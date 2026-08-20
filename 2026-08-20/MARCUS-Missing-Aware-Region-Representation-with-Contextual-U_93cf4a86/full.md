# MARCUS: Missing-Aware Region Representation with Contextual Urban Signals for Rent Prediction

Chenya Huang<sup>1</sup>, Bin Liang<sup>1,\*</sup>, Zhidong Li<sup>1</sup>, Yuxi Lu<sup>1</sup>, Kunqi Li<sup>1</sup>, Justin Wang<sup>2</sup>, and Fang Chen<sup>1</sup>

<sup>1</sup>Data Science Institute University of Technology Sydney Sydney, Australia

{Bin.Liang, Zhidong.Li, Fang.Chen}@uts.edu.au {Chenya.Huang, Yuxi.Lu, Kunqi.Li}@student.uts.edu.au

Corresponding author: Bin Liang.

<sup>2</sup>The Property Investors Alliance Sydney, Australia Justin@pia.com.au

Abstract—Multimodal urban data has expanded the applications of urban region representation learning, such as functional zone identification and real estate appraisal, but also introduces challenges caused by data incompleteness. Existing studies usually handle missing data through imputation, treating missingness as noise while ignoring its potential semantic value. To address this issue, we propose MARCUS, a missingaware region representation model that treats missingness as a contextual urban signal. MARCUS models missingness in three stages: Intra Learning jointly encodes observed features and missing patterns, Inter Learning estimates modality reliability to guide cross-modal interaction, and Fusion uses missing-aware and time-aware gating to generate the final region embedding. We apply MARCUS to rent prediction, a task with long-term trends and seasonal fluctuations, using real-world datasets from Sydney and New York. Experimental results show that MARCUS achieves state-of-the-art performance, reducing MAE by 51.35% on Sydney and 12.62% on New York compared with the best baselines. Additional experiments, including an imputation-based ablation study and randomized additional-missingness analysis, further demonstrate the effectiveness of the proposed method. For reproducibility, we release the implementation details and source code at: https://github.com/Santooops/MARCUS

Index Terms—Urban Region Embedding, Multimodal Learning, Rental Price Prediction, Incomplete Learning

## I. INTRODUCTION

In recent years, urban computing has attracted increasing attention from the research community [1]. This trend is driven by the growing availability of multimodal urban data [2]. In urban computing, cities are commonly partitioned into multiple regions with well-defined boundaries but irregular shapes [3]. These regions differ substantially in functional attributes, demographic structure, and economic activities. As a result, effectively modelling and understanding urban structure has become a key problem. High-quality urban region representations help characterize the spatial structure and functional distribution of cities. They also support a variety of downstream tasks, such as housing price prediction [4], [5], rent estimation [6], crime prediction [7], [8], and urban vitality assessment [9].

However, multimodal urban data is often incomplete in real urban environments. Coverage across multiple modalities typically exhibits systematic missingness. Some modalities may be entirely unavailable in certain regions. Others are only partially observed in space or time. More importantly, such missingness is usually not random noise. It is highly correlated with regional attributes and urban structure. For example, population-related indicators are semantically invalid in uninhabited areas, and POI or mobility signals tend to be sparser in newly developed or more remote regions. The resulting missingness patterns therefore carry urban semantics. They reflect functional status and development disparities across regions. Hence, missingness should be treated as an urban context signal, rather than merely a data issue to be resolved [10].

![](images/34f8d034d058b55e8c7d5a62ae98b9452da84abad410e872e5ccc83b81fa6468.jpg)  
Fig. 1. Additional missing ratio analysis on the Sydney dataset. The additional missing ratio denotes the proportion of originally observed data that are artificially masked. Naturally missing records in the original data are kept unchanged.

Existing methods [6], [39], [40] often treat missingness as an imputation problem, or adopt simple masking strategies during fusion. However, these approaches suffer from two key limitations.

1) They make it difficult for the model to explicitly distinguish reliable information from signals originating from sparse or missing sources during cross-modal interaction and fusion. This can introduce bias and noise.

2) They discard the regional semantics embedded in missingness patterns, preventing such information from contributing to representation learning. Even with dedicated imputation models, additional assumptions and imputation errors are inevitable, which may further amplify bias.

To address these limitations, we propose a missing-aware urban region representation learning model. It explicitly characterizes availability patterns in multimodal urban data and allows missingness information to permeate key stages of representation learning, especially during cross-modal interaction and fusion. Specifically, in intra learning, we jointly model observed-value features and missingness features by projecting them into the same latent space, then fusing the two embeddings into a unified representation. In inter learning, we estimate modality reliability based on missing status, thereby adaptively modulating interaction strength. In the final fusion stage, we further incorporate missingness patterns and temporal context to perform gated weighting, yielding region representations that are more robust and semantically informative under incomplete multimodal urban data. In addition, to evaluate model performance under different degrees of missingness, we conduct an additional missing ratio experiment, as shown in Fig. 1. MARCUS consistently achieves the best performance when the additional missing ratio is below approximately 55%. More detailed analysis is provided in Section “Analysis on Missing Ratio”.

## Our contributions can be summarized as follows:

• We propose a technical formulation that treats multimodal missingness as an informative urban context signal. For each region, time step, and modality, we construct missingness descriptors containing availability, missing ratio, and missing-location information, and inject them into intra learning, inter learning, and final modality fusion. This allows MARCUS to adaptively assess modality reliability and learn robust region representations from incomplete multimodal urban data.

• We propose a missingness-driven region representation learning framework, where missingness information permeates the entire pipeline, from intra-modality encoding to cross-modality interaction and final fusion.

• We validate MARCUS on real-world rental price prediction using datasets from Sydney and New York. MARCUS consistently outperforms traditional predictors and region embedding baselines across all metrics. Further comparisons with an imputation-first strategy and additional missingness experiments (Fig. 1) show that explicitly modelling missingness provides useful predictive information and enables MARCUS to remain robust under severe modality incompleteness.

## II. RELATED WORK

Early studies on urban region representation learning primarily relied on human mobility data to construct region embeddings, as mobility patterns capture dynamic interactions and behavioral connections between regions [2], [18], [22]– [24]. However, mobility data alone reflects only one aspect of urban dynamics and may overlook the semantic, functional, and socioeconomic characteristics of urban regions.

To obtain a more comprehensive representation of regions, recent studies have begun to jointly model mobility data with region attributes [25], [26], [27], [23]. For instance, POI distributions provide cues about regional functions [28], [29], [30], [31]. Census statistics capture demographic structure and socioeconomic context. Street-view or remote-sensing imagery further describes the built environment and land-use patterns [25], [32], [20], [33], [34], [35]. In addition, text, as a form of human language, also contributes to the enhancement of urban embedding [29], [20], [36]. These heterogeneous sources provide complementary perspectives on urban regions and have motivated multi-source region embedding learning. In parallel, researchers construct inter-region graphs based on spatial adjacency, functional similarity, or mobility [37], [38], [23], [18], [17]. These methods help capture cross-region dependencies and improve the structural consistency of the learned representations.

TABLE I DATASET STATISTICS
<table><tr><td>Datasets</td><td>Description</td><td>Sydney</td><td>New York</td></tr><tr><td>Regions</td><td>Partition scheme # Regions</td><td>SA2 373</td><td>ZIP code 177</td></tr><tr><td>Census data</td><td># Surveys Missing ratio</td><td>19,750 0.54%</td><td>85,154 0%</td></tr><tr><td>POI data</td><td># Records # Categories</td><td>145,548 13</td><td>20,570 13</td></tr><tr><td>Traffic data</td><td>Missing ratio # Mobility records</td><td>27.67% 22,816</td><td>19.66% 240,558,409</td></tr><tr><td>Rent data</td><td>Missing ratio # Records</td><td>40.40% 608,652</td><td>6.87%</td></tr><tr><td></td><td></td><td></td><td>6,372</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Temporal data</td><td></td><td>07/2021</td><td>01/2022</td></tr><tr><td></td><td>Time Period</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>07/2024</td><td>12/2024</td></tr></table>

However, existing fusion and relational modelling methods often rely on an implicit assumption. It is that most modalities are available across regions with comparable quality, enabling alignment and joint learning on a unified spatial unit. When this assumption does not hold, missingness and noise can be amplified during fusion and message passing. This should bias region representations and undermine structural consistency. Motivated by this practical constraint, we propose a novel missing-aware multimodal urban region representation learning framework to address this issue. For heterogeneous data, we characterize data availability through missingness patterns and use them to guide cross-modal fusion weights, resulting in more robust and transferable urban region representations.

In addition, some studies have introduced prompt-based task adaptation into urban region representation learning.HREP adopts a prefix-tuning–style prompt to guide heterogeneous region graph representations to learn more effectively across tasks [19]. FlexiReg further leverages prompting to customize task-oriented region representations [20]. These studies show that prompt-based adaptation can improve the flexibility and transferability of region embeddings. However, they mainly focus on task adaptation and do not explicitly model multimodal missingness, which is the focus of this work.

Black boundary: SA2 cells

![](images/43fd9e40b404ff47a1ddebea503fdb02ce1c160ad8a6fcb9c4dd2036e39b99bb.jpg)  
Fig. 2. Example of POIs in the Sydney Region

## III. DATA DESCRIPTION AND ANALYSIS

In this section, we introduce the datasets used in our research and provide a preliminary analysis. We use two realworld urban datasets, Sydney and New York, which correspond to major real-estate markets in Australia and the United States, respectively. We collected Point of Interest(POI) data, census statistics, and traffic records for both cities to construct region representations. These data were obtained from the Australian Bureau of Statistics (ABS) [11], NYC Open Data [12] and Office of Policy Development and Research [13]. In addition, we collected rental datasets for the downstream task. Table 1 summarizes the dataset statistics, including missing status. The missing ratio is computed at the region level.

## A. Region Data

Due to differences in the availability and usability of other data sources across geographic scales, we adopt two different spatial partition schemes for Sydney and New York. Sydney follows the ABS statistical geography, Statistical Area Level 2 (SA2), while New York uses ZIP codes as the regional units. This setting also allows us to examine how our model performs under different spatial granularities.

## B. Census Data

The raw census datasets contain 12,176 and 85,154 records for Sydney and New York. Both are reported at the finest available statistical units. To align with other data sources, we aggregate Sydney’s census data from Statistical Area Level(SA1) to SA2, and New York’s census data from Census Tracts to ZIP codes. Each record includes key socioeconomic attributes such as total population, average household income, educational attainment, and employment rate. In the Sydney census data, there is only a very small amount of missingness (0.54%), whereas the New York census data is complete.

## C. POI Data

We collect all POIs in Sydney and New York, which are officially categorized into 13 functional groups in each city. Although the category definitions differ slightly between the two datasets, they broadly cover common urban functions such as community, education, health, and recreational facilities. In total, there are 145,548 POIs in Sydney and 20,570 POIs in New York. For each region, we construct a textual description of the POIs it contains, as illustrated in Fig. 2. In the POI data, Sydney has a notable amount of missingness (27.67%), while New York has a lower missing ratio (19.66%).

## D. Traffic Data

Traffic data characterize city mobility flows. Due to data availability, we use public transport station entry and exit volumes for Sydney, and taxi trip records for New York. Taxi trip records include both street-hail and e-hail (app-based) trips, covering the entirety of New York City. These mobility data include information such as origins and destinations, and reflect the travel mode (e.g., train, metro and taxi). Since SA2 is a community-level geography designed for statistical reporting, transport stations do not cover all regions. As a result, the Sydney traffic dataset contains a substantial amount of missingness. The missing ratio of the New York traffic data is 6.87%. A large portion of this missingness arises because the raw records contain only origins or only destinations.

## E. Rent Data

In this paper, rent data are collected from real estate agencies. Sydney rent data are sourced from PriceFinder [14] and the Property Industry Alliance (PIA) [15], while the New York rent data are obtained from Zillow [16]. The New York rent data are provided as ZIP-code–level rent index, rather than transaction-level rental records. The scope of the Sydney dataset spans from July 2021 to July 2024, while the New York dataset covers January 2022 to December 2024. Fig. 3(a) shows the monthly transaction volume from January 2023 to July 2024, revealing pronounced seasonal peaks. Fig. 3(b) plots the rental price trend over 43 months for Mascot, a representative area in Sydney, where we observe both seasonal fluctuations and a clear long-term upward trend. For a more clear presentation, we use data spanning January 2021 to July 2024. Fig. 3(c) presents the overall rent distribution in Sydney.

## IV. PROPOSED MODEL

We start with a few basic concepts and a problem statement. The frequently used symbols are summarized in Table 2.

## A. Features description

Regions. The set of regions R is obtained by partitioning the spatial area of interest into non-overlapping units using a predefined scheme (e.g., postcode-based areas or censusdefined regions). We denote the i-th region by r<sub>i</sub>. For each region, we consider three types of features, including two static modalities (POI, census) and one dynamic modality (traffic), and collect them into the set M.

POI. For each region, we collect all points of interest (POIs) from the Australian Bureau of Statistics and NYC Open Data, and follow the official POI taxonomies provided by the corresponding sources. Both cities contain 13 POI categories (e.g., education and transportation), although the category systems are city-specific and not required to be aligned across cities. For each region $r _ { i } .$ , we first count the number of POIs in each category, and then serialize these statistics into a compact textual representation in the form of category count pairs (e.g., education:12; transport:3). We encode the resulting text into a fixed-dimensional semantic embedding, which serves as the POI modality input to our model.

![](images/42e0baec17c171953e5b03b55bb3299e0a97f0412dc22bdab46072be42e89aee.jpg)  
(a) Temporal Distribution of Transaction Volume

![](images/2ea96ae7b155599f5b93104e7668f7a0f1ab16a90de95808924301fd4b4d7a43.jpg)  
(b) Temporal Trend of Rental Price

![](images/6c5c57466439a013cdc7db22c3ee9f0c80be1418e8e04b3a3e8fa6b2d52300cf.jpg)  
(c) Rental Price Distribution

Fig. 3. Analysis and Visualization on Sydney Rent Data  
TABLE II  
FREQUENTLY USED SYMBOLS
<table><tr><td>Symbol</td><td>Description</td></tr><tr><td>R</td><td>Set of regions (e.g., SA2)</td></tr><tr><td> $N$ </td><td>Number of regions</td></tr><tr><td> $\mathcal { M } = \{ P , C , F \}$ </td><td>Modality set (POI, census, traffic)</td></tr><tr><td>S</td><td>Missingness vector for modality m at (i, t) (flag, ratio, and missing-location encoding).</td></tr><tr><td>h</td><td>Intra-modality embedding for modality m at (i, t)</td></tr><tr><td>H</td><td>Inter-updated embedding after cross-modal interaction</td></tr><tr><td>Z</td><td>Final fused region embedding for downstream prediction</td></tr><tr><td> $y _ { i , t + 1 }$ </td><td>Ground-truth rent at t+1</td></tr><tr><td>yi,t+1</td><td>Predicted rent at t+1</td></tr></table>

Census. Consistent with the POI modality, we collect the most recent census data for both cities and extract region-level socioeconomic features, such as household income, employment rate, and educational attainment. For each region $r _ { i }$ , we represent the census attributes as a feature vector $\mathbf { C } _ { i }$

Traffic (Flow). For each region $r _ { i }$ , we construct month-level inter-region mobility information. For each month t, we define the flow $F _ { i , j , t }$ for a region pair $( r _ { i } , r _ { j } )$ , representing the travel volume from $r _ { i }$ to $r _ { j }$ during month t (which can be summarized as inflow/outflow statistics). As the most readily available mobility data differ across cities, we use train station tap-in/tap-out counts as a proxy for travel intensity in Sydney, and taxi trip volumes to characterize inter-region mobility in New York City.

## Missingness.

We treat missingness as a contextual urban signal, and construct a missingness feature vector $\mathbf { s } _ { i , t } ^ { ( m ) }$ for each region $r _ { i } ,$ month $t ,$ and modality $m \in { \mathcal { M } }$ . Specifically, $\mathbf { s } _ { i , t } ^ { ( m ) }$ includes an availability indicator $b _ { i , t } ^ { ( m ) }$ , a missingness ratio $\rho _ { i , t } ^ { ( m ) }$ , and a missing-location encoding $\mathbf { l } _ { i , t } ^ { ( m ) }$ that captures where missingness occurs and its spatial pattern. Here, $b _ { i , t } ^ { ( m ) } \in \{ 0 , 1 \}$ is defined at the modality level: $b _ { i , t } ^ { ( m ) } = 1$ if any feature in modality m is missing for region i at time t, and $b _ { i , t } ^ { ( m ) } = 0$ otherwise. The missingness ratio $\rho _ { i , t } ^ { ( m ) }$ is defined as the fraction of missing features in modality m for region i at time t:

$$
\rho _ { i , t } ^ { ( m ) } = \frac { \left| \mathcal { F } _ { \mathrm { m i s s } } ^ { ( m ) } ( i , t ) \right| } { \left| \mathcal { F } ^ { ( m ) } ( i , t ) \right| } .\tag{1}
$$

We incorporate $\mathbf { s } _ { i , t } ^ { ( m ) }$ into representation learning and crossmodal fusion, enabling the model to adaptively adjust modality contributions under incomplete observations.

$$
\mathbf { s } _ { i , t } ^ { ( m ) } = \big [ b _ { i , t } ^ { ( m ) } \big \lVert \rho _ { i , t } ^ { ( m ) } \big \rVert \mathbf { l } _ { i , t } ^ { ( m ) } \big ] .\tag{2}
$$

## B. Problem statement

Given a set of regions $\mathcal { R } = \{ r _ { i } \} _ { i = 1 } ^ { N }$ and a sequence of months indexed by t, we observe multi-modal region attributes from three modalities $\mathcal { M } = \{ P , C , F \}$ , where $P , C ,$ and $F$ denote POI, census, and traffic (flow), respectively. For each region $r _ { i }$ at month t, let $\mathbf { x } _ { i , t } ^ { ( m ) }$ denote the input of modality $m \in { \mathcal { M } } .$ , and let $\mathbf { s } _ { i , t } ^ { ( m ) }$ denote its missingness vector (including missing flag, missing ratio, and a missing-location encoding). Our goal is to learn a missing-aware representation function f that maps the multi-modal inputs of $( r _ { i } , t )$ to a d-dimensional region embedding:

$$
f : ( r _ { i } , t , \{ \mathbf { x } _ { i , t } ^ { ( m ) } , \mathbf { s } _ { i , t } ^ { ( m ) } \} _ { m \in \mathcal { M } } )  \mathbf { Z } _ { i , t } \in \mathbb { R } ^ { d } .\tag{3}
$$

The learned embedding $\mathbf { Z } _ { i , i }$ <sub>t</sub> should capture region characteristics while being robust to structured missingness, and is used for the downstream house rent prediction task:

$$
\hat { y } _ { i , t + 1 } = M L P ( \mathbf { Z } _ { i , t } ) ,\tag{4}
$$

where $y _ { i , t + 1 }$ is the ground-truth rent for the next month.

In this section, we provide a detailed description of our proposed model MARCUS. First, we introduce Missing-aware

Embedding Learning, which consists of two branches: Intra Learning and Inter Learning. These two branches jointly produce embeddings that are sensitive to missingness patterns and automatically calibrate modality trustworthiness. Then, we present Missing-aware Fusion, which dynamically weights and fuses modality representations based on missingness patterns and temporal context to obtain more robust embeddings. The overall architecture of the model is illustrated in Fig. 4. It mainly consists of two modules. First, we feed the preprocessed multimodal inputs into Missing-aware Embedding Learning. The inputs include modality features and the corresponding missing summary. This module has two stages, Intra Learning and Inter Learning. Intra Learning jointly models observed-value features and missingness features within each modality. It produces a missing-aware representation for each modality. Inter Learning then performs cross-modal interaction. It estimates modality reliability from the missingness status. It also adaptively calibrates the strength of information exchange across modalities. This produces updated cross-modal embeddings. Next, these embeddings are sent to Missing-aware Fusion. Fusion uses the missing summary and the temporal context. It dynamically weights and fuses modality representations. Finally, it outputs the region representation for downstream rent prediction tasks.

## C. Missing-aware Embedding Learning

This section describes our missing-aware embedding learning, where missingness is incorporated into intra-modal encoding and cross-modal interaction.

1) Intra Learning: Modality-specific Missing-aware Encoding: For each region $r _ { i }$ and month t, we consider a set of modalities $\mathcal { M } \left( \mathrm { e . g . , P O I } , \right.$ Census, and Traffic). For each modality $m \in \mathcal { M }$ , we construct the modality input by concatenating its content embedding and the missingness feature vector:

$$
\mathbf { x } _ { i , t } ^ { ( m ) } = \big [ \mathbf { e } _ { i , t } ^ { ( m ) } \parallel \mathbf { s } _ { i , t } ^ { ( m ) } \big ] ,\tag{5}
$$

where $\mathbf { e } _ { i , t } ^ { ( m ) } \in \mathbb { R } ^ { d _ { e } ^ { m } }$ denotes the modality embedding, and the missingness feature vector is defined as

$$
\mathbf { s } _ { i , t } ^ { ( m ) } = \left[ b _ { i , t } ^ { ( m ) } , \ \rho _ { i , t } ^ { ( m ) } , \ 1 _ { i , t } ^ { ( m ) } \right] \in \mathbb { R } ^ { d _ { s } ^ { m } } .\tag{6}
$$

Note that POI and Census are static modalities, thus ${ \bf e } _ { i , t } ^ { ( m ) } \equiv { \bf \Psi }$ ${ \bf e } _ { i } ^ { ( m ) }$ for $m \in \{ \mathbf { P } , \mathbf { C } \}$ . Importantly, Intra Learning does not take the month index t as an explicit input; temporal context is incorporated later in the fusion stage.

We instantiate Intra Learning with a lightweight MLP-based encoder block $f _ { \mathrm { e n c } } ^ { ( m ) } ( \cdot )$ for each modality m. All modalities share the same encoder architecture while using modalityspecific parameters. The encoder maps $( \mathbf { e } _ { i , t } ^ { ( m ) } , \mathbf { s } _ { i , t } ^ { ( m ) } )$ to a unified hidden representation $\mathbf { h } _ { i , t } ^ { ( m ) } \in \mathbb { R } ^ { d _ { h } }$ by 1) projecting the content embedding and the missingness vector into the same hidden space via two separate branches and 2) fusing the two branches through a fusion layer.

Specifically, we first compute two branch-wise hidden vectors:

$$
\tilde { \mathbf { h } } _ { e } ^ { ( m ) } = \sigma \Big ( \mathbf { W } _ { e } ^ { ( m ) } \mathbf { e } _ { i , t } ^ { ( m ) } + \mathbf { b } _ { e } ^ { ( m ) } \Big ) ,\tag{7}
$$

$$
\tilde { \mathbf { h } } _ { s } ^ { ( m ) } = \sigma \Big ( \mathbf { W } _ { s } ^ { ( m ) } \mathbf { s } _ { i , t } ^ { ( m ) } + \mathbf { b } _ { s } ^ { ( m ) } \Big ) .\tag{8}
$$

where $\sigma ( \cdot )$ is a non-linear activation function. We then concatenate the two hidden vectors and apply a fusion layer to obtain the final intra-modality representation:

$$
\mathbf { h } _ { i , t } ^ { ( m ) } = \sigma \Big ( \mathbf { W } _ { f } ^ { ( m ) } \big [ \tilde { \mathbf { h } } _ { e } ^ { ( m ) } \ \lVert \ \tilde { \mathbf { h } } _ { s } ^ { ( m ) } \big ] + \mathbf { b } _ { f } ^ { ( m ) } \Big ) .\tag{9}
$$

The resulting $\mathbf { h } _ { i , t } ^ { ( m ) }$ serves as the modality-specific hidden representation for subsequent Inter Learning and Missingaware Fusion.

2) Inter Learning: Reliability-aware Cross-modality Interaction: After Intra Learning, we obtain intra-modality representations $\mathbf { h } _ { i , t } ^ { ( m ) } ~ \in ~ \mathbb { R } ^ { d _ { h } }$ for each modality $m \in \mathcal { M } \ =$ $\{ P , C , F \}$ . Since these modalities provide complementary urban signals $( \mathrm { e . g . }$ , amenities, socioeconomic attributes, and mobility patterns), Inter Learning aims to explicitly model cross-modality dependencies so that each modality can absorb complementary cues from the others and refine its representation, resulting in inter-updated embeddings $\mathbf { H } _ { i , t } ^ { ( m ) }$ . Meanwhile, real-world observations are often incomplete and exhibit heterogeneous data quality across modalities. Simply performing cross-modality interaction may propagate noise from severely missing modalities. To address this issue, we introduce a reliability-aware interaction mechanism that adaptively suppresses the influence of unreliable modalities during crossmodality interaction.

Modality Reliability Estimation. We treat missingness as a useful signal reflecting data availability and quality. Given the missingness vector $\mathbf { s } _ { i , t } ^ { ( m ) } = \left[ b _ { i , t } ^ { ( m ) } , \rho _ { i , t } ^ { ( m ) } , \mathbf { l } _ { i , t } ^ { ( m ) } \right]$ , Inter Learning focuses on the two most direct quality indicators: the missing flag and the missingness ratio. Specifically, we estimate a scalar reliability weight $\alpha _ { i , t } ^ { ( m ) } \in ( 0 , \bar { 1 } )$ from $\left[ b _ { i , t } ^ { ( m ) } , \rho _ { i , t } ^ { ( m ) } \right]$ via a lightweight mapping:

$$
\alpha _ { i , t } ^ { ( m ) } = \mathrm { s i g m o i d } \left( \mathbf { w } _ { \alpha } ^ { \top } \big [ b _ { i , t } ^ { ( m ) } , ~ \rho _ { i , t } ^ { ( m ) } \big ] + b _ { \alpha } \right) , \quad m \in \{ P , C , F \} .\tag{10}
$$

Intuitively, $\alpha _ { i , t } ^ { ( m ) }$ quantifies the trustworthiness of modality m under the current observation pattern. It decreases when the modality is less available or more severely missing, and is therefore used to down-weight its contribution during crossmodal interaction.

Reliability-modulated Cross-modality Interaction. We model cross-modality interaction by applying multi-head selfattention over the modality dimension, where each modality representation attends to the others to aggregate complementary information. Let $\Delta \mathbf { h } _ { i , t } ^ { ( m ) }$ denote the interaction-induced update for modality m produced by the cross-modality attention module. To prevent unreliable modalities from amplifying noisy interaction effects, we modulate the update magnitude using the estimated reliability and fuse it with a residual connection:

![](images/4bf674a9a1cd8ea6b4dc31651b4cbc77409db9468045e5aeb71c4cc839ca5412.jpg)  
Fig. 4. Overview of the Proposed MARCUS Framework

$$
\mathbf { H } _ { i , t } ^ { ( m ) } = \mathbf { h } _ { i , t } ^ { ( m ) } + \alpha _ { i , t } ^ { ( m ) } \cdot \Delta \mathbf { h } _ { i , t } ^ { ( m ) } , \quad m \in \{ P , C , F \} .\tag{11}
$$

In this way, modalities with higher reliability preserve more interaction updates and benefit more from complementary cues, while unreliable modalities are explicitly suppressed to reduce noise propagation. In practice, normalization and a feed-forward network can be further applied for training stability and expressive power. However, the core idea remains that missingness-driven reliability adaptively controls the strength of cross-modality interaction.

The resulting $\{ \mathbf { H } _ { i , t } ^ { ( m ) } \} _ { m \in \mathcal { M } }$ are then fed into the subsequent Missing-aware Fusion module to produce the final region embedding.

D. Missing-aware Fusion: Dynamic Fusion with Missingness and Temporal Context

After Inter Learning, we obtain inter-updated modality representations $\mathbf { H } _ { i , t } ^ { ( m ) } \in \mathbb { R } ^ { d _ { h } }$ for each modality $m \in \mathcal { M } =$ $\{ P , C , F \}$ . Since the contribution of each modality may vary across months (for example, traffic dynamics) and observations are often incomplete with heterogeneous missingness patterns, a fixed fusion strategy (such as uniform averaging) can be suboptimal and unstable. We therefore propose Missing-aware Fusion, which leverages both missingness patterns and temporal context to dynamically assign modality weights and produce a robust region representation $\mathbf { Z } _ { i , t }$ for downstream prediction.

Temporal Context Encoding. We map the month index t to a temporal context vector $\mathbf { c } _ { t }$ via a learnable embedding followed by a projection. Importantly, this temporal signal is not directly used in Intra or Inter encoding. Instead, it is introduced at the fusion stage to capture month-specific modality contributions and seasonal effects.

Missingness and Time-conditioned Gating. For each modality m, we compute a scalar fusion weight $\bar { \lambda } _ { i , t } ^ { ( m ) } \in ( 0 , 1 )$ using a modality-specific gating function (a lightweight MLP) conditioned on the full missingness vector and temporal context:

$$
\lambda _ { i , t } ^ { ( m ) } = g _ { m } \left( \left[ \mathbf { s } _ { i , t } ^ { ( m ) } \parallel \mathbf { c } _ { t } \right] \right) , \quad m \in \{ P , C , F \} .\tag{12}
$$

Compared to Inter Learning, which estimates reliability using only $\left[ b _ { i , t } ^ { ( m ) } , \rho _ { i , t } ^ { ( m ) } \right]$ , Missing-aware Fusion exploits the complete missingness vector $\mathbf { s } _ { i , t } ^ { ( m ) } = \left[ b _ { i , t } ^ { ( m ) } , \rho _ { i , t } ^ { ( m ) } , \mathbf { l } _ { i , t } ^ { ( m ) } \right]$ to capture where missingness occurs and its spatial pattern, enabling finergrained and context-dependent weighting.

Normalized Weighted Aggregation. Given the dynamic weights, we fuse modality representations via a normalized weighted sum:

$$
\mathbf { Z } _ { i , t } = \frac { \sum _ { m \in \mathcal { M } } \lambda _ { i , t } ^ { ( m ) } \mathbf { H } _ { i , t } ^ { ( m ) } } { \sum _ { m \in \mathcal { M } } \lambda _ { i , t } ^ { ( m ) } + \epsilon } .\tag{13}
$$

where ϵ is a small constant for numerical stability. This design allows the model to down-weight modalities that are severely missing or less informative under the current month, and to emphasize more reliable and month-relevant modalities, resulting in robust region embeddings $\mathbf { Z } _ { i , }$ <sub>t</sub> for downstream tasks like rent prediction.

## V. EXPERIMENTS

Our experiments are designed to address four research objectives:

1) Performance comparison with traditional machine learning methods and existing region-embedding approaches.

2) Applicability of MARCUS across different countries and geographic scales;

3) Effectiveness of region embeddings for rent prediction tasks with pronounced seasonal fluctuations, as shown in Fig. 3(b);

4) Contribution of each module to overall performance (via ablation studies).

Dataset. We conduct experiments on real-world datasets from two cities: Sydney and New York. We collect region partitions, census data, POI data, traffic records, and rent data. For region partitioning, we use SA2 area for Sydney and ZIP codes area (postcodes) for New York. More details on the data description and analysis can be found in Section 2.

Baselines. We compare MARCUS with two categories of baselines. The first category includes traditional machine learning predictors: LR, SVR, and RF. These baselines are used to examine whether incorporating region embeddings improves rent prediction performance. The second category includes existing region embedding models which are used to evaluate whether the representations learned by MARCUS outperform prior methods.They are MVURE [17], MGFN [18], HREP [19] and FLEXIREG [20].

FLEXIREG proposes learning more flexible representations of urban regions. It learns cell-level embeddings on a hexagonal grid from multimodal data, including POIs, streetview images and land-use data. Then, FLEXIREG customizes the representations for different region partitions and downstream tasks via adaptive aggregation and prompt learning.

HREP targets heterogeneous region embedding. The core innovation lies in introducing prompt learning to region embedding. Task-specific prompts guide downstream task learning, thereby eliminating the need to retrain region representations for each individual task.

MGFN uses human mobility as the sole modality to mine urban structure and patterns. It fuses mobility graphs from multiple time steps into mobility patterns, and learns region representations via intra-pattern message passing combined with inter-pattern cross-attention.

MVURE uses multi-view urban data, including mobility views and urban attribute views. It encodes each view with GAT, shares information via cross-view self-attention, and fuses the views with adaptive weights into a region embedding.

We similarly learn region embeddings by adaptively fusing multimodal urban data. MARCUS further addresses coverage imbalance and missingness by treating them as reliability signals that guide cross-modal interaction and fusion, thereby enabling reliable downstream performance even when certain features are missing.

We also include a zero-information baseline (ZIB), which predicts the training-set mean of the target for every test sample without using any input features.It serves as a twosided reference: compared with ZIB, lower error indicates retained region-discriminative information, while higher error suggests that the model is misled by its inputs under severe missingness.

TABLE III  
PERFORMANCE COMPARISON ON SYDNEY AND NEW YORK
<table><tr><td rowspan="2">Model</td><td colspan="3">Sydney</td><td colspan="3">New York</td></tr><tr><td>MAE ↓</td><td></td><td>RMSE ↓ MAPE(%) ↓ MAE ↓</td><td></td><td>RMSE ↓ MAPE(%) ↓</td><td></td></tr><tr><td>ZIB</td><td>126.46</td><td>158.65</td><td>24.05</td><td>558.94</td><td>862.07</td><td>16.14</td></tr><tr><td>LR</td><td>112.59</td><td>142.80</td><td>21.69</td><td>353.18</td><td>511.62</td><td>10.92</td></tr><tr><td>SVR</td><td>94.40</td><td>105.73</td><td>16.72</td><td>336.75</td><td>381.17</td><td>11.14</td></tr><tr><td>RF</td><td>84.46</td><td>96.61</td><td>15.19</td><td>187.42</td><td>261.18</td><td>6.04</td></tr><tr><td>FLEXIREG</td><td>77.18</td><td>88.83</td><td>14.18</td><td>160.10</td><td>239.55</td><td>5.14</td></tr><tr><td>HREP</td><td>75.51</td><td>87.33</td><td>13.97</td><td>174.81</td><td>251.69</td><td>5.61</td></tr><tr><td>MGFN</td><td>90.28</td><td>102.25</td><td>16.36</td><td>196.41</td><td>276.09</td><td>6.42</td></tr><tr><td>MVURE</td><td>72.54</td><td>85.87</td><td>13.45</td><td>189.48</td><td>287.29</td><td>6.17</td></tr><tr><td>MARCUS</td><td>35.29</td><td>45.91</td><td>7.08</td><td>139.90</td><td>213.87</td><td>4.51</td></tr><tr><td>Enhance(%)</td><td>51.35</td><td>46.53</td><td>47.36</td><td>12.62</td><td>10.72</td><td>12.26</td></tr></table>

Metrics. For the downstream task, we use three evaluation metrics to assess prediction performance: mean absolute error (MAE), root mean square error (RMSE), and mean absolute percentage error (MAPE). Lower values indicate better performance.

## A. Overall Performance

Table 3 reports the overall results of MARCUS and all baselines across the three metrics. MARCUS achieves stateof-the-art performance on all metrics. For clarity, the best overall result is highlighted in bold, while the best-performing baseline is marked with underlining. Based on these results, we have the following findings:

1) MARCUS achieves strong overall performance and outperforms all baseline models. In terms of MAE, MARCUS improves by up to 51.35% over the best baseline MVURE in the Sydney rental market. It also consistently surpasses the previous SOTA baseline FLEXIREG across all evaluated metrics and datasets, achieving improvements of 54.2% in Sydney and 12.62% in New York. The larger gain in Sydney reflects baseline degradation under native missingness rather than instability of MARCUS. Relative to ZIB, MARCUS stays at 25–29% of the zero-information error in both cities, whereas the strongest competing model degrades from 28–32% on New York to 54–57% on Sydney. This is largely due to Sydney’s data, where 40.4% of traffic records are natively missing, a regime that baseline models without explicit missingness handling struggle to handle.

2) MARCUS remains effective under different geographic partition schemes. As mentioned earlier, Sydney follows a statistical geography standard, while New York uses postal zones. The consistent performance across these two different rental market settings and spatial scales suggests that MARCUS has good applicability beyond a single city-specific partition scheme.

3) MARCUS is effective for rent forecasting with temporal dynamics. Since rental markets exhibit pronounced seasonal fluctuations, we incorporate time into downstream prediction by appending the same time embedding to all models. We do not modify the baseline representation learning models, ensuring a fair and consistent use of temporal information. Under this setting, MARCUS still outperforms all baselines, further demonstrating its effectiveness for seasonal rent forecasting.

![](images/baa0c70130f76d05ba116ebd4fe7ec9b74387670032f1b7027e551ba5611176e.jpg)  
Fig. 5. Ablation Study Results (Sydney)

## B. Ablation Study

We conduct the following studies to validate the effectiveness of the components in MARCUS:

1) MARCUS-w/o-IAM: We disable the use of missingness features in the Intra Learning module. This investigates whether missingness should be incorporated early during each modality’s representation learning, or whether it is sufficient to introduce it only in the subsequent Inter learning and Fusion stages.

2) MARCUS-w/o-RW: We remove the reliability weights computed from the missing flag and missing ratio, thereby treating all modalities equally during cross-modal interactions in the Inter Learning module. It tests whether the gains are mainly due to missingness-based reliability weighting, instead of the cross-modal attention module itself.

3) MARCUS-w/o-FG: We remove the missing-aware fusion gate in the Fusion stage.It examines whether the final representation benefits from missing-aware fusion weighting, or whether naive fusion is already sufficiently robust.

4) MARCUS-w/o-TG: We keep the gating structure but remove the time context input, computing fusion weights solely from missingness features. This variant assesses whether timeconditioned gating is necessary for rent prediction tasks with trends and seasonality.

5) MARCUS-w/o-M: We completely remove missingness from the model. This control experiment is designed to quantify the benefit of treating missingness as an urban contextual signal.

As shown in Fig. 5, MARCUS consistently outperforms all variants, highlighting the contribution of each component to its overall effectiveness. The main observation is that missingness should be treated as a useful contextual signal, rather than being handled only as a data quality issue. When missingness is completely removed (MARCUS-w/o-M), the prediction performance degrades drastically that MAE nearly doubles from 35.29 to 70.15. Second, the missing-aware fusion gate and its temporal conditioning constitute the key contributors to performance. Removing the fusion gate (MARCUS-w/o-FG) results in the largest performance drop. This implies that missingness should inform the fusion weights at the final stage, as naive fusion may override earlier missing-aware benefits. Moreover, removing time context from the gate (MARCUSw/o-TG) leads to a substantial degradation. Because in rent prediction, long-term trends and seasonality make missingness effects time-dependent. The same missingness pattern can reflect different semantics and modality reliability across months, motivating temporal conditioning in the fusion gate.

TABLE IV  
PERFORMANCE COMPARISON OF MARCUS VARIANTS.
<table><tr><td>Model</td><td>MAE</td><td>RMSE</td><td>MAPE(%)</td></tr><tr><td>MARCUS-w/o-IAM</td><td>38.24</td><td>50.34</td><td>8.48</td></tr><tr><td>MARCUS-w/o-RW</td><td>40.24</td><td>50.63</td><td>7.96</td></tr><tr><td>MARCUS-w/o-FG</td><td>75.39</td><td>90.08</td><td>14.20</td></tr><tr><td>MARCUS-w/o-TG</td><td>68.86</td><td>84.27</td><td>12.87</td></tr><tr><td>MARCUS-w/o-M</td><td>70.15</td><td>85.09</td><td>13.24</td></tr><tr><td>MARCUS-Full</td><td>35.29</td><td>45.91</td><td>7.08</td></tr></table>

After confirming the decisive role of the fusion stage, we further examine the effect of missingness in earlier stages, including reliability modulation in cross-modal interactions (MARCUS-w/o-RW) and intra-modality missing-aware encoding (MARCUS-w/o-IAM). Specifically, without reliability weighting, MAE increases from 35.29 to 40.24, indicating that missingness-driven reliability modeling effectively suppresses noise propagation in cross-modal attention. In contrast, removing intra-modality missing-aware encoding results in a smaller degradation. Overall, these observations suggest that missingness matters most in interaction and fusion, whereas intra-modality missing-aware encoding offers complementary gains in alignment and robustness. The ablation results are detailed in Table 4.

## C. Analysis on Missing Ratio

We further conduct additional experiments with different levels of additional missingness on the rental price prediction task. Specifically, the original missing data are kept unchanged, while a certain proportion of originally observed data is randomly masked.

Fig. 1 compares MARCUS with baseline models on the Sydney dataset. As the additional missing ratio increases, MARCUS degrades smoothly and monotonically. When all originally observed modality cells are masked, its MAE reaches 116.2, which remains below ZIB. This indicates that even when all originally observed modality information is removed, MARCUS can fall back on a meaningful learned prior through explicit missingness encoding, rather than being misled by absent inputs.

In contrast, the baselines fail in distinct ways. MVURE’s MAE jumps by 55% from 71.4 to 110.5 with only 5% additional masking, and then fluctuates around ZIB. FlexiReg crosses ZIB at a ratio of around 0.3 and plateaus above it, suggesting limited region-discriminative information. MGFN deteriorates monotonically to 176.8, about 40% worse than ZIB, indicating that masked inputs may actively mislead it. HREP appears robust but is mainly insensitive to attribute modalities. Its MAE without additional masking is already 2.3× that of MARCUS (81.4 versus 35.3), so it has little modality information to lose when originally observed modality data are removed.

Overall, MARCUS outperforms all baselines throughout the practically relevant regime, where the additional missing ratio is no larger than 0.5, while being the only model that both fully exploits available modalities and remains below ZIB under severe missingness.

## VI. REAL-WORLD DEPLOYMENT

The proposed MARCUS model has been deployed in a real-world rent prediction system operated by our industry partner, Property Investors Alliance (PIA), as shown in Fig. 6 and Fig. 7. The system provides two modes: a simple view and a professional view. The simple view supports basic rent prediction, while the professional view extends it with additional functions, including file upload and map-based location.

PIA is an Australian residential property investment and property management platform that provides data-driven market analysis and advisory services, and manages approximately 6,000 properties. In the deployed system, MARCUS generates region-level rent estimates across Greater Sydney, with Statistical Area Level 2 (SA2) regions serving as the default spatial unit. These estimates are combined with predictions at other spatial scales (SA1 and suburb) to provide comprehensive rental recommendations. The online deployment incorporates continuously updated multimodal data beyond the temporal range used in offline experiments, including newly collected rental transactions and urban signals from July 2024 onward, reflecting realistic operational conditions with persistent and heterogeneous missingness.

Under this real-world deployment, MARCUS achieves a 57.1% improvement in rent prediction accuracy at the SA2 level compared with PIA’s previously adopted model, ST-ResMAE [6]. Beyond accuracy gains, the deployment demonstrates clear practical benefits in reducing uncertainty in region-level rent estimation under incomplete multimodal observations. By producing more stable and informative regional rent reference ranges, the system supports downstream pricing and investment decision-making while avoiding reliance on individual-level or sensitive personal data.

An additional practical advantage of MARCUS is its flexibility with respect to spatial granularity. Although SA2 regions are used by default due to their alignment with official statistical reporting, the model can be readily adapted to finergrained units such as SA1 regions or to suburb-level partitions that are more intuitive for end users. This adaptability allows the deployed system to support diverse consumer-facing, business, and policy-oriented urban analytics, demonstrating that explicitly modelling missingness delivers measurable practical value beyond offline benchmarks.

![](images/bcbae136296c488632c3271cbc77dbba18293ff6c68a9a21810e151d0b28f494.jpg)

Fig. 6. Sydney Rent Prediction System-Simple View  
![](images/94a924293c35f00fad93a403298a3468fd9746796185f9f53dd4846f62260340.jpg)  
Fig. 7. Sydney Rent Prediction System-Professional View

## VII. CONCLUSION

In this paper, we studied urban region representation learning under incomplete multimodal urban data, where systematic missingness is prevalent and often correlated with regional attributes and urban structure. Rather than treating missingness as noise to be imputed or masked, we propose MARCUS, which models missingness patterns as contextual urban signals to guide region representation learning and multimodal fusion. Specifically, MARCUS incorporates missingness into intramodality encoding and reliability-aware cross-modality interaction. It further performs missingness and time-conditioned fusion to learn region embeddings that are robust to incompleteness and semantically informative. Experiments on realworld datasets from Sydney and New York for rent prediction show that MARCUS consistently outperforms all baselines. In addition, we include an ablation setting that imputes missing values and disables all missing-aware components. Performance drops noticeably under this setting, confirming that explicitly modelling missingness is effective. Overall, these results highlight the practical value of missing signals in multimodal urban computing.

## REFERENCES

[1] Y. Zheng, L. Capra, O. Wolfson, and H. Yang, “Urban computing: concepts, methodologies, and applications,” ACM Transactions on Intelligent Systems and Technology (TIST), vol. 5, no. 3, pp. 1–55, 2014.

[2] C. Huang, B. Liang, Z. Li, and F. Chen, “Multimodal machine learning for real estate appraisal: a comprehensive survey,” in Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, 2025, pp. 345–361.

[3] A. Schneider, M. A. Friedl, and D. Potere, “Mapping global urban areas using MODIS 500-m data: new methods and datasets based on ‘urban ecoregions’,” Remote Sensing of Environment, vol. 114, no. 8, pp. 1733– 1746, 2010.

[4] P. Jenkins, A. Farag, S. Wang, and Z. Li, “Unsupervised representation learning of spatial data via multimodal embedding,” in Proc. 28th ACM Int. Conf. Inf. Knowl. Manage., 2019, pp. 1993–2002.

[5] Q. Zhang et al., “HGAurban: heterogeneous graph autoencoding for urban spatial-temporal learning,” in Proc. 34th ACM Int. Conf. Inf. Knowl. Manage., 2025, pp. 4139–4148.

[6] C. Huang, B. Liang, Z. Li, J. Wang, and F. Chen, “Spatio-temporal residual masked autoencoder for urban rent estimation,” in Proc. 34th ACM Int. Conf. Inf. Knowl. Manage., 2025, pp. 4807–4811.

[7] M. Hou, X. Hu, J. Cai, X. Han, and S. Yuan, “An integrated graph model for spatial–temporal urban crime prediction based on attention mechanism,” ISPRS International Journal of Geo-Information, vol. 11, no. 5, p. 294, 2022.

[8] J. Jin et al., “Urban region pre-training and prompting: a graph-based approach,” in Proc. 31st ACM SIGKDD Conf. Knowl. Discovery Data Mining V. 2, 2025, pp. 1071–1082.

[9] W. Yue, Y. Chen, Q. Zhang, and Y. Liu, “Spatial explicit assessment of urban vitality using multi-source data: a case of Shanghai, China,” Sustainability, vol. 11, no. 3, p. 638, 2019.

[10] Z. Che, S. Purushotham, K. Cho, D. Sontag, and Y. Liu, “Recurrent neural networks for multivariate time series with missing values,” Scientific Reports, vol. 8, no. 1, p. 6085, 2018.

[11] Australian Bureau of Statistics, “Australian Bureau of Statistics,” 2026. [Online]. Available: https://www.abs.gov.au/. Accessed: Feb. 9, 2026.

[12] NYC Open Data, “NYC Open Data,” 2026. [Online]. Available: https://opendata.cityofnewyork.us/. Accessed: Feb. 9, 2026.

[13] U.S. Department of Housing and Urban Development, “Office of Policy Development and Research (PD&R),” 2026. [Online]. Available: https://www.huduser.gov/portal/. Accessed: Feb. 9, 2026.

[14] Pricefinder, “Pricefinder,” 2026. [Online]. Available: https://www.pricefinder.com.au/. Accessed: Feb. 9, 2026.

[15] Property Investment Advisors (PIA), “Property Investment Advisors (PIA),” 2026. [Online]. Available: https://pia.com.au/. Accessed: Feb. 9, 2026.

[16] Zillow Research, “Zillow Research,” 2026. [Online]. Available: https://www.zillow.com/research/. Accessed: Feb. 9, 2026.

[17] M. Zhang, T. Li, Y. Li, and P. Hui, “Multi-view joint graph representation learning for urban region embedding,” in Proc. 29th Int. Joint Conf. Artif. Intell., 2021, pp. 4431–4437.

[18] S. Wu et al., “Multi-graph fusion networks for urban region embedding,” arXiv preprint arXiv:2201.09760, 2022.

[19] S. Zhou, D. He, L. Chen, S. Shang, and P. Han, “Heterogeneous region embedding with prompt learning,” in Proc. AAAI Conf. Artif. Intell., vol. 37, no. 4, 2023, pp. 4981–4989.

[20] F. Sun, Y. Chang, E. Tanin, S. Karunasekera, and J. Qi, “Flexireg: flexible urban region representation learning,” in Proc. 31st ACM SIGKDD Conf. Knowl. Discovery Data Mining V. 2, 2025, pp. 2702–2713.

[21] Z. Yao, Y. Fu, B. Liu, W. Hu, and H. Xiong, “Representing urban functions through zone embedding with human mobility patterns,” in Proc. 27th Int. Joint Conf. Artif. Intell., 2018.

[22] P. Wang, Y. Fu, J. Zhang, X. Li, and D. Lin, “Learning urban community structures: a collective embedding perspective with periodic spatialtemporal mobility graphs,” ACM Transactions on Intelligent Systems and Technology (TIST), vol. 9, no. 6, pp. 1–28, 2018.

[23] H. Wang and Z. Li, “Region representation learning via mobility flow,” in Proc. 2017 ACM Conf. Inf. Knowl. Manage., 2017, pp. 237–246.

[24] Y. Zhang, W. Huang, Y. Yao, S. Gao, L. Cui, and Z. Yan, “Urban region representation learning with human trajectories: a multi-view approach incorporating transition, spatial, and temporal perspectives,” GIScience & Remote Sensing, vol. 61, no. 1, p. 2387392, 2024.

[25] Y. Li, W. Huang, G. Cong, H. Wang, and Z. Wang, “Urban region representation learning with OpenStreetMap building footprints,” in Proc. 29th ACM SIGKDD Conf. Knowl. Discovery Data Mining, 2023, pp. 1363–1373.

[26] F. Sun et al., “Urban region representation learning with attentive fusion,” in Proc. 2024 IEEE 40th Int. Conf. Data Eng. (ICDE), 2024, pp. 4409–4421.

[27] Z. Xu and X. Zhou, “CGAP: urban region representation learning with coarsened graph attention pooling,” arXiv preprint arXiv:2407.02074, 2024.

[28] L. Bai et al., “Geographic mapping with unsupervised multi-modal representation learning from VHR images and POIs,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 201, pp. 193–208, 2023.

[29] S. Hu et al., “A framework for extracting urban functional regions based on multiprototype word embeddings using points-of-interest data,” Computers, Environment and Urban Systems, vol. 80, p. 101442, 2020.

[30] W. Huang, D. Zhang, G. Mai, X. Guo, and L. Cui, “Learning urban region representations with POIs and hierarchical graph infomax,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 196, pp. 134–145, 2023.

[31] R. Xu, W. Huang, J. Zhao, M. Chen, and L. Nie, “A spatial and adversarial representation learning approach for land use classification with POIs,” ACM Transactions on Intelligent Systems and Technology, vol. 14, no. 6, pp. 1–25, 2023.

[32] Y. Lin et al., “Dual contrastive prediction for incomplete multi-view representation learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 4, pp. 4447–4461, 2022.

[33] N. Tempelmeier, S. Gottschalk, and E. Demidova, “GeoVectors: a linked open corpus of OpenStreetMap embeddings on world scale,” in Proc. 30th ACM Int. Conf. Inf. & Knowl. Manage., 2021, pp. 4604–4612.

[34] Z. Wang, H. Li, and R. Rajagopal, “Urban2vec: incorporating street view imagery and POIs for multi-modal urban neighborhood embedding,” in Proc. AAAI Conf. Artif. Intell., vol. 34, no. 1, 2020, pp. 1013–1020.

[35] C. Xiao, J. Zhou, Y. Xiao, J. Huang, and H. Xiong, “Refound: crafting a foundation model for urban region understanding upon language and visual foundations,” in Proc. 30th ACM SIGKDD Conf. Knowl. Discovery Data Mining, 2024, pp. 3527–3538.

[36] Y. Yan et al., “Urbanclip: learning text-enhanced urban region profiling with contrastive language-image pretraining from the web,” in Proc. ACM Web Conf. 2024, 2024, pp. 4006–4017.

[37] M. Chen et al., “MGRL4RE: a multi-graph representation learning approach for urban region embedding,” ACM Transactions on Intelligent Systems and Technology, vol. 16, no. 2, pp. 1–23, 2025.

[38] Y. Luo, F.-l. Chung, and K. Chen, “Urban region profiling via multigraph representation learning,” in Proc. 31st ACM Int. Conf. Inf. & Knowl. Manage., 2022, pp. 4294–4298.

[39] X. Yi, Y. Zheng, J. Zhang, and T. Li, “ST-MVL: filling missing values in geo-sensory time series data,” in Proc. 25th Int. Joint Conf. Artif. Intell., 2016.

[40] M. Liu et al., “Pristi: a conditional diffusion framework for spatiotemporal imputation,” in Proc. 2023 IEEE 39th Int. Conf. Data Eng. (ICDE), 2023, pp. 1927–1939.