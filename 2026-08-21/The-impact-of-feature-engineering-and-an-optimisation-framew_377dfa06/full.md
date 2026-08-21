# The impact of feature engineering and an optimisation framework for ocean colour machine learning

Edson Silva<sup>1</sup> <sup>\*</sup>, Julien Brajard<sup>1</sup>, Simon Cappe<sup>2</sup>, Lasse H. Pettersson<sup>2</sup> and François Counillon<sup>1</sup>

<sup>1</sup>Nansen Environmental and Remote Sensing Center, and Bjerknes Centre for Climate Research, Bergen, Vestland, Norway <sup>2</sup>Nansen Environmental and Remote Sensing Center, Bergen, Vestland, Norway

\*Corresponding author. Email: edson.silva@nersc.no

Received xx xxx xxxx

Keywords: remote sensing, machine learning, ocean colour, feature engineering

## Abstract

Machine learning (ML) is widely used for the development of ocean colour algorithms, but most studies focus on model parameter training and hyperparameter tuning. The optimisation of the data that feeds the models — i.e., Feature Engineering (FE) — is not fully explored. We assess the impact of FE in ocean colour machine learning models and we propose an optimisation framework that includes seven sequenced levels of data transformation: i. band choice, ii. log scaling, iii. spectral shape normalisation, iv. index extraction, v. principal component analysis, vi. feature scaling, and vii. zero-to-one scaling. We demonstrate the application for Multi-layer perceptron, Support Vector Machines, and eXtreme Gradient Boosting Trees on Sentinel-3 OLCI observations in the Norwegian coastal waters. The models are trained to estimate Chlorophyll-a concentration [Chl-a] and Secchi disk depth $( Z _ { \mathrm { s d } } )$ . Results show that accuracy is highly variable among FE found in six studies using Sentinel-3 OLCI and the ones that we optimise. The � range from 0.01 to 0.55 for [Chl-a] and from 0.15 to 0.68 for<sup>[</sup> $Z _ { \mathrm { s d } }$ , where the optimised FE shows the top results. The ML models with optimised FE could also improve by two times the � and reduce up to 63% of the mean absolute error when compared to CHL\_OC4ME and CHL\_NN standard algorithms. Nevertheless, no common optimised FE is found for all target variables and ML models, suggesting that FE optimisation is necessary for each application. Therefore, our proposed framework can be key for improving the accuracy of water quality monitoring in coastal waters.

## Impact Statement

Machine learning (ML) is commonly used for satellite monitoring of water quality in coastal areas. While most studies focus on improving ML parameters and hyperparameters, here we focus on optimising the data that the ML is trained on (i.e., feature engineering — FE). Accuracy varies widely across the FE found in the Sentinel-3 OLCI literature. Hence, we propose an optimisation framework that employs a search algorithm in the FE space to find the best option for feeding the machine learning model. These options include which bands to use, scaling transformations, and index extraction. The optimised FE options depend on the machine learning model and target variable. Nevertheless, applying this framework to each application yields more accurate retrieval models for satellite monitoring of water quality.

## 1. Introduction

Satellite ocean colour is challenging in coastal waters where the optically active constituents are not covarying and chlorophyll-a concentration [Chl-a] estimation by blue and green band ratios regularly fails (Pettersson and Pozdnyakov, 2013). Consequently, other water quality parameters, such as Secchi depth $( Z _ { \mathrm { s d } } )$ , cannot be derived by the same [Chl-a] estimations (Morel et al., 2007a). In these oftencalled Case-2 waters (Morel and Prieur, 1977), a wide range of methodologies and approaches are employed for developing regionally tuned algorithms (IOCCG, 2000; IOCCG et al., 2018). Recently, the use of machine learning (ML) for estimating bio-geochemical and water quality parameters has become popular in the remote sensing monitoring of coastal and inland waters (Brando et al., 2021; Joshi et al., 2024; Lapucci et al., 2023; Pahlevan et al., 2020; Shen et al., 2020; Zoffoli et al., 2025). The expansion of in-situ databases matching satellite observations fosters the increase of ML as it can easily exploit large databases to calibrate retrieval models. Improving the models’ accuracy involves a training procedure to optimise the model parameters, but depends in large part on tuning hyperparameters, such as the type of ML architecture, the size of the model, among others. The transformation of the data that feeds the models – i.e., Feature Engineering (FE) – also strongly affects ML models’ accuracy (Gada et al., 2021; Mumuni and Mumuni, 2025; Verdonck et al., 2024), but its impact on ocean colour ML models is little understood and probably remains ad-hoc. Ocean colour ML models frequently employ FE, such as band selection, remote sensing reflectance $( R _ { r s } )$ scaling, and computation of indexes (Brando et al., 2021; Lapucci et al., 2023; Pahlevan et al., 2020; Zoffoli et al., 2025), which can span a large number of possible unique options when combined. Hence, feature engineering in ocean colour might benefit more by using optimisation techniques than a manual searching approach.

FE consists of preparing and transforming the data into features that an ML algorithm can train on. In a broad definition, FE includes feature selection, univariate transformations, multivariate changes, and domain-specific FE (Verdonck et al., 2024). Despite modern machine learning architectures — e.g., deep learning — being able to extract features during the model optimisation, they are still improved by FE techniques (Fazliani et al., 2025; Mumuni and Mumuni, 2025; Heaton, 2017, 2016). Besides, deep learning typically benefits most when large volumes of data are available (Rather et al., 2024; Safonova et al., 2023), which is often not feasible in ocean colour ML. Calibration relies on in-situ water quality data matching radiometric measurements (in-situ or satellite), which are scarce due to field campaigns costs or geographical constraints for the calibration of regional models. For example, regional algorithms for Sentinel-3 OLCI have been calibrated with the number of samples spanning from 367 to 2,943 (Brando et al., 2021; Joshi et al., 2024; Lapucci et al., 2023; Pahlevan et al., 2020; Shen et al., 2020; Zoffoli et al., 2025). Consequently, these studies use traditional machine-learning algorithms that can better fit small datasets, such as Optimizable Gaussian Regression Models (Zoffoli et al., 2025), Multi-Layer Perceptron (MLP) (Brando et al., 2021), Random Forests (Joshi et al., 2024; Lapucci et al., 2023; Shen et al., 2020), and Mixture Density Networks (Pahlevan et al., 2020). Traditional ML methods are more impacted by FE when compared to deep learning (Heaton, 2016), and, therefore, FE optimisation could greatly enhance ocean colour ML calibration.

Ocean colour FE often starts by choosing which bands to feed into the ML model. Some studies use a few bands (Brando et al., 2021; Shen et al., 2020; Zoffoli et al., 2025), while others employ the whole optical spectrum (Joshi et al., 2024), or only remove a few inadequate bands (Pahlevan et al., 2020). The reason for selecting a few bands or using all of them is not always clear. We suppose that selecting a few bands may avoid over-fitting in data data-poor configuration, but can also be subop timal. Univariate transformations help improve normal distribution, which may increase convergence of the training, such as log scaling (Brando et al., 2021; Pahlevan et al., 2020; Zoffoli et al., 2025), z-score scaling (Brando et al., 2021), and inter-quartile scaling (Pahlevan et al., 2020). These transformations accommodate samples skewed towards low $R _ { r s }$ (Zoffoli et al., 2025), help against outliers, and enforce homogeneous scale across all bands (Pahlevan et al., 2020). Multivariate transformations in ocean colour models, however, are lacking to the extent of our knowledge, but a commonly known method is Principal Component Analysis (PCA) (Verdonck et al., 2024). Domain-specific FE in ocean colour has been little explored as well, but one example is combining maximum band ratio (MBR) with $R _ { r s }$ to feed the ML models (Lapucci et al., 2023). The band ratio may maximise the relationship with [Chl-a] and improve the models’ accuracy (Lapucci et al., 2023). Many more domain-specific FE could be considered, such as normalising the $R _ { r s }$ (Soja-Wo´zniak et al., 2017), estimation of Slopes and Normalised Difference Chlorophyll Index (NDCI), as well as other band ratios (BR) (Cairo et al., 2020). All the aforementioned FE methods can stack in many possible combinations. Pahlevan et al. (2020) FE with Sentinel-3 OLCI, for example, starts with excluding the band at 400 nm, log scaling the $R _ { r s }$ normalising using inter-quartile scaling, and scaling from 0 to 1. Although different FE choices are probably explored during the ML model calibration, FE may remain little optimised considering the many possible combinations. Nevertheless, the FE could be optimised by mirroring the optimisation of ML hyperparameters.

The hyperparameters of ML models span a variety of options and carefully tuning them impacts the accuracy of models. For example, the hyperparameters of the Support Vector Machine regressor (SVM) include the kernel function, penalty factor, epsilon, and other kernel-dependent hyperparameters, such as gamma and degree (Cortes and Vapnik, 1995; Drucker et al., 1997). Finding the optimal hyperparameter configuration involves creating a search space and employing a search algorithm to find the optimal choice evaluated through cross-validation. The search space dimension is given by the number of hyperparameters that span a variety of options. Considering SVM, for example, the options can be few, such as the kernel among linear, radial basis, and sigmoid options, and many, such as any real number above 0 for the penalty factor. Since the search space can extend to many options, Tree of Parzen Estimators (TPE) is a commonly used search algorithm that optimises hyperparameters. This search algorithm is a sequential optimisation that uses Gaussian Mixture Models (GMM) to infer the following hyperparameters in a sequence of trials that maximise the likelihood of improving accuracy (Bergstra et al., 2011, 2013). Roughly speaking, the search can start with 50 random hyperparameter configurations. Then, the best and worst hyperparameters are fitted in GMMs to infer the hyperparameters more likely to increase the accuracy in the next trial. The optimisation stops when the number of trials reaches a given limit and the optimised hyperparameters are the trial that reached the highest accuracy among all. TPE not only converges to optimal configurations faster but also creates ML models more accurate than randomly searching hyperparameters (Bergstra et al., 2013).

Considering that search algorithms are quite successful for optimising the hyperparameter choice by giving a hyperparameter space, our hypothesis is that we can create an FE space and employ the same search algorithm to optimise the FE choice. Hence, our main objective is to assess whether optimising FE yields more accurate ML models in coastal waters for the retrieval of water quality. Our study focuses on the Norwegian coastal and fjord waters, characterised by high colored dissolved organic matter (CDOM) concentration and coccolithophore blooms, and where open ocean (Case-1) optical algorithms perform poorly (Tessin et al., 2024). The target variables are the [Chl-a] and Secchi depth $( Z _ { \mathrm { s d } } )$ , and the predictors are the Sentinel-3 OLCI reflectance. We focus on three ML models: MLP, SVM, and eXtreme Gradient Boosting trees (XGBoost). We calibrate the ML models by adapting FE found in previous studies (Brando et al., 2021; Joshi et al., 2024; Lapucci et al., 2023; Pahlevan et al., 2020; Shen et al., 2020; Zoffoli et al., 2025) to assess the impact of FE on the ML models’ accuracy. Then, we create an FE space to employ the TPE search algorithm and optimise the FE. Finally, we choose the best model to demonstrate its application in satellite images and evaluate increased performance against Sentinel-3 OLCI standard algorithms.

![](images/1c368ee73681d819083bf24dab36175244655f4a76b811bd19808f43c168bb11.jpg)

![](images/dc9e282b3b7c61e2fa5cc63cfe191645a8d282f6c1faee10a5dd15571c2089f3.jpg)

![](images/b3a5fad433520e22a22e5b79114f95ab069d64ce59612d30fe50cffffb4b9b4e.jpg)

![](images/f60ce9262c7fac12386551374c13412d380c5fde96ee7beaaa9319d2b4522f14.jpg)

![](images/286d5525c68f67f01fcc4cd4a67a11c1b1cc57eb2e7d5982713e2f1ba3bda762.jpg)

![](images/55cb236552d9203909aa43d24e4add585a456ceaa4e98c4edaf2c2b11f58d308.jpg)

![](images/bb21e01f94ebd93244eb90b35a6f55ca3914d9273d1a4b58dafcc6afe2c75488.jpg)  
Figure 1. The Norwegian coast and in-situ monitoring stations. Subplot a) shows the subdivision of regions that are zoomed in subplots b-g). Red triangles are the Secchi depth $( Z _ { s d } )$ stations and green circles are the Chlorophyll-a concentration [Chl-a] stations. Most stations have more than one repeated observation as they comefrom continuous monitoring programs (e.g., Økokyst)..

## 2. Material and Methods

## 2.1. Study region

In Norwegian coastal waters, CDOM dominates the absorption at wavelengths shorter than 600 nm, followed consecutively by phytoplankton and non-algal particles (NAP) (Nima et al., 2016). In southern Norway (Figure 1), the inflow from the Baltic Sea and rivers increases the CDOM during the melting of snow in summer. This rich CDOM waters flows through the Norwegian Coastal Current (NCC) and moves northwards along the coast until the Barents Sea Opening. In the fjords of northern Norway, a rapid increase in freshwater inflow caused by storms may lead to a sudden increase in CDOM (Frigstad et al., 2020). Coccolithophore blooms are common in the fjords, NCC and open ocean areas and have a large influence on the back-scattering (Kondrik et al., 2017; Tessin et al., 2024). Furthermore, the optical variability in $R _ { r s }$ , estimated by satellite sensors, is likely influenced by the complicated geometric light field in the region. The sun zenith angle reaches more than 70 degrees towards the winter months, which reduces the total light backscattered to the orbital sensor and increases the signal-to-noise ratio. The mountains around the fjords can potentially create adjacency effects in some areas in the estimated $R _ { r s }$ The low $R _ { r s }$ and the ambient light field make the standard Sentinel-3 OLCI bio-optical algorithms of [Chl-a] perform poorly in the region (Tessin et al., 2024) and pose a challenging task for the calibration of regional algorithms.

## 2.2. Sentinel-3 Ocean and Land Colour Instrument (OLCI) observations

Sentinel-3 OLCI (satellite A) images matching the in-situ samples from national monitoring programs on the same day are accessed from 2016 to 2023. We first access the Sentinel-3 OLCI L1B product with the bands in radiance in 21 bands centred at 400 nm to 1020 nm at 300 m spatial resolution. We then use the Polymer atmospheric correction (Steinmetz et al., 2011) to retrieve the surface reflectance, as they are shown to perform slightly better than other atmospheric corrections in Norwegian fjord waters (Tessin et al., 2024). Polymer employs a spectral matching method relying on a polynomial atmospheric model that encompasses the atmospheric scattering and sun-glint and a bio-optical ocean water reflectance model (Steinmetz et al., 2011). For Sentinel-3 OLCI, the ocean water reflectance model accounts for coastal Case-2 waters and works continuously with open ocean waters (Steinmetz and Ramon, 2018). We use the Polymer version v4.17beta2 and set it to use the NASA auxiliary data for the atmospheric information and correction. We follow the suggested bitmask quality flagging for valid pixels (bitmask $\& \ 1 0 2 3 = = 0 )$ and we remove too bright pixels where the Near-infrared (NIR) reflectance is above 0.1.

The Sentinel-3 OLCI reflectance is extracted from a 3x3 pixel average over each match-up. Since some stations are close to the coastline, some 3x3 window retrieves a few pixels masked as land. About 60% of all match-ups have the whole 3x3 pixel area with valid reflectance estimations. Removing the remaining 40% match-ups would largely affect the quality of ML models and also create a geographical bias, e.g. in the narrow fjord areas, which are important to be represented in the model. Hence, we decided to use all match-ups with a minimum of one pixel available within the 3x3 window.

We assess Sentinel-3 L2B Non-Time Critical (NT) [Chl-a] products estimated by the CHL\_OC4ME and CHL\_NN algorithms and use them as a benchmark for comparing the improvements reached by our FE optimisation framework. Both CHL\_OC4ME and CHL\_NN are standard products of Sentinel-3 OLCI level 2 available on the EUMETSAT catalogue and thus a suitable benchmark. The CHL\_OC4ME algorithm uses the Baseline Atmospheric Correction (BAC) and maximum band ratio algorithm that is designed for estimating [Chl-a] in Case-1 waters (EUMETSAT, 2021). The CHL\_NN is a C2RCC product that estimates [Chl-a] using neural networks with input the reflectance on the top of the atmosphere (EUMETSAT, 2021; Glukhovets et al., 2020; Schiller and Doerffer, 1999). Furthermore, we create a $Z _ { \mathrm { s d } }$ benchmark by applying the following equation (Morel et al., 2007a):

$$
\begin{array} { r } { Z _ { \_ s d } = 8 . 5 - 1 2 . 6 \log _ { 1 0 } \bigl ( [ \mathrm { C h l - a } ] \bigr ) + 7 . 3 6 \log _ { 1 0 } \bigl ( [ \mathrm { C h l - a } ] \bigr ) ^ { 2 } - 1 . 4 3 \log _ { 1 0 } \bigl ( [ \mathrm { C h l - a } ] \bigr ) ^ { 3 } } \end{array}\tag{1}
$$

where $Z _ { \mathrm { s d } }$ (m) is estimated by using [Chl-a] from CHL\_OC4ME and CHL\_NN.

## 2.3. Characterisation of optical water variability

The Sentinel-3 OLCI observations matching the in-situ data are clustered into Optical Water Types (OWTs). We characterise the water masses by OWTs separately for [Chl-a] and $Z _ { \mathrm { s d } }$ match-ups. Only for the OWT characterisation, we divide reflectance by � to retrieve the remote sensing reflectance $( R _ { r s } )$ for comparison with previous studies (Tessin et al., 2024). The K-means algorithm with the Kmeans++ initialisation (Arthur and Vassilvitskii, 2007) is used to cluster the 16 Sentinel-3 OLCI ocean colour bands from 400 nm to 1020 nm. We set the number of OWTs—i.e., clusters—to 9. Note that we use OWTs for describing water masses, and they are not considered in the calibration or optimisation of models.

## 2.4. In-situ Chlorophyll-a and Secchi disk depth observations

In-situ [Chl-a] and Secchi disk depth $( Z _ { \mathrm { s d } } )$ are retrieved from several sources available on a public repository. The [Chl-a] measurements are from Økokyst, a Norwegian ecosystem monitoring program for assessing the water quality of the Norwegian coastal waters, commissioned by the Norwegian Environment Agency. The program runs in cycles every 4 years, with annual reports on the ecosystem status split into several sub-regions along the Norwegian coast. The sampling is made at several permanent locations along the fjords that are repeated every month (Figure 1). The [Chl-a] is made available at the Vannmiljø portal (https://vannmiljo.miljodirektoratet.no/), a web portal from the Norwegian Environment Agency that comprises data retrieved from several monitoring projects. The samples cover different depths, and only the stations with surface samples at the upper 10 m are utilised and averaged when more than one sample is present. We retrieved 625 match-up samples where 506 follow the NS-ISO 5667-9A protocol method, 12 are measured on ferry boxes (automated measurements onboard commercial vessels), and 107 without measurement method description. Since the samples are used for the official assessment of the coastal ecosystem status in Norway, and follow good quality control as defined in Classification of environmental state in water (vanndirektivet 2018, 2018), we have also used the 107 samples without a method description to maximise the number of samples.

Measurements of $Z _ { \mathrm { s d } }$ are retrieved from several sources and commissioned monitoring programs that make their data available through the Vannmiljø web portal. Since $Z _ { \mathrm { s d } }$ is little affected by different methodologies, we use all $Z _ { \mathrm { s d } }$ measurements available in the Vannmiljø web portal (Figure 1) to maximise the number of samples. Hence, 1378 $Z _ { \mathrm { s d } }$ measurements are available for training and testing the models.

## 2.5. Feature engineering

We build a Feature Engineering (FE) space consisting of seven successive levels of feature selection and data processing, where the data flows from the Sentinel-3 OLCI reflectance as input of the first level to the ML model input as the output of the seventh level (Figure 2a). The used Sentinel-3 OLCI reflectance bands are centred at wavelengths (�) 400, 412.5, 442.5, 490, 510, 560, 620, 665, 673.75, 681.25. 708.75. 753.75. 778.75. 865, 885, and 1020 nm. The Bands Choice (BC) is the first level of FE and subsets the data depending on the choice among 8 different options, as described in Algorithm 1. The BC indicates the option (e.g., Red or Blue), X denotes the feature, and L1 refers to the output of level 1. Since this is the first level, X is the Sentinel-3 OLCI reflectance. The options Blue, Red, and Visible are created to select different or all bands in the visible spectrum, while the remaining options are adapted from recent studies following the name of the first authors (Brando et al., 2021; Joshi et al., 2024; Lapucci et al., 2023; Pahlevan et al., 2020; Shen et al., 2020; Zoffoli et al., 2025). Note that some studies use other input information in the ML models, such as day of the year (Zoffoli et al., 2025) and bathymetry (Lapucci et al., 2023), but we have not included them as we want to limit the experiment to the optical calibration. Brando et al. (2021) employs an ensemble of ML models and we took only the option that showed the best accuracy.

The second level is the log scaling (LS) that log-scales the Sentinel-3 OLCI reflectance as described in Algorithm 2. Note that the 0.03 constant is necessary as negative reflectance is present in a few observations.

The third level is the Spectral Shape (SS) normalisation and is described in Algorithm 3. The summing is over all bands on the same sample that depends on the subset bands at level BC.

The fourth level is the Index Extraction (IE), which gives several options that include the band ratios (BR), Normalised Difference of Chlorophyll Index (NDCI), Slopes, and Maximum Band Ratio (MBR). IE is described in Algorithm 4. Note that f means feature and replace � at level 4 as the observations can include features without �.

The fifth level is the PCA and is described in Algorithm 5. The PCA computes the eigenvectors and eigenvalues for the training dataset, while the test dataset is only projected onto the eigenvectors derived from the training data. The resulting transformed test data retains the variance structure defined by the training eigenvectors.

The feature scaling is the sixth level and can apply scaling by the standard deviation (STD) and inter-quartile range (IQR) as described in Algorithm 6.

The zero-to-one scaling (ZOS) is the last level and scales the data between 0 and 1 for using the minimum and maximum range of the feature as described in Algorithm 7. Here, this is the last output and what is fed to the ML models.

Algorithm 1 Selection of $X _ { L 1 }$ bands based on the BC option   
Require: BC option, Sentinel-3 OLCI Reflectance   
Ensure: Selected band set $X _ { L 1 }$   
1: if BC = Blue then   
2: $X _ { L 1 } \gets \{ X _ { 4 0 0 } , X _ { 4 1 2 . 5 } , X _ { 4 4 2 . 5 } , X _ { 4 9 0 } , X _ { 5 1 0 } \}$   
3: else if BC = Red then   
4: $X _ { L 1 } \gets \{ X _ { 5 6 0 } , X _ { 6 2 0 } , X _ { 6 6 5 } , X _ { 6 7 3 . 7 5 } , X _ { 6 8 1 . 2 5 } , X _ { 7 0 8 . 7 5 } \}$   
else if BC = Visible then   
6: $X _ { L 1 } \gets \{ X _ { 4 0 0 } , X _ { 4 1 2 . 5 } , X _ { 4 4 2 . 5 } , X _ { 4 9 0 } , X _ { 5 1 0 } , X _ { 5 6 0 } , X _ { 6 2 0 } , X _ { 6 6 5 } , X _ { 6 7 3 . 7 5 } , X _ { 6 8 1 . 2 5 } \}$   
7: else if BC = Zoffoli then   
8: $X _ { L 1 }  \{ X _ { 4 4 2 . 5 } , X _ { 4 9 0 } , X _ { 5 1 0 } , X _ { 5 6 0 } \}$   
9: else if BC = Brando then   
10: $X _ { L 1 } \gets \{ X _ { 4 9 0 } , X _ { 5 1 0 } , X _ { 5 6 0 } , X _ { 6 7 3 . 7 5 } \}$   
11: else if BC = Shen then   
12: $X _ { L 1 } \gets \{ X _ { 4 1 2 . 5 } , X _ { 4 4 2 . 5 } , X _ { 4 9 0 } , X _ { 5 1 0 } , X _ { 5 6 0 } , X _ { 6 6 5 } \}$   
13: else if BC = Lappucci then   
14: $X _ { L 1 } \gets \{ X _ { 4 1 2 . 5 } , X _ { 4 4 2 . 5 } , X _ { 4 9 0 } , X _ { 5 1 0 } , X _ { 5 6 0 } , X _ { 6 2 0 } , X _ { 6 6 5 } , X _ { 6 7 3 . 7 5 } , X _ { 6 8 1 . 2 5 } \}$   
15: else if BC = Pahlevan then   
16: $X _ { L 1 } \gets \{ X _ { 4 1 2 . 5 } , X _ { 4 4 2 . 5 } , X _ { 4 9 0 } , X _ { 5 1 0 } , X _ { 5 6 0 } , X _ { 6 2 0 } , X _ { 6 6 5 } , X _ { 6 7 3 . 7 5 } , X _ { 6 8 1 . 2 5 } , X _ { 7 0 8 . 7 5 } , X _ { 7 5 3 . 7 5 } , X _ { 7 7 8 . 7 5 } \}$   
17: else if BC = Joshi then   
18: $X _ { L 1 } \gets \{ X _ { 4 0 0 } , X _ { 4 1 2 . 5 } , X _ { 4 4 2 . 5 } , X _ { 4 9 0 } , X _ { 5 1 0 } , X _ { 5 6 0 } , X _ { 6 2 0 } , X _ { 6 6 5 } , X _ { 6 7 3 . 7 5 } , X _ { 6 8 1 . 2 5 } , X _ { 7 0 8 . 7 5 } , X _ { 7 5 3 . 7 5 } \} ,$   
19: $X _  7 7 8 . 7 5 , X _ { 8 6 5 } , X _ { 8 8 5 } , X _ { 1 0 2 0 } \}$   
20: end if

```perl
Algorithm 2 Computation of $X _ { L 2 , \lambda }$ based on the LS option
Require: LS option, $X _ { L 1 , \lambda }$
Ensure: Computed $X _ { L 2 , \lambda }$
1: if $\mathrm { L } S = \mathrm { O N }$ then
2: $X _ { L 2 , \lambda } \gets \log _ { 1 0 } ( X _ { L 1 , \lambda } + 0 . 0 3 )$
3: else if $\mathrm { L S } = \mathrm { O F F }$ then
4: $X _ { L 2 , \lambda }  X _ { L 1 , \lambda }$
5: end if
```

Algorithm 3 Computation of $X _ { L 3 , \lambda }$ based on the SS flag   
Require: SS option, $X _ { L 2 , \lambda }$   
Ensure: Computed $X _ { L 3 , \lambda }$   
1: if SS = ON then   
2: $X _ { L 3 , \lambda }  X _ { L 2 , \lambda } / \sum X _ { L 2 }$   
3: else if SS = OFF then   
4: $X _ { L 3 , \lambda }  X _ { L 2 , \lambda }$   
5: end if

## 2.6. Machine learning models

We employ MLP, SVM, and XGBoost models for covering a variety of ML architectures, as FE can depend on the ML model. MLP is based on neural networks, SVM is based on support vectors, and XGBoost is based on decision trees. MLP is a type of artificial neural network that consists of several blocks (layers) of interconnected neurons that are linearly combined before being passed through a nonlinear function (activation). During training, the model learns the coefficients of the weighted sum performed in each layer by minimising a loss function that quantifies the error between predicted and true values. The optimisation techniques are often based on the gradient descent algorithm, with gradients efficiently computed through back-propagation. When samples are processed during optimization, the amount by which the coefficients are updated depends on the learning rate. One epoch is when all samples are passed as batches through the optimisation function. Hence, the MLP hyperparameters needed to be optimised are the number of neurons in each layer, the activation function, the learning rate, and the regularisation strength (Figure 2). A detailed explanation of MLPs can be found in Hinton (1989).

Algorithm 4 Computation of $X _ { L 4 , f }$ based on the IE option   
Require: IE option, $X _ { L 3 , \lambda }$   
Ensure: Computed $X _ { L 4 , f }$   
1: if IE = BR then   
2: $X _ { L 4 , f }  X _ { L 3 , \lambda } \parallel ( X _ { L 3 , 4 4 2 . 5 } / X _ { L 3 , 5 6 0 } ) \parallel ( X _ { L 3 , 4 4 2 . 5 } / X _ { L 3 , 5 1 0 } )$   
3: $\parallel \left( X _ { L 3 , 4 4 2 , 5 } / X _ { L 3 , 4 0 0 } \right) \parallel \left( X _ { L 3 , 5 6 0 } / X _ { L 3 , 6 6 5 } \right) \parallel \left( X _ { L 3 , 6 7 3 , 7 5 } / X _ { L 3 , 6 6 5 } \right) \parallel \left( X _ { L 3 , 6 8 1 , 2 5 } / X _ { L 3 , 6 6 5 } \right)$   
4: else if $\mathrm { I E } = \mathrm { N D C I }$ then   
5: $\begin{array} { r } { X _ { L 4 , f }  X _ { L 3 , \lambda } \parallel \frac { X _ { L 3 , 6 7 3 , 7 5 } - X _ { L 3 , 6 6 5 } } { X _ { L 3 , 6 7 3 , 7 5 } + X _ { L 3 , 6 6 5 } } \parallel \frac { X _ { L 3 , 6 8 1 , 2 5 } - X _ { L 3 , 6 6 5 } } { X _ { L 3 , 6 8 1 , 2 5 } + X _ { L 3 , 6 6 5 } } \parallel \frac { X _ { L 3 , 5 6 0 } - X _ { L 3 , 6 6 5 } } { X _ { L 3 , 5 6 0 } + X _ { L 3 , 6 6 5 } } } \end{array}$   
6: else if IE = Slope then   
7: $X _ { L 4 , f }  X _ { L 3 , \lambda } \parallel$ �<sub>�3,673.75</sub> −�<sub>�3,665</sub> ∥ $\frac { X _ { L 3 , 6 8 1 . 2 5 } - X _ { L 3 , 6 6 5 } } { \textmd { c e } }$ 681.25−665 ∥ $\underbrace { X _ { L 3 , 5 6 0 } - X _ { L 3 , 6 6 5 } } _ { = }$ 665−560   
8: else if IE = MBR then   
9: $X _ { L 4 , f }$ ← $X _ { L 3 , \lambda }$ ∥ $( X _ { L 3 , 4 4 2 . 5 } / X _ { L 3 , 5 1 0 } )$ ∥ $( X _ { L 3 , 4 9 0 } / X _ { L 3 , 5 1 0 } )$ ∥   
(max $( X _ { L 3 , 4 4 2 . 5 } , X _ { L 3 , 4 9 0 } ) / X _ { L 3 , 5 1 0 } )$   
10: else if IE = All then   
11: $X _ { L 4 , f }  X _ { L 3 , \lambda }$ ∥ BR ∥ NDCI ∥ Slope ∥ MBR   
12: else if IE = OFF then   
13: $X _ { L 4 , f }  X _ { L 3 , \lambda }$   
14: end if   
Algorithm 5 Computation of $X _ { L 5 , f }$ based on the PCA option   
Require: PCA flag, $X _ { L 4 , f }$   
Ensure: Computed $X _ { L 5 , f }$   
1: if $\mathrm { P C A } = \mathrm { O N }$ then   
2: $X _ { L 5 , f } \gets \mathrm { P C A } ( X _ { L 4 , f } )$   
3: else if PCA = OFF then   
4: $X _ { L 5 , f }  X _ { L 4 , f }$   
5: end if   
Algorithm 6 Computation of $X _ { L 6 , f }$ based on the FS option   
Require: FS flag, $X _ { L 5 , f }$   
Ensure: Computed $X _ { L 6 , f }$   
1: if $\mathrm { F S } = \mathrm { S T D }$ then   
2: $\begin{array} { r } { X _ { L 6 , f } \gets \frac { \overline { { X } } _ { L 5 , f , s } ^ { - } - \mathrm { M E A N } ( X _ { L 5 , f } ) } { \mathrm { S T D } ( X _ { \tau \varsigma } \bf { \Sigma } _ { \epsilon } ) } } \end{array}$ $\overline { { \mathrm { S T D } ( X _ { L 5 , f } ) } }$   
3: else if $\mathrm { F S } = \mathrm { I Q R }$ then   
4: $\begin{array} { r } { X _ { L 6 , f } \gets \frac { \mathbf { \tilde { { X } } } _ { L 5 , f , s } - \mathbf { M E D I A N } ( X _ { L 5 , f } ) } { \Pi \mathrm { { R } } ( X _ { r \textnormal { \tiny { c } } \epsilon } ) } } \end{array}$   
$\overline { { \mathrm { I Q R } ( X _ { L 5 , f } ) } }$   
5 else if FS = OFF then   
6: $X _ { L 6 , f }  X _ { L 5 , f }$   
7: end if   
Algorithm 7 Computation of $X _ { L 7 , f }$ based on the ZOS option   
Require: ZOS flag, $X _ { L 6 , f }$   
Ensure: Computed $X _ { L 7 , f }$   
1: if $\mathrm { \Delta ^ { \cdot } Z O S = O N }$ then   
2: $\begin{array} { r } { X _ { L 7 , f } \gets \frac { X _ { L 6 , f } - \operatorname* { m i n } ( X _ { L 6 , f } ) } { \operatorname* { m a x } ( X _ { L 6 , f } ) - \operatorname* { m i n } ( X _ { L 6 , f } ) } } \end{array}$   
3: else if ${ \mathrm { Z O S } } = { \mathrm { O F F } }$ then   
4: $X _ { L 7 , f }  X _ { L 6 , f }$   
5: end if

![](images/f86f85e77a4e281fd740583a8399347cb8d9cb8be237824ddce2d3daf6d1b312.jpg)  
Figure 2. Scheme offeature engineering (FE) and Machine Learning (ML) hyperparameters optimisation. Subplot a) exhibits the FE space where the Sentinel-3 OLCI reflectance feeds and is transformed across seven levels until reaching the ML model. Subplot b) shows the hyperparameter spacefor MLP, SVM, and XGBoost. Subplot c) shows the cross-validation methodfor estimating the �<sup>2</sup> ofa given FE and ML hyperparameters configuration.

Support Vector Machines (SVM) regressor is the generalisation of the SVM. The kernel function configures the hyperplane to fit the samples, and popular choices are linear, radial basis function (RBF), and sigmoid. For a given kernel function, the SVM finds the hyperplane that holds the maximum number of observations within a tube of width � to solve the optimisation problem. Samples falling inside the �-width tube contribute 0 to the optimisation. The penalty factor � is a regularisation parameter in the optimisation function that controls the trade-off between minimising the error and the complexity of the model. Finally, the RBF kernel introduces a hyperparameter , which controls the influence of a single sample. A smooth $\gamma ,$ in our experience, gives well-adjusted ML models and can be obtained as a function of the input variance: $\gamma _ { \mathrm { s m o o t h } } = 1 / ( n _ { \mathrm { f e a t u r e s } } \times \mathrm { V A R } ( X _ { L 7 } ) )$ , where $n _ { \mathrm { f e a t u r e s } }$ is the number of features and VAR is the variance estimator. Hence, we introduce a scaling factor $\gamma _ { \mathrm { s c a l i n g } }$ to estimate deviations from $\gamma _ { \mathrm { s m o o t h } }$ , given by $\gamma = \gamma _ { \mathrm { s m o o t h } } \times \gamma _ { \mathrm { s c a l i n g } }$ . During hyperparameter tuning, we optimise $\gamma _ { \mathrm { s c a l i n g } }$ rather than the absolute value of �. A deeper explanation of SVMs and regression tasks can be found in Cortes and Vapnik (1995) and Drucker et al. (1997).

XGBoost (for eXtreme Gradient Boosting) is a version of the gradient boosted trees algorithm. It is an ensemble of classification and regression trees (CART) that can be applied for classification and regression tasks similar to Random Forest. However, the key difference is in how the trees are trained. Gradient boosting refers to sequentially adding weak estimators (CART) that are trained to correct the errors of their predecessors until the maximum number of boosting rounds, or a desired level of accuracy is reached. The term eXtreme comes from the strong regularisation that includes the number of leaves (gamma) in each CART and the terms on weights (lambda). Hence, XGBoost hyperparameters include the number of estimators, gamma, lambda, and learning rate, which is multiplied by the output of each sequential tree. Furthermore, other hyperparameters include the minimum and maximum depth of the trees, sub-sampling rate, and feature sampling rate. More details are found in Chen and Guestrin (2016).

## 2.7. Optimisation offeature engineering and ML hyperparameters

The FE and ML hyperparameters are combined in a single space – hereafter referred to as pipeline space – for optimising both concurrently. The optimisation only uses the train dataset for [Chl-a] (2016- 2019) and $\mathrm { Z _ { s d } } \left( 2 0 1 6 - 2 0 2 0 \right)$ . First, the training dataset is randomly divided into 3 splits, each yielding 3 folds (Figure 2a). One fold is held for validation, while the two remaining folds are for fitting the ML models. The three validation folds are combined in a single validation dataset where the coefficient of determination $( \mathbb { R } ^ { 2 } )$ is estimated. The $\mathbf { R } ^ { 2 }$ is computed for each of the random splits and a final averaged $\mathbf { R } ^ { 2 }$ is used for ranking the pipeline options.

The TPE optimisation starts with 20 trials using random pipeline configurations by using probability distributions of each configuration space. The initial probability distributions are nominal (e.g., all FE options), discrete (e.g., number of neurons in MLP), and continuous (e.g., C in SVM). For the categorical pipeline configurations, the space is the given optional categories with a uniform probability distribution in each option, such as Red and Blue for the BC FE. For discrete and continuous configurations in the hyperparameter space, the space is defined by a minimum and a maximum in a uniform or log-uniform distribution. In the TPE jargon, the categorical, uniform, and log uniform are described as choice, uniform, and log-uniform spaces (Figure 2b). Hence, the space for initially ranking the configurations is given and described for FE in Figure 2a and ML hyperparameters in Figure 2b.

The initial 20 random configurations are ranked and split into the best group for the top ranking, where N is the number of trials (N=20 at the start), and the worst group with the remaining trials. For each pipeline choice – FE and ML hyperparameter choices – TPE fits a Gaussian Mixture Model (GMM) for the best group with the prior optimised configuration and another for the worst group. Then, 24 candidates are drawn from and the one with the maximum value is chosen as the configuration for the next trial. In other words, the TPE algorithm selects the candidate by balancing exploitation (high probability under the good distribution) with exploration (low probability under the bad distribution). These optimisations progress one by one until reaching 500 trials. The best configuration among all those trials is chosen as the optimal configuration.

## 2.8. Experiments design

We draw several experiments for assessing the impact of FE choices and FE optimisation using TPE. For the experiment named “Optimised FE”, we employ TPE for the whole pipeline space, which means that we optimise the FE and ML hyperparameters concurrently. For the remaining experiments, the FE space is constrained by the configuration found in the literature and TPE optimises only the ML hyperparameters. The FE experiments are described in Table 1. For all experiments, the TPE optimisation is run 30 times to assess the accuracy uncertainty that resulted from the TPE. Furthermore, the optimal FE frequency in the FE optimised experiment is estimated through the relative frequency of the FE choice in the 30 Optimised FE TPE runs.

<table><tr><td rowspan=1 colspan=1>Experiment name</td><td rowspan=1 colspan=1>FE configuration</td></tr><tr><td rowspan=1 colspan=1>Optimised FE</td><td rowspan=1 colspan=1>TPE is employed to the whole FE space.</td></tr><tr><td rowspan=1 colspan=1>Zoffoli et al. (2025) FE</td><td rowspan=1 colspan=1>BC = Zoff. and LS = ON.</td></tr><tr><td rowspan=1 colspan=1>Brando et al. (2021) FE</td><td rowspan=1 colspan=1>BC = Bran., LS = ON, and FS = STD.</td></tr><tr><td rowspan=1 colspan=1>Joshi et al. (2024) FE</td><td rowspan=1 colspan=1>BC = Joshi.</td></tr><tr><td rowspan=1 colspan=1>Pahlevan et al. (2020) FE</td><td rowspan=1 colspan=1>BC = Pahl., LS = ON, FS = IQR, and ZOS = ON.</td></tr><tr><td rowspan=1 colspan=1>Lapucci et al. (2023) FE</td><td rowspan=1 colspan=1>BC = Lapp. and IE = Lapp.</td></tr><tr><td rowspan=1 colspan=1>Shen et al. (2020) FE</td><td rowspan=1 colspan=1>BC = Shen.</td></tr></table>

Table 1. Feature engineering (FE) configurations for the different experiments. Omitted options are set to OFF.

## 2.9. Testing of the models

The test dataset for [Chl-a] ranges from 2021 to 2023, and $Z _ { \mathrm { s d } }$ ranges from 2022 to 2023 and comprises 38% and 45% of total match-ups, respectively. The number of $\mathrm { { Z _ { s d } } }$ observations is largely biased to recent years, and hence we need to shift one extra year for training while reducing one for testing. Setting the test dataset to the recent years helps to avoid any potential geographical and temporal bias (McGovern et al., 2022) caused by autocorrelation of samples between the train and test datasets. Furthermore, the temporal split considers whether the calibrated ML models still remain accurate in any potential recent changes in the optical variability. For example, darkening of coastal waters has been observed in the Norwegian coast in the last decades (Aksnes et al., 2009). The statistical evaluation employed includes the Pearson correlation (�), the Root Mean Squared Error (RMSE), and the Mean Absolute Percent Error (MAPE). Note that � is estimated using the targets log-scaled at base 10, while MAPE and RMSE are computed in the original scale of the target variable. The Pearson correlation coefficient is defined as:

$$
R = \frac { \sum _ { i = 1 } ^ { n } ( y _ { i } - \bar { y } ) ( \hat { y } _ { i } - \bar { \hat { y } } ) } { \sqrt { \sum _ { i = 1 } ^ { n } ( y _ { i } - \bar { y } ) ^ { 2 } } \sqrt { \sum _ { i = 1 } ^ { n } ( \hat { y } _ { i } - \bar { \hat { y } } ) ^ { 2 } } } ,\tag{1}
$$

where $y _ { i }$ and $\hat { y } _ { i }$ denote the observed and predicted values, respectively, and �¯ and $\bar { \hat { y } }$ are their means. The Root Mean Squared Error is given by:

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } } ,\tag{2}
$$

and the Mean Absolute Percent Error is expressed as:

$$
\mathrm { M A P E } = \frac { 1 0 0 } { n } \sum _ { i = 1 } ^ { n } \left| \frac { y _ { i } - \hat { y } _ { i } } { y _ { i } } \right| .\tag{3}
$$

## 3. Results

## 3.1. Optical variability ofNorwegian coastal waters and match-ups

The Norwegian coastal waters exhibit a rich OWT variability that can be illustrated in a single OLCI scene (Figure 3a). Along the southern coast and in the fjords, dark waters with low $\mathrm { R } _ { \mathrm { r s } }$ occur in the NCC. Blue waters occur towards the open ocean and, in some inner fjords, high scattering waters in the green and blue range (cyan colour) appear. Green waters are also present and occur along the coastline. Dark waters are likely the CDOM-rich waters that are often carried by the NCC or more common through episodic high freshwater inputs in northern Norway (Frigstad et al., 2020). Blue waters are the oligotrophic open ocean waters (Case-1) with little influence from the terrestrial input of CDOM and sediments. The high scattering and cyan waters are probably coccolithophore blooms that often occur in the fjords, by the coast and in the open ocean (Tessin et al., 2024). Green waters could be non-coccolithophore blooms that occur during the summer and are potentially harmful (Pettersson and Pozdnyakov, 2013; John et al., 2022).

Considering the [Chl-a] match-ups (Figure 3b), dark OWTs (1-3) with $\mathrm { R } _ { \mathrm { r s } }$ average lower than 0.002 are 67.8% of observations. Cyan and green like OWTs (4 and 8) with relatively high $\mathrm { R } _ { \mathrm { r s } }$ values are 8.8% of the retrieved matches. OWTs with higher $\mathrm { \sf R } _ { \mathrm { r s } }$ in blue (5, 6, 7, and 9), which indicate little influence of CDOM and blooms, are 13% of the observations. The OWTs observed in the $Z _ { \mathrm { s d } }$ matchups exhibit a similar distribution but with the ones related to a blue shape exhibiting negative $\mathrm { R } _ { \mathrm { r s } }$ around 700 nm (Figure 3c). Negative $\mathrm { R } _ { \mathrm { r s } }$ also occur in a few samples in the blue region of the spectrum in dark waters in both [Chl-a] and $Z _ { \mathrm { s d } }$ match-ups (not shown). Although they might present some constraints to physical models, from an ML point of view, they are not a problem as long as the reflectance variability corresponds to the optically active constituents.

## 3.2. FE impact on ML models’ accuracy

Using the FE available in the literature causes a large variability in the model’s accuracy across all ML models and target variables (Figure 4). The R, MAE, and MAPE variability between FE choices is higher in the MLP models but still present in the SVM and XGBoost for both [Chl-a] and $Z _ { \mathrm { s d } }$ . Furthermore, the accuracy variability across TPE optimisation runs (n=30) is higher (wider box-plots) in MLP models than SVM and XGBoost, indicating that the MLP configurations may remain unoptimized after 500 trials in some instances. Once the FE is optimised, the calibrated models reach the highest accuracy in both [Chl-a] and $Z _ { \mathrm { s d } }$ . Remarkably, Pahlevan et al. (2020) FE reaches accuracy close to the optimised FE for SVM.

## 3.3. Optimised FE choices

The optimised FE exhibits irregular patterns across ML models and targets (Figure 5). In the [Chla], all ML models’ band choices level converge to visible options (Vis. and Lap.) in more than 90% of optimisation runs (n=30). The $Z _ { \mathrm { s d } }$ models bands’ choice level converges mostly to visible with NIR options (Pahl. and Joshi). Log scaling level converges to ON for most of $Z _ { \mathrm { s d } }$ ML models and [Chl-a] XGBoost, while indifferent (close to 50%) for [Chl-a] MLP and SVM. The spectral shape level convergence exhibits no clear pattern, being predominantly ON and OFF or indifferent in several instances. The convergence in the index extraction level seems indifferent for the $Z _ { \mathrm { s d } }$ ML models, except for XGBoost, where it converges to the All option in 53.3% of runs. For [Chl-a], the index extraction level for MLP and SVM converges to Lapp. option while XGBoost converges more often to the NDCI option in 40% of the runs. The PCA level converges to ON in most cases and up to 100% XGBoost runs for [Chl-a] and $Z _ { \mathrm { s d } }$ . Lastly, the feature scaling and zero-to-one scaling exhibit no clear pattern among the target and ML models.

![](images/1512f9d08daf36bc683f2c21944b85d26451a28cba4b8aeef33bea7c900de4eb.jpg)

![](images/2eddb2d2b46b428543c8951980316433969094da70cb0371d1dc786d22fe11c0.jpg)

![](images/8c228c98c99a35343558f4d66ff99e0dfa1f84682d16ea448f01dfe6cf25e581.jpg)  
Figure 3. Optical water type variability along the Norwegian coast and in-situ match-ups. Subplot a) is an example of a Sentinel-3 OCLI RGB-image (665 nm, 560 nm, and 443 nm) composition covering the southern Norwegian coastal andfjord waters (2023-06-15) centred at $5 9 . 5 \mathrm { ^ \circ N }$ and scaled with $R _ { r s }$ between 0 and 0.01 $s r ^ { - 1 }$ , resolving the colour contrast in the fjord-coastal waters. Subplot b) exhibits the estimated OWTsfor the [Chl-a] match-ups $( n = 6 2 5 )$ . Subplot c) shows the estimated OWTsfor the $Z _ { s d }$ match-ups (n = 1378).

## 3.4. Improvement against standard OC retrieval models

Sentinel-3 OLCI standard retrieval models – CHL\_OC4ME and CHL\_NN – exhibit limited accuracy for estimating [Chl-a] in the Norwegian coastal waters (Figure 6). Considering both options, the OC4ME algorithm that is designed for Case-1 waters demonstrates slightly better results than the CHL\_NN that is designed for turbid waters. Compared with the OC4ME, the average estimated by the 30 SVM models (referred to as NOR\_SVM) increases the R by 2.4 times and reduces the MAE by 52%. When aggregating the match-ups in a seasonal time series (Figure 6d-f), in-situ observations exhibit a [Chl-a] peak in April and a later peak in October, which are the timing of the spring and autumn blooms of high latitudes (Silva et al., 2021). The CHL\_OC4ME and CHL\_NN algorithms fail to reproduce such a seasonal pattern and show high-frequency variability along the months. Our new NOR\_SVM, on the other hand, better demark the spring and autumn blooms similarly to in-situ observations, although with a slight underestimation. Furthermore, NOR\_SVM demonstrate a high geographical variability with low [Chl-a] estimated in the open ocean and part of the inner fjords, moderate [Chl-a] in the inner fjords and in the cyan waters, and high [Chl-a] in the green waters (Figure 6i). CHL\_OC4ME estimates high [Chl-a] in most inner fjord areas where cyan blooms and dark waters are present. CHL\_OC4ME has an overall high value that is probably related to the overestimation due to CDOM and particles $( \mathrm { e . g . , C a C O _ { 3 } }$ from coccolithophore). Both non-pigment constituents reduce the blue and green ratios that lead to Case-1 water algorithms to overestimate [Chl-a]. CHL\_NN, on the contrary, avoid the overestimation issue as it is designed for turbid and coastal waters. However, the CHL\_NN fails to estimate areas of high [Chl-a] in the green waters.

![](images/f0d3484d0fd093d3f20fc734c32ad6ad1b51752901c37b9537851b1ca14d7e34.jpg)  
Figure 4. Statistical results (R, MAE, and MAPE) for the FE experiments conducted. Subplots a, c, and e) are for [Chl-a] and b, d, andf) are for $Z _ { s d } .$ R is shown in subplots a and b), MAE is shown in subplots c and d), and MAPE is shown in subplots e and f). The box plots represent the distribution over 30 TPE optimisations for each FE experiment, shown in different colours- The optimised FE are highlighted with red ellipses.

![](images/bf9643cdd32956a0a12cfb501fc70fd559bab5e275dc5ab624c500501caa7c22.jpg)  
Figure 5. Distribution of optimised FE choices. Results for [Chl-a] are shown in subplots a, c, and e) and for $Z _ { s d }$ in b, d, and f). FE choices are shown for MLP in subplots a and b), for SVM in subplots c and d), and for XGBoost in subplots e and f). The percent values in each choice are the relative frequency of that choice, estimated as the best option across the 30 TPE optimisations. For example, 93.3% ofLS in ONfor $Z _ { s d }$ XGBoost models (in d, e, andf) corresponds to 28 of30 runs.

![](images/ce72705bf1140e56bcd29223f3783f1016e1d5f7f0b105347b8be687d45cad34.jpg)  
Figure 6. Sentinel-3 OLCI [Chl-a] estimations. Subplots a–c) exhibit the matching of in-situ samples with the estimates by a) standard CHL\_OC4ME, b) CHL\_NN, and c) our NOR\_SVM algorithms. Note that fewer samples are matched here because some points are masked as invalid data from the standard L2B OC product. The matching observations aggregated in seasonal time series are shownfor d) standard CHL\_OC4ME, e) CHL\_NN, and f) our NOR\_SVM algorithms, where dots are the monthly averages and bars represent standard deviations. The retrieved maps (2023-06-15) of [Chl-a] distribution are shown for g) standard CHL\_OC4ME, h) CHL\_NN, and i) our NOR\_SVM. NOR\_SVM corresponds to the average ofthe 30 SVM models calibrated in this study.

The $Z _ { \mathrm { s d } }$ derived by the equation 1 using the standard CHL\_OC4ME and CHL\_NN also demonstrates limited accuracy along the Norwegian coast (Figure 7). Hence, the NOR\_SVM substantially increase R by 1.9 times and reduces MAE by 63% when compared to the CHL\_OC4ME. The seasonal average in-situ $Z _ { \mathrm { s d } }$ exhibits peaks in April and in July to August that probably result from variations in incident light and light attenuation. Next to the winter months (e.g., February and October), the incident light on the ocean surface is low as the sun zenith angle is high, which might result in the low $\mathrm { { Z _ { s d } } }$ . During the summer, CDOM increases due to river runoff (Frigstad et al., 2020) and might explain the reduction observed in May and July. CHL\_OC4ME fails to estimate this pattern and CHL\_NN slightly describes the same in-situ pattern, but with much larger seasonal variations. On the other hand, the NOR\_SVM provides a closer relationship with the pattern observed in-situ. Since $\mathrm { { Z _ { s d } } }$ using CHL\_OC4ME and CHL\_NN are derived from the [Chl-a] estimations, the spatial distributions follow a similar pattern. The CHL\_OC4ME estimates shallow $\mathrm { { Z _ { s d } } }$ in the inner fjords associated with dark, cyan, and green waters. The standard CHL\_NN estimate deeper $Z _ { \mathrm { s d } }$ in most regions, including inner fjords, as a result of the [Chl-a] underestimation. The NOR\_SVM estimates shallow $Z _ { \mathrm { s d } }$ mostly in the green and cyan waters, while dark and blue waters exhibit deeper $Z _ { \mathrm { s d } }$

## 4. Discussion

Regional bio-optical algorithms often perform better than the ones designed to open oceans (Case-1) or to other coastal areas with contrasting optical variability. Tessin et al. (2024) has previously demonstrated this limitation in the Norwegian coastal waters for Sentinel-3 OLCI products and we corroborate the same result for our match-up data. The weak performance of standard Sentinel-3 OLCI algorithms might be due to high optical variability and the prevalence of low reflectance waters dominated by CDOM. The calibration of regional algorithms shows up as necessary and the ML usefulness for solving this optimisation task will continue gaining popularity as the records of public field monitoring data and satellite observations continue growing. The increasing use of ML will benefit from know-how practices that improve the optimisation of ML models. One example is including the day of the year with $R _ { r s }$ that may help the ML model to account for seasonal variations in bio-optical relationships (Zoffoli et al., 2025). Another example is employing an ensemble of ML models with varying wavelengths that can account for the temporal and spatial variation of atmospheric correction uncertainties (Brando et al., 2021). To the extent of our knowledge, feature engineering is so far explored ad hoc and left as an initial step in data pre-processing. However, our experiment demonstrates that the FE plays a large role in the accuracy of ML models. When adapting FE from previous studies to our experiments in the Norwegian coastal waters, we demonstrate that accuracy variability in terms of R, RMSE, and MAPE is higher among the FE configurations than across three different ML architectures. Since FE can span a variety of options, we propose a framework that can optimise FE with similar rigour to that used for calibrating ML hyperparameters. Our study demonstrates that employing TPE across different ML models and target variables for optimising FE yields more accurate models than using FE available in the literature.

The main challenge — similar to hyperparameters — is that no common optimal FE is found for all cases. Our results suggest that the optimal FE options depend on the ML model and target variable. Despite some FE optimisations exhibiting some common convergences, such as log-scaling reflectance $Z _ { \mathrm { s d } }$ models and maximising the number of visible bands selected for [Chl-a] models, this outcome may depend on the regional optical conditions. The surface reflectance broadly varies along the Norwegian coast as it is composed of waters with high absorption caused by CDOM and high scattering caused by coccolithophore blooms, both contributing to the reduction of water transparency and $Z _ { \mathrm { s d } }$ Log scaling might help allocate the high reflectance variability to a normal distribution and support the training of ML models. For [Chl-a], on the other hand, log scaling reflectance seems less important as it is only found relevant for XGBoost. Furthermore, high NIR reflectance commonly observed in eutrophicate environments (Cairo et al., 2019) is not present in our match-ups since the measured [Chl-a] are relatively low. Hence, the NIR bands are probably a source of noise rather than something useful for estimating [Chl-a] using ML models in our region. This might explain the band choice optimisations converging mostly to visible bands without the NIR being included. In an eutrophicate environment, optimising FE might include the NIR bands.

![](images/6b2c06f901e7bc95a189cc33b0aa5cb6ac156947e19a7adb239ce4598ace4371.jpg)  
Figure 7. Sentinel-3 OLCI $Z _ { s d }$ estimations. Subplots a–c) exhibit the matching of in-situ samples with the estimates by a) standard CHL\_OC4ME, b) CHL\_NN, and c) our NOR\_SVM algorithms. Note that fewer samples are matched here because some points are masked as invalid data from the standard L2B OC product. The matching observations aggregated in seasonal time series are shown for d) CHL\_OC4ME, e) CHL\_NN, and f) NOR\_SVM algorithms, where dots are the monthly averages and bars represent standard deviations. The retrieved maps (2023-06-15) of $Z _ { s d }$ distribution are shownfor g) standard CHL\_OC4ME, h) CHL\_NN, and i) our NOR\_SVM algorithms. NOR\_SVM corresponds to the average ofthe 30 SVM models calibrated in this study, and CHL\_OC4ME and CHL\_NN use Eq. 1 with their estimated [Chl-a] as inputfor retrieval ofthe Secchi disk depth.

A common criticism made against ML models is whether physical meaning exists from the optical input (e.g., reflectance) to the target variable (e.g., [Chl-a] and $\mathrm { \sf Z _ { s d } } )$ . While semi-empirical and semianalytical models exhibit a more transparent relationship (Lee et al., 2002; Morel et al., 2007b), ML models transform the data in such a way that might limit the physical understanding. Optimising FE can, to some extent, restrict this understanding, as reflectance might be transformed several times prior to feeding the ML model. For example, the relationship between reflectance and [Chl-a], where the original reflectance is sequentially log-scaled, has its principal components extracted, scaled by their variance, scaled from zero to one, and transformed by the given ML model architecture. Despite this drawback, employing such transformations yields substantially improved estimations in coastal waters, and it should not be easily discarded. Besides, we demonstrate that despite the sequence of transformations, we can still correct common issues in coastal waters and represent well the expected natural variability.

The main concern in our region is the [Chl-a] overestimation caused by high CDOM that can lead to geographical and seasonal biases. For example, the Case-1 waters CHL\_OC4ME algorithm estimates excessively high [Chl-a] in many areas where the NOR\_SVM estimate low values. The CHL\_OC4ME led to several false positives of high biomass algae blooms, which can potentially mislead the detection of harmful ones that occur in Norway (Pettersson and Pozdnyakov, 2013; John et al., 2022). Although the CHL\_NN is advocated as an alternative for coastal waters, its accuracy performs worse than CHL\_OC4ME along the Norwegian coast and fjords, where our match-up data is located. Conversely, our calibrated models correct the overestimation issue and provide a more realistic geographical and seasonal variability of [Chl-a] — as well as $Z _ { \mathrm { s d } }$ — in the waters they are trained for. The NOR\_SVM slightly underestimate the high and overestimates the low [Chl-a] observed in situ, but this is probably caused by the smoothing of the grid cell area in satellite observations. The same issue is also observed in the open ocean of the Norwegian Sea when using the Ocean Colour Climate Change Initiative (OC-CCI) data (Silva et al., 2021). Despite the smoothing of [Chl-a], the NOR\_SVM can distinguish well the bloom areas from oligotrophic regions along the coast.

## 5. Conclusion

We introduce a framework for optimising FE and improving the accuracy of ML ocean colour coastal products. The framework is demonstrated in the Norwegian coastal waters, which depicts a wide panel of challenges met: Atmospheric pollution, complex coastline, and high optical variability and complexity. We show that FE can be more important than the pure ML model choice. The TPE search algorithm optimises the FE configuration among many options drawn in an FE space by using the same rigour as for hyperparameter tuning. The FE space works as a pipeline where the input data (reflectance) flows through feature selection and data transformations applied in seven sequenced levels. Although our FE space is drawn by adapting the current literature of ML models applied to Sentinel-3 OLCI, we expect that future applications are not confined to those. Expanding the FE space may be used to improve our framework in the future, in particular, expanding the domain-specific FE by harvesting the extensive ocean colour literature. For example, the FE space could integrate atmospheric correction, glint removal, adjacency correction, and semi-analytical models. Those potential new FE levels could explore different techniques and the parameters belonging to them. Considering the continuing growth of observational databases and following the use of ML models, optimising FE can be key for the calibration of accurate regional retrieval models in coastal areas. Hence, contributing to making the ocean colour data more reliable for environmental ecosystem assessment, also in optically complex coastal and fjord waters.

Acknowledgments. We appreciate the availability of in situ data from the national Økosystemovervåking i kystvann (Økokyst) from the Norwegian Environment Agency and the Sentinel-3 OLCI data from the Copernicus Data Space.

Funding Statement. ES and SC were funded by the Bjerknes Centre for Climate Research (BCCR) – Centre of Climate Dynamics (SKD) Strategic Project NorHAB-ML and by the institute research fellowship (INSTSTIP) funded by the basic institutional funding through the Norwegian Research Council (#318085). LP was funded by the Research Council of Norway through the Nansen Center basic funds grant no. 342624. FC acknowledges the NFR Climate Futures (309562). This work has also received a storage Grant (NORSTORE, NS9039K)

## Competing Interests. None

Data Availability Statement. Chlrophyll-a and Secchi disk depth data are publicly available on the Vannmiljø web portal (https://vannmiljo.miljodirektoratet.no/). Sentinel-3 OLCI observations are publicly available on Copernicus Data Space Ecosystem (https://dataspace,copernicus.eu/). Codes for reproducing the study will be made available once accepted for publication.

Ethical Standards. The research meets all ethical guidelines, including adherence to the legal requirements of the study country.

Author Contributions. Conceptualisation: E.S., Data Curation: E.S. Formal analysis: E.S., S.C., J.B. Funding Acquisition: J.B., F.C., L.P. Investi ation: E.S., J.B., S.C., F.C., L.P. Methodolo : E.S., J.B. Validation: E.S., J.B., S.C., F.C., L.P. Visualisation: E.S. Writing - Original Draft: E.S. Writing - Review and Editing: E.S., J.B., S.C., F.C., L.P.

Supplementary Material. None.

## References

Aksnes, D., Dupont, N., Staby, A., Fiksen, Ø., Kaartvedt, S., and Aure, J. (2009). Coastal water darkening and implications for mesopelagic regime shifts in Norwegian fjords. Marine Ecology Progress Series, 387:39–49.

Arthur, D. and Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms, ofSODA.

Bergstra, J., Yamins, D., and Cox, D. D. (2013). Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. Atlanta

Bergstra, J. S., Bardenet, R., Bengio, Y., and Kégl, B. (2011). Algorithms for Hyper-Parameter Optimization. Advances in Neural Information Processing Systems 24 (NIPS 2011).

Brando, V. E., Sammartino, M., Colella, S., Bracaglia, M., Di Cicco, A., D’Alimonte, D., Kajiyama, T., Kaitala, S., and Attila, J. (2021). Phytoplankton Bloom Dynamics in the Baltic Sea Using a Consistently Reprocessed Time Series of Multi-Sensor fl d l hl h ll i l Remote Sensin 6

Cairo, C., Barbosa, C., Lobo, F., Novo, E., Carlos, F., Maciel, D., Flores Júnior, R., Silva, E., and Curtarelli, V. (2019). Hybrid Chlorophyll-a Algorithm for Assessing Trophic States of a Tropical Brazilian Reservoir Based on MSI/Sentinel-2 Data. Remote Sensing, 12(1):40.

Cairo, C., Barbosa, C., Lobo, F., Novo, E., Carlos, F., Maciel, D., Flores Júnior, R., Silva, E., and Curtarelli, V. (2020). Hybrid Chlorophyll-a Algorithm for Assessing Trophic States of a Tropical Brazilian Reservoir Based on MSI/Sentinel-2 Data. Remote Sensing, 12(1):40.

Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 785–794. arXiv:1603.02754 [cs].

Cortes, C. and Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3):273–297. Publisher: Springer Science and Business Media LLC.

Drucker, H., Burges, C., Kaufman, L., Smola, A., and Vapnik, V. (1997). Support vector regression machines. In Advances in Neural Information Processing Systems.

EUMETSAT (2021). Sentinel-3 OLCI L2 report for baseline collection OL\_l2m\_003.

Fazliani, S., Frangella, Z., and Udell, M. (2025). Enhancing Physics-Informed Neural Networks Through Feature Engineering. arXiv:2502.07209 [cs].

Frigstad, H., Kaste, Ø., Deininger, A., Kvalsund, K., Christensen, G., Bellerby, R. G. J., Sørensen, K., Norli, M., and King, A. L. (2020). Influence of Riverine Input on Norwegian Coastal Systems. Frontiers in Marine Science, 7:332.

Gada, M., Haria, Z., Mankad, A., Damania, K., and Sankhe, S. (2021). Automated Feature Engineering and Hyperparameter optimization for Machine Learning. In 2021 7th International Conference on Advanced Computing and Communication Systems (ICACCS), pages 981–986, Coimbatore, India. IEEE.

Glukhovets, D., Kopelevich, O., Yushmanova, A., Vazyulya, S., Sheberstov, S., Karalli, P., and Sahling, I. (2020). Evaluation of the CDOM Absorption Coefficient in the Arctic Seas Based on Sentinel-3 OLCI Data. Remote Sensing, 12(19):3210.

Heaton, J. (2016). An Empirical Analysis of Feature Engineering for Predictive Modeling. In SoutheastCon 2016, pages 1–6. arXiv:1701.07852 [cs].

Heaton, J. T. (2017). Automated Feature Engineeringfor Deep Neural Networks with Genetic Programming. PhD thesis, Nova Southeastern University.

Hinton, G. E. (1989). Connectionist learning procedures. Artificial Intelligence, 40(1-3):185–234.

IOCCG (2000). Remote Sensing of Ocean Colour in Coastal, and Other Optically-Complex, Waters. Technical report, IOCCG, Dartmouth.

IOCCG, Greb, S., Dekker, A., and Binding, C. (2018). Earth Observations in Support of Global Water Quality. Technical report, International Ocean Colour Coordinating Group (IOCCG). Medium: 125pp.

John, U., Šupraha, L., Gran-Stadniczeñko, S., Bunse, C., Cembella, A., Eikrem, W., Janouškovec, J., Klemm, K., Kühne, N., Naustvoll, L., Voss, D., Wohlrab, S., and Edvardsen, B. (2022). Spatial and biological oceanographic insights into the massive fish-killing bloom of the haptophyte Chrysochromulina leadbeateri in northern Norway. Harmful Algae, 118:102287.

Joshi, N., Park, J., Zhao, K., Londo, A., and Khanal, S. (2024). Monitoring Harmful Algal Blooms and Water Quality Using Sentinel-3 OLCI Satellite Imagery with Machine Learning. Remote Sensing, 16(13):2444.

Kondrik, D., Pozdnyakov, D., and Pettersson, L. (2017). Tendencies in Coccolithophorid Blooms in Some Marine Environments of the Northern Hemisphere according to the Data of Satellite Observations in 1998–2013. Izvestiya, Atmospheric and Oceanic Physics, 53(9):955–964.

Lapucci, C., Antonini, A., Böhm, E., Organelli, E., Massi, L., Ortolani, A., Brandini, C., and Maselli, F. (2023). Use of Sentinel-3 OLCI Images and Machine Learning to Assess the Ecological Quality of Italian Coastal Waters. Sensors, 23(22):9258.

Lee, Z., Carder, K. L., and Arnone, R. A. (2002). Deriving inherent optical properties from water color: a multiband quasianalytical algorithm for optically deep waters. Applied Optics, 41(27):5755.

McGovern, A., Ebert-Uphoff, I., Gagne, D. J., and Bostrom, A. (2022). Why we need to focus on developing ethical, responsible, and trustworthy artificial intelligence approaches for environmental science. Environmental Data Science, 1:e6.

Morel, A., Gentili, B., Claustre, H., Babin, M., Bricaud, A., Ras, J., and Tièche, F. (2007a). Optical properties of the “clearest” natural waters. Limnology and Oceanography, 52(1):217–229.

Morel, A., Huot, Y., Gentili, B., Werdell, P. J., Hooker, S. B., and Franz, B. A. (2007b). Examining the consistency of products derived from various ocean color sensors in open ocean (Case 1) waters in the perspective of a multi-sensor approach. Remote Sensing ofEnvironment, 111(1):69–88.

Morel, A. and Prieur, L. (1977). Analysis of variations in ocean color. Limnology and Oceanography, 22(4):709–722.

Mumuni, A. and Mumuni, F. (2025). Automated data processing and feature engineering for deep learning and big data applications: A survey. Journal ofInformation and Intelligence, 3(2):113–153. Publisher: Elsevier BV.

Nima, C., Frette, Ø., Hamre, B., Erga, S. R., Chen, Y.-C., Zhao, L., Sørensen, K., Norli, M., Stamnes, K., and Stamnes, J. J. (2016). Absorption properties of high-latitude Norwegian coastal water: The impact of CDOM and particulate matter. Estuarine, Coastal and ShelfScience, 178:158–167.

Pahlevan, N., Smith, B., Schalles, J., Binding, C., Cao, Z., Ma, R., Alikas, K., Kangro, K., Gurlin, D., Hà, N., Matsushita, B., Moses, W., Greb. S., Lehmann, M. K., Ondrusek, M., Oppelt, N., and Stumpf, R. (2020). Seamless retrievals of chlorophyll-a from Sentinel-2 (MSI) and Sentinel-3 (OLCI) in inland and coastal waters: A machine-learning approach. Remote Sensing of Environment, 240:111604.

Pettersson, L. H. and Pozdnyakov, D. (2013). Monitoring of Harmful Algal Blooms. Springer Berlin Heidelberg, Berlin, Heidelberg.

Rather, I. H., Kumar, S., and Gandomi, A. H. (2024). Breaking the data barrier: a review of deep learning techniques for democratizing AI with small datasets. Artificial Intelligence Review, 57(9):226.

Safonova, A., Ghazaryan, G., Stiller, S., Main-Knorn, M., Nendel, C., and Ryo, M. (2023). Ten deep learning techniques to address small data problems with remote sensing. International Journal ofApplied Earth Observation and Geoinformation, 125:103569

Schiller, H. and Doerffer, R. (1999). Neural network for emulation of an inverse model operational derivation of Case II water properties from MERIS data. International Journal ofRemote Sensing, 20(9):1735–1746.

Shen, M., Duan, H., Cao, Z., Xue, K., Qi, T., Ma, J., Liu, D., Song, K., Huang, C., and Song, X. (2020). Sentinel-3 OLCI observations of water clarity in large lakes in eastern China: Implications for SDG 6.3.2 evaluation. Remote Sensing of Environment, 247:111950.

Silva, E., Counillon, F., Brajard, J., Korosov, A., Pettersson, L. H., Samuelsen, A., and Keenlyside, N. (2021). Twenty-One Years of Phytoplankton Bloom Phenology in the Barents, Norwegian, and North Seas. Frontiers in Marine Science, 8:746327.

Soja-Wo´zniak, M., Craig, S., Kratzer, S., Wojtasiewicz, B., Darecki, M., and Jones, C. (2017). A Novel Statistical Approach for Ocean Colour Estimation of Inherent Optical Properties and Cyanobacteria Abundance in Optically Complex Waters. Remote Sensing, 9(4):343.

Steinmetz, F., Deschamps, P.-Y., and Ramon, D. (2011). Atmospheric correction in presence of sun glint: application to MERIS. Optics Express, 19(10):9783.

Steinmetz, F. and Ramon, D. (2018). Sentinel-2 MSI and Sentinel-3 OLCI consistent ocean colour products using POLYMER. In Frouin, R. J. and Murakami, H., editors, Remote Sensing of the Open and Coastal Ocean and Inland Waters, page 13, Honolulu, United States. SPIE.

Tessin, E., Hamre, B., and Kristoffersen, A. S. (2024). Testing the Limits of Atmospheric Correction over Turbid Norwegian Fjords. Remote Sensing, 16(21):4082.

vanndirektivet 2018, D. (2018). Klassifisering av miljøtilstand i vann. Technical Report Veileder 02:2018.

Verdonck, T., Baesens, B., Óskarsdóttir, M., and Vanden Broucke, S. (2024). Special issue on feature engineering editorial. Machine Learning, 113(7):3917–3928. Publisher: Springer Science and Business Media LLC.

Zoffoli, M. L., Brando, V., Volpe, G., González Vilas, L., Davies, B. F. R., Frouin, R., Pitarch, J., Oiry, S., Tan, J., Colella, S., and Marchese, C. (2025). CIAO: A machine-learning algorithm for mapping Arctic Ocean Chlorophyll-a from space. Science of Remote Sensing, 11:100212.