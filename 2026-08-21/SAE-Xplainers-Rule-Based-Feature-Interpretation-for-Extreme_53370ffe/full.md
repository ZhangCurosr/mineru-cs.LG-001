# SAE-Xplainers: Rule-Based Feature Interpretation for Extreme Earth Events

Hugo Porta1\*, Emanuele Dalsasso2, 3, Chang Xu1, Theo Gnassounou¹, Devis Tuia1

1EPFL, Switzerland

2Université Grenoble Alpes, France

3Inria, France

{hugo.porta, chang.xu, theo.gnassounou, devis.tuia} @epfl.ch, emanuele.dalsasso@inria.fr

## Abstract

The emergence of large-scale Weather and Climate (W&C) datasets offers new opportunities for modeling extreme Earth events (ExEE) and their impacts using deep learning. However, their adoption in operational settings remains limited by the lack of models’ interpretability. While for conventional text and image modalities, tools such as Sparse Autoencoders (SAEs) have proven effective for extracting humanunderstandable concepts, their use for the analysis of ExEE remains challenging due to the nature of W&C data. To address this, we introduce (i) a geographic location–based modulation of the inputs of SAE to capture the local semantic meaning of environmental patterns, and (ii) an ensemble of rule-based SAE-Xplainers to interpret the resulting high-dimensional features derived from complex, multi-modal environmental predictors. We evaluate our method on three ExEE types: the prediction of fires, and the detection of tropical cyclones and atmospheric rivers. We show that SAE input modulation improves both reconstruction performance and feature utilization, and that our SAE-Xplainers enable faithful interpretation of complex climatic patterns by unfolding them into humanunderstandable rules that are consistent with the scientific literature, while also supporting the identification of feature absorption.

Code — https://github.com/eceo-epfl/SAE-Xplainers

## Introduction

Human-induced climate change is causing a widespread increase in the frequency and intensity of ExEE (e.g., wildfires, floods, droughts, storms), with severe consequences for human populations and ecosystems (Seneviratne et al. 2023). Improving our ability to forecast, detect, and assess the impact of these events is therefore critical for effective preparedness and risk management. The rapid growth of W&C data and Earth observation imagery (Tuia et al. 2025) creates new opportunities to address these tasks using deep learning-based models capable of integrating heterogeneous data structures and predictor types (Camps-Valls et al. 2025).

Nonetheless, the "black box" nature of deep learning models remains a barrier to their adoption in the Earth sciences, especially for extremes, where transparency is essential (Camps-Valls et al. 2025), prompting the need for eXplainable AI (XAI) methods that interpret model behavior through the location (where) (Selvaraju et al. 2017; Lundberg and Lee 2017) and semantic meaning (what) (Kim et al 2018; Koh et al. 2020) of influential features. In ExEE, most methods aim to answer where the model focuses, using feature importance (Dikshit and Pradhan 2021) or attribution maps (Wei et al. 2025).

Alternatively, concept-based methods uncover humaninterpretable concepts (Kim et al. 2018; Koh et al. 2020) within model representations. Among these, Sparse Autoencoders (SAEs) (Huben et al. 2024) project polysemantic activations into a sparse latent space to reveal monosemantic "true features" (Bricken et al. 2023), with strong effectiveness shown for LLMs (Gao et al. 2025) and vision models (Lim et al. 2025). However, the direct use of SAEs on ExEE data presents limitations in their training and the interpretation of the extracted concepts (see Figure 1 and 3). First, ExEE drivers are often inherently local: similar weather conditions can lead to different outcomes due to regional climate dynamics and environmental context. This represents a challenge for SAEs, which are not designed to learn location-dependent activations (see Figure 3). Second, existing approaches for the interpretation of SAEs' features, such as LLM autointerpretations (Huben et al. 2024) and concept activation maps (Thasarathan et al. 2025), rely on humans or language models. For ExEE, the large number of input modalities and the complex spatio-temporal patterns limit the semantic interpretation of SAEs’features through activation maps in the input space (see Figure 1).

In this paper, we propose an approach for the interpretation of ExEE models that first extracts monosemantic features via a location-aware SAE, and then interprets them using an ensemble of human-understandable rule-based models: SAE-Xplainers. To account for the localized nature of ExEE drivers, we introduce a new geographic location encoder acting as a FiLM adapter (Perez et al. 2018), modulating activations prior to a k-sparse autoencoder (GeoTopK) (Makhzani and Frey 2014) to enforce a location-aware projection. We validate our approach on two encoders: ViT (Dosovitskiy et al. 2021) (application-agnostic) and ClimaX (Nguyen et al. 2023) (application-specific), across two datasets covering three ExEE types: fres, tropical cyclones, atmospheric rivers, for both forecasting and detection. We further show that the monosemanticity of the learned features (Bricken et al. 2023) enables SAE-Xplainers to faithfully approximate

![](images/08bea1f8afec770cd8107d8feaa00a4cb5fa886fedb35164f44cbba69dc01697.jpg)  
Figure 1: (Top): In computer vision, human users or Vision-Language Models (VLMs) can easily infer the semantic meaning of the SAE features via their activations. (Bottom): In ExEE tasks, the semantic meaning of SAE features cannot be inferred from their activation maps. We propose to learn rule-based models to interpret the features from the input variables (T2M, Rel-Hum, ..., see Table 3 in supplementary material for definition), replacing human or VLM analysis.

SAE activations (see Figure 1), and that comparing semantic similarity across rules reveals feature absorption (Chanin et al. 2025): a sparsity-driven phenomenon producing overly specific, non-atomic features from hierarchical concepts.

In summary, our contributions are as follows: (i) we introduce a geographical location-aware SAE, GeoTopK, to our knowledge, the first domain-conditioned SAE training strategy, which outperforms its location-agnostic counterpart in terms of reconstruction, dead features rate, and on two new application-specific metrics that account for label imbalance and domain shift. (ii) We propose an ensemble of rule-based SAE-Xplainers, which allows faithful interpretation of the SAE features and identification of feature absorption, building a complete SAE interpretability framework for ExEE.

## Related Work

Explainable AI in Extreme Earth Events. In W&C, interpretability has mainly been pursued through feature attribution methods addressing where the model looks, decomposed into which predictors, locations, or time-steps most influence the prediction. Shapley additive explanations (SHAP) (Lundberg and Lee 2017) and local interpretable modelagnostic explanations (LIME) (Ribeiro, Singh, and Guestrin 2016) have been used to identify the dominant predictors for drought (Dikshit and Pradhan 2021), tropical cyclones (Cui et al. 2025), wildfires (Cilli et al. 2022), and extreme rainfalls (Hrast Essenfelder, Toreti, and Seguini 2025) while locationsensitive methods such as attention (Dikshit et al. 2022) and Grad-CAM (Srinivasan, Wang, and Bulleid 2020) have been used to visualize where informative signals arise across space and time. However, these approaches risk increasing user confirmation bias (Adebayo et al. 2018), lack global interpretations (Van Der Linden, Haned, and Kanoulas 2019), raise faithfulness concerns (Rudin 2019), and face scalability constraints (Covert, Kim, and Lee 2023) for SHAP. Crucially, none address what the semantic meaning of influential features is. Disentangling neurons into interpretable concepts remains an open challenge for transparent W&C models.

Interpretable rule-based models. By-design interpretable models, particularly rule-based approaches, have been extensively studied for ExEE applications (Rodrigues et al. 2022; Lovo et al. 2025): decision trees offer clear decision paths (Breiman 2001), and methods like Skope-Rules extract compact i f-t hen rules from tree ensembles, though both struggle with complex tasks; Bayesian approaches (Wang et al. 2017; Letham et al. 2015) offer probabilistic rule learning but remain computationally expensive and hard to scale (Yang, Rudin, and Seltzer 2017). To address this, several works combine deep learning with rule-based structures, including differentiable neural decision trees (Kontschieder et al. 2015), hypernetworks generating rule parameters from learned representations (Yang, Ren, and Li 2024), and Gradient Grafting directly optimizing rules via gradient descent (Wang et al. 2021), but these methods rely on complex architectures that limit scalability and increase computational cost. Post-hoc methods, which extract rules from a trained model without modifying it, offer an attractive alternative: Anchors, for instance, generate high-precision if-then rules explaining individual predictions (Ribeiro, Singh, and Guestrin 2018), but remain only local, describing single-sample predictions. Global feature extraction followed by rule-based interpreters thus appears promising to bridge global understanding with sample-level interpretations.

Sparse Autoencoders for XAI. SAEs are a natural candidate for such global feature extraction as they aim to disentangle the model's polysemantic neurons by projecting them to a sparse high-dimensional space where the dimensions represent monosemantic features (Bricken et al. 2023). They have been applied beyond LLMs (Huben et al. 2024; Templeton et al. 2024; Gao et al. 2025) to computer vision, including steering and adaptation analysis of visionlanguage models (Lim et al. 2025), cross-model analysis of vision encoders and pretraining objectives (Thasarathan et al. 2025), and concept unlearning in diffusion models (Bohacek et al. 2026). Interpreting SAE dictionary vectors as humanunderstandable concepts typically relies on auxiliary semantic inference via LLM autointerpretations (Huben et al. 2024; Paulo et al. 2025), vision-text similarity measures (Lim et al. 2025; Rao et al. 2024), or human inference from activation maps (Thasarathan et al. 2025; Fel et al. 2025).

In ExEE, the absence of Weather-Language models prevents the automatic interpretation of SAEs’features, while the complexity of the input modalities and spatial patterns makes human analysis of SAE activations challenging, as shown in Figure 1. We address this with a machine-driven framework for feature interpretations via an ensemble of rulebased SAE-Xplainers. Representing features as rule sets allows us to identify feature absorption (Karvonen et al. 2025; Chanin et al. 2025), a sparsity-driven phenomenon where a hierarchical concept is partially captured by a general feature and partly by more specific ones, causing inconsistent activations and fragmented representations (Leask et al. 2025).

![](images/101e8dda0045c2c540ff44aa9a194d4e3f111ce0d4372fccf632e7dc7397b8c0.jpg)  
Figure 2: Overview of our framework. (Top): GeoTopK is trained on the modulated patch activations jointly with the location encoder. (Bottom): For a given SAE feature, a SAE-Xplainer is trained via Skope-Rules to predict the binarized activations of selected input patches, where the negatives are sampled adversarially in the neighborhood of positives.

## Geographic Location-Aware SAE

SAE Training. The top panel of Figure 2 illustrates our approach. We consider a vision transformer-based encoder $f : \mathcal { X } \ :  \ : \mathcal { Y } .$ with $f ^ { ( l ) } ~ : ~ \mathcal { X } ~  ~ \mathbb { R } ^ { T \times D }$ the mapping from the input to the sequence T of tokens activations at layer (l). For an input sample $x \in \mathcal { X }$ , we define the matrix $\dot { H ^ { ( l ) } } = f ^ { ( l ) } ( x ) \in \mathbb { R } ^ { \dot { T } \times D }$ as the latent activations of the model for layer l. For a specific patch (token) index $t \in \{ 1 , . . . , T \}$ , we denote $x _ { t }$ the corresponding input patch and $h _ { t } ^ { ( l ) } = H _ { t , : } ^ { ( l ) } \in \mathbb { R } ^ { D }$ its latent activations at layer l. SAEs aim to learn à sparse and overcomplete decomposition of the token activation $h _ { t } ^ { ( l ) }$ into a high-dimensional space of size N. Experimentally, we focus on the last residual stream output of the transformer $\left( { { h _ { t } } } \right)$ as input to the SAE. $h _ { t }$ is passed through a k—sparse autoencoder to directly control the number of active features $\mathcal { Z }$ via the TopK function, which replaces the ReLU activation and the $L _ { 1 }$ sparsity penalty. The encoder and decoder operations are defined by:

$$
\begin{array} { c } { { z = \mathrm { T o p } \mathrm { K } ( W _ { e n c } h _ { t } + b _ { e n c } ) \ : , } } \\ { { { \hat { h } } _ { t } = W _ { d e c } z \ : , } } \end{array}\tag{1}
$$

![](images/b41a4e213359da075a8717dc280a099b2e632a92e5d0036c235a9bc2dfc60719.jpg)  
Figure 3: Similar environmental conditions (ENV) can lead to different ExEE probabilities due to local effects, requiring models to learn location-dependent features.

with $W _ { e n c } \in \mathbb { R } ^ { N \times D } , b _ { e n c } \in \mathbb { R } ^ { N }$ , and $W _ { d e c } \in \mathbb { R } ^ { D \times N }$ Following (Gao et al. 2025), we also apply ReLU before TopK to ensure non-negative activations in our code. The decoder matrix $W _ { d e c }$ consists of N column vectors, where each vector $d _ { i }$ represents a monosemantic feature learned by the model. The vector $z \in \mathbb { R } ^ { N }$ represents the feature activations and $\| z \| _ { 0 } = k$

SAEs are trained to minimize the mean squared error (MSE) as a reconstruction objective. Architectures like k—sparse autoencoders also require the use of an auxiliary loss $\mathcal { L } _ { \mathrm { a u x } }$ to revive inactive features (Gao et al. 2025), its computation is detailed in the supplementary material:

$$
\mathcal { L } _ { \mathrm { S A E } } = \Vert h _ { t } - \hat { h } _ { t } \Vert _ { 2 } ^ { 2 } + \gamma \mathcal { L } _ { \mathrm { a u x } } .\tag{2}
$$

Geographic Location Modulation. To capture the local nature of ExEE, we enforce geographic location information into the activations of the models, enhancing the SAE ability to disentangle location-specific characteristics of the input predictors, as illustrated in Figure 3. Given the geographic location set G, we learn a geographic location encoder Ψ : $\mathcal { G } \subset \mathbb { R } ^ { 2 } \to \mathbb { R } ^ { 2 D }$ , which maps latitude $\phi \in [ - 9 0 , 9 0 ]$ and longitude $\lambda \in [ - 1 8 0$ , 180] of an input patch $x _ { t }$ to the modulation parameters $( \alpha , \beta ) \doteq \mathbb { R } ^ { 2 D }$

$$
\begin{array} { r l } & { \Psi ( \phi , \lambda ) = \mathbf { M } \mathbf { L } \mathbf { P } ( [ \sin ( \phi ) , \cos ( \phi ) , \sin ( \lambda ) , \cos ( \lambda ) ] ) \mathrm { ~ , ~ } } \\ & { \Psi ( \phi , \lambda ) = ( \alpha , \beta ) \mathrm { ~ . ~ } } \end{array}\tag{3}
$$

The modulation parameters are used to perform featurewise affine modulation of the model activation $h _ { t } .$ acting as FiLM adapters. This approach is used in normalization layers, where shift-only transformations cannot adjust the importance of neurons in the model activations, and scalar modulations can lack expressiveness. Moreover, feature-wise modulation provides expressiveness without the computational overhead and neuron mixing of full linear layers. We further apply the tanh nonlinearity to constrain the magnitude of both scaling and shifting, ensuring stable conditioning and preventing the SAE from trivially encoding only geographic location, resulting in:

$$
h _ { t , \mathrm { m o d } } = ( 1 + \operatorname { t a n h } ( \alpha ) ) \odot h _ { t } + \operatorname { t a n h } ( \beta ) ~ .\tag{4}
$$

The vector $h _ { t , \mathrm { m o d } }$ is then passed to the SAE in Equation 1 in place of $h _ { t } .$ and the affine modulation is inverted before the reconstruction, as shown in Figure 2. Ablation results on the geographic location modulation parameterization and comparison with other methods (Eddin et al. 2023) are provided in the supplementary material Table 6.

Evaluation. ExEE tasks are highly imbalanced and span diverse climatic regimes across the globe, requiring models to generalize across domains. To account for these challenges, we extend the standard metrics used in SAE evaluation: $\mathbf { \bar { R } } ^ { 2 }$ MSE, and ratio of dead features.

First, we analyze reconstruction fidelity by conditioning the computation of $\mathbb { R } ^ { 2 }$ on extreme events, which we denote as the $\mathbf { R } _ { \mathrm { e v e n t } } ^ { \bar { 2 } } { = } \mathbf { R } ^ { 2 } ( h _ { t } , \hat { h _ { t } } ) \big | _ { { \mathcal { T } } _ { \mathrm { F v e n t } } }$ , where $\mathcal { T } _ { \mathrm { e v e n t } }$ are the indices of input patch samples in X’ belonging to extreme events. This metric is macro-averaged across event classes and provides an assessment of reconstruction fidelity on the target classes despite the severe class imbalance.

Second, we condition the computation of $\mathbb { R } ^ { 2 }$ across predefined latitude bands and report the ${ \bf R } _ { \mathrm { w o r s t } } ^ { 2 } =$ $\mathrm { m i n } _ { b \in \mathcal { B } } \mathrm { R } ^ { 2 } ( h _ { t } , \hat { h _ { t } } \ | \ \phi \in \mathfrak { b } )$ , with latitude intervals of $2 0 ^ { \circ }$ $\mathcal { B } \ : = \ : \left\{ [ - 9 0 , - 7 0 ) , [ - 7 0 , - 5 0 ) , \ldots , [ 7 0 , 9 0 ] \right\}$ . This metric captures the worst-case reconstruction performance of the SAE across geographically diverse domains, providing a lower bound on reconstruction fidelity at the global scale. This echoes the worst-group accuracy metric, often used in applications with domain shift to assess model robustness.

## Rule-based SAE-Xplainers

Training the SAE-Xplainers. First, we rewrite Equation 1 using the feature vectors $d _ { i } \in \mathbb { R } ^ { D } , \forall i \in [ 1 , N ] \colon$

$$
\hat { h } _ { t } = W _ { d e c } z = \sum _ { i = 1 } ^ { N } z _ { i } d _ { i } \ ,\tag{5}
$$

with $z _ { i }$ the activation of the feature vector $d _ { i }$ An emergent property of SAEs' high-dimensional and sparse latent space is the monosemanticity of its features (Elhage et al. 2022). As a consequence, we hypothesize that we can learn for each feature vector $d _ { i }$ a surrogate interpretable model $g _ { i } : \mathcal { X }  \{ 0 , 1 \}$ that faithfully approximates the mapping from the multi-modal input patch to its binarized activations $\tilde { z } _ { i } ( x _ { t } ) = \mathbb { 1 } [ z _ { i } ( x _ { t } ) > 0 ]$ , as illustrated in the bottom panel of Figure 2. The set of $\dot { N }$ surrogate models constitutes our ensemble of SAE-Xplainers. Due to the high sparsity of SAE features, their activations follow a zero-inflated distribution, with most samples corresponding to zero activations. Therefore, interpretability depends more on accurate distinctions between activating and non-activating samples, rather than on predicting precise activation magnitudes (Karvonen et al. 2025). A model $g _ { i }$ is defined as a finite set of $m _ { i }$ rules $\mathcal { R } _ { i } = \{ r _ { i , 1 } , . . . , r _ { i , m _ { i } } \}$ , where each rule $r _ { i , j } : \mathcal { X }  \{ 0 , 1 \}$ is a conjunction of threshold conditions on the input variables with a normalized weight $w _ { j } \colon$

$$
g _ { i } ( x _ { t } ) = \mathbb { 1 } [ \sum _ { j = 1 } ^ { m _ { i } } w _ { j } r _ { i , j } ( x _ { t } ) > 0 . 5 ] .\tag{6}
$$

The rule-set surrogate models are trained using the Skope-Rules algorithm under multiple parameterizations to identify the proper level of complexity (measured as the number of rule conditions per rule set) for predicting feature activations. To further improve interpretation quality, we extend the toprandom balanced sampling strategy introduced in LLM autointerpretability (Huben et al. 2024; Paulo et al. 2025) with a novel adversarial geographic negative sampling scheme: for each surrogate model $g _ { i } ,$ negative samples are drawn within a geographic radius around the positive activation set $\mathcal { T } _ { i } = \{ x _ { t } ^ {  } | \stackrel {  } { z _ { i } } ( \stackrel { \cdot } { x _ { t } } ) > 0 , x \in \mathcal { X } \}$ , ensuring better rule-based interpretations and preventing location-based discrimination (see Figure 2). The adversarial negative set is defined as:

$$
{ \mathcal { T } } _ { \mathtt { n e g } , i } = \left\{ x _ { t } \mid z _ { i } ( x _ { t } ) = 0 \ \wedge \ d ^ { * } ( x _ { t } ) \leq \sigma , \ x \in \mathcal { X } \right\} ,\tag{7}
$$

with $\begin{array} { r } { d ^ { * } ( x _ { t } ) \ = \ \operatorname* { m i n } _ { x _ { \star } ^ { + } \in \mathbb { Z } _ { i } } d _ { \mathbb { S } ^ { 2 } } \big ( c ( x _ { t } ) , c ( x _ { t } ^ { + } ) \big ) , \ d _ { \mathbb { S } ^ { 2 } } ( . , . ) } \end{array}$ the geodesic distance on the Earth as a sphere, and σ the sampling radius. A comparison of the interpretations across sampling strategies is provided in the supplementary material.

Our approach is similar to Anchors (Ribeiro, Singh, and Guestrin 2018), as it extracts rule-based interpretations of the model behavior. However, while Anchors generate local rules explaining individual predictions in the input space via perturbations, our SAE-Xplainers learn surrogate rules on sparse latent activations to provide scalable, global interpretations of learned features in the representation space.

Integrating Spatial Context. The surrogate model formulation (Equation 6) does not account for the influence of spatial context on feature activations. However, the importance of spatial interactions between patches on the learned features can vary depending on the event types, as shown in Figure 4a. To capture this effect, we extend Equation $^ 6$ so that the surrogate models account for statistics computed over the set of most attended patches for a given input patch $x _ { t } .$ , where attention scores are aggregated across all layers of the transformer model f up to layer l: $A ^ { l } ( x _ { t } ) \in \mathbb { R } ^ { T }$ . We denote this new set of most attended patches as $\mathcal { N } _ { n } ( x _ { t } ) = \mathrm { T o p K } _ { n } ( A ^ { l } ( x _ { t } ) \mid t \in$ $\{ 1 , . . . , T \} ,$ ). The spatially aware surrogate model $g _ { i }$ becomes:

$$
g _ { i } ( x _ { t } ) = \mathbb { 1 } [ \sum _ { j = 1 } ^ { m } w _ { j } r _ { i , j } ( x _ { t } , \mu _ { t } ^ { ( n ) } , m _ { t } ^ { ( n ) } , \ell _ { t } ^ { ( n ) } ) > 0 . 5 ] ,\tag{8}
$$

with $\mu _ { t } ^ { ( n ) } = \operatorname * { m e a n } _ { s \in \mathcal { N } _ { n } ( x _ { t } ) } x _ { s } , \ m _ { t } ^ { ( n ) } = \operatorname * { m a x } _ { s \in \mathcal { N } _ { n } ( x _ { t } ) } x _ { s } ,$

$$
\ell _ { t } ^ { ( n ) } = \operatorname* { m i n } _ { s \in \mathcal { N } _ { n } ( x _ { t } ) } x _ { s } .
$$

Evaluation. The rule-based interpretations from our ensemble of SAE-Xplainers are first evaluated on their ability to faithfully predict the activations of the SAE features, as shown in Figure 4a. Nonetheless, this metric highly depends on the sampling strategy used for building the training and test sets and is not enough to guarantee the quality of feature interpretations. To this end, we investigate the potential of SAE-Xplainers to identify feature absorption at scale across all SAE features. Feature absorption (Chanin et al. 2025) is a sparsity-induced phenomenon in SAEs, where hierarchical concepts are not represented as separate atomic features, but instead a specific feature may absorb part of a more general one, leading to fragmented, overly specific, and less semantically coherent representations. If our rule-based interpreters are able to capture feature absorption, then the learned rules are consistent with the SAE feature activations.

We use our SAE-Xplainers to identify groups of semantically related features. First, we filter the set of SAE features into groups of connected components, where edges between two features are defined by a Jaccard distance below a given threshold, computed over their rule sets using the Hungarian algorithm for optimal assignment, retaining only the M closest pairs. The identified set of $P$ partitions $\{ \dot { K } _ { j } \} _ { j = 1 } ^ { P } ,$ with $K _ { j } \subseteq \{ 1 , \dots , N \}$ , define candidate feature groups sharing highly similar rule interpretations. Within each group $\kappa _ { j }$ , we measure feature co-occurrence based on their activation sets $\mathcal { T } _ { i } = \{ x _ { t } ~ | ~ z _ { i } ( x _ { t } ) > 0 , ~ x \in \mathcal { X } \}$ . For a feature $d _ { i } \in { \cal K } _ { j }$ , the total group co-occurrence ratio is

$$
\rho _ { i } = \frac { \sum _ { d _ { l } \in { \mathcal { K } } _ { j } \backslash \{ d _ { i } \} } | \mathbb { Z } _ { i } \cap \mathbb { Z } _ { l } | } { | \mathbb { Z } _ { i } | } .\tag{9}
$$

A feature $d _ { i }$ is considered absorbed if $\rho _ { i } \geq \tau$ , where $\tau$ is a fixed threshold. We then define the co-occurrence score as the ratio of groups containing at least one absorbed feature.

Finally, we report the SAEBench absorption score (Karvonen et al. 2025) within co-occurring subgroups, using as main feature $d _ { i ^ { \star } } = \operatorname* { m a x } _ { { d _ { i } \in { \mathcal { K } } _ { i } } } \rho _ { i }$ , and defining the prediction task with its rule-based surrogate model $g _ { i ^ { \star } }$ . This evaluates whether the rules learned by the SAE-Xplainers can identify co-occurring features that encode absorbed concepts.

## SAE Performance Analysis

Experimental Setup. We test our approach on two global datasets and tasks. First is an 8-day horizon dense forecasting of fres using the SeasFire dataset (Karasante et al. 2025), comprising a set of 14 variables (including 10 W&C variables, a biosphere variable, a socioeconomic variable, and the geographic coordinates). Second is the detection of tropical cyclones and atmospheric rivers from ClimateNet (Prabhat et al. 2020) on a set of 6 variables (including 4 W&C variables, and the geographic coordinates). For both tasks, we fine-tuned a W&C foundational model: ClimaX (Nguyen et al. 2023), and a generic vision encoder: ViT (Dosovitskiy et al. 2021). Both use a patch size of $4 \times 4$ As a result, our input patch token $x _ { t }$ and its corresponding model activations $h _ { t }$ represent 4 × 4 pixels at SeasFire and ClimateNet native resolution of $\sim 0 . 2 5 ^ { \circ }$ . For each model and dataset, we train the TopK SAE baseline (Makhzani and Frey 2014) and our geographic location-aware GeoTopK. All results use $k = 4$ and SAE dimension $N = 4 \cdot D = 4 0 9 6$ , favoring high sparsity to strengthen interpretability for human experts. Training and dataset details are provided in supplementary material.

GeoTopK Performance. Table 1 shows the contribution of our geographic location-aware SAE: GeoTopK on SeasFire and ClimateNet test sets. We observe that across all reconstruction metrics and almost all setups, our GeoTopK outperforms the TopK baseline, showing the importance of geographic location modulation. Moreover, GeoTopK highly reduces the number of dead features, which is a key aspect for

<table><tr><td rowspan="2">Metric</td><td colspan="2">ClimaX</td><td colspan="2">ViT</td></tr><tr><td>TopK</td><td>GeoTopK</td><td>TopK</td><td>GeoTopK</td></tr><tr><td> $\mathbf { R } ^ { 2 } \uparrow$ </td><td>0.927</td><td>0.937</td><td>0.825</td><td>0.840</td></tr><tr><td> $\mathrm { R } _ { \mathrm { w o r s t } } ^ { 2 }$  ↑</td><td>0.881</td><td>0.901</td><td>0.767</td><td>0.786</td></tr><tr><td> $\mathrm { R } _ { \mathrm { e v e n t } } ^ { 2 }$  ←</td><td>0.801</td><td>0.807</td><td>0.651</td><td>0.633</td></tr><tr><td>MSE↓</td><td>1.84e-3</td><td>1.59e-3</td><td>1.77e-1</td><td>1.63e-1</td></tr><tr><td>Dead Feat. ↓</td><td>0.159</td><td>0.002</td><td>0.021</td><td>0.000</td></tr></table>

(a) Results on SeasFire.
<table><tr><td rowspan="2">Metric</td><td colspan="2">ClimaX</td><td colspan="2">ViT</td></tr><tr><td>TopK</td><td>GeoTopK</td><td>TopK</td><td>GeoTopK</td></tr><tr><td> $\mathbf { R } ^ { 2 } \uparrow$ </td><td>0.943</td><td>0.953</td><td>0.987</td><td>0.993</td></tr><tr><td> $\mathrm { R } _ { \mathrm { w o r s t } } ^ { 2 }$  ↑</td><td>0.910</td><td>0.924</td><td>0.983</td><td>0.988</td></tr><tr><td> $\mathrm { R } _ { \mathrm { e v e n t } } ^ { 2 }$  ←</td><td>0.880</td><td>0.893</td><td>0.974</td><td>0.980</td></tr><tr><td>MSE↓</td><td>1.01e-3</td><td>0.84e-3</td><td>12.85e-3</td><td>7.12e-3</td></tr><tr><td>Dead Feat. ↓</td><td>0.513</td><td>0.391</td><td>0.811</td><td>0.442</td></tr></table>

(b) Results on ClimateNet.  
Table 1: Quantitative Comparison of TopK and GeoTopK on ClimaX and ViT last residual stream, fine-tuned on SeasFire and ClimateNet, respectively. Bold means best score, underline means similar score (difference is ≤ 0.005).

SAEs that favors monosemanticity. This is more pronounced in the ClimateNet dataset, where standard TopK architectures fail to maintain even a 50% feature activation rate despite the implementation of auxiliary losses and feature resampling techniques. By integrating geographic location modulation, GeoTopK effectively prevents the emergence of polysemantic features typically caused by neural superposition. The impact of scaling the latent dimension size of the SAE is shown in supplementary material Figure 7.

## Rule-based Feature Interpretations

SAE-Xplainers Faithfulness. In this section, we analyse two sets of SAE-Xplainers, learned over all the active features with enough positive samples of the SeasFire and ClimateNet datasets. For both datasets, we resample train/test splits to retain as many active features (in particular for ClimateNet) In both cases, we analyze the ClimaX model, as it outperforms ViT on both datasets. To assign a level of complexity to each feature and identify the importance of spatial context, we train our SAE-Xplainers for three parametrizations of Skope-Rules: low, mid, and high complexity, with or without spatial context. Then, for a specific feature we select the most suitable parametrization and spatial context that provides a gain in training accuracy of at least $\tau _ { a c c } = 0 . 0 1$ compared to the current setup, following this order of complexity [low, low and spatial, mid, ..., high and spatial]. This is reflected, in Figure 4, by the six configurations per method and dataset showing an increase in the mean number of rule conditions across the SAE-Xplainers (details on SAE-Xplainers training, downstream model performance, and baselines are in the supplementary material).

In Figure 4a, we compare the faithfulness via the prediction accuracy proxy of our SAE-Xplainers trained on the monosemantic SAE features to that of similar rule-based models trained on the polysemantic ClimaX model neurons and on class-specific prototypes, following the interpretable per-design prototype network: ProtoSeg (Sacha et al. 2023; Porta et al. 2025). We observe for the final aggregated sets of our SAE-Xplainers an evaluation accuracy of \~ 93% (+22% with the neurons baseline and +9% with the prototypes) on SeasFire, and～ 85% (+9% with the neurons baseline and +5% with the prototypes) on ClimateNet. This shows first a strong faithfulness of the interpretations and validates our motivation to train accurate SAE-Xplainers on the GeoTopK features due to their monosemantic property as opposed to the model neurons. The faithfulness of our SAE-Xplainers, considering the adversarial sampling strategy used, also shows that GeoTopK does not only memorizes the geographic coordinates modulation, but uses it to disentangle the model representations. Additionally, we see that our SAE-Xplainers perform much better on SeasFire in comparison with ClimateNet, which can be explained by the higher dead features rate on ClimateNet (see Table 1), which causes superposition across the active SAE features, but also by the higher importance of spatial context in tropical cyclone and atmospheric river detection, contrarily to fire forecasting. This is illustrated in Figure 4a, with the large gain in model accuracy for our SAE-Xplainers on ClimateNet when integrating spatial context at low model complexity: +13%, with only +3% for SeasFire. In supplementary material Figure 8, the feature assignment in SAE-Xplainers also demonstrates the role of spatial context in ClimateNet.

![](images/45c73a3e621f2e7cbca7822fe9bb2108cd46951d960e20f06dafa3b6e826f0ff.jpg)  
Figure 4: SAE-Xplainers quantitative analysis. (a): Faithfulness of our SAE-Xplainers interpretations across Skope-Rules parametrization of ClimaX Neurons, Prototypes, and GeoTopK feature activations. (b): Evolution of the scores related to absorption across SAE-Xplainers parametrizations.

Feature Absorption. We evaluate our SAE-Xplainers' ability to detect absorbed features for ClimaX fine-tuned on SeasFire and ClimateNet, using the same parametrizations as in Figure 4a. As shown in Figure 4b, on ClimateNet, excluding the low-complexity edge case where SAE-Xplainer accuracy is limited, the absorption score increases with rule accuracy, as it does for SeasFire. Moreover, for SeasFire, since fire patterns are highly localized, more complex and accurate rules make co-occurrence of features with similar rule conditions more likely. For ClimateNet, however, ExEE patterns are far less localized, so co-occurrence is already high at low rule complexity, and the score decreases as complexity increases. The identification of feature absorption confirms the consistency between SAE-Xplainers' interpretations and SAE feature activations. Examples of absorbed features are provided in the supplementary material.

Samples Interpretation. The ensemble of SAE-Xplainers enables local, sample-level interpretations, as illustrated in Figure 5, where we analyze the activations of a ClimaX feature associated with fires on a sample from the SeasFire dataset. The rule-based surrogate model of the feature first identifies the most relevant input variables, for which the corresponding rule boundaries can then be directly extracted. This provides structured insights to support human understanding of complex ExEE models at the sample level Thanks to the high sparsity enforced during SAE training (with k = 4), we limit each sample to at most four active features, ensuring that sample-level interpretations remain concise and tractable for human analysis. Our method also enables global model interpretability by projecting SAE features into a 2D space and describing them via the SAE-Xplainers, as illustrated by the exploratory tool¹ presented in the supplementary material. Additional sample analyses over SeasFire and ClimateNet are also included in the supplementary material and provide qualitative examples of scientifically consistent rule semantics.

Scientific Semantics of SAE-Xplainers. We assess whether the extracted rules encode scientifically meaningful concepts by analyzing, on SeasFire, the top-100 features of SAE-Xplainers on GeoTopK with the highest fire ratio for nonspatial rules across all complexity levels (ratio score and other selection algorithms detailed in the supplementary material). Table 2 reports the proportion of rule sets containing established fire-risk conditions: high vpd, low tp, and low re1\_hum (Kudláčková et al. 2025), as $\mathbf { T r _ { c o n d } } ,$ and their inverse (low-risk) as $\mathbf { I n v _ { c o n d } } .$ SAE-Xplainers on GeoTopK contain literature-consistent conditions in 46% of cases, while inverse conditions occur in only 2%, compared with 16% for ClimaX neurons and 20% for prototypes, thus achieving semantic consistency comparable to the prototype interpretations while producing much fewer conflicting conditions.

![](images/24d89215a69948fd57219ef626775a1964559b35c55a5fbc5f6380880c5408bf.jpg)  
Figure 5: Sample-level feature interpretability for a ClimaX feature fine-tuned on SeasFire. (Left → Right): Feature activations. model prediction masks (no fires regions are masked), SAE-Xplainer rule visualizations with binary masks (unsatisfied rule regions are masked), and input variables highlighting the rule support. The complete rule set for this feature is displayed in supplementary material Figure 11.

![](images/36c33f2f3a818a6999b5b24c863a44e7b924ab9d9dac296a3634d43320497bd3.jpg)  
Figure 6: Semantic structure of the rules in the SAE-Xplainers of the 50 most fire-specific GeoTopK features. The levels represent the tree-like structure of each rule, while the nodes describe the categories of the conditions; N indicates the number of rules.

<table><tr><td>Metric</td><td></td><td>Neurons Prototypes GeoTopK</td><td></td></tr><tr><td> $\mathrm { T r } _ { \mathrm { c o n d } }$  ↑</td><td>0.33</td><td>0.50</td><td>0.46</td></tr><tr><td> $\mathrm { I n v } _ { \mathrm { c o n d } } .$  →</td><td>0.16</td><td>0.20</td><td>0.02</td></tr></table>

Table 2: Ratio of rule sets containing true (Tr) and inverse (Inv) conditions for fire-specific dimensions.

We then examine the complete rule structure of the top-50 SAE-Xplainers. As shown in Figure 6, 92% of the explanations contain conditions associated with at least one of seven literature-supported fire drivers (see supplementary material Table 8). For example, the NDVI Limits filters out extreme NDVI values, as fires typically ignite and spread at mid-range NDVI (Krawchuk et al. 2009). The remaining 8% are conservatively labeled as Unknown and reported in supplementary material Table 9. Overall, this targeted analysis shows that most fire-specific GeoTopK features admit compact interpretations grounded in established wildfire literature.

## Conclusion

SAEs offer a promising explainability alternative for ExEE tasks, beyond standard methods focusing mainly on where the model looks. We tackled two challenges related to using SAEs in ExEE research: the location-dependent patterns of the data, and the complex nature of input spaces for feature interpretation. We introduce GeoTopK, a locationaware SAE improving feature disentanglement, and SAE-Xplainers, a rule-based ensemble for feature interpretation. GeoTopK outperforms baselines across two ExEE datasets and transformer-based models, while SAE-Xplainers faithfully capture feature behaviors with semantic interpretations. Our approach could broadly generalize to other Earth system applications, such as species distribution modeling or air quality forecasting. Future work can study the robustness of the extracted rules, which can be improved via constrained SAE training (Fel et al. 2025) or knowledge-grounded surrogate models, and explore richer surrogate architectures to better capture the spatiotemporal context.

## Acknowledgments

We would like to thank Gaston Lenczner, IT engineer at ECEO, for developing the exploratory tool used in this work.

## References

Adebayo, J.; Gilmer, J.; Muelly, M.; Goodfellow, I.; Hardt, M.; and Kim, B. 2018. Sanity checks for saliency maps. Advances in neural information processing systems, 31.

Bohacek, M.; Fel, T.; Agrawala, M.; and Lubana, E. S. 2026. Uncovering Conceptual Blindspots in Generative Image Models Using Sparse Autoencoders. In The Fourteenth International Conference on Learning Representations.

Bowman, D. M.; Kolden, C. A.; Abatzoglou, J. T.; Johnston, F. H.; van der Werf, G. R.; and Flannigan, M. 2020. Vegetation fires in the Anthropocene. Nature Reviews Earth & Environment, 1(10): 500–515.

Breiman, L. 2001. Random Forests. Machine Learning.

Bricken, T.; Templeton, A.; Batson, J.; Chen, B.; Jermyn, A.; Conerly, T.; Turner, N.; Anil, C.; Denison, C.; Askell, A.; Lasenby, R.; Wu, Y.; Kravec, S.; Schiefer, N.; Maxwell, T.; Joseph, N.; Hatfield-Dodds, Z.; Tamkin, A.; Nguyen, K.; McLean, B.; Burke, J. E.; Hume, T.; Carter, S.; Henighan, T.; and Olah, C. 2023. Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits Thread.

Calef, M.; McGuire, A.; and Chapin III, F. 2008. Human influences on wildfire in Alaska from 1988 through 2005: an analysis of the spatial patterns of human impacts. Earth Interactions, 12(1): 1–17.

Camps-Valls, G.; Fernández-Torres, M.-Á.; Cohrs, K.-H.; Höhl, A.; Castelletti, A.; Pacal, A.; Robin, C.; Martinuzzi, F.; Papoutsis, I.; Prapas, I.; et al. 2025. Artificial intelligence for modeling and understanding extreme weather and climate events. Nature Communications, 16(1): 1919.

Chanin, D.; Wilken-Smith, J.; Dulka, T.; Bhatnagar, H.; Golechha, S.; and Bloom, J. I. 2025. A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. In The Thirtyninth Annual Conference on Neural Information Processing Systems.

Cilli, R.; Elia, M.; D'Este, M.; Giannico, V.; Amoroso, N.; Lombardi, A.; Pantaleo, E.; Monaco, A.; Sanesi, G.; Tangaro, S.; et al. 2022. Explainable artificial intelligence (XAI) detects wildfire occurrence in the Mediterranean countries of Southern Europe. Scientific reports, 12(1): 16349.

Covert, I. C.; Kim, C.; and Lee, S.-I. 2023. Learning to Estimate Shapley Values with Vision Transformers. In The Eleventh International Conference on Learning Representations.

Cui, H.; Tang, D.; Foltz, G. R.; Mei, W.; Liu, H.; Hoteit, I.; Li, C.; Zhang, H.; Sui, Y.; and Gu, X. 2025. Drivers of tropical cyclone-induced ocean cooling in different seasons over the Northwest Pacific from explainable machine learning. Journal of Geophysical Research: Machine Learning and Computation, 2(4): e2025JH000746.

Dikshit, A.; and Pradhan, B. 2021. Explainable AI in drought forecasting. Machine Learning with Applications, 6: 100192.

Dikshit, A.; Pradhan, B.; Assiri, M. E.; Almazroui, M.; and Park, H.- J. 2022. Solving transparency in drought forecasting using attention models. Science of The Total Environment, 837: 155856.

Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; Uszkoreit, J.; and Houlsby, N. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In International Conference on Learning Representations.

Eddin, M.; et al. 2023. Location-aware adaptive normalization for wildfire danger forecasting. IEEE TGRS.

Elhage, N.; Hume, T.; Olsson, C.; Schiefer, N.; Henighan, T.;Kravec, S.; Hatfield-Dodds, Z.; Lasenby, R.; Drain, D.; Chen, C.;

Grosse, R.; McCandlish, S.; Kaplan, J.; Amodei, D.; Wattenberg, M.; and Olah, C. 2022. Toy Models of Superposition. Transformer Circuits Thread.

Fel, T.; Lubana, E. S.; Prince, J. S.; Kowal, M.; Boutin, V.; Papadimitriou, I.; Wang, B.; Wattenberg, M.; Ba, D. E.; and Konkle, T. 2025. Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction in Large Vision Models. In International Conference on Machine Learning, 16543–16572. PMLR.

Gao, L.; la Tour, T. D.; Tillman, H.; Goh, G.; Troll, R.; Radford, A.; Sutskever, I.; Leike, J.; and Wu, J. 2025. Scaling and evaluating sparse autoencoders. In The Thirteenth International Conference on Learning Representations.

Hantson, S.; Arneth, A.; Harrison, S. P.; Kelley, D. I.; Prentice, I. C.; Rabin, S. S.; Archibald, S.; Mouillot, F.; Arnold, S. R.; Artaxo, P.; et al. 2016. The status and challenge of global fire modelling. Biogeosciences, 13(11): 3359–3375.

Hrast Essenfelder, A.; Toreti, A.; and Seguini, L. 2025. Expertdriven explainable artificial intelligence models can detect multiple climate hazards relevant for agriculture. Communications Earth & Environment, 6(1): 207.

Huben, R.; Cunningham, H.; Smith, L. R.; Ewart, A.; and Sharkey, L. 2024. Sparse Autoencoders Find Highly Interpretable Features in Language Models. In The Twelfth International Conference on Learning Representations.

Jain, P.; Castellanos-Acuna, D.; Coogan, S. C.; Abatzoglou, J. T.; and Flannigan, M. D. 2022. Observed increases in extreme fire weather driven by atmospheric humidity and temperature. Nature Climate Change, 12(1): 63–70.

Jolly, W. M.; Cochrane, M. A.; Freeborn, P. H.; Holden, Z. A.; Brown, T. J.; Williamson, G. J.; and Bowman, D. M. 2015. Climateinduced variations in global wildfire danger from 1979 to 2013. Nature communications, 6(1): 7537.

Karasante, I.; Alonso, L.; Prapas, I.; Ahuja, A.; Carvalhais, N.; and Papoutsis, I. 2025. SeasFire cube-a multivariate dataset for global wildfire modeling. Scientific Data, 12(1): 368.

Karvonen, A.; Rager, C.; Lin, J.; Tigges, C.; Bloom, J. I.; Chanin, D.; Lau, Y.-T.; Farrell, E.; Mcdougall, C. S.; Ayonrinde, K.; et al. 2025. SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability. In International Conference on Machine Learning, 29223–29264. PMLR.

Kim, B.; Wattenberg, M.; Gilmer, J.; Cai, C.; Wexler, J.; Viegas, F.; et al. 2018. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). In International conference on machine learning, 2668–2677. PMLR.

Koh, P. W.; Nguyen, T.; Tang, Y. S.; Mussmann, S.; Pierson, E.; Kim, B.; and Liang, P. 2020. Concept bottleneck models. In International conference on machine learning, 5338–5348. PMLR.

Kontschieder, P.; Fiterau, M.; Criminisi, A.; and Bulo, S. R. 2015. Deep neural decision forests. In Proceedings of the IEEE international conference on computer vision, 1467–1475.

Krawchuk, M. A.; Moritz, M. A.; Parisien, M.-A.; Van Dorn, J.; and Hayhoe, K. 2009. Global pyrogeography: the current and future distribution of wildfire. PloS one, 4(4): e5102.

Kudláčková, L.; Bartošová, L.; Linda, R.; Bláhová, M.; Poděbradská, M.; Fischer, M.; Balek, J.; Żalud, Z.; and Trnka, M. 2025. Assessing fire danger classes and extreme thresholds of the Canadian Fire Weather Index across global environmental zones: a review. Environmental Research Letters, 20(1): 013001.

Leask, P.; Bussmann, B.; Pearce, M. T.; Bloom, J. I.; Tigges, C.; Moubayed, N. A.; Sharkey, L.; and Nanda, N. 2025. Sparse Autoencoders Do Not Find Canonical Units of Analysis. In The Thirteenth International Conference on Learning Representations.

Letham, B.; Rudin, C.; McCormick, T. H.; and Madigan, D. 2015. Interpretable classifiers using rules and Bayesian analysis: Building a better stroke prediction model. The Annals of Applied Statistics, 9(3): 1350–1371.

Li, F.; Zeng, X.; and Levis, S. 2012. A process-based fire parameterization of intermediate complexity in a Dynamic Global Vegetation Model. Biogeosciences, 9(7): 2761–2780.

Lim, H.; Choi, J.; Choo, J.; and Schneider, S. 2025. Sparse autoencoders reveal selective remapping of visual concepts during adaptation. In The Thirteenth International Conference on Learning Representations.

Lovo, A.; Lancelin, A.; Herbert, C.; and Bouchet, F. 2025. Tackling the Accuracy-Interpretability Trade-Off in a Hierarchy of Machine Learning Models for the Prediction of Extreme Heatwaves. Artificial Intelligence for the Earth Systems, 4(3): 240094.

Lundberg, S. M.; and Lee, S.-I. 2017. A unified approach to interpreting model predictions. Advances in neural information processing systems, 30.

Makhzani, A.; and Frey, B. J. 2014. k-Sparse Autoencoders. In International Conference on Learning Representations.

Nguyen, T.; Brandstetter, J.; Kapoor, A.; Gupta, J. K.; and Grover, A. 2023. ClimaX: A foundation model for weather and climate. In International Conference on Machine Learning, 25904–25938. PMLR.

Paulo, G. S.; Mallen, A. T.; Juang, C.; and Belrose, N. 2025. Automatically Interpreting Millions of Features in Large Language Models. In International Conference on Machine Learning, 48393– 48421. PMLR.

Pechony, O.; and Shindell, D. 2009. Fire parameterization on a global scale. Journal of Geophysical Research: Atmospheres, 114(D16).

Perez, E.; et al. 2018. Film: Visual reasoning with a general conditioning layer. In AAAI.

Porta, H.; Dalsasso, E.; Marcos, D.; and Tuia, D. 2025. Multi-Scale Grouped Prototypes for Interpretable Semantic Segmentation. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2869–2880. IEEE.

Prabhat; Kashinath, K.; Mudigonda, M.; Kim, S.; Kapp-Schwoerer, L.; Graubner, A.; Karaismailoglu, E.; von Kleist, L.; Kurth, T.; Greiner, A.; et al. 2020. ClimateNet: an expert-labelled open dataset and deep learning architecture for enabling high-precision analyses of extreme weather. Geoscientific Model Development Discussions, 2020: 1–28.

Rao, S.; Mahajan, S.; Böhle, M.; and Schiele, B. 2024. Discoverthen-name: Task-agnostic concept bottlenecks via automated concept discovery. In European Conference on Computer Vision, 444– 461. Springer.

Ribeiro, M. T.; Singh, S.; and Guestrin, C. 2016. " Why should i trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, 1135–1144.

Ribeiro, M. T.; Singh, S.; and Guestrin, C. 2018. Anchors: Highprecision model-agnostic explanations. In Proceedings of the AAAI conference on artificial intelligence, volume 32.

Rodrigues, E. R.; Watson, C. D.; Zadrozny, B.; and Nyirjesy, G. 2022. FIRO: A Deep-neural Network for Wildfire Forecast with Interpretable Hidden States. In NeurIPS 2022 Workshop on Tackling Climate Change with Machine Learning.

Rudin, C. 2019. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature machine intelligence, 1(5): 206–215.

Sacha, M.; Rymarczyk, D.; Struski, Ł.; Tabor, J.; and Zieliński, B. 2023. Protoseg: Interpretable semantic segmentation with prototypical parts. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 1481–1492.

Selvaraju, R. R.; Cogswell, M.; Das, A.; Vedantam, R.; Parikh, D.; and Batra, D. 2017. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision, 618–626.

Seneviratne, S.; Zhang, X.; Adnan, M.; Badi, W.; Dereczynski C.; Di Luca, A.; Ghosh, S.; Iskander, I.; Kossin, J.; Lewis, S.; et al. 2023. Weather and Climate Extreme Events in a Changing Climate (Chapter 11). IPCC 2021: Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change, 1513–1766.

Srinivasan, R.; Wang, L.; and Bulleid, J. 2020. Machine learningbased climate time series anomaly detection using convolutional neural networks. Weather and Climate, 40(1): 16–31.

Templeton, A.; Conerly, T.; Marcus, J.; Lindsey, J.; Bricken, T.; Chen, B.; Pearce, A.; Citro, C.; Ameisen, E.; Jones, A.; Cunningham, H.; Turner, N. L.; McDougall, C.; MacDiarmid, M.; Freeman, C. D.; Sumers, T. R.; Rees, E.; Batson, J.; Jermyn, A.; Carter, S.; Olah, C.; and Henighan, T. 2024. Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. Transformer Circuits Thread.

Thasarathan, H.; Forsyth, J.; Fel, T.; Kowal, M.; and Derpanis, K. G. 2025. Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment. In International Conference on Machine Learning, 59304–59325. PMLR.

Tuia, D.; Schindler, K.; Demir, B.; Zhu, X. X.; Kochupillai, M.; Džeroski, S.; van Rijn, J. N.; Hoos, H. H.; Frate, F. D.; Datcu, M.; Markl, V.; Saux, B. L.; Schneider, R.; and Camps-Valls, G. 2025. Artificial Intelligence to Advance Earth Observation: A review of models, recent trends, and pathways forward. IEEE Geoscience and Remote Sensing Magazine, 13(4): 119–141.

Van Der Linden, I.; Haned, H.; and Kanoulas, E. 2019. Global aggregations of local explanations for black box models. arXiv preprint arXiv:1907.03039.

Wang, T.; Rudin, C.; Doshi-Velez, F.; Liu, Y.; Klampfl, E.; and MacNeille, P. 2017. A bayesian framework for learning rule sets for interpretable classification. Journal of Machine Learning Research, 18(70): 1–37.

Wang, Z.; Zhang, W.; Liu, N.; and Wang, J. 2021. Scalable rule-based representation learning for interpretable classification. Advances in Neural Information Processing Systems, 34: 30479– 30491.

Wei, J.; Bora, A.; Oommen, V.; Dong, C.; Yang, J.; Adie, J.; Chen, C.; See, S.; Karniadakis, G.; and Mengaldo, G. 2025. XAI4Extremes: An interpretable machine learning framework for understanding extreme-weather precursors under climate change. In ICLR 2025 Workshop on Tackling Climate Change with Machine Learning.

Yang, H.; Rudin, C.; and Seltzer, M. 2017. Scalable Bayesian rule lists. In International conference on machine learning, 3921–3930. PMLR.

Yang, Y.; Ren, W.; and Li, S. 2024. Hyperlogic: Enhancing diversity and accuracy in rule learning with hypernets. Advances in Neural Information Processing Systems, 37: 3564–3587.

## Implementation Details

The code for this paper is available at https://github.com/ eceo-epfl/SAE-Xplainers. The experiments are all done on an NVIDIA GeForce RTX 3090 Ti GPU with 24 GB of memory, and an Intel Xeon W-2275 CPU with 125 GB of memory on Ubuntu 22.04. If not specified, all runs are done once, and as shown in the code all seeds are set using PyTorch Lightning seed\_everything function.

## Dataset Descriptions

SeasFire is a global dataset for dense wildfire forecasting across multiple horizons (from 8 days to 128 days, in 8-day intervals). In this paper, we focus on the shorter 8-day horizon to ensure strong performance of the analyzed models: ClimaX and ViT, without the need to integrate the temporal dimension (leading to a more complex interpretation of the models' behavior). SeasFire spans 21 years from 2001 to 2021 with aggregated input predictors over 8-day windows and a spatial grid of resolution 0.25° from multiple sources, as detailed in Table 3. Samples in SeasFire are constructed as 80 × 80 images extracted from the global Earth representation, where at least one fire occurs, leading to 44k samples over 17.7M patches (4 × 4 pixels), with 14 input variables as described in Table 3. 98% of those patches are assigned to the no fire class, causing the strong imbalance.

ClimateNet is a global dataset for the detection of tropical cyclones and atmospheric rivers via human-expert annotations on daily simulations of a climate model: CAM5.1, from 1996 to 2013. In ClimateNet, samples represent Earth as a whole in images of spatial size 768 × 1152 pixels with 6 input predictors (see Table 3), at a spatial resolution of 25 km. This constitutes an important difference w.r.t. SeasFire, as the detection tasks leverage a large spatial context compared to wildfire forecasting. Therefore, ClimateNet complements SeasFire in two ways: it focuses on two different events, each one triggered by specific patterns and requiring the capture of long-range phenomena. ClimateNet is composed of 459 samples over 25.4M patches, with 94% of no event patches.

Our selection of these datasets is motivated first by the growing danger and frequency of these ExEEs under climate change, and second by their data characteristics: although they share input-space similarities that raise similar challenges for SAE feature interpretation, they differ substantially in task formulation, sample construction, and W&C data generation (reanalysis vs. simulation).

## Models fine-tuning

For both datasets, ClimaX is independently fine-tuned, leveraging its pretrained weights at \~ 1.4°. On top of the last layer of ClimaX (ht), we train a lightweight MLP decoder with two hidden layers. The set of pre-training variables from ClimaX only partially overlapped with the predictors from SeasFire and ClimateNet. As a consequence, for the initialization of the variable-specific embedding and projection, when there is no direct match in the pre-training set for SeasFire and ClimateNet predictors, we manually matched them to the closest (or the group of closest) pre-training variables. For instance, the vpd projection and embedding in SeasFire is initialized with the mean of the weights from the following pre-training variables: relative humidity and temperature at 2 meters. Finally, if no group of pre-training variables would match, we initialized the variable-specific weights with the average across all pre-training variables. Experimentally, we observe that, when trained from scratch, ClimaX only reaches \~ 0.5 PRAUC on SeasFire compared to the results in Table 4, highlighting the benefits of fine-tuning from its pre-trained weights.

On both datasets, ClimaX is fine-tuned via the ADAMW optimizer and a cosine annealing scheduler with linear warmup. The learning rate is also adjusted for the ClimaX backbone compared to the first projection heads and the decoder by a factor of 0.2. For the hyperparameters, we directly reuse the ones leveraged in the ClimaX fine-tuning use case from (Nguyen et al. 2023) with learning rate: 5e — 4, warm-up learning rate: 1e — 8, betas: [0.9, 0.999], and weight decay: 1e – 5. We fine-tune across a total of 50 epochs with 5 for warm-up and a batch size of 64 for SeasFire and 8 for ClimateNet (due to the large image size for this dataset). Across both datasets, fine-tuning with an adjusted learning rate always outperforms freezing the backbone weights. We use standard cross-entropy loss for SeasFire and crossentropy combined with dice loss on ClimateNet, after testing cross-entropy and dice loss alone. During fine-tuning on ClimateNet, we randomly center crop to a size of 768 × 768 due to computational constraints, but run inference on the original size of 768 × 1152. The selection of the fine-tuning hyperparameters is done following the criterion presented in Table 4. Nonetheless, the objective of our paper is not to improve downstream performance on either dataset, but rather to remain close to the original papers’ reported baselines while focusing on the interpretability of the resulting models.

A ViT backbone is also trained from scratch on both datasets with the same lightweight MLP decoder as ClimaX. We leverage similar training hyperparameters as with ClimaX for the optimizer and learning rate scheduler, without adjusting the learning rate for the backbone. Experimentally, we just adapt the learning rate to 1e — 5 for SeasFire and 5e — 5 for ClimateNet by testing across the following values: [5e – 4, 1e − 4, 5e – 5, 1e − 5], as we observe overfitting at 5e — 4. The loss is the same for SeasFire, but on ClimateNet we experimentally found that weighted cross-entropy combined with dice outperforms standard cross-entropy. The weights selected are 0.2 for the background class and 1 for both event classes. The results in Table 4 show that ClimaX outperforms the ViT on both datasets. Therefore, while SAE Performance Analysis is conducted on both models across SeasFire and ClimateNet, for the Rule-based Feature Interpretations of our SAE-Xplainers we focus on ClimaX only.

## SAE Training

On both datasets for ClimaX and ViT, we train a single linearlayer SAE with the TopK activation and a batch size of 256. To guarantee that the TopKSAE activations are non-negative, we add a ReLU function in the encoder as in (Gao et al. 2025). The geolocation encoder in GeoTopK is composed of an MLP with two hidden layers. All our SAEs are trained via mean squared error and an auxiliary loss term to decrease the dead features rate. We train for 20 epochs on SeasFire, and 60 epochs on ClimateNet, as the $\scriptstyle { \mathrm { R } } ^ { 2 }$ and dead features rate criterion converge at different speeds across the datasets. The auxiliary loss term is an adaptation of ghost gradient for TopK SAE from the Overcomplete library. The pseudo-code for the auxiliary loss term is available in Algorithm 1. Prior to computing the loss in Equation 2, the reconstructed patches $\hat { h } _ { t , \mathrm { m o d } }$ are projected back to the original activation space via the following inversion formula:

<table><tr><td>Dataset</td><td>Acronym</td><td>Name</td><td>Units</td><td>Type</td><td>Source</td></tr><tr><td rowspan="18">SeasFire ClimateNet</td><td>lst_day ndvi</td><td>Land Surface Temperature (Day) Normalized Difference Vegetation Index</td><td>K unitless</td><td>Land Surface (Weather) Vegetation</td><td>MODIS MODIS</td></tr><tr><td>mslp</td><td>Mean Sea Level Pressure</td><td> $\mathrm { P a }$ </td><td>Weather</td><td>ERA5</td></tr><tr><td>rel_hum</td><td>Relative Humidity</td><td> $\%$ </td><td>Weather</td><td>ERA5</td></tr><tr><td>ssrd</td><td>Surface Solar Radiation Downwards</td><td> $\mathbf { M J \cdot m ^ { - 2 } }$ </td><td>Weather</td><td></td></tr><tr><td>sst</td><td></td><td></td><td></td><td>ERA5</td></tr><tr><td></td><td>Sea Surface Temperature</td><td> $\mathbf { K }$ </td><td>Climate</td><td>ERA5</td></tr><tr><td>swvl1</td><td>Volumetric Soil Water Layer 1</td><td>unitless</td><td>Land Surface (Weather)</td><td>ERA5</td></tr><tr><td>t2m_mean</td><td>2m Temperature Mean</td><td> $\mathbf { K }$ </td><td>Weather</td><td>ERA5</td></tr><tr><td>tp</td><td>Total Precipitation</td><td> $\mathbf { m }$ </td><td>Weather</td><td>ERA5</td></tr><tr><td>vpd</td><td>Vapor Pressure Deficit</td><td> $\mathrm { h P a }$ </td><td>Weather</td><td>ERA5</td></tr><tr><td>ws10</td><td>Wind Speed at 10m</td><td> $\mathbf { m } \cdot \mathbf { s } ^ { - 1 }$ </td><td>Weather</td><td>ERA5</td></tr><tr><td>pop_dens lat</td><td>Population Density</td><td> $\mathrm { p e r s o n s \cdot k m ^ { - 2 } }$ </td><td>Anthropogenic</td><td>GPWv4</td></tr><tr><td>lon</td><td>Latitude Longitude</td><td> $\scriptstyle { \mathrm { d e g r e e s } }$  degrees</td><td>Geographic</td><td></td></tr><tr><td>TMQ</td><td>Total Vertically Integrated Precipitable Water</td><td> $\mathrm { k g \cdot m ^ { - 2 } }$ </td><td>Geographic</td><td></td></tr><tr><td>U850</td><td>Zonal Wind at 850 mbar</td><td> $\mathbf { m } \cdot \mathbf { s } ^ { - 1 }$ </td><td>Weather</td><td>CAM5.1</td></tr><tr><td>V850</td><td>Meridional Wind at 850 mbar</td><td> $\mathbf { m } \cdot \mathbf { s } ^ { - 1 }$ </td><td>Weather</td><td>CAM5.1</td></tr><tr><td>PSL</td><td></td><td></td><td>Weather</td><td>CAM5.1</td></tr><tr><td></td><td>Sea Level Pressure</td><td> $\mathrm { P a }$ </td><td>Weather</td><td>CAM5.1</td></tr><tr><td>lat</td><td>Latitude</td><td>degrees</td><td>Geographic</td><td></td></tr><tr><td>lon</td><td>Longitude</td><td>degrees</td><td>Geographic</td><td></td></tr></table>

Table 3: Summary of predictors used from SeasFire and ClimateNet datasets in our paper.
<table><tr><td rowspan="2">Metric</td><td colspan="4">SeasFire</td><td colspan="4">ClimateNet</td></tr><tr><td>ClimaX</td><td>ViT</td><td>ProtoSeg</td><td> $\mathbf { U } \mathbf { - } \mathbf { N e t + + } \cdots$ </td><td>ClimaX</td><td>ViT</td><td></td><td>ProtoSeg DeepLabV3+*</td></tr><tr><td>PRAUC ↑</td><td>0.6205</td><td>0.6150</td><td>0.5161</td><td>0.6100</td><td></td><td></td><td></td><td></td></tr><tr><td>IoU↑</td><td></td><td></td><td></td><td></td><td>0.5385</td><td>0.4972</td><td>0.4254</td><td>0.5247</td></tr></table>

Table 4: Downstream task performance of ClimaX, ViT, and ProtoSeg fine-tuned on SeasFire and ClimateNet. \* denotes results reported from the original papers.

$$
\hat { h } _ { t } = \frac { \hat { h } _ { t , \mathrm { m o d } } - \operatorname { t a n h } ( \beta ) } { 1 + \operatorname { t a n h } ( \alpha ) + \epsilon } \ : ,\tag{10}
$$

with $\epsilon = 1 e - 8 $ We leverage the ADAM optimizer with no scheduler to train the SAEs. For ClimaX fine-tuned on SeasFire, we test for TopK and GeoTopK, multiple levels of regularization for the auxiliary loss $\gamma \in [ 0 . 1 , 0 . 0 \bar { 1 } , 0 . 0 0 1 ]$ at a fixed learning rate: $5 e - 4 ,$ and identify $\gamma = 0 . 0 1$ as the best for both SAEs. For ClimaX fine-tuned on ClimateNet, we apply a similar strategy while also adapting the learning rate due to overfitting, finding as the best regularization: $\gamma = 0 . 1$ for TopK, with a learning rate: $5 e - { \bar { 5 } } .$ , and for GeoTopK:

$\gamma = 0 . 0 1$ and a learning rate: 5e — 6. The hyperparameters selection is based on the $\mathrm { R } ^ { 2 }$ and dead features rate criterion; when $\mathbb { R } ^ { 2 }$ reaches a high value, we focus on optimizing the dead features rate, as SAEs can often reach a degenerate optimum with high reconstruction performance but only a handful of active features.

Moreover, to decrease the dead features rate we tie at initialization the encoder and decoder weights: $W _ { e n c } = W _ { d e c } ^ { T }$ following (Huben et al. 2024) and, as the rate remains high on ClimateNet, we employ the feature resampling strategy from (Bricken et al. 2023) to further decrease it by resampling every epoch across 4.5 M patches.

For the ViT models on both datasets, we directly reuse the best hyperparameters selected for ClimaX. The impact of the latent dimension size on the SAEs’ performance is detailed in subsection Scaling SAE Latent Dimension.

## SAE-Xplainers Training

The attention scores leveraged in Equation 8 are aggregated by averaging across all layers up to l: $\begin{array} { r } { W _ { \mathrm { a t t } } ^ { l } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } A _ { i } } \end{array}$ with B the number of attention blocks and $A _ { i } ~ \in ~ \mathbb { R } ^ { T \times T }$ the attention matrix. Then patch-level scores are $A ^ { l } ( x _ { t } ) =$

Algorithm 1: Top-K Auxiliary Loss (AuxK)   
Require: Input $h _ { t } ,$ reconstruction $\hat { h } _ { t } ,$ pre-activations $z _ { \mathrm { p r e } } ,$   
activations $z ,$ dictionary $W _ { d e c }$   
1: $r \gets h _ { t } - { \hat { h } } _ { t }$ Compute residual   
2: $z _ { \mathrm { p r e } } \gets \mathrm { R e L U } ( z _ { \mathrm { p r e } } ) - z$ Isolate unchosen feature   
activations   
3: $\mathcal { T }  \mathrm { T o p K } ( z _ { \mathrm { p r e } } , k = \lfloor N / 2 \rfloor )$ Select top-50%   
unchosen features   
4: $\tilde { z }  0 ; \quad \tilde { z } _ { \mathbb { Z } }  z _ { \mathrm { p r e } , \mathbb { Z } }$ ▶ Build sparse auxiliary vector   
5: $\hat { r }  W _ { d e c } \tilde { z } \quad \triangleright$ Predict residual from auxiliary features   
6: $\mathcal { L } _ { \mathrm { a u x } }  \| r - \hat { r } \| _ { 2 } ^ { 2 }$   
return $\mathcal { L } _ { \mathrm { a u x } }$

[Watt, .

We train our set of SAE-Xplainers on features with enough active samples for both datasets. On SeasFire, due to a low dead features rate and the overall high activation counts of the learned features, we target dropping below 1% of them by applying a threshold of 1600 active samples across train and evaluation sets (test and validation). On the contrary, with ClimateNet, given our high dead features rate and low activation counts, we target dropping around 10% of the features with a threshold of 160 to ensure sufficient samples for training the surrogate models; as a comparison in (Huben et al. 2024), the LLM autointerpretability framework only uses 20 samples of 64-token sentence-fragments. Following the toprandom balanced sampling strategy of (Huben et al. 2024; Paulo et al. 2025), active patches are sampled by selecting the most activated half, and the remaining half are randomly sampled from the active distribution. For the neuron and prototype baselines, we first binarize patch activations by thresholding at the ${ \ 9 9 } ^ { t h }$ percentile of each dimension, then apply the same top-random balanced strategy.

In Table 5, we detail the Skope-Rules parametrization for the different complexity levels explored for the training of the SAE-Xplainers. Skope-Rules is a Python machine learning module that extracts interpretable if-then logical rules from an ensemble of decision trees, weighting each rule by its out-of-bag precision. For both datasets, as they are global we trained with a radius $\sigma = 1 0 0 0$ km as a good range to find proper negatives still close in local environmental patterns. We evaluate the adversarial sampling strategy against standard random sampling in the subsection Sampling Analysis. We leverage for both datasets a spatial context of 20 patches.

<table><tr><td rowspan="2">Levels</td><td colspan="4">Parameters</td></tr><tr><td>Max Depth # Estim. Min Precision Min Recall</td><td></td><td></td><td></td></tr><tr><td>Low</td><td>3</td><td>3</td><td>0.90</td><td>0.20</td></tr><tr><td>Mid</td><td>4</td><td>5</td><td>0.80</td><td>0.15</td></tr><tr><td>High</td><td>5</td><td>5</td><td>0.70</td><td>0.10</td></tr></table>

Table 5: Skope-Rules parametrization across complexity levels. # Estim. denotes the number of estimators.

## Absorption Scores

For the co-occurrence score, we select the $M = 5 0$ closest pairs and set the co-occurrence threshold to $\tau = 0 . 1$ Additionally, we consider two rule-condition threshold values equivalent if their difference is within 5% of the target variable's standard deviation.

For the SAEBench absorption score, we reuse the same probe and $\ell _ { 1 } \cdot$ -probe hyperparameters as in (Karvonen et al. 2025), and set the cosine similarity threshold to 0.0 and the F1 threshold to 0.2.

## Prototype Network Training

To study how our Skope-Rules sets compare against a different family of interpretable concept-extraction methods, we train a dense prediction prototype network, ProtoSeg (Sacha et al. 2023; Porta et al. 2025), on top of ClimaX fine-tuned on SeasFire and ClimateNet. Since ClimaX is already finetuned on the target task, we skip ProtoSeg's joint training stage and run only the warm-up and fine-tuning stages: the prototype layer is trained during warm-up with the decoder head frozen, and the decoder head is subsequently trained once the prototypes have been pushed to real patch samples.

We reuse the ClimaX fine-tuning hyperparameters, with two exceptions: we replace AdamW with Adam and omit the learning-rate scheduler, since we train for substantially fewer epochs (10 per stage). We also do not apply the adjusted learning rate, as the backbone is frozen throughout. For the loss, we use cross-entropy for SeasFire and a combined cross-entropy/dice loss for ClimateNet, consistent with the best-performing configurations reported for ClimaX. For regularization, we adopt the losses defined in ProtoSeg: a Kullback-Leibler divergence term weighted at 0.25 and an $\ell _ { 1 }$ penalty on the decoder head weighted at $1 e - 4$

We evaluate two distance formulations for the prototype layer: the standard Euclidean distance paired with ProtoSeg's logarithmic activation, and a cosine distance paired with a linear activation. On SeasFire, the cosine-distance variant outperforms the standard formulation, so we adopt it for all subsequent experiments. Finally, we use 20 prototypes per class, yielding 40 prototypes total for SeasFire (2 classes) and 60 for ClimateNet (3 classes).

## GeoTopK Additional Results

## Ablation Study

In our method, GeoTopK is parametrized via a feature-wise affine modulation. Through the ablation study in Table $^ { 6 , }$ we compare the performance of our GeoTopK with other potential parametrizations of the geographic location modulation. We train on ClimaX fine-tuned for SeasFire a scalar modulation with $( \alpha , \beta ) \in \mathbb { R } ^ { 2 }$ , a shift-only modulation with $\alpha = 1$ and $\beta \in \mathbb { R } ^ { D }$ , and a scale-only modulation with $\boldsymbol { \alpha } \in \mathbb { R } ^ { D }$ and $\beta = 0$ We additionally compare against location-aware adaptive normalization (LOAN) (Eddin et al. 2023) and perlatitude SAE band bias (LatBias). The results show that, across all metrics, GeoTopK parametrization performs best except on $\mathrm { R } _ { \mathrm { e v e n t } } ^ { 2 } ,$ and its performance is close to that of the shift-only parametrization. Computing the MSE loss in the modulated space leads to a trivial solution with zero scale and shift vectors $( \mathsf { R } ^ { 2 } < 0$ on SeasFire). Finally, tanh bounds the scale and shift magnitudes, unlike GELU or ReLU, to avoid that the SAE only learns to represent geographic locations.

<table><tr><td>Metric</td><td>Scalar</td><td>Shift-Only</td><td>Scale-Only LOAN LatBias</td><td></td><td></td><td>GeoTopK</td></tr><tr><td> $\mathsf { R } ^ { 2 } \uparrow$ </td><td>0.930</td><td>0.936</td><td>0.931</td><td>0.754</td><td>0.928</td><td>0.937</td></tr><tr><td> $\mathbf { R } _ { \mathrm { w o r s t } } ^ { 2 } \uparrow$ </td><td>0.884</td><td>0.899</td><td>0.889</td><td>0.616</td><td>0.881</td><td>0.901</td></tr><tr><td> $\mathrm { R } _ { \mathrm { e v e n t } } ^ { 2 } \uparrow$ </td><td>0.808</td><td>0.802</td><td>0.817</td><td>0.726</td><td>0.806</td><td>0.807</td></tr><tr><td>MSE↓</td><td>1.77e-3</td><td>1.61e-3</td><td>1.73e-3</td><td>0.243</td><td>1.82e-3</td><td>1.59e-3</td></tr><tr><td>Dead Features ↓</td><td>0.041</td><td>0.004</td><td>0.003</td><td>0.002</td><td>0.107</td><td>0.002</td></tr></table>

Table 6: Ablation study of the geographic modulation in GeoTopK on ClimaX fine-tuned on SeasFire. Bold means best score, underline means similar score (difference is $\leq 0 . 0 0 5 )$ 1

![](images/73a9a9bb632987882aff3dc61a0aff481a3b3b56d1a8f73fb3d4818541bb8584.jpg)  
Figure 7: TopK and GeoTopK performance (Left: Dead Features Rate, Right: $\mathsf { R } ^ { 2 } )$ when scaling the latent dimension size: $N ,$ at fixed sparsity ratios for ClimaX, fine-tuned on SeasFire.

## Scaling SAE Latent Dimension

We study whether the benefits of GeoTopK persist as we increase the latent dimension size of the SAE. Focusing on ClimaX for SeasFire, we scale this size from N : 4096 → 16384 at a fixed sparsity ratio $k : 4 \to 1 6 ( \mathrm { F i g u r e } 7 )$ . We observe on the SeasFire test that with more capacity (higher k and N), the TopK baseline performance converges towards GeoTopK with similar $\mathrm { R } ^ { 2 }$ . However, the baseline consistently suffers from a significantly higher percentage of dead features. We hypothesize that larger TopK models can explicitly approximate the geographic priors encoded in GeoTopK, but do so less efficiently. GeoTopK achieves better results with fewer parameters, enabling high-performance SAEs at a reduced computational cost.

## SAE-Xplainers Additional Results

## Features Assignment

We show in Figure 8 the different patterns between SeasFire and ClimateNet in feature assignment for the SAE-Xplainers. On SeasFire, the spatial dimension has less importance than for ClimateNet, where a majority of the features are assigned to Skope-Rules parametrization with spatial context. This can be explained by the types of extreme events and tasks tackled in ClimateNet, which are inherently more spatial, but also by the difference in the number of predictors available in both datasets: 14 vs $^ { 6 , }$ as such, each patch in SeasFire contains more information.

## Sampling Analysis

The sampling for the training of the SAE-Xplainers directly impacts their observed faithfulness and the feature interpretation quality. We analyze, across three runs of SAE-Xplainers for ClimaX fine-tuned on SeasFire, three metrics quantifying feature interpretation quality. We focus on the most used parametrization setup in the final ensemble of SAE-Xplainers: low, without spatial context (see Figure 8). We first assess if the interpretability of features remains consistent across independent rule sets by calculating the Stability: it measures, for each feature, the pairwise average Jaccard distance between the rule sets across runs, using the Hungarian algorithm to match corresponding rules. The $\mathbf { W e i g h t _ { l o c } }$ allows us to study the importance of the geographic coordinates in feature interpretations by measuring the ratio of rules with a condition on either latitude or longitude. Finally, the $\mathbf { T r _ { c o n d } }$ assesses the agreement between model's prediction and rule interpretation by measuring the proportion of the top-100 fire-specific features interpreted by rules that contain known patterns leading to high fire risk: high vpd, low tp and low re1\_hum, as defined in the Section Rule-based Feature Interpretations.

Results in Table 7 shows that both sampling strategies lead to similar stability, with an improvement in rules quality for the adversarial sampling: the rules focus on fire drivers other than the geographic coordinates (lower $\bf { W e i g h t } _ { l o c } )$ and both their interpretation are in agreement with known and interpretable fire patterns (high $\bf { T r _ { c o n d } } )$

![](images/36116ca8733927bee9da85cad0384e4d0123fef9d54e486a3325a385cdb3656f.jpg)

![](images/0dfc63f34d2ebe921efae94af2b6d2bd7eee1486e464154aa6271d9786b6e2ef.jpg)  
Figure 8: Feature assignment across the Skope-Rules parametrization: low, mid, and high, with or without spatial context

![](images/5f49be5ea77674d06b83c70e1fc42a6cdb80c71c922c7dc3768871e198b66f4d.jpg)  
Figure 9: Exploratory tool for global model interpretations showcasing feature vectors as points in 2 dimensions extracted through a UMAP projection, and color-coded by event class selectivity. When a feature is selected, its rule-based interpretations, geographic activations, and co-occurring features are shown.

## Global Model Interpretations.

GeoTopK and SAE-Xplainers can be leveraged in synergy for global model analysis on ExEE tasks. To support this, we developed an exploratory tool inspired by (Bohacek et al. 2026) (see Figure 9). The tool enables different visualizations that domain experts can build upon to derive insights: a UMAP projection of SAE dictionary vectors $d _ { i }$ into two dimensions to visualize feature similarity and enable cluster analysis of related concepts, interactive access to feature co-occurrences, feature descriptions with detailed rule information from SAE-Xplainers, and geographic activations. Leveraging the interpretability enabled by GeoTopK and SAE-Xplainers, the tool provides ExEE experts with intuitive visualizations that can be turned into climatic insights. The exploratory tool is available online at https://eceo-epfl.github.io/SAE-Xplainers-umap and focuses on non-spatial rules only.

<table><tr><td rowspan="2">Sampling</td><td colspan="3">Metric</td></tr><tr><td>Stability↓</td><td> $\mathbf { W e i g h t _ { l o c } \downarrow }$ </td><td> $\mathbf { T r _ { c o n d } } \uparrow$ </td></tr><tr><td>Random</td><td> ${ \bf 0 . 6 8 8 \pm 0 . 0 2 3 }$ </td><td> $0 . 6 3 3 \pm 0 . 0 5 9$ </td><td> $0 . 4 9 0 \pm 0 . 0 6 4$ </td></tr><tr><td>Adversarial</td><td> $0 . 7 0 5 \pm 0 . 0 1 7$ </td><td> $\mathbf { 0 . 5 7 4 \pm 0 . 0 7 1 }$ </td><td> $\mathbf { 0 . 4 9 7 \pm 0 . 0 7 4 }$ </td></tr></table>

Table 7: Comparison of the sampling strategies for training the SAE-Xplainers on ClimaX fine-tuned on SeasFire. The reported results are average across three runs with their standard deviation.

## Sample Interpretations

In Figure 11 and 12, we provide other instances of samplelevel interpretations for features linked to fires for ClimaX fine-tuned on SeasFire. Moreover, in Figure 13 and 14 we introduce examples of sample-level interpretations for features linked to atmospheric rivers for ClimaX fine-tuned on ClimateNet, where we zoom in on a specific window of high feature activations for the given sample. These additional visualizations show the high interpretability of our SAE-Xplainers over different datasets and features.

## Absorbed Features

In Figure 15 and 16, we share examples of groups of absorbed features on the ClimateNet and SeasFire datasets. In Figure 15, Feature 3696 is the main feature, partially absorbed when it co-occurs with Feature 1670. The SAE-Xplainers interpretations are very similar for those features, while they partially co-occurred. This caused their selection to compute the SAEBench absorption score, defining the prediction task with the rule-based surrogate model assigned to the main feature.

## Details on SAE-Xplainers Semantic Consistency.

In this section, we provide details and extend the analysis of the SAE-Xplainers semantic meaning for the fire forecasting use-case. First, we introduce the selection score for each dimension type: neurons, prototypes, and GeoTopK features, used in Table 2 and Figure 6. For prototypes, we simply select the ones assigned in the first training stage to the class fire before the push stage, as described in (Sacha et al. 2023; Porta et al. 2025). The neurons and SAE features are selected as dimensions with the highest fire class ratio $\mathcal { R } _ { f i r e , i } .$ The feature ratio for the target class fire is computed over the SeasFire test set with $z _ { i } ( x _ { t } )$ the activation of feature $d _ { i }$ for the input patch sample $x _ { t } ,$ and $\mathcal { C } ^ { * } = \{ x _ { t } \mid y _ { t } = c ^ { * } , x \in \mathcal { X } \}$ the set of input patch samples X' assigned to the target class $c ^ { * } ;$

$$
\mathcal { R } _ { c ^ { * } , i } = \frac { \sum _ { \boldsymbol { x } _ { t } \in \mathcal { C } ^ { * } } | \boldsymbol { z } _ { i } ( \boldsymbol { x } _ { t } ) | } { \sum _ { \boldsymbol { x } _ { t } \in \mathcal { X } } | \boldsymbol { z } _ { i } ( \boldsymbol { x } _ { t } ) | + \epsilon } .\tag{11}
$$

For GeoTopK, we have $| z _ { i } ( x _ { t } ) | ~ = ~ z _ { i } ( x _ { t } )$ due to the TopKSAE encoder that internally contains a ReLU layer.

In Table 8, we provide the scientific descriptions with references from the wildfire literature of the categories used in the semantic structure analysis from Figure 6. This shows that the topics that emerge from the fire-specific SAE-Xplainers interpretations are deeply rooted in the wildfire literature and represent key global drivers of fire risk. In Figure 10a, we plot the most used rule conditions across our top-50 rule sets and observe that they can all be assigned to one of the categories from Table 8. We also analyze in Figure 10b how the GeoTopK features specific to each category co-occur at the patch level on the SeasFire test set. For instance, the features assigned to the NDVI Limits category, a descriptor of the local fuel load, co-occur with fire weather categories such as High VPD or Low Humidity that indicate fuel and soil aridity. Lastly, we list in Table 9 the SAE-Xplainers that are mapped to the Unknown with the corresponding unidentified condition.

<table><tr><td>Category</td><td>Usage</td><td>Ex. Feature</td><td>Description</td><td>Reference</td></tr><tr><td>NDVI Limits</td><td>56%</td><td>632</td><td>NDVI, a proxy for vegetation productivity, shows a humped relationship with fire risk: both sparse and overly healthy (green) vegetation suppress it.</td><td>(Krawchuk et al. 2009; Hantson et al. 2016; Bowman et al. 2020)</td></tr><tr><td>Population Density Limits</td><td>34%</td><td>3734</td><td>Fire risk peaks at intermediate population density, the wildland-urban interface, since fire ignition rises but fire spread falls with density, producing a humped relationship.</td><td>(Calef, McGuire, and Chapin III 2008; Hantson et al. 2016; Pechony and Shindell 2009; Li, Zeng, and Levis 2012)</td></tr><tr><td>Low Humidity</td><td>30%</td><td>49</td><td>Low relative humidity dries fuel and soil, favoring ignition and spread, and is thus negatively correlated with fire risk.</td><td>(Li, Zeng, and Levis 2012; Jolly et al. 2015; Jain et al. 2022; Kudláčková et al. 2025)</td></tr><tr><td>High VPD</td><td>26%</td><td>3815</td><td>Vapor pressure deficit (VPD): air dryness at a given temperature, drives fuel aridity and is positively correlated with ignition and spread.</td><td>(Pechony and Shindell 2009; Jain et al. 2022; Kudláčková et al. 2025)</td></tr><tr><td>Low Precipitation</td><td>18%</td><td>2728</td><td>Precipitation, closely tied to humidity, sustains fuel and soil moisture, and is thus negatively correlated with fire risk.</td><td>(Hantson et al. 2016; Jolly et al. 2015; Kudláčková et al. 2025)</td></tr><tr><td>High Temperature</td><td>14%</td><td>700</td><td>High temperature promotes fuel aridity and is generally positively correlated with fire risk. Extreme temperature can suppress fire</td><td>(Hantson et al. 2016; Bowman et al. 2020; Jolly et al. 2015; Jain et al. 2022)</td></tr><tr><td>Non-High Temperature</td><td>8%</td><td>1910</td><td>risk in fuel-limited regions, mirroring the humped relationship seen for NDVI and population density. In our model, it always co-occurs with low humidity/precipitation.</td><td>(Hantson et al. 2016; Bowman et al. 2020)</td></tr></table>

Table 8: Details on the categories used for the fire-specific SAE-Xplainers semantic analysis in Figure 6 and their scientific meaning.

<table><tr><td colspan="3">GeoTopK Feature Eval. Accuracy Complexity Rule Set</td><td></td><td>Conditions</td></tr><tr><td>1160</td><td>0.92</td><td>Mid</td><td> $\mathrm { \ m s l p { < } = 1 0 1 1 7 4 . 0 0 ~ a n d ~ l a t { < } = 9 . 3 6 ~ a n d ~ l a t { > } 7 . 2 8 ~ a n d ~ n d v i > 0 . 3 1 }$   $\mathrm { m s l p } < = 1 0 1 1 4 0 . 8 3 \mathrm { ~ a n d } \mathrm { l a t } < = 9 . 3 6 \mathrm { ~ a n d } \mathrm { l a t } > 7 . 2 8 \mathrm { ~ a n d } \mathrm { l o n } > 2 2 . 7 2$   $\mathrm { m s l p } < = 1 0 1 1 5 7 . 6 9 \ \mathrm { a n d } \ \mathrm { l a t } > 3 . 1 2 \ \mathrm { a n d } \ \mathrm { l o n } > 2 6 . 8 8 \ \mathrm { a n d } \ \mathrm { n d v i } > 0 . 3 0$ </td><td> $\mathbf { m s l p < } \mathbf { = } \mathbf { q 1 }$  ndvi &gt; q2</td></tr><tr><td>1222</td><td>0.99</td><td>Low l</td><td> $\overline { { \mathrm { l a t } < = 1 0 . 0 0 \mathrm { a n d } \mathrm { l a t } > 7 . 2 8 \mathrm { a n d } \mathrm { l o n } < = 0 . 0 0 } }$   $\mathrm { l a t } < = 1 0 . 0 0 \mathrm { a n d } \mathrm { l o n } < = 0 . 0 0 \mathrm { a n d } \mathrm { s w v l l } > 0 . 0 8$   $\mathrm { \ t \_ d a y > 2 9 9 . 4 4 ~ a n d ~ l a t < } = 1 0 . 0 0 ~ \mathrm { a n d ~ l a t > 7 . 2 8 }$   $\mathrm { { l a t } < = - 1 0 . 0 0 \mathrm { { a n d } \ l a t > - 1 1 . 6 8 \mathrm { { a n d } \ s w v l l > 0 . 0 8 } } }$ </td><td>swvl1 &gt; q0  $\mathrm { l s t \_ d a y > q 2 }$ </td></tr><tr><td>1547</td><td>0.96</td><td>Low</td><td> $\mathrm { { l a t } < = - 1 0 . 0 0 ~ a n d ~ l a t > - 1 1 . 6 8 ~ a n d ~ s s t } < = 1 1 6 . 9 5$   $\mathrm { { l a t } < = - 1 0 . 0 0 ~ a n d ~ l a t > - 1 1 . 6 8 ~ a n d ~ p o p \_ d e n s > - 0 . 1 6 }$ </td><td>swvl1 &gt; q0 sst &lt;= q0  $\mathrm { p o p \_ d e n s } > \mathrm { N a N }$ </td></tr><tr><td>944</td><td>0.93</td><td>Low</td><td> $\mathrm { { l a t } < = - 4 . 1 6 ~ a n d ~ l a t > - 8 . 3 2 ~ a n d ~ p o p \_ d e n s > 0 . 4 0 }$  1at &lt;= -4.16 and lat &gt; -8.32 and sst &lt;= 74.93  $\mathrm { l o n < = 1 5 . 2 0 \ a n d \ l o n > 1 3 . 1 2 }$ </td><td>pop_dens &gt; q1  $\mathbf { s s t } < = \mathbf { q 0 }$ </td></tr></table>

Table 9: Description of Unknown SAE-Xplainer with in bold the rule condition to further analyze.

![](images/30f36d7e5e84c52b63e714a24db5a9010f7975b845dd80562756d150464004a6.jpg)  
(a) Histogram of most used conditions.

![](images/0ea0cd904b7539d44dbe31ba930224febbd227571e44b683c3241b6295379ed8.jpg)  
(b) Co-occurrence matrix of high fire risk categories.

Figure 10: (a): Top 10 most used conditions (ignoring latitude and longitude) across the fire-specific SAE-Xplainers. The 20% quantiles are denoted: q0, q1, q2, q3, and q4, and so ndvi $> \ \mathsf { q } 2$ means that NDVI is strictly superior to a threshold in the third 20% quantile. (b): Co-occurrence matrix of the categories assigned to top-50 GeoTopK features. This matrix only records co-occurrence from distinct GeoTopK features, as they can be assigned to multiple categories.

![](images/8886630f36a530c91271568d5041918fdee57b31e0e52658bfa97dbfdebaae0a.jpg)

![](images/bfb7dfcd2ace23572ee399afe876d129f2a25974e167415d64b5cb2f52fcbafc.jpg)

![](images/fe1eea6bc50a55b7be08394fc558b0ca2ddddf63f6e5d9f378a1cada48a0d934.jpg)

![](images/121261ab6efd23e917857c4b727135bc3fbaa9a3ef26c32541ed1fd9333eabe5.jpg)

![](images/772b3caa65c532412f70a40331065e1b41433b06ebcd55a6003b759e01586865.jpg)

![](images/1eb40f699c1c2fbe745721bc3c24e6ded0ad69a7d5c0d962b3b4e16da788b5c7.jpg)

![](images/8a4f3ed67bea86693a96ea48ac3afa3853740a803ace24f2e3638fbc34e0e1ac.jpg)

![](images/b721c5b36ae351bb336a4df11d9b3c87a9a214ddeaa48c2ac3e46a1a16558fe5.jpg)

![](images/205369d3b0c28ab8f6f2753d16f77dd6b4e39af8cab7e4d87564dd32bf83f95e.jpg)

![](images/cbae1f01f3444cda565c988b1d71f0a41e3147380a42bbe53fb86abf303896f1.jpg)  
Figure 11: Sample-level feature interpretability for a ClimaX feature fine-tuned on SeasFire. (Row 1): Feature activations and model prediction masks (no fires regions are masked). (Rows 2-3): SAE-Xplainer rule visualizations with binary masks (unsatisfied rule regions are masked) and input variables highlighting the rule support.

![](images/b8381960ac7966b3fa54c58ddf7cb0d7abc9f711fc58b31fec8d34f6d08d2e63.jpg)  
Figure 12: Sample-level feature interpretability for a ClimaX feature fine-tuned on SeasFire. (Row 1): Feature activations and model prediction masks (No Fire regions are masked). (Rows 2–3): SAE-Xplainer rule visualizations with binary masks (unsatisfied rule regions are masked) and input variables highlighting the rule support.

PSL & Analyzed Window  
![](images/98c6b6d82d4827684876283588046f3c4fe57db2e484682e285fdda138a7c85b.jpg)

![](images/befb503ca206d25bdd42f0c2fec2de19647a27b3f17f424c129eeb03c83e653f.jpg)

![](images/1918bd1729312172f768d2f9795acdebf862c8a5b93b25dfe7808cbc24949e62.jpg)  
Rule 1 : TMQ > 22.72 and V850 <= -4.54 and lat <= -28.27

![](images/6a28d13215041bf64f92e9f77d87c03e348e43ce2aacdb11337c6ab29bd0e767.jpg)

![](images/1a4fc1fe4affb9a3d3ef7ad9bb1582946af5bfefa31fd106256a6bb2c32b6d20.jpg)

![](images/0fd783c509d235f72dcfad6cfac2646b1a89291cefc7a282ae1756b13791e5ae.jpg)

![](images/4fddd16a7fac8f26d2857cb6e55818e915269d0349d4f745aacdc96c373cdec7.jpg)  
Rule 2 : V850 <= -3.97 and lon <= -98.69 and lon > -146.32

![](images/0117a9606e27c79cc159b1aae912667eb5f1603086beac86857e72234f4201e2.jpg)

![](images/f0d0b8bc7c5cd7e646640289e8d13e13a92c9b1fa60a2dc217d139906f034676.jpg)

![](images/4a9c06f042c89f703a64c5811412905db831f00038bbe81df813ba621320b730.jpg)  
Figure 13: Sample-level feature interpretability for a ClimaX feature fine-tuned on ClimateNet and linked to Atmospheric Rivers. (Row 1): Feature activations, model prediction masks (No Event regions are masked), and high feature activation window of analysis. (Rows 2-3): SAE-Xplainer rule visualizations with binary masks (unsatisfied rule regions are masked) and input variables highlighting the rule support.

![](images/5031b024d7e51e4cd1ac3de74d79a83e9dde846d5c01dd6ebb04ec2bc0508e5e.jpg)

![](images/a1a54e61456ea0844f4ffa7f286c1675965c27032c11eba4b0227d857b87dc80.jpg)

![](images/00f89f422a0ad6fb013bc4b390629036595a3bd958e2a7ae372ee904cd565b0f.jpg)

![](images/b6b91e02de00c4cd2142a423b9a9ee00c4d4ea70e18f6a0a7138d1dd7e8abb35.jpg)

![](images/ef2d0d6880422f803380e51180e904f9bd10b63cb8d21c0c6dfdd4c549ad9e8e.jpg)

![](images/c0d08c180b16ed8374a36de06a56b212b4f38f0d688199efb156036679a8b37b.jpg)

![](images/32288548bb27521f1f4d8250f773a61df77cfe16d77cb7055e19837ed1a1bfd1.jpg)

![](images/5e34118d3027ead70d5af7966f5af90737ad2f2a31ddb50810aa56f74656cf5a.jpg)

![](images/3880d1c006ff2364c8310f2bab1a1b4744ed54afa97d10aa9d9f81e7bea43dc8.jpg)

![](images/505d5c9dbbf04fb8c98a3dc0e2bcac201449888699bb3d88249e93295e3eccb2.jpg)

![](images/327bacf857b9a149e56129a379d7f3f9df3ae7036d0ced9daeae9b33910cd5ba.jpg)  
Figure 14: Sample-level feature interpretability for a ClimaX feature fine-tuned on ClimateNet and linked to Atmospheric Rivers. (Row 1): Feature activations, model prediction masks (No Event regions are masked), and high feature activation window of analysis. (Rows 2-3): SAE-Xplainer rule visualizations with binary masks (unsatisfied rule regions are masked) and input variables highlighting the rule support.

![](images/52588ee3b9ed6d32190c62c704142452b5898c18c9832b1c5f8b2cf6a2b01dc7.jpg)

![](images/b38379652e67f4b8c44e362d113268868cfb9a4cbc6df2a85040bbeec5a5b67c.jpg)  
Figure 15: (Left): Example of identified feature group with absorption for ClimateNet. (Right): Co-occurrence matrix for those features computed over the whole dataset.

![](images/f603fd5f274f60096d75c5d173e139649f84116e77ff49d354eb9c787311a130.jpg)

![](images/a24caac5696807277bdd12e9d406eec23f68a73f8e5c3817cc88925304ead37a.jpg)  
Figure 16: (Left): Example of identified feature group with absorption for SeasFire. (Right): Co-occurrence matrix for those features computed over the whole dataset.