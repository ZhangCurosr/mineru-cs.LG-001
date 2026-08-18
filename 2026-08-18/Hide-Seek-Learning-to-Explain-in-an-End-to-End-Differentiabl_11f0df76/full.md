# Hide&Seek: Learning to Explain in an End-to-End Differentiable Network

Tal Ellinson <sup>1</sup> Hadi Mohasel Afshar <sup>1</sup> Sally Cripps <sup>1</sup>

## Abstract

Instance-wise feature selection is a valuable too for interpreting labeled data and the predictions of black-box models. In contrast to global feature selection techniques, instance-wise methods dynamically identify important features for each instance. A growing number of methods learn a selector, which identifies important features, and a predictor, which uses these to make predictions. However, these pioneering methods face challenges including information leakage and lack of differentiability, which can slow training. In this paper, we present Hide&Seek, an end-to-end differentiable model for instance-wise feature selection. We jointly learn feature selection and prediction under a single objective without information leakage. Hide&Seek outperforms existing state-of-the-art models across a range of experiments and is fast to train. We achieve this by reformulating feature removal as a differentiable operation where instead of discretely removing features, we replace a proportion of each feature. Training is further stabilized via a parsimony-weight annealing framework.

## 1. Introduction and Context

Feature selection (FS) techniques help identify informative variables in input-output systems. They are used widely to interpret labeled datasets and explain the predictions of black-box models. For example, in genomics, FS has been used to analyze high-dimensional gene expression data and identify genes relevant to cancer diagnosis (Unger & Kather, 2024). Black-box models are now being used across domains from medical research (Kogan et al., 2020) to finance (Rudin & Shaposhnik, 2023) to the education sector (Levy et al., 2021). As public understanding of model risk rises, decision makers increasingly require reasons behind model predictions. By identifying the important input variables, FS helps peer inside a black box and present a reason for a prediction, which can be sense-checked against a policymaker’s expertise. Beyond interpretability, it can simplify models and reduce overfitting by removing unnecessary features. Thus, FS is a key driver of scientific discovery, increased transparency and improved model performance.

There is an established literature on FS, which spans traditional (Liu & Setiono, 1996; Kohavi & John, 1997; Guyon & Elisseeff, 2003) and deep-learning (Covert et al., 2021; Xu & Yang, 2025) methods. Historically, techniques focused on global importance, identifying features with high predictive contribution across a whole dataset (Tibshirani, 1996; Louppe et al., 2013). However, there are limitations in such an approach. For example, diagnostic pathways can differ between patients and educational policies might be more or less effective for different student subpopulations. Recent years have given rise to algorithms for instance-wise feature selection (IWFS), which ascribe feature importance for each instance (e.g. patient or student) in a dataset.

A common approach of IWFS algorithms is to remove features from a prediction model and thereby quantify their importance (Covert et al., 2021). A pioneering method is LIME (Ribeiro et al., 2016), which perturbs features in the neighborhood of the input and fits a linear surrogate model to explain the prediction. SHAP (Lundberg & Lee, 2017) uses a game-theoretic framework to estimate the average marginal contribution of a feature across all feature permutations. These are both additivefeature attribution methods, where predictions are decomposed into contributions from individual features.

A range of approaches exist and their classifications are not always mutually exclusive. Saliency methods include Integrated Gradients and DeepLIFT, which measure change in the output relative to a baseline for each input feature (Sundararajan et al., 2017; Shrikumar et al., 2017). Masking techniques include Dabkowski & Gal (2017) for image classification and Crabbé & Van Der Schaar (2021) for timeseries analysis. Other work leverages attention mechanisms (Arik & Pfister, 2021; Choi et al., 2016) and causal framings (Schwab & Karlen, 2019). See Covert et al. (2021) for a useful taxonomy.

Table 1. Comparison of similar instance-wise feature selection models. All four models use a selector–predictor framework.
<table><tr><td colspan="2">Hide&amp;Seek</td><td>REAL-X</td><td>INVASE</td><td>L2X</td></tr><tr><td>Adaptive number of important features</td><td> $\checkmark$ </td><td>√</td><td>√</td><td>×</td></tr><tr><td>End-to-end differentiable</td><td>√</td><td>X</td><td>X</td><td> $\checkmark$ </td></tr><tr><td>Avoids information leakage</td><td>√</td><td>√</td><td>×</td><td>×</td></tr></table>

Related Work. The class of algorithms most closely related to our work train a selector, which identifies important features, and a predictor, which uses the chosen features to make predictions. These methods include L2X (Chen et al., 2018), INVASE (Yoon et al., 2018) and REAL-x (Jethani et al., 2021).<sup>1</sup> Let (X, Y ) denote the input features and corresponding output labels. L2X aims to maximize the mutual information between $\mathbf { X } _ { S }$ , a subset of important features, and Y. Conversely, INVASE and REAL-x seek to minimize the KL divergence between the distributions p(Y | X) and $p ( Y \mid \mathbf { X } _ { S } )$ . In practice, these methods approximate conditioning on $\mathbf { X } _ { S }$ by using an ablated version of the full input vector.

In all three methods, the selector learns probabilities for feature inclusion, which are used to sample features sent to the predictor. This discrete sampling is a non-differentiable operation and presents a challenge that each technique handles separately. L2X uses a Gumbel–Softmax relaxation for subset sampling (Maddison et al., 2016; Jang et al., 2017). This makes their architecture end-to-end differentiable but introduces a key limitation: the number of important features must be decided in advance and must be the same for all instances. INVASE uses an actor–critic methodology from reinforcement learning (Peters & Schaal, 2008) to bypass the need for differentiability. This produces good results but makes the model slow to train. REAL-x employs REBAR gradients to approximate the discrete distribution with its continuous relaxation (Tucker et al., 2017).

Information Leakage. A challenge faced by many ablation-based explanation methods, including L2X (Chen et al., 2018) and INVASE (Yoon et al., 2018), is information leakage (Jethani et al., 2021). In jointly trained selector–predictor models, this issue can arise when unselected features are replaced with an ablated value, typically zero (Chen et al., 2018; Yoon et al., 2018; Jethani et al., 2021), or with a mean feature value (Imrie et al., 2022). The predictor may learn to exploit the resulting ablation pattern, rather than relying only on the values of the selected features. In this paper, we study this failure mode in the presence of switch features. A feature is a switch feature when its induced partition, rather than its exact value, determines the response. This can occur directly, or indirectly by governing the importance of other features.

As an example, consider a switch feature $X _ { 1 } .$ , temperature, which influences ${ \mathit { Y } } ,$ cafe profit, through the other features. When $X _ { 1 } < 2 0 ^ { \circ } \mathrm { C } ,$ , let $Y$ be a function of coffee sales and hot chocolate sales. When $X _ { 1 } \geq 2 0 ^ { \circ } \mathrm { C } _ { }$ , let $Y$ be a function of coffee sales, iced coffee sales and iced tea sales. In all instances, $X _ { 1 }$ is important. However, a jointly trained selector and predictor network whose goal is to minimize the number of important features can encode $X _ { 1 } = 0$ as a signal for the partition $X _ { 1 } \geq 2 0 ^ { \circ } \mathrm { C }$ . This allows the joint system to ablate $X _ { 1 }$ to 0 for all instances of warm (or cold) weather, while preserving prediction accuracy. The end result is that $X _ { 1 }$ is not identified as important in a significant number of instances. Leakage can also occur when a switch feature impacts $Y$ directly. Consider an alternative scenario where a switch feature $X _ { 1 }$ , temperature, directly determines $Y ,$ ice cream sales. Say that when $X _ { 1 } \geq 2 5 ^ { \circ } \mathrm { C } ,$ , sales are made and when $X _ { 1 } < 2 5 ^ { \circ } \mathrm { C } ,$ , they are not. As above, information leakage can occur by ablating $X _ { 1 }$ to 0 for all instances of warm (or cold) weather.

Jethani et al. (2021) were the first to identify some cases of information leakage arising from the joint training of selector–predictor networks. They address it with REAL-x by decoupling training: the predictor is first trained independently to predict Y from randomly ablated feature vectors. The selector network is then optimized to provide a minimal subset of important features that preserves prediction accuracy. Because of this, REAL-x avoids information leakage. However, the disjoint training of the predictor network creates two challenges: training the full model is not endto-end differentiable and the predictor must be sufficiently expressive to predict optimally under arbitrary subsets of features. This is difficult as the number of possible selected features grows exponentially with the dimensionality of the data. The challenges faced by these related methods are summarized in Table 1 and motivate our model.

We prove information leakage in Appendix A and derive a lower bound on the achievable misidentification rate in Corollary A.2. Appendix A.1 compares our treatment of information leakage to that in Jethani et al. (2021).

Contributions. In this paper, we make four contributions: (1) We introduce Hide&Seek, a novel method for instancewise feature selection. The selector and predictor are jointly trained in an end-to-end differentiable framework. We achieve differentiability by replacing a proportion of each feature, instead of discretely removing a feature subset. The model is fast to train and performs strongly in a range of experiments.

![](images/7adce5455318934f0805c43cd23bb45eb3eeed19c33becedaa24a2756a9124a5.jpg)  
Figure 1. Hide&Seek architecture.

(2) We provide a mechanism for preventing information leakage while preserving joint training. Instead of replacing unselected features with a fixed ablation value, Hide&Seek uses random draws from the feature distribution. This prevents the predictor from exploiting the ablation as an extra source of information.

(3) We stabilize the joint training of the selector and predictor modules with a parsimony-weight annealing schedule, which markedly improves performance. The schedule allows the model to focus on predictive performance early in training and increase parsimony in the later stages.

(4) We study information leakage in a setting more general than that previously formalized (Jethani et al., 2021). We show that the typical loss function can be an ill-posed proxy for the intended loss and can favour omitting genuinely important features (Appendix A). We derive a lower bound on the achievable feature-importance misidentification rate (Corollary A.2), and our empirical results show that the observed misidentification rate is often close to this bound in practice.

This paper has the following structure. In Section 2, we formalize the mathematical foundation of our method. In Section 3, we describe the structure of our model. In Section 4, we evaluate Hide&Seek by comparing it against several benchmarks.

## 2. Problem Formulation

Consider a d-dimensional feature space $\mathcal { X } = \mathcal { X } _ { 1 } \times \cdot \cdot \cdot \times \mathcal { X } _ { d }$ paired with an output space Y. Let $\mathbf { x } : = ( x _ { 1 } , \dots , x _ { d } )$ and y be realizations of the random vector $\mathbf { X } : = ( X _ { 1 } , \ldots , X _ { d } ) \in$ X and random variable $Y \in \mathcal { D }$ , respectively. Let $D : =$ $\{ 1 , \ldots , d \}$

In instance-wise feature selection, the aim is to find a minimal subset $S \subseteq D$ per realization x that indexes the important features for predicting the label $y .$ Let the reduced vector containing selected features be $\mathbf { x } _ { S }$ . We seek a minimum |S| satisfying:

$$
p ( y \mid \mathbf { x } _ { S } ) = p ( y \mid \mathbf { x } ) .\tag{1}
$$

However, modeling $p ( y \mid \mathbf { x } _ { S } )$ is non-trivial as the dimensionality of the reduced vector $\mathbf { x } _ { S }$ changes between instances. A common strategy is to approximate $\mathbf { x } _ { S }$ by a fixed-size vector $\mathbf { z } : = ( z _ { 1 } , \ldots , z _ { d } )$ in which

$$
z _ { j } = \left\{ { \begin{array} { l l } { x _ { j } , } & { { \mathrm { i f ~ } } j \in S , } \\ { { \hat { x } } _ { j } , } & { { \mathrm { i f ~ } } j \notin S . } \end{array} } \right.\tag{2}
$$

where $j \in \{ 1 , \ldots , d \}$ and $\hat { \mathbf { x } } : = ( \hat { x } _ { 1 } , \ldots , \hat { x } _ { d } ) \in \mathbb { R } ^ { d }$ represents uninformative replacement values such as 0 (Chen et al., 2018; Yoon et al., 2018; Tagaris & Stafylopatis, 2020; Jethani et al., 2021) or a vector of feature means (Imrie et al., 2022).

This approach introduces two problems that were discussed earlier. Specifically, removing features with eq. (2) requires discrete feature selection, which is non-differentiable, and ablation to fixed values, ${ \hat { x } } _ { j } ,$ can be exploited by a jointly trained selector and predictor network to produce information leakage. To address these limitations, we reformulate both the selection mechanism and the replacement strategy.

Selection Mechanism. To enable differentiable feature selection, we propose constructing z with a linear combination of the original signal, $x _ { j }$ , and a stochastic replacement value, ${ \hat { x } } _ { j }$ :

$$
z _ { j } = m _ { j } x _ { j } + ( 1 - m _ { j } ) \hat { x } _ { j }\tag{3}
$$

where $m _ { j } ~ \in ~ [ 0 , 1 ]$ and a higher $m _ { j }$ represents a higher likelihood that $j \in S$ . In other words, instead of replacing a proportion of features, we replace a proportion of each feature.

Under this setting, the instance-wise feature selection task is equivalent to finding the most parsimonious real-valued mask vector m $\in [ 0 , 1 ] ^ { d }$ such that

$$
\mathbf { z } = \mathbf { m } \odot \mathbf { x } + ( 1 - \mathbf { m } ) \odot { \hat { \mathbf { x } } } , \ { \mathrm { a n d } }\tag{4}
$$

$$
p ( y \mid \mathbf { z } ) = p ( y \mid \mathbf { x } )\tag{5}
$$

Replacement Strategy. If the modified signal, $\mathbf { z } ,$ is distributed according to the original feature distribution, i.e., $p ( \mathbf { Z } ) = p ( \mathbf { X } )$ , then the predictor cannot distinguish between the original and replaced features and information leakage becomes impossible. Consider the case where m $\in \{ 0 , 1 \bar  \} ^ { d }$ is binary. In this context, $\hat { \mathbf { x } } = ( \mathbf { x } _ { S } , \hat { \mathbf { x } } _ { \bar { S } } )$ , where $\bar { S } : = D \backslash S$ No leakage is guaranteed when replacement values are drawn from the conditional distribution:<sup>2</sup>

$$
\hat { \mathbf { x } } _ { \bar { S } } \sim p ( \mathbf { X } _ { \bar { S } } \mid \mathbf { x } _ { S } )\tag{6}
$$

However, accurately modeling and sampling from this conditional distribution is non-trivial. In practice, we sample replacement values from the product of the marginal distributions, akin to unary Quantitative Input Influence (Datta et al., 2016):

$$
{ \hat { \mathbf { x } } } \sim \prod _ { i \in D } p ( X _ { i } )\tag{7}
$$

Note that we write xˆ rather than $\hat { \mathbf { x } } _ { \bar { S } }$ because with our continuous relaxation, all features have a proportion selected (and unselected) via the continuous masks $\mathbf { m } \in [ 0 , 1 ] ^ { d }$

Marginal sampling is efficient to implement and is significantly harder to exploit than fixed-value ablation. For leakage to occur in fixed-value ablation, the predictor network simply needs to learn the significance of a single ablated value. For leakage to occur under marginal sampling, the predictor network must first learn the true joint distribution of the data, and then recognize when stochastic replacement values are out-of-distribution. Our experiments show the absence of leakage under marginal replacement, including in highly correlated settings. In Appendix A.2, we also explore an alternative method to approximate the conditional distribution.

## 2.1. Optimization

As in Yoon et al. (2018), we relax (1) using the KL (Kullback-Leibler) divergence. Let $S : \mathcal { X }  2 ^ { \{ 1 , . . . , d \} }$ be a selector function that maps each input instance x to an instance-specific subset of selected feature indices. For notational convenience, we write $\mathbf { X } _ { S } : = \{ X _ { i } \} _ { i \in \pmb { S } ( \mathbf { x } ) }$ . Minimizing the KL divergence between $p ( Y \mid \mathbf { X } )$ and $\dot { p } ( Y \mid \mathbf { X } _ { S } )$ is equivalent to minimizing the cross-entropy with respect to the selector function $s { \mathrm { : } }$

$$
\operatorname* { m i n } _ { S } \mathbb { E } _ { \mathbf { X } } \left[ \mathrm { K L } \left( p ( Y \mid \mathbf { X } ) \parallel p ( Y \mid \mathbf { X } _ { S } ) \right) \right]
$$

$$
= \operatorname* { m i n } _ { \boldsymbol { s } } \mathbb { E } _ { \mathbf { X } } \mathbb { E } _ { Y | \mathbf { X } } \left[ \log \frac { p ( Y \mid \mathbf { X } ) } { p ( Y \mid \mathbf { X } _ { S } ) } \right]
$$

$$
= \underset { \cal { S } } { \mathrm { m i n } } \mathbb { E } _ { \mathbf { X } } \mathbb { E } _ { Y | \mathbf { X } } \left[ \log p ( Y \mid \mathbf { X } ) - \log p ( Y \mid \mathbf { X } _ { \cal { S } } ) \right]
$$

$$
\begin{array} { r l } & { = \underset { \mathcal { S } } { \mathrm { m i n } } \underbrace { \mathbb { E } _ { \mathbf { X } } \mathbb { E } _ { Y \mid \mathbf { X } } [ \log p ( Y \mid \mathbf { X } ) ] } _ { \mathrm { c o n s t a n t w . r . t . } \mathcal { S } } } \\ & { ~ + \underset { \mathcal { S } } { \mathrm { m i n } } \mathbb { E } _ { \mathbf { X } } \mathbb { E } _ { Y \mid \mathbf { X } } [ - \log p ( Y \mid \mathbf { X } _ { S } ) ] } \\ & { \equiv \underset { \mathcal { S } } { \mathrm { m i n } } \mathbb { E } _ { \mathbf { X } } \mathbb { E } _ { Y \mid \mathbf { X } } [ - \log p ( Y \mid \mathbf { X } _ { S } ) ] . } \end{array}\tag{8}
$$

In practice, the true conditional distribution of $p ( Y \mid \mathbf { X } _ { S } )$ is unknown and is approximated by the model induced distribution $p _ { \theta } ( Y \mid \mathbf { X } _ { S } )$ . This allows us to define the following loss:

$$
\begin{array} { r l } & { \mathcal { L } ( S , \theta ) = \mathbb { E } _ { ( \mathbf { X } , Y ) \sim p } \left[ - \log p _ { \theta } ( Y \mid \mathbf { X } _ { S } ) \right] } \\ & { ~ + ~ \lambda \mathbb { E } _ { \mathbf { X } \sim p _ { X } } [ \| S ( \mathbf { X } ) \| ] } \end{array}\tag{9}
$$

where ∥ · ∥ represents the $l _ { 1 }$ norm, encouraging sparse outputs from S. λ is a regularization hyperparameter that balances minimizing the KL divergence against sparsity. For classification, the negative log-likelihood reduces to the standard cross-entropy loss.

## 3. Proposed Model

Take a dataset $\mathcal { D } = \{ ( \mathbf { x } ^ { ( i ) } , \mathbf { y } ^ { ( i ) } ) \} _ { i = 1 } ^ { n }$ with n i.i.d. samples drawn from the joint distribution $p ( \mathbf { X } , \mathbf { Y } )$ , where $\mathbf { \bar { y } } \in \{ 0 , 1 \} ^ { C }$ represents the one-hot encoding of the target variable.<sup>3</sup> The Hide&Seek architecture consists of two feed-forward fully connected neural network modules: Hide and Seek.<sup>4</sup> As shown in Figure 1, Hide takes an input vector x and finds a continuous mask vector $\mathbf { m } \in [ 0 , 1 ] ^ { d }$ , representing the importance of d features. z is determined by (4) and (7) and Seek maps it to the predicted output $\hat { \mathbf { y } }$ . With these choices, the loss function (9) becomes:

![](images/989e8ea5627d2f863e7380d8238fd37a3e901df1951ab5c292d1beb4b061d26c.jpg)  
(c)

![](images/b0f6425bcb4cfc31be187caa96f55f3f6ceb20b79263c4e688712ca5294ed979.jpg)

![](images/e951a5319998244a242ee630511fed074b7ef385339b6b42a88c3b8e9d898de5.jpg)

(d)  
![](images/1aa7053fe7be39b19866a11cc847ed8936e03f57969181f32846196285b6ab63.jpg)  
Figure 2. The annealing schedule for λ<sub>t</sub> over t epochs, showing the impact of different choices of $q = \{ 0 , 1 , 2 , 3 \}$ in $\begin{array} { r } { \lambda _ { t } = \left( \frac { t } { T } \right) ^ { q } } \end{array}$ λ<sub>max</sub> on the loss function (12). (a) The functional form of $\lambda _ { t } . \ ( \mathfrak { b } )$ The cross-entropy term $- \sum _ { c = 1 } ^ { C }$ y<sub>c</sub> log ˆy<sub>c</sub> vs t. (c) The regularized parsimony term $\frac { \lambda _ { t } } { d } \left\| \mathbf { m } \right\| _ { 1 }$ vs t. (d) The combined loss $\ell ( \alpha , \beta , t )$ vs t. The metrics in $(  { \mathbf { b } } ) \ – (  { \mathbf { d } } )$ are calculated on a hold-out validation set using the same $\lambda _ { \mathrm { m a x } }$ and data as in the Syn4 experiment in section 4.1. See Appendix D.7 for other choices of $\lambda _ { \mathrm { m a x } }$ and more detail.

$$
\ell ( \boldsymbol { \alpha } , \boldsymbol { \beta } ) = \operatorname { \mathbb { E } } _ { ( \mathbf { x } , \mathbf { y } ) \sim p ( \mathbf { X } , \mathbf { Y } ) } \left[ - \sum _ { c = 1 } ^ { C } y _ { c } \log \hat { y } _ { c } + \frac { \lambda } { d } \| \mathbf { m } \| _ { 1 } \right]\tag{10}
$$

$$
\begin{array} { r l } { \mathrm { w h e r e } \ } & { \mathbf { m } = \mathrm { H i d e } _ { \alpha } ( \mathbf { x } ) } \\ & { \mathbf { z } = \mathbf { m } \odot \mathbf { x } + ( 1 - \mathbf { m } ) \odot \hat { \mathbf { x } } } \\ & { \hat { \mathbf { y } } = \mathrm { S e e k } _ { \beta } ( \mathbf { z } ) } \end{array}
$$

In summary, feature importance is achieved without discrete sampling of features to retain or replace. Rather, using continuous masks, we replace a proportion of each feature with marginal noise, enabling end-to-end differentiability and the joint training of our network.

Further details on the model architecture are provided in Appendix B.<sup>5</sup>

## 3.1. Parsimony-Weight Annealing

Optimizing the parameters of a joint network to both minimize cross-entropy and increase parsimony presents a challenge. Specifically, placing excessive emphasis on parsimony early in training can cause the optimization to become trapped in poor local minima. To address this, we introduce an annealing schedule, in line with Bowman et al. (2016)

and Tagaris & Stafylopatis (2020). In our method, we progressively increase the weight on our parsimony term by quadratically growing the regularization parameter towards a set maximum value. Let $\lambda _ { \mathrm { m a x } }$ be a fixed hyperparameter, and $t \in \{ 1 , \ldots , T \}$ index the training over $T$ epochs. We replace λ in (10) with $\lambda _ { t }$ where:

$$
\lambda _ { t } = \left( \frac { t } { T } \right) ^ { 2 } \lambda _ { \operatorname* { m a x } }\tag{11}
$$

and the final loss becomes:

$$
\ell ( \boldsymbol { \alpha } , \boldsymbol { \beta } , t ) = \mathbb { E } _ { ( \mathbf { x } , \mathbf { y } ) \sim p ( \mathbf { X } , \mathbf { Y } ) } \left[ - \sum _ { c = 1 } ^ { C } y _ { c } \log \hat { y } _ { c } + \frac { \lambda _ { t } } { d } \| \mathbf { m } \| _ { 1 } \right]\tag{12}
$$

As shown in Figure 2, this schedule allows the model to prioritize prediction accuracy early in training and focus on mask parsimony in later stages, leading to a lower final loss. Experimentally, this led to strong performance in the feature importance metrics. See Appendix D.7 for more detail.

## 4. Experiments

We present five analyses. The Synthetic data experiment evaluates Hide&Seek’s ability to recover ground-truth feature importance against existing benchmarks. Switch analysis compares model performance in identifying switchfeatures, that is, features whose induced partition, rather than their exact feature value, affects the response. Credit default data measures Hide&Seek’s ability to recover ground truth feature importance from highly correlated features, as well as comparing the predictive power of the model in semi-synthetic and real-world settings. The fourth experiment presents a Hide&Seek explanation for MNIST images. The final experiment, Breast cancer subtype classification applies Hide&Seek to genetic microarray data.

## 4.1. Synthetic Data

Our first experiment uses the same synthetic datasets as in Yoon et al. (2018), Chen et al. (2018) and Arik & Pfister (2021). The output Y is sampled from a Bernoulli distribution, where $\begin{array} { r } { P ( \dot { Y } = 1 | U ) = \frac { 1 } { 1 + e ^ { U } } } \end{array}$ . U is a function of 11 Gaussian iid variables $X _ { 1 } , . . . , \bar { X _ { 1 1 } }$ , where $X _ { j } \sim \mathcal { N } ( 0 , 1 )$ There are six settings.

$$
\mathbf { \partial } \cdot \mathbf { S y n 1 } \colon U = X _ { 1 } X _ { 2 }
$$

$$
\textstyle \bullet \ : \mathbf { S y n 2 } \colon U = \sum _ { i = 3 } ^ { 6 } X _ { i } ^ { 2 } - 4
$$

$$
\bullet \ \mathbf { S y n 3 : } \ U = - 1 0 \sin ( 0 . 2 X _ { 7 } ) + \left| X _ { 8 } \right| + X _ { 9 } + e ^ { - X _ { 1 0 } } - 2 . 4
$$

• Syn4: U: if $X _ { 1 1 } < 0$ , follow Syn1, else Syn2

Table 2. Performance of eight algorithms across six datasets. Each TPR and FDR metric is the median of 20 experiments. See Appendix C for boxplots.
<table><tr><td rowspan="2">Model</td><td colspan="2">Hide&amp;Seek</td><td colspan="2">INVASE</td><td colspan="2">REAL-X</td><td colspan="2">SHAP</td><td colspan="2">LIME</td><td colspan="2">L2X</td><td colspan="2">RForest</td><td colspan="2">LASSO</td></tr><tr><td>TPR</td><td>FDR</td><td>TPR</td><td>FDR</td><td>TPR</td><td>FDR</td><td>TPR</td><td>FDR</td><td>TPR</td><td>FDR</td><td>TPR</td><td>FDR</td><td>TPR FDR</td><td>TPR</td><td>FDR</td></tr><tr><td>Syn1</td><td>100</td><td>0</td><td>100</td><td>0</td><td>100</td><td>9</td><td>64</td><td>36</td><td>19</td><td>81</td><td>30</td><td>70</td><td>100 0</td><td>0</td><td>100</td></tr><tr><td>Syn2</td><td>100</td><td>0</td><td>100</td><td>0</td><td>100 4</td><td>95</td><td>5</td><td>100</td><td>0</td><td>75</td><td>25</td><td>100</td><td>0</td><td>50</td><td>50</td></tr><tr><td>Syn3</td><td>99</td><td>0</td><td>91</td><td>0</td><td>82 1</td><td>92</td><td>8</td><td>96</td><td>4</td><td>43</td><td>57</td><td>100</td><td>0</td><td>75</td><td>25</td></tr><tr><td>Syn4</td><td>99</td><td>4</td><td>90</td><td>4</td><td>98 15</td><td>72</td><td>38</td><td>54</td><td>51</td><td>44</td><td>65</td><td>67</td><td>40</td><td>58</td><td>55</td></tr><tr><td>Syn5</td><td>97</td><td>3</td><td>83</td><td>1</td><td>88 9</td><td>74</td><td>38</td><td>50</td><td>54</td><td>53</td><td>57</td><td>67</td><td>40</td><td>56</td><td>50</td></tr><tr><td>Syn6</td><td>98</td><td>4</td><td>90</td><td>9</td><td>91 4</td><td>72</td><td>28</td><td>51</td><td>49</td><td>53</td><td>47</td><td>60</td><td>40</td><td>60</td><td>40</td></tr></table>

• Syn5: U: if $X _ { 1 1 } < 0$ , follow Syn1, else Syn3

• Syn6: U: if $X _ { 1 1 } < 0 .$ , follow Syn2, else Syn3

The goal is to identify the specific features used in generating Y. Syn1–3 represent global feature importance problems. Syn4–6 use a switch feature, $X _ { 1 1 }$ , to create instance-wise feature importance settings. We evaluate each algorithm’s success using True Positive Rate (TPR) and False Discovery Rate (FDR), as in (Yoon et al., 2018).

We compare eight algorithms. Six are instance-wise algorithms: Hide&Seek, INVASE (Yoon et al., 2018), REAL-x (Jethani et al., 2021), SHAP (Lundberg & Lee, 2017), LIME (Ribeiro et al., 2016) and L2X (Chen et al., 2018). Two provide global feature importance: LASSO (Tibshirani, 1996) and Random Forest (Breiman, 2001). We train on 10,000 samples and test on 10,000 samples.

Hide&Seek outperforms the other models on instance-wise feature selection (IWFS) metrics, as shown in Table 2. Its end-to-end differentiability makes it easy to train, with model run times shown in Table 3. INVASE and REAL-x have good performance, but significantly longer training times due to more complex architectures. INVASE and L2X use different activation functions for different datasets (ReLU for Syn1-2 and SELU for Syn4–6) (Yoon et al., 2018; Chen et al., 2018). In Hide&Seek, the activation function (ReLU) is constant across all datasets.

Hide&Seek, INVASE and REAL-x have a natural mechanism for determining whether a feature is important, namely that its mask is greater than 0.5. This allows the number of important features to be automatically chosen for each instance. In all other methods, the number of important features, k, needs to be specified. In the experiments, k is chosen based on the number of ground truth important features sought for each dataset. Specifically, k = 2 for Syn1, k = 4 for Syn2–3 and k = 5 for Syn4–6. This can overestimate the FDR for Syn4 and Syn5, which have 5 important features when $X _ { 1 1 } \geq 0$ but only 3 important features when $X _ { 1 1 } < 0$ . To account for this, SHAP, LIME and L2X results for k = 3 and k = 4 are shown in Appendix D.4.

The distributions for the masks for each of the synthetic datasets are provided in Appendix D.1. Each model’s predictive performance is shown in Appendix D.2. Hide&Seek shows a higher predictive performance than other selector– predictor models despite having 17× (REAL-x, L2X) to 30× (INVASE) fewer parameters.

## 4.2. Switch Analysis

Corollary A.2 shows that, under the ill-posed loss proxy, a switch feature can be ablated, and hence misidentified as unimportant, at an achievable rate at least as large as the probability of its largest partition cell. In Syn4–6, the switch feature is $X _ { 1 1 }$ . Since $X _ { 1 1 } \sim \mathcal { N } ( 0 , 1 )$ , the partition cells $X _ { 1 1 } < 0$ and $X _ { 1 1 } \geq 0$ have equal probability, and the corollary therefore gives an achievable misidentification rate of 50%. Intuitively, a jointly trained selector–predictor network that ablates unselected features to zero can use the artificial value $X _ { 1 1 } = 0$ as a code for one of the two partition cells. The end result is that $X _ { 1 1 }$ is not identified as important 50% of the time, without degrading prediction accuracy.

To test for this information leak, we analyze each model’s ability to correctly identify $X _ { 1 1 }$ as an important feature. As in Section 4.1, we run each of Syn4–6 20 times. We then test how often each model correctly identifies the switchfeature, $X _ { 1 1 }$ . Across Syn4–6, REAL-x has perfect switch accuracy, due to its disjoint training of the predictor network. The median switch accuracy of Hide&Seek is also 100% for Syn4–6, significantly higher than INVASE (≈ 51%)

Table 3. Typical run times of selector–predictor models for the synthetic data, with IWFS metrics reported in table 2. Times include training (10,000 samples), prediction, and feature importance attribution (10,000 samples). See Appendix B.1 for hardware and Appendix D.5 for more detail.
<table><tr><td>Method</td><td>Run time (hh:mm:ss)</td></tr><tr><td>L2X</td><td>00:00:03</td></tr><tr><td>Hide&amp;Seek</td><td>00:00:05</td></tr><tr><td>REAL-X</td><td>00:01:16</td></tr><tr><td>INVASE</td><td>01:18:52</td></tr></table>

![](images/c99ddbd660d4a0a42aa87bccb696e7e194451e19d3f52d584503bf392a59f967.jpg)  
Figure 3. Switch accuracy. The percentage of instances where the switch-feature, $X _ { 1 1 }$ , was correctly identified as important. Each boxplot represents the distribution across 20 runs.

and L2X (≈ 58%). These results in Figure 3 empirically confirm the benefit of our feature replacement strategy.

To assess the impact of this information leakage, we analyzed the INVASE results from the experiment in Section 4.1. We manually corrected failures to identify the switch feature, $X _ { 1 1 }$ , in Syn4–6 and recomputed the results. If $X _ { 1 1 }$ were correctly identified, INVASE’s TPR would rise to 99.9, 99.8, and 99.9 for Syn4, Syn5, and Syn6, respectively. Note that FDR would be unchanged as $X _ { 1 1 }$ is always important. In other words, INVASE’s TPR gap in Table 2 could be closed entirely by correctly identifying the switch feature.

## 4.3. Correlated Data

This experiment investigates the robustness of Hide&Seek under multicollinearity. We generate features and the target as in section 4.1, except that instead of drawing each feature independently from $X _ { j } \sim \mathcal { N } ( 0 , 1 )$ , we draw from a multivariate normal distribution $\mathbf { X } \sim { \mathcal { N } } ( \mathbf { 0 } , { \boldsymbol { \Sigma } } )$ . The covariance matrix Σ is defined using a constant pairwise correlation parameter $\rho ,$ such that $\Sigma _ { i , j } = \rho$ for $i \neq j$ and $\Sigma _ { i , j } = 1$ for $i = j$ . Thus, $\rho$ controls feature dependence.

We ran each model on Syn1-6, under this correlated setting, for each of $\rho \in \{ 0 , 0 . 1 , 0 . 2 , \ldots , 0 . 8 , 0 . 9 \}$ . The results in Figure 4 show that Hide&Seek consistently outperforms the other models at instance-wise feature selection under multicollinearity.

Hide&Seek also maintained a switch accuracy greater than 99.4% (averaged across Syn4–6) for all choices of $\left| \rho . \right.$ This indicates the absence of leakage even when the switch feature has high pairwise correlations. Switch feature accuracies for each $\rho$ and model are shown in Appendix A.4.

![](images/5a467a9b44330e9f3d1028cceedd7c13e898856c7453eb17d9e869c67fbf8fab.jpg)  
Figure 4. Model performance under the multicollinear setting. Each point represents the average IWFS F1 score across Syn1-6 for a given model and a given value of $\rho .$ For these synthetic datasets, $\mathbf { X } \sim { \mathcal { N } } ( \mathbf { 0 } , { \boldsymbol { \Sigma } } )$ , where $\Sigma _ { i , j } = \rho$ for $i \neq j$ and $\begin{array} { r } { \dot { \Sigma } _ { i , j } = 1 } \end{array}$ for $i = j$

## 4.4. Credit Default Data

In this experiment, we evaluate the strongest selector– predictor models against correlated real-world features in a semi-synthetic IWFS setting. We also report on their predictive power.

The Default ofCredit Card Clients dataset contains 30,000 instances and 23 features describing demographic and financial attributes of credit card holders (Yeh & Lien, 2009). We generate a binary label synthetically to establish a groundtruth feature importance against which to evaluate our models, using the same functional forms for Syn4-Syn6 as in section 4.1. For $\{ X _ { 1 } , \ldots , X _ { 1 0 } \}$ we use highly correlated features including customer bill and payment amounts across consecutive months and set AGE as the switch feature, $X _ { 1 1 }$ Appendix F.1 presents the full list of features and the correlation matrix. Table 4 shows the instance-wise feature selection results.

While the primary goal of these models is to identify important instance-wise features, each model’s predictive power may be of interest. A significantly low prediction rate could indicate an upper bound on a model’s feature selection ability. We evaluated the predictive performance on the Credit Default data in each of the semi-synthetic settings and in predicting the true Y values, which indicate whether or not a customer defaulted on their payment. The results are shown in Table 5.

Table 4. IWFS performance on highly correlated credit default data, where the target is generated as in Section 4.1. Each metric is the median of 20 experiments.
<table><tr><td rowspan="2">Model</td><td colspan="3">Hide&amp;Seek</td><td colspan="3">INVASE</td><td colspan="3">REAL-X</td></tr><tr><td>TPR</td><td>FDR</td><td>F1</td><td>TPR</td><td>FDR</td><td>F1</td><td>TPR</td><td>FDR</td><td>F1</td></tr><tr><td rowspan="5">Syn1 Syn2 Syn3 Syn4</td><td>100</td><td>1</td><td>99</td><td>100</td><td>0</td><td>100</td><td>100</td><td>22</td><td>86</td></tr><tr><td>67</td><td>26</td><td>67</td><td>100</td><td>0</td><td>100</td><td>49</td><td>13</td><td>59</td></tr><tr><td>100</td><td>0</td><td>100</td><td>98</td><td>0</td><td>99</td><td>63</td><td>11</td><td>69</td></tr><tr><td>87</td><td>34</td><td>74</td><td>58</td><td>31</td><td>57</td><td>63</td><td>35</td><td>62</td></tr><tr><td>96</td><td>28</td><td>81</td><td>80</td><td>28</td><td>74</td><td>79</td><td>41</td><td>66</td></tr><tr><td>Syn5 Syn6</td><td>87</td><td>39</td><td>71</td><td>73</td><td>31</td><td>62</td><td>48</td><td>38</td><td>52</td></tr></table>

Table 5. Prediction performance for highly correlated credit default data with synthetically generated labels (Syn1-6) and the credit default binary Y labels (CDY) that indicate whether or not a customer defaulted on their payment. The metric is mean AUROC (± standard error) across 20 runs.
<table><tr><td></td><td colspan="3">AUROC</td></tr><tr><td>Model</td><td>Hide&amp;Seek</td><td>INVASE</td><td>REAL-X</td></tr><tr><td>Syn1</td><td> $\mathbf { 0 . 6 6 4 \pm 0 . 0 0 2 }$ </td><td> $\overline { { 0 . 6 0 1 \pm 0 . 0 0 2 } }$ </td><td> $\overline { { 0 . 6 4 1 \pm 0 . 0 0 3 } }$ </td></tr><tr><td>Syn2</td><td> $\mathbf { 0 . 8 8 5 \pm 0 . 0 0 2 }$ </td><td> $0 . 8 6 3 \pm 0 . 0 0 2$ </td><td> $0 . 8 7 8 \pm 0 . 0 0 3$ </td></tr><tr><td>Syn3</td><td> ${ \bf 0 . 7 6 7 \pm 0 . 0 0 2 }$ </td><td> $0 . 7 5 2 \pm 0 . 0 0 3$ </td><td> $0 . 7 2 9 \pm 0 . 0 0 5$ </td></tr><tr><td>Syn4</td><td> $\mathbf { 0 . 8 2 8 \pm 0 . 0 0 1 }$ </td><td> $0 . 8 1 5 \pm 0 . 0 0 2$ </td><td> $0 . 8 0 9 \pm 0 . 0 0 2$ </td></tr><tr><td>Syn5</td><td> $\mathbf { 0 . 7 0 0 \pm 0 . 0 0 2 }$ </td><td> $0 . 6 7 5 \pm 0 . 0 0 2$ </td><td> $0 . 6 6 3 \pm 0 . 0 0 3$ </td></tr><tr><td>Syn6</td><td> $\mathbf { 0 . 8 6 7 \pm 0 . 0 0 2 }$ </td><td> $0 . 8 5 8 \pm 0 . 0 0 1$ </td><td> $0 . 8 4 5 \pm 0 . 0 0 2$ </td></tr><tr><td>CDY</td><td> $0 . 7 7 0 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 7 7 1 \pm 0 . 0 0 1 }$ </td><td> $0 . 7 5 6 \pm 0 . 0 0 2$ </td></tr></table>

## 4.5. MNIST

The MNIST dataset contains handwritten digits represented as grayscale images (LeCun et al., 2002). We train on 11,172 images and test on 1,397 images. As in (Chen et al., 2018), we select only the 3s and 8s, with the intention that the minor differences between the two digits could imply feature importance where the ground truth is unknown.

We compare five algorithms: Hide&Seek, INVASE, REALx, SHAP and LIME. To assign feature importance, SHAP and LIME first require a baseline model to interpret. For SHAP, we train XGBoost, a tree-based gradient boosting algorithm (Chen & Guestrin, 2016). For LIME we train an independent neural network. Hide&Seek, INVASE and REAL-x perform predictions by design and we use their predictor networks without adjusting any architecture. In all five cases, we achieve a classification accuracy of 99.1% or greater.

Each image contains $2 8 \times 2 8 = 7 8 4$ pixels. To identify the most important patches, we assign an importance score to each pixel using the given model. We then apply a $3 \times 3$ sliding window over the image, assigning each 3 × 3 patch an aggregated importance score. Figure 5 shows the four most important patches in each image, identified by each model.

These instance-wise explanation models are expected to adapt to each drawing and highlight important pixels in each context. Only Hide&Seek and REAL-x consistently identify the ‘left arcs’ of each digit, which indicate an 8 when present and a 3 when absent. By contrast, LIME fails to identify important patches. INVASE and SHAP display less varied explainability.

![](images/408124f6477459a33f23b628628b41f5b951c35946dc9b5d320ad3ef46f25e4d.jpg)  
Figure 5. Important patches in MNIST data, as identified by each explanation method. Each 3x3 red square is one of the four most important patches in the image.

## 4.6. Breast Cancer Subtype Classification

In this experiment, we use Hide&Seek to analyze breast cancer (BRCA) microarray data from The Cancer Genome Atlas (TCGA) where tumors are categorized into four different molecular subtypes (Berger et al., 2018). We select the same subset of 100 genes (out of 17,814) as in Covert et al. (2021) to reduce overfitting and enable comparison between our experiments. The dataset is relatively small, containing 572 instances. Using a test set of size 100, we rank the genes by mean mask size, thereby aggregating instance-wise importance into a global analysis. As shown in Table 6, the top two genes ESR1 and CCNB2 exhibit significantly higher importance than the remainder. Both ESR1 (Robinson et al., 2013) and CCNB2 (Shubbar et al., 2013) have established associations with BRCA and were also identified as important by Covert et al. (2021). We provide a comparison with other models, for which we updated the code of INVASE, REAL-x and L2X to handle classification with more than two classes. This experiment highlights the potential of Hide&Seek to identify biologically relevant gene associations, including candidates beyond those previously reported in the literature. See Appendix F.3 for further details.

Table 6. Top 10 important genes across different explanation methods. Feature importance represents the mean mask size (Hide&Seek, INVASE, REAL-x), mean importance scores (SHAP, LIME), or mean selection frequency (L2X). Standard errors are provided in the appendix.
<table><tr><td colspan="2">Hide&amp;Seek</td><td colspan="2">INVASE</td><td colspan="2">REAL-X</td><td colspan="2">SHAP</td><td colspan="2">L2X</td><td colspan="2">LIME</td></tr><tr><td>Gene</td><td>Imp.</td><td>Gene</td><td>Imp.</td><td>Gene</td><td>Imp.</td><td>Gene</td><td>Imp.</td><td>Gene</td><td>Imp.</td><td>Gene</td><td>Imp.</td></tr><tr><td>ESR1</td><td>0.999</td><td>CCNB2</td><td>0.723</td><td>ESR1</td><td>0.969</td><td>ESR1</td><td>0.695</td><td>PENK</td><td>0.640</td><td>TUBB</td><td>0.044</td></tr><tr><td>CCNB2</td><td>0.951</td><td>ESR1</td><td>0.711</td><td>CCNB2</td><td>0.897</td><td>CCNB2</td><td>0.402</td><td>BIRC3</td><td>0.600</td><td>HACE1</td><td>0.044</td></tr><tr><td>STATH</td><td>0.827</td><td>ZNF775</td><td>0.680</td><td>NUP210</td><td>0.833</td><td>C6orf15</td><td>0.212</td><td>TMEM52</td><td>0.590</td><td>C6orf26</td><td>0.038</td></tr><tr><td>C6orf26</td><td>0.804</td><td>KLF3</td><td>0.619</td><td>C6orf15</td><td>0.758</td><td>ZNF385</td><td>0.126</td><td>HPS4</td><td>0.590</td><td>PENK</td><td>0.033</td></tr><tr><td>TUBB</td><td>0.784</td><td>C6orf15</td><td>0.610</td><td>SLC25A3</td><td>0.754</td><td>NUP210</td><td>0.121</td><td>OTUD3</td><td>0.590</td><td>ESR1</td><td>0.033</td></tr><tr><td>C7</td><td>0.773</td><td>TMSB10</td><td>0.597</td><td>SPOCD1</td><td>0.606</td><td>TMSB10</td><td>0.077</td><td>CAPZB</td><td>0.590</td><td>C7</td><td>0.032</td></tr><tr><td>NCAPH2</td><td>0.734</td><td>NCAPH2</td><td>0.586</td><td>C6orf26</td><td>0.593</td><td>C6orf26</td><td>0.071</td><td>C6orf26</td><td>0.580</td><td>CCNB2</td><td>0.031</td></tr><tr><td>UPK3B</td><td>0.727</td><td>C20orf111</td><td>0.570</td><td>TUBB</td><td>0.506</td><td>C7</td><td>0.069</td><td>STXBP1</td><td>0.580</td><td>KIAA1949</td><td>0.029</td></tr><tr><td>PARP1</td><td>0.710</td><td>NUP210</td><td>0.562</td><td>OR52E8</td><td>0.502</td><td>HACE1</td><td>0.064</td><td>ACLY</td><td>0.580</td><td>NCAPH2</td><td>0.029</td></tr><tr><td>HACE1</td><td>0.680</td><td>CAPZB</td><td>0.553</td><td>HACE1</td><td>0.498</td><td>GPX2</td><td>0.062</td><td>COL25A1</td><td>0.580</td><td>OAS2</td><td>0.029</td></tr></table>

## 5. Conclusion

We introduced Hide&Seek, an end-to-end differentiable method for instance-wise feature selection. The method jointly trains a selector and predictor while avoiding a key failure mode: information leakage. Instead of ablating unselected features with a fixed value, Hide&Seek replaces them with random draws from the feature distribution. This prevents the predictor from using the ablation pattern itself as a source of information. Our use of continuous masks enables efficient joint optimization: rather than sampling a discrete subset of features, we replace a proportion of each feature. Our parsimony-weight annealing schedule stabilizes training by prioritizing predictive accuracy early and sparsity in later stages.

We also provided a theoretical analysis of information leakage in jointly trained ablation models. In particular, we showed that the usual training objective for ablation-based models can be an ill-posed proxy for the intended expected KL objective. This causes genuinely important features to be omitted without degrading predictive performance. We show that for switch features, which affect the response through their induced partition, this yields a lower bound on the achievable misidentification rate. Our experiments support this analysis: leakage-prone methods often fail to identify the switch feature at rates close to the theoretical bound, whereas Hide&Seek maintains high switch accuracy.

Under the idealized Hide&Seek construction: a binary mask and drawing replacement values conditioned on the selected features, information leakage is theoretically impossible. Our empirical results further show that Hide&Seek remains robust to information leakage when using a continuous relaxation of the mask and drawing replacement values from the marginal feature distributions.

Across synthetic benchmarks, correlated semi-synthetic credit-default experiments, MNIST explanations, and genetic microarray data, Hide&Seek achieves strong instancewise feature-selection performance while remaining fast to train. These results suggest that replacing fixed ablations with distributional feature replacement preserves the benefits of joint selector–predictor training without introducing an artificial information channel.

## Acknowledgements

This project was made possible by CSIRO’s Next Generation Graduates program and the Paul Ramsay Foundation through the Thrive: Finishing School Well program. We thank the reviewers for their thoughtful feedback.

## Author Contributions

H.M.A. conceived the main idea, developed the theoretical findings and proofs, and designed the information-leakageresistant modelling approach. T.E. implemented and developed the model, including the annealing framework, designed and conducted the experiments and wrote the first draft. S.C. supervised the research, provided conceptual guidance and contributed to the interpretation of the results. All authors contributed to discussions and reviewed the final manuscript.

## Impact Statement

This paper presents work whose goal is to advance the field of Explainable AI (XAI). Improved interpretability of blackbox models can support transparency and informed decisionmaking. However, incorrect or misleading explanations can introduce risk, which underscores the need for careful validation.

We also prove that a seemingly natural loss function used in prior XAI work is ill-posed, in the sense that it can lead to unintended information leakage. This highlights the broader challenge of designing neural network training objectives that do not encourage unintended behavior.

## References

Arik, S. Ö. and Pfister, T. Tabnet: Attentive interpretable tabular learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 35, 2021.

Berger, A. C. et al. A comprehensive pan-cancer molecular study of gynecologic and breast cancers. Cancer Cell, 33 (4):690–705, 2018.

Bowman, S. R., Vilnis, L., Vinyals, O., Dai, A. M., Jozefowicz, R., and Bengio, S. Generating sentences from a continuous space. In Proceedings of the 20th SIGNLL Conference on Computational Natural Language Learning (CoNLL), pp. 10–21. Association for Computational Linguistics, 2016.

Breiman, L. Random forests. Machine Learning, 45(1): 5–32, 2001.

Candès, E., Fan, Y., Janson, L., and Lv, J. Panning for gold: ‘model-X’ knockoffs for high dimensional controlled variable selection. Journal of the Royal Statistical Society Series B: Statistical Methodology, 80(3):551–577, 2018.

Chen, J., Song, L., Wainwright, M. J., and Jordan, M. I. Learning to explain: An information-theoretic perspective on model interpretation. In International Conference on Machine Learning, pp. 2703–2712. PMLR, 2018.

Chen, T. and Guestrin, C. Xgboost: A scalable tree boosting system. In Proceedings ofthe 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794. ACM, 2016.

Choi, E., Bahadori, M. T., Schuetz, A., Stewart, W. F., and Sun, J. Retain: An interpretable predictive model for healthcare using reverse time attention mechanism. In Advances in Neural Information Processing Systems, volume 29, 2016.

Covert, I., Lundberg, S., and Lee, S.-I. Explaining by removing: A unified framework for model explanation. Journal ofMachine Learning Research, 22(209):1–90, 2021.

Crabbé, J. and Van Der Schaar, M. Explaining time series predictions with dynamic masks. In International Conference on Machine Learning, pp. 2555–2565. PMLR, 2021.

Dabkowski, P. and Gal, Y. Real time image saliency for black box classifiers. Advances in neural information processing systems, 30, 2017.

Datta, A., Sen, S., and Zick, Y. Algorithmic transparency via quantitative input influence: Theory and experiments with learning systems. In Proceedings ofthe IEEE Symposium on Security and Privacy, pp. 598–617. IEEE, 2016.

Guyon, I. and Elisseeff, A. An introduction to variable and feature selection. Journal of Machine Learning Research, 3:1157–1182, 2003.

Hooker, S., Erhan, D., Kindermans, P.-J., and Kim, B. A benchmark for interpretability methods in deep neural networks. In Advances in Neural Information Processing Systems, volume 32, pp. 9734–9745, 2019.

Imrie, F., van der Schaar, M., and Lane, N. D. Composite feature selection using deep ensembles. In Advances in Neural Information Processing Systems, volume 35, pp. 36142–36160, 2022.

Jang, E., Gu, S., and Poole, B. Categorical reparameterization with Gumbel-Softmax. In Proceedings of the International Conference on Learning Representations, 2017.

Jethani, N., Sudarshan, M., Aphinyanaphongs, Y., and Ranganath, R. Have we learned to explain?: How interpretability methods can learn to encode predictions in their interpretations. In International Conference on Artificial Intelligence and Statistics, pp. 1459–1467. PMLR, 2021.

Kogan, E. et al. Assessing stroke severity using electronic health record data: a machine learning approach. BMC Medical Informatics and Decision Making, 20(1):8, 2020.

Kohavi, R. and John, G. H. Wrappers for feature subset selection. Artificial Intelligence, 97(1–2):273–324, 1997.

LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradientbased learning applied to document recognition. Proceedings ofthe IEEE, 86(11):2278–2324, 2002.

Levy, K., Chasalow, K. E., and Riley, S. Algorithms and decision-making in the public sector. Annual Review of Law and Social Science, 17(1):309–334, 2021. doi: 10.1146/annurev-lawsocsci-041221-023808.

Liu, H. and Setiono, R. A probabilistic approach to feature selection — a filter solution. In Proceedings ofthe International Conference on Machine Learning, volume 96, pp. 319–327, 1996.

Louppe, G., Wehenkel, L., Sutera, A., and Geurts, P. Understanding variable importances in forests of randomized trees. In Advances in Neural Information Processing Systems, volume 26, 2013.

Lundberg, S. M. and Lee, S.-I. A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems, volume 30, 2017.

Maddison, C. J., Mnih, A., and Teh, Y. W. The concrete distribution: A continuous relaxation of discrete random variables. arXiv preprint arXiv:1611.00712, 2016.

Pace, R. K. and Barry, R. Sparse spatial autoregressions. Statistics & Probability Letters, 33(3):291–297, 1997.

Peters, J. and Schaal, S. Natural actor-critic. Neurocomputing, 71(7–9):1180–1190, 2008.

Petsiuk, V., Das, A., and Saenko, K. Rise: Randomized input sampling for explanation of black-box models. arXiv preprint arXiv:1806.07421, 2018.

Ribeiro, M. T., Singh, S., and Guestrin, C. “why should i trust you?” explaining the predictions of any classifier. In Proceedings ofthe 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1135–1144. ACM, 2016.

Robinson, D. R., Wu, Y.-M., Vats, P., Su, F., Lonigro, R. J., Cao, X., Kalyana-Sundaram, S., Wang, R., Ning, Y., Hodges, L., et al. Activating ESR1 mutations in hormoneresistant metastatic breast cancer. Nature Genetics, 45 (12):1446–1451, 2013.

Romano, Y., Sesia, M., and Candès, E. Deep knockoffs. Journal ofthe American Statistical Association, 115(532): 1861–1872, 2020.

Rudin, C. and Shaposhnik, Y. Globally-consistent rulebased summary-explanations for machine learning models: Application to credit-risk evaluation. Journal of Machine Learning Research, 24(16):1–44, 2023.

Schwab, P. and Karlen, W. Cxplain: Causal explanations for model interpretation under uncertainty. Advances in neural information processing systems, 32, 2019.

Shrikumar, A., Greenside, P., and Kundaje, A. Learning important features through propagating activation differences. In International Conference on Machine Learning, pp. 3145–3153. PMLR, 2017.

Shubbar, E., Helou, K., Kovacs, A., Einbeigi, Z., Olsson, H., Malmström, P., Skoog, L., and Bergh, J. Elevated cyclin b2 expression in invasive breast carcinoma is associated with unfavorable clinical outcome. BMC Cancer, 13(1): 1, 2013.

Sundararajan, M., Taly, A., and Yan, Q. Axiomatic attribution for deep networks. In International conference on machine learning, pp. 3319–3328. PMLR, 2017.

Tagaris, T. and Stafylopatis, A. Hide-and-seek: A template for explainable ai. arXiv preprint arXiv:2005.00130, 2020.

Tibshirani, R. Regression shrinkage and selection via the lasso. Journal ofthe Royal Statistical Society: Series B (Methodological), 58(1):267–288, 1996.

Tucker, G., Mnih, A., Maddison, C. J., Lawson, J., and Sohl-Dickstein, J. Rebar: Low-variance, unbiased gradient estimates for discrete latent variable models. In Advances in Neural Information Processing Systems, volume 30, 2017.

Unger, M. and Kather, J. N. Deep learning in cancer genomics and histopathology. Genome Medicine, 16(1):44, 2024. doi: 10.1186/s13073-024-01315-6.

Xu, B. and Yang, G. Interpretability research of deep learning: A literature survey. Information Fusion, 115:102721, 2025.

Yeh, I.-C. and Lien, C.-h. The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2):2473–2480, 2009.

Yoon, J., Jordon, J., and Van der Schaar, M. Invase: Instancewise variable selection using neural networks. In International conference on learning representations, 2018.

Zeiler, M. D. and Fergus, R. Visualizing and understanding convolutional networks. In European conference on computer vision, pp. 818–833. Springer, 2014.

## A. On Feature Ablation and Information Leakage

Let the random vector $\mathbf { X } : = ( X _ { 1 } , \ldots , X _ { d } )$ , defined on $x ,$ , represent the input features and let $Y ,$ , defined on $\mathcal { V } ,$ represent the response. Let $P ^ { \star } ( { \bf X } )$ and $P ^ { \star } ( Y \mid \mathbf { X } = \mathbf { x } )$ denote the empirical marginal distribution of X and conditional distribution of $Y$ induced by the observational data

The ultimate objective of instance-wise feature selection (IWFS) setting is to find a selector function $\boldsymbol { S } ( \mathbf { x } )$ and and a predictor distribution $P _ { \cdot | \mathbb { X } } ( Y | \{ X _ { k } = x _ { k } \} _ { k \in { \cal S } ( { \bf x } ) } )$ such that the selector ${ \bar { \mathcal { S } } } : { \mathcal { X } }  2 ^ { \{ 1 , . . . , d \} }$ maps each instance, x, to a subset of $\{ 1 , \ldots d \}$ indicating the indices of the important features and the pair $( S , P _ { \cdot | \mathbb { X } } )$ minimizes the loss:

$$
\ell _ { \mathbb { X } } ( S , P _ { \cdot | \mathbb { X } } ) = \mathbb { E } _ { \mathbf { x } \sim P ^ { \star } ( \mathbf { X } ) } \left[ \mathrm { K L } \big ( P ^ { \star } ( Y \mid \mathbf { X } = \mathbf { x } ) \mathrm { ~ } \| ~ P _ { \cdot | \mathbb { X } } ( Y \mid \{ X _ { k } = x _ { k } \} _ { k \in S ( \mathbf { x } ) } ) \big ) + \lambda \| S ( \mathbf { x } ) \| \right] .
$$

We assume that $\lambda > 0$ is chosen sufficiently small so that the minimiser of the intended loss $\ell _ { \mathbb { X } }$ selects every important feature.

As mentioned in the main text, modelling this loss is challenging because it requires conditioning on arbitrary subsets of random variables. Ablation-based methods address this by converting $\{ X _ { k } = x _ { k } \} _ { k \in { \cal S } ( { \bf x } ) }$ into a fixed-size ablated vector, Z, viafeature ablation.

That is, ${ \mathrm { i f ~ } } k \in \left\{ 1 , \ldots , d \right\}$ and $k \not \in S ( \mathbf { x } )$ , then the corresponding ablated element, $Z _ { k } ,$ is replaced by a symbol, $\mathrm { e . g . , * . }$ , that is not in the original support, $\mathcal { X } _ { k } . { } ^ { 6 }$ In many XAI methods, $* = 0$ (Zeiler & Fergus, 2014; Petsiuk et al., 2018; Yoon et al., 2018; Schwab & Karlen, 2019), though user-defined values are also common (Ribeiro et al., 2016; Dabkowski & Gal, 2017).

Ablating the features of an instance $\mathbf { X } = \mathbf { x }$ according to the selector $\boldsymbol { S } ( \mathbf { x } )$ is equivalent to constructing a random vector Z using a binary ablation mask $\mathcal { M } \in \{ 0 , 1 \} ^ { d }$

$$
\mathbf { Z } = \mathbf { x } \circledast \mathcal { M } ( \mathcal { S } ( \mathbf { x } ) ) \mathrm { ~ w h e r e ~ } [ \mathcal { M } ( \mathcal { S } ( \mathbf { x } ) ) ] _ { i } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } i \in \mathcal { S } ( \mathbf { x } ) , } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right. \mathrm { ~ a n d ~ } [ \mathbf { x } \circledast \mathbf { m } ] _ { i } = \left\{ \begin{array} { l l } { x _ { i } , } & { m _ { i } = 1 , } \\ { * , } & { m _ { i } = 0 . } \end{array} \right.
$$

These methods then minimise the following proxy loss:

$$
\ell _ { \mathbb { Z } } ( S , P _ { \cdot | \mathbb { Z } } ) : = \mathbb { E } _ { \mathbf { x } \sim P ^ { \star } ( \mathbf { X } ) } \left[ \mathrm { K L } \big ( P ^ { \star } ( Y \mid \mathbf { X } = \mathbf { x } ) \mathrm { ~ } \| ~ P _ { \cdot | \mathbb { Z } } ( Y \mid \mathbf { Z } = \mathbf { x } \otimes \mathcal { M } ( \mathcal { S } ( \mathbf { x } ) ) ) \big ) + \lambda \| ( \mathcal { S } ( \mathbf { x } ) ) \| \right] .
$$

The following theorem shows that $\ell _ { \mathbb { Z } } ( S , P _ { \cdot | \mathbb { Z } } )$ can be an ill-posed proxy for $\ell _ { \mathbb { X } } ( S , P _ { \cdot | \mathbb { X } } )$ . In particular, in a broad class of scenarios, the selector and predictor that minimise the intended loss $\ell _ { \mathbb { X } } ( S , P _ { \cdot | \mathbb { X } } )$ are not minimisers of the proxy loss $\ell _ { \mathbb { Z } } ( S , P _ { \cdot | \mathbb { Z } } )$ . Equivalently, there exist achievable selector–predictor pairs with lower proxy loss than the intended selector– predictor pair. As the theorem shows, such pairs exploit the ablation mask in an unintended way, allowing information to leak between the selector and predictor.

Theorem A.1. Let a subspace, $\mathcal { A } _ { i } \subseteq \mathcal { X } _ { i } ,$ , of a random variable $X _ { i } \in \mathbf { X }$ be partitioned as follows: $\begin{array} { r } { A _ { i } : = \bigcup _ { k = 1 } ^ { N } A _ { i , k } } \end{array}$ (where $N \geq 1 ) .$ . Assume Y depends on $X _ { i }$ through its partition $i . e . ,$ , if the realization, $\mathbf { x } _ { - i } ,$ of the other random variables, $\mathbf { X } _ { - i } : = \mathbf { X } \backslash \{ X _ { i } \}$ , are given, $Y$ and $X _ { i }$ are dependent but they are independent ifthe $X _ { i } \ ' _ { s }$ partition is given:

$$
Y \downarrow \mid X _ { i } \mid \mathbf { X } _ { - i } = \mathbf { x } _ { - i } , \qquad Y \downarrow \mid X _ { i } \mid X _ { i } \in A _ { i , k } , \mathbf { X } _ { - i } = \mathbf { x } _ { - i } , \qquad \forall \mathbf { x } _ { - i } \in \mathcal { X } _ { - i } , \forall A _ { i , k } \subset A _ { i } .\tag{13}
$$

Under this assumption, the loss function of the form:

$$
\ell _ { \Xi } ( S , P _ { \cdot } | \mathbf { z } ) : = \mathbb { E } _ { \mathbf { x } \sim P ^ { \star } ( \mathbf { x } ) } \left[ K L \big ( P ^ { \star } ( Y \mid \mathbf { X } = \mathbf { x } ) \mid \mid P _ { \cdot | \mathbf { Z } } ( Y \mid \mathbf { Z } = \mathbf { x } \oplus \mathcal { M } ( S ( \mathbf { x } ) ) ) \big ) + \lambda \| ( \mathcal { S } ( \mathbf { x } ) ) \| \right] .\tag{14}
$$

is an ill-posed proxy for

$$
\ell _ { \mathbb { X } } ( \boldsymbol { \mathcal { S } } , \boldsymbol { P } _ { \cdot | \mathbb { X } } ) = \mathbb { E } _ { \mathbf { x } \sim \boldsymbol { P } ^ { \star } ( \mathbf { X } ) } \left[ K L \big ( \boldsymbol { P } ^ { \star } ( \boldsymbol { Y } \mid \mathbf { X } = \mathbf { x } ) \mathrm { ~ } \| \mathrm { ~ } \boldsymbol { P } _ { \cdot | \mathbb { X } } ( \boldsymbol { Y } \mid \{ X _ { k } = x _ { k } \} _ { k \in \mathcal { S } ( \mathbf { x } ) } ) \big ) + \lambda \| \boldsymbol { \mathcal { S } } ( \mathbf { x } ) \| \right] .\tag{15}
$$

In the sense that:

$$
\arg \operatorname* { m i n } _ { S } \ell _ { \mathbb { Z } } ( S , P _ { \cdot | \mathbb { Z } } ) \neq \arg \operatorname* { m i n } _ { S } \ell _ { \mathbb { X } } ( S , P _ { \cdot | \mathbb { X } } ) .\tag{16}
$$

Proof. For any ablated vector z, define $\mathcal { T } ( \mathbf { z } )$ as the set of feature indices that are not ablated:

$$
\mathcal { T } ( \mathbf { z } ) : = \{ k \in \{ 1 , \ldots , d \} \mathrm { s . t . } z _ { k } \neq * \} .\tag{17}
$$

It is easy to verify that, for any selector function $s ,$ if $\mathbf { z } = \mathbf { x } \circledast \mathcal { M } ( \mathcal { S } ( \mathbf { x } ) )$ and $* \notin \mathcal { X } ,$ , then

$$
{ \mathcal { T } } ( \mathbf { z } ) = S ( \mathbf { x } ) , { \mathrm { a n d } } \{ z _ { k } \} _ { k \in { \mathcal { Z } } ( \mathbf { z } ) } = \{ x _ { k } \} _ { k \in S ( \mathbf { x } ) } ,\tag{18}
$$

This follows because

$$
\begin{array} { r l } & { k \in \mathcal { S } ( \mathbf { x } ) \implies [ \mathcal { M } ( \mathbf { x } ) ] _ { k } = 1 \implies z _ { k } = x _ { k } \implies k \in \mathcal { T } ( \mathbf { z } ) } \\ & { k \not \in \mathcal { S } ( \mathbf { x } ) \implies [ \mathcal { M } ( \mathbf { x } ) ] _ { k } = 0 \implies z _ { k } = \ast \implies k \not \in \mathcal { T } ( \mathbf { z } ) } \end{array} , \qquad \forall \mathbf { x } \in \mathcal { X } , \forall k \in \{ 1 , \ldots , d \} .
$$

Let

$$
( S ^ { * } , P _ { \cdot | \mathbb { X } } ^ { * } ) = \arg \operatorname* { m i n } _ { ( S , P _ { \cdot | \mathbb { X } } ) } \ell _ { \mathbb { X } } ( S , P _ { \cdot | \mathbb { X } } ) .\tag{19}
$$

denote the intended selector–predictor pair, i.e., the pair that minimises the intended loss. We define $P _ { \cdot | \mathbb { Z } } ^ { * }$ as the predictor on ablated vectors corresponding to the intended predictor $P _ { \cdot | \mathbb { X } } ^ { * }$

$$
P _ { \cdot | \mathbb { Z } } ^ { * } ( Y | \mathbf { Z } = \mathbf { z } ) : = P _ { \cdot | \mathbb { X } } ^ { * } ( Y \mid \{ X _ { k } = z _ { k } \} _ { k \in \mathbb { Z } ( \mathbf { z } ) } )\tag{20}
$$

Therefore,

$$
P _ { : | \mathbb { Z } } ^ { * } ( Y \mid \mathbf { Z } = \mathbf { x } \oplus { \mathcal { M } } ( S ( \mathbf { x } ) ) ) \stackrel { ( 2 0 ) } { : = } P _ { : | \mathbb { X } } ^ { * } ( Y \mid \{ X _ { k } = z _ { k } \} _ { k \in { \mathcal { I } } ( \mathbf { z } ) } ) \stackrel { ( 1 8 ) } { = } P _ { : | \mathbb { X } } ^ { * } ( Y \mid \{ X _ { k } = x _ { k } \} _ { k \in S ( \mathbf { x } ) } ) .\tag{21}
$$

Substituting (21) into the proxy loss (14) gives

$$
\ell _ { \mathbb { Z } } ( S ^ { \ast } , P _ { \cdot | \mathbb { Z } } ^ { \ast } ) = \ell _ { \mathbb { X } } ( S ^ { \ast } , P _ { \cdot | \mathbb { X } } ^ { \ast } ) .\tag{22}
$$

Thus, to prove the theorem, it is enough to construct another pair $( S ^ { \prime } , P _ { \cdot | \mathbb { Z } } ^ { \prime } )$ such that $\ell _ { \mathbb { Z } } ( S ^ { \prime } , P _ { \cdot | \mathbb { Z } } ^ { \prime } ) < \ell _ { \mathbb { Z } } ( S ^ { \ast } , P _ { \cdot | \mathbb { Z } } ^ { \ast } )$

Assuming that the premises (13) of the theorem hold, define

$$
P _ { \cdot | \mathbb { Z } } ^ { \prime } ( Y | \mathbf { Z } = \mathbf { z } ) : = \left\{ \begin{array} { l l } { P _ { \cdot | \mathbb { X } } ^ { * } ( Y \mid \{ X _ { k } = z _ { k } \} _ { k \in \mathcal { Z } ( \mathbf { z } ) } , X _ { i } \in A _ { i , 1 } ) , } & { \mathrm { i f ~ } z _ { i } = * , } \\ { P _ { \cdot | \mathbb { X } } ^ { * } ( Y \mid \{ X _ { k } = z _ { k } \} _ { k \in \mathcal { Z } ( \mathbf { z } ) } ) , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{23}
$$

and

$$
\begin{array} { r } { \mathcal { S } ^ { \prime } ( \mathbf { x } ) : = \displaystyle \left. \begin{array} { l l } { \mathcal { S } ^ { * } ( \mathbf { x } ) \backslash \{ i \} , } & { \mathrm { i f } x _ { i } \in A _ { i , 1 } , } \\ { \mathcal { S } ^ { * } ( \mathbf { x } ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{24}
$$

Intuitively, $S ^ { \prime }$ acts as the intended selector $S ^ { * }$ , except that when the i-th feature lies in the partition cell $A _ { i , 1 }$ , this feature is not selected. The predictor $P _ { \cdot | \mathbb { Z } } ^ { \prime }$ also acts like the intended predictor, except that when the i-th entry of its ablated input is ∗, it interprets this as evidence tha $X _ { i } \in A _ { i , 1 }$

The dependence assumption, Y ̸⊥⊥ $X _ { i } \mid \mathbf { X } _ { - i } = \mathbf { x } _ { - i }$ and the choice of $\lambda ,$ entails that regardless of the realisation of $\mathbf { X } , X _ { i }$ i selected by the intended selector $S ^ { * }$ . Hence,

$$
i \in \mathcal { S } ^ { * } ( \mathbf { x } ) \implies [ \mathcal { M } ( \mathcal { S } ^ { * } ( \mathbf { x } ) ) ] _ { i } = 1 \implies [ \mathbf { x } \circledast \mathcal { M } ( \mathcal { S } ^ { * } ( \mathbf { x } ) ) ] _ { i } = x _ { i } \not = * , \qquad \forall \mathbf { x } \in \mathcal { X } , \forall i \in \mathcal { S } ^ { * } ( \mathbf { x } ) .\tag{25}
$$

Now consider two cases, First, suppose $x _ { i } \notin A _ { i , . }$ <sub>1</sub>. Then,

$$
S ^ { \prime } ( \mathbf { x } ) \stackrel { ( 2 4 ) } { = } S ^ { * } ( \mathbf { x } )\tag{26}
$$

Hence, if $\mathbf { z } = \mathbf { x } \circledast \mathcal { M } ( S ^ { \prime } ( \mathbf { x } ) )$ ), then

$$
z _ { i } \overset { ( 2 6 ) } { = } [ { \mathbf { x } } \oplus \mathcal { M } ( { S ^ { * } } ( { \mathbf { x } } ) ) ] _ { i } \overset { ( 2 5 ) } { = } x _ { i } \neq * \implies P _ { { \mathbf { \xi } } ^ { \prime } \mid { \mathbb { Z } } } ^ { \prime } ( Y \mid { \mathbf { Z } } = { \mathbf { z } } ) \overset { ( 2 3 ) } { = } P _ { \cdot \mid { \mathbb { Z } } } ^ { * } ( Y \mid \{ X _ { k } = z _ { k } \} _ { k \in \mathcal { Z } ( { \mathbf { z } } ) } ) .\tag{27}
$$

Thus (26) and (27) entail that on this region, the instance-wise loss contribution of $( S ^ { \prime } , P _ { \cdot | \mathbb { Z } } ^ { \prime } )$ , is the same as that of $( S ^ { * } , P _ { \cdot | \mathbb { X } } ^ { * } )$

$$
\ell _ { \mathbb { Z } } ( \mathcal S ^ { \prime } , P _ { \cdot | \mathbb { Z } } ^ { \prime } ) = \ell _ { \mathbb { X } } ( \mathcal S ^ { * } , P _ { \cdot | \mathbb { X } } ^ { * } ) , \qquad \forall \mathbf { x } \in \mathcal { X } \mathrm { ~ s . t . } x _ { i } \notin A _ { i , 1 } .\tag{28}
$$

Second suppose $\mathbf { x } \in A _ { i , 1 }$ . Then $S ^ { \prime } ( \mathbf { x } ) \stackrel { ( 2 4 ) } { = } S ^ { * } ( \mathbf { x } ) \backslash \{ i \}$ which entails:

$$
z _ { i } = * , \mathrm { a n d } \mathcal { T } ( \mathbf { z } ) = S ^ { * } ( \mathbf { x } ) \backslash \{ i \} .\tag{29}
$$

By (29) and (23):

$$
P _ { \cdot \mid \mathbb { Z } } ^ { \prime } ( Y \mid \mathbf { Z } = \mathbf { z } ) = P _ { \cdot \mid \mathbb { Z } } ^ { \ast } ( Y \mid \{ X _ { k } = z _ { k } \} _ { k \in \mathcal { S } ^ { \ast } ( \mathbf { x } ) \backslash \{ i \} } , X _ { i } \in A _ { i , 1 } ) = P _ { \cdot \mid \mathbb { Z } } ^ { \ast } ( Y \mid \{ X _ { k } = z _ { k } \} _ { k \in \mathcal { S } ^ { \ast } ( \mathbf { x } ) } ) ,
$$

in which the last equality holds by the theorem’s assumption that Y depends on $X _ { i }$ through its partition.

Therefore, for all $\mathbf { x } \in \mathcal { X }$ such that $x _ { i } \in A _ { i , 1 }$ , the divergence terms in (15) and (14) are equal but $\| S ^ { \prime } ( \mathbf { x } ) \| = \| S ^ { * } ( \mathbf { x } ) \backslash \{ i \} \| =$ $\| S ^ { * } ( \mathbf { x } ) \| - 1$ . Thus, on this region, the instance-wise loss contribution of $( S ^ { \prime } , P _ { \cdot | \mathbb { Z } } ^ { \prime } )$ is strictly smaller than that of $( S ^ { * } , P _ { \cdot | \mathbb { X } } ^ { * } )$ while as we showed by (28), on the complement region $x _ { i } \notin A _ { i , 1 }$ , the instance-wise loss contributions of $( S ^ { \prime } , P _ { \cdot | \mathbb { Z } } ^ { \prime } )$ and $( S ^ { * } , P _ { \cdot | \mathbb { X } } )$ are equal. Therefore

$$
\ell _ { \mathbb { Z } } ( S ^ { \prime } , P _ { \cdot | \mathbb { Z } } ^ { \prime } ) < \ell _ { \mathbb { Z } } ( S ^ { \ast } , P _ { \cdot | \mathbb { Z } } ^ { \ast } ) \overset { ( 2 2 ) } { = } \ell _ { \mathbb { X } } ( S ^ { \ast } , P _ { \cdot | \mathbb { X } } ^ { \ast } ) .
$$

which completes the proof.

Corollary A.2 (Lower bound on the achievable feature-importance misidentification rate). Suppose the support $\mathcal { X } _ { i }$ ofa random variable $X _ { i } \in \mathbf { X }$ be partitioned as $\textstyle \mathcal { X } _ { i } : = \bigcup _ { k = 1 } ^ { N } A _ { i , k }$ and suppose Y depends on $X _ { i }$ only through its partition, in the sense of (13). Then a lower bound on the achievable rate at which $X _ { i }$ is ablated, and hence misidentified as unimportant, is $P ^ { \star } ( X _ { i } \in A _ { i , \operatorname* { m a x } } )$ where

$$
A _ { i , \operatorname* { m a x } } \in \arg \operatorname* { m a x } _ { A _ { i , k } \in \{ A _ { i , 1 } , \ldots , A _ { i , N } \} } P ^ { \star } ( X _ { i } \in A _ { i , k } ) .
$$

Proof. In Theorem A.1, any partition cell $A _ { i , k }$ of $\mathcal { A } _ { i } = \mathcal { X } _ { i }$ can be chosen to play the role of $A _ { i , 1 }$ . For any such choice, the constructed leakage solution ablates $X _ { i }$ whenever $X _ { i } \in A _ { i , k }$ . Therefore, choosing the cell $A _ { i , \operatorname* { m a x } }$ yields an achievable misidentification rate of $P ^ { \star } ( X _ { i } \in A _ { i , \operatorname* { m a x } } )$ □

Remark. Corollary A.2 is an existence result. It does not imply that every selector–predictor pair trained to minimize the proxy loss will attain the stated misidentification rate. This is because the training procedure may fail to discover the leakage strategy constructed in the proof, for example, by converging to a local minimum. Nor does the corollary imply that the stated rate $P ^ { \star } ( X _ { i } \in A _ { i , \operatorname* { m a x } } )$ is the largest possible misidentification rate: other leakage strategies may achieve lower proxy loss while ablating $X _ { i }$ on a larger subset of the input space. Rather, the corollary guarantees that there exists a selector–predictor pair whose proxy loss is lower than that of the intended selector–predictor pair and whose misidentification rate is $P ^ { \star } ( X _ { i } \in A _ { i , \operatorname* { m a x } } )$ . Thus, this quantity provides a lower bound on the largest misidentification rate achievable through information leakage.

## A.1. Comparison to Jethani et al. (2021)

Jethani et al. (2021) were the first to identify information leakage in jointly trained selector–predictor networks, which they refer to as joint amortized explanation methods (JAMs). They proved this problem in two scenarios. First, in a classification task where the number of possible outputs does not exceed the number of input features, the selector can encode the class label through the binary ablation mask, for instance, via a one-hot vector. This allows the predictor to decode the output from the mask rather than from genuinely informative selected features. The second scenario is when the output is generated by a tree-structured process whose leaves use distinct predictive feature sets. Here, the selected leaf-specific feature set can reveal the active branch. As a result, non-leaf (control-flow) features can be omitted while preserving the predictive likelihood, and the sparsity penalty then favors their omission.

In this paper, we study the JAM information leakage problem in a setting where the output $Y$ may depend on a feature X through the partition of the input space induced by that feature. As a simple example, consider a binary response whose probability changes according to the sign of X, e.g., $P ( Y = 1 \mid X ) = p _ { - } \mathrm { i f } X < 0$ and $P ( Y = 1 \mid X ) = p _ { + } \mathrm { i f } X \ge 0 ,$ with $p _ { - } \neq p _ { + }$ . In this case, the relevant information is whether $X \in ( - \infty , 0 ) \operatorname { o r } X \in [ 0 , + \infty )$ , rather than the exact value of X. We refer to such variables as switchfeatures throughout. This setting is more general than the scenarios studied by

Jethani et al. (2021) and covers cases where the output is either real-valued or discrete, corresponding to regression and classification tasks, respectively. There is no restriction on the number of classification labels, and if the output is generated by a tree-structured process, there is no restriction on the feature sets used in different leaves. In Section A, we show that, in this relaxed setting, JAM algorithms such as L2X and INVASE can still suffer from information leakage. Specifically, we show that the loss function used by such algorithms is an ill-posed proxy for the expected KL loss that they aim to minimize in the sense that the selected features that minimize the two loss functions are not the same.

## A.2. Sampling xˆ Using Model-X Knockoffs

As outlined in the paper, leakage can occur when a predictor network learns extra information from its masked inputs. A jointly trained selector–predictor network can exploit this knowledge by masking important features while retaining prediction accuracy, for example with ablation to 0. In Hide&Seek, we draw our replacement values from the product of the marginals, per eq. (30). Our results in Section 4.2, Section 4.3 and Section A.4 demonstrate the absence of leakage in this setting. Nonetheless, as outlined in Section 2, a stronger theoretical guarantee against leakage can be achieved by sampling replacement values from eq. (31). We present an alternative method using Model-X Knockoffs to more closely approximate this sampling.

$$
{ \hat { \mathbf { x } } } \sim \prod _ { i \in D } p ( X _ { i } )\tag{30}
$$

$$
\hat { \mathbf { x } } _ { \bar { S } } \sim p ( \mathbf { X } _ { \bar { S } } \mid \mathbf { x } _ { S } )\tag{31}
$$

Model-X Knockoffs. A random vector $\tilde { X } \in \mathbb { R } ^ { p }$ is a Model-X knockoff of X constructed to satisfy the following two properties (Candès et al., 2018):

(a) $( X , { \tilde { X } } ) _ { \operatorname { s w a p } ( S ) } \mathrel { \mathop { = } } ( X , { \tilde { X } } )$ for any subset $S \subset \{ 1 , \ldots , p \}$

where $( \cdot ) _ { \mathrm { s w a p } ( S ) }$ swaps the entries $X _ { j }$ and $\tilde { X } _ { j }$ for each $j \in S .$

(b) If there is a response variable Y , then ${ \tilde { X } } \perp \perp Y \mid X$

In our implementation, we generate Model-X knockoffs (Romano et al., 2020) using the package provided at https:// web.stanford.edu/group/candes/deep-knockoffs/. The results for Experiment 4.1 are shown in Figure 6a and the results for Experiment 4.3 are shown in Figure 6b. These results show that marginal sampling, which was used in the paper, outperforms knockoff sampling.

<table><tr><td>Model</td><td>Marginal sampling TPR FDR</td><td>Knockoff sampling TPR</td><td>FDR</td></tr><tr><td>Syn1</td><td>100</td><td>0 99</td><td>0</td></tr><tr><td>Syn2</td><td>100</td><td>0 100</td><td>0</td></tr><tr><td>Syn3</td><td>99 0</td><td>98</td><td>0</td></tr><tr><td>Syn4</td><td>99 4</td><td>99</td><td>4</td></tr><tr><td>Syn5</td><td>97 3</td><td>96</td><td>3</td></tr><tr><td>Syn6</td><td>98 4</td><td>89</td><td>4</td></tr></table>

(a)

![](images/c749b70df0e16c285f1b3e274a5de4b255aa7dd0e18d307189fe604bef59f25d.jpg)  
(b)  
Figure 6. Comparison of alternative methods for sampling xˆ in the Hide&Seek architecture, showing that marginal sampling outperforms knockoff sampling. (a) can be directly compared to Table 2 in the paper and (b) can be directly compared to fig. 4.

Note that while the exchangeability property $( X , { \tilde { X } } ) _ { \operatorname { s w a p } ( S ) } \mathrel { \mathop { = } } ( X , { \tilde { X } } )$ is guaranteed for any fixed subset S, this invariance does not hold in our context. This is because the selector–predictor architecture determines the masked subset using the selector network, meaning the chosen features are a function of the data itself, $S = f ( X )$ . Consequently, $( X , { \tilde { X } } ) _ { \operatorname { s w a p } ( f ( X ) ) } \neq$ $( X , { \tilde { X } } )$

## A.3. Preventing Information Leakage

Let us recall the following:

$\mathbf { X } : = ( X _ { 1 } , \ldots , X _ { d } ) \in \mathcal { X } ,$

$D : = \{ 1 , \ldots , d \} .$

$S \subseteq D$ the indices of the selected features,

$\bar { S } : = D \backslash S$ the indices of the unselected features

$$
\mathbf { \nabla } \bullet \mathbf { z } = \mathbf { m } \odot \mathbf { x } + ( 1 - \mathbf { m } ) \odot { \hat { \mathbf { x } } }
$$

In Section 2 we proposed that information leakage is prevented when replacement values are drawn from $\hat { \mathbf { x } } _ { \bar { S } } \sim p ( \mathbf { X } _ { \bar { S } } \mid \mathbf { x } _ { S } )$ The justification is as follows:

If the modified signal, z, received by the predictor network is distributed according to the original feature distribution, i.e. $p ( \mathbf { Z } ) = p ( \mathbf { X } )$ , then there is no way for the predictor to differentiate between the two and the information leakage becomes impossible. While, in our paper, we use a continuous relaxation where m $\in [ 0 , 1 ] ^ { d }$ , let us consider the binary condition where m $\in \{ 0 , 1 \} ^ { d }$

In this setting, let $\mathbf { Z } : = ( \mathbf { X } _ { S } , \hat { \mathbf { X } } _ { \bar { S } } )$ . Thus, $p ( \mathbf { Z } ) = p ( \mathbf { X } _ { S } , \hat { \mathbf { X } } _ { \bar { S } } ) = p ( \mathbf { X } _ { S } ) p ( \hat { \mathbf { X } } _ { \bar { S } } \mid \mathbf { X } _ { S } )$ . This distribution is equal to $p ( \mathbf { X } ) = p ( \mathbf { X } _ { S } , \mathbf { X } _ { \bar { S } } )$ if and only if $\dot { { \mathbf { \xi } } } p ( \hat { \mathbf { X } } _ { \bar { S } } \mid \mathbf { X } _ { S } ) = p ( \mathbf { X } _ { \bar { S } } \mid \mathbf { X } _ { S } )$ . Thus to prevent any possible leakage xˆ should ideally be drawn from the conditional distribution: $\hat { \mathbf { x } } _ { \bar { S } } \sim p ( \mathbf { X } _ { \bar { S } } \mid \mathbf { x } _ { S } )$

## A.4. Correlated Synthetic Data

Table 7 shows the switch feature accuracy for each of the models and $\rho$ values in the Section 4.3 experiment. It demonstrates that even with marginal sampling per eq. (7), the Seek module in Hide&Seek was unable to learn that replacement values for the switch feature were out-of-distribution.

Table 7. Switch accuracy (identifying $X _ { 1 1 }$ as important) across ρ and models. Each value is the mean switch accuracy across Syn4, Syn5 and Syn6 for a single run of each. Note that some variability in results is expected, as indicated by the boxplots in Figure 3.
<table><tr><td>ρ</td><td>Hide&amp;Seek</td><td>INVASE</td><td>L2X</td><td>LIME</td><td>REAL-X</td><td>SHAP</td></tr><tr><td>0.0</td><td>0.998</td><td>0.547</td><td>0.577</td><td>0.179</td><td>1.000</td><td>0.813</td></tr><tr><td>0.1</td><td>0.999</td><td>0.059</td><td>0.557</td><td>0.156</td><td>1.000</td><td>0.824</td></tr><tr><td>0.2</td><td>0.996</td><td>0.720</td><td>0.578</td><td>0.196</td><td>1.000</td><td>0.850</td></tr><tr><td>0.3</td><td>0.999</td><td>0.561</td><td>0.547</td><td>0.285</td><td>1.000</td><td>0.815</td></tr><tr><td>0.4</td><td>1.000</td><td>0.695</td><td>0.613</td><td>0.210</td><td>1.000</td><td>0.802</td></tr><tr><td>0.5</td><td>0.999</td><td>0.788</td><td>0.571</td><td>0.339</td><td>1.000</td><td>0.820</td></tr><tr><td>0.6</td><td>1.000</td><td>0.747</td><td>0.563</td><td>0.372</td><td>1.000</td><td>0.841</td></tr><tr><td>0.7</td><td>1.000</td><td>0.688</td><td>0.554</td><td>0.675</td><td>0.967</td><td>0.901</td></tr><tr><td>0.8</td><td>1.000</td><td>0.786</td><td>0.538</td><td>0.936</td><td>0.992</td><td>0.906</td></tr><tr><td>0.9</td><td>0.994</td><td>0.844</td><td>0.555</td><td>0.875</td><td>0.993</td><td>0.967</td></tr></table>

## B. Model infrastructure

Hide&Seek. Hide&Seek consists of two fully connected, feed-forward neural networks with ReLU activation functions. The last layer activation function of Hide is an element-wise sigmoid to ensure that the mask $\mathbf { m } \in [ 0 , 1 ] ^ { d }$ , while the last layer of Seek uses a softmax activation. Each network has two hidden layers with ReLU activation functions and each hidden layer has 32 dimensions. The model is trained in 500 epochs without batching. We use the Adam optimizer (learning rate = 0.001), and model weights are initialized using the default PyTorch setting. The implementation is based on PyTorch v2.7.1 with CUDA 12.8. At each epoch, the training data columns are internally shuffled with replacement to create a dataset from which to draw ${ \hat { x } } _ { j }$ values. $\lambda _ { \mathrm { m a x } } = 0 . 3$ for the synthetic data in Section 4.1. Our code is available at https://github.com/talellinson/hide-and-seek-icml2026.

INVASE. The implementation of INVASE uses the code in https://github.com/iclr2018invase/INVASE. Specifically, the selector (actor) network has two hidden layers, each with 100 dimensions. The predictor (critic) network has two hidden layers, each with 200 dimensions. The number of training epochs is 10,000, the batch size is 1, 000 and λ = 0.1 for the synthetic data in Section 4.1.

REAL-x. The implementation of REAL-x uses the code in https://github.com/rajesh-lab/realx. Specifically, the selector network has two hidden layers, each with 100 dimensions. The predictor network has two hidden layers, each with 200 dimensions. The number of training epochs is 500, the batch size is 1, 000 and λ = 0.15 for the synthetic data in Section 4.1.

L2X. The implementation of L2X uses the code in https://github.com/Jianbo-Lab/L2X/tree/master. Like INVASE, there are two networks, each with two hidden layers. Each hidden layer of the first network has 100 dimensions and each hidden layer of the second network has 200 dimensions.

LIME. The implementation of LIME uses the code in https://github.com/marcotcr/lime/tree/master.   
The baseline models for our Synthetic and MNIST data can be found in our repository.

SHAP. The implementation of SHAP uses the code in https://github.com/shap/shap. We explored two implementations of the SHAP package: KernelExplainer and TreeExplainer. Kernel SHAP uses weighted linear regression, similarly to LIME (Lundberg & Lee, 2017) and can be run on neural networks. Tree SHAP is a fast, tree-based algorithm that works with ensembles of trees. Tree SHAP performed better on our synthetic data, so we used it with a base XGBoost predictor (Chen & Guestrin, 2016). This combination explains the fast run time in Appendix D.5. The XGBoost model uses the code in https://pypi.org/project/xgboost/. The hyperparameters were chosen after tuning and are: {’objective’: ’binary:logistic’, ’eval\_metric’: ’logloss’, ’max\_depth’: 5, ’eta’: 0.1, ’colsample\_bytree’: 0.9, ’num\_boost\_round’: 100} in Section 4.1.

RForest. The implementation of RForest uses the code in https://scikit-learn.org/stable/modules/ generated/sklearn.ensemble.RandomForestClassifier.html. The hyperparameters are: {crite rion=’gini’, n\_estimators=100, max\_depth=5}.

LASSO. The implementation of LASSO uses the code in https://scikit-learn.org/stable/modules/ generated/sklearn.linear\_model.LogisticRegression.html. We use logistic regression with an $L _ { 1 }$ penalty.

## B.1. Hardware

Experiments were conducted on the following hardware:

• AMD EPYC 9354P 3.25GHz 32 cores 256MB L3 Cache (Max Turbo Freq. 3.75GHz)

• 192GB 4800MHz ECC DDR5-RAM (Twelve Channel)

• 1.92TB NVMe SSD Drive and 1.92TB NVMe SSD Drive

• 2x NVIDIA L4 (7,680 Cores, 240 Tensor Cores, 24GB Memory) GPUs

## C. Metric calculations

Explaining the TPR, FDR and F1 metrics. To compute the TPR, FDR and F1 values across a dataset, we first calculated the TPR, FDR and F1 score for each input instance (e.g. row in tabular data) using eq. (32) and eq. (33). The mean was then taken across the entire dataset. Where specified, each experiment was run 20 times using different seeds, with the median values reported in the tables. The full distributions for Section 4.1 are shown in Figure 7 as boxplots.

$$
\mathrm { T P R } = \frac { \mathrm { t r u e ~ p o s i t i v e s } } { \mathrm { t r u e ~ p o s i t i v e s } + \mathrm { f a l s e ~ n e g a t i v e s } }
$$

$$
\mathrm { F D R } = { \frac { \mathrm { f a l s e ~ p o s i t i v e s } } { \mathrm { t r u e ~ p o s i t i v e s } + \mathrm { f a l s e ~ p o s i t i v e s } } }\tag{32}
$$

$$
{ \mathrm { F 1 ~ s c o r e } } = 2 \cdot { \frac { { \mathrm { P r e c i s i o n } } \cdot { \mathrm { R e c a l l } } } { { \mathrm { P r e c i s i o n } } + { \mathrm { R e c a l l } } } }\tag{33}
$$

where Recall = TPR and Precision = 1 − FDR.

![](images/30063195238aa107c96174dcdb92cf3440cc0c249157ebf12d29e57c849ad201.jpg)  
(a)

![](images/130ae7ab42a881e48e7f1aeb94658b4ed7ffaca243d07d7ab84f4d0fdcd3b504.jpg)  
(b)

![](images/7678652c13fcd56ae9a407b168765925c86bff19769b63d30d6beed358329c68.jpg)  
(c)  
Figure 7. Distributions of (a) mean TPR, (b) mean FDR, and (c) mean F1 scores for feature identification across Syn1-6. Each boxplot shows the distribution across 20 experiments. The medians of the TPR and FDR boxplots are reported in Table 2.

## D. Further analyses

Section D contains analyses and experiments relating to the synthetic data experiment 4.1.

## D.1. Mask Distributions

Figure 8 shows the learned mask distributions for the six synthetic experiments in Section 4.1. Note the close alignment with the data-generating rules defined in Section 4.1.

![](images/26828ac9ff2541a1bf41262d3654536bd30338571939fba2da54667e4e682ed6.jpg)  
Figure 8. The mask distribution for the 6 synthetic experiments in Section 4.1, for one of the 20 experiments. Shown are the histograms and associated KDE plots. $X _ { 1 1 }$ is the switch-feature, used in Syn4–6.

## D.2. Predictive Performance

Figure 9 shows the predictive performance (AUROC) of the models used in Section 4.1. Hide&Seek has consistently higher predictive performance than comparable selector–predictor models INVASE, L2X and REAL-x, despite having significantly fewer parameters.

![](images/144dc733bc9d381a4d4433961671b462785bb81a1244b5e80cb051dcaf1bfffc.jpg)  
Figure 9. Predictive performance (AUROC) of the models in Section 4.1.

The poorer predictive performance of REAL-x is likely due to out-of-distribution draws when predicting on highly parsimonious data (Hooker et al., 2019). Recall that the predictor network in REAL-x is trained disjointly on feature masks drawn from a Bernoulli(0.5) distribution.

## D.3. λ Sensitivity

Three of the models: Hide&Seek, INVASE and REAL-x have the parameter λ (or $\lambda _ { \mathrm { m a x } }$ for Hide&Seek, which we will refer to as λ, here) which balances the trade-off between parsimony and prediction accuracy. When tuning, AUROC is typically calculated on a validation dataset for varying values of λ. Another available metric is the percentage significance, which reports the proportion of instance-wise features that the model has deemed important. The goal is to choose a λ which provides high parsimony while preserving high prediction accuracy. Figure 10 shows these metrics on a validation dataset for each of the models, for varying values of λ. It provides a number of useful insights.

Firstly, we see that Hide&Seek has a robust sensitivity to varying values of λ. While INVASE and REAL-x quickly become too parsimonious as λ increases, Hide&Seek maintains a percentage significance close to the ground truth. This is likely due to the annealing schedule, which ramps up the weight of the parsimony term in the loss function towards the end of training. Secondly, Hide&Seek maintains a strong AUROC, with almost all of its AUROC values above 0.8. Conversely, REAL-x quickly loses prediction accuracy as λ increases. Recall that the predictor network in REAL-x is trained disjointly on feature masks drawn from a Bernoulli(0.5) distribution. Therefore, when a highly parsimonious feature set is received, REAL-x performance could degrade due to out-of-distribution (OOD) shift (Hooker et al., 2019).

We have also added, post-hoc, the F1 instance-wise feature importance scores to the graphs to demonstrate the relationship between the AUROC used in tuning and the underlying IWFS metric we are trying to optimize. Note that there is a strong correlation between Hide&Seek’s AUROC and the IWFS F1 results. Specifically, the AUROC initially increases as parsimony increases, implying that the model’s predictions could be benefiting from discarding unimportant features. Conversely, the highest AUROC for REAL-x occurs when it has the most features available. INVASE has instances of high prediction accuracy corresponding to a low F1 score. The graphs of AUPRC vs λ displayed similar patterns.

![](images/936b7479950d33fed908f8e59f9aac3afd063f45da0def5be1622fb0e6537750.jpg)  
(a) Hide&Seek

![](images/570742de13b308d78e9a14a23311b96b10981a15f66bf5d8a36c217acf19b882.jpg)  
(b) INVASE

![](images/5ed4ddd438ee3a9562b391e7fa5cd572c735504a43fddcf6b6e928dcca6c2683.jpg)  
(c) REAL-x  
Figure 10. λ sensitivity across different models, with AUROC (predicting Y) and F1 (identifying important features) averaged over Syn1-6.

The INVASE results reported in Section 4.1 use $\lambda = 0 . 1$ , which is the value used in the original INVASE paper for the same synthetic data experiments. As shown, this corresponds to high IWFS F1 performance. If, instead, the λ with highest AUROC had been selected $( \lambda = 0 . 9 5 \mathrm { o r } \lambda = 0 . 4 5 )$ , the INVASE IWFS results would have been lower.

## D.4. Results for k = 3 and $k = 4$

For SHAP, LIME, L2X, LASSO, and RForest, the top k important features need to be specified. In the experiments of the main text, k is chosen based on the number of ground truth important features sought for each dataset. Specifically, k = 2 for Syn1, $k = 4$ for Syn2–3 and k = 5 for Syn4–6. This may overestimate the FDR for Syn4 and Syn5, which have only 3 important features when $X _ { 1 1 } < 0$ . To account for this, SHAP, LIME and L2X results for k = 3 and k = 4 are shown in Table 8.

Table 8. TPR and FDR for Syn4–5 for different values of $k ,$ as explained in Section 4.1. Each metric is the median of 20 experiments.
<table><tr><td rowspan="2">Model</td><td rowspan="2">k</td><td colspan="2">Syn4</td><td colspan="2">Syn5</td></tr><tr><td>TPR</td><td>FDR</td><td>TPR</td><td>FDR</td></tr><tr><td>Hide&amp;Seek</td><td></td><td>99</td><td>4</td><td>97</td><td>3</td></tr><tr><td rowspan="4">SHAP</td><td>5</td><td>72</td><td>38</td><td>74</td><td>38</td></tr><tr><td>4</td><td>60</td><td>35</td><td>63</td><td>34</td></tr><tr><td>3</td><td>46</td><td>34</td><td>48</td><td>32</td></tr><tr><td>5</td><td>54</td><td>51</td><td>50</td><td>54</td></tr><tr><td rowspan="5">L2X</td><td>4 3</td><td>40</td><td>50</td><td>41</td><td>51</td></tr><tr><td>5</td><td>30 44</td><td>50</td><td>30</td><td>50</td></tr><tr><td>4</td><td></td><td>65</td><td>53</td><td>57</td></tr><tr><td></td><td>36</td><td>65</td><td>45</td><td>55</td></tr><tr><td>3</td><td>27</td><td>65</td><td>35</td><td>53</td></tr></table>

## D.5. Run Time Analysis

Table 9 reports run times for the instance-wise feature selection models. INVASE’s training time is substantially longer than other methods, due to its REINFORCE architecture. REAL-x employs differentiable training using REBAR gradients (Tucker et al., 2017) but requires separate training of the selector and predictor networks. There is also a significant difference in model complexity. INVASE uses ≈100k parameters, REAL-x uses ≈57k and Hide&Seek uses ≈3k. Hide&Seek does not use batching, which is present in REAL-x, INVASE and L2X. See Appendix B for further detail on model designs.

Table 9. Typical model run times on the synthetic data, with IWFS results reported in table 2. Times include training (10,000 samples), prediction, and feature attribution (10,000 samples). All models were run on identical hardware, described in Appendix B.1.
<table><tr><td>Method</td><td>Run time (hh:mm:ss)</td></tr><tr><td>SHAP</td><td>00:00:02</td></tr><tr><td>L2X</td><td>00:00:03</td></tr><tr><td>Hide&amp;Seek</td><td>00:00:05</td></tr><tr><td>REAL-X</td><td>00:01:16</td></tr><tr><td>LIME</td><td>00:10:54</td></tr><tr><td>INVASE</td><td>01:18:52</td></tr></table>

## D.6. Syn3 - Specification vs. Implementation

We note a minor discrepancy between the specification of model Syn3 in the previous works (Yoon et al., 2018; Chen et al., 2018) and the code linked in their publication. For our experiments, we use the data-generating model of the previous code, so that the results are comparable.

Table 10. Paper and code expressions for Syn3 model in INVASE, L2X, and Hide&Seek.
<table><tr><td>Method</td><td>Source Syn3</td><td></td></tr><tr><td>INVASE</td><td>Paper</td><td> $- 1 0 \sin ( 2 X _ { 7 } ) + 2 | X _ { 8 } | + X _ { 9 } + \exp \bigl ( - X _ { 1 0 } \bigr )$ </td></tr><tr><td rowspan="2">L2X</td><td>Code</td><td> $- 1 0 \sin { \left( 0 . 2 X _ { 7 } \right) } + \left| X _ { 8 } \right| + X _ { 9 } + \exp { \left( - X _ { 1 0 } \right) } - 2 . 4$ </td></tr><tr><td>Paper</td><td> $- 1 0 0 \sin ( 2 X _ { 1 } ) + 2 | X _ { 2 } | + X _ { 3 } + \exp \bigl ( - X _ { 4 } \bigr )$ </td></tr><tr><td></td><td>Code</td><td> $- 1 0 0 \sin ( 0 . 2 X _ { 1 } ) + | X _ { 2 } | + X _ { 3 } + \exp \bigl ( - X _ { 4 } \bigr ) - 2 . 4$ </td></tr><tr><td>Hide&amp;Seek</td><td>Both</td><td> $- 1 0 \sin { \left( 0 . 2 X _ { 7 } \right) } + \left| X _ { 8 } \right| + X _ { 9 } + \exp { \left( - X _ { 1 0 } \right) } - 2 . 4$ </td></tr></table>

## D.7. Parsimony-Weight Annealing Analysis

As outlined in section 3.1, the annealing schedule for $\lambda _ { t }$ over t epochs is $\begin{array} { r } { \lambda _ { t } = \left( \frac { t } { T } \right) ^ { q } \lambda _ { \operatorname* { m a x } } } \end{array}$ , where $q = 2$ for all our experiments. Table 11 shows the stability of the annealing schedule for different choices of $q .$ It includes a mix of instance-wise feature selection (TPR, FDR, F1) and prediction (AUROC) metrics. We see that for square root and linear growth, $q \in \{ 0 . 5 , 1 \}$ , the results are poor, as expected. However, for $q \in \{ 2 , 3 , 4 , 5 \}$ , the results are stable. This is because the values in the second set allow the model to emphasize prediction accuracy early in training and parsimony later, as demonstrated in Figure 2. $\operatorname { A s } q$ increases (for a fixed $\lambda _ { \operatorname* { m a x } } )$ , the number of epochs spent in larger values of $\lambda _ { t }$ decreases, resulting in less parsimony.

Table 11. Sensitivity analysis for choices of q in $\begin{array} { r } { \lambda _ { t } = \left( \frac { t } { T } \right) ^ { q } \lambda _ { \operatorname* { m a x } } } \end{array}$ . TPR, FDR and F1 are instance-wise feature selection metrics and AUROC is a prediction metric. Each value represents the mean across the six synthetic datasets in Section 4.1 and 20 seeds. $\lambda _ { \mathrm { m a x } } = 0 . 3$ for all runs.
<table><tr><td> $q$ </td><td>TPR</td><td>FDR</td><td>F1</td><td>AUROC</td></tr><tr><td>0.5 1.0 2.0</td><td>38.34 78.43 97.66</td><td>5.33 2.71 2.49 98.53 4.30</td><td>44.45 82.88 97.26 96.67</td><td>0.68 0.79 0.83 0.83</td></tr></table>

Figure 11 shows the impact of different choices of $q = \{ 0 , 1 , 2 , 3 \}$ on the loss function (12) for different values of $\lambda _ { \operatorname* { m a x } } .$ Each row shows (a) The cross-entropy term $\textstyle \sum _ { c = 1 } ^ { C } y$ <sub>c</sub> log ˆy<sub>c</sub> vs t. (b) The regularized parsimony term $\frac { \lambda _ { t } } { d } \lVert \mathbf { m } \rVert _ { 1 }$ vs t. (c) The combined loss $\ell ( \alpha , \beta , t )$ vs t. The metrics are calculated on a hold-out validation set using the same data as in the Syn4 experiment in section 4.1. Figure 11 shows that once $\lambda _ { \mathrm { m a x } }$ is set large enough to impose parsimony, the choice $q = 2$ provides the best results. It allows Hide&Seek to prioritize prediction accuracy early and then mask parsimony in later training epochs. It is more stable than $q = 3$ and results in a lower final combined loss. Note that when $t = 5 0 0$ , the loss function has the same value for all four values of $q$ and is therefore comparable.

![](images/10b750f78079e7b15d776d24a42d99894c38f1ca2c7e74641164249b08bd5fc2.jpg)  
Figure 11. Parsimony-weight annealing analysis. These graphs show the impact of different choices of $q = \{ 0 , 1 , 2 , 3 \}$ in $\begin{array} { r } { \lambda _ { t } = \left( \frac { t } { T } \right) ^ { q } \lambda _ { \operatorname* { m a x } } } \end{array}$ on the loss function (12), for different values of $\lambda _ { \operatorname* { m a x } } .$ . Each row shows (a) The cross-entropy term $\textstyle \sum _ { c = 1 } ^ { C } y _ { c } \log \hat { y } _ { c }$ vs t. (b) The regularized parsimony term $\begin{array} { r l } {  { \frac { \lambda _ { t } } { d } \| \mathbf { m } \| . } } \end{array}$ vs t. (c) The combined loss $\ell ( \alpha , \beta , t )$ vs t. See section D.7 for more detail. 24

## E. Further experiments

This section provides the following experiments: training the models on 1,000,000 samples; performance on 100 features;   
and California Housing data.

## E.1. Training on 1,000,000 Samples

Table 12 shows the results from a single experiment for each model and dataset, where the training data was 1,000,000 samples. Note that L2X has improved results when trained on more data. Hide&Seek still outperforms other models.

Table 12. Performance of IWFS algorithms on Syn1-6 when trained on 1,000,000 samples.
<table><tr><td rowspan="2">Model</td><td colspan="2">Hide&amp;Seek</td><td colspan="2">INVASE</td><td colspan="2">REAL-X SHAP</td><td colspan="2">LIME TPR</td><td colspan="2">L2X</td></tr><tr><td>TPR</td><td>FDR 0</td><td>TPR 100</td><td>FDR 0</td><td>TPR FDR 100 0</td><td>TPR 98</td><td>FDR 2</td><td>FDR 24 76</td><td>TPR 100</td><td>FDR 0</td></tr><tr><td>Syn1 Syn2</td><td>100 100</td><td>0</td><td>100</td><td>0</td><td>87 6</td><td>100</td><td>0</td><td>100</td><td>0 100</td><td>0</td></tr><tr><td>Syn3</td><td>99</td><td>0</td><td>100</td><td>0</td><td>94 0</td><td>100</td><td>0</td><td>98 2</td><td>85</td><td>15</td></tr><tr><td>Syn4</td><td>100</td><td>1</td><td>90</td><td>1</td><td>95 4</td><td>70</td><td>38</td><td>55 49</td><td>83</td><td>32</td></tr><tr><td>Syn5</td><td>98</td><td>1</td><td>84</td><td>1</td><td>89 1</td><td>73</td><td>36</td><td>50 53</td><td>90</td><td>29</td></tr><tr><td>Syn6</td><td>98</td><td>1</td><td>90</td><td>1</td><td>95 6</td><td>76</td><td>24</td><td>51 49</td><td>91</td><td>9</td></tr></table>

Note that unlike Hide&Seek and INVASE, we found that the parsimony regularizer λ in REAL-x had to be tuned down as the number of training samples grew from N = 10, 000 (as in section 4.1) to N = 1, 000, 000.

## E.2. Training on 100 Features and Ensembling

We conduct an experiment in which we increase the number of synthetic features from 11 to 100. The relationship between features remains as described in Section 4.1. There are now an additional 89 noise signals.

Additionally, we introduce Hide&Seek<sub>ens</sub>, an extension of our base architecture that leverages ensembling and column subsampling. Taking advantage of the model’s fast training, we train an ensemble of 10 independent models, each observing a random 90% subset of the features. Instance-wise feature importance is then found by averaging the masks across the ensemble and applying the standard > 0.5 selection threshold.

We also present two runs of INVASE, with no architectural change, showing the results on two values of λ. As noted in Section D.3, INVASE is hard to tune and the optimal λ might not be easily read off a λ − AUROC tuning curve (see Figure 10b for an example). In Table 13, INVASE corresponds to the logical choice of λ during tuning (1.2) while $\mathrm { I N V A S E _ { i d e a l } }$ corresponds to the results if the ideal λ (0.6) was known.

In the results, we see that the base Hide&Seek is competitive. $\mathrm { H i d e } \& \mathrm { S e e k } _ { \mathrm { e n s } }$ outperforms all other models and is as good as, if not better than, the ideal INVASE. These results show the applicability of Hide&Seek to large datasets and outline that ensembling and column subsampling could be useful in improving results.

Table 13. F1 values for high-dimensional (100 features) synthetic datasets. The target is the same function of $\{ X _ { 1 } , \ldots , X _ { 1 1 } \}$ as in the earlier experiments. This experiment includes 89 extra unimportant features of independent noise. Each F1 value is the median of 20 experiments. $\mathrm { H i d e } \& \mathrm { S e e k } _ { \mathrm { e n s } }$ is an ensemble of 10 independent Hide&Seek models with 90% column subsampling and $\mathrm { I N V A S E } _ { \mathrm { i d e a l } }$ represents INVASE’s performance if the ideal λ is known.
<table><tr><td>Model</td><td>Hide&amp;Seek F1</td><td> $\mathrm { H i d e } \& \mathrm { S e e k _ { e n s } }$  F1</td><td>INVASE F1</td><td> $\mathrm { I N V A S E _ { i d e a l } }$  F1</td><td>REAL-X F1</td><td>SHAP F1</td><td>LIME F1</td><td>L2X F1</td></tr><tr><td>Syn1</td><td>90</td><td>100</td><td>0</td><td>100</td><td>48</td><td>14</td><td>2</td><td>2</td></tr><tr><td>Syn2</td><td>96</td><td>100</td><td>100</td><td>100</td><td>72</td><td>92</td><td>100</td><td>10</td></tr><tr><td>Syn3</td><td>97</td><td>99</td><td>100</td><td>100</td><td>93</td><td>88</td><td>78</td><td>4</td></tr><tr><td>Syn4</td><td>66</td><td>73</td><td>53</td><td>63</td><td>68</td><td>56</td><td>25</td><td>4</td></tr><tr><td>Syn5</td><td>77</td><td>79</td><td>63</td><td>75</td><td>68</td><td>53</td><td>25</td><td>4</td></tr><tr><td>Syn6</td><td>70</td><td>75</td><td>84</td><td>85</td><td>68</td><td>72</td><td>40</td><td>5</td></tr></table>

## E.3. California Housing

We present a new experiment that evaluates the models against correlated real-world features in a semi-synthetic setting. This experiment includes a switch feature with three partitions. The California Housing dataset contains information on housing block groups in California from the US 1990 Census (Pace & Barry, 1997). Each block group represents an average of 1425.5 people living in proximity. There are 20,640 block groups. We use the following variables: Median Income, Median House Age, Average Rooms, Average Bedrooms, Population, Occupancy (average members per household) and Longitude.

We establish a ground truth feature importance by sampling Y from a Bernoulli distribution: $\begin{array} { r } { P ( Y = 1 | U ) = \frac { 1 } { 1 + e ^ { U } } } \end{array}$ , where the feature importance is based on the switch-feature Longitude, with three geographic partitions:

• Where Longitude < −121.5: U = Average Rooms − Average Bedrooms

• Where −121 ≤ Longitude $\begin{array} { r } { < - 1 1 8 \colon U = \frac { \mathrm { P o p u l a t i o n } } { \mathrm { O c c u p a n c y } } } \end{array}$

• Where Longitude ≥ −118.: U = 5 × Median Income − 2 × (Median House $\mathrm { A g e } ) ^ { 2 }$

We split the data into 12,828 training samples, 1,604 validation samples and 1,604 test samples. Features were standardized using the training data. For Hide&Seek, REAL-x and INVASE, we tuned λ across [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]. We then assessed each model’s IWFS performance in recovering the ground truth features (housing attributes and longitude). The results are shown in Table 14.

Table 14. IWFS performance on California Housing data. Each TPR, FDR and F1 metric is the median of 10 runs of each model on the same test data.

<table><tr><td></td><td>Hide&amp;Seek</td><td>INVASE</td><td>REAL-X</td><td>SHAP</td><td>LIME</td><td>L2X</td></tr><tr><td>TPR</td><td>97</td><td>84</td><td>79</td><td>74</td><td>56</td><td>52</td></tr><tr><td>FDR</td><td>17</td><td>17</td><td>10</td><td>26</td><td>44</td><td>48</td></tr><tr><td>F1</td><td>88</td><td>83</td><td>83</td><td>74</td><td>56</td><td>52</td></tr></table>

## F. Extra detail

This section provides further detail on the non-synthetic data experiments of the paper.

## F.1. Credit Default Data Detail

The features in the Credit default data experiment in Section 4.4 are: LIMIT\_BAL, BILL\_AMT1, BILL\_AMT2, BILL\_AMT3, BILL\_AMT4, BILL\_AMT5, BILL\_AMT6, PAY\_AMT1, PAY\_AMT2, PAY\_AMT3, AGE, PAY\_AMT4, PAY\_AMT5, PAY\_AMT6, SEX, EDUCATION, MARRIAGE, PAY\_0, PAY\_2, PAY\_3, PAY\_4, PAY\_5 and PAY\_6. The binary target for the raw data is default payment next month, which is not used in Syn4–6. See https: //archive.ics.uci.edu/dataset/350/default+of+credit+card+clients for more details. The cor relation matrix is shown in Figure 12.

## F.2. MNIST Detail

In the MNIST experiment in section 4.5, the hyperparameter λ for Hide&Seek, INVASE and REAL-x was tuned over {0.05, 0.1, 0.2, 0.3, 0.4, 0.5}. We also compared two scaling methods: global rescaling to the [0, 1] range and feature-wise standardization to zero mean and unit variance (using the training set). The settings associated with the highest prediction accuracy were selected. For Hide&Seek, zero mean and unit-variance scaling resulted in a higher prediction accuracy for all λ values. λ = 0.05 was used, although explanation patches were largely similar for all 6 values.

## F.3. Breast Cancer Subtype Classification Detail

Table 15 contains a list of the 100 genes used in the experiment in section 4.6. Table 6 shows the top 10 genes identified by each model, ranked by their mean importance scores. Table 16 provides more detail inclduing standard errors.

<table><tr><td colspan="2">Hide&amp;Seek</td><td colspan="2">INVASE</td><td colspan="2">REAL-X</td></tr><tr><td>Gene</td><td>Importance ± SEM</td><td>Gene</td><td> $\mathrm { I m p o r t a n c e } \pm \mathrm { S E M }$ </td><td>Gene</td><td>Importance ± SEM</td></tr><tr><td>ESR1</td><td> $0 . 9 9 9 \pm 0 . 0 0 0$ </td><td>CCNB2</td><td> $0 . 7 2 3 \pm 0 . 0 3 7$ </td><td>ESR1</td><td> $0 . 9 6 9 \pm 0 . 0 0 3$ </td></tr><tr><td>CCNB2</td><td> $0 . 9 5 1 \pm 0 . 0 1 4$ </td><td>ESR1</td><td> $0 . 7 1 1 \pm 0 . 0 4 0$ </td><td>CCNB2</td><td> $0 . 8 9 7 \pm 0 . 0 0 6$ </td></tr><tr><td>STATH</td><td> $0 . 8 2 7 \pm 0 . 0 1 8$ </td><td>ZNF775</td><td> $0 . 6 8 0 \pm 0 . 0 3 9$ </td><td>NUP210</td><td> $0 . 8 3 3 \pm 0 . 0 1 0$ </td></tr><tr><td>C6orf26</td><td> $0 . 8 0 4 \pm 0 . 0 1 7$ </td><td>KLF3</td><td> $0 . 6 1 9 \pm 0 . 0 3 8$ </td><td>C6orf15</td><td> $0 . 7 5 8 \pm 0 . 0 1 3$ </td></tr><tr><td>TUBB</td><td> $0 . 7 8 4 \pm 0 . 0 2 4$ </td><td>C6orf15</td><td> $0 . 6 1 0 \pm 0 . 0 4 0$ </td><td>SLC25A3</td><td> $0 . 7 5 4 \pm 0 . 0 1 0$ </td></tr><tr><td>C7</td><td> $0 . 7 7 3 \pm 0 . 0 2 2$ </td><td>TMSB10</td><td> $0 . 5 9 7 \pm 0 . 0 3 8$ </td><td>SPOCD1</td><td> $0 . 6 0 6 \pm 0 . 0 1 1$ </td></tr><tr><td>NCAPH2</td><td> $0 . 7 3 4 \pm 0 . 0 2 1$ </td><td>NCAPH2</td><td> $0 . 5 8 6 \pm 0 . 0 3 2$ </td><td>C6orf26</td><td> $0 . 5 9 3 \pm 0 . 0 1 5$ </td></tr><tr><td>UPK3B</td><td> $0 . 7 2 7 \pm 0 . 0 3 3$ </td><td>C20orf111</td><td> $0 . 5 7 0 \pm 0 . 0 3 6$ </td><td>TUBB</td><td> $0 . 5 0 6 \pm 0 . 0 1 5$ </td></tr><tr><td>PARP1 HACE1</td><td> $0 . 7 1 0 \pm 0 . 0 3 6$ </td><td>NUP210 CAPZB</td><td> $0 . 5 6 2 \pm 0 . 0 4 1$ </td><td>OR52E8 HACE1</td><td> $0 . 5 0 2 \pm 0 . 0 1 9$ </td></tr><tr><td></td><td> $0 . 6 8 0 \pm 0 . 0 3 1$ </td><td></td><td> $0 . 5 5 3 \pm 0 . 0 3 6$ </td><td></td><td> $0 . 4 9 8 \pm 0 . 0 1 4$ </td></tr><tr><td colspan="2">SHAP</td><td colspan="2">L2X</td><td colspan="2">LIME</td></tr><tr><td>Gene</td><td>Importance ± SEM</td><td>Gene</td><td>Importance ± SEM</td><td>Gene</td><td>Importance ± SEM</td></tr><tr><td>ESR1</td><td> $0 . 6 9 5 \pm 0 . 0 2 9$ </td><td>PENK</td><td> $0 . 6 4 0 \pm 0 . 0 4 8$ </td><td>TUBB</td><td> $0 . 0 4 4 \pm 0 . 0 0 4$ </td></tr><tr><td>CCNB2</td><td> $0 . 4 0 2 \pm 0 . 0 1 6$ </td><td>BIRC3</td><td> $0 . 6 0 0 \pm 0 . 0 4 9$ </td><td>HACE1</td><td> $0 . 0 4 4 \pm 0 . 0 0 5$ </td></tr><tr><td>C6orf15</td><td> $0 . 2 1 2 \pm 0 . 0 0 7$ </td><td>TMEM52</td><td> $0 . 5 9 0 \pm 0 . 0 4 9$ </td><td>C6orf26</td><td> $0 . 0 3 8 \pm 0 . 0 0 4$ </td></tr><tr><td>ZNF385</td><td> $0 . 1 2 6 \pm 0 . 0 0 7$ </td><td>HPS4</td><td> $0 . 5 9 0 \pm 0 . 0 4 9$ </td><td>PENK</td><td> $0 . 0 3 3 \pm 0 . 0 0 4$ </td></tr><tr><td>NUP210</td><td> $0 . 1 2 1 \pm 0 . 0 0 9$ </td><td>OTUD3</td><td> $0 . 5 9 0 \pm 0 . 0 4 9$ </td><td>ESR1</td><td> $0 . 0 3 3 \pm 0 . 0 0 4$ </td></tr><tr><td>TMSB10</td><td> $0 . 0 7 7 \pm 0 . 0 0 6$ </td><td>CAPZB</td><td> $0 . 5 9 0 \pm 0 . 0 4 9$ </td><td>C7</td><td> $0 . 0 3 2 \pm 0 . 0 0 4$ </td></tr><tr><td>C6orf26</td><td>0.071 ± 0.004</td><td>C6orf26</td><td>0.580 ± 0.050</td><td>CCNB2</td><td>0.031 ± 0.003</td></tr><tr><td>C7</td><td> $0 . 0 6 9 \pm 0 . 0 0 4$ </td><td>STXBP1</td><td> $0 . 5 8 0 \pm 0 . 0 5 0$ </td><td>KIAA1949</td><td> $0 . 0 2 9 \pm 0 . 0 0 3$ </td></tr><tr><td>HACE1</td><td> $0 . 0 6 4 \pm 0 . 0 0 6$ </td><td>ACLY</td><td> $0 . 5 8 0 \pm 0 . 0 5 0$ </td><td>NCAPH2</td><td> $0 . 0 2 9 \pm 0 . 0 0 4$ </td></tr><tr><td>GPX2</td><td> $0 . 0 6 2 \pm 0 . 0 0 4$ </td><td>COL25A1</td><td> $0 . 5 8 0 \pm 0 . 0 5 0$ </td><td>OAS2</td><td> $0 . 0 2 9 \pm 0 . 0 0 3$ </td></tr></table>

![](images/83ac3cb85197568e8fbca2c9d45a12b9979e9c0751085d3a5911548f58ba14a1.jpg)  
Figure 12. The correlation matrix for the 23 input features in the Credit default data experiments. The 11 features surrounded by the black box are used for $\{ X _ { 1 } , \ldots , X _ { 1 1 } \}$ in generating Syn4–6 in Section 4.4.

Table 15. 100 genes used for analysis in section 4.6 to match the same experiment as in Covert et al. (2021).
<table><tr><td colspan="8">100 genes</td></tr><tr><td>OSTbeta</td><td>STATH</td><td>MAPK10</td><td>PLEKHG5</td><td>ERO1L</td><td>ZNF711</td><td>ZNF385</td><td>OR52E8</td><td>SLC5A11</td></tr><tr><td>P4HA3</td><td>LHFPL4</td><td>MGC33657</td><td>CAPZB</td><td>RBM15B</td><td>C1orf176</td><td>KLF3</td><td>OLFM4</td><td>NBR2</td></tr><tr><td>CCDC64</td><td>NUP210</td><td>HEMGN</td><td>SLC25A3</td><td>LEF1</td><td>MVD</td><td>OTUD3</td><td>KIAA1949</td><td>SLC44A3</td></tr><tr><td>ZNF775</td><td>THY1</td><td>DYNC1I2</td><td>CYP1A1</td><td>SPTA1</td><td>CLEC4M</td><td>RXFP3</td><td>TSHR</td><td>C7</td></tr><tr><td>CRYBB2</td><td>PPAPDC3</td><td>TXNL4B</td><td>CHST9</td><td>HACE1</td><td>AYTL1</td><td>PRSS35</td><td>ZNF408</td><td>DDC</td></tr><tr><td>CSTL1</td><td>OR2F1</td><td>C12orf50</td><td>SH3YL1</td><td>SNUPN</td><td>COL25A1</td><td>HPS4</td><td>ZFPM1</td><td>OAS2</td></tr><tr><td>TUBA1C</td><td>OR8K5</td><td>THSD3</td><td>ATP6V0C</td><td>RAB22A</td><td>AP1B1</td><td>CTAGE6</td><td>C6orf26</td><td>ESR1</td></tr><tr><td>UPK3B</td><td>ROBO4</td><td>TMEFF1</td><td>KIAA1279</td><td>ZFP36L1</td><td>GRINA</td><td>YTHDF3</td><td>TMCC1</td><td>UBE1DC1</td></tr><tr><td>C6orf15</td><td>PDE6A</td><td>PEO1</td><td>TMEM52</td><td>PARP1</td><td>GSS</td><td>RDH11</td><td>STXBP1</td><td>ACLY</td></tr><tr><td>TMSB10</td><td>TUBB</td><td>LIPK</td><td>HRC</td><td>C20orf111</td><td>OMA1</td><td>NCAPH2</td><td>GPX2</td><td>BPY2C</td></tr><tr><td>ZNF324 TAS2R9</td><td>CDC27</td><td>CCNB2</td><td>CNOT7</td><td>BIRC3</td><td>GAL3ST3</td><td>PLEKHM1</td><td>SPOCD1</td><td>PENK</td></tr></table>

Table 16. Detailed gene importance (mean ± standard error). Importance represents mean mask size (Hide&Seek, INVASE, REAL-x), mean importance scores (SHAP, LIME), or mean selection frequency (L2X).