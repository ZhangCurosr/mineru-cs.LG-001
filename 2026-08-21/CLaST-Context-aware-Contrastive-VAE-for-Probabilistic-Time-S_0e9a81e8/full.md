# CLaST: Context-aware Contrastive VAE for Probabilistic Time Series Forecasting

1<sup>st</sup> Alexander Marusov

Applied AI Institute

Moscow, Russia

4<sup>th</sup> Aleksandr Yugay Applied AI Institute Moscow, Russia

2<sup>nd</sup> Dmitry Anikin

Applied AI Institute

Moscow, Russia

5<sup>th</sup> Petr Sokerin

Applied AI Institute Moscow, Russia

7<sup>th</sup> Alexey Zaytsev

Applied AI Institute

Moscow, Russia

3<sup>rd</sup> Vitaliy Pozdnyakov Applied AI Institute Moscow, Russia

6<sup>th</sup> Ilya Kuleshov

Applied AI Institute

Moscow, Russia

Abstract—Probabilistic forecasting models are widely used for time series forecasting in domains such as energy systems, finance, medicine, and transportation. In recent years, deep generative models have shown strong results on probabilistic forecasting, yet many conventional approaches struggle to capture internal temporal dependencies, leading to latent representations with limited expressive power. To address this limitation, we propose CLaST, a VAE framework for probabilistic multivariate time series forecasting. Unlike existing generative models, CLaST learns embeddings that preserve contextual similarity between observations through our contrastive loss function. Experiments across nine widely adopted benchmarks demonstrate that CLaST consistently surpasses strong baseline methods. In short-term forecasting tasks, our approach achieves improvements of up to 16.4% in CRPS and 14.4% in NMAE over the second-best method. Furthermore, in long-term prediction CLaST attains superior overall performance, exceeding the second-best method by up to 48.6% and 25.1% in CRPS and NMAE, respectively.

Index Terms—probabilistic time series forecasting, similarity learning, generative models, contrastive learning

## I. INTRODUCTION

Time series analysis presents a unique set of challenges: temporal dependencies, non-stationarity, multimodality, missing data, and the need to quantify predictive uncertainty. A central task in this domain is time-series forecasting, in which models are required to infer future dynamics from imperfect and often volatile historical data. In a wide range of applications, leveraging deep generative models has consistently improved predictive performance. Recurrent (DeepAR [1]), attention-based (PatchTST [2]), diffusion (TSDiff-Cond [3]), and flow-based (TFM [4]) approaches offer strong predictive performance. VAE-based methods like K<sup>2</sup>VAE [5] and LaST [6] further structure latents using dynamical systems or trend/seasonal decomposition.

However, learning informative latent representations remains a significant challenge. While estimating mutual information (MI) between input observations and their representations offers a promising approach to enhance embedding informativeness, existing neural MI estimators—such as MINE [7], CLUB [8], and STUB [6]—are computationally intensive, prone to high variance, and exhibit slow convergence in highdimensional settings [9]–[11]. Such instability undermines the optimization process and ultimately restricts the quality of the learned representations.

Furthermore, current approaches often fail to capture latent temporal dependencies embedded within the contextual structure of sequential data. Here contextual similarity reflects how closely two observations align in terms of their semantics. For instance, July 1 and July 2 temperature readings share a summer heat contextual regime, whereas July 1 and January 1 belong to distinct seasonal regimes. Crucially, this notion diverges from conventional statistical correlation: variables may co-vary due to shared confounders yet remain contextually distinct. For example, electricity demand and ice cream sales often peak simultaneously in summer, but they describe fundamentally different physical processes and thus occupy separate contextual domains.

To address these limitations, we propose CLaST, a variational autoencoder framework for probabilistic time series forecasting that explicitly preserves contextual structure. To adequately capture contextual similarity, we consider a class of time series in which the covariance between observations depends solely on the temporal lag δ (i.e., $\begin{array} { r l } { \mathrm { c o v } ( \mathbf { x } _ { t } , \mathbf { x } _ { t + \delta } ) = } & { { } } \end{array}$ f(δ)), while a time-varying mean maintains overall nonstationarity. Although classic statistical literature has implicitly addressed such processes, to the best of our knowledge, they lack a unified nomenclature. We therefore introduce the term Lag-Invariant Non-stationary Time Series (LINTS) process. Unlike strict stationarity assumptions that frequently limit applicability to real-world data, the LINTS formulation retains realistic temporal dynamics through its evolving mean while imposing only a single, interpretable constraint on the dependence structure. Under the LINTS process our approach leverages similarity matrices—a cornerstone of contrastive self-supervised learning [12]—to model the degree of closeness between observations in time. Specifically, we constrain the latent similarity matrices to mirror the structural patterns inherent in the input sequences. Extending the LaST architecture, we replace conventional mutual information (MI) estimators, which are prone to instability, with a contrastivebased penalty that preserves contextual similarity between objects. This formulation simultaneously enhances training stability and facilitates a more robust capture of underlying similarity dependencies.

The main contributions of this work are as follows:

• CLaST: Context-aware Contrastive VAE for Probabilistic Time Series Forecasting. To enhance probabilistic forecasting performance, we introduce a variational autoencoder (VAE) framework grounded in a novel objective function that explicitly accounts for the contextual similarity between samples. Our method ensures that the learned embeddings preserve meaningful sequential dependencies while effectively capturing intrinsic temporal relationships.

• Theoretical properties of our loss function. We derive the proposed objective function analytically under the LINTS process. Notably, Theorem III.3 establishes the theoretical optimality of our loss function within the specified class of similarity matrices. By formalizing lag-structured priors within the optimization objective, the loss injects structural knowledge into the learning pipeline. Given sufficiently high-capacity encoders, this formulation compels the model to extract representations that preserve the lag-aware information of the input time series.

• Empirical validation. Extensive experiments on both short- and long-term forecasting across nine established datasets from diverse domains, including Electricity, ETT, and Weather, demonstrate that CLaST generalizes effectively across heterogeneous time-series modalities. For the short-term setting it consistently outperforms strong baselines, achieving up to 16.4% relative improvement in CRPS and 14.4% in NMAE. Regarding the long-term predictions our approach surpasses second-best result up to 48.6% and 25.1% in CRPS and NMAE correspondingly. We provide the code implementation of the proposed method:https://anonymous.4open.science/r/CLaST-3407/README.md.

## II. RELATED WORK

a) Classic and deep learning methods.: Classic approaches (ARIMA [13], VAR [14], SVR [15], exponential smoothing [16]) struggle with high-dimensional, nonlinear dynamics [17]. Deep learning models address this: RNNs like LSTM [18] enable probabilistic forecasting (e.g., DeepAR [1]), while Transformers like PatchTST [2] capture long-range dependencies via patch-based attention. Conversely, linear models like DLinear [19] remain strong baselines by decomposing series into trend and seasonal components.

b) Variational Autoencoders.: VAEs [20] offer probabilistic forecasting but often yield unstructured latents. Structured variants address this: LaST [6] factorizes latents into trend/seasonal components with MI regularization, while K<sup>2</sup>VAE [5] combines VAEs with Koopman operators and Kalman filtering to explicitly model temporal dynamics and improve long-horizon accuracy.

c) Ordinary Differential Equations.: Continuous-time models (Neural ODE [21], Latent ODE [22], GRU-ODE [23]) handle irregular sampling naturally but incur high computational costs and lack latent regularization. Flow-based alternatives like Trajectory Flow Matching (TFM) [4] efficiently learn conditional trajectory derivatives via continuous normalizing flows without stochastic simulation. DeNOTS [24] introduces a negative feedback mechanism to stabilize vector fields, thereby enabling the modeling of long-term dependencies.

d) Diffusion models.: Diffusion [25] generates probabilistic forecasts by reversing a noise process conditioned on past observations. While effective for multi-horizon forecasting [26] and imputation [27], they are computationally intensive and lack latent interpretability. TSDiff-Cond [3] improves scalability by integrating S4 layers with convolutional channel mixing in an SSSD-based architecture.

Following the literature review, we chose LaST [6], K<sup>2</sup>VAE [5], DeNOTS [24], DeepAR [1], TFM [4], TSDiff [3], and PatchTST [2] as baseline models due to their strong reported performance across their respective categories.

## III. METHOD

To present our proposed method, we first start with the problem statement. Then we outline the overall pipeline of the CLaST approach, followed by a detailed description of its key components. The notation appears throughout the text as needed for clarity. For a complete list of notation used throughout the paper, we refer the reader to the online documentation available on our GitHub repository.

## A. Problem Statement.

We study probabilistic forecasting for multivariate time series. Let $\mathbf { \bar { x } } _ { t } \in \mathbb { R } ^ { F }$ denote the observations of F variables at time t. Given a historical window $\mathbf { x } _ { t - N + 1 : t }$ of length N, the task is to predict the future sequence $\mathbf { x } _ { t + 1 : t + L }$ over a horizon L. In the probabilistic setting, the model outputs a predictive distribution $p ( \mathbf { x } _ { t + 1 : t + L } \mid \mathbf { x } _ { t - N + 1 : t } )$ , which captures both future values and their uncertainty.

## B. CLaST

We now introduce CLaST (Contrastive LaST), a generative– contrastive model for probabilistic time series forecasting. CLaST builds upon the LaST architecture [6] while fundamentally revising its latent regularization strategy.

To address data non-stationarity, LaST employs dedicated seasonal and trend encoders and decoders, providing informative representations for each component. Thus, LaST separately produces latent representations for trend $\mathbf { Z } ^ { t }$ and seasonality $\mathbf { Z } ^ { s }$ , along with a predictor module that uses the Discrete Fourier Transform (DFT) for seasonal forecasting.

A central component of LaST is its loss function. The evidence lower bound (ELBO) term $\mathcal { L } _ { \mathrm { E L B O } }$ ensures that the learned trend and seasonal embeddings are sufficiently informative to both reconstruct the input time series and enable accurate predictions. The authors employ a slightly modified MINE-based mutual information estimator to maximize the amount of information preserved from the original input X in the trend and seasonal representations, $\mathbf { Z } ^ { t }$ and Z<sup>s</sup>. The corresponding objectives are denoted as $I _ { \mathrm { M I N E } } ( { \mathbf { X } } , { \mathbf { Z } } ^ { t } )$ and $I _ { \mathrm { M I N E } } ( \mathbf { X } , \mathbf { Z } ^ { s } )$ , respectively. Meanwhile, the term $I _ { \mathrm { S T U B } } ( \mathbf { Z } ^ { s } , \mathbf { Z } ^ { t } )$ promotes disentanglement between the trend and seasonal representations, encouraging them to capture complementary, non-overlapping information. The final loss function for LaST is defined in eq. 1.

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { L a S T } } = \mathcal { L } _ { \mathrm { E L B O } } + I _ { \mathrm { M I N E } } ( \mathbf { X } , \mathbf { Z } ^ { s } ) + I _ { \mathrm { M I N E } } ( \mathbf { X } , \mathbf { Z } ^ { t } ) } \\ { - I _ { \mathrm { S T U B } } ( \mathbf { Z } ^ { s } , \mathbf { Z } ^ { t } ) . } \end{array}\tag{1}
$$

However, as discussed earlier, MI estimations could be unstable in practice. We replace the MI-based objective with a contextual similarity alignment loss that encourages latent representations to preserve the contextual structure of the input time series. The central idea is to align the estimated similarity matrix $\widehat { \mathbf { G } } ( \mathbf { Z } )$ with the data-induced similarity matrix ${ \textbf { G } } =$ $\mathbf { G } ( \mathbf { X } )$ , which is called ground truth similarity matrix. Here, $\widehat { \mathbf { G } } ( \mathbf { Z } )$ is computed from the trend and seasonal representations $\mathbf { Z } ^ { t }$ and $\mathbf { Z } ^ { s }$ , while G serves as the similarity matrix.

Although the similarity matrices estimated from trend embeddings, $\widehat { \mathbf { G } } ( \mathbf { Z } ^ { t } )$ , and seasonal embeddings, $\widehat { \mathbf { G } } ( \mathbf { Z } ^ { s } )$ , both approximate the same ground-truth matrix $\mathbf { G } ,$ the corresponding representations remain separable in the embedding space, as illustrated in our experiments<sup>1</sup>. This separation arises because the LaST framework explicitly models trend and seasonal components as distinct components.

All other parts include the LaST’s core architecture, in particular, separate trend and seasonal latents, and the DFTbased predictor would remain the same. The full CLaST objective with three terms is then given by

$$
\mathcal { L } _ { \mathrm { C L a S T } } = \mathcal { L } _ { \mathrm { E L B O } } + \left. \mathbf { G } - \widehat { \mathbf { G } } ( \mathbf { Z } ^ { t } ) \right. _ { \mathcal { F } } ^ { 2 } + \left. \mathbf { G } - \widehat { \mathbf { G } } ( \mathbf { Z } ^ { s } ) \right. _ { \mathcal { F } } ^ { 2 } .\tag{2}
$$

Here,

$$
\left\| \mathbf { G } - { \widehat { \mathbf { G } } } ( \mathbf { Z } ) \right\| _ { \mathcal { F } } ^ { 2 } = \sum _ { i , j } { \left( g _ { i j } - { \widehat { g } } _ { i j } ( f _ { \pmb { \theta } } ) \right) ^ { 2 } } .\tag{3}
$$

Detailed definitions and discussions of these estimated and ground truth similarity matrices are postponed to Section III-D.

## C. Method Overview

In Section III-D, we formally define the ground-truth G and estimated similarity matrices $\widehat { \mathbf { G } }$ that underpin our loss function.

<sup>1</sup>https://anonymous.4open.science/r/CLaST-3407/figs/trend seasonal.png

Our approach for constructing the ground truth similarity matrix G from time series is described in Section III-E. We start from a correlation matrix R, which captures the statistical dependencies in the data. Since correlations do not directly reflect similarity relationships, we introduce a mapping that transforms the correlation space into a similarity space, yielding the matrix G with elements $g _ { i j }$ presented in eq. (6).

In Section III-F, we derive a closed-form expression for the estimated similarity matrix $\widehat { \mathbf { G } }$ by formulating and solving a corresponding optimization problem. In Theorem III.3 we define the elements $\widehat { g } _ { i j } ( f _ { \pmb { \theta } } )$ of Gb . We prove that this closedform solution is the unique global minimizer.

Substituting the ground truth matrix G (eq. (6)) and the estimated matrix $\widehat { \mathbf { G } } ( \mathbf { Z } )$ (Thm. III.3) into the equation (3) yields the final loss function (2).

## D. Theoretical Background

To explain our method, we consider a multivariate time series consisting of N observations $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { N }$ , where each $\mathbf { x } _ { i } ~ \in ~ \mathbb { R } ^ { F }$ represents the F-dimensional observation at time step i. The time series can thus be represented as a matrix $\mathbf { X } \in \mathbb { R } ^ { N \times F }$

To utilize a loss function based on pairwise comparisons, we introduce two complementary matrices: a ground truth similarity matrix and an estimated similarity matrix that are core concepts in contrastive self-supervised learning [12].

The ground truth similarity matrix $\textbf { G } = \{ g _ { i j } \} _ { i , j = 1 } ^ { N } \in$ $\mathbb { R } ^ { N \times N }$ encodes prior knowledge about the contextual structure of the time series. Each entry $g _ { i j }$ represents the desired level of similarity closeness between observations collected at time moments $i \in [ 1 ; N ]$ and $j ~ \in ~ [ 1 ; N ]$ . This matrix, therefore, defines the target relational structure that the learned representations are expected to preserve.

The estimated similarity matrix $\widehat { \bf G } ( { \bf Z } ) \ = \ \{ \widehat { g } _ { i j } \} _ { i , j = 1 } ^ { N } \ \in$ $\mathbb { R } ^ { N \times N }$ , on the other hand, captures the similarities inferred by the model itself. The matrix $\widehat { \mathbf { G } } ( \mathbf { Z } )$ is computed from the embedding matrix $\mathbf Z = \{ \mathbf z _ { i } \} _ { i = 1 } ^ { N } \in \mathbb { R } ^ { N \times h }$ , where each representation $\mathbf { z } _ { i } = f _ { \theta } ( \mathbf { x } _ { i } ) \in \mathbb { R } ^ { h }$ is obtained by applying an encoder function $f _ { \theta }$ to the input signal $\mathbf { x } _ { i }$ . As a result, Gb reflects how close the samples are in the learned representation space.

By comparing $\widehat { \mathbf { G } }$ with the ground truth matrix G, we obtain a natural objective for training the encoder. Minimizing the discrepancy between these two matrices encourages the learned embeddings to preserve the contextual relationships specified by the prior knowledge encoded in G:

$$
\sum _ { i , j } ( g _ { i j } - \widehat { g } _ { i j } ( f _ { \pmb \theta } ) ) ^ { 2 }  \operatorname* { m i n } _ { \pmb \theta } .\tag{4}
$$

We use objective function 4 as our penalty in eq. (3) for trend and seasonal components. Defining this penalty requires specifying the ground truth matrix G (Section III-E) and the estimated matrix $\widehat { \mathbf { G } }$ (Section III-F).

## E. Ground Truth Similarity Matrix G

1) Correlation As A Contextual Proxy: To define the matrix G, we will rely on statistical correlations between observations. In particular, we consider a multivariate time series $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { N }$ modeled as a Lag-Invariant Non-stationary Time Series (LINTS). In this framework, the inter-observation covariance depends exclusively on the temporal lag: cov $\begin{array} { r l } { \mathbf { \nabla } \cdot ( \mathbf { x } _ { i } , \mathbf { x } _ { i + \delta } ) = } \end{array}$ f(δ). While the dependence kernel is shift-invariant, the process itself remains non-stationary due to its time-varying mean.

The sample correlation matrix R is constructed as follows.

1) Centering and normalizing. For each time point $i \in$ $[ 1 ; N ]$ , subtract a location estimate $\begin{array} { r } { \overline { { \mathbf { x } } } _ { i } = \frac { 1 } { F } \sum _ { j = 1 } ^ { F } x _ { i } ^ { ( j ) } } \end{array}$ to obtain the centered vector, then normalize it to unit Euclidean length:

$$
\tilde { \mathbf { x } } _ { i } = \frac { \mathbf { x } _ { i } - \overline { { \mathbf { x } } } _ { i } } { \lVert \mathbf { x } _ { i } - \overline { { \mathbf { x } } } _ { i } \rVert _ { 2 } } .
$$

This yields observations $\tilde { \mathbf { x } } _ { i }$ that lie on the unit sphere in $\mathbb { R } ^ { \bar { F } }$

2) Computing pairwise similarities. The correlation matrix Re is obtained as the Gram matrix of $\begin{array} { r l } { \widetilde { \mathbf { X } } } & { { } = } \end{array}$ $[ \tilde { \mathbf { x } } _ { 1 } ^ { \top } , \ldots , \tilde { \mathbf { x } } _ { N } ^ { \top } ] :$

$$
\begin{array} { r } { \widetilde { \mathbf { R } } = \widetilde { \mathbf { X } } \widetilde { \mathbf { X } } ^ { \top } \in \mathbb { R } ^ { N \times N } . } \end{array}
$$

3) Symmetric Toeplitz structure. For each lag $\delta \in \mathsf { \Gamma }$ $\{ 0 , \ldots , N - 1 \}$ we compute the median empirical cor relation

$$
r _ { \delta } = \mathrm { m e d i a n } \left\{ \widetilde { r } _ { i j } : | i - j | = \delta \right\} .
$$

Replacing each entry $\widetilde { r } _ { i j }$ by $r _ { | i - j | }$ yields a symmetric Toeplitz matrix R whose $( i , j )$ -entry depends only on the time lag $| i - j |$

$$
r _ { i j } = r _ { | i - j | } , \qquad i , j = 1 , \ldots , N .
$$

Thus the entire correlation structure is summarized by the N numbers $r _ { 0 } , r _ { 1 } , \dots , r _ { N - 1 }$ , with $r _ { 0 } = 1$ since $\mathbf { R } _ { i i } = 1$ . The Toeplitz property reflects that covariance depends only on lag, while the symmetry $\widetilde { \mathbf { R } } _ { i j } = \widetilde { \mathbf { R } } _ { j i }$ follows from the definition of correlation.

We now define a matrix G derived from the correlation matrix R. Each entry $g _ { i j } ~ \in ~ [ 0 , 1 ]$ quantifies the degree to which observations collected at time points i and j play similar roles within the underlying dynamical process. Because the correlations $r _ { i j }$ lie in [−1; 1], we need a mapping $m : [ - 1 ; 1 ] \  \ [ 0 ; 1 ]$ that converts correlation values into probabilities of similarity. Following [12], we set the main diagonal of G to zero, as these entries correspond to the trivial self-similarity of each sample.

We seek a principled mapping from correlations to similarity scores. Theorem III.1 is derived using classical Bayesian decision theory under Gaussian class-conditional assumptions. We show that for LINTS-process, the Bayes-optimal posterior mapping from the observed lag-τ correlation to contextual similarity is logistic (sigmoid). For simplicity, Theorem III.1 is stated for the univariate case.

Theorem III.1 (Optimal mapping from feature to similarity space). Let $\{ x _ { i } \} _ { i = 1 } ^ { N }$ be a LINTS-process: covariance depends only on the fixed lag $\delta \geq 0$ , so $\operatorname { c o v } ( x _ { i } , x _ { i + \delta } ) = f ( \delta )$ . Let $Y _ { \delta } \in$ $\{ 0 , 1 \}$ be a similarity indicator, and let $\rho _ { y } \in \left( - 1 , 1 \right)$ denote the population lag-δ correlation associated with $Y _ { \delta } = y$ Assume:

(A1) there exists continuously differentiable function $g \in C ^ { 1 } ( ( - 1 ; 1 ) )$ such that $z _ { \delta } : = g ( r _ { \delta } )$ satisfies 1 (z<sub>δ</sub> − µ<sub>y</sub>)<sup>2</sup> p(z<sub>δ</sub> | Y<sub>δ</sub> = y) = exp √<sub>2πσ2</sub> 2σ<sup>2</sup> $\mu _ { y } : = g ( \rho _ { y } ) , \quad \sigma ^ { 2 }$ independent of y;   
(A2) $\rho _ { 0 } \neq \rho _ { 1 } .$

Then, in the small-correlation regime $| r _ { \delta } | \ll 1 _ { \ L }$

$$
P ( Y _ { \delta } = 1 \mid r _ { \delta } ) = \sigma ( \alpha r _ { \delta } + \beta ) ,\tag{5}
$$

for some $\alpha , \beta \in \mathbb { R }$ , where $\sigma ( x ) = ( 1 + e ^ { - x } ) ^ { - 1 }$

Remark III.2 (Example of a valid transformation). A classical choice satisfying Assumption A1 is the Fisher z-transform

$$
g ( r ) = { \frac { 1 } { 2 } } \log { \frac { 1 + r } { 1 - r } } ,
$$

for which, under mild mixing conditions, $g ( r _ { \delta } )$ is asymptotically Gaussian with variance independent of the true correlation. This motivates the modeling assumption in (A1).

The proof of the Theorem III.1 can be found in the Appendix A.

2) Target contextual Similarity: The intuition underlying the connection between the data space and the similarity space, formalized in Theorem III.1, is illustrated in Figure 1. Accordingly, the entries of the true dependency matrix G are defined using a sigmoid function as follows:

$$
g _ { i j } = \left\{ \begin{array} { l l } { s _ { \mathrm { m i n } } + \left( 1 - s _ { \mathrm { m i n } } \right) \sigma ( \alpha r _ { | i - j | } ) , } & { i \neq j , } \\ { 0 , } & { i = j , } \end{array} \right.\tag{6}
$$

where $s _ { \mathrm { m i n } } \in ( 0 , 0 . 5 )$ specifies a lower bound on contextual affinity and $\alpha \in [ 1 , 1 0 ]$ controls the sensitivity of dependency to variations in correlation.

We introduce $s _ { \mathrm { m i n } }$ to reflect the assumption that even maximally negatively correlated vectors still share structural information and should not have zero contextual similarity. Our ablation studies F demonstrate that the sigmoid function provides the optimal mapping from correlation space to similarity space, a result validated both theoretically and empirically.

## F. Estimated Similarity Matrix $\widehat { \mathbf { G } }$

Now we need to derive an expression of the estimated similarity matrix $\widehat { \mathbf { G } }$ . To do this, we consider the optimization problem of learning an estimated similarity matrix $\widehat { \mathbf { G } } \in \mathbb { R } ^ { N \times N }$ from a symmetric distance matrix $\mathbf { D } \dot { = } \{ d _ { i j } \} _ { i , j = 1 } ^ { N } \in \mathbb { R } ^ { N \times N }$ where $d _ { i i } = 0$ . Here $d _ { i j } = d ( z _ { i } , z _ { j } )$ denotes the distance $( \mathrm { e . g . }$ euclidean) between representations $z _ { i }$ and $z _ { j }$ at time moments i and $j .$ The matrix $\widehat { \mathbf { G } }$ is assumed to be Toeplitz, i.e., its entries depend only on the temporal lag $| i - j |$

![](images/b1956c27b182a8b264df3ed5512f7fa0ab52bafe6480cfbbf9d8c2dd441af70c.jpg)  
Fig. 1: Connection between the original feature space and the contextual similarity space. In the feature space, $x _ { t }$ is the observation at the current time step, and $x _ { t + 1 }$ and $x _ { t + 2 }$ are the inputs at the subsequent time steps. In our formulation, the correlation coefficient r between normalized observations corresponds to the cosine of the angle between them $( r = \cos \theta )$ Correlations are mapped to contextual similarity scores via a sigmoid with a lower bound $s _ { \mathrm { m i n } }$ , yielding a smooth notion of contextual proximity.

Following [12], we cast the problem of estimating the similarity matrix Gb as a Laplacian estimation task in graph signal processing [28], [29]. We therefore obtain Gb by solving an optimization problem specifically tailored to time-series data, restricting the search space to G, the set of symmetric, non-negative Toeplitz matrices with a zero main diagonal.

$$
\operatorname { T r } ( \mathbf { D } \widehat { \mathbf { G } } ) + \mathcal { R } _ { \log } ( \widehat { \mathbf { G } } ) \to \operatorname* { m i n } _ { \widehat { \mathbf { G } } \in \mathcal { G } } ,\tag{7}
$$

where $\begin{array} { r } { \mathcal { R } _ { \log } ( \widehat { \mathbf { G } } ) = \tau \sum _ { i \neq j } \widehat { g } _ { i j } \left( \ln ( \widehat { g } _ { i j } ) - 1 \right) } \end{array}$ is a logarithmic regularizer.

Theorem III.3. (Necessary and sufficient conditions). Let D be the matrix of pairwise distances, and the regularizer $\begin{array} { r } { \mathcal { R } _ { \log } ( \widehat { \mathbf { G } } ) = \tau \sum _ { i \neq j } \widehat { g } _ { i j } ( \ln ( \widehat { g } _ { i j } ) - 1 ) } \end{array}$ . The matrix Gb with elements

$$
\widehat { g } _ { i j } = \exp \left( - \frac { \phi _ { \vert i - j \vert } } { \tau } \right) \mathbb { 1 } _ { \{ i \neq j \} } ,
$$

with $\begin{array} { r } { \phi _ { k } = \frac { 1 } { N - k } \sum _ { | i - j | = k } d _ { i j } } \end{array}$ is the unique global minimizer for the optimization problem (7).

The detailed derivation and proof of Theorem III.3 are given in Appendix B.

## IV. EXPERIMENTAL RESULTS

## A. Experiments Setup

a) Datasets: We evaluate our models on nine publicly available datasets that span several domains: ETT (ETTh1, ETTh2, ETTm1, ETTm2) [30], Traffic [31], Electricity<sup>2</sup>,

Weather<sup>3</sup>. Detailed information can be found in the $\mathsf { A p - }$ pendix C.

b) Metrics.: We evaluate the forecasts using two commonly used metrics: the Continuous Ranked Probability Score (CRPS) and the Normalized Mean Absolute Error (NMAE) that correspond to the quality of uncertainty estimation and forecasting correspondingly. The detailed descriptions of the metrics are in the Appendix D.

c) Baselines.: We compare our approach against several generative models, including modern $K ^ { \bar { 2 } } \mathrm { V A E }$ [5], LaST [6], TSDiff [3], and TFM [4]. Additionally, we include PatchTST [32], a point-wise forecasting model, for reference.

d) Backbone.: To choose an appropriate backbone encoder, we considered three types of backbone architectures: DLinear, MLP and PatchTST. The hyperparameters for the selected backbones are detailed in Appendix E. For both short- and long-term forecasting tasks, we consistently employ PatchTST as the backbone encoder.

## B. Main Results

a) Short-term forecasting $( N ~ = ~ 9 6 , L ~ = ~ 2 4 ) \colon$ For short-term prediction, all experiments are performed with a forecasting horizon of $L \ = \ 2 4$ and an context length of $N = 9 6$

As shown in Table I, CLaST delivers the best performance across several datasets, highlighting its effectiveness for probabilistic forecasting. It consistently outperforms the secondbest method on datasets ETTh1, ETTh2, ETTm1, ETTm2, and Weather, achieving relative improvements of 0.9–16.4% in CRPS and 4.4–14.4% in NMAE. Our comparative analysis reveals that when PatchTST is trained within the CLaST generative framework, it demonstrably surpasses its pointwisetrained counterpart across every benchmark dataset, as measured by the NMAE. The observed performance gains range from a minimum improvement of 5.86% on the Traffic dataset to a substantial maximum of 26.97% on ETTh2, underscoring the significant advantage conferred by our CLaST generative learning paradigm.

TABLE I: Comparison of short-term probabilistic time series forecasting performance (N = 96, L = 24) across various real-world datasets. Lower CRPS and NMAE scores indicate superior forecasting accuracy. Reported results show the mean values and standard errors, computed from three independent runs involving model retraining and evaluation. Boldface highlights the best-performing method, while underlining denotes the second-best result.
<table><tr><td>Dataset</td><td>Metric</td><td> $\mathrm { C L a S T \ ( o u r s ) }$ </td><td>LaST</td><td> $K ^ { 2 } \mathrm { { V A E } }$ </td><td>DeepAR</td><td>TFM</td><td>TSDiff</td><td>PatchTST</td><td>DeNOTS</td></tr><tr><td>Electricity</td><td>CRPS NMAE</td><td> $\underline { { 0 . 0 8 9 \pm 0 . 0 0 0 } }$   $\overline { { 0 . 1 0 8 \pm 0 . 0 0 0 } }$ </td><td> $0 . 1 4 4 \pm 0 . 0 2 1$   $0 . 1 6 6 \pm 0 . 0 2 2$ </td><td> $\mathbf { 0 . 0 8 2 \pm 0 . 0 0 0 }$   $\mathbf { 0 . 1 0 7 \pm 0 . 0 0 0 }$ </td><td> $0 . 1 1 7 \pm 0 . 0 0 1$   $0 . 1 5 1 \pm 0 . 0 0 2$ </td><td> $0 . 1 5 0 \pm 0 . 0 1 1$  0.201 ± 0.014</td><td> $0 . 1 1 5 \pm 0 . 0 0 6$   $0 . 1 5 0 \pm 0 . 0 0 8$ </td><td> $0 . 1 1 6 \pm 0 . 0 0 2$   $0 . 1 1 6 \pm 0 . 0 0 2$ </td><td> $0 . 2 0 1 \pm 0 . 0 0 2$   $0 . 2 7 4 \pm 0 . 0 0 1$ </td></tr><tr><td>Traffic</td><td>CRPS</td><td> $0 . 2 0 3 \pm 0 . 0 0 1$ </td><td> $0 . 3 0 9 \pm 0 . 0 1 1$ </td><td></td><td></td><td></td><td></td><td></td><td> $0 . 3 8 6 \pm 0 . 0 1 9$ </td></tr><tr><td></td><td>NMAE</td><td>0.241 ± 0.000</td><td> $0 . 3 6 0 \pm 0 . 0 1 0$ </td><td> $\mathbf { 0 . 1 6 4 \pm 0 . 0 0 1 }$   $\mathbf { 0 . 2 1 1 \pm 0 . 0 0 2 }$ </td><td> $0 . 2 0 7 \pm 0 . 0 0 1$   $0 . 2 6 4 \pm 0 . 0 0 2$ </td><td> $0 . 3 2 1 \pm 0 . 0 1 3$   $0 . 4 2 6 \pm 0 . 0 2 2$ </td><td> $0 . 2 0 8 \pm 0 . 0 0 4$   $0 . 2 5 3 \pm 0 . 0 0 4$ </td><td> $0 . 2 5 6 \pm 0 . 0 0 1$   $0 . 2 5 6 \pm 0 . 0 0 1$ </td><td> $0 . 5 0 0 \pm 0 . 0 3 2$ </td></tr><tr><td>ETTh1</td><td>CRPS</td><td> $\mathbf { 0 . 2 2 0 \pm 0 . 0 0 2 }$ </td><td> $0 . 2 5 4 \pm 0 . 0 0 4$ </td><td>0.222 ± 0.002</td><td></td><td></td><td> $0 . 3 7 3 \pm 0 . 0 4 1$ </td><td> $0 . 2 9 4 \pm 0 . 0 0 6$ </td><td> $0 . 5 0 3 \pm 0 . 0 0 8$ </td></tr><tr><td></td><td>NMAE</td><td>0.267 ± 0.003</td><td> $0 . 2 8 9 \pm 0 . 0 0 5$ </td><td>0.283 ± 0.006</td><td> $0 . 3 8 4 \pm 0 . 0 7 1$   $0 . 4 9 8 \pm 0 . 0 8 8$ </td><td> $0 . 5 4 0 \pm 0 . 0 1 8$   $0 . 6 2 3 \pm 0 . 0 2 6$ </td><td> $0 . 4 9 2 \pm 0 . 0 3 7$ </td><td> $0 . 2 9 4 \pm 0 . 0 0 6$ </td><td> $0 . 6 4 5 \pm 0 . 0 1 7$ </td></tr><tr><td>ETTh2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>CRPS NMAE</td><td> $\mathbf { 0 . 3 1 6 \pm 0 . 0 0 4 }$  0.371 ± 0.004</td><td> $0 . 4 6 0 \pm 0 . 0 5 8$  0.546 ± 0.077</td><td> $0 . 3 7 2 \pm 0 . 0 0 5$  0.405 ± 0.005</td><td> $1 . 1 6 8 \pm 0 . 0 9 1$  1.485 ± 0.138</td><td> $1 . 7 3 8 \pm 0 . 4 5 1$ </td><td> $0 . 9 8 9 \pm 0 . 2 1 3$ </td><td> $0 . 5 0 8 \pm 0 . 0 5 5$  0.508 ± 0.055</td><td> $1 . 9 2 0 \pm 0 . 3 9 8$  2.207 ± 0.421</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>1.892 ± 0.384</td><td> $1 . 2 8 1 \pm 0 . 2 4 5$ </td><td></td><td></td></tr><tr><td>ETTm1</td><td>CRPS</td><td> $\mathbf { 0 . 1 8 0 \pm 0 . 0 0 4 }$ </td><td> $0 . 2 1 7 \pm 0 . 0 0 4$ </td><td>0.186 ± 0.005</td><td> $0 . 3 5 9 \pm 0 . 0 0 8$ </td><td> $0 . 5 5 6 \pm 0 . 0 3 8$ </td><td> $0 . 2 9 9 \pm 0 . 0 3 6$ </td><td> $0 . 2 4 7 \pm 0 . 0 0 5$ </td><td> $0 . 5 6 1 \pm 0 . 0 4 8$ </td></tr><tr><td></td><td>NMAE</td><td> $\mathbf { 0 . 2 1 8 \pm 0 . 0 0 4 }$ </td><td> $0 . 2 5 7 \pm 0 . 0 0 8$ </td><td> $\overline { { 0 . 2 2 8 \pm 0 . 0 0 3 } }$ </td><td> $0 . 4 7 3 \pm 0 . 0 0 5$ </td><td> $0 . 6 3 4 \pm 0 . 0 3 8$ </td><td> $0 . 3 9 6 \pm 0 . 0 5 9$ </td><td> $0 . 2 4 7 \pm 0 . 0 0 5$ </td><td> $0 . 7 2 1 \pm 0 . 0 5 0$ </td></tr><tr><td>ETTm2</td><td>CRPS</td><td> $\mathbf { 0 . 2 2 9 \pm 0 . 0 0 0 }$ </td><td> $0 . 2 7 4 \pm 0 . 0 0 4$ </td><td>0.314 ± 0.027</td><td> $0 . 9 9 9 \pm 0 . 1 8 2$ </td><td>1.784 ± 0.121</td><td> $0 . 4 6 8 \pm 0 . 0 6 6$ </td><td>0.369 ± 0.022</td><td>1.591 ± 0.889</td></tr><tr><td></td><td>NMAE</td><td> $\mathbf { 0 . 2 7 1 \pm 0 . 0 0 2 }$ </td><td> $\overline { { 0 . 3 2 5 \pm 0 . 0 0 7 } }$ </td><td> $\underline { { 0 . 3 1 6 \pm 0 . 0 1 3 } }$ </td><td> $1 . 2 7 3 \pm 0 . 2 0 1$ </td><td> $1 . 9 0 4 \pm 0 . 1 5 1$ </td><td> $0 . 6 3 0 \pm 0 . 0 8 7$ </td><td> $0 . 3 6 9 \pm 0 . 0 2 2$ </td><td> $1 . 8 1 5 \pm 0 . 9 0 6$ </td></tr><tr><td>Weather</td><td>CRPS</td><td> $\mathbf { 0 . 2 9 6 \pm 0 . 0 1 1 }$ </td><td> $0 . 5 3 8 \pm 0 . 0 5 7$ </td><td> $0 . 6 8 3 \pm 0 . 0 1 1$ </td><td> $0 . 3 6 0 \pm 0 . 0 2 5$ </td><td> $1 . 0 0 8 \pm 0 . 3 2 2$ </td><td> $\underline { { 0 . 3 2 7 \pm 0 . 0 6 0 } }$ </td><td> $0 . 4 0 3 \pm 0 . 0 3 5$ </td><td> $0 . 4 3 4 \pm 0 . 0 2 2$ </td></tr><tr><td></td><td>NMAE</td><td> $\mathbf { 0 . 3 4 5 \pm 0 . 0 1 6 }$ </td><td> $0 . 6 7 6 \pm 0 . 0 9 0$ </td><td>0.491 ± 0.014</td><td> $0 . 4 2 5 \pm 0 . 0 1 7$ </td><td>1.206 ± 0.403</td><td>0.412 ± 0.093</td><td> $\underline { { 0 . 4 0 3 \pm 0 . 0 3 5 } }$ </td><td> $0 . 5 6 2 \pm 0 . 0 3 0$ </td></tr></table>

TABLE II: Comparison of long-term probabilistic time series forecasting performance $( N = 9 6 , L = 7 2 0 )$ across various real-world datasets. Lower CRPS and NMAE scores indicate superior forecasting accuracy. Reported results show the mean values and standard errors, computed from three independent runs involving model retraining and evaluation. Boldface highlights the best-performing method, while underlining denotes the second-best result.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Metric</td><td rowspan="2">CLaST (ours)</td><td rowspan="2">LaST</td><td rowspan="2">K²VAE</td><td rowspan="2">DeepAR</td><td rowspan="2">TFM</td><td rowspan="2">TSDiff</td><td rowspan="2">PatchTST</td><td rowspan="2">DeNOTS</td></tr><tr><td></td></tr><tr><td rowspan="3">Electricity</td><td>CRPS</td><td> $0 . 1 2 6 \pm 0 . 0 0 2$ </td><td> $0 . 1 4 6 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 1 1 5 \pm 0 . 0 0 6 }$ </td><td> $0 . 1 6 3 \pm 0 . 0 0 9$ </td><td> $0 . 2 1 0 \pm 0 . 0 1 7$ </td><td> $0 . 1 2 6 \pm 0 . 0 0 3$ </td><td> $0 . 1 5 6 \pm 0 . 0 0 2$ </td><td> $0 . 1 8 6 \pm 0 . 0 0 9$ </td></tr><tr><td>NMAE</td><td> $\overline { { 0 . 1 4 6 \pm 0 . 0 0 2 } }$ </td><td> $0 . 1 6 9 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 1 4 0 \overset { - } { \pm } 0 . 0 0 4 }$ </td><td> $0 . 2 0 4 \pm 0 . 0 1 1$ </td><td> $0 . 2 9 2 \pm 0 . 0 2 2$ </td><td> $0 . 1 6 5 \pm 0 . 0 0 4$ </td><td> $0 . 1 5 6 \pm 0 . 0 0 2$ </td><td> $0 . 1 8 6 \pm 0 . 0 0 9$ </td></tr><tr><td>CRPS</td><td>0.247 ± 0.001</td><td> $0 . 3 0 1 \pm 0 . 0 0 3$ </td><td>0.189 ± 0.002</td><td> $0 . 2 2 0 \pm 0 . 0 0 3$ </td><td> $0 . 4 7 3 \pm 0 . 0 1 8$ </td><td> $0 . 2 9 7 \pm 0 . 0 2 7$ </td><td> $0 . 2 7 6 \pm 0 . 0 0 2$ </td><td> $0 . 4 1 4 \pm 0 . 1 4 3$ </td></tr><tr><td rowspan="3">ETTh1</td><td>NMAE</td><td> $0 . 2 8 8 \pm 0 . 0 0 1$ </td><td> $0 . 3 4 3 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 2 4 3 \pm 0 . 0 0 0 }$ </td><td> $\overline { { 0 . 2 8 6 \pm 0 . 0 0 6 } }$ </td><td> $0 . 6 3 7 \pm 0 . 0 2 3$ </td><td> $0 . 3 6 8 \pm 0 . 0 3 3$ </td><td> $\underline { { 0 . 2 7 6 \pm 0 . 0 0 2 } }$ </td><td> $0 . 4 1 9 \pm 0 . 1 5 1$ </td></tr><tr><td>CRPS</td><td> $0 . 3 4 2 \pm 0 . 0 0 5$ </td><td> $0 . 5 7 3 \pm 0 . 0 0 9$ </td><td> $\mathbf { 0 . 3 3 4 \pm 0 . 0 0 8 }$ </td><td> $0 . 6 9 0 \pm 0 . 2 1 1$ </td><td> $0 . 5 9 0 \pm 0 . 0 6 0$ </td><td> $0 . 5 7 5 \pm 0 . 0 5 9$ </td><td>0.511 ± 0.001</td><td> $0 . 5 5 5 \pm 0 . 0 1 4$ </td></tr><tr><td>NMAE</td><td> $\mathbf { 0 . 3 7 9 \pm 0 . 0 0 4 }$ </td><td> $0 . 6 1 4 \pm 0 . 0 1 0$ </td><td> $\underline { { 0 . 4 2 8 \pm 0 . 0 1 2 } }$ </td><td> $0 . 8 3 0 \pm 0 . 2 4 0$ </td><td> $0 . 8 3 9 \pm 0 . 0 8 8$ </td><td> $0 . 7 0 1 \pm 0 . 0 5 4$ </td><td>0.511 ± 0.001</td><td> $0 . 7 5 0 \pm 0 . 0 1 9$ </td></tr><tr><td rowspan="3">ETTh2</td><td>CRPS</td><td> $\mathbf { 0 . 5 4 4 \pm 0 . 0 0 1 }$ </td><td> $2 . 5 3 1 \pm 0 . 0 5 4$ </td><td>1.000 ± 0.235</td><td> $2 . 3 3 2 \pm 0 . 0 1 3$ </td><td> $2 . 4 9 5 \pm 0 . 6 8 8$ </td><td> $2 . 2 5 4 \pm 0 . 1 0 6$ </td><td> $1 . 5 4 1 \pm 0 . 0 2 2$ </td><td> $2 . 0 8 2 \pm 0 . 1 3 6$ </td></tr><tr><td>NMAE</td><td> $\mathbf { 0 . 5 8 8 \pm 0 . 0 0 2 }$ </td><td> $2 . 6 0 2 \pm 0 . 0 3 8$ </td><td>0.769 ± 0.083</td><td> $2 . 6 1 1 \pm 0 . 0 5 3$ </td><td>3.204 ± 0.992</td><td>2.488 ± 0.104</td><td>1.541 ± 0.022</td><td>2.356 ± 0.104</td></tr><tr><td>CRPS</td><td> $0 . 3 0 9 \pm 0 . 0 0 0$ </td><td> $0 . 4 2 1 \pm 0 . 0 1 3$ </td><td> $\mathbf { 0 . 2 6 6 \pm 0 . 0 2 4 }$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">ETTm1 ETTm2</td><td>NMAE</td><td> $\overline { { 0 . 3 4 1 \pm 0 . 0 0 1 } }$ </td><td> $0 . 4 8 7 \pm 0 . 0 1 5$ </td><td> $\mathbf { 0 . 3 2 8 \pm 0 . 0 3 9 }$ </td><td> $0 . 4 6 6 \pm 0 . 0 7 3$   $0 . 6 2 3 \pm 0 . 0 6 9$ </td><td> $0 . 5 9 2 \pm 0 . 0 1 0$   $0 . 8 5 5 \pm 0 . 0 2 5$ </td><td> $0 . 4 8 3 \pm 0 . 0 2 0$   $0 . 6 2 9 \pm 0 . 0 2 5$ </td><td> $0 . 4 0 4 \pm 0 . 0 2 9$ </td><td> $0 . 5 7 6 \pm 0 . 0 0 9$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $0 . 4 0 4 \pm 0 . 0 2 9$ </td><td> $0 . 6 1 4 \pm 0 . 0 1 0$ </td></tr><tr><td>CRPS NMAE</td><td> $\mathbf { 0 . 5 0 8 \pm 0 . 0 0 1 }$   $\mathbf { 0 . 5 3 4 \pm 0 . 0 0 1 }$ </td><td>1.871 ± 0.241</td><td>0.989 ± 0.275</td><td> $2 . 0 6 4 \pm 0 . 6 4 6$ </td><td>2.899 ± 0.782</td><td> $1 . 6 4 2 \pm 0 . 4 5 0$ </td><td>1.240 ± 0.152</td><td>1.931 ± 0.016</td></tr><tr><td rowspan="3">Weather</td><td></td><td></td><td> $2 . 0 2 2 \pm 0 . 2 5 1$ </td><td> $\overline { { 0 . 7 1 3 \pm 0 . 1 0 5 } }$ </td><td> $2 . 3 8 8 \pm 0 . 6 3 3$ </td><td> $3 . 7 8 3 \pm 0 . 9 7 5$ </td><td> $1 . 9 1 5 \pm 0 . 4 3 0$ </td><td> $1 . 2 4 0 \pm 0 . 1 5 2$ </td><td> $2 . 0 0 8 \pm 0 . 0 3 6$ </td></tr><tr><td>CRPS</td><td> $0 . 4 4 7 \pm 0 . 0 0 1$ </td><td> $0 . 6 7 2 \pm 0 . 0 2 8$ </td><td> $0 . 6 6 1 \pm 0 . 0 0 7$ </td><td> $0 . 4 6 5 \pm 0 . 0 6 4$ </td><td> $1 . 3 4 4 \pm 0 . 2 4 6$ </td><td> $\mathbf { 0 . 3 7 5 \pm 0 . 0 3 0 }$ </td><td> $0 . 6 8 5 \pm 0 . 0 3 4$ </td><td> $0 . 8 0 3 \pm 0 . 0 7 1$ </td></tr><tr><td>NMAE</td><td> $\overline { { 0 . 4 8 2 \pm 0 . 0 0 1 } }$ </td><td> $0 . 8 3 5 \pm 0 . 0 4 3$ </td><td> $0 . 6 0 6 \pm 0 . 0 0 8$ </td><td> $0 . 5 5 5 \pm 0 . 0 6 7$ </td><td> $1 . 5 4 2 \pm 0 . 4 2 2$ </td><td> $\mathbf { 0 . 4 6 5 \pm 0 . 0 3 3 }$ </td><td> $0 . 6 8 5 \pm 0 . 0 3 4$ </td><td> $0 . 8 6 5 \pm 0 . 0 8 4$ </td></tr></table>

b) Long-term forecasting $( N = 9 6 , L = 7 2 0 ) \colon$ As presented in Table II, CLaST achieves superior CRPS and NMAE performance on the demanding ETTh2 and ETTm2 datasets, yielding reductions of up to 48.6% in CRPS on ETTh2 and 25.1% in NMAE on ETTm2. Integrating PatchTST into the CLaST framework yields substantial NMAE improvements across most benchmarks, with relative gains ranging from 6.41% on Electricity to 61.84% on ETTh2.

extended our evaluation to two additional energy-domain datasets, Solar and ERCOT [33]. Due to the substantial computational resources already expended on the main Table II for long-term experiments running architectures like diffusion-based (TSDiff), attention-based (PatchTST), flowbased (TFM), and ODE-based (e.g. DeNOTS) was not feasible within our remaining budget. We therefore restrict this comparison to the two most relevant VAE-based baselines: $K ^ { 2 } V A E ,$ the strongest empirical competitor on the main benchmarks, and LaST, the foundational architecture upon which CLaST is built. As demonstrated in Table III, CLaST consistently attains the best long-term forecasting results, outperforming the second-best model by margins of up to 12.9% in CRPS and 11% in NMAE. As shown in Figure 3 our CLasT effectively models both trend and seasonal components.

The exceptionally high dimensionality of the Traffic (862 channels) and Electricity (321 channels) datasets renders longterm forecasting particularly challenging. Consequently, we

c) Ablation studies: To verify that the observed performance gains stem from our regularization strategy rather than the choice of encoder, we conduct additional ablation experiments on three datasets (ETTh1, ETTh2, Weather) using different backbone architectures. With PatchTST as the backbone, CLaST improves both CRPS and NMAE by up to 46.8% and 45.6%, respectively <sup>4</sup>. When using the original FeedNet backbone from LaST, the gains reach 46.17% (CRPS) and 52.37% (NMAE) <sup>5</sup>.

TABLE III: Comparison of long-term probabilistic time series forecasting performance on Solar and ERCOT datasets $( N ~ = ~ 9 6 , ~ L ~ = ~ 7 2 0 )$ . Lower CRPS and NMAE scores indicate superior forecasting accuracy. Reported results show the mean values and standard errors, computed from three independent runs involving model retraining and evaluation. Boldface highlights the best-performing method, while underlining denotes the second-best result.
<table><tr><td>Dataset</td><td>Metric</td><td> $\mathbf { C L a S T } \ ( \mathrm { o u r s } )$ </td><td> $\mathrm { K } ^ { 2 } \mathrm { V A E }$ </td><td>LaST</td></tr><tr><td>ERCOT</td><td>CRPS</td><td> $\mathbf { 0 . 0 8 0 \pm 0 . 0 0 1 }$ </td><td> $0 . 0 8 4 \pm 0 . 0 0 1$ </td><td> $0 . 0 9 5 \pm 0 . 0 0 4$ </td></tr><tr><td></td><td>NMAE</td><td> $\mathbf { 0 . 0 9 0 \pm 0 . 0 0 2 }$ </td><td> $\overline { { 0 . 1 0 3 \pm 0 . 0 0 2 } }$ </td><td> $0 . 1 0 6 \pm 0 . 0 0 5$ </td></tr><tr><td>Solar</td><td>CRPS</td><td> $\mathbf { 0 . 5 2 9 \pm 0 . 0 1 0 }$ </td><td> $0 . 6 9 6 \pm 0 . 0 2 0$ </td><td> $0 . 6 0 7 \pm 0 . 0 1 5$ </td></tr><tr><td></td><td>NMAE</td><td> $\mathbf { 0 . 5 7 3 \pm 0 . 0 1 6 }$ </td><td> $0 . 6 3 7 \pm 0 . 0 0 6$ </td><td> $\overline { { 0 . 6 4 4 \pm 0 . 0 1 1 } }$ </td></tr></table>

We conducted a detailed ablation study to understand the model analysis. In particular to examine the effect of hyperparameter choices, we present a sensitivity analysis in Appendix G. Appendix G1 studies the effect of the latent space dimensionality while Appendix G2 analyzes the impact of the context length. Additionally, we study how well temporal structure is preserved in the latent space. Our results in $\mathsf { A p - }$ pendix H indicate that the learned embeddings retain temporal dependencies. The implementation details are provided in the Appendix E.

## V. DISCUSSION

How the proposed loss enhances forecasting. Accurate probabilistic forecasting requires capturing diverse temporal dynamics and quantifying uncertainty. When latent representations collapse, decoders produce over-smoothed predictions and degraded CRPS. Our loss prevents this by structuring the latent space to preserve lag-decay correlations, keeping temporally distinct inputs separable and enabling sharper, better-calibrated forecasts (Fig. 2, Fig. 3). The resulting gains in CRPS and NMAE directly reflect this principled temporal regularisation.

Theoretical limitations. CLaST assumes inter-observation covariance depends solely on temporal lag, not absolute time. This aligns with many real-world domains where dependency structures remain stable despite non-stationary means: climatological series with long-term trends, financial indicators under trend-stationary models [34], and industrial sensor data with invariant physical dynamics [35].

Practical limitations. The contextual similarity alignment loss incurs $\mathcal { O } ( N ^ { 2 } )$ complexity due to full similarity-matrix construction. In practice, this overhead is modest: LaST trains ${ \sim } 2 8 \%$ faster per epoch than CLaST <sup>6</sup>, yet CLaST delivers substantially better performance. As we mentioned earlier in our main results with PatchTST backbone, CLaST improves CRPS/NMAE by up to 46.8%/45.6% Using the FeedNet, gains reach 46.17%/52.37 This marginal computational cost is strongly justified by the consistent, backbone-agnostic improvements in predictive quality.

## VI. CONCLUSION

This paper presented CLaST, a Context-aware VAE framework for probabilistic time series forecasting. Standard generative objectives often fail to encourage informative latent representations, particularly in sequential settings. CLaST mitigates this issue by introducing a natural theoretically-grounded loss function that accounts for contextual similarity between observations. It transfers temporal structure from the input space to the latent space using theoretically grounded loss functions, ensuring that the embeddings preserve contextual structure of dependent data.

Across a range of benchmarks covering diverse domains such as Electricity, ETT, and Weather, the proposed method demonstrates consistent and meaningful gains over strong baselines. These improvements reflect CLaST’s ability to capture intrinsic temporal relationships while maintaining the flexibility of latent-variable models. In short-term forecasting tasks, our approach achieves improvements of up to 16.4% in CRPS and 14.4% in NMAE. Furthermore, for the longterm forecasting CLaST attains superior overall performance, exceeding the second-best method by up to 48.6% and 25.1% in CRPS and NMAE, respectively, indicating enhanced uncertainty quantification and reduced forecasting error without undermining prior modeling advances.

## REFERENCES

[1] D. Salinas, V. Flunkert, J. Gasthaus, and T. Januschowski, “Deepar: Probabilistic forecasting with autoregressive recurrent networks,” International journal offorecasting, vol. 36, no. 3, pp. 1181–1191, 2020.

[2] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in International Conference on Learning Representations, 2023.

[3] M. Kollovieh, A. F. Ansari, M. Bohlke-Schneider, J. Zschiegner, H. Wang, and Y. B. Wang, “Predict, refine, synthesize: Self-guiding diffusion models for probabilistic time series forecasting,” Advances in Neural Information Processing Systems, vol. 36, pp. 28 341–28 364, 2023.

[4] X. N. Zhang, Y. Pu, Y. Kawamura, A. Loza, Y. Bengio, D. Shung, and A. Tong, “Trajectory flow matching with applications to clinical time series modelling,” Advances in Neural Information Processing Systems, vol. 37, pp. 107 198–107 224, 2024.

[5] X. Wu, X. Qiu, H. Gao, J. Hu, B. Yang, and C. Guo, “K2vae: A koopman-kalman enhanced variational autoencoder for probabilistic time series forecasting,” arXiv preprint arXiv:2505.23017, 2025.

[6] Z. Wang, X. Xu, W. Zhang, G. Trajcevski, T. Zhong, and F. Zhou, “Learning latent seasonal-trend representations for time series forecasting,” Advances in Neural Information Processing Systems, vol. 35, pp. 38 775–38 787, 2022.

[7] M. I. Belghazi, A. Baratin, S. Rajeshwar, S. Ozair, Y. Bengio, A. Courville, and D. Hjelm, “Mutual information neural estimation,” in International conference on machine learning. PMLR, 2018, pp. 531–540.

[8] P. Cheng, W. Hao, S. Dai, J. Liu, Z. Gan, and L. Carin, “CLUB: A contrastive log-ratio upper bound of mutual information,” in Proceedings of the 37th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, H. D. III and A. Singh, Eds., vol. 119. PMLR, 13–18 Jul 2020, pp. 1779–1788. [Online]. Available: https://proceedings.mlr.press/v119/cheng20b.html

[9] D. Shin and H. Kim, “An experimental study of neural estimators of the mutual information between random vectors modeling power spectrum features,” EURASIP Journal on Advances in Signal Processing, vol. 2024, 01 2024.

[10] Q. Guo, J. Chen, D. Wang, Y. Yang, X. Deng, J. Huang, L. Carin, F. Li, and C. Tao, “Tight mutual information estimation with contrastive fenchel-legendre optimization,” Advances in Neural Information Processing Systems, vol. 35, pp. 28 319–28 334, 2022.

[11] C. Chan, A. Al-Bashabsheh, H. P. Huang, M. Lim, D. S. H. Tam, and C. Zhao, “Neural entropic estimation: A faster path to mutual information estimation,” arXiv preprint arXiv:1905.12957, 2019.

[12] R. Balestriero and Y. LeCun, “Contrastive and non-contrastive selfsupervised learning recover global and local spectral embedding methods,” in Advances in Neural Information Processing Systems (NeurIPS 2022), 2022, pp. 26 671–26 685.

[13] G. E. Box and D. A. Pierce, “Distribution of residual autocorrelations in autoregressive-integrated moving average time series models,” Journal of the American statistical Association, vol. 65, no. 332, pp. 1509–1526, 1970.

[14] B. Biller and B. L. Nelson, “Modeling and generating multivariate timeseries input processes using a vector autoregressive technique,” ACM Transactions on Modeling and Computer Simulation (TOMACS), vol. 13, no. 3, pp. 211–237, 2003.

[15] L.-J. Cao and F. E. H. Tay, “Support vector machine with adaptive parameters in financial time series forecasting,” IEEE Transactions on neural networks, vol. 14, no. 6, pp. 1506–1518, 2003.

[16] P. R. Winters, “Forecasting sales by exponentially weighted moving averages,” Management science, vol. 6, no. 3, pp. 324–342, 1960.

[17] M. Jin, H. Y. Koh, Q. Wen, D. Zambon, C. Alippi, G. I. Webb, I. King, and S. Pan, “A survey on graph neural networks for time series: Forecasting, classification, imputation, and anomaly detection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[18] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural computation, vol. 9, no. 8, pp. 1735–1780, 1997.

[19] A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are transformers effective for time series forecasting?” arXiv preprint arXiv:2205.13504, 2022.

[20] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” arXiv e-prints, pp. arXiv–1312, 2013.

[21] R. T. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud, “Neural ordinary differential equations,” Advances in neural information processing systems, vol. 31, 2018.

[22] Y. Rubanova, R. T. Chen, and D. K. Duvenaud, “Latent ordinary differential equations for irregularly-sampled time series,” Advances in neural information processing systems, vol. 32, 2019.

[23] E. De Brouwer, J. Simm, A. Arany, and Y. Moreau, “Gru-ode-bayes: Continuous modeling of sporadically-observed time series,” Advances in neural information processing systems, vol. 32, 2019.

[24] I. Kuleshov, E. Romanenkova, V. A. Zhuzhel, G. Boeva, E. Vorsin, and A. Zaytsev, “DeNOTS: Stable deep neural ODEs for time series,” in The Fourteenth International Conference on Learning Representations (ICLR), 2026. [Online]. Available: https://openreview.net/forum?id= SFoDJZ1sSk

[25] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840– 6851, 2020.

[26] K. Rasul, C. Seward, I. Schuster, and R. Vollgraf, “Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting,” in International conference on machine learning. PMLR, 2021, pp. 8857–8868.

[27] Y. Tashiro, J. Song, Y. Song, and S. Ermon, “Csdi: Conditional scorebased diffusion models for probabilistic time series imputation,” Advances in neural information processing systems, vol. 34, pp. 24 804– 24 816, 2021.

[28] X. Dong, D. Thanou, P. Frossard, and P. Vandergheynst, “Learning laplacian matrix in smooth graph signal representations,” IEEE Transactions on Signal Processing, vol. 64, no. 23, pp. 6160–6173, 2016.

[29] V. Kalofolias, “How to learn a graph from smooth signals,” in Artificial intelligence and statistics. PMLR, 2016, pp. 920–929.

[30] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, “Informer: Beyond efficient transformer for long sequence time-series forecasting,” in The Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Virtual Conference, vol. 35, no. 12. AAAI Press, 2021, pp. 11 106–11 115.

[31] G. Lai, W.-C. Chang, Y. Yang, and H. Liu, “Modeling long-and short-term temporal patterns with deep neural networks,” in The 41st international ACM SIGIR conference on research & development in information retrieval, 2018, pp. 95–104.

[32] Y. Nie, “A time series is worth 64words: Long-term forecasting with transformers,” arXiv preprint arXiv:2211.14730, 2022.

[33] O. Shchur, A. F. Ansari, C. Turkmen, L. Stella, N. Erickson, P. Guerron-Quintana, M. Bohlke-Schneider, and Y. Wang, “fev-bench: A realistic benchmark for time series forecasting,” Boston College Department of Economics, Tech. Rep., 2025.

[34] J. D. Hamilton, Time series analysis. Princeton university press, 2020.

[35] C. K. Williams and C. E. Rasmussen, Gaussian processes for machine learning. MIT press Cambridge, MA, 2006, vol. 2, no. 3.

[36] J. Zhang, X. Wen, Z. Zhang, S. Zheng, J. Li, and J. Bian, “Probts: Benchmarking point and distributional forecasting across diverse prediction horizons,” Advances in Neural Information Processing Systems, vol. 37, pp. 48 045–48 082, 2024.

[37] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, “Informer: Beyond efficient transformer for long sequence time-series forecasting,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, no. 12, 2021, pp. 11 106–11 115.

[38] H. Wu, J. Xu, J. Wang, and M. Long, “Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting,” Advances in neural information processing systems, vol. 34, pp. 22 419– 22 430, 2021.

[39] M. Zamo and P. Naveau, “Estimation of the continuous ranked probability score with limited information and applications to ensemble weather forecasts,” Mathematical Geosciences, vol. 50, no. 2, pp. 209–234, 2018.

[40] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, “Optuna: A nextgeneration hyperparameter optimization framework,” in Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, 2019, pp. 2623–2631.

## APPENDIX

Below, we provide the complete mathematical derivations, lemma proofs, and reductions used to obtain the final closedform solution.

## A. Optimal mapping from feature to similarity space

Proof. By Assumptions $\mathrm { A l - A } 2 , z _ { \delta } : = g ( r _ { \delta } )$ is Gaussian with class-conditional means $\mu _ { y } = g ( \rho _ { y } ) ~ ( \mu _ { 0 } \neq \mu _ { 1 } )$ and common variance $\sigma ^ { 2 }$ . The log-likelihood ratio is linear:

$$
\Lambda ( z _ { \delta } ) = \log \frac { p ( z _ { \delta } \mid Y _ { \delta } = 1 ) } { p ( z _ { \delta } \mid Y _ { \delta } = 0 ) } = \alpha z _ { \delta } + \beta ^ { \prime } , \quad \alpha : = \frac { \mu _ { 1 } - \mu _ { 0 } } { \sigma ^ { 2 } } .
$$

By Bayes’ rule, the posterior is $P ( Y _ { \delta } = 1 \mid z _ { \delta } ) = \sigma ( \alpha z _ { \delta } + \beta )$ with $\beta : = \beta ^ { \prime } + \log ( \pi _ { 1 } / \pi _ { 0 } )$ . For small r , a first-order Taylor expansion $g ( r _ { \delta } ) = g ( 0 ) + g ^ { \prime } ( 0 ) r _ { \delta } + o ( r _ { \delta } )$ yields

$$
\begin{array} { r l r } & { } & { P ( Y _ { \delta } = 1 \mid r _ { \delta } ) = \sigma ( \tilde { \alpha } r _ { \delta } + \tilde { \beta } ) + o ( r _ { \delta } ) , } \\ & { } & { \tilde { \alpha } : = \alpha g ^ { \prime } ( 0 ) , \quad \tilde { \beta } : = \beta + \alpha g ( 0 ) , } \end{array}
$$

completing the proof.

## B. Necessary and sufficient conditions

Proof. Since $\mathcal { G }$ consists of symmetric, non-negative Toeplitz matrices with zero diagonal, any $\hat { \textbf { G } } \in \mathcal G$ is parameterized by $\widehat { \boldsymbol { g } } = \left( \widehat { g } _ { 1 } , \ldots , \widehat { g } _ { N - 1 } \right)$ with $\widehat { g } _ { k } \ \geq \ 0 ,$ , where $\widehat { g } _ { i j } \ = \ \widehat { g } _ { | i - j | }$

Defining the lag-averaged distance $\begin{array} { r } { \phi _ { k } : = \frac { 1 } { N - k } \sum _ { | i - j | = k } d _ { i j } } \end{array}$ problem (7) reduces to

$$
\operatorname* { m i n } _ { { \widehat { g } } _ { k } > 0 } { \mathcal { I } } ( { \widehat { g } } ) = 2 \sum _ { k = 1 } ^ { N - 1 } ( N - k ) \left[ \phi _ { k } { \widehat { g } } _ { k } + \tau { \widehat { g } } _ { k } ( \ln { \widehat { g } } _ { k } - 1 ) \right] .\tag{8}
$$

$\mathcal { I }$ is separable and strictly convex on $\{ \widehat { g } _ { k } ~ > ~ 0 \}$ because each term has second derivative $2 \tau ( N - k ) / \widehat { g } _ { k } > 0$ . Setting $\partial \mathcal { I } / \partial \widehat { g } _ { k } = 0$ yields the unique stationary point

$$
\widehat { g } _ { k } ^ { \star } = \exp ( - \phi _ { k } / \tau ) , \qquad k = 1 , \ldots , N - 1 ,
$$

which, by strict convexity, is the unique global minimizer of (8). The corresponding Toeplitz matrix $\widehat { \mathbf { G } }$ is therefore the unique solution to (7).

□

## C. Datasets Description

We evaluate on nine standard multivariate time-series benchmarks: ETTh1/2 and ETTm1/2 (hourly/15-min transformer measurements, 2016–2018), Electricity (hourly consumption for 321 users, 2012–2014), Weather (10-min meteorological records, 2020), Traffic (hourly road occupancy across 862 sensors, 2015–2016), Solar (hourly energy consumption and weather data with solar irradiance and meteorological variables), and ERCOT (hourly electricity load across 8 U.S. regions, 2004–2021) [5], [33], [36]–[38]. These datasets span diverse temporal resolutions, dimensionalities, and seasonal patterns. To encode temporal context, we append 11 auxiliary features: sinusoidal embeddings (cos(2πt), sin(2πt)) for year, quarter, month, week, and day cycles, plus a normalized intersample interval $( t _ { i } - t _ { i - 1 } ) / \Delta t$ . All series are chronologically partitioned into 70% training, 10% validation, and 20% testing splits.

## D. Evaluation Metrics

We report forecast performance using two standard metrics: Normalized Mean Absolute Error (NMAE) for point accuracy and Continuous Ranked Probability Score (CRPS) for probabilistic calibration. NMAE is computed as

$$
\mathrm { N M A E } ( \boldsymbol { x } , \hat { \boldsymbol { x } } ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \frac { \sum _ { i = 1 } ^ { F } \left| x _ { i } ^ { n } - \hat { x } _ { i } ^ { n } \right| } { \sum _ { i = 1 } ^ { F } \left| x _ { i } ^ { n } \right| } ,
$$

where xˆ denotes the median of 100 sampled trajectories. CRPS measures the discrepancy between the predictive CDF F(z) and the observation x:

$$
\mathrm { C R P S } = \int _ { \mathbb { R } } \bigl ( F ( z ) - \mathbb { I } \{ x \leq z \} \bigr ) ^ { 2 } d z ,
$$

approximated via Monte Carlo sampling [39]. For both metrics, lower values indicate better performance.

## E. Implementation Details

We tune CLaST hyperparameters on the validation set via Optuna [40], searching $\alpha \in [ 1 , 1 0 ] , s _ { \mathrm { m i n } } \in ( 0 , 0 . 5 )$ , and $\tau \in$ (0.01, 10). We evaluate three encoder backbones:

DLinear: We modify the original architecture [19] to project trend/seasonal components into latent space (dim 64, or 128 for Electricity/Traffic) using a running-mean window of 25.

MLP: A two-layer ReLU MLP with hidden dimension tuned over {64, 128, 256} and fixed latent output dimension 64.

PatchTST: Separate PatchTST encoders [2] model trend and seasonal components, each followed by linear heads for latent mean/variance estimation. Patch length and stride are fixed to 16 and 8; latent dimension is 64 (128 for Electricity/Traffic). Remaining hyperparameters are optimized via Optuna<sup>7</sup> or follow the original implementation<sup>8</sup>.

## F. Mapping functions from correlation to similarity space

We evaluate the impact of different mappings from the correlation space to the similarity space. Specifically, we compare the theoretically motivated sigmoid mapping with alternative choices, including ReLU, absolute value, and a parabolic transformation. As shown in the Table IV, the sigmoid mapping consistently yields the best performance across both datasets (Electricity and Traffic) according to both CRPS and NMAE.

TABLE IV: Impact of mapping functions from correlation to similarity space on probabilistic forecasting performance for Electricity and Traffic datasets. Lower CRPS and NMAE indicate better performance. Results are reported as mean ± standard error
<table><tr><td>Dataset</td><td>Metric</td><td>Sigmoid</td><td>ReLU</td><td>Absolute</td><td>Parabola</td></tr><tr><td>Electricity</td><td>CRPS</td><td> $\mathbf { 0 . 1 0 7 \pm 0 . 0 0 0 }$ </td><td> $0 . 1 1 1 \pm 0 . 0 0 0$ </td><td> $0 . 1 1 1 \pm 0 . 0 0 1$ </td><td> $0 . 1 1 1 \pm 0 . 0 0 1$ </td></tr><tr><td></td><td>NMAE</td><td> $\mathbf { 0 . 1 3 2 \pm 0 . 0 0 0 }$ </td><td> $0 . 1 3 7 \pm 0 . 0 0 0$ </td><td> $0 . 1 4 0 \pm 0 . 0 0 1$ </td><td>0.139 ± 0.001</td></tr><tr><td>Traffic</td><td>CRPS</td><td> $\mathbf { 0 . 2 7 0 \pm 0 . 0 0 4 }$ </td><td> $0 . 2 7 4 \pm 0 . 0 0 1$ </td><td> $0 . 2 7 8 \pm 0 . 0 0 3$ </td><td> $0 . 2 8 5 \pm 0 . 0 1 2$ </td></tr><tr><td></td><td>NMAE</td><td> $\mathbf { 0 . 3 2 3 \pm 0 . 0 0 9 }$ </td><td> $0 . 3 3 2 \pm 0 . 0 0 1$ </td><td> $0 . 3 3 8 \pm 0 . 0 0 4$ </td><td> $0 . 3 4 5 \pm 0 . 0 1 3$ </td></tr></table>

## G. Sensitivity Analysis

1) Sensitivity to Latent Space Dimensionality: The dimensionality of the latent space is selected based on validation set optimization. However, it is important to assess the model’s sensitivity to this parameter, as a stable model should not exhibit significant performance degradation when using comparable dimensionalities. Our empirical results <sup>9</sup> demonstrate that CLaST achieves consistent forecasting performance across latent dimensions of 32, 64, 128, and 256.

2) Backbone Sensitivity to Context Length: All encoders are tested with different context lengths of 24, 48, 96, 192, and 384 time steps. The impact of the encoder choice and the context length is provided in our supplementary materials <sup>10</sup>. The Transformer-based CLaST model generally performs best, with the MLP backbone typically ranking second. However, for some long-horizon forecasts–particularly on the Traffic and ETTh1 datasets–the MLP model can outperform the

![](images/9f8b3c28a4d7ee2ea0428784c507853f2e863643d5e035622a7590532c515bd2.jpg)

![](images/8be18b288f7e9b586631858fc83dfc6b310ed202bb6ce5cc1605153998939d73.jpg)  
Fig. 2: Average cosine similarity between embeddings of input data across time steps for LaST and CLaST on the Weather dataset with context length and horizon $N = L = 2 4$

Transformer. On the other hand, for the Weather dataset, the Transformer model achieves better results than the MLP in certain long-horizon settings, while underperforming in others.

## H. Temporal Structure in Learned Embeddings

To better understand the effect of the proposed training objective on the learned representations, we analyze the temporal structure of the resulting embedding space.

Figure 2 presents the average pairwise cosine similarity of input embeddings across different temporal positions for LaST and CLaST. Both approaches exhibit a diagonal structure in the resulting similarity matrices, indicating that embeddings corresponding to temporally adjacent timestamps are more similar, while similarity decreases with increasing temporal distance. However, LaST shows uniformly high average cosine similarity across the matrix, suggesting overly smooth representations with limited discriminability between distant timestamps. In contrast, CLaST yields lower and more heterogeneous similarity values, leading to a more pronounced diagonal pattern and improved preservation of temporal structure, due to the contrastive regularization and temporal bias imposed by the training loss.

![](images/20d4b02a5c4045d257e6656ce3890ed15dca7de054d2aa14c3621a8ce7a96e26.jpg)  
(a) ETTm2.

![](images/9cc3c61e38b8035352b832ab23da912cf99d47e83259d60f845f6fce59c65ac0.jpg)  
(b) Electricity.  
Fig. 3: CLaST decomposed forecasts $( N = 9 6 , L = 7 2 0 )$ Each figure shows ground truth, seasonal, and trend with 90% confidence intervals from 100 samples.