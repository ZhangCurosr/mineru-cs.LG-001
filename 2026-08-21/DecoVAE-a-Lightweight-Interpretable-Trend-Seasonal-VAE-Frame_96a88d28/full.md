# DecoVAE: a Lightweight Interpretable Trend-Seasonal VAE Framework for Efficient Probabilistic Time Series Forecasting

1<sup>st</sup> Alexander Marusov

Applied AI Institute

Moscow, Russia

2<sup>nd</sup> Dmitry Anikin

Applied AI Institute

Moscow, Russia

3<sup>rd</sup> Alexey Zaytsev

Applied AI Institute

Moscow, Russia

Abstract—Probabilistic time series forecasting remains challenging, largely because modeling distinct trend and seasonal dynamics requires specialized approaches. Existing methods often fail to capture the unique inner properties of these components, lack interpretability, or suffer from heavy memory and runtime overhead. To address these limitations, we propose Deco-VAE, a lightweight interpretable trend-seasonal VAE framework that explicitly decomposes time series into trend and seasonal components by applying domain-specific inductive biases. The trend stream enforces structural smoothness using a differential regularizer on the latent trajectory, analogous to the Hodrick-Prescott filter. Concurrently, the seasonal stream operates in the frequency domain via a complex Gaussian VAE, natively capturing the amplitude and phase of periodic patterns. Extensive evaluations across seven real-world benchmarks show that DecoVAE consistently outperforms strong baselines. It achieves reductions of up to 14.96% in CRPS and 23.30% in NMAE for short-term forecasting, and up to 52.68% and 26.51% for longterm horizons. Crucially, DecoVAE yields these accuracy gains while remaining highly efficient, reducing model weight by up to 93% and accelerating speed by up to 74% compared to the second-best method.

Index Terms—probabilistic time series forecasting, trendseasonal decomposition, variational autoencoder, lightweight interpretable models

## I. INTRODUCTION

Probabilistic time series forecasting is essential for energy, finance, weather, and traffic applications [1]–[3], yet remains challenging due to the need for accurate uncertainty quantification and effective modeling of trend (long-term, low-frequency movements) and seasonality (periodic oscillations) [4].

Current methods face several critical limitations: most treat trend and seasonality uniformly, ignoring their distinct structures; time-domain models struggle with global periodicity [5], while frequency-domain approaches suffer from spectral mixing that degrades both point forecasts and uncertainty estimates [6], [7]. Although explicit decomposition improves accuracy [8], it intensifies the quality-computation trade-off, as high-fidelity generative models like diffusion networks remain prohibitively slow due to iterative sampling [9]–[12]. Consequently, developing a computationally efficient framework that accurately captures both temporal dynamics for probabilistic forecasting remains an open challenge.

To address this gap, we introduce DecoVAE (Trend-Seasonal Decomposition Variational Autoencoder), which explicitly separates time series into trend and seasonal components, modeling each via specialized mechanisms. We enforce trend smoothness through Hodrick–Prescott-inspired regularization, while modeling seasonality in the complex domain to naturally capture periodicity. Integrated into a variational framework, DecoVAE simultaneously captures structured temporal dynamics and provides well-calibrated probabilistic forecasts.

The main contributions of this work are summarized as follows:

• DecoVAE: a Lightweight Trend-Seasonal VAE Framework for Efficient Probabilistic Time Series Forecasting We propose DecoVAE, a variational autoencoder that performs explicit trend–seasonal decomposition with component-specific subnetworks, enabling accurate modeling of distinct temporal dynamics while yielding wellcalibrated predictive distributions.

Theoretically Grounded Trend Regularization. We enforce trend smoothness via a Hodrick–Prescott (HP)- inspired penalty [13]. Theorem III.1 shows equivalence to the classical HP objective under linear decoding, bridging statistical filtering with deep probabilistic modeling. Theorem III.2 further guarantees that, for Lipschitz decoders, the extracted trend exhibits negligible high-frequency energy, ensuring faithful long-term dynamic capture.

• Complex-Domain Seasonal Modeling. We model seasonality in the complex frequency domain to natively encode phase and amplitude information. Inspired by FITS [14], our complex-valued transformations overcome representational limits of real-valued time-domain processing, enhancing multi-periodic pattern learning within the variational framework.

• Empirical validation. Evaluations across seven benchmarks confirm DecoVAE’s robustness across diverse forecasting scenarios. In short-term settings, it achieves up to 14.96% CRPS and 23.30% NMAE improvements over strong baselines; for long-term forecasts, gains reach 52.68% CRPS and 26.51% NMAE. Code is available at GitHub<sup>1</sup>.

## II. RELATED WORK

This section reviews decomposition-based probabilistic forecasting methods by focusing on three key aspects: (A) probabilistic forecasting models, (B) techniques for decomposing and modeling trend and seasonal components (C), and loss functions that exploit decomposed temporal structure (D).

## A. Probabilistic Time Series Forecasting

Generative models form the backbone of modern probabilistic time series forecasting. DeepAR [15] establishes a strong autoregressive baseline via global latent variables. VAE-based approaches capture uncertainty, with LaST [16] using a factorized latent space to disentangle components, and $K ^ { 2 } \mathrm { V A E }$ [17] integrating Koopman operator theory for long-term forecasting. Diffusion models like TSDiff-Cond [18] offer flexible generation but at high computational cost. Alternatively, Trajectory Flow Matching (TFM) [19] provides more efficient solution via continuous normalizing flows. Building on Neural ODE-based trajectory modeling, DeNOTS [20] further improves stability and expressiveness while better preserving long-term temporal dependencies. Despite their accuracy, these methods typically trade off computational efficiency and interpretability, motivating a unified framework excelling across all three dimensions.

## B. Trend-Seasonal Decomposition Techniques

Classical methods like STL [21] and moving-average decomposition [22], [23] separate trend (long-term direction) from seasonality (fixed-interval repetitions). Recent work enhances this: FedFormer [24] adaptively combines multi-scale moving averages, while Learnable Decomposition [25] replaces fixed kernels with trainable convolutions. Empirical studies confirm that explicit trend-seasonality separation consistently improves forecasting performance [26].

## C. Latent Modeling of Trend and Seasonality

Modern architectures embed component-specific inductive biases. Autoformer [27] applies asymmetric attention (complex for seasonality, simple for trend). FEDformer [24] operates in the frequency domain via sparse Fourier mode selection. Alternative designs include xPatch [28] (separate MLP/convolution streams), TimeMixer (multi-scale pooling), TimesNet [29] (2D period-aligned CNNs), and LaST [16] (VAE with smoothness/periodicity constraints on latent components). These approaches balance expressiveness, efficiency, and interpretability while natively modeling distinct temporal structures.

## D. Loss Function

Loss design critically affects decomposition-based forecasting. LaST [16] applies auxiliary objectives to enforce smooth trend latents and periodic seasonal latents. DBLoss [30] decomposes both predictions and targets via exponential smoothing, computing separate trend/seasonal losses. This explicit decomposition in the objective accelerates convergence and improves accuracy over standard MSE, even when the model internally disentangles components.

## III. METHOD

Problem statement. To address the probabilistic time series forecasting problem, we consider a multivariate time series with F features, where $\mathbf { x } _ { \tau } \in \mathbb { R } ^ { F }$ represents the feature vector at timestamp τ . The objective is to map historical observations $\mathbf { X } _ { \tau - T + 1 : \tau }$ (lookback window T) to future values $\mathbf { X } _ { \mathcal { T } + 1 : \mathcal { T } + L }$ (prediction horizon L).

Method pipeline. To tackle the probabilistic time series forecasting problem we propose DecoVAE (Fig. 1), a dualstream Variational Autoencoder. It is designed to explicitly model the trend and seasonal components of a time series using structurally appropriate inductive biases:

• a smooth latent trajectory for the trend;

• a frequency-domain variational representation for seasonality.

Thus, DecoVAE operates through the following pipeline:

A) Trend-Seasonal Decomposition Techniques: The input series is decomposed into trend and seasonal components using moving average filtering.

B) Latent Modeling of Trend and Seasonality: Each component is modeled via a dedicated VAE stream. Specifically, the trend component is processed directly within the time domain, whereas the seasonal component is mapped into the frequency domain using a Fourier transform.

C) Forecasting Pipeline: Each stream generates independent predictions in its respective domain, which are combined to produce the final probabilistic forecast.

D) Loss Function: For each component, we designed a specific objective that accounts for the concrete specifics:

• The trend stream operates in the time domain with smoothness regularization applied to the latent trajectory, inspired by the Hodrick-Prescott filter.

• The seasonal stream operates in the frequency domain using complex-valued representations to naturally capture amplitude and phase information of periodic patterns.

The remainder of this section details each stage of the pipeline. We first examine input decomposition strategies for isolating trend and seasonal components, followed by a description of the specialized processing pathways applied to each. Finally, we present the prediction mechanism and formalize the corresponding loss function.

## A. Trend-Seasonal Decomposition Techniques

Given an input series $\mathbf { X } \in \mathbb { R } ^ { T \times F }$ with T time steps and F features, our goal is to decompose it into trend and seasonal components via operator $\mathcal { D } _ { t }$ , which extracts trend component of the X:

$$
{ \mathbf X } = { \mathbf X } _ { t } + { \mathbf X } _ { s } ,\tag{1}
$$

where $\mathbf { X } _ { t } = { \mathcal { D } } _ { t } ( \mathbf { X } )$ and $\mathbf { X } _ { s } = X - { \mathcal { D } } _ { t } ( \mathbf { X } )$ . Here, $\mathbf { X } _ { t } , \mathbf { X } _ { s } \in$ $\mathbb { R } ^ { T \times F }$ denote the trend and seasonal components respectively.

We use a moving average (MA) trend extractor for its simplicity, with FEDFormer’s MoE and EMA compared in

![](images/b499033eea28aaf751f5f5136f0c79214fac032bf758677b4fe84e5d580f0195.jpg)  
Fig. 1: The DecoVAE architecture decomposes the input time series X into trend $\left( \mathbf { X } _ { t } \right)$ and seasonal $( \mathbf { X } _ { s } )$ components. These are processed independently, accounting for trend smoothness and seasonal frequencies. The final prediction $\widehat { \mathbf Y }$ is synthesized as the sum of the forecasted trend $\widehat { \mathbf Y } _ { t }$ and seasonality $\widehat { \mathbf { Y } } _ { s }$

Appendix D. The separated components feed into specialized VAE pathways: the trend stream applies smoothness constraints to the latent trajectory, while the seasonal stream employs frequency-domain processing to capture periodicity.

## B. Latent Modeling of Trend and Seasonality

We outline below our approach for modeling trend and seasonal components to yield their respective representations.

## 1) Trend Component:

a) Trend Encoding: The trend $\mathbf { X } _ { t }$ is processed by a shared encoder applied independently along the channel dimension:

$$
\mathbf { H } = \mathrm { M L P } _ { s h a r e d } ^ { e n c } ( \mathbf { X } _ { t } ) , \quad \mathbf { H } \in \mathbb { R } ^ { T \times d } ,\tag{2}
$$

where $T$ is the context length and d is the embedding dimension. Following the standard VAE structure, we parameterize a Gaussian latent distribution. The mean and variance are computed from the encoder output:

$$
\pmb { \mu } = \mathrm { L i n e a r } _ { \pmb { \mu } } ( \mathbf { H } ) \in \mathbb { R } ^ { T \times d } ,\tag{3}
$$

$$
\pmb { \sigma } ^ { 2 } = \mathrm { S o f t p l u s } \left( \mathrm { L i n e a r } _ { \sigma } ( \mathbf { H } ) \right) \in \mathbb { R } ^ { T \times d } .\tag{4}
$$

b) Reparameterization and Sampling: We use the standard reparameterization trick to obtain latent trend representations:

$$
\begin{array} { r } { \mathbf { Z } _ { t } = \pmb { \mu } + \pmb { \sigma } \odot \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , I ) , } \end{array}\tag{5}
$$

where $\mathbf { Z } _ { t } ~ \in ~ \mathbb { R } ^ { T \times d }$ is the latent trajectory, and ⊙ denotes element-wise multiplication.

c) Decoding: The latent trajectory is decoded to reconstruct the trend component. This is performed via MLP operating along the feature dimension (from embedding space to the original feature space):

$$
\widehat { \mathbf { X } } _ { t } = \mathrm { M L P } _ { \mathrm { t r e n d } } ^ { \mathrm { d e c } } ( \mathbf { Z } _ { t } ) \in \mathbb { R } ^ { T \times F } .\tag{6}
$$

2) Seasonal Component: To model seasonal dynamics, we operate directly in the frequency domain. This choice is motivated by the fact that periodic patterns correspond to localized structures in the spectrum, making them easier to represent and manipulate compared to the time domain [31], [32].

We evaluated three alternative parameterizations for the frequency-domain seasonal component, ultimately selecting a complex-valued FITS-style approach [14]. By modeling the signal directly within the complex Fourier domain via complex-valued linear layers, this method preserves the Fourier transform’s native structure while jointly encoding amplitude and phase into a unified latent representation. As detailed in Appendix C1, this variant consistently outperforms alternatives, providing the most efficient spectral modeling and validating its adoption as our default.

a) Frequency Domain Encoding: Let $\mathbf { X } _ { s } \in \mathbb { R } ^ { T \times F }$ denote the seasonal component. After applying normalization, we transform seasonal component to the frequency domain:

$$
\mathbf { X } _ { s } ^ { ( f r e q ) } = \mathcal { F } ( \mathbf { X } _ { s } ) \in \mathbb { C } ^ { K \times F } ,\tag{7}
$$

where $\mathcal { F }$ denotes the FFT and $K = \lfloor T / 2 \rfloor + 1$ represents the number of frequency bins. The spectrum is encoded by a shared complex-valued linear encoder applied independently across frequency bins:

$$
\mathbf { H } = \operatorname { L i n e a r } _ { e n c } ( \mathbf { X } _ { s } ^ { ( f r e q ) } ) \in \mathbb { C } ^ { K \times d } .\tag{8}
$$

b) Complex Latent Distribution: We parameterize a complex Gaussian distribution to preserve both amplitude and phase information. The mean is computed directly from the complex encoder output via complex linear layer. The variance, which must be real and positive, is computed from the concatenated real ℜ(H) and imaginary ℑ(H) parts:

$$
\pmb { \mu } = \mathrm { L i n e a r } _ { \mu } ( \mathbf { H } ) \in \mathbb { C } ^ { K \times d } ,\tag{9}
$$

$$
\pmb { \sigma } ^ { 2 } = \mathrm { S o f t p l u s } \left( \operatorname { L i n e a r } _ { \sigma } ( [ \Re ( \mathbf { H } ) , \Im ( \mathbf { H } ) ] ) \right) \in \mathbb { R } ^ { K \times d } .\tag{10}
$$

Sampling is performed via complex reparameterization:

$$
\mathbf { Z } _ { s } = \mu + \sqrt { \frac { \pmb { \sigma } ^ { 2 } } { 2 } } \odot ( \epsilon _ { r } + i \epsilon _ { i } )\tag{11}
$$

where $\mathbf { Z } _ { s } \in \mathbb { C } ^ { K \times d }$ and $\epsilon _ { r } , \epsilon _ { i } \sim \mathcal { N } ( 0 , I )$ . Symbol ⊙ denotes element-wise multiplication.

c) Spectral Decoding: The latent representation is decoded back to the frequency domain:

$$
\widehat { \mathbf { X } } _ { s } ^ { ( f r e q ) } = \operatorname { L i n e a r } _ { d e c } ( \mathbf { Z } _ { s } ) \in \mathbb { C } ^ { K \times F } .\tag{12}
$$

## C. Forecasting Pipeline

At inference time, the model generates probabilistic forecasts by sampling from each stream and combining their predictions.

1) Trend Forecasting: The latent trend trajectory $\mathbf { Z } _ { t }$ is projected to the prediction horizon using two MLPs operating on temporal and feature dimensions:

$$
\widehat { \mathbf { Z } } _ { t } = \mathrm { M L P } _ { 1 } ^ { p r e d } ( \mathbf { Z } _ { t } ) \in \mathbb { R } ^ { L \times d }\tag{13}
$$

$$
\widehat { \mathbf { Y } } _ { t } = \mathrm { M L P } _ { 2 } ^ { p r e d } ( \widehat { \mathbf { Z } } _ { t } ) \in \mathbb { R } ^ { L \times F } .\tag{14}
$$

Unlike the decoder which reconstructs the observed window, these prediction-specific MLPs learn temporal extrapolation from context length $T$ to forecast horizon $L ,$ accommodating potential distribution shifts between past and future.

2) Seasonal Forecasting: For forecasting a future window of length L, we use a separate predictor that upsamples the latent representation to $M \ = \ \lfloor L / 2 \rfloor \ + \ 1$ frequency bins corresponding to the prediction horizon:

$$
\widehat { \mathbf { Z } } _ { s } = \mathrm { L i n e a r } _ { 1 } ^ { p r e d } ( \mathbf { Z } _ { s } ) \in \mathbb { C } ^ { M \times d } .\tag{15}
$$

This representation is decoded to the output channels:

$$
\widehat { \mathbf { Y } } _ { s } ^ { ( f r e q ) } = \mathrm { L i n e a r } _ { 2 } ^ { p r e d } ( \widehat { \mathbf { Z } } _ { s } ) \in \mathbb { C } ^ { M \times F } .\tag{16}
$$

The final seasonal forecast is obtained by inverse Fourier transform and and subsequent denormalization:

$$
\widehat { Y } _ { s } = \mathcal { F } ^ { - 1 } ( \widehat { \mathbf { Y } } _ { s } ^ { ( f r e q ) } ) \in \mathbb { R } ^ { L \times F } .\tag{17}
$$

Such architecture enables efficient frequency-domain interpolation for forecasting, where the learned latent representation captures the essential seasonal patterns that can be extrapolated to future time horizons through spectral upsampling.

3) Final Prediction: The final forecast combines both components:

$$
\widehat { \mathbf Y } = \widehat { \mathbf Y } _ { t } + \widehat { \mathbf Y } _ { s } , \quad \widehat { \mathbf Y } \in \mathbb R ^ { L \times F } .\tag{18}
$$

Probabilistic predictions are obtained by sampling multiple times from the latent distributions of both streams.

## D. Loss Function

The model is trained by minimizing a composite loss function that combines the objectives for the trend and seasonal streams.

1) Trend Loss: To enforce smoothness of trend, we apply a differential regularizer on this latent trajectory, penalizing its first and second derivatives (velocity and acceleration). This approach is analogous to the Hodrick-Prescott filter applied in the latent space. The trend regularizer $\mathcal { R } _ { t r e n d }$ has the form:

$$
\mathcal { R } _ { t r e n d } = \alpha _ { 1 } \mathcal { R } _ { 1 } ( \mathbf { Z } _ { t } ) + \alpha _ { 2 } \mathcal { R } _ { 2 } ( \mathbf { Z } _ { t } )\tag{19}
$$

where $\mathcal { R } _ { 1 }$ and $\mathcal { R } _ { 2 }$ are the normalized squared norms of the first and second finite differences:

$$
\mathcal { R } _ { 1 } ( \mathbf { Z } _ { t } ) = \frac { 1 } { T - 1 } \sum _ { \tau = 1 } ^ { T - 1 } \Vert \mathbf { z } _ { t , \tau + 1 } - \mathbf { z } _ { t , \tau } \Vert ^ { 2 } ,\tag{20}
$$

$$
\mathcal { R } _ { 2 } ( \mathbf { Z } _ { t } ) = \frac { 1 } { T - 2 } \sum _ { \tau = 2 } ^ { T - 1 } \| \mathbf { z } _ { t , \tau + 1 } - 2 \mathbf { z } _ { t , \tau } + \mathbf { z } _ { t , \tau - 1 } \| ^ { 2 } .\tag{21}
$$

The hyperparameters $\alpha _ { 1 } > 0$ and $\alpha _ { 2 } > 0$ control the contribution of velocity and curvature regularization, respectively. Both terms are necessary: the second derivative regularization allows piecewise smooth trends with potential jumps, while only first derivative regularization leads to linear trends. Their combination produces nonlinear but smooth trajectories appropriate for real-world time series.

Thus, the trend stream is trained by combining reconstruction error, KL divergence, and the smoothness regularizer:

$$
\mathcal { L } _ { t r e n d } = \mathcal { L } _ { r e c } ^ { t r e n d } + \mathcal { L } _ { K L } ^ { t r e n d } + \mathcal { R } _ { t r e n d } ,\tag{22}
$$

where:

$\mathcal { L } _ { r e c } ^ { t r e n d } = \| \mathbf { X } _ { t } - \widehat { \mathbf { X } } _ { t } \| _ { 2 } ^ { 2 }$ is the time-domain reconstruction loss.

$\mathcal { L } _ { K L } ^ { t r e n d } = \mathrm { K L } \big ( q _ { \phi } ( \mathbf { Z } _ { t } ) \| p ( \mathbf { Z } _ { t } ) \big )$ is the KL divergence for the trend latent variables.

$\mathcal { R } _ { t r e n d }$ is the latent smoothness regularizer.

The proposed trend regularization has theoretical properties. Specifically, Theorem III.1 establishes that, under a linear trend decoder, our objective function closely approximates the classical Hodrick–Prescott loss.

Theorem III.1 (Hodrick–Prescott Filter Approximation in Latent Space). Assume the trend decoder is linear in the temporal domain, i.e., $\mathcal { G } _ { t } ( \mathbf { z } _ { t , \tau } ) = W _ { t } \mathbf { z } _ { t , \tau }$ with $W _ { t } \in \mathbb { R } ^ { F \times d }$ and let $\alpha _ { 1 } = 0 .$ . Define the latent objective functional:

$$
J [ Z _ { t } ] : = \sum _ { \tau = 1 } ^ { T } \| X _ { t , \tau } - W _ { t } z _ { t , \tau } \| ^ { 2 } + \alpha _ { 2 } \sum _ { \tau = 2 } ^ { T - 1 } \| \Delta ^ { 2 } z _ { t , \tau } \| ^ { 2 } .\tag{23}
$$

Then, for any latent sequence $Z _ { t } ,$ the discrepancy between the classical Hodrick–Prescott functional $J _ { \mathrm { H P } } [ W _ { t } Z _ { t } ]$ and $J [ Z _ { t } ]$ satisfies the two-sided bound:

$$
\begin{array} { r l r } {  { \big ( \lambda _ { \mathrm { H P } } \sigma _ { \operatorname* { m i n } } ^ { 2 } ( W _ { t } ) - \alpha _ { 2 } \big ) \sum _ { \tau = 2 } ^ { T - 1 } \| \Delta ^ { 2 } z _ { t , \tau } \| ^ { 2 } } } \\ & { } & { \leq J _ { \mathrm { H P } } [ W _ { t } Z _ { t } ] - J [ Z _ { t } ] } \\ & { } & { \leq \alpha _ { 2 } ( \frac { \sigma _ { \operatorname* { m a x } } ^ { 2 } ( W _ { t } ) } { \sigma _ { \operatorname* { m i n } } ^ { 2 } ( W _ { t } ) } - 1 ) \sum _ { \tau = 2 } ^ { T - 1 } \| \Delta ^ { 2 } z _ { t , \tau } \| ^ { 2 } , } \end{array}\tag{24}
$$

TABLE I: Comparison of short-term probabilistic time series forecasting performance $( T = 9 6 , L = 2 4 )$ . Lower CRPS and NMAE scores indicate better performance. Results are reported as mean ± standard error, computed from three independent runs involving model retraining and evaluation. Boldface highlights the best-performing method, while underlining denotes the second-best result.
<table><tr><td>Dataset</td><td>Metric</td><td> $\mathrm { D e c o V A E ~ ( o u r s ) }$ </td><td> $K ^ { 2 } \mathrm { V A E }$ </td><td>LaST</td><td>DeepAR</td><td>TFM</td><td>TSDiff</td><td>DeNOTS</td></tr><tr><td rowspan="2">Electricity</td><td>CRPS</td><td> $0 . 0 8 8 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 0 8 2 \pm 0 . 0 0 0 }$ </td><td> $0 . 1 4 4 \pm 0 . 0 2 1$ </td><td> $0 . 1 1 7 \pm 0 . 0 0 1$ </td><td> $0 . 1 5 0 \pm 0 . 0 1 1$ </td><td> $0 . 1 1 5 \pm 0 . 0 0 6$ </td><td> $0 . 2 0 1 \pm 0 . 0 0 2$ </td></tr><tr><td>NMAE</td><td> $\mathbf { 0 . 1 0 4 \pm 0 . 0 0 1 }$ </td><td> $\underline { { 0 . 1 0 7 \pm 0 . 0 0 0 } }$ </td><td> $0 . 1 6 6 \pm 0 . 0 2 2$ </td><td> $0 . 1 5 1 \pm 0 . 0 0 2$ </td><td> $0 . 2 0 1 \pm 0 . 0 1 4$ </td><td> $0 . 1 5 0 \pm 0 . 0 0 8$ </td><td> $0 . 2 7 4 \pm 0 . 0 0 1$ </td></tr><tr><td rowspan="3">Traffic</td><td>CRPS</td><td> $0 . 1 9 4 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 1 6 4 \pm 0 . 0 0 1 }$ </td><td> $0 . 3 0 9 \pm 0 . 0 1 1$ </td><td> $0 . 2 0 7 \pm 0 . 0 0 1$ </td><td> $0 . 3 2 1 \pm 0 . 0 1 3$ </td><td> $0 . 2 0 8 \pm 0 . 0 0 4$ </td><td> $0 . 3 8 6 \pm 0 . 0 1 9$ </td></tr><tr><td>NMAE</td><td> $\overline { { 0 . 2 2 2 \pm 0 . 0 0 1 } }$ </td><td> $\mathbf { 0 . 2 1 1 \pm 0 . 0 0 2 }$ </td><td> $0 . 3 6 0 \pm 0 . 0 1 0$ </td><td> $0 . 2 6 4 \pm 0 . 0 0 2$ </td><td> $0 . 4 2 6 \pm 0 . 0 2 2$ </td><td> $0 . 2 5 3 \pm 0 . 0 0 4$ </td><td> $0 . 5 0 0 \pm 0 . 0 3 2$ </td></tr><tr><td>CRPS</td><td> $\mathbf { 0 . 2 1 7 \pm 0 . 0 0 5 }$ </td><td> $0 . 2 2 2 \pm 0 . 0 0 2$ </td><td> $0 . 2 5 4 \pm 0 . 0 0 4$ </td><td> $0 . 3 8 4 \pm 0 . 0 7 1$ </td><td> $0 . 5 4 0 \pm 0 . 0 1 8$ </td><td> $0 . 3 7 3 \pm 0 . 0 4 1$ </td><td> $0 . 5 0 3 \pm 0 . 0 0 8$ </td></tr><tr><td rowspan="2">ETTh1</td><td>NMAE</td><td> $\mathbf { 0 . 2 6 2 \pm 0 . 0 0 2 }$ </td><td> $\overline { { 0 . 2 8 3 \pm 0 . 0 0 6 } }$ </td><td> $0 . 2 8 9 \pm 0 . 0 0 5$ </td><td> $0 . 4 9 8 \pm 0 . 0 8 8$ </td><td> $0 . 6 2 3 \pm 0 . 0 2 6$ </td><td> $0 . 4 9 2 \pm 0 . 0 3 7$ </td><td> $0 . 6 4 5 \pm 0 . 0 1 7$ </td></tr><tr><td>CRPS</td><td> $\mathbf { 0 . 3 1 9 \pm 0 . 0 0 5 }$ </td><td> $0 . 3 7 2 \pm 0 . 0 0 5$ </td><td> $0 . 4 6 0 \pm 0 . 0 5 8$ </td><td> $1 . 1 6 8 \pm 0 . 0 9 1$ </td><td> $1 . 7 3 8 \pm 0 . 4 5 1$ </td><td> $0 . 9 8 9 \pm 0 . 2 1 3$ </td><td> $1 . 9 2 0 \pm 0 . 3 9 8$ </td></tr><tr><td rowspan="2">ETTh2</td><td>NMAE</td><td> $\mathbf { 0 . 3 8 2 \pm 0 . 0 0 5 }$ </td><td> $\overline { { 0 . 4 0 5 \pm 0 . 0 0 5 } }$ </td><td> $0 . 5 4 6 \pm 0 . 0 7 7$ </td><td> $1 . 4 8 5 \pm 0 . 1 3 8$ </td><td> $1 . 8 9 2 \pm 0 . 3 8 4$ </td><td> $1 . 2 8 1 \pm 0 . 2 4 5$ </td><td> $2 . 2 0 7 \pm 0 . 4 2 1$ </td></tr><tr><td>CRPS</td><td> $0 . 1 8 8 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 1 8 6 \pm 0 . 0 0 5 }$ </td><td> $0 . 2 1 7 \pm 0 . 0 0 4$ </td><td> $0 . 3 5 9 \pm 0 . 0 0 8$ </td><td> $0 . 5 5 6 \pm 0 . 0 3 8$ </td><td></td><td> $0 . 5 6 1 \pm 0 . 0 4 8$ </td></tr><tr><td rowspan="2">ETTm1</td><td>NMAE</td><td> $\mathbf { 0 . 2 2 5 \pm 0 . 0 0 2 }$ </td><td> $\underline { { 0 . 2 2 8 \pm 0 . 0 0 3 } }$ </td><td> $0 . 2 5 7 \pm 0 . 0 0 8$ </td><td> $0 . 4 7 3 \pm 0 . 0 0 5$ </td><td> $0 . 6 3 4 \pm 0 . 0 3 8$ </td><td> $0 . 2 9 9 \pm 0 . 0 3 6$   $0 . 3 9 6 \pm 0 . 0 5 9$ </td><td> $0 . 7 2 1 \pm 0 . 0 5 0$ </td></tr><tr><td>CRPS</td><td> $\mathbf { 0 . 2 3 3 \pm 0 . 0 0 1 }$ </td><td> $0 . 3 1 4 \pm 0 . 0 2 7$ </td><td> $\underline { { 0 . 2 7 4 \pm 0 . 0 0 4 } }$ </td><td> $0 . 9 9 9 \pm 0 . 1 8 2$ </td><td> $1 . 7 8 4 \pm 0 . 1 2 1$ </td><td></td><td></td></tr><tr><td rowspan="2">ETTm2</td><td>NMAE</td><td> $\mathbf { 0 . 2 8 6 \pm 0 . 0 0 1 }$ </td><td> $0 . 3 1 6 \pm 0 . 0 1 3$ </td><td> $\overline { { 0 . 3 2 5 \pm 0 . 0 0 7 } }$ </td><td> $1 . 2 7 3 \pm 0 . 2 0 1$ </td><td> $1 . 9 0 4 \pm 0 . 1 5 1$ </td><td> $0 . 4 6 8 \pm 0 . 0 6 6$   $0 . 6 3 0 \pm 0 . 0 8 7$ </td><td> $1 . 5 9 1 \pm 0 . 8 8 9$   $1 . 8 1 5 \pm 0 . 9 0 6$ </td></tr><tr><td></td><td> $\mathbf { 0 . 2 8 6 \pm 0 . 0 0 3 }$ </td><td> $0 . 6 8 3 \pm 0 . 0 1 1$ </td><td> $0 . 5 3 8 \pm 0 . 0 5 7$ </td><td> $0 . 3 6 0 \pm 0 . 0 2 5$ </td><td> $1 . 0 0 8 \pm 0 . 3 2 2$ </td><td></td><td></td></tr><tr><td rowspan="2">Weather</td><td>CRPS NMAE</td><td> $\mathbf { 0 . 3 1 6 \pm 0 . 0 0 1 }$ </td><td> $0 . 4 9 1 \pm 0 . 0 1 4$ </td><td> $0 . 6 7 6 \pm 0 . 0 9 0$ </td><td> $0 . 4 2 5 \pm 0 . 0 1 7$ </td><td> $1 . 2 0 6 \pm 0 . 4 0 3$ </td><td> $\underline { { 0 . 3 2 7 \pm 0 . 0 6 0 } }$   $\overline { { 0 . 4 1 2 \pm 0 . 0 9 3 } }$ </td><td> $0 . 4 3 4 \pm 0 . 0 2 2$   $0 . 5 6 2 \pm 0 . 0 3 0$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Avg. Rank</td><td>CRPS</td><td>1.429</td><td>2.286</td><td>3.714</td><td>4.286</td><td>6.429</td><td>3.571</td><td>6.286</td></tr><tr><td>NMAE</td><td>1.143</td><td>2.143</td><td>4.000</td><td>4.429</td><td>6.286</td><td>3.429</td><td>6.571</td></tr></table>

where $\sigma _ { \operatorname* { m i n } } ( W _ { t } )$ and $\sigma _ { \operatorname* { m a x } } ( W _ { t } )$ denote the smallest and largest singular values of $W _ { t } ,$ , respectively. By selecting $\lambda _ { \mathrm { H P } } =$ $\alpha _ { 2 } / \sigma _ { \operatorname* { m i n } } ^ { 2 } ( W _ { t } )$ , the lower bound vanishes, and the functional difference is controlled solely by the condition number of the decoder matrix.

The complete proof is provided in the supplementary repository<sup>2</sup>.

Building on this result, Theorem III.2 demonstrates that even under a nonlinear Lipschitz-continuous decoder, the reconstructed trend retains exclusively low-frequency spectral content. This establishes a formal guarantee of spectral disentanglement, ensuring that the trend component remains strictly smooth and free from high-frequency seasonal artifacts.

Theorem III.2 (Preservation of Spectral Separation under Nonlinear Decoding). Let the trend decoder $\bar { \mathcal { G } } _ { t } : \mathbb { R } ^ { d }  \mathbb { R } ^ { F }$ be $L _ { t } – L i p s c h i t z$ continuous in the temporal domain, i.e.,

$$
\begin{array} { r } { \Vert \mathcal { G } _ { t } ( \mathbf { z } _ { t , \tau _ { 1 } } ) - \mathcal { G } _ { t } ( \mathbf { z } _ { t , \tau _ { 2 } } ) \Vert \leq L _ { t } \Vert \mathbf { z } _ { t , \tau _ { 1 } } - \mathbf { z } _ { t , \tau _ { 2 } } \Vert , \quad \forall \tau _ { 1 } , \tau _ { 2 } . } \end{array}\tag{25}
$$

Then, the spectral energy of the reconstructed trend in the high-frequency band $[ \omega _ { h } , \pi ]$ satisfies:

$$
\begin{array} { r l } & { \displaystyle \int _ { \omega _ { h } } ^ { \pi } \| \mathcal { F } ( \mathcal { G } _ { t } \circ \mathbf { z } _ { t } ) ( \omega ) \| ^ { 2 } d \omega } \\ & { \quad \le 2 L _ { t } ^ { 2 } \displaystyle \int _ { \omega _ { h } / 2 } ^ { \pi } \| \mathcal { F } ( \mathbf { z } _ { t } ) ( \omega ) \| ^ { 2 } d \omega + \frac { 8 L _ { t } ^ { 2 } \| \mathbf { z } _ { t } \| _ { \ell ^ { 2 } } ^ { 2 } } { T \omega _ { h } ^ { 2 } } . } \end{array}\tag{26}
$$

The proof can be found in the supplementary repository<sup>3</sup>.

2) Seasonal Loss: In the final model configuration, no explicit spectral regularization is employed for the seasonal module. However, for completeness and symmetry with the trend module, we additionally explored several alternative regularization schemes, which are reported in the Appendix C.

Aside from the regularization term, the seasonal loss is formulated analogously to the trend loss. The key distinction lies in the reconstruction term, which is applied in the frequency domain to emphasize spectral fidelity:

$$
\mathcal { L } _ { r e c } ^ { s e a s o n } = { \left\| \widehat { \mathbf { X } } _ { s } ^ { \left( f r e q \right) } - \widetilde { \mathbf { X } } _ { s } \right\| } _ { 2 } ^ { 2 } = \sum _ { k = 1 } ^ { K } \sum _ { f = 1 } ^ { F } | \widehat { X } _ { s , k , f } - \widetilde { X } _ { s , k , f } | ^ { 2 } .\tag{27}
$$

Thus, the seasonal stream is trained with a multi-component loss in the frequency domain:

$$
\mathcal { L } _ { s e a s o n } = \mathcal { L } _ { r e c } ^ { f r e q } + \mathcal { L } _ { K L } ^ { s e a s o n } ,\tag{28}
$$

where:

$\mathcal { L } _ { r e c } ^ { f r e q }$ is the spectral reconstruction loss.

$\mathcal { L } _ { K L } ^ { s e a s o n } = \mathrm { K L } \big ( q _ { \psi } ( \mathbf { Z } _ { s } ) \| p ( \mathbf { Z } _ { s } ) \big )$ is the KL divergence for the seasonal latent variables.

3) Final Loss Functional: The total loss for DecoVAE is:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { p r e d } } + \mathcal { L } _ { \mathrm { t r e n d } } + \mathcal { L } _ { \mathrm { s e a s o n } } ,\tag{29}
$$

where the prediction loss is:

$$
\mathcal { L } _ { \mathrm { p r e d } } = \| \mathbf { Y } - ( \widehat { \mathbf { Y } } _ { t } + \widehat { \mathbf { Y } } _ { s } ) \| _ { 2 } ^ { 2 } .\tag{30}
$$

## IV. EXPERIMENTS

A. Experiments Setup

a) Datasets.: We evaluate the prediction quality using seven real-world datasets from diverse application domains. These include four variants of the Electricity Transformer Temperature dataset (ETTh1, ETTh2, ETTm1, ETTm2) [33], traffic flow measurements [34], household electricity consumption<sup>4</sup>, and meteorological observations<sup>5</sup>. Complete dataset specifications are provided in Appendix A.

TABLE II: Comparison of long-term probabilistic time series forecasting performance $( T = 9 6 , L = 7 2 0 )$ . Lower CRPS and NMAE scores indicate better performance. Results are reported as mean ± standard error, computed from three independent runs involving model retraining and evaluation. Boldface highlights the best-performing method, while underlining denotes the second-best result.
<table><tr><td>Dataset</td><td>Metric</td><td> $\mathrm { D e c o V A E ~ ( o u r s ) }$ </td><td>K²VAE</td><td>LaST</td><td>DeepAR</td><td>TFM</td><td>TSDiff</td><td>DeNOTS</td></tr><tr><td>Electricity</td><td>CRPS</td><td> $0 . 1 2 0 \pm 0 . 0 0 0$ </td><td> $\mathbf { 0 . 1 1 5 \pm 0 . 0 0 6 }$ </td><td> $0 . 1 4 6 \pm 0 . 0 0 1$ </td><td> $0 . 1 6 3 \pm 0 . 0 0 9$ </td><td> $0 . 2 1 0 \pm 0 . 0 1 7$ </td><td> $0 . 1 2 6 \pm 0 . 0 0 3$ </td><td> $0 . 1 8 6 \pm 0 . 0 0 9$ </td></tr><tr><td></td><td>NMAE</td><td> $\mathbf { 0 . 1 3 8 \pm 0 . 0 0 1 }$ </td><td> $\underline { { 0 . 1 4 0 \pm 0 . 0 0 4 } }$ </td><td> $0 . 1 6 9 \pm 0 . 0 0 1$ </td><td> $0 . 2 0 4 \pm 0 . 0 1 1$ </td><td> $0 . 2 9 2 \pm 0 . 0 2 2$ </td><td> $0 . 1 6 5 \pm 0 . 0 0 4$ </td><td> $0 . 1 8 6 \pm 0 . 0 0 9$ </td></tr><tr><td>Traffic</td><td>CRPS</td><td> $0 . 2 1 8 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 1 8 9 \pm 0 . 0 0 2 }$ </td><td> $0 . 3 0 1 \pm 0 . 0 0 3$ </td><td> $0 . 2 2 0 \pm 0 . 0 0 3$ </td><td> $0 . 4 7 3 \pm 0 . 0 1 8$ </td><td> $0 . 2 9 7 \pm 0 . 0 2 7$ </td><td> $0 . 4 1 4 \pm 0 . 1 4 3$ </td></tr><tr><td></td><td>NMAE</td><td> $\overline { { 0 . 2 4 6 \pm 0 . 0 0 2 } }$ </td><td> $\mathbf { 0 . 2 4 3 \pm 0 . 0 0 0 }$ </td><td> $0 . 3 4 3 \pm 0 . 0 0 2$ </td><td> $0 . 2 8 6 \pm 0 . 0 0 6$ </td><td> $0 . 6 3 7 \pm 0 . 0 2 3$ </td><td> $0 . 3 6 8 \pm 0 . 0 3 3$ </td><td> $0 . 4 1 9 \pm 0 . 1 5 1$ </td></tr><tr><td>ETTh1</td><td>CRPS</td><td> $\mathbf { 0 . 2 9 8 \pm 0 . 0 0 3 }$ </td><td> $0 . 3 3 4 \pm 0 . 0 0 8$ </td><td> $0 . 5 7 3 \pm 0 . 0 0 9$ </td><td> $0 . 6 9 0 \pm 0 . 2 1 1$ </td><td> $0 . 5 9 0 \pm 0 . 0 6 0$ </td><td> $0 . 5 7 5 \pm 0 . 0 5 9$ </td><td> $0 . 5 5 5 \pm 0 . 0 1 4$ </td></tr><tr><td></td><td>NMAE</td><td> $\mathbf { 0 . 3 6 0 \pm 0 . 0 0 4 }$ </td><td> $\overline { { 0 . 4 2 8 \pm 0 . 0 1 2 } }$ </td><td> $0 . 6 1 4 \pm 0 . 0 1 0$ </td><td> $0 . 8 3 0 \pm 0 . 2 4 0$ </td><td> $0 . 8 3 9 \pm 0 . 0 8 8$ </td><td> $0 . 7 0 1 \pm 0 . 0 5 4$ </td><td> $0 . 7 5 0 \pm 0 . 0 1 9$ </td></tr><tr><td>ETTh2</td><td>CRPS</td><td> $\mathbf { 0 . 4 7 8 \pm 0 . 0 0 8 }$ </td><td> ${ \underline { { 1 . 0 0 0 } } } \pm 0 . 2 3 5$ </td><td> $2 . 5 3 1 \pm 0 . 0 5 4$ </td><td> $2 . 3 3 2 \pm 0 . 0 1 3$ </td><td> $2 . 4 9 5 \pm 0 . 6 8 8$ </td><td> $2 . 2 5 4 \pm 0 . 1 0 6$ </td><td> $2 . 0 8 2 \pm 0 . 1 3 6$ </td></tr><tr><td></td><td>NMAE</td><td> $\mathbf { 0 . 5 8 0 \pm 0 . 0 0 1 }$ </td><td> $\overline { { 0 . 7 6 9 \pm 0 . 0 8 3 } }$ </td><td> $2 . 6 0 2 \pm 0 . 0 3 8$ </td><td> $2 . 6 1 1 \pm 0 . 0 5 3$ </td><td> $3 . 2 0 4 \pm 0 . 9 9 2$ </td><td> $2 . 4 8 8 \pm 0 . 1 0 4$ </td><td> $2 . 3 5 6 \pm 0 . 1 0 4$ </td></tr><tr><td>ETTm1</td><td>CRPS</td><td> $0 . 2 9 3 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 2 6 6 \pm 0 . 0 2 4 }$ </td><td> $0 . 4 2 1 \pm 0 . 0 1 3$ </td><td> $0 . 4 6 6 \pm 0 . 0 7 3$ </td><td> $0 . 5 9 2 \pm 0 . 0 1 0$ </td><td> $0 . 4 8 3 \pm 0 . 0 2 0$ </td><td> $0 . 5 7 6 \pm 0 . 0 0 9$ </td></tr><tr><td></td><td>NMAE</td><td> $\mathbf { 0 . 3 2 7 \pm 0 . 0 0 2 }$ </td><td> $\underline { { 0 . 3 2 8 \pm 0 . 0 3 9 } }$ </td><td> $0 . 4 8 7 \pm 0 . 0 1 5$ </td><td> $0 . 6 2 3 \pm 0 . 0 6 9$ </td><td> $0 . 8 5 5 \pm 0 . 0 2 5$ </td><td> $0 . 6 2 9 \pm 0 . 0 2 5$ </td><td> $0 . 6 1 4 \pm 0 . 0 1 0$ </td></tr><tr><td>ETTm2</td><td>CRPS</td><td> $\mathbf { 0 . 4 6 8 \pm 0 . 0 0 5 }$ </td><td> $\underline { { 0 . 9 8 9 \pm 0 . 2 7 5 } }$ </td><td> $1 . 8 7 1 \pm 0 . 2 4 1$ </td><td> $2 . 0 6 4 \pm 0 . 6 4 6$ </td><td> $2 . 8 9 9 \pm 0 . 7 8 2$ </td><td> $1 . 6 4 2 \pm 0 . 4 5 0$ </td><td> $1 . 9 3 1 \pm 0 . 0 1 6$ </td></tr><tr><td></td><td>NMAE</td><td> $\mathbf { 0 . 5 2 4 \pm 0 . 0 0 1 }$ </td><td> $\overline { { 0 . 7 1 3 \pm 0 . 1 0 5 } }$ </td><td> $2 . 0 2 2 \pm 0 . 2 5 1$ </td><td> $2 . 3 8 8 \pm 0 . 6 3 3$ </td><td> $3 . 7 8 3 \pm 0 . 9 7 5$ </td><td> $1 . 9 1 5 \pm 0 . 4 3 0$ </td><td> $2 . 0 0 8 \pm 0 . 0 3 6$ </td></tr><tr><td>Weather</td><td>CRPS</td><td> $\underline { { 0 . 4 3 1 \pm 0 . 0 0 2 } }$ </td><td> $0 . 6 6 1 \pm 0 . 0 0 7$ </td><td> $0 . 6 7 2 \pm 0 . 0 2 8$ </td><td> $0 . 4 6 5 \pm 0 . 0 6 4$ </td><td> $1 . 3 4 4 \pm 0 . 2 4 6$ </td><td> $\mathbf { 0 . 3 7 5 \pm 0 . 0 3 0 }$ </td><td> $0 . 8 0 3 \pm 0 . 0 7 1$ </td></tr><tr><td></td><td>NMAE</td><td> $\mathbf { 0 . 4 5 9 \pm 0 . 0 0 5 }$ </td><td> $0 . 6 0 6 \pm 0 . 0 0 8$ </td><td> $0 . 8 3 5 \pm 0 . 0 4 3$ </td><td> $0 . 5 5 5 \pm 0 . 0 6 7$ </td><td> $1 . 5 4 2 \pm 0 . 4 2 2$ </td><td> $0 . 4 6 5 \pm 0 . 0 3 3$ </td><td> $0 . 8 6 5 \pm 0 . 0 8 4$ </td></tr><tr><td>Avg. Rank</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>CRPS NMAE</td><td>1.571 1.143</td><td>1.857 2.143</td><td>4.571 4.143</td><td>4.714 5.000</td><td>6.714 7.000</td><td>3.571 3.857</td><td>5.000 4.714</td></tr></table>

b) Evaluation Protocol.: Model performance is quantified using two complementary metrics: the Continuous Ranked Probability Score (CRPS) [35], [36], which assesses distributional forecast quality, and Normalized Mean Absolute Error (NMAE), which evaluates point prediction accuracy. We refer readers to Appendix B for complete metric definitions.

c) Baselines.: We compare our approach against several generative models, including modern $K ^ { 2 } \mathrm { V A E }$ [17], LaST [16], TSDiff [18], DeepAR [15], DeNOTS [20] and TFM [19].

## B. Main Results

a) Short-term forecasting $( T = 9 6 , L = 2 4 ) \colon$ For shortterm forecasting (Table I), DecoVAE yields substantial CRPS gains on ETTh2 (+14.25%), ETTm2 (+14.96%), and Weather (+12.54% over TSDiff). On high-dimensional data, it secures the best NMAE on Electricity $( 0 . 1 0 4 { \pm } 0 . 0 0 1 )$ and outperforms most baselines by over 30% on Traffic. Overall, DecoVAE surpasses non-decomposition methods by over 30% in CRPS, significantly outperforming LaST on ETTh2 (+30.65%) and Weather (+46.84%), while matching or exceeding $K ^ { 2 } \mathrm { V A E }$ despite a simpler architecture. Finally, DecoVAE achieves the best average ranks: 1.429 and 1.143 for CRPS and NMAE correspondingly surpassing all previous approaches.

b) Long-term forecasting $( T ~ = ~ 9 6 , L ~ = ~ 7 2 0 ) .$ : For long-term forecasting (Table II), DecoVAE’s explicit decomposition yields substantial advantages. On ETT datasets, it improves CRPS by up to 52.68% and NMAE by up to

26.51% over second-best methods. Against LaST, it achieves up to 81.11% CRPS improvement on ETTh2, highlighting the limitations of latent space decomposition over extended horizons, while vastly outperforming non-decomposition baselines like DeepAR. Additionally, it matches $K ^ { 2 } \mathrm { { V A E } \mathrm { { s } } }$ NMAE $( 0 . 1 3 8 \pm 0 . 0 0 1 )$ on Electricity with greater efficiency, and secures the best NMAE $( 0 . 4 5 9 { \pm } 0 . 0 0 5 )$ on the Weather dataset.

## C. Computational Efficiency Analysis

Beyond predictive accuracy, we evaluate DecoVAE’s computational characteristics across training speed, memory consumption, model size, and inference time. We focus on the two largest datasets–Electricity (321 channels) and Traffic (862 channels)–considering both short-term $( L = 2 4 )$ and long-term $( L = 7 2 0 )$ horizons, comparing against $\mathrm { K ^ { 2 } V A E }$ and TSDiff, the best and second-best performing methods from our main experiments.

a) Time Efficiency: As shown in Figure 2, DecoVAE demonstrates substantial training efficiency advantages: 30– 74% faster per epoch compared to $\mathrm { K ^ { 2 } V A E }$ , with inference speed improvements of 10–35%. Against TSDiff, DecoVAE achieves approximately 50% faster training and 32–76% faster inference, with greater gains in long-term forecasting.

b) Memory Consumption and Model Size: Figure 3 shows DecoVAE achieves 48–85% reduction in test memory usage and 80–93% smaller model size compared to K<sup>2</sup>VAE (5MB vs 70MB in extreme cases), corresponding to 85–95% fewer parameters (under 1M vs up to 18M+). While TSDiff has 2–3× smaller model size than DecoVAE, this comes at the cost of significantly slower inference due to its iterative diffusion mechanism. DecoVAE remains 10–20× more compact than $\mathrm { K ^ { 2 } V A E }$ while delivering comparable forecasting performance. These results establish DecoVAE as computationally efficient, achieving favorable trade-offs between operational requirements and forecasting quality.

![](images/9327554f7b81413c350e60e0298cfa3e2f285dcf69f598edbb8f03d41c1df4fb.jpg)  
Fig. 2: Comparison of training and inference time across models for short-term $( L = 2 4 )$ and long-term $( L = 7 2 0 )$ forecasting tasks on the Electricity and Traffic datasets. Lower values indicate better computational efficiency.

![](images/3a89baa85bdd261bae3b679a05b1f9f3926d9142ea26a7421ca1ee3bec4d2733.jpg)  
Fig. 3: Comparison of training and inference memory consumption for short-term $( L = 2 4 )$ and long-term $( L = 7 2 0 )$ forecasting tasks on the Electricity and Traffic datasets. Lower values indicate better computational efficiency.

## D. Interpretability

DecoVAE provides interpretable predictions through explicit trend-seasonal decomposition. Figure 4 illustrates a representative example on the Electricity dataset, where the model successfully disentangles long-term directional movement (trend, blue) from short-term periodic fluctuations (seasonality, yellow). DecoVAE achieves interpretable decomposition, preserving the input’s structural and temporal dynamics.

Figure 5 presents a Principle Component Analysis (PCA) of trend and seasonal embeddings, with the seasonal embeddings first converted from the frequency domain to the time domain. Crucially, the two groups are highly separable. Trend embeddings form an almost distinct path, likely due to firstand second-derivative regularization, whereas seasonal embeddings appear more clustered. This clustering may indicate that, within a given periodicity, seasonal embeddings corresponding to similar periods are closely grouped, or as a result of the frequency-to-time domain transformation.

![](images/09b7dcd5726105ac6b2a594378d193ab28349d8bae79b86a3a51002f95c9f0c0.jpg)  
Fig. 4: Probabilistic trend-seasonal forecasts generated by DecoVAE on the Electricity dataset, with ground-truth observations overlaid for comparison.

![](images/6dbe2bd0b984d9077521c88825728db4c7c1d3267c8f15a4d5f8e13d8ffbc4eb.jpg)  
Fig. 5: Principal component analysis (PCA) of trend and seasonal embeddings projected onto a 2D space on the Electricity dataset.

## E. Regularization Effect

To investigate the impact of our Hodrick-Prescott-inspired trend regularization, we conduct an ablation study varying the regularization strengths $\alpha _ { 1 }$ and $\alpha _ { 2 }$ on the ETTh1 dataset $( T =$ 96, $L = 7 2 0 )$ . Figure 6 and Figure 7 demonstrate that both regularization terms improve forecasting performance.

Without first-order regularization $( \alpha _ { 1 } ~ = ~ 0 )$ , the model achieves CRPS of 0.33 and NMAE of 0.39. Introducing moderate regularization $( \alpha _ { 1 } \in [ 0 . 5 , 7 . 0 ] )$ yields improvements of approximately 9% in CRPS and 8% in NMAE, stabilizing performance at $\mathrm { C R P S } \approx 0 . 3 0$ and $\mathrm { N M A E } \approx 0 . 3 6$ . Similarly, second-order regularization provides consistent gains, with performance improving from the unregularized baseline when $\alpha _ { 2 } ~ \in ~ [ 1 . 0 , 5 . 0 ]$ . These results empirically validate Theorems III.1 and III.2, confirming that differential regularization successfully enforces trend smoothness and spectral separation, directly translating to improved forecasting accuracy.

![](images/b486d05b618e67862673baa4548fae884bd8f05c21dafd30a667847326864c46.jpg)  
Fig. 6: Effect of first-order latent trend regularization $( \alpha _ { 1 } )$ on long-term forecasting performance on ETTh1 $( T = 9 6 , L =$ 720).

![](images/2c30b193a13614b4ab7910ea06ce1ea5831a032097cea7f17fcf28ffb85fc1f3.jpg)  
Fig. 7: Effect of second-order latent trend regularization $\left( \alpha _ { 2 } \right)$ on long-term forecasting performance on ETTh1 $( T = 9 6$ $L = 7 2 0 )$ .

## V. CONCLUSION

We introduce DecoVAE, an efficient, interpretable variational autoencoder for probabilistic time series forecasting. By isolating trend and seasonal dynamics into distinct latent pathways, it applies domain-specific biases without interference: a time-domain differential regularizer ensures trend smoothness, while a complex Gaussian VAE captures seasonal cyclical behaviors in the frequency domain. Evaluated across seven realworld benchmarks, DecoVAE consistently outperforms strong baselines, improving CRPS and NMAE by up to 14.96% and 23.30% for short-term tasks, and 52.68% and 26.51% for longterm horizons. Crucially, it achieves these accuracy gains with substantial computational efficiency, reducing model weights by up to 93% and accelerating runtime by 74% versus the second-best baseline.

## REFERENCES

[1] Z. Qiao, S. Pan, A. Wang, V. Zhukova, Y. Liu, X. Jiang, Q. Wen, M. Long, M. Jin, and C. Liu, “It’s time: Towards the next generation of time series forecasting benchmarks,” arXiv preprint arXiv:2602.12147, 2026.

[2] X. Kong, Z. Chen, W. Liu, K. Ning, L. Zhang, S. Muhammad Marier, Y. Liu, Y. Chen, and F. Xia, “Deep learning for time series forecasting: a survey,” International Journal of Machine Learning and Cybernetics, vol. 16, no. 7, pp. 5079–5112, 2025.

[3] J. Zhang, X. Wen, Z. Zhang, S. Zheng, J. Li, and J. Bian, “Probts: Benchmarking point and distributional forecasting across diverse prediction horizons,” Advances in Neural Information Processing Systems, vol. 37, pp. 48 045–48 082, 2024.

[4] T. Proietti and D. J. Pedregal, “Seasonality in high frequency time series,” Econometrics and Statistics, vol. 27, pp. 62–82, 2023. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S2452306222000090

[5] Q. Zhang, Y. Sun, H. Wen, P. Yang, X. Li, M. Li, K.-Y. Lam, S.-M. Yiu, and H. Yin, “Time series analysis in frequency domain: A survey of open challenges, opportunities and benchmarks,” arXiv preprint arXiv:2504.07099, 2025.

[6] Z. An, J. You, J. Li, Y. Tang, W. Li, H. Du, and S. Du, “Fredn: Spectral disentanglement for time series forecasting via learnable frequency decomposition,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 24, 2026, pp. 19 623–19 631.

[7] J. Lu, P. Chen, C. Guo, Y. Shu, M. Wang, and B. Yang, “Towards non-stationary time series forecasting with temporal stabilization and frequency differencing,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 29, 2026, pp. 24 070–24 078.

[8] J. Deng, F. Ye, D. Yin, X. Song, I. Tsang, and H. Xiong, “Parsimony or capability? decomposition delivers both in long-term time series forecasting,” Advances in Neural Information Processing Systems, vol. 37, pp. 66 687–66 712, 2024.

[9] Y. Yang, M. Jin, H. Wen, C. Zhang, Y. Liang, L. Ma, Y. Wang, C. Liu, B. Yang, Z. Xu, S. Pan, and Q. Wen, “A survey on diffusion models for time series and spatio-temporal data,” ACM Comput. Surv., vol. 58, no. 8, Feb. 2026. [Online]. Available: https://doi.org/10.1145/3783986

[10] Y. Tashiro, J. Song, Y. Song, and S. Ermon, “Csdi: Conditional scorebased diffusion models for probabilistic time series imputation,” Advances in neural information processing systems, vol. 34, pp. 24 804– 24 816, 2021.

[11] K. Alkilane, Y. He, and D.-H. Lee, “Expert denoising for enhanced diffusion-based time series forecasting,” Expert Systems with Applications, vol. 299, p. 130269, 2026. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0957417425038849

[12] K. Rasul, C. Seward, I. Schuster, and R. Vollgraf, “Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting,” in International conference on machine learning. PMLR, 2021, pp. 8857–8868.

[13] R. J. Hodrick and E. C. Prescott, “Postwar u.s. business cycles: An empirical investigation,” Journal of Money, Credit and Banking, vol. 29, no. 1, pp. 1–16, 1997. [Online]. Available: http://www.jstor. org/stable/2953682

[14] Z. Xu, A. Zeng, and Q. Xu, “Fits: Modeling time series with 10k parameters,” in The Twelfth International Conference on Learning Representations.

[15] D. Salinas, V. Flunkert, J. Gasthaus, and T. Januschowski, “Deepar: Probabilistic forecasting with autoregressive recurrent networks,” International journal offorecasting, vol. 36, no. 3, pp. 1181–1191, 2020.

[16] Z. Wang, X. Xu, W. Zhang, G. Trajcevski, T. Zhong, and F. Zhou, “Learning latent seasonal-trend representations for time series forecasting,” Advances in Neural Information Processing Systems, vol. 35, pp. 38 775–38 787, 2022.

[17] X. Wu, X. Qiu, H. Gao, J. Hu, B. Yang, and C. Guo, “K2vae: A koopman-kalman enhanced variational autoencoder for probabilistic time series forecasting,” arXiv preprint arXiv:2505.23017, 2025.

[18] M. Kollovieh, A. F. Ansari, M. Bohlke-Schneider, J. Zschiegner, H. Wang, and Y. B. Wang, “Predict, refine, synthesize: Self-guiding diffusion models for probabilistic time series forecasting,” Advances in Neural Information Processing Systems, vol. 36, pp. 28 341–28 364, 2023.

[19] X. N. Zhang, Y. Pu, Y. Kawamura, A. Loza, Y. Bengio, D. Shung, and A. Tong, “Trajectory flow matching with applications to clinical time

series modelling,” Advances in Neural Information Processing Systems, vol. 37, pp. 107 198–107 224, 2024.

[20] I. Kuleshov, E. Romanenkova, V. A. Zhuzhel, G. Boeva, E. Vorsin, and A. Zaytsev, “DeNOTS: Stable deep neural ODEs for time series,” in The Fourteenth International Conference on Learning Representations (ICLR), 2026. [Online]. Available: https://openreview.net/forum?id= SFoDJZ1sSk

[21] R. B. Cleveland, W. S. Cleveland, J. E. McRae, I. Terpenning et al., “Stl: A seasonal-trend decomposition,” J. off. Stat, vol. 6, no. 1, pp. 3–73, 1990.

[22] R. J. Hyndman, Moving Averages. Berlin, Heidelberg: Springer Berlin Heidelberg, 2011, pp. 866–869. [Online]. Available: https: //doi.org/10.1007/978-3-642-04898-2 380

[23] T. Kreuzer, J. Zdravkovic, and P. Papapetrou, “Unpacking the trend: decomposition as a catalyst to enhance time series forecasting models,” Data Mining and Knowledge Discovery, vol. 39, 07 2025.

[24] T. Zhou, Z. Ma, Q. Wen, X. Wang, L. Sun, and R. Jin, “Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting,” in International conference on machine learning. PMLR, 2022, pp. 27 268–27 286.

[25] G. Yu, J. Zou, X. Hu, A. I. Aviles-Rivero, J. Qin, and S. Wang, “Revitalizing multivariate time series forecasting: Learnable decomposition with inter-series dependencies and intra-series variations modeling,” in Forty-first International Conference on Machine Learning, 2024. [Online]. Available: https://openreview.net/forum?id=87CYNyCGOo

[26] D. Luo and X. Wang, “Decompnet: Enhancing time series forecasting models with implicit decomposition,” Poster presented at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025), San Diego, CA, USA, Dec. 2025. [Online]. Available: https://neurips.cc/virtual/2025/loc/san-diego/poster/116526

[27] H. Wu, J. Xu, J. Wang, and M. Long, “Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting,” Advances in neural information processing systems, vol. 34, pp. 22 419– 22 430, 2021.

[28] A. Stitsyuk and J. Choi, “xpatch: Dual-stream time series forecasting with exponential seasonal-trend decomposition,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 19, 2025, pp. 20 601–20 609.

[29] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “Timesnet: Temporal 2d-variation modeling for general time series analysis,” in The Eleventh International Conference on Learning Representations.

[30] X. Qiu, X. Wu, H. Cheng, X. Liu, C. Guo, J. Hu, and B. Yang, “Dbloss: Decomposition-based loss function for time series forecasting,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems.

[31] P. Liu, B. Wu, N. Li, T. Dai, F. Lei, J. Bao, Y. Jiang, and S.-T. Xia, “Wftnet: Exploiting global and local periodicity in long-term time series forecasting,” in Icassp 2024-2024 ieee international conference on acoustics, speech and signal processing (icassp). IEEE, 2024, pp. 5960–5964.

[32] J. Crabbe, N. Huynh, J. P. Stanczuk, and M. Van Der Schaar, “Time´ series diffusion in the frequency domain,” in International Conference on Machine Learning. PMLR, 2024, pp. 9407–9438.

[33] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, “Informer: Beyond efficient transformer for long sequence time-series forecasting,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, no. 12, 2021, pp. 11 106–11 115.

[34] G. Lai, W.-C. Chang, Y. Yang, and H. Liu, “Modeling long-and short-term temporal patterns with deep neural networks,” in The 41st international ACM SIGIR conference on research & development in information retrieval, 2018, pp. 95–104.

[35] T. Gneiting and A. E. Raftery, “Strictly proper scoring rules, prediction, and estimation,” Journal of the American statistical Association, vol. 102, no. 477, pp. 359–378, 2007.

[36] M. Zamo and P. Naveau, “Estimation of the continuous ranked probability score with limited information and applications to ensemble weather forecasts,” Mathematical Geosciences, vol. 50, no. 2, pp. 209–234, 2018.

[37] R. Yamamoto, E. Song, and J.-M. Kim, “Parallel wavegan: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram,” in ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020, pp. 6199–6203.

[38] K. Barmpas, N. Lee, Y. Panagakis, D. Adamos, N. Laskaris, and S. Zafeiriou, “Advancing brainwave modelling with a codebook-based foundation model,” in NeurIPS 2025 Workshop on Foundation Models for the Brain and Body.

## APPENDIX

## A. Dataset Specifications

Our experimental study uses seven widely-adopted multivariate time series datasets with varying dimensionality, granularity, and domains [3], [17], [27], [33]:

• ETT: Four variants (ETTh1/h2 hourly, ETTm1/m2 15- min) of transformer temperature metrics (2016–2018).

• Electricity: Hourly consumption from 321 clients (2012– 2014).

• Weather: 10-min meteorological observations (temperature, humidity, pressure) throughout 2020.

• Traffic: Hourly road occupancy from 862 San Francisco Bay Area sensors (2015–2016).

We augment each series with 11 time-dependent covariates: sinusoidal encodings for yearly to daily periodicities and a normalized time-delta feature. Following standard practice, we use chronological splitting (70% train, 10% validation, 20% test).

## B. Performance Metrics

We quantify forecasting quality through two established metrics that assess complementary aspects of prediction performance:

a) Normalized Mean Absolute Error (NMAE).: This metric evaluates point forecast accuracy by normalizing absolute deviations relative to the true values:

$$
\mathrm { N M A E } ( \boldsymbol { x } , \hat { \boldsymbol { x } } ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \frac { \sum _ { i = 1 } ^ { F } \lvert x _ { i } ^ { n } - \hat { x } _ { i } ^ { n } \rvert } { \sum _ { i = 1 } ^ { F } \lvert x _ { i } ^ { n } \rvert } ,
$$

where $\boldsymbol { x } _ { i } ^ { n }$ represents the ground truth and $\hat { x } _ { i } ^ { n }$ is the predicted median from 100 sampled trajectories.

b) Continuous Ranked Probability Score (CRPS).: To assess the quality of predictive distributions, we use CRPS, which compares the cumulative distribution function F(z) against the empirical distribution of observations:

$$
\mathrm { C R P S } = \int _ { \mathbb { R } } \bigl ( F ( z ) - \mathbb { I } \{ x \leq z \} \bigr ) ^ { 2 } d z .
$$

We compute this integral using Monte Carlo approximation following [36]. Both metrics are negatively oriented, with smaller values indicating superior performance.

## C. Seasonal Module Variants

We compare real-valued, complex-valued, and amplitudephase factorized spectral parameterizations for the seasonal component. To stabilize frequency-domain training, each variant employs tailored spectral regularization alongside standard reconstruction and KL terms.

Complex-valued FITS-style model [14]. Seasonal embeddings are derived via complex-valued linear layers directly in the frequency domain, preserving the Fourier transform’s structure by jointly capturing amplitude and phase. Consequently, spectral regularization enforces explicit energy matching between the input and latent representations. Let $E _ { x }$ and $E _ { z }$ denote the average energy of the input and latent signals, respectively:

TABLE III: Comparison of different DecoVAE variants on short-term forecasting task. $( T = 9 6 , L = 2 4 )$ Lower CRPS and NMAE scores indicate better performance. Results are reported as mean ± standard error over three independent runs.
<table><tr><td>Dataset</td><td>Metric</td><td> $\mathrm { F r e q M L P \ ( w / o \ R e g . ) }$ </td><td> $\mathrm { F r e q M L P \ ( w / \ R e g . ) }$ </td><td>AmpPhase</td><td> $\mathrm { F I T S \ ( w / o \ R e g . ) }$ </td><td>FITS (w/ Reg.)</td></tr><tr><td rowspan="2">ETTh1</td><td>CRPS</td><td> $0 . 2 2 9 \pm 0 . 0 0 2$ </td><td> $0 . 2 3 0 \pm 0 . 0 0 0$ </td><td> $0 . 2 3 4 \pm 0 . 0 0 2$ </td><td> $0 . 2 1 7 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 2 1 5 \pm 0 . 0 0 1 }$ </td></tr><tr><td>NMAE</td><td> $0 . 2 6 5 \pm 0 . 0 0 2$ </td><td> $0 . 2 6 7 \pm 0 . 0 0 0$ </td><td> $0 . 2 6 8 \pm 0 . 0 0 2$ </td><td> $\overline { { 0 . 2 6 2 \pm 0 . 0 0 2 } }$ </td><td> $\mathbf { 0 . 2 5 9 \pm 0 . 0 0 1 }$ </td></tr><tr><td rowspan="2">ETTh2</td><td>CRPS</td><td> $0 . 3 3 6 \pm 0 . 0 0 3$ </td><td> $0 . 3 4 5 \pm 0 . 0 0 5$ </td><td> $0 . 3 4 7 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 3 1 9 \pm 0 . 0 0 5 }$ </td><td> $0 . 3 2 7 \pm 0 . 0 1 4$ </td></tr><tr><td>NMAE</td><td> $\mathbf { 0 . 3 8 1 \pm 0 . 0 0 4 }$ </td><td> $0 . 3 8 6 \pm 0 . 0 0 8$ </td><td> $0 . 3 8 6 \pm 0 . 0 0 0$ </td><td> $\underline { { 0 . 3 8 2 \pm 0 . 0 0 5 } }$ </td><td> $\overline { { 0 . 3 8 3 \pm 0 . 0 0 0 } }$ </td></tr><tr><td rowspan="2">ETTm1</td><td>CRPS</td><td> $\mathbf { 0 . 1 8 4 \pm 0 . 0 0 2 }$ </td><td> $0 . 1 9 4 \pm 0 . 0 0 1$ </td><td> $0 . 1 9 7 \pm 0 . 0 0 3$ </td><td> $0 . 1 8 8 \pm 0 . 0 0 2$ </td><td> $0 . 1 9 0 \pm 0 . 0 0 1$ </td></tr><tr><td>NMAE</td><td> $\mathbf { 0 . 2 1 7 \pm 0 . 0 0 1 }$ </td><td> $0 . 2 2 9 \pm 0 . 0 0 2$ </td><td> $0 . 2 2 7 \pm 0 . 0 0 3$ </td><td> $\overline { { 0 . 2 2 5 \pm 0 . 0 0 2 } }$ </td><td> $0 . 2 2 5 \pm 0 . 0 0 1$ </td></tr><tr><td rowspan="2">ETTm2</td><td>CRPS</td><td> $0 . 2 4 6 \pm 0 . 0 0 7$ </td><td> $0 . 2 4 7 \pm 0 . 0 0 1$ </td><td> $0 . 2 5 2 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 2 3 3 \pm 0 . 0 0 1 }$ </td><td> $0 . 2 3 4 \pm 0 . 0 0 1$ </td></tr><tr><td>NMAE</td><td> $\mathbf { 0 . 2 8 3 \pm 0 . 0 0 7 }$ </td><td> $0 . 2 8 8 \pm 0 . 0 0 2$ </td><td> $0 . 2 8 6 \pm 0 . 0 0 2$ </td><td> $\underline { { 0 . 2 8 6 \pm 0 . 0 0 1 } }$ </td><td> $\overline { { 0 . 2 8 7 \pm 0 . 0 0 3 } }$ </td></tr><tr><td rowspan="2">Weather</td><td>CRPS</td><td> $0 . 2 9 7 \pm 0 . 0 1 5$ </td><td> $0 . 2 9 3 \pm 0 . 0 0 8$ </td><td> $0 . 3 0 4 \pm 0 . 0 0 4$ </td><td> $0 . 2 8 6 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 2 7 8 \pm 0 . 0 0 6 }$ </td></tr><tr><td>NMAE</td><td> $0 . 3 3 0 \pm 0 . 0 1 4$ </td><td> $0 . 3 2 2 \pm 0 . 0 0 9$ </td><td> $0 . 3 2 4 \pm 0 . 0 0 4$ </td><td> $\overline { { 0 . 3 1 6 \pm 0 . 0 0 1 } }$ </td><td> $\mathbf { 0 . 3 0 5 \pm 0 . 0 0 7 }$ </td></tr></table>

$$
E _ { x } = \mathbb { E } \left[ \| \mathbf { X } \| _ { 2 } ^ { 2 } \right] ,
$$

$$
E _ { z } = \mathbb { E } \left[ \lVert \mathbf { Z } _ { s } \rVert _ { 2 } ^ { 2 } \right] .\tag{31}
$$

(32)

The regularization term is defined as:

$$
\mathcal { L } _ { s p e c } ^ { \mathrm { F I T S } } = \left( E _ { x } - E _ { z } \right) ^ { 2 } + \lambda \cdot \mathbb { E } \left[ \Vert \mathbf { Z } _ { s } \Vert _ { 1 } \right] ,\tag{33}
$$

which jointly enforces energy preservation while penalizing noisy latent activations.

Real-valued Frequency MLP (FreqMLP). Decomposing the complex spectrum into real and imaginary parts enables processing via standard real-valued networks, though this sacrifices explicit preservation of the Fourier domain’s complex structure. Consequently, the real-valued frequency MLP applies a simple power spectral density (PSD) penalty to the latent representation:

$$
\mathrm { P S D } ( \mathbf { Z } _ { s } ) = \frac { | \mathbf { Z } _ { s } | ^ { 2 } } { d } , \qquad \mathcal { L } _ { s p e c } ^ { \mathrm { F r e q M L P } } = \sum \mathrm { P S D } ( \mathbf { Z } _ { s } ) ,\tag{34}
$$

which encourages bounded spectral energy and discourages excessive high-frequency amplification.

Amplitude-Phase factorized model (AmpPhase). The Fourier spectrum is decomposed into amplitude and phase, modeled as independent stochastic variables via separate variational encoders for disentangled spectral modeling. Reconstruction occurs separately for magnitude and phase, with the amplitude loss integrating relative Frobenius error and logmagnitude consistency [37]:

$$
\mathcal { L } _ { a m p } = \frac { \Vert \widehat { A } - A \Vert _ { F } } { \Vert A \Vert _ { F } + \epsilon } + \sum \left| \log ( \widehat { A } + \epsilon ) - \log ( A + \epsilon ) \right| ,\tag{35}
$$

while the phase loss enforces angular alignment [38]:

$$
\mathcal { L } _ { p h a s e } = \sum \left( 1 - \cos ( \phi - \widehat { \phi } ) \right) .\tag{36}
$$

The total spectral loss is:

$$
\begin{array} { r } { \mathcal { L } _ { s p e c } ^ { \mathrm { A m p P h a s e } } = \mathcal { L } _ { a m p } + \mathcal { L } _ { p h a s e } . } \end{array}\tag{37}
$$

Further implementation details of each variant are available in the accompanying GitHub repository<sup>6</sup>.

1) Ablation Study on Spectral Parameterizations performance: An extended ablation study (Table III) compares seasonal spectral parameterizations, including real-valued FreqMLP, amplitude-phase factorized, and complex-valued FITSstyle formulations, both with and without regularization. The results validate the FITS-based variant as the default choice for the DecoVAE seasonal module.

## D. Comparison of decomposition methods

TABLE IV: Comparison of decomposition methods in DecoVAE-FITS $( \mathrm { T } = 9 6 , \mathrm { L } = 7 2 0 )$ on long-term forecasting task. Lower CRPS and NMAE scores indicate better performance. Results are reported as mean ± standard error over three independent runs. Boldface highlights the bestperforming method, while underlining denotes the second-best result.
<table><tr><td>Dataset</td><td>Metric</td><td>MA</td><td>MoE</td><td>EMA</td></tr><tr><td>ETTh1</td><td>CRPS</td><td> $\mathbf { 0 . 2 9 8 \pm 0 . 0 0 3 }$ </td><td> $0 . 3 0 4 \pm 0 . 0 0 9$ </td><td> $0 . 3 5 6 \pm 0 . 0 0 3$ </td></tr><tr><td></td><td>NMAE</td><td> $\underline { { 0 . 3 6 0 \pm 0 . 0 0 4 } }$ </td><td> $\mathbf { 0 . 3 5 7 \pm 0 . 0 0 3 }$ </td><td> $0 . 3 9 3 \pm 0 . 0 0 2$ </td></tr><tr><td>ETTh2</td><td>CRPS</td><td> $\mathbf { 0 . 4 9 8 \pm 0 . 0 0 6 }$ </td><td> $\frac { 0 . 5 1 0 \pm 0 . 0 0 6 } { 0 . 5 8 4 \pm 0 . 0 0 5 }$ </td><td> $0 . 5 1 5 \pm 0 . 0 0 6$ </td></tr><tr><td></td><td>NMAE</td><td> $0 . 5 8 5 \pm 0 . 0 0 2$ </td><td></td><td> $\mathbf { 0 . 5 8 4 \pm 0 . 0 0 4 }$ </td></tr><tr><td>ETTm1</td><td>CRPS</td><td></td><td> $\frac { 0 . 3 0 1 \pm 0 . 0 0 1 } { 0 . 3 2 7 \pm 0 . 0 0 2 }$ </td><td> $0 . 3 0 5 \pm 0 . 0 0 1$ </td></tr><tr><td></td><td>NMAE</td><td> $\begin{array} { c } { 0 . 2 9 7 \pm 0 . 0 0 4 } \\ { 0 . 3 2 9 \pm 0 . 0 0 4 } \end{array}$ </td><td></td><td> $0 . 3 3 0 \pm 0 . 0 0 1$ </td></tr><tr><td>ETTm2</td><td>CRPS</td><td> $0 . 4 6 8 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 4 6 7 \pm 0 . 0 0 7 }$ </td><td> $0 . 4 7 2 \pm 0 . 0 0 1$ </td></tr><tr><td></td><td>NMAE</td><td> $\mathbf { 0 . 5 2 4 } \pm \mathbf { 0 . 0 0 1 }$ </td><td> $0 . 5 3 6 \pm 0 . 0 0 8$ </td><td> $\underline { { 0 . 5 2 8 \pm 0 . 0 0 1 } }$ </td></tr></table>