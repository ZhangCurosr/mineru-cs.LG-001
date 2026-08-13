# Latent variable models for simultaneous EOV identification and removal in population-based SHM

M. D. Champneys <sup>1,2</sup>, M. R. Jones <sup>1</sup>, A. J. Hughes <sup>1</sup>, T. J. Rogers <sup>1</sup>, E. J. Cross <sup>1</sup>, K. Worden <sup>1</sup>

<sup>1</sup> Dynamics Research Group, University of Sheffield, Mappin Street, Sheffield S1 3JD, UK

<sup>2</sup> Centre for Machine Intelligence, University of Sheffield, Mappin Street, Sheffield S1 3JD, UK

## Abstract

The robust treatment of environmental and operational variability (EOV) is an open challenge in populationbased structural health monitoring (PBSHM). The difficulty is compounded in the case that the EOV signals are unmeasured. A common approach in conventional SHM is to apply projection-based methods that discard subspaces of healthy feature data, reasoning that the EOV signal dominates the variance of the measured features. However, a common pitfall of projection-based approaches is that when damage acts close to the same variance-dominant direction, damage sensitivity is removed along with the EOV. An alternative identifying assumption for the removal of particular unmeasured EOVs is slowness; the latent EOV process is characterised by its long temporal correlation. In this paper, the latent EOV is cast as a state-space Gaussian process, enabling tractable O(T) inference via a Kalman filter. A robust hierarchical Bayesian identification framework is developed that enables population-level identification of latent EOVs and EOV-free residual features, using a Laplace approximation. The approach is first validated on a single laboratory-scale bench mark structure from the literature, subject to thermal EOVs, demonstrating robust damage detection and EOV recovery. The method is then applied to a simulated nine-turbine offshore wind farm with staggered deployment and damage, where it delivers a substantial true-positive uplift over projection and cointegration-based baselines at matched false-positive rates.

## 1 Introduction

Structural health monitoring (SHM) seeks to detect, locate, and characterise structural damage from measured response data [1]. A persistent obstacle is environmental and operational variability (EOV); ambient temperature, wind speed, and operational load shifts all alter the measured feature distribution, causing false alarms if damage detection is framed as distributional outlier detection on the raw features [2].

To date, many approaches have been proposed to manage the effects of EOVs [3]. The central distinction between existing paradigms is the observability of the EOV. In the case that the EOV signal is observed (e.g. measured temperatures, operational logs etc.), regression<sup>1</sup> approaches model the feature-EOV relationship directly and detect damage in the regression residuals [4].

When the EOV is unmeasured<sup>2</sup>, the EOV subspace must be inferred from the feature data itself. The dominant paradigms are cointegration and projection-based removal. In the former case, the existence of a cointegrating relationship is assumed, whereby the features are individually non-stationary, driven by a shared integrated environmental trend, yet a linear combination of them is stationary [5, 6]. Damage is inferred by the violation of stationarity. Latent-variable treatments have also been proposed; factor analysis has been used to eliminate unmeasured EOVs and to distinguish their effects from those of sensor faults and damage [7, 8].

This paper focuses on the case of unmeasured EOVs. For projection-based approaches such as minor component analysis (MCA) [9, 10], the identifying assumption is variance-dominance, the top principal components (or other similarly-identified subspaces) of healthy feature data are discarded, with the reasoning that they are dominated by the EOV. However, for structural systems where damage induces proportional shifts across all monitored vibration modes (e.g. global foundation degradation from scour or corrosion) the damage direction may be close to collinear with the EOV direction, and the removal of the high-variance subspace removes the damage signature alongside the EOV.

An alternative to variance-dominance is slowness; the identifying assumption is that the EOV varies on longer timescales than damage, noise or the structural dynamics, so it can be identified by a latent process with a long-correlation prior, regardless of its variance signature [11, 12]. So-called slow feature analysis (SFA) is an approach also used for nuisance-variable removal in other monitoring domains; e.g. condition monitoring [13]. A common limitation of slowness-based removal, however, is that abnormalities with a gradual onset (e.g. crack onset and propagation) can themselves be absorbed by the slow latent process, masking the presence of damage.

The challenges facing both variance-dominance and slowness approaches can be amplified in a populationbased setting. Structures in a population may share a common environmental driver yet differ in their perstructure response to EOVs. Furthermore, assets commissioned (or instrumented) at different times accumulate healthy training data at different rates, leaving newly deployed members with poorly estimated loadings at the critical early-detection stage. However, methodologies that share data between structures can improve robustness to EOVs compared to the single-structure setting.

Contribution This work proposes a shared Gaussian-process latent EOV (GLEOV)<sup>3</sup> model for simultaneous identification and removal of EOVs across a population of structures. The latent EOV signal is modelled as a Matern Gaussian process (GP) with a long lengthscale. The result of [14] is used to cast the GP in´ a linear-Gaussian, state-space form, enabling efficient computation of the marginal likelihood and model predictions via a Kalman-filtering framework. The same state-space treatment of Gaussian processes underpins the Gaussian-process latent force models used for joint input-state-parameter estimation in structural dynamics [15].

In the proposed approach, the latent EOV is identified by its temporal correlation structure rather than its variance, avoiding the limitations of variance-dominance as an assumption for EOV removal. The latent EOV is shared across the population under a hierarchical prior on per-structure EOV loadings. Data-rich structures anchor the shared EOV and lend statistical strength to data-poor neighbours. Damage is detected using the Kalman innovations (the residuals on the projected EOVs) as damage sensitive features, with an approximate Laplace posterior supplying an uncertainty-aware exceedance probability at any given threshold.

The principal contributions of this work are:

• A Gaussian-process latent EOV (GLEOV) model for efficient simultaneous identification and removal of EOVs in SHM.

• A hierarchical Bayesian extension to the case of populations of structures.

• Validation on an experimental single-structure benchmark from the literature, and evaluation on a nineturbine synthetic offshore wind farm with staggered deployment and damage acting along the EOV direction, demonstrating true-positive uplift over projection-based baselines at matched false-positive rates.

Related work The problem of accounting for EOVs in SHM has been well studied in the literature [16]. Much of the research effort has been focussed on the treatment of single structures and observed EOVs [17, 18, 19]. However, there are several works that consider unobserved EOVs. In [20], a projection-based approach based on minor-component analysis is compared to other implicit methods including cointegration [5, 6]. Although successful, these results are limited to a single structure and do not recover the latent EOV signal.

Several approaches based on SFA (that can recover unobserved and observed EOVs) have also been proposed in process monitoring and SHM [13, 21], although these also have been focussed on single structures.

Additional single-structure treatments are to be found in the literature, using similar identifying assumptions. A localised principal-component method for environment-induced bridge modal variability [22] uses a variance-dominance argument and applies the projection locally rather than globally, while a nonlinear system-identification approach for output-only monitoring under changing environments [23] removes the environmental influence implicitly via kernel PCA, a nonlinear extension of the variance-dominance assumption. As with the works above, both are confined to a single structure and share no information across a population. A Kalman-filtering approach is adopted in [24], where the filter innovations serve as damage-sensitive features under changing environmental conditions, although again for a single structure and without recovery of the latent environmental signal. In [25], the unmeasured EOV instead enters a random-coefficient functional model as a latent scheduling variable, although it is marginalised rather than recovered. What unites all of these methods is that the environmental process is treated as a nuisance to be removed. Conversely, the authors of [26] invert this view, treating the slow latent process as the signal rather than the nuisance and inferring a monotonic degradation state from fast operational dynamics with a hierarchical neural controlled differential equation.

The treatment of EOVs in a population-based context has had comparatively less attention in the literature. The foundations of PBSHM are set out in [27, 28, 29], and in [16] the major challenges are outlined, including the treatment of EOVs. An earlier population treatment is given in [30], using unsupervised multiple-model time-series methods across nominally-identical structures, but without an explicit EOV model. Several papers employ hierarchical Bayesian methods to population-based SHM [31, 32]. While these approaches are able to recover functional relationships between observed features and EOVs, they rely on at least partial observation of the EOV signal, in contrast to the fully-unobserved setting considered here. Closer to the present setting, a hierarchical Bayesian approach to anomaly detection in the natural frequencies of offshore wind turbines under temperature variation is presented in [33], although the EOV is not there recovered as a latent signal.

Two relevant works in this vein are a GP regression-surface approach to EOVs in [34, 35] and an adaptive methodology in [36] that combines data from two nominally-identical structures, governed by a mixing hyperparameter, to build adaptive stochastic models of the damage-sensitive features. This latter work demonstrates results on a population of two structures in a manner related to the present work, but it does not recover the underlying latent EOV signal.

Perhaps closest to the present work is the single-structure method of [37], which combines a Gaussian process with probabilistic PCA to account for both observed and latent EOVs. There, the GP models the structural response as a function of the measured EOVs, while PCA removes the unmeasured component. By contrast, GLEOV represents the unobserved EOV itself as a slow latent Gaussian process that requires no EOV observations, shares this latent process across a population of structures under a hierarchical prior, and retains O(T) state-space inference.

Although many approaches have been proposed to address the problem of EOVs in SHM, the combination of slow-latent EOV removal with population-level information sharing, so that data-rich structures anchor the shared EOV and provide an informative prior for data-poor neighbours, has not been treated in a unified probabilistic framework.

Paper structure The remainder of this paper is structured as follows: The following section describes the proposed method. A third section presents two case-study examples of the approach on a single-structure laboratory dataset and a simulated population of structures respectively. A final section presents some discussion and conclusions.

## 2 The Gaussian-process latent EOV (GLEOV) model

Generative model In this work, a data-generating process of the following form is assumed. For structure $i \in \{ 1 , \ldots , N \}$ at time t, the observed M-dimensional feature vector $\pmb { x } _ { i , t } \in \mathbb { R } ^ { M }$ is modelled as,

$$
{ \pmb x } _ { i , t } = { \pmb \mu } _ { i } + { \cal W } _ { i } ( z _ { t } + \rho _ { i , t } ) + \epsilon _ { i , t } , \quad \epsilon _ { i , t } \sim \mathcal { N } \left( \mathbf { 0 } , \sigma _ { e } ^ { 2 } I _ { M } \right) , \quad { \pmb \rho } _ { i , t } \sim \mathcal { N } \left( \mathbf { 0 } , \tau _ { T } ^ { 2 } I _ { K } \right) ,\tag{1}
$$

where $\pmb { \mu } _ { i } \in \mathbb { R } ^ { M }$ is the per-structure mean, $z _ { t } \in \mathbb { R } ^ { K }$ are the shared slow latent EOVs. Intuitively, this treats the observed features in time as a Gaussian, i.i.d. process, per-structure, corrupted additively by a correlated latent process z via the action of the per-structure EOV loading weights $W _ { i }$ . The $\rho _ { i , t }$ term accounts for the fact that even shared EOV signals (e.g. temperature) may vary stochastically across the population.

Note that the above can be re-written in the form,

$$
{ \pmb x } _ { i , t } = { \pmb \mu } _ { i } + W _ { i } { \boldsymbol z } _ { t } + { \pmb \nu } _ { i , t } , \quad { \pmb \nu } _ { i , t } \sim \mathcal { N } \Big ( \mathbf { 0 } , \ \sigma _ { e } ^ { 2 } { \pmb I } _ { M } + \boldsymbol \tau _ { T } ^ { 2 } W _ { i } W _ { i } ^ { \top } \Big ) ,\tag{2}
$$

Thus, given estimates of the weightings, means and latent EOV signal, an EOV-removed residual can be computed as,

$$
\pmb { \nu } _ { i , t } = \pmb { x } _ { i , t } - \pmb { \hat { \mu } _ { i } } - \hat { W } _ { i } \hat { z } _ { t | t - 1 } ,\tag{3}
$$

where ˆ· denotes an estimated quantity. In the absence of damage, this residual is Gaussian and independent across time, with an approximately stationary covariance, making it an ideal candidate for novelty detection in SHM. A probabilistic graphical model of the proposed data-generating process is given in Figure 1.

![](images/2bbb1ae0ef9294966b431af2bab4175ff8211653b62c670a9236401afba51123.jpg)  
Figure 1: Probabilistic graphical model of the GLEOV generative process. The shared latent EOV $z _ { t }$ evolves as a temporal Gaussian process (self-loop) and is mapped to the observed features $\mathbf { \Delta } _ { \mathbf { x } _ { i , t } }$ (shaded node) by the per-structure loading $W _ { i } ,$ offset by the per-structure mean $\mu _ { i } ;$ loadings are drawn from a population consensus $W _ { 0 }$ . Plates index the latent dimension (K), time $( T )$ , and structure $( N )$

Priors In this work, the shared latent $z _ { t }$ is given a zero-mean Matern-3/2 GP´ $\mathrm { p r i o r } ^ { 4 }$ . It can be shown that such a GP can be written as a linear-Gaussian state-space-model as [14],

$$
\left[ \boldsymbol { \xi } _ { t } \right] = \boldsymbol { A } _ { d } \left[ \begin{array} { l } { \boldsymbol { z } _ { t - 1 } } \\ { \dot { \boldsymbol { z } } _ { t - 1 } } \end{array} \right] + { \bf w } _ { t } , \qquad { \bf w } _ { t } \sim \mathcal { N } ( \mathbf { 0 } , \boldsymbol { Q } _ { d } ) , \qquad \left[ \begin{array} { l } { \boldsymbol { z } _ { 0 } } \\ { \dot { \boldsymbol { z } } _ { 0 } } \end{array} \right] \sim \mathcal { N } ( \mathbf { 0 } , P _ { \infty } )\tag{4}
$$

$$
A _ { d } = e ^ { - \lambda \Delta _ { t } } \left[ \begin{array} { c c } { 1 + \lambda \Delta _ { t } } & { \Delta _ { t } } \\ { - \lambda ^ { 2 } \Delta _ { t } } & { 1 - \lambda \Delta _ { t } } \end{array} \right] , \quad P _ { \infty } = \left[ \begin{array} { c c } { 1 } & { 0 } \\ { 0 } & { \lambda ^ { 2 } } \end{array} \right] , \quad Q _ { d } = P _ { \infty } - A _ { d } P _ { \infty } A _ { d } ^ { \intercal }\tag{5}
$$

Here $\lambda = { \sqrt { 3 } } / \ell$ , where ℓ is the GP lengthscale and $\Delta _ { t }$ the sampling period. The matrix $P _ { \infty }$ is the stationary covariance of the state at which the filter is initialised so that the recursion begins from the $\mathrm { G P ^ { \circ } s }$ stationary distribution. Note that in order to avoid identifiability issues with the magnitude of the loading vectors, the signal variance of the GP is here assumed to be equal to 1 without loss of generality.

The loadings receive an informative hierarchical prior, $W _ { i } \sim \mathcal { N } ( W _ { 0 } , 1 0 ^ { - 6 } I )$ , it is reasoned here that the EOV loading direction $W _ { i }$ will be highly similar between nominally identical structures, and that data-poor structures must borrow information from the population in order to identify it accurately. In contrast, the per-structure means receive a deliberately uninformative prior, $\pmb { \mu } _ { i } \sim \mathcal { N } ( \mathbf { 0 } , 1 0 ^ { 2 } I )$ , with the reasoning that the mean values will be structure specific and not necessarily shared among population members. All prior distributions and hyperparameters are listed in Table 1.

Both the GP lengthscale and time step are considered fixed in this work; the lengthscale is expressed in units of the sampling period $\Delta _ { t }$ . In practice, the value of the lengthscale can be set according to knowledge of the population. For example, an annual seasonal trend could be given a lengthscale parameter in the range 50–150 days, whereas daily variation would require a shorter lengthscale on the order of hours. Note that the fixed value of the lengthscale encodes the assumption that the dynamics of the latent GP are slow. The robustness of the method to this choice is confirmed by a sensitivity study presented in Appendix A.

Table 1: Prior distributions and fixed hyperparameters of the GLEOV model.
<table><tr><td>Parameter</td><td>Symbol</td><td>Prior / value</td></tr><tr><td colspan="3">Priors</td></tr><tr><td>Per-structure mean</td><td> $\pmb { \mu } _ { i }$ </td><td> $\overline { { \mathcal { N } ( \mathbf { 0 } , 1 0 ^ { 2 } I ) } }$ </td></tr><tr><td>Per-structure loading</td><td> $W _ { i }$ </td><td> $\mathcal { N } ( W _ { 0 } , 1 0 ^ { - 6 } I )$ </td></tr><tr><td>Consensus loading</td><td> $W _ { 0 }$ </td><td> $\mathrm { { f l a t } } \left( \propto 1 \right)$ </td></tr><tr><td>Measurement-noise scale</td><td> $\sigma _ { e }$ </td><td>flat (∝ 1) on log  $\sigma _ { e }$ </td></tr><tr><td>Population EOV scale</td><td> $\tau _ { T }$ </td><td> $\log \tau _ { T } \sim \mathcal { N } ( \log 0 . 1 , 1 )$ </td></tr><tr><td colspan="3">Hyperparameters (fixed)</td></tr><tr><td>GP lengthscale</td><td> $\ell$ </td><td>100 (units of  $\Delta _ { t } )$ </td></tr><tr><td>Sampling period</td><td> $\Delta _ { t }$ </td><td>1</td></tr><tr><td>Latent EOV dimension</td><td> $K$ </td><td>1</td></tr><tr><td>Gating false-positive rate</td><td> $\alpha _ { \mathrm { g a t e } }$ </td><td>0.01</td></tr><tr><td>Detection false-positive rate</td><td> $\alpha$ </td><td>0.001</td></tr></table>

## 2.1 Inference

The model is learned from data using a recursive Bayesian estimation approach based on a Kalman filter. Because the GP prior is discretised exactly as a linear-Gaussian SSM, $z _ { t }$ can be marginalised (conditional on the parameters $\theta = \{ \mu , W , W _ { 0 } , \sigma _ { e } , \tau _ { T } \} )$ in closed form [38]. Let $m _ { t }$ and $P _ { t }$ be the filtering mean and covariance at time t. Let also H be a suitable selection matrix such that $z _ { t } = H m _ { t }$ . Then the Kalman filter equations are given by,

$$
p \bigg ( \bigg [ \frac { z _ { t } } { \dot { z } _ { t } } \bigg ] \bigg | x _ { 1 : t - 1 } , \theta \bigg ) = \mathcal { N } ( m ^ { - } , P ^ { - } ) , \qquad m ^ { - } = A _ { d } m _ { t - 1 } , \qquad P ^ { - } = A _ { d } P _ { t - 1 } A _ { d } ^ { \top } + Q _ { d } .\tag{6}
$$

$$
p ( x _ { t } \mid x _ { 1 : t - 1 } , \theta ) = \mathcal { N } ( \mu + W \hat { z } _ { t \mid t - 1 } , S _ { t } ) , \qquad \hat { z } _ { t \mid t - 1 } = H m ^ { - } , \qquad S _ { t } = W H P ^ { - } H ^ { \top } W ^ { \top } + R e V ^ { \top } W ^ { \top }\tag{7}
$$

where,

$$
R = \mathrm { b l o c k d i a g } ( R _ { 1 } , \ldots , R _ { N } ) , \qquad R _ { i } = \sigma _ { e } ^ { 2 } { \cal I } _ { M } + \tau _ { T } ^ { 2 } W _ { i } W _ { i } ^ { \top }\tag{8}
$$

is the block-diagonal observation noise model for each structure in the population.

$$
p \bigg ( \bigg [ \mathcal { z } _ { t } \bigg ] \bigg | x _ { 1 : t } \bigg ) = \mathcal { N } ( m _ { t } , P _ { t } ) , \qquad m _ { t } = m ^ { - } + K _ { t } \nu _ { t } , \qquad P _ { t } = P ^ { - } - K _ { t } S _ { t } K _ { t } ^ { \top } ,\tag{9}
$$

with Kalman gain $K _ { t } = P ^ { - } H ^ { \top } W ^ { \top } S _ { t } ^ { - 1 }$ and innovation $\nu _ { t } = x _ { t } - \mu - W \hat { z } _ { t | t - 1 }$ . The marginal likelihood can thus be computed recursively as the product of the one-step-ahead predictive densities,

$$
p ( x _ { 1 : T } \mid \theta ) = \prod _ { t = 1 } ^ { T } { \mathcal { N } } ( x _ { t } \mid \mu + W { \hat { z } } _ { t \mid t - 1 } , S _ { t } ) .\tag{10}
$$

However, the above is still conditional on the unknown parameters θ. Unfortunately, the full posterior distribution is intractable, and so a Laplace approximation is employed.

MAP The maximum a posteriori (MAP) values of the unknown parameters ${ \widehat { \theta } } ,$ are computed by minimisation of the negative log joint distribution,

$$
{ \hat { \theta } } = \arg \operatorname* { m i n } _ { \theta } - \log p ( x _ { 1 : T } \mid \theta ) - \log p ( \theta )\tag{11}
$$

where $p ( \theta )$ is given by the product of the parameter priors above. Here, the optimisation is run using the L-BFGS-B solver [39, 40], as implemented in scipy [41].

To aid convergence, the optimisation is preconditioned: L-BFGS-B is run in a whitened space in which each parameter block is rescaled by a counting-heuristic estimate of its Fisher information $( \propto 1 / \sqrt { n _ { \mathrm { o b s } } } )$ . Without this step the gradient magnitudes across the heterogeneous blocks (loadings, means, and log-scaled noise terms) differ by orders of magnitude and the optimiser fails to converge.

The optimiser is initialised at the pooled-PCA solution: The per-structure means $\pmb { \mu } _ { i }$ are set to the observed training-window means, and the consensus loading $W _ { 0 }$ and every per-structure loading $W _ { i }$ are initialised to the $\mathrm { t o p } { - } K$ right singular vectors of the per-structure mean-centred training features, pooled across all structures. The noise scales are initialised to a fraction of the residual spread $( \sigma _ { e }$ at one tenth of the pooled standard deviation) and to $\tau _ { T } = 0 . 1$ . From this warm start, L-BFGS-B refines the loadings under the GP prior, replacing the purely variance-based initial subspace with one that additionally respects the temporalcorrelation structure of the latent. Because each marginal-likelihood evaluation is linear in the series length, the fit is inexpensive in practice; the full nine-turbine population model here (Section 3.2, 365 observations) converges in approximately 70 s on a single CPU core.

Laplace approximation The Laplace approximation is a Gaussian surrogate centred at the MAP estimate, with covariance the inverse Hessian of the negative log-joint at the mode,

$$
p ( \theta \mid \alpha _ { 1 : T } ) \approx \mathcal { N } \big ( \hat { \theta } , \hat { \Sigma } \big ) , \qquad \hat { \Sigma } = \big [ - \nabla _ { \theta } ^ { 2 } \log p ( \theta , x _ { 1 : T } ) \big | _ { \hat { \theta } } \big ] ^ { - 1 }\tag{12}
$$

The Hessian matrix is evaluated exactly by automatic differentiation of the negative log-joint at the mode, and hence $\hat { \Sigma }$ is obtained by its inversion.

## 2.2 Damage detection

Once the model is identified, the EOV-removed residual for each structure is the Kalman innovation,

$$
\pmb { \nu } _ { i , t } = \pmb { x } _ { i , t } - \pmb { \hat { \mu } _ { i } } - \hat { W } _ { i } \hat { \pmb { z } } _ { t | t - 1 }\tag{13}
$$

Under healthy conditions, (and the assumption of a well-specified model) these residuals are zero-mean, Gaussian, and i.i.d. Thus, the onset of damage can be inferred as a departure from their healthy distribution. To obtain estimates of the normal condition, a Gaussian is fitted to the training-window innovations for each structure,

$$
\begin{array} { r } { \hat { \pmb { \mu } } _ { i } ^ { \nu } = \mathbb { E } \left[ \pmb { \nu } _ { i , t } \right] , \quad \hat { \Sigma } _ { i } ^ { \nu } = \mathbb { V } \left[ \pmb { \nu } _ { i , t } \right] , \quad \forall t \in \mathrm { T r a i n i n g ~ w i n d o w } } \end{array}\tag{14}
$$

and score every time step by the Mahalanobis distance (MD) to this empirical reference [42],

$$
D _ { i , t } ^ { 2 } = ( \pmb { \nu } _ { i , t } - \hat { \pmb { \mu } } _ { i } ^ { \nu } ) ^ { \top } ( \hat { \Sigma } _ { i } ^ { \nu } ) ^ { - 1 } ( \pmb { \nu } _ { i , t } - \hat { \pmb { \mu } } _ { i } ^ { \nu } )\tag{15}
$$

For healthy structures, the MD follows a chi-squared distribution with M degrees of freedom, $D _ { i , t } ^ { 2 } \sim \chi _ { M } ^ { 2 } { } ^ { 5 }$ A detection threshold $\tau$ is therefore obtained at a tolerable false-positive rate α as the corresponding upper quantile, $\tau = \chi _ { M } ^ { 2 } ( 1 { - } \alpha )$ , the value exceeded by only a fraction of α healthy observations. Here, and throughout, $\chi _ { M } ^ { 2 } ( \cdot )$ denotes the quantile function (inverse CDF) of the chi-squared distribution with M degrees of freedom.

Robust observation gating A complication of the proposed approach in the population-based setting is that damage has the potential to bias the EOV estimation. An anomalous observation, if assimilated by the Kalman filter, is partially absorbed into the shared latent $z _ { t } ,$ potentially masking future damage and contaminating the estimate for every other structure. To address this, an online gating strategy that downweights highly anomalous observations is applied, in the spirit of robust Kalman filtering approaches that guard the state estimate against outlying measurements [43, 44].

At each test step and for each structure, a predictive gating distance $G _ { i , t } ^ { 2 } ,$ is computed from the scaled Kalman innovations and predictive covariance,

$$
G _ { i , t } ^ { 2 } = \frac { 1 } { d _ { i } } \pmb { \nu } _ { i , t } ^ { \top } S _ { i , t } ^ { - 1 } \pmb { \nu } _ { i , t }\tag{16}
$$

with,

$$
d _ { i } = \frac { 1 } { M T _ { \mathrm { t r a i n } } } \sum _ { t \in T _ { \mathrm { t r a i n } } } \nu _ { i , t } ^ { \top } S _ { i , t } ^ { - 1 } \nu _ { i , t } ,\tag{17}
$$

being the average per-structure dispersion, estimated on the training window. This normalised-innovationsquared statistic is the same quantity used to form validation gates in target tracking [45] and to test the whiteness of the innovation sequence for fault detection in dynamic systems [46]. Observations that exceed the threshold $G _ { i , t } ^ { 2 } > \chi _ { M } ^ { 2 } ( 1 - \mathrm { \bar { \alpha } } _ { \mathrm { g a t e } } )$ are reasoned to be sufficiently outlying that their inclusion is at risk of biasing the latent EOV removal and are thus excluded. Note that the gating procedure is applied perobservation. In this way, it is possible to exclude an outlier but include an observation afterwards. This is in contrast to some damage detection schemes that exclude all observations after an outlier is observed.

It is noteworthy that the gating procedure allows one to be much more aggressive with the threshold selection than might otherwise be possible during damage detection. The objective is not to classify observations as damaged, only to exclude samples suspected of being damaged such that they do not bias the estimation. It is therefore possible to set the value of $\alpha _ { \mathrm { g a t e } }$ to a very permissive false-positive rate (FPR) without necessarily incurring a corresponding cost of many false activations in practice.

Finally, the Laplace posterior is propagated at each step to turn the point estimate into an uncertainty-aware decision signal. Drawing samples $\hat { \theta ^ { ( s ) } } \sim \mathcal { N } ( \hat { \theta } , \Sigma )$ and recomputing $D ^ { 2 }$ for each, the decision signal is thus the posterior exceedance probability, given by,

$$
\mathrm { P r } \left( D _ { i , t } ^ { 2 } > \tau \right) \approx \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \mathbb { I } \big [ D _ { i , t } ^ { 2 , ( s ) } > \tau \big ]\tag{18}
$$

for $S$ samples, where $\tau = \chi _ { M } ^ { 2 } ( 1 - \alpha )$ for some target FPR $\alpha ,$ and $\mathbb { I } [ \cdot ]$ is an indicator function. Values near zero indicate a confidently healthy structure, and near unity a confident detection. An overview of the proposed approach is given in Algorithm 1.

Algorithm 1 GLEOV: simultaneous EOV identification and damage detection   
Require: features ${ \bf { x } } _ { i , t }$ , commissioning masks, lengthscale ℓ, period $\Delta _ { t } ,$ , priors p(θ), α<sub>gate</sub>, α   
Identification ▷ training window   
1: form the Matern SSM´ $( A _ { d } , Q _ { d } , P _ { \infty } )$ from $( \ell , \Delta _ { t } )$ ▷ Eq. 5   
2: $\hat { \theta } \gets \mathbf { M A P }$ estimate ▷ Eq. 11, L-BFGS-B [39, 40]   
3: Σ<sup>ˆ</sup> ← Laplace covariance; draw $\theta ^ { ( s ) } \sim \mathcal { N } ( \hat { \theta } , \hat { \Sigma } ) , s = 1 , \dots , S$ ▷ Eq. 12   
4: run KF at $\hat { \theta }$ over training; fit normal condition $( \hat { \mu } _ { i } ^ { \nu } , \hat { \Sigma } _ { i } ^ { \nu } )$ and dispersion $d _ { i }$ ▷ Eqs. 14, 17   
Robust observation gating   
5: for $t = 1 , \dots , T$ do   
6: KF predict; form innovation $\nu _ { i , t }$ and predictive covariance $S _ { i , t }$ ▷ Eqs. 3, 7   
7: $\mathbf { i f } t > T _ { \mathrm { t r a i n } }$ and $G _ { i , t } ^ { 2 } > \chi _ { M } ^ { 2 } ( 1 - \alpha _ { \mathrm { g a t e } } )$ then ▷ Eq. 16   
8: mark $x _ { i , t }$ as gated and exclude from observation model   
9: end if   
10: KF update using non-gated observations   
11: end for   
Damage detection   
12: for $\theta ^ { \prime } \in \{ \hat { \theta } , \theta ^ { ( 1 ) } , \dots , \theta ^ { ( S ) } \}$ do   
13: run KF without gated observations; score $D _ { i , t } ^ { 2 } ( \theta ^ { \prime } )$ ▷ Eq. 15   
14: end for   
15: compute exceedance probability $p _ { i , t }$ ▷ Eq. 18   
16: return MAP score $D _ { i , t } ^ { 2 } ( \hat { \theta } )$ and exceedance probability $p _ { i , t }$

## 3 Case studies

In order to demonstrate the effectiveness of the proposed GLEOV approach, two case-study examples of detection and removal of EOVs in SHM datasets are presented.

## 3.1 Single structure

Before demonstrating the proposed approach on a population of structures, it is important to first establish that the method is effective in identification and removal of EOV signals in the case of a single structure $( N = 1 )$ . The first case study is therefore a laboratory-scale benchmark SHM dataset. The dataset under consideration comes from a Lamb-wave inspection investigation of a composite panel subject to temperature variations in an environmental chamber. The data were collected as part of the Brite-Euram project DAMAS-COS (BE97 4213) [20]. The dataset consists of Lamb-wave recordings collected every minute, comprising three distinct phases.

• Phase I: For the first 1355 minutes, the chamber temperature was held at approximately $2 5 \mathrm { { ^ \circ C } }$

• Phase II: For the next 1128 minutes, the chamber temperature was varied between $1 0 ^ { \circ } \mathrm { C }$ and $3 0 ^ { \circ } \mathrm { C }$ with a cycle time of approximately 370 minutes.

• Phase III: At minute 2483, the chamber was opened and damage was introduced by drilling a 10mm hole in the centre of the plate. Data collection continued under the same $1 0 ^ { \circ } \mathrm { C } { - } 3 0 ^ { \circ } \mathrm { C }$ cycling; the present analysis uses the series up to minute 3700, giving 1217 minutes of post-damage monitoring.

The features under investigation are the magnitudes of 50 spectral lines in the vicinity of the peak of the frequency spectrum of the received Lamb-wave signal, these features are plotted in Figure 2. For a more complete description of the dataset and experimental setup, the interested reader is directed to [20]. A training set (comprised of data from minutes 1000 to 2000) was extracted from the overall dataset. In order to capture both the stationary and non-stationary behaviour of the features, the training set comprises examples from both phases I and II.

![](images/252d15e349910633f7794e818219231c1b980f0721f04c051a56412b6b532171.jpg)  
Figure 2: Raw features for the DAMASCOS dataset.

Because this dataset comprises only a single structure, an invariance exists between the two variance terms in equation (2). As such, the two terms cannot be separately identified and so $\rho$ is removed from the model $( \mathrm { i } . \mathbf { e } . \tau _ { T } = 0 )$

The GLEOV algorithm was applied to this dataset as per Algorithm 1, with prior distributions and parameters as in Table 1. Here, the sampling period is $\Delta _ { t } = 1$ minute, so the fixed lengthscale $\ell = 1 0 0$ corresponds to 100 minutes, roughly a quarter of the ∼370-minute thermal cycle. During the inference, S = 200 Laplace posterior samples are drawn. The posterior predictive MD values $( D _ { t } ^ { 2 } )$ are plotted in Figure 3. The samples from the distribution are extremely sensitive to the presence of damage $( \mathrm { A U C - R O C } = 1 . 0 0$ over the test window), while remaining insensitive to the variation in temperature. Also visible in the figure are the gated observation points. It is clear that the gating has successfully removed all observations corresponding to damage as well as some of the more outlying points in the healthy testing regime. It is interesting that although several healthy observations are gated (22.6%) this does not come at any cost to damage sensitivity or EOV reconstruction.

Figure 4 depicts the posterior over the inferred latent $\hat { z } _ { t }$ . It is remarkable that, despite never observing any temperature measurements, the recovered signal shows strong agreement with the qualitative thermal history of the campaign<sup>6</sup>. In the damage regime, all observations are excluded from the Kalman filter by the gating process and so in the absence of any information, the EOV returns correctly to the GP prior. In the context of a single structure, this behaviour is inevitable. However, in the context of a population of structures, the latent EOV can still be informed by healthy structures, as shown below.

## 3.2 Population of structures

To demonstrate the effectiveness of the proposed approach on a population of structures, a simulated dataset corresponding to an offshore wind turbine farm subject to an unmeasured temperature EOV is introduced. This dataset has been deliberately designed to offer a robust modelling challenge in that the damage and EOV are highly confounded and that the EOV signal dominates the variance of the features.

![](images/e0bb302292a1a07cd37d98fb3338148676847722ef0b3da84daf9e69c3297e0a.jpg)  
Figure 3: Mahalanobis $D ^ { 2 }$ detection trace (posterior samples) for the DAMASCOS dataset. Also shown (grey line) are the gated observations, for which $G _ { i , t } ^ { 2 } > \chi _ { M } ^ { 2 } ( 1 - \alpha _ { \mathrm { g a t e } } )$ that are excluded during prediction.

![](images/7dec1b708ff5d3f374f58f27c90336c9c8dfd7793110880352adad867256dad2.jpg)  
Figure 4: Inferred latent EOV signal $\hat { z } _ { t }$ (posterior samples) over the full monitoring series, note that although the ground truth temperature is unavailable, good agreement is seen with the qualitative description of the test campaign in [20].

Simulated offshore wind turbine farm The simulated population comprises $N = 9$ nominally-identical turbines, each monitored via $M = 3$ natural frequencies over $T = 7 3 0$ daily samples, split into a 365- day healthy training period and a 365-day test period with a sampling period of $\Delta _ { t } \ = \ 1$ day. A single environmental driver (temperature), shared across the farm, is considered. It is generated as a Matern-3/2´ Gaussian process in time with a 60-day lengthscale about a $9 \mathrm { { } ^ { \circ } C }$ mean. At a sampling rate of $\Delta _ { t } = 1$ day, daily temperature variations (e.g. day-night cycles) are considered to be averaged over and are not considered here. The temperature field is treated as common to the whole farm, with local Gaussian perturbations at each turbine. For features, the natural frequencies of the turbine’s monopile are simulated; each turbine’s features are set to depend affinely on its local temperature,

$$
\pmb { x } _ { i , t } = \pmb { \mu } _ { i } + \phi _ { i } ( z _ { t } + \pmb { \rho } _ { i , t } ) + \pmb { \varepsilon } _ { i , t }\tag{19}
$$

with,

$$
\varepsilon _ { i , t } \sim \mathcal { N } ( \mathbf { 0 } , \ \sigma _ { \varepsilon } ^ { 2 } I _ { M } ) , \quad \rho _ { i , t } \sim \mathcal { N } ( \mathbf { 0 } , \ \sigma _ { \rho } ^ { 2 } I _ { K } )\tag{20}
$$

The per-turbine baseline frequencies $\pmb { \mu } _ { i }$ and temperature sensitivities $\phi _ { i }$ are sampled heterogeneously across the population to account for inter-structure variance. Damage is introduced asynchronously to five of the nine turbines, with onsets staggered across the test window days 485 − 665. In order to ensure that this dataset represents a stiff challenge to SHM algorithms, damage is introduced primarily along the per-turbine thermal loading vector $\phi _ { i }$

$$
\Delta x _ { i , t } = \beta _ { i } \left( \phi _ { i } + 0 . 1 \left\| \phi _ { i } \right\| g _ { \perp } \right)\tag{21}
$$

with $\mathbf { \delta } \mathbf { \mathcal { g } } \bot \mathrm { ~ a ~ }$ unit vector orthogonal to $\phi _ { i }$ and where $\beta _ { i }$ is a damage severity parameter. Thus, the damage signal is highly confounded with the EOV, with only a small component that acts independently. The motivation for this choice of damage model (other than it being challenging from an identification point of view) is that under a proportional stiffness perturbation $\kappa  \kappa ( 1 \overset { \cdot } { + } \overset { \cdot } { \beta } )$ with mass unchanged (caused by either temperature variation or stiffness-reducing damage, e.g. scour) the natural frequencies shift as $f _ { n }  f _ { n } \sqrt { 1 + \beta } \approx f _ { n } ( 1 + \beta / 2 )$ to first order. The induced feature-space displacement therefore lies along the same direction as the thermal loading, independent of the severity. This suggests that when EOVs and damage both cause proportional stiffness changes, the resulting changes to the natural-frequency features can lie in the same subspace.

In order to demonstrate the benefits of the GLEOV approach in a population-based context, the dataset implements a synthetic staggered deployment. For each turbine in the population, the number of data available for training is spaced linearly from the full 365 down to 30 across the nine turbines. The simulated features, training windows and damage regions are depicted in Figure 5. All parameter values pertaining to the creation of the synthetic OWT farm are collected in Table 2. It is clear from the figure that this dataset represents a robust modelling challenge. The onset of damage is barely perceptible and is dwarfed by the variation caused by the EOV. Furthermore, several turbines have remarkably few data available for training, particularly T8 which sees only a 30-day window.

With the dataset established, the proposed GLEOV approach is applied as described above; the sampling period is $\Delta _ { t } = 1$ day, so the fixed lengthscale $\ell = 1 0 0$ corresponds to 100 days. During the inference, $S = 5 0 0$ Laplace posterior samples are drawn. In Figure $^ { 6 , }$ the posterior predictive MD $D _ { i , t } ^ { 2 }$ are plotted for all turbines in the population. As can be seen in the figure, the sensitivity to damage is excellent. Furthermore, the gated observations are plotted (coloured in grey). Here, it can be seen that almost all observations from damaged regimes are excluded. Although some false positive readings are visible in T8, it is encouraging that these readings are also excluded by the gating mechanism so that these observations do not bias the estimation of the latent EOV.

The latent EOV signal is plotted in Figure 7. Because this is a simulated dataset, the ground truth is available (although there is an affine invariance). In the figure, the latent EOV is aligned to the ground truth by a leastsquares fit to the affine transformation on the training window. As can be seen in the figure, the posterior prediction of the EOV is excellent. As expected, uncertainty shrinks over the course of the training window as data from more turbines become available. At the onset of damage, observations from damaged turbines are excluded, and the tracking remains strong. At the very end of the test window (with damage in over half of the population), performance begins to degrade and the tracking diverges slightly from the ground truth.

![](images/3a7350ddf5f09e93d53479c006d71601297ebde4818df8e8177d5f102f9572c8.jpg)  
Figure 5: Raw natural frequency features (scaled to zero-mean for plotting) for the simulated OWT farm.

![](images/65ec5e94077a1c42710e53e22b3ae7ff5c388d1416e7c082705225413928ceeb.jpg)  
Figure 6: GLEOV posterior distribution over the Mahalanobis distances $D _ { i , t } ^ { 2 } ,$ for the simulated OWT farm dataset. Also shown (grey line) are the gated observations, for which $G _ { i , t } ^ { 2 } > \chi _ { M } ^ { 2 } ( 1 - \alpha _ { \mathrm { g a t e } } )$ that are excluded during prediction.

Table 2: Parameters of the simulated nine-turbine offshore wind farm.
<table><tr><td>Parameter</td><td>Symbol</td><td>Value</td></tr><tr><td colspan="3">Dimensions</td></tr><tr><td>Turbines</td><td>N</td><td>9</td></tr><tr><td>Frequencies per turbine</td><td>M</td><td>3</td></tr><tr><td>Duration (train/test)</td><td>T</td><td>730 (365 /365) days 365 → 30 (linearly spaced)</td></tr><tr><td colspan="3">Training days per turbine Temperature EOV</td></tr><tr><td>Mean temperature GP amplitude standard deviation</td><td> $T _ { \mathrm { b a s e } }$ </td><td> $\overline { { 9 . 0 ^ { \circ } C } }$   $4 . 0 ^ { \circ } \mathrm { C }$   $6 0 \mathrm { d a y s }$ </td></tr><tr><td colspan="3">Per turbine noise standard deviation  $\sigma _ { \rho }$  Feature model</td></tr><tr><td>Baseline frequencies (per turbine) mean standard deviation Temperature sensitivity (per turbine)</td><td> $[ 5 , 1 0 , 1 5 ] \times 1 0 ^ { - 3 }$ </td><td> $[ 0 . 3 0 , 0 . 6 5 , 1 . 0 5 ] \mathrm { H z }$  Hz  $[ - 5 , - 1 0 , - 1 5 ] \times 1 0 ^ { - 4 } \mathrm { H z ^ { \circ } C ^ { - 1 } }$   $[ 1 , 2 , 3 ] \times 1 0 ^ { - 4 } \mathrm { H z ^ { \circ } C ^ { - 1 } }$ </td></tr><tr><td colspan="3">Frequency noise standard deviation  $\sigma _ { \varepsilon }$   $5 \times 1 0 ^ { - 4 } \mathrm { H z }$  Damage model</td></tr><tr><td>Damaged turbines</td><td> $\beta _ { i }$ </td><td>5 of 9 2.5-3.0</td></tr><tr><td>Severity (per damaged turbine) Off-axis fraction</td><td></td><td>0.10 days 485–665</td></tr></table>

In Figure 8, the exceedance probability is plotted for each turbine in the population, with the threshold set to a target FPR of $\alpha = 0 . 0 0 1$ on the training window. As can be seen in the figure, the exceedance probability is high in the damage regimes and remains close to 0 at almost all other times for all turbines. The exception to this trend is T8, the most data-poor member of the population, commissioned with only 30 days of healthy training data. This is not an unexpected result, for T8, the EOV loading direction $W _ { 8 }$ is only weakly identified, and the corresponding posterior over the $D ^ { 2 }$ is broad (as can be seen in Figure 6). The exceedance probability inherits this uncertainty. Although the exceedance probability is raised during the healthy portion of the testing window for T8, it remains lower than during the damaged region, indicating sensitivity to damage.

Baseline comparison GLEOV is compared here against five baselines, each scored by the same Mahalanobis distance statistic (equation (15)) to provide a fair comparison between methods. The first, raw features, performs no EOV removal and detects damage directly in the M-dimensional feature space. The next two are variants of projection-based minor-component analysis (MCA) [9, 10], the canonical variancedominance approach. In MCA, the top-K principal directions of the healthy training features (assumed to span the dominant environmental variation) are discarded by singular value decomposition, and detection is carried out in the residual (M − K)-dimensional minor-component subspace. The two MCA variants dif fer only in how this subspace is estimated. Per-structure MCA estimates it independently for each turbine, whereas pooled MCA estimates a single shared subspace jointly from the whole population.

![](images/cbf37df65be25b7cca275e0a2cfdd4475c5f1e541bbed7bc08f1be0aaf8a9c42.jpg)  
Figure 7: GLEOV posterior predictions of the latent temperature EOV $\hat { z } _ { t }$ for the simulated OWT farm dataset, for plotting these predictions are aligned affinely to the ground truth signal by least squares on the training data. The darkening red regions indicate damage onset events in the population with darker shading corresponding to a greater damaged proportion of the population.

The final two are cointegration baselines, the other canonical implicit EOV-removal paradigm [5, 6]. Here the features are assumed to share a common non-stationary environmental trend, and a stationary linear combination of them is sought that cancels this trend. Damage is then inferred by a loss of stationarity of that combination. The cointegrating vector is estimated by the Johansen procedure [47], following the approach of [5]. Detection is carried out on the resulting one-dimensional cointegration residual. As for MCA, a per-structure variant estimates the cointegrating vector independently per turbine, while a pooled variant estimates a single shared vector across the population.

Rather than present results at only a single threshold level, it is of interest to consider the performance of the GLEOV method and baselines across threshold levels. A convenient way to achieve this is by computing the receiver operating characteristic (ROC) curve [48]. The ROC traces the true-positive rate (TPR) against the false-positive rate (FPR) as the detection threshold is swept across its full range. A convenient single scalar summary value is available as the area under the ROC curve (AUC-ROC), which is equivalent to the probability that a randomly-chosen damaged observation is scored higher than a randomly-chosen healthy one [48], so that AUC = 1 denotes perfect separation and AUC = 0.5 implies chance.

Figure 9 shows the pooled ROC over all test observations for GLEOV and the five baselines. GLEOV attains an AUC of 0.965 in the OWT test and lies above every baseline at every operating point. The baselines remain close to the diagonal (per-structure MCA 0.605, raw features 0.600, pooled MCA 0.564, per-structure cointegration 0.537 and pooled cointegration 0.515), barely exceeding chance. This gap is a direct consequence of the adversarial damage design of equation (21). Because the damage acts predominantly along the EOV direction, the baselines that remove the environmental subspace (either by discarding its high-variance directions (MCA) or by cancelling the shared environmental trend with a stationary combination (cointegration)) remove the damage signature along with it and retain only the small off-axis residual. By identifying the EOV via its slowness rather than its variance, GLEOV separates damage from the environment in time and recovers the collinear component that the baselines remove.

The shaded region in Figure 9 quantifies the effect of parameter uncertainty. Each Laplace posterior sample yields its own ROC, and the band spans their 99.7% central interval, with the ROC curve corresponding to the MAP curve shown as the central line. The band is narrow, indicating that the posterior uncertainty in θ has little bearing on the relative performance of the GLEOV and baseline methods.

![](images/0ab569ea3e3a7ab128986240fdfa5d0794eb371833659b0f6b8eed2965fd747d.jpg)  
Figure 8: GLEOV exceedance probability $p _ { i , t }$ for each turbine in the OWT farm dataset at a threshold level targeting 0.001 FPR on the training window.

![](images/9ba07b886473211a4a904e11dabe975689ccefcb362e5b1bd43387a2f06c4560.jpg)  
Figure 9: Pooled ROC curves over all OWT test observations for GLEOV and the five baselines.

In Figure 10, the ROC curves are reported for each of the damaged OWTs in the dataset independently. GLEOV (solid) attains per-turbine AUCs of 0.92–0.99 in every panel, while the projection and cointegration baselines stay near the diagonal (0.41–0.77). The lower-right panel plots AUC against training-set size, with GLEOV high and approximately flat (dropping only to 0.92 on the most data-poor turbine) and the projection baselines below it at every level. The same panel also reports an ablation study in which the GLEOV population sharing is switched off and each turbine is fit independently, with no shared consensus loading $W _ { 0 }$ and no shared latent EOV (GLEOV, no pooling). Stripped of the population-level data sharing, detection collapses to the per-structure MCA at every training-set size (0.54–0.77).

This result is a clear motivation for the population-based approach proposed in this work; the shared EOV is anchored by the healthy members, while the data-poor loadings are additionally shrunk toward the consensus $W _ { 0 }$ . The most data-poor turbine, T8, with only 30 days of data, still benefits measurably from the pooling (AUC 0.77 → 0.92).

## 4 Discussion

This work has presented GLEOV, a Bayesian recursive-estimation framework for the simultaneous identi fication and removal of EOVs in population-based structural health monitoring. The unobserved EOV is represented by a latent Gaussian process in time, shared across the population and identified by its slowness rather than its variance. Cast in linear-Gaussian state-space form, the GP is marginalised exactly by a Kalman filter at $\mathcal { O } ( T )$ cost, while a hierarchical prior over the per-structure loadings lets data-rich structures lend statistical strength to data-poor ones. The approach has been demonstrated on two case studies: A single laboratory-scale structure subject to thermal variability, and a simulated population of offshore wind turbines with damage acting along the environmental direction under staggered deployment.

In the first case study, the framework was validated on real monitoring data from a single laboratory-scale structure subject to thermal variability. Without observing temperature, GLEOV recovered the latent environmental signal directly from the measured features, in agreement with the qualitative thermal history of the test campaign. The resulting EOV-removed residuals are highly damage sensitive (AUC = 1.00). This study demonstrates identification and detection performance on real-world data, before the benefits of population sharing are examined in the second case study.

![](images/93379dbaafe6d190bb3852150baa97b87dfb2cb514c3eb05d1e1b9b19b6feefb.jpg)  
Figure 10: Per-turbine ROC curves for each of the five damaged turbines, comparing GLEOV against the raw-feature, MCA and cointegration baselines. The final panel plots AUC against training-set size for each method, together with the no-pooling ablation of GLEOV in which each turbine is fit independently by the GLEOV approach.

The second case study examined the population setting on a simulated nine-turbine farm, constructed deliberately so that damage acts along the same feature-space direction as the temperature EOV. Here, GLEOV recovers with uncertainty the damage signal that is missed by the projection baselines. The pooled ROC is above every baseline at every operating point (AUC = 0.965) compared to the baseline results (0.52–0.61). This robustness to data scarcity is the central benefit of treating EOV removal within a population-based framework; the joint identification of shared latent EOVs means that the data-poor members are better able to remove the effect of the EOV by borrowing strength from other turbines with more observations.

Modelling a shared latent EOV is, however, not without risk. An anomalous observation on one structure can be absorbed into the common EOV and bias the estimate for the whole population. The gating strategy proposed in this work guards against this by withholding observations inconsistent with the predicted environmental response from the latent update. In this way, damage cannot be accidentally interpreted as environmental variation, overcoming one of the main limitations of SFA in SHM.

A further advantage of the proposed gating is that it can be operated far more permissively than the detection threshold. An operator would be free to gate observations liberally (accepting that some observations would be falsely excluded) without the corresponding cost of false positive activations (potentially very expensive if many superfluous inspections are triggered).

The Bayesian framework adopted in this work lends further robustness to the proposed detection approach. Propagating the Laplace posterior gives access to an uncertainty-aware exceedance probability that correctly concentrates uncertainty on the structures with the lowest amount of data available.

Although the proposed approach is shown here to be highly promising, there remain some important limitations. The first, is that the Laplace approximation, being a Gaussian surrogate centred at the mode, can be inaccurate when the true posterior departs substantially from Gaussianity [49]. Future work by the authors will address this by employing more advanced Bayesian inference techniques (Hamiltonian Monte-Carlo, Stochastic variational inference). Another limitation is the choice of an isotropic noise structure for $\epsilon _ { i , t }$ and $\rho _ { i , t }$ . If the EOVs or EOV weights vary on drastically different timescales this strong assumption may not be warranted and a more rich correlation structure would have to be considered. Also of interest are alternative methods for the MAP optimisation procedure used herein. Methods such as expectation maximisation [50, 51] may offer performance benefits over the LBFGS-B approach. A final limitation is that a single latent EOV dimension (K = 1) is used throughout both case studies, reflecting the single dominant EOV in each dataset. Where several environmental factors act concurrently (for example, temperature alongside wind or humidity), a higher-dimensional latent (K > 1) would be required. The state-space formulation accommodates this directly by inflating the latent state, but its behaviour in the population setting remains to be explored and is left to future investigation.

In this work, the lengthscale parameter of the Gaussian process has been treated as a fixed hyperparameter rather than inferred. Although the hyperparameter sensitivity study in Appendix A demonstrates broad robustness to this choice, incorporating ℓ as an inference parameter under a prior that enforces slowness nonetheless remains a promising avenue for future work.

In conclusion, unobserved environmental and operational variability remains a central obstacle to reliable SHM in a population-based context. By identifying the latent EOV through its temporal correlation, sharing it across a population, and robustly gating anomalous observations, GLEOV recovers the confounding trend rather than discarding variance, sharpening the separation between benign environmental change and genuine damage and offering a promising route towards more robust operation and maintenance decision-making.

## Statements and declarations

Ethical considerations

Not applicable.

Consent to participate

Not applicable.

Consent for publication

Not applicable.

Declaration of conflicting interest

Not applicable.

## Funding

The authors of this paper gratefully acknowledge the support of the UK Engineering and Physical Sciences Research Council (EPSRC) via grant reference EP/W005816/1 (ROSEHIPS). For the purpose of open access, the authors have applied a Creative Commons Attribution (CC BY) licence to any Author Accepted Manuscript version arising.

EJC, TJR and MRJ would like to acknowledge the support of Innovate UK through the OLLGA project grant 10040817.

MDC gratefully acknowledges the support of the Centre for Machine Intelligence (CMI) within the University of Sheffield.

MRJ gratefully acknowledges the support of the University of Sheffield through a Research Excellence Fellowship.

## Data availability

The experimental data analysed in the first case study were collected under the Brite-Euram project DAMAS-COS (BE97 4213) and are not owned by the authors; a description of the dataset and experimental campaign is given in [20].

The code implementing the GLEOV model and all baseline methods will be made available upon acceptance of the paper.

## References

[1] Charles R Farrar and Keith Worden. Structural Health Monitoring: a Machine Learning Perspective. John Wiley & Sons, 2012.

[2] Hoon Sohn, Keith Worden, and Charles R Farrar. Statistical damage classification under changing environmental and operational conditions. Journal of Intelligent Material Systems and Structures, 13(9):561–574, 2002.

[3] Hoon Sohn. Effects of environmental and operational variability on structural health monitoring. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 365(1851):539–560, 2007.

[4] Bart Peeters and Guido De Roeck. One-year monitoring of the Z24-Bridge: environmental effects versus damage events. Earthquake Engineering & Structural Dynamics, 30(2):149–171, 2001.

[5] Elizabeth J Cross, Keith Worden, and Qian Chen. Cointegration: a novel approach for the removal of environmental trends in structural health monitoring data. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 467(2133):2712–2732, 2011.

[6] Elizabeth J Cross and Keith Worden. Cointegration and why it works for SHM. In Journal of Physics: Conference Series, volume 382, page 012046, 2012.

[7] Jyrki Kullaa. Eliminating environmental or operational influences in structural health monitoring using the missing data analysis. Journal of Intelligent Material Systems and Structures, 20(11):1327–1336, 2009.

[8] Jyrki Kullaa. Distinguishing between sensor fault, structural damage, and environmental or operational effects in structural health monitoring. Mechanical Systems and Signal Processing, 25(8):2976–2989, 2011.

[9] A-M Yan, G Kerschen, P De Boe, and J-C Golinval. Structural damage diagnosis under varying environmental conditions—Part I: A linear analysis. Mechanical Systems and Signal Processing, 19(4):847–864, 2005.

[10] A-M Yan, G Kerschen, P De Boe, and J-C Golinval. Structural damage diagnosis under varying environmental conditions—Part II: Local PCA for non-linear cases. Mechanical Systems and Signal Processing, 19(4):865–880, 2005.

[11] Laurenz Wiskott and Terrence J. Sejnowski. Slow feature analysis: Unsupervised learning of invariances. Neural Computation, 14(4):715–770, 2002.

[12] Richard E Turner and Maneesh Sahani. A maximum-likelihood interpretation for slow feature analysis. Neural Computation, 19(4):1022–1038, 2007.

[13] Shumei Zhang and Chunhui Zhao. Slow-feature-analysis-based batch process monitoring with comprehensive interpretation of operation condition deviation and dynamic anomaly. IEEE Transactions on Industrial Electronics, 66(5):3773–3783, 2019.

[14] Jouni Hartikainen and Simo Sarkk¨ a. Kalman filtering and smoothing solutions to temporal Gaussian¨ process regression models. In 2010 IEEE International Workshop on Machine Learning for Signal Processing, pages 379–384, 2010.

[15] T. J. Rogers, K. Worden, and E. J. Cross. On the application of Gaussian process latent force models for joint input-state-parameter estimation: With a view to Bayesian operational identification. Mechanical Systems and Signal Processing, 140:106580, 2020.

[16] Keith Worden, Lawrence A Bull, Paul Gardner, Julian Gosliga, Timothy J Rogers, Elizabeth J Cross, Evangelos Papatheou, Weijiang Lin, and Nikolaos Dervilis. A brief introduction to recent developments in population-based structural health monitoring. Frontiers in Built Environment, 6:146, 2020.

[17] Luis David Avendano-Valencia, Eleni N Chatzi, and Dmitri Tcherniak. Gaussian process models for˜ mitigation of operational variability in the structural health monitoring of wind turbines. Mechanical Systems and Signal Processing, 142:106686, 2020.

[18] Keith Worden and E.J. Cross. On switching response surface models, with applications to the structural health monitoring of bridges. Mechanical Systems and Signal Processing, 98:139–156, 2018.

[19] Soren M¨ oller, Matthew R. Jones, Clemens Jonscher, Jasper Ragnitz, Elizabeth J. Cross, and Raimund¨ Rolfes. Grey-box Gaussian processes for mode shape normalisation: Damage localisation under environmental and operational variability. Mechanical Systems and Signal Processing, 253:114353, 2026.

[20] Elizabeth J Cross, Graham Manson, Keith Worden, and Stephen G Pierce. Features for damage detection with insensitivity to environmental and operational variations. Proceedings ofthe Royal Society A: Mathematical, Physical and Engineering Sciences, 468(2148):4098–4122, 2012.

[21] Zhen Wang, Xin-Yan Tang, Jian Zhang, Guo-Hong Liu, and Foysal Bin Shakil. A dynamic early warning method for bridge damage via data clustering-aided subdomain feature extraction considering nonlinear environmental effects. International Journal of Structural Stability and Dynamics, page 2750429, 2026.

[22] Zhen Wang, Ting-Hua Yi, Dong-Hui Yang, Hong-Nan Li, Guan-Hua Zhang, and Ji-Gang Han. Early anomaly warning of environment-induced bridge modal variability through localized principal component differences. ASCE-ASME Journal of Risk and Uncertainty in Engineering Systems, Part A: Civil Engineering, 8(4):04022044, 2022.

[23] Edwin Reynders, Gersom Wursten, and Guido De Roeck. Output-only structural health monitoring in changing environmental conditions by means of nonlinear system identification. Structural Health Monitoring, 13(1):82–93, 2014.

[24] Kalil Erazo, Debarshi Sen, Satish Nagarajaiah, and Limin Sun. Vibration-based structural health monitoring under changing environmental conditions using Kalman filtering. Mechanical Systems and Signal Processing, 117:1–15, 2019.

[25] T.-C.I. Aravanis, J.S. Sakellariou, and S.D. Fassois. A stochastic functional model based method for random vibration based robust fault detection under variable non-measurable operating conditions with application to railway vehicle suspensions. Journal ofSound and Vibration, 466:115006, 2020.

[26] Mengjie Zhao and Olga Fink. Disentangling slow and fast temporal dynamics in degradation inference with hierarchical differential models. Reliability Engineering & System Safety, 277:112943, 2027.

[27] L.A. Bull, P.A. Gardner, J. Gosliga, T.J. Rogers, N. Dervilis, E.J. Cross, E. Papatheou, A.E. Maguire, C. Campos, and K. Worden. Foundations of population-based SHM, Part I: Homogeneous populations and forms. Mechanical Systems and Signal Processing, 148:107141, 2021.

[28] J. Gosliga, P.A. Gardner, L.A. Bull, N. Dervilis, and K. Worden. Foundations of population-based SHM, Part II: Heterogeneous populations – graphs, networks, and communities. Mechanical Systems and Signal Processing, 148:107144, 2021.

[29] P. Gardner, L.A. Bull, J. Gosliga, N. Dervilis, and K. Worden. Foundations of population-based SHM, Part III: Heterogeneous populations – mapping and transfer. Mechanical Systems and Signal Processing, 149:107142, 2021.

[30] K.J. Vamvoudakis-Stefanou, J.S. Sakellariou, and S.D. Fassois. Vibration-based damage detection for a population of nominally identical structures: Unsupervised Multiple Model (MM) statistical time series type methods. Mechanical Systems and Signal Processing, 111:149–171, 2018.

[31] TA Dardeno, LA Bull, N Dervilis, and K Worden. A population form via hierarchical Bayesian modelling of the FRF. In Data Science in Engineering, Volume 10: Proceedings of the 41st IMAC, A Conference and Exposition on Structural Dynamics 2023, volume 10, pages 95–103. Springer Nature, 2023.

[32] T.A. Dardeno, K. Worden, N. Dervilis, R.S. Mills, and L.A. Bull. On the hierarchical Bayesian modelling of frequency response functions. Mechanical Systems and Signal Processing, 208:111072, 2024.

[33] S.M. Smith, A.J. Hughes, T.A. Dardeno, L.A. Bull, N. Dervilis, and K. Worden. Anomaly detection in offshore wind turbine structures using hierarchical Bayesian modelling, 2024. Presented at the 14th International Workshop on Structural Health Monitoring (IWSHM 2023).

[34] Weijiang Lin, Keith Worden, Andrew Eoghan Maguire, and Elizabeth J Cross. Towards populationbased structural health monitoring, part VII: EOV fields–environmental mapping. In Topics in Modal Analysis & Testing, Volume 8: Proceedings ofthe 38th IMAC, A Conference and Exposition on Structural Dynamics 2020, pages 297–304. Springer, 2020.

[35] Weijiang Lin, Keith Worden, Andrew E Maguire, and Elizabeth J Cross. A mapping method for anomaly detection in a localized population of structures. Data-Centric Engineering, 3:e25, 2022.

[36] Kevin Qu, Alasdair Logan, Euan Miller, and David Garcia Cava. Multi-phase adaptive methodology for mitigating environmental and operational variability in slowly changing time-variant engineering structures. Mechanical Systems and Signal Processing, 229:112494, 2025.

[37] Yi-Chen Zhu, Wen Xiong, and Xiao-Dong Song. Structural performance assessment considering both observed and latent environmental and operational conditions: A Gaussian process and probability principal component analysis method. Structural Health Monitoring, 21(6):2531–2546, 2022.

[38] Simo Sarkk¨ a and Lennart Svensson.¨ Bayesian Filtering and Smoothing. Institute of Mathematical Statistics Textbooks. Cambridge University Press, New York, 2nd edition, 2023.

[39] Richard H. Byrd, Peihuang Lu, Jorge Nocedal, and Ciyou Zhu. A limited memory algorithm for bound constrained optimization. SIAM Journal on Scientific Computing, 16(5):1190–1208, 1995.

[40] Ciyou Zhu, Richard H. Byrd, Peihuang Lu, and Jorge Nocedal. Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization. ACM Transactions on Mathematical Software, 23(4):550–560, 1997.

[41] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stefan J. van der Walt, Matthew´ Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, <sup>˙</sup>Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antonio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0:ˆ Fundamental algorithms for scientific computing in Python. Nature Methods, 17:261–272, 2020.

[42] Keith Worden, Graham Manson, and Nigel R. J. Fieller. Damage detection using outlier analysis. Journal ofSound and Vibration, 229(3):647–667, 2000.

[43] Jo-Anne Ting, Evangelos Theodorou, and Stefan Schaal. A Kalman filter for robust outlier detection. In 2007 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1514–1519, 2007.

[44] Gabriel Agamennoni, Juan I. Nieto, and Eduardo M. Nebot. An outlier-robust Kalman filter. In 2011 IEEE International Conference on Robotics and Automation (ICRA), pages 1551–1558, 2011.

[45] Yaakov Bar-Shalom and Thomas E. Fortmann. Tracking and Data Association, volume 179 of Mathematics in Science and Engineering. Academic Press, San Diego, CA, 1988.

[46] Raman K. Mehra and J. Peschon. An innovations approach to fault detection and diagnosis in dynamic systems. Automatica, 7(5):637–640, 1971.

[47] Søren Johansen. Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. Econometrica, 59(6):1551–1580, 1991.

[48] Kevin P. Murphy. Probabilistic Machine Learning: Advanced Topics. MIT Press, Cambridge, MA, 2023.

[49] Malte Kuss and Carl Edward Rasmussen. Assessing approximate inference for binary Gaussian process classification. Journal ofMachine Learning Research, 6(57):1679–1704, 2005.

[50] Arthur P Dempster, Nan M Laird, and Donald B Rubin. Maximum likelihood from incomplete data via the EM algorithm. Journal of the Royal Statistical Society: Series B (Methodological), 39(1):1–22, 1977.

[51] Robert H Shumway and David S Stoffer. An approach to time series smoothing and forecasting using the EM algorithm. Journal ofTime Series Analysis, 3(4):253–264, 1982.

## A Lengthscale sensitivity

Throughout this work the Gaussian-process lengthscale ℓ is treated as a fixed hyperparameter rather than inferred, encoding the slowness assumption that the latent EOV varies on longer timescales than damage, noise or the structural dynamics. To verify that the reported performance does not hinge on the precise value chosen, the population case study of Section 3.2 is re-run across a grid of lengthscales, with all other priors and hyperparameters held at the values of Table 1. At each ℓ, the full pooled model is refit and the pooled population AUC-ROC over the test window is reported. The results are shown in Figure 11.

As can be seen in the figure, the AUC-ROC performance across the population remains largely insensitive to ℓ across a large range of choices of lengthscale, encompassing the range of values that might be selected to model annual seasonal variation. AUC-ROC remains at ≈ 0.96 for lengthscales between 40–150 days, comfortably bracketing both the 60-day lengthscale of the simulated environmental field and the ℓ = 100 days value adopted in the main text.

Damage identification performance degradation is only seen at the extremes, and even there it remains far above the best projection baseline (0.61, Section 3.2). The fixed choice of ℓ is therefore not a significant limitation of the results in this work.

![](images/13ab4bf8c85f5fe023132a97e55c17333ef8a3f9b268db946ec6e7d86cdc5921.jpg)  
Figure 11: Pooled population AUC-ROC for the OWT case study as the GP lengthscale ℓ is swept, with all other hyperparameters and priors fixed as in the main text.