# �-VAEs as Efective Theories: Tolerance-Dependent Dimension

Johannes Hirn<sup>1</sup>

<sup>1</sup>Image Processing Laboratory (IPL), Universitat de València, Paterna, València 46980, Spain<sup>∗</sup>

In a �-VAE, increasing the regularization strength acts as a spectral cutof by collapsing low-utility latent coordinates.

In the linear Gaussian VAE, the collapse order matches the ranking of reconstruction utilities exactly, because both are set by the PCA spectrum. We ask which parts of this picture survive in fully connected nonlinear VAEs trained on WorldClim.

We find that nonlinear interactions shift and broaden collapse onsets, so thresholds no longer coincide exactly with utilities. However, the common ordering is preserved over the resolved ranks, so the spectral cutof still acts as a utility cutof and the efective-description logic carries through.

The resulting efective-dimension curves reveal a head–tail tradeof: increasing depth concentrates utility into the first few coordinates but worsens tail fidelity.

## I. INTRODUCTION

Many high-dimensional datasets admit descriptions that are substantially simpler than their ambient representation [1–4]. The manifold hypothesis summarizes this observation by saying that data often lie near lowerdimensional structure. The word “near” matters: it leaves open how much residual variation is treated as noise rather than modeled explicitly. In the idealized sharp-gap limit, a small number of coordinates captures the dataset, while additional coordinates contribute only near the noise floor.

In real datasets, however, there may be no sharp gap. In such cases, efective dimension is not a single integer, but depends on the task, model class, and tolerated error.

A tolerance is only meaningful once a task and distortion metric have been fixed. When no downstream objective is specified, reconstruction is the natural baseline task: it measures how much input variation is retained, even though it can preserve redundant or nuisance varia tion because these remain part of the target.

Here we fix reconstruction as the task, reconstruction distortion as the metric, and fully connected �-VAEs as the model class, with depth as the main architectural variable.

With task, metric, and model class fixed, tolerance becomes the main variable. We ask how efective dimension changes as the allowed reconstruction error is varied. This suggests an efective-theory viewpoint: choosing a reconstruction tolerance amounts to choosing a resolution, so that some variation is represented explicitly while the rest is left unresolved in the residual term. Unlike in field theory however, the residual is not absorbed into counterterms to keep predictions cutof-independent; it is allowed to increase as the cutof is raised.

In that broad Wilsonian sense, this is the efectivetheory logic we use here: the variables needed in the description can change with the chosen cutof [5, 6]. In a VAE, the KL weight � plays the role of a tunable cutof over latent coordinates. This is a cutof language for finite latent degrees of freedom, not a full field-theory framework of counterterms and cutof-independent observables. It also difers from RG analogies for deep networks, where layers are compared to coarse-graining transformations [7, 8]: here the question is not layer-by-layer flow, but the cutof-dependent survival of latent coordinates.

The measured object is therefore the rank–distortion curve: reading it at fixed tolerance � gives the efective dimension $d _ { \mathrm { e f f } } ( \epsilon )$ , while the associated marginal utilities give the ordered spectrum of reconstruction importance.

But a rank–distortion curve by itself is not enough. For the corresponding coordinates to serve as efective variables, they need to satisfy several properties: they should be shared across the dataset, few enough to inspect, ordered by marginal contribution, and controllable by a cutof so that a chosen tolerance selects an active subset. Sharp on/of thresholds are ideal but not required. Semantic interpretation is a further step: the spectrum tells us which variables deserve inspection, but it does not by itself name them [9].

Variational autoencoders (VAEs [10]) are natural candidates here because their latent variables are probabilistic degrees of freedom: the ELBO averages reconstruction over the approximate posterior and charges a KL cost for keeping that posterior input-dependent and distinct from the prior. At a chosen information price, coordinates whose reconstruction benefit does not justify this cost collapse toward the prior. This latent sparsity reduces an overcomplete embedding to an active rank, which is read of as the efective dimension at the corresponding reconstruction tolerance.

Collapse can also be caused or amplified by optimization efects or decoder mismatch and is therefore often presented as a failure mode [11, 12]. Here we focus on the equilibrium, �-controlled part of collapse and use it as a selection mechanism. In a �-VAE [13, 14], � is the price, in reconstruction-loss units, of one nat of latent information. After normalization by the data variance in decoder-variance units, � is the same price in normalized reconstruction units.

The linear Gaussian analysis in Ref. [15] shows that this selection picture is exact in the linear VAE case. As $T$ is raised, latent coordinates collapse one by one. Each collapse threshold equals both the marginal reconstruction utility of that coordinate and the corresponding PCA explained-variance ratio. The collapse spectrum, PCA spectrum, and pruning-utility spectrum therefore coin cide component by component.

This paper asks which parts of this exactly calibrated picture survive in nonlinear VAEs. We test this on the WorldClim bioclimatic dataset [16], using rank– distortion curves to ask how many variables are needed to reach a given normalized reconstruction error. Dinnage studies the same dataset with a VAE at a fixed operating point and asks how useful the learned latent representation is for downstream analysis [17]. That is a natural question once one commits to a single bottle neck strength. Our question is diferent: we vary the regularization strength and ask how the active rank and reconstruction-utility spectrum change with the cutof.

To answer this question, we scan the regularization strength $\beta ,$ rank the latent coordinates, and measure how reconstruction utility and active rank evolve with the cutof.

## II. NONLINEAR LATENT SPECTROSCOPY

The nonlinear scan uses the same normalized control parameter as the linear theory of Ref. [15]. The VAE objective is

$$
\begin{array} { r } { \mathcal { L } = D + \beta R , } \end{array}
$$

where $D$ is reconstruction distortion, i.e. the reconstruction negative log-likelihood, and � is the latent KL, or information rate. The raw parameter $\beta$ sets the strength of the KL regularization. An overbar denotes an average over data samples, and brackets denote posterior averages.

$$
D = - \overline { { \langle \log p _ { \theta } ( x \mid z ) \rangle } } , \qquad R = \overline { { \mathrm { K L } \big [ q _ { \phi } ( z \mid x ) \| p ( z ) \big ] } } .
$$

For the fixed-variance Gaussian decoder used here,

$$
D = \frac { 1 } { 2 \sigma _ { \mathrm { d e c } } ^ { 2 } } \overline { { \langle \| x - \hat { x } _ { \theta } ( z ) \| ^ { 2 } \rangle } } + \mathrm { c o n s t } ,
$$

To compare $\beta$ meaningfully with the data, and with the linear baseline of Ref. [15], it must be measured against the total data variance in decoder-variance units, $\bar { V / \sigma _ { \mathrm { d e c } } ^ { 2 } }$ For a fixed-variance Gaussian decoder, we therefore use the normalized control parameter

$$
T = { \frac { \beta } { V / \sigma _ { \mathrm { d e c } } ^ { 2 } } } = { \frac { \beta \sigma _ { \mathrm { d e c } } ^ { 2 } } { V } } ,
$$

where � is the total data variance. Thus $T$ is the information price in normalized distortion units.

The KL regularization acts spectrally: coordinates collapse one by one as this price is raised, rather than being uniformly shrunk. In the linear case this cutof is exactly calibrated by utility; in the nonlinear case this calibration is part of what we test below.

The thermodynamic analogy has a precise but limited scope here. The latent variables are internal probabilistic degrees of freedom averaged over in the ELBO through the expectation over $q _ { \phi } ( z \mid x )$ and through the KL term relative to the prior. By contrast, the encoder and decoder parameters are not averaged over: they are selected by optimization of the objective.

This means that the VAE is not a thermodynamic or field-theoretic system in the usual sense, and thus has no large-� or continuum limit. The useful analogy is in stead one of efective description with a finite number of learned internal variables: some variation is represented explicitly through active latent coordinates, while the rest is absorbed into the residual distortion.

In the linear Gaussian case this structure becomes exactly solvable mode by mode; in the nonlinear case it remains a useful language for thinking about cutof, activity, and ranked utility without becoming a literal statistical-mechanics construction.

For each latent coordinate $k ,$ we monitor the posterior mean-square amplitude, posterior variance, posterior log variance, and rate,

$$
\overline { { \mu _ { k } ( x ) ^ { 2 } } } , \qquad \overline { { \sigma _ { k } ( x ) ^ { 2 } } } , \qquad \overline { { \log \sigma _ { k } ( x ) ^ { 2 } } } , \qquad R _ { k } ( T ) .
$$

The primary order parameter is the scale-invariant signal fraction

$$
M _ { k } ^ { 2 } ( T ) = \frac { \overline { { { \mu _ { k } ( x ) ^ { 2 } } } } } { \overline { { { \mu _ { k } ( x ) ^ { 2 } } } } + \overline { { { \sigma _ { k } ( x ) ^ { 2 } } } } } .
$$

The denominator is the aggregate second moment $A _ { k } ^ { 2 } = \overline { { \langle z _ { k } ^ { 2 } \rangle } } = \overline { { \mu _ { k } ( x ) ^ { 2 } + \sigma _ { k } ( x ) ^ { 2 } } }$ , so $M _ { k } ^ { 2 }$ measures the signal fraction of the aggregate coordinate without assuming a fixed latent scale.

The general idea is to call a coordinate active once an order parameter shows a clear departure from the collapsed prior-like branch. In the scan diagnostics used here, we classify coordinate � as active when $M _ { k } ^ { 2 } > 0 . 1$ This choice is not unique: one could instead threshold the SNR-like score, the rate, or the posterior log-variance. We checked that such choices shift exact thresholds and active counts slightly, but do not change the qualitative ranking or the concentration of utility in the leading coordinates.

Equivalently,

$$
M _ { k } ^ { 2 } ( T ) = \frac { \mathrm { S N R } _ { k } ( T ) } { 1 + \mathrm { S N R } _ { k } ( T ) } , \qquad \mathrm { S N R } _ { k } ( T ) = \frac { \overline { { { \mu _ { k } ( x ) ^ { 2 } } } } } { \overline { { { \sigma _ { k } ( x ) ^ { 2 } } } } } .
$$

This ratio is invariant under a common rescaling of the posterior mean and standard deviation of a latent coordinate, so it provides a scale-free order parameter for collapse.

For squared-error reconstruction, we report normalized distortion in the same convention as Ref. [15],

$$
\tilde { D } _ { K } \equiv \frac { \overline { { \| x - \hat { x } ^ { ( K ) } \| ^ { 2 } } } } { V } ,
$$

where $\hat { x } ^ { ( K ) }$ is the reconstruction obtained after retaining the first � ranked latent coordinates and pruning the rest. For a fixed ranking, these truncated reconstructions define the nonlinear utility spectrum. The marginal utility of rank � is

$$
\Delta \tilde { D } _ { K } = \tilde { D } _ { K - 1 } - \tilde { D } _ { K } .
$$

The ranked sequence $\{ \Delta \tilde { D } _ { K } \}$ is the spectrum compared across architectures. It is empirical: it depends on the trained nonlinear representation, the scan protocol, and the chosen ranking observable.

## III. LINEAR BASELINE AND NONLINEAR SPECTRAL DEFORMATION

The linear Gaussian VAE provides the calibrated baseline. The reconstruction geometry is quadratic and diagonal in the PCA basis, so each eigendirection behaves independently, and activating one mode thus does not change the utility of another. The signal-fraction branch follows the one-mode form

$$
M _ { k } ^ { 2 } ( T ) = \left[ 1 - \frac { T } { T _ { k } } \right] _ { + } .
$$

and the collapse threshold, reconstruction utility, and PCA weight coincide:

$$
T _ { k } = \Delta \tilde { D } _ { k } = \frac { \lambda _ { k } } { V } .
$$

The linear result is a calibrated null model. Assuming a centered dataset, a fixed-variance Gaussian decoder, a standard normal prior, and a linear encoder–decoder pair, the equilibrium $\beta$ scan does not merely recover the PCA subspace. It resolves the PCA spectrum mode by mode: the �-th latent becomes active for $T _ { k } \le \lambda _ { k } / V$ , its signal fraction follows $M _ { k } ^ { 2 } = [ 1 - T / T _ { k } ] _ { + }$ , and retaining that mode lowers the normalized distortion by the same amount, $\Delta \tilde { D } _ { k } = T _ { k }$ . The nonlinear experiments below keep this measurement protocol and ask which parts of this calibration remain once the decoder can mix modes.

We call such departures nonlinear spectral deformations. Several deformations are possible: collapse thresh olds can shift relative to PCA, the coordinate order can change, leading modes can absorb a larger fraction of the total distortion reduction, and onset branches can soften into crossovers when a new coordinate activates in the background created by already-active coordinates.

This last efect requires separating a local instability from a finite reconstruction utility. Let � denote the set of already-active coordinates, and let $q _ { k }$ be a small activity amplitude for a collapsed coordinate �. Locally, the objective can be expanded as

$$
\mathcal { L } _ { A + k } = \mathcal { L } _ { A } + \frac { 1 } { 2 } \left[ T - g _ { k } ( A , T ) \right] q _ { k } ^ { 2 } + O ( q _ { k } ^ { 4 } ) ,
$$

where $g _ { k } ( A , T )$ is the local reconstruction gain available to coordinate � in the background of the active set. The infinitesimal onset is controlled by

$$
T = g _ { k } ( A , T ) .
$$

The finite utility measured by truncation is instead

$$
\Delta \tilde { D } _ { k } ( A ) = \tilde { D } ( A ) - \tilde { D } ( A \cup \{ k \} ) .
$$

In the linear Gaussian case these two notions coincide because the quadratic reconstruction operator is global and diagonal in the PCA basis:

$$
g _ { k } ( A , T ) = \Delta \tilde { D } _ { k } ( A ) = \Delta \tilde { D } _ { k } .
$$

The onset condition is therefore independent of which other modes are active. This does not mean that earlier modes have saturated when later modes appear: even in the linear branch, an already-active coordinate continues to grow as $T$ decreases. The special feature of the linear case is that the local quadratic operator is fixed once and for all, so the growth of one mode does not change the instability scale of another.

A nonlinear representation has no reason to obey this additivity. The relevant quadratic operator is local to the current active background and can change along the scan. Schematically,

$$
g _ { k } ( A , T ) = g _ { k } ^ { ( 0 ) } + \sum _ { \ell \in A } J _ { k \ell } ( T ) + \sum _ { \ell , m \in A } J _ { k \ell m } ( T ) + \cdots ,
$$

and the finite utility need not equal this local gain at every point along the branch. The coeficients in this expansion summarize how the marginal usefulness of one coordinate is dressed by the nonlinear reconstruction background built from the coordinates that are already active. This explains why nonlinear onsets need not follow the linear branch exactly. We now turn to the measured scans.

## IV. RANKED UTILITY SPECTRUM IN WORLDCLIM

Our main empirical object is the ranked utility spectrum for WorldClim. Unless noted otherwise, the singlearchitecture scans below use a representative fully connected VAE with four hidden layers of width 64 in both encoder and decoder, and 32 latent coordinates. The depth comparison keeps the width fixed at 64 and the latent dimension fixed at 32 while varying depth over 2, 4, 8, and 16 hidden layers, so the comparison isolates nonlinear composition rather than a broad architecture sweep.

Fig. 1 shows the signal-fraction scan used to rank coordinates. The first coordinates activate in a clear sequence as $T$ is lowered, but the branches do not appear to follow exact one-mode Landau curves. In the nonlinear model, the onset can be rounded or shifted by the active back ground and by optimization. We therefore use $M _ { k } ^ { 2 }$ primarily as a scale-invariant ranking observable, not as the main source of fitted thresholds.

![](images/9d9b2832dd6acb3c6516af1f242f9916bf8cfa35c3db18649b143d3598ff8f58.jpg)  
Figure 1: Signal-fraction scan for the representative 4-layer VAE. Colors denote rank after sorting by the SNR-like score, equivalently by final $\breve { M } _ { k } ^ { 2 }$

Fig. 2 measures utilities directly. For each point along the scan, we retain only the first � ranked latents and prune the rest. The left panel follows the resulting normalized distortion as a function of $T ;$ the right panel records the best distortion reached at each retained rank. The marginal drops in this curve define $\Delta \tilde { D } _ { K }$ . This step is independent of any threshold fit.

This pruning measurement is a property of the trained representation, not a comparison with separately retrained �-latent models. The distinction is intentional: during training, reconstruction utility can be distributed across more coordinates than are ultimately needed at a chosen tolerance. The reported curve asks how much distortion remains after truncating the trained representation to its first � ranked variables.

![](images/37261c0c266adfc6f8a2f8d102abf8b66f633e61f01820eab0c39a90ef3a8c3c.jpg)  
Figure 2: Truncated distortion under ranked pruning. Left: distortion after retaining the first � ranked latents at each temperature. Right: best distortion at each rank; successive drops define $\Delta \tilde { D } _ { K }$

Because the nonlinear onset shape is not known a priori, we do not treat the rounded $\bar { M _ { k } ^ { 2 } }$ branches as exact fit objects. Instead, we use $M _ { k } ^ { 2 }$ to rank coordinates, measure utilities directly by pruning, and test the linear activebranch form on the rate, where the scaling is empirically cleanest. In the linear theory, the same number is both a truncation utility and a collapse threshold. We therefore ask whether the rate curves overlay when scaled by the measured utility:

$$
R _ { k } ( T ) \simeq \left[ - { \frac { 1 } { 2 } } \log { \frac { T } { \Delta \tilde { D } _ { k } } } \right] _ { + } .
$$

Fig. 3 shows this rescaled overlay for the information rate ${ \cal R } _ { k } ( T )$ using the utilities from Fig. 2. The black curve is the one-mode Landau prediction with no fitted threshold. The colored points are still ranked by $M _ { k } ^ { 2 }$ so the overlay also checks agreement between the orderparameter ranking and the rate ranking. In diagnostic checks, alternative rankings by information rate or posterior log-variance mainly reorder weak or near-degenerate coordinates and do not change the leading shortlist.

There is no parameter fitting involved in Fig. 3: the point is that a utility measured from reconstruction predicts the rate scale of the active branch. At much lower relative temperature, roughly two decades below threshold, some higher-rank branches steepen; this deep active tail is not used for later fits of thresholds.

The same rescaled overlay identifies a natural active window for a secondary threshold check. For each rank, we fit the unrescaled rate branch to the fixed-slope form

$$
R _ { k } ( T ) \simeq \left[ - \frac { 1 } { 2 } \log \frac { T } { T _ { c , k } ^ { \mathrm { r a t e } } } \right] _ { + }
$$

over

$$
1 0 ^ { - 2 } \Delta \tilde { D } _ { k } \le T \le \Delta \tilde { D } _ { k } .
$$

![](images/701c07b28189671fe00affef877715e002c27575a95d9961d756239e5a603c9d.jpg)  
Figure 3: Rate curves rescaled by pruning utilities $\Delta \tilde { D } _ { k }$ The black curve is the one-mode prediction; colored points are nonlinear branches ranked by $M _ { k } ^ { 2 }$

The window is defined from the pruning utility, but the intercept $T _ { c , k } ^ { \mathrm { r a t e } }$ is extracted from the rate branch. Fig. 4 compares these fitted thresholds with the directly measured utilities. The diagonal agreement is summarized by an identity $R ^ { 2 }$ on log-scaled values, log $\Delta \tilde { D } _ { k }$ versus log $T _ { c , k } ^ { \mathrm { r a t e } }$ . This is a predictive identity score on the log values, not the square of a correlation coeficient, so over all ofsets are penalized.

On the test split, the check gives $R ^ { 2 } \simeq 0 . 9 8$ over the plotted ranks. For the representative WorldClim run, the threshold ranking agrees with the utility ranking through the first thirteen modes; beyond that point, neighboring fitted thresholds begin to swap while the pruning utilities remain clearly separated. This is not the primary way to define the spectrum, but a consistency check on the utility calibration.

A compact sparsity diagnostic is the required rank at a prescribed normalized distortion. Fig. 5 shows the measured rank–distortion curves. Lower curves are more spectrally sparse: they recover the same distortion re duction with fewer retained variables. This comparison is clearest on a linear rank axis and logarithmic distortion axis, because it exposes both the low-rank head and the residual tail. The plotted comparison fixes the hid den width and varies depth. The first few ranks form a depth-sensitive head: increasing depth can concentrate more utility into the leading efective variables. Beyond rank 8, however, the deepest models often saturate at a higher residual floor.

Reading the same curves at fixed tolerance gives the efective dimension,

$$
d _ { \mathrm { e f f } } ( \epsilon ) = \operatorname* { m i n } \{ K : D ( K ) / D _ { 0 } \leq \epsilon \} .
$$

Table I reports this fixed-tolerance read-of from 5%

![](images/9937024eb3f2b93f016bb853892e5a58d8c828be564c37ad547aef8baad6897f.jpg)  
Figure 4: Utility–threshold comparison from fixed-slope rate fits. Each point compares $\Delta \tilde { D } _ { k }$ with the corresponding rate-fit threshold $T _ { c , k } ^ { \mathrm { r a t e } }$

![](images/0b25bcf9aa93e25a8b81ba951fdc2a34ff31be8fc100e7b7e2632aca0e24c5fa.jpg)  
Figure 5: Rank–distortion curves under ranked pruning. Lower curves indicate a more concentrated utility spectrum.

down to $1 0 ^ { - 5 }$

Fig. 6 suggests that the same fall-of can be described locally by a power law. On log–log axes, the head–tail separation appears as a crossover near $K _ { c } \simeq 4 :$ the first few ranks form the head, and subsequent ranks follow the residual tail until architecture-dependent floors appear. We interpret $K _ { c }$ only as a spectral crossover rank, not as a unique intrinsic dimension of the dataset. Fig. 6 shows the representative four-layer width-64, 32-latent VAE on log–log axes, with a fit over the low-rank window used in the architecture comparison. For WorldClim, the residual distortion falls approximately as $D ( K ) \sim K ^ { - 4 }$ over ranks 4–17. The corresponding spectral exponent is

Table I: Required truncation rank to reach each normalized distortion target. For each target, the VAE entry is the lowest rank among the compared VAE architectures. A dash indicates that the target is not reached.
<table><tr><td colspan="4">Distortion Model 5% 1% 0.5% 0.1% 0.01% 0.001%</td></tr><tr><td></td><td>9</td><td>10 12</td><td>15 17</td></tr><tr><td>PCA 6 VAE 2</td><td>5 5</td><td>8 14</td><td></td></tr></table>

closer to 5 since the marginal utility spectrum $\Delta \tilde { D } _ { K }$ is the discrete derivative of �(�) and is therefore roughly one power steeper.

![](images/99d11cce6183ac290b639ea168a042901b8b21938c964a1a3e83d992332c6350.jpg)  
Figure 6: Log–log rank–distortion curve for the representative four-layer width-64, 32-latent VAE. The dashed line is the power-law fit over the chosen rank window.

For this protocol, the full threshold spectrum is not required. The direct object is the truncation curve: train across a grid of normalized spectral cutofs, rank the latent variables by a stable order parameter, and record the normalized distortion reached after retaining the first � variables. At fixed width, increasing depth mostly sharpens the low-rank head of this curve: more utility is concentrated into the first few retained variables. The same increase in depth is not uniformly beneficial in the tail, since the deepest models can plateau at higher residual distortion. We therefore describe depth as a head–tail tradeof at fixed width, not as a monotonic improvement.

## V. CONCLUSIONS AND OUTLOOK

For WorldClim, the nonlinear scan confirms the cutof picture, but it also shows that direct reconstruction utilities are more robust than fitted collapse thresholds. Utilities are therefore the preferred spectrum: they directly measure the feature importance of each ranked latent coordinate, remain usable at higher ranks, and avoid the fit-window and outlier sensitivity of threshold extraction. Thresholds serve mainly as a check of the utility– threshold duality.

This changes the practical role of the scan. The scan is the calibration step, not the end goal. Once the spectralcutof picture has been checked for a given model class and normalization, one need not extract thresholds or run a full scan on every dataset. Instead, one can choose the normalized regularization strength to match the smallest reconstruction utility one is willing to trust, for example a noise floor or modeling tolerance.

Under this interpretation, the main output is not a single preferred bottleneck size, but a tolerance-dependent efective dimension. Efective dimension is therefore not a fixed property of the data alone; it depends on the task, distortion metric, model class, and tolerated reconstruction error.

Architecture changes the shape of that dependence: at fixed width, depth mainly redistributes utility toward the head of the spectrum, often at the cost of a higher tail floor. This is a latent sparsity tradeof: deeper models concentrate more reconstruction value into the first few coordinates, but can leave a less eficient residual tail.

These results support an efective-theory reading of the �-VAE scan. The value of � specifies the price of resolving latent information; the trained model then decides which learned variables remain explicit and which variation is absorbed into the residual distribution. In that sense, the scan does not assume an efective description in advance; it reveals one by showing which variables survive at the chosen cutof.

This selection mechanism also clarifies what distinguishes VAEs from deterministic and sparse autoencoders. VAEs introduce probabilistic latent degrees of freedom, average over them in the objective, and penalize their departure from a prior. This is structurally close to a finite-dimensional variational free-energy problem over latent variables, in a way that deterministic embeddings are not. A deterministic autoencoder can learn a lowdimensional bottleneck, but the bottleneck dimension is fixed by design: the model tests whether that chosen dimension is suficient, not which coordinates should survive as the tolerance changes. Sparse autoencoders define a diferent notion of sparsity that can be useful in other contexts [18, 19]: they impose sparsity at the level of individual codes, with each input activating only a small subset of dictionary elements, and the active sub set changing from one data point to the next. This is well suited to feature discovery, especially in overcomplete representations, but it does not by itself produce a small globally ranked set of coordinates shared across the dataset. Other regularizers can also suppress dimensions or produce sparse codes, but they generally do not retain the same prior-relative, information-theoretic interpretation of the thresholds.

In the VAE, this cutof picture comes from a priorrelative rate cost for making an entire coordinate inputdependent across the dataset. A coordinate either earns its KL cost or collapses toward the prior globally. Rather than asking whether a system is emergent in the abstract, one can ask how many efective variables are required to reproduce the behavior of interest at a given tolerance.

Because the present work ranks variables by reconstruction utility, it can retain redundant or nuisance variation whenever that variation helps reproduce the input distribution. A natural next step is therefore to embed the same variational bottleneck inside a supervised predictor and rank variables by task utility. Since the bottleneck would then lie on the prediction path, pruning would measure the importance of variables for the model actually used for inference, rather than for a post-hoc surrogate.

In that setting, choosing $\beta$ becomes an operating-point question: one would trade of task performance, active dimension, and the interpretability of the surviving coordinates.

## VI. APPENDIX: WORLDCLIM EXPERIMENT DETAILS

The WorldClim experiments use the 19 bioclimatic variables from WorldClim [16] at 10 arc-minute resolution. We keep grid cells that are valid in all 19 layers and standardize each feature using the training split only. The data split uses seed 0 and the spatial-block protocol used throughout these experiments: equal-area blocks of side length 500 km, area-weighted sampling, a target of 500,000 training samples, and an equal split of the non training blocks between validation and test. The batch size is 256.

[1] J. B. Tenenbaum, V. de Silva, and J. C. Langford, A global geometric framework for nonlinear dimensionality reduction, Science 290, 2319 (2000).

[2] S. T. Roweis and L. K. Saul, Nonlinear dimensionality reduction by locally linear embedding, Science 290, 2323 (2000).

[3] L. Cayton, Algorithms for Manifold Learning, Tech. Rep. CS2005-0839 (University of California, San Diego, 2005).

[4] Y. Bengio, A. Courville, and P. Vincent, Representation learning: A review and new perspectives, IEEE Transactions on Pattern Analysis and Machine Intelligence 35, 1798 (2013).

[5] K. G. Wilson and J. Kogut, The renormalization group and the � expansion, Physics Reports 12, 75 (1974).

[6] N. Goldenfeld, Lectures on Phase Transitions and the Renormalization Group, 1st ed. (CRC Press, 2018).

[7] P. Mehta and D. J. Schwab, An exact mapping between the variational renormalization group and deep learning, arXiv preprint (2014), arXiv:1410.3831.

[8] H. W. Lin, M. Tegmark, and D. Rolnick, Why does deep and cheap learning work so well?, Journal of Statistical Physics 168, 1223 (2017).

The representative nonlinear run used in Figs. 1, 2, $^ { 3 , }$ and 4 is the saved VAE run where the encoder and decoder are fully connected networks with four hidden layers, 64 hidden units per layer, and 32 latent coordinates. We use SiLU activations, Kaiming initialization, and initial log-variance bias −4. The reconstruction loss is mean-squared error with fixed decoder variance convention $\sigma _ { \mathrm { d e c } } ^ { 2 } = 1$ and no additional reconstruction normalization.

The scan evaluates an independent grid of normalized temperatures, with $T$ running from 1 to $1 0 ^ { - 6 }$ Each value of $T$ is trained from a fresh initialization rather than warm-started from neighboring scan points, so the ordering of the grid is not an annealing path and does not introduce scan hysteresis. Here “equilibrium” means that each independently trained scan point is run until the configured convergence criterion is met; it does not imply a proof of global optimality for the nonlinear objective. In the saved resolved configuration this corresponds to 121 scan points and raw $\bar { \beta }$ values from 19.0 to $1 . 9 \times 1 0 ^ { - 5 }$ , because the measured WorldClim variance scale is $V = 1 9$ . Distortions are reported in normalized squared-error units, i.e. reconstruction error divided by the total data variance $V .$ Each scan point is trained with Adam at learning rate $3 \times 1 0 ^ { - 4 }$ , with a maximum budget of $1 0 ^ { 6 }$ optimizer updates, relative tolerance $1 0 ^ { - 3 } ,$ and patience fraction 0.01. The reported quantities are evaluated on the test split. PCA references use the same normalized data.

Latent coordinates are ranked by the saved SNR-like score, equivalently by the signal fraction $M _ { k } ^ { 2 }$ using descending score order. The active threshold used for scan diagnostics is 0.1. Ranked-pruning curves retain the first $K$ coordinates in this order and prune the rest.

[9] F. Locatello, S. Bauer, M. Lucic, G. Raetsch, S. Gelly, B. Schölkopf, and O. Bachem, Challenging common assumptions in the unsupervised learning of disentangled representations, in Proceedings of the 36th International Conference on Machine Learning (ICML), Proceedings of Machine Learning Research, Vol. 97 (PMLR, 2019) pp. 4114–4124.

[10] D. P. Kingma and M. Welling, Auto-encoding variational bayes, in 2nd International Conference on Learning Representations (ICLR) (2014).

[11] B. Dai, Z. Wang, and D. Wipf, The usual suspects? reassessing blame for VAE posterior collapse, in Proceedings of the 37th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 119 (PMLR, 2020) pp. 2313–2322.

[12] J. Lucas, G. Tucker, R. B. Grosse, and M. Norouzi, Don’t blame the ELBO! a linear VAE perspective on posterior collapse, in Advances in Neural Information Processing Systems 32 (2019).

[13] I. Higgins, L. Matthey, A. Pal, C. P. Burgess, X. Glorot, M. M. Botvinick, S. Mohamed, and A. Lerchner, betavae: Learning basic visual concepts with a constrained

variational framework, in 5th International Conference on Learning Representations (ICLR) (OpenReview.net, 2017).

[14] C. P. Burgess, I. Higgins, A. Pal, L. Matthey, N. Watters, G. Desjardins, and A. Lerchner, Understanding disentangling in ����-vae, arXiv preprint (2018), arXiv:1804.03599.

[15] J. Hirn, Latent spectroscopy: Posterior collapse as a fea ture, arXiv preprint (2026), arXiv:2605.22691v2.

[16] S. E. Fick and R. J. Hijmans, Worldclim 2: new 1-km spatial resolution climate surfaces for global land areas, International Journal of Climatology 37, 4302 (2017).

[17] R. Dinnage, How many variables does wordclim have, really? generative A.I. unravels the intrinsic dimension of bioclimatic variables, bioRxiv preprint (2023), bioRxiv 2023.06.12.544623.

[18] A. Bricken, C. Templeton, J. Templeton, T. Conerly, N. McDougall, T. Henighan, A. Hume, B. Carter, and C. Olah, Towards monosemanticity: Analyzing 1m+ features in sparse autoencoders, Transformer Circuits Thread (2023).

[19] H. Cunningham, A. Eisma, L. Chan, and L. Turner, Sparse autoencoders find highly interpretable directions in language model activations, arXiv preprint (2023), arXiv:2309.09942.