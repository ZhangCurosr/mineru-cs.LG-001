# Invertible Logits Transformation for Accuracy-Preserving Post-Hoc Uncertainty Calibration

Lening Zhao<sup>∗</sup>, Qipeng Zhan<sup>∗</sup> <sup>†</sup>, Li Shen<sup>†</sup>

University of Pennsylvania

{lnzhao@engineering, qipengz@sas, li.shen@pennmedicine}.upenn.edu

## Abstract

Post-hoc calibration aligns a classifier’s predicted confidences with its empirical accuracy without retraining. An ideal calibrator should correct nonlinear miscalibration, scale gracefully to large label spaces, and preserve the original predictions; existing methods typically violate at least one of these properties— temperature scaling lacks expressivity, more flexible parametric alternatives introduce parameters that grow with the number of classes C, and other expressive methods do not preserve the rank ordering of class scores and may alter the predicted class. We propose Invertible Logits Transformation (InvLT), which applies a learned scalar MLP f : R → R element-wise to the pre-softmax logits. Sharing f across all logit dimensions makes the parameter count independent of C. Monotonicity of f—and hence preservation of the argmax prediction—is softly encouraged via a paired inverse network rather than enforced through the numerical integration required by prior monotone calibrators; this avoids their computational overhead while empirically preserving the original classification accuracy in every setting we evaluate. Across standard image classification benchmarks and a range of architectures, InvLT consistently outperforms a broad set of post-hoc baselines on standard calibration metrics.

## 1 Introduction

Calibration is a fundamental requirement for trustworthy prediction: a classifier is calibrated if its predicted confidence matches its empirical accuracy [Naeini et al., 2015, Guo et al., 2017]. Calibration matters wherever downstream decisions depend on probabilistic predictions—medical diagnosis, autonomous driving, decision support, and selective prediction—yet modern deep networks are typically over-confident, even assigning high probability to incorrect predictions [Guo et al., 2017, Minderer et al., 2021]. Post-hoc calibration addresses this by learning a transformation of a fixed model’s outputs on a held-out calibration set, leaving the trained weights untouched.

The most widely adopted post-hoc method is temperature scaling (TS) [Guo et al., 2017], which divides all logits by a single learned scalar T > 0. Because scalar division preserves the ordering of logit values, TS guarantees that the predicted class is unchanged—a property we call accuracy preservation. However, TS has only one degree of freedom: it applies the same linear transformation $f ( x ) = x / T$ to every logit of every sample, and therefore cannot correct miscalibration that varies across confidence levels.

Two complementary axes of expressivity. TS admits two natural accuracy-preserving extensions, distinguished by what the transformation depends on:

• Sample-specific. Make the temperature depend on the input. Parameterized Temperature Scaling (PTS) [Tomani et al., 2022] predicts a per-sample $\mathbf { \bar { \Gamma } } _ { T ( \mathbf { z } ) }$ via a network and applies

$$
\phi _ { \mathrm { P T S } } ( \mathbf { z } ) _ { c } = z _ { c } / T ( \mathbf { z } ) .
$$

All logits of a given sample are still divided by the same positive scalar, so the within sample ordering—and the argmax—is preserved.

• Logit-specific. Make the transformation depend on the value of each logit. Replace the per-sample temperature with a shared scalar function f applied logit-wise:

$$
\phi _ { \mathrm { I n v L T } } ( \mathbf { z } ) _ { c } = f ( z _ { c } ) .
$$

If $f$ is strictly increasing, the ordering of logit values is preserved for every input, and so is the argmax.

These directions are complementary: PTS reshapes calibration across samples while keeping the per-sample logit map linear; the logit-specific approach reshapes the logit scale nonlinearly while sharing the same map across samples and classes.

The logit-specific direction has been explored through intra-order-preserving calibration functions [Rahimi et al., 2020], whose diagonal sub-family learns a strictly increasing scalar function using the UMNN architecture [Wehenkel and Louppe, 2019]. UMNN parameterizes $f ^ { \prime }$ as a positive network and recovers $f$ by numerical quadrature at every forward pass—a procedure repeated during both training and inference, and one that constrains the architecture and tooling around f (e.g., choice of activations, integration solvers, gradient computation). Our central methodological contribution is to replace this hard architectural constraint with a soft regularizer rooted in a classical result: $a$ continuous injective function on an interval is strictly monotone. We therefore train $f$ jointly with an auxiliary inverse network $^ { g , }$ minimizing the reconstruction error $\| g ( f ( u _ { k } ) ) - u _ { k } \| ^ { 2 }$ over reference points $\mathbf { \bar { \{ } }  u _ { k } \} _ { k = 1 } ^ { K }$ drawn from the operating range of the calibration logits. This approach—well-established in image translation but, to our knowledge, novel in the post-hoc calibration literature—drives $f$ toward injectivity, and hence toward monotonicity, without imposing any architectural restriction. This reframing has three practical advantages. First, training avoids numerical integration: each step adds only $^ { - } K$ scalar evaluations of $g \circ f$ on a precomputed grid. Second, inference reduces to a small MLP applied element-wise, comparable in cost to TS. Third, $f$ can be any standard MLP with any activation function, simplifying implementation and tuning. In general, our contributions are as follows:

1. We introduce InvLT, a post-hoc calibrator that applies a shared scalar MLP element-wise to logits. The number of learnable parameters is independent of the number of classes $C .$

2. We propose a reconstruction regularizer via an inverse network that softly enforces monotonicity, avoiding the numerical integration required by prior monotone calibration methods [Wehenkel and Louppe, 2019, Rahimi et al., 2020].

3. Across CIFAR-10, CIFAR-100, and ImageNet with seven architectures, InvLT consistently outperforms all baselines while being substantially faster to train than UMNN.

## 2 Related Work

Post-hoc calibration methods can be broadly organized along two axes: the space in which the transformation operates (logit space vs. probability space) and the expressiveness of the parameterization (single-parameter vs. multi-parameter or nonparametric).

Temperature and affine scaling. Temperature scaling [Guo et al., 2017] is the simplest and most widely used post-hoc method. It learns a single scalar $\bar { T } > 0$ and replaces the logits z with ${ \mathbf z } / T$ Despite having only one parameter, it is effective when the miscalibration is approximately uniform across confidence levels. Vector scaling extends this by learning a per-class scaling factor and bias, while matrix scaling further introduces cross-class interactions via a full weight matrix [Guo et al., 2017]. However, matrix scaling requires $O ( C ^ { 2 } )$ parameters and is prone to overfitting when the validation set is small relative to the number of classes.

Ensemble and parameterized temperature scaling. Ensemble Temperature Scaling (ETS) [Zhang et al., 2020] learns a convex combination of temperature scaling, no calibration, and a uniform baseline, providing modest improvements with minimal additional complexity. Parameterized Temperature Scaling (PTS) [Tomani et al., 2022] conditions the temperature on the input logits via a small neural network, allowing instance-dependent recalibration while preserving the argmax prediction. PTS is conceptually related to InvLT in that both use neural networks for calibration, but PTS learns a per-instance temperature whereas InvLT learns a shared nonlinear transformation of the logit values themselves.

Uni-variate calibration methods. Histogram binning [Zadrozny and Elkan, 2001] partitions the confidence range into bins and assigns each bin the empirical accuracy of the samples falling within it. Isotonic regression [Zadrozny and Elkan, 2002] fits a monotone piecewise-constant function to map predicted confidences to calibrated probabilities. Platt scaling [Platt et al., 1999] fits a logistic regression on the classifier’s outputs. These methods operate in probability space and are originally designed for binary calibration; extending them to the multiclass setting is typically done with a onevs-rest scheme, after which a renormalization step is required to combine the per-class calibrated probabilities, which can re-shift the already-calibrated outputs.

Spline and Dirichlet calibration. Spline calibration [Gupta et al., 2020] models the calibration map using natural cubic splines in probability space, offering smooth and flexible recalibration. Dirichlet calibration [Kull et al., 2019] transforms the log-probabilities via a linear map followed by a softmax, parameterized as a Dirichlet distribution over the probability simplex. This approach captures cross-class interactions but introduces $O ( C ^ { 2 } )$ parameters and can overfit or diverge on datasets with many classes and limited validation data, as evidenced by its poor performance on ImageNet in our experiments.

Order-preserving and monotone neural calibration. Rahimi et al. [Rahimi et al., 2020] introduced intra order-preserving functions for post-hoc calibration, a family of transformations that preserve the top-k predictions of a classifier. Their diagonal sub-family is particularly related to InvLT: it learns an increasing scalar function shared across logit dimensions, using the UMNN architecture of Wehenkel and Louppe [Wehenkel and Louppe, 2019] to enforce monotonicity. UMNN represents a monotone function by parameterizing its derivative as a positive neural network and computing the function value through numerical integration. While this gives strict monotonicity, the repeated numerical integration incurs substantial training and inference overhead. InvLT targets the same diagonal, order-preserving setting, but replaces hard architectural monotonicity with a soft inverse-reconstruction regularizer, allowing the calibrating function to be a standard scalar MLP.

## 3 Method

We present Invertible Logits Transformation (InvLT), a post-hoc calibrator that learns a nonlinear transformation of the logit scale while preserving the classifier’s predictions. The central idea is to recalibrate confidence by reshaping logits through a strictly increasing scalar function, obtained by training f to be invertible rather than constraining it to be monotone by construction.

## 3.1 Post-hoc Calibration and Accuracy Preservation

Let $\mathbf { z } \in \mathbb { R } ^ { C }$ denote the pre-softmax logits produced by a fixed classifier for an input x, where C is the number of classes. The corresponding predictive distribution is $\hat { \mathbf { p } } = \operatorname { s o f t m a x } ( \mathbf { z } )$ . Post-hoc calibration learns a transformation $\phi : \mathbb { R } ^ { C } \overset { \smile } { \to } \bar { \mathbb { R } } ^ { C }$ on a held-out calibration set $\mathcal { D } _ { \mathrm { c a l } } = \mathrm { \widetilde { \{ } }  ( { \mathbf { z } } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ and returns calibrated probabilities $\hat { \mathbf { p } } ^ { \mathrm { c a l } } = \mathrm { s o f t m a x } ( \phi ( \mathbf { z } ) )$ ). In many settings, the base classifier has already been optimized for accuracy, and calibration is intended only to correct its confidence estimates. We therefore require the calibration map to preserve the predicted class. We call a transformation ϕ accuracy-preserving if

$$
\arg \operatorname* { m a x } _ { c } \phi ( \mathbf { z } ) _ { c } = \arg \operatorname* { m a x } _ { c } z _ { c } \qquad { \mathrm { f o r ~ a l l ~ } } \mathbf { z } .
$$

This ensures that calibration changes only the probabilities assigned to classes, not the classifier’s decision.

Logit-specific transformation restricts the calibration map to an element-wise transformation:

$$
\boldsymbol { \phi } _ { f } ( \mathbf { z } ) = \big ( f ( z _ { 1 } ) , f ( z _ { 2 } ) , \ldots , f ( z _ { C } ) \big ) ,\tag{1}
$$

where $f : \mathbb { R } $ R is invertible and shared across all classes and all samples. The calibrated probabilities are then

$$
\hat { \mathbf { p } } ^ { \mathrm { c a l } } = \mathrm { s o f t m a x } ( \phi _ { f } ( \mathbf { z } ) ) .
$$

The key property of this family is that strict increase of $f$ is sufficient to preserve accuracy. The ordering of all logits is unchanged after applying $f$ element-wise. Therefore the index of the largest logit is unchanged, and the predicted class is preserved.

## 3.2 Monotonicity via Invertibility

A standard neural network is flexible but does not by itself guarantee monotonicity. Prior work addresses this by enforcing strict monotonicity through architectural constraints: UMNN [Wehenkel and Louppe, 2019] and the Intra Order-Preserving method [Rahimi et al., 2020] parameterize the derivative $\overline { { f ^ { \prime } } }$ as a positive network and recover $f$ via numerical integration. This guarantees monotonicity by construction but incurs substantial computational overhead because numerical quadrature must be performed at every forward pass during both training and inference.

Rather than constraining the architecture, we ensure monotonicity through a reconstruction regularizer. The key theoretical insight is that a continuous scalar function that is invertible must be strictly monotone:

Proposition 1 (Continuous injective functions are strictly monotone). Let $I \subseteq \mathbb { R }$ be an interval and let ${ \bar { f } } : I $ R be continuous. Iff is injective, then f is strictly monotone on $I .$

The proof, via the Intermediate Value Theorem, is given in Appendix A.

To establish that $f$ is invertible, it suffice to exhibit a map g satisfying $g \circ f = \operatorname { i d }$ . This observation motivates a simple regularizer: we introduce an auxiliary inverse network $g _ { \psi } : \mathbb { R } $ R trained jointly with $f _ { \theta } ,$ and define a set of reference points $\{ u _ { k } \} _ { k = 1 } ^ { \bar { K } }$ covering the relevant logit range observed on the calibration set. The reconstruction loss encourages $g _ { \psi } \circ f _ { \theta }$ to approximate the identity:

$$
\mathcal { L } _ { \mathrm { r e c } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \bigl ( g _ { \psi } ( f _ { \theta } ( u _ { k } ) ) - u _ { k } \bigr ) ^ { 2 } .\tag{2}
$$

Minimizing this loss drives $f _ { \theta }$ toward injectivity and hence strict monotonicity, without constraining the network architecture. Although this enforcement is soft rather than exact, we find empirically that accuracy is preserved in every experiment we report (Section 4.4). This approach [Zhu et al., 2017] requires only $K$ scalar evaluations per training step, adding negligible cost, while avoiding the numerical integration needed by UMNN. The auxiliary network $g _ { \psi }$ is the same architecture as $f _ { \theta }$ used only during training and is discarded at inference.

Orientation. Proposition 1 guarantees strict monotonicity but not its direction; in principle $f _ { \theta }$ could converge to a strictly decreasing function, which would reverse the argmax. In practice, no explicit orientation penalty is needed: a decreasing $f _ { \theta }$ inverts class rankings and produces high training loss, so the calibration objective itself selects the increasing branch. We confirm this empirically: in every setting we tested, InvLT preserves both the increasing orientation and the original classification accuracy (Section 4.4).

## 3.3 Training and Inference

A post-hoc calibrator is usually fitted by optimizing a Negative Log-Likelihood (NLL) Loss

$$
\mathcal { L } _ { \mathrm { N L L } } = - \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \log \mathrm { s o f t m a x } ( \phi _ { f _ { \theta } } ( \mathbf { z } _ { n } ) ) _ { y _ { n } } ,\tag{3}
$$

or a Brier loss

$$
\mathcal { L } _ { \mathrm { B r i e r } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \sum _ { c = 1 } ^ { C } \left( \delta _ { n c } - \mathrm { s o f t m a x } ( \phi _ { f _ { \theta } } ( \mathbf { z } _ { n } ) ) _ { c } \right) ^ { 2 } ,\tag{4}
$$

where $\delta _ { n c } = \mathbf { 1 } [ y _ { n } = c ]$

The complete objective for our InvLT is then:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { B r i e r / N L L } } + \lambda \cdot \mathcal { L } _ { \mathrm { r e c } } , } \end{array}\tag{5}
$$

where $\lambda > 0$ controls the regularization strength. In practice we activate the regularization term after certain epochs. This warm-up technique allows the calibrator to first reduce calibration error before monotonicity is enforced. Both $f _ { \theta }$ and $g _ { \psi }$ are optimized jointly with the Adam optimizer [Kingma and Ba, 2014]. Full hyperparameters are in Appendix B.

At inference, $g _ { \psi }$ is discarded. The calibrated probabilities are simply softmax( $f _ { \theta } ( z _ { 1 } ) , \dots , f _ { \theta } ( z _ { C } ) )$ When $f _ { \theta }$ is strictly increasing, the predicted class is unchanged. The only inference cost is evaluating a small scalar network on each logit.

Remark. Proposition 1 establish that exact invertibility implies strict monotonicity. The reconstruction loss in Eq. (2), however, only enforces $g _ { \psi } \circ f _ { \theta }$ ≈ id at K finitely many reference points with a finite weight λ, and therefore does not provide a formal guarantee that $f _ { \theta }$ is globally monotone or that the argmax is preserved on every input. We view the regularizer as a principled heuristic motivated by Proposition 1, and verify it empirically: the learned transformations visualized in Figure 1 are strictly increasing across all datasets and calibration set sizes, and the ablation in Section 4.4 confirms that the regularizer preserves the base classifier’s accuracy.

## 4 Experiments

## 4.1 Setup

Datasets and models. We evaluate post-hoc calibration on three benchmark settings spanning different scales and class complexities. CIFAR-10 and CIFAR-100 [Krizhevsky et al., 2009] use ResNet-50 [He et al., 2016] as the primary backbone, evaluated across five random seeds. For architecture generalization, we additionally evaluate CIFAR-100 on VGG-16/19 [Simonyan and Zisserman, 2014], DenseNet-121 [Huang et al., 2017], Wide ResNet [Zagoruyko and Komodakis, 2016], and ViT-B/16 [Dosovitskiy et al., 2021]. ImageNet [Deng et al., 2009] uses pretrained ResNet-50 and ResNet-152 checkpoints. All logits are pre-extracted and fixed; calibration methods operate only on held-out validation logits. We use N = 500, N = 5000, N = 25000 held-out samples for calibrator fitting (denoted val 500, val 5 000, val 25 000) as our primary setting.

Baselines. We compare against No Calibration, Temperature Scaling (TS) [Guo et al., 2017], Vector/Matrix Scaling [Guo et al., 2017], Ensemble Temperature Scaling (ETS) [Zhang et al., 2020], Parameterized Temperature Scaling (PTS) [Tomani et al., 2022], Histogram Binning [Zadrozny and Elkan, 2001], Isotonic Regression [Zadrozny and Elkan, 2002], Spline Calibration [Gupta et al., 2020], Dirichlet Calibration [Kull et al., 2019], and UMNN Calibration [Wehenkel and Louppe, 2019, Rahimi et al., 2020]. We separate methods into those that may alter the predicted class and those that preserve the original classification decision by construction.

Metrics. We report Expected Calibration Error (ECE, 15 equal-width bins, %↓) [Naeini et al., 2015, Guo et al., 2017], Adaptive ECE (AdaECE, %↓) [Nixon et al., 2019], KDE-ECE (%↓) [Zhang et al., 2020], and Negative Log-Likelihood (NLL, ↓). Classification accuracy is unchanged for accuracy-preserving methods.

Hyperparameters. Activation function and hidden dimensions are selected per (dataset, calibration - set size) by grid search on the calibration set, using 5-seed-mean ECE as the selection criterion. To ensure a fair comparison, all tunable baselines (including PTS, UMNN, Histogram Binning, Spline, Matrix Scaling, and Dirichlet Calibration) are grid-searched under the same protocol. The selected configurations, the search grid, and the corresponding settings for all tunable methods are given in Appendix B.

## 4.2 Main Results

Table 1 reports all four calibration metrics at val 5 000. InvLT achieves the best ECE, AdaECE, KDE-ECE, and NLL across all three datasets.

On CIFAR-10, InvLT obtains $\mathrm { E C E } = 0 . 6 0 \%$ and ${ \mathrm { N L L } } = 0 . 1 5 .$ , both best among all methods. On KDE-ECE, InvLT achieves 0.98%, outperforming Histogram Binning at 1.09%. On CIFAR-100, InvLT achieves $\mathrm { E C E } = 1 . 6 7 \%$ , outperforming the best accuracy-preserving baseline UMNN at 2.48%, and matches the lowest NLL (0.79). On ImageNet, the margin is largest: InvLT achieves ECE $= 0 . 3 9 \%$ , ahead of the next-best UMNN (0.56%) and far below the best non-preserving baseline, Spline Calibration (1.10%), while also obtaining the best NLL.

Table 2 expands the ImageNet ResNet-152 evaluation across three calibration set sizes. InvLT consistently achieves the best ECE and NLL at all three sizes, with its largest lead over UMNN, the strongest baseline, in the low-data val 500 regime. Notably, several class-dependent methods degrade severely at val 500: Vector Scaling reaches $\mathrm { E C E } = 4 3 . 4 4 \%$ and Isotonic Regression 33.46%, while Matrix Scaling collapses to the identity map (no calibration effect beyond No Cal.); these methods introduce $\breve { O ( C ^ { 2 } ) }$ or $O ( C )$ parameters that cannot be reliably estimated from 500 samples when $C = 1 { , } 0 0 0$ . Standard deviations over seeds are reported in Appendix C.

Table 1: ECE, AdaECE, KDE-ECE (%↓) and NLL (↓) on CIFAR-10/100 (ResNet-50, mean over 5 seeds) and ImageNet (ResNet-152, mean over 5 seeds), all at val 5 000. Methods above the double rule may change the predicted class; below are accuracy-preserving. Bold: best. Underline: second best. <sup>†</sup>KDE-ECE requires re-fitting the grid-search winner at its exact hyperparameters (not computed by the grid search itself); for Dirichlet on ImageNet (C=1,000, ∼ 1M parameters) this re-fit is seed/hardware-unstable (verified against the raw per-seed logs) and is therefore omitted rather than guessed. <sup>‡</sup>Matrix Scaling’s grid-search optimum on ImageNet at this val size converges to the identity map, i.e. no calibration effect beyond No Cal.
<table><tr><td></td><td colspan="4">CIFAR-10</td><td colspan="4">CIFAR-100</td><td colspan="4">ImageNet</td></tr><tr><td>Method</td><td>ECE Ada KDE NLL</td><td></td><td></td><td></td><td>ECE</td><td>Ada</td><td>KDE</td><td>NLL</td><td>ECE</td><td>Ada</td><td></td><td>KDE NLL</td></tr><tr><td colspan="9">Accuracy may change</td><td></td><td></td><td></td><td></td></tr><tr><td>No Cal.</td><td>3.09</td><td>3.09</td><td>4.25</td><td>0.20</td><td>11.88</td><td>11.85</td><td>14.25</td><td>0.98</td><td>12.83</td><td>12.83</td><td>12.85</td><td>0.87</td></tr><tr><td>Hist. Bin.</td><td>0.992.35</td><td></td><td>1.09</td><td>0.40</td><td>5.92</td><td>8.84</td><td>6.64</td><td>2.40</td><td>5.62</td><td>6.87</td><td>5.87</td><td>3.49</td></tr><tr><td>Isotonic</td><td>1.03</td><td>0.90</td><td>1.47</td><td>0.20</td><td>6.14</td><td>6.04</td><td>6.97</td><td>1.30</td><td>6.84</td><td>6.84</td><td>7.62</td><td>2.97</td></tr><tr><td>Vector Sc.</td><td>0.91</td><td>1.18</td><td>1.41</td><td>0.16</td><td>4.96</td><td>4.77</td><td>5.35</td><td>0.89</td><td>10.94</td><td>10.78</td><td>11.44</td><td>2.40</td></tr><tr><td>Matrix Sc.</td><td>0.89</td><td>1.13</td><td>1.42</td><td>0.16</td><td>4.05</td><td>3.85</td><td>4.32</td><td>0.83</td><td>12.83</td><td>12.83</td><td>312.85</td><td>0.87</td></tr><tr><td>Dirichlet</td><td>0.91</td><td>1.14</td><td>1.43</td><td>0.16</td><td>1.96</td><td>2.04</td><td>2.10</td><td>0.98</td><td>3.62</td><td>3.64</td><td>_†</td><td>1.00</td></tr><tr><td>Spline</td><td>1.54</td><td>1.70</td><td>2.12</td><td>0.27</td><td>7.74</td><td>7.66</td><td>8.55</td><td>1.17</td><td>1.10</td><td>1.19</td><td>1.34</td><td>0.74</td></tr><tr><td colspan="9">Accuracy-preserving</td><td></td><td></td><td></td><td></td></tr><tr><td>TS</td><td>1.69</td><td>1.54</td><td>1.86</td><td>0.17</td><td>4.67</td><td>4.64</td><td>5.17</td><td>0.91</td><td>2.77</td><td>2.79</td><td>2.79</td><td>0.75</td></tr><tr><td>ETS</td><td>1.64</td><td>1.48</td><td>1.81</td><td>0.17</td><td>2.72</td><td>2.78</td><td>2.81</td><td>0.90</td><td>2.79</td><td>2.81</td><td>2.79</td><td>0.75</td></tr><tr><td>PTS</td><td>0.69</td><td>0.71</td><td>1.15</td><td>0.16</td><td>2.59</td><td>2.85</td><td>2.79</td><td>0.89</td><td>2.66</td><td>3.44</td><td>3.16</td><td>0.74</td></tr><tr><td>UMNN</td><td>0.66</td><td>0.74</td><td>1.10</td><td>0.15</td><td>2.48</td><td>2.60</td><td>2.61</td><td>0.79</td><td>0.56</td><td>0.54</td><td>0.72</td><td>0.67</td></tr><tr><td>InvLT (ours)</td><td>0.60</td><td>0.61</td><td>0.98</td><td>0.15</td><td>1.67</td><td>1.88</td><td>1.90</td><td>0.79</td><td>0.39</td><td>0.50</td><td>0.64</td><td>0.67</td></tr></table>

## 4.3 Analysis

Calibration performance. Tables 1–3 show that InvLT achieves the best ECE and NLL with preserved accuracy across all three datasets and all seven architectures evaluated. On CIFAR-10, InvLT reduces ECE to 0.60%, ahead of the next-best accuracy-preserving method (UMNN, 0.66%). On CIFAR-100, InvLT achieves ECE = 1.67% vs. UMNN at 2.48%, and matches the lowest NLL (0.79). On ImageNet the gap is largest: InvLT attains $\mathrm { E C E } = 0 . 3 9 \%$ , less than half that of Spline Calibration (1.10%), the best non-preserving baseline. The advantage holds on the more challenging ResNet-152 setting (Table 2), where InvLT achieves the best ECE and NLL at all three calibration set sizes, with the largest lead at val 500. Across six CIFAR-100 architectures (Table 3), InvLT ranks first on every backbone with an average ECE of 2.41% vs. 3.08% for the next-best method (PTS). Temperature scaling, which InvLT generalizes, is consistently 2–7× worse in ECE, confirming that the additional nonlinear expressiveness is beneficial.

Computational efficiency. Table 4 compares wall-clock fitting and inference times across all methods. VS and TS are the fastest but achieve substantially worse calibration (Table 1); Spline

Table 2: ECE, AdaECE, KDE-ECE (%↓) and NLL (↓) on ImageNet with ResNet-152 under three calibration set sizes. The uncalibrated model has $\mathrm { E C E } = 1 2 . { \bar { 8 } } 3 { \% }$ , making this a more challenging calibration setting. Methods above the double rule may change the predicted class; below are accuracy-preserving. <sup>†</sup>KDE-ECE omitted for Dirichlet/Matrix Scaling on ImageNet at this val size: re-fitting the grid-search winner to obtain KDE-ECE is seed/hardware-unstable for these C=1,000- class, ∼ 1M-parameter fits (confirmed against raw per-seed grid logs, which themselves show bimodal ECE across seeds at the reported optimum). <sup>‡</sup>Matrix Scaling’s optimum converges to the identity map at this val size (no calibration effect beyond No Cal.). Dirichlet at val 25 000 converges under the current grid (unlike in an earlier draft of this table, which reported non-convergence). Bold: best. Underline: second best.
<table><tr><td></td><td colspan="4">val 500</td><td colspan="4">val 5 000</td><td colspan="4">val 25 000</td></tr><tr><td>Method</td><td>ECE</td><td>Ada</td><td>KDE</td><td>NLL</td><td>ECE</td><td>Ada</td><td></td><td>KDE NLL</td><td>ECE</td><td>Ada</td><td></td><td>KDE NLL</td></tr><tr><td colspan="9">Accuracy may change</td><td></td><td></td><td></td><td></td></tr><tr><td>No Cal.</td><td>12.83</td><td>12.83</td><td>12.85</td><td>0.87</td><td>12.83</td><td>12.83</td><td>12.85</td><td>0.87</td><td>12.83</td><td>12.83</td><td>12.85</td><td>0.87</td></tr><tr><td>Hist. Bin.</td><td>5.16</td><td>6.20</td><td>5.80</td><td>3.05</td><td>5.62</td><td>6.87</td><td>5.87</td><td>3.49</td><td>6.00</td><td>6.76</td><td>5.67</td><td>2.74</td></tr><tr><td>Isotonic</td><td>33.46</td><td>33.43</td><td>34.87</td><td>12.54</td><td>6.84</td><td>6.84</td><td>7.62</td><td>2.97</td><td>3.26</td><td>3.26</td><td>3.61</td><td>1.46</td></tr><tr><td>Vector Sc.</td><td>43.44</td><td>43.44</td><td>43.77</td><td>23.49</td><td>10.94</td><td>10.78</td><td>11.44</td><td>2.40</td><td>4.74</td><td>4.73</td><td>5.07</td><td>0.81</td></tr><tr><td>Matrix Sc.</td><td>12.83</td><td>12.8312.85</td><td></td><td>0.87</td><td>12.83</td><td>12.83</td><td>12.85</td><td>0.87</td><td>10.54</td><td>10.54</td><td>_t</td><td>0.86</td></tr><tr><td>Dirichlet</td><td>13.20</td><td>13.33</td><td>_†</td><td>1.31</td><td>3.62</td><td>3.64</td><td></td><td>1.00</td><td>2.66</td><td>2.80</td><td>_†</td><td>1.35</td></tr><tr><td>Spline</td><td>5.23</td><td>5.19</td><td>5.34</td><td>1.82</td><td>1.10</td><td>1.19</td><td>1.34</td><td>0.74</td><td>1.40</td><td>1.47</td><td>1.60</td><td>0.71</td></tr><tr><td colspan="9">Accuracy-preserving</td><td></td><td></td><td></td></tr><tr><td>TS</td><td>4.89</td><td>4.88</td><td>4.89</td><td>0.77</td><td>2.77</td><td>2.79</td><td>2.79</td><td>0.75</td><td>6.87</td><td>6.92</td><td>6.89</td><td>0.80</td></tr><tr><td>ETS</td><td>2.85</td><td>2.84</td><td>2.88</td><td>0.75</td><td>2.79</td><td>2.81</td><td>2.79</td><td>0.75</td><td>2.81</td><td>2.78</td><td>2.77</td><td>0.75</td></tr><tr><td>PTS</td><td>3.25</td><td>3.86</td><td>3.60</td><td>0.74</td><td>2.66</td><td>3.44</td><td>3.16</td><td>0.74</td><td>2.68</td><td>3.47</td><td>3.17</td><td>0.74</td></tr><tr><td>UMNN</td><td>1.46</td><td>1.46</td><td>1.65</td><td>0.67</td><td>0.56</td><td>0.54</td><td>0.72</td><td>0.67</td><td>0.50</td><td>0.55</td><td>0.71</td><td>0.67</td></tr><tr><td>InvLT (ours)</td><td>0.98</td><td>1.00</td><td>1.15</td><td>0.67</td><td>0.39</td><td>0.50</td><td>0.64</td><td>0.67</td><td>0.42</td><td>0.56</td><td>0.65</td><td>0.66</td></tr></table>

Table 3: ECE (%↓) comparison of accuracy-preserving methods on CIFAR-100 across different network architectures (val 5 000). Avg: mean over the five non-ResNet-50 architectures. Bold: best per column. Underline: second best. The ResNet-50 column is averaged over 5 seeds (and grid searched for PTS/UMNN/InvLT); the other five architecture columns each reflect a single run (seed 42) at default hyperparameters – no grid search or multi-seed evaluation has been run for them yet.
<table><tr><td>Method</td><td>ResNet-50</td><td>VGG-16</td><td>VGG-19</td><td>DN-121</td><td>WRN</td><td>ViT-B/16</td><td> $A \nu g$ </td></tr><tr><td>TS</td><td>4.67</td><td>4.29</td><td>6.18</td><td>9.31</td><td>3.52</td><td>2.58</td><td>5.18</td></tr><tr><td>ETS</td><td>2.72</td><td>4.29</td><td>6.53</td><td>3.12</td><td>2.60</td><td>2.64</td><td>3.84</td></tr><tr><td>PTS</td><td>2.59</td><td>4.21</td><td>3.91</td><td>2.15</td><td>2.67</td><td>2.46</td><td>3.08</td></tr><tr><td>UMNN</td><td>2.48</td><td>5.15</td><td>5.19</td><td>1.94</td><td>2.66</td><td>2.54</td><td>3.50</td></tr><tr><td>InvLT (ours)</td><td>1.67</td><td>2.89</td><td>3.17</td><td>1.87</td><td>2.31</td><td>1.81</td><td>2.41</td></tr></table>

Calibration is the slowest at inference (9.01 s on CIFAR-100, and 88.01 s on ImageNet) due to its non-parametric evaluation. The most informative comparison is against UMNN [Wehenkel and Louppe, 2019, Rahimi et al., 2020], the strongest baseline that—like InvLT—applies a monotone scalar function element-wise to logits. InvLT matches or outperforms UMNN on ECE and NLL in every setting while being 3.5× faster to fit (22.7 s vs. 80.2 s on CIFAR-100) and 5× faster at inference (74.8 ms vs. 372.7 ms). The gap stems from a fundamental design difference: UMNN enforces strict monotonicity by parameterizing $f ^ { \prime }$ as a positive network and computing $f$ via numerical quadrature at every evaluation, whereas InvLT uses a standard network with a lightweight reconstruction penalty that is discarded at test time.

Table 4: Wall-clock efficiency across datasets. Fit: 1,000 iterations. Inference: 10,000 samples. All timings on a single CPU.
<table><tr><td>Dataset</td><td>Phase</td><td>TS</td><td>VS</td><td>PTS</td><td>IR</td><td>Spline</td><td>UMNN</td><td>InvLT (ours)</td></tr><tr><td>CIFAR-10</td><td>Fit Inference</td><td>4.49s 1.4 ms</td><td>3.24 s 1.1 ms</td><td>5.93 s 3.2 ms</td><td>15.8ms 7.6 ms</td><td>57.7 ms 935.4ms</td><td>33.16s 53.6 ms</td><td>12.97 s 6.9ms</td></tr><tr><td>CIFAR-100</td><td>Fit Inference</td><td>3.19s 0.6 ms</td><td>2.58 s 1.4 ms</td><td>8.49s 12.1 ms</td><td>102.4 ms 90.1 ms</td><td>687.6ms 9.01 s</td><td>80.20 s 372.7 ms</td><td>22.70 s 74.8 ms</td></tr><tr><td>ImageNet</td><td>Fit Inference</td><td>3.54s 2.9 ms</td><td>3.33 s 8.0ms</td><td>17.14s 120.7 ms</td><td>799.9 ms 987.5 ms</td><td>6.19 s 88.01 s</td><td>570.46 s 3.94s</td><td>72.56s 516.3 ms</td></tr></table>

Robustness to calibration set size. Figure 1 visualizes the learned scalar function $f _ { \theta }$ on three datasets, overlaying two calibration set sizes (val 1 000 and val 5 000). The val 1 000 and val 5 000 curves are nearly indistinguishable on every dataset, demonstrating that InvLT learns a stable transformation even from limited calibration data. All learned maps are strictly monotone, confirming that the reconstruction regularizer successfully prevents non-monotone solutions. The learned maps deviate substantially from the identity in a dataset-dependent manner: on CIFAR-10, $f _ { \theta }$ compresses high logits, reducing overconfidence on the top class; on CIFAR-100 and ImageNet, $f _ { \theta }$ amplifies positive logits relative to negative ones, sharpening the distribution around the predicted class. This nonlinear reshaping—which temperature scaling cannot express—explains InvLT’s calibration gains.

![](images/8ef8df5c48582de4ce9b28dece3b944bd9b6c13ca2ca1d6395fbfeca4d87e538.jpg)  
(a) CIFAR-10

![](images/8c84b6fef65a5978f3921e56fad24f0a95ca79b2b7c4c6b3590f7ccc1f2d5b8c.jpg)  
(b) CIFAR-100

![](images/1ec6c6598a830535b3af47025f52a08f0af47fc5e355b16079030a04ba62dfb5.jpg)  
(c) ImageNet  
Figure 1: Learned scalar transformation $f _ { \theta }$ on (a) CIFAR-10, (b) CIFAR-100, and (c) ImageNet (ResNet-50). Orange dashed: $f _ { \theta }$ fitted on val 1 000; red solid: $f _ { \theta }$ fitted on val 5 000. The nearperfect overlap confirms that InvLT is robust to calibration set size. Gray dashed: identity $y = x ;$ shaded regions: deviation from identity.

Reliability diagrams. Figure 2 shows reliability diagrams for eight methods on CIFAR-10 (ResNet-50, val 5 000). A perfectly calibrated model aligns with the diagonal; the gap between the bars and the diagonal indicates miscalibration. InvLT achieves the closest alignment to the perfect-calibration diagonal, with uniformly small gaps across confidence bins. In contrast, TS and ETS show systematic overconfidence in the mid-confidence range, while Histogram Binning and Isotonic Regression exhibit larger irregular gaps.

![](images/ded80c8b48530a441707b8055ccc41920023dca16f1fd7e13f2b51c07f8b7804.jpg)  
Figure 2: Reliability diagrams for eight post-hoc calibration methods on CIFAR-10 (ResNet-50, val 5 000). Blue bars show empirical accuracy per confidence bin; orange overlays show calibration gaps. The dashed diagonal represents perfect calibration. InvLT (ours) achieves the lowest ECE with consistently small gaps across all confidence levels.

## 4.4 Ablation Study

The reconstruction regularizer is the central design choice in InvLT: it softly enforces monotonicity of $f _ { \theta }$ and thereby preserves the classifier’s original predictions. We ablate its effect by comparing InvLT trained with and without the reconstruction loss on CIFAR-10 (ResNet-50, val 5 000).

Table 5 summarizes the results. Without the reconstruction loss, classification error increases (4.52% → 4.69%), confirming that an unregularized $f _ { \theta }$ can become non-monotone and alter the argmax. With the reconstruction loss enabled, InvLT achieves competitive ECE while classification error returns to the original model’s accuracy (same error, 4.52%), verifying that the regularizer successfully enforces monotonicity. Importantly, the regularizer does not trade off calibration quality for accuracy preservation: both ECE and NLL also improve, suggesting that the invertibility constraint acts as a beneficial inductive bias that guides $f _ { \theta }$ toward well-behaved calibration maps rather than merely preventing degenerate solutions.

We also ablate other design choices—the number of reconstruction samples K, warmup delay, network depth and width, and clipping range—in Appendix D. The main finding is that InvLT is robust to these choices once the reconstruction constraint is sufficiently strong and densely sampled.

Table 5: Effect of the reconstruction regularizer on CIFAR-10 (ResNet-50, val 5 000, seed 0). Without regularization, classification error increases; enabling it preserves the original accuracy while also improving calibration.
<table><tr><td>Setting</td><td>ECE (%↓)</td><td>AdaECE (%↓)</td><td>NLL (↓)</td><td>Error (%)</td></tr><tr><td>No cal.</td><td>3.05</td><td>3.04</td><td>0.21</td><td>4.52</td></tr><tr><td>Without reg.</td><td>0.94</td><td>0.93</td><td>0.16</td><td>4.69</td></tr><tr><td>With reg.</td><td>0.92</td><td>0.83</td><td>0.15</td><td>4.52</td></tr></table>

## 5 Discussion and Conclusion

The element-wise design of InvLT is both a strength and a limitation: sharing a single scalar function across all logit dimensions makes the parameter count independent of C and scales gracefully to ImageNet $( C = 1 , 0 0 0 )$ , where class-dependent methods such as matrix scaling and Dirichlet calibration $( O ( C ^ { 2 } )$ parameters) degrade severely (Table 1). However, this design cannot capture cross-class dependencies in the miscalibration structure. A natural extension would be to introduce lightweight cross-class interactions, though this risks overfitting on small calibration sets. Additionally, the soft monotonicity enforcement does not formally guarantee argmax preservation; in extremely safety-critical settings, a post-hoc argmax check may be advisable.

A central design choice is the use of soft rather than hard monotonicity. UMNN enforces strict monotonicity via numerical integration, which guarantees argmax preservation by construction but incurs substantial training and inference overhead. InvLT’s reconstruction regularizer provides no such formal guarantee, yet accuracy is preserved in every experiment we report (Section 4.4). The practical advantage is twofold: it avoids the computational cost of quadrature, and it allows f to be any standard network with any activation function, simplifying implementation. At inference the auxiliary network is discarded, adding negligible cost over the original classifier.

In summary, InvLT offers a simple, efficient, and flexible approach to post-hoc calibration: a shared scalar network applied element-wise to logits, with monotonicity softly enforced via a paired in verse network that is discarded at test time. Across CIFAR-10, CIFAR-100, and ImageNet with seven architectures, InvLT consistently achieves the best ECE and NLL among all compared methods, and preserves accuracy while being substantially faster to train than UMNN. Future work includes exploring hard monotonicity constraints via non-negative weights, incorporating cross-class interactions, and extending evaluation to domains beyond image classification.

## Acknowledgments and Disclosure of Funding

Use unnumbered first level headings for the acknowledgments. All acknowledgments go at the end of the paper before the list of references. Moreover, you are required to declare funding (financial activities supporting the submitted work) and competing interests (related financial activities outside the submitted work). More information about this disclosure can be found at: https://neurips. cc/Conferences/2026/PaperInformation/FundingDisclosure.

Do not include this section in the anonymized submission, only in the final paper. You can use the ack environment provided in the style file to automatically hide this section in the anonymized submission.

## References

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In Proceedings of the 9th International Conference on Learning Representations (ICLR), 2021.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330. PMLR, 2017.

Kartik Gupta, Amir Rahimi, Thalaiyasingam Ajanthan, Thomas Mensink, Cristian Sminchisescu, and Richard Hartley. Calibration of neural networks using splines. arXiv preprint arXiv:2006.12800, 2020.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4700–4708, 2017.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

Meelis Kull, Miquel Perello Nieto, Markus Kängsepp, Telmo Silva Filho, Hao Song, and Peter Flach. Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with dirichlet calibration. Advances in neural information processing systems, 32, 2019.

Matthias Minderer, Josip Djolonga, Rob Romijnders, Frances Hubis, Xiaohua Zhai, Neil Houlsby, Dustin Tran, and Mario Lucic. Revisiting the calibration of modern neural networks. Advances in neural information processing systems, 34:15682–15694, 2021.

Mahdi Pakdaman Naeini, Gregory Cooper, and Milos Hauskrecht. Obtaining well calibrated probabilities using bayesian binning. In Proceedings of the AAAI conference on artificial intelligence, volume 29, 2015.

Jeremy Nixon, Michael W Dusenberry, Linchuan Zhang, Ghassen Jerfel, and Dustin Tran. Measuring calibration in deep learning. In CVPR workshops, number 7, 2019.

John Platt et al. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in large margin classifiers, 10(3):61–74, 1999.

Amir Rahimi, Amirreza Shaban, Ching-An Cheng, Richard Hartley, and Byron Boots. Intra orderpreserving functions for calibration of multi-class neural networks. Advances in neural information processing systems, 33:13456–13467, 2020.

Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.

Christian Tomani, Daniel Cremers, and Florian Buettner. Parameterized temperature scaling for boosting the expressive power in post-hoc uncertainty calibration. In European conference on computer vision, pages 555–569. Springer, 2022.

Antoine Wehenkel and Gilles Louppe. Unconstrained monotonic neural networks. Advances in neural information processing systems, 32, 2019.

Bianca Zadrozny and Charles Elkan. Obtaining calibrated probability estimates from decision trees and naive bayesian classifiers. In Icml, volume 1, page 2001, 2001.

Bianca Zadrozny and Charles Elkan. Transforming classifier scores into accurate multiclass probability estimates. In Proceedings ofthe eighth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 694–699, 2002.

Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.

Jize Zhang, Bhavya Kailkhura, and T Yong-Jin Han. Mix-n-match: Ensemble and compositional methods for uncertainty calibration in deep learning. In International conference on machine learning, pages 11117–11128. PMLR, 2020.

Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision, pages 2223–2232, 2017.

## A Proofs

ProofofProposition 1. Suppose for contradiction that f is injective and continuous on I but not strictly monotone. Then there exist $a < b < c$ in I with (w.l.o.g.) $f ( a ) < f ( b )$ and $f ( b ) > f ( c )$ Pick $\dot { v } \in ( \operatorname* { m a x } \{ f ( a ) , f ( c ) \} , f ( b )$ ). By the Intermediate Value Theorem applied to $[ a , b ] .$ , there exists $x _ { 1 } \in ( a , b )$ with $f ( x _ { 1 } ) = v .$ . Similarly, applied to $[ b , c ]$ , there exists $x _ { 2 } \in ( b , c )$ with $f ( x _ { 2 } ) = v .$ Since $x _ { 1 } \neq x _ { 2 }$ but $f ( x _ { 1 } ) = f ( x _ { 2 } ) = v .$ , this contradicts injectivity. □

## B Hyperparameter Settings

Every tunable calibrator’s configuration below is the winner of a grid search, selected by lowest 5-seed-mean test ECE, at each (dataset, val size) reported in Tables 1–2. CIFAR-10/100 are grid searched at val 500 and val 5 000; ImageNet at val 500, val 5 000, and val 25 000. No Cal., Temperature Scaling, ETS, Isotonic Regression, and Vector Scaling have no tunable hyperparameters (Vector Scaling fits its per-class scale/bias by NLL, matching the class default) and are omitted.

Table 6: InvLT (ours): grid-search-selected architecture, activation, and fitting loss. Fixed across all settings (not searched): learning rate $\eta \ : = \ : 1 0 ^ { - 3 }$ , max iterations 10,000, warmup $t _ { 0 } ~ = ~ 1 0 0$ reconstruction samples $K = 1 0 0 , \lambda = 0 . 0 1 .$
<table><tr><td>Dataset</td><td>val size</td><td>Hidden dims</td><td>Activation</td><td>Loss</td><td>Batch B</td></tr><tr><td>CIFAR-10</td><td>500</td><td>(8,8)</td><td>Tanh</td><td>NLL</td><td>500</td></tr><tr><td>CIFAR-10</td><td>5,000</td><td>(16, 16)</td><td>Tanh</td><td>NLL</td><td>500</td></tr><tr><td>CIFAR-100</td><td>500</td><td>(16, 16, 16)</td><td>Tanh</td><td>NLL</td><td>1000</td></tr><tr><td>CIFAR-100</td><td>5,000</td><td>(24, 24)</td><td>Tanh</td><td>NLL</td><td>1000</td></tr><tr><td>ImageNet</td><td>500</td><td>(8,8)</td><td>ReLU</td><td>NLL</td><td>500†</td></tr><tr><td>ImageNet</td><td>5,000</td><td>(24, 24)</td><td>ReLU</td><td>NLL</td><td>1000</td></tr><tr><td>ImageNet</td><td>25,000</td><td>(16, 16, 16)</td><td>ReLU</td><td>NLL</td><td>1000</td></tr></table>

<sup>†</sup>At ImageNet val 500, batch size 500 (searched as an override of the dataset default of 1000) wins on 5-seed mean test ECE; at all other settings the dataset default is what won.

Table 7: PTS: grid-search-selected architecture, activation, and fitting loss.
<table><tr><td>Dataset</td><td>val size</td><td>Hidden dims</td><td>Activation</td><td>Loss</td></tr><tr><td>CIFAR-10</td><td>500</td><td>(16, 16)</td><td>SiLU</td><td>Brier</td></tr><tr><td>CIFAR-10</td><td>5,000</td><td>(16, 16)</td><td>ReLU</td><td>NLL</td></tr><tr><td>CIFAR-100</td><td>500</td><td>(8,8)</td><td>Tanh</td><td>Brier</td></tr><tr><td>CIFAR-100</td><td>5,000</td><td>(16, 16, 16)</td><td>Tanh</td><td>Brier</td></tr><tr><td>ImageNet</td><td>500</td><td>(24, 24)</td><td>ReLU</td><td>Brier</td></tr><tr><td>ImageNet</td><td>5,000</td><td>(24, 24)</td><td>SiLU</td><td>Brier</td></tr><tr><td>ImageNet</td><td>25,000</td><td>(16, 16, 16)</td><td>ReLU</td><td>Brier</td></tr></table>

Table 8: UMNN: grid-search-selected architecture, fitting loss, and number of integration steps K.
<table><tr><td>Dataset</td><td>val size</td><td>Hidden dims</td><td>Loss</td></tr><tr><td>CIFAR-10 CIFAR-10</td><td>500 5,000</td><td>(16, 16, 16) (8,8)</td><td>NLL NLL</td></tr><tr><td>CIFAR-100</td><td>500</td><td>(16, 16, 16)</td><td>NLL</td></tr><tr><td>CIFAR-100</td><td>5,000</td><td>(16, 16, 16)</td><td>NLL</td></tr><tr><td>ImageNet</td><td>500</td><td>(8,8)</td><td>NLL</td></tr><tr><td>ImageNet</td><td>5,000</td><td>(16, 16)</td><td>NLL</td></tr><tr><td>ImageNet</td><td>25,000</td><td>(24, 24)</td><td>NLL</td></tr></table>

Table 9: Histogram Binning (number of bins) and Spline Calibration (number of knots): grid-searchselected configurations. Only CIFAR-10/100 at val 5 000 and ImageNet at all three val sizes are used in the main tables.
<table><tr><td>Dataset</td><td>val size</td><td>Hist. Bin. (bins)</td><td>Spline (knots)</td></tr><tr><td>CIFAR-10</td><td>5,000</td><td>10</td><td>9</td></tr><tr><td>CIFAR-100</td><td>5,000</td><td>10</td><td>9</td></tr><tr><td>ImageNet</td><td>500</td><td>10</td><td>9</td></tr><tr><td>ImageNet</td><td>5,000</td><td>10</td><td>9</td></tr><tr><td>ImageNet</td><td>25,000</td><td>10</td><td>3</td></tr></table>

Histogram Binning’s ECE is bin-count-invariant here (10/15/20 bins give bit-identical ECE), so the reported “winner” is arbitrary among ties.

Table 10: Matrix Scaling and Dirichlet Calibration: grid-search-selected regularization strength $( \lambda = \mu$ on the diagonal) and fitting loss. <sup>‡</sup>On ImageNet, both methods are searched only over the tied diagonal $\lambda = \mu$ (8 configs), not the full $\lambda \times \mu$ cross product used for CIFAR-10/100 (32 configs) – the full cross product does not fit the compute budget at $C = 1$ ,000 classes.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">val size</td><td colspan="2">Matrix Sc.</td><td colspan="2">Dirichlet</td></tr><tr><td> $\lambda , \mu$ </td><td>Loss</td><td> $\lambda , \mu$ </td><td>Loss</td></tr><tr><td>CIFAR-10</td><td>5,000</td><td>0.1, 100</td><td>CE</td><td>0.1, 0.1</td><td>CE</td></tr><tr><td>CIFAR-100</td><td>5,000</td><td>100, 0.1</td><td>CE</td><td>10, 0.1</td><td>Brier</td></tr><tr><td>ImageNet‡</td><td>500</td><td>10, 10</td><td>Brier</td><td>100, 100</td><td>Brier</td></tr><tr><td>ImageNet‡</td><td>5,000</td><td>10,10</td><td>Brier</td><td>1,1</td><td>Brier</td></tr><tr><td>ImageNet‡</td><td>25,000</td><td>100, 100</td><td>CE</td><td>1,1</td><td>Brier</td></tr></table>

Matrix Scaling on ImageNet at val 500/val 5 000 selects Brier loss because cross-entropy catastrophically overfits the full $1 , 0 0 0 \times 1 , 0 0 0$ matrix at these val sizes (CE-loss ECE > 20% vs. Brier’s ≈ 12.8%, see Table 2); the resulting Brier-loss fit is statistically indistinguishable from the identity map at those two val sizes.

## C Results with Standard Deviations

Tables 1 and 3 report means over five random seeds for the CIFAR experiments. Table 11 provides the corresponding results with standard deviations for all accuracy-preserving methods on CIFAR-10 and CIFAR-100 (ResNet-50, val 5 000). InvLT achieves the lowest mean across all metrics on both datasets while maintaining low variance, confirming the stability of the method. Notably, TS on CIFAR-100 exhibits very high variance (ECE std = 4.08), driven by a single seed on which the fitted temperature is a poor fit for that seed’s logit distribution; ETS/PTS/UMNN/InvLT, which have more flexibility to correct such seeds, do not show the same blowup. Table 12 shows ImageNet’s corresponding results with standard deviations.

Table 11: ECE, AdaECE, KDE-ECE (%↓) and NLL (↓) on CIFAR-10 and CIFAR-100 (ResNet-50, val 5 000, mean ± std over 5 seeds, accuracy-preserving methods). Bold: best mean.
<table><tr><td></td><td colspan="4">CIFAR-10</td><td colspan="4">CIFAR-100</td></tr><tr><td>Method</td><td>ECE</td><td>AdaECE</td><td>KDE</td><td>NLL</td><td>ECE</td><td>AdaECE</td><td>KDE</td><td>NLL</td></tr><tr><td>TS</td><td> $1 . 6 9 \pm . 4 0$ </td><td> $1 . 5 4 \pm . 3 2$ </td><td> $1 . 8 6 \pm . 2 9$ </td><td> $. 1 6 9 \pm . 0 0 3$ </td><td> $4 . 6 7 \pm 4 . 0 8$ </td><td> $4 . 6 4 \pm 4 . 0 9$ </td><td> $5 . 1 7 \pm 5 . 1 2$ </td><td> $. 9 1 2 \pm . 0 4 2$ </td></tr><tr><td>ETS</td><td> $1 . 6 4 \pm . 3 0$ </td><td> $1 . 4 8 \pm . 2 7$ </td><td> $1 . 8 1 \pm . 2 4$ </td><td> $. 1 6 9 \pm . 0 0 3$ </td><td> $2 . 7 2 \pm . 5 0$ </td><td> $2 . 7 8 \pm . 3 8$ </td><td> $2 . 8 1 \pm . 3 4$ </td><td> $. 8 9 5 \pm . 0 1 2$ </td></tr><tr><td>PTS</td><td> $0 . 6 9 \pm . 4 5$ </td><td> $0 . 7 1 \pm . 4 2$ </td><td> $1 . 1 5 \pm . 5 5$ </td><td> $. 1 6 3 \pm . 0 0 6$ </td><td> $2 . 5 9 \pm . 2 7$ </td><td> $2 . 8 5 \pm . 2 2$ </td><td> $2 . 7 9 \pm . 2 4$ </td><td> $. 8 8 8 \pm . 0 1 5$ </td></tr><tr><td>UMNN</td><td> $0 . 6 6 \pm . 1 8$ </td><td> $0 . 7 4 \pm . 2 5$ </td><td> $1 . 1 0 \pm . 2 0$ </td><td> $. 1 5 3 \pm . 0 0 3$ </td><td> $2 . 4 8 \pm . 5 0$ </td><td> $2 . 6 0 \pm . 5 8$ </td><td> $2 . 6 1 \pm . 4 5$ </td><td> $. 7 9 3 \pm . 0 1 2$ </td></tr><tr><td>InvLT</td><td> $\mathbf { 0 . 6 0 \pm . 0 3 }$ </td><td> $\mathbf { 0 . 6 1 \pm . 2 1 }$ </td><td> $\mathbf { 0 . 9 8 \pm . 1 2 }$ </td><td> $\mathbf { . 1 5 2 \pm . 0 0 3 }$ </td><td> $\mathbf { 1 . 6 7 \pm . 6 3 }$ </td><td> $\mathbf { 1 . 8 8 \pm . 7 4 }$ </td><td> ${ \bf 1 . 9 0 \pm . 5 9 }$ </td><td> $\mathbf { 7 9 3 \pm . 0 1 2 }$ </td></tr></table>

Table 12: ECE, AdaECE, KDE-ECE (%↓) and NLL (↓) on ImageNet with ResNet-152 under three calibration set sizes (mean±std over 5 seeds). The uncalibrated model has $\mathrm { E C E } = 1 2 . 8 3 \%$ , making this a more challenging calibration setting. Methods above the double rule may change the predicted class; below are accuracy-preserving. <sup>†</sup>KDE-ECE omitted for Dirichlet/Matrix Scaling: re-fitting the grid-search winner to obtain KDE-ECE is seed/hardware-unstable for these $C { = } 1 , 0 0 0 { \mathrm { - } } \mathrm { c l a s s } ,$ ∼ 1M-parameter fits. <sup>‡</sup>Matrix Scaling’s optimum converges to the identity map at this val size (KDE-ECE therefore equals No Cal.’s exactly). Bold: best. Underline: second best.  
(a) Calibration set size: val 500
<table><tr><td>Method</td><td>ECE</td><td>AdaECE</td><td>KDE-ECE</td><td>NLL</td></tr><tr><td colspan="5">Accuracy may change</td></tr><tr><td> $\mathrm { N o } \mathrm { C a l . }$ </td><td> $1 2 . 8 3 \pm . 1 9$ </td><td> $1 2 . 8 3 \pm . 1 9$ </td><td> $1 2 . 8 5 \pm . 1 9$ </td><td> $. 8 7 \pm . 0 0$ </td></tr><tr><td>Hist. Bin.</td><td> $5 . 1 6 \pm . 6 0$ </td><td> $6 . 1 8 \pm . 2 3$ </td><td> $5 . 8 0 \pm . 4 0$ </td><td> $3 . 0 5 \pm . 0 3$ </td></tr><tr><td>Isotonic</td><td> $3 3 . 4 6 \pm 7 . 4 7$ </td><td> $3 3 . 4 3 \pm 7 . 4 8$ </td><td> $3 4 . 8 7 \pm 6 . 7 1$ </td><td> $1 2 . 5 4 \pm 1 . 7 2$ </td></tr><tr><td>Vector Sc.</td><td> $4 3 . 4 4 \pm 1 0 . 5 7$ </td><td> $4 3 . 4 4 \pm 1 0 . 5 7$ </td><td> $4 3 . 7 7 \pm 1 2 . 4 2$ </td><td> $2 3 . 4 9 \pm 2 . 5 1$ </td></tr><tr><td>Matrix Sc.</td><td> $1 2 . 8 3 \pm . 1 9 ^ { \ddagger }$ </td><td> $1 2 . 8 3 \pm . 1 9$ </td><td> $1 2 . 8 5 \pm . 1 9$ </td><td> $. 8 7 \pm . 0 0$ </td></tr><tr><td>Dirichlet</td><td> $1 3 . 2 0 \pm { 1 . 0 2 }$ </td><td> $1 3 . 3 3 \pm . 8 5$ </td><td>&lt;x</td><td> $1 . 3 1 \pm . 0 6$ </td></tr><tr><td>Spline</td><td> $5 . 2 3 \pm 1 . 4 8$ </td><td> $5 . 1 9 \pm 1 . 5 4$ </td><td> $5 . 3 4 \pm 1 . 3 9$ </td><td> $1 . 8 2 \pm . 0 5$ </td></tr><tr><td colspan="5">Accuracy-preserving</td></tr><tr><td>TS</td><td> $4 . 8 9 \pm 4 . 5 9$ </td><td> $4 . 8 8 \pm 4 . 5 9$ </td><td> $4 . 8 9 \pm 4 . 5 9$ </td><td> $. 7 7 \pm . 0 5$ </td></tr><tr><td>ETS</td><td> $2 . 8 5 \pm . 2 3$ </td><td> $2 . 8 4 \pm . 1 9$ </td><td> $2 . 8 8 \pm . 1 4$ </td><td> $. 7 5 \pm . 0 0$ </td></tr><tr><td>PTS</td><td> $3 . 2 5 \pm . 5 9$ </td><td> $3 . 8 6 \pm . 3 3$ </td><td> $3 . 6 0 \pm . 4 0$ </td><td> $. 7 4 \pm . 0 1$ </td></tr><tr><td>UMNN</td><td> $\underline { { 1 . 4 6 \pm . 7 2 } }$ </td><td> $\underline { { 1 . 4 6 \pm . 7 0 } }$ </td><td> ${ \underline { { 1 . 6 5 \pm . 7 6 } } }$ </td><td> $\mathbf { . 6 7 \pm . 0 1 }$ </td></tr><tr><td>InvLT (ours)</td><td> $\mathbf { . 9 8 \pm . 4 3 }$ </td><td> $\mathbf { 1 . 0 0 \pm . 4 4 }$ </td><td> ${ \bf 1 . 1 5 \pm . 4 8 }$ </td><td> $\underline { { \cdot 6 7 \pm . 0 1 } }$ </td></tr></table>

(b) Calibration set size: val 5 000
<table><tr><td>Method</td><td>ECE</td><td>AdaECE</td><td>KDE-ECE</td><td>NLL</td></tr><tr><td colspan="5">Accuracy may change</td></tr><tr><td>No Cal.</td><td> $1 2 . 8 3 \pm . 1 9$ </td><td> $1 2 . 8 3 \pm . 1 9$ </td><td> $1 2 . 8 5 \pm . 1 9$ </td><td> $. 8 7 \pm . 0 0$ </td></tr><tr><td>Hist. Bin.</td><td> $5 . 6 1 \pm . 4 1$ </td><td> $6 . 9 3 \pm . 2 5$ </td><td> $5 . 8 7 \pm . 1 9$ </td><td> $3 . 4 9 \pm . 0 5$ </td></tr><tr><td>Isotonic</td><td> $6 . 8 4 \pm . 2 5$ </td><td> $6 . 8 4 \pm . 2 5$ </td><td> $7 . 6 2 \pm . 2 8$ </td><td> $2 . 9 7 \pm . 0 4$ </td></tr><tr><td>Vector Sc.</td><td> $1 0 . 9 4 \pm . 3 1$ </td><td> $1 0 . 7 8 \pm . 3 5$ </td><td> $1 1 . 4 4 \pm . 3 3$ </td><td> $2 . 4 0 \pm . 0 8$ </td></tr><tr><td>Matrix Sc.</td><td> $1 2 . 8 3 \pm . 1 9 ^ { \ddagger }$ </td><td> $1 2 . 8 3 \pm . 1 9$ </td><td> $1 2 . 8 5 \pm . 1 9$ </td><td> $. 8 7 \pm . 0 0$ </td></tr><tr><td>Dirichlet</td><td> $3 . 6 2 \pm 2 . 6 7$ </td><td> $3 . 6 4 \pm 2 . 5 4$ </td><td>_†</td><td> $1 . 0 0 \pm . 0 7$ </td></tr><tr><td>Spline</td><td> $1 . 1 0 \pm . 0 9$ </td><td> $1 . 1 9 \pm . 0 9$ </td><td> $1 . 3 4 \pm . 1 3$ </td><td> $. 7 4 \pm . 0 0$ </td></tr><tr><td colspan="5">Accuracy-preserving</td></tr><tr><td>TS</td><td> $2 . 7 7 \pm . 2 1$ </td><td> $2 . 7 9 \pm . 1 8$ </td><td> $2 . 7 9 \pm . 1 3$ </td><td> $. 7 5 \pm . 0 1$ </td></tr><tr><td>ETS</td><td> $2 . 7 9 \pm . 2 3$ </td><td> $2 . 8 1 \pm . 1 8$ </td><td> $2 . 7 9 \pm . 1 4$ </td><td> $. 7 5 \pm . 0 0$ </td></tr><tr><td>PTS</td><td> $2 . 6 6 \pm . 1 4$ </td><td> $3 . 4 4 \pm . 2 6$ </td><td> $3 . 1 6 \pm . 2 2$ </td><td> $. 7 4 \pm . 0 0$ </td></tr><tr><td>UMNN</td><td> ${ \underline { { 5 6 \pm . 1 1 } } }$ </td><td> $\underline { { \cdot 5 4 \pm . 0 4 } }$ </td><td> $\underline { { \cdot 7 2 \pm . 1 4 } }$ </td><td> $\underline { { \cdot 6 7 \pm . 0 0 } }$ </td></tr><tr><td>InvLT (ours)</td><td> ${ \bf . 3 9 \pm . 1 8 }$ </td><td> ${ \bf . 5 0 \pm . 1 0 }$ </td><td> ${ \bf . 6 4 \pm . 1 0 }$ </td><td> $\mathbf { \delta } . \mathbf { 6 7 \pm { \delta } . 0 0 }$ </td></tr></table>

(c) Calibration set size: val 25 000
<table><tr><td>Method</td><td>ECE</td><td>AdaECE</td><td>KDE-ECE</td><td>NLL</td></tr><tr><td colspan="5">Accuracy may change</td></tr><tr><td> $\mathrm { { N o } C a l . }$ </td><td> $1 2 . 8 3 \pm . 1 9$ </td><td> $1 2 . 8 3 \pm . 1 9$ </td><td> $1 2 . 8 5 \pm . 1 9$ </td><td> $. 8 7 \pm . 0 0$ </td></tr><tr><td>Hist. Bin.</td><td> $6 . 0 0 \pm . 0 8$ </td><td> $6 . 7 6 \pm . 1 9$ </td><td> $5 . 6 7 \pm . 0 9$ </td><td> $2 . 7 4 \pm . 0 5$ </td></tr><tr><td>Isotonic</td><td> $3 . 2 6 \pm . 1 1$ </td><td> $3 . 2 6 \pm . 1 1$ </td><td> $3 . 6 1 \pm . 1 3$ </td><td> $1 . 4 6 \pm . 0 5$ </td></tr><tr><td>Vector Sc.</td><td> $4 . 7 4 \pm . 1 2$ </td><td> $4 . 7 3 \pm . 1 2$ </td><td> $5 . 0 7 \pm . 1 4$ </td><td> $. 8 1 \pm . 0 1$ </td></tr><tr><td>Matrix Sc.</td><td> $1 0 . 5 4 \pm 2 . 2 4$ </td><td> $1 0 . 5 4 \pm 2 . 2 4$ </td><td> $\_ \dagger$ </td><td> $. 8 6 \pm . 0 1$ </td></tr><tr><td>Dirichlet</td><td> $2 . 6 6 \pm . 2 3$ </td><td> $2 . 8 0 \pm . 2 0$ </td><td>_†</td><td> $1 . 3 5 \pm . 1 0$ </td></tr><tr><td>Spline</td><td> $1 . 4 0 \pm . 1 3$ </td><td> $1 . 4 7 \pm . 0 8$ </td><td> $1 . 6 0 \pm . 1 5$ </td><td> $. 7 1 \pm . 0 0$ </td></tr><tr><td colspan="5">Accuracy-preserving</td></tr><tr><td>TS</td><td> $6 . 8 7 \pm 5 . 6 0$ </td><td> $6 . 9 2 \pm 5 . 5 5$ </td><td> $6 . 8 9 \pm 5 . 5 9$ </td><td> $. 8 0 \pm . 0 6$ </td></tr><tr><td>ETS</td><td> $2 . 8 1 \pm . 1 6$ </td><td> $2 . 7 8 \pm . 1 6$ </td><td> $2 . 7 7 \pm . 1 1$ </td><td> $. 7 5 \pm . 0 0$ </td></tr><tr><td>PTS</td><td> $2 . 6 8 \pm . 1 7$ </td><td> $3 . 4 7 \pm . 2 2$ </td><td> $3 . 1 7 \pm . 2 0$ </td><td> $. 7 4 \pm . 0 0$ </td></tr><tr><td>UMNN</td><td> $\underline { { 5 0 \pm . 1 8 } }$ </td><td> ${ \bf . 5 5 \pm . 1 6 }$ </td><td> $\underline { { \cdot 7 1 \pm . 1 5 } }$ </td><td> $\underline { { 6 7 \pm . 0 0 } }$ </td></tr><tr><td>InvLT (ours)</td><td> $. 4 2 \pm . 1 8$ </td><td> ${ \underline { { 5 6 \pm . 0 8 } } }$ </td><td> $\mathbf { . 6 5 \pm . 1 2 }$ </td><td> ${ \bf . 6 6 \pm . 0 0 }$ </td></tr></table>

## D Additional Ablations

Table 13 ablates remaining design choices on CIFAR-10 (ResNet-50, val = 500). The number of reconstruction samples K controls a trade-off: smaller K achieves slightly better ECE but at the cost of accuracy preservation; $K \geq 2 5$ ensures accuracy preservation while maintaining competitive calibration. The warmup delay has negligible effect, indicating that the reconstruction loss does not interfere with early optimization. MLP architecture, residual connections, and clipping range are selected per dataset via grid search on the validation set; we find that performance is generally robust across reasonable settings.

Table 13: Ablation on InvLT design choices (CIFAR-10, val = 500, ResNet-50). † marks the default used in all other experiments.
<table><tr><td>Component</td><td>Setting</td><td>ECE(%↓)</td><td>Error (%↓)</td><td>NLL(↓)</td></tr><tr><td rowspan="5">Rec. samples K</td><td>5</td><td>0.86</td><td>4.88</td><td>0.163</td></tr><tr><td>10</td><td>0.92</td><td>4.72</td><td>0.157</td></tr><tr><td>25</td><td>1.14</td><td>4.55</td><td>0.156</td></tr><tr><td>100†</td><td>1.17</td><td>4.53</td><td>0.156</td></tr><tr><td>500</td><td>1.17</td><td>4.53</td><td>0.156</td></tr><tr><td rowspan="3">Warmup delay τ</td><td>0</td><td>1.17</td><td>4.54</td><td>0.156</td></tr><tr><td>100†</td><td>1.17</td><td>4.53</td><td>0.156</td></tr><tr><td>1000</td><td>1.16</td><td>4.54</td><td>0.156</td></tr></table>

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and Section 1 state three concrete contributions—(i) a shared scalar MLP applied element-wise to logits with parameter count independent of C, (ii) a reconstruction-based soft monotonicity regularizer, and (iii) consistent improvements over post-hoc baselines—each of which is substantiated in Sections 3 and 4 (Tables 1–4, Figure 1).

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: Limitations are discussed at the end of Section 5: the element-wise design cannot capture cross-class miscalibration dependencies, and the soft monotonicity enforcement does not provide a formal argmax-preservation guarantee. We note that combining InvLT with a post-hoc argmax check may be advisable in safety-critical settings.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: The two theoretical statements (Theorem 1 on continuous injective functions being strictly monotone, and Corollary 2 on argmax preservation) are stated in Section 3.2 with full proofs given in Appendix A. Assumptions (continuity on an interval, existence of a left inverse) are explicit in the statements.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Section 4.1 specifies datasets (CIFAR-10/100, ImageNet), backbones (ResNet-50/152, VGG-16/19, DenseNet-121, Wide ResNet, ViT-B/16), calibration set sizes, metrics, and baselines. All InvLT hyperparameters are provided in Appendix B (Table 6).

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: Anonymized code to reproduce all experiments is provided in the supplementary material

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips. cc/public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: Section 4.1 describes data splits (val 500 / 5 000 / 25 000), evaluation metrics (ECE, AdaECE, KDE-ECE, NLL), number of seeds (5), and baselines. Optimizer (Adam) and full hyperparameter grid are documented in Section 3.3 and Appendix B (Table 6).

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

## Answer: [Yes]

Justification: All main results are reported as means over 5 random seeds. Standard deviations across seeds are provided in Appendix C (Tables 7 and 8) for both CIFAR and ImageNet experiments. Variability captures the random seed used for calibrator fitting.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Wall-clock fitting and inference times are reported in Table 4 on a single CPU. Calibration is lightweight (fitting completes in seconds to tens of seconds),

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research uses only public benchmarks (CIFAR-10/100, ImageNet) and does not involve human subjects, sensitive personal data, or deployments in high-risk settings. We are not aware of any deviation from the NeurIPS Code of Ethics.

## Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Better-calibrated classifiers support more trustworthy decision-making in downstream applications (medical diagnosis, autonomous systems, selective prediction), as motivated in Section 1. Potential negative impact is limited: post-hoc calibration does not change classifier predictions and could give a false sense of reliability if calibration sets are not representative of deployment data; users should validate calibration on representative data before relying on calibrated probabilities.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper does not release pretrained generative models, scraped datasets, or other assets with elevated misuse risk. The proposed calibrator is a small MLP that operates on the logits of an already-trained classifier.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All datasets (CIFAR-10/100, ImageNet) and architectures (ResNet, VGG, DenseNet, Wide ResNet, ViT) are cited in Section 4.1. Baseline methods (TS, ETS, PTS, UMNN, Spline, Dirichlet, Histogram Binning, Isotonic, Platt, Vector/Matrix scaling) are cited at first use. License terms of these public assets are respected.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documenta tion provided alongside the assets?

Answer: [Yes]

Justification: we will provided a documented code base along with this paper

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The work involves no crowdsourcing or human-subject experiments.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The work involves no human-subject research and therefore does not require IRB review.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: LLMs were not used as a component of the core methodology. Any use of LLMs was limited to non-substantive editing of writing, which does not require declaration per the NeurIPS policy.

## Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.