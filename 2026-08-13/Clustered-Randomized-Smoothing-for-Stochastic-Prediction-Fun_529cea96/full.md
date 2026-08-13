# Clustered Randomized Smoothing for Stochastic Prediction Functions

Eduardo Figueiredo<sup>⋆,1</sup> Frederik Mathiesen<sup>⋆,1</sup> Julian Schumann<sup>2</sup>

Jens Kober<sup>3</sup> Arkady Zgonnikov<sup>2</sup> Luca Laurenti<sup>1,4</sup>

<sup>1</sup>Delft Center for Systems and Control, Delft University of Technology

<sup>2</sup>Cognitive Robotics, Delft University of Technology <sup>3</sup>University of Stuttgart <sup>4</sup>AI4I {e.figueiredo, f.b.mathiesen, j.f.schumann, a.zgonnikov, l.laurenti}@tudelft.nl

## Abstract

Modern stochastic predictors can model rich, multi-modal outcome distributions. However, this expressive power comes with challenges in ensuring robust predictions—a critical requirement in safety-critical domains. Randomized smoothing is a leading technique for improving robustness, particularly against adversarial perturbations. Yet, in stochastic multi-modal regression settings, randomized smoothing often fails due to mode collapse, yielding averaged predictions that do not reflect the underlying distribution. To address this limitation, we propose clustered α-smoothing, a framework that (1) partitions noisy samples using an arbitrary clustering algorithm, (2) applies α-smoothing locally within each cluster, and (3) combines the resulting predictions into a mixture distribution. By interpreting the smoothing distribution as a mixture of α-smoothers, we derive a lower bound on the probability that the smoothed prediction lies within a union of compact regions corresponding to distinct modes. We empirically evaluate our framework on two benchmarks, demonstrating substantial improvements over state-of-the-art methods. In stochastic trajectory prediction on a driving simulator dataset, our approach achieves, on average, a 27% lower Wasserstein distance to the groundtruth distribution compared to α-smoothing. In quadrotor control, where modes correspond to distinct feasible paths to a target, our method reduces the collision rate by 81% relative to the state-of-the-art randomized smoothing.

## 1 Introduction

Stochastic predictors, such as variational autoencoders [10], normalizing flows [28], or Bayesian neural networks [43], are capable of modeling rich, multi-modal outcome distributions [21, 12]. However, this expressive power comes with challenges in ensuring robust predictions – a fundamental requirement in safety-critical applications such as electrical grids, healthcare, robotics, or autonomous driving, where this type of brittle behavior can have catastrophic consequences [9, 24, 23, 37]. A particular concern is vulnerability to adversarial attacks [15, 22, 35, 29, 6], in which carefully crafted, small perturbations of the input cause significant deviations in the network’s output. To address this issue, a wide range of methods has been developed for formally verifying desired input–output properties of neural networks, including mixed-integer programming [16], abstract interpretation [13], linear relaxations with branch-and-bound [48, 45], satisfiability modulo theory solvers [44, 11], and reachability analysis [41, 1]. Despite their strong guarantees, these methods are often computationally expensive, which limits their practical applicability.

Randomized smoothing is a scalable alternative for certifying robustness [20, 17, 8]. Given a base predictor, randomized smoothing injects noise into the input and averages the resulting predictions, yielding a smoothed predictor with certified robustness guarantees. While most prior work focuses on classification [20, 17, 8], recent efforts extend these ideas to regression [7, 31, 30, 36]. In this setting, given a base predictor $h : \mathbb { R } ^ { d }  \mathbb { R } ^ { q }$ and a Gaussian noise $\varepsilon \sim \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } I )$ with $\sigma ^ { 2 } > 0$ , the smoothed predictor is commonly defined as (some variant of) $\mathcal { H } ( \boldsymbol { x } ) = \mathbb { E } _ { \boldsymbol { \epsilon } } \left[ h ( \boldsymbol { x } + \boldsymbol { \varepsilon } ) \right]$ ]. However, this approach is fundamentally limited for stochastic neural network predictors. In these models, predictions are often inherently multi-modal [10, 28, 43], yet randomized smoothing collapses this structure by averaging over noisy outputs, yielding a unimodal and potentially uninformative estimate (Figure 1). This limitation is particularly problematic in safety-critical applications where multi-modality is essential, such as trajectory prediction [27, 26, 4, 37] or failure prediction [33, 49].

![](images/f8d72f6bb10a6d4dcec2f4ee7c8391c53d7936ea8b4cece9dda6154a68fa12ee.jpg)  
Figure 1: Conceptual illustration of mode collapse in randomized smoothing. In (a), an ego vehicle (red) is predicting the trajectory of another vehicle (green) at a cross junction. The stochastic predictor outputs a distribution with two modes: turning left before the ego vehicle passes the intersection, or waiting for it to pass before initiating the turn. In (b), current randomized smoothing methods (e.g., [30]) would produce an averaged prediction, failing to capture any of the two modes. Instead, in (c), our approach is to smooth each mode individually and combine as a mixture model, which preserves the modes corresponding to the two tactical behaviors. Sets represent the sets that can be certified with each method.

To address this limitation and capture the multi-modal behavior of the base predictor, we propose clustered α-smoothing. Given an input x and a set of noise samples $\left\{ \varepsilon _ { 1 } , \dots , \varepsilon _ { N } \right\}$ , our method (i) clusters the prediction samples $\{ h _ { \bf w } ( x + \varepsilon _ { i } ) , \dots , h _ { \bf w } ( x + \varepsilon _ { N } ) \}$ , (ii) trims α outlier samples per cluster, and (iii) smooths locally within each cluster. The clusters are then combined according to a mixture distribution. By combining the modes/clusters as mixture model rather than averaging, we avoid mode collapse, and thus, the limitation of the previous state-of-the-art. Further, the per-mode α-trimming allows greater control over outliers. We prove by classical probability theory results and statistical methods that the output of the smoothed predictor belongs to a coverage region with high probability for all inputs in a radius r. The guaranteed coverage region is provided as the union of the cluster-wise coverage regions $\mathcal { R } _ { m }$ , which are regions containing the samples of each cluster with high probability (see Figure 1), and the probability of the prediction belonging to the coverage region is provided both individually and jointly for all clusters. Our framework is agnostic to the clustering algorithm and provides flexibility in how to determine the number of clusters and sample assignment. We demonstrate the framework on two benchmarks: (1) stochastic multi-modal trajectory prediction in traffic (Figure 1), a key prerequisite for safe and efficient automated driving, and (2) robustification of a multi-modal deep RL quadrotor controller, where the framework help improve robustness while preserving multiple feasible navigation strategies around obstacles.

In summary, our main contributions are: (i) we introduce a framework for randomized smoothing for multi-modal stochastic predictors via clustering and α-trimming, (ii) we prove that the output of the clustered α-smoothed predictor with high probability falls within a small coverage region per cluster for all inputs in a radius, and (iii) we demonstrate the effectiveness of our framework in benchmarks from stochastic trajectory prediction in traffic and a multi-modal RL controller for a quadrotor, showing improved performance compared to state-of-the-art.

## 2 Preliminaries on randomized smoothing

Notation. For a vector $x \in \mathbb { R } ^ { d }$ , we denote by $\boldsymbol { x } ^ { ( i ) }$ its i-element. Given $N \in \mathbb { N } _ { > 0 } ,$ the set $\{ 1 , \ldots , N \}$ is denoted by [N]. I denotes the identity matrix. The Pontryagin difference between sets A, B is denoted by $A \ominus B$ . For a set $\boldsymbol { \mathcal { X } } \subseteq \mathbb { R } ^ { \boldsymbol { \dot { d } } }$ , the indicator function for is denoted as $\mathbf { 1 } _ { \mathcal { X } } ( x ) : = 1 \mathrm { i f } x \in \mathcal { X } ;$ otherwise 0. Given a Borel measurable space $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ , we denote by ( ) the Borel sigma algebra over and by $\mathcal { P } ( \mathcal { X } )$ the set of probability distributions on . For a random variable x taking values in $\mathcal { X } , \mathbf { x } \sim \overset { \mathbf { \bar { p } } } { \mathbb { P } } _ { \mathbf { x } } \overset { } { \in } \mathcal { P } ( \mathcal { X } )$ represents the probability measure associated to x. Let $\begin{array} { r } { f _ { n , k } ( p ) = \binom { n } { k } p ^ { k } ( 1 - p ) ^ { n - k } } \end{array}$ denote throughout the paper the probability mass function of a binomial random variable with parameters n and k. For a vector $x \in { \mathcal { X } } ,$ the k-th order statistic of x is denoted by $x _ { ( k ) }$ . Further, we define a stochastic predictor as follows.

Definition 1 (Stochastic Predictor). Consider a deterministicfunction $h _ { w } : \mathcal { X } \to \mathcal { Y }$ with $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ and $\mathcal { V } \subset \mathbb { R } ^ { q }$ , where $w \in \mathbb { R } ^ { m }$ is a parameter vector. Then, for $x \in \mathcal { X }$ and a possibly input dependent random vector $\mathbf { w } \sim \mathbb { P } _ { \mathbf { w } \mid x } o n \mathbb { R } ^ { m }$ , the stochastic prediction induced by $h _ { w }$ and w at x is the random variable $h _ { \mathbf { w } } ( x )$

Our definition of stochastic predictor is sufficiently general to encompass important stochastic neural models such as Bayesian neural networks (BNNs) [43], variational autoencoders (VAEs) [10], and normalizing flows [28]. In what follows, for simplicity, we assume that the distribution of w does not depend on the input x. Accordingly, we write $\mathbb { P } _ { \mathbf { w } }$ instead of $\mathbb { P } _ { \mathbf { w } | x }$ . However, our methods in Sections 3 and 4 also apply to the case where w depends on the input x.

Randomized smoothing. Although the literature on randomized smoothing has been extensively developed for classification problems [8, 19, 46, 18], recent work has recently sparked interest in regression settings [7, 31, 30]. Given a stochastic predictor $h _ { \mathbf { w } }$ and $x \in \mathcal { X }$ , randomized smoothing can be viewed as the transformation $\mathcal { H } ( \boldsymbol { x } ) = \mathbb { E } _ { \boldsymbol { \epsilon } } [ \bar { h } _ { \mathbf { w } } ( \boldsymbol { x } + \boldsymbol { \varepsilon } ) ]$ ], i.e., the original stochastic function is replaced by its expectation w.r.t. noisy perturbations around x. In practice, computing $\mathcal { H } ( x )$ in closed-form is often infeasible [14], thus commonly replaced by its sampled approximation $\begin{array} { r } { \mathcal { H } _ { N } ( x ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } h _ { \mathbf { w } } ( x + \varepsilon _ { i } ) } \end{array}$ , where $\left\{ \varepsilon _ { 1 } , \dots , \varepsilon _ { N } \right\}$ are i.i.d. samples from the distribution of ε. However, as discussed in [30], such smoother presents a major weakness: the smoothed function is vulnerable to outliers in the prediction, with a single sample being capable of pushing the smoothed output beyond the desired range . Therefore, [30] proposes a α-trimming smoothing technique, which allows trimming these outliers. The α-smoothed predictor is defined as follows.

Definition 2 (α-smoothing [30]). Let $h _ { \mathbf { w } }$ be a stochastic predictor with $x , y$ , being its input and output spaces and $x \in \mathcal { X }$ . Given $N \in  { \mathbb { N } } _ { > 0 }$ i.i.d. samples $\left\{ \varepsilon _ { 1 } , \dots , \varepsilon _ { N } \right\}$ from a distribution $\mathbb { P } _ { \varepsilon } \in \mathcal { P } ( \mathcal { X } \ominus \{ x \} )$ , define the random vector $\mathcal { H } ( x ) = \left( h _ { \mathbf { w } } ( x + \varepsilon _ { i } ) \right) _ { i \in [ N ] } .$ Then, for $\alpha \in [ 0 , \frac { 1 } { 2 } )$ , the α-smoothed predictor is defined as

$$
\tilde { \mathcal { H } } _ { N , \alpha } ( x ) = \frac { 1 } { N - 2 \lfloor \alpha N \rfloor } \sum _ { i = 1 + \lfloor \alpha N \rfloor } ^ { N - \lfloor \alpha N \rfloor } \mathcal { H } ( x ) _ { ( i ) } ,\tag{1}
$$

where the i-th order statistic in $\mathcal { H } ( x ) _ { ( i ) }$ is taken element-wise.

α-smoothing allows tighter control over the probability that $\tilde { \mathcal { H } } _ { N , \alpha } ( x ^ { \prime } ) \in \mathcal { R }$ for any convex set $\mathcal { R } \subseteq \mathcal { V }$ . However, when the predictor distribution is multi-modal, the above-mentioned technique inadvertently collapses its modes (Section 3). To solve this issue, we propose clustered α-smoothing. Before introducing clustered α-smoothing, we should note that clustering has previously been applied successfully in randomized smoothing to robustify large language models [39, 42] under the justification that adversarial examples are often located in small clusters adjacent to the dominant mode. However, unlike these previous works, the safety-critical settings considered in this manuscript require that we account for behavior of the predictor in those minor clusters and thus cannot simply prune them, as done in [39, 42].

## 3 Clustered α-Smoothing

In this section, we introduce a novel randomized smoothing method tailored to stochastic predictors, which we call clustered α-smoothing. The key idea is to (i) cluster samples in the output space, (ii) apply smoothing locally within each cluster, and (iii) combine the resulting predictors into a mixture distribution, with weights proportional to the size of each cluster. By combining the locally smoothed predictors as a mixture, as illustrated in Figure 2, we prevent the mode collapse observed in α-smoothing [30]. To formally define clustered α-smoothing, we let $\pmb { \nu } = \{ \nu _ { 1 } , \dots , \nu _ { M } \}$ be a partition of the output space $\mathcal { V }$ into M disjoint sets, each corresponding to a mode of the prediction. For now, we assume that V is given; in Section 4, we describe how to construct V via clustering so that it aligns with the modes of the predictor.

Definition 3 (Clustered α-smoothing). Let $h _ { \mathbf { w } }$ be a stochastic predictor, $x \in \mathcal { X }$ a point in its domain, and $\pmb { \nu } = \{ \nu _ { 1 } , \dots , \nu _ { M } \}$ a partition of . Given $N \in  { \mathbb { N } } _ { > 0 }$ i.i.d. samples $\left\{ \varepsilon _ { 1 } , \dots , \varepsilon _ { N } \right\}$ from a distribution $\mathbb { P } _ { \varepsilon } \in \mathcal { P } ( \mathcal { X } \ominus \{ x \} )$ , we define the set of indexes $I _ { m } \subset [ N ]$ as $I _ { m } = \left\{ i \in [ N ] \right.$

![](images/b8a790d0e1eb8b7d569ffd8a2eb29374cd44a3cc2e4170b68206122a5d36c45e.jpg)

![](images/798ac691951e03d7a90a5e2470a654f8371ab368474efcf17c0872fcd91da916.jpg)  
Figure 2: An example of α-smoothing and clustered α-smoothing at $x = 2$ of a stochastic base predictor $h _ { \mathbf { w } } ( x ) = \mathbf { w } x$ where $\begin{array} { r } { \mathbf { w } \sim \sum _ { i = 1 } ^ { 3 } { \pi } _ { i } \mathcal { N } ( \cdot \mid \mu _ { i } , \sigma _ { i } ^ { 2 } ) } \end{array}$ is a mixture of three Gaussians with means $\mu = ( 1 , 0 , 2 )$ and equal variance $\sigma _ { i } ^ { 2 } = 0 . 0 \dot { 1 }$ , and weights $\pi = ( 0 . 2 , 0 . 2 , 0 . 6 )$ . The histograms are obtained by Monte Carlo sampling the distributions of $h _ { \mathbf { w } } ( x + \varepsilon ) , \tilde { \mathcal { H } } _ { N , \alpha , \{ \mathcal { V } \} } ( x )$ (in (a), equivalent to α-smoothing, see Remark 1), and our clustered α-smoothing technique, $\ddot { \mathcal { H } } _ { N , \alpha , \nu } ( x )$ , in (b) constructed by Algorithm 2. The Voronoi cell split represents the partition $\bar { \pmb { \nu } } = \{ \mathcal { V } _ { 1 } , \mathcal { V } _ { 2 } \}$ into two clusters, and the robustness regions in (b) denote sets that $\tilde { \mathcal { H } } _ { N , \alpha , \nu } ( x )$ belongs to with high probability (see Algorithm 1).

$$
h _ { \mathbf { w } } ( x + \varepsilon _ { i } ) \in \mathcal { V } _ { m } \Big \} . T h e n , d e f i n e \mathcal { H } _ { \mathcal { V } _ { m } } ( x ) = \big ( h _ { \mathbf { w } } ( x + \varepsilon _ { i } ) \big ) _ { i \in I _ { m } } a n d , f o r \alpha \in [ 0 , \frac { 1 } { 2 } ) \ c a l l
$$

$$
\mathcal { \tilde { H } } _ { N , \alpha , \mathcal { V } _ { m } } ( x ) = \frac { 1 } { | I _ { m } | - 2 \lfloor \alpha \lvert I _ { m } \rvert \rfloor } \sum _ { i = 1 + \lfloor \alpha \lvert I _ { m } \rvert \rfloor } ^ { | I _ { m } | - \lfloor \alpha \lvert I _ { m } \rvert \rfloor } \mathcal { H } _ { \mathcal { V } _ { m } } ( x ) _ { ( i ) } ,\tag{2}
$$

where the i-th ordering statist in $\mathcal { H } _ { \mathcal { V } _ { m } } ( x ) _ { ( i ) }$ is taken element-wise. Then, the clustered α-smoothing is defined asfollows

$$
\tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } } ( x ) = \sum _ { m = 1 } ^ { M } \mathbf { 1 } _ { \mathbf { z } = m } \tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } _ { m } } ( x )\tag{3}
$$

where z is a categorical random variable with class weights $\begin{array} { r } { \pi ^ { ( m ) } = \frac { | I _ { m } | } { N } } \end{array}$

Intuitively, the set of indexes $I _ { m }$ identifies to which partition cell each prediction sample $h _ { \mathbf { w } } ( x + \varepsilon _ { i } )$ belongs. Then $\tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } _ { m } }$ applies the α-trimming average to each partition cell individually. Finally, the clustered α-smoother $\tilde { \mathcal { H } } _ { N , \alpha , \nu } ( x )$ is constructed by randomly selecting each of the $\tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } _ { m } }$ with a probability proportional to how many samples are in each $\nu _ { m }$ . The parameter α in Definition 3 refers to trimming a 2α fraction of samples, i.e., treating them as outliers, per cluster. When $\alpha = 0$ $\tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } _ { m } } ( x )$ reduces to the mean smoothing, $\begin{array} { r } { \tilde { \mathcal { H } } _ { N , 0 , \mathcal { V } _ { m } } ( x ) = \frac { 1 } { | I _ { m } | } \sum _ { i \in I _ { m } } h _ { \mathbf { w } } ( x + \varepsilon _ { i } ) } \end{array}$ . When $\alpha  \frac { 1 } { 2 }$ $\tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } _ { m } } ( x )$ is equivalent to the median-smoothing, $\tilde { \mathcal { H } } _ { N , \alpha , \nu _ { m } } ( x ) =$ median $i \in I _ { m } \left( h _ { \mathbf { w } } ( x + \varepsilon _ { i } ) \right)$

Remark 1 (Relation with α-smoothing in [30]). When $\pmb { \nu } = \{ \mathcal { V } \} = \{ \mathcal { V } \}$ , then Definition 3 is equivalent to the α-smoothing in (1). However, when distributions exhibit multi-modality or are not strongly symmetric, then the flexibility of our approach can lead to a substantial improvement of performance over [30], as we will show in Section 5.

Robustness Guarantees. A core advantage of randomized smoothing is the ability to guarantee that, for sufficiently small perturbations, the output remains within a desired set with high probability. In this subsection, we develop this type of guarantee for clustered α-smoothing. Our approach, which results in Theorem 3.1 below, is to apply the Neyman-Pearson Lemma [8] to each cluster/mode, which are then combined to obtain worst-case bounds for the smoothed predictor via a union-bound argument combined with correction to the fact that V is a partition of .To apply the lemma, we follow the predominant convention in the randomized smoothing literature and adopt $\varepsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ for $\sigma ^ { 2 } > 0$ , which requires $\mathcal { X } = \mathbb { R } ^ { d }$

Theorem 3.1 (Robustness certification for clustered α-smoothing). Let $\pmb { \nu } = \{ \nu _ { 1 } , \dots , \nu _ { M } \}$ be a partition of the output space . Assume that $\begin{array} { r } { \check { p } \nu _ { m } \leq \mathbb { P } _ { ( \mathbf { w } , \varepsilon ) } \left( h _ { \mathbf { w } } ( x + \varepsilon ) \in \mathcal { V } _ { m } \right) \leq \hat { p } \nu _ { m } f o r } \end{array}$ each $\nu _ { m }$ . Furthermore, let ${ \mathcal { L } } \subset [ M ]$ and define $\tilde { \mathcal { R } } = \cup _ { l \in \mathcal { L } } \mathcal { R } _ { l }$ for convex sets $\mathcal { R } _ { l } \subset \mathcal { V } _ { l }$ . Assume that $\mathbb { P } _ { ( \mathbf { w } , \varepsilon ) } \left( h _ { \mathbf { w } } ( x + \varepsilon ) \in \mathcal { R } _ { l } \right) \geq \breve { p } _ { \mathcal { R } _ { l } }$ for each $\mathcal { R } _ { l } .$ . For $r \geq 0 ,$ definefor each $m \in [ M ]$ and $l \in { \mathcal { L } }$

$$
\underline { { p } } _ { \mathcal { V } _ { m } } = \Phi \left( \Phi ^ { - 1 } ( \tilde { p } _ { \mathcal { V } _ { m } } ) - \frac { r } { \sigma } \right) , \ \bar { p } _ { \mathcal { V } _ { m } } = \Phi \left( \Phi ^ { - 1 } ( \hat { p } _ { \mathcal { V } _ { m } } ) + \frac { r } { \sigma } \right) , \ a n d \ \underline { { p } } _ { \mathcal { R } _ { l } } = \Phi \left( \Phi ^ { - 1 } ( \tilde { p } _ { \mathcal { R } _ { l } } ) - \frac { r } { \sigma } \right) .\tag{4}
$$

![](images/041319f0e2ad13de8b46532b064b3e6779d799241dae867fb043d1c13919a706.jpg)  
Figure 3: Probability bounds $\begin{array} { r } { \sum _ { s = 1 } ^ { N } \frac { s } { N } \sum _ { j = s - \lfloor \alpha s \rfloor } ^ { s } f _ { s , j } \left( \frac { p _ { \mathcal { R } } } { p _ { \mathcal { V } } } \right) f _ { N , s } \left( p _ { \mathcal { V } } \right) } \end{array}$ from (6) attributable to each mode for different combinations of number of smoothing samples $N ,$ trimming parameter $\alpha ,$ partition set probability $p _ { \nu }$ and coverage ratio $p _ { \mathcal { R } } / p _ { \mathcal { V } }$ . The right-most column (where $p \nu = 1 )$ is equivalent to the bound in [30], by noting that only the term $s = N$ in (6) is non-zero when $\underline { { p } } _ { \nu } = p \nu = \bar { p } \nu = 1$

Then, for any $\delta \in \mathbb { R } ^ { d }$ such that $\begin{array} { r } { \| \delta \| _ { 2 } \leq r , } \end{array}$ , it holds that

$$
\begin{array} { r l } & { \mathbb { P } _ { ( \mathbf { w } , \varepsilon _ { 1 } , \ldots , \varepsilon _ { N } ) } \left( \mathcal { \tilde { H } } _ { N , \alpha , \mathcal { V } } ( x + \delta ) \in \tilde { \mathcal { R } } \right) } \\ & { \qquad \quad \overset { \mathrm { i n f } } { \geq } \underset { p _ { \mathcal { V } _ { m } } \in [ \underline { { p } } _ { \mathcal { V } _ { m } } , \overline { { p } } _ { \mathcal { V } _ { m } } ] } { \operatorname* { i n f } } \underset { l \in \mathcal { L } } { \sum } \underset { s = 1 } { \overset { N } { \sum } } \frac { s } { N } \underset { j = s - \lfloor \alpha s \rfloor } { \sum ^ { s } } f _ { s , j } \left( \frac { p _ { \mathcal { R } _ { l } } } { p _ { \mathcal { V } _ { l } } } \right) f _ { N , s } \left( p _ { \mathcal { V } _ { l } } \right) . } \end{array}\tag{5}
$$

The proof of Theorem 3.1 can be found in Appendix A. For each partition region $\nu _ { m } .$ , the interval $[ \underline { { p } } _ { \nu _ { m } } , \bar { p } \nu _ { m } ]$ bounds the probability that $h _ { \mathbf { w } } ( \bar { { \boldsymbol { x } } ^ { \cdot } } + \delta + \varepsilon ) \in \mathcal { V } _ { m }$ for any perturbation δ with norm $\left\| \delta \right\| _ { 2 } \leq r .$ . Analogously, $\underline { { p } } _ { \mathcal { R } _ { l } }$ represents a probability lower bounds on the event $h _ { \mathbf { w } } ( x + \delta + \varepsilon ) \in \mathcal { R } _ { l }$ Consequently, Theorem 3.1 combines these bounds to obtain a lower bound for the probability that $\tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } } ( x + \delta ) \in \tilde { \mathcal { R } }$ . Figure 3 shows how the various parameters affect the resulting bounds. It can be observed that a higher trimming parameter α leads to a significantly faster increase of the bound, as outlier output values are eliminated from the average predictor in (2), while higher ratios $\frac { p _ { \mathcal { R } _ { l } } } { p _ { \mathcal { V } _ { l } } } - \mathrm { i . e }$ larger coverage in the partition set – naturally increases the bound.

Before showing how to compute (5) in practice, it is instructive to examine the properties of the case $| \mathcal { L } | = 1$ first. In particular, the objective function in this case is independent from all other indices $[ M ] \setminus { \mathcal { L } }$ , and so, as we formally prove in Lemma A.1, the bound in (5) simply reduces to

$$
\begin{array} { r l } & { \mathbb { P } _ { ( \mathbf { w } , \varepsilon _ { 1 } , \ldots , \varepsilon _ { N } ) } \left( \mathcal { \tilde { H } } _ { N , \alpha , \mathcal { N } } ( x + \delta ) \in \mathcal { R } _ { l } \right) } \\ & { \qquad \geq \displaystyle \sum _ { s = 1 } ^ { N } \frac { s } { N } \sum _ { j = s - \lfloor \alpha s \rfloor } ^ { s } f _ { s , j } \left( \frac { p _ { \mathcal { R } _ { l } } } { \bar { p } _ { \mathcal { V } _ { l } } } \right) \operatorname* { m i n } \left\{ f _ { N , s } \left( \underline { { p } } _ { \mathcal { V } _ { l } } \right) , f _ { N , s } \left( \bar { p } _ { \mathcal { V } _ { l } } \right) \right\} . } \end{array}\tag{6}
$$

Solving the general case. Unfortunately, the computational convenience of the case $| \mathcal { L } | = 1$ does not extend to the general case $| { \mathcal { L } } | > 1$ , as in this setting (5) consists of a non-convex optimization problem. Instead, to achieve a computationally tractable result, we show how the optimization problem in (5) can be lower bounded by a linear program via anchor points and an error that can be controlled by the user. This is presented in Proposition 3.2.

Proposition 3.2. Assume that $\underline { { p } } _ { \nu _ { m } } , \bar { p } { \nu } _ { m }$ for every m $\in ~ [ M ]$ and $\underline { { p } } _ { \mathcal { R } _ { l } } f o r$ every $l \in \mathcal L$ are given. Define for each component $l \in \mathcal L$ the anchor points $p _ { \mathcal { V } _ { l } } ^ { ( 1 ) } < p _ { \mathcal { V } _ { l } } ^ { ( 2 ) } < \cdots < p _ { \mathcal { V } _ { l } } ^ { ( K ) } , p _ { \mathcal { V } _ { l } } ^ { ( k ) } \in [ \underline { { p } } _ { \mathcal { V } _ { l } } , \bar { p } \underline { { \nu } } _ { l } ] .$

Then, it holds that

$$
\begin{array} { r l r } {  { \operatorname* { i n f } _ { p \nu _ { m } \in [ \underline { { p } } _ { \mathcal { V } _ { m } } , \bar { p } \nu _ { m } ] } \sum _ { l \in \mathcal { L } } \sum _ { s = 1 } ^ { N } \frac { s } { N } \sum _ { j = s - \lfloor \alpha s \rfloor } ^ { s } f _ { s , j } ( \frac { p _ { \mathcal { R } _ { l } } } { p \nu _ { l } } ) f _ { N , s } ( p \nu _ { l } ) } } \end{array}\tag{7}
$$

$$
\ge \operatorname* { i n f } _ { \substack { p , \lambda _ { 1 } , \dots , \lambda _ { | { \mathcal { L } } | } } } \sum _ { l \in { \mathcal { L } } } \sum _ { k = 1 } ^ { K } \lambda _ { l } ^ { ( k ) } g _ { l } ( p _ { \gamma _ { l } } ^ { ( k ) } ) - \sum _ { l \in { \mathcal { L } } } \left[ \frac { N } { 1 - \bar { p } \nu _ { l } } + \frac { N } { \underline { { p } } _ { \mathcal { V } _ { l } } - \underline { { p } } _ { \mathcal { R } _ { l } } } \right] h _ { l , K }\tag{8}
$$

$$
s . t . \quad p _ { \mathcal N _ { m } } \in [ \underline { { p } } _ { \mathcal N _ { m } } , \bar { p } _ { \mathcal N _ { m } } ] , \sum _ { m = 1 } ^ { M } p _ { \mathcal N _ { m } } = 1 , \sum _ { k = 1 } ^ { K } \lambda _ { l } ^ { ( k ) } p _ { \mathcal N _ { l } } ^ { ( k ) } = p _ { \mathcal N _ { l } } , \lambda _ { l } \in \Delta _ { K } ,\tag{9}
$$

where the variable $h _ { l , K }$ and function g<sub>l</sub> are given by $h _ { l , K } = \operatorname* { m a x } _ { p \in [ \underline { { p } } _ { \mathcal { V } _ { l } } , \bar { p } \nu _ { l } ] } \operatorname* { m i n } _ { k \in [ K ] } | p - p _ { \mathcal { V } _ { l } } ^ { ( k ) } |$ and   
gl(p) = P<sup>N</sup><sub>s=1</sub> <sup>s</sup><sub>N</sub> fN,s(p) P<sup>s</sup><sub>j=s−⌊αs⌋</sub> fs,j  p p

The intuition behind Proposition 3.2 is the following. We seek a tractable lower bound on the infimum in (7). This is generally intractable. Consequently, instead, we replace $p _ { \mathcal R _ { l } } \ \mathsf { b y } \underline { { p } } _ { \mathcal R _ { l } }$ to obtain a valid lower bound, and then approximate the convex envelope of each inner component using a finite set of evaluation points – the anchor points – leading to a linear program in which the coupling constraint is handled natively. The bound is asymptotically exact as the number of anchor points K increases, at the cost of increasing the size of the linear program. We call the second term of (8) the residual, which represents an adjustment needed due to the fact that we only approximate the convex envelope. Note that, as we show in Appendix G, one can always achieve an arbitrarily small residual value by increasing K.

Remark 2 (Extension to $L _ { \rho } .$ -norm attack). Given $\rho \in [ 2 , \infty ) \cup \{ \infty \}$ , the guarantee (5) also holds $f o r \delta \in \mathbb { R } ^ { d }$ such that $\left. \delta \right. _ { \rho } \leq d ^ { \frac { 1 } { \rho } - \frac 1 2 } r$ by a straight-forward application ofHolder’s inequality [34].

## 4 Algorithm

In Theorem 3.1, given a point $x \in \mathcal { X }$ , we need to compute probability bounds for the events $h _ { \mathbf { w } } ( x + \varepsilon ) \in \mathcal { V } _ { m }$ and $h _ { \mathbf { w } } ( \bar { x } + \varepsilon ) \in \mathcal { R } _ { l }$ , where $\pmb { \nu } = \{ \gamma _ { m } \} _ { m = 1 } ^ { \hat { M } }$ is a partition of the output space into M sets, and $\mathcal { R } _ { l } \subseteq \mathcal { V } _ { l }$ are given subsets. Given the (possibly) non-linearity of the underlying deterministic function $h _ { w }$ on both the parameters w and input x, computing those quantities explicitly is generally intractable. Thus, one needs to resort to sample-based methods, for which probability bounds are estimated with high-confidence.

Proposition 4.1 (High-confidence probability bounds for sets). Let $\pmb { \nu } = \{ \gamma _ { m } \} _ { m = 1 } ^ { M }$ be a partition of the space , and $\bar { \mathcal { R } _ { l } } \subseteq \mathcal { V } _ { l }$ be given subsets. Given a confidence parameter $\beta \in [ 0 , 1 ]$ , and a number ofsamples $\bar { N } \in \mathbb { N } _ { > 0 }$ , let $\begin{array} { r } { Z _ { \mathcal { V } _ { m } } = \sum _ { i = 1 } ^ { N } \mathbf { 1 } _ { h _ { w _ { i } } ( x + \varepsilon _ { i } ) \in \mathcal { V } _ { m } } } \end{array}$ and $\begin{array} { r } { Z _ { \mathcal { R } _ { m } } = \sum _ { i = 1 } ^ { N } \mathbf { 1 } _ { h _ { w _ { i } } ( x + \varepsilon _ { i } ) \in \mathcal { R } _ { m } ; } } \end{array}$ , where $( w _ { i } , \varepsilon _ { i } )$ are i.i.d. samplesfrom $\mathbb { P } _ { ( \mathbf { w } , \varepsilon ) } . \ I f \breve { p } _ { \mathcal { V } _ { m } } , \hat { p } _ { \mathcal { V } _ { m } } , \breve { p } _ { \mathcal { R } _ { m } } \in [ 0 , 1 ]$ are such that<sup>1</sup>

$$
\frac { \beta } { 3 M } = \sum _ { j = { Z } _ { \nu _ { m } } } ^ { \tilde { N } } f _ { \tilde { N } , j } \left( \tilde { p } \nu _ { m } \right) , \ \frac { \beta } { 3 M } = \sum _ { j = 0 } ^ { Z _ { \nu _ { m } } } f _ { \tilde { N } , j } \left( \hat { p } \nu _ { m } \right) , \ a n d \ \frac { \beta } { 3 M } = \sum _ { j = { Z } _ { \kappa _ { m } } } ^ { \tilde { N } } f _ { \tilde { N } , j } \left( \tilde { p } _ { \mathcal { R } _ { m } } \right) ,\tag{10}
$$

then, the assumptions in Theorem 3.1 hold with confidence $1 - \beta .$

Proposition 4.1 comes from the Clopper-Pearson Lemma [8] combined with a union-bound argument. It gives a natural way to construct the partition V and sets R: after observing the samples $\{ \bar { h _ { w _ { i } } } ( x +$ $\varepsilon _ { i } ) \}$ , one may place the sets $\nu _ { m }$ and $\mathcal { R } _ { m }$ so that the number of outputs inside them (approximately) induces the desired probability level. This reasoning is the core principle of Algorithm 1.

Given the user’s desired coverage level p, Algorithm 1 generates the sets $\mathcal { R } _ { m }$ to cover a fraction p of the samples within $\nu _ { m }$ (Line 9). This parameter can be used to calibrate the trade-off between the size of the coverage sets and their certified probability. Next, by choosing the partition of the input space as a generalized Voronoi partition (closest region instead of point, Line 11; see Appendix E for details) [25], we enforce that the regions are contained in their partition, which is a requirement for Theorem 3.1. Although computing the (generalized) Voronoi partition associated with a set of regions R is hard (see Algorithm 2), we only need to check whether a point belongs to $\nu _ { m } .$ , which is equivalent to checking if $\mathcal { R } _ { m }$ is the closest region to this point. Thus, in practice, V is implicitly defined by R. With partition $\nu ,$ we can compute the robustness bounds introduced in Theorem 3.1 for the clustered α-smoother ${ \mathcal { H } } _ { N , \alpha , \nu } ,$ , as presented in Algorithm 2. Appendix C expands on the parameters of Algorithm 2 and their impact on the coverage regions and certified probabilities.

Algorithm 1 Construction of partition $\pmb { \nu } = \{ \nu _ { 1 } , . . . , \nu _ { M } \}$ and sets $\pmb { \mathcal { R } } = \left\{ \mathcal { R } _ { 1 } , . . . , \mathcal { R } _ { M } \right\}$ from   
samples such that $\mathcal { R } _ { i } \subset \mathcal { V } _ { i }$ and $\mathbb { P } ( h _ { \mathbf { w } } ( x + \varepsilon ) \in \mathcal { R } _ { i } ) / \mathbb { P } ( h _ { \mathbf { w } } ( x + \varepsilon ) \in \mathcal { V } _ { i } ) \approx p .$   
1: function CONSTRUCTCOVERAGESETS(x, $h _ { \mathbf { w } } , p , N )$   
2: Outputs $ \emptyset$   
3: for $i \gets 1 , \ldots , N$ do   
4: Draw $\varepsilon _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ and $w _ { i } \sim \mathbb { P } _ { \mathbf { w } }$   
5: Compute $h _ { w _ { i } } ( x + \varepsilon _ { i } )$ and append to Outputs   
6: Cluster the elements of Outputs ▷ $e . g .$ , using DBSCAN, DP-means, UMAP.   
7: Denote the resulting clusters by $C _ { 1 } , \ldots , C _ { M }$   
8: for $m  1 , \dots , M$ do   
9: $C _ { m } ^ { \left( p \right) }$ Select approximately a fraction p of the points in $C _ { m }$   
10: $\mathscr { R } _ { m } \gets \mathrm { E N V E L O P E } ( C _ { m } ^ { ( p ) } )$ $\triangleright \ell . g _ { * }$ , convex hull, hyperrectangle.   
11: V generalized Voronoi partition associated with R   
12: return partition V and coverage regions R

Algorithm 2 Robustness certification of $\tilde { \mathcal { H } } _ { N , \alpha , \nu }$ at $x \in \mathcal { X } ^ { 2 }$   
Require: Point $x \in \mathbb { R } ^ { d }$ , stochastic function $h _ { \mathbf { w } } ,$ smoothing parameters $\alpha \in [ 0 , 1 ]$ and $N \in \mathbb { N } _ { > 0 } ,$   
coverage level $p \in [ 0 , 1 ] ,$ , robustness radius $r \geq 0 ,$ , number of samples $\bar { N } \in \mathbb { N } _ { > 0 } .$ , and   
confidence level $\beta \in ( 0 , 1 )$   
Ensure: With confidence $1 - \beta ,$ for any m $\in [ M ] , \tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } } ( x + \delta ) \in \tilde { \mathcal { R } }$ with probability given by   
(5) for any $\delta \in \mathbb { R } ^ { d }$ such that $\left\| \delta \right\| _ { 2 } \leq r .$   
1: V, R CONSTRUCTCOVERAGESET $\mathsf { s } ( x , h _ { \mathbf { w } } , p , \bar { N } )$   
2: Construct $\tilde { \mathcal { H } } _ { N , \alpha , \nu } ( x )$   
3: Compute $\check { p } \nu _ { m } , \hat { p } \nu _ { m } , \check { p } \mathcal { R } _ { m } \in [ 0 , 1 ]$ with Proposition 4.1 for m $\in [ M ]$ (using $\beta$ and $\bar { N } )$   
4: Compute $\underline { { p } } _ { \mathcal { V } _ { m } } , \bar { p } \nu _ { m } , \underline { { p } } _ { \mathcal { R } _ { m } }$ using (4) for m $\dot { \in } [ M ]$   
5: return $\tilde { \mathcal { H } } _ { N , \alpha , \nu } ( x )$ and (5) certificate.

The time complexity of Algorithm 2 is generally dominated by the repeated inference cost of the base classifier. It is thus governed by the number of samples required, which is determined by (i) the data patterns and clustering algorithm, (ii) the confidence bound (binomial inference, see Proposition 4.1), and (iii) the probability bound in Theorem 3.1.

## 5 Experiments

We evaluated our framework on two benchmarks: stochastic trajectory prediction and multi-modal RL quadrotor control. For both experiments, we use the minimum bounding hyperrectangle for the envelope of each cluster (Appendix D provides further details on the experiment setup)<sup>3</sup>. All experiments were run on an Intel Core i7-1365U CPU with 16GB of RAM.

Trajectory Prediction. We consider the stochastic TrajFlow prediction model [26], a normalizing flow-based model (details in Appendix D.1), and apply it to the $L – G A P$ driving simulator dataset [47].

![](images/949fdf0e2d24163762bd61629ea93c9cdf679645838c1da81b1075ed24476bd6.jpg)  
Figure 4: Safety analysis for 200 randomly selected inputs $x _ { i }$ from the L-GAP dataset.

In this dataset, human drivers face a decision whether to turn left in front of an oncoming vehicle (go) or wait for it to pass before turning (yield). The recorded trajectories are converted into control inputs (acceleration and curvature) using the unicycle model [37]. The resulting stochastic predictor is then defined as $h _ { \mathbf { w } } : \mathbb { R } ^ { 1 8 } \to \mathbb { R } ^ { 4 0 }$ , where the input space $\mathbb { R } ^ { 1 8 }$ encodes the two control actions that underlie the 9 transitions between the 10 past recorded positions of the target agent. The output space $\mathbb { R } ^ { 4 0 }$ corresponds to the position of the target vehicle over 20 time-steps, here corresponding to a horizon of 2 s.

We start with a safety analysis comparing our approach to α-smoothing and RS-Reg [31] (which is equivalent to smoothing without any trimming, i.e. $\alpha = 0$ in (1)) using the L-GAP dataset. In particular, we evaluate the predicted probability that the vehicle will adopt the behavior $s t a y ^ { 4 }$ and we make the ego vehicle go if it predicts a stay probability for the other higher than a decision threshold of 95%. We then use the actual trajectory of the predicted vehicle to assess the risk of this decision: if both vehicles decided to go, we classify it as a risky decision. We then report the risk rate for 200 inputs in our database. We show that RS-Reg has a risk rate of 7.5%, α-smoothing of 16.5%, while ours only 2.5%. The main mechanism behind this result is that while α-smoothing and RS-Reg are often highly sure of the oncoming vehicle’s behavior (due to the mode collapse for both RS-Reg and α-smoothing), our clustered smoother allows for behavioral uncertainty that makes it more cautious about adopting the go decision, as illustrated in Figure 4.

Second, for a quantitative comparison of the discrepancy between the output distributions from each approach, we select a random subset of 200 input trajectories $x _ { i }$ from L-GAP and compute a sample approximation of the 2-Wasserstein distance $\mathbb { W } _ { 2 }$ between the noisy predictor $h _ { \mathbf { w } } ( x _ { i } + \varepsilon )$ and α-smoothing, RS-Reg, and ours note that we are thus comparing distributions embedded in a 40-dimensional space. We found that our clustered α-smoothing reduces the mean 2-Wasserstein distance to $h _ { \mathbf { w } } ( x + \varepsilon )$ in this subset of data in 27% compared to α-smoothing (3.80 vs. 5.18; Table 1). This confirms the intuition that the distributional flexibility of our method generally yields predictions that are closer to the original predictor, thereby better capturing its uncertainty (as also illustrated in Figure 7 in Appendix B).

<table><tr><td></td><td colspan="2">Wasserstein  $\mathbb { W } _ { 2 }$ </td></tr><tr><td>Method</td><td colspan="2">Trajectory  $( \mathbb { R } ^ { 4 0 } )$  Last step  $( \mathbb { R } ^ { 2 } )$ </td></tr><tr><td>Ours</td><td> ${ \bf 3 . 8 0 \pm 1 . 5 9 }$ </td><td> ${ \bf 1 . 4 9 \pm 0 . 6 3 }$ </td></tr><tr><td>RS-Reg</td><td> $4 . 7 0 \pm 2 . 0 9$ </td><td> $1 . 8 4 \pm 0 . 8 4$ </td></tr><tr><td>α-smoothing</td><td> $5 . 1 8 \pm 2 . 1 9$ </td><td> $2 . 0 4 \pm 0 . 8 4$ </td></tr></table>

Table 1: Comparison between smoothers and $h _ { \mathbf { w } }$ (x<sub>i</sub> + ε) in terms of $\mathbb { W } _ { 2 }$ for 200 inputs $x _ { i } .$

![](images/1c307062d8d185abe05ff23cb1fa89ea6e049470d5fe26402428032865ff253b.jpg)  
Figure 5: Robustness lower-bounds (with one std. dev. shaded band) for 20 distinct inputs $x _ { i } .$

Finally, we perform a parameter analysis of the robustness lower-bounds in Theorem 3.1 for various trimming parameters $\bar { \alpha } \in \{ 0 . 0 , 0 . 2 , 0 . 4 \}$ and robustness radii $r \geq 0 .$ , where the sets are constructed according to Algorithm 1 enveloping a fraction $p = 0 . 8 5$ , for different inputs in our database. In Figure 5, we observe that certified lower bounds are higher when the robustness radius is smaller, as this allows smaller perturbations from the reference input, and when the trimming parameter α is higher (so that outliers are filtered out, so we can be more certain about where the smoother probability mass is concentrated). Despite the solid certification suggested by Figure 5, decreasing the conservativeness of the lower bounds in Theorem 3.1 constitutes an interesting research direction.

![](images/3c991fa85be9baa3218ed26d9d8d452e7119dda8b871c9f3687d8f90b43eede4.jpg)

![](images/4f795e61a3a6b3a3a17ffcc23e3ac1710280aa0e6306022fe8b249a97faadf5b.jpg)

![](images/aa6944f22e4e376f9621e2ce91210ce94cf3c183b16358509454c53aca3ab5ef.jpg)

![](images/09dd0987a82a666feb2af4772221221d505f782d942cad7adc85bdd8307db745.jpg)  
(a) nominal policy

![](images/efcdce24e19ffb3d6ebea1cbc0bc20f83971c42b055f01c29608c7428300d649.jpg)  
(b) α-smoothing [30]

![](images/87680abc934ad7911bad98d95833f8cc26208a25274f027a68cff2bd59d1f9a2.jpg)  
(c) clustered α-smoothing (ours)  
Figure 6: Top row: rollouts of the quadrotor from the starting state (blue marker) where the controller predicts a bi-modal action distribution, corresponding to navigating through two “holes” in the obstacles towards the goal region (green). Bottom row: action distributions at the initial state $x = ( - 3 . 5 , 0 , - 0 . 3 5 , 0 , 0 . 7 , 0 )$ . The smoothed predictors are constructed using 30 samples. Insets: the certified probability of robustness of the smoothed policies with respect to their coverage regions (boxes) as a function of perturbation radius. The quantization in (b) arises from cross-mode smoothing with a limited number of samples (30 samples, α-trimmed to 6) (analyzed further in Appendix F).

Multi-modal quadrotor control. Quadrotors are safety-critical systems where robustness directly affects deployability. We consider a navigation task in which the quadrotor must reach a goal while avoiding numerous obstacles (see Figure 6), under stochastic dynamics with full state observability [3]. Details of benchmark can be found in Appendix D.2. The task is inherently multi-modal: obstacles between the quadrotor and its goal admit passage on either side, and wind can shift probability mass between such modes. The controller is an RL policy that outputs a Gaussian mixture over actions with predicted component means and mixture weights, and fixed variances [32]. The prediction of a mixture enables navigation around obstacles through either path and rejection of wind-induced disturbances. This is precisely the setting where randomized and α-smoothing fails: perturbations are averaged such that different modes are collapsed to their (weighted) mean, which often is to navigate directly into an obstacle. Clustered smoothing recovers meaningful guarantees by treating each cluster individually.

Visual inspection of policies (Figure 6) shows that the nominal policy is bi-modal, corresponding to navigating through one of two paths through the obstacles to the goal. However, the α-smoothing destroys the multi-modality and ultimately predicts the (trimmed) average, which corresponds to navigating directly at the obstacles. Instead, our framework preserves the modes and only tightens the clusters. In the simulation shown in Figure 6, the nominal policy reaches the goal in 72 of the 100 trajectories sampled and crashes in 28 trajectories. The α-smoothed policy crashes in 48 trajectories, while our clustered α-smoothed policy only crashes in 9 trajectories. The certified probability as a function of perturbation radius in Figure 6 suggests that our method is penalized more for larger input radii r, by the correction of $\underline { { p } } _ { \nu _ { m } } , \bar { p } { \nu } _ { m }$ in (4).

Note that the guarantees of the smoothed predictors are with respect to the controller output at each time step, while the benchmark operates in a receding horizon fashion. Nonetheless, the empirical results show significant benefit in applying our method compared to both the base policy and regular α-smoothing. Moreover, our approach is agnostic to the underlying predictor and is thus applicable to controllers such as MPC and motion planners such as RRT, both of which are important methods in robotic applications.

## 6 Limitations

Randomized smoothing relies on repeated inference. In certain contexts (e.g. large neural networks), this may be prohibitive, in particular when the base predictor has a non-negligible computational cost. That our method is agnostic to the clustering algorithm is both a strength and a weakness: the quality of the result depends strongly on the choice, and thus, one will need to assess the clustering to choose a suitable algorithm. Theory-wise, we identify four primary limitations. First, defining as a union of disjoint and convex sets is a key component to enabling the guarantees, but this is restrictive, e.g., if the clusters cannot be cleanly separated as convex sets such as the famous half-moon toy example. The second limitation – that certification is with respect to the smoothed predictor – is key, as one cannot apply our method to robustify the prediction if parity with the base regressor is crucial, such as physics and dynamical systems simulations. Third, in (4), if one wants to reduce the noise but maintain the probability bound, it is necessary to fix $r / \sigma ,$ and thus, r must be proportional to the reciprocal of σ. Fourth, as we have shown empirically, the lower bounds from Theorem 3.1 can become more conservative with the inclusion of new clusters. Consequently, either a lower certified radius or a larger noise level may be needed to guarantee robustness in highly multi-modal settings.

## 7 Conclusion

In this work, we addressed a key limitation of randomized smoothing for regression: the collapse of multi-modal predictions under averaging. We proposed clustered α-smoothing, which preserves multi-modality by applying α-trimmed smoothing within clusters and combining the results as a mixture. We established probabilistic robustness guarantees and demonstrated significant empirical gains, reducing the Wasserstein distance by 27% in trajectory prediction and the collision rate by 81% in quadrotor control. Several directions remain for future work. In particular, our current approach assumes a fixed partition of the output space V. However, recent work in randomized smoothing [40, 2] suggests that input-dependent smoothing distributions can improve performance. Extending our framework to allow input-dependent smoothing and partitions $\nu ( x )$ is therefore a promising and challenging direction.

## References

[1] Steven Adams, Andrea Patane, Morteza Lahijanian, and Luca Laurenti. Bnn-dp: robustness certification of bayesian neural networks via dynamic programming. In International Conference on Machine Learning, pages 133–151. PMLR, 2023.

[2] Motasem Alfarra, Adel Bibi, Philip HS Torr, and Bernard Ghanem. Data dependent randomized smoothing. In Uncertainty in Artificial Intelligence, pages 64–74. PMLR, 2022.

[3] Thom Badings, Licio Romao, Alessandro Abate, David Parker, Hasan A Poonawala, Marielle Stoelinga, and Nils Jansen. Robust control for dynamical systems with non-gaussian noise via formal abstractions. Journal ofArtificial Intelligence Research, 76:341–391, 2023.

[4] Inhwan Bae, Young-Jae Park, and Hae-Gon Jeon. Singulartrajectory: Universal trajectory predictor using diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17890–17901, 2024.

[5] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuScenes: A Multimodal Dataset for Autonomous Driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11618–11628. IEEE Computer Society, June 2020.

[6] Yipu Chen, Haotian Xue, and Yongxin Chen. Diffusion policy attacker: Crafting adversarial attacks for diffusion-based policies. Advances in Neural Information Processing Systems, 37:119614–119637, 2024.

[7] Ping-yeh Chiang, Michael Curry, Ahmed Abdelkader, Aounon Kumar, John Dickerson, and Tom Goldstein. Detection as regression: Certified object detection with median smoothing. Advances in Neural Information Processing Systems, 33:1275–1286, 2020

[8] Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized smoothing. In international conference on machine learning, pages 1310–1320. PMLR, 2019.

[9] Ruilong Deng, Gaoxi Xiao, Rongxing Lu, Hao Liang, and Athanasios V Vasilakos. False data injection on state estimation in power systems—attacks, impacts, and defense: A survey. IEEE transactions on industrial informatics, 13(2):411–423, 2016.

[10] P Kingma Diederik and Welling Max. An introduction to variational autoencoders. Foundations and Trends® in Machine Learning, 12(4):307–392, 2019.

[11] Hai Duong, ThanhVu Nguyen, and Matthew B Dwyer. Neuralsat: A high-performance verification tool for deep neural networks. In International Conference on Computer Aided Verification, pages 409–423. Springer, 2025.

[12] Andrew Foong, David Burt, Yingzhen Li, and Richard Turner. On the expressiveness of approximate inference in bayesian neural networks. Advances in Neural Information Processing Systems, 33:15897–15908, 2020.

[13] Timon Gehr, Matthew Mirman, Dana Drachsler-Cohen, Petar Tsankov, Swarat Chaudhuri, and Martin Vechev. Ai2: Safety and robustness certification of neural networks with abstract interpretation. In 2018 IEEE symposium on security and privacy (SP), pages 3–18. IEEE, 2018.

[14] Agathe Girard, Carl Rasmussen, Joaquin Q Candela, and Roderick Murray-Smith. Gaussian process priors with uncertain inputs application to multiple-step ahead time series forecasting. Advances in neural information processing systems, 15, 2002.

[15] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.

[16] Guy Katz, Clark Barrett, David L Dill, Kyle Julian, and Mykel J Kochenderfer. Reluplex: An efficient smt solver for verifying deep neural networks. In International conference on computer aided verification, pages 97–117. Springer, 2017.

[17] Mathias Lecuyer, Vaggelis Atlidakis, Roxana Geambasu, Daniel Hsu, and Suman Jana. Certified robustness to adversarial examples with differential privacy. In 2019 IEEE symposium on security and privacy (SP), pages 656–672. IEEE, 2019.

[18] Alexander Levine and Soheil Feizi. Wasserstein smoothing: Certified robustness against wasserstein adversarial attacks. In International conference on artificial intelligence and statistics, pages 3938–3947. PMLR, 2020.

[19] Bai Li, Changyou Chen, Wenlin Wang, and Lawrence Carin. Certified adversarial robustness with additive noise. Advances in neural information processing systems, 32, 2019.

[20] Xuanqing Liu, Minhao Cheng, Huan Zhang, and Cho-Jui Hsieh. Towards robust neural networks via random selfensemble. In Proceedings ofthe european conference on computer vision (ECCV), pages 369–385, 2018.

[21] Yulong Lu and Jianfeng Lu. A universal approximation theorem of deep neural networks for expressing probabilit distributions. Advances in neural information processing systems, 33:3094–3105, 2020.

[22] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learnin models resistant to adversarial attacks. In 6th International Conference on Learning Representations, ICLR, 2018.

[23] Charles Meyers, Tommy Löfstedt, and Erik Elmroth. Safety-critical computer vision: an empirical survey of adversarial evasion attacks and defenses on computer vision systems. Artificial Intelligence Review, 56(Suppl 1):217–251, 2023.

[24] Rhiannon Michelmore, Matthew Wicker, Luca Laurenti, Luca Cardelli, Yarin Gal, and Marta Kwiatkowska. Uncertainty quantification with statistical guarantees in end-to-end autonomous driving control. In 2020 IEEE international conference on robotics and automation (ICRA), pages 7344–7350. IEEE, 2020.

[25] Victor Milenkovic. Robust construction of the voronoi diagram of a polyhedron. In Canadian Conference on Computa tional Geometry, volume 93, pages 473–478, 1993.

[26] Anna Mészáros, Julian F. Schumann, Javier Alonso-Mora, Arkady Zgonnikov, and Jens Kober. TrajFlow: Learning Distributions over Trajectories for Human Behavior Prediction. In 2024 IEEE Intelligent Vehicles Symposium (IV), Jeju, June 2024.

[27] Nigamaa Nayakanti, Rami Al-Rfou, Aurick Zhou, Kratarth Goel, Khaled S. Refaat, and Benjamin Sapp. Wayformer: Motion Forecasting via Simple & Efficient Attention Networks. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 2980–2987, May 2023.

[28] George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. Journal ofMachine Learning Research, 22(57):1–64, 2021.

[29] Zeyu Qin, Yanbo Fan, Yi Liu, Li Shen, Yong Zhang, Jue Wang, and Baoyuan Wu. Boosting the transferability of adversarial attacks with reverse adversarial perturbation. Advances in neural information processing systems, 35:29845– 29858, 2022.

[30] Aref Rekavandi, Farhad Farokhi, Olga Ohrimenko, and Benjamin Rubinstein. Certified adversarial robustness via randomized α-smoothing for regression models. Advances in Neural Information Processing Systems, 37:134127– 134150, 2024.

[31] Aref Miri Rekavandi, Olga Ohrimenko, and Benjamin IP Rubinstein. Rs-reg: Probabilistic and robust certified regressio through randomized smoothing. Transactions on Machine Learning Research, 2025.

[32] Jie Ren, Yewen Li, Zihan Ding, Wei Pan, and Hao Dong. Probabilistic mixture-of-experts for efficient deep reinforcement learning. arXiv preprint arXiv:2104.09122, 2021.

[33] Andy Rivas, Gregory Kyriakos Delipei, and Jason Hou. Predictions of component remaining useful lifetime using bayesian neural network. Progress in Nuclear Energy, 146:104143, 2022.

[34] Walter Rudin. Real and complex analysis, 1987.

[35] Hadi Salman, Jerry Li, Ilya Razenshteyn, Pengchuan Zhang, Huan Zhang, Sebastien Bubeck, and Greg Yang. Provably robust deep learning via adversarially trained smoothed classifiers. Advances in neural information processing systems, 32, 2019.

[36] Julian F Schumann, Eduardo Figueiredo, Frederik Baymler Mathiesen, Luca Laurenti, Jens Kober, and Arkady Zgonnikov. Evaluating randomized smoothing as a defense against adversarial attacks in trajectory prediction. arXiv preprint arXiv:2603.10821, 2026.

[37] Julian F. Schumann, Jeroen Hagenus, Frederik Baymler Mathiesen, and Arkady Zgonnikov. Realistic Adversarial Attacks for Robustness Evaluation of Trajectory Prediction Models via Future State Perturbation. ACM Journal on Autonomous Transportation Systems, 2026.

[38] Julian F Schumann, Anna Mészáros, Jens Kober, and Arkady Zgonnikov. STEP: Structured training and evaluation platform for benchmarking trajectory prediction models. arXiv preprint arXiv:2509.14801, 2025.

[39] Guangzhi Su, Shuchang Huang, Yutong Ke, Zhuohang Liu, Long Qian, and Kaizhu Huang. Smoothguard: Defending multimodal large language models with noise perturbation and clustering aggregation. arXiv preprint arXiv:2510.26830, 2025.

[40] Peter Sukenik, Aleksei Kuvshinov, and Stephan Günnemann. Intriguing properties of input-dependent randomized smoothing. In International Conference on Machine Learning, 2021.

[41] Hoang-Dung Tran, Xiaodong Yang, Diego Manzanas Lopez, Patrick Musau, Luan Viet Nguyen, Weiming Xiang, Stanley Bak, and Taylor T Johnson. Nnv: the neural network verification tool for deep neural networks and learning-enabled cyber-physical systems. In International conference on computer aided verification, pages 3–17. Springer, 2020.

[42] Zixia Wang, Gaojie Jin, Jia Hu, and Ronghui Mu. Clucert: Certifying llm robustness via clustering-guided denoising smoothing. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 37998–38006, 2026.

[43] Anqi Wu, Sebastian Nowozin, Edward Meeds, Richard E. Turner, José Miguel Hernández-Lobato, and Alexander L. Gaunt. Deterministic variational inference for robust bayesian neural networks. In 7th International Conference on Learning Representations, ICLR, 2019.

[44] Haoze Wu, Omri Isac, Aleksandar Zeljic, Teruhiro Tagomori, Matthew Daggitt, Wen Kokke, Idan Refaeli, Guy Amir,´ Kyle Julian, Shahaf Bassan, et al. Marabou 2.0: a versatile formal analyzer of neural networks. In International Conference on Computer Aided Verification, pages 249–264. Springer, 2024.

[45] Kaidi Xu, Zhouxing Shi, Huan Zhang, Yihan Wang, Kai-Wei Chang, Minlie Huang, Bhavya Kailkhura, Xue Lin, and Cho-Jui Hsieh. Automatic perturbation analysis for scalable certified robustness and beyond. Advances in Neural Information Processing Systems, 33:1129–1141, 2020.

[46] Greg Yang, Tony Duan, J Edward Hu, Hadi Salman, Ilya Razenshteyn, and Jerry Li. Randomized smoothing of all shapes and sizes. In International conference on machine learning, pages 10693–10705. PMLR, 2020.

[47] Arkady Zgonnikov, David Abbink, and Gustav Markkula. Should I Stay or Should I Go? Cognitive Modeling of Left-Turn Gap Acceptance Decisions in Human Drivers. Human Factors: The Journal of the Human Factors and Ergonomics Society, 66(5):1399–1413, May 2024.

[48] Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh, and Luca Daniel. Efficient neural network robustnes certification with general activation functions. Advances in neural information processing systems, 31, 2018.

[49] Liangliang Zhuang, Ancha Xu, and Xiao-Lin Wang. A prognostic driven predictive maintenance framework based on bayesian deep learning. Reliability Engineering & System Safety, 234:109181, 2023.

## A Technical Proofs

## A.1 Proof of Theorem 3.1

As Theorem 3.1 is built from a union-bound argument over $\mathcal { R } _ { m } ,$ we start by introducing the following Lemma A.1, which bounds the probability $\mathbb { P } _ { ( \mathbf { w } , \pmb { \varepsilon } _ { 1 } , . . . , \pmb { \varepsilon } _ { N } ) } \left( \tilde { \mathcal { H } } _ { N , \alpha , \pmb { \nu } } ( x + \delta ) \in \mathcal { R } _ { m } \right)$ . For notational convenience, we write ε to mean $\varepsilon _ { 1 } , \ldots , \varepsilon _ { N }$ for the remainder of this appendix.

Lemma A.1. Let $\pmb { \nu } = \{ \nu _ { 1 } , \dots , \nu _ { M } \}$ be a partition of the space . Assume that $\begin{array} { l } { { \check { p } } { \nu _ { m } } } \\ { { t } } \end{array} \overset { \leq } { \mathcal { R } } _ { m } \overset { \leq } { \subset }$ $\mathbb { P } _ { ( \mathbf { w } , \varepsilon ) } \left( h _ { \mathbf { w } } ( x + \varepsilon ) \in \mathcal { V } _ { m } \right) \leq \hat { p } _ { \mathcal { V } _ { m } }$ and $\begin{array} { r } { \mathbb { P } _ { ( \mathbf { w } , \varepsilon ) } \left( h _ { \mathbf { w } } ( x + \varepsilon ) \in \mathcal { R } _ { m } \right) \geq \breve { p } _ { \mathcal { R } _ { m } } f o r } \end{array}$ a convex se $\mathcal { V } _ { m } . \ F o r \ r \ge 0 , d e f i n e$

$$
\underline { { p } } _ { \mathcal { V } _ { m } } = \Phi \left( \Phi ^ { - 1 } ( \breve { p } _ { \mathcal { V } _ { m } } ) - \frac { r } { \sigma } \right) , \ : \ : \bar { p } _ { \mathcal { V } _ { m } } = \Phi \left( \Phi ^ { - 1 } ( \hat { p } _ { \mathcal { V } _ { m } } ) + \frac { r } { \sigma } \right) , \ : \ : \ : a n d \ : \underline { { p } } _ { \mathcal { R } _ { m } } = \Phi \left( \Phi ^ { - 1 } ( \breve { p } _ { \mathcal { R } _ { m } } ) - \frac { r } { \sigma } \right) .\tag{11}
$$

Then, for any $\delta \in \mathbb { R } ^ { d }$ such that $\| \delta \| _ { 2 } \leq r ,$ it holds that

$$
\begin{array} { r l } & { \mathbb { P } _ { ( \mathbf { w } , \underline { { \varepsilon } } ) } \left( \mathcal { \tilde { H } } _ { N , \alpha , \mathcal { V } } ( x + \delta ) \in \mathcal { R } _ { m } \right) } \\ & { \qquad \geq \displaystyle \sum _ { s = 1 } ^ { N } \frac { s } { N } \sum _ { j = s - \lfloor \alpha s \rfloor } ^ { s } f _ { s , j } \left( \frac { p _ { \mathcal { R } _ { m } } } { \bar { p } \mathcal { V } _ { m } } \right) \operatorname* { m i n } \left\{ f _ { N , s } \left( \underline { { p } } _ { \mathcal { V } _ { m } } \right) , f _ { N , s } \left( \bar { p } \mathcal { V } _ { m } \right) \right\} . } \end{array}\tag{12}
$$

Proof. First, we observe that using the Neyman-Pearson’s Lemma [8], for any $\delta \in \mathbb { R } ^ { d }$ such that $\| \delta \| _ { 2 } \le r$

$$
\mathbb { P } _ { ( \mathbf { w } , \varepsilon ) } \left( h _ { \mathbf { w } } ( x + \delta + \varepsilon ) \in \mathcal { V } _ { m } \right) \geq \Phi \left( \Phi ^ { - 1 } ( \tilde { p } \nu _ { m } ) - \frac { \| \delta \| _ { 2 } } { \sigma } \right) \geq \Phi \left( \Phi ^ { - 1 } ( \tilde { p } \nu _ { m } ) + \frac { r } { \sigma } \right) = \underline { { p } } _ { \nu _ { m } } ,\tag{13}
$$

$$
\mathbb { P } _ { ( \mathbf { w } , \varepsilon ) } \left( h _ { \mathbf { w } } ( x + \delta + \varepsilon ) \in \mathcal { V } _ { m } \right) \leq \Phi \left( \Phi ^ { - 1 } ( \hat { p } _ { \mathcal { V } _ { m } } ) + \frac { \| \delta \| _ { 2 } } { \sigma } \right) \leq \Phi \left( \Phi ^ { - 1 } ( \hat { p } _ { \mathcal { V } _ { m } } ) + \frac { r } { \sigma } \right) = \bar { p } _ { \mathcal { V } _ { m } } ,\tag{14}
$$

and

$$
\begin{array} { r } { \mathbb { P } _ { ( \mathbf { w } , \varepsilon ) } \left( h _ { \mathbf { w } } ( x + \delta + \varepsilon ) \in \mathcal { R } _ { m } \right) \geq \Phi \left( \Phi ^ { - 1 } ( \tilde { p } _ { \mathcal { R } _ { m } } ) - \frac { \left\| \delta \right\| _ { 2 } } { \sigma } \right) \geq \Phi \left( \Phi ^ { - 1 } ( \tilde { p } _ { \mathcal { R } _ { m } } ) - \frac { r } { \sigma } \right) = \underline { { p } } _ { \mathcal { R } _ { m } } , } \end{array}\tag{15}
$$

due to the monotonicity of the function Φ. Further, because by construction $\mathcal { R } _ { m } \subset \mathcal { V } _ { m }$ , we have that

$$
\mathbb { P } _ { ( \mathbf { w } , \varepsilon ) } \left( h _ { \mathbf { w } } ( x + \delta + \varepsilon ) \in \mathcal { R } _ { m } \mid h _ { \mathbf { w } } ( x + \delta + \varepsilon ) \in \mathcal { V } _ { m } \right) \geq \frac { \underline { { p } } _ { \mathcal { R } _ { m } } } { \bar { p } \nu _ { m } } .\tag{16}
$$

Having these probability bounds for $h _ { \mathbf { w } } ( x + \delta + \varepsilon )$ , we can now proceed to study the probability of the event $\tilde { \mathcal { H } } _ { N , \alpha , \nu } ( x + \delta ) \in \mathcal { R } _ { m }$ . For any $x ^ { \prime } \in \mathcal { X }$ , by applying the Law of Total Probability twice, it holds that

$$
\begin{array} { r l } & { \mathbb { P } _ { ( \mathbf { w } , \mathbf { g } ) } \left( \tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } } ( x ^ { \prime } ) \in \mathcal { R } _ { m } \right) = } \\ & { \mathbb { P } _ { ( \mathbf { w } , \mathbf { g } ) } \left( \tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } } ( x ^ { \prime } ) \in \mathcal { R } _ { m } \mid \tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } _ { m } } ( x ^ { \prime } ) \in \mathcal { R } _ { m } \right) \mathbb { P } _ { ( \mathbf { w } , \mathbf { g } ) } \left( \tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } _ { m } } ( x ^ { \prime } ) \in \mathcal { R } _ { m } \right) + } \\ & { \qquad \underbrace { \mathbb { P } _ { ( \mathbf { w } , \mathbf { g } ) } \left( \tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } } ( x ^ { \prime } ) \in \mathcal { R } _ { m } \mid \tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } _ { m } } ( x ^ { \prime } ) \notin \mathcal { R } _ { m } \right) } _ { = 0 } \mathbb { P } _ { ( \mathbf { w } , \mathbf { g } ) } \left( \tilde { \mathcal { H } } _ { N , \alpha } , { \mathcal { V } _ { m } } ( x ^ { \prime } ) \notin \mathcal { R } _ { m } \right) = } \end{array}
$$

$$
\begin{array} { r l } & { \displaystyle \sum _ { s = 0 } ^ { N } \mathbb { P } _ { ( \mathbf { w } , \underline { { \varepsilon } } ) } \left( \tilde { \mathcal { H } } _ { N , \alpha , \pmb { \nu } } ( x ^ { \prime } ) \in \mathcal { R } _ { m } \mid \tilde { \mathcal { H } } _ { N , \alpha , \gamma _ { m } } ( x ^ { \prime } ) \in \mathcal { R } _ { m } , \lvert I _ { m } \rvert = s \right) . } \\ & { \quad \quad \quad \mathbb { P } _ { ( \mathbf { w } , \underline { { \varepsilon } } ) } \left( \tilde { \mathcal { H } } _ { N , \alpha , \gamma _ { m } } ( x ^ { \prime } ) \in \mathcal { R } _ { m } \mid \lvert I _ { m } \rvert = s \right) \mathbb { P } _ { ( \mathbf { w } , \underline { { \varepsilon } } ) } ( | I _ { m } | = s ) . } \end{array}\tag{17}
$$

In particular, let $x ^ { \prime } = x + \delta$ . We will examine each of these terms separately. Recall that, by construction, the random set of indexes $I _ { m }$ is given by $I _ { m } = \{ i \in [ N ] : h _ { \mathbf { w } } ( x + \bar { \delta } + \varepsilon _ { i } ) \in \mathcal { V } _ { m } \}$ , so that $| I _ { j } | \sim$ Binomial $\left( N , \mathbb { P } _ { ( \mathbf { w } , \pmb \varepsilon ) } \left( h _ { \mathbf { w } } ( x + \delta + \pmb \varepsilon ) \in \mathcal { V } _ { m } \right) \right)$ . Therefore,

$$
\begin{array} { r } { \mathbb { P } _ { ( \mathbf { w } , \underline { { \varepsilon } } ) } ( | I _ { m } | = s ) \geq \operatorname* { m i n } \left\{ f _ { N , s } \left( \underline { { p } } _ { \mathcal { V } _ { m } } \right) , f _ { N , s } \left( \bar { p } _ { \mathcal { V } _ { m } } \right) \right\} , } \end{array}\tag{18}
$$

as $\mathbb { P } _ { ( \mathbf { w } , \epsilon ) } \left( h _ { \mathbf { w } } ( x + \delta + \varepsilon ) \in \mathcal { V } _ { m } \right) \in \left[ \underline { { p } } _ { \nu _ { m } } , \bar { p } \nu _ { m } \right]$ by (13) and (14). That the minimum is attained at $\underline { { p } } _ { v _ { m } }$ or $\bar { p } _ { \nu _ { m } }$ follows from the fact that the continuous function $\begin{array} { r } { f _ { N , s } ( p ) = \binom { N } { s } p ^ { s } ( 1 - p ) ^ { N - s } } \end{array}$ has a positive derivative $f ^ { \prime } ( p ) > 0$ for $\begin{array} { r } { p < \frac { s } { N } } \end{array}$ (resp. $f ^ { \prime } ( p ) < 0$ for $\begin{array} { r } { p > \frac { s } { N } ) } \end{array}$ . Hence, $\begin{array} { r } { p \ = \ \frac { s } { N } } \end{array}$ is the maximizer within $[ 0 , \dot { 1 } ]$ and the minimum of $f ( \boldsymbol p )$ for $p \in [ \underline { { p } } , \bar { p } ]$ must be attained at either $p = \underline { { p } }$ or $p = { \bar { p } } .$

We then move on to the event $\tilde { \mathcal { H } } _ { N , \alpha , \pmb { \nu } } ( \boldsymbol { x } + \delta ) \in \mathcal { R } _ { m }$ given $\tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } _ { m } } ( x + \delta ) \in \mathcal { R } _ { m } , | I _ { m } | = s$ Trivially, conditioned on $\tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } _ { m } } ( x + \delta ) \in \mathcal { R } _ { m }$ , the event $\tilde { \mathcal { H } } _ { N , \alpha , \nu } ( x + \delta ) \in \mathcal { R } _ { m }$ takes place if the component $\tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } _ { m } }$ is sampled from the mixture, which by construction is equal to $\frac { \left| I _ { m } \right| } { N }$ . Thus,

$$
\mathbb { P } _ { ( \mathbf { w } , \mathbf { \Xi } \mathbf { \Xi } ) } \left( \tilde { \mathcal { H } } _ { N , \alpha , \gamma } ( x + \delta ) \in \mathcal { R } _ { m } \mid \tilde { \mathcal { H } } _ { N , \alpha , \gamma _ { m } } ( x + \delta ) \in \mathcal { R } _ { m } , | I _ { m } | = s \right) = \frac { s } { N } .\tag{19}
$$

Finally, to conclude, we need to lower-bound the probability of the event $\tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } _ { m } } ( x + \delta ) \in \mathcal { R } _ { m }$ given $| I _ { m } | = s$ . We observe that conditioned on the knowledge $| I _ { m } | = s ,$ , the random variable $\tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } _ { m } } ( x +$ δ) conditioned on $| I _ { m } | = s$ behaves as the α-smoother introduced in [30] with number of samples s. Therefore, we follow a similar proof. Let $\begin{array} { r } { Z _ { m } = \sum _ { i = 1 } ^ { s } \mathbf { 1 } _ { h _ { \mathbf { w } } ( x + \delta + \varepsilon _ { i } ) \in \mathcal { R } _ { m } | h _ { \mathbf { w } } ( x + \delta + \varepsilon _ { i } ) \in \mathcal { V } _ { m } } } \end{array}$ , thus

$$
Z _ { m } \sim \mathrm { B i n o m i a l } \left( s , \mathbb { P } _ { ( \mathbf { w } , \varepsilon ) } \left( h _ { \mathbf { w } } ( x + \delta + \varepsilon ) \in \mathcal { R } _ { m } \mid h _ { \mathbf { w } } ( x + \delta + \varepsilon ) \in \mathcal { V } _ { m } \right) \right) .\tag{20}
$$

From the Law of Total Probability, it holds that

$$
\begin{array} { r l } & { \mathbb { P } _ { ( \mathbf { w } , \varepsilon ) } \left( \mathcal { \tilde { H } } _ { N , \alpha , \gamma _ { m } } ( x + \delta ) \in \mathcal { R } _ { m } \middle \vert | I _ { m } | = s \right) = } \\ & { \qquad \displaystyle \sum _ { z = 0 } ^ { s } \mathbb { P } _ { ( \mathbf { w } , \varepsilon ) } \left( \mathcal { \tilde { H } } _ { N , \alpha , \gamma _ { m } } ( x + \delta ) \in \mathcal { R } _ { m } \middle \vert | I _ { m } | = s , Z _ { m } = z \right) \mathbb { P } ( Z _ { m } = z ) , } \end{array}\tag{21}
$$

Now, note that the event $z \ge s - \lfloor \alpha s \rfloor$ means that the α-trimming only discards points out of $\mathcal { R } _ { m }$ (i.e., the average in (2) only contains points in $\mathcal { R } _ { m } )$ , thus it must hold that $\tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } _ { m } } ( x + \delta ) \in \mathcal { R } _ { m }$ conditioned on the given events by the convexity of the set $\mathcal { R } _ { m }$ . On the other hand, when $z \ <$ $N - \lfloor \alpha s \rfloor$ , the average in (2) contains points out of $\mathcal { R } _ { m }$ . In the absence of more information about how far those points are, we have no guarantees that the average over them will belong to $\mathcal { R } _ { m }$ (as only one point suffices to displace the average arbitrarily far from $\mathcal { R } _ { m } \left[ 3 1 \right] )$ . Therefore, conditioned on these given events, we may conservatively consider tha $\tilde { \mathcal { H } } _ { N , \alpha , \mathcal { V } _ { m } } ( x + \delta ) \notin \mathcal { R } _ { m }$ . Thus, from (21),

$$
\begin{array} { r l } & { \mathbb { P } _ { ( \mathbb { N } , \alpha ) , \gamma _ { m } } ( \frac { 1 } { H _ { N , \alpha } } \middle | \mathcal { F } _ { m } ( x + \delta ) \in \mathcal { R } _ { m } \middle | | I _ { m } | = s ) = } \\ & { \qquad \underset {  \sum = - 1 } { s - 1 \alpha } \mathbb { P } _ { ( \mathbb { N } , \alpha ) } ( \tilde { \mathcal { H } } _ { N , \alpha , \gamma _ { m } } ( x + \delta ) \in \mathcal { R } _ { m } \middle | | I _ { m } | = s , Z _ { m } = z ) \mathbb { P } _ { Z _ { m } } ( Z _ { m } = z ) + } \\ & { \qquad \underset {  \sum = - 1 } { s - 1 \alpha } \mathbb { P } _ { ( \mathbb { N } , \alpha ) , \gamma _ { m } } ( \tilde { \mathcal { H } } _ { N , \alpha , \gamma _ { m } } ( x + \delta ) \in \mathcal { R } _ { m } \middle | | I _ { m } | = s , Z _ { m } = z ) ) \mathbb { E } _ { Z _ { \alpha } } ( Z _ { m } = z ) } \\ & { \qquad \quad \sum _ { z = - 1 , \alpha + 1 } ^ { s } \mathbb { P } _ { Z _ { m } } ( Z _ { m } = z ) } \\ & { \geq \underset { \alpha = - s - 1 , \alpha + 1 } { \overset { s } { \sum } } \mathbb { P } _ { Z _ { m } } ( Z _ { m } = z ) } \\ & { \geq \underset { \delta = - 1 , \alpha + 1 } { \overset { s } { \sum } } \int _ { \alpha , \delta } ( \frac { P _ { \mathcal { R } _ { m } } } { \tilde { \mathcal { W } } _ { \gamma _ { m } } } ) , } \end{array}\tag{22}
$$

from the fact that $Z _ { m }$ is distributed as the binomial in (20) and its success probability is lower bounded by (16). The proof is concluded by replacing (18), (19), and (22) in (17). □

We are now ready to prove Theorem 3.1.

Proof. By assumption, we have $\mathcal { R } _ { l } \cap \mathcal { R } _ { l ^ { \prime } } = \emptyset$ for $l , l ^ { \prime } \in \mathcal { L }$ where ${ \mathit { l } } \neq { \mathit { l } } ^ { \prime }$ . Therefore, we can employ a union-bound argument over Lemma A.1 for the following bound

$$
\begin{array} { r l } {  { \mathbb { P } _ { ( \mathbf { w } , \mathbf { g } ) } ( \tilde { \mathcal { H } } _ { N , \alpha , \nu } ( x + \delta ) \in \tilde { \mathcal { R } } ) } \quad } & { } \\ & { = \sum _ { l \in \mathcal { L } } \mathbb { P } _ { ( \mathbf { w } , \mathbf { g } ) } ( \tilde { \mathcal { H } } _ { N , \alpha , \nu } ( x + \delta ) \in \mathcal { R } _ { l } ) } \\ & { \geq \sum _ { l \in \mathcal { L } } \operatorname* { i n f } _ { p _ { \nu _ { l } } \in [ p _ { \nu _ { l } } , \tilde { p } _ { \nu _ { l } } ] } \sum _ { s = 1 } ^ { N } \frac { s } { N } \sum _ { j = s - \lfloor \alpha s \rfloor } ^ { s } f _ { s , j } ( \frac { \underline { { p } } _ { \mathcal { R } _ { l } } } { p _ { \nu _ { l } } } ) f _ { N , s } ( p _ { \nu _ { l } } ) } \\ & { = \operatorname* { i n f } _ { p _ { \nu _ { m } } \in [ p _ { \nu _ { m } } , \tilde { p } _ { \nu _ { m } } ] } \sum _ { l \in \mathcal { L } } \sum _ { s = 1 } ^ { N } \frac { s } { N } \sum _ { j = s - \lfloor \alpha s \rfloor } ^ { s } ( \frac { p _ { \mathcal { R } _ { l } } } { p _ { \nu _ { l } } } ) f _ { N , s } ( p _ { \nu _ { l } } ) , } \end{array}\tag{23}
$$

where the last infimum is for all $m \in [ M ]$ . Naturally, one would compute this infimum independently per $l \in \mathcal L$ . However, extracting it reveals that we can add the constraint $\begin{array} { r } { \sum _ { m = 1 } ^ { M } p _ { \nu _ { m } } = 1 } \end{array}$ , i.e., ensure that the probability over all partition cells $\nu _ { m }$ sum to one. Thus, we arrive at

$$
\begin{array} { r l } & { \mathbb { P } _ { ( \mathbf { w } , \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } ) } \left( \mathcal { \tilde { H } } _ { N , \alpha , \mathbf { \Xi } \mathbf { \mathcal { V } } } ( x + \delta ) \in \mathcal { \tilde { R } } \right) } \\ & { \qquad \quad \overset { \mathrm { i n f } } { \geq } \underset { p _ { \mathcal { V } _ { m } } \in [ \underline { { p } } _ { \mathcal { V } _ { m } } , \overline { { p } } _ { \mathcal { V } _ { m } } ] } { \operatorname* { i n f } } \underset { l \in \mathcal { L } } { \sum } \underset { s = 1 } { \overset { N } { \sum } } \frac { s } { N } \underset { j = s - \lfloor \alpha s \rfloor } { \sum ^ { s } } f _ { s , j } \left( \frac { p _ { \mathcal { R } _ { l } } } { p _ { \mathcal { V } _ { l } } } \right) f _ { N , s } \left( p _ { \mathcal { V } _ { l } } \right) , } \end{array}\tag{24}
$$

which concludes the proof.

## A.2 Proof of Proposition 3.2

Before proving Proposition 3.2, we first prove an auxiliary Lemma.

Lemma A.2. Let $g : [ \ell , u ] \to$ R be a Lipschitz continuousfunction with Lipschitz constant $\mathcal { L } _ { g }$ . Let $\{ p ^ { ( 1 ) } , \ldots , p ^ { ( K ) } \}$ be a set of $K \in \mathbb { N } _ { > 0 }$ anchor points such that $\ell = p ^ { ( 1 ) } < p ^ { ( 2 ) } < \dots < p ^ { ( K ) } = u$ Let conv $\because ( g ) ( p )$ be defined as

$$
\mathrm { c o n v } _ { K } ( g ) ( p ) = \operatorname* { m i n } _ { \lambda \in \Delta _ { K } } \left\{ \sum _ { k = 1 } ^ { K } \lambda ^ { ( k ) } g ( p ^ { ( k ) } ) s . t . \sum _ { k = 1 } ^ { K } \lambda ^ { ( k ) } p ^ { ( k ) } = p \right\} .\tag{25}
$$

Then,

$$
\mathrm { c o n v } _ { K } ( g ) ( p ) \leq g ( p ) + { \mathcal L } _ { g } h _ { K } ,\tag{26}
$$

$$
\begin{array} { r } { w h e r e \ : h _ { K } = \operatorname* { m a x } _ { p ^ { \prime } \in [ \ell , u ] } \operatorname* { m i n } _ { k \in [ K ] } | p ^ { \prime } - p ^ { ( k ) } | . } \end{array}
$$

Proof. For any $p \in [ \ell , u ]$ , there exist consecutive anchor points $p ^ { ( k ) } , p ^ { ( k + 1 ) }$ such that $p ^ { ( k ) } \leq p \leq$ $p ^ { ( k + 1 ) }$ . We can write p as a convex combination of those anchor points, i.e. $p = \lambda p ^ { ( k ) } + ( 1 - \lambda ) p ^ { ( k + 1 ) }$ by taking $\begin{array} { r } { \lambda = \frac { p ^ { ( k + 1 ) } - p } { p ^ { ( k + 1 ) } - p ^ { ( k ) } } \in [ 0 , 1 ] } \end{array}$ . Thus, the vector $\pmb { \lambda } = [ 0 , \ldots , 0 , \lambda , 1 - \lambda , 0 , \ldots , 0 ] \ ( \mathrm { i . e }$ ., non-zero at positions k and $k + 1 )$ belongs to $\Delta _ { K }$ and satisfies the constraint in (25), thus

$$
\operatorname { c o n v } _ { K } ( g ) ( p ) \leq \lambda g ( p ^ { ( k ) } ) + ( 1 - \lambda ) g ( p ^ { ( k + 1 ) } ) .\tag{27}
$$

Let $h _ { K } = \mathrm { m a x } _ { p \in [ \ell , u ] } \mathrm { m i n } _ { k \in [ K ] } | p - p ^ { ( k ) } |$ . Then,

$$
\begin{array} { r l r } {  { \lambda g ( p ^ { ( k ) } ) + ( 1 - \lambda ) g ( p ^ { ( k + 1 ) } ) \le \lambda g ( p ) + ( 1 - \lambda ) g ( p ) + \lambda | g ( p ^ { ( k ) } ) - g ( p ) | + ( 1 - \lambda ) | g ( p ^ { ( k + 1 ) } ) - g ( p ) | } } \\ & { } & { \le g ( p ) + L _ { g } h _ { K } . } \end{array}
$$

where $L _ { g }$ is the Lipschitz constant of $g$ in $[ \ell , u ]$ . Thus, from (27) and (28), we have that, for any $p \in [ \ell , u ]$

$$
\operatorname { c o n v } _ { K } ( g ) ( p ) \leq g ( p ) + L _ { g } h _ { K } .
$$

We are now ready to prove Proposition 3.2.

Proof. The goal of Proposition 3.2 is to bound the following optimization problem.

$$
\begin{array} { r l } & { \underset { p v _ { m } \in [ \underline { { p } } _ { \mathcal { V } _ { m } } , \overline { { p } } v _ { m } ] } { \operatorname* { i n f } } \sum _ { l \in \mathcal { L } } \sum _ { s = 1 } ^ { N } \frac { s } { N } \displaystyle \sum _ { j = s - \lfloor \alpha s \rfloor } ^ { s } f _ { s , j } \left( \frac { \underline { { p } } _ { { \mathcal R } _ { l } } } { p \gamma _ { l } } \right) f _ { N , s } \left( p \gamma _ { l } \right) . } \end{array}\tag{29}
$$

Due to the non-convexity of (29), we will search for a tractable convex lower-bound. First, note that the objective function is separable in $p _ { \nu _ { l } }$ as a sum of functions g , which allows the objective in (29) to be written as $\begin{array} { r } { G ( \pmb { p } ) = \dot { \sum } _ { l \in \mathcal { L } } g _ { l } ( \tilde { p _ { \mathcal { V } _ { l } } } ) } \end{array}$ , where $\pmb { p } = \{ p \nu _ { l } \} _ { l \in \mathcal { L } }$ . For each component l, select K anchor points:

$$
p _ { \mathcal { V } _ { l } } ^ { ( 1 ) } < p _ { \mathcal { V } _ { l } } ^ { ( 2 ) } < \cdots < p _ { \mathcal { V } _ { l } } ^ { ( K ) } , \quad p _ { \mathcal { V } _ { l } } ^ { ( k ) } \in [ \underline { { p } } _ { \mathcal { V } _ { l } } , \bar { p } _ { \mathcal { V } _ { l } } ] ,\tag{30}
$$

and evaluate $g _ { l } ^ { ( k ) } : = g _ { l } ( p _ { \mathcal { V } _ { l } } ^ { ( k ) } )$ ) at each anchor. Then, let

$$
\mathrm { c o n v } _ { K } ( g _ { l } ) ( p ) = \operatorname* { m i n } _ { \lambda \in \Delta _ { K } } \left\{ \sum _ { k = 1 } ^ { K } \lambda ^ { ( k ) } g _ { l } ( p ^ { ( k ) } ) \mathrm { s . t . } \sum _ { k = 1 } ^ { K } \lambda ^ { ( k ) } p ^ { ( k ) } = p \right\} .\tag{31}
$$

From Lemma A.2, it holds that

$$
\mathrm { c o n v } _ { K } ( g _ { l } ) ( p _ { \mathcal { V } _ { l } } ) \leq g _ { l } ( p _ { \mathcal { V } _ { l } } ) + \mathcal { L } _ { g _ { l } } h _ { l , K } ,\tag{32}
$$

where $L _ { g \imath }$ is the Lipschitz constant of the function $g _ { l }$ in the interval $[ \underline { { p } } _ { \nu _ { l } } , \bar { p } \nu _ { l } ]$ and the maximum interval size $h _ { l , K } = \operatorname* { m a x } _ { p \in [ \underline { { p } } _ { \mathcal { V } _ { l } } , \bar { p } \nu _ { l } ] } \operatorname* { m i n } _ { k \in [ K ] } | p - p _ { \mathcal { V } _ { l } } ^ { ( k ) } |$ . Therefore,

$$
\begin{array} { r } { \underset { p v _ { m } \in [ \underline { { p } } _ { \mathcal { V } _ { m } } , \overline { { p } } \nu _ { m } ] } { \operatorname* { i n f } } \sum _ { l \in \mathcal { L } } \mathrm { c o n v } _ { K } ( g _ { l } ) ( p \nu _ { l } ) \leq \underset { p \nu _ { m } \in [ \underline { { p } } _ { \mathcal { V } _ { m } } , \overline { { p } } \nu _ { m } ] } { \operatorname* { i n f } } \sum _ { l \in \mathcal { L } } g _ { l } ( p \nu _ { l } ) + \sum _ { l \in \mathcal { L } } L _ { g _ { l } } h _ { l , K } . } \end{array}\tag{33}
$$

As the λ variables within the definition of each conv $ \cdot _ { K } ( g _ { l } ) ( p \nu _ { l } )$ are independent from each other, we can write the left-hand side term as the following LP problem

$$
\operatorname* { i n f } _ { \substack { p \nu _ { m } \in [ \underline { { p } } _ { \nu _ { m } } , \bar { p } \nu _ { m } ] } } \sum _ { l \in \mathcal { L } } \mathrm { c o n v } _ { K } ( g _ { l } ) ( p \nu _ { l } ) = \operatorname* { i n f } _ { \substack { p , \lambda _ { 1 } , \ldots , \lambda _ { | \mathcal { L } | } } } \sum _ { l \in \mathcal { L } } \sum _ { k = 1 } ^ { K } \lambda _ { l } ^ { ( k ) } g _ { l } ( p _ { \nu _ { l } } ^ { ( k ) } )\tag{34}
$$

$$
\begin{array} { r l } { \mathrm { s . t . } } & { \left\{ \begin{array} { l } { p \nu _ { m } \in [ \underline { { p } } _ { \mathcal { V } _ { m } } , \overline { { p } } \nu _ { m } ] , } \\ { \displaystyle \sum _ { m = 1 } ^ { M } p \nu _ { m } = 1 , } \\ { \displaystyle \sum _ { k = 1 } ^ { K } \lambda _ { l } ^ { ( k ) } p _ { \mathcal { V } _ { l } } ^ { ( k ) } = p \nu _ { l } , } \\ { \displaystyle \lambda _ { l } \in \Delta _ { K } . } \end{array} \right. } \end{array}\tag{35}
$$

To conclude, we apply Lemma A.3 to obtain

$$
L _ { g _ { l } } \le \frac { N } { 1 - \bar { p } _ { \mathcal { V } _ { l } } } + \frac { N } { \underline { { p } } _ { \mathcal { V } _ { l } } - \underline { { p } } _ { \mathcal { R } _ { l } } } .
$$

Lemma A.3. Let $g : ( 0 , 1 ] \to \mathbb { R }$ be the function defined by

$$
g ( p ) = \sum _ { s = 1 } ^ { N } \frac { s } { N } \sum _ { \substack { j = s - \lvert \alpha s \rvert } } ^ { s } \binom { s } { j } \left( \frac { \tilde { p } } { p } \right) ^ { j } \left( 1 - \frac { \tilde { p } } { p } \right) ^ { s - j } \binom { N } { s } p ^ { s } ( 1 - p ) ^ { N - s } ,
$$

where $\alpha \in [ 0 , \frac { 1 } { 2 } ) , N \in \mathbb { N } _ { > 0 }$ , and $0 < \tilde { p } < p < \bar { p } < 1$ . Then g is Lipschitz continuous on $[ \underline { { p } } , \bar { p } ]$ with Lipschitz constant

$$
L _ { g } \leq \frac { N } { 1 - \bar { p } } + \frac { N } { \underline { { p } } - \bar { p } } .\tag{36}
$$

Proof. Because g is a sum of continuous and differentiable functions on $[ \underline { { p } } , \bar { p } ]$ , its Lipschitz constant is given by $L _ { g } = \operatorname* { s u p } _ { p \in [ \underline { { p } } , \bar { p } ] } | g ^ { \prime } ( p )$ . For notational simplicity, we write the summand as

$$
T _ { s , j } ( p ) = C _ { s , j } \left( \frac { \tilde { p } } { p } \right) ^ { j } \left( 1 - \frac { \tilde { p } } { p } \right) ^ { s - j } p ^ { s } ( 1 - p ) ^ { N - s } , \quad C _ { s , j } = \binom { s } { j } \binom { N } { s } .
$$

First, note that, for $\begin{array} { r } { p ^ { s } ( 1 - p ) ^ { N - s } , \left| \frac { d } { d p } \big [ p ^ { s } ( 1 - p ) ^ { N - s } \big ] \right| \leq p ^ { s } ( 1 - p ) ^ { N - s } \frac { N } { 1 - \bar { p } } } \end{array}$ . Then, for $( \tilde { p } / p ) ^ { j } ( 1 - $ $\tilde { p } / p \big ) ^ { s - j }$ , the absolute value of the derivative is at most $\left( \tilde { p } / p \right) ^ { j } \left( 1 - \tilde { p } / p \right) ^ { s - j } \frac { N } { \underline { { p } } - \tilde { p } }$ . Since all coefficients are nonnegative and sum to at most 1, it holds that

$$
| g ^ { \prime } ( p ) | \leq \frac { N } { 1 - \bar { p } } + \frac { N } { \underline { { p } } - \tilde { p } } , \quad p \in [ \underline { { p } } , \bar { p } ] ,
$$

which concludes the proof.

## A.3 Proof of Proposition 4.1

Before proving Proposition 4.1, we introduce the well-known Clopper-Pearson Lemma.

Lemma A.4 (Clopper-Pearson lower bound). Let $\left\{ \mathbf { z } _ { 1 } , \hdots , \mathbf { z } _ { M } \right\}$ be i.i.d. samples from P, and let $h : \mathcal { X } \to \mathcal { Y }$ and $\mathcal { R } \subseteq \mathcal { V }$ be given. Define $\begin{array} { r } { Z = \sum _ { i = 1 } ^ { M } \mathbf { 1 } _ { h ( \mathbf { z } _ { i } ) \in \mathcal { R } } . } \end{array}$ . Given a confidence level $\beta \in [ 0 , 1 ]$ $i f { \check { p } } \in [ 0 , 1 ]$ is such that $\begin{array} { r } { \beta = \sum _ { j = Z } ^ { M } f _ { M , j } ( \check { p } ) } \end{array}$ , then

$$
\mathbb { P } _ { ( \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { M } ) } \left( \mathbb { P } _ { \mathbf { z } } \left( h ( \mathbf { z } ) \in \mathcal { R } \right) \geq j \right) \geq 1 - \beta\tag{37}
$$

We are now ready to prove Proposition 4.1.

Proof. From the Fréchet bound and Lemma A.4,

$$
\begin{array} { r l } & { \mathbb { P } _ { \{ ( \mathbf { w } _ { i } , \varepsilon _ { i } ) \} _ { i = 1 } ^ { N } } \left( \cap _ { m = 1 } ^ { M } \left\{ \widetilde { p } _ { \mathcal { V } _ { m } } \leq \mathbb { P } _ { ( \mathbf { w } , \varepsilon ) } \left( h _ { \mathbf { w } } ( x + \varepsilon ) \in \mathcal { V } _ { m } \right) \leq \widehat { p } _ { \mathcal { V } _ { m } } , \check { p } _ { \mathcal { R } _ { m } } \leq \mathbb { P } _ { ( \mathbf { w } , \varepsilon ) } \left( h _ { \mathbf { w } } ( x + \varepsilon ) \in \mathcal { R } _ { m } \right) \right\} \right) } \\ & { \qquad \geq \displaystyle \sum _ { i = 1 } ^ { 3 M } \left( 1 - \frac { \beta } { 3 M } \right) - ( 3 M - 1 ) = 1 - \beta . } \end{array}
$$

## B Additional Experimental Results

## B.1 Trajectory Prediction

We examine a specific input instance from the dataset and demonstrate that the introduction of a small input perturbation $\varepsilon \sim \dot { \mathcal { N } } ( 0 , I )$ increases the behavioral variability of the target agent by raising the probability of the tactical behavior go. While α-smoothing tends to suppress the go mode – thereby suggesting that the target vehicle is likely to yield – our clustered α-trimming preserves the predicted multi-modality of $h _ { \mathbf { w } } ( x + \varepsilon )$ , as illustrated in Figure 7, where we show the last time step of the sampled trajectories.

## C Summary of method parameters

The clustered α-smoothing certificate is parameterized by three groups. The clustering group depends on the strategy. For the purposes of the experiments in Section 5, we use DBSCAN. The parameters are listed in Table 2.

## D Experiment Details

## D.1 Trajectory Prediction

TrajFlow architecture and training procedure. TrajFlow [26] is built with a combination of a normalizing flow network F with a recurrent autoencoder, consisting of encoder $E _ { \mathrm { R N N } }$ and decoder

![](images/e384666b9cbdf721b99eac9ef48ef26c7310ac7b969168019afd8f66a8c239eb.jpg)  
Figure 7: Given a particular trajectory input x from the L-GAP dataset [47], we show the last time step of trajectory samples from $h _ { \mathbf { w } } ( x ) , h _ { \mathbf { w } } ( x + \bar { \varepsilon } )$ , where $\varepsilon \sim \mathcal { N } ( 0 , I )$ , the α-smoothing in (1), and our clustered α-smoothing. We also show the projection to the last time step subspace of the sets $\mathcal { R } _ { m }$ produced with Algorithm 1 for coverage $p = 0 . 8$

Table 2: Clustered α-smoothing parameters.
<table><tr><td>Parameter</td><td>Symbol</td><td>Impact</td></tr><tr><td>Clustering (DBSCAN strategy)</td><td></td><td></td></tr><tr><td>Maximum number of clusters</td><td> $M _ { \mathrm { m a x } }$ </td><td>If the output of DBSCAN has more than  $M _ { \mathrm { m a x } }$  clusters, then we merge them in order of fewest to most samples.</td></tr><tr><td>Coverage</td><td> $p$ </td><td>See Line 9 of Algorithm 1</td></tr><tr><td>Neighborhood radius</td><td> $\varepsilon _ { \mathrm { d b } }$ </td><td>DBSCAN neighbourhood radius in output space.</td></tr><tr><td>Min. samples</td><td></td><td>DBSCAN core-point threshold.</td></tr><tr><td>Smoothing core</td><td></td><td></td></tr><tr><td>Std.dev. of Gaussian input-perturbation</td><td>σ</td><td>Larger std.dev. allows larger perturbation radius  $^ { r , }$  at the cost of larger deviation from  $\bar { h } _ { \mathbf { w } } ( \bar { x } )$  . See (4).</td></tr><tr><td>Number of samples for smoothing</td><td> $N$ </td><td>More samples tightens the modes of the smoothed distribution.</td></tr><tr><td>Trimming fraction</td><td>α</td><td>The certificate discards the most extreme 2α tails of each per-cluster output coordinate,  $0 \leq \alpha < 1 / 2 .$ </td></tr><tr><td>Confidence</td><td></td><td></td></tr><tr><td>Number of samples for confidence</td><td>N</td><td>Number of samples used for Proposition 4.1. More samples leads to better confidence, at an increased computational cost.</td></tr><tr><td>Confidence parameter</td><td> $\beta$ </td><td>The joint confidence for all estimated probability bounds is  $1 - \bar { \beta } .$ </td></tr></table>

E<sub>RNN</sub> (see Figure 8). Furthermore, the model is completed with the networks ϕ<sub>RNN</sub> (a gated recurring unit encoding the predicted agents past trajectory), ψ<sub>GNN</sub> (a graph neural network encoding the other agents), and ϕ<sub>CNN</sub> (a convolutional network encoding the environment, which is given as a bird’s-eye image). This model is trained in a two-stage process, where first the recurrent autoencoder is trained to encode the future trajectories of all involved agents. Once done, the normalizing flow is trained on the encoded future sampled as well as the past and environment information. An detailed description of the model components and the training loss can be found in the original work [26].

For model training, we employ the STEP framework [38] (MIT License) to train TrajFlow for our chosen scenario. In particular, the model is trained on a dataset combining NuScenes [5] (CC BY-NC-SA) and L-GAP [47] (CC-BY). Following previous works [37, 36], we do not use the standard input for NuScenes (i.e., 4 past time steps at a frequency of 2 Hz), but instead use STEP to extract past trajectories with 10 time steps at 10 Hz. This choice is mostly motivated by the fact that this results in a smoother trajectory, for which we can extract the underlying control states much easier. Importantly, while these datasets provide trajectories with states including positions, velocities and accelerations, for TrajFlow, we only consider the recorded positions. The code is available at https://github.com/DAI-Lab-HEReALD/General-Framework/blob/ main/Framework/simulations\_randomized\_smoothing.py.

Behavioral clustering. For our experiments, we run Algorithm 1 with a behavior-based two-cluster heuristic in endpoint space, using prior knowledge of two tactical modes (stay and go). For each sample, let $\Delta = y _ { \mathrm { e n d } } - y _ { \mathrm { i n i t } }$ . A sample is tagged as stay when $\| \Delta \| _ { 2 } \le r _ { \mathrm { s t a y } }$ , and as go when $\Delta ^ { ( 2 ) } \leq - \delta _ { \downarrow }$ and $\Delta ^ { ( 1 ) } \leq - \delta _ {  }$ , excluding already-tagged stay samples. Endpoint prototypes are then built from these tagged sets (with fallback rules if one set is empty), and all samples are assigned to the nearest prototype in 2D endpoint space. Finally, for each cluster $j \in { 1 , 2 }$ , we construct a tight axis-aligned hyperrectangle $\mathcal { R } _ { j }$ in the full trajectory space (40D) that covers a fraction p of the samples assigned to that cluster.

![](images/957774505e75976bdc08abb8d60c4a9b26aeda8df255b6bfb3ddf4e3ff634b36.jpg)  
Figure 8: An overview of TrajFlow’s architecture, taken from Figure 2 in [26]. During training, future trajectories y are encoded with the encoder $E _ { \mathrm { R N N } }$ <sup>and transformed to the abstracted features</sup> yb<sup>. Using</sup> the normalizing direction, they are then transformed into a sample $z _ { 0 } = F ( \widehat { \pmb { y } } )$ , which follows a standard normal distribution $p _ { 0 }$ . For inference we then use the generative direction, in which a sample $z _ { 0 } \sim p _ { 0 }$ is inversely transformed by the Normalizing Flow to generate the abstracted future trajectories $\widehat { \pmb { y } } = \pmb { F } ^ { - 1 } ( \pmb { z } _ { 0 } )$ that are decoded with $\dot { D } _ { \mathrm { R N N } }$ into the actual trajectories $y _ { \mathrm { p r e d } } .$ <sup>. The likelihood of the encoded trajectory</sup> yb <sup>is</sup> obtained with $p _ { 0 } ( z _ { 0 } )$ det $J _ { F ^ { - 1 } } ( z _ { 0 } ) | ^ { - 1 }$ . The encoding ϕ<sub>CNN</sub> of map $E ,$ and the encoding ψ<sub>GNN</sub> of social interactions are optional blocks, which can provide richer context information.

## D.2 Quadrotor Control

The quadrotor control benchmark was first described in [3]. We use it solely to obtain a safety-critical, stochastic, multi-modal closed-loop system on which to perform clustered randomized smoothing. Relative to [3], we augment the system with a reward structure and extended termination criteria.

Environment. The state is $s _ { k } = [ x _ { k } , \dot { x } _ { k } , y _ { k } , \dot { y } _ { k } , z _ { k } , \dot { z } _ { k } ] \in \mathbb { R } ^ { 6 }$ and the action $u _ { k } = [ u _ { k } ^ { x } , u _ { k } ^ { y } , u _ { k } ^ { z } ] \in$ $[ - 4 , 4 ] ^ { 3 }$ . The dynamics are a per-axis double integrator with unit time step,

$$
s _ { k + 1 } = A s _ { k } + B u _ { k } + C w _ { k } , \qquad w _ { k } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )\tag{38}
$$

with velocities clipped to $[ - 7 , 7 ]$ after each update. In particular, per axis the dynamics are $x _ { k + 1 } =$ $\begin{array} { r } { x _ { k } + \tau \dot { x } _ { k } + \frac { 1 } { 2 } \tau ^ { 2 } \dot { u _ { k } ^ { x } } } \end{array}$ and $\dot { x } _ { k + 1 } = \dot { x } _ { k } + \tau u _ { k } ^ { x } + w _ { k }$ where $w _ { k } \sim \mathcal { N } ( \boldsymbol { 0 } , \sigma ^ { 2 } )$ . The corridor is $[ - 1 5 , \dot { 1 5 } ] \times$ $[ - 9 , 9 ] \times [ - \bar { 7 } , 7 ] .$ , the goal region is $[ 1 1 , 1 \ddot { 5 } ] \times [ 1 , 5 ] \times [ - 7 , - 3 ]$ , and 14 axis-aligned hyperrectangular obstacles match [3]. The termination criteria are in order:

1. if $[ x _ { k } , y _ { k } , z _ { k } ]$ is outside the corridor,

2. if any point on the line segment between $[ x _ { k } , y _ { k } , z _ { k } ]$ and $[ x _ { k + 1 } , y _ { k + 1 } , z _ { k + 1 } ]$ touches an obstacle,

3. if $[ x _ { k } , y _ { k } , z _ { k } ]$ is inside the goal region, or

4. if the max episode length is reached.

The parameters of the environment are summarized in Table 3.

Policy and value networks. The policy is a tanh-squashed Gaussian mixture: at state s, sample a mode $z \sim \mathrm { C a t } ( \operatorname { s o f t m a x } ( g _ { \phi } ( s ) ) )$ ) where Cat is a categorical distribution, sample an unscaled action from this mode $r \sim \mathcal { N } ( \mu _ { \theta } ^ { z } ( s )$ , diag $e ^ { 2 \log \sigma _ { z } } )$ , and rescale the action $\boldsymbol { a } = \operatorname { t a n h } ( \boldsymbol { r } ) \cdot \boldsymbol { a } _ { \mathrm { s c a l e } }$ . The mixture is results in a multi-modal policy, which indeed requires clustered α-smoothing. The architecture is summarized in Table 4.

Training. The policy is trained using standard clipped PPO with GAE and a curriculum on the start state. The curriculum using a linear schedule where a scalar $f \in [ 0 , 1 ]$ linearly anneals from 0 to 1 over the first 40,000 episodes. At each rollout we sample a valid (collision-free, outside the goal) start position uniformly inside an axis-aligned box around the goal center with half-width $r = 3 + 2 5 f$ , and initial velocities $\mathcal { N } ( 0 , ( 0 . 1 + \mathbf { \bar { 0 } } . 4 f ) ^ { 2 } )$ . Without curriculum learning, the policy frequently terminates in collisions before it ever sees the goal. The remaining training parameters are listed in Table 5.

Table 3: Environment parameters. $\Delta$ dist denotes the change in distance $\| \mathrm { p o s - g o a l } \|$ to the goal center goal = (13, 3, 5) between subsequent steps.
<table><tr><td>Quantity</td><td>Value</td><td>Notes</td></tr><tr><td>State / action dim</td><td>6/3</td><td></td></tr><tr><td>Action bounds</td><td> $[ - 4 , 4 ] ^ { 3 }$ </td><td>matches JAIR benchmark</td></tr><tr><td>Process noise σ</td><td>0.05</td><td>on velocity coords; small enough that obstacles dominate</td></tr><tr><td>Velocity clip</td><td>[-7,7]</td><td>prevents PPO blow-up under exploration</td></tr><tr><td>Max steps / episode</td><td>64</td><td>matches rollout horizon</td></tr><tr><td>Reward (goal)</td><td>+10</td><td>terminal</td></tr><tr><td>Reward (collision / exit)</td><td>-5</td><td>terminal</td></tr><tr><td>Reward (timeout)</td><td> $- 0 . 0 5 \cdot \left\| \mathrm { p o s } - \mathrm { g o a l } \right\|$ </td><td>distance-shaped: far timeouts penalised more</td></tr><tr><td>Reward (alive)</td><td> $0 . 5 \cdot \Delta \ddot { \mathrm { d i s t } } - 0 . \bar { 0 } 1$ </td><td>dense progress shaping</td></tr><tr><td>State normalization</td><td> $s / [ 1 5 , 7 , 9 , 7 , 7 , 7 ]$ </td><td>per-dim rescale for the policy net</td></tr></table>

Table 4: Network details. A denotes the action dimension.
<table><tr><td>Component</td><td>Specification</td></tr><tr><td>Trunk</td><td>2 × Linear(128) + tanh, orthogonal init (gain  $\sqrt { 2 } )$ </td></tr><tr><td>Mean head</td><td>Linear(K · A), orthogonal init gain 0.01</td></tr><tr><td>Mixture head  $( K > 1 )$ </td><td>Linear(K), orthogonal init gain 0.01</td></tr><tr><td>Log-std</td><td>global per-component  $[ K , { \bar { A } } ]$  , init —1.2, clamped to  $[ - 5 , 2 ]$ </td></tr><tr><td>Critic</td><td>identical 2 × Linear(128) + tanh trunk, scalar head</td></tr><tr><td>K (mixture components)</td><td>2</td></tr><tr><td>ascale</td><td>4.0 (matches action bound; tanh saturation gives the true control limit)</td></tr></table>

Table 5: PPO hyperparameters.
<table><tr><td>Hyperparameter</td><td>Value</td><td>Notes</td></tr><tr><td>Optimizer</td><td>Adam, l  $\mathrm { ~ r ~ 3 ~ } \times 1 0 ^ { - 4 }$ </td><td>shared across actor and critic</td></tr><tr><td>Total episodes</td><td>200,000</td><td></td></tr><tr><td>Rollout horizon T</td><td>64 steps</td><td>matches env timeout</td></tr><tr><td>Episodes per update</td><td>32</td><td>≈ 2k transitions / batch</td></tr><tr><td>PPO epochs / batch</td><td>10</td><td></td></tr><tr><td>Minibatch size</td><td>512</td><td></td></tr><tr><td>Discount  $\gamma$ </td><td>0.99</td><td></td></tr><tr><td>GAE λ</td><td>0.95</td><td></td></tr><tr><td>Clip range</td><td>0.2</td><td></td></tr><tr><td>Value loss coefficient</td><td>0.5</td><td></td></tr><tr><td>Entropy coefficient</td><td>0.02</td><td>mild; the mixture itself adds exploration</td></tr><tr><td>Max grad norm</td><td>0.5</td><td>global clip across actor + critic</td></tr><tr><td>Advantage normalization</td><td>per-batch (mean / std)</td><td></td></tr></table>

Randomized smoothed policy parameters. Figure 6 is produced with the clustered α-smoothing parameters in Table 6 below. α-smoothing uses the same parameters, except for no clustering.

## E Generalized Voronoi partition

Given a collection of points ${ \mathcal { C } } = \{ c _ { 1 } , \ldots , c _ { M } \} \subset { \mathcal { X } } _ { : }$ , the Voronoi partition $\tilde { \nu }$ induced by $c$ is defined<sup>5</sup> as $\tilde { \mathcal { V } } _ { i } = \left\{ x \in \mathcal { X } : \| x - c _ { i } \| _ { 2 } \leq \| x - c _ { j } \| _ { 2 } , i \neq j \right\}$ , i.e., the point x belongs to the cell $\tilde { \mathcal { V } } _ { i }$ if it is closer to $c _ { i }$ than to any other point $c _ { j }$ in $c .$ . The Generalized Voronoi partition extend this notion

Table 6: Clustered α-smoothing parameters for the quadrotor benchmarks.
<table><tr><td>Parameter</td><td>Value</td><td>Notes</td></tr><tr><td>σ</td><td>0.05</td><td></td></tr><tr><td>N</td><td>30</td><td></td></tr><tr><td>α</td><td>0.4</td><td></td></tr><tr><td> $M _ { \mathrm { m a x } }$ </td><td>3</td><td></td></tr><tr><td>Coverage p</td><td>0.9</td><td></td></tr><tr><td>Clustering alg.</td><td>DBSCAN</td><td></td></tr><tr><td>εdb</td><td>0.45</td><td>Output-space DBSCAN radius.</td></tr><tr><td>dbscan_min_samples</td><td>50</td><td>Core-point threshold relative.</td></tr><tr><td> $\bar { N }$ </td><td>4000</td><td>Samples for Proposition 4.1.</td></tr><tr><td> $\beta$ </td><td> $1 0 ^ { - 2 }$ </td><td></td></tr></table>

![](images/c6aaa3ee2f8243e3f21e0251ba1df0dab42bc6e45da477239a632d0b09b66fe5.jpg)

![](images/24787d9fd4a838f15ddf201a206d439838e153aea20a1a90850d3952ff8f947e.jpg)  
Figure 9: Generalized Voronoi partition of four disjoint axis-aligned rectangles $\mathcal { R } _ { 1 } , . . . , \mathcal { R } _ { 4 } \subset \mathbb { R } ^ { 2 }$ under the Euclidean metric. (a) The distance from point p to the closest point in each region $\mathcal { R } _ { i }$ is measured and p is assigned $\nu _ { 3 }$ as $\mathcal { R } _ { 3 }$ is the closest. (b) Each cell $\nu _ { i }$ collects every point closer to $\mathcal { R } _ { i }$ than to any other site. Cell boundaries in the case of rectangles in 2D consist of straight segments and parabolic arcs and meet at Voronoi vertices where three cells agree on the distance to a common point.

to sets, provided a shortest-distance measure from x to a set $\mathcal { R } \subseteq \mathcal { X }$ . We write

$$
d ( x , { \mathcal { R } } ) = \operatorname* { i n f } _ { r \in { \mathcal { R } } } \| x - r \| _ { 2 }\tag{39}
$$

for the (Euclidean) point-to-set distance, which equals 0 if and only ${ \mathrm { i f ~ } } x \in { \mathcal { R } }$ for closed .

Definition 4 (Generalized Voronoi partition). For a collection $\pmb { \mathcal { R } } = \{ \mathcal { R } _ { 1 } , . . . , \mathcal { R } _ { M } \}$ of pairwise disjoint closed sets in  , the generalized Voronoi cell associated with $\mathcal { R } _ { i }$ is given by

$$
\mathcal { V } _ { i } = \big \{ x \in \mathcal { X } : d ( x , \mathcal { R } _ { i } ) \leq d ( x , \mathcal { R } _ { j } ) f o r a l l j \neq i \big \} ,
$$

and $\pmb { \nu } = \{ \mathcal { V } _ { i } \} _ { i = 1 } ^ { M }$ is the generalized Voronoi partition induced by R.

Unlike $\tilde { \nu } ,$ , the cells in the generalized Voronoi partition V are generally not convex, even when the sets $\mathcal { R } _ { i }$ are convex or simple shapes such as rectangles, as shown in Figure 9. This fact, together with the combinatorial size of the partition that grows rapidly with both the number of sets M and the dimension of the ambient space , implies that computing the partition is almost certainly intractable. Indeed, even for the point-induced partition $\tilde { \nu } ,$ the worst-case complexity in $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ is already $\Theta \left( M ^ { \lceil d / 2 \rceil } \right)$ . When the sets $\mathcal { R } _ { i }$ are convex, the cell boundaries are $( d - 1 )$ )-dimensional algebraic surfaces, and for arbitrary compact sets the topology can be unbounded. Explicit construction of the partition is, therefore, intractable in $\mathbb { R } ^ { d }$ beyond toy examples. Crucially, however, our procedure does not require the explicit construction of the partition, but only identifying to which cell $\nu _ { i }$ a given point $x \in \mathcal { X }$ belongs, which reduces to computing M point-to-set distances (39). This admits a trivial implementation when, for instance, the sets $\mathcal { R } _ { i }$ are axis-aligned hyperrectangles.

## F Quantization artifact in α-smoothing

As shown in Figure 6, α-smoothing on bi-modal distributions unintuitively outputs a “quantized” smoothed prediction. The prediction concentrates on a small, near-uniformly spaced set of quantization levels whose positions and weights are fully determined by the mixture and trimming parameters.

We describe this artifact with the simplest non-trivial case, a 1-D two-component Gaussian mixture, which is enough to expose the mechanism. The same reasoning extends mode-by-mode to mixtures with more components or to higher-dimensional outputs (where each axis is quantized independently after coordinate-wise sorting).

## F.1 Setup

Let the output distribution be the bi-modal Gaussian mixture

$$
p ( y ) = \pi _ { A } { \mathcal { N } } ( y \mid \mu _ { A } , \sigma ^ { 2 } ) + \pi _ { B } { \mathcal { N } } ( y \mid \mu _ { B } , \sigma ^ { 2 } ) , \qquad \pi _ { A } + \pi _ { B } = 1 ,\tag{40}
$$

with $\mu _ { A } < \mu _ { B }$ and noise scale σ small relative to the mode gap $\mu _ { B } - \mu _ { A }$ . At a fixed smoothed input, the predictor draws N samples $Y _ { 1 } , \dots , Y _ { N } \overset { \mathrm { i . i . d . } } { \sim } p .$ , sorts them as $Y _ { ( 1 ) } \le \cdots \le Y _ { ( N ) }$ , and returns the symmetric α-trimmed mean

$$
\hat { \mu } _ { \alpha } = \frac { 1 } { M } \sum _ { i = L + 1 } ^ { N - L } Y _ { ( i ) } , \qquad L = \lfloor \alpha N \rfloor , \quad M = N - 2 L .\tag{41}
$$

Each Monte Carlo realization of the smoothed predictor is one independent draw of $\hat { \mu } _ { \alpha }$ . We call the set $\left\{ Y _ { ( L + 1 ) } , \dots , Y _ { ( N - L ) } \right\}$ the trimmed core.

## F.2 Quantization in the noiseless limit

Let K be the (random) number of samples drawn from mode A. By construction $K \sim \operatorname { B i n o m } ( N , \pi _ { A } )$ In the noiseless limit $\sigma  0$ the modes are perfectly separated, so after sorting the first K values come from mode A and the remaining $N - K$ from mode B. The trimmed core then contains $c _ { A } ( K ) = \mathrm { c l i p } \left( K - L , 0 , M \right)$ values from mode A and $c _ { B } ( K ) = M - c _ { A } ( K )$ from mode B. The trimmed mean then reduces to

$$
\hat { \mu } _ { \alpha } ^ { \sigma = 0 } ( K ) = \frac { c _ { A } ( K ) \mu _ { A } + c _ { B } ( K ) \mu _ { B } } { M } .\tag{42}
$$

Since $c _ { A }$ takes only integer values $0 , 1 , \ldots , M$ , the predictor collapses onto $M + 1$ discrete quantization levels

$$
\ell _ { j } = \frac { j \mu _ { A } + ( M - j ) \mu _ { B } } { M } , \qquad j = 0 , 1 , \ldots , M ,\tag{43}
$$

spaced uniformly between $\mu _ { A }$ and $\mu _ { B }$ at increments of $( \mu _ { A } - \mu _ { B } ) / M$ . Each level carries the binomial probability that K takes a value mapping to it,

$$
\pi _ { j } = \left\{ \begin{array} { l l } { \operatorname* { P r } [ K \leq L ] } & { j = 0 , } \\ { \operatorname* { P r } [ K = L + j ] } & { j = 1 , \ldots , M - 1 , } \\ { \operatorname* { P r } [ K \geq N - L ] } & { j = M . } \end{array} \right.\tag{44}
$$

The two boundary levels absorb the tails of $\operatorname { B i n o m } ( N , \pi _ { A } )$ because any $K \leq L$ saturates $c _ { A } ( K ) = 0$ and any $K \ge \bar { N _ { } } - L$ saturates $c _ { A } ( K ) = M$

Addressing the failure mode of α smoothing. Increasing N and decreasing α help alleviate the behavior by enlarging the trimmed core, and as $N \to \infty \mathrm { o r } \alpha \to 0$ , the distribution approaches a continuous distribution. However, both are problematic. Decreasing α means including more outliers, and increasing N collapses the distribution to a tight peak around the mean, which is even more disjoint from the individual modes. Our clustered α-smoothing avoids the artifact by summarizing each mode with its own trimmed mean and encoding the binomial split in the partition weights $\pi ^ { ( m ) }$ instead of in the level positions.

## F.3 Order-statistic correction at finite σ

The noiseless scenario highlights the origin of the quantization behavior, but neglects a bias in the level positions for $\sigma > 0$ . The mode-A samples that survive trimming are not arbitrary mode-A draws but specific order statistics of the K mode-A samples (those closer to mode-B after sorting), and analogously for mode B. Both contributions are biased toward the gap between the modes.

Conditioning on $K = k$ , the i-th overall order statistic is

$$
\begin{array} { r } { Y _ { ( i ) } \mid K = k \ \stackrel { d } { = } \ \left\{ { A } _ { ( i : k ) } , \right. \ ~ i \le k , } \\ { \left. B _ { ( i - k : N - k ) } , \right. \ ~ i > k , } \end{array}\tag{45}
$$

where $A _ { ( j : n ) }$ and $B _ { ( j : n ) }$ denote the j-th order statistic among n i.i.d. draws from the corresponding mode. Writing $\nu _ { j : n } = \mathbb { E } [ Z _ { ( j : n ) } ]$ for the expected j-th order statistic of n standard normals, the conditional expected trimmed mean is

$$
\mathbb { E } \left[ { \hat { \mu } } _ { \alpha } \mid K = k \right] = { \frac { 1 } { M } } \left[ c _ { A } ( k ) \mu _ { A } + c _ { B } ( k ) \mu _ { B } + \sigma \left( S _ { A } ( k ) + S _ { B } ( k ) \right) \right] ,\tag{46}
$$

with order-statistic correction terms

$$
S _ { A } ( k ) = \sum _ { i = L + 1 } ^ { \operatorname* { m i n } ( k , N - L ) } \nu _ { i : k } \quad ( 0 \mathrm { i f } k \le L ) ,\tag{47}
$$

$$
S _ { B } ( k ) = \sum _ { i = \operatorname * { m a x } ( L + 1 , k + 1 ) } ^ { N - L } \nu _ { i - k : N - k } \quad ( 0 \mathrm { i f } k \geq N - L ) .\tag{48}
$$

Because the expected order statistic is monotone in its rank, $S _ { A } ( k )$ is non-negative (we are averaging the upper order statistics of the mode-A sub-sample whenever $k > L )$ and $S _ { B } ( k )$ is non-positive (lower order statistics of mode B). Both shifts pull $\operatorname { \mathbb { E } } [ \hat { \mu } _ { \alpha } \mid K = k ]$ inward, i.e., toward the middle of the gap. The shifts are equal and opposite when $c _ { A } ( \dot { k } ) \stackrel { \cdot } { = } c _ { B } ( k ) \stackrel { \cdot } { = } M / 2$ , so the central level $\ell _ { M / 2 }$ remains at $( \mu _ { A } + \mu _ { B } ) / 2$

The marginal expected level for the j-th quantization bin is the binomial weighted average of (46) over the k values mapping to that bin (a single k for interior bins, a tail of values for the two boundary bins).

## F.4 Worked example: $N = 3 0 , \alpha = 0 . 4 , \pi _ { A } = 0 . 4$

With these parameters, the trimmed core is $M = 6$ samples, giving $M + 1 = 7$ quantization bins. The bins are nearly uniformly spaced; the order-statistic correction compresses the outer bins inward by roughly $0 . 2 2 \approx 0 . 7 \sigma$ and leaves the center bin invariant. Most of the probability mass concentrates at the boundary level closest to $\mu _ { B }$ because $\mathbb { E } [ K ] = N \pi _ { A } = 1 2 = { \overset { \cdot } { L } } L .$ , placing the median of the binomial split exactly at the lower trimming boundary.

Figure 10 shows the underlying mixture (top) alongside the empirical histogram of $\hat { \mu } _ { \alpha }$ over 20,000 Monte Carlo realizations (bottom). The dashed lines mark the corrected levels $\bar { \ell } _ { j }$ and labels report the binomial weights $\pi _ { j }$ from (44). Both line up tightly with the empirical peaks, confirming (43)-(46).

## G Control of the residual in Proposition 3.2

One can achieve a desired residual level $\epsilon > 0$ by considering a uniform partition of the interval $[ \underline { { p } } _ { \nu _ { l } } , \bar { p } \nu _ { l } ]$ into $\begin{array} { r } { K > \operatorname* { m a x } _ { l \in \mathcal { L } } 1 + \frac { | \mathcal { L } | } { \epsilon } ( \bar { p } \gamma _ { l } - \underline { { p } } _ { \mathcal { V } _ { l } } ) \left| \frac { N } { 1 - \bar { p } \gamma _ { l } } + \frac { N } { \underline { { p } } _ { \mathcal { V } _ { l } } - \underline { { p } } _ { \mathcal { R } _ { l } } } \right| } \end{array}$ anchor points. Then, it holds that $\begin{array} { r } { \sum _ { l \in \mathcal { L } } \left[ \frac { N } { 1 - \bar { p } \nu _ { l } } + \frac { N } { \underline { { p } } _ { \mathcal { V } _ { l } } - \underline { { p } } _ { \mathcal { R } _ { l } } } \right] h _ { l , K } \le \epsilon , } \end{array}$

Underlying 2-component Gaussian mixture  
![](images/d7ec528686595f075e1978f68d3b93543056212ae0c3397d011183687e4000c3.jpg)

![](images/3bc41b7a01a451e242c8a674a31f520022cc237a90ac62a78f09f4c4be05b95c.jpg)  
Figure 10: Quantization of the single-cluster α-trimmed predictor on a bimodal output. Top: the underlying mixture $0 . 4 \mathsf { \tilde { N } } ( - 3 , 0 . 3 ^ { 2 } ) + 0 . 6 \mathsf { \tilde { N } } ( + 3 , 0 . 3 ^ { 2 } )$ . Bottom: histogram of $\hat { \mu } _ { \alpha }$ for $N = \hat { 3 } 0 , \alpha \stackrel { - } { = } 0 . 4$ over 20,000 realizations. Dashed lines are the predicted quantization levels $\boldsymbol { \bar { \ell } } _ { j }$ from (46); labels are the binomial weights $\pi _ { j }$ from (44).