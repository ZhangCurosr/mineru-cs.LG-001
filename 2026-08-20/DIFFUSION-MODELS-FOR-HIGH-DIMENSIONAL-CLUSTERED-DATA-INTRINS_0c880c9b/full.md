# DIFFUSION MODELS FOR HIGH-DIMENSIONAL CLUSTERED DATA: INTRINSIC-DIMENSION ADAPTIVITY VIA BAYESIAN CLASSIFICATION

By Yuga Iguchi and Paul Fearnhead

School ofMathematical Sciences, Lancaster University, Lancaster, UK

## Abstract

The empirical success of difusion models in generative modelling has motivated theoretical work, including quantitative error bounds and qualitative analyses that characterise the diferent phases of denoising. We bring these two areas together by studying the adaptivity of difusion models to the structured geometry of multimodal high-dimensional data that consists of multiple clusters in R<sup>�</sup>, each with its own low-dimensional structure, and inter-cluster separation depending on �. We employ �-mixture Gaussian distributions as a canonical framework to capture this geometry and establish two theoretical results. First, we interpret denoising as a dynamical Bayesian classifier: the mixture score is a posterior-weighted average of cluster-wise scores, and we show that, with high probability, the posterior class probabilities concentrate on a single cluster once the signalto-noise ratio reaches the scale Θ(log(��)/�). Second, by separately analysing the denoising process in its mixing and cluster-commitment phases, we prove that the KL error bound depends linearly on the maximum intrinsic dimension of a cluster, up to a logarithmic factor, even when � grows polynomially with �. This improves on ambient-dimensional bounds and extends existing low-dimensional adaptivity analyses to multimodal distributions with heterogeneous, approximately low-rank covariances.

Key words. Bayesian classification; Difusion models; Dimensional scaling; Intrinsic dimension; Phase transitions.

## 1. Introduction

Denoising Difusion Probabilistic Models (DDPMs) [Song and Ermon, 2019, Ho et al., 2020, Song et al., 2020] are general methods for generative modelling that have produced state-of-the-art results across many applications [Dhariwal and Nichol, 2021, Popov et al., 2021, Lugmayr et al., 2022, Trippe et al., 2023]. They define a process that takes data and progressively adds noise to it to produce a realisation from (a close approximation) to some simple distribution. They then learn the reverse, denoising, process, so that we can generate new data examples by denoising realisations from the simple distribution.

Relatedly, there has been much interest in understanding why DDPMs are so efective. Much of this work has been theoretical, producing bounds on the accuracy of DDPM. Initial work gave bounds in terms of the Kullback-Leibler (KL) divergence between the true data distribution and the distribution sampled by a DDPM [Benton et al., 2024, Conforti et al., 2025], though these have been extended to other metrics such as total variation distance [Liang et al., 2025, Brešar and Mijatović, 2025], 1-Wasserstein distance [De Bortoli, 2022] or 2-Wasserstein distance [Gao et al., 2025, Silveri and Ocello, 2025, Arsenyan et al., 2026, Koike, 2026].

Within this body of work there has been particular interest in understanding why DDPMs scale well to high-dimensional distributions. Under KL divergence, standard bounds on the accuracy of DDPMs [Benton et al., 2024, Conforti et al., 2025] have terms that, up to a logarithmic factor, scale linearly with the ambient dimension, �, and suggest that the computational cost, such as the number of time-steps used when denoising, needs to scale with �. While earlier works have established such a bound under minimal conditions on the target, e.g., bounded second moment for data distribution, more recent work has made assumptions that the data lie on a low-dimensional manifold with a bounded support and then shows that in such cases the error can scale with the dimension of the manifold rather than the ambient dimension. For example, Potaptchik et al. [2025] assume that data lies on a bounded manifold of intrinsic dimension � and show that the KL-bound is then linear in �. Li and Yan [2024] define the dimension of the manifold, �, in terms of metric entropy (which implicitly also implies the data distribution has bounded support) and show that to bound the total variation error the number of time-steps when denoising needs to scale quadratically in �. This was subsequently improved to linearly in � by Li and Yan [2025] and Huang et al. [2026].

Separately, work has focussed on whether DDPMs generalise or just memorise the data [Li et al., 2023, Dupuis et al., 2025, Bonnaire et al., 2025, Maillard and Goldt, 2026]. That is, does simulating from a DDPM produce new examples from the underlying data distribution, or just reproduce members of the data set it was trained on. Many applications of DDPMs involve data distributions with separate clusters, for example image data with images of diferent types. Biroli et al. [2024] studied the noising/denoising process for such data and was able to give rigorous qualitative insight into its behaviour albeit assuming exact simulation of the denoising process using the true empirical score function. They show that the denoising process can be split into three phases, one where there is general exploration, a second where the process fixes the cluster it will sample from, and then a final stage where the process will be attracted to a specific data point from the training sample. The last stage corresponds to memorisation, and can be avoided by early stopping of the denoiser. Achilli et al. [2026] further extended this study of DDPMs for mixture distributions.

## 1.1. Our Contributions

Our focus is on analysing the geometric adaptivity of DDPMs for target data from a �-mixture of distributions on R<sup>�</sup>, exhibiting both low dimensionality and distinct clustering. This is motivated by the notion of the union ofmanifolds hypothesis [Brown et al., 2022]: that high-dimensional data such as images lie on a disjoint union of manifolds of varying intrinsic dimensions. Our contributions are summarised as follows:

• We rigorously extend the general framework of Biroli et al. [2024] and Achilli et al. [2026] to study a situation where each cluster distribution samples from, or is close to, a low-dimensional manifold. Under the structured geometry specified later, we show that the denoising process fixes on a single cluster to sample from with high probability after a suitably defined signal-to-noise ratio (SNR) reaches a threshold Θ(log(��)/�).

• Exploiting the above characterisation of cluster commitment in terms of SNR, we give rigorous bounds on the error of DDPM accounting for the various approximations such as time-discretisation of the denoising process, which, under suitable regularity conditions, is shown to scale linearly with the maximum intrinsic dimension of any of the clusters, rather than the ambient dimension �.

As such our work brings together the mixture framework of [Biroli et al., 2024] and Achilli et al. [2026] with the literature on KL-bounds for DDPMs, and the ability of DDPMs to adapt to low-dimensional structure within each distinct cluster. To demonstrate the intrinsic adaptivity in the KL error despite the distinct multimodality, we first characterise the phase transition where the DDPM switches from exploring between clusters to fixing on a single cluster. By leveraging this, and separately analysing the behaviour of the DDPM before and after the phase transition, we are able to show that the KL error bound depends linearly on the intrinsic dimensions up to a logarithmic factor of �, even when the number of clusters grows polynomially in the ambient dimension �. A detailed description of the link between the phase transition and the intrinsic dimension adaptivity in KL error is provided in Section 3.1.

In order to get our results, we make a number of assumptions on the data distribution. The strongest of which is that each component of our mixture is Gaussian. i.e., Gaussian mixture models (GMMs). This is in line with recent papers that study simplified models in order to gain insight into DDPMs [Li et al., 2026, Oymak and Gulcu, 2021, Shah et al., 2023, Wu et al., 2024, Wang et al., 2025]. In particular, Li et al. [2026] establish dimension-free convergence of DDPM in total variation under the common isotropic covariance structure. In contrast, we consider GMMs with heterogeneous and anisotropic covariances and assume that each cluster is well-separated, in that the squared distances between cluster means scale linearly with $D ,$ while each cluster has its own low-dimensional structure. This inter-cluster separation induces the phase transition described above, leading to our intrinsic-dimension adaptivity result under the heterogeneous and anisotropic covariance geometry. In our work, the intrinsic dimensionality is imposed by an assumption that the spectrum of the cluster covariance matrix decays quickly: for cluster $i ,$ there is some $M _ { i }$ such that the sum of all but the $M _ { i }$ -largest eigenvalues is bounded. This is a weaker condition than assuming the data in cluster � lies on an $M _ { i } .$ -dimensional manifold. Importantly, we do not require that the support of the data distribution is bounded. We stress that our main result focuses on the dimensional scaling of the noise or time-discretisation schedule for DDPMs or associated reverse SDEs, given a suficiently accurate estimate of the score, which analytically explains why a reasonable number of steps is enough in practice to sample from high-dimensional clustered data. In contrast, Wang et al. [2025] focuses on score estimation, and clarifies the relation between the training samples and the intrinsic dimensionality to minimise a training loss when the target is a mixture of exactly low-rank Gaussians.

This paper is organised as follows. Section 2 summarises the basic set-up of difusion models and introduces the structured geometry under GMMs. Section 3 presents our two main analytical results on the phase transition and intrinsic-dimension adaptivity in KL error. We provide empirical experiments using real data in Section 4 to demonstrate that our analytical findings on the phase transition can be observed beyond GMMs. We prove the second main result in Section 5 and then conclude in Section 6.

Notation. For a vector $\boldsymbol { x } \in \mathbb { R } ^ { D }$ and matrix $A \in \mathbb { R } ^ { D \times D }$ , we write $\| x \| _ { A } = { \sqrt { x ^ { \top } A x } }$ and $\left. \boldsymbol { x } \right. = \sqrt { \sum _ { i = 1 } ^ { D } x _ { i } ^ { 2 } }$ Denote by $\lambda _ { i } ( A )$ the �-th largest eigenvalue of the matrix A and write $\| A \| _ { 2 } ~ = ~ \sqrt { \lambda _ { 1 } ( A ^ { \top } A ) }$ and $\| A \| _ { F } = \sqrt { \mathrm { t r } ( A ^ { \top } A ) }$ . For two vectors $x , y \in \mathbb { R } ^ { D } , \langle x , y \rangle$ denotes the inner product. For two matrices $A , B \in \mathbb { R } ^ { D \times D } , A \preceq B$ mean that $B - A$ is positive semidefinite. For positive integer $M ,$ we write $[ M ] \equiv \left\{ 1 , \dots , M \right\}$ . For $x > 0$ , we write $f ( x ) = \Theta ( x )$ if there exist constants $0 < c _ { 1 } \leq c _ { 2 } <$ ∞ such that $c _ { 1 } x \leq f ( x ) \leq c _ { 2 } x$ . For two sequences $\{ a _ { n } \} _ { n }$ and $\{ b _ { n } \} _ { n }$ , we write $a _ { n } = O ( b _ { n } )$ if there exists a constant $C > 0$ such that $a _ { n } \leq C b _ { n }$ for all suficiently large �. We also write $a _ { n } = o ( b _ { n } )$ if $\scriptstyle \operatorname* { l i m } _ { n \to \infty } a _ { n } / b _ { n } = 0$ For any two probability measures P and $\mathbb { Q }$ with P being absolutely continuous with respect to $\mathbb { Q } ,$ we write $\begin{array} { r } { \mathrm { K L } \bigl ( \mathbb { P } \left| \right| \mathbb { Q } \bigr ) = \int \log \frac { d \mathbb { P } } { d \mathbb { Q } } ( \omega ) d \mathbb { P } ( \omega ) } \end{array}$ for the KL divergence between them. We write $\gamma ^ { D } : \mathcal { B } ( \mathbb { R } ^ { D } )  [ 0 , 1 ]$ as the standard Gaussian measure defined as $\begin{array} { r } { \gamma ^ { D } ( A ) = ( 2 \pi ) ^ { - D / 2 } \int _ { A } \exp ( - \frac { \| z \| ^ { 2 } } { 2 } ) d z , A \in \mathcal { B } ( \mathbb { R } ^ { D } ) } \end{array}$ . Also, for ${ \boldsymbol { m } } \in \mathbb { R } ^ { D }$ and $V \in \mathbb { R } ^ { D \times D } , N ( m , V )$ denotes the normal distribution with mean � and covariance �. For a random variable �, Law(�) denotes the probability law of �.

## 2. Setup

## 2.1. Score-Based Generative Models

We first review score-based generative models [Song et al., 2020], which provide a continuous-time framework for sampling from a target distribution. Let $( \Omega , { \mathcal { F } } , { \mathbb { P } } )$ be the probability space where the �-dimensional standard Brownian motion $( B _ { t } ) _ { t \geq 0 }$ is equipped, and $P _ { \mathrm { d a t a } }$ denote the target distribution on $\mathbb { R } ^ { D }$ . Denoising difusion models are designed to generate samples from $P _ { \mathrm { d a t a } }$ by reversing a forward noising process, where the latter process can be formulated as the following Ornstein-Uhlenbeck

(OU)-type SDE:

$$
\vec { d X } _ { t } = - \overrightarrow { X } _ { t } d t + \sqrt { 2 } d B _ { t } , \quad \overrightarrow { X } _ { 0 } \sim P _ { \mathrm { d a t a } } , \quad t \in [ 0 , T ] ,
$$

where $T > 0$ is a fixed time horizon. We denote by $\vec { p } _ { t } : \mathcal { B } ( \mathbb { R } ^ { D } )  [ 0 , 1 ]$ the marginal distribution of $\smash { \vec { X } } _ { t }$ , given as

$$
\overrightarrow { p } _ { t } ( A ) = \int \overrightarrow { p } _ { t | 0 } ( A | x ) P _ { \mathrm { d a t a } } ( d x ) , \quad A \in \mathcal { B } ( \mathbb { R } ^ { D } ) ,
$$

where $\begin{array} { r } { \overrightarrow { p } _ { t \vert 0 } ( A \vert x ) \equiv \mathbb { P } \big ( \overrightarrow { X } _ { t } \in A \vert \overrightarrow { X } _ { 0 } = x \big ) } \end{array}$ . The forward noise $\smash { \vec { X } } _ { i }$ conditional on the initial data point $\scriptstyle { \vec { X } } _ { 0 }$ follows a Gaussian distribution, and specifically, the solution is expressed as $\overrightarrow { X } _ { t } = m _ { t } \overrightarrow { X } _ { 0 } + \sigma _ { t } Z , Z \sim$ ${ \cal N } ( 0 , I _ { D } )$ , where $m _ { t }$ and $\sigma _ { t }$ represent the strength of signal and noise, defined as

$$
m _ { t } = \exp ( - t ) , \sigma _ { t } = \sqrt { 1 - m _ { t } ^ { 2 } } .\tag{1}
$$

With a slight abuse of notation, we write $\scriptstyle { \vec { p } } _ { t \mid 0 }$ as the associated Lebesgue density. Similarly, we write $\vec { p } _ { t }$ for the Lebesgue density of the marginal law of $\smash { \vec { X } } _ { t }$ if it exists. Then, the denoising SDE is defined as:

$$
\begin{array} { r } { d \overleftarrow { X } _ { t } = \big ( \overleftarrow { X } _ { t } + 2 \nabla \log \overrightarrow { p } _ { T - t } ( \overleftarrow { X } _ { t } ) \big ) d t + \sqrt { 2 } d B _ { t } , \quad \overleftarrow { X } _ { 0 } \sim \mathrm { L a w } ( \overrightarrow { X } _ { T } ) . } \end{array}\tag{2}
$$

The classical results of [Anderson, 1982, Haussmann and Pardoux, 1986] show that $( \overleftarrow { X } _ { t } ) _ { t \in [ 0 , T ] }$ admits a weak solution so that $\operatorname { L a w } ( \overleftarrow { X } _ { t } ) = \operatorname { L a w } ( \overrightarrow { X } _ { T - t } )$ for any $t \in [ 0 , T ]$ . In particular $\mathrm { L a w } ( \overleftarrow { X } _ { T } ) = \mathrm { L a w } ( \overrightarrow { X } _ { 0 } )$ , so exact simulation of the reverse SDE (2) allows for sampling from the target distribution $P _ { \mathrm { d a t a } }$

DDPMs [Ho et al., 2020] can be viewed as a tractable alternative to (2), which involves three approximations: (i) The use of an estimated score – the intractable score function is replaced with a function $\hat { s } : [ 0 , T ] \times \mathbb { R } ^ { D } \to \mathbb { R } ^ { D }$ parametrised with a neural network, obtained by optimising an appropriate loss function [Yang et al., 2023]; (ii) The use of the Gaussian prior ${ \cal N } ( 0 , I _ { D } )$ for initialisation rather than $\operatorname { L a w } ( { \overrightarrow { X } } _ { T } )$ ; and (iii) Time-discretisation of the continuous-time process (2).

Even with these approximations, it can be dificult to approximately simulate from (2) for $t \approx T$ as the drift, specifically the term 2∇ log $\overleftrightarrow { p } _ { T - t } ( \overleftarrow { X } _ { t } )$ , blows-up as $t \to T$ . To overcome this, simulation is often performed up to some time $T - \varepsilon ,$ for a suitable choice of early-stopping parameter $\varepsilon > 0$ . It is simple to bound the error between Law $( \overleftarrow { X } _ { T - \varepsilon } )$ and $\operatorname { L a w } ( \overleftarrow { X } _ { T } )$ in terms of � [Benton et al., 2024].

## 2.2. Bayesian Classification Perspective and Our Analytic Framework

Given the above setup, our primary goal is to rigorously investigate the generative mechanism and geometric adaptivity of the reverse process, characterised by (2), when $P _ { \mathrm { d a t a } }$ is a union of � clusters, written in the form

$$
P _ { \mathrm { d a t a } } ( A ) = \sum _ { k = 1 } ^ { K } \pi _ { k } \cdot P _ { \mathrm { d a t a } } ^ { [ k ] } ( A ) , \quad K \geq 2 , A \in \mathcal { B } ( \mathbb { R } ^ { D } ) ,
$$

where $\{ \pi _ { k } \}$ satisfying $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \pi _ { k } = 1 } \end{array}$ are the prior weights and $P _ { \mathrm { d a t a } } ^ { [ k ] }$ denotes the distribution associated with the �th class. In the mixture setting, the key observation is that the denoising procedure can be viewed as a dynamical Bayesian classification, as advocated also in recent work [Achilli et al., 2026]. More specifically, the score function ∇ log $\vec { p } _ { T - t } ( \cdot )$ contained in the drift of (2) is expressed as a weighted

average of the cluster-wise scores, that is,

$$
\nabla \log \overrightarrow { p } _ { T - t } ( x ) = \sum _ { k = 1 } ^ { K } W _ { D } ^ { [ k ] } ( t , x ) \nabla \log \overrightarrow { p } _ { T - t } ^ { [ k ] } ( x ) , \qquad x \in \mathbb { R } ^ { D } ,
$$

where $\vec { p } _ { T - t } ^ { [ k ] } ( x )$ is the Lebesgue density of $\stackrel { \triangledown } { \vec { X } } _ { T - t }$ given initial data point ${ \vec { X } } _ { 0 }$ belonging to the �-th cluster, i.e.,

$$
\overrightarrow { p } _ { T - t } ^ { [ k ] } ( x ) = \int \overrightarrow { p } _ { T - t | 0 } ( x | z ) \cdot P _ { \mathrm { d a t a } } ^ { [ k ] } ( d z )
$$

and $W _ { D } ^ { [ k ] } ( t , x )$ is the posterior mixture weight or Bayes classifier, i.e., the conditional probability of the label taking � given the state $\vec { X } _ { t } = x$ , specifically,

$$
W _ { D } ^ { [ k ] } ( t , x ) = \frac { \pi _ { k } \cdot \overrightarrow { p } _ { T - t } ^ { [ k ] } ( x ) } { \sum _ { \ell = 1 } ^ { K } \pi _ { \ell } \cdot \overrightarrow { p } _ { T - t } ^ { [ \ell ] } ( x ) } .\tag{3}
$$

Thus, the reverse process (2) evolves according to the posterior weights given the current denoising state $\overleftarrow { X } _ { t }$ , indicating that it implicitly performs a dynamic Bayesian classification within the clustered data setting.

The insight presented above suggests that understanding the generative mechanism of difusion models for clustered data relies on studying the stochastic behaviour of the posterior weights $W _ { D } ^ { [ k ] } ( t , \overleftarrow { X } _ { t } )$ for all $t \in [ 0 , T )$ . In this paper, we focus on Gaussian mixture models (GMMs) as the target data, as they provide a solid analytical framework for rigorously examining the dynamic Bayesian classification. Under this setting, the marginal distribution of the forward noise also follows a Gaussian mixture. Also, cluster-wise low-dimensional structure can be employed within each covariance. Thus, we define $P _ { \mathrm { d a t a } } ^ { [ k ] } ( A ) = N ( A ; \mu _ { k } , \Sigma _ { k } ) , A \in \mathcal { B } ( \mathbb { R } ^ { D } )$ and specify the target distribution as:

$$
P _ { \mathrm { d a t a } } ( A ) = \sum _ { k = 1 } ^ { K } \pi _ { k } \cdot N ( A ; \mu _ { k } , \Sigma _ { k } ) , \qquad K \geq 2 , A \in \mathcal { B } ( \mathbb { R } ^ { D } ) ,\tag{4}
$$

where $\mu _ { k } \in \mathbb { R } ^ { D } , \Sigma _ { k } \in \mathbb { R } ^ { D \times D }$ with $\Sigma _ { k } \succeq 0$ and $N ( A ; \mu _ { k } , \Sigma _ { k } )$ is a Gaussian measure on $\mathbb { R } ^ { D }$ defined as

$$
N ( A ; \mu _ { k } , \Sigma _ { k } ) \equiv \gamma ^ { D } \big ( \{ z \in \mathbb { R } ^ { D } | \mu _ { k } + \Sigma _ { k } ^ { 1 / 2 } z \in A \} \big ) .
$$

The measure admits a well-defined Lebesgue density if $\Sigma _ { k }$ is positive definite. Also, note that the distribution of the cluster-wise forward noising $\vec { p } _ { T - t } ^ { [ k ] } ( \cdot ) , t \in [ 0 , T ) , k \in [ K ]$ follows a Gaussian with mean $m _ { T - t } \mu _ { k }$ and covariance $S _ { T - t } ^ { k }$ given as

$$
S _ { T - t } ^ { k } = m _ { T - t } ^ { 2 } \Sigma _ { k } + \sigma _ { T - t } ^ { 2 } I _ { D }\tag{5}
$$

which is positive definite due to the added noise for $t \in [ 0 , T )$ .

We now introduce the following structure for the target measure (4):

Assumption 1 (Prior Weights). There exists a constant $C _ { 1 } \geq 0$ such that

$$
\begin{array} { r } { \operatorname* { m a x } _ { i , j \in [ K ] } \log \frac { \pi _ { i } } { \pi _ { j } } \leq C _ { 1 } . } \end{array}
$$

Assumption 2 (Covariance). (i) $\Sigma _ { i }$ is symmetric positive semi-definite for any cluster $i \in [ K ]$ . Also,

there exist a constant $C _ { 2 } > 0$ and $\alpha \in [ 0 , 1 )$ such that for all $D \in \mathbb { N }$

$$
\operatorname* { m a x } _ { i \in [ K ] } \| \Sigma _ { i } \| _ { 2 } \leq C _ { 2 } \cdot D ^ { \alpha } .
$$

(ii) (Low-dimensional structure) For each cluster $i \in [ K ]$ , there exist a positive fixed integer $M _ { i } < D$ and $\nu _ { i } \geq 0$ such that for all $D \in \mathbb { N }$ , we have

$$
\sum _ { k = M _ { i } + 1 } ^ { D } \lambda _ { k } ( \Sigma _ { i } ) \leq \nu _ { i } , \qquad \frac { \nu _ { i } } { M _ { i } } \leq R\tag{6}
$$

for some universal constant $R \in [ 0 , 1 )$

Assumption 3 (Mean). For any two clusters $i , j \in [ K ]$ , there exist constants $C _ { 3 } , C _ { 4 } > 0$ such that for all $D \in \mathbb { N }$

$$
C _ { 3 } \cdot D \le \| \mu _ { i } - \mu _ { j } \| ^ { 2 } \le C _ { 4 } \cdot D .\tag{7}
$$

The first assumption is weak as it just bounds the ratio of the mixture weights. The second assumption imposes the structure of each component. Assumption 2(i) allows at most sub-linear dependency on � for the maximum eigenvalue of each cluster’s covariance, which is weaker than assuming a uniform bound w.r.t. �. Sublinear growth is a technical condition that facilitates the phase transition analysis later in Section 3. Assumption 2(ii) together with (i) covers both exact or approximately low-rank structure; if data from component � is confined to a manifold of dimension $M _ { i }$ , then (6) would hold with $\nu _ { i } = 0 $ Notably, Assumption 2(ii) is a weaker condition that only requires data from each component to lie close to a low-dimensional manifold, with � being a global measure of how close data from each mixture component is to its manifold.

Finally, Assumption 3 defines the separation of clusters as a function of the ambient dimension �. The square distance between the cluster means is assumed to increase linearly with �. It is also natural for image data where increasing � corresponds to observing the same image at increasing resolution. We will also empirically assess these data geometries using high-dimensional datasets (images and RNA sequencing data) in Section 4.

## 3. Posterior Classification Analysis and Intrinsic Dimension Adaptivity

## 3.1. Overview

This section provides an overview of the generative mechanism of the reverse process within the GMM framework (4) detailed in Section 2.2. We will also explain how the dynamic Bayesian classification perspective can be linked to the KL convergence analysis, thereby demonstrating its ability to adapt to the geometry of the low-dimensional structure within each cluster.

Sections 3.2–3.4 analyse the stochastic behaviour of the posterior weights $W _ { D } ^ { [ k ] } ( t , \overleftarrow { X } _ { t } )$ given in (3), revealing that the denoising process decomposes into the following two phases: In the Phase I – the mixing phase – several posterior weights remain non-negligible, and the trajectory is afected by global inter-cluster geometry. Then, in Phase II – the commitment phase –, one posterior weight becomes dominant, that is the posterior weight concentrates on a single class, i.e., $W _ { D } ^ { [ k ] ^ { \star } } \approx 1$ for some $j \in [ K ]$ and $W _ { D } ^ { [ i ] } \approx 0 , i \in [ K ] \setminus \{ j \}$ , and the score is efectively reduced to the corresponding cluster-wise score

$$
\nabla \log \overrightarrow { p } _ { T - t } ( \overleftarrow { X } _ { t } ) \approx \nabla \log \overrightarrow { p } _ { T - t } ^ { [ j ] } ( \overleftarrow { X } _ { t } ) .
$$

The analytic results presented in Sections 3.3–3.4 demonstrate a key efect of the ambient dimension � on the posterior weight concentration. Roughly, the transition from Phase I to II or mode commitment

can occur with probability at least $1 - D ^ { - q } , \ q \geq 1$ after the signal-to-noise ratio (SNR)

$$
\begin{array} { r } { \mathrm { S N R } _ { T - t } \equiv \frac { m _ { T - t } ^ { 2 } } { \sigma _ { T - t } ^ { 2 } } , \qquad t \in [ 0 , T ) , } \end{array}\tag{8}
$$

reaches scale log $( K D ) / D$ , see Theorem 1 in Section 3.4. These observations imply that inter-class separation that scales with the ambient dimension � facilitates earlier resolution of the cluster component identification, and the rest of the denoising process then focuses on building fine details from the single cluster. Table 1 summarises this two-phase picture.

Table 1. Summary of the dynamical regimes of the denoising difusion for high-dimensional �-clustered data.
<table><tr><td></td><td>Phase I</td><td>Phase II</td></tr><tr><td>Phase Characterisation</td><td>Mixing phase</td><td>Commitment to a single cluster j</td></tr><tr><td>Signal-to-noise ratio  $( \mathrm { S N R } _ { T - t } )$ </td><td> $O \big ( \log ( K D ) / D \big )$ </td><td>0(1)</td></tr><tr><td>Posterior Weights</td><td> $W _ { D } ^ { [ k ] } \ge 0 , k \in [ K ]$  (Onset of posterior collapse)</td><td> $W _ { D } ^ { [ j ] } \approx 1 , W _ { D } ^ { [ k ] } \approx 0 , k \neq j$  (Posterior concentration)</td></tr><tr><td>Effective Geometry</td><td>Inter-cluster inference + local covariances</td><td>Single local covariance (cluster j)</td></tr></table>

Given the above characterisation of the denoising process, we bound the KL error between the target distribution and its approximation realised by DDPM samplers in Section 3.5. The main analytic result therein is that, up to a logarithmic factor, the time-discretisation error component of the KL divergence scales linearly with the intrinsic dimension, i.e., the low-rank property within each cluster, rather than with the ambient dimension �. To show this we use diferent approaches to bound the KL error in Phase I and Phase II: For Phase I we use the fact that the KL error involves terms that depend on the SNR, and in this phase the SNR is $O ( \log ( K D ) / D )$ , which ofsets the inter-cluster separation depending on � in the first and second moments. For Phase II, due to cluster-commitment, the denoiser behaves as one targeting a single cluster, so we can bound the error in terms of the cluster’s intrinsic dimension. A more detailed overview of our argument is given in Section 3.6.

Remark 1. Similar attempts to split the denoising process into distinct phases are considered in [Biroli et al., 2024, Achilli et al., 2026]. One of the focuses in these works is on characterising when the marginal density of the process transitions from a single-well structure (when the process is contractive) to a multi-well one. They study this in an asymptotic manner, i.e., consider a Taylor series expansion of the score function in very small signal $m _ { T - t } \ll 1$ and analyse the curvature of the approximate Hessian. In contrast, our analysis proceeds with a probabilistic characterisation of posterior weights concentration without relying on the Taylor expansion in $m _ { T - t }$

## 3.2. High Probability Bounds for Posterior Weights

We begin by stating the following dynamic characterisation of the posterior weights, which serves as a basis to develop the phase transition arguments hereafter:

Lemma 3.1. Let $T \geq 1$ and $\delta \in ( 0 , 1 )$ . Then, for each $i , j \in [ K ]$ , it holds that:

$$
\mathbb { P } \Big ( \log W _ { D } ^ { [ i ] } ( t , \overleftarrow { X } _ { t } ) \leq \operatorname* { m i n } \{ - C _ { D } ^ { [ i , j ] } ( t , \delta ) , 0 \} \mid \overleftarrow { X } _ { T } \sim P _ { \mathrm { d a t a } } ^ { [ j ] } \Big ) \geq 1 - \delta
$$

for each $t \in [ 0 , T )$ , where we have set:

$$
\begin{array} { r l r } {  { C _ { D } ^ { [ i , j ] } ( t , \delta ) = \log \frac { \pi _ { j } } { \pi _ { i } } + \mathrm { K L } \big ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \mid \mid \overrightarrow { p } _ { T - t } ^ { [ i ] } \big ) } } \\ & { } & { - \sqrt { 2 \mathcal { T } _ { D } ^ { [ i , j ] } ( t ) \log \frac { 2 } { \delta } } - \sqrt { 2 \mathcal { U } _ { D } ^ { [ i , j ] } ( t ) \log \frac { 2 } { \delta } } - 2 \mathcal { V } _ { D } ^ { [ i , j ] } ( t ) \log \frac { 2 } { \delta } , } \end{array}\tag{9}
$$

with

$$
\begin{array} { r l } & { \mathcal { T } _ { D } ^ { [ i , j ] } ( t ) = m _ { T - t } ^ { 2 } \Big \Vert \mu _ { i } - \mu _ { j } \Big \Vert _ { ( S _ { T - t } ^ { i } ) ^ { - 1 } ( S _ { T - t } ^ { j } ) ( S _ { T - t } ^ { i } ) ^ { - 1 } } ^ { 2 } ; } \\ & { \mathcal { U } _ { D } ^ { [ i , j ] } ( t ) = \frac { 1 } { 2 } \mathrm { t r } \Big ( \big ( I _ { D } - S _ { T - t } ^ { j } ( S _ { T - t } ^ { i } ) ^ { - 1 } \big ) ^ { 2 } \Big ) ; } \\ & { \mathcal { V } _ { D } ^ { [ i , j ] } ( t ) = \big \Vert I _ { D } - S _ { T - t } ^ { j } ( S _ { T - t } ^ { i } ) ^ { - 1 } \big \Vert _ { 2 } . } \end{array}
$$

The proof of Lemma 3.1 is postponed to Appendix B.1. We further introduce event sets

$$
A ^ { [ i , j ] } ( t , \delta ) = \Big \{ \omega \in \Omega \mid \log W _ { D } ^ { [ i ] } ( t , \overleftarrow { X } _ { t } ) \le \operatorname* { m i n } \big \{ - C _ { D } ^ { [ i , j ] } ( t , \delta ) , 0 \big \} \Big \Big \} , \qquad i , j \in [ K ] .\tag{10}
$$

Then, the following result is immediate:

Proposition 1. Let $T \geq 1 , \delta \in ( 0 , 1 )$ and $j \in [ K ]$ . It holds that:

$$
\mathbb { P } \bigg ( \bigcap _ { i \in [ K ] } A ^ { [ i , j ] } \big ( t , \frac { \delta } { K } \big ) | \overleftarrow { X } _ { T } \sim P _ { \mathrm { d a t a } } ^ { [ j ] } \bigg ) \geq 1 - \delta ,
$$

for each $t \in [ 0 , T )$ .

Proposition 1 tells us that for denoising trajectories $\{ \overleftarrow { X } _ { t } \}$ eventually sampling from the cluster $j$ at the terminal time �, the posterior weights $W _ { D } ^ { [ i ] } ( t , \overleftarrow { X } _ { t } )$ with any cluster index � rather than $j$ are bounded by min $\left\{ \exp \bigl ( - C _ { D } ^ { [ i , j ] } ( t , \delta / K ) \bigr ) , 1 \right\}$ with probability at least $1 - \delta , \delta \in ( 0 , 1 )$ .

ProofofProposition 1. We have that:

$$
\begin{array} { r l } & { \mathbb { P } \Big ( \Big ( \displaystyle \bigcap _ { i \in [ K ] } A ^ { [ i , j ] } \big ( t , \frac \delta K \big ) \Big ) ^ { c } \mid \overleftarrow { X } _ { T } \sim P _ { \mathrm { d a t a } } ^ { [ j ] } \Big ) = \mathbb { P } \Big ( \bigcup _ { i \in [ K ] } \left( A ^ { [ i , j ] } \big ( t , \frac \delta K \big ) \right) ^ { c } \mid \overleftarrow { X } _ { T } \sim P _ { \mathrm { d a t a } } ^ { [ j ] } \Big ) } \\ & { \qquad \le \displaystyle \sum _ { i \in [ K ] } \mathbb { P } \Big ( \left( A ^ { [ i , j ] } \big ( t , \frac \delta K \big ) \right) ^ { c } \mid \overleftarrow { X } _ { T } \sim P _ { \mathrm { d a t a } } ^ { [ j ] } \Big ) \le \displaystyle \sum _ { i \in [ K ] } \frac \delta K = \delta , } \end{array}
$$

where we used Lemma 3.1 in the last line. The proof is complete.

## 3.3. Posterior Weights in High-Dimensional Limit

Proposition 1 indicates that the decay of the posterior weights is dynamically controlled by how large the signal KL $\big ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \mid \mid \overrightarrow { p } _ { T - t } ^ { [ i ] } \big )$ is compared to the prior ratio and the fluctuation terms. The following result demonstrates how the signal component dominates in the term $C _ { D } ^ { [ i , j ] }$ in the high-ambient-dimensional regime, $D \to \infty$

Proposition 2. Let $T \geq 1$ and $j \in [ K ]$ . Also, let Assumptions $1 , 2 , 3$ hold and assume thatfor each $t \in [ 0 , T )$ and $i , j \in [ K ]$ , the following limit exists:

$$
\operatorname* { l i m } _ { D \to \infty } \frac { 1 } { D } \times \mathrm { K L } \big ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \mid \mid \overrightarrow { p } _ { T - t } ^ { [ i ] } \big ) \equiv \widetilde { C } _ { \infty } ^ { [ i , j ] } ( t ) \geq 0 .
$$

Then, we have that: conditionally on the event $\left\{ \omega \in \Omega | \overleftarrow { X } _ { T } \sim P _ { \mathrm { d a t a } } ^ { [ j ] } \right\}$

$$
\operatorname* { l i m } _ { D \to \infty } \operatorname* { s u p } _ { D } \log W _ { D } ^ { [ i ] } ( t , \overleftarrow { X } _ { t } ) \leq - \widetilde { C } _ { \infty } ^ { [ i , j ] } ( t ) , \qquad \forall i \in [ K ] ,
$$

holds true almost surely for each $t \in [ 0 , T )$ .

Proposition 2 implies that in the high ambient dimensional limit $D \to \infty ,$ , the decay of the posterior weight is mainly governed by the KL divergence between pairs of noised cluster distributions. The data geometry stated in Assumptions 1, 2 and 3 ensure that the KL divergence (signal) eventually dominates the fluctuation terms (noise) specified in (9) as $D \to \infty$ . If the clusters in the target distribution are well-separated, e.g., in the sense of the first or second moment, then the signal (KL divergence) is also expected to increase as � approaches the terminal time �. That is, for small $t \in [ 0 , T - \varepsilon ]$ , the noised clusters are nearly indistinguishable, but the gap gradually increases as the denoising proceeds, where those diferences are measured by the KL divergence. In the early stage of denoising, the posterior weights are basically similar to the prior weights, but, for some classes, these posterior weights start to degenerate toward 0, followed by a rapid posterior concentration for a single cluster $j .$ In particular, such a rapid cluster commitment occurs with an exponential rate after the signal KL $\cdot \big ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \big | \big | \overrightarrow { p } _ { T - t } ^ { [ i ] } \big )$ reaches a certain level so that $- \mathrm { K L } \big ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \mid \mid \overrightarrow { p } _ { T - t } ^ { [ i ] } \big ) + o ( D )$ takes a suficiently large negative value.

ProofofProposition 2. Set $\delta = D ^ { - q } , q \geq 2$ . Recall the definition of the event $A ^ { [ i , j ] }$ in (10) and define $\begin{array} { r } { E _ { D } ^ { [ j ] } ( t , \delta ) = \bigcap _ { i \in [ K ] } A ^ { [ i , j ] } \big ( t , \frac { \delta } { K } \big ) } \end{array}$ and $B ^ { [ j ] } = \{ \omega \in \Omega | \overleftarrow { X } _ { T } \sim P _ { \mathrm { d a t a } } ^ { [ j ] } \}$ . Then, Proposition 1 yields

$$
\sum _ { D = 1 } ^ { \infty } \mathbb { P } \Big ( \big ( E _ { D } ^ { [ j ] } ( t , D ^ { - q } ) \big ) ^ { c } | B ^ { [ j ] } \Big ) \leq \sum _ { D = 1 } ^ { \infty } D ^ { - q } < \infty .
$$

Thus, the first Borel-Cantelli lemma gives that:

$$
\mathbb { P } \Big ( \operatorname* { l i m } _ { D \to \infty } \boldsymbol { \mathrm { i n f } } E _ { D } ^ { [ j ] } ( t , D ^ { - q } ) | B ^ { [ j ] } \Big ) = 1 ,
$$

that is, for $\mathbb { P } ( \cdot | B ^ { [ j ] } )$ almost every $\omega ,$ there exists $D ^ { \dagger } \in \mathbb { N }$ such that for all $D \geq D ^ { \dagger }$ we have

$$
\log W _ { D } ^ { [ i ] } ( t , \overleftarrow { X } _ { t } ) + C _ { D } ^ { [ i , j ] } ( t , \frac { \delta } { K } ) \leq 0\tag{11}
$$

for all $i \in [ K ]$ . Dividing the both sides of (11) by � and taking lim $\operatorname* { s u p } _ { D \to \infty }$ gives

$$
\operatorname* { l i m } _ { D \to \infty } \operatorname* { s u p } _ { D } \log W _ { D } ^ { [ i ] } ( t , \overleftarrow { X } _ { t } ) \leq - \operatorname* { l i m } _ { D \to \infty } \operatorname* { s u p } _ { D } \frac { 1 } { D } C _ { D } ^ { [ i , j ] } ( t , \frac { \delta } { K } ) .
$$

The proof completes once we show that for $\delta = D ^ { - q } , q \geq 2$

$$
\begin{array} { r } { \operatorname* { l i m } _ { D \to \infty } \mathop { \operatorname* { l p } } _ { D } ^ { 1 } C _ { D } ^ { [ i , j ] } ( t , \frac \delta K ) = \widetilde C _ { \infty } ^ { [ i , j ] } ( t ) , } \end{array}\tag{12}
$$

where all the fluctuation terms in (9) multiplied by $1 / D$ converges to 0 as $D \to \infty$ . To this end, we exploit the following upper bounds for the fluctuation terms in (9) whose proof is contained in Appendix D:

Lemma 3.2. It holds thatfor each $t \in [ 0 , T )$ and all $i , j \in [ K ]$

$$
\begin{array} { r } { \mathcal { T } _ { D } ^ { [ i , j ] } ( t ) \leq \big ( 1 + \mathrm { S N R } _ { T - t } \cdot \| \Sigma _ { i } - \Sigma _ { j } \| _ { 2 } \big ) \cdot m _ { T - t } ^ { 2 } \| \mu _ { i } - \mu _ { j } \| _ { ( S _ { T - t } ^ { j } ) ^ { - 1 } } ^ { 2 } ; } \end{array}\tag{13}
$$

$$
\begin{array} { r } { \mathcal { U } _ { D } ^ { [ i , j ] } ( t ) \leq \frac { 1 } { 2 } { \bf S N R } _ { T - t } ^ { 2 } \left. \Sigma _ { i } - \Sigma _ { j } \right. _ { F } ^ { 2 } ; } \end{array}\tag{14}
$$

$$
\begin{array} { r } { \mathcal { V } _ { D } ^ { [ i , j ] } ( t ) \leq \mathrm { S N R } _ { T - t } \cdot \| \Sigma _ { i } - \Sigma _ { j } \| _ { 2 } . } \end{array}\tag{15}
$$

Lemma 3.2 with Assumptions $1 - 3$ yields that: for each $t \in [ 0 , T )$ with a fixed $T \geq 1 , \mathcal { T } _ { D } ^ { [ i , j ] } ( t ) =$ $O ( D ^ { \alpha + 1 } ) , \mathcal { U } _ { D } ^ { [ i , j ] } ( t ) = O ( D ^ { 2 \alpha } )$ and $\mathcal { V } _ { D } ^ { [ i , j ] } ( t ) = O ( D ^ { \alpha } )$ , leading to the desired equation (12), where we used $\| \Sigma _ { i } - \Sigma _ { j } ^ { - } \| _ { F } ^ { 2 } = O ( D ^ { 2 \alpha } )$ shown in (75). The proof of Proposition 2 is now complete.

## 3.4. Posterior Concentration and the Scale of SNR

We hereby examine the time scale at which the posterior weight decay, or concentration, occurs. Inspired by the high-dimensional asymptotics presented in Proposition 2, we analytically identify the point in time after which the main signal consistently outweighs other components in equation (9), such as the ratio of prior weights and fluctuation terms. Specifically, by setting a fixed early stopping criterion $ { \varepsilon } \in ( 0 , 1 )$ , we consider the time $t ^ { \dagger } \in [ 0 , T - \varepsilon ]$ such that $C _ { D } ^ { [ i , \dot { j } ] } ( \dot { t } , \delta / K ) \bar { \geq } \eta \mathrm { K L } \big ( \dddot { p } _ { T - t } ^ { [ j ] } \big | \big | \dddot { p } _ { T - t } ^ { [ i ] } \big )$ for all $t \geq t ^ { \dagger }$ , for a fixed constant $\eta \in ( 0 , 1 )$ . To analytically characterise the scale of $t ^ { \dagger }$ , we introduce an additional condition:

Assumption 4. There exists a constant $C _ { 5 } > 0$ such that for all $i , j \in [ K ]$

$$
\operatorname* { m a x } _ { k \in [ D ] } \left| \langle \mu _ { i } - \mu _ { j } , u _ { k } ( j ) \rangle \right| \leq C _ { 5 } ,\tag{16}
$$

where $u _ { k } ( j )$ is the eigenvector associated with the eigenvalue $\lambda _ { k } ( \Sigma _ { j } )$

Assumption 4 states that the projection of the mean diference vector between clusters onto the eigenvectors of one cluster does not concentrate on a few principal components. We then have the following result, whose proof is postponed to Section B.2:

Theorem 1 (SNR Scale for Posterior Concentration). Let $T \geq 1$ and $ { \varepsilon } \in ( 0 , 1 )$ be a fixed early stopping. Also set $\eta \in ( 0 , 1 )$ and $q \geq 2 .$ . Let Assumptions $I , 2 , 3 ,$ 4 hold. Then, we have thefollowing: There exists $D ^ { \dagger } = D ^ { \dagger } ( \eta , q , \varepsilon )$ such that for any integer $D \geq D ^ { \dagger }$

$$
\mathbb { P } \Bigg ( \bigcap _ { i \in [ K ] } \left\{ W _ { D } ^ { [ i ] } ( t , \overleftarrow { X } _ { t } ) \leq \exp \Big ( - \eta \mathrm { K L } \big ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \mid \big | \overrightarrow { p } _ { T - t } ^ { [ i ] } \big ) \Big ) \right\} | \overleftarrow { X } _ { T } \sim P _ { \mathrm { d a t a } } ^ { [ j ] } \Bigg ) \geq 1 - D ^ { - q } , \quad j \in [ K ] ,\tag{17}
$$

for each $\begin{array} { r } { t \in \left[ \operatorname* { m a x } \bigl \{ 0 , T + \frac { 1 } { 2 } \log \frac { S } { 1 + \underline { { S } } } \bigr \} , T - \varepsilon \right] } \end{array}$ , equivalently $\mathrm { S N R } _ { T - t } \in \left\lceil \operatorname* { m a x } \big \{ \mathrm { S N R } _ { T } , \underline { { S } } \big \} , \mathrm { S N R } _ { \varepsilon } \right\rceil$ , where $\underline { { S } } \equiv \underline { { S } } ( D , q , \eta )$ satisfies:

$$
\begin{array} { r } { \underline { { \cal S } } = \Theta \left( \frac { \log \left( K D \right) } { D } \right) . } \end{array}\tag{18}
$$

Besides, for any $\beta \in ( 0 , 1 )$

$$
\mathbb { P } \Bigg ( \underset { i \neq j } { \bigcap } \left\{ W _ { D } ^ { [ i ] } ( t , \overleftarrow { X } _ { t } ) \leq \beta \right\} \mid \overleftarrow { X } _ { T } \sim P _ { \mathrm { d a t a } } ^ { [ j ] } \Bigg ) \geq 1 - D ^ { - q } , \quad j \in [ K ] ,
$$

whenever the reverse time � satisfies

$$
\operatorname { S N R } _ { T - t } \geq L \cdot { \frac { \operatorname* { m a x } \left\{ \log ( K D ) , \log { \frac { 1 } { \beta } } \right\} } { D } }\tag{19}
$$

for some constant $L > 0$ independent of �.

Theorem 1 establishes an SNR scaling law for posterior concentration. That is, for trajectories eventually sampling from cluster $j ,$ the posterior weights assigned to all other classes are suficiently small with high probability once the SNR reaches a threshold $\Theta \big ( \log ( K D ) / D \big )$ . In the proof, Assumption 3 (mean separation ) together with Assumption 4 plays a role in quantifying the signal KL $\big ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \big | \big | \overrightarrow { p } _ { T - t } ^ { [ i ] } \big )$ leading to the obtained scale of the SNR threshold $\underline { { S } } .$

The above result also gives rise to the following key implications on the appropriate scaling of � with respect to the ambient dimension � and its efect on the length of the mixing phase:

Corollary 1. Consider the same setting as the statement in Theorem 1 and let $\begin{array} { r } { t ^ { \dagger } = \operatorname* { m a x } \big \{ 0 , T + \frac { 1 } { 2 } \log \frac { S } { 1 + \underline { { S } } } \big \} } \end{array}$ Then, we have the following:

(i). If the time horizon $T > 0$ is a fixed constant, then $t ^ { \dagger }  0$ as $D \to \infty ,$ i.e., (17) holds for each $t \in [ 0 , T - \varepsilon ]$

(ii). $\begin{array} { r } { I f T \geq \frac { 1 } { 2 } \log D } \end{array}$ , then $t ^ { \dagger } \gtrsim$ log log �. Thus, the threshold $t ^ { \dagger }$ does not collapse to 0 as $D \to \infty , i . e .$ ., the mixing phase can remain non-trivial.

We remark that for $\begin{array} { r } { T \geq \frac { 1 } { 2 } \log D } \end{array}$ , we have $\mathrm { S N R } _ { T } \leq 2 / D , D \geq 2$ , and so $\mathrm { S N R } _ { T } \leq \underline { { \mathbf { S } } }$ for any suficiently large �. The choice of time interval $\begin{array} { r } { T \geq \frac { 1 } { 2 } } \end{array}$ log � is also reasonable in reducing the gap between $\scriptstyle { \vec { p } } _ { T }$ and ${ \cal N } ( 0 , I _ { D } )$ so that the forward noising distribution at time � is suficiently close to the normal distribution. See also Theorem 2 below.

Proof of Corollary 1. The first statement (i) is immediate by noticing that $\begin{array} { r } { T + \frac { 1 } { 2 } \log \frac { S } { 1 + \underline { { S } } } \to - \infty } \end{array}$ as $D \to \infty$ given (18). For (ii), $\begin{array} { r } { T \geq \frac { 1 } { 2 } \log D } \end{array}$ with (18) gives $\begin{array} { r } { T + \frac { 1 } { 2 } \log \frac { S } { 1 + \underline { { S } } } \geq \frac { 1 } { 2 } \log \frac { D \underline { { S } } } { 1 + \underline { { S } } } \gtrsim } \end{array}$ log log � for any suficiently large �. The proof is now complete.

## 3.5. KL Convergence Analysis – Intrinsic Dimension Adaptivity via Posterior Concentration

Given the above arguments on the dynamical regimes, we now argue how the time-discretisation error scales with the ambient dimension � and the intrinsic dimensionality $M _ { i }$ involved in each cluster specified in Assumption 2.

Let $ { \varepsilon } \in ( 0 , 1 )$ be a fixed early stopping. We write the time-discretisation schedule $\{ t _ { i } \} _ { i \in [ N ] }$ satisfying $0 = t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot < t _ { N } = T - \varepsilon$ with $T \geq 1$ and the number of steps $N \in \mathbb { N }$ . We define the time-discretised denoising sampler :

$$
\begin{array} { r } { d \overleftarrow { Y } _ { t } = \Big ( \overleftarrow { Y } _ { t } - 2 \frac { \overleftarrow { Y } _ { t } } { \sigma _ { T - t } ^ { 2 } } + 2 \frac { m _ { T - t } } { \sigma _ { T - t } ^ { 2 } } \hat { \mu } ( T - t _ { i } , \overleftarrow { Y } _ { t _ { i } } ) \Big ) d t + \sqrt { 2 } d B _ { t } , \qquad t \in [ t _ { i } , t _ { i + 1 } ] , } \end{array}
$$

with $\overleftarrow { Y } _ { 0 } \sim N ( 0 , I _ { D } )$ , where $\hat { \mu } _ { T - t _ { i } } ( y )$ is an estimator of the true posterior mean $\mu _ { T - t _ { i } } ( y ) = \mathbb { E } [ \overrightarrow { X } _ { 0 } \vert \overrightarrow { X } _ { T - t _ { i } } =$ $y ]$ . We write $\overrightarrow { \mathbb { P } } _ { t }$ and as the law of $\smash { \vec { X } } _ { t }$ and $\overleftarrow { Y } _ { t _ { i } }$ , respectively and study the upper bound for $\mathrm { K L } \big ( \overrightarrow { \mathbb { P } } _ { \varepsilon } \mid | \overleftarrow { \mathbb { Q } } _ { t _ { N } } \big )$ . We then need to decide the schedule for the time discretisation. In this work, we adopt the following clear and simple update rule for grid points: for a precision parameter $\tau \in ( 0 , 1 / 4 ]$

$$
\begin{array} { r l } & { t _ { 0 } = 0 ; } \\ & { t _ { j + 1 } = \operatorname* { m i n } \bigl \{ t _ { j } + \tau \sigma _ { T - t _ { j } } ^ { 2 } , T - \varepsilon \bigr \} , \qquad j \in \{ 0 , \ldots , N - 1 \} } \end{array}\tag{20}
$$

where � is the minimum integer s.t. $t _ { N - 1 } + \tau \sigma _ { T - t _ { N - 1 } } ^ { 2 } \geq T - \varepsilon$ . The design of the schedule is determined by the local noise $\sigma _ { T - t _ { i } } ^ { 2 }$ , but is independent of the geometry of multimodal distributions. Let $h _ { j + 1 } = t _ { j + 1 } - t _ { j } , j = 0 , 1 , . . . , N - 1$ , then this schedule has two properties. First, we have $h _ { j + 1 } \leq 2 \tau \operatorname* { m i n } \{ 1 , T - t _ { j } \}$ , which aligns with the exponential-decreasing-type schedules often employed in convergence analyses; see, e.g., Benton et al. [2024], Potaptchik et al. [2025], Huang et al. [2026]. The second property is that the number of time-steps � increases as $1 / \tau$ , as shown in the following lemma, whose proof is postponed to Appendix A:

Lemma 3.3. For a given precision parameter $\tau \in ( 0 , 1 / 4 ]$ , the number of steps � specified in the time-discretisation schedule (20) satisfies

$$
\begin{array} { r } { N = \Theta \Big ( 1 + \frac { 1 } { \tau } \big ( T - 1 + \log \frac { 1 } { \varepsilon } \big ) \Big ) . } \end{array}\tag{21}
$$

To state our main result, we introduce the following assumption regarding the accuracy of score estimation.

Assumption 5 (Score Estimate Accuracy). There exists $\varepsilon _ { \mathrm { s c o r e } } > 0$ such that

$$
\frac { 1 } { T } \sum _ { j = 0 } ^ { N - 1 } h _ { j + 1 } \times \mathbb { E } \big [ \| \nabla \log p _ { T - t _ { j } } ( \overleftarrow { X } _ { t _ { j } } ) - \hat { s } ( T - t _ { j } , \overleftarrow { X } _ { t _ { j } } ) \| ^ { 2 } \big ] \leq \varepsilon _ { \mathrm { s c o r e } } ^ { 2 } ,
$$

where

$$
\hat { s } ( T - t _ { j } , x ) \equiv - \frac { x - m _ { T - t _ { j } } \hat { \mu } ( T - t _ { j } , x ) } { \sigma _ { T - t _ { j } } ^ { 2 } } , \qquad x \in \mathbb { R } ^ { D } .\tag{22}
$$

Treating the accuracy as a black box, as per this assumption, is commonly adopted in difusion model convergence analysis [Benton et al., 2024, Potaptchik et al., 2025, Huang et al., 2026]. We now have the following KL convergence guarantee:

Theorem 2 (KL Convergence under Dynamical Regimes). Let $T \geq \operatorname* { m a x } \left\{ 1 , { \frac { 1 } { 2 } } \log D \right\}$ and $ { \varepsilon } \in ( 0 , 1 )$ be a fixed early stopping parameter. Consider the time-discretisation schedule (20) with $\tau \in ( 0 , 1 / 4 ]$ Let Assumptions $I , 2 , 3 ,$ 4 and 5 hold. Also, assume $\mathbb { E } _ { P _ { \mathrm { d a t a } } } [ \| X \| ^ { 2 } ] < \infty , K \lesssim D ^ { \gamma } f o r c$ fixed $\gamma > 0$ and $\varepsilon \gtrsim R / T$ , where $R \in [ 0 , 1 )$ isfound in Assumption 2. Set $M ^ { \dagger } = \mathrm { m a x } _ { k \in [ K ] } M _ { k }$ . Then, there exists $D ^ { \dagger } \in \mathbb { N }$ such that for all $D \geq D ^ { \dagger }$ , we have KL $. ( \overrightarrow { \mathbb { P } } _ { \varepsilon } | | \overleftarrow { \mathbb { Q } } _ { t _ { N } } ) \lesssim \mathcal { E } _ { 1 } + \mathcal { E } _ { 2 } + \mathcal { E } _ { 3 }$ , where

$$
\begin{array} { r } { \mathcal { E } _ { 1 } \equiv \tau M ^ { \dagger } \Big ( T + \log \frac { 1 } { \varepsilon } \Big ) , \qquad \mathcal { E } _ { 2 } \equiv T \varepsilon _ { \mathrm { s c o r e } } ^ { 2 } , \qquad \mathcal { E } _ { 3 } \equiv \left( D + \mathbb { E } _ { P _ { \mathrm { d a t a } } } [ \| X \| ^ { 2 } ] \right) e ^ { - 2 T } . } \end{array}\tag{23}
$$

The proof of Theorem 2 is postponed to Section 5. The bound (23) shows the three sources of errors $- \mathcal { E } _ { \mathrm { l } } \colon$ time-discretisation error; $\varepsilon _ { 2 } { \mathrm { : } }$ score-estimation error; and E<sub>3</sub>: prior gap – which aligns well with the standard theoretical results in the literature. The main highlight is that the discretisation error scales with the intrinsic dimension $M ^ { \dagger }$ , rather than the ambient dimension $D _ { ; }$ , despite the $O ( D )$ inter-class separation, while the number of mixture components � is allowed to grow with $D$ polynomially. Moreover, combining (21) with Theorem 2 gives the following complexity bound:

Corollary 2. Consider the setting of Theorem $^ 2$ and set $\begin{array} { r } { A \equiv M ^ { \dagger } \left( T + \log \frac { 1 } { \varepsilon } \right) } \end{array}$ . For any $\chi \in \left( 0 , A / 4 \right]$ choosing $\tau = \chi / A \in ( 0 , 1 / 4 ]$ leads to $\mathcal { E } _ { 1 } = \chi$ , where the number ofsteps � satisfies

$$
\begin{array} { r } { N = O \left( \frac { M ^ { \dagger } } { \chi } \left( T + \log \frac { 1 } { \varepsilon } \right) ^ { 2 } \right) . } \end{array}
$$

Remark 2. The requirement for the early stopping criterion $\varepsilon \gtrsim R / T$ gives insight into the role of � and its relation to the geometry of the data. When $\nu ^ { \dagger } = \mathrm { m a x } _ { k \in [ K ] } \nu _ { k }$ is exactly 0 (each cluster’s covariance is exactly low-rank) or nearly 0, i.e., data within clusters concentrates on each thin subspace, then the corresponding � is exactly or nearly 0, and one can choose a very small �. If � is large, i.e., clusters no longer have a low-dimensional structure, then one instead obtains an error bound depending on � linearly without relying on such a requirement for �.

## 3.6. Overview of the Proof of Theorem 2

We briefly outline the main arguments for the proof of Theorem 2 and highlight how the phase transition analysis is employed to obtain the intrinsic-dimension $( M ^ { \dagger } )$ scaling in the time-discretisation error, even when the number of clusters � grows polynomially with �.

The proof consists of five main steps. The first step in Section 5.1 uses standard arguments [Benton et al., 2024, Conforti et al., 2025, Huang et al., 2026] to bound KL $, ( \overrightarrow { \mathbb { P } } _ { \varepsilon } | | \overleftarrow { \mathbb { Q } } _ { t _ { N } } )$ as the sum of three terms that correspond to the error from approximating the score, the error in how we initiate the denoiser and the time-discretisation error. The main novelty in our proof is how we bound the time-discretisation error, which is covered in Steps 2 to 4 and outlined below. Step 5 then collects together all the error terms, and obtains the simplified bound presented in (23).

In Step 2 (see Section 5.2) we provide an upper bound on the time-discretisation error involving the following term:

$$
\tau \sum _ { j = 1 } ^ { N - 1 } \left( \mathrm { S N R } _ { T - t _ { j + 1 } } - \mathrm { S N R } _ { T - t _ { j } } \right) \cdot \mathbb { E } \Big [ \mathrm { t r } \big ( \Sigma _ { T - t _ { j } } \big ( \overrightarrow { X } _ { T - t _ { j } } \big ) \big ) \Big ] ,
$$

where $\Sigma _ { T - t } ( y ) = \mathrm { C o v } [ \overrightarrow { X } _ { 0 } \vert \overrightarrow { X } _ { T - t } = y ] , ( t , y ) \in [ 0 , T ) \times \mathbb { R } ^ { D }$ , is the posterior covariance. Under the GMM framework, the posterior covariance can be decomposed as shown by the following lemma.

## Lemma 3.4. It holds that for all $( t , y ) \in [ 0 , T ) \times \mathbb { R } ^ { D } , \Sigma _ { T - t } ( y ) = \Sigma _ { T - t } ^ { I } ( y ) + \Sigma _ { T - t } ^ { I I } ( y )$ with

$$
\begin{array} { l } { { \displaystyle \Sigma _ { T - t } ^ { I } ( y ) = \sum _ { k = 1 } ^ { K } W _ { D } ^ { [ k ] } ( t , y ) \sigma _ { T - t } ^ { 2 } \Sigma _ { k } ( S _ { T - t } ^ { k } ) ^ { - 1 } ; } } \\ { { \displaystyle \Sigma _ { T - t } ^ { I I } ( y ) = \frac { 1 } { 2 } \sum _ { j , k \in [ K ] } W _ { D } ^ { [ j ] } ( t , y ) W _ { D } ^ { [ k ] } ( t , y ) \big ( \mu ^ { [ j ] } ( t , y ) - \mu ^ { [ k ] } ( t , y ) \big ) ^ { \otimes 2 } , } } \end{array}\tag{24}
$$

where we have set

$$
\mu ^ { [ j ] } ( t , y ) = \mathbb { E } \big [ \overrightarrow { X } _ { 0 } \big | \overrightarrow { X } _ { T - t } = y , \overrightarrow { X } _ { 0 } \sim P _ { \mathrm { d a t a } } ^ { [ j ] } \big ] = \frac { \sigma _ { T - t } ^ { 2 } } { m _ { T - t } } \nabla \log \overrightarrow { p } _ { T - t } ^ { [ j ] } ( y ) + \frac { y } { m _ { T - t } } .\tag{25}
$$

The proof of Lemma 3.4 is provided in Appendix E.1. The term $\Sigma _ { T - i } ^ { I }$ captures the local within-cluster geometry, whereas $\Sigma _ { T - } ^ { I I }$ reflects the between-cluster geometry and may therefore scale with � through diferences between components.

In Section 5.3 (Step 3), we show the upper bound on the term $\mathbb { E } \big [ \mathrm { t r } \big ( \Sigma _ { T - t _ { j } } ^ { I } ( \overrightarrow { X } _ { T - t _ { j } } ) \big ) \big ]$ depends linearly on $M ^ { \dag }$ but is independent of �. Section 5.4 (Step 4) deals with the inter-class term $\mathbb { E } \big [ \mathrm { t r } \big ( \Sigma _ { T - t _ { i } } ^ { I I } ( \overrightarrow { X } _ { T - t _ { j } } ) \big ) \big ]$ by exploiting the phase transition. Roughly, we divide the time interval $[ 0 , T - \varepsilon ]$ into two regimes according to whether $\mathrm { S N R } _ { T - t _ { i } } \leq L \bigl ( \log ( K D ) \bigr ) / D$ for an arbitrary fixed constant $L > 0$ . During the first regime (mixing phase), the small SNR ofsets the mismatches in the first and second moments that can grow polynomially with �. For the latter regime (cluster commitment phase), Theorem 1 implies that, conditional on a trajectory terminating in a cluster $k ,$ at least one of the two posterior weights $W _ { D } ^ { [ i ] } ( t _ { j } , \vec { X } _ { T - t _ { j } } ) , W _ { D } ^ { [ j ] } ( t _ { j } , \vec { X } _ { T - t _ { j } } ) , i \ne j ,$ is exponentially small in � with high probability, thereby suppressing the inter-class diference. Furthermore, the dependence on � is shown to be logarithmic via a finite exponential-moment bound for the maximum over the Gaussian components.

## 4. Numerical Experiments of Posterior Commitments in Real Data

So far, we have studied the intrinsic dimension adaptivity of difusion models for target distributions formulated as a union of distinct (linear) manifolds within a Gaussian mixture framework. The key ingredient for showing such adaptivity in the KL error analysis (Theorem 2) is that the scale of the SNR for posterior weights concentration is approximately $\Theta ( 1 / D )$ , as shown in Theorem 1 under cluster separation. This section provides empirical evidence that this SNR scale at posterior concentration can also be observed for high-dimensional data beyond the GMM setting. All technical details of the implementation can be found in Appendix F, and the code to reproduce the experiments is available at https://github.com/YugaIgu/diffusion\_models\_phase\_transition

## 4.1. Set-Up

We consider two types of high-dimensional data, which are typical examples of the application of difusion models: (a) Single cell RNA sequencing (scRNA-seq) of peripheral blood mononuclear cells (PBMCs) with 32738 genes and (b) Image data (STL-10) consisting of a resolution of 96 by 96 pixels with 3 colour channels (RGB). For dataset (a), we vary the ambient dimension � by retaining the top $D \in \{ 5 0 0 , 1 0 0 0 , 2 0 0 0 , 4 0 0 0 , 8 0 0 0 \}$ highly variable genes (HVGs). For the STL-10 image data, we construct lower-resolution versions to obtain three resolutions: $D = 3 2 \times 3 2 \times 3 , D = 6 4 \times 6 4 \times 3$ and $D = 9 6 \times 9 6 \times 3$ (raw data), by using bilinear interpolation, implemented in PyTorch. Then, we select three classes (labels), specifically, C<sub>PBMC</sub> = {CD4 T, CD14+ Monocytes, CD19+ B} and $C _ { \mathrm { i m a g e } } = \{ { \mathrm { d o g } } , { \mathrm { c a t } } .$ airplane} for datasets (a) and (b).

We simulate the denoising SDE to sample from the target clusters. We directly use the score of the noised empirical distribution rather than a trained neural-network score, in the spirit of the exact-empirical-score setting considered in Biroli et al. [2024]. This allows us to isolate the dynamical phase transition from the uncertainty due to score estimation error. We perform the following four steps to observe the empirical SNR at the posterior weight concentration:

I. Simulating trajectories. We set the terminal time as $T = 6$ and simulate � = 500 independent backward trajectories $\{ \bar { X } _ { t _ { j } } ^ { [ k ] } \} _ { 1 \leq j \leq n , 1 \leq k \leq M }$ using the Euler-Maruyama discretisation with a uniform time-step $h = t _ { j + 1 } - t _ { j } = 0 . 0 0 3$ and so the number of steps $n = T / h = 2 0 0 0$ . These are used as numerical approximations for the exact backward process.

II. Labelling. For each trajectory, we assign a label of the class from which the path eventually samples, which is obtained as the argument of the maximum of the posterior weight (3): $C ^ { [ k ] } \equiv$ $\mathrm { a r g m a x } _ { c \in C } W _ { D } ^ { [ c ] } ( t _ { J } , \bar { X } _ { t _ { J } } ^ { [ k ] } ) , k = 1 , \ldots , M$ , where $J = n - 1 0$ (to avoid numerical instability) and C is C<sub>PBMC</sub> or C<sub>image</sub>.

III. Sorting. Those paths are sorted within the three distinct classes as $\mathbb { X } ^ { c } = \Big \{ \Big \{ \bar { X } _ { t _ { j } } ^ { [ k ] } \Big \} _ { j = 1 , \dots , J } | C ^ { [ k ] } =$ $c , k = 1 , \ldots , M \} , c \in C$ , that is, the set of paths eventually sampling from the class �.

IV. Recording. We then record the values of three types of posterior weights $W _ { D } ^ { [ c ] } , c \in C$ , evaluated with the trajectories contained in $\mathbb { X } ^ { c ^ { \dagger } }$ for some target class $c ^ { \dagger } \in C$ at each time-step $j = 1 , \dots , J .$

In the above procedure, the prior weights are assumed to be equal across the classes, and the (intractable) posterior weights are estimated by plugging an empirical approximation of the forward noised density into the definition of the weight (3).

## 4.2. Results

We first present the data geometry for the datasets (a) and (b) in Tables 2 and 3. Table 2 indicates that for the STL-10 datasets the 90% efective rank is stable for varying ambient dimension �, while the maximum eigenvalue $\lambda _ { \mathrm { m a x } }$ seems to grow approximately linearly with �. On the other hand, $\lambda _ { \mathrm { m a x } }$ for PBMC data exhibits $O ( D ^ { \alpha } )$ growth with $\alpha < 1$ . We observe from Table 3 that the mean separation of STL-10 datasets scales linearly with � and that covariance separations are very clear. The pairwise geometry for PBMC data also exhibits a similar separation growth as � increases – presented in Appendix F.

Table 2. Empirical covariance geometry for PBMC and STL-10 across ambient dimensions. For each class, we report the maximum covariance eigenvalue $\left( \lambda _ { \operatorname* { m a x } } \right)$ , the trace $( \operatorname { t r } ( { \widehat { \Sigma } } ) )$ , and the 90% efective rank $( r _ { \mathrm { e f f } , 9 0 } )$ , computed from the empirical data.  
PBMC
<table><tr><td>Class</td><td>D</td><td> $\lambda _ { \mathrm { m a x } }$ </td><td> $\operatorname { t r } ( { \widehat { \Sigma } } )$ </td><td> $r _ { \mathrm { e f f } , 9 0 }$ </td></tr><tr><td rowspan="6">CD4 T</td><td>500</td><td>3.42</td><td>469.88</td><td>313</td></tr><tr><td>1000</td><td>5.41</td><td>943.59</td><td>507</td></tr><tr><td>2000</td><td>9.85</td><td>1807.21</td><td>686</td></tr><tr><td>4000</td><td>16.30</td><td>3600.29</td><td>817</td></tr><tr><td>8000</td><td>25.72</td><td>7465.22</td><td>903</td></tr><tr><td>500</td><td>8.41</td><td>485.49</td><td>209</td></tr><tr><td rowspan="6">CD14+ Mono.</td><td>1000</td><td>14.14</td><td>985.57</td><td>289</td></tr><tr><td>2000</td><td>23.32</td><td>2057.65</td><td>339</td></tr><tr><td>4000</td><td>36.64</td><td>4001.79</td><td>372</td></tr><tr><td>8000</td><td>57.85</td><td>7781.14</td><td>395</td></tr><tr><td>500</td><td>9.03</td><td>554.19</td><td>182</td></tr><tr><td>1000</td><td>13.03</td><td>1050.05</td><td>229</td></tr><tr><td rowspan="3">CD19+B</td><td>2000</td><td>22.68</td><td>2177.58</td><td>254</td></tr><tr><td>4000</td><td>37.03</td><td>4723.49</td><td>272</td></tr><tr><td>8000</td><td>55.90</td><td>9230.85</td><td>285</td></tr></table>

STL-10
<table><tr><td>Class</td><td>D</td><td> $\lambda _ { \mathrm { m a x } }$ </td><td> $\operatorname { t r } ( { \widehat { \Sigma } } )$ </td><td> $r _ { \mathrm { e f f } , 9 0 }$ </td></tr><tr><td rowspan="3">Dog</td><td>3072</td><td>160.69</td><td>783.73</td><td>145</td></tr><tr><td>12288</td><td>638.36</td><td>3022.54</td><td>130</td></tr><tr><td>27648</td><td>1437.94</td><td>7061.82</td><td>154</td></tr><tr><td rowspan="3">Cat</td><td>3072</td><td>132.58</td><td>700.49</td><td>150</td></tr><tr><td>12288</td><td>531.07</td><td>2679.13</td><td>134</td></tr><tr><td>27648</td><td>1195.06</td><td>6308.94</td><td>163</td></tr><tr><td rowspan="3">Airplane</td><td>3072</td><td>193.60</td><td>756.55</td><td>105</td></tr><tr><td>12288</td><td>778.14</td><td>2920.58</td><td>97</td></tr><tr><td>27648</td><td>1750.48</td><td>6823.12</td><td>119</td></tr></table>

Table 3. Pairwise data geometry of the STL-10 clusters across image resolutions. For each pair and ambient dimension �, we report the squared mean separation, the maximum mean diference projection defined in (16), and the squared Frobenius covariance separation, computed from empirical data.
<table><tr><td>Pair</td><td>D</td><td> $\| \widehat { \mu } _ { i } - \widehat { \mu } _ { j } \| ^ { 2 }$ </td><td> $M _ { \mu }$ </td><td> $\| \widehat { \Sigma } _ { i } - \widehat { \Sigma } _ { j } \| _ { F } ^ { 2 }$ </td></tr><tr><td rowspan="4">Dog-Cat</td><td>3072</td><td>9.20</td><td>2.09</td><td> $4 . 0 3 \times 1 0 ^ { 3 }$ </td></tr><tr><td>12288</td><td>35.10</td><td>4.10</td><td> $5 . 9 8 \times 1 0 ^ { 4 }$ </td></tr><tr><td>27648</td><td>80.46</td><td>6.17</td><td> $3 . 1 9 \times 1 0 ^ { 5 }$ </td></tr><tr><td>3072</td><td>232.68</td><td>12.81</td><td> $1 . 3 5 \times 1 0 ^ { 4 }$ </td></tr><tr><td rowspan="3">Cat-Airplane</td><td>12288</td><td>928.48</td><td>25.59</td><td> $2 . 1 2 \times 1 0 ^ { 5 }$ </td></tr><tr><td>27648</td><td>2091.26</td><td>38.39</td><td> $1 . 0 9 \times 1 0 ^ { 6 }$ </td></tr><tr><td>3072</td><td>189.10</td><td>10.74</td><td> $1 . 1 8 \times 1 0 ^ { 4 }$ </td></tr><tr><td rowspan="2">Dog-Airplane</td><td>12288</td><td>753.39</td><td>21.46</td><td> $1 . 8 5 \times 1 0 ^ { 5 }$ </td></tr><tr><td>27648</td><td>1697.45</td><td>32.19</td><td> $9 . 5 5 \times 1 0 ^ { 5 }$ </td></tr></table>

Figure 1 demonstrates the dynamical concentration or decay of the posterior weights, evaluated for trajectories committing to a particular class, in the case of (a) PBMC data. Note that the reverse process proceeds from the left (low SNR) to the right (high SNR). The lines labelled with a target class $c ^ { \dagger } \in C$ (‘class name’ (target) in the legend) indicate the lower envelope of the posterior probability $W _ { D } ^ { [ c ^ { \dagger } ] }$ achieved by 99% of the paths in $\mathbb { X } ^ { c ^ { \dagger } }$ , while the lines with competing classes $c \in C \setminus \{ c ^ { \dagger } \}$ indicate the upper envelope of $W _ { D } ^ { [ { \boldsymbol { \hat { c } } } ] }$ by 99% of the paths in $\mathbb { X } ^ { c ^ { \dagger } }$ . As expected, for trajectories labelled with the chosen target cluster $c ^ { \dagger }$ (e.g., ‘CD4 T’), the posterior weight $W _ { D } ^ { [ c ^ { \dagger } ] }$ eventually concentrates on 1 as the SNR increases (as the physical time � approaches the terminal �) while the others $W _ { D } ^ { [ c ] }$ (e.g., ‘CD14+ Monocyte’ and $\mathbf { \hat { C D 1 9 + B ^ { \prime } } } )$ degenerate to 0. More importantly, these plots showcase the rapid posterior concentration, or the phase transition, for each ambient dimension $D _ { ; }$ , and its timing and corresponding SNR shifts earlier as � increases. In particular, the sudden decay of non-target posterior weights, followed by the sharp rise of the target posterior weight, is interpreted as a sign that the KL signal between the corresponding noisy clusters begins to dominate the fluctuation terms in the log-posterior ratio.

![](images/c34d5789aff6cefaae422fb950a5b67812bddcb6b21f69c92b7d058a3576f035.jpg)

![](images/83fb166e4b919d36751d068c54f84144dbafd597a2f7d4e0ff6065491dc54918.jpg)

![](images/d5c56e4b7d455d8e5b50b99bf8f0dcab79f3a14885564c7ccd28832e4bae3875.jpg)  
Figure 1. Posterior probability dynamics in the PBMC experiment. We construct increasing-dimensional representations by retaining the top-� highly variable genes, with � ∈ {500, 1000, 2000, 4000, 8000}. The posterior concentration timing shifts to lower SNR as � increases.

In Figure 2, we show the dynamical behaviour of the posterior probability for the image datasets with a fixed ambient dimension $D = 9 6 \times 9 6 \times 3$ , but now with plots of the KL divergence between the noised target cluster $c ^ { \dagger }$ and the competing class $c \in C \setminus \{ c ^ { \dagger } \}$ , precisely, KL $\cdot ( \vec { p } _ { T - t _ { i } } ^ { [ c ^ { \dagger } ] } | | \vec { p } _ { T - t _ { i } } ^ { [ c ] } ) , j = 1 , \ldots , n $ The KL divergence is not available in closed form and is therefore estimated from empirical data. The left panel of Figure 2 shows that, for trajectories eventually committing to the Dog class, the posterior weight of Airplane decays earlier than that of Cat. This behaviour is also reflected in the noised KL curves: the noised KL divergence between Dog and Airplane is already larger than that between Dog and Cat in the early-SNR phase. This also agrees with the pairwise class separation presented in Table 3. Thus, the reverse trajectories committing to Dog first rule out the Airplane class, while the finer ambiguity between Dog and Cat persists until a larger SNR. By contrast, in the right panel of Figure 2, for trajectories committing to Airplane, the KL curves against Dog and Cat are much closer to each other. Correspondingly, the Dog and Cat posterior weights decay in a more symmetric manner. These observations are consistent with the phase-transition analysis in Section 3, where the decay of non-target posterior weights is shown to be governed by the KL divergence between noised clusters.

Finally, Figure 3 shows the mean and standard deviation of the first SNR level after which the lower 1% quantile of the target posterior weight $W _ { D } ^ { [ c ^ { \dag } ] }$ , evaluated among trajectories eventually assigned to the target cluster $c ^ { \dagger }$ , remains above a threshold 0.99 for all subsequent time steps. This indicates that for at least 99% of the trajectories, the posterior weights permanently exceed the threshold after that time. The mean and standard deviation are computed over five independent repetitions of the four-step procedure explained in Section 4.1. The metric provides an empirical proxy for the phase transition timing analysed in Section 3. In both the STL-10 and PBMC experiments, the SNR at the phase transition decreases as the ambient dimension � increases, exhibiting an empirical ${ \cal O } ( 1 / D )$ scaling, in agreement with the posterior-commitment scaling predicted by the theory for GMMs (Theorem 1) up to logarithmic factors.

![](images/57ab13710370b627a13b299f0f917b65affbb473b3fefe96ddca58eb2a4fc759.jpg)

![](images/656ea8fe2132c7cddf0035ed82eae3466d39486b3ad93c2cfd9f4fc68c9d25c3.jpg)  
Figure 2. Posterior commitment in the STL-10 experiment. Posterior probabilities are evaluated using the reverse trajectories, grouped by their final committed class. Solid curves show the posterior probabilities of the three classes, while dashed curves show the corresponding noised KL divergences from the target class to competing classes.

![](images/57b77073351a68585d7f50ae4a3b2f4b901aa0f4e2ddacfd0302996adfb51aa4.jpg)  
(a) PBMC single-cell data.

![](images/940600081b73a9cef1bc0c526d24164cabf8065ab140cf7f5b2aade84dc58d3e.jpg)  
(b) STL-10 image data.  
Figure 3. Average empirical SNR at the posterior probability concentration over five repetitions of experiments, with the error bar showing their standard deviations. The plots show the SNR at which the lower 1 % quantile of the posterior weight among trajectories assigned to a given cluster permanently exceeds 0.99. In both STL-10 and PBMC experiments, the transition SNR exhibits an approximately $O ( 1 / D )$ decay.

## 5. Proof of Theorem 2

We write the measures of path $\overleftarrow { X }$ and $\overleftarrow { Y }$ over time interval $\mathbb { T } = [ a , b ] , 0 < a < b$ , as $\left. _ { \mathbb { T } } \right.$ and , respectively, and $\mathrm { S N R } _ { j } \equiv m _ { T - t _ { i } } ^ { 2 } / \sigma _ { T - t _ { i } } ^ { 2 } , \ j = 0 , 1 , \ldots , N$ , for ease of notation.

## 5.1. Step 1: Error Decomposition of KL Divergence

We first decompose the KL divergence into three sources of error by following the arguments in convergence analysis for difusion models [Benton et al., 2024, Conforti et al., 2025, Huang et al., 2026]. First, the data processing inequality and the equivalence of measures $\begin{array} { r } { \stackrel { \right. } { \mathbb { P } } _ { \varepsilon } \equiv \stackrel { \left. } { \mathbb { P } } _ { T - \varepsilon } } \end{array}$ lead to

$$
\begin{array} { r } { { \mathrm { K L } } \big ( \overrightarrow { \mathbb { P } } _ { \varepsilon } | | \overleftarrow { \mathbb { Q } } _ { t _ { N } } \big ) \leq { \mathrm { K L } } \big ( \overleftarrow { \mathbb { P } } _ { [ 0 , T - \varepsilon ] } \big | | \overleftarrow { \mathbb { Q } } _ { [ 0 , T - \varepsilon ] } \big ) . } \end{array}
$$

Further, employing the arguments in [Benton et al., 2024, Section 3.3], together with the use of Girsanov’s theorem (see e.g., [Benton et al., 2024, Appendix F] or [Chen et al., 2023b, Section 5.2]), we get $\ K \mathbf { L } \big ( \overleftarrow { \mathbb { P } } _ { [ 0 , T - \varepsilon ] } | | \overleftarrow { \mathbb { Q } } _ { [ 0 , T - \varepsilon ] } \big ) = R _ { 1 } + R _ { 2 }$ with

$$
R _ { 1 } \equiv { \bf K L } \big ( \overrightarrow { \mathbb { P } } _ { T } | | { \cal N } ( 0 , I _ { D } ) \big ) , \quad R _ { 2 } \equiv \sum _ { j = 0 } ^ { N - 1 } \int _ { t _ { j } } ^ { t _ { j + 1 } } \frac { m _ { T - t } ^ { 2 } } { \sigma _ { T - t } ^ { 4 } } \times \mathbb { E } \big [ \big \| \mu ( T - t , \overleftarrow { X } _ { t } ) - \hat { \mu } ( T - t _ { j } , \overleftarrow { X } _ { t _ { j } } ) \big \| ^ { 2 } \big ] d t ,
$$

where we recall $\mu ( t , x ) = \mathbb { E } [ \vec { X } _ { 0 } | \vec { X } _ { t } = x ]$ . For the first term $R _ { 1 }$ , following the proof arguments of [Chen et al., 2023a, Lemma 9] or [Benton et al., 2024, Proposition 9], we have:

$$
R _ { 1 } \lesssim ( D + \mathbb { E } _ { P _ { \mathrm { d a t a } } } \big [ \| X \| ^ { 2 } \big ] ) e ^ { - 2 T } .\tag{26}
$$

The second term $R _ { 2 }$ is further decomposed as $R _ { 2 } \ s \ R _ { 2 } ^ { I } + R _ { 2 } ^ { I I }$ , with

$$
R _ { 2 } ^ { I } = \sum _ { j = 0 } ^ { N - 1 } \int _ { t _ { j } } ^ { t _ { j + 1 } } \frac { m _ { T - t } ^ { 2 } } { \sigma _ { T - t } ^ { 4 } } \times \mathbb { E } \big [ \big \lVert \hat { \mu } ( T - t _ { j } , \overleftarrow { X } _ { t _ { j } } ) - \mu ( T - t _ { j } , \overleftarrow { X } _ { t _ { j } } ) \big \rVert ^ { 2 } \big ] d t ;
$$

$$
R _ { 2 } ^ { I I } = \sum _ { j = 0 } ^ { N - 1 } \int _ { t _ { j } } ^ { t _ { j + 1 } } \frac { m _ { T - t } ^ { 2 } } { \sigma _ { T - t } ^ { 4 } } \times \mathbb { E } \big [ \big \lVert \mu ( T - t _ { j } , \overleftarrow { X } _ { t _ { j } } ) - \mu ( T - t , \overleftarrow { X } _ { t } ) \big \rVert ^ { 2 } \big ] d t .\tag{27}
$$

For the term $R _ { 2 } ^ { I }$ , we have from (22), Tweedie’s formula and Assumption 5 that

$$
R _ { 2 } ^ { I } \lesssim \sum _ { j = 0 } ^ { N - 1 } h _ { j + 1 } \times \mathbb { E } \big [ \big \| \hat { s } ( T - t _ { j } , \overleftarrow { X } _ { t _ { j } } ) - \nabla \log \overleftarrow { p } _ { T - t _ { j } } ( \overleftarrow { X } _ { t _ { j } } ) \big \| ^ { 2 } \big ] \leq T \varepsilon _ { \mathrm { s c o r e } } ^ { 2 } .\tag{28}
$$

where we also used $\frac { m _ { T - t } ^ { 2 } } { \sigma _ { T - t } ^ { 4 } } \cdot \frac { \sigma _ { T - t _ { j } } ^ { 4 } } { m _ { T - t _ { j } } ^ { 2 } } \leq \frac { m _ { T - t _ { j + 1 } } ^ { 2 } } { \sigma _ { T - t _ { j + 1 } } ^ { 4 } } \cdot \frac { \sigma _ { T - t _ { j } } ^ { 4 } } { m _ { T - t _ { j } } ^ { 2 } } \leq 4 \exp ( 2 \tau )$ from Lemma A.1.

## 5.2. Step 2. Decomposition of the Time-Discretisation Error

Our main focus is thus on bounding the term $R _ { 2 } ^ { I I }$ in (27). We exploit the following result established in the literature. The proof can be found, e.g., in [Huang et al., 2026, Lemma 9].

Lemma 5.1. Let $j \in [ N ]$ . It then holds that for any $t \in \left[ t _ { j - 1 } , t _ { j } \right]$

$$
\begin{array} { r } { \mathbb { E } \Big [ \big \| \mu ( T - t _ { j - 1 } , \overleftarrow { X } _ { t _ { j - 1 } } ) - \mu ( T - t , \overleftarrow { X } _ { t } ) \big \| ^ { 2 } \Big ] = \mathbb { E } \Big [ \mathrm { t r } \big ( \sum _ { T - t _ { j - 1 } } ( \overleftarrow { X } _ { t _ { j - 1 } } ) \big ) \Big ] - \mathbb { E } \Big [ \mathrm { t r } \big ( \Sigma _ { T - t } ( \overleftarrow { X } _ { t } ) \big ) \Big ] . } \end{array}
$$

Lemma 5.1 implies that $t \mapsto \mathbb { E } \Big | \mathrm { t r } \big ( \Sigma _ { T - t } ( \overleftarrow { X } _ { t } ) \big ) \Big |$ is a non-increasing function, thus

$$
\mathbb { E } \Big [ \big \| \mu ( T - t _ { j - 1 } , \overleftarrow { X } _ { t _ { j - 1 } } ) - \mu ( T - t , \overleftarrow { X } _ { t } ) \big \| ^ { 2 } \Big ] \leq \mathbb { E } \Big [ \mathbf { t r } \big ( \sum _ { T - t _ { j - 1 } } ( \overleftarrow { X } _ { t _ { j - 1 } } ) \big ) \Big ] - \mathbb { E } \Big [ \mathbf { t r } \big ( \sum _ { T - t _ { j } } ( \overleftarrow { X } _ { t _ { j } } ) \big ) \Big ] ,
$$

for any $t \in \left[ t _ { j - 1 } , t _ { j } \right]$ . Recall $h _ { j + 1 } \leq \tau \sigma _ { T - t _ { j } } ^ { 2 }$ where the equality holds for $j = 0 , 1 , \ldots , N - 2$ . Then, we

apply Lemma 5.1 to the term $R _ { 2 } ^ { I I }$ and obtain

$$
\begin{array} { r l } & { R _ { 2 } ^ { I I } \leq \displaystyle \sum _ { j = 0 } ^ { N - 1 } h _ { j + 1 } \frac { m _ { T - t _ { j + 1 } } ^ { 2 } } { \sigma _ { T - t _ { j + 1 } } ^ { 4 } } \cdot \left( \mathbb { E } \big [ \mathrm { t r } \big ( \Sigma _ { T - t _ { j } } ( \overleftarrow \boldsymbol X _ { t _ { j } } ) \big ) \big ] - \mathbb { E } \big [ \mathrm { t r } \big ( \Sigma _ { T - t _ { j + 1 } } ( \overleftarrow \boldsymbol X _ { t _ { j + 1 } } ) \big ) \big ] \right) } \\ & { \qquad \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } } \\ &  { \leq 2 \tau \displaystyle \sum _ { j = 0 } ^ { N - 1 } \mathrm { S N R } _ { j + 1 } \left( \mathbb { E } \big [ \mathrm { t r } \big ( \Sigma _ { T - t _ { j } } ( \overleftarrow X _ { t _ { j } } ) \big ) \big ] - \mathbb { E } \big [ \mathrm { t r } \big ( \Sigma _ { T - t _ { j + 1 } } ( \overleftarrow X _ { t _ { j + 1 } } ) \big ) \big ] \right) } \\ & { \qquad \mathrm { ~ } } \\ & { \lesssim \tau \mathrm { S N R } _ { 1 } \cdot \mathbb { E } \big [ \mathrm { t r } \big ( \Sigma _ { T } ( \overrightarrow X _ { T } ) \big ) \big ] + \tau \displaystyle \sum _ { j = 1 } ^ { N - 1 } \big ( \mathrm { S N R } _ { j + 1 } - \mathrm { S N R } _ { j } \big ) \cdot \mathbb { E } \big [ \mathrm { t r } \big ( \Sigma _ { T - t _ { j } } ( \overrightarrow X _ { T - t _ { j } } ) \big ) \big ] \equiv \mathcal { T } , } \end{array}\tag{29}
$$

where in the second line we used the bound $\sigma _ { T - t _ { j } } ^ { 2 } / \sigma _ { T - t _ { j + 1 } } ^ { 2 } \leq 2$ , which is presented in Lemma A.1. For the ease of notation, we write:

$$
\Delta _ { j } = ( { \bf S } { \bf N } { \bf R } _ { j + 1 } - { \bf S } { \bf N } { \bf R } _ { j } ) , \quad Z _ { j } = ( t _ { j } , \overrightarrow { X } _ { T - t _ { j } } ) , \qquad j = 0 , \ldots , N - 1 .\tag{30}
$$

Using the closed-form expression of posterior covariance (24) in Lemma 3.4, we decompose the term $\mathcal { T }$ as $\mathcal { T } = \mathcal { T } _ { 1 } + \mathcal { T } _ { 2 }$ , where

$$
\mathcal { T } _ { 1 } = \tau \sum _ { j = 0 } ^ { N - 1 } \Bigl ( \mathrm { S N R } _ { 1 } \mathbf { 1 } _ { j = 0 } + \Delta _ { j } \mathbf { 1 } _ { j \geq 1 } \Bigr ) \sum _ { k = 1 } ^ { K } \mathbb { E } \bigl [ W _ { D } ^ { [ k ] } ( Z _ { j } ) \bigr ] \cdot \sigma _ { T - t _ { j } } ^ { 2 } \mathrm { t r } \Bigl ( \sum _ { k } \bigl ( m _ { T - t _ { j } } ^ { 2 } \Sigma _ { k } + \sigma _ { T - t _ { j } } ^ { 2 } I _ { D } \bigr ) ^ { - 1 } \Bigr ) ;\tag{31}
$$

$$
\mathcal { T } _ { 2 } = \frac { \tau } { 2 } \sum _ { j = 0 } ^ { N - 1 } \biggl ( \mathrm { S N R } _ { 1 } \mathbf { 1 } _ { j = 0 } + \Delta _ { j } \mathbf { 1 } _ { j \geq 1 } \biggr ) \sum _ { \boldsymbol { k } _ { 1 } , \boldsymbol { k } _ { 2 } = 1 \atop k _ { 1 } \neq k _ { 2 } } ^ { K } \mathbb { E } \biggl [ W _ { D } ^ { [ k _ { 1 } ] } ( Z _ { j } ) W _ { D } ^ { [ k _ { 2 } ] } ( Z _ { j } ) \bigl \| \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ) - \mu ^ { [ k _ { 2 } ] } ( Z _ { j } ) \bigr \| ^ { 2 } \biggr ]\tag{32}
$$

where we recall that fo $\mathfrak { r } \ z \equiv ( t , x ) \in [ 0 , T ) \times \mathbb { R } ^ { D } , \mu ^ { [ i ] } ( z ) = \mathbb { E } \big [ \ \overrightarrow { X } _ { 0 } \big | \overrightarrow { X } _ { T - t } = x , \ \overrightarrow { X } _ { 0 } \sim P _ { \mathrm { d a t a } } ^ { [ i ] } \big ] , \ i \in [ K ] .$

## 5.3. Step 3. Bounding the Term T<sub>1</sub>

The eigen-decomposition $\Sigma _ { k } \equiv U \Lambda _ { k } U ^ { \top }$ together with Assumption 2 gives that: for $j = 0 , 1 , \ldots , N - 1$

$$
\begin{array} { r l } & { \sigma _ { T - t _ { j } } ^ { 2 } \mathrm { t r } \left( \Sigma _ { k } \left( m _ { T - t _ { j } } ^ { 2 } \Sigma _ { k } + \sigma _ { T - t _ { j } } ^ { 2 } I _ { D } \right) ^ { - 1 } \right) = \sigma _ { T - t _ { j } } ^ { 2 } \mathrm { t r } \left( \Lambda _ { k } \left( m _ { T - t _ { j } } ^ { 2 } \Lambda _ { k } + \sigma _ { T - t _ { j } } ^ { 2 } I _ { D } \right) ^ { - 1 } \right) } \\ & { \qquad = \sigma _ { T - t _ { j } } ^ { 2 } \displaystyle \sum _ { \ell = 1 } ^ { D } \frac { \lambda _ { \ell } ( \Sigma _ { k } ) } { m _ { T - t _ { j } } ^ { 2 } \lambda _ { \ell } ( \Sigma _ { k } ) + \sigma _ { T - t _ { j } } ^ { 2 } } \leq \frac { \sigma _ { T - t _ { j } } ^ { 2 } } { m _ { T - t _ { j } } ^ { 2 } } M _ { k } + \nu _ { k } . } \end{array}\tag{33}
$$

Thus, we get

$$
\begin{array} { r l } & { \mathcal { T } _ { 1 } \leq \tau \mathrm { S N R } _ { 1 } \displaystyle \sum _ { k = 1 } ^ { K } \mathbb { E } \big [ W _ { D } ^ { [ k ] } ( Z _ { 0 } ) \big ] \cdot \Big ( \frac { \sigma _ { T } ^ { 2 } } { m _ { T } ^ { 2 } } M _ { k } + \nu _ { k } \Big ) + \tau \displaystyle \sum _ { j = 1 } ^ { N - 1 } \Delta _ { j } \displaystyle \sum _ { k = 1 } ^ { K } \mathbb { E } \big [ W _ { D } ^ { [ k ] } ( Z _ { j } ) \big ] \cdot \Big ( \frac { \sigma _ { T - t _ { j } } ^ { 2 } } { m _ { T - t _ { j } } ^ { 2 } } M _ { k } + \nu _ { k } \Big ) } \\ & { \quad \leq \tau \Big ( \frac { \mathrm { S N R } _ { 1 } } { \mathrm { S N R } _ { 0 } } + \mathrm { S N R } _ { 1 } R \Big ) M ^ { \dagger } + \tau \displaystyle \sum _ { j = 1 } ^ { N - 1 } \frac { \Delta _ { j } } { \mathrm { S N R } _ { j } } \big ( 1 + \mathrm { S N R } _ { j } R \big ) M ^ { \dagger } } \\ & { \quad = \tau \Big ( \frac { \mathrm { S N R } _ { 1 } } { \mathrm { S N R } _ { 0 } } + \displaystyle \sum _ { j = 1 } ^ { N - 1 } \frac { \Delta _ { j } } { \mathrm { S N R } _ { j } } \Big ) M ^ { \dagger } + \tau \mathrm { S N R } _ { N } R , } \end{array}
$$

where $M ^ { \dagger } = \mathrm { m a x } _ { k \in [ K ] } M _ { k } , R \in [ 0 , 1 )$ is a universal constant defined in Assumption 2 and we used $\begin{array} { r } { \sum _ { k = 1 } ^ { K } W _ { D } ^ { [ k ] } ( t , x ) = 1 } \end{array}$ for all $( t , x ) \in [ 0 , T ) \times \mathbb { R } ^ { D }$ . We then get under $\varepsilon \gtrsim R / T$ that

$$
\begin{array} { r } { \mathcal { T } _ { 1 } \lesssim \tau \Big ( 1 + \log \frac { \mathrm { S N R } _ { N } } { \mathrm { S N R } _ { 1 } } \Big ) M ^ { \dagger } + \tau \frac { R } { \varepsilon } \lesssim \tau \Big ( \log \frac { 1 } { \varepsilon } + T \Big ) M ^ { \dagger } , } \end{array}\tag{34}
$$

since it follows that for $j \in [ N - 1 ]$ ，

$$
\frac { \Delta _ { j } } { \mathrm { S N R } _ { j } } = \frac { \mathrm { S N R } _ { j + 1 } } { \mathrm { S N R } _ { j } } \int _ { \mathrm { S N R } _ { j } } ^ { \mathrm { S N R } _ { j + 1 } } \frac { 1 } { \mathrm { S N R } _ { j + 1 } } d y \leq \frac { \mathrm { S N R } _ { j + 1 } } { \mathrm { S N R } _ { j } } \int _ { \mathrm { S N R } _ { j } } ^ { \mathrm { S N R } _ { j + 1 } } \frac { 1 } { y } d y \lesssim \log \frac { \mathrm { S N R } _ { j + 1 } } { \mathrm { S N R } _ { j } }\tag{35}
$$

with $\mathrm { S N R } _ { k + 1 } / \mathrm { S N R } _ { k } \leq 4 , k = 0 , 1 , \ldots , N - 1$ from Lemma A.1 and

$$
\begin{array} { r } { \log \frac { 1 } { \mathrm { S N R } _ { 1 } } \leq - \log m _ { T - t _ { 1 } } ^ { 2 } = 2 ( T - t _ { 1 } ) \lesssim T , } \end{array}
$$

$$
\begin{array} { r } { \log \operatorname { S N R } _ { N } = \log \Bigl ( \frac { 1 } { \exp ( 2 \varepsilon ) - 1 } \Bigr ) \leq \log \frac { 1 } { \varepsilon } , \quad \bigl ( \because \exp ( x ) - 1 \geq x , \forall x > 0 \bigr ) . } \end{array}
$$

## 5.4. Step 4. Bounding the Term $\mathcal { T } _ { 2 }$

We hereby focus on bounding the term involving the inter-class diference between clusters, i.e., $\left\| \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ) - \mu ^ { [ k _ { 2 } ] } ( Z _ { j } ) \right\| ^ { 2 }$ that can depend on $\| \mu _ { i } - \mu _ { j } \| ^ { 2 } = O ( D )$ under Assumption 3. To avoid the polynomial dependence on � in the bound, we make use of the phase transition from mixing to single-cluster commitment, as argued in Section 3. We thus separate the time interval $[ 0 , T - \varepsilon ]$ into two parts and get $\mathcal { T } _ { 2 } = \mathcal { T } _ { 2 } ^ { I } + \mathcal { T } _ { 2 } ^ { I I }$ with

$$
\mathcal { T } _ { 2 } ^ { I } = \frac { \tau } { 2 } \sum _ { j = 0 } ^ { N _ { 1 } - 1 } \left( \mathrm { S N R } _ { 1 } \mathbf { 1 } _ { j = 0 } + \Delta _ { j } \mathbf { 1 } _ { j \geq 1 } \right) \sum _ { k _ { 1 } , k _ { 2 } = 1 } ^ { K } \mathbb { E } \Big [ W _ { D } ^ { [ k _ { 1 } ] } ( Z _ { j } ) W _ { D } ^ { [ k _ { 2 } ] } ( Z _ { j } ) \big \| \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ) - \mu ^ { [ k _ { 2 } ] } ( Z _ { j } ) \big \| ^ { 2 } \Big ] ;\tag{36}
$$

$$
\mathcal { T } _ { 2 } ^ { I I } = \frac { \tau } { 2 } \sum _ { j = N _ { 1 } } ^ { N - 1 } \Delta _ { j } \sum _ { k _ { 1 } , k _ { 2 } = 1 } ^ { K } \mathbb { E } \Big [ W _ { D } ^ { [ k _ { 1 } ] } ( Z _ { j } ) W _ { D } ^ { [ k _ { 2 } ] } ( Z _ { j } ) \big \lVert \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ) - \mu ^ { [ k _ { 2 } ] } ( Z _ { j } ) \big \rVert ^ { 2 } \Big ] ,\tag{37}
$$

where in the above we have set:

$$
\begin{array} { r } { N _ { 1 } \equiv \operatorname* { i n f } \left\{ j \in \left[ N \right] \vert \mathrm { S N R } _ { j } > L \frac { \log \left( K D \right) } { D } \right\} } \end{array}\tag{38}
$$

for an arbitrary fixed constant $L > 0$ satisfying $L \log ( K D ) / D \geq \underline { { S } } .$ , with � specified in the statement of Theorem 1. We note that $N _ { 1 } \geq 1$ under $\begin{array} { r } { T \geq \frac { 1 } { \mathit { \varepsilon } } } \end{array}$ log � due to Corollary 1 and $\mathrm { S N R } _ { j } \ \lesssim \ \frac { \log ( K D ) } { D }$ for $j \in \{ 1 , \dots , N _ { 1 } - 1 \}$ . In what follows, we write $Z _ { j } ^ { [ i ] } = ( t _ { j } , \overrightarrow { X } _ { T - t _ { j } } ^ { [ i ] } ) , j \in \{ 0 , 1 , . . . , N - 1 \} , i \in [ K ]$ where $\overrightarrow { X } _ { T - t _ { j } } ^ { [ i ] }$ follows the distribution $\vec { p } _ { T - t _ { j } } ^ { [ i ] }$

## 5.4.1. Bounding the term $\mathcal { T } _ { 2 } ^ { I }$

We begin with the term $\mathcal { T } _ { 2 } ^ { I }$ , that is, the error during the mixing phase. We then get that

$$
\mathcal { T } _ { 2 } ^ { I } = \frac { \tau } { 2 } \sum _ { j = 0 } ^ { N _ { 1 } - 1 } \left( \mathrm { S N R } _ { 1 } \mathbf { 1 } _ { j = 0 } + \Delta _ { j } \mathbf { 1 } _ { j \geq 0 } \right) \sum _ { i \in [ K ] } \sum _ { k _ { 1 } , k _ { 2 } = 1 } \pi _ { i } \cdot Q _ { k _ { 1 } , k _ { 2 } } ^ { [ i ] } ( j )
$$

with

$$
Q _ { k _ { 1 } , k _ { 2 } } ^ { [ i ] } ( j ) = \mathbb { E } \Big [ W _ { D } ^ { [ k _ { 1 } ] } ( Z _ { j } ^ { [ i ] } ) W _ { D } ^ { [ k _ { 2 } ] } ( Z _ { j } ^ { [ i ] } ) \big \| \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ^ { [ i ] } ) - \mu ^ { [ k _ { 2 } ] } ( Z _ { j } ^ { [ i ] } ) \big \| ^ { 2 } \Big ] ,
$$

where the expectation operator acts on the second component of the variable $Z _ { j } ^ { [ i ] }$ under the law of $\vec { X } _ { T - t _ { j } } ^ { [ i ] }$ We have that

$$
\begin{array} { r l r l } & { \underset { k _ { 1 } , k _ { 2 } \in [ K ] } { \sum } Q _ { k _ { 1 } , k _ { 2 } } ^ { [ i ] } ( j ) \leq \mathbb { E } \underset { k _ { 1 } , k _ { 2 } \in [ K ] } { \operatorname* { m a x } } \| \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ^ { [ i ] } ) - \mu ^ { [ k _ { 2 } ] } ( Z _ { j } ^ { [ i ] } ) \| ^ { 2 } ] } & & { ( : \sum _ { k \in [ K ] } W _ { D } ^ { [ k ] } ( Z _ { j } ^ { [ i ] } ) = 1 ) } \\ & { = \frac { 1 } { \xi _ { j } } \log \exp ( \mathbb { E } [ \xi _ { j } \cdot \underset { k _ { 1 } , k _ { 2 } \in [ K ] } { \operatorname* { m a x } } ] \| \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ^ { [ i ] } ) - \mu ^ { [ k _ { 2 } ] } ( Z _ { j } ^ { [ i ] } ) \| ^ { 2 } ] ) } \\ & { \leq \frac { 1 } { \xi _ { j } } \log ( \mathbb { E } [ \exp \Bigl ( \xi _ { j } \underset { k _ { 1 } , k _ { 2 } \in [ K ] } { \operatorname* { m a x } } \Bigr \| \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ^ { [ i ] } ) - \mu ^ { [ k _ { 2 } ] } ( Z _ { j } ^ { [ i ] } ) \rVert ^ { 2 } \Bigr ) ] ) } & & { ( : \mathrm { : ~ J e n s e n ' s i n e q u a l i t y } ) } \\ & { = \frac { 1 } { \xi _ { j } } \log ( \mathbb { E } [ \underset { k _ { 1 } , k _ { 2 } \in [ K ] } { \operatorname* { m a x } } \exp \Bigl ( \xi _ { j } \big \| \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ^ { [ i ] } ) - \mu ^ { [ k _ { 2 } ] } ( Z _ { j } ^ { [ i ] } ) \big \| ^ { 2 } ) ] ) } \end{array}\tag{39}
$$

for any $\xi _ { j } > 0$ and for each fixed $i \in [ K ]$ and $j \in \{ 0 , 1 , \ldots , N _ { 1 } - 1 \}$ . To further bound the above term, we exploit the following result whose proof is postponed to Appendix E.2:

Lemma 5.2. Under Assumptions 2, 3, we have the following: for any $i \in [ K ] , j \in \{ 0 , 1 , \ldots , N - 1 \}$ and any $\rho \in \left( 0 , m _ { T - t _ { i } } ^ { 2 } / ( 1 6 \sigma _ { T - t _ { i } } ^ { 4 } \operatorname* { m a x } _ { i , k \in [ K ] } \Vert H _ { k , i } ( j ) \Vert _ { 2 } ) \right]$ , it holds

$$
\log \biggl ( \mathbb { E } \biggl [ \operatorname* { m a x } _ { k _ { 1 } , k _ { 2 } \in [ K ] } \exp \Bigl ( \rho \bigl \| \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ^ { [ i ] } ) - \mu ^ { [ k _ { 2 } ] } ( Z _ { j } ^ { [ i ] } ) \bigr \| ^ { 2 } \Bigr ) \biggr ] \biggr ) \lesssim \log K + \rho \bigl ( D + \mathrm { S N R } _ { j } M ^ { \dagger } D ^ { 2 \alpha } \bigr )
$$

where we have set

$$
{ \cal H } _ { k , i } ( j ) = m _ { T - t _ { j } } ^ { 4 } \{ ( S _ { T - t _ { j } } ^ { k } ) ^ { - 1 } ( \Sigma _ { k } - \Sigma _ { i } ) ( S _ { T - t _ { j } } ^ { i } ) ^ { - 1 } ( \Sigma _ { k } - \Sigma _ { i } ) ( S _ { T - t _ { j } } ^ { k } ) ^ { - 1 } \} .\tag{40}
$$

Setting $\begin{array} { r } { \xi _ { j } = m _ { T - t _ { j } } ^ { 2 } / ( 1 6 \sigma _ { T - t _ { j } } ^ { 4 } \operatorname* { m a x } _ { i , k \in [ K ] } \Vert H _ { k , i } ( j ) \Vert _ { 2 } ) } \end{array}$ in (39), we apply Lemma 5.2 to get

$$
\begin{array} { r l } { \mathcal { T } _ { 2 } ^ { I } \lesssim \tau \frac { \mathrm { S N R } _ { 1 } } { \xi _ { 0 } } \big ( \log K + \xi _ { 0 } ( D + \mathrm { S N R } _ { j } M ^ { \dagger } D ^ { 2 \alpha } ) \big ) + \tau \displaystyle \sum _ { j = 1 } ^ { N _ { 1 } - 1 } \frac { \Delta _ { j } } { \xi _ { j } } \big ( \log K + \xi _ { j } ( D + \mathrm { S N R } _ { j } M ^ { \dagger } D ^ { 2 \alpha } ) \big ) } & { } \\ { \lesssim \tau \Big ( \mathrm { S N R } _ { 1 } + \displaystyle \sum _ { j = 1 } ^ { N _ { 1 } - 1 } \Delta _ { j } \Big ) \Big ( ( \log K ) \mathrm { S N R } _ { N _ { 1 } - 1 } D ^ { 2 \alpha } + D + \mathrm { S N R } _ { N _ { 1 } - 1 } M ^ { \dagger } D ^ { 2 \alpha } \Big ) } & { } \\ { = \tau \mathrm { S N R } _ { N _ { 1 } } \Big ( \mathrm { S N R } _ { N _ { 1 } - 1 } \big ( \log K + M ^ { \dagger } \big ) D ^ { 2 \alpha } + D \Big ) } & { } \\ { \lesssim \tau \mathrm { S N R } _ { N _ { 1 } - 1 } \Big ( \mathrm { S N R } _ { N _ { 1 } - 1 } \big ( \log K + M ^ { \dagger } \big ) D ^ { 2 \alpha } + D \Big ) } & { \big ( \cdot \cdot \mathrm { L e m m a ~ A . 1 } \big ) } \\ { \lesssim \tau \Big ( \displaystyle \frac { \log ( K D ) } { D ^ { 1 - \alpha } } \Big ) ^ { 2 } \big ( \log K + M ^ { \dagger } \big ) + \log ( K D ) \Big ] \quad } & { \big ( \cdot \cdot \mathrm { S N R } _ { N _ { 1 } - 1 } \lesssim \frac { \log ( K D ) } { D } \big ) } \end{array}\tag{41}
$$

where in the second line we used

$$
\begin{array} { r } { \frac { 1 } { \xi _ { j } } \lesssim \frac { \sigma _ { T - t _ { j } } ^ { 4 } } { m _ { T - t _ { j } } ^ { 2 } } \operatorname* { m a x } _ { i , k \in [ K ] } \| H _ { k , i } \| _ { 2 } \lesssim \mathrm { S N R } _ { j } D ^ { 2 \alpha } \lesssim \mathrm { S N R } _ { N _ { 1 } - 1 } D ^ { 2 \alpha } . } \end{array}\tag{42}
$$

## 5.4.2. Bounding the term $\mathcal { T } _ { 2 } ^ { I I }$

We then consider the error during the cluster commitment phase, that is, $\mathcal { T } _ { 2 } ^ { I I }$ defined in (37). In this phase, we can exploit the high-probability concentration of the posterior weight stated in Theorem 1.

Introducing an event set $E _ { j } ^ { [ i ] } ( \eta ) \subseteq \Omega , \eta \in ( 0 , 1 )$ as

$$
E _ { j } ^ { [ i ] } ( \eta ) = \Big \{ \omega \in \Omega \ \big | \bigcap _ { k \in [ K ] \backslash \{ i \} } \big \{ W _ { D } ^ { [ k ] } ( Z _ { j } ^ { [ i ] } ) \leq \exp \big ( { - \eta \mathrm { K L } \big ( \overrightarrow { p } _ { T - t _ { j } } ^ { [ i ] } \mid \mid \overrightarrow { p } _ { T - t _ { j } } ^ { [ k ] } \big ) } \big ) \big \} \Big \} ,
$$

we express the term $\mathcal { T } _ { 2 } ^ { I I }$ as $\mathcal { T } _ { 2 } ^ { I I } = \mathcal { T } _ { 2 } ^ { I I , E } + \mathcal { T } _ { 2 } ^ { I I , E ^ { c } }$ with

$$
\mathcal { T } _ { 2 } ^ { I I , E } = \textstyle \frac { \tau } { 2 } \sum _ { j = N _ { 1 } } ^ { N - 1 } \Delta _ { j } \sum _ { \mathfrak { i } \in [ K ] } \sum _ { k _ { 1 } , k _ { 2 } \in [ K ] } \pi _ { \mathfrak { i } } \cdot Q _ { j } ^ { [ \mathfrak { i } ] , k _ { 1 } , k _ { 2 } } \big ( E _ { j } ^ { [ \mathfrak { i } ] } ( \eta ) \big ) ;\tag{43}
$$

$$
\mathcal { T } _ { 2 } ^ { I I , E ^ { c } } = \textstyle { \frac { \tau } { 2 } } \sum _ { j = N _ { 1 } } ^ { N - 1 } \Delta _ { j } \sum _ { i \in [ K ] } \sum _ { k _ { 1 } , k _ { 2 } \in [ K ] } \pi _ { i } \cdot Q _ { j } ^ { [ i ] , k _ { 1 } , k _ { 2 } } \big ( E _ { j } ^ { [ i ] } ( \eta ) ^ { c } \big ) ,\tag{44}
$$

where we have set: for an event $A \subseteq \Omega$

$$
Q _ { j } ^ { [ i ] , k _ { 1 } , k _ { 2 } } ( A ) = \mathbb { E } \Big [ W _ { D } ^ { [ k _ { 1 } ] } ( Z _ { j } ^ { [ i ] } ) W _ { D } ^ { [ k _ { 2 } ] } ( Z _ { j } ^ { [ i ] } ) \left. \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ^ { [ i ] } ) - \mu ^ { [ k _ { 2 } ] } ( Z _ { j } ^ { [ i ] } ) \right. ^ { 2 } \mathbf { 1 } _ { A } \Big ] .
$$

For notational simplicity, we write:

$$
D _ { k } ^ { [ i ] } ( j ) = \mathbb { E } \bigg [ \big \| \mu ^ { [ k ] } ( Z _ { j } ^ { [ i ] } ) - \mu ^ { [ i ] } ( Z _ { j } ^ { [ i ] } ) \big \| ^ { 2 } \bigg ] , \qquad k \in \{ k _ { 1 } , k _ { 2 } \} .
$$

Then, for the term $\mathcal { T } _ { 2 } ^ { I I , E }$ , we have the following bound on the event set $E _ { j } ^ { [ i ] } ( \eta )$

$$
\begin{array} { r } { Q _ { j } ^ { [ i ] , k _ { 1 } , k _ { 2 } } ( E _ { j } ^ { [ i ] } ( \eta ) ) \lesssim \exp \Bigl ( - \eta \Bigl \{ \mathrm { K L } \bigl ( \vec { p } _ { T - t _ { j } } ^ { [ i ] } \bigr ) \bigl | \stackrel {  } { p } _ { T - t _ { j } } ^ { [ k _ { 1 } ] } \bigr ) + \mathrm { K L } \bigl ( \vec { p } _ { T - t _ { j } } ^ { [ i ] } \bigr | \bigr | \stackrel {  } { p } _ { T - t _ { j } } ^ { [ k _ { 2 } ] } \bigr ) \Bigr \} \cdot \bigl \{ D _ { k _ { 1 } } ^ { [ i ] } ( j ) + D _ { k _ { 2 } } ^ { [ i ] } ( j ) \bigr \} } \\ { \lesssim \bigl ( \bigl \| \Sigma _ { i } \bigr \| _ { 2 } + \mathrm { S N R } _ { j } ^ { - 1 } \bigr ) \exp \Bigl ( - \eta \Bigl \{ \mathrm { K L } \bigl ( \vec { p } _ { T - t _ { j } } ^ { [ i ] } \bigr | \stackrel {  } { p } _ { T - t _ { j } } ^ { [ k _ { 1 } ] } \bigr ) + \mathrm { K L } \bigl ( \vec { p } _ { T - t _ { j } } ^ { [ i ] } \bigr | \stackrel {  } { p } _ { T - t _ { j } } ^ { [ k _ { 2 } ] } \bigr ) \Bigr \} \Bigr ) } \\ { \times \Bigl \{ \mathrm { K L } \bigl ( \vec { p } _ { T - t _ { j } } ^ { [ i ] } \bigr | \stackrel {  } { p } _ { T - t _ { j } } ^ { [ i ] } \bigr ) + \mathrm { K L } \bigl ( \vec { p } _ { T - t _ { j } } ^ { [ i ] } \bigr | \stackrel {  } { p } _ { T - t _ { j } } ^ { [ k _ { 2 } ] } \bigr ) \Bigr \} } \\  \lesssim \bigl ( \bigl \| \Sigma _ { i } \bigr \| _ { 2 } + \mathrm { S N R } _ { j } ^ { - 1 } \bigr ) \exp \Bigl ( - \tilde { \eta } \cdot \bigl \{ \mathrm { K L } \bigl ( \vec { p } _  T - \end{array}
$$

for some $\widetilde { \eta } \in ( 0 , \eta )$ and ${ \widetilde { C } = C _ { 3 } / C _ { 5 } }$ , where in the second line above we used the following result whose proof is postponed to Appendix E.3:

Lemma 5.3. It holds that:

$$
D _ { k } ^ { [ i ] } ( j ) \leq 4 \big ( \| \Sigma _ { i } \| _ { 2 } + \mathrm { S N R } _ { j } ^ { - 1 } \big ) \cdot \mathrm { K L } \big ( \overrightarrow { p } _ { T - t _ { j } } ^ { [ i ] } \vert \vert \overrightarrow { p } _ { T - t _ { j } } ^ { [ k ] } \big ) .\tag{45}
$$

To further bound the term $Q _ { j } ^ { [ i ] , k _ { 1 } , k _ { 2 } } ( E _ { j } ^ { [ i ] } ( \eta ) )$ under the case $j \in \{ N _ { 1 } , \ldots , N - 1 \}$ , we define an integer $N _ { 2 }$ as

$$
N _ { 2 } = \left\{ \begin{array} { l l } { \operatorname* { i n f } \big \{ j = N _ { 1 } , N _ { 1 } + 1 , \dots , N | \mathrm { ~ S N R } _ { j } > D / \operatorname* { m a x } _ { k \in [ K ] } \mathrm { t r } ( \Sigma _ { k } ) \big \} , } & { \mathrm { i f ~ t h e ~ s e t ~ i s ~ n o t ~ e m p t y } ; } \\ { N , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{46}
$$

We then consider the following two cases for $j \colon \left( \mathrm { a } \right)$ $j = N _ { 1 } , \ldots , N _ { 2 } - 1$ and (b). $j = N _ { 2 } , \ldots , N - 1$ where case (b) is understood to be empty when $N _ { 2 } ~ = ~ N$ . For case (a), we have $\mathrm { S N R } _ { j } \| \Sigma _ { i } \| _ { 2 } ~ \leq$

SNR $_ j \operatorname* { m a x } _ { k \in [ K ] } \operatorname { t r } ( \Sigma _ { k } ) \leq D$ and thus

$$
\begin{array} { r l } & { Q _ { j } ^ { [ i ] , k _ { 1 } , k _ { 2 } } ( E _ { j } ^ { [ i ] } ( \eta ) ) \lesssim \mathrm { S N R } _ { j } ^ { - 1 } ( D + 1 ) \exp \Bigl ( - \frac { \tilde { \eta } \tilde { C } ^ { 2 } } { 2 } \mathrm { S N R } _ { j } D \Bigr ) } \\ & { \qquad \le \mathrm { S N R } _ { j } ^ { - 1 } \bigl ( D + 1 \bigr ) \cdot ( K D ) ^ { - c L } , \qquad \bigl ( \because \mathrm { ~ S N R } _ { j } \ge L \log ( K D ) / D \bigr ) , } \end{array}\tag{47}
$$

where $c = \widetilde { \eta } \widetilde { C } ^ { 2 } / 2$ . On the other hand, under case (b), we have

$$
\begin{array} { r l } & { Q _ { j } ^ { [ i ] , k _ { 1 } , k _ { 2 } } ( E _ { j } ^ { [ i ] } ( \eta ) ) } \\ & { \lesssim \mathrm { S N R } _ { j } ^ { - 1 } \big ( \mathrm { S N R } _ { j } D ^ { \alpha } + 1 \big ) \exp \Big ( { - \frac { \widetilde \eta C ^ { 2 } } { 2 } \cdot \frac { D ^ { 2 } } { \operatorname* { m a x } _ { k \in [ K ] } \mathrm { t r } ( \Sigma _ { k } ) } } \Big ) \qquad \big ( \cdot \cdot \mathrm { A s s u m p t i o n } 2 \big ) } \\ & { \lesssim \mathrm { S N R } _ { j } ^ { - 1 } \cdot \frac { D ^ { \alpha } } { \varepsilon } \cdot \exp \Big ( { - \frac { \widetilde \eta C ^ { 2 } } { 2 } \cdot \frac { D ^ { 2 } } { M ^ { \dagger } ( C _ { 2 } D ^ { \alpha } + R ) } } \Big ) \quad \big ( \cdot \cdot \mathrm { A s s u m p t i o n } 2 \& \mathrm { S N R } _ { \varepsilon } \lesssim \frac { 1 } { \varepsilon } \big ) } \\ & { \le \mathrm { S N R } _ { j } ^ { - 1 } \cdot \frac { D ^ { \alpha } } { \varepsilon } \cdot \exp \Big ( { - \frac { \widetilde \eta \widetilde \zeta ^ { 2 } } { 4 M ^ { \dagger } C _ { 2 } } \cdot D ^ { 2 - \alpha } } \Big ) \lesssim \mathrm { S N R } _ { j } ^ { - 1 } \cdot \frac { D ^ { 2 \alpha - 2 } } { \varepsilon } \cdot \exp \Big ( { - \widetilde { c } \cdot D ^ { 2 - \alpha } } \Big ) } \end{array}\tag{48}
$$

for some $\widetilde { c } \in ( 0 , \widetilde { \eta } \widetilde { C } ^ { 2 } / 4 M ^ { \dagger } C _ { 2 } )$ . Substituting the bounds (47) and (48) into (43), we obtain

$$
\begin{array} { r l r } {  { \mathcal { T } _ { 2 } ^ { I I , E } \lesssim \tau \Bigg [ K ^ { 2 } D ( K D ) ^ { - c L } \cdot \sum _ { j = N _ { 1 } } ^ { N _ { 2 } - 1 } \frac { \Delta _ { j } } { \mathrm { S N R } _ { j } } + \frac { K ^ { 2 } } { \varepsilon } \exp ( - \widetilde { c } D ^ { 2 - \alpha } ) \cdot \sum _ { j = N _ { 2 } } ^ { N - 1 } \frac { \Delta _ { j } } { \mathrm { S N R } _ { j } } \Bigg ] } } \\ & { } & { \lesssim \tau \Big [ \big ( \log \frac { 1 } { \varepsilon } + \log D \big ) \cdot \Big \{ K ^ { 2 } D ( K D ) ^ { - c L } + \frac { K ^ { 2 } } { \varepsilon } \exp ( - \widetilde { c } D ^ { 2 - \alpha } ) \Big \} \Big ] , } \end{array}\tag{49}
$$

where we used

$$
\begin{array} { r } { \operatorname* { m a x } \bigg \{ \displaystyle \sum _ { j = N _ { 1 } } ^ { N _ { 2 } - 1 } \frac { \Delta _ { j } } { { \mathrm { S N R } _ { j } } } , \sum _ { j = N _ { 2 } } ^ { N - 1 } \frac { \Delta _ { j } } { { \mathrm { S N R } _ { j } } } \bigg \} \leq \displaystyle \sum _ { j = N _ { 1 } } ^ { N - 1 } \frac { \Delta _ { j } } { { \mathrm { S N R } _ { j } } } \leq \log \frac { 1 } { \varepsilon } + \log D . } \end{array}\tag{50}
$$

We now consider the term $\mathcal { T } _ { 2 } ^ { I I , E ^ { c } }$ defined in (44). For ease of notation, we write $B = ( E _ { j } ^ { [ i ] } ( \eta ) ) ^ { c }$ and $\mathbb { P } _ { 1 } = \overset { \vartriangle } { \vec { p } } _ { T - t _ { j } } ^ { [ i ] }$ , and introduce a probability measure $\mathbb { P } _ { 2 }$ defined as $\begin{array} { r } { \mathrm { d } \mathbb { P } _ { 2 } = \frac { \mathbf { 1 } _ { B } } { \mathbb { P } _ { 1 } \left( B \right) } \mathrm { d } \mathbb { P } _ { 1 } } \end{array}$ . We then get

$$
\begin{array} { r l } & { \displaystyle \sum _ { k _ { 1 } , k _ { 2 } = 1 } ^ { K } Q _ { j } ^ { [ i ] , k _ { 1 } , k _ { 2 } } ( B ) \le \mathbb { B } _ { \mathbb { P } _ { 1 } } \Big [ \operatorname* { m a x } _ { k _ { 1 } , k _ { 2 } \in [ K ] } \big \| \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ^ { [ i ] } ) - \mu ^ { [ k _ { 2 } ] } ( Z _ { j } ^ { [ i ] } ) \big \| ^ { 2 } \cdot \mathbf { 1 } _ { B } \Big ] } \\ & { \quad = \frac { \mathbb { P } _ { 1 } \big ( B \big ) } { \xi _ { j } } \cdot \mathbb { B } _ { \mathbb { P } _ { 2 } } \Big [ \xi _ { j } \operatorname* { m a x } _ { k _ { 1 } , k _ { 2 } \in [ K ] } \big \| \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ^ { [ i ] } ) - \mu ^ { [ k _ { 2 } ] } ( Z _ { j } ^ { [ i ] } ) \big \| ^ { 2 } \Big ] } \\ & { \quad \le \frac { \mathbb { P } _ { 1 } \big ( B \big ) } { \xi _ { j } } \cdot \Bigg \{ \log \mathbb { E } _ { \mathbb { P } _ { 1 } } \Big [ \exp \Big ( \xi _ { j } \operatorname* { m a x } _ { k _ { 1 } , k _ { 2 } \in [ K ] } \big \| \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ^ { [ i ] } ) - \mu ^ { [ k _ { 2 } ] } ( Z _ { j } ^ { [ i ] } ) \big \| ^ { 2 } \Big ) \Big ] + \mathrm { K L } \big ( \mathbb { P } _ { 2 } \mid \big \| \mathbb { P } _ { 1 } \big ) \Bigg \} , } \end{array}
$$

for $\xi _ { j } > 0$ such that $\begin{array} { r } { \mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ \exp \left( \xi _ { j } \operatorname* { m a x } _ { k _ { 1 } , k _ { 2 } \in [ K ] } \big \| \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ^ { [ i ] } ) - \mu ^ { [ k _ { 2 } ] } ( Z _ { j } ^ { [ i ] } ) \big \| ^ { 2 } \right) \right] < \infty } \end{array}$ , where the second bound is obtained via Donsker-Varadhan’s variational formula [Dupuis and Ellis, 1997]. We immediately have KL $\mathcal { \left( \mathbb { P } _ { 2 } \left. \right. \mathbb { P } _ { 1 } \right) } = - \log \mathbb { P } _ { 1 } ( B )$ . Also, Theorem 1 gives that for any $q \ge 1 , \eta \in ( 0 , 1 )$ there exists $D ^ { \dagger } \in \mathbb { N }$ such that for any $D \ \geq \ D ^ { \dagger }$ , we have $\mathbb { P } _ { 1 } ( B ) \le D ^ { - q }$ . Thus, by setting $\xi _ { j } \ : = \ :$

$$
m _ { T - t _ { j } } ^ { 2 } / ( 1 6 \sigma _ { T - t _ { j } } ^ { 4 } \mathrm { m a x } _ { i , k \in [ K ] } \| H _ { k , i } ( j ) \| _ { 2 } ) , \mathrm { w e ~ o b t a i n : }
$$

$$
\begin{array} { r l r } {  { \sum _ { k _ { 1 } , k _ { 2 } = 1 } ^ { K } Q _ { j } ^ { [ i ] , k _ { 1 } , k _ { 2 } } ( B ) \lesssim \mathbb { P } _ { 1 } ( B ) \cdot \Big \{ \frac { 1 } { \xi _ { j } } \log K + D + \mathrm { S N R } _ { j } M ^ { \dagger } D ^ { 2 \alpha } + \frac { 1 } { \xi _ { j } } \log \frac { 1 } { \mathbb { P } _ { 1 } ( B ) } \Big \} } } & { \big ( \cdot ; \mathrm { ~ L e m m a ~ 5 . 2 } \big ) } \\ & { } & { \lesssim D ^ { 2 \alpha - q } \cdot \mathrm { S N R } _ { j } \cdot \big ( \log K + M ^ { \dagger } \big ) + D ^ { 1 - q } + \mathrm { S N R } _ { j } D ^ { 2 \alpha } \mathbb { P } _ { 1 } ( B ) \log \frac { 1 } { \mathbb { P } _ { 1 } ( B ) } \quad \big ( \cdot ; ( 4 2 ) \big ) } \\ & { } & { \lesssim D ^ { 2 \alpha - q } \cdot \mathrm { S N R } _ { j } \cdot \big ( \log K + M ^ { \dagger } \big ) + D ^ { 1 - q } + D ^ { 2 \alpha - q } \cdot \mathrm { S N R } _ { j } \cdot \log D } \\ & { } & { \lesssim \mathrm { S N R } _ { j } ^ { - 1 } \big ( D ^ { 2 \alpha - q } \cdot \varepsilon ^ { - 2 } \cdot ( \log K + M ^ { \dagger } + \log D ) + D ^ { 1 - q } \cdot \varepsilon ^ { - 1 } \big ) , \qquad ( 5 1 ) } \end{array}
$$

where we used for � log $( 1 / x ) \leq r \log ( 1 / r ) x \in [ 0 , r ] , r \leq e ^ { - 1 }$ , and $\mathrm { S N R } _ { j } \lesssim \varepsilon ^ { - 1 }$ in the third and fourth inequality, respectively. Thus, the bound (51) together with (50) leads to

$$
\begin{array} { r l } & { \mathcal { T } _ { 2 } ^ { I I , E ^ { c } } \lesssim \tau \Big [ \big ( D ^ { 2 \alpha - q } \cdot \varepsilon ^ { - 2 } \cdot ( \log K + M ^ { \dagger } + \log D ) + D ^ { 1 - q } \cdot \varepsilon ^ { - 1 } \big ) \sum _ { j = N _ { 1 } } ^ { N } \frac { \Delta _ { j } } { \mathrm { S N R } _ { j } } \Big ] } \\ & { \qquad \lesssim \tau \Big [ \big ( D ^ { 2 \alpha - q } \cdot \varepsilon ^ { - 2 } \cdot ( \log K + M ^ { \dagger } + \log D ) + D ^ { 1 - q } \cdot \varepsilon ^ { - 1 } \big ) \big ( \log \frac { 1 } { \varepsilon } + \log D \big ) \Big ] . } \end{array}\tag{52}
$$

## 5.5. Step 5: Final Bound

Combining the upper bounds (34), (41), (49) and (52) into the term $R _ { 2 } ^ { I I }$ defined in (29), the timediscretisation error is bounded as follows:

$$
\begin{array} { r l } & { R _ { 2 } ^ { I I } \lesssim \mathcal { T } _ { 1 } + \mathcal { T } _ { 2 } ^ { I } + \mathcal { T } _ { 2 } ^ { I I , E } + \mathcal { T } _ { 2 } ^ { I I , E ^ { c } } } \\ & { \qquad \lesssim \tau \Big [ M ^ { \dagger } \Big ( \log \frac { 1 } { \varepsilon } + T \Big ) + \left( \frac { \log ( K D ) } { D ^ { 1 - \alpha } } \right) ^ { 2 } \left( \log K + M ^ { \dagger } \right) + \log ( K D ) } \\ & { \qquad + \left( \log \frac { 1 } { \varepsilon } + \log D \right) \cdot \left\{ K ^ { 2 } D ( K D ) ^ { - c L } + \frac { K ^ { 2 } } { \varepsilon } \exp ( - \widetilde { c } D ^ { 2 - \alpha } ) \right\} } \\ & { \qquad + \left( D ^ { 2 \alpha - q } \cdot \varepsilon ^ { - 2 } \cdot ( \log K + M ^ { \dagger } + \log D ) + D ^ { 1 - q } \cdot \varepsilon ^ { - 1 } \right) \left( \log \frac { 1 } { \varepsilon } + \log D \right) \Big ] . } \end{array}\tag{53}
$$

For the number mixture of � that grows polynomially in �, by choosing suficiently large $q \geq 2$ and $L > 0$ , we have under $\begin{array} { r } { T \geq \frac { 1 } { 2 } \log D } \end{array}$ that for any suficiently large $D$

$$
\begin{array} { r } { R _ { 2 } ^ { I I } \lesssim \tau M ^ { \dagger } \Bigl ( \log \frac { 1 } { \varepsilon } + T \Bigr ) . } \end{array}\tag{54}
$$

We thus conclude Theorem 2 from (26), (28) and (54).

## 6. Conclusion and Discussion

This work has analytically studied the geometric adaptivity of denoising difusion models when the target distributions to sample from are characterised as a union of well-separated classes with their own low-dimensional structure, motivated by the union of manifolds hypothesis [Brown et al., 2022]. By focusing on �-Gaussian mixture models, our analysis has focused on the dynamical behaviour of the score function and, in particular, the behaviour of the posterior weights for sampling from each cluster component. Under the conditions, particularly the $\Theta ( D )$ inter-class separation in cluster means, we have shown that the concentration of the posterior weights occurs with high probability after the SNR reaches a threshold $\Theta \big ( \log ( K D ) / D \big )$ . We then use this result to obtain a KL divergence error bound, where the time-discretisation error is shown to scale linearly with the intrinsic dimensionality rather than the ambient dimensionality �. The main diference between our analysis and the literature (e.g., Benton et al. [2024], Conforti et al. [2025]) is that our analysis accounts for the dynamical regimes of the denoising process. Specifically, during the mixing phase (before the posterior commitment), the small

SNR suppresses the efect of the inter-class separation of means and covariances in terms of $D ;$ whereas after the cluster commitment, when the SNR is no longer suficiently small, the posterior weight decay neutralises the large cluster separations. We also note that the literature on difusion models targeting GMMs [Li et al., 2026] establishes a dimension-free bound under shared isotropic covariances, while our work treats GMMs with heterogeneous anisotropic covariances and derives a desirable error bound by employing the phase transition. Our analysis exploits $\Theta ( D )$ separation in the mean vectors to induce the phase transition, with the key SNR scaling with �, but it could be extended to cases where covariance separation plays a central role in the transition.

The numerical experiments in Section 4 demonstrate the SNR scaling Θ(log �/�) in the phase transition for real high-dimensional datasets beyond our geometric framework (GMMs). Given this, our phase transition-driven error analysis can be applied to a wider class of mixture models where we believe similar intrinsic-dimension adaptivity would hold, provided we can still prove concentration results on the posterior weights and the score function for each mixture component is suficiently well-behaved. The former is likely to hold if the mixture components are sub-Gaussian, while the latter would apply under assumptions such as strong log-concavity of the cluster densities.

Another interesting extension would be to hierarchical clusters. For a mixture model with ‘supercomponents’, each of which is a Gaussian mixture model, with $O ( D )$ separation between the supercomponent means, and smaller separation, say ${ \cal O } ( 1 )$ , between the means of cluster components within each super-component. For this setting, we would still get concentration of the posterior weights to a super-component with high probability. Then, the KL divergence error is controlled by the intrinsic dimensionality in a similar manner, so that the relatively small separation within each super-component does not degrade the dimensional scaling in the error bound.

## Acknowledgement

We thank Prof Kenji Fukumizu and Prof Paul Jenkins for detailed comments and feedback on this work.   
The authors acknowledge support from EPSRC grant EP/Y028783/1.

## References

Beatrice Achilli, Marco Benedetti, Giulio Biroli, and Marc Mézard. Theory of speciation transitions in difusion models with general class structure. arXiv preprint arXiv:2602.04404, 2026.

Brian D.O. Anderson. Reverse-time difusion equation models. Stochastic Processes and their Applications, 12(3):313–326, May 1982. ISSN 0304-4149. doi: 10.1016/0304-4149(82)90051-5.

Vahan Arsenyan, Elen Vardanyan, and Arnak Dalalyan. Assessing the quality of denoising difusion models in Wasserstein distance: noisy score and optimal bounds. Advances in Neural Information Processing Systems, 38:19548–19591, 2026.

Joe Benton, Valentin De Bortoli, Arnaud Doucet, and George Deligiannidis. Nearly �-linear convergence bounds for difusion models via stochastic localization. In The Twelfth International Conference on Learning Representations, 2024.

Giulio Biroli, Tony Bonnaire, Valentin De Bortoli, and Marc Mézard. Dynamical regimes of difusion models. Nature Communications, 15(1):9957, 2024.

Tony Bonnaire, Raphaël Urfin, Giulio Biroli, and Marc Mezard. Why difusion models don’t memorize: The role of implicit dynamical regularization in training. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, pages 141266–141286. Curran Associates, Inc., 2025.

Miha Brešar and Aleksandar Mijatović. Nonasymptotic bounds for forward processes in denoising difusions: Ornstein–uhlenbeck is hard to beat. The Annals ofApplied Probability, 35(6):4439–4463, 2025.

Bradley CA Brown, Anthony L Caterini, Brendan Leigh Ross, Jesse C Cresswell, and Gabriel Loaiza-Ganem. Verifying the union of manifolds hypothesis for image data. arXiv preprint arXiv:2207.02862, 2022.

Hongrui Chen, Holden Lee, and Jianfeng Lu. Improved analysis of score-based generative modeling: User-friendly bounds under minimal smoothness assumptions. In International Conference on Machine Learning, pages 4735–4763. PMLR, 2023a.

Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, and Anru R Zhang. Sampling is as easy as learning the score: theory for difusion models with minimal data assumptions. International Conference on Learning Representations, 2023b.

Giovanni Conforti, Alain Durmus, and Marta Gentiloni Silveri. KL convergence guarantees for score difusion models under minimal data assumptions. SIAM Journal on Mathematics of Data Science, 7 (1):86–109, 2025.

Valentin De Bortoli. Convergence of denoising difusion models under the manifold hypothesis. Transactions on Machine Learning Research, 2022.

Prafulla Dhariwal and Alexander Nichol. Difusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34:8780–8794, 2021.

Benjamin Dupuis, Dario Shariatian, Maxime Haddouche, Alain Durmus, and Umut Simsekli. Algorithmand data-dependent generalization bounds for difusion models. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, pages 101897–101936. Curran Associates, Inc., 2025.

Paul Dupuis and Richard S. Ellis. A Weak Convergence Approach to the Theory of Large Deviations. Wiley Series in Probability and Statistics. John Wiley & Sons, New York, 1997. doi: 10.1002/9781118165904.

Xuefeng Gao, Hoang M Nguyen, and Lingjiong Zhu. Wasserstein convergence guarantees for a genera class of score-based generative models. Journal ofMachine Learning Research, 26(43):1–54, 2025.

U. G. Haussmann and E. Pardoux. Time reversal of difusions. The Annals ofProbability, 14(4), October 1986. ISSN 0091-1798. doi: 10.1214/aop/1176992362.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.

Daniel Hsu, Sham Kakade, and Tong Zhang. A tail inequality for quadratic forms of subGaussian random vectors. Electronic Communications in Probability, 17(none), January 2012. ISSN 1083-589X. doi: 10.1214/ecp.v17-2079. URL http://dx.doi.org/10.1214/ecp.v17-2079.

Zhihan Huang, Yuting Wei, and Yuxin Chen. Denoising difusion probabilistic models are optimally adaptive to unknown low dimensionality. Mathematics ofOperations Research, March 2026. ISSN 1526-5471. doi: 10.1287/moor.2024.0769.

Yuta Koike. Wasserstein bounds for denoising difusion probabilistic models via the Föllmer process. arXiv preprint arXiv:2605.18069, 2026.

Gen Li and Yuling Yan. Adapting to unknown low-dimensional structures in score-based difusion models. Advances in Neural Information Processing Systems, 37:126297–126331, 2024.

Gen Li and Yuling Yan. �(�/�) convergence theory for difusion probabilistic models under minimal assumptions. Journal ofMachine Learning Research, 26(292):1–55, 2025.

Gen Li, Changxiao Cai, and Yuting Wei. Dimension-free convergence of difusion models for approximate Gaussian mixtures. 2026.

Puheng Li, Zhong Li, Huishuai Zhang, and Jiang Bian. On the generalization properties of difusion models. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 2097–2127. Curran Associates, Inc., 2023.

Jiadong Liang, Zhihan Huang, and Yuxin Chen. Low-dimensional adaptation of difusion models: Convergence in total variation (extended abstract). In Nika Haghtalab and Ankur Moitra, editors, Proceedings ofThirty Eighth Conference on Learning Theory, volume 291 of Proceedings ofMachine Learning Research, pages 3723–3729. PMLR, 30 Jun–04 Jul 2025.

Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising difusion probabilistic models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11461–11471, 2022.

Antoine Maillard and Sebastian Goldt. Memorisation, convergence and generalisation in generative models, 2026.

Samet Oymak and Talha Cihad Gulcu. A theoretical characterization of semi-supervised learning with self-training for Gaussian mixture models. In International Conference on Artificial Intelligence and Statistics, pages 3601–3609. PMLR, 2021.

Kaare Brandt Petersen, Michael Syskind Pedersen, et al. The matrix cookbook. Technical University of Denmark, 7(15):510, 2008.

Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, and Mikhail Kudinov. Grad-tts: A difusion probabilistic model for text-to-speech. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8599–8608. PMLR, 18–24 Jul 2021.

Peter Potaptchik, Iskander Azangulov, and George Deligiannidis. Linear convergence of difusion models under the manifold hypothesis. In The Thirty Eighth Annual Conference on Learning Theory, pages 4668–4685. PMLR, 2025.

Rahul Satija, Jefrey A. Farrell, David Gennert, Alexander F. Schier, and Aviv Regev. Spatial reconstruction of single-cell gene expression data. Nature Biotechnology, 33(5):495–502, 2015.

Kulin Shah, Sitan Chen, and Adam Klivans. Learning mixtures of Gaussians using the DDPM objective. Advances in Neural Information Processing Systems, 36:19636–19649, 2023.

Marta Gentiloni Silveri and Antonio Ocello. Beyond log-concavity and score regularity: Improved convergence bounds for score-based generative models in W2-distance. In Forty-second International Conference on Machine Learning, 2025.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic diferential equations. arXiv preprint arXiv:2011.13456, 2020.

Tim Stuart, Andrew Butler, Paul Hofman, Christoph Hafemeister, Efthymia Papalexi, William M. Mauck, Yuhan Hao, Marlon Stoeckius, Peter Smibert, and Rahul Satija. Comprehensive integration of single-cell data. Cell, 177(7):1888–1902.e21, 2019.

Brian L. Trippe, Jason Yim, Doug Tischer, David Baker, Tamara Broderick, Regina Barzilay, and Tommi S. Jaakkola. Difusion probabilistic modeling of protein backbones in 3d for the motif-scafolding problem. In The Eleventh International Conference on Learning Representations, 2023.

Peng Wang, Huijie Zhang, Zekai Zhang, Siyi Chen, Yi Ma, and Qing Qu. Difusion Models Learn Low-Dimensional Distributions VIA Subspace Clustering. In 2025 IEEE 10th International Workshop on Computational Advances in Multi-Sensor Adaptive Processing (CAMSAP), page 211–215, 2025.

F. Alexander Wolf, Philipp Angerer, and Fabian J. Theis. SCANPY: large-scale single-cell gene expression data analysis. Genome Biology, 19(1):15, 2018.

Yuchen Wu, Minshuo Chen, Zihao Li, Mengdi Wang, and Yuting Wei. Theoretical insights for difusion guidance: A case study for Gaussian mixture models. In International Conference on Machine Learning, pages 53291–53327. PMLR, 2024.

Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. Difusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 56(4):1–39, 2023.

## Appendix

## A. Property of Time-Discretisation Schedule

To show Lemma 3.3, we first prove the following result, which is also key in the proof of Theorem 2:

Lemma A.1. For the time-discretisation schedule (20), it holds that:

$$
\begin{array} { r } { \frac { \sigma _ { T - t _ { j } } ^ { 2 } } { \sigma _ { T - t _ { j + 1 } } ^ { 2 } } \leq 2 , \quad \frac { \mathrm { S N R } _ { T - t _ { j + 1 } } } { \mathrm { S N R } _ { T - t _ { j } } } \leq 4 , \qquad j = 0 , \ldots , N - 1 . } \end{array}
$$

Proof. Noticing that $h _ { i + 1 } = \tau \sigma _ { T - t _ { i } } ^ { 2 } , i = 0 , 1 , \dots , N - 2$ and $h _ { N } \leq \tau \sigma _ { T - t _ { N - 1 } } ^ { 2 } .$ , we have that for all $j = 0 , 1 , \ldots , N - 1$

$$
\sigma _ { T - t _ { j } } ^ { 2 } - \sigma _ { T - t _ { j + 1 } } ^ { 2 } = \int _ { T - t _ { j + 1 } } ^ { T - t _ { j } } \frac { d } { d s } \sigma _ { s } ^ { 2 } d s \leq 2 h _ { j + 1 } \leq 2 \tau \sigma _ { T - t _ { j } } ^ { 2 } ,
$$

which leads to $\sigma _ { T - t _ { i } } ^ { 2 } / \sigma _ { T - t _ { i + 1 } } ^ { 2 } \ \le \ 1 / ( 1 - 2 \tau ) \ \le \ 2$ for $\tau \in ( 0 , 1 / 4 ]$ . For the latter inequality, using $m _ { T - t _ { i + 1 } } ^ { 2 } / m _ { T - t _ { i } } ^ { 2 } \leq \exp \bigl ( 2 h _ { j + 1 } \bigr ) \leq \exp ( 2 \tau )$ , we get

$$
\begin{array} { r } { \frac { \mathrm { S N R } _ { T - t _ { j + 1 } } } { \mathrm { S N R } _ { T - t _ { j } } } = \frac { m _ { T - t _ { j + 1 } } ^ { 2 } } { m _ { T - t _ { j } } ^ { 2 } } \cdot \frac { \sigma _ { T - t _ { j } } ^ { 2 } } { \sigma _ { T - t _ { j + 1 } } ^ { 2 } } \leq 2 \exp ( 2 \tau ) \leq 4 , } \end{array}
$$

which concludes the proof.

(ProofofLemma 3.3) We consider the term $\begin{array} { r } { T _ { j } \equiv \int _ { t _ { j } } ^ { t _ { j + 1 } } \frac { d s } { \sigma _ { T - s } ^ { 2 } } , j = 0 , 1 , . . . , N - 1 } \end{array}$ . Because $s \mapsto 1 / \sigma _ { T - s } ^ { 2 }$ is an increasing function, (20) and Lemma A.1 give that $\bar { T _ { j } } \le \tau \sigma _ { T - t _ { i } } ^ { 2 } / \sigma _ { T - t _ { i + 1 } } ^ { 2 } \le 2 \tau , \ j = 0 , 1 , \ldots , N - 1$ where for $j = N - 1$ we considered $h _ { N } \leq \tau \sigma _ { T - t _ { N - 1 } } ^ { 2 }$ . On the other hand, we have

$$
T _ { j } \ge \left\{ \begin{array} { l l } { \tau } & { j = 0 , 1 , \dots , N - 2 ; } \\ { 0 } & { j = N - 1 . } \end{array} \right.
$$

Taking summation, we get $\begin{array} { r } { \tau ( N - 1 ) \le \sum _ { j = 0 } ^ { N - 1 } T _ { j } \le 2 N } \end{array}$ · � and

$$
\sum _ { j = 0 } ^ { N - 1 } T _ { j } = \int _ { 0 } ^ { T - \varepsilon } \frac { d s } { \sigma _ { T - s } ^ { 2 } } = T - \varepsilon + \frac { 1 } { 2 } \log \frac { 1 - \exp ( - 2 T ) } { 1 - \exp ( - 2 \varepsilon ) } = \Theta \bigl ( T - 1 + \log \frac { 1 } { \varepsilon } \bigr ) ,
$$

which concludes Lemma 3.3.

## B. Proofs for Posterior Classification Analysis

## B.1. Proof of Lemma 3.1

The definition of posterior weight yields

$$
\begin{array} { r } { \log W _ { D } ^ { [ i ] } ( t , y ) = \log \frac { \pi _ { i } \cdot \overrightarrow { p } _ { T - t } ^ { [ i ] } ( y ) } { \sum _ { \ell \in [ K ] } \pi _ { \ell } \cdot \overrightarrow { p } _ { T - t } ^ { [ \ell ] } ( y ) } \leq \operatorname* { m i n } \Big \{ \log \frac { \pi _ { i } } { \pi _ { j } } + Z _ { t } ^ { ( i , j ) } ( y ) , 0 \Big \} , \quad y \in \mathbb { R } ^ { D } , j \in [ K ] , } \end{array}
$$

with $\begin{array} { r } { Z _ { t } ^ { ( i , j ) } ( y ) = \log \frac { N ( y ; m _ { T - t } \mu _ { i } , S _ { T - t } ^ { i } ) } { N ( y ; m _ { T - t } \mu _ { j } , S _ { T - t } ^ { j } ) } } \end{array}$ . The conditional random variable can be expressed as $\widetilde { Y } _ { T - t } ^ { [ j ] } \sim$ $N ( m _ { T - t } \mu _ { j } , S _ { T - t } ^ { j } )$ due to Law $\prime ( \overrightarrow { X } _ { t } ) = \mathrm { L a w } ( \overleftarrow { X } _ { T - t } )$ , where we recall $S _ { T - t } ^ { j }$ in (5). We then decompose $Z _ { t } ^ { ( i , j ) } ( \widetilde { Y } _ { T - t } ^ { [ j ] } ) = \Psi _ { 1 } + \Psi _ { 2 } + \Psi _ { 3 }$ where

$$
\Psi _ { 1 } \equiv - m _ { T - t } ( \widetilde { Y } _ { T - t } ^ { [ j ] } - m _ { T - t } \mu _ { j } ) ^ { \top } ( S _ { T - t } ^ { i } ) ^ { - 1 } ( \mu _ { j } - \mu _ { i } ) ;
$$

$$
\Psi _ { 2 } \equiv { \textstyle \frac { 1 } { 2 } } ( \widetilde { Y } _ { T - t } ^ { [ j ] } - m _ { T - t } \mu _ { j } ) ^ { \top } \big ( ( S _ { T - t } ^ { j } ) ^ { - 1 } - ( S _ { T - t } ^ { i } ) ^ { - 1 } \big ) ( \widetilde { Y } _ { T - t } ^ { [ j ] } - m _ { T - t } \mu _ { j } ) ;
$$

$$
\begin{array} { r } { \Psi _ { 3 } \equiv \frac { 1 } { 2 } \log \frac { \operatorname* { d e t } S _ { T - t } ^ { j } } { \operatorname* { d e t } S _ { T - t } ^ { i } } - \frac { m _ { T - t } ^ { 2 } } { 2 } ( \mu _ { j } - \mu _ { i } ) ^ { \top } ( S _ { T - t } ^ { i } ) ^ { - 1 } ( \mu _ { j } - \mu _ { i } ) . } \end{array}
$$

The first term follows a Gaussian distribution with mean 0 and variance given as

$$
\mathrm { V a r } [ \Psi _ { 1 } ] = m _ { T - t } ^ { 2 } ( \mu _ { j } - \mu _ { i } ) ^ { \top } ( S _ { T - t } ^ { i } ) ^ { - 1 } ( S _ { T - t } ^ { j } ) ( S _ { T - t } ^ { i } ) ^ { - 1 } ( \mu _ { j } - \mu _ { i } ) \equiv \mathcal { T } _ { D } ^ { [ i , j ] } ( t ) .
$$

Thus, the standard high probability argument for the Gaussian variable gives: for $\delta \in ( 0 , 1 )$ ,

$$
\begin{array} { r } { \mathbb { P } \Big ( | \Psi _ { 1 } | \geq \sqrt { 2 \mathrm { V a r } [ \Psi _ { 1 } ] \log \frac { 2 } { \delta } } \Big ) \leq \frac { \delta } { 2 } . } \end{array}
$$

The second term $\Psi _ { 2 }$ , quadratic $\mathbf { W . r . t }$ . the Gaussian $\widetilde { Y } _ { T - t } ^ { [ j ] } - m _ { T - t } \mu _ { j } \sim N ( 0 , S _ { T - t } ^ { j } )$ is expressed as $\Psi _ { 2 } = z ^ { \top } A z$ with

$$
A \equiv { \textstyle \frac { 1 } { 2 } } \big ( I _ { D } - ( S _ { T - t } ^ { j } ) ^ { 1 / 2 } ( S _ { T - t } ^ { i } ) ^ { - 1 } ( S _ { T - t } ^ { j } ) ^ { 1 / 2 } \big ) , \qquad z \sim N ( 0 , I _ { D } ) .
$$

Then, application of the concentration inequality [Hsu et al., 2012, Proposition 1] to $\Psi _ { 2 }$ yields: for $\delta \in ( 0 , 1 )$

$$
\begin{array} { r } { \mathbb { P } \Big ( \Psi _ { 2 } \geq \operatorname { t r } ( A ) + \sqrt { 4 \mathrm { t r } ( A ^ { 2 } ) \log \frac { 2 } { \delta } } + 2 \| A \| _ { 2 } \log \frac { 2 } { \delta } \Big ) \leq \frac { \delta } { 2 } . } \end{array}
$$

Thus with probability at least $1 - \delta .$ , we have:

$$
\begin{array} { r l } & { Z _ { t } ^ { ( i , j ) } ( \widetilde { Y } _ { T - t } ^ { [ j ] } ) \leq \Psi _ { 3 } + \mathrm { t r } ( A ) + \sqrt { 2 \mathrm { V a r } [ \Psi _ { 1 } ] \log \frac { 2 } { \delta } } + \sqrt { 2 \mathrm { V a r } [ \Psi _ { 2 } ] \log \frac { 2 } { \delta } } + 2 \| A \| _ { 2 } \log \frac { 2 } { \delta } } \\ & { = - \mathrm { K L } \big ( \overline { { \mathcal { P } } } _ { T - t } ^ { [ j ] } \mid \big | \ \overline { { \mathcal { P } } } _ { T - t } ^ { [ i ] } \big ) + \sqrt { 2 \mathcal { T } _ { D } ^ { [ i , j ] } ( t ) \log \frac { 2 } { \delta } } + \sqrt { 2 \mathcal { U } _ { D } ^ { [ i , j ] } ( t ) \log \frac { 2 } { \delta } } + 2 \mathcal { V } _ { D } ^ { [ i , j ] } ( t ) \log \frac { 2 } { \delta } , } \end{array}
$$

where we used: $\mathrm { V a r } [ \Psi _ { 2 } ] = 2 \mathrm { t r } ( A ^ { 2 } ) \equiv \mathcal { U } _ { D } ^ { [ i , j ] } ( t )$ and

$$
\Psi _ { 3 } + \mathrm { t r } ( A ) = - \mathrm { K L } \bigl ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \mid \mid \overrightarrow { p } _ { T - t } ^ { [ i ] } \bigr ) ,
$$

which completes the proof of Lemma 3.1.

## B.2. Proof of Theorem 1

Recall the definition of $C _ { D } ^ { [ i , j ] }$ in (9). We set $\delta = D ^ { - q } , q \geq 2$ and write � ≡ log $\begin{array} { r } { \frac { 2 K } { \delta } = \Theta \bigl ( \log ( K D ) \bigr ) } \end{array}$ , and $S = { \mathrm { S N R } } _ { T - t }$ for convenience. We also write $\mathcal { T } ( t ) = \mathcal { T } _ { D } ^ { [ i , j ] } ( t ) , \mathcal { U } ( t ) = \mathcal { U } _ { D } ^ { [ i , j ] } ( t )$ and $\mathcal { V } ( t ) = \mathcal { V } _ { D } ^ { [ i , j ] } ( t )$ whose definitions are found in (9). Then, the inequality

$$
C _ { D } ^ { [ i , j ] } ( t , \delta / K ) \geq \eta \mathrm { K L } \big ( \vec { p } _ { T - t } ^ { [ j ] } | | \vec { p } _ { T - t } ^ { [ i ] } \big )\tag{55}
$$

is equivalent to $( 1 - \eta ) \ge G ( t )$ with $G ( t ) = N ( t ) / D ( t )$ , where

$$
\begin{array} { r } { N ( t ) \equiv \log \frac { \pi _ { i } } { \pi _ { j } } + \sqrt { 2 \mathcal { T } ( t ) r } + \sqrt { 2 \mathcal { U } ( t ) r } + 2 \mathcal { V } ( t ) r ; } \end{array}\tag{56}
$$

$$
D ( t ) \equiv \mathrm { K L } \bigl ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \vert \vert \overrightarrow { p } _ { T - t } ^ { [ i ] } \bigr ) .\tag{57}
$$

We will derive a suficient condition for (55). To this end, we consider a simplified upper bound for $G ( t )$ denoted by $\widetilde { G } ( t )$ , that involves only $S = { \mathrm { S N R } } _ { T - t }$ as a function of time �, and then solve the inequality $1 - \eta \geq \widetilde { G } ( t )$ , which is suficient for the original condition (55). First, we consider an upper bound of the term $N ( t )$ . Noticing that KL $\cdot \big ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \mid \big | \overrightarrow { p } _ { T - t } ^ { [ i ] } \big ) = \mathrm { K L } _ { m } \big ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \mid \big | \overrightarrow { p } _ { T - t } ^ { [ i ] } \big ) + \mathrm { K L } _ { c } \big ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \mid \big | \overrightarrow { p } _ { T - t } ^ { [ i ] } \big )$ with

$$
\begin{array} { r } { { \mathrm { K L } } _ { m } \big ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \vert \vert \overrightarrow { p } _ { T - t } ^ { [ i ] } \big ) : = \frac { m _ { T - t } ^ { 2 } } { 2 } \vert \vert \mu _ { i } - \mu _ { j } \vert \vert _ { ( S _ { T - t } ^ { i } ) ^ { - 1 } } ^ { 2 } ; } \end{array}\tag{58}
$$

$$
\begin{array} { r } { { \bf K L } _ { c } \big ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \big | \big | \overrightarrow { p } _ { T - t } ^ { [ i ] } \big ) : = \frac { 1 } { 2 } \Big ( \mathrm { t r } \big ( S _ { T - t } ^ { j } ( S _ { T - t } ^ { i } ) ^ { - 1 } \big ) - D - \log \frac { \mathrm { d e t } S _ { T - t } ^ { j } } { \mathrm { d e t } S _ { T - t } ^ { i } } \Big ) , } \end{array}\tag{59}
$$

we deduce from the bound (13) in Lemma 3.2 that

$$
\begin{array} { r l r } & { } & { \sqrt { 2 \mathcal { T } ( t ) r } \leq \sqrt { 4 r \big ( 1 + S \cdot \| \Sigma _ { i } - \Sigma _ { j } \| _ { 2 } \big ) \mathrm { K L } _ { m } \big ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \mid \mid \overrightarrow { p } _ { T - t } ^ { [ i ] } \big ) } } \\ & { } & { \leq \frac { 2 r \big ( 1 + S \cdot \| \Sigma _ { i } - \Sigma _ { j } \| _ { 2 } \big ) } { ( 1 - \eta ) } + \frac { 1 - \eta } { 2 } { \mathrm { K L } _ { m } \big ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \mid \mid \overrightarrow { p } _ { T - t } ^ { [ i ] } \big ) } , } \end{array}\tag{60}
$$

where we used Arithmetic/Geometric Mean inequality: $\begin{array} { r } { \sqrt { a b } \le \frac { a } { 2 ( 1 - \eta ) } + \frac { ( 1 - \eta ) b } { 2 } , a , b > 0 } \end{array}$ . Similarly, we obtain from Lemma C.1 in Appendix C that

$$
\begin{array} { r } { \sqrt { 2 \mathcal { U } ( t ) r } \le \frac { 2 r \left( 1 + S \cdot \| \Sigma _ { j } \| _ { 2 } \right) } { ( 1 - \eta ) } + \frac { 1 - \eta } { 2 } \mathrm { K L } _ { c } \bigl ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \vert \vert \overrightarrow { p } _ { T - t } ^ { [ i ] } \bigr ) . } \end{array}\tag{61}
$$

Substitution of the bounds (60–61) and (15) into (56) with Assumptions 1, 2, 3 leads to

$$
\begin{array} { r } { N ( s ) \leq C _ { 1 } + \frac { 4 r } { 1 - \eta } + C _ { 2 } D ^ { \alpha } \Big ( \frac { 1 0 - 4 \eta } { 1 - \eta } \Big ) r \cdot S + \frac { 1 - \eta } { 2 } \mathrm { K L } \big ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \vert \vert \overrightarrow { p } _ { T - t } ^ { [ i ] } \big ) , } \end{array}
$$

where the constants $C _ { 1 } , C _ { 2 }$ are collected in Assumptions 1 and 2. Thus, we obtain

$$
\begin{array} { r } { G ( t ) \le \frac { C _ { 1 } + \frac { 4 r } { 1 - \eta } + C _ { 2 } D ^ { \alpha } \left( \frac { 1 0 - 4 \eta } { 1 - \eta } \right) r \cdot S } { \mathrm { K L } \left( \overrightarrow { p } _ { T - t } ^ { [ j ] } \mid \big | \overrightarrow { p } _ { T - t } ^ { [ i ] } \right) } + \frac { 1 - \eta } { 2 } \le \frac { C _ { 1 } + \frac { 4 r } { 1 - \eta } + C _ { 2 } D ^ { \alpha } \left( \frac { 1 0 - 4 \eta } { 1 - \eta } \right) r \cdot S } { \frac { S } { 2 } \left( \frac { C _ { 3 } } { C _ { 5 } } \right) ^ { 2 } \cdot \frac { D ^ { 2 } } { S \cdot \Pi ( \Sigma _ { i } ) + D } } + \frac { 1 - \eta } { 2 } \equiv \widetilde { G } ( t ) , } \end{array}
$$

where in the second inequality we used Lemma C.2 under Assumptions 3 and 4, provided in Appendix C. Then, $1 - \eta \ge G ( t )$ is satisfied if $1 - \eta \geq \widetilde { G } ( t )$ holds, which is simplified as:

$$
\begin{array} { r } { \frac { 1 - \eta } { 2 } \cdot \frac { \widetilde { C } ^ { 2 } D ^ { 2 } } { S \cdot \mathrm { t r } ( \Sigma _ { i } ) + D } \cdot \frac { S } { 2 } \geq \mathcal { B } + \mathcal { C } S , } \end{array}
$$

where we have set:

$$
\begin{array} { r } { \mathcal { B } \equiv C _ { 1 } + \frac { 4 r } { 1 - \eta } > 0 , \quad \mathcal { C } \equiv C _ { 2 } D ^ { \alpha } \left( \frac { 1 0 - 4 \eta } { 1 - \eta } \right) r > 0 , \quad \widetilde { C } \equiv \frac { C _ { 3 } } { C _ { 5 } } > 0 . } \end{array}
$$

Rearranging the terms with respect to �, we obtain the following quadratic inequality:

$$
\begin{array} { r } { \mathcal { C } \operatorname { t r } ( \Sigma _ { i } ) S ^ { 2 } - \mathcal { A } S + \mathcal { B } D \leq 0 , \quad \mathrm { w h e r e } \quad \mathcal { A } \equiv \frac { 1 - \eta } { 4 } \widetilde { C } ^ { 2 } D ^ { 2 } - \mathcal { C } D - \mathcal { B } \operatorname { t r } ( \Sigma _ { i } ) . } \end{array}\tag{62}
$$

Under Assumption 2, we have

$$
\mathrm { t r } ( \Sigma _ { i } ) = \sum _ { k = 1 } ^ { D } \lambda _ { k } ( \Sigma _ { i } ) \leq C _ { 2 } M _ { i } D ^ { \alpha } + \nu _ { i } \leq C _ { 2 } ( D ^ { \alpha } + R ) M ^ { \dagger }\tag{63}
$$

where the constants $C _ { 2 } > 0 , \nu _ { i } \geq 0 , R \in [ 0 , 1 )$ are collected in Assumption 2. Thus, there exists $D _ { 1 } \in \mathbb { N }$ such that for all $D \geq D _ { 1 }$ , we have the discriminant strictly positive, i.e.,

$$
\Delta \equiv \mathcal { A } ^ { 2 } - 4 \mathcal { C } \mathrm { t r } ( \Sigma _ { i } ) \mathcal { B } D > 0 .
$$

This is because $\mathcal { A } ^ { 2 } \ : = \ : \Theta ( D ^ { 4 } )$ and $\mathcal { C } \mathrm { t r } ( \Sigma _ { i } ) \mathcal { B } D \ = \ O ( D ^ { 1 + 2 \alpha } ( \log D ) ^ { 2 } )$ , and thus the leading term dominates the lower-order term for any $\alpha \in [ 0 , 1 )$ . Under $\Delta > 0$ , solving (62) under $S \geq S \mathrm { N R } _ { T }$ leads to $S \in \left[ \operatorname* { m a x } \{ \mathrm { S N R } _ { T } , \underline { { S } } \} , \overline { { S } } \right]$ , where

$$
\begin{array} { r } { \underline { { \boldsymbol { S } } } = \frac { \boldsymbol { \mathcal { A } } - \sqrt { \Delta } } { 2 \boldsymbol { \mathcal { C } } \mathrm { t r } ( \Sigma _ { i } ) } , \qquad \overline { { \boldsymbol { S } } } = \frac { \boldsymbol { \mathcal { A } } + \sqrt { \Delta } } { 2 \boldsymbol { \mathcal { C } } \mathrm { t r } ( \Sigma _ { i } ) } . } \end{array}
$$

First, we note from (63) that for all suficiently large �,

$$
\begin{array} { r } { \overline { { S } } > \frac { \mathcal { A } } { 2 \mathcal { C } \mathrm { t r } ( \Sigma _ { i } ) } \gtrsim \frac { D ^ { 2 } } { D ^ { 2 \alpha } \log D } = \frac { D ^ { 2 - 2 \alpha } } { \log D } , } \end{array}
$$

where the right-hand side is an increasing function of $D$ since $\alpha < 1$ . Therefore, we have: for any fixed early stopping criterion $ { \varepsilon } \in \left( 0 , 1 \right)$ , there exists $D _ { 2 }$ such that for all $D \geq D _ { 2 }$ , the upper root $\overline { { S } }$ is strictly larger than the early stopping threshold $S _ { \varepsilon } = \mathrm { S N R } _ { \varepsilon } = m _ { \varepsilon } ^ { 2 } / \sigma _ { \varepsilon } ^ { 2 }$ , which is independent of $D$ . Notice that $S = \mathrm { S N R } _ { T - t }$ is a non-decreasing function of the physical time $t .$ Thus, we have: for all $D \geq D ^ { \dagger } = \operatorname* { m a x } \{ D _ { 1 } , D _ { 2 } \}$ , the target condition (55) holds for all $t ~ \in ~ [ 0 , T - \varepsilon ]$ satisfying SNR $\mathbf { \chi } _ { T - t } ^ { } \in \left[ \operatorname* { m a x } \{ \mathbf { S N R } _ { T } , \underline { { S } } \} , \mathbf { S N R } _ { \varepsilon } \right]$ , equivalently to $\begin{array} { r } { t \in \left[ \operatorname* { m a x } \{ 0 , T + \frac { 1 } { 2 } \log \frac { S } { 1 + S } \} , T - \varepsilon \right] } \end{array}$

We now study the scale of the threshold $\underline { { S } } .$ We multiply the numerator and denominator by the conjugate $\mathcal { A } + \sqrt { \Delta }$ and have from $\Delta > 0$ and the definition of terms $\mathcal { B }$ and $\mathcal { A }$ that

$$
\begin{array} { r } { \underline { { S } } = \frac { 2 \mathcal { B } D } { \mathcal { A } + \sqrt { \Delta } } \le \frac { 2 \mathcal { B } D } { \mathcal { A } } = O \left( \frac { D \log ( K D ) } { D ^ { 2 } } \right) = O \left( \frac { \log ( K D ) } { D } \right) , } \end{array}\tag{64}
$$

where we used $r = \Theta \bigl ( \log ( K D ) \bigr )$ . Similarly, we have $\begin{array} { r } { \underline { { S } } \ge \frac { { \mathcal B } D } { { \mathcal A } } \gtrsim \frac { \log ( K D ) } { D } } \end{array}$ . The first part of the statement has now been proved.

Finally, we prove the second part of the statement. For any $\beta ~ \in ~ ( 0 , 1 )$ , directly solving $\exp \bigl ( - \eta \dot { \mathrm { K L } } \bigl ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \vert \vert \overrightarrow { p } _ { T - t } ^ { [ i ] } \bigr ) \bigr ) \leq \beta$ gives

$$
\begin{array} { r } { \mathrm { K L } \bigl ( \underset { { T - t } } { \vec { p } _ { T - t } ^ { [ j ] } } | | \underset { { P _ { T - t } } } { \vec { p } _ { T - t } ^ { [ i ] } } \bigr ) \geq \frac { 1 } { \eta } \log \frac { 1 } { \beta } } \end{array}\tag{65}
$$

under $\begin{array} { r } { t \in \left[ \operatorname* { m a x } \bigl \{ 0 , T + \frac { 1 } { 2 } \log \frac { S } { 1 + S } \bigr \} , T - \varepsilon \right] } \end{array}$ . We see that the inequality (65) is satisfied if the lower bound of left-hand-side given in Lemma C.2 is no lower than $\frac { 1 } { \eta } \log \frac { 1 } { \beta }$ , that is, $\begin{array} { r } { \frac { \mathrm { S N R } _ { T - t } } { 2 } \Big ( \frac { C _ { 3 } } { C _ { 5 } } \Big ) ^ { 2 } \cdot \frac { D ^ { 2 } } { \mathrm { S N R } _ { T - t } \cdot \mathrm { t r } ( \Sigma _ { i } ) + D } \geq } \end{array}$ $\frac { 1 } { \eta } \log \frac { 1 } { \beta }$ . Solving the latter inequality with respect to $\mathrm { S N R } _ { T - t }$ under the range $\mathrm { S N R } _ { T - t } \in \lbrack \underline { { S } } , \mathrm { S N R } _ { \varepsilon } ]$

leads to (19), and then the second part of the statement has been shown due to the high probability bound (17). Proof of Theorem 1 is now complete.

## C. Auxiliary Results for Theorem 1

## C.1. First Result

Lemma C.1. Let $i , k \in [ K ]$ and $Y = ( S _ { T - t } ^ { i } ) ^ { 1 / 2 } ( S _ { T - t } ^ { k } ) ^ { - 1 } ( S _ { T - t } ^ { i } ) ^ { 1 / 2 } - I _ { D }$ . We have:

$$
\mathrm { t r } ( Y ^ { 2 } ) \leq 4 \Big ( \mathrm { S N R } _ { T - t } \| \Sigma _ { i } \| _ { 2 } + 1 \Big ) \cdot \mathrm { K L } _ { c } \big ( \overrightarrow { p } _ { T - t } ^ { [ i ] } \mid \mid \overrightarrow { p } _ { T - t } ^ { [ k ] } \big ) ,
$$

where $\mathrm { K L } _ { c } \big ( \vec { p } _ { T - t } ^ { [ i ] } \big | \big | \vec { p } _ { T - t } ^ { [ k ] } \big )$ is defined in (59).

Proof. Let $\{ \eta _ { l } \} _ { l = 1 , \dots , D }$ be the eigenvalues of $( S _ { T - t } ^ { i } ) ^ { 1 / 2 } ( S _ { T - t } ^ { k } ) ^ { - 1 } ( S _ { T - t } ^ { i } ) ^ { 1 / 2 } .$ Then, the eigenvalues of � are $\{ \eta _ { l } \ : - \ : 1 \} _ { l }$ , yielding $\mathrm { t r } ( Y ^ { 2 } ) = \textstyle \sum _ { l = 1 } ^ { \hat { D } } \dot { ( } \eta _ { l } - \dot { 1 } ) ^ { 2 } .$ Because $S _ { T - t } ^ { i } ( S _ { T - t } ^ { k } ) ^ { - 1 }$ is similar to $( S _ { T - t } ^ { i } ) ^ { 1 / 2 } ( S _ { T - t } ^ { k } ) ^ { - 1 } ( S _ { T - t } ^ { i } ) ^ { 1 / 2 }$ , they share the identical eigenvalues $\eta _ { l } .$ We thus get $\begin{array} { r c l } { { \mathrm { K L } _ { c } \bigl ( \overrightarrow { p } _ { T - t } ^ { [ i ] } \mid \mid \overrightarrow { p } _ { T - t } ^ { [ k ] } \bigr ) } } & { { = } } & { { \frac { 1 } { 2 } \sum _ { l = 1 } ^ { D } ( \eta _ { l } - 1 - \log \eta _ { l } ) } } \end{array}$ . The maximum eigenvalue $\eta _ { 1 }$ is bounded by $\left\| S _ { T - t } ^ { i } \right\| _ { 2 } ^ { \cdot } \left\| \dot { ( S _ { T - t } ^ { k } ) ^ { - 1 } } \right\| _ { 2 } \leq \bar { C } _ { \eta } ( t )$ with $C _ { \eta } ( t ) \ \equiv \ ( \mathrm { S N R } _ { T - t } \| \Sigma _ { i } \| _ { 2 } + 1 )$ . By elementary calculus, the algebraic inequality $( \eta - 1 ) ^ { 2 } \le 2 C _ { \eta } ( t ) ( \eta - 1 - \log \eta )$ holds strictly for all $\eta \in ( 0 , C _ { \eta } ( t ) ]$ . Applying this inequality to each eigenvalue $\eta _ { l }$ and summing them completes the proof. □

## C.2. Second Result

Lemma C.2 (Upper and lower bounds for KL divergence). It holds under Assumptions 3 and 4 that: $f o r$ any $i , j \in [ K ]$

$$
\begin{array} { r l } & { \frac { \mathrm { S N R } _ { T - t } } { 2 } \cdot \bigg ( \frac { C _ { 3 } } { C _ { 5 } } \bigg ) ^ { 2 } \frac { D ^ { 2 } } { \mathrm { S N R } _ { T - t } \mathrm { t r } ( \Sigma _ { i } ) + D } } \\ & { \qquad \quad \leq \mathrm { ~ K L } \big ( \overrightarrow { p } _ { T - t } ^ { [ j ] } \mid \mid \overrightarrow { p } _ { T - t } ^ { [ i ] } \big ) \leq \frac { \mathrm { S N R } _ { T - t } } { 2 } \cdot C _ { 4 } D + \frac { \mathrm { S N R } _ { T - t } ^ { 2 } } { 4 } \cdot \| \Sigma _ { i } - \Sigma _ { j } \| _ { F } ^ { 2 } , } \end{array}\tag{66}
$$

where the constants $C _ { 3 } , C _ { 4 } , C _ { 5 } > 0$ are collected in Assumptions 3 and 4.

Proof. We begin with proving the lower bound of (66). Since the covariance matrix $\Sigma _ { i }$ is symmetric due to Assumption 2, its eigen-decomposition gives $\Sigma _ { i } = U ^ { ( i ) } \Lambda ^ { ( i ) } U ^ { ( i ) , \top }$ with the orthogonal matrix $U ^ { ( i ) } \equiv [ u _ { 1 } ( i ) , \ldots , u _ { D } ( i ) ]$ and $\Lambda ^ { ( i ) } = \mathrm { d i a g } \big ( \lambda _ { 1 } ( \Sigma _ { i } ) , \dots , \lambda _ { D } ( \Sigma _ { i } ) \big )$ . For notational simplicity, we write $\nu _ { k } = \langle \mu _ { j } - \mu _ { i } , u _ { k } ( i ) \rangle$ . Then, $S _ { T - t } ^ { i } = U ^ { ( i ) } \bigl ( m _ { T - t } ^ { 2 } \Lambda ^ { ( i ) } + \sigma _ { T - t } ^ { 2 } I _ { D } \bigr ) U ^ { ( i ) , \top }$ and (58) yield that

$$
\begin{array} { r } { { \bf K L } ( \underset { T - t } { \longrightarrow } | | \underset { J _ { T - t } } { \longrightarrow } | | \underset { P _ { T - t } } { \longrightarrow } | | \underset { + } { \longrightarrow } { \bf K L } _ { m } ( \underset { P _ { T - t } } { \longrightarrow } | | \underset { P _ { T - t } } { \longrightarrow } | | \underset { + } { \longrightarrow } \frac { D } { 2 } \underset { k = 1 } { \longrightarrow } \frac { \nu _ { k } ^ { 2 } } { m _ { T - t } ^ { 2 } \lambda _ { k } ( \Sigma _ { i } ) + \sigma _ { T - t } ^ { 2 } } .  } \end{array}\tag{67}
$$

Further, the Cauchy-Schwarz inequality gives

$$
\sum _ { k = 1 } ^ { D } \frac { \nu _ { k } ^ { 2 } } { m _ { T - t } ^ { 2 } \lambda _ { k } ( \Sigma _ { i } ) + \sigma _ { T - t } ^ { 2 } } \geq \frac { \left( \sum _ { k = 1 } ^ { D } | \nu _ { k } | \right) ^ { 2 } } { \sum _ { k = 1 } ^ { D } ( m _ { T - t } ^ { 2 } \lambda _ { k } ( \Sigma _ { i } ) + \sigma _ { T - t } ^ { 2 } ) } = \frac { \left( \sum _ { k = 1 } ^ { D } | \nu _ { k } | \right) ^ { 2 } } { m _ { T - t } ^ { 2 } \mathfrak { t r } ( \Sigma _ { i } ) + \sigma _ { T - t } ^ { 2 } D } .\tag{68}
$$

For the numerator in the above lower bound, we have $\begin{array} { r } { \sum _ { k = 1 } ^ { D } \left| \nu _ { k } \right| \geq \frac { C _ { 3 } } { C _ { 5 } } D } \end{array}$ under Assumption 3, which comes from

$$
\sum _ { k = 1 } ^ { D } \nu _ { k } ^ { 2 } = \| \mu _ { i } - \mu _ { j } \| ^ { 2 } \geq C _ { 3 } D , \quad \sum _ { k = 1 } ^ { D } \nu _ { k } ^ { 2 } \leq C _ { 5 } \sum _ { k = 1 } ^ { D } | \nu _ { k } | .
$$

We thus obtain the desired lower bound by recalling the definition of SNR: $\mathrm { S N R } _ { T - t } = m _ { T - t } ^ { 2 } / \sigma _ { T - t } ^ { 2 }$ . We now show the upper bound of (66). The upper bound for the mean part of the KL (58) is immediately obtained from (67) with Assumption 3 as:

$$
\begin{array} { r } { { \mathrm { K L } } _ { m } \Big ( \underset { p } { \longrightarrow } _ { T - t } ^ { [ j ] } \big \lVert \underset { \mathcal { P } _ { T - t } } { \longrightarrow } \Big ) \leq \frac { m _ { T - t } ^ { 2 } } { 2 \sigma _ { T - t } ^ { 2 } } \displaystyle \sum _ { k = 1 } ^ { D } \nu _ { k } ^ { 2 } = \frac { { \mathrm { S N R } } _ { T - t } } { 2 } \big \lVert \mu _ { i } - \mu _ { j } \big \rVert ^ { 2 } \leq \frac { { \mathrm { S N R } } _ { T - t } } { 2 } \cdot C _ { 4 } D . } \end{array}
$$

We now consider the covariance part of the KL divergence, given in (59). Introducing $\widetilde { S } _ { \lambda } = I _ { D } +$ $\lambda ( S _ { T - t } ^ { i } ( S _ { T - t } ^ { k } ) ^ { - 1 } - I _ { D } ) , \lambda \in [ 0 , 1 ]$ , we use standard matrix calculus [Petersen et al., 2008] to get:

$$
\log \operatorname* { d e t } \bigl ( S _ { T - t } ^ { i } ( S _ { T - t } ^ { k } ) ^ { - 1 } \bigr ) = \int _ { 0 } ^ { 1 } \frac { d } { d \lambda } \log \operatorname* { d e t } \widetilde { S } _ { \lambda } d \lambda = \int _ { 0 } ^ { 1 } \mathrm { t r } \Big [ ( \widetilde { S } _ { \lambda } ) ^ { - 1 } \big ( S _ { T - t } ^ { i } ( S _ { T - t } ^ { k } ) ^ { - 1 } - I _ { D } \big ) \Big ] d \lambda .
$$

Thus, (59) is expressed as:

$$
\begin{array} { r l r } & { } & { { \bf K L } _ { c } \Big ( \overrightarrow { p } _ { T - t } ^ { [ i ] } \vert \vert \overrightarrow { p } _ { T - t } ^ { [ k ] } \Big ) = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } { \mathrm { t r } } \Big [ \big ( I _ { D } - ( \widetilde { S } _ { \lambda } ) ^ { - 1 } \big ) \big ( S _ { T - t } ^ { i } ( S _ { T - t } ^ { k } ) ^ { - 1 } - I _ { D } \big ) \Big ] d \lambda } \\ & { } & { = \frac { 1 } { 2 } \displaystyle \int _ { 0 } ^ { 1 } \lambda \cdot { \mathrm { t r } } \Big [ ( \widetilde { S } _ { \lambda } ) ^ { - 1 } \big ( S _ { T - t } ^ { i } ( S _ { T - t } ^ { k } ) ^ { - 1 } - I _ { D } \big ) ^ { 2 } \Big ] d \lambda . } \end{array}\tag{69}
$$

The expression $\widetilde { S } _ { \lambda } = \left( \lambda S _ { T - t } ^ { i } + ( 1 - \lambda ) S _ { T - t } ^ { k } \right) ( S _ { T - t } ^ { k } ) ^ { - 1 }$ and the cycliciy of the trace operator yield

$$
\begin{array} { r l } & { \mathrm { t r } \Big [ ( \widetilde S _ { \lambda } ) ^ { - 1 } \big ( S _ { T - t } ^ { i } ( S _ { T - t } ^ { k } ) ^ { - 1 } - I _ { D } \big ) ^ { 2 } \Big ] } \\ & { \quad = \mathrm { t r } \Big [ \big ( \lambda S _ { T - t } ^ { i } + ( 1 - \lambda ) S _ { T - t } ^ { k } \big ) ^ { - 1 } ( S _ { T - t } ^ { i } - S _ { T - t } ^ { k } ) ( S _ { T - t } ^ { k } ) ^ { - 1 } ( S _ { T - t } ^ { i } - S _ { T - t } ^ { k } ) \Big ] } \\ & { \quad \le \| S _ { T - t } ^ { i } - S _ { T - t } ^ { k } \| _ { F } ^ { 2 } \cdot \| ( S _ { T - t } ^ { k } ) ^ { - 1 } \| _ { 2 } \cdot \| ( \lambda S _ { T - t } ^ { i } + ( 1 - \lambda ) S _ { T - t } ^ { k } ) ^ { - 1 } \| _ { 2 } \le \| S _ { T - t } ^ { i } - S _ { T - t } ^ { k } \| _ { F } ^ { 2 } \cdot \frac { 1 } { \sigma _ { T - t } ^ { 4 } } , } \end{array}\tag{70}
$$

where we used $\begin{array} { r } { ( S _ { T - t } ^ { k } ) ^ { - 1 } \preceq \frac { 1 } { \sigma _ { T - t } ^ { 2 } } I _ { D } } \end{array}$ and the operator convexity of $z \mapsto z ^ { - 1 }$ for positive definite matrices to get

$$
\begin{array} { r } { ( \lambda S _ { T - t } ^ { i } + ( 1 - \lambda ) S _ { T - t } ^ { k } ) ^ { - 1 } \preceq \lambda ( S _ { T - t } ^ { i } ) ^ { - 1 } + ( 1 - \lambda ) ( S _ { T - t } ^ { k } ) ^ { - 1 } \preceq \frac { 1 } { \sigma _ { T - t } ^ { 2 } } I _ { D } . } \end{array}
$$

Substitution of (70) into (69) together with $S _ { T - t } ^ { i } - S _ { T - t } ^ { k } = m _ { T - t } ^ { 2 } \big ( \Sigma _ { i } - \Sigma _ { k } \big )$ yields

$$
\begin{array} { r } { \mathrm { K L } _ { c } \big ( \vec { p } _ { T - t } ^ { [ i ] } \big | \big | \vec { p } _ { T - t } ^ { [ k ] } \big ) \leq \frac { \mathrm { S N R } _ { T - t } ^ { 2 } } { 4 } \big \| \Sigma _ { i } - \Sigma _ { k } \big \| _ { F } ^ { 2 } , } \end{array}
$$

which completes the proof.

## D. Proof of Lemma 3.2

Proofof(15). We start with the easiest term $\mathcal { V } _ { D } ^ { [ i , j ] } ( t ) = \| I _ { D } - S _ { T - t } ^ { i } ( S _ { T - t } ^ { j } ) ^ { - 1 } \| ,$ . We have

$$
I _ { D } - S _ { T - t } ^ { i } ( S _ { T - t } ^ { j } ) ^ { - 1 } = \bigl ( S _ { T - t } ^ { j } - S _ { T - t } ^ { i } \bigr ) ( S _ { T - t } ^ { j } ) ^ { - 1 } = m _ { T - t } ^ { 2 } \bigl ( \Sigma _ { j } - \Sigma _ { i } \bigr ) ( S _ { T - t } ^ { j } ) ^ { - 1 }\tag{71}
$$

and immediately get the desired bound from $\begin{array} { r } { ( S _ { T - t } ^ { j } ) ^ { - 1 } \preceq \frac { 1 } { \sigma _ { T - t } ^ { 2 } } I _ { D } } \end{array}$

Proof of (13). We have

$$
\begin{array} { r l } & { \mathcal { T } _ { D } ^ { [ i , j ] } ( t ) : = m _ { T - t } ^ { 2 } \Big \Vert \mu _ { i } - \mu _ { j } \Big \Vert _ { ( S _ { T - t } ^ { i } ) ^ { - 1 } ( S _ { T - t } ^ { j } ) ( S _ { T - t } ^ { i } ) ^ { - 1 } } ^ { 2 } } \\ & { \qquad = m _ { T - t } ^ { 2 } \Big \Vert \mu _ { i } - \mu _ { j } \Big \Vert _ { ( S _ { T - t } ^ { i } ) ^ { - 1 } } ^ { 2 } + m _ { T - t } ^ { 2 } \Big \Vert \mu _ { i } - \mu _ { j } \Big \Vert _ { ( S _ { T - t } ^ { i } ) ^ { - 1 } ( S _ { T - t } ^ { j } ) ( S _ { T - t } ^ { i } ) ^ { - 1 } - ( S _ { T - t } ^ { i } ) ^ { - 1 } } ^ { 2 } . } \end{array}\tag{72}
$$

and

$$
\begin{array} { r l } & { ( S _ { T - t } ^ { i } ) ^ { - 1 } ( S _ { T - t } ^ { j } ) ( S _ { T - t } ^ { i } ) ^ { - 1 } - ( S _ { T - t } ^ { i } ) ^ { - 1 } } \\ & { \qquad = m _ { T - t } ^ { 2 } ( S _ { T - t } ^ { i } ) ^ { - 1 } \bigl ( \Sigma _ { j } - \Sigma _ { i } \bigr ) ( S _ { T - t } ^ { i } ) ^ { - 1 } \preceq \mathrm { S N R } _ { T - t } \| \Sigma _ { i } - \Sigma _ { j } \| _ { 2 } ( S _ { T - t } ^ { i } ) ^ { - 1 } . } \end{array}
$$

Substitution of this into (72) yields the desired bound.

Proof of (14). The formula (71) gives

$$
\begin{array} { r } { \mathcal { U } _ { D } ^ { [ i , j ] } ( t ) : = \frac { 1 } { 2 } \mathrm { t r } \big ( \big ( I _ { D } - S _ { T - t } ^ { j } ( S _ { T - t } ^ { i } ) ^ { - 1 } \big ) ^ { 2 } \big ) = \frac { m _ { T - t } ^ { 4 } } { 2 } \mathrm { t r } \Big ( \big ( \Sigma _ { i } - \Sigma _ { j } \big ) ^ { 2 } ( S _ { T - t } ^ { i } ) ^ { - 2 } \Big ) \leq \frac { \mathrm { S N R } _ { T - t } ^ { 2 } } { 2 } \| \Sigma _ { i } - \Sigma _ { j } \| _ { F } ^ { 2 } , } \end{array}
$$

which completes the proof of Lemma 3.2.

## E. Proofs of the Auxiliary Results for Theorem 2

## E.1. Proof of Lemma 3.4

From the definition of posterior covariance, we have $\begin{array} { r c l c r c l } { { \sum _ { T - t } ( y ) } } & { { \equiv } } & { { \mathrm { C o v } [ \overrightarrow { X } _ { 0 } \vert \overrightarrow { X } _ { T - t } } } & { { = } } & { { y ] } } & { { = } } & { { } } \end{array}$ $\mathbb { E } [ ( \overrightarrow { X } _ { 0 } ) ^ { \otimes 2 } | \overrightarrow { X } _ { T - t } ~ = ~ y ] ~ - ~ \left( \mathbb { E } [ \overrightarrow { X } _ { 0 } | \overrightarrow { X } _ { T - t } ~ = ~ y ] \right) ^ { \otimes 2 }$ . Using the posterior weights and (25), we express the conditional means as:

$$
\mathbb { E } [ ( \vec { X } _ { 0 } ) ^ { \otimes 2 } | \vec { X } _ { T - t } = y ] = \sum _ { k = 1 } ^ { K } W _ { D } ^ { [ k ] } ( t , y ) \cdot \Big \{ \mathrm { C o v } \big [ \vec { X } _ { 0 } | \vec { X } _ { T - t } = y , \vec { X } _ { 0 } \sim P _ { \mathrm { d a t a } } ^ { [ k ] } \big ] + \big ( \mu ^ { [ k ] } ( t , y ) \big ) ^ { \otimes 2 } \Big \} ;
$$

$$
\mathbb { E } [ \overrightarrow { X } _ { 0 } \vert \overrightarrow { X } _ { T - t } = y ] = \sum _ { k = 1 } ^ { K } W _ { D } ^ { [ k ] } ( t , y ) \mu ^ { [ k ] } ( t , y ) .
$$

Substituting these into the formula of $\Sigma _ { T - t } ( y )$ gives

$$
\begin{array} { l } { { \displaystyle \Sigma _ { T - t } ( y ) = \sum _ { k = 1 } ^ { K } W _ { D } ^ { [ k ] } ( t , y ) \operatorname { C o v } [ \vec { X } _ { 0 } | \vec { X } _ { T - t } = y , \vec { X } _ { 0 } \sim P _ { \mathrm { d a t a } } ^ { [ k ] } ] + \sum _ { k = 1 } ^ { K } W _ { D } ^ { [ k ] } ( t , y ) \big ( \mu ^ { [ k ] } ( t , y ) \big ) ^ { \otimes 2 } } } \\ { ~ - \sum _ { k _ { 1 } , k _ { 2 } = 1 } ^ { K } W _ { D } ^ { [ k _ { 1 } ] } ( t , y ) W _ { D } ^ { [ k _ { 2 } ] } ( t , y ) \mu ^ { [ k _ { 1 } ] } ( t , y ) \mu ^ { [ k _ { 2 } ] } ( t , y ) ^ { \top } } \\ { { \displaystyle = \sum _ { k = 1 } ^ { K } W _ { D } ^ { [ k ] } ( t , y ) \operatorname { C o v } [ \vec { X } _ { 0 } | \vec { X } _ { T - t } = y , \vec { X } _ { 0 } \sim P _ { \mathrm { d a t a } } ^ { [ k ] } ] } } \\ { { ~ + \frac { 1 } { 2 } \sum _ { k _ { 1 } , k _ { 2 } = 1 } ^ { K } W _ { D } ^ { [ k _ { 1 } ] } ( t , y ) W _ { D } ^ { [ k _ { 2 } ] } ( t , y ) \big ( \mu ^ { [ k _ { 1 } ] } ( t , y ) - \mu ^ { [ k _ { 2 } ] } ( t , y ) \big ) ^ { \otimes 2 } . } } \end{array}
$$

Finally, the conditional covariance formula for joint Gaussian distributions deduces that

$$
\mathrm { C o v } [ \overrightarrow { X } _ { 0 } \vert \overrightarrow { X } _ { T - t } = y , \overrightarrow { X } _ { 0 } \sim P _ { \mathrm { d a t a } } ^ { [ k ] } ] = \Sigma _ { k } - m _ { T - t } ^ { 2 } \Sigma _ { k } ( S _ { T - t } ^ { k } ) ^ { - 1 } \Sigma _ { k } = \sigma _ { T - t } ^ { 2 } \Sigma _ { k } ( S _ { T - t } ^ { k } ) ^ { - 1 } ,
$$

which concludes Lemma 3.4.

## E.2. Proof of Lemma 5.2

First, we have for $\rho > 0$

$$
\begin{array} { r l } & { \log \Big ( \mathbb { E } \Big [ \underset { k _ { 1 } , k _ { 2 } \in [ K ] } { \operatorname* { m a x } } \exp \Big ( \rho \cdot \big \| \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ^ { [ i ] } ) - \mu ^ { [ k _ { 2 } ] } ( Z _ { j } ^ { [ i ] } ) \big \| ^ { 2 } \Big ) \Big ] \Big ) } \\ & { \leq \log \Big ( \underset { k _ { 1 } , k _ { 2 } \in [ K ] } { \sum } \mathbb { E } \big [ \exp \big ( \rho \cdot \big \| \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ^ { [ i ] } ) - \mu ^ { [ k _ { 2 } ] } ( Z _ { j } ^ { [ i ] } ) \big \| ^ { 2 } ) \big ] \Big ) } \\ & { \leq \log \Big ( \underset { k _ { 1 } , k _ { 2 } \in [ K ] } { \sum } \underset { k \in \{ k _ { 1 } , k _ { 2 } \} } { \prod } \sqrt { \mathbb { E } \big [ \exp \big ( 4 \rho \cdot \big \| \mu ^ { [ k ] } ( Z _ { j } ^ { [ i ] } ) - \mu ^ { [ i ] } ( Z _ { j } ^ { [ i ] } ) \big \| ^ { 2 } \big ) \big ] } \Big ) . } \end{array}
$$

where in the last line we used $\begin{array} { r l r } { \| \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ^ { [ i ] } ) \ - \mu ^ { [ k _ { 2 } ] } ( Z _ { j } ^ { [ i ] } ) \| ^ { 2 } } & { \leq } & { 2 \big \{ \| \mu ^ { [ k _ { 1 } ] } ( Z _ { j } ^ { [ i ] } ) \ - \ \mu ^ { [ i ] } ( Z _ { j } ^ { [ i ] } ) \| ^ { 2 } \ + } \end{array}$ $\| \mu ^ { [ k _ { 2 } ] } ( Z _ { i } ^ { [ i ] } ) - \mu ^ { [ i ] } ( Z _ { i } ^ { [ i ] } ) \| ^ { 2 } \big \}$ and Cauchy-Schwartz inequality. For notational simplicity, we write $r _ { j } = \sigma _ { T - t _ { i } } ^ { 2 } / m _ { T - t _ { j } }$ and recall $\mu ^ { [ k ] } ( t , x ) = \mathbb { E } [ \overrightarrow { X } _ { 0 } \vert \overrightarrow { X } _ { T - t } = x , \overrightarrow { X } _ { 0 } \sim P _ { \mathrm { d a t a } } ^ { [ k ] } ]$ . Then, Tweedie’s formula for ∇ log $\vec { p } _ { T - t } ^ { [ k ] } ( x ) , ( t , x ) \in [ 0 , T ) \times \mathbb { R } ^ { D }$ gives

$$
\begin{array} { r } { \mu ^ { [ k ] } ( t , x ) = \frac { 1 } { m _ { T - t } } \cdot ( x + \sigma _ { T - t } ^ { 2 } \nabla \log  \overrightarrow { p } _ { T - t } ^ { [ k ] } ( x ) ) } \end{array}
$$

and then

$$
\begin{array} { r l } & { \mu ^ { [ k ] } ( Z _ { j } ^ { [ i ] } ) - \mu ^ { [ i ] } ( Z _ { j } ^ { [ i ] } ) = r _ { j } \Big ( \nabla \log \overrightarrow { p } _ { T - t _ { j } } ^ { [ k ] } ( \overrightarrow { X } _ { T - t _ { j } } ^ { [ i ] } ) - \nabla \log \overrightarrow { p } _ { T - t _ { j } } ^ { [ i ] } ( \overrightarrow { X } _ { T - t _ { j } } ^ { [ i ] } ) \Big ) } \\ & { \quad \quad \quad \quad \stackrel { d } { = } r _ { j } \cdot \big ( F _ { k , i } ( j ) + G _ { k , i } ( j ) \big ) } \end{array}\tag{73}
$$

where

$$
F _ { k , i } ( j ) = m _ { T - t _ { j } } \cdot ( S _ { T - t _ { j } } ^ { k } ) ^ { - 1 } ( \mu _ { k } - \mu _ { i } ) , \quad G _ { k , i } ( j ) \sim \cal N \big ( 0 , H _ { k , i } ( j ) \big )
$$

with $H _ { k , i } ( j )$ defined in (40). The moment generating function for the quadratic of Gaussians under the expression (73) gives

$$
\begin{array} { r l } & { \log \mathbb E \biggl [ \exp \biggl ( 4 \rho \cdot \big \| \mu ^ { [ k ] } ( Z _ { j } ^ { [ i ] } ) - \mu ^ { [ i ] } ( Z _ { j } ^ { [ i ] } ) \big \| ^ { 2 } \biggr ) \biggr ] } \\ & { \qquad = - \frac 1 2 \log \operatorname* { d e t } \bigl ( I _ { D } - 8 \rho r _ { j } ^ { 2 } H _ { k , i } ( j ) \bigr ) + 4 \rho r _ { j } ^ { 2 } ( F _ { k , i } ( j ) ) ^ { \top } \bigl ( I _ { D } - 8 \rho r _ { j } ^ { 2 } H _ { k , i } ( j ) \bigr ) ^ { - 1 } F _ { k , i } ( j ) } \end{array}\tag{74}
$$

for any $\rho > 0$ such that $I _ { D } - 8 \rho r _ { i } ^ { 2 } H _ { k , i } ( j )$ is positive definite, which is satisfied i $\dot { } \rho < 1 / ( 8 r _ { j } ^ { 2 } \| H _ { k , i } ( j ) \| _ { 2 } )$ Thus, under $\begin{array} { r } { \rho \leq 1 / ( 1 6 r _ { { j } } ^ { 2 } \operatorname* { m a x } _ { k , i \in [ K ] } \| H _ { k , i } ( j ) \| _ { 2 } ) } \end{array}$ , we have

$$
\begin{array} { r } { - \frac { 1 } { 2 } \log \operatorname* { d e t } \bigl ( I _ { D } - 8 \rho r _ { j } ^ { 2 } H _ { k , i } ( j ) \bigr ) = - \frac { 1 } { 2 } \displaystyle \sum _ { \ell = 1 } ^ { D } \log \bigl ( 1 - 8 \rho r _ { j } ^ { 2 } \lambda _ { \ell } ( H _ { k , i } ( j ) ) \bigr ) \leq 8 \rho r _ { j } ^ { 2 } \cdot \operatorname { t r } \bigl ( H _ { k , i } ( j ) \bigr ) , } \end{array}
$$

because $- \log ( 1 - x ) \leq 2 x$ for $x \in [ 0 , 1 / 2 ]$ . We have

$$
\begin{array} { r l } & { \mathrm { t r } ( H _ { k , i } ( j ) ) = m _ { T - t } ^ { 4 } \mathrm { t r } \big ( ( S _ { T - t _ { j } } ^ { k } ) ^ { - 1 } ( \Sigma _ { k } - \Sigma _ { i } ) ( S _ { T - t _ { j } } ^ { i } ) ^ { - 1 } ( \Sigma _ { k } - \Sigma _ { i } ) ( S _ { T - t _ { j } } ^ { k } ) ^ { - 1 } \big ) } \\ & { \qquad \le \frac { m _ { T - t _ { j } } ^ { 4 } } { \sigma _ { T - t _ { j } } ^ { 6 } } \| \Sigma _ { k } - \Sigma _ { i } \| _ { F } ^ { 2 } \lesssim \frac { m _ { T - t _ { j } } ^ { 4 } } { \sigma _ { T - t _ { j } } ^ { 6 } } M ^ { \dagger } D ^ { 2 \alpha } } \end{array}
$$

where we used $\begin{array} { r } { ( S _ { T - t _ { j } } ^ { k } ) ^ { - 1 } \preceq \frac { 1 } { \sigma _ { T - t } ^ { 2 } } I _ { D } } \end{array}$ in the first inequality and the following bound in the second:

$$
\begin{array} { r } { \| \Sigma _ { k } - \Sigma _ { i } \| _ { F } ^ { 2 } \lesssim \| \Sigma _ { k } \| _ { F } ^ { 2 } + \| \Sigma _ { i } \| _ { F } ^ { 2 } \leq 2 C _ { 2 } D ^ { \alpha } ( C _ { 2 } D ^ { \alpha } + R ) M ^ { \dagger } \lesssim D ^ { 2 \alpha } M ^ { \dagger } . } \end{array}\tag{75}
$$

where the constants $C _ { 2 } ~ > ~ 0 , R ~ \in ~ [ 0 , 1 )$ are collected in Assumption 2. In the above, we used $\begin{array} { r } { \Vert \Sigma _ { k } \Vert _ { F } ^ { 2 } = \mathrm { t r } \big ( \Sigma _ { k } ^ { \top } \Sigma _ { k } \big ) \ = \ \mathrm { t r } \big ( \Lambda _ { k } ^ { \top } \Lambda _ { k } \big ) \ \le \ \lambda _ { 1 } ( \Sigma _ { k } ) \cdot \mathrm { t r } ( \Lambda _ { k } ) \ \le \ C _ { 2 } D ^ { \alpha } ( C _ { 2 } D ^ { \alpha } M _ { k } + \nu _ { k } ) \ \le \ C _ { 2 } D ^ { \alpha } ( C _ { 2 } D ^ { \alpha } + C _ { 2 } D ^ { \alpha } + C _ { 2 } D ^ { \alpha } ) , } \end{array}$ $R ) M ^ { \dagger }$ under Assumption 2 with decomposition $\Sigma _ { k } ~ = ~ U \Lambda _ { k } U ^ { \top }$ . We thus have that for $\rho \_ { \perp }$ $1 / ( 1 6 r _ { j } ^ { 2 } \operatorname* { m a x } _ { k , i \in [ K ] } \| H _ { k , i } ( j ) \| _ { 2 } )$

$$
\begin{array} { r } { - \frac 1 2 \log \operatorname* { d e t } \bigl ( I _ { D } - 8 \rho r _ { j } ^ { 2 } H _ { k , i } ( j ) \bigr ) \le \rho \operatorname { S N R } _ { j } M ^ { \dagger } D ^ { 2 \alpha } . } \end{array}\tag{76}
$$

Similarly, we also have that for $\begin{array} { r } { \rho \leq 1 / ( 1 6 r _ { j } ^ { 2 } \operatorname* { m a x } _ { k , i \in [ K ] } \| H _ { k , i } ( j ) \| _ { 2 } ) } \end{array}$

$$
\begin{array} { r } { \rho r _ { j } ^ { 2 } ( F _ { k , i } ( j ) ) ^ { \top } \big ( I _ { D } - 8 \rho r _ { j } ^ { 2 } H _ { k , i } ( j ) \big ) ^ { - 1 } F _ { k , i } ( j ) \le 2 \rho r _ { j } ^ { 2 } \| F _ { k , i } ( j ) \| ^ { 2 } \lesssim \rho D . \quad \big ( \cdot \cdot \ \mathrm { A s s u m p t i o n 3 } \big ) } \end{array}\tag{77}
$$

Substituting (76) and (77) into (74), we conclude Lemma 5.2.

## E.3. Proof of Lemma 5.3

Writing $\overrightarrow { X } _ { T - t _ { j } } ^ { [ i ] }$ as $m _ { T - t _ { j } } \mu _ { i } + Z$ with $Z \sim { \cal N } ( 0 , S _ { T - t _ { j } } ^ { i } )$ , we have

$$
\begin{array} { r l } & { D _ { k } ^ { [ i ] } ( j ) = \frac { \sigma _ { T - t _ { j } } ^ { 4 } } { m _ { T - t _ { j } } ^ { 2 } } \times { \mathbb { E } } \Big [ \big \| m _ { T - t _ { j } } ( S _ { T - t _ { j } } ^ { k } ) ^ { - 1 } ( \mu _ { i } - \mu _ { k } ) + \big \{ ( S _ { T - t _ { j } } ^ { k } ) ^ { - 1 } - ( S _ { T - t _ { j } } ^ { i } ) ^ { - 1 } \big \} Z \big \| ^ { 2 } \Big ] } \\ & { \qquad = \frac { \sigma _ { T - t _ { j } } ^ { 4 } } { m _ { T - t } ^ { 2 } } \times \Big \{ m _ { T - t _ { j } } ^ { 2 } \| \mu _ { i } - \mu _ { k } \| _ { ( S _ { T - t _ { i } } ^ { k } ) ^ { - 2 } } ^ { 2 } + \mathrm { t r } \big ( \big \{ ( S _ { T - t _ { j } } ^ { k } ) ^ { - 1 } - ( S _ { T - t _ { j } } ^ { i } ) ^ { - 1 } \big \} \big ) ^ { 2 } S _ { T - t _ { j } } ^ { i } \big ) \Big \} , } \end{array}\tag{78}
$$

where in the first equation, we used: for any $( t , x ) \in [ 0 , T ) \times \mathbb { R } ^ { D }$

$$
\begin{array} { r } { \mu ^ { [ k ] } ( T - t , x ) - \mu ^ { [ i ] } ( T - t , x ) = \frac { \sigma _ { T - t } ^ { 2 } } { m _ { T - t } } \bigl ( \nabla \log p _ { T - t } ^ { [ i ] } ( x ) - \nabla \log p _ { T - t } ^ { [ k ] } ( x ) \bigr ) , } \end{array}
$$

with ∇ log $p _ { T - t } ^ { [ k ] } ( x ) = - ( S _ { T - t } ^ { k } ) ^ { - 1 } \big ( x - m _ { T - t } \mu _ { k } \big )$ . The matrix inequality $\begin{array} { r } { ( S _ { T - t _ { j } } ^ { i } ) ^ { - 1 } \preceq \frac { 1 } { \sigma _ { T - t } ^ { 2 } } I _ { D } } \end{array}$ together with (58) yields

$$
\begin{array} { r } { m _ { T - t _ { j } } ^ { 2 } \Vert \mu _ { i } - \mu _ { k } \Vert _ { ( S _ { T - t _ { j } } ^ { k } ) ^ { - 2 } } ^ { 2 } \leq \frac { m _ { T - t _ { j } } ^ { 2 } } { \sigma _ { T - t _ { j } } ^ { 2 } } \Vert \mu _ { i } - \mu _ { k } \Vert _ { ( S _ { T - t _ { j } } ^ { k } ) ^ { - 1 } } ^ { 2 } = \frac { 2 } { \sigma _ { T - t _ { j } } ^ { 2 } } \mathrm { K L } _ { \mathrm { m } } \bigl ( \overrightarrow { p } _ { T - t } ^ { [ i ] } \mid \big | \overrightarrow { p } _ { T - t } ^ { [ k ] } \bigr ) . } \end{array}\tag{79}
$$

By setting $Y : = ( S _ { T - t _ { j } } ^ { i } ) ^ { 1 / 2 } ( S _ { T - t _ { j } } ^ { k } ) ^ { - 1 } ( S _ { T - t _ { j } } ^ { i } ) ^ { 1 / 2 } - I _ { D }$ , we have

$$
\begin{array} { r l } & { \mathrm { t r } \big ( \big \{ ( S _ { T - t _ { j } } ^ { k } ) ^ { - 1 } - ( S _ { T - t _ { j } } ^ { i } ) ^ { - 1 } \big \} ^ { 2 } S _ { T - t _ { j } } ^ { i } \big ) = \mathrm { t r } \big ( ( S _ { T - t _ { j } } ^ { i } ) ^ { - 1 } Y ^ { 2 } \big ) \leq \frac { 1 } { \sigma _ { T - t _ { j } } ^ { 2 } } \mathrm { t r } ( Y ^ { 2 } ) } \\ & { \qquad \leq \frac { 4 } { \sigma _ { T - t _ { j } } ^ { 2 } } \Big ( \frac { m _ { T - t } ^ { 2 } } { \sigma _ { T - t } ^ { 2 } } \| \Sigma _ { i } \| _ { 2 } + 1 \Big ) \cdot \mathrm { K L } _ { \mathrm { c } } \big ( \overrightarrow { p } _ { T - t } ^ { [ i ] } \mid \vert \overrightarrow { p } _ { T - t } ^ { [ k ] } \big ) } \end{array}\tag{80}
$$

where in the last line we used Lemma C.1 provided in Appendix C. Substituting the inequalities (79) and (80) into (78), we conclude.

## F. Details on Numerical Experiments

This section provides details of the numerical experiments presented in Section 4, and additional information to support the arguments of the section.

## F.1. Implementation details

Data processing For the scRNA-seq experiments on PBMC, we use the PBMC3K cell-by-gene count matrix as the raw data and apply a standard single-cell preprocessing workflow; see e.g., Satija et al. [2015] and Stuart et al. [2019]. First, we remove genes expressed in fewer than three cells and apply total-count normalisation with a target sum of $1 0 ^ { 4 }$ , followed by log-transformation. Then, for each ambient dimension $D \in \{ 5 0 0 , 1 0 0 0 , 2 0 0 0 , 4 0 0 0 , 8 0 0 0 \}$ , we select the top � highly variable genes using the Seurat-flavour procedure implemented in the Python package Scanpy [Wolf et al., 2018]. We retain the CD4 T, CD14+ Monocyte, and CD19+ B populations with the number of samples 1144, 480 and 342 respectively. We then apply gene-wise standardisation, and the resulting �-dimensional gene-expression vectors are used as the empirical target distribution for the reverse trajectories.

For the image experiment, we use the training split of the STL-10 dataset, whose images contain $9 6 \times 9 6$ pixels with 3 RGB channels. We retain three classes: Dog, Cat, and Airplane, and use 500 images from each class. To vary the ambient dimension, we construct three image resolutions $s \in \{ 3 2 , 6 4 , 9 6 \}$ corresponding to $D = 3 s ^ { 2 }$ . For � = 32 and $s = 6 4$ , the original images are downsampled using bilinear interpolation implemented by torch.nn.functional.interpolate in PyTorch. For $s = 9 6$ , we use the original STL-10 resolution. Finally, pixel intensities are rescaled from [0, 255] to [−1, 1].

Calculation of posterior weights and KL divergence We approximate the true intractable KL divergence between two noised clusters, using a Monte Carlo approximation based on the empirical target distribution. That is, for two distinct clusters �, � ∈ C or $C _ { \mathrm { i m a g e } }$ and $t \in [ 0 , T )$

$$
\mathrm { K L } \big ( \overrightarrow { p } _ { T - t } ^ { [ A ] } \mid | \overrightarrow { p } _ { T - t } ^ { [ B ] } \big ) = \int \log \frac { \overrightarrow { p } _ { T - t } ^ { [ A ] } ( x ) } { \overrightarrow { p } _ { T - t } ^ { [ B ] } ( x ) } \cdot \overrightarrow { p } _ { T - t } ^ { [ A ] } ( x ) d x \approx \int \log \frac { \overrightarrow { p } _ { T - t } ^ { [ A ] , N _ { A } } ( x ) } { \overrightarrow { p } _ { T - t } ^ { [ B ] , N _ { B } } ( x ) } \cdot \overrightarrow { p } _ { T - t } ^ { [ A ] , N _ { A } } ( x ) d x ,
$$

where $N _ { A }$ and $N _ { B }$ denote the numbers of empirical samples in classes � and �, respectively, and we define

$$
\begin{array} { r } { \overrightarrow { p } _ { T - t } ^ { [ i ] , N _ { i } } ( x ) \equiv \frac { 1 } { N _ { i } } \displaystyle \sum _ { j = 1 } ^ { N _ { i } } \overrightarrow { p } _ { T - t | 0 } ( x | Z _ { j } ) , \qquad \{ Z _ { j } \} _ { j } ^ { \mathrm { ~ i . i . d . } } p _ { 0 } ^ { [ i ] } , \quad i \in \{ A , B \} , } \end{array}\tag{81}
$$

where $p _ { 0 } ^ { [ i ] }$ is the population distribution for the class $\mathbf { \ddot { \boldsymbol { \ i } } } _ { i } ^ { \flat } .$ . Then, the estimate of the KL is finally given as:

$$
\begin{array} { r } { { \mathrm { K L } } \Big ( \underset { { T - t } } { \longrightarrow } \Vert \underset { { T - t } } { \longrightarrow } \Vert \underset { { T - t } } { \longrightarrow } \Big ) \approx \frac { 1 } { N _ { A } M } \underset { { l = 1 } } { \overset { N _ { A } } { \sum } } \underset { { k = 1 } } { \overset { M } { \sum } } \log \frac { \underset { { T - t } } { \longrightarrow } \dots { \Vert { A } } \dots { N _ { A } } ( { X } ^ { k , l } ) } { \underset { { P } _ { { T - t } } } { \longrightarrow } \dots { \Vert { B } } ( { X } ^ { k , l } ) } } \end{array}
$$

where $X ^ { k , l } = m _ { T - t } Z _ { l } + \sigma _ { T - t } \xi _ { k } , \ : \{ \xi _ { k } \} _ { k = 1 , \ldots , M } \stackrel { \mathrm { i . i . d . } } { \sim } N ( 0 , I _ { D } ) , \ : \{ Z _ { j } \} _ { j = 1 , \ldots , N _ { A } } \stackrel { \mathrm { i . i . d . } } { \sim } p _ { 0 } ^ { [ A ] } \ : \mathrm { i . i . d . }$ and � is the number of MC samples for the white noise. Since the true posterior weight for general target distributions is intractable, we estimate it using the empirical forward density (81) again as follows: for the set of classes C,

$$
\widehat { W } _ { D } ^ { [ i ] , N } ( t , x ) = \frac { \underset { T _ { T - t } } { \longrightarrow } [ i ] , N _ { i } } { \sum _ { c \in C } \underset { P _ { T - t } } { \longrightarrow } [ c ] , N _ { c } } , \quad i \in C , \quad ( t , x ) \in [ 0 , T ) \times \mathbb { R } ^ { D } ,
$$

where $N _ { c }$ denotes the number of samples in the class $c \in C$ , and we assume equal priors across all the classes.

Reverse SDE with Empirical Score As mentioned in Section 4, we generate reverse trajectories using the empirical score function in the drift. Following the notation above, we write the reverse SDE as:

$$
\begin{array} { r } { d \overleftarrow { X } _ { t } = ( \overleftarrow { X } _ { t } + 2 \nabla \log \overrightarrow { p } _ { T - t } ^ { N } ( \overleftarrow { X } _ { t } ) ) d t + \sqrt { 2 } d B _ { t } , \qquad \overleftarrow { X } _ { 0 } \sim N ( 0 , I _ { D } ) , } \end{array}
$$

where � is the total number of empirical data, including all classes. In the above, the empirical score is given as

$$
\nabla \log \overrightarrow { p } _ { T - t } ^ { N } ( x ) = \sum _ { c \in C } \widehat { W } _ { D } ^ { [ c ] , N } ( t , x ) \cdot \nabla \log \overrightarrow { p } _ { T - t } ^ { [ c ] , N _ { c } } ( x ) , \qquad x \in \mathbb { R } ^ { D } ,
$$

where ∇ log $\vec { p } _ { T - t } ^ { [ c ] , N _ { c } } ( x )$ is the cluster-wise empirical score with the number of samples $N _ { c }$ written as

$$
\begin{array} { r } { \nabla \log \overrightarrow { p } _ { T - t } ^ { [ c ] , N _ { c } } ( x ) = - \frac { 1 } { \sigma _ { T - t } ^ { 2 } } \big ( x - m _ { T - t } \mathbb { E } _ { N _ { c } } [ \overrightarrow { X } _ { 0 } ^ { [ c ] } | \overrightarrow { X } _ { T - t } ^ { [ c ] } = x ] \big ) , } \end{array}
$$

with

$$
\mathbb { E } _ { N _ { c } } \big [ \overrightarrow { X } _ { 0 } ^ { [ c ] } \big | \overrightarrow { X } _ { T - t } ^ { [ c ] } = x \big ] = \frac { 1 } { N _ { c } } \sum _ { j = 1 } ^ { N _ { c } } Z _ { j } \cdot \frac { \overrightarrow { p } _ { T - t | 0 } ( x | Z _ { j } ) } { \overrightarrow { p } _ { T - t } ^ { [ c ] , N _ { c } } ( x ) } , \{ Z _ { j } \} _ { j = 1 , \dots , N _ { c } } \cdot \operatorname { i . i . d } _ { n } \cdot p _ { 0 } ^ { [ c ] } .
$$

## F.2. Additional Information

We present the empirical pairwise geometry for the PBMC dataset. We observe that the mean separation increases with the ambient dimension � approximately linearly, while the maximum mean diference projection $M _ { \mu }$ is relatively stable across varying �.

Table 4. Pairwise empirical geometry of the PBMC clusters under variance-ordered feature truncation. The labels are the same as in Table 3.
<table><tr><td>Pair</td><td>D</td><td> $\| \widehat { \mu } _ { i } - \widehat { \mu } _ { j } \| ^ { 2 }$ </td><td> $M _ { \mu }$ </td><td> $\| \widehat { \Sigma } _ { i } - \widehat { \Sigma } _ { j } \| _ { F } ^ { 2 }$ </td></tr><tr><td rowspan="6">CD4 T-CD14+ Monocyte</td><td>500</td><td>31.55</td><td>1.48</td><td> $1 . 3 4 \times 1 0 ^ { 3 }$ </td></tr><tr><td>1000</td><td>91.95</td><td>2.67</td><td> $4 . 1 6 \times 1 0 ^ { 3 }$ </td></tr><tr><td>2000</td><td>240.40</td><td>4.37</td><td> $1 . 5 5 \times 1 0 ^ { 4 }$ </td></tr><tr><td>4000</td><td>352.70</td><td>4.62</td><td> $5 . 3 5 \times 1 0 ^ { 4 }$ </td></tr><tr><td>8000</td><td>509.33</td><td>5.68</td><td> $1 . 9 0 \times 1 0 ^ { 5 }$ </td></tr><tr><td>500</td><td>42.77</td><td>1.24</td><td> $2 . 4 0 \times 1 0 ^ { 3 }$ </td></tr><tr><td rowspan="4">CD14+ Monocyte-CD19+ B</td><td>1000</td><td>111.03</td><td>2.39</td><td> $7 . 3 1 \times 1 0 ^ { 3 }$ </td></tr><tr><td>2000</td><td>296.89</td><td>3.41</td><td> $2 . 9 4 \times 1 0 ^ { 4 }$ </td></tr><tr><td>4000</td><td>440.36</td><td>4.01</td><td> $1 . 1 6 \times 1 0 ^ { 5 }$ </td></tr><tr><td>8000</td><td>614.00</td><td>5.29</td><td> $4 . 0 4 \times 1 0 ^ { 5 }$ </td></tr><tr><td rowspan="5"> $\mathrm { C D 4 T \mathrm { - C D 1 9 + B } }$ </td><td>500</td><td>20.84</td><td>1.96</td><td> $1 . 7 0 \times 1 0 ^ { 3 }$ </td></tr><tr><td>1000</td><td>53.11</td><td>1.49</td><td> $5 . 2 1 \times 1 0 ^ { 3 }$ </td></tr><tr><td>2000</td><td>133.63</td><td>1.73</td><td> $2 . 0 9 \times 1 0 ^ { 4 }$ </td></tr><tr><td>4000</td><td>205.03</td><td>1.60</td><td> $8 . 8 5 \times 1 0 ^ { 4 }$ </td></tr><tr><td>8000</td><td>271.85</td><td>1.99</td><td> $3 . 1 9 \times 1 0 ^ { 5 }$ </td></tr></table>