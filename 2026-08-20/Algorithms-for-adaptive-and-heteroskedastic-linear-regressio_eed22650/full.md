# Algorithms for adaptive and heteroskedastic linear regression at the computational threshold

Spencer Compton and Tselil Schramm

## Abstract

We study finite-sample linear regression in the presence of varied and unknown label noise, focusing on the heteroskedastic and adaptive linear regression models.

Heteroskedastic linear regression models regression problems in which the labels or measurements are of varying quality. We receive n covariate, label pairs $( X _ { i } , Y _ { i } )$ with labels $Y _ { i } = X _ { i } ^ { \top } \beta + \varepsilon _ { i }$ , where each $\varepsilon _ { i } \sim N ( 0 , \sigma _ { i } ^ { 2 } )$ for a possibly distinct variance $\sigma _ { i } ^ { 2 }$ , and the variances $\sigma _ { i } ^ { 2 }$ are unknown to the estimator. One natural measurement of the dificulty of this problem is the number of samples m for which $\sigma _ { i } ^ { 2 } \leq 1$ (larger m is easier). Building on Compton and Valiant’s results for the simpler problem of mean estimation (STOC 2024), we obtain a polynomial-time estimator with rate ${ \tilde { O } } ( ( n d ^ { 3 } / m ^ { 4 } ) ^ { 1 / 6 } )$ when $m \gg d ^ { 3 / 4 } n ^ { 1 / 4 }$ , as well as nearlymatching lower bounds. For $d = O ( 1 )$ , our estimator achieves error $o ( 1 )$ so long as $m \gg n ^ { 1 / 4 }$ ， whereas $L _ { 1 }$ regression and other traditional approaches require $m \gg n ^ { 1 / 2 }$

Adaptive linear regression models regression problems in which we have limited information about the nature of the label noise. Here, the errors $\varepsilon _ { i }$ are drawn i.i.d. from an unknown distribution $p ,$ and our goal is to design a generic estimator that performs nearly as well as the best custom estimator that knows $p .$ To appreciate the challenge of this task, note that diferent parametric families exhibit dramatically diferent behavior: for example, when p is Normal the optimal rate is $\Theta ( \sqrt { d / n } )$ , whereas ${ \mathrm { i f ~ } } p$ is Uniform[−1, 1] the optimal rate is $\tilde { \Theta } ( d / n )$ , and the maximum likelihood estimator in each case is quite diferent. We introduce a (computationally ineficient) adaptive estimator that, so long as $p$ is a mixture of k symmetric log-concave densities, achieves error comparable with the optimal estimator that knows p and has $\tilde { \Theta } ( \boldsymbol n / \boldsymbol k )$ samples. For the special case $k = 1$ , we show that $L _ { q }$ regression (with data-dependent q) gives a polynomialtime estimator.

Finally, in order to study the computational limits of both problems, we introduce the planted linear regression problem, where $X _ { i } \sim N ( 0 , I _ { d } )$ , m unknown samples are noiseless, and the rest have error $\varepsilon _ { i } \sim N ( 0 , 1 )$ . We conjecture that recovering $\beta$ up to error $\ll \sqrt { d / n }$ (or exactly) may have an information-computation gap between $m = d + 1$ and $m \sim d ^ { 3 / 4 } \dot { n } ^ { 1 / 4 }$ , as is suggested by our near-matching polynomial-time estimator and statistical query (SQ) lower bound.

## Contents

1 Introduction 3   
1.1 Preliminaries 6   
1.2 Our contributions 8   
1.3 Related work 13   
1.4 Discussion 16   
2 Heteroskedastic linear regression 18   
2.1 A simple estimator for standard Gaussian covariates 18   
2.2 An estimator for well-conditioned, sub-Gaussian covariates 19   
3 Adaptive linear regression 27   
3.1 Adaptive guarantees for symmetric, log-concave mixture errors 27   
3.2 Eficient estimation for symmetric, log-concave errors (k = 1) 31   
3.3 Statistical lower bound 43   
Planted linear regression 52   
4.1 Exact recovery via heteroskedastic linear regression 52   
4.2 Statistical Query lower bound . 53   
4.3 Barrier for convex M-estimation 60   
5 Simulations 63   
5.1 Planted linear regression. 64   
5.2 Adaptive linear regression. 65   
A Subset-of-Signals statistical lower bound 75   
B Additional simulation figures 79   
C Proof of uniform convergence results 79   
C.1 Proof of Proposition 2.5 79   
C.2 Proof of Lemma 3.2 85   
D Additional deferred proofs 86   
D.1 Proof of Claim D.1 86   
D.2 Proof of Proposition 3.5 86   
D.3 Proof of Corollary 3.6 90   
D.4 Smoothed uniform errors example . 91

## 1 Introduction

Linear regression is the bread and butter of statistical prediction and modeling. Developing linear regression estimators that best adapt to unknown noise distributions is a core challenge. There is a mature, insightful body of work for this task in the asymptotic setting, where typically the noise distribution is fixed, has finite Fisher information, and then $n \to \infty$

In our work, we study linear regression with unknown noise in the finite-sample setting, which is much less understood. We allow the noise distribution to be chosen after $n ,$ and we do not assume finite Fisher information. A strikingly diferent picture emerges when we relax these assumptions: the optimal rates are diferent, the estimators achieving these rates are diferent, and the guarantees pertain to more problems of recent interest. Our study focuses on three representative tasks: heteroskedastic, adaptive, and planted linear regression.

Heteroskedastic linear regression. How much signal can you extract from data when its labels have unknown, varying quality? This is a question that naturally must be answered when handling data from distinctive sources; for example, when data is scraped from multiple collections, or crowdsourced from users of unequal skill. Formally, in heteroskedastic linear regression we receive n samples of the form $( X _ { i } , Y _ { i } )$ , and each $Y _ { i } = X _ { i } ^ { \top } \beta + \varepsilon _ { i }$ , where $\varepsilon _ { i } \sim N ( 0 , \sigma _ { i } ^ { 2 } )$ for diferent variances $\sigma _ { i } ^ { 2 }$ with unknown values.

A comprehensive treatment of such problems was initiated by Chierichetti, Dasgupta, Kumar, and Lattanzi [CDKL14] for heteroskedastic mean estimation: where we aim to learn a onedimensional mean $\mu$ from samples $X _ { i } \sim N ( \mu , \sigma _ { i } ^ { 2 } )$ with unknown $\sigma _ { i } ^ { 2 }$ . In this setting, estimators such as the empirical mean are quite undesirable; even just one sample with extremely large variance can render the estimator useless. This has motivated the study of alternative estimators which are better-suited for this task, such as the empirical median, iterative trimming, and modal estimators [CDKL14, Xia19, YL20, PJL22, DLLZ23, Lou25]. In the heteroskedastic setting, the rate achieved by each estimator may be a sophisticated function of $\sigma _ { 1 } , \ldots , \sigma _ { n }$ , making it dificult to compare estimators. As a point of comparison, Liang and Yuan [LY20] suggested an interpretable “Subset-of-Signals” benchmark: “What is the best error guaranteed by an estimator (as a function of n, m) if at least m of the samples have $\sigma _ { i } \leq 1$ , but the estimator is not told which samples?” This perspective makes comparing the guarantees of diferent estimators much easier; estimators such as the median and other previously-discussed works require $m \geq \Omega ( n ^ { 1 / 2 } )$ samples to attain estimation error $\| \widehat { \mu } - \mu \| _ { 2 } \ll 1$ , whereas a lower bound of [LY20] only requires $m \ge \Omega ( n ^ { 1 / 4 } )$ . Compton and Valiant [CV24] designed a new balance-finding estimator for this setting, obtaining near-optimal errors for the Subset-of-Signals benchmark and demonstrating that error $\ll 1$ is possible even with only $m \geq n ^ { 1 / 4 + o ( 1 ) }$ bounded-variance samples.

Just as the empirical mean fails in the heteroskedastic setting, the OLS estimator degrades dramatically in the presence of even one high-variance label. There has been a similar study into alternative estimators for heteroskedastic regression [YL20, PJL22, CB26], and understanding the statistical and computational landscape has been highlighted as an open direction in the recent survey of $[ \mathrm { M S B ^ { + } 2 6 } ]$ , where they remark that $^ { 6 6 } \mathrm { { \dot { 1 } t } }$ is largely unclear whether polynomial-time algorithms can achieve optimal rates in these heterogeneous high-dimensional settings.” A natural open question is whether there exists a polynomial-time estimator which similarly succeeds in the Subset-of-Signals setting for regression when only $m \geq n ^ { 1 / 4 + o ( 1 ) }$ out of the n labels have bounded variance. In our work, we resolve this positively: obtaining error $\ll 1$ when $m \gg d ^ { 3 / 4 } n ^ { 1 / 4 }$ . More generally, we show this is statistically optimal, and obtain near-optimal Subset-of-Signals error in this regime of $m$

![](images/be1a0e65073104f3090b06ba2cf1a7f0772b7302ec843eb5ab89b627dd526c52.jpg)  
(a)

![](images/54f5457c7d28a7c5bdbc821fa16757a36d3848c9563834d747133b440f5f4e1f.jpg)  
(b)

![](images/afb2854f04a7c38601569c21a639a7f245d1bf9354cbd9f0e919c5b58a28d479.jpg)  
(c)  
Figure 1: Performance for Gaussian, uniform, and smoothed uniform errors (see Section 5).

Adaptive linear regression. In the task of adaptive linear regression, errors are drawn i.i.d. from an unknown distribution $p .$ The main challenge is to understand which conditions enable an adaptive estimator, which does not know $p ,$ to estimate nearly as well as a custom estimator with full knowledge of $p .$

Even if we are willing to restrict to p which are symmetric and log-concave, this problem is quite nuanced. Consider three examples (with simulations in Fig. 1):

• Gaussian errors: $N ( 0 , 1 )$ . The optimal estimation error $\left\| \widehat { \beta } - \beta \right\| _ { 2 }$ is on the order of ${ \sqrt { d / n } } ;$ this is achieved by OLS.

• Uniform errors: $\operatorname { U n i f } [ - 1 , 1 ]$ . Dramatically better estimation error of ${ \tilde { O } } ( d / n )$ is possible; this is achieved by minimizing the $L _ { \infty }$ norm of the residuals [YN24, Theorem 2.2, Example 2.10].

• Smoothed uniform errors: $\mathrm { U n i f } [ - 1 , 1 ] { * } N ( 0 , \sigma ^ { 2 } )$ . When $\sigma = n ^ { - a }$ for $a \in ( 0 , 1 ]$ , then previouslyknown techniques do not determine the optimal error (to our knowledge); our later results will imply that the optimal error is on the order of max $\textstyle \left\{ { \sqrt { \frac { d } { n ^ { 1 + a } } } } , { \frac { d } { n } } \right\}$ (ignoring logarithmic factors; see Section D.4), which is attained by neither OLS nor $L _ { \infty }$ regression.

In each case, the optimal rate is diferent, and is achieved by a diferent estimator. A key question is whether there exists a single estimator that may adapt to all such distributions.

While analogous questions in the asymptotic setting are much better understood (we point to [FKXS26] as an illuminating study), the limits of adaptivity with finite samples (and no bound on Fisher information) are comparatively uncharted. This concerns not only the simple example of uniform errors (with infinite Fisher information), but also more interesting errors such as: (i) smoothed uniform errors (where the smoothing may depend on n), or (ii) a mixture of two heteroskedastic mean-zero Gaussians (where the mixing weight and variances may depend on n). The heteroskedastic mixture is an instructive case for the importance of the finite-sample perspective; if we treated it asymptotically, both components would have a constant-fraction of the mixing weight, and we would not observe the interesting phase transition behaviors that appear in Subset-of-Signals heteroskedastic estimation.

In the case of symmetric, log-concave errors (which includes all examples in Fig. 1), we show that polynomial-time $L _ { q }$ regression has the desired adaptive behavior when q is chosen in a datadependent manner. This style of estimator is comprehensively studied by Kao, Xu, and Zhang [KXZ24], but our result does not follow from existing analyses; we discuss the technical obstacles later. Roughly, our result proves that for any symmetric, log-concave $p _ { \mathrm { : } }$ , this $L _ { q }$ estimator given n samples performs nearly as well as the best estimator that knows $p$ when it is given access to $\frac { n } { \log ( n ) }$ samples.

For more general errors, the most relevant prior work is a diferent paper of Compton and Valiant [CV26], where they study the analogous problems in the setting of univariate mean estimation. Their main result considers the setting where $p$ is a mixture of $k$ symmetric, log-concave distributions (which includes all examples mentioned so far); they design an adaptive estimator that, with $n$ samples, performs nearly as well as the best estimator that knows $p$ yet has $\frac { n } { k \operatorname { p o l y l o g } ( n ) }$ samples. They also prove that when $p$ is only symmetric and unimodal, it is impossible to attain the analogous result competing against an estimator with $\frac { n } { \log ( n ) }$ samples that knows $p .$ . Under the hood, both of these results leverage the Hellinger modulus $o f$ continuity to characterize the optimal error (we discuss the Hellinger modulus in the preliminaries).

In our work, we design a computationally ineficient estimator that yields the analogous positive result for linear regression with symmetric log-concave mixture errors. Hence, the adaptive $L _ { q }$ estimator runs in polynomial-time for $k = 1$ , yet we have no eficient estimator for $k \geq 2$

Planted linear regression. For the purpose of studying the information-computation landscape of both heteroskedastic and adaptive linear regression, we suggest the following problem, which we call “planted linear regression.”

Definition 1.1 (Planted linear regression). In planted linear regression, the covariates $X _ { i } \sim N ( 0 , I _ { d } )$ and the labels $Y _ { i } = X _ { i } ^ { \top } \beta + \varepsilon _ { i }$ for exchangeable $\varepsilon _ { i } ,$ exactly m of which are zero and the remaining are drawn i.i.d. from $N ( 0 , 1 )$ . A similar i.i.d. version draws each $\varepsilon _ { i }$ from the mixture $\begin{array} { r } { \frac { m } { n } \delta _ { 0 } + \frac { n - m } { n } N ( 0 , 1 ) } \end{array}$ . The goal is to recover $\beta$ exactly (or with much better error than the OLS rate $\sqrt { d / n } )$

Once $m \geq d + 1$ , this task is information-theoretically trivial: any hyperplane containing at least d+1 samples must be the true $\beta$ almost surely. However, it is not clear how to do this in polynomial time; a naive implementation would run in $n ^ { \Omega ( d ) }$ time. Most well-studied eficient estimators for linear regression are convex M-estimators, yet we later show that no convex M-estimator exactly recovers $\beta$ when $m \ll { \sqrt { d n } }$ . In contrast, our estimator for heteroskedastic regression can recover $\beta$ once $m \gg d ^ { 3 / 4 } n ^ { 1 / 4 }$ . The success of our method in the regime $n ^ { 1 / 4 } d ^ { 3 / 4 } \ll m \ll \sqrt { d n }$ highlights the value of going beyond traditional estimators in finite-sample settings. We note that planted linear regression is a special case of noiseless linear regression (see $[ \mathrm { D G K ^ { + } 2 5 } ]$ ; we discuss more in Section 1.3).

In the regime $d + 1 \leq m \ll d ^ { 3 / 4 } n ^ { 1 / 4 }$ , recovering $\beta$ is information-theoretically possible, yet we know no polynomial-time algorithm; we conjecture that this is an information-computation gap, and we prove a statistical query (SQ) lower bound in support of this claim (Theorem 4.3).

Additionally, we remark that planted linear regression is a special case of heteroskedastic linear regression, and of adaptive linear regression (in the i.i.d. formulation). Our heteroskedastic regression estimator only yields polynomial-time Subset-of-Signals guarantees when m $\gg d ^ { 3 / 4 } n ^ { 1 / 4 }$ ; obtaining poly $( n , d , 1 / \delta )$ Subset-of-Signals estimation error in poly $( n , d )$ -time when $m \ll d ^ { 3 / 4 } n ^ { 1 / 4 }$ would imply an estimator for planted linear regression below the suggested computationally-hard regime. Likewise, a polynomial-time version of the adaptive linear regression guarantee (Theorem 1.7) for $k = 2$ would imply an estimator for planted linear regression when $m \gg d \mathrm { p o l y l o g } ( n , d )$ This indicates an intertwined computational-statistical landscape for these three problems. We illustrate the planted linear regression landscape in Fig. 2a.

![](images/272721feb64eb254d1da63e0e74dea258e0e15a51e94763af60405a0cfc2c4b6.jpg)

![](images/355a520293c81cabd82641994c6fcfb2a069e3dd940c6f64cd200d7943052601.jpg)  
(a) Our understanding of exact recovery for planted linear regression as a function of m.  
(b) Empirical probability of recovery within $\left. \widehat { \beta } - \beta \right. _ { 2 } \leq 1 0 ^ { - 5 }$ (over 100 repetitions) for planted linear regression (see discussion in Section 5).  
Figure 2: Planted linear regression.

## 1.1 Preliminaries

For all linear regression tasks, the samples will take the form $( X _ { i } , Y _ { i } ) \in \mathbb { R } ^ { d } \times \mathbb { R }$ , where $Y _ { i } = X _ { i } ^ { \top } \beta { + } \varepsilon _ { i }$ The errors $\varepsilon _ { i }$ are independent from other samples, and independent of $X _ { i }$ . We will exclusively consider the setting where $X _ { i }$ are i.i.d., and for constants $K , \kappa \geq 1$ , we assume they have a κ well-conditioned second-moment matrix

$$
\Sigma = \operatorname { E } [ X X ^ { \top } ] \qquad \kappa ^ { - 1 } I \preceq \Sigma \preceq \kappa I\tag{1}
$$

and are K sub-Gaussian

$$
\begin{array} { r } { \| \langle u , X \rangle \| _ { \psi _ { 2 } } \leq K \sqrt { u ^ { \top } \Sigma u } \qquad \mathrm { f o r ~ e v e r y ~ } u \in \mathbb { R } ^ { d } , } \end{array}\tag{2}
$$

as would be the case when the variables have already been appropriately preprocessed/whitened. Constants throughout the proofs may depend on K and κ implicitly; we will be explicit in our theorem statements when the constants depend on $K , \kappa .$

Throughout the paper we will use $c , C > 0$ to denote constants, which may vary line-by-line and only depend on K, κ. We use $a \lesssim b$ to denote the same meaning as $a \leq C b$ (meaning a is at most a constant multiplied by b); and similarly for $\gtrsim$ notation. Let $a \asymp b$ denote both $a \lesssim b$ and $a \gtrsim b$ $O _ { K , \kappa } ( \cdot )$ notation suppresses multiplicative factors depending only on $K , \kappa .$ . We use log to denote the natural logarithm. We only study and discuss measurable sets.

For a candidate estimate $b \in \mathbb { R } ^ { d }$ , we denote the ith residual as

$$
r _ { i } ( b ) = y _ { i } - X _ { i } ^ { \top } b .\tag{3}
$$

We often encounter symmetric, unimodal random variables (in 1-dimension), and use the following known decomposition (e.g. see [Jon02]) of any such random variable:

Definition 1.2 (Symmetric unimodal decomposition). A real random variable ε is symmetric unimodal about zero if

$$
\varepsilon \overset { d } { = } T U ,
$$

where $U \sim \operatorname { U n i f } [ - 1 , 1 ] , T \geq 0$ , and $T$ is independent of $U$

In this representation, $\mathrm { V a r } ( \varepsilon ) = \mathrm { E } T ^ { 2 } / 3 ;$ this means $\mathrm { V a r } ( \varepsilon ) \leq 1$ is equivalent to $\mathrm { E } T ^ { 2 } \leq 3$

For a 1-dimensional distribution $p ,$ let $p _ { \mu } ( x )$ denote the same distribution translated by $\mu ,$ meaning $p _ { \mu } ( x ) = p ( x - \mu )$

Hellinger modulus of continuity. Hellinger distance is central to our work. The squared Hellinger distance between two distributions $P , Q$ with densities $p , q$ is defined as

$$
\mathrm { H } ^ { 2 } ( P , Q ) = { \frac { 1 } { 2 } } \int _ { \mathcal { X } } \left( { \sqrt { p ( x ) } } - { \sqrt { q ( x ) } } \right) ^ { 2 } d x .
$$

A crucial property of Hellinger distance is that it tensorizes, meaning the distance between product distributions may be written in terms of the distance between the marginals:

Fact 1.3 (e.g. page 45 of $\big [ \mathrm { L C Y 0 0 } \big ] \big )$ . Let $P ^ { \otimes n }$ and $Q ^ { \otimes n }$ denote the distribution of n i.i.d. samples from $P$ and $Q$ respectively. Then,

$$
\mathrm { H } ^ { 2 } ( P ^ { \otimes n } , Q ^ { \otimes n } ) = 1 - ( 1 - \mathrm { H } ^ { 2 } ( P , Q ) ) ^ { n } .
$$

Squared Hellinger distance is also quadratically related to TV distance:

Fact 1.4 (e.g. page 44 of [LCY00]). $\mathrm { H } ^ { 2 } ( P , Q ) \leq \mathrm { T V } ( P , Q ) \leq \sqrt { 2 \mathrm { H } ^ { 2 } ( P , Q ) } .$

Using both facts, the TV distance $\mathrm { T V } ( P ^ { \otimes n } , Q ^ { \otimes n } )$ can be bounded in terms of the squared Hellinger distance between the marginals,

$$
\begin{array} { r } { 1 - ( 1 - \mathrm { H } ^ { 2 } ( P , Q ) ) ^ { n } \lesssim \mathrm { T V } ( P ^ { \otimes n } , Q ^ { \otimes n } ) \lesssim \sqrt { 1 - ( 1 - \mathrm { H } ^ { 2 } ( P , Q ) ) ^ { n } } . } \end{array}
$$

Thus, the value of $n$ where $\mathrm { T V } ( P ^ { \otimes n } , Q ^ { \otimes n } )$ goes from ≈ 0 to ≈ 1 is $n = \Theta ( 1 / \mathrm { H } ^ { 2 } ( p , q ) )$ . In this sense, the squared Hellinger distance between two distributions characterizes how many samples are required to distinguish them (while TV distance does not). This is helpful for cleanly establishing lower bounds; we will focus on lower bounds in the context of mean estimation. If $\begin{array} { r } { \mathrm { H } ^ { 2 } ( p , p _ { \mu } ) \leq \frac { 1 } { n } . } \end{array}$ then with only n samples, any hypothesis tester must fail with some constant probability, and hence (by Le Cam’s two-point testing method) must incur error $\mu / 2$ with constant probability. Donoho and Liu [DL87, DL91a, DL91b] established the influential concept of the Hellinger modulus of continuity, which captures this style of argument. In the context of mean estimation, the Hellinger modulus is the function

$$
\omega _ { p } ( \varepsilon ) : = \operatorname* { s u p } \{ | \mu _ { 1 } - \mu _ { 2 } | : \mathrm { H } ^ { 2 } ( p _ { \mu _ { 1 } } , p _ { \mu _ { 2 } } ) \leq \varepsilon , \mu _ { 1 } , \mu _ { 2 } \in \mathbb { R } \} = \operatorname* { s u p } \{ t : H _ { p } ( t ) \leq \varepsilon , t \geq 0 \} ,
$$

where $H _ { p } ( t )$ is shorthand for $\mathrm { H } ^ { 2 } ( p , p _ { t } )$ . By the same reasoning, any mean estimator for the class of translations of $p$ with access to only n samples must incur error $c \cdot \omega _ { p } ( { \textstyle { \frac { 1 } { n } } } )$ with at least constant probability.

This perspective is widely useful both for providing lower bounds, and for obtaining upper bounds in terms of the Hellinger modulus (for just a few notable examples of the influence of this modulus perspective, see [JN09, CL15, FKQR21, DR24, PW26, HSS26]). In the context of adaptive location estimation, the lower bound is immediate by definition, and the related work of [CV26] studies conditions where the Hellinger modulus error is nearly achievable. For mixtures of k symmetric, log-concave distributions, they design an adaptive estimator with error on the order of $\overset { \cdot } { \omega } _ { p } \bigl ( \frac { k \mathrm { p o l y l o g } ( n ) } { n } \bigr )$ ; combined with the immediate Hellinger modulus lower bound, this implies that their adaptive estimator with n samples performs nearly as well as the best custom estimator that knows p and has $\frac { n } { k \operatorname { p o l y l o g } ( n ) }$ samples. Our work will prove upper and lower bounds for adaptive linear regression in terms of the Hellinger modulus for 1-dimensional mean estimation (neither direction is immediate).

## 1.2 Our contributions

We now outline our results for heteroskedastic, adaptive, and planted linear regression. We additionally run experiments in Section 5.

Polynomial-time heteroskedastic linear regression. We design a polynomial-time algorithm that yields near-optimal guarantees for the Subset-of-Signals benchmark (we allow any symmetric, unimodal error, not just $N ( 0 , \sigma _ { i } ^ { 2 } ) )$ ):

Theorem 1.5. Suppose $\| \beta \| _ { 2 } \le B$ and the distribution of X is κ well-conditioned and K sub-Gaussian. There exists a constant $C _ { K , \kappa } \geq 1$ depending only on K, κ such that the following holds. Our residual-balance descent estimator (RB-Desc) takes in parameters $n , d , B , \delta , K , \kappa$ where $n \geq$ $d \geq 1 , B , K , \kappa \geq 1$ , and $\delta \in ( 0 , 1 / 2 ]$ . Suppose

$$
m \geq C _ { K , \kappa } \log ^ { 5 } \left( \frac { n d B } { \delta } \right) d ^ { 3 / 4 } n ^ { 1 / 4 } .\tag{4}
$$

Assume that each $\varepsilon _ { i }$ is distributed according to a symmetric unimodal distribution $p _ { i }$ , m of which have variance at most 1. Given access only to the samples (and not to m nor the variances of the $p _ { i } )$ , with probability at least $1 - \delta$ , RB-Desc returns $\widehat { \beta }$ satisfying

$$
\left\| \widehat { \boldsymbol { \beta } } - \boldsymbol { \beta } \right\| _ { 2 } \leq C _ { K , \kappa } \log ^ { 5 } \left( \frac { n d B } { \delta } \right) \cdot \sqrt { d } \left( \frac { n } { m ^ { 4 } } \right) ^ { 1 / 6 } ,\tag{5}
$$

and runs in time

$$
O _ { K , \kappa } \left( n ^ { 2 } d \log ^ { 2 } \left( \frac { n d B } { \delta } \right) \right) .
$$

We give a nearly matching minimax lower bound in Section $\mathrm { A }$

We briefly ofer some intuition for the proof. We take inspiration from the balance-finding algorithm of Compton and Valiant [CV24] for 1-dimensional heteroskedastic mean estimation. The main idea of their algorithm is that any suficiently bad estimate $\widehat { \mu }$ of the mean will exhibit “imbalance” at some scale $w > 0 \mathrm { : }$ if $\widehat \mu < \mu$ and $\| { \widehat { \mu } } - \mu \| _ { 2 }$ is suficiently large, this means that there exists some $w > 0$ where the number of samples in $[ \widehat { \mu } - w , \widehat { \mu } ]$ is noticeably smaller than the number of samples in $\textstyle \left[ { \widehat { \mu } } , { \widehat { \mu } } + w \right]$ . Accordingly, whenever some candidate estimate $\widehat { \mu }$ has large imbalance to the left or right, we may restrict our search to that direction. As w increases, the expected diference in sample counts becomes more favorable, yet the empirical fluctuations get worse; the optimal choice of w entails a tradeof that depends on the collection of variances. For example, if m samples have $\sigma _ { i } \leq 1$ and the remaining $n - m$ samples have extremely large $\sigma _ { i }$ , then a “bump” in the histogram may be noticeable around $\mu$ at scale $w = 1$ ; however, if the $n - m$ samples have moderate values of $\sigma _ { i } .$ then they may obscure the bump at scale 1, yet the signal may be visible with a larger choice of w. [CV24] search over many choices of w to obtain near-optimal Subset-of-Signals guarantees for mean estimation. Unfortunately, there is not an obvious analog for this approach in linear regression.

A well-known reduction from linear regression to mean estimation entails using that if X is mean zero with identity covariance, then $\operatorname { E } [ X _ { i } y _ { i } ] = \beta _ { \mathrm { { : } } }$ , and hence we may simply run a mean estimation procedure on $X _ { i } y _ { i }$ (e.g. see [DK23, Section $7 . 2 ] )$ . This reduction does not work for our heteroskedastic setting (the distribution of $X y$ is not even Gaussian), but we will use a similar intuition. When $y _ { i }$ is very positive, it stands to reason that typically $X _ { i } ^ { \top } \beta$ is very positive; similarly, when $y _ { i }$ is very negative, typically $X _ { i } ^ { \top } \beta$ is very negative. Accordingly, it makes sense that $\operatorname { E } [ X _ { i } \operatorname { s g n } ( y _ { i } ) ]$ may correlate with $\beta ;$ this has been understood since at least the work of Manski [Man85]. In our algorithm, we use a related statistic with a scale parameter $w$

$$
S _ { w } ( b ) : = \sum _ { i = 1 } ^ { n } X _ { i } \mathrm { s g n } ( r _ { i } ( b ) ) { \bf 1 } \left\{ | r _ { i } ( b ) | \leq w \right\} .
$$

This is the (negative) subgradient of a clipped $\mathrm { L A D }$ objective $\scriptstyle \sum _ { i = 1 } ^ { n }$ min $\{ | r _ { i } ( b ) | , w \}$ , and is studied in the works of [Rou82, HRRS86, LL23]. Our analysis focuses on proving that when some candidate b is suficiently far from $\beta ,$ then for some scale $w ,$ the statistic $S _ { w } ( b )$ will strongly correlate with $\beta - b$ This can be viewed as a form of imbalance in this direction. Our algorithm is a residual-balance descent, where we repeatedly choose a set of scales, aggregate their statistics $S _ { w } ( b )$ , and move in this direction.

We remark that in the special case of standard Gaussian covariates $( X _ { i } \sim N ( 0 , I _ { d } ) )$ , designing an algorithm is much simpler; we discuss this in Section 2.1.

Adaptive linear regression for symmetric, log-concave errors. We design a polynomialtime algorithm (with guarantees in terms of the Hellinger modulus) for adaptive linear regression when the errors are symmetric and log-concave:

Theorem 1.6. Suppose the distribution of X is κ well-conditioned, K sub-Gaussian, and the distribution $o f \ \varepsilon _ { i } \stackrel { i . i . d . } { \sim } \ p$ is symmetric and log-concave. There exists a constant $C _ { K , \kappa } \geq 1$ depending only on K, κ such that, with probability at least $1 - \delta$ (for $\delta \in ( 0 , 1 / 2 ] )$ , our adaptive $L _ { q }$ estimator returns $\widehat { \beta }$ satisfying

$$
\left\| \widehat { \beta } - \beta \right\| _ { 2 } \leq C _ { K , \kappa } \cdot \omega _ { p } \left( C _ { K , \kappa } \frac { d \log ^ { 4 } ( 6 4 n / \delta ) } { n } \right) + \gamma ,\tag{6}
$$

for $\gamma \in ( 0 , 1 ]$ , where the estimator runs in poly $\cdot ( n , d ) \cdot \mathrm { p o l y l o g } ( 1 / \delta , \| \mathbf { Y } \| _ { \infty } , 1 / \gamma )$ time.

We complement this result with a lower bound which shows it is impossible to obtain a rate better than $c \cdot \omega _ { p } ( c d / n )$

Again, we ofer some intuition for the proof. Our algorithm outputs an optimizer for $L _ { q }$ regression

$$
\underset { b \in \mathbb { R } ^ { d } } { \mathrm { a r g } } \operatorname* { m i n } \sum _ { i = 1 } ^ { n } | r _ { i } ( b ) | ^ { q } ,
$$

where the power $q$ is chosen in a data-dependent manner. Kao, Xu, and Zhang [KXZ24] initiate a helpful study of this style of estimator, primarily in the context of adaptive location estimation (but also touching on adaptive linear regression). For mean estimation, their main analyses (Theorems 2.1 and 3.1) demonstrate guarantees for their adaptive estimator that relate to the performance of the $L _ { 2 }$ and $L _ { \infty }$ estimator. Though their guarantees are strong, they fail to cover all symmetric log-concave $p .$ For example, both $q = 2$ and $q = \infty$ give polynomially-suboptimal guarantees for the previously-discussed “smoothed uniform errors” example $( \mathrm { U n i f } [ - 1 , 1 ] * N ( 0 , n ^ { - 2 a } )$ for $0 < a < 1 )$ , which is symmetric and log-concave. With examples like this in mind, it is a priori unclear whether one should expect an adaptive $L _ { q }$ estimator to work for all symmetric, log-concave distributions: is there always a choice of $q$ that recovers the desired guarantees?

We answer this question in the afirmative. As indicated by prior works (e.g. [LL05, EHE23, KXZ24]), the performance on $L _ { q }$ regression is informed by the moments of the 1-dimensional error distribution $p ;$ we must prove the appropriate moment bounds. With work, this will follow by demonstrating a certain control over the tails of $p .$ At first glance, it is not clear why p must have tail control that relates in a meaningful way to its Hellinger modulus. The key tool is that for any p and $p _ { \mu }$ with large Hellinger distance, they may be distinguished by counting the number of samples inside a particular halfline (this is observed by [CV26] and follows from the reverse data processing inequality of [PJL23]). In this sense, the distance between $p$ and $p _ { \mu }$ is almost entirely from the tails, and this ultimately yields the necessary tail control (a more substantive proof intuition is given in Section 3.2).

Proving the existence of a good $q$ is the main dificulty; adaptively choosing the value of $q$ is much simpler. Roughly, we break the samples into blocks, perform $L _ { q }$ regression for all blocks and a collection of choices for $q ,$ and then choose the value of $q$ for which the empirical $L _ { q }$ estimates concentrate best (this style of technique has appeared before in e.g. [NRS07, HS14]).

Adaptive linear regression for mixtures of k symmetric, log-concave errors. We design a (computationally intractable) estimator for adaptive linear regression when the errors are mixtures of symmetric, log-concave distributions:

Theorem 1.7. Suppose the distribution of X is κ well-conditioned, K sub-Gaussian, and the distribution of $\varepsilon _ { i } \stackrel { i . \ i . d . } { \sim }$ p is a mixture of k symmetric, log-concave densities. There exists a constant $C _ { K , \kappa } \geq 1$ depending only on K, κ such that, with probability at least $1 - \delta$ , our (computationally intractable) estimator $\widehat { \beta }$ satisfies

$$
\left\| { \widehat { \beta } } - \beta \right\| _ { 2 } \leq C _ { K , \kappa } \cdot \omega _ { p } \left( C _ { K , \kappa } { \frac { d k \log ( 2 n / \delta ) \log ( 2 k n ) \log ( 2 n ) } { n } } \right) .\tag{7}
$$

Compton and Valiant [CV26] design an analogous polynomial-time estimator for the task of 1- dimensional adaptive location estimation when $p$ is likewise a mixture of k symmetric, log-concave densities; our estimator will be a natural high-dimensional generalization of their technique.

In their adaptive location estimation algorithm, they output any estimate $\widehat { \mu }$ such that the empirical samples look symmetric around $\widehat { \mu }$ according to interval tests: meaning the number of samples in $[ \widehat { \mu } - b , \widehat { \mu } - a ]$ is approximately the same as in $[ \widehat { \mu } + a , \widehat { \mu } + b ]$ , for all $0 \leq a < b < \infty$ . The success of this estimator is proven by a technical result where they show that whenever $p$ and $p _ { t }$ have large Hellinger distance (and $p$ is a symmetric, log-concave mixture with small $k )$ , then there

exists an interval $[ l , r ]$ where

$$
\left( \sqrt { \mathbb { P } _ { p } ( x \in [ l , r ] ) } - \sqrt { \mathbb { P } _ { p _ { t } } ( x \in [ l , r ] ) } \right) ^ { 2 }
$$

is large. In other words, an interval $[ l , r ]$ “witnesses” the Hellinger distance between translations.   
The proof of this builds upon the reverse data processing inequality of Pensia, Jog, and Loh [PJL23].

We design an estimator with an analysis that leverages this technical result. Suppose $b \in \mathbb { R } ^ { d }$ is a candidate estimate where $\| \boldsymbol { b } - \boldsymbol { \beta } \| _ { 2 }$ is large, and let $h = \beta - b$ . The residuals satisfy

$$
r _ { i } ( b ) = y _ { i } - X _ { i } ^ { \top } b = \varepsilon _ { i } + X _ { i } ^ { \top } h .
$$

Suppose we defined a threshold $s > 0$ , and two sets where $S _ { + }$ contains all the residuals for samples where $X _ { i } ^ { \top } h \geq s .$ , and $S _ { - }$ contains all the residuals for samples where $X _ { i } ^ { \top } h \leq - s$ . Observe how each sample in $S _ { + }$ , conditioned on $X _ { i } .$ , is distributed like $\varepsilon _ { i }$ with a positive translation of at least s (and similarly for S<sub>−</sub> with negative translations). If b is a very poor estimate, we may expect that for the right choice of $s ,$ the means of $S _ { + }$ and S<sub>−</sub> will be very diferent, and we want to detect when this occurs, so we may rule out this candidate estimate.

For any candidate estimate $b ,$ if there exists a choice of projection direction $u \in \mathbb { R } ^ { d }$ , threshold $s > 0$ , and an interval $I = [ l , r ]$ , where the number of samples from $S _ { + }$ in I is substantially diferent than the number from $S _ { - }$ , we will conclude b is a poor estimate. By comparison, if $b = \beta$ , then $X _ { i } ^ { \top } u$ is independent of $r _ { i } ( \beta )$ for any $u ,$ so the sets $S _ { + } , S _ { - }$ <sub>−</sub> are drawn from the same distribution, and $\beta$ should not fail any such test. The estimator is ineficient since we require a search over all $u \in \mathbb { R } ^ { d }$ to test a candidate estimate; a poor estimate b will violate the test for $u = h$ . Naturally, our analysis heavily leverages the technical interval witness result in [CV26]. In reality, we use slightly diferent sets than $S _ { + } , S _ { - }$ <sub>−</sub>, but the intuition remains the same.

A complementary lower bound for linear regression in terms of the Hellinger modulus. We prove a general lower bound for standard Gaussian covariates with symmetric, unimodal errors:

Theorem 1.8 (Hellinger modulus lower bound for regression). There exists a universal constant $0 < c \leq 1$ such that for every symmetric unimodal density $p ,$ every $n \geq d \geq 1$ , and every estimator $\widehat { \beta }$ given n samples where $X _ { i } \sim N ( 0 , I _ { d } ) , \varepsilon _ { i } \sim p , Y _ { i } = X _ { i } ^ { \top } \beta + \varepsilon _ { i }$ , it holds that

$$
\operatorname* { s u p } _ { \beta \in \mathbb { R } ^ { d } } \mathbb { P } \left( \left\| \beta - \widehat { \beta } \right\| _ { 2 } \geq c \cdot \omega _ { p } \left( c \frac { d } { n } \right) \right) \geq 3 / 8 .
$$

We aim to leverage a proof similar in spirit to Fano’s method (see e.g. [PW25, Section 31.4] for related exposition), where we draw the parameter $\beta$ from a prior and show: (i) if the estimate $\widehat { \beta }$ is often suficiently close to $\beta ,$ then $I ( \beta ; X ^ { n } , Y ^ { n } )$ must be large, and (ii) an upper bound for $I ( \beta ; X ^ { n } , Y ^ { n } )$ . Together, these imply a tradeof that yields a lower bound. In a typical lower bound, the proof of (ii) can be implied by a suitable upper bound for $\operatorname { E } _ { \beta , X _ { i } } [ \mathrm { K L } ( p _ { X _ { i } ^ { \top } \beta } , p ) ]$ ; this works well when $p \sim N ( 0 , 1 )$ , but can fail miserably when $p$ is an arbitrary symmetric unimodal distribution (e.g. when $p = \mathrm { U n i f } [ - 1 , 1 ]$ , the KL for translations are infinite). The main dificulty in our lower bound is sidestepping this technical obstacle.

A key observation is that whenever two distributions P and $Q$ have small Hellinger distance, then P can be decomposed into two components $P = ( 1 - \alpha ) G + \alpha R$ , where $\alpha \lesssim \mathrm { H } ^ { 2 } ( P , Q )$ and $\mathrm { K L } ( G , Q ) \lesssim \mathrm { H } ^ { 2 } ( P , Q )$ . The decomposition follows intuitively from leveraging how Hellinger and KL are closely related when the densities are within a constant factor of each other, and that the mass where densities are not within a constant factor is bounded in terms of the Hellinger distance. While there are more technical steps, our proof aims to handle samples corresponding to G in the more typical Fano style, and handle samples corresponding to R separately.

This gives a user-friendly lower bound for a more general class of errors than we are aware of in existing literature. For example, when the errors are uniform Unif[−1, 1], the Hellinger modulus is $\omega _ { p } ( t ) = 2 t \mathrm { f o r } t \in [ 0 , 1 )$ , and hence any estimator must incur $\Omega ( \textstyle { \frac { d } { n } } )$ error with constant probability. $\mathrm { A s }$ far as we know, the optimal rate’s dependence on d for uniform errors was previously unknown, and in prior work it is suggested that ${ \sqrt { d } } / n$ might be the optimal rate.<sup>1</sup> We expect there is a simpler lower bound which specifically targets the uniform setting, but our lower bound must handle all symmetric, unimodal distributions.

Adaptive estimators that nearly match the optimal non-adaptive estimator. Let us denote the minimum possible worst-case expected error (risk) for standard Gaussian design, known noise p, and m samples as

$$
\mathcal { R } ( \boldsymbol { p } , \boldsymbol { m } ) : = \operatorname* { i n f } _ { \mathrm { \tiny ~ e s t i m a t o r ~ } \widehat { \beta } } \operatorname* { s u p } _ { \beta \in \mathbb { R } ^ { d } } \operatorname* { l i } _ { \boldsymbol { x } _ { i } \sim N ( 0 , I _ { d } ) } \left[ \Big \| \widehat { \beta } ( X _ { 1 } , \ldots , X _ { m } , Y _ { 1 } , \ldots , Y _ { m } ) - \beta \Big \| _ { 2 } \right] .
$$

When combined with our upper bounds, this lets us compare the performance of our adaptive estimator to the best estimator that knows $p \mathrm { : }$

Corollary 1.9 (Adaptive linear regression for symmetric, log-concave errors). There exists a constant $C _ { K , \kappa } \geq 1$ such that when $n \geq C _ { K , \kappa } d \log ^ { 4 } ( 6 4 n / \delta )$ and the conditions of Theorem 1.6 hold, then with probability at least $1 - \delta$ , the adaptive $L _ { q }$ estimator incurs error

$$
\left. \widehat { \beta } - \beta \right. _ { 2 } \leq \mathcal { R } \left( p , \left\lfloor \frac { n } { C _ { K , \kappa } \log ^ { 4 } ( 6 4 n / \delta ) } \right\rfloor \right) + 3 \gamma .
$$

Corollary 1.10 (Adaptive linear regression for symmetric, log-concave mixture errors). There exists a constant $C _ { K , \kappa } \geq 1$ such that when $n \geq C _ { K , \kappa } d k \log ( 2 n / \delta )$ log(2kn) log(2n) and the conditions of Theorem 1.7 hold, then with probability at least $1 - \delta ,$ , the computationally intractable adaptive estimator incurs error

$$
\left\| \widehat { \beta } - \beta \right\| _ { 2 } \leq \mathcal { R } \left( p , \left\lfloor \frac { n } { C _ { K , \kappa } k \log ( 2 n / \delta ) \log ( 2 k n ) \log ( 2 n ) } \right\rfloor \right) .
$$

Both follow from the lower bound of Theorem 1.8 and after a small simplification using Claim D.1.

Planted linear regression. We suggest the planted linear regression problem as a simple, special case of both heteroskedastic and adaptive linear regression which may encapsulate the computational barriers to improving on our results. We diagram our understanding in Fig. 2a.

In Corollary 4.1, we show that our heteroskedastic linear regression algorithm recovers $\beta$ in the m ≫ $d ^ { 3 / 4 } n ^ { 1 / 4 }$ regime. In contrast, Theorem 4.7 demonstrates that convex M-estimators fail to recover $\beta$ even when $m \ll { \sqrt { d n } }$ . In Theorem 4.3, we prove a statistical query (SQ) lower bound for the $m \ll d ^ { 3 / 4 } n ^ { 1 / 4 }$ regime; this provides supporting evidence for an information-computation gap in the regime $d + 1 \ll m \ll d ^ { 3 / 4 } n ^ { 1 / 4 }$

## 1.3 Related work

Asymptotic adaptive estimation. In the setting of adaptive location estimation, the goal is to adaptively learn the location parameter of a distribution without knowing $p .$ A substantial body of work (see [S<sup>+</sup>56, VE70, Sto75, Sac75, Ber78]) captures the setting where $p$ is fixed and symmetric, the Fisher information is finite, and $n  \infty ;$ in this case, estimators may asymptotically attain the Fisher information rate. The work of Bickel [Bic82] includes a study into the analogous setting for adaptive linear regression. The work of Laha [Lah21] studies adaptive location estimation for symmetric, log-concave $p ,$ and proposes an estimator that requires no tuning parameter. Even when the Fisher information $\mathcal { T }$ is infinite, it is still possible to achieve $n ^ { 1 / 2 } ( \widehat { \theta } _ { n } - \theta )  N ( 0 , 1 / \mathcal { T } )$ as $n \to \infty$ for symmetric yet unknown $p ;$ see the work of Stone [Sto75] on adaptive location estimation and of Koul and Susarla [KS83] on adaptive linear regression. We note that this still requires $p$ to be fixed as $n  \infty$ , and when $\mathcal { T } = \infty$ this implies convergence faster than $n ^ { - 1 / 2 }$ , but not necessarily the optimal rate (e.g. $p = \mathrm { U n i f } [ - 1 , 1 ]$ permits $n ^ { - 1 }$ convergence rate). For location estimation where $p$ is known, the work of Le Cam [LC73] implies Hellinger modulus guarantees in the asymptotic setting under particular metric dimensionality assumptions. To our knowledge, we do not know of any result in the asymptotic setting that always yields the optimal convergence rate when $p$ may have infinite Fisher information, and $p$ is symmetric yet unknown (for context, [CV26, Theorem 1.5] implies that the analogous style of result is impossible in the finite-sample setting, even when assuming $p$ is symmetric and unimodal, but it does not rule out the asymptotic setting).

Feng, Kao, Xu, and Samworth [FKXS26] study adaptive linear regression with convex $M -$ estimators (this subsumes $L _ { q }$ losses). Their antitonic score matching (ASM) estimator chooses the convex loss in a data-dependent manner, where the loss is chosen to minimize the asymptotic variance among all convex losses. They focus on the setting where $p$ is unknown yet has finite Fisher information. The estimator’s performance in experiments is very strong, and we include their estimator as a benchmark in our simulations.

Finite-sample adaptive estimation. The most directly related work is that of Compton and Valiant [CV26], which studies location estimation. When $p$ is known, they attain nearly-optimal Hellinger modulus guarantees if $p$ is unimodal, yet show this is impossible when only assuming p is symmetric. In the adaptive setting where $p$ is unknown, they attain nearly-optimal Hellinger modulus guarantees if $p$ is a mixture of $k = O ( 1 )$ symmetric, log-concave distributions, yet show this is impossible when only assuming $p$ is symmetric and unimodal.

The work of Kao, Xu, and Zhang [KXZ24] primarily aims to better understand adaptive location estimation and regression when $p$ is symmetric, unknown, and may have infinite Fisher information. Focusing mostly on adaptive location estimation, they study an adaptive $L _ { q }$ estimator, where $q$ is chosen in a data-dependent manner; this yields nearly optimal rates for examples such as when $p$ is Gaussian or Uniform. Our work will leverage this style of estimator for adaptive linear regression with symmetric, log-concave errors. As noted by [KXZ24], this estimator is unable to leverage discontinuities in the interior of $p ;$ as an example, the mixture $\begin{array} { r } { p = \frac { 1 } { 2 } N ( 0 , 1 ) + \frac { 1 } { 2 } \operatorname { U n i f } [ - 1 , 1 ] } \end{array}$ has optimal rate on the order of $n ^ { - 1 }$ for location estimation, but an $L _ { q }$ estimator would only attain an $n ^ { - 1 / 2 }$ rate.

Gupta, Lee, Price, and Valiant study location estimation when $p$ is known [GLPV22], and when $p$ is symmetric yet unknown [GLP23]. They obtain guarantees in terms of the smoothed Fisher information of the $p ;$ when $p$ has variance $\sigma ^ { 2 }$ , this corresponds to the Fisher information of $p$ after it is convolved with $N ( 0 , \sigma ^ { 2 } r ^ { 2 } )$ , where $r$ is roughly $n ^ { - c }$ for a small constant $c > 0$ . For some distributions (e.g. Cauchy, Gaussian), this yields extremely strong guarantees with nearly-sharp constant factors. For other distributions (e.g. Uniform, heteroskedastic mixtures), this convolution destroys most of the important signal. The same authors prove that when $p$ is known, a variant of the MLE is optimal for location estimation [GLPV23]. A related line of work [Cat12, LV22b, LV22a, GHP24] studies estimators with sharp constant factor dependence for the sub-Gaussian error rate (where $p$ is unknown and has finite variance).

The method of ρ-estimation (see [BBS17, BB16, BB18]) is also a very relevant technique; at a minimum, this technique would yield certain finite-sample guarantees in terms of the Hellinger modulus when $p$ is known.

Heteroskedastic estimation. Chierichetti, Dasgupta, Kumar, and Lattanzi [CDKL14] initiated a study of heteroskedastic mean estimation: in one dimension, consider samples $X _ { i } \sim N ( \mu , \sigma _ { i } ^ { 2 } )$ where the goal is to estimate $\mu$ without knowing the variances. They studied the performance of an estimator that leveraged an empirical median confidence interval, and the k-shorth estimator (which finds the smallest interval containing at least k samples). This motivated an interesting body of work studying the performance of diferent estimators for heteroskedastic mean estimation. Pensia, Jog, and Loh [PJL22] designed hybrid median/modal/shorth estimators (in higher-dimensional settings as well). Devroye, Lattanzi, Lugosi, and Zhivotovskiy [DLLZ23] studied error guarantees for a diferent hybrid median/modal estimator. Xia [Xia19] and Louati [Lou25] analyzed the performance of the empirical median. Liang and Yuan [YL20] studied an iterative trimming estimator, and also introduced the Subset-of-Signals model, for which they proved lower bounds [LY20]. Compton and Valiant [CV24] designed a balance-finding estimator that matched these lower bounds, attaining near-optimal Subset-of-Signals guarantees (in near-linear time). Han, Shetty, and Shkrob [HSS26] also attained near-optimal Subset-of-Signals guarantees, and further proved strong Hellinger modulus guarantees, with an estimator that leverages empirical Bayes methods.

There has also been interest in higher-dimensional heteroskedastic mean estimation. The work of [CDKL14] studies the task where $X _ { i } \sim N ( \mu , \sigma _ { i } ^ { 2 } I _ { d } )$ , inspired by a setting where users of unknown, varying skills rate $d$ items each (higher-skill users have smaller $\sigma _ { i } )$ . In general, heteroskedastic mean estimation is much easier when the $\sigma _ { i }$ for each sample is known by the estimator; the optimal rate is much better, as an estimator may use a weighted mean with $X _ { i }$ weighted proportionally to $1 / \sigma _ { i } ^ { 2 } \ \mathrm { [ I H 1 3 ] }$ . In this high-dimensional setting, [CDKL14] show that the known-variance rate is nearly attainable even without knowing the variances, so long as $d = \Omega ( \log n )$ . Later, the work of [CV24] demonstrated this is possible for any $d \ge 2$ . Li [Li24] proposed an alternative highdimensional mean estimation task where $X _ { i } \sim N ( \mu , \Sigma _ { i } )$ for $\mathrm { P S D } \ \Sigma _ { i } .$ , and $m$ of the samples satisfy $\Sigma _ { i } \preceq I _ { d }$ . Diakonikolas, Kane, Liu, and Pittas [DKLP25] designed a near-optimal estimator in this high-dimensional Subset-of-Signals setting; their estimator employs an iterative dimension-reduction strategy, and uses the 1-dimensional estimator of [CV24] as a subroutine.

The works of [LY20, PJL22, CB26] have also studied the task of heteroskedastic linear regression, and the survey of $[ \mathrm { M S B ^ { + } 2 6 }$ , Section 3.4.3] highlighted the open problem of whether optimal guarantees are achievable in polynomial time. Even information-theoretically, the optimal guarantees were not yet established; for example, all previous Subset-of-Signals guarantees attaining o(1) error required $m \geq \Omega ( { \sqrt { n d } } )$ (this is achieved by the Huber loss estimator via the analysis of [dNS21]).

We note that heteroscedastic/heteroskedastic linear regression may also refer to a task where the error scales with an unknown direction (scaling with $X _ { i } ^ { \top } f$ for an $f \in \mathbb { R } ^ { d } )$ ; this is a very diferent task from what we study, and we refer to [DNNB23] for discussion of this problem.

Oblivious contamination and noiseless linear regression. In the setting of linear regression with oblivious contamination, an adversary adds noise $\varepsilon _ { 1 } , \ldots , \varepsilon _ { n }$ , but the adversary does not observe $X _ { 1 } , \ldots , X _ { n }$ (and hence the noise is independent, or “oblivious,” to the covariates). Framed in terms of our notation, suppose $m$ of the noise entries satisfy $| \varepsilon _ { i } | \leq 1$ . A main question in oblivious contamination is understanding what values of m permit $o ( 1 )$ coeficient error. A line of research including [TJSO14, BJKK17, SBRJ19], and culminating in the work of d’Orsi, Novikov, and Steurer [dNS21], proved that $o ( 1 )$ error is possible once $m \gg \sqrt { n d }$ (with only mild assumptions on the design matrix); as noted in [dNS21], a near-matching statistical lower bound follows immediately from the setting where all $\varepsilon _ { i } \sim N ( 0 , n / d )$ . Note that the Subset-of-Signals model for heteroskedastic regression is (roughly) a special case of oblivious contamination: where m samples have oblivious error $N ( 0 , 1 )$ , and $n { - } m$ samples have oblivious error $N ( 0 , \sigma _ { i } ^ { 2 } )$ for any $\sigma _ { i }$ . While $m \gg \sqrt { n d }$ is optimal for oblivious contamination, this contrasts with the lower optimal threshold of $m \gg d ^ { 3 / 4 } n ^ { 1 / 4 }$ for heteroskedastic regression (ignoring logarithmic factors).

The main predecessor of planted linear regression is the task of noiseless linear regression. In this task, the covariates are also $N ( 0 , I _ { d } )$ , and m samples are noiseless, yet the remaining $n - m$ samples have error drawn from any fixed distribution E (a related i.i.d. version has all noise drawn from a distribution $E _ { \mathrm { { i } } }$ , where $\begin{array} { r } { \mathbb { P } _ { Z \sim E } ( Z = 0 ) \geq \frac { m } { n } ) } \end{array}$ . Diakonikolas, Gao, Kane, Laferty, and Pensia $[ \mathrm { D G K ^ { + } 2 5 } ]$ study a possible information-computation gap for this problem. When $m \geq d + 1$ , this problem is information-theoretically trivial for the same reasons as planted linear regression. When $m \gg \sqrt { n d }$ (ignoring logarithmic factors), the work of [GL20, Theorem 3.2] demonstrates that LAD attains exact recovery of $\beta$ . The work of [DGK<sup>+</sup>25] conjectures that any polynomial-time algorithm requires $m = \Omega ( { \sqrt { n d } } )$ to exactly recover $\beta ,$ and they provide Statistical Query (SQ) lower bounds to support their claim (the SQ lower bounds are not yet tight to their conjecture, this is left as an open problem). Note that noiseless linear regression is a special case of oblivious contamination, and further, planted linear regression is a special case of noiseless linear regression. While $m \gg \sqrt { n d }$ is the conjectured computational threshold for noiseless linear regression (and is attained by LAD, a convex M-estimator), our task of planted linear regression permits polynomial-time recovery when $m \gg d ^ { 3 / 4 } n ^ { 1 / 4 }$ (ignoring logarithmic factors), even though we show that convex M-estimators cannot recover $\beta$ when $m \ll { \sqrt { d n } }$

We refer to $[ \mathrm { D G K ^ { + } 2 6 } ]$ for discussion of works where the labels may be adversarially corrupted.

Information-computation gaps. When an estimation problem is statistically tractable, but no time-eficient algorithm is known, the problem is said to exhibit an “information-computation gap.” Any such gap could be a product of researchers’ ignorance and/or laziness. Hence, when confronted with an information-computation gap, it is desirable to prove a lower bound that suggests that no state-of-the-art algorithmic technique could overcome the gap. There are two common approaches: the first is to give a reduction from a statistical problem that is widely believed to be hard, for example planted clique (e.g. [BPP18, BB20, BDT24]). The second approach is to prove a lower bound against a restricted class of algorithms or computational model; popular frameworks include statistical query (SQ) algorithms (e.g. [FGR<sup>+</sup>17, DGK<sup>+</sup>25]), low-degree algorithms (e.g. [Hop18, Wei25]), and generalized first-order methods (e.g. [CMW20]). Some problems are more natural to consider in one particular framework (though the frameworks are directly comparable for many canonical problems [BBH<sup>+</sup>21, MW25]). Problems in which the signal-to-noise ratio is measured using the number of samples, and in which no one sample carries too much information, are often most natural to consider in the SQ framework. In this work, we encounter an informationcomputation gap for planted linear regression, and substantiate it with an SQ lower bound: we show that when our algorithm fails, it is also the case that no query-eficient SQ algorithm can achieve error which is significantly better than the OLS estimator. We note that because it is essential that the planted linear regression problem is noiseless on a subset of labels, the low-degree framework does not make sense for this problem (and indeed, it may be checked that the low-degree method obtains a pessimistic threshold).

Reverse data processing inequalities. The results of Bhatt, Nazer, Ordentlich, and Polyanskiy [BNOP21, Theorem 1] and Pensia, Jog, and Loh [PJL23, Corollary 3.4] both imply a “reverse data processing inequality” for Hellinger distance: for any $P , Q$ , there exists a $\tau > 0$ where the function $f _ { \tau } ( x ) = \mathbf { 1 } \{ P ( x ) / Q ( x ) \geq \tau \}$ is such that $\mathrm { H } ^ { 2 } ( f _ { \tau } ( P ) , f _ { \tau } ( Q ) )$ is at worst a logarithmic factor smaller than $\mathrm { H } ^ { 2 } ( P , Q )$ . This result is naturally useful for communication-constrained settings, but it has recently been helpful in settings with no communication constraints as well (see [CV26, BCW25, CL26]). We will often use the form given by [PJL23, Corollary 3.4] in this work. The work of [KPJ25] more generally proves reverse data processing inequalities for “TV-like f-divergences.”

## 1.4 Discussion

In this work, we have studied the computational-statistical landscape of heteroskedastic, adaptive, and planted linear regression. For heteroskedastic linear regression, we design a polynomial-time algorithm with near-optimal Subset-of-Signals guarantees when m $\gg d ^ { 3 / 4 } n ^ { 1 / \breve { 4 } }$ (Theorem 1.5). For adaptive linear regression, we design estimators for mixtures of k symmetric, log-concave $p ;$ our estimators perform nearly as well as the optimal estimator that knows $p$ and has $\tilde { \Theta } ( n / k )$ samples. Our estimator for $k = 1$ uses adaptive $L _ { q }$ regression and runs in polynomial time (Theorem 1.6); for $k \geq 2$ , we design an ineficient estimator (Theorem 1.7). Finally, we introduce the task of planted linear regression, for which we demonstrate a polynomial-time estimator and a near-matching SQ lower bound at $m \sim d ^ { 3 / 4 } n ^ { 1 / 4 }$ ; any computationally tractable estimator improving on our heteroskedastic or adaptive results would imply an estimator for planted linear regression that succeeds below this threshold.

More broadly, our work highlights the helpfulness of the finite-sample perspective for these problems, and how this lens inspires new estimators with guarantees that might not be possible with classical techniques.

We make some suggestions for follow-up work. First, a few technical (and perhaps minor) points for improvement: For heteroskedastic linear regression, it would be desirable to remove the dependence on $B ,$ the radius of the ball containing $\beta$ in Theorem 1.5. We also omit a Subset-of-Signals information-theoretic upper bound for the $m \ll d ^ { 3 / 4 } n ^ { 1 / 4 }$ regime; we expect an analysis in the style of Theorem 1.7 would yield the correct result, or possibly an analysis in the style of the metric entropy techniques in [HSS26]. For adaptive linear regression (Theorem 1.8), our Hellinger modulus lower bound is stated in terms of standard Gaussian covariates, instead of arbitrary well-conditioned and sub-Gaussian covariates. For planted linear regression, our SQ lower bound (Theorem 4.3) adds arbitrarily small noise (e.g. superexponentially small in n) to the m noiseless samples for technical convenience.

We next outline three broader potential directions for future study:

Practical implementations. We believe our simulations in Section 5 are generally positive, yet there is still room for improvement. First, our RB-Desc estimator needed important modifications for strong empirical performance (we discuss this in Section 5); it would be nice if there is a simpler estimator with the same theoretical guarantees that attains strong empirical performance more naturally. Second, the estimators have favorable empirical performance, yet no single estimator attains the best performance simultaneously for all five test distributions in Fig. 4. Designing a natural estimator that simultaneously achieves the best performance for all these settings seems interesting. On a related note, we were unable to produce a reasonable implementation for our ineficient estimator in Theorem 1.7, even for very small $d ;$ a core obstacle is that the estimator requires a search over an extremely fine net. A practical version of this estimator for small d (yet still moderately large n) would be quite interesting, as the analogous estimator for mean estimation simultaneously nearly achieves the best empirical performance for all examples in [CV26].

Alternative estimators. We are optimistic there may be alternative estimators that also achieve the theoretical guarantees we study. For example, in the case of 1-dimensional heteroskedastic mean estimation, the known estimators that achieve error $\ll 1$ when $m \gg n ^ { 1 / 4 }$ are the balance-finding algorithm of [CV24] and the empirical Bayes estimator of [HSS26]. We remark that a sharper analysis of a hybrid median/modal estimator (in the style of [DLLZ23]<sup>2</sup>) can actually recover this guarantee as well; this may be helpful knowledge when designing future estimators.

Wider applicability of the (computationally ineficient) adaptive linear regression estimator. Our result in Theorem 1.7 focuses on the setting where $p$ is a mixture of k log-concave distributions, each symmetric around 0. This estimator applies to more general $p ,$ and better understanding which classes of distributions could be interesting. This would entail proving that the Hellinger distance between translations is witnessed by intervals for a larger class of distributions (generalizing Corollary 2.18 of [CV26]). As an example, the Cauchy distribution is not contained within our Theorem 1.7, but it is simple to prove that its translations have Hellinger distance witnessed by intervals; in this case, $p ( x ) \geq \tau p _ { t } ( x )$ in precisely the region where a degree-2 polynomial in x is non-negative, and hence the region is at most two intervals. The desired adaptivity guarantees should follow for any symmetric density that can be ε-approximated in Hellinger distance by a piecewise polynomial with $\mathrm { p o l y l o g } ( n / \varepsilon )$ pieces and degree (related ideas are studied for Hellinger density estimation in [CL26], which builds upon [CDSS13, CDSS14, ADH<sup>+</sup>15, ADLS17]). See also the discussion in [CV26] on “adaptive location estimation for more general distributions.”

## 2 Heteroskedastic linear regression

In this section, we discuss polynomial-time algorithms for heteroskedastic linear regression. We begin with a discussion of a simpler setting with standard Gaussian covariates in Section 2.1, and then introduce our algorithm for the more general setting of well-conditioned, sub-Gaussian covariates in Section 2.2.

## 2.1 A simple estimator for standard Gaussian covariates

As a warm-up, we first discuss how the task of heteroskedastic regression is simpler when the covariates are drawn according to $X _ { i } ~ \sim ~ N ( 0 , I _ { d } )$ (and the errors are $\varepsilon _ { i } \sim N ( 0 , \sigma _ { i } ^ { 2 } ) )$ . In this restricted setting, we observe that the reduction from linear regression to mean estimation used in the work of d’Orsi, Novikov, and Steurer [dNS21, Section 4.2.1] is amenable to using recent algorithms for heteroskedastic mean estimation. We will give brief intuition on how this works, but omit a rigorous analysis since this will be subsumed by our Theorem 1.5 for more general covariates; this is a standalone section and other parts of the paper do not use these techniques.

Consider a candidate $b \in \mathbb { R } ^ { d }$ , and the diference $h \ : = \ : \beta \ : - \ : b$ . Let us focus on the task of estimating $h _ { 1 }$ . In [dNS21], they observe how the following manipulation helps reduce the task to mean estimation:

$$
{ \frac { r _ { i } ( b ) } { X _ { i } ( 1 ) } } = h _ { 1 } + { \frac { \varepsilon _ { i } + \sum _ { j = 2 } ^ { d } h _ { j } \cdot X _ { i } ( j ) } { X _ { i } ( 1 ) } }\tag{8}
$$

Moreover, conditioned on the value of $X _ { i } ( 1 )$ , the distribution is exactly the Gaussian

$$
{ \frac { r _ { i } ( b ) } { X _ { i } ( 1 ) } } \triangleq N \left( h _ { 1 } , { \frac { \sigma _ { i } ^ { 2 } + \| h _ { - 1 } \| _ { 2 } ^ { 2 } } { X _ { i } ( 1 ) ^ { 2 } } } \right) ,\tag{9}
$$

where $h _ { - 1 }$ denotes the vector without the first coordinate. Thus, if we condition on the values of $X _ { i } ( 1 )$ , and only use the samples where $| X _ { i } ( 1 ) | \geq 1$ (this is a constant fraction), then we immediately have a mean estimation problem with new variances bounded by $\sigma _ { i } ^ { 2 } + \| h _ { - 1 } \| _ { 2 } ^ { 2 }$

We now roughly outline how to get a Subset-of-Signals guarantee with this approach when $m \gg d ^ { 3 / 4 } n ^ { 1 / 4 }$ . Consider two regimes: (i) $\| h _ { - 1 } \| _ { 2 } ^ { 2 } > 1$ and (ii) $\| h _ { - 1 } \| _ { 2 } ^ { 2 } \leq 1$

For regime (i), using existing Subset-of-Signals heteroskedastic mean estimation guarantees (such as in [CV24]) implies that we learn $h _ { 1 }$ with error $\ll \| h _ { - 1 } \| _ { 2 } / \sqrt { d }$ . If we simultaneously estimate every coordinate of $h$ in this manner, and adjust our estimate of $\beta$ accordingly by adding the estimated means, then the norm of the new $h$ will be at most, say, half of what it was originally. This is the same paradigm used in [dNS21], where all coordinates are estimated and improved iteratively. We continue doing so until we are in regime (ii).

In regime (ii), we now have ≈ m samples with variance bounded by a constant, and can run one last round of mean estimation with the existing Subset-of-Signals guarantees again; this will yield error $\approx ( n / m ^ { 4 } ) ^ { 1 / 6 }$ for each coordinate, implying total error

$$
\approx { \sqrt { d } } \left( { \frac { n } { m ^ { 4 } } } \right) ^ { 1 / 6 } ,
$$

which is the desired guarantee. Hence, the problem follows rather simply from mean estimation in the setting where covariates are sampled from a standard Gaussian. However, this approach does not seem conducive to handling more general covariates, and thus in Section 2.2 we design a new algorithm for heteroskedastic regression.

## 2.2 An estimator for well-conditioned, sub-Gaussian covariates

As is outlined in Section 1.2, the main intuition for this estimator is that we aim to show an estimate $b \in \mathbb { R } ^ { d }$ is either close to $\beta ,$ or it exhibits a notion of imbalance in the direction of $\beta - b$ that we may use to iteratively improve the estimate. We start by introducing the relevant notation. Let

$$
L = c _ { \log } \log \left( \frac { e n d B } { \delta } \right) .\tag{10}
$$

You should think of $c _ { \mathrm { l o g } }$ as the last constant chosen in the proof (throughout the proof we will often say “for suficiently large choice of constant in $L ^ { \dag }$ to denote that $c _ { \mathrm { l o g } }$ must be chosen suficiently large); it only depends on $K , \kappa$ (meaning it is independent of $n , d , m , B , \delta , { \mathrm { e t c } } )$

For $1 \leq s \leq n$ , define

$$
\rho ( s ) = L ^ { 5 } \left[ \sqrt { d } \left( \frac { n } { s ^ { 4 } } \right) ^ { 1 / 6 } + \frac { d } { s } \right] .\tag{11}
$$

The function $s \mapsto \rho ( s )$ is nonincreasing, and the first summand is larger in the regime of m we study (4). In this notation, the desired error bound (5) for Theorem 1.5 may be replaced with $C _ { K , \kappa } \rho ( m )$

For a window $w > 0$ , we define

$$
\psi _ { w } ( t ) = \mathrm { s g n } ( t ) \mathbf { 1 } \{ | t | \leq w \} , \qquad \mathrm { s g n } ( 0 ) = 0 .\tag{12}
$$

The residual-balance vector is

$$
S _ { w } ( b ) = \sum _ { i = 1 } ^ { n } X _ { i } \psi _ { w } ( r _ { i } ( b ) ) ,\tag{13}
$$

which we will show is correlated with the direction $\beta - b$ when they are suficiently far apart. The local residual count is

$$
A _ { w } ( b ) = \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ | r _ { i } ( b ) | \leq w \} ,\tag{14}
$$

which will serve as a helpful benchmark for the level of noise at scale w. Let $w _ { 0 } = w _ { 0 } ( K , \kappa )$ be a constant defined later in Claim 2.4. Define the dyadic grid (with ∞)

$$
{ \mathcal { W } } = \{ w _ { 0 } , 2 w _ { 0 } , 4 w _ { 0 } , \ldots , 2 ^ { J } w _ { 0 } \} \cup \{ \infty \} ,\tag{15}
$$

where J is the smallest nonnegative integer satisfying $2 ^ { J } \ge \operatorname* { m a x } \{ 1 , 2 \kappa ^ { 1 / 2 } B \}$ ; this will capture all the potentially relevant scales. The grid construction satisfies

$$
| \mathcal { W } | \leq L ,\tag{16}
$$

since $| \mathcal { W } | \lesssim 1 + \log ( B )$ is less than L when it is defined with a suficiently large constant. For w $\in \mathcal { W }$ , we use $A _ { w } ( b )$ to define a threshold for the uncertainty in the residual-balance vector

$$
\tau _ { w } ( b ) = L ^ { 2 } \left( \sqrt { d ( A _ { w } ( b ) + d ) } + d \right) .\tag{17}
$$

A scale w is called violated at b if

$$
\| S _ { w } ( b ) \| _ { 2 } > | \mathcal { W } | \cdot \tau _ { w } ( b ) .\tag{18}
$$

Consider the collection of violated scales

$$
\mathcal { V } ( b ) = \{ w \in \mathcal { W } : w \mathrm { ~ i s ~ v i o l a t e d ~ a t ~ } b \} .
$$

The aggregate residual-balance direction is then

$$
G _ { \mathrm { a g g } } ( b ) = \sum _ { w \in \mathcal { V } ( b ) } \frac { S _ { w } ( b ) } { \tau _ { w } ( b ) } ,\tag{19}
$$

with the convention $G _ { \mathrm { a g g } } ( b ) = 0 { \mathrm { ~ i f ~ } } \mathcal { V } ( b ) = \emptyset$

With these definitions in hand, the estimator is simple: we repeatedly normalize $G _ { \mathrm { a g g } } ( b )$ and move in that direction. We next prove guarantees about $G _ { \mathrm { a g g } } ( b )$ , and later in Section 2.2.2 we provide detailed pseudocode and conclude performance of the algorithm.

## 2.2.1 Residual-balance signal

In this section, we prove results about how well $G _ { \mathrm { a g g } } ( b )$ aligns with $\beta - b$ . We will begin by showing in-expectation bound. We define population analogs of $S _ { w }$ and $A _ { w }$ with

$$
\overline { { S } } _ { w } ( b ) = \mathrm { E } S _ { w } ( b ) , \qquad Q _ { w } ( b ) = \sum _ { i = 1 } ^ { n } \mathbb { P } \left( \left| y _ { i } - X _ { i } ^ { \top } b \right| \leq w \right) .
$$

Fixing $b \in B _ { 2 } ( B )$ , we will also use the notation

$$
h = \beta - b , \quad \quad r = \| h \| _ { \Sigma } , \quad \quad \eta ( r ) = \operatorname* { m i n } \{ r , 1 \} , \quad \quad s ( r ) = \operatorname* { m a x } \{ r , 1 \} .
$$

We will choose a relevant scale $w = w ( r )$ in terms of r and a later-specified constant $w _ { 0 } ;$ if $r \leq 1$ we choose $w = w _ { 0 }$ , if $r > 1$ we choose a dyadic $w \in \mathcal W$ such that $w _ { 0 } r \le w \le 2 w _ { 0 } r$ (such a $w \in \mathcal W$ always exists by $r \leq 2 \kappa ^ { 1 / 2 } B$ and (15)). We now prove the in-expectation signal guarantee:

Lemma 2.1 (Population signal). For the scale $w = w ( r )$ chosen above,

$$
\left. h , \overline { { S } } _ { w } ( b ) \right. \geq c m r \eta ( r ) ,\tag{20}
$$

$$
\left. h , \overline { { S } } _ { \infty } ( b ) \right. \geq c r \eta ( r ) Q _ { w } ( b ) ,\tag{21}
$$

and for every tested scale $u \in \mathcal W$

$$
\left. h , { \overline { { S } } } _ { u } ( b ) \right. \geq 0 .\tag{22}
$$

Proof. If $r = 0$ , then $h = 0$ and all three inequalities are immediate. Assume $r > 0$ . We define the random variable

$$
Z _ { i } = \frac { X _ { i } ^ { \top } h } { r } ,
$$

which conveniently lets us rewrite

$$
\langle h , { \overline { { S } } } _ { w } ( b ) \rangle = \sum _ { i = 1 } ^ { n } \mathbf { E } [ X _ { i } ^ { \top } h \cdot \psi _ { w } ( X _ { i } ^ { \top } h + \varepsilon _ { i } ) ] = \sum _ { i = 1 } ^ { n } \mathbf { E } [ r Z _ { i } \cdot \psi _ { w } ( r Z _ { i } + \varepsilon _ { i } ) ] .\tag{23}
$$

Proving this lemma will entail providing the right guarantees for the RHS of (23), where we may use that $\mathrm { E } Z _ { i } ^ { 2 } = 1$ and (2) gives $\| Z _ { i } \| _ { \psi _ { 2 } } \leq K$

First, we work towards proving (22) by showing each summand $\mathrm { E } [ X _ { i } ^ { \top } h \cdot \psi _ { w } ( X _ { i } ^ { \top } h + \varepsilon _ { i } ) ] \geq 0$ Recall the decomposition for symmetric, unimodal variables given in Definition 1.2. In this spirit, for a fixed $T \geq 0$ , let $U \sim \mathrm { U n i f } [ - 1 , 1 ]$ , and define

$$
\begin{array} { r } { q _ { T , w } ( a ) = \mathrm { E } _ { U } [ \mathrm { s g n } ( a + T U ) \mathbf { 1 } \{ | a + T U | \leq w \} ] , \qquad s _ { T } ( a ) = \mathrm { E } _ { U } [ \mathrm { s g n } ( a + T U ) ] . } \end{array}
$$

For $T > 0$

$$
s _ { T } ( a ) = \mathrm { c l i p } ( a / T , - 1 , 1 ) ,\tag{24}
$$

and for $T = 0 , s _ { 0 } ( a ) = \operatorname { s g n } ( a )$ . Then:

Claim 2.2 (Non-negative contribution). For every $a \in \mathbb { R } , T \geq 0$ , and $w > 0$

$$
a \cdot q _ { T , w } ( a ) \geq 0 , \qquad a \cdot s _ { T } ( a ) \geq 0 .
$$

Consequently, $i f \varepsilon$ is symmetric unimodal about zero, then

$$
a \mathrm { E } [ \psi _ { w } ( a + \varepsilon ) ] \geq 0 .
$$

Proof. The claim $a \cdot s _ { T } ( a ) \geq 0$ follows from (24). For the other claim, since $q _ { T , w }$ is odd it sufices to consider $a > 0$ . The $T = 0$ case is immediate. For $T > 0$ , the density of $a + T U$ is

$$
f _ { a } ( t ) = { \frac { 1 } { 2 T } } { \bf 1 } \{ | t - a | \leq T \} .
$$

Thus

$$
q _ { T , w } ( a ) = \int _ { 0 } ^ { w } \{ f _ { a } ( t ) - f _ { a } ( - t ) \} d t .
$$

For every $t \geq 0 , | t - a | \leq | t + a |$ . Hence $f _ { a } ( - t ) > 0$ implies $f _ { a } ( t ) > 0$ , and the integrand is nonnegative. Therefore $q _ { T , w } ( a ) \geq 0$ □

(22) follows immediately by conditioning on $X _ { i }$ and applying Claim 2.2 to $a = X _ { i } ^ { \top } h$ . For the remaining claims, we use that $| Z _ { i } |$ is often in a constant-size window bounded away from 0:

Claim 2.3. Let Z be a real random variable satisfying $\mathrm { E } Z ^ { 2 } = 1$ and $\| Z \| _ { \psi _ { 2 } } \le K$ . There exist $a , p \in ( 0 , 1 )$ , depending only on $K ,$ such that

$$
\mathbb { P } ( a \leq | Z | \leq a ^ { - 1 } ) \geq p .
$$

Proof. The sub-Gaussian bound gives $\mathrm { E } Z ^ { 4 } \leq C$ . Paley-Zygmund applied to $Z ^ { 2 }$ gives

$$
\mathbb { P } \left( Z ^ { 2 } \geq { \frac { 1 } { 2 } } \mathrm { E } Z ^ { 2 } \right) \geq { \frac { ( 1 / 2 ) ^ { 2 } ( \mathrm { E } Z ^ { 2 } ) ^ { 2 } } { \mathrm { E } Z ^ { 4 } } } \geq c .
$$

Choose $R \geq 1$ so that $\mathbb { P } ( | Z | > R ) \le c / 2$ , and set $a = \operatorname* { m i n } \{ 1 / \sqrt { 2 } , R ^ { - 1 } \}$ and $p = c / 2$

We are now ready to state the main quantitative bounds in terms of $Z _ { i } , \varepsilon _ { i }$

Claim 2.4. Let Z satisfy $\mathrm { E } Z ^ { 2 } = 1$ and $\| Z \| _ { \psi _ { 2 } } \leq K _ { \mathrm { \ell } }$ ; Z may have nonzero mean. Let ε be independent $o f Z$ and symmetric unimodal about zero. There are constants $c , w _ { 0 } > 0$ , depending only on K, such that the following hold. For $r > 0$ , recall

$$
\eta ( r ) = \operatorname* { m i n } \{ r , 1 \} , \qquad s ( r ) = \operatorname* { m a x } \{ r , 1 \} .
$$

Also recall that $w = w ( r ) = w _ { 0 } \ i f \ 0 < r \leq 1$ , and $w _ { 0 } r \le w \le 2 w _ { 0 } r \ i f r > 1$ . Then:

1. $I f \operatorname { V a r } ( \varepsilon ) \leq 1$

$$
\begin{array} { r } { \operatorname { E } [ Z \psi _ { w } ( r Z + \varepsilon ) ] \ge c \eta ( r ) . } \end{array}\tag{25}
$$

2. For arbitrary symmetric unimodal $\varepsilon _ { \mathrm { { ; } } }$

$$
\begin{array} { r } { \mathrm { E } [ Z \mathrm { s g n } ( r Z + \varepsilon ) ] \ge c \eta ( r ) \mathbb { P } ( | r Z + \varepsilon | \le w ) . } \end{array}\tag{26}
$$

Proof. Use the decomposition $\varepsilon = T U$ as in Definition 1.2. Let $a , p$ be supplied by Claim 2.3. If $\mathrm { V a r } ( \varepsilon ) \leq 1$ , then $\mathrm { E } T ^ { 2 } \leq 3$ . Choose

$$
M = \operatorname* { m a x } \{ a ^ { - 1 } + 1 , ( 6 / p ) ^ { 1 / 2 } \} , \qquad w _ { 0 } = a ^ { - 1 } + M .
$$

This was chosen such that $\mathbb { P } ( T \le M ) \ge 1 - p / 2$ . Since $Z$ and T are independent, the event

$$
\mathcal { E } = \{ a \leq | Z | \leq a ^ { - 1 } \mathrm { ~ a n d ~ } T \leq M \}
$$

satisfies $\mathbb { P } ( \mathcal { E } ) \ge p / 2$

Proving (25). We prove this in two cases. First, suppose $0 < r \leq 1$ . On $\mathcal { E } ,$ the support of $r Z + T U$ is contained in $[ - w _ { 0 } , w _ { 0 } ]$ , so the windowed sign equals the full sign. Using (24),

$$
Z \mathrm { E } _ { U } [ \mathrm { s g n } ( r Z + T U ) ] = | Z | \operatorname * { m i n } \left\{ \frac { r | Z | } { T } , 1 \right\} .
$$

On $\mathcal { E } ,$ this quantity is at least $r a ^ { 2 } / M$ . By Claim 2.2, the contribution outside $\mathcal { E }$ is nonnegative. Hence $\begin{array} { r } { \mathrm { E } [ Z \psi _ { w _ { 0 } } ( r Z + \varepsilon ) ] \ge c r = c \eta ( r ) } \end{array}$

Otherwise, $r > 1$ . We use $w \ge w _ { 0 } r \implies w \ge r ( a ^ { - 1 } + M ) \ge r a ^ { - 1 } + M$ . Thus, on $\mathcal { E }$ the window again contains the full support of $r Z + T U$ , and

$$
| Z | \operatorname* { m i n } \left\{ \frac { r \left| Z \right| } { T } , 1 \right\} \geq a \operatorname* { m i n } \left\{ \frac { r a } { M } , 1 \right\} \geq c = c \eta ( r ) .
$$

Again, contribution outside $\mathcal { E }$ is nonnegative, concluding (25).

Proving (26). Condition on $T = t$ . Using (24), let

$$
D _ { t } : = \mathrm { E } _ { Z } [ Z s _ { t } ( r Z ) ] = \mathrm { E } _ { Z } \left[ | Z | \operatorname* { m i n } \left\{ \frac { r | Z | } { t } , 1 \right\} \right] .
$$

Claim 2.3 gives

$$
D _ { t } \geq c \operatorname* { m i n } \left\{ 1 , { \frac { r } { t } } \right\} .\tag{27}
$$

When $T = 0$ , the following bounds will hold trivially. Otherwise, conditioning on $Z$ and $T = t > 0$ the random variable $t U$ has density bounded by $( 2 t ) ^ { - 1 }$ on an interval of length 2t, so

$$
\mathbb { P } _ { U } ( | r Z + t U | \le w \mid Z , T = t ) \le \operatorname* { m i n } \left\{ 1 , \frac { w } { t } \right\} .\tag{28}
$$

Since $w \asymp s ( r )$ and $\eta ( r ) s ( r ) = r$ , we may control the quantity relating to the RHS of (26),

$$
\eta ( r ) \mathbb { P } ( \left| r Z + t U \right| \leq w \mid T = t ) \leq C \operatorname* { m i n } \left\{ 1 , { \frac { r } { t } } \right\} .
$$

Combining this with (27) and integrating over $T$ proves (26).

We are finally prepared to prove (20) and (21). For i where $\mathrm { V a r } ( \varepsilon _ { i } ) \leq 1$ , applying Claim 2.4 to $Z _ { i } = X _ { i } ^ { \top } h / r$ gives

$$
\begin{array} { r } { \operatorname { E } [ Z _ { i } \psi _ { w } ( r Z _ { i } + \varepsilon _ { i } ) ] \ge c \eta ( r ) . } \end{array}
$$

Multiplying by r gives the form of (23), and summing over the good indices yields (20); other indices have nonnegative contribution by Claim 2.2. Claim 2.4 also gives, for every i,

$$
\begin{array} { r } { \mathrm { E } [ Z _ { i } \mathrm { s g n } ( r Z _ { i } + \varepsilon _ { i } ) ] \ge c \eta ( r ) \mathbb { P } ( | r Z _ { i } + \varepsilon _ { i } | \le w ) . } \end{array}
$$

Similarly, multiplying by r and summing over i gives (21).

Our algorithm will need to empirically witness signals demonstrated by the in-expectation bounds of Lemma 2.1. The following uniform convergence event will be crucial (the proof follows from existing techniques and is deferred to Section C.1):

Proposition 2.5 (Uniform convergence). With probability at least $1 - \delta$ , all of the following statements hold simultaneously.

(i) Residual-balance vector accuracy. For every $b \in B _ { 2 } ( B )$ and every $w \in \mathcal W$

$$
\begin{array} { r } { \left\| S _ { w } ( b ) - \overline { { S } } _ { w } ( b ) \right\| _ { 2 } \leq \tau _ { w } ( b ) . } \end{array}\tag{29}
$$

(ii) Count comparability. For every $b \in B _ { 2 } ( B )$ and every $w \in \mathcal W$

$$
A _ { w } ( b ) \leq 2 Q _ { w } ( b ) + C L d , \qquad Q _ { w } ( b ) \leq 2 A _ { w } ( b ) + C L d .\tag{30}
$$

(iii) Feature norms.

$$
\operatorname* { m a x } _ { 1 \leq i \leq n } \| X _ { i } \| _ { 2 } \leq { \sqrt { L d } } .\tag{31}
$$

Note how the concentration of $S _ { w } ( b )$ degrades as $\tau _ { w } ( b )$ increases (or equivalently, as $A _ { w } ( b )$ increases). This illuminates a tradeof at the heart of the algorithm’s analysis: (i) if $A _ { w } ( b )$ is small, then the signal at scale $S _ { w } ( b )$ is visible; otherwise, if $A _ { w } ( b )$ is large, then Lemma 2.1 implies there will be strong signal at scale $S _ { \infty } ( b )$ . We now make this intuition rigorous:

Lemma 2.6 (A far estimate has a good residual-balance vector). Assume the lower bound condition on m (4) and let $b \in B _ { 2 } ( B ) . \ I f \ \| b - \beta \| _ { \Sigma } > \rho ( m )$ , then some $w _ { \star } \in \mathcal { W } \cup \{ \infty \}$ satisfies

$$
\begin{array} { r } { \langle S _ { w _ { \star } } ( b ) , \beta - b \rangle \geq 2 \kappa ^ { 1 / 2 } | \mathcal { W } | \| b - \beta \| _ { \Sigma } \tau _ { w _ { \star } } ( b ) , } \end{array}\tag{32}
$$

under the uniform convergence event. In particular, $w _ { \star }$ is violated at b.

Proof. Let $w = w ( r )$ as before. Lemma 2.1 and Proposition 2.5 imply

$$
\langle S _ { w } ( b ) , h \rangle \geq c m \eta ( r ) r - \kappa ^ { 1 / 2 } r \tau _ { w } ( b ) .\tag{33}
$$

If $m \eta ( r ) \geq C | \mathcal { W } | \tau _ { w } ( b )$ for a large enough constant $C \geq 1$ , then (33) proves (32) with $w _ { \star } = w$ Otherwise, by (16) and (17),

$$
m \eta ( r ) \lesssim L ^ { 3 } \left( \sqrt { d A _ { w } ( b ) } + d \right) .\tag{34}
$$

We claim $L ^ { 3 } d$ term cannot account for more than a fixed fraction (say, half) of the LHS. For $r \leq 1$ that would imply $r \lesssim L ^ { 3 } d / m$ , whereas $r > \rho ( m ) \geq L ^ { 5 } d / m$ . For $r > 1$ , it would imply $m \lesssim L ^ { 3 } d ,$ yet $n \geq d$ and (4) imply $m \geq L ^ { 5 } d$ . Both cases incur a contradiction when L is defined with a suficiently large constant. Thus the square-root term in (34) must account for a fixed fraction, and

$$
A _ { w } ( b ) \gtrsim \frac { m ^ { 2 } \eta ( r ) ^ { 2 } } { L ^ { 6 } d } .\tag{35}
$$

The count comparison (30) gives $Q _ { w } ( b ) \geq A _ { w } ( b ) / 2 - C L d$ . To absorb this additive term, we claim that

$$
\frac { m ^ { 2 } \eta ( r ) ^ { 2 } } { L ^ { 6 } d } \geq L ^ { 4 } d .
$$

If $r \leq 1$ , this follows from $r > \rho ( m ) \geq L ^ { 5 } d / m ;$ otherwise, if $r > 1$ , this follows from (4) and $n \geq d .$ Together with (35), this shows $A _ { w } ( b ) \gtrsim L ^ { 4 } d$ , so when L is defined with a suficiently large constant,

$$
Q _ { w } ( b ) \gtrsim \frac { m ^ { 2 } \eta ( r ) ^ { 2 } } { L ^ { 6 } d } .\tag{36}
$$

We now aim to use this lower bound on $Q _ { w } ( b )$ to imply signal at scale $S _ { \infty } ( b )$ . Since $n \geq d ,$

$$
\tau _ { \infty } ( b ) = L ^ { 2 } \left( \sqrt { d ( n + d ) } + d \right) \leq 3 L ^ { 2 } \sqrt { d n } .
$$

Using (21) and (29) yields

$$
\langle S _ { \infty } ( b ) , h \rangle \ge c \frac { m ^ { 2 } \eta ( r ) ^ { 3 } r } { L ^ { 6 } d } - C L ^ { 2 } r \sqrt { d n } .\tag{37}
$$

It remains only to compare the leading term with the concentration error. $\mathrm { I f } \ r \leq 1$ , then the first term in (11) and $r > \rho ( m )$ give

$$
\frac { m ^ { 2 } \eta ( r ) ^ { 3 } } { L ^ { 6 } d } = \frac { m ^ { 2 } r ^ { 3 } } { L ^ { 6 } d } \geq L ^ { 9 } \sqrt { d n } .
$$

If $r > 1$ , then $\eta ( r ) = 1$ and (4) imply $m ^ { 2 } / ( L ^ { 6 } d ) \geq L ^ { 4 } \sqrt { d n }$ . Thus in both cases the leading term in (37) is at least $c L ^ { 4 } r { \sqrt { d n } }$ . When L is defined with a suficiently large constant, the leading term dominates the concentration error and the desired margin

$$
| \mathcal { W } | r \tau _ { \infty } ( b ) \leq 3 L ^ { 3 } r \sqrt { d n } . \mathrm { ( v i a ~ ( 1 6 ) ) }
$$

Thus (32) holds with $w _ { \star } = \infty$

Finally, $\| h \| _ { 2 } \le \kappa ^ { 1 / 2 } r$ , so (32) gives

$$
\| S _ { w _ { \star } } ( b ) \| _ { 2 } \geq 2 | \mathcal { W } | \tau _ { w _ { \star } } ( b ) > | \mathcal { W } | \tau _ { w _ { \star } } ( b ) .
$$

Hence, we conclude that $w _ { \star }$ is violated.

We now show the signal is still preserved once aggregated into $G _ { \mathrm { a g g } } ( b )$

Lemma 2.7 (Normalized aggregate direction). Define $D = \sqrt { \kappa n } \ge 1$ . Assume the m lower bound (4) and $b \in B _ { 2 } ( B )$ . Under the uniform convergence event, $i f \parallel b - \beta \parallel _ { \Sigma } > \rho ( m )$ , then $\mathcal { V } ( b ) \neq \emptyset$ ， $G _ { \mathrm { a g g } } ( b ) \neq 0$ , and

$$
\left. \frac { G _ { \mathrm { a g g } } ( b ) } { \| G _ { \mathrm { a g g } } ( b ) \| _ { 2 } } , \beta - b \right. \geq \frac { \| b - \beta \| _ { 2 } } { D } .\tag{38}
$$

Proof. Lemma 2.1 and Proposition 2.5 imply that for every tested scale $u ,$

$$
\langle S _ { u } ( b ) / \tau _ { u } ( b ) , h \rangle \geq - \kappa ^ { 1 / 2 } r .
$$

Lemma 2.6 supplies one violated scale $w _ { \star }$ for which

$$
\langle S _ { w _ { \star } } ( b ) / \tau _ { w _ { \star } } ( b ) , h \rangle \geq 2 \kappa ^ { 1 / 2 } | \mathcal { W } | r .
$$

There are at most $| \mathcal { W } | - 1$ other violated scales. Hence,

$$
\langle G _ { \mathrm { a g g } } ( b ) , h \rangle \geq 2 \kappa ^ { 1 / 2 } | \mathcal { W } | r - ( | \mathcal { W } | - 1 ) \kappa ^ { 1 / 2 } r \geq | \mathcal { W } | r .\tag{39}
$$

The feature norms guarantee of Proposition 2.5, $\| S _ { w } ( b ) \| _ { 2 } \le \sqrt { L d } A _ { w } ( b )$ , and $A _ { w } ( b ) \leq n$ yield

$$
\frac { \| S _ { w } ( b ) \| _ { 2 } } { \tau _ { w } ( b ) } \leq \frac { \sqrt { L d } \cdot A _ { w } ( b ) } { L ^ { 2 } \sqrt { d A _ { w } ( b ) } } \leq \sqrt { n } .
$$

Therefore,

$$
\| G _ { \mathrm { a g g } } ( b ) \| _ { 2 } \leq | \mathcal { W } | \cdot \sqrt { n } .\tag{40}
$$

Equations (39) and (40), together with $r \geq \| h \| _ { 2 } / { \sqrt { \kappa } } .$ , prove (38). They also show $G _ { \mathrm { a g g } } ( b ) \neq 0$ .

## 2.2.2 Concluding algorithmic guarantees

Algorithm description. We now more formally introduce our algorithm that uses the residualbalance vectors to descend towards an estimate close to $\beta .$ Our algorithm will repeatedly move in the direction of $G _ { \mathrm { a g g } } ( b )$ , with step sizes changing in diferent stages. Define the smallest stage radius as

$$
\rho _ { \mathrm { m i n } } = \rho ( n ) = L ^ { 5 } \left( \sqrt { d / n } + d / n \right) .\tag{41}
$$

Since $m \leq n$ and $\rho$ is nonincreasing, $\rho _ { \mathrm { m i n } } \le \rho ( m )$ . If $B \le \rho _ { \mathrm { m i n } }$ , the zero vector already has the desired accuracy, so the algorithm returns it. Otherwise set

$$
q = { \frac { 3 } { 4 } } , \qquad T _ { \mathrm { s t a g e } } = \left\lceil { \frac { \log ( B / \rho _ { \mathrm { m i n } } ) } { \log ( 4 / 3 ) } } \right\rceil .\tag{42}
$$

For $\ell = 0 , \ldots , T _ { \mathrm { s t a g e } }$ , let $R _ { \ell } = q ^ { \ell } B$ , which is chosen such that $R _ { T _ { \mathrm { s t a g e } } } \leq \rho _ { \mathrm { m i n } }$ . Finally, for each stage we choose the step size and number of iterations as

$$
\alpha _ { \ell } = \frac { R _ { \ell } } { 8 D } , \qquad N _ { \mathrm { i t } } = \lceil 6 4 D ^ { 2 } \rceil .\tag{43}
$$

Algorithm RB-Desc. If $B ~ \leq ~ \rho _ { \mathrm { m i n } }$ , return 0. Otherwise initialize $b ~ = ~ 0$ . For stages $\ell \ =$ $0 , 1 , \ldots , T _ { \mathrm { s t a g e } } - 1$

1. Set $\alpha = \alpha _ { \ell }$

2. For $j = 1 , \ldots , N _ { \mathrm { i t } }$

(a) Compute all scores $S _ { w } ( b )$ , counts $A _ { w } ( b )$ , and thresholds $\tau _ { w } ( b )$

(b) If $\mathcal { V } ( b ) = \emptyset$ , return b.

(c) Compute $\begin{array} { r } { G _ { \mathrm { a g g } } ( b ) = \sum _ { w \in \mathcal { V } ( b ) } \frac { S _ { w } ( b ) } { \tau _ { w } ( b ) } } \end{array}$ . If $G _ { \mathrm { a g g } } ( b ) = 0$ , return b.

(d) Set $u = G _ { \mathrm { a g g } } ( b ) / \left. G _ { \mathrm { a g g } } ( b ) \right. _ { 2 }$ and update

$$
\begin{array} { r } { b  \Pi _ { B _ { 2 } ( B ) } ( b + \alpha u ) , } \end{array}
$$

where $\Pi _ { B _ { 2 } ( B ) }$ is Euclidean projection onto $B _ { 2 } ( B )$

Return the final iterate.

Algorithm analysis. The following will let us show b gets close to $\beta \colon$

Lemma 2.8 (One-stage contraction). Assume (4) and work on the uniform convergence event. Consider a stage with radius R and the step size and iteration count as in (43). If the stage begins at b with $\| b - \beta \| _ { 2 } \leq R$ and

$$
R \geq 4 \sqrt { \kappa } \rho ( m ) ,\tag{44}
$$

then, unless the algorithm returns during the stage, its output $b ^ { + }$ satisfies

$$
\left\| b ^ { + } - \beta \right\| _ { 2 } \leq { \frac { 3 } { 4 } } R .
$$

Proof. For the t-th iteration in a stage, let $b _ { t }$ be the value of b at the start of the iteration, $D _ { t } =$ $\| b _ { t } - \beta \| _ { 2 } ^ { 2 }$ , and $u _ { t }$ is the value of u during this iteration. Whenever $\lVert b _ { t } - \beta \rVert _ { 2 } > R / 2$ , (44) implies $\| b _ { t } - \beta \| _ { \Sigma } > \rho ( m )$ . Hence Lemma 2.7 gives

$$
\langle u _ { t } , \beta - b _ { t } \rangle \geq \frac { \| b _ { t } - \beta \| _ { 2 } } { D } > \frac { R } { 2 D } = 4 \alpha .
$$

Since projection onto a closed convex set containing $\beta$ satisfies $\left\| \Pi _ { B _ { 2 } ( B ) } ( z ) - \beta \right\| _ { 2 } \leq \| z - \beta \| _ { 2 } ,$

$$
D _ { t + 1 } \leq \| b _ { t } + \alpha u _ { t } - \beta \| _ { 2 } ^ { 2 } = D _ { t } - 2 \alpha \left. u _ { t } , \beta - b _ { t } \right. + \alpha ^ { 2 } \leq D _ { t } - 7 \alpha ^ { 2 } .\tag{45}
$$

If the stage never entered $B _ { 2 } ( \beta , R / 2 )$ , this decrease would occur on every iteration, whereas

$$
N _ { \mathrm { i t } } \alpha ^ { 2 } \geq 6 4 D ^ { 2 } \frac { R ^ { 2 } } { 6 4 D ^ { 2 } } = R ^ { 2 } ,
$$

contradicting $D _ { 0 } \leq R ^ { 2 }$ and $D _ { t } > 0$ . Thus, the half-radius ball is entered. From a point within $R / 2$ the next update has length at most $R / ( 8 D ) \leq R / 8$ and therefore lands within $5 R / 8$ . From a point whose distance lies between $R / 2$ and $5 R / 8$ , the preceding descent inequality makes the distance decrease. Induction shows that every later iterate remains within $5 R / 8 < 3 R / 4$ □

Similar logic shows that once b is close to $\beta ,$ it will not move away:

Lemma 2.9. Assume (4) and work on the uniform convergence event. Define $\overline { { R } } = 4 \sqrt { \kappa } \rho ( m )$ . If $\| b - \beta \| _ { 2 } \leq \overline { { R } }$ and an update is made during a stage with $R _ { \ell } \leq \overline { { R } }$ , then the next iterate also lies in $B _ { 2 } ( \beta , { \overline { { R } } } )$

Proof. The update length is at most $\overline { { R } } / ( 8 D ) \leq \overline { { R } } / 8$ . Thus an iterate in $B _ { 2 } ( \beta , \overline { { R } } / 2 )$ remains inside $B _ { 2 } ( \beta , { \overline { { R } } } )$ after one update. If instead $\overline { { R } } / 2 < \| b - \beta \| _ { 2 } \leq \overline { { R } }$ , then $\| b - \beta \| _ { \Sigma } > \rho ( m )$ and

$$
\left. u , \beta - b \right. \geq \frac { \| b - \beta \| _ { 2 } } { D } > \frac { \overline { { R } } } { 2 D } \geq 4 \alpha _ { \ell } .
$$

By the same argument as (45), the squared Euclidean distance decreases, and hence no update crosses the boundary of $B _ { 2 } ( \beta , { \overline { { R } } } )$ □

We combine Lemmas 2.8 and 2.9 to conclude our desired guarantee:

Lemma 2.10. Assume (4) and work on the uniform convergence event. The output of RB-Desc satisfies

$$
\left\| { \widehat { \beta } } - \beta \right\| _ { 2 } \leq C \rho ( m ) .
$$

Proof. If $B \le \rho _ { \mathrm { m i n } }$ , then $\| \beta \| _ { 2 } \le B \le \rho ( m )$ , so the initial return is valid. Moving forward, we assume $B > \rho _ { \mathrm { m i n } }$

An early return with $\mathcal V ( b ) = \emptyset \mathrm { ~ o r ~ } G _ { \mathrm { a g g } } ( b ) = 0$ is within $C \rho ( m )$ by Lemma 2.7.

Suppose no early return occurs. As long as $R \ell \geq { \overline { { R } } } ,$ Lemma 2.8 and induction show that the stage begins within $R _ { \ell }$ and ends within $R _ { \ell + 1 } = 3 R _ { \ell } / 4$ . If some stage is the first with $R _ { \ell } < \overline { { R } }$ , its starting point is already in $B _ { 2 } ( \beta , \overline { { R } } )$ , and Lemma 2.9 keeps all later iterates there. Otherwise, the last stage ends within

$$
R _ { T _ { \mathrm { s t a g e } } } \leq \rho _ { \mathrm { m i n } } \leq \rho ( m ) < \overline { { R } } .
$$

Thus in every case $\left\| { \widehat { \beta } } - \beta \right\| _ { 2 } \leq { \overline { { R } } } = 4 { \sqrt { \kappa } } \rho ( m )$ , proving the claim.

This concludes the high-probability estimation error guarantee. We also observe the algorithm has the desired runtime:

Proposition 2.11. RB-Desc runs in time

$$
O _ { K , \kappa } ( L ^ { 2 } n ^ { 2 } d ) = O _ { K , \kappa } \left( n ^ { 2 } d \log ^ { 2 } \left( \frac { e n d ( B + 1 ) } { \delta } \right) \right) .
$$

Proof. There are $O ( \log ( 2 + B / \rho _ { \operatorname* { m i n } } ) )$ stages, and each stage has $O ( D ^ { 2 } ) = O _ { \kappa } ( n )$ inner iterations. One inner iteration costs $O ( n d | W | )$ arithmetic operations. In total, the number of arithmetic operations is

$$
O _ { K , \kappa } ( L ^ { 2 } n ^ { 2 } d ) = O _ { K , \kappa } \left( n ^ { 2 } d \log ^ { 2 } \left( \frac { e n d ( B + 1 ) } { \delta } \right) \right) .
$$

Proof of Theorem 1.5. The theorem follows immediately from the uniform convergence event (Proposition 2.5), estimation error guarantee (Lemma 2.10), and runtime bound (Proposition 2.11).

## 3 Adaptive linear regression

In this section, we study the task of adaptive linear regression. In Section 3.1, we give an (ineficient) estimator for when the errors are mixtures of symmetric, log-concave distributions. In Section 3.2, we give an eficient estimator for when the errors are a single symmetric, log-concave distribution.

## 3.1 Adaptive guarantees for symmetric, log-concave mixture errors

In this section, we show positive guarantees for an (ineficient) statistical estimator when the errors are mixtures of symmetric log-concave densities. This regression setting generalizes the main result of [CV26], and our estimator is a natural high-dimensional version of their approach.

We outline our approach in Section 1.2; we now set up our estimator more technically.

Formally, the error density belongs to the class

$$
\mathcal { P } _ { k } : = \left\{ p = \sum _ { \ell = 1 } ^ { k } w _ { \ell } p _ { \ell } : w _ { \ell } \geq 0 , \sum _ { \ell = 1 } ^ { k } w _ { \ell } = 1 , \ p _ { \ell } \mathrm { ~ l o g - c o n c a v e ~ a n d ~ s y m m e t r i c ~ a b o u t ~ } 0 \right\} .
$$

Later, we will also use the following condition:

Definition 3.1 (Small-ball condition). The design satisfies an $( c _ { 1 } , c _ { 2 } )$ -small-ball condition if

$$
\mathbb P \Big ( | X ^ { \top } u | \geq c _ { 1 } \left\| u \right\| _ { \Sigma } \Big ) \geq c _ { 2 } \qquad \mathrm { f o r ~ e v e r y ~ } u \in \mathbb R ^ { d } \mathrm { ~ w i t h ~ } \left\| u \right\| _ { \Sigma } > 0 .\tag{46}
$$

Symmetrizing the covariate distribution. Since our algorithm relies on testing some notion of symmetry, we will require that $X ^ { \top } v \overset { d } { = } X ^ { \top } ( - v )$ . This is not automatically true, since we do not assume any symmetry in the covariate distribution, yet the following classical transformation (e.g. this also appears in [dNS21, Section A.2]) resolves this issue.

Let $S _ { i }$ be independent Rademacher signs, independent of the sample, and define

$$
\widetilde { X } _ { i } = S _ { i } X _ { i } , \qquad \widetilde { y } _ { i } = S _ { i } y _ { i } , \qquad \widetilde { \varepsilon } _ { i } = S _ { i } \varepsilon _ { i } .\tag{47}
$$

Then,

$$
\widetilde { y } _ { i } = \widetilde { X } _ { i } ^ { \top } \beta + \widetilde { \varepsilon } _ { i } .
$$

Since the distribution of $\varepsilon _ { i }$ is symmetric, the distribution of $\widetilde { \varepsilon _ { i } }$ is unchanged, and $\widetilde { X } _ { i }$ and $\widetilde { \varepsilon } _ { i }$ are independent. ${ \widetilde { X } } _ { i }$ now satisfies our desired symmetry condition, while its tail bounds and second moment matrix are unafected since

$$
| \widetilde { X } ^ { \top } u | = | X ^ { \top } u | , \qquad \mathrm { E } [ \widetilde { X } \widetilde { X } ^ { \top } ] = \Sigma .
$$

For ease of notation, we will assume this preprocessing has already been applied in the remainder of this section, meaning that $X _ { i } , y _ { i }$ will implicitly represent $\widetilde { X } _ { i } , \widetilde { y } _ { i }$

Designing the detection sets. For a candidate $b \in \mathbb { R } ^ { d }$ , recall that the residual is $r _ { i } ( b ) =$ $y _ { i } - X _ { i } ^ { \top } b .$ . For $c \in \mathbb { R } ^ { d } , s > 0$ , and an interval $I = [ - b _ { 0 } , - a _ { 0 } ]$ with $0 \leq a _ { 0 } < b _ { 0 } < \infty$ , define

$$
N _ { b , c , s , I } ^ { + } : = \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ X _ { i } ^ { \top } c \geq s , \ r _ { i } ( b ) + X _ { i } ^ { \top } c \in I \} ,\tag{48}
$$

$$
N _ { b , c , s , I } ^ { - } : = \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ X _ { i } ^ { \top } c \leq - s , \ r _ { i } ( b ) - X _ { i } ^ { \top } c \in I \} ,\tag{49}
$$

and let

$$
\ell _ { \delta } : = \operatorname* { m a x } \{ 1 , \log ( 2 e n / \delta ) \} , \qquad A _ { n , d , \delta } : = ( d + 1 ) \ell _ { \delta } , \qquad \Gamma : = C _ { \Gamma } \sqrt { A _ { n , d , \delta } } ,\tag{50}
$$

where $C _ { \Gamma }$ is chosen to be suficiently large later. Then, we define a confidence set

$$
\mathcal { C } _ { n } : = \left. b \in \mathbb { R } ^ { d } : \left| \sqrt { N _ { b , c , s , I } ^ { + } } - \sqrt { N _ { b , c , s , I } ^ { - } } \right| \leq \Gamma \mathrm { ~ f o r ~ e v e r y ~ } c , s , I \right. .\tag{51}
$$

Currently, it should not be clear to the reader why this confidence set will typically contain only good estimates. Let us defer that question and first discuss some simpler intuition for why the confidence set will usually contain $\beta .$ . For the correct estimate $b = \beta$ , the residuals $r _ { i } ( b ) \ { \stackrel { d } { = } } \ \varepsilon _ { i }$ and are independent of $X _ { i }$ . Additionally, by symmetry of $X _ { i } .$ , the distribution of $X _ { i } ^ { \top } { \boldsymbol { \cdot } }$ c restricted to $X _ { i } ^ { \top } c \geq s$ should be exactly the same as the distribution of $- X _ { i } ^ { \top }$ c restricted to $X _ { i } ^ { \mp } c \leq - s$ . Hence, the expectation of both counts $N _ { b , c , s , I } ^ { + } , N _ { b , c , s , I } ^ { - }$ are the same when $b = \beta$ , and if we employ a uniform convergence result and set the threshold accordingly, then we should be able to show that $\beta \in { \mathcal { C } } _ { n }$ with high probability. The more interesting component of our proof is showing that only good estimates will be inside this confidence set.

Analysis. Moving forward more rigorously, observe how each detection set is an intersection of three halfspaces in $\mathbb { R } ^ { d + 1 }$ , so the relevant concept class has VC dimension $O ( d )$ . As we examine the counts inside these sets, the following relative uniform convergence guarantee will be helpful (the proof is standard and is deferred to Section C.2):

Lemma 3.2 (Uniform convergence for square-root counts). Let $Z _ { 1 } , \ldots , Z _ { n }$ be i.i.d. random elements of a Euclidean space. Let A be a concept class with VC dimension at most $V \geq 1$ . For $A \in { \mathcal { A } }$ define

$$
N ( A ) : = \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ Z _ { i } \in A \} , \qquad Q ( A ) : = \sum _ { i = 1 } ^ { n } \mathbb { P } ( Z _ { i } \in A ) .
$$

For $\delta \in ( 0 , 1 / 2 )$ , let $\ell _ { \delta } = \log ( 2 e n / \delta )$ . Then, with probability at least $1 - \delta$

$$
\operatorname* { s u p } _ { A \in \mathcal { A } } \left. \sqrt { N ( A ) } - \sqrt { Q ( A ) } \right. \leq C \sqrt { V \ell _ { \delta } } ,\tag{52}
$$

for a universal constant $C > 0$

The following is the main theorem, which will immediately imply Theorem 1.7:

Theorem 3.3. Assume $p \in \mathcal { P } _ { k }$ , the covariates are κ-well-conditioned, and satisfy $\textit { a } \left( c _ { 1 } , c _ { 2 } \right) - s m a l l -$ ball condition, where $\kappa \geq 1$ and $c _ { 1 } , c _ { 2 } > 0$ are all constants. There exists a constant $C _ { c _ { 1 } , c _ { 2 } , \kappa } \geq$ 1 depending only on $c _ { 1 } , c _ { 2 }$ , κ such that the following holds. If C is suficiently large, then with probability at least $1 - \delta$ the confidence set (51) satisfies:

(a) $\beta \in { \mathcal { C } } _ { n } ;$

(b) for every $b \in { \mathcal { C } } _ { n }$

$$
\| b - \beta \| _ { 2 } \leq C _ { c _ { 1 } , c _ { 2 } , \kappa } \omega _ { p } \left( C _ { c _ { 1 } , c _ { 2 } , \kappa } \frac { d k \log ( 2 n / \delta ) \log ( 2 k n ) \log ( 2 n ) } { n } \right) .\tag{53}
$$

Proof. Work on the event of Lemma 3.2 which holds with probability $1 - \delta$

The true parameter passes. This follows from the earlier-discussed intuition. At $b = \beta , r _ { i } ( b ) = \varepsilon _ { i }$ For any $c , s , I ,$ , since $X ^ { \top } c ^ { \underline { { d } } } - X ^ { \top } c$ and $X ^ { \top } c$ is independent of $\varepsilon _ { i }$

$$
\begin{array} { r } { \mathbb { P } \{ X ^ { \top } c \geq s , \varepsilon + X ^ { \top } c \in I \} = \mathbb { P } \{ X ^ { \top } c \leq - s , \varepsilon - X ^ { \top } c \in I \} . } \end{array}
$$

Thus, the expected counts are equal. By Lemma 3.2, their empirical square-root counts difer by at most $2 C \sqrt { A _ { n , d , \delta } } ;$ we choose $C _ { \Gamma }$ large enough to cover this constant, and thus $\beta \in { \mathcal { C } } _ { n }$

A bad estimate produces a population imbalance. Fix $b \in \mathcal { C } _ { n }$ and let $u = b - \beta$ . We will aim to witness a poor estimate with a test that chooses $c = u$ . We focus on the case where $u \ne 0$ as otherwise $b = \beta$ . For convenience, we define

$$
T = X ^ { \top } u , \qquad T _ { i } = X _ { i } ^ { \top } u ,
$$

and remark that $r _ { i } ( b ) = \varepsilon _ { i } - T _ { i }$ . At this point, it will be very informative to revisit the detection sets with these choices,

$$
N _ { b , c , s , I } ^ { + } : = \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ X _ { i } ^ { \top } c \geq s , \ r _ { i } ( b ) + X _ { i } ^ { \top } c \in I \} = \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ T _ { i } \geq s , \ \varepsilon _ { i } \in I \} ,\tag{54}
$$

$$
N _ { b , c , s , I } ^ { - } : = \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ X _ { i } ^ { \top } c \leq - s , ~ r _ { i } ( b ) - X _ { i } ^ { \top } c \in I \} = \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ T _ { i } \leq - s , ~ \varepsilon _ { i } - 2 T _ { i } \in I \} .\tag{55}
$$

This is the key insight for the analysis: the first statistic (54) roughly corresponds to counting the number of samples where $\varepsilon _ { i }$ is inside an interval I, and the second statistic (55) roughly corresponds to counting the number of samples where a translation of $\varepsilon _ { i }$ is inside I, with the translation being at least $2 s .$ . The main point is that we have now roughly reduced the problem to distinguishing samples from $\varepsilon _ { i }$ against samples from a translation, via an interval statistic. This is precisely the main technical ingredient in [CV26], and we may use their following result:

Lemma 3.4 (Interval witness; follows from Corollary 2.18 of [CV26]). There is a universal constant $c _ { 0 } > 0$ such that the following holds. Let $p \in \mathcal { P } _ { k } , \Delta > 0$ , and $h = H _ { p } ( \Delta ) \in ( 0 , 1 ]$ . There is an interval

$$
I = [ - b , - a ] , \qquad 0 \leq a < b < \infty ,
$$

such that

$$
\left( \sqrt { \mathbb { P } _ { p } ( I ) } - \sqrt { \mathbb { P } _ { p _ { \Delta } } ( I ) } \right) ^ { 2 } \ge c _ { 0 } \frac { h } { k \log ( 4 k / h ) \log ( 4 / h ) } .\tag{56}
$$

Moreover, for every $\Delta ^ { \prime } \ge \Delta$

$$
\mathbb { P } _ { p _ { \Delta ^ { \prime } } } ( I ) \leq \mathbb { P } _ { p _ { \Delta } } ( I ) \leq \mathbb { P } _ { p } ( I ) .\tag{57}
$$

We now work to use this in our setting. Define

$$
m ( s ) : = \mathbb { P } ( T \geq s ) = \mathbb { P } ( T \leq - s ) , \qquad h ( s ) : = H _ { p } ( 2 s ) .
$$

Invoking Lemma 3.4 with $\Delta = 2 s$ yields an interval $I _ { s }$ . We compute the probability of a sample contributing to $N _ { b , c , s , I _ { s } } ^ { + }$ as

$$
P _ { s } ^ { + } : = \mathbb { P } ( T \geq s , \ \varepsilon \in I _ { s } ) = m ( s ) \mathbb { P } _ { p } ( I _ { s } ) .
$$

For $N _ { b , c , s , I _ { s } } ^ { - }$ , we use that the conditional translation is $- 2 T \geq 2 s$ , and (57), to bound

$$
P _ { s } ^ { - } : = \mathbb { P } ( T \leq - s , \ \varepsilon - 2 T \in I _ { s } ) \leq m ( s ) \mathbb { P } _ { p _ { 2 s } } ( I _ { s } ) .
$$

Consequently,

$$
\left( \sqrt { P _ { s } ^ { + } } - \sqrt { P _ { s } ^ { - } } \right) ^ { 2 } \geq m ( s ) \left( \sqrt { \mathbb { P } _ { p } ( I _ { s } ) } - \sqrt { \mathbb { P } _ { p _ { 2 s } } ( I _ { s } ) } \right) ^ { 2 } \geq c _ { 0 } m ( s ) \frac { h ( s ) } { k \log ( 4 k / h ( s ) ) \log ( 4 / h ( s ) ) } .
$$

Leveraging the $( c _ { 1 } , c _ { 2 } )$ -small-ball condition, by choosing $s = c _ { 1 } \| u \| _ { \Sigma }$ , we may use $m ( s ) \geq c _ { 2 } / 2$ which implies

$$
\left( \sqrt { P _ { s } ^ { + } } - \sqrt { P _ { s } ^ { - } } \right) ^ { 2 } \gtrsim \frac { h ( s ) } { k \log ( 4 k / h ( s ) ) \log ( 4 / h ( s ) ) } .\tag{58}
$$

This roughly indicates that as $h ( \| b - \beta \| _ { \Sigma } )$ gets large, this becomes detectable by our tests.

Passing the empirical test bounds the estimation error. Since $b \in \mathcal { C } _ { n }$ and both counts satisfy Lemma 3.2,

$$
\begin{array} { r l } & { \sqrt { n } \left| \sqrt { { P } _ { s } ^ { + } } - \sqrt { { P } _ { s } ^ { - } } \right| \leq \left| \sqrt { n { P } _ { s } ^ { + } } - \sqrt { { N } _ { b , c , s , l _ { s } } ^ { + } } \right| + \left| \sqrt { { N } _ { b , c , s , l _ { s } } ^ { + } } - \sqrt { { N } _ { b , c , s , l _ { s } } ^ { - } } \right| + \left| \sqrt { n { P } _ { s } ^ { - } } - \sqrt { { N } _ { b , c , s , l _ { s } } ^ { - } } \right| } \\ & { \qquad \leq ( { C } _ { \Gamma } + 2 C ) \sqrt { { A } _ { n , d , \hat { s } } } . } \end{array}
$$

Combining with (58) yields

$$
h ( s ) \leq C ^ { \prime } { \frac { k A _ { n , d , \delta } \log { \frac { 4 k } { h ( s ) } } \log { \frac { 4 } { h ( s ) } } } { n } } ,
$$

for some constant $C ^ { \prime } \geq 1$ . We observe that this further implies

$$
h ( s ) \leq C ^ { \prime } { \frac { k A _ { n , d , \delta } \log ( 4 k n ) \log ( 4 n ) } { n } } .\tag{59}
$$

For sake of contradiction, if (59) did not hold, then using $h ( s ) \geq 1 / n$

$$
C ^ { \prime } \frac { k A _ { n , d , \delta } \log ( 4 k n ) \log ( 4 n ) } { n } < h ( s ) \leq C ^ { \prime } \frac { k A _ { n , d , \delta } \log \frac { 4 k } { h ( s ) } \log \frac { 4 } { h ( s ) } } { n } \leq C ^ { \prime } \frac { k A _ { n , d , \delta } \log ( 4 k n ) \log ( 4 n ) } { n } ,
$$

which is a contradiction. We recall our choice of $s = c _ { 1 } \| u \| _ { \Sigma } = c _ { 1 } \| b - \beta \| _ { \Sigma }$ to conclude

$$
\begin{array} { r l } & { H _ { p } ( 2 c _ { 1 } \| b - \beta \| _ { \Sigma } ) \leq C \frac { k A _ { n , d , \delta } \log ( 4 k n ) \log ( 4 n ) } { n } } \\ & { \implies 2 c _ { 1 } \| b - \beta \| _ { \Sigma } \leq \omega _ { p } \left( C \frac { k A _ { n , d , \delta } \log ( 4 k n ) \log ( 4 n ) } { n } \right) } \\ & { \implies \| b - \beta \| _ { 2 } \leq C \omega _ { p } \left( C \frac { d k \log ( 2 n / \delta ) \log ( 2 k n ) \log ( 2 n ) } { n } \right) \ \sqsubset } \end{array}
$$

Proof of Theorem 1.7. This follows immediately from Theorem 3.3; the small-ball condition is implied, for example, by Claim 2.3 (where $c _ { 1 } , c _ { 2 }$ depend only on K). We output any $b \in { \mathcal { C } } _ { n }$ (tiebreak arbitrarily), or an arbitrary estimate if ${ \mathcal { C } } _ { n }$ is empty. □

## 3.2 Eficient estimation for symmetric, log-concave errors (k = 1)

In this section, we eficiently obtain Hellinger modulus rates for symmetric, log-concave errors (Theorem 1.6).

Proof intuition. Our estimator for this task will be a form of $L _ { q }$ regression where we choose q depending on the data. This style of estimator has been comprehensively studied by [KXZ24]; in Section 1.2 we discuss some of the technical obstacles for this technique. When analyzing the estimator, there are two main components: (i) proving there exists a q where $L _ { q }$ regression produces a good enough estimate, and (ii) adaptively choosing the right $q .$

The adaptive choice (ii) of q follows simply for our purposes. Suppose we have divided our samples into a small number of blocks, and we are considering a fixed q. If we compute the $L _ { q }$ estimator for each block, and most of these estimates are close to some value ${ \widehat { \beta } } ,$ then we may show that it is very likely the true $\beta$ is close to ${ \widehat { \beta } } .$ . This intuition leads to a rigorous method for choosing the estimate, where you choose the value of $q$ with the best concentration. (In particular, we choose an estimate that has a large fraction of the other estimates in the smallest-possible radius around it; this style of technique appears previously in works such as [NRS07, HS14, PJL22].)

The main dificulty is in (i): showing there exists a choice of $q$ where $L _ { q }$ regression recovers the Hellinger modulus error. A guiding intuition comes from the work of [CV26], where they observe the Hellinger distance between a log-concave distribution and its translation is nearly captured by counting the number of samples inside a particular halfline (this uses the reverse data processing inequality of [PJL23]). This may be converted into a quantitative guarantee about the tail of $p$ that relates to the Hellinger modulus; we will use this tail control to prove guarantees for $L _ { q }$ regression.

Another useful intuition is that $L _ { q }$ regression may have guarantees in terms of the moments of $\varepsilon \sim p ;$ this is crucially leveraged in prior works. Consider the following quantities,

$$
M _ { q } = \mathrm { E } [ | \varepsilon | ^ { q } ] , \qquad V _ { p } ( q ) = \frac { M _ { 2 q - 2 } } { ( q - 1 ) ^ { 2 } M _ { q - 2 } ^ { 2 } } \qquad Q _ { p } ( q ) = \frac { M _ { 2 q - 4 } } { M _ { q - 2 } ^ { 2 } } .
$$

We will leverage guarantees in terms of such quantities, as is done in $\mathrm { e . g . }$ . [EHE23, KXZ24]. Roughly, $Q _ { p } ( q )$ is a term that controls curvature, and when $\begin{array} { r } { Q _ { p } ( q ) \ll \frac { n } { d } } \end{array}$ , then $L _ { q }$ estimation will typically have error at the scale

$$
\left. \widehat { \beta } - \beta \right. _ { 2 } \lesssim \sqrt { \frac { d V _ { p } ( q ) } { n } } .
$$

This form of guarantee is quite strong; for example, it yields the right error for Gaussian errors with $q = 2$ , and uniform errors with $q \asymp n$

The main goal is to turn the earlier-mentioned tail control into a strong enough bound for $V _ { p } ( q )$ with the correct $q ;$ the hope is that the right choice of $q$ can extract the signal from the tail decay. We now very loosely describe the structure of this proof. Consider any t where $\begin{array} { r } { \mathrm { H } ^ { 2 } ( p , p _ { t } ) \approx \frac { d } { n } } \end{array}$ . If we could show there exists a choice of $q$ where

$$
V _ { p } ( q ) \ll \frac { t ^ { 2 } n } { d } , \qquad Q _ { p } ( q ) \ll \frac { n } { d } ,\tag{60}
$$

then this would be suficient for our goals. Suppose we know the distribution $p$ has a tail such that

$$
\begin{array} { r } { \mathbb { P } _ { \varepsilon \sim p } ( | \varepsilon | \ge r ) \approx \tau , } \end{array}
$$

and the remaining $\tau$ mass decays exponentially over a local length $\ell .$ In the most optimistic setting, there might be a choice of $q$ where the moment bounds satisfy

$$
M _ { q - 2 } \approx \tau r ^ { q - 2 } , \qquad M _ { 2 q - 2 } \approx \tau r ^ { 2 q - 2 } , \qquad M _ { 2 q - 4 } \approx \tau r ^ { 2 q - 4 } .
$$

For this target, a natural choice of $q$ is the largest value such that the integrand of the moment integral is decreasing at $r ,$ which gives

$$
\begin{array} { l } { \displaystyle \frac { d } { d x } \left( q ( r + x ) ^ { q - 1 } \mathbb { P } ( | \varepsilon | \geq r + x ) \right) \Big | _ { x = 0 } \approx \frac { d } { d x } \left( q ( r + x ) ^ { q - 1 } \tau \exp ( - x / \ell ) \right) \Big | _ { x = 0 } \propto - r ^ { q - 1 } \frac { 1 } { \ell } + ( q - 1 ) r ^ { q - 2 } } \\ { \displaystyle \implies q \approx \frac { r } { \ell } . } \end{array}
$$

If this optimistic heuristic were true, then we could use the bounds

$$
V _ { p } ( q ) \approx \frac { \ell ^ { 2 } } { \tau } , \qquad Q _ { p } ( q ) \approx \frac { 1 } { \tau } .
$$

Fortuitously, the Hellinger tail control result will control precisely these two quantities $\frac { \ell ^ { 2 } } { \tau }$ and $\scriptstyle { \frac { 1 } { \tau } }$ such that $V _ { p } ( q )$ and $Q _ { p } ( q )$ would satisfy the desired conditions (60). While many of these calculations are heuristic, such a proof follows with technical work.

Algorithm description. Our algorithm will consist of computing $L _ { q }$ regression estimators for many blocks of samples. Let

$$
B = \lceil C \log ( 6 4 n / \delta ) \rceil , \qquad m = \left\lfloor \frac { n } { B } \right\rfloor ,
$$

where $C > 0$ is a suficiently large universal constant. We split the data into B blocks of size $m ,$ , where the blocks will be used to generate candidate estimates. As is discussed and justified in Section 3.1, we will consider a preprocessing of all samples where $( X _ { i } , Y _ { i } )  ( S _ { i } X _ { i } , S _ { i } Y _ { i } )$ for independent Rademacher signs $S _ { i } ;$ hence, we may consider the $X _ { i }$ as centrally symmetric and the $\varepsilon _ { i }$ as still independent of $X _ { i }$ . Our algorithm will only consider $L _ { q }$ regression for $q$ that are powers of 2 (as is done in [KXZ24]):

$$
\mathcal { G } _ { m } = \{ 2 ^ { j } : j \in \{ 1 , \dots , \lfloor \log _ { 2 } m \rfloor \} \} .
$$

We will compute the $L _ { q }$ regression estimator for every block and every $q \in \mathcal { G } _ { m }$ . For a fixed $q ,$ we denote our polynomial-time approximate optimizers for each of the B blocks as

$$
\widehat { A } _ { q , 1 } , \ldots , \widehat { A } _ { q , B } ,
$$

and denote the exact optimizers as

$$
A _ { q , 1 } , \dotsc , A _ { q , B } ,
$$

where they can be any exact optimizer if there are multiple. Roughly, we will use that an estimate $\widehat { A } _ { q , i }$ is better if there are many estimates $\widehat { A } _ { q , j }$ near it. We introduce $r _ { q , i }$ as an efective radius for the estimate $\widehat { A } _ { q , i } ;$ we define it as the empirical $3 / 4$ quantile of the Euclidean distances

$$
\left\| \widehat { A } _ { q , j } - \widehat { A } _ { q , i } \right\| _ { 2 } , \qquad j \in \{ 1 , \ldots , B \} \setminus \{ i \} .
$$

Finally, we will choose the estimate with the smallest efective radius, denoted by

$$
( \widehat { q } , \widehat { i } ) = \underset { q \in { \mathcal G } _ { m } , i \in \{ 1 , \ldots , B \} } { \arg \operatorname* { m i n } } r _ { q , i } , \widehat { \beta } = \widehat { A } _ { \widehat { q } , \widehat { i } } .
$$

The algorithm runs in polynomial time since there are polylog $( n / \delta )$ instances of $L _ { q }$ regression computed, each taking polynomial time.

$L _ { q }$ regression guarantee in terms of moment control. We now provide a regression guarantee in terms of the moment control. We do not believe this guarantee requires anything conceptually novel compared to prior works (e.g. [EHE23, KXZ24]), yet we require a non-asymptotic guarantee with the right dependence on q (since our q may take super-constant values) in a way that we do not know an external result we may invoke as a black-box. We defer the proof of the following to Section D.2:

Proposition 3.5 (Fixed-q regression bound). There exists a constant $C _ { K , \kappa }$ such that, if

$$
m \geq C _ { K , \kappa } d Q _ { p } ( q ) \log ( 2 m ) ,\tag{61}
$$

then with probability at least $1 7 / 1 8$ , the exact empirical $L _ { q }$ optimizer $( f o r \ q \ge 2 )$ given m samples satisfies

$$
\left\| \widehat { \beta } _ { q } - \beta \right\| _ { 2 } \leq C _ { K , \kappa } \sqrt { \frac { d V _ { p } ( q ) } { m } } ,\tag{62}
$$

and the design matrix $\mathbf { X } \in \mathbb { R } ^ { m \times d }$ has $\operatorname { r a n k } ( \mathbf { X } ) = d ,$ , implying the optimizer is unique.

Note that ${ \widehat { \beta } } _ { q }$ in Proposition 3.5 refers to the exact empirical $L _ { q }$ optimizer. $L _ { q }$ regression is a convex optimization task which permits eficient algorithms, but not with exact solutions. We use existing algorithms that enable an approximate solution (proof deferred to Section D.3):

Corollary 3.6 (Polynomial-time regression accuracy). There exist constants $c _ { K , \kappa } > 0$ and $C _ { K , \kappa } \geq 1$ such that, if

$$
m \geq C _ { K , \kappa } d \log ( 2 m ) ,\tag{63}
$$

then with probability at least 17/18,

$$
\sigma _ { m i n } ^ { 2 } ( { \bf X } ) \geq c _ { K , \kappa } m\tag{64}
$$

holds for the design matrix $\mathbf { X } \in \mathbb { R } ^ { m \times d }$ . Further, when (64) holds, the algorithm of [AKPS24] outputs an estimate $\widehat { A } _ { q }$ such that it is close to the unique, exact $L _ { q }$ optimizer ${ \widehat { \beta } } _ { q }$

$$
\left\| \widehat { A } _ { q } - \widehat { \beta } _ { q } \right\| _ { 2 } \leq \gamma ,
$$

and only requires

$$
O \left( q ^ { 2 } m ^ { 1 / 3 } \operatorname* { m a x } \left\{ 1 , \log \left( \frac { C _ { K , \kappa } m \left\| \mathbf { Y } \right\| _ { \infty } } { \gamma } \right) \right\} \right)
$$

calls to a linear system solver $( f o r 0 < \gamma \leq 1 )$

Note that the condition on m (63) holds since we may assume $n \geq C _ { K , \kappa } d \log ^ { 4 } ( 6 4 n / \delta )$ (for a large enough choice of constant), as otherwise the desired coeficient bound for the theorem is infinite.

We will use the algorithm of [AKPS24] for eficiently estimating the $L _ { q }$ regression parameters when the singular value condition of (64) holds, and otherwise we may output anything.

Proving desirable moment control. We will leverage the reverse data processing inequality proven by [PJL23]; we state a version that follows after using [CV26, Remark 2.3]:

Theorem 3.7 (Reverse data processing inequality; follows from Corollary 3.4 of [PJL23]). Let $P , Q$ be distributions with densities $p , q ,$ and let

$$
\eta = \mathrm { H } ^ { 2 } ( P , Q ) \in ( 0 , 1 ] .
$$

After exchanging P and Q if necessary, there exists a $\lambda \geq 1$ where the likelihood-threshold event

$$
A = \{ x : p ( x ) / q ( x ) \geq \lambda \}
$$

satisfies

$$
\left( \sqrt { P ( A ) } - \sqrt { Q ( A ) } \right) ^ { 2 } \geq \frac { \eta } { 1 8 0 0 \log ( 4 / \eta ) } , \qquad P ( A ) \geq \frac { \eta } { 1 8 0 0 \log ( 4 / \eta ) } .\tag{65}
$$

Further (as noted in [CV26]), when $P = p$ and $Q = p _ { t }$ for a log-concave $p ,$ then the likelihood ratio is monotone, and hence the set A is simply a halfline. For some rough intuition, this result indicates that if $p$ and $p _ { t }$ are distinguishable from $N \approx 1 / \eta$ samples, then they are also distinguishable from $\approx N$ log $N$ samples just by counting the number of samples in some halfline. Theorem 3.7 will enable helpful tail control for $p$ that aligns with the desired control in the proof intuition:

Lemma 3.8 (Informative tail scale). Let $t > 0$ , p is symmetric and log-concave, and

$$
\eta = H _ { p } ( t ) , \qquad L _ { \eta } = \log ( 4 / \eta ) ,
$$

where $0 < \eta \leq 1$ . There exists a choice of $r \geq 0$ , one-sided tail mass

$$
\tau = \mathbb { P } _ { \varepsilon \sim p } ( \varepsilon \geq r ) ,
$$

and a local tail length

$$
\ell = \frac { \tau } { p ( r ) } ,\tag{66}
$$

such that

$$
\tau \geq \frac { \eta } { 3 6 0 0 L _ { \eta } } ,\tag{67}
$$

and

$$
\frac { \ell ^ { 2 } } { \tau } \leq 1 8 0 0 \frac { t ^ { 2 } L _ { \eta } } { \eta } .\tag{68}
$$

Proof. Apply Theorem 3.7 to $p$ and $p _ { t }$ . By symmetry, there exists a choice of A of the form $A = ( - \infty , a ]$ . Let F be the CDF of $p ,$ and denote

$$
\alpha = P ( A ) = F ( a ) , \qquad \beta = Q ( A ) = F ( a - t ) , \qquad \Delta = \alpha - \beta \ge 0 ,
$$

where Theorem 3.7 implies

$$
g : = ( \sqrt { \alpha } - \sqrt { \beta } ) ^ { 2 } \geq \frac { \eta } { 1 8 0 0 L _ { \eta } } .
$$

We distinguish three positions of the interval $[ a - t , a ]$ relative to the mode zero.

Case $( i ) \colon a \leq 0$ . Choose $r = - a$ and $\tau = F ( a ) = \mathbb { P } ( \varepsilon \geq r )$ by symmetry. Since $p$ is nondecreasing on $( - \infty , 0 ]$

$$
\Delta = \int _ { a - t } ^ { a } p ( x ) d x \leq t p ( a ) = t p ( r ) .
$$

Moreover,

$$
\frac { \Delta ^ { 2 } } { \tau } = g \frac { ( \sqrt { \alpha } + \sqrt { \beta } ) ^ { 2 } } { \alpha } \geq g , \qquad \tau = \alpha \geq g .
$$

Case $( i i ) \colon a - t \geq 0$ . Choose $r = a - t$ and $\tau = 1 - F ( a - t ) = \mathbb { P } ( \varepsilon \geq r )$ . Since $p$ is nonincreasing on $[ 0 , \infty )$ 2

$$
\Delta = \int _ { r } ^ { r + t } p ( x ) d x \leq t p ( r ) ,
$$

and similarly using $\alpha , \beta \ge 1 / 2$

$$
\frac { \Delta ^ { 2 } } { \tau } = g \frac { ( \sqrt { \alpha } + \sqrt { \beta } ) ^ { 2 } } { 1 - \beta } \ge g , \qquad \tau \ge \alpha - \beta \ge g .
$$

Case (iii): $a - t < 0 < a$ . Choose $r = 0$ and $\tau = 1 / 2$ . Since $p ( x ) \leq p ( 0 )$ 2

$$
\Delta = \int _ { a - t } ^ { a } p ( x ) d x \leq t p ( r ) .
$$

and similarly using $\alpha \ge 1 / 2$

$$
\frac { \Delta ^ { 2 } } { \tau } = 2 g ( \sqrt { \alpha } + \sqrt { \beta } ) ^ { 2 } \geq g , \qquad \tau = 1 / 2 \geq g / 2 .
$$

Combining cases. All cases satisfy

$$
\tau \geq \frac { \eta } { 3 6 0 0 L _ { \eta } } , \qquad \frac { \Delta ^ { 2 } } { \tau } \geq \frac { \eta } { 1 8 0 0 L _ { \eta } } , \qquad \Delta \leq t p ( r ) .
$$

Finally,

$$
\frac { \ell ^ { 2 } } { \tau } = \frac { \tau } { p ( r ) ^ { 2 } } \leq \frac { t ^ { 2 } \tau } { \Delta ^ { 2 } } \leq 1 8 0 0 \frac { t ^ { 2 } L _ { \eta } } { \eta } .
$$

This enables the desired moment control (this is mostly calculation):

Lemma 3.9 (Moment control). Under the assumptions of Lemma 3.8, there is a positive integer $2 \leq q \leq 2 + \frac { 9 0 0 L _ { \eta } } { \eta }$ such that

$$
V _ { p } ( q ) \lesssim \frac { t ^ { 2 } L _ { \eta } ^ { 3 } } { \eta } , \qquad Q _ { p } ( q ) \lesssim \frac { L _ { \eta } ^ { 2 } } { \eta } .\tag{69}
$$

Proof. Let

$$
R = | \varepsilon | , \qquad S ( x ) = \mathbb { P } ( R \geq x ) ,
$$

and let $( r , \tau , \ell )$ be supplied by Lemma 3.8. Thus

$$
S ( r ) = 2 \tau , \qquad - \left. \frac { d } { d x } \log S ( x ) \right| _ { x = r } = \frac { 1 } { \ell } .\tag{70}
$$

First, we note that $S ( x )$ is log-concave. To prove this, consider the function $F ( x , y ) = \mathbf { 1 } \left[ y \geq x \right] R ( y )$ Observe that F is log-concave because both $\mathbf { 1 } \left[ y \geq x \right]$ and $R ( y )$ are log-concave, and the product of log-concave functions is log-concave. We can write

$$
S ( x ) = \int _ { \mathbb { R } } F ( x , y ) d y ,
$$

and by Prékopa’s theorem [Pré73, Theorem 6] this implies $S ( x )$ is log-concave. Using (70) and log-concavity of $S ( x )$ , we may bound that for every $y \geq 0$ and every $0 \leq y \leq r ,$ respectively,

$$
S ( r + y ) \leq 2 \tau e ^ { - y / \ell } , \qquad S ( r - y ) \leq \operatorname* { m i n } \{ 1 , 2 \tau e ^ { y / \ell } \} .\tag{71}
$$

Let us denote

$$
L _ { \tau } = \log ( e / \tau ) , \qquad h = r / \ell .
$$

Using (67),

$$
L _ { \tau } \leq \log \left( \frac { 3 6 0 0 e L _ { \eta } } { \eta } \right) \lesssim L _ { \eta } .
$$

We will give the desired moment control based on two separate cases, depending on whether h is small (corresponding to when r is not too far out in terms of the local tail length), or whether h is large.

Case $( i ) \colon h \leq 2 4 L _ { \tau }$ . In this case, the tail is not too far out and we will choose $q = 2$ . We may write the second moment as

$$
M _ { 2 } = 2 \int _ { 0 } ^ { \infty } x S ( x ) d x .
$$

We split the integral at r and bound

$$
\begin{array} { l } { { \displaystyle M _ { 2 } = 2 \int _ { 0 } ^ { r } x S ( x ) d x + 2 \int _ { r } ^ { \infty } x S ( x ) d x } } \\ { { \displaystyle \quad \leq 2 \int _ { 0 } ^ { r } x d x + 4 \tau \int _ { 0 } ^ { \infty } ( r + y ) e ^ { - y / \ell } d y \quad \mathrm { ( v i a ~ } S ( x ) \leq 1 \mathrm { ~ a n d ~ t h e ~ t a i l ~ b o u n d ~ i n ~ ( 7 1 ) ) } } } \\ { { \displaystyle = r ^ { 2 } + 4 \tau ( r \ell + \ell ^ { 2 } ) \leq r ^ { 2 } + 4 r \ell + 4 \ell ^ { 2 } = ( r + 2 \ell ) ^ { 2 } } } \\ { { \displaystyle \quad \lesssim L _ { \tau } ^ { 2 } \ell ^ { 2 } . \quad \mathrm { ( v i a ~ } r = h \ell \leq 2 4 L _ { \tau } \ell ) } } \end{array}
$$

Hence, we may bound $V _ { p } ( 2 )$ and $Q _ { p } ( 2 )$ as desired,

$$
\begin{array} { l } { { V _ { p } ( 2 ) = M _ { 2 } \lesssim L _ { \eta } ^ { 2 } \displaystyle \frac { \ell ^ { 2 } } { \tau } \lesssim \displaystyle \frac { t ^ { 2 } L _ { \eta } ^ { 3 } } { \eta } \quad \mathrm { ( v i a ~ ( 6 8 ) ) } } } \\ { { Q _ { p } ( 2 ) = 1 \lesssim L _ { \eta } ^ { 2 } / \eta } } \end{array}
$$

Case $( i i ) \colon h > 2 4 L _ { \tau }$ . In this case, we choose a larger $q .$ Consider any even integer k satisfying

$$
h \left( 1 - \frac { 1 } { 3 L _ { \tau } } \right) \leq k \leq h \left( 1 - \frac { 1 } { 4 L _ { \tau } } \right) .\tag{72}
$$

This interval has length $h / ( 1 2 L _ { \tau } ) > 2$ , so it contains an even integer. We choose

$$
q = { \frac { k + 2 } { 2 } } .
$$

The relevant moments in the numerators of $V _ { p } ( q )$ and $Q _ { p } ( q )$ are $M _ { s }$ for $s \in [ k - 2 , k ]$ . We will prove that for every such $s \in [ k - 2 , k ]$ 2

$$
M _ { s } \lesssim L _ { \tau } \tau r ^ { s } .\tag{73}
$$

We again use the expression

$$
M _ { s } = s \int _ { 0 } ^ { \infty } x ^ { s - 1 } S ( x ) d x
$$

and will split this integral into

$$
M _ { s } = I _ { - } + I _ { + } , \qquad I _ { - } = s \int _ { 0 } ^ { r } x ^ { s - 1 } S ( x ) d x , \qquad I _ { + } = s \int _ { r } ^ { \infty } x ^ { s - 1 } S ( x ) d x .
$$

For the right-tail integral, write $x = r + y$ . Since

$$
\left( r + y \right) ^ { s - 1 } = r ^ { s - 1 } \left( 1 + \frac { y } { r } \right) ^ { s - 1 } \leq r ^ { s - 1 } e ^ { ( s - 1 ) y / r } ,
$$

we may bound $I _ { + }$ by

$$
\begin{array} { r l } & { I _ { + } \leq 2 s \tau r ^ { s - 1 } \displaystyle \int _ { 0 } ^ { \infty } e ^ { - y / \ell + ( s - 1 ) y / r } d y \quad \mathrm { ( v i a ~ ( 7 1 ) ) } } \\ & { \quad = 2 s \tau r ^ { s - 1 } \displaystyle \int _ { 0 } ^ { \infty } e ^ { - \delta _ { s } y } d y , } \end{array}\tag{74}
$$

if we define $\delta _ { s }$ as

$$
\delta _ { s } = \frac { 1 } { \ell } - \frac { s - 1 } { r } .
$$

This will be an important quantity that controls the exponent. Using $s \leq k$ and (72), we may lower bound $\delta _ { s }$ by

$$
\delta _ { s } \geq \frac { 1 } { 4 L _ { \tau } \ell } .
$$

Similarly, using $s - 1 \geq k - 3$ and (72) yields

$$
\begin{array} { r l } & { \delta _ { s } \leq \displaystyle \frac { 1 } { 3 L _ { \tau } \ell } + \frac { 3 } { r } } \\ & { \quad \leq \displaystyle \frac { 1 } { 3 L _ { \tau } \ell } + \frac { 1 } { 8 L _ { \tau } \ell } \quad ( \mathrm { v i a } h > 2 4 L _ { \tau } \mathrm { ~ a n d ~ } 3 / r = 3 / ( h \ell ) \leq 1 / ( 8 L _ { \tau } \ell ) ) } \\ & { \quad \leq \displaystyle \frac { 1 } { 2 L _ { \tau } \ell } . } \end{array}\tag{75}
$$

Together these imply

$$
\frac { 1 } { 4 L _ { \tau } \ell } \leq \delta _ { s } \leq \frac { 1 } { 2 L _ { \tau } \ell } .\tag{76}
$$

Revisiting our bound on $I _ { + }$ with (74) and (76),

$$
\begin{array} { r l r } {  { I _ { + } \le 2 s \tau r ^ { s - 1 } \int _ { 0 } ^ { \infty } e ^ { - \delta _ { s } y } d y } } \\ & { } & \\ & { } & { = 2 s \tau r ^ { s - 1 } \frac { 1 } { \delta _ { s } } } \\ & { } & \\ & { } & { \le 8 s L _ { \tau } \ell \tau r ^ { s - 1 } } \\ & { } & { \le 8 L _ { \tau } \tau r ^ { s } . ~ \mathrm { \ ( v i a ~ } s \le k \le h = r / \ell ) } \end{array}
$$

We now turn towards bounding I<sub>−</sub>. Since

$$
( r - y ) ^ { s - 1 } = r ^ { s - 1 } \left( 1 - { \frac { y } { r } } \right) ^ { s - 1 } \leq r ^ { s - 1 } e ^ { - ( s - 1 ) y / r } ,
$$

using (71) yields

$$
I _ { - } \leq s r ^ { s - 1 } \int _ { 0 } ^ { r } e ^ { - ( s - 1 ) y / r } \operatorname* { m i n } \{ 1 , 2 \tau e ^ { y / \ell } \} d y .
$$

We will split this bound into $I _ { - } \leq I _ { - , 1 } + I _ { - , 2 }$ where

$$
I _ { - , 1 } = 2 s \tau r ^ { s - 1 } \int _ { 0 } ^ { y _ { 0 } } e ^ { \delta _ { s } y } d y , \qquad I _ { - , 2 } = s r ^ { s - 1 } \int _ { y _ { 0 } } ^ { r } e ^ { - ( s - 1 ) y / r } d y , \qquad y _ { 0 } = \ell \log \frac { 1 } { 2 \tau } .
$$

Observe that $0 \leq y _ { 0 } < r$ because $\tau \leq 1 / 2 , \log ( 1 / ( 2 \tau ) ) \leq L _ { \tau }$ , and $h > 2 4 L _ { \tau }$

For $I _ { - , 1 }$ , we bound

$$
\begin{array} { r l r } {  { I _ { - , 1 } = 2 s \tau r ^ { s - 1 } \frac { e ^ { \delta _ { s } y _ { 0 } } - 1 } { \delta _ { s } } } } \\ & { \leq 2 s \tau r ^ { s - 1 } \frac { e ^ { 1 / 2 } - 1 } { \delta _ { s } } } & { \bigg ( \mathrm { v i a ~ } ( 7 6 ) \mathrm { ~ a n d ~ } \delta _ { s } y _ { 0 } \leq \frac { 1 } { 2 L _ { \tau } \ell } \ell \log \frac { 1 } { 2 \tau } \leq \frac { 1 } { 2 } \bigg ) } \\ & { \lesssim s L _ { \tau } \ell \tau r ^ { s - 1 } \lesssim L _ { \tau } \tau r ^ { s } . } & { ( \mathrm { a g a i n ~ v i a ~ } s \leq k \leq h = r / \ell ) } \end{array}
$$

For $I _ { - , 2 }$ , we bound

$$
\begin{array} { r l } & { I _ { - , 2 } = \displaystyle \frac { s } { s - 1 } r ^ { s } \left( e ^ { - ( s - 1 ) y _ { 0 } / r } - e ^ { - ( s - 1 ) } \right) } \\ & { \mathrm { ~ \ ~ \ } \leq 2 r ^ { s } e ^ { - ( s - 1 ) y _ { 0 } / r } \left( \mathrm { v i a \ } s \geq h \left( 1 - \displaystyle \frac 1 { 3 L _ { \tau } } \right) - 2 > 2 4 L _ { \tau } - 1 0 > 2 \right) } \\ & { \mathrm { \ ~ \ } = 2 r ^ { s } ( 2 \tau ) ^ { ( s - 1 ) \ell / r } } \\ & { \mathrm { \ ~ \ } \leq 2 r ^ { s } ( 2 \tau ) ^ { 1 - 1 / ( 2 L _ { \tau } ) } \left( \mathrm { v i a \ } \displaystyle \frac { ( s - 1 ) \ell } { r } \geq 1 - \displaystyle \frac 1 { 2 L _ { \tau } } \mathrm { \ b y \ t h e \ s a m e \ s t e p s \ a s \ } ( 7 5 ) \right) } \\ & { \mathrm { \ ~ \ ~ \ } = 4 \tau r ^ { s } e ^ { \log ( 1 / ( 2 \tau ) ) / ( 2 L _ { \tau } ) } \leq 4 \tau r ^ { s } e ^ { 1 / 2 } \lesssim \tau r ^ { s } . } \end{array}
$$

Combining the bounds for $I _ { + } , I _ { - , 1 } , I _ { - , 2 }$ proves our desired bound (73) for $M _ { s } ;$ this corresponds to the quantities in the numerators of $V _ { p } ( q )$ and $Q _ { p } ( q )$ . For the moments in the denominators, we may use the much simpler bound

$$
M _ { j } = \mathrm { E } R ^ { j } \geq r ^ { j } \mathbb { P } ( R \geq r ) = 2 \tau r ^ { j } .\tag{77}
$$

We may now bound $V _ { p } ( q )$ using (73) and (77)

$$
\begin{array} { r l } & { V _ { p } ( q ) = \frac { M _ { k } } { \left( k / 2 \right) ^ { 2 } M _ { k / 2 - 1 } ^ { 2 } } } \\ & { \qquad \lesssim \frac { L _ { \tau } \tau r ^ { k } } { \left( k ^ { 2 } / 4 \right) \left( 4 \tau ^ { 2 } r ^ { k - 2 } \right) } \lesssim \frac { L _ { \tau } r ^ { 2 } } { k ^ { 2 } \tau } } \\ & { \qquad \lesssim \frac { L _ { \tau } r ^ { 2 } } { h ^ { 2 } \tau } = \frac { L _ { \tau } \ell ^ { 2 } } { \tau } \quad \mathrm { ( v i a ~ } k \geq ( 2 / 3 ) h \mathrm { ~ b y ~ ( 7 2 ) ) } } \\ & { \qquad \lesssim \frac { L _ { \tau } \ell ^ { 2 } L _ { \eta } } \eta \lesssim \frac { t ^ { 2 } L _ { \eta } ^ { 2 } } { \eta } . \quad \mathrm { ( v i a ~ ( 6 8 ) ) } } \end{array}
$$

For $Q _ { p } ( q )$ we similarly bound

$$
\begin{array} { c } { { \displaystyle Q _ { p } ( q ) = \frac { M _ { k - 2 } } { M _ { k / 2 - 1 } ^ { 2 } } \lesssim \frac { L _ { \tau } \tau r ^ { k - 2 } } { 4 \tau ^ { 2 } r ^ { k - 2 } } \lesssim \frac { L _ { \tau } } { \tau } } } \\ { { \displaystyle ~ \lesssim \frac { L _ { \tau } L _ { \eta } } { \eta } \lesssim \frac { L _ { \eta } ^ { 2 } } { \eta } ~ ( \mathrm { v i a } ~ ( 6 7 ) ) . } } \end{array}
$$

Finally, we upper bound the value of q by

$$
\begin{array} { l } { \displaystyle { q = \frac { k + 2 } { 2 } \leq 1 + \frac { h } { 2 } \quad \mathrm { ( v i a ~ ( 7 2 ) ) } } } \\ { \displaystyle { \quad \leq 1 + \frac { 1 } { 4 \tau } \left( \mathrm { v i a } \ h = \frac { r } { \ell } = \frac { r p ( r ) } { \tau } \leq \frac { \int _ { 0 } ^ { r } p ( x ) d x } { \tau } \leq \frac { 1 } { 2 \tau } \right) } } \\ { \displaystyle { \quad \leq 2 + \frac { 9 0 0 L _ { \eta } } { \eta } . \quad \mathrm { ( v i a ~ ( 6 7 ) ) } \quad \mathrm { ~ } } \displaystyle { \stackrel { \prod } { = } } } \end{array}
$$

We further use that there exists a choice of $q$ that is a power of $2 \colon$

Corollary 3.10. There is a choice of q that is a power of 2 and satisfies the conditions of Lemma 3.9. Proof. We will show that for any

$$
2 \leq s \leq q < 2 s ,
$$

it holds that

$$
V _ { p } ( s ) \leq 9 V _ { p } ( q ) ,\tag{78}
$$

and

$$
\begin{array} { r } { Q _ { p } ( s ) \leq Q _ { p } ( q ) . } \end{array}\tag{79}
$$

Together, these would immediately imply our desired result: consider the $q$ given by Lemma 3.9, and choose this value rounded down to the nearest power of 2.

It will be helpful to consider the function

$$
\phi ( a ) : = \log M _ { a } .
$$

$\phi$ is convex since $M _ { a }$ is log-convex. To prove the latter, we use that for $a , b \geq 0$ and $\theta \in [ 0 , 1 ]$ 2 Hölder’s inequality gives

$$
M _ { \theta a + ( 1 - \theta ) b } = \operatorname { E } \left[ ( R ^ { a } ) ^ { \theta } ( R ^ { b } ) ^ { 1 - \theta } \right] \leq ( \operatorname { E } R ^ { a } ) ^ { \theta } ( \operatorname { E } R ^ { b } ) ^ { 1 - \theta } = M _ { a } ^ { \theta } M _ { b } ^ { 1 - \theta } .
$$

The convexity of $\phi$ will almost immediately imply our desired result. For any $c \geq 0$ , consider

$$
F _ { c } ( x ) = \phi ( 2 x + c ) - 2 \phi ( x ) , \qquad x \geq 0 .
$$

$F _ { c }$ is non-decreasing since $\phi$ has non-decreasing derivative

$$
F _ { c } ^ { \prime } ( x ) = 2 \phi ^ { \prime } ( 2 x + c ) - 2 \phi ^ { \prime } ( x ) \geq 0 .
$$

In these terms,

$$
\log Q _ { p } ( a ) = F _ { 0 } ( a - 2 ) ,
$$

and hence $Q _ { p } ( a )$ is non-decreasing for $a \geq 2$ , which implies (79). We may similarly bound $V _ { p } ( s )$ by

$$
V _ { p } ( s ) = { \frac { e ^ { F _ { 2 } ( s - 2 ) } } { ( s - 1 ) ^ { 2 } } } \leq { \frac { e ^ { F _ { 2 } ( q - 2 ) } } { ( s - 1 ) ^ { 2 } } } = \left( { \frac { q - 1 } { s - 1 } } \right) ^ { 2 } V _ { p } ( q ) \leq \left( { \frac { 2 s - 1 } { s - 1 } } \right) ^ { 2 } V _ { p } ( q ) \leq 9 V _ { p } ( q ) .
$$

Choosing the estimate and concluding the analysis. Together, our bound for regression error in terms of moments (Proposition 3.5), our moment control (Corollary 3.10), and the polynomial-time regression error (Corollary 3.6), imply the following guarantee:

Corollary 3.11. For $t > 0$ , let $\eta = H _ { p } ( t )$ , and assume $\begin{array} { r } { \eta \ge \frac { 1 } { m } } \end{array}$ . There exists a constant $C _ { K , \kappa } \geq 1$ such that $i f$

$$
m \geq C _ { K , \kappa } \frac { d L _ { \eta } ^ { 2 } \log ( 2 n ) } { \eta } ,
$$

then there exists a $q ^ { * } \in \mathcal { G } _ { m }$ such that for each $i \in \{ 1 , \ldots , B \}$ , it holds with probability at least $\frac { 7 } { 8 }$ that

$$
\begin{array} { r } { \| A _ { q ^ { * } , i } - \beta \| _ { 2 } \leq t \qquad \textstyle a n d \qquad \left\| \widehat { A } _ { q ^ { * } , i } - A _ { q ^ { * } , i } \right\| _ { 2 } \leq \gamma . } \end{array}
$$

Proof. Corollary 3.10 yields a power of two $q ^ { * } \leq 2 + 9 0 0 L _ { \eta } / \eta$ . For a large enough choice of $C _ { K , \kappa }$ it holds that $q ^ { * } \leq m$ and hence $q ^ { * } \in \mathcal { G } _ { m }$ . Further, $q ^ { * }$ satisfies

$$
V _ { p } ( q ^ { * } ) \lesssim \frac { t ^ { 2 } L _ { \eta } ^ { 3 } } { \eta } , \qquad Q _ { p } ( q ^ { * } ) \lesssim \frac { L _ { \eta } ^ { 2 } } { \eta } .
$$

This satisfies (61) of Proposition 3.5 when our $C _ { K , \kappa }$ is suficiently large. This yields

$$
\bigl \| A _ { q ^ { * } , i } - \beta \bigr \| _ { 2 } \leq C _ { K , \kappa } \sqrt { \frac { d V _ { p } ( q ^ { * } ) } { m } } \lesssim C _ { K , \kappa } t \sqrt { \frac { d L _ { \eta } ^ { 3 } / \eta } { m } } \leq t
$$

with probability at least $1 7 / 1 8$ , when our choice of $C _ { K , \kappa }$ is large enough. Corollary 3.6 yields the final guarantee $\left\| \widehat { A } _ { q ^ { * } , i } - A _ { q ^ { * } , i } \right\| _ { 2 } \leq \gamma$ □

Consider two parameters

$$
\eta _ { 0 } = C _ { K , \kappa } \frac { d \log ^ { 4 } ( 6 4 n / \delta ) } { n } , \qquad t _ { 0 } = \omega _ { p } ( \eta _ { 0 } ) + \gamma .
$$

If $\eta _ { 0 } \geq 1$ , then our desired theorem is always vacuously true; we otherwise focus on $\eta _ { 0 } < 1$ . By Lemma 3.13, $H _ { p } ( t )$ is non-decreasing in [0, ∞). Hence, $H _ { p } ( t _ { 0 } ) > \eta _ { 0 }$ . We may invoke Corollary 3.11 with $t _ { 0 }$ since the condition on m holds when

$$
m \ge \frac { n } { C _ { K , \kappa } \log ( 6 4 n / \delta ) } ,
$$

where the constant in the denominator grows with the constant in the choice of $\eta _ { 0 }$ , and hence the condition holds from our choice of B when we last set the constant in $\eta _ { 0 }$ large enough.

For a large enough choice of constant in our choice of B, we may invoke Corollary 3.11 and a Chernof bound to demonstrate that, with probability at least $1 - \delta / 1 0$ , more than $3 / 4$ of $A _ { q ^ { * } , 1 } , \dotsc , A _ { q ^ { * } , B }$ will be within $t _ { 0 }$ of $\beta$ and have $\left\| A _ { q ^ { * } , i } - \widehat { A } _ { q ^ { * } , i } \right\| _ { 2 } \leq \gamma$ . For any such $i ,$ we observe

$$
r _ { q ^ { * } , i } \leq 2 t _ { 0 } + 2 \gamma .
$$

All that remains is to show that all balls nearly contain $\beta ,$ meaning

$$
\beta \in B ( { \widehat { A } } _ { q , i } , r _ { q , i } + \gamma ) , \qquad { \mathrm { f o r ~ a l l ~ } } q \in { \mathcal { G } } _ { m } { \mathrm { ~ a n d ~ } } i \in \{ 1 , \ldots , B \} .\tag{80}
$$

This would immediately imply

$$
\left\| \widehat { \beta } - \beta \right\| _ { 2 } \leq \operatorname* { m i n } _ { \substack { q \in \mathcal { G } _ { m } , i \in \{ 1 , \ldots , B \} } } r _ { q , i } + \gamma \leq 2 t _ { 0 } + 3 \gamma .\tag{81}
$$

We finally turn towards proving the ball containment claim (80).

Proof of (80). We use the fact

Claim 3.12. Let P be a distribution that is centrally symmetric around $\beta$ . If

$$
P ( B ( c , r ) ) > 1 / 2 ,
$$

then $\| c - \beta \| _ { 2 } \leq r$

Proof. For sake of contradiction, suppose this were not true. Then, $\boldsymbol { B } ( \boldsymbol { c } , \boldsymbol { r } )$ and $B ( 2 \beta - c , r )$ are disjoint and have the same probability under $P ;$ yet this would imply $P ( B ( c , r ) ) + P ( B ( 2 \beta - c , r ) ) > 1$ which is a contradiction. □

Claim 3.12 will be helpful because the distribution of the exact $L _ { q }$ optimizer is centrally symmetric around $\beta$ for any fixed X where $\operatorname { r a n k } ( \mathbf { X } ) = d$ (so the estimator is unique). This follows from the fact that if v is the unique minimizer for a fixed $\beta ,$ the design matrix $\mathbf { X } .$ and the collection of errors ε, then $2 \beta - v$ is the unique estimator for $\beta , \mathbf { X } , - \varepsilon$ , and by independence and symmetry of $\varepsilon ,$ both are equally likely.

Consider some function $f _ { q } ( X _ { 1 } , \ldots , X _ { m } , Y _ { 1 } , \ldots , Y _ { m } )$ which always outputs an exact optimizer of the $L _ { q }$ objective on m samples (it may output any optimizer if there are multiple). Let $P _ { \mathrm { a l l } }$ denote the distribution of $f _ { q }$ when the m samples $( X _ { i } , Y _ { i } )$ are sampled i.i.d., and let $P _ { \mathrm { g o o d } }$ denote the distribution of $f _ { q }$ conditioned on the minimum singular value condition of (64) holding (this also implies rank $\mathbf { \partial } ( \mathbf { X } ) = d ,$ , so the optimizer is unique). We have already shown that $P _ { \mathrm { g o o d } }$ is centrally symmetric around $\beta .$ Moreover, by Corollary 3.6, the singular value condition holds with probability at least $1 7 / 1 8$ , and hence we may write

$$
P _ { \mathrm { a l l } } = \alpha P _ { \mathrm { g o o d } } + ( 1 - \alpha ) Q ,
$$

for some distribution $Q$ and $\alpha \in [ 1 7 / 1 8 , 1 ]$ . Similarly, let $\widehat { P } _ { \mathrm { a l l } }$ denote the distribution of our algorithm’s polynomial time estimate of the optimizer given m i.i.d. samples, and let $\widehat { P } _ { \mathrm { g o o d } }$ denote the distribution of our algorithm’s polynomial time estimate of the optimizer conditioned on the minimum singular value condition holding. Note that $\widehat { P } _ { \mathrm { g o o d } }$ need not be symmetric around $\beta .$ . We may likewise write

$$
\widehat { P } _ { \mathrm { a l l } } = \alpha \widehat { P } _ { \mathrm { g o o d } } + ( 1 - \alpha ) \widehat { Q } ,
$$

for some distribution $\widehat { Q }$ and $\alpha \in [ 1 7 / 1 8 , 1 ]$

Condition on a particular fixed candidate estimate $\widehat { A } _ { q , i }$ . For each $\widehat { A } _ { q , j } \ ( i \neq j )$ , let us denote the distance from $\widehat { A } _ { q , i }$ as $D _ { j } = \left. \widehat { A } _ { q , i } - \widehat { A } _ { q , j } \right. _ { 2 }$ . Denote the CDF of this corresponding random variable as $F _ { ; }$ , and the empirical CDF from $D _ { 1 } , \ldots , D _ { B }$ as $F _ { B }$ . By DKW inequality [Mas90], we know

$$
\mathbb { P } ( \operatorname* { s u p } _ { x } | F _ { B } ( x ) - F ( x ) | > a ) \le 2 e ^ { - 2 ( B - 1 ) a ^ { 2 } } .
$$

Hence, using $a = 0 . 1$ , it holds that

$$
\mathbb { P } ( \widehat { P } _ { \mathrm { a l l } } ( B ( \widehat { A } _ { q , i } , r _ { q , i } ) ) < 0 . 6 5 ) \le 2 e ^ { - 0 . 0 2 ( B - 1 ) } .
$$

In total, there are at most $B \log _ { 2 } ( m )$ candidate estimates, and thus by union bound all candidate estimates satisfy

$$
\widehat { P } _ { \mathrm { a l l } } ( B ( \widehat { A } _ { q , i } , r _ { q , i } ) ) \geq 0 . 6 5
$$

with probability at least $1 - \delta / 1 0$ when the constant in $B$ is chosen large enough. Furthermore, this implies that

$$
\widehat { P } _ { \mathrm { g o o d } } ( B ( \widehat { A } _ { q , i } , r _ { q , i } ) ) > 0 . 5
$$

for all candidates, as otherwise

$$
\widehat { P } _ { \mathrm { a l l } } ( B ( \widehat { A } _ { q , i } , r _ { q , i } ) ) \leq \frac { 1 7 } { 1 8 } \cdot \frac { 1 } { 2 } + \frac { 1 } { 1 8 } < 0 . 6 5 ,
$$

which is a contradiction.

Further, $\mathrm { i f }$ we consider the relationship between $P _ { \mathrm { g o o d } }$ and $\widehat { P } _ { \mathrm { g o o d } }$ , recall that the algorithmic optimizer is always within $\gamma$ of the true optimizer under the singular value condition, and hence

$$
\begin{array} { r } { P _ { \mathrm { g o o d } } ( B ( \widehat { A } _ { q , i } , r _ { q , i } + \gamma ) ) \geq \widehat { P } _ { \mathrm { g o o d } } ( B ( \widehat { A } _ { q , i } , r _ { q , i } ) ) > 0 . 5 . } \end{array}
$$

Since $P _ { \mathrm { g o o d } }$ is centrally symmetric around $\beta ,$ Claim 3.12 implies that $B ( \widehat { A } _ { q , i } , r _ { q , i } + \gamma )$ contains $\beta$ for all candidates, concluding our proof. □

This implies our desired guarantee by (81)

$$
\left\| \widehat { \beta } - \beta \right\| _ { 2 } \leq 2 t _ { 0 } + 3 \gamma \leq C _ { K , \kappa } \omega _ { p } \left( C _ { K , \kappa } \frac { d \log ^ { 4 } ( 6 4 n / \delta ) } { n } \right) + 5 \gamma .
$$

We may use the above guarantee with $\gamma ^ { \prime } = \gamma / 5$ for the desired theorem statement.

Finally, we discuss the running time. The algorithm makes $O ( B \log ( n ) ) = O ( \log ^ { 2 } ( 2 n / \delta ) )$ calls to the optimizer of [AKPS24]. By Corollary 3.6, each call solves $O \left( q ^ { 2 } m ^ { 1 / 3 } \right.$ max $\begin{array} { r } { \{ 1 , \log ( \frac { C _ { K , \kappa } m \| \mathbf { Y } \| _ { \infty } } { \gamma } ) \} ) } \end{array}$ linear systems. Outside of this, the runtime is dominated by computing pairwise distances between candidate estimates and choosing $( \widehat { q } , \widehat { i } )$ ; this runs in $O ( \log ( n ) \cdot d B ^ { 2 } ) = O ( d \log ^ { 3 } ( n / \delta ) )$ time. In total, the algorithm runs in poly $( n , d )$ $\cdot \ \mathrm { p o l y l o g } ( 1 / \delta , \| \mathbf { Y } \| _ { \infty } , 1 / \gamma )$ time. We refer to [AKPS24] for a more fine-grained description of optimization subroutine’s runtime.

## 3.3 Statistical lower bound

In this section, we will provide a statistical lower bound for adaptive linear regression in terms of the Hellinger modulus. Our lower bound (Theorem 1.8) will focus on the setting where $X _ { i } \sim N ( 0 , I _ { d } )$ $\varepsilon _ { i } \sim p .$ , and $p$ is symmetric unimodal.

Further proof intuition. Please recall the proof intuition discussed in Section 1.2. In this section, we will denote our linear regression parameter as Θ, where it is drawn from a prior.

Recall the original technical obstacle of bounding $\operatorname { E } _ { \Theta , X _ { i } } [ \mathrm { K L } ( p _ { X _ { i } ^ { \top } \Theta } , p ) ]$ , and recall the KL decomposition observation discussed in Section 1.2, where $P$ is decomposed into two components $P = ( 1 - \alpha ) G + \alpha R$ , where $\alpha \lesssim \mathrm { H } ^ { 2 } ( P , Q )$ and $\mathrm { K L } ( G , Q ) \lesssim \mathrm { H } ^ { 2 } ( P , Q )$ . A more typical Fano approach will eventually work well for the samples corresponding to G in our decomposition, yet samples corresponding to R may be more unwieldy. In these cases, our lower bound will consider a more extreme version where instead of observing just $Y _ { i } = R _ { X _ { i } ^ { \top } \Theta }$ , we reveal directly the value of $X _ { i } ^ { \top } \Theta$ which is strictly more informative. The key point is that we will tune parameters such that there are typically, say, $\leq d / 2$ samples corresponding to $R ,$ and hence Θ is still uncertain over a subspace of dimension $\geq d / 2$ . Extra care is required for a rigorous proof. A notable consideration is the revealed subspace is dependent with $\Theta$ , since samples from $R$ are more likely when $\mathrm { H } ^ { 2 } ( p _ { X _ { i } ^ { \top } \Theta } , p )$ is larger, and hence the revealed subspace might correlate with directions where Θ is large; this will be handled more clearly later in the full proof.

Proof structure. In Section 3.3.1, we begin our proof with some Hellinger helper lemmas that show the Hellinger distance has controlled growth for translations of symmetric unimodal distributions, and the existence of this decomposition that is amenable to KL. In Section 3.3.2, we prove our desired lower bound, conditional on a lemma that yields a lower bound whenever the dimension of the revealed subspace is not too large, the choice of subspace is not too informative about Θ, and the other samples are not too informative about Θ restricted to the unrevealed subspace. Finally, in Section 3.3.3, we prove said lemma.

## 3.3.1 Hellinger lemmas

We first show control for Hellinger between translations:

Lemma 3.13 (Translation Hellinger growth). $I f p$ is symmetric unimodal, then $H _ { p }$ is even, nondecreasing on $[ 0 , \infty )$ , and satisfies

$$
H _ { p } ( t ) \leq \left( 1 + { \frac { | t | } { a } } \right) ^ { 2 } H _ { p } ( a ) \qquad ( a > 0 , \ t \in \mathbb { R } ) .\tag{82}
$$

Consequently, every $| r | < \omega _ { p } ( u )$ satisfies $H _ { p } ( r ) \leq u$

Proof. Let $f = { \sqrt { p } }$ and

$$
C _ { f } ( t ) = \int f ( x ) f ( x - t ) d x .
$$

Then $H _ { p } ( t ) = 1 - C _ { f } ( t )$ . Since f is even and non-increasing on $[ 0 , \infty )$ , its superlevel sets are centered intervals. $C _ { f } ( t )$ can be written as a double integral over the overlap lengths of the sets where $f ( x ) \geq a$ and $f ( x - t ) \geq b$ for $a , b \geq 0 ;$ since each of these overlap lengths are non-increasing in t for $t \geq 0$ , then $C _ { f }$ is non-increasing and $H _ { p }$ is non-decreasing for $t \geq 0$ . Further, define

$$
\rho ( t ) = \| f - f ( \cdot - t ) \| _ { 2 } = \sqrt { 2 H _ { p } ( t ) } .
$$

Translation invariance and the triangle inequality imply $\rho ( s + t ) \leq \rho ( s ) + \rho ( t )$ . Thus,

$$
\rho ( t ) \leq \lceil | t | / a \rceil \rho ( t / \lceil | t | / a \rceil ) \leq \lceil | t | / a \rceil \rho ( a ) \leq \left( 1 + { \frac { | t | } { a } } \right) \rho ( a ) ,
$$

using the monotonicity of $\rho ;$ squaring proves (82).

We next show the desirable KL decomposition:

Lemma 3.14 (Hellinger-to-KL decomposition). There are universal constants $0 < C , K < \infty$ such that, for arbitrary densities $P , Q$ , there exist densities $G , R$ and

$$
\alpha = \operatorname* { m i n } \{ 1 , K \mathrm { H } ^ { 2 } ( P , Q ) \} ,
$$

for which

$$
P = ( 1 - \alpha ) G + \alpha R , \qquad \mathrm { K L } ( G , Q ) \leq C \mathrm { H } ^ { 2 } ( P , Q ) .\tag{83}
$$

Proof. We work towards using the following convenient fact, which shows the $\operatorname { K L } ( P , Q )$ is bounded in terms of Hellinger when the ratio of $P ( x ) / Q ( x )$ is bounded (this is a well-used fact in the area, and results of this form also appear in e.g. [Bir83, Lemma 4.4], [BM98, Lemma 5], [Tan22, Fact 4]):

Lemma 3.15 (KL control from Hellinger and bounded ratio; Lemma 1 of $\mathrm { [ C L M ^ { + } 2 6 ] ) }$ . Assume that $p , q$ are two densities on $\mathcal { X }$ with respect to $\mu .$ . It holds that

$$
\mathrm { K L } ( p , q ) \leq { \frac { 2 } { ( { \sqrt { e } } - 1 ) ^ { 2 } } } ~ \mathrm { H } ^ { 2 } ( p , q ) ~ \mathrm { m a x } \left\{ 1 , ~ \operatorname* { s u p } _ { x \in { \mathcal { X } } } \log \left( { \frac { p ( x ) } { q ( x ) } } \right) \right\} .
$$

This fact illuminates that the problematic regions of P for $\operatorname { K L } ( P , Q )$ are only those where, say, $Q ( x ) < P ( x ) / 2$ . We define a set corresponding to the favorable region of the domain as

$$
A = \{ x : Q ( x ) \geq P ( x ) / 2 \} .
$$

Further, we can bound the mass outside this region by

$$
\begin{array} { l } { \displaystyle \frac 1 2 \int _ { A ^ { c } } \left( \sqrt { P ( x ) } - \sqrt { Q ( x ) } \right) ^ { 2 } \geq \frac 1 2 \int _ { A ^ { c } } P ( x ) \left( \sqrt { 1 } - \sqrt { 1 / 2 } \right) ^ { 2 } \geq 0 . 0 4 \int _ { A ^ { c } } P ( x ) } \\ { \displaystyle \implies P ( A ^ { c } ) \leq 2 5 \mathrm { H } ^ { 2 } ( P , Q ) . } \end{array}\tag{84}
$$

We will choose $K = 5 0$ , and hence we may focus on $P , Q$ where $\begin{array} { r } { \mathrm { H } ^ { 2 } ( P , Q ) \le \frac { 1 } { 5 0 } } \end{array}$ and $\begin{array} { r } { P ( A ^ { c } ) \leq \frac { 1 } { 2 } } \end{array}$ (When $\begin{array} { r } { \mathrm { H } ^ { 2 } ( P , Q ) \geq \frac { 1 } { 5 0 } } \end{array}$ , set $G = Q$ and $R = P . )$ We now choose the distribution $G = ( P \mid A )$ , and the distribution of R so that P has the right marginal (this will be valid since $( 1 - \alpha ) \leq P ( A ) )$ .

We now must bound KL(G, Q). Observe how $G ( x ) = 0$ outside $A ,$ , and

$$
G ( x ) = { \frac { 1 } { 1 - P ( A ^ { c } ) } } P ( x ) = \left( 1 + { \frac { P ( A ^ { c } ) } { 1 - P ( A ^ { c } ) } } \right) P ( x ) \leq ( 1 + 2 P ( A ^ { c } ) ) P ( x ) \leq 2 P ( x ) \leq 4 Q ( x )
$$

inside A. Using this and Lemma 3.15, we finally bound

$$
\begin{array} { r l } & { \mathrm { K L } ( G , Q ) \lesssim \mathrm { H } ^ { 2 } ( G , Q ) \lesssim \mathrm { H } ^ { 2 } ( P , Q ) + \mathrm { H } ^ { 2 } ( G , P ) } \\ & { \qquad \leq \mathrm { H } ^ { 2 } ( P , Q ) + \displaystyle \int _ { A } \left( \sqrt { G ( x ) } - \sqrt { P ( x ) } \right) ^ { 2 } + \displaystyle \int _ { A ^ { c } } \left( \sqrt { G ( x ) } - \sqrt { P ( x ) } \right) ^ { 2 } } \\ & { \qquad \leq \mathrm { H } ^ { 2 } ( P , Q ) + \displaystyle \int _ { A } P ( x ) \left( \sqrt { 1 + 2 P ( A ^ { c } ) } - 1 \right) ^ { 2 } + \displaystyle \int _ { A ^ { c } } P ( x ) } \\ & { \qquad \lesssim \mathrm { H } ^ { 2 } ( P , Q ) + \displaystyle \left( \int _ { A } P ( x ) P ( A ^ { c } ) ^ { 2 } \right) + P ( A ^ { c } ) \lesssim \mathrm { H } ^ { 2 } ( P , Q ) ~ \bigm \sqsupsetneqq } \end{array}
$$

## 3.3.2 Proof of Theorem 1.8

We now prove Theorem 1.8. Let

$$
\eta = c { \frac { d } { n } } ,
$$

for a $c > 0$ later chosen to be suficiently small. We will choose a lower bound incurring error relative to a benchmark

$$
r = \omega _ { p } ( \eta ) / 2
$$

(where the division by 2 is an arbitrary constant, but it implies $H _ { p } ( r ) \leq \eta )$ , and choose the prior

$$
\Theta \sim N \left( 0 , \frac { r ^ { 2 } } { d } I _ { d } \right) .\tag{85}
$$

For an independent $X \sim N ( 0 , I _ { d } )$ , let $T = X ^ { \top } \Theta$ be the exact inner product before adding any error distribution. For any realization t, we will compute the decomposition given by Lemma 3.14 with $P = p _ { t } , Q = p _ { \mathrm { ; } }$ , and $\alpha ( t ) = \operatorname* { m i n } \{ 1 , K \mathrm { H } ^ { 2 } ( p , p _ { t } ) \}$ . In terms of this decomposition, the i-th observation can be generated by drawing

$$
B _ { i } \mid ( \Theta , X _ { i } ) \sim \mathrm { B e r n } ( \alpha ( T _ { i } ) ) , \qquad T _ { i } = X _ { i } ^ { \top } \Theta ,
$$

and then drawing $Y _ { i }$ from $G _ { T _ { i } }$ if $B _ { i } = 0$ , or from $R _ { T _ { i } } { \mathrm { ~ i f ~ } } B _ { i } = 1$

We will modify this setup, by revealing even more information

$$
X ^ { n } , \qquad B = ( B _ { 1 } , \dots , B _ { n } ) , \qquad T _ { B } = ( T _ { i } : B _ { i } = 1 ) , \qquad Y ^ { n } .
$$

This is strictly more informative than the original observations, as it also tells you the exact inner product $T _ { i }$ whenever $B _ { i } = 1$ . Given $\left( X ^ { n } , B , T _ { B } \right)$ , the outputs $Y _ { i }$ with $B _ { i } = 1$ have known laws $R _ { T _ { i } }$ independent of the remaining uncertainty about $\Theta ;$ hence, these values of $Y _ { i }$ may be deleted. We define the modified version of $Y ^ { n }$ as $W = ( W _ { 1 } , \dots , W _ { n } )$ , where

$$
W _ { i } = \left\{ \begin{array} { l l } { Y _ { i } , } & { B _ { i } = 0 , } \\ { * , } & { B _ { i } = 1 . } \end{array} \right.
$$

It sufices to provide a lower bound for estimators that observe

$$
( X ^ { n } , B , T _ { B } , W ) ,
$$

where $T _ { B }$ denotes the collection of exact linear equations $X _ { i } ^ { \top } \Theta = T _ { i }$ for $B _ { i } = 1$ . In the remaining proof, we will show that $T _ { B }$ usually does not contain too many entries, B is not too informative about Θ conditioned on $X ^ { n }$ , and W is not too informative about Θ conditioned on $X ^ { n } , B , T _ { B }$ Together, these conditions will imply a lower bound by invoking the following (proof deferred to Section 3.3.3):

Lemma 3.16. Let $\Theta \sim N ( 0 , \sigma ^ { 2 } I _ { d } )$ , and let X be side information independent of Θ. Conditional on (X, Θ), let J be a random subset of a finite set. For each $( x , j )$ , let $A _ { j } ( x )$ be a known matrix with |j| rows, and reveal $U _ { j } = A _ { j } ( X ) \Theta$ . Let W be any additional observation. Suppose

$$
\mathbb { P } ( | J | > d / 2 ) \leq \frac { 1 } { 8 }\tag{86}
$$

and

$$
2 I ( \Theta ; J \mid X ) + I ( \Theta ; W \mid X , J , U _ { J } ) \le d / 2 .\tag{87}
$$

Then, every estimator based on $( X , J , U _ { J } , W )$ satisfies

$$
\mathbb { P } \left( \left\| \widehat { \Theta } - \Theta \right\| _ { 2 } \geq e ^ { - 8 } \sigma \sqrt { d / 2 } \right) \geq \frac { 3 } { 8 } .\tag{88}
$$

We remark that the constants in the statement of Lemma 3.16 are somewhat arbitrary. We invoke Lemma 3.16 with $X = X ^ { n } , J = \{ i : B _ { i } = 1 \}$ , and $A _ { J } ( X )$ equal to the matrix of selected rows of $X ^ { n }$ ; in the remaining proof we verify the lemma’s conditions.

Few exact measurements. We must prove $\begin{array} { r } { \mathbb { P } ( | B | > d / 2 ) \le \frac { 1 } { 8 } } \end{array}$ . Since the Bernoulli parameters for $B _ { i }$ are $\alpha ( T _ { i } ) = \operatorname* { m i n } \{ 1 , K \mathrm { H } ^ { 2 } ( p , p _ { T _ { i } } ) \}$ , we naturally want to bound $\operatorname { E } _ { \Theta , X } H _ { p } ( T )$ . Using $\mathrm { E } _ { \Theta , X } T ^ { 2 } = r ^ { 2 }$ $\operatorname { E } _ { \Theta , X } | T | \leq r .$ , and Lemma 3.13 yields

$$
\mathrm { E } _ { \Theta , X } H _ { p } ( T ) \leq H _ { p } ( r ) \mathrm { E } _ { \Theta , X } \left( 1 + \frac { | T | } { r } \right) ^ { 2 } \leq 4 H _ { p } ( r ) .\tag{89}
$$

This directly implies

$$
\mathrm { E } _ { \Theta , X ^ { n } , B } | B | = n \mathrm { E } _ { \Theta , X } \alpha ( T ) \leq K n \mathrm { E } _ { \Theta , X } H _ { p } ( T ) \leq 4 K n H _ { p } ( r ) \leq 4 K c d .\tag{90}
$$

Our desired condition holds by Markov’s inequality when we choose a suficiently small $c .$

Limited information from B. We must bound $I ( \Theta ; B \mid X ^ { n } )$ . Condition on $X _ { i } = x$ . Then, $T _ { i } = x ^ { \top } \Theta$ is Gaussian with variance $\| x \| _ { 2 } ^ { 2 } r ^ { 2 } / d ,$ and $B _ { i }$ depends on $\Theta$ only through $T _ { i }$ . This means

$$
I ( \Theta ; B _ { i } \mid X _ { i } = x ) = I ( T _ { i } ; B _ { i } \mid X _ { i } = x )\tag{91}
$$

We will bound this quantity using the following result:

Lemma 3.17 (Information in a Gaussian-shift label). Let $\alpha : \mathbb { R }  [ 0 , 1 ]$ be even, nondecreasing in $| t | _ { i }$ , and satisfy

$$
\alpha ( s z ) \leq ( 1 + | z | ) ^ { 2 } \alpha ( s ) \qquad ( s \geq 0 , z \in \mathbb { R } ) .\tag{92}
$$

Let $Z \sim N ( 0 , 1 )$ and, conditionally on $Z$ , let

$$
B \sim \operatorname { B e r n } ( \alpha ( s Z ) ) .
$$

Then,

$$
I ( Z ; B ) \le C \mathrm { E } _ { Z } \alpha ( s Z )\tag{93}
$$

for a universal constant $C > 0$

Proof. Let $A = \alpha ( s Z )$ and $\mu = \operatorname { E } _ { Z } A$ . Monotonicity implies

$$
\mu \geq \mathbb { P } ( | Z | \geq 1 ) \alpha ( s ) \gtrsim \alpha ( s ) ,
$$

and the (92) therefore gives

$$
A \leq C \mu ( 1 + | Z | ) ^ { 2 } .\tag{94}
$$

Since the marginal of B is Bern $( \mu )$ , the mutual information is

$$
I ( Z ; B ) = \operatorname { E } _ { Z } \operatorname { K L } ( \operatorname { B e r n } ( A ) , \operatorname { B e r n } ( \mu ) ) .
$$

If $\mu \geq 1 / 2$ , then

$$
I ( Z ; B ) \leq H ( B ) \leq \log 2 \leq 2 \mu \log 2 .
$$

Otherwise, if $\mu < 1 / 2$ , we use that KL is bounded by chi-square divergence, and (94):

$$
I ( Z ; B ) \leq \frac { \mathrm { E } _ { Z } ( A - \mu ) ^ { 2 } } { \mu } + \frac { \mathrm { E } _ { Z } ( A - \mu ) ^ { 2 } } { 1 - \mu } \leq \frac { 2 \mathrm { E } _ { Z } ( A - \mu ) ^ { 2 } } { \mu } \leq C \mu \mathrm { E } _ { Z } ( 1 + | Z | ) ^ { 4 } \lesssim \mu .
$$

We bound the desired quantity by

$$
\begin{array} { l } { { \displaystyle I ( \Theta ; B \mid X ^ { n } ) = H ( B \mid X ^ { n } ) - H ( B \mid \Theta , X ^ { n } ) \leq \sum _ { i = 1 } ^ { n } H ( B _ { i } \mid X ^ { n } ) - \sum _ { i = 1 } ^ { n } H ( B _ { i } \mid \Theta , X ^ { n } ) } } \\ { { \displaystyle ~ = \sum _ { i = 1 } ^ { n } \left( H ( B _ { i } \mid X _ { i } ) - H ( B _ { i } \mid X _ { i } , \Theta ) \right) = \sum _ { i = 1 } ^ { n } I ( \Theta ; B _ { i } \mid X _ { i } ) } } \\ { { \displaystyle ~ = \sum _ { i = 1 } ^ { n } I ( T _ { i } ; B _ { i } \mid X _ { i } ) } } \\ { { \displaystyle ~ \leq C n \mathrm { E } _ { \Theta , X ^ { \alpha } } ( T ) } } \\ { { \displaystyle ~ \leq C n H _ { p } ( r ) \leq C c d , } } \end{array}\tag{via (91)}
$$

(via Lemma 3.17)

(95)

which satisfies our desired guarantees when c is chosen suficiently small.

Limited information from $W$ . Conditional on $( \Theta , X ^ { n } , B , T _ { B } )$ , the $W _ { i }$ where $B _ { i } = 0$ are independent with density $G _ { T _ { i } }$ . Compare them with a product reference having density $p$ on every $B _ { i } = 0$ coordinate and the ∗ symbol on every $B _ { i } = 1$ coordinate. This yields the bound

$$
\begin{array} { r l } { I ( \Theta ; W \mid X ^ { n } , B , T _ { B } ) \le \mathrm { { E } } _ { \Theta , X ^ { n } , B } \displaystyle \sum _ { i : B _ { i } = 0 } \mathrm { { K L } } ( G _ { T _ { i } } , p ) } & { } \\ { \le C n \mathrm { { E } } _ { \Theta , X } H _ { p } ( T ) \le C n H _ { p } ( r ) \le C c d . } \end{array}\tag{96}
$$

Concluding the lower bound. Together (90), (95) (96) imply the required conditions when c is chosen small enough. This implies that with probability at least $3 / 8$ any estimator incurs error

$$
c \omega _ { p } ( \eta ) = c \omega _ { p } \left( c \frac { d } { n } \right) .
$$

## 3.3.3 Proof of Lemma 3.16

Proof intuition. After $J , U _ { J }$ are revealed, the only remaining uncertainty in $\Theta$ is in the orthogonal complement of row $( A _ { j } ( x ) )$ . In the extreme case where $J$ was chosen independently from Θ, and there is no additional observation $W$ , then $\Theta$ on this remaining subspace would be distributed like $N ( 0 , \sigma ^ { 2 } I _ { d - | j | } )$ , and it would be impossible to estimate $\Theta$ with error much better than $\sigma \sqrt { d - | j | }$ However, in our setting J is not chosen independently of $\Theta ,$ , and there is an additional observation $W$ , but we assume a bounded mutual information condition for $J$ and $W$ with Θ. Let $P$ be the actual joint distribution over $( X , J , U _ { J } , W , \Theta )$ , and let $Q$ denote the same distribution over the observable variables, except the distribution of $\Theta$ on the unmeasured subspace is replaced with $N ( 0 , \sigma ^ { 2 } I _ { d - | j | } )$ We work to bound $\mathrm { K L } ( P , Q )$ , and then since estimation is hard under $Q ,$ , we will show hardness for estimation under $P .$

Proof. Fix a realization $( x , j )$ . Let

$$
\mathcal { R } _ { x , j } = \mathrm { r o w } ( A _ { j } ( x ) ) , \qquad \mathcal { N } _ { x , j } = \mathcal { R } _ { x , j } ^ { \perp } ,
$$

We split Θ into two components, $\Theta = U + V$ , where

$$
U = \Pi _ { \mathcal { R } _ { x , j } } \Theta , \qquad V = \Pi _ { \mathcal { N } _ { x , j } } \Theta .
$$

Given $( x , j )$ , the measurement $U _ { j } = A _ { j } ( x ) \Theta$ determines $U$ (and vice versa). Thus, an estimator based on $( X , J , U _ { J } , W )$ has exactly the same information as one based on $( X , J , U , W )$ . For a fixed $( x , j )$ , we denote the related Gaussian distributions over the corresponding subspaces as

$$
\Gamma _ { \mathcal { R } _ { x , j } } = N ( 0 , \sigma ^ { 2 } I _ { \mathcal { R } _ { x , j } } ) , \qquad \Gamma _ { \mathcal { N } _ { x , j } } = N ( 0 , \sigma ^ { 2 } I _ { \mathcal { N } _ { x , j } } ) .
$$

Let $P$ be the joint distribution over $( X , J , U , V , W )$ . We define the distribution Q by the process:

(i) draw $( X , J , U , W )$ from its marginal under $P ;$

(ii) conditional on $( X , J , U , W ) = ( x , j , u , w )$ , draw $V \sim \Gamma _ {  { \mathcal { N } } _ { x , \mathcal { I } } }$ independently of $( u , w )$

Observe how P and $Q$ have exactly the same marginal for everything observable to the estimator $( X , J , U , W )$ ; they difer only in the conditional distribution of the hidden component V . Our proof will leverage that estimation of $\Theta = U + V$ under $Q$ is hard because there is no observable information about $V ,$ yet $V$ is still distributed in a well-spread manner under $Q .$ By showing $\operatorname { K L } ( P , Q )$ is bounded, this will imply hardness for estimating Θ under $P$

Bounding $\mathrm { K L } ( P , Q )$ . For ease of notation, let the observables be represented by $O = ( X , J , U , W )$ Since $P , Q$ have the same distribution over $O ,$ it holds that

$$
\operatorname { K L } ( P , Q ) = \operatorname { E } _ { O \sim P } \operatorname { K L } ( P _ { V | O } , \Gamma _ { \mathcal { N } _ { x , j } } ) .
$$

It will be helpful to introduce an intermediary $P _ { V | X , J }$ and use

$$
\log \frac { P _ { V | X , J , U , W } } { \Gamma _ { \mathcal { N } _ { X , J } } } ( v ) = \log \frac { P _ { V | X , J , U , W } } { P _ { V | X , J } } ( v ) + \log \frac { P _ { V | X , J } } { \Gamma _ { \mathcal { N } _ { X , J } } } ( v ) .
$$

Taking expectation under $P ,$ this yields

$$
\operatorname { K L } ( P , Q ) = I ( V ; U , W \mid X , J ) + \operatorname { E } _ { ( X , J ) \sim P } \operatorname { K L } ( P _ { V \mid X , J } , \Gamma _ { \mathcal { N } _ { X , J } } ) .\tag{97}
$$

The rest of the proof will bound this quantity. It will be helpful to work with $I ( \Theta ; J | X )$ since we assume this quantity is small in (87). We use

$$
I ( \Theta ; J \mid X ) = \mathrm { E } _ { ( X , J ) \sim P } \mathrm { K L } ( P _ { \Theta | X , J } , N ( 0 , \sigma ^ { 2 } I _ { d } ) ) = \mathrm { E } _ { ( X , J ) \sim P } \mathrm { K L } ( P _ { U , V | X , J } , \Gamma _ { \mathcal { R } _ { X , J } } \otimes \Gamma _ { \mathcal { N } _ { X , J } } ) ,\tag{98}
$$

where the last step used that for any fixed $( x , j )$ , there is a bijection between Θ and $( U , V )$ . We use the fact that for any joint distribution $M _ { U , V }$ and product distribution $G _ { U } \otimes G _ { V }$ , then it holds that

$$
\mathrm { K L } ( M _ { U , V } , G _ { U } \otimes G _ { V } ) = \mathrm { K L } ( M _ { U } , G _ { U } ) + \mathrm { K L } ( M _ { V } , G _ { V } ) + I _ { M } ( U ; V ) ,
$$

where $I _ { M } ( U ; V )$ denotes the mutual information between $U , V$ when they are distributed according to $M _ { U , V }$ . Invoking this with (98) yields

$$
I ( \Theta ; J | X ) = \mathrm { E } _ { ( X , J ) \sim P } \mathrm { K L } ( P _ { U | X , J } , \Gamma _ { \mathcal { R } _ { X , J } } ) + \mathrm { E } _ { ( X , J ) \sim P } \mathrm { K L } ( P _ { V | X , J } , \Gamma _ { \mathcal { N } _ { X , J } } ) + I ( U ; V | X , J ) ,
$$

which by non-negativity of all terms implies

$$
\begin{array} { r l r } & { } & { \mathrm { E } _ { ( X , J ) \sim P } \mathrm { K L } ( P _ { V \mid X , J } , \Gamma _ { \mathcal { N } _ { X , J } } ) \leq I ( \Theta ; J \mid X ) , } \\ & { } & { I ( U ; V \mid X , J ) \leq I ( \Theta ; J \mid X ) . } \end{array}
$$

Revisiting (97) with these implications gives

$$
\begin{array} { r l } & { { \mathrm { K L } } ( P , Q ) = I ( U ; V | X , J ) + I ( V ; W | U , X , J ) + { \mathrm { E } } _ { ( X , J ) \sim P } { \mathrm { K L } } ( P _ { V | X , J } , \Gamma _ { \mathcal { N } _ { X , J } } ) } \\ & { \qquad \leq 2 I ( \Theta ; J \mid X ) + I ( V ; W | U , X , J ) . } \end{array}
$$

All that remains is to bound $I ( V ; W | U , X , J )$ . Since there is a bijection between U and $U _ { J }$ given $( X , J )$ , and a bijection between V and Θ given $( U , X , J )$ , this implies

$$
I ( V ; W | U , X , J ) = I ( \Theta ; W | X , J , U _ { J } ) ,
$$

and hence our final bound via assumption (87),

$$
\operatorname { K L } ( P , Q ) \leq 2 I ( \Theta ; J \mid X ) + I ( \Theta ; W | X , J , U _ { J } ) \leq d / 2 .\tag{99}
$$

Showing hardness of estimation under $Q .$ . Set a target accuracy parameter

$$
\rho = e ^ { - 8 } \sigma \sqrt { d / 2 } .
$$

Consider an event corresponding to when $| J |$ is not too large and the estimator $\widehat { \Theta }$ is close to $\theta ,$

$$
E = \left\{ | J | \leq d / 2 , \ \left\| { \widehat { \Theta } } - \Theta \right\| _ { 2 } \leq \rho \right\} .
$$

Fix the observed values $( X , J , U , W ) = ( x , j , u , w )$ with $| j | \leq d / 2$ , and let $q = \mathrm { d i m } (  { \mathcal { N } } _ { x , j } ) \geq d / 2$ Under $Q ,$ , conditional on the observed variables, V is independent of the estimator and is distributed according to $N ( 0 , \sigma ^ { 2 } I _ { q } )$ on $\mathcal { N } _ { x , j }$ . Observe how estimating Θ well requires estimating $V$ well, since

$$
\begin{array} { r } { \left\| \widehat { \Theta } - \Theta \right\| _ { 2 } \leq \rho \quad \Longrightarrow \quad \left\| V - \Pi _ { \mathcal { N } _ { x , j } } \widehat { \Theta } \right\| _ { 2 } \leq \rho . } \end{array}
$$

However, for any estimate ${ \widehat { \Theta } } ,$ , since $V$ is independent of the observations, we will be able to show $v _ { 0 } = \Pi _ { \mathcal { N } _ { x , j } } \widehat { \Theta }$ is often far from $V .$ . This follows from how the Gaussian density is at most $( 2 \pi \sigma ^ { 2 } ) ^ { - q / 2 }$ everywhere, so

$$
\begin{array} { r l } & { \begin{array} { r l } & { Q ( \| V - v _ { 0 } \| _ { 2 } \leq \rho \mid x , j , u , w ) \leq ( 2 \pi \sigma ^ { 2 } ) ^ { - q / 2 } \operatorname { v o l } ( B _ { 2 } ^ { q } ( \rho ) ) } \\ & { \qquad = \frac { \left( \rho / \left( \sigma \sqrt { 2 } \right) \right) ^ { q } } { \Gamma \left( q / 2 + 1 \right) } } \\ & { \qquad \leq \left( \frac { e \rho ^ { 2 } } { \sigma ^ { 2 } q } \right) ^ { q / 2 } } \\ & { \qquad < e ^ { - 1 5 ( q / 2 ) } . } \end{array} } \end{array}
$$

Here we used $\Gamma ( q / 2 + 1 ) \ge ( q / ( 2 e ) ) ^ { q / 2 }$ and $\rho ^ { 2 } = e ^ { - 1 6 } \sigma ^ { 2 } ( d / 2 ) \le e ^ { - 1 6 } \sigma ^ { 2 } q$ . Using $q \geq d / 2$ yields

$$
Q ( E ) \leq e ^ { - 1 5 ( d / 4 ) } .\tag{100}
$$

Transferring hardness from $Q$ to $P \cdot$ For ease of notation, let $\mathrm { K L } ( a , b )$ denote the KL divergence between Bern(a), Bern(b) when $a , b \in [ 0 , 1 ]$ . Let $p _ { E } = P ( E )$ and $q _ { E } = Q ( E )$ . Then, by data processing,

$$
\begin{array} { l } { { \displaystyle \mathrm { K L } ( P , Q ) \geq \mathrm { K L } ( p _ { E } , q _ { E } ) = p _ { E } \log \frac { 1 } { q _ { E } } + ( 1 - p _ { E } ) \log \frac { 1 } { 1 - q _ { E } } - \left( p _ { E } \log \frac { 1 } { p _ { E } } + ( 1 - p _ { E } ) \log \frac { 1 } { 1 - p _ { E } } \right) } } \\ { { \displaystyle \phantom { \frac { 1 } { 1 - p _ { E } } } \sum p _ { E } \log \frac { 1 } { q _ { E } } - \log 2 . } } \end{array}
$$

Combining this with (99) and (100) yields

$$
P ( E ) \leq { \frac { \mathrm { K L } ( P , Q ) + \log 2 } { \log { \frac { 1 } { q _ { E } } } } } \leq { \frac { d / 2 + \log ( 2 ) } { 1 5 ( d / 4 ) } } \leq { \frac { 1 } { 2 } } .
$$

Finally, if the estimator has error at most $\rho ,$ then either E occurs or $\vert J \vert > d / 2$ . Using (86),

$$
\mathbb { P } \Big ( \Big \| \widehat { \Theta } - \Theta \Big \| _ { 2 } \leq \rho \Big ) \leq P ( E ) + \mathbb { P } ( | J | > d / 2 ) \leq \frac { 1 } { 2 } + \frac { 1 } { 8 } = \frac { 5 } { 8 }
$$

$$
\implies \mathbb { P } \Big ( \Big \lVert \widehat { \Theta } - \Theta \Big \rVert _ { 2 } \geq e ^ { - 8 } \sigma \sqrt { d / 2 } \Big ) \geq \frac { 3 } { 8 } . \quad \bigsqcup
$$

## 4 Planted linear regression

In this section, we present our results for planted linear regression. In Section 4.1, we demonstrate that our heteroskedastic regression estimator yields exact recovery for planted linear regression when $m \gg d ^ { 3 / 4 } n ^ { 1 / 4 }$ . In Section 4.2, we present a Statistical Query (SQ) lower bound nearly matching this threshold. Finally, in Section 4.3, we show that no convex M-estimator exactly recovers $\beta$ when $m \ll { \sqrt { n d } }$ . Combined, these results inform the computational-statistical landscape in Fig. 2a.

## 4.1 Exact recovery via heteroskedastic linear regression

We briefly sketch how planted linear regression follows from heteroskedastic regression:

Corollary 4.1. Consider the planted linear regression setting with $\| \beta \| _ { 2 } \le B , n \ge d \ge 1 , B \ge 1$ and $\delta \in ( 0 , 1 / 2 ]$ . There exists a constant $C \geq 1$ such that when

$$
m \geq C \log ^ { 5 } \left( \frac { n d B } { \delta } \right) d ^ { 3 / 4 } n ^ { 1 / 4 } ,\tag{101}
$$

then a polynomial-time estimator recovers $\beta$ exactly (up to the precision of solving a linear system) with probability at least $1 - \delta$

Proof. Divide the samples into two halves of size $n / 2$ . Using a Chernof bound, each half will have at least $m / 4$ noiseless samples, with probability at least $1 - \delta / 4$

Consider a scaling factor

$$
R : = n ^ { 1 0 } \log ^ { 5 } \left( \frac { n B } { \delta } \right) / \delta .
$$

Then, consider a rescaled $y _ { i } ^ { \prime } = R y _ { i }$ for the first half of samples. If we run RB-Desc on $( X _ { i } , y _ { i } ^ { \prime } )$ , this corresponds to a model $y _ { i } ^ { \prime } = X _ { i } ^ { \top } \beta ^ { \prime } + \varepsilon _ { i } ^ { \prime }$ , where $\beta ^ { \prime } = R \beta , \varepsilon _ { i } ^ { \prime } = R \varepsilon _ { i }$ , and the norm bound for $\beta ^ { \prime }$ is now RB.

We now aim to use Theorem 1.5. Observe that the required condition on m holds, as long as we choose a large enough constant in (101).

Let ${ \widehat { \beta } } ^ { \prime }$ denote the estimate given by RB-Desc on $( X _ { i } , y _ { i } ^ { \prime } )$ . Since the rescaled $\varepsilon _ { i } ^ { \prime }$ still have at least $m / 4$ noiseless samples, we may conclude by Theorem 1.5 (with $\delta ^ { \prime } = \delta / 4 )$ that

$$
\left\| \widehat { \beta } ^ { \prime } - \beta ^ { \prime } \right\| _ { 2 } \lesssim \log ^ { 5 } \left( \frac { n d B ^ { \prime } } { \delta } \right) \cdot \sqrt { d } \left( \frac { n } { m ^ { 4 } } \right) ^ { 1 / 6 } \lesssim \log ^ { 5 } \left( \frac { n d B } { \delta } \right) \cdot \sqrt { d } \left( \frac { n } { m ^ { 4 } } \right) ^ { 1 / 6 } .
$$

We now convert ${ \widehat { \beta } } ^ { \prime }$ into an estimate of $\beta ,$ by defining ${ \widehat { \beta } } = { \widehat { \beta } } ^ { \prime } / R$ . This yields

$$
\left\| \widehat { \beta } - \beta \right\| _ { 2 } \lesssim \frac { \log ^ { 5 } \left( \frac { n d B } { \delta } \right) \cdot \sqrt { d } \left( \frac { n } { m ^ { 4 } } \right) ^ { 1 / 6 } } { R } \leq \frac { \delta } { n ^ { 9 } } ,
$$

for a large enough constant in (101).

With the second half of the samples, consider $\tilde { y } _ { i } = y _ { i } - X _ { i } ^ { \top } \widehat { \beta } .$ . By a Chernof bound, at least d of the noiseless samples in this half will have $\textstyle | { \tilde { y } } _ { i } | \leq { \frac { \delta } { n ^ { 9 } } }$ , with probability at least $1 - \delta / 4$ . Further, by a union bound, none of the noisy samples in this half will have $\textstyle | { \tilde { y } } _ { i } | \leq { \frac { \delta } { n ^ { 9 } } }$ , with probability at least $1 - \delta / 4$ . Thus, we may take any d samples with $\textstyle | { \tilde { y } } _ { i } | \leq { \frac { \delta } { n ^ { 9 } } }$ , and output the line that exactly fits all of them (this line is unique with probability 1). □

The simpler heteroskedastic linear regression algorithm for standard Gaussian covariates (see Section 2.1) would have also suficed. We remark that the dependence on B in this theorem could be removed by first running OLS to localize within a ball of radius on the order of $\sqrt { d / n }$

Using the same rescaling argument as Corollary 4.1 implies that any polynomial-time Subset-of-Signals heteroskedastic regression estimator with poly $( n , d , 1 / \delta )$ error when $m \ll d ^ { 3 / 4 } n ^ { 1 / 4 }$ , would yield a planted linear regression estimator in the same regime.

## 4.2 Statistical Query lower bound

Let $d \leq m \leq n$ be positive integers, and $0 < \eta \ll \mathrm { p o l y } ( d / n )$ arbitrarily small (say, superexponentially small in n). For each $\boldsymbol { \beta } \in \mathbb { R } ^ { d }$ , we define a distribution $\mathcal { D } _ { \beta }$ over $\mathbb { R } ^ { d } \times \mathbb { R }$ from which one samples $( x , y )$ as follows:

$$
\begin{array} { r l } & { x \sim { \mathcal { N } } ( 0 , \mathrm { I } _ { d } ) , } \\ & { \varepsilon \sim \frac { m } { n } \cdot { \mathcal { N } } ( 0 , \eta ^ { 2 } ) + ( 1 - \frac { m } { n } ) \cdot { \mathcal { N } } ( 0 , 1 ) , } \\ & { y = \langle \beta , x \rangle + \varepsilon . } \end{array}
$$

Suppose we observe a sample from $\mathcal { D } _ { \beta } ^ { \otimes n }$ for some $\beta$ with $\| \beta \| > 1 / 2$ . The OLS estimator produces an estimate $\hat { \beta } _ { \mathrm { O L S } }$ with $\begin{array} { r } { \| \beta - \hat { \beta } _ { \mathrm { O L S } } \| ^ { 2 } = ( 1 + o ( 1 ) ) \frac { d } { n } } \end{array}$ with high probability. We will consider the ability of statistical query $( S Q )$ algorithms with access to the VSTAT oracle to improve over the OLS estimator.

Definition 4.2. Given a function $\phi : \mathbb { R } ^ { k }  [ 0 , 1 ]$ , the VSTAT(n) oracle to a distribution $\mathcal { D }$ over $\mathbb { R } ^ { k }$ returns $\mathbb { E } _ { \mathcal { D } } \phi + \tau$ for an arbitrary $\tau \in \mathbb { R }$ satisfying $| \tau | \leq$ max $\scriptstyle \left( { \frac { 1 } { n } } , { \sqrt { \frac { \left( 1 - \mathbb { E } _ { \mathcal { D } } \phi \right) \mathbb { E } _ { \mathcal { D } } \phi } { n } } } \right)$

Theorem 4.3. Let $n , d , m$ be non-negative integers satisfying log $n \ll d \ll m \ll n$ . Suppose $\beta \sim \mathcal { N } ( 0 , \mathrm { I } _ { d } )$ is unknown. When $m \ \ll \ ( n d ^ { 3 } ) ^ { 1 / \bar { 4 } }$ , any estimate $\hat { \beta }$ produced by $1 < q \le \exp ( c$ $\begin{array} { r } { \operatorname* { m i n } ( d , \frac { ( n d ^ { 3 } ) ^ { 1 / 6 } } { m ^ { 2 / 3 } } ) \big ) } \end{array}$ or fewer VSTAT(n) queries to $\mathcal { D } _ { \beta }$ will have error $\begin{array} { r } { \| \beta - \hat { \beta } \| ^ { 2 } = \Omega ( \frac { d } { n \log ^ { 2 } q } ) } \end{array}$ with high probability.

To prove the theorem, we will invoke the framework of $[ \mathrm { F G R ^ { + } 1 7 } ]$ , wherein SQ lower bounds are a consequence of bounds on a quantity called the statistical dimension. Here we will use slightly diferent definition:

Definition 4.4. Let $\mathcal { P } = \{ P _ { \theta } \} _ { \theta \in \Theta }$ be a family of probability distributions and $\pi \mathrm { ~ a ~ }$ distribution over Θ. Let $Q$ be a reference distribution with the same domain. The statistical dimension of $\mathcal { P }$ under π relative to $Q$ with average correlation $\xi$ is the largest value $q$ such that for any event E with $\mathrm { P r } _ { \pi } [ E ] \geq { \frac { 1 } { q } } .$

$$
\chi ^ { 2 } \left( \underset { \theta \sim \pi | E } { \mathbb { E } } P _ { \theta } \| Q \right) \le \xi .
$$

Remark 4.5. In $\mathrm { [ F G R ^ { + } 1 7 ] }$ , a slightly diferent notion of statistical dimension is used, in which $\begin{array} { r } { \chi ^ { 2 } ( \mathrm { E } _ { \theta } P _ { \theta } \| Q ) = \mathrm { E } _ { \theta , \theta ^ { \prime } \sim \pi | E } \langle \frac { P _ { \theta } } { Q } , \frac { P _ { \theta ^ { \prime } } } { Q } \rangle _ { Q } - 1 } \end{array}$ is replaced with the potentially larger $\begin{array} { r } { \mathrm { E } _ { \theta , \theta ^ { \prime } \sim \pi | E } \left| \langle \frac { P _ { \theta } } { Q } , \frac { P _ { \theta ^ { \prime } } } { Q } \rangle _ { Q } - 1 \right| } \end{array}$ The proof can easily be adapted to accommodate our definition, at a loss of constant factors.

We’ll say that an SQ algorithm “fails to reject” Q if for every query $\phi$ that it makes, the answer $\mathrm { E } _ { Q } \phi$ is valid for the SQ oracle.

Theorem 4.6 (Mild adaptation of Theorem 2.7 in $\mathrm { [ F G R ^ { + } 1 7 ] } )$ . Let $\delta > 0$ . If the statistical dimension of P under π relative to $Q$ with average correlation $\xi$ is at least $q / \delta$ , then any adaptive SQ algorithm making q or fewer queries to $\mathrm { V S T A T } ( 1 / 9 \xi )$ fails to reject Q with probability at least 1−2δ over the choice of $\theta \sim \pi$

We provide a proof of this adaptation; the main diference is that we work with the chi-squared divergence rather than the signed version, for which we introduce the sets $\Theta _ { i } ^ { + }$ and $\Theta _ { i } ^ { - }$ (whereas in [FGR<sup>+</sup>17] they are merged together and handled with an absolute value). Also, for us π need not be uniform over $\Theta$ , though this is mainly a matter of bookkeeping.

Proof. Let A be any adaptive, deterministic algorithm making q queries to $\mathrm { V S T A T } ( 1 / 9 \xi )$ (the result can be extended to randomized algorithms by averaging over the random seed). Let $\phi _ { 1 } , . . . , \phi _ { q }$ be the sequence of queries made by the algorithm when each $\phi _ { i }$ is answered with $\mathbb { E } _ { Q } \phi _ { i }$ . For convenience, define the query means $p _ { i } = \mathbb { E } _ { Q } \phi _ { i }$ and $p _ { i , \theta } \ = \ \mathbb { E } _ { P _ { \theta } } \phi _ { i } .$ , as well as the $\mathrm { V S T A T } ( 1 / 9 \xi )$ tolerances $\tau _ { i , Q } = \operatorname* { m a x } \left( 9 \xi , 3 \sqrt { \xi p _ { i } ( 1 - p _ { i } ) } \right)$ and $\tau _ { i , \theta } = \operatorname* { m a x } \left( 9 \xi , 3 \sqrt { \xi p _ { i , \theta } ( 1 - p _ { i , \theta } ) } \right)$

Let $\Theta ^ { \ast } \in \Theta$ be the set of parameters θ for which, when receiving oracle answers about $P _ { \theta } { \mathrm { . } }$ , A rejects $Q ,$ and suppose that $\pi ( \Theta ^ { * } ) \ge 2 \delta$ . Then, for each $i \in [ q ]$ , let $\Theta _ { i } ^ { + } \subset \Theta$ be the set of parameters $\theta$ such that

$$
p _ { i , \theta } - p _ { i } > \tau _ { i , \theta }
$$

and define $\Theta _ { i } ^ { - }$ analogously as the set $\{ \theta \in \Theta \mid p _ { i , \theta } - p _ { i } < - \tau _ { i , \theta } \}$ . Since the event of rejecting $Q$ necessitates at least one $\phi _ { i }$ to be answered with a value other than $p _ { i } .$ , it is necessarily the case that

$$
\pi ( \Theta ^ { * } ) \leq \sum _ { i = 1 } ^ { q } \pi ( \Theta _ { i } ^ { + } ) + \pi ( \Theta _ { i } ^ { - } ) ,
$$

and thus there must exist some $i \in [ q ] , s \in \{ \pm 1 \}$ such that $\textstyle \pi ( \Theta _ { i } ^ { s } ) \geq { \frac { \pi ( \Theta ^ { * } ) } { 2 q } } \geq { \frac { \delta } { q } }$ ; note that this implies our bound on the statistical dimension applies to the event that $\bar { { \boldsymbol { \theta } } } \in \Theta _ { i } ^ { s }$ . Let $\xi _ { i } ^ { s }$ be the average absolute diference in the means on this set:

$$
\xi _ { i } ^ { s } = \mathbb { E } _ { \theta \sim \pi | \Theta _ { i } ^ { s } } s \cdot ( p _ { i , \theta } - p _ { i } ) = \underset { \theta \sim \pi | \Theta _ { i } ^ { s } } { \mathbb { E } } s \cdot \left. \phi _ { i } , \frac { d P _ { \theta } } { d Q } - 1 \right. _ { Q } = \underset { \theta \sim \pi | \Theta _ { i } ^ { s } } { \mathbb { E } } s \cdot \left. \phi _ { i } - p _ { i } , \frac { d P _ { \theta } } { d Q } - 1 \right. _ { Q } .
$$

By Cauchy-Schwarz,

$$
\begin{array} { r } { ( \xi _ { i } ^ { s } ) ^ { 2 } \le \operatorname { V a r } _ { Q } ( \phi _ { i } ) \cdot \chi ^ { 2 } ( \underset { \theta \sim \pi | \Theta _ { i } ^ { s } } { \mathbb { E } } P _ { \theta }  Q ) \ \le \operatorname { V a r } _ { Q } ( \phi _ { i } ) \cdot \xi , } \end{array}
$$

where the last inequality follows by our bound on the statistical dimension. We can also lower bound $\xi _ { i } ^ { s }$ by noting that for each $\theta \in \Theta _ { i } ^ { s }$ i

$$
| p _ { i , \theta } - p _ { i } | \geq { \frac { 3 } { 2 } } { \sqrt { \xi p _ { i } ( 1 - p _ { i } ) } } .
$$

Indeed, the 1-Lipschitzness of the function $x \mapsto x ( 1 - x )$ implies that

$$
\sqrt { \xi p _ { i } ( 1 - p _ { i } ) } \leq \sqrt { \xi p _ { i , \theta } ( 1 - p _ { i , \theta } ) } + \sqrt { \xi \cdot | p _ { i } - p _ { i , \theta } | } \leq \frac { 1 } { 3 } \tau _ { i , \theta } + \frac { 1 } { 3 } \sqrt { 9 \xi \cdot | p _ { i } - p _ { i , \theta } | } < \frac { 2 } { 3 } | p _ { i } - p _ { i , \theta } | ,
$$

where we have used that $A \leq B + C \implies \sqrt { A } \leq \sqrt { B } + \sqrt { C }$ , the AM-GM inequality, and the fact that $| p _ { i } - p _ { i , \theta } | > \tau _ { i , \theta } \geq 9 \xi$ by assumption. Thus

$$
( \xi _ { i } ^ { s } ) ^ { 2 } = \bigg ( \underset { \theta \sim \pi \vert \Theta _ { i } ^ { s } } { \mathbb { E } } \vert p _ { i } - p _ { i , \theta } \vert \bigg ) ^ { 2 } > \frac { 9 } { 4 } \xi \cdot p _ { i } ( 1 - p _ { i } ) = \frac { 9 } { 4 } \operatorname { V a r } _ { Q } ( \phi _ { i } ) \cdot \xi .
$$

But our upper bound on $( \xi _ { i } ^ { s } ) ^ { 2 }$ is smaller than our lower bound, a contradiction. Hence we must have $\pi ( \Theta ^ { * } ) < 2 \delta$ □

Our proof will bound the statistical dimension of an appropriately chosen family of distributions, then appeal to the theorem of $\mathrm { [ F G R ^ { + } 1 7 ] }$ . As usual, we will use $C , c > 0$ to denote constants independent of $n , d , m , q .$ whose value may change from line to line.

Proof of Theorem 4.3. We begin with a mathematically convenient way to account for the information available to the algorithm from the OLS estimator: we will use the classic “genie” trick and give the algorithm access to a random vector which is at least as correlated with $\beta$ as the OLS. In particular, let $s = \sqrt { d / ( \ell n ) }$ with $\ell = C \log ^ { 2 } q$ . We decompose

$$
\begin{array} { r } { \beta = \tilde { \beta } + \alpha , \qquad \mathrm { w i t h ~ } \tilde { \beta } \sim \sqrt { 1 - \frac { s ^ { 2 } } { d } } \cdot \mathcal { N } ( 0 , \mathrm { I } _ { d } ) , \quad \alpha \sim \frac { s } { \sqrt { d } } \cdot \mathcal { N } ( 0 , \mathrm { I } _ { d } ) \quad \mathrm { i n d e p e n d e n t l y } . } \end{array}
$$

With probability $1 - \exp ( - c d ) , a = \| \alpha \|$ satisfies $c \leq a \cdot { \sqrt { \ell n / d } } \leq C ;$ condition on this event from now on. We reveal to the algorithm $\beta$ and $a = \| \alpha \|$ , after which

$$
\beta = \tilde { \beta } + a \cdot v ,
$$

where the posterior on $v = \alpha / \| \alpha \|$ is uniform over the sphere $\mathbb { S } ^ { d - 1 }$ . Let $\pi$ denote the uniform measure on $\mathbb { S } ^ { d - 1 }$ ; we will make repeated use of the fact that given $\tilde { \beta } , a , v \sim \pi$

Given ${ \tilde { \beta } } , a ,$ any statistical query $\phi$ depends on each datapoint x, y only through x and the residual $z = y - \langle x , \tilde { \beta } \rangle = a \langle x , v \rangle + \varepsilon$ . Let $\mathcal { D } _ { v }$ be the marginal law of $( x , z )$ given $\tilde { \beta } , a$

Fix $\delta > 0$ . Our result will follow from the following fact: there exist constants $C , c > 0$ depending on δ but independent of $d ,$ n such that for all $n , d$ suficiently large, for all $A \subset \mathbb { S } ^ { d - 1 }$ with $1 / e \geq \pi ( A ) \geq e ^ { - c d }$

$$
\chi ^ { 2 } \left( \underset { u \sim \mathrm { U n i f } ( A ) } { \mathbb { E } } \mathcal { D } _ { u } \parallel Q \right) \leq C \log ^ { 2 } \pi ( A ) \cdot \left( \frac { a ^ { 2 } } { d } + \frac { m ^ { 2 } } { a d n ^ { 2 } } \right) .\tag{102}
$$

Using convexity of chi-squared divergence (any A with $\pi ( { \cal A } ) \ge 1 / e$ can be written as a mixture of subsets with each A<sup>′</sup> satisfying $\pi ( A ^ { \prime } ) \leq 1 / e )$ , this implies nearly the same bound for the broader range $\pi ( A ) \in [ e ^ { - c d } , 1 ]$ 2

$$
\chi ^ { 2 } \left( \underset { u \sim \mathrm { U n i f } ( A ) } { \mathbb { E } } \mathcal { D } _ { u } \parallel Q \right) \leq C \log ^ { 2 } \frac { e } { \pi ( A ) } \cdot \left( \frac { a ^ { 2 } } { d } + \frac { m ^ { 2 } } { a d n ^ { 2 } } \right) .\tag{103}
$$

Since $a \sim \sqrt { d / ( n \log ^ { 2 } q ) }$ and $m \leq c ( \frac { n d ^ { 3 } } { \log ^ { 6 } q } ) ^ { 1 / 4 }$ , the right-hand side of Eq. (103) is $\leq 1 / ( 9 n )$ for all A with $\textstyle { \frac { \delta } { q } } \leq \pi ( A ) < 1$ . Thus, invoking Theorem 4.6, any adaptive SQ algorithm which makes at most q queries to $\mathrm { V S T A T } ( n )$ cannot reject Q when run on $\mathcal { D } _ { v }$ . To see why this implies Theorem 4.3,

consider any deterministic algorithm which fails to reject $Q$ . To estimate $\beta$ given $a , { \tilde { \beta } } .$ , the algorithm must output a deterministic $\hat { \beta } = \tilde { \beta } + \hat { \alpha }$ which is not a function of $\mathcal { D } _ { v }$ . Then

$$
\| \beta - \hat { \beta } \| ^ { 2 } = \| a \cdot v - \hat { \alpha } \| ^ { 2 } = a ^ { 2 } + \| \hat { \alpha } \| ^ { 2 } - 2 a \langle \hat { \alpha } , v \rangle .
$$

As $v \sim \pi$ the inner product $\langle \hat { \alpha } , v \rangle$ is subgaussian with variance proxy $\scriptstyle { \frac { 1 } { d } } \parallel { \hat { \alpha } } \parallel ^ { 2 }$ , and so except with probability $e ^ { - c d } , \| \beta - \hat { \beta } \| ^ { 2 } \geq { \textstyle { \frac { 1 } { 2 } } } a ^ { 2 }$ . The conclusion follows by averaging over the random seed.

We’ll prove Eq. (102) by directly bounding the integral of the centered, squared density of $\mathrm { E } _ { u \sim \pi _ { A } } \mathcal { D } _ { u }$ relative to $Q .$ . It is crucial that we center using the mixture density $Q ;$ however, we will be able to substitute a simpler expression in the denominator to make the bound more tractable. The subsequent bound follows from straightforward (and tedious) analytic arguments. We’ll use Gaussian integration to express the integral as an analytic function of $\rho = \langle u , w \rangle$ in expectation under $u , w \sim \pi$ and $u , w \sim \pi | A $ , and then use a polynomial expansion to get bounds in terms of the diferences of low-order moments in $\rho$ under both distributions. We will then show that the moments are within poly log $\pi ( A )$ factors, completing the proof.

Plowing on, let $\gamma ^ { d }$ be the density of $\mathcal { N } ( 0 ,  { \mathrm { I } _ { d } } )$ in $\mathbb { R } ^ { d } .$ , let $\gamma _ { \sigma }$ be the density of ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ in $\mathbb { R } .$ . For each $v \in \mathbb { S } ^ { d - 1 }$ , we can express the density $p _ { v } ( x , z )$ of $\mathcal { D } _ { v }$ as a mixture:

$$
p _ { v } ( x , z ) = \frac { m } { n } \cdot h _ { v } ( x , z ) + \Big ( 1 - \frac { m } { n } \Big ) \cdot g _ { v } ( x , z ) ,
$$

where $h _ { v }$ is the small-label-noise component $h _ { v } ( x , z ) = \gamma ^ { d } ( x ) \cdot \gamma _ { \eta } ( z - a \langle x , v \rangle )$ and $g _ { v }$ is the largelabel-noise component $g _ { v } ( x , z ) = \gamma ^ { d } ( x ) \cdot \gamma _ { 1 } ( z - a \langle x , v \rangle )$

Fix the event A and let $\pi _ { A } = \pi \mid A$ . For convenience let $p _ { \pi } = \mathbb { E } _ { u \sim \pi } p _ { u }$ , let $p _ { A } = \mathbb { E } _ { u \sim \pi _ { A } } p _ { u }$ , and let $g _ { \pi } , g _ { A } , h _ { \pi } , h _ { A }$ be defined similarly. We can express the chi-squared divergence

$$
\begin{array} { r l } & { \chi ^ { 2 } \left( \underset { w \sim \pi _ { A } } { \mathbb { E } } ~ \mathcal { D } _ { u } ~ \lVert Q \right) = \displaystyle \int \int \int \frac { ( p _ { A } ( x , z ) - p _ { \pi } ( x , z ) ) ^ { 2 } } { p _ { \pi } ( x , z ) } d x d z } \\ & { \qquad \leq \displaystyle \frac { 1 } { 1 - \frac { m } { n } } \int \int \frac { ( p _ { A } ( x , z ) - p _ { \pi } ( x , z ) ) ^ { 2 } } { g _ { \pi } ( x , z ) } d x d z \leq 2 \int \int \frac { \left( p _ { A } ( x , z ) - p _ { \pi } ( x , z ) \right) ^ { 2 } } { g _ { \pi } ( x , z ) } d x d z , } \end{array}\tag{104}
$$

where the final inequality follows because m $\ll n$ . By Jensen’s inequality, we have that

$$
g _ { \pi } ( x , z ) = \mathop { \mathbb { E } } _ { v \sim \pi } ^ { \mathbb { E } } g _ { v } ( x , z ) = \gamma ^ { d } ( x ) \cdot \gamma _ { 1 } ( z ) \cdot \underbrace { \mathbb { E } } _ { v \sim \pi } \exp \bigl ( z a \langle x , v \rangle - \frac { 1 } { 2 } a ^ { 2 } \langle x , v \rangle ^ { 2 } \bigr ) \geq \gamma ^ { d } ( x ) \cdot \gamma _ { 1 } ( z ) \cdot \exp \left( - \frac { a ^ { 2 } } { 2 d } \| x \| ^ { 2 } \right)
$$

So we may substitute the right-hand side above into the denominator in Eq. (104),

$$
E q . \ ( 1 0 4 ) \leq 2 \int \int \frac { ( p _ { A } ( x , z ) - p _ { \pi } ( x , z ) ) ^ { 2 } } { \gamma ^ { d } ( x ) \gamma _ { 1 } ( z ) e ^ { - \frac { a ^ { 2 } } { 2 d } \| x \| ^ { 2 } } } d x d z ,
$$

By Cauchy-Schwarz,

$$
\leq 4 \left( \frac { m } { n } \right) ^ { 2 } \int \int \frac { \left( h _ { A } ( x , z ) - h _ { \pi } ( x , z ) \right) ^ { 2 } } { \gamma ^ { d } ( x ) \gamma _ { 1 } ( z ) e ^ { - \frac { a ^ { 2 } } { 2 d } \| x \| ^ { 2 } } } d x d z + 4 \int \int \frac { \left( g _ { A } ( x , z ) - g _ { \pi } ( x , z ) \right) ^ { 2 } } { \gamma ^ { d } ( x ) \gamma _ { 1 } ( z ) e ^ { - \frac { a ^ { 2 } } { 2 d } \| x \| ^ { 2 } } } d x d z .\tag{105}
$$

We will now integrate over $x , z$ . To make use of the common form of both integrals, let $k _ { A } =$ $\mathbb { E } _ { u \sim \pi _ { A } } \gamma ^ { d } ( x ) \gamma _ { \sigma } ( z - a \langle x , u \rangle )$ , and define $k _ { \pi }$ similarly. Note that when $\sigma = 1$ then $k _ { A } = g _ { A }$ and when $\sigma = \eta$ then $k _ { A } = h _ { A }$ . We factor out a multiple of $\gamma ^ { d } ( \boldsymbol { x } ) \gamma _ { 1 } ( z )$ and obtain the bound

$$
\begin{array} { r l } & { \displaystyle \int \int \int \frac { ( k _ { A } ( x , z ) - k _ { \pi } ( x , z ) ) ^ { 2 } } { \gamma ^ { d } ( x ) \gamma _ { 1 } ( z ) e ^ { - \frac { a ^ { 2 } } { 2 d } \| x \| ^ { 2 } } } d x d z } \\ & { \quad \quad \quad = \frac { 1 } { \sigma ^ { 2 } } \underset { x \sim \gamma _ { 1 } } { \mathbb { E } } e ^ { - \left( \frac { 1 } { \sigma ^ { 2 } } - 1 \right) z ^ { 2 } + \frac { a ^ { 2 } } { 2 d } \| x \| ^ { 2 } } \left( \underset { w \sim \pi _ { A } } { \mathbb { E } } e ^ { \frac { a } { \sigma ^ { 2 } } z \langle x , u \rangle - \frac { a ^ { 2 } } { 2 \sigma ^ { 2 } } \langle x , u \rangle ^ { 2 } } - \underset { v \sim \pi } { \mathbb { E } } e ^ { \frac { a } { \sigma ^ { 2 } } z \langle x , v \rangle - \frac { a ^ { 2 } } { 2 \sigma ^ { 2 } } \langle x , v \rangle ^ { 2 } } \right) ^ { 2 } , } \\ & { \quad \quad \quad = \underset { u _ { 1 } , u _ { 2 } \sim \pi _ { A } } { \mathbb { E } } M _ { \sigma } ( u _ { 1 } , u _ { 2 } ) + \underset { v _ { 1 } , v _ { 2 } \sim \pi } { \mathbb { E } } M _ { \sigma } ( v _ { 1 } , v _ { 2 } ) - 2 \underset { w \sim \pi _ { A } , v \sim \pi } { \mathbb { E } } M _ { \sigma } ( u , v ) , } \end{array}\tag{06}
$$

Where $M _ { \sigma }$ is obtained via Gaussian integration over $x , z ,$ , using that $\sigma \leq 1$ and $a \ll 1$ to ensure the integral is defined. Note that because of the rotational symmetry of the law of $x , \ M _ { \sigma } ( w , w ^ { \prime } )$ can depend on $w , w ^ { \prime }$ only through their inner product. In particular, letting $\textstyle t = 1 - { \frac { a ^ { 2 } } { d } }$ and letting $\rho = \langle w , w ^ { \prime } \rangle$ 2

$$
M _ { \sigma } ( w , w ^ { \prime } ) = M _ { \sigma } ( \rho ) = { t ^ { - ( d - 2 ) / 2 } } \bigg ( \left( t ( 2 - \sigma ^ { 2 } ) - a ^ { 2 } ( 1 + \rho ) \right) \left( t \sigma ^ { 2 } + a ^ { 2 } ( 1 - \rho ) \right) \bigg ) ^ { - 1 / 2 } .
$$

Since $M _ { \sigma } ( w , w ^ { \prime } )$ depends only on the inner product $\rho = \langle w , w ^ { \prime } \rangle$ , and because for any unit vector w the law of $\langle v , w \rangle$ when $v \sim \pi$ is independent of $w ,$ we can further simplify

$$
\begin{array} { r } { E q . \ ( 1 0 6 ) = \underset { \rho _ { A } \sim \pi _ { A } } { \mathbb { E } } M _ { \sigma } ( \rho _ { A } ) - M _ { \sigma } ( \rho ) , } \end{array}\tag{107}
$$

where we abuse notation slightly and denote by $\rho _ { A } \sim \pi _ { A }$ the distribution of the inner product of $u , u ^ { \prime } \sim \pi _ { A }$ (and similarly for $\rho \sim \pi )$

We now specialize to the diferent cases $\sigma = 1 , \eta _ { \mathrm { { } } }$ , as we will require a diferent bound in each case. Considering first the case $\sigma = 1$ ，

$$
M _ { 1 } ( \rho ) = c _ { 0 } \left( 1 - c _ { 1 } a ^ { 2 } \rho + c _ { 2 } ( a ^ { 2 } \rho ) ^ { 2 } \right) ^ { - 1 / 2 }
$$

where we can apply the fact that $t = 1 - a ^ { 2 } / d$ and the fact that $a \ = \ o ( 1 )$ to conclude that $c _ { 0 } , c _ { 2 } = 1 \pm o ( 1 )$ and $c _ { 1 } = 2 \pm o ( 1 )$ . Thinking of the coeficients $c _ { 0 } , c _ { 2 }$ as fixed quantities of magnitude at most 1.01 and $c _ { 1 }$ as a fixed quantity of magnitude at most 2.01, because the quantity $\textstyle | a ^ { 2 } \rho | < { \frac { 1 } { 2 } }$ by assumption, the Taylor series of $M _ { 1 } ( \rho )$ in $a ^ { 2 } \rho$ converges:

$$
M _ { 1 } ( \rho ) = c _ { 0 } \left( 1 + \frac { c _ { 1 } } { 2 } a ^ { 2 } \rho + R _ { 1 } ( \rho ) \right)
$$

where $| R _ { 1 } ( \rho ) | \le ( a ^ { 2 } \rho ) ^ { 2 }$

Hence, applying the above and Eq. (107) to bound the integral of the $N ( 0 , 1 )$ )-noise component,

$$
\begin{array} { r l r } {  { \int \int \int \frac { ( g _ { A } ( x , z ) - g _ { \pi } ( x , z ) ) ^ { 2 } } { \gamma ^ { d } ( x ) \gamma _ { 1 } ( z ) e ^ { - \frac { a ^ { 2 } } { 2 d } \| x \| ^ { 2 } } } d x d z = \underbrace { \mathbb { E } } _ { \rho _ { A } \sim \pi _ { A } } M _ { 1 } ( \rho _ { A } ) - M _ { 1 } ( \rho ) } } \\ & { } & { \leq \frac { c _ { 0 } c _ { 1 } a ^ { 2 } } { 2 } | \underset { \rho _ { \sim \pi } \sim \pi } { \underbrace { \mathbb { E } } } \rho _ { A } - \rho | + c _ { 0 } \underset { \rho \sim \pi } { \underbrace { \mathbb { E } } } | R _ { 1 } ( \rho ) | + | R _ { 1 } ( \rho _ { A } ) | } \\ & { } & { \leq a ^ { 2 } | \underset { \rho _ { \exp } \pi } { \underbrace { \mathbb { E } } } \rho _ { A } - \rho | + 2 \underset { \rho \sim \pi _ { A } } { \underbrace { \mathbb { E } } } a ^ { 4 } ( \rho ^ { 2 } + \rho _ { A } ^ { 2 } ) . } \end{array}\tag{108}
$$

When $\sigma = \eta$ , we take the parameterization

$$
M _ { \eta } ( \rho ) = \frac { b _ { 0 } } { a } \left( 1 - b _ { 1 } \rho + b _ { 2 } a ^ { 2 } \rho ^ { 2 } \right) ^ { - 1 / 2 }
$$

where b<sub>0</sub>, $b _ { 2 } = 2 ^ { - 1 / 2 } \pm o ( 1 ) , b _ { 1 } = 1 \pm o ( 1 )$ , and all three are independent of $\rho .$ Here, we take a Taylor expansion in $\rho \colon$

$$
M _ { \eta } ( \rho ) = \frac { b _ { 0 } } { a } \left( 1 + \frac { b _ { 1 } } { 2 } \rho + R _ { \eta } ( \rho ) \right) ,
$$

where for all $| \rho | \leq { \frac { 1 } { 2 } } , | R _ { \eta } ( \rho ) | \leq \rho ^ { 2 }$ . We also require a bound on $M _ { \eta }$ for all $\rho$ which we will apply when $| \rho | \geq { \frac { 1 } { 2 } }$

$$
M _ { \eta } ( \rho ) \leq \frac { 2 } { a } \frac { 1 } { \sqrt { 1 - \rho } } ,
$$

which can be verified by inspection of the formula for $M _ { \sigma }$ (the first two factors are bounded by 2 when $\sigma = \eta$ , the last term is smaller than $( a ^ { 2 } ( 1 - \rho ) ) ^ { - 1 / 2 } )$

Applying the above in Eq. (107) to bound the integral of the $N ( 0 , \eta ^ { 2 } )$ -noise component,

$$
\begin{array} { r l r } {  { \int \int \int \frac { ( h _ { A } ( x , z ) - h _ { \pi } ( x , z ) ) ^ { 2 } } { \gamma ^ { d } ( x ) \gamma _ { 1 } ( z ) e ^ { - \frac { a ^ { 2 } } { 2 d } \| x \| ^ { 2 } } } d x d z = \underbrace { \mathbb { E } } _ { \rho _ { \underset { \rho \sim \pi } { \sim } \pi } } M _ { \eta } ( \rho _ { A } ) - M _ { \eta } ( \rho ) } } \\ & { } & { \leq \frac { b _ { 0 } b _ { 1 } } { 2 a } | \underset { \rho \sim \pi } { \underbrace { \mathbb { E } } } _ { \rho \sim \pi } \rho _ { A } - \rho | + \frac { b _ { 0 } } { a } \underset { \rho \sim \pi } { \underbrace { \mathbb { E } } } _ { a \sim \pi } | R _ { \eta } ( \rho ) | + | R _ { \eta } ( \rho _ { A } ) | . } \end{array}\tag{109}
$$

We will bound the remainders by separately considering the event $E ( \rho ) ~ = ~ \{ | \rho | ~ \leq ~ { \frac { 1 } { 2 } } \}$ and its complement. Treating $\rho$ (the logic is identical for $\rho _ { A } )$ ,

$$
\begin{array} { r l r } {  { \mathbb { E } | R _ { \eta } ( \rho ) | = \mathbb { E } | R _ { \eta } ( \rho ) | \cdot \mathbf { 1 } _ { E ( \rho ) } + | R _ { \eta } ( \rho ) | \cdot \mathbf { 1 } _ { \overline { { E } } ( \rho ) } } } \\ & { } & { \leq \mathbb { E } \rho ^ { 2 } + \frac { \mathbb { E } } { \rho } [ ( \frac { a } { b _ { 0 } } | M _ { \eta } ( \rho ) | + | 1 + \frac { b _ { 1 } } { 2 } \rho | ) \mathbf { 1 } _ { \overline { { E } } ( \rho ) } ] } \\ & { } & { \leq \mathbb { E } \rho ^ { 2 } + \frac { \mathbb { E } } { \rho } [ ( \frac { 2 } { b _ { 0 } } ( 1 - \rho ) ^ { - 1 / 2 } + 2 ) \mathbf { 1 } _ { \overline { { E } } ( \rho ) } ] } \\ & { } & { \leq \mathbb { E } \rho ^ { 2 } + \frac { \mathbb { E } } { \rho } [ ( 1 0 ( 1 - \rho ) ^ { - 1 / 2 } ) \mathbf { 1 } _ { \overline { { E } } ( \rho ) } ] . } \end{array}
$$

So we have

$$
E q . \left( 1 0 9 \right) \leq \frac { 1 } { a } \left| \mathbb { E } \rho _ { A } - \rho \right| + \frac { 2 } { a } \mathbb { E } ( \rho _ { A } ^ { 2 } + \rho ^ { 2 } ) + \frac { 1 1 } { a } \left( \mathbb { E } \frac { \mathbf { 1 } _ { \overline { { E } } ( \rho ) } } { \sqrt { 1 - \rho } } + \frac { \mathbf { 1 } _ { \overline { { E } } ( \rho _ { A } ) } } { \sqrt { 1 - \rho _ { A } } } \right) .\tag{110}
$$

So to finish up given Eqs. (108) and (110), we require a bound on the diference in the first moment of $\rho , \rho _ { A }$ , an upper bound on the second moments, and control on the expectation under the tail event $\overline { E }$

To control the moments, we invoke Lemma 4.4 of [LMSY24], where it is shown that the optimal transport coupling of $\pi _ { A } , \pi$ provides a way to jointly sample u $\sim \pi _ { A } , v \sim \pi$ such that $\Delta = u - v$ satisfies, for all $t > 0 , \operatorname* { P r } ( \| \Delta \| > t + \sqrt { - 2 \log \pi ( A ) / d } ) \le \exp \left( - \frac { d - 1 } { 8 } t ^ { 2 } \right)$ . In particular, letting

$( v , v + \Delta )$ and $( v ^ { \prime } , v ^ { \prime } + \Delta ^ { \prime } )$ be two independent samples from the optimal transport coupling of $( \pi , \pi _ { A } )$

$$
\mathbb { E } \rho _ { A } - \rho = \mathbb { E } \langle v + \Delta , v ^ { \prime } + \Delta ^ { \prime } \rangle - \langle v , v ^ { \prime } \rangle = \mathbb { E } \langle \Delta , \Delta ^ { \prime } \rangle \leq ( \mathbb { E } \left. \Delta \right. ) ^ { 2 } \leq C \frac { - \log \pi ( A ) } { d } ,
$$

where we have used that $\mathrm { E } [ v ] = 0$ , Cauchy-Schwarz, the independence of $\Delta , \Delta ^ { \prime }$ , and the fact that $\| \Delta \| - \sqrt { - \log \pi ( A ) / d }$ is subgaussian with variance proxy $O ( 1 / d )$

The same coupling allows us to upper bound the second moments. First, we directly compute E $\begin{array} { r } { \rho ^ { 2 } = \mathbb { E } ( v ^ { \prime } ) ^ { \top } ( v v ^ { \top } ) v ^ { \prime } = \frac { 1 } { d } } \end{array}$ . We further have that

$$
\begin{array} { r } { \mathbb { E } \rho _ { A } ^ { 2 } = \mathbb { E } \langle v + \Delta , v ^ { \prime } + \Delta ^ { \prime } \rangle ^ { 2 } \leq \mathbb { E } 4 \langle v , v ^ { \prime } \rangle ^ { 2 } + 8 \langle v , \Delta ^ { \prime } \rangle ^ { 2 } + 4 \langle \Delta , \Delta ^ { \prime } \rangle ^ { 2 } \leq \frac { 4 } { d } + \frac { 8 } { d } \mathbb { E } [ \| \Delta ^ { \prime } \| ^ { 2 } ] + 4 \mathbb { E } [ \| \Delta \| ^ { 2 } ] ^ { 2 } \leq \frac { C } { d } + \frac { C ( \log \pi ( A ) ) ^ { 2 } } { d ^ { 2 } } , } \end{array}
$$

where we have again used Cauchy-Schwarz and the subgaussianity of $\| \Delta \|$

Finally, we note that $( 1 - \rho ) ^ { - 1 / 2 } \mathbf { 1 } _ { | \rho | \geq { \frac { 1 } { 2 } } } \geq 0$ for $| \rho | \le 1$ , and for any non-negative function $f ( \rho )$

$$
\mathbb { E } f ( \rho _ { A } ) = \frac { \mathbb { E } _ { v , v ^ { \prime } \sim \pi } f ( \langle v , v ^ { \prime } \rangle ) \mathbf { 1 } _ { v , v ^ { \prime } \in A } } { \operatorname* { P r } [ v , v ^ { \prime } \in A ] } \leq \frac { 1 } { \pi ( A ) ^ { 2 } } \mathrm { E } [ f ( \rho ) ] .
$$

And using that the density of $\rho$ under $\pi$ is $\pi ( \rho ) = C _ { d } ( 1 - \rho ^ { 2 } ) ^ { ( d - 3 ) / 2 }$ for $\begin{array} { r } { C _ { d } = \frac { \Gamma ( d / 2 ) } { \sqrt { \pi } \Gamma ( ( d - 1 ) / 2 ) } , } \end{array}$

$$
\begin{array} { r l } & { \mathbb { E } \frac { { \mathbf 1 } _ { | \rho | > \frac { 1 } { 2 } } } { \sqrt { 1 - \rho } } = \displaystyle \int _ { - 1 } ^ { 1 } \frac { { \mathbf 1 } _ { | \rho | > \frac { 1 } { 2 } } } { \sqrt { 1 - \rho } } \cdot C _ { d } \cdot \left( 1 - \rho ^ { 2 } \right) ^ { ( d - 3 ) / 2 } d \rho } \\ & { \quad \quad \quad \quad = \displaystyle \int _ { - 1 } ^ { 1 } { \mathbf 1 } _ { | \rho | > \frac { 1 } { 2 } } C _ { d } \cdot ( 1 + \rho ) ^ { 1 / 2 } \cdot \left( 1 - \rho ^ { 2 } \right) ^ { ( d - 4 ) / 2 } d \rho } \\ & { \quad \quad \quad \quad \leq \displaystyle \int _ { - 1 } ^ { 1 } { \mathbf 1 } _ { | \rho | > \frac { 1 } { 2 } } C _ { d } \cdot \sqrt { 2 } \cdot \left( 1 - \rho ^ { 2 } \right) ^ { ( d - 4 ) / 2 } d \rho } \\ & { \quad \quad \quad \quad = \sqrt { 2 } \frac { C _ { d } } { C _ { d - 1 } } \operatorname* { P r } _ { w , w ^ { \prime } \sim \mathrm { U n i f } ( \mathbb { S } ^ { d - 2 } ) } [ | \langle w , w ^ { \prime } \rangle | \geq \frac { 1 } { 2 } ] } \\ & { \quad \quad \quad \quad \leq C \exp ( - c ^ { \prime } d ) , } \end{array}
$$

where the last line follows for all d suficiently large by the $O ( 1 / d )$ -subgaussianity of the inner product $\langle w , w ^ { \prime } \rangle$ and because $C _ { d } / C _ { d - 1 } = 1 + o _ { d } ( 1 )$

We can now use these estimates to complete our bound from $\operatorname { E q . }$ (105) using Eqs. (108) and (110) and the assumption that $1 / e \geq \pi ( A ) \geq e ^ { - c d }$ for $c < c ^ { \prime } / 2 \colon$

$$
\begin{array} { r l } & { E q . \left( 1 0 5 \right) \leq C \left( \frac { m } { n } \right) ^ { 2 } \cdot \left( \frac { \log ^ { 2 } \pi ( A ) } { a d } + \frac { e ^ { - c ^ { \prime } d } } { a \pi ( A ) ^ { 2 } } \right) + C \left( a ^ { 2 } \frac { - \log \pi ( A ) } { d } + a ^ { 4 } \frac { \log ^ { 2 } \pi ( A ) } { d } \right) } \\ & { \qquad \leq C \log ^ { 2 } \pi ( A ) \left( \frac { m ^ { 2 } } { a d n ^ { 2 } } + \frac { a ^ { 2 } } { d } \right) , } \end{array}
$$

which completes the proof of Eq. (102).

## 4.3 Barrier for convex M-estimation

We will show that convex M-estimation cannot exactly recover $\beta$ in planted linear regression once $m \ll \sqrt { n d }$

In particular, let $\rho : \mathbb { R }  \mathbb { R }$ be any finite convex function chosen independently of the sample.<sup>3</sup> We define the M-estimation objective as

$$
L _ { \rho } ( b ) : = \sum _ { i = 1 } ^ { n } \rho ( y _ { i } - X _ { i } ^ { \top } b ) ,
$$

and we say the estimator exactly identifies $\beta$ if it is the unique minimizer, meaning

$$
\operatorname * { a r g m i n } _ { b \in \mathbb { R } ^ { d } } L _ { \rho } ( b ) = \{ \beta \} .
$$

We show that no such M-estimator succeeds in exactly identifying $\beta$ when $m \ll { \sqrt { n d } } .$

Theorem 4.7. There exists a universal constant $c > 0$ such that for any finite convex loss $\rho ,$ if $m \leq \operatorname* { m i n } \{ c { \sqrt { n d } } , n / 2 \}$ then

$$
\begin{array} { r } { \mathbb { P } \left( \underset { b \in \mathbb { R } ^ { d } } { \arg \operatorname* { m i n } } L _ { \rho } ( b ) = \{ \beta \} \right) \lesssim e ^ { - c d } + e ^ { - c m } . } \end{array}
$$

Proof. Our proof will follow by studying the conditions where $0 \in \partial L _ { \rho } ( \beta )$ . We leverage that $\rho$ has subderivatives everywhere, and is diferentiable almost everywhere. We denote

$$
\partial \rho ( 0 ) = [ l , r ] , \qquad R : = \operatorname* { m a x } \{ | l | , | r | \} .
$$

For $i \not \in S _ { \mathrm { : } }$ , it will be helpful to define

$$
a _ { i } : = \rho ^ { \prime } ( \varepsilon _ { i } ) ,
$$

which is well-defined with probability 1 because $\rho$ is diferentiable almost everywhere and $\varepsilon _ { i }$ is a continuous distribution. We now study $\partial L _ { \rho } ( \beta )$ ; since the residual is 0 for i in the noiseless set $S _ { ; }$ it holds that

$$
\partial L _ { \rho } ( \beta ) = - \left\{ \sum _ { i \notin S } a _ { i } X _ { i } + \sum _ { i \in S } s _ { i } X _ { i } : s _ { i } \in [ l , r ] \right\} .\tag{111}
$$

We separate this into two terms

$$
Z : = \sum _ { i \notin S } a _ { i } X _ { i } , \qquad K : = \left\{ \sum _ { i \in S } s _ { i } X _ { i } : s _ { i } \in [ l , r ] \right\} .
$$

In this language, we may use the equivalent characterizations

$$
\begin{array} { r } { \beta \in \arg \operatorname* { m i n } L _ { \rho } \quad \Longleftrightarrow \quad 0 \in \partial L _ { \rho } ( \beta ) \quad \Longleftrightarrow \quad - Z \in K . } \end{array}
$$

We separate our proof into two cases: (i) when $R > 0$ , we show $\beta$ is usually not a minimizer, and (ii) when $R = 0$ , the flatness at 0 will let us show that $\beta$ is usually not a unique minimizer.

Case (i): $R > 0$ . We will prove that with high probability it holds that both

$$
\| Z \| _ { 2 } \geq c R { \sqrt { n d } } ,\tag{112}
$$

$$
\operatorname* { s u p } _ { w \in K } \left. - Z / \left. Z \right. _ { 2 } , w \right. \leq C R m .\tag{113}
$$

Together, these imply that if $m \leq c _ { 0 } { \sqrt { n d } }$ for small enough $c _ { 0 }$ , then $- Z \notin K$ and hence $\beta \notin$ arg min $L _ { \rho }$

Proof of (112). By monotonicity of $\rho ^ { \prime } ,$ it holds that $| \rho ^ { \prime } ( t ) | \geq R$ either for all $t > 0$ , or for all $t < 0$ . Hence, for each $i \not \in S ,$ it holds that $| a _ { i } | \geq R$ with probability at least $1 / 2$ . Using a Chernof bound, this implies that at least $\frac { n - m } { 4 } \geq n / 8$ samples from $i \not \in S$ satisfy $| a _ { i } | \geq R$ , with probability at least $1 - e ^ { - c n }$ . Conditioned on this event, we denote

$$
q : = \sum _ { i \notin S } a _ { i } ^ { 2 } \geq \frac { n R ^ { 2 } } { 8 } .
$$

Conditioned on the values of $a _ { i }$ , the value of $Z$ is distributed like

$$
Z \mid ( a _ { i } ) _ { i \not \in S } \sim N ( 0 , q I _ { d } ) .
$$

We may use another Chernof bound to conclude that its norm satisfies

$$
\| Z \| _ { 2 } \gtrsim \sqrt { q d } \gtrsim R \sqrt { n d }
$$

with probability at least $1 - e ^ { - c d } .$

Proof of (113). Let $\begin{array} { r } { v = \frac { Z } { \Vert Z \Vert _ { 2 } } } \end{array}$ denote the relevant vector in (113); observe how v is independent of $( X _ { i } ) _ { i \in S }$ . For every $\begin{array} { r } { w = \sum _ { i \in S } s _ { i } X _ { i } \in K } \end{array}$ , it holds that

$$
\langle - v , w \rangle = \sum _ { i \in S } ( - s _ { i } ) \langle v , X _ { i } \rangle \leq R \sum _ { i \in S } | \langle v , X _ { i } \rangle | .
$$

Conditioned on v, the random variables $| \langle v , X _ { i } \rangle |$ are i.i.d. $| N ( 0 , 1 )$ |, and hence

$$
\sum _ { i \in S } | \langle v , X _ { i } \rangle | \lesssim m
$$

with probability at least $1 - e ^ { - c m }$ by Bernstein’s inequality.

In total, this yields the desired guarantee for $R > 0$ with probability at least $1 - e ^ { - c d } - e ^ { - c m }$ Case (ii): $R = 0$ . When $R = 0$ , it holds that $\partial \rho ( 0 ) = \{ 0 \}$ , and hence

$$
\partial L _ { \rho } ( \beta ) = \{ - Z \} ,
$$

meaning $\beta$ can only be an optimizer when $Z = 0$ . If at least one value of $a _ { i }$ is nonzero, then conditioned on the values of $( a _ { i } ) _ { i \notin S }$ , Z is a nondegenerate Gaussian vector and hence $Z \neq 0$ with probability 1 (in which case $\beta$ is not an optimizer).

Hence, we may focus on the event where $\rho ^ { \prime } ( \varepsilon _ { i } ) = 0$ for every $i \not \in S$ . Consider the region

$$
M : = \underset { t \in \mathbb { R } } { \arg \operatorname* { m i n } } \rho ( t ) ,
$$

or equivalently $M = \{ t : 0 \in \partial \rho ( t ) \}$ ; this is a nonempty interval containing 0. If no $\varepsilon _ { i }$ is at the boundary of M, then $\beta$ is not a unique minimizer since a small perturbation of $\beta$ is also a minimizer. For any $i \not \in S$ , an $\varepsilon _ { i }$ is realized exactly at the boundary with probability 0. We only need to consider the possibility where 0 is at the boundary of M. In this case, it must hold that either: (i) all $\varepsilon _ { i } < 0$ for $i \not \in S _ { \mathrm { : } }$ , or (ii) all $\varepsilon _ { i } > 0$ for $i \not \in S$ . This event occurs with probability at most $e ^ { - c n }$ , thus implying our desired result. □

## 5 Simulations

In this section we discuss our simulations. We have modified the estimators given in the theorems in order to make them more practical; we detail these changes, and then discuss the results of our simulation for planted and adaptive linear regression.

Estimator implementations. The existing estimators we compare to include OLS, LAD $( L _ { 1 }$ regression), $L _ { \infty }$ regression, and the ASM (antitonic score matching) estimator of Feng, Kao, $\mathrm { X u }$ and Samworth [FKXS26]. For the ASM estimator, we use version 0.2.4 of their R package with the default parameters except we set alt\_iter = 1 (which corresponds to their ASM estimator), and we use the default pilot estimator LAD in all settings other than in Fig. 4e (where the pilot estimator is OLS, since this is done in their analogous experiment).

We now discuss our estimator implementations, which are also available at https://github. com/SpencerCompton/adaptive-regression.

Adaptive $L _ { q }$ . Compared to the mathematically-analyzed version of this estimator, there are two main diferences. First, we include $q = 1 \left( \mathrm { L A D } \right)$ as an option for the estimator. For some theoretical justification, the only concern is whether the candidate selection in Theorem 1.6 is still valid when including $q = 1$ . The proof only uses the property that when $\operatorname { r a n k } ( \mathbf { X } ) = d ,$ the optimizer of the objective must be unique, and the optimizer’s distribution is symmetric around $\beta .$ . This does not immediately hold for $q = 1$ (the optimizer need not be unique), but it is true when we tiebreak among the LAD optimizers by choosing the optimizer whose residuals have the smallest $L _ { 2 }$ norm; we use this tiebreaking version in our implementation for adaptive $L _ { q }$ . Second, instead of outputting the selected block estimate $\widehat { A } _ { \widehat { q } , \widehat { i } } .$ we run $L _ { q }$ regression on the whole dataset with this choice of ${ \widehat { q } } ,$ and output this estimate (this version improves performance, but because $\widehat { q }$ is not independent of the data, our proof no longer applies).

Residual-balance descent (RB-Desc). Our implementation incorporates substantial changes to optimize empirical performance.

First, we discuss some relatively minor changes. We initialize with the OLS estimate instead of the all-zeros vector (this is not precluded by our analysis). Instead of taking in bounds on $B , K , \kappa ,$ we apply a whitening preprocessing step, and then use a B at the scale of the typical residual magnitude; if the estimator ever leaves the initial B-radius, we double B and restart (repeating as necessary). To improve runtime, we reduce the number of iterations per stage from $O ( n )$ to $O ( { \sqrt { n } } )$ . The mathematically-analyzed version chooses W as powers of 2 starting at a constant w<sub>0</sub>; we instead consider W as all the j-th smallest absolute value of residuals at the current iteration, for j of the form $( d + 1 ) 2 ^ { i / 4 } \in [ d + 1 , n ]$ for integer i (this is a scale-invariant choice of W).

The main obstacle for practical performance is that the estimator only uses $S _ { w } ( b )$ with norm larger than a threshold, and the empirical performance is very sensitive to minor perturbations to this threshold. If the threshold is too large, the estimator may not leverage important signal; if the threshold is too small, the estimator may descend in erroneous directions.

We use empirical simulation to choose our thresholding parameter for our “standard” implementation of RB-Desc: At the start of the algorithm, we fit OLS and compute the residuals for all the samples. Our aim is to choose a threshold such that if the $X _ { i }$ were symmetrized independently of the $r _ { i } ( b )$ via random signing, then our threshold is larger than all the considered $\| S _ { w } ( b ) \| _ { 2 } ;$ this is because after random signing, there is no correlation anymore between $r _ { i } ( b )$ and $X _ { i }$ . This gives a quantitative indicator for the typical scale of $\| S _ { w } ( b ) \| _ { 2 }$ under the “null hypothesis” of no correlation. Thus, at the start of the algorithm, our implementation does this Rademacher signing 256 times, and chooses as the threshold the smallest number such that no $\| S _ { w } ( b ) \| _ { 2 }$ exceeds the threshold for at least 95% of those samplings. Given this threshold constant, we proceed with the iterative estimator normally (and do not change the constant again). This is the threshold-selection approach we generally recommend.

In the case of planted linear regression, the thresholds chosen by this “standard” variant are too conservative. In our “aggressive” variant, we use the same threshold as the “standard” variant to determine when a w is violated, yet we also consider a w as violated whenever it is at the scale of the current stage diameter $2 R _ { \ell }$ (or smaller). This is quite aggressive, so it does not throw away signal as much as the “standard” variant, yet it may also erroneously move in directions caused by noise (so we do not recommend this in general).

We also include a “hybrid” variant of our estimator, which is designed with planted linear regression in mind. This first runs the “standard” variant, then checks if the d + 1 smallest residuals are all zero (up to numerical tolerance), and uses this estimate if so. Otherwise, it tries the “aggressive” variant, and uses its output if its d + 1 smallest residuals are zero; else, we use the standard variant estimate. This corresponds to using the aggressive strategy only when it clearly attains exact recovery, and the standard strategy does not.

We emphasize there are significant modifications in these implementations, and that many ablations were run to choose and understand what modifications seemed conducive for practical performance. A natural hope for future work is to design other estimators with the same desirable guarantees, yet with fewer degrees of freedom that require tuning for practical performance.

## 5.1 Planted linear regression.

Our simulations evaluate the empirical likelihood of exactly recovering $\beta$ for planted linear regression (which is the definitive criterion for the estimator’s success in this case; for completeness we include figures showing the median coeficient error and average runtime in Section B). The model is as described in Definition 1.1, and we sample $\beta$ from the unit sphere. Each data point corresponds to 100 repetitions. We evaluate the three versions of our RB-Desc estimator, with comparisons to LAD and OLS. Recall that our theoretical guarantees show that a version of our RB-Desc estimator attains exact recovery when $m \gg d ^ { 3 / 4 } n ^ { 1 / 4 }$ , and that no convex M-estimator (including LAD and OLS) may attain exact recovery when $m \ll { \sqrt { d n } }$ . We accordingly observe a notable gap between what is possible with the best, practical versions of our RB-Desc estimators, and what is achieved by LAD and OLS.

![](images/c5d3553ffa7971e05188260025bbcd076cbf0c3627d8c00fafed4348b897e33f.jpg)  
(a)

![](images/0d7133e6593131d19fff0c873f8ead8836937711661ba74cc4defc5c1b201b1b.jpg)  
(b)

![](images/6778b0389d6b1af4312f8cddb21673bab784e3ada394917959b589081f93d8bc.jpg)  
(c)  
Figure 3: Empirical probability of recovery within $\left\| \widehat { \beta } - \beta \right\| _ { 2 } \leq 1 0 ^ { - 5 }$ (for 100 repetitions) for planted linear regression.

The aggressive variant of RB-Desc performs even better than the standard variant. As discussed in our implementation details, this is because the aggressive variant considers more scales of w violated (which may erroneously move in bad directions in general, but is helpful for stronger performance in the case of planted regression). While it is somewhat undesirable that these practical implementations of RB-Desc required substantial changes, we still believe this is an interesting demonstration of significant improvement relative to classical estimators.

Lastly, we remark that LAD achieves exact recovery when m is moderately large (consistent with the $m \gg \sqrt { n d }$ exact recovery guarantee given by [GL20, Theorem 3.2]), but OLS does not achieve exact recovery in these regimes. For some intuition, consider a toy 1-dimensional setting where m samples are exactly 0, and $n - m$ samples are $N ( 0 , 1 )$ . The empirical median (analogous to LAD) will output 0 when $m \gg { \sqrt { n } } ,$ yet the empirical mean (analogous to OLS) will not output 0 until $m = n$

## 5.2 Adaptive linear regression.

We analyze the performance of OLS, LAD, $L _ { \infty }$ regression, ASM, adaptive $L _ { q } ,$ , and RB-Desc on 5 symmetric noise distributions. The first coordinate of each $X _ { i }$ is 1 (representing an intercept term), and the remaining coordinates are sampled from $N ( 0 , I _ { d - 1 } )$ $\beta$ is sampled from the unit sphere. Each data point corresponds to 100 repetitions. For context, the average running time for Gaussian $N ( 0 , 1 )$ errors with $d = 1 0$ and $n = 1 0 { , } 0 0 0$ was 0.00158 seconds for OLS, 0.526 for LAD, 0.455 for $L _ { \infty } , 1 0 . 3$ for ASM, 3.67 for adaptive $L _ { q }$ , and 0.214 for RB-Desc (standard). The runtime is mostly similar for the other errors, except for in the location mixture simulation, the average runtime of RB-Desc is significantly larger (189.3 seconds).

We first explain why one should only expect RB-Desc and adaptive $L _ { q }$ to have strong performance for some of these distributions.

![](images/870165c3a6d82652f972c4204e939620d31fedd1974adf12e89d8b972f8b8c9f.jpg)

(a)  
![](images/aebef557850b6eac10a85de9e7481557a84f6b497523494e006ac565898ce5e6.jpg)  
(b)

![](images/34e6645a94a118aee75b848f80839b7cb69e7193d7239b8d442d806e3f75ec69.jpg)  
(d)

![](images/25f81936eedc30a88f7fabdf54a412270d27af5441991133b12f42c87fcffe16.jpg)  
(c)

![](images/6889713a4077400880ec593fa0e0c02658b9c537c93bb40f184ad22b4f80d5f9.jpg)  
(e)  
Figure 4: Average $\left\| \widehat { \beta } - \beta \right\| _ { 2 }$ error for adaptive linear regression.

RB-Desc is designed for heteroskedastic estimation, and it mostly extracts signal by trying to leverage small “bumps” in the noise density near 0. It achieves nearly the best performance (like many estimators) for standard Gaussian noise (Fig. 4a), and achieves the best performance for the two-component heteroskedastic mixture (Fig. 4d), where it has a big jump in performance once it has enough samples to detect the small-variance component. RB-Desc has suboptimal performance for uniform and smoothed uniform noise (Figs. 4b and 4c) where important signal is in the tails, since RB-Desc aims to leverage interior bumps and not tail information. Its performance for the location mixture (Fig. 4e) is predictably poor, since the noise distribution is not unimodal (and RB-Desc may unfortunately be trying to leverage bumps centered far away from 0).

On the other hand, adaptive $L _ { q }$ is designed to leverage information in the tails. This is provably suficient for near-optimal estimation with symmetric, log-concave noise (see Theorem 1.6), and this matches our empirical observations for the three such noise distributions (Figs. 4a to 4c). Adaptive $L _ { q }$ achieves nearly the best performance (like many estimators) for standard Gaussian noise (Fig. 4a); achieves strong performance for uniform noise (Fig. 4b), where only $L _ { \infty }$ has better performance (which is an MLE for uniform errors); and achieves the best performance for smoothed uniform noise (Fig. 4c), where our discussion in the introduction already indicated that OLS and $L _ { \infty }$ should have suboptimal guarantees. Since adaptive $L _ { q }$ leverages information in the tails, it is less able to leverage information in the interior of the noise distribution (this is also discussed in [KXZ24]), and hence achieves middling performance on the heteroskedastic mixture (Fig. 4d). For the location mixture (Fig. 4e), the performance is only worse than the ASM estimator; we remark that the plot shows their errors approaching each other for large n, yet we expect that ASM might have superior performance in an even larger regime of n if the variance in the mixture components was tuned diferently.

In terms of general performance, the adaptive $L _ { q }$ and ASM estimators seem to generally have the most versatile performance over these five simulations. We suspect the ASM estimator of [FKXS26] may have comparatively stronger performance for asymmetric noise, and its strong performance in Fig. 4e gives an example of a type of signal that neither of our implemented estimators can leverage even with symmetric noise. More broadly, there is no single estimator that simultaneously achieved the best performance for all five noise distributions. We suspect that some version of our ineficient estimator (Theorem 1.7) could leverage the signal in all of these distributions, yet we do not have a reasonable implementation of this estimator even for extremely small d (the estimator requires a search over a prohibitively fine net). Designing a natural, practical estimator that can leverage the signal in all such distributions would be an interesting line of work.

Some of these noise distributions have been studied with similar simulation setups in previouslydiscussed related work. [KXZ24] have linear regression simulations for Gaussian and uniform errors. [FKXS26] have linear regression simulations corresponding to all five of these styles of noise distributions other than uniform noise (the location mixture we use is exactly their location mixture). [CV26] have 1-dimensional mean estimation simulations corresponding to all five of these styles of noise distributions other than the location mixture; they also observe an analogous jump in improvement for heteroskedastic mixtures when n becomes large enough to detect the small-variance component.

## Acknowledgements

We thank Frederic Koehler and Gregory Valiant for discussions in earlier stages of this project. We thank Adityanand Guntuboyina and Richard Samworth for bringing related work to our attention. ChatGPT was used during the process of developing proofs and implementing simulations; we take full responsibility for the content of our paper. T.S. and S.C. are supported by T.S.’s NSF CAREER Grant no. 2143246 and the NSF AI institute for Foundations of Machine Learning (IFML). S.C. is also supported by a Two Sigma PhD Fellowship, and by Gregory Valiant’s Simons Foundation Investigator Award and NSF award AF-2341890.

## References

[Ada08] Radoslaw Adamczak. A tail inequality for suprema of unbounded empirical processes with applications to markov chains. Electronic Journal of Probability, 13:1000–1034, 2008.

[ADH<sup>+</sup>15] Jayadev Acharya, Ilias Diakonikolas, Chinmay Hegde, Jerry Zheng Li, and Ludwig Schmidt. Fast and near-optimal algorithms for approximating distributions by histograms. In Proceedings of the 34th ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database Systems, pages 249–263, 2015.

[ADLS17] Jayadev Acharya, Ilias Diakonikolas, Jerry Li, and Ludwig Schmidt. Sample-optimal density estimation in nearly-linear time. In Proceedings of the Twenty-Eighth Annual ACM-SIAM Symposium on Discrete Algorithms, pages 1278–1289. SIAM, 2017.

[AKPS24] Deeksha Adil, Rasmus Kyng, Richard Peng, and Sushant Sachdeva. Fast algorithms for ℓ<sub>p</sub>-regression. Journal of the ACM, 71(5):1–45, 2024.

[BB16] Yannick Baraud and Lucien Birgé. Rho-estimators for shape restricted density estimation. Stochastic Processes and their Applications, 126(12):3888–3912, 2016.

[BB18] Yannick Baraud and Lucien Birgé. Rho-estimators revisited: General theory and applications. Annals of Statistics, 46(6B), 2018.

[BB20] Matthew Brennan and Guy Bresler. Reducibility and statistical-computational gaps from secret leakage. In Conference on Learning Theory, pages 648–847. PMLR, 2020.

[BBH<sup>+</sup>21] Matthew S Brennan, Guy Bresler, Sam Hopkins, Jerry Li, and Tselil Schramm. Statistical query algorithms and low degree tests are almost equivalent. In Mikhail Belkin and Samory Kpotufe, editors, Proceedings of Thirty Fourth Conference on Learning Theory, volume 134 of Proceedings of Machine Learning Research, pages 774–774. PMLR, 15–19 Aug 2021. URL: https://proceedings.mlr.press/v134/brennan21a.html.

[BBS17] Yannick Baraud, Lucien Birgé, and Mathieu Sart. A new method for estimation and model selection: ρ-estimation. Inventiones mathematicae, 207(2):425–517, 2017.

[BCW25] Guy Blanc, Clément L Canonne, and Erik Waingarten. Instance-optimal uniformity testing and tracking. In 2025 IEEE 66th Annual Symposium on Foundations of Computer Science (FOCS), pages 1351–1365. IEEE, 2025.

[BDT24] Rares-Darius Buhai, Jingqiu Ding, and Stefan Tiegel. Computational-statistical gaps for improper learning in sparse linear regression. In Conference on Learning Theory, pages 752–771. PMLR, 2024.

[Ber78] Rudolf Beran. An eficient and robust adaptive estimator of location. The Annals of Statistics, pages 292–313, 1978.

[Bic82] Peter J Bickel. On adaptive estimation. The Annals of Statistics, pages 647–671, 1982.

[Bir83] Lucien Birgé. Approximation dans les espaces métriques et théorie de l’estimation. Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete, 65(2):181–237, 1983.

[BJKK17] Kush Bhatia, Prateek Jain, Parameswaran Kamalaruban, and Purushottam Kar. Consistent robust regression. Advances in Neural Information Processing Systems, 30, 2017.

[BM98] Lucien Birgé and Pascal Massart. Minimum contrast estimators on sieves: exponential bounds and rates of convergence. Bernoulli, 4(3):329–375, 1998.

[BNOP21] Alankrita Bhatt, Bobak Nazer, Or Ordentlich, and Yury Polyanskiy. Informationdistilling quantizers. IEEE Transactions on Information Theory, 67(4):2472–2487, 2021.

[BPP18] Guy Bresler, Sung Min Park, and Madalina Persu. Sparse pca from sparse linear regression. In Advances in Neural Information Processing Systems, volume 31, 2018.

[Cat12] Olivier Catoni. Challenging the empirical mean and empirical variance: a deviation study. In Annales de l’IHP Probabilités et statistiques, volume 48, pages 1148–1185, 2012.

[CB26] Siddhant Chaudhary and Aditya Bhaskara. Linear regression with heteroskedastic errors. In Forty-Second Annual Conference on Uncertainty in Artificial Intelligence, 2026. URL: https://openreview.net/forum?id=gK8yBRaUik.

[CDKL14] Flavio Chierichetti, Anirban Dasgupta, Ravi Kumar, and Silvio Lattanzi. Learning entangled single-sample gaussians. In Proceedings of the twenty-fifth annual ACM-SIAM symposium on Discrete algorithms, pages 511–522. SIAM, 2014.

[CDSS13] Siu-On Chan, Ilias Diakonikolas, Xiaorui Sun, and Rocco A Servedio. Learning mixtures of structured distributions over discrete domains. In Proceedings of the twenty-fourth annual ACM-SIAM symposium on Discrete algorithms, pages 1380–1394. SIAM, 2013.

[CDSS14] Siu-On Chan, Ilias Diakonikolas, Rocco A Servedio, and Xiaorui Sun. Eficient density estimation via piecewise polynomial approximation. In Proceedings of the forty-sixth annual ACM symposium on Theory of computing, pages 604–613, 2014.

[CFU24] Matias D Cattaneo, Yingjie Feng, and William G Underwood. Uniform inference for kernel density estimators with dyadic data. Journal of the American Statistical Association, 119(548):2695–2708, 2024.

[CL15] T Tony Cai and Mark G Low. A framework for estimation of convex functions. Statistica Sinica, pages 423–456, 2015.

[CL26] Spencer Compton and Jerry Li. Density estimation for hellinger via minimum-distance estimators: mixtures of gaussians, log-concave, and more. In Steve Hanneke and Tor Lattimore, editors, Proceedings of Thirty Ninth Conference on Learning Theory, volume 336 of Proceedings of Machine Learning Research, pages 1436–1475. PMLR, 29 Jun–03 Jul 2026.

[CLM<sup>+</sup>26] Spencer Compton, Gábor Lugosi, Jaouad Mourtada, Jian Qian, and Nikita Zhivotovskiy. Ratio covers of convex sets and optimal mixture density estimation. arXiv preprint arXiv:2602.16142, 2026.

[CMW20] Michael Celentano, Andrea Montanari, and Yuchen Wu. The estimation error of general first order methods. In Conference on Learning Theory, pages 1078–1141. PMLR, 2020.

[CV24] Spencer Compton and Gregory Valiant. Near-optimal mean estimation with unknown, heteroskedastic variances. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing, pages 194–200, 2024.

[CV26] Spencer Compton and Gregory Valiant. Attainability of two-point testing rates for finite-sample location estimation. The Annals of Statistics, 54(3):1474–1501, 2026.

[DGK<sup>+</sup>25] Ilias Diakonikolas, Chao Gao, Daniel Kane, John Laferty, and Ankit Pensia. Information-computation tradeofs for noiseless linear regression with oblivious contamination. Advances in Neural Information Processing Systems, 38:108089–108126, 2025.

[DGK<sup>+</sup>26] Ilias Diakonikolas, Chao Gao, Daniel M Kane, Ankit Pensia, and Dong Xie. Robust regression with adaptive contamination in response: Optimal rates and computational barriers. arXiv preprint arXiv:2604.04228, 2026.

[DK23] Ilias Diakonikolas and Daniel M Kane. Algorithmic high-dimensional robust statistics. Cambridge university press, 2023.

[DKLP25] Ilias Diakonikolas, Daniel M Kane, Sihan Liu, and Thanasis Pittas. Entangled mean estimation in high dimensions. In Proceedings of the 57th Annual ACM Symposium on Theory of Computing, pages 1680–1688, 2025.

[DL87] David L. Donoho and Richard C. Liu. Geometrizing rates of convergence, i. Technical Report 137a, Department of Statistics, University of California, Berkeley, 1987.

[DL91a] David L Donoho and Richard C Liu. Geometrizing rates of convergence, ii. The Annals of Statistics, pages 633–667, 1991.

[DL91b] David L Donoho and Richard C Liu. Geometrizing rates of convergence, iii. The Annals of Statistics, pages 668–701, 1991.

[DLLZ23] Luc Devroye, Silvio Lattanzi, Gábor Lugosi, and Nikita Zhivotovskiy. On mean estimation for heteroscedastic random variables. In Annales de l’Institut Henri Poincare (B) Probabilites et statistiques, volume 59, pages 1–20. Institut Henri Poincaré, 2023.

[DNNB23] Aniket Das, Dheeraj M Nagaraj, Praneeth Netrapalli, and Dheeraj Baby. Near optimal heteroscedastic regression with symbiotic learning. In The Thirty Sixth Annual Conference on Learning Theory, pages 3696–3757. PMLR, 2023.

[dNS21] Tommaso d’Orsi, Gleb Novikov, and David Steurer. Consistent regression when oblivious outliers overwhelm. In International Conference on Machine Learning, pages 2297–2306. PMLR, 2021.

[DR24] John C Duchi and Feng Ruan. The right complexity measure in locally private estimation: It is not the fisher information. The Annals of Statistics, 52(1):1–51, 2024.

[EHE23] Ayoub El Hanchi and Murat A Erdogdu. Optimal excess risk bounds for empirical risk minimization on p-norm linear regression. Advances in Neural Information Processing Systems, 36:24368–24387, 2023.

[FGR<sup>+</sup>17] Vitaly Feldman, Elena Grigorescu, Lev Reyzin, Santosh S Vempala, and Ying Xiao. Statistical algorithms and a lower bound for detecting planted cliques. Journal of the ACM (JACM), 64(2):1–37, 2017.

[FKQR21] Dylan J Foster, Sham M Kakade, Jian Qian, and Alexander Rakhlin. The statistical complexity of interactive decision making. arXiv:2112.13487, 2021. arXiv:2112.13487.

[FKXS26] Oliver Y Feng, Yu-Chun Kao, Min Xu, and Richard J Samworth. Optimal convex m-estimation via score matching. The Annals of Statistics, 54(1):408–441, 2026.

[GHP24] Shivam Gupta, Samuel Hopkins, and Eric Price. Beyond catoni: Sharper rates for heavy-tailed and robust mean estimation. In The Thirty Seventh Annual Conference on Learning Theory, pages 2232–2269. PMLR, 2024.

[GL20] Chao Gao and John Laferty. Model repair: Robust recovery of over-parameterized statistical models. arXiv preprint arXiv:2005.09912, 2020.

[GLP23] Shivam Gupta, Jasper CH Lee, and Eric Price. Finite-sample symmetric mean estimation with fisher information rate. In The Thirty Sixth Annual Conference on Learning Theory, pages 4777–4830. PMLR, 2023.

[GLPV22] Shivam Gupta, Jasper Lee, Eric Price, and Paul Valiant. Finite-sample maximum likelihood estimation of location. Advances in Neural Information Processing Systems, 35:30139–30149, 2022.

[GLPV23] Shivam Gupta, Jasper Lee, Eric Price, and Paul Valiant. Minimax-optimal location estimation. Advances in Neural Information Processing Systems, 36:900–915, 2023.

[Hop18] Samuel Hopkins. Statistical inference and the sum of squares method. Cornell University, 2018.

[HRRS86] Frank Hampel, Elvezio Ronchetti, Peter Rousseeuw, and Werner Stahel. Robust Statistics: The Approach Based on Influence Functions. 03 1986. doi:10.1002/ 9781118186435.

[HS14] Daniel Hsu and Sivan Sabato. Heavy-tailed regression with a generalized median-ofmeans. In International Conference on Machine Learning, pages 37–45. PMLR, 2014.

[HSS26] Yanjun Han, Abhishek Shetty, and Jacob Shkrob. An empirical bayes perspective on heteroskedastic mean estimation. In The Thirty Ninth Annual Conference on Learning Theory, pages 3076–3108. PMLR, 2026.

[IH13] Ildar Abdulovich Ibragimov and Rafail Zalmanovich Has’minskii. Statistical estimation: asymptotic theory. Springer Science & Business Media, 2013.

[JN09] Anatoli B Juditsky and Arkadii S Nemirovski. Nonparametric estimation by convex programming. Annals of Statistics, pages 2278–2300, 2009.

[Jon02] MC Jones. On khintchine’s theorem and its place in random variate generation. The American Statistician, 56(4):304–307, 2002.

[KPJ25] Hadi Kazemi, Ankit Pensia, and Varun Jog. The sample complexity of distributed simple binary hypothesis testing under information constraints. In Nika Haghtalab and Ankur Moitra, editors, Proceedings of Thirty Eighth Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 3213–3214. PMLR, 30 Jun–04 Jul 2025.

[KS83] HL Koul and V Susarla. Adaptive estimation in linear regression. Statistics & Risk Modeling, 1(4-5):379–400, 1983.

[KXZ24] Yu-Chun Kao, Min Xu, and Cun-Hui Zhang. Choosing the p in lp loss: adaptive rates for symmetric mean estimation. In The Thirty Seventh Annual Conference on Learning Theory, pages 2795–2839. PMLR, 2024.

[Lah21] Nilanjana Laha. Adaptive estimation in symmetric location model under log-concavity constraint. Electronic Journal of Statistics, 15(1):2939–3014, 2021.

[LC73] Lucien Le Cam. Convergence of estimates under dimensionality restrictions. The Annals of Statistics, pages 38–53, 1973.

[LCY00] Lucien Marie Le Cam and Grace Lo Yang. Asymptotics in statistics: some basic concepts. Springer Science & Business Media, 2000.

[Li24] Jerry Li. Entangled mean estimation in high dimensions. In Open Problems Session at Workshop on New Frontiers in Robust Statistics, TTI-Chicago, 2024.

[LL05] Pui-Ying Lai and Stephen M S Lee. An overview of asymptotic properties of lp regression under general classes of error distributions. Journal of the American Statistical Association, 100(470):446–458, 2005.

[LL23] Hao Lia and Seokho Lee. Least clipped absolute deviation for robust regression using skipped median. Communications for Statistical Applications and Methods, 30(2):135– 147, 2023.

[LMSY24] Siqi Liu, Sidhanth Mohanty, Tselil Schramm, and Elizabeth Yang. Testing thresholds for high-dimensional sparse random geometric graphs. SIAM Journal on Computing, pages STOC22–125–STOC22–181, 2024. doi:10.1137/23M1545203.

[Lou25] Sirine Louati. Performance of the empirical median for location estimation in heteroscedastic settings. arXiv preprint arXiv:2501.16956, 2025.

[LV22a] Jasper CH Lee and Paul Valiant. Optimal sub-gaussian mean estimation in R. In 2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS), pages 672–683. IEEE, 2022.

[LV22b] Jasper CH Lee and Paul Valiant. Optimal sub-gaussian mean estimation in very high dimensions. In 13th Innovations in Theoretical Computer Science Conference (ITCS 2022). Schloss-Dagstuhl-Leibniz Zentrum für Informatik, 2022.

[LY20] Yingyu Liang and Hui Yuan. Learning entangled single-sample gaussians in the subsetof-signals model. In Conference on Learning Theory, pages 2712–2737. PMLR, 2020.

[Man85] Charles F Manski. Semiparametric analysis of discrete response: Asymptotic properties of the maximum score estimator. Journal of econometrics, 27(3):313–333, 1985.

[Mas90] Pascal Massart. The tight constant in the dvoretzky-kiefer-wolfowitz inequality. The annals of Probability, pages 1269–1283, 1990.

[MSB<sup>+</sup>26] Arian Maleki, Subhabrata Sen, Sivaraman Balakrishnan, Verena Zuber, Chao Gao, Rishabh Dudeja, Christos Thrampoulidis, Anru Zhang, Weijie Su, Jason M Klusowski, et al. High-dimensional statistics: Reflections on progress and open problems. arXiv preprint arXiv:2605.05076, 2026.

[MW25] Andrea Montanari and Alexander S Wein. Equivalence of approximate message passing and low-degree polynomials in rank-one matrix estimation. Probability Theory and Related Fields, 191(1):181–233, 2025.

[NRS07] Kobbi Nissim, Sofya Raskhodnikova, and Adam Smith. Smooth sensitivity and sampling in private data analysis. In Proceedings of the thirty-ninth annual ACM symposium on Theory of computing, pages 75–84, 2007.

[PJL22] Ankit Pensia, Varun Jog, and Po-Ling Loh. Estimating location parameters in sampleheterogeneous distributions. Information and Inference: A Journal of the IMA, 11(3):959–1036, 2022.

[PJL23] Ankit Pensia, Varun Jog, and Po-Ling Loh. Communication-constrained hypothesis testing: Optimality, robustness, and reverse data processing inequalities. IEEE Transactions on Information Theory, 70(1):389–414, 2023.

[Pré73] András Prékopa. On logarithmic concave measures and functions. Acta Sci. Math., 34:335, 1973.

[PW25] Yury Polyanskiy and Yihong Wu. Information theory: From coding to learning. Cambridge university press, 2025.

[PW26] Yury Polyanskiy and Yihong Wu. Dualizing Le Cam’s method for functional estimation I: General theory. The Annals of Statistics, 54(1):1 – 24, 2026. doi:10.1214/25-AOS2498.

[Rou82] Peter J Rousseeuw. Most robust m-estimators in the infinitesimal sense. Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete, 61(4):541–551, 1982.

[S<sup>+</sup>56] Charles Stein et al. Eficient nonparametric testing and estimation. In Proceedings of the third Berkeley symposium on mathematical statistics and probability, volume 1, pages 187–195, 1956.

[Sac75] Jerome Sacks. An asymptotically eficient sequence of estimators of a location parameter. The Annals of Statistics, pages 285–298, 1975.

[SBRJ19] Arun Sai Suggala, Kush Bhatia, Pradeep Ravikumar, and Prateek Jain. Adaptive hard thresholding for near-optimal consistent robust regression. In Conference on Learning Theory, pages 2892–2897. PMLR, 2019.

[Sto75] Charles J Stone. Adaptive maximum likelihood estimators of a location parameter. The Annals of Statistics, pages 267–284, 1975.

[Tan22] Jennifer Tang. Divergence covering. PhD thesis, Massachusetts Institute of Technology, 2022.

[TJSO14] Efthymios Tsakonas, Joakim Jaldén, Nicholas D Sidiropoulos, and Björn Ottersten. Convergence of the huber regression m-estimate in the presence of dense outliers. IEEE Signal Processing Letters, 21(10):1211–1214, 2014.

[vdVW96] Aad W van der Vaart and Jon A Wellner. Weak convergence and empirical processes: with applications to statistics. Springer, 1996.

[VE70] Constance Van Eeden. Eficiency-robust estimation of location. The Annals of Mathematical Statistics, 41(1):172–181, 1970.

[Wei25] Alexander S Wein. Computational complexity of statistics: New insights from low-degree polynomials. arXiv preprint arXiv:2506.10748, 2025.

[Xia19] Dong Xia. Non-asymptotic bounds for percentiles of independent non-identical random variables. Statistics & Probability Letters, 152:111–120, 2019.

[YL20] Hui Yuan and Yingyu Liang. Learning entangled single-sample distributions via iterative trimming. In International Conference on Artificial Intelligence and Statistics, pages 2666–2676. PMLR, 2020.

[YN24] Yufei Yi and Matey Neykov. Non-asymptotic bounds for the $\ell _ { \infty }$ estimator in linear regression with uniform noise. Bernoulli, 30(1):534–553, 2024.

## A Subset-of-Signals statistical lower bound

Theorem A.1 (Subset-of-Signals lower bound). There exist universal constants $c , c _ { 0 } > 0$ such that the following holds. Suppose $1 \leq d , m \leq n$ . For every estimator $\widehat { \beta } = \widehat { \beta } ( ( X _ { i } , Y _ { i } ) _ { i = 1 } ^ { n } )$ that is not given the standard-deviation vector, there exists a $\pmb { \sigma } = ( \sigma _ { 1 } , \ldots , \sigma _ { n } )$ with at least m values of $\sigma _ { i } = 1$ , and a $\beta \in \mathbb { R } ^ { d }$ such that, under

$$
X _ { i } \sim N ( 0 , I _ { d } ) , \qquad Y _ { i } = X _ { i } ^ { \top } \beta + \varepsilon _ { i } , \qquad \varepsilon _ { i } \ \sim N ( 0 , \sigma _ { i } ^ { 2 } ) ,
$$

it holds that

$$
\mathbb { P } \left( \| \widehat { \beta } - \beta \| _ { 2 } \geq c r _ { n , d , m } \right) \geq \frac { 1 } { 4 } ,
$$

where

$$
r _ { n , d , m } : = \left\{ \begin{array} { l l } { \displaystyle \frac { \sqrt { n } d ^ { 3 / 2 } } { m ^ { 2 } } , } & { m < c _ { 0 } d ^ { 3 / 4 } n ^ { 1 / 4 } , } \\ { \displaystyle \sqrt { \frac { d } { n } } \left( \frac { n } { m } \right) ^ { 2 / 3 } , } & { m \geq c _ { 0 } d ^ { 3 / 4 } n ^ { 1 / 4 } . } \end{array} \right.
$$

Proof. This proof will follow by invoking the general Hellinger modulus lower bound of Theorem 1.8, and then analyzing the Hellinger modulus for a 1-dimensional mixture with two variances (the latter is mostly calculations, and is not conceptually novel beyond related analyses in [LY20, HSS26]).

We first analyze the Hellinger distance for translations of such mixtures. Let

$$
\varphi _ { \tau } ( x ) : = \frac { 1 } { \sqrt { 2 \pi } \tau } \exp \left( - \frac { x ^ { 2 } } { 2 \tau ^ { 2 } } \right)
$$

be the density of $N ( 0 , \tau ^ { 2 } )$ , and define

$$
p _ { q , \sigma } : = q \varphi _ { 1 } + ( 1 - q ) \varphi _ { \sigma } .
$$

This density is symmetric and unimodal. We bound the distance between translations:

Lemma A.2 (Translation bound). For $0 < q \leq 1 / 2 , \sigma \geq 2$ , and $| t | \leq \sigma ,$

$$
H _ { p q , \sigma } ( t ) \lesssim \left[ q ^ { 2 } \sigma \operatorname* { m i n } \{ t ^ { 2 } , 1 \} + \frac { t ^ { 2 } } { \sigma ^ { 2 } } \right] .
$$

Proof. We use

$$
\mathrm { H } ^ { 2 } ( f , g ) \lesssim \chi ^ { 2 } ( g \| f ) = \int { \frac { ( g ( x ) - f ( x ) ) ^ { 2 } } { f ( x ) } } .
$$

Since $p _ { q , \sigma } \geq ( 1 - q ) \varphi _ { \sigma } \geq \varphi _ { \sigma } / 2$

$$
H _ { p q , \sigma } ( t ) \lesssim \int \frac { \left( q \left( \varphi _ { 1 } ( x - t ) - \varphi _ { 1 } ( x ) \right) + \left( \varphi _ { \sigma } ( x - t ) - \varphi _ { \sigma } ( x ) \right) \right) ^ { 2 } } { \varphi _ { \sigma } ( x ) } \lesssim q ^ { 2 } A _ { \sigma } ( t ) + B _ { \sigma } ( t ) ,
$$

where

$$
A _ { \sigma } ( t ) : = \int \frac { ( \varphi _ { 1 } ( x - t ) - \varphi _ { 1 } ( x ) ) ^ { 2 } } { \varphi _ { \sigma } ( x ) } d x , \qquad B _ { \sigma } ( t ) : = \int \frac { ( \varphi _ { \sigma } ( x - t ) - \varphi _ { \sigma } ( x ) ) ^ { 2 } } { \varphi _ { \sigma } ( x ) } d x .
$$

The value of $B _ { \sigma } ( t )$ has the closed-form

$$
B _ { \sigma } ( t ) = \chi ^ { 2 } \bigl ( N ( t , \sigma ^ { 2 } ) \bigr | \bigr | N ( 0 , \sigma ^ { 2 } ) \bigr ) = e ^ { t ^ { 2 } / \sigma ^ { 2 } } - 1 \stackrel { < } { \sim } \frac { t ^ { 2 } } { \sigma ^ { 2 } } ,
$$

using $| t | \leq \sigma$ . For the $A _ { \sigma } ( t )$ term, it will be helpful to write $z = x - u$ and bound

$$
\begin{array} { r l r } {  { \frac { \varphi _ { 1 } ( x - u ) ^ { 2 } } { \varphi _ { \sigma } ( x ) } = \frac { \sigma } { \sqrt { 2 \pi } } \exp ( - z ^ { 2 } + \frac { ( z + u ) ^ { 2 } } { 2 \sigma ^ { 2 } } ) } } \\ & { \leq \frac { \sigma } { \sqrt { 2 \pi } } \exp ( - z ^ { 2 } + \frac { z ^ { 2 } } { \sigma ^ { 2 } } + \frac { u ^ { 2 } } { \sigma ^ { 2 } } ) } \\ & { \lesssim \sigma e ^ { - 3 z ^ { 2 } / 4 } . \ } & { \ } & { \ \mathrm { ( v i a ~ } \sigma \geq 2 \mathrm { ~ a n d ~ } | u | \leq \sigma ) } \end{array}
$$

Consequently, for $| u | \le \sigma$

$$
\int \frac { \varphi _ { 1 } ( x - u ) ^ { 2 } } { \varphi _ { \sigma } ( x ) } d x \lesssim \sigma , \qquad \int \frac { \varphi _ { 1 } ^ { \prime } ( x - u ) ^ { 2 } } { \varphi _ { \sigma } ( x ) } d x \lesssim \sigma ,\tag{114}
$$

because $\varphi _ { 1 } ^ { \prime } ( x - u ) = - ( x - u ) \varphi _ { 1 } ( x - u ) = - z \varphi _ { 1 } ( z )$ , and both $\int e ^ { - 3 z ^ { 2 } / 4 } d z$ and $\int z ^ { 2 } e ^ { - 3 z ^ { 2 } / 4 } d z$ are finite constants.

If $| t | \leq 1$ , using Cauchy-Schwarz gives, for each $x ,$

$$
\big ( \varphi _ { 1 } ( x - t ) - \varphi _ { 1 } ( x ) \big ) ^ { 2 } \leq | t | \int _ { 0 } ^ { | t | } \varphi _ { 1 } ^ { \prime } ( x - \mathrm { s g n } ( t ) u ) ^ { 2 } ~ d u .
$$

Divide by $\varphi _ { \sigma } ( x )$ , integrate in $x ,$ and use (114):

$$
A _ { \sigma } ( t ) \leq | t | \int _ { 0 } ^ { | t | } \left[ \int \frac { \varphi _ { 1 } ^ { \prime } ( x - \mathrm { s g n } ( t ) u ) ^ { 2 } } { \varphi _ { \sigma } ( x ) } d x \right] d u \lesssim \sigma t ^ { 2 } .
$$

Otherwise, when $1 < | t | \leq \sigma$ , then $( a - b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ and (114) give

$$
A _ { \sigma } ( t ) \leq 2 \int \frac { \varphi _ { 1 } ( x - t ) ^ { 2 } } { \varphi _ { \sigma } ( x ) } d x + 2 \int \frac { \varphi _ { 1 } ( x ) ^ { 2 } } { \varphi _ { \sigma } ( x ) } d x \lesssim \sigma .
$$

Combining the two cases yields

$$
\begin{array} { r } { A _ { \sigma } ( t ) \leq C \sigma \operatorname* { m i n } \{ t ^ { 2 } , 1 \} . } \end{array}
$$

This gives the desired guarantee.

This enables the Hellinger modulus bound:

Lemma A.3 (Piecewise Hellinger modulus bound). There exists a universal constant $C > 0$ such that, for every $0 < a < 1$ and $0 < q \le 1 / 2$ , one can choose $\sigma \geq 2$ for which

$$
\omega _ { p _ { q , \sigma } } ( C a ) \geq \left\{ \begin{array} { l l } { 2 a ^ { 3 / 2 } q ^ { - 2 } , } & { q \leq a ^ { 3 / 4 } , } \\ { a ^ { 1 / 2 } q ^ { - 2 / 3 } , } & { q \geq a ^ { 3 / 4 } . } \end{array} \right.
$$

Proof. Case (i) $q \geq a ^ { 3 / 4 }$ . Choose

$$
\sigma = 2 q ^ { - 2 / 3 } , \qquad t = a ^ { 1 / 2 } q ^ { - 2 / 3 } .
$$

This satisfies $\sigma \geq 2$ and $t \leq 1$ ; invoking Lemma $\mathrm { A . 2 }$ yields

$$
H _ { p _ { q , \sigma } } ( t ) \lesssim \bigl ( q ^ { 2 } \sigma t ^ { 2 } + t ^ { 2 } / \sigma ^ { 2 } \bigr ) \leq C a .
$$

Case (ii): $q \leq a ^ { 3 / 4 }$ . Choose

$$
\sigma = 2 a q ^ { - 2 } , \qquad t = \sigma \sqrt { a } = 2 a ^ { 3 / 2 } q ^ { - 2 } .
$$

This satisfies $\sigma \geq 2$ and $t \leq \sigma ;$ invoking Lemma $\mathrm { A . 2 }$ yields

$$
H _ { p _ { q , \sigma } } ( t ) \lesssim \bigl ( q ^ { 2 } \sigma \operatorname* { m i n } \{ t ^ { 2 } , 1 \} + t ^ { 2 } / \sigma ^ { 2 } \bigr ) \lesssim a .
$$

Concluding the lower bound. Let $\kappa \geq 1$ be a suficiently large universal constant. We first focus on the case where $m \le n / ( 2 \kappa )$ , and set

$$
q : = \kappa \frac { m } { n } \leq \frac { 1 } { 2 } .
$$

Independently for each observation, draw a hidden scale label: the noise standard deviation is 1 with probability $q$ and is $\sigma$ otherwise. When the labels are hidden, the errors are i.i.d. $p _ { q , \sigma }$

Let

$$
a : = \frac { c _ { 1 } } { C _ { 2 } } \frac { d } { n } ,
$$

where $c _ { 1 }$ is the constant in Theorem 1.8 and $C _ { 2 }$ is the constant in Lemma A.3. Choose σ according to Lemma $\mathrm { A . 3 }$ with parameter $a .$ . Theorem 1.8 then implies that with probability at least $3 / 8$ , any estimator incurs error of order

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { a ^ { 3 / 2 } q ^ { - 2 } , } & { q \leq a ^ { 3 / 4 } , } \\ { a ^ { 1 / 2 } q ^ { - 2 / 3 } , } & { q \geq a ^ { 3 / 4 } . } \end{array} \right. } \end{array}\tag{115}
$$

We now relate this error in the i.i.d. setting to the Subset-of-Signals setting where at least m scale labels are 1. If $N _ { 1 } \sim \mathrm { B i n } ( n , q )$ denotes the number of observations with $\sigma _ { i } = 1$ , then $\mathrm { E } N _ { 1 } = \kappa m$ , and a Chernof bound gives

$$
\mathbb { P } ( N _ { 1 } < m ) \leq e ^ { - c \kappa m } \leq \frac { 1 } { 8 } .
$$

for suficiently large κ. For any estimator, the failure probability on i.i.d. samples from $p _ { q , \sigma ; \ l }$ is the average of its failure probabilities over the hidden scale vectors. Since our use of Theorem 1.8 gave a failure probability of at least $\frac { 3 } { 8 }$ , then subtracting the event $\{ N _ { 1 } < m \}$ shows that there exists a fixed scale vector with at least m entries having variance 1, where the estimator fails with probability at least $\textstyle { \frac { 3 } { 8 } } - { \frac { 1 } { 8 } } = { \frac { 1 } { 4 } }$

We now substitute our values into the error radius given by (115). Since $q \asymp m / n$ and $a \asymp d / n$ the two quantities in (115) are

$$
a ^ { 3 / 2 } q ^ { - 2 } \asymp \left( \frac { d } { n } \right) ^ { 3 / 2 } \left( \frac { n } { m } \right) ^ { 2 } = \frac { \sqrt { n } d ^ { 3 / 2 } } { m ^ { 2 } } ,
$$

and

$$
a ^ { 1 / 2 } q ^ { - 2 / 3 } \asymp \sqrt { \frac { d } { n } } \left( \frac { n } { m } \right) ^ { 2 / 3 } .
$$

The branching point $q \asymp a ^ { 3 / 4 }$ is equivalent to

$$
m \asymp d ^ { 3 / 4 } n ^ { 1 / 4 } .
$$

Changing the branch point by a fixed multiplicative constant changes the displayed piecewise rate by only a fixed multiplicative constant, so this proves the theorem for $m \le n / ( 2 \kappa )$

For the $m > n / ( 2 \kappa )$ case, we will simply consider the instance where all variances are equal to 1. The classical lower bound in this setting is $c \sqrt { d / n }$ (this would also follow from invoking Theorem 1.8). By choosing $c _ { 0 }$ in the theorem statement suficiently small, the transition $c _ { 0 } d ^ { 3 / 4 } n ^ { 1 / \bar { 4 } }$ lies below $n / ( 2 \kappa )$ . Hence, this case corresponds to the second branch, and

$$
{ \sqrt { \frac { d } { n } } } \left( { \frac { n } { m } } \right) ^ { 2 / 3 } \leq ( 2 \kappa ) ^ { 2 / 3 } { \sqrt { \frac { d } { n } } } .
$$

Thus, this lower bound yields our desired result.

## B Additional simulation figures

In this section, we provide additional simulation figures for planted linear regression. We find these figures less insightful than Fig. 3, but we still include them for completeness. The experiment setup is the same as Section 5.1, and each data point corresponds to 100 repetitions.

![](images/56cde1341502fd023714d7dcb28db59999e6a84f1c6d2318ad5c62f78a103a22.jpg)  
(a)

![](images/0c1359ed00848b29e096b737df608d278227e4954c7334903b0b899951257e6b.jpg)  
(b)

![](images/be49328fb6e15a7eb5839940a423f9330d293de1a9371cdc582f83719e259b2b.jpg)  
(c)

Figure 5: Median $\left\| \widehat { \beta } - \beta \right\| _ { 2 }$ error for planted linear regression.  
![](images/315013559462829d025b55b48fe8980a580adc5f112162c5056b51cf06897042.jpg)  
(a)

![](images/1e8695064c38e3236392f8f90ea3db66c5cf735d96a6f9137d20a3d0e817e359.jpg)  
(b)

![](images/82fecff60436162888feb7e71144e4539017c868ad822e8d44167371dd41de12.jpg)  
(c)  
Figure 6: Average runtime for planted linear regression.

Fig. 5 illustrates the median error of the estimators as m varies. The diferences in median error between LAD and RB-Desc in the large-m regime are likely only due to numerical precision diferences. Plotting the median error seemed more informative than the mean error, as the latter was functionally a (harder to read) transformation of the likelihood of exact recovery (seen in Fig. 3). Fig. 6 plots the average runtime of the estimators as m varies.

## C Proof of uniform convergence results

## C.1 Proof of Proposition 2.5

In this section, we use relatively standard techniques to prove the desired uniform convergence result. We begin by stating a VC-style result (Lemma C.1), then in Section C.1.1 we show how this implies our desired uniform convergence result (Proposition 2.5), and finally in Section C.1.2 we prove Lemma C.1.

Lemma C.1 (VC-selected counts and feature sums). Let $W _ { i } = ( X _ { i } , Y _ { i } )$ be independent random elements $o f \mathbb { R } ^ { d } \times \mathbb { R }$ . Assume that $X _ { 1 } , \ldots , X _ { n }$ are i.i.d. copies of a covariate X; the conditional laws of $Y _ { i }$ given $X _ { i }$ may difer with i. Let A be a pointwise measurable class of measurable subsets of $\mathbb { R } ^ { d + 1 }$ with VC dimension at most $V \geq 1$ . For $A \in { \mathcal { A } }$ , define

$$
N ( A ) = \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ W _ { i } \in A \} , \qquad Q ( A ) = \sum _ { i = 1 } ^ { n } \mathbb { P } ( W _ { i } \in A ) ,
$$

and

$$
Z ( A ) = \sum _ { i = 1 } ^ { n } { \bigl ( } X _ { i } \mathbf { 1 } \{ W _ { i } \in A \} - \operatorname { E } [ X _ { i } \mathbf { 1 } \{ W _ { i } \in A \} ] { \bigr ) } .
$$

For $\varepsilon \in ( 0 , 1 / 2 )$ , put

$$
\ell _ { \varepsilon } = \operatorname* { m a x } \{ 1 , \log ( e n / \varepsilon ) \} , \qquad D = d + V .
$$

There is a universal constant C such that, with probability at least $1 - \varepsilon _ { i }$ , simultaneously for all $A \in { \mathcal { A } }$

$$
| N ( A ) - Q ( A ) | \leq C \big ( \sqrt { V Q ( A ) \ell _ { \varepsilon } } + V \ell _ { \varepsilon } \big ) .\tag{116}
$$

Further, if X satisfies (1) and (2), then also simultaneously for all $A \in { \mathcal { A } }$

$$
\begin{array} { r } { \| Z ( A ) \| _ { 2 } \le C _ { K , \kappa } \ell _ { \varepsilon } ^ { 2 } \big ( \sqrt { D ( N ( A ) + D ) } + D \big ) . } \end{array}\tag{117}
$$

## C.1.1 Concluding Proposition 2.5

Proof of Proposition 2.5. Let $W _ { i } = ( X _ { i } , y _ { i } )$ , and let A be the union of the following three residualset classes, with $b \in B _ { 2 } ( B )$ and $w > 0$ varying over their full ranges:

$$
\begin{array} { r l r l } & { \{ ( x , y ) : \left| y - x ^ { \top } b \right| \leq w \} , } \\ & { \{ ( x , y ) : 0 < y - x ^ { \top } b \leq w \} , } & & { \{ ( x , y ) : - w \leq y - x ^ { \top } b < 0 \} } \end{array}
$$

Since each set is the intersection of two halfspaces in $\mathbb { R } ^ { d + 1 }$ , then $\mathrm { V C } ( \mathcal { A } ) \leq C d$ . By applying Lemma C.1 to A with failure probability $\delta / 2$ , then for every $A \in { \mathcal { A } }$

$$
| N ( A ) - Q ( A ) | \leq C \left( \sqrt { d \ell _ { \delta / 2 } Q ( A ) } + d \ell _ { \delta / 2 } \right) ,\tag{118}
$$

$$
\begin{array} { r } { \| Z ( A ) \| _ { 2 } \lesssim \ell _ { \delta / 2 } ^ { 2 } \left( \sqrt { d ( N ( A ) + d ) } + d \right) . } \end{array}\tag{119}
$$

We now focus separately on the three claims of Proposition 2.5.

Counts. For a two-sided residual slab, $N ( A ) = A _ { w } ( b )$ and $Q ( A ) = Q _ { w } ( b )$ . Using

$$
| A - Q | \leq C ( { \sqrt { a Q } } + a ) \quad \Longrightarrow \quad A \leq 2 Q + C a , \qquad Q \leq 2 A + C a
$$

with $a = d \ell _ { \delta / 2 }$ proves (30), after using $\ell _ { \delta / 2 } \leq L$ (where the constant in L was chosen to be suficiently large).

Scores. For a given w, set

$$
A _ { + } ( b , w ) = \{ ( x , y ) : 0 < y - x ^ { \top } b \leq w \} , \qquad A _ { - } ( b , w ) = \{ ( x , y ) : - w \leq y - x ^ { \top } b < 0 \} .
$$

In these terms, it is clear that

$$
S _ { w } ( b ) - \overline { { S } } _ { w } ( b ) = Z ( A _ { + } ( b , w ) ) - Z ( A _ { - } ( b , w ) ) , \qquad N ( A _ { + } ( b , w ) ) + N ( A _ { - } ( b , w ) ) \leq A _ { w } ( b ) .
$$

Using the triangle inequality, (119), and $\sqrt { x } + \sqrt { y } \leq \sqrt { 2 ( x + y ) }$ implies

$$
\begin{array} { r } { \left\| S _ { w } ( b ) - \overline { { S } } _ { w } ( b ) \right\| _ { 2 } \lesssim \ell _ { \delta / 2 } ^ { 2 } \left( \sqrt { d ( A _ { w } ( b ) + d ) } + d \right) , } \end{array}
$$

which immediately yields (29) (again using $\ell _ { \delta / 2 } \leq L )$

Feature norms. A 1/2-net of $S ^ { d - 1 }$ has at most $5 ^ { d }$ points. The sub-Gaussian assumption and $\Sigma \preceq \kappa I$ therefore give

$$
\mathbb { P } \Big ( \| X _ { i } \| _ { 2 } > C \sqrt { d + t } \Big ) \leq e ^ { - t }
$$

for $t \geq 1$ . Choosing $t = C \log ( 2 n / \delta )$ and union-bounding over $i ,$ shows that with probability at least $1 - \delta / 2$

$$
\operatorname* { m a x } _ { i } \| X _ { i } \| _ { 2 } \leq C \sqrt { d + \log ( n / \delta ) } \leq \sqrt { L d } ,
$$

where the constant in L was chosen to be suficiently large. Intersecting this event with the uniform convergence event over A gives our desired result. □

## C.1.2 Proof of Lemma C.1

The proof follows by combining three prior results that we now describe.

Lemma C.2 (VC covering bound; e.g. van der Vaart and Wellner Section 2.6 [vdVW96]). Let A be a class of sets of VC dimension at most V. There are universal constants $A _ { 0 } \geq e$ and $c _ { 0 } \geq 1$ such that, for every finite discrete probability measure $Q$ and every $0 < \eta \leq 1$

$$
N ( \{ \mathbf { 1 } _ { A } : A \in \mathcal { A } \} , L _ { 2 } ( Q ) , \eta ) \leq \left( \frac { A _ { 0 } } { \eta } \right) ^ { c _ { 0 } V } .\tag{120}
$$

We will leverage a variant of (120) for weighted classes. An envelope for a class of functions H is a non-negative measurable function H satisfying $| h ( u ) | \leq H ( u )$ for every $h \in \mathcal H$ and every u. The indicator class in Lemma C.2 has envelope $H = 1$ . If r is a fixed measurable function and

$$
\mathcal { H } _ { r } = \{ r \mathbf { 1 } _ { A } : A \in \mathcal { A } \} ,
$$

then $H = | r |$ is an envelope because $| r ( u ) \mathbf { 1 } _ { A } ( u ) | \leq | r ( u ) |$ | for every A.

The covering consequence used below is

$$
N \Big ( \mathscr { H } _ { r } , L _ { 2 } ( Q ) , \eta \| r \| _ { L _ { 2 } ( Q ) } \Big ) \leq \left( \frac { A _ { 0 } } { \eta } \right) ^ { c _ { 0 } V } .\tag{121}
$$

Here the radius is scaled by the $L _ { 2 } ( Q )$ size of the envelope, $\lVert r \rVert _ { L _ { 2 } ( Q ) } = \lVert r \rVert _ { L _ { 2 } ( Q ) }$ . To verify (121), suppose first that $\| r \| _ { L _ { 2 } ( Q ) } > 0$ and define a probability measure $Q _ { r }$ by

$$
d Q _ { r } ( u ) = \frac { r ( u ) ^ { 2 } } { \left. r \right. _ { L _ { 2 } ( Q ) } ^ { 2 } } d Q ( u ) .
$$

Lemma C.2 gives indicators $\mathbf { 1 } _ { A _ { 1 } } , \dotsc , \mathbf { 1 } _ { A _ { N } }$ , with $N \leq ( A _ { 0 } / \eta ) ^ { c _ { 0 } V }$ , such that for every $A \in { \mathcal { A } }$ some $A _ { j }$ obeys $\Vert \mathbf { 1 } _ { A } - \mathbf { 1 } _ { A _ { j } } \Vert _ { L _ { 2 } ( Q _ { r } ) } \leq \eta$ . For that same $A _ { j }$

$$
\begin{array} { l } { \displaystyle \big \| r { \bf 1 } _ { A } - r { \bf 1 } _ { A _ { j } } \big \| _ { L _ { 2 } ( Q ) } ^ { 2 } = \int r ( u ) ^ { 2 } | { \bf 1 } _ { A } ( u ) - { \bf 1 } _ { A _ { j } } ( u ) | ^ { 2 } d Q ( u ) } \\ { \displaystyle \qquad = \| r \| _ { L _ { 2 } ( Q ) } ^ { 2 } \big \| { \bf 1 } _ { A } - { \bf 1 } _ { A _ { j } } \big \| _ { L _ { 2 } ( Q _ { r } ) } ^ { 2 } } \\ { \displaystyle \qquad \leq \eta ^ { 2 } \| r \| _ { L _ { 2 } ( Q ) } ^ { 2 } . } \end{array}
$$

Thus the same indicator net becomes a weighted-function net after multiplication by r. $\operatorname { I f } \| r \| _ { L _ { 2 } ( Q ) } =$   
$0 ,$ every member of $\mathcal { H } _ { r }$ is zero in $L _ { 2 } ( Q )$ , so the claim is immediate.

Lemma C.3 (i.n.i.d. VC maximal inequality; Lemma SA25 of Cattaneo, Feng, and Underwood [CFU24]). Let $U _ { 1 } , \dots , U _ { n }$ be independent, with laws $P _ { 1 } , \ldots , P _ { n }$ , and put $\overline { { P } } = n ^ { - 1 } \sum _ { i } P _ { i }$ . Let $\mathcal { F }$ be a pointwise measurable class with envelope F such that $0 < \| F \| _ { L _ { 2 } ( \overline { { P } } ) } < \infty$ , and suppose that, for some $A \geq e$ and $V \geq 1$

$$
\operatorname* { s u p } _ { Q } N \Big ( \mathcal { F } , L _ { 2 } ( Q ) , \eta \| F \| _ { L _ { 2 } ( Q ) } \Big ) \leq \Big ( \frac { A } { \eta } \Big ) ^ { V } , \qquad 0 < \eta \leq 1 ,\tag{122}
$$

where the supremum is over finite discrete probability measures. Suppose

$$
\operatorname* { s u p } _ { f \in \mathscr { F } } \| f \| _ { L _ { 2 } ( \overline { { P } } ) } \leq \sigma \leq \| F \| _ { L _ { 2 } ( \overline { { P } } ) } , \qquad M = \operatorname* { m a x } _ { 1 \leq i \leq n } F ( U _ { i } ) .
$$

Then

$$
\mathrm { E } \operatorname* { s u p } _ { f \in \mathcal { F } } \left| \sum _ { i = 1 } ^ { n } \{ f ( U _ { i } ) - P _ { i } f \} \right| \leq C \left[ \sqrt { n } \sigma \sqrt { V \log \left( \frac { A \| F \| _ { L _ { 2 } ( \overline { { P } } ) } } { \sigma } \right) } + \| M \| _ { 2 } V \log \left( \frac { A \| F \| _ { L _ { 2 } ( \overline { { P } } ) } } { \sigma } \right) \right] .\tag{123}
$$

We leverage the following to yield a high-probability bound from the in-expectation bound:

Theorem C.4 (Deviation inequality; Theorem 4 of Adamczak [Ada08]). Let $U _ { 1 } , \ldots , U _ { n }$ be independent and let $\mathcal { G }$ be a countable class such that $\operatorname { E } g ( U _ { i } ) = 0$ for every $g \in { \mathcal { G } }$ and every i. Define

$$
Z = \operatorname* { s u p } _ { g \in \mathcal { G } } \left| \sum _ { i = 1 } ^ { n } g ( U _ { i } ) \right| , \qquad \sigma ^ { 2 } = \operatorname* { s u p } _ { g \in \mathcal { G } } \sum _ { i = 1 } ^ { n } \mathrm { E } g ( U _ { i } ) ^ { 2 } ,
$$

and

$$
B = \left\| \operatorname* { m a x } _ { 1 \leq i \leq n } \operatorname* { s u p } _ { g \in \mathcal { G } } | g ( U _ { i } ) | \right\| _ { \psi _ { 1 } } .
$$

There is a universal constant C such that, for every $u \geq 1$ , with probability at least $1 - 4 e ^ { - u }$

$$
Z \leq 2 \mathrm { E } Z + C \big ( \sigma \sqrt { u } + B u \big ) .\tag{124}
$$

This is the ψ<sub>1</sub> case of Theorem $\it 4$ of [Ada08] (fix the two auxiliary constants in that theorem and take $t = C ( \sigma { \sqrt { u } } + B u ) )$

Proof of Lemma C.1. Choose

$$
\ell = \ell _ { \varepsilon } , \qquad D = d + V , \qquad \ell _ { q } = \operatorname* { m a x } \{ 1 , \log ( e n / q ) \} \quad ( 1 \leq q \leq n ) .
$$

1. Bounding $| N ( A ) - Q ( A ) |$ . For a fixed $q ,$ let $\mathcal { A } _ { q } = \{ A \in \mathcal { A } : Q ( A ) \leq q \}$ and apply Lemma C.3 to $\mathcal { F } _ { q } = \{ \mathbf { 1 } _ { A } : A \in \mathcal { A } _ { q } \}$ . By Lemma C.2, its entropy exponent is $c _ { 0 } V ;$ its envelope is $1 ;$ and

$$
\operatorname* { s u p } _ { A \in \mathcal { A } _ { q } } \left\| \mathbf { 1 } _ { A } \right\| _ { L _ { 2 } ( \overline { { P } } ) } ^ { 2 } = \frac { 1 } { n } \operatorname* { s u p } _ { A \in \mathcal { A } _ { q } } Q ( A ) \leq \frac { q } { n } .
$$

Thus (123), with $\sigma = { \sqrt { q / n } }$ , gives

$$
\mathrm { ~ E ~ } \operatorname* { s u p } _ { A \in { \mathcal { A } } _ { q } } | N ( A ) - Q ( A ) | \leq C \left( { \sqrt { q V \ell _ { q } } } + V \ell _ { q } \right) .\tag{125}
$$

We will turn this into a high-probability guarantee by invoking Theorem C.4 on the centered indicator class; the variance proxy $\sigma ^ { 2 } \leq q$ (centering can only decrease the second moment) and $B \leq 1$ Combining (124) and (125) therefore yields, for every $u \geq 1$ , with failure probability at most $4 e ^ { - u }$

$$
\operatorname* { s u p } _ { A \in { \mathcal { A } } _ { q } } | N ( A ) - Q ( A ) | \leq C \left[ { \sqrt { q ( V \ell _ { q } + u ) } } + V \ell _ { q } + u \right] .\tag{126}
$$

Apply (126) at dyadic levels $q = 1 , 2 , 4 , \dots$ , with the last level capped at $n ,$ and take $u = c \ell .$ For a suficiently large universal c, the union-bound gives a total probability of failure of at most $4 ( 1 + \log _ { 2 } n ) e ^ { - c \ell } \leq \varepsilon / 2$ . Since $\ell _ { q } \leq \ell$ and $V \geq 1$ , we obtain simultaneously at all dyadic levels

$$
\operatorname* { s u p } _ { A \in { \mathcal { A } } _ { q } } | N ( A ) - Q ( A ) | \leq C \left( { \sqrt { V q \ell } } + V \ell \right) .\tag{127}
$$

If $Q ( A ) \geq 1$ , choose q so that $Q ( A ) \leq q < 2 Q ( A ) ; { \mathrm { i f ~ } } Q ( A ) < 1$ , use $q = 1$ and absorb $\sqrt { V \ell }$ into $V \ell .$ This proves (116).

2. Bounding $\| Z ( A ) \| _ { 2 } .$ Our plan is to bound $\| Z ( A ) \| _ { 2 }$ by invoking our VC-style bounds with the class of functions associated with 1-dimensional projections $r _ { v } ( x , y ) = \langle v , x \rangle$ for a fixed $v \in S ^ { d - 1 }$ Eventually, we will employ a net of such v to yield our desired $\| Z ( A ) \| _ { 2 }$ bound, but first we focus on just a fixed v. The sub-Gaussian assumption and $\Sigma \preceq \kappa I$ imply

$$
\| r _ { v } ( W _ { i } ) \| _ { \psi _ { 2 } } \leq C ( K , \kappa ) , \qquad \| r _ { v } \| _ { L _ { 2 } ( \overline { { P } } ) } \leq C ( \kappa ) .\tag{128}
$$

As before, we will handle classes separately with $A _ { q }$ for diferent q. Naturally, it is helpful to note that if an event E has probability $p ,$ integration of the sub-Gaussian tail gives

$$
\begin{array} { r } { \operatorname { E } [ r _ { v } ( W _ { i } ) ^ { 2 } \mathbf { 1 } _ { E } ] \le C p \log ( e / p ) . } \end{array}\tag{129}
$$

To see this directly, choose $t _ { p } = C \log ( e / p )$ and use $\mathbb { P } ( r _ { v } ( W _ { i } ) ^ { 2 } > t ) \le 2 e ^ { - c t }$ , for the bound

$$
\operatorname { E } [ r _ { v } ( W _ { i } ) ^ { 2 } \mathbf { 1 } _ { E } ] \leq \int _ { 0 } ^ { t _ { p } } p d t + \int _ { t _ { p } } ^ { \infty } 2 e ^ { - c t } d t \leq C p \log ( e / p ) .
$$

Writing $p _ { i } = P _ { i } ( A )$ and using concavity of $p \mapsto p \log ( e / p )$ , we obtain for every $A \in { \mathcal { A } } _ { q }$

$$
\sum _ { i = 1 } ^ { n } \operatorname { E } [ r _ { v } ( W _ { i } ) ^ { 2 } \mathbf { 1 } _ { A } ( W _ { i } ) ] \leq C \sum _ { i = 1 } ^ { n } p _ { i } \log ( e / p _ { i } ) \leq C q \ell _ { q } .\tag{130}
$$

We now apply Lemma C.3 to $\mathcal { F } _ { q , v } = \{ r _ { v } \mathbf { 1 } _ { A } : A \in \mathcal { A } _ { q } \}$ . We choose the strictly positive envelope

$$
F _ { v } ( x , y ) = 1 + | r _ { v } ( x , y ) | ,
$$

satisfying $| r _ { v } \mathbf { 1 } _ { A } | \le F _ { v }$ . The weighted covering bound (121) gives the required entropy estimate with normalizing radius $\eta \parallel r _ { v } \parallel _ { L _ { 2 } ( Q ) }$ . Since $\lVert r _ { v } \rVert _ { L _ { 2 } ( Q ) } \leq \lVert F _ { v } \rVert _ { L _ { 2 } ( Q ) }$ , the same net is also an $\eta \parallel F _ { v } \parallel _ { L _ { 2 } ( Q ) ^ { - } }$ net. Thus, the entropy condition in Lemma C.3 holds with exponent $c _ { 0 } V$ and envelope $F _ { v }$

Equation (130) controls the variance term for $\mathcal { F } _ { q , v } .$ , since

$$
\operatorname* { s u p } _ { A \in \mathcal { A } _ { q } } \| r _ { v } \mathbf { 1 } _ { A } \| _ { L _ { 2 } ( \overline { { P } } ) } ^ { 2 } = \frac { 1 } { n } \operatorname* { s u p } _ { A \in \mathcal { A } _ { q } } \sum _ { i = 1 } ^ { n } \mathrm { E } [ r _ { v } ( W _ { i } ) ^ { 2 } \mathbf { 1 } _ { A } ( W _ { i } ) ] \leq \frac { C q \ell _ { q } } { n }
$$

lets us choose

$$
\sigma _ { q , v } = \operatorname* { m i n } \left\{ \| F _ { v } \| _ { L _ { 2 } ( \overline { { P } } ) } , C \sqrt { \frac { q \ell _ { q } } { n } } \right\} .
$$

We may also control the random variable M in Lemma C.3:

$$
M _ { v } : = \operatorname* { m a x } _ { 1 \leq i \leq n } F _ { v } ( W _ { i } ) = 1 + \operatorname* { m a x } _ { 1 \leq i \leq n } | r _ { v } ( W _ { i } ) | , \qquad \| M _ { v } \| _ { 2 } \leq C \sqrt { \log ( e n ) } .\tag{131}
$$

The last inequality follows by integrating the union bound for the uniformly sub-Gaussian variables $r _ { v } ( W _ { i } )$ . We now invoke Lemma C.3 and use that $1 \leq \| F _ { v } \| _ { L _ { 2 } ( \overline { { P } } ) } \leq C$ and log $\left( \frac { A \| F _ { v } \| _ { L _ { 2 } ( \overline { { P } } ) } } { \sigma _ { q , v } } \right) \leq C \ell _ { q }$ to conclude

$$
\operatorname { E } \operatorname* { s u p } _ { A \in \mathcal { A } _ { q } } \left. \sum _ { i = 1 } ^ { n } \lbrace r _ { v } ( W _ { i } ) \mathbf { 1 } _ { A } ( W _ { i } ) - P _ { i } ( r _ { v } \mathbf { 1 } _ { A } ) \rbrace \right. \leq C \left[ \ell _ { q } \sqrt { q V } + V \ell _ { q } \sqrt { \log ( e n ) } \right] .\tag{132}
$$

To apply Theorem C.4, we define $U _ { i } = ( i , W _ { i } )$ and $g _ { A , v } ( i , w ) = r _ { v } ( w ) \mathbf { 1 } _ { A } ( w ) - P _ { i } ( r _ { v } \mathbf { 1 } _ { A } )$ . Note that $\operatorname { E } g _ { A , v } ( U _ { i } ) = 0$ . The variance proxy $\sigma ^ { 2 }$ is at most $C q \ell _ { q }$ by (130), because centering can only decrease the second moment. A convenient envelope for this centered class is

$$
G _ { v } ( i , w ) = | r _ { v } ( w ) | + P _ { i } | r _ { v } | \leq | r _ { v } ( w ) | + C .
$$

Therefore the envelope term B in Theorem C.4 satisfies

$$
B \leq \left. \operatorname* { m a x } _ { 1 \leq i \leq n } G _ { v } ( i , W _ { i } ) \right. _ { \psi _ { 1 } } \leq C \sqrt { \log ( e n ) }\tag{133}
$$

since the maximum of n uniformly sub-Gaussian variables has $\psi _ { 1 }$ norm $O ( { \sqrt { \log ( e n ) } } )$ . Using Theorem C.4 and (132) now proves, that with failure probability at most $4 e ^ { - u }$

$$
\begin{array} { r } { \displaystyle \operatorname* { s u p } _ { A \in \mathcal { A } _ { q } } \left| \sum _ { i = 1 } ^ { n } \lbrace r _ { v } ( W _ { i } ) \mathbf { 1 } _ { A } ( W _ { i } ) - P _ { i } ( r _ { v } \mathbf { 1 } _ { A } ) \rbrace \right| \leq C \big [ \ell _ { q } \sqrt { q V } + V \ell _ { q } \sqrt { \log ( e n ) } } \\ { + \sqrt { q \ell _ { q } u } + \sqrt { \log ( e n ) } u \big ] . } \end{array}\tag{134}
$$

We now turn towards using a net argument to imply a bound for $\| Z ( A ) \| _ { 2 }$ . Let $\mathcal { N }$ be a $, 1 / 2 \cdot$ -net of $S ^ { d - 1 }$ with $| \mathcal { N } | \leq 5 ^ { d }$ . Apply (134) to every $v \in \mathcal N$ and every dyadic q, taking $u = c ( d + \ell )$ . For suficiently large universal $c ,$ a union-bound gives failure probability at most

$$
4 ( 1 + \log _ { 2 } { n } ) 5 ^ { d } e ^ { - c ( d + \ell ) } \leq \varepsilon / 2 .
$$

Since $\ell _ { q } , \log ( e n ) \leq \ell$ and $d + \ell \leq D \ell .$ , the right side of (134) is at most

$$
C \left( \ell \sqrt { D q } + D \ell ^ { 3 / 2 } \right) .\tag{135}
$$

Using the dyadic q and $\left\| z \right\| _ { 2 } \leq 2 \operatorname* { m a x } _ { v \in \mathcal { N } } | \left. v , z \right. |$ yields that, simultaneously for every $A \in { \mathcal { A } } .$

$$
\| Z ( A ) \| _ { 2 } \le C \left[ \ell \sqrt { D ( Q ( A ) + 1 ) } + D \ell ^ { 3 / 2 } \right] .\tag{136}
$$

3. Replace population mass by the observed count. Equation (116) implies

$$
Q ( A ) \leq 2 N ( A ) + C V \ell .
$$

Substituting this into (136), using $V , d \leq D$ and $\ell \geq 1$ , yields (117):

$$
\| Z ( A ) \| _ { 2 } \leq C \ell ^ { 2 } \left( \sqrt { D ( N ( A ) + D ) } + D \right) .
$$

## C.2 Proof of Lemma 3.2

Proof. This follows nearly immediately from Lemma C.1 and typical algebraic manipulations.

Fix $A \in { \mathcal { A } }$ and abbreviate $N = N ( A ) , Q = Q ( A )$ , and $B = V \ell _ { \delta }$

If $Q \le B$ , then (116) implies

$$
N \leq Q + C ( { \sqrt { Q B } } + B ) \leq C B .
$$

Consequently,

$$
| { \sqrt { N } } - { \sqrt { Q } } | \leq { \sqrt { N } } + { \sqrt { Q } } \leq ( { \sqrt { C } } + 1 ) { \sqrt { B } } .
$$

If $Q > B ;$ , use the exact identity

$$
| \sqrt { N } - \sqrt { Q } | = \frac { | N - Q | } { \sqrt { N } + \sqrt { Q } } \leq \frac { | N - Q | } { \sqrt { Q } } .
$$

Together with (116), this yields

$$
| { \sqrt { N } } - { \sqrt { Q } } | \leq C \left( { \sqrt { B } } + { \frac { B } { \sqrt { Q } } } \right) \leq 2 C { \sqrt { B } } .
$$

Both cases prove (52) simultaneously for all $A \ \in \ A$ under the uniform convergence event of Lemma C.1. □

## D Additional deferred proofs

## D.1 Proof of Claim D.1

The following enables moving constant factors from outside the Hellinger modulus to inside the Hellinger modulus:

Claim D.1. For $\begin{array} { r } { t < \frac { 1 } { 9 } } \end{array}$ and any symmetric, unimodal $p ,$ it holds that

$$
\omega _ { p } ( t ) \leq \frac { 1 } { 2 } \omega _ { p } ( 9 t )
$$

Proof. This follows immediately from Lemma 3.13.

## D.2 Proof of Proposition 3.5

Proof. For $u \in \mathbb { R } ^ { d }$ , let

$$
L _ { m } ( u ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left| Y _ { i } - X _ { i } ^ { \top } ( \beta + u ) \right| ^ { q } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left| \varepsilon _ { i } - X _ { i } ^ { \top } u \right| ^ { q } .
$$

Our plan is to show that for all u satisfying $\left. u \right. _ { 2 } \geq R$ for a later chosen $R ,$ it holds that

$$
\langle \nabla L _ { m } ( u ) , u \rangle > 0 .\tag{137}
$$

By convexity of $L _ { m }$ , this would prove $\left\| \widehat { \beta } _ { q } - \beta \right\| _ { 2 } \leq R$

Fix $u \in \mathbb { R } ^ { d }$ and let $v _ { i } = X _ { i } ^ { \top } u$ . For each observation, define the one-dimensional function

$$
\phi _ { i } ( t ) = | \varepsilon _ { i } - t v _ { i } | ^ { q } , \qquad 0 \leq t \leq 1 .
$$

This will be a relevant function, since

$$
\langle \nabla L _ { m } ( u ) , u \rangle = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \phi _ { i } ^ { \prime } ( 1 ) .
$$

We compute the first and second derivatives of $\phi _ { i }$ as

$$
\phi _ { i } ^ { \prime } ( t ) = - q v _ { i } \mathrm { \mathrm { ~ s g n } } ( \varepsilon _ { i } - t v _ { i } ) | \varepsilon _ { i } - t v _ { i } | ^ { q - 1 } , \qquad \phi _ { i } ^ { \prime \prime } ( t ) = q ( q - 1 ) v _ { i } ^ { 2 } | \varepsilon _ { i } - t v _ { i } | ^ { q - 2 } .
$$

Consequently, we may bound

$$
\begin{array} { r l } & { \phi _ { i } ^ { \prime } ( 1 ) = \phi _ { i } ^ { \prime } ( 0 ) + \big ( \phi _ { i } ^ { \prime } ( 1 ) - \phi _ { i } ^ { \prime } ( 0 ) \big ) = \phi _ { i } ^ { \prime } ( 0 ) + q ( q - 1 ) v _ { i } ^ { 2 } \int _ { 0 } ^ { 1 } \left| \varepsilon _ { i } - t v _ { i } \right| ^ { q - 2 } d t } \\ & { \qquad \geq \phi _ { i } ^ { \prime } ( 0 ) + q ( q - 1 ) \left| \varepsilon _ { i } \right| ^ { q - 2 } v _ { i } ^ { 2 } \mathbf { 1 } \{ \varepsilon _ { i } v _ { i } \leq 0 \} \quad \mathrm { ( v i a ~ } \varepsilon _ { i } v _ { i } \leq 0 \implies \left| \varepsilon _ { i } - t v _ { i } \right| \geq \left| \varepsilon _ { i } \right| \big ) } \\ & { \qquad = - q v _ { i } \operatorname { s g n } ( \varepsilon _ { i } ) \left| \varepsilon _ { i } \right| ^ { q - 1 } + q ( q - 1 ) \left| \varepsilon _ { i } \right| ^ { q - 2 } v _ { i } ^ { 2 } \mathbf { 1 } \{ \varepsilon _ { i } v _ { i } \leq 0 \} } \\ & { \qquad \implies \langle \nabla L _ { m } ( u ) , u \rangle \geq - q G ^ { \top } u + q ( q - 1 ) C _ { n } ( u ) , } \end{array}
$$

where

$$
G = { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } X _ { i } \operatorname { s g n } ( \varepsilon _ { i } ) | \varepsilon _ { i } | ^ { q - 1 }
$$

relates to the gradient of the objective at the truth $\left( \nabla L _ { m } ( 0 ) \right)$ , and

$$
C _ { n } ( u ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } | \varepsilon _ { i } | ^ { q - 2 } ( X _ { i } ^ { \top } u ) ^ { 2 } \mathbf { 1 } \{ \varepsilon _ { i } X _ { i } ^ { \top } u \leq 0 \}
$$

is a curvature-related term. We will bound both terms separately.

Bounding the gradient term. Let

$$
\xi _ { i } = X _ { i } \operatorname { s g n } ( \varepsilon _ { i } ) | \varepsilon _ { i } | ^ { q - 1 } .
$$

Since $p$ is symmetric and $\varepsilon _ { i }$ and $X _ { i }$ are independent, it holds that $\operatorname { E } [ \xi _ { i } ] = 0$ . We may then bound the second moment by

$$
\begin{array} { r l } & { \displaystyle { \mathbb E } \| G \| _ { 2 } ^ { 2 } = \frac { 1 } { m ^ { 2 } } \sum _ { i = 1 } ^ { m } { \mathbb E } \| \xi _ { i } \| _ { 2 } ^ { 2 } } \\ & { \quad \quad \quad = \frac { 1 } { m } { \mathbb E } \left[ | \varepsilon | ^ { 2 q - 2 } X ^ { \top } X \right] } \\ & { \quad \quad \quad = \frac { M _ { 2 q - 2 } } { m } \operatorname { t r } ( \Sigma ) } \\ & { \quad \quad \quad \le \frac { \kappa d M _ { 2 q - 2 } } { m } . } \end{array}
$$

Applying Markov’s inequality to $\| G \| _ { 2 } ^ { 2 }$ yields

$$
\mathbb { P } \left\{ \| G \| _ { 2 } > 6 \sqrt { \frac { \kappa d M _ { 2 q - 2 } } { m } } \right\} \leq \frac { 1 } { 3 6 } .\tag{138}
$$

Bounding the curvature term. For ease of reading, we will separately prove the required result for this term in Lemma D.2. We invoke this result with error probability $\delta = 1 / 3 6 ;$ its sample-size requirement is enforced by our condition (61) with large enough $C _ { K , \kappa }$ . Hence, with probability at least 35/36, it holds that

$$
\begin{array} { r } { C _ { n } ( u ) \geq c _ { K , \kappa } M _ { q - 2 } \left\| u \right\| _ { \Sigma } ^ { 2 } \geq c _ { K , \kappa } M _ { q - 2 } \left\| u \right\| _ { 2 } ^ { 2 } \qquad \mathrm { f o r ~ e v e r y ~ } u \in \mathbb { R } ^ { d } . } \end{array}\tag{139}
$$

This also implies the rank condition, as otherwise a vector in the null space of X would violate (139).

Concluding $\langle \nabla L _ { m } ( u ) , u \rangle > 0 ~ f o r ~ \| u \| _ { 2 } \geq R$ . We finally prove the desired bound as in (137) for $\left. u \right. _ { 2 } \geq R$ . With probability at least $1 7 / 1 8$ , by using the events of (138) and (139), we bound

$$
\begin{array} { r l } & { \langle \nabla L _ { m } ( u ) , u \rangle \ge - q G ^ { \top } u + q ( q - 1 ) C _ { n } ( u ) } \\ & { \qquad \ge - q \| G \| _ { 2 } \| u \| _ { 2 } + q ( q - 1 ) C _ { n } ( u ) } \\ & { \qquad \ge - 6 q \sqrt { \frac { \kappa d M _ { 2 q - 2 } } { m } } \| u \| _ { 2 } + q ( q - 1 ) c _ { K , \kappa } M _ { q - 2 } \| u \| _ { 2 } ^ { 2 } } \\ & { \qquad = \| u \| _ { 2 } \cdot \left( q ( - 1 ) c _ { K , \kappa } M _ { q - 2 } \| u \| _ { 2 } - 6 q \sqrt { \frac { \kappa d M _ { 2 q - 2 } } { m } } \right) . } \end{array}
$$

Hence, we may conclude that $\langle \nabla L _ { m } ( u ) , u \rangle > 0$ whenever

$$
\begin{array} { r } { \| u \| _ { 2 } \geq C _ { K , \kappa } \frac { 6 q \sqrt { \frac { \kappa d M _ { 2 q - 2 } } { m } } } { q ( q - 1 ) c _ { K , \kappa } M _ { q - 2 } } \iff \| u \| _ { 2 } \geq C _ { K , \kappa } \sqrt { \frac { d V _ { p } ( q ) } { m } } , } \end{array}
$$

for suficiently large $C _ { K , \kappa }$ , proving our desired result.

We now prove the desired curvature term bound:

Lemma D.2. There exists a $C _ { K , \kappa } \geq 1$ such that, if

$$
m \geq C _ { K , \kappa } Q _ { p } ( q ) d \log ( e m / \delta ) ,\tag{140}
$$

then for $\delta \in ( 0 , 1 / 2 )$ , with probability at least $1 - \delta$

$$
\frac { 1 } { m } \sum _ { i = 1 } ^ { m } | \varepsilon _ { i } | ^ { q - 2 } ( X _ { i } ^ { \top } u ) ^ { 2 } \mathbf { 1 } \{ \varepsilon _ { i } X _ { i } ^ { \top } u \leq 0 \} \geq c _ { K , \kappa } M _ { q - 2 } u ^ { \top } \Sigma u\tag{141}
$$

for every $u \in \mathbb { R } ^ { d }$

Proof. Observe that the claim is scale-invariant, so it sufices to focus on the case where $\left. u \right. _ { 2 } = 1$ Let

$$
\xi = | \varepsilon | ^ { q - 2 } , \qquad \widetilde X = - \operatorname { s g n } ( \varepsilon ) X .
$$

In this notation, recall that

$$
C _ { n } ( u ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \xi _ { i } ( \widetilde { X } _ { i } ^ { \top } u ) ^ { 2 } \mathbf { 1 } \{ \widetilde { X } _ { i } ^ { \top } u \geq 0 \}
$$

By symmetry we may use

$$
\operatorname { E } \big [ ( \widetilde { X } ^ { \top } u ) ^ { 2 } \mathbf { 1 } \{ \widetilde { X } ^ { \top } u \geq 0 \} \big ] = \frac { 1 } { 2 } \mathbf { E } ( \widetilde { X } ^ { \top } u ) ^ { 2 } \geq c _ { K , \kappa } .
$$

Using the same logic as Claim 2.3 implies there exist constants $a _ { K , \kappa } , b _ { K , \kappa } > 0$ where

$$
\mathbb { P } ( \widetilde { X } ^ { \top } u \ge a _ { K , \kappa } ) \ge b _ { K , \kappa }
$$

for all $\left. u \right. _ { 2 } = 1$ . Our proof will only require lower bounding the contributions from terms where $\widetilde { X } _ { i } ^ { \top } u \ge a _ { K , \kappa }$ . In particular, we will prove

$$
\frac { 1 } { m } \sum _ { i = 1 } ^ { m } \xi _ { i } { \bf 1 } \{ { \widetilde X } _ { i } ^ { \top } u \geq a _ { K , \kappa } \} \geq c _ { K , \kappa } M _ { q - 2 } ,\tag{142}
$$

which directly implies our desired bound. We will prove such a lower bound by invoking uniform convergence guarantees for the counts of certain sets, and showing how we may represent this quantity in a way that corresponds to an integral over certain counts.

It will be helpful to work with a modified version of this quantity where the values are restricted below a maximum value. Consider

$$
U = 8 M _ { q - 2 } Q _ { p } ( q ) , \qquad \overline { { { \xi } } } = \mathrm { m i n } ( { \xi } , U ) .
$$

By definition, $\mathrm { E } [ \xi ^ { 2 } ] = M _ { q - 2 } ^ { 2 } Q _ { p } ( q )$ . We show that $\mathbb { E } [ \overline { { \xi } } ]$ is not much smaller than $\mathbb { E } [ \xi ]$

$$
\begin{array} { l } { { \displaystyle \mathrm { E } [ \overline { { \xi } } ] = M _ { q - 2 } - \mathrm { E } [ ( \xi - U ) _ { + } ] } } \\ { { \displaystyle \quad \geq M _ { q - 2 } - \mathrm { E } \big [ \xi \mathbf { 1 } \{ \xi > U \} \big ] } } \\ { { \displaystyle \quad \geq M _ { q - 2 } - \frac { \mathrm { E } \xi ^ { 2 } } { U } = M _ { q - 2 } - \frac { M _ { q - 2 } ^ { 2 } Q _ { p } ( q ) } { 8 M _ { q - 2 } Q _ { p } ( q ) } = \frac { 7 } { 8 } M _ { q - 2 } . } } \end{array}\tag{143}
$$

We now define the relevant sets on which we will leverage uniform convergence. For $s \in ( 0 , U ]$ and $\| v \| _ { 2 } = 1$ , let

$$
E _ { v , s } = \{ \xi \geq s , \ \widetilde { X } ^ { \top } v \geq a _ { K , \kappa } \} .
$$

The probabilities of these events satisfy

$$
b _ { K , \kappa } \mathbb { P } ( \boldsymbol { \xi } \geq s ) \leq \mathbb { P } ( E _ { v , s } ) \leq \mathbb { P } ( \boldsymbol { \xi } \geq s ) .\tag{144}
$$

In terms of the observations $( X , Y )$ , the concept class has VC dimension $\lesssim d .$ For $q > 2 , E _ { v , s }$ can be written as the union of two intersections of two halfspaces

$$
\{ Y - X ^ { \top } \beta \geq s ^ { 1 / ( q - 2 ) } , \ X ^ { \top } v \leq - a _ { K , \kappa } \} , \qquad \{ Y - X ^ { \top } \beta \leq - s ^ { 1 / ( q - 2 ) } , \ X ^ { \top } v \geq a _ { K , \kappa } \} .
$$

For $q = 2 , E _ { v , s }$ is empty when $s > 1$ , and for $s \leq 1$ is similarly

$$
\{ Y - X ^ { \top } \beta > 0 , ~ X ^ { \top } v \leq - a _ { K , \kappa } \} , \qquad \{ Y - X ^ { \top } \beta < 0 , ~ X ^ { \top } v \geq a _ { K , \kappa } \} .
$$

We now employ the uniform convergence guarantee (116) in Lemma C.1, with failure probability δ. Under this event, simultaneously for every v and s,

$$
N ( E _ { v , s } ) = \sum _ { i = 1 } ^ { m } \mathbf { 1 } \big \{ ( X _ { i } , Y _ { i } ) \in E _ { v , s } \big \} \geq m b _ { K , \kappa } \mathbb { P } ( \xi \geq s ) - C _ { K , \kappa } \left( \sqrt { d m \mathbb { P } ( \xi \geq s ) \ell _ { \delta } } + d \ell _ { \delta } \right) ,\tag{145}
$$

where

$$
\ell _ { \delta } = \operatorname* { m a x } \{ 1 , \log ( e m / \delta ) \} .
$$

We use this to bound

$$
\begin{array} { l l } & { \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \overline { { \xi } } _ { i } { \bf 1 } \{ \tilde { X } _ { i } ^ { \top } v \geq a _ { K , \kappa } \} = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left( \int _ { 0 } ^ { U } { \bf 1 } \{ \tilde { \xi } _ { i } \geq s \} d s \right) { \bf 1 } \{ \tilde { X } _ { i } ^ { \top } v \geq a _ { K , \kappa } \} = \frac { 1 } { m } \int _ { 0 } ^ { U } N ( E _ { v , s } ) d s } \\ & { \geq \displaystyle \frac { 1 } { m } \int _ { 0 } ^ { U } m b _ { K , \kappa } \mathbb { P } ( \xi \geq s ) - C _ { K , \kappa } \left( \sqrt { d m \mathbb { P } ( \xi \geq s ) \ell _ { \delta } } + d \ell _ { \delta } \right) d s } \\ & { = b _ { K , \kappa } { \bf E } [ \overline { { \xi } } ] - C _ { K , \kappa } \sqrt { \displaystyle \frac { d \ell _ { \delta } } { m } } \left( \int _ { 0 } ^ { U } \sqrt { \mathbb { P } ( \xi \geq s ) } d s \right) - \frac { C _ { K , \kappa } d \ell _ { \delta } U } { m } \left( \mathrm { v i a ~ } \mathbb { E } [ \overline { { \xi } } ] = \int _ { 0 } ^ { U } \mathbb { P } ( \xi \geq s ) d s \right) } \end{array}
$$

Using Cauchy-Schwarz gives $\begin{array} { r } { \int _ { 0 } ^ { U } \sqrt { \mathbb { P } ( \xi \geq s ) } d s \leq \left( U \int _ { 0 } ^ { U } \mathbb { P } ( \xi \geq s ) d s \right) ^ { 1 / 2 } = \sqrt { U \mathrm { E } \overline { { \xi } } } \leq \sqrt { 8 Q _ { p } ( q ) } M _ { q - 2 } , } \end{array}$ and hence we may continue

$$
\begin{array} { r l } & { \geq b _ { K , \kappa } \mathsf { E } [ \overline { { \xi } } ] - C _ { K , \kappa } M _ { q - 2 } \sqrt { \frac { d \ell _ { \delta } Q _ { p } ( q ) } { m } } - \frac { C _ { K , \kappa } d \ell _ { \delta } M _ { q - 2 } Q _ { p } ( q ) } { m } } \\ & { \geq \frac { 7 b _ { K , \kappa } } { 8 } M _ { q - 2 } - C _ { K , \kappa } M _ { q - 2 } \sqrt { \frac { d \ell _ { \delta } Q _ { p } ( q ) } { m } } - \frac { C _ { K , \kappa } d \ell _ { \delta } M _ { q - 2 } Q _ { p } ( q ) } { m } \quad \mathrm { ( v i a ~ ( 1 4 3 ) ) } } \\ & { \geq \frac { b _ { K , \kappa } } { 2 } M _ { q - 2 } , \quad \mathrm { ( v i a ~ ( 1 4 0 ) ) } } \end{array}
$$

when the constant in our condition (140) is chosen suficiently large. This proves (142), and hence completes our proof. □

## D.3 Proof of Corollary 3.6

Proof. The condition of (64) immediately holds with probability at least $1 7 / 1 8$ by invoking Lemma D.2 with $q = 2$ and $\delta = 1 / 1 8 .$ , since

$$
\sigma _ { \operatorname* { m i n } } ^ { 2 } ( \mathbf { X } ) = \operatorname* { m i n } _ { \| u \| _ { 2 } = 1 } \sum _ { i = 1 } ^ { m } ( X _ { i } ^ { \top } u ) ^ { 2 } \geq \operatorname* { m i n } _ { \| u \| _ { 2 } = 1 } \sum _ { i = 1 } ^ { m } | \varepsilon _ { i } | ^ { q - 2 } \left( X _ { i } ^ { \top } u \right) ^ { 2 } \mathbf { 1 } \{ \varepsilon _ { i } X _ { i } ^ { \top } u \leq 0 \} \geq c _ { K , \kappa } m .
$$

We now turn towards the implication for the algorithm given by Adil, Kyng, Peng, and Sachdeva [AKPS24, Theorem 1.1]. Recall

$$
\widehat { \beta } _ { q } = \mathop { \mathrm { a r g } \operatorname* { m i n } } _ { b \in \mathbb { R } ^ { d } } \| \mathbf { X } b - \mathbf { Y } \| _ { q } ^ { q } .
$$

We invoke their Theorem 1.1 to yield a $\widehat { A } _ { q }$ where

$$
\begin{array} { r } { \left\| \mathbf { X } \widehat { A } _ { q } - \mathbf { Y } \right\| _ { q } ^ { q } \leq \left\| \mathbf { X } \widehat { \beta } _ { q } - \mathbf { Y } \right\| _ { q } ^ { q } + \varepsilon , } \end{array}
$$

for a later-chosen $\varepsilon > 0$ . We now briefly set up our invocation in their notation. Consider the equivalent optimization problem

$$
\operatorname* { m i n } _ { \substack { r \in \mathbb { R } ^ { m } , \beta \in \mathbb { R } ^ { d } : r + \mathbf { X } \beta = \mathbf { Y } } } \| r \| _ { q } ^ { q } ,
$$

where r represents the residuals. In their notation, we represent $x = \left( r \quad \beta \right)$ and let

$$
\mathbf { A } = [ I _ { m } \quad \mathbf { X } ] , \qquad b = \mathbf { Y } , \qquad \mathbf { d } = \mathbf { M } = 0 , \qquad \mathbf { N } = \mathrm { d i a g } ( I _ { m } , 0 _ { d } ) , \qquad \mathbf { x } ^ { ( 0 ) } = ( \mathbf { Y } \quad \mathbf { 0 } _ { d } ) .
$$

Their algorithm uses $O \left( q m ^ { 1 / 3 } \operatorname* { m a x } \left\{ 1 , \log \left( \frac { \| \mathbf { Y } \| _ { q } ^ { q } } { \varepsilon } \right) \right\} \right)$ calls to a linear system solver.

We must relate the objective error to the coeficient error $\left\| \widehat { A } _ { q } - \widehat { \beta } _ { q } \right\| _ { 2 } ,$ , so we may choose the appropriate value of ε. We use the following form of Clarkson’s inequalities

$$
\left\| \frac { f + g } 2 \right\| _ { q } ^ { q } + \left\| \frac { f - g } 2 \right\| _ { q } ^ { q } \leq \frac 1 2 \left( \| f \| _ { q } ^ { q } + \| g \| _ { q } ^ { q } \right) ,
$$

with $f = \mathbf { Y } - \mathbf { X } { \widehat { \beta } } _ { q } { \mathrm { ~ a n d ~ } } g = \mathbf { Y } - \mathbf { X } { \widehat { A } } _ { q }$ , which yields,

$$
\begin{array} { r l } & { \quad | { \mathbb { Y } } - { \mathbb { X } } ( \frac { \hat { \mathcal { X } } _ { 1 } } { 2 } , \frac { \hat { \mathcal { X } } _ { 2 } } { 2 } ) | _ { \phi } ^ { \phi } \| { \mathbb { X } } ( \frac { \hat { \mathcal { X } } _ { 2 } } { 2 } , \frac { \hat { \mathcal { X } } _ { 2 } } { 2 } ) \| _ { \phi } ^ { \phi } \leq \frac { 1 } { 2 } ( \| { \mathbb { Y } } - { \mathbb { X } } \hat { \mathcal { X } } _ { 1 } \| _ { \phi } ^ { 2 } + \| { \mathbb { Y } } - { \mathbb { X } } \hat { \mathcal { X } } _ { 2 } \| _ { \phi } ^ { 2 } ) } \\ & { \implies \| { \mathbb { Y } } - { \mathbb { X } } \hat { \mathcal { X } } _ { 1 } \| _ { \phi } ^ { \phi } \| { \mathbb { X } } ( \hat { \mathcal { Y } } ^ { \hat { \mathcal { X } } _ { 2 } } - \hat { \mathcal { X } } _ { 2 } ^ { \hat { \mathcal { X } } } ) \| _ { \phi } ^ { \phi } \leq \frac { 1 } { 2 } ( | { \mathbb { Y } } - { \mathbb { X } } \hat { \mathcal { X } } _ { 1 } ^ { \phi } | _ { \phi } ^ { 3 } ) \| { \mathbb { Y } } - { \mathbb { X } } \hat { \mathcal { X } } _ { 1 } \| _ { \phi } ^ { \phi } \| _ { \phi } ^ { 3 } } \\ &  \implies | { \mathbb { X } } ( \frac { \hat { \mathcal { X } } _ { 2 } } { 2 } , \frac { \hat { \mathcal { X } } _ { 2 } } { 2 } ) \| _ { \phi } ^ { \phi } \leq \frac { 1 } { 2 } ( \| { \mathbb { Y } } - { \mathbb { X } } \hat { \mathcal { X } } _ { 1 } \| _ { \phi } ^ { 2 } - \| { \mathbb { Y } } - { \mathbb { X } } \end{array}
$$

Hence, $\left\| \widehat { \beta } _ { q } - \widehat { A } _ { q } \right\| _ { 2 } \leq \gamma$ if we choose

$$
\varepsilon = m \left( c _ { K , \kappa } \gamma ^ { 2 } \right) ^ { q / 2 } ,
$$

for a small enough $c _ { K , \kappa }$ . Thus, the number of calls to a linear system solver is at most

$$
O \left( q m ^ { 1 / 3 } \operatorname* { m a x } \left\{ 1 , \log \left( \frac { \| { \bf Y } \| _ { q } ^ { q } } { \varepsilon } \right) \right\} \right) \leq O \left( q ^ { 2 } m ^ { 1 / 3 } \operatorname* { m a x } \left\{ 1 , \log \left( \frac { C _ { K , \kappa } m \| { \bf Y } \| _ { \infty } } { \gamma } \right) \right\} \right)
$$

## D.4 Smoothed uniform errors example

In this section, we discuss the example where the errors are distributed like $\mathrm { U n i f } [ - 1 , 1 ] * N ( 0 , n ^ { - 2 a } )$ We first control the Hellinger modulus:

Lemma D.3. There exist constants $c , C , \varepsilon _ { 0 } > 0$ such that the following holds. Let $\varepsilon \in ( 0 , \varepsilon _ { 0 } ]$ ， $a \in [ 0 , 1 ]$ , and the distribution $p$ is the convolution $\mathrm { U n i f } [ - 1 , 1 ] * N ( 0 , n ^ { - 2 a } )$ . Then, the Hellinger modulus satisfies

$$
c \left( \sqrt { n ^ { - a } \varepsilon } + \varepsilon \right) \leq \omega _ { p } ( \varepsilon ) \leq C \left( \sqrt { n ^ { - a } \varepsilon } + \varepsilon \right) .
$$

Proof. Consider $t \in [ 0 , 1 ]$ . We will control $H _ { p } ( t )$ , and then invert this to control the Hellinger modulus. Let $u ( x )$ be the density of $\mathrm { U n i f } [ - 1 , 1 ] , \sigma = n ^ { - a } , \phi _ { \sigma } ( x )$ is the density of $N ( 0 , \sigma ^ { 2 } )$ , and $\overline { { \Phi } } _ { \sigma } ( x ) = \mathbb { P } _ { z \sim N ( 0 , \sigma ^ { 2 } ) } [ z \geq x ]$

Upper bounding $H _ { p } ( t )$ . First, by data processing inequality, we may bound

$$
\mathrm { H } ^ { 2 } ( p , p _ { t } ) \lesssim \mathrm { T V } ( u , u _ { t } ) = \frac { t } { 2 } .
$$

For our second upper bound, let $f ( x ) = { \sqrt { p ( x ) } }$ . Our bound will require control over $f ^ { \prime } ( x )$ , and hence control over $p ^ { \prime } ( x )$ . We may write $p ^ { \prime } ( x )$ as

$$
p ^ { \prime } ( x ) = - \int _ { - 1 } ^ { 1 } \frac { x - y } { \sigma ^ { 2 } } \cdot \frac { 1 } { 2 } \phi _ { \sigma } ( x - y ) d y .
$$

Using Cauchy-Schwarz, we may bound,

$$
\begin{array} { l } { p ^ { \prime } ( x ) ^ { 2 } \leq \left( \displaystyle \int _ { - 1 } ^ { 1 } \frac { ( x - y ) ^ { 2 } } { \sigma ^ { 4 } } \cdot \frac { 1 } { 2 } \phi _ { \sigma } ( x - y ) d y \right) \left( \displaystyle \int _ { - 1 } ^ { 1 } \frac { 1 } { 2 } \phi _ { \sigma } ( x - y ) d y \right) } \\ { \leq \left( \displaystyle \int _ { - \infty } ^ { \infty } \frac { z ^ { 2 } } { \sigma ^ { 4 } } \cdot \frac { 1 } { 2 } \phi _ { \sigma } ( z ) d z \right) p ( x ) = \frac { p ( x ) } { 2 \sigma ^ { 2 } } } \\ { \Longrightarrow \left| p ^ { \prime } ( x ) \right| \lesssim \sqrt { p ( x ) } / \sigma . } \end{array}
$$

This further implies $\| f ^ { \prime } \| _ { \infty } \lesssim 1 / \sigma$ . We may thus bound the squared Hellinger distance by

$$
\begin{array} { r l r } {  { H _ { p } ( t ) \lesssim \| f - f ( \cdot - t ) \| _ { 2 } ^ { 2 } \leq \| f - f ( \cdot - t ) \| _ { \infty } \| f - f ( \cdot - t ) \| _ { 1 } \leq \big ( t \| f ^ { \prime } \| _ { \infty } \big ) \cdot \big ( t \| f ^ { \prime } \| _ { 1 } \big ) } } \\ & { } & { \lesssim \frac { t ^ { 2 } } { \sigma } \| f ^ { \prime } \| _ { 1 } = \frac { t ^ { 2 } } { \sigma } \cdot 2 f ( 0 ) \lesssim \frac { t ^ { 2 } } { \sigma } . } \end{array}
$$

Together, both upper bounds yield

$$
H _ { p } ( t ) \lesssim \operatorname* { m i n } \left\{ t , \frac { t ^ { 2 } } { \sigma } \right\} .
$$

Lower bounding $H _ { p } ( t )$ . Consider the tail halfline $A = ( 1 , \infty )$ , and let $\alpha = \mathbb { P } _ { x \sim p } ( x \in A )$ ， $\beta = \mathbb { P } _ { x \sim p _ { t } } ( x \in A )$ . Using data processing inequality,

$$
H _ { p } ( t ) \geq \mathrm { H } ^ { 2 } ( \mathrm { B e r n } ( \alpha ) , \mathrm { B e r n } ( \beta ) ) \geq \frac { 1 } { 2 } \left( \sqrt { \alpha } - \sqrt { \beta } \right) ^ { 2 } = \frac { ( \beta - \alpha ) ^ { 2 } } { 2 ( \sqrt { \alpha } + \sqrt { \beta } ) ^ { 2 } } \gtrsim \frac { ( \beta - \alpha ) ^ { 2 } } { \alpha + \beta } .
$$

The relevant quantities are

$$
\alpha = \frac { 1 } { 2 } \int _ { 0 } ^ { 2 } \overline { { \Phi } } _ { \sigma } ( x ) d x = \frac { 1 } { 2 } \int _ { 0 } ^ { 2 } \overline { { \Phi } } _ { 1 } ( x / \sigma ) d x \lesssim \sigma + \int _ { \sigma } ^ { 2 } \phi _ { 1 } ( x / \sigma ) d x \lesssim \sigma , \qquad \beta - \alpha = \int _ { 1 - t } ^ { 1 } p ( x ) d x \asymp t .
$$

This immediately yields the bound

$$
H _ { p } ( t ) \gtrsim \frac { t ^ { 2 } } { \sigma + t } \asymp \operatorname* { m i n } \left\{ t , \frac { t ^ { 2 } } { \sigma } \right\} .
$$

Concluding the Hellinger modulus control. We have shown that for any $t \in [ 0 , 1 ]$ it holds that

$$
H _ { p } ( t ) \asymp \frac { t ^ { 2 } } { \sigma + t } .
$$

Thus, for $\varepsilon \le \varepsilon _ { 0 }$ for a suficiently small constant $\varepsilon _ { 0 } .$ , the largest possible t is

$$
t \asymp \sqrt { \sigma \varepsilon } + \varepsilon .
$$

Now that we have control over the Hellinger modulus, we may apply our results to this setting. Our lower bound (Theorem 1.8) implies that any estimator must incur error

$$
\gtrsim \operatorname* { m a x } \left\{ \sqrt { \frac { d } { n ^ { 1 + a } } } , \frac { d } { n } \right\} ,
$$

with probability at least $\frac { 3 } { 8 }$ , for standard Gaussian covariates.

Under the conditions of our upper bound (Theorem 1.6) and $n \geq C _ { K , \kappa } d \log ^ { 4 } ( 6 4 n / \delta )$ for a suficiently large $C _ { K , \kappa } \geq 1$ , then an estimator may incur error at most

$$
C _ { K , \kappa } \mathrm { p o l y l o g } ( n , 1 / \delta ) \cdot \operatorname* { m a x } \left\{ \sqrt { \frac { d } { n ^ { 1 + a } } } , \frac { d } { n } \right\} ,
$$

with probability at least $1 - \delta$