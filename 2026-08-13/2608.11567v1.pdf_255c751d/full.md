# Sparse and robust geometric twin support vector machine via asymmetric RoBoSS loss function

Kai Qi<sup>a</sup>, Xinji Huang<sup>b</sup>, Hongchun Wang<sup>b,∗</sup>

<sup>a</sup>National Center for Applied Mathematics in Chongqing, Chongqing Normal University, Chongqing, 401331, China.

<sup>b</sup>School of Mathematical Sciences, Chongqing Normal University, Chongqing, 401331, China.

## Abstract

In real-world scenarios, the training data usually contains redundant features, label noise and feature noise, which provide severe challenges for the eficiency of machine learning methods. Since standard support vector machine (SVM) adopts $l _ { 2 } .$ -norm penalty and hinge loss function, it lacks the ability of selecting significant features and is sensitive to noise. To address these issues, this paper proposes a novel asymmetric, robust, bounded, sparse and smooth (aR) loss function for $l _ { 1 } { \mathrm { - n o r m } }$ penalized geometric twin SVM (aRSGTSVM) to handle classification and regression tasks. The $l _ { 1 }$ -norm penalty can achieve the feature selection. The proposed aR loss function can not only efectively mitigate the impact of label noise, but also significantly enhance the stability to resampling noise, i.e., the zero-mean feature noise around the boundary hyperplanes. Furthermore, a statistical analysis of the robustness of aRSGTSVM was also conducted using the influence function. Since aRSGTSVM involves nonconvex and nonsmooth optimization, we develop a fast and stable proximal gradient descent based solving algorithm. Compared with related state-of-the-art methods, experimental results demonstrate the superiority of the proposed aRSGTSVM on both synthetic and UCI datasets. Furthermore, we apply aRSGTSVM to index tracking tasks, where results for tracking the diferent indices in the China stock market show that it can achieve satisfactory performance.

Keywords: Feature selection, robustness, geometric twin classification and regression, asymmetric RoBoSS loss function, nonconvex and nonsmooth optimization

## 1. Introduction

Twin support vector machine (TSVM) [1] is one of the well-known variants of the support vector machine, which has experienced prosperous developments in the past decade and has been widely applied into diferent fields, such as digit recognition [2], medical diagnosis [3, 4] and bioanalysis [5]. From a geometric perspective, TSVM aims to find a pair of nonparallel separating hyperplanes, where each hyperplane is required to be close to the samples of one class while being far from those of the other class. This makes TSVM more flexible and eficient than standard SVM. Moreover, similar to SVM, the standard TSVM can also be explained from a statistical point of view, i.e., it fits the “loss + penalty” regularization framework [3, 6]. Specifically, most of the existing TSVM adopt hinge loss and $l _ { 2 } .$ -norm penalty. However, hinge loss is proved to be sensitive to label noise [7, 8] and resampling noise (zero-mean feature noise around the boundary hyperplanes) [9]. The $l _ { 2 } .$ -norm penalty lacks the ability to perform feature selection [10, 11]. Therefore, there remains significant room for improving the eficiency of TSVM in handling complex real-world problems.

The label noise sensitivity of SVMs and TSVMs stems mainly from the upperunbounded nature of the hinge loss. Specifically, label noise usually lies near or even on the wrong side of the separating hyperplane. For such samples, the losses can be quite large due to the unboundedness of hinge loss function. Consequently, the final obtained separating hyperplane tends to be deviated by label noise by the minimization of the optimization objective. To address this issue, researchers have turned to nonconvex loss functions to enhance robustness against label noise. By considering the diference between the logistic loss and its shifted version, Krause and Singer [12] proposed logistic diference (LD) loss for SVM to reduce the influence of label noise. Wu and Liu [7] integrated a truncated hinge loss, so-called ramp loss, with SVM to enhance the robustness. Later, Liu et al. [13] combined the ramp loss with TSVM to propose robust nonparallel SVM (RNPSVM). Wang et al. [14] constructed a capped $l _ { 1 }$ -norm loss for robust TSVM. Wang et al. [8] designed a new truncated squared loss to obtain a robust SVM $( L _ { t s l }  – \mathbf { S } \mathbf { V } \mathbf { M } )$ . In addition to truncation, correntropy is also an effective approach of designing robust losses [15, 16]. Xu et al. [17] developed the idea of correntropy to SVM and proposed rescaled hinge loss SVM, which can alleviate the disturbance of outliers. Xu et al. [18] studied the correntropy based loss (C-loss) for least squares SVM. Ma et al. [19] constructed a novel adaptive capped loss derived from correntropy to enhance TSVM’s resistance to label noise. Recently, Akhtar et al. [20] introduced a novel robust, bounded, sparse and smooth (RoBoSS) loss function for SVM, to mitigate the impact of label noise. However, RoBoSS loss ignores the perturbation of resampling noise and the performance needs further improving.

Regarding resampling noise, Huang et al. [9] first demonstrated that hinge lossbased separating hyperplanes are sensitive to zero-mean feature noise near the boundary. Consequently, solutions derived from resampling procedures, such as K-fold cross-validation lack stability. Inspired by statistical quantiles, they introduced the pinball loss to SVM (PinSVM) to enhance its resampling stability. The authors [21] further extended this idea to squared loss, developing an asymmetric least squares SVM with a smooth, easily optimized objective function. A key limitation of PinSVM is the loss of sample sparsity. Specifically, all training samples become support vectors. This drawback can increase the training burden [9]. To encourage the sparsity, Shen et al. [22] proposed a truncated pinball loss, which can provide a flexible trade-of between sparsity and feature noise insensitivity. Motivated by quantiles and correntropy, Yang and Dong [23] proposed a generalized quantile loss. Based on the least squares SVM, He et al. [24] established an asymmetric kernel-based learning classifier.

For redundant features, they often provide irrelevant or even misleading information for the modelling process. As a result, the predicting performance may degrade. Zhu et al. [10] replaced $l _ { 2 } { \mathrm { - n o r m } }$ penalty by $l _ { 1 } { \mathrm { - n o r m } }$ penalty to propose 1-norm SVM, which can select significant features and remove redundant features, simultaneously. Ikeda and Murata [25] investigated the geometrical properties of $l _ { p }$ -norm penalized ν- SVM, where $1 \leq p \leq \infty$ . Zhu et al. [11] integrated elastic net with SVM to construct doubly regularized SVM (DrSVM). In many real-world scenarios, input features exhibit a grouped structure. Consequently, there is greater interest in performing feature selection or elimination at the group level rather than on an individual basis. To achieve group selection, Zou and Yuan [26] applied the infinity norm for each group of features to propose $F _ { \infty } \mathrm { . }$ -nrom SVM. Gao et al. [27] employed $l _ { 1 }$ -norm penalty for least squares TSVM to automatically select input features. Moosaei and Hladík [28] proposed $l _ { p ^ { - } }$ norm $( 0 < p < 1 )$ least squares twin multi-class SVM. With $p \geq 1$ , Xie et al. [29] designed Laplacian $l _ { p }$ -norm least squares TSVM.

Despite advances in robust SVM methods, existing approaches sufer critical limitations that hinder practical use. Label-noise-robust losses such as ramp loss, capped loss, and RoBoSS typically ignore resampling noise, while resampling-stable variants like pinball loss remain sensitive to label noise. Most robust losses either sacrifice sparsity or introduce nonconvexity with hyperparameters lacking theoretical tuning guidance. Moreover, few prior methods address label noise, resampling instability, and redundant feature interference simultaneously within a unified framework. Notably, RoBoSS lacks resampling-noise resistance and its feature-selection performance remains unexplored. In this paper, to address the aforementioned issues, we propose a novel asymmetric RoBoSS loss-based sparse geometric TSVM (aRSGTSVM). The proposed asymmetric RoBoSS loss is smooth and bounded, which can mitigate the impact of label noise and resampling noise, simultaneously. Moreover, aRSGTSVM incorporates an $l _ { 1 }$ -norm penalty to enable feature selection.

In short, the main contributions are summarized as follows:

1) We propose a new asymmetric and robust (aR) loss function. Especially, compared with RoBoSS loss function, the proposed aR loss function can not only efectively mitigate the impact of label noise, but also enhance the stability to the zero-mean noise near the boundary hyperplanes (so-called resampling noise).

2) We employ influence function to demonstrate the robustness of aR loss function. Although there is extensive literature on enhancing SVM robustness by constructing non-convex functions, few studies theoretically elucidate the robustness of the constructed functions. From a statistical perspective, we further proved that the influence function of aR loss is bounded, thereby providing a theoretical guarantee of its robustness.

3) Based on the aR loss and $l _ { 1 }$ -norm penalty, the aRSGTSVM models are proposed for robust classification and regression learning problems (In the following content, we refer to the regression model as aRSGTSVR to avoid confusion). In addition to robustness, the proposed aRSGTSVM and aRSGTSVR can also reduce the influence of noise features.

4) To optimize the nonconvex and nonsmooth aRSGTSVM and aRSGTSVR optimization problems, we design an eficient algorithm based on the proximal gradient descent algorithm, termed as iPiano algorithm. The implemented algorithm is demonstrated to be fast and stable, and is applicable to high-dimensional learning problems.

5) A lot of numerical studies on synthetic and benchmark datasets demonstrate that the proposed aRSGTSVM and aRSGTSVR are robust against label noise and resampling noise, while maintaining excellent feature selection capability in highdimensional settings.

6) To further validate the generalization capability, we apply the model into index tracking task. The results show that aRSGTSVR delivers consistently superior performance across diferent underlying indices.

The rest of paper is organized as follows: Section 2 introduces the related work of this paper. Section 3 proposes the aR loss function and applies it to both classification and regression problems. Section 4 presents a lot of experimental results of the proposed models on artificial datasets, UCI datasets and real-world stock datasets. Finally, in Section 5, we revisit the main contributions of this paper and provide a concluding remark.

## 2. Preliminaries and related work

First, we define some necessary notation. Consider a supervised learning data $\{ ( x _ { i } ^ { T } , y _ { i } ) ^ { T } \} _ { i = 1 } ^ { n }$ with n samples and p features, where $x _ { i } = ( x _ { i 1 } , x _ { i 2 } , \cdot \cdot \cdot , x _ { i p } ) ^ { T } \in \mathbb { R } ^ { p }$ is the i-th sample. Note that $y _ { i } ~ \in ~ \{ \pm 1 \}$ means a binary classification problem, while $y _ { i } \in \mathbb { R }$ means a regression problem. Furthermore, let $Y = ( y _ { 1 } , y _ { 2 } , \cdot \cdot \cdot y _ { n } ) ^ { T } \in \mathbb { R } ^ { n }$ and $X = ( x _ { 1 } , x _ { 2 } , \cdot \cdot \cdot x _ { n } ) ^ { T } \in \mathbb { R } ^ { n \times p }$ . The positive and the negative samples are reorganized as $X _ { + }$ and $X _ { - }$ , respectively. Unless otherwise specified, all the vectors mentioned below are in column-form and the norm defaults to the $l _ { 2 } { \mathrm { - n o r m } }$

## 2.1. ENNHSVM

For the standard TSVM, its training process does not fully consider the core requirement of comparing distances to each separating hyperplane in the prediction stage, leading to an inconsistency between the training and prediction processes. As [30] pointed out, this inconsistency can possibly reduce the prediction performance, especially in the datasets with heteroscedastic noise.

Recently, Qi and Yang [3] proposed a novel consistent ENNHSVM, which performs better than the standard TSVM. Specifically, ENNHSVM pursues two nonparallel hyperplanes by optimizing the following problems:

$$
\begin{array} { l } { \operatorname* { m i n } \frac { C _ { 1 } } { 2 } \left( \| X _ { + } w _ { + } + e _ { + } b _ { + } \| ^ { 2 } + \| X _ { - } w _ { - } + e _ { - } b _ { - } \| ^ { 2 } \right) } \\ { \quad \ + \ \frac { 1 } { 2 } \left( \| w _ { + } \| ^ { 2 } + b _ { + } ^ { 2 } + \| w _ { - } \| ^ { 2 } + b _ { - } ^ { 2 } \right) } \\ { \quad \ + \ \frac { C _ { 2 } } { 2 } \left( \| \xi _ { + } \| ^ { 2 } + \| \xi _ { - } \| ^ { 2 } \right) + c _ { 3 } \left( e _ { + } ^ { T } \xi _ { + } + e _ { - } ^ { T } \xi _ { - } \right) } \\ { \ } \\ { \mathrm { s . t . } \ \left\{ X _ { + } w _ { + } + e _ { + } b _ { + } + X _ { + } w _ { - } + e _ { + } b _ { - } \geq e _ { + } - \xi _ { + } , \xi _ { + } \geq 0 , \right. } \\ { } \\ { \ } \\ { X _ { - } w _ { - } + e _ { - } b _ { - } + X _ { - } w _ { + } + e _ { - } b _ { + } \leq \xi _ { - } - e _ { - } , \xi _ { - } \geq 0 , } \end{array}\tag{1}
$$

where $w _ { \pm }$ and $b _ { \pm }$ are the normal vectors and intercepts of the positive and negative hyperplanes. $c _ { i } ( i = 1 , 2 , 3 )$ are positive tuning parameters, $\xi _ { \pm }$ are slack variables and $e _ { \pm }$ are two vectors with all elements equaling to one.

After obtaining $w _ { \pm }$ and $b _ { \pm }$ , for a given new sample $x _ { \mathrm { n e w } }$ , its label can be classified via the following decision function:

$$
f ( x _ { \mathrm { n e w } } ) = \mathrm { s i g n } \big ( x _ { \mathrm { n e w } } ^ { T } ( w _ { + } + w _ { - } ) + ( b _ { + } + b _ { - } ) \big ) ,\tag{2}
$$

where sign(·) denotes the sign function.

Although ENNHSVM is consistent and demonstrated to be eficient, it cannot perform feature selection and exhibits poor performance in the presence of redundant variables. Moreover, its application in regression has not yet been explored in depth.

## 2.2. RoBoSS loss for SVM

To reduce the impact of label noise, recently, Akhtar et al. [20] proposed a robust, bounded, sparse and smooth loss (RoBoSS). They incorporated the RoBoSS loss into the SVM framework, resulting in a novel robust RoBoSS-SVM.

In detail, RoBoSS loss is defined as

$$
L _ { R o B o S S } ( u ) = \left\{ \begin{array} { l l } { \lambda ( 1 - ( a u + 1 ) \exp ( - a u ) ) , } & { u > 0 , } \\ { } & { u \leq 0 , } \end{array} \right.\tag{3}
$$

where a is the shape parameter and λ is the boundary parameter. The primal problem of RoBoSS-SVM can be formulated as

$$
\operatorname* { m i n } _ { w , b } \frac { 1 } { 2 } \| w \| ^ { 2 } + \frac { C } { n } \sum _ { i = 1 } ^ { n } L _ { \mathrm { R o B o S S } } \left( 1 - y _ { i } \left( w ^ { T } x _ { i } + b \right) \right) .\tag{4}
$$

where w and b are the normal vector and intercept of the separating hyperplane. $C > 0$   
is the tuning parameter.

## 3. The proposed aRSGTSVM(R)

As discussed earlier, although the RoBoSS loss is robust to noise, it exhibits instability to resampling. In this chapter, an asymmetric version of the RoBoSS loss, termed aR loss, is proposed to address more complex noise in high-dimensional learning. Specifically, Section 3.1 introduces the aR loss and investigates its theoretical properties. Section 3.2 presents the aRSGTSVM method for classification problems, while Section 3.3 proposes the aRSGTSVR model for regression scenarios. Finally, Section 3.4 provides the corresponding optimization algorithm.

## 3.1. asymmetric RoBoSS loss

Although the RoBoSS loss demonstrates considerable robustness to label noise, Huang et al. [31] point out that in many practical problems, beyond outliers, highdimensional complex data may also be afected by other types of noise, such as resampling noise or zero-mean noise near the bounding hyperplane. However, one-sided loss functions like the hinge loss and RoBoSS loss struggle to efectively mitigate the impact of such noise, leading to unstable model performance and leaving room for improvement. Therefore, to further enhance the capability of the RoBoSS loss in handling complex data, we propose a new loss function named aR, which is defined as follows:

$$
L _ { a R } ( u ) = \left\{ \begin{array} { l l } { \lambda ( 1 - ( a u + 1 ) \exp ( - a u ) ) , } & { u > 0 , } \\ { \tau \lambda ( 1 - ( a u ^ { 2 } + 1 ) \exp ( - a u ^ { 2 } ) ) , } & { u \leq 0 , } \end{array} \right.\tag{5}
$$

where the shape parameter $a > 0$ , the margin parameter $\lambda > 0$ , and the parameter $\tau \in [ 0 ,$ 1] control the asymmetry of the loss function, thereby enhancing the model’s robustness to resampling noise.

Next, we elaborate on the properties of the proposed aR loss function.

Property 1. The aR loss function is $C ^ { 1 }$ -smooth.

Proof. First, by (5), it follows that

$$
\begin{array} { r } { \nabla L _ { a R } ( u ) = \left\{ \begin{array} { l l } { \lambda a ^ { 2 } u \exp ( - a u ) , } & { u > 0 , } \\ { 2 \tau \lambda a ^ { 2 } u ^ { 3 } \exp ( - a u ^ { 2 } ) , } & { u \leq 0 , } \end{array} \right. } \end{array}\tag{6}
$$

Thus, we can deduce that $\nabla L _ { a R } ( 0 ^ { + } ) = 0$ and $\nabla L _ { a R } ( 0 ^ { - } ) = 0$ . According to (5), we have $\begin{array} { r } { \operatorname* { l i m } _ { u  0 ^ { + } } L _ { a R } ( u ) = \operatorname* { l i m } _ { u  0 ^ { - } } L _ { a R } ( u ) = L _ { a R } ( 0 ) = 0 } \end{array}$ , which implies that $L _ { a R } ( u )$ is continuous at $u = 0$ . Consequently, the aR loss is a $C ^ { 1 }$ -smooth function. □

Property 2. The aR loss function is bounded.

Proof. For $u \ > \ 0 ;$ , we have $\nabla L _ { a R } ( u ) \ = \ \lambda a ^ { 2 } u \exp ( - a u ) \ > \ 0 ,$ which implies that $L _ { a R } ( u )$ is strictly monotonically increasing on $( 0 , + \infty )$ . For $u \ < \ 0$ , we have $\nabla L _ { a R } ( u ) = 2 \tau \lambda a ^ { 2 } u ^ { 3 } \exp ( - a u ^ { 2 } ) < 0$ , which implies that $L _ { a R } ( u )$ is strictly monotonically decreasing on $( - \infty , 0 )$ . According to the monotonicity above, the function reaches its global minimum at $u = 0$ , and $L _ { a R } ( 0 ) = 0$ , which means that the loss function is lower bounded by 0.

Next, we analyze the upper bound of the function by calculating the limits at infinity:

$$
\operatorname * { l i m } _ { u  + \infty } { \cal L } _ { a R } ( u ) = \lambda , \ : \ : \ : \ : \ : \operatorname * { l i m } _ { u  - \infty } { \cal L } _ { a R } ( u ) = \tau \lambda .\tag{7}
$$

Since $0 < \tau \leq 1$ , we have τλ $\leq \lambda .$ . Combined with monotonicity on both intervals, all function values satisfy

$$
0 \leq L _ { a R } ( u ) \leq \lambda , \quad \forall u \in \mathbb { R } .\tag{8}
$$

Therefore, the aR loss function is both lower bounded and upper bounded, namely, it is bounded on R. □

Property 3. The aR loss is robust to label noise (outliers), which can be theoretically guaranteed by influence function.

Proof. Hampel introduced the influence function [32], which is mainly used to measure the stability of an estimator when subjected to infinitesimal contamination. For an ideal roust loss function, the influence function of the induced estimator is bounded.

Following [33] and by (6), we have that $\nabla l _ { a R }$ achieves its maximum at $\textstyle u = { \frac { 1 } { a } }$ for $u > 0$ , while it achieves its maximum at $u = - \sqrt { \frac { 3 } { 2 a } }$ for $u \leq 0$ . Therefore, it follows that

$$
| \mathrm { I F } ( u ) | \leq \operatorname* { m a x } \left( \frac { \lambda a } { \mathrm { e } } , \frac { 3 \tau \lambda \sqrt { 6 a } } { 2 \mathrm { e } ^ { 3 / 2 } } \right) < \infty , \quad \forall u \in \mathbb { R } .\tag{9}
$$

Therefore, according to the influence function theory, for the SVM model designed based on the aR loss, even if the loss u caused by label noise is extremely large, their impact on the model is still limited. This theoretically guarantees the robustness of the aR loss.

Figure 1 illustrates the various forms of the aR loss function under diferent parameters. Combined with the preceding propositions, the asymmetric RoBoSS (aR) loss function is a bounded, asymmetric, smooth, and non-convex function. Consequently, compared to the original RoBoSS loss, our proposed loss function is not only robust to outliers but also resilient to zero-mean feature noise, enabling it to handle more complex problems.

![](images/576d6ebae8ee91ad30c40350822e00324a8b6286ae6ce902bb73ce172129ca61.jpg)  
(a) Diferent a values

![](images/0366d47abbc99d211e336382d3c9ddc4296b60e70b00f2c41a3cbc94c27cce92.jpg)  
(b) Diferent λ values

![](images/0763b29a50a0d783280c95039c3f3a0c45daf04231e18f34850cbfd9a5239075.jpg)  
(c) Diferent τ values  
Figure 1: Illustrations of diferent parameter values of the proposed aR loss function

## 3.2. aR loss-based sparse geometric TSVM

In this section, we mainly study the classification problems. Since the proposed aR loss can efectively mitigate the impact of outliers and strengthen the stability to resampling noise, simultaneously. We propose a novel sparse and robust geometric twin support vector machine(aRSGTSVM), which integrates the aR loss function with ENNHSVM, thereby inheriting the consistency properties of ENNHSVM. The resulting expression is as follows:

$$
\begin{array} { r l } { \operatorname* { m i n } } & { \displaystyle \frac { C _ { 1 } } { 2 } \left( \| X _ { + } w _ { + } + e _ { + } b _ { + } \| ^ { 2 } + \| X _ { - } w _ { - } + e _ { - } b _ { - } \| ^ { 2 } \right) } \\ & { + \lambda ( \| w _ { + } \| _ { 1 } + | b _ { + } | + \| w _ { - } \| _ { 1 } + | b _ { - } | ) } \\ & { \displaystyle + c _ { 2 } \Bigg ( \sum _ { i \in { \cal N } _ { \uparrow } } { \cal L } _ { a b } \big ( ( \xi _ { + } ) _ { i } \big ) + \sum _ { i \in { \cal N } _ { \uparrow } } { \cal L } _ { a b } \big ( ( \xi _ { - } ) _ { i } \big ) \Bigg ) } \\ & { \displaystyle \left( X _ { + } w _ { + } + e _ { + } b _ { + } + X _ { + } w _ { - } + e _ { + } b _ { - } \geq e _ { + } - \xi _ { + } , \right. } \\ { \mathrm { S . t . } } &  \displaystyle \left. \sum _ { i \in \mathcal { N } _ { - } + \mathcal { C } - b _ { - } + X _ { - } w _ { + } + \mathcal { C } - b _ { + } \leq \xi _ { - } - e _ { - } \right) } \\ & { \displaystyle \Bigg \{ \xi _ { + } \geq 0 , \xi _ { - } \geq 0 , } \end{array}\tag{10}
$$

where $N _ { \pm }$ denotes by the index sets of positive and negative samples. $( \xi _ { \pm } ) _ { i }$ means the i-th element of the slack variables. Other notations are similar to ENNHSVM.

In contrast to ENNHSVM, our proposed classifier replaces the loss function in the original model with the aR loss and incorporates an l<sub>1</sub>-norm penalty to equip the model with feature selection capability. The core idea of the proposed aRSGTSVM is outlined as follows:

1) Minimizing the first term of the objective of (10) forces each hyperplane to be closer to its corresponding class samples and farther from the other class, enhancing inter-class discrimination.

2) The second term ∥cdot∥<sub>1</sub> induces sparsity in the model solution, enabling efective variable selection and handling of high-dimensional complex problems.

3) The third term employs the aR loss function, whose inherent properties provide robustness against label noise and resampling noise, thereby improving the model’s generalization performance.

To simplify subsequent derivations, define the following notations: $\tilde { X } _ { + } = ( X _ { + } , e _ { + } )$ , $\tilde { X } _ { - } = ( X _ { - } , e _ { - } ) , \tilde { w } _ { \pm } = ( w _ { \pm } ^ { T } , b _ { \pm } ) ^ { T }$ and $\tilde { w } = ( \tilde { w } _ { + } , \tilde { w } _ { - } )$ . Furthermore, let

$$
A = \left( \begin{array} { c c } { { \tilde { X } _ { + } ^ { T } \tilde { X } _ { + } } } & { { 0 } } \\ { { 0 } } & { { \tilde { X } _ { - } ^ { T } \tilde { X } _ { - } } } \end{array} \right) , \quad B = \left( \begin{array} { c c } { { \tilde { X } _ { + } } } & { { \tilde { X } _ { + } } } \\ { { - \tilde { X } _ { - } } } & { { - \tilde { X } _ { - } } } \end{array} \right) , \quad e = \left( \begin{array} { c } { { e _ { + } } } \\ { { e _ { - } } } \end{array} \right) , \quad \xi = \left( \begin{array} { c } { { \xi _ { + } } } \\ { { \xi _ { - } } } \end{array} \right) .\tag{11}
$$

Then, the aRSGTSVM problem (10) can be rewritten as:

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } \frac { c _ { 1 } } { 2 } \tilde { \boldsymbol { w } } ^ { T } \boldsymbol { A } \tilde { \boldsymbol { w } } + \lambda \| \tilde { \boldsymbol { w } } \| _ { 1 } + \frac { c _ { 2 } } { 2 } \| \boldsymbol { \xi } \| ^ { 2 } } \\ { \displaystyle \operatorname { s . t . } \quad \left\{ B \tilde { \boldsymbol { w } } \geq e - \boldsymbol { \xi } , \right. } \\ { \displaystyle \operatorname { s . t . } \quad \left\{ \xi \geq 0 . \right. } \end{array}\tag{12}
$$

which is equivalent to the following "loss + penalty" form, i.e.,

$$
\operatorname* { m i n } _ { \tilde { \boldsymbol { w } } } \frac { c _ { 1 } } { 2 } \tilde { \boldsymbol { w } } ^ { T } \boldsymbol { A } \tilde { \boldsymbol { w } } + c _ { 2 } \sum _ { i = 1 } ^ { n } L _ { a R } ( 1 - \boldsymbol { b } _ { i } ^ { T } \boldsymbol { \tilde { w } } ) + \boldsymbol { \lambda } \| \tilde { \boldsymbol { w } } \| _ { 1 } .\tag{13}
$$

where $b _ { i }$ represents the i-th row vector of matrix B.

## 3.3. aR loss-based sparse geometric TSVR

To extend the application scope from discrete classification to continuous prediction, we apply aRSGTSVM to regression problems and propose aR loss-based sparse geometric TSVR (aRSGTSVR).

Using the idea of TSVR [34], let $X _ { + } = ( X _ { + } , Y + \varepsilon _ { 1 } e _ { + } ) , X _ { - } = ( X _ { - } , Y - \varepsilon _ { 2 } e _ { - } ) , w _ { + } =$ $\left( w _ { + } ^ { T } , \eta _ { + } \right) ^ { T } , w _ { - } = \left( w _ { - } ^ { T } , \eta _ { - } \right) ^ { T }$ , where $\varepsilon _ { 1 } , \varepsilon _ { 2 } > 0$ are tuning parameters. We first transform (10) into the following regression problem:

$$
\begin{array} { l } { \operatorname* { m i n } \frac { c _ { 1 } } { 2 } \left( \| X _ { + } w _ { + } + \eta _ { + } ( Y + \varepsilon _ { 1 } e _ { + } ) + e _ { + } b _ { + } \| ^ { 2 } + \| X _ { - } w _ { - } + \eta _ { - } ( Y - \varepsilon _ { 2 } e _ { - } ) + e _ { - } b _ { - } \| ^ { 2 } \right) } \\ { \quad \quad + \lambda \left( \| w _ { + } \| _ { 1 } + | \eta _ { + } | + | b _ { + } | + \| w _ { - } \| _ { 1 } + | \eta _ { - } | + | b _ { - } | \right) } \\ { \quad \quad + c _ { 2 } \Bigg ( \displaystyle \sum _ { i \in \mathcal { N } _ { 1 } } L _ { a b } \left( ( \xi _ { + } ) _ { i } \right) + \displaystyle \sum _ { i \in \mathcal { N } _ { 1 } } L _ { a b } \left( ( \xi _ { - } ) _ { i } \right) \Bigg ) } \\ { \mathrm { ~ s i n . ~ } } \\ { \quad \quad \quad \Bigg \{ X _ { + } w _ { + } + \eta _ { + } ( Y + \varepsilon _ { 1 } e _ { + } ) + e _ { + } b _ { + } + X _ { + } w _ { - } + \eta _ { - } ( Y + \varepsilon _ { 1 } e _ { + } ) + e _ { + } b _ { - } \geq e _ { + } - \xi _ { + } , } \\  \quad \quad \Bigg \} _ { X = \nu _ { - } + \eta _ { - } ( Y - \varepsilon _ { 2 } e _ { - } ) + e _ { - } b _ { - } + X _ { - } w _ { + } + \eta _ { + } ( Y - \varepsilon _ { 2 } e _ { - } ) + e _ { - } b _ { + } \leq \xi _ { - } - e _ { - } < 0 , } \\  \quad \quad \Bigg \} _ { \xi _ { + } \geq 0 , \quad \xi _ { - } \geq 0 , } \end{array}\tag{14}
$$

Analogously to the classification problem, we further define the following variables: $\begin{array} { r } { \tilde { X } _ { + } \ = \ ( X _ { + } , Y + \varepsilon _ { 1 } e _ { + } , 1 ) , \ \tilde { X } _ { - } \ = \ ( X _ { - } , Y - \varepsilon _ { 2 } e _ { - } , 1 ) , \ \tilde { w } _ { + } \ = \ \left( w _ { + } ^ { T } , \eta _ { + } , b _ { + } \right) ^ { T } , \ \tilde { w } _ { - } \ = \ ( X _ { - } , Y - \varepsilon _ { 2 } e _ { - } , 1 ) , \ \tilde { w } _ { - } \ = \ ( X _ { + } , Y + \varepsilon _ { 1 } e _ { + } , 1 ) , } \end{array}$ $\left( w _ { - } ^ { T } , \eta _ { - } , b _ { - } \right) ^ { T } , \ \tilde { w } = \left( \tilde { w } _ { + } ^ { T } , \tilde { w } _ { - } ^ { T } \right) ^ { T }$ , as well as the corresponding matrices and vectors:

$$
A = \left( \begin{array} { c c c } { { \tilde { X } _ { + } ^ { T } \tilde { X } _ { + } } } & { { O } } \\ { { O } } & { { \tilde { X } _ { - } ^ { T } \tilde { X } _ { - } } } \end{array} \right) , B = \left( \begin{array} { c c c } { { \tilde { X } _ { + } } } & { { \tilde { X } _ { + } } } \\ { { - \tilde { X } _ { - } } } & { { - \tilde { X } _ { - } } } \end{array} \right) , e = \left( \begin{array} { c } { { e _ { + } } } \\ { { e _ { - } } } \end{array} \right) , \xi = \left( \begin{array} { c } { { \xi _ { + } } } \\ { { \xi _ { - } } } \end{array} \right) .
$$

We rewrite (14) as:

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } \frac { c _ { 1 } } { 2 } \tilde { \boldsymbol { w } } ^ { T } \boldsymbol { A } \tilde { \boldsymbol { w } } + \lambda \| \tilde { \boldsymbol { w } } \| _ { 1 } + \frac { c _ { 2 } } { 2 } \| \boldsymbol { \xi } \| ^ { 2 } } \\ { \displaystyle \operatorname { s . t . } \quad \left\{ B \tilde { \boldsymbol { w } } \geq e - \boldsymbol { \xi } , \right. } \\ { \displaystyle \operatorname { s . t . } \quad \left\{ \xi \geq 0 . \right. } \end{array}\tag{15}
$$

Similar to the previous steps, we obtain the linear case of the aRSGTSVR model:

$$
\operatorname* { m i n } \frac { c _ { 1 } } { 2 } \tilde { w } ^ { T } A \tilde { w } + c _ { 2 } \sum _ { i = 1 } ^ { n } L _ { a R } \left( 1 - b _ { i } ^ { T } \tilde { w } \right) + \lambda \| \tilde { w } \| _ { 1 } .\tag{16}
$$

By utilizing the feature mapping Φ(·) that maps raw input space into a high dimensional Reproducing Kernel Hilbert Space, the proposed aRSGTSVM(R) model can be extended to the sparse kernel case. In fact, by only replacing X with Φ(X), we derive

the aRSGTSVM(R) model for the sparse kernel scenario. Note that feature selection is one of the key focuses of this study, so we primarily consider the linear case

## 3.4. iPiano algorithm

Due to the non-convex aR loss function and non-smooth penalty term, Problem (10) constitutes a non-convex optimization problem. Conventional solution approaches typically employ the diference of convex functions (DC) algorithm or the half-quadratic (HQ) algorithm, which require iteratively solving a series of subproblems and incur substantial computational costs. Therefore, we opt to utilize the iPiano (Inertial Proximal Algorithm for Nonconvex Optimization) algorithm [35] for optimization, which ofers advantages of rapid convergence and numerical stability, particularly suited for high-dimensional complex problems. The following derivation will be presented using the classification problem as an illustrative example.

## 3.4.1. iPiano for aRSGTSVM

The iPiano algorithm can solve optimization problems of the form:

$$
\operatorname* { m i n } _ { x \in \mathbb { R } ^ { n } } h ( x ) = f ( x ) + g ( x ) ,\tag{17}
$$

where $g$ is convex (possibly nonsmooth), $f$ is $C ^ { 1 }$ -smooth (possibly nonconvex).

Thus, (10) can also be interpreted as having the form $" f + g "$

$$
\operatorname* { m i n } _ { \tilde { \boldsymbol { w } } } \underbrace { \frac { c _ { 1 } } { 2 } \tilde { \boldsymbol { w } } ^ { T } \boldsymbol { A } \tilde { \boldsymbol { w } } + c _ { 2 } \sum _ { i = 1 } ^ { n } L _ { a R } \left( 1 - b _ { i } ^ { T } \tilde { \boldsymbol { w } } \right) } _ { \boldsymbol { f } } + \underbrace { \boldsymbol { \lambda } \| \tilde { \boldsymbol { w } } \| _ { 1 } } _ { \boldsymbol { g } } .\tag{18}
$$

Let $u _ { i } = 1 - b _ { i } ^ { T } \tilde { w }$ , then the objective function f is given by:

$$
f ( \tilde { w } ) = \left\{ \begin{array} { l l } { \frac { c _ { 1 } } { 2 } \tilde { w } ^ { T } A \tilde { w } + c _ { 2 } \sum _ { i = 1 } ^ { n } \lambda \big ( 1 - ( a u _ { i } + 1 ) \exp { ( - a u _ { i } ) } \big ) , } & { u _ { i } > 0 , } \\ { \frac { c _ { 1 } } { 2 } \tilde { w } ^ { T } A \tilde { w } + c _ { 2 } \sum _ { i = 1 } ^ { n } \tau \lambda \left( 1 - \Big ( a u _ { i } ^ { 2 } + 1 \Big ) \exp { \left( - a u _ { i } ^ { 2 } \right) } \right) , } & { u _ { i } \le 0 . } \end{array} \right.\tag{19}
$$

Then $\nabla f$ is obtained:

$$
\nabla f ( \tilde { w } ) = \left\{ \begin{array} { l l } { \displaystyle c _ { 1 } A \tilde { w } - c _ { 2 } \sum _ { i = 1 } ^ { n } a ^ { 2 } \lambda b _ { i } u _ { i } \exp ( - a u _ { i } ) , } & { \displaystyle u _ { i } > 0 , } \\ { \displaystyle c _ { 1 } A \tilde { w } - c _ { 2 } \sum _ { i = 1 } ^ { n } 2 a ^ { 2 } \tau \lambda b _ { i } u _ { i } ^ { 3 } \exp ( - a u _ { i } ^ { 2 } ) , } & { \displaystyle u _ { i } \leq 0 . } \end{array} \right.\tag{20}
$$

Here, $L > 0$ denotes an upper bound constant for the Lipschitz continuity of $\nabla f ,$ which satisfies the inequality

$$
| \nabla f ( \tilde { w } _ { 1 } ) - \nabla f ( \tilde { w } _ { 2 } ) | \leq L \| \tilde { w } _ { 1 } - \tilde { w } _ { 2 } \| , \quad \forall \tilde { w } _ { 1 } , \tilde { w } _ { 2 } .\tag{21}
$$

Because

$$
\begin{array} { r } { \nabla ^ { 2 } f ( \tilde { w } ) = \left\{ \begin{array} { l l } { c _ { 1 } A - c _ { 2 } \sum _ { i = 1 } ^ { n } a ^ { 2 } \lambda b _ { i } b _ { i } ^ { T } ( - 1 + a u _ { i } ) \exp ( - a u _ { i } ) , } & { u _ { i } > 0 , } \\ { c _ { 1 } A - c _ { 2 } \sum _ { i = 1 } ^ { n } 2 a ^ { 2 } \tau \lambda b _ { i } b _ { i } ^ { T } \left( - 3 u _ { i } ^ { 2 } + 2 a u _ { i } ^ { 4 } \right) \exp ( - a u _ { i } ^ { 2 } ) , } & { u _ { i } \leq 0 , } \end{array} \right. } \end{array}\tag{22}
$$

we can get

$$
L = c _ { 1 } \| A \| + c _ { 2 } \lambda a ^ { 2 } \operatorname* { m a x } \left( 1 , \frac { 2 \left| \tau \right| } { a \sqrt { e } } \right) \sum _ { i = 1 } ^ { n } \| b _ { i } \| ^ { 2 } .\tag{23}
$$

The general framework of the iPiano algorithm for the problem (17) is presented in the algorithm 1. In this work, we empirically set the maximum iteration number $n _ { \mathrm { i t e r } }$ to 500 and the error threshold ε to $1 0 ^ { - 6 }$

The proximal map is defined by

$$
( I + \alpha \partial g ) ^ { - 1 } ( \hat { x } ) : = \underset { x \in \mathbb { R } ^ { n } } { \arg \operatorname* { m i n } } \left. \alpha g ( x ) + \frac 1 2 \| x - \hat { x } \| _ { 2 } ^ { 2 } \right. ,\tag{24}
$$

where I is the identity map, $\alpha > 0$ is a given step size parameter, and $g$ is a proper lower semicontinuous convex function whose specific definition is given in (18).

## 3.4.2. iPiano for aRSGTSVR

According to (16) and (13), the aRSGTSVR model can be reformulated into a form consistent with that of the aRSGTSVM. Thus, similarly to algorithm 1, the overall procedure for solving the aRSGTSVR algorithm is shown in algorithm 2.

Algorithm 1 iPiano for aRSGTSVM   
1: Input: Set step size parameters $\beta = 0 . 5 , \alpha = \frac { 1 - \beta } { L }$ , where L is the Lipschitz   
constant of $\nabla f ,$ choose $\tilde { w } ^ { 0 } \in \mathrm { d o m } , h , \tilde { w } ^ { - 1 } = \tilde { w } ^ { 0 }$ . The maximal iteration number   
$n _ { i t e r } ,$ and the convergent error $\varepsilon .$   
2: Output: Optimal solution $\tilde { w } ^ { * }$ of (10)   
3: Initialize iteration counter $n = 0$   
4: while $n < n _ { i t e r }$ do   
5: $\tilde { w } ^ { n + 1 } = ( I + \alpha _ { n } \partial g ) ^ { - 1 } \big ( \tilde { w } ^ { n } - \alpha _ { n } \nabla f ( \tilde { w } ^ { n } ) + \beta _ { n } ( \tilde { w } ^ { n } - \tilde { w } ^ { n - 1 } ) \big )$   
6: if $\| \tilde { w } ^ { n + 1 } - \tilde { w } ^ { n } \| _ { 2 } < \varepsilon$  then   
7: break   
8: end if   
9: $n = n + 1$   
10: end while   
11: return $\tilde { w } ^ { n + 1 }$

Algorithm 2 iPiano for aRSGTSVR   
1: Input: Set step size parameters $\beta = 0 . 5 , \alpha = \frac { 1 - \beta } { L }$ , where L is the Lipschitz   
constant of $\nabla f ,$ choose $\tilde { w } ^ { 0 } \in$ dom, h, $\tilde { w } ^ { - 1 } = \tilde { w } ^ { 0 }$ . The maximal iteration number   
$n _ { i t e r } ,$ and the convergent error $\varepsilon .$   
2: Output: Optimal solution $\tilde { w } ^ { * }$ of (16)   
3: Initialize iteration counter $n = 0$   
4: while $n < n _ { i t e r }$ do   
5: $\tilde { w } ^ { n + 1 } = ( I + \alpha _ { n } \partial g ) ^ { - 1 } \big ( \tilde { w } ^ { n } - \alpha _ { n } \nabla f ( \tilde { w } ^ { n } ) + \beta _ { n } ( \tilde { w } ^ { n } - \tilde { w } ^ { n - 1 } ) \big )$   
6: if $\| \tilde { w } ^ { n + 1 } - \tilde { w } ^ { n } \| _ { 2 } < \varepsilon$  then   
7: break   
8: end if   
9: $n = n + 1$   
10: end while   
11: return $\tilde { w } ^ { n + 1 }$

It is worth noting that the convergence of the proposed iPiano-based algorithm is guaranteed by Theorem 4.8 in the iPiano framework established by Ochs et al. [35]. Specifically, for nonconvex and possibly nonsmooth objective functions, under the condition that the objective satisfies the KurdykaLojasiewicz property and appropriate step size parameters are adopted, it can be verified that the iterative sequence of the iPiano algorithm converges to critical points. Consequently, the algorithm developed has a solid theoretical convergence guaranty.

## 4. Numerical studies

In this section, to further investigate the performance of our proposed algorithm, we design a series of numerical studies to compare aRSGTSVM and aRSGTSVR with some well-known and recent robust methods to verify the performance of our proposed aRSGTSVM and aRSGTSVR.

For the classification problem, we compared the classical 1-SVM [36], TPMSVM [37], Pin-TSVM [31], rhingeSVM [17], and RoBoSS-SVM [20]. The regularization parameter λ in 1-SVM is selected from the candidate set $\Lambda = \{ 0 . 0 1 , 0 . 0 2 , 0 . 0 3 , \ldots , 0 . 9 9 \}$ For TPMSVM, we select the values of ν from the set $\{ 0 . 1 , 0 . 2 , \ldots , 0 . 8 , 0 . 9 \}$ . For Pin-TSVM, the parameter ν from set $\{ 2 ^ { i } | i = - 8 , - 7 , . . . , 7 , 8 \}$ and τ is tuned in $\{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$ . In the rhingeSVM, the η is assumed to be 1. For RoBoSS-SVM, the parameters a and λ are selected from the set {1, 2, 3, 4, 5} and {0.5, 1, 1.5, 2} , respectively. For the aRSGTSVM, the parameters a and λ are the same as those of RoBoSS-SVM, and we take τ in the range {0.2, 0.5, 0.8}.

For the regression problem, we compared aRSGTSVR with SVR, LASSO, Elastic Net, TSVR [38] and Res-TSVR [39]. In the SVR, the parameter ε is optimized in the range $\{ 2 ^ { i } \mid i = - 8 , - 7 , \ldots , 7 , 8 \}$ . For TSVR, we set $C _ { 1 } = C _ { 2 }$ and $\varepsilon _ { 1 } = \varepsilon _ { 2 }$ . For Res-TSVR, the regularization parameters $C _ { 1 } = C _ { 2 }$ and η in the range of $\{ 1 , 3 , \ldots , 1 6 \}$ . For the aRSGTSVR, the hyperparameters a, λ, and τ are optimized through a grid search over the discrete sets {1, 2, 3, 4, 5}, {0.5, 1, 1.5, 2}, and {0.2, 0.5, 0.8}, respectively.

For all of the methods mentioned above, we have chosen the values of C and ε from the following sets: $\{ 2 ^ { i } \mid i = - 8 , - 7 , \ldots , 7 , 8 \}$ . In the nonlinear case, we consider the Gaussian kernel function, i.e. $K ( x _ { 1 } , x _ { 2 } ) = \exp ( - \gamma \| x _ { 1 } - x _ { 2 } \| _ { 2 } ^ { 2 } )$ . The parameters γ of the Gaussian kernel function take values in the range $\{ 2 ^ { i } \mid i = - 8 , - 7 , . . . , 7 , 8 \}$ No training-test partition was performed in this study. All models were trained on the entire dataset, and the reported accuracy values are averaged results from five-fold cross-validation, with each fold acting as the validation set sequentially.

For classification problems, using acc (accuracy) as our evaluation metric. For regression problems, we evaluated the performance of the model using RMSE. MAE was used during the simulations to select the optimal parameters. Their specific definitions

![](images/262c98b8f67e2527c858ee56d295a3d2075d0ea979b96e5a45de1d14db08fc29.jpg)  
(a) 1-SVM

![](images/2cee996f415a78dc40b15c323f715b1c7ef1a3185c562ca2b3c21d0a2d4c8cf9.jpg)  
(b) TPMSVM

![](images/2b592ffdf1e4b6bb768b8229e30153e95c717bb1c89abd9266ee180d239e5791.jpg)  
(c) Pin-TSVM

![](images/1004f001fc3f837fbe536c486add4c2decd6d522bca8e031ca6feb3371a6169e.jpg)  
(d) rhingeSVM

![](images/455828448093ff3c1202cbf80b47faef72da3c4039f79888a9964948190f0df2.jpg)  
(e) RoBoSS-SVM

![](images/d84d0ebd62fd49aa368151d4329aaa1c5013dac77c0ad4b54af5357632a8a2d8.jpg)  
(f) aRSGTSVM

Figure 2: The red solid lines represent the separating hyperplanes obtained by 1-SVM, TPMSVM, Pin-TSVM, rhingeSVM, RoBoSS-SVM and aRSGTSVM. The Bayes classifier is shown as a black dotted line.  
Table 1: The accuracy(acc) and standard deviation (sd) with linear kernel. The bold is the best one.
<table><tr><td>label noise</td><td>1-SVM  $\mathbf { a c c } \pm \mathbf { s d }$ </td><td>TPMSVM  $\mathbf { a c c } \pm \mathbf { s d }$ </td><td>Pin-TSVM  $\mathbf { a c c } \pm \mathbf { s d }$ </td><td> $\mathbf { r h i n g e S V M }$   $\mathbf { a c c } \pm \mathbf { s d }$ </td><td>RoBoSS-SVM  $\mathbf { a c c } \pm \mathbf { s d }$ </td><td>aRSGTSVM  $\mathbf { a c c } \pm \mathbf { s d }$ </td></tr><tr><td>no noise</td><td> $0 . 9 5 0 \pm 0 . 0 3 5$ </td><td> $0 . 9 5 0 \pm 0 . 0 5 5$ </td><td> $0 . 9 5 0 \pm 0 . 0 3 5$ </td><td> $0 . 9 6 0 \pm 0 . 0 5 5$ </td><td> $0 . 9 5 0 \pm 0 . 0 3 5$ </td><td> $\mathbf { 0 . 9 6 0 \pm 0 . 0 2 2 }$ </td></tr><tr><td>5%</td><td> $0 . 9 1 0 \pm 0 . 0 8 2$ </td><td> $0 . 9 1 0 \pm 0 . 0 0 4$ </td><td> $0 . 9 2 0 \pm 0 . 0 5 7$ </td><td> $0 . 9 1 0 \pm 0 . 0 4 2$ </td><td> $0 . 9 1 0 \pm 0 . 0 6 5$ </td><td> $\mathbf { 0 . 9 2 0 \pm 0 . 0 2 7 }$ </td></tr><tr><td>10%</td><td> $0 . 8 5 0 \pm 0 . 0 8 7$ </td><td> $0 . 8 5 0 \pm 0 . 0 0 4$ </td><td> $0 . 8 5 0 \pm 0 . 0 6 1$ </td><td> $0 . 8 5 0 \pm 0 . 0 6 1$ </td><td> $0 . 8 5 0 \pm 0 . 0 9 4$ </td><td> $\mathbf { 0 . 8 7 0 \pm 0 . 0 5 7 }$ </td></tr><tr><td>20%</td><td> $0 . 7 8 0 \pm 0 . 1 0 4$ </td><td> $0 . 7 7 0 \pm 0 . 0 0 4$ </td><td> $0 . 7 7 0 \pm 0 . 1 3 0$ </td><td> $0 . 7 7 0 \pm 0 . 1 6 8$ </td><td> $0 . 7 7 0 \pm 0 . 0 7 6$ </td><td> $\mathbf { 0 . 7 9 0 \pm 0 . 0 6 5 }$ </td></tr><tr><td>35%</td><td> $0 . 6 2 0 \pm 0 . 0 7 6$ </td><td> $0 . 6 3 0 \pm 0 . 0 0 4$ </td><td> $0 . 6 3 0 \pm 0 . 0 9 1$ </td><td> $0 . 6 3 0 \pm 0 . 0 7 6$ </td><td> $0 . 6 3 0 \pm 0 . 0 6 5$ </td><td> $\mathbf { 0 . 6 6 0 \pm 0 . 0 7 4 }$ </td></tr></table>

are as follows:

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( y _ { i } - \hat { y _ { i } } ) ^ { 2 } } , \quad \mathrm { M A E } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | y _ { i } - \hat { y _ { i } } | ,\tag{25}
$$

where n represents the number of samples used for testing.

## 4.1. Artificial datasetsfor classification

Example 1. To evidence the robustness of aRSGTSVM to label noise, we conducted an experiment using a two-dimensional synthetic dataset containing 100 samples. In this example, the samples were generated from two Gaussian distributions with equal probability: $x _ { i } , i \in \{ 1 , 2 , \ldots , 5 0 \} \ \sim \ N ( u _ { 1 } , \Sigma _ { 1 } ) , x _ { i } , i \ \in \ \{ 5 1 , 5 2 , \ldots , 1 0 0 \} \ \sim$ $N ( u _ { 2 } , \Sigma _ { 2 } )$ , where $u _ { 1 } = [ 2 , 5 ] ^ { T } , u _ { 2 } = [ 4 , 2 ] ^ { T }$ and $\Sigma _ { 1 } = \Sigma _ { 2 } = \mathrm { d i a g } ( 1 , 2 )$ . In this case, the

![](images/a4d6e94580f83d3b2703b994b795a91cf25eae2bfd9f6777ec50cc9eb56b8df4.jpg)  
(a) 1-SVM

![](images/fb6d96cf8cdf70202c063c2b2fa3d596a70fdc6357c21efc7102627bfedb5d54.jpg)  
(b) TPMSVM

![](images/78c7db2d03ce16597f3bfd95f042fff0f35c375b162852b2944f59f134317950.jpg)  
(c) Pin-TSVM

![](images/2f367558d5172499a88a69523204625bdcd20b31b42b62d8e2be37f23c212eb5.jpg)  
(d) rhingeSVM

![](images/f283986c915ac19f889bea4462c904f041921847bda7efc7f2ca9657c002ce8b.jpg)  
(e) RoBoSS-SVM

![](images/6feffdb29aa9addcf28aa126b6c51da3eb7a3b3df57f17b844964b25c4dcce16.jpg)  
(f) aRSGTSVM  
Figure 3: The black solid lines represent the separating hyperplanes obtained by 1-SVM, TPMSVM, Pin-TSVM, rhingeSVM, RoBoSS-SVM and aRSGTSVM.

Bayes classifier is $f _ { C } ( x ) = 4 x _ { 1 } - 3 x _ { 2 } - \frac { 3 } { 2 }$ . To systematically investigate the impact of label noise on model performance, experiments were conducted under varying noise ratios.

In Figure 2 presents a noise-free condition, in which the black dotted line represents the Bayesian optimal decision boundary and the red solid line represents the decision boundary obtained from each model. Among the six models evaluated, our proposed aRSGTSVM demonstrates the most satisfactory performance, with its decision boundary showing the closest alignment to that of the Bayesian classifier. The complete experimental results are presented in Table 1, which demonstrate the remarkable robustness and noise insensitivity of our proposed aRSGTSVM.

Example 2. To exemplify the stability of SVM under resampling, a total of 160 samples were generated based on Example 1, with the parameters set to $u _ { 1 } = [ 0 . 8 , - 0 . 5 ] ^ { T }$ $u _ { 2 } = [ - 0 . 8 , 0 . 5 ] ^ { T }$ , and $\Sigma _ { 1 } = \Sigma _ { 2 } = \mathrm { d i a g } ( 0 . 2 , 0 . 2 )$ . The experiment was repeated 30 times.

As visualized in Figure 3, the proposed aRSGTSVM demonstrates exceptional robustness and stability in constructing decision hyperplanes. Specifically, the hyperplanes of aRSGTSVM form a remarkably narrow and highly concentrated cluster, with almost no variation across diferent resampling runs, highlighting its strong resistance to sample perturbations. While RoBoSS-SVM exhibits improved stability relative to traditional SVM variants, its hyperplanes still show non-negligible dispersion. In sharp contrast, the baseline models, including 1-SVM, TPMSVM, Pin-TSVM and rhingeSVM, generate widely scattered hyperplanes with substantial fluctuations, particularly in the boundary regions between classes. This dispersion reveals their sensitivity to minor changes in the training data, resulting in unstable decision boundaries. Beyond its robustness advantage, aRSGTSVM also achieves the most accurate separation between the two classes (red and blue points), efectively distinguishing between the samples and delivering the best overall classification performance among all evaluated models.

Table 2: The accuracy (acc) and standard deviation (sd) in diferent $( n , p ) .$ . The bold is the best one.
<table><tr><td colspan="7">(a) 0% label noise</td></tr><tr><td></td><td>1-SVM</td><td>TPMSVM</td><td> $\mathbf { P i n - T S V M }$ </td><td>rhingeSVM</td><td>RoBoSS-SVM</td><td>aRSGTSVM</td></tr><tr><td> $n = 5 0 , p = 1 0 0$ </td><td> $0 . 9 0 0 \pm 0 . 0 7 1$ </td><td> $0 . 9 2 0 \pm 0 . 1 3 0$ </td><td> $\mathbf { 0 . 9 4 0 \pm 0 . 0 5 5 }$ </td><td> $0 . 9 0 0 \pm 0 . 0 7 1$ </td><td> $0 . 9 2 0 \pm 0 . 1 1 0$ </td><td> $0 . 9 4 0 \pm 0 . 0 8 9$ </td></tr><tr><td> $n = 5 0 , p = 1 5 0$ </td><td> $0 . 9 4 0 \pm 0 . 0 5 5$ </td><td> $0 . 9 2 0 \pm 0 . 0 8 4$ </td><td> $0 . 9 4 0 \pm 0 . 0 5 5$ </td><td> $0 . 9 2 0 \pm 0 . 0 8 4$ </td><td> $0 . 9 2 0 \pm 0 . 0 8 4$ </td><td> $\mathbf { 0 . 9 6 0 \pm 0 . 0 5 5 }$ </td></tr><tr><td> $n = 1 0 0 , p = 1 5 0$ </td><td> $0 . 9 2 0 \pm 0 . 0 2 7$ </td><td> $0 . 9 3 0 \pm 0 . 0 2 7$ </td><td> $0 . 9 3 0 \pm 0 . 0 7 6$ </td><td> $0 . 9 2 0 \pm 0 . 0 6 7$ </td><td> $0 . 9 3 0 \pm 0 . 0 5 7$ </td><td> $\mathbf { 0 . 9 7 0 \pm 0 . 0 4 5 }$ </td></tr><tr><td> $n = 1 0 0 , p = 2 0 0$ </td><td> $0 . 9 2 0 \pm 0 . 0 7 6$ </td><td> $0 . 9 1 0 \pm 0 . 0 5 5$ </td><td> $0 . 9 2 0 \pm 0 . 0 8 4$ </td><td> $0 . 9 1 0 \pm 0 . 0 7 4$ </td><td> $0 . 9 3 0 \pm 0 . 0 4 5$ </td><td> $\mathbf { 0 . 9 6 0 \pm 0 . 0 4 2 }$ </td></tr><tr><td colspan="7">(b) 5% label noise</td></tr><tr><td> $n = 5 0 , p = 1 0 0$ </td><td> $0 . 8 6 0 \pm 0 . 1 1 4$ </td><td> $0 . 8 8 0 \pm 0 . 0 8 4$ </td><td> $0 . 9 0 0 \pm 0 . 1 2 2$ </td><td> $0 . 8 6 0 \pm 0 . 1 3 4$ </td><td> $0 . 8 8 0 \pm 0 . 0 4 5$ </td><td> $\mathbf { 0 . 9 2 0 \pm 0 . 0 8 4 }$ </td></tr><tr><td> $n = 5 0 , p = 1 5 0$ </td><td> $0 . 9 4 0 \pm 0 . 0 5 5$ </td><td> $0 . 9 0 0 \pm 0 . 0 7 1$ </td><td> $0 . 9 4 0 \pm 0 . 0 5 5$ </td><td> $0 . 9 2 0 \pm 0 . 0 4 5$ </td><td> $0 . 9 4 0 \pm 0 . 0 8 9$ </td><td> $\mathbf { 0 . 9 6 0 \pm 0 . 0 5 5 }$ </td></tr><tr><td> $n = 1 0 0 , p = 1 5 0$ </td><td> $0 . 9 0 0 \pm 0 . 0 3 5$ </td><td> $0 . 8 8 0 \pm 0 . 0 5 7$ </td><td> $0 . 8 9 0 \pm 0 . 1 0 8$ </td><td> $0 . 8 8 0 \pm 0 . 0 8 4$ </td><td> $0 . 9 0 0 \pm 0 . 1 0 0$ </td><td> $\mathbf { 0 . 9 4 0 \pm 0 . 0 4 2 }$ </td></tr><tr><td> $n = 1 0 0 , p = 2 0 0$ </td><td> $0 . 9 3 0 \pm 0 . 0 8 4$ </td><td> $0 . 9 2 0 \pm 0 . 0 7 6$ </td><td> $0 . 9 2 0 \pm 0 . 0 5 7$ </td><td> $0 . 9 1 0 \pm 0 . 0 4 2$ </td><td> $0 . 9 3 0 \pm 0 . 0 4 5$ </td><td> $\mathbf { 0 . 9 5 0 \pm 0 . 0 3 5 }$ </td></tr><tr><td colspan="7"></td></tr><tr><td> $n = 5 0 , p = 1 0 0$ </td><td> $0 . 8 4 0 \pm 0 . 1 3 4$ </td><td> $0 . 8 4 0 \pm 0 . 1 5 2$ </td><td>(c) 10% label noise  $0 . 8 6 0 \pm 0 . 1 1 4$ </td><td> $0 . 8 4 0 \pm 0 . 1 1 4$ </td><td> $0 . 8 6 0 \pm 0 . 0 5 5$ </td><td></td></tr><tr><td> $n = 5 0 , p = 1 5 0$ </td><td> $0 . 9 0 0 \pm 0 . 1 0 0$ </td><td> $0 . 8 8 0 \pm 0 . 0 8 4$ </td><td> $0 . 8 8 0 \pm 0 . 0 8 4$ </td><td> $0 . 8 8 0 \pm 0 . 1 6 4$ </td><td> $0 . 9 0 0 \pm 0 . 0 7 1$ </td><td> $\mathbf { 0 . 8 8 0 \pm 0 . 1 3 0 }$   $\mathbf { 0 . 9 2 0 \pm 0 . 0 8 4 }$ </td></tr><tr><td> $n = 1 0 0 , p = 1 5 0$ </td><td> $0 . 8 3 0 \pm 0 . 1 0 4$ </td><td> $0 . 8 4 0 \pm 0 . 1 1 4$ </td><td> $0 . 8 3 0 \pm 0 . 0 7 6$ </td><td> $0 . 8 1 0 \pm 0 . 1 0 8$ </td><td> $0 . 8 5 0 \pm 0 . 0 8 7$ </td><td> $\mathbf { 0 . 8 7 0 \pm 0 . 0 9 7 }$ </td></tr><tr><td> $n = 1 0 0 , p = 2 0 0$ </td><td> $0 . 8 3 0 \pm 0 . 0 9 7$ </td><td> $0 . 8 0 0 \pm 0 . 0 7 9$ </td><td> $0 . 8 1 0 \pm 0 . 0 8 2$ </td><td> $0 . 8 1 0 \pm 0 . 1 2 4$ </td><td> $0 . 8 3 0 \pm 0 . 1 4 4$ </td><td> $\mathbf { 0 . 8 7 0 \pm 0 . 0 2 7 }$ </td></tr><tr><td colspan="7">(d) 20% label noise</td></tr><tr><td> $n = 5 0 , p = 1 0 0$ </td><td> $0 . 7 2 0 \pm 0 . 2 2 8$ </td><td> $0 . 7 0 0 \pm 0 . 1 2 2$ </td><td> $0 . 7 4 0 \pm 0 . 2 0 7$ </td><td> $0 . 7 2 0 \pm 0 . 1 3 0$ </td><td> $0 . 7 2 0 \pm 0 . 1 7 9$ </td><td> $\mathbf { 0 . 7 8 0 \pm 0 . 1 1 0 }$ </td></tr><tr><td> $n = 5 0 , p = 1 5 0$ </td><td> $0 . 8 0 0 \pm 0 . 0 7 1$ </td><td> $0 . 8 4 0 \pm 0 . 0 8 9$ </td><td> $0 . 7 8 0 \pm 0 . 1 1 0$ </td><td> $0 . 7 8 0 \pm 0 . 1 1 0$ </td><td> $0 . 7 8 0 \pm 0 . 0 8 4$ </td><td> ${ \bf 0 . 8 4 0 \pm 0 . 1 9 5 }$ </td></tr><tr><td> $n = 1 0 0 , p = 1 5 0$ </td><td> $0 . 7 7 0 \pm 0 . 1 3 0$ </td><td> $0 . 7 7 0 \pm 0 . 0 5 7$ </td><td> $0 . 7 5 0 \pm 0 . 1 1 2$ </td><td> $0 . 7 7 0 \pm 0 . 1 1 5$ </td><td> $0 . 7 7 0 \pm 0 . 0 8 4$ </td><td> $\mathbf { 0 . 8 0 0 \pm 0 . 0 8 7 }$ </td></tr><tr><td> $n = 1 0 0 , p = 2 0 0$ </td><td> $0 . 7 7 0 \pm 0 . 1 1 5$ </td><td> $0 . 7 6 0 \pm 0 . 1 3 4$ </td><td> $0 . 7 6 0 \pm 0 . 0 5 5$ </td><td> $0 . 7 4 0 \pm 0 . 0 8 9$ </td><td> $0 . 7 7 0 \pm 0 . 1 0 4$ </td><td> $\mathbf { 0 . 7 9 0 \pm 0 . 0 8 9 }$ </td></tr><tr><td colspan="7">(e) 35% label noise</td></tr><tr><td> $n = 5 0 , p = 1 0 0$ </td><td> $0 . 6 0 0 \pm 0 . 1 0 0$ </td><td> $0 . 6 0 0 \pm 0 . 1 4 1$ </td><td> $0 . 6 2 0 \pm 0 . 0 8 4$ </td><td> $0 . 5 6 0 \pm 0 . 1 8 2$ </td><td> $0 . 6 6 0 \pm 0 . 1 9 5$ </td><td> $\mathbf { 0 . 6 8 0 \pm 0 . 1 3 0 }$ </td></tr><tr><td> $n = 5 0 , p = 1 5 0$ </td><td> $0 . 7 0 0 \pm 0 . 0 7 1$ </td><td> $0 . 6 6 0 \pm 0 . 1 6 7$ </td><td> $0 . 7 4 0 \pm 0 . 0 8 9$ </td><td> $0 . 6 6 0 \pm 0 . 1 5 2$ </td><td> $0 . 7 2 0 \pm 0 . 1 4 8$ </td><td> $\mathbf { 0 . 7 8 0 \pm 0 . 2 0 5 }$ </td></tr><tr><td> $n = 1 0 0 , p = 1 5 0$ </td><td> $0 . 6 2 0 \pm 0 . 0 5 7$ </td><td> $0 . 6 1 0 \pm 0 . 1 0 8$ </td><td> $0 . 6 3 0 \pm 0 . 0 9 7$ </td><td> $0 . 6 0 0 \pm 0 . 1 2 7$ </td><td> $0 . 6 4 0 \pm 0 . 0 8 9$ </td><td> $\mathbf { 0 . 6 6 0 \pm 0 . 0 8 2 }$ </td></tr><tr><td> $n = 1 0 0 , p = 2 0 0$ </td><td> $0 . 6 2 0 \pm 0 . 1 1 5$ </td><td> $0 . 6 0 0 \pm 0 . 0 3 5$ </td><td> $0 . 6 4 0 \pm 0 . 0 5 5$ </td><td> $0 . 5 9 0 \pm 0 . 1 5 2$ </td><td> $0 . 6 2 0 \pm 0 . 1 0 4$ </td><td> $\mathbf { 0 . 6 7 0 \pm 0 . 0 6 7 }$ </td></tr></table>

Example 3. In this example, the samples were generated from two multivariate Gaussian distributions with equal probability: $x _ { i } , i \in \{ 1 , 2 , . . . , \frac { n } { 2 } \} \sim N ( u _ { 1 } , \Sigma _ { 1 } ) , x _ { i } , i$ ∈ $\{ \frac { n } { 2 } + 1 , \frac { n } { 2 } + 2 , \ldots , n \} \sim N ( u _ { 2 } , \Sigma _ { 2 } )$ , where $u _ { 1 } = ( 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 , 0 , . . . , 0 ) ^ { T } \in \mathbb { R } ^ { n }$ $u _ { 2 } ~ = ~ - u _ { 1 }$ and $\Sigma _ { 1 } = \Sigma _ { 2 } = ( \sigma _ { i j } )$ is defined with non-zero elements: $\sigma _ { i i } = 1$ , for $i \ =$

![](images/0c064ac2437d5df950aaf7a41ab0be682698e2a87fd3930dd1beed287e027379.jpg)  
Figure 4: The results of the ablation experiments under diferent ratios of label noise and number of features.

## $1 , 2 , \ldots , n$ and $\sigma _ { i j } = - 0 . 2 ,$ , for $1 \leq i \neq j \leq 5 .$

As shown in Table 2, we systematically compared the classification performance of the proposed aRSGTSVM with several state-of-the-art SVM-based methods across diferent combinations of sample size n, feature dimension p, and varying label noise levels ranging from 0% to 35%. The results, measured by mean classification accuracy and standard deviation, consistently demonstrate that aRSGTSVM achieves the highest or tied-for-highest accuracy in nearly all experimental settings, including highdimensional scenarios where $p > n ,$ while maintaining stable performance across repeated trials. As label noise increases, aRSGTSVM consistently outperforms competing methods, showing its superiority in both clean high-dimensional environments and noisy conditions. This comprehensive comparison confirms that the proposed aR-SGTSVM exhibits stronger robustness against label noise and better adaptability to high-dimensional data compared with baseline models, validating the efectiveness of its asymmetric regularized twin SVM framework.

Example 4. To further verify the efectiveness of the proposed asymmetric Ro-BoSS loss function, we conduct ablation experiments. Based on the setting of Example 3, we fix the sample size at $n = 2 0 0 .$ , set the feature dimension $p \in \{ 1 0 0 , 2 0 0 , 3 0 0 \}$ and the label noise ratio to {0%, 10%, 20%, 30%, 40%}. The experimental results are presented in Figure 4.

![](images/8ca7ce6512325a2d70fe103d3ab46c7d0b2d22a8727dd0a20a35999ef30da4e1.jpg)  
Figure 5: The one-run CPU time cost of 1-SVM, TPMSVM, Pin-TSVM, rhingeSVM, RoBoSS-SVM and aRSGTSVM under diferent sample sizes.

As shown in Figure 4, under diferent feature dimensions $( p ~ = ~ 1 0 0 , 2 0 0 , 3 0 0 )$ and varying label noise ratios (0% to 40%), the proposed aRSGTSVM consistently achieves higher prediction accuracy than the ablation method SGTSVM, which removes the asymmetric robust loss. As the noise ratio increases from 0% to 40%, the prediction accuracy of both methods decreases. However, aRSGTSVM exhibits a much smaller decline, demonstrating superior robustness to label noise. Moreover, as the feature dimension increases from 100 to 300, aRSGTSVM maintains stable performance, whereas SGTSVM shows a more pronounced drop in accuracy. These results validate the efectiveness and robustness of the proposed asymmetric RoBoSS loss function in scenarios with label noise and high-dimensional features.

Example 5. In this example, we investigate the CPU time cost of our proposed algorithm (Algorithm 1) with varying numbers of samples and features, respectively. Based on the setting of Example 1, we fix the feature dimension at $p = 2$ , set the sample size n ranging from 50 to 10000 with a label noise ratio of 15%, and set all parameters to 1 except for $\tau = 0 . 5 .$ . The results are presented in Figure 5. Based on the setting of Example 3, We further conduct experiments with a fixed sample size $n = 1 0 0$ and feature dimension p ranging from 50 to 5000, under the same noise ratio and parameter

![](images/78927ef0dcc9c88c0e74faf88471a9d324b5acb81ac4693e924dea146c8ab250.jpg)  
Figure 6: The one-run CPU time cost of 1-SVM, TPMSVM, Pin-TSVM, rhingeSVM, RoBoSS-SVM and aRSGTSVM under diferent feature numbers.

settings, as shown in Figure 6.

Figure 5 presents the CPU running time of various classifiers under diferent sample sizes. As the sample size increases from 50 to 10000, the running time of most compared algorithms increases significantly. Among them, rhingeSVM and Pin-TSVM show the fastest growth in time complexity, with their time consumption far exceeding other models in large-sample scenarios. In contrast, the proposed aRSGTSVM maintains an extremely low running time throughout. Even when the sample size reaches 10000, its time consumption remains at a low level, only slightly higher than 1-SVM and much lower than other compared models, demonstrating excellent scalability to large sample sizes.

Figure 6 illustrates the single-run CPU time (log scale) of diferent algorithms as the feature dimension increases from 50 to 5,000, with a fixed sample size of n = 100. For the proposed aRSGTSVM, the running time grows slowly when the number of features is below 200, but increases sharply thereafter, becoming the highest among all compared methods at 5,000 dimensions. In contrast, Pin-TSVM and TPMSVM maintain consistently low running times, while 1-SVM and RoBoSS-SVM exhibit an approximately linear increase with a much lower growth rate than aRSGTSVM. In summary, aRSGTSVM is sensitive to feature dimensionality, incurring high computational costs in high-dimensional scenarios, though its running time remains acceptable

![](images/6162b65cb68eab3f495d6a0eb49de71083c74d8d4c2c2d9e9d287630f71e8ef6.jpg)  
(a) No noise

![](images/13b17f745c4cdd76d80a9e77a59b6dee28529b76197e21040eddc65a36b272f3.jpg)  
(b) Uniform noise

![](images/7bcb4b23443a9009b18552279d56fb560ab8c3c191a26a43702bcb3f1b75274e.jpg)  
(c) Gaussian noise  
Figure 7: Predictions of SVR, TSVR, Res-TSVR and aRSGTSVR on Sinc function with diferent noises.

in low-to-medium dimensions.

## 4.2. Artificial datasetsfor regression

Example 1. In this example, we designed a two-dimensional case to demonstrate that aRSGTSVR is robust under diferent types of noise, namely uniform noise and Gaussian noise. We generate 160 samples as follows:

$$
y = \operatorname { s i n c } ( x _ { i } ) = { \frac { \sin x _ { i } } { x _ { i } } } + e _ { i } , \quad x _ { i } \sim U [ - 3 \pi , 3 \pi ] .\tag{26}
$$

Noise is added based on (26), which is defined as follows:

$$
\mathrm { T y p e ~ A } \colon e _ { i } \sim U [ - 0 . 4 , 0 . 4 ] , \mathrm { T y p e ~ B } \colon e _ { i } \sim N ( 0 , 0 . 2 ^ { 2 } ) .
$$

We plotted the fitting curve of each model in Figure 7. Since it is not intuitively obvious which model fits the true curve better, we present the corresponding root mean square error and standard deviation obtained with the nonlinear kernel setting in Table 3. The results consistently show that the proposed aRSGTSVR achieves the best overall performance in terms of RMSE across diferent noise conditions, including noiseless, uniform noise, and Gaussian noise scenarios, while maintaining a relatively low standard deviation. Compared with competing methods such as SVR, LASSO, Elastic Net, TSVR, and Res-TSVR, aRSGTSVR exhibits more stable and superior regression performance, which confirms that our method is more robust than other comparative models under diferent types of noise when handling nonlinear regression tasks.

Table 3: The RMSE and standard deviation (sd) with nonlinear kernel. The bold is the best one.
<table><tr><td></td><td>SVR</td><td>LASSO</td><td>Elastic Net</td><td>TSVR</td><td>Res-TSVR</td><td>aRSGTSVR</td></tr><tr><td>No noise</td><td> $0 . 0 8 0 \pm 0 . 0 1 3$ </td><td> $0 . 0 7 4 \pm 0 . 0 1 2$ </td><td> $0 . 0 7 5 \pm 0 . 0 1 1$ </td><td> $0 . 0 7 5 \pm 0 . 0 3 0$ </td><td> $0 . 1 2 6 \pm 0 . 1 1 5$ </td><td> $\mathbf { 0 . 0 7 1 \pm 0 . 0 1 0 }$ </td></tr><tr><td>Uniform noise</td><td> $0 . 1 2 8 \pm 0 . 0 2 6$ </td><td> $0 . 1 2 9 \pm 0 . 0 1 4$ </td><td> $0 . 1 3 0 \pm 0 . 0 1 5$ </td><td> $0 . 1 2 9 \pm 0 . 0 2 1$ </td><td> $0 . 1 2 8 \pm 0 . 0 2 2$ </td><td> $\mathbf { 0 . 1 1 8 \pm 0 . 0 1 7 }$ </td></tr><tr><td>Gaussian noise</td><td> $0 . 2 6 7 \pm 0 . 0 5 9$ </td><td> $0 . 2 7 0 \pm 0 . 0 6 2$ </td><td> $0 . 2 6 5 \pm 0 . 0 6 0$ </td><td> $0 . 2 6 5 \pm 0 . 0 5 5$ </td><td> $0 . 2 6 4 \pm 0 . 0 5 7$ </td><td> $\mathbf { 0 . 2 6 0 \pm 0 . 0 7 3 }$ </td></tr></table>

Table 4: The RMSE and standard deviation (sd) in diferent $( n , p ) .$ The bold is the best one.
<table><tr><td colspan="7">(a) 0% label noise</td></tr><tr><td></td><td>SVR</td><td>LASSO</td><td>Elastic Net</td><td>TSVR</td><td> $\mathbf { R e s { \mathbf { - } } T S V R }$ </td><td>aRSGTSVR</td></tr><tr><td> $n = 5 0 , p = 3 0$ </td><td> $1 . 5 7 0 \pm 1 . 3 4 2$ </td><td> $1 . 9 6 8 \pm 1 . 9 8 8$ </td><td> $1 . 8 1 9 \pm 1 . 8 3 8$ </td><td> $1 . 9 4 9 \pm 1 . 2 5 4$ </td><td> $1 . 8 6 1 \pm 0 . 8 9 2$ </td><td> $\mathbf { 1 . 4 3 8 \pm 0 . 6 7 5 }$ </td></tr><tr><td> $n = 5 0 , p = 5 0$ </td><td> $1 . 7 1 2 \pm 1 . 9 0 2$ </td><td> $2 . 3 3 1 \pm 2 . 3 5 4$ </td><td> $2 . 3 3 1 \pm 2 . 3 5 4$ </td><td> $2 . 1 0 5 \pm 1 . 9 9 8$ </td><td> $1 . 9 7 6 \pm 0 . 9 0 7$ </td><td> $\mathbf { 1 . 4 8 1 \pm 1 . 0 4 1 }$ </td></tr><tr><td> $n = 5 0 , p = 1 0 0$ </td><td> $1 . 8 1 1 \pm 1 . 9 0 0$ </td><td> $2 . 3 3 1 \pm 2 . 3 5 4$ </td><td> $2 . 3 3 1 \pm 2 . 3 5 4$ </td><td> $2 . 1 4 0 \pm 0 . 8 8 4$ </td><td> $2 . 0 4 1 \pm 2 . 5 3 2$ </td><td> $\mathbf { 1 . 5 0 1 \pm 1 . 1 3 5 }$ </td></tr><tr><td colspan="7">(b) 5% label noise</td></tr><tr><td> $n = 5 0 , p = 3 0$ </td><td> $1 . 5 7 4 \pm 1 . 3 4 1$ </td><td> $1 . 9 5 8 \pm 1 . 9 7 7$ </td><td> $1 . 9 5 8 \pm 1 . 9 7 7$ </td><td> $1 . 9 4 2 \pm 1 . 4 6 9$ </td><td> $1 . 8 6 2 \pm 0 . 7 7 0$ </td><td> $\mathbf { 1 . 4 0 6 \pm 1 . 1 0 1 }$ </td></tr><tr><td> $n = 5 0 , p = 5 0$ </td><td> $1 . 8 0 2 \pm 1 . 6 0 7$ </td><td> $2 . 3 2 5 \pm 2 . 3 4 9$ </td><td> $2 . 3 2 5 \pm 2 . 3 4 9$ </td><td> $2 . 0 9 7 \pm 1 . 6 1 8$ </td><td> $1 . 9 0 4 \pm 1 . 1 4 2$ </td><td> $\mathbf { 1 . 4 6 1 \pm 1 . 5 4 3 }$ </td></tr><tr><td> $n = 5 0 , p = 1 0 0$ </td><td> $1 . 7 7 3 \pm 1 . 4 7 8$ </td><td> $2 . 3 2 5 \pm 2 . 3 4 9$ </td><td> $2 . 3 2 5 \pm 2 . 3 4 9$ </td><td> $2 . 1 5 7 \pm 2 . 2 0 4$ </td><td> $1 . 9 8 6 \pm 1 . 3 5 0$ </td><td> $\mathbf { 1 . 4 8 0 \pm 1 . 4 3 8 }$ </td></tr><tr><td colspan="7"> $( \mathrm { c } ) 1 0 \% 1 \mathrm { a b e l n o i s e }$ </td></tr><tr><td> $n = 5 0 , p = 3 0$ </td><td> $1 . 6 3 2 \pm 0 . 5 7 4$ </td><td> $1 . 9 9 0 \pm 2 . 0 1 0$ </td><td> $1 . 9 9 0 \pm 2 . 0 1 0$ </td><td> $1 . 9 7 8 \pm 1 . 1 5 5$ </td><td> $1 . 8 9 2 \pm 0 . 9 2 2$ </td><td> $\mathbf { 1 . 4 4 4 \pm 0 . 7 1 7 }$ </td></tr><tr><td> $n = 5 0 , p = 5 0$ </td><td> $1 . 8 3 5 \pm 0 . 7 7 4$ </td><td> $2 . 3 3 4 \pm 2 . 3 5 8$ </td><td> $2 . 3 3 4 \pm 2 . 3 5 8$ </td><td> $2 . 1 7 8 \pm 2 . 0 5 5$ </td><td> $2 . 0 9 1 \pm 0 . 7 8 4$ </td><td> $\mathbf { 1 . 4 8 7 \pm 1 . 2 9 0 }$ </td></tr><tr><td> $n = 5 0 , p = 1 0 0$ </td><td> $1 . 8 7 8 \pm 1 . 1 1 5$ </td><td> $2 . 3 3 4 \pm 2 . 3 5 8$ </td><td> $2 . 3 3 4 \pm 2 . 3 5 8$ </td><td> $2 . 1 6 5 \pm 1 . 6 4 5$ </td><td> $2 . 1 2 0 \pm 1 . 3 4 8$ </td><td> $\mathbf { 1 . 4 8 3 \pm 1 . 2 7 9 }$ </td></tr><tr><td colspan="7">(d) 20% label noise</td></tr><tr><td> $n = 5 0 , p = 3 0$ </td><td> $1 . 5 9 0 \pm 0 . 4 7 7$ </td><td> $1 . 9 6 2 \pm 1 . 9 8 2$ </td><td> $1 . 8 2 5 \pm 1 . 8 4 3$ </td><td> $1 . 9 4 6 \pm 1 . 5 9 6$ </td><td> $1 . 7 0 9 \pm 1 . 4 7 7$ </td><td> $\mathbf { 1 . 4 3 5 \pm 0 . 9 9 4 }$ </td></tr><tr><td> $n = 5 0 , p = 5 0$ </td><td> $1 . 8 1 5 \pm 1 . 8 7 6$ </td><td> $2 . 3 3 6 \pm 2 . 3 6 0$ </td><td> $2 . 3 3 6 \pm 2 . 3 6 0$ </td><td> $2 . 1 5 0 \pm 1 . 2 5 7$ </td><td> $1 . 9 4 1 \pm 1 . 9 9 9$ </td><td> $\mathbf { 1 . 4 8 5 \pm 1 . 3 1 5 }$ </td></tr><tr><td> $n = 5 0 , p = 1 0 0$ </td><td> $1 . 8 3 2 \pm 2 . 4 5 9$ </td><td> $2 . 3 3 6 \pm 2 . 3 6 0$ </td><td> $2 . 3 3 6 \pm 2 . 3 6 0$ </td><td> $2 . 1 6 2 \pm 1 . 7 0 2$ </td><td> $1 . 9 2 4 \pm 1 . 4 9 1$ </td><td> $\mathbf { 1 . 5 3 5 \pm 1 . 9 8 4 }$ </td></tr><tr><td colspan="7">(e) 35% label noise</td></tr><tr><td> $n = 5 0 , p = 3 0$ </td><td> $1 . 5 9 0 \pm 0 . 3 6 0$ </td><td> $1 . 9 7 2 \pm 1 . 9 9 2$ </td><td> $1 . 8 2 3 \pm 1 . 8 4 1$ </td><td> $1 . 8 9 6 \pm 1 . 0 4 4$ </td><td> $1 . 8 5 8 \pm 1 . 0 4 1$ </td><td> $\mathbf { 1 . 4 0 0 \pm 0 . 9 5 9 }$ </td></tr><tr><td> $n = 5 0 , p = 5 0$ </td><td> $1 . 7 6 7 \pm 1 . 5 6 6$ </td><td> $2 . 3 5 2 \pm 2 . 3 7 6$ </td><td> $2 . 3 5 2 \pm 2 . 3 7 6$ </td><td> $2 . 1 6 9 \pm 1 . 7 9 8$ </td><td> $1 . 9 3 9 \pm 1 . 7 5 8$ </td><td> $\mathbf { 1 . 5 2 8 \pm 1 . 3 0 2 }$ </td></tr><tr><td> $n = 5 0 , p = 1 0 0$ </td><td> $1 . 7 8 7 \pm 1 . 5 8 3$ </td><td> $2 . 3 5 2 \pm 2 . 3 7 6$ </td><td> $2 . 3 5 2 \pm 2 . 3 7 6$ </td><td> $2 . 1 5 7 \pm 2 . 1 4 4$ </td><td> $2 . 0 0 2 \pm 1 . 2 4 8$ </td><td> $\mathbf { 1 . 5 1 9 \pm 1 . 6 8 2 }$ </td></tr></table>

Example 2. To illustrate that aRSGTSVR exhibits excellent variable selection capability under diferent noise ratios for high-dimensional data, in this example, the samples are generated from the following multivariate Gaussian distribution: $x _ { i } , i \in$ $\{ 1 , 2 , \ldots , n \} \sim N ( u , \Sigma )$ , where $u = ( 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 , 0 , . . . , 0 ) ^ { T } \in \mathbb { R } ^ { n }$ and $\Sigma = ( \sigma _ { i j } )$ is defined with non-zero elements: $\sigma _ { i i } = 1$ , for $i = 1 , 2 , \dots$ , n and $\sigma _ { i j } = - 0 . 2 $ , for $1 \leq$ $i \neq j \leq 5 .$

The response variable y for each sample is constructed via the function:

$$
y = \sin ( 3 x _ { 1 } ) + x _ { 2 } ^ { 2 } + \exp ( - | x _ { 3 } | ) + \log ( | x _ { 4 } | + 1 ) + x _ { 1 } x _ { 5 } .\tag{27}
$$

Table 4 presents the RMSE and standard deviation results of all compared methods across diferent combinations of sample size n, feature dimension $p ,$ and label noise levels ranging from 0% to 35%, while Figure 8 further visualizes the performance trends of each method under varying noise conditions for diferent dimensional settings. As shown in both the table and figure, the proposed aRSGTSVR consistently achieves the lowest RMSE values across all experimental scenarios, including low-dimensional, boundary high-dimensional, and classical high-dimensional cases, while maintaining stable performance with relatively small standard deviations. Compared with baseline models such as SVR, LASSO, Elastic Net, TSVR, and Res-TSVR, aRSGTSVR exhibits superior regression accuracy and stronger robustness against increasing label noise, confirming its combined advantages in prediction accuracy and stability for both low- and high-dimensional nonlinear regression tasks.

![](images/a82e7cbd929e8fbf7711d7a571d96e3202955f033f381720144caeba1a4765cf.jpg)  
(a) n = 50, p = 30

![](images/cbe5d237192ee4a376e2d6abd3c4d7cc902a9313ce874cbcceb8fb5444c9319e.jpg)  
(b) $n = 5 0 , p = 5 0$

![](images/a2c6cf76c8a893d329de68b4c12d11210d69fa0f47cf25d68b0a60d9cd397578.jpg)  
(c) n = 50, p = 100  
Figure 8: Predictions of SVR, LASSO, Elastic Net, TSVR, Res-TSVR and aRSGTSVR on Sinc function with diferent types of noises.

## 4.3. UCI datasets for classification

## 4.3.1. Experimental results

To assess the generalization performance of the models, we conducted tests on a lot of UCI datasets (see Table 5). Furthermore, to examine the robustness of the SVMs to outliers, we introduced artificial noise by randomly selecting 15% and 35% of the training samples and swapping their labels.

Table 6 presents the performance comparison of various SVM models using the initial data. In terms of average prediction accuracy, our proposed aRSGTSVM achieves the most optimal results, followed by RoBoSS-SVM, while rhingeSVM also demonstrates strong competitiveness. Although 1-SVM performs well only on certain datasets, all models can efectively predict data labels. TPMSVM and Pin-TSVM show the relatively poorest performance. Overall, the vast majority of models are capable of efectively classifying the data.

Table 5: The information of seleced UCI datasets.
<table><tr><td>Dataset</td><td>#Samples</td><td>#Features</td></tr><tr><td>acoustic</td><td>400</td><td>50</td></tr><tr><td>amphibians</td><td>189</td><td>21</td></tr><tr><td>autism</td><td>702</td><td>15</td></tr><tr><td>bads</td><td>1372</td><td>4</td></tr><tr><td>dccc</td><td>30000</td><td>23</td></tr><tr><td>diabetes</td><td>520</td><td>16</td></tr><tr><td>gait</td><td>47</td><td>321</td></tr><tr><td>garments</td><td>1197</td><td>7</td></tr><tr><td>heartfailure</td><td>299</td><td>12</td></tr><tr><td>lsvt</td><td>126</td><td>310</td></tr><tr><td>messidor</td><td>1151</td><td>19</td></tr><tr><td>raisin</td><td>900</td><td>7</td></tr><tr><td>shillbidding</td><td>6321</td><td>9</td></tr><tr><td>tripadvisor</td><td>980</td><td>9</td></tr><tr><td>vertebral</td><td>310</td><td>6</td></tr><tr><td>waveform</td><td>5000</td><td>40</td></tr><tr><td>wp</td><td>4746</td><td>15</td></tr></table>

As shown in Table 7, the aRSGTSVM model still achieves the optimal performance. In comparison, the competitive advantages of rhingeSVM, RoBoSS-SVM, and 1-SVM models gradually diminish, while TPMSVM shows an improving trend. Pin-TSVM consistently demonstrates relatively poor performance. From Table 8, it can be observed that aRSGTSVM performs best on all UCI datasets, and as the noise ratio increases, the superiority of aRSGTSVM becomes more pronounced, indicating that our model exhibits strong robustness to label noise.

## 4.3.2. Comparisons by statistical test

To further validate the efectiveness of our proposed aRSGTSVM model, we compare it with other SVM models on multiple datasets using the Friedman test and the corresponding Nemenyi post-hoc test[40]. The null hypothesis of the Friedman test is that there is no significant diference among all models. If the null hypothesis is rejected, the Nemenyi test is conducted. The Friedman statistic is defined as:

$$
F _ { F } = \frac { ( D _ { n } - 1 ) \chi _ { F } ^ { 2 } } { D _ { n } ( C _ { k } - 1 ) - \chi _ { F } ^ { 2 } } ,
$$

Table 6: The results of UCI datasets with 0% label noise. The bold is the best one.
<table><tr><td></td><td>1-SVM</td><td>TPMSVM</td><td>Pin-TSVM</td><td>rhingeSVM</td><td>RoBoSS-SVM</td><td>aRSGTSVM</td></tr><tr><td>acoustic</td><td> $0 . 7 5 5 { \scriptstyle \pm 0 . 0 8 4 }$ </td><td> $0 . 8 2 5 { \scriptstyle \pm 0 . 0 6 2 }$ </td><td> $0 . 7 6 7 { \scriptstyle \pm 0 . 0 5 9 }$ </td><td> $0 . 8 6 8 { \scriptstyle \pm 0 . 0 4 4 }$ </td><td> $0 . 8 5 3 { \scriptstyle \pm 0 . 0 4 6 }$ </td><td> $\mathbf { 0 . 8 7 3 { \scriptstyle \pm 0 . 0 2 2 } }$ </td></tr><tr><td>amphibians</td><td> $0 . 7 0 4 { \scriptstyle \pm 0 . 0 8 0 }$ </td><td> $0 . 7 4 6 { \scriptstyle \pm 0 . 0 9 0 }$ </td><td> $0 . 7 1 4 { \pm } 0 . 1 1 5$ </td><td> $0 . 7 9 3 { \scriptstyle \pm 0 . 0 6 2 }$ </td><td> $0 . 8 0 9 { \scriptstyle \pm 0 . 0 7 5 }$ </td><td> $\mathbf { 0 . 8 1 4 2 0 . 0 8 3 }$ </td></tr><tr><td>autism</td><td> $\mathbf { 1 . 0 0 0 { \pm } 0 . 0 0 0 }$ </td><td> $0 . 9 6 7 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $0 . 9 7 0 { \scriptstyle \pm 0 . 0 3 8 }$ </td><td> $\mathbf { 1 . 0 0 0 { \pm } 0 . 0 0 0 }$ </td><td> $\mathbf { 1 . 0 0 0 { \pm } 0 . 0 0 0 }$ </td><td> $\mathbf { 1 . 0 0 0 { \div } 0 . 0 0 0 }$ </td></tr><tr><td>bads</td><td> $0 . 9 8 1 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td> $0 . 9 8 5 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $0 . 9 6 1 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $0 . 9 8 7 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 9 8 4 { \pm } 0 . 0 1 3$ </td><td> $\mathbf { 0 . 9 8 8 { \pm } 0 . 0 1 3 }$ </td></tr><tr><td>dccc</td><td> $0 . 7 5 3 { \scriptstyle \pm 0 . 0 6 4 }$ </td><td> $0 . 8 2 1 { \scriptstyle \pm 0 . 0 5 3 }$ </td><td> $0 . 8 1 2 { \scriptstyle \pm 0 . 0 4 9 }$ </td><td> $0 . 8 0 3 { \scriptstyle \pm 0 . 0 5 4 }$ </td><td> $0 . 8 3 2 { \scriptstyle \pm 0 . 0 5 9 }$ </td><td> $\mathbf { 0 . 8 3 6 { \scriptstyle \pm 0 . 0 3 1 } }$ </td></tr><tr><td>diabetes</td><td> $0 . 9 1 7 { \scriptstyle \pm 0 . 0 3 6 }$ </td><td> $0 . 9 0 4 { \scriptstyle \pm 0 . 0 3 7 }$ </td><td> $0 . 8 9 2 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $0 . 9 3 7 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td> $\mathbf { 0 . 9 4 0 { \pm } 0 . 0 3 9 }$ </td><td> $0 . 9 1 0 { \scriptstyle \pm 0 . 0 2 6 }$ </td></tr><tr><td>gait</td><td> $0 . 9 8 0 { \scriptstyle \pm 0 . 0 6 3 }$ </td><td> $0 . 9 1 5 { \scriptstyle \pm 0 . 1 4 5 }$ </td><td> $0 . 9 1 5 { \scriptstyle \pm 0 . 1 4 5 }$ </td><td> $0 . 9 8 0 { \scriptstyle \pm 0 . 0 6 3 }$ </td><td> $0 . 9 4 0 { \scriptstyle \pm 0 . 1 3 5 }$ </td><td> $\mathbf { 1 . 0 0 0 { \pm } 0 . 0 0 0 }$ </td></tr><tr><td>garments</td><td> $0 . 9 9 9 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 9 9 8 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 9 9 3 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $\mathbf { 1 . 0 0 0 { \pm } 0 . 0 0 0 }$ </td><td> $\mathbf { 1 . 0 0 0 { \pm } 0 . 0 0 0 }$ </td><td> $0 . 9 9 9 { \scriptstyle \pm 0 . 0 0 3 }$ </td></tr><tr><td>heartfailure</td><td> $0 . 7 6 2 { \scriptstyle \pm 0 . 0 7 1 }$ </td><td> $0 . 8 3 6 { \scriptstyle \pm 0 . 0 5 8 }$ </td><td> $0 . 8 4 3 { \scriptstyle \pm 0 . 0 5 2 }$ </td><td> $0 . 8 3 3 { \scriptstyle \pm 0 . 0 7 9 }$ </td><td> $0 . 8 3 6 { \scriptstyle \pm 0 . 0 6 0 }$ </td><td> $\mathbf { 0 . 8 5 9 { \scriptstyle \pm 0 . 0 6 3 } }$ </td></tr><tr><td>lsvt</td><td> $\mathbf { 0 . 9 8 5 { \pm } 0 . 0 3 2 }$ </td><td> $0 . 7 6 0 { \scriptstyle \pm 0 . 1 2 3 }$ </td><td> $0 . 8 2 4 { \scriptstyle \pm 0 . 0 9 4 }$ </td><td> $0 . 9 5 3 { \scriptstyle \pm 0 . 0 7 5 }$ </td><td> $0 . 9 0 4 { \scriptstyle \pm 0 . 1 2 3 }$ </td><td> $0 . 9 1 2 { \scriptstyle \pm 0 . 0 9 0 }$ </td></tr><tr><td>messidor</td><td> $0 . 6 0 0 { \scriptstyle \pm 0 . 0 5 1 }$ </td><td> $0 . 6 9 9 { \scriptstyle \pm 0 . 0 4 2 }$ </td><td> $0 . 6 0 1 { \scriptstyle \pm 0 . 0 7 1 }$ </td><td> $\mathbf { 0 . 7 5 1 { \pm } 0 . 0 3 6 }$ </td><td> $0 . 6 7 3 { \scriptstyle \pm 0 . 0 6 0 }$ </td><td> $\mathbf { 0 . 7 5 1 { \pm } 0 . 0 3 4 }$ </td></tr><tr><td>raisin</td><td> $0 . 7 8 8 { \pm } 0 . 0 3 4$ </td><td> $0 . 8 6 5 { \scriptstyle \pm 0 . 0 2 8 }$ </td><td> $0 . 8 6 2 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $0 . 8 7 1 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $\mathbf { 0 . 8 7 5 { \scriptstyle \pm 0 . 0 2 7 } }$ </td><td> $0 . 8 6 9 { \scriptstyle \pm 0 . 0 2 7 }$ </td></tr><tr><td>shillbidding</td><td> $0 . 9 8 2 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $0 . 9 8 4 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 9 7 9 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $0 . 9 8 4 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 9 8 2 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $\mathbf { 0 . 9 8 8 { \pm } 0 . 0 0 9 }$ </td></tr><tr><td>tripadvisor</td><td> $0 . 6 4 4 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $0 . 7 4 7 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $0 . 7 5 3 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $0 . 7 3 3 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $0 . 7 5 9 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $\mathbf { 0 . 7 6 1 { \pm } 0 . 0 3 4 }$ </td></tr><tr><td>vertebral</td><td> $0 . 8 1 3 { \scriptstyle \pm 0 . 0 5 2 }$ </td><td> $0 . 8 0 3 { \scriptstyle \pm 0 . 0 5 1 }$ </td><td> $0 . 8 0 6 { \scriptstyle \pm 0 . 0 5 9 }$ </td><td> $0 . 8 5 5 { \scriptstyle \pm 0 . 0 6 1 }$ </td><td> $\mathbf { 0 . 8 5 8 { \pm } 0 . 0 4 6 }$ </td><td> $0 . 8 3 9 { \scriptstyle \pm 0 . 0 6 1 }$ </td></tr><tr><td>waveform</td><td> $0 . 8 3 2 { \scriptstyle \pm 0 . 0 4 7 }$ </td><td> $0 . 8 6 6 { \scriptstyle \pm 0 . 0 3 4 }$ </td><td> $0 . 8 3 6 { \scriptstyle \pm 0 . 0 2 8 }$ </td><td> $0 . 8 7 7 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td> $0 . 8 8 1 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $\mathbf { 0 . 8 8 7 { \pm } 0 . 0 2 5 }$ </td></tr><tr><td>wp</td><td> $0 . 8 0 9 { \scriptstyle \pm 0 . 0 4 4 }$ </td><td> $0 . 8 5 3 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $0 . 8 5 3 { \scriptstyle \pm 0 . 0 2 8 }$ </td><td> $0 . 8 5 6 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $0 . 8 5 7 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $\mathbf { 0 . 8 6 1 { \pm } 0 . 0 3 2 }$ </td></tr></table>

Table 7: The results of UCI datasets with 15% label noise. The bold is the best one.
<table><tr><td></td><td>1-SVM</td><td>TPMSVM</td><td>Pin-TSVM</td><td>rhingeSVM</td><td>RoBoSS-SVM</td><td> $\mathbf { a R S G T S V M }$ </td></tr><tr><td>acoustic</td><td> $0 . 6 7 0 { \scriptstyle \pm 0 . 0 7 9 }$ </td><td> $0 . 7 6 0 { \scriptstyle \pm 0 . 0 7 9 }$ </td><td> $0 . 7 5 5 { \scriptstyle \pm 0 . 0 7 7 }$ </td><td> $0 . 7 6 7 { \scriptstyle \pm 0 . 0 6 0 }$ </td><td> $0 . 7 8 0 { \scriptstyle \pm 0 . 0 8 8 }$ </td><td> $\mathbf { 0 . 8 5 0 { \scriptstyle \pm 0 . 0 5 7 } }$ </td></tr><tr><td>amphibians</td><td> $0 . 6 8 8 { \pm } 0 . 0 7 5$ </td><td> $0 . 7 3 0 { \scriptstyle \pm 0 . 0 6 7 }$ </td><td>0.730±0.114</td><td> $0 . 7 7 8 { \scriptstyle \pm 0 . 0 7 7 }$ </td><td> $0 . 7 4 0 { \scriptstyle \pm 0 . 0 7 8 }$ </td><td> $\pm 0 . 7 8 4 \pm 0 . 1 0 6$ </td></tr><tr><td>autism</td><td> $0 . 8 2 9 { \scriptstyle \pm 0 . 1 2 7 }$ </td><td> $0 . 9 5 9 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td>0.953±0.025</td><td> $0 . 9 4 9 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td> $0 . 9 6 3 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $\mathbf { 0 . 9 9 2 { \scriptstyle \pm 0 . 0 1 5 } }$ </td></tr><tr><td>bads</td><td> $0 . 8 1 7 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $0 . 9 7 0 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $0 . 9 5 5 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td> $0 . 9 7 3 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td> $0 . 9 7 7 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $\mathbf { 0 . 9 7 8 { \pm } 0 . 0 2 0 }$ </td></tr><tr><td>dccc</td><td> $0 . 6 2 8 { \scriptstyle \pm 0 . 0 6 3 }$ </td><td> $\mathbf { 0 . 8 2 0 { \scriptstyle \pm 0 . 0 4 1 } }$ </td><td>0.807±0.040</td><td> $0 . 8 1 4 { \scriptstyle \pm 0 . 0 3 9 }$ </td><td> $0 . 8 1 7 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td> $0 . 8 1 0 { \scriptstyle \pm 0 . 0 4 1 }$ </td></tr><tr><td>diabetes</td><td> $0 . 6 9 1 { \scriptstyle \pm 0 . 0 8 4 }$ </td><td> $0 . 9 0 8 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $0 . 9 0 2 { \scriptstyle \pm 0 . 0 4 4 }$ </td><td> $0 . 8 6 9 { \scriptstyle \pm 0 . 0 3 9 }$ </td><td> $\mathbf { 0 . 9 1 7 { \pm } 0 . 0 1 6 }$ </td><td> $0 . 9 0 4 { \scriptstyle \pm 0 . 0 3 5 }$ </td></tr><tr><td>gait</td><td> $0 . 8 5 0 { \scriptstyle \pm 0 . 1 4 1 }$ </td><td> $0 . 8 7 5 { \scriptstyle \pm 0 . 1 4 4 }$ </td><td> $0 . 8 5 5 { \scriptstyle \pm 0 . 1 3 8 }$ </td><td> $0 . 8 7 5 { \scriptstyle \pm 0 . 1 4 4 }$ </td><td> $0 . 8 5 5 { \scriptstyle \pm 0 . 1 3 8 }$ </td><td> $\mathbf { 0 . 9 6 0 { \pm } 0 . 0 8 4 }$ </td></tr><tr><td>garments</td><td> $0 . 8 1 8 { \scriptstyle \pm 0 . 0 7 4 }$ </td><td> $0 . 9 9 4 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 9 9 2 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 9 8 0 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $\mathbf { 0 . 9 9 9 { \scriptstyle \pm 0 . 0 0 3 } }$ </td><td> $0 . 9 9 8 { \scriptstyle \pm 0 . 0 0 4 }$ </td></tr><tr><td>heartfailure</td><td> $0 . 6 5 9 { \scriptstyle \pm 0 . 0 7 6 }$ </td><td> $0 . 8 2 6 { \scriptstyle \pm 0 . 0 7 5 }$ </td><td> $0 . 8 2 6 { \scriptstyle \pm 0 . 0 7 5 }$ </td><td> $0 . 8 2 3 { \scriptstyle \pm 0 . 0 5 9 }$ </td><td> $0 . 8 3 0 { \scriptstyle \pm 0 . 0 6 9 }$ </td><td> $\mathbf { 0 . 8 5 6 { \pm } 0 . 0 7 4 }$ </td></tr><tr><td>lsvt</td><td> $0 . 8 4 9 { \scriptstyle \pm 0 . 0 6 0 }$ </td><td> $0 . 7 5 4 { \pm } 0 . 1 3 6$ </td><td> $0 . 6 9 3 { \scriptstyle \pm 0 . 1 2 0 }$ </td><td> $0 . 8 4 8 { \scriptstyle \pm 0 . 0 6 3 }$ </td><td> $0 . 8 1 9 { \scriptstyle \pm 0 . 1 2 0 }$ </td><td> $\mathbf { 0 . 9 2 1 { \pm } 0 . 0 7 4 }$ </td></tr><tr><td>messidor</td><td> $0 . 5 6 1 { \scriptstyle \pm 0 . 0 4 2 }$ </td><td> $0 . 6 9 8 { \scriptstyle \pm 0 . 0 4 5 }$ </td><td> $0 . 6 4 2 { \scriptstyle \pm 0 . 0 3 5 }$ </td><td> $0 . 6 9 9 { \scriptstyle \pm 0 . 0 4 0 }$ </td><td> $0 . 6 4 6 { \scriptstyle \pm 0 . 0 5 3 }$ </td><td> $\pm 0 . 7 3 1 { \pm } 0 . 0 5 1$ </td></tr><tr><td>raisin</td><td> $0 . 7 3 5 { \scriptstyle \pm 0 . 1 8 3 }$ </td><td> $0 . 8 5 9 { \scriptstyle \pm 0 . 0 3 5 }$ </td><td> $0 . 8 5 4 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $0 . 8 5 9 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td> $0 . 8 6 3 { \scriptstyle \pm 0 . 0 3 5 }$ </td><td> $\mathbf { 0 . 8 7 1 { \pm } 0 . 0 3 3 }$ </td></tr><tr><td>shillbidding</td><td> $0 . 9 0 3 { \scriptstyle \pm 0 . 0 5 8 }$ </td><td> $0 . 9 7 3 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $0 . 9 6 8 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $\mathbf { 0 . 9 7 8 { \pm } 0 . 0 1 1 }$ </td><td> $\mathbf { 0 . 9 7 8 { \pm } 0 . 0 1 0 }$ </td><td> $\mathbf { 0 . 9 7 8 { \pm } 0 . 0 1 3 }$ </td></tr><tr><td>tripadvisor</td><td> $0 . 6 0 1 { \scriptstyle \pm 0 . 0 3 5 }$ </td><td> $\mathbf { 0 . 7 5 2 { \scriptstyle \pm 0 . 0 3 5 } }$ </td><td> $0 . 7 4 7 { \scriptstyle \pm 0 . 0 3 5 }$ </td><td> $0 . 7 3 1 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td> $0 . 7 5 0 { \scriptstyle \pm 0 . 0 4 2 }$ </td><td> $0 . 7 3 9 { \scriptstyle \pm 0 . 0 4 4 }$ </td></tr><tr><td>vertebral</td><td> $0 . 6 8 7 { \scriptstyle \pm 0 . 0 8 7 }$ </td><td> $0 . 8 0 6 { \scriptstyle \pm 0 . 0 8 6 }$ </td><td> $0 . 7 4 2 { \scriptstyle \pm 0 . 0 8 3 }$ </td><td> $0 . 8 2 6 { \scriptstyle \pm 0 . 0 7 2 }$ </td><td> $\mathbf { 0 . 8 5 8 { \pm } 0 . 0 5 9 }$ </td><td> $\mathbf { 0 . 8 5 8 { \pm } 0 . 0 5 5 }$ </td></tr><tr><td>waveform</td><td> $0 . 7 0 7 { \scriptstyle \pm 0 . 0 5 0 }$ </td><td> $0 . 8 1 5 { \scriptstyle \pm 0 . 0 2 8 }$ </td><td> $0 . 8 2 8 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $0 . 8 4 3 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $0 . 8 3 1 { \scriptstyle \pm 0 . 0 3 4 }$ </td><td> $\mathbf { 0 . 8 6 0 { \pm } 0 . 0 3 2 }$ </td></tr><tr><td>wp</td><td> $0 . 6 6 9 { \scriptstyle \pm 0 . 0 4 3 }$ </td><td> $0 . 8 2 7 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $0 . 8 1 8 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $0 . 8 2 4 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $0 . 8 2 7 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $\mathbf { 0 . 8 3 1 { \pm } 0 . 0 3 3 }$ </td></tr></table>

where $D _ { n }$ is the number of datasets, and $C _ { k }$ is the number of models,

$$
\chi _ { F } ^ { 2 } = \frac { 1 2 D _ { n } } { C _ { k } ( C _ { k } + 1 ) } \left( \sum _ { i = 1 } ^ { C _ { k } } R _ { i } ^ { 2 } - \frac { C _ { k } ( C _ { k } + 1 ) ^ { 2 } } { 4 } \right) ,
$$

where $R _ { i }$ is the average rank of the i-th model (calculated from the results in Table 6-8). Given $D _ { n } = 1 7$ and $C _ { k } = 6 .$ , then for a noise ratio of 0%, $F _ { F } = 1 4 . 5 0 7$ . For 15%, $F _ { F } = 2 7 . 7 8 8$ . For 35%, $F _ { F } = 2 5 . 1 0 7$ . In our experiment, $F _ { F }$ follows an $F ( 5 , 8 0 )$ distribution. At a significance level of 0.05, the critical value is $F _ { 0 . 0 5 } ( 5 , 8 0 ) = 3 . 3 2 9$

Table 8: The results of UCI datasets with 35% label noise. The bold is the best one.
<table><tr><td></td><td>1-SVM</td><td>TPMSVM</td><td>Pin-TSVM</td><td>rhingeSVM</td><td>RoBoSS-SVM</td><td>aRSGTSVM</td></tr><tr><td>acoustic</td><td> $0 . 5 8 5 { \scriptstyle \pm 0 . 1 1 0 }$ </td><td> $0 . 6 0 5 { \scriptstyle \pm 0 . 1 1 2 }$ </td><td> $0 . 5 9 5 { \scriptstyle \pm 0 . 1 1 2 }$ </td><td> $0 . 6 4 7 { \scriptstyle \pm 0 . 0 8 8 }$ </td><td> $0 . 6 4 2 { \scriptstyle \pm 0 . 0 9 3 }$ </td><td> $\mathbf { 0 . 8 5 3 { \pm } 0 . 0 5 3 }$ </td></tr><tr><td>amphibians</td><td> $0 . 5 2 9 { \scriptstyle \pm 0 . 1 5 3 }$ </td><td> $0 . 5 4 6 { \scriptstyle \pm 0 . 1 5 2 }$ </td><td> $0 . 5 9 3 { \scriptstyle \pm 0 . 0 9 8 }$ </td><td> $0 . 6 0 9 { \scriptstyle \pm 0 . 1 2 8 }$ </td><td> $0 . 5 5 6 { \scriptstyle \pm 0 . 1 1 3 }$ </td><td> $\mathbf { 0 . 7 9 3 { \pm } 0 . 0 6 9 }$ </td></tr><tr><td>autism</td><td> $0 . 6 1 9 { \scriptstyle \pm 0 . 2 9 4 }$ </td><td> $0 . 8 8 2 { \scriptstyle \pm 0 . 0 5 0 }$ </td><td> $0 . 9 2 9 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $0 . 8 9 5 { \scriptstyle \pm 0 . 0 5 7 }$ </td><td> $0 . 9 0 0 { \scriptstyle \pm 0 . 0 5 6 }$ </td><td> $\mathbf { 0 . 9 9 3 { \scriptstyle \pm 0 . 0 2 2 } }$ </td></tr><tr><td>bads</td><td> $0 . 6 2 0 { \scriptstyle \pm 0 . 0 5 4 }$ </td><td> $0 . 9 7 2 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $0 . 9 1 8 { \pm } 0 . 0 3 8$ </td><td> $0 . 9 6 5 { \scriptstyle \pm 0 . 0 1 8 }$ </td><td> $0 . 9 6 9 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $\mathbf { 0 . 9 8 1 { \pm } 0 . 0 1 1 }$ </td></tr><tr><td>dccc</td><td> $0 . 5 7 0 { \scriptstyle \pm 0 . 1 1 3 }$ </td><td> $0 . 7 7 8 { \scriptstyle \pm 0 . 0 5 6 }$ </td><td> $0 . 5 4 9 { \scriptstyle \pm 0 . 1 1 0 }$ </td><td> $0 . 7 8 7 { \scriptstyle \pm 0 . 0 4 7 }$ </td><td> $0 . 7 7 9 { \scriptstyle \pm 0 . 0 4 3 }$ </td><td> $\mathbf { 0 . 7 9 9 { \pm } 0 . 0 5 0 }$ </td></tr><tr><td>diabetes</td><td> $0 . 6 1 0 { \scriptstyle \pm 0 . 1 7 1 }$ </td><td> $0 . 8 1 0 { \scriptstyle \pm 0 . 0 6 6 }$ </td><td> $0 . 8 3 5 { \scriptstyle \pm 0 . 0 5 2 }$ </td><td> $0 . 8 2 1 { \scriptstyle \pm 0 . 0 4 8 }$ </td><td> $0 . 8 3 5 { \scriptstyle \pm 0 . 0 4 7 }$ </td><td> $\mathbf { 0 . 9 0 2 { \pm 0 . 0 3 2 } }$ </td></tr><tr><td>gait</td><td> $0 . 7 5 5 { \scriptstyle \pm 0 . 2 2 7 }$ </td><td> $0 . 6 4 0 { \scriptstyle \pm 0 . 2 5 1 }$ </td><td> $0 . 6 8 5 { \scriptstyle \pm 0 . 2 2 1 }$ </td><td> $0 . 6 8 0 { \scriptstyle \pm 0 . 2 5 8 }$ </td><td> $0 . 6 2 0 { \scriptstyle \pm 0 . 2 5 3 }$ </td><td> $\mathbf { 0 . 9 8 0 { \pm } 0 . 0 6 3 }$ </td></tr><tr><td>garments</td><td> $0 . 7 0 5 { \scriptstyle \pm 0 . 3 4 8 }$ </td><td> $0 . 9 7 3 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td> $0 . 9 8 5 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $0 . 9 6 9 { \scriptstyle \pm 0 . 0 2 3 }$ </td><td> $0 . 9 8 9 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $\mathbf { 0 . 9 9 9 { \scriptstyle \pm 0 . 0 0 3 } }$ </td></tr><tr><td>heartfailure</td><td> $0 . 5 1 9 { \scriptstyle \pm 0 . 1 5 6 }$ </td><td> $0 . 7 3 6 { \pm } 0 . 1 3 9$ </td><td> $0 . 6 9 0 { \scriptstyle \pm 0 . 1 3 7 }$ </td><td> $0 . 7 1 9 { \scriptstyle \pm 0 . 1 1 0 }$ </td><td> $0 . 7 3 3 { \scriptstyle \pm 0 . 1 0 8 }$ </td><td> $\mathbf { 0 . 8 5 3 { \scriptstyle \pm 0 . 0 3 9 } }$ </td></tr><tr><td>lsvt</td><td> $0 . 6 6 6 { \pm } 0 . 1 5 8$ </td><td> $0 . 6 0 8 { \scriptstyle \pm 0 . 1 9 1 }$ </td><td> $0 . 5 8 6 { \scriptstyle \pm 0 . 1 9 3 }$ </td><td> $0 . 6 6 8 { \scriptstyle \pm 0 . 1 6 4 }$ </td><td> $0 . 6 6 6 { \pm } 0 . 1 4 8$ </td><td> $\mathbf { 0 . 9 1 2 { \scriptstyle \pm 0 . 0 8 1 } }$ </td></tr><tr><td>messidor</td><td> $0 . 5 5 4 { \scriptstyle \pm 0 . 0 4 5 }$ </td><td> $0 . 6 1 7 { \scriptstyle \pm 0 . 0 5 2 }$ </td><td> $0 . 5 5 7 { \scriptstyle \pm 0 . 0 6 3 }$ </td><td> $0 . 5 9 0 { \scriptstyle \pm 0 . 0 7 1 }$ </td><td> $0 . 6 0 2 { \scriptstyle \pm 0 . 0 6 5 }$ </td><td> $\mathbf { 0 . 7 1 7 { \pm } 0 . 0 3 5 }$ </td></tr><tr><td>raisin</td><td> $0 . 6 1 5 { \scriptstyle \pm 0 . 2 9 5 }$ </td><td> $0 . 8 4 8 { \pm } 0 . 0 3 8$ </td><td> $0 . 8 4 6 { \scriptstyle \pm 0 . 0 3 9 }$ </td><td> $0 . 8 3 6 { \scriptstyle \pm 0 . 0 3 5 }$ </td><td> $0 . 8 5 1 { \scriptstyle \pm 0 . 0 4 0 }$ </td><td> $\mathbf { 0 . 8 6 8 { \pm } 0 . 0 2 7 }$ </td></tr><tr><td>shillbidding</td><td> $0 . 6 3 2 { \scriptstyle \pm 0 . 0 7 3 }$ </td><td> $0 . 9 7 3 { \scriptstyle \pm 0 . 0 1 8 }$ </td><td> $0 . 9 7 5 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $0 . 9 7 0 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $0 . 9 7 1 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td> $\mathbf { 0 . 9 8 1 { \pm } 0 . 0 1 4 }$ </td></tr><tr><td>tripadvisor</td><td> $0 . 5 3 5 { \scriptstyle \pm 0 . 0 4 6 }$ </td><td> $0 . 6 6 8 { \scriptstyle \pm 0 . 0 8 4 }$ </td><td> $0 . 6 0 2 { \scriptstyle \pm 0 . 1 7 9 }$ </td><td> $0 . 7 2 8 { \pm } 0 . 0 5 8$ </td><td> $0 . 7 1 4 { \scriptstyle \pm 0 . 0 7 1 }$ </td><td> $\mathbf { 0 . 7 4 2 { \pm } 0 . 0 5 3 }$ </td></tr><tr><td>vertebral</td><td> $0 . 6 3 5 { \scriptstyle \pm 0 . 1 7 1 }$ </td><td> $0 . 7 1 6 { \scriptstyle \pm 0 . 0 8 7 }$ </td><td> $0 . 6 0 3 { \scriptstyle \pm 0 . 1 4 3 }$ </td><td> $0 . 7 0 7 { \scriptstyle \pm 0 . 0 6 2 }$ </td><td> $0 . 6 8 7 { \scriptstyle \pm 0 . 0 6 3 }$ </td><td> $\mathbf { 0 . 8 5 5 { \pm 0 . 0 6 0 } }$ </td></tr><tr><td>waveform</td><td> $0 . 6 1 2 { \scriptstyle \pm 0 . 0 6 5 }$ </td><td> $0 . 8 0 2 { \scriptstyle \pm 0 . 0 5 1 }$ </td><td> $0 . 8 0 4 { \scriptstyle \pm 0 . 0 4 7 }$ </td><td> $0 . 7 7 9 { \scriptstyle \pm 0 . 0 5 1 }$ </td><td> $0 . 7 9 3 { \scriptstyle \pm 0 . 0 6 0 }$ </td><td> $\mathbf { 0 . 8 7 7 { \scriptstyle \pm 0 . 0 2 1 } }$ </td></tr><tr><td>wp</td><td> $0 . 5 6 6 { \scriptstyle \pm 0 . 0 8 8 }$ </td><td> $0 . 7 9 8 { \pm } 0 . 0 3 3$ </td><td> $0 . 7 8 6 { \scriptstyle \pm 0 . 0 6 1 }$ </td><td> $0 . 8 0 4 { \scriptstyle \pm 0 . 0 5 1 }$ </td><td> $0 . 8 1 6 { \scriptstyle \pm 0 . 0 3 4 }$ </td><td> $\pm 0 . 8 3 2 { \pm } 0 . 0 3 4$ </td></tr></table>

![](images/5b2e44fffc453e0f5d5e8e8ab49f7c12a0b6db11de0806e6d6c20aeaf4b9a8eb.jpg)  
(a) 0% label noise

![](images/b7c9b14ac0e77dc4f2fd022f107d490d3735cddcda99cea12689f6240d60f207.jpg)  
(b) 15% label noise

![](images/b4bc93cdc4df9c2e2345362fceeb2c44b16f174a377c1903e737f051a6e4b391.jpg)  
Figure 9: The average ranks of six classifiers on the UCI datasets with diferent label noise.

Since $F _ { F } > 3 . 3 2 9$ , the null hypothesis of the Friedman test should be rejected, and the Nemenyi test is conducted next.

If there is a significant diference between two models, the diference in their average ranks must be at least the critical diference, defined as:

$$
C D = q _ { 0 . 0 5 } \sqrt { \frac { C _ { k } ( C _ { k } + 1 ) } { 6 D _ { n } } } .
$$

Given that $D _ { n } = 1 7 , C _ { k } = 6 ,$ and $q _ { 0 . 0 5 } ( 6 ) = 2 . 8 5$ , we obtain $C D = 1 . 7 7 7$

Figure 9 visualizes the average ranking comparison of six Support Vector Regression (SVR) models. According to the Critical Diference (CD) diagram analysis criterion, when the distance between two models on the horizontal axis exceeds the CD value (red solid line), it indicates a statistically significant diference between them. In the subfigure 9(a), aRSGTSVM, rhingeSVM, and RoBoSS-SVM are grouped together, while Pin-TSVM, 1-SVM, and TPMSVM form another group, with significant performance diferences between these two groups. In the subfigure 9(b), the grouping changes to aRSGTSVM, RoBoSS-SVM, and TPMSVM as one group, and 1-SVM, Pin-TSVM, and rhingeSVM as the other group, maintaining significant inter-group diferences. In the subfigure 9(c), aRSGTSVM stands alone as one group, while the remaining models (RoBoSS-SVM, rhingeSVM, 1-SVM, Pin-TSVM, and TPMSVM) form another group, with equally significant performance diferences between groups. It can be observed that aRSGTSVM significantly outperforms other comparative models, and its advantage becomes more pronounced as the noise ratio increases, fully demonstrating the model’s performance stability across diferent label noise scenarios.

## 4.4. Index tracking for regression

In this section, to assess the models out-of-sample performance, we employ six index tracking datasets spanning from January 1, 2025, to July 1, 2025, for experimentation. The dataset is partitioned such that the first 70% (rounded down if not an integer) is used as the training set, with the remainder serving as the test set. During training, the Mean Absolute Error (MAE) is utilized to select the optimal parameters, while the model’s performance is evaluated based on the annual tracking error. The annual tracking error is defined as:

Table 9: TrackingError $Y e a r$ for diferent datasets. The bold is the best one.
<table><tr><td>Index</td><td>SVR</td><td>LASSO</td><td>Elastic Net</td><td>TSVR</td><td>Res-TSVR</td><td>aRSGTSVR</td></tr><tr><td>bz50</td><td>0.138</td><td>0.126</td><td>0.133</td><td>0.120</td><td>0.135</td><td>0.117</td></tr><tr><td>cy200</td><td>0.059</td><td>0.134</td><td>0.078</td><td>0.045</td><td>0.045</td><td>0.040</td></tr><tr><td>hs300</td><td>0.277</td><td>0.277</td><td>0.380</td><td>0.122</td><td>0.138</td><td>0.117</td></tr><tr><td>xf100</td><td>0.638</td><td>0.328</td><td>0.250</td><td>0.368</td><td>0.925</td><td>0.226</td></tr><tr><td>ys50</td><td>0.191</td><td>0.065</td><td>0.068</td><td>0.128</td><td>0.124</td><td>0.057</td></tr><tr><td>zz500</td><td>0.297</td><td>0.271</td><td>0.185</td><td>0.162</td><td>0.109</td><td>0.105</td></tr></table>

![](images/70faaac7a6fe709a712c27fd3dea285afda6d50458f3d8f47f84b27a7b3943e3.jpg)  
(a) bz50

![](images/0850ed9f59775dea3a15f376156e533e7ed8b5d954e2315ff2496d87cbaf1ab2.jpg)  
(b) cy200

![](images/ef4ea0f76c8c1ff66c77a91810758627b0f829a12ff042ed5f1902fb7a9746ce.jpg)  
(c) hs300

![](images/8d5f521c956b616787487e6efcb869175c814caa59c6f8cf5bd3346f54811752.jpg)  
(d) xf100

![](images/3425b84cd93e55066562670e5f6d71aae266d79868a0c412612753725ba8042b.jpg)  
(e) ys50

![](images/a2bc35559d5e10ba6d98a1e62d77b4868ed9d1330aac50cdd743c9831ff9e491.jpg)  
(f) zz500  
Figure 10: Index tracking plots for Stock Indices. The red solid lines are the true values, the blue solid lines are the fitted values on the training set, and the blue dashed lines are the predicted values on the test set.

$$
{ \mathrm { T r a c k i n g E r r o r } } _ { \mathrm { y e a r } } = { \sqrt { 2 5 2 } } \cdot { \sqrt { \frac { \sum _ { t = 1 } ^ { T } \left( e r r _ { t } - { \overline { { e r r } } } \right) ^ { 2 } } { T - 1 } } } ,
$$

where $e r r _ { t } = y _ { t } - { \hat { y } } _ { t }$ for $t = 1 , 2 , \cdots , T , y _ { t }$ denotes the return of an index at time $t , \hat { y } _ { t }$ is the estimated value of $y _ { t } ,$ and err represents the mean of $e r r _ { t }$ over $t = 1 , 2 , \cdots , T$

To mitigate spurious regression and enable the model to focus on replicating relative price movements, converting raw price data to returns is crucial for index tracking. For a raw price series $P _ { t }$ , the return at time t is calculated as:

$$
r _ { t } = \frac { P _ { t } } { P _ { t - 1 } } - 1 , \quad t = 1 , 2 , \cdots , T .
$$

Similarly, the index value $y _ { t }$ at time t is converted to its corresponding return, which

we continue to denote as $y _ { t }$ for consistency.

The experimental results are presented in Table 9. As evidenced by the annual tracking errors, the proposed aRSGTSVR achieves the best performance across all six index tracking datasets, demonstrating strong generalization capability of the model. As illustrated in Figure 10, the fitted and predicted values of the aRSGTSVR model closely align with the true values. This visually confirms the model’s superior fitting performance and robust generalization ability.

## 5. Conclusion

This paper designs a novel asymmetric loss function $L _ { a R }$ based on the RoBoSS loss, which is integrated with ENNHSVM to establish aRSGTSVM for classification and aRSGTSVR for regression tasks. To tackle the challenging optimization issue arising from the combination of a nonconvex loss $L _ { a R }$ and a sparse penalty $l _ { 1 }$ in the objective function, the iPiano algorithm is adopted to achieve eficient and stable optimization. Experimental results in synthetic datasets demonstrate that aRSGTSVM maintains strong robustness against label noise, delivers stable performance under resampling, achieves satisfactory results in high-dimensional scenarios, and presents competitive computational eficiency. Meanwhile, aRSGTSVR also exhibits prominent noise resistance and efective high-dimensional variable selection capability. In realworld datasets, both proposed models acquire favorable generalization performance. The core contribution of this study lies in that the designed asymmetric loss naturally balances model robustness and sparsity, which provides a reliable and superior alternative to conventional symmetric loss functions for practical scenarios sufering from label noise, high-dimensional features, and resampling instability.

It is noteworthy that although this study employs a cross-validation strategy for model parameter optimization, this method can only detect the presence or absence of overfitting rather than alleviate it. Alleviating overfitting requires strategies such as regularization. Furthermore, cross-validation necessitates repeated data partitioning and model retraining, resulting in substantial computational overhead. This issue becomes particularly pronounced in high-dimensional data or complex model scenarios. Given that information criteria (e.g., AIC, BIC) can construct objective functions by integrating model goodness of fit and complexity without requiring repeated data partitioning and retraining, they significantly enhance parameter tuning eficiency while balancing model generalization performance. Therefore, exploring the optimal information criterion tailored to the characteristics of our model represents a highly valuable direction for future research. Additionally, distributed or parallel algorithms can be employed to further improve the computational eficiency of the model.

## Acknowledgments

This research is supported by the National Natural Science Foundation of China (Grant No. 12401664), the Natural Science Foundation of Chongqing Municipality (Grant No. CSTB2024NSCQ-MSX0855), and the Science and Technology Research Program of Chongqing Municipal Education Commission (Grant No. KJQN202400514).

## References

[1] Jayadeva, R. Khemchandani, S. Chandra, Twin support vector machines for pattern classification, IEEE Transactions on Pattern Analysis and Machine Intelligence 29 (2007) 905–910.

[2] X. Chen, J. Yang, Q. Ye, J. Liang, Recursive projection twin support vector machine via within-class variance minimization, Pattern Recognition 44 (2011) 2643–2655.

[3] K. Qi, H. Yang, Elastic net nonparallel hyperplane support vector machine and its geometrical rationality, IEEE Transactions on Neural Networks & Learning Systems 33 (2022) 7199–7209.

[4] Z. Liang, S. Ding, Fuzzy twin support vector machines with distribution inputs, IEEE Transactions on Fuzzy Systems 32 (2024) 240–254.

[5] A. Quadir, M. Sajid, M. Tanveer, Granular ball twin support vector machine, IEEE Transactions on Neural Networks & Learning Systems 36 (2025) 12444– 12453.

[6] Y. Shao, C. Zhang, X. Wang, N. Deng, Improvements on twin support vector machines, IEEE Transactions on Neural Networks 22 (2011) 962–968.

[7] Y. Wu, Y. Liu, Robust truncated hinge loss support vector machines, Journal of the American Statistical Association 102 (2007) 974–983.

[8] H. Wang, Z. Zhu, Y. Shao, Fast support vector machine with low-computational complexity for large-scale classification, IEEE Transactions on Systems, Man, & Cybernetics: Systems 54 (2024) 4151–4163.

[9] X. Huang, L. Shi, J. Suykens, Support vector machine classifier with pinball loss, IEEE Transactions on Pattern Analysis & Machine Intelligence 36 (2014) 984–997.

[10] J. Zhu, S. Rosset, T. Hastie, R. Tibshirani, 1-norm support vector machines, in: Advances in Neural Information Processing Systems, 2004, pp. 1–8.

[11] L. Wang, J. Zhu, H. Zou, The doubly regularized support vector machine, Statistica Sinica (2006) 589–615.

[12] N. Krause, Y. Singer, Leveraging the margin more carefully, in: Proceedings of the Twenty-First International Conference on Machine Learning, 2004, p. 63.

[13] D. Liu, Y. Shi, Y. Tian, Ramp loss nonparallel support vector machine for pattern classification, Knowledge-Based Systems 85 (2015) 224–233.

[14] C. Wang, Q. Ye, P. Luo, N. Ye, L. Fu, Robust capped l1-norm twin support vector machine, Neural Networks 114 (2019) 47–59.

[15] W. Liu, P. Pokharel, J. Principe, Correntropy: Properties and applications in nongaussian signal processing, IEEE Transactions on Signal Processing 55 (2007) 5286–5298.

[16] A. Singh, R. Pokharel, J. Principe, The c-loss function for pattern classification, Pattern Recognition 47 (2014) 441–453.

[17] G. Xu, Z. Cao, B. Hu, J. Principe, Robust support vector machines based on the rescaled hinge loss function, Pattern Recognition 63 (2017) 139–148.

[18] G. Xu, B. Hu, J. Principe, Robust c-loss kernel classifiers, IEEE Transactions on Neural Networks & Learning Systems 29 (2018) 510–529.

[19] J. Ma, L. Yang, Q. Sun, Adaptive robust learning framework for twin support vector machine classification, Knowledge-Based Systems 211 (2021) 106536.

[20] M. Akhtar, M. Tanveer, M. Arshad, Roboss: A robust, bounded, sparse, and smooth loss function for supervised learning, IEEE Transactions on Pattern Analysis & Machine Intelligence 47 (2025) 149–160.

[21] X. Huang, L. Shi, J. Suykens, Asymmetric least squares support vector machine classifiers, Computational Statistics & Data Analysis 70 (2014) 395–405.

[22] X. Shen, L. Niu, Z. Qi, Y. Tian, Support vector machine classifier with truncated pinball loss, Pattern Recognition 68 (2017) 199–210.

[23] L. Yang, H. Dong, Robust support vector machine with generalized quantile loss for classification and regression, Applied Soft Computing 81 (2019) 105483.

[24] M. He, F. He, L. Shi, X. Huang, J. Suykens, Learning with asymmetric kernels: Least squares and feature interpretation, IEEE Transactions on Pattern Analysis & Machine Intelligence 45 (2023) 10044–10054.

[25] K. Ikeda, N. Murata, Geometrical properties of nu support vector machines with diferent norms, Neural Computation 17 (2005) 2508–2529.

[26] H. Zou, M. Yuan, The F<sub>∞</sub>-norm support vector machine, Statistica Sinica (2008) 379–398.

[27] S. Gao, Q. Ye, N. Ye, 1-norm least squares twin support vector machines, Neurocomputing 74 (2011) 3590–3597.

[28] H. Moosaei, M. Hladík, Sparse solution of least-squares twin multi-class support vector machine using ℓ<sub>0</sub> and ℓ<sub>p</sub>-norm for classification and feature selection, Neural Networks 166 (2023) 471–486.

[29] X. Xie, F. Sun, J. Qian, L. Guo, R. Zhang, X. Ye, Z. Wang, Laplacian $l _ { p }$ norm least squares twin support vector machine, Pattern Recognition 136 (2023) 109192.

[30] Y.-H. Shao, W.-J. Chen, N.-Y. Deng, Nonparallel hyperplane support vector machine for binary classification problems, Information Sciences 263 (2014) 22–35.

[31] Y. Xu, Z. Yang, X. Pan, A novel twin support-vector machine with pinball loss, IEEE Transactions on Neural Networks & Learning Systems 28 (2017) 359–370.

[32] F. R. Hampel, Contributions to the theory of robust estimation, University of California, Berkeley, 1968.

[33] M. Akhtar, M. Tanveer, M. Arshad, Robots: A robust bounded twin svm based on roboss loss function, Pattern Recognition (2026) 113653.

[34] R. Khemchandani, K. Goyal, S. Chandra, Twsvr: regression via twin support vector machine, Neural Networks 74 (2016) 14–21.

[35] P. Ochs, Y. Chen, T. Brox, T. Pock, ipiano: Inertial proximal algorithm for nonconvex optimization, SIAM Journal on Imaging Sciences 7 (2014) 1388–1419.

[36] P. S. Bradley, O. L. Mangasarian, Feature selection via concave minimization and support vector machines., in: ICML, volume 98, 1998, pp. 82–90.

[37] X. Peng, Tpmsvm: a novel twin parametric-margin support vector machine for pattern recognition, Pattern recognition 44 (2011) 2678–2692.

[38] X. Peng, Tsvr: an eficient twin support vector machine for regression, Neural Networks 23 (2010) 365–372.

[39] M. Singla, D. Ghosh, K. Shukla, W. Pedrycz, Robust twin support vector regression based on rescaled hinge loss, Pattern Recognition 105 (2020) 107395.

[40] J. Demšar, Statistical comparisons of classifiers over multiple data sets, Journal of Machine learning research 7 (2006) 1–30.