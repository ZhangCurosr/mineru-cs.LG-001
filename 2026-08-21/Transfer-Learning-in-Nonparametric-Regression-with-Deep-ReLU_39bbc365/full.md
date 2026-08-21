# Transfer Learning in Nonparametric Regression with Deep ReLU Networks

Junpeng Ren <sup>1</sup> Carlos Misael Madrid Padilla <sup>2</sup> Yanzhen Chen <sup>3</sup> Oscar Hernan Madrid Padilla <sup>1</sup>

## Abstract

This paper develops a general transfer learning framework for nonparametric regression with data consisting of multiple groups. Under the assumption that groups share a common structure along with group-specific deviations in additive form, the proposed method employs a two-stage offset learning procedure: the first stage pools data from all groups to estimate an overall mean function, and the second stage estimates offsets for each group, yielding final group-level estimators through additive combination. Upper bounds on the $\mathcal { L } _ { 2 }$ error are established for the proposed framework, covering a broad class of nonparametric estimators under mild complexity and noise conditions. When instantiated with deep ReLU networks, explicit convergence rates are derived under hierarchical composition models, demonstrating the ability to overcome the curse of dimensionality. Conditions that enable positive transfer with faster rates are considered, including learning with simpler functions and data augmentation through pooling samples across groups. Various simulations and real-data experiments further validate the effectiveness of the proposed method.

## 1. Introduction

Nonparametric regression is a flexible modeling paradigm that captures complex data relationships without assuming specific functional forms. With the increasing diversity of modern data, transfer learning provides an effective framework for leveraging information from related sources to improve performance on a target, achieving remarkable empirical success in fields such as natural language processing (Daume III´ , 2007; Howard & Ruder, 2018), computer vision (Gong et al., 2012; Tzeng et al., 2017), bioinformatics (Schweikert et al., 2008), transportation (Lu et al., 2019), and epidemiology (Apostolopoulos & Mpesiana, 2020). Consequently, enhancing nonparametric regression through transfer learning becomes a meaningful problem. Meanwhile, the growing complexity of modern data is also reflected in higher dimensionality, as exemplified by text and image data. Deep neural networks (DNNs), as a cornerstone class of nonparametric estimators, have demonstrated outstanding empirical performance on such tasks (Krizhevsky et al., 2012; Vaswani et al., 2017; Radford et al., 2018) and have been theoretically shown to possess strong advantages in modeling such high-dimensional nonlinear structures (Schmidt-Hieber, 2020; Kohler & Langer, 2021; Chen et al., 2022). Therefore, studying how transfer learning can further improve neural network estimators provides a practically meaningful instantiation of this problem. Motivated by these developments, this work investigates nonparametric regression under the transfer learning framework, focusing on deep neural networks as the primary estimator.

We formalize nonparametric regression with data composed of distinct groups that share structural similarities. Given n independent copies of $( X , Y , Z )$ , denoted by $\{ ( x _ { i } , y _ { i } , z _ { i } ) \} _ { i = 1 } ^ { n }$ , where $x _ { i } \in \mathcal { X } \subset \mathbb { R } ^ { p }$ represents covariates, $y _ { i } \in \mathbb { R }$ is the response, and $z _ { i } \in \{ 1 , \ldots , L \}$ indexes the group membership, we assume the data are generated from the model

$$
y _ { i } = f _ { 0 } ( x _ { i } ) + f _ { 0 , z _ { i } } ( x _ { i } ) + \epsilon _ { i } ,\tag{1}
$$

where $\epsilon _ { i }$ are independent errors with $\mathbb { E } ( \epsilon _ { i } \ | \ x _ { i } , z _ { i } ) = 0$ Here, $f _ { 0 }$ represents a shared function common to all groups, while $f _ { 0 , \ell }$ denotes the group-specific deviation for group $\ell \in \{ 1 , \ldots , L \}$ . Our goal is to estimate the group-specific conditional mean function for each group:

$$
g _ { 0 , \ell } ( x ) : = \mathbb { E } ( Y \mid X = x , Z = \ell ) = f _ { 0 } ( x ) + f _ { 0 , \ell } ( x ) .\tag{2}
$$

To tackle this problem, we propose a two-stage offset transfer learning framework inspired by pretraining. In the first stage, we pool data from all L groups to estimate the overall conditional mean function, defined as the average of the conditional mean functions across all groups. In the second stage, for each target group ℓ, we use its group-specific data to estimate the offset between the overall mean function and the group-specific conditional mean. The final estimator combines the overall mean function with the estimated group-specific offset in an additive form.

## Our main contributions are summarized as follows:

General Theoretical Framework. We study a general transfer learning framework for nonparametric regression. In Theorem 3.2, we derive an $\mathcal { L } _ { 2 }$ error upper bound for a broad class of nonparametric estimators under mild complexity and noise conditions that accommodate subexponential noise. We additionally study the variant where the two stages are estimated on independent data via sample splitting, and provide a complementary general bound (Theorem 3.7) that yields a tighter rate. Notably, our framework accommodates the regime in which the number of groups L grows with the total sample size n, capturing the essence of modern machine learning on data of growing size and diversity, in contrast to existing transfer learning results that focus on a fixed number of groups. As a direct consequence of the generality of our framework, we yield the first convergence guarantees for trend filtering (Tibshirani, 2014) in a transfer learning setting (Appendix D.1), and recover the existing convergence rates established by (Wang et al., 2016) for orthogonal series regression using Sobolev sieves up to logarithmic factors (Appendix D.2).

Transfer Learning with Dense ReLU Networks. Building on the general theory, in Theorem 3.4 and Corollary 3.8 we derive explicit upper bounds for transfer learning with dense ReLU neural networks under hierarchical composition models (Kohler & Langer, 2021), showing that the ability to overcome the curse of dimensionality is preserved within the transfer learning framework. As emphasized in (Cagnetta et al., 2024; Danhofer et al., 2025), such property remains critical for explaining the empirical success of DNNs, even as dataset sizes have scaled dramatically in modern architectures such as ImageNet (Deng et al., 2009), ResNet (He et al., 2016), and GPT (Brown et al., 2020). We further identify regimes in which transfer learning with DNNs achieves strictly faster convergence rates than singlegroup estimation, providing theoretical justification for the empirical success of pretrained models. Through extensive simulations across diverse scenarios and two real-data applications, we demonstrate that the proposed estimators consistently outperform a range of competitors.

## 1.1. Related Literature

Transfer Learning. There are vast works of transfer learning having been discussed, including but not limited to transfer learning in models like linear regression (Chen et al., 2015), high dimensional linear regression (Gross & Tibshirani, 2016; Bastani, 2021; Li et al., 2022; Lai et al., 2024), high dimensional generalized linear model (Tian &

Feng, 2023; Li et al., 2024), functional linear regression (Lin & Reimherr, 2022), and graphical models (Li et al., 2023). Various regularization schemes have also been explored to facilitate transfer, including $\ell _ { 1 }$ (Craig et al., 2026), $\ell _ { 2 }$ (Duan & Wang, 2023), and graph-based penalties (Dinh et al., 2022).

In the nonparametric regime, (Cai & Wei, 2021) investigates transfer learning for classification, while a series of works (Wang & Schneider, 2015; Wang et al., 2016; Du et al., 2017; Lin & Reimherr, 2024; Cai & Pu, 2024) focus on nonparametric regression with kernel-based estimators. Specifically, (Wang & Schneider, 2015) introduces a twostage kernel ridge regression (KRR) framework that models the target function as an additive combination of a source function and an offset, demonstrating that transfer learning can improve estimation when the offset is smoother than the target. (Wang et al., 2016) further formalizes this framework under Sobolev smoothness assumptions. Subsequent work (Du et al., 2017) generalizes it via a nonlinear transformation linking the source and target functions, and (Lin & Reimherr, 2024) develops adaptive KRR algorithms that automatically adjust to unknown smoothness levels. Beyond KRR, (Cai & Pu, 2024) considers transfer learning using local polynomial estimators. In contrast to these studies, our theory establishes convergence rates for general nonparametric estimators and achieves faster rates that overcome the curse of dimensionality for hierarchically composed function classes by leveraging deep neural networks.

Another important line of research studies transfer learning through shared representations, where source and target tasks are assumed to depend on a common low-dimensional structure along with task-specific prediction functions (Maurer et al., 2016; Du et al., 2020; Tripuraneni et al., 2020; 2021; Xu & Tewari, 2021; Tian et al., 2023). However, most previous studies in this line discuss neural networks under parametric formulations, including the analyses in (Tripuraneni et al., 2020; Xu & Tewari, 2021). A recent work (Jiao et al., 2024) studies nonparametric transfer with deep ReLU networks under a representation learning framework. Our work differs by adopting an additive formulation and offering relaxed conditions on noise and network constraints, establishing rates that overcome the curse of dimensionality.

Transfer learning with shared representations is also closely tied to multi-task learning, which jointly learns multiple related tasks in parallel to improve performance, often by assuming shared structures across tasks, with a rich body of early work (Caruana, 1997; Baxter, 2000; Evgeniou & Pontil, 2004; Argyriou et al., 2008). The multi-task learning setting typically treats tasks symmetrically and aims to estimate functions for all tasks jointly without distinguishing source and target, in contrast to the transfer learning formulation of leveraging a data-rich source for a data-scarce target. However, the distinction between transfer learning and multi-task learning is not sharply drawn in the literature; for instance, the works cited above (Tripuraneni et al., 2020; Jiao et al., 2024) use transfer learning terminology while adopting symmetric multi-task learning setups, or analyze transfer to a data-scarce target within a multi-task framework. We retain the terminology of transfer learning throughout this work. Nevertheless, our framework accommodates both regimes within a unified theory: the target group proportion over the whole dataset is allowed to vanish as the total sample size n grows (transfer learning with data-scarce target), while our algorithm simultaneously produces estimators for all groups on equal footing (multi-task learning).

Deep Neural Networks as Nonparametric Estimators. A fundamental challenge in nonparametric estimation is the curse of dimensionality (Donoho et al., 2000). Recent theoretical work shows that deep ReLU networks can mitigate this issue under structured function classes, such as hierarchical composition and low-dimensional manifold assumptions (Schmidt-Hieber, 2020; Kohler & Langer, 2021; Chen et al., 2022; Padilla et al., 2022; Jiao et al., 2023; Padilla et al., 2024b;a). In particular, hierarchical composition has been identified as a natural and practically relevant assumption for explaining the empirical success of deep networks (Cagnetta et al., 2024). Building on this line of research, we study transfer learning with deep ReLU networks under hierarchical composition models and establish explicit convergence rates within our two-stage framework.

## 1.2. Notation

The following notation is used throughout the article: $\mathbb { Z } ^ { + }$ denotes the set of positive integers, and N denotes the set of natural numbers. For two positive sequences $a _ { n }$ and $b _ { n } ,$ let $a _ { n } = O ( b _ { n } )$ or $a _ { n } \lesssim b _ { n }$ when $a _ { n } \leq C b _ { n }$ for some constant $C > 0$ which is independent of $n ,$ and $a _ { n } = \Theta ( b _ { n } )$ or $a _ { n } \asymp b _ { n }$ when $a _ { n } = O ( b _ { n } )$ and $b _ { n } = O ( a _ { n } )$ . Besides, we write $a _ { n } \ = \ O ( \mathrm { p o l y } ( \log n ) )$ if there exists a polynomial h such that $a _ { n } \ = \ O ( h ( \log n ) )$ . For a sequence of random variables $X _ { n }$ and a positive sequence $a _ { n } .$ , we write $X _ { n } = O _ { \mathbb { P } } ( a _ { n } )$ if for any $\varepsilon > 0 .$ , there exist constants $M > 0$ and $N > 0$ such that $\mathbb { P } ( | X _ { n } | > M a _ { n } ) < \varepsilon$ for all $n > N$ The $\mathcal { L } _ { \infty }$ -norm of a function $f ( \cdot ) : \mathbb { R } ^ { d } $ R is defined by $\| f \| _ { \infty } = \operatorname* { s u p } _ { \mathbf { x } \in \mathbb { R } ^ { d } } | f ( \mathbf { x } ) |$ . Given the probability distribution $\mathbb { P } _ { \mathbf { X } }$ of the random vector X over , define the $\mathcal { L } _ { 2 } ( \mathbb { P } _ { \mathbf { X } } )$ -norm as $\begin{array} { r } { \| f \| _ { \mathcal L _ { 2 } ( \mathbb { R } _ { \mathbf { X } } ) } : = \left( \int _ { \mathcal { X } } f ^ { 2 } ( \mathbf { x } ) \mathbb { P } _ { \mathbf { X } } ( d \mathbf { x } ) \right) ^ { 1 / 2 } = } \end{array}$ E $\bar { \boldsymbol { \mathrm { z } } } \left[ f ^ { 2 } ( \mathbf { X } ) \right] ^ { 1 / 2 }$ . Besides, given n random samples $\mathbf { x } _ { 1 } ^ { n } : =$ $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { n }$ independently and identically distributed according to $\mathbb { P } _ { \mathbf { X } }$ , the corresponding empirical probability measure is $\begin{array} { r } { \mathbb { P } _ { n } ( \mathbf { x } _ { 1 } ^ { n } ) ~ : = ~ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { \mathbf { x } _ { i } } ( \mathbf { \bar { x } } ) } \end{array}$ Define the empirical <sub>2</sub>-norm as $\begin{array} { r } { \| f \| _ { \mathcal L _ { 2 } ( \mathbb { P } _ { n } ) } : = \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } f ^ { 2 } ( \mathbf { x } _ { i } ) \right) ^ { 1 / 2 } = } \end{array}$ $\begin{array} { r } { \left( \int _ { \mathcal { X } } f ^ { 2 } ( \mathbf { x } ) \mathbb { P } _ { n } ( d \mathbf { x } ) \right) ^ { 1 / 2 } } \end{array}$ For a metric space $( \mathcal { X } , d )$ , let $\kappa \subseteq { \mathcal { X } }$ and $r > 0$ . A subset $C \subset { \mathcal { X } }$ is an r-external cover of if $\textstyle { \mathcal { K } } \subseteq \bigcup _ { x \in C } B _ { r } ( x , d )$ , where $B _ { r } ( x , d )$ denotes the ball of radius r centered at x. The external covering number $N ( r , { \boldsymbol { \kappa } } , d )$ is the minimal cardinality among all r-external covers of $\kappa .$ . For any index set ${ \mathcal { T } } \subseteq \{ 1 , \ldots , n \}$ , we use $\| f \| _ { \mathcal { I } }$ to denote the empirical $\mathcal { L } _ { 2 }$ -norm computed over the subset of data points $\{ \mathbf { x } _ { i } : i \in \mathcal { T } \}$ . For a function $f : \mathbb { R } ^ { d } $ R and $A _ { n } > 0 ,$ let $f _ { A _ { n } }$ denote the truncation of f at level $\mathcal { A } _ { n }$ , defined as $f _ { A _ { n } } ( x ) = \operatorname* { m a x } \{ - \mathcal { A } _ { n } , \operatorname* { m i n } \{ f ( x ) , \mathcal { A } _ { n } \} \}$

## 1.3. Outline

The remainder of this paper is organized as follows. Section 2 introduces the general methodology of the proposed transfer learning framework. Section 3.1 establishes general theoretical upper bounds for our estimator, covering a broad class of nonparametric estimators. Section 3.2 provides concrete upper bounds specialized to deep ReLU networks, with discussion on conditions for positive transfer. Section 3.3 further provides theoretical results for the independent two-stage transfer learning estimator based on data splitting. Section 4 presents simulation studies and real-data experiments, comparing the proposed transfer learning approach with alternative training strategies and estimators in both low- and high-dimensional settings. Finally, Section 5 concludes the paper and discusses potential extensions.

## 2. Methodology

We formalize our transfer learning framework based on a two-stage offset learning procedure, which shares similarities with the approaches in (Wang & Schneider, 2015; Wang et al., 2016; Du et al., 2017; Lin & Reimherr, 2024). Under model (1), we define the overall mean function ${ \bar { f } } ( x ) : = \mathbb { E } ( Y \mid X = x )$ , which can be written as

$$
\bar { f } ( x ) = f _ { 0 } ( x ) + \sum _ { \ell = 1 } ^ { L } \mathbb { P } ( Z = \ell \mid X = x ) \ f _ { 0 , \ell } ( x ) .
$$

In the first stage, we use a nonparametric estimator to esti-<sub>mate</sub> ¯f b<sub>ase</sub>d <sub>on</sub> d<sub>ata poo</sub>l<sub>e</sub>d f<sub>rom a</sub>ll <sub>groups. Let</sub> $\mathcal { F }$ denote the function class of the nonparametric estimator. We construct the estimator as

$$
{ \hat { f } } : = { \underset { f \in { \mathcal { F } } } { \operatorname { a r g m i n } } } \left\{ \sum _ { i = 1 } ^ { n } ( y _ { i } - f ( x _ { i } ) ) ^ { 2 } \right\} ,\tag{3}
$$

and consider its clipped version $\hat { f } _ { A _ { n } }$ with $A _ { n } > 0$ . Next, in the second stage, for $\ell = 1 , \ldots , L$ , we construct groupspecific estimators for the offset between the overall mean $\mathbb { E } ( Y | X = x )$ and the group-specific mean function $g _ { \boldsymbol { 0 } , \ell }$ as

$$
\hat { f } _ { \ell } : = \underset { f \in \mathcal { F } _ { \ell } } { \arg \operatorname* { m i n } } \left. \sum _ { i : z _ { i } = \ell } ( y _ { i } - \hat { f } _ { A _ { n } } ( x _ { i } ) - f ( x _ { i } ) ) ^ { 2 } \right. ,\tag{4}
$$

where $\mathcal { F } _ { \ell }$ denotes the function class for the group-specific nonparametric estimator. The group data used in this stage may be either identical or independent of the data used in the first stage. As in the first stage, we consider the truncated version of the second-stage estimator, denoted by $\hat { f } _ { \ell , B _ { n } }$ The final estimator, which is our main object of theoretical interest, is then constructed as the sum of the two estimators:

$$
\hat { g } _ { \ell } ( x ) : = \hat { f } _ { A _ { n } } ( x ) + \hat { f } _ { \ell , B _ { n } } ( x ) .\tag{5}
$$

## 2.1. Background of Deep ReLU Neural Networks as Nonparametric Estimators

We now introduce the basic notation for deep ReLU neural networks, which serve as the primary class of nonparametric estimators in this paper. Let $\tau ( x ) = \operatorname* { m a x } \{ 0 , x \}$ denote the ReLU activation function, applied componentwise to vectors. A fully connected feedforward neural network with M hidden layers defines a function

$$
f = \Phi _ { M } \circ \tau \circ \Phi _ { M - 1 } \circ \cdots \circ \tau \circ \Phi _ { 1 } ,
$$

where each $\Phi _ { s } : \mathbb { R } ^ { w _ { s - 1 } }  \mathbb { R } ^ { w _ { s } }$ is an affine map of the form $\Phi _ { s } ( x ) = W _ { s } x + b _ { s }$ . Here $w _ { 0 } = d , w _ { M } = 1 , W _ { s } \in$ $\mathbb { R } ^ { w _ { s } \times w _ { s - 1 } }$ , and $b _ { s } \in \mathbb { R } ^ { w _ { s } }$

Following (Kohler & Langer, 2021; Padilla et al., 2024a), we restrict attention to dense architectures in which all hidden layers have the same width ν. We denote by $\mathcal { F } ( M , \nu )$ the corresponding class of deep ReLU networks. Under the proposed transfer learning framework, the first-stage estimator $\hat { f }$ is obtained by empirical risk minimization over $\mathcal { F } ( M , \nu )$ while the second-stage group-specific offset estimator $\hat { f } _ { \ell }$ is obtained from an analogous class $\mathcal { F } ( M _ { \ell } , \nu _ { \ell } )$

## 3. Theory

In this section, we provide theoretical guarantees for the two-stage offset learning framework introduced in Section 2. Suppose the covariate space is $\mathcal { X } = [ 0 , 1 ] ^ { d }$ , which is standard in nonparametric theory (Gyorfi et al. ¨ , 2002). We begin with a mild overlap assumption requiring the group membership probabilities to be bounded away from 0 and 1 given the covariates, similar to the overlap condition commonly assumed in causal inference (Rosenbaum & Rubin, 1983).

Assumption 3.1. For every $\ell \in \{ 1 , \ldots , L \}$ there exists $\underline { { \pi } } _ { \ell } , \overline { { \pi } } _ { \ell }$ such that $\underline { { \pi } } _ { \ell } \asymp \overline { { \pi } } _ { \ell }$ and $\forall x \in { \mathcal { X } }$

$$
0 < \underline { { \pi } } _ { \ell } < \mathbb { P } ( Z = \ell | X = x ) < \overline { { \pi } } _ { \ell } < 1 .
$$

## 3.1. General Result

Our general theoretical analysis proceeds by first establishing a bound for the first-stage estimator $\hat { f } _ { A _ { n } }$ and then leveraging it to provide an upper bound for the final estimator $\hat { g } _ { \ell }$ in (5). Let $\mathcal { F }$ d<sub>enote a gener</sub>i<sub>c</sub> f<sub>unct</sub>i<sub>on c</sub>l<sub>ass</sub> f<sub>rom w</sub>hi<sub>c</sub>h ˆf is obtained. Leveraging Theorem 1 of (Padilla et al., 2024a), under mild complexity conditions on ${ \mathcal { F } } _ { : }$ , we obtain

$$
\Vert \bar { f } - \hat { f } _ { A _ { n } } \Vert _ { \mathcal { L } _ { 2 } } ^ { 2 } = O _ { \mathbb { P } } ( r _ { n } ) ,
$$

where the rate $r _ { n }$ consists of an approximation error term $\phi _ { n } .$ , defined via the existence of $\tilde { f } \in \mathcal { F }$ satisfying $\parallel \bar { f } -$ $\tilde { f } \| _ { \infty } \leq \sqrt { \phi _ { n } }$ , and an estimation error term determined by the complexity of $\mathcal { F }$ (see Theorem B.1 in the Appendix for a formal statement).

In the second stage, for each group $\ell \in \{ 1 , \ldots , L \}$ , we use a separate group-specific nonparametric function class $\mathcal { F } _ { \ell }$ to estimate the offset between overall mean and group mean, which is

$$
G _ { \ell } ( x ) : = f _ { 0 , \ell } ( x ) - \sum _ { k = 1 } ^ { L } f _ { 0 , k } ( x ) \mathbb { P } ( Z = k | X = x ) .\tag{6}
$$

This results in the estimator $\hat { f } _ { \ell , B _ { n } }$ in (4). Building upon the first-stage bound, we now establish bounds for the final group-specific estimator $\hat { g } _ { \ell }$

Theorem 3.2. Under the conditions of Theorem $B . I ,$ and where $\hat { f }$ has been constructed as in (3). Suppose for $\ell \in$ $\{ 1 , \ldots , L \}$ there exists $\widetilde { G } _ { \ell } \in \mathcal { F } _ { \ell }$ such that

$$
\| G _ { \ell } - \widetilde { G } _ { \ell } \| _ { \infty } \leq \sqrt { \phi _ { \ell , n } } ,
$$

where $\phi _ { \ell , n }$ denotes the approximation error. Let $\eta _ { \ell , n } :$ $\mathbb { R } _ { + } \to \mathbb { R } _ { + } f o r \ell = 1 , . . . , L$ be functions such that $\forall \delta \in$ (0, 1),

$$
\operatorname* { m a x } _ { \substack { \frac { n \pi _ { \ell } } { 2 } \leq n _ { \ell } \leq 2 n \overline { { \pi } } _ { \ell } } } \operatorname* { s u p } _ { \{ x _ { i } \} _ { i \in \mathcal { Z } _ { \ell } } } \log N ( \delta , \mathcal { F } _ { \ell , B _ { n } } , \Vert \cdot \Vert _ { \mathcal { Z } _ { \ell } } ) \leq \eta _ { \ell , n } ( \delta ) ,\tag{7}
$$

where $n _ { \ell } = | \mathcal { T } _ { \ell } |$ , where $\mathcal { F } _ { \ell , B _ { n } } : = \{ f _ { B _ { n } } / ( 2 \mathcal { B } _ { n } ) : f \in \mathcal { F } _ { \ell } \}$ and $B _ { n }$ is chosen to be sufficiently large.

Suppose thefunction classes $\mathcal { F } _ { \ell , B _ { n } }$ and $\mathcal { F } _ { A _ { n } }$ satisfy appropriate complexity conditions (see Theorem B.2for aformal statement), and $\mathbb { P } ( \| \epsilon \| _ { \infty } > \mathcal { U } _ { n } ) \to 0 a s n \to \infty ,$ ,for some ${ { \mathcal { U } } _ { n } } > 0$ . Then, $f o r \delta \in ( 0 , 1 )$

$$
\Vert g _ { 0 , \ell } - \hat { g } _ { \ell } \Vert _ { \mathcal { L } _ { 2 } } ^ { 2 } = O _ { \mathbb { P } } \left( \phi _ { \ell , n } + \mathcal { B } _ { n } ^ { 2 } \delta ^ { 2 } + \frac { \mathcal { U } _ { n } ^ { 2 } \eta _ { \ell , n } ( \delta ) } { n \underline { { \pi } } _ { \ell } } + r _ { n } \right)
$$

provided that $\delta ^ { 2 } n \underline { { \pi } } _ { \ell } \to \infty .$

(8)

Theorem 3.2 establishes a general upper bound for twostage transfer learning, where the convergence rate in (8) decomposes into the first-stage error $r _ { n }$ and the other terms as errors from the second-stage estimation of $G _ { \ell } ,$ with $\phi _ { \ell , n }$ and $B _ { n } ^ { 2 } \delta ^ { 2 } + \mathcal { U } _ { n } ^ { 2 } \eta _ { \ell , n } ( \delta ) / ( n \underline { { \pi } } _ { \ell } )$ denoting the approximation and estimation errors, respectively. The result holds under $\delta ^ { 2 } n \underline { { \pi } } _ { \ell } \to \infty ,$ , a mild condition which implies that the expected number of observations in group ℓ grows to . In addition to complexity conditions on $\mathcal { F } _ { \ell }$ similar in form to those in Theorem B.1, the two-stage estimation procedure requires two additional localized complexity constraints, namely conditions (19) and (20) in the detailed version of Theorem 3.2, imposed on the function classes $\mathcal { F }$ and $\mathcal { F } _ { \ell }$ with respect to group ℓ. These complexity conditions are mild, and in Section 3.2 we demonstrate that they are satisfied by standard nonparametric function classes, such as deep ReLU neural networks. Moreover, in Section 3.3, we show that sample splitting can further relax these constraints and potentially accelerate the convergence rate.

## 3.2. Theory for Transfer Learning with Deep ReLU Networks

Our proposed general theorems serve as powerful tools for studying the convergence rates of nonparametric estimators within a two-stage transfer learning framework. In this section, we focus on deep ReLU networks as nonparametric estimators in both stages to further investigate the properties of the proposed framework. Throughout, we assume that the overall mean and offset functions satisfy a hierarchical composited structure (Kohler & Langer, 2021).

Assumption 3.3. Assume the overall mean $\bar { f }$ and offset $G _ { \ell }$ for $\ell \in \{ 1 , \ldots , L \}$ satisfy the hierarchical compositional structure (see Appendix A for a formal statement), with $\bar { f } \in \mathcal { H } ( l _ { 0 } , \mathcal { P } _ { 0 } )$ and $G _ { \ell } \in \mathcal { H } ( R _ { \ell } , \mathcal { P } _ { \ell } )$ . Furthermore, assume that each component function m in the definition of $\bar { f } \mathrm { o r } G _ { \ell }$ can have different smoothness $p _ { m } = q _ { m } + s _ { m }$ , for $q _ { m } \in \mathbb { N }$ $s _ { m } \in ( 0 , 1 ]$ , and of potentially different input dimension $K _ { m } .$ , so that $( p _ { m } , K _ { m } ) \in \mathcal { P } _ { 0 } \cup ( \cup _ { \ell } \mathcal { P } _ { \ell } )$ . Let $K _ { \mathrm { m a x } }$ be the largest input dimension and $p _ { \mathrm { m a x } }$ the largest smoothness of any of the functions $m .$ . Suppose that all the partial derivatives of order less than or equal to $q _ { m }$ are uniformly bounded by constant $c _ { 2 } ,$ , and each function m is Lipschitz continuous with Lipschitz constant $C _ { \mathrm { L i p } } \geq 1$ . Also, assume that max $\{ p _ { \operatorname* { m a x } } , K _ { \operatorname* { m a x } } \} = O ( 1 )$

With Assumption 3.3 in place, we are ready to present our results. Specifically, the convergence rate for the first-stage estimator $\hat { f } _ { A _ { n } }$ , which is estimated by deep ReLU networks class $\mathcal { F } ( M , \nu )$ with appropriately chosen depth and width, is established by leveraging Theorem $2$ of (Padilla et al., 2024a). Consequently, it holds that $\Vert \bar { f } - \hat { f } _ { A _ { n } } \Vert _ { \mathcal { L } _ { 2 } } ^ { 2 } = O _ { \mathbb { P } } ( r _ { n } )$ where the rate $r _ { n }$ depends only on

$$
\phi _ { n } = \operatorname* { m a x } _ { ( p , K ) \in \mathcal { P } _ { 0 } } n ^ { - 2 p / ( 2 p + K ) } ,\tag{9}
$$

the approximation error of $\bar { f }$ by $\mathcal { F } ( M , \nu )$ , with $A _ { n }$ and $\mathcal { U } _ { n }$ chosen appropriately. We refer readers to Theorem B.3 for a formal statement.

Building upon this first-stage result, we present the upper bound for the final transfer-learning estimator $\hat { g } _ { \ell }$ of the group-specific conditional mean, also constructed using

deep neural networks.

Theorem 3.4. Suppose that the conditions of Theorem B.3 hold. Let

$$
\phi _ { \ell , n } = \operatorname* { m a x } _ { ( p , K ) \in \mathcal { P } _ { \ell } } \left( n \underline { { \pi } } _ { \ell } \right) ^ { \frac { - 2 p } { ( 2 p + K ) } } .\tag{10}
$$

Then there exists sufficiently large positive constants $c _ { 3 }$ and $c _ { 4 }$ such that if $M _ { \ell }$ and $\nu _ { \ell }$ are appropriately chosen (see Theorem B.4for aformal statement), then $\hat { g } _ { \ell }$ as defined in (5) with $\mathcal { F } _ { \ell } : = \mathcal { F } ( M _ { \ell } , \nu _ { \ell } )$ , satisfies,

$$
\begin{array} { r l } & { \Vert g _ { 0 , \ell } - \hat { g } _ { \ell } \Vert _ { \mathcal { L } _ { 2 } } ^ { 2 } = O _ { \mathbb { P } } \Big ( \phi _ { \ell , n } \operatorname* { m a x } \{ \mathcal { U } _ { n } ^ { 2 } , \mathcal { B } _ { n } ^ { 2 } \} \log ^ { 4 } \left( \mathcal { B } _ { n } ^ { 2 } n \overline { { \pi } } _ { \ell } \right) } \\ & { \qquad + \phi _ { n } \underline { { \pi } } _ { \ell } ^ { - 1 } \operatorname* { m a x } \{ \mathcal { A } _ { n } , \mathcal { U } _ { n } ^ { 2 } \} \log ^ { 3 } ( n ) \log \left( \mathcal { A } _ { n } ^ { 2 } n \overline { { \pi } } _ { \ell } \right) \Big ) . } \end{array}\tag{11}
$$

provided that log $n \lesssim n \underline { { \pi } } _ { \ell } ,$ and $B _ { n }$ is chosen to satisfy (18). Remark 3.5. Suppose that the following quantity

$$
\operatorname* { m a x } \{ \mathcal { U } _ { n } , \mathcal { A } _ { n } , \mathcal { B } _ { n } , \| \bar { f } \| _ { \infty } , \operatorname* { m a x } _ { \ell = 1 , \ldots , L } \| f _ { 0 , \ell } \| _ { \infty } , \| G _ { \ell } \| _ { \infty } \} ,\tag{12}
$$

grows at most in $O ( \log n ) _ { \mathrm { { \scriptscriptstyle } } }$ . This is a general setting that includes the case where the error distribution can be sub-Gaussian or even sub-exponential. Let us now compare the result in Theorem 3.4 (based on pretraining) with the alternative approach of estimating each $g _ { 0 , \ell }$ in (2) separately:

$$
\hat { h } _ { \ell } : = \underset { g \in \mathcal { F } _ { \ell } } { \arg \operatorname* { m i n } } \sum _ { i : z _ { i } = \ell } \big ( y _ { i } - g ( x _ { i } ) \big ) ^ { 2 } .\tag{13}
$$

The convergence rate of $\hat { h } _ { \ell }$ depends on the approximation error of the class $\mathcal { F } _ { \ell }$ relative to $g _ { 0 , \ell } ( x ) = f _ { 0 } ( x ) + f _ { 0 , \ell } ( x )$ as well as the estimation error $( \mathrm { e . g . }$ ., for ReLU neural networks) based on approximately $n \underline { { \pi } } _ { \ell }$ observations, assuming $\underline { { \pi } } _ { \ell } \asymp$ $\overline { { \pi } } \ell .$ . In contrast, under Theorem 3.4—ignoring logarithmic factors and assuming $\underline { { \pi } } _ { \ell } \asymp 1$ —the rate for our estimator $\hat { g } _ { \ell }$ becomes

$$
\phi _ { n } + \phi _ { \ell , n } ,
$$

where $\phi _ { n } .$ , as defined in $( 9 ) .$ , is the rate for estimating $\bar { f }$ based on n samples, and $\phi _ { \ell , n }$ is the convergence rate for estimating $G _ { \ell } ( x )$ based on approximately $n \underline { { \pi } } _ { \ell } \asymp n$ samples.

Thus, when the functions $\bar { f }$ and $G _ { \ell }$ are less complex than $g _ { 0 , \ell } ,$ pretraining can yield better performance than estimating each $g _ { 0 , \ell }$ separately via $\hat { h } _ { \ell } .$ . This scenario is reasonable: $\bar { f }$ involves a weighted average of the difference functions $f _ { 0 , k } ,$ which can mitigate the influence of extreme values associated with a particular $f _ { 0 , k }$ , while $G _ { \ell }$ may be small or close to zero in regions where the functions $f _ { 0 , k }$ take similar values.

As another example, consider the case where

$$
\frac { \phi _ { n } } { \phi _ { \ell , n } } \lesssim \underline { { \pi } } _ { \ell } ,
$$

for some fixed $\ell \in \{ 1 , \ldots , L \}$ , which can allow for $\overline { { \pi } } \ell \to 0$ Then the convergence rate of our estimator depends only on estimating the function $G _ { \ell } ( x )$ using a neural network class and a sample size of the same order as that required for estimating $g _ { 0 , \ell }$ with a neural network class. In this setting, the comparison between our pretraining-based estimator $\hat { g } _ { \ell }$ and the naive estimator $\hat { h } _ { \ell }$ reduces to determining which function is easier to estimate with the same number of samples using neural networks. When $g _ { 0 , \ell }$ is not substantially different from the functions $\{ g _ { 0 , k } \}$ , the function $G _ { \ell }$ is typically easier to estimate than $g _ { 0 , \ell } ,$ as in problems with shared information across groups (see 6)—precisely the type of setting in which pretraining is most beneficial.

Remark 3.6. Our upper bounds demonstrate two key points: (1) the convergence rate of transfer learning with deep ReLU networks can overcome the curse of dimensionality, and (2) the benefits of transfer learning. However, we do not claim optimality. We emphasize that this is not a limitation of our work. In fact, lower bounds for the hierarchical function class are only known in very specific cases (see Remark 2 in (Kohler & Langer, 2021)).

## 3.3. Independent Two Stages Estimator

In this subsection, we assume that the estimator $\hat { f }$ in (3) is computed using data independent of that in (4), e.g., via sample splitting.

Theorem 3.7. Consider the conditions and notationsfrom Theorem 3.2. However, instead of assuming that (20) holds, suppose that ˆf is computed with data independent of that used in (4). Then $\| g _ { 0 , \ell } - \hat { g } _ { \ell } \| _ { \mathcal { L } _ { 2 } } ^ { 2 }$ is of order

$$
O _ { \mathbb { P } } \left( \phi _ { \ell , n } + r _ { n } + \frac { \mathscr { A } _ { n } ^ { 2 } \log n } { n \underline { { \pi } } _ { \ell } } + \frac { \mathscr { U } _ { n } ^ { 2 } \eta _ { \ell , n } ( \delta ) } { n \underline { { \pi } } _ { \ell } } + \mathcal { B } _ { n } ^ { 2 } \delta ^ { 2 } \right) .
$$

It is important to note that if $\hat { f }$ is computed using data independent of that used in (4), the conclusion of Theorem 3.2 continues to hold under its original assumptions. The main difference between the upper bound in Theorem 3.7 and that in Theorem 3.2 is the relaxation of the complexity constraints (specifically, the replacement of condition (20)) with an additional term $\mathcal { A } _ { n } ^ { 2 } \log n / ( n \underline { { \pi } } \ell )$ . The potential advantage of Theorem 3.7 is that the additional term may yield a faster convergence rate than that in Theorem 3.2. We illustrate this next.

Corollary 3.8. Consider the conditions of Theorems B.3 and 3.4, with the onl<sub>y</sub> modification that ˆf is independent of the data used in (4). Then, the ReLU neural network pretraining estimator $\hat { g } _ { \ell }$ satisfies

$$
\begin{array} { r l } & { \| g _ { 0 , \ell } - \hat { g } _ { \ell } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } = O _ { \mathbb { P } } \Big ( \phi _ { \ell , n } \operatorname* { m a x } \{ \mathcal { U } _ { n } ^ { 2 } , \mathcal { B } _ { n } ^ { 2 } , \mathcal { A } _ { n } ^ { 2 } \} \log ^ { 4 } \big ( \mathcal { B } _ { n } ^ { 2 } n \overline { { \pi } } _ { \ell } \big ) } \\ & { \qquad + \phi _ { n } \log ^ { 3 } ( n ) \log ( \mathcal { A } _ { n } ) \operatorname* { m a x } \{ \mathcal { A } _ { n } , \mathcal { U } _ { n } ^ { 2 } \} \Big ) . } \end{array}
$$

Remark 3.9. Corollary 3.8 shows that the ReLU pretraining estimator (ignoring logarithmic factors) attains the rate $\phi _ { n } + \phi _ { \ell , n } .$ . This rate naturally decomposes into two components: the rate for estimating $\dot { \boldsymbol { f } }$ based on n samples, plus the rate for estimating $G _ { \ell }$ based on $n \underline { { \pi } } _ { \ell }$ samples. The key difference from Theorem 3.4 is that here we are able to remove the factor $\underline { { \pi } } _ { \ell }$ from the denominator on the right-hand side of (11). This distinction matters primarily when $\underline { { \pi } } _ { \ell } \to 0$ If the latter holds, the first stage estimation is done with significantly more samples than if the estimation was done only using the data from the ℓth group (n ${ \mathrm { v s } } n \underline { { \pi } } _ { \ell } )$ , while the second stage estimation is done with the same number of samples (roughly $n \underline { { \pi } } _ { \ell } )$ though potentially for estimating a simpler function.

We conjecture that the appearance of $\underline { { \pi } } _ { \ell }$ in Theorem 3.4 is an artifact of the proof. Indeed, in practice we observe that the estimator performs better without sample splitting—that is, in the setting considered in Theorem 3.4—than with sample splitting, as in Corollary 3.8. That said, both results coincide and yield the same rate when $\underline { { \pi } } _ { \ell } \asymp 1$

Remark 3.10. We illustrate the benefit of transfer learning with DNNs in comparison with classical estimators whose rates depend on the ambient dimension and therefore suffer from the curse of dimensionality, such as transfer learning by kernel smoothing (Du et al., 2017), through a simple example applying Corollary 3.8. Let $X \in [ 0 , \bar { 1 } ] ^ { 1 0 }$ and $\ell \in \{ 1 , 2 , 3 \}$ index three groups. For simplicity, suppose that $\bar { \pi } _ { \ell } = \underline { { \pi } } _ { \ell } = \pi _ { \ell }$ for all ℓ. Consider $g _ { 0 , \ell } ( x ) =$ $f _ { 0 } ( x ) + f _ { 0 , \ell } ( x )$ with $f _ { 0 } ( x ) = | x _ { 1 } + \cdot \cdot \cdot + x _ { 1 0 } - 1 / 2 |$ and $f _ { 0 , \ell } ( x ) = \ell \cdot | x _ { 1 } + x _ { 2 } + x _ { 3 } + x _ { 4 } - 1 / 2 | ^ { 3 / 2 }$ Then $g _ { \boldsymbol { 0 } , \ell }$ <sub>an</sub>d ¯f h<sub>as smoot</sub>h<sub>ness</sub> $p = 1$ on the full $K = 1 0$ ambient space, while the Stage-2 offset $G _ { 1 } ( x ) = ( 1 - \pi _ { 1 } -$ $2 \pi _ { 2 } - 3 \pi _ { 3 } ) | x _ { 1 } + x _ { 2 } + x _ { 3 } + x _ { 4 } - 1 / 2 | ^ { 3 / 2 }$ for group 1 is much simpler, with smoothness $p = 3 / 2$ and intrinsic dimension $K = 4 . ^ { 1 }$ Table 1 provides the convergence rates of four estimators across regimes. As $\pi _ { 1 }$ shrinks, classical TL estimator degrades to $( n \pi _ { 1 } ) ^ { - 3 / 1 3 }$ , while NN with TL remains at $n ^ { - 1 / \bar { 6 } }$ until $\pi _ { 1 } \ll n ^ { - 1 1 / 1 8 }$ and then transitions to $( n \pi _ { 1 } ) ^ { - 3 / 7 }$ , strictly faster than classical TL. This example clearly illustrates how a small $\pi _ { \ell }$ can drastically worsen the rate of transfer learning using classical estimators while NN with TL remains robust. We refer readers to Appendix E for a detailed explanation of this example and other concrete examples illustrating the rate $\phi _ { n } + \phi _ { \ell , n }$ using DNNs.

## 4. Experiments

## 4.1. Numerical Experiments

We conduct experiments in diverse simulation settings to evaluate our proposed transfer learning strategy (denoted as 2-Stage). For deep ReLU networks (NN), we compare

Table 1. Convergence rates of four estimators under different regimes of $\pi _ { 1 }$ . CLS and NN stand for classical estimators and deep ReLU networks, respectively.
<table><tr><td>Regime of π1</td><td>CLS-no-TL NN-no-TL</td><td></td><td>CLS-TL</td><td>NN-TL</td></tr><tr><td> $\pi _ { 1 } \asymp 1$ </td><td> $n ^ { - \frac { 1 } { 6 } }$ </td><td> $n ^ { - \frac { 1 } { 6 } }$ </td><td> $n ^ { - \frac { 1 } { 6 } }$ </td><td> $n ^ { - \frac { 1 } { 6 } }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $n ^ { - \frac { \mathrm { o } } { 1 8 } } \ll \pi _ { 1 } \ll 1$ </td><td>(nπ1)− 116</td><td> $( n \pi _ { 1 } ) ^ { - { \frac { 1 } { 6 } } }$ </td><td> $n ^ { - \frac { 1 } { 6 } }$ </td><td> $n ^ { - \frac { 1 } { 6 } }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $n ^ { - { \frac { 1 1 } { 1 8 } } } \lesssim \pi _ { 1 } \lesssim n ^ { - { \frac { 5 } { 1 8 } } }$ </td><td>(nπ₁)− (nπ₁)−</td><td></td><td> $( n \pi _ { 1 } ) ^ { - { \frac { 3 } { 1 3 } } }$ </td><td> $n ^ { - \frac { 1 } { 6 } }$ </td></tr><tr><td> $n ^ { - 1 } \ll \pi _ { 1 } \ll n ^ { - \frac { 1 1 } { 1 8 } }$ </td><td> $( n \pi _ { 1 } ) ^ { - { \frac { 1 } { 6 } } }$ </td><td> $( n \pi _ { 1 } ) ^ { - { \frac { 1 } { 6 } } }$ </td><td> $( n \pi _ { 1 } ) ^ { - { \frac { 3 } { 1 3 } } }$ </td><td> $( n \pi _ { 1 } ) ^ { - { \frac { 3 } { 7 } } }$ </td></tr></table>

2-Stage with several training strategies, including Pooled, Separate, Pool-w-L, and Top-FT. Specifically, the Pooled strategy trains a single model $\hat { f }$ on all data while ignoring group labels, following (3), and also serves as the first stage of our procedure. We further consider Separate, which trains independent models for each group; Pool-w-L, which pools the data and appends group labels as additional inputs to train a single model; and Top-FT, which fine-tunes only the output layer of the pooled NN for each group. The method of (Jiao et al., 2024), conceptually similar to Top-FT, was also considered but omitted as it did not show advantages in most of our simulations. Besides NN, we compare with Random Forest (denoted as RF, (Breiman, 2001)), implementing the same strategies except for Top-FT. The ptLasso (Craig et al., 2026) is also considered in the high-dimensional setting as a parametric competitor.

We generate n 5000, 10000, 30000, 50000 independent samples for each scenario. Gaussian noise is used in all scenarios. In the low-dimensional settings, we control the noise level via a signal-to-noise ratio (SNR), defined as the ratio between the empirical variance of the noiseless signal computed within each generated dataset and Var(ϵ), with $\mathrm { S N R } \in \{ 2 , 5 , 1 0 \}$ . Each sample is assigned to one of $L = 5$ groups, where group sizes are unbalanced and proportional to the group index $z \in \{ 1 , \ldots , 5 \}$ . In the high-dimensional scenarios, the noise variance is set to $\mathrm { V a r } ( \epsilon ) \in \{ 0 . 1 ^ { 2 } , 1 \}$ To mimic settings with a large number of different groups, each sample is assigned to one of $L = 3 0$ groups with equal group sizes. For each generated dataset, 15% of the samples within each group are randomly held out as a test set. All experiments are repeated over 50 Monte Carlo replications, and performance is evaluated using the mean squared error (MSE) on the test set. Due to space limitations, we present two low-dimensional scenarios with SNR = 5 and two high-dimensional scenario with $\mathrm { V a r } ( \epsilon ) = 1$ , and refer readers to Appendix F for other simulation scenarios and detailed experimental configurations.

Scenario 1. We consider a low-dimensional setting with an additive structure. Let the input vector be $q \quad = { }$ $( q _ { 1 } , \dots , q _ { 1 0 } ) \overset { \mathrm { i . i . d . } } { \sim } \mathrm { U n i f } ( [ 0 , 1 ] ^ { 1 0 } )$ , holding for Scenario 1 and

2. We define four intermediate quantities:

$$
\begin{array} { c c } { { \displaystyle h _ { 1 } ( q ) = \sum _ { i = 1 } ^ { 1 0 } q _ { i } ^ { 2 } , } } & { { \displaystyle h _ { 2 } ( q ) = \sum _ { i = 1 } ^ { 1 0 } | q _ { i } | , } } \\ { { \displaystyle h _ { 3 } ( q ; z ) = \sum _ { i = 1 } ^ { 5 } q _ { i } ^ { 2 } - q _ { z } ^ { 2 } , } } & { { \displaystyle h _ { 4 } ( q ; z ) = \sum _ { i = 1 } ^ { 5 } | q _ { i } | - | q _ { z } | . } } \end{array}
$$

The shared and group-specific components are given by:

$$
\begin{array} { r l } & { f _ { 0 } ( q ) = \log { ( 1 + h _ { 1 } ( q ) \cdot h _ { 2 } ( q ) ) } , } \\ & { f _ { 0 , z } ( q ) = \sqrt { 1 + | h _ { 3 } ( q ; z ) + h _ { 4 } ( q ; z ) | } , } \end{array}
$$

and the response is generated according to the additive model $y = f _ { 0 } ( x ) + f _ { 0 , z } ( x ) + \epsilon$

Scenario 2. We consider a low-dimensional setting with a nonlinear mixture of shared and group-specific components, without an explicit additive structure. Based on the four intermediate quantities defined in Scenario 1, we further define the following compositions:

$$
\begin{array} { l } { { h _ { 5 } ( q ; z ) = \sqrt { { h _ { 1 } ( q ) h _ { 2 } ( q ) + h _ { 3 } ( q ; z ) h _ { 4 } ( q ; z ) } } , } } \\ { { h _ { 6 } ( q ; z ) = ( h _ { 1 } ( q ) + h _ { 3 } ( q ; z ) ) ^ { 2 } / ( h _ { 2 } ( q ) + h _ { 4 } ( q ; z ) ) ^ { 2 } . } } \end{array}
$$

The group-specific conditional mean is then given by

$$
f _ { z } ( x ) = h _ { 5 } ( q ; z ) \cdot h _ { 6 } ( q ; z ) ,
$$

and the response is generated with $y = f _ { z } ( x ) + \epsilon .$

Scenario 3. We consider a high-dimensional setting with shifted latent representations generated from a lowdimensional latent factor model. Let the latent variable be $u = ( u _ { 1 } , \ldots , u _ { 1 0 } ) \sim \mathcal { N } ( 0 , I _ { 1 0 } )$ . The response is generated directly from the latent variable as $\begin{array} { r } { y = \sum _ { j = 1 } ^ { 1 0 } u _ { j } ^ { 2 } + \epsilon } \end{array}$ However, the latent variable u is unobserved; instead, for each group $z ~ \in ~ \{ 1 , \ldots , L \}$ , the observed covariates $q ~ \in \ \mathbb { R } ^ { 1 0 0 }$ are obtained via a group-specific nonlinear transformation of u. Specifically, each group is randomly assigned parameters $w _ { z } \in \{ 1 / 2 , 1 / 3 , \dotsc , 1 / 1 0 \}$ $\nu _ { z } \in \{ 1 / 5 , 1 / 4 , 1 / 3 , 1 / 2 , 1 , 2 , 3 , 4 , 5 \}$ , and $\psi _ { z } \in \{ 2 \pi / k$ $k = 1 , \ldots , 6 \}$ . We define a shifted latent vector $\tilde { u } ^ { ( z ) }$

$$
\tilde { u } _ { j } ^ { ( z ) } = \left\{ \begin{array} { l l } { u _ { j } , } & { \mathrm { i f ~ } j \in \{ 1 , \ldots , 5 \} , } \\ { u _ { j } + w _ { z } \cdot \sin ( \psi _ { z } + \nu _ { z } \cdot u _ { j } ) , } & { \mathrm { i f ~ } j \in \{ 6 , \ldots , 1 0 \} . } \end{array} \right.
$$

The observed covariates are then generated from $\tilde { u } ^ { ( z ) }$ via a two-layer neural network mapping $T : \mathbb { R } ^ { 1 0 }  \mathbb { R } ^ { 1 0 0 }$ defined as $q = T ( \tilde { u } ^ { ( z ) } ) = V \operatorname { t a n h } ( W \tilde { u } ^ { ( z ) } + b )$ , where $W \in \mathbb { R } ^ { 5 0 \times \bar { 1 0 } } , b \in \bar { \mathbb { R } } ^ { 5 0 }$ , and $V \in \mathbb { R } ^ { 1 0 0 \times 5 0 }$ are fixed across all groups and randomly initialized as $W _ { i j } \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { N } \big ( 0 , 1 0 ^ { - 1 } \big )$ $b _ { j } \overset { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , 0 . 1 ^ { 2 } )$ , and $V _ { i j } \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { N } \big ( 0 , 5 0 ^ { - 1 } \big )$ . Under this construction, the estimation target $\dot { f } _ { z } ( q ) = \dot { \mathbb { E } } [ y \mid q , z ]$ is a highly nonlinear and implicit function of the observed q.

Scenario 4. This scenario features a shared lowdimensional latent structure with sparse group-specific signals. Let $u = ( u _ { 1 } , \ldots , u _ { 1 0 } ) \sim \mathcal { N } ( 0 , I _ { 1 0 } )$ be the shared latent variable. The observed covariates $q \in \mathbb { R } ^ { 5 0 0 }$ consist of two parts: the first 410 coordinates arise from a shared nonlinear transformation $q _ { \mathrm { s h a r e } } = V \operatorname { t a n h } ( W u + b )$ where $W ~ \in ~ \mathbb { R } ^ { 5 0 \times 1 0 } , ~ b ~ \in ~ \mathbb { R } ^ { 5 0 } , ~ V ~ \in ~ \mathbb { R } ^ { 4 1 0 \times 5 0 }$ are randomly initialized as $W _ { i j } \stackrel { \mathrm { \scriptsize ~ i . i . d . } } { \sim } \ N ( 0 , 1 0 ^ { - 1 } ) , \ b _ { j } \stackrel { \mathrm { \scriptsize ~ i . i . d . } } { \sim }$ $\mathcal { N } ( 0 , 0 . 1 ^ { 2 } ) , V _ { i j } \overset { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , 5 0 ^ { - 1 } )$ ; the remaining 90 coordinates $q _ { \mathrm { t a i l } } \sim \mathcal { N } ( 0 , I _ { 9 0 } )$ form group-specific sparse signals. Each group z is assigned three disjoint indices from q<sub>tail</sub>, each associated with a randomly selected function from $\{ \sin ( \nu _ { z } \cdot )$ , tanh $( \nu _ { z } . ) , ( 1 + \exp ( \nu _ { z } . ) ) ^ { - 1 } \}$ with a randomly chosen parameter $\nu _ { z } \ \in \ \{ 1 / 5 , 1 / 4 , 1 / 3 , 1 / 2 , 1 , 2 , 3 , 4 , 5 \}$ The response is $\begin{array} { r } { y = \sum _ { j = 1 } ^ { 1 0 } u _ { j } ^ { 2 } + \sum _ { k = 1 } ^ { 3 } \theta _ { z , k } ( q _ { s _ { z , k } } ) + \epsilon , } \end{array}$ where $\theta _ { z , k }$ denotes the function for group z at index $s _ { z , k } .$

Table 2. Average MSE for Scenario 1 and 2 with SNR = 5 over 50 independent trials. For each sample size, the lowest MSE is highlighted in bold.
<table><tr><td>Model / Size</td><td>5000</td><td>10000</td><td>30000</td><td>50000</td></tr><tr><td colspan="5">Scenario 1 (MSE × 102)</td></tr><tr><td>Pooled (NN)</td><td>2.28 (0.2)</td><td>2.13 (0.2)</td><td>2.34 (0.7)</td><td>1.93 (0.3)</td></tr><tr><td>2-Stage (NN)</td><td>0.93 (0.2)</td><td>0.64 (0.1)</td><td>0.34 (0.1)</td><td>0.27 (0.1)</td></tr><tr><td>Separate (NN)</td><td>6.64 (1.2)</td><td>0.97 (0.4)</td><td>0.60 (0.1)</td><td>0.49 (0.1)</td></tr><tr><td>Top-FT (NN)</td><td>2.12 (0.2)</td><td>1.77 (0.2)</td><td>1.14 (0.2)</td><td>0.92 (0.2)</td></tr><tr><td>Pool-w-L (NN)</td><td>1.03 (0.2)</td><td>0.74 (0.2)</td><td>1.03 (1.0)</td><td>0.48 (0.2)</td></tr><tr><td>Pooled (RF)</td><td>7.08 (0.4)</td><td>6.73 (0.3)</td><td>6.55 (0.2)</td><td>6.69 (0.1)</td></tr><tr><td>2-Stage (RF)</td><td>4.10 (0.3)</td><td>3.37 (0.2)</td><td>2.52 (0.1)</td><td>2.36 (0.1)</td></tr><tr><td>Separate (RF)</td><td>7.75 (0.5)</td><td>6.50 (0.3)</td><td>5.32 (0.2)</td><td>5.00 (0.1)</td></tr><tr><td>Pool-w-L (RF)</td><td>6.92 (0.4)</td><td>6.56 (0.3)</td><td>6.37 (0.2)</td><td>6.52 (0.1)</td></tr><tr><td colspan="5">Scenario 2 (MSE × 102)</td></tr><tr><td>Pooled (NN)</td><td>3.97 (0.5)</td><td>3.51 (0.3)</td><td>3.67 (1.1)</td><td>2.77 (0.2)</td></tr><tr><td>2-Stage (NN)</td><td>2.75 (0.3)</td><td>2.09 (0.2)</td><td>1.44 (0.1)</td><td>1.09 (0.1)</td></tr><tr><td>Separate (NN)</td><td>8.84 (1.4)</td><td>3.75 (0.8)</td><td>2.00 (0.1)</td><td>1.75 (0.2)</td></tr><tr><td>Top-FT (NN)</td><td>3.52 (0.3)</td><td>2.92 (0.3)</td><td>1.98 (0.2)</td><td>1.64 (0.2)</td></tr><tr><td>Pool-w-L (NN)</td><td>3.52 (0.7)</td><td>2.73 (0.5)</td><td>2.53 (1.2)</td><td>1.51 (0.2)</td></tr><tr><td>Pooled (RF)</td><td>12.37 (1.0)</td><td>11.43 (0.6)</td><td>10.87 (0.3)</td><td>11.03 (0.3)</td></tr><tr><td>2-Stage (RF)</td><td>7.96 (0.8)</td><td>6.58 (0.5)</td><td>5.12 (0.2)</td><td>4.76 (0.2)</td></tr><tr><td>Separate (RF)</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>15.70 (1.3)</td><td>13.12 (0.7)</td><td>10.31 (0.3)</td><td>9.57 (0.2)</td></tr><tr><td>Pool-w-L (RF)</td><td>12.26 (1.0)</td><td>11.33 (0.6)</td><td>10.78 (0.3)</td><td>10.97 (0.3)</td></tr></table>

Based on the numerical simulations, the deep neural network estimator generally outperforms the other considered estimators, demonstrating the advantages of NN as a nonparametric estimator in handling complex functional composite structures across various dimensions. Notably, the NN trained with the proposed two-stage transfer learning strategy consistently outperforms both alternative training strategies and other estimators, highlighting the advantage of the proposed strategy. Moreover, in low-dimensional settings within RF–based estimators, the two-stage transfer learning strategy achieves the lowest MSE, demonstrating the generalizability of our framework to other estimators.

Table 3. Average MSE for Scenario 3 and 4 with $\mathrm { V a r } ( \epsilon ) = 1$ over 50 independent trials. Random forest–based estimators are omitted due to high computational cost and poor performance. For each sample size, the lowest MSE is highlighted in bold.
<table><tr><td>Model / Size</td><td>5000</td><td>10000</td><td>30000</td><td>50000</td></tr><tr><td colspan="5">Scenario 3 (MSE)</td></tr><tr><td>Pooled (NN)</td><td>4.08 (0.4)</td><td>3.22 (0.4)</td><td>2.63 (0.4)</td><td>2.12 (0.3)</td></tr><tr><td>2-Stage (NN)</td><td>3.95 (0.4)</td><td>2.97 (0.3)</td><td>2.12 (0.2)</td><td>1.77 (0.2)</td></tr><tr><td>Separate (NN)</td><td>18.52 (1.4)</td><td>16.84 (1.1)</td><td>8.01 (0.6)</td><td>5.70 (0.4)</td></tr><tr><td>Top-FT (NN)</td><td>3.97 (0.4)</td><td>3.05 (0.3)</td><td>2.26 (0.3)</td><td>1.87 (0.2)</td></tr><tr><td>Pool-w-L (NN)</td><td>4.37 (0.5)</td><td>3.19 (0.3)</td><td>2.34 (0.3)</td><td>1.85 (0.2)</td></tr><tr><td>ptLasso</td><td>20.14 (1.4)</td><td>19.14(1.2)</td><td>18.76 (0.8)</td><td>18.51 (0.7)</td></tr><tr><td colspan="5">Scenario 4 (MSE)</td></tr><tr><td>Pooled (NN)</td><td>9.13 (0.9)</td><td>6.46 (0.6)</td><td>4.40 (0.5)</td><td>3.56 (0.3)</td></tr><tr><td>2-Stage (NN)</td><td>9.08 (0.9)</td><td>6.33 (0.6)</td><td>3.89 (0.3)</td><td>3.19 (0.2)</td></tr><tr><td>Separate (NN)</td><td>22.33 (1.5)</td><td>21.59 (0.9)</td><td>15.16 (0.6)</td><td>13.50 (0.7)</td></tr><tr><td>Top-FT (NN)</td><td>9.12 (0.9)</td><td>6.40 (0.6)</td><td>4.11 (0.3)</td><td>3.44 (0.3)</td></tr><tr><td>Pool-w-L (NN)</td><td>9.21 (0.9)</td><td>6.47 (0.6)</td><td>4.59 (0.7)</td><td>3.51 (0.3)</td></tr><tr><td> $\mathrm { \small { ~ \frac { \partial ~ } { \ p t L a s s o } ~ } } = \mathrm { \small { ~ \frac { \partial ~ } { \partial ~ } } } - \mathrm { \small { ~ \frac { \partial ~ } { \partial ~ } } } - \mathrm { \small { ~ \frac { \partial ~ } { \partial ~ } } } - \mathrm { \small { ~ \frac { \partial ~ } { \partial ~ } } } \mathrm { \small { ~ \frac { \partial ~ } { \partial ~ } } } - \mathrm { \small { ~ \frac { \partial ~ } { \partial ~ } } } \mathrm { \small { ~ \frac { \partial ~ } { \partial ~ } } } - \mathrm { \small { ~ \frac { \partial ~ } { \partial ~ } } } - \mathrm { \small { ~ \frac { \partial ~ } { \partial ~ } } } - \mathrm { \small { ~ \frac { \partial ~ } { \partial ~ } } } - \mathrm { \small { ~ \frac { \partial ~ } { \partial ~ } } } - \mathrm { \small { ~ \frac { \partial ~ } { \partial ~ } } } - \mathrm { \small { ~ \frac { \partial ~ } { \partial ~ } } } - \mathrm { \small { ~ \frac { \partial ~ } { \partial ~ } } } - \mathrm { \small { ~ \frac { \partial ~ } { \partial ~ } } } - \mathrm { \small { ~ \frac { \partial ~ } { \partial ~ } } } - \mathrm { \small { ~ \frac { \partial ~ } { \partial ~ } } } - \mathrm { \small { ~ \frac { \partial ~ } { \partial ~ } } }$ </td><td></td><td></td><td></td><td></td></tr></table>

## 4.2. Real Data Experiments

We conduct two real-data experiments: a low-dimensional study based on the Beijing PM2.5 dataset (Chen, 2015) and a high-dimensional study based on the UTKFace dataset (Zhang et al., 2017). In this section, we present partial results for UTKFace and refer readers to Appendix G for the complete results across all experiments.

Experiments on the UTKFace Dataset. We conduct realdata experiments to image data for facial age estimation, where the goal is to estimate age conditioned on a facial image input. In this task, shared visual attributes such as skin texture and wrinkles provide universal cues for age estimation, while facial morphology varies across different ethnic groups, making it a natural setting for transfer learning.

Specifically, experiments are conducted on the UTKFace dataset (Zhang et al., 2017), which contains 23,705 facial images annotated with ages ranging from 0 to 116, along with ethnicity labels across five groups: White (43%), Black (19%), Asian (14%), Indian (17%), and Other (7%). We treat ethnicity as the group label and perform neural network–based transfer learning across groups. Following (Baumann et al., 2021), we adopt a more practical approach in which rather than learning directly from raw pixels, we first extract 512-dimensional latent features for each image using a pretrained FaceNet model (Schroff et al., 2015), with weights obtained from an open-source implementation trained on the VGGFace2 dataset (Cao et al., 2018). These features then serve as input covariates for both our proposed framework and competing methods.

We evaluate the five learning strategies from the previous section, focusing exclusively on DNN estimators. For data partitioning, we randomly split the data while maintaining the ethnic group proportions across training (70%), validation (15%), and test (15%) sets. Each neural network instance is a four-layer MLP with 64 hidden units per layer and ReLU activations, trained with early stopping on the validation set. Additional configurations are provided in Section G.2. Table 4 reports the average MSE over 20 random trials on the test set, excluding samples labeled as “Other” ethnicity due to their ambiguous demographic attribution.

Table 4. Overall and group-wise average MSE of age regression models over 20 independent trials. Random forest–based estimators are omitted due to poor performance. The best performance within each group is bolded.
<table><tr><td>Model</td><td>Overall</td><td>White</td><td>Black</td><td>Asian</td><td>Indian</td></tr><tr><td>Pooled</td><td>61.8 (1.7)</td><td>72.2 (2.7)</td><td>60.1 (3.8)</td><td>47.9 (7.8)</td><td>49.2 (2.9)</td></tr><tr><td>2-Stage</td><td>59.6 (1.9)</td><td>68.9 (3.2)</td><td>59.1 (4.1)</td><td>46.3 (8.4)</td><td>48.3 (2.7)</td></tr><tr><td>Separate</td><td>63.1 (2.5)</td><td>72.4 (3.8)</td><td>65.4 (6.0)</td><td>48.2 (8.6)</td><td>49.6 (3.6)</td></tr><tr><td>Top-FT</td><td>61.4 (1.7)</td><td>71.8 (2.5)</td><td>59.8 (3.7)</td><td>47.8 (7.9)</td><td>48.7 (3.0)</td></tr><tr><td>Pool-w-L</td><td>60.7 (2.5)</td><td>71.1 (3.9)</td><td>59.5 (3.4)</td><td>46.7 (8.4)</td><td>47.5 (2.7)</td></tr></table>

From Table 4, the three hybrid strategies combining common and group-specific information consistently outperform the simple Pooled and Separated baselines, demonstrating the value of both leveraging shared patterns across groups and accommodating differences among groups in age estimation. Among all strategies, the proposed two-stage transfer learning achieves the lowest overall MSE, attaining optimal performance on White, Black, and Asian subgroups while yielding substantial improvements on the Indian group relative to Pooled. These results validate the effectiveness of our proposed method. For additional results, including analyses that incorporate gender as an additional grouping factor and experiments using raw image inputs trained directly with MLPs, we refer readers to Appendix G.3.

## 5. Conclusion

This paper investigates transfer learning for nonparametric regression, using fully connected deep neural networks with ReLU activation as the estimator of interest. We assume that each group-specific conditional mean consists of a shared component common across all groups and a group-specific deviation. Our approach employs a two-stage offset learning framework: in the first stage, the overall conditional mean is estimated using pooled data from all groups to capture the shared structure; in the second stage, a group-specific offset is estimated to learn group-level characteristics, which are then combined to form the final estimator. We further discuss scenarios where transfer learning with deep neural networks can be beneficial, including cases where averaging similar groups yields a simpler overall function, where the offset function is simpler than the group-specific conditional mean, and data augmentation with all groups’ data improves the estimation of the overall conditional mean.

Several directions remain open for future work. First, our framework formally assumes an additive decomposition; while our simulations do not require this explicit form to hold, extending this to other coupling forms remains interesting. Second, negative transfer is theoretically possible in our setting when the offset $G _ { \ell }$ is highly complex or when <sub>t</sub>h<sub>e</sub> fi<sub>rst-stage aggregat</sub>i<sub>on pro</sub>d<sub>uces a comp</sub>li<sub>cate</sub>d ¯f<sub>. Suc</sub>h a case may arise when the groups are substantially different; in practice, however, groups in many real-world datasets exhibit strong cross-group similarity (e.g., our real-data experiments), and formal detection of negative transfer is left as future work. Third, our two-stage offset framework is conceptually related to the practically popular last-layer fine-tuning strategy, and developing analogous theory for fine-tuning is an open and important topic. Beyond these, aligning our theory with current practice motivates extensions to more advanced deep architectures, such as convolutional neural networks (LeCun et al., 2002; Krizhevsky et al., 2012) and Transformers (Vaswani et al., 2017). Finally, while this work assumes discrete group differences, a natural extension would be to model continuous variation across smoothly related tasks.

## Software and Data

Implementation of this work is available at https:// github.com/RenJump/Twostage-Trans-DNN.

## Acknowledgements

We greatly thank the anonymous reviewers for their helpful comments and suggestions, which have greatly helped us improve the quality of this work. In preparing the final revision of the paper, OHMP was partially supported by the NSF CAREER Award (No. DMS-2541747).

## Impact Statement

This paper studies transfer learning for nonparametric regression with data from multiple groups and proposes a general two-stage framework with theoretical guarantees. While the proposed methods may have potential applications in areas such as scientific data analysis and other domains involving grouped data, the primary contributions of this work are focuses on improving statistical efficiency and convergence properties. As such, the societal impact of this work is expected to be indirect, mainly by advancing the theoretical understanding of transfer learning and nonparametric regression with deep neural networks, and by informing future research in related areas.

## References

Apostolopoulos, I. D. and Mpesiana, T. A. Covid-19: automatic detection from x-ray images utilizing transfer learning with convolutional neural networks. Physical and engineering sciences in medicine, 43(2):635–640, 2020.

Argyriou, A., Evgeniou, T., and Pontil, M. Convex multitask feature learning. Machine learning, 73(3):243–272, 2008.

Bartlett, P. L., Harvey, N., Liaw, C., and Mehrabian, A. Nearly-tight vc-dimension and pseudodimension bounds for piecewise linear neural networks. Journal of Machine Learning Research, 20(63):1–17, 2019.

Bastani, H. Predicting with proxies: Transfer learning in high dimension. Management Science, 67(5):2964–2984, 2021.

Baumann, P. F., Hothorn, T., and Rugamer, D. Deep condi-¨ tional transformation models. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 3–18. Springer, 2021.

Baxter, J. A model of inductive bias learning. Journal of artificial intelligence research, 12:149–198, 2000.

Bradski, G. The opencv library. Dr. Dobb’s Journal: Software Toolsfor the Professional Programmer, 25(11):120– 123, 2000.

Breiman, L. Random forests. Machine learning, 45(1): 5–32, 2001.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877–1901, 2020.

Cagnetta, F., Petrini, L., Tomasini, U. M., Favero, A., and Wyart, M. How deep neural networks learn compositional data: The random hierarchy model. Physical Review X, 14(3):031001, 2024.

Cai, T. T. and Pu, H. Transfer learning for nonparametric regression: Non-asymptotic minimax analysis and adaptive procedure. arXiv preprint arXiv:2401.12272, 2024.

Cai, T. T. and Wei, H. Transfer learning for nonparametric classification. The Annals of Statistics, 49(1):100–128, 2021.

Cao, Q., Shen, L., Xie, W., Parkhi, O. M., and Zisserman, A. Vggface2: A dataset for recognising faces across pose and age. In 2018 13th IEEE international conference on automatic face & gesture recognition (FG 2018), pp. 67–74. IEEE, 2018.

Caruana, R. Multitask learning. Machine learning, 28(1): 41–75, 1997.

Chen, A., Owen, A. B., and Shi, M. Data enriched linear regression. 2015.

Chen, M., Jiang, H., Liao, W., and Zhao, T. Nonparametric regression on low-dimensional manifolds using deep relu networks: Function approximation and statistical recovery. Information and Inference: A Journal of the IMA, 11 (4):1203–1253, 2022.

Chen, S. Beijing PM2.5. UCI Machine Learning Repository, 2015. DOI: https://doi.org/10.24432/C5JS49.

Craig, E., Pilanci, M., Le Menestrel, T., Narasimhan, B., Rivas, M. A., Gullaksen, S.-E., Dehghannasiri, R., Salzman, J., Taylor, J., and Tibshirani, R. Pretraining and the lasso. Journal of the Royal Statistical Society Series B: Statistical Methodology, 88(1):261–281, 2026.

Danhofer, D. A., D’Ascenzo, D., Dubach, R., and Poggio, T. A. Position: A theory of deep learning must include compositional sparsity. In International Conference on Machine Learning, pp. 81199–81210. PMLR, 2025.

Daume III, H. Frustratingly easy domain adaptation. In´ Proceedings of the 45th annual meeting of the association ofcomputational linguistics, pp. 256–263, 2007.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Dinh, C. T., Vu, T. T., Tran, N. H., Dao, M. N., and Zhang, H. A new look and convergence rate of federated multitask learning with laplacian regularization. IEEE Transactions on Neural Networks and Learning Systems, 35(6):8075– 8085, 2022.

Donoho, D. L. et al. High-dimensional data analysis: The curses and blessings of dimensionality. AMS math challenges lecture, 1(2000):32, 2000.

Du, S. S., Koushik, J., Singh, A., and Poczos, B. Hypothesis´ transfer learning via transformation functions. Advances in neural information processing systems, 30, 2017.

Du, S. S., Hu, W., Kakade, S. M., Lee, J. D., and Lei, Q. Few-shot learning via learning the representation, provably. arXiv preprint arXiv:2002.09434, 2020.

Duan, Y. and Wang, K. Adaptive and robust multi-task learning. The Annals of Statistics, 51(5):2015–2039, 2023.

Evgeniou, T. and Pontil, M. Regularized multi–task learning. In Proceedings ofthe tenth ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 109–117, 2004.

Gong, B., Shi, Y., Sha, F., and Grauman, K. Geodesic flow kernel for unsupervised domain adaptation. In 2012 IEEE conference on computer vision and pattern recognition, pp. 2066–2073. IEEE, 2012.

Gross, S. M. and Tibshirani, R. Data shared lasso: A novel tool to discover uplift. Computational statistics & data analysis, 101:226–235, 2016.

Gyorfi, L., Kohler, M., Krzy ¨ zak, A., and Walk, H. ˙ A distribution-free theory of nonparametric regression. Springer, 2002.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Howard, J. and Ruder, S. Universal language model finetuning for text classification. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 328–339, 2018.

Jiao, Y., Shen, G., Lin, Y., and Huang, J. Deep nonparametric regression on approximate manifolds: Nonasymptotic error bounds with polynomial prefactors. The Annals of Statistics, 51(2):691–716, 2023.

Jiao, Y., Lin, H., Luo, Y., and Yang, J. Z. Deep transfer learning: Model framework and error analysis. arXiv preprint arXiv:2410.09383, 2024.

Kohler, M. and Langer, S. On the rate of convergence of fully connected deep neural network regression estimates. The Annals ofStatistics, 49(4):2231–2249, 2021.

Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012.

Lai, D., Padilla, O. H. M., and Gu, T. Bayesian transfer learning for enhanced estimation and inference. arXiv preprint arXiv:2412.02986, 2024.

LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradientbased learning applied to document recognition. Proceedings ofthe IEEE, 86(11):2278–2324, 2002.

Li, S., Cai, T. T., and Li, H. Transfer learning for highdimensional linear regression: Prediction, estimation and minimax optimality. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(1):149–173, 2022.

Li, S., Cai, T. T., and Li, H. Transfer learning in largescale gaussian graphical models with false discovery rate control. Journal ofthe American Statistical Association, 118(543):2171–2183, 2023.

Li, S., Zhang, L., Cai, T. T., and Li, H. Estimation and inference for high-dimensional generalized linear models with knowledge transfer. Journal ofthe American Statistical Association, 119(546):1274–1285, 2024.

Liang, X., Zou, T., Guo, B., Li, S., Zhang, H., Zhang, S., Huang, H., and Chen, S. X. Assessing beijing’s pm2. 5 pollution: severity, weather impact, apec and winter heating. Proceedings ofthe Royal Society A: Mathematical, Physical and Engineering Sciences, 471(2182):20150257, 2015.

Lin, H. and Reimherr, M. On transfer learning in functional linear regression. CoRR, 2022.

Lin, H. and Reimherr, M. Smoothness adaptive hypothesis transfer learning. In International Conference on Machine Learning, pp. 30286–30316. PMLR, 2024.

Loshchilov, I. and Hutter, F. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016.

Lu, C., Hu, F., Cao, D., Gong, J., Xing, Y., and Li, Z. Transfer learning for driver model adaptation in lane-changing scenarios using manifold alignment. IEEE transactions on intelligent transportation systems, 21(8):3281–3293, 2019.

Madrid Padilla, O. H., Sharpnack, J., Chen, Y., and Witten, D. M. Adaptive nonparametric regression with the knearest neighbour fused lasso. Biometrika, 107(2):293– 310, 2020.

Mammen, E. and Van De Geer, S. Locally adaptive regression splines. The Annals of Statistics, 25(1):387–413, 1997.

Maurer, A., Pontil, M., and Romera-Paredes, B. The benefit of multitask representation learning. Journal of Machine Learning Research, 17(81):1–32, 2016.

Padilla, C. M. M., Padilla, O. H. M., Kei, Y. L., Zhang, Z., and Chen, Y. Confidence interval construction and conditional variance estimation with dense relu networks. arXiv preprint arXiv:2412.20355, 2024a.

Padilla, C. M. M., Zhang, Z., Luo, X., Wang, D., and Padilla, O. H. M. Dense relu neural networks for temporal-spatial model. arXiv preprint arXiv:2411.09961, 2024b.

Padilla, O. H. M., Tansey, W., and Chen, Y. Quantile regression with relu networks: Estimators and minimax rates. Journal of Machine Learning Research, 23(247):1–42, 2022.

Radford, A., Narasimhan, K., Salimans, T., Sutskever, I., et al. Improving language understanding by generative pre-training. 2018.

Rosenbaum, P. R. and Rubin, D. B. The central role of the propensity score in observational studies for causal effects. Biometrika, 70(1):41–55, 1983.

Sadhanala, V. and Tibshirani, R. J. Additive models with trend filtering. The Annals ofStatistics, 47(6):3032–3068, 2019.

Schmidt-Hieber, J. Nonparametric regression using deep neural networks with relu activation function. 2020.

Schroff, F., Kalenichenko, D., and Philbin, J. Facenet: A unified embedding for face recognition and clustering. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pp. 815–823, 2015.

Schweikert, G., Ratsch, G., Widmer, C., and Sch¨ olkopf, B.¨ An empirical analysis of domain adaptation algorithms for genomic sequence analysis. Advances in neural information processing systems, 21, 2008.

Tian, Y. and Feng, Y. Transfer learning under highdimensional generalized linear models. Journal of the American Statistical Association, 118(544):2684–2697, 2023.

Tian, Y., Gu, Y., and Feng, Y. Learning from similar linear representations: Adaptivity, minimaxity, and robustness. arXiv preprint arXiv:2303.17765, 2023.

Tibshirani, R. J. Adaptive piecewise polynomial estimation via trend filtering. 2014.

Tripuraneni, N., Jordan, M., and Jin, C. On the theory of transfer learning: The importance of task diversity. Advances in neural information processing systems, 33: 7852–7862, 2020.

Tripuraneni, N., Jin, C., and Jordan, M. Provable metalearning of linear representations. In International conference on machine learning, pp. 10434–10443. PMLR, 2021.

Tzeng, E., Hoffman, J., Saenko, K., and Darrell, T. Adversarial discriminative domain adaptation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7167–7176, 2017.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Vershynin, R. High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, 2 edition, 2026.

Wang, X. and Schneider, J. G. Generalization bounds for transfer learning under model shift. In UAI, pp. 922–931, 2015.

Wang, X., Oliva, J. B., Schneider, J. G., and Poczos, B.´ Nonparametric risk and stability analysis for multi-task learning problems. In IJCAI, pp. 2146–2152, 2016.

Wasserman, L. All of nonparametric statistics. Springer, 2006.

Xu, Z. and Tewari, A. Representation learning beyond linear prediction functions. Advances in Neural Information Processing Systems, 34:4792–4804, 2021.

Zhang, Z., Song, Y., and Qi, H. Age progression/regression by conditional adversarial autoencoder. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5810–5818, 2017.

## A. Formal Definitions of Hierarchical Composition Models

Definition A.1 $( ( p , C )$ -Smoothness). Let $p = q + s$ for some $q \in \mathbb { N } = \mathbb { Z } ^ { + } \cup \{ 0 \}$ and $0 < s \le 1$ . We say that a function $g : \mathbb { R } ^ { d } $ R is called $( p , C )$ -smooth, if for every $\alpha = ( \alpha _ { 1 } , \ldots , \alpha _ { d } ) \in \mathbb { N } ^ { d }$ , with $d \in \mathbb { Z } ^ { + }$ , where $\textstyle \sum _ { j = 1 } ^ { d } \alpha _ { j } = q$ the partial derivative $\partial ^ { q } g / \left( \partial a _ { 1 } ^ { \alpha _ { 1 } } \ldots \partial a _ { d } ^ { \alpha _ { d } } \right)$ exists and satisfies

$$
\left. \frac { \partial ^ { q } g } { \partial a _ { 1 } ^ { \alpha _ { 1 } } \ldots \partial a _ { d } ^ { \alpha _ { d } } } \left( \boldsymbol { a } \right) - \frac { \partial ^ { q } g } { \partial a _ { 1 } ^ { \alpha _ { 1 } } \ldots \partial a _ { d } ^ { \alpha _ { d } } } \left( \boldsymbol { b } \right) \right. \leq C \Vert \boldsymbol { a } - \boldsymbol { b } \Vert ^ { s }
$$

for all $a , b \in \mathbb { R } ^ { d }$

Building on the definition of $( p , C )$ -smoothness, we now introduce the class of hierarchical composition models.

Definition A.2 (Space of Hierarchical Composition Models, (Kohler & Langer, 2021)). For $l = 1$ and smoothness constraint $\mathcal { P } \subseteq ( 0 , \infty ) \times \mathbb { N }$ the space of hierarchical composition models is given as

$$
\begin{array} { r l r } { \mathcal { H } ( 1 , \mathcal { P } ) } & { : = } & { \big \{ h : \mathbb { R } ^ { d } \to \mathbb { R } : h ( a ) = m \left( a _ { ( \pi ( 1 ) ) } , \dots , a _ { ( \pi ( K ) ) } \right) , \mathrm { ~ w h e r e ~ } m : \mathbb { R } ^ { K } \to \mathbb { R } \mathrm { ~ i s ~ } } \\ & { } & { ( p , C ) \ – \mathrm { s m o o t h ~ f o r ~ s o m e ~ } ( p , K ) \in \mathcal { P } \mathrm { ~ a n d ~ } \pi : \{ 1 , \dots , K \} \to \{ 1 , \dots , d \} \big \} . } \end{array}
$$

For $l > 1$ , we recursively construct

$$
\begin{array} { r l r } { \mathcal { H } ( l , \mathcal { P } ) } & { : = } & { \left\{ h : \mathbb { R } ^ { d } \to \mathbb { R } : h ( a ) = m \left( f _ { 1 } ( a ) , \dots , f _ { K } ( a ) \right) , \mathrm { ~ w h e r e ~ } m : \mathbb { R } ^ { K } \to \mathbb { R } \mathrm { ~ i s ~ } } \\ & { } & { ( p , C ) \mathrm { \cdot s m o o t h ~ f o r ~ s o m e ~ } ( p , K ) \in \mathcal { P } \mathrm { ~ a n d ~ } f _ { i } \in \mathcal { H } ( l - 1 , \mathcal { P } ) \right\} . } \end{array}
$$

Figure 1 gives an illustration of a hierarchical composition model from the class $\mathcal { H } ( 2 , \mathcal { P } )$ , where a complex multivariate function is constructed via a two-level composition of low-dimensional smooth functions acting on subsets of the 10- dimensional input variables.

![](images/cfb19469d936f70c174c3b18cbe4646befd5887f8721216c18af0b21cb46e3be.jpg)  
Figure 1. Illustration of a hierarchical composition model of the class $\mathcal { H } ( 2 , \mathcal { P } )$ .

## B. Formal Statements of Theorems

Theorem B.1. [General Error Boundfor First-Stage Estimation (Theorem 1from (Padilla et al., 2024a)]). Let ${ { \mathcal { U } } _ { n } } > 0$ and suppose that $\tilde { f } \in \mathcal { F }$ is such that

$$
\| \bar { f } - \tilde { f } \| _ { \infty } \leq \sqrt { \phi _ { n } } ,
$$

so that $\phi _ { n }$ is the approximating error. Suppose that $A _ { n }$ is chosen to satisfy

$$
\begin{array} { r } { A _ { n } \geq 8 \operatorname* { m a x } \{ \| \bar { f } \| _ { \infty } + \mathcal { U } _ { n } , 8 \| \bar { f } \| _ { \infty } , 8 \sqrt { \phi _ { n } } \} . } \end{array}
$$

Moreover, let $\mathcal { F } _ { A _ { n } } : = \{ f _ { A _ { n } } / ( 2 A _ { n } ) : f \in \mathcal { F } \}$ and assume that

$$
\operatorname* { s u p } _ { x _ { 1 } , \ldots , x _ { n } \in [ 0 , 1 ] ^ { d } } \log N \left( \delta , \mathcal { F } _ { { A _ { n } } } , \lVert \cdot \rVert _ { n } \right) \leq \eta _ { n } ( \delta )
$$

for some decreasing function $\eta _ { n } : ( 0 , 1 ) \to \mathbb { R } _ { > 0 } .$ . If

$$
\operatorname* { l i m } _ { n \to \infty } \left[ \sum _ { k = 0 } ^ { \infty } \sum _ { \substack { h ^ { \prime } = 1 } } ^ { \infty } \exp \left( - C _ { 1 } \eta _ { n } ( 2 ^ { - k - k ^ { \prime } - 1 } ) \right) + \sum _ { k = 0 } ^ { \infty } \exp \left( - C _ { 2 } \eta _ { n } ( 2 ^ { - k - 1 } ) \right) + \mathbb { P } ( \| \bar { \epsilon } \| _ { \infty } > \mathcal { U } _ { n } ) \right] = 0 ,\tag{14}
$$

for some constants $C _ { 1 } , C _ { 2 } > 0$ and

$$
\operatorname* { s u p } _ { k \in \mathbb { N } } \sum _ { k ^ { \prime } = 1 } ^ { \infty } { \frac { \eta _ { n } ( 2 ^ { - k - k ^ { \prime } } ) } { 2 ^ { 2 k ^ { \prime } } \eta _ { n } ( 2 ^ { - k } ) } } \leq 1 ,\tag{15}
$$

then

$$
\operatorname* { m a x } \{ \| \bar { f } - \hat { f } _ { A _ { n } } \| _ { n } ^ { 2 } , \| \bar { f } - \hat { f } _ { A _ { n } } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } \} = O _ { \mathbb { P } } \left( \phi _ { n } + \frac { \mathcal { U } _ { n } ^ { 2 } \eta _ { n } ( \delta _ { n } ) } { n } + \mathcal { A } _ { n } \delta _ { n } ^ { 2 } \right) ,\tag{16}
$$

where $\delta _ { n }$ is a critical radius of $\cdot _ { \mathcal { F } _ { A _ { n } } }$ , see Definition C.1.

Theorem B.2. [Detailed Statement of Theorem 3.2]. With the notation and conditions from Theorem $B . I ,$ let $r _ { n }$ be the rate ofconvergence of $\hat { f } _ { A _ { n } }$ towards ${ \bar { f } } ,$ namely

$$
r _ { n } : = \phi _ { n } + \frac { \mathcal { U } _ { n } ^ { 2 } \eta _ { n } ( \delta _ { n } ) } { n } + \mathcal { A } _ { n } \delta _ { n } ^ { 2 } ,
$$

and where $\hat { f }$ has been constructed as in (3). Suppose for $\ell \in \{ 1 , \ldots , L \}$ there exists $\widetilde { G } _ { \ell } \in \mathcal { F } _ { \ell }$ such that

$$
\| G _ { \ell } - \widetilde { G } _ { \ell } \| _ { \infty } \leq \sqrt { \phi _ { \ell , n } } ,
$$

where $\phi _ { \ell , n }$ denotes the approximation error. Let $\eta _ { \ell , n } : \mathbb { R } _ { + } \to \mathbb { R } _ { + }$ , befunctions such that

$$
\operatorname* { m a x } _ { \substack { n \frac { \pi } { 2 } \ell } \leq n _ { \ell } \leq 2 n \overline { { \pi } } _ { \ell } } \operatorname* { s u p } _ { \left\{ x _ { i } \right\} _ { i \in \mathcal { I } _ { \ell } } \subset [ 0 , 1 ] ^ { d } } \log N ( \delta , \mathcal { F } _ { \ell , \mathcal { B } _ { n } } , \Vert \cdot \Vert _ { \mathcal { I } _ { \ell } } ) \leq \eta _ { \ell , n } ( \delta ) \forall \delta \in ( 0 , 1 ) ,\tag{17}
$$

where $n _ { \ell } = | \mathcal { T } _ { \ell } |$ , where $\mathcal { F } _ { \ell , B _ { n } } : = \{ f _ { B _ { n } } / ( 2 \mathcal { B } _ { n } ) : f \in \mathcal { F } _ { \ell } \}$ , and $B _ { n }$ is chosen such that

$$
\mathcal { B } _ { n } \geq \operatorname* { m a x } \{ \mathcal { U } _ { n } + \| \bar { f } \| _ { \infty } + \mathcal { A } _ { n } + 2 \operatorname* { m a x } _ { \ell = 1 , \ldots , L } \| f _ { 0 , \ell } \| _ { \infty } , \| G _ { \ell } \| _ { \infty } + \sqrt { \phi _ { \ell , n } } \} .\tag{18}
$$

Suppose that $\delta \in ( 0 , 1 )$ ) satisfies

$$
\operatorname* { m a x } _ { \substack { m : \frac { n \pi } { 2 } \leq m } } \operatorname* { s u p } _ { \substack { \zeta _ { 2 } \sqrt { m } _ { \ell } \left\{ { x _ { j } } \right\} _ { j = 1 } ^ { m } \subset \left[ 0 , 1 \right] ^ { d } } } \left\{ \frac { 1 } { \sqrt { m } } \int _ { \delta ^ { 2 } / 4 8 } ^ { \delta } \sqrt { \log N \left( t / 4 , \mathcal { F } _ { \ell , B _ { n } } , \Vert \cdot \Vert _ { m } \right) } d t + \frac { \delta } { \sqrt { m } } \sqrt { c _ { 1 } \log ( 4 8 / \delta ^ { 2 } ) } \right\} \lesssim \delta ^ { 2 } ,\tag{19}
$$

and

$$
\operatorname* { m a x } _ { \substack { m : \frac { n \pi } { 2 } \leq m } } \operatorname* { s u p } _ { \varepsilon \downarrow \gamma _ { j = 1 } ^ { m } \subseteq [ 0 , 1 ] ^ { d } } \left\{ \frac { 1 } { \sqrt { m } } \int _ { \delta ^ { 2 } / 4 8 } ^ { \delta } \sqrt { \log N \left( t / 4 , \mathcal { F } _ { A _ { n } } , \Vert \cdot \Vert _ { m } \right) } d t + \frac { \delta } { \sqrt { m } } \sqrt { c _ { 1 } \log ( 4 8 / \delta ^ { 2 } ) } \right\} \lesssim \delta ^ { 2 } ,\tag{20}
$$

then

$$
\Vert g _ { 0 , \ell } - \hat { g } _ { \ell } \Vert _ { \mathcal { L } _ { 2 } } ^ { 2 } = O _ { \mathbb { P } } \left( \phi _ { \ell , n } + r _ { n } + \frac { \mathcal { U } _ { n } ^ { 2 } \eta _ { \ell , n } ( \delta _ { n } ) } { n \underline { { \pi } } _ { \ell } } + \mathcal { B } _ { n } ^ { 2 } \delta _ { n } ^ { 2 } \right)\tag{21}
$$

provided that $n \underline { { { \pi } } } _ { \ell } ^ { 3 } / \overline { { { \pi } } } _ { \ell } ^ { 2 } \to \infty , \delta _ { n } ^ { 2 } n \underline { { { \pi } } } _ { \ell } \to \infty ,$ , and

$$
\operatorname* { l i m } _ { n \to \infty } \left[ \sum _ { k = 0 } ^ { \infty } \sum _ { k ^ { \prime } = 1 } ^ { \infty } \exp \left( - C _ { 1 } \eta _ { \ell , n } ( 2 ^ { - k - k ^ { \prime } - 1 } ) \right) + \sum _ { k = 0 } ^ { \infty } \exp \left( - C _ { 2 } \eta _ { \ell , n } ( 2 ^ { - k - 1 } ) \right) + \mathbb { P } ( \left\| \epsilon \right\| _ { \infty } > \mathcal { U } _ { n } ) \right] = 0\tag{22}
$$

for some constants $C _ { 1 } , C _ { 2 } > 0$ , and

$$
\operatorname* { s u p } _ { k \in \mathbb { N } } \sum _ { k ^ { \prime } = 1 } ^ { \infty } { \frac { \eta _ { \ell , n } ( 2 ^ { - k - k ^ { \prime } } ) } { 2 ^ { 2 k ^ { \prime } } \eta _ { \ell , n } ( 2 ^ { - k } ) } } \leq 1 .\tag{23}
$$

Theorem B.3. [Upper Bound for First-Stage Estimation with Deep ReLU Network (Theorem 2 in (Padilla et al., 2024a))]. Suppose that $\bar { f } \in \mathcal { H } ( l _ { 0 } , \mathcal { P } _ { 0 } )$ for some $l _ { 0 } \in \mathbb { N }$ and $\mathcal { P } _ { 0 } \subset [ 1 , \infty ) \times \mathbb { N }$ . Furthermore, assume that each function m in the definition of <sup>¯</sup>f can have different smoothness $p _ { m } = q _ { m } + s _ { m } ,$ for $q _ { m } \in  { \mathbb { N } } , s _ { m } \in ( 0 , 1 ]$ , and ofpotentially different input dimension $K _ { m } ,$ , so that $( p _ { m } , K _ { m } ) \in \mathcal { P } _ { 0 }$ . Let $K _ { \mathrm { m a x } }$ be the largest input dimension and $p _ { \mathrm { m a x } }$ the largest smoothness ofany ofthefunctions m. Suppose that all the partial derivatives oforder less than or equal to $q _ { m }$ are uniformly bounded by constant $c _ { 2 } ,$ and eachfunction m is Lipschitz continuous with Lipschitz constant $C _ { \mathrm { L i p } } \geq 1$ . Also, assume that max $\{ p _ { \operatorname* { m a x } } , K _ { \operatorname* { m a x } } \} = O ( 1 )$ . Let

$$
\phi _ { n } = \operatorname* { m a x } _ { ( p , K ) \in \mathcal { P } _ { 0 } } n ^ { \frac { - 2 p } { ( 2 p + K ) } } .\tag{24}
$$

Then there exists sufficiently large positive constants $c _ { 3 }$ and $c _ { 4 }$ such that if

$$
M = \left\lceil c _ { 3 } \log n \right\rceil \quad a n d \quad \nu = \left\lceil c _ { 4 } \operatorname* { m a x } _ { ( p , K ) \in \mathcal { P } _ { 0 } } n ^ { \frac { K } { 2 ( 2 p + K ) } } \right\rceil\tag{25}
$$

or

$$
M = \left\lceil c _ { 3 } \operatorname* { m a x } _ { ( p , K ) \in \mathcal { P } _ { 0 } } n ^ { \frac { K } { 2 ( 2 p + K ) } } \log n \right\rceil \quad a n d \quad \nu = \left\lceil c _ { 4 } \right\rceil ,\tag{26}
$$

then $\hat { f } _ { A _ { n } }$ , with $\hat { f }$ as defined in (3) with $\mathcal { F } : = \mathcal { F } ( M , \nu )$ , satisfies,

$$
\Vert \bar { f } - \hat { f } _ { A _ { n } } \Vert _ { \mathcal { L } _ { 2 } } ^ { 2 } = O _ { \mathbb { P } } \left( r _ { n } \right)\tag{27}
$$

where

$$
r _ { n } : = \frac { \operatorname* { m a x } \{ A _ { n } , \mathcal { U } _ { n } ^ { 2 } \} \log n } { n } + \phi _ { n } \log ^ { 3 } ( n ) \log ( A _ { n } ) \operatorname* { m a x } \{ A _ { n } , \mathcal { U } _ { n } ^ { 2 } \} ,\tag{28}
$$

provided that

$$
\operatorname* { l i m } _ { n \to \infty } \mathbb { P } ( \| \epsilon \| _ { \infty } > \mathcal { U } _ { n } ) = 0
$$

and

$$
\mathcal { A } _ { n } \geq 8 \operatorname* { m a x } \{ \| \bar { f } \| _ { \infty } + \mathcal { U } _ { n } , 8 \| \bar { f } \| _ { \infty } , 8 \sqrt { \phi _ { n } } \}
$$

hold.

Theorem B.4. [Detailed Statement of Theorem 3.4.] Suppose that the conditions of Theorem B.3 hold. Also, for $\ell \in \{ 1 , \ldots , L \}$ , let

$$
G _ { \ell } ( x ) : = f _ { 0 , \ell } ( x ) - \sum _ { k = 1 } ^ { L } f _ { 0 , k } ( x ) \mathbb { P } ( Z = k | X = x ) .
$$

Suppose that $G _ { \ell } \in \mathcal { H } ( R _ { \ell } , \mathcal { P } _ { \ell } )$ for some $R _ { \ell } \in \mathbb { N }$ and $\mathcal { P } _ { \ell } \subset [ 1 , \infty ) \times \mathbb { N }$ . Furthermore, assume that each function m in the definition of G can have different smoothness $p _ { m } = q _ { m } + s _ { m } ,$ for $q _ { m } \in  { \mathbb { N } } , s _ { m } \in ( 0 , 1 ]$ , and of potentially different input dimension $K _ { m } ,$ , so that $\left( p _ { m } , K _ { m } \right) \in \mathcal { P } _ { \ell } .$ . Let $K _ { \ell , \mathrm { m a x } }$ be the largest input dimension and $p _ { \ell , \mathrm { m a x } }$ the largest smoothness ofany ofthefunctions m. Suppose that all the partial derivatives oforder less than or equal to $q _ { m }$ are uniformly bounded by constant $c _ { 2 } ,$ , and eachfunction m is Lipschitz continuous with Lipschitz constant $C _ { \mathrm { L i p } } \geq 1$ . Also, assume that max $\{ p _ { \ell , \mathrm { { m a x } } } , K _ { \ell , \mathrm { { m a x } } } \} = O ( 1 )$ . Let

$$
\phi _ { \ell , n } = \operatorname* { m a x } _ { ( p , K ) \in \mathcal { P } _ { \ell } } \left( n \underline { { \pi } } _ { \ell } \right) ^ { \frac { - 2 p } { ( 2 p + K ) } } .\tag{29}
$$

Then there exists sufficiently large positive constants $c _ { 3 }$ and $c _ { 4 }$ such that if

$$
M _ { \ell } = \lceil c _ { 3 } \log ( n \underline { { \pi } } _ { \ell } ) \rceil \quad a n d \quad \nu _ { \ell } = \left\lceil c _ { 4 } \operatorname* { m a x } _ { ( p , K ) \in \mathcal { P } _ { \ell } } ( n \underline { { \pi } } _ { \ell } ) ^ { \frac { K } { 2 ( 2 p + K ) } } \right\rceil\tag{30}
$$

or

$$
M _ { \ell } = \left\lceil c _ { 3 } \operatorname* { m a x } _ { ( p , K ) \in \mathcal { P } _ { \ell } } ( n _ { \underline { { \pi } } _ { \ell } } ) ^ { \frac { K } { 2 ( 2 p + K ) } } \log ( n _ { \underline { { \pi } } _ { \ell } } ) \right\rceil \quad a n d \quad \nu _ { \ell } = \left\lceil c _ { 4 } \right\rceil ,\tag{31}
$$

then $\hat { g } _ { \ell }$ as defined in (5) with $\mathcal { F } _ { \ell } : = \mathcal { F } ( M _ { \ell } , \nu _ { \ell } )$ , satisfies,

$$
\begin{array} { r } { \| g _ { 0 , \ell } - \hat { g } _ { \ell } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } = O _ { \mathbb { P } } \left( \frac { \phi _ { n } \operatorname* { m a x } \left\{ \mathcal { A } _ { n } , \mathcal { U } _ { n } ^ { 2 } \right\} \log ^ { 3 } \left( n \right) \log \left( \mathcal { A } _ { n } ^ { 2 } n \overline { { \pi } } _ { \ell } \right) } { \underline { { \pi } } _ { \ell } } + \phi _ { \ell , n } \operatorname* { m a x } \{ \mathcal { U } _ { n } ^ { 2 } , \mathcal { B } _ { n } ^ { 2 } \} \log ^ { 4 } ( \mathcal { B } _ { n } ^ { 2 } n \overline { { \pi } } _ { \ell } ) \right) } \end{array}\tag{32}
$$

provided that $n \underline { { \pi } } _ { \ell } ^ { 3 } / \overline { { \pi } } _ { \ell } ^ { 2 } \to \infty , B _ { n }$ is chosen to satisfy (18).

## C. Proofs

## C.1. Preliminaries

Definition C.1. Define $B _ { \infty } ( 1 ) : = \{ f : [ 0 , 1 ] ^ { d } \to \mathbb { R } : \| f \| _ { \infty } \leq 1 \}$ . Given a function class $\mathcal { F }$ with $\mathcal { F } \subset B _ { \infty } ( 1 )$ , we call $\delta _ { n } > 0$ a critical radius for if

$$
\mathbb { E } \left( \operatorname* { s u p } _ { f \in \mathrm { s t a r } ( \mathcal { F } ) : \| f \| _ { n } \leq \delta _ { n } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \xi _ { i } f ( x _ { i } ) \biggm | \{ x _ { i } \} _ { i = 1 } ^ { n } \right) \leq \delta _ { n } ^ { 2 } ,
$$

where $\xi _ { 1 } , \ldots , \xi _ { n }$ are independent Rademacher variables independent of $\{ x _ { i } \} _ { i = 1 } ^ { n }$ and where star( ) is defined as

$$
\operatorname { s t a r } ( { \mathcal { F } } ) : = \{ \lambda f : \lambda \in [ 0 , 1 ] , f \in { \mathcal { F } } \} .
$$

## C.2. Auxiliary Lemmas

Lemma C.2. Let $\tilde { y } _ { i } = y _ { i } - \hat { f } _ { A _ { n } } ( x _ { i } ) f o r i = 1 , \dots , n$ . Then

$$
\tilde { y } _ { i } = f _ { 0 , z _ { i } } ( x _ { i } ) - \sum _ { \ell = 1 } ^ { L } f _ { 0 , \ell } ( x _ { i } ) \mathbb { P } ( Z = \ell | X = x _ { i } ) + ( \bar { f } ( x _ { i } ) - \hat { f } _ { A _ { n } } ( x _ { i } ) ) + \epsilon _ { i } ,
$$

f $r i = 1 , \ldots , n .$

Proof. Notice that

$$
\begin{array} { r c l } { \tilde { y } _ { i } } & { = } & { y _ { i } - \hat { f } _ { A _ { n } } ( x _ { i } ) } \\ & { = } & { y _ { i } + \big ( \bar { f } ( x _ { i } ) - \hat { f } _ { A _ { n } } ( x _ { i } ) \big ) - \bar { f } ( x _ { i } ) } \\ & { = } & { f _ { 0 } ( x _ { i } ) + f _ { 0 , z _ { i } } ( x _ { i } ) + \epsilon _ { i } + \big ( \bar { f } ( x _ { i } ) - \hat { f } _ { A _ { n } } ( x _ { i } ) \big ) - \bar { f } ( x _ { i } ) } \\ & { = } & { f _ { 0 } ( x _ { i } ) + f _ { 0 , z _ { i } } ( x _ { i } ) + \epsilon _ { i } + \big ( \bar { f } ( x _ { i } ) - \hat { f } _ { A _ { n } } ( x _ { i } ) \big ) - \big [ f _ { 0 } ( x _ { i } ) + \sum _ { \ell = 1 } ^ { L } f _ { 0 , \ell } ( x _ { i } ) \mathbb { P } ( Z = \ell | X = x _ { i } ) . \big ] } \\ & { = } & { f _ { 0 , z _ { i } } ( x _ { i } ) - { \displaystyle \sum _ { \ell = 1 } ^ { L } } f _ { 0 , \ell } ( x _ { i } ) \mathbb { P } ( Z = \ell | X = x _ { i } ) + \big ( \bar { f } ( x _ { i } ) - \hat { f } _ { A _ { n } } ( x _ { i } ) \big ) + \epsilon _ { i } . } \end{array}
$$

Lemma C.3. Suppose that the event $\Omega _ { 1 } = \{ \| \epsilon \| _ { \infty } \leq \mathcal { U } _ { n } \}$ holds and $B _ { n }$ satisfies

$$
\mathcal { B } _ { n } \geq \mathcal { U } _ { n } + \| \bar { f } \| _ { \infty } + \mathcal { A } _ { n } + 2 \operatorname* { m a x } _ { \ell = 1 , \ldots , L } \| f _ { 0 , \ell } \| _ { \infty } .\tag{33}
$$

Then $\| \tilde { y } \| _ { \infty } \leq B _ { n }$

Proof. Notice that by Lemma C.2,

$$
\begin{array} { r l r } { | \tilde { y } _ { i } | } & { \leq } & { \| f _ { 0 , z _ { i } } \| _ { \infty } + \underset { \ell = 1 , \ldots , L } { \operatorname* { m a x } } \| f _ { 0 , \ell } \| _ { \infty } + \| \bar { f } \| _ { \infty } + \mathcal { A } _ { n } + \mathcal { U } _ { n } } \end{array}
$$

and the claim follows.

Lemma C.4. Let

$$
G _ { \ell } ( x ) : = f _ { 0 , z _ { i } } ( x ) - \sum _ { k = 1 } ^ { L } f _ { 0 , k } ( x ) \mathbb { P } ( Z = k | X = x )
$$

and $\widetilde G _ { \ell } \in \mathcal { F } _ { \ell }$ be such that $\| G _ { \ell } - \widetilde { G } _ { \ell } \| _ { \infty } \leq \sqrt { \phi _ { \ell , n } } .$ . Such $\widetilde { G } _ { \ell }$ exists by Theorem 3 in (Kohler & Langer, 2021). Then $i f$ $B _ { n } \geq \| G _ { \ell } \| _ { \infty } + \sqrt { \phi _ { \ell , n } }$ we have that $\widetilde { G } _ { \ell } = \widetilde { G } _ { \ell , B _ { n } } : = ( \widetilde { G } _ { \ell } ) _ { B _ { n } }$

Proof. Simply observe that

$$
\| \widetilde { G } _ { \ell } \| _ { \infty } \leq \| \widetilde { G } _ { \ell } - G _ { \ell } \| _ { \infty } + \| G _ { \ell } \| _ { \infty } \leq \sqrt { \phi _ { \ell , n } } + \| G _ { \ell } \| _ { \infty } \leq \mathcal B _ { n }
$$

and the claim follows.

Lemma C.5. Suppose that Assumption 3.1 holds. Then letting $n _ { \ell } = | \{ i \in [ n ] : z _ { i } = \ell \} | , f o r \ell \in \{ 1 , \dots , L \}$ the event

$$
\Omega _ { 2 } ( \ell ) : = \Big \{ \frac { n \pi _ { \ell } } { 2 } \ : \leq \ : n _ { \ell } \leq 2 n \overline { { \pi } } _ { \ell } \Big \}
$$

happens with probability at least

$$
1 - \exp \left( - \frac { n \underline { { { \pi } } } _ { \ell } ^ { 3 } } { 2 7 \overline { { { \pi } } } _ { \ell } ^ { 2 } } \right) .
$$

Proof. First, notice that $n _ { \ell }$ is Binomial $( n , p _ { \ell } )$ where

$$
p _ { \ell } = \int \mathbb { P } ( z _ { i } = \ell | x _ { i } = x ) d F _ { X } ( d x ) \in [ \underline { { \pi } } _ { \ell } , \overline { { \pi } } _ { \ell } ] ,
$$

by Assumption 3.1.

Hence, by the binomial concentration inequality, for instance see Lemma 5 in (Madrid Padilla et al., 2020), letting $\kappa _ { \ell } = \underline { { \pi } } _ { \ell } / ( 3 \overline { { \pi } } _ { \ell } ) \in [ 0 , 1 ]$

$$
\begin{array} { r c l } { \mathbb { P } ( | n p _ { \ell } - n _ { \ell } | \geq \kappa _ { \ell } n \overline { { \pi } } _ { \ell } ) } & { \leq } & { \mathbb { P } ( | n p _ { \ell } - n _ { \ell } | \geq \kappa _ { \ell } n p _ { \ell } ) } \\ & { \leq } & { \exp \left( - \frac { \pi _ { \ell } ^ { 2 } n p _ { \ell } } { 2 7 \overline { { \pi } } _ { \ell } ^ { 2 } } \right) } \\ & { \leq } & { \exp \left( - \frac { n \underline { { \pi } } _ { \ell } ^ { 3 } } { 2 7 \overline { { \pi } } _ { \ell } ^ { 2 } } \right) . } \end{array}
$$

However, $| n p \ell - n \ell | < \kappa \ell ^ { n \overline { { \pi } } \ell }$ holds if and only if

$$
n p _ { \ell } - \kappa _ { \ell } n \overline { { { \pi } } } _ { \ell } < n _ { \ell } < n p _ { \ell } + \kappa _ { \ell } n \overline { { { \pi } } } _ { \ell }
$$

which implies

$$
\frac { n \underline { { \pi } } _ { \ell } } { 2 } < n _ { \ell } < 2 n \overline { { \pi } } _ { \ell }
$$

given our choice of $\kappa _ { \ell } .$ The claim then follows.

Lemma C.6. Let be afunction class such that $| f | \leq M$ for all $f \in { \mathcal { F } } .$ . For a set $S = \{ a _ { 1 } , . . . , a _ { W } \} \subset [ 0 , 1 ] ^ { d }$ , we write

$$
\| f \| _ { S } : = \sqrt { \sum _ { w = 1 } ^ { W } f ( a _ { w } ) ^ { 2 } } .
$$

We also write

$$
\| f \| _ { \ell , \mathcal L _ { 2 } } : = \sqrt { \int f ( x ) ^ { 2 } d P _ { X | Z = \ell } ( x ) }
$$

with $P _ { X \mid Z = \ell }$ the distribution of X conditioning on $Z = \ell .$ Then for any $\delta > 0$ we have that

$$
\operatorname* { s u p } _ { S = \{ a _ { w } \} _ { w = 1 } ^ { W } \subset [ 0 , 1 ] ^ { d } : \frac { n \pi _ { \ell } } { 2 } \le W \le 2 n \overline { { \pi } } _ { \ell } } \left\{ \int _ { \delta ^ { 2 } / 4 8 } ^ { \delta } \sqrt { \frac { \log N \left( t / 4 , \mathcal { F } , \| \cdot \| _ { S } \right) } { W } } d t + \delta \sqrt { \frac { \log ( 4 8 / \delta ^ { 2 } ) } { W } } \right\} \lesssim \delta ^ { 2 } ,\tag{34}
$$

implies that the event

$$
\mathcal { E } _ { \ell } ( \mathcal { F } ) : = \left\{ \operatorname* { s u p } _ { \boldsymbol { g } \in \mathcal { F } } \left| \frac { | \| \boldsymbol { g } \| _ { \ell , n } ^ { 2 } - \| \boldsymbol { g } \| _ { \ell , \mathcal { L } _ { 2 } } ^ { 2 } | } { \frac { 1 } { 2 } \| \boldsymbol { g } \| _ { \ell , \mathcal { L } _ { 2 } } ^ { 2 } + \frac { \delta ^ { 2 } } { 2 } } \right| \leq 1 \right\}
$$

happens with probability at least

$$
\left[ 1 - c _ { 1 } \exp ( - c _ { 3 } n \underline { { { \pi } } } _ { \ell } \delta ^ { 2 } ) \right] \cdot \left[ 1 - \exp \left( - \frac { n \underline { { { \pi } } } _ { \ell } ^ { 3 } } { 2 7 \overline { { { \pi } } } _ { \ell } ^ { 2 } } \right) \right] .
$$

Proof. We notice that

$$
\begin{array} { r c l } { \mathbb { P } ( { \mathcal E } _ { \ell } ( { \mathcal F } ) ) } & { \geq } & { \displaystyle \sum _ { W : \frac { n \pi _ { \ell } } { 2 } \leq W \leq 2 n \overline { { \pi } } _ { \ell } } \mathbb { P } ( { \mathcal E } _ { \ell } ( { \mathcal F } ) | n _ { \ell } = W ) \cdot \mathbb { P } ( n _ { \ell } = W ) . } \end{array}\tag{35}
$$

Let $\{ \xi _ { i } \} _ { i = 1 } ^ { n }$ be Radamacher variables, and observe that

$$
\begin{array} { r l } & { \mathbb E (  \underset { f \in \mathrm { s t a r } ( \mathcal F ) : \| f \| _ { \ell , n } \leq \delta } { \operatorname* { s u p } } \frac { 1 } { n _ { \ell } } \sum _ { i : z _ { i } = \ell } \xi _ { i } f ( x _ { i } ) \biggm | n _ { \ell } = W ) } \\ & { = \mathbb E ( \mathbb E (  \underset { f \in \mathrm { s t a r } ( \mathcal F ) : \| f \| _ { \ell , n } \leq \delta } { \operatorname* { s u p } } \frac { 1 } { n _ { \ell } } \sum _ { i : z _ { i } = \ell } \xi _ { i } f ( x _ { i } ) \biggm | \{ x _ { i } , z _ { i } \} _ { i = 1 } ^ { n } , \ n _ { \ell } = W ) ) . } \end{array}
$$

Hence, proceeding as in the proof Lemma 6 in (Padilla et al., 2024a), based on our choice of δ, we obtain that

$$
\begin{array} { r } { \mathbb { P } ( \mathcal { E } _ { \ell } ( \mathcal { F } ) | n _ { \ell } = W ) \ge 1 - c _ { 1 } \exp ( - c _ { 2 } W \delta ^ { 2 } ) } \end{array}
$$

for some constants $c _ { 1 } , c _ { 2 } > 0$

Therefore, from (35), for some constant $c _ { 3 } > 0$ , we have that

$$
\begin{array} { r c l } { { \mathbb { P } ( \mathcal { E } _ { \ell } ( \mathcal { F } ) ) } } & { { \geq } } & { { [ 1 - c _ { 1 } \exp ( - c _ { 3 } n \underline { { \pi } } _ { \ell } \delta ^ { 2 } ) ] \cdot { \mathbb { P } } ( \frac { n \underline { { \pi } } _ { \ell } } { 2 } \leq W \leq 2 n \overline { { \pi } } _ { \ell } ) } } \\ { { } } & { { \geq } } & { { [ 1 - c _ { 1 } \exp ( - c _ { 3 } n \underline { { \pi } } _ { \ell } \delta ^ { 2 } ) ] \cdot \left[ 1 - \exp \left( - \frac { n \underline { { \pi } } _ { \ell } ^ { 3 } } { 2 7 \overline { { \pi } } _ { \ell } ^ { 2 } } \right) \right] } } \end{array}
$$

where the last inequality follows from Lemma C.5. The claim then follows.

Lemma C.7. Suppose that Assumption 3.1 holds. There exists constants $c _ { 1 } , c _ { 2 } > 0$ such that

$$
c _ { 1 } \| f \| _ { \ell , \angle _ { 2 } } ^ { 2 } \leq \| f \| _ { \mathcal { L } _ { 2 } } ^ { 2 } \leq c _ { 2 } \| f \| _ { \ell , \mathcal { L } _ { 2 } } ^ { 2 }
$$

for any measurable function f with $\| f \| _ { \mathcal { L } _ { 2 } } ^ { 2 } < \infty$

Proof. By Assumption 3.1, there exist $0 < \underline { { \pi } } _ { \ell } \le \overline { { \pi } } _ { \ell } < 1$ such that

$$
\underline { { \pi } } _ { \ell } \leq \mathbb { P } ( Z = \ell \vert X = x ) \leq \overline { { \pi } } \ell
$$

for all x. Hence

$$
\underline { { \pi } } _ { \ell } \leq \mathbb { P } ( Z = \ell ) = \int \mathbb { P } ( Z = \ell \mid X = x ) d P _ { X } ( x ) \leq \overline { { \pi } } _ { \ell } .
$$

Moreover, consider the Radon–Nikodym derivative

$$
{ \frac { d P _ { X | Z = \ell } } { d P _ { X } } } ( x ) = { \frac { \mathbb { P } ( Z = \ell | X = x ) } { \mathbb { P } ( Z = \ell ) } } .
$$

Therefore,

$$
\frac { \pi _ { \ell } } { \overline { { \pi } } _ { \ell } } \leq \frac { d P _ { X | Z = \ell } } { d P _ { X } } ( x ) \leq \frac { \overline { { \pi } } _ { \ell } } { \underline { { \pi } } _ { \ell } } .
$$

Thus, for any nonnegative measurable function $^ { g , }$

$$
\frac { \pi _ { \ell } } { \overline { { { \pi } } } _ { \ell } } \int g ( x ) d P _ { X } ( x ) \leq \int g ( x ) d P _ { X | Z = \ell } ( x ) \leq \frac { \overline { { { \pi } } } _ { \ell } } { \underline { { { \pi } } } _ { \ell } } \int g ( x ) d P _ { X } ( x ) .
$$

Equivalently,

$$
\frac { \pi _ { \ell } } { \overline { { \pi } } _ { \ell } } \int g ( x ) d P _ { X | Z = \ell } ( x ) \leq \int g ( x ) d P _ { X } ( x ) \leq \frac { \overline { { \pi } } _ { \ell } } { \underline { { \pi } } _ { \ell } } \int g ( x ) d P _ { X | Z = \ell } ( x ) .
$$

Taking $g ( x ) = f ^ { 2 } ( x )$ , we obtain

$$
\frac { \pi _ { \ell } } { \overline { { \pi } } _ { \ell } } \| f \| _ { \ell , \mathcal { L } _ { 2 } } ^ { 2 } \leq \| f \| _ { \mathcal { L } _ { 2 } } ^ { 2 } \leq \frac { \overline { { \pi } } _ { \ell } } { \underline { { \pi } } _ { \ell } } \| f \| _ { \ell , \mathcal { L } _ { 2 } } ^ { 2 } .
$$

Hence the result holds with

$$
c _ { 1 } = \frac { \underline { { \pi } } _ { \ell } } { \overline { { \pi } } _ { \ell } } , \qquad c _ { 2 } = \frac { \overline { { \pi } } _ { \ell } } { \underline { { \pi } } _ { \ell } } .
$$

## C.3. Proof of Theorem B.2

Proof. For a fixed $\ell \in \{ 1 , \ldots , L \}$ let ${ \mathcal { T } } _ { \ell } : = \{ i \in [ n ] : z _ { i } = \ell \}$ and $n _ { \ell } : = | \mathcal { T } _ { \ell } |$ , and we condition on z being in the set

$$
\{ z \in \{ 1 , \ldots , L \} ^ { n } : { \frac { n \pi _ { \ell } } { 2 } } \leq | \{ i : z _ { i } = \ell \} | \leq 2 n \overline { { \pi } } _ { \ell } \} .\tag{36}
$$

By Lemma C.5 this happens with probability at least

$$
1 - \exp \left( - \frac { n \underline { { { \pi } } } _ { \ell } ^ { 3 } } { 2 7 \overline { { { \pi } } } _ { \ell } ^ { 2 } } \right) .
$$

Let $\delta > 0$ to be defined later. Suppose that the event $\Omega _ { 1 }$ defined in Lemma C.3 holds. Then, given our choice of $B _ { n }$ , from Lemmas C.3 and C.4 we have $\widetilde { G } _ { \ell } = \widetilde { G } _ { \ell , B _ { \ i } }$ and $\| \tilde { y } \| _ { \infty } \leq B _ { n }$ . Hence,

$$
\frac { 1 } { n _ { \ell } } \sum _ { i \in \mathcal { I } _ { \ell } } ( \tilde { y } _ { i } - \hat { f } _ { \ell , \mathcal { B } _ { n } } ( x _ { i } ) ) ^ { 2 } \leq \frac { 1 } { n _ { \ell } } \sum _ { i \in \mathcal { I } _ { \ell } } ( \tilde { y } _ { i } - \hat { f } _ { \ell } ( x _ { i } ) ) ^ { 2 }
$$

where $\hat { f } _ { l , B _ { n } } = ( \hat { f } _ { l } ) _ { B _ { n } }$ . Also, by the basic inequality,

$$
\frac { 1 } { n _ { \ell } } \sum _ { i \in \mathbb { Z } _ { \ell } } ( \tilde { y } _ { i } - \hat { f } _ { \ell } ( x _ { i } ) ) ^ { 2 } \le \frac { 1 } { n _ { \ell } } \sum _ { i \in \mathbb { Z } _ { \ell } } ( \tilde { y } _ { i } - \widetilde { G } _ { \ell } ( x _ { i } ) ) ^ { 2 } = \frac { 1 } { n _ { \ell } } \sum _ { i \in \mathbb { Z } _ { \ell } } ( \tilde { y } _ { i } - \widetilde { G } _ { \ell , \mathcal { B } _ { n } } ( x _ { i } ) ) ^ { 2 } .
$$

As a result,

$$
\frac { 1 } { n _ { \ell } } \sum _ { i \in \mathcal { I } _ { \ell } } ( \widetilde { y } _ { i } - \widehat { f } _ { \ell , \mathcal { B } _ { n } } ( x _ { i } ) ) ^ { 2 } \ : \leq \ : \frac { 1 } { n _ { \ell } } \sum _ { \in \mathcal { I } _ { \ell } } ( \widetilde { y } _ { i } - \widetilde { G } _ { \ell , \mathcal { B } _ { n } } ( x _ { i } ) ) ^ { 2 } .
$$

However, by Lemma C.2 and the notation of Lemma $\Sigma . 4 , \tilde { y } _ { i } = G _ { \ell } ( x _ { i } ) + ( \bar { f } ( x _ { i } ) - \hat { f } _ { A _ { n } } ( x _ { i } ) ) + \epsilon _ { i }$ for i with $z _ { i } = \ell ,$ then

$$
\frac { 1 } { n _ { \ell } } \sum _ { i \in \mathbb { Z } _ { \ell } } [ G _ { \ell } ( x _ { i } ) - \hat { f } _ { \ell , B _ { n } } ( x _ { i } ) + ( \bar { f } ( x _ { i } ) - \hat { f } _ { A _ { n } } ( x _ { i } ) ) + \epsilon _ { i } ] ^ { 2 } \leq \frac { 1 } { n _ { \ell } } \sum _ { i \in \mathbb { Z } _ { \ell } } [ G _ { \ell } ( x _ { i } ) - \widetilde { G } _ { \ell , B _ { n } } + ( \bar { f } ( x _ { i } ) - \hat { f } _ { A _ { n } } ( x _ { i } ) ) + \epsilon _ { i } ] ^ { 2 }
$$

which implies

$$
\begin{array} { r c l } { \displaystyle \frac { 1 } { n \varepsilon } \sum _ { i \in \mathcal { Z } _ { \varepsilon } } ( G _ { \varepsilon } ( x _ { i } ) - \hat { f } _ { \varepsilon , B _ { n } } ( x _ { i } ) ) ^ { 2 } } & { \leq } & { \displaystyle \frac { 1 } { n \varepsilon } \sum _ { i \in \mathcal { T } _ { \varepsilon } } ( G _ { \varepsilon } ( x _ { i } ) - \widetilde { G } _ { \varepsilon , B _ { n } } ( x _ { i } ) ) ^ { 2 } + \displaystyle \frac { 2 } { n \varepsilon } \sum _ { i \in \mathcal { Z } _ { \varepsilon } } \epsilon _ { i } ( \hat { f } _ { \varepsilon , B _ { n } } ( x _ { i } ) - \widetilde { G } _ { \varepsilon , B _ { n } } ( x _ { i } ) ) + } \\ & & { \displaystyle \frac { 2 } { n \varepsilon } \sum _ { i \in \mathcal { Z } _ { \varepsilon } } ( \hat { f } _ { \varepsilon , B _ { n } } ( x _ { i } ) - \widetilde { G } _ { \varepsilon , B _ { n } } ( x _ { i } ) ) ( \widetilde { f } ( x _ { i } ) - \hat { f } _ { A _ { n } } ( x _ { i } ) ) } \\ & { \leq } & { \displaystyle \| \widetilde { G } _ { \varepsilon } - G _ { \varepsilon } \| _ { \infty } ^ { 2 } + \frac { 2 } { n \varepsilon } \sum _ { i \in \mathcal { Z } _ { \varepsilon } } \epsilon _ { i } ( \hat { f } _ { \varepsilon , B _ { n } } ( x _ { i } ) - \widetilde { G } _ { \varepsilon , B _ { n } } ( x _ { i } ) ) + } \\ & & { \displaystyle \frac { 2 } { n \varepsilon } \sum _ { i \in \mathcal { T } _ { \varepsilon } } ( \hat { f } _ { \varepsilon , B _ { n } } ( x _ { i } ) - \widetilde { G } _ { \varepsilon , B _ { n } } ( x _ { i } ) ) ( \widetilde { f } ( x _ { i } ) - \hat { f } _ { A _ { n } } ( x _ { i } ) ) } \end{array}
$$

$$
\begin{array} { r l } { \leq } & { \phi _ { \ell , n } + \displaystyle \frac { 2 } { n \ell } \sum _ { i \in \mathcal { Z } _ { \ell } } \mathsf { c } _ { i } ( \hat { f } _ { \ell , B _ { n } } ( x _ { i } ) - \widetilde { G } _ { \ell , B _ { n } } ( x _ { i } ) ) + } \\ & { \displaystyle \frac { 2 } { n \ell } \left[ \displaystyle \frac { 1 } { 3 2 } \sum _ { i \in \mathcal { Z } _ { \ell } } \left( \hat { f } _ { \ell , B _ { n } } ( x _ { i } ) - \widetilde { G } _ { \ell , B _ { n } } ( x _ { i } ) \right) ^ { 2 } + 8 \sum _ { i \in \mathcal { Z } _ { \ell } } \left( \widetilde { f } ( x _ { i } ) - \hat { f } _ { A _ { n } } ( x _ { i } ) \right) ^ { 2 } \right] } \\ { \leq } & { \phi _ { \ell , n } + \displaystyle \frac { 1 } { n } \sum _ { i \in \mathcal { Z } _ { \ell } } \sum _ { s = 1 } \mathsf { c } _ { i } ( \hat { f } _ { \ell , B _ { n } } ( x _ { i } ) - \widetilde { G } _ { \ell , B _ { n } } ( x _ { i } ) ) + \displaystyle \frac { 1 } { 8 n \ell } \sum _ { i \in \mathcal { Z } _ { \ell } } \left( G _ { \ell } ( x _ { i } ) - \hat { f } _ { \ell , B _ { n } } ( x _ { i } ) \right) ^ { 2 } } \\ & { \displaystyle \frac { 1 } { 8 n \ell } \sum _ { i \in \mathcal { Z } _ { \ell } } \left( G _ { \ell } ( x _ { i } ) - \widetilde { G } _ { \ell } ( x _ { i } ) \right) ^ { 2 } + \displaystyle \frac { 1 6 } { n \ell } \sum _ { i \in \mathcal { Z } _ { \ell } } \left( \widetilde { f } ( x _ { i } ) - \hat { f } _ { A _ { n } } ( x _ { i } ) \right) ^ { 2 } } \\ { \leq } &  \displaystyle \frac { 9 \phi _ { \ell , n } } { 8 } + \frac { 2 } { n \ell } \sum _  i \in \mathcal { Z } _  \ell \end{array}
$$

Therefore,

$$
\begin{array} { r c l } { \displaystyle \frac { 7 } { 8 n _ { \ell } } \sum _ { i \in \mathbb { Z } _ { \ell } } ( G _ { \ell } ( x _ { i } ) - \widehat f _ { \ell } , B _ { n } ( x _ { i } ) ) ^ { 2 } } & { \le } & { \displaystyle \frac { 9 \phi _ { \ell , n } } { 8 } + \frac { 2 } { n _ { \ell } } \sum _ { i \in \mathbb { Z } _ { \ell } } \epsilon _ { i } ( \widehat f _ { \ell , B _ { n } } ( x _ { i } ) - \widetilde { G } _ { \ell , B _ { n } } ( x _ { i } ) ) + \frac { 1 6 } { n _ { \ell } } \sum _ { i \in \mathbb { Z } _ { \ell } } ( \bar { f } ( x _ { i } ) - \widehat f _ { A _ { n } } ( x _ { i } ) ) ^ { 2 } . } \end{array}\tag{37}
$$

Next let $\mathcal { H } _ { \ell } : = \{ ( f - g ) / 2 : f , g \in \mathcal { F } _ { \ell , B _ { n } } \}$ . Then $f \in \mathcal H _ { \ell }$ implies $\| f \| _ { \infty } \leq 1$ and

$$
\begin{array} { r } { \log N ( \delta , \mathcal { H } _ { \ell } , \lVert \cdot \rVert _ { \boldsymbol { \mathcal { Z } } _ { \ell } } ) \leq 2 \log N ( \delta / 2 , \mathcal { F } _ { \ell , \mathcal { B } _ { n } } , \lVert \cdot \rVert _ { \boldsymbol { \mathcal { Z } } _ { \ell } } ) \leq 2 \eta _ { \ell , n } ( \delta / 2 ) , } \end{array}\tag{38}
$$

where the second inequality follows since we are conditioning on z satisfiying (36) and due to our assumption that (17) holds.

Hence, by Lemma 4 from (Padilla et al., 2024a), for some positive constant $C > 0$

$$
\begin{array} { r c l } { \displaystyle \frac { 2 \mathcal { B } _ { n } } { n _ { \ell } } \sum _ { i \in \mathcal { T } _ { \ell } } \frac { \big ( \widehat f _ { \ell , B _ { n } } ( x _ { i } ) - \widetilde { G } _ { \ell , B _ { n } } ( x _ { i } ) \big ) } { 2 B _ { n } } \epsilon _ { i } } & { \leq } & { 2 C \mathcal { B } _ { n } \mathcal { U } _ { n } \| ( \widehat f _ { \ell , B _ { n } } - \widetilde { G } _ { \ell , B _ { n } } ) / ( 2 B _ { n } ) \| _ { \mathcal { U } _ { \ell } } . } \\ & & { \sqrt { \eta _ { \ell , n } ( \| ( \widehat f _ { \ell , B _ { n } } - \widetilde { G } _ { \ell , B _ { n } } ) / ( 4 B _ { n } ) \| _ { \mathcal { L } _ { \ell } } ) / n _ { \ell } } } \\ & { \leq } & { C \mathcal { U } _ { n } \| \widehat f _ { \ell , B _ { n } } - \widetilde { G } _ { \ell , B _ { n } } \| _ { \mathcal { U } _ { \ell } } \cdot \sqrt { \eta _ { \ell , n } ( \| ( \widehat f _ { \ell , B _ { n } } - \widetilde { G } _ { \ell , B _ { n } } ) / ( 4 B _ { n } ) \| _ { \mathcal { L } _ { \ell } } ) / n _ { \ell } } } \end{array}\tag{39}
$$

with probability at least

$$
1 - 4 \sum _ { k = 0 } ^ { \infty } \sum _ { k ^ { \prime } = 1 } ^ { \infty } \exp \Big ( { - C _ { 1 } \eta _ { \ell , n } \big ( 2 ^ { - k - k ^ { \prime } - 1 } \big ) } \Big ) - 4 \sum _ { k = 0 } ^ { \infty } \exp \big ( { - C _ { 2 } \eta _ { \ell , n } \big ( 2 ^ { - k - 1 } \big ) } \big ) - \mathbb { P } \big ( \vert \vert \epsilon \| _ { \infty } > \mathcal { U } _ { n } \big ) .
$$

Define $\Omega _ { 3 }$ , the event such that (39) holds and let $\delta > 0$ . From now on suppose that $\Omega _ { 1 } \cap \Omega _ { 3 }$ holds. If

$$
\| ( \hat { f } _ { \ell , \mathcal { B } _ { n } } - \widetilde { G } _ { \ell , \mathcal { B } _ { n } } ) / ( 4 \mathcal { B } _ { n } ) \| _ { \mathcal { T } _ { \ell } } \leq \delta
$$

we obtain

$$
\begin{array} { r } { \| \big ( \widehat f _ { \ell , \mathcal { B } _ { n } } - \widetilde G _ { \ell , \mathcal { B } _ { n } } \big ) \| _ { \mathcal { T } _ { \ell } } ^ { 2 } \leq 1 6 \mathcal { B } _ { n } ^ { 2 } \delta ^ { 2 } , } \end{array}
$$

which implies

$$
\| \hat { f } _ { \ell ,  { \mathcal { B } } _ { n } } - G _ { \ell } \| _ { \mathcal { T } _ { \ell } } ^ { 2 } \leq 3 2  { \mathcal { B } } _ { n } ^ { 2 } \delta ^ { 2 } + 2 \phi _ { \ell , n } ,
$$

Suppose now that

$$
\| ( \hat { f } _ { \ell , \mathcal { B } _ { n } } - \widetilde { G } _ { \ell , \mathcal { B } _ { n } } ) / ( 4 \mathcal { B } _ { n } ) \| _ { \mathcal { T } _ { \ell } } > \delta .
$$

Then (37) and (39) imply

$$
\begin{array} { r c l } { \displaystyle \frac { 7 } { 8 m _ { \ell } } \sum _ { i \in \mathbb { Z } _ { \ell } } ( G _ { \ell } ( x _ { i } ) - \widehat f _ { \ell , \mathbb { B } _ { n } } ( x _ { i } ) ) ^ { 2 } } & { \leq } & { \displaystyle \frac { 9 \phi _ { \ell , n } } { 8 } + \frac { 1 6 } { m _ { \ell } } \sum _ { i \in \mathbb { Z } _ { \ell } } ( \widehat f ( x _ { i } ) - \widehat f _ { \mathcal { A } _ { n } } ( x _ { i } ) ) ^ { 2 } } \\ & & { \displaystyle + 2 G \mathcal { A } _ { n } \lVert \widehat f _ { \ell , \mathbb { B } _ { n } } - \widehat G _ { \ell , \mathbb { B } _ { n } } \rVert _ { \mathbb { Z } _ { \ell } } \cdot \sqrt { \eta _ { \ell , n } ( \delta ) / n _ { \ell } } } \\ & { \leq } & { \displaystyle \frac { 9 \phi _ { \ell , n } } { 8 } + \frac { 1 6 } { m _ { \ell } } \sum _ { i \in \mathbb { Z } _ { \ell } } ( \widehat f ( x _ { i } ) - \widehat f _ { \mathcal { A } _ { n } } ( x _ { i } ) ) ^ { 2 } + } \\ & & { \displaystyle \frac { \lVert \widehat f _ { \ell , \mathbb { B } _ { n } } - \widehat G _ { \ell , \mathbb { B } _ { n } } \rVert _ { \mathbb { Z } _ { \ell } } ^ { 2 } } { 4 } + \frac { 4 C ^ { 2 } M _ { n } ^ { 2 } \eta _ { \ell , n } ( \delta ) } { n _ { \ell } } } \\ & { \leq } & { \displaystyle \frac { 9 \phi _ { \ell , n } } { 8 } + \frac { 1 6 } { m _ { \ell } } \sum _ { i \in \mathbb { Z } _ { \ell } } ( \widehat f ( x _ { i } ) - \widehat f _ { \mathcal { A } _ { n } } ( x _ { i } ) ) ^ { 2 } + } \\ & &  \displaystyle \frac { \lVert \widehat f _ { \ell , \mathbb { B } _ { n } } - G _ { \ell } \rVert _ { \mathbb { Z } _ { \ell } } ^ { 2 } } { 2 } + \frac  \end{array}\tag{40}
$$

Hence,

$$
\begin{array} { l c l } { \displaystyle \frac { 3 } { 8 } \| G _ { \ell } - \hat { f } _ { \ell , \mathcal { B } _ { n } } \| _ { \mathcal { L } _ { \ell } } ^ { 2 } } & { \leq } & { \displaystyle \frac { 1 7 \phi _ { \ell , n } } { 8 } + 1 6 \| \bar { f } - \hat { f } _ { \mathcal { A } _ { n } } \| _ { \mathcal { L } _ { \ell } } ^ { 2 } + \frac { 4 C ^ { 2 } \mathcal { U } _ { n } ^ { 2 } \eta _ { \ell , n } ( \delta ) } { n _ { \ell } } } \\ & { \leq } & { \displaystyle \frac { 1 7 \phi _ { \ell , n } } { 8 } + 3 2 \| \tilde { f } _ { A _ { n } } - \hat { f } _ { \mathcal { A } _ { n } } \| _ { \mathcal { L } _ { \ell } } ^ { 2 } + 3 2 \| \bar { f } - \tilde { f } \| _ { \mathcal { L } _ { \ell } } ^ { 2 } + \frac { 4 C ^ { 2 } \mathcal { U } _ { n } ^ { 2 } \eta _ { \ell , n } ( \delta ) } { n _ { \ell } } } \\ & { \leq } & { \displaystyle \frac { 1 7 \phi _ { \ell , n } } { 8 } + 3 2 \| \tilde { f } _ { A _ { n } } - \hat { f } _ { \mathcal { A } _ { n } } \| _ { \mathcal { L } _ { \ell } } ^ { 2 } + 3 2 \phi _ { n } + \frac { 4 C ^ { 2 } \mathcal { U } _ { n } ^ { 2 } \eta _ { \ell , n } ( \delta ) } { n _ { \ell } } } \end{array}\tag{41}
$$

with $\tilde { f }$ and $\phi _ { n }$ as in the notation of Theorem B.1.

Moreover, by choosing δ to satisfy

$$
\operatorname* { m a x } _ { \substack { m : \frac { n \pi } { 2 } \leq m } } \operatorname* { s u p } _ { \substack { \zeta \left\{ x \right\} _ { j } ^ { 1 } \zeta _ { 2 } \Vert ^ { \infty } \Vert \mathcal { C } \Vert 0 , 1 \Vert ^ { d } } } \left\{ \frac { 1 } { \sqrt { m } } \int _ { \delta ^ { 2 } / 4 8 } ^ { \delta } \sqrt { \log N \left( t / 4 , \mathcal { F } _ { \ell , B _ { n } } , \Vert \cdot \Vert _ { m } \right) } d t + \frac { \delta } { \sqrt { m } } \sqrt { c _ { 1 } \log ( 4 8 / \delta ^ { 2 } ) } \right\} \lesssim \delta ^ { 2 } ,\tag{42}
$$

Lemma C.6 implies that the event

$$
\Omega _ { 4 } : = \left\{ \operatorname* { s u p } _ { h \in \mathcal { H } _ { \ell } } \left| \frac { | \| h \| _ { \mathcal { L } _ { \ell } } ^ { 2 } - \| h \| _ { \ell , \mathcal { L } _ { 2 } } ^ { 2 } | } { \frac { 1 } { 2 } \| h \| _ { \ell , \mathcal { L } _ { 2 } } ^ { 2 } + \frac { \delta ^ { 2 } } { 2 } } \right| \leq 1 \right\}
$$

holds with probability at least

$$
\left[ 1 - c _ { 2 } \exp ( - c _ { 1 } \delta ^ { 2 } n _ { \underline { { { \pi } } } \ell } ) \right] \cdot \left[ 1 - \exp \left( - \frac { n _ { \underline { { { \pi } } } } 3 } { 2 7 \overline { { { \pi } } } _ { \ell } ^ { 2 } } \right) \right]
$$

for constants $c _ { 1 } , c _ { 2 } > 0 .$

Therefore, from (41) and Lemma C.7, in the event $\Omega _ { 1 } \cap \Omega _ { 3 } \cap \Omega _ { 4 }$ , we obtain that

$$
\begin{array} { r c l } { \| G _ { \ell } - \hat { f } _ { \ell , B _ { n } } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } } & { \lesssim } & { 2 \| G _ { \ell } - \widetilde { G } _ { \ell } \| _ { \ell , \mathcal { L } _ { 2 } } ^ { 2 } + 2 \| \widetilde { G } _ { \ell } - \hat { f } _ { \ell , B _ { n } } \| _ { \ell , \mathcal { L } _ { 2 } } ^ { 2 } } \\ & { \lesssim } & { 2 \| G _ { \ell } - \widetilde { G } _ { \ell } \| _ { \infty } ^ { 2 } + 2 \| 2 \| \widetilde { G } _ { \ell } - \hat { f } _ { \ell , B _ { n } } \| _ { \mathcal { L } _ { \ell } } ^ { 2 } + \delta ^ { 2 } \| } \\ & { \leq } & { 2 \| G _ { \ell } - \widetilde { G } _ { \ell } \| _ { \infty } ^ { 2 } + \frac { 3 2 } { 3 } \big [ \frac { 1 7 \phi _ { \ell , n } } { 8 } + 3 2 \| \tilde { f } _ { A _ { n } } - \hat { f } _ { A _ { n } } \| _ { \mathcal { L } _ { \ell } } ^ { 2 } + 3 2 \phi _ { n } + \frac { 4 C ^ { 2 } \mathcal { U } _ { n } ^ { 2 } \eta _ { \ell , n } ( \delta ) } { n _ { \ell } } \big ] + 2 \delta ^ { 2 } } \\ & { \lesssim } & { \phi _ { \ell , n } + \| \tilde { f } _ { A _ { n } } - \hat { f } _ { A _ { n } } \| _ { \mathcal { L } _ { \ell } } ^ { 2 } + \phi _ { n } + \frac { \mathcal { U } _ { n } ^ { 2 } \eta _ { \ell , n } ( \delta ) } { n _ { \ell } } + \delta ^ { 2 } . } \\ & { \lesssim } &  \phi _ { \ell , n } + \| \tilde { f } _ { A _ { n } } - \hat { f } _ { A _ { n } } \| _ { \mathcal { L } _ { \ell } } ^ { 2 } + \phi _ { n } + \frac  \mathcal { U } _ { n } ^ { 2 } \eta \end{array}\tag{43}
$$

Furthermore, let $\mathcal { F } _ { A _ { n } } : = \{ f _ { A _ { n } } / ( 2 A _ { n } ) : f \in \mathcal { F } \}$ and

$$
\begin{array} { r } { \mathcal { H } : = \{ f - g : f , g \in \mathcal { F } _ { A _ { n } } \} . } \end{array}
$$

Then, by choosing δ to satisfy

$$
\operatorname* { m a x } _ { \substack { m : \frac { n \pi } { 2 } \leq m } } \operatorname* { s u p } _ { \varepsilon \downarrow \gamma _ { j = 1 } ^ { m } \subseteq [ 0 , 1 ] ^ { d } } \left\{ \frac { 1 } { \sqrt { m } } \int _ { \delta ^ { 2 } / 4 8 } ^ { \delta } \sqrt { \log N \left( t / 4 , \mathcal { F } _ { A _ { n } } , \Vert \cdot \Vert _ { m } \right) } d t + \frac { \delta } { \sqrt { m } } \sqrt { c _ { 1 } \log ( 4 8 / \delta ^ { 2 } ) } \right\} \lesssim \delta ^ { 2 } ,\tag{44}
$$

Lemma C.7 also implies that the event

$$
\Omega _ { 5 } : = \left\{ \operatorname* { s u p } _ { h \in \mathcal { H } } \left| \frac { | \| h \| _ { \mathcal { T } _ { \ell } } ^ { 2 } - \| h \| _ { \ell , \mathcal { L } _ { 2 } } ^ { 2 } | } { \frac { 1 } { 2 } \| h \| _ { \ell , \mathcal { L } _ { 2 } } ^ { 2 } + \frac { \delta ^ { 2 } } { 2 } } \right| \leq 1 \right\}
$$

holds with probability at least

$$
\left[ 1 - c _ { 2 } \exp ( - c _ { 1 } \delta ^ { 2 } n \pi _ { \ell } ) \right] \cdot \left[ 1 - \exp \left( - \frac { n \underline { { { \pi } } } _ { \ell } ^ { 3 } } { 2 7 \overline { { { \pi } } } _ { \ell } ^ { 2 } } \right) \right] .
$$

As a result, from (43), in the event $\Omega _ { 1 } \cap \Omega _ { 3 } \cap \Omega _ { 4 } \cap \Omega _ { 5 }$ , we have that

$$
\begin{array} { r c l } { \| G _ { \ell } - \hat { f } _ { \ell , \mathcal { B } _ { n } } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } } & { \lesssim } & { \phi _ { \ell , n } + \| \tilde { f } _ { A _ { n } } - \hat { f } _ { A _ { n } } \| _ { \mathcal { L } _ { \ell } } ^ { 2 } + \phi _ { n } + \frac { \mathcal { U } _ { n } ^ { 2 } \eta _ { \ell , n } ( \delta ) } { n _ { \ell } } + \delta ^ { 2 } } \\ & { \lesssim } & { \phi _ { \ell , n } + \| \tilde { f } _ { A _ { n } } - \hat { f } _ { A _ { n } } \| _ { \ell , \mathcal { L } _ { 2 } } ^ { 2 } + \phi _ { n } + \frac { \mathcal { U } _ { n } ^ { 2 } \eta _ { \ell , n } ( \delta ) } { n _ { \ell } } + \delta ^ { 2 } } \\ & { \lesssim } & { \phi _ { \ell , n } + \| \tilde { f } _ { A _ { n } } - \hat { f } _ { A _ { n } } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } + \phi _ { n } + \frac { \mathcal { U } _ { n } ^ { 2 } \eta _ { \ell , n } ( \delta ) } { n _ { \ell } } + \delta ^ { 2 } } \\ & { \lesssim } & { \phi _ { \ell , n } + r _ { n } + \frac { \mathcal { U } _ { n } ^ { 2 } \eta _ { \ell , n } ( \delta ) } { n _ { \pi _ { \ell } } } + \delta ^ { 2 } . } \end{array}\tag{45}
$$

Finally, the claim follows from the inequality

$$
\begin{array} { r } { \| g _ { 0 , \ell } - \hat { g } _ { \ell } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } \lesssim \| G _ { \ell } - \hat { f } _ { \ell , \mathcal { B } _ { n } } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } + \| \bar { f } - \hat { f } _ { A _ { n } } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } . } \end{array}
$$

## C.4. Proof of Theorem B.4

We apply Theorem B.2 and proceed as in the proof of Theorem 2 in (Padilla et al., 2024a). Specifically, let $\begin{array} { r } { \mathcal { F } _ { \mathcal { B } _ { n } } ( M _ { \ell } , \nu _ { \ell } ) : = \ d } \end{array}$ $\{ f _ { \mathcal { B } _ { n } } : f \in \mathcal { F } ( M _ { \ell } , \nu _ { \ell } ) \}$ . Then, by Theorem 6 from (Bartlett et al., 2019), it holds that the VC dimension of $\mathcal { C } ( M _ { \ell } , \nu _ { \ell } )$ satisfies

$$
\mathrm { V C } ( \mathcal { F } ( M _ { \ell } , \nu _ { \ell } ) ) \lesssim M _ { \ell } ^ { 2 } \nu _ { \ell } ^ { 2 } \log ( M _ { \ell } \nu _ { \ell } ) .
$$

As a result, by Lemma 9.2 and Theorem 9.4 from (Gyorfi et al. ¨ , 2002), for any $\delta \in ( 0 , 1 )$ and for any $\{ x _ { j } \} _ { j = 1 } ^ { m }$ with induced norm $\| \cdot \| _ { m }$ , it holds that

$$
\begin{array} { l l l } { \log N ( \delta , \mathcal { F } _ { B _ { n } } ( M _ { \ell } , \nu _ { \ell } ) , \| \cdot \| _ { m } ) } & { \lesssim } & { M ^ { 2 } \nu ^ { 2 } \log ( M _ { \ell } \nu _ { \ell } ) \cdot \big [ \log ( \mathcal { B } _ { n } ^ { 2 } \delta ^ { - 2 } ) + \log \log ( \mathcal { B } _ { n } ^ { 2 } \delta ^ { - 2 } ) \big ] } \\ & { \lesssim } & { M _ { \ell } ^ { 2 } \nu _ { \ell } ^ { 2 } \log ( M _ { \ell } \nu _ { \ell } ) \log ( \mathcal { B } _ { n } ^ { 2 } \delta ^ { - 1 } ) } \\ & { \leq } & { C _ { 0 } ( n \underline { { \pi } } _ { \ell } ) \phi _ { \ell , n } \log ^ { 3 } ( n \underline { { \pi } } _ { \ell } ) \log ( \mathcal { B } _ { n } ^ { 2 } \delta ^ { - 1 } ) } \end{array}\tag{46}
$$

where $C _ { 0 } > 0$ is a constant, and the last inequality holds from our choice of $M _ { \ell }$ and $\nu _ { \ell } .$ . Here, $m \in \mathbb { N }$ is arbitray.

Therefore, with $\mathcal { F } _ { \ell } : = \mathcal { F } ( M _ { \ell } , \nu _ { \ell } )$ , we have that

$$
\operatorname* { m a x } _ { \stackrel { n \pi _ { \ell } } { 2 } \leq n _ { \ell } \leq 2 n \pi _ { \ell } } \operatorname* { s u p } _ { \{ x _ { i } \} _ { i \in \mathcal { I } _ { \ell } } \subset [ 0 , 1 ] ^ { d } } \log N ( \delta , \mathcal { F } _ { \ell , \mathcal { B } _ { n } } , \Vert \cdot \Vert _ { \mathcal { I } _ { \ell } } ) \leq \eta _ { \ell , n } ( \delta ) \ \forall \delta \in ( 0 , 1 ) ,
$$

for

$$
\eta _ { \ell , n } ( \delta ) : = C _ { 0 } ( n \underline { { \pi } } _ { \ell } ) \phi _ { \ell , n } \log ^ { 3 } ( n \underline { { \pi } } _ { \ell } ) \log ( \mathcal { B } _ { n } ^ { 2 } \delta ^ { - 1 } ) .\tag{47}
$$

Thus, (17) holds with $\eta _ { \ell , n } ( \delta )$ as in (47).

Furthermore, for any positive constants $C _ { 1 }$ and $C _ { 2 }$ , we have that

$$
\begin{array} { r l } & { \displaystyle \sum _ { l = 0 } ^ { \infty } \displaystyle \sum _ { \gamma = 1 } ^ { \infty } \exp ( - C _ { 1 } \eta _ { \ell , n } ( 2 ^ { - l - l ^ { \prime } - 1 } ) ) + \displaystyle \sum _ { l = 0 } ^ { \infty } \exp ( - C _ { 2 } \eta _ { \ell , n } ( 2 ^ { - l - 1 } ) ) } \\ & { = \displaystyle \sum _ { l = 0 } ^ { \infty } \sum _ { \gamma = 1 } ^ { \infty } \exp ( - C _ { 1 } C _ { 0 } ( l + l ^ { \prime } + 1 ) \pi _ { \alpha , \ell } \phi _ { \ell , n } \log ^ { 3 } ( n \pi _ { \ell } ) \log ( 2 B _ { n } ^ { 2 } ) ) } \\ & { \displaystyle + \sum _ { l = 0 } ^ { \infty } \exp ( - C _ { 2 } C _ { 0 } \pi \underline { { \pi } } _ { \ell } \phi _ { \ell , n } \log ^ { 3 } ( n \pi _ { \ell } ) \log ( 2 B _ { n } ^ { 2 } ) ) } \\ & { = [ \exp ( - C _ { 1 } C _ { 0 } \pi \underline { { \pi } } _ { \ell } \phi _ { \ell , n } \log ^ { 3 } ( n \pi _ { \ell } ) \log ( 2 B _ { n } ^ { 2 } ) ) / [ 1 - \exp ( - C _ { 1 } C _ { 0 } n \pi _ { \underline { { \ell } } } \phi _ { \ell , n } \log ^ { 3 } ( n \pi _ { \ell } ) \log ( 2 B _ { n } ^ { 2 } ) ) ] ] ^ { 2 } + } \\ &  \displaystyle \exp ( - C _ { 2 } C _ { 0 } n \pi _ { \underline { { \ell } } } \phi _ { \ell , n } \log ^ { 3 } ( n \pi _ { \underline { { \ell } } } ) \log ( 2 B _ { n } ^ { 2 } ) ) / [ 1 - \exp ( - C _ { 2 } C _ { 0 } n \pi _ { \underline { { \ell } } } \phi _ { \ell , n } \log ^ { 3 } ( n \pi _ { \underline { { \ell } } } ) \log ( 2 B _ { n } ^  2 \end{array}\tag{48}
$$

since

$$
\operatorname * { l i m } _ { n \to \infty } n \underline { { { \pi } } } _ { \ell } \phi _ { \ell , n } \log ^ { 3 } ( n \underline { { { \pi } } } _ { \ell } ) \log ( 2 B _ { n } ^ { 2 } ) = \infty
$$

holds provided that

$$
\operatorname* { l i m } _ { n \to \infty } n \underline { { { \pi } } } _ { \ell } = \infty ,
$$

which holds by assumption. Thus, we have verified (22).

Next, notice that

$$
\operatorname* { s u p } _ { k \in \mathbb { N } } \sum _ { k ^ { \prime } = 1 } ^ { \infty } { \frac { \eta _ { \ell , n } ( 2 ^ { - k - k ^ { \prime } } ) } { 2 ^ { 2 k ^ { \prime } } \eta _ { \ell , n } ( 2 ^ { - k } ) } } \leq \operatorname* { s u p } _ { k \in \mathbb { N } } \sum _ { k ^ { \prime } = 1 } ^ { \infty } { \frac { ( k + k ^ { \prime } ) } { 2 ^ { 2 k ^ { \prime } } k } } \leq 1 ,
$$

which verifies (23).

Now, it remains to verify (19) and (20) for an appropriate $\delta > 0$ . Towards that end, notice that by the proof of Lemma 6 in (Padilla et al., 2024a), (19) holds if δ is chosen to satisfy

$$
\delta \geq \operatorname* { m a x } _ { \substack { m : \frac { n \pi } { 2 } \ell } \leq m \leq 2 n \overline { { \pi } } _ { \ell } } c _ { 1 } \left[ \sqrt { \frac { \log m } { m } } + \sqrt { \frac { \log N \left( 1 / ( 4 8 m ) , \mathcal { F } _ { \ell , { B _ { n } } } , \| \cdot \| _ { m } \right) } { m } } \right]\tag{49}
$$

for a positive constant $c _ { 1 }$ . However, by (46), we can take

$$
\sqrt { \frac { \log ( n \overline { { \pi } } \ell ) } { n \underline { { \pi } } _ { \ell } } } + \sqrt { \phi _ { \ell , n } \log ^ { 3 } ( n \underline { { \pi } } _ { \ell } ) \log ( B _ { n } ^ { 2 } n \overline { { \pi } } _ { \ell } ) } \lesssim \delta
$$

to ensure that (19) holds. Similarly, by the proof of Lemma 6 in (Padilla et al., 2024a), (20) holds if δ is chosen to satisfy

$$
\delta \geq \operatorname* { m a x } _ { \substack { m : \frac { n \pi _ { \ell } } { 2 } \leq m \leq 2 n \overline { { \pi } } _ { \ell } } } c _ { 1 } \left[ \sqrt { \frac { \log m } { m } } + \sqrt { \frac { \log N \left( 1 / ( 4 8 m ) , \mathcal { F } _ { A _ { n } } , \| \cdot \| _ { m } \right) } { m } } \right]\tag{50}
$$

which holds if we take

$$
\sqrt { \frac { \log ( n \overline { { \pi } } \ell ) } { n \underline { { \pi } } _ { \ell } } } + \sqrt { \frac { \phi _ { n } \log ^ { 3 } ( n ) \log ( \mathcal { A } _ { n } ^ { 2 } n \overline { { \pi } } \ell ) } { \underline { { \pi } } _ { \ell } } } \lesssim \delta ,
$$

by our choice of $\mathcal { F }$ in Theorem B.2. Therefore, taking

$$
\sqrt { \frac { \log ( n \overline { { \pi } } _ { \ell } ) } { n \underline { { \pi } } _ { \ell } } } + \sqrt { \frac { \phi _ { n } \log ^ { 3 } ( n ) \log ( \mathcal { A } _ { n } ^ { 2 } n \overline { { \pi } } _ { \ell } ) } { \underline { { \pi } } _ { \ell } } } + \sqrt { \phi _ { \ell , n } \log ^ { 3 } ( n \underline { { \pi } } _ { \ell } ) \log ( \mathcal { B } _ { n } ^ { 2 } n \overline { { \pi } } _ { \ell } ) } \lesssim \delta ,
$$

it follows that (19) and (20) both hold.

## C.5. Proof of Theorem 3.7

Proof. Proceeding as in the proof of Theorem B.2, supposing that

$$
\| ( \hat { f } _ { \ell , \mathcal { B } _ { n } } - \widetilde { G } _ { \ell , \mathcal { B } _ { n } } ) / ( 4 \mathcal { B } _ { n } ) \| _ { \mathcal { Z } _ { \ell } } > \delta ,
$$

we obtain that, in the event $\Omega _ { 1 } \cap \Omega _ { 3 } \cap \Omega _ { 4 }$ 9

$$
\| G _ { \ell } - \hat { f } _ { \ell , \mathcal { B } _ { n } } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } \quad \lesssim \quad \phi _ { \ell , n } + \| \tilde { f } _ { A _ { n } } - \hat { f } _ { A _ { n } } \| _ { \mathcal { L } _ { \ell } } ^ { 2 } + \phi _ { n } + \frac { \mathcal { U } _ { n } ^ { 2 } \eta _ { \ell , n } ( \delta ) } { n \underline { { \pi } } _ { \ell } } + \delta ^ { 2 } .\tag{51}
$$

Next, notice that since $\hat { f } _ { A _ { n } }$ is independent of $\{ ( x _ { i } ) \} _ { i \in \mathbb { Z } _ { \ell } }$ , we obtain that

$$
\mathbb { E } ( \Vert \tilde { f } _ { A _ { n } } - \hat { f } _ { A _ { n } } \Vert _ { \mathcal { I } _ { \ell } } ^ { 2 } \vert \hat { f } _ { A _ { n } } ) = \Vert \tilde { f } _ { A _ { n } } - \hat { f } _ { A _ { n } } \Vert _ { \ell , \mathcal { L } _ { 2 } } ^ { 2 } .
$$

Furthermore, for all $i \in \mathcal { Z } _ { \ell }$ , we have

$$
0 \leq q _ { i } : = \frac { ( \tilde { f } _ { A _ { n } } ( x _ { i } ) - \hat { f } _ { A _ { n } } ( x _ { i } ) ) ^ { 2 } } { 4 \mathcal { A } _ { n } ^ { 2 } } \leq 1 .
$$

Also,

$$
\begin{array} { l l l } { \mathbb { E } ( q _ { i } ^ { 2 } \vert z _ { i } = \ell , \tilde { f } _ { A _ { n } } , \hat { f } _ { A _ { n } } ) } & { \leq } & { \mathbb { E } ( q _ { i } \vert z _ { i } = \ell , \tilde { f } _ { A _ { n } } , \hat { f } _ { A _ { n } } ) } \\ & { = } & { \displaystyle \frac { \| \tilde { f } _ { A _ { n } } - \hat { f } _ { A _ { n } } \| _ { \ell , \mathcal { L } _ { 2 } } ^ { 2 } } { 4 A _ { n } ^ { 2 } } } \end{array}
$$

Hence, conditionining on $\Omega _ { 2 } ( \ell )$ by Bernstein’s inequality and Lemma C.7,

$$
\begin{array} { l l l } { \| \tilde { f } _ { { A } _ { n } } - \hat { f } _ { { A } _ { n } } \| _ { \mathcal { L } _ { \ell } } ^ { 2 } } & { \le } & { \| \tilde { f } _ { { A } _ { n } } - \hat { f } _ { { A } _ { n } } \| _ { \ell , \mathcal { L } _ { 2 } } ^ { 2 } + 2 A _ { n } \sqrt { \displaystyle \frac { \| \tilde { f } _ { { A } _ { n } } - \hat { f } _ { { A } _ { n } } \| _ { \ell , \mathcal { L } _ { 2 } } ^ { 2 } \log ( 2 n ) } { n _ { \ell } } } + \frac { 4 A _ { n } ^ { 2 } \log ( 2 n ) } { 3 n _ { \ell } } } \\ & { \le } & { 2 \| \tilde { f } _ { { A } _ { n } } - \hat { f } _ { { A } _ { n } } \| _ { \ell , \mathcal { L } _ { 2 } } ^ { 2 } + \displaystyle \frac { A _ { n } ^ { 2 } \log ( 2 n ) } { n _ { \ell } } } \\ & { \lesssim } & { \| \tilde { f } _ { { A } _ { n } } - \hat { f } _ { { A } _ { n } } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } + \displaystyle \frac { A _ { n } ^ { 2 } \log ( 2 n ) } { n \pi _ { \ell } } , } \end{array}\tag{52}
$$

with probability at least

$$
1 - 1 / n .
$$

Let $\Omega _ { 5 }$ be the event that (52) holds. Then with $\Omega _ { 1 } , \ldots , \Omega _ { 4 }$ as in the proof of Theorem B.2, we obtain that in $\Omega _ { 1 } \cap \Omega _ { 2 } \cap$ $\Omega _ { 3 } \cap \Omega _ { 4 } \cap \Omega _ { 5 }$

$$
\begin{array} { r c l } { \| G _ { \ell } - \widehat f _ { \ell , \mathscr { B } _ { n } } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } } & { \lesssim } & { \phi _ { \ell , n } + \| \widetilde f _ { A _ { n } } - \widehat f _ { A _ { n } } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } + A _ { n } ^ { 2 } \frac { \log n } { n \underline { { \pi } } _ { \ell } } + \phi _ { n } + \frac { \mathcal { U } _ { n } ^ { 2 } \eta _ { \ell , n } ( \delta ) } { n _ { \ell } } + \delta ^ { 2 } } \\ & { \lesssim } & { \phi _ { \ell , n } + r _ { n } + A _ { n } ^ { 2 } \frac { \log n } { n \underline { { \pi } } _ { \ell } } + \frac { \mathcal { U } _ { n } ^ { 2 } \eta _ { \ell , n } ( \delta ) } { n \underline { { \pi } } _ { \ell } } + \delta ^ { 2 } . } \end{array}\tag{53}
$$

Finally, the claim follows as in the proof of Theorem B.2 and the inequality

$$
\begin{array} { r } { \| g _ { 0 , \ell } - \hat { g } _ { \ell } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } \lesssim \| G _ { \ell } - \hat { f } _ { \ell , \mathcal { B } _ { n } } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } + \| \bar { f } - \hat { f } _ { A _ { n } } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } . } \end{array}
$$

## C.6. Proof of Corollary 3.8

Proof. The proof follows as that of Theorem B.4. The only difference is that we do not need to enforce (20). Hence, δ can be chosen simply to satisfy (50) which as in the proof of Theorem B.4 can be done with δ satisfying

$$
\sqrt { \frac { \log ( n \overline { { \pi } } _ { \ell } ) } { n \underline { { \pi } } _ { \ell } } } + \sqrt { \phi _ { \ell , n } \log ^ { 3 } ( n \underline { { \pi } } _ { \ell } ) \log ( \mathcal { B } _ { n } ^ { 2 } n \overline { { \pi } } _ { \ell } ) } \lesssim \delta .
$$

Then the conclusion follows from Theorem B.2 and by noticing that

$$
\frac { \log n } { n \pi _ { \ell } } < < \phi _ { \ell , n } .
$$

## D. Applications to Other Nonparametric Estimators

The general theoretical framework developed in Section 3.1 applies to a broad class of nonparametric estimators beyond deep ReLU networks. In this section, we present two additional examples: orthogonal series regression and trend filtering. For trend filtering, we instantiate Theorem B.2 (our general result under shared two-stage estimation), and to the best of our knowledge, the transfer learning setting has not been previously studied for this estimator, with our results yielding the first convergence guarantees in this regime. For orthogonal series regression, we instantiate Theorem 3.7 (our resul under independent two-stage estimation) and recover, up to logarithmic factors, the existing convergence rates established in (Wang et al., 2016), providing a unified theoretical perspective on offset-based transfer learning.

## D.1. Trend Filtering

We now apply Theorem B.1 to total variation–based estimators, commonly referred to as trendfiltering (Mammen & Van De Geer, 1997; Tibshirani, 2014). Trend filtering estimates a univariate regression function by penalizing the discrete total variation of its rth-order derivative, and is known to adapt to spatially varying smoothness. To the best of our knowledge, the pretraining setting has not been previously studied for this class of estimators.

Let $s$ be a space of univariate functions invariant to scalar multiplication and $R : S  \mathbb { R }$ a regularizer that acts in $s$ and it is a seminorm.

Assumption D.1. Suppose that there exists constants $K > 0$ and $0 < w \leq 1$ such that for all $m \in \mathbb { N }$ and $\delta \in ( 0 , 1 )$ , it holds that

$$
\operatorname* { s u p } _ { x _ { 1 } , \ldots , x _ { m } \subset [ 0 , 1 ] } \log N ( \delta , B _ { R } ( 1 ) \cap B _ { \infty } ( 1 ) , \| \cdot \| _ { m } ) \leq K \delta ^ { - w } ,
$$

where $\| \cdot \| _ { m }$ is the empirical norm induced by $x _ { 1 } , \ldots , x _ { m }$ , and $B _ { R } ( 1 )$ is the unit ball defined by $R \colon$

$$
B _ { R } ( 1 ) : = \{ f \in { \mathcal { S } } : R ( f ) \leq 1 \}
$$

and $B _ { \infty } ( 1 ) = \{ f \in S : \| f \| _ { \infty } \leq 1 \}$

Assumption D.2. Suppose that $\mathbb { P } ( Z = \ell | X = x ) = \pi _ { \ell } \in ( 0 , 1 ) \mathrm { ~ f o r ~ a l l ~ } x \in [ 0 , 1 ] \mathrm { ~ a n d ~ } \ell \in \{ 1 , \ldots , L \}$

Assumption D.2 restricts the probabilities to be constant for notational simplicity. Consider the model in (1) with $d = 1$ and, for simplicity, with $f _ { 0 } = 0$ . Our goal is to estimate the functions $f _ { 0 , 1 } , \ldots , f _ { 0 , L }$ which now coincide with $g _ { 0 , 1 } , \ldots , g _ { 0 , L }$ respectively.

As an estimator we consider $\hat { g } _ { \ell }$ defined in (3) and (4), and with

$$
{ \mathcal { F } } : = \{ f \in S : R ( f ) \leq V \} \ { \mathrm { ~ a n d ~ } } \ { \mathcal { F } } _ { \ell } : = \{ f \in S : R ( f ) \leq V _ { \ell } \} .\tag{54}
$$

Corollary D.3. Suppose that Assumptions D.1 and D.2 hold and $V \geq V _ { 0 }$ and $V _ { \ell } \geq V _ { 0 , \ell }$ where

$$
V _ { 0 } : = R \left( \sum _ { k = 1 } ^ { L } \pi _ { k } g _ { 0 , k } \right) , \quad V _ { 0 , \ell } : = R \left( g _ { 0 , \ell } - \sum _ { k = 1 } ^ { L } \pi _ { k } g _ { 0 , k } , \right) ,
$$

and min $\{ V _ { 0 } , V _ { 0 , \ell } \} \gtrsim \log n / n ^ { 1 / 2 - 1 / ( 2 + w ) }$ . Suppose that $A _ { n }$ is chosen to satisfy $A _ { n } \to \infty$ and

$$
\begin{array} { r } { A _ { n } \geq 8 \operatorname* { m a x } \{ \| \bar { f } \| _ { \infty } + \mathcal { U } _ { n } , 8 \| \bar { f } \| _ { \infty } , 8 \sqrt { \phi _ { n } } , 1 \} . } \end{array}
$$

Moreover, assume that $B _ { n }$ is taken to satisfy $B _ { n }$ and (18). Then the estimator $\hat { g } _ { \ell }$ defined in (5) satisfies

$$
\| g _ { 0 , \ell } - \hat { g } _ { \ell } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } = O _ { \mathbb { P } } \left( \frac { \operatorname* { m a x } \{ \mathcal { U } _ { n } ^ { 2 } , \mathcal { A } _ { n } , \mathcal { B } _ { n } ^ { 2 } \} ( V ^ { 2 w / ( 2 + w ) } \mathcal { A } _ { n } ^ { 2 w / ( 2 + w ) } + V _ { \ell } ^ { 2 w / ( 2 + w ) } \mathcal { B } _ { n } ^ { 2 w / ( 2 + w ) } ) } { ( n \pi _ { \ell } ) ^ { 2 / ( 2 + w ) } } \right)\tag{55}
$$

provided that $n \pi _ { \ell }  \infty .$

Remark D.4. In the natural setting where max $\begin{array} { r } { \left\{ \mathcal { U } _ { n } , \mathcal { A } _ { n } , \mathcal { B } _ { n } , \| \bar { f } \| _ { \infty } , \operatorname* { m a x } _ { \ell = 1 , \ldots , L } \| f _ { 0 , \ell } \| _ { \infty } , \| G _ { \ell } \| _ { \infty } \right\} = O ( \mathrm { p o l y } ( \log n ) ) . } \end{array}$ and ignoring logarithmic factors, the convergence rate in (55) simplifies to

$$
\left( V _ { 0 } ^ { 2 w / ( 2 + w ) } + V _ { 0 , \ell } ^ { 2 w / ( 2 + w ) } \right) ( n \pi _ { \ell } ) ^ { - 2 / ( 2 + w ) } ,
$$

provided that $V \times V _ { 0 }$ and $V _ { \ell } \asymp V _ { 0 , \ell }$ . In contrast, by Theorem B.1, the naive estimator $\hat { h } _ { \ell } { \mathrm { - } } \mathrm { w h i c h }$ only uses data from the ℓth group and is defined as in (13) with $\mathcal { F } _ { \ell }$ from (54)—satisfies

$$
\| g _ { 0 , \ell } - \hat { h } _ { \ell } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } = O _ { \mathbb { P } } \left( \frac { \widetilde { V } _ { \ell } ^ { 2 w / ( 2 + w ) } } { ( n \pi _ { \ell } ) ^ { 2 / ( 2 + w ) } } \right)
$$

ignoring logarithmic factors, and where $\widetilde { V } _ { \ell } \asymp R ( g _ { 0 , \ell } )$ . Thus, the pretraining estimator $\hat { g } _ { \ell }$ will achieve a faster rate of convergence than the naive estimator $\hat { h } _ { \ell }$ provided that

$$
R \left( \sum _ { k = 1 } ^ { L } \pi _ { k } g _ { 0 , k } \right) + R \left( g _ { 0 , \ell } - \sum _ { k = 1 } ^ { L } \pi _ { k } g _ { 0 , k } \right) < < R ( g _ { 0 , \ell } ) ,
$$

which can occur when the functions $g _ { \mathrm { 0 } , k }$ are relatively similar, and the averaging of these functions results in a representation with lower complexity, in terms of $R ,$ than that of $g _ { 0 , \ell } .$

We now apply Corollary D.3 to total variation–based estimators, commonly referred to as trend filtering (see (Mammen & Van De Geer, 1997; Tibshirani, 2014)). Towards that end, for a function $f : [ 0 , 1 ] \to \mathbb { R }$ , we define its total variation as

$$
\mathrm { T V } ( f ) = \operatorname* { s u p } _ { M \in \mathbb { N } : M \geq 1 } \mathrm { T V } ( f , M ) ,\tag{56}
$$

where

$$
\mathbf { T V } ( f , M ) = \operatorname* { s u p } _ { \substack { 0 \leq a _ { 1 } \leq \ldots \leq a _ { M } \leq 1 } } \sum _ { j = 1 } ^ { M - 1 } | f ( a _ { j } ) - f ( a _ { j + 1 } ) | .
$$

Then, we let $s$ be the class of functions

$$
S : = \{ f : [ 0 , 1 ] \to \mathbb { R } \ : \ \operatorname { T V } ( f ^ { ( r - 1 ) } ) < \infty \}
$$

where $r \in \mathbb { N } \backslash \{ 0 \}$ is fixed and $f ^ { ( r - 1 ) }$ denotes the $( r - 1 )$ th weak derivative of $f .$ . With this notation, we are now ready to state our next result.

Corollary D.5. Let the $( r - 1 ) t h$ order total variation offunction be defined as $R ( f ) = \mathrm { T V } ( f ^ { ( r - 1 ) } )$ and the corresponding function classes $\mathcal { F }$ and $\mathcal { F } _ { \ell }$ as in $( 5 4 )$ . Suppose that the notation and conditions ofCorollary D.3 hold. Then the estimator $\hat { g } _ { \ell }$ defined in (5) satisfies

$$
\| g _ { 0 , \ell } - \hat { g } _ { \ell } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } = O _ { \mathbb { P } } \left( \frac { \operatorname* { m a x } \{ \mathcal { U } _ { n } ^ { 2 } , \mathcal { A } _ { n } , \mathcal { B } _ { n } ^ { 2 } \} ( V ^ { 2 / ( 2 r + 1 ) } A _ { n } ^ { 2 / ( 2 r + 1 ) } + V _ { \ell } ^ { 2 / ( 2 r + 1 ) } B _ { n } ^ { 2 / ( 2 r + 1 ) } ) } { ( n \pi _ { \ell } ) ^ { 2 r / ( 2 r + 1 ) } } \right)\tag{57}
$$

Remark D.6. Let $\begin{array} { r } { \operatorname* { m a x } \Bigl \{ \mathcal { U } _ { n } , A _ { n } , \mathcal { B } _ { n } , \| \bar { f } \| _ { \infty } , \operatorname* { m a x } _ { \ell = 1 , \ldots , L } \| f _ { 0 , \ell } \| _ { \infty } , \| G _ { \ell } \| _ { \infty } \Bigr \} = O ( \mathrm { p o l y } ( \log n ) ) } \end{array}$ just as in Remark $_ { \mathrm { D . 4 , } }$ then the rate achieved by trend filtering with pretraining—namely, the estimator $\hat { g } _ { \ell } - \mathrm { i } \mathrm { s }$

$$
\left( V _ { 0 } ^ { 2 / ( 2 r + 1 ) } + V _ { 0 , \ell } ^ { 2 / ( 2 r + 1 ) } \right) ( n \pi _ { \ell } ) ^ { - 2 r / ( 2 r + 1 ) } ,
$$

while the rate attained by the usual trend filtering estimator is

$$
\left( \mathrm { T V } ( g _ { 0 , \ell } ^ { ( r - 1 ) } ) \right) ^ { 2 / ( 2 r + 1 ) } ( n \pi _ { \ell } ) ^ { - 2 r / ( 2 r + 1 ) } .
$$

Therefore, the pretraining estimator achieves a faster convergence rate than the usual trend filtering estimator whenever the functions

$$
\sum _ { k = 1 } ^ { L } \pi _ { k } g _ { 0 , k } \quad { \mathrm { a n d } } \quad g _ { 0 , \ell } - \sum _ { k = 1 } ^ { L } \pi _ { k } g _ { 0 , k }
$$

are substantially smoother than $g _ { 0 , \ell } ,$ in terms of their total variation.

## Proof of Corollary D.3

Proof. With the notation from Theorem B.1, we let $\mathcal { F } _ { A _ { n } } : = \{ f _ { A _ { n } } / ( 2 A _ { n } ) : f \in \mathcal { F } \}$ . Then

$$
\begin{array} { l l l } { \log N \left( \delta , \mathcal { F } _ { \mathcal { A } _ { n } } , \left. \cdot \right. _ { n } \right) } & { \leq } & { \log N \left( \delta / V , \mathcal { F } _ { \mathcal { A } _ { n } } / V , \left. \cdot \right. _ { n } \right) } \\ & { \leq } & { \log N \left( \delta / V , B _ { R } ( 1 ) \cap B _ { \infty } ( 1 ) , \left. \cdot \right. _ { n } \right) } \\ & { \leq } & { K V ^ { w } \delta ^ { - w } } \\ & { \leq } & { K \mathcal { A } _ { n } ^ { w } V ^ { w } \delta ^ { - w } } \end{array}\tag{58}
$$

Hence, we define

$$
\eta _ { n } ( \delta ) : = K \mathcal { A } _ { n } ^ { w } V ^ { w } \delta ^ { - w } ,
$$

and so by Assumption D.2 we proceed to check the conditions of Theorem B.1.

To verify (14), we observe that

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 0 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \exp \left( - C _ { 1 } \eta _ { t } \big ( 2 ^ { t - t } \ell ^ { - 1 } \big ) \right) + \displaystyle \sum _ { t = 0 } ^ { \infty } \exp \left( - C _ { 2 } \eta _ { t } \big ( 2 ^ { t - t - 1 } \big ) \right) } \\ & { = \displaystyle \sum _ { t = 0 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \exp \left( - C _ { 1 } K ^ { \infty } A _ { \ell } ^ { \mathrm { a v } } V ^ { \infty } 2 ^ { t ( \ell + t + 1 ) \kappa } \right) + \displaystyle \sum _ { t = 0 } ^ { \infty } \exp \left( - C _ { 2 } K ^ { \infty } A _ { \ell } ^ { \mathrm { a v } } V ^ { \infty } 2 ^ { t ( \ell + 1 ) \kappa } \right) } \\ & { \le \displaystyle \sum _ { t = 0 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \exp ( - C _ { 1 } K ^ { \infty } A _ { \ell } ^ { \mathrm { a v } } V ^ { \infty } ( t + t ^ { \prime } + 1 ) w ) + \displaystyle \sum _ { t = 0 } ^ { \infty } \exp \left( - C _ { 2 } K ^ { \infty } A _ { \ell } ^ { \mathrm { a v } } V ^ { \infty } ( t + 1 ) w \right) } \\ & { = \displaystyle \sum _ { t = 0 } ^ { \infty } \left[ \exp \left( - C _ { 1 } K ^ { \infty } A _ { \ell } ^ { \mathrm { a v } } V ^ { \infty } ( t + 1 ) w \right) \cdot \displaystyle \sum _ { t = 1 } ^ { \infty } \left[ \exp \left( - C _ { 1 } K ^ { \infty } A _ { \ell } ^ { \mathrm { a v } } V ^ { \infty } w \right) \right] ^ { t ^ { \prime } } \right] + } \\ & { \displaystyle \sum _ { t = 0 } ^ { \infty } \left[ \exp \left( - C _ { 2 } K ^ { \infty } A _ { \ell } ^ { \mathrm { a v } } V ^ { \infty } ( t + 1 ) \right) \right] ^ { t _ { 1 } } } \\ &  \displaystyle \sum _ { t = 0 } ^  \ \end{array}\tag{59}
$$

since by construction $A _ { n } \to \infty$ . Thus, (14) holds.

To verify (15), notice that

$$
\operatorname* { s u p } _ { k \in \mathbb { N } } \sum _ { k ^ { \prime } = 1 } ^ { \infty } \frac { \eta _ { n } ( 2 ^ { - k - k ^ { \prime } } ) } { 2 ^ { 2 k ^ { \prime } } \eta _ { n } ( 2 ^ { - k } ) } = \operatorname* { s u p } _ { k \in \mathbb { N } } \sum _ { k ^ { \prime } = 1 } ^ { \infty } \frac { 1 } { 2 ^ { ( 2 - w ) k ^ { \prime } } } \le 1 .\tag{60}
$$

Next, we construct a $\delta$ that is a critical radius. Towards that end, we notice that as in Lemma 6 of (Padilla et al., 2024a), it is enough to have δ satisfy

$$
\frac { 1 } { \sqrt { n } } \int _ { \delta ^ { 2 } / 4 } ^ { \delta } \sqrt { \log N \left( t / 2 , \mathcal { F } _ { A _ { n } } , \Vert \cdot \Vert _ { n } \right) } d t + \frac { \delta } { \sqrt { n } } \sqrt { \log ( 4 8 / \delta ^ { 2 } ) } \lesssim \delta ^ { 2 } .\tag{61}
$$

However,

$$
\begin{array} { r l } & { \frac { 1 } { \sqrt { n } } \displaystyle \int _ { \delta ^ { 2 } / 4 } ^ { \delta } \sqrt { \log { N \left( t / 2 , \mathscr { F } _ { A _ { n } } , \| \cdot \| _ { n } \right) } } d t + \frac { \delta } { \sqrt { n } } \sqrt { \log { \left( 4 8 / \delta ^ { 2 } \right) } } } \\ & { \leq \frac { 1 } { \sqrt { n } } \displaystyle \int _ { 0 } ^ { \delta } \sqrt { \log { N \left( t / 2 , \mathscr { F } _ { A _ { n } } , \| \cdot \| _ { n } \right) } } d t + \frac { \delta _ { n } } { \sqrt { n } } \sqrt { \log { \left( 4 8 / \delta ^ { 2 } \right) } } } \\ & { \lesssim \frac { 1 } { \sqrt { n } } \displaystyle \int _ { 0 } ^ { \delta } \sqrt { A _ { n } ^ { w } V ^ { w } t ^ { - w } } d t + \frac { \delta } { \sqrt { n } } \sqrt { \log { \left( 4 8 / \delta ^ { 2 } \right) } } } \\ & { \lesssim \frac { V ^ { w / 2 } A _ { n } ^ { w / 2 } \delta ^ { 1 - w / 2 } } { \sqrt { n } } + \frac { \delta } { \sqrt { n } } \sqrt { \log { \left( 4 8 / \delta ^ { 2 } \right) } } . } \end{array}\tag{62}
$$

Hence, (61) holds for $\delta > 0$ satisfying

$$
\delta ^ { 2 } \asymp n ^ { - 2 / ( 2 + w ) } V ^ { 2 w / ( 2 + w ) } A _ { n } ^ { 2 w / ( 2 + w ) } .\tag{63}
$$

Thus, δ is a critical radius for $\mathcal { F } _ { A _ { n } }$ . Moreover, in this case, we also have that

$$
\frac { \eta _ { n } ( \delta ) } { n } \lesssim \delta ^ { 2 } .
$$

Therefore, from Theorem B.1, we obtain that

$$
\operatorname* { m a x } \{ \| \bar { f } - \hat { f } _ { A _ { n } } \| _ { n } ^ { 2 } , \| \bar { f } - \hat { f } _ { A _ { n } } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } \} = O _ { \mathbb { P } } \left( r _ { n } \right) ,\tag{64}
$$

where

$$
r _ { n } : = \frac { \operatorname* { m a x } \{ \mathcal { U } _ { n } ^ { 2 } , \mathcal { A } _ { n } \} \mathcal { A } _ { n } ^ { 2 w / ( 2 + w ) } V ^ { 2 w / ( 2 + w ) } } { n ^ { 2 / ( 2 + w ) } } .
$$

Next, we analyze the second stage estimator using Theorem B.2. Let $\mathcal { F } _ { \ell , B _ { n } } : = \{ f _ { B _ { n } } / ( 2 \mathcal { B } _ { n } ) : f \in \mathcal { F } _ { \ell } \}$ . Then, taking

$$
\eta _ { \ell , n } ( \delta ) : = K B _ { n } ^ { w } V _ { \ell } ^ { w } \delta ^ { - w } ,
$$

proceeding as in (58), we obtain that

$$
\begin{array} { r } { \underset { \frac { n \pi _ { \ell } } 2 \leq n _ { \ell } \leq 2 n \pi _ { \ell } } { \operatorname* { m a x } } \ \underset { \{ x _ { i } \} _ { i \in I _ { \ell } } \subset [ 0 , 1 ] ^ { d } } { \operatorname* { s u p } } \log N ( \delta , \mathcal { F } _ { \ell , \mathcal { B } _ { n } } , \| \cdot \| _ { I _ { \ell } } ) \leq \eta _ { \ell , n } ( \delta ) . } \end{array}
$$

which shows (17). Moreover, we can verify (22) proceeding as in (59), and (23) as in (60).

Next, we proceed to find δ for which (19) and (20) hold. However, based on our previous calculations (19) and (20) are equivalent to

$$
\frac { 1 } { \sqrt { n \pi _ { \ell } } } \int _ { \delta ^ { 2 } / 4 8 } ^ { \delta } \sqrt { \mathcal { B } _ { n } ^ { w } V _ { \ell } ^ { w } t ^ { - w } } d t + \frac { \delta } { \sqrt { n \pi _ { \ell } } } \sqrt { \log ( 4 8 / \delta ^ { 2 } ) } \lesssim \delta ^ { 2 } ,
$$

and

$$
\frac { 1 } { \sqrt { n \pi _ { \ell } } } \int _ { \delta ^ { 2 } / 4 8 } ^ { \delta } \sqrt { A _ { n } ^ { w } V ^ { w } t ^ { - w } } d t + \frac { \delta } { \sqrt { n \pi _ { \ell } } } \sqrt { \log ( 4 8 / \delta ^ { 2 } ) } \lesssim \delta ^ { 2 } ,
$$

respectively. Hence, (19) and (20) hold for a choice of δ satisfying

$$
\delta ^ { 2 } \asymp ( n \pi _ { \ell } ) ^ { - 2 / ( 2 + w ) } \left\lceil V ^ { 2 w / ( 2 + w ) } \mathcal { A } _ { n } ^ { 2 w / ( 2 + w ) } + V _ { \ell } ^ { 2 w / ( 2 + w ) } \mathcal { B } _ { n } ^ { 2 w / ( 2 + w ) } \right\rceil .
$$

Therefore, the claim follows from Theorem B.2.

## Proof of Corollary D.5

Proof. The statement of Corollary D.5 follows from Corollary D.3, since Assumption D.1 is satisfied for the trend filtering regularizer. In particular, by Corollary 1 of (Sadhanala & Tibshirani, 2019), Assumption D.1 holds with $w = 1 / r$ for $R ( f ) = \mathrm { T V } ( f ^ { ( r - 1 ) } )$ , corresponding to $k = r - 1$ in their notation. □

## D.2. Orthogonal Series Regression

We now apply Theorem 3.7 to orthogonal series estimator, instantiating the function classes $\mathcal { F }$ and $\mathcal { F } _ { \ell }$ as Sobolev ellipsoids on $[ 0 , 1 ] ^ { d }$ . This setting recovers, up to logarithmic factors, the two-task transfer learning rate established in (Wang et al., 2016), whose offset-based decomposition (Model 4.1 of (Wang et al., 2016)) parallels our two-stage procedure.

Setup. Following the set up in (Wang et al., 2016), we specialize the model in (1) to $L = 2$ groups, with group $\ell = 1$ playing the role of the target. The data consist of n i.i.d. copies of $( X , Y , Z )$ , partitioned into the target group ${ \mathcal { T } } _ { 1 } : = \{ i : z _ { i } = 1 \}$ of size $n _ { 1 } : = | \mathbb { Z } _ { 1 } |$ and the source group ${ \mathcal { T } } _ { 2 } : = \{ i : z _ { i } = 2 \}$ of size $n - n _ { 1 }$ . Following (2), the target-group conditional mean function decomposes as

$$
g _ { 0 , 1 } ( x ) : = \mathbb { E } [ Y \mid X = x , Z = 1 ] = { \bar { f } } ( x ) + G _ { 1 } ( x ) ,
$$

where

$$
{ \bar { f } } ( x ) : = \operatorname { \mathbb { E } } [ Y \mid X = x ] \quad { \mathrm { a n d } } \quad G _ { 1 } ( x ) : = g _ { 0 , 1 } ( x ) - { \bar { f } } ( x )
$$

denote the overall mean and the target-specific offset, respectively, as introduced in Section 2. Throughout this subsection, we work under the independent two-stage estimation setting of Theorem 3.7: the Stage-1 estimator $\hat { f }$ is constructed using a sample independent of the one used in Stage 2.

According to (Wang et al., 2016), let $L _ { 2 } ( [ 0 , 1 ] ^ { d } )$ denote the space of square-integrable functions on $[ 0 , 1 ] ^ { d }$ equipped with the inner product $\begin{array} { r } { \langle f , g \rangle = \int _ { [ 0 , 1 ] ^ { d } } f ( x ) g ( x ) } \end{array}$ dx. Let $\{ \varphi _ { j } \} _ { j \in \mathbb { Z } }$ be an orthonormal basis of $L _ { 2 } ( [ 0 , 1 ] )$ with su $\begin{array} { r } { \mathsf { \Delta } _ { j } \parallel \varphi _ { j } \parallel _ { \infty } \le C _ { \varphi } } \end{array}$ for some constant $C _ { \varphi } > 0$ . For example, we can consider the cosine basis in the following:

$$
\varphi _ { j } ( x ) = { \left\{ \begin{array} { l l } { 1 , } & { j = 0 , } \\ { { \sqrt { 2 } } \cos ( j \pi x ) , } & { j \geq 1 , } \end{array} \right. }
$$

where the index $j$ orders the basis functions by frequency and $\operatorname* { s u p } _ { j } \| \varphi _ { j } \| _ { \infty } \leq C _ { \varphi }$ with $C _ { \varphi } = { \sqrt { 2 } } .$ . The tensor product basis is defined as $\{ \varphi _ { \alpha } \} _ { \alpha \in \mathbb { Z } ^ { d } }$ , where

$$
\varphi _ { \alpha } ( x ) = \prod _ { w = 1 } ^ { d } \varphi _ { \alpha _ { w } } ( x _ { w } ) , \qquad \alpha = ( \alpha _ { 1 } , \ldots , \alpha _ { d } ) \in \mathbb { Z } ^ { d } ,
$$

forms an orthonormal basis of $L _ { 2 } ( [ 0 , 1 ] ^ { d } )$ , so every $f \in L _ { 2 } ( [ 0 , 1 ] ^ { d } )$ admits the expansion $\begin{array} { r } { f = \sum _ { \alpha } a _ { \alpha } ( f ) \varphi _ { \alpha } } \end{array}$ with $a _ { \alpha } ( f ) : = \langle \varphi _ { \alpha } , f \rangle$

Definition D.7 (Sobolev Ellipsoid). Fix smoothness $s > 0 ,$ , radius $A > 0 ,$ , and uniform bound $f _ { \mathrm { m a x } } > 0$ , where A and $f _ { \mathrm { m a x } }$ are taken to be common across all smoothness levels s for simplicity. The (isotropic) Sobolev ellipsoid on $[ 0 , 1 ] ^ { d } \mathrm { i }$ s

$$
\mathcal { W } ^ { s } : = \left\{ f \in L _ { 2 } ( [ 0 , 1 ] ^ { d } ) : \sum _ { \alpha \in \mathbb { Z } ^ { d } } a _ { \alpha } ( f ) ^ { 2 } \kappa _ { s } ^ { 2 } ( \alpha ) \leq A ^ { 2 } , \| f \| _ { \infty } \leq f _ { \operatorname* { m a x } } \right\} ,
$$

where

$$
\kappa _ { s } ^ { 2 } ( \alpha ) : = \sum _ { w = 1 } ^ { d } | \alpha _ { w } | ^ { 2 s } .
$$

The quantity $\kappa _ { s } ^ { 2 } ( \alpha )$ assigns larger weights to higher-frequency basis functions. Hence the constraint $\begin{array} { r } { \sum _ { \alpha \in \mathbb { Z } ^ { d } } a _ { \alpha } ( f ) ^ { 2 } \kappa _ { s } ^ { 2 } ( \alpha ) \le } \end{array}$ $A ^ { 2 }$ forces the coefficients $a _ { \alpha } ( f )$ corresponding to high-frequency basis functions to decay sufficiently fast. In this sense, larger values of s correspond to smoother functions.

For each smoothness level $s > 0$ and threshold $t > 0$ , define the set

$$
M _ { s } ( t ) : = \Bigl \{ \alpha \in  { \mathbb { Z } } ^ { d } : \kappa _ { s } ^ { 2 } ( \alpha ) \leq t \Bigr \} ,
$$

which collects all multi-indices whose Sobolev weight does not exceed t. We then write $J : = | M _ { s } ( t ) |$ for the cardinality of $M _ { s } ( t )$ , which stands for the number of basis functions retained at threshold t.

Function Classes. We assume that both the overall mean function and the target offset belong to Sobolev ellipsoids:

$$
\bar { f } \in \mathcal { W } ^ { \tau } , \qquad G _ { 1 } \in \mathcal { W } ^ { \nu } .
$$

The smoothness levels $\tau , \nu > 0$ are allowed to differ. In particular, $\nu > \tau$ corresponds to the regime in which the offset $G _ { 1 }$ is smoother than the overall mean ${ \bar { f } } ,$ which is the setting where transfer learning is expected to be most beneficial.

Now consider the Stage-1 estimator. Define the Stage-1 threshold $t _ { 0 } > 0$ and let $J _ { 0 } : = | M _ { \tau } ( t _ { 0 } )$ denote the number of basis functions retained. For each $\alpha \in M _ { \tau } ( t _ { 0 } )$ , let $\theta _ { \alpha } \in \mathbb { R }$ be the scalar coefficient associated with the basis function $\varphi _ { \alpha }$ collecting these $J _ { 0 }$ coefficients into the vector $\theta = ( \theta _ { \alpha } ) _ { \alpha \in M _ { \tau } ( t _ { 0 } ) } \in \mathbb { R } ^ { J _ { 0 } }$ , we estimate the Stage-1 target function $\bar { f }$ using $\hat { f } \in \mathcal { F } _ { t _ { 0 } }$ , where

$$
\mathcal { F } _ { t _ { 0 } } : = \Big \{ \sum _ { \alpha \in M _ { \tau } ( t _ { 0 } ) } \theta _ { \alpha } \varphi _ { \alpha } : \| \theta \| _ { 2 } \leq R _ { 0 } \Big \} ,
$$

with a fixed $R _ { 0 } \in \mathbb { R } _ { + }$ , and

$$
\hat { f } : = \arg \operatorname* { m i n } _ { f \in \mathcal { F } _ { t _ { 0 } } } \sum _ { i = 1 } ^ { n } \bigl ( y _ { i } - f ( x _ { i } ) \bigr ) ^ { 2 } .
$$

Here $\mathcal { F } _ { t _ { 0 } }$ consists of all linear combinations of the $J _ { 0 }$ lowest-frequency tensor-product basis functions selected under the Sobolev weight corresponding to the smoothness τ. Analogously, for the Stage-2 estimator, let $t _ { 1 } > 0$ and $J _ { 1 } : = | M _ { \nu } ( t _ { 1 } ) |$ For each $\alpha \in M _ { \tau } ( t _ { 1 } )$ , let $\gamma _ { \alpha } ~ \in$ R be the scalar coefficient associated with the basis function $\varphi _ { \alpha } ;$ collecting these $J _ { 1 }$ coefficients into the vector $\gamma = ( \gamma _ { \alpha } ) _ { \alpha \in M _ { \tau } ( t _ { 1 } ) } \in \mathbb { R } ^ { J _ { 1 } }$ , we estimate the Stage-2 target function $G _ { 1 }$ using $\hat { f } _ { 1 } \in \mathcal { G } _ { t _ { 1 } }$ , where

$$
\mathcal { G } _ { t _ { 1 } } : = \Big \{ \sum _ { \alpha \in M _ { \nu } ( t _ { 1 } ) } \gamma _ { \alpha } \varphi _ { \alpha } : \| \gamma \| _ { 2 } \leq R _ { 1 } \Big \} ,
$$

with a fixed $R _ { 1 } \in \mathbb { R } _ { + }$ , and

$$
\hat { f } _ { 1 } : = \arg \operatorname* { m i n } _ { f \in \mathcal { G } _ { t _ { 1 } } } \sum _ { i : z _ { i } = 1 } \big ( y _ { i } - \hat { f } _ { A _ { n } } ( x _ { i } ) - f ( x _ { i } ) \big ) ^ { 2 } ,
$$

where we focus on group 1 for simplicity. The final estimator would be

$$
\hat { g } _ { 1 } ( x ) : = \hat { f } _ { A _ { n } } ( x ) + \hat { f } _ { 1 , B _ { n } } ( x ) .
$$

Now we are ready to state our results.

Corollary D.8. Suppose that $\mathbb { P } ( Z = \ell \mid X = x ) = \pi _ { \ell } \in ( 0 , 1 )$ for all $x \in [ 0 , 1 ]$ and $\ell \in \{ 1 , 2 \}$ , that $\bar { f } \in \mathcal { W } ^ { \tau }$ and $G _ { 1 } \in \mathcal { W } ^ { \nu }$ . Assumefurther that the two stages are estimated on independent samples as in Theorem $3 . 7 ,$ with truncation levels $\mathcal { A } _ { n } = \mathcal { B } _ { n } = f _ { \mathrm { m a x } }$ . Suppose that $\mathcal { U } _ { n }$ grows at most in $O ( \log n ) _ { \mathrm { { \scriptscriptstyle } } }$ . With

$$
J _ { 0 } \asymp \left( \frac { n } { \log n } \right) ^ { d / ( 2 \tau + d ) } , \qquad J _ { 1 } \asymp \left( \frac { n _ { 1 } } { \log n } \right) ^ { d / ( 2 \nu + d ) } ,
$$

and $n _ { \mathrm { 1 } } \asymp n \pi _ { \mathrm { 1 } }$ , the transfer learning estimator $\hat { g } _ { 1 }$ defined in (5) satisfies

$$
\| g _ { 0 , 1 } - \hat { g } _ { 1 } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } = O _ { \mathbb { P } } \left( n ^ { - \frac { 2 \tau } { 2 \tau + d } } + { n _ { 1 } } ^ { - \frac { 2 \nu } { 2 \nu + d } } \right) ,
$$

up to logarithmic factors.

Remark D.9. The two terms in the rate correspond, respectively, to the Stage-1 error of estimating $\bar { f }$ from the full sample of size n, and the Stage-2 error of estimating the offset $G _ { 1 }$ from the target-group sample of size $n _ { 1 }$ . To compare with the simplified rate of (Wang et al., 2016), parameterize $n = n _ { 1 } ^ { \lambda }$ for some $\lambda \geq 1 -$ in their two-task setting, λ captures the relative abundance of source data, with $\lambda = 1$ corresponding to balanced source and target sample sizes and $\lambda \to \infty$ corresponding to a much larger source pool. Substituting yields

$$
\| g _ { 0 , 1 } - \hat { g } _ { 1 } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } = O _ { \mathbb { P } } \bigg ( n _ { 1 } ^ { - \frac { 2 \lambda \tau } { 2 \tau + d } } + n _ { 1 } ^ { - \frac { 2 \nu } { 2 \nu + d } } \bigg )
$$

up to logarithmic factors, which matches the simplified rate stated immediately after Theorem 4.3 of (Wang et al., 2016). Consistent with Remark 3.5, the pretraining estimator achieves a faster rate than the naive estimator using only target-group data — whose rate is $( \log n _ { 1 } / \dot { n _ { 1 } } ) ^ { 2 \sigma / ( 2 \sigma + \overline { { d } } ) }$ when $g _ { 0 , 1 } \in \mathcal { W } ^ { \sigma }$ — whenever $\bar { f }$ and $G _ { 1 }$ are smoother than $g _ { 0 , 1 }$ itself (i.e., $\tau , \nu > \sigma ,$ with λ large enough).

## D.3. Proof of Corollary D.8

We first establish the approximation rate for d-dimensional Sobolev ellipsoids by sieve classes. At the beginning, we state a standard result for approximating a generic function f in a one-dimensional Sobolev ellipsoid. The purpose of the lemma is to bound the projection error incurred by keeping only the first J low-frequency basis functions.

Lemma D.10 (One-Dimensional Approximation by Sobolev Sieves (Lemma 8.4 in (Wasserman, 2006))). Consider $f \in \mathcal { W } ^ { s }$ Let $d = 1$ , so that the multi-index α reduces to a scalar index $\alpha \in \mathbb { Z } ,$ , and $\kappa _ { s } ^ { 2 } ( \alpha ) = | \alpha | ^ { 2 s }$ . Let $M _ { s } ( t ) \subset \mathbb { Z }$ denote the indices ofthe J one-dimensional basisfunctions having $J : = | M _ { \tau } ( t ) |$ , and define the projection

$$
f _ { J } : = \sum _ { \alpha \in M _ { s } ( t ) } a _ { \alpha } ( f ) \varphi _ { \alpha } .
$$

$\begin{array} { r } { I f \sum _ { \alpha \in \mathbb { Z } } | \alpha | ^ { 2 s } a _ { \alpha } ( f ) ^ { 2 } \le C , } \end{array}$ then

$$
\| f - f _ { J } \| _ { \mathcal L _ { 2 } } ^ { 2 } = \sum _ { \alpha \not \in M _ { s } ( t ) } a _ { \alpha } ( f ) ^ { 2 } \leq C J ^ { - 2 s } .
$$

Now we extend this result to d-dimension.

Proposition D.11. Let $f : [ 0 , 1 ] ^ { d } \to \mathbb { F }$ be a function in $\mathcal { W } ^ { s }$ , and let $M _ { s } ( t )$ be the set of indices defined earlier, which contains the J multi-indices with the Sobolev weight $\kappa _ { s } ^ { 2 } ( \alpha )$ truncated by t. For all J sufficiently large, it holds that

$$
\operatorname* { i n f } _ { \tilde { f } \in \mathcal { F } _ { t } } \| f - \tilde { f } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } = \sum _ { \alpha \not \in M _ { s } ( t ) } a _ { \alpha } ( f ) ^ { 2 } \leq C \cdot J ^ { - 2 s / d } ,
$$

with $\alpha \in \mathbb { Z } ^ { d } .$ for some constant C not depending on J.

Proof. For α $\notin M _ { s } ( t ) , \kappa _ { s } ^ { 2 } ( \alpha ) > t .$ , so

$$
\sum _ { \alpha \notin M _ { s } ( t ) } a _ { \alpha } ( f ) ^ { 2 } \leq t ^ { - 1 } \sum _ { \alpha \in \mathbb { Z } ^ { d } } a _ { \alpha } ( f ) ^ { 2 } \kappa _ { s } ^ { 2 } ( \alpha ) \leq \frac { A ^ { 2 } } { t } .\tag{65}
$$

where $\begin{array} { r } { M _ { s } ( t ) \ = \ \{ \alpha \ : \ \sum _ { w = 1 } ^ { d } | \alpha _ { w } | ^ { 2 s } \ \leq \ t \} } \end{array}$ , which forces $| \alpha _ { w } | ~ \le ~ t ^ { 1 / ( 2 s ) }$ for each dimension w. Hence $J \_ { \perp }$ $\left( 2 \lfloor t ^ { 1 / ( 2 s ) } \rfloor + 1 \right) ^ { d } \le \left( 3 t ^ { 1 / ( 2 s ) } \right) ^ { d } \mathrm { f o r } t ^ { 1 / ( 2 s ) } \ge 1$ , giving

$$
t \geq { \frac { J ^ { 2 s / d } } { 3 ^ { 2 s } } } .\tag{66}
$$

The case $t ^ { 1 / ( 2 s ) } < 1$ is degenerate: since $\kappa _ { s } ^ { 2 } ( \alpha ) \geq 1$ for every nonzero integer multi-index $\alpha , \mathrm { i t }$ forces $M _ { s } ( t ) = \{ ( 0 , \dots , 0 ) \}$ and $J = 1$ , which is excluded in the regime of interest where $J \to \infty$ . Combining (65) and (66) yields the claim. □

Remark D.12. The proof of Proposition D.11 establishes the upper bound $J \lesssim t ^ { d / ( 2 s ) }$ . A matching lower bound $J \gtrsim t ^ { d / ( 2 s ) }$ follows analogously by counting the integer points in a box contained in $M _ { s } ( t )$ . Let $\Lambda : = \lfloor ( t / d ) ^ { 1 / ( 2 s ) } \rfloor$ . If $\alpha \in \mathbb { Z } ^ { d }$ satisfies $| \alpha _ { w } | \le \Lambda$ for all $w = 1 , \ldots , d ,$ then

$$
\sum _ { w = 1 } ^ { d } | \alpha _ { w } | ^ { 2 s } \leq d \Lambda ^ { 2 s } \leq d \cdot \frac { t } { d } = t .
$$

Hence $\{ - \Lambda , \ldots , \Lambda \} ^ { d } \subseteq M _ { s } ( t )$ , and therefore

$$
J = | M _ { s } ( t ) | \geq ( 2 \Lambda + 1 ) ^ { d } = \Big ( 2 \Big \lfloor ( t / d ) ^ { 1 / ( 2 s ) } \Big \rfloor + 1 \Big ) ^ { d } .
$$

For $( t / d ) ^ { 1 / ( 2 s ) } \geq 1$ , this implies $J \gtrsim t ^ { d / ( 2 s ) }$ . Combining the two bounds gives $J \asymp t ^ { d / ( 2 s ) }$ . For simplicity we track the dimension J rather than the truncation t.

Approximation Error. By Proposition D.11,

$$
\phi _ { n } : = \operatorname* { i n f } _ { f \in { \mathscr F } _ { t _ { 0 } } } \| \bar { f } - f \| _ { { \mathscr L } _ { 2 } } ^ { 2 } \lesssim J _ { 0 } ^ { - 2 \tau / d } , \qquad \phi _ { 1 , n } : = \operatorname* { i n f } _ { g \in { \mathscr G } _ { t _ { 1 } } } \| G _ { 1 } - g \| _ { { \mathscr L } _ { 2 } } ^ { 2 } \lesssim J _ { 1 } ^ { - 2 \nu / d } .
$$

Empirical Covering Entropy. Since $\mathcal { F } _ { t _ { 0 } }$ and $\mathcal { G } _ { t _ { 1 } }$ are finite-dimensional linear spans of bounded basis functions, each function is uniquely parameterized by its coefficient vector. Moreover, for any sample $\{ x _ { i } \} _ { i = 1 } ^ { n }$ and any coefficient vectors $\theta , \theta ^ { \prime }$

$$
\lVert f _ { \theta } - f _ { \theta ^ { \prime } } \rVert _ { n } ^ { 2 } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Bigl ( \sum _ { \alpha \in M _ { - } ( t _ { \alpha } ) } ( \theta _ { \alpha } - \theta _ { \alpha } ^ { \prime } ) \varphi _ { \alpha } ( x _ { i } ) \Bigr ) ^ { 2 } \leq \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Bigl ( \sum _ { \alpha \in M _ { - } ( t _ { 0 } ) } ( \theta _ { \alpha } - \theta _ { \alpha } ^ { \prime } ) ^ { 2 } \Bigr ) \Bigl ( \sum _ { \alpha \in M _ { - } ( t _ { \alpha } ) } \varphi _ { \alpha } ( x _ { i } ) ^ { 2 } \Bigr ) \leq J _ { 0 } C _ { \varphi } ^ { 2 d } \lVert \theta - \theta ^ { \prime } \rVert _ { 2 } ^ { 2 } ,
$$

where the first inequality follows from the Cauchy–Schwarz inequality and the second uses the uniform bound $\operatorname* { s u p } _ { j } \| \varphi _ { j } \| _ { \infty } \leq$ $C _ { \varphi }$ . Equivalently, $\| f _ { \theta } - f _ { \theta ^ { \prime } } \| _ { m } \leq C _ { \varphi } ^ { 2 d } \sqrt { J _ { 0 } } \| \theta - \theta ^ { \prime } \| _ { 2 }$ . The same bound holds for $\mathcal { G } _ { t _ { 1 } }$ with $J _ { 1 }$ in place of $J _ { 0 } .$ . Consequently,

any $\big ( \delta / ( C _ { \varphi } ^ { 2 d } \sqrt { J _ { 0 } } ) \big )$ -cover of the Euclidean ball $B _ { 2 } ^ { J _ { 0 } } ( R _ { 0 } ) : = \{ \theta \in \mathbb { R } ^ { J _ { 0 } } : \| \theta \| _ { 2 } \leq R _ { 0 } \}$ in $\mathbb { R } ^ { J _ { 0 } }$ induces a δ-cover of $\mathcal { F } _ { t _ { 0 } }$ under $\| \cdot \| _ { n } { \mathrm { : } }$

$$
N ( \delta , \mathcal { F } _ { t _ { 0 } } , \| \cdot \| _ { n } ) \le N \Big ( \delta / ( C _ { \varphi } ^ { 2 d } \sqrt { J _ { 0 } } ) , \ B _ { 2 } ^ { J _ { 0 } } ( R _ { 0 } ) , \ \| \cdot \| _ { 2 } \Big ) \ .
$$

Combined with the standard covering bound $N ( \eta , B _ { 2 } ^ { J } ( R ) , \Vert \cdot \Vert _ { 2 } ) \leq ( 3 R / \eta ) ^ { J }$ (Corollary 4.2.11 of (Vershynin, 2026)), we obtain

$$
\begin{array} { r } { \log N ( \delta , \mathcal { F } _ { t _ { 0 } } , \lVert \cdot \rVert _ { m } ) \lesssim J _ { 0 } \log ( C \sqrt { J _ { 0 } } / \delta ) , \quad \log N ( \delta , \mathcal { G } _ { t _ { 1 } } , \lVert \cdot \rVert _ { m } ) \lesssim J _ { 1 } \log ( C \sqrt { J _ { 1 } } / \delta ) . } \end{array}\tag{67}
$$

Verification of Conditions (14) and (15) for Stage 1. By the entropy bound established above, we may take the majorant

$$
\eta _ { n } ( \delta ) : = c _ { 0 } J _ { 0 } \log ( C \sqrt { J _ { 0 } } / \delta ) ,
$$

under our assumption $\mathcal { A } _ { n } = O ( 1 )$ , which satisfies log $N ( \delta , \mathcal { F } _ { t _ { 0 } , \mathcal { A } _ { n } } , \| \cdot \| _ { n } ) \le \eta _ { n } ( \delta )$ as required by Theorem B.1.

Step 1: Verification of (14). Under increasing $\sqrt { J _ { 0 } }$ with $n , \log ( C \sqrt { J _ { 0 } } ) > 0 .$ , so for any $k \geq 0$ and $k ^ { \prime } \ge 1$

$$
\eta _ { n } ( 2 ^ { - k - k ^ { \prime } - 1 } ) = c _ { 0 } J _ { 0 } \big [ ( k + k ^ { \prime } + 1 ) \log 2 + \log ( C \sqrt { J _ { 0 } } ) \big ] \geq c _ { 0 } J _ { 0 } ( k + k ^ { \prime } + 1 ) \log 2 .
$$

Hence for any $C _ { 1 } > 0$

$$
\sum _ { k = 0 } ^ { \infty } \sum _ { k ^ { \prime } = 1 } ^ { \infty } \exp \left( - C _ { 1 } \eta _ { n } ( 2 ^ { - k - k ^ { \prime } - 1 } ) \right) \leq \sum _ { k = 0 } ^ { \infty } \sum _ { k ^ { \prime } = 1 } ^ { \infty } \rho ^ { k + k ^ { \prime } + 1 } = \rho \cdot \frac { 1 } { 1 - \rho } \cdot \frac { \rho } { 1 - \rho } = \frac { \rho ^ { 2 } } { ( 1 - \rho ) ^ { 2 } } \to 0
$$

as $J _ { 0 } \to \infty$ , where $\rho : = \exp ( - C _ { 1 } c _ { 0 } J _ { 0 } \log 2 ) \in ( 0 , 1 )$ . The second summation in (14) is bounded analogously:

$$
\sum _ { k = 0 } ^ { \infty } \exp \bigl ( - C _ { 2 } \eta _ { n } \bigl ( 2 ^ { - k - 1 } \bigr ) \bigr ) \leq \sum _ { k = 0 } ^ { \infty } \rho ^ { \prime k + 1 } = \frac { \rho ^ { \prime } } { 1 - \rho ^ { \prime } } \to 0 ,
$$

where $\rho ^ { \prime } : = \exp ( - C _ { 2 } c _ { 0 } J _ { 0 } \log 2 ) \in ( 0 , 1 )$ . Together with the noise tail assumption $\mathbb { P } ( \| \epsilon \| _ { \infty } > \mathcal { U } _ { n } )  0$ , condition (14) is verified.

Step 2: Verification of (15). For $k \geq 1$ and $k ^ { \prime } \ge 1$ , since log $( C \sqrt { J _ { 0 } } ) > 0$

$$
\frac { \eta _ { n } ( 2 ^ { - k - k ^ { \prime } } ) } { \eta _ { n } ( 2 ^ { - k } ) } = \frac { ( k + k ^ { \prime } ) \log 2 + \log ( C \sqrt { J _ { 0 } } ) } { k \log 2 + \log ( C \sqrt { J _ { 0 } } ) } \leq \frac { k + k ^ { \prime } } { k } .
$$

Therefore

$$
\operatorname* { s u p } _ { k \geq 1 } \sum _ { k ^ { \prime } = 1 } ^ { \infty } { \frac { \eta _ { n } ( 2 ^ { - k - k ^ { \prime } } ) } { 2 ^ { 2 k ^ { \prime } } \eta _ { n } ( 2 ^ { - k } ) } } \leq \operatorname* { s u p } _ { k \geq 1 } \sum _ { k ^ { \prime } = 1 } ^ { \infty } { \frac { k + k ^ { \prime } } { 2 ^ { 2 k ^ { \prime } } k } } = \operatorname* { s u p } _ { k \geq 1 } \left[ \sum _ { k ^ { \prime } = 1 } ^ { \infty } { \frac { 1 } { 4 ^ { k ^ { \prime } } } } + { \frac { 1 } { k } } \sum _ { k ^ { \prime } = 1 } ^ { \infty } { \frac { k ^ { \prime } } { 4 ^ { k ^ { \prime } } } } \right] = \operatorname* { s u p } _ { k \geq 1 } \left[ { \frac { 1 } { 3 } } + { \frac { 4 } { 9 k } } \right] = { \frac { 7 } { 9 } } < 1 ,
$$

where we used the geometric-series identities $\begin{array} { r } { \sum _ { k ^ { \prime } \ge 1 } 4 ^ { - k ^ { \prime } } = 1 / 3 } \end{array}$ and $\textstyle \sum _ { k ^ { \prime } \geq 1 } k ^ { \prime } \cdot 4 ^ { - k ^ { \prime } } = 4 / 9$ . The case $k = 0$ is handled identically. This verifies (15).

Verification of Condition (19) for Stage 2. Under the independent two-stage estimation setting of Theorem 3.7, condition (20) is replaced by the independence assumption between the data used in Stages 1 and 2, so only condition (19) on the Stage-2 requires verification.

Consider $m \in [ n \pi _ { \ell } / 2 , 2 n \pi _ { \ell } ]$ . By the entropy bound for $\mathcal { G } _ { t }$ established above,

$$
\sqrt { \log N ( t / 4 , \mathcal G _ { t _ { 1 } , B _ { n } } , \Vert \cdot \Vert _ { m } ) } \lesssim \sqrt { J _ { 1 } \log ( C \sqrt { J _ { 1 } } / t ) } .
$$

Therefore, the Stage-2 complexity integral satisfies

$$
\frac { 1 } { \sqrt { m } } \int _ { \delta ^ { 2 } / 4 8 } ^ { \delta } \sqrt { \log { N ( t / 4 , \mathcal { G } _ { t _ { 1 } , B _ { n } } , \| \cdot \| _ { m } ) } } d t \lesssim \frac { \sqrt { J _ { 1 } } } { \sqrt { m } } \int _ { \delta ^ { 2 } / 4 8 } ^ { \delta } \sqrt { \log ( C \sqrt { J _ { 1 } } / t ) } d t .
$$

Since $t \mapsto { \sqrt { \log ( C { \sqrt { J _ { 1 } } } / t ) } }$ is decreasing on $( 0 , C \sqrt { J _ { 1 } } )$

$$
\int _ { \delta ^ { 2 } / 4 8 } ^ { \delta } \sqrt { \log ( C \sqrt { J _ { 1 } } / t ) } d t \lesssim \delta \sqrt { \log ( C \sqrt { J _ { 1 } } / \delta ^ { 2 } ) } .
$$

Hence

$$
\frac { 1 } { \sqrt { m } } \int _ { \delta ^ { 2 } / 4 8 } ^ { \delta } \sqrt { \log N ( l / 4 , \mathcal { G } _ { t _ { 1 } , \mathcal { B } _ { n } } , \| \cdot \| _ { m } ) } d l + \frac { \delta } { \sqrt { m } } \sqrt { c _ { 1 } \log ( 4 8 / \delta ^ { 2 } ) } \lesssim \frac { \delta } { \sqrt { m } } \left( \sqrt { J _ { 1 } \log ( C \sqrt { J _ { 1 } } / \delta ^ { 2 } ) } + \sqrt { \log ( 1 / \delta ^ { 2 } ) } \right) .
$$

Thus condition (19) holds whenever $\delta ^ { 2 } \gtrsim J _ { 1 } \log ( C \sqrt { J _ { 1 } } / \delta ^ { 2 } ) / ( n \pi \varepsilon )$ . Under $n _ { 1 } \asymp n \pi _ { 1 }$ , the choice

$$
\delta ^ { 2 } \asymp \frac { J _ { 1 } \log ( J _ { 1 } \cdot n ) } { n _ { 1 } }
$$

suffices.

Combining the Two Stages. By Theorem B.1, applied to the Stage-1 sieve class $\mathcal { F } _ { t _ { 0 } }$ , the Stage-1 error satisfies

$$
r _ { n } \lesssim J _ { 0 } ^ { - 2 \tau / d } + \frac { J _ { 0 } \log ( J _ { 0 } n ) } { n }
$$

up to polylogarithmic factors. Applying Theorem 3.7 with $\phi _ { 1 , n } \lesssim J _ { 1 } ^ { - 2 \nu / d }$ and effective target sample size $n _ { \mathrm { 1 } } \asymp n \pi _ { \mathrm { 1 } }$

$$
\| g _ { 0 , 1 } - \hat { g } _ { 1 } \| _ { \mathcal { L } _ { 2 } } ^ { 2 } = O _ { \mathbb { P } } \bigg ( J _ { 0 } ^ { - 2 \tau / d } + \frac { J _ { 0 } \log ( J _ { 0 } n ) } { n } + J _ { 1 } ^ { - 2 \nu / d } + \frac { J _ { 1 } \log ( J _ { 1 } n ) } { n _ { 1 } } \bigg )
$$

up to polylogarithmic factors. Balancing bias and variance in each stage yields the choices of $J _ { 0 }$ and $J _ { 1 }$ stated in Corollary D.8, and substituting back gives the stated rate.

## E. Illustrative Examples to Interpret the Theoretical Results

We provide three illustrative examples to complement the theoretical results in the main text. The first example shows that the two-stage transfer learning procedure can yield estimation targets that are considerably simpler than direct estimation of the group-specific functions. The second example instantiates the convergence rate $\phi _ { n } + \phi _ { \ell , n }$ under a concrete compositional structure with 10-dimensional input. The third example compares convergence rates across four estimators under different regimes of the group proportion $\pi _ { \ell }$ , illustrating the advantage of NN with TL over classical methods.

## E.1. Two-Stage Targets Can Be Simpler Than Direct Estimation

We provide a concrete example illustrating that, under the hierarchical compositional structure, both the shared and groupspecific functions can individually be simple, yet their sum is complex, thus yielding simpler estimation targets in the two-stage transfer learning procedure. Consider $L = 2$ groups with proportions $\pi _ { 1 } + \pi _ { 2 } = 1$ and group-specific regression functions

$$
g _ { 0 , 1 } ( x _ { 1 } , \ldots , x _ { 5 } ) = f _ { 0 } ( x _ { 1 } , x _ { 2 } ) + f _ { 0 , 1 } ( x _ { 3 } , x _ { 4 } , x _ { 5 } ) , \quad g _ { 0 , 2 } ( x _ { 1 } , \ldots , x _ { 5 } ) = f _ { 0 } ( x _ { 1 } , x _ { 2 } ) + f _ { 0 , 2 } ( x _ { 3 } , x _ { 4 } , x _ { 5 } ) .
$$

Each $g _ { 0 , \ell }$ depends on all 5 variables, so direct estimation incurs rates governed by dimension 5. Now suppose the group deviation functions satisfy

$$
\pi _ { 1 } f _ { 0 , 1 } ( x _ { 3 } , x _ { 4 } , x _ { 5 } ) + \pi _ { 2 } f _ { 0 , 2 } ( x _ { 3 } , x _ { 4 } , x _ { 5 } ) \approx c
$$

for some constant c. Then the first-stage target

$$
\bar { f } = f _ { 0 } ( x _ { 1 } , x _ { 2 } ) + \pi _ { 1 } f _ { 0 , 1 } + \pi _ { 2 } f _ { 0 , 2 } \approx f _ { 0 } ( x _ { 1 } , x _ { 2 } ) + c
$$

effectively depends only on 2 variables, yielding faster first-stage convergence rates. The second-stage offset for group 2,

$$
G _ { 2 } = \pi _ { 2 } ( f _ { 0 , 2 } - f _ { 0 , 1 } ) ,
$$

depends on 3 variables and becomes nearly zero as $\pi _ { 2 }  0$ , making second-stage estimation easy.

## E.2. Concrete Instantiation of Convergence Rates

We provide a concrete example to illustrate the final convergence rate $\phi _ { n } + \phi _ { \ell , n }$ under specific settings. Consider $L = 2$ groups with proportions $\pi _ { 1 } + \pi _ { 2 } = 1$ and 10-dimensional input $\boldsymbol { x } \in \mathbb { R } ^ { 1 0 }$ . Both groups share the same compositional structure as illustrated in Figures 2 and 3, where the shared components $h _ { 1 } ^ { ( 1 ) } , h _ { 2 } ^ { ( 1 ) }$ have input dimension $K = 2 , 3 , 3$ and smoothness $p = 1 . 5 , 1 . 5 , 2 . 5$ , respectively. The two groups differ only in the group-specific component $h _ { 3 , \mathrm { g r o u p } _ { \ell } } ^ { ( 1 ) }$ (in red color), with $K = 2 , p = 1 . 5$ for both groups.

![](images/0be2a3919688c1a5900132e841e1957e6d680d5636d0fe0679c9b050a53d1264.jpg)  
Figure 2. Illustration of the estimated group mean function $g _ { 0 , 1 } ( x )$ for Group 1, where $\boldsymbol { x } \in \mathbb { R } ^ { 1 0 }$

![](images/7728ee5637493e9c209117258f7b529a708223a1d4f04d1c3913ea60599096b4.jpg)  
Figure 3. Illustration of the estimated group mean function $g _ { 0 , 2 } ( x )$ for Group 2, where $\boldsymbol { r } \in \mathbb { R } ^ { 1 0 }$

Estimating $\bar { f } = \pi _ { 1 } g _ { 0 , 1 } + \pi _ { 2 } g _ { 0 , 2 }$ in the first stage preserves the same compositional structure as $g _ { 0 , 1 }$ and $g _ { 0 , 2 } ,$ , giving an upper bound

$$
\phi _ { n } = \operatorname* { m a x } _ { ( p , K ) \in \{ ( 1 . 5 , 2 ) , ( 1 . 5 , 3 ) , ( 2 . 5 , 3 ) \} } n ^ { - 2 p / ( 2 p + K ) } = n ^ { - 1 / 2 } .
$$

In the second stage, the offset $G _ { 1 } = \pi _ { 2 } ( g _ { 0 , 1 } - g _ { 0 , 2 } )$ cancels all shared components, reducing to a 2-layer model with a single branch, giving an upper bound

$$
\phi _ { 1 , n } = \operatorname* { m a x } _ { ( p , K ) \in \{ ( 1 . 5 , 2 ) , ( 2 . 5 , 1 ) \} } ( n \pi _ { 1 } ) ^ { - 2 p / ( 2 p + K ) } = ( n \pi _ { 1 } ) ^ { - 3 / 5 } .
$$

The overall rate $n ^ { - 1 / 2 } + ( n \pi _ { 1 } ) ^ { - 3 / 5 }$ depends only on the intrinsic dimensions of the compositional structure, not on the ambient dimension $d = 1 0$ . In contrast, standard estimators that treat all 10 dimensions yield the slower rate $n ^ { - 3 / 1 3 } +$ $( n \pi _ { 1 } ) ^ { - 3 / 1 3 }$ , demonstrating the advantage of our method in overcoming the curse of dimensionality.

## E.3. Benefit of Transfer Learning with DNNs over Classical Estimators

We illustrate the benefit of transfer learning (TL) with DNNs in comparison with classical estimators, such as transfer learning with kernel smoothing (Du et al., 2017), through a simple example applying Corollary 3.8. Let $X \in [ 0 , 1 ] ^ { 1 0 }$ and $\ell \in \{ 1 , 2 , 3 \}$ denote the group index. For simplicity, suppose that $\bar { \pi } _ { \ell } = \underline { { \pi } } _ { \ell } = \pi _ { \ell }$ for all ℓ. We observe

$$
Y _ { i } = g _ { 0 , \ell } ( X _ { i } ) + \varepsilon _ { \ell , i } ,
$$

and consider the additive decomposition

$$
g _ { 0 , \ell } ( x ) = f _ { 0 } ( x ) + f _ { 0 , \ell } ( x ) , \quad \ell = 1 , 2 , 3 ,
$$

with shared component $f _ { 0 } ( x ) = | x _ { 1 } + \cdot \cdot \cdot + x _ { 1 0 } - 1 / 2 |$ and group-specific components

$$
f _ { 0 , 1 } ( x ) = | x _ { 1 } + x _ { 2 } + x _ { 3 } + x _ { 4 } - 1 / 2 | ^ { 3 / 2 } , \quad f _ { 0 , 2 } ( x ) = 2 | x _ { 1 } + x _ { 2 } + x _ { 3 } + x _ { 4 } - 1 / 2 | ^ { 3 / 2 } , \quad f _ { 0 , 3 } ( x ) = 3 | x _ { 1 } + x _ { 2 } + x _ { 3 } + x _ { 4 } - 1 / 2 | ^ { 3 / 2 } .
$$

Each group may further have its own noise distribution; for example, $\varepsilon _ { \ell , i } \sim \mathcal { N } ( 0 , \ell \cdot \sqrt { X _ { i } ^ { ( \ell ) } } )$ , where $X _ { i } ^ { ( \ell ) }$ denotes the ℓ-th coordinate of $X _ { i }$ . Since each $g _ { 0 , \ell }$ and the distribution of $\varepsilon _ { \ell }$ can differ across groups. We now use the example above to compare convergence rates under different $\pi _ { \ell }$ with directly comparable classical estimators under the similar offset TL framework, and illustrate the novelty of our results.

(a) Without Transfer Learning. In this example, the smoothness of $g _ { 0 , \ell }$ is $p = 1$ with intrinsic dimensionalities $K = 1 0$ Training each group separately on $n \pi _ { \ell }$ samples, classical estimators must use all $d = 1 0$ dimensions, yielding rate

$$
( n \pi _ { \ell } ) ^ { - { \frac { 2 p } { 2 p + d } } } = ( n \pi _ { \ell } ) ^ { - 1 / 6 } .
$$

Direct NN estimation without transfer learning exploits intrinsic dimensionality $K = 1 0$ , also giving

$$
( n \pi _ { \ell } ) ^ { - { \frac { 2 p } { 2 p + K } } } = ( n \pi _ { \ell } ) ^ { - 1 / 6 } .
$$

This is a deliberately chosen example where direct NN estimation without TL cannot improve upon the classical rate, specifically constructed to motivate the benefit of TL with NN in the subsequent comparison. In most settings, NN estimation does outperform classical methods by exploiting intrinsic dimensionality; in particular, when $d > 1 0$ but $K = 1 0$ , the NN achieves a strictly faster rate by overcoming the curse of dimensionality. (We note that the same function may admit different hierarchical compositions with different intrinsic dimensionalities, since our upper bounds apply at the function class level rather than to a specific function instance. The composition with intrinsic dimension $K = 1 0$ chosen here is just for illustrative purposes.)

(b) With Transfer Learning. The first-stage target is

$$
\bar { f } ( x ) = | x _ { 1 } + \cdot \cdot \cdot + x _ { 1 0 } - 1 / 2 | + ( \pi _ { 1 } + 2 \pi _ { 2 } + 2 \pi _ { 3 } ) \cdot | x _ { 1 } + x _ { 2 } + x _ { 3 } + x _ { 4 } - 1 / 2 | ^ { 3 / 2 } ,
$$

which has smoothness $p = 1$ and intrinsic dimension $K = 1 0$ . The second-stage offset (taking group 1 as an example) is

$$
G _ { 1 } ( x ) = ( 1 - \pi _ { 1 } - 2 \pi _ { 2 } - 3 \pi _ { 3 } ) \cdot | x _ { 1 } + x _ { 2 } + x _ { 3 } + x _ { 4 } - 1 / 2 | ^ { 3 / 2 } ,
$$

which has smoothness $p = 3 / 2$ and intrinsic dimension $K = 4 ,$ hence simpler than $g _ { \boldsymbol { 0 } , \ell }$ and ${ \bar { f } } .$ Under this setting, classica estimators, which require all 10 dimensions for both $\bar { f }$ and $G _ { \mathrm { 1 } } , { \mathrm { y i e l d } }$ rate

$$
n ^ { - 1 / 6 } + ( n \pi _ { 1 } ) ^ { - 3 / 1 3 } .
$$

Our Theorem 3.7, by exploiting the intrinsic dimensionality of $G _ { 1 }$ , yields the faster rate

$$
n ^ { - 1 / 6 } + ( n \pi _ { 1 } ) ^ { - 3 / 7 } .
$$

The four regimes of convergence rates for all four estimators as $\pi _ { 1 }$ varies are summarized in the table below. Specifically: 1. When $\pi _ { 1 } \asymp 1$ , all methods achieve $n ^ { - 1 / 6 }$ and TL provides no additional gain.

2. When $n ^ { - 5 / 1 8 } \ll \pi _ { 1 } \ll 1$ , both classical and NN methods with TL achieve the faster rate $n ^ { - 1 / 6 }$ , while classical and NN without TL are slower at $( n \pi _ { 1 } ) ^ { - 1 / 6 }$

3. When $n ^ { - 1 1 / 1 8 } \lesssim \pi _ { 1 } \lesssim n ^ { - 5 / 1 8 }$ , NN with TL still achieves $n ^ { - 1 / 6 }$ while the classical TL rate degrades to $( n \pi _ { 1 } ) ^ { - 3 / 1 3 }$ ; for instance, at $\pi _ { 1 } \asymp n ^ { - 1 1 / 1 8 }$ , our rate is $n ^ { - 1 / 6 }$ versus $n ^ { - 7 / 7 8 }$ for classical TL.

4. When $n ^ { - 1 } \ll \pi _ { 1 } \ll n ^ { - 1 1 / 1 8 }$ , NN with TL at $( n \pi _ { 1 } ) ^ { - 3 / 7 }$ remains strictly faster than classical TL at $( n \pi _ { 1 } ) ^ { - 3 / 1 3 }$ . The condition $n ^ { - 1 } \ll \pi _ { 1 }$ follows from the group probability requirement in Theorem B.2.

This example clearly illustrates how a small $\pi _ { 1 }$ can drastically worsen the classical rate while NN with TL remains robust.

<table><tr><td>Classical w/o TL NN w/o TL Classical w/ TL NN w/ TL</td><td></td><td></td><td></td><td></td></tr><tr><td> $\pi _ { 1 } \asymp 1$ </td><td> $n ^ { - \frac { 1 } { 6 } }$ </td><td> $n ^ { - \frac { 1 } { 6 } }$ </td><td> $n ^ { - \frac { 1 } { 6 } }$ </td><td> $n ^ { - \frac { 1 } { 6 } }$ </td></tr><tr><td> $n ^ { - \frac { 5 } { 1 8 } } \ll \pi _ { 1 } \ll 1$ </td><td> $( n \pi _ { 1 } ) ^ { - { \frac { 1 } { 6 } } }$ </td><td> $( n \pi _ { 1 } ) ^ { - { \frac { 1 } { 6 } } }$ </td><td> $n ^ { - \frac { 1 } { 6 } }$ </td><td> $n ^ { - \frac { 1 } { 6 } }$ </td></tr><tr><td> $n ^ { - { \frac { 1 1 } { 1 8 } } } \lesssim \pi _ { 1 } \lesssim n ^ { - { \frac { 5 } { 1 8 } } }$ </td><td> $( n \pi _ { 1 } ) ^ { - { \frac { 1 } { 6 } } }$ </td><td> $( n \pi _ { 1 } ) ^ { - { \frac { 1 } { 6 } } }$ </td><td> $( n \pi _ { 1 } ) ^ { - { \frac { 3 } { 1 3 } } }$ </td><td> $n ^ { - \frac { 1 } { 6 } }$ </td></tr><tr><td> $n ^ { - 1 } \ll \pi _ { 1 } \ll n ^ { - \frac { 1 1 } { 1 8 } }$ </td><td> $( n \pi _ { 1 } ) ^ { - { \frac { 1 } { 6 } } }$ </td><td> $( n \pi _ { 1 } ) ^ { - { \frac { 1 } { 6 } } }$ </td><td> $( n \pi _ { 1 } ) ^ { - { \frac { 3 } { 1 3 } } }$ </td><td> $( n \pi _ { 1 } ) ^ { - { \frac { 3 } { 7 } } }$ </td></tr></table>

Table 5. Convergence rates under different regimes of $\pi _ { 1 }$

## F. Details of Numerical Experiments

## F.1. Complete Numerical Results

In Section 4 of the main text, we selectively present four scenarios from our simulation experiments. Scenarios 1 and 2 correspond to low-dimensional settings, while Scenarios 3 and 4 are high-dimensional. In this supplementary section, we report results for four additional low-dimensional scenarios (Extra Scenarios 1–4) to provide a more comprehensive evaluation of method performance in low-dimensional settings.

Extra Scenario 1. This scenario features a shared additive model with simple group-specific shifts. Let the input vector be $q = ( q _ { 1 } , \dots , q _ { 1 0 } ) \overset { \mathrm { i . i . d . } } { \sim } \mathrm { U n i f } ( [ 0 , 1 ] ^ { 1 0 } )$ , holding for Extra Scenario 1–4. With a group-varying scale $\begin{array} { r } { w _ { z } = \frac { L - z + 1 } { L ( L + 1 ) / 2 } } \end{array}$ for group z, the shifts are applied to the last five dimensions as follows:

$$
\tilde { q } _ { j } ^ { ( z ) } = \left\{ \begin{array} { l l } { q _ { j } } & { \mathrm { i f ~ } j = 5 + ( z \bmod 5 ) \mathrm { ~ o r ~ } j = 6 + ( z \bmod 5 ) } \\ { q _ { j } + w _ { z } \cdot \sin ( z \cdot q _ { j } ) } & { \mathrm { i f ~ } j \in \{ 6 , 7 , 8 , 9 , 1 0 \} \setminus \{ 5 + ( z \bmod 5 ) , 6 + ( z \bmod 5 ) \} . } \end{array} \right.
$$

Define $y = g _ { z } ( q ) + \epsilon ,$ where $\begin{array} { r } { g _ { z } ( q ) = f _ { 0 } ( q ) + f _ { 0 , z } ( q ) = \sum _ { j = 1 } ^ { 5 } \theta _ { j } ( q _ { j } ^ { ( z ) } ) + \sum _ { j = 6 } ^ { 1 0 } \theta _ { j } ( \tilde { q } _ { j } ^ { ( z ) } ) } \end{array}$ as the group-specific conditiona mean function. The component functions $\{ \theta _ { j } : \mathbb { R } \to \mathbb { R } \} _ { j = 1 } ^ { 1 0 }$ are defined as follows:

$$
\begin{array} { l l } { { \theta _ { 1 } ( q _ { 1 } ) = \sin ( \operatorname { t a n h } ( q _ { 1 } ^ { 2 } ) ) , } } & { { \theta _ { 6 } ( q _ { 6 } ) = \cos ( \sqrt { 1 + q _ { 6 } ^ { 2 } } ) , } } \\ { { \theta _ { 2 } ( q _ { 2 } ) = \log ( 1 + | q _ { 2 } | ) , } } & { { \theta _ { 7 } ( q _ { 7 } ) = - \log ( 1 + \operatorname { t a n h } ( q _ { 7 } ^ { 2 } ) ) , } } \\ { { \theta _ { 3 } ( q _ { 3 } ) = \exp ( - \sqrt { 1 + q _ { 3 } ^ { 2 } } ) , } } & { { \theta _ { 8 } ( q _ { 8 } ) = - \sin ( \log ( 1 + q _ { 8 } ^ { 2 } ) ) , } } \\ { { \theta _ { 4 } ( q _ { 4 } ) = \cos ( \log ( 1 + q _ { 4 } ^ { 2 } ) ) , } } & { { \theta _ { 9 } ( q _ { 9 } ) = \cos ( \operatorname { t a n h } ( q _ { 9 } ^ { 2 } ) ) , } } \\ { { \theta _ { 5 } ( q _ { 5 } ) = - \sin ( \operatorname { t a n h } ( | q _ { 5 } | ) ) , } } & { { \theta _ { 1 0 } ( q _ { 1 0 } ) = - \exp ( - \sqrt { 1 + q _ { 1 0 } ^ { 2 } } ) . } } \end{array}
$$

Extra Scenario 2. This scenario combines covariate shifts with mild concept shifts. The shared function remains identical to that in Extra Scenario 1, given by $\begin{array} { r } { f _ { 0 } ( q ) = \sum _ { j = 1 } ^ { 5 } \theta _ { j } ( q _ { j } ) } \end{array}$ . The group-specific effects introduce covariate shifts through the group scale $w _ { z } ,$ , defined as

$$
\tilde { q } _ { j } ^ { ( z ) } = \left\{ \begin{array} { l l } { q _ { j } } & { \mathrm { i f ~ } j = 5 + z } \\ { q _ { j } + w _ { z } \cdot \sin ( z \cdot q _ { j } ) } & { \mathrm { i f ~ } j \in \{ 6 , 7 , 8 , 9 , 1 0 \} \setminus \{ 5 + z \} . } \end{array} \right.
$$

The deviation function $f _ { 0 , z }$ further incorporates concept shift by applying a group-specific scaling $f _ { 0 , z } ( q ) = ( 1 + w _ { z } )$ $\textstyle \sum _ { j = 6 } ^ { 1 0 } \theta _ { j } ( \tilde { q } _ { j } ^ { ( z ) } )$ , where $\theta _ { 6 } , \ldots , \theta _ { 1 0 }$ are defined as in Extra Scenario 1.

Extra Scenario 3. This scenario introduces a two-layer hierarchical structure where inputs are first mapped to intermediate representations before final transformation. With the scaling factor $w _ { z }$ , let

$$
\tilde { q } _ { j } ^ { ( z ) } = \left\{ \begin{array} { l l } { q _ { j } } & { \mathrm { i f ~ } j = 5 + ( z \bmod 5 ) \mathrm { ~ o r ~ } j = 6 + ( z \bmod 5 ) } \\ { q _ { j } + w _ { z } \cdot \sin ( z \cdot q _ { j } ) } & { \mathrm { i f ~ } j \in \{ 6 , 7 , 8 , 9 , 1 0 \} \setminus \{ 5 + ( z \bmod 5 ) , 6 + ( z \bmod 5 ) \} } \end{array} \right.
$$

The intermediate transformations are:

$$
\begin{array} { r l } & { h _ { 1 } ( q ) = \sin ( q _ { 1 } ) \cdot q _ { 2 } ^ { 2 } + \exp ( q _ { 3 } ) - q _ { 4 } \cdot q _ { 5 } , } \\ & { h _ { 2 } ( q ) = \cos ( q _ { 2 } ) + q _ { 3 } \cdot \operatorname { t a n h } ( q _ { 4 } ) + q _ { 5 } ^ { 3 } , } \\ & { h _ { 3 } ( q ; z ) = \log \big ( 1 + \tilde { q } _ { 6 } \cdot \tilde { q } _ { 7 } - \sin ( \tilde { q } _ { 8 } ) \big ) - \operatorname { t a n h } ( \tilde { q } _ { 9 } \cdot \tilde { q } _ { 1 0 } ) , } \\ & { h _ { 4 } ( q ; z ) = \exp ( - | \tilde { q } _ { 8 } | ) + \tilde { q } _ { 9 } \cdot ( 1 + \tilde { q } _ { 1 0 } ^ { 2 } ) ^ { - 1 } . } \end{array}
$$

The shared and group-specific functions are constructed as:

$$
f _ { 0 } ( q ) = \sqrt { 1 + h _ { 1 } ( q ) ^ { 2 } } + \sqrt { 1 + h _ { 2 } ( q ) ^ { 2 } } \quad \mathrm { a n d } \quad f _ { 0 , z } ( q ) = | h _ { 3 } ( q ; z ) - h _ { 4 } ( q ; z ) | ,
$$

and the final response is generated according to

$$
y = f _ { 0 } ( q ) + f _ { 0 , z } ( q ) + \epsilon .
$$

Extra Scenario 4. This scenario extends Extra Scenario 3 beyond the additive model by introducing more complex group effects through mixed dimensional shifts. Specifically, for even indices $j \in \{ 2 , 4 , \dots , 1 0 \}$ , we define

$$
\tilde { q } _ { j } ^ { ( z ) } = q _ { j } + w _ { z } \cdot \sin ( z q _ { j } ) .
$$

Group effects are then incorporated by modifying the even-indexed dimensions in the following components:

$$
\begin{array} { r l } & { h _ { 1 } ( q ; z ) = \sin ( q _ { 1 } ) \cdot \tilde { q } _ { 2 } ^ { 2 } + \exp ( q _ { 3 } ) - \tilde { q } _ { 4 } \cdot q _ { 5 } , } \\ & { h _ { 2 } ( q ; z ) = \cos ( \tilde { q } _ { 2 } ) + q _ { 3 } \cdot \operatorname { t a n h } ( \tilde { q } _ { 4 } ) + q _ { 5 } ^ { 3 } , } \\ & { h _ { 3 } ( q ; z ) = \log \left( 1 + \tilde { q } _ { 6 } \cdot q _ { 7 } - \sin ( \tilde { q } _ { 8 } ) \right) - \operatorname { t a n h } ( q _ { 9 } \cdot \tilde { q } _ { 1 0 } ) , } \\ & { h _ { 4 } ( q ; z ) = \exp ( - | \tilde { q } _ { 8 } | ) + q _ { 9 } \cdot ( 1 + \tilde { q } _ { 1 0 } ^ { 2 } ) ^ { - 1 } . } \end{array}
$$

The final conditional mean for group z is defined as

$$
f _ { z } ( q ) = \sqrt { \left| h _ { 1 } ( q ; z ) \cdot h _ { 3 } ( q ; z ) \right| + \left( h _ { 2 } ( q ; z ) + h _ { 4 } ( q ; z ) \right) ^ { 2 } } ,
$$

and the response is generated according to $y = f _ { z } ( q ) + \epsilon$

In the following, we present the complete set of experimental results, including all signal-to-noise ratio (SNR) settings, as well as group-level results for $\mathrm { S N R } = 5$ in the low-dimensional settings. The results are organized from low-dimensional settings (Scenarios 1 and 2, and Extra Scenarios 1–4) to high-dimensional settings (Scenarios 3 and 4). In most cases, the proposed two-stage transfer learning method achieves the best performance as the sample size increases.

Table 6. Average MSE for Scenario 1 with different SNR levels over 50 independent trials. The best performance in terms of lower MSE within each SNR group is bolded.
<table><tr><td>SNR</td><td>Model / Sample size</td><td>1000</td><td>5000</td><td>10000</td><td>30000</td><td>50000</td></tr><tr><td rowspan="11">10</td><td>Pooled (NN)</td><td>0.0352 (0.0356)</td><td>0.0214 (0.0016)</td><td>0.0197 (0.0011)</td><td>0.0209 (0.0056)</td><td>0.0184 (0.0016)</td></tr><tr><td>2-Stage (NN)</td><td>0.0175 (0.0064)</td><td>0.0067 (0.0012)</td><td>0.0045 (0.0007)</td><td>0.0025 (0.0003)</td><td>0.0020 (0.0002)</td></tr><tr><td>Separate (NN)</td><td>0.8039 (0.0735)</td><td>0.0673 (0.0130)</td><td>0.0079 (0.0031)</td><td>0.0041 (0.0005)</td><td>0.0034 (0.0005)</td></tr><tr><td>Top-FT (NN)</td><td>0.0315 (0.0249)</td><td>0.0188 (0.0022)</td><td>0.0154 (0.0017)</td><td>0.0093 (0.0020)</td><td>0.0074 (0.0015)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0177 (0.0037)</td><td>0.0069 (0.0011)</td><td>0.0055 (0.0016)</td><td>0.0069 (0.0061)</td><td>0.0036 (0.0016)</td></tr><tr><td>Pooled (RF)</td><td>0.0957 (0.0109)</td><td>0.0721 (0.0037)</td><td>0.0690 (0.0028)</td><td>0.0675 (0.0019)</td><td>0.0691 (0.0012)</td></tr><tr><td>2-Stage (RF)</td><td>0.0675 (0.0085)</td><td>0.0410 (0.0026)</td><td>0.0338 (0.0016)</td><td>0.0255 (0.0011)</td><td>0.0242 (0.0008)</td></tr><tr><td>Separate (RF)</td><td>0.1300 (0.0198)</td><td>0.0785 (0.0046)</td><td>0.0661 (0.0026)</td><td>0.0548 (0.0015)</td><td>0.0518 (0.0011)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.0957 (0.0109)</td><td>0.0707 (0.0035)</td><td>0.0674 (0.0028)</td><td>0.0658 (0.0019)</td><td>0.0675 (0.0012)</td></tr><tr><td>Pooled (NN)</td><td>0.0364 (0.0355)</td><td>0.0228 (0.0017)</td><td>0.0213 (0.0022)</td><td>0.0234 (0.0068)</td><td>0.0193 (0.0029)</td></tr><tr><td rowspan="8">5</td><td>2-Stage (NN)</td><td>0.0222 (0.0078)</td><td>0.0093 (0.0016)</td><td>0.0064 (0.0009)</td><td>0.0034 (0.0005)</td><td>0.0027 (0.0004)</td></tr><tr><td>Separate (NN)</td><td>0.8183 (0.0788)</td><td>0.0664 (0.0116)</td><td>0.0097 (0.0035)</td><td>0.0060 (0.0007)</td><td>0.0049 (0.0008)</td></tr><tr><td>Top-FT (NN)</td><td>0.0331 (0.0249)</td><td>0.0212 (0.0018)</td><td>0.0177 (0.0018)</td><td>0.0114 (0.0019)</td><td>0.0092 (0.0017)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0219 (0.0044)</td><td>0.0103 (0.0021)</td><td>0.0074 (0.0017)</td><td>0.0103 (0.0097)</td><td>0.0048 (0.0023)</td></tr><tr><td>Pooled (RF)</td><td>0.0952 (0.0105)</td><td>0.0708 (0.0039)</td><td>0.0673 (0.0028)</td><td>0.0655 (0.0018)</td><td>0.0669 (0.0012)</td></tr><tr><td>2-Stage (RF)</td><td>0.0680 (0.0085)</td><td>0.0410 (0.0028)</td><td>0.0337 (0.0016)</td><td>0.0252 (0.0011)</td><td>0.0236 (0.0008)</td></tr><tr><td>Separate (RF)</td><td>0.1318 (0.0195)</td><td>0.0775 (0.0047)</td><td>0.0650 (0.0028)</td><td>0.0532 (0.0015)</td><td>0.0500 (0.0011)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.0945 (0.0106)</td><td>0.0692 (0.0038)</td><td>0.0656 (0.0028)</td><td>0.0637 (0.0018)</td><td>0.0652 (0.0012)</td></tr><tr><td rowspan="11">2</td><td>Pooled (NN)</td><td></td><td></td><td></td><td>0.0271 (0.0094)</td><td></td></tr><tr><td>2-Stage (NN)</td><td>0.0395 (0.0350) 0.0346 (0.0100)</td><td>0.0248 (0.0018)</td><td>0.0243 (0.0027)</td><td>0.0059 (0.0010)</td><td>0.0212 (0.0046) 0.0045 (0.0008)</td></tr><tr><td>Separate (NN)</td><td></td><td>0.0151 (0.0032)</td><td>0.0110 (0.0019)</td><td></td><td></td></tr><tr><td>Top-FT (NN)</td><td>0.8097 (0.0875)</td><td>0.0685 (0.0112)</td><td>0.0115 (0.0011)</td><td>0.0087 (0.0008)</td><td>0.0078 (0.0015)</td></tr><tr><td></td><td>0.0370 (0.0244)</td><td>0.0239 (0.0017)</td><td>0.0212 (0.0014)</td><td>0.0155 (0.0018)</td><td>0.0129 (0.0019)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0306 (0.0081)</td><td>0.0151 (0.0024)</td><td>0.0120 (0.0026)</td><td>0.0146 (0.0105)</td><td>0.0071 (0.0030)</td></tr><tr><td>Pooled (RF)</td><td>0.0970 (0.0106)</td><td>0.0703 (0.0041)</td><td>0.0658 (0.0032)</td><td>0.0625 (0.0019)</td><td>0.0631 (0.0012)</td></tr><tr><td>2-Stage (RF)</td><td>0.0738 (0.0089)</td><td>0.0446 (0.0028)</td><td>0.0366 (0.0021)</td><td>0.0267 (0.0013)</td><td>0.0243 (0.0010)</td></tr><tr><td>Separate (RF)</td><td>0.1401 (0.0216)</td><td>0.0793 (0.0050)</td><td>0.0655 (0.0033)</td><td>0.0520 (0.0015) 0.0603 (0.0018)</td><td>0.0480 (0.0011)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.0960 (0.0105)</td><td>0.0683 (0.0038)</td><td>0.0636 (0.0031)</td><td></td><td>0.0611 (0.0012)</td></tr></table>

Table 7. Average per-group MSE for Scenario 1 over 50 independent trials at an SNR of 5 using neural networks. The best performance in terms of lower MSE is bolded.
<table><tr><td>n</td><td>Group</td><td>Pooled</td><td>2-Stage</td><td>Separate</td><td>Top-FT</td><td>Pool-w-L</td></tr><tr><td rowspan="5">1000</td><td>1</td><td>0.0367 (0.0296)</td><td>0.0435 (0.0341)</td><td>0.8045 (0.3100)</td><td>0.0426 (0.0378)</td><td>0.0353 (0.0154)</td></tr><tr><td>2</td><td>0.0441 (0.0444)</td><td>0.0302 (0.0302)</td><td>0.8241 (0.2425)</td><td>0.0417 (0.0380)</td><td>0.0283 (0.0100)</td></tr><tr><td>3</td><td>0.0359 (0.0325)</td><td>0.0222 (0.0117)</td><td>0.8402 (0.2157)</td><td>0.0342 (0.0303)</td><td>0.0206 (0.0077)</td></tr><tr><td>4</td><td>0.0373 (0.0426)</td><td>0.0205 (0.0151)</td><td>0.8012 (0.1283)</td><td>0.0328 (0.0272)</td><td>0.0208 (0.0075)</td></tr><tr><td>5</td><td>0.0326 (0.0351)</td><td>0.0157 (0.0051)</td><td>0.8198 (0.1723)</td><td>0.0272 (0.0198)</td><td>0.0180 (0.0059)</td></tr><tr><td rowspan="5">5000</td><td>1</td><td>0.0302 (0.0058)</td><td>0.0164 (0.0078)</td><td>0.8092 (0.1464)</td><td>0.0294 (0.0064)</td><td>0.0171 (0.0059)</td></tr><tr><td>2</td><td>0.0266 (0.0039)</td><td>0.0107 (0.0031)</td><td>0.0209 (0.0421)</td><td>0.0259 (0.0043)</td><td>0.0118 (0.0033)</td></tr><tr><td>3</td><td>0.0248 (0.0036)</td><td>0.0088 (0.0029)</td><td>0.0151 (0.0212)</td><td>0.0232 (0.0043)</td><td>0.0105 (0.0029)</td></tr><tr><td>4</td><td>0.0219 (0.0027)</td><td>0.0087 (0.0024)</td><td>0.0104 (0.0016)</td><td>0.0205 (0.0030)</td><td>0.0094 (0.0023)</td></tr><tr><td>5</td><td>0.0191 (0.0026)</td><td>0.0082 (0.0020)</td><td>0.0094 (0.0016)</td><td>0.0170 (0.0027)</td><td>0.0090 (0.0027)</td></tr><tr><td rowspan="5">10000</td><td>1</td><td>0.0285 (0.0039)</td><td>0.0090 (0.0028)</td><td>0.0224 (0.0510)</td><td>0.0261 (0.0041)</td><td>0.0103 (0.0032)</td></tr><tr><td>2</td><td>0.0254 (0.0036)</td><td>0.0069 (0.0012)</td><td>0.0104 (0.0018)</td><td>0.0214 (0.0038)</td><td>0.0083 (0.0022)</td></tr><tr><td>3</td><td>0.0231 (0.0028)</td><td>0.0067 (0.0019)</td><td>0.0089 (0.0015)</td><td>0.0184 (0.0036)</td><td>0.0076 (0.0023)</td></tr><tr><td>4</td><td>0.0206 (0.0031)</td><td>0.0061 (0.0016)</td><td>0.0083 (0.0012)</td><td>0.0167 (0.0030)</td><td>0.0069 (0.0019)</td></tr><tr><td>5</td><td>0.0176 (0.0026)</td><td>0.0057 (0.0010)</td><td>0.0085 (0.0014)</td><td>0.0150 (0.0027)</td><td>0.0066 (0.0023)</td></tr><tr><td rowspan="4">30000</td><td>1</td><td>0.0308 (0.0070)</td><td>0.0045 (0.0013)</td><td>0.0091 (0.0014)</td><td>0.0206 (0.0053)</td><td>0.0122 (0.0094)</td></tr><tr><td>2</td><td>0.0280 (0.0069)</td><td>0.0040 (0.0012)</td><td>0.0075 (0.0012)</td><td>0.0154 (0.0037)</td><td>0.0109 (0.0107)</td></tr><tr><td>3</td><td>0.0251 (0.0071)</td><td>0.0036 (0.0008)</td><td>0.0062 (0.0016)</td><td>0.0124 (0.0033)</td><td>0.0107 (0.0104)</td></tr><tr><td>4</td><td>0.0224 (0.0073)</td><td>0.0031 (0.0007)</td><td>0.0053 (0.0013)</td><td>0.0105 (0.0031)</td><td>0.0099 (0.0110)</td></tr><tr><td rowspan="3"></td><td>5</td><td>0.0197 (0.0068)</td><td>0.0032 (0.0007)</td><td>0.0053 (0.0014)</td><td>0.0079 (0.0023)</td><td>0.0097 (0.0088)</td></tr><tr><td>1</td><td>0.0263 (0.0039)</td><td>0.0040 (0.0012)</td><td>0.0089 (0.0015)</td><td>0.0187 (0.0044)</td><td>0.0060 (0.0026)</td></tr><tr><td>2</td><td>0.0235 (0.0036)</td><td>0.0030 (0.0006)</td><td>0.0061 (0.0014)</td><td></td><td>0.0055 (0.0032)</td></tr><tr><td rowspan="4">50000</td><td>3</td><td></td><td></td><td></td><td>0.0124 (0.0037)</td><td></td></tr><tr><td></td><td>0.0209 (0.0033)</td><td>0.0030 (0.0008)</td><td>0.0053 (0.0020)</td><td>0.0096 (0.0029)</td><td>0.0048 (0.0029)</td></tr><tr><td>4 5</td><td>0.0189 (0.0029)</td><td>0.0026 (0.0006)</td><td>0.0042 (0.0017)</td><td>0.0084 (0.0025)</td><td>0.0045 (0.0023)</td></tr><tr><td></td><td>0.0156 (0.0032)</td><td>0.0024 (0.0005)</td><td>0.0039 (0.0013)</td><td>0.0063 (0.0020)</td><td>0.0044 (0.0025)</td></tr></table>

Table 8. Average MSE for Scenario 2 with different SNR levels over 50 independent trials. The best performance in terms of lower MSE within each SNR group is bolded.
<table><tr><td>SNR</td><td>Model / Sample size</td><td>1000</td><td>5000</td><td>10000</td><td>30000</td><td>50000</td></tr><tr><td rowspan="11">10</td><td>Pooled (NN)</td><td>0.0592 (0.0098)</td><td>0.0358 (0.0049)</td><td>0.0331 (0.0039)</td><td>0.0355 (0.0132)</td><td>0.0263 (0.0026)</td></tr><tr><td>2-Stage (NN)</td><td>0.0537 (0.0097)</td><td>0.0220 (0.0020)</td><td>0.0169 (0.0011)</td><td>0.0109 (0.0010)</td><td>0.0083 (0.0005)</td></tr><tr><td>Separate (NN)</td><td>0.6172 (0.0661)</td><td>0.0759 (0.0129)</td><td>0.0273 (0.0053)</td><td>0.0160 (0.0014)</td><td>0.0140 (0.0015)</td></tr><tr><td>Top-FT (NN)</td><td>0.0579 (0.0103)</td><td>0.0303 (0.0027)</td><td>0.0252 (0.0027)</td><td>0.0166 (0.0018)</td><td>0.0135 (0.0011)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0817 (0.0191)</td><td>0.0271 (0.0026)</td><td>0.0218 (0.0039)</td><td>0.0190 (0.0068)</td><td>0.0112 (0.0020)</td></tr><tr><td>Pooled (RF)</td><td>0.1841 (0.0239)</td><td>0.1249 (0.0103)</td><td>0.1161 (0.0066)</td><td>0.1112 (0.0031)</td><td>0.1134 (0.0033)</td></tr><tr><td>2-Stage (RF)</td><td>0.1339 (0.0198)</td><td>0.0770 (0.0077)</td><td>0.0642 (0.0047)</td><td>0.0504 (0.0020)</td><td>0.0475 (0.0019)</td></tr><tr><td>Separate (RF)</td><td>0.2683 (0.0426)</td><td>0.1566 (0.0124)</td><td>0.1315 (0.0067)</td><td>0.1045 (0.0026)</td><td>0.0978 (0.0024)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.1851 (0.0240)</td><td>0.1241 (0.0105)</td><td>0.1152 (0.0066)</td><td>0.1106 (0.0031)</td><td>0.1129 (0.0033)</td></tr><tr><td>Pooled (NN)</td><td>0.0779 (0.0146)</td><td>0.0397 (0.0051)</td><td>0.0351 (0.0033)</td><td>0.0367 (0.0105)</td><td>0.0277 (0.0021)</td></tr><tr><td rowspan="8">5</td><td>2-Stage (NN)</td><td>0.0774 (0.0162)</td><td>0.0275 (0.0025)</td><td>0.0209 (0.0015)</td><td>0.0144 (0.0011)</td><td>0.0109 (0.0008)</td></tr><tr><td>Separate (NN)</td><td>0.6209 (0.0756)</td><td>0.0884 (0.0138)</td><td>0.0375 (0.0076)</td><td>0.0200 (0.0011)</td><td>0.0175 (0.0019)</td></tr><tr><td>Top-FT (NN)</td><td>0.0782 (0.0155)</td><td>0.0352 (0.0026)</td><td>0.0292 (0.0026)</td><td>0.0198 (0.0021)</td><td>0.0164 (0.0015)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.1094 (0.0195)</td><td>0.0352 (0.0071)</td><td>0.0273 (0.0054)</td><td>0.0253 (0.0121)</td><td>0.0151 (0.0021)</td></tr><tr><td>Pooled (RF)</td><td>0.1828 (0.0245)</td><td>0.1237(0.0100)</td><td>0.1143 (0.0062)</td><td>0.1087 (0.0032)</td><td>0.1103 (0.0032)</td></tr><tr><td>2-Stage (RF)</td><td>0.1366 (0.0205)</td><td>0.0796 (0.0078)</td><td>0.0658 (0.0045)</td><td>0.0512 (0.0021)</td><td>0.0476 (0.0019)</td></tr><tr><td>Separate (RF)</td><td>0.2722 (0.0436)</td><td>0.1570 (0.0129)</td><td>0.1312 (0.0067)</td><td>0.1031 (0.0027)</td><td>0.0957 (0.0024)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.1835 (0.0245)</td><td>0.1226 (0.0100)</td><td>0.1133 (0.0061)</td><td>0.1078 (0.0032)</td><td>0.1097 (0.0032)</td></tr><tr><td rowspan="11">2</td><td>Pooled (NN)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2-Stage (NN)</td><td>0.1080 (0.0168)</td><td>0.0513 (0.0075)</td><td>0.0420 (0.0056)</td><td>0.0460 (0.0172)</td><td>0.0342 (0.0076)</td></tr><tr><td>Separate (NN)</td><td>0.1187 (0.0214)</td><td>0.0433 (0.0059)</td><td>0.0298 (0.0030)</td><td>0.0202 (0.0018)</td><td>0.0165 (0.0014)</td></tr><tr><td></td><td>0.6185 (0.0803)</td><td>0.1192 (0.0148)</td><td>0.0615 (0.0108)</td><td>0.0307 (0.0038)</td><td>0.0249 (0.0028)</td></tr><tr><td>Top-FT (NN)</td><td>0.1101 (0.0169)</td><td>0.0463 (0.0045)</td><td>0.0369 (0.0025)</td><td>0.0254 (0.0021)</td><td>0.0217 (0.0017)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.1423 (0.0272)</td><td>0.0565 (0.0100)</td><td>0.0395 (0.0082)</td><td>0.0291 (0.0111)</td><td>0.0209 (0.0040)</td></tr><tr><td>Pooled (RF)</td><td>0.1878 (0.0260)</td><td>0.1244 (0.0107)</td><td>0.1131 (0.0061)</td><td>0.1046 (0.0030)</td><td>0.1047 (0.0029)</td></tr><tr><td>2-Stage (RF)</td><td>0.1508 (0.0225)</td><td>0.0892 (0.0085)</td><td>0.0729 (0.0046)</td><td>0.0548 (0.0021)</td><td>0.0496 (0.0019)</td></tr><tr><td>Separate (RF)</td><td>0.2912 (0.0448)</td><td>0.1625 (0.0138)</td><td>0.1346 (0.0071)</td><td>0.1033 (0.0031)</td><td>0.0940 (0.0022)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.1883 (0.0261)</td><td>0.1228 (0.0104)</td><td>0.1116 (0.0061)</td><td>0.1034 (0.0030)</td><td>0.1037 (0.0029)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 9. Average per-group MSE for Scenario 2 over 50 independent trials at an SNR of 5 using neural networks. The best performance in terms of lower MSE is bolded.
<table><tr><td>n</td><td>Group</td><td>Pooled</td><td>2-Stage</td><td>Separate</td><td>Top-FT</td><td>Pool-w-L</td></tr><tr><td rowspan="5">1000</td><td>1</td><td>0.0768 (0.0401)</td><td>0.0852 (0.0420)</td><td>0.6384 (0.2559)</td><td>0.0820 (0.0420)</td><td>0.1501 (0.0883)</td></tr><tr><td>2</td><td>0.0804 (0.0265)</td><td>0.0843 (0.0322)</td><td>0.6267 (0.1901)</td><td>0.0810 (0.0262)</td><td>0.1269 (0.0490)</td></tr><tr><td>3</td><td>0.0861 (0.0384)</td><td>0.0862 (0.0416)</td><td>0.6510 (0.1942)</td><td>0.0871 (0.0407)</td><td>0.1133 (0.0402)</td></tr><tr><td>4</td><td>0.0811 (0.0211)</td><td>0.0770 (0.0212)</td><td>0.6245 (0.1700)</td><td>0.0807 (0.0224)</td><td>0.1038 (0.0327)</td></tr><tr><td>5</td><td>0.0698 (0.0201)</td><td>0.0680 (0.0194)</td><td>0.5940 (0.1198)</td><td>0.0691 (0.0197)</td><td>0.0957 (0.0258)</td></tr><tr><td rowspan="5">5000</td><td>1</td><td>0.0467 (0.0128)</td><td>0.0398 (0.0123)</td><td>0.6086 (0.1238)</td><td>0.0435 (0.0103)</td><td>0.0507 (0.0131)</td></tr><tr><td>2</td><td>0.0450 (0.0087)</td><td>0.0302 (0.0061)</td><td>0.0865 (0.0717)</td><td>0.0408 (0.0069)</td><td>0.0418 (0.0102)</td></tr><tr><td>3</td><td>0.0421 (0.0083)</td><td>0.0281 (0.0054)</td><td>0.0523 (0.0174)</td><td>0.0379 (0.0075)</td><td>0.0368 (0.0089)</td></tr><tr><td>4</td><td>0.0393 (0.0070)</td><td>0.0261 (0.0051)</td><td>0.0461 (0.0100)</td><td>0.0344 (0.0058)</td><td>0.0336 (0.0103)</td></tr><tr><td>5</td><td>0.0350 (0.0054)</td><td>0.0246 (0.0041)</td><td>0.0389 (0.0077)</td><td>0.0304 (0.0036)</td><td>0.0298 (0.0072)</td></tr><tr><td rowspan="5">10000</td><td>1</td><td>0.0424 (0.0089)</td><td>0.0275 (0.0068)</td><td>0.1050 (0.1076)</td><td>0.0373 (0.0079)</td><td>0.0382 (0.0101)</td></tr><tr><td>2</td><td>0.0406 (0.0064)</td><td>0.0235 (0.0049)</td><td>0.0456 (0.0092)</td><td>0.0349 (0.0064)</td><td>0.0311 (0.0068)</td></tr><tr><td>3</td><td>0.0374 (0.0061)</td><td>0.0212 (0.0031)</td><td>0.0346 (0.0054)</td><td>0.0308 (0.0060)</td><td>0.0273 (0.0057)</td></tr><tr><td>4</td><td>0.0340 (0.0038)</td><td>0.0199 (0.0031)</td><td>0.0298 (0.0059)</td><td>0.0277 (0.0044)</td><td>0.0251 (0.0066)</td></tr><tr><td>5</td><td>0.0309 (0.0048)</td><td>0.0192 (0.0027)</td><td>0.0285 (0.0062)</td><td>0.0255 (0.0038)</td><td>0.0253 (0.0063)</td></tr><tr><td rowspan="4">30000</td><td>1</td><td>0.0441 (0.0133)</td><td>0.0182 (0.0032)</td><td>0.0353 (0.0068)</td><td>0.0301 (0.0068)</td><td>0.0322 (0.0170)</td></tr><tr><td>2</td><td>0.0416 (0.0119)</td><td>0.0163 (0.0027)</td><td>0.0231 (0.0035)</td><td>0.0219 (0.0040)</td><td>0.0280 (0.0132)</td></tr><tr><td>3</td><td>0.0387 (0.0107)</td><td>0.0145 (0.0022)</td><td>0.0198 (0.0031)</td><td>0.0208 (0.0035)</td><td>0.0247 (0.0114)</td></tr><tr><td>4</td><td>0.0352 (0.0108)</td><td>0.0139 (0.0019)</td><td>0.0182 (0.0034)</td><td>0.0185 (0.0033)</td><td>0.0236 (0.0141)</td></tr><tr><td rowspan="3"></td><td>5</td><td>0.0331 (0.0105)</td><td>0.0131 (0.0016)</td><td>0.0172 (0.0024)</td><td>0.0174 (0.0023)</td><td>0.0245 (0.0128)</td></tr><tr><td>1</td><td>0.0359 (0.0047)</td><td>0.0147 (0.0028)</td><td>0.0284 (0.0057)</td><td>0.0256 (0.0048)</td><td>0.0218 (0.0043)</td></tr><tr><td>2</td><td>0.0328 (0.0044)</td><td>0.0126 (0.0016)</td><td>0.0206 (0.0045)</td><td>0.0193 (0.0037)</td><td>0.0171 (0.0036)</td></tr><tr><td rowspan="4">50000</td><td>3</td><td>0.0293 (0.0030)</td><td>0.0110 (0.0012)</td><td>0.0182 (0.0049)</td><td>0.0170 (0.0024)</td><td>0.0152 (0.0030)</td></tr><tr><td>4</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>5</td><td>0.0265 (0.0035)</td><td>0.0104 (0.0012)</td><td>0.0156 (0.0027)</td><td>0.0153 (0.0020)</td><td>0.0142 (0.0032)</td></tr><tr><td></td><td>0.0242 (0.0033)</td><td>0.0097 (0.0011)</td><td>0.0152 (0.0034)</td><td>0.0139 (0.0019)</td><td>0.0136 (0.0029)</td></tr></table>

Table 10. Average MSE for Extra Scenario 1 with different SNR levels over 50 independent trials. The best performance in terms of lower MSE within each SNR group is bolded.
<table><tr><td>SNR</td><td>Model / Sample size</td><td>1000</td><td>5000</td><td>10000</td><td>30000</td><td>50000</td></tr><tr><td rowspan="11">10</td><td>Pooled (NN)</td><td>0.0146 (0.0021)</td><td>0.0106 (0.0009)</td><td>0.0096 (0.0009)</td><td>0.0100 (0.0024)</td><td>0.0085 (0.0007)</td></tr><tr><td>2-Stage (NN)</td><td>0.0109 (0.0016)</td><td>0.0051 (0.0007)</td><td>0.0035 (0.0003)</td><td>0.0021 (0.0002)</td><td>0.0016 (0.0001)</td></tr><tr><td>Separate (NN)</td><td>0.4149 (0.0643)</td><td>0.0367 (0.0080)</td><td>0.0059 (0.0005)</td><td>0.0032 (0.0003)</td><td>0.0026 (0.0003)</td></tr><tr><td>Top-FT (NN)</td><td>0.0114 (0.0020)</td><td>0.0060 (0.0006)</td><td>0.0044 (0.0003)</td><td>0.0030 (0.0002)</td><td>0.0026 (0.0002)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0115 (0.0019)</td><td>0.0059 (0.0008)</td><td>0.0042 (0.0008)</td><td>0.0043 (0.0026)</td><td>0.0019 (0.0006)</td></tr><tr><td>Pooled (RF)</td><td>0.0405 (0.0056)</td><td>0.0278 (0.0017)</td><td>0.0255 (0.0013)</td><td>0.0247 (0.0006)</td><td>0.0254 (0.0005)</td></tr><tr><td>2-Stage (RF)</td><td>0.0279 (0.0039)</td><td>0.0157 (0.0010)</td><td>0.0124 (0.0007)</td><td>0.0093 (0.0004)</td><td>0.0086 (0.0003)</td></tr><tr><td>Separate (RF)</td><td>0.0640 (0.0092)</td><td>0.0326 (0.0019)</td><td>0.0261 (0.0012)</td><td>0.0201 (0.0006)</td><td>0.0188 (0.0004)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.0402 (0.0055)</td><td>0.0271 (0.0017)</td><td>0.0247 (0.0012)</td><td>0.0240 (0.0006)</td><td>0.0247 (0.0005)</td></tr><tr><td>Pooled (NN)</td><td>0.0170 (0.0027)</td><td>0.0121 (0.0011)</td><td>0.0106 (0.0011)</td><td>0.0119 (0.0039)</td><td>0.0089 (0.0011)</td></tr><tr><td rowspan="8">5</td><td>2-Stage (NN)</td><td>0.0140 (0.0028)</td><td>0.0068 (0.0008)</td><td>0.0047 (0.0005)</td><td>0.0028 (0.0003)</td><td>0.0022 (0.0002)</td></tr><tr><td>Separate (NN)</td><td>0.4180 (0.0605)</td><td>0.0365 (0.0099)</td><td>0.0079 (0.0020)</td><td>0.0045 (0.0006)</td><td>0.0037 (0.0005)</td></tr><tr><td>Top-FT (NN)</td><td>0.0140 (0.0026)</td><td>0.0074 (0.0007)</td><td>0.0054 (0.0005)</td><td>0.0035 (0.0003)</td><td>0.0030 (0.0002)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0148 (0.0028)</td><td>0.0078 (0.0011)</td><td>0.0059 (0.0012)</td><td>0.0050 (0.0023)</td><td>0.0026 (0.0007)</td></tr><tr><td>Pooled (RF)</td><td>0.0412 (0.0060)</td><td>0.0277 (0.0017)</td><td>0.0252 (0.0013)</td><td>0.0241 (0.0006)</td><td>0.0246 (0.0005)</td></tr><tr><td>2-Stage (RF)</td><td>0.0294 (0.0043)</td><td>0.0164 (0.0011)</td><td>0.0130 (0.0008)</td><td>0.0097 (0.0004)</td><td>0.0088 (0.0004)</td></tr><tr><td>Separate (RF)</td><td>0.0653 (0.0096)</td><td>0.0330 (0.0021)</td><td>0.0263 (0.0013)</td><td>0.0200 (0.0006)</td><td>0.0185 (0.0004)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.0408 (0.0059)</td><td>0.0269 (0.0016)</td><td>0.0243 (0.0013)</td><td>0.0233 (0.0006)</td><td>0.0239 (0.0005)</td></tr><tr><td rowspan="11">2</td><td>Pooled (NN)</td><td></td><td></td><td></td><td>0.0122 (0.0030)</td><td></td></tr><tr><td>2-Stage (NN)</td><td>0.0216 (0.0044)</td><td>0.0143 (0.0017)</td><td>0.0128 (0.0013)</td><td></td><td>0.0098 (0.0010)</td></tr><tr><td>Separate (NN)</td><td>0.0220 (0.0060)</td><td>0.0101 (0.0017)</td><td>0.0074 (0.0011)</td><td>0.0045 (0.0006)</td><td>0.0037 (0.0004)</td></tr><tr><td></td><td>0.4198 (0.0632)</td><td>0.0411 (0.0119)</td><td>0.0113 (0.0019)</td><td>0.0070 (0.0010)</td><td>0.0057 (0.0007)</td></tr><tr><td>Top-FT (NN)</td><td>0.0197 (0.0042)</td><td>0.0097 (0.0009)</td><td>0.0074 (0.0007)</td><td>0.0047 (0.0004)</td><td>0.0040 (0.0003)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0207 (0.0036)</td><td>0.0108 (0.0017)</td><td>0.0084 (0.0015)</td><td>0.0069 (0.0028)</td><td>0.0040 (0.0009)</td></tr><tr><td>Pooled (RF)</td><td>0.0444 (0.0071)</td><td>0.0290 (0.0018)</td><td>0.0259 (0.0013)</td><td>0.0235 (0.0007)</td><td>0.0236 (0.0005)</td></tr><tr><td>2-Stage (RF)</td><td>0.0350 (0.0055)</td><td>0.0196 (0.0014)</td><td>0.0156 (0.0009)</td><td>0.0112 (0.0005)</td><td>0.0099 (0.0004)</td></tr><tr><td>Separate (RF)</td><td>0.0696 (0.0104)</td><td>0.0353 (0.0020)</td><td>0.0280 (0.0014)</td><td>0.0210 (0.0007)</td><td>0.0189 (0.0005)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.0440 (0.0068)</td><td>0.0279 (0.0017)</td><td>0.0248 (0.0013)</td><td>0.0226 (0.0007)</td><td>0.0226 (0.0005)</td></tr></table>

Table 11. Average per-group MSE for Extra Scenario 1 over 50 independent trials at an SNR of 5 using neural networks. The bes performance in terms of lower MSE is bolded.
<table><tr><td>n</td><td>Group</td><td>Pooled</td><td>2-Stage</td><td>Separate</td><td>Top-FT</td><td>Pool-w-L</td></tr><tr><td rowspan="5">1000</td><td>1</td><td>0.0254 (0.0121)</td><td>0.0224 (0.0112)</td><td>0.4225 (0.2332)</td><td>0.0225 (0.0106)</td><td>0.0253 (0.0109)</td></tr><tr><td>2</td><td>0.0271 (0.0116)</td><td>0.0199 (0.0085)</td><td>0.5008 (0.1793)</td><td>0.0227 (0.0096)</td><td>0.0208 (0.0088)</td></tr><tr><td>3</td><td>0.0152 (0.0051)</td><td>0.0136 (0.0050)</td><td>0.4046 (0.1851)</td><td>0.0128 (0.0044)</td><td>0.0151 (0.0056)</td></tr><tr><td>4</td><td>0.0110 (0.0035)</td><td>0.0118 (0.0036)</td><td>0.3859 (0.0916)</td><td>0.0108 (0.0036)</td><td>0.0123 (0.0037)</td></tr><tr><td>5</td><td>0.0170 (0.0035)</td><td>0.0117 (0.0030)</td><td>0.4167 (0.0992)</td><td>0.0119 (0.0025)</td><td>0.0120 (0.0035)</td></tr><tr><td rowspan="5">5000</td><td>1</td><td>0.0213 (0.0064)</td><td>0.0112 (0.0034)</td><td>0.4110 (0.1464)</td><td>0.0157 (0.0042)</td><td>0.0131 (0.0032)</td></tr><tr><td>2</td><td>0.0180 (0.0046)</td><td>0.0085 (0.0024)</td><td>0.0135 (0.0027)</td><td>0.0114 (0.0026)</td><td>0.0100 (0.0022)</td></tr><tr><td>3</td><td>0.0101 (0.0026)</td><td>0.0075 (0.0017)</td><td>0.0107 (0.0018)</td><td>0.0073 (0.0011)</td><td>0.0088 (0.0019)</td></tr><tr><td>4</td><td>0.0069 (0.0010)</td><td>0.0064 (0.0012)</td><td>0.0085 (0.0011)</td><td>0.0060 (0.0008)</td><td>0.0070 (0.0014)</td></tr><tr><td>5</td><td>0.0131 (0.0049)</td><td>0.0053 (0.0014)</td><td>0.0074 (0.0010)</td><td>0.0053 (0.0010)</td><td>0.0060 (0.0011)</td></tr><tr><td rowspan="4">10000</td><td>1</td><td>0.0207 (0.0053)</td><td>0.0067 (0.0019)</td><td>0.0178 (0.0281)</td><td>0.0121 (0.0029)</td><td>0.0094 (0.0028)</td></tr><tr><td>2</td><td>0.0166 (0.0040)</td><td>0.0056 (0.0011)</td><td>0.0093 (0.0014)</td><td>0.0088 (0.0020)</td><td>0.0074 (0.0019)</td></tr><tr><td>3</td><td>0.0088 (0.0027)</td><td>0.0052 (0.0011)</td><td>0.0083 (0.0013)</td><td>0.0053 (0.0007)</td><td>0.0065 (0.0015)</td></tr><tr><td>4</td><td>0.0057 (0.0009)</td><td>0.0045 (0.0007)</td><td>0.0066 (0.0011)</td><td>0.0044 (0.0006)</td><td>0.0052 (0.0012)</td></tr><tr><td rowspan="3"></td><td>5</td><td>0.0112 (0.0049)</td><td>0.0038 (0.0009)</td><td>0.0060 (0.0012)</td><td>0.0036 (0.0005)</td><td>0.0047 (0.0016)</td></tr><tr><td>1</td><td>0.0221 (0.0093)</td><td>0.0039 (0.0007)</td><td>0.0085 (0.0014)</td><td>0.0076 (0.0017)</td><td>0.0073 (0.0032)</td></tr><tr><td>2</td><td>0.0181 (0.0079)</td><td>0.0038 (0.0006)</td><td>0.0058 (0.0009)</td><td>0.0055 (0.0009)</td><td>0.0061 (0.0028)</td></tr><tr><td rowspan="4">30000</td><td>3</td><td>0.0101 (0.0059)</td><td>0.0034 (0.0007)</td><td>0.0050 (0.0010)</td><td>0.0038 (0.0004)</td><td>0.0052 (0.0024)</td></tr><tr><td>4</td><td>0.0071 (0.0036)</td><td>0.0028 (0.0005)</td><td>0.0038 (0.0010)</td><td>0.0030 (0.0003)</td><td>0.0047 (0.0025)</td></tr><tr><td>5</td><td>0.0124 (0.0115)</td><td>0.0020 (0.0005)</td><td>0.0034 (0.0013)</td><td>0.0021 (0.0003)</td><td></td></tr><tr><td>1</td><td>0.0184 (0.0050)</td><td>0.0035 (0.0006)</td><td>0.0078 (0.0012)</td><td></td><td>0.0043 (0.0029)</td></tr><tr><td rowspan="5">50000</td><td></td><td></td><td></td><td></td><td>0.0064 (0.0012)</td><td>0.0040 (0.0010)</td></tr><tr><td>2</td><td>0.0146 (0.0046)</td><td>0.0030 (0.0005)</td><td>0.0048 (0.0010)</td><td>0.0050 (0.0009)</td><td>0.0033 (0.0011)</td></tr><tr><td>3</td><td>0.0067 (0.0031)</td><td>0.0025 (0.0004)</td><td>0.0039 (0.0009)</td><td>0.0033 (0.0003)</td><td>0.0029 (0.0010)</td></tr><tr><td>4</td><td>0.0042 (0.0014)</td><td>0.0021 (0.0003)</td><td>0.0033 (0.0011)</td><td>0.0027 (0.0003)</td><td>0.0023 (0.0008)</td></tr><tr><td>5</td><td>0.0098 (0.0040)</td><td>0.0016 (0.0003)</td><td>0.0025 (0.0008)</td><td>0.0017 (0.0002)</td><td>0.0021 (0.0008)</td></tr></table>

Table 12. Average MSE for Extra Scenario 2 with different SNR levels over 50 independent trials. The best performance in terms of lower MSE within each SNR group is bolded.
<table><tr><td>SNR</td><td>Model / Sample size</td><td>1000</td><td>5000</td><td>10000</td><td>30000</td><td>50000</td></tr><tr><td rowspan="11">10</td><td>Pooled (NN)</td><td>0.0223 (0.0040)</td><td>0.0163 (0.0014)</td><td>0.0153 (0.0016)</td><td>0.0147 (0.0021)</td><td>0.0133 (0.0011)</td></tr><tr><td>2-Stage (NN)</td><td>0.0163 (0.0036)</td><td>0.0065 (0.0008)</td><td>0.0045 (0.0004)</td><td>0.0028 (0.0003)</td><td>0.0020 (0.0002)</td></tr><tr><td>Separate (NN)</td><td>0.4959 (0.0843)</td><td>0.0482 (0.0115)</td><td>0.0079 (0.0007)</td><td>0.0043 (0.0004)</td><td>0.0037 (0.0004)</td></tr><tr><td>Top-FT (NN)</td><td>0.0181 (0.0038)</td><td>0.0082 (0.0011)</td><td>0.0065 (0.0008)</td><td>0.0039 (0.0004)</td><td>0.0031 (0.0005)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0157 (0.0026)</td><td>0.0076 (0.0010)</td><td>0.0054 (0.0009)</td><td>0.0059 (0.0046)</td><td>0.0024 (0.0006)</td></tr><tr><td>Pooled (RF)</td><td>0.0518 (0.0078)</td><td>0.0355 (0.0023)</td><td>0.0331 (0.0015)</td><td>0.0318 (0.0010)</td><td>0.0325 (0.0007)</td></tr><tr><td>2-Stage (RF)</td><td>0.0358 (0.0055)</td><td>0.0199 (0.0014)</td><td>0.0159 (0.0008)</td><td>0.0117 (0.0006)</td><td>0.0108 (0.0004)</td></tr><tr><td>Separate (RF)</td><td>0.0773 (0.0103)</td><td>0.0398 (0.0023)</td><td>0.0316 (0.0014)</td><td>0.0240 (0.0007)</td><td>0.0222 (0.0005)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.0510 (0.0076)</td><td>0.0338 (0.0022)</td><td>0.0308 (0.0014)</td><td>0.0291 (0.0009)</td><td>0.0295 (0.0006)</td></tr><tr><td>Pooled (NN)</td><td>0.0248 (0.0049)</td><td>0.0180 (0.0018)</td><td>0.0159 (0.0013)</td><td>0.0157 (0.0027)</td><td>0.0141 (0.0012)</td></tr><tr><td rowspan="8">5</td><td>2-Stage (NN)</td><td>0.0200 (0.0044)</td><td>0.0088 (0.0010)</td><td>0.0060 (0.0006)</td><td>0.0037 (0.0003)</td><td>0.0029 (0.0003)</td></tr><tr><td>Separate (NN)</td><td>0.4885 (0.0793)</td><td>0.0504 (0.0136)</td><td>0.0099 (0.0007)</td><td>0.0059 (0.0006)</td><td>0.0049 (0.0006)</td></tr><tr><td>Top-FT (NN)</td><td>0.0212 (0.0046)</td><td>0.0103 (0.0013)</td><td>0.0077 (0.0008)</td><td>0.0048 (0.0005)</td><td>0.0040 (0.0004)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0200 (0.0038)</td><td>0.0098 (0.0014)</td><td>0.0078 (0.0019)</td><td>0.0065 (0.0029)</td><td>0.0039 (0.0013)</td></tr><tr><td>Pooled (RF)</td><td>0.0524 (0.0076)</td><td>0.0357 (0.0021)</td><td>0.0329 (0.0016)</td><td>0.0312 (0.0009)</td><td>0.0317 (0.0007)</td></tr><tr><td>2-Stage (RF)</td><td>0.0376 (0.0058)</td><td>0.0209 (0.0014)</td><td>0.0167 (0.0009)</td><td>0.0122 (0.0006)</td><td>0.0110 (0.0005)</td></tr><tr><td>Separate (RF)</td><td>0.0793 (0.0115)</td><td>0.0404 (0.0024)</td><td>0.0320 (0.0014)</td><td>0.0239 (0.0007)</td><td>0.0219 (0.0005)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.0513 (0.0076)</td><td>0.0338 (0.0020)</td><td>0.0306 (0.0015)</td><td>0.0284 (0.0009)</td><td>0.0287 (0.0007)</td></tr><tr><td rowspan="11">2</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Pooled (NN)</td><td>0.0306 (0.0067)</td><td>0.0208 (0.0023)</td><td>0.0188 (0.0020)</td><td>0.0191 (0.0054)</td><td>0.0152 (0.0017)</td></tr><tr><td>2-Stage (NN)</td><td>0.0296 (0.0067)</td><td>0.0129 (0.0016)</td><td>0.0096 (0.0013)</td><td>0.0055 (0.0006)</td><td>0.0045 (0.0005)</td></tr><tr><td>Separate (NN)</td><td>0.5007 (0.0691)</td><td>0.0556 (0.0130)</td><td>0.0147 (0.0031)</td><td>0.0091 (0.0008)</td><td>0.0076 (0.0009)</td></tr><tr><td>Top-FT (NN)</td><td>0.0285 (0.0065)</td><td>0.0138 (0.0015)</td><td>0.0106 (0.0013)</td><td>0.0065 (0.0005)</td><td>0.0055 (0.0007)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0281 (0.0058)</td><td>0.0140 (0.0017)</td><td>0.0111 (0.0016)</td><td>0.0095 (0.0042)</td><td>0.0060 (0.0024)</td></tr><tr><td>Pooled (RF)</td><td>0.0563 (0.0088)</td><td>0.0374 (0.0022)</td><td>0.0338 (0.0017)</td><td>0.0307 (0.0009)</td><td>0.0306 (0.0008)</td></tr><tr><td>2-Stage (RF)</td><td>0.0439 (0.0075)</td><td>0.0249 (0.0017)</td><td>0.0199 (0.0012)</td><td>0.0141 (0.0006)</td><td>0.0123 (0.0005)</td></tr><tr><td>Separate (RF)</td><td>0.0851 (0.0128)</td><td>0.0434 (0.0029)</td><td>0.0344 (0.0017)</td><td>0.0252 (0.0009)</td><td>0.0226 (0.0006)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.0554 (0.0088)</td><td>0.0353 (0.0022)</td><td>0.0314 (0.0016)</td><td>0.0279 (0.0009)</td><td>0.0277 (0.0007)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 13. Average per-group MSE for Extra Scenario 2 over 50 independent trials at an SNR of 5 using neural networks. The bes performance in terms of lower MSE is bolded.
<table><tr><td>n</td><td>Group</td><td>Pooled</td><td>2-Stage</td><td>Separate</td><td>Top-FT</td><td>Pool-w-L</td></tr><tr><td rowspan="5">1000</td><td>1</td><td>0.0463 (0.0228)</td><td>0.0426 (0.0209)</td><td>0.5862 (0.3274)</td><td>0.0436 (0.0225)</td><td>0.0459 (0.0173)</td></tr><tr><td>2</td><td>0.0631 (0.0245)</td><td>0.0372 (0.0186)</td><td>0.6185 (0.2614)</td><td>0.0505 (0.0221)</td><td>0.0373 (0.0180)</td></tr><tr><td>3</td><td>0.0168 (0.0058)</td><td>0.0183 (0.0089)</td><td>0.4944 (0.1956)</td><td>0.0162 (0.0059)</td><td>0.0175 (0.0059)</td></tr><tr><td>4</td><td>0.0141 (0.0044)</td><td>0.0144 (0.0044)</td><td>0.4476 (0.1183)</td><td>0.0134 (0.0037)</td><td>0.0154 (0.0050)</td></tr><tr><td>5</td><td>0.0176 (0.0040)</td><td>0.0137 (0.0037)</td><td>0.4433 (0.1004)</td><td>0.0135 (0.0039)</td><td>0.0124 (0.0034)</td></tr><tr><td rowspan="5">5000</td><td>1</td><td>0.0393 (0.0091)</td><td>0.0198 (0.0088)</td><td>0.5857 (0.2020)</td><td>0.0259 (0.0095)</td><td>0.0205 (0.0049)</td></tr><tr><td>2</td><td>0.0465 (0.0112)</td><td>0.0127 (0.0034)</td><td>0.0201 (0.0038)</td><td>0.0216 (0.0057)</td><td>0.0160 (0.0032)</td></tr><tr><td>3</td><td>0.0100 (0.0022)</td><td>0.0086 (0.0022)</td><td>0.0131 (0.0023)</td><td>0.0082 (0.0012)</td><td>0.0099 (0.0023)</td></tr><tr><td>4</td><td>0.0097 (0.0022)</td><td>0.0077 (0.0018)</td><td>0.0101 (0.0017)</td><td>0.0069 (0.0010)</td><td>0.0081 (0.0016)</td></tr><tr><td>5</td><td>0.0136 (0.0037)</td><td>0.0059 (0.0011)</td><td>0.0084 (0.0013)</td><td>0.0066 (0.0012)</td><td>0.0064 (0.0013)</td></tr><tr><td rowspan="4">10000</td><td>1</td><td>0.0396 (0.0075)</td><td>0.0111 (0.0028)</td><td>0.0217 (0.0049)</td><td>0.0203 (0.0054)</td><td>0.0146 (0.0050)</td></tr><tr><td>2</td><td>0.0428 (0.0068)</td><td>0.0085 (0.0021)</td><td>0.0146 (0.0023)</td><td>0.0159 (0.0029)</td><td>0.0121 (0.0031)</td></tr><tr><td>3</td><td>0.0084 (0.0020)</td><td>0.0063 (0.0012)</td><td>0.0102 (0.0015)</td><td>0.0065 (0.0010)</td><td>0.0082 (0.0021)</td></tr><tr><td>4</td><td>0.0077 (0.0020)</td><td>0.0053 (0.0009)</td><td>0.0081 (0.0012)</td><td>0.0051 (0.0006)</td><td>0.0067 (0.0023)</td></tr><tr><td rowspan="3"></td><td>5</td><td>0.0114 (0.0032)</td><td>0.0043 (0.0008)</td><td>0.0069 (0.0011)</td><td>0.0047 (0.0008)</td><td>0.0054 (0.0019)</td></tr><tr><td>1</td><td>0.0380 (0.0077)</td><td>0.0067 (0.0010)</td><td>0.0127 (0.0022)</td><td>0.0134 (0.0031)</td><td>0.0105 (0.0041)</td></tr><tr><td>2</td><td>0.0418 (0.0122)</td><td>0.0055 (0.0008)</td><td>0.0084 (0.0012)</td><td>0.0094 (0.0019)</td><td>0.0084 (0.0038)</td></tr><tr><td rowspan="4">30000</td><td>3</td><td>0.0082 (0.0040)</td><td>0.0038 (0.0005)</td><td>0.0069 (0.0017)</td><td>0.0044 (0.0006)</td><td>0.0065 (0.0031)</td></tr><tr><td>4</td><td>0.0079 (0.0038)</td><td>0.0035 (0.0006)</td><td>0.0049 (0.0009)</td><td>0.0034 (0.0004)</td><td>0.0056 (0.0028)</td></tr><tr><td>5</td><td>0.0115 (0.0052)</td><td>0.0025 (0.0004)</td><td>0.0037 (0.0007)</td><td>0.0026 (0.0004)</td><td></td></tr><tr><td>1</td><td>0.0366 (0.0051)</td><td>0.0053 (0.0008)</td><td>0.0118 (0.0029)</td><td></td><td>0.0056 (0.0035)</td></tr><tr><td rowspan="4">50000</td><td></td><td></td><td></td><td></td><td>0.0108 (0.0022)</td><td>0.0067 (0.0024)</td></tr><tr><td>2 3</td><td>0.0398 (0.0075)</td><td>0.0041 (0.0008)</td><td>0.0069 (0.0019)</td><td>0.0083 (0.0020)</td><td>0.0054 (0.0021)</td></tr><tr><td>4</td><td>0.0062 (0.0021)</td><td>0.0031 (0.0006)</td><td>0.0052 (0.0011)</td><td>0.0035 (0.0005)</td><td>0.0043 (0.0014)</td></tr><tr><td>5</td><td>0.0064 (0.0020) 0.0101 (0.0029)</td><td>0.0028 (0.0006) 0.0019 (0.0003)</td><td>0.0040 (0.0010) 0.0034 (0.0014)</td><td>0.0029 (0.0003) 0.0021 (0.0004)</td><td>0.0035 (0.0013) 0.0027 (0.0013)</td></tr></table>

Table 14. Average MSE for Extra Scenario 3 with different SNR levels over 50 independent trials. The best performance in terms of lower MSE within each SNR group is bolded.
<table><tr><td>SNR</td><td>Model / Sample size</td><td>1000</td><td>5000</td><td>10000</td><td>30000</td><td>50000</td></tr><tr><td rowspan="11">10</td><td>Pooled (NN)</td><td>0.0536 (0.0095)</td><td>0.0375 (0.0031)</td><td>0.0345 (0.0026)</td><td>0.0374 (0.0114)</td><td>0.0298 (0.0025)</td></tr><tr><td>2-Stage (NN)</td><td>0.0393 (0.0090)</td><td>0.0164 (0.0020)</td><td>0.0112 (0.0011)</td><td>0.0068 (0.0007)</td><td>0.0050 (0.0004)</td></tr><tr><td>Separate (NN)</td><td>1.4760 (0.1806)</td><td>0.1328 (0.0264)</td><td>0.0200 (0.0018)</td><td>0.0112 (0.0012)</td><td>0.0091 (0.0010)</td></tr><tr><td>Top-FT (NN)</td><td>0.0393 (0.0082)</td><td>0.0191 (0.0018)</td><td>0.0150 (0.0013)</td><td>0.0106 (0.0010)</td><td>0.0086 (0.0008)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0480 (0.0100)</td><td>0.0181 (0.0021)</td><td>0.0142 (0.0025)</td><td>0.0143 (0.0062)</td><td>0.0064 (0.0014)</td></tr><tr><td>Pooled (RF)</td><td>0.0875 (0.0139)</td><td>0.0624 (0.0044)</td><td>0.0578 (0.0035)</td><td>0.0543 (0.0016)</td><td>0.0535 (0.0013)</td></tr><tr><td>2-Stage (RF)</td><td>0.0631 (0.0107)</td><td>0.0346 (0.0029)</td><td>0.0276 (0.0019)</td><td>0.0195 (0.0007)</td><td>0.0172 (0.0006)</td></tr><tr><td>Separate (RF)</td><td>0.1239 (0.0140)</td><td>0.0652 (0.0049)</td><td>0.0503 (0.0026)</td><td>0.0366 (0.0009)</td><td>0.0331 (0.0008)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.0843 (0.0138)</td><td>0.0567 (0.0042)</td><td>0.0510 (0.0029)</td><td>0.0460 (0.0013)</td><td>0.0450 (0.0011)</td></tr><tr><td>Pooled (NN)</td><td>0.0610 (0.0100)</td><td>0.0407 (0.0032)</td><td>0.0376 (0.0042)</td><td>0.0406 (0.0103)</td><td>0.0312 (0.0038)</td></tr><tr><td rowspan="8">5</td><td>2-Stage (NN)</td><td>0.0500 (0.0117)</td><td>0.0210 (0.0027)</td><td>0.0144 (0.0015)</td><td>0.0088 (0.0006)</td><td>0.0068 (0.0006)</td></tr><tr><td>Separate (NN)</td><td>1.4668 (0.1573)</td><td>0.1385 (0.0261)</td><td>0.0262 (0.0019)</td><td>0.0148 (0.0014)</td><td>0.0119 (0.0012)</td></tr><tr><td>Top-FT (NN)</td><td>0.0472 (0.0091)</td><td>0.0223 (0.0019)</td><td>0.0175 (0.0016)</td><td>0.0121 (0.0011)</td><td>0.0099 (0.0008)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0546 (0.0099)</td><td>0.0240 (0.0037)</td><td>0.0181 (0.0042)</td><td>0.0174 (0.0066)</td><td>0.0098 (0.0036)</td></tr><tr><td>Pooled (RF)</td><td>0.0910 (0.0142)</td><td>0.0641 (0.0044)</td><td>0.0588 (0.0037)</td><td>0.0542 (0.0016)</td><td>0.0530 (0.0014)</td></tr><tr><td>2-Stage (RF)</td><td>0.0688 (0.0118)</td><td>0.0376 (0.0030)</td><td>0.0301 (0.0020)</td><td>0.0209 (0.0008)</td><td>0.0182 (0.0007)</td></tr><tr><td>Separate (RF)</td><td>0.1296 (0.0150)</td><td>0.0684 (0.0051)</td><td>0.0530 (0.0028)</td><td>0.0380 (0.0010)</td><td>0.0339 (0.0009)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.0873 (0.0138)</td><td>0.0584 (0.0043)</td><td>0.0519 (0.0031)</td><td>0.0458 (0.0014)</td><td>0.0444 (0.0012)</td></tr><tr><td rowspan="11">2</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Pooled (NN)</td><td>0.0720 (0.0121)</td><td>0.0483 (0.0051)</td><td>0.0422 (0.0035)</td><td>0.0449 (0.0160)</td><td>0.0344 (0.0043)</td></tr><tr><td>2-Stage (NN)</td><td>0.0649 (0.0166)</td><td>0.0302 (0.0037)</td><td>0.0214 (0.0023)</td><td>0.0131 (0.0012)</td><td>0.0105 (0.0011)</td></tr><tr><td>Separate (NN)</td><td>1.4359 (0.1835)</td><td>0.1539 (0.0312)</td><td>0.0370 (0.0031)</td><td>0.0222 (0.0025)</td><td>0.0185 (0.0024)</td></tr><tr><td>Top-FT (NN)</td><td>0.0596 (0.0119)</td><td>0.0294 (0.0035)</td><td>0.0220 (0.0018)</td><td>0.0155 (0.0014)</td><td>0.0128 (0.0010)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0654 (0.0114)</td><td>0.0354 (0.0054)</td><td>0.0255 (0.0043)</td><td>0.0260 (0.0153)</td><td>0.0145 (0.0043)</td></tr><tr><td>Pooled (RF)</td><td>0.1018 (0.0160)</td><td>0.0701 (0.0051)</td><td>0.0624 (0.0040)</td><td>0.0556 (0.0019)</td><td>0.0532 (0.0015)</td></tr><tr><td>2-Stage (RF)</td><td>0.0853 (0.0142)</td><td>0.0473 (0.0041)</td><td>0.0372 (0.0023)</td><td>0.0254 (0.0010)</td><td>0.0215 (0.0009)</td></tr><tr><td>Separate (RF)</td><td>0.1460 (0.0177)</td><td>0.0796 (0.0057)</td><td>0.0615 (0.0031)</td><td>0.0433 (0.0011)</td><td>0.0377 (0.0011)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.0982 (0.0151)</td><td>0.0642 (0.0049)</td><td>0.0555 (0.0035)</td><td>0.0471 (0.0017)</td><td>0.0445 (0.0013)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 15. Average per-group MSE for Extra Scenario 3 over 50 independent trials at an SNR of 5 using neural networks. The bes performance in terms of lower MSE is bolded.
<table><tr><td>n</td><td>Group</td><td>Pooled</td><td>2-Stage</td><td>Separate</td><td>Top-FT</td><td>Pool-w-L</td></tr><tr><td rowspan="5">1000</td><td>1</td><td>0.2043 (0.0972)</td><td>0.0975 (0.0573)</td><td>1.8615 (0.9071)</td><td>0.1011 (0.0584)</td><td>0.1085 (0.0573)</td></tr><tr><td>2</td><td>0.0747 (0.0239)</td><td>0.0681 (0.0414)</td><td>1.6906 (0.5349)</td><td>0.0609 (0.0212)</td><td>0.0675 (0.0227)</td></tr><tr><td>3</td><td>0.0646 (0.0187)</td><td>0.0468 (0.0169)</td><td>1.3591 (0.4425)</td><td>0.0452 (0.0140)</td><td>0.0503 (0.0186)</td></tr><tr><td>4</td><td>0.0401 (0.0114)</td><td>0.0395 (0.0105)</td><td>1.3825 (0.3397)</td><td>0.0373 (0.0107)</td><td>0.0473 (0.0115)</td></tr><tr><td>5</td><td>0.0392 (0.0078)</td><td>0.0426 (0.0124)</td><td>1.4208 (0.3058)</td><td>0.0390 (0.0084)</td><td>0.0459 (0.0113)</td></tr><tr><td rowspan="5">5000</td><td>1</td><td>0.2000 (0.0467)</td><td>0.0556 (0.0215)</td><td>1.6041 (0.3878)</td><td>0.0699 (0.0161)</td><td>0.0528 (0.0175)</td></tr><tr><td>2</td><td>0.0496 (0.0129)</td><td>0.0249 (0.0051)</td><td>0.0507 (0.0119)</td><td>0.0272 (0.0050)</td><td>0.0295 (0.0061)</td></tr><tr><td>3</td><td>0.0421 (0.0116)</td><td>0.0199 (0.0054)</td><td>0.0368 (0.0076)</td><td>0.0191 (0.0037)</td><td>0.0235 (0.0046)</td></tr><tr><td>4</td><td>0.0203 (0.0049)</td><td>0.0168 (0.0032)</td><td>0.0272 (0.0043)</td><td>0.0157 (0.0020)</td><td>0.0201 (0.0045)</td></tr><tr><td>5</td><td>0.0203 (0.0029)</td><td>0.0165 (0.0033)</td><td>0.0260 (0.0047)</td><td>0.0179 (0.0027)</td><td>0.0193 (0.0038)</td></tr><tr><td rowspan="4">10000</td><td>1</td><td>0.1944 (0.0449)</td><td>0.0283 (0.0075)</td><td>0.0599 (0.0170)</td><td>0.0569 (0.0129)</td><td>0.0375 (0.0106)</td></tr><tr><td>2</td><td>0.0456 (0.0153)</td><td>0.0181 (0.0039)</td><td>0.0378 (0.0082)</td><td>0.0210 (0.0033)</td><td>0.0237 (0.0075)</td></tr><tr><td>3</td><td>0.0386 (0.0161)</td><td>0.0140 (0.0027)</td><td>0.0247 (0.0043)</td><td>0.0143 (0.0023)</td><td>0.0170 (0.0046)</td></tr><tr><td>4</td><td>0.0178 (0.0073)</td><td>0.0119 (0.0016)</td><td>0.0204 (0.0041)</td><td>0.0120 (0.0013)</td><td>0.0152 (0.0037)</td></tr><tr><td rowspan="3"></td><td>5</td><td>0.0179 (0.0041)</td><td>0.0124 (0.0020)</td><td>0.0204 (0.0030)</td><td>0.0145 (0.0018)</td><td>0.0150 (0.0038)</td></tr><tr><td>1</td><td>0.2025 (0.0765)</td><td>0.0161 (0.0030)</td><td>0.0342 (0.0048)</td><td>0.0455 (0.0090)</td><td>0.0304 (0.0089)</td></tr><tr><td>2</td><td>0.0503 (0.0297)</td><td>0.0108 (0.0017)</td><td>0.0205 (0.0036)</td><td>0.0152 (0.0021)</td><td>0.0207 (0.0086)</td></tr><tr><td rowspan="4">30000</td><td>3</td><td>0.0395 (0.0333)</td><td>0.0088 (0.0015)</td><td>0.0140 (0.0026)</td><td>0.0094 (0.0014)</td><td>0.0162 (0.0075)</td></tr><tr><td>4</td><td>0.0210 (0.0169)</td><td>0.0078 (0.0012)</td><td>0.0119 (0.0026)</td><td>0.0083 (0.0010)</td><td>0.0154 (0.0072)</td></tr><tr><td>5</td><td>0.0207 (0.0101)</td><td>0.0073 (0.0008)</td><td>0.0116 (0.0025)</td><td>0.0088 (0.0011)</td><td></td></tr><tr><td>1</td><td>0.1874 (0.0393)</td><td>0.0128 (0.0020)</td><td></td><td></td><td>0.0157 (0.0067)</td></tr><tr><td rowspan="5">50000</td><td></td><td></td><td></td><td>0.0284 (0.0049)</td><td>0.0394 (0.0076)</td><td>0.0189 (0.0049)</td></tr><tr><td>2</td><td>0.0393 (0.0140)</td><td>0.0082 (0.0014)</td><td>0.0156 (0.0025)</td><td>0.0123 (0.0016)</td><td>0.0123 (0.0052)</td></tr><tr><td>3</td><td>0.0316 (0.0165)</td><td>0.0068 (0.0011)</td><td>0.0123 (0.0030)</td><td>0.0076 (0.0010)</td><td>0.0094 (0.0042)</td></tr><tr><td>4</td><td>0.0123 (0.0079)</td><td>0.0063 (0.0010)</td><td>0.0096 (0.0020)</td><td>0.0069 (0.0008)</td><td>0.0090 (0.0040)</td></tr><tr><td>5</td><td>0.0116 (0.0039)</td><td>0.0053 (0.0007)</td><td>0.0088 (0.0023)</td><td>0.0069 (0.0007)</td><td>0.0079 (0.0035)</td></tr></table>

Table 16. Average MSE for Extra Scenario 4 with different SNR levels over 50 independent trials. The best performance in terms of lower MSE within each SNR group is bolded.
<table><tr><td>SNR</td><td>Model / Sample size</td><td>1000</td><td>5000</td><td>10000</td><td>30000</td><td>50000</td></tr><tr><td rowspan="11">10</td><td>Pooled (NN)</td><td>0.0159 (0.0026)</td><td>0.0089 (0.0009)</td><td>0.0076 (0.0010)</td><td>0.0075 (0.0017)</td><td>0.0060 (0.0008)</td></tr><tr><td>2-Stage (NN)</td><td>0.0144 (0.0029)</td><td>0.0058 (0.0007)</td><td>0.0042 (0.0005)</td><td>0.0025 (0.0002)</td><td>0.0019 (0.0001)</td></tr><tr><td>Separate (NN)</td><td>0.3787 (0.0438)</td><td>0.0354 (0.0068)</td><td>0.0077 (0.0006)</td><td>0.0043 (0.0004)</td><td>0.0037 (0.0004)</td></tr><tr><td>Top-FT (NN)</td><td>0.0145 (0.0025)</td><td>0.0066 (0.0007)</td><td>0.0052 (0.0006)</td><td>0.0035 (0.0004)</td><td>0.0028 (0.0003)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0192 (0.0049)</td><td>0.0068 (0.0008)</td><td>0.0054 (0.0015)</td><td>0.0052 (0.0032)</td><td>0.0027 (0.0009)</td></tr><tr><td>Pooled (RF)</td><td>0.0323 (0.0039)</td><td>0.0217 (0.0013)</td><td>0.0193 (0.0009)</td><td>0.0179 (0.0006)</td><td>0.0179 (0.0005)</td></tr><tr><td>2-Stage (RF)</td><td>0.0253 (0.0034)</td><td>0.0135 (0.0009)</td><td>0.0105 (0.0006)</td><td>0.0079 (0.0003)</td><td>0.0073 (0.0003)</td></tr><tr><td>Separate (RF)</td><td>0.0501 (0.0064)</td><td>0.0285 (0.0019)</td><td>0.0229 (0.0011)</td><td>0.0174 (0.0005)</td><td>0.0159 (0.0004)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.0322 (0.0041)</td><td>0.0214 (0.0013)</td><td>0.0190 (0.0009)</td><td>0.0176 (0.0006)</td><td>0.0176 (0.0005)</td></tr><tr><td>Pooled (NN)</td><td>0.0202 (0.0047)</td><td>0.0101 (0.0011)</td><td>0.0087 (0.0010)</td><td>0.0089 (0.0029)</td><td>0.0066 (0.0009)</td></tr><tr><td rowspan="8">5</td><td>2-Stage (NN)</td><td>0.0194 (0.0049)</td><td>0.0076 (0.0008)</td><td>0.0056 (0.0006)</td><td>0.0034 (0.0003)</td><td>0.0026 (0.0002)</td></tr><tr><td>Separate (NN)</td><td>0.3739 (0.0474)</td><td>0.0381 (0.0071)</td><td>0.0101 (0.0009)</td><td>0.0057 (0.0006)</td><td>0.0048 (0.0006)</td></tr><tr><td>Top-FT (NN)</td><td>0.0191 (0.0048)</td><td>0.0080 (0.0008)</td><td>0.0063 (0.0007)</td><td>0.0041 (0.0004)</td><td>0.0034 (0.0003)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0244 (0.0047)</td><td>0.0092 (0.0014)</td><td>0.0067 (0.0009)</td><td>0.0061 (0.0029)</td><td>0.0032 (0.0006)</td></tr><tr><td>Pooled (RF)</td><td>0.0332 (0.0042)</td><td>0.0219 (0.0012)</td><td>0.0195 (0.0009)</td><td>0.0177 (0.0006)</td><td>0.0176 (0.0005)</td></tr><tr><td>2-Stage (RF)</td><td>0.0270 (0.0037)</td><td>0.0145 (0.0010)</td><td>0.0114 (0.0007)</td><td>0.0083 (0.0003)</td><td>0.0075 (0.0003)</td></tr><tr><td>Separate (RF)</td><td>0.0514 (0.0066)</td><td>0.0292 (0.0021)</td><td>0.0235 (0.0011)</td><td>0.0176 (0.0005)</td><td>0.0159 (0.0004)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.0331 (0.0043)</td><td>0.0216 (0.0012)</td><td>0.0191 (0.0009)</td><td>0.0173 (0.0006)</td><td>0.0173 (0.0005)</td></tr><tr><td rowspan="11">2</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Pooled (NN)</td><td>0.0266 (0.0051)</td><td>0.0132 (0.0016)</td><td>0.0110 (0.0016)</td><td>0.0103 (0.0033)</td><td>0.0079 (0.0016)</td></tr><tr><td>2-Stage (NN)</td><td>0.0280 (0.0053)</td><td>0.0114 (0.0015)</td><td>0.0081 (0.0010)</td><td>0.0050 (0.0004)</td><td>0.0040 (0.0003)</td></tr><tr><td>Separate (NN)</td><td>0.3740 (0.0515)</td><td>0.0448 (0.0090)</td><td>0.0153 (0.0016)</td><td>0.0087 (0.0009)</td><td>0.0070 (0.0007)</td></tr><tr><td>Top-FT (NN)</td><td>0.0259 (0.0052)</td><td>0.0112 (0.0012)</td><td>0.0083 (0.0009)</td><td>0.0055 (0.0005)</td><td>0.0046 (0.0004)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0302 (0.0045)</td><td>0.0140 (0.0025)</td><td>0.0099 (0.0016)</td><td>0.0093 (0.0056)</td><td>0.0052 (0.0014)</td></tr><tr><td>Pooled (RF)</td><td>0.0362 (0.0053)</td><td>0.0234 (0.0014)</td><td>0.0206 (0.0010)</td><td>0.0178 (0.0006)</td><td>0.0173 (0.0005)</td></tr><tr><td>2-Stage (RF)</td><td>0.0321 (0.0049)</td><td>0.0178 (0.0012)</td><td>0.0142 (0.0008)</td><td>0.0098 (0.0003)</td><td>0.0086 (0.0003)</td></tr><tr><td>Separate (RF)</td><td>0.0571 (0.0074)</td><td>0.0324 (0.0022)</td><td>0.0256 (0.0012)</td><td>0.0189 (0.0006)</td><td>0.0169 (0.0004)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.0359 (0.0055)</td><td>0.0230 (0.0014)</td><td>0.0201 (0.0011)</td><td>0.0174 (0.0006)</td><td>0.0169 (0.0005)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 17. Average per-group MSE for Extra Scenario 4 over 50 independent trials at an SNR of 5 using neural networks. The bes performance in terms of lower MSE is bolded.
<table><tr><td>n</td><td>Group</td><td>Pooled</td><td>2-Stage</td><td>Separate</td><td>Top-FT</td><td>Pool-w-L</td></tr><tr><td rowspan="5">1000</td><td>1</td><td>0.0294 (0.0171)</td><td>0.0317 (0.0186)</td><td>0.4555 (0.2110)</td><td>0.0294 (0.0169)</td><td>0.0454 (0.0236)</td></tr><tr><td>2</td><td>0.0337 (0.0126)</td><td>0.0298 (0.0134)</td><td>0.4091 (0.1315)</td><td>0.0306 (0.0126)</td><td>0.0342 (0.0103)</td></tr><tr><td>3</td><td>0.0190 (0.0071)</td><td>0.0190 (0.0074)</td><td>0.3641 (0.1010)</td><td>0.0183 (0.0065)</td><td>0.0255 (0.0080)</td></tr><tr><td>4</td><td>0.0151 (0.0046)</td><td>0.0154 (0.0049)</td><td>0.3518 (0.0806)</td><td>0.0149 (0.0046)</td><td>0.0205 (0.0065)</td></tr><tr><td>5</td><td>0.0173 (0.0048)</td><td>0.0161 (0.0043)</td><td>0.3653 (0.0886)</td><td>0.0158 (0.0044)</td><td>0.0183 (0.0051)</td></tr><tr><td rowspan="5">5000</td><td>1</td><td>0.0160 (0.0045)</td><td>0.0148 (0.0046)</td><td>0.3776 (0.1010)</td><td>0.0147 (0.0046)</td><td>0.0178 (0.0062)</td></tr><tr><td>2</td><td>0.0193 (0.0046)</td><td>0.0120 (0.0024)</td><td>0.0251 (0.0064)</td><td>0.0150 (0.0032)</td><td>0.0142 (0.0030)</td></tr><tr><td>3</td><td>0.0083 (0.0019)</td><td>0.0074 (0.0016)</td><td>0.0152 (0.0033)</td><td>0.0071 (0.0014)</td><td>0.0096 (0.0019)</td></tr><tr><td>4</td><td>0.0065 (0.0014)</td><td>0.0060 (0.0010)</td><td>0.0116 (0.0027)</td><td>0.0058 (0.0009)</td><td>0.0074 (0.0013)</td></tr><tr><td>5</td><td>0.0090 (0.0022)</td><td>0.0058 (0.0011)</td><td>0.0094 (0.0019)</td><td>0.0062 (0.0010)</td><td>0.0066 (0.0015)</td></tr><tr><td rowspan="4">10000</td><td>1</td><td>0.0146 (0.0039)</td><td>0.0102 (0.0028)</td><td>0.0240 (0.0064)</td><td>0.0124 (0.0032)</td><td>0.0122 (0.0028)</td></tr><tr><td>2</td><td>0.0178 (0.0045)</td><td>0.0090 (0.0016)</td><td>0.0163 (0.0033)</td><td>0.0121 (0.0025)</td><td>0.0106 (0.0024)</td></tr><tr><td>3</td><td>0.0070 (0.0021)</td><td>0.0053 (0.0009)</td><td>0.0100 (0.0013)</td><td>0.0053 (0.0009)</td><td>0.0067 (0.0013)</td></tr><tr><td>4</td><td>0.0053 (0.0010)</td><td>0.0046 (0.0008)</td><td>0.0076 (0.0011)</td><td>0.0043 (0.0005)</td><td>0.0055 (0.0012)</td></tr><tr><td rowspan="3"></td><td>5</td><td>0.0076 (0.0022)</td><td>0.0044 (0.0010)</td><td>0.0068 (0.0012)</td><td>0.0050 (0.0007)</td><td>0.0050 (0.0010)</td></tr><tr><td>1</td><td>0.0141 (0.0042)</td><td>0.0068 (0.0011)</td><td>0.0129 (0.0025)</td><td>0.0093 (0.0016)</td><td>0.0095 (0.0036)</td></tr><tr><td>2</td><td>0.0164 (0.0061)</td><td>0.0053 (0.0008)</td><td>0.0088 (0.0011)</td><td>0.0082 (0.0014)</td><td>0.0091 (0.0043)</td></tr><tr><td rowspan="4">30000</td><td>3</td><td>0.0066 (0.0039)</td><td>0.0033 (0.0004)</td><td>0.0061 (0.0009)</td><td>0.0035 (0.0005)</td><td>0.0060 (0.0026)</td></tr><tr><td>4</td><td>0.0058 (0.0033)</td><td>0.0027 (0.0003)</td><td>0.0043 (0.0011)</td><td>0.0028 (0.0003)</td><td>0.0052 (0.0030)</td></tr><tr><td>5</td><td>0.0086 (0.0054)</td><td>0.0025 (0.0003)</td><td>0.0038 (0.0009)</td><td>0.0030 (0.0005)</td><td></td></tr><tr><td>1</td><td>0.0122 (0.0025)</td><td>0.0050 (0.0008)</td><td>0.0110 (0.0021)</td><td></td><td>0.0049 (0.0028)</td></tr><tr><td rowspan="5">50000</td><td></td><td></td><td></td><td></td><td>0.0084 (0.0014)</td><td>0.0062 (0.0015)</td></tr><tr><td>2</td><td>0.0149 (0.0034)</td><td>0.0040 (0.0006)</td><td>0.0071 (0.0009)</td><td>0.0070 (0.0013)</td><td>0.0051 (0.0009)</td></tr><tr><td>3</td><td>0.0049 (0.0020)</td><td>0.0026 (0.0003)</td><td>0.0050 (0.0012)</td><td>0.0027 (0.0003)</td><td>0.0033 (0.0007)</td></tr><tr><td>4</td><td>0.0034 (0.0011)</td><td>0.0021 (0.0004)</td><td>0.0036 (0.0009)</td><td>0.0022 (0.0003)</td><td>0.0026 (0.0007)</td></tr><tr><td>5</td><td>0.0057 (0.0025)</td><td>0.0020 (0.0003)</td><td>0.0036 (0.0013)</td><td>0.0024 (0.0004)</td><td>0.0024 (0.0006)</td></tr></table>

Table 18. Average MSE for Scenario 3 (high-dimensional) with different SNR levels over 50 independent trials. The best performance in terms of lower MSE within each SNR group is bolded.
<table><tr><td>Std</td><td>Model / Sample size</td><td>5000</td><td>10000</td><td>30000</td><td>50000</td></tr><tr><td rowspan="6">0.1</td><td>Pooled (NN)</td><td>3.5772 (0.4119)</td><td>2.8049 (0.3413)</td><td>2.4122 (0.3941)</td><td>1.8712 (0.2405)</td></tr><tr><td>2-Stage (NN)</td><td>3.4045 (0.3687)</td><td>2.5317 (0.2908)</td><td>1.7826 (0.2254)</td><td>1.4377 (0.1909)</td></tr><tr><td>Separate (NN)</td><td>18.4700 (1.3653)</td><td>16.7562 (1.0588)</td><td>7.3540 (0.6464)</td><td>5.1880 (0.3722)</td></tr><tr><td>Top-FT (NN)</td><td>3.4864 (0.3970)</td><td>2.6559 (0.3170)</td><td>2.0301 (0.2903)</td><td>1.6499 (0.2149)</td></tr><tr><td>Pool-w-L (NN)</td><td>3.7586 (0.4649)</td><td>2.6799 (0.3126)</td><td>2.0207 (0.3539)</td><td>1.5020 (0.2305)</td></tr><tr><td>ptLasso</td><td>19.2066 (1.1992)</td><td>18.1132 (1.1836)</td><td>17.7140 (0.8321)</td><td>17.5210 (0.6882)</td></tr><tr><td rowspan="6">1</td><td>Pooled (NN)</td><td>4.0795 (0.4407)</td><td>3.2150 (0.3589)</td><td>2.6265 (0.3937)</td><td>2.1231 (0.2551)</td></tr><tr><td>2-Stage (NN)</td><td>3.9457 (0.4051)</td><td>2.9710 (0.3132)</td><td>2.1172 (0.2096)</td><td>1.7750 (0.1908)</td></tr><tr><td>Separate (NN)</td><td>18.5199 (1.4433)</td><td>16.8376 (1.1317)</td><td>8.0144 (0.6275)</td><td>5.7036 (0.4192)</td></tr><tr><td>Top-FT (NN)</td><td>3.9666 (0.4319)</td><td>3.0476 (0.3328)</td><td>2.2643 (0.2707)</td><td>1.8705 (0.2096)</td></tr><tr><td>Pool-w-L (NN)</td><td>4.3708 (0.4846)</td><td>3.1855 (0.3481)</td><td>2.3404 (0.2989)</td><td>1.8470 (0.2286)</td></tr><tr><td>ptLasso</td><td>20.1427 (1.3865)</td><td>19.1391 (1.2099)</td><td>18.7576 (0.8002)</td><td>18.5122 (0.7150)</td></tr></table>

Table 19. Average MSE for Scenario 4 (high-dimensional) with different SNR levels over 50 independent trials. The best performance in terms of lower MSE within each SNR group is bolded.
<table><tr><td>Std</td><td>Model / Sample size</td><td>5000</td><td>10000</td><td>30000</td><td>50000</td></tr><tr><td rowspan="6">0.1</td><td>Pooled (NN)</td><td>8.5086 (0.8878)</td><td>5.8025 (0.5703)</td><td>3.8477 (0.6878)</td><td>2.9454 (0.3160)</td></tr><tr><td>2-Stage (NN)</td><td>8.4782 (0.8924)</td><td>5.6875 (0.5791)</td><td>3.2175 (0.3005)</td><td>2.5256 (0.2120)</td></tr><tr><td>Separate (NN)</td><td>22.0682 (1.5491)</td><td>21.4115 (1.0980)</td><td>14.8534 (0.6357)</td><td>13.0925 (0.7355)</td></tr><tr><td>Top-FT (NN)</td><td>8.5038 (0.8945)</td><td>5.7637 (0.5768)</td><td>3.5023 (0.3075)</td><td>2.8225 (0.2286)</td></tr><tr><td>Pool-w-L (NN)</td><td>8.5776 (0.8611)</td><td>5.8783 (0.5824)</td><td>3.8614 (0.7056)</td><td>2.9559 (0.2845)</td></tr><tr><td>ptLasso</td><td>21.7040 (1.6185)</td><td>19.7228 (0.9760)</td><td>19.2816 (0.8314)</td><td>19.2234 (0.7995)</td></tr><tr><td rowspan="6">1</td><td>Pooled (NN)</td><td>9.1348 (0.9145)</td><td>6.4555 (0.6181)</td><td>4.3988 (0.5348)</td><td>3.5638 (0.2752)</td></tr><tr><td>2-Stage (NN)</td><td>9.0841 (0.9062)</td><td>6.3287 (0.5981)</td><td>3.8919 (0.3088)</td><td>3.1943 (0.2400)</td></tr><tr><td>Separate (NN)</td><td>22.3329 (1.5473)</td><td>21.5863 (0.9121)</td><td>15.1587 (0.5992)</td><td>13.5028 (0.7384)</td></tr><tr><td>Top-FT (NN)</td><td>9.1156 (0.9069)</td><td>6.4022 (0.5908)</td><td>4.1114 (0.3084)</td><td>3.4415 (0.2636)</td></tr><tr><td>Pool-w-L (NN)</td><td>9.2112 (0.9200)</td><td>6.4724 (0.6281)</td><td>4.5905 (0.6670)</td><td>3.5074 (0.3140)</td></tr><tr><td>ptLasso</td><td>21.7648 (1.6522)</td><td>19.7493 (0.9892)</td><td>19.3060 (0.8403)</td><td>19.2003 (0.8071)</td></tr></table>

## F.2. Detailed Configurations for Numerical Simulation

This section presents the detailed configurations used in our numerical simulations. Regarding the neural network setups, the Pooled, Pool-w-L, and Top-FT strategies each employ a single neural network instance, whereas the 2-Stage strategy involves two networks: one trained on the full dataset and another trained separately for each group. Within each scenario, all NN instances share the same network architecture.

For Scenarios 1,2 and Extra Scenarios 1–4, which correspond to the low-dimensional settings, each NN is implemented as a four-layer MLP with 64 hidden units per layer, using ReLU activations after the first three layers. For the high-dimensional settings, we adopt a five-layer MLP with 128 hidden units per layer for Scenario 3, and an eight-layer MLP with 256 hidden units per layer for Scenario 4. All neural networks are trained using the Adam optimizer with a learning rate of $1 0 ^ { - 3 }$ and a batch size of 256. For each setting, data are randomly partitioned using a group-wise stratified split into training (70%), validation (15%), and testing (15%) subsets. Early stopping is applied with a patience of 10 epochs, based on validation performance.

For the competing methods, random forests are implemented with 100 tree estimators and a maximum tree depth of 10, while other hyperparameters follow the default settings from the sklearn.ensemble package. For ptLasso, we perform 10-fold cross-validation over $\alpha \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 \}$ , keeping other parameters at their default values.

## F.3. Experiments with Highly Unbalanced Groups

To further investigate the robustness of our proposed framework under highly unbalanced groups, we examine the six low-dimensional scenarios previously introduced with unbalanced group assignment proportions of [0.02, 0.25, 0.25, 0.24, 0.24], where the first group represents only 2% of the total sample. Here we focus on the DNN realization and present our experimental results at SNR=5 across total sample sizes n 5000, 10000, 30000, 50000 . Table 20 summarizes the MSE performance of the unbalanced group (Group 1) over 50 independent Monte Carlo trials across the six simulation scenarios. The results demonstrate that as the sample size increases, the proposed two-stage transfer learning method yields increasingly pronounced performance improvements for the extremely low-proportion group, achieving the best performance in most cases.

Table 20. Average MSE for the unbalanced group (Group 1) across six low-dimensional scenarios at SNR = 5 over 50 independent trials. The best performance for each sample size is highlighted in bold.
<table><tr><td></td><td>Model / Sample size</td><td>5000</td><td>10000</td><td>30000</td><td>50000</td></tr><tr><td rowspan="5">Scenario 1</td><td>Pooled (NN)</td><td>0.0339 (0.0164)</td><td>0.0333 (0.0113)</td><td>0.0297 (0.0047)</td><td>0.0308 (0.0065)</td></tr><tr><td>2-Stage (NN)</td><td>0.0327 (0.0179)</td><td>0.0184 (0.0061)</td><td>0.0088 (0.0018)</td><td>0.0060 (0.0009)</td></tr><tr><td>Separate (NN)</td><td>0.6937 (0.4308)</td><td>0.7003 (0.3028)</td><td>0.0236 (0.0047)</td><td>0.0176 (0.0020)</td></tr><tr><td>Top-FT (NN)</td><td>0.0236 (0.0099)</td><td>0.0232 (0.0099)</td><td>0.0132 (0.0039)</td><td>0.0131 (0.0041)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0334 (0.0149)</td><td>0.0184 (0.0050)</td><td>0.0155 (0.0076)</td><td>0.0108 (0.0038)</td></tr><tr><td rowspan="5">Scenario 2</td><td>Pooled (NN)</td><td>0.0488 (0.0211)</td><td>0.0456 (0.0144)</td><td>0.0437 (0.0124)</td><td>0.0375 (0.0086)</td></tr><tr><td>2-Stage (NN)</td><td>0.0498 (0.0210)</td><td>0.0415 (0.0199)</td><td>0.0250 (0.0074)</td><td>0.0185 (0.0050)</td></tr><tr><td>Separate (NN)</td><td>0.6019 (0.2203)</td><td>0.5966 (0.1470)</td><td>0.1108 (0.0858)</td><td>0.0571 (0.0147)</td></tr><tr><td>Top-FT (NN)</td><td>0.0514 (0.0257)</td><td>0.0410 (0.0121)</td><td>0.0358 (0.0073)</td><td>0.0312 (0.0067)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0918 (0.0427)</td><td>0.0594 (0.0255)</td><td>0.0413 (0.0223)</td><td>0.0286 (0.0069)</td></tr><tr><td rowspan="5">Extra Scenario 1</td><td>Pooled (NN)</td><td>0.0206 (0.0103)</td><td>0.0186 (0.0061)</td><td>0.0169 (0.0061)</td><td>0.0163 (0.0049)</td></tr><tr><td>2-Stage (NN)</td><td>0.0160 (0.0078)</td><td>0.0112 (0.0038)</td><td>0.0053 (0.0020)</td><td>0.0042 (0.0011)</td></tr><tr><td>Separate (NN)</td><td>0.4917 (0.2299)</td><td>0.4940 (0.1773)</td><td>0.0145 (0.0032)</td><td>0.0113 (0.0016)</td></tr><tr><td>Top-FT (NN)</td><td>0.0156 (0.0069)</td><td>0.0146 (0.0051)</td><td>0.0099 (0.0021)</td><td>0.0085 (0.0020)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0200 (0.0096)</td><td>0.0128 (0.0047)</td><td>0.0095 (0.0050)</td><td>0.0060 (0.0018)</td></tr><tr><td rowspan="5">Extra Scenario 2</td><td>Pooled (NN)</td><td>0.0344 (0.0184)</td><td>0.0337 (0.0119)</td><td>0.0297 (0.0079)</td><td>0.0291 (0.0043)</td></tr><tr><td>2-Stage (NN)</td><td>0.0355 (0.0156)</td><td>0.0167 (0.0064)</td><td>0.0089 (0.0015)</td><td>0.0064 (0.0016)</td></tr><tr><td>Separate (NN)</td><td>0.6453 (0.3275)</td><td>0.6625 (0.3514)</td><td>0.0228 (0.0029)</td><td>0.0177 (0.0032)</td></tr><tr><td>Top-FT (NN)</td><td>0.0255 (0.0147)</td><td>0.0187 (0.0086)</td><td>0.0133 (0.0028)</td><td>0.0120 (0.0029)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0321 (0.0112)</td><td>0.0182 (0.0035)</td><td>0.0116 (0.0044)</td><td>0.0098 (0.0038)</td></tr><tr><td rowspan="5">Extra Scenario 3</td><td>Pooled (NN)</td><td>0.1963 (0.0784)</td><td>0.1868 (0.0610)</td><td>0.2073 (0.0686)</td><td>0.2012 (0.0375)</td></tr><tr><td>2-Stage (NN)</td><td>0.0696 (0.0393)</td><td>0.0545 (0.0231)</td><td>0.0231 (0.0061)</td><td>0.0182 (0.0049)</td></tr><tr><td>Separate (NN)</td><td>1.9146 (0.7185)</td><td>1.8240 (0.4825)</td><td>0.0644 (0.0147)</td><td>0.0505 (0.0119)</td></tr><tr><td>Top-FT (NN)</td><td>0.0747 (0.0376)</td><td>0.0703 (0.0223)</td><td>0.0534 (0.0134)</td><td>0.0494 (0.0102)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0710 (0.0376)</td><td>0.0562 (0.0214)</td><td>0.0407 (0.0142)</td><td>0.0267 (0.0074)</td></tr><tr><td rowspan="5">Extra Scenario 4</td><td>Pooled (NN)</td><td>0.0149 (0.0070)</td><td>0.0143 (0.0062)</td><td>0.0121 (0.0036)</td><td>0.0100 (0.0017)</td></tr><tr><td>2-Stage (NN)</td><td>0.0174 (0.0098)</td><td>0.0124 (0.0060)</td><td>0.0081 (0.0020)</td><td>0.0065 (0.0013)</td></tr><tr><td>Separate (NN)</td><td>0.4438 (0.2062)</td><td>0.4219 (0.1147)</td><td>0.0263 (0.0059)</td><td>0.0197 (0.0063)</td></tr><tr><td>Top-FT (NN)</td><td>0.0155 (0.0076)</td><td>0.0123 (0.0050)</td><td>0.0084 (0.0017)</td><td>0.0078 (0.0015)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0283 (0.0128)</td><td>0.0180 (0.0068)</td><td>0.0122 (0.0047)</td><td>0.0083 (0.0023)</td></tr></table>

## G. Details of Real Data Experiments

## G.1. Beijing PM2.5 Dataset

In this section, we evaluate our framework on an air quality prediction task using an open-source dataset provided by (Chen, 2015) and following the study of (Liang et al., 2015). This dataset records hourly meteorological data from Beijing, China over several years, with particulate matter with a diameter of 2.5 micrometers (PM2.5), a central index for air pollution levels, as the predicted response variable. Specifically, the dataset spans January 1, 2010 to December 31, 2014, containing 43824 hourly records after removing missing values. Features include six numerical meteorological covariates (dew point, temperature, pressure, cumulated wind speed, and cumulated hours of snow and rain) and one categorical variable for dominant wind direction with four categories: northeast (NE), northwest (NW), south (S), and calm (CV). Due to Beijing’s geography with mountains to the north and west and industrial zones to the south and east, wind direction significantly influences air quality by determining whether clean air flows in or pollutants become trapped (Liang et al., 2015). This makes wind direction a natural group label, as the PM2.5-meteorological relationship varies substantially across wind directions. We conduct transfer learning based on these groups, with 4997, 14150, 9387, and 15290 samples for NE, NW, CV, and S winds, respectively.

We specify a similar autoregressive model to that stated in (Liang et al., 2015) equation (4.3) for predicting PM2.5 as

follows:

$$
y _ { t } = f _ { z _ { t } } ( x _ { t } , y _ { t - 1 } ) + \epsilon _ { t } ,
$$

where $y _ { t }$ is the PM2.5 concentration at hour $t ,$ and $y _ { t - 1 }$ is the lag-1 PM2.5 value to capture temporal dependencies. The covariate vector $x _ { t }$ includes the six numerical meteorological variables, along with four additional variables from sine and cosine transformations of the hour-of-day and month-of-year to help capture diurnal and seasonal patterns. The group label z corresponds to one of the four dominant wind directions.

We follow a similar setting to the low-dimensional numerical experiments, where we compare DNN and RF as estimators across various learning strategies. To prevent potential data leakage risks in time series prediction tasks, we use data from 2014 as the hold-out test set. We repeat the entire procedure for 20 independent trials using different random seeds and report the average test MSE across trials. For the neural network implementations, each model instance is a five-layer MLP with 64 neurons per layer, using ReLU activation functions after the first four layers. All networks are trained with the Adam optimizer at a learning rate of $1 0 ^ { - 3 }$ , a batch size of 256, and early stopping with a patience of 10 epochs based on validation performance. For the random forest models, we use 100 decision trees with a maximum depth of 10, while other hyperparameters remain at their default settings. All input features are standardized to have zero mean and unit variance.

Table 21. Overall and per-group average MSE for PM2.5 models over 20 independent trials. The best performance in terms of lower MSE within each group is bolded.
<table><tr><td>Model</td><td>Overall</td><td>NE</td><td>NW</td><td>S</td><td>CV</td></tr><tr><td>Pooled (NN)</td><td>0.0555 (0.0023)</td><td>0.0509 (0.0024)</td><td>0.0566 (0.0014)</td><td>0.0617 (0.0030)</td><td>0.0469 (0.0046)</td></tr><tr><td>2-Stage (NN)</td><td>0.0519 (0.0007)</td><td>0.0486 (0.0016)</td><td>0.0548 (0.0024)</td><td>0.0573 (0.0009)</td><td>0.0418 (0.0007)</td></tr><tr><td>Separate (NN)</td><td>0.0541 (0.0009)</td><td>0.0555 (0.0039)</td><td>0.0561 (0.0027)</td><td>0.0591 (0.0010)</td><td>0.0434 (0.0011)</td></tr><tr><td>Top-FT (NN)</td><td>0.0529 (0.0011)</td><td>0.0481 (0.0017)</td><td>0.0561 (0.0036)</td><td>0.0581 (0.0010)</td><td>0.0435 (0.0019)</td></tr><tr><td>Pool-w-L (NN)</td><td>0.0538 (0.0023)</td><td>0.0521 (0.0020)</td><td>0.0557 (0.0031)</td><td>0.0585 (0.0022)</td><td>0.0452 (0.0039)</td></tr><tr><td>Pooled (RF)</td><td>0.0556 (0.0002)</td><td>0.0514 (0.0004)</td><td>0.0581 (0.0005)</td><td>0.0626 (0.0004)</td><td>0.0437 (0.0004)</td></tr><tr><td>2-Stage (RF)</td><td>0.0534 (0.0003)</td><td>0.0456 (0.0004)</td><td>0.0555 (0.0005)</td><td>0.0606 (0.0006)</td><td>0.0438 (0.0005)</td></tr><tr><td>Separate (RF)</td><td>0.0580 (0.0002)</td><td>0.0502 (0.0005)</td><td>0.0602 (0.0005)</td><td>0.0643 (0.0004)</td><td>0.0494 (0.0010)</td></tr><tr><td>Pool-w-L (RF)</td><td>0.0542 (0.0002)</td><td>0.0503 (0.0004)</td><td>0.0558 (0.0005)</td><td>0.0610 (0.0003)</td><td>0.0435 (0.0005)</td></tr></table>

From Table 21, DNN trained with two-stage transfer learning achieves the lowest overall prediction error and wins in most subgroup metrics, highlighting the effectiveness of our proposed training strategy in low-dimensional settings. Notably, the two-stage training strategy also performs best when applied to RF, demonstrating the potential applicability of our framework and theorems to other nonparametric estimators.

## G.2. Detailed Configurations for UTKFace Experiment Using Pretrained Model

As required by the pretrained face feature extraction model, all images were resized to $1 6 0 \times 1 6 0 \times 3$ resolution using OpenCV (Bradski, 2000), with pixel values normalized to the range [0, 1] prior to feature extraction. For the neural networks used for age regression based on the extracted features, in addition to the MLP architecture described in the main text, each MLP instance was trained using the Adam optimizer with an initial learning rate of $1 0 ^ { - 4 } .$ . The model was trained under a cosine annealing learning rate schedule (Loshchilov & Hutter, 2016), where the rate smoothly decreased to $1 0 ^ { - 6 }$ over the course of training. Early stopping was applied with a patience of 5 epochs based on the validation loss.

## G.3. Complementary Experiments on the UTKFace Dataset

Joint Grouping by Ethnicity and Gender. In addition to ethnicity, gender is another potential factor that may influence facial characteristics and consequently affect age estimation. To further investigate this, we conducted complementary experiments where both gender and ethnicity were considered simultaneously, resulting in 10 subgroups. The following table presents the results, with all other configurations kept consistent with those in the experiments that considered only ethnicity. As shown in the table, the proposed two-stage transfer learning method improves models’ performance on each subgroup in comparison with the pooled training, achieving the lowest average MSE both overall and across most subgroups.

Table 22. Overall and per-group average MSE for age regression models over 20 independent trials, where both ethnicity and gender are used as grouping factors. Standard deviation is computed across seeds.
<table><tr><td>Group</td><td>Pooled</td><td>2-Stage</td><td>Separate</td><td>Top-FT</td><td>Pool-w-L</td></tr><tr><td>Overall</td><td>59.1 (2.6)</td><td>56.4 (2.1)</td><td>63.4 (3.2)</td><td>58.8 (2.5)</td><td>57.0 (1.8)</td></tr><tr><td>White-M</td><td>63.8 (4.8)</td><td>61.2 (4.2)</td><td>65.0 (6.6)</td><td>63.6(4.7)</td><td>62.0 (4.3)</td></tr><tr><td>White-F</td><td>77.2 (5.0)</td><td>71.7 (3.3)</td><td>78.5 (8.2)</td><td>76.7 (4.8)</td><td>73.7 (4.3)</td></tr><tr><td>Black-M</td><td>58.0 (5.8)</td><td>56.4 (6.0)</td><td>62.3 (7.0)</td><td>58.0 (5.8)</td><td>56.5 (6.0)</td></tr><tr><td>Black-F</td><td>57.9 (6.8)</td><td>57.1 (6.6)</td><td>68.0 (5.5)</td><td>57.7 (6.8)</td><td>56.3 (5.5)</td></tr><tr><td>Asian-M</td><td>49.1 (12.6)</td><td>47.4 (12.7)</td><td>60.0 (12.1)</td><td>48.8 (12.5)</td><td>48.3 (12.2)</td></tr><tr><td>Asian-F</td><td>34.7 (8.4)</td><td>32.0 (8.2)</td><td>39.4 (9.8)</td><td>34.4 (8.2)</td><td>33.7 (8.4)</td></tr><tr><td>Indian-M</td><td>52.5 (3.7)</td><td>51.0 (4.1)</td><td>58.0 (6.3)</td><td>52.2 (3.9)</td><td>50.2 (4.0)</td></tr><tr><td>Indian-F</td><td>43.1 (6.4)</td><td>41.0 (5.3)</td><td>50.0 (9.9)</td><td>43.1 (6.2)</td><td>40.8 (5.8)</td></tr></table>

Direct Modeling with MLP. As discussed in the main text, we can also directly employ MLPs to perform age estimation from raw image inputs. As a toy experiment, all images are resized to a resolution of 32 32 3 using OpenCV, with pixel values normalized to the range [0, 1]. Each image is then flattened into a 3,072-dimensional feature vector, which serves as the input x<sub>i</sub>.

We compare five learning strategies using MLPs as nonparametric estimator, following the same data-splitting setting as before. For each neural network instance, we employ an MLP with six layers and ReLU activation functions. Specifically, the input vector has 3,072 dimensions for all strategies except Pooled-with-Label, where we concatenate the one-hot encoded ethnicity feature with the image vector, resulting in a 3,077-dimensional input. This input passes through layers with 2048, 1024, 512, 256, 128, and 64 units, respectively, and the network concludes with a scalar output for age prediction. We adopt the AdamW optimizer $( \mathrm { l r } = 1 0 ^ { - 3 }$ , weight $\mathrm { d e c a y } = 1 0 ^ { - 4 } )$ and train the model under a cosine annealing learning rate schedule. Early stopping is applied with a patience of 10 epochs based on validation loss. In Table 23, we report the average MSE over 20 random trials, where the two-stage transfer learning achieves the lowest overall MSE and wins in the most subgroups.

Table 23. Overall and per-group average MSE for age regression models over 20 independent trials.
<table><tr><td>Model</td><td>Overall</td><td>White</td><td>Black</td><td>Asian</td><td>Indian</td></tr><tr><td>Pooled (NN)</td><td>140.8 (13.7)</td><td>165.9 (13.0)</td><td>135.2 (14.3)</td><td>112.6 (16.4)</td><td>107.8 (12.0)</td></tr><tr><td>2-Stage (NN)</td><td>136.7 (12.1)</td><td>161.2 (11.3)</td><td>129.9 (12.3)</td><td>110.9 (15.7)</td><td>104.6 (9.8)</td></tr><tr><td>Pool-w-L (NN)</td><td>139.3 (12.5)</td><td>164.4 (10.6)</td><td>134.0 (12.2)</td><td>111.8 (18.8)</td><td>105.4 (10.3)</td></tr><tr><td>Separate (NN)</td><td>161.9 (14.8)</td><td>180.0 (15.2)</td><td>157.3 (16.1)</td><td>147.5 (17.0)</td><td>133.9 (14.3)</td></tr><tr><td>Top-FT (NN)</td><td>140.3 (13.5)</td><td>165.2 (13.0)</td><td>135.1 (13.5)</td><td>112.8 (16.8)</td><td>107.0 (11.6)</td></tr></table>