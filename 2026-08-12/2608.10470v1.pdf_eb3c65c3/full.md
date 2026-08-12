# A JOINT-DISTRIBUTION ROUTE TO FAIR REPRESENTATIONS WITH CONTINUOUS SENSITIVE ATTRIBUTES<sup>∗</sup>

YIJIN NI<sup>†</sup> AND XIAOMING HUO<sup>†</sup>

Abstract. Fair representation learning with a continuous sensitive attribute S requires a representation Z that is statistically independent of S. Existing criteria, including generalized demographic parity, the expectation of integral probability metrics (EIPM), and mutual information, enforce this independence by averaging a per-value discrepancy between the conditional law $P _ { Z \mid S = s }$ and the marginal $P _ { Z }$ over the law of S. This approach requires a nonparametric surrogate for the conditional law at each sensitive value. We propose evaluating independence through a single joint discrepancy $d ( P _ { Z , S } , P _ { Z } \otimes P _ { S } )$ between the joint law and the product of its marginals. We establish a disintegration identity; on decomposable witness classes it equals the conditional-integral functional that EIPM and generalized demographic parity instantiate. By reaching the same target without the conditional law, this discrepancy can be estimated directly from samples via a dependence statistic rather than conditional smoothing. We take the Hilbert–Schmidt independence criterion (HSIC) as an instance of the joint discrepancy d to investigate the statistical eficiency of replacing the conditional formulation. The HSIC estimator is a closed-form $O ( n ^ { 2 } )$ statistic that converges at the $O ( n ^ { - 1 / 2 } )$ rate, in contrast to the nonparametric $O ( n ^ { - 2 / 5 } )$ rate of the conditional-route estimators. We prove this instance is equivalent to the conditional maximum mean discrepancy (MMD) integral up to an explicit spectral tail. The corresponding algorithmic implementation, i.e., FRHSIC, attains fairness–accuracy tradeofs comparable to conditional-route baselines while reducing per-epoch training time.

Key words. fair representation learning, continuous sensitive attributes, Hilbert–Schmidt in dependence criterion, kernel methods, demographic parity

MSC codes. 68T05, 62G05, 62G20, 46E22

1. Introduction. Fair representation learning seeks a representation Z that is statistically independent of a sensitive attribute $S \in { \mathcal { S } } , { \mathrm { ~ i . e . , ~ } } Z \mathrm { ~ \bot ~ } S .$ so that any downstream predictor built on Z inherits demographic parity with respect to S (Zemel et al., 2013; Madras et al., 2018). An encoder $h : \mathcal { X }  \mathcal { Z }$ maps an input $X \in { \mathcal { X } }$ to the representation $Z = h ( X ) \in { \mathcal { Z } }$ , and any prediction head acting on $Z$ is fair once Z carries no information about S, which makes $Z \perp S$ the representation-level target. Writing $P _ { Z }$ and $P _ { S }$ for the marginal laws of Z and $S$ and $P _ { Z \mid S = s }$ for the conditional law of $Z$ given $S = s .$ , this target is the equality $P _ { Z | S = s } = P _ { Z }$ for $P _ { S }$ -almost every s. We study it when $S$ is continuous, as with age, income, or a risk score in domains such as hiring, lending, and criminal justice (Calders and Verwer, 2010), where independence must hold across a continuum of sensitive values rather than across finitely many groups.

For continuous S, existing criteria enforce this independence by approximating a localized conditional object indexed by $S = s { \mathrm { : } }$ the conditional law $P _ { Z | S = s } , \mathrm { o r }$ , for generalized demographic parity (GDP), the conditional mean $\mathbb { E } [ \cdot \ | \ S \ = \ s ]$ of the prediction head. They share one form, the S-average of a per-value discrepancy d between the conditional law and the marginal,

$$
\begin{array} { r } { \mathcal { T } _ { d } ( Z ; S ) : = \mathbb { E } _ { S } \left[ d ( P _ { Z | S } , P _ { Z } ) \right] , } \end{array}\tag{1.1}
$$

and difer only in d: GDP uses a first-moment diference (Jiang et al., 2022), the expectation-of-IPM criterion (EIPM) of Kong et al. (2025) uses an integral probability metric (IPM), realized by the kernel-weighted estimator FREM, and mutualinformation objectives use the Kullback–Leibler divergence (Cho et al., 2020b). Because a finite sample of size n has no repeated observations at an exact value $S =$ $s ,$ each criterion replaces exact conditioning by a localized or density-based surrogate, such as FREM’s leave-one-out kernel-weighted conditional empirical measure or $\mathrm { G D P ^ { \circ } s }$ Nadaraya–Watson kernel smoothing on S (Nadaraya, 1964; Watson, 1964). These smoothing-based estimators carry a localization bandwidth and converge at the one-dimensional nonparametric rate ${ \dot { O ( n ^ { - 2 / 5 } ) } }$

We propose to enforce independence through a single discrepancy between the joint law $P _ { Z , S }$ of the pair $( Z , S )$ and the product of its marginals $P _ { Z } \otimes P _ { S }$ , the product measure on ${ \mathcal { Z } } \times { \mathcal { S } } ;$ ; Figure 1 contrasts this joint route with the conditional route it replaces. For a class $\mathcal { F }$ of bounded measurable functions on ${ \mathcal { Z } } \times { \mathcal { S } } $ , we measure this discrepancy by the joint-vs-product integral probability metric

$$
\mathrm { I P M } _ { \mathcal { F } } \bigl ( P _ { Z , S } , P _ { Z } \otimes P _ { S } \bigr ) : = \operatorname* { s u p } _ { f \in \mathcal { F } } \bigl | \mathbb { E } _ { P _ { Z , S } } [ f ] - \mathbb { E } _ { P _ { Z } \otimes P _ { S } } [ f ] \bigr | ,
$$

which vanishes if and only if $Z \perp S$ whenever $\mathcal { F }$ is rich enough to separate distinct laws (a characteristic class), and at this target it agrees with the conditional-integral criteria of (1.1), namely GDP, EIPM, and mutual information (Proposition 3.1). The averaging over sensitive values is then intrinsic to the joint law rather than an imposed weighting, and no per-value conditional object enters the definition.

The link to the conditional-integral route is not confined to this zero level: the joint discrepancy disintegrates exactly into the same averaged form. A disintegration identity (Theorem 3.2) rewrites the joint-vs-product IPM, for any class ${ \mathcal F } .$ , as the $P _ { S } .$ -average of a conditional contrast with the supremum over $\mathcal { F }$ taken outside the average,

$$
\mathrm { I P M } _ { \mathcal F } \big ( P _ { Z , S } , P _ { Z } \otimes P _ { S } \big ) = \operatorname* { s u p } _ { f \in \mathcal F } \left| \int _ { S } \int _ { \mathcal Z } f ( z , s ) d \big ( P _ { Z | S = s } - P _ { Z } \big ) ( z ) d P _ { S } ( d s ) \right|
$$

and on the decomposable classes of Definition 3.3, generated by per-value witness functions, this average equals the conditional-integral functional $\mathcal { T } _ { d }$ of (1.1) that GDP and EIPM instantiate (Corollary 3.4). Mutual information is the classical precedent for this coincidence: as the Kullback–Leibler discrepancy it equals both the joint-vsproduct divergence $\mathrm { K L } ( P _ { Z , S } \parallel P _ { Z } \otimes P _ { S } )$ and the $P _ { S }$ -average of the conditional divergences $\mathrm { K L } ( P _ { Z | S = s } \parallel P _ { Z } )$ by the chain rule of relative entropy (Cover and Thomas, 2006, Ch. 2), so it already measures independence from the joint law without con structing the conditional family $\{ P _ { Z \mid S = s } \}$ s

Because the target is a single functional of the paired law, it is estimated directly from the joint sample $\{ ( Z _ { i } , S _ { i } ) \} _ { i = 1 } ^ { n }$ using only those paired observations and the witness class $\mathcal { F }$ . How fast this direct estimate converges then depends on $\mathcal { F }$ rather than on a smoothing bandwidth.

We take the Hilbert–Schmidt independence criterion (HSIC) as the instance obtained by choosing $\mathcal { F }$ to be the unit ball of a product reproducing-kernel Hilbert space (RKHS). The empirical HSIC is a closed-form statistic, computable in $O ( n ^ { 2 } )$ time from the Gram matrices of $\{ Z _ { i } \} _ { i = 1 } ^ { n }$ and $\{ S _ { i } \} _ { i = 1 } ^ { n } ,$ whose kernel bandwidths are fixed scale parameters rather than per-value localization for the conditional law. Being a non-degenerate V -statistic, it converges to its population value at the $O ( n ^ { - 1 / 2 } )$

rate, in contrast to the nonparametric $O ( n ^ { - 2 / 5 } )$ of the smoothing-based EIPM estimator (FREM); our synthetic study fits log–log slopes of −0.46 for HSIC and −0.44 for EIPM, close $\mathrm { t o \ t i { \Omega } ^ { \mathrm { { t o } } } - 1 / 2 }$ and −2/5 (§6.2, Figure 2). This rate is not only pointwise: up to logarithmic factors it holds uniformly over encoder classes of controlled complexity, such as bounded linear encoders and fixed-architecture bounded multilayer perceptrons (Theorem 5.1, building on Ni and Huo (2024)), so the data-dependent encoder that minimizes the penalty still has small population dependence. We further prove that HSIC is equivalent to the $P _ { S ^ { - } \mathrm { a v e r a g e d } }$ conditional maximum mean discrepancy (MMD) integral up to an explicit spectral tail of the sensitive-attribute kernel (Theorem 4.5). As a minibatch regularizer, the resulting objective, which we call FRHSIC, trains about 36× faster per epoch than FREM at $n = 2 0 , 0 0 0 \ ( \ S 6 . 5 )$ ; penalizing dependence with HSIC for fairness is itself established (P´erez-Suay et al., 2017; Li et al., 2022), and our contribution is the theory that follows.

## 1.1. Summary of Contributions.

A joint-discrepancy formulation of the continuous-S criteria. We show that the existing continuous-S fairness criteria arise, on decomposable witness classes, as the conditional reading of a single joint discrepancy. A disintegration identity (Theorem 3.2) rewrites the joint-vs-product IPM as a P<sub>S</sub>-averaged conditional contrast, and on decomposable witness classes (Corollary 3.4) it coincides with the conditionalintegral functional $\mathcal { T } _ { d }$ of (1.1) that generalized demographic parity and EIPM instantiate, while mutual information is the Kullback–Leibler instance of the same S-averaged form.

A closed-form HSIC estimator with spectral and uniform control. We give a closed-form $O ( n ^ { 2 } )$ estimator of the joint discrepancy, the HSIC obtained from a product RKHS, that is controlled by the $P _ { S }$ -averaged conditional object without ever estimating it. We prove it is equivalent to the conditional MMD integral up to an explicit spectral tail of the sensitive-attribute kernel (Theorem 4.5), bounds the GDP of an RKHS head (Corollary 4.6), and concentrates uniformly at the rate $n ^ { - 1 / 2 }$ , up to logarithmic factors, under a controlled-complexity condition on the encoder class (Theorem 5.1, building on Ni and Huo (2024)).

A training algorithm for fair representations, FRHSIC. We instantiate the estimator as a minibatch regularizer that attains fairness–accuracy tradeofs comparable to conditional-route baselines while training about 36× faster per epoch than FREM at $n = 2 0 , 0 0 0 \ ( \ S 6 . 5 )$ . On synthetic data and five real datasets, FRHSIC attains comparable fairness–accuracy tradeofs, shows estimator convergence consistent with the predicted $O ( n ^ { - 1 / 2 } )$ rate (§6.2, Figure 2), and keeps fairness stable across fresh downstream heads.

![](images/71d91789bbae7d4ae6d85b2cf78235eeabf305342ba6f01aad6d9277fcd98fd0.jpg)  
Fig. 1. Conditional and joint routes for continuous-sensitive fair representation learning. The conditional route approximates local objects indexed by $S = s ;$ the joint route estimates dependence directly from paired samples.

2. Preliminaries and Existing Criteria. This section fixes notation, recalls the integral probability metrics used throughout, and shows that the existing continuous-S fairness criteria share a common conditional-integral form, the baseline against which our joint route is compared.

2.1. Setup and representation-level fairness. Representation-level fairness asks for a representation that is statistically independent of the sensitive attribute while remaining predictive of the target. Let $X \in { \mathcal { X } }$ denote the input random vector, $S \in { \mathcal { S } } \subseteq \mathbb { R }$ the continuous sensitive attribute, and $Y \in \mathcal { V }$ the target variable. A representation function $h : \mathcal { X } \to \mathcal { Z }$ yields $Z = h ( X )$ , and a prediction head $f : \mathcal { Z } \to \mathcal { V }$ yields ${ \widehat { Y } } = f ( Z ) = f \circ h ( X )$ . Write $P _ { Z }$ for the marginal law of $Z , \ P _ { Z | S = s }$ for the conditional law of $Z$ given $S = s$ , and $P _ { Z , S }$ for the joint law. For a reproducing kernel Hilbert space (RKHS) F with kernel $k ,$ the mean embedding of a distribution $P$ is $\mu _ { P } = \mathbb { E } _ { X \sim P } [ k ( X , \cdot ) ] \in \mathcal { F }$ . For binary $S \in \{ 0 , 1 \}$ , a predictor $\bar { \widehat { Y } }$ satisfies demographic parity $( \mathrm { D P } ) \operatorname { i f } { \widehat { Y } } \bot S$ , which for binary $\widehat { Y }$ reduces to $\mathbb { E } [ \widehat { Y } \mid S = 0 ] = \mathbb { E } [ \widehat { Y } \mid S = 1 ]$ . The goal of fair representation learning is a representation $Z = h ( X )$ that is independent of $S$ while remaining informative about $Y$ . For continuous $S$ the exact representationlevel target is

$$
( 2 . 1 ) \qquad Z \perp S , \qquad \mathrm { e q u i v a l e n t l y } \qquad P _ { Z | S = s } = P _ { Z } \quad \mathrm { f o r ~ } P _ { S ^ { - } } \mathrm { a l m o s t ~ e v e r y ~ } s
$$

for any regular conditional distribution $\{ P _ { Z | S = s } \} _ { s \in \mathcal { S } }$ . This independence condition is the target that every criterion in this paper enforces; empirical criteria approximate it through a discrepancy between $P _ { Z \mid S = s }$ and $P _ { Z }$ , introduced next.

2.2. Discrepancies between probability laws. Distances between distributions in this paper are integral probability metrics, and the maximum mean discrepancy is the kernel instance that, for a characteristic kernel, vanishes exactly when two laws coincide. For a class $\mathcal { F }$ of bounded measurable functions, the in tegral probability metric (IPM) between distributions $P$ and $Q$ is $\mathrm { I P M } _ { \mathcal { F } } ( P , Q ) : = $ $\textstyle \operatorname* { s u p } _ { f \in { \mathcal { F } } } | \mathbb { E } _ { P } [ f ] - \mathbb { E } _ { Q } [ f ] |$ . Taking $\mathcal { F }$ to be the unit ball of an RKHS gives the maximum mean discrepancy,

$$
\mathrm { M M D } ( P , Q ) = \operatorname* { s u p } _ { \| f \| _ { \mathcal { F } } \leq 1 } | \mathbb { E } _ { P } [ f ] - \mathbb { E } _ { Q } [ f ] | = \| \mu _ { P } - \mu _ { Q } \| _ { \mathcal { F } } .
$$

When k is characteristic, $\mathrm { M M D } ( P , Q ) = 0$ if and only if $P = Q$ (Gretton et al., 2012). Section 3 applies the IPM with $P = P _ { Z , S }$ and $Q = P _ { Z } \otimes P _ { S }$

2.3. Conditional-integral criteria for continuous sensitive attributes. Existing continuous-S fairness criteria share one form, the S-average of a per-value discrepancy between the conditional and marginal laws of $Z ,$ and difer only in that discrepancy. For a discrepancy d between distributions, the integral functional is

$$
\begin{array} { r } { \mathcal { T } _ { d } ( Z ; S ) : = \mathbb { E } _ { S } \left[ d ( P _ { Z | S } , P _ { Z } ) \right] , } \end{array}
$$

which is well-defined for Polish ${ \mathcal { Z } } , S$ once a regular conditional probability is selected; existence follows from the disintegration theorem (Kallenberg, 2002, Theorem 5.4). Three instances recur. Generalized demographic parity (GDP) (Jiang et al., 2022) takes d to be a first-moment diference evaluated through a measurable head $f : { \mathcal { Z } } \to$ R,

$$
\Delta _ { \mathrm { G D P } } ( f ) = \mathbb { E } _ { S } \left| \mathbb { E } _ { Z } [ f ( Z ) \mid S ] - \mathbb { E } _ { Z } [ f ( Z ) ] \right| , \qquad Z = h ( X ) ,\tag{2.2}
$$

with the conditional expectation estimated by Nadaraya–Watson kernel smoothing on S (Nadaraya, 1964; Watson, 1964); GDP reduces to DP when S is binary. The expectation of IPM (EIPM) (Kong et al., 2025) takes d to be an IPM,

$$
\mathrm { E I P M } _ { \mathcal { V } } ( Z ; S ) : = \mathbb { E } _ { S } [ \mathrm { I P M } _ { \mathcal { V } } ( P _ { Z | S } , P _ { Z } ) ] , \qquad \mathrm { I P M } _ { \mathcal { V } } ( P , Q ) = \operatorname* { s u p } _ { f \in \mathcal { V } } | \mathbb { E } _ { P } [ f ] - \mathbb { E } _ { Q } [ f ] | ,
$$

and controlling EIPM controls GDP (Kong et al., 2025). Mutual information is the Kullback–Leibler instance,

$$
I ( Z ; S ) = \mathbb { E } _ { S } \left[ \operatorname { K L } ( P _ { Z | S } \parallel P _ { Z } ) \right] ,
$$

and characterizes independence, $I ( Z ; S ) = 0 \iff Z \perp S$ (Cover and Thomas, 2006, Theorem 2.6.3). Section 3 relates this conditional-integral route to a single joint-vsproduct discrepancy through disintegration, recovering the conditional-integral IPM on decomposable witness classes.

2.4. Joint dependence measures and HSIC. An alternative measures dependence directly between the joint law and the product of marginals, and the Hilbert– Schmidt independence criterion is the kernel instance that, under characteristic product kernels, vanishes exactly under independence. This is the object the method of this paper is built on.

Definition 2.1 (Mean embeddings and HSIC). Let $\mathcal { F } _ { \mathcal { Z } }$ and $\mathcal { F } _ { \mathcal { S } }$ be RKHSs on $\mathcal { Z }$ and S with bounded characteristic kernels $k z$ and $k _ { S }$ , and let ${ \mathcal { F } } _ { { \mathcal { Z } } \otimes { \mathcal { S } } }$ be the tensorproduct RKHS on ${ \mathcal { Z } } \times { \mathcal { S } }$ with product kernel $k \ z \otimes k \ s$ . The marginal and joint mean embeddings are

$$
\mu _ { Z } : = \mathbb { E } _ { Z } [ k _ { \mathcal { Z } } ( Z , \cdot ) ] \in \mathscr { F } _ { \mathcal { Z } } , \qquad \mu _ { S } : = \mathbb { E } _ { S } [ k _ { S } ( S , \cdot ) ] \in \mathscr { F } _ { S } ,
$$

$$
\mu _ { Z , S } : = \mathbb { E } _ { ( Z , S ) } \left[ k _ { \mathcal { Z } } ( Z , \cdot ) \otimes k _ { S } ( S , \cdot ) \right] \in \mathscr { F } _ { \mathcal { Z } \otimes S } ,
$$

and $\mu _ { Z } \otimes \mu _ { S } \in \mathcal { F } _ { \mathcal { Z } \otimes S }$ is the embedding of the product law $P _ { Z } \otimes P _ { S }$ . The conditional mean embedding of Z given $S = s$ is $\mu _ { Z | S = s } : = \mathbb { E } [ k _ { \mathcal { Z } } ( Z , \cdot ) \mid S = s ] \in \mathcal { F } _ { \mathcal { Z } } ,$ ; we write $\mu _ { Z \mid S }$ for the $\mathcal { F } _ { \mathcal { Z } }$ -valued random element $s \mapsto \mu _ { Z | S = s }$ evaluated at S. The Hilbert– Schmidt independence criterion (HSIC) is the squared maximum mean discrepancy between the joint law and the product of marginals,

$$
\operatorname { H S I C } ( Z , S ) : = \left\| \mu _ { Z , S } - \mu _ { Z } \otimes \mu _ { S } \right\| _ { \mathcal { F } _ { Z \otimes S } } ^ { 2 } .
$$

Under characteristic product kernels, $\mathrm { H S I C } ( Z , S ) = 0$ if and only if $Z \perp S$ (Gretton et al., 2005). Section 3 relates this joint discrepancy to conditional-integral criteria through a disintegration identity. The equality with the conditional-integral IPM requires the decomposability condition stated in Corollary 3.4.

## 3. The Joint Discrepancy: An Intrinsic Metric for the Conditional-Integral Criteria.

3.1. Zero-level relation to existing criteria. This section develops the central object of this paper: a single joint discrepancy between $P _ { Z , S }$ and $P _ { Z } \otimes P _ { S }$ that serves as an intrinsic metric for the conditional-integral criteria. We begin at the zero level, where a characteristic witness class makes the joint discrepancy vanish exactly at the representation-level fairness target shared by the existing criteria; the nonzero regime is developed in §3.2–§4.2.

Proposition 3.1 (Zero-level characterization of representation fairness). Let $\mathcal { Z }$ and S be Polish spaces, let $\mathcal { F }$ be a characteristic class of bounded measurable functions on ${ \mathcal { Z } } \times { \mathcal { S } }$ , that is, one for which I $\mathrm { P M } _ { \mathcal { F } }$ separates probability laws on ${ \mathcal { Z } } \times { \mathcal { S } }$ , and let $( Z , S )$ be a random pair on ${ \mathcal { Z } } \times { \mathcal { S } } .$ . Then the following are equivalent:

1. $\mathrm { I P M } _ { \mathcal { F } } ( P _ { Z , S } , P _ { Z } \otimes P _ { S } ) = 0 ;$

2. $Z \perp S$ , equivalently $P _ { Z , S } = P _ { Z } \otimes P _ { S } $ ;

3. for some, equivalently every, regular conditional distribution $\{ P _ { Z | S = s } \} _ { s \in \mathcal { S } }$

$$
P _ { Z | S = s } = P _ { Z } \quad f o r \ : P _ { S ^ { - } } a . e . \ : s ;
$$

4. for every bounded measurable $g : { \mathcal { Z } }  \mathbb { R }$

$$
\begin{array} { r } { { \mathbb E } [ g ( Z ) \mid S = s ] = { \mathbb E } [ g ( Z ) ] \quad f o r \ P _ { S } - a . e . \ s . } \end{array}
$$

Proof. The equivalences are proved in Appendix A.1: condition (1) is equivalent to (2) because a characteristic class separates $P _ { Z , S }$ from $P _ { Z } \otimes P _ { S } ,$ , and $( 2 ) - ( 4 )$ are equivalent by disintegration. The MMD-based EIPM instance, $\mathrm { E I P M } _ { \mathcal { V } } ( Z ; S ) = 0$ for the unit ball $\nu \subset \mathcal { F } _ { \mathcal Z }$ , and the downstream-head (GDP) form are recorded in Appendices $\mathrm { A . 2 – A . 3 }$

Remark (zero-level equivalence only). Proposition 3.1 is a zero-level statement. Away from zero the joint discrepancy and the conditional-integral criteria are diferent functionals: the conditional-integral criteria average a per-value discrepancy between $P _ { Z \mid S = s }$ and $P _ { Z }$ over the law of $S _ { ; }$ , whereas the joint discrepancy applies a single supremum to the paired law.

3.2. Conditional-integral and joint formulations. Conditional-integral criteria compare $P _ { Z \mid S = s }$ with $P _ { Z }$ and average the comparison over $S ,$ whereas the joint route compares the joint law $P _ { Z , S }$ with the product of marginals $P _ { Z } \otimes P _ { S }$ . When S takes finitely many values the two formulations coincide as aggregated two-sample MMDs, recorded as Lemma B.2 in Appendix B.1. The case of practical interest is continuous $S ,$ where the conditional family $\{ P _ { Z \mid S = s } \}$ <sub>s</sub> is uncountable and the two formulations difer in how they are estimated. The next theorem disintegrates the joint-vs-product IPM into a $P _ { S }$ -weighted conditional contrast.

Theorem 3.2 (Disintegration of the joint-vs-product IPM). Let Z and S be standard Borel spaces, and let $\{ P _ { Z | S = s } \} _ { s \in \mathcal { S } }$ be a regular conditional distribution of Z given $S _ { ☉ }$ . For any class $\mathcal { F }$ of bounded measurable functions on ${ \mathcal { Z } } \times { \mathcal { S } }$

$$
\mathrm { I P M } _ { \mathcal F } \big ( P _ { Z , S } , P _ { Z } \otimes P _ { S } \big ) = \operatorname* { s u p } _ { f \in \mathcal F } \left| \int _ { S } \int _ { \mathcal Z } f ( z , s ) d \big ( P _ { Z | S = s } - P _ { Z } \big ) ( z ) d P _ { S } ( d s ) \right| .
$$

Proof. See Appendix $\mathrm { A . 4 } .$

The theorem shows that the $P _ { S }$ weighting arises from disintegrating the joint law, rather than from an external weighting choice. For any class ${ \mathcal F } .$ , it places a single supremum outside the $P _ { S }$ -average of the conditional contrast; this average equals the conditional-integral functional $\mathcal { T } _ { d }$ exactly when $\mathcal { F }$ decomposes across $s ,$ which Definition 3.3 and Corollary 3.4 make precise.

Definition 3.3 (Decomposable witness class). Let V be a symmetric class of bounded measurable functions on $\mathcal { Z }$ , meaning $g \in \mathcal V$ implies $- g \in \mathcal { V }$ . A class $\mathcal { F }$ of bounded measurable functions on ${ \mathcal { Z } } \times { \mathcal { S } }$ is decomposable over $\gamma \ i f .$

1. for every measurable selector $s \mapsto g _ { s } \in \mathcal { V }$ , the function $( z , s ) \mapsto g _ { s } ( z )$ belongs to ${ \mathcal { F } } ;$

2. for every $f \in { \mathcal { F } }$ , the slice $f ( \cdot , s )$ belongs to V for $P _ { S } - a l m o s t$ every s.

Corollary 3.4 (Recovery of the conditional-integral IPM under decomposability). Adopt the assumptions of Theorem 3.2: Z and $s$ are standard Borel spaces and $\{ P _ { Z | S = s } \} _ { s \in { \mathcal { S } } }$ is a regular conditional distribution of Z given S. Let V be a symmetric uniformly bounded class of measurable functions on ${ \mathcal { Z } } ,$ and define

$$
d _ { \mathcal { V } } ( P , Q ) : = \mathrm { I P M } _ { \mathcal { V } } ( P , Q ) .
$$

Let $\mathcal { F }$ be decomposable over V in the sense of Definition 3.3. Assume that for every $\varepsilon > 0$ there exists a measurable selector $s \mapsto g _ { s } ^ { \varepsilon } \in \mathcal { V }$ such that

$$
\int g _ { s } ^ { \varepsilon } ( z ) d ( P _ { Z | S = s } - P _ { Z } ) ( z ) \geq d \nu ( P _ { Z | S = s } , P _ { Z } ) - \varepsilon \quad f o r P _ { S ^ { - } } a . e . s .
$$

Then

$$
\mathrm { I P M } _ { \mathcal { F } } ( P _ { Z , S } , P _ { Z } \otimes P _ { S } ) = \mathbb { E } _ { S } \left[ d _ { \mathcal { V } } ( P _ { Z | S = s } , P _ { Z } ) \right] = \mathcal { T } _ { d _ { \mathcal { V } } } ( Z ; S ) .
$$

Proof. See Appendix A.5.

Corollary 3.4 therefore recovers the conditional-integral IPM, which GDP and EIPM instantiate, exactly when the witness class decomposes across $s ;$ mutual information is the Kullback–Leibler instance of the same S-averaged form.

4. HSIC: A Closed-Form Equivalent with Demographic-Parity Control. HSIC is equivalent, up to an explicit spectral tail, to the conditional MMD integral it replaces, and that equivalence carries over to the fairness metric itself. §4.1 proves the equivalence with the conditional MMD integral, the conditional-integral criterion HSIC replaces. §4.2 turns it into control of the demographic-parity gap at the population level and on a finite sample.

4.1. Equivalence with the conditional MMD integral. HSIC is the kernel joint discrepancy obtained by comparing $P _ { Z , S }$ with $P _ { Z } \otimes P _ { S }$ . Instantiating the joint IPM of Theorem 3.2 with the unit ball of the tensor-product RKHS $\mathcal { F } _ { \mathcal { Z } \otimes \mathcal { S } }$ of Definition 2.1 gives the maximum mean discrepancy $\mathrm { M M D } _ { k _ { z } \otimes k _ { S } } ( P _ { Z , S } , P _ { Z } \otimes P _ { S } )$ , and HSIC is its square,

$$
\begin{array} { r } { \mathrm { H S I C } ( Z , S ) = \mathrm { M M D } _ { k z \otimes k _ { S } } ^ { 2 } \bigl ( P _ { Z , S } , P _ { Z } \otimes P _ { S } \bigr ) . } \end{array}
$$

We assume throughout that the product kernel $k _ { \mathcal { Z } } \otimes k _ { \mathcal { S } }$ is characteristic to the relevant joint and product measures on ${ \mathcal { Z } } \times { \mathcal { S } } ;$ ; this holds under standard conditions for the Gaussian kernels used in our experiments. The first of the two bounds comes from a decomposition. The following proposition writes HSIC over the conditional deviations $\mu _ { Z | S = s } - \mu _ { Z }$ and bounds it by the squared expected MMD, so a small conditional MMD makes HSIC small.

Proposition 4.1 (Decomposition of HSIC and its expected-MMD bound). Suppose $k z$ and $k _ { S }$ are bounded, i.e., $k _ { \mathcal { Z } } ( z , z ) \le \kappa _ { \mathcal { Z } }$ and $k _ { S } ( s , s ) \leq \kappa _ { S }$ for all $z \in { \mathcal { Z } }$ $s \in S$ . Then HSIC admits the decomposition

$$
\begin{array} { r } { \mathrm { H S I C } ( Z , S ) = \mathbb { E } _ { S , S ^ { \prime } } \left[ \left. \mu _ { Z | S } - \mu _ { Z } , \mu _ { Z | S ^ { \prime } } - \mu _ { Z } \right. _ { \mathcal { F } _ { Z } } \cdot k _ { S } ( S , S ^ { \prime } ) \right] , } \end{array}\tag{4.1}
$$

where $S , S ^ { \prime }$ are independent copies. Moreover, with $\mathrm { M M D } _ { k _ { z } } ( P _ { Z | S = s } , P _ { Z } ) = \| \mu _ { Z | S = s } -$ $\mu _ { Z } \| _ { \mathcal { F } _ { \mathcal { Z } } }$ the maximum mean discrepancy of $\mathcal { S } \mathcal { Q }$ under $k _ { \mathcal { Z } }$ , HSIC is bounded by the squared expected MMD:

$$
\mathrm { H S I C } ( Z , S ) \leq \kappa _ { S } \cdot \left( \mathbb { E } _ { S } \left[ \mathrm { M M D } _ { k _ { z } } \left( P _ { Z | S } , P _ { Z } \right) \right] \right) ^ { 2 } .\tag{4.2}
$$

Proof. See Appendix A.6.

Remark 4.2 (Relation to the MMD-based EIPM). The expected MMD $\mathbb { E } _ { S } [ \mathrm { M M D } _ { k _ { z } } ( P _ { Z | S } , P _ { Z } ) ]$ is the MMD-based EIPM, the expectation-of-IPM criterion realized with the RKHS witness class $\nu \subset \mathcal { F } _ { \mathcal Z }$ . Equation (4.1) writes HSIC as a covariance-weighted aggregation of the conditional mean-embedding deviations, and Inequality (4.2) bounds HSIC by $\kappa _ { \mathcal { S } }$ times its square. The reverse control, of the conditional MMD integral $\mathbb { E } _ { S } [ \mathrm { M M D } _ { k _ { z } } ^ { 2 } ( P _ { Z | S } , P _ { Z } ) ]$ by HSIC up to an explicit spectral tail, is Theorem 4.5.

The second bound comes from the same decomposition. Because $\parallel \mu _ { Z | S = s } \mathrm { ~ - ~ }$ $\mu _ { Z } \| _ { \mathcal { F } _ { Z } } = \mathrm { M M D } _ { k _ { Z } } ( P _ { Z | S = s } , P _ { Z } )$ , the conditional deviation $\Delta ( s ) : = \mu _ { Z | S = s } - \mu _ { Z }$ is the conditional MMD in vector form. The next theorem shows that a small HSIC makes every RKHS-smoothed average of $\Delta$ small, so the joint statistic controls the conditional MMD after smoothing.

Theorem 4.3 (Smoothed conditional-distribution control by HSIC). Let $k z$ and $k _ { S }$ be bounded measurable kernels with RKHSs $\mathcal { F } _ { \mathcal { Z } }$ and $\mathcal { F } _ { \mathcal { S } }$ . Let

$$
\Delta ( s ) : = \mu _ { Z | S = s } - \mu _ { Z } \in \mathscr { F } _ { \mathcal { Z } } , \qquad \mu _ { Z | S = s } : = \mathbb { E } [ k _ { \mathcal { Z } } ( Z , \cdot ) \mid S = s ] .
$$

Then $\Delta \in L ^ { 2 } ( P _ { S } ; \mathcal { F } _ { \mathcal { Z } } )$ and

$$
\begin{array} { r } { \mathrm { H S I C } ( Z , S ) = \left\| \mathbb { E } _ { S } [ \Delta ( S ) \otimes k _ { S } ( S , \cdot ) ] \right\| _ { \mathcal { F } _ { Z \otimes S } } ^ { 2 } . } \end{array}
$$

Moreover, for every $g \in { \mathcal { F } } _ { S }$

$$
\begin{array} { r } { \left\| \mathbb { E } _ { S } [ g ( S ) \Delta ( S ) ] \right\| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 } \le \| g \| _ { \mathcal { F } _ { S } } ^ { 2 } \operatorname { H S I C } ( Z , S ) . } \end{array}\tag{4.3}
$$

Proof. See Appendix A.7.

At a single sensitive value $s _ { 0 }$ , the same bound controls the conditional MMD localized near $s _ { 0 } .$ , the contrast a per-value estimator would target.

Corollary 4.4 (Localized conditional-MMD control by HSIC). Under the assumptions of Theorem $4 . 9 ,$ for any $s _ { 0 } \in S$ with $k _ { S } ( s _ { 0 } , s _ { 0 } ) > 0$ 2

$$
\big \| \mathbb { E } _ { S } [ k _ { S } ( S , s _ { 0 } ) ( \mu _ { Z | S } - \mu _ { Z } ) ] \big \| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 } \leq k _ { S } ( s _ { 0 } , s _ { 0 } ) \mathrm { H S I C } ( Z , S ) .
$$

More generally, for any $s _ { 0 } , s _ { 1 } \in \mathcal S$

$$
\begin{array} { r } { \big \| \mathbb { E } _ { S } [ \{ k _ { S } ( S , s _ { 0 } ) - k _ { S } ( S , s _ { 1 } ) \} ( \mu _ { Z | S } - \mu _ { Z } ) ] \big \| _ { \mathcal { F } _ { \mathbb { Z } } } ^ { 2 } \leq \| k _ { S } ( s _ { 0 } , \cdot ) - k _ { S } ( s _ { 1 } , \cdot ) \| _ { \mathcal { F } _ { S } } ^ { 2 } \mathrm { H S I C } ( Z , S ) . } \end{array}
$$

Proof. See Appendix A.8.

Resolving $\Delta$ along the spectrum of the sensitive-attribute kernel turns the control of Theorem 4.3 into a bound on the conditional MMD integral; combined with the bound of Proposition 4.1, the two give a two-sided equivalence: HSIC and the condi tional MMD integral control each other up to an explicit spectral tail, so making the closed-form statistic small makes the conditional MMD integral that EIPM targets small up to the residual tail $\rho _ { m } ^ { 2 }$

Theorem 4.5 (Spectral equivalence of HSIC and the conditional MMD integral). Assume the conditions of Theorem $4 . 9 ,$ and let $\kappa _ { \mathcal { S } }$ be the bound on $k _ { S }$ from Proposition 4.1. Let $\Delta ( s ) : = \mu _ { Z | S = s } - \mu _ { Z }$ and let $T _ { S } : L ^ { 2 } ( P _ { S } )  L ^ { 2 } ( P _ { S } )$ be the kernel integral operator

$$
( T _ { S } u ) ( s ) : = \int k s ( s , t ) u ( t ) d P _ { S } ( t ) ,
$$

with orthonormal eigensystem $( \lambda _ { \ell } , \psi _ { \ell } ) _ { \ell \ge 1 } , \lambda _ { 1 } \ge \lambda _ { 2 } \ge \dots \ge 0$ , and let $P _ { m }$ denote the $L ^ { 2 } ( P _ { S } )$ projection onto span $\{ \psi _ { 1 } , \ldots , \psi _ { m } \}$ , extended componentwise to $L ^ { 2 } ( P _ { S } ; \mathcal { F } _ { \mathcal { Z } } )$ Then, for any m with $\lambda _ { m } > 0$ such that $\| ( I - P _ { m } ) \Delta \| _ { L ^ { 2 } ( P _ { S } ; \mathcal { F } _ { \mathcal { Z } } ) } ^ { 2 } \le \rho _ { m } ^ { 2 }$

$$
\frac { 1 } { \kappa _ { \mathscr { S } } } \mathrm { H S I C } ( Z , S ) \leq \mathbb { E } _ { S } \big [ \mathrm { M M D } _ { k _ { \mathscr { Z } } } ^ { 2 } ( P _ { Z | S } , P _ { Z } ) \big ] \leq \frac { 1 } { \lambda _ { m } } \mathrm { H S I C } ( Z , S ) + \rho _ { m } ^ { 2 } .
$$

Hence HSIC(Z, S) and the conditional MMD integral control each other up to the constants $\kappa _ { \mathcal { S } } , \lambda _ { m }$ and the explicit spectral tail $\rho _ { m } ^ { 2 }$

Proof. The upper bound is the spectral argument of Appendix A.9: with $\begin{array} { r c l } { { \mathbb { E } _ { S } [ \mathrm { M M D } _ { k _ { z } } ^ { 2 } ( P _ { Z | S } , \bar { P } _ { Z } ) ] } } & { { = } } & { { \| \Delta \| _ { L ^ { 2 } ( P _ { S } ; \mathcal { F } _ { \mathcal { Z } } ) } ^ { 2 } , } } \end{array}$ , expanding $\Delta$ in the eigenbasis of $T _ { S }$ gives $\| \Delta \| _ { L ^ { 2 } ( P _ { S } ; \mathcal { F } _ { \mathcal { Z } } ) } ^ { 2 } \le \lambda _ { m } ^ { - 1 } \mathrm { H S I C } ( Z , \bar { S } ) + \rho _ { m } ^ { 2 }$ . For the lower bound, Proposition 4.1 gives $\begin{array} { r } { \mathrm { H S I C } ( Z , S ) \ \le \ \kappa _ { S } \big ( \mathbb { E } _ { S } [ \mathrm { M M D } _ { k _ { Z } } ( P _ { Z | S } , P _ { Z } ) ] \big ) ^ { 2 } } \end{array}$ , and Jensen’s inequality gives $\left( \mathbb { E } _ { S } [ \mathrm { M M D } _ { k _ { z } } ( P _ { Z | S } , P _ { Z } ) ] \right) ^ { 2 } \le \mathbb { E } _ { S } [ \mathrm { M M D } _ { k _ { z } } ^ { 2 } ( P _ { Z | S } , P _ { Z } ) ]$

HSIC controls the conditional-distribution deviations lying in the well-conditioned spectral directions of $T _ { S }$ . Because the kernel integral operator on $s$ is compact, the projected tail $\rho _ { m }$ decreases in m and the remaining directions enter only through it, so when $\rho _ { m }$ is small, small HSIC implies small conditional-integral MMD. This is the population counterpart of the finite-sample spectral factor $\hat { \lambda } _ { m } ^ { - 1 }$ in Theorem 4.7.

4.2. Control of the demographic-parity gap. The equivalence transfers to the fairness metric, in population and then on a finite sample. At the population level, the same spectral bound reaches generalized demographic parity: for an RKHS prediction head, small HSIC controls $\Delta _ { \mathrm { G D P } } ( f )$ up to the spectral tail of its conditional mean $m _ { f }$

Corollary 4.6 (Spectral control of GDP for RKHS heads). Under the assumptions of Theorem $4 . 5 ,$ let $f \in \mathcal { F } _ { \mathcal { Z } }$ and define

$$
m _ { f } ( s ) : = \mathbb { E } [ f ( Z ) \mid S = s ] - \mathbb { E } [ f ( Z ) ] .
$$

Let $P _ { m }$ be the spectral projection from Theorem 4.5. Then

$$
\| P _ { m } m _ { f } \| _ { L ^ { 2 } ( P _ { S } ) } ^ { 2 } \leq \frac { \| f \| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 } } { \lambda _ { m } } \operatorname { H S I C } ( Z , S ) .
$$

$I f \| ( I - P _ { m } ) m _ { f } \| _ { L ^ { 2 } ( P _ { S } ) } ^ { 2 } \leq r _ { m , f } ^ { 2 } ,$ , then

$$
\Delta _ { \mathrm { G D P } } ( f ) ^ { 2 } = \left( \mathbb { E } _ { S } | m _ { f } ( S ) | \right) ^ { 2 } \leq \frac { \| f \| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 } } { \lambda _ { m } } \operatorname { H S I C } ( Z , S ) + r _ { m , f } ^ { 2 } .
$$

Proof. See Appendix A.10.

On a finite sample, the same control holds for the gap computed from data. The next theorem is the finite-sample form of Corollary 4.6: the empirical HSIC statistic controls the empirical demographic-parity gap of any RKHS head, with the eigenvalue $\hat { \lambda } _ { m }$ of the centered sensitive Gram matrix Le in the role of $\lambda _ { m }$

Theorem 4.7 (Finite-sample control of the empirical demographic-parity gap by HSIC). Let $\{ ( Z _ { i } , S _ { i } ) \} _ { i = 1 } ^ { n }$ be a sample. Let $K , L \in \mathbb { R } ^ { n \times n }$ be the Gram matrices

$$
K _ { i j } = k _ { \mathcal { Z } } ( Z _ { i } , Z _ { j } ) , \qquad L _ { i j } = k _ { \mathcal { S } } ( S _ { i } , S _ { j } ) ,
$$

and let

$$
H : = I _ { n } - n ^ { - 1 } \mathbf { 1 } \mathbf { 1 } ^ { \top } , \qquad \widetilde { K } : = H K H , \qquad \widetilde { L } : = H L H .
$$

Define the biased empirical HSIC statistic

$$
\widehat { \mathrm { H S I C } } _ { n } ( Z , S ) : = n ^ { - 2 } \operatorname { t r } ( \widetilde { K } \widetilde { L } ) .\tag{4.4}
$$

Let $\hat { \lambda } _ { 1 } \geq \dots \geq \hat { \lambda } _ { r } > 0$ be the positive eigenvalues $o f n ^ { - 1 } { \widetilde { L } }$ with orthonormal eigenvectors $\boldsymbol { \hat { e } } _ { 1 } , \dots , \boldsymbol { \hat { e } } _ { r } \in \mathbb { R } ^ { n }$ , where $r = \mathrm { r a n k } ( \widetilde { L } )$ , and $f o r \ 1 \leq m \leq r$ let $P _ { m }$ be the orthogonal projection onto span $\{ \hat { e } _ { 1 } , \dots , \hat { e } _ { m } \}$ . For $f \in { \mathcal { F } } _ { \mathcal { Z } }$ , define

$$
\bar { f } _ { n } : = n ^ { - 1 } \sum _ { j = 1 } ^ { n } f ( Z _ { j } ) , \qquad \hat { \delta } _ { f , i } : = f ( Z _ { i } ) - \bar { f } _ { n } , \qquad \delta _ { f } : = ( \hat { \delta } _ { f , 1 } , \ldots , \hat { \delta } _ { f , n } ) ^ { \top } ,
$$

and let $\widehat { m } _ { f } : = P _ { m } \delta _ { f }$ be the empirical conditional-mean gap of $f ,$ the prediction’s dependence on S resolved on the m best-conditioned sensitive directions. Then the empirical demographic-parity gap $\begin{array} { r } { \widehat { \Delta } _ { \mathrm { G D P } } ( f ) : = n ^ { - 1 } \sum _ { i = 1 } ^ { n } \vert ( \widehat { m } _ { f } ) _ { i } \vert } \end{array}$ satisfies

$$
{ \widehat { \Delta } } _ { \mathrm { G D P } } ( f ) ^ { 2 } \ \leq \ { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } ( { \widehat { m } } _ { f } ) _ { i } ^ { 2 } \ \leq \ { \frac { \| f \| _ { \mathcal { F } _ { \mathbb { Z } } } ^ { 2 } } { \widehat { \lambda } _ { m } } } { \widehat { \mathrm { H S I C } } } _ { n } ( Z , S ) .\tag{4.5}
$$

Proof. See Appendix A.11.

Interpretation. Equation (4.5) is the term-by-term empirical form of Corollary 4.6: the centered Gram matrix $\widetilde { L }$ replaces the operator $T _ { S }$ , its m-th eigenvalue $\hat { \lambda } _ { m }$ replaces $\lambda _ { m } .$ , and the projection $\widehat { m } _ { f } ~ = ~ P _ { m } \delta _ { f }$ of the prediction onto the m best-conditioned sensitive directions replaces the projected conditional mean $P _ { m } m _ { f }$ The free parameter $m$ trades resolution for conditioning, as in the population bound: a larger m resolves more of the prediction’s dependence on S but lowers $\hat { \lambda } _ { m }$ . A small $\widehat { \mathrm { H S I C } } _ { n }$ therefore bounds $\widehat { \Delta } _ { \mathrm { G D P } } ( f )$ for every RKHS head $f ,$ through the head-dependent factor $\| f \| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 } / \hat { \lambda } _ { m }$

5. FRHSIC: A Regularizer with a Faster Train-to-Population Rate. FRHSIC turns the population control of the previous section into a trainable penalty whose empirical value bounds the learned encoder’s population dependence at the $n ^ { - 1 / 2 }$ rate. It is the empirical objective (5.1) together with its minibatch training algorithm: §5.1 defines the objective, and §5.2 gives the training algorithm (the Implementation paragraph) and the uniform train-to-population guarantee. The jointdistribution framework underlying the HSIC penalty was developed in Section $_ { 3 ; }$ this section is self-contained for a reader who wants only the method and its statistical guarantee.

5.1. Empirical FRHSIC objective. FRHSIC trains the representation $Z : =$ $h ( X )$ (as in $\ S 2 )$ by penalizing empirical dependence between $Z$ and $S$ over a feasible encoder class H and prediction-head class ${ \mathcal F } .$ . With fixed bounded characteristic kernels $k _ { \mathcal { Z } } , k _ { \mathcal { S } }$ , the empirical HSIC penalty ${ \widehat { \mathrm { H S I C } } } _ { n } ( h ( X ) , S )$ is the biased V -statistic (4.4) formed from the paired sample $\{ ( h ( X _ { i } ) , S _ { i } ) \} _ { i = 1 } ^ { n }$ . FRHSIC solves the empirical regularized problem

$$
\operatorname* { m i n } _ { h \in \mathcal H , \ f \in \mathcal F } \widehat { \mathcal L } _ { n } ( f \circ h ) + \lambda \widehat { \mathrm { H S I C } } _ { n } ( h ( X ) , S ) ,\tag{5.1}
$$

where ${ \widehat { \mathcal { L } } } _ { n }$ is the empirical prediction loss (cross-entropy for classification, MSE for regression) and $\lambda > 0$ controls the fairness–accuracy tradeof. Equation (5.1) is the

Lagrangian relaxation of the empirical constrained problem mi $\mathsf { 1 } _ { h , f } \widehat { \mathcal { L } } _ { n } ( f \circ h )$ subject to $\widehat { \mathrm { H S I C } } _ { n } ( h ( X ) , S ) \leq \delta$ . We use the biased estimator because it is nonnegative and hence a valid penalty; the unbiased U-statistic estimator (Song et al., 2012) can be negative in finite samples, though both are consistent.

The penalty is a joint-sample statistic on $\{ ( h ( X _ { i } ) , S _ { i } ) \}$ and does not construct local estimates of $P _ { Z \mid S = s }$ . The concentration result below is stated for fixed kernels. In the experiments the kernel bandwidths are set by the median heuristic and the resulting statistic is used as a practical training penalty; the theorem isolates the fixed-kernel statistical behavior.

5.2. Uniform train-to-population control. Empirical HSIC concentrates on its population value uniformly over the encoder class $\mathcal { H } .$ , so the data-dependent encoder that minimizes the FRHSIC penalty still has small population dependence, up to a uniform error of order $n ^ { - 1 / 2 }$ under controlled encoder complexity. Uniformity is essential because the penalty is evaluated on an encoder selected from the same sample, so a pointwise concentration bound for a fixed encoder would not sufice.

Theorem 5.1 (Uniform concentration of empirical HSIC over the encoder class). Let $\{ ( X _ { i } , S _ { i } ) \} _ { i = 1 } ^ { n }$ be i.i.d. copies of $( X , S )$ , and let H be a class of encoders $h : \mathcal { X } $ $\mathcal { Z } \subseteq \mathbb { R } ^ { d _ { \mathcal { Z } } }$ . For $h \in \mathcal H$ write $Z ^ { h } : = h ( X )$ and $Z _ { i } ^ { h } : = h ( X _ { i } )$ . Let $k _ { \mathcal { Z } } , k _ { \mathcal { S } }$ be fixed kernels with feature maps $\phi _ { \mathcal { Z } } ( z ) : = k _ { \mathcal { Z } } ( z , \cdot ) \in \mathcal { F } _ { \mathcal { Z } }$ and $\phi _ { S } ( s ) : = k _ { S } ( s , \cdot ) \in \mathcal { F } _ { S }$ , satisfying the boundedness and Lipschitz conditions

$$
\operatorname* { s u p } _ { z \in \mathcal { Z } } k _ { \mathcal { Z } } ( z , z ) \leq \nu _ { \mathcal { Z } } , \qquad \operatorname* { s u p } _ { s \in \mathcal { S } } k _ { \mathcal { S } } ( s , s ) \leq \nu _ { \mathcal { S } } ,
$$

$$
\| \phi _ { \mathcal { Z } } ( z ) - \phi _ { \mathcal { Z } } ( z ^ { \prime } ) \| _ { \mathcal { F } _ { \mathcal { Z } } } \le \ell _ { \mathcal { Z } } \| z - z ^ { \prime } \| , \qquad \| \phi _ { \mathcal { S } } ( s ) - \phi _ { \mathcal { S } } ( s ^ { \prime } ) \| _ { \mathcal { F } _ { \mathcal { S } } } \le \ell _ { \mathcal { S } } \| s - s ^ { \prime } \| ,
$$

where $k _ { \mathcal { Z } } ( z , z ) ~ = ~ \| \phi _ { \mathcal { Z } } ( z ) \| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 }$ and $k _ { S } ( s , s ) ~ = ~ \| \phi _ { S } ( s ) \| _ { \mathcal { F } _ { S } } ^ { 2 }$ . For $\textit { h } \in \textit { \textbf { \mathscr { H } } }$ , write $\widehat { \mathrm { H S I C } } _ { n } ( h )$ for the biased empirical HSIC estimator (4.4) computed from the paired sample $\{ ( Z _ { i } ^ { h } , S _ { i } ) \} _ { i = 1 } ^ { n } ,$ , and $\mathrm { H S I C } ( h )$ for the population HSIC of Definition 2.1 with $Z \ = \ Z ^ { h } ,$ ; these are the estimator and functional already defined, evaluated at the encoder-transformed representation $Z ^ { h }$ with the fixed kernels $k _ { \mathcal { Z } } , k _ { \mathcal { S } }$ unchanged. Let

$$
\begin{array} { r } { \widehat { \mathcal { G } } _ { n } ( \mathcal { H } ) : = \mathbb { E } _ { \boldsymbol { \xi } } \Big [ \underset { h \in \mathcal { H } } { \operatorname* { s u p } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \langle \xi _ { i } , h ( X _ { i } ) \rangle \Big | X _ { 1 } , \ldots , X _ { n } \Big ] , \quad \xi _ { i } \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( 0 , I _ { d _ { \boldsymbol { z } } } ) , } \end{array}
$$

be the empirical Gaussian complexity of H, and set $\mathcal { G } _ { n } ( \mathcal { H } ) : = \mathbb { E } [ \widehat { \mathcal { G } } _ { n } ( \mathcal { H } ) ]$ . Define

$$
\begin{array} { r } { B _ { n } ( \mathcal { H } , \delta ) : = 8 \sqrt { 2 } \nu _ { \mathcal { Z } } \nu _ { S } \sqrt { \frac { \log ( 2 / \delta ) } { n } } + \frac { 4 \nu _ { \mathcal { Z } } \nu _ { S } } { n } + 4 8 \sqrt { \pi } \operatorname* { m a x } \{ \nu _ { \mathcal { Z } } \ell _ { S } , \nu _ { S } \ell _ { \mathcal { Z } } \} \mathcal { G } _ { n } ( \mathcal { H } ) . } \end{array}\tag{5.2}
$$

Then, for every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$

$$
\operatorname* { s u p } _ { h \in \mathcal { H } } \left| \widehat { \mathrm { H S I C } } _ { n } ( h ) - \mathrm { H S I C } ( h ) \right| \leq B _ { n } ( \mathcal { H } , \delta ) .\tag{5.3}
$$

Proof. See Appendix G.2. The proof maps FRHSIC to the uniform HSIC concentration theorem of Ni and Huo (2024): for each h, the empirical HSIC (4.4) at $\{ ( Z _ { i } ^ { h } , S _ { i } ) \} _ { i = 1 } ^ { n }$ is the biased estimator for the product kernel $\kappa _ { h } \big ( ( x , s ) , ( x ^ { \prime } , s ^ { \prime } ) \big ) \ : =$ $k _ { \mathcal { Z } } ( h ( x ) , h ( x ^ { \prime } ) ) k _ { \mathcal { S } } ( s , s ^ { \prime } )$ ; the sensitive coordinate is untransformed, so the singleton class on $s$ contributes zero Gaussian complexity; and the cited result applied to $\{ \kappa _ { h } : h \in \mathcal { H } \}$ yields (5.3) with the constants of (5.2) (Proposition G.2). □

Corollary 5.2 (Population HSIC control for the learned encoder). Suppose the assumptions of Theorem 5.1 hold, and let $\widehat { h } \in \mathcal { H }$ be any data-dependent encoder selected from the same sample, including one obtained by minimizing the FRHSIC objective (5.1). Then, with probability at least $1 - \delta$

$$
\operatorname { H S I C } ( \widehat { h } ) \leq \widehat { \operatorname { H S I C } } _ { n } ( \widehat { h } ) + B _ { n } ( \mathcal { H } , \delta ) .
$$

Proof. The event in Theorem 5.1 holds simultaneously for every $h \in \mathcal H$ . Since $\widehat { h } \in \mathcal { H }$ , evaluating (5.3) at $h = \widehat { h }$ gives the claim. □

Corollary 5.2 is the train-to-population bridge: small empirical HSIC for the selected encoder implies small population dependence of the learned representation, up to the uniform concentration error, independently of the stochastic optimization used to minimize (5.1).

Corollary 5.3 (Root-n rate under controlled encoder complexity). Under $t h e$ assumptions of Theorem $5 . 1 ,$ suppose $\mathcal { G } _ { n } ( \mathcal { H } ) = O ( n ^ { - 1 / 2 } )$ . Then, for every fixed $\delta \in$ (0, 1),

$$
\operatorname* { s u p } _ { h \in \mathcal { H } } \left| \widehat { \mathrm { H S I C } } _ { n } ( h ) - \mathrm { H S I C } ( h ) \right| = \widetilde { O } _ { p } ( n ^ { - 1 / 2 } ) ,
$$

where $\widetilde O$ hides logarithmic factors, and the same rate controls the learned encoder $\widehat { h }$ through Corollary 5.2.

Proof. Substituting $\mathcal { G } _ { n } ( \mathcal { H } ) = O ( n ^ { - 1 / 2 } )$ into (5.2) gives $B _ { n } ( \mathcal { H } , \delta ) = \widetilde { O } ( n ^ { - 1 / 2 } )$ ; the claim then follows from (5.3) and Corollary 5.2. □

Encoder classes covered by the rate. The condition $\mathcal { G } _ { n } ( \mathcal { H } ) = O ( n ^ { - 1 / 2 } )$ is a fixedefective-complexity condition on the feasible encoder class. For bounded linear encoders $\mathcal { H } _ { \mathrm { l i n } } = \{ h _ { W } ( x ) = W x : \| W \| _ { F } \leq R \}$ with $\| X \| \leq M$ almost surely,

$$
\begin{array} { r } { \mathcal { \widehat { G } } _ { n } ( \mathcal { H } _ { \mathrm { l i n } } ) \leq R \left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \xi _ { i } X _ { i } ^ { \top } \right\| _ { F } , \qquad \mathrm { h e n c e } \qquad \mathcal { G } _ { n } ( \mathcal { H } _ { \mathrm { l i n } } ) \leq R M \sqrt { d \boldsymbol { z } / n } , } \end{array}
$$

which is $O ( n ^ { - 1 / 2 } )$ for fixed representation dimension $d _ { \mathcal { Z } }$ . The same $n ^ { - 1 / 2 }$ order holds for fixed-depth, fixed-width MLP encoders with bounded weights, bounded inputs, and Lipschitz activations, and for finite-dimensional parametric families over compact parameter sets, under the corresponding complexity bounds of Ni and Huo (2024). The rate does not extend to arbitrary unregularized or growing-width networks.

Implementation. Training optimizes (5.1) with minibatch stochastic gradient methods: on a minibatch of size m the HSIC penalty is the same trace formula (4.4) restricted to the batch, at $O ( m ^ { 2 } )$ cost. Kernel bandwidths use the median heuristic. The penalty extends to Equal Opportunity by restricting the statistic to the $Y = 1$ subset and to multiple sensitive attributes through a product kernel on the sensitive space, and λ is chosen by held-out validation; details are in Appendix E.2 and Appendix H.6.

6. Experimental Validation. The experiments are consistent with the theory: the empirical estimator follows the predicted $\bar { O } ( n ^ { - 1 / 2 } )$ rate, FRHSIC attains fairness– accuracy tradeofs comparable to conditional-route baselines while keeping fairness stable across fresh downstream heads, and it trains about 36× faster than FREM per epoch. We evaluate FRHSIC along four questions. §6.2 checks whether the empirical estimator follows the predicted convergence rate. §6.3 compares its fairness–accuracy tradeofs against the baselines on real data. §6.4 tests whether the enforced fairness transfers to fresh downstream prediction heads. §6.5 measures training time relative to conditional-route methods.

Code, cached results, and figures are available in a public repository at https: $/ / \mathrm { g i t h u b . c o m / Y i j i n 9 1 1 / F R H S I C }$ , with an archived DOI release accompanying the camera-ready version.

6.1. Experimental setup. We use five real datasets with continuous sensitive attributes. Adult, ACS Income, MEPS, and COMPAS are classification tasks scored by accuracy; Crime is a regression task scored by mean squared error. Fairness is measured by held-out $\Delta _ { \mathrm { G D P } }$ , with lower values indicating less dependence on the sensitive attribute S.

• Adult: $n \approx 3 0 { , } 0 0 0$ individuals; income $\mathrm { > 5 0 K ; }$ ; sensitive attribute age.

• ACS Income: $n = 2 0 { , } 0 0 0$ from the 2018 California American Community Survey via folktables (Ding et al., 2021); income $> 5 0 \mathrm { K } ;$ age.

• MEPS: $n \approx 1 3 , 0 0 0$ from the 2015 Medical Expenditure Panel Survey (Romano et al., 2020); healthcare utilization $\geq 1 0$ visits; age.

• Crime (the Communities and Crime dataset): n ≈ 2,000 communities; violent crimes per population; racial composition.

• COMPAS: $n \approx 6 { , } 0 0 0$ defendants; two-year recidivism; age.

Features and S are min–max scaled to [0, 1], fit on the training split only. We compare FRHSIC with the continuous-sensitive methods FREM (Kong et al., 2025) and Reg GDP (Jiang et al., 2022), the adversarial method ADV (Grari et al., 2022), the binaryattribute methods MMD (Deka and Sutherland, 2023) and LAFTR (Madras et al., 2018) applied to quartile-binned $S ,$ and an unconstrained baseline (Unfair, $\lambda = 0 )$ Following Kong et al. (2025), all methods share a two-layer SELU encoder with hidden and representation dimension 50 and a linear prediction head, train for 200 epochs with Adam at learning rate $1 0 ^ { - 3 }$ , and are swept over $\lambda \in \{ 0 . 1 , 1 , 1 0 , 1 0 0 , 5 0 0 \}$ with results averaged over five random $8 0 / 2 0$ splits, reported as mean ± standard deviation. Each method keeps the kernel bandwidth convention of its source publication; we do not retune bandwidths to favor any method, and the only fairness-specific hyperparameter FRHSIC tunes is $\lambda ,$ with kernel bandwidths fixed by the median heuristic; λ is selected by the held-out procedure of Appendix H.6. Full preprocessing, hyperparameter grids, hardware, and the FREM 32-anchor subsampling approximation are documented in Appendix D.

6.2. Synthetic estimator behavior. Empirical HSIC converges to its population value at the predicted $O ( n ^ { - 1 / 2 } )$ rate, faster than the conditional-route EIPM estimator’s $O ( n ^ { - 2 / { \bar { 5 } } } )$ We draw (Z, S) jointly Gaussian with correlation $\rho \ : = \ : 0 . 5$ $( Z \sim \mathcal { N } ( 0 , 1 ) , \ S \ = \ \rho Z + \sqrt { 1 - \rho ^ { 2 } } \varepsilon , \ \varepsilon \sim \mathcal { N } ( 0 , 1 ) )$ ), approximate each estimator’s population value by its $n = 2 0 { , } 0 0 0$ estimate, and report the absolute error at sample sizes from 50 to 5,000, averaged over 20 replications with Gaussian kernels at bandwidth 1. The EIPM estimator is the weighted MMD estimator of Kong et al. (2025), the same estimator used by the FREM baseline. On this dependence, HSIC<sup>\</sup> converges at the $O ( n ^ { - 1 / 2 } )$ rate that underlies the uniform train-to-population control of §5.2, while the EIPM estimator converges at the slower $O ( n ^ { - 2 \bar { / } 5 } )$ rate. Figure 2 reports both estimation errors against sample size. By Theorem 4.5, HSIC and the conditional MMD integral bound each other up to an explicit spectral tail, so the joint estimator’s faster $O ( n ^ { - 1 / 2 } )$ rate is a statistical-eficiency gain in estimating a closely related fairness quantity, equal to the conditional MMD integral up to that tail. The conditional-gap bound of Theorem 4.7 is separately validated on a synthetic classification setup in Appendix E.1.

![](images/28320c320c8b264627cb98ae4159a3b759aeb6bba4460d212658375b8cd33e72.jpg)  
Fig. 2. Synthetic estimator convergence. Absolute estimation error is plotted against sample size on log–log axes. Empirical HSIC follows the predicted $O ( n ^ { - 1 / 2 } )$ behavior and the smoothingbased EIPM estimator follows the slower $O ( n ^ { - 2 / 5 } )$ trend: the fitted log–log slopes are −0.46 for HSIC and −0.44 for EIPM, close to the predicted −1/2 and −2/5. The dashed and dotted gray lines are the $n ^ { - 1 / 2 }$ and $n ^ { - 2 / 5 }$ references.

6.3. Fairness–accuracy tradeofs. On the five real datasets of §6.1, scored by held-out $\Delta _ { \mathrm { G D P } }$ against accuracy (mean squared error for Crime), FRHSIC achieves fairness–accuracy tradeofs comparable to the strongest continuous-sensitive baselines, although it is not uniformly best on any dataset. Because HSIC and GDP penalties have diferent scales, equal λ does not mean equal fairness, so Table 1 compares at a matched operating point: for each method we report the lowest $\Delta _ { \mathrm { G D P } }$ among models whose prediction performance stays within 1% of the Unfair baseline, together with the bandwidth-free normalized area under the accuracy–fairness Pareto envelope, nAUP. Figures 3 and 4 show the full frontiers over the λ sweep, for the classification datasets and for Crime respectively.

No method dominates across datasets. By nAUP, FRHSIC is best on COMPAS, within .004 of the best on Crime, and mid-pack on Adult, ACS Income, and MEPS, staying within the spread of FREM, Reg-GDP, and the adversarial baselines. Among the binned baselines, LAFTR is consistently among the weakest by nAUP, while MMD on quartile-binned S is competitive on Adult and ACS Income; binning S thus yields no systematic advantage over the continuous-sensitive methods on these datasets. The matched band also removes a Reg-GDP artifact: Reg-GDP reaches near-zero $\Delta _ { \mathrm { G D P } }$ only by collapsing accuracy outside the 1% band, from .849 to .758 on Adult and from .790 to .590 on ACS Income at its unconstrained point. Diferences within $\pm 0 . 0 1$ accuracy or ±0.05 on $\Delta _ { \mathrm { G D P } }$ fall inside the five-split spread and should be read as practically equivalent. COMPAS age is integer-valued with few distinct values, so its centered sensitive Gram matrix has small rank, and the finite-sample bound of Theorem 4.7 is governed by a few well-conditioned sensitive directions.

6.4. Transfer to fresh downstream heads. Fairness enforced by FRHSIC stays stable across fresh downstream heads, consistent with a penalty that constrains all dependence between Z and S rather than the single head used in training. On Adult (at λ = 10) and Crime $( \mathrm { a t } ~ \lambda = 1 0 0 )$ , we freeze the learned representation of each method and train four fresh heads on it: linear, two-layer MLP, random forest, and SVM. Table 2 reports the standard deviation of $\Delta _ { \mathrm { G D P } }$ across the four heads, a

## Table 1

Matched-performance comparison on the five real datasets. For each method we report the lowest ∆ among models whose performance stays within 1% of the Unfair baseline. Perf is accuracy, except Crime where lower MSE × 10<sup>2</sup> is better; ∆ lower is fairer; nAUP is the normalized area under the accuracy–fairness Pareto envelope, higher is better. Bold marks the best fairness method per column; “—” means no swept λ meets the band. Setup and bandwidth conventions are in §6.1.
<table><tr><td>Dataset</td><td>Method</td><td>Perf.</td><td>∆GDP (↓)</td><td>nAUP (↑)</td></tr><tr><td rowspan="6">Adult (Acc ↑)</td><td>Unfair</td><td>.849</td><td>.749</td><td></td></tr><tr><td>FRHSIC (Ours)</td><td>.845</td><td>.232</td><td>.786</td></tr><tr><td>FREM</td><td>.845</td><td>.167</td><td>.835</td></tr><tr><td>Reg-GDP</td><td>.842</td><td>.082</td><td>.872</td></tr><tr><td>ADV</td><td>.841</td><td>.466</td><td>.434</td></tr><tr><td>MMD (binned)</td><td>.847</td><td>.126</td><td>.880</td></tr><tr><td rowspan="7"></td><td>LAFTR (binned)</td><td>.842</td><td>.385</td><td>.555</td></tr><tr><td>Unfair</td><td>.790</td><td>.514</td><td></td></tr><tr><td>FRHSIC (Ours)</td><td>.783</td><td>.228</td><td>.805</td></tr><tr><td>FREM</td><td>.787</td><td>.274</td><td>.865</td></tr><tr><td>Reg-GDP</td><td>.787</td><td>.292</td><td>.852</td></tr><tr><td>ADV</td><td>.785 .783</td><td>.458</td><td>.713</td></tr><tr><td>MMD (binned) LAFTR (binned)</td><td>.787</td><td>.157 .474</td><td>.881 .754</td></tr><tr><td rowspan="7">MEPS (Acc ↑)</td><td>Unfair</td><td>.804</td><td>.626</td><td></td></tr><tr><td>FRHSIC (Ours)</td><td>.797</td><td>.329</td><td>.464</td></tr><tr><td>FREM</td><td>.800</td><td>.317</td><td>.549</td></tr><tr><td>Reg-GDP</td><td>.801</td><td>.370</td><td>.469</td></tr><tr><td>ADV</td><td>.803</td><td>.480</td><td>.487</td></tr><tr><td>MMD (binned)</td><td>.800</td><td>.407</td><td>.492</td></tr><tr><td>LAFTR (binned)</td><td>.803</td><td>.530</td><td>.477</td></tr><tr><td rowspan="7">Crime (MSE ↓)</td><td>Unfair</td><td>1.91</td><td>.029</td><td></td></tr><tr><td>FRHSIC (Ours)</td><td>1.91</td><td>.029</td><td>.686</td></tr><tr><td>FREM</td><td>1.87</td><td>.023</td><td>.680</td></tr><tr><td>Reg-GDP</td><td>1.91</td><td>.025</td><td>.690</td></tr><tr><td>ADV</td><td>1.92</td><td>.029</td><td>.002</td></tr><tr><td>MMD (binned)</td><td></td><td></td><td>.573</td></tr><tr><td>LAFTR (binned)</td><td>1.89</td><td>.029</td><td>.103</td></tr><tr><td rowspan="7">COMPAS (Acc ↑)</td><td>Unfair</td><td>.654</td><td>.123</td><td></td></tr><tr><td>FRHSIC (Ours)</td><td>.649</td><td>.090</td><td>.679</td></tr><tr><td>FREM</td><td>.648</td><td>.101</td><td>.530</td></tr><tr><td>Reg-GDP</td><td>.650</td><td>.079</td><td>.480</td></tr><tr><td>ADV</td><td>.649</td><td>.095</td><td>.314</td></tr><tr><td>MMD (binned)</td><td>.651</td><td>.082</td><td>.485</td></tr><tr><td>LAFTR (binned)</td><td>.655</td><td>.122</td><td>.475</td></tr></table>

stability diagnostic in which lower values indicate fairness less tied to the head used during training. FRHSIC’s cross-head variability is comparable to FREM and, on Adult, well below the Unfair baseline. Reg-GDP attains lower variability on Adult, but only at the accuracy collapse noted in §6.3. This is a stability diagnostic rather than a guarantee that every downstream head is fair; the full per-head breakdown is in Appendix H.2.

![](images/62d26492d392c913dd7ffb5d78fe073a86c2e3c7012fc66c47d0a4a4f97b952c.jpg)  
Fig. 3. Fairness–accuracy frontiers on the classification datasets. Each point corresponds to one value of the fairness regularization parameter λ; lower ∆ is fairer and higher accuracy is better. Lines connect the Pareto-eficient points of each method. Axes are scaled separately by dataset to show the frontier structure. The black star marks the Unfair baseline, the unconstrained model trained without any fairness penalty $( \lambda = 0 )$ . FRHSIC is competitive with continuous-sensitive baselines across datasets, but no method uniformly dominates. Table 1 gives the matched-performance summary.

Table 2  
Cross-head $\Delta _ { \mathrm { G D P } }$ standard deviation across four fresh heads (linear, MLP, random forest, SVM) trained on a single frozen representation at λ = 10 (Adult) and λ = 100 (Crime); lower indicates fairness less tied to the training head. Full per-head results are in Appendix H.2.
<table><tr><td>Method</td><td>Adult (λ = 10)</td><td>Crime (λ = 100)</td></tr><tr><td>Unfair</td><td>0.348</td><td></td></tr><tr><td>FRHSIC (Ours)</td><td>0.244</td><td>0.001</td></tr><tr><td>FREM</td><td>0.247</td><td>0.002</td></tr><tr><td>Reg-GDP</td><td>0.058†</td><td>0.002</td></tr><tr><td>ADV</td><td>0.252</td><td>0.005</td></tr></table>

<sup>†</sup> Reg-GDP attains lower cross-head variance on Adult but with accuracy collapsing to 0.758 (Table 1).

6.5. Computational eficiency. FRHSIC trains substantially faster than FREM because it replaces FREM’s per-sample weighted conditional-MMD computation with a single $O ( m ^ { 2 } )$ HSIC trace per minibatch. Figure 5 reports per-epoch wall-clock time, averaged over five epochs after a warm-up epoch, on a synthetic binary-classification task $( S \sim \mathrm { U n i f } ( 0 , 1 )$ , features $\boldsymbol { X } = ( \boldsymbol { S } + \varepsilon , \varepsilon ^ { \prime } )$ with $\varepsilon \sim \mathcal { N } ( 0 , 0 . 3 ^ { 2 } )$ and $\varepsilon ^ { \prime } \sim \mathcal { N } ( 0 , 1 )$ , label $Y \ \sim \ \mathrm { B e r n o u l l i } ( \sigma ( X _ { 1 } ) ) )$ of varying size, at batch size 256 from $n = 5 0 0 \ { \mathrm { t o } } \ n = 2 0 , 0 0 0$ , with all methods sharing the encoder and prediction head. $\mathrm { A t } \ n = 2 0 { , } 0 0 0$ and batch size 256, FRHSIC completes an epoch in 2.82s versus $1 0 0 . 7 \mathrm { s }$ for FREM, a roughly 36× speedup. Reg-GDP is faster still at 1.69s because it evaluates conditional expectations on a fixed grid rather than kernel matrices, but it optimizes a head-specific smoothed moment criterion, whereas FRHSIC penalizes a representation-level distributional dependence that binds every downstream head. The full-batch $O ( n ^ { 2 } )$ versus $O ( n ^ { 3 } )$ analysis is given in Appendix D.

![](images/d725617e44bcb97b5bc79e90f2aabd830caf113b677f774a2e2533abb6dbb9b5.jpg)

Fig. 4. Fairness–prediction-error frontier on Crime. Each point corresponds to one value of λ; lower $\Delta _ { \mathrm { G D P } }$ is fairer and lower MSE is better. Lines connect the Pareto-eficient points of each method; the black star marks the Unfair baseline $( \lambda = 0 ;$ , no fairness penalty). The separate panel avoids mixing regression error with classification accuracy.  
![](images/4f5479397ddadb67d1ed73dd61abfe9e623dadb8b17a3a9a7502938c3d830f60.jpg)  
Fig. 5. Per-epoch wall-clock time versus sample size at batch size 256, on log–log axes; error bars are ±1 standard deviation over repeats and are negligible except for FRHSIC at the smallest n. FRHSIC uses one HSIC trace per batch and trains about 36× faster than FREM at $n = 2 0 , 0 0 0 \ ( \xi 6 . 5 )$ . Reg-GDP is faster but optimizes a head-specific smoothed criterion rather than a representation-level distributional penalty.

Additional experiments are reported in Appendices H.2–H.6: an empirical tightness sweep for the Theorem 4.7 bound, a high-dimensional sensitive-attribute comparison, a mutual-information diagnostic, ablations over kernel choice and representation dimensionality, and validation of the λ-selection rule of Appendix H.6.

## 7. Related Work.

Fair representation learning. Zemel et al. (2013) introduced the framework of mapping data to a latent space that preserves task information while obfuscating S. Subsequent work extended this through variational autoencoders (Louizos et al., 2016), adversarial training (Madras et al., 2018), disentangled representations (Creager et al., 2019), MMD- and Sinkhorn-based formulations (Cho et al., 2020a; Deka and Sutherland, 2023), and data-domain transformations (Quadrianto et al., 2019). Most of these methods are designed for categorical sensitive attributes and discretize continuous $S$ at training time.

Continuous-sensitive-attribute fairness criteria. GDP (Jiang et al., 2022) is estimated via Nadaraya–Watson kernel smoothing on S. EIPM/FREM (Kong et al., 2025) extends GDP to higher moments via a weighted empirical MMD with bandwidth $\gamma _ { n }$ and leave-one-out correction. Kernel HGR / NLD (Mary et al., 2019; Grari et al., 2022) estimates a global dependence criterion via regularized kernel canonical correlation. Mutual information (Cho et al., 2020b) is estimated via density or copula approximations. Theorem 3.2 of §3.2 disintegrates the joint-vs-product IPM into a $P _ { S } .$ -weighted conditional contrast, and Corollary 3.4 recovers the conditional-integral IPM that GDP and EIPM instantiate when the witness class decomposes across s; kHGR is a supremum-of-correlations criterion outside this form.

HSIC and dependence measures. HSIC was introduced by Gretton et al. (2005) and has been used for independence testing (Gretton et al., 2007; Albert et al., 2022), feature selection (Song et al., 2007, 2012), dimensionality reduction (Ma et al., 2018), and as a training objective for deep networks (Ma et al., 2020); its equivalence with distance covariance (Sejdinovic et al., 2013) places it within a single family of jointdistribution dependence measures. HSIC has also been used directly as a fairness penalty: P´erez-Suay et al. (2017) introduced it as the fairness term in kernel regression and dimensionality reduction, Li et al. (2022) added closed-form ridge solutions and a Gaussian-process treatment, and Quadrianto et al. (2019) used it for independence from the sensitive attribute in a data-domain model. Our contribution is the continuous-S disintegration that ties the HSIC penalty to the conditional-integral criteria (Theorem 3.2, Corollary 3.4) and the spectral and train-to-population guarantees (Theorems 4.5 and 5.1) that these empirical and ridge-based treatments lack.

## 8. Discussion.

The joint-distribution view. The joint-distribution view reaches the same independence target as the conditional-integral criteria but changes the object that is estimated. Rather than a conditional law at each sensitive value, it estimates a single closed-form V-statistic, which controls smoothed conditional deviations and, under spectral regularity, the conditional MMD integral up to a spectral tail (Theorems 4.3 and 4.5).

Extensions. The framework extends to representation-level Equal Opportunity by computing the joint discrepancy on the $Y = 1$ subset, and to multiple continuous sensitive attributes by taking a product kernel on $S _ { 1 } \times S _ { 2 }$ . Both extensions are derived in Appendix E.2. Replacing the reference distribution $P _ { Z } \otimes P _ { S }$ by $P _ { Z } \otimes \operatorname { U n i f } ( S )$ yields a minority-protection target in which rare values of S receive equal weight; this is a modeling choice and does not afect the structural identity of Theorem 3.2.

Open questions. Two questions remain. First, the empirical bound in Theorem 4.7 scales as $\| f \| _ { \mathcal { F } _ { Z } } ^ { 2 } / \hat { \lambda } _ { m } ;$ ; tighter spectral arguments exploiting the full singularvalue distribution of $\widetilde { L }$ would yield sharper guarantees on the integral form. Second, the framework treats the joint discrepancy with an RKHS witness class; non-RKHS instances (Wasserstein, total variation) inherit the structural identity but lose the $O ( n ^ { - 1 / 2 } )$ rate and closed-form estimator. We leave these to future work.

## Appendix A. Proofs.

This section collects the proofs of the results stated in the main text, in the order in which the results appear there. The technical lemma used in the proof of Proposition 4.1 (Lemma B.1) is recorded in Appendix B below.

## A.1. Proof of Proposition 3.1: conditions (1)–(4).

Proof. We show $( i ) \Rightarrow ( i i ) \Rightarrow ( i i i ) \Rightarrow ( i v ) \Rightarrow ( i )$

$( i ) \Rightarrow ( i i )$ . Since $\mathcal { F }$ is characteristic, $\mathrm { I P M } _ { \mathcal { F } } ( P _ { Z , S } , P _ { Z } \otimes P _ { S } ) = 0$ implies $P _ { Z , S } =$ $P _ { Z } \otimes P _ { S }$

$( i i ) \Rightarrow ( i i i )$ . On the Polish product space ${ \mathcal { Z } } \times { \mathcal { S } }$ , the disintegration theorem (Kallenberg, 2002, Thm. 5.4) yields a regular conditional probability $\{ P _ { Z | S = s } \} _ { s \in { \mathcal { S } } }$ such that for every Borel $A \subseteq { \mathcal { Z } } , B \subseteq S$ ，

$$
P _ { Z , S } ( A \times B ) = \int _ { B } P _ { Z | S = s } ( A ) d P _ { S } ( s ) .
$$

Under $( i i )$ , the same Borel set $A \times B$ also satisfies $P _ { Z , S } ( A \times B ) = P _ { Z } ( A ) P _ { S } ( B ) =$ $\textstyle \int _ { B } P _ { Z } ( A ) d P _ { S } ( s )$ . Comparing the two expressions, $\begin{array} { r } { \int _ { B } [ P _ { Z | S = s } ( A ) - P _ { Z } ( A ) ] d P _ { S } ( s ) = 0 } \end{array}$ for every Borel $B ,$ so $P _ { Z | S = s } ( A ) = P _ { Z } ( A )$ for $P _ { S ^ { - } \mathrm { a } . \mathrm { e } . \ s }$ . Choose a countable algebra $\mathcal { A }$ generating the Borel σ-algebra of $\mathcal { Z }$ (which exists since $\mathcal { Z }$ is Polish); the exceptional $P _ { S }$ -null sets, taken over the countable family $\mathcal { A } .$ have countable union of $P _ { S }$ -measure zero. Hence $P _ { Z | S = s } | _ { A } = P _ { Z } | _ { A }$ for $P _ { S ^ { - } \mathrm { a . e . } ~ s . }$ and by the $\pi - \lambda$ theorem $P _ { Z | S = s } = P _ { Z }$ on the full Borel σ-algebra for $P _ { S ^ { - } \mathrm { a } . \mathrm { e } . \ s }$ . Equivalence of “some” and “every” regular conditional version follows from the $P _ { S ^ { - } \mathrm { a . e } }$ . uniqueness clause of the disintegration theorem.

$( i i i ) \Rightarrow ( i v )$ . For every bounded measurable $f$ and $P _ { S ^ { - } \mathrm { a . e . } ~ s . }$

$$
\mathbb { E } [ f ( Z ) \mid S = s ] = \int f ( z ) d P _ { Z \mid S = s } ( z ) = \int f ( z ) d P _ { Z } ( z ) = \mathbb { E } [ f ( Z ) ] ,
$$

where the second equality uses (iii).

$( i v ) \Rightarrow ( i )$ . Indicator functions of Borel sets are bounded and measurable, so applying (iv) with $f = \mathbf { 1 } _ { A }$ gives $P _ { Z | S = s } ( A ) = P _ { Z } ( A )$ for $P _ { S ^ { - } \mathrm { a } . \mathrm { e } . \ s }$ , for every Borel $A \subseteq { \mathcal { Z } }$ . Take a countable algebra A generating the Borel σ-algebra of the Polish space $\mathcal { Z } ;$ the exceptional $P _ { S ^ { - } \mathrm { n u l l } }$ sets over the countable family A have null union, so $P _ { Z | S = s } | _ { A } = P _ { Z } | _ { A }$ for $P _ { S ^ { - } \mathrm { a . e . } ~ s . }$ , and the $\pi - \lambda$ theorem gives $P _ { Z | S = s } = P _ { Z }$ for $P _ { S ^ { - } \mathrm { a . e } }$ s. Hence $Z \perp S , { \mathrm { i . e . , } } P _ { Z , S } = P _ { Z } \otimes P _ { S }$ , so $\mathrm { I P M } _ { \mathcal { F } } ( P _ { Z , S } , P _ { Z } \otimes P _ { S } ) = 0$ □

## A.2. Proof of Proposition 3.1: the MMD-based EIPM instance.

Proof. (⇒) If $\mathrm { H S I C } ( Z , S ) = 0$ , then $\mu _ { Z , S } = \mu _ { Z } \otimes \mu _ { S }$ in ${ \mathcal { F } } _ { { \mathcal { Z } } \otimes { \mathcal { S } } }$ . Because $k z$ and $k _ { S }$ are characteristic, the product kernel $k \ z \otimes k \ s$ is I-characteristic, i.e., its joint mean embedding distinguishes $P _ { Z , S }$ from $P _ { Z } \otimes P _ { S }$ (Theorem 3 of Szab´o and Sriperumbudur, 2018, taking $M = 2 )$ . Hence $\mu _ { Z , S } = \mu _ { Z } \otimes \mu _ { S }$ implies $P _ { Z , S } = P _ { Z } \otimes P _ { S }$ ${ \mathrm { i . e . , ~ } } Z \perp S$ . Hence $P _ { Z | S = s } = P _ { Z }$ for $P _ { S ^ { - } } \mathrm { { a } }$ lmost every s, so $\mathrm { M M D } ( P _ { Z | S = s } , P _ { Z } ) = 0$ a.s., and $\mathrm { E I P M } _ { \mathcal { V } } ( Z ; S ) \stackrel { \cdot } { = } \mathbb { E } _ { S } [ \mathrm { M M D } ( P _ { Z | S } , P _ { Z } ) ] = 0$

(⇐) Suppose $\mathrm { E I P M } _ { \mathcal { V } } ( Z ; S ) = 0$ . Since $\mathrm { M M D } ( P _ { Z | S = s } , P _ { Z } ) \ge 0$ , this implies $\mathrm { M M D } ( P _ { Z | S = s } , P _ { Z } ) = 0$ for P<sub>S</sub>-almost every s. Because $k z$ is characteristic, MMD is a metric on probability distributions, so $P _ { Z | S = s } = P _ { Z } { \mathrm { a . s . , i . e . , } } Z \bot S .$ . Independence then implies that the joint mean embedding factorizes, $\mu _ { Z , S } = \mu _ { Z } \otimes \mu _ { S }$ , so $\operatorname { H S I C } ( Z , S ) = \| \mu _ { Z , S } - \mu _ { Z } \otimes \mu _ { S } \| _ { \mathcal { F } _ { Z \otimes S } } ^ { 2 } = 0 .$ □

A.3. Proof of Proposition 3.1: downstream-head (GDP) form. Since $k _ { \mathcal { Z } } \otimes k _ { \mathcal { S } }$ is characteristic, $\mathrm { H S I C } ( Z , S ) = 0$ is the product-RKHS instance of condition (1) of Proposition 3.1, so $P _ { Z | S = s } = P _ { Z }$ for $P _ { S } { \mathrm { - a l m o s t } }$ every s by condition (3), hence $\mathbb { E } [ f ( Z ) \mid S = s ] = \mathbb { E } [ f ( Z ) ]$ for P<sub>S</sub>-almost every s and every measurable $f : \mathcal { Z } \to$ R with $\mathbb { E } | f ( Z ) | < \infty$ . Therefore $\Delta _ { \mathrm { G D P } } ( f ) = \mathbb { E } _ { S } \left| \mathbb { E } [ f ( Z ) \mid S ] - \mathbb { E } [ f ( Z ) ] \right| = 0$

A.4. Proof of Theorem 3.2 (Disintegration of the joint-vs-product IPM). Because Z and S are standard Borel, the disintegration theorem (Kallenberg, 2002, Theorem 5.4) gives $P _ { Z , S } ( d z d s ) ~ = ~ P _ { Z \mid S = s } ( d z ) ~ P _ { S } ( d s )$ , and by construction $( P _ { Z } \otimes P _ { S } ) ( d z d s ) = P _ { Z } ( d z ) P _ { S } ( d s )$ . Fix $f ~ \in ~ { \mathcal { F } }$ with $\| f \| _ { \infty } ~ \le ~ b ~ < ~ \infty ~$ . Both $P _ { Z | S = s } ( d z ) P _ { S } ( d s )$ and $P _ { Z } ( d z ) P _ { S } ( d s )$ are probability measures and $| f | \leq b ,$ so each iterated integral below is absolutely convergent and the Fubini–Tonelli theorem applies:

$$
\mathbb { E } _ { P _ { Z , s } } [ f ] - \mathbb { E } _ { P _ { Z } \otimes P _ { S } } [ f ] = \int _ { S } \left( \int _ { { \mathcal Z } } f ( z , s ) P _ { Z | S = s } ( d z ) - \int _ { { \mathcal Z } } f ( z , s ) P _ { Z } ( d z ) \right) P _ { S } ( d s ) .
$$

The inner diference is $\begin{array} { r } { \int _ { \mathcal Z } f ( z , s ) d ( P _ { Z | S = s } - P _ { Z } ) ( z ) } \end{array}$ . Taking absolute values and the supremum over $f \in { \mathcal { F } }$ yields the identity.

A.5. Proof of Corollary 3.4 (Recovery of the conditional-integral IPM). By Theorem 3.2,

$$
\mathrm { I P M } _ { \mathcal { F } } ( P _ { Z , S } , P _ { Z } \otimes P _ { S } ) = \operatorname* { s u p } _ { f \in \mathcal { F } } \Big | \mathbb { E } _ { S } \int _ { \mathcal { Z } } f ( z , S ) d ( P _ { Z | S } - P _ { Z } ) ( z ) \Big | .
$$

Upper bound. Fix $f \in { \mathcal { F } }$ . By Definition $3 . 3 ( 2 ) , f ( \cdot , s ) \in \mathcal { V }$ for $P _ { S ^ { - } \mathrm { a . e . } ~ s . }$ , so for those s

$$
\Big | \int _ { \mathcal Z } f ( z , s ) d \big ( P _ { Z | S = s } - P _ { Z } \big ) ( z ) \Big | \leq \operatorname* { s u p } _ { g \in \mathcal { V } } \Big | \int _ { \mathcal Z } g d \big ( P _ { Z | S = s } - P _ { Z } \big ) \Big | = d _ { \mathcal { V } } \big ( P _ { Z | S = s } , P _ { Z } \big ) .
$$

By the triangle inequality for the P<sub>S</sub>-integral, $\begin{array} { r l } { \big | \mathbb { E } _ { S } \int f ( z , S ) d ( P _ { Z | S } - P _ { Z } ) \big | } & { { } \le } \end{array}$ $\mathbb { E } _ { S } \big [ d _ { \mathcal { V } } \big ( P _ { Z | S } , P _ { Z } \big ) \big ]$ ; taking the supremum over $f \in { \mathcal { F } }$ gives $\mathrm { I P M } _ { \mathcal { F } } \leq \mathbb { E } _ { S } [ d _ { \mathcal { V } } ( P _ { Z | S } , P _ { Z } ) ]$ Lower bound. Fix $\varepsilon \quad > \quad 0$ Since V is symmetric $( g ~ \in ~ \mathcal { V } ~ \Rightarrow ~ - g ^ { \cdot } \in ~ \mathcal { V } )$ $\begin{array} { r } { d _ { \mathcal { V } } ( P _ { Z | S = s } , P _ { Z } ) \ = \ \operatorname* { s u p } _ { q \in \mathcal { V } } \left( \int g d P _ { Z | S = s } - \int g d P _ { Z } \right) \ } \end{array}$ the absolute value being attained because $- g \in \bar { \mathcal { V } } . \quad \mathrm { B y }$ the selector assumption of Corollary 3.4 there is a measurable map $s \mapsto g _ { s } ^ { \varepsilon } \in \mathcal { V }$ with $\begin{array} { r } { \int g _ { s } ^ { \varepsilon } d ( P _ { Z | S = s } - P _ { Z } ) \ge d \nu ( P _ { Z | S = s } , P _ { Z } ) - \varepsilon } \end{array}$ for $P _ { S ^ { - } \mathrm { a . e . } ~ s . }$ . By Definition $3 . 3 ( 1 ) , f _ { \star } ( z , s ) : = g _ { s } ^ { \varepsilon } ( \dot { z } )$ lies in $\mathcal { F }$ , and since the integrand is nonnegative up to $\varepsilon ,$

$$
\mathrm { I P M } _ { \mathcal { F } } \geq \left| \mathbb { E } _ { S } \int g _ { S } ^ { \varepsilon } d \big ( P _ { Z | S } - P _ { Z } \big ) \right| \geq \mathbb { E } _ { S } \big [ d \nu \big ( P _ { Z | S } , P _ { Z } \big ) \big ] - \varepsilon .
$$

Letting $\varepsilon \downarrow 0$ gives $\mathrm { I P M } _ { \mathcal { F } } \geq \mathbb { E } _ { S } [ d _ { \mathcal { V } } ( P _ { Z | S } , P _ { Z } ) ]$ . Combining the two bounds and recalling $\mathcal { T } _ { d _ { \mathcal { V } } } ( Z ; S ) = \mathbb { E } _ { S } [ d _ { \mathcal { V } } ( P _ { Z | S } , P _ { Z } ) ]$ from the preliminaries yields the identity.

## A.6. Proof of Proposition 4.1 (HSIC and Expected MMD).

Proof. Step 1: Decomposition. By definition of the joint mean embedding and the tower property of conditional expectation,

$$
\mu _ { Z , S } = \mathbb { E } _ { Z , S } [ k _ { \mathcal { Z } } ( Z , \cdot ) \otimes k _ { \mathcal { S } } ( S , \cdot ) ] = \mathbb { E } _ { S } \left[ \mu _ { Z | S } \otimes k _ { \mathcal { S } } ( S , \cdot ) \right] ,
$$

and by the definition $\mu _ { S } = \mathbb { E } _ { S ^ { \prime } } [ k _ { S } ( S ^ { \prime } , \cdot ) ]$ and linearity of the tensor product (with $\mu _ { Z }$ constant in $S ^ { \prime } ) , \mu _ { Z } \otimes \mu _ { S } = \mathbb { E } _ { S ^ { \prime } } [ \mu _ { Z } \otimes k _ { S } ( S ^ { \prime } , \cdot ) ]$ ]. Subtracting and using linearity of the Bochner integral (Lemma B.1),

$$
\mu _ { Z , S } - \mu _ { Z } \otimes \mu _ { S } = \mathbb { E } _ { S } \left[ \left( \mu _ { Z | S } - \mu _ { Z } \right) \otimes k _ { S } ( S , \cdot ) \right] .\tag{A.1}
$$

Squaring the norm and writing it as an inner product gives, with $D _ { S } : = \left( \mu _ { Z | S } - \mu _ { Z } \right) ($ ⊗ $k _ { S } ( S , \cdot )$

$$
\begin{array} { r } { \mathrm { H S I C } ( Z , S ) = \langle \mathbb { E } _ { S } [ D _ { S } ] , \mathbb { E } _ { S ^ { \prime } } [ D _ { S ^ { \prime } } ] \rangle _ { \mathcal { F } _ { Z \otimes S } } } \\ { = \mathbb { E } _ { S , S ^ { \prime } } \left[ \langle D _ { S } , D _ { S ^ { \prime } } \rangle _ { \mathcal { F } _ { Z \otimes S } } \right] , } \end{array}
$$

where the exchange of expectation and inner product is justified by Lemma B.1 together with the bound $\| ( \mu _ { Z | S } - \mu _ { Z } ) \otimes k _ { S } ( S , \cdot ) \| _ { \mathcal { F } _ { Z \otimes S } } \leq 2 \sqrt { \kappa _ { Z } \kappa _ { S } }$ . By the tensorproduct inner product identity $\langle a \otimes b , a ^ { \prime } \otimes b ^ { \prime } \rangle = \langle \bar { a } , a ^ { \prime } \rangle \langle b , \dot { b } ^ { \prime } \rangle$ and the reproducing property $\langle k _ { S } ( S , \cdot ) , k _ { S } ( S ^ { \prime } , \cdot ) \rangle _ { \mathcal { F } _ { S } } = k _ { S } ( S , S ^ { \prime } )$ ,

$$
\mathrm { H S I C } ( Z , S ) = \mathbb { E } _ { S , S ^ { \prime } } \left[ \langle \mu _ { Z | S } - \mu _ { Z } , \mu _ { Z | S ^ { \prime } } - \mu _ { Z } \rangle _ { \mathcal { F } _ { Z } } \cdot k _ { S } ( S , S ^ { \prime } ) \right] ,
$$

which establishes (4.1).

Step 2: Squared expected MMD bound. Starting from (A.1), apply the triangle inequality (Jensen’s inequality for the convex norm):

$$
\begin{array} { r l } & { \mathrm { H S I C } ( Z , S ) ^ { 1 / 2 } = \left\| \mathbb { E } _ { S } [ ( \mu _ { Z | S } - \mu _ { Z } ) \otimes k _ { S } ( S , \cdot ) ] \right\| _ { \mathcal { F } _ { Z \otimes S } } } \\ & { \qquad \leq \mathbb { E } _ { S } \left[ \| ( \mu _ { Z | S } - \mu _ { Z } ) \otimes k _ { S } ( S , \cdot ) \| _ { \mathcal { F } _ { Z \otimes S } } \right] . } \end{array}
$$

The tensor-product norm factorizes: $\| a \otimes b \| _ { \mathcal { F } _ { \mathcal { Z } \otimes S } } = \| a \| _ { \mathcal { F } _ { \mathcal { Z } } } \| b \| _ { \mathcal { F } _ { S } }$ , and $\| k _ { S } ( S , \cdot ) \| _ { \mathcal { F } _ { S } } =$ $\sqrt { k _ { S } ( S , S ) } \le \sqrt { \kappa _ { S } }$ . Therefore

$$
\mathrm { H S I C } ( Z , S ) ^ { 1 / 2 } \leq \mathbb { E } _ { S } \Big [ \| \mu _ { Z | S } - \mu _ { Z } \| _ { \mathcal { F } _ { \Xi } } \sqrt { k _ { S } ( S , S ) } \Big ] \leq \sqrt { \kappa _ { S } } \cdot \mathbb { E } _ { S } \big [ \mathrm { M M D } _ { k _ { Z } } ( P _ { Z | S } , P _ { Z } ) \big ] .
$$

Squaring both sides yields (4.2).

## A.7. Proof of Theorem 4.3 (Smoothed conditional-distribution control by HSIC).

Proof. Since $k z$ is bounded, $\| \mu _ { Z | S = s } \| _ { \mathcal { F } _ { \mathcal { Z } } } \le \sqrt { \kappa _ { \mathcal { Z } } }$ and $\| \mu _ { Z } \| _ { \mathcal { F } _ { \mathcal { Z } } } \le \sqrt { \kappa _ { \mathcal { Z } } }$ , so $\Delta \in$ $L ^ { 2 } ( P _ { S } ; \mathcal { F } _ { \mathcal { Z } } )$ . The joint mean embedding satisfies

$$
\mu _ { Z , S } - \mu _ { Z } \otimes \mu _ { S } = \mathbb { E } _ { S } [ ( \mu _ { Z | S = S } - \mu _ { Z } ) \otimes k _ { S } ( S , \cdot ) ] = \mathbb { E } _ { S } [ \Delta ( S ) \otimes k _ { S } ( S , \cdot ) ] .
$$

Taking the squared norm in $\mathcal { F } _ { \mathcal { Z } \otimes \mathcal { S } }$ gives the first identity.

For $g \in { \mathcal { F } } _ { S }$ , the partial contraction of the tensor against $g$ over the $\mathcal { F } _ { \mathcal { S } }$ factor gives an element of $\mathcal { F } _ { \mathcal { Z } } \mathrm { i }$

$$
\begin{array} { r } { \left( \mathrm { i d } _ { \mathcal { F } _ { \mathcal { Z } } } \otimes \langle \cdot , g \rangle _ { \mathcal { F } _ { S } } \right) \left( \mathbb { E } _ { S } [ \Delta ( S ) \otimes k _ { S } ( S , \cdot ) ] \right) = \mathbb { E } _ { S } [ g ( S ) \Delta ( S ) ] . } \end{array}
$$

Hence

$$
\begin{array} { r } { \left\| \mathbb { E } _ { S } [ g ( S ) \Delta ( S ) ] \right\| _ { \mathcal { F } _ { z } } \le \| g \| _ { \mathcal { F } _ { s } } \left\| \mathbb { E } _ { S } [ \Delta ( S ) \otimes k _ { S } ( S , \cdot ) ] \right\| _ { \mathcal { F } _ { z \otimes s } } , } \end{array}
$$

which, with the first identity, proves (4.3). Two equivalent reformulations follow. Writing $m _ { f } ( s ) : = \mathbb { E } [ f ( Z ) \mid S = s ] - \mathbb { E } [ f ( Z ) ] = \langle f , \Delta ( s ) \rangle _ { \mathcal { F } _ { Z } }$ for $f \in \mathcal { F } _ { \mathcal Z }$ , the same contraction argument applied to the ${ \mathcal { F } } _ { { \mathcal { Z } } } .$ -coordinate gives the fixed-head form

$$
\big \| \mathbb { E } _ { S } [ m _ { f } ( S ) k _ { S } ( S , \cdot ) ] \big \| _ { \mathcal { F } _ { S } } ^ { 2 } \leq \| f \| _ { \mathcal { F } _ { Z } } ^ { 2 } \mathrm { H S I C } ( Z , S ) .\tag{A.2}
$$

Finally, since $\mathbb { E } _ { S } [ g ( S ) m _ { f } ( S ) ] \ = \ \big \langle f , \ \mathbb { E } _ { S } [ g ( S ) \Delta ( S ) ] \big \rangle _ { \mathcal { F } _ { Z } }$ , Cauchy–Schwarz together with (4.3) gives the bilinear form

$$
\big | \mathbb { E } _ { S } [ g ( S ) m _ { f } ( S ) ] \big | ^ { 2 } \leq \| f \| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 } \| g \| _ { \mathcal { F } _ { \mathcal { S } } } ^ { 2 } \mathrm { H S I C } ( Z , S )\tag{A.3}
$$

for all $f \in { \mathcal { F } } _ { \mathcal { Z } }$ and $g \in { \mathcal { F } } _ { S }$

A.8. Proof of Corollary 4.4 (Localized sensitive-value contrasts).

Proof. Apply Inequality (4.3) with $g \ = \ k _ { S } ( \cdot , s _ { 0 } ) \ \in \ { \mathcal { F } } _ { S } { \mathrm { : } }$ by the reproducing property $g ( S ) ~ = ~ k _ { S } ( S , s _ { 0 } )$ and $\| g \| _ { \mathcal { F } _ { S } } ^ { 2 } ~ = ~ \langle k _ { S } ( \cdot , s _ { 0 } ) , k _ { S } ( \cdot , s _ { 0 } ) \rangle _ { \mathcal { F } _ { S } } ~ = ~ k _ { S } ( s _ { 0 } , s _ { 0 } )$ , so $\big \| \mathbb { E } _ { S } [ k _ { S } ( S , s _ { 0 } ) \Delta ( S ) ] \big \| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 } \ \le \ k _ { S } ( s _ { 0 } , s _ { 0 } )$ HSIC(Z, S), which is the single-point bound since $\Delta ( S ) ~ = ~ \mu _ { Z \mid S } - \mu _ { Z } .$ The two-point form follows by the same argument with $g = k _ { S } ( \cdot , s _ { 0 } ) - k _ { S } ( \cdot , s _ { 1 } ) \in \mathcal { F } _ { S }$ , for which $g ( S ) = k _ { S } ( S , s _ { 0 } ) - k _ { S } ( S , s _ { 1 } )$ and $\| g \| _ { \mathcal { F } _ { \mathcal { S } } } ^ { 2 } = \| k _ { S } ( s _ { 0 } , \cdot ) - k _ { S } ( s _ { 1 } , \cdot ) \| _ { \mathcal { F } _ { \mathcal { S } } } ^ { 2 } .$ □

A.9. Proof of Theorem 4.5 (upper bound: spectral control of the conditional MMD integral).

Proof. Since $k _ { S }$ is bounded, $T _ { S }$ is a self-adjoint, positive, trace-class operator on $L ^ { 2 } ( P _ { S } )$ ; let $( \lambda _ { \ell } , \psi _ { \ell } ) _ { \ell \geq 1 }$ be its eigensystem with $\lambda _ { \ell } \geq 0$ and $\{ \psi _ { \ell } \}$ orthonormal in $L ^ { 2 } ( P _ { S } )$ , and recall the Mercer expansion $\begin{array} { r } { k _ { S } ( s , t ) = \sum _ { \ell \geq 1 } \lambda _ { \ell } \psi _ { \ell } ( s ) \psi _ { \ell } ( t ) } \end{array}$ in $L ^ { 2 } ( P _ { S } \otimes P _ { S } )$ . By Theorem 4.3,

$$
\begin{array} { r } { \mathrm { H S I C } ( Z , S ) = \left\| \mathbb { E } _ { S } [ \Delta ( S ) \otimes k _ { \mathcal { S } } ( S , \cdot ) ] \right\| _ { \mathcal { F } _ { Z \otimes \mathcal { S } } } ^ { 2 } . } \end{array}
$$

Substituting the Mercer expansion and writing $\begin{array} { r l r } { \Delta _ { \ell } } & { { } : = } & { \mathbb { E } _ { S } [ \Delta ( S ) \psi _ { \ell } ( S ) ] \quad = } \end{array}$ $\begin{array} { r } { \int \Delta ( s ) \psi _ { \ell } ( s ) d P _ { S } ( s ) \in \mathcal { F } _ { \mathcal Z } } \end{array}$ gives, by orthonormality of {ψ<sub>ℓ</sub>},

$$
\mathbb { E } _ { S } [ \Delta ( S ) \otimes k _ { S } ( S , \cdot ) ] = \sum _ { \ell \geq 1 } \lambda _ { \ell } \Delta _ { \ell } \otimes \psi _ { \ell } , \qquad { \mathrm { H S I C } } ( Z , S ) = \sum _ { \ell \geq 1 } \lambda _ { \ell } \| \Delta _ { \ell } \| _ { \mathcal { F } _ { Z } } ^ { 2 } ,
$$

which is the first identity. For any m with $\lambda _ { m } > 0$ , since $\lambda _ { \ell } \geq \lambda _ { m }$ for $\ell \leq m$ and every term is nonnegative,

$$
\sum _ { \ell = 1 } ^ { m } \| \Delta _ { \ell } \| _ { \mathcal { F } _ { z } } ^ { 2 } \leq \frac { 1 } { \lambda _ { m } } \sum _ { \ell = 1 } ^ { m } \lambda _ { \ell } \| \Delta _ { \ell } \| _ { \mathcal { F } _ { z } } ^ { 2 } \leq \frac { 1 } { \lambda _ { m } } \mathrm { H S I C } ( Z , S ) .
$$

Because $\{ \psi _ { \ell } \}$ is orthonormal in $L ^ { 2 } ( P _ { S } )$ , the $\mathcal { F } _ { \mathcal { Z } }$ -valued coeficients of $\Delta$ are $\Delta _ { \ell } .$ , so $\begin{array} { r } { \| \dot { P } _ { m } \hat { \Delta } \| _ { L ^ { 2 } ( P _ { S } ; \mathcal { F } _ { \mathcal { Z } } ) } ^ { 2 } \ = \ \sum _ { \ell = 1 } ^ { m } \| \Delta _ { \ell } \| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 } , } \end{array}$ , giving the projected bound. Finally, $P _ { m }$ is an orthogonal projection on $L ^ { 2 } ( P _ { S } ; \mathcal { F } _ { \mathcal { Z } } )$ , so $\| \Delta \| ^ { 2 } ~ = ~ \| P _ { m } \Delta \| ^ { 2 } + \| ( I -$ $P _ { m } ) \Delta \| ^ { 2 } \ \leq \ \lambda _ { m } ^ { - 1 } \bar { \mathrm { H S I C } } ( \bar { Z } , S ) + \rho _ { m } ^ { 2 }$ ; the identity $\| \Delta \| _ { L ^ { 2 } ( P _ { S } ; \mathcal { F } _ { \mathcal { Z } } ) } ^ { 2 } = \mathbb { E } _ { S } [ \| \Delta ( S ) \| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 } ] =$ $\mathbb { E } _ { S } [ \mathrm { M M D } _ { k _ { z } } ^ { 2 } ( P _ { Z | S } , P _ { Z } ) ]$ uses $\| \Delta ( s ) \| _ { \mathcal { F } _ { z } } = \mathrm { M M D } _ { k _ { z } } ( P _ { Z | S = s } , P _ { Z } )$ □

## A.10. Proof of Corollary 4.6 (Spectral control of GDP for RKHS heads).

Proof. By the reproducing property, $m _ { f } ( s ) \ = \ \mathbb { E } [ f ( Z ) \mid S = s ] - \mathbb { E } [ f ( Z ) ] \ =$ $\langle f , \Delta ( s ) \rangle _ { \mathcal { F } _ { \mathcal { Z } } }$ , so $m _ { f } ~ \in ~ L ^ { 2 } ( P _ { S } )$ with $L ^ { 2 } ( \boldsymbol { P _ { S } } )$ coeficients $\begin{array} { r l } { \int m _ { f } ( s ) \psi _ { \ell } ( s ) d P _ { S } ( s ) } & { { } = } \end{array}$ $\langle f , \Delta _ { \ell } \rangle _ { \mathcal { F } _ { \mathcal { Z } } } ,$ , where $\dot { \Delta _ { \ell } } : = \mathbb { E } _ { S } [ \Delta ( S ) \psi _ { \ell } ( S ) ]$ as in the proof of Theorem 4.5. Hence, by Cauchy–Schwarz and that theorem,

$$
\| P _ { m } m _ { f } \| _ { L ^ { 2 } ( P _ { S } ) } ^ { 2 } = \sum _ { \ell = 1 } ^ { m } \langle f , \Delta _ { \ell } \rangle _ { \mathcal { F } _ { Z } } ^ { 2 } \leq \| f \| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 } \sum _ { \ell = 1 } ^ { m } \| \Delta _ { \ell } \| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 } \leq \frac { \| f \| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 } } { \lambda _ { m } } \mathrm { H S I C } ( Z , S ) .
$$

Since $P _ { m }$ is an orthogonal projection on $L ^ { 2 } ( P _ { S } ) , \| m _ { f } \| _ { L ^ { 2 } ( P _ { S } ) } ^ { 2 } = \| P _ { m } m _ { f } \| ^ { 2 } + \| ( I -$ $P _ { m } ) m _ { f } \| ^ { 2 } \leq \lambda _ { m } ^ { - 1 } \| f \| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 } \mathrm { H S I C } ( Z , S ) + r _ { m , f } ^ { 2 }$ . Finally, by Jensen’s inequality $\Delta _ { \mathrm { G D P } } ( f ) =$ $\mathbb { E } _ { S } | m _ { f } ( S ) | \leq ( \mathbb { E } _ { S } m _ { f } ( S ) ^ { 2 } ) ^ { 1 / 2 } = \| m _ { f } \| _ { L ^ { 2 } ( P _ { S } ) }$ , so $\Delta _ { \mathrm { G D P } } ( f ) ^ { 2 } \leq \lambda _ { m } ^ { - 1 } \| f \| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 } \mathrm { H S I C } ( Z , S ) +$ $r _ { m , f } ^ { 2 } .$ □

A.11. Proof of Theorem 4.7 (Empirical demographic-parity control by HSIC).

Proof of Theorem 4.7. We work with empirical quantities throughout. Denote by $\hat { d } _ { S _ { i } } : = k _ { \mathcal { Z } } ( Z _ { i } , \cdot ) - \hat { \mu } _ { Z }$ the empirical centered embedding at $S = S _ { i }$ , with $\hat { \mu } _ { Z } =$ $\begin{array} { r } { n ^ { - 1 } \sum _ { j = 1 } ^ { n } k \boldsymbol { z } ( Z _ { j } , \cdot ) } \end{array}$ , and write $\delta _ { f } : = ( \widehat { \delta } _ { f , 1 } , \ldots , \widehat { \delta } _ { f , n } ) ^ { \top }$ for the centered predictiondeviation vector.

Step 1: Spectral identity for HSIC<sup>\</sup> using positive eigenpairs only. Let $K \in \mathbb { R } ^ { n \times n }$ have entries $K _ { i j } = k _ { \mathcal { Z } } ( Z _ { i } , Z _ { j } )$ and $\tilde { K } = H K H$ . The biased V-statistic estimator is

$$
{ \widehat { \mathrm { H S I C } } } ( Z , S ) = { \frac { 1 } { n ^ { 2 } } } \mathrm { t r } ( K H L H ) = { \frac { 1 } { n ^ { 2 } } } \mathrm { t r } ( { \widetilde { K } } { \widetilde { L } } ) .
$$

Since $L$ is symmetric positive semi-definite and $H = H ^ { \top }$ with $H ^ { 2 } = H , \widetilde { L } = H L H$ is symmetric PSD. Let $\hat { \lambda } _ { 1 } \geq \dots \geq \hat { \lambda } _ { r } > 0$ be the positive eigenvalues of $n ^ { - 1 } \widetilde L$ with associated orthonormal eigenvectors $\boldsymbol { \hat { e } } _ { 1 } , \dots , \boldsymbol { \hat { e } } _ { r } \in \mathbb { R } ^ { n }$ , where $r = \mathrm { r a n k } ( \widetilde { L } )$ . Then

$$
\frac 1 n \widetilde L = \sum _ { j = 1 } ^ { r } \hat { \lambda } _ { j } \hat { e } _ { j } \hat { e } _ { j } ^ { \top } , \qquad P _ { m } = \sum _ { j = 1 } ^ { m } \hat { e } _ { j } \hat { e } _ { j } ^ { \top } .
$$

Substituting and using ${ \widehat { \mathrm { H S I C } } } = n ^ { - 1 } \mathrm { t r } ( { \widetilde { K } } \cdot n ^ { - 1 } { \widetilde { L } } )$

$$
\widehat { \mathrm { H S I C } } ( Z , S ) = \frac { 1 } { n } \sum _ { j = 1 } ^ { r } \widehat { \lambda } _ { j } \cdot \widehat { e } _ { j } ^ { \top } \widetilde { K } \widehat { e } _ { j } .\tag{A.4}
$$

Step 2: Lower bound via the m-th eigenvalue. Since $\widetilde { K } = H K H$ is PSD, every term in (A.4) is nonnegative; dropping the terms $j > m$ and using $\hat { \lambda } _ { j } \ge \hat { \lambda } _ { m }$ for $j \leq m$ ,

$$
\widehat { \mathrm { H S I C } } ( Z , S ) \geq \frac { 1 } { n } \sum _ { j = 1 } ^ { m } \hat { \lambda } _ { j } \hat { e } _ { j } ^ { \top } \widetilde { K } \hat { e } _ { j } \geq \frac { \hat { \lambda } _ { m } } { n } \sum _ { j = 1 } ^ { m } \hat { e } _ { j } ^ { \top } \widetilde { K } \hat { e } _ { j } = \frac { \hat { \lambda } _ { m } } { n } \mathrm { t r } \big ( P _ { m } \widetilde { K } P _ { m } \big ) ,\tag{A.5}
$$

where we used $\begin{array} { r } { \sum _ { j = 1 } ^ { m } \hat { e } _ { j } ^ { \top } \widetilde { K } \hat { e } _ { j } = \mathrm { t r } ( P _ { m } \widetilde { K } P _ { m } ) } \end{array}$ (since $\begin{array} { r } { P _ { m } = \sum _ { j = 1 } ^ { m } \hat { e } _ { j } \hat { e } _ { j } ^ { \top } , P _ { m } ^ { 2 } = P _ { m } } \end{array}$ , and the cyclic-trace identity).

Step 3: Bound the projected prediction deviation by $\mathrm { t r } ( P _ { m } \widetilde { K } P _ { m } )$ . By the reproducing property, $\hat { \delta } _ { f , i } ~ = ~ \langle f , \hat { d } _ { S _ { i } } \rangle _ { \mathcal { F } _ { \mathcal { Z } } }$ , so the vector $\delta _ { f }$ has entries $( \delta _ { f } ) _ { i } \ =$

$\langle f , \hat { d } _ { S _ { i } } \rangle _ { \mathcal { F } _ { \mathcal { Z } } }$ . For any unit vector $u \in \mathbb { R } ^ { n }$ , the linear combination $\textstyle \sum _ { i } u _ { i } { \hat { d } } _ { S _ { i } } \in { \mathcal { F } } _ { { \mathcal { Z } } }$ and $\begin{array} { r } { u ^ { \top } \delta _ { f } = \langle f , \sum _ { i } u _ { i } \hat { d } _ { S _ { i } } \rangle _ { \mathcal { F } _ { \mathcal { Z } } } } \end{array}$ . By Cauchy–Schwarz in $\mathcal { F } _ { \mathcal { Z } }$

$$
( \boldsymbol { u } ^ { \top } \delta _ { f } ) ^ { 2 } \ \leq \ \| \boldsymbol { f } \| _ { \mathcal { F } _ { \boldsymbol { z } } } ^ { 2 } \left\| \sum _ { i } u _ { i } \hat { d } _ { S _ { i } } \right\| _ { \mathcal { F } _ { \boldsymbol { z } } } ^ { 2 } \ = \ \| \boldsymbol { f } \| _ { \mathcal { F } _ { \boldsymbol { z } } } ^ { 2 } \boldsymbol { u } ^ { \top } \widetilde { \boldsymbol { K } } \boldsymbol { u } ,
$$

where the last equality uses $\langle \hat { d } _ { S _ { i } } , \hat { d } _ { S _ { j } } \rangle _ { \mathcal { F } _ { \mathcal { Z } } } = \widetilde { K } _ { i j }$ . Applying this with $u \ : = \ : \hat { e } _ { j }$ for $j = 1 , \ldots , m$ and summing,

$$
\sum _ { j = 1 } ^ { m } ( \hat { e } _ { j } ^ { \top } \delta _ { f } ) ^ { 2 } \leq \| f \| _ { \mathcal { F } _ { z } } ^ { 2 } \sum _ { j = 1 } ^ { m } \hat { e } _ { j } ^ { \top } \widetilde { K } \hat { e } _ { j } \ = \ \| f \| _ { \mathcal { F } _ { z } } ^ { 2 } \operatorname { t r } \bigl ( P _ { m } \widetilde { K } P _ { m } \bigr ) .
$$

The left-hand side equals $\begin{array} { r } { \| P _ { m } \delta _ { f } \| _ { 2 } ^ { 2 } = \sum _ { i = 1 } ^ { n } ( P _ { m } \delta _ { f } ) _ { i } ^ { 2 } } \end{array}$ since $\{ \hat { e } _ { j } \} _ { j = 1 } ^ { m }$ is an orthonormal basis for the range of $P _ { m }$ . Dividing by n,

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( P _ { m } \delta _ { f } ) _ { i } ^ { 2 } \ \leq \ \frac { \| f \| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 } } { n } \operatorname { t r } \bigl ( P _ { m } \widetilde { K } P _ { m } \bigr ) .\tag{A.6}
$$

Step 4: Combine. Combining (A.5) (which gives $\mathrm { t r } ( P _ { m } \widetilde { K } P _ { m } ) \leq n \widehat { \mathrm { H S I C } } / \hat { \lambda } _ { m } )$ with (A.6),

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( P _ { m } \delta _ { f } ) _ { i } ^ { 2 } \ \leq \ \frac { \| f \| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 } } { \hat { \lambda } _ { m } } \widehat { \mathrm { H S I C } } ( Z , S ) ,
$$

which is the second inequality of (4.5) since $\widehat { m } _ { f } ~ = ~ P _ { m } \delta _ { f }$ . The first inequality, $\begin{array} { r } { \widehat { \Delta } _ { \mathrm { G D P } } ( f ) ^ { 2 } = \left( n ^ { - 1 } \sum _ { i } | ( \widehat { m } _ { f } ) _ { i } | \right) ^ { 2 } \leq n ^ { - 1 } \sum _ { i } ( \widehat { m } _ { f } ) _ { i } ^ { 2 } } \end{array}$ , is the Cauchy–Schwarz inequality (equivalently, Jensen applied to $x \mapsto x ^ { 2 } )$ □

The population counterpart of this finite-sample spectral control is the upper bound of Theorem 4.5, which handles the compact kernel operator on $s$ through its spectral tail $\rho _ { m }$ without assuming a spectral gap.

## Appendix B. Additional Notation and Technical Lemmas.

Lemma B.1 (Fubini’s theorem for RKHS-valued integrals). Let k be a bounded kernel, $\begin{array} { r } { \operatorname* { s u p } _ { x \in \mathcal { X } } k ( x , x ) \le \kappa < \infty , } \end{array}$ , with RKHS ${ \mathcal F } .$ . For any random variable X on $x ,$ the mean embedding $\mu _ { X } = \operatorname { \mathbb { E } } [ k ( X , \cdot ) ]$ exists as a Bochner integral in ${ \mathcal F } .$ . Moreover, for any F-valued Bochner-integrable random element $U$ and any $v \in { \mathcal { F } } , \langle \mathbb { E } [ U ] , v \rangle _ { \mathcal { F } } =$ $\mathbb { E } [ \langle U , v \rangle _ { \mathcal { F } } ] ,$ in particular $\langle \mu _ { X } , v \rangle _ { \mathcal { F } } = \mathbb { E } [ v ( X ) ]$

Proof. By the reproducing property,

$$
\| k ( x , \cdot ) \| _ { \mathcal { F } } = \sqrt { k ( x , x ) } \leq \sqrt { \kappa } , \quad \mathrm { s o } \quad \mathbb { E } [ \| k ( X , \cdot ) \| _ { \mathcal { F } } ] \leq \sqrt { \kappa } < \infty ,
$$

guaranteeing existence of the Bochner integral. The exchange of expectation and inner product follows from continuity of the inner product together with Fubini’s theorem.

B.1. Discrete sensitive attribute: coincidence of the two formulations. When S takes finitely many values, the joint and conditional-integral formulations of representation-level fairness reduce to the same aggregated two-sample MMD object, which motivates studying the continuous case in the main text.

Lemma B.2 (Discrete sensitive attribute coincidence). Use the notation: $\mathcal { F } _ { \mathcal { Z } }$ is the RKHS on $\mathcal { Z }$ with characteristic kernel $k _ { \mathcal Z } , ~ \mu _ { Z } : = \mathbb { E } [ k _ { \mathcal Z } ( Z , \cdot ) ] \in \mathcal { F } _ { \mathcal Z }$ is the marginal mean embedding, and $\mu _ { Z | s _ { k } } : = \mathbb { E } [ k _ { \mathcal { Z } } ( Z , \cdot ) \mid S = s _ { k } ] \in \mathcal { F } _ { \mathcal { Z } }$ is the conditional mean embedding. Let $\boldsymbol { S } = \{ s _ { 1 } , \ldots , s _ { K } \}$ be finite with $\pi _ { k } : = \operatorname* { P r } ( S = s _ { k } ) > 0$ , let $k _ { S }$ be the strictly positive-definite sensitive-attribute kernel, write $\delta _ { k } : = \mu _ { Z | s _ { k } } - \mu _ { Z }$ , and let $\mathcal { T } _ { \mathrm { M M D } ^ { 2 } } ( Z ; S ) : = \mathbb { E } _ { S } [ \mathrm { M M D } ^ { 2 } ( P _ { Z | S } , P _ { Z } ) ]$ be the conditional-integral functional with local discrepancy $d = \mathrm { M M D ^ { 2 } }$ . Then

$$
\operatorname { H S I C } ( Z , S ) = \sum _ { k , k ^ { \prime } = 1 } ^ { K } \pi _ { k } \pi _ { k ^ { \prime } } k _ { \mathcal { S } } { \left( s _ { k } , s _ { k ^ { \prime } } \right) } \left. { \delta _ { k } , \delta _ { k ^ { \prime } } } \right. _ { \mathcal { F } _ { \mathcal { Z } } } .
$$

HSI $\mathrm { C } ( Z , S )$ and $\mathcal { T } _ { \mathrm { M M D } ^ { 2 } } ( Z ; S )$ both vanish if and only if $P _ { Z | S = s _ { k } } = P _ { Z }$ for every $k ;$ both reduce to aggregated two-sample MMDs.

Proof. Expand HSIC via $\mu _ { P _ { Z , S } } - \mu _ { P _ { Z } \otimes P _ { S } } = \mathbb { E } _ { S } [ ( \mu _ { Z | S } - \mu _ { Z } ) \otimes k _ { S } ( S , \cdot ) ]$ and integrate the finite measure on $s$ □

## B.2. Bandwidth scaling of the spectral constant.

Remark B.3 (Bandwidth scaling of the full-resolution constant $\hat { \lambda } _ { S } )$ . The fullresolution constant $\hat { \lambda } _ { S } = \lambda _ { \operatorname* { m i n } } ^ { + } ( n ^ { - 1 } \widetilde { L } )$ of Theorem 4.7 at $m = r$ reflects the conditioning of the centered Gram matrix on S. As the sensitive-kernel bandwidth tends to zero and $L \to I _ { n } , { \widetilde { L } } \to H$ (whose positive eigenvalues are all 1), and $\hat { \lambda } _ { S }  1 / n$ . As the bandwidth tends to infinity and $L  \mathbf { 1 1 } ^ { \top } , \widetilde { L }  0$ and $\hat { \lambda } _ { S }  0$ . Very large bandwidths therefore make the conditional-gap bound loose, while extremely small bandwidths give a finite-sample point-mass notion of conditioning; the median heuristic selects an intermediate regime. Standard results on Gaussian-kernel Gram matrices for samples from densities with compact support $( \mathrm { e . g . }$ , Koltchinskii and Gin´e, 2000) confirm a polynomial dependence on σ between these extremes; in our experiments at $n = 1 5 0 0$ this yields $\hat { \lambda } _ { S } \sim 1 0 ^ { - 9 } – 1 0 ^ { - 7 }$ across the bandwidth sweep (Appendix H.3).

## Appendix C. Regularized empirical kHGR.

This appendix records a regularized empirical bound linking $\widehat { \mathrm { H S I C } }$ to a Tikhonovregularized empirical kHGR. The unregularized empirical kHGR requires inverting empirical covariance operators with arbitrarily small positive eigenvalues, so a meaningful finite-sample statement is naturally regularized; see Fukumizu et al. (2007) for the kCCA interpretation. The FRHSIC guarantees in the main text do not depend on this result.

Empirical covariance operators. Define the empirical centered cross-covariance operator $\hat { \Sigma } _ { Z S } : \mathcal { F } _ { S }  \mathcal { F } _ { \mathcal { Z } }$ by

$$
\hat { \Sigma } _ { Z S } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( k _ { Z } ( Z _ { i } , \cdot ) - \hat { \mu } _ { Z } \right) \otimes \big ( k _ { S } ( S _ { i } , \cdot ) - \hat { \mu } _ { S } \big ) ,
$$

where $\hat { \mu } _ { Z } , \hat { \mu } _ { S }$ are the empirical mean embeddings, and define $\hat { \Sigma } _ { Z Z } , \hat { \Sigma } _ { S S }$ analogously. All three operators are finite-rank (rank at most $n )$ , hence Hilbert–Schmidt without further assumptions on the kernels. By a direct expansion, $\widehat { \mathrm { H S I C } } ( Z , S ) = \| \hat { \Sigma } _ { Z S } \| _ { \mathrm { H S } } ^ { 2 }$

Regularized empirical kHGR. For ridge parameters $\eta _ { Z } , \eta _ { S } \ > \ 0$ , define the Tikhonov-regularized empirical kHGR

$$
\begin{array} { r } { \widehat { \mathrm { k H G R } } _ { \eta _ { Z } , \eta _ { S } } ( Z , S ) : = \big \| ( \hat { \Sigma } _ { Z Z } + \eta _ { Z } I ) ^ { - 1 / 2 } \hat { \Sigma } _ { Z S } ( \hat { \Sigma } _ { S S } + \eta _ { S } I ) ^ { - 1 / 2 } \big \| _ { \mathrm { o p } } . } \end{array}
$$

Proposition C.1 (Regularized empirical sandwich bound). Let $k _ { \mathcal { Z } } , k _ { \mathcal { S } }$ be characteristic kernels and let η<sub>Z</sub>, $\eta _ { S } > 0$ . Then

$$
\widehat { \mathrm { k H G R } } _ { \eta _ { Z } , \eta _ { S } } ( Z , S ) ^ { 2 } \ \leq \ \widehat { \frac { \mathrm { H S I C } } { \eta _ { Z } \eta _ { S } } } .\tag{C.1}
$$

Proof. Set $A = ( \hat { \Sigma } _ { Z Z } + \eta _ { Z } I ) ^ { - 1 / 2 } , B = \hat { \Sigma } _ { Z S } , C = ( \hat { \Sigma } _ { S S } + \eta _ { S } I ) ^ { - 1 / 2 }$ . Each operator $\hat { \Sigma } _ { Z Z } , \hat { \Sigma } _ { S S }$ is positive semi-definite, so $\hat { \Sigma } _ { Z Z } + \eta _ { Z } I \succeq \eta _ { Z } I$ and $\hat { \Sigma } _ { S S } + \eta _ { S } I \succeq \eta _ { S } I$ Consequently

$$
\| A \| _ { \mathrm { o p } } \ \leq \ 1 / { \sqrt { \eta z } } , \qquad \| C \| _ { \mathrm { o p } } \ \leq \ 1 / { \sqrt { \eta s } } .
$$

By the operator-norm ≤ Hilbert–Schmidt-norm inequality and the standard submultiplicative bound $\left\| A B C \right\| _ { \mathrm { o p } } \leq \left\| A \right\| _ { \mathrm { o p } } \left\| B \right\| _ { \mathrm { H S } } \left\| C \right\| _ { \mathrm { o p } }$ (valid because the Hilbert–Schmidt norm dominates the operator norm),

$$
\begin{array} { r } { \widehat { \mathrm { K H G R } } _ { \eta _ { Z } , \eta _ { S } } ( Z , S ) = \| A B C \| _ { \mathrm { o p } } \leq \| A \| _ { \mathrm { o p } } \| B \| _ { \mathrm { H S } } \| C \| _ { \mathrm { o p } } \leq \frac { \| \hat { \Sigma } _ { Z S } \| _ { \mathrm { H S } } } { \sqrt { \eta _ { Z } \eta _ { S } } } . } \end{array}
$$

Squaring and using $\widehat { \mathrm { H S I C } } ( Z , S ) = \| \hat { \Sigma } _ { Z S } \| _ { \mathrm { H S } } ^ { 2 }$ gives (C.1).

The proposition makes no claim about the unregularized limit $\eta _ { Z } , \eta _ { S } \downarrow 0 _ { ; }$ for which the spectrum of the empirical covariance operators must be controlled separately. Population-level zero-equivalence of HSIC and kHGR under characteristic kernels is independent of this regularization choice.

## Appendix D. Experimental Details.

## D.1. Datasets.

Adult Income. The UCI Adult dataset contains n ≈ 30,000 records (after removing missing values). Features include numeric variables and one-hot encoded categoricals (d = 107 after encoding). The target is whether income exceeds \$50K/year. The sensitive attribute is age (continuous, range 17–90). All features and the sensitive attribute are min-max scaled to [0, 1].

ACS Income. The ACS Income dataset (Ding et al., 2021) is drawn from the 2018 American Community Survey (California) via the folktables package. We subsample $n \ : = \ : 2 0 , 0 0 0$ individuals. The target is whether income exceeds $\$ 508/9$ . The sensitive attribute is age (continuous, range 17–94). Features include 9 demographic and employment variables.

MEPS. The Medical Expenditure Panel Survey (Panel 19, 2015) (Romano et al., 2020) contains $n \approx 1 3 , 0 0 0$ respondents after removing records with missing values (negative sentinel codes). The target is healthcare utilization $\geq 1 0$ visits (binary classification). The sensitive attribute is age (continuous, range 17–85). Features include 19 variables covering demographics, health conditions, insurance status, and income.

Communities & Crime. The UCI Communities & Crime dataset contains n ≈ 2,000 communities. The target is violent crimes per population (regression). The sensitive attribute is racial composition (continuous). Columns with ${ > } 2 0 \%$ missing values are dropped; remaining missing values are imputed with column medians.

COMPAS. The ProPublica COMPAS dataset contains $n \approx 6 { , } 0 0 0$ defendants after standard filtering (Angwin et al., 2016). The target is two-year recidivism (binary classification). The sensitive attribute is age (continuous). Features include prior counts, charge degree (one-hot), sex, and race (one-hot), yielding d = 16 features.

D.2. Model Architecture. Following Kong et al. (2025), the encoder $h$ is a 2-layer MLP: $d _ { X }  5 0  5 0$ with SELU activations (hidden dimension $H = 5 0$ output representation dimension $Z = 5 0 )$ . The prediction head $f$ is a linear layer: $5 0  1$ (no hidden layers). This architecture matches FREM exactly, ensuring a fair comparison. The adversary in LAFTR and ADV uses the same 2-layer structure $( 5 0 \to 5 0 \to n _ { \mathrm { o u t } } )$ with SELU activation.

D.3. Training Details. All models are trained for 200 epochs with the Adam optimizer (learning rate $1 0 ^ { - 3 }$ , default $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 , \varepsilon = 1 0 ^ { - 8 }$ , weight decay 0), batch size 256, and the standard $\mathrm { P y }$ Torch default initialization. For FRHSIC, kernel bandwidths are set via the median heuristic: $\sigma _ { S }$ is computed once on the training data, and $\sigma _ { \mathcal { Z } }$ is recomputed every 20 epochs on the current mini-batch of encoded representations. For baselines (FREM, Reg-GDP, etc.), we use fixed $\sigma _ { Z } = 1 . 0$ for the Z-kernel after min-max scaling to [0, 1], following Kong et al. (2025); the additional S-smoothing bandwidths used by FREM $( \gamma = 0 . 5 )$ and Reg-GDP (Nadaraya–Watson bandwidth 0.2) are taken from their source publications and described per-method below. We do not modify per-method bandwidth conventions, ensuring each method is evaluated as published.

Train/validation/test splits. For all real datasets we use an $8 0 \% / 2 0 \%$ random train/test split. Hyperparameter selection (i.e., choosing λ along the Pareto frontier) is performed using the same training split as the model fit; for each method and each dataset we sweep λ over the full grid and report results across the resulting Pareto frontier (no separate validation split is used to pick a single $\lambda ,$ since the goal of the comparison is to characterize the fairness–accuracy tradeof curve, not to choose one operating point). We repeat 5 times with diferent random splits and report mean ± standard deviation on the held-out test set.

Regularization-strength $( \lambda )$ grids. For all methods on all real datasets, we sweep

$$
\lambda \in \{ 0 . 1 , 1 . 0 , 1 0 . 0 , 1 0 0 . 0 , 5 0 0 . 0 \} .
$$

For the Equal Opportunity and multi-sensitive-attribute experiments (Appendix E) the grid is $\lambda \in \{ 0 . 1 , 1 . 0 , 1 0 . 0 , 5 0 . 0 \}$

## D.4. Baseline Implementations and Hyperparameter Grids.

Reg-GDP. We implement the GDP regularizer from Jiang et al. (2022) using kernel-smoothed conditional expectations on a grid of 30 evenly-spaced points over the range of S. Reg-GDP uses two distinct bandwidths: a Gaussian Z-kernel bandwidth $\sigma _ { Z } = 1 . 0$ (matching the FREM convention of Kong et al. (2025) on min-max-scaled features) and a Nadaraya–Watson smoothing bandwidth 0.2 on S for the conditional expectation; both follow the values reported by Jiang et al. (2022).

FREM. We implement the weighted EIPM estimator from Kong et al. (2025) using the MMD discriminator. Kernel-smoothed weights are computed with a Gaussian kernel on S with bandwidth $\gamma = 0 . 5$ (following Kong et al. (2025)). For computational tractability we subsample $n _ { \mathrm { a n c h o r } } = 3 2$ anchor points per batch for the EIPM computation; see Appendix D.5 below for details.

LAFTR. We bin the continuous sensitive attribute into 4 quartiles and apply the adversarial FRL method of Madras et al. (2018). The adversary is trained with alternating gradient steps (one adversary step per encoder step). The adversary is a 2-layer MLP $( 5 0  5 0  4 )$ with SELU activations.

ADV. The continuous-target adversary of Grari et al. (2022), with the same $2 \AA$ layer architecture as LAFTR but with a single scalar output (predicting S); trained with one adversary step per encoder step.

MMD-binned. Bins continuous $S$ into $n _ { \mathrm { b i n s } } = 1 0$ equal-width bins and applies pairwise MMD across bins; bin centers serve as the smoothing grid.

dCor. We use the unbiased empirical distance correlation of Sz´ekely et al. (2007) as the fairness penalty; no further hyperparameters beyond λ.

D.5. FREM 32-anchor subsampling. The FREM EIPM objective of Kong et al. (2025) computes, at each mini-batch, an integral over s of a weighted MMD term:

$$
\widehat { \mathrm { E I P M } } _ { n } ( Z ; S ) = \int _ { S } \widehat { \mathrm { M M D } } ^ { 2 } \big ( \widehat { P } _ { Z | S = s } , \widehat { P } _ { Z } \big ) d \widehat { P } _ { S } ( s ) .
$$

Estimating this expectation by Monte Carlo with all n training points as anchors would require $O ( n ^ { 3 } )$ work per batch (an $n \times n$ kernel matrix evaluated at each of n anchors). For computational tractability we instead draw $n _ { \mathrm { a n c h o r } } = 3 2$ uniformlyrandom anchor points per mini-batch and average the MMD terms over those anchors. With batch size 256 this is a $3 2 / 2 5 6 = 1 2 . 5 \%$ Monte Carlo subsample and reduces the per-batch cost to $O ( n _ { \mathrm { a n c h o r } } \cdot n ^ { 2 } )$ . The number 32 was chosen to match the FREM implementation in our codebase (see experiments/real data.py, function train frem); we report results with $n _ { \mathrm { a n c h o r } } = 3 2$ throughout.

FREM anchor-subsampling sensitivity (limitation). The 32-anchor subsampling we use for FREM is a computational approximation; an anchor-count sweep is the natural next step to characterize the sensitivity of FREM’s reported Pareto frontier to the anchor budget, and we plan to include such a sweep in a follow-up version of this work. The per-epoch wall-clock cost grows linearly with the anchor count, so the principal cost of the sweep is compute rather than implementation.

Baseline bandwidth sensitivity (limitation). The bandwidths reported in Appendix D (FRHSIC median heuristic, FREM $\gamma = 0 . 5$ and $\sigma _ { Z } = 1 . 0$ , Reg-GDP Nadaraya– Watson bandwidth 0.2 and $\sigma _ { Z } = 1 . 0 ) $ are the values reported in the source publications. We retain published bandwidths in this paper to avoid favoring any particular method by retuning, but a full bandwidth-sensitivity sweep for the FREM and Reg-GDP baselines is the natural next step, and we plan to include such a sweep in a follow-up version of this work.

Paired significance test (limitation). We report mean and standard deviation across $n = 5$ random splits in Table 1 but do not run a formal paired test $( \mathrm { e . g . } ,$ Wilcoxon signed-rank), since 5 paired observations is too few for reliable inference at conventional significance levels. Increasing the number of repeats to enable a credible paired comparison is the natural next step and we plan to include such an analysis in a follow-up version of this work.

D.6. Reproducibility checklist. We summarize the elements needed to reproduce the experiments end to end.

• Random seed. A single base seed SEED = 42 is set at the top of each experiment script for numpy, torch, and the train/test split RNG. The 5 repeats use seeds SEED, $\mathtt { S E E D } + 1 , \dotsc , \mathtt { S E E D } + 4$ . See experiments/real\_data.py:27 and experiments/utils.py.

• Train/validation/test split. 80%/20% random train/test split; no separate validation split (rationale above). Repeated 5 times with diferent splits.

• Preprocessing. Min–max scaling for the features and the sensitive attribute is fit on the training split only and then applied to the held-out test split (experiments/real\_data.py, run\_single\_dataset). All cells of Table 1 in the main text are reported under this leakage-free scaling protocol; the resulting numbers agree with our earlier leakage-afected numbers to within standard deviations on essentially every cell, with the most visible shifts in the MMD and ADV rows of Crime (where the small sample size n ≈ 2000 amplifies any scaling diference). Categoricals are one-hot encoded; rows with missing values are dropped except in Communities & Crime, where columns with >20% missing are dropped and remaining missing values are imputed with column medians.

• Model architecture. Encoder h: 2-layer MLP $d _ { X }  5 0  5 0$ with SELU activations. Predictor $f \colon$ linear $5 0  1$ . Adversaries (LAFTR / ADV / MMD-binned): 2-layer $\mathrm { M L P ~ 5 0 } \to 5 0 \to n _ { \mathrm { o u t } }$ with SELU.

• Optimizer. Adam, learning rate $1 0 ^ { - 3 } , \ \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 , \varepsilon = 1 0 ^ { - 8 }$ weight decay 0. PyTorch defaults.

• Batch size and epochs. Batch size 256, 200 epochs.

• Kernel bandwidths. FRHSIC: median heuristic for $\sigma _ { \mathcal { S } }$ (fixed at start of training, on training data) and for $\sigma _ { \mathcal { Z } }$ (recomputed every 20 epochs on the current mini-batch). Baselines: fixed $\sigma _ { Z } = 1 . 0$ on min-max-scaled $Z ,$ FREM $\gamma = 0 . 5$ , Reg-GDP NW bandwidth 0.2.

• λ grids. Main experiments: $\lambda \in \{ 0 . 1 , 1 . 0 , 1 0 . 0 , 1 0 0 . 0 , 5 0 0 . 0 \}$ . EO and multisensitive: $\lambda \in \{ 0 . 1 , 1 . 0 , 1 0 . 0 , 5 0 . 0 \}$

• Baseline hyperparameter grids. See Appendix D, “Baseline Implementations”; all baselines use the same encoder, predictor, optimizer, batch size, and number of epochs as FRHSIC, varying only the per-method fairnesspenalty hyperparameters listed above. The adversary step ratio (LAFTR, ADV) is fixed at 1 : 1.

• FREM anchor count. $n _ { \mathrm { a n c h o r } } = 3 2$ uniformly-sampled anchors per mini batch; see Appendix D.5.

• Hardware. All experiments were run on a single workstation with one NVIDIA RTX-class GPU; the experiment code falls back to CPU if no CUDA device is available (see the device selector in experiments/utils.py). Runtime numbers in Section 6 are wall-clock per-epoch on this single device, averaged over 5 epochs after a warm-up epoch. Training the full λ-sweep across all methods and datasets fits within ∼24 GPU-hours.

• Software. Python 3.11, PyTorch 2.x, NumPy 1.26, scikit-learn 1.3, folktables (for ACS Income). See experiments/README.md for the exact versions used at submission time.

• Code availability. All scripts to reproduce the figures and tables are in the experiments/ directory of the public code repository (https://github.com/Yijin911/FRHSIC); each main-paper figure/table is generated by a single named script (e.g. run\_single\_method.py, regen\_pareto\_mi.py, runtime.py, convergence.py).

D.7. Pareto summary: lowest GDP within 1% of the unfair-baseline performance. For a single-number summary of the fairness–accuracy tradeof that complements the matched-performance comparison in Section $6 ,$ Table 3 reports, for each (method, dataset) pair, the operating point on the λ-sweep that attains the lowest GDP subject to the constraint that the prediction performance is within 1% of the unfair baseline $( \mathrm { A c c } \geq 0 . 9 9 \cdot \mathrm { A c c } _ { \mathrm { U n f a i r } }$ for classification; $\mathrm { M S E } \leq 1 . 0 1 { \cdot } \mathrm { M S E } _ { \mathrm { U n f a i r } }$ for regression). Entries are read of the Pareto sweep underlying Figure $3 . \quad ^ { 6 6 } - ^ { 5 9 }$ denotes that no λ in our grid satisfies the constraint for that method on that dataset. Values are means across 5 random splits.

Table 3  
Pareto summary: (Acc $\ ⁄ M S E , G D P )$ at the lowest-GDP operating point with prediction performance within 1% of the unfair baseline. Adult/ACS/MEPS/COMPAS report Acc; Crime reports MSE $\times 1 0 ^ { 2 } . \quad \stackrel { \scriptscriptstyle { a } } { \scriptscriptstyle \mathscr { a } } _ { \mathscr { T } } \stackrel { \scriptscriptstyle { m } } { \scriptscriptstyle \mathscr { n } }$ indicates no λ in the sweep satisfies the constraint. Read alongside Figure 3, since a tight 1% band can exclude operating points that are visually close on the frontier.
<table><tr><td>Method</td><td>Adult</td><td>ACS Income</td><td>MEPS</td><td>Crime</td><td>COMPAS</td></tr><tr><td>Unfair (ref.)</td><td>(.849, .749)</td><td>(.790, .514)</td><td>(.804, .626)</td><td>(1.91, .029)</td><td>(.654, .123)</td></tr><tr><td>FRHSIC (Ours)</td><td>(.845, .232)</td><td>(.792, .399)</td><td>(.803, .517)</td><td>(1.91, .029)</td><td>(.655, .107)</td></tr><tr><td>FREM</td><td>(.845, .167)</td><td>(.787, .274)</td><td>(.800, .317)</td><td>(1.87, .023)</td><td>(.648, .101)</td></tr><tr><td>Reg-GDP</td><td>(.842, .082)</td><td>(.787, .292)</td><td>(.801, .370)</td><td>(1.91, .025)</td><td>(.650, .079)</td></tr><tr><td>ADV</td><td>(.841, .466)</td><td>(.785, .458)</td><td>(.803, .480)</td><td>(1.92, .029)</td><td>(.649, .095)</td></tr><tr><td>MMD (binned)</td><td>(.847, .126)</td><td>(.783, .157)</td><td>(.800, .407)</td><td></td><td>(.651, .082)</td></tr><tr><td>LAFTR (binned)</td><td>(.842, .385)</td><td>(.787, .474)</td><td>(.803, .530)</td><td>(1.89, .029)</td><td>(.655, .122)</td></tr><tr><td>dCor</td><td>(.846, .194)</td><td>(.783, .203)</td><td>(.800, .487)</td><td>(1.92, .023)</td><td>(.653, .091)</td></tr></table>

Caveats. The 1% accuracy band is intentionally tight, and on some datasets (notably COMPAS, where multiple methods cluster near the unfair-baseline accuracy) the band excludes operating points that are visually close on the Pareto frontier (Figure 3). Reg-GDP can drive GDP to near zero on Adult and ACS, but only at accuracy losses well outside the 1% band; those points are visible on Figure 3 but do not appear in the table above. The single-number summary should therefore be read alongside the full Pareto curves rather than as a standalone ranking.

## Appendix E. Additional Experimental Results.

E.1. Synthetic validation of the conditional-gap bound. On a controlled synthetic classification setup, enforcing HSIC drives GDP monotonically to zero at stable accuracy, confirming the conditional-gap bound of Theorem 4.7. We generate $n = 5 0 0 0$ samples with $S \sim$ Uniform(0, 1), $\boldsymbol { X } = ( S + \epsilon _ { 1 } , \epsilon _ { 2 } )$ where $\epsilon _ { 1 } \sim \mathcal { N } ( 0 , 0 . 0 9 )$ $\epsilon _ { 2 } \sim \mathcal { N } ( 0 , 1 )$ , and $Y \sim \mathrm { B e r n o u l l i } ( 1 / ( 1 + e ^ { - X _ { 1 } } ) )$ . The encoder h is a 2-layer MLP (hidden dimension 64, representation dimension 8) and the prediction head f is linear; a smaller architecture than the real-data experiments isolates the efect of the HSIC penalty in a controlled setting. We train with objective $\mathcal { L } _ { \mathrm { C E } } ( f \circ h ) + \lambda \cdot \widehat { \mathrm { H S I C } } ( Z , S )$ for $\lambda \in \{ 0 , 0 . 1 , 0 . 5 , 1 , 5 , 1 0 , 5 0 \}$ . As λ increases, GDP decreases monotonically toward zero (Figure 6a), GDP and HSIC decrease together in a monotone relationship consistent with the upper bound in Theorem 4.7 (Figure 6c), and accuracy remains stable (around 0.62), indicating that the sensitive attribute contributes primarily to unfair discrimination rather than useful predictive signal in this setting.

E.2. Equal Opportunity and multiple sensitive attributes. The HSIC penalty extends to Equal Opportunity (Madras et al., 2018), which requires fairness conditional on the positive outcome $Y = 1$ , by restricting the estimator to the $n _ { 1 }$ samples with $Y _ { i } = 1$

$$
\widehat { \mathrm { H S I C } } _ { \mathrm { E O } } ( Z , S ) = \frac { 1 } { n _ { 1 } ^ { 2 } } \mathrm { t r } ( K _ { 1 } H _ { 1 } L _ { 1 } H _ { 1 } ) ,
$$

where $K _ { 1 } , L _ { 1 } , H _ { 1 }$ are the kernel and centering matrices restricted to that subset. It extends to multiple continuous sensitive attributes by taking a product kernel on $S _ { 1 } \times S _ { 2 }$ (Appendix E.3).

We evaluate the EO extension on Adult and COMPAS, comparing FRHSIC-DP (HSIC on all samples) to FRHSIC-EO (HSIC on $Y = 1$ subset only). Table 4 reports accuracy and EO-GDP (GDP restricted to the Y = 1 subset) across λ values. On COMPAS, the DP variant achieves comparable or better EO-GDP than the EO variant at similar accuracy across all λ values. On Adult, the EO variant achieves slightly lower EO-GDP at high λ (0.352 vs. 0.418 at $\lambda = 5 0 )$ , but the DP variant simultaneously reduces full GDP from 0.889 to $0 . 2 4 0 { \mathrm { - } } \mathrm { a }$ broader fairness guarantee. Overall, enforcing full demographic parity via HSIC provides a strong baseline for Equal Opportunity, with the dedicated EO variant ofering marginal gains on larger datasets.

![](images/ab25d699ca74d5152c8296461494f3373e30195a365d33b7cab92a4feff72679.jpg)

![](images/da332d01ceeddf34e1e6b7de2f45498a657252fdbc80552a3bc07293638760c1.jpg)

![](images/fe8a6dfb3bcc6e8782abc94a4426b6dddd5fbdbce78470d097e97bcc8c38fa80.jpg)  
Fig. 6. Synthetic validation. (a) GDP decreases as λ increases. (b) Accuracy remains stable across regularization strengths. (c) GDP vs. HSIC shows a monotone relationship, consistent with the conditional-gap-control interpretation of Theorem 4.7 (the plot displays monotonic dependence, not the LHS/RHS of the bound).

Table 4  
Equal Opportunity results: FRHSIC-DP vs. FRHSIC-EO. EO-GDP is GDP restricted to $Y = 1$ samples. FRHSIC-DP achieves comparable EO fairness while also enforcing DP.
<table><tr><td>Dataset</td><td>λ</td><td>Method</td><td>Acc</td><td>EO-GDP (4)</td></tr><tr><td rowspan="7">Adult</td><td>0.1</td><td>DP</td><td>0.851</td><td>0.453</td></tr><tr><td></td><td>EO</td><td>0.849</td><td>0.406</td></tr><tr><td>1.0</td><td>DP EO</td><td>0.846 0.850</td><td>0.403 0.416</td></tr><tr><td>10.0</td><td>DP EO</td><td>0.846</td><td>0.437</td></tr><tr><td>50.0</td><td>DP</td><td>0.850 0.840</td><td>0.392 0.418</td></tr><tr><td></td><td>EO</td><td>0.849</td><td>0.352</td></tr><tr><td>0.1</td><td>DP</td><td>0.659</td><td>0.067</td></tr><tr><td rowspan="6">COMPAS</td><td></td><td>EO</td><td>0.662</td><td>0.070</td></tr><tr><td>1.0</td><td>DP</td><td>0.662</td><td>0.070</td></tr><tr><td></td><td>EO</td><td>0.662</td><td>0.075</td></tr><tr><td></td><td>DP</td><td>0.657</td><td>0.064</td></tr><tr><td>10.0</td><td>EO</td><td>0.655</td><td>0.065</td></tr><tr><td>50.0</td><td>DP EO</td><td>0.599 0.659</td><td>0.033 0.076</td></tr></table>

E.3. Multiple Sensitive Attributes. We test FRHSIC with two continuous sensitive attributes (age and hours-per-week) on Adult. Table 5 compares two approaches: (1) a product kernel on joint $( S _ { 1 } , S _ { 2 } )$ (FRHSIC-Joint), and (2) a sum of per-attribute HSIC terms (FRHSIC-Sum). Both approaches reduce GDP with respect to each attribute as λ increases while preserving accuracy, confirming that HSIC naturally extends to the multi-attribute setting without modification. The Joint approach achieves slightly lower GDP at $\lambda = 5 0 \ \mathrm { ( A g e \colon 0 . 3 0 6 \ v s . \ 0 . 6 7 8 ; }$ Hours: 0.517 vs. 0.759), likely because the product kernel captures cross-attribute structure. Figure 7 visualizes the Pareto frontiers.

Table 5  
Multiple sensitive attributes on Adult (age + hours-per-week). GDP is reported separately for each attribute. Both product-kernel (Joint) and sum-of-HSIC (Sum) approaches reduce GDP for both attributes as λ increases.
<table><tr><td>Method (λ)</td><td> $\operatorname { A c c }$  GDP(Age) (↓)</td><td>GDP(Hours) (↓)</td></tr><tr><td>Unfair</td><td>0.843 0.837</td><td>1.142</td></tr><tr><td>FRHSIC-Joint (λ = 1)</td><td>0.844 0.779</td><td>1.029</td></tr><tr><td>FRHSIC-Joint (λ = 10)</td><td>0.846 0.553</td><td>0.857</td></tr><tr><td>FRHSIC-Joint (λ = 50)</td><td>0.851 0.306</td><td>0.517</td></tr><tr><td>FRHSIC-Sum (λ = 1)</td><td>0.847 0.753</td><td>1.055</td></tr><tr><td>FRHSIC-Sum (λ = 10)</td><td>0.848 0.485</td><td>0.826</td></tr><tr><td>FRHSIC-Sum (λ = 50) 0.849</td><td>0.678</td><td>0.759</td></tr></table>

![](images/9e21acd311b133e89d36a5b66642dc6f4833d84105633413b4621fd97e9e8f2d.jpg)

![](images/541a173a458ac02934c72ec71c7911b45d08604dc7d61d6f93ddd9cd146f5132.jpg)  
Fig. 7. Multiple sensitive attributes: fairness–accuracy Pareto frontiers for FRHSIC-Joint (product kernel) and FRHSIC-Sum (sum of per-attribute HSIC), evaluated on Adult with age and hours-per-week as sensitive attributes.

Appendix F. Structural Causal Models.

F.1. Structural Causal Models and Counterfactual Fairness. Following Pearl (2009) and Kusner et al. (2017), a Structural Causal Model (SCM) over (X, S, Y ) specifies structural equations

$$
S = U _ { S } , \qquad X = f _ { X } ( S , U _ { X } ) , \qquad Y = f _ { Y } ( X , S , U _ { Y } ) ,
$$

with jointly independent exogenous noise $U _ { X } , U _ { S } , U _ { Y }$ . The counterfactual prediction $\widehat { Y } _ { S  s ^ { \prime } }$ is the value of $\widehat { Y }$ obtained by replacing $S = U _ { S }$ with $S = s ^ { \prime }$ in the structural equations and propagating downstream. A predictor $\widehat { Y }$ is counterfactually fair

(Kusner et al., 2017) if, for every observable $( X = x , S = s )$ and every alternative $s ^ { \prime } { \mathrm { . } }$

$$
\widehat { Y } _ { S  s ^ { \prime } } = \widehat { Y } _ { S  s } \mathrm { a . s . \ g i v e n \ } X = x , S = s .
$$

## Appendix G. Additional Theoretical Augmentations.

This section collects two augmentations referenced from the main text: a necessity-for-counterfactual-fairness result and a uniform-concentration bound that bridges training-sample HSIC to its population counterpart.

G.1. Necessity for Counterfactual Fairness. The joint-distribution route to independence also clarifies the relationship between HSIC-fairness and the causalfairness literature.

Proposition G.1 (HSIC-fairness as a necessary condition for counterfactual fairness). Assume the SCM in Appendix F.1. Suppose the encoder h is a deterministic function so that $Z = h ( X ) = h ( f _ { X } ( S , U _ { X } ) )$ . If the prediction ${ \widehat { Y } } = f ( Z )$ is counterfactually fair (Appendix $F . 1 ) \ f ($ r every f in a class F rich enough to identify equality of distributions in $\mathcal { Z } \ ( e . g . , \mathcal { F }$ is the unit ball of an RKHS with characteristic kernel $k _ { \mathcal { Z } } )$ , then $\mathrm { H S I C } ( Z , S ) = 0$

Proof. Counterfactual fairness for every $f \in { \mathcal { F } }$ implies, in particular, that for all $s ^ { \prime } \in \mathcal { S }$ in the support of S and all $f \in { \mathcal { F } }$

$$
\mathbb { E } [ f ( Z _ { S  s ^ { \prime } } ) ] = \mathbb { E } [ f ( Z ) ] ,
$$

where $Z _ { S  s ^ { \prime } } = h ( f _ { X } ( s ^ { \prime } , U _ { X } ) )$ . Since $S = U _ { S }$ and $U _ { X }$ are independent under the SCM, and the marginal of $Z _ { S  s ^ { \prime } }$ over $U _ { X }$ has the same distribution as the marginal of Z given $S = s ^ { \prime }$ (because intervening with $S = s ^ { \prime }$ matches the conditional given $S = s ^ { \prime }$ when S is exogenous (Pearl, 2009, Section 3.2.2)), we obtain

$$
\begin{array} { r } { \mathbb { E } [ f ( Z ) \mid S = s ^ { \prime } ] = \mathbb { E } [ f ( Z ) ] \quad \mathrm { f o r } \ P _ { S ^ { - } } \mathrm { a . e . } \ s ^ { \prime } . } \end{array}
$$

By Proposition 3.1, this is condition (4), equivalent to $Z \perp S$ and hence, for the characteristic product kernel, to $\mathrm { H S I C } ( Z , S ) = 0$

In words, if a representation can support counterfactually fair predictions for every (suficiently rich) prediction head, the representation must already satisfy the observational-independence condition that HSIC enforces. The converse does not hold: $\mathrm { H S I C } ( Z , S ) = 0$ ensures that the marginal distribution of $f ( Z )$ does not depend on S, but counterfactual fairness is a strictly pointwise statement about individuallevel interventions, which can fail when the structural dependence on S is observationally invisible $( \mathrm { e . g . } , Z = S \oplus U _ { X }$ with binary $S , U _ { X }$ uniform yields $Z \perp S$ marginally but $Z _ { S  s ^ { \prime } } \neq Z _ { S  s }$ pointwise). HSIC-fairness should therefore be understood as the strongest observational fairness consequence of counterfactual fairness, not as a substitute for it.

G.2. Uniform Concentration over the Encoder Class. The empirical Theorem 4.7 controls the empirical conditional-mean gap $\begin{array} { r } { \frac { 1 } { n } \sum _ { i } ( P _ { m } \delta _ { f } ) _ { i } ^ { 2 } } \end{array}$ through ${ \widehat { \mathrm { H S I C } } } ( h ( X ) , S )$ for a single encoder h. In practice, h is learned from the same data, so the bound must hold uniformly over a hypothesis class H of encoders. We invoke Corollary 23 of Ni and Huo (2024) (an HSIC specialization of their Theorem 12), which we adapt to the one-sided encoder class arising in FRHSIC. We do not reprove that result; this appendix only maps FRHSIC onto it.

Mapping FRHSIC to the prior uniform concentration result. First, HSIC(Z, S) is the squared MMD between $P _ { Z , S }$ and $P _ { Z } \otimes P _ { S }$ under the fixed product kernel $k { \big ( } ( z , s ) , ( z ^ { \prime } , s ^ { \prime } ) { \big ) } = k _ { \mathcal { Z } } ( z , z ^ { \prime } ) k _ { \mathcal { S } } ( s , s ^ { \prime } )$ , and HSIC<sup>\</sup> $\mathsf { \Omega } _ { n } ( h ( X ) , S ) = n ^ { - 2 } \operatorname { t r } ( K _ { h } H L H )$ is exactly the empirical kernel-based two-sample statistic to which Theorem 12 of Ni and Huo (2024) applies, with $( Z ^ { \prime } , S ^ { \prime } ) \sim P _ { Z } \otimes P _ { S }$ . Second, the encoder class H acts only on the Z-coordinate, inducing the transformed paired samples $\{ ( h ( X _ { i } ) , S _ { i } ) \}$ and the encoder-indexed product-kernel class $\kappa _ { \mathcal { H } }$ . Third, the boundedness and Lipschitz hypotheses needed by Ni and Huo (2024) (Assumption 20) are the bounded-kernel and Lipschitz-feature-map conditions stated in Proposition G.2, which $k _ { \mathcal { Z } } , k _ { \mathcal { S } }$ , and $h \in \mathcal H$ satisfy under the standing assumptions. Fourth, bounded linear maps over a compact parameter set, and fixed-architecture MLPs with bounded weights, bounded inputs, and Lipschitz activations, have empirical Gaussian (equivalently, up to logarithmic factors, Rademacher) complexity of order $n ^ { - 1 / 2 }$ , so the complexity term in the imported bound is controlled; growing-width or unregularized networks need not satisfy this. Data-dependent kernel selection, which would enlarge $\kappa _ { \mathcal { H } }$ to a composite-kernel class, is outside the scope of this statement.

Proposition G.2 (Uniform concentration of HSIC over an encoder class).<sup>\</sup> Let H be a class of measurable encoders $h : \mathcal { X }  \mathcal { Z }$ . Suppose the kernels $k _ { \mathcal { Z } } , k _ { \mathcal { S } }$ satisfy Assumption 20 of Ni and Huo (2024): they are bounded with sup<sub>z</sub> $k _ { \mathcal { Z } } ( z , z ) \leq \nu _ { \mathcal { Z } }$ and sup<sub>s</sub> $k _ { S } ( s , s ) \leq \nu _ { S }$ , and the feature maps $z \mapsto k _ { \mathcal { Z } } ( z , \cdot )$ and $s \mapsto k _ { S } ( s , \cdot )$ are Lipschitz with constants $\ell _ { \mathcal { Z } } , \ell _ { S } > 0$ , respectively (the constants $\nu _ { \mathcal { Z } } , \nu _ { S } , \ell _ { \mathcal { Z } } , \ell _ { S }$ are exactly those of Theorem 5.1). Let

$$
\hat { \mathcal { G } } _ { n } ( \mathcal { H } ) : = \mathbb { E } _ { \boldsymbol \xi } [ \operatorname* { s u p } _ { h \in \mathcal { H } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \langle \xi _ { i } , h ( X _ { i } ) \rangle | \{ X _ { i } \} _ { i = 1 } ^ { n } ]
$$

denote the empirical Gaussian complexity of H (Ni and Huo, 2024, Equation $\it 3 . 5 \AA$ 2 where $\xi _ { 1 } , \ldots , \xi _ { n }$ are i.i.d. $\mathcal { N } ( 0 , I _ { \mathrm { d i m } ( \mathcal { Z } ) } )$ random vectors; write $\mathcal { G } _ { n } ( \mathcal { H } ) : = \mathbb { E } [ \hat { \mathcal { G } } _ { n } ( \mathcal { H } ) ]$ for its expectation, as in the main text. Then for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$

$$
\begin{array} { r l } & { \underset { h \in \mathcal { H } } { \operatorname* { s u p } } \left| \widehat { \mathrm { H S I C } } ( h ( X ) , S ) - \mathrm { H S I C } ( h ( X ) , S ) \right| \leq 8 \sqrt { 2 } \nu _ { \mathbb { Z } } \nu _ { S } \sqrt { \frac { \log ( 2 / \delta ) } { n } } + \frac { 4 \nu _ { \mathbb { Z } } \nu _ { S } } { n } } \\ & { \qquad + 4 8 \sqrt { \pi } \operatorname* { m a x } \{ \nu _ { \mathbb { Z } } \ell _ { S } , \nu _ { S } \ell _ { Z } \} \mathcal { G } _ { n } ( \mathcal { H } ) . } \end{array}\tag{G.1}
$$

Proof. Apply Corollary 23 of Ni and Huo (2024) to the two-sample statistic $\gamma _ { k } ^ { 2 }$ on the product RKHS with reproducing kernel $k \big ( ( z , s ) , ( z ^ { \prime } , s ^ { \prime } ) \big ) = k _ { \mathcal { Z } } ( z , z ^ { \prime } ) k _ { \mathcal { S } } ( s , s ^ { \prime } )$ identified with HSIC via $\mathrm { H S I C } ( Z , S ) = \gamma _ { k } ^ { 2 } ( ( Z , S ) , ( Z ^ { \prime } , S ^ { \prime } ) )$ for $( Z ^ { \prime } , S ^ { \prime } ) \sim P _ { Z } \otimes P _ { S }$ (Ni and Huo, 2024, Equation 3). The bivariate function class $\mathcal { H } \times \{ \mathrm { i d } _ { \mathcal { S } } \}$ , where id<sub>S</sub> is the identity on S, satisfies $\vec { \mathbb { E } } [ \mathcal { G } ( ( \mathcal { H } \times \{ \mathrm { i d } _ { \mathcal { S } } \} ) ( \mathbf { X } , \mathbf { S } ) ) ] \le \vec { \mathbb { E } } [ \hat { \mathcal { G } } _ { n } ( \mathcal { H } ) ]$ ] by Lemma 19 of Ni and Huo (2024), since the second-coordinate class is a singleton: under the noabsolute-value convention adopted by Ni and Huo (2024, Section 3.1), $\mathcal { G } ( \{ \mathrm { i d } _ { \boldsymbol { S } } \} ( \mathbf { S } ) ) =$ $\begin{array} { r } { \mathbb { E } _ { \xi } \big [ { \frac { 1 } { n } } \sum _ { i } \langle \xi _ { i } , S _ { i } \rangle \big ] = 0 } \end{array}$ by symmetry of the Gaussian distribution. Substituting into the HSIC bound of Corollary 23 yields (G.1). □

Corollary G.3 (Train-population HSIC bridge). Under the assumptions of Proposition G.2, for any encoder $\hat { h } \in \mathcal { H }$ obtained from n i.i.d. training samples and any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$ ，

$$
\begin{array} { r } { \mathrm { H S I C } ( \hat { h } ( X ) , S ) \leq \widehat { \mathrm { H S I C } } ( \hat { h } ( X ) , S ) + \varepsilon _ { n } ( \delta ) , } \end{array}\tag{G.2}
$$

where

$$
\varepsilon _ { n } ( \delta ) : = 8 \sqrt { 2 } \nu _ { \Xi } \nu _ { \mathcal { S } } \sqrt { \frac { \log ( 2 / \delta ) } { n } } + \frac { 4 \nu _ { \Xi } \nu _ { \mathcal { S } } } { n } + 4 8 \sqrt { \pi } \operatorname* { m a x } \{ \nu _ { \Xi } \ell _ { \mathcal { S } } , \nu _ { \mathcal { S } } \ell _ { \Xi } \} \mathcal { G } _ { n } ( \mathcal { H } )
$$

is the uniform-concentration penalty from Proposition G.2.

Proof. By Proposition G.2, with probability at least $1 - \delta .$

$$
\operatorname* { s u p } _ { h \in \mathcal { H } } \left| \widehat { \mathrm { H S I C } } ( h ( X ) , S ) - \mathrm { H S I C } ( h ( X ) , S ) \right| \leq \varepsilon _ { n } ( \delta ) .
$$

Specializing to the data-dependent $\hat { h } \in \mathcal { H }$ , $\mathrm { H S I C } ( \hat { h } ( X ) , S ) - \widehat { \mathrm { H S I C } } ( \hat { h } ( X ) , S ) \leq \varepsilon _ { n } ( \delta ) .$ which rearranges to (G.2). □

Remark G.4 (Train-test fairness bridge). Corollary G.3 delivers a train-test fairness guarantee when combined with Theorem 4.7 applied on an independent test sample. Theorem 4.7 controls the empirical conditional-mean gap $\begin{array} { r } { \frac { 1 } { n } \sum _ { i } ( P _ { m } \delta _ { f } ) _ { i } ^ { 2 } } \end{array}$ which by Jensen’s inequality dominates the squared empirical demographic-parity gap $\widehat { \Delta } _ { \mathrm { G D P } } ( f ) ^ { 2 }$ for any prediction head $f \in \mathcal { F } _ { \mathcal Z }$ . Concretely, given a test sample $\{ ( \widetilde { Z } _ { i } , \widetilde { S } _ { i } ) \} _ { i = 1 } ^ { n }$ independent of the training sample, Theorem 4.7 gives the deterministic bound

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } \big ( P _ { m } ^ { \mathrm { t e s t } } \delta _ { f } ^ { \mathrm { t e s t } } \big ) _ { i } ^ { 2 } \ \le \ \frac { \| f \| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 } } { \widetilde { \lambda } _ { m } } \widehat { \mathrm { H S I C } } ^ { \mathrm { t e s t } } ( \hat { h } ( X ) , S ) ,
$$

where $\widetilde { \mathrm { H S I C } } ^ { \mathrm { t e s t } } , \widetilde { \lambda } _ { m } .$ , and the projection $P _ { m } ^ { \mathrm { t e s t } }$ are computed on the test sample. $\mathrm { A p - }$ plying Proposition G.2 once on training and once pointwise on test (the latter being the standard single-encoder $O ( n ^ { - 1 / 2 } )$ deviation, which is dominated by $\textstyle { \varepsilon _ { n } } )$

$$
\widehat { \mathrm { H S I C } } ^ { \mathrm { t e s t } } ( \hat { h } ( X ) , S ) \leq \mathrm { H S I C } ( \hat { h } ( X ) , S ) + \varepsilon _ { n } ( \delta ) \leq \widehat { \mathrm { H S I C } } ^ { \mathrm { t r a i n } } ( \hat { h } ( X ) , S ) + 2 \varepsilon _ { n } ( \delta )
$$

with probability at least 1 − 2δ. Hence, with the same probability,

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } \big ( P _ { m } ^ { \mathrm { t e s t } } \delta _ { f } ^ { \mathrm { t e s t } } \big ) _ { i } ^ { 2 } \ \leq \ \frac { \| f \| _ { \mathcal { F } _ { \mathcal { Z } } } ^ { 2 } } { \widetilde { \lambda } _ { m } } \ \big [ \widehat { \mathrm { H S I C } } ^ { \mathrm { t r a i n } } ( \widehat { h } ( X ) , S ) + 2 \varepsilon _ { n } ( \delta ) \big ] .
$$

This is the generalization statement for the FRHSIC fairness surrogate: trainingsample HSIC, plus the uniform-concentration penalty, controls the held-out projected empirical second-moment conditional gap up to the test-sample spectral factor $1 / \widetilde { \lambda } _ { m }$

## Appendix H. Additional Experiments.

H.1. Representation-level mutual information. As a representation-level diagnostic complementing the transfer experiment, FRHSIC tends to attain lower $\operatorname { M I } ( Z , S )$ than ${ \mathrm { R e g - G D P } }$ at comparable prediction performance, consistent with the learned representation carrying less information about S rather than only making a single trained head fair. Figure 8 reports $\operatorname { M I } ( Z , S )$ versus prediction performance across the λ sweep. We treat $\operatorname { M I } ( Z , S )$ as a diagnostic, not a primary fairness audit: the KSG estimator can underestimate MI for high-dimensional learned representations whose dependence structure violates the smoothness assumptions underlying nearest-neighbor entropy estimation (Gretton et al., 2012), so we interpret it together with HSIC, GDP, and downstream performance. The adversarial methods illustrate this: ADV and LAFTR can report $\mathrm { M I } \approx 0$ while their GDP remains high, which is inconsistent with ${ \mathrm { M I } } = 0 \Rightarrow Z \bot S \Rightarrow { \mathrm { G D P } } = 0$ and indicates that adversarially trained representations, often concentrated on low-dimensional manifolds, are poorly suited to nearest-neighbor entropy estimation. COMPAS is omitted because the KSG estimator is unreliable on its low-dimensional, discrete-valued features.

![](images/0116722be61e812c23102dcd243bf3b20602dad3a1ee1c7312450eb6751af0e2.jpg)  
Fig. 8. FRHSIC yields representations that leak less about S: it tends to attain lower MI(Z, S) than Reg-GDP at comparable prediction performance, consistent with representation-level rather than single-head fairness. Axes are MI(Z, S) (lower = fairer) versus performance across λ.

H.2. Transfer Experiment. The key advantage of FRHSIC over Reg-GDP is transferability: since HSIC enforces $Z \perp S .$ , any downstream prediction head built on the learned representation is guaranteed to be fair. To test this empirically, we train representations using each method (at λ = 10), freeze the encoder, and fit four diferent downstream heads on the frozen representation: linear, 2-layer MLP, random forest, and SVM. We then measure GDP for each head.

Figure 9 shows the results on all five datasets; we highlight two representative findings. On Crime $( \lambda = 1 0 0 )$ , FRHSIC attains low GDP variance across heads (std $= 0 . 0 0 1 )$ , comparable to or lower than Reg-GDP (0.002) and ADV (0.005). On Adult $( \lambda = 1 0 )$ , FRHSIC reduces cross-head GDP variance from 0.348 (Unfair) to 0.244, a 30% reduction. Notably, Reg-GDP achieves lower variance (0.058) on Adult but at the cost of collapsing accuracy to 0.758 (Table 1). Among methods that preserve accuracy, FRHSIC provides the most consistent fairness across downstream heads. The MI results in Table 1 provide complementary evidence: MI measures the total information about S in $Z ,$ bounding the discrimination achievable by any downstream head (including adversarial ones not tested here). FRHSIC’s low MI (0.008 on Adult, 0.015 on Crime) thus guarantees fairness even for worst-case downstream use, while Reg-GDP’s higher MI (0.017 on Adult, 0.056 on Crime) leaves room for adversarial exploitation despite its $\mathrm { G D P = 0 }$ for the training head.

![](images/22663deca695edef09b24918995e309f74c3f758fb2d88df0547b2f03d41528e.jpg)

![](images/4b0a1e3401e84eada568948a5839d21210c56a8cc454adfd3762541b2cfa0835.jpg)

![](images/beb474e6010a25d7918db488b6baafc2cd5bf9e670f1cf93c95632ff7e4c4bf8.jpg)  
Fig. 9. Transfer experiment: GDP of diferent downstream prediction heads trained on frozen representations. In these experiments FRHSIC maintains comparably low GDP across heads, while Reg-GDP and FREM exhibit more head-dependent variation.

H.3. Empirical Tightness of the Theorem 4.7 Bound. We empirically evaluate the tightness of the bound in Theorem 4.7 at full resolution $( m = r ,$ , where $\hat { \lambda } _ { m } = \hat { \lambda } _ { S }$ is the smallest positive eigenvalue and $P _ { m } \delta _ { f } = \delta _ { f } )$ on two real datasets (Adult and Crime; we omit COMPAS because age in this dataset takes only a small number of distinct integer values, so the centered Gram matrix $\widetilde { L }$ on S is near-low-rank and $\hat { \lambda } _ { S }$ is numerically unstable). For each dataset and each regularization strength $\lambda ,$ we compute (i) the empirical RHS $\| f \| ^ { 2 } / \widehat { \lambda } _ { S } \cdot \widehat { \mathrm { H S I C } } ( Z , S )$ on the test split, where $\hat { \lambda } _ { S }$ is the smallest positive eigenvalue of $n ^ { - 1 } \widetilde L$ , with $\widetilde { L } = H L H$ the centered Gram matrix on $S$ (eigenvalues below $1 0 ^ { - 6 } \hat { \lambda } _ { \operatorname* { m a x } }$ are treated as numerical zero when identifying the smallest positive eigenvalue, mirroring standard practice for truncated/regularized kernel inverses); and (ii) the empirical LHS $\begin{array} { r } { \frac { 1 } { n } \sum _ { i } \hat { \delta } _ { f , . } ^ { 2 } } \end{array}$ , averaged over RKHS test functions $f ( \cdot ) = k _ { \mathcal { Z } } ( z _ { 0 } , \cdot )$ for randomly sampled anchors $z _ { 0 }$ . We plot both as a function of λ on log–log axes.

Figure 10 shows the results. The bound is valid: the empirical RHS upper-bounds the LHS at every λ on both datasets. The vertical gap between the two curves can span several orders of magnitude and reflects the well-known looseness of HSIC-based bounds when $\hat { \lambda } _ { S }$ is small (the multiplier $1 / \hat { \lambda } _ { S }$ is large because $\hat { \lambda } _ { S }$ scales as a small fraction of the largest eigenvalue of $n ^ { - 1 } \widetilde L$ for the Gaussian kernel at the medianheuristic bandwidth). Crucially, the empirical bound remains finite and non-vacuous in sharp contrast with the population-level version, where the analogous constant collapses to zero for Gaussian kernels on continuous $S$ and the bound becomes vacuous so the empirical reformulation of Theorem 4.7 is quantitatively useful as a tractable surrogate. Choosing $m < r$ replaces $\hat { \lambda } _ { S }$ by the larger $\hat { \lambda } _ { m }$ and tightens the bound, at the cost of resolving only the top-m sensitive directions.

Bandwidth scaling of $\hat { \lambda } _ { S }$ . To validate the polynomial-scaling claim in Remark B.3, we additionally sweep the bandwidth $\sigma \in \{ 0 . 1 , 0 . 3 , 1 . 0 , 3 . 0 , 1 0 . 0 \}$ and record $\hat { \lambda } _ { S }$ for Adult and Crime at n = 1500 (Figure 11). On log–log axes the curves are approximately linear, with negative slopes; the magnitudes are within a factor of two of $d _ { s } \ = \ 1$ , and the negative sign confirms that the centered Gram matrix on $S$ approaches rank-one as σ grows. This empirical polynomial dependence is consistent with Remark B.3.

![](images/30bbfb92b8afa5fc0691686d93cecc8d43c5684254bdbeae09b7d43c832bdf00.jpg)  
Fig. 10. Empirical tightness of the bound from Theorem 4.7. Solid line: empirical LHS $\textstyle { \frac { 1 } { n } } \sum _ { i } \hat { \delta } _ { f , i } ^ { 2 }$ Dashed line: empirical RHS $\| f \| ^ { 2 } / \hat { \lambda } _ { S } \cdot \widehat { \mathrm { H S I C } }$ Validity: the RHS upper-bounds the LHS at every λ. The vertical gap can span several orders of magnitude and reflects the looseness of HSIC-based bounds when $\hat { \lambda } _ { S }$ is small; the bound nonetheless remains finite and non-vacuous, in contrast to the population version, where the constant collapses to zero.

![](images/10cca5233b1709f1f9a52d7a08de400aeff4caf364176c23c1b76d19af4dd5f5.jpg)  
Fig. 11. Empirical $\hat { \lambda } _ { S }$ as a function of bandwidth σ on Adult and Crime, $n = 1 5 0 0$ . Log– log axes; the approximately linear scaling (with empirical slopes shown in the legend) supports the polynomial bandwidth dependence noted in Remark B.3.

H.4. High-Dimensional Sensitive Attributes. A practical consequence of computing on the joint distribution rather than on conditional families is that FRHSIC tolerates increasing dim(S) gracefully, while methods built on the integral framework $\mathcal { T } _ { d }$ rely internally on a kernel-smoothed conditional weighting $\hat { w } _ { \gamma } ( j ; i )$ on $s$ whose statistical accuracy is subject to the curse of dimensionality of nonparametric conditional density estimation. We illustrate the per-iteration computational consequences on the Adult dataset by constructing S as a d-dimensional vector from the continuous Adult features (age, capital gain, capital loss, hours per week, plus a synthetic Gaussian feature for $d = 5 )$ , sweeping $d \in \{ 1 , 2 , 3 , 4 , 5 \}$ , and comparing the per-epoch training time of FRHSIC and FREM at fixed $\lambda = 1 0$ and fixed bandwidths $( \sigma _ { S } = 1 . 0$ for FRHSIC, $\gamma = 0 . 5$ for FREM); see Figure 12.

![](images/af1adfe46715cc956759096ce9beacd0dcbde032e1240c0054d725033d703e75.jpg)  
Fig. 12. Per-epoch training time on Adult $( n = 2 0 0 0 )$ as a function of dim(S), at fixed $\lambda = 1 0$ and fixed kernel bandwidths. $F R H S I C ^ { \prime } s$ HSIC loss is essentially insensitive to d (the dimension enters only through the n × n kernel evaluation on S). FREM is consistent $\cdot { } l y \ 2 \substack { - 3 \times }$ slower per epoch than FRHSIC across all d; the slowdown is dominated by the kernel-smoothed conditional weighting $\hat { w } _ { \gamma } ( j ; i )$ , which evaluates a d-dimensional Gaussian kernel and an anchor-subsampled MMD on every batch.

In our wall-clock measurements both curves are roughly flat in d over the range we test, with FREM exhibiting a uniform $\sim 2 \ – 3 \times$ overhead at this scale; the gap grows to roughly 36× at $n = 2 0 { , } 0 0 0$ (see Section 6.5), where FREM’s $O ( n ^ { 3 } )$ scaling dominates. At $n = 2 0 0 0$ with an anchor-subsampled MMD (32 anchors), the perbatch cost is dominated by the $n \times n$ kernel matrix on $\mathcal { Z }$ , which is d-independent. The joint-distribution route is therefore primarily a statistical (rather than computational) advantage at moderate n: nonparametric conditional weighting on $s$ inherits the slow rates of d-dimensional density estimation, while the HSIC kernel-matrix computation is d-agnostic. The constant gap we observe nonetheless reflects the per-iteration overhead of constructing conditional weights in $\mathcal { T } _ { d } .$ -style estimators.

## H.5. Ablation Studies.

Kernel choice. We evaluate FRHSIC with diferent kernel functions for $k z$ and $k _ { S } \colon$ Gaussian (RBF), Laplacian, and inverse multiquadric (IMQ). Table 6 shows that performance is robust across kernel choices, with the Gaussian kernel performing slightly better overall. This is consistent with the theoretical requirement that the kernels be characteristic, which all three satisfy.

Representation dimensionality. We vary the representation dimension $d _ { Z } \in$ $\{ 2 , 8 , 3 2 , 6 4 \}$ with $\lambda \ : = \ : 5$ (Table 7). All dimensions achieve comparable accuracy $( \approx ~ 0 . 6 2 )$ , but GDP grows with dimensionality, from 0.0001 at $d _ { Z } \ = \ 2$ to 0.0034 at $d _ { Z } = 6 4$ . This confirms that higher-dimensional representations make the HSIC penalty less efective per unit of λ. The default $d _ { Z } = 8$ provides a good balance.

Table 6  
Ablation over kernel choice (synthetic data, $\lambda = 5 )$ . All kernels achieve comparable performance.
<table><tr><td> $k z$ </td><td> $k _ { S }$ </td><td>Accuracy</td><td>GDP</td></tr><tr><td>Gaussian</td><td>Gaussian</td><td>0.620</td><td>0.0015</td></tr><tr><td>Gaussian</td><td>Laplacian</td><td>0.620</td><td>0.0003</td></tr><tr><td>Gaussian</td><td>IMQ</td><td>0.620</td><td>0.0006</td></tr><tr><td>Laplacian</td><td>Gaussian</td><td>0.619</td><td>0.0091</td></tr><tr><td>Laplacian</td><td>Laplacian</td><td>0.620</td><td>0.0001</td></tr><tr><td>IMQ</td><td>Gaussian</td><td>0.620</td><td>0.0002</td></tr><tr><td>IMQ</td><td>IMQ</td><td>0.620</td><td>0.0027</td></tr></table>

Table 7

Ablation over representation dimensionality (synthetic data, $\lambda = 5 )$
<table><tr><td> $d _ { Z }$ </td><td>Accuracy</td><td>GDP</td></tr><tr><td>2</td><td>0.620</td><td>0.0001</td></tr><tr><td>8</td><td>0.620</td><td>0.0001</td></tr><tr><td>32</td><td>0.618</td><td>0.0025</td></tr><tr><td>64</td><td>0.620</td><td>0.0034</td></tr></table>

H.6. Validation-based regularization selection. Selecting the regularization strength λ is a practical challenge in fair representation learning. We use the HSIC independence test (Gretton et al., 2005) as a validation-based stopping criterion. The HSIC test statistic under the null hypothesis $H _ { 0 } : Z \perp S$ follows a weighted sum of chi-squared variables; we use the gamma approximation (Gretton et al., 2005), under which $\widehat { n \cdot \mathrm { H S I C } }$ is approximately gamma-distributed with parameters estimated from data.

Algorithm: HSIC-test-based λ selection.

1. Input: sample $\{ ( X _ { i } , S _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n } ;$ kernels $k _ { \mathcal { Z } } , k _ { \mathcal { S } }$ ; increasing λ grid $\Lambda = \{ \lambda ^ { ( 1 ) } <$ $\cdots < \lambda ^ { ( K ) } \big \}$ ; significance level $\alpha _ { \mathrm { t e s t } } ~ ( \mathrm { e . g . , 0 . 0 5 ) }$

2. For each $\lambda ^ { ( j ) } \in \Lambda$ in increasing order: train FRHSIC at $\lambda ^ { ( j ) }$ to obtain $\hat { h } _ { \lambda ^ { ( j ) } } ;$ compute $\widehat { \mathrm { H S I C } } ( \hat { h } _ { \lambda ( j ) } ( X ) , S )$ on a held-out validation split; run the gammaapproximated HSIC independence test at level $\alpha _ { \mathrm { t e s t } }$ (Gretton et al., 2005); if the test fails to reject $H _ { 0 } : Z \perp S$ , return $\lambda ^ { ( j ) }$

3. Output: the smallest $\lambda ^ { ( j ) }$ at which the test fails to reject independence, or $\overline { { \lambda ^ { ( K ) } } } \mathrm { { { i f } } }$ no such λ is found.

This is a practical validation heuristic based on a familiar independence test, not a principled selection rule. On large validation sets the test can be overpowered, detecting statistically significant but practically negligible dependence. The practicalsignificance variant stops at the smallest $\lambda ^ { ( j ) }$ such that $\widehat { \mathrm { H S I C } } < \epsilon$ or the test fails to reject at level $\alpha _ { \mathrm { t e s t } }$ , where $\epsilon > 0$ is a user-specified threshold $( \mathrm { e . g . } , \epsilon = 1 0 ^ { - 3 } )$ , keeping the procedure useful regardless of sample size.

We validate the procedure as follows. For each dataset, we train FRHSIC over the grid

$$
\lambda ^ { ( j ) } \in \{ 0 . 0 1 , 0 . 1 , 0 . 5 , 1 , 5 , 1 0 , 5 0 , 1 0 0 , 5 0 0 \} ,
$$

evaluate $\widehat { \mathrm { H S I C } }$ on a held-out validation set (15% of data), and apply the gamma-

![](images/654e08126ad61eac259e7ef5768a548563580b1291e44206f77982fcd6ffd826.jpg)

![](images/76308e6285177471b0d9e848faea24b5f345b43244fdb514c3e2d459b79f31af.jpg)

![](images/df6782a0b9e42039f4cf61b0eab55a6cee7a61fc032c1093c8a7708bda604b63.jpg)  
Fig. 13. Regularization strength selection via the HSIC independence test (Appendix $H . 6 )$ . Red circles: empirical HSIC on the validation set. Blue squares: test threshold at $\alpha = 0 . 0 5$ Green shading: λ values where the test fails to reject independence. The vertical dashed line marks the selected λ.

approximation independence test at level $\alpha = 0 . 0 5$

Figure 13 shows the results. On Crime and COMPAS, the test successfully identifies a transition: as λ increases, the HSIC test statistic falls below the rejection threshold, selecting $\lambda \approx 5 – 1 0 0$ depending on the dataset and random split. At the selected λ, the representation achieves low GDP while preserving prediction performance.

On Adult $( n \approx 3 0 , 0 0 0 )$ , the test rejects independence at all λ values, including $\lambda =$ 500 where $\mathrm { G D P }$ is near zero. This reflects a well-known property of hypothesis testing: with large samples, the test detects statistically significant but practically negligible dependence. In such settings, we recommend supplementing the independence test with a practical significance threshold $\left( \mathrm { e . g . } \right.$ ., requiring $\widehat { \mathrm { H S I C } } < \epsilon$ for a user-specified ϵ) or using the test on a smaller held-out subsample to reduce power.

## References.

M´elisande Albert, B´eatrice Laurent, Amandine Marrel, and Anouar Meynaoui. Adaptive test of independence based on HSIC measures. The Annals of Statistics, 50(2): 858–879, 2022.

Julia Angwin, Jef Larson, Surya Mattu, and Lauren Kirchner. Machine bias. ProPublica, 2016.

Toon Calders and Sicco Verwer. Three naive Bayes approaches for discrimination-free classification. Data mining and knowledge discovery, 21:277–292, 2010.

Jaewoong Cho, Gyeongjo Hwang, and Changho Suh. A fair classifier using kernel density estimation. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 15088–15099. Curran Associates, Inc., 2020a. URL https://proceedings.neurips. cc/paper files/paper/2020/file/ac3870fcad1cfc367825cda0101eee62-Paper.pdf.

Jaewoong Cho, Gyeongjo Hwang, and Changho Suh. A fair classifier using mutual in formation. In 2020 IEEE International Symposium on Information Theory (ISIT), pages 2521–2526, 2020b.

Thomas M. Cover and Joy A. Thomas. Elements of Information Theory. Wiley-Interscience, 2 edition, 2006.

Elliot Creager, David Madras, J¨orn-Henrik Jacobsen, Marissa Weis, Kevin Swersky, Toniann Pitassi, and Richard Zemel. Flexibly fair representation learning by disentanglement. In International conference on machine learning, pages 1436–1445. PMLR, 2019.

Namrata Deka and Danica J Sutherland. MMD-B-Fair: Learning fair representations with statistical testing. In International Conference on Artificial Intelligence and

Statistics, pages 9564–9576. PMLR, 2023.

Frances Ding, Moritz Hardt, John Miller, and Ludwig Schmidt. Retiring adult: New datasets for fair machine learning. In Advances in Neural Information Processing Systems, volume 34, pages 6478–6490, 2021.

Kenji Fukumizu, Francis R. Bach, and Arthur Gretton. Statistical consistency of kernel canonical correlation analysis. Journal of Machine Learning Research, 8: 361–383, 2007.

Vincent Grari, Sylvain Lamprier, and Marcin Detyniecki. Fairness without the sensitive attribute via causal variational autoencoder. In International Joint Conference on Artificial Intelligence, 2022.

Arthur Gretton, Olivier Bousquet, Alex Smola, and Bernhard Sch¨olkopf. Measuring statistical dependence with Hilbert-Schmidt norms. In Algorithmic Learning Theory: 16th International Conference, ALT 2005, Singapore, October 8-11, 2005. Proceedings 16, pages 63–77. Springer, 2005.

Arthur Gretton, Kenji Fukumizu, Choon Teo, Le Song, Bernhard Sch¨olkopf, and Alex Smola. A kernel statistical test of independence. Advances in neural information processing systems, 20, 2007.

Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Sch¨olkopf, and Alexander Smola. A kernel two-sample test. The Journal of Machine Learning Research, 13(1):723–773, 2012.

Zhimeng Jiang, Xiaotian Han, Chao Fan, Fan Yang, Ali Mostafavi, and Xia Hu. Generalized demographic parity for group fairness. In International Conference on Learning Representations, 2022.

Olav Kallenberg. Foundations of Modern Probability. Springer, 2 edition, 2002.

Vladimir Koltchinskii and Evarist Gin´e. Random matrix approximation of spectra of integral operators. Bernoulli, 6(1):113–167, 2000.

Insung Kong, Kunwoong Kim, and Yongdai Kim. Fair representation learning for continuous sensitive attributes using expectation of integral probability metrics. IEEE transactions on pattern analysis and machine intelligence, 2025.

Matt J Kusner, Joshua Loftus, Chris Russell, and Ricardo Silva. Counterfactual fairness. In Advances in Neural Information Processing Systems, volume 30, 2017.

Zhu Li, Adri´an P´erez-Suay, Gustau Camps-Valls, and Dino Sejdinovic. Kernel dependence regularizers and Gaussian processes with applications to algorithmic fairness. Pattern Recognition, 132:108922, 2022.

Christos Louizos, Kevin Swersky, Yujia Li, Max Welling, and Richard Zemel. The variational fair autoencoder. In International Conference on Learning Representations, 2016.

Wan-Duo Kurt Ma, JP Lewis, and W Bastiaan Kleijn. The HSIC bottleneck: Deep learning without back-propagation. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 5085–5092, 2020.

Zhengming Ma, Zengrong Zhan, Xiaoyuan Ouyang, and Xue Su. Nonlinear dimensionality reduction based on HSIC maximization. IEEE Access, 6:55537–55555, 2018.

David Madras, Elliot Creager, Toniann Pitassi, and Richard Zemel. Learning adversarially fair and transferable representations. In International Conference on Machine Learning, pages 3384–3393. PMLR, 2018.

Jean Mary, Cl´ement Calauz\`enes, and Noureddine El Karoui. Fairness-aware learning for continuous attributes and treatments. In International Conference on Machine Learning, pages 4382–4391, 2019.

Elizbar A Nadaraya. On estimating regression. Theory of Probability & Its Applica-

tions, 9(1):141–142, 1964.

Yijin Ni and Xiaoming Huo. A uniform concentration inequality for kernel-based two-sample statistics. arXiv preprint arXiv:2405.14051, 2024.

Judea Pearl. Causality: Models, Reasoning, and Inference. Cambridge University Press, 2 edition, 2009.

Adri´an P´erez-Suay, Valero Laparra, Gonzalo Mateo-Garc´ıa, Jordi Mu˜noz-Mar´ı, Luis G´omez-Chova, and Gustau Camps-Valls. Fair kernel learning. In Machine Learning and Knowledge Discovery in Databases (ECML PKDD), pages 339–355. Springer, 2017.

Novi Quadrianto, Viktoriia Sharmanska, and Oliver Thomas. Discovering fair representations in the data domain. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

Yaniv Romano, Stephen Bates, and Emmanuel J Cand\`es. Achieving equalized odds by resampling sensitive attributes. In Advances in Neural Information Processing Systems, volume 33, pages 361–371, 2020.

Dino Sejdinovic, Bharath Sriperumbudur, Arthur Gretton, and Kenji Fukumizu. Equivalence of distance-based and RKHS-based statistics in hypothesis testing. The annals of statistics, pages 2263–2291, 2013.

Le Song, Alex Smola, Arthur Gretton, Karsten M Borgwardt, and Justin Bedo. Supervised feature selection via dependence estimation. In Proceedings of the 24th international conference on Machine learning, pages 823–830, 2007.

Le Song, Alex Smola, Arthur Gretton, Justin Bedo, and Karsten Borgwardt. Feature selection via dependence maximization. The Journal of Machine Learning Research, 13(1):1393–1434, 2012.

Zolt´an Szab´o and Bharath K Sriperumbudur. Characteristic and universal tensor product kernels. Journal of Machine Learning Research, 18(233):1–29, 2018.

G´abor J Sz´ekely, Maria L Rizzo, and Nail K Bakirov. Measuring and testing dependence by correlation of distances. The annals of statistics, 35(6):2769–2794, 2007.

Geofrey S Watson. Smooth regression analysis. Sankhy¯a: The Indian Journal of Statistics, Series A, pages 359–372, 1964.

Rich Zemel, Yu Wu, Kevin Swersky, Toni Pitassi, and Cynthia Dwork. Learning fair representations. In Sanjoy Dasgupta and David McAllester, editors, Proceedings of the 30th International Conference on Machine Learning, volume 28 of Proceedings of Machine Learning Research, pages 325–333, Atlanta, Georgia, USA, 17–19 Jun 2013. PMLR. URL https://proceedings.mlr.press/v28/zemel13.html.