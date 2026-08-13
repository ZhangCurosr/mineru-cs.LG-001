# An Eficient Near-Optimal Algorithm for Adversarial m-Set Bandits

Francesco Bacchiocchi<sup>1</sup>, Tommaso Cesari<sup>2</sup>, and Roberto Colomboni<sup>3</sup>

<sup>1</sup>DEIB, Politecnico di Milano, Milano, Italy

<sup>2</sup>School of Electrical Engineering and Computer Science, University of Ottawa, Ottawa, Canada <sup>3</sup>School of Mathematics, University of Bristol, Bristol, United Kingdom francesco.bacchiocchi@polimi.it, tcesari@uottawa.ca, roberto.colomboni@bristol.ac.uk

August 13, 2026

## Abstract

We study adversarial combinatorial bandits with m-set actions, where at each round the learner selects m out of d items and observes only the aggregate loss of the selected items. The resulting action set contains $K = { \binom { d } { m } }$ elements and can therefore be exponentially large. Nevertheless, the loss of every action is determined by the same d-dimensional vector of item losses. We propose a computationally eficient algorithm that exploits this structure without explicitly enumerating the action set. Against adaptive non-anticipating adversaries, it guarantees, with probability at least 1 − δ, regret against the best fixed action of

$$
R _ { T } = O \left( { \sqrt { d T \log ( K / \delta ) } } \right) .
$$

This matches the high-probability regret bound of the finite-action EXP3–KW algorithm of Zimmert and Lattimore [29, Theorem 6 and Algorithm 3], whose direct implementation may require exponential space. Our algorithm instead represents each sampling distribution with d parameters and runs in polynomial time without enumerating the action set. Thus, it resolves the open problem posed by Maiti et al. [26].

## 1 Introduction

In this paper, we study adversarial m-set bandits. At each round, the learner selects m out of d items and incurs the sum of their losses, but observes only the aggregate loss, not the loss of each selected item. Since any m-subset may be chosen, the learner has $K = { \binom { d } { m } }$ possible actions, and its goal is to minimize regret with respect to the best fixed m-set in hindsight.

For finite action sets, Zimmert and Lattimore [29, Theorem 6 and Algorithm 3] show that EXP3–KW achieves a high-probability regret bound that, for m-sets, becomes $\widetilde { O } ( \sqrt { d m T } )$ when $m \leq d / 2 . 1$ However, its direct implementation maintains one weight for every action, and since $K \ = \ { \overset { . } { ( } } { \overset { d } { _ { m } } } )$ , this can require exponential space. A possible workaround is to use the results in [26]. Specifically, Maiti et al. [26, Appendix E.6] give a polynomial-time high-probability algorithm based on a general directed acyclic graph representation that, for m-sets, turns into an algorithm achieving $\widetilde { O } ( d \sqrt { m T } )$ regret bound. Despite this implementation being computationally eficient, it remains unclear whether one can eficiently obtain the sharper rate $\widetilde { O } ( \sqrt { d m T } )$

We provide a positive answer to this question with a polynomial-time algorithm. Against adaptive non-anticipating adversaries, our algorithm guarantees $\widetilde { O } ( \sqrt { d s T } )$ regret with high probability, where $s : = \dim \{ m , d - m \}$ . In the regime $m \le d / 2$ , we have $s ~ = ~ m$ , and the bound becomes $\widetilde { O } ( \sqrt { d m T } )$ , and thus it matches the regret rate of finite-action EXP3–KW by Zimmert and Lattimore [29, Theorem 6 and Algorithm 3].

To obtain our eficient algorithm, we work with weighted m-set distributions, a family of distribution that can be represented by d item weights, from which one can sample and compute the required moments in polynomial time [11, 24]. The high-level idea is to implement the finite-action EXP3–KW algorithm while keeping the sampling distributions at all iterates within this family. The main obstacle is that the updates of EXP3-KW come with a quadratic correction, which does not preserve the weighted m-set structure. We overcome this dificulty by using the covariance bound of Cesari and Colomboni [10, Corollary 1.3] to construct a carefully chosen afine majorant of the quadratic correction on the m-set action space: exponential-weights updates using this majorant preserve the family of weighted m-set distributions, while the majorant also provides the estimation-error control needed to obtain the desired high-probability regret rate.

## 2 Setting and problem formulation

Preliminary definitions. For a positive integer n, we write $[ n ] : = \{ 1 , . . . , n \}$ . For a vector $z = ( z _ { 1 } , \ldots , z _ { n } )$ and $k \in [ n ]$ , we define its k-th elementary symmetric polynomial as

$$
e _ { k } ( z ) : = \sum _ { S \subseteq [ n ] : | S | = k } \prod _ { i \in S } z _ { i } , \qquad e _ { 0 } ( z ) : = 1 , \qquad e _ { k } ( z ) : = 0 \quad { \mathrm { f o r ~ } } k < 0 { \mathrm { ~ o r ~ } } k > n .
$$

Fix two integers $d \geq 2$ and $1 \leq m \leq d - 1$ . We define the action set and its size as

$$
\mathcal { X } : = \mathcal { X } _ { d , m } : = \Big \{ x \in \{ 0 , 1 \} ^ { d } : \| x \| _ { 1 } = m \Big \} , \qquad K : = | \mathcal { X } | = { \binom { d } { m } } .\tag{1}
$$

Each action $x \in \mathcal { X }$ selects exactly m of the d items, $x _ { i } = 1$ if item i is selected and $x _ { i } = 0$ otherwise. Its support is sup $) ( x ) : = \{ i \in [ d ] : x _ { i } = 1 \}$ . Thus, the support of each action is an m-element subset of [d], which we call an m-set. Conversely, each m-set $S \subseteq [ d ]$ corresponds to the indicator vector ${ \mathbf { 1 } } _ { S } \in { \mathcal { X } }$ . We therefore use actions in X and m-sets interchangeably.

Let $s : = \operatorname* { m i n } \{ m , d - m \}$ . By the symmetry of the binomial coeficient, log $K \leq s \log \left( e d / s \right)$

Linear-algebra notation. For a symmetric matrix $A \in \mathbb { R } ^ { d \times d }$ , we denote its kernel by ker $( A ) : =$ $\{ z \in \mathbb { R } ^ { d } : A z = 0 \}$ and its Moore–Penrose pseudoinverse by $A ^ { + }$ . For two symmetric matrices $A , B \in \mathbb { R } ^ { d \times d }$ , we write $A \succeq B$ if $A - B$ is positive semidefinite or, equivalently

$$
z ^ { \top } A z \geq z ^ { \top } B z \qquad { \mathrm { f o r ~ e v e r y ~ } } z \in \mathbb { R } ^ { d } .
$$

We write $A \succ 0 \mathrm { ~ i f ~ } z ^ { \top } A z > 0$ for every nonzero $z \in \mathbb { R } ^ { d }$ . In particular, A is positive semidefinite if $A \succeq 0$ , and positive definite if $A \succ 0$ . Let $\mathbf { 1 } : = ( 1 , \ldots , 1 ) ^ { \top }$ , and define

$$
\operatorname { s p a n } \{ \mathbf { 1 } \} : = \{ a \mathbf { 1 } : a \in \mathbb { R } \} , \qquad H _ { 0 } : = \{ y \in \mathbb { R } ^ { d } : \langle y , \mathbf { 1 } \rangle = 0 \} .
$$

Thus, $H _ { 0 }$ is the subspace orthogonal to span{1}. If A is positive semidefinite and ker $( A ) = \operatorname { s p a n } \{ { \bf 1 } \}$ then A is invertible on $H _ { 0 }$ . For every $y \in H _ { 0 }$ , the vector $A ^ { + } y$ is the inverse of A applied to y within this subspace.

We will use the order-reversing property of matrix inversion [18, Corollary $\mathrm { 7 . 7 . 4 ( a ) ] }$ ]. Specifically, suppose that A and B are positive semidefinite linear operator with

$$
\ker ( A ) = \ker ( B ) = \operatorname { s p a n } \{ \mathbf { 1 } \} ,
$$

then their restrictions to $H _ { 0 } = \mathbf { 1 } ^ { \perp }$ are positive definite. Consequently, if $A \succeq c B$ , for $c > 0$ , then

$$
\left( { \cal A } | _ { { \cal H } _ { 0 } } \right) ^ { - 1 } \preceq \frac { 1 } { c } \left( { \cal B } | _ { { \cal H } _ { 0 } } \right) ^ { - 1 } .
$$

Equivalently,

$$
A ^ { + } \preceq { \frac { 1 } { c } } B ^ { + } .
$$

In particular,

$$
y ^ { \top } A ^ { + } y \leq { \frac { 1 } { c } } y ^ { \top } B ^ { + } y \qquad { \mathrm { f o r ~ e v e r y ~ } } y \in H _ { 0 } .
$$

Probability notation. For a finite set $\mathcal { V } \subseteq \mathbb { R } ^ { d }$ , we denote its probability simplex by

$$
\Delta ( \mathcal { V } ) : = \{ p : \mathcal { V }  [ 0 , 1 ] : \sum _ { y \in \mathcal { Y } } p ( y ) = 1 \} .
$$

For $p , q \in \Delta ( \mathcal { V } )$ , we define the entropy of $p$ and the Kullback–Leibler (KL) divergence from $p$ to q as

$$
H ( p ) : = - \sum _ { y \in \mathcal { Y } } p ( y ) \log p ( y ) , \qquad \mathrm { K L } ( p \parallel q ) : = \sum _ { y \in \mathcal { Y } } p ( y ) \log \frac { p ( y ) } { q ( y ) } .
$$

We use the convention $0 \log ( 0 / q ) = 0$ , and set K $\operatorname { L } ( p \parallel q ) : = + \infty { \mathrm { ~ i f ~ } } p ( y ) > 0 = q ( y )$ for some $y \in \mathcal { V }$

Let X be a random vector drawn from $p \in \Delta ( \mathfrak { V } )$ . We define the marginal vector, second-moment matrix, and covariance matrix of p as

$$
\mu ( p ) : = \mathbb { E } _ { X \sim p } [ X ] , \qquad M ( p ) : = \mathbb { E } _ { X \sim p } [ X X ^ { \top } ] , \qquad \Sigma ( p ) : = \mathrm { C o v } _ { X \sim p } ( X ) = M ( p ) - \mu ( p ) \mu ( p ) ^ { \top } .
$$

When $\mathcal { V } = \mathcal { X }$ , the i-th marginal probability is

$$
\mu _ { i } ( p ) = \mathbb { E } _ { X \sim p } [ X _ { i } ] = \mathbb { P } _ { X \sim p } [ X _ { i } = 1 ] .
$$

Thus, $\mu _ { i } ( p )$ is the probability that action X selects item i. If $M ( p )$ is invertible, we refer to the inverse-design quadratic form

$$
x ^ { \top } M ( p ) ^ { - 1 } x
$$

as the leverage score of x under $p .$

Weighted m-set distributions. Given $\boldsymbol { \theta } \in \mathbb { R } ^ { d }$ , define $w _ { i } : = e ^ { \theta _ { i } } > 0$ for every $i \in [ d ]$ . For every m-set $S \subseteq [ d ]$ , the weighted m-set distribution $p _ { \theta }$ is

$$
p _ { \theta } ( S ) : = \frac { \exp \left( \sum _ { i \in S } \theta _ { i } \right) } { \sum _ { A \subseteq [ d ] : | A | = m } \exp \left( \sum _ { i \in A } \theta _ { i } \right) } = \frac { \prod _ { i \in S } w _ { i } } { e _ { m } ( w ) } .
$$

Its log-partition function is

$$
F ( \theta ) : = \log \sum _ { \substack { A \subseteq [ d ] : | A | = m } } \exp \left( \sum _ { i \in A } \theta _ { i } \right) = \log e _ { m } ( e ^ { \theta _ { 1 } } , \ldots , e ^ { \theta _ { d } } ) .
$$

If $S \sim p _ { \theta }$ and $X : = \mathbf { 1 } _ { S } \in \mathcal { X }$ is its indicator vector, then

$$
\nabla F ( \theta ) = \mathbb { E } _ { X \sim p _ { \theta } } [ X ] = \mu ( p _ { \theta } ) , \qquad \nabla ^ { 2 } F ( \theta ) = \operatorname { C o v } _ { X \sim p _ { \theta } } ( X ) = \Sigma ( p _ { \theta } ) .
$$

Although $p _ { \theta }$ may assign positive probability to exponentially many actions, the d parameters $\theta _ { 1 } , \ldots , \theta _ { d }$ represent it. Finally, we define a set of distributions whose marginal probabilities stay away from 0 and 1. Let $r : = m / d$ and fix $\lambda \in ( 0 , 1 )$ . Define

$$
\mathcal { P } _ { \lambda } : = \left\{ p \in \Delta ( \mathcal { X } ) : \lambda r \leq \mu _ { i } ( p ) \leq 1 - \lambda ( 1 - r ) \mathrm { ~ f o r ~ e v e r y ~ } i \in [ d ] \right\} .\tag{2}
$$

The uniform distribution U over X satisfies $\mu _ { i } ( U ) = r$ for every $i \in [ d ]$ . Therefore, $U \in { \mathcal { P } } _ { \lambda }$

Adversarial m-set bandit protocol. Fix a time horizon $T \geq 1$ . Let $( \mathcal { F } _ { t } ) _ { t = 0 } ^ { T }$ denote the filtration generated by the interaction history, where $\mathcal { F } _ { t - 1 }$ contains the information available before round t. At each round $t \in [ T ]$ , the learner selects an $\mathcal { F } _ { t - 1 }$ -measurable distribution $p _ { t } \in \Delta ( \mathcal X )$ , where $x ,$ defined in Equation (1), represents the set of actions available to the learner. The adversary selects a loss vector $\boldsymbol { \ell } _ { t } \in \mathbb { R } ^ { d }$ that is also $\mathcal { F } _ { t - 1 }$ -measurable. Thus, the adversary is adaptive nonanticipating. The loss vector $\ell _ { t }$ may depend on the entire interaction history up to round $t - 1$ but not on the learner’s random draw at round t. Therefore, conditionally on $\mathcal { F } _ { t - 1 }$ , both $p _ { t }$ and $\ell _ { t }$ are fixed. The learner then draws $X _ { t } \sim p _ { t }$ and observes only the loss $Y _ { t } : = \langle X _ { t } , \ell _ { t } \rangle$ . We use the standard action-level normalization

$$
| \langle x , \ell _ { t } \rangle | \leq 1 \qquad \forall x \in \mathcal { X } , \ t \in [ T ] ,
$$

also adopted by Maiti et al. [26, Assumption 1] and Zimmert and Lattimore [29, Section 1]. <sup>2</sup> The learner’s realized regret after T rounds, measured against the best fixed action in hindsight, is

$$
R _ { T } : = \sum _ { t = 1 } ^ { T } \langle X _ { t } , \ell _ { t } \rangle - \operatorname* { m i n } _ { x \in \mathcal { X } } \sum _ { t = 1 } ^ { T } \langle x , \ell _ { t } \rangle .
$$

Throughout, we use the shorthand $\mathbb { E } _ { t } [ \cdot ] : = \mathbb { E } [ \cdot | \mathcal { F } _ { t - 1 } ]$ for conditional expectation given the information available before round $t \in [ T ]$

## 3 Original contributions

Our main contribution is an algorithm (Algorithm 1) for adversarial m-set bandits that achieves the regret guarantee summarized in the following.

Theorem (Informal version of Theorem 1). There exists an algorithm that can be implemented in polynomial time such that, against any adaptive non-anticipating adversary, with probability at least $1 - \delta ,$ , it achieves

$$
R _ { T } = O \left( \sqrt { d T \left( \log \left( \int _ { m } ^ { d } \right) + \log \frac { 1 } { \delta } \right) } \right) .
$$

For example, one may take Algorithm 1.

We first discuss the state-of-the-art results for adversarial m-set bandits. Maiti et al. [26, Appendix E.6] obtain a high-probability regret bound of $\widetilde { O } ( d \sqrt { m T } )$ for m-set bandits and leave open whether the extra $\sqrt { d }$ factor in this bound can be removed with an eficient algorithm. Although this consequence is not stated explicitly in their paper, applying their general DAG guarantee [26, Theorem 7] to the m-set DAG constructed in Maiti et al. [26, Appendix E.6], which has $O ( m d )$

edges and $K = { \binom { d } { m } }$ paths, and using log $K \leq m \log ( e d / m )$ yields, for $m \leq d / 2$ , the regret bound $\widetilde { O } ( m \sqrt { d T } )$

Our algorithm achieves a regret upper bound of $\widetilde { O } ( \sqrt { d s T } )$ , where $s : = \dim \{ m , d - m \}$ . In particular, when $m \leq d / 2$ , we have $s = m$ , and the bound becomes $\widetilde { O } ( \sqrt { d m T } )$ . Thus, our bound improves the $\widetilde { O } ( d \sqrt { m T } )$ guarantee of Maiti et al. [26, Appendix $\mathrm { E . 6 ] }$ by a factor of ${ \sqrt { d } } ,$ and the derived $\widetilde { O } ( m \sqrt { d T } )$ consequence above [26, Theorem 7 and Appendix E.6] by a factor of $\sqrt { m }$ Consequently, relative to the better of these two bounds in the regime $m \leq d / 2$ , the improvement is a factor of $\sqrt { m }$

Zimmert and Lattimore [29, Theorem 6 and Algorithm 3] obtain a high-probability regret bound that, when $m \leq d / 2 .$ , yields the rate $\widetilde { O } ( \sqrt { d m T } )$ , matching our regret bound. However, their finiteaction exponential-weights algorithm is stated with one weight for each of the $\begin{array} { r } { K = { \binom { d } { m } } } \end{array}$ actions, which can be exponential in d. Our algorithm achieves the same regret rate using only d parameters and running in polynomial time (see Section 5.3 for the implementation details).

In Table 1, we summarize the best known high-probability guarantees with $\sqrt { T }$ dependence for adversarial m-set bandits. All results reported there consider action-normalized losses and adaptive non-anticipating adversaries.

We next discuss the regret lower bound in the regime m $\leq d / 2$ , making its dependence on $d - m$ explicit. Let $\rho : = ( d - m ) / d$ . When the loss vectors belong to $\{ 0 , 1 \} ^ { d }$ Ito et al. [19, Theorem 2] prove the expected-regret lower bound

$$
\Omega \left( \operatorname* { m i n } \left\{ \rho ^ { 2 } \sqrt { d m ^ { 3 } T } , \rho m ^ { 3 / 4 } T \right\} \right) .
$$

To compare this result with our action-normalized setting, observe that an m-set may incur a loss as large as m when the loss vectors belong to $\{ 0 , 1 \} ^ { d }$ . We therefore divide the losses by m. Since regret scales linearly with the losses, this yields the action-normalized lower bound

$$
\Omega \left( \operatorname* { m i n } \left\{ \rho ^ { 2 } \sqrt { d m T } , \frac { \rho T } { m ^ { 1 / 4 } } \right\} \right) .
$$

Consequently, when $m \leq d / 2$ and $T \geq d m ^ { 3 / 2 }$ , this lower bound is $\Omega ( { \sqrt { d m T } } )$ . On the other hand, setting $\delta = T ^ { - 2 }$ in our high-probability guarantee and using $R _ { T } \leq 2 T$ yields an expected-regret upper bound of $\widetilde { O } ( \sqrt { d m T } )$ . Therefore, our rate matches the minimax expected-regret lower bound up to logarithmic factors in this regime. In particular, any improvement in the polynomial dependence of a uniform high-probability bound valid for $\delta = T ^ { - 2 }$ would imply the same improvement in expected regret and would therefore contradict this lower bound.

<table><tr><td>Work</td><td>Regret bound</td><td>Computational complexity</td></tr><tr><td>Zimmert and Lattimore [29] (Theorem 6 and Algorithm 3)</td><td> $\widetilde { O } ( \sqrt { d m T } )$ </td><td>Exponential</td></tr><tr><td>Maiti et al. [26] (Appendix E.6)</td><td> $\widetilde { O } ( d \sqrt { m T } )$ </td><td>Polynomial</td></tr><tr><td>Maiti et al. [26] (Theorem 7 applied here to the DAG of Appendix E.6)</td><td> $\widetilde { O } ( m \sqrt { d T } )$ </td><td>Polynomial</td></tr><tr><td>This work</td><td> $\widetilde { O } ( \sqrt { d m T } )$ </td><td>Polynomial</td></tr></table>

Table 1: Statistical and computational comparison for the regime $m \leq d / 2$ . All upper bounds hold with high probability against adaptive non-anticipating adversaries.

Challenges and techniques. In their finite-action EXP3–KW algorithm, Zimmert and Lattimore [29, Algorithm 3] obtain a high-probability regret bound using the biased surrogate loss

$$
\langle x , \widehat { \ell } _ { t } \rangle - \eta x ^ { \top } M _ { t } ^ { - 1 } x ,
$$

where $\eta > 0$ is the learning rate,

$$
M _ { t } = \mathbb { E } _ { X \sim p _ { t } } [ X X ^ { \top } ]
$$

is the second-moment matrix of the sampling distribution $p _ { t }$ , and

$$
\widehat { \ell _ { t } } : = M _ { t } ^ { - 1 } X _ { t } \langle X _ { t } , \ell _ { t } \rangle
$$

is the loss estimate constructed after drawing $X _ { t } \sim p _ { t }$ and observing the scalar loss $\langle X _ { t } , \ell _ { t } \rangle$

The main obstacle to an eficient implementation of this approach is that the leverage score is quadratic in x. Even if we start from a weighted m-set distribution and we insert this term directly into an action-level multiplicative-weights update, the updated distribution need not remain in the weighted m-set family and, consequently, it may require one weight for each of the $K = { \binom { d } { m } }$ actions.

Our first step is to replace the leverage score with an afine upper bound. Starting from the covariance inequality of Cesari and Colomboni [10, Corollary 1.3], we derive

$$
x ^ { \top } M _ { t } ^ { - 1 } x \leq \phi _ { t } ( x ) : = 1 + 2 \sum _ { i = 1 } ^ { d } \frac { ( x _ { i } - \mu _ { t , i } ) ^ { 2 } } { \mu _ { t , i } ( 1 - \mu _ { t , i } ) } .\tag{3}
$$

Although the expression defining $\phi _ { t }$ appears quadratic, it is afine on $\mathcal { X }$ . Indeed, every $x \in \mathcal { X }$ satisfies $x _ { i } ^ { 2 } = x _ { i }$ and $\textstyle \sum _ { i = 1 } ^ { d } x _ { i } = m$ . For later convenience, we scale this afine majorant by four and write

$$
c _ { t } ^ { \top } x = 4 \phi _ { t } ( x ) \qquad \mathrm { f o r ~ e v e r y ~ } x \in \mathcal { X } .
$$

We then define the surrogate loss

$$
Z _ { t } ( x ) : = \langle x , \widehat { \ell _ { t } } \rangle - \eta c _ { t } ^ { \top } x .\tag{4}
$$

The correction in Equation (4) plays the same role as the quadratic correction used by Zimmert and Lattimore [29, Algorithm 3], but remains afine in x. Consequently, the multiplicative-weights step maps a weighted m-set distribution to another weighted m-set distribution.

A further dificulty is that the afine leverage majorant can become arbitrarily large when a marginal probability approaches 0 or 1. Consequently, the estimated action losses $\langle x , \widehat { \ell } _ { t } \rangle$ can have arbitrarily large one-step magnitude and conditional second moment, preventing both the secondorder exponential-weights analysis and the required high-probability concentration arguments. To obtain the uniform leverage bound needed in the analysis, we project the updates toward the set $\mathcal { P } _ { \lambda }$ introduced in Section 2.

More precisely, the exact constrained OMD iterate is the KL projection of the unprojected exponential-weights distribution $\widetilde { p } _ { t + 1 }$ onto $\mathcal { P } _ { \lambda }$ . Our polynomial-time approximation is instead guaranteed to belong to $\mathcal { P } _ { \lambda / 2 }$ , where the extra slack serves to accommodate the finite accuracy of the projection procedure. The approximate KL update also preserves the weighted m-set family, as shown in Lemma 11. Hence, every iterate can be represented as $p _ { \theta _ { t } }$ using the d parameters

$$
\theta _ { t , 1 } , \ldots , \theta _ { t , d } ,
$$

rather than one weight for each of the K actions.

The comparator in the OMD analysis must also belong to $\mathcal { P } _ { \lambda }$ . However, the distribution $\delta _ { x } ,$ which puts all its probability on $x ,$ has marginal probabilities in {0, 1} and does not belong to ${ \mathcal P } _ { \lambda }$ We therefore use the smoothed comparator

$$
q _ { x } : = ( 1 - \lambda ) \delta _ { x } + \lambda U ,\tag{5}
$$

where $U$ is the uniform distribution over $\mathcal { X }$ . By construction, $q _ { x } \in { \mathcal { P } } _ { \lambda }$ . Since the losses are action-normalized, using $q _ { x }$ in place of $\delta _ { x }$ adds at most 2λT to the regret.

The remaining analytical challenge is to control the estimation error under the smoothed comparator $q _ { x }$ , namely,

$$
\sum _ { t = 1 } ^ { T } \Big ( \mathbb { E } _ { X \sim q _ { x } } [ \langle X , \widehat { \ell } _ { t } \rangle ] - \mathbb { E } _ { X \sim q _ { x } } [ \langle X , \ell _ { t } \rangle ] \Big ) .
$$

Applying a concentration bound to this estimation error introduces the positive term

$$
\frac { \eta } { 4 } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { X \sim q _ { x } } [ c _ { t } ^ { \top } X ] .
$$

Controlling this term separately would not give the desired regret bound. However, the afine correction in Equation (4) contributes

$$
- \eta \sum _ { t = 1 } ^ { T } \mathbb { E } _ { X \sim q _ { x } } [ c _ { t } ^ { \top } X ]
$$

to the OMD bound. Since $c _ { t } ^ { \top } X = 4 \phi _ { t } ( X ) \geq 0$ , this negative contribution dominates the positive concentration term. Their sum is therefore non-positive and may be dropped from the regret upper bound.

Finally, the elementary-symmetric-polynomial recurrence of Chen and Liu [11, Section $^ { 2 , }$ Method 2] allows us to compute the marginal probabilities and second-moment matrix of $p _ { t }$ in polynomial time. The associated conditional-Bernoulli procedure gives an exact sample from $p _ { t }$ without enumerating $x ;$ see Chen and Liu [11, Section 4, Procedure 3], as well as the alternative backward-path construction in Procedure 5. This sampling rule is also the $L = \mathrm { d i a g } ( w )$ specialization of Kulesza and Taskar [24, Section 3.1 and Algorithm 2]. Together with the fact that both the afine update and the approximate KL update preserve the family of weighted m-set distributions, these routines establish the polynomial-time implementation in Proposition 1.

## 4 Related work

Optimal design and finite-action linear bandits. Kiefer [21] introduced the general framework of approximate optimal design, and Kiefer and Wolfowitz [22] proved the equivalence between $D _ { - }$ - and G-optimality. For a finite action set $\mathcal { A } \subset \mathbb { R } ^ { d }$ , Bubeck et al. [8, Theorem 4] combined exponential weights with John’s exploration and obtained $O ( \sqrt { d T \log | \mathcal { A } | } )$ expected regret under action-normalized losses. For general convex action sets, Abernethy et al. [2, Theorem 1] used self-concordant barriers to obtain an eficient expected-regret bound of $\widetilde { O } ( d ^ { 3 / 2 } \sqrt { T } )$

The first general high-probability bound with $\sqrt { T }$ dependence against adaptive adversaries had a larger dependence on the dimension. Bartlett et al. [5, Theorem 1] obtained $\widetilde { O } ( d ^ { 3 / 2 } \sqrt { T } )$ regret. Abernethy and Rakhlin [1, Theorem 4 and Section 5.4] then gave a general high-probability reduction. Its eficient implementation requires afine confidence bounds tailored to the geometry of the action set. Later, Lee et al. [25, Theorem $3 . 1 ]$ gave the first general polynomial-time highprobability algorithm, with worst-case regret $\widetilde { O } ( d ^ { 7 / 2 } \sqrt { T } )$ ). Zimmert and Lattimore [29, Theorem 4] improved this rate to $\widetilde { O } ( d ^ { 2 } \sqrt { T } )$

For a finite action set A, Zimmert and Lattimore [29, Theorem 6 and Algorithm 3] also combined EXP3 with Kiefer–Wolfowitz exploration and obtained $O \left( \sqrt { d T \log ( | \mathcal { A } | / \delta } ) \right)$ regret with high probability. We use this regret bound as our statistical benchmark. When applied to our action set $\mathcal { X } = \mathcal { X } _ { d , m }$ , a direct implementation maintains $K = { \binom { d } { m } }$ weights. However, K can be exponential in d.

Combinatorial optimization with bandit feedback. Early adversarial bandit algorithms for structured action sets were given by McMahan and Blum [27] and Awerbuch and Kleinberg [4]. Cesa-Bianchi and Lugosi [9] subsequently introduced ComBand, a general framework for adversarial combinatorial bandits. This line of work was subsequently developed by Audibert et al. [3, Section 4, after Theorem 5], who considered full-information, semi-bandit, and full-bandit feedback and established several lower bounds. In particular, they conjectured that the minimax regret in the bandit setting should scale as $m \sqrt { d T }$ , where each action selects m out of d items and T is the time horizon. On the computational side, Combes et al. [14, Theorem 6 and 7] later introduced CombEXP, improving the computational complexity for some combinatorial action classes while preserving the existing regret scaling.

Cohen et al. [13, Theorems 1 and 5] disproved the conjecture of Audibert et al. [3]. Using correlated losses, they proved an expected-regret lower bound of $\Omega ( \sqrt { d m ^ { 3 } T / \log T } )$ for a multitask problem in which every action selects m items. Ito et al. [19, Theorem $2 ]$ later removed the $1 / { \sqrt { \log T } }$ factor using binary correlated losses. For m-set bandits, their result also describes the dependence on $d - m$ and on the time horizon, as discussed in Section 3.

Eficient algorithms for structured action sets. Several works avoid enumerating exponentially large action sets by using a compact representation or an optimization oracle. For expected or pseudo-regret, Hazan and Karnin [17, Corollary 35] use volumetric spanners, while Ito et al. [20, Theorem 1] use an optimization oracle. Both approaches run in polynomial time and achieve $\widetilde { O } ( \sqrt { T } )$ regret with polynomial dependence on the dimension.

Earlier eficient high-probability guarantees with $T ^ { 2 / 3 }$ dependence were obtained by Braun and Pokutta [7, Theorem 3.1], using a linear-optimization oracle over the action polytope, and by Sakaue et al. [28, Theorem 1], using a zero-suppressed binary decision diagram. These approaches are eficient when the required oracle or compact representation is available, but do not attain the $\sqrt { T }$ dependence achieved here.

The work most closely related to ours is Maiti et al. [26, Appendix E.6]. They represent the actions as paths in a DAG and obtain an eficient algorithm with a high-probability regret guarantee against adaptive adversaries. For $m { \mathrm { - s e t s } }$ , their algorithm achieves $\widetilde { O } ( d \sqrt { m T } )$ regret, and they ask whether $\widetilde { O } ( \sqrt { d m T } )$ can be achieved eficiently. Our result answers this question with a polynomial-time algorithm.

Finally, Kontogiannis et al. [23, Theorem 3.2 and Appendix G.5] show that their kernelized payof-based framework applies to m-sets and yields ${ \widetilde O } ( d ^ { 2 / 3 } m ^ { 4 / 3 } T ^ { 2 / 3 } )$ high-probability regret. Our result improves the dependence on the horizon to $\sqrt { T }$

Weighted m-set distributions. The weighted m-set distribution introduced in Section 2 is also known as a conditional Bernoulli or rejective-sampling distribution [11, 12, 16]. Classical methods based on elementary symmetric polynomials allow us to compute its normalizing constant and marginal probabilities, and to sample from it eficiently [11, 24].

Beyond these computational properties, weighted m-set distributions have a covariance structure central to our analysis: Cesari and Colomboni [10, Corollary 1.3] prove exactly the halfnormalized covariance bound used below.

## 5 A near-optimal algorithm for m-set bandits

## 5.1 Afine upper bound on the leverage score

In this section, we derive the afine upper bound on the leverage score $x ^ { \top } M ^ { - 1 } x$ used by our algorithm. Complete proofs of the consequences derived here are in Section A. The covariance bound itself is imported from Cesari and Colomboni [10, Corollary 1.3] and is not reproved here.

Lemma 1 (Cesari–Colomboni covariance bound). Let $p \in \Delta ( \mathcal X )$ be a weighted m-set distribution with positive weights, where $1 \leq m \leq d - 1$ , and let Σ be its covariance matrix. For every $i \in [ d ]$ define $v _ { i } : = \Sigma _ { i i }$ . Then

$$
\Sigma \succeq \frac { 1 } { 2 } \left( D - \frac { v v ^ { \top } } { V } \right) ,
$$

where $\begin{array} { r } { v : = ( v _ { 1 } , \dotsc , v _ { d } ) ^ { \top } , D : = \operatorname { d i a g } ( v ) \ , a n d V : = \sum _ { i = 1 } ^ { d } v _ { i } } \end{array}$

Using Lemma 1, we prove the following bound on the Moore–Penrose pseudoinverse $\Sigma ^ { + }$ of Σ.

Lemma 2. Let $p \in \Delta ( \mathcal X )$ be a weighted m-set distribution with positive weights, where $1 \leq m \leq$ $d - 1$ . Let Σ be its covariance matrix, and define $v _ { i } : = \Sigma _ { i i }$ for every $i \in [ d ]$ . Then, ker $( \Sigma ) = \operatorname { s p a n } \{ { \bf 1 } \}$ , and, for every $y \in H _ { 0 } : = \{ z \in \mathbb { R } ^ { d } : \langle z , \mathbf { 1 } \rangle = 0 \}$ , we have

$$
y ^ { \top } \Sigma ^ { + } y \leq 2 \sum _ { i = 1 } ^ { d } { \frac { y _ { i } ^ { 2 } } { v _ { i } } } .
$$

Proof sketch. Let $P : = D - v v ^ { \top } / V$ . A direct calculation, given in Section $\mathrm { A . 1 }$ , shows that ker $( P ) =$ ke $\mathbf { \partial } \cdot ( \Sigma ) = \operatorname { s p a n } \{ \mathbf { 1 } \}$ . Therefore, both matrices are positive definite, and hence invertible, on $H _ { 0 }$ . On $H _ { 0 } ,$ the bound in Lemma 1 gives

$$
\Sigma \succeq \frac 1 2 P .
$$

Thus, on $H _ { 0 }$ , we have

$$
y ^ { \top } \Sigma ^ { + } y \leq 2 y ^ { \top } P ^ { + } y \qquad { \mathrm { f o r ~ e v e r y ~ } } y \in H _ { 0 } .\tag{6}
$$

We now relate $P ^ { + }$ to $D ^ { - 1 }$ . Since $y \in H _ { 0 }$ , we have $\begin{array} { r } { \sum _ { i = 1 } ^ { d } y _ { i } = 0 } \end{array}$ . Therefore,

$$
P D ^ { - 1 } y = \left( D - { \frac { v v ^ { \top } } { V } } \right) D ^ { - 1 } y = y - { \frac { v } { V } } \sum _ { i = 1 } ^ { d } y _ { i } = y .\tag{7}
$$

We also have

$$
P ( P ^ { + } y ) = y ,\tag{8}
$$

because $P ^ { + }$ coincides with the inverse of $P$ on $H _ { 0 }$ . Subtracting Equation (8) from Equation (7) gives $P ( D ^ { - 1 } y - P ^ { + } y ) = 0$ . It follows that

$$
D ^ { - 1 } y - P ^ { + } y \in \ker ( P ) = \operatorname { s p a n } \{ { \bf 1 } \} .
$$

Since $y \in H _ { 0 }$ , this diference is orthogonal to y. Thus,

$$
y ^ { \top } P ^ { + } y = y ^ { \top } D ^ { - 1 } y = \sum _ { i = 1 } ^ { d } \frac { y _ { i } ^ { 2 } } { v _ { i } } .\tag{9}
$$

Combining (6) and (9) proves the result. See Section A.1 for the complete proof.

The next lemma relates the pseudoinverse $\Sigma ^ { + }$ of the covariance matrix of a weighted m-set distribution to the inverse $M ^ { - 1 }$ of its second-moment matrix.

Lemma 3. Let $p \in \Delta ( \mathcal X )$ be a weighted m-set distribution with positive weights and $1 \leq m \leq d - 1$ Then its second-moment matrix M is positive definite and, for every $y \in H _ { 0 }$ ，

$$
\boldsymbol { y } ^ { \top } \boldsymbol { M } ^ { - 1 } \boldsymbol { y } = \boldsymbol { y } ^ { \top } \boldsymbol { \Sigma } ^ { + } \boldsymbol { y } .
$$

Consequently, for every $x \in \mathcal { X }$

$$
x ^ { \top } M ^ { - 1 } x = 1 + ( x - \mu ) ^ { \top } \Sigma ^ { + } ( x - \mu ) ,
$$

where $\mu = \mu ( p )$ is the marginal vector of $p .$

Proof sketch. If $a ^ { \top } M a = 0$ , full support implies $a ^ { \top } x = 0$ for every $x \in \mathcal { X }$ . Comparing two actions that difer by swapping items i and $j$ forces $a _ { i } = a _ { j }$ . Hence $a = c \mathbf { 1 }$ , and then $0 = a ^ { \top } x = c m$ gives $c = 0$ . Thus $M \succ 0$ . Hence, $M \mathbf { 1 } = \mathbb { E } _ { X \sim p } [ X X ^ { \top } \mathbf { 1 } ] = m \mu$ . In addition, since M is invertible,

$$
M ^ { - 1 } \mu = \frac { \mathbf { 1 } } { m } .\tag{10}
$$

Now fix $y \in H _ { 0 }$ . By Equation (10),

$$
\mu ^ { \top } M ^ { - 1 } y = ( M ^ { - 1 } \mu ) ^ { \top } y = \frac { 1 } { m } \mathbf { 1 } ^ { \top } y = 0 .
$$

Therefore,

$$
\Sigma M ^ { - 1 } y = ( M - \mu \mu ^ { \top } ) M ^ { - 1 } y = y .
$$

Since $\Sigma ^ { + }$ is the inverse of $\Sigma$ on $H _ { 0 }$ , the vectors $M ^ { - 1 } y$ and $\Sigma ^ { + } y$ difer by an element of $\ker ( \Sigma ) =$ span{1}. This diference is orthogonal to $y ,$ and thus

$$
y ^ { \top } M ^ { - 1 } y = y ^ { \top } \Sigma ^ { + } y .\tag{11}
$$

Finally, fix $x \in \mathcal { X }$ . Since both $x$ and $\mu$ have coordinates that sum to $m ,$ we have $x - \mu \in H _ { 0 }$ Expanding $x = \mu + ( x - \mu )$ and using Equation (10) and Equation (11) gives

$$
x ^ { \top } M ^ { - 1 } x = 1 + ( x - \mu ) ^ { \top } \Sigma ^ { + } ( x - \mu ) .
$$

See Section A.2 for the complete proof.

We now combine Lemmas 2 and 3 to obtain the afine upper bound on the leverage score used by our algorithm.

Lemma 4. Let $p \in { \mathcal { P } } _ { \lambda }$ be a weighted m-set distribution, where $1 \leq m \leq d - 1$ , and let $\mu$ and M be its marginal vector and second-moment matrix. For every $x \in \mathcal { X }$ , define

$$
\phi _ { p } ( x ) : = 1 + 2 \sum _ { i = 1 } ^ { d } \frac { ( x _ { i } - \mu _ { i } ) ^ { 2 } } { \mu _ { i } ( 1 - \mu _ { i } ) } .
$$

Then, for every $x \in \mathcal { X } , x ^ { \top } M ^ { - 1 } x \leq \phi _ { p } ( x )$ . Moreover, $4 \phi _ { p } ( x ) = c _ { p } ^ { \top } x$ , where

$$
c _ { p , i } = 4 \left( \frac { 1 + 2 \sum _ { j = 1 } ^ { d } \frac { \mu _ { j } } { 1 - \mu _ { j } } } { m } + 2 \frac { 1 - 2 \mu _ { i } } { \mu _ { i } ( 1 - \mu _ { i } ) } \right) \qquad f o r \ e v e r y \ i \in [ d ] .
$$

Finally, $\mathbb { E } _ { X \sim p } [ \phi _ { p } ( X ) ] = 2 d + 1$ , and, for every $x \in \mathcal { X } , 0 \leq \phi _ { p } ( x ) \leq 1 + 4 d / \lambda$

Proof sketch. Fix $x \in \mathcal { X }$ . Since both x and $\mu$ have coordinates that sum to m, we have $x - \mu \in H _ { 0 }$ Therefore, Lemmas 2 and 3 give

$$
x ^ { \top } M ^ { - 1 } x = 1 + ( x - \mu ) ^ { \top } \Sigma ^ { + } ( x - \mu ) \leq 1 + 2 \sum _ { i = 1 } ^ { d } \frac { ( x _ { i } - \mu _ { i } ) ^ { 2 } } { \mu _ { i } ( 1 - \mu _ { i } ) } = \phi _ { p } ( x ) .
$$

Since $\mathbb { E } _ { X \sim p } [ ( X _ { i } - \mu _ { i } ) ^ { 2 } ] = \mu _ { i } ( 1 - \mu _ { i } )$ , we have $\mathbb { E } _ { X \sim p } [ \phi _ { p } ( X ) ] = 1 + 2 d$ . Moreover, $p \in { \mathcal { P } } _ { \lambda }$ implies $\mu _ { i } \geq \lambda r$ and $1 - \mu _ { i } \geq \lambda ( 1 - r )$ , where $r = m / d$ . Separating the m coordinates for which $x _ { i } = 1$ from the $d - m$ coordinates for which $x _ { i } = 0$ gives

$$
\sum _ { i = 1 } ^ { d } \frac { ( x _ { i } - \mu _ { i } ) ^ { 2 } } { \mu _ { i } ( 1 - \mu _ { i } ) } \leq \frac { m } { \lambda r } + \frac { d - m } { \lambda ( 1 - r ) } = \frac { 2 d } { \lambda } .
$$

Hence,

$$
0 \leq \phi _ { p } ( x ) \leq 1 + \frac { 4 d } { \lambda } .
$$

Finally, using $x _ { i } ^ { 2 } = x _ { i }$ , we obtain

$$
\phi _ { p } ( x ) = 1 + 2 \sum _ { j = 1 } ^ { d } \frac { \mu _ { j } } { 1 - \mu _ { j } } + 2 \sum _ { i = 1 } ^ { d } \frac { 1 - 2 \mu _ { i } } { \mu _ { i } ( 1 - \mu _ { i } ) } x _ { i } .
$$

Since $\textstyle \sum _ { i } x _ { i } = m$ , we can write the constant term as a linear function of x. Multiplying the resulting afine representation by four gives $4 \phi _ { p } ( x ) = c _ { p } ^ { \top } x$ , with $c _ { p }$ defined in the statement. See Section $\mathrm { A . 3 }$ for the complete proof. □

## 5.2 Algorithm and high-probability regret

In this section, we present Algorithm 1 and prove its high-probability regret guarantee. At each round $t ,$ the learner draws an action $X _ { t }$ from a weighted m-set distribution $p _ { t } \in \mathcal { P } _ { \lambda / 2 }$ and observes only the scalar loss $\langle X _ { t } , \ell _ { t } \rangle$ . It uses this observation to construct the KW loss estimate $\widehat { \ell } _ { t }$

At round t, we set $\phi _ { t } : = \phi _ { p _ { t } }$ and $c _ { t } : = c _ { p _ { t } }$ , where $\phi _ { p }$ and $c _ { p }$ are defined in Lemma 4. The learner forms a biased surrogate loss by subtracting the afine correction $\eta c _ { t } ^ { \top }$ x from $\langle x , \widehat { \ell } _ { t } \rangle$ . Finally, the learner computes an approximate entropic OMD update toward $\mathcal { P } _ { \lambda }$ (defined in Eq. (2)). The update is accurate enough for the OMD analysis and returns a distribution in $\mathcal { P } _ { \lambda / 2 }$ , so its marginal probabilities stay away from 0 and 1. We state the resulting regret bound in Theorem 1.

```latex
Algorithm 1 Afine KW–OMD with approximate entropic updates toward $\mathcal { P } _ { \lambda }$
Require: $\overline { { T \geq 1 , \delta \in ( 0 , 1 ) } }$
1: Set $\eta $ min $\left\{ { \frac { 1 } { 2 5 6 d } } , { \sqrt { \frac { \log ( 1 2 K / \delta ) } { 3 2 0 d T } } } \right\}$
2: Set λ ← 128ηd and $\varepsilon _ { \mathrm { p } } \gets \eta / T$
3: Initialize $\theta _ { 1 }  0$ and $p _ { 1 }  p _ { \theta _ { 1 } } = U$ ▷ U is a uniform distribution
4: for $t = 1 , \dots , T$ do
5: Compute $\mu _ { t }  \mu ( p _ { t } ) , M _ { t }  M ( p _ { t } ) = \mathbb { E } _ { X \sim p _ { t } } [ X X ^ { \top } ]$
6: Set $\phi _ { t }  \phi _ { p _ { t } }$ and $c _ { t } \gets c _ { p _ { t } }$ , so that, for every $x \in { \mathcal { X } } .$
$\phi _ { t } ( x )  1 + 2 \sum _ { i = 1 } ^ { d } \frac { ( x _ { i } - \mu _ { t , i } ) ^ { 2 } } { \mu _ { t , i } ( 1 - \mu _ { t , i } ) }$ and $c _ { t } ^ { \top } x = 4 \phi _ { t } ( x )$
7: Draw $X _ { t } \sim p _ { t }$
8: Observe $Y _ { t } = \langle X _ { t } , \ell _ { t } \rangle$
9: Compute $\widehat { \ell } _ { t } \gets M _ { t } ^ { - 1 } X _ { t } Y _ { t }$
10: Define $Z _ { t } ( x ) \gets \langle x , \widehat { \ell } _ { t } \rangle - \eta c _ { t } ^ { \top } x ,$ for every $x \in \mathcal { X }$
11: Set $z _ { t } \gets \widehat { \ell } _ { t } - \eta c _ { t } , \widetilde { \theta } _ { t + 1 } \gets \theta _ { t } - \eta z _ { t } .$ and $\widetilde { p } _ { t + 1 } \gets p _ { \widetilde { \theta } _ { t + 1 } }$
12: Apply a fixed deterministic implementation of the procedure in Lemma 11 to obtain $p _ { t + 1 } =$
$p _ { \theta _ { t + 1 } }$ satisfying
$p _ { t + 1 } \in \mathcal { P } _ { \lambda / 2 }$
and, simultaneously for every $q \in { \mathcal { P } } _ { \lambda }$
$\begin{array} { r } { \mathrm { K L } ( q \| p _ { t + 1 } ) + \mathrm { K L } ( p _ { t + 1 } \| \widetilde { p } _ { t + 1 } ) \leq \mathrm { K L } ( q \| \widetilde { p } _ { t + 1 } ) + \varepsilon _ { \mathrm { p } } . } \end{array}$
13: end for
```

Theorem 1 (High-probability KW bound). There exists a universal constant $C > 0$ such that the following holds. Let $T \geq 1$ be an integer, let $\delta \in ( 0 , 1 )$ , and assume $1 \leq m \leq d - 1$ . Run Algorithm 1. Then, with probability at least $1 - \delta _ { i }$

$$
R _ { T } \leq C \sqrt { d T \left( \log K + \log \frac { 1 } { \delta } \right) } ,
$$

where $K = { \binom { d } { m } }$ . One may take $C = 1 6 0$

Proof sketch. For every x $\in { \mathcal { X } }$ , define the smoothed comparator

$$
q _ { x } : = ( 1 - \lambda ) \delta _ { x } + \lambda U ,
$$

where $\delta _ { x }$ puts all its probability on x and $U$ is the uniform distribution over $\mathcal { X }$ . By construction, $q _ { x } \in { \mathcal { P } } _ { \lambda }$ . The concentration bounds below hold simultaneously for every $x \in \mathcal { X }$ , so we can then choose x to be a best action.

By adding and subtracting the expected losses under $p _ { t }$ and $q _ { x }$ , and applying Lemma 9, we

obtain, with high probability,

$$
R _ { T } \leq \underbrace { \sum _ { t = 1 } ^ { T } \left( \mathbb { E } _ { X \sim p _ { t } } [ \langle X , \ell _ { t } \rangle ] - \mathbb { E } _ { X \sim q _ { x } } [ \langle X , \ell _ { t } \rangle ] \right) } _ { = : ( \star ) } + O \left( \sqrt { T \log \frac { 1 } { \delta } } \right) + O ( \lambda T ) .\tag{12}
$$

The square-root term controls the diference between the learner’s realized and expected losses. The term $O ( \lambda T )$ is the cost of using $q _ { x }$ in place of x. It remains to bound (⋆). Adding and subtracting the expectations of the estimated losses gives

$$
\begin{array} { r l } & { ( \star ) = \displaystyle \sum _ { t = 1 } ^ { T } \Big ( \mathbb { E } _ { X \sim p _ { t } } [ \langle X , \ell _ { t } \rangle ] - \mathbb { E } _ { X \sim p _ { t } } [ \langle X , \widehat { \ell } _ { t } \rangle ] \Big ) } \\ & { \qquad + \displaystyle \sum _ { t = 1 } ^ { T } \Big ( \mathbb { E } _ { X \sim p _ { t } } [ \langle X , \widehat { \ell } _ { t } \rangle ] - \mathbb { E } _ { X \sim q _ { x } } [ \langle X , \widehat { \ell } _ { t } \rangle ] \Big ) + \displaystyle \sum _ { t = 1 } ^ { T } \Big ( \mathbb { E } _ { X \sim q _ { x } } [ \langle X , \widehat { \ell } _ { t } \rangle ] - \mathbb { E } _ { X \sim q _ { x } } [ \langle X , \ell _ { t } \rangle ] \Big ) . } \end{array}\tag{13}
$$

The first and third terms in (13) can be bounded by applying Lemma 7 under $p _ { t }$ and $q _ { x }$ , respectively. To bound the second term, recall the surrogate loss

$$
Z _ { t } ( x ) : = \langle x , { \widehat { \ell } } _ { t } \rangle - \eta c _ { t } ^ { \top } x , \qquad x \in \mathcal { X } .
$$

Since

$$
\langle x , \widehat { \ell } _ { t } \rangle = Z _ { t } ( x ) + \eta c _ { t } ^ { \top } x ,
$$

we can apply Lemma 6 and Lemma 8 to the surrogate losses $Z _ { t }$ . Moreover, Lemma 4 gives

$$
\mathbb { E } _ { X \sim p _ { t } } [ c _ { t } ^ { \top } X ] = 4 ( 2 d + 1 ) = O ( d ) .
$$

Combining these results, we have

$$
( \star ) \leq O \left( \frac { \log K + \log ( 1 / \delta ) } { \eta } + \eta d T + 1 \right) + \underbrace { \left( - \eta + \frac { \eta } { 4 } \right) \sum _ { t = 1 } ^ { T } \mathbb E _ { X \sim q _ { x } } [ c _ { t } ^ { \top } X ] } _ { = : ( \star \star ) \leq 0 } .\tag{14}
$$

The coeficient $- \eta$ in $( \star \star )$ comes from the afine correction $- \eta c _ { t } ^ { \top } x$ in $Z _ { t }$ , while the coeficient $\eta / 4$ comes from the concentration bound for the third term in (13). Since

$$
c _ { t } ^ { \top } X = 4 \phi _ { t } ( X ) \geq 0 ,
$$

the term $( \star \star )$ is nonpositive. The additive constant in (14) is the accumulated error of the approximate KL updates: by the choice $\varepsilon _ { \mathrm { p } } = \eta / T$ , its total contribution is $T \varepsilon _ { \mathrm { p } } / \eta = 1$ . Substituting (14) into (12) gives

$$
R _ { T } \leq { \cal O } \left( \frac { \log K + \log ( 1 / \delta ) } { \eta } + \eta d T + 1 + \lambda T + \sqrt { T \log \frac 1 \delta } \right) .\tag{15}
$$

Finally, substituting $\lambda = 1 2 8 \eta d$ and the value of $\eta ,$ and combining the resulting high-probability estimate with the deterministic bound $R _ { T } \ \leq \ 2 T$ , gives the stated result with $C = 1 6 0$ . The complete two-case calculation is given in Section B.1. □

Let $s : = \operatorname* { m i n } \{ m , d - m \}$ . Since log $K \leq s \log \left( e d / s \right)$ , the bound in Theorem 1 is $\widetilde { O } ( \sqrt { d s T } )$ . In particular, when $m \leq d / 2$ , it is $\widetilde { O } ( \sqrt { d m T } )$

## 5.3 Polynomial-time implementation

We now show that Algorithm 1 can be implemented in polynomial time without enumerating $\mathcal { X } .$

The key idea is to maintain $p _ { t }$ in the weighted m-set form $p _ { \theta _ { t } }$ , with $\theta _ { t } \in \mathbb { R } ^ { d }$ , thereby allowing all steps of the algorithm to be performed without enumerating $\mathcal { X } .$ . The convex problem used in the update is solved to the explicit inverse-polynomial objective accuracy established in Lemma 11.

Proposition 1. Algorithm 1 can be implemented in $T \cdot \mathrm { p o l y } ( d , m , \log T )$ time and poly(d, m) space, without enumerating X . At every round, $p _ { t }$ is a weighted m-set distribution $p _ { \theta _ { t } }$ represented by the d parameters $\theta _ { t , 1 } , \ldots , \theta _ { t , d }$

Proof sketch. We observe that if $p _ { t }$ has a weighted m-set representation, its moments can be computed and a sample can be drawn in polynomial time using the recurrences in Lemma 12.

The strategy update in Algorithm 1 can be decomposed into two steps. First, define the exponential update

$$
\widetilde { p } _ { t + 1 } ( x ) \propto p _ { t } ( x ) \exp \bigl ( - \eta Z _ { t } ( x ) \bigr ) .
$$

The exact update toward which we compute is the KL projection

$$
\begin{array} { r } { p _ { t + 1 } ^ { \star } \in \underset { p \in \mathcal { P } _ { \lambda } } { \arg \operatorname* { m i n } } \mathrm { K L } ( p \| \widetilde { p } _ { t + 1 } ) . } \end{array}
$$

Since $Z _ { t }$ is afine, the exponential update only changes the d parameters, and $\widetilde { p } _ { t + 1 }$ remains a weighted m-set distribution. The exact projection is characterized by a convex problem in 2d variables. Solving this problem to the objective accuracy in Lemma 11 returns a weighted m-set distribution $p _ { t + 1 } \in \mathcal { P } _ { \lambda / 2 }$ and introduces at most $\varepsilon _ { \mathrm { p } } = \eta / T$ error in the KL inequality used by OMD. The required objective accuracy is inverse-polynomial, and the ellipsoid method obtains it using polynomially many evaluations of the objective and its gradient. Since the initial distribution $p _ { 1 }$ is uniform and admits a weighted m-set representation, this representation is preserved at every round. Thus, all steps of the algorithm can be performed in polynomial time without enumerating X. The complete proof is given in Section C.3. □

## 6 Conclusion and future work

We introduced an algorithm for adversarial m-set bandits whose high-probability regret matches the finite-action EXP3–KW rate up to logarithmic factors, while running in polynomial time without enumerating the action set.

Our results leave two main questions open. First, it remains open to determine whether the log K term in the leading bound can be replaced by

$$
s : = \operatorname* { m i n } \{ m , d - m \} ,
$$

yielding $O ( \sqrt { d s T } )$ realized regret at constant confidence, or whether an additional logarithmic factor is unavoidable against adaptive non-anticipating adversaries. We remark, to the best of our knowledge, this issue is open already for ordinary adversarial multi-armed bandits, corresponding to $m = 1$ . Second, it would be interesting to determine whether our approach extends beyond m-sets. A natural test case is that of matroid-base action sets, since m-sets are precisely the bases of a uniform matroid. Although afine exponential-weights updates still preserve weighted matroidbase distributions, our analysis also requires an eficiently computable afine leverage majorant with controlled expectation and range, as well as eficient moment and KL-projection routines. These properties do not follow from our present covariance-based construction, so extending the nearoptimal high-probability guarantee to broader matroid classes would require additional structural and computational ideas.

## Appendix

## A Proofs from Section 5.1

## A.1 Proof of Lemma 2

Lemma 2. Let $p \in \Delta ( \mathcal X )$ be a weighted m-set distribution with positive weights, where $1 \leq m \leq$ $d - 1$ . Let Σ be its covariance matrix, and define $v _ { i } : = \Sigma _ { i i }$ for every $i \in [ d ]$ . Then, ker $( \Sigma ) = \operatorname { s p a n } \{ { \bf 1 } \}$ and, for every $y \in H _ { 0 } : = \{ z \in \mathbb { R } ^ { d } : \langle z , \mathbf { 1 } \rangle = 0 \}$ , we have

$$
y ^ { \top } \Sigma ^ { + } y \leq 2 \sum _ { i = 1 } ^ { d } { \frac { y _ { i } ^ { 2 } } { v _ { i } } } .
$$

Proof. Let

$$
P : = D - { \frac { v v ^ { \top } } { V } } .
$$

We first identify the kernels of P and Σ. Since p has positive weights, it has full support on X . Therefore,

$$
z ^ { \top } \Sigma z = \operatorname { V a r } _ { X \sim p } ( \langle z , X \rangle ) = 0
$$

only if $\langle z , x \rangle$ is the same for every $x \in \mathcal { X }$

Fix $i \neq j$ . Since $1 \leq m \leq d - 1$ , we can choose two actions that difer only by exchanging item i with item $j .$ The equality of their inner products with z gives $z _ { i } = z _ { j }$ . Thus, all the coordinates of z are equal. Conversely, $\langle \mathbf { 1 } , X \rangle = m$ for every $X \in { \mathcal { X } }$ , so $\Sigma \mathbf { 1 } = 0$ . Hence,

$$
\ker ( \Sigma ) = \operatorname { s p a n } \{ { \bf 1 } \} .\tag{16}
$$

Moreover, for every $z \in \mathbb { R } ^ { d }$

$$
z ^ { \top } P z = \sum _ { i = 1 } ^ { d } v _ { i } z _ { i } ^ { 2 } - \frac { \left( \sum _ { i = 1 } ^ { d } v _ { i } z _ { i } \right) ^ { 2 } } { V } = \sum _ { i = 1 } ^ { d } v _ { i } ( z _ { i } - \bar { z } _ { v } ) ^ { 2 } ,
$$

where

$$
\bar { z } _ { v } : = \frac { \sum _ { i = 1 } ^ { d } v _ { i } z _ { i } } { V } .
$$

Full support also gives

$$
v _ { i } = \mathrm { V a r } ( X _ { i } ) = \mu _ { i } ( 1 - \mu _ { i } ) > 0 \qquad \mathrm { f o r ~ e v e r y ~ } i \in [ d ] .
$$

It follows that $z ^ { \top } P z = 0$ if and only if all the coordinates of z are equal. Therefore,

$$
\ker ( P ) = \operatorname { s p a n } \{ { \bf 1 } \} .\tag{17}
$$

By (16) and (17), the pseudoinverse order property stated in the preliminaries applies to the bound in Lemma 1. Hence, for every $y \in H _ { 0 }$ 2

$$
y ^ { \top } \Sigma ^ { + } y \leq 2 y ^ { \top } P ^ { + } y .\tag{18}
$$

It remains to compute $y ^ { \top } P ^ { + } y$ . Since $D = \deg ( v )$ , we have $v ^ { \top } D ^ { - 1 } = { \bf 1 } ^ { \top }$ . Thus, for every $y \in H _ { 0 }$ ，

$$
P D ^ { - 1 } y = y - \frac { v } { V } \mathbf { 1 } ^ { \top } y = y .\tag{19}
$$

Moreover, range $\begin{array} { r } { \langle \boldsymbol { P } \rangle = H _ { 0 } , } \end{array}$ so

$$
P P ^ { + } y = y .\tag{20}
$$

Equations (19) and (20) show that $D ^ { - 1 } y$ and $P ^ { + } y$ both solve the equation $P z = y$ . Therefore,

$$
D ^ { - 1 } y - P ^ { + } y \in \ker ( P ) = \operatorname { s p a n } \{ { \bf 1 } \} .
$$

Since $y \in H _ { 0 }$ , it is orthogonal to this diference. Consequently,

$$
y ^ { \top } P ^ { + } y = y ^ { \top } D ^ { - 1 } y = \sum _ { i = 1 } ^ { d } \frac { y _ { i } ^ { 2 } } { v _ { i } } .\tag{21}
$$

Combining (18) and (21) proves the result.

## A.2 Proof of Lemma 3

Lemma 3. Let $p \in \Delta ( \mathcal X )$ be a weighted m-set distribution with positive weights and $1 \leq m \leq d - 1$ Then its second-moment matrix M is positive definite and, for every $y \in H _ { 0 }$ 2

$$
y ^ { \top } M ^ { - 1 } y = y ^ { \top } \Sigma ^ { + } y .
$$

Consequently, for every $x \in \mathcal { X }$

$$
x ^ { \top } M ^ { - 1 } x = 1 + ( x - \mu ) ^ { \top } \Sigma ^ { + } ( x - \mu ) ,
$$

where $\mu = \mu ( p )$ is the marginal vector of $p .$

Proof. We first show that M is positive definite. Suppose that $a ^ { \top } M a = 0$ . Since

$$
\begin{array} { r } { \boldsymbol { a } ^ { \top } M \boldsymbol { a } = \mathbb { E } _ { X \sim p } \big [ ( \boldsymbol { a } ^ { \top } X ) ^ { 2 } \big ] , } \end{array}
$$

we have $a ^ { \top } x = 0$ for every $x \in \mathcal { X }$ , because p has full support.

Fix $i \neq j$ . Since $1 \leq m \leq d - 1$ , we can choose two actions that difer only by exchanging item i with item $j .$ Comparing their inner products with a gives $a _ { i } = a _ { j }$ . Thus, $a = c \mathbf { 1 }$ for some $c \in \mathbb { R }$ Since every $x \in \mathcal { X }$ has m nonzero coordinates,

$$
0 = a ^ { \top } x = c m .
$$

Therefore, $c = 0 ,$ and hence $a = 0$ . This proves that $M \succ 0$ . By Lemma 2, ker $( \Sigma ) = \operatorname { s p a n } \{ { \bf 1 } \}$ Since Σ is symmetric, this also gives range $ { \left( \Sigma \right) } = H _ { 0 }$ . Every $X \in { \mathcal { X } }$ satisfies $\langle X , \mathbf { 1 } \rangle = m$ . Therefore,

$$
M \mathbf { 1 } = \mathbb { E } _ { X \sim p } \bigl [ X \langle X , \mathbf { 1 } \rangle \bigr ] = m \mu .
$$

Since M is invertible,

$$
M ^ { - 1 } \mu = \frac { 1 } { m } { \bf 1 } .\tag{22}
$$

Now fix $y \in H _ { 0 }$ and let $z : = M ^ { - 1 } y$ . By (22),

$$
\mu ^ { \top } z = ( M ^ { - 1 } \mu ) ^ { \top } y = \frac { 1 } { m } \mathbf { 1 } ^ { \top } y = 0 .
$$

It follows that

$$
\begin{array} { r } { \Sigma { z } = ( M - \mu \mu ^ { \top } ) z = y . } \end{array}
$$

Moreover, $\Sigma ^ { + } y$ also solves the equation $\Sigma u = y$ . Therefore,

$$
z - \Sigma ^ { + } y \in \ker ( \Sigma ) = \operatorname { s p a n } \{ { \bf 1 } \} .
$$

Since $y \in H _ { 0 }$ , it is orthogonal to this diference. Hence,

$$
\boldsymbol { y } ^ { \top } \boldsymbol { M } ^ { - 1 } \boldsymbol { y } = \boldsymbol { y } ^ { \top } \boldsymbol { \Sigma } ^ { + } \boldsymbol { y } .\tag{23}
$$

Finally, fix $x \in \mathcal { X }$ and let $y : = x - \mu$ . Since $\mathbf { 1 } ^ { \top } x = \mathbf { 1 } ^ { \top } \boldsymbol { \mu } = m$ , we have $y \in H _ { 0 }$ . Expanding $x = \mu + y$ gives

$$
\begin{array} { c } { { x ^ { \top } M ^ { - 1 } x = \mu ^ { \top } M ^ { - 1 } \mu + 2 y ^ { \top } M ^ { - 1 } \mu + y ^ { \top } M ^ { - 1 } y } } \\ { { = 1 + y ^ { \top } \Sigma ^ { + } y , } } \end{array}
$$

where the second equality follows from (22) and (23). Substituting $y = x - \mu$ proves that

$$
x ^ { \top } M ^ { - 1 } x = 1 + ( x - \mu ) ^ { \top } \Sigma ^ { + } ( x - \mu ) .
$$

This concludes the proof.

## A.3 Proof of Lemma 4

Lemma 4. Let $p \in { \mathcal { P } } _ { \lambda }$ be a weighted m-set distribution, where $1 \leq m \leq d - 1$ , and let $\mu$ and M be its marginal vector and second-moment matrix. For every $x \in \mathcal { X }$ , define

$$
\phi _ { p } ( x ) : = 1 + 2 \sum _ { i = 1 } ^ { d } \frac { ( x _ { i } - \mu _ { i } ) ^ { 2 } } { \mu _ { i } ( 1 - \mu _ { i } ) } .
$$

Then, for every $x \in \mathcal { X } , x ^ { \top } M ^ { - 1 } x \leq \phi _ { p } ( x )$ . Moreover, $4 \phi _ { p } ( x ) = c _ { p } ^ { \top } x$ , where

$$
c _ { p , i } = 4 \left( \frac { 1 + 2 \sum _ { j = 1 } ^ { d } \frac { \mu _ { j } } { 1 - \mu _ { j } } } { m } + 2 \frac { 1 - 2 \mu _ { i } } { \mu _ { i } ( 1 - \mu _ { i } ) } \right) \qquad f o r \ e v e r y \ i \in [ d ] .
$$

Finally, $\mathbb { E } _ { X \sim p } [ \phi _ { p } ( X ) ] = 2 d + 1$ , and, for every $x \in \mathcal { X } , 0 \leq \phi _ { p } ( x ) \leq 1 + 4 d / \lambda$

Proof. For $x \in \mathcal { X }$ , put $y = x - \mu \in H _ { 0 }$ . By Lemma $3 .$

$$
x ^ { \top } M ^ { - 1 } x = 1 + y ^ { \top } \Sigma ^ { + } y .
$$

Lemma 2 gives

$$
y ^ { \top } \Sigma ^ { + } y \leq 2 \sum _ { i } { \frac { y _ { i } ^ { 2 } } { \mu _ { i } ( 1 - \mu _ { i } ) } } ,
$$

and hence $x ^ { \top } M ^ { - 1 } x \leq \phi _ { p } ( x )$ . Taking expectations and using $\mathbb { E } [ ( X _ { i } - \mu _ { i } ) ^ { 2 } ] = \mu _ { i } ( 1 - \mu _ { i } )$ yields

$$
\mathbb { E } _ { X \sim p } [ \phi _ { p } ( X ) ] = 2 d + 1 .
$$

For the range bound, if $x _ { i } = 1$ , then

$$
2 \frac { ( x _ { i } - \mu _ { i } ) ^ { 2 } } { \mu _ { i } ( 1 - \mu _ { i } ) } = 2 \frac { 1 - \mu _ { i } } { \mu _ { i } } \leq \frac { 2 } { \lambda r } ;
$$

if $x _ { i } = 0$ , then

$$
2 { \frac { ( x _ { i } - \mu _ { i } ) ^ { 2 } } { \mu _ { i } ( 1 - \mu _ { i } ) } } = 2 { \frac { \mu _ { i } } { 1 - \mu _ { i } } } \leq { \frac { 2 } { \lambda ( 1 - r ) } } .
$$

Since x has m ones and $d - m$ zeros,

$$
\phi _ { p } ( x ) \leq 1 + \frac { 2 m } { \lambda r } + \frac { 2 ( d - m ) } { \lambda ( 1 - r ) } = 1 + \frac { 4 d } { \lambda } .
$$

Finally, because $x _ { i } \in \{ 0 , 1 \}$ ,

$$
2 \frac { ( x _ { i } - \mu _ { i } ) ^ { 2 } } { \mu _ { i } ( 1 - \mu _ { i } ) } = 2 \frac { \mu _ { i } } { 1 - \mu _ { i } } + 2 x _ { i } \frac { 1 - 2 \mu _ { i } } { \mu _ { i } ( 1 - \mu _ { i } ) } .
$$

Thus

$$
\phi _ { p } ( x ) = 1 + 2 \sum _ { i } \frac { \mu _ { i } } { 1 - \mu _ { i } } + 2 \sum _ { i } x _ { i } \frac { 1 - 2 \mu _ { i } } { \mu _ { i } ( 1 - \mu _ { i } ) } .
$$

Distribute the constant term evenly over the selected coordinates using $\textstyle \sum _ { i } x _ { i } = m$ , and multiply the resulting afine representation by four, to obtain $4 \phi _ { p } ( x ) = c _ { p } ^ { \top } x$ □

## B Proofs from Section 5.2

Assumption 1. Throughout this appendix, $T \geq 1$ is an integer, $\delta \in ( 0 , 1 )$ ,

$$
0 < \eta \leq \frac { 1 } { 6 4 } , \qquad \lambda = 1 2 8 \eta d \leq \frac { 1 } { 2 } .
$$

The loss sequence $( \ell _ { t } ) _ { t = 1 } ^ { T }$ is predictable, that is, $\ell _ { t }$ is $\mathcal { F } _ { t - 1 }$ -measurable for every $t \in [ T ]$ , and satisfies the action-normalized condition in Section 2.

The initialization and the update in Algorithm 1 ensure that, at every round t, the sampling distribution $p _ { t }$ is a weighted m-set distribution with positive weights and $p _ { t } \in \mathcal { P } _ { \lambda / 2 }$ . Hence, the assumptions of Lemmas 1 and $\it 4$ hold at every round.

Recall from Algorithm 1 that

$$
\begin{array} { r } { \phi _ { t } : = \phi _ { p _ { t } } , \qquad c _ { t } : = c _ { p _ { t } } , \qquad c _ { t } ^ { \top } x = 4 \phi _ { t } ( x ) , \qquad Z _ { t } ( x ) : = \widehat { y } _ { t } ( x ) - \eta c _ { t } ^ { \top } x . } \end{array}
$$

We also write

$$
\widehat { y } _ { t } ( x ) : = \langle x , \widehat { \ell } _ { t } \rangle = x ^ { \top } M _ { t } ^ { - 1 } X _ { t } Y _ { t } .
$$

Set

$$
B : = 1 + { \frac { 8 d } { \lambda } } .
$$

Applying Lemma 4 with $\lambda / 2 ,$ , for every $x \in \mathcal { X }$ , we have

$$
0 \leq c _ { t } ^ { \top } x \leq 4 B , \qquad \mathbb { E } _ { X \sim p _ { t } } [ c _ { t } ^ { \top } X ] = 4 ( 2 d + 1 ) , \qquad x ^ { \top } M _ { t } ^ { - 1 } x \leq \frac { c _ { t } ^ { \top } x } { 4 } .
$$

Lemma 5. Fix $t \in [ T ]$ . Suppose that $0 < \eta \leq 1 / 6 4 , \lambda = 1 2 8 \eta d \leq 1 / 2$ , the loss at round t satisfies the action-normalized condition, and $p _ { t }$ is a weighted m-set distribution with positive weights in $\mathcal { P } _ { \lambda / 2 }$ . Then, for every $x \in \mathcal { X } , \eta | Z _ { t } ( x ) | \le 1 / 4$

Proof. Since $| Y _ { t } | \le 1$ , the Cauchy–Schwarz inequality in the norm induced by $M _ { t } ^ { - 1 }$ gives

$$
\left| \widehat { y } _ { t } ( x ) \right| = | x ^ { \top } M _ { t } ^ { - 1 } X _ { t } Y _ { t } | \leq \sqrt { x ^ { \top } M _ { t } ^ { - 1 } x } \sqrt { X _ { t } ^ { \top } M _ { t } ^ { - 1 } X _ { t } } .
$$

Applying Lemma 4 to $p _ { t } \in \mathcal { P } _ { \lambda / 2 }$ , both leverage scores in the last inequality are at most $\phi _ { t } ( x )$ and $\phi _ { t } ( X _ { t } )$ , respectively, and both ϕ<sub>t</sub>-values are at most

$$
B = 1 + { \frac { 8 d } { \lambda } } .
$$

Thus

$$
| { \widehat { y } } _ { t } ( x ) | \leq B .
$$

Also $0 \leq c _ { t } ^ { \top } x \leq 4 B$ . Hence

$$
| Z _ { t } ( x ) | = | \widehat { y } _ { t } ( x ) - \eta c _ { t } ^ { \top } x | \leq B + 4 \eta B .
$$

The parameter choice implies

$$
\eta B = \eta + \frac { 8 \eta d } { \lambda } = \eta + \frac { 1 } { 1 6 } \leq \frac { 5 } { 6 4 } .
$$

Therefore

$$
\eta | Z _ { t } ( x ) | \leq ( 1 + 4 \eta ) \eta B \leq \left( 1 + \frac { 1 } { 1 6 } \right) \frac { 5 } { 6 4 } = \frac { 8 5 } { 1 0 2 4 } < \frac { 1 } { 4 } ,
$$

where we used $\eta \leq 1 / 6 4$ . This proves the claim.

Lemma 6. Under Assumption 1, for every distribution $q \in { \mathcal { P } } _ { \lambda }$

$$
\sum _ { t = 1 } ^ { T } \left( \mathbb { E } _ { X \sim p _ { t } } [ Z _ { t } ( X ) ] - \mathbb { E } _ { X \sim q } [ Z _ { t } ( X ) ] \right) \leq \frac { \mathrm { K L } ( q \| p _ { 1 } ) } { \eta } + 1 + 2 \eta \sum _ { t = 1 } ^ { T } \mathbb { E } _ { X \sim p _ { t } } [ Z _ { t } ( X ) ^ { 2 } ] .
$$

Proof. Fix $t \in [ T ]$ and define the unprojected multiplicative-weights distribution

$$
\widetilde { p } _ { t + 1 } ( x ) : = \frac { p _ { t } ( x ) \exp ( - \eta Z _ { t } ( x ) ) } { \mathbb { E } _ { X \sim p _ { t } } [ \exp ( - \eta Z _ { t } ( X ) ) ] } , \qquad x \in \mathcal { X } .
$$

Since $Z _ { t } ( x ) = z _ { t } ^ { \top } x$ and $p _ { t } = p _ { \theta _ { t } }$ , the distribution above is precisely $p _ { \widetilde { \theta } _ { t + 1 } }$ from Algorithm 1. The guarantee in Algorithm 1 therefore gives, for every $q \in { \mathcal { P } } _ { \lambda }$

$$
\begin{array} { r } { \mathrm { K L } ( q \| p _ { t + 1 } ) \leq \mathrm { K L } ( q \| \widetilde { p } _ { t + 1 } ) + \varepsilon _ { \mathrm { p } } , } \end{array}\tag{24}
$$

where we dropped the nonnegative term $\mathrm { K L } ( p _ { t + 1 } | | \widetilde { p } _ { t + 1 } )$ . By the definition of $\widetilde { p } _ { t + 1 }$

$$
\mathrm { K L } ( q \| \widetilde { p } _ { t + 1 } ) = \mathrm { K L } ( q \| p _ { t } ) + \eta \mathbb { E } _ { X \sim q } [ Z _ { t } ( X ) ] + \log \mathbb { E } _ { X \sim p _ { t } } [ \exp ( - \eta Z _ { t } ( X ) ) ] .
$$

Combining this identity with (24) and rearranging gives

$$
\begin{array} { r l } & { \eta \left( \mathbb { E } _ { X \sim p _ { t } } [ Z _ { t } ( X ) ] - \mathbb { E } _ { X \sim q } [ Z _ { t } ( X ) ] \right) \le \mathrm { K L } ( q \| p _ { t } ) - \mathrm { K L } ( q \| p _ { t + 1 } ) } \\ & { \qquad + \eta \mathbb { E } _ { X \sim p _ { t } } [ Z _ { t } ( X ) ] + \log \mathbb { E } _ { X \sim p _ { t } } [ \exp ( - \eta Z _ { t } ( X ) ) ] + \varepsilon _ { \mathrm { p } } . } \end{array}
$$

By Lemma 5,

$$
| \eta Z _ { t } ( x ) | \leq { \frac { 1 } { 4 } } \qquad { \mathrm { f o r ~ e v e r y ~ } } x \in \mathcal { X } .
$$

Using

$$
e ^ { - u } \leq 1 - u + 2 u ^ { 2 } \qquad { \mathrm { f o r ~ } } | u | \leq { \frac { 1 } { 4 } } ,
$$

we obtain

$$
\mathbb { E } _ { X \sim p _ { t } } [ \exp ( - \eta Z _ { t } ( X ) ) ] \le 1 - \eta \mathbb { E } _ { X \sim p _ { t } } [ Z _ { t } ( X ) ] + 2 \eta ^ { 2 } \mathbb { E } _ { X \sim p _ { t } } [ Z _ { t } ( X ) ^ { 2 } ] .
$$

The inequality log $\left( 1 + s \right) \leq s$ then gives

$$
\eta \mathbb { E } _ { X \sim p _ { t } } [ Z _ { t } ( X ) ] + \log \mathbb { E } _ { X \sim p _ { t } } [ \exp ( - \eta Z _ { t } ( X ) ) ] \le 2 \eta ^ { 2 } \mathbb { E } _ { X \sim p _ { t } } [ Z _ { t } ( X ) ^ { 2 } ] .
$$

Therefore,

$$
\mathbb { E } _ { X \sim p _ { t } } [ Z _ { t } ( X ) ] - \mathbb { E } _ { X \sim q } [ Z _ { t } ( X ) ] \le \frac { \mathrm { K L } ( q \| p _ { t } ) - \mathrm { K L } ( q \| p _ { t + 1 } ) } { \eta } + 2 \eta \mathbb { E } _ { X \sim p _ { t } } [ Z _ { t } ( X ) ^ { 2 } ] + \frac { \varepsilon _ { \mathrm { p } } } { \eta } .
$$

Summing over t telescopes the KL terms. Finally, $\mathrm { K L } ( q \| p _ { T + 1 } ) \geq 0 .$ , so

$$
\sum _ { t = 1 } ^ { T } ( \mathbb { E } _ { X \sim p _ { t } } [ Z _ { t } ( X ) ] - \mathbb { E } _ { X \sim q } [ Z _ { t } ( X ) ] ) \leq \frac { \mathrm { K L } ( q \| p _ { 1 } ) } { \eta } + 2 \eta \sum _ { t = 1 } ^ { T } \mathbb { E } _ { X \sim p _ { t } } [ Z _ { t } ( X ) ^ { 2 } ] + \frac { T \varepsilon _ { \mathrm { p } } } { \eta } .
$$

Since $\varepsilon _ { \mathrm { p } } = \eta / T$ , the last term equals one.

Lemma 7. Under Assumption 1, let $\rho \in ( 0 , 1 )$ , and let $( q _ { t } ) _ { t = 1 } ^ { T }$ be a predictable sequence of distributions in $\Delta ( \mathcal { X } )$ . For every $t \in [ T ]$ , define

$$
A _ { t } ( q _ { t } ) : = \mathbb { E } _ { X \sim q _ { t } } [ \langle X , \ell _ { t } \rangle ] , \qquad \widehat { A } _ { t } ( q _ { t } ) : = \mathbb { E } _ { X \sim q _ { t } } [ \widehat { y } _ { t } ( X ) ] .
$$

Then, with probability at least $1 - \rho _ { ; }$

$$
\left| \sum _ { t = 1 } ^ { T } \left( \widehat { A } _ { t } ( q _ { t } ) - A _ { t } ( q _ { t } ) \right) \right| \leq \frac { \eta } { 4 } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { X \sim q _ { t } } [ c _ { t } ^ { \top } X ] + \frac { \log ( 2 / \rho ) } { \eta } .
$$

Proof. For every $t \in [ T ]$ , define

$$
\Delta _ { t } : = \widehat { A } _ { t } ( q _ { t } ) - A _ { t } ( q _ { t } ) .
$$

Recall that $\mathbb { E } _ { t } [ \cdot ] = \mathbb { E } [ \cdot \mid \mathcal { F } _ { t - 1 } ]$ . Since $p _ { t } , \ q _ { t }$ , and $\ell _ { t }$ are predictable, they are fixed conditionally on $\mathcal { F } _ { t - 1 }$ . The only randomness at round t comes from $X _ { t } \sim p _ { t }$ . We first show that $\Delta _ { t }$ has conditional mean zero. By the definition of the KW loss estimate,

$$
\begin{array} { r } { \mathbb { E } _ { t } [ \widehat \ell _ { t } ] = M _ { t } ^ { - 1 } \mathbb { E } _ { t } [ X _ { t } { X _ { t } ^ { \top } } ] \ell _ { t } = M _ { t } ^ { - 1 } M _ { t } \ell _ { t } = \ell _ { t } . } \end{array}
$$

Therefore,

$$
\mathbb { E } _ { t } [ \widehat { A } _ { t } ( q _ { t } ) ] = A _ { t } ( q _ { t } ) , \qquad \mathbb { E } _ { t } [ \Delta _ { t } ] = 0 .
$$

Define $\bar { x } _ { t } : = \mathbb { E } _ { X \sim q _ { t } } [ X ]$ . Then

$$
\widehat { A } _ { t } ( q _ { t } ) = \bar { x } _ { t } ^ { \top } M _ { t } ^ { - 1 } X _ { t } Y _ { t } .
$$

Since $| Y _ { t } | \le 1$

$$
\mathbb { E } _ { t } [ \widehat { A } _ { t } ( q _ { t } ) ^ { 2 } ] \leq \bar { x } _ { t } ^ { \top } M _ { t } ^ { - 1 } \mathbb { E } _ { t } [ X _ { t } X _ { t } ^ { \top } ] M _ { t } ^ { - 1 } \bar { x } _ { t } = \bar { x } _ { t } ^ { \top } M _ { t } ^ { - 1 } \bar { x } _ { t } .
$$

The matrix $M _ { t } ^ { - 1 }$ is positive semidefinite. Hence, Jensen’s inequality gives

$$
\bar { x } _ { t } ^ { \top } M _ { t } ^ { - 1 } \bar { x } _ { t } \leq \mathbb { E } _ { X \sim q _ { t } } \left[ X ^ { \top } M _ { t } ^ { - 1 } X \right] .
$$

By Lemma 4 and $c _ { t } ^ { \top } X = 4 \phi _ { t } ( X )$ ，

$$
X ^ { \top } M _ { t } ^ { - 1 } X \leq \frac { c _ { t } ^ { \top } X } { 4 } .
$$

It follows that

$$
\mathbb { E } _ { t } [ \widehat { A } _ { t } ( q _ { t } ) ^ { 2 } ] \leq \frac { 1 } { 4 } \mathbb { E } _ { X \sim q _ { t } } [ c _ { t } ^ { \top } X ] .
$$

Since $\mathbb { E } _ { t } [ \widehat { A } _ { t } ( q _ { t } ) ] = A _ { t } ( q _ { t } )$ , we obtain

$$
\mathbb { E } _ { t } [ \Delta _ { t } ^ { 2 } ] \leq \frac { 1 } { 4 } \mathbb { E } _ { X \sim q _ { t } } [ c _ { t } ^ { \top } X ] .\tag{25}
$$

We next bound $\Delta _ { t }$ uniformly. Since $c _ { t } ^ { \top } x \leq 4 B$ for every $x \in \mathcal { X }$ , the previous leverage bound gives

$$
\begin{array} { r } { \bar { x } _ { t } ^ { \top } M _ { t } ^ { - 1 } \bar { x } _ { t } \leq B , \qquad X _ { t } ^ { \top } M _ { t } ^ { - 1 } X _ { t } \leq B . } \end{array}
$$

Using Cauchy–Schwarz in the norm induced by $M _ { t } ^ { - 1 }$ , we obtain

$$
\begin{array} { r } { | \widehat { A } _ { t } ( q _ { t } ) | = \mathopen { } \mathclose \bgroup \left| \bar { x } _ { t } ^ { \top } M _ { t } ^ { - 1 } X _ { t } Y _ { t } \aftergroup \egroup \right| \leq \sqrt { \bar { x } _ { t } ^ { \top } M _ { t } ^ { - 1 } \bar { x } _ { t } } \sqrt { X _ { t } ^ { \top } M _ { t } ^ { - 1 } X _ { t } } \leq B . } \end{array}
$$

The action-normalized condition gives $| A _ { t } ( q _ { t } ) | \leq 1$ . Since $B \geq 1$ 2

$$
\begin{array} { r } { | \Delta _ { t } | \le B + 1 \le 2 B . } \end{array}
$$

Moreover,

$$
\eta B = \eta + \frac { 8 \eta d } { \lambda } = \eta + \frac { 1 } { 1 6 } \leq \frac { 5 } { 6 4 } .
$$

Therefore,

$$
\left| \eta \Delta _ { t } \right| \leq 2 \eta B \leq \frac { 5 } { 3 2 } < 1 .\tag{26}
$$

For every u with $| u | \leq 1$ ，

$$
e ^ { u } \leq 1 + u + u ^ { 2 } .
$$

Using (26), $\mathbb { E } _ { t } [ \Delta _ { t } ] = 0$ , and (25), we obtain

$$
\begin{array} { r l } & { \mathbb { E } _ { t } [ \exp ( \eta \Delta _ { t } ) ] \leq 1 + \eta ^ { 2 } \mathbb { E } _ { t } [ \Delta _ { t } ^ { 2 } ] } \\ & { \qquad \leq \exp \big ( \eta ^ { 2 } \mathbb { E } _ { t } [ \Delta _ { t } ^ { 2 } ] \big ) } \\ & { \qquad \leq \exp \left( \displaystyle \frac { \eta ^ { 2 } } { 4 } \mathbb { E } _ { X \sim q _ { t } } [ c _ { t } ^ { \top } X ] \right) . } \end{array}
$$

For $s \in \{ 0 , \ldots , T \}$ , define

$$
\mathcal { M } _ { s } : = \exp \left( \eta \sum _ { t = 1 } ^ { s } \Delta _ { t } - \frac { \eta ^ { 2 } } { 4 } \sum _ { t = 1 } ^ { s } \mathbb { E } _ { X \sim q _ { t } } [ c _ { t } ^ { \top } X ] \right) .
$$

The previous conditional moment bound shows that $( \mathcal { M } _ { s } ) _ { s = 0 } ^ { T }$ is a nonnegative supermartingale with $\mathcal { M } _ { 0 } = 1$ . Hence, $\mathbb { E } [ \mathcal { M } _ { T } ] \leq 1$ . By Markov’s inequality, with probability at least $1 - \rho / 2$ 2

$$
\sum _ { t = 1 } ^ { T } \Delta _ { t } \leq \frac { \eta } { 4 } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { X \sim q _ { t } } [ c _ { t } ^ { \top } X ] + \frac { \log ( 2 / \rho ) } { \eta } .
$$

The same argument applied $\mathrm { t o } \mathrm { ~ - } \Delta _ { t }$ gives, with probability at least $1 - \rho / 2$

$$
- \sum _ { t = 1 } ^ { T } \Delta _ { t } \le \frac { \eta } { 4 } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { X \sim q _ { t } } [ c _ { t } ^ { \top } X ] + \frac { \log ( 2 / \rho ) } { \eta } .
$$

A union bound over the two events proves the result.

Lemma 8. Under Assumption 1, for every $\rho \in ( 0 , 1 )$ , with probability at least $1 - \rho _ { ; }$

$$
\sum _ { t = 1 } ^ { T } \phi _ { t } ( X _ { t } ) \leq 2 ( 2 d + 1 ) T + 2 \left( 1 + \frac { 8 d } { \lambda } \right) \log \frac { 1 } { \rho } .
$$

Proof. Recall that $B = 1 + 8 d / \lambda$ . Conditionally on $\mathcal { F } _ { t - 1 }$ , the distribution $p _ { t }$ and the function $\phi _ { t }$ are fixed, while $X _ { t } \sim p _ { t }$ . Therefore, Lemma 4 gives

$$
\mathbb { E } _ { t } [ \phi _ { t } ( X _ { t } ) ] = \mathbb { E } _ { X \sim p _ { t } } [ \phi _ { t } ( X ) ] = 2 d + 1 , \qquad 0 \le \phi _ { t } ( X _ { t } ) \le B .
$$

Since $\phi _ { t } ( X _ { t } ) / B \in [ 0 , 1 ]$ , convexity of the exponential gives

$$
e ^ { u } \leq 1 + ( e - 1 ) u , \qquad u \in [ 0 , 1 ] .
$$

It follows that

$$
\mathbb { E } _ { t } \left[ \exp \left( \frac { \phi _ { t } ( X _ { t } ) } { B } \right) \right] \leq 1 + \frac { ( e - 1 ) ( 2 d + 1 ) } { B } \leq \exp \left( \frac { ( e - 1 ) ( 2 d + 1 ) } { B } \right) .
$$

For $s \in \{ 0 , \ldots , T \}$ , define

$$
\mathcal { M } _ { s } : = \exp \left( \frac { 1 } { B } \sum _ { t = 1 } ^ { s } \phi _ { t } ( X _ { t } ) - \frac { ( e - 1 ) ( 2 d + 1 ) s } { B } \right) .
$$

The previous conditional moment bound shows that $( \mathcal { M } _ { s } ) _ { s = 0 } ^ { T }$ is a nonnegative supermartingale with $\mathcal { M } _ { 0 } = 1$ . Hence,

$$
\mathbb { E } [ \mathcal { M } _ { T } ] \leq 1 .
$$

By Markov’s inequality, with probability at least $1 - \rho ,$

$$
\sum _ { t = 1 } ^ { T } \phi _ { t } ( X _ { t } ) \leq ( e - 1 ) ( 2 d + 1 ) T + B \log \frac { 1 } { \rho } .
$$

Using $e - 1 \leq 2$ and substituting the definition of B proves the result.

Lemma 9. Under Assumption 1, for every $\rho \in ( 0 , 1 )$ , with probability at least $1 - \rho ,$

$$
\sum _ { t = 1 } ^ { T } \left( \langle X _ { t } , \ell _ { t } \rangle - \mathbb { E } _ { X \sim p _ { t } } [ \langle X , \ell _ { t } \rangle ] \right) \leq { \sqrt { 2 T \log { \frac { 1 } { \rho } } } } .
$$

Proof. Define

$$
D _ { t } : = \langle X _ { t } , \ell _ { t } \rangle - \mathbb { E } _ { X \sim p _ { t } } [ \langle X , \ell _ { t } \rangle ] .
$$

Since $p _ { t }$ and $\ell _ { t }$ are determined before $X _ { t }$ is sampled,

$$
\mathbb { E } [ D _ { t } \mid \mathcal { F } _ { t - 1 } ] = 0 .
$$

Thus, $( D _ { t } ) _ { t = 1 } ^ { T }$ is a martingale diference sequence.

By the action-normalized condition, $\left. X _ { t } , \ell _ { t } \right. \in [ - 1 , 1 ]$ . Therefore, conditionally on $\mathcal { F } _ { t - 1 } , ~ D _ { t }$ belongs to an interval of length at most 2. The Azuma–Hoefding inequality then gives, for every $u > 0$ ，

$$
\mathbb { P } \left( \sum _ { t = 1 } ^ { T } D _ { t } \geq u \right) \leq \exp \left( - \frac { u ^ { 2 } } { 2 T } \right) .
$$

Taking $u = \sqrt { 2 T \log ( 1 / \rho ) }$ proves the result.

## B.1 Proof of the main regret theorem

Theorem 1 (High-probability KW bound). There exists a universal constant $C > 0$ such that the following holds. Let $T \geq 1$ be an integer, let $\delta \in ( 0 , 1 )$ , and assume $1 \leq m \leq d - 1$ . Run Algorithm 1. Then, with probability at least $1 - \delta$

$$
R _ { T } \leq C \sqrt { d T \left( \log K + \log \frac { 1 } { \delta } \right) } ,
$$

where $K = { \binom { d } { m } }$ . One may take $C = 1 6 0$

Proof. For every $x \in \mathcal { X }$ , define the smoothed comparator

$$
q _ { x } : = ( 1 - \lambda ) \delta _ { x } + \lambda U ,
$$

where $\delta _ { x }$ assigns probability one to x, and U is the uniform distribution over $\mathcal { X }$ . If $x _ { i } = 1$ , the i-th marginal of $q _ { x }$ is

$$
1 - \lambda ( 1 - r ) ,
$$

while, if $x _ { i } = 0$ , it is λr. Therefore, $q _ { x } \in { \mathcal { P } } _ { \lambda }$ . Moreover, since $p _ { 1 } = U$

$$
\mathrm { K L } ( q _ { x } \| p _ { 1 } ) = \mathrm { K L } ( q _ { x } \| U ) = \log K - H ( q _ { x } ) \leq \log K .
$$

Applying Lemma 6 with $q = q _ { x }$ gives

$$
\sum _ { t = 1 } ^ { T } ( \mathbb { E } _ { X \sim p _ { t } } [ Z _ { t } ( X ) ] - \mathbb { E } _ { X \sim q _ { x } } [ Z _ { t } ( X ) ] ) \leq \frac { \log K } { \eta } + 1 + 2 \eta \sum _ { t = 1 } ^ { T } \mathbb { E } _ { X \sim p _ { t } } [ Z _ { t } ( X ) ^ { 2 } ] .\tag{27}
$$

We first bound the second-order term. Since $Z _ { t } ( X ) = { \widehat { y } } _ { t } ( X ) - \eta c _ { t } ^ { \top } X$ , we have

$$
Z _ { t } ( X ) ^ { 2 } \leq 2 \widehat { y } _ { t } ( X ) ^ { 2 } + 2 \eta ^ { 2 } ( c _ { t } ^ { \top } X ) ^ { 2 } .
$$

Furthermore, we have

$$
\begin{array} { r l } { \mathbb { E } _ { X \sim p _ { t } } [ \widehat { y } _ { t } ( X ) ^ { 2 } ] = Y _ { t } ^ { 2 } X _ { t } ^ { \top } M _ { t } ^ { - 1 } \mathbb { E } _ { X \sim p _ { t } } [ X X ^ { \top } ] M _ { t } ^ { - 1 } X _ { t } } & { } \\ { = Y _ { t } ^ { 2 } X _ { t } ^ { \top } M _ { t } ^ { - 1 } X _ { t } } & { } \\ { \ } & { \leq \phi _ { t } ( X _ { t } ) , } \end{array}
$$

where the last inequality follows from $| Y _ { t } | \le 1$ and Lemma 4. Recall that

$$
0 \leq c _ { t } ^ { \top } X \leq 4 B , \qquad \mathbb { E } _ { X \sim p _ { t } } [ c _ { t } ^ { \top } X ] = 4 ( 2 d + 1 ) .
$$

Hence,

$$
\begin{array} { r } { \mathbb { E } _ { X \sim p _ { t } } [ ( c _ { t } ^ { \top } X ) ^ { 2 } ] \le 4 B \mathbb { E } _ { X \sim p _ { t } } [ c _ { t } ^ { \top } X ] = 1 6 B ( 2 d + 1 ) . } \end{array}
$$

It follows that

$$
2 \eta \mathbb { E } _ { X \sim p _ { t } } [ Z _ { t } ( X ) ^ { 2 } ] \leq 4 \eta \phi _ { t } ( X _ { t } ) + 6 4 \eta ^ { 3 } B ( 2 d + 1 ) .\tag{28}
$$

Using $Z _ { t } ( X ) = { \widehat { y } } _ { t } ( X ) - \eta c _ { t } ^ { \top } X$ in (27), and then applying (28), gives

$$
\begin{array} { r l } {  { \sum _ { t = 1 } ^ { T } ( \mathbb { E } _ { X \sim p _ { t } } [ \widehat { y } _ { t } ( X ) ] - \mathbb { E } _ { X \sim q _ { x } } [ \widehat { y } _ { t } ( X ) ] ) } } \\ & { \leq \frac { \log K } { \eta } + 1 + 4 \eta \sum _ { t = 1 } ^ { T } \phi _ { t } ( X _ { t } ) + 6 4 \eta ^ { 3 } B ( 2 d + 1 ) T + 4 \eta ( 2 d + 1 ) T - \eta \sum _ { t = 1 } ^ { T } \mathbb { E } _ { X \sim q _ { x } } [ c _ { t } ^ { \top } X ] . } \end{array}\tag{29}
$$

We now apply Lemma 8 with $\rho = \delta / 4$ . With probability at least $1 - \delta / 4$

$$
4 \eta \sum _ { t = 1 } ^ { T } \phi _ { t } ( X _ { t } ) \leq 8 \eta ( 2 d + 1 ) T + 8 \eta B \log { \frac { 4 } { \delta } } .
$$

Since

$$
\eta B = \eta + \frac { 8 \eta d } { \lambda } = \eta + \frac { 1 } { 1 6 } \leq \frac { 5 } { 6 4 } ,
$$

we have $8 \eta B \le 5 / 8$ . Therefore,

$$
4 \eta \sum _ { t = 1 } ^ { T } \phi _ { t } ( X _ { t } ) \leq 8 \eta ( 2 d + 1 ) T + \log \frac { 4 } { \delta } .\tag{30}
$$

Next, apply Lemma $7$ to the predictable sequence $q _ { t } = p _ { t }$ , with $\rho = \delta / 4$ . With probability at least $1 - \delta / 4$ 2

$$
\sum _ { t = 1 } ^ { T } \left( \mathbb { E } _ { X \sim p _ { t } } [ \langle X , \ell _ { t } \rangle ] - \mathbb { E } _ { X \sim p _ { t } } [ \widehat { y } _ { t } ( X ) ] \right) \leq \eta ( 2 d + 1 ) T + \frac { \log ( 8 / \delta ) } { \eta } .\tag{31}
$$

For every $x \in \mathcal { X }$ , apply the same lemma to the constant predictable sequence $q _ { t } = q _ { x } .$ with $\rho = \delta / ( 4 K )$ . A union bound over the K actions shows that, with probability at least $1 - \delta / 4$ simultaneously for every $x \in \mathcal { X }$ ，

$$
\sum _ { t = 1 } ^ { T } \left( \mathbb { E } _ { X \sim q _ { x } } [ \widehat { y } _ { t } ( X ) ] - \mathbb { E } _ { X \sim q _ { x } } [ \langle X , \ell _ { t } \rangle ] \right) \leq \frac { \eta } { 4 } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { X \sim q _ { x } } [ c _ { t } ^ { \top } X ] + \frac { \log ( 8 K / \delta ) } { \eta } .\tag{32}
$$

Combining (29), (30), (31), and (32), we obtain, simultaneously for every $x \in \mathcal { X }$

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T } ( \mathbb { E } _ { X \sim p _ { t } } [ \langle X , \ell _ { t } \rangle ] - \mathbb { E } _ { X \sim q _ { x } } [ \langle X , \ell _ { t } \rangle ] ) \le \frac { \log K + \log ( 8 / \delta ) + \log ( 8 K / \delta ) } { \eta } } \\ & { \displaystyle \qquad + 1 + 1 3 \eta ( 2 d + 1 ) T + 6 4 \eta ^ { 3 } B ( 2 d + 1 ) T + \log \frac { 4 } { \delta } - \eta \sum _ { t = 1 } ^ { T } \mathbb { E } _ { X \sim q _ { x } } [ c _ { t } ^ { \top } X ] + \frac { \eta } { 4 } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { X \sim q _ { x } } [ c _ { t } ^ { \top } X ] . } \end{array}\tag{33}
$$

The last two terms satisfy

$$
- \eta \sum _ { t = 1 } ^ { T } \mathbb { E } _ { X \sim q _ { x } } [ c _ { t } ^ { \top } X ] + \frac { \eta } { 4 } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { X \sim q _ { x } } [ c _ { t } ^ { \top } X ] = - \frac { 3 \eta } { 4 } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { X \sim q _ { x } } [ c _ { t } ^ { \top } X ] \le 0 ,
$$

because $c _ { t } ^ { \top } X \ge 0$ . Moreover, $6 4 \eta ^ { 3 } B = 6 4 \eta ^ { 2 } ( \eta B ) \leq 5 \eta ^ { 2 } \leq 5 \eta$ . Therefore, (33) gives

$$
\sum _ { t = 1 } ^ { T } ( \mathbb { E } _ { X \sim \rho _ { t } } [ \langle X , \ell _ { t } \rangle ] - \mathbb { E } _ { X \sim q _ { x } } [ \langle X , \ell _ { t } \rangle ] ) \le \frac { \log K + \log ( 8 / \delta ) + \log ( 8 K / \delta ) } { \eta } + 1 + 1 8 \eta ( 2 d + 1 ) T + \log \frac { 4 } { \delta } .
$$

Using $2 d + 1 \leq 3 d$ and enlarging the logarithmic terms, we obtain

$$
\sum _ { \ell = 1 } ^ { T } ( \mathbb { E } _ { X \sim p _ { t } } [ \langle X , \ell _ { t } \rangle ] - \mathbb { E } _ { X \sim q _ { x } } [ \langle X , \ell _ { t } \rangle ] ) \leq \frac { 3 \log K + 3 \log ( 1 2 / \delta ) } { \eta } + 1 + 5 4 \eta d T + \log \frac { 4 } { \delta } .\tag{34}
$$

Let $x ^ { \star }$ be a best action in hindsight. Since $q _ { x ^ { \star } } = ( 1 - \lambda ) \delta _ { x ^ { \star } } + \lambda U$ and every action loss belongs to $[ - 1 , 1 ]$

$$
\mathbb { E } _ { X \sim q _ { x ^ { \star } } } [ \langle X , \ell _ { t } \rangle ] \leq \langle x ^ { \star } , \ell _ { t } \rangle + 2 \lambda .
$$

Applying (34) with $x = x ^ { \star }$ gives

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { T } ( \mathbb { E } _ { X \sim p _ { t } } [ \langle X , \ell _ { t } \rangle ] - \langle x ^ { \star } , \ell _ { t } \rangle ) } } \\ & { } & { \leq \frac { 3 \log K + 3 \log ( 1 2 / \delta ) } { \eta } + 1 + 5 4 \eta d T + 2 \lambda T + \log \frac { 4 } { \delta } } \\ & { } & { \leq \frac { 3 \log K + 3 \log ( 1 2 / \delta ) } { \eta } + 1 + 3 1 0 \eta d T + \log \frac { 4 } { \delta } , } \end{array}
$$

where the last inequality uses $\lambda = 1 2 8 \eta d .$ Finally, apply Lemma 9 with $\rho = \delta / 4$ . With probability at least $1 - \delta / 4$

$$
\sum _ { t = 1 } ^ { T } \big ( \langle X _ { t } , \ell _ { t } \rangle - \mathbb { E } _ { X \sim p _ { t } } [ \langle X , \ell _ { t } \rangle ] \big ) \leq \sqrt { 2 T \log \frac { 4 } { \delta } } \leq 2 \sqrt { T \log \frac { 4 } { \delta } } .
$$

Since $3 1 0 \leq 3 2 0$ and $1 + \log ( 4 / \delta ) \leq 4 \log ( 1 2 / \delta )$ , a union bound over the four failure events gives

$$
R _ { T } \le \frac { 3 \log K + 3 \log ( 1 2 / \delta ) } { \eta } + 3 2 0 \eta d T + 4 \log \frac { 1 2 } { \delta } + 2 \sqrt { T \log \frac { 1 2 } { \delta } } .\tag{35}
$$

It remains to substitute the value of η. Define

$$
A : = \log K + \log { \frac { 1 2 } { \delta } } .
$$

Also set

$$
S : = \log K + \log { \frac { 1 } { \delta } } .
$$

Since $\begin{array} { r } { K = { \binom { d } { m } } \ge 2 } \end{array}$ , we have $S \geq \log 2$ , and therefore

$$
A = S + \log 1 2 \leq \left( 1 + { \frac { \log 1 2 } { \log 2 } } \right) S \leq { \frac { 2 3 } { 5 } } S ,\tag{36}
$$

where the last inequality follows from $1 2 ^ { 5 } < 2 ^ { 1 8 }$ . If $\eta = \sqrt { A / ( 3 2 0 d T ) }$ , then the first two terms in (35) are exactly

$$
\frac { 3 A } { \eta } + 3 2 0 \eta d T = 4 \sqrt { 3 2 0 d T A } .
$$

Moreover,

$$
4 \log { \frac { 1 2 } { \delta } } \leq 4 A , \qquad 2 \sqrt { T \log { \frac { 1 2 } { \delta } } } \leq 2 \sqrt { d T A } .
$$

It remains to consider the case in which $\eta = 1 / ( 2 5 6 d )$ . In this case,

$$
{ \sqrt { \frac { A } { 3 2 0 d T } } } \geq { \frac { 1 } { 2 5 6 d } } ,
$$

which implies

$$
T \leq { \frac { 2 5 6 ^ { 2 } } { 3 2 0 } } d A .
$$

Since the action-normalized condition gives the deterministic bound $R _ { T } \leq 2 T$ , we also have

$$
R _ { T } \leq 2 \sqrt { \frac { 2 5 6 ^ { 2 } } { 3 2 0 } } \sqrt { d T A } \leq 2 9 \sqrt { d T A } .
$$

In the non-clipped case, Equation (35) is at most $7 4 { \sqrt { d T A } } + 4 A$ . Since $2 9 < 7 4$ and $A > 0$ , the same bound also holds in the clipped case. Hence, by (36),

$$
R _ { T } \leq 7 4 \sqrt { \frac { 2 3 } { 5 } } \sqrt { d T S } + \frac { 9 2 } { 5 } S .\tag{37}
$$

Set

$$
\gamma : = \sqrt { \frac { S } { d T } } > 0 .
$$

Since

$$
S = \gamma { \sqrt { d T S } } ,
$$

if $\gamma \leq 1 / 1 6 0$ , then

$$
\begin{array} { r l r } & { } & { { R _ { T } } \leq \left( 7 4 \sqrt { \frac { 2 3 } { 5 } } + { \frac { 9 2 } { 5 } } \gamma \right) \sqrt { d T S } } \\ & { } & { \quad \leq \left( 7 4 \frac { 4 3 } { 2 0 } + \frac { 9 2 } { 5 } \frac { 1 } { 1 6 0 } \right) \sqrt { d T S } } \\ & { } & { \quad = \frac { 3 1 8 4 3 } { 2 0 0 } \sqrt { d T S } < 1 6 0 \sqrt { d T S } , } \end{array}
$$

where we used

$$
{ \sqrt { \frac { 2 3 } { 5 } } } < { \frac { 4 3 } { 2 0 } } .
$$

If instead $\gamma > 1 / 1 6 0$ , the deterministic bound $R _ { T } \leq 2 T$ and $d \geq 2$ give

$$
R _ { T } \leq 2 T = \frac { 2 } { d \gamma } \sqrt { d T S } \leq \frac { 1 } { \gamma } \sqrt { d T S } < 1 6 0 \sqrt { d T S } .
$$

Thus, in all cases,

$$
R _ { T } \leq 1 6 0 \sqrt { d T \left( \log K + \log \frac { 1 } { \delta } \right) } ,
$$

which proves the theorem with $C = 1 6 0$

## C Polynomial-time implementation

This appendix proves Proposition 1. We show that our algorithm can be implemented in polynomial time without enumerating X . It maintains each $p _ { t }$ as a weighted m-set distribution, computes the required moments and draws samples in polynomial time, and reduces the target KL projection to a convex problem in 2d variables. The convex problem is solved only to the inverse-polynomial accuracy specified below.

## C.1 Updates and KL projection

We first identify the exact KL projection toward which the algorithm computes and show that it reduces to a convex problem in 2d variables. We then prove that solving this problem to an explicit objective accuracy is suficient for the approximate update in Algorithm 1.

Suppose that $p _ { t } = p _ { \theta _ { t } }$ . Define the unprojected multiplicative-weights update

$$
\widetilde { p } _ { t + 1 } ( x ) : = \frac { p _ { t } ( x ) \exp ( - \eta Z _ { t } ( x ) ) } { \mathbb { E } _ { X \sim p _ { t } } [ \exp ( - \eta Z _ { t } ( X ) ) ] } .
$$

For every $p \in { \mathcal { P } } _ { \lambda }$

$$
\eta \mathbb { E } _ { X \sim p } [ Z _ { t } ( X ) ] + \mathrm { K L } ( p \| p _ { t } ) = \mathrm { K L } ( p \| \widetilde p _ { t + 1 } ) - \log \mathbb { E } _ { X \sim p _ { t } } [ \exp ( - \eta Z _ { t } ( X ) ) ] .
$$

The last term does not depend on $p .$ Therefore, the exact constrained OMD iterate is the KL projection of $\widetilde { p } _ { t + 1 }$ onto $\mathcal { P } _ { \lambda }$ , namely,

$$
\begin{array} { r } { p _ { t + 1 } ^ { \star } \in \underset { p \in \mathcal { P } _ { \lambda } } { \arg \operatorname* { m i n } } \mathrm { K L } ( p \| \widetilde { p } _ { t + 1 } ) . } \end{array}
$$

The algorithm computes the approximate replacement specified in Algorithm 1 rather than assuming access to this exact projection.

We next show that the first step preserves the weighted m-set form. Since $c _ { t } ^ { \top } x$ is afine in $x ,$ the surrogate loss is afine on X:

$$
Z _ { t } ( x ) = \langle x , \widehat { \ell } _ { t } \rangle - \eta c _ { t } ^ { \top } x = z _ { t } ^ { \top } x , \qquad z _ { t } : = \widehat { \ell } _ { t } - \eta c _ { t } .
$$

Consequently,

$$
\widetilde { p } _ { t + 1 } ( x ) = \frac { p _ { \theta _ { t } } ( x ) \exp ( - \eta z _ { t } ^ { \top } x ) } { \mathbb { E } _ { X \sim p _ { \theta _ { t } } } [ \exp ( - \eta z _ { t } ^ { \top } X ) ] } = p _ { \widetilde { \theta } _ { t + 1 } } ( x ) , \qquad \widetilde { \theta } _ { t + 1 } : = \theta _ { t } - \eta z _ { t } .\tag{38}
$$

Hence, if $p _ { t }$ is a weighted m-set distribution, then so is $\widetilde { p } _ { t + 1 }$

It remains to analyze the target KL projection. The next lemma shows that the projection of a weighted m-set distribution onto $\mathcal { P } _ { \lambda }$ is again a weighted m-set distribution. It also shows that the projection can be obtained by solving a convex problem in 2d variables.

Lemma 10. Define

$$
a _ { i } : = \lambda r , \qquad b _ { i } : = 1 - \lambda ( 1 - r ) , \qquad i \in [ d ] .
$$

For every $\widetilde { \theta } \in \mathbb { R } ^ { d }$ , the KL projection

$$
\begin{array} { c } { \operatorname { a r g m i n } \operatorname { K L } ( p \| p _ { \widetilde { \theta } } ) } \\ { p { \in } \mathcal { P } _ { \lambda } } \end{array}
$$

is a weighted m-set distribution $p _ { \theta ^ { + } }$ , where $\theta ^ { + } = \widetilde { \theta } + \alpha ^ { \star } - \beta ^ { \star }$ . Here, $( \alpha ^ { \star } , \beta ^ { \star } ) \in \mathbb { R } _ { + } ^ { d } \times \mathbb { R } _ { + } ^ { d }$ minimizes the convex function

$$
\Psi _ { \widetilde { \theta } } ( \alpha , \beta ) : = F ( \widetilde { \theta } + \alpha - \beta ) - \alpha ^ { \top } a + \beta ^ { \top } b .
$$

Moreover,

$$
\frac { \partial \Psi _ { \widetilde { \theta } } } { \partial \alpha _ { i } } = \mu _ { i } ( \widetilde { \theta } + \alpha - \beta ) - a _ { i } , \qquad \frac { \partial \Psi _ { \widetilde { \theta } } } { \partial \beta _ { i } } = b _ { i } - \mu _ { i } ( \widetilde { \theta } + \alpha - \beta ) .
$$

Proof. The projection minimizes $\mathrm { K L } ( p \Vert p _ { \widetilde { \theta } } )$ over $p \in \Delta ( \mathcal X )$ , subject to

$$
a _ { i } - \mathbb { E } _ { X \sim p } [ X _ { i } ] \leq 0 , \qquad \mathbb { E } _ { X \sim p } [ X _ { i } ] - b _ { i } \leq 0 .
$$

The uniform distribution has all marginals equal to $r ,$ strictly between $a _ { i }$ and $b _ { i }$ . Hence, Slater’s condition holds. We may therefore use the standard strong-duality and KKT results of Boyd and Vandenberghe [6, Sections 5.2.3 and 5.5.3].

Introduce multipliers $\alpha _ { i } , \beta _ { i } \geq 0$ . The Lagrangian is

$$
\begin{array} { r } { \mathcal { L } ( p , \boldsymbol { \alpha } , \beta ) = \mathrm { K L } ( p \| p _ { \widetilde { \theta } } ) + \boldsymbol { \alpha } ^ { \top } \boldsymbol { a } - \boldsymbol { \beta } ^ { \top } \boldsymbol { b } + \mathbb { E } _ { \boldsymbol { X } \sim p } [ ( \beta - \boldsymbol { \alpha } ) ^ { \top } \boldsymbol { X } ] . } \end{array}
$$

For fixed $( \alpha , \beta )$ , minimizing over p gives

$$
p ( x ) \propto p _ { \widetilde { \theta } } ( x ) e ^ { ( \alpha - \beta ) ^ { \top } x } = p _ { \widetilde { \theta } + \alpha - \beta } ( x ) .
$$

The resulting dual maximization is equivalent to minimizing $\Psi _ { \widetilde { \theta } } .$ . Strong duality proves the first claim, and the derivative identities follow from $\nabla F = \mu$ □

Lemma 11. Let $0 < \lambda \leq 1 / 2$ , let $\varepsilon > 0$ , and let $\widetilde { \theta } \in \mathbb { R } ^ { d }$ satisfy

$$
\mathrm { K L } ( U \| p _ { \widetilde { \theta } } ) \leq T .
$$

Define

$$
h : = \frac \lambda 2 \operatorname* { m i n } \{ r , 1 - r \} = \frac { \lambda s } { 2 d } , \qquad R : = 2 d T ,
$$

and

$$
\tau : = \operatorname* { m i n } \left\{ 2 h ^ { 2 } , \frac { \varepsilon } 2 , \frac { \varepsilon ^ { 2 } } { 2 R ^ { 2 } } \right\} .\tag{39}
$$

Let

$$
\mathcal { D } _ { R } : = \Big \{ ( \alpha , \beta ) \in \mathbb { R } _ { + } ^ { d } \times \mathbb { R } _ { + } ^ { d } : \| \alpha \| _ { 1 } + \| \beta \| _ { 1 } \leq R \Big \} .
$$

Suppose that $( \alpha , \beta ) \in \mathcal { D } _ { R }$ satisfies

$$
\Psi _ { \widetilde { \theta } } ( \alpha , \beta ) \leq \operatorname* { m i n } _ { ( u , v ) \in \mathcal { D } _ { R } } \Psi _ { \widetilde { \theta } } ( u , v ) + \tau .\tag{40}
$$

Set

$$
\theta ^ { + } : = \widetilde { \theta } + \alpha - \beta , \qquad p ^ { + } : = p _ { \theta ^ { + } } .
$$

Then $p ^ { + } \in \mathcal { P } _ { \lambda / 2 }$ , and, for every $q \in { \mathcal { P } } _ { \lambda }$ ，

$$
\mathrm { K L } ( q \| p ^ { + } ) + \mathrm { K L } ( p ^ { + } \| p _ { \widetilde { \theta } } ) \leq \mathrm { K L } ( q \| p _ { \widetilde { \theta } } ) + \varepsilon .\tag{41}
$$

Moreover, a pair satisfying (40) can be computed by the ellipsoid method in time

$$
\mathrm { p o l y } \left( d , m , \log R , \log { \frac { 1 } { \tau } } \right) .
$$

Proof. Write

$$
a _ { i } : = \lambda r , \qquad b _ { i } : = 1 - \lambda ( 1 - r ) .
$$

Let $( \alpha ^ { \star } , \beta ^ { \star } )$ be an exact minimizer of $\Psi _ { \widetilde { \theta } } ,$ let

$$
\theta ^ { \star } : = \widetilde { \theta } + \alpha ^ { \star } - \beta ^ { \star } , \qquad p ^ { \star } : = p _ { \theta ^ { \star } } ,
$$

and write $\mu ^ { \star } : = \mu ( p ^ { \star } )$ . We first show that an exact minimizer belongs to $\mathcal { D } _ { R }$ . By strong duality, the value of the dual function at $( \alpha ^ { \star } , \beta ^ { \star } )$ equals the value of the KL projection problem and is therefore nonnegative. Evaluating the Lagrangian from the proof of Lemma 10 at the uniform distribution gives

$$
0 \leq \mathrm { K L } ( U \| p _ { \widetilde { \theta } } ) - ( 1 - \lambda ) r \| \alpha ^ { \star } \| _ { 1 } - ( 1 - \lambda ) ( 1 - r ) \| \beta ^ { \star } \| _ { 1 } .
$$

Consequently,

$$
\| \alpha ^ { \star } \| _ { 1 } + \| \beta ^ { \star } \| _ { 1 } \leq \frac { T } { ( 1 - \lambda ) \operatorname* { m i n } \{ r , 1 - r \} } \leq \frac { 2 d T } { s } \leq R .
$$

Thus the minimum over $\mathcal { D } _ { R }$ in (40) is the global minimum.

Let $p : = p ^ { + }$ , let $\mu : = \mu ( p )$ , and abbreviate $\Psi : = \Psi _ { \widetilde { \theta } } .$ The KKT conditions for the exact projection give

$$
a _ { i } \le \mu _ { i } ^ { \star } \le b _ { i } , \qquad \alpha _ { i } ^ { \star } ( \mu _ { i } ^ { \star } - a _ { i } ) = 0 , \qquad \beta _ { i } ^ { \star } ( b _ { i } - \mu _ { i } ^ { \star } ) = 0 .
$$

Using

$$
\operatorname { K L } ( p ^ { \star } \| p ) = F ( \theta ^ { + } ) - F ( \theta ^ { \star } ) - ( \mu ^ { \star } ) ^ { \top } ( \theta ^ { + } - \theta ^ { \star } ) ,
$$

and the complementary-slackness identities above, direct expansion gives the exact identity

$$
\Psi ( \alpha , \beta ) - \Psi ( \alpha ^ { \star } , \beta ^ { \star } ) = \operatorname { K L } ( p ^ { \star } \| p ) + \alpha ^ { \top } ( \mu ^ { \star } - a ) + \beta ^ { \top } ( b - \mu ^ { \star } ) .\tag{42}
$$

All three terms on the right-hand side are nonnegative. Hence,

$$
\mathrm { K L } ( p ^ { \star } \| p ) \leq \tau .
$$

Pinsker’s inequality and the fact that every coordinate of an action belongs to [0, 1] imply

$$
\| \mu - \mu ^ { \star } \| _ { \infty } \leq \mathrm { T V } ( p , p ^ { \star } ) \leq \sqrt { \frac { \tau } { 2 } } \leq h .
$$

Every marginal of $p ^ { \star }$ belongs to $\left[ \lambda r , 1 - \lambda ( 1 - r ) \right]$ . The distance from this interval to the corresponding boundary of $[ \lambda r / 2 , 1 - \lambda ( 1 - r ) / 2 ]$ is at least h. Therefore, $p \in \mathcal { P } _ { \lambda / 2 }$

Define the residual

$$
\kappa : = \alpha ^ { \top } ( \mu - a ) + \beta ^ { \top } ( b - \mu ) .
$$

By (42) and $( \alpha , \beta ) \in \mathcal { D } _ { R }$

$$
\begin{array} { l } { \kappa = { \alpha } ^ { \top } ( { \mu } ^ { \star } - a ) + { \beta } ^ { \top } ( b - { \mu } ^ { \star } ) + ( { \alpha } - { \beta } ) ^ { \top } ( { \mu } - { \mu } ^ { \star } ) } \\ { \quad \le { \tau } + R \sqrt { \frac { \tau } { 2 } } \le \frac { \varepsilon } { 2 } + \frac { \varepsilon } { 2 } = \varepsilon , } \end{array}
$$

where the last inequality follows from (39).

For every $q \in { \mathcal { P } } _ { \lambda }$ , the exponential-family representation gives the three-point identity

$$
\begin{array} { r l } & { \mathrm { K L } ( q \| p _ { \widetilde { \theta } } ) - \mathrm { K L } ( q \| p ) - \mathrm { K L } ( p \| p _ { \widetilde { \theta } } ) } \\ & { \qquad = ( \mu ( q ) - \mu ) ^ { \top } ( \alpha - \beta ) . } \end{array}
$$

Since $a \leq \mu ( q ) \leq b$ coordinatewise and $\alpha , \beta \ge 0$ , the right-hand side is at least −κ. The bound $\kappa \leq \varepsilon$ proves (41).

It remains to justify the computational claim. Put

$$
L : = \sqrt { 2 d } , \qquad r _ { 0 } : = \frac { R } { 8 d } , \qquad w _ { 0 } : = \frac { R } { 4 d } { \bf 1 } _ { 2 d } .
$$

For $\boldsymbol { v } \in \mathbb { R } ^ { 2 d }$ and $r > 0$ , write $B ( v , r ) : = \{ u \in \mathbb { R } ^ { 2 d } : \| u - v \| _ { 2 } \leq r \}$ . The set $\mathcal { D } _ { R }$ has an explicit separation procedure, is contained in the Euclidean ball of radius $R ,$ and contains $B ( w _ { 0 } , r _ { 0 } )$ . Indeed, every point of this ball has nonnegative coordinates and $\ell _ { 1 } { \mathrm { - n o r m } }$ at most $R / 2 + \sqrt { 2 d } r _ { 0 } \leq R$ . By the derivative formulas in Lemma 10,

$$
\| \nabla \Psi _ { \widetilde { \theta } } ( w ) \| _ { 2 } \leq L \qquad \mathrm { f o r ~ e v e r y ~ } w \in \mathbb { R } ^ { 2 d } .
$$

Thus $\Psi _ { \widetilde { \theta } }$ is globally L-Lipschitz. Its value and gradient can be evaluated using poly(d, m) operations by Lemma 12.

Let $w ^ { \star }$ minimize $\Psi _ { \widetilde { \theta } }$ over $\mathcal { D } _ { R }$ . For $0 < \zeta \leq r _ { 0 }$ , define

$$
w _ { \zeta } : = \left( 1 - \frac { \zeta } { r _ { 0 } } \right) w ^ { \star } + \frac { \zeta } { r _ { 0 } } w _ { 0 } .
$$

Convexity of $\mathcal { D } _ { R }$ and $B ( w _ { 0 } , r _ { 0 } ) \subseteq { \mathcal { D } } _ { R }$ imply $B ( w _ { \zeta } , \zeta ) \subseteq \mathcal { D } _ { R }$ . Moreover, $\| w ^ { \star } - w _ { 0 } \| _ { 2 } \le 2 R$ , and hence

$$
\Psi _ { \widetilde { \theta } } ( w _ { \zeta } ) \leq \Psi _ { \widetilde { \theta } } ( w ^ { \star } ) + 1 6 d L \zeta .
$$

Set

$$
\zeta : = \operatorname* { m i n } \left\{ r _ { 0 } , \frac { \tau } { 1 + ( 1 6 d + 1 ) L } \right\} .
$$

Weak constrained convex-function minimization returns a point $\widehat { w }$ at Euclidean distance at most $\zeta$ from $\mathcal { D } _ { R }$ , whose objective is at most $\zeta$ above the objective of every point whose closed ζ-ball is contained in $\mathcal { D } _ { R } \ [ 1 5 $ , Problem (2.1.22) and Theorem 4.3.13]. In particular,

$$
\Psi _ { \widetilde { \theta } } ( \widehat { w } ) \leq \Psi _ { \widetilde { \theta } } ( w ^ { \star } ) + ( 1 6 d L + 1 ) \zeta .
$$

Let w be the Euclidean projection of wb onto $\mathcal { D } _ { R }$ . Since $\mathcal { D } _ { R }$ is the nonnegative $\ell _ { 1 } { \mathrm { - b a l l } }$ , its coordinates have the form

$$
\overline { { w } } _ { i } = \operatorname* { m a x } \{ \widehat { w } _ { i } - \chi , 0 \} ,
$$

where $\chi = 0$ if the positive part of $\widehat { w }$ has $\ell _ { 1 } { \mathrm { - n o r m ~ a t } }$ most R, and otherwise $\chi$ is chosen so that $\begin{array} { r } { \sum _ { i } \operatorname* { m a x } \{ \widehat { w } _ { i } - \chi , 0 \} = R } \end{array}$ . Sorting the coordinates finds $\chi$ in $O ( d \log d )$ time. Furthermore, $\| \overline { { w } } - \widehat { w } \| _ { 2 } \leq \zeta$ , so global Lipschitzness gives

$$
\begin{array} { r l } & { \Psi _ { \widetilde { \theta } } ( \overline { { w } } ) \leq \Psi _ { \widetilde { \theta } } ( w ^ { \star } ) + \big ( 1 + ( 1 6 d + 1 ) L \big ) \zeta } \\ & { \qquad \leq \Psi _ { \widetilde { \theta } } ( w ^ { \star } ) + \tau . } \end{array}
$$

Writing $\overline { { w } } = ( \alpha , \beta )$ gives a pair in $\mathcal { D } _ { R }$ satisfying (40). Since $R / r _ { 0 } = 8 d$ and log $( 1 / \zeta ) = O ( \log d +$ log $R + \log ( 1 / \tau ) )$ ), the weak optimizer and the final projection use poly(d, m, log R, log(1/τ )) operations. □

## C.2 Moment and sampling routines

We now show how to compute the moments of a weighted m-set distribution and draw samples from it without enumerating X. Given its parameters θ, we compute its marginals, its second-moment matrix, and an exact sample using dynamic programs for elementary symmetric polynomials.

Lemma 12. Given $\theta \in \mathbb { R } ^ { d } .$ , the log-partition function $F ( \theta )$ , all marginals $\mu _ { i } ( p _ { \theta } )$ , all pairwise marginals

$$
\pi _ { i j } ( p _ { \theta } ) : = \mathbb { P } _ { X \sim p _ { \theta } } ( X _ { i } = X _ { j } = 1 ) ,
$$

and a sample from p<sub>θ</sub> can be drawn using a number of arithmetic operations polynomial in d and m, without enumerating X.

Proof. Let $w _ { i } : = e ^ { \theta _ { i } }$ , and define

$$
E [ i , k ] : = e _ { k } ( w _ { 1 } , \dots , w _ { i } ) .
$$

The recurrence

$$
E [ i , k ] = E [ i - 1 , k ] + w _ { i } E [ i - 1 , k - 1 ]\tag{43}
$$

computes all entries with $0 \leq i \leq d$ and $0 \leq k \leq m$ in $O ( d m )$ arithmetic operations. In particular, $E [ 0 , 0 ] = 1$ and $E [ i , k ] = 0$ whenever $k < 0$ or $k > i$ . Moreover,

$$
F ( \theta ) = \log E [ d , m ] .
$$

Write $E _ { k } : = e _ { k } ( w )$ . For each $i \in [ d ]$ , define

$$
E _ { 0 } ^ { ( - i ) } : = 1 , \qquad E _ { k } ^ { ( - i ) } : = E _ { k } - w _ { i } E _ { k - 1 } ^ { ( - i ) } .
$$

For $i \neq j$ , define

$$
E _ { 0 } ^ { ( - i , - j ) } : = 1 , \qquad E _ { k } ^ { ( - i , - j ) } : = E _ { k } ^ { ( - i ) } - w _ { j } E _ { k - 1 } ^ { ( - i , - j ) } .
$$

We use the value zero for both arrays at negative indices. These recurrences give

$$
\mu _ { i } ( p _ { \theta } ) = \frac { w _ { i } E _ { m - 1 } ^ { ( - i ) } } { E _ { m } } , \qquad \pi _ { i i } ( p _ { \theta } ) = \mu _ { i } ( p _ { \theta } ) , \qquad \pi _ { i j } ( p _ { \theta } ) = \frac { w _ { i } w _ { j } E _ { m - 2 } ^ { ( - i , - j ) } } { E _ { m } } \quad ( i \neq j ) .\tag{44}
$$

All pairwise marginals can be obtained in $O ( d ^ { 2 } m )$ arithmetic operations. The recurrence is Method 2 of Chen and Liu [11, Section 2].

For sampling, start from $( i , k ) = ( d , m )$ . At state (i, k), include item i with probability

$$
{ \frac { w _ { i } E [ i - 1 , k - 1 ] } { E [ i , k ] } } .\tag{45}
$$

If item i is selected, continue from $( i - 1 , k - 1 )$ ; otherwise, continue from $( i - 1 , k )$ . The recurrence (43) shows that (45) is the correct conditional probability. Thus, the procedure samples from p<sub>θ</sub> using $O ( d m )$ arithmetic operations after the table has been computed. It is the reverse-indexed conditional-Bernoulli procedure of Chen and Liu [11, Section 4, Procedure 3]; it is also equivalent to their direct backward-path construction (Procedure 5) and to the diagonal-L specialization of Kulesza and Taskar [24, Section 3.1 and Algorithm 2]. □

## C.3 Proof of Proposition 1

Proposition 1. Algorithm 1 can be implemented in T ·poly(d, m, log T) time and poly $( d , m )$ space, without enumerating X . At every round, $p _ { t }$ is a weighted m-set distribution $p _ { \theta _ { t } }$ represented by the d parameters $\theta _ { t , 1 } , \ldots , \theta _ { t , d }$

Proof. Recall that

$$
\varepsilon _ { \mathrm { p } } = { \frac { \eta } { T } } .
$$

We prove by induction that every $p _ { t }$ is a weighted m-set distribution in $\mathcal { P } _ { \lambda / 2 }$ and that

$$
\mathrm { K L } ( U \| p _ { t } ) \leq ( t - 1 ) \left( { \frac { 1 } { 2 } } + \varepsilon _ { \mathrm { p } } \right) .\tag{46}
$$

This holds at $t = 1$ , because $\theta _ { 1 } = 0$ and $p _ { 1 } = p _ { \theta _ { 1 } } = U$

Suppose that the claim holds at round t. Given $\theta _ { t } .$ Lemma 12 computes the marginals and the second-moment matrix $M _ { t }$ in $O ( d ^ { 2 } m )$ operations. The same marginals determine $\phi _ { t }$ and $c _ { t }$ in $O ( d )$ operations. The sampling procedure in Lemma 12 draws $X _ { t }$ from $p _ { t }$ without enumerating $\mathcal { X } \mathrm { : }$ at each step, it draws one uniform random variable and compares it with the conditional probability in (45).

By Lemma 3, $M _ { t }$ is positive definite. Therefore, $M _ { t } ^ { - 1 }$ and $\widehat { \ell } _ { t }$ can be computed in $O ( d ^ { 3 } )$ operations. The surrogate loss is represented by the d-dimensional vector $z _ { t } = \widehat { \ell } _ { t } - \eta c _ { t }$ , and (38) gives $\widetilde { p } _ { t + 1 } = p _ { \widetilde { \theta } _ { t + 1 } }$

We next verify the hypothesis of Lemma 11. By the definition of the unprojected update,

$$
\mathrm { K L } ( U \| \widetilde { p } _ { t + 1 } ) = \mathrm { K L } ( U \| p _ { t } ) + \eta \mathbb { E } _ { X \sim U } [ Z _ { t } ( X ) ] + \log \mathbb { E } _ { X \sim p _ { t } } [ e ^ { - \eta Z _ { t } ( X ) } ] .
$$

Under the induction hypothesis, Lemma 5 gives $| \eta Z _ { t } ( x ) | \le 1 / 4$ for every $x \in \mathcal { X }$ . In particular,

$$
\eta \mathbb { E } _ { X \sim U } [ Z _ { t } ( X ) ] \le \frac { 1 } { 4 } , \qquad \log \mathbb { E } _ { X \sim p _ { t } } [ e ^ { - \eta Z _ { t } ( X ) } ] \le \log ( e ^ { 1 / 4 } ) = \frac { 1 } { 4 } .
$$

Hence,

$$
\mathrm { K L } ( U \| \widetilde { p } _ { t + 1 } ) \leq \mathrm { K L } ( U \| p _ { t } ) + \frac { 1 } { 2 } \leq \frac { t } { 2 } + ( t - 1 ) \varepsilon _ { \mathrm { p } } \leq T ,
$$

where the last inequality uses $t \leq T$ and $T \varepsilon _ { \mathrm { p } } = \eta \leq 1 / 6 4$

Apply Lemma 11 with $\varepsilon = \varepsilon _ { \mathrm { p } }$ . It computes $( \alpha _ { t } , \beta _ { t } ) \in \mathcal { D } _ { 2 d T }$ so that (40) holds with the value τ in (39). Set

$$
\theta _ { t + 1 } : = \widetilde { \theta } _ { t + 1 } + \alpha _ { t } - \beta _ { t } , \qquad p _ { t + 1 } : = p _ { \theta _ { t + 1 } } .
$$

The lemma gives $p _ { t + 1 } \in \mathcal { P } _ { \lambda / 2 }$ and the KL inequality required in Algorithm 1. Since $U \in { \mathcal { P } } _ { \lambda }$ , that inequality also gives

$$
\mathrm { K L } ( U \| p _ { t + 1 } ) \leq \mathrm { K L } ( U \| \widetilde p _ { t + 1 } ) + \varepsilon _ { \mathrm { p } } \leq t \left( \frac { 1 } { 2 } + \varepsilon _ { \mathrm { p } } \right) .
$$

This closes the induction.

It remains to bound the number of operations uniformly over the rounds. Since $K \geq 2$ and $\delta < 1$ , the learning rate in Algorithm 1 satisfies

$$
\eta \geq { \frac { 1 } { 2 5 6 d { \sqrt { T } } } } .
$$

Consequently,

$$
\varepsilon _ { \mathrm { p } } \geq \frac { 1 } { 2 5 6 d T ^ { 3 / 2 } } , \qquad \frac { \lambda s } { 2 d } \geq \frac { 1 } { 4 d \sqrt { T } } , \qquad R = 2 d T .
$$

The value τ in (39) therefore satisfies

$$
\tau \geq { \frac { 1 } { 2 ^ { 1 9 } d ^ { 4 } T ^ { 5 } } } .
$$

Thus log $R + \log ( 1 / \tau ) = O ( \log ( d T ) )$ , and Lemma 11 runs in $\operatorname { p o l y } ( d , m , \log T )$ time per round. All other steps run in time polynomial in d and $m$ . The algorithm therefore runs in $T \cdot \mathrm { p o l y } ( d , m , \log T )$ time, uses poly(d, m) space, and never enumerates X. □

## References

[1] Jacob Abernethy and Alexander Rakhlin. Beating the adaptive bandit with high probability. In Proceedings of the 22nd Annual Conference on Learning Theory, 2009. URL https://www.learningtheory.org/colt2009/papers/025.pdf.

[2] Jacob Abernethy, Elad Hazan, and Alexander Rakhlin. Competing in the dark: An eficient algorithm for bandit linear optimization. In Proceedings of the 21st Annual Conference on Learning Theory, pages 263–274, 2008. URL https://www.learningtheory.org/colt2008/papers/123-Abernethy.pdf.

[3] Jean-Yves Audibert, S´ebastien Bubeck, and G´abor Lugosi. Regret in online combinatorial optimization. Mathematics of Operations Research, 39(1):31–45, 2014. doi: 10.1287/moor.2013.0598. URL https://doi.org/10.1287/moor.2013.0598.

[4] Baruch Awerbuch and Robert D. Kleinberg. Adaptive routing with end-to-end feedback: Distributed learning and geometric approaches. In Proceedings of the Thirty-Sixth Annual ACM Symposium on Theory of Computing, pages 45–53. ACM, 2004. doi: 10.1145/1007352.1007367. URL https://doi.org/10.1145/1007352.1007367.

[5] Peter L. Bartlett, Varsha Dani, Thomas P. Hayes, Sham M. Kakade, Alexander Rakhlin, and Ambuj Tewari. High-probability regret bounds for bandit online linear optimization. In Proceedings of the 21st Annual Conference on Learning Theory, pages 335–342. Omnipress, 2008. URL https://www.learningtheory.org/colt2008/papers/30-Bartlett.pdf.

[6] Stephen Boyd and Lieven Vandenberghe. Convex Optimization. Cambridge University Press, 2004.

[7] G´abor Braun and Sebastian Pokutta. An eficient high-probability algorithm for linear bandits, 2016. URL https://arxiv.org/abs/1610.02072.

[8] S´ebastien Bubeck, Nicol\`o Cesa-Bianchi, and Sham M. Kakade. Towards minimax policies for online linear optimization with bandit feedback. In Shie Mannor, Nathan Srebro, and Robert C. Williamson, editors, Proceedings of the 25th Annual Conference on Learning Theory, volume 23 of Proceedings of Machine Learning Research, pages 41.1–41.14. PMLR, 2012. URL https://proceedings.mlr.press/v23/bubeck12a.html.

[9] Nicol\`o Cesa-Bianchi and G´abor Lugosi. Combinatorial bandits. Journal of Computer and System Sciences, 78(5):1404–1422, 2012. doi: 10.1016/j.jcss.2012.01.001. URL https://doi.org/10.1016/j.jcss.2012.01.001.

[10] Tommaso Cesari and Roberto Colomboni. Efective resistance in fixed-rank external-field measures and constant-stretch correlated sampling on the hypersimplex, 2026. URL https://arxiv.org/abs/2607.13990.

[11] Sean X. Chen and Jun S. Liu. Statistical applications of the poisson-binomial and conditional bernoulli distributions. Statistica Sinica, 7(4):875–892, 1997. URL https://www3.stat.sinica.edu.tw/statistica/j7n4/j7n44/j7n44.htm.

[12] Xiang-Hui Chen, Arthur P. Dempster, and Jun S. Liu. Weighted finite population sampling to maximize entropy. Biometrika, 81(3):457–469, 1994. doi: 10.1093/biomet/81.3.457. URL https://doi.org/10.1093/biomet/81.3.457.

[13] Alon Cohen, Tamir Hazan, and Tomer Koren. Tight bounds for bandit combinatorial optimization. In Satyen Kale and Ohad Shamir, editors, Proceedings of the 30th Conference on Learning Theory, volume 65 of Proceedings of Machine Learning Research, pages 629–642. PMLR, 2017. URL https://proceedings.mlr.press/v65/cohen17a.html.

[14] Richard Combes, Mohammad Sadegh Talebi Mazraeh Shahi, Alexandre Prouti\`ere, and Marc Lelarge. Combinatorial bandits revisited. In Advances in Neural Information Processing Systems, volume 28, pages 2116–2124. Curran Associates, Inc., 2015. URL https://proceedings.neurips.cc/paper/2015/hash/ 0ce2ffd21fc958d9ef0ee9ba5336e357-Abstract.html.

[15] Martin Gr¨otschel, L´aszl´o Lov´asz, and Alexander Schrijver. Geometric Algorithms and Combinatorial Optimization, volume 2 of Algorithms and Combinatorics. Springer, Berlin, 1988. doi: 10.1007/978-3-642-97881-4. URL https://doi.org/10.1007/978-3-642-97881-4.

[16] Jaroslav H´ajek. Asymptotic theory of rejective sampling with varying probabilities from a finite population. The Annals of Mathematical Statistics, 35(4):1491–1523, 1964. doi: 10.1214/aoms/1177700375. URL https://doi.org/10.1214/aoms/1177700375.

[17] Elad Hazan and Zohar Karnin. Volumetric spanners: An eficient exploration basis for learning. Journal of Machine Learning Research, 17(119):1–34, 2016. URL https://jmlr.org/papers/v17/hazan16a.html.

[18] Roger A. Horn and Charles R. Johnson. Matrix Analysis. Cambridge University Press, 2 edition, 2012. doi: 10.1017/CBO9781139020411. URL https://doi.org/10.1017/CBO9781139020411.

[19] Shinji Ito, Daisuke Hatano, Hanna Sumita, Kei Takemura, Takuro Fukunaga, Naonori Kakimura, and Ken-ichi Kawarabayashi. Improved regret bounds for bandit combinatorial optimization. In Advances in Neural Information Processing Systems, volume 32, pages 12027–12036. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/ paper/2019/hash/d5b3d8dadd770c460b1cde910a711987-Abstract.html.

[20] Shinji Ito, Daisuke Hatano, Hanna Sumita, Kei Takemura, Takuro Fukunaga, Naonori Kakimura, and Ken-ichi Kawarabayashi. Oracle-eficient algorithms for online linear optimization with bandit feedback. In Advances in Neural Information Processing Systems, volume 32, pages 10589–10598. Curran Associates, Inc., 2019. URL https://proceedings. neurips.cc/paper/2019/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html.

[21] Jack Kiefer. Optimum experimental designs. Journal of the Royal Statistical Society: Series B (Methodological), 21(2):272–304, 1959. doi: 10.1111/j.2517-6161.1959.tb00338.x. URL https://doi.org/10.1111/j.2517-6161.1959.tb00338.x.

[22] Jack Kiefer and Jacob Wolfowitz. The equivalence of two extremum problems. Canadian Journal of Mathematics, 12:363–366, 1960. doi: 10.4153/CJM-1960-030-4. URL https://doi.org/10.4153/CJM-1960-030-4.

[23] Andreas Kontogiannis, Vasilis Pollatos, Gabriele Farina, Panayotis Mertikopoulos, and Ioannis Panageas. Eficient kernelized learning in polyhedral games beyond full information: From Colonel Blotto to congestion games. In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/hash/d6b452383b070d5ccf042d8d25e0cdb2-Abstract-Conference.html. NeurIPS 2025 Main Conference Track.

[24] Alex Kulesza and Ben Taskar. k-DPPs: Fixed-size determinantal point processes. In Proceedings of the 28th International Conference on Machine Learning, pages 1193–1200. Omnipress, 2011. URL https://icml.cc/2011/papers/611\_icmlpaper.pdf.

[25] Chung-Wei Lee, Haipeng Luo, Chen-Yu Wei, and Mengxiao Zhang. Bias no more: High-probability data-dependent regret bounds for adversarial bandits and MDPs. In Advances in Neural Information Processing Systems, volume 33, pages 15522–15533. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/hash b2ea5e977c5fc1ccfa74171a9723dd61-Abstract.html.

[26] Arnab Maiti, Zhiyuan Fan, Kevin Jamieson, Lillian J. Ratlif, and Gabriele Farina. Eficient near-optimal algorithm for online shortest paths in directed acyclic graphs with bandit feedback against adaptive adversaries. In Nika Haghtalab and Ankur Moitra, editors, Proceedings of the Thirty-Eighth Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 3881–3932. PMLR, 2025. URL https://proceedings.mlr.press/v291/maiti25a.html.

[27] H. Brendan McMahan and Avrim Blum. Online geometric optimization in the bandit setting against an adaptive adversary. In Proceedings of the 17th Annual Conference on Learning Theory, volume 3120 of Lecture Notes in Computer Science, pages 109–123. Springer, 2004. doi: 10.1007/978-3-540-27819-1 8. URL https://doi.org/10.1007/978-3-540-27819-1\_8.

[28] Shinsaku Sakaue, Masakazu Ishihata, and Shin-ichi Minato. Eficient bandit combinatorial optimization algorithm with zero-suppressed binary decision diagrams. In Amos Storkey and Fernando Perez-Cruz, editors, Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics, volume 84 of Proceedings of Machine Learning Research, pages 585–594. PMLR, 2018. URL https://proceedings.mlr.press/v84/sakaue18a.html.

[29] Julian Zimmert and Tor Lattimore. Return of the bias: Almost minimax optimal high probability bounds for adversarial linear bandits. In Po-Ling Loh and Maxim Raginsky, editors, Proceedings of the Thirty-Fifth Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 3285–3312. PMLR, 2022. URL https://proceedings.mlr.press/v178/zimmert22b.html.