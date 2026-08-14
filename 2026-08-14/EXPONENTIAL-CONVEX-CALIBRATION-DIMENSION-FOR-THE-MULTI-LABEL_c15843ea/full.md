# EXPONENTIAL CONVEX CALIBRATION DIMENSION FOR THE MULTI-LABEL JACCARD MEASURE

PREPRINT

Mingyuan Zhang Independent Researcher myz@alumni.upenn.edu

## ABSTRACT

The per-instance Jaccard score, or intersection over union (IoU), is standard in multi-label classification and binary segmentation. With s labels, its loss matrix has $2 ^ { s }$ outcomes and reports. Under the convention Jac $( \varnothing , \varnothing ) = 1$ , we prove that the Jaccard score, shifted-loss, and ordinary loss matrices are nonsingular and that the loss columns have affine dimension $2 ^ { s } - 1$ . The proof combines a finite MinHash Gram representation with Boolean Möbius inversion.

For exact calibration, we prove

$$
2 ^ { s - 1 } \leq \mathrm { C C d i m } ( L ^ { \mathrm { J a c } } ) \leq 2 ^ { s } - 1 .
$$

The lower bound uses a factorially weighted distribution with $2 ^ { s - 1 } + 1$ supported outcomes and Bayesoptimal reports. Consequently, every exactly calibrated convex surrogate requires exponentially many prediction coordinates.

We also give two polynomial-dimensional approximation guarantees with explicit regret transfers. A new $F _ { 1 }$ -to-Jaccard transfer turns an existing $\bar { ( s ^ { 2 } + 1 ) }$ )-dimensional $F _ { 1 }$ surrogate into a polynomial-time rule with asymptotic Jaccard regret at most $3 - 2 \sqrt { 2 }$ . For any $\alpha > 0$ and $0 < \rho < 1$ , a MinHash squareloss surrogate attains Jaccard-regret floor α uniformly over arbitrary conditional label distributions. With probability at least $1 - \rho ,$ the direct construction has dimension $O ( ( s ^ { 2 } + s \log ( 1 / \rho ) ) / \alpha ^ { 2 } )$ while a signed variant has dimension $O ( ( s + \log ( 1 / \rho ) ) / \alpha ^ { 2 } )$ . Thus zero-regret calibration requires exponential dimension, whereas every fixed additive regret tolerance admits polynomial prediction dimension.

Keywords multi-label classification · Jaccard loss · intersection over union · convex calibration dimension · loss-matrix rank · approximate consistency · MinHash random features · F-measure

## 1 Introduction

In multi-label classification, an outcome and a prediction are subsets of a ground set of s labels. The Jaccard score compares them by the size of their intersection divided by the size of their union. The same quantity is widely called intersection over union (IoU) in image segmentation (Berman et al., 2018). Its dependence on the entire predicted and true sets makes the instance-wise loss nondecomposable across labels. We study the decision-theoretic setting in which this per-instance ratio is averaged under a conditional label distribution, rather than a population-utility setting in which confusion counts are averaged before a ratio is formed.

The output space contains 2<sup>s</sup> sets, but an exponential output space alone does not determine the dimension required by a statistically consistent convex surrogate. Some structured losses admit low-dimensional affine representations or calibrated surrogates even when their report spaces are exponential. Convex calibration dimension formalizes the smallest Euclidean dimension in which an exactly calibrated convex surrogate can exist (Ramaswamy and Agarwal, 2012, 2016).

Our contributions are threefold. First, we determine the exact ranks of the Jaccard score, shifted-loss, and ordinary loss matrices, together with the affine dimension of the loss columns. The nonempty score matrix is known to be strictly positive definite (Bouchard et al., 2013); we give a self-contained finite proof tailored to the power set using MinHash and Boolean Möbius inversion. Under our empty-set convention, all three matrix ranks equal $\bar { 2 ^ { s } }$ , and the affine dimension is $2 ^ { s } - 1$ . The resulting affine-dimension bound gives an exactly calibrated surrogate in $2 ^ { s } - 1$ dimensions, although matrix rank alone cannot lower-bound arbitrary nonlinear convex surrogates.

Second, we prove the exponential bounds

$$
2 ^ { s - 1 } \leq \mathrm { C C d i m } ( L ^ { \mathrm { J a c } } ) \leq 2 ^ { s } - 1 .
$$

For the lower bound, fix one core label and assign each outcome containing d optional labels weight proportional to $1 / d !$ . A combinatorial identity makes all reports containing the core label tie, while adding the core label strictly improves every nonempty report that omits it. Mixing with the empty outcome makes the empty report tie as well. The supported outcomes and Bayes-optimal reports then form the same $( 2 ^ { s - 1 } + 1 )$ -element family. Its principal score submatrix is nonsingular, making the relevant two-sided feasible subspace trivial and yielding the lower bound. Hence the convex calibration dimension for exact calibration is $\Theta ( 2 ^ { s } )$

Third, we give two complementary polynomial-dimensional approximation guarantees. Pointwise, $\mathrm { J a c } = F _ { 1 } / ( 2 - F _ { 1 } )$ but expectation does not commute with this nonlinear transformation, so $F _ { 1 } \cdot$ - and Jaccard-optimal reports need not agree. Under the same empty-set convention, the multi-label $F _ { 1 }$ loss has convex calibration dimension $\ \mathbf { \bar { \Theta } } _ { \mathbf { \Theta } } ( s ^ { 2 } )$ (Zhang, 2026), and quadratic-dimensional convex calibrated surrogates for $F _ { 1 }$ are known (Nowak et al., 2019; Zhang et al., 2020). Our regret transfer turns one such surrogate into a polynomial-time Jaccard rule whose regret is at most $3 - 2 { \sqrt { 2 } }$ plus a term controlled by its $F _ { 1 }$ regret.

Our second approximation route uses MinHash directly. For every $\alpha > 0$ and $0 < \rho < 1$ , a finite MinHash feature map uniformly approximates the entire Jaccard score matrix with probability at least $1 - \rho .$ Regressing the conditional feature mean with a convex square loss gives an α-approximately consistent surrogate: vanishing surrogate regret guarantees asymptotic Jaccard regret at most α. A signed MinHash variant has dimension $O ( ( s + \log ( \bar { 1 } / \rho ) ) / \bar { \alpha } ^ { 2 } )$ These are prediction-dimension guarantees, not efficient-decoding results: the exact link may still maximize over all $2 ^ { s }$ reports, while a τ-approximate decoder increases the regret floor by at most τ. There is no conflict with the exact-calibration lower bound, because these constructions allow a positive tolerance and their dimension diverges as α ↓ 0.

## 2 Setup and calibration background

Fix an integer $s \geq 1$ . Let $[ s ] = \{ 1 , . . . , s \} , y = 2 ^ { [ s ] }$ , and $N = | y | = 2 ^ { s }$ . Outcomes and reports are denoted by $A , B \in \mathcal { V }$ . For any finite index set I, let $\mathbf { 1 } _ { I } \in \mathbb { R } ^ { I }$ denote the all-ones vector, and let $\delta _ { A }$ denote the point mass at $A \in { \dot { \mathcal { y } } }$ Define the Jaccard score by

$$
\operatorname { J a c } ( A , B ) = { \left\{ \begin{array} { l l } { { \frac { | A \cap B | } { | A \cup B | } } , } & { A \cup B \neq \varnothing , } \\ { 1 , } & { A = B = \varnothing . } \end{array} \right. }\tag{1}
$$

Let $S \in \mathbb { R } ^ { N \times N }$ be the score matrix, $S _ { A , B } = \operatorname { J a c } ( A , B )$ , let $U = \mathbf { 1 } _ { N } \mathbf { 1 } _ { N } ^ { \top }$ , and let

$$
L : = L ^ { \mathrm { J a c } } = U - S\tag{2}
$$

be the Jaccard loss matrix. Thus $L - U = - S$

For a vector x indexed by a finite set I and $T \subseteq I$ , x<sub>T</sub> denotes its restriction to T. For a matrix M, $M _ { T , R }$ denotes the corresponding submatrix and $M _ { \cdot , j }$ its column indexed by $j .$ . For a finite family $V \subseteq \mathbb { R } ^ { r }$ , let $\operatorname { a f f d i m } ( V ) = \dim \operatorname { a f f } ( V )$ For a matrix $\mathbf { \bar { M } } = [ m _ { 1 } \ \cdot \cdot \cdot \ m _ { k } ]$ , define its column-affine dimension by

$$
\operatorname { a f f d i m } ( M ) = \operatorname { a f f d i m } \{ m _ { 1 } , \dots , m _ { k } \} .
$$

Let $\Delta _ { N } = \{ p \in \mathbb { R } _ { + } ^ { N } : \mathbf { 1 } _ { N } ^ { \top } p = 1 \}$ , and write supp $( p ) = \{ A \in \mathcal { Y } : p _ { A } > 0 \}$ for $p \in \Delta _ { N }$ . The conditional risk of report B is $p ^ { \top } L . . _ { B } .$ , and

$$
{ \mathrm { o p t } } _ { L } ( p ) = { \mathrm { a r g } } \operatorname* { m i n } _ { B \in \mathcal { V } } p ^ { \top } L . , _ { B }
$$

is the set of Bayes-optimal reports. Its trigger probability set is

$$
\begin{array} { r } { Q _ { B } ^ { L } = \left\{ p \in \Delta _ { N } : p ^ { \top } L . , _ { B } \leq p ^ { \top } L . , _ { B ^ { \prime } } \mathrm { ~ f o r ~ e v e r y ~ } B ^ { \prime } \in \mathcal { V } \right\} . } \end{array}\tag{3}
$$

For completeness, let $\mathcal { C } \subseteq \mathbb { R } ^ { d }$ be convex, let $\psi : { \mathcal { C } } \to \mathbb { R } _ { + } ^ { N }$ have convex coordinate functions, and let pred : $\mathcal { C }  \mathcal { V }$ be a link. The pair $( \psi , \mathrm { p r e d } )$ is L-calibrated if, for every $p \in \Delta _ { N }$

$$
\operatorname* { i n f } _ { \boldsymbol { u } \in \mathcal { C } : \mathrm { { \scriptsize ~ p r e d } } ( \boldsymbol { u } ) \notin \mathrm { { o p t } } _ { L } ( \boldsymbol { p } ) } p ^ { \top } \boldsymbol { \psi } ( \boldsymbol { u } ) > \operatorname* { i n f } _ { \boldsymbol { u } \in \mathcal { C } } p ^ { \top } \boldsymbol { \psi } ( \boldsymbol { u } ) .
$$

The convex calibration dimension CCdim(L) is the smallest such d (Ramaswamy and Agarwal, 2016, Definitions 1 and 10).

We use two general results. First,

$$
\mathrm { C C d i m } ( L ) \leq \mathrm { a f f i m } ( L ) .\tag{4}
$$

Second, for $p \in Q _ { B } ^ { L } ,$

$$
\mathrm { C C d i m } ( L ) \geq \| p \| _ { 0 } - \mu _ { Q _ { B } ^ { L } } ( p ) - 1 ,\tag{5}
$$

where $\| p \| _ { 0 } = | \operatorname { s u p p } ( p ) |$ . To define $\mu ,$ for $Q \subseteq \Delta _ { N }$ and $p \in Q$ , let

$$
\begin{array} { r } { \mathrm { { d i r } } _ { Q } ( p ) = \left\{ v \in \mathbb { R } ^ { N } : p + \varepsilon v \in Q \mathrm { ~ f o r ~ a l l ~ s u f f i c i e n t l y ~ s m a l l ~ } \varepsilon > 0 \right\} \mathrm { , ~ } } \end{array}
$$

and set

$$
\operatorname * { l i m } _ { Q } ( p ) = \mathrm { d i r } _ { Q } ( p ) \cap \bigl ( - \mathrm { d i r } _ { Q } ( p ) \bigr ) , \qquad \mu _ { Q } ( p ) = \mathrm { d i m } \operatorname * { l i n } _ { Q } ( p ) .
$$

Equations (4) and (5) are Theorems 12 and 16 of Ramaswamy and Agarwal (2016).

## 3 Related work

Jaccard matrices and MinHash. The collision probability underlying MinHash is the Jaccard similarity (Broder, 1997). Following an earlier positive-semidefiniteness result of Gower (1971), Bouchard et al. (2013) proved strict positive definiteness of the complete nonempty Jaccard index matrix by an infinite positive-semidefinite decomposition. We give an alternative finite proof based on MinHash and Boolean Möbius inversion, then use the result to analyze the associated loss matrix. We also subsample the same Gram representation to obtain a uniformly accurate approximate factorization and elicit its conditional mean with a convex square loss.

Instance-wise Jaccard prediction. Dembczynski et al.´ (2012) studied decision-theoretic risk minimization for several multi-label losses and highlighted the difficulty of exact Jaccard-risk minimization from an arbitrary conditional joint distribution. For an empirical distribution, Bayes-optimal Jaccard prediction is the Jaccard-median problem; Chierichetti et al. (2010) established its computational hardness and gave approximation algorithms. Although Jaccard is a pointwise monotone transform of the $F _ { 1 }$ score, the corresponding expected-score maximizers need not agree (Waegeman et al., 2014). If δ(P) denotes the Bayes-optimal expected $F _ { 1 }$ score, Waegeman et al. (2014, Theorem 3) bound the Jaccard regret of an $F _ { 1 }$ -optimal report by $\bar { 1 - \delta ( P ) / 2 } ;$ ; this is an approximation guarantee, not exact Jaccard consistency. They take $F _ { 1 } ( \varnothing , \varnothing ) \stackrel { \textstyle = } { = } 1$ but Jac $( \partial , \partial ) = 0$ , whereas our main convention aligns the two empty-set values. Under the aligned convention, we derive a convention-matched uniform transfer and combine it with the quadratic-dimensional $F _ { 1 }$ surrogate of Zhang et al. (2020).

This is the setting considered here: the Jaccard loss is evaluated separately on each outcome–report pair and then averaged under a conditional distribution. It differs from population-confusion or expected-utility formulations of micro- and instance-averaged linear-fractional metrics, for which threshold and calibrated-utility methods are available (Koyejo et al., 2015; Bao and Sugiyama, 2020).

Convex surrogates for IoU. The Lovász hinge and Lovász–Softmax losses provide tractable objectives motivated by submodular losses and IoU (Yu and Blaschko, 2015; Berman et al., 2018). These objectives have been influential in segmentation, but empirical usefulness does not imply calibration for the finite instance-wise loss studied here. Indeed, Finocchiaro et al. (2022) showed that the Lovász hinge is inconsistent for its intended structured target unless the underlying set function is modular. Dai and Li (2023) derived an IoU-calibrated ranking-based plug-in rule under conditional independence of the component labels. Their exact guarantee relies on conditional independence. Our exact lower bound allows arbitrary conditional dependence and concerns all convex calibrated surrogates, independently of smoothness or polyhedrality. Our MinHash construction gives a distribution-free approximate guarantee without an independence assumption.

Prediction dimension. Convex calibration dimension was introduced by Ramaswamy and Agarwal (2012) and developed using trigger-set geometry by Ramaswamy and Agarwal (2016). For comparison, under the aligned emptyset convention, Zhang (2026) proved that the $F _ { 1 }$ score, shifted-loss, and ordinary loss matrices have rank $s ^ { 2 } - s + \bar { 2 } .$ that the loss columns have affine dimension $s ^ { 2 } - s + 1$ , and that $\mathrm { C C d i m } ( L ^ { F _ { 1 } } ) ^ { \bullet } = \Theta ( s ^ { 2 } )$ . Thus exact $F _ { 1 }$ prediction dimension is quadratic, whereas the Jaccard bounds proved here are exponential. The present exact lower bound applies the feasible-subspace theorem of Ramaswamy and Agarwal (2016) to a Jaccard-specific family of tied Bayes reports. We are not aware of a previous exponential prediction-dimension lower bound for the instance-wise Jaccard loss. Convex calibration dimension is an exact notion and does not itself lower-bound the dimension needed for a fixed additive target-regret tolerance.

## 4 Exact rank and affine dimension

$\mathcal { Y } _ { + } = \mathcal { Y } \setminus \{ \emptyset \}$ and let $K = S _ { \mathcal { V } + , \mathcal { V } _ { + } }$ be the score matrix restricted to nonempty sets.

Lemma 4.1 (Strict positive definiteness). For every $s \geq 1$ , the matrix K is positive definite. Consequently, rank $( K ) =$ $2 ^ { s } - 1$

The proof is given in Appendix A.1. It writes K as a MinHash Gram matrix and uses Boolean Möbius inversion to show that the Gram features have no nontrivial common null vector.

Theorem 4.2 (Exact Jaccard ranks). For every $s \geq 1$ , under the convention in (1),

$$
\operatorname { r a n k } ( S ) = \operatorname { r a n k } ( L - U ) = \operatorname { r a n k } ( L ) = 2 ^ { s } .
$$

Moreover,

$$
\mathrm { a f f d i m } ( L ) = 2 ^ { s } - 1 .
$$

Proof sketch. Ordering the empty set first gives $S = \mathrm { d i a g } ( 1 , K )$ , so Lemma 4.1 makes $S$ and $L - U = - S$ nonsingular. Let $e \in \mathbb { R } ^ { 2 ^ { s } - 1 }$ indicate the singleton sets. For every nonempty A,

$$
( K e ) _ { A } = \sum _ { j = 1 } ^ { s } \mathrm { J a c } ( A , \{ j \} ) = 1 ,
$$

so $K e = { \bf 1 } _ { N - 1 }$ . The block kernel equations for L then force every null vector to vanish, proving that L is nonsingular. Finally, the $2 ^ { s }$ score columns are linearly independent and hence have affine dimension $2 ^ { s } - 1$ ; the invertible affine map $x \mapsto \mathbf { 1 } _ { N } - x$ transfers this dimension to the loss columns. The full proof is in Appendix A.2.

Corollary 4.3 (Calibration-dimension upper bound). For every $s \geq 1$

$$
\mathrm { C C d i m } ( L ^ { \mathrm { J a c } } ) \leq 2 ^ { s } - 1 .
$$

This follows immediately from Theorem 4.2 and (4). Equivalently, one may estimate an arbitrary $2 ^ { s }$ -class conditional distribution in $2 ^ { s } - 1$ coordinates and decode by minimizing the estimated Jaccard risk.

Remark 4.4 (Alternative empty-set convention). Ifinstead $\mathrm { J a c } ( \emptyset , \emptyset ) = 0$ , then

$$
\operatorname { r a n k } ( S ) = \operatorname { r a n k } ( L - U ) = 2 ^ { s } - 1 , \qquad \operatorname { r a n k } ( L ) = 2 ^ { s } , \qquad \operatorname { a f f d i m } ( L ) = 2 ^ { s } - 1 .
$$

The verification is given in Appendix E.

## 5 An exponential calibration-dimension lower bound

The lower-bound witness uses a factorial balancing identity. For an integer $n \geq 0 .$ , write $[ n ] = \{ 1 , \dots , n \}$ , with $[ 0 ] = \emptyset$

Lemma 5.1 (Factorial balancing). For every $C \subseteq [ n ]$

$$
\sum _ { D \subseteq [ n ] } { \frac { 1 + | C \cap D | } { ( 1 + | C \cup D | ) | D | ! } } = \sum _ { d = 0 } ^ { n } { \binom { n } { d } } { \frac { 1 } { ( d + 1 ) ! } } .\tag{6}
$$

In particular, the left-hand side is independent ofC.

The proof is a finite binomial calculation and is given in Appendix B.

Theorem 5.2 (Exponential CC-dimension bounds). For every $s \geq 1$

$$
2 ^ { s - 1 } \leq \mathrm { C C d i m } ( L ^ { \mathrm { J a c } } ) \leq 2 ^ { s } - 1 .\tag{7}
$$

Consequently, CCdim $( L ^ { \mathrm { J a c } } ) = \Theta ( 2 ^ { s } )$

Proof sketch. Fix a core label 1 and put $O = [ s ] \setminus \{ 1 \} , { \mathrm { s o } } | O | = s - 1$ . Let

$$
\mathcal { U } = \{ \{ 1 \} \cup D : D \subseteq O \} .
$$

First define a full-support distribution q on U by assigning $\{ 1 \} \cup D$ probability proportional to $1 / | D | !$ . For a report $\{ 1 \} \cup C$ , its expected score under $q$ is the normalized left-hand side of (6); hence all $2 ^ { s - 1 }$ reports in U tie. Every nonempty report missing label 1 is pointwise improved by adding it, because all outcomes in U contain label 1.

Let $\kappa > 0$ be the common score of the reports in U. Mixing q with the empty outcome using weights $1 / ( 1 + \kappa )$ and $\kappa / ( 1 + \kappa )$ , respectively, makes the empty report tie with every report in U. These are exactly the Bayes-optimal reports, and the support is

$$
\mathcal { A } = \{ \emptyset \} \cup \mathcal { U } , \qquad | \mathcal { A } | = 2 ^ { s - 1 } + 1 .
$$

The active score submatrix indexed by A is $\mathrm { d i a g } ( 1 , S _ { \boldsymbol { U } , \boldsymbol { U } } )$ and is nonsingular by Lemma 4.1. Its columns therefore have affine dimension $| { \mathcal { A } } | - 1 = 2 ^ { s - 1 }$ . At the witness distribution, their difference span is the entire hyperplane orthogonal to the positive probability vector. Adding the normalization constraint leaves no nonzero two-sided feasible direction, so $\mu = 0$ . Substitution into (5) gives the lower bound $| { \mathcal { A } } | - 1 = 2 ^ { s - 1 }$ . Corollary 4.3 gives the upper bound. The detailed feasible-subspace argument is in Appendix C.

Remark 5.3 (Size of the remaining gap). The bounds in (7) differ by less than afactor oftwo. Determining the exact convex calibration dimension, or improving either constant, remains open. The theorem rules out every polynomialdimensional convex surrogate that is exactly calibrated uniformly over all conditional label distributions. Section 6 shows that the qualifier “exactly” is essential.

Remark 5.4 (Alternative empty-set convention). Under $\operatorname { J a c } ( \varnothing , \varnothing ) = 0 ;$ , the same factorial witness without the empty-outcome mixture yields

$$
2 ^ { s - 1 } - 1 \leq \mathrm { C C d i m } ( L ^ { \mathrm { J a c } } ) \leq 2 ^ { s } - 1 .
$$

Thus the exponential conclusion is unchanged. See Appendix E.

## 6 Polynomial-dimensional approximate surrogates

The exponential lower bound in Theorem 5.2 concerns exact calibration. We now allow a nonzero, distribution-free target-regret floor, combine a prior $F _ { 1 }$ surrogate with a new regret transfer, and construct a MinHash surrogate of polynomial prediction dimension.

We use both conditional and population regrets. For a score $G : \mathcal { V } \times \mathcal { V } \to [ 0 , 1 ]$ , a distribution $p \in \Delta _ { N }$ , and a report $B \in \mathcal { V }$ , define

$$
r _ { G } ( p , B ) = \operatorname* { m a x } _ { C \in \mathcal { V } } \sum _ { A \in \mathcal { V } } p _ { A } G ( A , C ) - \sum _ { A \in \mathcal { V } } p _ { A } G ( A , B ) .\tag{8}
$$

For a surrogate $\Psi : \mathcal { V } \times \mathbb { R } ^ { d } \to \mathbb { R } _ { + }$ , define its conditional regret by

$$
r _ { \Psi } ( p , u ) = \sum _ { A \in \mathcal { V } } p _ { A } \Psi ( A , u ) - \operatorname* { i n f } _ { v \in \mathbb { R } ^ { d } } \sum _ { A \in \mathcal { V } } p _ { A } \Psi ( A , v ) .\tag{9}
$$

For a distribution D on $\mathcal { X } \times \mathcal { V }$ , let $p _ { x }$ be the conditional distribution of the outcome given $X = x ,$ , and set

$$
\operatorname { R e g } _ { G } ( h ) = \mathbb { E } _ { X } { \big [ } r _ { G } ( p _ { X } , h ( X ) ) { \big ] } ,\tag{10}
$$

$$
\mathrm { R e g } _ { \Psi } ( f ) = \mathbb { E } _ { ( X , A ) \sim \mathcal { D } } [ \Psi ( A , f ( X ) ) ] - \operatorname* { i n f } _ { g } \mathbb { E } _ { ( X , A ) \sim \mathcal { D } } [ \Psi ( A , g ( X ) ) ] ,\tag{11}
$$

where the infimum is over all measurable $g : \mathcal { X }  \mathbb { R } ^ { d }$ . For $\alpha \geq 0$ , we call a surrogate–link pair α-approximately consistent for Jaccard if, for every D and every sequence $\left( f _ { n } \right)$

$$
\mathrm { R e g } _ { \Psi } ( f _ { n } ) \longrightarrow 0 \quad \Longrightarrow \quad \operatorname* { l i m } _ { n \to \infty } \mathrm { R e g } _ { \mathrm { J a c } } ( \mathrm { p r e d } \circ f _ { n } ) \le \alpha .\tag{12}
$$

For both surrogates below, the infimum in (11) decomposes pointwise, and hence

$$
\mathrm { R e g } _ { \Psi } ( f ) = \mathbb { E } _ { X } \left[ r _ { \Psi } ( p _ { X } , f ( X ) ) \right] .\tag{13}
$$

## 6.1 An $F _ { 1 }$ -proxy surrogate

Define the instance-wise $F _ { 1 }$ score using the same empty-set convention as in (1):

$$
F ( A , B ) = \left\{ \frac { 2 | A \cap B | } { | A | + | B | } , | A | + | B | > 0 , \right.\tag{14}
$$

Pointwise,

$$
\operatorname { J a c } ( A , B ) = g ( F ( A , B ) ) , \qquad g ( t ) = { \frac { t } { 2 - t } } .\tag{15}
$$

Proposition 6.1 $( F _ { 1 }$ -to-Jaccard regret transfer). Let $c _ { \star } = 3 - 2 \sqrt { 2 }$ and define $H : [ 0 , 1 ]  [ 0 , 1 ]$ by

$$
H ( r ) = \left\{ \begin{array} { l l } { c _ { \star } + r , } & { 0 \leq r \leq \sqrt { 2 } - 1 , } \\ { \displaystyle \frac { 2 r } { 1 + r } , } & { \sqrt { 2 } - 1 \leq r \leq 1 . } \end{array} \right.\tag{16}
$$

Then, for every distribution D and every classifier h : $: \mathcal { X }  \mathcal { Y } _ { \mathrm { ~ } }$

$$
\mathrm { R e g } _ { \mathrm { J a c } } ( h ) \leq H \big ( \mathrm { R e g } _ { F } ( h ) \big ) \leq c _ { \star } + \mathrm { R e g } _ { F } ( h ) .\tag{17}
$$

In particular, an $F _ { 1 }$ -Bayes classifier has Jaccard regret at most $c _ { \star } \approx 0 . 1 7 1 6$

The proof is given in Appendix D.1.

Let $( \Psi _ { F } , \mathrm { p r e d } _ { F } )$ denote the $( s ^ { 2 } + 1 )$ )-dimensional convex surrogate and polynomial-time link for $F _ { 1 }$ of Zhang et al. (2020). Their calibration and regret-transfer results imply that, for every distribution D and every sequence $\left( f _ { n } \right)$

$$
\mathrm { R e g } _ { \Psi _ { F } } ( f _ { n } ) \longrightarrow 0 \quad \Longrightarrow \quad \mathrm { R e g } _ { F } ( \mathrm { p r e d } _ { F } \circ f _ { n } ) \longrightarrow 0 ;
$$

see Zhang et al. (2020, Theorems 2 and 5). Proposition 6.1 therefore immediately gives

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname* { l p R e g } _ { \mathrm { J a c } } ( \mathrm { p r e d } _ { F } \circ f _ { n } ) \leq c _ { \star } .\tag{18}
$$

Thus their surrogate–link pair is $c _ { \star }$ -approximately consistent for Jaccard. We refer to Zhang et al. (2020) for the construction, decoder, and proof. Its quadratic order is necessary for exact $F _ { 1 }$ calibration (Zhang, 2026); after the transfer above, however, it provides only a constant-floor approximate guarantee for Jaccard.

Proposition 6.1 is a convention-matched refinement of Waegeman et al. (2014, Theorem 3). For $\delta ( p ) \ =$ max<sub>C</sub> $\mathbb { E } _ { p } [ F ( A , C ) ]$ ], their coarse comparison gives an $F _ { 1 }$ -optimal report $B _ { F }$ the bound $r _ { \mathrm { J a c } } ( p , B _ { F } ) \leq 1 - \delta ( p ) / 2$ Under the aligned empty-set convention, the exact identity Jac $= g ( \bar { F } )$ and Jensen’s inequality instead $\mathrm { g i v e }$ , for any report B with $F _ { 1 }$ regret r,

$$
r _ { \mathrm { J a c } } ( p , B ) \leq \delta ( p ) - g ( \delta ( p ) - r ) \leq H ( r ) .
$$

In particular,

$$
r _ { \mathrm { J a c } } ( p , B _ { F } ) \leq \frac { \delta ( p ) ( 1 - \delta ( p ) ) } { 2 - \delta ( p ) } \leq 3 - 2 \sqrt { 2 } ,
$$

whereas the earlier bound is at least $1 / 2 .$ Thus the proposition both sharpens the exact-optimizer guarantee and covers approximate $F _ { 1 }$ optimization; concavity of H supplies the population transfer used above. Because $F _ { 1 }$ - and Jaccard-optimal reports need not agree, the resulting guarantee remains approximate rather than exact.

## 6.2 A MinHash random-feature surrogate

Write a permutation of [s] as $\pi = ( \pi ( 1 ) , \ldots , \pi ( s ) )$ . For every nonempty $A \subseteq [ s ]$ , its MinHash value is the first element of A in this order:

$$
m _ { \pi } ( A ) = \pi ( k _ { A } ) , \qquad k _ { A } = \operatorname* { m i n } \{ k \in [ s ] : \pi ( k ) \in A \} .
$$

Introduce an additional symbol ⊥ and extend this map to all $A \in \mathcal { V }$ by setting

$$
{ \overline { { m } } } _ { \pi } ( A ) = { \left\{ \begin{array} { l l } { m _ { \pi } ( A ) , } & { A \neq \varnothing , } \\ { \perp , } & { A = \varnothing , } \end{array} \right. }
$$

Under the convention $\mathrm { J a c } ( \emptyset , \emptyset ) = 1$ , a uniformly random permutation then satisfies the MinHash identity for every $A , B \in \mathcal { V }$

$$
\mathrm { J a c } ( A , B ) = \mathrm { P r } ( { \overline { { m } } } _ { \pi } ( A ) = { \overline { { m } } } _ { \pi } ( B ) ) .\tag{19}
$$

Lemma 6.2 (Uniform finite-sample MinHash approximation). Let $\pi _ { 1 } , \ldots , \pi _ { M }$ be independent uniformly random permutations $o f [ s ]$ , and let $\{ e _ { j } : \dot { j } \in [ s ] \cup \{ \perp \} \}$ be the standard basis $o f \mathbb { R } ^ { s + 1 }$ . Define

$$
\Phi ( A ) = \frac { 1 } { \sqrt { M } } \left( e _ { \overline { { m } } _ { \pi _ { 1 } } ( A ) } , \ldots , e _ { \overline { { m } } _ { \pi _ { M } } ( A ) } \right) \in \mathbb { R } ^ { M ( s + 1 ) }\tag{20}
$$

and

$$
\widetilde { S } ( A , B ) = \left. \Phi ( A ) , \Phi ( B ) \right. = \frac { 1 } { M } \sum _ { r = 1 } ^ { M } { \bf 1 } \{ \overline { { { m } } } _ { \pi _ { r } } ( A ) = \overline { { { m } } } _ { \pi _ { r } } ( B ) \} .\tag{21}
$$

For every η $\eta > 0 _ { ; }$

$$
\operatorname* { P r } \biggr ( \operatorname* { m a x } _ { A , B \in \mathcal { V } } | \widetilde { S } ( A , B ) - \operatorname { J a c } ( A , B ) | > \eta \biggr ) \leq 2 4 ^ { s } \exp ( - 2 M \eta ^ { 2 } ) .\tag{22}
$$

Consequently, for $0 < \rho < 1$ , the uniform error is at most η with probability at least $1 - \rho$ whenever

$$
M \geq \frac { 2 s \log 2 + \log ( 2 / \rho ) } { 2 \eta ^ { 2 } } .\tag{23}
$$

The proof is given in Appendix D.2.

Fix a realization of the feature map and define the convex square-loss surrogate and link by

$$
\Psi _ { M } ( A , u ) = \| u - \Phi ( A ) \| _ { 2 } ^ { 2 } , \qquad u \in \mathbb { R } ^ { M ( s + 1 ) } ,\tag{24}
$$

$$
\operatorname { p r e d } _ { M } ( u ) \in \arg \operatorname* { m a x } _ { B \in \mathcal { V } } \langle u , \Phi ( B ) \rangle ,\tag{25}
$$

using any fixed rule to break ties.

Theorem 6.3 (MinHash approximate consistency). On the uniform-approximation event in Lemma 6.2, for every $p \in \Delta _ { N }$ and every $u \in \mathbb { R } ^ { \hat { M } ( s + 1 ) }$

$$
r _ { \mathrm { J a c } } ( p , \mathrm { p r e d } _ { M } ( u ) ) \leq 2 \eta + \sqrt { 2 r _ { \Psi _ { M } } ( p , u ) } .\tag{26}
$$

The corresponding population guarantee is

$$
\mathrm { R e g } _ { \mathrm { J a c } } ( \mathrm { p r e d } _ { M } \circ f ) \leq 2 \eta + \sqrt { 2 \mathrm { R e g } _ { \Psi _ { M } } ( f ) } .\tag{27}
$$

In particular, for every $\alpha > 0 ;$ , taking $\eta = \alpha / 2$ makes the pair α-approximately consistent. With probability at least $1 - \rho ,$ this is achieved in dimension

$$
d _ { \mathrm { M H } } = M ( s + 1 ) , \qquad M = \left\lceil \frac { 2 \bigl ( 2 s \log 2 + \log ( 2 / \rho ) \bigr ) } { \alpha ^ { 2 } } \right\rceil ,\tag{28}
$$

and hence

$$
d _ { \mathrm { M H } } = { \cal O } \left( \frac { s ^ { 2 } + s \log ( 1 / \rho ) } { \alpha ^ { 2 } } \right) .
$$

The probabilistic construction also proves the existence of a fixed deterministic feature map with this guarantee.

The proof is given in Appendix D.3.

For the compressed construction, fix an integer $d \geq 1$ , draw independent uniform permutations $\pi _ { 1 } , \ldots , \pi _ { d } ,$ , and, independently of the permutations and of one another, draw Rademacher variables $\xi _ { r , j } \in \{ - 1 , 1 \}$ for $r \in [ d ]$ and $j \in [ s ] \cup \{ \bot \}$ . Define

$$
\Phi _ { \pm } ( A ) = \frac { 1 } { \sqrt { d } } \left( \xi _ { 1 , \overline { { { m } } } _ { \pi _ { 1 } } ( A ) } , \ldots , \xi _ { d , \overline { { { m } } } _ { \pi _ { d } } ( A ) } \right) \in \mathbb { R } ^ { d } ,\tag{29}
$$

and set

$$
\Psi _ { \pm } ( A , u ) = \| u - \Phi _ { \pm } ( A ) \| _ { 2 } ^ { 2 } ,\tag{30}
$$

$$
\operatorname { p r e d } _ { \pm } ( u ) \in \arg \operatorname* { m a x } _ { B \in \mathcal { V } } \langle u , \Phi _ { \pm } ( B ) \rangle ,\tag{31}
$$

using any fixed rule to break ties.

Corollary 6.4 (Rademacher-compressed approximate consistency). For every $\alpha > 0$ and $0 < \rho < 1$ , take the preceding random surrogate–link pair with $d = d _ { \pm }$ , where

$$
d _ { \pm } = \left\lceil \frac { 8 \left( 2 s \log 2 + \log ( 2 / \rho ) \right) } { \alpha ^ { 2 } } \right\rceil\tag{32}
$$

Then, with probability at least $1 - \rho$ over itsfeature construction, simultaneouslyfor every distribution D and every measurable $f ,$

$$
\begin{array} { r } { \mathrm { R e g } _ { \mathrm { J a c } } ( \mathrm { p r e d } _ { \pm } \circ f ) \leq \alpha + 2 \sqrt { \mathrm { R e g } _ { \Psi _ { \pm } } ( f ) } . } \end{array}\tag{33}
$$

Thus the pair is α-approximately consistent in dimension $O ( ( s + \log ( 1 / \rho ) ) / \alpha ^ { 2 } )$ . Fixing $\rho = 1 / 2$ and any successful realization proves the deterministic existence of such a pair in dimension $\scriptstyle { \dot { O } } ( s / \alpha ^ { 2 } )$

The proof is given in Appendix D.4.

Remark 6.5 (Prediction dimension versus decoding). The MinHash construction has polynomial prediction dimension, but the exact link in (25) still maximizes over all $2 ^ { s }$ reports. No polynomial-time decoding claim is implicit in Theorem 6.3. Ifa decoder returns $\widehat { B }$ satisfying

$$
\langle u , \Phi ( { \widehat { B } } ) \rangle \geq \operatorname* { m a x } _ { B \in \mathcal { V } } \langle u , \Phi ( B ) \rangle - \tau ,
$$

the argument in Appendix D.3 adds only τ to the regret bounds. The same argument applies to the signed construction. If a signed-feature decoder returns $\widehat { B }$ satisfying

$$
\langle u , \Phi _ { \pm } ( \widehat { B } ) \rangle \geq \operatorname* { m a x } _ { B \in \mathcal { V } } \langle u , \Phi _ { \pm } ( B ) \rangle - \tau ,
$$

then, on the uniform-approximation event used in Corollary 6.4,

$$
r _ { \mathrm { J a c } } ( p , \widehat { B } ) \leq 2 \eta + \tau + 2 \sqrt { r _ { \Psi _ { \pm } } ( p , u ) } .
$$

Consequently, with the choice $\eta = \alpha / 2$ in that corollary, the population boundfor the corresponding approximate link $\widehat { \mathrm { p r e d } } _ { \pm }$ becomes

$$
\begin{array} { r } { \mathrm { R e g } _ { \mathrm { J a c } } ( \widehat { \mathrm { p r e d } _ { \pm } \circ f } ) \leq \alpha + \tau + 2 \sqrt { \mathrm { R e g } _ { \Psi _ { \pm } } ( f ) } . } \end{array}
$$

Thus a fixed decoding error gives (α + τ)-approximate consistency; to retain an overall floor $\alpha ,$ choose the kernelapproximation and decoding budgets so that $2 \eta + \tau \leq \alpha$ . In contrast, the $F _ { 1 }$ proxy of Zhang et al. (2020) has polynomial-time decoding but, after the transfer in Proposition 6.1, retains the nonzero worst-case floor $c _ { \star }$

Remark 6.6 (Alternative empty-set convention). $I f \operatorname { J a c } ( \varnothing , \varnothing ) = 0 ,$ , define a convention-matched $F _ { 1 }$ score with $F ( \varnothing , \varnothing ) = 0 .$ . Then (15) and Proposition 6.1 remain valid; any surrogate calibratedfor this convention-matched score inherits the same transfer. For the MinHash construction, set $\Phi ( \boldsymbol { \mathcal { O } } ) = 0$ and use the ordinary nonempty-setfeatures in (20). Allfeature norms are then at most one, and the concentration and regret bounds remain unchanged; the direct feature dimension becomes Ms. The signed construction likewise remains valid after setting $\Phi _ { \pm } ( \alpha ) = 0$

For every fixed $\alpha > 0 ,$ , the MinHash constructions have polynomial dimension. There is no contradiction with Theorem 5.2: that theorem concerns the zero-floor case $\alpha = 0 ,$ whereas the dimensions above grow as the requested approximation floor tends to zero.

## 7 Conclusion

The instance-wise multi-label Jaccard loss has maximal matrix rank and affine dimension. A finite MinHash Gram representation and Boolean Möbius inversion establish the rank results, while a factorially weighted family with a trivial two-sided feasible subspace yields $2 ^ { s - 1 } \leq \mathrm { C C d i m } ( L ^ { \mathrm { J a c } } ) \leq 2 ^ { s } - 1$ . Thus exact calibration requires $\Theta ( { \bar { 2 } } ^ { s } )$ prediction coordinates, although the precise dimension remains open within a factor of two.

Approximation exposes an exactness–dimension tradeoff. Our $F _ { 1 }$ transfer, combined with the polynomial-time decoder of Zhang et al. (2020), gives asymptotic Jaccard regret at most $3 - 2 { \sqrt { 2 } }$ . For any $\alpha > 0$ , MinHash square-loss surrogates attain regret floor α in polynomial dimension; a signed variant achieves $\bar { O ( ( s + \log ( 1 / \rho ) ) / \alpha ^ { 2 } ) }$ dimensions with probability at least $1 - \rho .$ . These prediction-dimension results do not resolve inference complexity: the MinHash link may still search over $2 ^ { s }$ reports. Natural next steps are efficient links with arbitrarily small floors, matching approximate-dimension lower bounds, and closing the factor-of-two gap in $\mathrm { C C d i m } ( L ^ { \mathrm { J a c } } )$ .

## A Proofs for the exact-rank results

## A.1 Proof of Lemma 4.1

Proof of Lemma 4.1. For a permutation π of $[ s ]$ and a nonempty $A \subseteq [ s ]$ , let $m _ { \pi } ( A )$ be the first element of A under π. The MinHash identity (Broder, 1997) states that, for nonempty A, B,

$$
\operatorname { J a c } ( A , B ) = \operatorname* { P r } _ { \pi } { \big ( } m _ { \pi } ( A ) = m _ { \pi } ( B ) { \big ) } .\tag{34}
$$

Indeed, the first element of $A \cup B$ under a uniformly random permutation is uniform on $A \cup B$ , and the two minima agree exactly when that element lies in $A \cap B$

For every permutation π and $j \in [ s ]$ , define the column vector

$$
h _ { \pi , j } : = \left( \mathbf { 1 } \{ m _ { \pi } ( A ) = j \} \right) _ { A \in \mathcal { V } _ { + } } \in \mathbb { R } ^ { \mathcal { V } _ { + } } \cong \mathbb { R } ^ { N - 1 } .
$$

Thus $h _ { \pi , j } h _ { \pi , j } ^ { \top }$ is an $( N - 1 ) \times ( N - 1 )$ matrix indexed by pairs of nonempty sets. For every A, $B \in \mathcal { V } _ { + }$

$$
\begin{array} { r l } {  { [ \frac { 1 } { s ! } \sum _ { \pi } \sum _ { j = 1 } ^ { s } h _ { \pi , j } h _ { \pi , j } ^ { \top } ] _ { A , B } = \frac { 1 } { s ! } \sum _ { \pi } \sum _ { j = 1 } ^ { s } \mathbf { 1 } \{ m _ { \pi } ( A ) = j \} \mathbf { 1 } \{ m _ { \pi } ( B ) = j \} } } \\ & { = \operatorname* { P r } ( m _ { \pi } ( A ) = m _ { \pi } ( B ) ) = \operatorname { J a c } ( A , B ) = K _ { A , B } , } \end{array}
$$

where the sum over $j$ is the indicator that the two MinHashes agree. Consequently, Equation (34) gives the Gram representation

$$
K = \frac { 1 } { s ! } \sum _ { \pi } \sum _ { j = 1 } ^ { s } h _ { \pi , j } h _ { \pi , j } ^ { \top } ,
$$

Indeed, for every $x \in \mathbb { R } ^ { y _ { + } }$ ，

$$
x ^ { \top } K x = \frac { 1 } { s ! } \sum _ { \pi } \sum _ { j = 1 } ^ { s } x ^ { \top } h _ { \pi , j } h _ { \pi , j } ^ { \top } x = \frac { 1 } { s ! } \sum _ { \pi } \sum _ { j = 1 } ^ { s } \bigl ( h _ { \pi , j } ^ { \top } x \bigr ) ^ { 2 } \geq 0 ,
$$

so $K \succeq 0 . \mathrm { I f } x ^ { \top } K x = 0$ , this finite sum of nonnegative squares is zero, and therefore $h _ { \pi , j } ^ { \top } x = 0$ for every π and j. By the definition of $h _ { \pi , j }$

$$
h _ { \pi , j } ^ { \top } x = \sum _ { A \in \mathcal { V } _ { + } } \mathbf { 1 } \{ m _ { \pi } ( A ) = j \} x _ { A } = \sum _ { A : m _ { \pi } ( A ) = j } x _ { A } .
$$

Consequently,

$$
\sum _ { A : m _ { \pi } ( A ) = j } x _ { A } = 0 \qquad { \mathrm { f o r ~ e v e r y ~ } } \pi { \mathrm { ~ a n d ~ } } j .\tag{35}
$$

Fix $j$ and any $T \subseteq [ s ] \setminus \{ j \}$ . Choose a permutation in which every element of $[ s ] \setminus ( T \cup \{ j \} )$ precedes $j$ and every element of T follows j. The sets whose first element is j are exactly $\{ j \} \cup R$ with $R \subseteq T$ . Hence (35) becomes

$$
\sum _ { R \subseteq { T } } x _ { \{ j \} \cup R } = 0 \qquad { \mathrm { f o r } } \operatorname { e v e r y } T \subseteq [ s ] \backslash \{ j \} .
$$

Boolean Möbius inversion gives $x _ { \{ j \} \cup T } = 0$ for every T. Every nonempty set contains some $j ,$ , so $x = 0$ . Therefore $K \succ 0$ □

## A.2 Proof of Theorem 4.2

Proof of Theorem 4.2. Order the empty set first. Since its score with a nonempty set is zero,

$$
S = \binom { 1 } { 0 } \left( \begin{array} { c c } { { 0 } } & { { 0 } } \\ { { 0 } } & { { K } } \end{array} \right) .
$$

Lemma 4.1 makes S nonsingular, so rank $\mathrm { \hat { \rho } } ( S ) = \mathrm { r a n k } ( L - U ) = N$

Let $e \in \mathbb { R } ^ { N - 1 }$ be the indicator of the singleton subsets. For every nonempty A,

$$
( K e ) _ { A } = \sum _ { j = 1 } ^ { s } \operatorname { J a c } ( A , \{ j \} ) = \sum _ { j \in A } { \frac { 1 } { | A | } } = 1 .
$$

Thus $K e = { \bf 1 } _ { N - 1 }$ and $\mathbf { 1 } _ { N - 1 } ^ { \top } e = s .$

The loss has block form

$$
L = \binom { 0 } { \mathbf 1 _ { { N } - 1 } } \quad \mathbf 1 _ { { N } - 1 } ^ { \top } \mathbf 1 _ { { N } - 1 } ^ { } - { \boldsymbol K } \Big ) .\tag{36}
$$

Suppose $L ( \alpha , x ) ^ { \top } = 0$ . The first block equation gives $\mathbf { 1 } _ { N - 1 } ^ { \top } x = 0$ . The second then reduces to $K x = \alpha \mathbf { 1 } _ { N - 1 }$ . Since K is invertible and $K e = \mathbf { 1 } _ { N - 1 } , x = \alpha e$ . Therefore $0 = \mathbf { 1 } _ { N - 1 } ^ { \top } x = \alpha s .$ so $\alpha = 0$ and $x = 0$ . Hence rank $( L ) = N$

The N score columns are linearly independent, and therefore affinely independent, so $\mathrm { a f f d i m } ( S ) = N - 1$ . Applying the invertible affine map $z \mapsto \mathbf { 1 } _ { N } - z$ to each score column gives the corresponding loss column. Thus afdim $( L ) =$ $N - 1$ □

## B Proof of the factorial balancing identity

Proof of Lemma 5.1. Put $c = | C |$ . Every $D \subseteq [ n ]$ has the unique decomposition

$$
D = I \sqcup E , \qquad I = D \cap C \subseteq C , \qquad E = D \setminus C \subseteq [ n ] \setminus C .
$$

Let $i = | I |$ and $j = | E |$ . For this decomposition,

$$
| C \cap D | = i , \qquad | C \cup D | = | C | + | E | = c + j , \qquad | D | = i + j .
$$

For fixed i and j, there are $\binom { c } { i }$ choices for I and $\binom { n - c } { j }$ choices for E. Consequently, the left-hand side of (6) becomes

$$
\sum _ { j = 0 } ^ { n - c } \sum _ { i = 0 } ^ { c } { \binom { n - c } { j } } { \binom { c } { i } } { \frac { i + 1 } { ( c + j + 1 ) ( i + j ) ! } } .
$$

Factoring the terms that do not depend on i gives

$$
\sum _ { j = 0 } ^ { n - c } { \binom { n - c } { j } } { \frac { 1 } { c + j + 1 } } \sum _ { i = 0 } ^ { c } { \binom { c } { i } } { \frac { i + 1 } { ( i + j ) ! } } .\tag{37}
$$

We claim that

$$
{ \frac { 1 } { c + j + 1 } } \sum _ { i = 0 } ^ { c } { \binom { c } { i } } { \frac { i + 1 } { ( i + j ) ! } } = \sum _ { r = 0 } ^ { c } { \binom { c } { r } } { \frac { 1 } { ( j + r + 1 ) ! } } .\tag{38}
$$

To verify the claim, multiply its right-hand side by $c + j + 1$ and use $c + j + 1 = ( j + r + 1 ) + ( c - r )$ in each summand. This gives

$$
\begin{array} { l } { { ( c + j + 1 ) \displaystyle \sum _ { r = 0 } ^ { c } \binom { c } { r } \displaystyle \frac { 1 } { ( j + r + 1 ) ! } } } \\ { { = \displaystyle \sum _ { r = 0 } ^ { c } \binom { c } { r } \displaystyle \frac { j + r + 1 } { ( j + r + 1 ) ! } + \displaystyle \sum _ { r = 0 } ^ { c } \binom { c } { r } \displaystyle \frac { c - r } { ( j + r + 1 ) ! } } } \\ { { = \displaystyle \sum _ { r = 0 } ^ { c } \binom { c } { r } \displaystyle \frac { 1 } { ( j + r ) ! } + \displaystyle \sum _ { r = 0 } ^ { c - 1 } \binom { c } { r } \displaystyle \frac { c - r } { ( j + r + 1 ) ! } . } } \end{array}
$$

In the second sum, the $r = c$ term is zero. Using

$$
( c - r ) { \binom { c } { r } } = ( r + 1 ) { \binom { c } { r + 1 } }
$$

and then setting $i = r + 1$ , we obtain

$$
\sum _ { r = 0 } ^ { c - 1 } { \binom { c } { r } } { \frac { c - r } { ( j + r + 1 ) ! } } = \sum _ { i = 1 } ^ { c } i { \binom { c } { i } } { \frac { 1 } { ( j + i ) ! } } .
$$

Renaming r as i in the first sum and combining the two sums therefore yields

$$
\sum _ { i = 0 } ^ { c } { \binom { c } { i } } { \frac { 1 } { ( j + i ) ! } } + \sum _ { i = 1 } ^ { c } i { \binom { c } { i } } { \frac { 1 } { ( j + i ) ! } } = \sum _ { i = 0 } ^ { c } { \binom { c } { i } } { \frac { i + 1 } { ( j + i ) ! } } .
$$

This is the numerator on the left-hand side of (38); dividing by $c + j + 1$ proves the claim.

Substituting (38) into (37) yields

$$
\sum _ { j = 0 } ^ { n - c } \sum _ { r = 0 } ^ { c } { \binom { n - c } { j } } { \binom { c } { r } } { \frac { 1 } { ( j + r + 1 ) ! } } .
$$

Set $d = j + r$ . Since $0 \leq j \leq n - c$ and $0 \leq r \leq c .$ the new index runs from 0 to $n .$ . For fixed $d ,$ the denominator is $( d + 1 ) !$ , while the sum of the binomial coefficients over all admissible pairs $( j , r )$ is

$$
\sum _ { \stackrel { 0 \leq r \leq c } { 0 \leq d - r \leq n - c } } { \binom { c } { r } } { \binom { n - c } { d - r } } = { \binom { n } { d } }
$$

by Vandermonde’s identity. Hence the preceding double sum equals

$$
\sum _ { d = 0 } ^ { n } { \frac { 1 } { ( d + 1 ) ! } } \sum _ { { 0 \leq r \leq c \atop 0 \leq d - r \leq n - c } } { \binom { c } { r } } { \binom { n - c } { d - r } } = \sum _ { d = 0 } ^ { n } { \binom { n } { d } } { \frac { 1 } { ( d + 1 ) ! } } ,
$$

which is the right-hand side of (6).

## C Proof of the exponential lower bound

Proof of Theorem 5.2. The upper bound is Corollary 4.3. We prove the lower bound.

Step 1: the factorial distribution. Fix the core label 1, let $O = [ s ] \setminus \{ 1 \}$ and $n = s - 1$ , and set

$$
\mathcal { U } = \{ \{ 1 \} \cup D : D \subseteq O \} .
$$

Define

$$
Z _ { n } = \sum _ { d = 0 } ^ { n } { \binom { n } { d } } { \frac { 1 } { d ! } } , \qquad H _ { n } = \sum _ { d = 0 } ^ { n } { \binom { n } { d } } { \frac { 1 } { ( d + 1 ) ! } } , \qquad \kappa = { \frac { H _ { n } } { Z _ { n } } } > 0 .
$$

Let $q$ be supported on U with

$$
q _ { \{ 1 \} \cup D } = { \frac { 1 } { Z _ { n } | D | ! } } , \qquad D \subseteq O .\tag{39}
$$

Every coordinate in this support is positive.

For $C \subseteq O$ , let $A \sim q$ and write $A = \{ 1 \} \cup D$ . Lemma 5.1 gives

$$
\begin{array} { l l r } {  { \mathbb { E } _ { A \sim q } [ \operatorname { J a c } ( A , \{ 1 \} \cup C ) ] = \frac { 1 } { Z _ { n } } \sum _ { D \subseteq O } \frac { 1 + | C \cap D | } { ( 1 + | C \cup D | ) | D | ! } } } \\ & { } \\ & { } & { = \frac { H _ { n } } { Z _ { n } } = \kappa . } \end{array}
$$

Thus all reports in U tie under $q .$

If $B \neq \varnothing$ and $1 \notin B _ { i }$ , then for every outcome $A \in { \mathcal { U } }$

$$
\operatorname { J a c } ( A , B \cup \{ 1 \} ) - \operatorname { J a c } ( A , B ) = { \frac { 1 } { | A \cup B | } } > 0 .\tag{40}
$$

Hence $B$ is strictly suboptimal under $q .$ The empty report has score zero, whereas $\kappa > 0 .$ . Therefore

$$
\mathrm { o p t } _ { L } ( q ) = \mathcal { U } .\tag{41}
$$

Step 2: tie the empty report. Define

$$
p = \frac { \kappa } { 1 + \kappa } \delta _ { \infty } + \frac { 1 } { 1 + \kappa } q .\tag{42}
$$

The empty report has expected score $\kappa / ( 1 + \kappa )$ . Every report in U has the same score because its score on the empty outcome is zero and its score under q is κ. Equations (41) and (40) show that all remaining reports have strictly lower score. Thus

$$
\mathrm { o p t } _ { L } ( p ) = \mathcal { A } : = \{ \emptyset \} \cup \mathcal { U } , \qquad \mathrm { s u p p } ( p ) = \mathcal { A } , \qquad | \mathcal { A } | = 2 ^ { s - 1 } + 1 .\tag{43}
$$

Step 3: active-column dimension. The score submatrix with rows and columns in $\mathcal { A }$ is

$$
\begin{array} { r } { S _ { \mathcal { A } , \mathcal { A } } = \binom { 1 } { 0 } \begin{array} { c c } { 0 } \\ { S _ { \mathcal { U } , \mathcal { U } } } \end{array} } \end{array}
$$

The lower-right block is a principal submatrix of the positive-definite matrix in Lemma 4.1; hence $S _ { \mathbf { \mathcal { A } } , \mathbf { \mathcal { A } } }$ is nonsingular. Its $| { \cal A } |$ columns are linearly independent and therefore have affine dimension $| { \mathcal { A } } | - 1$

Fix $B _ { 0 } = \varnothing$ and define the active loss-difference space

$$
E = \mathrm { s p a n } \left\{ L _ { A , B } - L _ { A , B _ { 0 } } : B \in \mathcal { A } \right\} \subseteq \mathbb { R } ^ { A } .
$$

Loss differences are the negatives of score differences, so

$$
\dim E = | A | - 1 = 2 ^ { s - 1 } .\tag{44}
$$

All reports in $\mathcal { A }$ tie at $p ,$ so $p _ { \mathcal { A } } ^ { \top } z = 0$ for every $z \in E$ . Since $p _ { \mathcal { A } }$ is nonzero and (44) has codimension one,

$$
E = p _ { A } ^ { \perp } .\tag{45}
$$

Step 4: the two-sided feasible subspace. $\operatorname { A t } p ,$ the reports in A are exactly tied and every other report has a strict risk gap. It follows directly from the simplex constraints and the active trigger inequalities that

$$
\begin{array} { r } { \operatorname* { l i n } _ { Q _ { B _ { 0 } } ^ { L } } ( p ) = \left\{ v \in \mathbb { R } ^ { N } : v _ { A ^ { c } } = 0 , \quad \mathbf { 1 } _ { \mathcal { A } } ^ { \top } v _ { \mathcal { A } } = 0 , \quad v _ { \mathcal { A } } \bot E \right\} . } \end{array}\tag{46}
$$

Indeed, two-sided feasibility forces zero motion on the zero-probability coordinates, preserves total mass, and turns every active comparison into an equality. Conversely, any vector satisfying these conditions remains in the simplex and preserves the active equalities for sufficiently small motion in either direction; the strict inactive gaps remain strict.

By (45), the last condition in (46) gives $v _ { \mathcal { A } } = a p _ { \mathcal { A } }$ for some scalar $a .$ Normalization gives $0 = \mathbf { 1 } _ { \mathcal { A } } ^ { \top } v _ { \mathcal { A } } = a .$ because $\mathbf { 1 } _ { \mathcal { A } } ^ { \top } p _ { \mathcal { A } } = 1$ . Thus the lineality space is $\{ 0 \}$ and

$$
\mu _ { Q _ { B _ { 0 } } ^ { L } } ( p ) = 0 .
$$

Applying (5) and (43) gives

$$
\mathrm { C C d i m } ( L ^ { \mathrm { J a c } } ) \geq | { \cal A } | - 0 - 1 = 2 ^ { s - 1 } .
$$

## D Proofs for the polynomial-dimensional approximation results

## D.1 Proof of Proposition 6.1

ProofofProposition 6.1. Fix $p \in \Delta _ { N }$ and $B \in \mathcal { V }$ . Let A be a $\mathcal { D } \cdot$ -valued random variable with $\operatorname* { P r } ( A = D ) = p _ { D }$ , and write

$$
\mathbb { E } _ { p } [ \varphi ( A ) ] : = \sum _ { D \in \mathcal { D } } p _ { D } \varphi ( D )
$$

for any function $\varphi : \mathcal { V }  \mathbb { R }$ . Define the optimal conditional $F _ { 1 }$ score and the conditional $F _ { 1 }$ regret of $B$ by

$$
\delta = \operatorname* { m a x } _ { C \in \mathcal { V } } \mathbb { E } _ { p } [ F ( A , C ) ] , \qquad r = r _ { F } ( p , B ) ,
$$

respectively. By the definition of $r _ { F }$ in (8),

$$
\mathbb { E } _ { p } [ F ( A , B ) ] = \delta - r .
$$

Moreover, $0 \leq r \leq \delta \leq 1$ , since $F$ takes values in [0, 1]. Because $g ( t ) \leq t \mathrm { o n } \left[ 0 , 1 \right]$

$$
\operatorname* { m a x } _ { C \in \mathcal { V } } \mathbb { E } _ { p } [ \operatorname { J a c } ( A , C ) ] \leq \operatorname* { m a x } _ { C \in \mathcal { V } } \mathbb { E } _ { p } [ F ( A , C ) ] = \delta .
$$

The function g is convex. Jensen’s inequality and (15) therefore give

$$
\begin{array} { r } { \mathbb { E } _ { p } [ \mathrm { J a c } ( A , B ) ] = \mathbb { E } _ { p } [ g ( F ( A , B ) ) ] \geq g \big ( \mathbb { E } _ { p } [ F ( A , B ) ] \big ) = g ( \delta - r ) . } \end{array}
$$

Consequently,

$$
r _ { \mathrm { J a c } } ( p , B ) \leq \delta - g ( \delta - r ) .\tag{47}
$$

Set

$$
z = \delta - r = \mathbb { E } _ { p } [ F ( A , B ) ] .
$$

Then $z \geq 0 ,$ , and $z + r = \delta \leq 1 , \mathrm { s o } z \in [ 0 , 1 - r ]$ . The right-hand side of (47) becomes

$$
r + z - g ( z ) = r + \frac { z ( 1 - z ) } { 2 - z } .
$$

Let

$$
a ( z ) : = z - g ( z ) = { \frac { z ( 1 - z ) } { 2 - z } } .
$$

The function a increases on $[ 0 , z _ { \star } ]$ and decreases on $[ z _ { \star } , 1 ]$ , where $z _ { \star } = 2 - \sqrt { 2 }$ and $a ( z _ { \star } ) = c _ { \star }$ . Since the actual value of z lies in $[ 0 , 1 - r ]$ , we have

$$
r + a ( z ) \leq r + \operatorname* { m a x } _ { 0 \leq u \leq 1 - r } a ( u ) = H ( r ) ,
$$

with H as defined in (16): if $r \leq { \sqrt { 2 } } - 1$ , the interval contains $z _ { \star }$ and the maximum is $r + c _ { \star } \mathrm { { : } }$ otherwise the maximum occurs at $u = 1 - r$ and equals

$$
r + a ( 1 - r ) = \frac { 2 r } { 1 + r } .
$$

It follows that

$$
r _ { \mathrm { J a c } } ( p , B ) \leq H \big ( r _ { F } ( p , B ) \big ) .
$$

It remains to pass from conditional to population regret. Let $( X , A )$ have distribution ${ \mathcal { D } } ,$ let $p _ { X }$ denote the conditional law of A given X, and let h be the classifier in the proposition. Define the random variable

$$
R _ { X } : = r _ { F } ( p _ { X } , h ( X ) ) \in [ 0 , 1 ] .
$$

The two pieces of H agree in both value and first derivative at $r = { \sqrt { 2 } } - 1$ , and the second piece is increasing and concave. Hence H is increasing and concave on [0, 1]. Applying the conditional inequality with $p = p _ { X }$ and $B = \bar { h } ( X )$ , then using Jensen’s inequality and the population-regret definition (10), gives

$$
\begin{array} { r l } { \mathrm { R e g } _ { \mathrm { J a c } } ( h ) = \mathbb { E } _ { X } \left[ r _ { \mathrm { J a c } } ( p _ { X } , h ( X ) ) \right] } & { } \\ { \leq \mathbb { E } _ { X } [ H ( R _ { X } ) ] } & { } \\ { \leq H ( \mathbb { E } _ { X } [ R _ { X } ] ) = H ( \mathrm { R e g } _ { F } ( h ) ) . } \end{array}
$$

Finally, $a ( z ) \leq c _ { \star }$ for every $z \in [ 0 , 1 ]$ , so the representation $\begin{array} { r } { H ( r ) = r + \operatorname* { m a x } _ { 0 \leq u \leq 1 - r } a ( u ) } \end{array}$ gives $H ( r ) \leq r + c _ { \star }$ This proves the second inequality in (17). In particular, if h is $F _ { \mathrm { 1 } } { \mathrm { - } } \mathbf { B } \mathbf { a } \mathbf { y } \mathbf { e } \mathbf { s }$ , then $\mathrm { R e g } _ { F } ( h ) = 0$ and the first inequality gives $\mathrm { \bar { R e g } _ { J a c } } ( h ) \le H ( 0 ) = { \bar { c } } _ { \star }$ □

## D.2 Proof of Lemma 6.2

ProofofLemma 6.2. For fixed $A , B ,$ , the summands in (21) are independent Bernoulli random variables with mean Jac $( { \dot { A } } , { \dot { B } } )$ by (19). Hoeffding’s inequality gives

$$
\operatorname* { P r } \Bigl ( | \widetilde { S } ( A , B ) - \mathrm { J a c } ( A , B ) | > \eta \Bigr ) \le 2 \exp ( - 2 M \eta ^ { 2 } ) .
$$

There are $| \mathcal { D } | ^ { 2 } = 4 ^ { s }$ ordered outcome–report pairs. A union bound proves (22), and solving its right-hand side for M gives (23). □

## D.3 Proof of Theorem 6.3

Proof of Theorem 6.3. Fix a realization for which the uniform-approximation event in Lemma 6.2 holds. For $p \in \Delta _ { N }$ set

$$
\mu _ { p } = \sum _ { A \in \mathcal { Y } } p _ { A } \Phi ( A ) , \qquad g _ { p } ( B ) = \sum _ { A \in \mathcal { Y } } p _ { A } \operatorname { J a c } ( A , B ) , \qquad \widetilde { g } _ { p } ( B ) = \langle \mu _ { p } , \Phi ( B ) \rangle .
$$

By the definitions of $\mu _ { p }$ and ${ \widetilde { S } } ,$

$$
\widetilde { g } _ { p } ( B ) = \sum _ { A \in \mathcal { V } } p _ { A } \langle \Phi ( A ) , \Phi ( B ) \rangle = \sum _ { A \in \mathcal { V } } p _ { A } \widetilde { S } ( A , B ) .\tag{48}
$$

On the uniform-approximation event,

$$
\begin{array} { l } { \displaystyle | g _ { p } ( B ) - \widetilde g _ { p } ( B ) | = \left| \sum _ { A \in \mathcal { V } } p _ { A } \big ( \mathrm { J a c } ( A , B ) - \widetilde { S } ( A , B ) \big ) \right| } \\ { \displaystyle \qquad \leq \sum _ { A \in \mathcal { V } } p _ { A } | \mathrm { J a c } ( A , B ) - \widetilde { S } ( A , B ) | } \\ { \displaystyle \qquad \leq \eta \sum _ { A \in \mathcal { V } } p _ { A } = \eta } \end{array}\tag{49}
$$

for every $p$ and $B .$ Let $B ^ { \star } \in$ arg max<sub>B</sub> $g _ { p } ( B )$ and $\widehat { B } = \mathrm { p r e d } _ { M } ( u )$ . Adding and subtracting the two approximate scores gives

$$
\begin{array} { r l } & { r _ { \mathrm { J a c } } ( p , \widehat { B } ) = g _ { p } ( B ^ { \star } ) - g _ { p } ( \widehat { B } ) } \\ & { \qquad = \big ( g _ { p } ( B ^ { \star } ) - \widetilde { g } _ { p } ( B ^ { \star } ) \big ) + \big ( \widetilde { g } _ { p } ( B ^ { \star } ) - \widetilde { g } _ { p } ( \widehat { B } ) \big ) + \big ( \widetilde { g } _ { p } ( \widehat { B } ) - g _ { p } ( \widehat { B } ) \big ) } \\ & { \qquad \leq 2 \eta + \widetilde { g } _ { p } ( B ^ { \star } ) - \widetilde { g } _ { p } ( \widehat { B } ) } \\ & { \qquad = 2 \eta + \langle \mu _ { p } , \Phi ( B ^ { \star } ) - \Phi ( \widehat { B } ) \rangle } \\ & { \qquad = 2 \eta + \langle \mu _ { p } - u , \Phi ( B ^ { \star } ) - \Phi ( \widehat { B } ) \rangle + \langle u , \Phi ( B ^ { \star } ) - \Phi ( \widehat { B } ) \rangle } \\ & { \qquad \leq 2 \eta + \| \mu _ { p } - u \| _ { 2 } \| \Phi ( B ^ { \star } ) - \Phi ( \widehat { B } ) \| _ { 2 } . } \end{array}
$$

Here

$$
\langle u , \Phi ( B ^ { \star } ) - \Phi ( { \widehat { B } } ) \rangle \leq 0
$$

because $\widehat { B }$ maximizes $B \mapsto \langle u , \Phi ( B ) \rangle$ ; the final line then follows from Cauchy–Schwarz. Moreover, each of the M blocks in $\Phi ( B )$ is a scaled standard basis vector, so

$$
\| \Phi ( B ) \| _ { 2 } ^ { 2 } = \frac { 1 } { M } \sum _ { r = 1 } ^ { M } \| e _ { \overline { { { m } } } _ { \pi _ { r } } ( B ) } \| _ { 2 } ^ { 2 } = 1 .
$$

All feature coordinates are nonnegative, and hence $\langle \Phi ( B ) , \Phi ( B ^ { \prime } ) \rangle \geq 0$ . It follows that

$$
\begin{array} { r } { \| \Phi ( B ) - \Phi ( B ^ { \prime } ) \| _ { 2 } ^ { 2 } = \| \Phi ( B ) \| _ { 2 } ^ { 2 } + \| \Phi ( B ^ { \prime } ) \| _ { 2 } ^ { 2 } - 2 \langle \Phi ( B ) , \Phi ( B ^ { \prime } ) \rangle = 2 - 2 \langle \Phi ( B ) , \Phi ( B ^ { \prime } ) \rangle \leq 2 . } \end{array}
$$

Combining the last two displays yields

$$
r _ { \mathrm { J a c } } ( p , \widehat { B } ) \leq 2 \eta + \sqrt { 2 } \lVert u - \mu _ { p } \rVert _ { 2 } .\tag{50}
$$

It remains to identify the distance on the right with square-loss regret. For every A,

$$
u - \Phi ( A ) = ( u - \mu _ { p } ) + ( \mu _ { p } - \Phi ( A ) ) .
$$

After squaring and averaging, the cross term vanishes because

$$
\sum _ { A } p _ { A } ( \mu _ { p } - \Phi ( A ) ) = \mu _ { p } - \sum _ { A } p _ { A } \Phi ( A ) = 0 .
$$

Therefore the square-loss bias–variance identity is

$$
\sum _ { A } p _ { A } \| u - \Phi ( A ) \| _ { 2 } ^ { 2 } = \| u - \mu _ { p } \| _ { 2 } ^ { 2 } + \sum _ { A } p _ { A } \| \Phi ( A ) - \mu _ { p } \| _ { 2 } ^ { 2 } .\tag{51}
$$

The second term is independent of u, so the conditional surrogate risk is minimized at $u = \mu _ { p }$ , and

$$
r _ { \Psi _ { M } } ( p , u ) = \| u - \mu _ { p } \| _ { 2 } ^ { 2 } .
$$

Substitution into (50) proves (26).

Now apply the conditional bound at each x with $p = p _ { x }$ and $u = f ( x )$ . Using (10), then Cauchy–Schwarz (equivalently, Jensen’s inequality for the square root), gives

$$
\begin{array} { r l } & { \mathrm { R e g } _ { \mathrm { J a c } } ( \mathrm { p r e d } _ { M } \circ f ) = \mathbb { E } _ { X } [ r _ { \mathrm { J a c } } ( p _ { X } , \mathrm { p r e d } _ { M } ( f ( X ) ) ) ] } \\ & { \qquad \leq 2 \eta + \sqrt { 2 } \mathbb { E } _ { X } \left[ \sqrt { r _ { \Psi _ { M } } ( p _ { X } , f ( X ) ) } \right] } \\ & { \qquad \leq 2 \eta + \sqrt { 2 \mathbb { E } _ { X } \left[ r _ { \Psi _ { M } } ( p _ { X } , f ( X ) ) \right] } } \\ & { \qquad = 2 \eta + \sqrt { 2 \mathrm { R e g } _ { \Psi _ { M } } ( f ) } , } \end{array}
$$

where the last equality uses (13). This proves (27).

Finally, set $\eta = \alpha / 2$ . If $\mathrm { R e g } _ { \Psi _ { M } } ( f _ { n } ) \to 0$ , the population bound gives

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname* { s u p } \operatorname { R e g } _ { \operatorname { J a c } } ( \operatorname { p r e d } _ { M } \circ f _ { n } ) \leq 2 \eta = \alpha ,
$$

which is approximate consistency. With this value of η, (23) becomes

$$
M \geq \frac { 2 \left( 2 s \log 2 + \log ( 2 / \rho ) \right) } { \alpha ^ { 2 } } .
$$

Taking the ceiling gives (28); multiplying by the block size $s + 1$ gives the stated order for $d _ { \mathrm { M H } }$ . Since the uniformapproximation event has positive probability, at least one fixed realization of the feature map satisfies all these bounds. □

## D.4 Proof of Corollary 6.4

ProofofCorollary 6.4. Take $d = d _ { \pm }$ in (29). For every A, $B , \mathbb { E } \langle \Phi _ { \pm } ( A ) , \Phi _ { \pm } ( B ) \rangle = \operatorname { J a c } ( A , B )$ : conditional on $\pi _ { r }$ the sign product has expectation one when the two hashes agree and zero otherwise. The summands lie in $[ - 1 , 1 ]$ , so Hoeffding’s inequality and the same union bound give uniform error η with probability at least $1 - \rho$ provided

$$
d _ { \pm } \geq \frac { 2 \bigl ( 2 s \log 2 + \log ( 2 / \rho ) \bigr ) } { \eta ^ { 2 } } .
$$

The proof of Theorem 6.3 applies to (30)–(31), with $\| \Phi _ { \pm } ( B ) - \Phi _ { \pm } ( B ^ { \prime } ) \| _ { 2 } \leq 2$ . Taking $\eta = \alpha / 2$ gives (33), while (32) is exactly the preceding sample-size condition with this choice of η. □

## E The alternative empty-set convention

Suppose Jac $( \varnothing , \varnothing ) = 0$ . Ordering the empty set first, the score matrix becomes diag(0, K). Lemma 4.1 therefore gives ran $\bar { \mathrm { k } } ( S ) =$ rank $\dot { ( L - U ) } = N - \dot {  }$ . The empty score column is zero and the other $\dot { N } - 1$ columns are independent, so afdim $( S ) = N - 1$ and hence afdim $( L ) \stackrel { - } { = } \dot { N } - 1$

The loss block is now

$$
L = \binom { 1 } { \mathbf 1 _ { { N } - 1 } } \quad \mathbf 1 _ { { N } - 1 } ^ { \top } \mathbf 1 _ { { N } - 1 } ^ { } - { \boldsymbol K } \Big ) .
$$

If $L ( \alpha , x ) ^ { \top } = 0$ , the first equation is $\alpha + \mathbf { 1 } _ { N - 1 } ^ { \top } x = 0$ . Substitution into the second gives $K x = 0 , { \mathrm { s o ~ } } x = 0$ and then $\alpha = 0$ . Thus $\mathrm { r a n k } ( L ) = N$

For the calibration lower bound, use the factorial distribution q from (39) without mixing in the empty outcome. Its support and its exact Bayes-optimal report set are both $u ,$ of size $2 ^ { s - 1 }$ . The restricted active score matrix $S _ { u , u }$ is positive definite. Repeating Steps 3–4 of the proof of Theorem 5.2 gives zero lineality and hence the lower bound $2 ^ { s - 1 } - 1$ . The affine-dimension upper bound remains $2 ^ { s } - 1$

## References

Han Bao and Masashi Sugiyama. Calibrated surrogate maximization of linear-fractional utility in binary classification. In Proceedings ofthe Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of PMLR, pages 2337–2347, 2020.

Maxim Berman, Amal Rannen Triki, and Matthew B. Blaschko. The Lovász–Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 4413–4421, 2018.

Mathieu Bouchard, Anne-Laure Jousselme, and Pierre-Emmanuel Doré. A proof for the positive definiteness of the Jaccard index matrix. International Journal ofApproximate Reasoning, 54(5):615–626, 2013.

Andrei Z. Broder. On the resemblance and containment of documents. In Compression and Complexity ofSequences 1997, pages 21–29. IEEE, 1997.

Flavio Chierichetti, Ravi Kumar, Sandeep Pandey, and Sergei Vassilvitskii. Finding the Jaccard median. In Proceedings of the Twenty-First Annual ACM–SIAM Symposium on Discrete Algorithms, pages 293–311, 2010.

Ben Dai and Chunlin Li. RankSEG: A consistent ranking-based framework for segmentation. Journal ofMachine Learning Research, 24(224):1–50, 2023.

Krzysztof Dembczynski, Willem Waegeman, Weiwei Cheng, and Eyke Hüllermeier. On label dependence and loss´ minimization in multi-label classification. Machine Learning, 88(1–2):5–45, 2012.

Jessica J. Finocchiaro, Rafael Frongillo, and Enrique B. Nueve. The structured abstain problem and the Lovász hinge. In Proceedings of the 35th Conference on Learning Theory, volume 178 of PMLR, pages 3718–3740, 2022.

John C. Gower. A general coefficient of similarity and some of its properties. Biometrics, 27(4):857–871, 1971.

Oluwasanmi O. Koyejo, Nagarajan Natarajan, Pradeep K. Ravikumar, and Inderjit S. Dhillon. Consistent multilabel classification. In Advances in Neural Information Processing Systems 28, pages 3321–3329, 2015.

Alex Nowak, Francis Bach, and Alessandro Rudi. Sharp analysis of learning with discrete losses. In Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, volume 89 of PMLR, pages 1920–1929, 2019.

Harish G. Ramaswamy and Shivani Agarwal. Classification calibration dimension for general multiclass losses. In Advances in Neural Information Processing Systems 25, pages 2087–2095, 2012.

Harish G. Ramaswamy and Shivani Agarwal. Convex calibration dimension for multiclass loss matrices. Journal of Machine Learning Research, 17(14):1–45, 2016.

Willem Waegeman, Krzysztof Dembczynski, Arkadiusz Jachnik, Weiwei Cheng, and Eyke Hüllermeier. On the´ Bayes-optimality of F-measure maximizers. Journal ofMachine Learning Research, 15(103):3513–3568, 2014.

Jiaqian Yu and Matthew Blaschko. Learning submodular losses with the Lovász hinge. In Proceedings of the 32nd International Conference on Machine Learning, volume 37 of PMLR, pages 1623–1631, 2015.

Mingyuan Zhang, Harish G. Ramaswamy, and Shivani Agarwal. Convex calibrated surrogates for the multi-label F-measure. In Proceedings ofthe 37th International Conference on Machine Learning, volume 119 of PMLR, pages 11246–11255, 2020.

Mingyuan Zhang. Exact rank and convex calibration dimension lower bounds for the multi-label $F _ { 1 }$ loss. arXiv preprint arXiv:2608.08399, 2026.