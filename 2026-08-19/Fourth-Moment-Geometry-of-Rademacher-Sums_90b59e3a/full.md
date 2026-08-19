# Fourth-Moment Geometry of Rademacher Sums

Peigan Gao<sup>∗</sup>

Jian Qian<sup>†</sup>

August 18, 2026

## Abstract

Let $\varepsilon _ { 1 } , \ldots , \varepsilon _ { n }$ be independent Rademacher signs and let $a = ( a _ { 1 } , \ldots , a _ { n } ) \in \mathbb { R } ^ { n }$ satisfy the normalization below. For the normalized Rademacher sum, we determine how its higher moments depend on the fourth-order mass. Combining a sharp fixed-q moment envelope with a separate argument below the convexity threshold gives the Gaussian stability inequality for the full range $p \geq 4$ of this linear-in-q bound. The same fourth-order framework determines the sharp finite dimensional $L _ { p } / L _ { 4 }$ Khintchine constant for $p \geq 5 ,$ , with the flat coeficient vector as the extremizer. These results settle the conjectures of Jakimiuk and of Bara´nski, Murawski, Nayar, and Oleszkiewicz stated below [13, 14]. We also prove Jakimiuk’s conjectured quadratic stability estimate at $p = 3 .$ . The resulting bounds retain information about sparsity and efective dimension, with applications to Rademacher random projections and randomly signed errors; those applications are not developed further here. Their Laplace-transform form also gives coeficient-sensitive tail bounds. The proofs are discovered with substantial assistance from ChatGPT 5.6 Sol.

## 1 Introduction

Let $\varepsilon _ { 1 } , \varepsilon _ { 2 } , \ldots$ . be independent Rademacher random variables. For a finitely supported vector $a = ( a _ { i } ) \in \mathbb { R } ^ { n }$ , write

$$
S = \sum _ { i = 1 } ^ { n } a _ { i } \varepsilon _ { i } .
$$

We also use the unnormalized partial sums $\textstyle S _ { m } : = \sum _ { i = 1 } ^ { m } \varepsilon _ { i }$ . For $r > 0$ , write $\| X \| _ { r } : = ( \mathbb { E } | X | ^ { r } ) ^ { 1 / r }$ Rademacher sums are fundamental models of randomly signed fluctuations. They appear throughout the probability and Banach space theory, as well as in the Fourier analysis of Boolean functions and reduction in the dimension of the random sign [1, 2]. Their moments are governed at first order by the classical Khintchine inequality. Under normalization

$$
\sum _ { i = 1 } ^ { n } a _ { i } ^ { 2 } = 1 ,
$$

its sharp upper bound is

$$
\begin{array} { r } { \mathbb { E } | S | ^ { p } \leq \mu _ { p } , \qquad \mu _ { p } : = \mathbb { E } | G | ^ { p } , \qquad p > 2 , } \end{array}
$$

where G is standard Gaussian. The determination of the optimal constants goes back to Khintchine and culminated in Haagerup’s theorem; see [3, 4] and the later approach in [5]. By the classical central limit theorem, the Gaussian constant is approached by the flat sums

$$
\frac { \varepsilon _ { 1 } + \cdot \cdot \cdot + \varepsilon _ { N } } { \sqrt { N } }
$$

as $N \to \infty$ , but it is not attained by any nonzero finite Rademacher sum when $p > 2$

There is also a finite-dimensional endpoint. The Schur comparison of Eaton [6], together with Komorowski’s sharp finite-dimensional result [7], shows that for fixed N and $p \geq 3$ , the moment $\mathbb { E } | S _ { N } | ^ { p }$ is maximized by the flat coeficient vector. This does not determine the finite-dimensional $L _ { p } / L _ { 4 }$ ratio, because flattening changes both moments simultaneously, but it identifies the second extremal regime that will reappear below.

The classical Khintchine inequality, however, suppresses two pieces of information relevant here: the concentration behavior of S and the geometry of the optimizer in fixed dimension. A natural parameter for both questions is

$$
q : = \sum _ { i } a _ { i } ^ { 4 } .
$$

Indeed,

$$
\mathbb { E } S ^ { 4 } = 3 - 2 \sum _ { i } a _ { i } ^ { 4 } = 3 - 2 q , \qquad \kappa _ { 4 } ( S ) = - 2 \sum _ { i } a _ { i } ^ { 4 } = - 2 q ,\tag{1}
$$

where $\kappa _ { 4 } ( S )$ denotes the fourth cumulant. Thus q directly measures the fourth-moment defect from Gaussianity. It also records coeficient concentration: $q = 1$ for a coordinate vector, whereas $q = 1 / N$ for the flat vector in dimension N. Thus $q ^ { - 1 }$ serves as an efective number of active coordinates. The central problem is to determine the largest possible higher moment when this fourth-order mass is prescribed.

This problem belongs to the broader stability theory of sharp Khintchine-type inequalities. Earlier quantitative forms measured the distance from the two-coordinate extremizers in the $L _ { 2 } / L _ { 1 }$ inequality and led to applications in Boolean Fourier analysis and convex geometry [8, 9, 10]. Distributional stability, in which the law of the summands is perturbed, was developed in [11]. Recent work also treats Gaussian and coordinate stability for wider classes of symmetric laws at even exponents [12].

## 1.1 Main results

The coeficient-stability question was introduced by Jakimiuk [13]. For every $p \geq 3$ , he proved that there is a constant $c _ { p } > 0$ such that

$$
\mathbb { E } \left| \sum _ { i = 1 } ^ { n } a _ { i } \varepsilon _ { i } \right| ^ { p } \leq \mathbb { E } | G | ^ { p } - c _ { p } \sum _ { i = 1 } ^ { n } a _ { i } ^ { 4 }
$$

and conjectured that the optimal coeficient is $c _ { p } = \mu _ { p } - 1$ for every $p \geq 3$ . He also observed that the even-integer cases follow from moment calculations. The proposed range needs a correction at its lower endpoint. For $S _ { 2 } = ( \varepsilon _ { 1 } + \varepsilon _ { 2 } ) / \sqrt { 2 }$ , one has $q = 1 / 2$ , and the chord inequality fails for every $2 < p < 4 ;$ see Proposition 4.3. Ch´avez and Sheng subsequently obtained related stability estimates for even moments in a broader distributional setting [12]. For Rademacher variables as a special case, their Gaussian-side coeficient is $2 \mu _ { p } / 3$ for even integer $p ;$ this agrees with the sharp coeficient at $p = 4$ and is smaller for $p > 4$

We use the Gaussian-to-coordinate chord

$$
\Lambda _ { p } ( q ) : = q + ( 1 - q ) \mu _ { p } .\tag{2}
$$

The first main result is the following sharp stability bound.

Theorem 1.1 (Sharp Gaussian stability). Let $n \geq 1$ and $a \in \mathbb { R } ^ { n }$ satisfy $\textstyle \sum _ { i = 1 } ^ { n } a _ { i } ^ { 2 } = 1$ . For every real $p \geq 4$ 2

$$
\mathbb { E } \left| \sum _ { i = 1 } ^ { n } a _ { i } \varepsilon _ { i } \right| ^ { p } \leq \mu _ { p } - ( \mu _ { p } - 1 ) \sum _ { i = 1 } ^ { n } a _ { i } ^ { 4 } = \Lambda _ { p } ( q ) .\tag{3}
$$

The coeficient $\mu _ { p } - 1$ is optimal. At $p = 4$ , equality holds for every coeficient vector. For $p > 4$ equality holds exactly for coordinate vectors, up to signs and permutations.

The afine estimate is the strongest bound that is linear in $q ,$ but it does not identify the extremal law at an intermediate value of $q .$ . The finer fixed-q problem reveals the structure behind the chord. For $q \in ( 0 , 1 ]$ , define

$$
\mathcal { A } ( q ) : = \left\{ a : \mathrm { ~ } a \mathrm { ~ i s ~ f i n i t e l y ~ s u p p o r t e d } , \quad \sum _ { i } a _ { i } ^ { 2 } = 1 , \quad \sum _ { i } a _ { i } ^ { 4 } = q \right\} ,
$$

and define

$$
Y _ { q } : = q ^ { 1 / 4 } \varepsilon _ { 0 } + \sqrt { 1 - \sqrt { q } } G ,
$$

where $\varepsilon _ { 0 }$ and $G$ are independent. Thus $Y _ { q }$ has one Bernoulli coordinate carrying all fourth-order mass and a Gaussian remainder carrying the difuse variance.

Theorem 1.2 (Fixed-fourth-moment extremal principle). Let $\Phi \in C ^ { 4 } ( \mathbb { R } )$ be even and of polynomial growth, and suppose that $\Phi ^ { ( 4 ) }$ is convex. Then, for every $q \in ( 0 , 1 ]$ ，

$$
\operatorname* { s u p } _ { a \in { \mathcal { A } } ( q ) } \mathbb { E } \Phi ( S ) = \mathbb { E } \Phi ( Y _ { q } ) .\tag{4}
$$

In particular, for every $p \geq 5$

$$
\operatorname* { s u p } _ { a \in A ( q ) } \mathbb { E } | S | ^ { p } = U _ { p } ( q ) : = \mathbb { E } \left| q ^ { 1 / 4 } \varepsilon _ { 0 } + \sqrt { 1 - \sqrt { q } } G \right| ^ { p } .
$$

Moreover, $U _ { p }$ is strictly convex on (0, 1).

The supremum in Theorem 1.2 is generally attained only in the closure of finite Rademacher sums. One coeficient remains macroscopic, while the remaining variance is split among increasingly many vanishing coeficients. Since

$$
U _ { p } ( 0 ) = \mu _ { p } , \qquad U _ { p } ( 1 ) = 1 ,
$$

strict convexity yields

$$
U _ { p } ( q ) < ( 1 - q ) \mu _ { p } + q , \qquad 0 < q < 1 .
$$

The theorem therefore explains the afine bound and shows that it is strict throughout the interior. The interval $4 < p < 5$ is diferent: $| x | ^ { p }$ does not have a convex fourth derivative there, so

the fixed-moment principle is unavailable. A separate argument completes Theorem 1.1 in this interval.

The same fourth-order parameter organizes the finite-dimensional reverse H¨older problem. For $N \geq 1$ and $p \geq 4$ , define the scale-invariant constant

$$
C _ { p , 4 , N } : = \operatorname* { s u p } _ { a \in \mathbb { R } ^ { N } \setminus \{ 0 \} } \frac { \left\| \sum _ { i = 1 } ^ { N } a _ { i } \varepsilon _ { i } \right\| _ { p } } { \left\| \sum _ { i = 1 } ^ { N } a _ { i } \varepsilon _ { i } \right\| _ { 4 } } ,
$$

For the normalized sum $\begin{array} { r } { S = \sum _ { i = 1 } ^ { N } a _ { i } \varepsilon _ { i } } \end{array}$ with $\textstyle \sum _ { i = 1 } ^ { N } a _ { i } ^ { 2 } = 1$ ，

$$
\| S \| _ { 4 } ^ { 4 } = 3 - 2 q ,
$$

so maximizing $\| S \| _ { p } / \| S \| _ { 4 }$ couples the higher moment to precisely the same fourth-order geometry. Bara´nski, Murawski, Nayar, and Oleszkiewicz proved that the dimension-free $L _ { p } / L _ { 4 }$ Khintchine constant is Gaussian for $p \geq 4$ . For $p \geq 5$ , they reduced the finite-dimensional constant to the one-parameter family

$$
C _ { p , 4 , N + 1 } = \operatorname* { s u p } _ { x \geq 1 } { \frac { \| x + \varepsilon _ { 1 } + \cdot \cdot \cdot + \varepsilon _ { N } \| _ { p } } { \| x + \varepsilon _ { 1 } + \cdot \cdot \cdot + \varepsilon _ { N } \| _ { 4 } } } .
$$

They conjectured that the supremum is attained at $x = 1 \ [ 1 4 ]$ . We prove the stronger monotonicity statement below, which resolves that conjecture.

Theorem 1.3 (Finite-dimensional $L _ { p ^ { - } } L _ { 4 }$ constant). For every real $p \geq 5$ and every $N \geq 2$

$$
C _ { p , 4 , N } = { \frac { \| S _ { N } \| _ { p } } { \| S _ { N } \| _ { 4 } } } .\tag{5}
$$

More strongly, for $n = N - 1$

$$
x \longmapsto { \frac { \| x + S _ { n } \| _ { p } } { \| x + S _ { n } \| _ { 4 } } } \quad i s \ s t r i c t l y \ d e c r e a s i n g \ o n \ [ 1 , \infty ) .\tag{6}
$$

After normalization, the optimizer is unique up to coordinate signs and permutations; without normalization, also up to nonzero scalar multiplication.

This is the finite-dimensional counterpart of the Gaussian extremal regime. As the number of coordinates grows, flat vectors converge to the Gaussian endpoint $q = 0$ . In fixed dimension, the constraint

$$
q \geq \frac { 1 } { N }
$$

prevents that limit, and the least possible fourth-order mass is attained exactly by the flat vector. The fixed-q and finite-dimensional theorems therefore resolve two versions of the same extremal problem: the former identifies the extremal law after a Gaussian remainder is allowed, while the latter identifies the extremal coeficient vector when only N coordinates are available.

The preceding theorems do not cover the critical exponent $p = 3$ . For this paragraph, let $\begin{array} { r } { S = \sum _ { i = 1 } ^ { N } a _ { i } \varepsilon _ { i } } \end{array}$ . Jakimiuk also asked whether diagonal stability could be strengthened to a dimension-free quadratic deficit. We prove that there is a universal constant $c > 0$ such that for every $N \geq 2$ and every normalized $a \in \mathbb { R } ^ { N }$

$$
\mathbb { E } | S | ^ { 3 } \le \mathbb { E } \left| \frac { \varepsilon _ { 1 } + \cdot \cdot \cdot + \varepsilon _ { N } } { \sqrt { N } } \right| ^ { 3 } - c \sum _ { i = 1 } ^ { N } \left( a _ { i } ^ { 2 } - \frac { 1 } { N } \right) ^ { 2 } .
$$

No linear-in-q chord statement is asserted here for the intermediate range $3 < p < 4 ;$ the twocoordinate example above rules out the broader range $2 < p < 4$ . For the statement and proof below, write

$$
\overline { { S } } _ { n } : = \frac { S _ { n } } { \sqrt { n } } = \frac { 1 } { \sqrt { n } } \sum _ { i = 1 } ^ { n } \varepsilon _ { i } , \qquad \Delta _ { n } ( a ) : = \sum _ { i = 1 } ^ { n } \left( a _ { i } ^ { 2 } - \frac { 1 } { n } \right) ^ { 2 } = q - \frac { 1 } { n } .
$$

Theorem 1.4 (Dimension-free stability at the third moment). For every $n \geq 1$ and every $a \in \mathbb { R } ^ { n }$ with $\textstyle \sum _ { i } a _ { i } ^ { 2 } = 1$

$$
\mathbb { E } \left| \sum _ { i = 1 } ^ { n } a _ { i } \varepsilon _ { i } \right| ^ { 3 } \leq \mathbb { E } \left| { \overline { { S } } } _ { n } \right| ^ { 3 } - { \frac { 1 } { 1 0 0 } } \Delta _ { n } ( a ) .\tag{7}
$$

In particular, the optimal dimension-free constant

$$
C _ { 3 } ^ { \mathrm { o p t } } : = \operatorname* { i n f } _ { \substack { n \geq 2 , \sum _ { i } a _ { i } ^ { 2 } = 1 } } \frac { \mathbb { E } \left| \overline { { S } } _ { n } \right| ^ { 3 } - \mathbb { E } \left| \sum _ { i } a _ { i } \varepsilon _ { i } \right| ^ { 3 } } { \Delta _ { n } ( a ) }\tag{8}
$$

satisfies

$$
\frac { 3 ( 5 \sqrt { 2 } - 7 ) } { 1 6 } \leq C _ { 3 } ^ { \mathrm { o p t } } \leq 5 \sqrt { 3 } - 6 \sqrt { 2 } .\tag{9}
$$

In addition, in dimension three, it satisfies

$$
\mathbb { E } \left| \sum _ { i = 1 } ^ { 3 } a _ { i } \varepsilon _ { i } \right| ^ { 3 } \leq \mathbb { E } \left| \overline { { S } } _ { 3 } \right| ^ { 3 } - ( 5 \sqrt { 3 } - 6 \sqrt { 2 } ) \Delta _ { 3 } ( a ) .
$$

The upper bound is attained in dimension three by $( a _ { 1 } ^ { 2 } , a _ { 2 } ^ { 2 } , a _ { 3 } ^ { 2 } ) = ( 1 / 2 , 1 / 2 , 0 )$ , up to signs and permutations.

This confirms the second conjecture in [13] and describes the endpoint at which the usual smooth-convexity argument degenerates.

The fixed-fourth-moment principle also yields coeficient-sensitive concentration bounds. $\mathrm { A p \mathrm { - } }$ plying it to Taylor polynomials of cosh and passing to the limit gives, for every $t \in \mathbb { R }$

$$
\mathbb { E } e ^ { t S } \leq \exp \left( \frac { 1 - \sqrt { q } } { 2 } t ^ { 2 } \right) \cosh \left( q ^ { 1 / 4 } t \right) .
$$

Consequently, for $u > 0$

$$
\mathbb { P } ( | S | \geq u ) \leq 2 \operatorname* { i n f } _ { t > 0 } \exp \left( - t u + \frac { 1 - \sqrt { q } } { 2 } t ^ { 2 } \right) \cosh \left( q ^ { 1 / 4 } t \right) .
$$

The right-hand side is the moment generating function of the one-spike-plus-Gaussian extremizer $Y _ { q } ,$ and hence is optimal among normalized Rademacher sums with prescribed fourth-order mass. Chernof’s method gives the corresponding refinement of the usual sub-Gaussian tail bound. The estimate distinguishes coeficient vectors with the same variance but diferent efective support sizes, interpolating between the Gaussian regime $q \to 0$ and a single random sign at $q = 1$ . In this sense the extremal principle controls the full Laplace transform, not only individual moments.

Theorems 1.1 and 1.3 address diferent extremal questions, and neither is a formal consequence of the other. Their proofs share a fourth-order input: for $p \geq 5$ , extremizers with fixed second and fourth moments have at most one exceptional coeficient. Letting the number of small coeficients tend to infinity gives $Y _ { q }$ and the Gaussian stability problem; keeping the dimension fixed gives the family $( x , 1 , \ldots , 1 )$ and the finite-dimensional ratio problem. The critical third-moment theorem has a diferent structure. Repeatedly averaging two extreme squared coeficients produces a pointwise nonnegative cubic gain, and a small-ball estimate makes that gain uniformly quadratic. The new contributions are:

1. the corrected real-p form of Jakimiuk’s Conjecture 1 for $p \geq 4$ , together with a counterexample for $2 < p < 4$ ;

2. a proof of the Bara´nski–Murawski–Nayar–Oleszkiewicz flat-point conjecture; and

3. the exact fixed-q moment and Laplace-transform envelopes;

4. a proof of Jakimiuk’s Conjecture 2 with an explicit universal constant.

## 1.2 Proof architecture

Section 2 proves the stability theorem for $4 \le p \le 6$ by a coeficientwise argument related to the one in [14]; this route reaches below the fourth-order convexity threshold. For $p \geq 5 ,$ Section 3 combines the fixed-moment reduction with a Riccati comparison for noncentral Gaussian moments and completes the proof of Theorem 1.1. Section 4 records the exact fixed-q envelope from the same reduction. Next, Section 5 uses the finite-dimensional one-spike reduction of [14]; a determinant identity and a cubic-quotient lemma show that every non-flat spike lowers the $L _ { p } / L _ { 4 }$ ratio. Finally, Section 6 proves Theorem 1.4 by repeatedly averaging the largest and smallest squared coeficients. The local gain comes from an explicit scalar smoothing identity together with the Rademacher small-ball estimate of Dzindzalieta and G¨otze.

## 1.3 Statement of AI use

Initial versions of the proofs of the sharp Gaussian stability inequality and the finite-dimensional $L _ { p ^ { - } } L _ { 4 }$ constant theorem were developed with assistance from ChatGPT 5.6 Sol. The authors checked and revised the arguments, take full responsibility for their mathematical content, and independently verified the extensions presented here.

## 2 A direct proof of sharp Gaussian stability for $4 \le p \le 6$

## 2.1 A coeficientwise leave-one-out estimate

The proof of Theorem 1.1 uses two estimates with overlapping ranges. When $q \leq 1 / 2$ , every coeficient is small enough for a uniform coeficientwise bound. When $q \geq 1 / 2$ , interpolation between the exact fourth moment and an upper bound for the sixth moment is stronger. We use $q = 1 / 2$ as a convenient dividing point. Put

$$
x _ { i } : = a _ { i } ^ { 2 } , \quad \quad \sum _ { i = 1 } ^ { n } x _ { i } = 1 , \quad \quad q = \sum _ { i = 1 } ^ { n } x _ { i } ^ { 2 } .
$$

The next identity isolates this fourth-power mass. It is closely related to the leave-one-out argument in [14], but the coeficientwise form is needed here.

Lemma 2.1. Let $p \geq 4$ . Then

$$
\mathbb { E } \left| S \right| ^ { p } \leq \mu _ { p } \sum _ { i = 1 } ^ { n } x _ { i } \left( 1 - \frac { 2 x _ { i } } { 3 } \right) ^ { ( p - 2 ) / 2 } .\tag{10}
$$

Proof. Fix $i ,$ let $U _ { i }$ be uniform on [−1, 1] and independent of all signs, and set

$$
S _ { i } : = \sum _ { j \neq i } a _ { j } \varepsilon _ { j } + a _ { i } U _ { i } .
$$

With $\textstyle T _ { i } = \sum _ { j \neq i } a _ { j } \varepsilon _ { j }$ , the fundamental theorem of calculus gives, for every continuously diferentiable $f ,$

$$
\mathbb { E } _ { \varepsilon _ { i } } \big [ \varepsilon _ { i } f ( T _ { i } + a _ { i } \varepsilon _ { i } ) \big ] = \frac { f ( T _ { i } + a _ { i } ) - f ( T _ { i } - a _ { i } ) } { 2 } = a _ { i } \mathbb { E } _ { U _ { i } } f ^ { \prime } ( T _ { i } + a _ { i } U _ { i } ) .
$$

After multiplication by $a _ { i } .$ , summation over $i ,$ and expectation,

$$
\mathbb { E } [ S f ( S ) ] = \sum _ { i = 1 } ^ { n } a _ { i } ^ { 2 } \mathbb { E } f ^ { \prime } ( S _ { i } ) .
$$

Taking $f ( s ) = | s | ^ { p - 2 }$ s gives

$$
\mathbb { E } | S | ^ { p } = ( p - 1 ) \sum _ { i = 1 } ^ { n } x _ { i } \mathbb { E } | S _ { i } | ^ { p - 2 } .\tag{11}
$$

It remains to estimate each leave-one-out moment. On an enlarged product space, realize

$$
U _ { i } = \sum _ { k \geq 1 } 2 ^ { - k } \varepsilon _ { i , k } \quad { \mathrm { a l m o s t ~ s u r e l y } } ,
$$

where the $\varepsilon _ { i , k }$ are independent Rademacher variables, also independent of the original signs. Set

$$
S _ { i } ^ { ( m ) } : = \sum _ { j \neq i } a _ { j } \varepsilon _ { j } + a _ { i } \sum _ { k = 1 } ^ { m } 2 ^ { - k } \varepsilon _ { i , k } .
$$

The sharp Khintchine inequality applied to this finite Rademacher sum gives

$$
\mathbb { E } | S _ { i } ^ { ( m ) } | ^ { p - 2 } \le \mu _ { p - 2 } \left( \sum _ { j \ne i } a _ { j } ^ { 2 } + a _ { i } ^ { 2 } \sum _ { k = 1 } ^ { m } 4 ^ { - k } \right) ^ { ( p - 2 ) / 2 } .
$$

As $m  \infty , S _ { i } ^ { ( m ) }  S _ { i }$ almost surely and $\textstyle \sum _ { k = 1 } ^ { m } 4 ^ { - k } \to 1 / 3$ . Moreover,

$$
| S _ { i } ^ { ( m ) } | \leq \sum _ { j \neq i } | a _ { j } | + | a _ { i } | ,
$$

uniformly in $m$ . The right-hand side is a finite deterministic bound, so its (p − 2)-power is integrable; dominated convergence therefore yields

$$
\mathbb { E } | S _ { i } | ^ { p - 2 } \leq \mu _ { p - 2 } \left( 1 - \frac { 2 x _ { i } } { 3 } \right) ^ { ( p - 2 ) / 2 } .
$$

Substituting this bound into Equation (11) and using $( p - 1 ) \mu _ { p - 2 } = \mu _ { p }$ proves Equation (10).

To express the coeficientwise estimate only in terms of $q ,$ we need a uniform lower bound. Since $( p - 2 ) / 2 \geq 1$ , define

$$
\phi _ { p } ( x ) : = \frac { 1 - ( 1 - 2 x / 3 ) ^ { ( p - 2 ) / 2 } } { x } \quad ( x > 0 ) , \qquad \phi _ { p } ( 0 ) : = \frac { p - 2 } { 3 } .\tag{12}
$$

The numerator is concave as a function of x and vanishes at the origin. For a concave function $f$ with $f ( 0 ) = 0$ , the quotient $f ( x ) / x$ is nonincreasing; hence $\phi _ { p }$ is nonincreasing on [0, 1].

Lemma 2.2. For $p \geq 4$ and $x _ { 0 } = 2 ^ { - 1 / 2 }$ ，

$$
\phi _ { p } ( x _ { 0 } ) \geq 1 - \frac { 1 } { \mu _ { p } } .\tag{13}
$$

For $p > 4$ , the inequality is strict.

A self-contained proof is given in Section C.1.

Proposition 2.3. If $p \geq 4$ and $q \leq 1 / 2$ , then Equation (3) holds.

Proof. Let $m _ { * } = \operatorname* { m a x } _ { i } x _ { i }$ . The assumption $q \leq 1 / 2$ and the inequality $\begin{array} { r } { m _ { * } ^ { 2 } \le \sum _ { i } x _ { i } ^ { 2 } = q } \end{array}$ give $m _ { * } \leq 2 ^ { - 1 / 2 }$ . Monotonicity of $\phi _ { p }$ and Lemma 2.2 now give

$$
\begin{array} { l } { 1 - \displaystyle \sum _ { i } x _ { i } \left( 1 - \frac { 2 x _ { i } } { 3 } \right) ^ { ( p - 2 ) / 2 } = \displaystyle \sum _ { i } x _ { i } ^ { 2 } \phi _ { p } ( x _ { i } ) \geq \sum _ { i } x _ { i } ^ { 2 } \phi _ { p } ( m _ { * } ) } \\ { \geq q \phi _ { p } ( m _ { * } ) \geq q \phi _ { p } ( 2 ^ { - 1 / 2 } ) \geq q \left( 1 - \frac { 1 } { \mu _ { p } } \right) . } \end{array}
$$

Multiplication by $\mu _ { p }$ , followed by Lemma 2.1, gives $\mathbb { E } | S | ^ { p } \leq \Lambda _ { p } ( q )$

## 2.2 Interpolation between the fourth and sixth moments

$$
A ( q ) : = 3 - 2 q , \qquad B ( q ) : = 1 5 - 3 0 q + 1 6 q ^ { 3 / 2 } .\tag{14}
$$

The exact sixth-moment formula and the inequality

$$
\sum _ { i } x _ { i } ^ { 3 } \leq \left( \sum _ { i } x _ { i } ^ { 2 } \right) ^ { 3 / 2 } = q ^ { 3 / 2 } ,
$$

which is the monotonicity of finite-dimensional ℓ<sub>r</sub>-norms, give

$$
\mathbb { E } S ^ { 6 } = 1 5 - 3 0 q + 1 6 \sum _ { i } x _ { i } ^ { 3 } \leq 1 5 - 3 0 q + 1 6 q ^ { 3 / 2 } = B ( q ) .\tag{15}
$$

Lemma 2.4. For $1 / 2 \le q \le 1$ and $4 \le p \le 6$

$$
( 3 - 2 q ) ^ { ( 6 - p ) / 2 } ( 1 5 - 3 0 q + 1 6 q ^ { 3 / 2 } ) ^ { ( p - 4 ) / 2 } \leq q + ( 1 - q ) \mu _ { p } = \Lambda _ { p } ( q ) .\tag{16}
$$

The elementary calculus proof appears in Section C.2.

Proposition 2.5. $I f 4 \leq p \leq 6$ and $q \geq 1 / 2$ , then Equation (3) holds.

Proof. Fix $q \in [ 1 / 2 , 1 ]$ . For $4 \le p \le 6$ , log-convexity between the fourth and sixth moments, together with $\mathbb { E } S ^ { 4 } = 3 - 2 q$ , gives

$$
\begin{array} { r } { \mathbb { E } | S | ^ { p } \leq ( \mathbb { E } S ^ { 4 } ) ^ { ( 6 - p ) / 2 } ( \mathbb { E } S ^ { 6 } ) ^ { ( p - 4 ) / 2 } \leq A ( q ) ^ { ( 6 - p ) / 2 } B ( q ) ^ { ( p - 4 ) / 2 } . } \end{array}
$$

The scalar bound Equation (16) then yields $\mathbb { E } | S | ^ { p } \leq \Lambda _ { p } ( q )$

Corollary 2.6. The inequality Equation (3) holds for every $4 \le p \le 6$

Proof. Combine Proposition 2.3 when $q \leq 1 / 2$ with Proposition 2.5 when $q \geq 1 / 2$

## 3 The fixed-moment reduction for $p \geq 5$

For $p \geq 5$ , the extremal principle in Lemma 3.1 reduces an arbitrary Rademacher sum to one distinguished coeficient together with equal smaller coeficients. Passing to infinitely many small coeficients leaves one Gaussian shift, so the remaining comparison is one-dimensional. This gives a second proof of Theorem 1.1 in that range.

## 3.1 Reduction to a Gaussian shift

Lemma 3.1 ([14]). Let $N \geq 2$ , let $\Phi \in C ^ { 4 } ( \mathbb { R } )$ be even with convex fourth derivative, and fix feasible numbers $\sigma _ { 2 } , \sigma _ { 4 }$ . On

$$
\left\{ b \in \mathbb { R } ^ { N } : \sum _ { i = 1 } ^ { N } b _ { i } ^ { 2 } = \sigma _ { 2 } , \ \sum _ { i = 1 } ^ { N } b _ { i } ^ { 4 } = \sigma _ { 4 } \right\} ,
$$

the functional

$$
b \longmapsto \mathbb { E } \Phi \left( \sum _ { i = 1 } ^ { N } b _ { i } \varepsilon _ { i } \right)
$$

has a maximizer of the form $( c , d , \ldots , d )$ , where $c \geq d \geq 0$

For $\Phi ( x ) = | x | ^ { p }$ , the hypothesis holds whenever $p \geq 5$ , because

$$
\Phi ^ { ( 4 ) } ( x ) = p ( p - 1 ) ( p - 2 ) ( p - 3 ) | x | ^ { p - 4 }
$$

is convex.

Proposition 3.2 (Gaussian reduction). Let $p \geq 5$ . Then

$$
\mathbb { E } | S | ^ { p } \leq \mathbb { E } \left| q ^ { 1 / 4 } + \sqrt { 1 - \sqrt { q } } G \right| ^ { p } = U _ { p } ( q ) .\tag{17}
$$

Proof. Append zeros and regard the coeficient vector as an element of $\mathbb { R } ^ { N }$ , where $N \geq n$ . For each such N, Lemma 3.1 gives

$$
\mathbb { E } \left| S \right| ^ { p } \leq \mathbb { E } \left| c _ { N } \varepsilon _ { 1 } + d _ { N } \sum _ { i = 2 } ^ { N } \varepsilon _ { i } \right| ^ { p } ,
$$

where $c _ { N } \geq d _ { N } \geq 0$ satisfy

$$
c _ { N } ^ { 2 } + ( N - 1 ) d _ { N } ^ { 2 } = 1 , \qquad c _ { N } ^ { 4 } + ( N - 1 ) d _ { N } ^ { 4 } = q .
$$

The two constraints can be solved explicitly. Set

$$
\delta _ { N , q } : = \sqrt { \frac { N q - 1 } { N - 1 } } .\tag{18}
$$

Then

$$
c _ { N } ^ { 2 } = \frac { 1 + ( N - 1 ) \delta _ { N , q } } { N } , \qquad d _ { N } ^ { 2 } = \frac { 1 - \delta _ { N , q } } { N } .\tag{19}
$$

As $N \to \infty$

$$
c _ { N } \longrightarrow q ^ { 1 / 4 } , \qquad ( N - 1 ) d _ { N } ^ { 2 } \longrightarrow 1 - \sqrt { q } , \qquad d _ { N } \longrightarrow 0 .
$$

The Lindeberg–Feller central limit theorem therefore gives

$$
d _ { N } \sum _ { i = 2 } ^ { N } \varepsilon _ { i } \stackrel { D } { \longrightarrow } \sqrt { 1 - \sqrt { q } } G .
$$

The leading sign is independent of the remaining sum, and $c _ { N } \varepsilon _ { 1 } \to q ^ { 1 / 4 } \varepsilon _ { 1 }$ almost surely. Thus, with an independent copy ε<sub>0</sub>,

$$
X _ { N } : = c _ { N } \varepsilon _ { 1 } + d _ { N } \sum _ { i = 2 } ^ { N } \varepsilon _ { i } \stackrel { D } { \longrightarrow } q ^ { 1 / 4 } \varepsilon _ { 0 } + \sqrt { 1 - \sqrt { q } } G .
$$

To pass from convergence in distribution to convergence of the p-th moments, apply the sharp Khintchine inequality at exponent $p + 1$

$$
\operatorname* { s u p } _ { N \geq n } \mathbb { E } | X _ { N } | ^ { p + 1 } \leq \mu _ { p + 1 } .
$$

Thus $\{ | X _ { N } | ^ { p } \} _ { N \geq n }$ is uniformly integrable, and convergence in distribution implies

$$
\operatorname* { l i m } _ { N \to \infty } \mathbb { E } | X _ { N } | ^ { p } = \mathbb { E } \left| q ^ { 1 / 4 } \varepsilon _ { 0 } + \sqrt { 1 - \sqrt { q } } G \right| ^ { p } .
$$

Since $\mathbb { E } | S | ^ { p } \le \mathbb { E } | X _ { N } | ^ { p }$ for every $N \geq n$ , the limit above, together with the symmetry of $G ,$ proves Equation (17). □

## 3.2 The Gaussian-shift chord

Lemma 3.3 (Gaussian-shift chord). Let $p > 4 , 0 \leq u \leq 1$ , and $G \sim N ( 0 , 1 )$ . Then

$$
\begin{array} { r } { \mathbb { E } \left| u + \sqrt { 1 - u ^ { 2 } } G \right| ^ { p } \leq \mu _ { p } - ( \mu _ { p } - 1 ) u ^ { 4 } . } \end{array}\tag{20}
$$

For $0 < u < 1$ , the inequality is strict.

Proof. Consider the fixed-moment profile

$$
g ( s ) : = \mathbb { E } \left| s ^ { 1 / 4 } + { \sqrt { 1 - { \sqrt { s } } } } G \right| ^ { p } , \qquad 0 \leq s \leq 1 .
$$

Its endpoint values are $g ( 0 ) = \mu _ { p }$ and $g ( 1 ) = 1$ . Strict convexity on (0, 1) would therefore place g below the chord joining these endpoints, which is precisely Equation (20).

To verify the required convexity, write $u = s ^ { 1 / 4 }$ and

$$
h ( u ) = \mathbb { E } \left| u + \sqrt { 1 - u ^ { 2 } } G \right| ^ { p } .
$$

Diferentiating gives

$$
g ^ { \prime \prime } ( s ) = { \frac { u h ^ { \prime \prime } ( u ) - 3 h ^ { \prime } ( u ) } { 1 6 u ^ { 7 } } } .\tag{21}
$$

Hence the sign of $g ^ { \prime \prime }$ is determined by $u h ^ { \prime \prime } ( u ) - 3 h ^ { \prime } ( u )$ , and it remains to prove $u h ^ { \prime \prime } ( u ) - 3 h ^ { \prime } ( u ) > 0$ for $0 < u < 1$

For the curvature calculation, put

$$
v = \sqrt { 1 - u ^ { 2 } } , \qquad t = \frac { u } { v } .
$$

Since $p > 4$ , set $\alpha = p - 4$ and

$$
m ( t ) = \mathbb { E } \left| G + t \right| ^ { \alpha } .
$$

Moreover, the Gaussian convolution is smooth in t. Gaussian integration by parts gives

$$
m ^ { \prime \prime } ( t ) + t m ^ { \prime } ( t ) - \alpha m ( t ) = 0 .\tag{22}
$$

Using this identity, the curvature reduces to

$$
\begin{array} { c } { { u h ^ { \prime \prime } ( u ) - 3 h ^ { \prime } ( u ) = p ( \alpha + 2 ) v ^ { \alpha + 3 } \Big \{ \alpha t \big ( 3 + ( \alpha + 2 ) t ^ { 2 } \big ) m ( t ) } } \\ { { - \big ( 3 + ( 2 \alpha + 3 ) t ^ { 2 } \big ) m ^ { \prime } ( t ) \Big \} . } } \end{array}\tag{23}
$$

The calculation is given in Section A.1. The bracket is positive exactly when

$$
{ \frac { m ^ { \prime } ( t ) } { t m ( t ) } } < \alpha { \frac { 3 + ( \alpha + 2 ) t ^ { 2 } } { 3 + ( 2 \alpha + 3 ) t ^ { 2 } } } , \qquad t > 0 .\tag{24}
$$

Write $R ( t )$ for the logarithmic derivative on the left of Equation (24), and $Q ( t )$ for the explicit function on the right. The function Q is the threshold at which the bracket in Equation (23) changes sign. The diferential equation Equation (22) gives

$$
t R ^ { \prime } ( t ) = \alpha - ( 1 + t ^ { 2 } ) R ( t ) - t ^ { 2 } R ( t ) ^ { 2 } .\tag{25}
$$

Direct substitution shows that Q is a strict supersolution of the same Riccati equation:

$$
t Q ^ { \prime } ( t ) > \alpha - ( 1 + t ^ { 2 } ) Q ( t ) - t ^ { 2 } Q ( t ) ^ { 2 } , \qquad t > 0 .\tag{26}
$$

The calculation is recorded in Section A.2. Since m is even and Equation (22) gives $m ^ { \prime \prime } ( 0 ) = \alpha m ( 0 )$ both $R ( t )$ and $Q ( t )$ tend to α as $t \downarrow 0$ . The positive remainder in the supersolution inequality shows that $Q - R$ satisfies a first-order linear equation with positive forcing. Because $Q ( t ) - R ( t )  0$ at the origin, variation of constants gives

$$
R ( t ) < Q ( t ) , \qquad t > 0 .
$$

The endpoint comparison and variation-of-constants formula are written out in Section A.2.

Hence the right-hand side of Equation (23) is positive. By Equation (21), $g ^ { \prime \prime } ( s ) > 0$ on $( 0 , 1 )$ The endpoint values of g then give Equation (20), with strict inequality for $0 < s < 1$ □

Proposition 3.4. The inequality Equation (3) holds for every $p \geq 5$

Proof. Apply Proposition 3.2, followed by Lemma 3.3 with $u = q ^ { 1 / 4 }$

Proof of Theorem 1.1. Corollary 2.6 covers $4 \le p \le 6$ , while Proposition 3.4 covers $p \geq 5$ . Their union covers the full range $p \geq 4 .$ . At $p = 4 .$ , the theorem is exactly Equation (1). Coordinate vectors give equality for every $p > 4$ , so the coeficient $\mu _ { p } - 1$ cannot be increased. The equality cases are proved in Corollary 4.2. □

## 4 The exact fixed-q upper envelope

This section records the full upper envelope supplied by the fixed-moment principle, rather than only its afine consequence.

## 4.1 Finite-dimensional extremizers

Fix $N \geq 2$ and a feasible $q \in [ 1 / N , 1 ]$ . Let $c _ { N , q } , d _ { N , q }$ be given by Equations (18) and (19), and define

$$
Y _ { N , q } ^ { + } : = c _ { N , q } \varepsilon _ { 1 } + d _ { N , q } \sum _ { j = 2 } ^ { N } \varepsilon _ { j } .\tag{27}
$$

Then

$$
\sum _ { i = 1 } ^ { N } a _ { i } ^ { 2 } = 1 , \qquad \sum _ { i = 1 } ^ { N } a _ { i } ^ { 4 } = q
$$

for the coeficients of $Y _ { N , q } ^ { + }$

Proposition 4.1 (Exact finite-dimensional optimization). Let $\Phi \in C ^ { 4 } ( \mathbb { R } )$ be even with convex fourth derivative, let $N \geq 2$ , and let $q \in [ 1 / N , 1 ]$ . Then

$$
\operatorname* { m a x } _ { \stackrel { a \in \mathbb { R } ^ { N } } { \sum _ { i } a _ { i } ^ { 2 } = 1 , \ \sum _ { i } a _ { i } ^ { 4 } = q } } \mathbb { E } \Phi \left( \sum _ { i } a _ { i } \varepsilon _ { i } \right) = \mathbb { E } \Phi ( Y _ { N , q } ^ { + } ) .\tag{28}
$$

Proof. Lemma 3.1 reduces the maximizer to coeficients $( c , d , \ldots , d )$ . Solving

$$
c ^ { 2 } + ( N - 1 ) d ^ { 2 } = 1 , \qquad c ^ { 4 } + ( N - 1 ) d ^ { 4 } = q , \qquad c \geq d \geq 0 ,
$$

gives Equation (19) and hence the stated maximum.

## 4.2 Passage to one spike and a Gaussian sea

Proof of Theorem 1.2. Fix $N \geq n .$ , append zeros to the coeficient vector of S, and apply Proposition 4.1:

$$
\mathbb { E } \Phi ( S ) \leq \mathbb { E } \Phi ( Y _ { N , q } ^ { + } ) .
$$

By Equation (19),

$$
c _ { N , q } \to q ^ { 1 / 4 } , \qquad ( N - 1 ) d _ { N , q } ^ { 2 } \to 1 - \sqrt { q } , \qquad d _ { N , q } \to 0 .
$$

The Lindeberg–Feller and independence argument from Proposition 3.2 gives $Y _ { N , q } ^ { + } \xrightarrow { D } Y _ { q }$ . If Φ has polynomial growth of degree at most $m _ { : }$ the sharp Khintchine inequality uniformly bounds a moment of order strictly larger than m. Thus $\{ \Phi ( Y _ { N , q } ^ { + } ) \}$ is uniformly integrable, and

$$
\mathbb { E } \Phi ( S ) \le \operatorname* { l i m } _ { N \to \infty } \mathbb { E } \Phi ( Y _ { N , q } ^ { + } ) = \mathbb { E } \Phi ( Y _ { q } ) .
$$

Conversely, $Y _ { N , q } ^ { + }$ is feasible whenever $N q \geq 1$ . Its expectations converge to $\mathbb { E } \Phi ( Y _ { q } )$ , so the upper bound is approached by finite Rademacher sums and is therefore the exact supremum. □

## 4.3 Exact moment profiles and strict improvements

Proof of the moment-profile and convexity assertions in Theorem 1.2. For $p \ge 5 , \Phi ( x ) = | x | ^ { p }$ satisfies the hypotheses of Theorem 1.2, giving the fixed-q supremum formula. The Riccati argument in Lemma 3.3 proves that the profile $g ( s )$ in Lemma 3.3 is strictly convex for $p > 4$ Since s is the same fourth-moment parameter q, this is strict convexity of $q \mapsto U _ { p } ( q )$ . Together with $U _ { p } ( 0 ) = \mu _ { p }$ and $U _ { p } ( 1 ) = 1$ , this gives $U _ { p } ( q ) < \Lambda _ { p } ( q )$ for $0 < q < 1$ □

Corollary 4.2 (Equality cases). At $p = 4$ , equality in Equation (3) holds for every coeficient vector. For every $p > 4$ , equality holds if and only if $q = 1$ , equivalently, exactly one coeficient is nonzero and has magnitude 1.

Proof. The case $p = 4$ is Equation (1). For $p \geq 5$ , strict convexity in Theorem 1.2 gives strictness whenever $0 < q < 1$ . For $4 < p < 5$ , Lemma 2.2 makes the proof of Proposition 2.3 strict when $q \leq 1 / 2$ . When $1 / 2 \le q < 1$ , the supporting-tangent comparison that leads from Equation (64) to Equation (16) is strict. Finally, under $\textstyle \sum _ { i } a _ { i } ^ { 2 } = 1$ , the condition $q = 1$ is equivalent to a coordinate vector. □

The following two-coordinate example shows why the original conjectured range cannot hold.

Proposition 4.3. For every $2 < p < 4$ , the inequality Equation (3) fails for $S _ { 2 } = ( \varepsilon _ { 1 } + \varepsilon _ { 2 } ) / \sqrt { 2 }$

Proof. Since $G ^ { 2 }$ is nonconstant, strict log-convexity of its moments gives

$$
\begin{array} { r } { \mu _ { p } = \mathbb { E } ( G ^ { 2 } ) ^ { p / 2 } < ( \mathbb { E } G ^ { 2 } ) ^ { 2 - p / 2 } ( \mathbb { E } G ^ { 4 } ) ^ { p / 2 - 1 } = 3 ^ { p / 2 - 1 } . } \end{array}
$$

Since $0 < p / 2 - 1 < 1$ , the concavity of $x \mapsto x ^ { p / 2 - 1 }$ yields

$$
\frac { 1 + 3 ^ { p / 2 - 1 } } { 2 } < \left( \frac { 1 + 3 } { 2 } \right) ^ { p / 2 - 1 } = 2 ^ { p / 2 - 1 } .
$$

Combining these two strict inequalities gives $1 + \mu _ { p } < 2 ^ { p / 2 }$ . For $S _ { 2 }$ , one has $q = 1 / 2$ and $\mathbb { E } | S _ { 2 } | ^ { p } = 2 ^ { p / 2 - 1 }$ , hence

$$
\mathbb { E } | S _ { 2 } | ^ { p } > \frac { \mu _ { p } + 1 } { 2 } .
$$

## 4.4 Laplace transform

The fixed-q principle also determines the sharp moment generating function of S.

Theorem 4.4 (Sharp fixed-q Laplace transform). Let $q ~ \in ~ ( 0 , 1 ]$ , let $a \ \in \ A ( q )$ , and write $\textstyle S = \sum _ { i } a _ { i } \varepsilon _ { i }$ . Then, for every $t \in \mathbb { R }$ 2

$$
\mathbb { E } e ^ { t S } = \prod _ { i } \cosh ( t a _ { i } ) \leq \exp \left( \frac { 1 - \sqrt { q } } { 2 } t ^ { 2 } \right) \cosh ( q ^ { 1 / 4 } t ) .\tag{29}
$$

More precisely, the right-hand side equals $\begin{array} { r } { \operatorname* { s u p } _ { a \in \mathcal { A } ( q ) } \mathbb { E } \exp ( t \sum _ { i } a _ { i } \varepsilon _ { i } ) } \end{array}$ ; it is the exact supremum at fixed q, approached in the finite-sum closure.

Proof. Since S is symmetric, $\mathbb { E } e ^ { t S } = \mathbb { E } \cosh ( t S )$ . Apply Lemma 3.1 to $\Phi ( x ) = \cosh ( t x )$ , whose fourth derivative $t ^ { 4 }$ cosh(tx) is convex. After appending zeros, for every $N \geq n$

$$
\mathbb { E } e ^ { t S } \le \cosh ( t c _ { N , q } ) \cosh ( t d _ { N , q } ) ^ { N - 1 } .
$$

By Equation (19), $( N - 1 ) d _ { N , q } ^ { 4 } \to 0$ . Hence the expansion log cosh $z = z ^ { 2 } / 2 + O ( z ^ { 4 } )$ at the origin gives

$$
\cosh ( t c _ { N , q } ) \cosh ( t d _ { N , q } ) ^ { N - 1 } \longrightarrow \cosh ( q ^ { 1 / 4 } t ) \exp \left( \frac { 1 - \sqrt { q } } { 2 } t ^ { 2 } \right) .
$$

Because the finite upper vectors are feasible and converge to the limiting law, the right-hand side is the exact supremum. □

## 5 The finite-dimensional $L _ { p } { - } L _ { 4 }$ conjecture

Bara´nski, Murawski, Nayar, and Oleszkiewicz proved, for $p \geq 5$ , that

$$
C _ { p , 4 , n + 1 } = \operatorname* { s u p } _ { x \geq 1 } { \frac { \| x + S _ { n } \| _ { p } } { \| x + S _ { n } \| _ { 4 } } } ,\tag{30}
$$

and conjectured that the supremum is attained at $x = 1 \ [ 1 4 ]$ ]. We prove the stronger monotonicity statement Equation (6), which resolves the conjecture.

## 5.1 A determinant identity

For an integer $r \geq 0$ and $c \in \mathbb { R }$ , set

$$
J _ { p , r } ( c ) : = \mathbb { E } { \bigl [ } ( c + S _ { r } ) | c + S _ { r } | ^ { p - 2 } { \bigr ] } , \qquad H _ { r } ( c ) : = \mathbb { E } ( c + S _ { r } ) ^ { 3 } = c ( c ^ { 2 } + 3 r ) .\tag{31}
$$

For fixed $n \geq 1$ , define

$$
A _ { p } ( x ) : = \operatorname { \mathbb { E } } | x + S _ { n } | ^ { p } , \qquad B ( x ) : = \operatorname { \mathbb { E } } | x + S _ { n } | ^ { 4 } .
$$

For use below, set $S _ { 0 } = 0$ and put $r = n - 1$

Lemma 5.1 (Determinant identity). For $x > 1$ ，

$$
4 B ( x ) \frac { \partial A _ { p } } { \partial ( x ^ { 2 } ) } ( x ) - p A _ { p } ( x ) \frac { \partial B } { \partial ( x ^ { 2 } ) } ( x ) = \frac { p n } { x } \big [ H _ { r } ( x + 1 ) J _ { p , r } ( x - 1 ) - H _ { r } ( x - 1 ) J _ { p , r } ( x + 1 ) \big ] .\tag{32}
$$

Proof. The identity follows by expressing both moments through the two neighboring shifts $x - 1$ and $x + 1$ . Condition on the last sign in $S _ { n }$ . Exchangeability gives

$$
A _ { p } ( x ) = { \frac { 1 } { 2 } } { \big [ } ( x + n ) J _ { p , r } ( x + 1 ) + ( x - n ) J _ { p , r } ( x - 1 ) { \big ] } ,\tag{33}
$$

$$
B ( x ) = \frac { 1 } { 2 } \big [ ( x + n ) H _ { r } ( x + 1 ) + ( x - n ) H _ { r } ( x - 1 ) \big ] .\tag{34}
$$

To see the first identity, write $| X | ^ { p } = X \cdot X | X | ^ { p - 2 }$ , where $X = x + S _ { n }$ , and separate the deterministic part x from the sum of signs; exchangeability handles the sum. The second identity follows in the same way from $X ^ { 4 } = X \cdot X ^ { 3 }$

Diferentiating the two shift representations, and using $\partial _ { x ^ { 2 } } = ( 2 x ) ^ { - 1 } \partial _ { x }$ , yields

$$
\frac { \partial A _ { p } } { \partial ( x ^ { 2 } ) } = \frac { p } { 4 x } \big ( J _ { p , r } ( x - 1 ) + J _ { p , r } ( x + 1 ) \big ) , \qquad \frac { \partial B } { \partial ( x ^ { 2 } ) } = \frac { 1 } { x } \big ( H _ { r } ( x - 1 ) + H _ { r } ( x + 1 ) \big ) .
$$

Substitution of Equations (33) and (34) cancels the diagonal products and leaves Equation (32).

Since

$$
{ \frac { \mathrm { d } } { \mathrm { d } ( x ^ { 2 } ) } } \log { \frac { A _ { p } ( x ) ^ { 1 / p } } { B ( x ) ^ { 1 / 4 } } } = { \frac { 4 B \partial _ { x ^ { 2 } } A _ { p } - p A _ { p } \partial _ { x ^ { 2 } } B } { 4 p A _ { p } B } } ,
$$

the desired decrease of the norm ratio will follow once $J _ { p , r } / H _ { r }$ is shown to be increasing.

## 5.2 The cubic quotient lemma

Lemma 5.2 (Cubic quotient). Let $g \in C ^ { 3 } ( \mathbb { R } )$ be odd, and suppose that $h = g ^ { \prime \prime \prime }$ is even and convex. Let $a \geq 0$ and assume

$$
a h ( 0 ) \geq 6 g ^ { \prime } ( 0 ) .\tag{35}
$$

Then

$$
c \longmapsto { \frac { g ( c ) } { c ( c ^ { 2 } + a ) } }\tag{36}
$$

is nondecreasing on $( 0 , \infty )$ . It is strictly increasing if $a h ( 0 ) > 6 g ^ { \prime } ( 0 )$

Proof. The proof turns the derivative of the quotient into two successive monotonicity statements. Its numerator is

$$
N ( c ) = c ( c ^ { 2 } + a ) g ^ { \prime } ( c ) - ( 3 c ^ { 2 } + a ) g ( c ) .
$$

To control $N _ { ; }$ , set

$$
M ( c ) = ( c ^ { 2 } + a ) g ^ { \prime \prime } ( c ) - 6 g ( c ) .
$$

Diferentiation gives the useful relation

$$
N ^ { \prime } ( c ) = c M ( c ) .\tag{37}
$$

Since g is odd, $M ( 0 ) = 0$ , and

$$
M ^ { \prime } ( 0 ) = a h ( 0 ) - 6 g ^ { \prime } ( 0 ) \geq 0 .
$$

Because the convex function h is locally absolutely continuous, the following computation holds for almost every $c > 0$ 0:

$$
M ^ { \prime \prime } ( c ) = ( c ^ { 2 } + a ) h ^ { \prime } ( c ) + 4 \left( c h ( c ) - \int _ { 0 } ^ { c } h ( s ) \mathrm { d } s \right) \geq 0 .\tag{38}
$$

Here evenness and convexity make h nondecreasing on $[ 0 , \infty )$ , which also makes the last integrand nonnegative. Hence $M ^ { \prime \prime } \geq 0$ almost everywhere. Since $M ^ { \prime } ( 0 ) \geq 0$ , the function $M ^ { \prime }$ is nonnegative and $M ( c ) \geq 0$ for $c \geq 0$ . Equation 37, together with $N ( 0 ) = 0$ , now gives $N ( c ) \geq 0$ . If the initial inequality is strict, then $M ^ { \prime } ( 0 ) > 0$ , and the same chain gives $N ( c ) > 0$ for every $c > 0$ □

The initial condition in Lemma 5.2 will follow from the next real-moment recurrence. Its proof uses the same conditioning idea as Lemma 2.1.

Lemma 5.3 (Rademacher moment recurrence). For every integer $r \geq 1$ and every real $s \geq 3$

$$
\begin{array} { r } { \mathbb { E } \left| S _ { r } \right| ^ { s } \leq ( s - 1 ) r \mathbb { E } \left| S _ { r } \right| ^ { s - 2 } . } \end{array}\tag{39}
$$

Proof. Let $F _ { s } ( y ) = y \left| y \right| ^ { s - 2 }$ . Expanding $S _ { r }$ coeficient by coeficient,

$$
\mathbb { E } \left| S _ { r } \right| ^ { s } = \mathbb { E } [ S _ { r } F _ { s } ( S _ { r } ) ] = \sum _ { i = 1 } ^ { r } \mathbb { E } [ \varepsilon _ { i } F _ { s } ( S _ { r } ) ] .
$$

Conditioning on $T _ { i } = S _ { r } - \varepsilon _ { i } ,$ , the fundamental theorem of calculus gives

$$
\mathbb { E } _ { \varepsilon _ { i } } [ \varepsilon _ { i } F _ { s } ( T _ { i } + \varepsilon _ { i } ) ] = \frac { F _ { s } ( T _ { i } + 1 ) - F _ { s } ( T _ { i } - 1 ) } { 2 } = \frac { s - 1 } { 2 } \int _ { T _ { i } - 1 } ^ { T _ { i } + 1 } | z | ^ { s - 2 } \ \mathrm { d } z .
$$

The function $z \mapsto | z | ^ { s - 2 }$ is convex. The upper Hermite–Hadamard inequality therefore gives

$$
{ \frac { 1 } { 2 } } \int _ { T _ { i } - 1 } ^ { T _ { i } + 1 } | z | ^ { s - 2 } ~ \mathrm { d } z \leq { \frac { | T _ { i } - 1 | ^ { s - 2 } + | T _ { i } + 1 | ^ { s - 2 } } { 2 } } .
$$

After expectation, the right-hand side equals $\mathbb { E } | S _ { r } | ^ { s - 2 }$ by symmetry of the conditioned sign. Summing over $i = 1 , \ldots , r$ proves Equation (39). □

## 5.3 Proof of the finite-dimensional theorem

Proof of Theorem 1.3. By Equation (30), the theorem reduces to Equation (6). First suppose $n \geq 2 .$ so $r = n - 1 \geq 1$ . We apply Lemma 5.2 to

$$
g ( c ) = J _ { p , r } ( c ) , \qquad a = 3 r .
$$

Indeed, $g$ is the convolution of the odd function $f _ { p } ( x ) = x | x | ^ { p - 2 }$ , and

$$
f _ { p } ^ { \prime \prime \prime } ( x ) = ( p - 1 ) ( p - 2 ) ( p - 3 ) | x | ^ { p - 4 } .
$$

with the symmetric law of $S _ { r }$ . For $p \geq 5$ , the displayed third derivative is even and convex;   
convolution preserves both properties, so $g ^ { \prime \prime \prime }$ is even and convex.

Let $s = p - 2 \geq 3$ . Diferentiation under the finite expectation gives

$$
g ^ { \prime } ( 0 ) = ( p - 1 ) \mathbb { E } | S _ { r } | ^ { s } , \qquad g ^ { \prime \prime \prime } ( 0 ) = ( p - 1 ) s ( s - 1 ) \mathbb { E } | S _ { r } | ^ { s - 2 } .
$$

The moment recurrence now verifies the strict initial condition:

$$
3 r g ^ { \prime \prime \prime } ( 0 ) - 6 g ^ { \prime } ( 0 ) \geq 3 r ( p - 1 ) ( s - 1 ) ( s - 2 ) \mathbb { E } | S _ { r } | ^ { s - 2 } > 0 .
$$

The cubic-quotient lemma therefore shows that

$$
c \longmapsto { \frac { J _ { p , r } ( c ) } { H _ { r } ( c ) } } = { \frac { J _ { p , r } ( c ) } { c ( c ^ { 2 } + 3 r ) } }\tag{40}
$$

is strictly increasing on $( 0 , \infty )$

For $x > 1$ , one has $0 < x - 1 < x + 1$ , and hence

$$
\frac { J _ { p , r } ( x - 1 ) } { H _ { r } ( x - 1 ) } < \frac { J _ { p , r } ( x + 1 ) } { H _ { r } ( x + 1 ) } ,
$$

After cross multiplication by the positive values of $H _ { r }$ , this is exactly the negativity of the bracket in Equation (32). Therefore,

$$
{ \frac { \mathrm { d } } { \mathrm { d } ( x ^ { 2 } ) } } \log { \frac { \| x + S _ { n } \| _ { p } } { \| x + S _ { n } \| _ { 4 } } } < 0 , \qquad x > 1 .
$$

The ratio is strictly decreasing on $( 1 , \infty )$

When $n = 1$ , use the same determinant identity with $J _ { p , 0 } ( c ) = c ^ { p - 1 }$ and $H _ { 0 } ( c ) = c ^ { 3 } ;$ their quotient is $c ^ { p - 4 }$ , which is strictly increasing for $p > 4$ . The ratio is therefore strictly decreasing on $( 1 , \infty )$ in this case as well.

It remains to include $x = 1$ . For $s \in \{ 4 , p \}$ , the reverse triangle inequality gives

$$
| \| x + S _ { n } \| _ { s } - \| y + S _ { n } \| _ { s } | \leq | x - y | .
$$

Hence $x \mapsto \| x + S _ { n } \| _ { s }$ is continuous. Since $\| 1 + S _ { n } \| _ { 4 } > 0$ , their ratio

$$
x \longmapsto { \frac { \left\| x + S _ { n } \right\| _ { p } } { \left\| x + S _ { n } \right\| _ { 4 } } }
$$

is continuous at $x = 1$ . For any $x > 1$ , choose $y \in ( 1 , x )$ . Strict decrease gives $f \left( x \right) < f \left( y \right)$ , where $f$ denotes the ratio, and continuity gives $f ( y ) \to f ( 1 )$ as $y \downarrow 1$ . Hence

$$
\frac { \| x + S _ { n } \| _ { p } } { \| x + S _ { n } \| _ { 4 } } < \frac { \| 1 + S _ { n } \| _ { p } } { \| 1 + S _ { n } \| _ { 4 } } , \qquad x > 1 .
$$

The reduction Equation (30) now gives

$$
C _ { p , 4 , N } = { \frac { \| 1 + S _ { N - 1 } \| _ { p } } { \| 1 + S _ { N - 1 } \| _ { 4 } } } = { \frac { \| S _ { N } \| _ { p } } { \| S _ { N } \| _ { 4 } } } ,
$$

where the last equality follows by restoring the independent leading sign.

For uniqueness, normalize an optimizer by $\textstyle \sum _ { i } a _ { i } ^ { 2 } = 1$ and put $q = \textstyle \sum _ { i } a _ { i } ^ { 4 }$ . At this fixed $q ,$ Lemma 3.1 bounds its ratio by that of $Y _ { N , q } ^ { + } .$ . If $q > 1 / N$ , the upper vector has coeficient ratio $c _ { N , q } / d _ { N , q } > 1$ . At $q = 1 , d _ { N , q } = 0$ and the upper vector is already a coordinate vector; otherwise the ratio is finite and strictly larger than one. Its $L _ { p } / L _ { 4 }$ ratio equals 1. On the other hand, since $N \geq 2$ , the random variable $\vert S _ { N } \vert$ is nonconstant. Hence, for $p > 4$ , strict monotonicity of $L _ { r } .$ -norms gives $\frac { \| S _ { N } \| _ { p } } { \| S _ { N } \| _ { 4 } } > 1$ . Thus the coordinate vector is also strictly suboptimal. Consequently, every optimizer must satisfy $q = 1 / N$ . Equality in Cauchy–Schwarz forces $| a _ { 1 } | = \cdots = | a _ { N } | .$ proving uniqueness up to signs and permutations. □

Corollary 5.4 (Explicit fixed-dimensional constant). For $p \geq 5$ and $N \geq 2$

$$
C _ { p , 4 , N } = \frac { \left( 2 ^ { - N } \sum _ { k = 0 } ^ { N } \binom { N } { k } | N - 2 k | ^ { p } \right) ^ { 1 / p } } { ( 3 N ^ { 2 } - 2 N ) ^ { 1 / 4 } } .\tag{41}
$$

After normalization, the flat coeficient vector is the unique optimizer up to coordinate signs and permutations; without normalization, its nonzero scalar multiples are also optimizers.

Proof. The formula follows from the binomial law of $S _ { N }$ and $\mathbb { E } S _ { N } ^ { 4 } = 3 N ^ { 2 } - 2 N$ . Uniqueness was proved at the end of the proof of Theorem 1.3. □

## 6 Dimension-free stability at the critical third moment

At $p = 3$ , the fourth-order convexity used above is no longer available. We work instead with the squared coeficients. At each step, the largest and smallest squared coeficients are replaced by their average. This preserves their sum, moves the vector toward the flat point, and decreases q. Conditioning on the remaining signs reduces the corresponding increase in the third moment to a two-variable calculation; a small-ball estimate makes that increase uniform in the residual sum.

## 6.1 The simplex formulation

Signs of the coeficients may be absorbed into the Rademacher variables. Write

$$
x _ { i } = a _ { i } ^ { 2 } , \qquad x _ { i } \geq 0 , \qquad \sum _ { i = 1 } ^ { n } x _ { i } = 1 ,
$$

and, for x in the probability simplex, set

$$
F _ { n } ( x ) : = \mathbb { E } \left| \sum _ { i = 1 } ^ { n } { \sqrt { x _ { i } } } \varepsilon _ { i } \right| ^ { 3 } , \qquad q ( x ) : = \sum _ { i = 1 } ^ { n } x _ { i } ^ { 2 } .\tag{42}
$$

At the flat point $u _ { n } = ( 1 / n , \ldots , 1 / n )$ 2

$$
F _ { n } ( u _ { n } ) = \mathbb { E } \left| \overline { { S } } _ { n } \right| ^ { 3 } ,
$$

while

$$
\sum _ { i = 1 } ^ { n } \left( x _ { i } - { \frac { 1 } { n } } \right) ^ { 2 } = q ( x ) - { \frac { 1 } { n } } .\tag{43}
$$

The proof will compare the increase of $F _ { n }$ with the decrease of $q$ at each averaging step.

We use the following small-ball estimate.

Lemma 6.1 (Dzindzalieta–G¨otze [15]). Let

$$
R = \sum _ { k = 1 } ^ { m } b _ { k } \varepsilon _ { k } , \qquad \sum _ { k = 1 } ^ { m } b _ { k } ^ { 2 } \leq 1 ,
$$

and suppose that max<sub>k</sub> $| b _ { k } | \leq \tau \leq 1$ . Then

$$
\mathbb { P } \{ | R | \le \tau \} \ge \frac { 3 } { 1 6 } \tau .\tag{44}
$$

## 6.2 A two-point smoothing estimate

Put

$$
\kappa : = { \frac { 5 { \sqrt { 2 } } - 7 } { 2 } } .\tag{45}
$$

The next lemma quantifies the efect of averaging two squared coeficients. Its expectation is only over the two displayed Rademacher variables.

Lemma 6.2 (Cubic two-point smoothing). Let $x \geq y \geq 0 , A = x + y > 0$ , and $r \in \mathbb { R } . \ I f \varepsilon , \varepsilon ^ { \prime }$ are independent Rademacher variables, then

$$
\begin{array} { r } { \Gamma ( r ; x , y ) : = \mathbb { E } _ { \varepsilon , \varepsilon ^ { \prime } } \left| r + \sqrt { A / 2 } \varepsilon + \sqrt { A / 2 } \varepsilon ^ { \prime } \right| ^ { 3 } } \\ { - \mathbb { E } _ { \varepsilon , \varepsilon ^ { \prime } } \left| r + \sqrt { x } \varepsilon + \sqrt { y } \varepsilon ^ { \prime } \right| ^ { 3 } \geq 0 . } \end{array}\tag{46}
$$

Moreover, when $| r | \leq { \sqrt { A } }$

$$
\Gamma ( r ; x , y ) \ge \kappa \frac { ( x - y ) ^ { 2 } } { \sqrt { A } } .\tag{47}
$$

Proof. By homogeneity, normalize $A = 1$ ; symmetry allows us to replace r by $s = | r |$ . The two possible magnitudes of ${ \sqrt { x } } \varepsilon + { \sqrt { y } } \varepsilon ^ { \prime }$ are described by

$$
u = { \sqrt { x } } + { \sqrt { y } } , \qquad v = { \sqrt { x } } - { \sqrt { y } } .
$$

Then $1 \leq u \leq \sqrt { 2 } , 0 \leq v \leq 1 , u ^ { 2 } + v ^ { 2 } = 2$ , and u $v = x - y$ . For $s , t \geq 0$ , write

$$
H ( s , t ) = { \frac { | s + t | ^ { 3 } + | s - t | ^ { 3 } } { 2 } } .
$$

Averaging first over the common sign of each magnitude gives

$$
\Gamma ( s ; x , y ) = \frac { 1 } { 2 } H ( s , \sqrt { 2 } ) + \frac { 1 } { 2 } { s } ^ { 3 } - \frac { 1 } { 2 } H ( s , u ) - \frac { 1 } { 2 } H ( s , v ) .\tag{48}
$$

Since $H ( s , t ) = t ^ { 3 } + 3 t s ^ { 2 }$ for $s \leq t$ and $H ( s , t ) = s ^ { 3 } + 3 s t ^ { 2 }$ for $s \geq t ,$ diferentiating separately on $[ 0 , v ] , [ v , u ]$ , and $[ u , 1 ]$ shows that the right-hand side of Equation (48) is nonincreasing on [0, 1]. The derivatives are recorded in Section B.1. Hence

$$
\Gamma ( s ; x , y ) \geq \Gamma ( 1 ; x , y ) .
$$

At $s = 1$ , substitution of the appropriate branches gives

$$
\Gamma ( 1 ; x , y ) = \frac { 5 \sqrt { 2 } } { 2 } - 3 - \frac { u ^ { 3 } } { 2 } + \frac { 3 u ^ { 2 } } { 2 } - \frac { 3 u } { 2 } .
$$

Using $( x - y ) ^ { 2 } = u ^ { 2 } ( 2 - u ^ { 2 } )$ , one obtains

$$
\Gamma ( 1 ; x , y ) - \kappa ( x - y ) ^ { 2 } = \kappa ( u - 1 ) ^ { 2 } ( u - \sqrt { 2 } ) ( u - 5 - 4 \sqrt { 2 } ) \geq 0 .\tag{49}
$$

This proves Equation (47) when $A = 1$

For $1 \leq s \leq u ,$ the same formulas show that the expression in Equation (48) decreases to ${ \scriptstyle { \frac { 1 } { 2 } } } ( { \sqrt { 2 } } - u ) ^ { 3 }$ . It then equals ${ \scriptstyle { \frac { 1 } { 2 } } } ( { \sqrt { 2 } } - s ) ^ { 3 } { \mathrm { ~ o n ~ } } [ u , { \sqrt { 2 } } ]$ and vanishes for $s \geq \sqrt { 2 }$ . It is therefore nonnegative for every $s \geq 1$ , proving Equation (46). Rescaling from $A = 1$ completes the proof. □

The conditional estimate can now be averaged over the remaining coordinates.

Lemma 6.3 (Extreme-pair gain). Let $x \ge y \ge 0$ , put $A = x + y > 0$ , and let

$$
R = \sum _ { k } b _ { k } \varepsilon _ { k }
$$

be independent of two additional Rademacher variables $\varepsilon , \varepsilon ^ { \prime }$ . Assume

$$
\sum _ { k } b _ { k } ^ { 2 } \leq 1 , \qquad \operatorname* { m a x } _ { k } | b _ { k } | \leq { \sqrt { A } } , \qquad A \leq 1 .\tag{50}
$$

Then

$$
\begin{array} { r l } & { \mathbb { E } \left| R + \sqrt { A / 2 } \varepsilon + \sqrt { A / 2 } \varepsilon ^ { \prime } \right| ^ { 3 } - \mathbb { E } \left| R + \sqrt { x } \varepsilon + \sqrt { y } \varepsilon ^ { \prime } \right| ^ { 3 } } \\ & { \qquad \geq \displaystyle \frac { 3 \kappa } { 1 6 } ( x - y ) ^ { 2 } . } \end{array}\tag{51}
$$

Proof. Condition on R. By Lemma 6.2, the conditional gain is always nonnegative, and on the event $\{ | R | \leq { \sqrt { A } } \}$ it is at least

$$
\kappa { \frac { ( x - y ) ^ { 2 } } { \sqrt { A } } } .
$$

If necessary, pad the residual sum with a zero coeficient; this does not change its law or any assumption. Since $A \leq 1$ , we may apply Lemma 6.1 with $\tau = { \sqrt { A } }$ , obtaining

$$
\mathbb { P } \{ | R | \le { \sqrt { A } } \} \geq { \frac { 3 } { 1 6 } } { \sqrt { A } } .
$$

Averaging the conditional estimate proves Equation (51).

## 6.3 The dimension-three case

Lemma 6.4. In dimension three,

$$
\mathbb { E } \left| \sum _ { i = 1 } ^ { 3 } a _ { i } \varepsilon _ { i } \right| ^ { 3 } \leq \mathbb { E } \left| \overline { { S } } _ { 3 } \right| ^ { 3 } - ( 5 \sqrt { 3 } - 6 \sqrt { 2 } ) \Delta _ { 3 } ( a ) .
$$

The corresponding quotient is attained by $( a _ { 1 } ^ { 2 } , a _ { 2 } ^ { 2 } , a _ { 3 } ^ { 2 } ) = ( 1 / 2 , 1 / 2 , 0 )$ , up to signs and permutations. Proof. Set $C _ { \star } = 5 \sqrt { 3 } - 6 \sqrt { 2 }$ . After changing signs and permuting coordinates, assume $a \geq b \geq$ $c \geq 0$ and $a ^ { 2 } + b ^ { 2 } + c ^ { 2 } = 1$ . The identities

$$
\mathbb { E } \left| \overline { { S } } _ { 3 } \right| ^ { 3 } = \frac { 5 } { 2 \sqrt { 3 } } , \qquad \sqrt { 2 } + \frac { C _ { \star } } { 2 } = \frac { 5 } { 2 \sqrt { 3 } } + \frac { C _ { \star } } { 3 } ,
$$

show that the desired inequality is equivalent to

$$
\mathbb { E } | a \varepsilon _ { 1 } + b \varepsilon _ { 2 } + c \varepsilon _ { 3 } | ^ { 3 } + C _ { \star } ( a ^ { 4 } + b ^ { 4 } + c ^ { 4 } ) \leq \sqrt { 2 } + \frac { C _ { \star } } { 2 } .\tag{52}
$$

The cases $a \geq b + c$ and $a \leq b + c$ exhaust the ordered region. Suppose first that $a \geq b + c$ Conditioning on the last two signs gives

$$
\mathbb { E } \left| a \varepsilon _ { 1 } + b \varepsilon _ { 2 } + c \varepsilon _ { 3 } \right| ^ { 3 } = a ( 3 - 2 a ^ { 2 } ) .
$$

Also $b ^ { 4 } + c ^ { 4 } \leq ( b ^ { 2 } + c ^ { 2 } ) ^ { 2 }$ . With $t = a ^ { 2 } \geq 1 / 2$ , the left-hand side of Equation (52) is therefore at most

$$
\sqrt { t } \left( 3 - 2 t \right) + C _ { \star } \bigl ( t ^ { 2 } + ( 1 - t ) ^ { 2 } \bigr ) .
$$

Its derivative is

$$
\left( 2 t - 1 \right) \left( 2 C _ { \star } - \frac { 3 } { 2 \sqrt { t } } \right) ,
$$

which is nonpositive on $[ 1 / 2 , 1 ]$ . Thus the maximum in this region occurs at $t = 1 / 2$ , proving Equation (52) there, with equality only at $( a , b , c ) = ( 1 / \sqrt { 2 } , 1 / \sqrt { 2 } , 0 )$

Now suppose $a \leq b + c , $ and put $s = a + b + c$ and $r = a b c$ . Expanding the four sign patterns in this triangle region gives

$$
\mathbb { E } \left| a \varepsilon _ { 1 } + b \varepsilon _ { 2 } + c \varepsilon _ { 3 } \right| ^ { 3 } = \frac { s ^ { 3 } - 1 2 r } { 2 } , \qquad a ^ { 4 } + b ^ { 4 } + c ^ { 4 } = 1 - \frac { 1 } { 2 } ( s ^ { 2 } - 1 ) ^ { 2 } + 4 s r .
$$

For fixed $s ,$ the left-hand side of Equation (52) is afine in $r ,$ with coeficient $4 C _ { \star } s - 6 < 0$ . It is therefore largest when abc is smallest. Under $a \geq b \geq c \geq 0 , a + b + c = s .$ and $a ^ { 2 } + b ^ { 2 } + c ^ { 2 } = 1$ that minimum occurs when the two largest coordinates coincide; see Section B.2. It remains to consider

$$
a = b = \frac { 1 } { \sqrt { 2 + z ^ { 2 } } } , \qquad c = \frac { z } { \sqrt { 2 + z ^ { 2 } } } , \qquad 0 \leq z \leq 1 .
$$

For this family Equation (52) becomes

$$
\frac { 4 + 3 z ^ { 2 } + z ^ { 3 } / 2 } { ( 2 + z ^ { 2 } ) ^ { 3 / 2 } } + C _ { \star } \frac { 2 + z ^ { 4 } } { ( 2 + z ^ { 2 } ) ^ { 2 } } \leq \sqrt { 2 } + \frac { C _ { \star } } { 2 } .\tag{53}
$$

The derivative calculation in Section B.2 shows that the left-hand side first decreases and then increases, so it has no interior maximum. Its endpoint values are equal, which proves Equation (53). The endpoints correspond respectively to $( 1 / \sqrt { 2 } , 1 / \sqrt { 2 } , 0 )$ and the flat vector. The non-flat endpoint also gives

$$
C _ { 3 } ^ { \mathrm { o p t } } \leq 5 \sqrt { 3 } - 6 \sqrt { 2 } .
$$

## 6.4 Iteration to the flat vector

Proof of Theorem $1 . 4 \cdot$ Start from $x ^ { ( 0 ) } = ( a _ { 1 } ^ { 2 } , \ldots , a _ { n } ^ { 2 } )$ . If it is flat, there is nothing to prove. Otherwise, let $M _ { k }$ and $m _ { k }$ be the largest and smallest coordinates of $x ^ { ( k ) }$ , and obtain $x ^ { ( k + 1 ) }$ by replacing this pair with their average. This operation keeps the vector in the simplex. If the flat point is reached after finitely many steps, keep the sequence constant thereafter.

The residual Rademacher sum has squared coeficient sum at most one, and every one of its coeficients has magnitude at most $\sqrt { M _ { k } } \le \sqrt { M _ { k } + m _ { k } }$ . Thus Lemma 6.3 gives

$$
F _ { n } ( x ^ { ( k + 1 ) } ) - F _ { n } ( x ^ { ( k ) } ) \geq { \frac { 3 \kappa } { 1 6 } } ( M _ { k } - m _ { k } ) ^ { 2 } .\tag{54}
$$

The same averaging step decreases $q$ by

$$
q ( x ^ { ( k ) } ) - q ( x ^ { ( k + 1 ) } ) = \frac { 1 } { 2 } ( M _ { k } - m _ { k } ) ^ { 2 } .\tag{55}
$$

Eliminating $( M _ { k } - m _ { k } ) ^ { 2 }$ between the two estimates gives

$$
F _ { n } ( x ^ { ( k + 1 ) } ) - F _ { n } ( x ^ { ( k ) } ) \geq { \frac { 3 ( 5 { \sqrt { 2 } } - 7 ) } { 1 6 } } { \big ( } q ( x ^ { ( k ) } ) - q ( x ^ { ( k + 1 ) } ) { \big ) } .\tag{56}
$$

Since $q \geq 1 / n$ on the simplex, summing Equation (55) shows that $\begin{array} { r } { \sum _ { k } ( M _ { k } - m _ { k } ) ^ { 2 } < \infty } \end{array}$ , and hence $M _ { k } - m _ { k }  0$ . The mean coordinate remains $1 / n ,$ so

$$
\operatorname* { m a x } _ { i } \left| x _ { i } ^ { ( k ) } - { \frac { 1 } { n } } \right| \leq M _ { k } - m _ { k } \longrightarrow 0 ;
$$

therefore $x ^ { ( k ) } \to u _ { n }$ . The function $F _ { n }$ is a finite average of continuous functions of ${ \sqrt { x _ { i } } } ,$ and is continuous on the simplex. Summing Equation (56) through step K and then letting $K  \infty$ gives

$$
F _ { n } ( u _ { n } ) - F _ { n } ( x ^ { ( 0 ) } ) \geq \frac { 3 ( 5 \sqrt { 2 } - 7 ) } { 1 6 } \left( q ( x ^ { ( 0 ) } ) - \frac { 1 } { n } \right) \geq \frac { 1 } { 1 0 0 } \left( q ( x ^ { ( 0 ) } ) - \frac { 1 } { n } \right) .
$$

Together with Equation (43), this proves Equation (7). The upper bound in Equation (9) is supplied by Lemma 6.4, completing the theorem. □

Remark 6.5. The proof establishes dimension-free stability but does not determine the exact value of $C _ { 3 } ^ { \mathrm { { o p t } } }$ . The dimension-three vector above is a natural sharpness candidate. Any improvement of the constant $3 / 1 6$ in Lemma 6.1 immediately improves the lower bound obtained by this method.

Conjecture 6.6. We conjecture that, for every $n \geq 1$ and every $a \in \mathbb { R } ^ { n }$ with $\textstyle \sum _ { i } a _ { i } ^ { 2 } = 1$

$$
\mathbb { E } \left| \sum _ { i = 1 } ^ { n } a _ { i } \varepsilon _ { i } \right| ^ { 3 } \leq \mathbb { E } \left| \overline { { S } } _ { n } \right| ^ { 3 } - ( 5 \sqrt { 3 } - 6 \sqrt { 2 } ) \Delta _ { n } ( a ) .\tag{57}
$$

## 7 Concluding perspective

The fourth-power mass q exposes two complementary extremal geometries. At fixed $q ,$ the upper law consists of one Bernoulli spike and a Gaussian cloud; comparison with its endpoint distributions yields Gaussian stability. At fixed dimension, the fourth-order principle leaves one possible spike among otherwise equal coeficients, and the cubic-quotient lemma shows that the flat choice is optimal. The interval $4 < p < 5$ lies below the convexity threshold of the fixed-moment principle and instead requires the direct coeficientwise argument. At $p = 3 ,$ the exact dimension-free quadratic stability constant remains open, as stated in Remark 6.5 and Conjecture 6.6. For $0 < q < 1$ , the fixed-q envelope is a supremum in the closure of finite Rademacher sums rather than an attained finite-dimensional law. The results therefore distinguish the proved sharp envelopes and finite-dimensional optimizers from the remaining question of the exact third-moment constant.

## References

[1] R. O’Donnell, Analysis of Boolean Functions, Cambridge University Press, Cambridge, 2014, doi:10.1017/CBO9781139814782.

[2] D. Achlioptas, Database-friendly random projections: Johnson–Lindenstrauss with binary coins, J. Comput. System Sci. 66 (2003), no. 4, 671–687, doi:10.1016/S0022-0000(03)00025-4.

[3] A. Khintchine, Uber dyadische Br¨uche <sup>¨</sup> , Math. Z. 18 (1923), 109–116.

[4] U. Haagerup, The best constants in the Khintchine inequality, Studia Math. 70 (1981), no. 3, 231–283.

[5] P. Nayar and K. Oleszkiewicz, Khinchine type inequalities with optimal constants via ultra log-concavity, Positivity 16 (2012), no. 2, 359–371, doi:10.1007/s11117-011-0130-z.

[6] M. L. Eaton, A note on symmetric Bernoulli random variables, Ann. Math. Statist. 41 (1970), no. 4, 1223–1226, doi:10.1214/aoms/1177696897.

[7] R. Komorowski, On the best possible constants in the Khintchine inequality for $p \geq 3$ , Bull. London Math. Soc. 20 (1988), no. 1, 73–75, doi:10.1112/blms/20.1.73.

[8] A. De, I. Diakonikolas, and R. A. Servedio, A robust Khintchine inequality, and algorithms for computing optimal constants in Fourier analysis and high-dimensional geometry, SIAM J. Discrete Math. 30 (2016), no. 2, 1058–1094, doi:10.1137/130919143.

[9] J. Melbourne and C. Roberto, Quantitative form of Ball’s cube slicing in R<sup>n</sup> and equality cases in the min-entropy power inequality, Proc. Amer. Math. Soc. 150 (2022), no. 8, 3595–3611, doi:10.1090/proc/15944.

[10] A. Eskenazis, P. Nayar, and T. Tkocz, Resilience of cube slicing in $\ell _ { p } ,$ , Duke Math. J. 173 (2024), no. 17, 3377–3412, doi:10.1215/00127094-2024-0004.

[11] A. Eskenazis, P. Nayar, and T. Tkocz, Distributional stability of the Szarek and Ball inequalities, Math. Ann. 389 (2024), no. 2, 1161–1185, doi:10.1007/s00208-023-02669-9.

[12] A. Ch´avez and S. Sheng, <sup>´</sup> Stability of Khintchine-type inequalities via log-monotonicity, arXiv:2606.19313v2 [math.PR], 2026.

[13] J. Jakimiuk, Stability of Khintchine inequalities with optimal constants between the second and the p-th moment for $p \geq 3$ , Bernoulli 32 (2026), no. 3, 2524–2542.

[14] A. Bara´nski, D. Murawski, P. Nayar, and K. Oleszkiewicz, On the optimal $L _ { p } { - } L _ { 4 }$ Khintchine inequality, arXiv:2503.11869 [math.PR], 2025.

[15] D. Dzindzalieta and F. G¨otze, Half-spaces with influential variable, Theory Probab. Appl. 65 (2020), no. 1, 114–120, doi:10.1137/S0040585X97T989866.

## A Calculations for the Gaussian-shift chord

This appendix supplies the curvature and comparison computations used in Lemma 3.3.

## A.1 The curvature calculation

For $\beta > 0$ , write

$$
\begin{array} { r } { M _ { \beta } ( t ) = \mathbb { E } \left| G + t \right| ^ { \beta } . } \end{array}
$$

Gaussian integration by parts gives

$$
M _ { \beta + 2 } ( t ) = { \left( \beta + 1 + t ^ { 2 } \right) } M _ { \beta } ( t ) + t M _ { \beta } ^ { \prime } ( t ) .\tag{58}
$$

Let $\alpha = p - 4$ and $m = M _ { \alpha }$ . Applying Equation (58) twice, and using Equation (22), yields

$$
\begin{array} { l } { { M _ { \alpha + 2 } ( t ) = \bigl ( \alpha + 1 + t ^ { 2 } \bigr ) m ( t ) + t m ^ { \prime } ( t ) , } } \\ { { M _ { \alpha + 4 } ( t ) = \Bigl ( \bigl ( \alpha + 1 \bigr ) ( \alpha + 3 ) + 3 ( \alpha + 2 ) t ^ { 2 } + t ^ { 4 } \Bigr ) m ( t ) } } \\ { { \qquad + t \bigl ( 2 \alpha + 5 + t ^ { 2 } \bigr ) m ^ { \prime } ( t ) . } } \end{array}
$$

Now $h ( u ) = v ^ { \alpha + 4 } M _ { \alpha + 4 } ( t )$ , where $v = \sqrt { 1 - u ^ { 2 } }$ and $t = u / v$ . Since

$$
\frac { \mathrm { d } v } { \mathrm { d } u } = - t , \qquad \frac { \mathrm { d } t } { \mathrm { d } u } = v ^ { - 3 } ,
$$

two diferentiations and one further use of Equation (22) give Equation (23).

## A.2 The Riccati comparison

With

$$
R ( t ) = \frac { m ^ { \prime } ( t ) } { t m ( t ) } , \qquad Q ( t ) = \alpha \frac { 3 + ( \alpha + 2 ) t ^ { 2 } } { 3 + ( 2 \alpha + 3 ) t ^ { 2 } } ,
$$

Equation (22) gives Equation (25). For $Q _ { i }$ , bringing the two sides of Equation (26) to a common denominator gives

$$
\begin{array} { l } { { t Q ^ { \prime } ( t ) - \left[ \alpha - ( 1 + t ^ { 2 } ) Q ( t ) - t ^ { 2 } Q ( t ) ^ { 2 } \right] } } \\ { { \ } } \\ { { \ } = { \frac { \alpha { \left( \alpha + 1 \right) } { \left( \alpha + 3 \right) } t ^ { 4 } { \left( 4 + { \left( \alpha + 2 \right) } t ^ { 2 } \right) } } { { \left( 3 + { \left( 2 \alpha + 3 \right) } t ^ { 2 } \right) } ^ { 2 } } } > 0 . } \end{array}\tag{59}
$$

This proves Equation (26).

To justify the comparison at the singular endpoint $t = 0$ , set $P = Q - R$ and denote the positive right-hand side of Equation (59) by $E ( t )$ . Subtracting the two Riccati equations gives

$$
t P ^ { \prime } ( t ) + \bigl ( 1 + t ^ { 2 } + t ^ { 2 } ( Q ( t ) + R ( t ) ) \bigr ) P ( t ) = E ( t ) .
$$

The integrating factor for this equation is

$$
I ( t ) = t \exp \left( \int _ { 0 } ^ { t } r \big ( 1 + Q ( r ) + R ( r ) \big ) \mathrm { d } r \right) .
$$

Since $E ( r ) = O ( r ^ { 4 } )$ and $I ( \boldsymbol { r } ) = { \cal { O } } ( \boldsymbol { r } )$ near the origin, the integrand below is integrable there. The condition $P ( t ) \to 0$ as $t \downarrow 0$ and variation of constants give

$$
P ( t ) = \frac { 1 } { I ( t ) } \int _ { 0 } ^ { t } \frac { I ( r ) } { r } E ( r ) \mathrm { d } r .
$$

The integrand is positive, and therefore $P ( t ) > 0$ for every $t > 0 ,$ , as used in the main proof.

## B Calculations for the third-moment argument

## B.1 The two-point profile

We verify the monotonicity used in Lemma 6.2. Use the normalized notation from its proof, so $x + y = 1$ :

$$
u = { \sqrt { x } } + { \sqrt { y } } , \qquad v = { \sqrt { x } } - { \sqrt { y } } , \qquad u ^ { 2 } + v ^ { 2 } = 2 .
$$

Let the right-hand side of Equation (48) be denoted by $\mathcal G ( s )$ . From the two branches of $H$ , for $0 \leq s \leq v _ { : }$

$$
\mathcal { G } ^ { \prime } ( s ) = \frac { 3 } { 2 } s \big [ s + 2 ( \sqrt { 2 } - u - v ) \big ] .\tag{60}
$$

Now

$$
2 u + v = 2 { \sqrt { 2 - v ^ { 2 } } } + v \geq 2 { \sqrt { 2 } } ,
$$

because the left-hand side is concave on $[ 0 , 1 ]$ and its endpoint values are at least $2 \sqrt { 2 }$ . Hence $v \leq 2 ( u + v - \sqrt { 2 } )$ , and Equation (60) is nonpositive.

For $v \leq s \leq u .$ 2

$$
\mathcal { G } ^ { \prime } ( s ) = 3 ( \sqrt { 2 } - u ) s - \frac { 3 } { 2 } v ^ { 2 } = 3 ( \sqrt { 2 } - u ) \left( s - \frac { \sqrt { 2 } + u } { 2 } \right) \leq 0 ,\tag{61}
$$

since $s \leq u \leq ( \sqrt { 2 } + u ) / 2$ . Thus $\mathcal { G }$ is nonincreasing on $[ 0 , u ] ;$ in particular it is nonincreasing on $[ 0 , 1 ]$

For $u \leq s \leq \sqrt { 2 }$ , direct substitution gives

$$
\mathscr { G } ( s ) = \frac { 1 } { 2 } ( \sqrt { 2 } - s ) ^ { 3 } ,
$$

and for $s \geq \sqrt { 2 }$ all terms lie in the second branch of H, so $\mathcal { G } ( s ) = 0$ . This proves the remaining nonnegativity assertion used in the main proof.

## B.2 The dimension-three reduction

We give the two calculations used in the proof of Lemma 6.4. Again write $C _ { \star } = 5 \sqrt { 3 } - 6 \sqrt { 2 }$ . Fix $s = a + b + c$ and assume $a \geq b \geq c \geq 0 , a ^ { 2 } + b ^ { 2 } + c ^ { 2 } = 1$ . Since

$$
b + c = s - a , \qquad b ^ { 2 } + c ^ { 2 } = 1 - a ^ { 2 } ,
$$

we have

$$
b c = a ^ { 2 } - s a + { \frac { s ^ { 2 } - 1 } { 2 } } , \qquad a b c = a ^ { 3 } - { s a ^ { 2 } } + { \frac { s ^ { 2 } - 1 } { 2 } } a .
$$

The ordered feasible interval for a starts when $a = b ,$ at

$$
a _ { 0 } = \frac { 2 s + \sqrt { 6 - 2 s ^ { 2 } } } { 6 } .
$$

Moreover,

$$
{ \frac { d } { d a } } ( a b c ) = 3 \left( a - { \frac { 2 s - { \sqrt { 6 - 2 s ^ { 2 } } } } { 6 } } \right) \left( a - { \frac { 2 s + { \sqrt { 6 - 2 s ^ { 2 } } } } { 6 } } \right) \geq 0
$$

on that interval. Thus abc is minimized at $a = b ,$ as claimed.

For Equation (53), denote its left-hand side by $J ( z )$ . Diferentiation yields

$$
J ^ { \prime } ( z ) = \frac { z ( 1 - z ) } { ( z ^ { 2 } + 2 ) ^ { 3 } } \left( 3 z \sqrt { z ^ { 2 } + 2 } - 8 C _ { \star } ( z + 1 ) \right) .
$$

The factor in parentheses is strictly increasing on [0, 1], since its derivative is

$$
\frac { 6 ( z ^ { 2 } + 1 ) } { \sqrt { z ^ { 2 } + 2 } } - 8 C _ { \star } \geq 3 \sqrt { 2 } - 8 C _ { \star } > 0 .
$$

It is negative at $z = 0$ and positive at $z = 1$ . Consequently J first decreases and then increases, and therefore has no interior maximum. Finally,

$$
J ( 0 ) = J ( 1 ) = \sqrt { 2 } + \frac { C _ { \star } } { 2 } ,
$$

which proves Equation (53).

## C The scalar estimates

The appendix records the elementary estimates used in Lemmas 2.2 and 2.4. No numerical optimization is involved.

## C.1 The estimate for small fourth-power mass

Proof of Lemma 2.2. At the point $x _ { 0 } = 2 ^ { - 1 / 2 }$ , put

$$
\rho = 1 - \frac { 2 x _ { 0 } } { 3 } = 1 - \frac { \sqrt { 2 } } { 3 } .
$$

The diference between the two sides of Equation (13) is

$$
D ( p ) : = \frac { 1 - \rho ^ { ( p - 2 ) / 2 } } { x _ { 0 } } - 1 + \frac { 1 } { \mu _ { p } } .
$$

Thus the desired inequality is $D ( p ) \geq 0$ . Substitution of $\mu _ { 4 } = 3$ gives $D ( 4 ) = 0$ , so it remains to prove that D is strictly increasing.

For this purpose, recall the logarithmic derivative of the Gaussian moment:

$$
\ell ( p ) : = { \frac { \mathrm { d } } { \mathrm { d } p } } \log \mu _ { p } = { \frac { 1 } { 2 } } \log 2 + { \frac { 1 } { 2 } } \psi \left( { \frac { p + 1 } { 2 } } \right) ,
$$

where $\psi$ is the digamma function. Diferentiation gives

$$
D ^ { \prime } ( p ) = - \frac { \rho ^ { ( p - 2 ) / 2 } \log \rho } { 2 x _ { 0 } } - \frac { \ell ( p ) } { \mu _ { p } } .\tag{62}
$$

Both $- \rho ^ { ( p - 2 ) / 2 } \log \rho / ( 2 x _ { 0 } )$ and $\ell ( p ) / \mu _ { p }$ are positive. Hence $D ^ { \prime } ( p ) > 0$ is equivalent to the following ratio being greater than one:

$$
\mathcal { R } ( p ) = \frac { - \rho ^ { ( p - 2 ) / 2 } \log \rho / ( 2 x _ { 0 } ) } { \ell ( p ) / \mu _ { p } } .
$$

The function ℓ is increasing and $\ell ^ { \prime }$ is decreasing; this follows from the positivity of the trigamma function and the negativity of its derivative. The special values at $p = 4$ satisfy

$$
\ell ( 4 ) = \frac { 4 } { 3 } - \frac { \gamma _ { E } + \log 2 } { 2 } > \frac { 2 } { 3 } , \qquad \ell ^ { \prime } ( 4 ) = \frac { \pi ^ { 2 } } { 8 } - \frac { 1 0 } { 9 } < \frac { 1 } { 8 } ,\tag{63}
$$

and $\rho > e ^ { - 2 / 3 }$ ; these elementary bounds are verified in Lemma C.1. Consequently, for $p \geq 4$

$$
\frac { \mathrm { d } } { \mathrm { d } p } \log \mathcal { R } ( p ) = \frac { 1 } { 2 } \log \rho + \ell ( p ) - \frac { \ell ^ { \prime } ( p ) } { \ell ( p ) } > - \frac { 1 } { 3 } + \frac { 2 } { 3 } - \frac { 3 } { 1 6 } = \frac { 7 } { 4 8 } > 0 .
$$

Thus R is strictly increasing. It remains to check that $\mathcal { R } ( 4 ) > 1$ . From $\gamma _ { E } + \log 2 > 1 9 / 1 5$ 2

$$
\frac { \ell ( 4 ) } { \mu _ { 4 } } = \frac { \ell ( 4 ) } { 3 } < \frac { 7 } { 3 0 } .
$$

Writing $y = { \sqrt { 2 } } / 3$ , so $\rho = 1 - y$ , the first four terms of $\textstyle - \log ( 1 - y ) = \sum _ { k > 1 } y ^ { k } / k$ give

$$
- \frac { \rho \log \rho } { \sqrt { 2 } } > \frac { 7 7 - 1 4 \sqrt { 2 } } { 2 4 3 } > \frac { 7 } { 3 0 } ,
$$

where the last inequality follows at once from ${ \sqrt { 2 } } < 1 0 / 7 .$ These two estimates show that $\mathcal { R } ( 4 ) > 1$ , and monotonicity gives $\mathcal { R } ( p ) > 1$ for every $p \geq 4$ . By Equation (62), this means $D ^ { \prime } ( p ) > 0$ . Together with $D ( 4 ) = 0$ , this proves the lemma and its strict form for $p > 4$ □

## C.2 The estimate for large fourth-power mass

Proof of Lemma ${ 2 . 4 } .$ First, Equation (63) and the estimate $\gamma _ { E } + 3 \log 2 < 8 / 3$ in Lemma C.1 give $\ell ( 4 ) > \log 2 $

We begin with the logarithmic estimate

$$
3 \log 2 \frac { 1 - q } { 3 - 2 q } + \frac { 1 } { 2 } \log \frac { 3 - 2 q } { 1 5 - 3 0 q + 1 6 q ^ { 3 / 2 } } \geq 0 .\tag{64}
$$

For Equation (64), put $x = { \sqrt { q } }$ and, temporarily, set

$$
A = 3 - 2 x ^ { 2 } , \qquad B = 1 5 - 3 0 x ^ { 2 } + 1 6 x ^ { 3 } .
$$

These temporary symbols are $A = A ( q )$ and $B = B ( q )$ from Equation (14), expressed in terms of x. Since $B ^ { \prime } ( x ) = 1 2 x ( 4 x - 5 ) < 0$ , we have $B ( x ) \geq B ( 1 ) = 1$ , so the logarithm below is well defined. Define

$$
K ( x ) = 3 ( \log 2 ) { \frac { 1 - x ^ { 2 } } { A } } + { \frac { 1 } { 2 } } \log { \frac { A } { B } } , \qquad 2 ^ { - 1 / 2 } \leq x \leq 1 .
$$

Let

$$
N ( x ) = 2 A ( 1 5 - 1 8 x + 4 x ^ { 3 } ) - 3 ( \log 2 ) B .
$$

Diferentiation gives

$$
K ^ { \prime } ( x ) = { \frac { 2 x N ( x ) } { A ^ { 2 } B } }\tag{65}
$$

and

$$
N ^ { \prime \prime } ( x ) = ( - 1 2 0 + 1 8 0 \log 2 ) + ( 5 7 6 - 2 8 8 \log 2 ) x - 3 2 0 x ^ { 3 } .
$$

$\mathrm { O n }$ the interval in question, $N ^ { \prime \prime \prime } ( x ) \ < \ 0$ , so $N ^ { \prime \prime }$ is decreasing. Since $x \ \leq \ 1$ and $N ^ { \prime \prime } ( 1 ) =$ $1 3 6 - 1 0 8 \log 2 > 0$ , it follows that $N ^ { \prime \prime } ( x ) > 0$ throughout the interval. Thus N is strictly convex. Moreover,

$$
N ( 1 ) = 2 - 3 \log 2 < 0 ,
$$

and

$$
N ( 2 ^ { - 1 / 2 } ) = 6 0 - 3 2 \sqrt { 2 } - 1 2 \sqrt { 2 } \log 2 > 0 ;
$$

for the latter it is enough to use log $2 < 7 / 1 0$ and ${ \sqrt { 2 } } < 1 0 / 7$ . Because $N ^ { \prime }$ is increasing, N decreases up to at most one minimum and then increases. The endpoint signs force exactly one zero, on the decreasing branch. By Equation (65), K first increases and then decreases. Since

$$
K ( 2 ^ { - 1 / 2 } ) = K ( 1 ) = 0 ,
$$

we conclude that $K \geq 0$ , which is Equation (64).

Fix $q \in [ 1 / 2 , 1 ]$ . By Equation $( 2 ) , \Lambda _ { p } ( q )$ is the p-th moment of a nonnegative random variable equal to 1 with probability q and to |G| with probability $1 - q .$ . Therefore $p \mapsto \log \Lambda _ { p } ( q )$ is convex. Since $\Lambda _ { 4 } ( q ) = { \cal { A } } ( q )$ and $\partial _ { p } \Lambda _ { p } ( q ) | _ { p = 4 } = 3 ( 1 - q ) \ell ( 4 )$ , its supporting tangent at $p = 4$ , followed by Equation (64), gives

$$
\begin{array} { l } { { \displaystyle \log \Lambda _ { p } ( q ) \geq \log A ( q ) + ( p - 4 ) \frac { 3 ( 1 - q ) \ell ( 4 ) } { A ( q ) } } } \\ { { \displaystyle \geq \log A ( q ) + \frac { p - 4 } { 2 } \log \frac { B ( q ) } { A ( q ) } . } } \end{array}
$$

Hence

$$
A ( q ) ^ { ( 6 - p ) / 2 } B ( q ) ^ { ( p - 4 ) / 2 } \leq \Lambda _ { p } ( q ) , \qquad p \geq 4 .
$$

## C.3 Elementary constants

Here $\gamma _ { E }$ is Euler’s constant, and $\textstyle H _ { n } : = \sum _ { k = 1 } ^ { n } k ^ { - 1 }$ denotes the n-th harmonic number.

Lemma C.1. The following estimates hold:

$$
\frac { 8 3 } { 1 2 0 } < \log 2 < \frac { 1 7 3 3 } { 2 5 0 0 } < \frac { 7 } { 1 0 } , \qquad \frac { 2 3 } { 4 0 } < \gamma _ { E } < \frac { 7 } { 1 2 } .\tag{66}
$$

Consequently,

$$
\frac { 1 9 } { 1 5 } < \gamma _ { E } + \log 2 < \frac { 4 } { 3 } , \qquad \gamma _ { E } + 3 \log 2 < \frac { 8 } { 3 } .
$$

Moreover,

$$
\frac { \pi ^ { 2 } } { 8 } - \frac { 1 0 } { 9 } < \frac { 1 } { 8 } , \qquad 1 - \frac { \sqrt { 2 } } { 3 } > e ^ { - 2 / 3 } .
$$

Proof. The identity

$$
\log 2 = 2 \sum _ { k = 0 } ^ { \infty } { \frac { 1 } { ( 2 k + 1 ) 3 ^ { 2 k + 1 } } }
$$

with three retained terms and a geometric bound on the tail gives the two stated bounds for log 2. The standard Euler–Maclaurin inequalities

$$
H _ { n } - \log n - { \frac { 1 } { 2 n } } < \gamma _ { E } < H _ { n } - \log n - { \frac { 1 } { 2 n } } + { \frac { 1 } { 1 2 n ^ { 2 } } }
$$

give the upper bound at $n = 1$ . For the lower bound, write log $7 = 3 \log 2 + \log ( 7 / 8 )$ ; the preceding bound for log 2, together with the first three terms of the power series for $\log ( 1 - z )$ at $z = 1 / 8$ gives log $7 < 1 0 9 / 5 6$ . The lower Euler–Maclaurin bound at $n = 7$ then gives $\gamma _ { E } > 2 3 / 4 0$ . The three consequences displayed immediately follow.

Finally, $\pi < 2 2 / 7$ gives $\pi ^ { 2 } / 8 - 1 0 / 9 < 1 / 8$ . Also, ${ \sqrt { 2 } } < 1 0 / 7$ gives $1 - { \sqrt { 2 } } / 3 > 1 1 / 2 1$ , while the first four terms of the exponential series give $e ^ { 2 / 3 } > 2 1 / 1 1$ . Hence $1 - \sqrt { 2 } / 3 > e ^ { - \dot { 2 } / 3 }$ □

Remark C.2. The split at $q = 1 / 2$ is deliberately unoptimized. It is the point at which the two scalar arguments have transparent endpoint checks and meet without numerical optimization.