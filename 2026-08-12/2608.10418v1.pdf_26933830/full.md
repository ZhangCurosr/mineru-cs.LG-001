# A lower bound for stepsize-based acceleration of gradient descent

Jianhao Ma<sup>∗</sup>

Yuxin Chen<sup>†</sup>

August 12, 2026

## Abstract

Recent work has shown that, for smooth convex optimization, plain gradient descent can be accelerated from its textbook convergence rate of ${ \cal O } ( T ^ { - 1 } )$ (where � denotes the number of iterations) to $O \big ( T ^ { - \log _ { 2 } ( 1 + \sqrt { 2 } ) } \big )$ using carefully designed stepsize schedules alone, without resorting to momentum or other algorithmic modifications. Despite this progress, however, little was known about lower bounds for such methods beyond the classical $\Omega ( T ^ { - 2 } )$ benchmark for general first-order methods. In this work, we present a new lower bound of $\Omega ( T ^ { - 1 . 9 3 1 9 } )$ for the last-iterate convergence rate of gradient descent with predetermined nonnegative stepsize schedules. This result provides rigorous evidence that stepsize schedules alone cannot accelerate plain GD to the optima ${ \cal O } ( T ^ { - 2 } )$ convergence rate. The proof was developed by GPT-5.6 Sol Pro under the authors’ guidance.

## 1 Introduction

Consider the unconstrained smooth convex optimization problem

$$
{ \mathrm { m i n i m i z e } } _ { x \in \mathbb { R } ^ { d } } \quad f ( x ) ,\tag{1}
$$

where $f : \mathbb { R } ^ { d } $ R is a convex function whose gradient is �-Lipschitz. Gradient descent (GD), a canonical first-order method for solving (1), generates a sequence of iterates according to

$$
x _ { t + 1 } = x _ { t } - \eta _ { t } \nabla f ( x _ { t } ) , \qquad 0 \leq t < T ,\tag{2}
$$

where $x _ { 0 }$ is the initialization, � is the prescribed number of iterations, and $\{ \eta _ { t } \} _ { 0 \le t < T }$ is the stepsize schedule. With the standard constant stepsize $\eta _ { t } = 1 / L$ , the suboptimality of its last iterate decreases at the rate ${ \cal O } ( T ^ { - 1 } )$ (Bubeck, 2015).

## 1.1 Motivation: stepsize-based acceleration of GD

Faster convergence has traditionally required the use of momentum or other modifications to the GD update; most prominently, Nesterov’s accelerated method attains the optimal ${ \cal O } ( T ^ { - 2 } )$ rate (Nesterov, 1983). Surprisingly, a recent line of work has shown that plain GD can also be accelerated through carefully designed stepsize schedules alone $- \mathrm { e . g . }$ schedules with occasional long steps — even though such steps may temporarily worsen the objective value (Altschuler and Parrilo, 2025b; Grimmer et al., 2025b).

The strongest known constructions are based on the silver ratio $\rho _ { \mathrm { s i l } } = 1 + { \sqrt { 2 } } .$ Specifically, the stepsize schedules proposed by Altschuler and Parrilo (2025b); Grimmer et al. (2025b) achieve the last-iterate convergence rate of

$$
O \left( T ^ { - \log _ { 2 } \rho _ { \mathrm { s i l } } } \right) \qquad \left( { \mathrm { w i t h ~ l o g } } _ { 2 } \rho _ { \mathrm { s i l } } \approx 1 . 2 7 1 5 \right)
$$

whenever the prescribed number of iterations satisfies $T = 2 ^ { k } - 1$ for some integer $k \geq 1$ . Subsequent concatenation and composition techniques extend the same convergence rate to arbitrary prescribed numbers of iterations (Zhang and Jiang, 2024; Grimmer et al., 2025a). These methods employ the standard GD update with predetermined nonnegative stepsize schedules containing occasional long steps, chosen with knowledge of the total number of iterations �. These developments naturally raise the following question:

How fast can plain GD converge when its stepsize schedule is chosen in advance with knowledge of the total number ofiterations?

Altschuler and Parrilo (2025b) conjectured that the optimal answer is given by the silver exponent $\log _ { 2 } ( 1 + { \sqrt { 2 } } )$ , but this conjecture remains open.

Resolving this question requires lower bounds tailored specifically to GD with predetermined stepsize schedules. The classical $\mathsf { \bar { Q } } ( T ^ { - \bar { 2 } } )$ lower bound for deterministic first-order methods applies to every method using at most � oracle calls and therefore does not distinguish plain GD from accelerated methods (Drori, 2016). Existing sharper lower bounds for GD with potentially large stepsizes either apply only to basic �-composable schedules (Grimmer et al., 2025a), or concern the “anytime” setting, where a single infinite stepsize schedule must be fixed in advance and perform well regardless of when the algorithm is stopped (Tsai et al., 2026). In fact, existing lower bounds do not preclude plain GD with suitably designed predetermined stepsize schedules from attaining the optimal ${ \cal O } ( T ^ { - 2 } )$ convergence rate.

## 1.2 This paper

Main contributions. In this work, we make progress on this question for the class of predetermined stepsize schedules designed for a prescribed horizon �, without imposing any restrictions on the magnitudes and ordering of the nonnegative stepsizes. Our main result establishes a new lower bound of

$$
\Omega \left( T ^ { - p } \right) \qquad { \mathrm { w h e r e ~ } } p > p _ { \star } = \sqrt { 2 + \sqrt { 3 } } \approx 1 . 9 3 1 9
$$

for the worst-case last-iterate convergence rate of gradient descent with arbitrary predetermined nonnegative stepsize schedules. The proof constructs hard instances in dimension at most $T + 1$ , and applies to schedules with zero or arbitrarily large stepsizes, arbitrary ordering, and no descent or monotonicity assumptions. To the best of our knowledge, this is the first rigorous evidence that adjusting predetermined stepsize schedules alone is insuficient to accelerate plain gradient descent to the optimal ${ \cal O } ( T ^ { - 2 } )$ convergence rate. The remaining gap between the best-known upper-bound exponent, $\log _ { 2 } ( 1 + { \sqrt { 2 } } ) \approx 1 . 2 7 1 5 .$ , and our impossibility threshold $p _ { \star }$ , remains unresolved.

Organization. The remainder of this paper is organized as follows. Section 2 presents the main theorem, along with a high-level description of the proof strategy. Section 3 reviews related work. Sections 4, 5, and 6 provide the analysis (which develops, respectively, the realization, the order-free matching bound, and the final cutof argument and scaling). Section 7 concludes the paper with additional discussion, whereas the appendices contain the remaining technical analysis.

Notation. Throughout this paper, � is a positive integer denoting the ambient dimension. We equip $\mathbb { R } ^ { d }$ with its Euclidean inner product and norm $\| \cdot \|$ . We let $L > 0$ denote the smoothness parameter, such that for all $x , y \in \mathbb { R } ^ { d }$

$$
\| \nabla f ( x ) - \nabla f ( y ) \| \leq L \| x - y \| .
$$

Additionally, let argmin � denote the set of minimizers of a function $f .$ We also let conv(�) denote the convex hull of a set �.

## 2 Main results

In this section, we formalize the setting and state our main lower bound. Throughout, we consider gradient descent with predetermined nonnegative stepsize schedules designed for a prescribed horizon �.

## 2.1 Main theorem

Let $\mathcal { F } _ { 0 , L } ( \mathbb { R } ^ { d } )$ denote the class of diferentiable convex functions from $\mathbb { R } ^ { d }$ to R that admit at least one minimizer and have �-Lipschitz continuous gradients. We consider the smooth convex optimization problem (1), where the objective function is assumed to obey $f \in \mathcal { F } _ { 0 , L } ( \mathbb { R } ^ { d } )$ . Given a prescribed horizon $T \geq 1$ and an initial point $\boldsymbol { x } _ { 0 } \in \mathbb { R } ^ { d }$ , we consider running plain GD (2) using a predetermined, nonnegative stepsize schedule $\eta = ( \eta _ { 0 } , \ldots , \eta _ { T - 1 } ) \in \mathbb { R } _ { > 0 } ^ { T }$ . The stepsize schedule is fixed before the problem dimension, objective function, and minimizer are chosen. Our lower bound concerns the convergence rate of the last iterate $x _ { T } ;$ intermediate objective values need not decrease.

Theorem 2.1. Let $p _ { \star } : = \sqrt { 2 + \sqrt { 3 } } .$ . For every $p \in ( p _ { \star } , 2 )$ , there exists a constant $c _ { p } > 0 ,$ depending only on $p ,$ such that the following holds. For every integer $T \geq 1 _ { \mathrm { { } } }$ , every $L , R > 0 ,$ and every predetermined nonnegative schedule $\eta = ( \eta _ { 0 } , \ldots , \eta _ { T - 1 } ) \in \mathbb { R } _ { \ge 0 } ^ { T } ,$ , there exists an integer � with $1 \leq d \leq T + 1$ such that, for every prescribed initial point $\boldsymbol { x } _ { 0 } \in \mathbb { R } ^ { d }$ , there exist afunction $f \in \mathcal { F } _ { 0 , L } ( \mathbb { R } ^ { d } )$ and a minimizer $x _ { \star } \in$ argmin � satisfying $\| x _ { 0 } - x _ { \star } \| = R$ such that GD $( c f . \ ( 2 ) )$ initialized at $x _ { 0 }$ with schedule � satisfies

$$
f ( x _ { T } ) - f ( x _ { \star } ) \geq c _ { p } L R ^ { 2 } ( T + 1 ) ^ { - p } .
$$

Remark 2.2. Theorem 2.1 establishes the lower bound separately for every prescribed horizon and every predetermined stepsize schedule; the adversarial instance may depend on both. Additionally, it does not establish the endpoint lower bound $\Omega ( T ^ { - p _ { \star } } )$ , corresponding to the case $p = p _ { \star }$ . Instead, the theorem applies only to exponents $p > p _ { \star }$

Comparison with lower bounds for anytime schedules. Our result should be contrasted with the concurrent anytime lower bound of Tsai et al. (2026), which rules out a worst-case last-iterate convergence rate of $o ( T ^ { - 4 / 3 } )$ for any single infinite sequence of strictly positive stepsizes. That result requires a single stepsize schedule to provide convergence guarantees uniformly over all stopping times, whereas Theorem 2.1 allows a diferent predetermined stepsize schedule for each prescribed horizon. Therefore, the anytime lower bound therein does not imply our lower bound.

## 2.2 Proof strategy

In this subsection, we give a roadmap for the proof of Theorem 2.1; the full proof is provided in Sections $4 { - } 6 .$ After normalizing the stepsizes as $h _ { t } = L \eta _ { i }$ , we assume without loss of generality that $L = R = 1$ . We decompose each normalized stepsize into its capped part and its excess part, and define

$$
y _ { t } : = ( h _ { t } - 1 ) _ { + } , \qquad B : = 1 + \sum _ { t = 0 } ^ { T - 1 } \operatorname* { m i n } \{ h _ { t } , 1 \} , \qquad r : = \big | \{ t : y _ { t } > 0 \} \big | .
$$

We call a step long if $h _ { t } > 1$ , equivalently if $y _ { t } > 0$ . Thus � is the base mass contributed by the unit-capped schedule, and � is the number of long steps. The proof proceeds in three stages. First, we construct an explicit hard instance tailored to selected long steps in the prescribed schedule. Second, we derive a lower bound on the suboptimality gap that depends only on the magnitudes of these long steps and is independent of their temporal order. Third, we combine a rank cutof argument with a Lyapunov estimate to obtain the desired convergence lower bound.

An explicit hard instance from selected long steps. Choose any � long steps in their original temporal order. They divide the schedule into $m + 1$ blocks. Section 4 associates with these blocks positive scales $H _ { 0 } , \ldots , H _ { m }$ and transition factors $\chi _ { 0 } , \ldots , \chi _ { m - 1 }$ . Here $H _ { i }$ records the displacement accumulated in block $i ,$ while $\chi _ { i }$ is the largest squared-amplitude ratio for which the projection comparisons in the construction remain valid. The construction places orthogonal anchors

$$
X _ { i } = \lambda _ { i } e _ { i } , \qquad \lambda _ { 0 } = 1 , \qquad \lambda _ { i + 1 } ^ { 2 } = \chi _ { i } \lambda _ { i } ^ { 2 } ,
$$

where $e _ { 0 } , \ldots , e _ { m }$ is an orthonormal basis of $\mathbb { R } ^ { m + 1 }$ . The block gradients $g _ { i }$ are chosen so that the trajectory moves along the ray $X _ { i } - \mathbb { R } _ { \geq 0 } g$ <sub>�</sub> during block $i ,$ and the selected long step at the end of the block sends the iterate exactly to the next anchor:

$$
X _ { i } - H _ { i } g _ { i } = X _ { i + 1 } \qquad ( 0 \leq i < m ) .
$$

The terminal gradient is chosen analogously along the final coordinate. Let $K : = \operatorname { c o n v } \{ 0 , g _ { 0 } , \dots , g _ { m } \}$ , and define the hard instance as the Moreau envelope of the support function of $K \colon$

$$
F ( x ) : = \operatorname* { m i n } _ { z \in \mathbb { R } ^ { m + 1 } } \left\{ \sigma _ { K } ( z ) + { \frac { 1 } { 2 } } \| x - z \| ^ { 2 } \right\} , \qquad \sigma _ { K } ( z ) : = \operatorname* { m a x } _ { g \in K } \langle g , z \rangle .
$$

The key property of this choice is the Moreau identity $\nabla F ( x ) = \Pi _ { K } ( x )$ . Projection comparisons show that the vertex $g _ { i }$ remains the projection throughout block �. Hence the unselected steps move along the ray $X _ { i } - \mathbb { R } _ { \geq 0 } g _ { i }$ , while the selected long step lands exactly at the next orthogonal anchor. At the terminal time, the construction yields

$$
F ( x _ { T } ) - F ( 0 ) = \frac { \lambda _ { m } ^ { 2 } } { 2 H _ { m } } = \frac { 1 } { 2 H _ { m } } \prod _ { i = 0 } ^ { m - 1 } \chi _ { i } .
$$

Maximizing $H _ { m } ^ { - 1 } \prod _ { i } \chi _ { i }$ over all admissible choices leads to a key functional $C _ { T } ( h )$ . The empty selection contributes $( 1 + 2 \textstyle \sum _ { t } h _ { t } ) ^ { - 1 }$ ; in particular, it already yields an $\Omega ( T ^ { - 1 } )$ gap when all steps are short. The remaining task is to show that long steps cannot make $C _ { T } ( h )$ too small.

Removing temporal order by two matchings. The main obstacle is that each factor $\chi _ { i }$ couples two neighboring blocks, so a chain value depends on the temporal order of the long steps, not only on their sizes. To eliminate this temporal dependence, rank the positive excesses as $a _ { 1 } \geq \dots \geq a _ { r } > 0$ and define the residual schedule mass $\begin{array} { r } { D _ { q } : = B + \sum _ { s = q + 1 } ^ { r } a _ { s } } \end{array}$ . For a fixed $2 \leq q \leq r$ , select the $q$ largest excesses while retaining their original temporal order. $\mathbf { A }$ budget-equalization argument converts the reciprocal of the corresponding chain value into the edge product of a path whose vertex weights are normalized reciprocals of the selected excesses. The odd and even edges of this path form two matchings. Replacing each by the corresponding optimal matching value removes the temporal order completely, at the cost of upper-bounding the reciprocal chain value. Therefore, after inversion, we obtain a lower bound on $C _ { T } ( h )$ . If $\mu _ { q }$ denotes the resulting geometric mean cost per edge, then

$$
C _ { T } ( h ) \geq \frac { q } { 2 D _ { q } ( q - 1 ) \mu _ { q } ^ { q } } .
$$

Thus a bound of the form $\mu _ { q } < \rho < 1$ immediately creates the favorable factor $\rho ^ { - q }$ in the lower bound.

The single statistic

$$
\zeta _ { q } : = \frac { D _ { q } } { q ^ { 2 } } \sum _ { s = 1 } ^ { q } \frac { 1 } { a _ { s } }
$$

controls the total normalized reciprocal weight of the selected excesses, and therefore controls the matching cost $\mu _ { q }$ . A total-weight estimate for the two matchings shows that, when $\zeta _ { q } \leq \vartheta$ and $q$ is large, $\mu _ { q }$ is bounded above by $2 \vartheta + 2 \vartheta ^ { 2 }$ plus an error that vanishes as $q \to \infty$ . This is the only property of the matching problems needed in the final rank argument.

Rank cutof and mass growth. Fix $p \in ( p _ { \star } , 2 )$ and set $\vartheta : = ( p ^ { 2 } - 1 ) ^ { - 1 }$ , which satisfies $2 \vartheta + 2 \vartheta ^ { 2 } < 1$ . Thus, for all suficiently large ranks, one can choose a fixed $\rho < 1$ such that $\zeta _ { q } \leq \vartheta$ forces $\mu _ { q } < \rho$ . This produces a useful dichotomy. If $\mu _ { q } < \rho$ , the matching bound already gives a large value of $C _ { T } ( h )$ through the factor $\rho ^ { - q }$ . Otherwise $\zeta _ { q } > \vartheta$ . In this second regime, define

$$
\nu _ { q } : = \frac { q a _ { q } } { D _ { q } } .
$$

Since $D _ { q - 1 } / D _ { q } = 1 + \nu _ { q } / q$ , the quantity $\nu _ { q }$ measures how quickly the residual schedule mass grows as the rank cutof decreases.

The pointwise bound on $\nu _ { q }$ is not suficient by itself. The decisive additional input is the exact adjacent-rank recursion

$$
\zeta _ { q + 1 } = \frac { q ^ { 2 } \zeta _ { q } } { ( q + 1 ) ( q + 1 + \nu _ { q + 1 } ) } + \frac { 1 } { ( q + 1 ) \nu _ { q + 1 } } , \qquad 1 \leq q < r .
$$

If $\nu _ { q }$ were roughly constant and equal to $\nu ,$ this recursion would drive $\zeta _ { q }$ toward $1 / ( \nu ( \nu + 2 ) )$ . Maintaining $\zeta _ { q } > \vartheta$ therefore limits the sustainable growth rate to $\nu \leq p - 1$

Section 6 makes this heuristic argument rigorous with the aid of a Lyapunov potential. Telescoping its one-step drift and using the mass-ratio identity yield a constant $K _ { p }$ , depending only on $p ,$ such that

$$
D _ { k } k ^ { p - 1 } \le K _ { p } B r ^ { p - 1 }
$$

along every interval on which the second alternative persists.

The proof now scans the ranks until either the favorable matching alternative occurs or the Lyapunov estimate controls all but a bounded number of excesses; in the latter case, the remaining excesses are handled directly. In both cases,

$$
C _ { T } ( h ) \geq \frac { c _ { p } } { B ( r + 1 ) ^ { p - 1 } } \geq c _ { p } ( T + 1 ) ^ { - p } .
$$

The threshold exponent. The matching and mass-growth arguments impose competing requirements on $\vartheta \colon$ the Lyapunov analysis gives $p = \sqrt { 1 + \vartheta ^ { - 1 } }$ , whereas the matching estimate requires

$$
2 \vartheta + 2 \vartheta ^ { 2 } < 1 \quad \Longleftrightarrow \quad \vartheta < \frac { \sqrt { 3 } - 1 } { 2 } \quad \Longleftrightarrow \quad p > \sqrt { 2 + \sqrt { 3 } } = p _ { \star } .
$$

At $p = p _ { \star }$ , the interval $( 2 \vartheta + 2 \vartheta ^ { 2 } , 1 )$ from which � must be chosen collapses. This explains why Theorem 2.1 establishes $\Omega ( T ^ { - } { } ^ { p } )$ for every $p \in ( p _ { \star } , 2 )$ , but is not yet able to reach the endpoint $p = p _ { \star }$ r

## 3 Related work

Classical bounds and horizon-dependent analysis. For smooth convex optimization, classical worst-case theory has established an $O ( L R ^ { 2 } / T )$ convergence guarantee for plain GD with the constant stepsize $\eta = 1 / L$ (Bubeck, 2015; Beck, 2017), while accelerated first-order methods achieve the optimal $O ( L R ^ { 2 } / T ^ { 2 } )$ rate (Nesterov, 1983). In dimensions $d \ge T + 1$ , the exact information-based minimax risk also scales on the order of $L R ^ { 2 } / T ^ { 2 }$ (Drori, 2016) and is attained by the Optimized Gradient Method (Drori and Teboulle, 2014; Kim and Fessler, 2016). Since this is an oracle lower bound—that is, it applies to the entire class of first-order methods that access the objective only through a first-order oracle—it characterizes the fundamental limits of first-order methods, rather than those of plain GD itself. A complementary line of work studies lower bounds for restricted classes of first-order methods. For instance, Arjevani and Shamir (2016) showed that restricting GD to time-invariant stepsizes cannot improve upon the classical ${ \cal O } ( T ^ { - 1 } )$ convergence rate, whereas our result allows arbitrary predetermined stepsize schedules.

Another related line of work studies the exact worst-case performance of first-order methods through the performance estimation problem (PEP) framework, introduced by Drori and Teboulle (2014). Building on smooth interpolation, Taylor et al. (2017) showed that, when $d \ge T + 2$ , the PEP can be formulated as a finite-dimensional semidefinite program (SDP); an additional rank constraint is required in smaller dimensions. For GD on �-smooth convex objectives, Daccache (2019) derived the exact worst-case two-step objective-gap formula for $( L \eta _ { 0 } , L \eta _ { 1 } ) \in [ 0 , 1 ] ^ { 2 }$ and conjectured a formula on $[ 0 , 2 ] ^ { 2 }$ , together with partial extensions to longer schedules. Diego (2022) subsequently constructed explicit lower-bound instances attaining conjectured expressions in several two-, three-, and longer-step regimes. Overall, exac worst-case characterizations of GD with variable stepsize schedules remain available only in a limited number of special cases. Branch-and-bound and local PEP methods have also been proposed for computing improved stepsize schedules for prescribed horizons (Das Gupta et al., 2024; Kamri et al., 2025); note, however, that the local method does not certify global optimality, and neither approach yields an asymptotic characterization of the achievable convergence exponent.

Additionally, Kim (2024) derived the exact worst-case formula for the terminal objective value of constant-step GD on smooth convex and strongly convex functions when $\eta \in ( 0 , 2 / L )$ ). Rotaru et al. (2026) established the corresponding exact formula for the terminal gradient norm. For possibly nonconvex �-smooth objectives with a finite minimum, Abbaszadehpeivasti et al. (2022) proved a PEP upper bound on $\mathrm { m i n } _ { 0 \leq k \leq T } \| \nabla f ( x _ { k } ) \|$ for prescribed $t _ { k } \in ( 0 , \sqrt { 3 } / L )$ and showed it to be exact when al $t _ { k } \in ( 0 , 1 / L ]$ . None of these results provides a uniform worst-case analysis of the last-iterate objective value for arbitrary nonconstant stepsize schedules over arbitrary horizons, particularly when stepsizes larger than $2 / L$ are allowed.

Acceleration by (occasional) long steps. Reciprocal Chebyshev steps have long been known to achieve ${ \cal O } ( \sqrt { \kappa } \log ( 1 / \varepsilon ) )$ complexity for Richardson iteration on linear systems with positive-definite matrices, where � denotes the condition number (Young, 1953). Fractal orderings of these steps further improve intermediate stability under bounded additive perturbations (Agarwal et al., 2021). These spectral arguments, however, do not yield worst-case guarantees for general smooth convex optimization.

More recently, several works have shown that carefully designed stepsize schedules with occasional long steps can improve the performance of GD beyond the quadratic setting. PEP-certified periodic schedules improve the constant in the classical $T ^ { - 1 }$ convergence bound (Grimmer, 2024), while recursive long-step (silver) schedules improve the dependence on the condition number in the strongly convex setting (Altschuler and Parrilo, 2025a). The left-heavy schedule, obtained by reversing the right-heavy schedule of Grimmer, Shu, and Wang, achieves the same accelerated exponent for the squared norm of the terminal gradient (Grimmer et al., 2025b). Related silver acceleration results have also been developed for projected and proximal gradient methods (Bok and Altschuler, 2025). Finally, for linearly separable logistic regression, a horizon-dependent constant stepsize of order � achieves a final loss of $\widetilde O ( T ^ { - 2 } )$ (Wu et al., 2024); since the logistic loss has no finite minimizer, this setting lies outside the class $\mathcal { F } _ { 0 , L } ( \mathbb { R } ^ { d } )$ considered herein.

Anytime stepsize-based acceleration. Kornowski and Shamir (2024) initiated the study of anytime stepsize schedules, asking whether a single predetermined infinite stepsize sequence can accelerate plain GD uniformly over all stopping times. An anytime schedule achieving the convergence rate ${ \cal O } ( T ^ { - 2 \log _ { 2 } \rho _ { \mathrm { s i l } } / ( 1 + \mathrm { { l o g } } _ { 2 } \rho _ { \mathrm { s i l } } ) } ) = { \cal O } ( T ^ { - 1 . \mathrm { { l i l } } 9 . . . } )$ was recently constructed by Zhang et al. (2025), while Tsai et al. (2026) proved that no fixed positive infinite stepsize schedule can attain a worst-case last-iterate convergence rate of $o ( T ^ { - 4 / 3 } )$ . In contrast, our work studies prescribed-horizon schedules rather than anytime schedules.

Other related models. Our lower bound concerns the unmodified last iterate. For constant stepsizes with $L \eta \in ( 0 , 1 ]$ and for the dynamically increasing Teboulle–Vaisbourd schedule, worst-case analyses show that convex averaging cannot improve either the objective value or the gradient norm. For the same constant-step range, the extrapolated output $x _ { 0 } + c \left( x _ { T } - x _ { 0 } \right)$ , for a suitable scalar $^ { c , }$ improves the worst-case objective-value bound by a lower-order term (Luner and Grimmer, 2025). Such output modifications fall outside the method class considered in this paper.

Our model also excludes adaptive, randomized, and signed stepsize rules. Gradient-feedback and AdaGrad-type methods use adaptive stepsizes (Malitsky and Mishchenko, 2020; Duchi et al., 2011), randomized stepsize schedules follow a diferent model than the deterministic predetermined schedules considered here (Altschuler and Parrilo, 2024), and existing evidence for negative stepsizes comes from convex–concave gradient descent–ascent rather than smooth convex minimization (Shugart and Altschuler, 2025).

## 4 Normalized setup and a geometric construction

We begin with a geometric construction that associates a finite-dimensional hard instance with any prescribed nonnegative stepsize schedule. The construction selects a subset of the steps whose normalized stepsizes exceed one, preserves their original order, and uses them to move the trajectory between orthogonal blocks. A Moreau envelope then realizes the resulting piecewise-constant gradient field as the gradient of a single convex 1-smooth function. Maximizing the last-iterate objective gap over all selections of long steps produces the geometric quantity that will be analyzed in Sections 5 and 6.

## 4.1 Normalized setting and stepsize decomposition

We first analyze the normalized setting, where the smoothness parameter and the initial distance are both equal to one, i.e., $L = R = 1$ . In Section 6.4, we will recover the general case $L , R > 0$ by rescaling the objective, the iterates, and the stepsize schedule.

Specifically, let us consider a convex 1-smooth objective function � and a normalized stepsize schedule $h =$ $( h _ { 0 } , \ldots , h _ { T - 1 } ) \in \mathbb { R } _ { > 0 } ^ { T }$ . The corresponding GD updates are

$$
x _ { t + 1 } = x _ { t } - h _ { t } \nabla F ( x _ { t } ) , \qquad 0 \leq t < T .\tag{3}
$$

The portion of each stepsize $h _ { t }$ that exceeds the unit stepsize plays a key role. To isolate this efect, we decompose each $h _ { t }$ into a capped component min $\{ h _ { t } , 1 \}$ and an excess component $( h _ { t } - 1 ) .$ <sub>+</sub>, where $( u ) _ { + } : = \operatorname* { m a x } \{ u , 0 \}$ . Accordingly, we define

$$
y _ { t } : = ( h _ { t } - 1 ) _ { + } , \qquad B : = 1 + \sum _ { t = 0 } ^ { T - 1 } \operatorname* { m i n } \{ h _ { t } , 1 \} .\tag{4}
$$

We also define the index set of long steps (i.e., those stepsizes larger than 1) and its cardinality as follows:

$$
\mathcal { I } _ { + } : = \{ t \in \{ 0 , \ldots , T - 1 \} : y _ { t } > 0 \} , \qquad r : = | \mathcal { I } _ { + } | .\tag{5}
$$

Thus, $r$ is the number of long steps, and clearly $r \leq T$ . Moreover, since each capped component of the stepsize is at most 1, it follows immediately that

$$
B \leq T + 1 .
$$

## 4.2 Block decomposition

Fix any integer $m \in \{ 0 , \ldots , r \}$ . When $m \geq 1$ , we choose indices $0 \leq t _ { 1 } < \cdots < t _ { m } < T$ from $\mathcal { I } _ { + }$ (cf. (5)). These indices identify � long steps and partition the remaining iterations into the following gaps:

$$
\begin{array} { r l } & { G _ { 0 } : = \{ 0 , \dotsc , t _ { 1 } - 1 \} , } \\ & { G _ { i } : = \{ t _ { i } + 1 , \dotsc , t _ { i + 1 } - 1 \} , \qquad 1 \leq i < m , } \\ & { G _ { m } : = \{ t _ { m } + 1 , \dotsc , T - 1 \} . } \end{array}
$$

Note that any of these gaps may be empty. In the degenerate case $m = 0$ , we simply set $G _ { 0 } : = \{ 0 , \ldots , T - 1 \}$

When $m > 0 .$ , for each nonterminal block $0 \leq i < m$ , we define the gap mass $U _ { i }$ and the associated block scale $H _ { i }$ by

$$
U _ { i } : = 1 + \sum _ { t \in G _ { i } } h _ { t } , \qquad H _ { i } : = U _ { i } + y _ { t _ { i + 1 } } , \qquad 0 \leq i < m ;\tag{6a}
$$

regarding the terminal gap, we define

$$
H _ { m } : = 1 + 2 \sum _ { t \in G _ { m } } h _ { t } .\tag{6b}
$$

In contrast, in the degenerate case $m = 0$ , we do not need definitions of $U _ { i }$ but can still define

$$
H _ { 0 } = 1 + 2 \sum _ { t = 0 } ^ { T - 1 } h _ { t } , \qquad { \mathrm { i f ~ } } m = 0 .\tag{6c}
$$

To help control the transition between consecutive blocks, we define, for each $0 \leq i < m$

$$
\chi _ { i } : = \frac { y _ { t _ { i + 1 } } H _ { i + 1 } } { U _ { i } ( H _ { i } + H _ { i + 1 } ) } .\tag{7}
$$

As will be shown later via a projection comparison argument, $\chi _ { i }$ is precisely the largest admissible value of the squared amplitude ratio between blocks � and $i + 1$ . Note that in the special case $m = 0$ , we do not need quantities $U _ { i } \ \mathrm { o r } \ \chi _ { i }$ . We are now ready to state the main result regarding realization of a schedule.

Theorem 4.1. Fix a nonnegative normalized stepsize schedule $h _ { 0 } , \ldots , h _ { T - 1 }$ . Choose any integer � $\in \{ 0 , \ldots , r \}$ and, when $m \geq 1$ , any indices $0 \leq t _ { 1 } < \cdots < t _ { m } < T$ from $\mathcal { T } _ { + }$ . Let the corresponding quantities $G _ { i }$ and $H _ { i } ,$ , as well as $U _ { i }$ and $\chi _ { i }$ when $m \geq 1$ , be defined as above. Let $( \gamma _ { i } ) _ { i = 0 } ^ { m - 1 }$ be any collection ofpositive amplitudes (interpreted as the empty collection when $m = 0 )$ satisfying

$$
0 < \gamma _ { i } ^ { 2 } \leq \chi _ { i } , \qquad 0 \leq i < m .
$$

Then there exist a convex 1-smooth function $F : \mathbb { R } ^ { m + 1 }  \mathbb { R }$ and an initial point $\boldsymbol { x } _ { 0 } \in \mathbb { R } ^ { m + 1 }$ such that $0 \in$ argmin $F ,$ $\left\| x _ { 0 } \right\| = 1$ , and gradient descent with the schedule $h = \left( h _ { 0 } , \ldots , h _ { T - 1 } \right)$ satisfies

$$
F ( x _ { T } ) - F ( 0 ) = \frac { 1 } { 2 H _ { m } } \prod _ { i = 0 } ^ { m - 1 } \gamma _ { i } ^ { 2 } .\tag{8}
$$

In other words, every selection of long steps and admissible amplitude factors can be realized by a convex 1-smooth instance, with the final objective value determined by the terminal block scale $H _ { m }$ and the product of the squared amplitude factors $( \gamma _ { i } ) _ { i = 0 } ^ { m - 1 }$

## 4.3 Proof of Theorem 4.1

The construction is based on the classical notions of support functions and Moreau envelopes. To be precise, for a compact convex set $K \subseteq \mathbb { R } ^ { d }$ obeying $0 \in K$ , we define its support function by

$$
\sigma _ { K } ( z ) : = \operatorname* { m a x } _ { g \in K } \langle g , z \rangle ,\tag{9}
$$

as well as its associated Moreau envelope

$$
\mathcal { E } _ { K } ( x ) : = \operatorname* { m i n } _ { z \in \mathbb { R } ^ { d } } \left\{ \sigma _ { K } ( z ) + \frac { 1 } { 2 } \| x - z \| ^ { 2 } \right\} .\tag{10}
$$

We also define the corresponding proximal operator by

$$
{ \mathrm { p r o x } } _ { \sigma _ { K } } ( x ) : = { \underset { z \in \mathbb { R } ^ { d } } { \operatorname { a r g m i n } } } \left\{ \sigma _ { K } ( z ) + { \frac { 1 } { 2 } } \| x - z \| ^ { 2 } \right\} .\tag{11}
$$

Note that the minimizer in (11) is unique because the objective is strongly convex in �.

We next record the standard relationship between this proximal operator, Euclidean projection onto $K ,$ and the gradient of the Moreau envelope. For completeness, we include the proof of this classical result in Appendix A.

Lemma 4.2. Let $K \subseteq \mathbb { R } ^ { d }$ be compact and convex with $0 \in K$ , and set $F : = { \mathcal { E } } _ { K }$ as defined in (10). Write $\Pi _ { K } ( x )$ for the Euclidean projection of� onto �. Then we have

$$
\mathrm { p r o x } _ { \sigma _ { K } } ( x ) = x - \Pi _ { K } ( x ) ,\tag{12}
$$

and

$$
\nabla F ( x ) = \Pi _ { K } ( x ) , \qquad F ( x ) = \frac 1 2 \| x \| ^ { 2 } - \frac 1 2 \operatorname { d i s t } ( x , K ) ^ { 2 } .\tag{13}
$$

Consequently, � is convex and its gradient is 1-Lipschitz.

Motivated by Lemma 4.2, which reveals that the gradient of $\mathcal { E } _ { K }$ is simply Euclidean projection onto $K ,$ we shall choose � as the convex hull of the origin and the prescribed block gradients, and then verify that, at each iterate visited by GD, the appropriate vertex of � is the Euclidean projection. This allows a single globally defined convex 1-smooth function to generate the desired piecewise-constant gradient pattern along the GD trajectory.

Upon fixing a block decomposition and admissible amplitude factors as in Theorem 4.1, we choose an orthonormal basis $e _ { 0 } , \ldots , e _ { m } \mathrm { o f } \mathbb { R } ^ { m + 1 }$ , and define

$$
\lambda _ { 0 } : = 1 , \quad \lambda _ { i + 1 } : = \gamma _ { i } \lambda _ { i } , \quad 0 \leq i < m , \quad \quad X _ { i } : = \lambda _ { i } e _ { i } , \quad 0 \leq i \leq m .
$$

Next, we define the block gradients by

$$
g _ { i } : = \frac { \lambda _ { i } } { H _ { i } } ( e _ { i } - \gamma _ { i } e _ { i + 1 } ) \quad ( 0 \leq i < m ) , \qquad g _ { m } : = \frac { \lambda _ { m } } { H _ { m } } e _ { m } .\tag{14}
$$

We write

$$
s _ { \mathrm { t a i l } } : = \sum _ { t \in G _ { m } } h _ { t } , \qquad \Longrightarrow \qquad H _ { m } = 1 + 2 s _ { \mathrm { t a i l } } ,
$$

and set

$$
\begin{array} { l } { { K : = \displaystyle \mathrm { c o n v } \left\{ 0 , g _ { 0 } , \ldots , g _ { m } \right\} , } } \\ { { \displaystyle F ( x ) : = { \mathcal E } _ { K } ( x ) = \operatorname* { m i n } _ { z \in { \mathbb R } ^ { m + 1 } } \left\{ \sigma _ { K } ( z ) + \frac 1 2 \| x - z \| ^ { 2 } \right\} , } } \\ { { \displaystyle x _ { 0 } : = X _ { 0 } = e _ { 0 } } . } \end{array}\tag{15}
$$

In view of Lemma 4.2, � is convex and 1-smooth. Moreover, we have $\left\| x _ { 0 } \right\| = 1$ and $0 \in$ argmin $F$ since $0 \in K$

The main ingredient of the proof is to show that, throughout block $i ,$ the Euclidean projection onto � is precisely the vertex $g _ { i } .$ The construction has a simple yet important local structure: $g _ { i } \in$ span $\{ e _ { i } , e _ { i + 1 } \}$ for $i < m$ , while $g _ { m } \in$ span $\{ e _ { m } \}$ . Consequently, on a nonterminal block $i ,$ only $g _ { i - 1 }$ (when $i > 0 ) , g _ { i }$ , and $g _ { i + 1 }$ can have a nonzero inner product with the shifted iterate. The predecessor contribution is nonpositive, so verifying the projection property reduces to comparing $g _ { i }$ with its successor $g _ { i + 1 }$ . The condition $\gamma _ { i } ^ { 2 } \leq \chi _ { i }$ is exactly the criterion ensuring that this comparison holds. Consequently, the gradient of the Moreau envelope remains equal to $g _ { i }$ throughout block �, and the GD iterates follow the prescribed block trajectory by construction.

Proof of Theorem 4.1. For any $x , g \in \mathbb { R } ^ { m + 1 }$ , the characterization of Euclidean projection states that

$$
g = \Pi _ { K } ( x ) \quad \iff \quad g \in K { \mathrm { ~ a n d ~ } } \langle x - g , \nu - g \rangle \leq 0 \quad { \mathrm { f o r ~ e v e r y ~ } } \nu \in K .
$$

Writing $z = x - g$ , we see that $g = \Pi _ { K } ( x )$ holds if and only $\operatorname { i f } g$ maximizes $\langle \cdot , z \rangle$ over �. Since � is the convex hull of $\{ 0 , g _ { 0 } , \ldots , g _ { m } \}$ , it is therefore suficient to verify this condition only for the generators $0 , g _ { 0 } , \ldots , g _ { m }$

Step 1: characterizing projection within each block. Our goal in this step is to verify that, throughout each block, the prescribed block gradient is the Euclidean projection onto �. By the discussion preceding the proof, this reduces to showing that the corresponding generator maximizes a suitable linear functional over �.

Specifically, for each nonterminal block $0 \leq i < m$ and each $s \in [ 0 , U _ { i } - 1 ]$ , set $u : = s + 1 \in [ 1 , U _ { i } ]$ , and define

$$
\begin{array} { r l r } {  { q _ { i } ( s ) : = X _ { i } - s g _ { i } , } } \\ & { } & { \quad \quad \quad z _ { i , u } : = q _ { i } ( s ) - g _ { i } = \cfrac { \lambda _ { i } } { H _ { i } } \{ ( H _ { i } - u ) e _ { i } + u \gamma _ { i } e _ { i + 1 } \} . } \end{array}\tag{16}
$$

The cumulative stepsize before each update in block � is a value of � in $[ 0 , U _ { i } - 1 ]$ , since $\begin{array} { r } { \sum _ { t \in G _ { i } } h _ { t } = U _ { i } - 1 } \end{array}$

For the terminal block, we similarly define, for every $s \in [ 0 , s _ { \mathrm { t a i l } } ]$

$$
q _ { m } ( s ) : = X _ { m } - s g _ { m } , \qquad z _ { m } ( s ) : = q _ { m } ( s ) - g _ { m } = \frac { \lambda _ { m } ( H _ { m } - s - 1 ) } { H _ { m } } e _ { m } .
$$

Note that every $z _ { m } ( s )$ is a nonnegative multiple of $e _ { m } .$ , since

$$
H _ { m } - s - 1 = 2 s _ { \mathrm { t a i l } } - s \geq s _ { \mathrm { t a i l } } \geq 0 .
$$

We first analyze a nonterminal block. By (14) and $( 1 6 ) , z _ { i , u }$ ∈ span $\big \{ e _ { i } , e _ { i + 1 } \big \}$ , while $g _ { j } \in \operatorname { s p a n } \{ e _ { j } , e _ { j + 1 } \}$ for $j < m$ and $g _ { m } \in \operatorname { s p a n } \{ e _ { m } \}$ . Since the basis vectors are orthonormal, we have

$$
\langle g _ { j } , z _ { i , u } \rangle = 0 \qquad \big ( j \notin \{ i - 1 , i , i + 1 \} , 0 \le j \le m \big ) .
$$

Thus, only the predecessor $g _ { i - 1 } \ ( \mathrm { w h e n } i > 0 ) , g _ { i }$ , and the successor $g _ { i + 1 }$ can have nonzero inner product with $z _ { i , u }$ . For $i > 0 ,$ , the predecessor satisfies

$$
\langle g _ { i - 1 } , z _ { i , u } \rangle = - \frac { \lambda _ { i } ^ { 2 } ( H _ { i } - u ) } { H _ { i - 1 } H _ { i } } \leq 0 ;
$$

and when $i = 0 .$ , there is no predecessor. Consequently, we only need to compare $g _ { i }$ with its successor and with the zero inner products. The two remaining inner products are

$$
\begin{array} { l } { { \displaystyle \langle g _ { i } , z _ { i , u } \rangle = \frac { \lambda _ { i } ^ { 2 } } { H _ { i } ^ { 2 } } \left\{ H _ { i } - u ( 1 + \gamma _ { i } ^ { 2 } ) \right\} , } } \\ { { \displaystyle \langle g _ { i + 1 } , z _ { i , u } \rangle = \frac { \lambda _ { i } ^ { 2 } \gamma _ { i } ^ { 2 } u } { H _ { i } H _ { i + 1 } } . } } \end{array}
$$

The second formula remains valid for $i = m - 1$ , where $g _ { i + 1 } = g _ { m } .$ . The first quantity decreases with �, whereas the second increases with �, so it sufices to compare them at $u = U _ { i }$ . At that endpoint, we have

$$
\begin{array} { c c c } { { \displaystyle { \langle g _ { i } , z _ { i , U _ { i } } \rangle = \frac { \lambda _ { i } ^ { 2 } \{ y _ { t _ { i + 1 } } - U _ { i } \gamma _ { i } ^ { 2 } \} } { H _ { i } ^ { 2 } } , } } }  \\ { { \ } } \\ { { \langle g _ { i + 1 } , z _ { i , U _ { i } } \rangle = \displaystyle { \frac { \lambda _ { i } ^ { 2 } \gamma _ { i } ^ { 2 } U _ { i } } { H _ { i } H _ { i + 1 } } } . } } \end{array}
$$

The successor value is positive because $\lambda _ { i } , \gamma _ { i } , U _ { i } , H _ { i } , H _ { i + 1 } > 0$ . Moreover,

$$
\left. g _ { i } , z _ { i } , U _ { i } \right. \geq \left. g _ { i + 1 } , z _ { i } , U _ { i } \right. \quad \Longleftrightarrow \quad \frac { y _ { t _ { i + 1 } } - U _ { i } \gamma _ { i } ^ { 2 } } { H _ { i } ^ { 2 } } \geq \frac { \gamma _ { i } ^ { 2 } U _ { i } } { H _ { i } H _ { i + 1 } } \quad \Longleftrightarrow \quad \gamma _ { i } ^ { 2 } \leq \frac { y _ { t _ { i + 1 } } H _ { i + 1 } } { U _ { i } \left( H _ { i } + H _ { i + 1 } \right) } = \chi _ { i } ,
$$

which holds by assumption. Therefore, for every $1 \leq u \leq U _ { i }$ , one has

$$
\begin{array} { r } { \langle g _ { i } , z _ { i , u } \rangle \geq \langle g _ { i } , z _ { i , U _ { i } } \rangle \geq \langle g _ { i + 1 } , z _ { i , U _ { i } } \rangle \geq \langle g _ { i + 1 } , z _ { i , u } \rangle . } \end{array}
$$

In particular, the inner product with $g _ { i }$ is positive and, therefore, dominates the generators with zero or negative inner product. Consequently, $g _ { i }$ maximizes $\langle g , z _ { i , u } \rangle$ over $g \in K$

The terminal block is even simpler. Observe that $\langle g _ { m } , z _ { m } ( s ) \rangle \geq 0$ . When $m > 0 , \langle g _ { m - 1 } , z _ { m } ( s ) \rangle \leq 0 \quad$ , and all other generators are orthogonal to $z _ { m } ( s )$ . Hence $g _ { m }$ maximizes the support functional at $z _ { m } ( s )$ . If $s _ { \mathrm { t a i l } } = 0$ , then $z _ { m } ( s ) = 0 .$ so every generator attains the same value, and in particular $g _ { m }$ is still a maximizer.

We have therefore established the projection property throughout every block. Indeed, for every $\nu \in K$

$$
\begin{array} { r } { \left. z _ { i , u } , \nu - g _ { i } \right. \leq 0 , \qquad \left. z _ { m } ( s ) , \nu - g _ { m } \right. \leq 0 . } \end{array}
$$

Since $q _ { i } ( s ) = z _ { i , u } + g _ { i }$ and $q _ { m } ( s ) = z _ { m } ( s ) + g _ { m }$ , these are exactly the variational inequalities characterizing Euclidean projection. It thus follows that

$$
\Pi _ { K } ( q _ { i } ( s ) ) = g _ { i } , \qquad \Pi _ { K } ( q _ { m } ( s ) ) = g _ { m } .
$$

Applying Lemma 4.2 then yields

$$
\begin{array} { r l r l r l } & { \nabla F ( q _ { i } ( s ) ) = g _ { i } , } & { \mathrm { p r o x } _ { \sigma _ { K } } ( q _ { i } ( s ) ) = z _ { i , u } , } & { 0 \le i < m , } & { 0 \le s \le U _ { i } - 1 , } \\ & { \nabla F ( q _ { m } ( s ) ) = g _ { m } , } & { \mathrm { p r o x } _ { \sigma _ { K } } ( q _ { m } ( s ) ) = z _ { m } ( s ) , } & { 0 \le s \le s _ { \mathrm { t a i l } } . } \end{array}\tag{17}
$$

Step 2: characterizing the GD trajectory. Set

$$
b _ { 0 } : = 0 , \qquad b _ { i } : = t _ { i } + 1 \quad ( 1 \leq i \leq m ) ,
$$

so that $b _ { i }$ is the first time index of block �. For $0 \leq i < m$ and $t \in G _ { i } \cup \{ t _ { i + 1 } \}$ , define

$$
s _ { i , t } : = \sum _ { \tau = b _ { i } } ^ { t - 1 } h _ { \tau } .
$$

Having established the gradient identities in Step 1, we now verify that the GD iterates follow the prescribed trajectory. Specifically, we claim that throughout block �,

$$
x _ { t } = q _ { i } ( s _ { i , t } ) = X _ { i } - s _ { i , t } g _ { i } .\tag{18}
$$

We prove the claim by induction over the blocks and, within each block, by induction over the iterations. For the first block, one has $x _ { b _ { 0 } } = x _ { 0 } = X _ { 0 } = q _ { 0 } ( 0 )$ . We now fix $i < m$ and suppose $x _ { b _ { i } } = X _ { i }$ . For any index $t \in G _ { i } \cup \{ t _ { i + 1 } \}$ , we have $0 \leq s _ { i , t } \leq U _ { i } - 1$ . Hence, whenever (18) holds, (17) gives $\nabla F ( x _ { t } ) = g _ { i }$ . If the next iterate remains in the same block, then it holds that

$$
x _ { t + 1 } = x _ { t } - h _ { t } \nabla F ( x _ { t } ) = X _ { i } - ( s _ { i , t } + h _ { t } ) g _ { i } = X _ { i } - s _ { i , t + 1 } g _ { i } ,
$$

which proves the within-block induction. At the selected transition index $t = t _ { i + 1 }$ ，

$$
s _ { i , t _ { i + 1 } } = \sum _ { t \in G _ { i } } h _ { t } = U _ { i } - 1 ,
$$

and the cumulative stepsize after the transition update is

$$
s _ { i , t _ { i + 1 } } + h _ { t _ { i + 1 } } = ( U _ { i } - 1 ) + ( 1 + y _ { t _ { i + 1 } } ) = H _ { i } .
$$

Thus the long step moves the iterate exactly to the starting point of the next block:

$$
x _ { b _ { i + 1 } } = x _ { t _ { i + 1 } + 1 } = X _ { i } - H _ { i } g _ { i } = \lambda _ { i } \gamma _ { i } e _ { i + 1 } = X _ { i + 1 } .
$$

This establishes the induction hypothesis for the next block and hence verifies the entire nonterminal trajectory.

For the terminal block, we define

$$
s _ { m , t } : = \sum _ { \tau = b _ { m } } ^ { t - 1 } h _ { \tau } , \qquad b _ { m } \leq t \leq T .
$$

We have $x _ { b _ { m } } = X _ { m } ;$ : this is the initial condition when $m = 0$ , and follows from the preceding block induction when $m > 0 . { \mathrm { \ A p p l y i n g \ : ( 1 7 ) } }$ and the same within-block induction gives

$$
x _ { t } = q _ { m } ( s _ { m , t } ) = X _ { m } - s _ { m , t } g _ { m } \quad ( b _ { m } \leq t \leq T ) .
$$

In particular,

$$
x _ { T } = q _ { m } ( s _ { \mathrm { t a i l } } ) = X _ { m } - s _ { \mathrm { t a i l } } g _ { m } .\tag{19}
$$

This completes the characterization of the GD trajectory. Thus every update, including each zero-stepsize update, is a GD step for the same objective function �; the construction also covers empty gaps.

Step 3: calculating the objective value at the last iterate. It remains to evaluate the objective value at the last iterate. By virtue of property (19) and the terminal part of (17),

$$
\Pi _ { K } ( x _ { T } ) = g _ { m } , \qquad z _ { T } : = \operatorname { p r o x } _ { \sigma _ { K } } ( x _ { T } ) = x _ { T } - g _ { m } = z _ { m } ( s _ { \mathrm { t a i l } } ) = \frac { \lambda _ { m } s _ { \mathrm { t a i l } } } { H _ { m } } e _ { m } .
$$

Thus dist $( x _ { T } , K ) = \| z _ { T } \|$ . Using (13) and $F ( 0 ) = 0$ , we obtain

$$
F ( x _ { T } ) - F ( 0 ) = \frac 1 2 \| z _ { T } + g _ { m } \| ^ { 2 } - \frac 1 2 \| z _ { T } \| ^ { 2 } = \langle g _ { m } , z _ { T } \rangle + \frac 1 2 \| g _ { m } \| ^ { 2 } = \frac { \lambda _ { m } ^ { 2 } } { H _ { m } ^ { 2 } } \left( s _ { \mathrm { t a i l } } + \frac 1 2 \right) = \frac { \lambda _ { m } ^ { 2 } } { 2 H _ { m } } ,
$$

where the last equality uses $H _ { m } = 1 + 2 s _ { \mathrm { t a i l } }$ . Finally, since $\begin{array} { r } { \lambda _ { m } ^ { 2 } = \prod _ { i = 0 } ^ { m - 1 } \gamma _ { i } ^ { 2 } } \end{array}$ , we readily see that

$$
F ( x _ { T } ) - F ( 0 ) = \frac { 1 } { 2 H _ { m } } \prod _ { i = 0 } ^ { m - 1 } \gamma _ { i } ^ { 2 } ,
$$

which is precisely (8). The same construction covers $m = 0$ and uses exactly $m + 1 \leq T + 1$ dimensions.

## 4.4 The resulting key functional

Theorem 4.1 shows that, for a fixed block decomposition, the final objective value is maximized by taking each amplitude factor as large as allowed, namely, $\gamma _ { i } ^ { 2 } = \chi _ { i }$ . Optimizing further over all admissible choices of transition indices therefore leads to the functional

$$
C _ { T } ( h ) : = \operatorname* { m a x } _ { 0 \leq m \leq r } \ \operatorname* { m a x } _ { \substack { 0 \leq t _ { 1 } < \cdots < t _ { m } < T } } \frac { 1 } { H _ { m } } \prod _ { i = 0 } ^ { m - 1 } \chi _ { i } .\tag{20}
$$

Here, the terminal block scale $H _ { m }$ and the local bounds $\chi _ { i }$ are determined by the selected indices through (6b) and $( 7 ) ,$ respectively. The purpose of $C _ { T } ( h )$ is to separate the geometric realization from the remaining schedule analysis. For a fixed schedule ℎ, the quantity $C _ { T } ( h ) / 2$ is the largest terminal objective gap attained within the family of instances constructed in Theorem 4.1. The following corollary makes this attainment precise and thereby reduces the remaining proof to lower-bounding $C _ { T } ( h )$ .

Corollary 4.3. For every nonnegative normalized stepsize schedule ℎ oflength $T ,$ there exist an integer $1 \leq d \leq T + 1$ a function $F \in \mathcal { F } _ { 0 , 1 } ( \mathbb { R } ^ { d } )$ with $0 \in$ argmin �, and an initial point $\boldsymbol { x } _ { 0 } \in \mathbb { R } ^ { d }$ with $\| x _ { 0 } \| = 1$ such that GD with schedule ℎ satisfies

$$
2 \{ F ( x _ { T } ) - F ( 0 ) \} = C _ { T } ( h ) .\tag{21}
$$

Proof. Since the maximization in (20) is over finitely many choices of transition indices, it is attained. Choose indices $t _ { 1 } < \cdots < t _ { m }$ and, for each nonterminal block, set $\gamma _ { i } : = \sqrt { \chi _ { i } }$ . Each $\chi _ { i }$ is positive by (7), and hence these factors

satisfy the assumptions of Theorem 4.1. The theorem therefore yields a convex 1-smooth instance in dimension $d = m + 1 \leq T + 1$ satisfying

$$
2 \{ F ( x _ { T } ) - F ( 0 ) \} = \frac { 1 } { H _ { m } } \prod _ { i = 0 } ^ { m - 1 } \chi _ { i } = C _ { T } ( h ) .
$$

If the maximum is attained at $m = 0$ , the same theorem applies with the empty product interpreted as one. Thus the claimed result holds in all cases. □

The empty selection $m = 0$ is always admissible and contributes

$$
\left( 1 + 2 \sum _ { t = 0 } ^ { T - 1 } h _ { t } \right) ^ { - 1 } .
$$

When all steps are short, this already gives an objective gap of order $T ^ { - 1 }$ . Large excesses can make this baseline much smaller, however, so the remaining analysis must exploit nonempty selections.

The main dificulty is that each local factor $\chi _ { i }$ couples two neighboring blocks, so the value of a selected chain depends on the temporal ordering of the long steps. Section 5 removes this dependence through two matchings, while Section 6 chooses an appropriate rank cutof and controls the residual schedule mass.

## 5 Order-independent bounds via matchings

The functional $C _ { T } ( h )$ in (20) depends on both the magnitudes and the temporal order of the selected long steps. To separate the magnitudes of the long-step excesses from their temporal arrangement, we first rank the positive excesses by size. $\operatorname { I f } r \geq 1$ , break ties arbitrarily and label the distinct indices in $\mathcal { I } _ { + }$ as $\tau _ { 1 } , \ldots , \tau _ { r }$ so that, with $a _ { s } : = y _ { \tau _ { s } }$

$$
a _ { 1 } \geq a _ { 2 } \geq \cdot \cdot \cdot \geq a _ { r } > 0 .
$$

When $r = 0 .$ , the ranked list is empty. For $0 \leq q \leq r ,$ define

$$
D _ { q } : = B + \sum _ { s = q + 1 } ^ { r } a _ { s } .\tag{22}
$$

We call $D _ { q }$ the residual schedule mass at rank $q .$ It consists of the capped base mass � together with all positive excesses except the � largest ones. In particular,

$$
D _ { r } = B , \qquad D _ { 0 } = 1 + \sum _ { t = 0 } ^ { T - 1 } h _ { t } ,
$$

because $h _ { t } = \operatorname* { m i n } \{ h _ { t } , 1 \} + y _ { t }$

The cutof � separates the � excesses used to construct a candidate chain from the remaining schedule mass. Once the selected excesses are restored to chronological order, $D _ { q }$ decomposes exactly into the intervening gap masses and the terminal mass, as shown in (24). It therefore serves as the common mass budget in the matching argument below and, later, as the state variable in the cutof analysis of Section 6

For $2 \leq q \leq r ,$ we restore the largest excesses to chronological order and associate the reciprocals of the corresponding local factors with the edges of a path, including a final edge for the terminal factor. The odd and even edges form two matchings. Bounding their products by the corresponding optimal matching values removes the temporal order, and a total reciprocal mass estimate then gives an order-independent lower bound on $C _ { T } ( h )$ . Corollary 4.3 connects this functional bound back to a valid normalized GD instance.

## 5.1 The top-ranked excesses in temporal order

Fix $1 \leq q \leq r ,$ , and consider the time labels $\tau _ { 1 } , \ldots , \tau _ { q }$ of the � largest excesses. We define � as the unique permutation of $\{ 1 , \ldots , q \}$ for which

$$
\tau _ { \pi ( 1 ) } < \cdot \cdot \cdot < \tau _ { \pi ( q ) } ,
$$

which is well defined because the time labels are distinct. We then set

$$
c _ { i } : = a _ { \pi ( i ) } = y _ { \tau _ { \pi ( i ) } } \qquad \mathrm { ~ f o r ~ } 1 \leq i \leq q ;\tag{23}
$$

these are the same excesses, now indexed by time rather than by rank. We use the gap quantities associated with

$$
t _ { i } : = \tau _ { \pi ( i ) } , \qquad 1 \leq i \leq q ,
$$

and denote the schedule mass following its last selected time by

$$
V : = 1 + \sum _ { t = \tau _ { \pi ( q ) } + 1 } ^ { T - 1 } h _ { t } .
$$

The quantity $D _ { q }$ (cf. (22)) then decomposes as

$$
\sum _ { i = 0 } ^ { q - 1 } U _ { i } + V = D _ { q } , \qquad H _ { q } = 2 V - 1 .\tag{24}
$$

The first identity holds because each selected long step contributes its unit part to �, each unselected excess remains in $D _ { q }$ , and every other schedule term belongs either to one of the gaps or to the terminal portion �. Since $V \geq 1 , ( 2 4 )$ also immediately gives

$$
\sum _ { i = 0 } ^ { q - 1 } U _ { i } \le D _ { q } - 1 , \qquad U _ { q - 1 } + H _ { q } \le 2 D _ { q } - 1 ,\tag{25}
$$

where the second inequality follows because $\begin{array} { r } { U _ { q - 1 } + H _ { q } \le \sum _ { i } U _ { i } + 2 V - 1 = D _ { q } + V - 1 \le 2 D _ { q } - 1 } \end{array}$ . For these selected indices, $H _ { i } = U _ { i } + c _ { i + 1 }$ for $0 \leq i < q$ , and the local factors are those in (7). We expand their reciprocals as

$$
\begin{array} { c c } { { \chi _ { i - 1 } ^ { - 1 } = \displaystyle \frac { U _ { i - 1 } } { c _ { i } } + \displaystyle \frac { U _ { i - 1 } } { H _ { i } } + \displaystyle \frac { U _ { i - 1 } ^ { 2 } } { c _ { i } H _ { i } } , } } & { { 1 \leq i < q , } } \\ { { H _ { q } \chi _ { q - 1 } ^ { - 1 } = U _ { q - 1 } \left( 1 + \displaystyle \frac { U _ { q - 1 } + H _ { q } } { c _ { q } } \right) . } } & { { } } \end{array}\tag{26}
$$

## 5.2 A path bound and its two matchings

Each nonterminal factor $\chi _ { i - 1 } ^ { - 1 } , 1 \leq i < q$ , couples two consecutive excesses $c _ { i } , c _ { i + 1 }$ in temporal order. The terminal factor involves only $c _ { q } ;$ after normalization, we represent it by adjoining one auxiliary terminal vertex. This produces a path on $q + 1$ vertices. Alternating edges along the path share no endpoints, which is why two matchings, one for each parity, give a bound that no longer depends on the particular temporal ordering of the selected excesses.

For $u , \nu > 0$ , define

$$
\psi ( u , \nu ) = \frac { u + \nu + u \nu } { 2 } .\tag{27}
$$

For a finite label set $\mathcal { V } ,$ , let $W = ( w _ { \alpha } ) _ { \alpha \in \mathcal { V } }$ be a family of positive vertex weights. For an integer $1 \leq k \leq \lfloor | \mathcal { V } | / 2 \rfloor$ define $P _ { \psi } ( W , k )$ as the maximum product of � over all �-edge matchings on the labeled elements of �:

$$
P _ { \psi } ( W , k ) : = \operatorname* { m a x } _ { \substack { \alpha _ { 1 } , \beta _ { 1 } , \ldots , \alpha _ { k } , \beta _ { k } \in \mathcal { V } } } \prod _ { j = 1 } ^ { k } \psi ( w _ { \alpha _ { j } } , w _ { \beta _ { j } } ) , \qquad P _ { \psi } ( W , 0 ) : = 1 .
$$

For a rank cutof $2 \leq q \leq r$ , we define

$$
w _ { s } ^ { ( q ) } = \frac { 2 D _ { q } } { q a _ { s } } \quad ( 1 \leq s \leq q ) , \qquad w _ { \dagger } ^ { ( q ) } = \frac { 1 } { q - 1 } , \qquad W _ { q } = \biggl \{ w _ { 1 } ^ { ( q ) } , \ldots , w _ { q } ^ { ( q ) } , w _ { \dagger } ^ { ( q ) } \biggr \} .\tag{28}
$$

The first � weights are normalized reciprocals of the selected excesses, while $w _ { \cdot \cdot } ^ { ( q ) }$ is the auxiliary terminal weight. The multiset $W _ { q }$ is labeled, so equal numerical weights remain distinct. We then set

$$
k _ { + } = \left\lceil \frac { q } { 2 } \right\rceil , \qquad k _ { - } = \left\lfloor \frac { q } { 2 } \right\rfloor , \qquad M _ { q } = P _ { \psi } ( W _ { q } , k _ { + } ) P _ { \psi } ( W _ { q } , k _ { - } ) .\tag{29}
$$

In temporal order, these vertex weights are

$$
\nu _ { i } : = w _ { \pi ( i ) } ^ { ( q ) } = \frac { 2 D _ { q } } { q c _ { i } } \quad ( 1 \leq i \leq q ) , \qquad \nu _ { q + 1 } : = w _ { \dagger } ^ { ( q ) } .
$$

Thus, the path corresponding to the chosen excesses is

$$
\underbrace { \nu _ { 1 } \to \dots \to \nu _ { q } } _ { \mathrm { n o r m a l i z e d ~ r e c i p r o c a l s ~ o f ~ s e l e c t e c t e d ~ e x c e s s e s } } \longrightarrow \underbrace { \nu _ { q + 1 } = w _ { \dagger } ^ { ( q ) } } _ { \mathrm { a u x i l i a r y ~ t e r m i n a l ~ v e r t e x } } .
$$

We now have the following order-independent lower bound.

Proposition 5.1. For every $2 \leq q \leq r ,$ one has

$$
C _ { T } ( h ) \geq \frac { q } { 2 D _ { q } ( q - 1 ) \mathscr { M } _ { q } } .\tag{30}
$$

Importantly, this bound is independent of the temporal order of the top � excesses.

## 5.3 Proof of Proposition 5.1

We first record the following product inequality, whose proof is deferred to Appendix B.1.

Lemma 5.2. Let $q \geq 1$ and $D > 0 ,$ , and put $\bar { u } : = D / q$ . If $\mathbf { \Phi } _ { u _ { 1 } , . . . , u _ { q } } > 0$ satisfy $\begin{array} { r } { \sum _ { i = 1 } ^ { q } u _ { i } \le D , } \end{array}$ and $A _ { i } , B _ { i } > 0 f o r$ $1 \leq i < q ,$ , then

$$
\left( \prod _ { i = 1 } ^ { q } u _ { i } \right) \prod _ { i = 1 } ^ { q - 1 } ( A _ { i } + B _ { i } u _ { i } ) \leq \bar { u } ^ { q } \prod _ { i = 1 } ^ { q - 1 } ( A _ { i } + 2 \bar { u } B _ { i } ) .\tag{31}
$$

ProofofProposition 5.1. We write $\xi _ { i } : = c _ { i } ^ { - 1 }$ for the reciprocal of the �th excess in temporal order. The identities in (24) give

$$
\sum _ { i = 0 } ^ { q - 1 } U _ { i } \le D _ { q } - 1 < D _ { q } , \qquad U _ { q - 1 } + H _ { q } \le 2 D _ { q } - 1 < 2 D _ { q } .
$$

Moreover, $U _ { i } \geq 1$ for $0 \leq i < q$ and $H _ { q } \geq 1$ because the schedule is nonnegative. Since every $c _ { i }$ is positive, so are all the quantities $\chi _ { i }$ in (7).

For $1 \leq i < q .$ , the first identity in (26), together with $H _ { i } = U _ { i } + c _ { i + 1 } \geq c _ { i + 1 }$ , gives

$$
\chi _ { i - 1 } ^ { - 1 } \leq U _ { i - 1 } ( \xi _ { i } + \xi _ { i + 1 } ) + U _ { i - 1 } ^ { 2 } \xi _ { i } \xi _ { i + 1 } .\tag{32}
$$

For the terminal factor, the second identity in (26) and $U _ { q - 1 } + H _ { q } < 2 D _ { q }$ give

$$
H _ { q } \chi _ { q - 1 } ^ { - 1 } \leq U _ { q - 1 } \big ( 1 + 2 D _ { q } \xi _ { q } \big ) .\tag{33}
$$

For $1 \leq i < q ,$ , we set

$$
S _ { i } : = \xi _ { i } + \xi _ { i + 1 } , \qquad \Xi _ { i } : = \xi _ { i } \xi _ { i + 1 } .
$$

Although the individual gap masses are not controlled separately, their sum is less than $D _ { q }$ . We multiply (32) over $1 \leq i < q$ and then use (33) to obtain

$$
H _ { q } \prod _ { j = 0 } ^ { q - 1 } \chi _ { j } ^ { - 1 } \leq ( 1 + 2 D _ { q } \xi _ { q } ) \left( \prod _ { j = 0 } ^ { q - 1 } U _ { j } \right) \prod _ { i = 1 } ^ { q - 1 } ( S _ { i } + \Xi _ { i } U _ { i - 1 } ) .
$$

We can therefore apply Lemma 5.2 with $D = D _ { q } , u _ { i } = U _ { i - 1 } , A _ { i } = S _ { i }$ , and $B _ { i } = \Xi _ { i }$ , where its hypotheses follow from $\begin{array} { r } { U _ { i } \geq 1 , \sum _ { i = 0 } ^ { q - 1 } U _ { i } < D _ { q } } \end{array}$ , and $S _ { i } , \Xi _ { i } > 0$ . This yields

$$
\begin{array} { l } { { \displaystyle { \cal H } _ { q } \prod _ { i = 0 } ^ { q - 1 } \chi _ { i } ^ { - 1 } \le ( 1 + 2 D _ { q } \xi _ { q } ) \left( \frac { D _ { q } } { q } \right) ^ { q } \prod _ { i = 1 } ^ { q - 1 } \left( S _ { i } + \frac { 2 D _ { q } } { q } \Xi _ { i } \right) } } \\ { { \displaystyle ~ = 2 D _ { q } \left( \frac { D _ { q } } { q } \right) ^ { q } \left( \xi _ { q } + \frac { 1 } { 2 D _ { q } } \right) \prod _ { i = 1 } ^ { q - 1 } \left( \xi _ { i } + \xi _ { i + 1 } + \frac { 2 D _ { q } } { q } \xi _ { i } \xi _ { i + 1 } \right) . } } \end{array}\tag{34}
$$

We now use the normalization $\nu _ { i } = 2 D _ { q } \xi _ { i } / q$ to express every factor in terms of the same function �. More precisely,

$$
\frac { D _ { q } } { q } \left( S _ { i } + \frac { 2 D _ { q } } { q } \Xi _ { i } \right) = \psi ( \nu _ { i } , \nu _ { i + 1 } ) \quad ( 1 \leq i < q ) , \qquad 2 D _ { q } \frac { D _ { q } } { q } \left( \xi _ { q } + \frac { 1 } { 2 D _ { q } } \right) = \frac { 2 D _ { q } ( q - 1 ) } { q } \psi ( \nu _ { q } , \nu _ { q + 1 } ) .\tag{35}
$$

Both sides of the second identity equal $( D _ { q } / q ) ( 1 + 2 D _ { q } \xi _ { q } )$ , so adjoining the auxiliary vertex introduces no further inequality: it merely puts the terminal factor in the same two-variable form as the internal factors. The resulting path has labeled vertex multiset $W _ { q }$ , with the selected excesses in their actual temporal order. We denote its edge product by

$$
\mathcal { P } _ { \mathrm { p a t h } } : = \prod _ { i = 1 } ^ { q } \psi ( \nu _ { i } , \nu _ { i + 1 } ) .
$$

Taken together, (34) and (35) give

$$
H _ { q } \prod _ { i = 0 } ^ { q - 1 } { \chi _ { i } ^ { - 1 } } \leq \frac { 2 D _ { q } ( q - 1 ) } { q } \mathcal { P } _ { \mathrm { p a t h } } .\tag{36}
$$

We next remove the dependence on that order by splitting the path edges according to parity:

$$
E _ { + } : = \Big \{ \{ \nu _ { i } , \nu _ { i + 1 } \} : 1 \leq i \leq q , \ i \ o d { \bf { d } } \Big \} , \qquad E _ { - } : = \Big \{ \{ \nu _ { i } , \nu _ { i + 1 } \} : 1 \leq i \leq q , \ i \ e v e n \Big \} .
$$

No two edges of the same parity share a vertex. Hence $E _ { + }$ and $E _ { - }$ are matchings on $W _ { q }$ with $k _ { + }$ and $k _ { - }$ edges, respectively, and the definition of $P _ { \psi }$ yields

$$
\mathcal { P } _ { \mathrm { p a t h } } = \left( \prod _ { \left\{ u , \nu \right\} \in E _ { + } } \psi ( u , \nu ) \right) \left( \prod _ { \left\{ u , \nu \right\} \in E _ { - } } \psi ( u , \nu ) \right) \leq P _ { \psi } ( W _ { q } , k _ { + } ) P _ { \psi } ( W _ { q } , k _ { - } ) = M _ { q } .
$$

We combine this inequality with (36) and take positive reciprocals to obtain

$$
\frac { 1 } { H _ { q } } \prod _ { i = 0 } ^ { q - 1 } { \chi _ { i } } = \left( H _ { q } \prod _ { i = 0 } ^ { q - 1 } { \chi _ { i } ^ { - 1 } } \right) ^ { - 1 } \ge \frac { q } { 2 D _ { q } ( q - 1 ) \mathcal { P } _ { \mathrm { p a t h } } } \ge \frac { q } { 2 D _ { q } ( q - 1 ) M _ { q } } .
$$

The left-hand side is the value in (20) corresponding to the indices of the $q$ largest excesses. Since $C _ { T } ( h )$ is the maximum over all admissible choices of selected indices, (30) follows. □

## 5.4 Bounding the matchings by total weight

The quantity $\textstyle { \mathcal { M } } _ { q }$ no longer depends on the temporal order of the selected excesses, but it still depends on each of the first $q$ excesses separately. We now bound it using only their scaled reciprocal sum. For $1 \leq q \leq r$ , define

$$
\zeta _ { q } : = \frac { D _ { q } } { q ^ { 2 } } \sum _ { s = 1 } ^ { q } \frac { 1 } { a _ { s } } .\tag{37}
$$

Thus, $\zeta _ { q }$ is $D _ { q } / q$ times the mean reciprocal of the first � excesses. For $2 \leq q \leq r ,$ , we also write

$$
\mu _ { q } : = M _ { q } ^ { 1 / q } .\tag{38}
$$

Since the two matching products contain $k _ { + } + k _ { - } = q$ edges in total, $\mu _ { q }$ is their geometric mean per edge. The sum of the $q$ excess-dependent vertex weights is

$$
\sum _ { s = 1 } ^ { q } w _ { s } ^ { ( q ) } = 2 q \zeta _ { q } ,
$$

and the auxiliary terminal vertex adds $( q - 1 ) ^ { - 1 }$ . Thus it remains to bound a matching product in terms of the total weight of its vertices.

Lemma 5.3. Let � be a finite labeled multiset of positive weights, set $\Sigma : = \textstyle \sum _ { w \in W }$ �, and let $1 \leq k \leq \lfloor | W | / 2 \rfloor$ . Then

$$
P _ { \psi } ( W , k ) ^ { 1 / k } \leq \frac { \Sigma } { 2 k } + \frac { \Sigma ^ { 2 } } { 8 k ^ { 2 } } .\tag{39}
$$

Consequently, for every $2 \leq q \leq r , $

$$
\mu _ { q } \leq \frac { 2 q \zeta _ { q } + ( q - 1 ) ^ { - 1 } } { q - 1 } + \frac { \{ 2 q \zeta _ { q } + ( q - 1 ) ^ { - 1 } \} ^ { 2 } } { 2 ( q - 1 ) ^ { 2 } } .\tag{40}
$$

Proof. Choose a matching attaining $P _ { \psi } ( W , k )$ , and list the weights of its selected vertices as $0 < x _ { 1 } \le \cdots \le x _ { 2 k }$ . We first note that these vertices may be paired from the outside inward. To see this, we take $0 < a \leq b \leq c \leq d$ and set $Y _ { s } : = 1 + s$ . Since $2 \psi ( s , t ) = Y _ { s } Y _ { t } - 1$ , direct subtraction gives

$$
\begin{array} { r l } & { 4 \{ \psi ( a , d ) \psi ( b , c ) - \psi ( a , c ) \psi ( b , d ) \} = ( Y _ { b } - Y _ { a } ) ( Y _ { d } - Y _ { c } ) \geq 0 , } \\ & { 4 \{ \psi ( a , d ) \psi ( b , c ) - \psi ( a , b ) \psi ( c , d ) \} = ( Y _ { c } - Y _ { a } ) ( Y _ { d } - Y _ { b } ) \geq 0 . } \end{array}
$$

We apply these inequalities with $a = x _ { 1 }$ and $d = x _ { 2 k }$ . If the corresponding labeled vertices are not paired with one another, we write $b \leq c$ for the weights of their current partners. Those two edges contribute either $\psi ( a , b ) \psi ( c , d )$ or $\psi ( a , c ) \psi ( b , d )$ , so the two inequalities show that replacing them by $\{ a , d \}$ and $\{ b , c \}$ does not decrease the product. The remaining pairs must still be optimal on the remaining selected vertices. Repeating the same argument recursively, we obtain a maximizing matching that pairs $x _ { i }$ with $x _ { 2 k + 1 - i }$ for every �.

Set $\Sigma _ { E } : = \textstyle \sum _ { i = 1 } ^ { 2 k } x _ { i }$ . The sequence $\left( x _ { i } \right) _ { i = 1 } ^ { k }$ is nondecreasing, whereas $\left( x _ { 2 k + 1 - i } \right) _ { i = 1 } ^ { k }$ is nonincreasing. Hence

$$
k \sum _ { i = 1 } ^ { k } x _ { i } x _ { 2 k + 1 - i } - \left( \sum _ { i = 1 } ^ { k } x _ { i } \right) \left( \sum _ { i = 1 } ^ { k } x _ { 2 k + 1 - i } \right) = \sum _ { 1 \leq i < j \leq k } ( x _ { i } - x _ { j } ) ( x _ { 2 k + 1 - i } - x _ { 2 k + 1 - j } ) \leq 0 .
$$

Therefore, the reverse Chebyshev inequality and the elementary inequality $a b \leq ( a + b ) ^ { 2 } / 4$ give

$$
\sum _ { i = 1 } ^ { k } x _ { i } x _ { 2 k + 1 - i } \leq \frac { 1 } { k } \left( \sum _ { i = 1 } ^ { k } x _ { i } \right) \left( \sum _ { i = k + 1 } ^ { 2 k } x _ { i } \right) \leq \frac { \Sigma _ { E } ^ { 2 } } { 4 k } .
$$

The arithmetic–geometric mean inequality for the � edge weights now yields

$$
P _ { \psi } ( W , k ) ^ { 1 / k } \leq \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \psi ( x _ { i } , x _ { 2 k + 1 - i } ) = \frac { \Sigma _ { E } } { 2 k } + \frac { 1 } { 2 k } \sum _ { i = 1 } ^ { k } x _ { i } x _ { 2 k + 1 - i } \leq \frac { \Sigma _ { E } } { 2 k } + \frac { \Sigma _ { E } ^ { 2 } } { 8 k ^ { 2 } } \leq \frac { \Sigma } { 2 k } + \frac { \Sigma ^ { 2 } } { 8 k ^ { 2 } } .
$$

The last inequality uses $\Sigma _ { E } \leq \Sigma$ and the fact that $x \mapsto x / ( 2 k ) + x ^ { 2 } / ( 8 k ^ { 2 } )$ is increasing on $[ 0 , \infty )$ . This proves (39).

For the particular vertex multiset $W _ { q }$ , we write its total weight as

$$
\Sigma _ { q } : = \sum _ { w \in W _ { q } } w = 2 q \zeta _ { q } + \frac { 1 } { q - 1 } .
$$

For fixed $\Sigma _ { q } > 0$ , the right-hand side of (39) decreases with �. Since $k _ { + } , k _ { - } \ge ( q - 1 ) / 2$ , we can apply the preceding bound to each matching to obtain

$$
P _ { \psi } ( W _ { q } , k _ { \pm } ) ^ { 1 / k _ { \pm } } \leq \frac { \Sigma _ { q } } { q - 1 } + \frac { \Sigma _ { q } ^ { 2 } } { 2 ( q - 1 ) ^ { 2 } } .
$$

Finally, using $k _ { + } + k _ { - } = q$ , we reach

$$
\mu _ { q } = \left( P _ { \psi } ( W _ { q } , k _ { + } ) ^ { 1 / k _ { + } } \right) ^ { k _ { + } / q } \left( P _ { \psi } ( W _ { q } , k _ { - } ) ^ { 1 / k _ { - } } \right) ^ { k _ { - } / q } \leq \frac { \Sigma _ { q } } { q - 1 } + \frac { \Sigma _ { q } ^ { 2 } } { 2 ( q - 1 ) ^ { 2 } } ,
$$

which is precisely (40).

## 6 A cutof argument and completion of the proof

For each rank cutof $2 \leq q \leq r ,$ the preceding section gives an order-independent matching bound expressed in terms of $\mu _ { q } .$ , the geometric mean per edge of the matching product. We compare this quantity with a fixed threshold $\rho < 1$ For all suficiently large ranks $q \geq Q$ , this comparison yields two alternatives. If $\mu _ { q } < \rho ,$ , then the matching bound already produces a large contribution to the objective gap. If $\mu _ { q } \geq \rho _ { \mathrm { \scriptsize ~ . ~ } }$ , then the smallest excess among the first � ranked excesses must be small relative to the remaining schedule mass. When this second alternative persists over many ranks, a Lyapunov argument controls the cumulative growth of the residual schedule mass. Once the argument has reduced the problem to only a fixed number of remaining ranks, a direct prefix estimate completes the lower bound. The admissible parameter choices lead exactly to the threshold $p _ { \star } = \sqrt { 2 + \sqrt { 3 } }$ . Finally, we restore the original scaling and complete the proof of Theorem 2.1.

## 6.1 Choice of parameters and the two cutof alternatives

Recall that (30) holds for every $2 \leq q \leq r$ . In this bound, $D _ { q }$ is the schedule mass left after the � largest excesses have been selected, while $\mu _ { q } ^ { q } = \mathcal { M } _ { q }$ is the corresponding matching product. We will choose a rank at which either this product is small or the fact that it is not small gives control of $D _ { q }$

Our target is the following schedule-independent lower bound.

Proposition 6.1. For every fixed $p \in ( p _ { \star } , 2 )$ , there is $c _ { p } > 0 \}$ , depending only on $p ,$ , such that every nonnegative normalized stepsize schedule ℎ satisfies

$$
C _ { T } ( h ) \geq \frac { c _ { p } } { B ( r + 1 ) ^ { p - 1 } } .\tag{41}
$$

For the rest of the normalized argument, we fix $p \in ( p _ { \star } , 2 )$ and define

$$
\vartheta : = { \frac { 1 } { p ^ { 2 } - 1 } } .\tag{42}
$$

Then we have

$$
2 \vartheta + 2 \vartheta ^ { 2 } < 1 \quad \Longleftrightarrow \quad p ^ { 2 } > 2 + { \sqrt { 3 } } .
$$

Thus, the strict inequality $p > p ,$ <sub>★</sub> allows us to choose $\rho \in ( 0 , 1 )$ so that

$$
2 \vartheta + 2 \vartheta ^ { 2 } < \rho < 1 .\tag{43}
$$

$\mathrm { A t } p = p _ { \star }$ , the left endpoint equals one, so no such $\rho$ exists. This explains why the argument requires the strict inequality $p > p _ { \star }$ 7

We next record how much the residual schedule mass changes when we move the cutof by one rank. For $1 \leq q \leq r ,$ we define

$$
\nu _ { q } : = \frac { q a _ { q } } { D _ { q } } .\tag{44}
$$

Because $D _ { q - 1 } = D _ { q } + a _ { q }$ , the quotient $\nu _ { q } / q$ is the relative increase of $D _ { q }$ when the cutof moves from $q \ \mathrm { t o } \ q - 1$ . This quantity is related to the matching quantity $\zeta _ { q }$ from (37). Indeed, since $a _ { s } \geq a _ { q }$ for $s \leq q$

$$
\zeta _ { q } { \nu _ { q } } = \frac { a _ { q } } { q } \sum _ { s = 1 } ^ { q } \frac { 1 } { a _ { s } } \le 1 .\tag{45}
$$

If $\zeta _ { q } \leq \vartheta$ , then

$$
\frac { 2 q \zeta _ { q } + ( q - 1 ) ^ { - 1 } } { q - 1 } \leq 2 \vartheta + \frac { 2 \vartheta } { q - 1 } + \frac { 1 } { ( q - 1 ) ^ { 2 } } \leq 2 \vartheta + \frac { 2 \vartheta + 1 } { q - 1 } ,
$$

where the last inequality uses $q \geq 2 .$ . Since $x \mapsto x + x ^ { 2 } / 2$ is increasing on $[ 0 , \infty )$ , (40) gives

$$
\mu _ { q } \leq 2 \vartheta + \frac { 2 \vartheta + 1 } { q - 1 } + \frac { 1 } { 2 } \left( 2 \vartheta + \frac { 2 \vartheta + 1 } { q - 1 } \right) ^ { 2 } .
$$

As $q \to \infty$ , the right-hand side decreases to $2 \vartheta + 2 \vartheta ^ { 2 } < \rho$ . We therefore choose an integer $Q \geq 2$ large enough such that

$$
Q > \vartheta ^ { - 1 } , \qquad 2 \vartheta + \frac { 2 \vartheta + 1 } { Q - 1 } + \frac { 1 } { 2 } \left( 2 \vartheta + \frac { 2 \vartheta + 1 } { Q - 1 } \right) ^ { 2 } < \rho .
$$

With this choice, uniformly for every $Q \leq q \leq r ,$ one has

$$
\zeta _ { q } \le \vartheta \quad \Longrightarrow \quad \mu _ { q } < \rho .\tag{46}
$$

Both $\rho$ and � depend only on � and will remain fixed.

For each $Q \leq q \leq r$ , we now consider the two cases $\mu _ { q } < \rho$ and $\mu _ { q } \geq \rho$ . If $\mu _ { q } < \rho$ , then combining (30) with $q / ( q - 1 ) > 1$ gives

$$
\mu _ { q } < \rho \quad \Longrightarrow \quad C _ { T } ( h ) > \frac { \rho ^ { - q } } { 2 D _ { q } } .\tag{47}
$$

On the other hand, if $\mu _ { q } \geq \rho$ , we first apply the contrapositive of (46) and then use (45) to obtain

$$
\mu _ { q } \geq \rho \quad \implies \quad \zeta _ { q } > \vartheta , \qquad 0 < \nu _ { q } < \vartheta ^ { - 1 } .\tag{48}
$$

In the proof below, we take the largest � for which the first alternative holds. At all larger ranks the second alternative holds. If there is no such �, the second alternative holds throughout $\{ Q , \ldots , r \}$ .

## 6.2 Growth of the residual schedule mass and the bounded-rank case

The pointwise bounds in (48) show that, under the large-product alternative, each individual rank can increase the residual schedule mass only by a controlled relative amount. However, these local bounds do not by themselves control the cumulative growth of $D _ { q }$ across many ranks. We therefore pass from pointwise control to an aggregate estimate. First, we record the exact relations between adjacent ranks; then we introduce a Lyapunov potential tailored to the critical exponent $p - 1$

## 6.2.1 Adjacent-rank dynamics

The following lemma collects the exact relations between successive ranks used in the Lyapunov analysis; its proof is deferred to Appendix B.2.

Lemma 6.2. For every $1 \leq q \leq r ,$

$$
\frac { D _ { q - 1 } } { D _ { q } } = 1 + \frac { \nu _ { q } } { q } .\tag{49}
$$

Consequently, for $0 \leq k \leq r ,$

$$
\frac { D _ { k } } { B } = \prod _ { s = k + 1 } ^ { r } \left( 1 + \frac { \nu _ { s } } { s } \right) .\tag{50}
$$

For $1 \leq q < r ,$ monotonicity of the ranked excesses implies

$$
\frac { \nu _ { q } } { q } \left( 1 + \frac { \nu _ { q + 1 } } { q + 1 } \right) \geq \frac { \nu _ { q + 1 } } { q + 1 } ,\tag{51}
$$

and hence, whenever $q > \nu _ { q } ,$

$$
\nu _ { q + 1 } \leq \frac { ( q + 1 ) \nu _ { q } } { q - \nu _ { q } } .\tag{52}
$$

The matching quantity satisfies the exact recursion

$$
\zeta _ { q + 1 } = \frac { q ^ { 2 } \zeta _ { q } } { ( q + 1 ) ( q + 1 + \nu _ { q + 1 } ) } + \frac { 1 } { \nu _ { q + 1 } ( q + 1 ) } \qquad ( 1 \leq q < r ) .\tag{53}
$$

## 6.2.2 A one-step Lyapunov estimate

To sum the changes in $D _ { q } ,$ , we need to control $\Sigma _ { q } \nu _ { q } / q$ . The afine recursion (53) suggests the Lyapunov potential

$$
\mathcal { L } _ { q } : = \frac { \nu _ { q } ( \zeta _ { q } - \vartheta ) } { \vartheta ( \nu _ { q } + p + 1 ) } .\tag{54}
$$

The coeficient of $\zeta _ { q } - \vartheta$ is chosen so that the contraction in (53) is absorbed by the preceding value of the potential.   
The resulting drift estimate bounds the relative mass increase, up to a summable quadratic error.

Lemma 6.3. There is a constant $C _ { p } \geq 1$ , depending only on $p ,$ with the following property. Let $n \geq 2$ be an integer satisfying $n - 1 > \vartheta ^ { - 1 }$ . Suppose

$$
\zeta _ { n - 1 } \ge \vartheta , \qquad 0 < \nu _ { n - 1 } , \nu _ { n } \le \vartheta ^ { - 1 } ,
$$

and suppose

$$
\nu _ { n } \leq \frac { n \nu _ { n - 1 } } { n - 1 - \nu _ { n - 1 } } , \qquad \zeta _ { n } = \frac { ( n - 1 ) ^ { 2 } \zeta _ { n - 1 } } { n ( n + \nu _ { n } ) } + \frac { 1 } { n \nu _ { n } } .
$$

Then

$$
\mathcal { L } _ { n } - \mathcal { L } _ { n - 1 } \leq \frac { p - 1 - \nu _ { n } } { n } + \frac { C _ { p } } { n ^ { 2 } } .\tag{55}
$$

The proof consists of a coeficient comparison followed by a direct calculation of the drift; we defer it to Appendix B.3. Telescoping this inequality then leads to the required bound on $D _ { q }$

Lemma 6.4. There is a constant $K _ { p } \geq 1$ , depending only on $p ,$ such that, whenever $Q \leq k \leq r$ and $\mu _ { q } \geq \rho f o r$ every $q \in \{ k + 1 , \ldots , r \}$

$$
D _ { k } k ^ { p - 1 } \leq K _ { p } B r ^ { p - 1 } .\tag{56}
$$

ProofofLemma 6.4. We take $C _ { p } \geq 1$ to be the constant from Lemma 6.3 and define

$$
K _ { p } : = 2 \exp \Biggl \{ \frac { 1 } { \vartheta ( p + 1 ) } + C _ { p } \frac { \pi ^ { 2 } } { 6 } \Biggr \} .
$$

This constant depends only on $p .$

Estimate on an interval of ranks. We first show that, whenever $Q \leq \ell \leq r$ and $\mu _ { q } \geq \rho$ for every $q \in \{ \ell , \ldots , r \}$ ,

$$
D _ { \ell } \ell ^ { p - 1 } \leq \frac { K _ { p } } { 2 } B r ^ { p - 1 } .\tag{57}
$$

If $\ell = r ,$ this follows from $D _ { r } = B$ and $K _ { p } / 2 \geq 1$ . We now assume $\ell < r .$ Applying (48) and (45) at each of these ranks gives

$$
\zeta _ { q } > \vartheta , \qquad 0 < \nu _ { q } < \vartheta ^ { - 1 } , \qquad \zeta _ { q } \nu _ { q } \leq 1 \quad ( \ell \leq q \leq r ) .\tag{58}
$$

Moreover, for every $q \in \{ \ell , \ldots , r - 1 \} , q \geq Q > \vartheta ^ { - 1 } > \nu _ { q } .$ , so (52) applies to every transition used below.

Specifically, we fix $n \in \{ \ell + 1 , \ldots , r \}$ and apply (52) and (53) with $q = n - 1$ . They give

$$
\nu _ { n } \leq \frac { n \nu _ { n - 1 } } { n - 1 - \nu _ { n - 1 } } , \qquad \zeta _ { n } = \frac { ( n - 1 ) ^ { 2 } \zeta _ { n - 1 } } { n ( n + \nu _ { n } ) } + \frac { 1 } { n \nu _ { n } } .
$$

Together with (58), these are precisely the hypotheses of Lemma 6.3. Hence that lemma is applicable for every $n \in \{ \ell + 1 , \ldots , r \}$

In view of (58), the potential $\mathcal { L } _ { q }$ is uniformly bounded as follows:

$$
0 \le \mathcal { L } _ { q } \le \frac { \nu _ { q } \zeta _ { q } } { \vartheta ( \nu _ { q } + p + 1 ) } \le \frac { 1 } { \vartheta ( \nu _ { q } + p + 1 ) } \le \frac { 1 } { \vartheta ( p + 1 ) } .\tag{59}
$$

In particular, the endpoint contribution satisfies

$$
\mathcal { L } _ { \ell } - \mathcal { L } _ { r } \leq \mathcal { L } _ { \ell } \leq \frac { 1 } { \vartheta ( p + 1 ) } .
$$

For each $n \in \{ \ell + 1 , \ldots , r \}$ , we rearrange (55) to obtain

$$
{ \frac { \nu _ { n } } { n } } \leq { \frac { p - 1 } { n } } + { \frac { C _ { p } } { n ^ { 2 } } } + { \mathcal { L } } _ { n - 1 } - { \mathcal { L } } _ { n } .
$$

We now sum these inequalities and telescope the potential terms to obtain

$$
\begin{array} { l } { \displaystyle \sum _ { n = \ell + 1 } ^ { r } \frac { \nu _ { n } } { n } \le \left( p - 1 \right) \sum _ { n = \ell + 1 } ^ { r } \frac { 1 } { n } + C _ { p } \displaystyle \sum _ { n = \ell + 1 } ^ { r } \frac { 1 } { n ^ { 2 } } + \displaystyle \sum _ { n = \ell + 1 } ^ { r } \left( \mathcal { L } _ { n - 1 } - \mathcal { L } _ { n } \right) } \\ { = \left( p - 1 \right) \displaystyle \sum _ { n = \ell + 1 } ^ { r } \frac { 1 } { n } + C _ { p } \displaystyle \sum _ { n = \ell + 1 } ^ { r } \frac { 1 } { n ^ { 2 } } + \mathcal { L } _ { \ell } - \mathcal { L } _ { r } } \\ { \le \left( p - 1 \right) \log \displaystyle \frac { r } { \ell } + \displaystyle \frac { 1 } { \vartheta ( p + 1 ) } + C _ { p } \displaystyle \frac { \pi ^ { 2 } } { 6 } . } \end{array}
$$

Here, we have used (59), $\textstyle \sum _ { n = \ell + 1 } ^ { r } n ^ { - 1 } \leq \log ( r / \ell )$ , and $\textstyle \sum _ { n = 1 } ^ { \infty } n ^ { - 2 } = \pi ^ { 2 } / 6$ . Finally, we set $k = \ell$ in (50), take logarithms, and use log(1 + �) ≤ � to derive

$$
\log \frac { D _ { \ell } } { B } = \sum _ { n = \ell + 1 } ^ { r } \log \left( 1 + \frac { \nu _ { n } } { n } \right) \leq \sum _ { n = \ell + 1 } ^ { r } \frac { \nu _ { n } } { n } \leq ( p - 1 ) \log \frac { r } { \ell } + \frac { 1 } { \vartheta ( p + 1 ) } + C _ { p } \frac { \pi ^ { 2 } } { 6 } .
$$

We then exponentiate this inequality and multiply by $B \ell ^ { p - 1 }$ to reach

$$
D _ { \ell } \ell ^ { p - 1 } \leq \exp \Biggl \{ \frac { 1 } { \vartheta ( p + 1 ) } + C _ { p } \frac { \pi ^ { 2 } } { 6 } \Biggr \} B r ^ { p - 1 } = \frac { K _ { p } } { 2 } B r ^ { p - 1 } ,
$$

which proves (57).

The remaining boundary rank. If $k = r .$ , then

$$
D _ { k } k ^ { p - 1 } = D _ { r } r ^ { p - 1 } = B r ^ { p - 1 } \leq K _ { p } B r ^ { p - 1 } .
$$

We now suppose $k < r$ . Since $\mu _ { q } \geq \rho$ for every $q \in \{ k + 1 , \ldots , r \} , ( 5 7 )$ at $\ell = k + 1$ gives

$$
D _ { k + 1 } ( k + 1 ) ^ { p - 1 } \leq \frac { K _ { p } } { 2 } B r ^ { p - 1 } .
$$

Moreover, $\mu _ { k + 1 } \geq \rho$ , and hence (48) gives $\nu _ { k + 1 } < \vartheta ^ { - 1 }$ . Therefore, it holds that

$$
\frac { D _ { k } } { D _ { k + 1 } } = 1 + \frac { \nu _ { k + 1 } } { k + 1 } < 1 + \frac { \vartheta ^ { - 1 } } { k + 1 } \leq 1 + \frac { \vartheta ^ { - 1 } } { Q } < 2 ,
$$

where the last inequality uses $Q > \vartheta ^ { - 1 }$ . Since $k ^ { p - 1 } \leq ( k + 1 ) ^ { p - 1 }$ , we can conclude that

$$
D _ { k } k ^ { p - 1 } < 2 D _ { k + 1 } ( k + 1 ) ^ { p - 1 } \leq K _ { p } B r ^ { p - 1 } ,
$$

thereby establishing (56).

## 6.2.3 The bounded-rank case

The preceding estimate controls the residual schedule mass down to any fixed rank. It remains to extract a lower bound using only a fixed number of the largest excesses. The key point is that, among the empty chain and the prefix chains formed from these excesses, at least one achieves a value comparable to the reciprocal of the remaining mass. More concretely, either the empty chain already sufices, or one of the nonempty prefixes has well-controlled local factors.

Lemma 6.5. For everyfixed integer $Q \geq 1$ , there exists a constant $c _ { Q } > 0 ,$ , depending only on $Q ,$ such that,for every $0 \leq q _ { 0 } \leq \operatorname* { m i n } \{ Q , r \}$

$$
C _ { T } ( h ) \geq \frac { c _ { Q } } { D _ { q _ { 0 } } } .\tag{60}
$$

The lemma follows by applying Lemma B.1 at rank $q _ { 0 }$ . The proof details are postponed to Appendix B.4.

## 6.3 Proof of Proposition 6.1

We now combine the growth estimate with the bounded-rank argument to prove Proposition 6.1.

Let $K _ { p } \geq 1$ be the constant supplied by Lemma 6.4. We consider two cases, depending on whether there exists a rank $q \in \{ { Q } , \ldots , { r } \}$ such that $\mu _ { q } < \rho .$

Case 1: $\mu _ { q } < \rho$ for some $q \in \{ { Q , \dots , r } \}$ . We define

$$
k : = \operatorname* { m a x } \big \{ q \in \{ Q , \dots , r \} : \mu _ { q } < \rho \big \} .
$$

By the maximality of $k , \mu _ { q } \geq \rho$ for every $q \in \{ k + 1 , \ldots , r \}$ . Lemma 6.4 therefore gives

$$
D _ { k } k ^ { p - 1 } \leq K _ { p } B r ^ { p - 1 } .
$$

Combining this with (47) yields

$$
C _ { T } ( h ) > \frac { \rho ^ { - k } } { 2 D _ { k } } = \frac { k ^ { p - 1 } \rho ^ { - k } } { 2 D _ { k } k ^ { p - 1 } } \geq \frac { Q ^ { p - 1 } \rho ^ { - Q } } { 2 K _ { p } B r ^ { p - 1 } } \geq \frac { Q ^ { p - 1 } \rho ^ { - Q } } { 2 K _ { p } B ( r + 1 ) ^ { p - 1 } } .
$$

Case 2: $\mu _ { q } \geq \rho$ for every $q \in \{ { Q } , \ldots , { r } \}$ . Set $q _ { 0 } : = \operatorname* { m i n } \{ r , Q \}$ . If $r < Q$ , then $q _ { 0 } = r$ , and hence $D _ { q _ { 0 } } = D _ { r } = B$ . If $r \geq Q$ , then $q _ { 0 } = Q ;$ ; moreover, the assumption of this case gives $\mu _ { q } \geq \rho$ for all $q \in \{ { Q } , \ldots , { r } \}$ . Applying Lemma 6.4 with $k = Q$ therefore yields

$$
D _ { q _ { 0 } } = D _ { Q } \leq K _ { p } B \left( \frac { r } { Q } \right) ^ { p - 1 } .
$$

Since $K _ { p } \geq 1$ and $Q \geq 1$ , these two subcases are both covered by the uniform bound

$$
D _ { q 0 } \leq K _ { p } B ( r + 1 ) ^ { p - 1 } .
$$

Lemma 6.5 then gives

$$
C _ { T } ( h ) \geq \frac { c _ { Q } } { D _ { q _ { 0 } } } \geq \frac { c _ { Q } / K _ { p } } { B ( r + 1 ) ^ { p - 1 } } .
$$

Putting the two cases together. Note that the quantities $\rho , Q , K _ { p }$ , and $c _ { Q }$ depend only on $p$ . We can therefore define

$$
c _ { p } : = \operatorname* { m i n } \left\{ \frac { Q ^ { p - 1 } \rho ^ { - Q } } { 2 K _ { p } } , \frac { c _ { Q } } { K _ { p } } \right\} > 0 .
$$

With this choice of $c _ { p } , ( 4 1 )$ follows from the two cases above.

## 6.4 Restoring the original scaling

Thus far, we have established a lower bound on the normalized quantity $C _ { T } ( h )$ . By Corollary 4.3, this value is attained by a finite-dimensional smooth convex instance. It remains only to restore the original smoothness parameter � and initial-distance scale �. To this end, we have the following lemma.

Lemma 6.6. Let $F \in \mathcal { F } _ { 0 , 1 } ( \mathbb { R } ^ { d } )$ , let $L , R > 0$ , and define

$$
f ( x ) : = L R ^ { 2 } F ( x / R ) .
$$

Then $f \in \mathcal { F } _ { 0 , L } ( \mathbb { R } ^ { d } ) a n d \nabla f ( x ) = L R \nabla F ( x / R ) . \ I f h _ { t } = L \eta _ { t }$ and

$$
\bar { x } _ { t + 1 } = \bar { x } _ { t } - h _ { t } \nabla F ( \bar { x } _ { t } ) , \qquad x _ { t } : = R \bar { x } _ { t } ,
$$

then $x _ { t + 1 } = x _ { t } - \eta _ { t } \nabla f ( x _ { t } )$ . Moreover, every ${ \bar { x } } _ { \star }$ ∈ argmin � gives $x _ { \star } : = R \bar { x }$ <sub>★</sub> ∈ argmin $f ,$ and

$$
\| x _ { 0 } - x _ { \star } \| = R \| \bar { x } _ { 0 } - \bar { x } _ { \star } \| , \qquad f ( x _ { T } ) - f ( x _ { \star } ) = L R ^ { 2 } \{ F ( \bar { x } _ { T } ) - F ( \bar { x } _ { \star } ) \} .
$$

Proof. We first note that positive scaling and composition with $x \mapsto x / R$ preserve convexity and carry minimizers to � argmin �. The chain rule then gives the stated gradient formula. Moreover, 1-smoothness of � yields

$$
\| \nabla f ( x ) - \nabla f ( y ) \| \leq L R \| ( x - y ) / R \| = L \| x - y \| .
$$

Finally, it is readily seen that

$$
x _ { t } - \eta _ { t } \nabla f ( x _ { t } ) = R \{ \bar { x } _ { t } - h _ { t } \nabla F ( \bar { x } _ { t } ) \} = R \bar { x } _ { t + 1 } .
$$

The distance and objective gap identities follow directly from the definitions.

ProofofTheorem 2.1. Fix $p \in ( p _ { \star } , 2 )$ , an integer $T \geq 1$ , parameters $L , R > 0 .$ , and a stepsize schedule $\eta \in \mathbb { R } _ { \geq 0 } ^ { T }$ . We then set $h _ { t } = L \eta _ { t }$ and choose $c _ { p } > 0$ small enough that Proposition 6.1 holds with $2 c _ { p }$ in place of its constant.

Since $B \leq T + 1$ and $r + 1 \leq T + 1$ , Proposition 6.1 gives

$$
C _ { T } ( h ) \geq \frac { 2 c _ { p } } { B ( r + 1 ) ^ { p - 1 } } \geq 2 c _ { p } ( T + 1 ) ^ { - p } .
$$

By Corollary 4.3, there exist an integer $\bar { d }$ with $1 \leq \bar { d } \leq T + 1$ , a function $F \in \mathcal { F } _ { 0 , 1 } ( \mathbb { R } ^ { \bar { d } } )$ with $0 \in$ argmin $F ,$ and a normalized GD trajectory satisfying

$$
\bar { x } _ { t + 1 } = \bar { x } _ { t } - h _ { t } \nabla F ( \bar { x } _ { t } ) \quad ( 0 \leq t < T ) , \qquad \| \bar { x } _ { 0 } \| = 1 ,
$$

for which

$$
2 \{ F ( \bar { x } _ { T } ) - F ( 0 ) \} = C _ { T } ( h ) \geq 2 c _ { p } ( T + 1 ) ^ { - p } .
$$

Apply Lemma 6.6 with $\bar { x } _ { \star } = 0$ and define

$$
f ( x ) = L R ^ { 2 } F ( x / R ) , \qquad x _ { t } = R \bar { x } _ { t } , \qquad x _ { \star } = 0 .
$$

Then $f \in \mathcal { F } _ { 0 , L } ( \mathbb { R } ^ { \bar { d } } )$ , the sequence $( x _ { t } ) _ { t = 0 } ^ { T }$ is exactly the GD trajectory with stepsizes $\eta _ { t }$ , and $\| x _ { 0 } - x _ { \star } \| = R$ . Moreover,

$$
f ( x _ { T } ) - f ( x _ { \star } ) \geq c _ { p } L R ^ { 2 } ( T + 1 ) ^ { - p } .
$$

This concludes the proof of Theorem 2.1.

## 7 Discussion

In this work, we have established a new lower bound for the last-iterate convergence rate of plain GD with arbitrary predetermined nonnegative stepsize schedules designed for a prescribed horizon. In particular, our result has provided the first rigorous evidence that stepsize schedules alone cannot accelerate plain GD to the optimal ${ \cal O } ( T ^ { - 2 } )$ convergence rate. The tight worst-case rate achievable by this class of stepsize-based acceleration methods, however, remains open. We conclude by highlighting several directions for future work.

• Determining the optimal convergence exponent. As mentioned previously, the best-known upper bound for stepsize-based acceleration of GD is $O ( T ^ { - \log _ { 2 } ( 1 + \sqrt { 2 } ) } ) = O ( T ^ { - 1 . 2 7 1 5 \ldots } )$ , whereas our lower bound rules out polynomial exponents exceeding $p _ { \star } \approx 1 . 9 3 1 9$ . Closing this gap by matching the upper and lower bounds remains a central open problem.

• Lower bounds for the best iterate. Thus far, Theorem 2.1 concerns only the last iterate $x _ { T } .$ A stronger result would establish a lower bound for

$$
\operatorname* { m i n } _ { 0 \leq t \leq T } { \left\{ f ( x _ { t } ) - f ( x _ { \star } ) \right\} } .
$$

Such a result cannot be obtained by applying Theorem 2.1 independently to each prefix, since diferent prefixes may require diferent hard instances.

• Lower bounds for strongly convex problems. The silver stepsize schedule also accelerates gradient descent in the strongly convex setting (Altschuler and Parrilo, 2025a). An analogous lower bound for predetermined nonnegative stepsize schedules would improve our understanding of the optimal convergence rate in the strongly convex regime.

• Allowing negative or adaptive stepsizes. Our lower bound relies on both the nonnegativity of the stepsizes and the fact that the entire schedule is fixed in advance. Whether analogous lower bounds hold for signed stepsizes or adaptively chosen stepsizes remains an interesting question for future exploration.

## Disclosure on the use of generative AI

The main proof was developed by GPT-5.6 Sol Pro. The authors provided the model with two inputs: the research objective of proving a lower bound for plain gradient descent showing that its worst-case rate is strictly slower than $1 / T ^ { 2 }$ and a high-level resisting-oracle strategy. That strategy was to first construct an adversarial gradient descent trajectory and then show that the trajectory could be realized by a smooth convex function. Beyond this objective and high-level strategy, the authors did not provide any nontrivial mathematical ingredients used in the final proof.

The proof was not produced in a single interaction. The authors queried GPT-5.6 Sol Pro multiple times, and the argument presented in this paper emerged only after several attempts. GPT-5.6 Sol was also used in later rounds to improve the organization and exposition of the proof. The authors spent substantial efort reviewing, verifying, and revising the generated material to make the proof correct and readable. The authors additionally used Codex to formalize the proof in Lean 4. The formalization is available at https://github.com/jianhaoma/gd-lower-bound-lean. The authors take full responsibility for the final manuscript and all of its mathematical claims

## Acknowledgement

Jianhao thanks Stephen Wright for helpful discussions during the preparation of this manuscript.

## A Proof of the projection formula for Moreau support envelopes

In this section, we prove Lemma 4.2.

Proof of Lemma 4.2. We fix $\boldsymbol { x } \in \mathbb { R } ^ { d }$ and set

$$
x _ { K } : = \Pi _ { K } ( x ) , \qquad z : = x - x _ { K } .
$$

The variational characterization of Euclidean projection gives

$$
\langle z , \nu - x _ { K } \rangle \leq 0 \qquad { \mathrm { f o r ~ e v e r y ~ } } \nu \in K .
$$

Thus, $x _ { K }$ maximizes $\langle \cdot , z \rangle$ over $K .$ . Since $\partial \sigma _ { K } ( z ) = \{ g \in K : \langle g , z \rangle = \sigma _ { K } ( z ) \}$ , we have $x _ { K } \in \partial \sigma _ { K } ( z )$ . Together with $z - x = - x _ { K }$ , this yields

$$
0 \in \partial \sigma _ { K } ( z ) + z - x .
$$

Hence, � is the unique minimizer of the problem in (11), because its objective is strongly convex. It follows that

$$
\operatorname { p r o x } _ { \sigma _ { K } } ( x ) = z = x - \Pi _ { K } ( x ) ,
$$

which proves (12).

The support equality $\sigma _ { K } ( z ) = \langle x _ { K } , z \rangle$ now gives

$$
\begin{array} { l } { \displaystyle { F ( x ) = \langle x _ { K } , z \rangle + \frac { 1 } { 2 } \| x _ { K } \| ^ { 2 } = \langle x _ { K } , x \rangle - \frac { 1 } { 2 } \| x _ { K } \| ^ { 2 } } } \\ { \displaystyle { \quad = \frac { 1 } { 2 } \| x \| ^ { 2 } - \frac { 1 } { 2 } \| x - x _ { K } \| ^ { 2 } = \frac { 1 } { 2 } \| x \| ^ { 2 } - \frac { 1 } { 2 } \operatorname { d i s t } ( x , K ) ^ { 2 } } . } \end{array}\tag{61}
$$

Completing the square gives the equivalent representation

$$
F ( x ) = \operatorname* { m a x } _ { g \in K } \left\{ \langle g , x \rangle - { \frac { 1 } { 2 } } \| g \| ^ { 2 } \right\} .\tag{62}
$$

Completing the square also shows that the maximizer in (62) is $x _ { K }$ , and strong concavity ensures that this maximizer is unique. Moreover, the same representation shows that � is a pointwise maximum of afine functions of � and is therefore convex.

To identify the gradient, we fix $\boldsymbol { y } \in \mathbb { R } ^ { d }$ and set

$$
y _ { K } : = \Pi _ { K } ( y ) , \qquad \Delta : = y - x .
$$

Using the maximization formula for $F ( \cdot )$ , we have

$$
F ( y ) \geq \langle x _ { K } , y \rangle - { \frac { 1 } { 2 } } \| x _ { K } \| ^ { 2 } = F ( x ) + \langle x _ { K } , \Delta \rangle ,
$$

$$
F ( x ) \geq \langle y _ { K } , x \rangle - { \frac { 1 } { 2 } } \| y _ { K } \| ^ { 2 } = F ( y ) - \langle y _ { K } , \Delta \rangle .
$$

Combining these inequalities gives

$$
\langle x _ { K } , \Delta \rangle \leq F ( y ) - F ( x ) \leq \langle y _ { K } , \Delta \rangle .
$$

Consequently,

$$
0 \leq F ( y ) - F ( x ) - \langle x _ { K } , \Delta \rangle \leq \langle y _ { K } - x _ { K } , \Delta \rangle \leq \| y _ { K } - x _ { K } \| \| \Delta \| \leq \| \Delta \| ^ { 2 } ,
$$

where the last inequality uses the nonexpansiveness of Euclidean projection. Dividing by $\| \Delta \|$ and letting $y $ � shows that � is Fréchet diferentiable at � with

$$
\nabla F ( x ) = x _ { K } = \Pi _ { K } ( x ) .
$$

Because � was arbitrary, the nonexpansiveness of $\Pi _ { K }$ also shows that $\nabla F$ is 1-Lipschitz. Together with (61), this proves (13) and completes the proof.

## B Technical proofs for the schedule analysis

We collect here four estimates used in the schedule analysis: the equalization bound under a mass constraint, the recursions as the rank cutof changes, the one-step Lyapunov estimate, and the bounded-rank argument.

## B.1 Proof of Lemma 5.2

Set $\widehat { u _ { i } } : = u _ { i } / \bar { u }$ and $\beta _ { i } : = { \bar { u } } B _ { i } / A _ { i }$ . For $x > 0$ and $\beta \geq 0$

$$
\frac { x ( 1 + \beta x ) } { 1 + 2 \beta } = \frac { x } { 1 + 2 \beta } + \frac { 2 \beta } { 1 + 2 \beta } \frac { x ^ { 2 } } { 2 } \leq e ^ { x - 1 } .
$$

Indeed, $x \leq e ^ { x - 1 }$ , and $x ^ { 2 } / 2 \leq e ^ { x - 1 }$ because $g ( x ) : = x ^ { 2 } e ^ { 1 - x } / 2$ satisfies $g ^ { \prime } ( x ) = g ( x ) ( 2 / x - 1 )$ and hence attains its maximum at $x = 2$ , with $g ( 2 ) = 2 / e < 1$ . The ratio of the left-hand side of (31) to its right-hand side is therefore

$$
\frac { ( \prod _ { i = 1 } ^ { q } u _ { i } ) \prod _ { i = 1 } ^ { q - 1 } ( A _ { i } + B _ { i } u _ { i } ) } { \bar { u } ^ { q } \prod _ { i = 1 } ^ { q - 1 } ( A _ { i } + 2 \bar { u } B _ { i } ) } = \widehat { u } _ { q } \prod _ { i = 1 } ^ { q - 1 } \frac { \widehat { u } _ { i } ( 1 + \beta _ { i } \widehat { u _ { i } } ) } { 1 + 2 \beta _ { i } } \leq \exp \left( \sum _ { i = 1 } ^ { q } \widehat { u } _ { i } - q \right) \leq 1 ,
$$

since $\begin{array} { r } { \sum _ { i } \widehat { u _ { i } } = \bar { u } ^ { - 1 } \sum _ { i } u _ { i } \le D / \bar { u } = q , } \end{array}$ . The same conclusion holds when $q = 1$ , in which case the product over $i < q$ is empty.

## B.2 Proof of Lemma 6.2

The identity $D _ { q - 1 } = D _ { q } + a _ { q }$ and the definition of $\nu _ { q }$ give (49). Multiplying these adjacent ratios and using $D _ { r } = B$ lead to (50), with an empty product when $k = r$

Next, we divide $a _ { q } \geq a _ { q + 1 }$ by $D _ { q }$ and use $D _ { q } = D _ { q + 1 } \{ 1 + \nu _ { q + 1 } / ( q + 1 ) \}$ to obtain

$$
\frac { \nu _ { q } } { q } = \frac { a _ { q } } { D _ { q } } \geq \frac { a _ { q + 1 } } { D _ { q } } = \frac { \nu _ { q + 1 } / ( q + 1 ) } { 1 + \nu _ { q + 1 } / ( q + 1 ) } .
$$

This is exactly (51). Rearranging terms gives

$$
\nu _ { q + 1 } ( q - \nu _ { q } ) \leq ( q + 1 ) \nu _ { q } .
$$

When $q > \nu _ { q }$ , division by the positive denominator proves (52).

Finally, the definition of $\zeta _ { q }$ gives

$$
\sum _ { s = 1 } ^ { q + 1 } \frac { 1 } { a _ { s } } = \frac { q ^ { 2 } \zeta _ { q } } { D _ { q } } + \frac { q + 1 } { \nu _ { q + 1 } D _ { q + 1 } } .
$$

Therefore, we obtain

$$
\begin{array} { l } { { \zeta _ { q + 1 } = \displaystyle \frac { D _ { q + 1 } } { ( q + 1 ) ^ { 2 } } \left( \displaystyle \frac { q ^ { 2 } \zeta _ { q } } { D _ { q } } + \displaystyle \frac { q + 1 } { \nu _ { q + 1 } D _ { q + 1 } } \right) } } \\ { { \mathrm { } = \displaystyle \frac { q ^ { 2 } \zeta _ { q } } { ( q + 1 ) ^ { 2 } } \displaystyle \frac { D _ { q + 1 } } { D _ { q } } + \displaystyle \frac { 1 } { \nu _ { q + 1 } ( q + 1 ) } } } \\ { { \mathrm { } = \displaystyle \frac { q ^ { 2 } \zeta _ { q } } { ( q + 1 ) ( q + 1 + \nu _ { q + 1 } ) } + \displaystyle \frac { 1 } { \nu _ { q + 1 } ( q + 1 ) } , } } \end{array}
$$

which establishes (53).

## B.3 Proof of Lemma 6.3

In this proof, we use the recursions and potential from Subsection 6.2. We shall also use the notation

$$
A ( \nu ) : = \frac { \nu } { \vartheta ( \nu + p + 1 ) } , \qquad \mathcal { L } _ { q } = A ( \nu _ { q } ) ( \zeta _ { q } - \vartheta ) ,
$$

and choose $C _ { p } \geq 1$ so that

$$
C _ { p } \geq \frac { \vartheta ^ { - 1 } \big \{ 1 + \vartheta ^ { - 1 } ( \vartheta ^ { - 1 } + 2 ) \big \} } { p + 1 } .
$$

Coeficient comparison. We abbreviate $\nu : = \nu _ { n }$ and $u : = \nu _ { n - 1 }$ , and we define

$$
\omega _ { n } ( \nu ) : = \frac { ( n - 1 ) ^ { 2 } } { n ( n + \nu ) } .
$$

Because $n - 1 > \vartheta ^ { - 1 } \geq u .$ , the denominator in the assumed upper bound on � is positive. We now set

$$
u _ { \mathrm { m i n } } : = \frac { ( n - 1 ) \nu } { n + \nu } .
$$

Rearranging that bound gives

$$
u \geq u _ { \mathrm { m i n } } .\tag{63}
$$

The derivative of � and the ratio $A ( x ) / x$ satisfy

$$
A ^ { \prime } ( x ) = { \frac { p + 1 } { \vartheta ( x + p + 1 ) ^ { 2 } } } > 0 , \qquad { \frac { A ( x ) } { x } } = { \frac { 1 } { \vartheta ( x + p + 1 ) } } .
$$

Thus, � is increasing on (0, ∞), whereas $A ( x ) / x$ is decreasing there. We also have $0 < u _ { \mathrm { m i n } } < \nu .$ , so

$$
A ( u ) \geq A ( u _ { \operatorname* { m i n } } ) \geq \frac { u _ { \operatorname* { m i n } } } { \nu } A ( \nu ) = \frac { n - 1 } { n + \nu } A ( \nu ) \geq \frac { ( n - 1 ) ^ { 2 } } { n ( n + \nu ) } A ( \nu ) = \omega _ { n } ( \nu ) A ( \nu ) .\tag{64}
$$

Drift calculation. We define the drift term

$$
\Delta _ { n } ( \nu ) : = \frac { 1 } { n \nu } - \frac { \vartheta \{ n ( \nu + 2 ) - 1 \} } { n ( n + \nu ) } .
$$

Subtracting � from the recursion gives

$$
\begin{array} { r } { \zeta _ { n } - \vartheta = \omega _ { n } ( \nu ) ( \zeta _ { n - 1 } - \vartheta ) + \Delta _ { n } ( \nu ) . } \end{array}
$$

Here we use the identity

$$
1 - \omega _ { n } ( \nu ) = \frac { n ( n + \nu ) - ( n - 1 ) ^ { 2 } } { n ( n + \nu ) } = \frac { n ( \nu + 2 ) - 1 } { n ( n + \nu ) } .
$$

Using (42), direct simplification gives

$$
A ( \nu ) \Delta _ { n } ( \nu ) = \frac { 1 } { n } \left[ p - 1 - \nu + \frac { \nu \{ 1 + \nu ( \nu + 2 ) \} } { ( n + \nu ) ( \nu + p + 1 ) } \right] .\tag{65}
$$

To verify this identity, we use $\vartheta ^ { - 1 } = ( p - 1 ) ( p + 1 ) { \mathrm { a n d } } ( p + 1 ) - ( p - 1 ) = 2$ to obtain

$$
n A ( \nu ) \Delta _ { n } ( \nu ) - ( p - 1 - \nu ) = \frac { \nu } { \nu + p + 1 } \left[ ( \nu + 2 ) - \frac { n ( \nu + 2 ) - 1 } { n + \nu } \right] = \frac { \nu \{ 1 + \nu ( \nu + 2 ) \} } { ( n + \nu ) ( \nu + p + 1 ) } ,
$$

which is precisely (65). Because $\zeta _ { n - 1 } - \vartheta \geq 0 , ( 6 4 )$ gives

$$
\begin{array} { r } { \mathcal { L } _ { n } = A ( \nu ) \omega _ { n } ( \nu ) ( \zeta _ { n - 1 } - \vartheta ) + A ( \nu ) \Delta _ { n } ( \nu ) \leq A ( u ) ( \zeta _ { n - 1 } - \vartheta ) + A ( \nu ) \Delta _ { n } ( \nu ) = \mathcal { L } _ { n - 1 } + A ( \nu ) \Delta _ { n } ( \nu ) . } \end{array}
$$

Moreover, $0 < \nu \leq \vartheta ^ { - 1 }$ implies

$$
0 \leq \frac { \nu \{ 1 + \nu ( \nu + 2 ) \} } { ( n + \nu ) ( \nu + p + 1 ) } \leq \frac { 1 } { n } \frac { \vartheta ^ { - 1 } \{ 1 + \vartheta ^ { - 1 } ( \vartheta ^ { - 1 } + 2 ) \} } { p + 1 } \leq \frac { C _ { p } } { n } .
$$

Here we use $n / ( n + \nu ) \leq 1 , \nu / ( \nu + p + 1 ) \leq \vartheta ^ { - 1 } / ( p + 1 )$ , and $1 + \nu ( \nu + 2 ) \leq 1 + \vartheta ^ { - 1 } ( \vartheta ^ { - 1 } + 2 )$ . Combining this estimate with (65) proves (55).

## B.4 The bounded-rank case

The next lemma provides the estimate used in the proof of Lemma 6.5.

Lemma B.1 (Bounded-rank prefix bound). Let $0 \leq q _ { 0 } \leq r ,$ and regard $D _ { q _ { 0 } }$ as the residual schedule mass after the top $q _ { 0 }$ excesses have been selected. When $q _ { 0 } \geq 1$ , consider the chronological chains whose selected sets are the global rank prefixes

$$
\emptyset , \quad \{ \tau _ { 1 } \} , \quad \{ \tau _ { 1 } , \tau _ { 2 } \} , \quad \dots , \quad \{ \tau _ { 1 } , \dots , \tau _ { q _ { 0 } } \} .
$$

Each nonempty prefix is restored to chronological order before applying the geometric construction, with equality in each local bound. The value ofa chain with � selected steps is

$$
{ \frac { 1 } { H _ { m } } } \prod _ { i = 0 } ^ { m - 1 } \chi _ { i } .
$$

This is the chain’s contribution to $C _ { T } ( h )$ in (20). When $q _ { 0 } \geq 1 ,$ , at least one of these chains has value at least

$$
\frac { 1 } { 4 D _ { q _ { 0 } } q _ { 0 } 8 ^ { q _ { 0 } - 1 } } .\tag{66}
$$

For $q _ { 0 } = 0 ,$ , the empty chain has value at least $( 2 D _ { 0 } ) ^ { - 1 }$

Proof of Lemma B.1. We use a difuse–dense dichotomy. Suppose first that, for every prefix, the product of its smallest excess and its length is smaller than the residual schedule mass behind it. Telescoping then shows that the top �<sub>0</sub> excesses increase the residual schedule mass $D _ { q _ { 0 } }$ by at most a factor $q _ { 0 } + 1$ , and the empty chain gives the desired bound. If a dense prefix exists instead, the largest such prefix has both a controlled residual schedule mass and a large smallest selected excess. These two facts control the local factors in the construction.

If $q _ { 0 } = 0 ;$ , the empty-chain value is $( 2 D _ { 0 } - 1 ) ^ { - 1 } \ge ( 2 D _ { 0 } ) ^ { - 1 }$ . For the rest of the proof, we assume that $q _ { 0 } \geq 1$ . For $0 \leq j \leq q _ { 0 }$ , the residual schedule masses satisfy

$$
D _ { j } = D _ { q _ { 0 } } + \sum _ { s = j + 1 } ^ { q _ { 0 } } a _ { s } , \qquad D _ { j - 1 } = D _ { j } + a _ { j } .
$$

We call the rank prefix of length $j \in \{ 1 , \dots , q _ { 0 } \}$ dense if $j a _ { j } \geq D _ { j }$

Case 1: all prefixes are difuse. In this case, $j a _ { j } < D _ { j }$ for every $1 \leq j \leq q _ { 0 }$ . The tail recursion telescopes to

$$
\frac { D _ { 0 } } { D _ { q _ { 0 } } } = \prod _ { j = 1 } ^ { q _ { 0 } } \left( 1 + \frac { a _ { j } } { D _ { j } } \right) < \prod _ { j = 1 } ^ { q _ { 0 } } \left( 1 + \frac { 1 } { j } \right) = q _ { 0 } + 1 .
$$

Consequently, the empty-chain value satisfies

$$
\frac { 1 } { 2 D _ { 0 } - 1 } \geq \frac { 1 } { 2 D _ { 0 } } > \frac { 1 } { 2 D _ { q _ { 0 } } ( q _ { 0 } + 1 ) } \geq \frac { 1 } { 4 D _ { q _ { 0 } } q _ { 0 } 8 ^ { q _ { 0 } - 1 } } .
$$

The last inequality follows from $2 ( q _ { 0 } + 1 ) \leq 4 q _ { 0 } 8 ^ { q _ { 0 } - 1 }$

Case 2: a dense prefix exists. We now take $q$ to be the largest index for which the rank prefix is dense. Every prefix of length $j > q$ is difuse, so telescoping the residual schedule mass beyond $q$ gives

$$
\frac { D _ { q } } { D _ { q _ { 0 } } } = \prod _ { j = q + 1 } ^ { q _ { 0 } } \left( 1 + \frac { a _ { j } } { D _ { j } } \right) \le \prod _ { j = q + 1 } ^ { q _ { 0 } } \left( 1 + \frac { 1 } { j } \right) = \frac { q _ { 0 } + 1 } { q + 1 } .
$$

Together with the density of the prefix of length $q ,$ this gives

$$
D _ { q } \leq D _ { q _ { 0 } } \frac { q _ { 0 } + 1 } { q + 1 } , \qquad a _ { q } \geq \frac { D _ { q } } { q } .\tag{67}
$$

The first inequality controls the residual schedule mass after the prefix, while the second controls every $\chi _ { i }$ through the smallest selected excess.

We select the labeled prefix $\{ \tau _ { 1 } , . . . , \tau _ { q } \}$ and restore its chronological order. Specifically, we define � as the unique permutation of $\{ 1 , \ldots , q \}$ such that

$$
\tau _ { \pi ( 1 ) } < \cdot \cdot \cdot < \tau _ { \pi ( q ) } ,
$$

and we then set $c _ { i } : = a _ { \pi ( i ) } = y _ { \tau _ { \pi ( i ) } }$ and use the block notation with $t _ { i } : = \tau _ { \pi ( i ) }$ for $1 \leq i \leq q$ . We also set

$$
V : = 1 + \sum _ { t = \tau _ { \pi ( q ) } + 1 } ^ { T - 1 } h _ { t } , \qquad H _ { q } = 2 V - 1 .
$$

Because the gaps partition the time indices outside the selected prefix,

$$
\sum _ { i = 0 } ^ { q - 1 } U _ { i } + V = q + 1 + \sum _ { \substack { t \notin \{ \tau _ { 1 } , \ldots , \tau _ { q } \} } } h _ { t } = B + \sum _ { s = q + 1 } ^ { r } a _ { s } = D _ { q } .\tag{68}
$$

For this chain, we set $\gamma _ { i } ^ { 2 } = \chi _ { i }$ for $0 \leq i < q$ and define $a _ { \mathrm { { m i n } } } : = a _ { q }$

The nonterminal factors. We define $\varphi ( x ) : = 2 x + x ^ { 2 } .$ . For $1 \leq i < q , c _ { i } , c _ { i + 1 } \geq a _ { \operatorname* { m i n } }$ and $H _ { i } = U _ { i } + c _ { i + 1 } \geq a _ { \operatorname* { m i n } }$ Hence, for $1 \leq i < q$ , the first identity in (26) gives

$$
\chi _ { i - 1 } ^ { - 1 } \leq \varphi ( U _ { i - 1 } / a _ { \mathrm { m i n } } ) .\tag{69}
$$

Indeed, the three terms in that identity are bounded by $U _ { i - 1 } / a _ { \mathrm { m i n } } , U _ { i - 1 } / a _ { \mathrm { m i n } }$ , and $U _ { i - 1 } ^ { 2 } / a _ { \operatorname* { m i n } } ^ { 2 }$ . Moreover,

$$
\sum _ { i = 0 } ^ { q - 2 } \frac { U _ { i } } { a _ { \operatorname* { m i n } } } \leq \frac { D _ { q } } { a _ { \operatorname* { m i n } } } \leq q
$$

by (68) and (67). Since

$$
\frac { d ^ { 2 } } { d x ^ { 2 } } \log \varphi ( x ) = - \frac { 1 } { x ^ { 2 } } - \frac { 1 } { ( x + 2 ) ^ { 2 } } < 0 ,
$$

log � is concave on (0, ∞). Moreover, $\varphi ^ { \prime } ( x ) = 2 + 2 x > 0$ , so $\varphi$ is increasing. For $q \geq 2$ , Jensen’s inequality and the preceding mass bound give

$$
\prod _ { i = 0 } ^ { q - 2 } { { \chi } _ { i } ^ { - 1 } } \leq \varphi \left( \frac { 1 } { q - 1 } \sum _ { i = 0 } ^ { q - 2 } { \frac { U _ { i } } { a _ { \operatorname* { m i n } } } } \right) ^ { q - 1 } \leq \varphi \left( \frac { q } { q - 1 } \right) ^ { q - 1 } \leq 8 ^ { q - 1 } .\tag{70}
$$

The last inequality uses $q / ( q - 1 ) \leq 2 { \mathrm { ~ a n d ~ } } \varphi ( 2 ) = 8$ . When $q = 1$ , the internal product is empty and the same bound holds. Thus

$$
\prod _ { i = 0 } ^ { q - 2 } \chi _ { i } ^ { - 1 } \leq 8 ^ { q - 1 } \qquad ( q \geq 1 ) .\tag{71}
$$

The terminal factor. The mass decomposition (68) also gives, with the sum interpreted as empty when $q = 1$

$$
2 D _ { q } - ( U _ { q - 1 } + H _ { q } ) = 2 \sum _ { i = 0 } ^ { q - 2 } U _ { i } + U _ { q - 1 } + 1 \ge 0 .
$$

Thus $U _ { q - 1 } \leq D _ { q }$ and $U _ { q - 1 } + H _ { q } \leq 2 D _ { q }$ . The second identity in (26), followed by (67), now yields

$$
H _ { q } \chi _ { q - 1 } ^ { - 1 } \leq D _ { q } \left( 1 + \frac { 2 D _ { q } } { a _ { \operatorname* { m i n } } } \right) \leq 3 q D _ { q } \leq 3 D _ { q _ { 0 } } q _ { 0 } .\tag{72}
$$

Here the last inequality uses $q ( q _ { 0 } + 1 ) / ( q + 1 ) \leq q _ { 0 }$

The selected prefix therefore has value

$$
\frac { 1 } { H _ { q } } \prod _ { i = 0 } ^ { q - 1 } { \chi _ { i } } = \frac { 1 } { ( H _ { q } \chi _ { q - 1 } ^ { - 1 } ) \prod _ { i = 0 } ^ { q - 2 } \chi _ { i } ^ { - 1 } } \ge \frac { 1 } { 3 D _ { q _ { 0 } } q _ { 0 } 8 ^ { q - 1 } } \ge \frac { 1 } { 4 D _ { q _ { 0 } } q _ { 0 } 8 ^ { q _ { 0 } - 1 } } .
$$

We state the result with 4 and $q _ { 0 } .$ , rather than 3 and the selected rank $q ,$ to obtain a simple bound uniform over all possible prefixes. This proves (66). □

Proof of Lemma 6.5. Lemma B.1 gives $C _ { T } ( h ) \geq ( 2 D _ { 0 } ) ^ { - 1 }$ when $q _ { 0 } = 0 ,$ , and for $1 \leq q _ { 0 } \leq \operatorname* { m i n } \{ Q , r \}$ it gives

$$
C _ { T } ( h ) \geq \frac { 1 } { 4 D _ { q _ { 0 } } q _ { 0 } 8 ^ { q _ { 0 } - 1 } } .
$$

We can therefore set

$$
c _ { Q } : = \frac { 1 } { 4 Q 8 ^ { Q - 1 } } .
$$

This choice works simultaneously for every $0 \leq q _ { 0 } \leq \operatorname* { m i n } \{ Q , r \}$ . Indeed, $c _ { Q } \leq 1 / 2$ , and $q _ { 0 } 8 ^ { q _ { 0 } - 1 } \leq Q 8 ^ { Q - 1 }$ for $1 \leq q _ { 0 } \leq Q$ □

## References

Hadi Abbaszadehpeivasti, Etienne de Klerk, and Moslem Zamani. The exact worst-case convergence rate of the gradient method with fixed step lengths for l-smooth functions. Optimization Letters, 16(6):1649–1661, 2022.

Naman Agarwal, Surbhi Goel, and Cyril Zhang. Acceleration via fractal learning rate schedules. In International Conference on Machine Learning, pages 87–99, 2021.

Jason M Altschuler and Pablo A Parrilo. Acceleration by random stepsizes: Hedging, equalization, and the arcsine stepsize schedule. arXiv preprint arXiv:2412.05790, 2024.

Jason M Altschuler and Pablo A Parrilo. Acceleration by stepsize hedging: Multi-step descent and the silver stepsize schedule. Journal ofthe ACM, 72(2):1–38, 2025a.

Jason M Altschuler and Pablo A Parrilo. Acceleration by stepsize hedging: Silver stepsize schedule for smooth convex optimization. Mathematical Programming, 213(1):1105–1118, 2025b.

Yossi Arjevani and Ohad Shamir. On the iteration complexity of oblivious first-order optimization algorithms. In International Conference on Machine Learning, pages 908–916, 2016.

Amir Beck. First-order methods in optimization. SIAM, 2017.

Jinho Bok and Jason M Altschuler. Accelerating proximal gradient descent via silver stepsizes. In Conference on Learning Theory, pages 421–453, 2025.

Sébastien Bubeck. Convex optimization: Algorithms and complexity. Foundations and trends in Machine Learning, 8 (3-4):231–357, 2015.

Antoine Daccache. Performance estimation of the gradient method with fixed arbitrary step sizes. Master’s thesis, Université Catholique de Louvain, 2019.

Shuvomoy Das Gupta, Bart PG Van Parys, and Ernest K Ryu. Branch-and-bound performance estimation programming: a unified methodology for constructing optimal optimization methods. Mathematical Programming, 204(1-2): 567–639, 2024.

ELOI Diego. Worst-case functions for the gradient method with fixed variable step sizes. 2022.

Yoel Drori. The exact information-based complexity of smooth convex minimization. arXiv preprint arXiv:1606.01424, 2016.

Yoel Drori and Marc Teboulle. Performance of first-order methods for smooth convex minimization: a novel approach. Mathematical Programming, 145(1):451–482, 2014.

John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal ofmachine learning research, 12(7), 2011.

Benjamin Grimmer. Provably faster gradient descent via long steps. SIAM Journal on Optimization, 34(3):2588–2608, 2024.

Benjamin Grimmer, Kevin Shu, and Alex L Wang. Composing optimized stepsize schedules for gradient descent. Mathematics ofOperations Research, 2025a.

Benjamin Grimmer, Kevin Shu, and Alex L Wang. Accelerated objective gap and gradient norm convergence for gradient descent via long steps. INFORMS Journal on Optimization, 7(2), 2025b.

Yassine Kamri, Julien M Hendrickx, and François Glineur. Numerical design of optimized first-order algorithms. arXiv preprint arXiv:2507.20773, 2025.

Donghwan Kim and Jefrey A Fessler. Optimized first-order methods for smooth convex minimization. Mathematical programming, 159(1):81–107, 2016.

Jungbin Kim. A proof of the exact convergence rate of gradient descent. arXiv preprint arXiv:2412.04427, 2024.

Guy Kornowski and Ohad Shamir. Open problem: Anytime convergence rate of gradient descent. In Conference on Learning Theory, pages 5335–5339, 2024.

Alan Luner and Benjamin Grimmer. On averaging and extrapolation for gradient descent. Mathematics of Operations Research, 2025.

Yura Malitsky and Konstantin Mishchenko. Adaptive gradient descent without descent. In International Conference on Machine Learning, pages 6702–6712, 2020.

Yurii E. Nesterov. A method of solving a convex programming problem with convergence rate $O ( 1 / k ^ { 2 } )$ . Soviet Mathematics Doklady, 27(2):372–376, 1983.

Teodor Rotaru, François Glineur, and Panagiotis Patrinos. Exact worst-case convergence rates of gradient descent: a complete analysis for all constant stepsizes over nonconvex and convex functions. Mathematical Programming, pages 1–67, 2026.

Henry Shugart and Jason M Altschuler. Negative stepsizes make gradient-descent-ascent converge. arXiv preprint arXiv:2505.01423, 2025.

Adrien B Taylor, Julien M Hendrickx, and François Glineur. Smooth strongly convex interpolation and exact worst-case performance of first-order methods. Mathematical Programming, 161(1):307–345, 2017.

Chung-En Tsai, Ilyas Fatkhullin, Liang Zhang, and Niao He. Lower bounds for anytime acceleration of gradient descent. arXiv preprint arXiv:2607.02053, 2026.

Jingfeng Wu, Peter L Bartlett, Matus Telgarsky, and Bin Yu. Large stepsize gradient descent for logistic loss: Nonmonotonicity of the loss improves optimization eficiency. In Conference on Learning Theory, pages 5019–5073, 2024.

David Young. On richardson’s method for solving linear systems with positive definite matrices. Journal of Mathematics and Physics, 32(1-4):243–255, 1953.

Zehao Zhang and Rujun Jiang. Accelerated gradient descent by concatenation of stepsize schedules. arXiv preprint arXiv:2410.12395, 2024.

Zihan Zhang, Jason Lee, Simon Du, and Yuxin Chen. Anytime acceleration of gradient descent. In Conference on Learning Theory, pages 5991–6013, 2025.