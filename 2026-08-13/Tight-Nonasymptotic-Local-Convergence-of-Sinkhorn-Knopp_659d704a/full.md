# Tight Nonasymptotic Local Convergence of Sinkhorn-Knopp

Wenzhi Gao<sup>∗1</sup>, Zhaonan Qu<sup>†2</sup>, Yinyu Ye<sup>‡1,3</sup>, and Madeleine Udell<sup>§1,3</sup>

<sup>1</sup>ICME, Stanford University

<sup>2</sup>Columbia University

<sup>3</sup>Department of Management Science and Engineering, Stanford University

August 13, 2026

## Abstract

We revisit the Sinkhorn-Knopp (SK) algorithm for the matrix scaling problem. Despite extensive literature on the global convergence of SK and its variants, its local linear convergence behavior remains less understood. We address this gap by providing the first nonasymptotic local analysis of SK that matches the rate obtained from existing asymptotic Jacobian-based arguments. We show that under certain connectivity conditions, SK is a polynomial-time algorithm for doubly stochastic matrix scaling. With the developed tools, we showcase the local suboptimality of SK and provide accelerated variants. Finally, for dense matrices, we improve the complexity of existing first-order matrix scaling algorithms from $\begin{array} { r l } { \mathcal { O } ( \frac { n ^ { 7 / 3 } } { \varepsilon ^ { 2 / 3 } } ) \mathrm { t o } \mathcal { O } ( \frac { n ^ { 9 / 4 } } { \sqrt { \varepsilon } } ) } & { { } } \end{array}$

## 1 Introduction

Given a nonnegative matrix $A \in \mathbb { R } _ { + } ^ { m \times n }$ and two positive margin vectors $p \in \mathbb { R } _ { + + } ^ { m } , q \in \mathbb { R } _ { + + } ^ { n }$ , the matrix scaling problem $( A , p , q )$ seeks two positive diagonal scaling matrices $D _ { 1 } ^ { \star } , D _ { 2 } ^ { \star }$ such that the scaled matrix $A ^ { \star } : = D _ { 1 } ^ { \star } A D _ { 2 } ^ { \star }$ (approximately) satisfies the target margin condition $A ^ { \star } \mathbf { 1 } _ { n } = p$ and $( A ^ { \star } ) ^ { \top } \mathbf { 1 } _ { m } = q$ . Matrix scaling arises in a wide range of applications, including computational optimal transport [3, 34], numerical linear algebra [25], choice modeling [35], neural architecture design [39], and many others [19].

Among the algorithms for matrix scaling, Sinkhorn-Knopp (SK) [36] is arguably the simplest and most widely used. It is also known in the literature as the RAS algorithm [4]. Algorithmically, SK alternates between the two scaling matrices: it fixes one scaling matrix and updates the other so that the corresponding marginal condition is satisfied. Theoretically, the global worst-case iteration complexity of SK is now well understood, following a sequence of works [21, 24, 6, 3, 11, 18]. In terms of target accuracy ε, and ignoring the dimension dependence, worst-case $\mathcal { O } ( \textstyle { \frac { 1 } { \varepsilon } } )$ complexity upper and lower bounds for SK have been established in the literature [11, 18].

Despite this conservative worst-case bound, SK often reaches medium-to-high accuracy in only a few iterations in practice. In particular, it is frequently observed that SK exhibits local linear convergence [34]. On one hand, a line of work has developed nonasymptotic global linear convergence guarantees for SK [13, 35, 18]. These rates are easily computable directly from the problem data $( A , p , q )$ , but are generally not tight. On the other hand, by viewing SK as a fixed-point iteration, the literature has obtained a much sharper local linear convergence rate via Jacobian linearization [25]. In particular, the asymptotic convergence rate is

$$
1 - \sigma _ { 2 } : = \lambda _ { m - 1 } ( P ^ { - 1 / 2 } A ^ { \star } Q ^ { - 1 } ( A ^ { \star } ) ^ { \top } P ^ { - 1 / 2 } ) ,\tag{1}
$$

where $P : = \mathrm { d i a g } ( p ) , Q : = \mathrm { d i a g } ( q )$ , and $\lambda _ { m - 1 }$ denotes the second-largest eigenvalue. However, since this Jacobianbased argument applies in the limit as the iterates approach the optimum, it only yields an asymptotic guarantee. This gap motivates us to establish a nonasymptotic local linear convergence analysis whose rate matches the sharp asymptotic rate in (1).

## Contributions.

• We provide the first tight nonasymptotic local linear convergence analysis of SK, establishing an $\mathcal { O } ( c +$ $\textstyle { \frac { 1 } { \sigma _ { 2 } } } \log ( { \frac { 1 } { \varepsilon } } ) \big )$ iteration complexity. In particular, by explicitly computing the problem-dependent constant $c ,$ we show that SK is a polynomial-time algorithm for the doubly-stochastic matrix scaling problem when $\sigma _ { 2 }$ is treated as a fixed constant. Our analysis also provides a general template for proving nonasymptotic local convergence rates of alternating minimization algorithms.

• Building on these tools, we derive accelerated variants of SK that achieve improved nonasymptotic global and local complexity bounds for the matrix scaling problem.

## 1.1 Related literature

There is a vast literature on the matrix scaling problem. We focus on recent complexity-theoretic advances and refer interested readers to [19] for a comprehensive review.

Matrix scaling and balancing. Matrix scaling is a widely studied problem in numerical linear algebra [25] and optimization [10]. Early works on the complexity of matrix scaling problem treat the problem as a structured convex problem and obtain polynomial-time algorithms that depend on log $\| A \| _ { \infty } , \log \| A \| _ { - \infty } , \log \| p \|$ 1 and $\log ( \frac { 1 } { \varepsilon } )$ [31, 23]. [29] further removes the dependence on log $\| A \| _ { - \infty }$ and obtains a strongly polynomial-time algorithm. Recent advances on the matrix scaling problem, including [2, 7], develop first- and second-order methods for the matrix scaling problem. The matrix scaling problem is also closely related to matrix balancing and equilibration. See [25, 10] for a more detailed discussion between these problems.

Sublinear convergence of Sinkhorn-Knopp. The global sublinear convergence of SK was first established in [22, 24] and later improved in a sequence of works [3, 11, 6]. The current state-of-the-art complexity of SK is $\mathcal { O } ( \textstyle { \frac { D } { \varepsilon } } )$ , where D is an upper bound on the minimum-norm solution pair in log space. Recently, [18] establishes an $\Omega ( \frac { \sqrt { n } } { \varepsilon } )$ iteration complexity lower bound. Hence, the two bounds match in terms of the order of ε.

Linear convergence of Sinkhorn-Knopp. This line of work aims to establish linear convergence of SK. Some of the results provide global linear convergence rates: for example, [13] establishes a linear convergence rate $\frac { \sqrt { \kappa } - 1 } { \sqrt { \kappa } + 1 }$ based on Hilbert’s projective metric, where $\begin{array} { r } { \kappa = \operatorname* { m a x } _ { i , j , k , l } \frac { a _ { i k } a _ { j l } } { a _ { j k } a _ { i l } } } \end{array}$ . Recently, [35] establishes a linear convergence rate based on the second smallest eigenvalue of $\left( \begin{array} { c } { { P } } \\ { { A ^ { \top } } } \end{array} _ { Q } ^ { A } \right)$ . Another notable recent result is [18], where the authors define a density parameter based on the normalized version of the matrix and show linear convergence when this parameter is fixed. However, the above results are often conservative upper bounds on the behavior of SK, and some only apply to strictly positive matrices. In contrast, another line of research obtains tight contraction factors based on local arguments [25, 34]. These arguments treat SK as a fixed-point iteration and obtain the tight local linear convergence rate in (1) by analyzing the spectrum of the Jacobian. The local rate matches the true performance of SK, but such arguments are only asymptotic. This paper closes this gap by providing a nonasymptotic local analysis for SK for nonnegative matrices.

## 2 Sinkhorn-Knopp algorithm for matrix scaling

Notation. Throughout the paper, we use $\langle \cdot , \cdot \rangle$ to denote the Euclidean inner product and $\| \cdot \|$ to denote the Euclidean norm. We denote $\begin{array} { r } { \| A \| _ { 1 } : = \sum _ { i , j } | a _ { i j } | } \end{array}$ and $\begin{array} { r } { \left. A \right. _ { - \infty } : = \operatorname* { m i n } _ { | a _ { i j } | > 0 } \left| a _ { i j } \right| } \end{array}$ . Given a vector d $\in \mathbb { R } ^ { n }$ , we use ${ \mathcal { D } } ( d ) = \mathrm { d i a g } ( d )$ to denote a corresponding diagonal matrix with d on its diagonal. Notation ${ \bf 1 } _ { n }$ denotes the allone vector of dimension n. We will frequently use the notation $U = { \mathcal { D } } ( u )$ and $V = { \mathcal { D } } ( v )$ . Notation $\exp ( \cdot ) = \mathrm { e } ^ { ( \cdot ) }$ and log(·) will be applied element-wise to a vector or to the nonzeros of a matrix. i.e., $\exp ( U ) = { \mathcal { D } } ( \exp ( u ) )$ Given two vectors $a , b$ of the same dimension, we use $a / b = \mathcal { D } ( b ) ^ { - 1 } a$ to denote the element-wise division. Given a symmetric matrix $A , \lambda _ { k } ( A )$ denotes its k-th smallest eigenvalue. We often use $( a , b ) : = ( a )$ to denote the concatenation of column vectors when the context is clear. We denote $P = \mathcal { D } ( p ) , Q = \mathcal { D } ( q )$ and $S = \left( \begin{array} { c c } { { P } } \\ { { Q } } \end{array} \right)$

## 2.1 Matrix scaling and Sinkhorn-Knopp iteration

We formally define the matrix scaling problem: given a nonnegative matrix $A \in \mathbb { R } _ { + } ^ { m \times n }$ and two target margins $p \in \mathbb { R } _ { + + } ^ { m } , q \in \mathbb { R } _ { + + } ^ { n }$ such that $\langle \mathbf { 1 } _ { m } , p \rangle \ = \ \langle \mathbf { 1 } _ { n } , q \rangle$ , the matrix scaling problem looks for two diagonal scaling matrices $D _ { 1 } , D _ { 2 }$ such that two margin conditions $D _ { 1 } A D _ { 2 } \mathbf { 1 } _ { n } = p , D _ { 2 } A ^ { \top } D _ { 1 } \mathbf { 1 } _ { m } = q$ are approximately satisfied:

$$
\operatorname* { m a x } \{ \| D _ { 1 } A D _ { 2 } { \bf 1 } _ { n } - p \| , \| D _ { 2 } A ^ { \top } D _ { 1 } { \bf 1 } _ { m } - q \| \} \leq \varepsilon .\tag{2}
$$

Remark 1. In the recent literature for matrix scaling, the residual of the margin condition is often measured in $\ell _ { 1 } { \mathrm { - n o r m } }$ when $p , q$ are probability margins. Since our focus is linear convergence, we will adopt $\ell _ { 2 } { \mathrm { - e r r o r } }$ , and converting linear convergence to $\ell _ { 1 }$ error loses a log n term.

A pair $( D _ { 1 } , D _ { 2 } )$ satisfying (2) is called an ε-approximate scaling. Without loss of generality, we reparametrize $D _ { 1 } , D _ { 2 }$ by $D _ { 1 } = \mathrm { e } ^ { U }$ and $D _ { 2 } = \mathrm { e } ^ { - V }$ for real diagonal matrices $U , V$ . SK alternates between the two scaling matrices by fixing one and forcing the other to satisfy the margin condition:

$$
D _ { 1 } ^ { k + 1 } = { \mathcal { D } } ( p / ( A D _ { 2 } ^ { k } \mathbf { 1 } _ { n } ) ) \quad { \mathrm { a n d } } \quad D _ { 2 } ^ { k + 1 } = { \mathcal { D } } ( q / ( A ^ { \top } D _ { 1 } ^ { k + 1 } \mathbf { 1 } _ { m } ) ) .
$$

Under mild conditions [19], this iteration will converge to an optimal scaling pair $( D _ { 1 } ^ { \star } , D _ { 2 } ^ { \star } )$ . Without loss of generality, we assume that such a pair exists and is finite. i.e., the matrix A is scalable with respect to $p , q .$

A1: The nonnegative matrix A is scalable with respect to margin $p , q .$

Dual potential. A useful technique for analyzing the Sinkhorn iteration is through the potential function [19]

$$
\varphi ( u , v ) : = \langle \mathrm { e } ^ { u } , A \mathrm { e } ^ { - v } \rangle - \langle p , u \rangle + \langle q , v \rangle ,\tag{3}
$$

and SK can be written as performing alternating minimization (AM) on it:

$$
\boldsymbol { u } ^ { k + 1 } = \underset { \boldsymbol { u } } { \arg \operatorname* { m i n } } ~ \boldsymbol { \varphi } ( \boldsymbol { u } , \boldsymbol { v } ^ { k } ) , \quad \boldsymbol { v } ^ { k + 1 } = \underset { \boldsymbol { v } } { \arg \operatorname* { m i n } } ~ \boldsymbol { \varphi } ( \boldsymbol { u } ^ { k + 1 } , \boldsymbol { v } ) ,
$$

This paper adopts the AM perspective and will stick to Algorithm 1.

```julia
Algorithm 1: Sinkhorn-Knopp (AM on the potential function (3))
input (A, p, q), initial v<sup>1</sup> (or u<sup>1</sup>)
for k = 1, 2, . . . do
u<sup>k+1</sup> = arg min<sub>u</sub> φ(u, v<sup>k</sup>)
v<sup>k+1</sup> = arg min φ(u<sup>k+1</sup>, v)
end
```

## 3 Global and local convergence of Sinkhorn-Knopp

This section presents our main result on the convergence of the Sinkhorn algorithm.

## 3.1 Algorithm analysis

Global convergence. The local behavior of an algorithm typically follows a phase of global convergence. Define the solution norm diameter constant

$$
D : = \operatorname* { m i n } _ { ( u ^ { \star } , v ^ { \star } ) \in \arg \operatorname* { m i n } \varphi } \| ( u ^ { \star } , v ^ { \star } ) \| _ { \infty } .\tag{4}
$$

Under A1, there exists a finite scaling pair $( u ^ { \star } , v ^ { \star } )$ , and the constant $D < \infty$ can be explicitly bounded under additional assumptions:

Lemma 3.1 ([28, 24, 2]). Suppose A1 holds. Then

• Square doubly-stochastic. If $\begin{array} { r } { { \bf \Pi } ^ { \prime } p = q = \frac { 1 } { n } { \bf 1 } _ { n } } \end{array}$ , then $\begin{array} { r } { D \leq \frac { 1 } { 2 } | \log ( \frac { 1 } { n \| A \| _ { - \infty } } ) | + n \log ( \frac { \| A \| _ { 1 } } { n \| A \| _ { - \infty } } ) } \end{array}$

• Doubly-stochastic. $\begin{array} { r } { I f p = { \frac { 1 } { m } } \mathbf { 1 } _ { m } , q = { \frac { 1 } { n } } \mathbf { 1 } _ { n } } \end{array}$ , and that m $\leq n ,$ then $\begin{array} { r } { D \leq \frac { 1 } { 2 } | \log ( \frac { 1 } { n \| A \| _ { - \infty } } ) | + n ^ { 2 } \log ( \frac { \| A \| _ { 1 } } { n \| A \| _ { - \infty } } ) . } \end{array}$

• Positive matrix. $I f A > 0$ and $\langle \mathbf { 1 } _ { m } , p \rangle = \langle \mathbf { 1 } _ { n } , q \rangle = 1$ , then $D \leq \log ( { \frac { n } { \| A \| _ { - \infty } \| ( p , q ) \| _ { - \infty } ^ { 2 } } } )$

Given Lemma 3.1, an explicit convergence rate for the suboptimality of $\varphi$ is immediate.

Theorem 3.1. Suppose A1 holds and let $( u ^ { k } , v ^ { k } )$ be generated by Algorithm 1, then

$$
\begin{array} { r } { \varphi ( u ^ { K + 1 } , v ^ { K + 1 } ) - \varphi ( u ^ { \star } , v ^ { \star } ) \leq \frac { 2 \| p \| _ { 1 } D ^ { 2 } } { K } , } \end{array}
$$

where the diameter D is defined in (4).

Local convergence. Given Theorem 3.1, the function value gap will vanish and finally enter the local convergence regime. Our local analysis hinges on a simple relation, given by Proposition 3.1 below.

Proposition 3.1. Suppose A1 holds. Then we have the following identity:

$$
\begin{array} { r } { \varphi ( u , v ) = \varphi ( u ^ { \star } , v ^ { \star } ) + \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } a _ { i j } ^ { \star } \phi ( \Delta _ { i j } ) , } \end{array}
$$

where $\phi ( \delta ) : = \mathrm { e } ^ { \delta } - \delta - 1 , A ^ { \star }$ is the optimally scaled matrix, and $\Delta _ { i j } : = \left( u _ { i } - u _ { i } ^ { \star } \right) - \left( v _ { j } - v _ { j } ^ { \star } \right)$

Proposition 3.1 suggests we could analyze the potential function $\varphi$ through the scalar function

$$
\phi ( \delta ) = \mathrm { e } ^ { \delta } - \delta - 1 = \frac { \delta ^ { 2 } } { 2 } + \delta ^ { 3 } \int _ { 0 } ^ { 1 } \frac { 1 - t } { 2 } \mathrm { e } ^ { \delta t } \mathrm { d } t ,
$$

whose local behavior is dictated by $\frac { \delta ^ { 2 } } { 2 }$ and high-order remainder terms. In particular, it is convenient to bound the first and second-order information with zeroth-order information:

$$
{ \bf L e m m a 3 . 2 . } ~ L e l ~ \phi ( \delta ) = \mathrm { e } ^ { \delta } - \delta - 1 . ~ T h e n ~ \phi ^ { \prime } ( \delta ) - \delta = \phi ( \delta ) ~ a n d ~ | \phi ^ { \prime } ( \delta ) | = | \phi ^ { \prime \prime } ( \delta ) - 1 | \leq \sqrt { 2 \phi ( \delta ) } + \phi ( \delta ) .
$$

As a standard component of local analysis, we introduce the second-order expansion that governs the local behavior. Define $\begin{array} { r } { \lambda ( u , v ) : = \varphi ( u ^ { \star } , v ^ { \star } ) + \frac { 1 } { 2 } \| ( \Delta u , \Delta v ) \| _ { \nabla ^ { 2 } \varphi ( u ^ { \star } , v ^ { \star } ) } ^ { 2 } . } \end{array}$ . By definition, we have

$$
\begin{array} { r } { \boldsymbol { \lambda } ( \boldsymbol { u } , \boldsymbol { v } ) = \varphi ( \boldsymbol { u } ^ { \star } , \boldsymbol { v } ^ { \star } ) + \sum _ { i , j } \frac { a _ { i j } ^ { \star } } { 2 } \Delta _ { i j } ^ { 2 } , \quad \nabla \boldsymbol { \lambda } ( \boldsymbol { u } , \boldsymbol { v } ) = \nabla ^ { 2 } \varphi ( \boldsymbol { u } ^ { \star } , \boldsymbol { v } ^ { \star } ) ( \Delta \boldsymbol { u } , \Delta \boldsymbol { v } ) , \quad \mathrm { a n d } \quad \nabla ^ { 2 } \boldsymbol { \lambda } ( \boldsymbol { u } , \boldsymbol { v } ) = \nabla ^ { 2 } \varphi ( \boldsymbol { u } ^ { \star } , \boldsymbol { v } ^ { \star } ) } \end{array}
$$

Since we will mostly focus on single-step progress, from now on, we will resort to the notation

$$
( u , v ) \ \to \ ( u ^ { + } , v ) \ \to \ ( u ^ { + } , v ^ { + } )
$$

to denote one iteration of SK (or AM). A useful fact is that AM on $\lambda ( u , v )$ coincides with preconditioned gradient descent with preconditioner $P ^ { - 1 }$ and $Q ^ { - 1 }$ , whose contraction can be explicitly computed:

Lemma 3.3. Let $( u , v ) , ( u ^ { + } , v )$ and $( u ^ { + } , v ^ { + } )$ be consecutive iterations obtained by running AM on λ. Then

$$
\begin{array} { r } { \lambda ( u ^ { + } , v ^ { + } ) - \varphi ( u ^ { \star } , v ^ { \star } ) \le ( 1 - \sigma _ { 2 } ) [ \lambda ( u ^ { + } , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] \le ( 1 - \sigma _ { 2 } ) ^ { 2 } [ \lambda ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] , } \end{array}
$$

where $\sigma _ { 2 } = \lambda _ { 2 } ( I - P ^ { - 1 / 2 } A ^ { \star } Q ^ { - 1 } ( A ^ { \star } ) ^ { \top } P ^ { - 1 / 2 } ) \in ( 0 , 1 )$ is the connectivity of the normalized Laplacian.

The quantity $\sigma _ { 2 }$ already appeared in the SK literature [34, 25, 35], and it is obtained by treating SK as a fixedpoint iteration and linearizing its Jacobian. This quantity tightly characterizes the asymptotic behavior of the algorithm and is therefore a desirable measure of local convergence. Given that $\varphi$ behaves like λ in the local regime, and that AM applied to quadractic function λ produces a contraction of $1 - \sigma _ { 2 }$ , it remains to connect $\varphi$ to λ. With the help of Lemma 3.2, Lemma 3.4 below makes this connection explicit.

Lemma 3.4. Suppose A1 holds and let $\varepsilon = \varphi ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) > 0$ . Then we have

$\begin{array} { r } { 1 . \ \| \nabla \varphi ( u , v ) \| _ { S ^ { - 1 } } \leq 2 \sqrt { \varepsilon } + \frac { 2 } { \sqrt { s } } \varepsilon } \end{array}$ and that $\begin{array} { r } { \| \nabla \varphi ( u , v ) - \nabla \lambda ( u , v ) \| _ { S ^ { - 1 } } \leq \frac { 2 } { \sqrt { s } } \varepsilon } \end{array}$ , where $s : = \| ( p , q ) \| _ { - \infty } ,$

$\begin{array} { r } { \mathcal { Z } . \| S ^ { - 1 / 2 } [ \nabla ^ { 2 } \varphi ( u , v ) - \nabla ^ { 2 } \lambda ( u , v ) ] S ^ { - 1 / 2 } \| \le 4 \sqrt { \frac { \varepsilon } { s } } + \frac { 2 \varepsilon } { s } } \end{array}$ , and recall that $S = \left( \begin{array} { c r c } { { P } } \\ { { Q } } \end{array} \right)$

3. $\begin{array} { r } { i f \varepsilon \le \frac { s \sigma _ { 2 } ^ { 2 } } { 1 2 9 6 } } \end{array}$ , then $\begin{array} { r } { | \varphi ( u , v ) - \lambda ( u , v ) | \le \frac { 1 6 8 } { \sigma _ { 2 } } ( \frac { 1 } { \sqrt { s } } \varepsilon ^ { 3 / 2 } + \frac { 1 } { s } \varepsilon ^ { 2 } ) } \end{array}$

Using Lemma 3.4, we obtain a local recursion on the potential function gap.

Lemma 3.5. Suppose A1 holds. $\begin{array} { r } { I f \varphi ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) \leq \frac { s \sigma _ { 2 } ^ { 2 } } { 1 2 9 6 } } \end{array}$ with $s : = \| ( p , q ) \| _ { - \infty }$ , then

$$
\begin{array} { r } { \varphi ( u ^ { + } , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) \le ( 1 - \sigma _ { 2 } ) [ \varphi ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] + \frac { 3 3 6 } { \sqrt { \delta } \sigma _ { 2 } } [ \varphi ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] ^ { 3 / 2 } + \frac { 3 3 8 } { \delta \sigma _ { 2 } } [ \varphi ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] ^ { 2 } , } \end{array}
$$

and the same result holds for $\varphi ( u ^ { + } , v ^ { + } ) - \varphi ( u ^ { \star } , v ^ { \star } )$ with respect to $\varphi ( u ^ { + } , v ) - \varphi ( u ^ { \star } , v ^ { \star } )$

We are ready to present the nonasymptotic complexity of Sinkhorn that only depends on $( A , p , q )$ and $\sigma _ { 2 }$

Theorem 3.2. Suppose A1 holds and $\varepsilon \in ( 0 , 1 6 \| p \| _ { 1 } ]$ . Then SK finds an ε-approximate scaling in

$$
\begin{array} { r } { K _ { \varepsilon } = \left\lceil \frac { 2 \cdot 1 3 4 \cdot 4 ^ { 2 } \| p \| _ { 1 } D ^ { 2 } } { \sigma _ { 2 } ^ { 4 } } + \frac { 1 } { \sigma _ { 2 } } \log ( 1 2 8 [ \| A \| _ { 1 } \| p \| _ { 1 } + \| p \| _ { 1 } ^ { 2 } ] D ) + \frac { 2 } { \sigma _ { 2 } } \log ( \frac { 1 } { \varepsilon } ) \right\rceil = \mathcal { O } \bigl ( \frac { \| p \| _ { 1 } D ^ { 2 } } { \operatorname* { m i n } \{ \sigma _ { 2 } ^ { 4 } , s \sigma _ { 2 } ^ { 2 } \} } + \frac { 1 } { \sigma _ { 2 } } \log ( \frac { 1 } { \varepsilon } ) \bigr ) } \end{array}
$$

iterations, where solution norm bound D is defined in (4), $\textstyle \| A \| _ { 1 } = \sum _ { i , j } a _ { i j }$ , and $s = \| ( p , q ) \| _ { - \infty }$

## 3.2 Implications and extensions

Tightness. Since $\sigma _ { 2 }$ has been shown to match the rate obtained by linearizing the Jacobian, it is easy to find instances where our analysis matches the asymptotic behavior of the algorithm.

Proposition 3.2 (Informal). There exists $( a f a m i l y o f )$ matrix scaling instances $\{ ( A , p , q ) \}$ such that $\varphi ( u ^ { k } , v ^ { k } ) -$ $\begin{array} { r } { \varphi ( u ^ { \star } , v ^ { \star } ) \geq \varepsilon ~ f o r ~ k = \mathcal { O } ( c + \frac { 1 } { \sigma _ { \circ } } \log ( \frac { 1 } { \varepsilon } ) ) } \end{array}$ , where $c \geq 0$ does not depend on ε.

Polynomiality of SK. For doubly stochastic matrix scaling, our analysis shows that SK is a polynomial-time algorithm if $\sigma _ { 2 } > 0$ is considered a fixed constant.

Corollary 3.1. Let $\begin{array} { r } { p = q = \frac { 1 } { n } \mathbf { 1 } _ { n } } \end{array}$ (case 1 in Lemma 3.1) and suppose $\sigma _ { 2 } > 0$ is a fixed constant. Then SK finds an ε-approximate scaling in $\begin{array} { r } { \mathcal { O } ( n ^ { 3 } + \log ( \frac { n } { \varepsilon } ) ) } \end{array}$ iterations. If $A > 0$ is positive (case 3 in Lemma 3.1), then the complexity is further improved to $\begin{array} { r } { \mathcal { O } ( n + \log ( \frac { n } { \varepsilon } ) ) } \end{array}$

Remark 2. The result exhibits an undesired sharp complexity transition between ${ \mathcal { O } } ( n )$ for positive matrices and $\mathcal { O } ( n ^ { 3 } )$ for general nonnegative matrices. This transition comes from the norm bound D in Lemma 3.1 and is likely an artifact of analysis. A smoothed measure of sparsity may overcome this weakness (e.g. [17, 18]).

Matrix balancing and equilibration. Given a real matrix $A \in \mathbb { R } ^ { m \times n }$ , the matrix $\ell _ { \omega }$ matrix equilibration problem finds positive diagonal matrices $D _ { 1 } , D _ { 2 }$ such that

$$
| D _ { 1 } A D _ { 2 } | ^ { \omega } { \bf 1 } _ { m } \approx p \quad \mathrm { a n d } \quad | D _ { 1 } A ^ { \top } D _ { 2 } | ^ { \omega } { \bf 1 } _ { n } \approx q ,
$$

where $| A |$ denotes element-wise absolute value and $\omega \in ( 0 , \infty )$ . This problem [12, 10] can be reduced to matrix scaling with data $( | A | ^ { \omega } , p , q )$ , and our analysis follows immediately.

Corollary 3.2. For $\omega \in ( 0 , 1 )$ , Algorithm 1 applied to $( | A | ^ { \omega } , p , q )$ finds an ε-approximate $\ell _ { \omega }$ -equilibration of A in $\begin{array} { r } { \mathcal { O } ( \frac { D ^ { 2 } } { \operatorname* { m i n } \{ \sigma _ { 2 } ^ { 4 } , s \sigma _ { 2 } ^ { 2 } \} } + \frac { 1 } { \sigma _ { 2 } } \log ( \frac { 1 } { \varepsilon } ) ) } \end{array}$ iterations, where $\sigma _ { 2 } = \lambda _ { 2 } ( I - P ^ { - 1 / 2 } | A ^ { \star } | ^ { \omega } Q ^ { - 1 } ( | A ^ { \star } | ^ { \omega } ) ^ { \top } P ^ { - 1 / 2 } )$

Finally, since SK can also be used for matrix balancing [25], our result are similarly applicable.

## 4 Nonasymptotic acceleration of matrix scaling

This section develops accelerated variants of SK. Given the tightness result Proposition 3.2, acceleration is generally unachievable without modifying the algorithm. Hence, we resort to the semi-dual formulation [8].

## 4.1 Semi-dual function

Since SK performs AM on $\varphi ( u , v )$ , it is natural to consider the partially minimized objective

$$
\begin{array} { r } { \zeta ( u ) : = \operatorname* { m i n } _ { \varphi } \varphi ( u , v ) = \varphi ( u , - \log ( q / A ^ { \top } \mathrm { e } ^ { u } ) ) = \sum _ { j = 1 } ^ { n } q _ { j } \log \langle a _ { [ : , j ] } , \mathrm { e } ^ { u } \rangle - \langle p , u \rangle - \sum _ { j = 1 } ^ { n } q _ { j } \log q _ { j } + \langle \mathbf { 1 } _ { n } , q \rangle , } \end{array}
$$

where $a _ { [ : , j ] }$ is the j-th column of A. Function ζ is known as the semi-dual in the literature [8]. Its gradient can be computed by a half SK iteration: define nonlinear map $v ( u ) : = - \log ( q / A ^ { \top } \mathrm { e } ^ { u } )$ and its Jacobian $\mathcal { I } _ { v }$ . We have

$$
\nabla \zeta ( u ) = \nabla _ { u } \varphi ( u , v ( u ) ) + \nabla _ { v } \varphi ( u , v ( u ) ) \mathcal { I } _ { v } ( u ) = \nabla _ { u } \varphi ( u , v ( u ) ) ,
$$

since optimality condition implies $\nabla _ { v } \varphi ( u , v ( u ) ) = 0$ . Therefore, minimizing residual $\| \nabla \varphi ( u , v ) \|$ reduces to reducing the gradient norm $\lVert \nabla \zeta ( u ) \rVert$ . Lemma 4.1 shows that $\zeta$ has desirable properties.

Lemma 4.1. The semi-dual function $\zeta$ satisfies the following properties

1. It is convex, with $L = { \frac { 1 } { 2 } } \| p \| _ { 1 }$ -Lipschitz gradient and $\begin{array} { r } { H = \frac { 3 } { 2 } \| p \| _ { 1 } – L i p s c h i t z \ H e s s i a n } \end{array}$ .

2. $\begin{array} { r } { I f \zeta ( u ) - \zeta ( u ^ { \star } ) \leq \frac { s \sigma _ { 2 } ^ { 2 } } { 1 2 9 6 } } \end{array}$ , then $\begin{array} { r } { \zeta ( u ) - \zeta ( u ^ { \star } ) \geq \frac { \sigma _ { 2 } } { 4 } \| \Pi ( u - u ^ { \star } ) \| _ { P } ^ { 2 } } \end{array}$ and

$$
\begin{array} { r } { \frac { 4 } { \sigma _ { 2 } } \| \nabla \zeta ( u ) \| _ { P ^ { - 1 } } ^ { 2 } \geq \zeta ( u ) - \zeta ( u ^ { \star } ) \geq \frac { 1 } { 1 6 } \| \nabla \zeta ( u ) \| _ { P ^ { - 1 } } ^ { 2 } . } \end{array}
$$

3. $\begin{array} { r } { I f \zeta ( u ) - \zeta ( u ^ { \star } ) \leq \operatorname* { m i n } \{ \frac { s \sigma _ { 2 } ^ { 2 } } { 1 2 9 6 } , \frac { \sigma _ { 2 } ^ { 3 } s ^ { 2 } } { 2 4 \| p \| _ { 1 } } \} } \end{array}$ , then $\begin{array} { r } { \frac { \sigma _ { 2 } } { 2 } \leq \lambda _ { 2 } ( P ^ { - 1 / 2 } \nabla ^ { 2 } \zeta ( u ) P ^ { - 1 / 2 } ) } \end{array}$ and $\lambda _ { m } ( P ^ { - 1 / 2 } \nabla ^ { 2 } \zeta ( u ) P ^ { - 1 / 2 } ) \leq 2$

where $\begin{array} { r } { \Pi = I - \frac { 1 } { \| p \| ^ { 2 } } p p ^ { \top } } \end{array}$ is the orthogonal projection onto $p ^ { \perp }$

Remark 3. To our knowledge, the global and local smoothness (case 1 of Lemma 4.1) of the semi-dual function has been established and leveraged in the literature for optimal transport [8, 16, 40]. However, the explicit local PL constants (case 2 and 3 of Lemma 4.1) are not yet available.

Given that ζ is a smooth, convex function with a finite-sum structure, there are several techniques from optimization theory for making its gradient small.

## 4.2 Global acceleration

Finding an ε-approximate scaling corresponds to making $\| \nabla \zeta ( u ) \| \ \leq \varepsilon$ . For an L-smooth convex function, Nesterov’s regularization technique achieves this goal with an iteration complexity of $\begin{array} { r } { \mathcal { O } ( \sqrt { \frac { L \| u ^ { \star } \| } { \varepsilon } } \log ( \frac { L \| u ^ { \star } \| } { \varepsilon } ) ) } \end{array}$ [32], where the log factor can be removed using the recently developed performance estimation techniques [26].

Theorem 4.1. Under the same assumptions as Theorem 3.2, there exists an algorithm $\mathcal { A } _ { 1 }$ that outputs an ε-approximate scaling in

$$
\begin{array} { r } { K _ { \varepsilon } = \left\lceil \frac { 8 \sqrt { \| p \| _ { 1 } \| u ^ { \star } \| } } { \sqrt { \varepsilon } } \right\rceil \leq \left\lceil \frac { 8 n ^ { 1 / 4 } \| p \| _ { 1 } ^ { 1 / 2 } \sqrt { D } } { \sqrt { \varepsilon } } \right\rceil } \end{array}
$$

iterations, where each iteration has the same cost as SK.

Following [2], assuming $D = \mathcal { O } ( 1 )$ , the arithmetic complexity of Theorem 4.1 is $\mathcal { O } ( \frac { \mathrm { n n z } ( A ) } { \sqrt { \varepsilon } } n ^ { 1 / 4 } \| p \| _ { 1 } ^ { 1 / 2 } \sqrt { D } )$ which improves on the $\mathcal { O } ( \frac { \mathrm { n n z } ( A ) } { \varepsilon ^ { 2 / 3 } } \| p \| _ { 1 } ^ { 1 / 3 } D ^ { 2 / 3 } )$ result from [2] in terms of ε.

Finally, given the finite-sum structure of ζ, applying stochastic gradient descent with variance reduction $( \mathrm { e . g . }$ Katyusha [1]) further reduces the complexity to $\begin{array} { r } { \tilde { \mathcal { O } } ( \frac { \mathrm { n n z } ( A ) } { \sqrt { \varepsilon } } n ^ { 1 / 4 } \| p \| _ { \infty } ^ { 1 / 2 } \sqrt { D } ) } \end{array}$ in achieving $\mathbb { E } [ \| \nabla \zeta ( u ) \| ] \le \varepsilon$

Theorem 4.2. Under the same assumptions as Theorem 3.2, there exists an algorithm $\mathcal { A } _ { 2 }$ that outputs uˆ such that $\mathbb { E } [ \| \nabla \zeta ( \hat { u } ) \| ] \le \varepsilon$ with arithmetic complexity $\begin{array} { r } { \tilde { \mathcal { O } } ( \frac { \mathrm { n n z } ( A ) } { \sqrt { \varepsilon } } n ^ { 1 / 4 } \| p \| _ { \infty } ^ { 1 / 2 } \sqrt { D } ) } \end{array}$ in expectation.

Remark 4. For $\begin{array} { r } { p = q = \mathbf { 1 } _ { n } , \tilde { \mathcal { O } } ( \frac { \mathrm { n n z } ( A ) } { \sqrt { \varepsilon } } n ^ { 1 / 4 } \sqrt { D } ) } \end{array}$ is better than $\mathcal { O } ( \frac { \mathrm { n n z } ( A ) } { \varepsilon ^ { 2 / 3 } } n ^ { 1 / 3 } D ^ { 2 / 3 } )$ in expectation.

Remark 5. Specializing to entropically-regularized optimal transport with smoothing parameter η and target accuracy εˆ, we have [28] $\mathrm { { n z } } ( A ) \ : = \ : n ^ { 2 } , D \ : = \ : \mathcal { O } ( \textstyle { \frac { \| C \| _ { \infty } } { \eta } } ) , \eta \ : = \ : \Theta ( \varepsilon )$ and $\| p \| _ { \infty } = n ^ { - 1 }$ . Taking SK accuracy $\varepsilon \gets n ^ { - 1 / 2 } \hat { \varepsilon }$ ensures $\| \nabla \zeta ( \hat { u } ) \| _ { 1 } \leq \sqrt { n } \| \nabla \zeta ( \hat { u } ) \| \leq \hat { \varepsilon } .$ . It results in an

$$
\begin{array} { r } { \tilde { \mathcal { O } } \Big ( \frac { n ^ { 2 } \cdot n ^ { 1 / 4 } } { \sqrt { \hat { \varepsilon } } n ^ { 1 / 4 } } \sqrt { \frac { \| C \| _ { \infty } } { \hat { \varepsilon } } } \Big ) = \tilde { \mathcal { O } } \big ( \frac { n ^ { 2 } \sqrt { \| C \| _ { \infty } } } { \hat { \varepsilon } } \big ) } \end{array}
$$

complexity for entropy-regularized optimal transport, with a better dependence on $\| C \| _ { \infty }$ than the previous $O ( \frac { n ^ { 2 } \| C \| _ { \infty } } { \hat { \varepsilon } } )$ result [20, 5]. This improvement comes at the cost of randomness in the algorithm.

While global acceleration is interesting from a worst-case perspective, it is local behavior that determines how quickly the algorithm reaches a high-accuracy solution. The rest of this section explores the local behavior of SK and introduces two acceleration mechanisms based on the semi-dual formulation: one adopts Nesterov’s acceleration and improves the complexity from $\begin{array} { r } { \mathcal { O } \big ( \frac { 1 } { \sigma _ { 2 } } \log \bigl ( \frac { 1 } { \varepsilon } \bigr ) \big ) } \end{array}$ to $\begin{array} { r } { \mathcal { O } ( \frac { 1 } { \sqrt { \sigma _ { 2 } } } \log ( \frac { 1 } { \varepsilon } ) ) } \end{array}$ ; the other notices that locally, SK corresponds to a suboptimal diagonal preconditioner. A better preconditioner also improves the local contraction.

## 4.3 Local suboptimality of Sinkhorn-Knopp

Consider a step of SK: $( u , v )  ( u ^ { + } , v )$ with $\nabla _ { v } \varphi ( u , v ) = 0$ . By our previous analysis, locally, we have

$$
u ^ { + } = u - P ^ { - 1 } \nabla _ { u } \varphi ( u , v ) = u - P ^ { - 1 } \nabla \zeta ( u ) .\tag{5}
$$

Therefore, the local convergence rate of SK can be reproduced by running preconditioned gradient descent on ζ with preconditioner $P ^ { - 1 } \stackrel { \textstyle - } { ( } \mathrm { o r } \ Q ^ { - 1 }$ if v is kept in the semi-dual formulation). Since a preconditioner can be viewed as a matrix stepsize, a natural question is whether this stepsize is optimal. To simplify the notation, for now we assume $p = q = { \mathbf { 1 } } _ { m }$ and $P = I ,$ , and (5) simplifies to vanilla gradient descent with stepsize 1. With this simplification, the local linear rate of preconditioned gradient descent is dictated by the ratio between smoothness and strong convexity over ${ \mathbf { 1 } } _ { m } ^ { \perp } ,$ the orthogonal complement of span $\left\{ \mathbf { 1 } _ { m } \right\}$

$$
\sigma _ { 2 } I \preceq _ { \Pi } \nabla ^ { 2 } \zeta ( u ^ { \star } ) = I - R R ^ { \top } \preceq _ { \Pi } \sigma _ { m } I ,
$$

where $\preceq _ { \Pi }$ denotes the semidefinite order over $\mathbf { 1 } _ { m } ^ { \perp }$ and $\sigma _ { m } : = \lambda _ { m } ( I - R R ^ { \intercal } ) \leq 1$ . The efective local condition number is $\begin{array} { r } { \frac { \sigma _ { m } } { \sigma _ { 2 } } \leq \frac { 1 } { \sigma _ { 2 } } } \end{array}$ , and the $( 1 - \sigma _ { 2 } ) ^ { 2 }$ contraction of SK (two half-iterations) matches this bound. However, this perspective also shows that SK is locally suboptimal: for an L-smooth µ-strongly convex problem, the minimax optimal stepsize $\begin{array} { r } { [ 3 7 ] \frac { 2 } { L + \mu } > \frac { 1 } { L } } \end{array}$ produces a better contraction

$$
\begin{array} { r } { ( \frac { L - \mu } { L + \mu } ) ^ { 2 } = ( 1 - \frac { 2 } { \kappa + 1 } ) ^ { 2 } < ( 1 - \frac { 1 } { \kappa } ) ^ { 2 } . } \end{array}
$$

Plugging in $L = 1$ and $\mu = \sigma _ { 2 }$ , we see that the stepsize $\displaystyle \frac { 2 } { \sigma _ { 2 } + 1 }$ provides a local contraction of $\begin{array} { r } { \big ( \frac { 1 - \sigma _ { 2 } } { 1 + \sigma _ { 2 } } \big ) ^ { 2 } < ( 1 - \sigma _ { 2 } ) ^ { 2 } } \end{array}$ If $\sigma _ { 2 }  1$ , this stepsize improves on the contraction of SK by nearly a factor of 4. Furthermore, we have been using $L = 1$ , an upper bound of $\sigma _ { m } ;$ in contrast, the true condition number $\frac { \sigma _ { m } } { \sigma _ { 2 } }$ can be smaller than $\frac { 1 } { \sigma _ { 2 } }$ , making this change even more impactful. Indeed, when R has full row rank, $R R ^ { \top }$ is positive definite, and

$$
\sigma _ { m } = \lambda _ { m } ( I - R R ^ { \top } ) < 1 .
$$

Together with the previous analysis, the stepsize $\frac { 2 } { \sigma _ { 2 } + \sigma _ { m } }$ further improves the contraction to $\begin{array} { r } { \big ( \frac { \sigma _ { m } - \sigma _ { 2 } } { \sigma _ { m } + \sigma _ { 2 } } \big ) ^ { 2 } < ( 1 - \sigma _ { 2 } ) ^ { 2 } } \end{array}$ Remark 6. Our analysis reveals an asymmetry between u and v blocks: to ensure R is full row rank, it is necessary that $m \leq n$ . In other words, $\sigma _ { m } < 1$ happens only if one uses the semi-dual objective on the block with a smaller dimension. We are not aware of this viewpoint being exploited in existing SK analyses.

As a summary, the local perspective reveals the intrinsic suboptimality of SK. This observation also carries over to the local behavior of other two-block alternating minimization algorithms. Using a slightly larger stepsize provably improves the local contraction. However, there is one challenge left: the above stepsizes rely on unknown information such as $\sigma _ { 2 }$ and $\sigma _ { m }$ . It is computationally expensive to obtain accurate estimates of these quantities. To use this strategy, we need an algorithm that automatically finds the locally best stepsize. This goal is achieved by the online scaled gradient method (OSGM) [15].

## 4.4 Local online acceleration

Consider optimizing ζ with gradient descent preconditioned (scaled) by a diagonal matrix $\mathcal { D } ( w )$ (vector w)

$$
\boldsymbol { u } ^ { + } = \boldsymbol { u } - \mathcal { D } ( \boldsymbol { w } ) \nabla \zeta ( \boldsymbol { u } ) .\tag{6}
$$

The descent lemma in P-norm motivates the definition of relative progress of the preconditioner vector w with respect to the scaled gradient norm $\begin{array} { r } { \| \nabla \zeta ( u ) \| _ { P ^ { - 1 } } ^ { 2 } \colon h _ { u } ( w ) \ : = \ \frac { \zeta ( u ^ { - } \mathcal { D } ( \overline { { w } } ) \nabla \zeta ( u ) ) - \zeta ( u \overline { { ) } } } { \| \nabla \zeta ( u ) \| _ { p ^ { - 1 } } ^ { 2 } } } \end{array}$ . The function $h _ { u }$ inherits properties like convexity and smoothness from ζ.

## Lemma 4.2. The function $h _ { u }$ has the following properties

1. It is convex and $L = { \frac { 1 } { 2 } } \| p \| _ { 1 }$ <sub>1</sub>-smooth in $P ^ { - 1 }$ -norm,

2. $h _ { u } ( P ^ { - 1 } \mathbf { 1 } _ { m } ) \leq - \frac 1 4$ for all u such that $\begin{array} { r } { \zeta ( u ) - \zeta ( u ^ { \star } ) \leq \frac { \sigma _ { 2 } ^ { 2 } s } { 1 2 9 6 } } \end{array}$

Algorithm 2: Matrix scaling with online scaling (OSMS)   
input $( A , p , q )$ with $\begin{array} { r } { \overline { { L = \frac { 1 } { 2 } \| p \| _ { 1 } , H = \frac { 3 } { 2 } \| p \| _ { 1 } } } } \end{array}$ (as defined in Lemma 4.1), initial $u ^ { 1 }$ and $w ^ { 1 }$   
for $k = 1 , 2 , \dots$ . do   
$\begin{array} { r } { w ^ { k + 1 } = w ^ { k } - \frac { 1 } { L } P ^ { - 1 } \nabla h _ { u ^ { k } } ( w ^ { k } ) } \end{array}$   
$u ^ { k + 1 / 2 } = u ^ { k } - \overline { { \mathcal { D } ( w ^ { k + 1 } ) \nabla \zeta ( u ^ { k } ) } }$   
Choose $u ^ { k + 1 }$ that yields smaller potential value ζ between $u ^ { k + 1 / 2 }$ and $u ^ { k } .$   
end

Online acceleration (Algorithm 2) alternates between 1) (hyper-)gradient descent on w (with gradient $\nabla h _ { u } ( w ) )$ to learn wˆ and 2) preconditioned gradient update (6) on u (with gradient $\nabla \zeta ( u ) )$ . Given any preconditioner

vector $\hat { w } ,$ define the learning potential

$$
\begin{array} { r } { \Omega _ { \hat { w } } ( u , w ) : = \log ( \zeta ( u ) - \zeta ( u ^ { \star } ) ) + \frac { L \sigma _ { 2 } } { 8 } \| w ^ { + } - \hat { w } \| _ { P } ^ { 2 } . } \end{array}
$$

The following result characterizes the local behavior of Algorithm 2.

Lemma 4.3. Under the same assumptions as Theorem 3.2, for any fixed scaling vector $\hat { w } \in \mathbb { R } ^ { m }$ , we have $\begin{array} { r } { \Omega _ { \hat { w } } ( u ^ { + } , w ^ { + } ) \leq \Omega _ { \hat { w } } ( u , w ) + \frac { \sigma _ { 2 } } { 4 } h _ { u } ( \hat { w } ) } \end{array}$ for all u such that $\begin{array} { r } { \zeta ( u ) - \zeta ( u ^ { \star } ) \leq \frac { s \sigma _ { 2 } ^ { 2 } } { 1 2 9 6 } } \end{array}$

Telescoping Lemma 4.3 implies the relation $\begin{array} { r } { \Omega _ { \hat { w } } ( u ^ { K + 1 } , w ^ { K + 1 } ) \leq \Omega _ { \hat { w } } ( u ^ { 1 } , w ^ { 1 } ) + \frac { \sigma _ { 2 } } { 4 } \sum _ { k = 1 } ^ { K } h _ { u ^ { k } } ( \hat { w } ) } \end{array}$ holds for any $\hat { w } .$ Given the freedom to choose $\hat { w } .$ , taking wˆ = P<sup>−1</sup>1 gives $h _ { u } ( P ^ { - 1 } \mathbf { 1 } _ { m } ) \leq - \frac 1 4$ for all local u (by Lemma 4.2), which gives $\Omega _ { \hat { w } } ( u ^ { K + 1 } , w ^ { K + 1 } ) \leq \Omega ( u ^ { 1 } , w ^ { \bar { 1 } } ) - { \textstyle \frac { \sigma _ { 2 } } { 1 6 } } K$ and a final complexity of

$$
\begin{array} { r } { K _ { \varepsilon } : = \left\lceil \frac { 1 6 } { \sigma _ { 2 } } \frac { L \sigma _ { 2 } } { 8 } \| w ^ { 1 } - P ^ { - 1 } \mathbf { 1 } _ { m } \| _ { P } ^ { 2 } + \frac { 1 6 } { \sigma _ { 2 } } \log ( \frac { 1 } { \varepsilon } ) \right\rceil = \left\lceil 2 L \| w ^ { 1 } - P ^ { - 1 } \mathbf { 1 } _ { m } \| _ { P } ^ { 2 } + \frac { 1 6 } { \sigma _ { 2 } } \log ( \frac { 1 } { \varepsilon } ) \right\rceil . } \end{array}
$$

Alternatively, we can take $\hat { w } = w ^ { \star }$ that minimizes $\textstyle \sum _ { k = 1 } ^ { K } h _ { u ^ { k } } ( \hat { w } )$ . Define $\begin{array} { r } { \sigma _ { 2 } ^ { \star } : = - \frac { \sigma _ { 2 } } { 4 } \operatorname* { m a x } _ { u } h _ { u } ( w ^ { \star } ) } \end{array}$ . Then $\Omega _ { w ^ { \star } } ( u ^ { K + 1 } , w ^ { K + 1 } ) \leq \Omega _ { w ^ { \star } } ( u ^ { 1 } , w ^ { 1 } ) - \sigma _ { 2 } ^ { \star } K$ implies a complexity of

$$
\begin{array} { r } { K _ { \varepsilon } : = \lceil \frac { L \sigma _ { 2 } } { 8 \sigma _ { 2 } ^ { \star } } \rceil | w ^ { 1 } - w ^ { \star } | \| _ { P } ^ { 2 } + \frac { 1 } { \sigma _ { 2 } ^ { \star } } \log ( \frac { 1 } { \varepsilon } ) \rceil , } \end{array}
$$

which can be faster than $\begin{array} { r } { \mathcal { O } ( \frac { 1 } { \sigma _ { 2 } } \log ( \frac { 1 } { \varepsilon } ) ) } \end{array}$ of SK if $\sigma _ { 2 } < \sigma _ { 2 } ^ { \star }$ . Finally, the above local arguments are automatically globalized since OSGM has an $\mathcal { O } ( \textstyle \frac { D ^ { 2 } } { K } )$ global convergence rate on smooth convex functions [15].

Theorem 4.3 (Informal). Algorithm 2 outputs an ε-approximate scaling in $\begin{array} { r } { \mathcal { O } \big ( \frac { 1 } { \sigma _ { 2 } ^ { \star } } \log \bigl ( \frac { 1 } { \varepsilon } \bigr ) \bigr ) } \end{array}$ iterations.

We see that while online acceleration can speed up matrix scaling, its efect is problem-dependent. On the other hand, extrapolation-based schemes often yield a strict acceleration efect. The next section explores Nesterov’s accelerated gradient method to accelerate matrix scaling locally.

## 4.5 Local Nesterov’s acceleration

A few attempts have been made in the literature to locally accelerate SK, mostly by interpreting it as fixed-point iteration and employing overrelaxation [27, 38, 33]. However, existing arguments based on fixed-point iteration are asymptotic. In this section, we instead make use of the preconditioned gradient descent perspective, whose accelerated variant is preconditioned accelerated gradient descent (Algorithm 3, PAGD).

Algorithm 3: Matrix scaling with accelerated preconditioned gradient descent (PAGD)   
input $( A , p , q )$ , initial point $\overline { { u ^ { 1 } = w ^ { 1 } } }$   
for $k = 1 , 2 , \ldots$ do   
$\begin{array} { r } { y ^ { k } = u ^ { k } + \frac { \sqrt { \sigma _ { 2 } } } { \sqrt { \sigma _ { 2 } } + 2 } ( w ^ { k } - u ^ { k } ) } \end{array}$   
$u ^ { k + 1 / 2 } = y ^ { \check { k } } - \textstyle { \frac { 1 } { 2 } } P ^ { - 1 } \nabla \zeta ( y ^ { k } )$   
$\begin{array} { r } { w ^ { k + 1 } = ( \bar { 1 } - \frac { 1 } { 2 } \sqrt [ ] { \sigma _ { 2 } } ) w ^ { k } + \frac { 1 } { 2 } \sqrt { \sigma _ { 2 } } ( y ^ { k } - \frac { 4 } { \sigma _ { 2 } } P ^ { - 1 } \nabla \zeta ( y ^ { k } ) ) } \end{array}$   
Let $u ^ { k + 1 }$ be the better between $u ^ { k + 1 / \bar { 2 } }$ and $u ^ { k }$   
end

Our analysis adopts the following potential function [9]

$$
\begin{array} { r } { f ( u , w ) : = \zeta ( u ) + \frac { \sigma _ { 2 } } { 4 } \| \Pi ( w - u ^ { \star } ) \| _ { P } ^ { 2 } , } \end{array}
$$

where recall that $\begin{array} { r } { \Pi = I - \frac { 1 } { \| p \| ^ { 2 } } p p ^ { \top } } \end{array}$ . By Lemma 4.1, ζ is 2-smooth and $\frac { \sigma _ { 2 } } { 2 }$ -strongly convex over $\mathbf { 1 } _ { m } ^ { \perp }$ in P-norm.   
Then, a standard potential function-based analysis [9] establishes the desired accelerated rate.

Lemma 4.4. Let $( u , w ) , ( u ^ { + } , w ^ { + } )$ be consecutive iterates generated by Algorithm 3. $I f \ \zeta ( u ^ { 1 } ) - \zeta ( u ^ { \star } ) \ \leq$ min $\{ \frac { \sigma _ { 2 } ^ { 2 } s } { 1 2 9 6 } , \frac { \sigma _ { 2 } ^ { 3 } s ^ { 2 } } { 2 4 \| p \| _ { 1 } } \} \cdot \operatorname* { m i n } \{ 4 \| p \| _ { 1 } ^ { - 1 } , 1 \}$ , then $\begin{array} { r } { f ( u ^ { + } , w ^ { + } ) - \zeta ( u ^ { \star } ) \leq ( 1 - \frac { 1 } { 2 } \sqrt { \sigma _ { 2 } } ) [ f ( u , w ) - \zeta ( u ^ { \star } ) ] } \end{array}$

Theorem 4.4. Suppose $\textstyle u ^ { 1 } = w ^ { 1 }$ satisfies the condition from Lemma 4.4, then Algorithm 3 outputs an ε-approximate scaling in $\begin{array} { r } { \mathcal { O } ( \frac { 2 } { \sqrt { \sigma _ { 2 } } } \log ( \frac { 1 } { \varepsilon } ) ) } \end{array}$ iterations.

## 5 Numerical experiments

This section conducts numerical experiments to validate our findings.

Experiment setup. We compare SK, gradient descent (GD) on ζ, OSMS (Algorithm 2), and PAGD (Algorithm 3). Both OSMS and PAGD warm-start with SK until $\lVert \mathrm { e } ^ { U } A \mathrm { e } ^ { - v } - p \rVert \leq 1 0 ^ { - 3 }$ ; OSMS initializes its preconditioner at $P _ { 1 } = P ^ { - 1 }$ (so its first step is a SK step).

Dataset. We use entropy-regularized optimal transport instances with Gibbs kernel $A = \mathrm { e } ^ { - C / \eta } , \eta = 2 \times 1 0 ^ { - 3 }$ where C is obtained by 1). random: support points uniform in $[ 0 , 1 ] ^ { 2 }$ , C the squared Euclidean cost, and $p , q \sim \mathcal { U } [ 0 , 1 ] \ 2 )$ . MNIST: $p , q$ are normalized digit images with $m = n = 7 8 4$

## 5.1 Local suboptimality of Sinkhorn-Knopp

This section validates our finding that SK is locally suboptimal (using fixed stepsize 1.0). In Figure 1, we compare the performance of SK to GD with two fixed stepsizes $\frac { 2 } { 1 + \sigma _ { 2 } }$ and $\frac { 2 } { \sigma _ { 2 } + \sigma _ { m } }$ from Section 4.3.

iteration (= one Sinkhorn step)  
![](images/ffde3d6f82b6f511cac3b4a9169801d20032bc64dbfeb597e86d229abb3dfa2a.jpg)

![](images/d8250993247e7e716575c1e9617f77ce29b7f914e4f3198d43caaefe0e835aac.jpg)

![](images/95450fd0c4961ad6259d534664c2fde49295dcccf7247cadcec482916f2a18e5.jpg)

![](images/a8a2810655a33377a64a50ec8e46726fee03affb6f8d67b2c54e6955e350c5d4.jpg)  
Figure 1: Local suboptimality of SK $( m = 3 , n = 3 0 0 )$ : the larger minimax $( \alpha = 2 / ( 1 + \sigma _ { 2 } ) )$ and optimal $( \alpha = 2 / ( \sigma _ { 2 } + \sigma _ { m } ) )$ stepsizes contract faster than $\mathtt { S K } \left( \alpha = 1 \right)$ 1

As our theory predicts, using a larger step size yields better local contraction than SK.

## 5.2 Accelerated variants of Sinkhorn-Knopp

Figures 2 and 3 compare the accelerated variants on eight random and eight MNIST instances. Both PAGD and OSMS beat SK by several orders of magnitude within the budget.

## 6 Conclusion

This paper investigates the nonasymptotic local convergence of SK and provides new insights into its behavior. We also develop nonasymptotic accelerated variants through the semi-dual formulation. Our techniques extend to related problems, such as unbalanced optimal transport [16], and our analysis template applies to general two-block alternating minimization algorithms.

![](images/67eb2803ce9b77d080fedb589f649f1972340b0fd85b5de54345162668877ed4.jpg)  
Figure 2: Accelerated variants of SK on eight random entropic optimal transport instances $( m = n = 2 0 0 )$ : residual versus number of matrix-vector products.

## References

[1] Zeyuan Allen-Zhu. Katyusha: The first direct acceleration of stochastic gradient methods. Journal of Machine Learning Research, 18(221):1–51, 2018. (cited on 7, 27)

[2] Zeyuan Allen-Zhu, Yuanzhi Li, Rafael Oliveira, and Avi Wigderson. Much faster algorithms for matrix scaling. In 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS), pages 890– 901. IEEE, 2017. (cited on 2, 4, 7)

[3] Jason Altschuler, Jonathan Niles-Weed, and Philippe Rigollet. Near-linear time approximation algorithms for optimal transport via sinkhorn iteration. Advances in Neural Information Processing Systems, 30, 2017. (cited on 1, 2)

[4] Michael Bacharach. Estimating nonnegative matrices from marginal data. International Economic Review, 6(3):294–310, 1965. (cited on 1)

[5] Jose Blanchet, Arun Jambulapati, Carson Kent, and Aaron Sidford. Towards optimal running times for optimal transport. arXiv preprint arXiv:1810.07717, 2018. (cited on 7)

[6] Deeparnab Chakrabarty and Sanjeev Khanna. Better and simpler error analysis of the sinkhorn–knopp algorithm for matrix scaling. Mathematical Programming, 188(1):395–407, 2021. (cited on 1, 2)

[7] Michael B Cohen, Aleksander Madry, Dimitris Tsipras, and Adrian Vladu. Matrix scaling and balancing via box constrained newton’s method and interior point methods. In 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS), pages 902–913. IEEE, 2017. (cited on 2)

[8] Marco Cuturi and Gabriel Peyré. Semidual regularized optimal transport. SIAM Review, 60(4):941–965, 2018. (cited on 6)

[9] Alexandre d’Aspremont, Damien Scieur, and Adrien Taylor. Acceleration methods. Foundations and Trends® in Optimization, 5(1-2):1–245, 2021. (cited on 9, 31)

![](images/98442894a71a996a403dd0e4583b72cc605384a5debbb089a1dd0deb9dcc64fa.jpg)  
Figure 3: Accelerated variants of SK on eight MNIST entropic optimal transport instances $( n = 7 8 4 , \eta = 2 \times 1 0 ^ { - 3 } )$

[10] Steven Diamond and Stephen Boyd. Stochastic matrix-free equilibration. Journal of Optimization Theory and Applications, 172(2):436–454, 2017. (cited on 2, 6)

[11] Pavel Dvurechensky, Alexander Gasnikov, and Alexey Kroshnin. Computational optimal transport: Com plexity by accelerated gradient descent is better than by sinkhorn’s algorithm. In International Conference on Machine Learning, pages 1367–1376. PMLR, 2018. (cited on 1, 2, 17)

[12] Christopher Fougner and Stephen Boyd. Parameter selection and preconditioning for a graph form solver. In Emerging Applications of Control and Systems Theory: A Festschrift in Honor of Mathukumalli Vidyasagar, pages 41–61. Springer, 2018. (cited on 6)

[13] Joel Franklin and Jens Lorenz. On the scaling of multidimensional matrices. Linear Algebra and its applications, 114:717–735, 1989. (cited on 1, 2)

[14] Wenzhi Gao. Non-asymptotic local convergence analysis of alternating minimization, April 2026. Blog Post #4.

[15] Wenzhi Gao, Ya-Chi Chu, Yinyu Ye, and Madeleine Udell. Gradient methods with online scaling part I. theoretical foundations. arXiv preprint arXiv:2505.23081, 2025. (cited on 8, 9)

[16] Ferdinand Genans. Fast and large-scale unbalanced optimal transport via its semi-dual and adaptive gradient methods. arXiv preprint arXiv:2602.10697, 2026. (cited on 6, 10)

[17] Kun He. On the eficiency of sinkhorn-knopp for entropically regularized optimal transport. arXiv preprint arXiv:2604.03787, 2026. (cited on 5)

[18] Kun He. Phase transition of the sinkhorn-knopp algorithm. In Proceedings of the 2026 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 4238–4284. SIAM, 2026. (cited on 1, 2, 5)

[19] Martin Idel. A review of matrix scaling and sinkhorn’s normal form for matrices and positive maps. arXiv preprint arXiv:1609.06349, 2016. (cited on 1, 2, 3)

[20] Arun Jambulapati, Aaron Sidford, and Kevin Tian. A direct tilde {O}(1/epsilon) iteration parallel algorithm for optimal transport. Advances in Neural Information Processing Systems, 32, 2019. (cited on 7)

[21] Bahman Kalantari. A theorem of the alternative for multihomogeneous functions and its relationship to diagonal scaling of matrices. Linear Algebra and its Applications, 236:1–24, 1996. (cited on 1)

[22] Bahman Kalantari and Leonid Khachiyan. On the rate of convergence of deterministic and randomized ras matrix scaling algorithms. Operations research letters, 14(5):237–244, 1993. (cited on 2)

[23] Bahman Kalantari and Leonid Khachiyan. On the complexity of nonnegative-matrix scaling. Linear Algebra and its applications, 240:87–103, 1996. (cited on 2)

[24] Bahman Kalantari, Isabella Lari, Federica Ricca, and Bruno Simeone. On the complexity of general matrix scaling and entropy minimization via the ras algorithm. Mathematical Programming, 112(2):371–401, 2008. (cited on 1, 2, 4, 16)

[25] Philip A Knight. The sinkhorn–knopp algorithm: convergence and applications. SIAM Journal on Matrix Analysis and Applications, 30(1):261–275, 2008. (cited on 1, 2, 5, 6)

[26] Jongmin Lee, Chanwoo Park, and Ernest Ryu. A geometric structure of acceleration and its role in making gradients small fast. Advances in Neural Information Processing Systems, 34:11999–12012, 2021. (cited on 7, 26)

[27] Tobias Lehmann, Max-K Von Renesse, Alexander Sambale, and André Uschmajew. A note on overrelaxation in the sinkhorn algorithm. Optimization Letters, 16(8):2209–2220, 2022. (cited on 9)

[28] Tianyi Lin, Nhat Ho, and Michael Jordan. On eficient optimal transport: An analysis of greedy and accelerated mirror descent algorithms. In International Conference on Machine Learning, pages 3982– 3991. PMLR, 2019. (cited on 4, 7, 16)

[29] Nathan Linial, Alex Samorodnitsky, and Avi Wigderson. A deterministic strongly polynomial algorithm for matrix scaling and approximate permanents. In Proceedings of the thirtieth annual ACM symposium on Theory of computing, pages 644–652, 1998. (cited on 2)

[30] Pravin Nair. Softmax is \$1/2\$-lipschitz: A tight bound across all \$\ell\_p\$ norms. Transactions on Machine Learning Research, 2026. (cited on 27)

[31] Arkadi Nemirovski and Uriel Rothblum. On complexity of matrix scaling. Linear Algebra and its Applications, 302:435–460, 1999. (cited on 2)

[32] Yurii Nesterov. Introductory lectures on convex optimization: A basic course, volume 87. Springer Science & Business Media, 2013. (cited on 7, 28)

[33] Gabriel Peyré, Lenaic Chizat, François-Xavier Vialard, and Justin Solomon. Quantum entropic regularization of matrix-valued optimal transport. European Journal of Applied Mathematics, 30(6):1079–1102, 2019. (cited on 9)

[34] Gabriel Peyré and Marco Cuturi. Computational optimal transport: With applications to data science. Now Foundations and Trends, 2019. (cited on 1, 2, 5)

[35] Zhaonan Qu, Alfred Galichon, Wenzhi Gao, and Johan Ugander. On sinkhorn’s algorithm and choice modeling. Operations Research, 2025. (cited on 1, 2, 5, 16)

[36] Richard Sinkhorn and Paul Knopp. Concerning nonnegative matrices and doubly stochastic matrices. Pacific Journal of Mathematics, 21(2):343–348, 1967. (cited on 1)

[37] Adrien B Taylor, Julien M Hendrickx, and François Glineur. Exact worst-case convergence rates of the proximal gradient method for composite convex minimization. Journal of Optimization Theory and Applications, 178(2):455–476, 2018. (cited on 8)

[38] Alexis Thibault, Lénaïc Chizat, Charles Dossal, and Nicolas Papadakis. Overrelaxed sinkhorn–knopp algorithm for regularized optimal transport. Algorithms, 14(5):143, 2021. (cited on 9)

[39] Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Kuai Yu, et al. mhc: Manifold-constrained hyper-connections. arXiv preprint arXiv:2512.24880, 2025. (cited on 1)

[40] Zeyi Xu and Long Chen. Accelerating sinkhorn for entropy-regularized optimal transport. arXiv preprint arXiv:2605.30267, 2026. (cited on 6)

## Appendix

## Table of Contents

A Proof of results in Section 3 16   
A.1 Auxiliary result . 16   
A.2 Proof of Lemma 3.1 16   
A.3 Proof of Theorem 3.1 16   
A.4 Proof of Proposition 3.1 17   
A.5 Proof of Lemma 3.2 17   
A.6 Proof of Lemma 3.3 18   
A.7 Proof of Lemma 3.4 19   
A.8 Proof of Lemma 3.5 23   
A.9 Proof of Theorem 3.2 . 25   
A.10 Proof of Theorem 3.2 . 25   
B Proof of results in Section 4 26   
B.1 Auxiliary results 26   
B.2 Proof of Lemma 4.1 27   
B.3 Proof of Theorem 4.1 28   
B.4 Proof of Theorem 4.2 28   
B.5 Proof of Lemma 4.2 29   
B.6 Proof of Lemma 4.3 30   
B.7 Proof of Lemma 4.4 31

## A Proof of results in Section 3

## A.1 Auxiliary result

Lemma A.1. Suppose A1 holds and that $\begin{array} { r } { p = { \frac { 1 } { m } } \mathbf { 1 } _ { m } , q = { \frac { 1 } { n } } \mathbf { 1 } _ { n } } \end{array}$ . Then there exists a solution $( u ^ { \star } , v ^ { \star } )$ such that

$$
\begin{array} { r } { \| ( u ^ { \star } , v ^ { \star } ) \| _ { \infty } \leq \frac { 1 } { 2 } | \log ( \frac { \| ( p , q ) \| _ { \infty } } { \| A \| _ { - \infty } } ) | + \mathrm { l c m } ( m , n ) \log ( \frac { \| ( p , q ) \| _ { \infty } \| A \| _ { 1 } } { \| p \| _ { 1 } \| A \| _ { - \infty } } ) , } \end{array}
$$

where lcm(a, b) denotes the least common multiple of m and $n .$

Proof. According to Theorem 5.1 of [24], there exist $( u ^ { \star } , v ^ { \star } )$ such that

$$
\begin{array} { r } { u _ { i } ^ { \star } = \frac { 1 } { 2 } \log ( \frac { \| ( p , q ) \| _ { \infty } } { \| A \| _ { - \infty } } ) + t \quad \mathrm { a n d } \quad v _ { j } ^ { \star } = \frac { 1 } { 2 } \log ( \frac { \| ( p , q ) \| _ { \infty } } { \| A \| _ { - \infty } } ) - t } \end{array}
$$

where

$$
\begin{array} { r } { | t | \leq \frac { 1 } { \delta _ { m , n } } \big | - \| p \| _ { 1 } \log ( \frac { \| A \| _ { 1 } } { \| p \| _ { 1 } } ) - \| p \| _ { 1 } \log \big ( \frac { \| ( p , q ) \| _ { \infty } } { \| A \| _ { - \infty } } \big ) \big | = \frac { 1 } { \delta _ { m , n } } \| p \| _ { 1 } \log \big ( \frac { \| ( p , q ) \| _ { \infty } \| A \| _ { 1 } } { \| p \| _ { 1 } \| A \| _ { - \infty } } \big ) , } \end{array}
$$

and $\begin{array} { r } { \delta _ { m n } = \frac { 1 } { \mathrm { l c m } ( m , n ) } } \end{array}$ , where lcm $( m , n )$ denotes the least common multiple between m and $n .$ . Therefore, with $\| p \| _ { 1 } = 1$ , we have

$$
\begin{array} { r } { | u _ { i } ^ { \star } | \leq \frac { 1 } { 2 } \big | \log ( \frac { \| ( p , q ) \| _ { \infty } } { \| A \| _ { - \infty } } ) \big | + \mathrm { { l c m } } ( m , n ) \log ( \frac { \| ( p , q ) \| _ { \infty } \| A \| _ { 1 } } { \| A \| _ { - \infty } } ) , } \end{array}
$$

and the same result holds for $v ^ { \star }$

Lemma A.2. Suppose A1 holds and that $m = n , A > 0$ and $\| p \| _ { 1 } = 1$ . Then there exists a solution $( u ^ { \star } , v ^ { \star } )$ such that $\begin{array} { r } { \| ( u ^ { \star } , v ^ { \star } ) \| _ { \infty } \leq \log ( \frac { n } { \| A \| _ { - \infty } \| ( p , q ) \| _ { - \infty } ^ { 2 } } ) } \end{array}$

Proof. The proof is immediate by taking $C _ { i j } = \mathbf { e } ^ { - a _ { i j } }$ and adopting $v  - v$ in [28].

## A.2 Proof of Lemma 3.1

The first two bullet points follow from Lemma A.1 with $\operatorname { l c m } ( m , n ) \leq$ mn and that $\operatorname { l c m } ( n , n ) = n$ . The last bullet point follows from Lemma A.2.

## A.3 Proof of Theorem 3.1

We start by showing that the Sinkhorn iterates satisfy $\| u ^ { k + 1 } - u ^ { \star } \| _ { \infty } \leq \| u ^ { 1 } - u ^ { \star } \| _ { \infty }$ . According to Lemma 2 of [35], consider any $i \in [ m ]$ and $\gamma , \eta$ such that

$$
\gamma \mathrm { e } ^ { u _ { i } ^ { \star } } \leq \mathrm { e } ^ { u _ { i } ^ { k } } \leq \eta \mathrm { e } ^ { u _ { i } ^ { \star } } ,
$$

we always have $\gamma \mathrm { e } ^ { u _ { i } ^ { \star } } \leq \mathrm { e } ^ { u _ { i } ^ { k + 1 } } \leq \eta \mathrm { e } ^ { u _ { i } ^ { \star } }$ . Taking $\gamma = \mathrm { e } ^ { u _ { i } ^ { \star } - u _ { i } ^ { 1 } }$ and $\eta = \mathrm { e } ^ { u _ { i } ^ { 1 } - u _ { i } ^ { \star } }$ , we have, inductively, that

$$
\mathrm { e } ^ { u _ { i } ^ { \star } - u _ { i } ^ { 1 } } \mathrm { e } ^ { u _ { j } ^ { \star } } \leq \mathrm { e } ^ { u _ { i } ^ { k } } \leq \mathrm { e } ^ { u _ { i } ^ { 1 } - u _ { i } ^ { \star } } \mathrm { e } ^ { u _ { i } ^ { \star } } ,
$$

which implies $\mathrm { e } ^ { u _ { i } ^ { \star } - u _ { i } ^ { 1 } } \leq \mathrm { e } ^ { u _ { i } ^ { k } - u _ { i } ^ { \star } } \leq \mathrm { e } ^ { u _ { i } ^ { 1 } - u _ { i } ^ { \star } }$ and that $| u _ { i } ^ { k } - u _ { j } ^ { \star } | \leq | u _ { i } ^ { 1 } - u _ { i } ^ { \star } |$ , giving $\| u ^ { k } - u ^ { \star } \| _ { \infty } \leq \| u ^ { 1 } - u ^ { \star } \| _ { \infty }$ . The same argument holds for $v ^ { k }$ . Next, we deduce that

$$
\varphi ( u ^ { k } , v ^ { k } ) - \varphi ( u ^ { \star } , v ^ { \star } ) \leq \langle \nabla \varphi ( u ^ { k } , v ^ { k } ) , ( u ^ { k } - u ^ { \star } , v ^ { k } - v ^ { \star } ) \rangle\tag{7}
$$

$$
\leq \| \nabla \varphi ( u ^ { k } , v ^ { k } ) \| _ { 1 } \| ( u ^ { k } - u ^ { \star } , v ^ { k } - v ^ { \star } ) \| _ { \infty }\tag{8}
$$

$$
\leq \| \nabla \varphi ( u ^ { k } , v ^ { k } ) \| _ { 1 } \| ( u ^ { 1 } - u ^ { \star } , v ^ { 1 } - v ^ { \star } ) \| _ { \infty }\tag{9}
$$

$$
= \| \nabla \varphi ( u ^ { k } , v ^ { k } ) \| _ { 1 } \| ( u ^ { \star } , v ^ { \star } ) \| _ { \infty }\tag{10}
$$

$$
\leq D \| \nabla \varphi ( u ^ { k } , v ^ { k } ) \| _ { 1 } ,
$$

where (7) uses convexity of $\varphi ; ( 8 )$ uses Hölder’s inequality $\langle a , b \rangle \leq \| a \| _ { 1 } \| b \| _ { \infty } ; ( 9 )$ uses the previously derived relation $\| u ^ { k } - u ^ { \star } \| _ { \infty } \leq \| u ^ { 1 } - u ^ { \star } \| _ { \infty }$ and $\| v ^ { k } - v ^ { \star } \| _ { \infty } \leq \| v ^ { \mathrm { i } } - v ^ { \star } \| _ { \infty } ;$ the relation (10) uses the initialization $( u ^ { 1 } , v ^ { 1 } ) \ = \ ( \mathbf { 0 } _ { m } , \mathbf { 0 } _ { n } )$ . Finally, the proof of sublinear convergence is similar to that in [11]. Denote $\delta _ { k } : = $ $\varphi ( u ^ { k } , v ^ { k } ) - \varphi ( u ^ { \star } , v ^ { \star } )$ . According to the Theorem 1 of [11], given $\varphi ( u ^ { k } , v ^ { k } ) - \varphi ( u ^ { \star } , v ^ { \star } ) \leq \operatorname { \dot { D } } \| \nabla \varphi ( u ^ { k } , v ^ { k } ) \| _ { 1 }$ , we have

$$
\begin{array} { r } { \delta _ { k + 1 } \leq \delta _ { k } - \frac { \delta _ { k } ^ { 2 } } { 2 \| p \| _ { 1 } D ^ { 2 } } , } \end{array}
$$

which implies $\begin{array} { r } { \frac { 1 } { \delta _ { k + 1 } } - \frac { 1 } { \delta _ { k } } = \frac { \delta _ { k } - \delta _ { k + 1 } } { \delta _ { k } \delta _ { k + 1 } } \geq \frac { 1 } { 2 \| p \| _ { 1 } D ^ { 2 } } \frac { \delta _ { k } } { \delta _ { k + 1 } } \geq \frac { 1 } { 2 \| p \| _ { 1 } D ^ { 2 } } } \end{array}$ since AM enforces $\delta _ { k + 1 } \leq \delta _ { k }$ . Hence

$$
\begin{array} { r } { \frac { 1 } { \varphi ( u ^ { K + 1 } , v ^ { K + 1 } ) - \varphi ( u ^ { \star } , v ^ { \star } ) } = \frac { 1 } { \delta _ { K + 1 } } \geq \frac { 1 } { \delta _ { 1 } } + \frac { K } { 2 \| p \| _ { 1 } D ^ { 2 } } } \end{array}
$$

and we arrive at $\begin{array} { r } { \varphi ( u ^ { K + 1 } , v ^ { K + 1 } ) - \varphi ( u ^ { \star } , v ^ { \star } ) \leq \frac { 2 \| p \| _ { 1 } D ^ { 2 } } { K } } \end{array}$ . This completes the proof.

## A.4 Proof of Proposition 3.1

This equation can be directly verified. Let $\Delta u : = u - u ^ { \star }$ and $\Delta v : = v - v ^ { \star }$ and consider

$$
\begin{array} { r l } & { \varphi ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) = \langle \mathrm { e } ^ { u } , A \mathrm { e } ^ { - v } \rangle - \langle p , u \rangle + \langle q , v \rangle - ( \langle \mathrm { e } ^ { u ^ { \star } } , A \mathrm { e } ^ { - v ^ { \star } } \rangle - \langle p , u ^ { \star } \rangle + \langle q , v ^ { \star } \rangle ) } \\ & { \qquad = \langle \mathrm { e } ^ { u } , A \mathrm { e } ^ { - v } \rangle - \langle \mathrm { e } ^ { u ^ { \star } } , A \mathrm { e } ^ { - v ^ { \star } } \rangle - \langle p , u - u ^ { \star } \rangle + \langle q , v - v ^ { \star } \rangle } \end{array}
$$

Using $\langle \mathrm { e } ^ { u } , A \mathrm { e } ^ { - v } \rangle = \langle \mathrm { e } ^ { u ^ { \star } } \circ \mathrm { e } ^ { u - u ^ { \star } } , A \mathrm { e } ^ { v ^ { \star } - v } \circ \mathrm { e } ^ { - v ^ { \star } } \rangle = \langle \mathrm { e } ^ { u - u ^ { \star } } , \mathrm { e } ^ { U ^ { \star } } A \mathrm { e } ^ { - V ^ { \star } } \mathrm { e } ^ { v ^ { \star } - v } \rangle$ , we deduce that

$$
\begin{array} { r l } & { \langle \mathrm { e } ^ { u } , A \mathrm { e } ^ { - v } \rangle - \langle \mathrm { e } ^ { u ^ { \star } } , A \mathrm { e } ^ { - v ^ { \star } } \rangle = \langle \mathrm { e } ^ { u - u ^ { \star } } , \mathrm { e } ^ { U ^ { \star } } A \mathrm { e } ^ { - V ^ { \star } } \mathrm { e } ^ { v ^ { \star } - v } \rangle - \langle \mathrm { e } ^ { u ^ { \star } } , A \mathrm { e } ^ { - v ^ { \star } } \rangle } \\ & { \phantom { x x x x x x x x x x x x x x x x x x x x x x x } } \\ & { \phantom { x x x x x x x x x x x x x x x x x x } = \sum _ { i , j } a _ { i j } \mathrm { e } ^ { u _ { i } ^ { \star } - v _ { j } ^ { \star } } \mathrm { e } ^ { \Delta u _ { i } - \Delta v _ { j } } - \sum _ { i , j } a _ { i j } \mathrm { e } ^ { u _ { i } ^ { \star } - v _ { j } ^ { \star } } } \\ & { \phantom { x x x x x x x x x x x x x x x x x } } \\ & { \phantom { x x x x x x x x x x x x x x x } = \sum _ { i , j } a _ { i j } \mathrm { e } ^ { u _ { i } ^ { \star } - v _ { j } ^ { \star } } [ \mathrm { e } ^ { \Delta u _ { i } - \Delta v _ { j } } - 1 ] . } \end{array}
$$

Since $\begin{array} { r } { p _ { i } = \sum _ { j } a _ { i j } \mathrm { e } ^ { u _ { i } ^ { \star } - v _ { j } ^ { \star } } } \end{array}$ and $\begin{array} { r } { q _ { j } = \sum _ { i } a _ { i j } \mathrm { e } ^ { u _ { i } ^ { \star } - v _ { j } ^ { \star } } } \end{array}$ , we have

$$
\begin{array} { r l } & { - \langle p , u - u ^ { \star } \rangle + \langle q , v - v ^ { \star } \rangle = - \sum _ { i , j } a _ { i j } \mathrm { e } ^ { u _ { i } ^ { \star } - v _ { j } ^ { \star } } \Delta u _ { i } + \sum _ { i , j } a _ { i j } \mathrm { e } ^ { u _ { i } ^ { \star } - v _ { j } ^ { \star } } \Delta v _ { i } } \\ & { \phantom { = } - \sum _ { i , j } a _ { i j } \mathrm { e } ^ { u _ { i } ^ { \star } - v _ { j } ^ { \star } } ( \Delta u _ { i } - \Delta v _ { j } ) . } \end{array}
$$

Plugging in $a _ { i j } \mathrm { e } ^ { u _ { i } ^ { \star } - v _ { j } ^ { \star } } = a _ { i j } ^ { \star }$ gives

$$
\begin{array} { r l } & { \boldsymbol { \varphi } ( u , v ) - \boldsymbol { \varphi } ( u ^ { \star } , v ^ { \star } ) = \sum _ { i , j } a _ { i j } \mathrm { e } ^ { u _ { i } ^ { \star } - v _ { j } ^ { \star } } [ \mathrm { e } ^ { \Delta u _ { i } - \Delta v _ { j } } - 1 ] - \sum _ { i , j } a _ { i j } \mathrm { e } ^ { u _ { i } ^ { \star } - v _ { j } ^ { \star } } ( \Delta u _ { i } - \Delta v _ { j } ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad = \sum _ { i , j } a _ { i j } \mathrm { e } ^ { u _ { i } ^ { \star } - v _ { j } ^ { \star } } [ \mathrm { e } ^ { \Delta u _ { i } - \Delta v _ { j } } - 1 - ( \Delta u _ { i } - \Delta v _ { j } ) ] } \\ & { \quad \quad \quad \quad \quad \quad \quad = \sum _ { i , j } a _ { i j } \mathrm { e } ^ { u _ { i } ^ { \star } - v _ { j } ^ { \star } } ( \mathrm { e } ^ { \Delta _ { i j } } - 1 - \Delta _ { i j } ) } \\ & { \quad \quad \quad \quad \quad \quad = \sum _ { i , j } a _ { i j } ^ { \star } \boldsymbol { \phi } ( \Delta _ { i j } ) , } \end{array}
$$

and this completes the proof.

## A.5 Proof of Lemma 3.2

Given $\begin{array} { r } { \phi ( \delta ) = \mathrm { e } ^ { \delta } - \delta - 1 } \end{array}$ . We have $\phi ^ { \prime } ( \delta ) = \mathrm { e } ^ { \delta } - 1$ and $\phi ^ { \prime \prime } ( \delta ) = \mathrm { e } ^ { \delta }$ . Hence $\phi ^ { \prime } ( \delta ) - \delta = \phi ( \delta )$ and

$$
\vert \phi ^ { \prime } ( \delta ) \vert = \vert \phi ^ { \prime \prime } ( \delta ) - 1 \vert = \vert \mathrm { e } ^ { \delta } - 1 \vert
$$

If $\delta > 0 .$ , by $\begin{array} { r } { \mathrm { e } ^ { \delta } - \delta - 1 \geq \frac { \delta ^ { 2 } } { 2 } } \end{array}$ , we have $| \delta | \leq \sqrt { 2 \phi ( \delta ) }$ and that

$$
| \mathrm { e } ^ { \delta } - 1 | \leq | \mathrm { e } ^ { \delta } - \delta - 1 | + | \delta | \leq \phi ( \delta ) + \sqrt { 2 \phi ( \delta ) } .
$$

If $\delta < 0$ , we deduce that

$$
| \mathrm { e } ^ { \delta } - 1 | = 1 - \mathrm { e } ^ { \delta } \leq \sqrt { 2 ( \mathrm { e } ^ { \delta } - \delta - 1 ) } = \sqrt { 2 \phi ( \delta ) } \leq \phi ( \delta ) + \sqrt { 2 \phi ( \delta ) } ,
$$

where the first inequality follows from defining

$$
h ( \delta ) : = ( 1 - \mathrm { e } ^ { \delta } ) ^ { 2 } - 2 ( \mathrm { e } ^ { \delta } - \delta - 1 ) = \mathrm { e } ^ { 2 \delta } - 4 \mathrm { e } ^ { \delta } + 2 \delta + 3
$$

and noticing that $h ( 0 ) = 0$ and that $h ^ { \prime } ( 0 ) = 2 \mathrm { e } ^ { 2 \delta } - 4 \mathrm { e } ^ { \delta } + 2 = 2 ( \mathrm { e } ^ { \delta } - 1 ) ^ { 2 } \geq 0$ , giving $h ( \delta ) \leq 0$ for $\delta \leq 0$ Combining these two cases completes the proof.

## A.6 Proof of Lemma 3.3

For brevity, in the proof we will denote $R : = A ^ { \star }$ and hence

$$
\nabla ^ { 2 } \varphi ( u ^ { \star } , v ^ { \star } ) = \binom { P } { - R ^ { \top } } \frac { - R } { Q } \Big ) .
$$

Since $( u , v )$ is generated by AM, assuming $\nabla _ { v } \lambda ( u , v ) = 0$ , we have

$$
v - v ^ { \star } = Q ^ { - 1 } R ^ { \top } ( u - u ^ { \star } )\tag{11}
$$

and consider the preconditioned update $\begin{array} { r } { u ^ { + } = u - P ^ { - 1 } \nabla \lambda _ { u } ( u , v ) } \end{array}$ . It gives

$$
\begin{array} { r l } & { u ^ { + } - u ^ { \star } = u - P ^ { - 1 } \nabla \lambda _ { u } ( u , v ) - u ^ { \star } } \\ & { \qquad = u - u ^ { \star } - P ^ { - 1 } ( P ( u - u ^ { \star } ) - R ( v - v ^ { \star } ) ) } \\ & { \qquad = P ^ { - 1 } R ( v - v ^ { \star } ) } \end{array}\tag{12}
$$

and it is easy to see that $\nabla _ { u } \lambda ( u ^ { + } , v ) = P ( u ^ { + } - u ^ { \star } ) - R ( v - v ^ { \star } ) = 0$ . Hence AM applied to λ yields the same $u ^ { + }$ Next, we consider the contraction, and we successively deduce that

$$
\begin{array} { r l } & { \lambda ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) = \frac { 1 } { 2 } \| u - u ^ { \star } \| _ { P } ^ { 2 } - \langle u - u ^ { \star } , R ( v - v ^ { \star } ) \rangle + \frac { 1 } { 2 } \| v - v ^ { \star } \| _ { Q } ^ { 2 } } \\ & { \qquad = \frac { 1 } { 2 } \| u - u ^ { \star } \| _ { P - R Q ^ { - 1 } R ^ { \tau } } ^ { 2 } , } \end{array}\tag{13}
$$

where (13) applies (11). Furthermore, plugging (11) into (12) gives $u ^ { + } - u ^ { \star } = P ^ { - 1 } R Q ^ { - 1 } R ^ { \top } ( u - u ^ { \star } )$ and

$$
\begin{array} { r l } & { \lambda ( u ^ { + } , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) = \frac { 1 } { 2 } \| u ^ { + } - u ^ { \star } \| _ { P } ^ { 2 } - \langle u ^ { + } - u ^ { \star } , R ( v - v ^ { \star } ) \rangle + \frac { 1 } { 2 } \| v - v ^ { \star } \| _ { Q } ^ { 2 } } \\ & { \qquad = \frac { 1 } { 2 } \| u - u ^ { \star } \| _ { R Q ^ { - 1 } R ^ { \top } - R Q ^ { - 1 } R ^ { \top } P ^ { - 1 } R Q ^ { - 1 } R ^ { \top } } ^ { 2 } . } \end{array}
$$

Now it sufices to show, for any z, that

$$
\begin{array} { r } { \| z \| _ { R Q ^ { - 1 } R ^ { \top } - R Q ^ { - 1 } R ^ { \top } P ^ { - 1 } R Q ^ { - 1 } R ^ { \top } } ^ { 2 } \leq ( 1 - \sigma _ { 2 } ) \| z \| _ { P - R Q ^ { - 1 } R ^ { \top } } ^ { 2 } . } \end{array}
$$

Denote the eigen-decomposition $M : = P ^ { - 1 / 2 } R Q ^ { - 1 } R ^ { \top } P ^ { - 1 / 2 } = U \Sigma U ^ { \top }$ and suppose the eigenvalues of Σ are non-increasing (left top is the largest). We may check that $P ^ { 1 / 2 } \mathbf { 1 } _ { m }$ is an eigenvector with eigenvalue 1:

$$
M P ^ { 1 / 2 } \mathbf { 1 } _ { m } = P ^ { - 1 / 2 } R Q ^ { - 1 } R ^ { \top } P ^ { - 1 / 2 } ( P ^ { 1 / 2 } \mathbf { 1 } ) = P ^ { 1 / 2 } \mathbf { 1 } _ { m } ,
$$

and therefore $P ^ { 1 / 2 } \mathbf { 1 } _ { m } \in \mathrm { N u l } ( I - M )$ . Besides, note that $0 \preceq \Sigma \preceq I$ since $\Big ( \begin{array} { c } { { P } } \\ { { - R ^ { \top } } } \end{array} \Big ) \Big ( \begin{array} { c } { { - R } } \\ { { Q } } \end{array} \Big ) \succeq 0$ implies that its Schur complement $P - R Q ^ { - 1 } R ^ { \top } \succeq 0$ is positive semidefinite, which further implies

$$
I \succeq P ^ { - 1 / 2 } R Q ^ { - 1 } R ^ { \top } P ^ { - 1 / 2 } \succeq 0
$$

Now we deduce that

$$
\begin{array} { r l } & { \| z \| _ { R Q ^ { - 1 } R ^ { \top } - R Q ^ { - 1 } R ^ { \top } P ^ { - 1 } R Q ^ { - 1 } R ^ { \top } } ^ { 2 } = \langle z , ( R Q ^ { - 1 } R ^ { \top } - R Q ^ { - 1 } R ^ { \top } P ^ { - 1 } R Q ^ { - 1 } R ^ { \top } ) z \rangle } \\ & { \qquad = \langle z , ( P ^ { 1 / 2 } M P ^ { 1 / 2 } - P ^ { 1 / 2 } M ^ { \top } M P ^ { 1 / 2 } ) z \rangle } \\ & { \qquad = \| P ^ { 1 / 2 } z \| _ { M - M ^ { \top } M } ^ { 2 } } \end{array}
$$

and $\| z \| _ { P - R O ^ { - 1 } R ^ { \top } } ^ { 2 } = \langle z , ( P - P ^ { 1 / 2 } M P ^ { 1 / 2 } ) z \rangle = \| P ^ { 1 / 2 } z \| _ { I - M } ^ { 2 }$ . Since $\operatorname { N u l } ( I - M ) \subseteq \operatorname { N u l } ( M - M ^ { \top } M )$ , we assume $P ^ { 1 / 2 } z \perp \mathrm { N u l } ( I - M )$ and deduce that

$$
\begin{array} { r } { \operatorname* { m a x } _ { \substack { P ^ { 1 / 2 } z \prod _ { M = 1 } ^ { 1 } \prod _ { T = M } ^ { 1 } \prod _ { T = M } ^ { 1 } } } = \operatorname* { m a x } _ { \substack { x \bot \mathrm { N u l } ( I - M ) } } \frac { \langle x , ( M - M ^ { \top } M ) x \rangle } { \langle x , ( I - M ) x \rangle } } \\ { = \operatorname* { m a x } _ { \substack { x \bot \mathrm { N u l } ( I - \Sigma ) } } \frac { \langle x , ( \Sigma - \Sigma ^ { 2 } ) x \rangle } { \langle x , ( I - \Sigma ) x \rangle } } \\ { = \operatorname* { m a x } _ { \substack { x \bot \mathrm { N u l } ( I - \Sigma ) } } \frac { \langle x , ( \Sigma - \Sigma ^ { 2 } ) x \rangle } { \langle x , ( I - \Sigma ) x \rangle } } \\ { \leq \operatorname* { m a x } _ { \substack { x \bot \mathrm { N u l } ( I - \Sigma ) } } \frac { \langle x , ( \Sigma - \Sigma ^ { 2 } ) x \rangle } { \langle x , ( I - \Sigma ) x \rangle } } \\ { \leq \lambda _ { 2 } ( \Sigma ) } \\ { = 1 - \lambda _ { 2 } ( I - \Sigma ) = 1 - \sigma _ { 2 } . } \end{array}
$$

Using $\begin{array} { r } { \Vert z \Vert _ { R Q ^ { - 1 } R ^ { \top } - R Q ^ { - 1 } R ^ { \top } P ^ { - 1 } R Q ^ { - 1 } R ^ { \top } } ^ { 2 } \leq ( 1 - \sigma _ { 2 } ) \Vert z \Vert _ { P - R Q ^ { - 1 } R ^ { \top } } ^ { 2 } } \end{array}$ , we have

$$
\lambda ( u ^ { + } , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) \leq ( 1 - \sigma _ { 2 } ) [ \lambda ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] ,
$$

and this completes the proof. Repeating the argument by switching the role of u, v and noticing that

$$
\begin{array} { r l } & { \lambda _ { 2 } ( I _ { m } - P ^ { - 1 / 2 } R Q ^ { - 1 } R ^ { \top } P ^ { - 1 / 2 } ) = \lambda _ { 2 } ( I _ { m } - P ^ { - 1 / 2 } R Q ^ { - 1 / 2 } ( P ^ { - 1 / 2 } R Q ^ { - 1 / 2 } ) ^ { \top } ) } \\ & { \phantom { \lambda _ { 2 } ( I _ { m } - } = \lambda _ { 2 } ( I _ { n } - ( P ^ { - 1 / 2 } R Q ^ { - 1 / 2 } ) ^ { \top } P ^ { - 1 / 2 } R Q ^ { - 1 / 2 } ) } \\ & { \phantom { \lambda _ { 2 } ( I _ { m } - } = \lambda _ { 2 } ( I _ { n } - Q ^ { - 1 / 2 } R ^ { \top } P ^ { - 1 } R Q ^ { - 1 / 2 } ) , } \end{array}
$$

we have $\lambda ( u ^ { + } , v ^ { + } ) - \varphi ( u ^ { \star } , v ^ { \star } ) \leq ( 1 - \sigma _ { 2 } ) ^ { 2 } [ \lambda ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) ]$ . This completes the proof.

## A.7 Proof of Lemma 3.4

Recall the notation $\Delta _ { i j } : = \Delta u _ { i } - \Delta v _ { j }$ and define $g ( t ) : = \varphi ( u + t d ^ { u } , v + t d ^ { v } )$ and $h ( t ) : = \lambda ( u + t d ^ { u } , v + t d ^ { v } )$ with $d = \left( d ^ { u } , d ^ { v } \right) \in \mathbb { R } ^ { m + n }$ . We have $g ( 0 ) = \varphi ( u , v ) , h ( 0 ) = \lambda ( u , v )$ , and that

$$
\begin{array} { r l } & { { g } ^ { \prime } ( t ) = \frac { \mathrm { d } } { \mathrm { d } t } [ \sum _ { i , j } a _ { i j } ^ { \star } \phi ( \Delta _ { i j } + t ( d _ { i } ^ { u } - d _ { j } ^ { v } ) ) ] } \\ & { \qquad = \sum _ { i , j } a _ { i j } ^ { \star } \phi ^ { \prime } ( \Delta _ { i j } + t ( d _ { i } ^ { u } - d _ { j } ^ { v } ) ) ( d _ { i } ^ { u } - d _ { j } ^ { v } ) } \\ & { { g } ^ { \prime \prime } ( t ) = \sum _ { i , j } a _ { i j } ^ { \star } \phi ^ { \prime \prime } ( \Delta _ { i j } + t ( d _ { i } ^ { u } - d _ { j } ^ { v } ) ) ( d _ { i } ^ { u } - d _ { j } ^ { v } ) ^ { 2 } . } \\ & { { h } ^ { \prime } ( t ) = \sum _ { i , j } a _ { i j } ^ { \star } ( \Delta _ { i j } + t ( d _ { i } ^ { u } - d _ { j } ^ { v } ) ) ( d _ { i } ^ { u } - d _ { j } ^ { v } ) } \\ & { { h } ^ { \prime \prime } ( t ) = \sum _ { i , j } a _ { i j } ^ { \star } ( d _ { i } ^ { u } - d _ { j } ^ { v } ) ^ { 2 } } \end{array}
$$

Taking $t = 0$ , we have

$$
\begin{array} { r l } & { g ^ { \prime } ( 0 ) = \langle \nabla \varphi ( u , v ) , d \rangle = \sum _ { i , j } a _ { i j } ^ { \star } \phi ^ { \prime } ( \Delta _ { i j } ) ( d _ { i } ^ { u } - d _ { j } ^ { v } ) } \\ & { g ^ { \prime \prime } ( 0 ) = \langle d , \nabla ^ { 2 } \varphi ( u , v ) d \rangle = \sum _ { i , j } a _ { i j } ^ { \star } \phi ^ { \prime \prime } ( \Delta _ { i j } ) ( d _ { i } ^ { u } - d _ { j } ^ { v } ) ^ { 2 } } \\ & { h ^ { \prime } ( 0 ) = \langle \nabla \lambda ( u , v ) , d \rangle = \sum _ { i , j } a _ { i j } ^ { \star } \Delta _ { i j } ( d _ { i } ^ { u } - d _ { j } ^ { v } ) } \\ & { h ^ { \prime \prime } ( 0 ) = \langle d , \nabla ^ { 2 } \lambda ( u ^ { \star } , v ^ { \star } ) d \rangle = \sum _ { i , j } a _ { i j } ^ { \star } ( d _ { i } ^ { u } - d _ { j } ^ { v } ) ^ { 2 } . } \end{array}
$$

Proof of relation 1. We invoke Lemma 3.2 and deduce that

$$
\begin{array} { r l } & { | \langle \nabla \varphi ( u , v ) - \nabla \lambda ( u , v ) , d \rangle | = | \sum _ { i , j } a _ { i j } ^ { \star } \phi ^ { \prime } ( \Delta _ { i j } ) ( d _ { i } ^ { u } - d _ { j } ^ { v } ) - \sum _ { i , j } a _ { i j } ^ { \star } \Delta _ { i j } ( d _ { i } ^ { u } - d _ { j } ^ { v } ) | } \\ & { \qquad = | \sum _ { i , j } a _ { i j } ^ { \star } [ \phi ^ { \prime } ( \Delta _ { i j } ) - \Delta _ { i j } ] ( d _ { i } ^ { u } - d _ { j } ^ { v } ) | } \\ & { \qquad = | \sum _ { i , j } a _ { i j } ^ { \star } \phi ( \Delta _ { i j } ) ( d _ { i } ^ { u } - d _ { j } ^ { v } ) | } \\ & { \qquad \le 2 [ \sum _ { i , j } a _ { i j } ^ { \star } \phi ( \Delta _ { i j } ) ] \| d \| _ { \infty } } \\ & { \qquad = 2 \varepsilon \| d \| _ { \infty } , } \end{array}\tag{14}
$$

(15)

where (14) uses $\phi ( \delta ) = \phi ^ { \prime } ( \delta ) - \delta$ from Lemma 3.2; (15) uses Proposition 3.1. Taking $\begin{array} { r } { d = \frac { \nabla \varphi ( u , v ) - \nabla \lambda ( u , v ) } { \Vert \nabla \varphi ( u , v ) - \nabla \lambda ( u , v ) \Vert } } \end{array}$ 1 gives $\begin{array} { r } { \| \nabla \varphi ( u , v ) - \nabla \lambda ( u , v ) \| \leq 2 \varepsilon \frac { \| d \| _ { \infty } } { \| d \| } \leq 2 \varepsilon } \end{array}$ and that

$$
\begin{array} { r } { \| \nabla \varphi ( u , v ) - \nabla \lambda ( u , v ) \| _ { S ^ { - 1 } } = \| S ^ { - 1 / 2 } [ \nabla \varphi ( u , v ) - \nabla \lambda ( u , v ) ] \| \leq \frac { 2 \varepsilon } { \sqrt { \| ( p , q ) \| _ { - \infty } } } . } \end{array}
$$

Next, consider

$$
\begin{array} { r l } & { | \langle \nabla \varphi ( u , v ) , S ^ { - 1 / 2 } d \rangle | = | \sum _ { i , j } a _ { i j } ^ { \star } \phi ^ { \prime } ( \Delta _ { i j } ) ( \frac { d _ { i } ^ { u } } { \sqrt { p _ { i } } } - \frac { d _ { j } ^ { v } } { \sqrt { q _ { j } } } ) | } \\ & { \qquad \leq \sum _ { i , j } a _ { i j } ^ { \star } | \phi ^ { \prime } ( \Delta _ { i j } ) | | \frac { d _ { i } ^ { u } } { \sqrt { p _ { i } } } - \frac { d _ { j } ^ { v } } { \sqrt { q _ { j } } } | } \\ & { \qquad \leq \sum _ { i , j } a _ { i j } ^ { \star } ( \sqrt { 2 \phi ( \Delta _ { i j } ) } + \phi ( \Delta _ { i j } ) ) | \frac { d _ { i } ^ { u } } { \sqrt { p _ { i } } } - \frac { d _ { j } ^ { v } } { \sqrt { q _ { j } } } | } \\ & { \qquad = \sum _ { i , j } a _ { i j } ^ { \star } \sqrt { 2 \phi ( \Delta _ { i j } ) } | \frac { d _ { i } ^ { u } } { \sqrt { p _ { i } } } - \frac { d _ { j } ^ { v } } { \sqrt { q _ { j } } } | + \sum _ { i , j } a _ { i j } ^ { \star } \phi ( \Delta _ { i j } ) | \frac { d _ { i } ^ { u } } { \sqrt { p _ { i } } } - \frac { d _ { j } ^ { v } } { \sqrt { q _ { j } } } | , } \end{array}\tag{16}
$$

where (16) invokes $\phi ^ { \prime } ( \delta ) \leq \sqrt { 2 \phi ( \delta ) } + \phi ( \delta )$ from Lemma 3.2. Now we bound these two terms respectively by

$$
\begin{array} { r l } & { \sum _ { i , j } a _ { i j } ^ { \star } \sqrt { 2 \phi ( \Delta _ { i j } ) } \Big | \frac { d _ { i i } ^ { u } } { \sqrt { p _ { i } } } - \frac { d _ { i j } ^ { u } } { \sqrt { q _ { j } } } \Big | = \sum _ { i , j } \sqrt { 2 a _ { i j } ^ { \star } \phi ( \Delta _ { i j } ) } \Big | \frac { \sqrt { a _ { i j } ^ { \star } d _ { i i } ^ { u } } } { \sqrt { \sum _ { k } a _ { i k } ^ { u } } } - \frac { \sqrt { a _ { i j } ^ { \star } d _ { j } ^ { u } } } { \sqrt { \sum _ { k } a _ { k j } ^ { u } } } \Big | } \\ & { \qquad \leq \sqrt { \sum _ { i , j } 2 a _ { i j } ^ { \star } \phi ( \Delta _ { i j } ) } \sqrt { \sum _ { i , j } \Big ( \frac { \sqrt { a _ { i j } ^ { \star } d _ { i i } ^ { u } } } { \sqrt { \sum _ { k } a _ { i k } ^ { u } } } - \frac { \sqrt { a _ { i j } ^ { \star } d _ { j } ^ { u } } } { \sqrt { \sum _ { k } a _ { k j } ^ { \star } } } \Big ) ^ { 2 } } } \\ & { \qquad = \sqrt { 2 \varepsilon } \sqrt { \sum _ { i , j } \Big ( \frac { \sqrt { a _ { i j } ^ { \star } d _ { i i } ^ { u } } } { \sqrt { \sum _ { k } a _ { i k } ^ { u } } } - \frac { \sqrt { a _ { i j } ^ { \star } d _ { j } ^ { u } } } { \sqrt { \sum _ { k } a _ { k j } ^ { u } } } \Big ) ^ { 2 } } } \\ &  \qquad \leq \sqrt { 2 \varepsilon } \sqrt  2 \Big [ \sum _ { i } \sum _ { j } \frac { a _ { i j } ^ { \star } ( d _ { i i } ^ { u } ) ^ { 2 } } { \sum _ { k } a _ { i k } ^ { u } } + \sum _ { j } \sum _ { i } \frac { a _ { i j } ^ { \star } ( d _ { j } ^ { u } ) ^ { 2 } }  \sum _ { k } \end{array}\tag{17}
$$

(18)

where (17) uses Cauchy-Schwarz $\begin{array} { r } { \sum _ { i , j } a _ { i j } b _ { i j } \le \sqrt { \sum _ { i , j } a _ { i j } } \sqrt { \sum _ { i , j } b _ { i j } } ; } \end{array}$ (18) uses $( a - b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ . Moreover, we have $\begin{array} { r } { \sum _ { i , j } a _ { i j } ^ { \star } \phi ( \Delta _ { i j } ) | \frac { d _ { i } ^ { u } } { \sqrt { p _ { i } } } - \frac { d _ { j } ^ { v } } { \sqrt { q _ { j } } } | \leq \varepsilon \operatorname* { m a x } _ { i j } | \frac { d _ { i } ^ { u } } { \sqrt { p _ { i } } } - \frac { d _ { j } ^ { v } } { \sqrt { q _ { j } } } | \leq \frac { 2 \varepsilon \| d \| _ { \infty } } { \sqrt { \| ( p , q ) \| _ { \infty } } } } \end{array}$ . Taking $\begin{array} { r } { d = \frac { S ^ { - 1 / 2 } \nabla \varphi ( u , v ) } { \| S ^ { - 1 / 2 } \nabla \varphi ( u , v ) \| } } \end{array}$ gives

$$
\begin{array} { r } { \| \nabla \varphi ( u , v ) \| _ { S ^ { - 1 } } = | \langle \nabla \varphi ( u , v ) , S ^ { - 1 / 2 } d \rangle | \leq 2 \sqrt { \varepsilon } + \frac { 2 \varepsilon } { \sqrt { \| ( p , q ) \| _ { - \infty } } } . } \end{array}
$$

Before concluding, we also prove a bound on $\| \nabla \varphi ( u , v ) \|$ . Consider

$$
\begin{array} { r l } { \langle \nabla \varphi ( u , v ) , d \rangle | = | \sum _ { i , j } a _ { i , j } ^ { * } \phi ^ { \prime } ( \Delta _ { i j } ) ( d _ { i } ^ { u } - d _ { j } ^ { v } ) | } & { } \\ & { \leq \sum _ { i , j } a _ { i , j } ^ { * } | \phi ^ { \prime } ( \Delta _ { i j } ) | | d _ { i } ^ { u } - d _ { j } ^ { v } | } \\ & { \leq 2 \sum _ { i , j } a _ { i , j } ^ { * } [ \sqrt { 2 \phi ( \Delta _ { i j } ) } + \phi ( \Delta _ { i j } ) ] | | d | | _ { \infty } } \\ & { = 2 | \sum _ { i , j } a _ { i , j } ^ { * } \sqrt { 2 \phi ( \Delta _ { i j } ) } + \sum _ { i , j } a _ { i , j } ^ { * } \phi ( \Delta _ { i j } ) | | d | | _ { \infty } } \\ & { = 2 | \sqrt { 2 } \sum _ { i , j } a _ { i , j } ^ { * } \sqrt { a _ { i , j } ^ { * } \phi ( \Delta _ { i j } ) } + \sum _ { i , j } a _ { i , j } ^ { * } \phi ( \Delta _ { i j } ) | | d | | _ { \infty } } \\ & { \leq 2 [ \sqrt { 2 } \sqrt { \sum _ { i , j } a _ { i , j } ^ { * } } \sqrt { \sum _ { i , j } a _ { i , j } ^ { * } \phi ( \Delta _ { i j } ) } + \sum _ { i , j } a _ { i , j } ^ { * } \phi ( \Delta _ { i j } ) ] | | d | _ { \infty } } \\ & { \leq ( 4 \sqrt { \| p \| _ { 1 } \mathclose { : } } d + 2 \varepsilon ) | | d | _ { \infty } , } \end{array}\tag{19}
$$

(20)

(21)

where (19) uses Lemma 3.2; (20) uses Cauchy-Schwarz; (21) uses $\| A ^ { \star } \| _ { 1 } = \| p \| _ { 1 }$ . Taking $\begin{array} { r } { d = \frac { \nabla \varphi ( u , v ) } { \lVert \nabla \varphi ( u , v ) \rVert } } \end{array}$ gives $\| \nabla \varphi ( u , v ) \| \leq 4 \sqrt { \| p \| _ { 1 } \varepsilon } + 2 \varepsilon$

Proof of relation 2. We deduce that

$$
\begin{array} { r l } { | \langle d , S ^ { - 1 / 2 } [ \nabla ^ { 2 } \varphi ( u , v ) - \nabla ^ { 2 } \lambda ( u , v ) ] S ^ { - 1 / 2 } , d \rangle | = | \langle S ^ { - 1 / 2 } d , [ \nabla ^ { 2 } \varphi ( u , v ) - \nabla ^ { 2 } \lambda ( u , v ) ] , S ^ { - 1 / 2 } d \rangle | } & { } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \leq \sum _ { i , j } a _ { i j } ^ { * } \big [ \phi ^ { \prime \prime } ( \Delta _ { i j } ) - 1 \big ] \big ( \frac { d _ { i } ^ { u } } { \sqrt { p _ { i } } } - \frac { d _ { j } ^ { u } } { \sqrt { q _ { j } } } \big ) ^ { 2 } \big | } \\ & { \leq \sum _ { i , j } a _ { i j } ^ { * } \big [ \phi ^ { \prime \prime } ( \Delta _ { i j } ) - 1 \big ] \big ( \frac { d _ { i } ^ { u } } { \sqrt { p _ { i } } } - \frac { d _ { j } ^ { u } } { \sqrt { q _ { j } } } \big ) ^ { 2 } } \\ & { \leq \sum _ { i , j } a _ { i j } ^ { * } \sqrt { 2 \phi ( \Delta _ { i j } ) } \big | \frac { d _ { i } ^ { u } } { \sqrt { p _ { i } } } - \frac { d _ { j } ^ { u } } { \sqrt { q _ { j } } } \big | ^ { 2 } + \sum _ { i , j } a _ { i j } ^ { * } \phi ( \Delta _ { i j } ) \big | \frac { d _ { i } ^ { u } } { \sqrt { p _ { i } } } - \frac { d _ { j } ^ { u } } { \sqrt { q _ { j } } } \big | ^ { 2 } , } \end{array}
$$

where the last inequality uses $\phi ^ { \prime \prime } ( \delta ) - 1 = \phi ^ { \prime } ( \delta )$ and Lemma 3.2. Hence, we can similarly bound

$$
\begin{array} { r } { \sum _ { i , j } a _ { i j } ^ { \star } \sqrt { 2 \phi ( \Delta _ { i j } ) } | \frac { d _ { i } ^ { u } } { \sqrt { p _ { i } } } - \frac { d _ { j } ^ { v } } { \sqrt { q _ { i } } } | ^ { 2 } } \end{array}\tag{22}
$$

$$
\begin{array} { r l r } & { } & { = \sum _ { i , j } \sqrt { 2 a _ { i j } ^ { \star } \phi ( \Delta _ { i j } ) } \sqrt { a _ { i j } ^ { \star } } | \frac { d _ { i } ^ { u } } { \sqrt { p _ { i } } } - \frac { d _ { j } ^ { v } } { \sqrt { q _ { j } } } | ^ { 2 } } \end{array}
$$

$$
\begin{array} { r l } & { \leq 2 \sum _ { i , j } \sqrt { 2 a _ { i j } ^ { \star } \phi ( \Delta _ { i j } ) } \Big [ \frac { \sqrt { a _ { i j } ^ { \star } } ( d _ { i } ^ { u } ) ^ { 2 } } { \sqrt { \sum _ { k } a _ { i k } ^ { \star } } } \frac { 1 } { \sqrt { p _ { i } } } + \frac { \sqrt { a _ { i j } ^ { \star } } ( d _ { j } ^ { v } ) ^ { 2 } } { \sqrt { \sum _ { k } a _ { k j } ^ { \star } } } \frac { 1 } { \sqrt { q _ { j } } } \Big ] } \end{array}\tag{23}
$$

$$
\begin{array} { r } { \leq \frac { 2 } { \sqrt { \| ( p , q ) \| _ { - \infty } } } \sum _ { i , j } \sqrt { 2 a _ { i j } ^ { \star } \phi ( \Delta _ { i j } ) } \Big [ \frac { \sqrt { a _ { i j } ^ { \star } } ( d _ { i } ^ { u } ) ^ { 2 } } { \sqrt { \sum _ { k } a _ { i k } ^ { \star } } } + \frac { \sqrt { a _ { i j } ^ { \star } } ( d _ { j } ^ { v } ) ^ { 2 } } { \sqrt { \sum _ { k } a _ { k j } ^ { \star } } } \Big ] } \end{array}\tag{24}
$$

$$
\begin{array} { r } { \leq \frac { 2 \sqrt { 2 } } { \sqrt { \| ( p , q ) \| _ { - \infty } } } \sqrt { \sum _ { i , j } a _ { i j } ^ { \star } \phi ( \Delta _ { i j } ) } \sqrt { \sum _ { i } \sum _ { j } \frac { a _ { i j } ^ { \star } ( d _ { i } ^ { u } ) ^ { 4 } } { \sum _ { k } a _ { i k } ^ { \star } } + \sum _ { j } \sum _ { i } \frac { a _ { i j } ^ { \star } ( d _ { j } ^ { v } ) ^ { 4 } } { \sum _ { k } a _ { k j } ^ { \star } } } } \end{array}\tag{25}
$$

$$
\begin{array} { r } { = \frac { 2 \sqrt { 2 } } { \sqrt { \| ( p , q ) \| _ { - \infty } } } \sqrt { \sum _ { i , j } a _ { i j } ^ { \star } \phi ( \Delta _ { i j } ) } \sqrt { \sum _ { i } ( d _ { i } ^ { u } ) ^ { 4 } + \sum _ { j } ( d _ { j } ^ { v } ) ^ { 4 } } = \frac { 2 \sqrt { 2 \varepsilon } } { \sqrt { \| ( p , q ) \| _ { - \infty } } } \| d \| _ { 4 } ^ { 2 } \leq \frac { 2 \sqrt { 2 \varepsilon } } { \sqrt { \| ( p , q ) \| _ { - \infty } } } \| d \| ^ { 2 } } \end{array}\tag{26}
$$

where (23) uses $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 } ; ( 2 4 )$ uses $p _ { i } , q _ { j } \geq \| ( p , q ) \| _ { - \infty } ; ( 2 5 )$ applies Cauchy-Schwarz and (26) uses $\| x \| _ { 4 } ^ { 2 } \leq \| x \| ^ { 2 }$ ; We also have $\begin{array} { r } { \sum _ { i , j } a _ { i j } ^ { \star } \phi ( \Delta _ { i j } ) | \frac { d _ { i } ^ { u } } { \sqrt { p _ { i } } } - \frac { d _ { j } ^ { v } } { \sqrt { q _ { j } } } | ^ { 2 } \leq \frac { 2 \varepsilon } { \| ( p , q ) \| _ { - \infty } } \| d \| _ { \infty } ^ { 2 } } \end{array}$ . Hence for any $d \neq 0 ,$ we have

$$
\begin{array} { r } { | \langle d , S ^ { - 1 / 2 } [ \nabla ^ { 2 } \varphi ( u , v ) - \nabla ^ { 2 } \lambda ( u , v ) ] S ^ { - 1 / 2 } , d \rangle | \leq 2 \sqrt { \frac { 2 \varepsilon } { \| ( p , q ) \| _ { - \infty } } } \| d \| ^ { 2 } + \frac { 2 \varepsilon } { \| ( p , q ) \| _ { - \infty } } \| d \| _ { \infty } ^ { 2 } } \end{array}
$$

$$
\begin{array} { r } { \mathrm { a n d } \| S ^ { - 1 / 2 } [ \nabla ^ { 2 } \varphi ( u , v ) - \nabla ^ { 2 } \lambda ( u , v ) ] S ^ { - 1 / 2 } \| \leq 2 \sqrt { \frac { 2 \varepsilon } { \| ( p , q ) \| _ { - \infty } } } + \frac { 2 \varepsilon } { \| ( p , q ) \| _ { - \infty } } . } \end{array}
$$

Proof of relation 3. Finally, we consider the zeroth-order relation. The proof uses the fact that $\nabla \varphi \approx 0$ to restrict $( u , v )$ and deduce a bound that only depends on $\sigma _ { 2 }$ . Since $\begin{array} { r } { \varepsilon \le \frac { \sigma _ { 2 } ^ { 2 } s } { 1 2 9 6 } \le s } \end{array}$ , we have $\begin{array} { r } { \frac { 1 } { \sqrt { s } } \varepsilon \leq \sqrt { \varepsilon } } \end{array}$ and that $\textstyle { \sqrt { \frac { \varepsilon } { s } } } \geq { \frac { \varepsilon } { s } }$ . By Taylor theorem, for any optimal $z ^ { \star } = ( u ^ { \star } , v ^ { \star } )$ , we define $z _ { t } : = ( u ^ { \star } + t ( u - u ^ { \star } ) , v ^ { \star } + t ( v - v ^ { \star } ) )$ and

deduce

$$
\begin{array} { r l } & { \varphi ( u , v ) - \lambda ( u , v ) = \int _ { 0 } ^ { 1 } ( 1 - t ) \langle z - z ^ { \star } , \vert \nabla ^ { 2 } \varphi ( z _ { t } ) - \nabla ^ { 2 } \varphi ( z ^ { \star } ) \vert ( z - z ^ { \star } ) \rangle \mathrm { d } t } \\ & { \qquad = \int _ { 0 } ^ { 1 } ( 1 - t ) \langle S ^ { 1 / 2 } ( z - z ^ { \star } ) , S ^ { - 1 / 2 } [ \nabla ^ { 2 } \varphi ( z _ { t } ) - \nabla ^ { 2 } \varphi ( z ^ { \star } ) ] S ^ { - 1 / 2 } S ^ { 1 / 2 } ( z - z ^ { \star } ) \rangle \mathrm { d } t } \\ & { \qquad \geq - ( 2 \sqrt { \frac { \varepsilon } { s } } + \frac { \varepsilon } { s } ) \Vert z - z ^ { \star } \Vert _ { S } ^ { 2 } \geq - 3 \sqrt { \frac { \varepsilon } { s } } \Vert z - z ^ { \star } \Vert _ { S } ^ { 2 } . } \end{array}\tag{27}
$$

Next, using $\begin{array} { r } { \| \nabla \varphi ( z ) - \nabla \lambda ( z ) \| _ { S ^ { - 1 } } \leq \frac { 2 } { \sqrt { s } } \varepsilon } \end{array}$ and relation (1) from Lemma 3.4, we have

$$
\begin{array} { r } { \| \nabla \lambda ( z ) \| _ { S ^ { - 1 } } \leq \| \nabla \varphi ( z ) \| _ { S ^ { - 1 } } + \| \nabla \lambda ( z ) - \nabla \varphi ( z ) \| _ { S ^ { - 1 } } \leq 2 \sqrt { \varepsilon } + \frac { 2 } { \sqrt { s } } \varepsilon + \frac { 2 } { \sqrt { s } } \varepsilon \leq 6 \sqrt { \varepsilon } . } \end{array}
$$

Given that

$$
\begin{array} { r } { \nabla \lambda ( z ) = \Bigl ( \begin{array} { c } { P ( u - u ^ { \star } ) - R ( v - v ^ { \star } ) } \\ { Q ( v - v ^ { \star } ) - R ^ { \top } ( u - u ^ { \star } ) } \end{array} \Bigr ) , } \end{array}
$$

without loss of generality, we let $( \delta _ { u } , \delta _ { v } ) = \nabla \lambda ( z )$ and

$$
\begin{array} { r l } & { P ( u - u ^ { \star } ) = R ( v - v ^ { \star } ) + \delta _ { u } } \\ & { Q ( v - v ^ { \star } ) = R ^ { \top } ( u - u ^ { \star } ) + \delta _ { v } , } \end{array}
$$

and that $\| \delta _ { u } \| _ { P ^ { - 1 } } \leq \| ( \delta _ { u } , \delta _ { v } ) \| _ { S ^ { - 1 } } = \| \nabla \lambda ( z ) \| _ { S ^ { - 1 } } \leq 6 \sqrt { \varepsilon }$ . Next, we lower bound $\lambda ( z ) - \lambda ( z ^ { \star } )$ and deduce that

$$
\begin{array} { r l } & { \lambda ( u , v ) - \lambda ( u ^ { \star } , v ^ { \star } ) = \frac { 1 } { 2 } \| u - u ^ { \star } \| _ { P } ^ { 2 } - \langle u - u ^ { \star } , R ( v - v ^ { \star } ) \rangle + \frac { 1 } { 2 } \| v - v ^ { \star } \| _ { Q } ^ { 2 } } \\ & { \qquad = \frac { 1 } { 2 } \| \delta _ { u } \| _ { P ^ { - 1 } } ^ { 2 } + \frac { 1 } { 2 } \| v - v ^ { \star } \| _ { Q } ^ { 2 } - \frac { 1 } { 2 } \| v - v ^ { \star } \| _ { R ^ { \tau } P ^ { - 1 } R } ^ { 2 } } \\ & { \qquad = \frac { 1 } { 2 } \| \delta _ { u } \| _ { P ^ { - 1 } } ^ { 2 } + \frac { 1 } { 2 } \| v - v ^ { \star } \| _ { Q - R ^ { \tau } P ^ { - 1 } R } ^ { 2 } , } \end{array}\tag{28}
$$

(29)

where (28) plugs in $u - u ^ { \star } = P ^ { - 1 } ( R ( v - v ^ { \star } ) + \delta _ { u } )$ . Since for any optimal solution $( u ^ { \star } , v ^ { \star } ) , ( u ^ { \star } + \alpha \mathbf { 1 } _ { m } , v ^ { \star } + \alpha \mathbf { 1 } _ { n } )$ is also optimal, we can pick $( u ^ { \star } , v ^ { \star } )$ such that $v - v ^ { \star } \perp q \Leftrightarrow Q ^ { 1 / 2 } ( v - v ^ { \star } ) \perp Q ^ { 1 / 2 } { \bf 1 } _ { n }$ . With this choice, we have

$$
\begin{array} { r l } & { \frac { 1 } { 2 } \| v - v ^ { \star } \| _ { Q - R ^ { \top } P ^ { - 1 } R } ^ { 2 } = \frac { 1 } { 2 } \| Q ^ { 1 / 2 } ( v - v ^ { \star } ) \| _ { I - Q ^ { - 1 / 2 } R ^ { \top } P ^ { - 1 } R Q ^ { - 1 / 2 } } ^ { 2 } } \\ & { \qquad \quad \geq \operatorname* { m i n } _ { \substack { w \perp Q ^ { 1 / 2 } \mathbf { 1 } _ { n } , w \neq 0 } } \frac { 1 } { 2 } \frac { \langle w , ( I - Q ^ { - 1 / 2 } R ^ { \top } P ^ { - 1 } R Q ^ { - 1 / 2 } ) w \rangle } { \| w \| ^ { 2 } } \| v - v ^ { \star } \| _ { Q } ^ { 2 } } \\ & { \qquad = \frac { \sigma _ { 2 } } { 2 } \| v - v ^ { \star } \| _ { Q } ^ { 2 } . } \end{array}\tag{30}
$$

Next, we deduce that

$$
\begin{array} { r l } & { \varepsilon \geq \varphi ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) } \\ & { \quad = \varphi ( u , v ) - \lambda ( u , v ) + \lambda ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) } \\ & { \quad \geq - 3 \sqrt { \frac { \varepsilon } { s } } \| z - z ^ { \star } \| _ { S } ^ { 2 } + \frac { 1 } { 2 } \| \delta _ { u } \| _ { P ^ { - 1 } } ^ { 2 } + \frac { 1 } { 2 } \| v - v ^ { \star } \| _ { Q - R ^ { \top } P ^ { - 1 } R } ^ { 2 } } \\ & { \quad \geq - 3 \sqrt { \frac { \varepsilon } { s } } \| z - z ^ { \star } \| _ { S } ^ { 2 } + \frac { 1 } { 2 } \| \delta _ { u } \| _ { P ^ { - 1 } } ^ { 2 } + \frac { \sigma _ { 2 } } { 2 } \| v - v ^ { \star } \| _ { Q } ^ { 2 } , } \end{array}\tag{31}
$$

(32)

where (31) plugs in (27) and (29); (32) applies (30). Using $P ( u - u ^ { \star } ) = R ( v - v ^ { \star } ) + \delta _ { u }$ , we have

$$
\begin{array} { r l } & { \| u - u ^ { \star } \| _ { P } ^ { 2 } = \| P ^ { - 1 } R ( v - v ^ { \star } ) + P ^ { - 1 } \delta _ { u } \| _ { P } ^ { 2 } } \\ & { \qquad = \langle P ^ { - 1 } R ( v - v ^ { \star } ) + P ^ { - 1 } \delta _ { u } , R ( v - v ^ { \star } ) + \delta _ { u } \rangle } \\ & { \qquad = \| v - v ^ { \star } \| _ { R ^ { \top } P ^ { - 1 } R } ^ { 2 } + 2 \langle P ^ { - 1 / 2 } R ( v - v ^ { \star } ) , P ^ { - 1 / 2 } \delta _ { u } \rangle + \| \delta _ { u } \| _ { P ^ { - 1 } } ^ { 2 } } \\ & { \qquad \leq ( 1 + \theta ) \| v - v ^ { \star } \| _ { R ^ { \top } P ^ { - 1 } R } ^ { 2 } + ( 1 + \frac { 1 } { \theta } ) \| \delta _ { u } \| _ { P ^ { - 1 } } ^ { 2 } } \end{array}
$$

for any $\theta > 0$ . Hence

$$
\begin{array} { r l } & { \| z - z ^ { \star } \| _ { S } ^ { 2 } = \| u - u ^ { \star } \| _ { P } ^ { 2 } + \| v - v ^ { \star } \| _ { Q } ^ { 2 } } \\ & { \qquad \leq ( 1 + \theta ) \| v - v ^ { \star } \| _ { R ^ { \top } P ^ { - 1 } R } ^ { 2 } + \| v - v ^ { \star } \| _ { Q } ^ { 2 } + ( 1 + \frac { 1 } { \theta } ) \| \delta _ { u } \| _ { P ^ { - 1 } } ^ { 2 } } \\ & { \qquad = ( 1 + \theta ) \| Q ^ { 1 / 2 } ( v - v ^ { \star } ) \| _ { Q ^ { - 1 / 2 } R ^ { \top } P ^ { - 1 } R Q ^ { - 1 / 2 } } ^ { 2 } + ( 1 + \frac { 1 } { \theta } ) \| \delta _ { u } \| _ { P ^ { - 1 } } ^ { 2 } } \\ & { \qquad \leq ( 2 + \theta ) \| v - v ^ { \star } \| _ { Q } ^ { 2 } + ( 1 + \frac { 1 } { \theta } ) \| \delta _ { u } \| _ { P ^ { - 1 } } ^ { 2 } . } \end{array}
$$

since $I \succeq Q ^ { - 1 / 2 } R ^ { \top } P ^ { - 1 } R Q ^ { - 1 / 2 }$ . Plugging the relation back into (32), we have

$$
\begin{array} { r l } & { \varepsilon \geq - 3 \sqrt { \frac { \varepsilon } { s } } \| z - z ^ { \star } \| _ { S } ^ { 2 } + \frac { 1 } { 2 } \| \delta _ { u } \| _ { P ^ { - 1 } } ^ { 2 } + \frac { \sigma _ { 2 } } { 2 } \| v - v ^ { \star } \| _ { Q } ^ { 2 } } \\ & { \geq - 3 \sqrt { \frac { \varepsilon } { s } } \big [ ( 2 + \theta ) \| v - v ^ { \star } \| _ { Q } ^ { 2 } + ( 1 + \frac { 1 } { \theta } ) \| \delta _ { u } \| _ { P ^ { - 1 } } ^ { 2 } \big ] + \frac { 1 } { 2 } \| \delta _ { u } \| _ { P ^ { - 1 } } ^ { 2 } + \frac { \sigma _ { 2 } } { 2 } \| v - v ^ { \star } \| _ { Q } ^ { 2 } } \\ & { = \big [ \frac { \sigma _ { 2 } } { 2 } - ( 6 + 3 \theta ) \sqrt { \frac { \varepsilon } { s } } \big ] \| v - v ^ { \star } \| _ { Q } ^ { 2 } + \big [ \frac { 1 } { 2 } - 3 \sqrt { \frac { \varepsilon } { s } } ( 1 + \frac { 1 } { \theta } ) \big ] \| \delta _ { u } \| _ { P ^ { - 1 } } ^ { 2 } } \end{array}
$$

Taking $\theta = 1$ and using $\begin{array} { r } { \varepsilon \le \frac { s \sigma _ { 2 } ^ { 2 } } { 1 2 9 6 } } \end{array}$ , we have

$$
\begin{array} { r } { \frac { 1 } { 2 } - 6 \sqrt { \frac { \varepsilon } { s } } \geq 0 , \quad \mathrm { a n d } \quad 9 \sqrt { \frac { \varepsilon } { s } } \leq \frac { \sigma _ { 2 } } { 4 } , } \end{array}
$$

giving $\begin{array} { r } { \varepsilon \geq \frac { \sigma _ { 2 } } { 4 } \lVert v - v ^ { \star } \rVert _ { Q } ^ { 2 } } \end{array}$ and that $\begin{array} { r } { \| v - v ^ { \star } \| _ { Q } ^ { 2 } \leq \frac { 4 \varepsilon } { \sigma _ { 2 } } } \end{array}$ . Finally,

$$
\begin{array} { r } { \| z - z ^ { \star } \| _ { S } ^ { 2 } \leq 3 \| v - v ^ { \star } \| _ { Q } ^ { 2 } + 2 \| \delta _ { u } \| _ { P ^ { - 1 } } ^ { 2 } \leq \frac { 1 2 } { \sigma _ { 2 } } \varepsilon + 7 2 \varepsilon = 1 2 ( \frac { 1 } { \sigma _ { 2 } } + 6 ) \varepsilon \leq \frac { 8 4 } { \sigma _ { 2 } } \varepsilon . } \end{array}
$$

and that

$$
\begin{array} { r } { | \varphi ( u , v ) - \lambda ( u , v ) | \le ( 2 \sqrt { \frac { \varepsilon } { s } } + \frac { \varepsilon } { s } ) \| z - z ^ { \star } \| _ { S } ^ { 2 } \le \frac { 1 6 8 } { \sigma _ { 2 } } ( \frac { 1 } { \sqrt { s } } \varepsilon ^ { 3 / 2 } + \frac { 1 } { s } \varepsilon ^ { 2 } ) . } \end{array}
$$

This completes the proof.

Remark 7. Symmetrically, we can also show that $\begin{array} { r } { \| u - u ^ { \star } \| _ { P } ^ { 2 } \leq \frac { 4 \varepsilon } { \sigma _ { \mathcal { D } } } } \end{array}$ by picking $u ^ { \star }$ such that $u - u ^ { \star } \perp p$ . But note that we may not simultaneously enforce both $\boldsymbol { u } - \boldsymbol { u } ^ { \star } \perp \boldsymbol { p }$ and $v ^ { \perp } v ^ { \star } \perp q$ with the same $z ^ { \star }$

## A.8 Proof of Lemma 3.5

Let $\varepsilon = \varphi ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } )$ . Fix v and define $\alpha ( \cdot ) : = \varphi ( \cdot , v )$ and $\ell ( \cdot ) : = \lambda ( \cdot , v )$ . We have $\nabla ^ { 2 } \alpha ( u ) = \nabla _ { u u } ^ { 2 } \varphi ( u , v )$ By Lemma 3.3, we have

$$
\ell ( u - P ^ { - 1 } \nabla \ell ( u ) ) - \varphi ( u ^ { \star } , v ^ { \star } ) \leq ( 1 - \sigma _ { 2 } ) [ \ell ( u ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] .
$$

By Lemma 3.4, we have

$$
\begin{array} { r l } & { \| \nabla \alpha ( u ) - \nabla \ell ( u ) \| _ { S ^ { - 1 } } = \| \nabla _ { u } \varphi ( u , v ) - \nabla _ { u } \lambda ( u , v ) \| _ { S ^ { - 1 } } } \\ & { \qquad \le \| \nabla \varphi ( u , v ) - \nabla \lambda ( u , v ) \| _ { S ^ { - 1 } } \le \frac { 2 } { \sqrt { s } } \varepsilon . } \end{array}
$$

Then we deduce that

$$
\begin{array} { r l } { \varphi ( u ^ { + } , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) = \alpha ( u ^ { + } ) - \varphi ( u ^ { \star } , v ^ { \star } ) } & { } \\ { = \underset { u } { \operatorname* { m i n } } \alpha ( u ) - \varphi ( u ^ { \star } , v ^ { \star } ) } & { } \\ { \leq \alpha ( u - P ^ { - 1 } \nabla \alpha ( u ) ) - \varphi ( u ^ { \star } , v ^ { \star } ) } & { } \\ { = \alpha ( u - P ^ { - 1 } \nabla \alpha ( u ) ) - \ell ( u - P ^ { - 1 } \nabla \alpha ( u ) ) + \ell ( u - P ^ { - 1 } \nabla \alpha ( u ) ) - \varphi ( u ^ { \star } , v ^ { \star } ) } & { } \\ { = \underbrace { \alpha ( u - P ^ { - 1 } \nabla \alpha ( u ) ) - \ell ( u - P ^ { - 1 } \nabla \alpha ( u ) ) } _ { \Delta _ { 1 } } + \underbrace { \ell ( u - P ^ { - 1 } \nabla \alpha ( u ) ) - \ell ( u - P ^ { - 1 } \nabla \ell ( u ) ) } _ { \Delta _ { 2 } } } & { } \\ { + \underbrace { \ell ( u - P ^ { - 1 } \nabla \ell ( u ) ) - \varphi ( u ^ { \star } , v ^ { \star } ) } _ { \Delta _ { 3 } } . } & { } \end{array}
$$

Now we bound $\Delta _ { 1 } , \Delta _ { 2 }$ , and $\Delta _ { 3 }$ respectively. For $\Delta _ { 1 } , \mathrm { i f } \alpha ( u - P ^ { - 1 } \nabla \alpha ( u ) ) \le \alpha ( u )$ , then $\begin{array} { r } { \varphi ( u - P ^ { - 1 } \nabla \alpha ( u ) , v ) \le } \end{array}$ $\varphi ( u , v )$ , and we have

$$
\begin{array} { r } { \Delta _ { 1 } \leq \frac { 1 6 8 } { \sigma _ { 2 } } \big ( \frac { 1 } { \sqrt { s } } \varepsilon ^ { 3 / 2 } + \frac { 1 } { s } \varepsilon ^ { 2 } \big ) . } \end{array}
$$

To show $\alpha ( u - P ^ { - 1 } \nabla \alpha ( u ) ) \leq \alpha ( u )$ , we first note that $\nabla ^ { 2 } \alpha ( u ) = \nabla _ { u u } ^ { 2 } \varphi ( u , v )$ and with $\varepsilon \ \leq \ { \frac { s \sigma _ { 2 } ^ { 2 } } { 1 2 9 6 } }$ , we have $\begin{array} { r } { - 2 \sqrt { \frac { \varepsilon } { s } } + \frac { 2 } { s } \varepsilon \leq \frac { 1 } { 2 } } \end{array}$ , which implies

$$
\begin{array} { r l } & { P ^ { - 1 / 2 } \nabla ^ { 2 } \alpha ( u ) P ^ { - 1 / 2 } = P ^ { - 1 / 2 } [ \nabla ^ { 2 } \ell ( u ) + \nabla ^ { 2 } \alpha ( u ) - \nabla ^ { 2 } \ell ( u ) ] P ^ { - 1 / 2 } } \\ & { \qquad \preceq I + \| P ^ { - 1 / 2 } [ \nabla ^ { 2 } \alpha ( u ) - \nabla ^ { 2 } \ell ( u ) ] P ^ { - 1 / 2 } \| I } \\ & { \qquad \preceq I + \| S ^ { - 1 / 2 } [ \nabla ^ { 2 } \varphi ( u , v ) - \nabla ^ { 2 } \lambda ( u , v ) ] S ^ { - 1 / 2 } \| I } \\ & { \qquad \preceq \frac { 3 } { 2 } I } \end{array}
$$

and $\nabla ^ { 2 } \alpha ( u ) \preceq \frac { 3 } { 2 } P$ . Next, consider $\gamma ( \theta ) : = \alpha ( u - \theta P ^ { - 1 } \nabla \alpha ( u ) )$ and let $\theta _ { \mathrm { m a x } } = \operatorname* { s u p } _ { \theta } \{ \theta > 0 : \gamma ( \theta ) = \alpha ( u ) \}$ $\theta _ { \mathrm { m a x } }$ is well-defined since $P ^ { - 1 } \succ 0$ and $P ^ { - 1 } \nabla \alpha ( u )$ is a descent direction of α at u. By convexity, we have $\alpha ( u - \theta P ^ { - 1 } \nabla \alpha ( u ) ) \leq \alpha ( u )$ for all $\theta \leq \theta _ { \mathrm { m a x } }$ . Then we deduce that

$$
\begin{array} { r l } & { \alpha ( u ) = \gamma ( \theta _ { \operatorname* { m a x } } ) } \\ & { \qquad = \alpha ( u ) - \theta _ { \operatorname* { m a x } } \langle \nabla \alpha ( u ) , P ^ { - 1 } \nabla \alpha ( u ) \rangle } \\ & { \qquad + \theta _ { \operatorname* { m a x } } ^ { 2 } \int _ { 0 } ^ { 1 } \langle \nabla \alpha ( u ) , P ^ { - 1 } \nabla ^ { 2 } \alpha ( u - t \theta _ { \operatorname* { m a x } } P ^ { - 1 } \nabla \alpha ( u ) ) P ^ { - 1 } \nabla \alpha ( u ) \rangle ( 1 - t ) \mathrm { d } t } \\ & { \qquad \le \alpha ( u ) - \theta _ { \operatorname* { m a x } } \langle \nabla \alpha ( u ) , P ^ { - 1 } \nabla \alpha ( u ) \rangle + \theta _ { \operatorname* { m a x } } ^ { 2 } \int _ { 0 } ^ { 1 } \langle \nabla \alpha ( u ) , P ^ { - 1 } ( \frac 3 2 P ) P ^ { - 1 } \nabla \alpha ( u ) \rangle ( 1 - t ) \mathrm { d } t , } \\ & { \qquad = \alpha ( u ) - \theta _ { \operatorname* { m a x } } \| \nabla \alpha ( u ) \| _ { P ^ { - 1 } } ^ { 2 } + \frac 3 4 \theta _ { \operatorname* { m a x } } ^ { 2 } \| \nabla \alpha ( u ) \| _ { P ^ { - 1 } } ^ { 2 } , } \end{array}
$$

where the inequality holds since $\alpha ( u - t \theta _ { \mathrm { { m a x } } } P ^ { - 1 } \nabla \alpha ( u ) ) \leq \alpha ( u )$ . Given

$$
\begin{array} { r } { \frac { 3 } { 4 } \theta _ { \mathrm { m a x } } ^ { 2 } \Vert \nabla \alpha ( u ) \Vert _ { P ^ { - 1 } } ^ { 2 } \geq \theta _ { \mathrm { m a x } } \Vert \nabla \alpha ( u ) \Vert _ { P ^ { - 1 } } ^ { 2 } , } \end{array}\tag{33}
$$

we have $\theta _ { \mathrm { { m a x } } } \geq \frac { 4 } { 3 }$ . Therefore, $\alpha ( u - P ^ { - 1 } \nabla \alpha ( u ) ) = \gamma ( 1 ) \leq \gamma ( \theta _ { \operatorname* { m a x } } ) = \alpha ( u )$ , and it implies

$$
\begin{array} { r } { \Delta _ { 1 } \leq \frac { 3 3 6 } { \sigma _ { 2 } \sqrt { s } } [ \varphi ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] ^ { 3 / 2 } . } \end{array}
$$

For $\Delta _ { 2 }$ , since $\ell ( u )$ is a quadratic function minimized at $\boldsymbol { u } - \boldsymbol { P } ^ { - 1 } \nabla \ell ( \boldsymbol { u } )$ , we have

$$
\begin{array} { r l } & { \Delta _ { 2 } = \ell ( u - P ^ { - 1 } \nabla \alpha ( u ) ) - \ell ( u - P ^ { - 1 } \nabla \ell ( u ) ) } \\ & { \quad = \frac { 1 } { 2 } \| \nabla \alpha ( u ) - \nabla \ell ( u ) \| _ { P ^ { - 1 } } ^ { 2 } } \\ & { \quad \le \frac { 1 } { 2 } \| \nabla \varphi ( u , v ) - \nabla \lambda ( u , v ) \| _ { S ^ { - 1 } } ^ { 2 } \le \frac { 2 } { s } \varepsilon ^ { 2 } . } \end{array}
$$

For $\Delta _ { 3 }$ , we have

$$
\begin{array} { r l } & { \ell ( u - P ^ { - 1 } \nabla \ell ( u ) ) - \varphi ( u ^ { \star } , v ^ { \star } ) \leq ( 1 - \sigma _ { 2 } ) [ \ell ( u ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] } \\ & { \qquad = ( 1 - \sigma _ { 2 } ) [ \ell ( u ) - \alpha ( u ) + \alpha ( u ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] } \\ & { \qquad \leq ( 1 - \sigma _ { 2 } ) [ \alpha ( u ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] + | \ell ( u ) - \alpha ( u ) | } \\ & { \qquad \leq ( 1 - \sigma _ { 2 } ) [ \alpha ( u ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] + \frac { 1 6 8 } { \sigma _ { 2 } } ( \frac { 1 } { \sqrt { s } } \varepsilon ^ { 3 / 2 } + \frac { 1 } { s } \varepsilon ^ { 2 } ) } \end{array}
$$

Putting the relations together, we have

$$
\begin{array} { r l } & { \quad \varphi ( u ^ { + } , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) } \\ & { \leq ( 1 - \sigma _ { 2 } ) [ \varphi ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] + \frac { 3 3 6 } { \sqrt { { s } } \sigma _ { 2 } } [ \varphi ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] ^ { 3 / 2 } + \frac { 3 3 8 } { { s } \sigma _ { 2 } } [ \varphi ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] ^ { 2 } , } \end{array}
$$

This completes the whole proof.

## A.9 Proof of Theorem 3.2

For $\begin{array} { r } { \varepsilon \leq \frac { s \sigma _ { 2 } ^ { 4 } } { 1 3 4 4 ^ { 2 } } } \end{array}$ , we have

$$
\begin{array} { r } { \frac { 3 3 6 } { \sigma _ { 2 } \sqrt { s } } [ \varphi ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] ^ { 3 / 2 } \leq \frac { \sigma _ { 2 } } { 4 } [ \varphi ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] . } \end{array}
$$

For $\begin{array} { r } { \varepsilon \le \frac { s \sigma _ { 2 } ^ { 2 } } { 1 3 5 2 } } \end{array}$ , we have

$$
\begin{array} { r } { \frac { 3 3 8 } { s \sigma _ { 2 } } [ \varphi ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] ^ { 2 } \leq \frac { \sigma _ { 2 } } { 4 } [ \varphi ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] . } \end{array}
$$

Hence $\begin{array} { r } { \varphi ( u ^ { + } , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) \le ( 1 - \frac { \sigma _ { 2 } } { 2 } ) [ \varphi ( u , v ) - \varphi ( u ^ { \star } , v ^ { \star } ) ] } \end{array}$ , and it remains to bound

$$
\begin{array} { r l } & { \varphi ( u ^ { 1 } , v ^ { 1 } ) - \varphi ( u ^ { \star } , v ^ { \star } ) = \varphi ( \ \mathbf { 0 } _ { m } , \mathbf { 0 } _ { n } ) - \varphi ( u ^ { \star } , v ^ { \star } ) } \\ & { \qquad \leq \| \nabla \varphi ( \mathbf { 0 } _ { m } , \mathbf { 0 } _ { n } ) \| _ { 1 } \| ( u ^ { \star } , v ^ { \star } ) \| _ { \infty } } \\ & { \qquad \leq [ \| A \mathbf { 1 } _ { m } - p \| _ { 1 } + \| A ^ { \top } \mathbf { 1 } _ { n } - q \| _ { 1 } ] D } \\ & { \qquad \leq 2 [ \| A \| _ { 1 } + \| p \| _ { 1 } ] D . } \end{array}
$$

It a total complexity of

$$
\begin{array} { r l } & { K _ { \varepsilon } \leq \frac { 2 \| p \| _ { 1 } D ^ { 2 } } { \frac { \sigma _ { 2 } ^ { 4 } } { 1 3 4 4 } } + \frac { 2 } { \sigma _ { 2 } } \log ( \frac { 2 [ \| A \| _ { 1 } + \| p \| _ { 1 } ] D } { \varepsilon } ) } \\ & { \qquad = \frac { 2 \cdot 1 3 4 4 ^ { 2 } \| p \| _ { 1 } D ^ { 2 } } { \sigma _ { 2 } ^ { 4 } } + \frac { 2 } { \sigma _ { 2 } } \log ( 2 [ \| A \| _ { 1 } + \| p \| _ { 1 } ] D ) + \frac { 2 } { \sigma _ { 2 } } \log ( \frac { 1 } { \varepsilon } ) } \end{array}
$$

To ensure $\| \nabla \varphi ( u , v ) \| \leq \varepsilon ^ { \prime } ;$ it sufices to have $\begin{array} { r } { \| \nabla \varphi ( u , v ) \| \leq 4 \sqrt { \| p \| _ { 1 } \varepsilon } + 2 \varepsilon \leq \varepsilon ^ { \prime } } \end{array}$ , and it sufices to take

$$
\begin{array} { r } { \varepsilon \le \operatorname* { m i n } \{ \frac { ( \varepsilon ^ { \prime } ) ^ { 2 } } { 6 4 \| p \| _ { 1 } } , \frac { \varepsilon ^ { \prime } } { 4 } \} . } \end{array}
$$

Since each step we analyze is a half-iteration, dividing the complexity by 2 completes the proof.

## A.10 Proof of Theorem 3.2

Fix parameters $\theta \in ( 0 , 1 )$ and $\rho _ { 1 } , \rho _ { 2 } , c _ { 1 } , c _ { 2 } > 0$ . Consider the matrix scaling problem with

$$
\begin{array} { r } { A = \left( \begin{array} { l } { \frac { 1 } { \rho _ { 1 } c _ { 1 } } \ \frac { \theta } { \rho _ { 1 } c _ { 2 } } } \\ { \frac { \theta } { \rho _ { 2 } c _ { 1 } } \ \frac { 1 } { \rho _ { 2 } c _ { 2 } } } \end{array} \right) , \qquad p = q = \left( \begin{array} { l } { 1 + \theta } \\ { 1 + \theta } \end{array} \right) , } \end{array}
$$

so that the solution is given by

$$
\begin{array} { r } { A ^ { \star } = D _ { 1 } ^ { \star } A D _ { 2 } ^ { \star } = \left( \begin{array} { l } { 1 \theta } \\ { \theta 1 } \end{array} \right) \quad \mathrm { w i t h } \quad D _ { 1 } ^ { \star } = \left( \begin{array} { l l } { \rho _ { 1 } } & { 0 } \\ { 0 } & { \rho _ { 2 } } \end{array} \right) , \qquad D _ { 2 } ^ { \star } = \left( \begin{array} { l l } { c _ { 1 } } & { 0 } \\ { 0 } & { c _ { 2 } } \end{array} \right) . } \end{array}
$$

Let $M = P ^ { - 1 / 2 } A ^ { \star } Q ^ { - 1 } ( A ^ { \star } ) ^ { \top } P ^ { - 1 / 2 }$ and $L = I - M .$ . Since $P = Q = ( 1 + \theta ) I ;$

$$
\begin{array} { r } { M = \frac { A ^ { \star } ( A ^ { \star } ) ^ { \top } } { ( 1 + \theta ) ^ { 2 } } = \frac { 1 } { ( 1 + \theta ) ^ { 2 } } \left( { 1 + \theta ^ { 2 } } \begin{array} { c } { 2 \theta } \\ { 2 \theta } \end{array} 1 + \theta ^ { 2 } \right) . } \end{array}
$$

Its top eigenvector is $\mathbf { 1 } _ { 2 }$ with eigenvalue 1, and its second largest eigenvalue is

$$
\begin{array} { r } { \mu _ { 2 } = \operatorname { t r } ( M ) - 1 = \frac { 2 ( 1 + \theta ^ { 2 } ) - ( 1 + \theta ) ^ { 2 } } { ( 1 + \theta ) ^ { 2 } } = ( \frac { 1 - \theta } { 1 + \theta } ) ^ { 2 } . } \end{array}
$$

Therefore,

$$
\begin{array} { r } { \sigma _ { 2 } ( \theta ) = \lambda _ { 2 } ( L ) = 1 - \mu _ { 2 } = \frac { 4 \theta } { ( 1 + \theta ) ^ { 2 } } . } \end{array}
$$

By the additive shift invariance of the dual potential, we can parametrize the deviation from the optimum by a single scalar s, taking $u ( s ) = u ^ { \star } + ( s , - s )$ and letting $v ( s )$ be the exact minimizer $v ( s ) = \arg \operatorname* { m i n } _ { v } \varphi ( u ( s ) , v )$ (a half Sinkhorn step). Next, starting at A and $( u , v )$ is equivalent to starting at $A ^ { \star }$ and $( u - u ^ { \star } , v - v ^ { \star } )$ , with identical dual potential gaps and imbalance sequences. We adopt this change of variables, and substituting v(s) into the dual potential gap $G ( s ) : = \varphi ( u ( s ) , v ( s ) ) - \varphi ( u ^ { \star } , v ^ { \star } )$ gives the closed form

$$
\begin{array} { r } { G ( s ) = ( 1 + \theta ) \log { \frac { ( 1 + \theta ) ^ { 2 } } { ( e ^ { s } + \theta e ^ { - s } ) ( \theta e ^ { s } + e ^ { - s } ) } } = \frac { 4 \theta } { 1 + \theta } s ^ { 2 } + O ( s ^ { 4 } ) } \end{array}
$$

In particular, G is locally a positive-definite quadratic in the imbalance s. Now we proceed to updating u from $v ( s )$ by $u ^ { + } = \log p - \log ( A ^ { \star } \mathrm { e } ^ { - v ( s ) } )$ , and compute $\begin{array} { r } { s ^ { + } : = \frac { 1 } { 2 } \log ( \mathrm { e } ^ { u _ { 1 } ^ { + } } / \mathrm { e } ^ { u _ { 2 } ^ { + } } ) } \end{array}$ . A direct diferentiation of the mapping $s \to s ^ { + }$ at $s = 0$ gives, for a single half-step,

$$
\begin{array} { r } { s ^ { + } = - \big ( \frac { 1 - \theta } { 1 + \theta } \big ) ^ { 2 } s + \mathcal { O } ( s ^ { 2 } ) = - ( 1 - \sigma _ { 2 } ) s + \mathcal { O } ( s ^ { 2 } ) , } \end{array}
$$

and substituting into $G ( s )$ , the dual potential gap $G _ { k + 1 } = G ( s ^ { + } )$ after one full Sinkhorn iteration is given by

$$
G _ { k + 1 } = ( 1 - \sigma _ { 2 } ) ^ { 2 } G _ { k } ( 1 + o ( 1 ) ) .
$$

This matches Lemma 3.5 with equality, confirming that the local factor $( 1 - \sigma _ { 2 } ) ^ { 2 }$ per iteration is attained on this family and cannot be improved. In conclusion, after a finite, ε-independent burn-in c(θ) (controlled by Theorem 3.1 and the threshold of Lemma 3.5) the iterates enter the local regime, where $\log ( 1 / G _ { k } )$ grows by at most $- 2 \log ( 1 - \sigma _ { 2 } ) + o ( 1 ) = 2 \sigma _ { 2 } + o ( \sigma _ { 2 } )$ per cycle. Hence reaching $G _ { k } \leq \varepsilon$ requires

$$
\begin{array} { r } { k \ge c ( \theta ) + \frac { \log ( 1 / \varepsilon ) } { - 2 \log ( 1 - \sigma _ { 2 } ) } = c ( \theta ) + \frac { 1 } { 2 \sigma _ { 2 } ( \theta ) } \log ( \frac { 1 } { \varepsilon } ) ( 1 + o ( 1 ) ) , } \end{array}
$$

matching the upper bound $\begin{array} { r } { \mathcal { O } \bigl ( c + \frac { 1 } { \sigma _ { 2 } } \log \bigl ( \frac { 1 } { \varepsilon } \bigr ) \bigr ) } \end{array}$ of Theorem 3.2 up to the additive constant.

## B Proof of results in Section 4

## B.1 Auxiliary results

Lemma B.1. Suppose f is L-smooth and convex. Then there exists an algorithm that outputs $x ^ { K }$ such that $\| \nabla f ( x ^ { K } ) \| ^ { 2 } \leq \varepsilon$ in $\begin{array} { r } { K _ { \varepsilon } : = \lceil ( \frac { 5 2 8 L ^ { 2 } \| x ^ { 1 } - x ^ { \star } \| ^ { 2 } } { \varepsilon } ) ^ { 1 / 4 } \rceil } \end{array}$ iterations, where $x ^ { \star }$ is an optimal solution.

Proof. The algorithm is $\mathtt { F I S T A } + \mathtt { F I S T A - G }$ . Invoking Corollary 1 from [26] completes the proof. □

Lemma B.2. Suppose $\begin{array} { r } { f ( x ) = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } f _ { j } ( x ) } \end{array}$ with each $f _ { j }$ is $L _ { \mathrm { m a x } }$ -smooth and σ-strongly convex. Then there exists an algorithm that outputs $x ^ { K }$ such that $\begin{array} { r } { \mathbb { E } [ f ( x ^ { K } ) ] - f ( x ^ { \star } ) \le \varepsilon \ i n \ K _ { \varepsilon } : = \mathcal { O } \big ( \big ( n + \textstyle \sqrt { \frac { n L _ { \operatorname* { m a x } } } { \sigma } } \big ) \log \big ( \frac { f ( x ^ { 1 } ) - f ( x ^ { \star } ) } { \varepsilon } \big ) \big ) } \end{array}$ stochastic gradient oracle calls.

Proof. The algorithm is Katyusha [1]. Invoking Theorem 2.1 of [1] completes the proof.

## B.2 Proof of Lemma 4.1

Let $a _ { [ : , j ] }$ denote the j-th column of A. Through direct calculation, we have

$$
\begin{array} { r l r } {  { \zeta ( u ) = \varphi ( u , - \log ( q / A ^ { \top } \mathrm { e } ^ { u } ) ) } } \\ & { } & { = \sum _ { j = 1 } ^ { n } q _ { j } \log \langle a _ { [ : , j ] } , \mathrm { e } ^ { u } \rangle - \langle p , u \rangle - \sum _ { j = 1 } ^ { n } q _ { j } \log q _ { j } + \langle \mathbf { 1 } _ { n } , q \rangle } \end{array}
$$

and that

$$
\begin{array} { r l } & { \nabla \zeta ( u ) = \sum _ { j = 1 } ^ { n } \frac { q _ { j } } { \langle a _ { [ : , j ] } , \mathrm { e } ^ { u } \rangle } \mathrm { e } ^ { U } a _ { [ : , j ] } - p } \\ & { \nabla ^ { 2 } \zeta ( u ) = \mathcal { D } ( \sum _ { j = 1 } ^ { n } q _ { j } \frac { \mathrm { e } ^ { U } a _ { [ : , j ] } } { \langle a _ { [ : , j ] } , \mathrm { e } ^ { u } \rangle } ) - \sum _ { j = 1 } ^ { n } q _ { j } \frac { \mathrm { e } ^ { U } a _ { [ : , j ] } a _ { [ : , j ] } ^ { \top } \mathrm { e } ^ { U } } { \langle a _ { [ : , j ] } , \mathrm { e } ^ { u } \rangle ^ { 2 } } } \\ & { \qquad = \mathcal { D } ( \sum _ { j = 1 } ^ { n } q _ { j } \frac { \mathrm { e } ^ { U } a _ { [ : , j ] } } { \langle \mathbf { 1 } _ { m } , \mathrm { e } ^ { U } a _ { [ : , j ] } \rangle } ) - \sum _ { j = 1 } ^ { n } q _ { j } \frac { \mathrm { e } ^ { U } a _ { [ : , j ] } a _ { [ : , j ] } ^ { \top } \mathrm { e } ^ { U } } { \langle \mathbf { 1 } _ { m } , \mathrm { e } ^ { U } a _ { [ : , j ] } \rangle ^ { 2 } } } \\ & { \qquad = : \sum _ { j = 1 } ^ { n } q _ { j } [ \mathcal { D } ( \sigma _ { j } ) - \sigma _ { j } \sigma _ { j } ^ { \top } ] , } \end{array}
$$

where we define $\begin{array} { r } { \sigma _ { j } : = \frac { \mathrm { e } ^ { U } a _ { [ : , j ] } } { \langle \mathbf { 1 } _ { m } , \mathrm { e } ^ { U } a _ { [ : , j ] } \rangle } } \end{array}$ and ${ \bf 0 } _ { m } \le \sigma _ { j } \le { \bf 1 } _ { m }$ by nonnegativity of A and $\mathrm { e } ^ { U }$ , and the fact that A must have at least nonzero per row. To establish smoothness, we have

$$
\begin{array} { r } { \| \nabla ^ { 2 } \zeta ( u ) \| \le \sum _ { j = 1 } ^ { n } q _ { j } \| \mathcal { D } ( \sigma _ { j } ) - \sigma _ { j } \sigma _ { j } ^ { \top } \| \le \frac { \| q \| _ { 1 } } { 2 } = \frac { \| p \| _ { 1 } } { 2 } , } \end{array}
$$

where we use Lipschitzness of softmax functions $[ 3 0 ] \colon \| \mathcal D ( x ) - x x ^ { \top } \| \leq \frac { 1 } { 2 }$ for $x \in \left\{ x \geq \mathbf { 0 } _ { m } : \left. \mathbf { 1 } _ { m } , x \right. = 1 \right\}$ . To establish Lipschitz Hessian, we similarly deduce, for any x, y, that

$$
\begin{array} { r l } { \| \nabla ^ { 2 } \zeta ( x ) - \nabla ^ { 2 } \zeta ( y ) \| = \| \sum _ { j = 1 } ^ { n } q _ { j } [ \mathcal { D } ( \sigma _ { j } ) - \sigma _ { j } \sigma _ { j } ^ { \top } ] - \sum _ { j = 1 } ^ { n } q _ { j } [ \mathcal { D } ( \sigma _ { j } ^ { \prime } ) - \sigma _ { j } ^ { \prime } ( \sigma _ { j } ^ { \prime } ) ^ { \top } ] \| } & { } \\ { \leq \sum _ { j = 1 } ^ { n } q _ { j } \| \sigma _ { j } - \sigma _ { j } ^ { \prime } \| _ { \infty } + \sum _ { j = 1 } ^ { n } q _ { j } \| \sigma _ { j } \sigma _ { j } ^ { \top } - \sigma _ { j } ^ { \prime } ( \sigma _ { j } ^ { \prime } ) ^ { \top } \| } & { } \\ { = \sum _ { j = 1 } ^ { n } q _ { j } \| \sigma _ { j } - \sigma _ { j } ^ { \prime } \| _ { \infty } + \sum _ { j = 1 } ^ { n } q _ { j } \| \sigma _ { j } \sigma _ { j } ^ { \top } - \sigma _ { j } ( \sigma _ { j } ^ { \prime } ) ^ { \top } + \sigma _ { j } ( \sigma _ { j } ^ { \prime } ) ^ { \top } - \sigma _ { j } ^ { \prime } ( \sigma _ { j } ^ { \prime } ) ^ { \top } | } & { } \\ { \leq \sum _ { j = 1 } ^ { n } q _ { j } \| \sigma _ { j } - \sigma _ { j } ^ { \prime } \| _ { \infty } + \sum _ { j = 1 } ^ { n } q _ { j } [ \| \sigma _ { j } \| + \| \sigma _ { j } ^ { \prime } \| ] \| \sigma _ { j } - \sigma _ { j } ^ { \prime } \| } & { } \\ { \leq 3 \sum _ { j = 1 } ^ { n } q _ { j } \| \sigma _ { j } - \sigma _ { j } ^ { \prime } \| , } & { } \end{array}\tag{34}
$$

where $\begin{array} { r } { \sigma _ { j } = \frac { \mathrm { e } ^ { X } a _ { [ : , j ] } } { \langle \mathbf { 1 } _ { m } , \mathrm { e } ^ { X } a _ { [ : , j ] } \rangle } } \end{array}$ and $\begin{array} { r } { \sigma _ { j } ^ { \prime } = \frac { \mathrm { e } ^ { Y } a _ { [ : , j ] } } { \langle \mathbf { 1 } _ { m } , \mathrm { e } ^ { Y } a _ { [ : , j ] } \rangle } } \end{array}$ are both outcomes of weighted softmax; (34) uses the fact that $\| \sigma _ { j } \| \leq 1$ , and Lipschitzness of softmax implies

$$
\begin{array} { r } { \| \sigma _ { j } - \sigma _ { j } ^ { \prime } \| \leq \frac { 1 } { 2 } \| x - y \| . } \end{array}
$$

Plugging back, we have $\begin{array} { r } { \| \nabla ^ { 2 } \zeta ( x ) - \nabla ^ { 2 } \zeta ( y ) \| \leq \frac { 3 } { 2 } \sum _ { i = 1 } ^ { n } q _ { j } \| x - y \| = \frac { 3 } { 2 } \| p \| _ { 1 } \| x - y \| } \end{array}$ . Finally, the quadratic growth $\begin{array} { r } { \frac { \sigma _ { 2 } } { 4 } \| \Pi ( u - u ^ { \star } ) \| _ { P } ^ { 2 } \leq \zeta ( u ) - \zeta ( u ^ { \star } ) } \end{array}$ follows from the symmetric argument in Remark 7. To show the PL inequality,

we deduce that, taking $u ^ { \star }$ such that $\Pi ( u - u ^ { \star } ) = u - u ^ { \star }$ , that,

$$
\zeta ( u ) - \zeta ( u ^ { \star } ) \leq \langle \nabla \zeta ( u ) , u - u ^ { \star } \rangle\tag{35}
$$

$$
= \langle P ^ { - 1 / 2 } \nabla \zeta ( u ) , P ^ { 1 / 2 } ( u - u ^ { \star } ) \rangle
$$

$$
\leq \| \nabla \zeta ( u ) \| _ { P ^ { - 1 } } \| \Pi ( u - u ^ { \star } ) \| _ { P }\tag{36}
$$

$$
\leq \| \nabla \zeta ( u ) \| _ { P ^ { - 1 } } \sqrt { \frac { 4 } { \sigma _ { 2 } } [ \zeta ( u ) - \zeta ( u ^ { \star } ) ] } ,\tag{37}
$$

where (35) uses convexity of $\zeta ; ( 3 6 )$ uses Cauchy-Schwarz and (37) uses quadratic growth. Squaring and dividing both sides by $\zeta ( u ) - \zeta ( u ^ { \star } )$ gives the desired relation. Finally, with Lemma 3.4, we have

$$
\begin{array} { r } { \| \nabla \zeta ( u ) \| _ { P ^ { - 1 } } ^ { 2 } = \| \nabla \varphi ( u , v ( u ) ) \| _ { S ^ { - 1 } } ^ { 2 } \leq 4 \varepsilon + \frac { 8 } { \sqrt { s } } \varepsilon ^ { 3 / 2 } + \frac 4 s \varepsilon ^ { 2 } } \end{array}
$$

and with $\varepsilon \leq s .$ , we have $\frac { 8 } { \sqrt { s } } \varepsilon ^ { 3 / 2 } \leq 8 \varepsilon$ and $\textstyle { \frac { 4 } { s } } \varepsilon ^ { 2 } \leq 4 \varepsilon$ , giving $\| \nabla \zeta ( u ) \| _ { P ^ { - 1 } } ^ { 2 } \leq 1 6 \varepsilon = 1 6 [ \zeta ( u ) - \zeta ( u ^ { \star } ) ]$ . To establish lower and upper bounds on the spectrum, using $\begin{array} { r } { \| \Pi ( u - u ^ { \star } ) \| _ { P } \leq \sqrt { \frac { 4 \varepsilon } { \sigma _ { 2 } } } } \end{array}$ and by Weyl’s inequality,

$$
\begin{array} { r } { \lambda _ { 2 } ( P ^ { - 1 / 2 } \nabla ^ { 2 } \zeta ( u ) P ^ { - 1 / 2 } ) \ge \lambda _ { 2 } ( P ^ { - 1 / 2 } \nabla ^ { 2 } \zeta ( u ^ { \star } ) P ^ { - 1 / 2 } ) - \frac { H } { s } \| \Pi ( u - u ^ { \star } ) \| _ { P } \ge \sigma _ { 2 } - \sqrt { \frac { 4 H ^ { 2 } \varepsilon } { s ^ { 2 } \sigma _ { 2 } } } . } \end{array}
$$

$$
\begin{array} { r } { \lambda _ { m } ( P ^ { - 1 / 2 } \nabla ^ { 2 } \zeta ( u ) P ^ { - 1 / 2 } ) \le \lambda _ { m } \bigl ( P ^ { - 1 / 2 } \nabla ^ { 2 } \zeta ( u ^ { \star } ) P ^ { - 1 / 2 } \bigr ) + \frac { H } { s } \| \Pi ( u - u ^ { \star } ) \| _ { P } \le 1 + \sqrt { \frac { 4 H ^ { 2 } \varepsilon } { s ^ { 2 } \sigma _ { 2 } } } } \end{array}
$$

Hence for $\begin{array} { r } { \varepsilon \leq \frac { \sigma _ { 2 } ^ { 3 } s ^ { 2 } } { 1 6 H ^ { 2 } } , \sigma _ { 2 } - \sqrt { \frac { 4 H ^ { 2 } \varepsilon } { s ^ { 2 } \sigma _ { 2 } } } \geq \frac { \sigma _ { 2 } } { 2 } } \end{array}$ and this completes the proof.

## B.3 Proof of Theorem 4.1

Given that ζ is L-smooth convex with $L = { \frac { 1 } { 2 } } \| p \| _ { 1 }$ , invoking Lemma B.1 with $L = { \frac { 1 } { 2 } } \| p \| _ { 1 }$ completes the proof. The inequality uses $\| u ^ { \star } \| \leq \sqrt { n } \| u ^ { \star } \| _ { \infty }$ . Computing $\nabla \zeta ( u )$ involves finding $v = p / A ^ { \top } u$ whose arithmetic complexity is the same as half of a SK iteration.

## B.4 Proof of Theorem 4.2

It sufices to adopt Nesterov’s regularization technique for making gradient small [32]. Define

$$
\begin{array} { r } { \zeta _ { \sigma } ( u ) : = \zeta ( u ) + \frac { \sigma } { 2 } \| u \| ^ { 2 } . } \end{array}
$$

Since $\zeta$ is L-smooth and convex, $\zeta _ { \sigma }$ is $( L + \sigma )$ -smooth and σ-strongly convex. Denote $\begin{array} { r } { u _ { \sigma } ^ { \star } = \arg \operatorname* { m i n } _ { u } \zeta _ { \sigma } ( u ) } \end{array}$ . By the optimality condition, $\nabla \zeta _ { \sigma } ( u _ { \sigma } ^ { \star } ) = \nabla \zeta ( u _ { \sigma } ^ { \star } ) + \sigma u _ { \sigma } ^ { \star } = 0$ and with quadratic growth,

$$
\begin{array} { r } { \zeta _ { \sigma } ( u ^ { \star } ) - \zeta _ { \sigma } ( u _ { \sigma } ^ { \star } ) \geq \frac { \sigma } { 2 } \| u _ { \sigma } ^ { \star } - u ^ { \star } \| ^ { 2 } . } \end{array}
$$

By definition, the above relation implies

$$
\begin{array} { r } { \frac { \sigma } { 2 } \| u _ { \sigma } ^ { \star } - u ^ { \star } \| ^ { 2 } \leq \zeta _ { \sigma } ( u ^ { \star } ) - \zeta _ { \sigma } ( u _ { \sigma } ^ { \star } ) = \zeta ( u ^ { \star } ) + \frac { \sigma } { 2 } \| u ^ { \star } \| ^ { 2 } - \zeta ( u _ { \sigma } ^ { \star } ) - \frac { \sigma } { 2 } \| u _ { \sigma } ^ { \star } \| ^ { 2 } \leq \frac { \sigma } { 2 } \| u ^ { \star } \| ^ { 2 } - \frac { \sigma } { 2 } \| u _ { \sigma } ^ { \star } \| ^ { 2 } } \end{array}
$$

since $\zeta ( u ^ { \star } ) \leq \zeta ( u _ { \sigma } ^ { \star } )$ . Rearranging, we have $\| u _ { \sigma } ^ { \star } \| ^ { 2 } \leq \| u _ { \sigma } ^ { \star } - u ^ { \star } \| ^ { 2 } + \| u _ { \sigma } ^ { \star } \| ^ { 2 } \leq \| u ^ { \star } \| ^ { 2 }$ and $\| \nabla \zeta ( u _ { \sigma } ^ { \star } ) \| = \sigma \| u _ { \sigma } ^ { \star } \| \leq$ $\sigma \| u ^ { \star } \|$ . Now suppose we run Katyusha on $\zeta _ { \sigma }$ , which, by Lemma B.2, outputs uˆ such that

$$
\mathbb { E } [ \zeta _ { \sigma } ( \hat { u } ) ] - \zeta _ { \sigma } ( u _ { \sigma } ^ { \star } ) \leq \hat { \varepsilon }
$$

in $\begin{array} { r } { \mathcal { O } \big ( \big ( n + \sqrt { \frac { n L _ { \mathrm { m a x } } } { \sigma } } \big ) \log \big ( \frac { \zeta _ { \sigma } ( u ^ { 1 } ) - \zeta _ { \sigma } ( u _ { \sigma } ^ { \star } ) } { \hat { \varepsilon } } \big ) \big ) } \end{array}$ stochastic gradient calls, where $\textstyle L _ { \mathrm { m a x } } = { \frac { n } { 2 } } \| p \| _ { \infty }$ . Now we deduce that

$$
\mathbb { E } [ \| \nabla \zeta ( \hat { u } ) \| ] = \mathbb { E } [ \| \nabla \zeta ( \hat { u } ) - \nabla \zeta ( u _ { \sigma } ^ { \star } ) + \nabla \zeta ( u _ { \sigma } ^ { \star } ) \| ]
$$

$$
\leq \| \nabla \zeta ( u _ { \sigma } ^ { \star } ) \| + \mathbb { E } [ \| \nabla \zeta ( \hat { u } ) - \nabla \zeta ( u _ { \sigma } ^ { \star } ) \| ]\tag{38}
$$

$$
\leq \sigma \| u ^ { \star } \| + L \mathbb { E } [ \| \hat { u } - u _ { \sigma } ^ { \star } \| ]\tag{39}
$$

$$
\leq \sigma \| u ^ { \star } \| + L \sqrt { \mathbb { E } [ \| \hat { u } - u _ { \sigma } ^ { \star } \| ^ { 2 } ] }\tag{40}
$$

$$
\begin{array} { r } { \leq \sigma \| u ^ { \star } \| + L \sqrt { \frac { 2 } { \sigma } \mathbb { E } [ \zeta _ { \sigma } ( \hat { u } ) - \zeta _ { \sigma } ( u _ { \sigma } ^ { \star } ) ] } } \end{array}\tag{41}
$$

$$
\begin{array} { r } { = \sigma \| u ^ { \star } \| + L \sqrt { \frac { 2 } { \sigma } \hat { \varepsilon } } , } \end{array}
$$

where (38) uses triangle inequality; (39) uses the previous relation $\| \nabla \zeta ( u _ { \sigma } ^ { \star } ) \| \leq \sigma \| u ^ { \star } \|$ and L-smoothness; (40) uses Jensen’s inequality $\mathbb { E } [ | X | ] \le \sqrt { \mathbb { E } [ X ^ { 2 } ] } ;$ ; (41) uses quadratic growth. Taking $\sigma = \frac { \varepsilon } { 2 \| u ^ { \star } \| }$ and $\begin{array} { r } { \hat { \varepsilon } = \frac { \sigma } { 8 L ^ { 2 } } = } \end{array}$ $\frac { \varepsilon } { 1 6 L ^ { 2 } \left\| u ^ { \star } \right\| }$ , we have

$$
\begin{array} { r } { \mathbb { E } [ \| \nabla \zeta ( \hat { u } ) \| ] \le \frac { \varepsilon } { 2 \| u ^ { \star } \| } \| u ^ { \star } \| + L \sqrt { \frac { 2 } { \sigma } \frac { \sigma } { 8 L ^ { 2 } } } = \frac { \varepsilon } { 2 } + \frac { \varepsilon } { 2 } = \varepsilon , } \end{array}
$$

which gives a total complexity of

$$
\begin{array} { r l } & { \mathcal { O } \big ( \big ( n + \sqrt { \frac { n L _ { \operatorname* { m a x } } } { \sigma } } \big ) \log \big ( \frac { \zeta _ { \sigma } ( u ^ { 1 } ) - \zeta _ { \sigma } ( u _ { \sigma } ^ { * } ) } { \xi } \big ) \big ) = \mathcal { O } \big ( \big ( n + \sqrt { \frac { 2 n L _ { \operatorname* { m a x } } \| u ^ { * } \| } { \varepsilon } } \big ) \log \big ( \frac { 1 6 L ^ { 2 } \| u ^ { * } \| \zeta _ { \sigma } ( u ^ { 1 } ) - \zeta _ { \sigma } ( u _ { \sigma } ^ { * } ) \| } { \varepsilon } \big ) \big ) } \\ & { \quad \quad \quad \quad \quad = \mathcal { O } \big ( \big ( n + n ^ { 3 / 4 } \sqrt { \frac { 2 L _ { \operatorname* { m a x } } \| u ^ { * } \| _ { \infty } } { \varepsilon } } \big ) \log \big ( \frac { 1 6 L ^ { 2 } \| u ^ { * } \| \zeta _ { \sigma } ( u ^ { 1 } ) - \zeta _ { \sigma } ( u _ { \sigma } ^ { * } ) \| } { \varepsilon } \big ) \big ) } \\ & { \quad \quad \quad \quad = \tilde { \mathcal { O } } \big ( n + n ^ { 3 / 4 } \sqrt { \frac { 2 L _ { \operatorname* { m a x } } \| u ^ { * } \| _ { \infty } } { \varepsilon } } \big ) , } \end{array}
$$

where we hide the terms in log since with $\boldsymbol { u } ^ { 1 } = \mathbf { 0 }$ , we have

$$
\begin{array} { r } { \zeta _ { \sigma } ( u ^ { 1 } ) - \zeta _ { \sigma } ( u _ { \sigma } ^ { \star } ) = \zeta ( \mathbf { 0 } _ { m } ) - \zeta _ { \sigma } ( u _ { \sigma } ^ { \star } ) \leq \zeta ( \mathbf { 0 } _ { m } ) - \zeta ( u _ { \sigma } ^ { \star } ) \leq \zeta ( \mathbf { 0 } _ { m } ) - \zeta ( u ^ { \star } ) \leq \| u ^ { \star } \| _ { \infty } \| \nabla \zeta ( \mathbf { 0 } _ { m } ) \| _ { 1 } \leq \| u ^ { \star } \| _ { \infty } ( \| p \| _ { 1 } + \| A \| _ { 1 } ) } \end{array}
$$

Plugging in $\textstyle L _ { \mathrm { m a x } } = { \frac { n } { 2 } } \| p \| _ { \infty }$ gives $\begin{array} { r } { \tilde { \mathcal { O } } ( n + n ^ { 5 / 4 } \sqrt { \frac { \| p \| _ { \infty } \| u ^ { \star } \| _ { \infty } } { \varepsilon } } ) } \end{array}$ complexity of stochastic gradients. Given that the amortized complexity of each iteration is ${ \mathcal { O } } ( { \frac { \mathrm { n n } \mathbf { \dot { z } } ( A ) } { n } } )$ , we have the expected total arithmetic complexity given by

$$
\begin{array} { r } { \tilde { \mathcal { O } } ( \mathrm { n n z } ( A ) + \mathrm { n n z } ( A ) \cdot n ^ { 1 / 4 } \sqrt { \frac { \| p \| _ { \infty } \| u ^ { \star } \| _ { \infty } } { \varepsilon } } ) . } \end{array}
$$

Noticing that $\| u ^ { \star } \| \leq D$ , this completes the proof.

## B.5 Proof of Lemma 4.2

Convexity of $h _ { u }$ follows from the composition rule of convex functions. To show smoothness, it sufices to show that $P ^ { - 1 / 2 } \nabla ^ { 2 } h _ { u } ( w ) P ^ { - 1 / 2 } \preceq L \cdot I$ , which holds since

$$
\begin{array} { r } { \nabla ^ { 2 } h _ { u } ( w ) = \mathcal { D } \big ( \frac { \nabla \zeta ( u ) } { \| P ^ { - 1 / 2 } \nabla \zeta ( u ) \| } \big ) \nabla ^ { 2 } \zeta ( u - \mathcal { D } ( w ) \nabla \zeta ( u ) ) \mathcal { D } \big ( \frac { \nabla \zeta ( u ) } { \| P ^ { - 1 / 2 } \nabla \zeta ( u ) \| } \big ) } \end{array}
$$

and for some wˆ, we have

$$
\begin{array} { r } { P ^ { - 1 / 2 } { \nabla ^ { 2 } } h _ { u } ( w ) P ^ { - 1 / 2 } = \mathcal { D } \big ( \frac { P ^ { - 1 / 2 } \nabla \zeta ( u ) } { \| P ^ { - 1 / 2 } \nabla \zeta ( u ) \| } \big ) \nabla ^ { 2 } \zeta ( \hat { w } ) \mathcal { D } \big ( \frac { P ^ { - 1 / 2 } \nabla \zeta ( u ) } { \| P ^ { - 1 / 2 } \nabla \zeta ( u ) \| } \big ) \preceq L \cdot I . } \end{array}
$$

Given $\begin{array} { r } { \zeta ( u ) - \zeta ( u ^ { \star } ) \leq \frac { s \sigma _ { 2 } ^ { 2 } } { 1 2 9 6 } } \end{array}$ , by Lemma 3.4, $\begin{array} { r } { \| \nabla \zeta ( u ) \| _ { P ^ { - 1 } } = \| \nabla \varphi ( u , v ( u ) ) \| _ { S ^ { - 1 } } \leq 2 \sqrt { \varepsilon } + \frac { 2 } { \sqrt { s } } \varepsilon } \end{array}$ and

$$
\begin{array} { r l } & { P ^ { - 1 / 2 } \nabla ^ { 2 } \zeta ( u ) P ^ { - 1 / 2 } \preceq P ^ { - 1 / 2 } ( { \cal D } ( \nabla \zeta ( u ) + p ) ) P ^ { - 1 / 2 } } \\ & { \preceq { \cal D } ( P ^ { - 1 } \nabla \zeta ( u ) ) + I } \\ & { \preceq \frac { 1 } { \sqrt { s } } \| \nabla \zeta ( u ) \| _ { P ^ { - 1 } } I + I } \\ & { \preceq ( \frac { 2 } { \sqrt { s } } \sqrt { \varepsilon } + \frac { 2 } { s } \varepsilon + 1 ) I \preceq \frac 3 2 I } \end{array}
$$

Denote $u _ { t } : = u - t P ^ { - 1 } \nabla \zeta ( u )$ . With the same reasoning as (33), we can show that $\zeta ( u - P ^ { - 1 } \nabla \zeta ( u ) ) \leq \zeta ( u )$ and $\zeta ( u _ { t } ) - \zeta ( u ^ { \star } ) \leq \varepsilon$ for all t, giving $\begin{array} { r } { \nabla ^ { 2 } \zeta ( u _ { t } ) \preceq \frac { 3 } { 2 } P } \end{array}$ . Then we deduce that

$$
\begin{array} { r l } { \| \nabla \zeta ( u ) \| _ { P ^ { - 1 } } ^ { 2 } h _ { u } ( P ^ { - 1 } \mathbf { 1 } _ { m } ) = \zeta ( u - P ^ { - 1 } \nabla \zeta ( u ) ) - \zeta ( u ) } & { } \\ { = - \left. \nabla \zeta ( u ) , P ^ { - 1 } \nabla \zeta ( u ) \right. + \int _ { 0 } ^ { 1 } ( 1 - t ) \langle P ^ { - 1 } \nabla \zeta ( u ) , \nabla ^ { 2 } \zeta ( u _ { t } ) P ^ { - 1 } \nabla \zeta ( u ) \rangle \mathrm { d } t } & { } \\ { \leq - \| \nabla \zeta ( u ) \| _ { P ^ { - 1 } } ^ { 2 } + \int _ { 0 } ^ { 1 } ( 1 - t ) \langle P ^ { - 1 } \nabla \zeta ( u ) , ( \frac { 3 } { 2 } P ) P ^ { - 1 } \nabla \zeta ( u ) \rangle \mathrm { d } t } & { } \\ { = - \| \nabla \zeta ( u ) \| _ { P ^ { - 1 } } ^ { 2 } + \frac { 3 } { 4 } \| \nabla \zeta ( u ) \| _ { P ^ { - 1 } } ^ { 2 } = - \frac { 1 } { 4 } \| \nabla \zeta ( u ) \| _ { P ^ { - 1 } } ^ { 2 } . } \end{array}
$$

Dividing both sides by $\| \nabla \zeta ( u ) \| _ { P ^ { - } } ^ { 2 }$ <sub>1</sub> completes the proof.

## B.6 Proof of Lemma 4.3

A good scaling vector w makes $h _ { u } ( w )$ small. Since $h _ { u }$ is convex, its negative gradient aligns with any direction that points some scaling matrix better than w being used, say wˆ:

$$
h _ { u } ( \hat { w } ) \geq h _ { u } ( w ) + \langle \nabla h _ { u } ( w ) , \hat { w } - w \rangle \quad \Rightarrow \quad \langle - P ^ { - 1 / 2 } \nabla h _ { u } ( w ) , P ^ { 1 / 2 } ( \hat { w } - w ) \rangle \geq h _ { u } ( w ) - h _ { u } ( \hat { w } ) \geq 0
$$

Hence, preconditioned gradient descent on w reduces $\begin{array} { r } { \frac { 1 } { 2 } \| w - \hat { w } \| _ { P } ^ { 2 } } \end{array}$ : define $\begin{array} { r } { w ^ { + } = w - \frac { 1 } { L } P ^ { - 1 } \nabla h _ { u } ( w ) } \end{array}$ . We have

$$
\begin{array} { r l } & { \frac { 1 } { 2 } \| w ^ { + } - \hat { w } \| _ { P } ^ { 2 } \leq \frac { 1 } { 2 } \| w - \hat { w } \| _ { P } ^ { 2 } - \frac { 1 } { L } [ h _ { u } ( w ) - h _ { u } ( \hat { w } ) ] + \frac { 1 } { 2 L ^ { 2 } } \| \nabla h _ { u } ( w ) \| _ { P ^ { - 1 } } ^ { 2 } } \\ & { \qquad \leq \frac { 1 } { 2 } \| w - \hat { w } \| _ { P } ^ { 2 } - \frac { 1 } { L } \underbrace { [ h _ { u } ( w ^ { + } ) - h _ { u } ( \hat { w } ) ] } _ { \triangle h } , } \end{array}\tag{42}
$$

where $\begin{array} { r } { h _ { u } ( w ^ { + } ) \leq h _ { u } ( w ) - \frac { 1 } { 2 L ^ { 2 } } \| \nabla h _ { u } ( w ) \| _ { P ^ { - } } ^ { 2 } } \end{array}$ by L-smoothness in Lemma 4.2. Next, we have, by definition, that

$$
\begin{array} { r l } & { \log ( \zeta ( u ^ { + } ) - \zeta ( u ^ { \star } ) ) - \log ( \zeta ( u ) - \zeta ( u ^ { \star } ) ) = \log ( \frac { \zeta ( u ^ { + } ) - \zeta ( u ^ { \star } ) } { \zeta ( u ) - \zeta ( u ^ { \star } ) } ) } \\ & { \qquad = \log ( \frac { \zeta ( u ^ { + } ) - \zeta ( u ) + \zeta ( u ) - \zeta ( u ^ { \star } ) } { \zeta ( u ) - \zeta ( u ^ { \star } ) } ) } \end{array}
$$

$$
\begin{array} { r } { = \log ( 1 + \frac { \zeta ( u ^ { + } ) - \zeta ( u ) } { \zeta ( u ) - \zeta ( u ^ { \star } ) } ) } \end{array}
$$

$$
\begin{array} { r l } { \mathbf { \epsilon } } & { { } = \log ( 1 + \frac { \zeta ( u ^ { + } ) - \zeta ( u ) } { \| \nabla \zeta ( u ) \| _ { P ^ { - 1 } } ^ { 2 } } \frac { \| \nabla \zeta ( u ) \| _ { P ^ { - 1 } } ^ { 2 } } { \zeta ( u ) - \zeta ( u ^ { \star } ) } ) } \end{array}
$$

$$
\begin{array} { r l } {  { \le \log ( 1 + \frac { \sigma _ { 2 } } { 4 } \operatorname* { m i n } \{ h _ { u } ( w ^ { + } ) , 0 \} ) } \quad } & { { } } \end{array}\tag{43}
$$

$$
\begin{array} { r } { \le \log ( 1 + \frac { \sigma _ { 2 } } { 4 } h _ { u } ( w ^ { + } ) ) } \end{array}\tag{44}
$$

$$
\begin{array} { r } { \leq \frac { \sigma _ { 2 } } { 4 } h _ { u } ( w ^ { + } ) , } \end{array}\tag{45}
$$

where (43) uses the PL inequality from Lemma 4.2 and the fact that $\zeta ( u ^ { + } ) \leq \operatorname* { m i n } \{ \zeta ( u ) , \zeta ( u - \mathcal { D } ( w ^ { + } ) \nabla \zeta ( u ) ) \} ;$ (44) uses monotonicity of log and (45) uses $\log ( 1 + x ) \leq x$ . Together with $\begin{array} { r } { \frac { L \sigma _ { 2 } } { 8 } \| w ^ { + } - \hat { w } \| _ { P } ^ { 2 } - \frac { L \sigma _ { 2 } } { 8 } \| w - \hat { w } \| _ { P } ^ { 2 } \leq } \end{array}$

$\begin{array} { r } { - \frac { \sigma _ { 2 } } { 4 } \big [ h _ { u } ( w ^ { + } ) - h _ { u } ( \hat { w } ) \big ] } \end{array}$ from (42), we have

$$
\begin{array} { r l } & { \Omega _ { \hat { w } } ( u ^ { + } , w ^ { + } ) = \log ( \zeta ( u ^ { + } ) - \zeta ( u ^ { \star } ) ) + \frac { L \sigma _ { 2 } } { 8 } \| w ^ { + } - \hat { w } \| _ { P } ^ { 2 } } \\ & { \qquad \leq \log ( \zeta ( u ) - \zeta ( u ^ { \star } ) ) + \frac { L \sigma _ { 2 } } { 8 } \| w - \hat { w } \| _ { P } ^ { 2 } - \frac { \sigma _ { 2 } } { 4 } [ h _ { u } ( w ^ { + } ) - h _ { u } ( \hat { w } ) ] + \frac { \sigma _ { 2 } } { 4 } h _ { u } ( w ^ { + } ) } \\ & { \qquad = \log ( \zeta ( u ) - \zeta ( u ^ { \star } ) ) + \frac { L \sigma _ { 2 } } { 8 } \| w - \hat { w } \| _ { P } ^ { 2 } + \frac { \sigma _ { 2 } } { 4 } h _ { u } ( \hat { w } ) } \\ & { \qquad = \Omega _ { \hat { w } } ( u , w ) + \frac { \sigma _ { 2 } } { 4 } h _ { u } ( \hat { w } ) , } \end{array}
$$

and this completes the proof.

## B.7 Proof of Lemma 4.4

For brevity, we drop the iteration index and use $u , u ^ { 1 / 2 } , w , y , u ^ { + } , w ^ { + }$ to denote $u ^ { k } , u ^ { k + 1 / 2 } , w ^ { k } , y ^ { k } , u ^ { k + 1 } , w ^ { k + 1 }$ Given that max $\begin{array} { r } { \{ \zeta ( u ) , \zeta ( w ) \} - \zeta ( u ^ { \star } ) \le \operatorname* { m i n } \{ \frac { s \sigma _ { 2 } ^ { 2 } } { 1 2 9 6 } , \frac { \sigma _ { 2 } ^ { 3 } s ^ { 2 } } { 2 4 \| p \| _ { 1 } } \} = : \tau } \end{array}$ , the point $\begin{array} { r } { y = u + \frac { \sqrt { \sigma _ { 2 } } } { 2 + \sqrt { \sigma _ { 2 } } } ( w - u ) = \frac { \sqrt { \sigma _ { 2 } } } { 2 + \sqrt { \sigma _ { 2 } } } w + ( 1 - } \end{array}$ $\frac { \sqrt { \sigma _ { 2 } } } { 2 + \sqrt { \sigma _ { 2 } } } \Big ) u$ is the convex combination between u and $w ,$ , and by convexity, $\begin{array} { r } { \zeta ( y ) \leq \frac { \sqrt { \sigma _ { 2 } } } { 2 + \sqrt { \sigma _ { 2 } } } \zeta ( w ) + ( 1 - \frac { \sqrt { \sigma _ { 2 } } } { 2 + \sqrt { \sigma _ { 2 } } } ) \zeta ( u ) \leq } \end{array}$ $\zeta ( u ^ { \star } ) + \tau$ . By Lemma 4.1, for all $t \in [ 0 , 1 ] , \lambda _ { 2 } ( P ^ { - 1 / 2 } \nabla ^ { 2 } \zeta ( y + t ( u ^ { \star } - y ) ) P ^ { - 1 / 2 } ) \geq \frac { \sigma _ { 2 } } { 2 }$ , and

$$
\begin{array} { r l } & { \quad \zeta ( u ^ { \star } ) - \zeta ( y ) - \langle \nabla \zeta ( y ) , u ^ { \star } - y \rangle } \\ & { = \int _ { 0 } ^ { 1 } ( 1 - t ) \langle u ^ { \star } - y , \nabla ^ { 2 } \zeta ( y + t ( u ^ { \star } - y ) ) ( u ^ { \star } - y ) \rangle \mathrm { d } t } \\ & { = \int _ { 0 } ^ { 1 } ( 1 - t ) \langle P ^ { 1 / 2 } \Pi ( u ^ { \star } - y ) , P ^ { - 1 / 2 } \nabla ^ { 2 } \zeta ( y + t ( u ^ { \star } - y ) ) P ^ { - 1 / 2 } P ^ { 1 / 2 } \Pi ( u ^ { \star } - y ) \rangle \mathrm { d } t } \\ & { \ge \int _ { 0 } ^ { 1 } ( 1 - t ) \frac { \sigma _ { 2 } } { 2 } \| \Pi ( u ^ { \star } - y ) \| _ { P } ^ { 2 } \mathrm { d } t = \frac { \sigma _ { 2 } } { 4 } \| \Pi ( u ^ { \star } - y ) \| _ { P } ^ { 2 } . } \end{array}
$$

Next, by $\zeta ( y ) \leq \zeta ( u ^ { \star } ) + \tau$ , we have $\zeta ( u ^ { 1 / 2 } ) \leq \zeta ( y ) \leq \zeta ( u ^ { \star } ) + \tau$ . Hence $P ^ { - 1 / 2 } \nabla ^ { 2 } \zeta ( y + t ( u ^ { + } - y ) ) P ^ { - 1 / 2 } \preceq 2$ for all $t \in [ 0 , 1 ]$ , giving

$$
\begin{array} { r l } & { \quad \zeta ( u ^ { 1 / 2 } ) - \zeta ( y ) - \langle \nabla \zeta ( y ) , u ^ { 1 / 2 } - y \rangle } \\ & { = \int _ { 0 } ^ { 1 } ( 1 - t ) \langle u ^ { 1 / 2 } - y , \nabla ^ { 2 } \zeta ( y + t ( u ^ { 1 / 2 } - y ) ) ( u ^ { 1 / 2 } - y ) \rangle \mathrm { d } t \leq \| \Pi ( u ^ { 1 / 2 } - y ) \| _ { P } ^ { 2 } . } \end{array}
$$

Finally, by convexity, we have $\zeta ( u ) \geq \zeta ( y ) + \langle \nabla \zeta ( y ) , u - y \rangle$ . By [9, Lemma 4.14], adding the three inequalities

$$
\begin{array} { r l r } & { } & { \zeta ( u ^ { \star } ) - \zeta ( y ) - \langle \nabla \zeta ( y ) , u ^ { \star } - y \rangle \ge \frac { \sigma _ { 2 } } { 4 } \| \Pi ( u ^ { \star } - y ) \| _ { P } ^ { 2 } } \\ & { } & { \zeta ( u ^ { 1 / 2 } ) - \zeta ( y ) - \langle \nabla \zeta ( y ) , u ^ { 1 / 2 } - y \rangle \le \| \Pi ( u ^ { 1 / 2 } - y ) \| _ { P } ^ { 2 } } \\ & { } & { \zeta ( u ) \ge \zeta ( y ) + \langle \nabla \zeta ( y ) , u - y \rangle } \end{array}
$$

with weights $\begin{array} { r } { \frac { \sqrt { \sigma _ { 2 } } } { 2 - \sqrt { \sigma _ { 2 } } } , 1 , \frac { 2 } { 2 - \sqrt { \sigma _ { 2 } } } } \end{array}$ gives $\begin{array} { r } { f ( u ^ { 1 / 2 } , w ^ { + } ) - \zeta ( u ^ { \star } ) \leq ( 1 - \frac { 1 } { 2 } \sqrt { \sigma _ { 2 } } ) [ f ( u , w ) - \zeta ( u ^ { \star } ) ] } \end{array}$ . Using $\zeta ( u ^ { + } ) \leq \zeta ( u ^ { 1 / 2 } )$ shows $\begin{array} { r } { f ( u ^ { + } , w ^ { + } ) - \zeta ( u ^ { \star } ) \leq ( 1 - \frac { 1 } { 2 } \sqrt { \sigma _ { 2 } } ) [ f ( u , w ) - \zeta ( u ^ { \star } ) ] } \end{array}$

Finally we show that $w ^ { + }$ satisfy $\zeta ( w ^ { + } ) - \zeta ( u ^ { \star } ) \leq \tau$ . Given that $f ( u , w ) - \zeta ( u ^ { \star } ) \leq \operatorname* { m i n } \{ 4 \| p \| _ { 1 } ^ { - 1 } , 1 \} \tau$ , we have then $\begin{array} { r } { \frac { 2 } { \sigma _ { \cdot } } \| \Pi ( w ^ { + } - u ^ { \star } ) \| _ { P } ^ { 2 } \leq f ( u , w ) - \zeta ( u ^ { \star } ) \leq \operatorname* { m i n } \{ 4 \| p \| _ { 1 } ^ { - 1 } , 1 \} \tau } \end{array}$ and

$$
\begin{array} { r } { \| \Pi ( w ^ { + } - u ^ { \star } ) \| _ { P } ^ { 2 } \leq \frac { \sigma _ { 2 } \operatorname* { m i n } \{ 4 \| p \| _ { 1 } ^ { - 1 } , 1 \} } { 2 } \tau . } \end{array}
$$

By L-smoothness, we have

$$
\begin{array} { r } { \zeta ( w ^ { + } ) - \zeta ( u ^ { \star } ) \le \frac { L } { 2 } \| \Pi ( w ^ { + } - u ^ { \star } ) \| _ { P } ^ { 2 } \le \frac { \sigma _ { 2 } L } { 4 } \le \frac { \| p \| _ { 1 } } { 4 } \operatorname* { m i n } \{ 4 \| p \| _ { 1 } ^ { - 1 } , 1 \} \tau \le \tau } \end{array}
$$

and this completes the proof.