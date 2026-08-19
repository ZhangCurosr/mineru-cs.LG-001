# Tight Bounds for Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function

Anh Tuan Nguyen<sup>1</sup>, Viet Anh Nguyen<sup>2</sup>

<sup>1</sup>Carnegie Mellon University

<sup>2</sup>The Chinese University of Hong Kong

atnguyen@cs.cmu.edu, nguyen@se.cuhk.edu.hk

## Abstract

Data-driven algorithm design frames hyperparameter tuning as a statistical learning problem, but establishing generalization guarantees remains challenging due to the implicit, nonsmooth dependence of model performance on hyperparameters. Existing multi-dimensional bounds under piecewisepolynomial assumptions remain theoretically loose and lack comprehensive lower bounds. We resolve this by establishing tight pseudo-dimension bounds for multi-dimensional datadriven tuning. First, we refine the learning-theoretic upper bound using real algebraic geometry; by analyzing invariant connected sign cells during block elimination rather than isolated sign vectors, we avoid topological over-counting to derive strictly sharper sample complexities. Second, we present a multi-regime lower-bound framework that disentangles combinatorial and algebraic capacities. By constructing shattered problem instances across distinct regimes, we prove our upper bounds are tightly saturated. Finally, we extend our topological framework to accommodate general bi-level validationloss tuning and broader semi-algebraic applications.

## 1 Introduction

The success of modern machine learning relies heavily on the meticulous selection of hyperparameters (Probst, Bischl, and Boulesteix 2018; a Ilemobayo et al. 2024). Despite its central role in model deployment, hyperparameter tuning is predominantly treated as an empirical art rather than a rigorous analytical discipline. Traditional approaches, such as exhaustive grid search or random search, discretize continuous parameter spaces to identify high-performing configurations through empirical validation. While straightforward in practice, these brute-force methods are theoretically unprincipled and fail to provide formal generalization guarantees across continuous hyperparameter spaces.

To automate this process, practitioners often rely on sophisticated heuristic search strategies, including Bayesian optimization (Bergstra et al. 2011) and spectral methods (Hazan, Klivans, and Yuan 2018). However, these techniques typically hinge on restrictive structural assumptions, such as modeling the loss landscape as a smooth Gaussian process, that often fail to capture the highly volatile, non-convex nature of modern objective functions. Furthermore, resourceallocation strategies like Hyperband (Li et al. 2017) excel at computational eficiency via early stopping, yet they primarily focus on discrete search spaces for fixed problem instances. Consequently, these frameworks lack the theoretical machinery to characterize the fundamental statistical complexity of identifying optimal continuous hyperparameters.

To bridge this theoretical gap, the data-driven algorithm design paradigm (Gupta and Roughgarden 2020; Balcan 2020) frames hyperparameter tuning as a formal statistical learning problem over an unknown, application-specific problem distribution D over the problem space X . Given a finite sample of training instances $S \sim \mathcal { D } ^ { \bar { N } }$ , the goal is to identify a hyperparameter configuration that provably generalizes to unseen problem instances $x \sim \mathcal { D }$ from the exact same distribution. This process is inherently bi-level: the induced loss $\ell _ { \alpha } ( x ) = \operatorname* { i n f } _ { \theta \in S ( x , \alpha ) } g ( x , \alpha , \theta )$ is evaluated on a target validation objective g, subject to the model parameters θ being optimized over a surrogate training objective f, that is, $S ( x , \alpha ) = { \mathrm { a r g } } \operatorname* { m i n } _ { \theta \in \Theta } f ( x , \alpha , \theta )$ . For example, in ridge regression, a problem instance $x = ( A , b , A ^ { \prime } , b ^ { \prime } ) \in \mathcal { X }$ consists of a training and a validation set. The hyperparameter α explicitly regularizes the training objective $\dot { f } ( x , \alpha , \theta ) = \| \dot { A } \theta - \dot { b } \| _ { 2 } ^ { 2 } + \alpha \| \theta \| _ { 2 } ^ { 2 }$ , but impacts the validation target $g ( x , \alpha , \theta ) \overset { = } { = } \| A ^ { \prime } \theta - \overline { { b ^ { \prime } } } \| _ { 2 } ^ { 2 }$ only implicitly via the optimal weights $\theta \in { \mathcal { S } } ( x , \alpha )$ . Bounding the statistical complexity of this non-smooth dependency, where α acts exclusively through the argmin of an auxiliary problem, is our fundamental theoretical hurdle.

To overcome this challenge, we leverage statistical learning theory to bound the generalization capacity, measured via the pseudo-dimension, of the hyperparameter-induced loss function class $\mathcal { L } = \{ \ell _ { \alpha } : \mathcal { X } \right. [ \overset { \left. } { - } H , H ] \mid \alpha \in \mathcal { A } \}$ . Our analysis relies on the structural observation that for many practical machine learning problems, the training objective $f _ { x } ( \alpha , \theta ) \triangleq f ( x , \alpha , \theta )$ admits a piecewise polynomial structure with respect to both the hyperparameter α and the model parameter θ; see e.g., Definition 7 for the formal definition and Figure 2 for a simple example. This piecewise polynomial assumption is highly ubiquitous: it has been rigorously established across a wide spectrum of applications in classical learning theory (Bartlett, Maiorov, and Meir 1998; Montúfar et al. 2014; Bartlett et al. 2019) and in recent data-driven algorithm design frameworks (Balcan et al. 2021; Balcan, Nguyen, and Sharma 2023; Nguyen and Nguyen 2026).

Despite the prevalence of this structure, existing generalization guarantees remain fundamentally limited. Balcan, Nguyen, and Sharma (2025b) provided the first formal framework for this setting, but their ad-hoc, low-dimensional geometric analysis is highly restrictive: it applies exclusively to one-dimensional hyperparameters $( \alpha \in \mathbb { R } )$ and single-level objectives $( f \equiv g )$ . Le, Nguyen, and Nguyen (2026b) successfully extended this analysis to multi-dimensional hyperparameters $( \alpha \in \mathbb { R } ^ { p } )$ and general bi-level objectives $( f \not \equiv g )$ using a model-theoretic approach. However, their statistical bounds are demonstrably suboptimal. Relying heavily on Quantifier Elimination (QE) (Basu, Pollack, and Roy 2006) followed by the Goldberg-Jerrum (GJ) framework (Bartlett, Indyk, and Wagner 2022), their approach generates overly conservative upper bounds due to topological under-counting and inflated algebraic dependencies. Furthermore, their approach fails to establish a lower bound that captures the true combinatorial capacity of the problem, specifically one that scales with the piecewise structural complexities, such as the number of pieces and the number of boundaries. These unresolved theoretical gaps directly motivate the tight bounding techniques developed in this work.

Contributions. In this work, we resolve the theoretical limitations of prior data-driven hyperparameter tuning frameworks by establishing improved pseudo-dimension bounds for the multi-dimensional setting. Our core contributions are:

• We establish in Theorem 4.2 that if a loss function is definable in polynomial first-order logic (FOL), its pseudodimension is tightly bounded by the topological complexity of a nested block elimination process. By analyzing invariant connected sign cells, this approach bypasses the inflated algebraic dependencies of prior quantifier elimination methods (Le, Nguyen, and Nguyen 2026b).

• We apply our nested block elimination framework to the standard training-loss setting $( f \equiv g )$ , where the underlying objective $\bar { f } ( x , \alpha , \theta )$ exhibits a piecewise polynomial structure (Theorem 5.1). By representing the induced loss $\ell _ { \alpha } ( x ) = \mathrm { m i n } _ { \theta \in \Theta } f ( x , \alpha , \theta )$ as a polynomial FOL, we derive strictly sharper pseudo-dimension bounds that directly resolve the suboptimal multi-dimensional guarantees of prior work (Le, Nguyen, and Nguyen 2026b).

• We complement our upper bounds with a novel multiregime lower-bound framework in Lemmas 5.2 and 5.3. By independently analyzing the combinatorial $( T _ { f } , M _ { f } )$ and algebraic $( \Delta _ { f } )$ capacities, we prove that our upper bounds are tight with respect to $T _ { f }$ and $\Delta _ { f } ,$ , and nearly tight with respect to $M _ { f }$ . This establishes the fundamental necessity of each structural parameter in our theoretical guarantees.

• We extend our framework to the general bi-level validationloss setting and broader semi-algebraic applications (Theorems 6.1 and 7.1). By applying nested block elimination to validation-loss tuning, we eliminate the inflated algebraic dependencies of prior work. Furthermore, we demonstrate the framework’s versatility by analyzing the Weighted Group Lasso, accommodating structures beyond piecewise polynomials and strictly improving existing sample complexity bounds by a full factor of $p .$

Technical diference and overviews. Bounding the generalization capacity of multi-dimensional, bi-level hyperparameter tuning requires capturing the implicit, non-smooth dependency of the validation loss on the hyperparameters. Prior model-theoretic work approached this by applying standard Quantifier Elimination (QE) followed by the Goldberg-Jerrum framework (Le, Nguyen, and Nguyen 2026b). However, this strategy yields highly suboptimal statistical bounds due to severe topological overcounting. To resolve this, we introduce a novel geometric framework based on nested block elimination. Rather than evaluating isolated sign conditions, we recursively construct a nested block-sign profile that tracks the logical invariance of polynomial formulas across entire connected sign cells. This topological shift completely bypasses the algebraic inflation inherent to standard QE, enabling us to derive strictly sharper, optimal sample-complexity guarantees for the multi-dimensional tuning problem. See Appendix A.2 for a detailed discussion.

Notations. For a real-valued $t \in \mathbb { R }$ , we denote $\mathrm { s i g n } ( t ) = 0$ if $t = 0 ,$ , sign(t) = 1 if $t > 0 ,$ , and sign(t) = −1 otherwise. For a variable $\boldsymbol { \tilde { z } } \in \mathbb { R } ^ { k }$ , we denote $\mathbb { R } [ z ]$ the polynomial ring of z, containing all (multivariate) polynomials of $z ; \mathrm { e . g . }$ $P ( z ) = z _ { 1 } ^ { 2 } + \bar { z _ { 2 } ^ { 2 } } \in \mathbb { R } [ z ]$ for $z \in \bar { \mathbb { R } } ^ { 2 }$ . Given a polynomial $P ( z )$ , we denote deg(P) the degree of $P ( z ) ; { \mathrm { e . g . , d e g } } ( P ) =$ 2 for $P ( z ) = z _ { 1 } ^ { 2 } + \bar { z _ { 2 } ^ { 2 } }$ . For a function $h ( z , x )$ that takes both variables z and x as inputs, we denote $h _ { x } ( z ) \triangleq h ( z , x )$ the induced function when we treat x as fixed and vary z. For a logic sentence A, we denote $\mathbb { I } ( A ) = 1$ if A is true, else 0; $\mathrm { e . g . , I } ( 1 > 0 ) = 1$ , and $\mathbb { I } ( 0 > \dot { 1 } ) \dot { = } 0$

## 1.1 Related Works

Data-driven algorithm design. Data-driven algorithm design (Gupta and Roughgarden 2020; Balcan 2020) adapts hyperparameters to specific problem distributions rather than relying on worst-case instances. This paradigm has achieved significant empirical success across diverse domains, including sketching (Li et al. 2023; Indyk, Vakilian, and Yuan 2019), linear and mixed-integer programming (Sakaue and Oki 2024; Nguyen and Nguyen 2026; Balcan et al. 2022a; Cheng and Basu 2026; Le, Nguyen, and Nguyen 2026a), and regularization tuning (Balcan et al. 2022b; Balcan, Nguyen, and Sharma 2023).

Theoretical guarantees for data-driven algorithm design. Motivated by empirical successes, recent work has sought to establish statistical guarantees for data-driven algorithm design (Balcan et al. 2021; Bartlett, Indyk, and Wagner 2022; Balcan, Nguyen, and Sharma 2025a). However, the nonsmooth, potentially discontinuous hyperparameter landscape $\ell _ { x } ( \alpha )$ makes this challenging, leading most analyses to avoid inner optimization over model parameters θ. While Balcan, Nguyen, and Sharma (2025b) tackled this harder bi-level case, their framework is restricted to one-dimensional hyperparameters $( \alpha \in \mathbb { R } )$ and single-level objectives $( f \equiv g )$ Le, Nguyen, and Nguyen (2026b) recently extended this to multi-dimensional $( \boldsymbol { \alpha } \in \mathbb { R } ^ { p } )$ and general bi-level settings $( f \not \equiv g )$ , but their complexity bounds remain theoretically loose and lack lower bounds; these are the gaps we resolve in this work.

## 2 Preliminaries

## 2.1 Backgrounds on Learning Theory

We first recall some standard results in learning theory, which play the central role in this work.

Definition 1 (Pseudo-dimension (Pollard 1984)). Consider a real-valued function class ${ \mathcal { L } } = \{ \ell _ { \alpha } : \chi \to \mathbb { R } \mid \alpha \in { \mathcal { A } } \}$ parameterized by $\alpha \in \ A .$ . Given a set of inputs $S \ =$ $( x _ { 1 } , \ldots , x _ { N } ) \subset \mathcal { X } ,$ , we say that S is shattered by $\textit { \textbf { L } } i f$ there exists a set ofreal-valued threshold $\tau _ { 1 } , \dots , \tau _ { N } \in \mathbb { R }$ such that $| \{ ( \mathbb { I } ( \ell _ { \alpha } ( x _ { 1 } ) \geq \tau _ { 1 } ) , \dots , \mathbb { I } ( \ell _ { \alpha } ( x _ { N } ) \geq \tau _ { N } ) ) \mid \ell _ { \alpha } \in \mathcal { L } \} |$ = $\dot { 2 } ^ { N }$ . The pseudo-dimension of L, denoted $\mathrm { P d i m } ( \dot { \mathcal { L } } )$ , is the maximum size N ofan input set that L can shatter.

A finite pseudo-dimension guarantees uniform convergence via empirical risk minimization (ERM).

Theorem 2.1 (Pollard (1984)). Consider a real-valued function class ${ \mathcal { L } } \ = \ \{ \ell _ { \alpha } : \ X \ \to \ [ - H , H ] \ | \ \alpha \ \in \ A \}$ parameterized by $\alpha \in \ A .$ Assume that Pdim(L) is $f ^ { - }$ nite. Then given $\epsilon > 0$ and $\delta \in \mathsf { \Gamma } ( 0 , 1 )$ , for any $N \geq$ $N ( \epsilon , \delta )$ , where $\begin{array} { r } { N ( \epsilon , \delta ) = \mathcal { O } \left( \frac { H ^ { 2 } } { \epsilon ^ { 2 } } ( \mathrm { P d i m } ( \mathcal { L } ) + \log ( 1 / \delta ) ) \right) } \end{array}$ with probability at least $1 - \delta$ over the draw $o f \ S \ =$ $( x _ { 1 } , \bar { \cdot } \cdot \cdot , x _ { N } ) \stackrel { \cdot } { \sim } \mathcal { D } ^ { N }$ , where $\mathcal { D }$ is a distribution over $x ,$ we have $\begin{array} { r } { \mathbb { E } _ { x \sim \mathcal { D } } [ \ell _ { \hat { \alpha } } ( x ) ] \le \operatorname* { i n f } _ { \alpha \in \mathcal { A } } \mathbb { E } _ { x \sim \mathcal { D } } [ \ell _ { \alpha } ( x ) ] + \epsilon } \end{array}$ . Here ${ \hat { \alpha } } \in$ arg $\textstyle \operatorname* { m i n } _ { \alpha \in A } \sum _ { x \in S } \ell _ { \alpha } ( x )$ is the ERM minimizer w.r.t. the set of instances S.

## 2.2 Backgrounds on Real-Algebraic Geometry

In this section, we present the necessary background for our block elimination approach. First, we introduce the notion of first-orderformula (FOL).

Definition 2 (First-order formula, quantified/free variables, and polynomial first-order logic (Renegar 1992)). A firstorderformula $( F O L ) \Phi ( \alpha )$ admits the form:

$$
( q _ { 1 } \theta ^ { [ 1 ] } \in \mathbb { R } ^ { d _ { 1 } } ) \dots ( q _ { K } \theta ^ { [ K ] } \in \mathbb { R } ^ { d _ { K } } ) \psi ( \alpha , \theta ^ { [ 1 ] } , \dots , \theta ^ { [ K ] } ) ,\tag{1}
$$

where

1. Each $q _ { k }$ is one of the quantifiers $\exists \ o r \ \forall .$ . The sequence $\{ q _ { k } \} _ { k = 1 } ^ { K }$ alternates between $\dot { \exists }$ and ∀ and we denote $K$ as the number of quantifier blocks.

2. $\theta ^ { [ 1 ] } , \theta ^ { [ 2 ] } , \dots , \bar { \theta ^ { [ \boldsymbol { K } ] } }$ are called the quantified variables, while $\alpha \in \mathbb { R } ^ { p }$ is called thefree variable.

3. $\psi ( \alpha , \theta ^ { [ 1 ] } , \theta ^ { [ 2 ] } , \dots , \theta ^ { [ K ] } )$ is a boolean combination of atomic predicates oftheform:

$$
P _ { j } ( \alpha , \theta ^ { [ 1 ] } , \theta ^ { [ 2 ] } , \dots , \theta ^ { [ K ] } ) \chi _ { j } 0 ,
$$

where $\chi _ { j } \in \{ > , \geq , < , \leq , = , \neq \}$ is relational operator.

The formula ψ is called the quantifier-free part of Φ. A FOL Φ is a polynomial FOL if each $P _ { j }$ is a polynomial of α and $\theta ^ { [ 1 ] } , \ldots , \theta ^ { [ K ] }$

While our bi-level optimization naturally maps to a polynomial FOL, applying standard QE yields conservative upper bounds. To derive a sharper sample complexity, we track logical invariance across geometric regions rather than isolated points. We formalize these regions using sign conditions.

![](images/db01af15828fccc830d8fd22ac6cdd5fc35da3f2b6b4a89ae67c531d8c863a20.jpg)

Figure 1: A demonstration of sign conditions, realizations, and cells for $\mathcal { P } ~ = ~ \{ P ( z ) ~ \mathbf { \bar { \alpha } } = ~ z ^ { 2 } ~ - ~ 1 \} ~ \subset ~ \mathbb { R } [ z ]$ The realization Reali<sub>P</sub>(1) yields two connected components: $C _ { 1 } ~ = ~ ( - \infty , - 1 )$ and $C _ { 5 } ~ = ~ ( 1 , \infty )$ . Conversely, $\mathrm { R e a l i } _ { \mathcal { P } } ( - 1 )$ yields a single component $C _ { 3 } ~ = ~ ( - 1 , 1 )$ and Real $\dot { \mathcal { P } } ^ { ( 0 ) }$ yields $\begin{array} { r } { C _ { 2 } ^ { \bf { \bar { \Phi } } } = \{ - 1 \} } \end{array}$ and $C _ { 4 } ~ = ~ \{ 1 \}$ . Ultimately, $\mathcal { P }$ induces $| \mathrm { C e l l } ( \mathcal { P } ) | ~ = ~ \bar { 5 }$ connected sign cells: $\operatorname { C e l l } ( \mathcal { P } ) = \{ C _ { 1 } , C _ { 2 } , \dot { C } _ { 3 } , \dot { C _ { 4 } } , \dot { C _ { 5 } } \}$

Definition 3 (Sign conditions and their realizations). Let $\mathcal { P } = \{ P _ { 1 } , . . . , \bar { P _ { s } } \} \subset \mathbb { R } [ z ]$ be a finite set of polynomials in $z ~ \in ~ \mathbb { R } ^ { p }$ , and let $Z \doteq \mathbb { R } ^ { p } .$ . A sign condition on $\mathcal { P }$ is a vector $\sigma = ( \sigma _ { 1 } , \ldots , \sigma _ { s } ) \in \{ - 1 , 0 , 1 \} ^ { s } ,$ . The realization $ { \mathrm { R e a l i } } _ { \mathcal { P } } ( \sigma , Z )$ of sign condition σ over $Z$ is a set $o f z \in Z$ such that sign $( \bar { P } _ { i } ( \bar { z } ) ) = \sigma _ { i }$ for every $i = 1 , \ldots , s ,$ that is,

$$
{ \mathrm { R e a l i } } _ { { \mathcal { P } } } ( \sigma , Z ) = \{ z \in Z \mid s i g n ( P _ { i } ( z ) ) = \sigma _ { i } , i = 1 , \ldots , s \} .
$$

We then denote $\mathrm { S i g n } ( \mathcal { P } , Z )$ the set of all possible sign conditions σ that we can have by varying $z \in Z ,$ , that is

$$
{ \mathrm { S i g n } } ( { \mathcal { P } } , Z ) = \{ \sigma \in \{ - 1 , 0 , 1 \} ^ { s } \mid { \mathrm { R e a l i } } _ { { \mathcal { P } } } ( \sigma , Z ) \neq \emptyset \} .
$$

Equivalently, Sign $\mathfrak { \backslash } ( \mathcal { P } , Z ) = \{ s i g n ( \mathcal { P } ( z ) ) : z \in Z \}$ , where we abuse notation and write

$$
s i g n ( \mathcal { P } ( z ) ) = ( s i g n ( P _ { 1 } ( z ) ) , \ldots , s i g n ( P _ { s } ( z ) ) .
$$

When $Z ~ = ~ \mathbb { R } ^ { p } ,$ , we shorten the notation and write $\mathrm { R e a l i } _ { \mathcal P } ( \sigma ) = \mathrm { R e a l i } _ { \mathcal P } ( \sigma , \mathbb { R } ^ { p } )$ and $\operatorname { S i g n } ( \mathcal { P } , \mathbb { R } ^ { p } ) = \operatorname { S i g n } ( \mathcal { P } )$ .

Definition 4 (Connected sign cells). Let $\mathcal { P } \subset \mathbb { R } [ z ]$ be a finite set of polynomials. For each realizable sign pattern $\sigma \in \operatorname { S i g n } ( \mathcal { P } )$ , let $\smash {  { \mathrm { C c } } (  { \mathrm { R e a l i } _ { \mathcal { P } } } ( \sigma ) ) }$ denote the set of connected components of the realization Reali $_ { \mathcal { P } } ( \sigma )$ . We then denote $\mathrm { C e l l } ( \mathcal { P } )$ the set ofconnected sign cells induced by P, that is

$$
\operatorname { C e l l } ( { \mathcal { P } } ) = \bigcup _ { \sigma \in \operatorname { S i g n } ( { \mathcal { P } } ) } \operatorname { C c } ( \operatorname { R e a l i } _ { { \mathcal { P } } } ( \sigma ) ) .
$$

We further denote $b _ { 0 } ( S )$ the number of connected components of S, and therefore

$$
\vert \mathrm { C e l l } ( \mathcal { P } ) \vert = \sum _ { \sigma \in \mathrm { S i g n } ( \mathcal { P } ) } b _ { 0 } \bigl ( \mathrm { R e a l i } _ { \mathcal { P } } ( \sigma ) \bigr ) .
$$

Figure 1 demonstrates the concepts above. We now formulate block elimination in terms of these connected sign cells. Given polynomials in free variables z and variables t to be eliminated, the block elimination process aims to produce a set of polynomials in z whose connected sign cells preserve the sign information attainable by varying t. Critically, our multi-dimensional tuning problem involves distinct blocks of quantified variables that must be eliminated sequentially. Evaluating them simultaneously destroys this block-wise grouping, leading to topological under-counting. To bypass this algebraic inflation, we track sign invariance sequentially by constructing nested profiles.

Definition 5 (Nested block-sign profiles). Let $\begin{array} { r l } { \mathcal { P } } & { { } = } \end{array}$ $\{ P _ { 1 } , \dots , P _ { s } \} \subset \mathbb { R } [ z , t ^ { ( 1 ) } , \dots , t ^ { ( K ) } ]$ , where $z ~ \in ~ \mathbb { R } ^ { p }$ and $t ^ { ( k ) } \in \mathbb { R } ^ { d _ { k } } f o r \ k = 1 , \dots , K$ . We define the terminal sign vector $\Sigma _ { K } \big ( \bar { \boldsymbol { z } } , t ^ { ( 1 ) } , \dots , t ^ { ( K ) } \big ) = s i g n \big ( \mathcal { P } ( \boldsymbol { z } , t ^ { ( 1 ) } , \dots , t ^ { ( K ) } \big ) \big )$ . Recursively, $f o r k = K , \ldots , 1$ , we define

$$
\begin{array} { r l } & { \Sigma _ { k - 1 } ( z , t ^ { ( 1 ) } , \dots , t ^ { ( k - 1 ) } ) = } \\ & { \qquad \ \{ \Sigma _ { k } ( z , t ^ { ( 1 ) } , \dots , t ^ { ( k - 1 ) } , u ) : u \in \mathbb { R } ^ { d _ { k } } \} . } \end{array}
$$

Then, we call $\Sigma _ { 0 } ( z )$ the nested block-sign profile $o f \mathcal { P }$ at a fixed point $z \in \mathbb { R } ^ { p }$ , relative to the ordered variable blocks $\mathbf { \check { \chi } } _ { t } ^ { ( 1 ) } , \dots , t ^ { ( K ) }$

Roughly speaking, in Definition 5, we can think of the terminal sign vector $\Sigma _ { K }$ as an ordinary sign vector, and subsequently $\Sigma _ { K - 1 }$ is a set of sign vectors, and so on. Finally, the nested block-sign profile $\begin{array} { r } { \dot { \Sigma _ { 0 } } ( z ) } \end{array}$ can be thought of as a depth-K sign tree. Specifically, unlike the ordinary set of sign conditions obtained by varying all quantified variables simultaneously, $\Sigma _ { 0 } ( z )$ retains the block-wise grouping of realizable sign patterns, which is crucial in our analysis. Based on this definition, we can now formally define the nested sign-invariant projection as follows.

Definition 6 (Nested sign-invariant projection). A set of polynomials $Q \subset \mathbb { R } [ z ]$ is a nested sign-invariant projection of $\mathcal { P } \subset \mathbb { R } [ z , t ^ { ( 1 ) } , \ldots , t ^ { ( K ) } ]$ , relative to the ordered blocks $t ^ { ( 1 ) } , \dots , t ^ { ( \bar { K } ) }$ , if the nested block-sign profile $\Sigma _ { 0 } ( z )$ is constant on every connected sign cell induced by Q. That $i s ,$ for every $C \in { \mathrm { C e l l } } ( { \mathcal { Q } } )$ and every $z _ { 1 } , z _ { 2 } \in C ,$ , we have $\mathrm { \dot { Z } } _ { 0 } ( z _ { 1 } ) = \Sigma _ { 0 } ( z _ { 2 } )$

Specially, when $K = 1$ , we have $\Sigma _ { 0 } ( z ) = \{ \mathrm { s i g n } ( \mathcal { P } ( z , u )$ $u \in \bar { \mathbb { R } } ^ { d _ { 1 } } \bar  \} = \mathrm { S i g n } ( \mathcal { P } _ { z } , \mathbb { R } ^ { d _ { 1 } } )$ . Thus, in the one-block case, Definition 6 simply requires the set of realizable sign conditions to be unchanged on every connected sign cell induced by Q. To help the audience quickly grasp the concepts of nested block-sign profiles in Definition 5 and nested-sign invariant projection in Definition 6, we present the following simple example.

Example 1. $L e t z , u , v \in \mathbb { R } ,$ , with ordered quantified blocks $t ^ { ( 1 ) } = u$ and $t ^ { ( 2 ) } = v .$ . Consider the set of polynomials $\mathcal { P } = \{ P _ { 1 } , P _ { 2 } \} \subset \mathbb { R } [ z , u , v ]$ , where $P _ { 1 } ( z , u , v ) \stackrel { \textstyle = } { = } u ^ { 2 } - z$ and $P _ { 2 } ( z , u , v ) = v .$ . Then the terminal sign vector $\Sigma _ { 2 } ( z , u , v )$ is $\Sigma _ { 2 } ( z , u , v ) = ( s i g n ( u ^ { 2 } - z ) , s i g n ( v ) )$ . For $a \in \{ - 1 , 0 , 1 \}$ we define $E _ { a } \doteq \bar { \{ ( a , - 1 ) , ( a , 0 ) , ( a , 1 ) \} }$ , then after varying the innermost block v, we obtain $\Sigma _ { 1 } ( \bar { z } , u ) = E _ { s i g n ( u ^ { 2 } - z ) } .$ After subsequently varying $u ,$ the nested sign profile $\Sigma _ { 0 } ( z )$ becomes

$$
\begin{array} { r } { \Sigma _ { 0 } ( z ) = \left\{ \begin{array} { l l } { \{ E _ { 1 } \} , } & { z < 0 , } \\ { \{ E _ { 0 } , E _ { 1 } \} , } & { z = 0 , } \\ { \{ E _ { - 1 } , E _ { 0 } , E _ { 1 } \} , } & { z > 0 . } \end{array} \right. } \end{array}
$$

Consequently, $\boldsymbol { \mathcal { Q } } = \left\{ \boldsymbol { z } \right\}$ is a nested sign-invariantprojection $o f \mathcal { P } a \bar { s } \sum _ { 0 } \bigl ( \bar { z } \bigr )$ is constant on each ofthe three connected sign cells $( - \infty , 0 ) , \{ 0 \}$ , and $( 0 , \infty )$ induced by Q.

To rigorously bound the statistical complexity of bi-level hyperparameter tuning, we must first formalize the implicit dependency in the validation loss. We achieve this by formulating continuous-optimization objectives in polynomia first-order logic (FOL).

![](images/224cc0c2f68887b08e39c8e4273c5cfa20744ae283897fe4b6b56a66fb15ed6d.jpg)  
Figure 2: An example of a piecewise polynomial function $f ( z )$ (Definition 7) with parabolic boundary $h _ { 1 } ( z ) = z _ { 1 } - z _ { 2 } ^ { 2 }$ The function evaluates to $f _ { ( - 1 ) } ( z ) = z _ { 1 }$ in the pink region $( \sigma ( z ) = - 1 ) , f _ { ( 1 ) } ( z ) = z _ { 2 }$ in the blue region $( \sigma ( z ) = 1 )$ and $f _ { ( 0 ) } ( z ) = z _ { 1 } ^ { 2 } + z _ { 2 } ^ { 2 }$ on the black boundary $( \sigma ( z ) = 0 )$ With 1 boundary, 3 pieces, and maximum polynomial degree 2, the function’s complexity is (1, 3, 2).

Theorem 2.2 (Nested block elimination, Adapted from Basu, Pollack, and Roy (2006, Chapter 14.1)). Fix $K \geq 1$ , let $z ~ \in ~ \mathbb { R } ^ { p } , ~ t ^ { ( k ) } ~ \in ~ \mathbb { R } ^ { d _ { k } }$ for $k = 1 , \ldots , K$ , and let $\mathcal { P } \subset$ $\mathbb { R } [ z , t ^ { ( 1 ) } , \dots , t ^ { ( K ) } ]$ be a finite set of polynomials satisfying (i) $| { \bar { \mathcal { P } } } | \leq s ,$ and (ii) deg $\dot { \left[ R \right) } \leq \Delta$ for every $R \in \mathcal P$ . Then there exists a nested sign-invariant projection $\mathcal { Q } \subset \mathbb { R } [ z ] o f \mathcal { P } ,$ , relative to the ordered blocks $t ^ { ( 1 ) } , \dots , t ^ { ( K ) }$ , such that the nested block-sign profile $\Sigma _ { 0 } ( z )$ is constant on every $C \in { \mathrm { C e l l } } ( { \mathcal { Q } } )$ Moreover, defining $\begin{array} { r } { A _ { K } \triangleq \prod _ { k = 1 } ^ { K } ( d _ { k } + 1 ) , B _ { K } \triangleq \prod _ { k = 1 } ^ { K } d _ { k } } \end{array}$ then there exists a constant $c _ { K } ,$ , depending only on $K$ , such that

$$
| \mathcal { Q } | \leq s ^ { A _ { K } } \Delta ^ { c _ { K } B _ { K } } , \mathrm { ~ a n d ~ } \operatorname* { m a x } _ { R \in \mathcal { Q } } \deg ( R ) \leq \Delta ^ { c _ { K } B _ { K } } .
$$

Finally, we recall a useful result that allows us to give an upper bound for the number of elements in the set of connected sign cells $\operatorname { C e l l } ( \mathcal { Q } )$ corresponding to a set of polynomials Q. Lemma 2.3 (Basu, Pollack, and Roy (2005, Theorem 1.1)). Let $\mathcal { Q } \subset \mathbb { R } [ z ]$ be a set ofs polynomials in $z \in \mathbb { R } ^ { p }$ of degree at most ∆, where s, ${ \mathrm { \Delta } p } , \Delta \geq 1 $ . Then there exists an universal constant C such that $\begin{array} { r } { | \mathrm { C e l l } ( \mathcal { Q } ) | \leq \left( \frac { C s \Delta } { p } \right) ^ { p } i f s \geq p , } \end{array}$ and $| \mathrm { C e l l } ( \mathcal { Q } ) | \leq ( C \Delta ) ^ { p } i f s < p .$

## 3 Problem Settings

Our data-driven framework models hyperparameter tuning across three foundational spaces: a problem instance space $\mathcal { X } \subset \mathbb { R } ^ { q }$ , a continuous hyperparameter space $A \subset \mathbb { R } ^ { p }$ , and a model parameter space $\Theta \subseteq \mathbb { R } ^ { d }$ . To ensure analytical compactness, we restrict the parameter domains to bounding boxes, defining $\mathcal { A } = [ \alpha _ { m i n } , \alpha _ { m a x } ] ^ { p }$ and $\Theta = [ \theta _ { m i n } , \theta _ { m a x } ] ^ { \tilde { d } }$ For any given instance $x \in { \mathcal { X } } .$ , the learning procedure evaluates a model parameterized by $\theta \in \Theta$ alongside hyperparameters $\alpha \in { \mathcal { A } }$ via two distinct objectives:

• Training Objective $f : \mathcal { X } \times \mathcal { A } \times \Theta  [ - H , H ]$ : The surrogate loss directly minimized by the algorithm during training.

• Validation Objective $g : \mathcal { X } \times \mathcal { A } \times \Theta  [ - H , H ] ;$ : The true target metric utilized to assess the quality of the model’s out-of-sample performance.

Here, H acts as a positive constant bounding the function values.

The Bi-level Optimization Objective. The ultimate performance of a hyperparameter configuration α on a specific problem instance x is quantified by the induced loss $\ell _ { \alpha } ( x )$ Because hyperparameters generally dictate model behavior implicitly through the training procedure, $\ell _ { \alpha } ( x )$ is naturally cast as a bi-level optimization problem $\begin{array} { r } { \ell _ { \alpha } ( x ) = \operatorname* { i n f } _ { \theta \in S ( x , \alpha ) } g ( x , \alpha , \theta ) } \end{array}$ where the set of optimal lower-level model parameters is defined as $s ( x , \alpha ) =$ arg mi $\displaystyle \boldsymbol { \Omega } _ { \theta \in \Theta } f ( \boldsymbol { x } , \alpha , \theta )$ . By evaluating the infimum over the set $s ( x , \alpha )$ , this framework employs the standard optimistic bi-level formulation. This guarantees that the validation loss remains strictly well-defined even if the inner surrogate optimization yields a non-singleton set of minimizers.

Statistical Learning Goal. We assume the existence of an unknown, application-specific distribution D from which problem instances are drawn. The theoretical goal of data-driven tuning is to isolate a configuration $\alpha ^ { * }$ that minimizes the expected validation loss, $\mathrm { i } . \mathrm { e } . , \ \alpha ^ { \ast } \in $ arg $\mathrm { m i n } _ { \alpha \in \mathcal { A } } \mathbb { E } _ { x \sim \mathcal { D } } [ \ell _ { \alpha } ( x ) ]$ . Given that the true distribution $\mathcal { D }$ is unobservable, the learner is instead provided with a finite dataset $S \ = \ \{ x _ { 1 } , . . . , x _ { N } \} \ \sim \ { \mathcal { D } } ^ { N }$ consisting of N training instances. The task therefore reduces to Empirical Risk Minimization (ERM), where the algorithm computes an empirical minimizer $\hat { \alpha } ( S )$ , that is, $\bar { \hat { \alpha } } ( S ) \in$ arg $\begin{array} { r } { \operatorname* { m i n } _ { \alpha \in \mathcal { A } } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \ell _ { \alpha } ( x _ { i } ) } \end{array}$

Structural Assumptions. As in prior works (Balcan, Nguyen, and Sharma 2025b; Le, Nguyen, and Nguyen 2026b), a central theoretical pillar of our subsequent complexity analysis requires the objective functions to exhibit specific algebraic geometries.

Definition 7 (Piecewise polynomial function (Balcan, Nguyen, and Sharma 2025b)). A real-valued function f admits a piecewise polynomial structure with complexity $( M _ { f } , T _ { f } , \Delta _ { f } )$ if there exists a set of boundary polynomials $\mathbb { H } \doteq \{ h _ { 1 } , \ldots , h _ { M _ { f } } \}$ and a set of value polynomials $\mathbb { F } = \{ f _ { \sigma } \} _ { \sigma \in \Sigma _ { f } } \subset \mathbb { R } [ z ]$ , where $\Sigma _ { f } \ \subseteq \ \{ - 1 , 0 , 1 \} ^ { M _ { f } }$ and $\vert \Sigma _ { f } \vert \le T _ { f }$ , both with polynomial degrees at most $\Delta _ { f } ,$ , such thatfor any z, we have $f ( z ) = f _ { \pmb { \sigma } ( z ) } ( z )$ , where σ $\cdot ( z ) \in \Sigma _ { f }$ is the sign pattern of z with respect to the set offunctions $\mathbb { H } , i . e . , \pmb { \sigma } ( z ) _ { j } = s i g n ( h _ { j } ( z ) )$ . The functions in F are called the piece functions and the functions in H are called the boundaryfunctions.

Assumption 1 (Structural assumption (Le, Nguyen, and Nguyen 2026b)). Given any problem instance $x , \ t h e$ dual parameter-dependent training objective $f _ { x } ( \alpha , \theta ) \ \triangleq$ $f ( x , \alpha , \theta )$ and dual parameter-dependent validation objective $g _ { x } ( \alpha , \theta ) \ \triangleq \ g ( x , \alpha , \theta )$ admits piecewise polynomial structure as functions of α and θ with complexity $( M _ { f } , T _ { f } , \Delta _ { f } )$ and $\left( M _ { g } , T _ { g } , \hat { \Delta _ { g } } \right)$ that are independent of the problem instance x, respectively. Moreover, the set $s ( x , \alpha )$ ofminimizes of $f ( x , \alpha , \cdot )$ is non-empty,for all $( x , \alpha )$

Remark 1 (Ubiquity of structure). As noted in prior works (Balcan, Nguyen, and Sharma 2025b; Le, Nguyen, and Nguyen 2026b), this structural assumption is ubiquitous. It has been formally established across a wide range of applications in both classical learning theory (Bartlett, Maiorov, and Meir 1998; Montúfar et al. 2014; Bartlett et al. 2019)) and data-driven algorithm design (Balcan et al. 2021; Balcan, Nguyen, and Sharma 2023; Nguyen and Nguyen 2026).

Although recent work has successfully extended datadriven algorithm design to multi-dimensional hyperparameters and bi-level objectives (Le, Nguyen, and Nguyen 2026b), their sample complexity guarantees are fundamentally suboptimal. By analyzing invariant connected sign cells rather than relying on the inflated algebraic dependencies of Quantifier Elimination, we now present a nested block elimination framework that entirely eliminates this theoretical looseness.

## 4 A General Learning-theoretic Complexity Framework via Block Elimination

In this section, we introduce a new general upper bound for learning-theoretic complexity by leveraging nested block elimination. First, we establish Proposition 4.1, which guarantees that the truth value of a polynomial first-order logic formula remains invariant across the connected sign cells of a nested sign-invariant projection.

Proposition 4.1 (Cellwise logical invariance). Let $\Phi ( z ) = $ $( \boldsymbol { q _ { 1 } t ^ { ( 1 ) } } \in \mathbb { R } ^ { d ^ { ( 1 ) } } ) \dots ( \boldsymbol { q _ { K } t ^ { ( K ) } } \in \mathbb { R } ^ { d ^ { ( K ) } } ) \psi ( \boldsymbol { z } , t ^ { ( 1 ) } , \dots , t ^ { ( K ) } )$ be a polynomial FOL, and let P be the family of atomic polynomials appearing in the atomic predicates of ψ. $H \mathcal { Q }$ is a nested sign-invariant projection of P relative to the ordered blocks $t ^ { ( 1 ) } , \dots , t ^ { ( \dot { K } ) } ,$ , then Φ has a constant truth value on every connected sign cell $C \in { \mathrm { C e l l } } ( { \mathcal { Q } } )$ . In other words,for every $C \in \mathbf { C e l l } ( \mathcal { Q } )$ and every $z _ { 1 } , z _ { 2 } \in C$ , we have $\Phi ( z _ { 1 } ) \Leftrightarrow \Phi ( z _ { 2 } )$

The proof of Proposition 4.1 is presented in Appendix B. Using this cellwise logical invariance alongside the nested block elimination process, we now state the following theorem, which establishes a rigorous pseudo-dimension upper bound for any function class whose threshold conditions can be formulated via branches of polynomial first-order logic.

Theorem 4.2 (Pseudo-dimension upper-bound via branchwise nested block elimination). Consider a real-valuedfunction class ${ \mathcal { F } } = \{ f _ { \alpha } : { \mathcal { X } }  \mathbb { R } \mid \alpha \in { \mathcal { A } } \}$ be a real-valued function class, where $\mathcal { A } = [ \alpha _ { \mathrm { m i n } } , \alpha _ { \mathrm { m a x } } ] ^ { p } \subseteq \mathbb { R } ^ { p }$ . Suppose that there are integers $L , M , K \ge 1$ , where K is fixed, block dimensions $d _ { 1 } , \dotsc , d _ { K } \geq 1$ , and a degree upper bound $\Delta \geq 1$ satisfying the following conditions: for any $x \in \mathcal { X }$ and a real-valued threshold $\tau \in \mathbb { R }$ , there exists

1. L polynomial FOLs $\Phi _ { x , \tau , 1 } ( \alpha ) , \dots , \Phi _ { x , \tau , L } ( \alpha )$ , and   
2. a Booleanfunction $\beta _ { x , \tau } : \{ 0 , 1 \} ^ { L } \to \{ 0 , 1 \}$

such that $\mathbb { I } ( f _ { \alpha } ( x ) \ge \tau ) = \beta _ { x , \tau } ( \Phi _ { x , \tau , 1 } ( \alpha ) , \ldots , \Phi _ { x , \tau , L } ( \alpha ) )$ for every $\alpha \in { \mathcal { A } } .$ Furthermore, assuming that every FOL $\Phi _ { x , \tau , \ell } .$

1. has K ordered quantified variables blocks of dimension $d _ { 1 } , \ldots , d _ { K }$ , respectively, and

2. contains at most M distinct atomic polynomials, each of degree at most $\Delta$

Define $\begin{array} { r } { A _ { K } = \prod _ { k = 1 } ^ { K } ( d _ { k } + 1 ) } \end{array}$ and $\begin{array} { r } { B _ { K } = \prod _ { k = 1 } ^ { K } d _ { k } , } \end{array}$ , then the pseudo-dimension of the function class $\mathcal { F }$ is at most

$\dim ( { \mathcal { F } } ) = { \mathcal { O } } \left( p \log ( 2 L ) + p A _ { K } \log ( 2 M ) + p B _ { K } \log ( 2 \Delta ) \right)$ Proof Sketch. Assume there exists N problem instances $x _ { 1 } , \ldots , x _ { N }$ that can be shattered by ${ \dot { \mathcal { F } } } ,$ with real-valued thresholds $\tau _ { 1 } , \ldots , \tau _ { N }$ witness the shattering. Let $S _ { K } \ =$ $M ^ { A _ { K } } \Delta ^ { c _ { K } B _ { K } }$ and $\Delta _ { K } = \Delta ^ { c _ { K } B _ { K } }$ , where $c _ { K }$ depends only on K from Theorem 2.2.

For every instance $i ~ = ~ 1 , \dots , N$ , and a branch $\ell \ =$ $1 , \ldots , L$ , let $\mathcal { P } _ { i , \ell }$ be the set of atomic polynomials appearing in $\Phi _ { x _ { i } , \tau _ { i } , \ell } .$ . From Theorem 2.2, there exists a nested signinvariant projection $\mathcal { Q } _ { i , \ell } \subset \mathbb { R } [ \alpha ]$ such that: $( 1 ) \left| \mathcal { Q } _ { i , \ell } \right| \le S _ { K }$ and (2) $\begin{array} { r } { \mathrm { n a x } _ { R \in { \mathcal { Q } } _ { i , \ell } } \deg ( R ) \leq \dot { D } _ { K } } \end{array}$ . By Proposition 4.1, the truth value of $\Phi _ { x _ { i } , \tau _ { i } , \ell }$ remains unchanged on every connected sign cell induced by $\mathcal { Q } _ { \ell _ { i } }$

Now, let $B _ { A }$ be a family of $2 p$ afine polynomials describing the region A, and let $\begin{array} { r } { \tilde { \mathcal { Q } } = \mathcal { B } _ { \cal A } \cup \bigcup _ { i = 1 } ^ { N } \bigcup _ { \ell = 1 } ^ { L } \mathcal { Q } _ { i , \ell } . } \end{array}$ . By definition, we have (1) $p \leq \left| \tilde { \mathcal { Q } } \right| \leq N L S _ { K } + 2 p ,$ and (2) $\operatorname* { m a x } _ { R \in \tilde { \mathcal { Q } } } \deg ( R ) \ \leq \ D _ { K }$ . Furthermore, from the previous observation about $\mathcal { Q } _ { i , \ell } ,$ , we claim that over every connected sign cell $C \in \mathrm { C e l l } ( \tilde { \mathcal { Q } } )$ , the threshold label

$$
\mathbb { I } ( f _ { \alpha } ( x ) \ge \tau ) = \beta _ { x , \tau } ( \Phi _ { x , \tau , 1 } ( \alpha ) , \ldots , \Phi _ { x , \tau , L } ( \alpha ) )
$$

remains unchanged. Therefore, we claim that

$$
2 ^ { N } \leq | \mathrm { C e l l } ( \tilde { \mathcal { Q } } ) | \leq \left( \frac { \kappa ( N L S _ { K } + 2 p ) D _ { K } } { p } \right) ^ { p } ,
$$

where the latter inequality comes from Lemma 2.3. Finally, solving this inequality gives the final conclusion. See $\mathsf { A p - }$ pendix B for the detailed proof. □

## 5 Tuning via Training Objective

In this section, we focus exclusively on the standard trainingloss setting, where the training and validation objectives are identical $( \mathrm { i } . \mathsf { e } . , f \equiv g )$ . Under this regime, we apply our nested block elimination framework to the underlying piecewise polynomial objective to derive a sharp pseudo-dimension upper bound in Section 5.1, followed by a comprehensive lower bound analysis in Section 5.2 to demonstrate tightness.

## 5.1 Upper Bound

We first bound the generalization capacity of the hyperparameter-induced loss function. By formulating the inner minimization problem as a polynomial first-order logic sentence, we can directly invoke the block elimination argument (Theorem 2.2) to derive the following upper bound.

Theorem 5.1 (Pseudo-dimension). Let $\mathcal { A } = [ \alpha _ { \mathrm { m i n } } , \alpha _ { \mathrm { m a x } } ] ^ { p }$ and $\Theta = [ \theta _ { \mathrm { m i n } } , \theta _ { \mathrm { m a x } } ] ^ { d }$ be the hyperparameter and parameter domains, respectively. Assume that for any problem instance $x \in \mathcal { X }$ , the training objective $f _ { x } ( \alpha , \theta ) \triangleq f ( x , \alpha , \theta )$ for any $( \alpha , \theta ) \in \mathcal { A } \times \Theta ,$ admits a piecewise polynomial structure with complexity $( M _ { f } , T _ { f } , \bar { \Delta _ { f } } )$ . Then, the pseudodimension ofthefunction class ${ \mathcal { L } } \breve { = } \{ \ell _ { \alpha } : \chi \to \mathbb { R } \mid \alpha \in { \mathcal { A } } \}$ where $\ell _ { \alpha } ( x ) = \operatorname* { m i n } _ { \theta \in \Theta } f ( x , \alpha , \theta )$ , is at most

Pdim(L) = O(p log T<sub>f</sub> + pd log(M<sub>f</sub> + d) + pd log ∆<sub>f</sub>).

Proof sketch. For any problem instance $x \in \mathcal { X }$ and threshold $\tau \in \mathbb { R }$ , our goal is to express the boolean indicator $\mathbb { I } ( \ell _ { x } ( \alpha ) ~ \geq ~ \tau )$ as a boolean combination of polynomial first-order logic (FOL) formulas to directly invoke Theorem 2.2. Because the induced loss is defined via inner minimization, $\ell _ { x } ( \alpha ) \ = \ \operatorname* { m i n } _ { \theta \in \Theta } f _ { x } ( \alpha , \theta )$ , the condition $\ell _ { x } ( \alpha ) \ \geq \ \tau$ is logically equivalent to the universal statement $( \forall \theta \in \Theta ) [ f _ { x } ( \alpha , \theta ) \geq \tau ]$ . Exploiting the piecewise polynomial structure of $f _ { x }$ , we partition the parameter space into invariant sign regions. Let $\mathrm { w h e r e } _ { x , \sigma } ( \alpha , \theta ) \triangleq$ $\textstyle \bigwedge _ { m = 1 } ^ { M _ { x } } [ \mathrm { s i g n } ( h _ { x , m } ( \alpha , \theta ) ) = \sigma _ { m } ]$ be the predicate indicating whether $( \alpha , \theta )$ falls into the region defined by the sign pattern $\sigma \in \Sigma _ { x }$ . We can rewrite $\mathbb { I } ( \ell _ { x } ( \alpha ) \geq \tau )$ as a conjunction over all valid regions $\textstyle \bigwedge _ { \sigma \in \Sigma _ { x } } \Phi _ { x , \tau , \sigma } ( \alpha )$ where

$$
\Phi _ { x , \tau , \sigma } ( \alpha ) \triangleq ( \forall \theta \in \Theta ) [ \mathrm { W h e r e } _ { x , \sigma } ( \alpha , \theta ) \Rightarrow P _ { x , \sigma } ( \alpha , \theta ) \geq \tau ] .
$$

By substituting the domain constraint $\begin{array} { r l } { { \mathrm { I n } } _ { \Theta } ( \theta ) } & { { } = } \end{array}$ $\textstyle \bigwedge _ { r = 1 } ^ { d } ( \theta _ { \operatorname* { m a x } } - \theta _ { r } \geq 0 ) \wedge ( \theta _ { r } - \theta _ { \operatorname* { m i n } } \geq 0 )$ and applying the logical identity $( A \overset { \cdot } { \Rightarrow } B ) \equiv ( \neg A \lor B )$ , we transform each branch into a standard polynomial FOL $\Phi _ { x , \tau , \sigma } ( \alpha ) =$

$$
( \forall \theta \in \mathbb { R } ^ { d } ) [ \neg \mathrm { I n } _ { \Theta } ( \theta ) \lor \neg \mathrm { W h e r e } _ { x , \sigma } ( \alpha , \theta ) \lor P _ { x , \sigma } ( \alpha , \theta ) \geq \tau ] .
$$

Crucially, each formula $\Phi _ { x , \tau , \sigma }$ requires exactly $K = 1$ quantifier block of dimension $\dot { d _ { 1 } } = d ,$ , and contains at most $\bar { M } _ { f } + 2 d + 1$ distinct atomic polynomial predicates (accounting for the active boundary pieces, domain edges, and the threshold comparison). Furthermore, the maximum degree across all atomic predicates is strictly bounded by $\Delta _ { f } .$ Substituting these uniform complexity parameters into our nested block elimination bound in Theorem 4.2 yields the final pseudo-dimension. The detailed proof can be found in Appendix C.1. □

Remark 2 (Comparison to prior works). Theorem 5.1 strictly improves upon the multi-dimensional generalization guarantees established in prior works in Le, Nguyen, and Nguyen (2026b, Theorem 5.1). Their approach, which relied on standard quantifier elimination, yielded a suboptimal upper bound of $\dot { \cdot } \dot { P } d i m \dot { ( } \mathcal { L } ) = \mathcal { O } ( p d \log ( M _ { f } + T _ { f } + d ) + \dot { p } ^ { 2 } d \log \dot { \Delta } _ { f } )$ $B y$ utilizing nested block elimination, we remove the inflated $p ^ { 2 }$ dependency on the algebraic capacity and significantly sharpen the combinatorial dependencies. Furthermore, while prior work only established a partial lower bound of $\Omega ( p d \log \Delta _ { f } )$ , we resolve this gap in Section 5.2 by providing a complete theoretical lower bound that explicitly captures the combinatorial complexities $M _ { f }$ and $\ddot { T _ { f } }$ . We provided more in-depth discussion in Appendix C.1.

## 5.2 Lower Bound

As noted above, prior work only provided an algebraic lower bound of Ω(pd log $\Delta _ { f } )$ in Le, Nguyen, and Nguyen (2026b, Theorem $5 . 2 )$ , leaving the necessity of the combinatorial parameters $T _ { f }$ and $M _ { f }$ an open question. We address this gap by introducing a multi-regime lower bound framework. By independently analyzing these capacities, we prove that our upper bounds are tight with respect to the number of pieces $T _ { f }$ and nearly tight with respect to the number of boundaries $\dot { M _ { f } }$ , establishing the fundamental necessity ofeach structural parameter in our theoretical guarantees. The detailed proofs of Lemmas 5.2 and 5.3 can be found in Appendix C.2.

Lemma 5.2 (Lower bound depending on $T _ { f } )$ . For every $p , d \geq 1$ and an integer $T _ { f } \geq 4$ , there exists box like regions $\mathcal { A } \subset \mathbb { R } ^ { p } , \Theta \subset \mathbb { R } ^ { d }$ , an instance space $x ,$ , and a training objective $f ( x , \alpha , \theta )$ that admits piecewise polynomial structure with complexity $\big ( \lfloor \frac { T _ { f } - 1 } { 2 } \rfloor , T _ { f } , 1 \big )$ such that the corresponding function class $\mathcal { L }$ satisfies: Pdim $\left( \mathcal { L } \right) = \Omega \left( p \log T _ { f } \right)$

Lemma 5.3 (Lower bound depending on $M _ { f } )$ . For every $p , d , M _ { f } \ge 1$ , there exists box-like hyperparameter domain $A \subset \mathbb { R } ^ { p }$ and parameter domain $\Theta \subseteq \mathbb { R } ^ { d }$ , a finite instance space $x ,$ , and a training objective f such that:

(i) For any $x \in { \mathcal { X } } ,$ , the function $f _ { x } ( \alpha , \theta ) \triangleq f ( x , \alpha , \theta )$ admits a piecewise polynomial structure with exacts $M _ { f }$ afine boundary polynomials, at most $\biggl ( 1 + \frac { 2 M _ { f } } { d } \biggr ) ^ { d }$ realized sign conditions, piece polynomials ofdegree at most 2.

(ii) The induced class ${ \mathcal { L } } ~ = ~ \{ \ell _ { \alpha } ~ : ~ { \mathcal { X } } ~ \to ~ { \mathbb { R } } ~ | ~ \alpha ~ \in ~$ ${ \mathcal A } \} , \ell _ { \alpha } ( x ) = \mathrm { m i n } _ { \theta \in \Theta } f ( x , \overset { \cdot } { \alpha } , \theta )$ satisfies ${ \mathrm { P d i m } } ( { \mathcal { L } } ) \ =$ $\begin{array} { r } { \Omega \left( p d \log \left( 1 + \frac { M _ { f } } { d } \right) \right) } \end{array}$

Remark 3 (Tightness of the bounds). Our multi-regime lower bounds, combined with prior results, demonstrate that the pseudo-dimension upper bound in Theorem 5.1 is tight in practically relevant regimes. Specifically, Lemma 5.2 yields a lower bound of Ω(p log $T _ { f } )$ , and Le, Nguyen, and Nguyen (2026b, Theorem $5 . { \overset { \cdot } { 2 } } )$ establishes an algebraic lower bound of $\Omega ( p d \log \Delta _ { f } )$ . These match the respective dependencies in Theorem ${ 5 . } I , $ proving our bound is strictly tight with respect to both $T _ { f }$ and $\Delta _ { f }$ . Furthermore, Lemma 5.3 isolates the boundary capacity to establish a lower bound of $\Omega ( p d \log ( 1 + { \frac { M _ { f } } { d } } ) )$ , confirming that the theoretical dependency on $M _ { f }$ is nearly tight.

## 6 Tuning via Validation Objective

We now consider the general bi-level setting in which model parameters are selected using the training objective $f ,$ while the hyperparameters are evaluated using a potentially diferent validation objective g. Recall that $\begin{array} { r } { \mathring { \ell _ { \alpha } ^ { \mathrm { v a l } } } ( x ) ~ = ~ \operatorname* { i n f } _ { \theta \in S ( x , \alpha ) } g _ { x } ( \alpha , \theta ) } \end{array}$ , where $\begin{array} { r l } { \mathcal { S } ( x , \alpha ) } & { { } = } \end{array}$ arg min<sub>θ∈Θ</sub> $f _ { x } ( \alpha , \theta )$ . Unlike the training-loss setting of Section 5, analyzing this loss requires encoding validation performance and lower-level optimality simultaneously.

Theorem 6.1. Under Assumption 1, let $\mathcal { L } _ { \nu a l } = \{ \ell _ { \alpha } ^ { \nu a l } : \mathcal { X } \to$ $[ - H , H ] \mid \alpha \in \mathcal { A } \} , \Delta _ { f , g } = \operatorname* { m a x } \{ 1 , \Delta _ { f } , \Delta _ { g } \}$ , and $M _ { \mathrm { t o t } } =$ $\dot { M } _ { f } + \dot { M } _ { g } + T _ { f } + \dot { d } ,$ , we have

$$
\mathrm { P } \mathrm { d i m } ( \mathcal { L } _ { \nu a l } ) = O \left( p \log ( T _ { f } T _ { g } ) + p d ^ { 2 } \log \left( 2 + M _ { \mathrm { t o t } } \Delta _ { f , g } \right) \right) .
$$

The detailed proof can be found in Appendix D. Compared with prior multi-dimensional validation bounds in Le, Nguyen, and Nguyen (2026b, Theorem 6.1), Theorem 6.1 is strictly tighter by removing one factor p from the degreedependent term and moves the $T _ { g }$ dependence outside the $d ^ { 2 }$ -scaled logarithm. See Appendix D for a more detailed comparison.

## 7 Applications

To demonstrate the versatility of our nested block elimination framework, we analyze the data-driven Weighted Group Lasso (Yuan and Lin 2006; Le, Nguyen, and Nguyen 2026b). The training and validation objectives are formulated as: $\begin{array} { r } { f (  { \boldsymbol { x } } ,  { \boldsymbol { \alpha } } , \theta ) \overset {  } { = } \|  { \boldsymbol { A } } \theta -  { \boldsymbol { b } } \| _ { 2 } ^ { 2 } + \sum _ { i = 1 } ^ { p } \widetilde { \alpha _ { i } } \| \theta _ { i } \| _ { 2 } } \end{array}$ , and $g ( x , \alpha , \theta ) =$ $\begin{array} { r l } { { \frac { 1 } { 2 } } \| A ^ { \prime } \theta - b ^ { \prime } \| ^ { 2 } } \end{array}$ . Here, the model parameter θ is partitioned into $p$ distinct groups $\boldsymbol { \theta } = ( \theta _ { 1 } , \ldots , \theta _ { p } ) \in \mathbb { R } ^ { d _ { 1 } } \times \cdot \cdot \cdot \times \mathbb { R } ^ { d _ { p } }$ where $\textstyle \sum _ { i = 1 } ^ { p } d _ { i } = { \dot { d } }$ , and the hyperparameter $\alpha \in \mathbb { R } ^ { p }$ controls the penalty weight for each group. Using our proposed framework, we have the following results.

Theorem 7.1. Consider the class ofvalidation lossfunctions $\mathcal { L } _ { \nu a l } = \{ \ell _ { \alpha } ^ { \nu a l } : \mathcal { X }  [ - H , H ] | \overline { { { \alpha } } } \in \mathcal { A } \}$ induced by the Weighted Group Lasso objectives. The pseudo-dimension is bounded by Pdim $( \mathcal { L } _ { \nu a l } ) \stackrel { } { = } \mathcal { O } ( p ( d + p ) ^ { 2 } \log p )$

The proof directly leverages our nested block elimination framework in Theorem 4.2 by introducing auxiliary variables to encode the semi-algebraic $L _ { 2 }$ norm as purely polynomial constraints. Prior work utilizing standard quantifier elimination yielded a looser upper bound of $\mathcal { O } ( p ^ { 3 } d + p ^ { 2 } d ^ { 2 } )$ in Le, Nguyen, and Nguyen (2026b, Theorem 8.1). By preserving invariant connected sign cells during block elimination, Theorem 7.1 removes a full factor of $p$ from the dominant algebraic term, providing a strictly sharper generalization guarantee. See Appendix E.1 for further discussions on the improvement as well as the detailed proof. We also provided other applications to showcase the applicability of our general frameworks, which can be found in Appendix E.2.

## 8 Conclusion and Future Work

In this paper, we studied the statistical complexity of multidimensional data-driven hyperparameter tuning for bi-level optimization. By replacing standard quantifier elimination with a nested block elimination framework based on invariant connected sign cells, we bypassed topological over-counting to derive strictly tighter pseudo-dimension bounds. For the training-loss setting, our novel multi-regime lower bounds proved that these upper bounds are tightly saturated with respect to both combinatorial and algebraic capacities. We further extended this framework to general validation-loss tuning and to broader semi-algebraic structures, such as the Weighted Group Lasso, thereby eliminating the inflated algebraic dependencies of prior work. While our lower bounds confirm the tightness of our guarantees in the training-loss setting $( f \equiv g )$ , determining the fundamental minimax lower bounds for the general bi-level validation-loss setting $( f \not \equiv g )$ remains a compelling open problem for future research.

## References

a Ilemobayo, J.; Durodola, O.; Alade, O.; J Awotunde, O.; T Olanrewaju, A.; Falana, O.; Ogungbire, A.; Osinuga, A.; Ogunbiyi, D.; Ifeanyi, A.; et al. 2024. Hyperparameter tuning in machine learning: A comprehensive review. Journal of Engineering Research and Reports, 26(6): 388–395.

Balcan, M.-F. 2020. Data-driven algorithm design. arXiv preprint arXiv:2011.07177.

Balcan, M.-F.; DeBlasio, D.; Dick, T.; Kingsford, C.; Sandholm, T.; and Vitercik, E. 2021. How much data is suficient to learn high-performing algorithms? generalization guarantees for data-driven algorithm design. In Proceedings ofthe 53rd Annual ACM SIGACT Symposium on Theory of Computing, 919–932.

Balcan, M. F.; Nguyen, A. T.; and Sharma, D. 2025a. Algorithm Configuration for Structured Pfafian Settings. Transactions on Machine Learning Research.

Balcan, M. F.; Nguyen, A. T.; and Sharma, D. 2025b. Sample complexity of data-driven tuning of model hyperparameters in neural networks with structured parameter-dependent dual function. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Balcan, M. F.; Prasad, S.; Sandholm, T.; and Vitercik, E. 2022a. Structural analysis of branch-and-cut and the learnability of Gomory mixed integer cuts. Advances in Neural Information Processing Systems.

Balcan, M.-F. F.; Khodak, M.; Sharma, D.; and Talwalkar, A. 2022b. Provably tuning the ElasticNet across instances. Advances in Neural Information Processing Systems, 35: 27769–27782.

Balcan, M.-F. F.; Nguyen, A.; and Sharma, D. 2023. New bounds for hyperparameter tuning of regression problems across instances. Advances in Neural Information Processing Systems, 36: 80066–80078.

Bartlett, P.; Indyk, P.; and Wagner, T. 2022. Generalization bounds for data-driven numerical linear algebra. In Conference on Learning Theory, 2013–2040. PMLR.

Bartlett, P.; Maiorov, V.; and Meir, R. 1998. Almost linear VC dimension bounds for piecewise polynomial networks. Advances in Neural Information Processing Systems, 11.

Bartlett, P. L.; Harvey, N.; Liaw, C.; and Mehrabian, A. 2019. Nearly-tight VC-dimension and pseudodimension bounds for piecewise linear neural networks. Journal ofMachine Learning Research, 20(63): 1–17.

Basu, S.; Pollack, R.; and Roy, M.-F. 2005. On the Betti numbers of sign conditions. Proceedings of the American Mathematical Society, 133(4): 965–974.

Basu, S.; Pollack, R.; and Roy, M.-F. 2006. Algorithms in Real Algebraic Geometry. Springer.

Bergstra, J.; Bardenet, R.; Bengio, Y.; and Kégl, B. 2011. Algorithms for hyper-parameter optimization. Advances in Neural Information Processing Systems, 24.

Cheng, H.; and Basu, A. 2026. Generalization guarantees for learning score-based branch-and-cut policies in integer programming. Advances in Neural Information Processing Systems, 38: 118669–118699.

Goldberg, P.; and Jerrum, M. 1993. Bounding the Vapnik-Chervonenkis dimension of concept classes parameterized by real numbers. In Proceedings ofthe Sixth Annual Conference on Computational Learning Theory, 361–369.

Gupta, R.; and Roughgarden, T. 2020. Data-driven algorithm design. Communications ofthe ACM, 63(6): 87–94.

Hazan, E.; Klivans, A.; and Yuan, Y. 2018. Hyperparameter optimization: A spectral approach. ICLR.

Indyk, P.; Vakilian, A.; and Yuan, Y. 2019. Learning-based low-rank approximations. Advances in Neural Information Processing Systems, 32.

Le, T. Q.; Nguyen, A. T.; and Nguyen, V. A. 2026a. Provably Data-driven Lagrangian Relaxation for Mixed Integer Linear Programming. In Forty-third International Conference on Machine Learning.

Le, T. Q.; Nguyen, A. T.; and Nguyen, V. A. 2026b. Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function. In Forty-third International Conference on Machine Learning.

Li, L.; Jamieson, K.; DeSalvo, G.; Rostamizadeh, A.; and Talwalkar, A. 2017. Hyperband: A novel bandit-based approach to hyperparameter optimization. Journal ofMachine Learning Research, 18(1): 6765–6816.

Li, Y.; Lin, H.; Liu, S.; Vakilian, A.; and Woodruf, D. P. 2023. Learning the positions in countsketch. arXiv preprint arXiv:2306.06611.

Montúfar, G.; Pascanu, R.; Cho, K.; and Bengio, Y. 2014. On the number of linear regions of deep neural networks. Advances in Neural Information Processing Systems, 27.

Nguyen, A. T.; and Nguyen, V. A. 2026. Provably data-driven projection method for quadratic programming. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 24541–24548.

Pollard, D. 1984. Convergence of Stochastic Processes. Springer New York.

Probst, P.; Bischl, B.; and Boulesteix, A.-L. 2018. Tunability: Importance of hyperparameters of machine learning algorithms. arXiv preprint arXiv:1802.09596.

Renegar, J. 1992. On the computational complexity and geometry of the first-order theory of the reals. Part I: Introduction. Preliminaries. The geometry of semi-algebraic sets. The decision problem for the existential theory of the reals. Journal ofSymbolic Computation, 13(3): 255–299.

Sakaue, S.; and Oki, T. 2024. Generalization bound and learning methods for data-driven projections in linear programming. Advances in Neural Information Processing Systems, 37: 12825–12846.

Yuan, M.; and Lin, Y. 2006. Model Selection and Estimation in Regression With Grouped Variables. Journal ofthe Royal Statistical Society Series B, 68: 49–67.

# A Additional Definitions and Results

## A.1 The Goldberg-Jerrum framework

The Goldberg-Jerrum (GJ) framework was introduced by Goldberg and Jerrum (1993) and subsequently refined by Bartlett, Indyk, and Wagner (2022). It provides a systematic mechanism to bound the pseudo-dimension of parameterized function classes. Specifically, it applies to any function class $\mathcal { L }$ where the evaluation of a function $\ell _ { \alpha }$ can be modeled as a GJ algorithm. Such algorithms are restricted to elementary arithmetic operations $( + , - , \times , \div )$ and conditional branching, producing intermediate values that naturally take the form of rational functions of the input parameter $\alpha .$ We formally define this computational model below:

Definition 8 (GJ algorithm, (Bartlett, Indyk, and Wagner 2022)). A GJ algorithm Γ operates on real-valued inputs and is restricted to two types of operations:

• Arithmetic assignments oftheform $v ^ { \prime \prime } = v \odot v ^ { \prime } ,$ , where $\odot \in \{ + , - , \times , { \div } \}$ , and

• Conditional branching of the form ${ { \mathrm { \Omega } } ^ { \ast } } i f v \geq 0$ . . . else $\cdots ^ { \prime \prime } .$

In both instances, the operands v and $v ^ { \prime }$ must either be external inputs or intermediate values previously generated by the algorithm.

Because the intermediate variables $v , v ^ { \prime } , v ^ { \prime \prime }$ are generated solely through sequential arithmetic operations on the input $\alpha ,$ they inherently manifest as rational functions. The complexity of a given GJ algorithm is entirely characterized by the algebraic properties of these intermediate rational functions, formally captured by two metrics: degree and predicate complexity.

Definition 9 (Complexities of GJ algorithm, (Bartlett, Indyk, and Wagner 2022)). The degree ofa GJ algorithm is the maximum degree of any rational function it computes based on its inputs. The predicate complexity is defined as the total number of distinct rationalfunctions evaluated within its conditional statements. For a rationalfunction $\begin{array} { r } { f ( \alpha ) = \frac { g ( \alpha ) } { h ( \alpha ) } } \end{array}$ , where g and h are polynomials in α, its degree is given by $\deg ( f ) = \operatorname* { m a x } \{ \deg ( g ) , \deg ( h ) \}$

When the evaluation of every function in a parameterized class L can be executed by a GJ algorithm with bounded algebraic and predicate complexities, we can invoke the following foundational theorem to explicitly bound its pseudo-dimension.

Theorem A.1 (Bartlett, Indyk, and Wagner (2022, Theorem 3.3)). Suppose that each function $\ell _ { \alpha } \in \mathcal { L }$ is specified by p real parameters $\alpha \in \mathbb { R } ^ { p }$ . Supposefurther thatfor every problem instance x ∈ X and real-valued threshold $t \in \mathbb { R } ,$ , there exists a GJ algorithm $\Gamma _ { x , t }$ that takes α as input and returns $t r u e ^ { , , } i f \ell _ { \alpha } ( x ) \geq t$ and $f a l s e '$ otherwise. ${ \cal I } f \Gamma _ { x , t }$ exhibits a maximum degree of $\Delta$ and a predicate complexity ofΛ, then Pdim $( \mathcal { L } ) = \hat { \mathcal { O } } ( p \log ( \Delta \Lambda ) )$ ).

In the next section, we will discuss in depth the idea in prior works, its limitation, and how we overcome this with the new block elimination argument.

## A.2 A detailed discussion on the technical diference compared to prior works

Technique in prior works and its limitation. In prior work, the standard approach to bounding the pseudo-dimension relies on a rigid two-step pipeline. First, the bi-level optimization objective is formulated as a polynomial First-Order Logic (FOL) statement, and a standard Quantifier Elimination (QE) algorithm is applied to convert it into a quantifier-free FOL. Second, the authors apply an of-the-shelf result—the Goldberg-Jerrum (GJ) framework—which treats this quantifier-free formula as a computational algorithm to derive the final pseudo-dimension bounds. However, the fundamental problem with this approach is that explicit QE is an exceptionally costly, worst-case algebraic procedure. To eliminate quantifiers, the algorithm forces the construction of a massive number of intermediate, redundant polynomial structures to algebraically separate every conceivable outcome. Because the of-the-shelf GJ framework blindly accounts for the quantities and degrees of all these newly generated, bloated polynomials, the resulting pseudo-dimension bounds sufer from severe algebraic inflation (such as the suboptimal $p ^ { 2 }$ dependency).

Our proposed technique, the diference, and how it resolves the limitation. In contrast, our proposed technique takes a more fundamental route. We recognize that the ultimate goal of bounding the pseudo-dimension does not actually require evaluating an explicit logic formula; it strictly requires bounding the total number of distinct sign patterns (the shattering coeficient) that the thresholded loss functions can realize across a dataset. Instead of algebraic manipulation, we use a geometric strategy based on nested block elimination to track how logical truth values remain invariant across topologically connected sign cells. By doing so, we completely bypass the need to ever construct the final quantifier-free FOL. Because we do not explicitly generate the redundant polynomial structures required by standard QE, we do not incur their associated complexity blowups. We simply bound the number of invariant sign regions directly, achieving the same ultimate goal—bounding the sign patterns—but doing so with drastically improved, tightly saturated pseudo-dimension bounds.

## A.3 Additional Definitions

In this section, we will formally define the notion of cells (or connected components).

Definition 10 (Connectedness and connected sign cells (Basu, Pollack, and Roy 2006)). $L e t S \subseteq \mathbb { R } ^ { d }$ be a subset $o f \mathbb { R } ^ { d } .$ . A subset $U \subseteq S$ is called open relative to S ifthere exists an open set $O \subseteq \mathbb { R } ^ { d }$ such that $U = S \cap O$ . The set S is called connected ifthere does not exist two non-empty, disjoint subsets $U , V _ { \ast }$ , both open relative to S, such that $S = U \cup V . A$ connected component C ofS is a maximal connected subset ofS, that is (i) C is connected, and (ii) there is no connected set $C ^ { \prime }$ such that $C \subseteq C ^ { \prime } \subseteq S .$

## A.4 Supporting Lemmas

The following inequality is a well-known and frequently used result in the learning theory literature. For completeness, we present its formal statement and detailed proof below.

Proposition A.2. For any $t \geq 0$ and $A \geq 2 , i f 2 ^ { t } \leq c ( t + 2 ) A$ for a constant $c \geq 1$ , then $t = \mathcal { O } ( \log A )$

Proof. For convenience, we write $\begin{array} { r } { x = \frac { t } { 2 } } \end{array}$ for any $t \geq 0$ . Note that $x \leq 2 ^ { x }$ for all $x \geq 0$ , we have

$$
t + 2 = 2 ( x + 1 ) \leq 2 ( 2 ^ { x } + 2 ^ { x } ) = 4 \cdot 2 ^ { t / 2 } .
$$

From the assumption, we have $2 ^ { t } \leq c ( t + 2 ) A$ . Combining with the above, we have

$$
2 ^ { t } \leq 4 c A 2 ^ { t / 2 } \Rightarrow 2 ^ { t / 2 } \leq 4 A c \Rightarrow t \leq 2 \log _ { 2 } ( 4 c A ) .
$$

Since $A \geq 2 ,$ we have

$$
\log _ { 2 } ( 4 A c ) = \log _ { 2 } A + \log _ { 2 } ( 4 c ) = { \mathcal { O } } ( \log A ) .
$$

From the above, we have the final conclusion.

## B Additional Results and Omitted Proofs for Section 4

Proposition 4.1 (restated). Let $\Phi ( z ) = \big ( q _ { 1 } t ^ { ( 1 ) } \in \mathbb { R } ^ { d ^ { ( 1 ) } } \big ) \dots \big ( q _ { K } t ^ { ( K ) } \in \mathbb { R } ^ { d ^ { ( K ) } } \big ) \psi ( z , t ^ { ( 1 ) } , \dots , t ^ { ( K ) } )$ be a polynomial $F O L ,$ and let $\hat { \mathcal { P } }$ be thefamily ofatomic polynomials appearing in the atomic predicates ofψ. IfQ is a nested sign-invariant projection of $\mathcal { P }$ relative to the ordered blocks $\mathbf { \tilde { \chi } } _ { t } ( 1 ) , \dots , t ^ { ( \tilde { K } ) }$ , then Φ has a constant truth value on every connected sign cell $C \in { \mathrm { C e l l } } ( { \mathcal { Q } } )$ . In other words, for every $C \in { \bf C e l l } ( \mathcal { Q } )$ and every $z _ { 1 } , z _ { 2 } \in C ,$ , we have $\Phi ( z _ { 1 } ) \Leftrightarrow \Phi ( z _ { 2 } )$

Proof. For every sign condition $\eta = ( \eta _ { 1 } , \dots , \eta _ { s } ) \in \{ - 1 , 0 , 1 \} ^ { s }$ Let $E _ { K } ( \eta )$ be the truth value obtained by evaluating every atomic predicate $P _ { i } \chi _ { i } 0$ using the polynomial sign $\eta _ { i }$ . Let $\Sigma _ { K } , \Sigma _ { K - 1 } , \dots , \Sigma _ { 0 }$ be the nested block-sign profiles of $\mathcal { P }$ (as in Definition 5). Recursively, for $k = \mathbf { \bar { \boldsymbol { K } } } , \mathbf { \bar { \boldsymbol { K } } } - 1 , \dots , 1$ , and every possible value $S$ of the nested profile $\Sigma _ { k - 1 }$ , we define

$$
E _ { k - 1 } ( S ) = { \binom { \bigvee } { \bigwedge _ { \xi \in S } E _ { k } ( \xi ) , q _ { k } = \bigtriangledown } } ,
$$

Indeed, $\Sigma _ { k - 1 }$ records all the values of $\Sigma _ { k }$ obtained by varying $t ^ { ( k ) }$ . Therefore, an existential quantifier $( \mathrm { e . g , \exists } )$ takes the logical OR over the these values, whereas a universal quantifier $( \mathrm { e . g . , \forall ) }$ takes the logical AND. Applying this argument recursively from the innermost block $t ^ { ( K ) }$ to the outermost block $t ^ { ( 1 ) }$ , the truth value of $\Phi ( z )$ is $\begin{array} { r } { E _ { 0 } ( \Sigma _ { 0 } ( z ) ) } \end{array}$ . Since $\mathcal { Q }$ is a nested sign invariant projection, by Definition 6, we have $\Sigma _ { 0 } ( z _ { 1 } ) = \Sigma _ { 0 } ( z _ { 2 } )$ , whenever $z _ { 1 } , z _ { 2 }$ belong to the same $C \in { \mathrm { C e l l } } ( { \mathcal { Q } } )$ . Therefore $E _ { 0 } ( \Sigma _ { 0 } ( z _ { 1 } ) ) \stackrel {  } { = } E _ { 0 } ( \Sigma _ { 0 } \stackrel {  } { ( } z _ { 2 } ) )$ , meaning that the truth value of $\Phi ( z _ { 1 } )$ and $\Phi ( z _ { 2 } )$ is identical. □

Theorem 4.2 (restated). Consider a real-valued function class ${ \mathcal { F } } = \{ f _ { \alpha } : { \mathcal { X } }  \mathbb { R } \mid \alpha \in { \mathcal { A } } \}$ be a real-valued function class, where $\mathcal { A } = [ \alpha _ { \mathrm { m i n } } , \alpha _ { \mathrm { m a x } } ] ^ { p } \subseteq \mathbb { R } ^ { p }$ . Suppose that there are integers $L , M , K \ge 1$ , where K is fixed, block dimensions $d _ { 1 } , \dotsc , d _ { K } \geq 1$ , and a degree upper bound $\Delta \geq 1$ satisfying the following conditions: for any $x \in \mathcal { X }$ and and real-valued threshold $\tau \in \mathbb { R } ,$ , there exists

1. L polynomial FOLs $\Phi _ { x , \tau , 1 } ( \alpha ) , \ldots , \Phi _ { x , \tau , L } ( \alpha ) ,$ , and

2. a Boolean function $\beta _ { x , \tau } : \{ 0 , 1 \} ^ { L } \to \{ 0 , 1 \}$

such that $\mathbb { I } ( f _ { \alpha } ( x ) \ge \tau ) = \beta _ { x , \tau } ( \Phi _ { x , \tau , 1 } ( \alpha ) , \ldots , \Phi _ { x , \tau , L } ( \alpha ) )$ for every $\alpha \in { \mathcal { A } } .$ Furthermore, assuming that every $F O L \Phi _ { x , \tau , \ell } .$

1. has K ordered quantified variables blocks of dimension $d _ { 1 } , \ldots , d _ { K }$ , respectively, and

2. contains at most M distinct atomic polynomials, each ofdegree at most $\Delta .$

Define $\begin{array} { r } { A _ { K } = \prod _ { k = 1 } ^ { K } ( d _ { k } + 1 ) } \end{array}$ and $\begin{array} { r } { B _ { K } = \prod _ { k = 1 } ^ { K } d _ { k } } \end{array}$ , then the pseudo-dimension of the function class $\mathcal { F }$ is at most

$$
\mathrm { P d i m } ( \mathcal { F } ) = \mathcal { O } \left( p \log ( 2 L ) + p A _ { K } \log ( 2 M ) + p B _ { K } \log ( 2 \Delta ) \right) .
$$

Proof. By the definition of pseudo-dimension $\operatorname { P d i m } ( { \mathcal { F } } )$ , our goal is to bound the maximum size $N$ of a set of problem instances $S = \{ x _ { 1 } , \ldots , x _ { N } \}$ that $\mathcal { F }$ can shattered and with real-valued thresholds $\tau = ( \tau _ { 1 } , \dots , \tau _ { N } ) \subset \mathbb { R }$ witness the shattering, that is, the binary vectors

$$
v ( ( S , \tau ) , f _ { \alpha } ) = \big ( \mathbb { I } ( f _ { \alpha } ( x _ { 1 } ) \ge \tau _ { 1 } ) , \dots , \mathbb { I } ( f _ { \alpha } ( x _ { N } ) \ge \tau _ { N } ) \big )
$$

admits all possible sign vectors $\{ 0 , 1 \} ^ { N }$ when varying $\alpha \in { \mathcal { A } }$ . This can be achieved by giving an upper bound for the shattering coeficient $\mathcal { N } ( \mathcal { F } , N )$ of the function class $\mathcal { F }$ with respect to $N$ , representing the maximum number of possible sign patterns $\{ v ( S , f _ { \alpha } ) | \stackrel { . } { \alpha } \in \stackrel { . } { A } \}$ for any $S \in \mathcal { X } ^ { N }$ and $\tau \in \mathbb { R } ^ { N }$ , and then solving for the maximum N that satisfies the inequality $\dot { 2 } ^ { N } \le \dot { \mathcal { N } } ( \dot { \mathcal { F } } , N )$ . We will proceed in the following steps.

Step 1: Branch elimination. For every $i = 1 , \ldots , N$ and $\ell = 1 , \ldots , L$ , let $\mathcal { P } _ { i , \ell }$ be the set of distinct atomic polynomials appearing in $\Phi _ { x _ { i } , \tau _ { i } , \ell } ( \alpha )$ . From assumption, we have: (1) $| \mathcal { P } _ { i , \ell } | \ \leq \ M$ , and (2) max $\mathfrak { r } \in { \mathcal { P } } _ { i , \ell } \deg ( R ) \ \leq \ \Delta$ . Let $c _ { K }$ be the constant appearing in Theorem 2.2, and define $S _ { K } = M ^ { A _ { K } } \Delta ^ { c _ { K } B _ { K } }$ and $D _ { K } = \Delta ^ { c _ { K } B _ { K } }$ . By Theorem $2 . 2 ,$ , for every pair $( i , \ell )$ , there exists a nested sign-invariant projection $\mathcal { Q } _ { i , \ell } \subseteq \mathbb { R } [ \alpha ]$ (Definition 6) of $\mathcal { P } _ { i , \ell }$ such that: $\left( 1 \right) \left| \mathcal { Q } _ { i , \ell } \right| \le S _ { K }$ , and (2) max $R \in \mathcal { Q } _ { i , \ell } \deg ( R ) \ \leq \ D _ { K }$ . By Proposition 4.1, the truth value of $\Phi _ { x _ { i } , \tau _ { i } , \ell } ( \alpha )$ is constant on every connected sign cel $C \in \mathrm { C e l l } _ { \mathcal { Q } _ { i , \ell } }$

Step 2: Construct one global polynomial family: Let $\mathcal { B } _ { A } = \{ \alpha _ { j } - \alpha _ { \operatorname* { m i n } } , \alpha _ { \operatorname* { m a x } } - \alpha _ { j } \ | \ j = 1 , \ldots , p \}$ be the family of $2 p$ afine polynomials defining the box-like region $\mathcal { A } = [ \alpha _ { \mathrm { m i n } } , \alpha _ { \mathrm { m a x } } ] ^ { j }$ . We then define the global polynomial set

$$
\tilde { \mathcal { Q } } = \mathcal { B } _ { \mathcal { A } } \cup \bigcup _ { i = 1 } ^ { N } \bigcup _ { \ell = 1 } ^ { L } \mathcal { Q } _ { i , \ell } .
$$

Let $q = \left| \tilde { \mathcal { Q } } \right|$ . Because there are $N L$ branchwise projection families $\mathcal { Q } _ { i , \ell }$ , each contains at most $S _ { K }$ polynomials, we have $q \leq N L S _ { K } + 2 p$ .Besides, every polynomial in $\tilde { \mathcal { Q } }$ has degree at most $D _ { K }$

Step 3: The labeling vector is constant on every global cell. Consider any connected sign cel $C \in { \bf C e l l } ( \tilde { \mathcal { Q } } )$ . Fix an instance $x _ { i }$ and a branch ℓ. Since $\mathcal { Q } _ { i , \ell } \subseteq \tilde { \mathcal { Q } } ,$ , the sign of all polynomials in $\mathcal { Q } _ { i , \ell }$ are constant on C. Therefore, there exists a sign condition $\sigma _ { i , \ell }$ on $\mathcal { Q } _ { i , \ell }$ such that $C \subseteq$ Reali $\boldsymbol { \mathbf { \rho } } _ { \mathcal { Q } _ { i , \ell } } \left( \boldsymbol { \sigma } _ { i , \ell } \right)$ , and therefore in one cell belonging to $\mathrm { C e l l } _ { \mathcal { Q } _ { i , \ell } } .$ Proposition 4.1 now implies that the truth value of $\Phi _ { x _ { i } , \tau _ { i } , \ell } ( \alpha )$ is unchanged as α varies over $C .$

Note that the above holds for every $\ell = 1 , \ldots , L$ . Therefore, the binary vector $( \Phi _ { x _ { i } , \tau _ { i } , 1 } ( \alpha ) , \dots , \Phi _ { x _ { N } , \tau _ { N } , 1 } ( \alpha ) )$ remains unchanged on C. Because $\beta _ { x _ { i } , \tau _ { i } }$ is a fixed boolean function, the truth value of

$$
\mathbb { I } ( f _ { \alpha } ( x _ { i } ) \ge \tau _ { i } ) = \beta _ { x _ { i } , \tau _ { i } } ( \Phi _ { x _ { i } , \tau _ { i } , 1 } ( \alpha ) , \dots , \Phi _ { x _ { i } , \tau _ { i } , L } ( \alpha ) )
$$

is also unchanged for $\alpha \in C$

Applying this argument to every $i = 1 , \ldots , N ,$ , we claim that

$$
v ( ( S , \tau ) , f _ { \alpha } ) = \big ( \mathbb { I } ( f _ { \alpha } ( x _ { 1 } ) \ge \tau _ { 1 } ) , \dots , \mathbb { I } ( f _ { \alpha } ( x _ { N } ) \ge \tau _ { N } ) \big )
$$

also remains unchanged for $\alpha \in C$ . Therefore, the task now is to bound $| \mathrm { C e l l } ( \tilde { \mathcal { Q } } |$ and solve for the maximum N that satisfies $2 ^ { N } \leq | \mathbf { C e l l } ( \tilde { \mathcal { Q } } |$

Step 4: Establish the pseudo-dimension upper bound. The set of polynomials $\tilde { \mathcal { Q } }$ contains $q \geq p$ polynomials in p variables, each of degree at most $\Delta _ { K }$ . From Lemma 2.3, there exists a universal constant $\kappa > 0 .$ , such that $\begin{array} { r } { \left| \mathrm { C e l l } ( \tilde { \mathcal { Q } } \right| \leq \left( \frac { \kappa q D _ { K } } { p } \right) ^ { p } } \end{array}$ . Note that $q \leq N L S _ { K } + 2 p$ , our task now is to solve the inequality

$$
\begin{array} { c } { 2 ^ { N } \displaystyle \leq \left( \frac { \kappa ( N L S _ { K } + 2 p ) D _ { K } } { p } \right) ^ { p } } \\ { \displaystyle \Leftrightarrow 2 ^ { N / p } \leq \left( \frac { \kappa ( N L S _ { K } + 2 p ) D _ { K } } { p } \right) . } \end{array}
$$

For convenience, let $\begin{array} { r } { u = \frac { N } { p } } \end{array}$ , and the above becomes

$$
2 ^ { u } \le \kappa D _ { K } ( u L S _ { K } + 2 ) \le \kappa ( u + 2 ) L S _ { K } D _ { K } .
$$

Applying Proposition A.2, we have $u ~ = ~ \mathcal { O } ( \log ( 2 L S _ { K } D _ { K } )$ , or $N ~ = ~ \mathcal { O } ( p \log ( 2 L S _ { K } D _ { K } ) )$ . Finally, substitute $\begin{array} { r l } { S _ { K } } & { { } = } \end{array}$ $\dot { M } ^ { \dot { A } \dot { K } } \Delta ^ { \check { c } _ { K } B _ { K } }$ and $D _ { K } = \Delta ^ { c _ { K } B _ { K } }$ onto that, we have the final conclusion. □

## C Additional Results and Omitted Proofs for Section 5

## C.1 Omitted Proofs for Section 5.1

In this section, we will present the detailed proof for Theorem 5.1, which establishes the guarantee for data-driven hyperparameter tuning. We then provided a detailed discussion on the improvement of our results, compared to prior works as below.

Theorem 5.1 (restated). Let $\mathcal { A } = [ \alpha _ { \mathrm { m i n } } , \alpha _ { \mathrm { m a x } } ] ^ { p }$ and $\Theta = [ \theta _ { \mathrm { m i n } } , \theta _ { \mathrm { m a x } } ] ^ { d }$ be the hyperparameter and parameter domains, respectively. Assume thatfor any problem instance x $: \in { \mathcal { X } } ,$ , the training objective $f _ { x } ( \alpha , \theta ) \triangleq f ( x , \alpha , \theta ) ,$ ,for any $( \alpha , \theta ) \in \mathcal { A } \times \Theta$ admits a piecewise polynomial structure with complexity $( M _ { f } , T _ { f } , \bar { \Delta _ { f } } )$ . Then, the pseudo-dimension of the function class ${ \mathcal { L } } = \{ \ell _ { \alpha } : { \mathcal { X } } \to \mathbb { R } \mid { \overline { { \alpha } } } \in { \mathcal { A } } \}$ , where $\ell _ { \alpha } ( x ) = \mathrm { m i n } _ { \theta \in \Theta } f ( x , \alpha , \theta )$ , is at most

$$
\mathrm { P d i m } ( \mathcal { L } ) = \Omega ( p \log T _ { f } + p d \log ( M _ { f } + d ) + p d \log \Delta _ { f } ) .
$$

Proof. Consider any problem instance $x \in \mathcal { X }$ and real-valued threshold $\tau \in \mathbb { R }$ . Our goal is to show that: the binary value $\mathbb { I } ( \ell _ { x } ( \alpha ) \ge \tau )$ , where $\ell _ { x } ( \alpha ) \triangleq \ell _ { \alpha } ( x ) = \operatorname* { m i n } _ { \theta \in \Theta } ( x , \alpha , \theta )$ , as a function of α can be rewritten as a boolean combination of at most $T _ { f }$ polynomial FOLs $\Phi _ { x , \tau , l } ,$ each satisfying the complexity conditions as in Theorem 4.2. This can be done as follows.

Step 1: Piecewise polynomial representation of $f _ { x } ( \alpha , \theta ) \triangleq f ( x , \alpha , \theta )$ . By assumption, for each problem instance x, there exists: (1) a set of boundary polynomials $\mathcal { H } _ { x } = \{ h _ { x , 1 } , \hdots , h _ { x , M _ { x } } \} \subset \bar { \mathbb { R } } [ \alpha , \theta ]$ where $M _ { x } \le M _ { f } , \mathsf { \bar { ( 2 ) } }$ a set of realized sign patterns $\Sigma _ { x } = \{ - 1 , 0 , 1 \} ^ { m _ { a } }$ , where $| \Sigma _ { x } | \le T _ { f }$ , and (3) a set of piece polynomials $\{ P _ { x , \sigma } \} _ { \sigma \in \Sigma _ { x } } \subset \mathbb { R } [ \alpha , \theta ]$ such that every boundary polynomials $\therefore \ r _ { m } ( \ r _ { 0 } r m = 1 , \ldots , M _ { x } )$ and piece polynomials $\bar { P } _ { x , \sigma } \left( \mathrm { f o r } \sigma \in \bar { \Sigma } _ { x } \right)$ has degree at most $\Delta _ { f }$ . Those piece and boundary polynomials determine the form of $f _ { x } ( \alpha , \theta )$ as follows.

For a sign pattern $\boldsymbol { \sigma } = \left( \sigma _ { 1 } , \ldots , \sigma _ { M _ { x } } \right) \in \Sigma _ { x }$ , we define the region predicate

$$
\operatorname { W h e r e } _ { x , \sigma } ( \alpha , \theta ) \triangleq \bigwedge _ { m = 1 } ^ { M _ { x } } [ \mathrm { s i g n } ( h _ { x , m } ( \alpha , \theta ) = \sigma _ { m } ] .
$$

For each $( \alpha , \theta ) \in \mathcal { A } \times \Theta$ , exact one sign pattern $\sigma \in \Sigma _ { x }$ is active, and on its corresponding region, $f _ { x } ( \alpha , \theta )$ admits the polynomial forms $f _ { x } ( \alpha , \theta ) = P _ { x , \sigma } ( \alpha , \theta )$

Step 2: Rewrite $\ell _ { x } ( \alpha )$ in the polynomial FOL form. Recall that

$$
\ell _ { x } ( \alpha ) = \operatorname* { m i n } _ { \theta \in \Theta } f _ { x } ( \alpha , \theta ) .
$$

Therefore $\ell _ { x } ( \alpha ) \geq \tau$ for a real-valued threshold τ if and only if for all $\theta \in \Theta$ , we have $f _ { x } ( \alpha , \theta ) \ge \tau$ . Combining with Step 2, we can write the binary value $\mathbb { I } ( \ell _ { x } ( \alpha ) \geq \tau )$ as

$$
\bigwedge _ { \sigma \in \Sigma _ { x } } ( \forall \theta \in \Theta ) \left[ \mathrm { W h e r e } _ { x , \sigma } ( \alpha , \theta ) \Rightarrow P _ { x , \sigma } ( \alpha , \theta ) \geq \tau \right] = \bigwedge _ { \sigma \in \Sigma _ { x } } \Phi _ { x , \tau , \sigma } ( \alpha ) ,
$$

where $\mathrm { W h e r e } _ { x , \sigma } ( \alpha , \theta )$ is a logical statement defined as in Step 2, and

$$
\begin{array} { r } { \Phi _ { x , \tau , \sigma } ( \alpha ) = ( \forall \theta \in \Theta ) [ \mathrm { W h e r e } _ { x , \sigma } ( \alpha , \theta ) \Rightarrow P _ { x , \sigma } ( \alpha , \sigma ) \geq \tau ] } \\ { = ( \forall \theta \in \mathbb { R } ^ { d } ) [ \neg \mathrm { I n } _ { \Theta } ( \theta ) \lor \neg \mathrm { W h e r e } _ { x , \sigma } ( \alpha , \theta ) \lor P _ { x , \sigma } ( \alpha , \sigma ) \geq \tau ] . } \end{array}
$$

Here, $\begin{array} { r } { \mathrm { I n } _ { \Theta } ( \theta ) = \bigwedge _ { r = 1 } ^ { d } [ \theta _ { r } \geq \theta _ { \mathrm { m i n } } ] \wedge [ \Theta _ { \mathrm { m a x } } \geq \theta _ { r } ] } \end{array}$ denotes the binary indicator if $\theta \in \Theta = [ \theta _ { \operatorname* { m i n } } , \theta _ { \operatorname* { m a x } } ] ^ { d } .$ , and in the above we use the logical identity that $( A \overset { \vartriangle } { \Rightarrow } B ) = ( \neg A \lor B )$ ). We now see that $\mathbb { I } ( \ell _ { x } ( \alpha ) \geq \tau )$ is written in the form $\beta _ { x , \tau } ( \{ \Phi _ { x , \tau , 1 } \} _ { \sigma \in \Sigma } )$ where $\beta _ { x , \tau } ( \{ b _ { \sigma } \} _ { \sigma \in \Sigma _ { x } } ) = \bigwedge _ { \sigma \in \Sigma _ { \sigma } } b _ { \sigma }$ , as required by Theorem 4.2.

Step 3: Count the complexity of each brand and apply Theorem 4.2. The task left is to bound the complexity of $\Phi _ { x , \tau , \sigma } .$ Since each polynomial FOL contains exactly one quantified block $\forall \theta \in \mathbb { R } ^ { d }$ , we have $K = 1$ and $d _ { 1 } = d .$

Note that we have: (1) $M _ { x } \le M _ { f }$ boundary polynomials $h _ { x , 1 } , \ldots , h _ { x , M _ { x } } , ( 2 )$ 2d domain polynomials $\theta _ { r } - \theta _ { \operatorname* { m i n } }$ and $\theta _ { \mathrm { m a x } } - \theta _ { \ i }$ for $r = 1 , \ldots , d ,$ and (3) a single value polynomial $P _ { x , \sigma } ( \alpha , \theta ) ^ { \cdot } - \tau$ . Therefore, each brand contains at most $M _ { f } + 2 d + 1$ distinct atomic polynomial predicates. Finally, the maximum degree of all atomic polynomial predicates is simply at most $\Delta _ { f }$ . Finally, applying Theorem 4.2, we obtain

$$
\mathrm { P d i m } ( \mathcal { L } ) = \mathcal { O } ( p \log T _ { f } + p d \log ( M _ { f } + d ) + p d \log \Delta _ { f }
$$

as desired.

Remark 4. Compared to prior work (Le, Nguyen, and Nguyen 2026b), whose standard quantifier-argument gives the upper bound

$$
\mathcal { O } ( p d \log ( M _ { f } + T _ { f } + d ) + p ^ { 2 } d \log \Delta _ { f } ) ,
$$

our proposed result (Theorem 5.1) instead gives

$$
\mathcal { O } ( p \log T _ { f } + p d \log ( M _ { f } + d ) + p d \log \Delta _ { f } ) .
$$

Therefore, nested block elimination removes on factor p from the degree-dependent term and decouples the number of pieces $T _ { f }$ from the d-scaled logarithm. These improvements are supported by complementary lower bounds, which we will present in details below.

## C.2 Proofs for Section 5.2

In this section, we will formally present the proofs of Lemmas 5.2 and 5.3, which establish the lower-bound depending on the number of pieces $T _ { f }$ and the number of boundary $M _ { f }$

ProofofLemma 5.2. Our goal is to show that there exists $N = \Omega ( p \log T _ { f } )$ problem instances $x _ { 1 } , \ldots , x _ { N }$ and real-valued threshold $\tau _ { 1 } , \dots , \tau _ { N }$ such that $\mathcal { L }$ can shatter all the problem instances $x _ { i }$ and $\dot { \tau } _ { i } ( \mathrm { f o r } i = 1 , \dots , N )$ witnesses the shattering, that is

$$
| \{ ( \mathbb { I } ( \ell _ { \alpha } ( x _ { 1 } ) \geq \tau _ { 1 } ) , \dots , ( \mathbb { I } ( \ell _ { \alpha } ( x _ { N } ) \geq \tau _ { N } ) \mid \alpha \in \mathcal { A } \} | = 2 ^ { N } .
$$

Let $K = \lfloor \textstyle { \frac { T _ { f } + 1 } { 2 } } \rfloor , B = \lfloor \log _ { 2 } K \rfloor , A = [ 0 , K - 1 ] ^ { p }$ , and $\Theta = [ 0 , 1 ] ^ { d } \subset \mathbb { R } ^ { d }$ be a box-like region in $\mathbb { R } ^ { d } .$ . We construct $p B = \Omega ( p \log T _ { f } ^ { \mathbf { \tilde { \alpha } } } )$ problem instances $x _ { j , b } , \mathrm { f o r } j = 0 , \dotsc , - 1 - 1$ and $b = 0 , \ldots , B - 1$ by first defining the boundary polynomials for each problem instance: for each $x _ { j , b }$ , introduce $K - 1$ afine boundary polynomials:

$$
h _ { j , q } ( \alpha , \theta ) = \alpha _ { j } - \left( q + \frac 1 2 \right) , q = 0 , \ldots , K - 2 .
$$

In other words, each boundary polynomial $h _ { j , q } ( \alpha , \theta )$ is just an afine function that does not depend on the parameter θ. The boundaries divide the hyperparameter $\alpha _ { j }$ axis into K open intervals $( - \infty , \frac { 1 } { 2 } ) , ( \frac { 1 } { 2 } , \frac { 3 } { 2 } ) , \dots , ( K - \textstyle \frac { 5 } { 2 } , K - \textstyle \frac { 3 } { 2 } ) , ( K - \textstyle \frac { 3 } { 2 } , + \infty )$ and $K - 1$ boundary points $\alpha _ { j } = \textstyle { \frac { 1 } { 2 } } , \frac { 3 } { 2 } , \dotsc , K - \frac { 3 } { 2 }$

The task now is to construct the piece function for each problem instance $x _ { j , b }$ . For convenience, we define the open interval containing $r \in \{ 0 , \ldots , K - 1 \}$ as $I _ { r }$ , and the sign condition realized in the interval $I _ { r }$ as $\sigma _ { r }$ . Let $\mathsf { b i t } _ { b } ( r )$ denote the $b ^ { t h }$ -binary bit of r. On the interval $I _ { r }$ , define

$$
P _ { x _ { j , b } , \sigma _ { r } } ( \alpha , \theta ) = \frac { 1 } { 2 } \mathbf { b i t } _ { b } ( r ) ,
$$

which is simply a constant piece function. For any $\alpha _ { j }$ that lies in one of $K - 1$ boundary points, we simply assign the value 1 to the corresponding piece functions. By this construction, the piece and boundary functions are independent of ${ \bar { \theta } } ,$ so minimizing over Θ changes nothing.

Now, given an arbitrary sign pattern $y = ( y _ { j , b } ) _ { j , b } \in \{ 0 , 1 \} ^ { p B }$ , we simply choose $\alpha = ( \alpha _ { 0 } , \ldots , \alpha _ { p - 1 } )$ such that

$$
\alpha _ { j } = \sum _ { b = 0 } ^ { B - 1 } y _ { j , b } 2 ^ { b } .
$$

From the construction, we claim that the function $f _ { x _ { i , b } } ( \alpha , \theta )$ admits piecewise polynomial structure with complexity $\big ( \lfloor \frac { T _ { f } - 1 } { 2 } \rfloor , T _ { f } , 1 \big )$ ) as expected. To see this, we have $\begin{array} { r } { K - 1 = \lfloor \frac { T _ { f } - 1 } { 2 } \rfloor } \end{array}$ boundary polynomials ofthe forms $\begin{array} { r } { h _ { j , q } ( \alpha , \theta ) = \alpha _ { j } - \left( q + \frac { 1 } { 2 } \right) } \end{array}$ for $\bar { q } = 0 , \ldots , K - 2$ . Those boundaries induce $K$ open intervals and $\dot { K ^ { - 1 } }$ boundary points, for a total of $2 K { \bar { - } } 1 { \mathrm { ~ p i e c e s } } .$ Therefore, the number of piece functions is at most

$$
| \Sigma _ { x } | \leq 2 K - 1 = 2 \left\lfloor \frac { T _ { f } + 1 } { 2 } \right\rfloor - 1 \leq T _ { f } .
$$

Finally, notice that all the piece and boundary functions are linear or constant functions with respect to α and $\theta ,$ so $\Delta _ { f } = 1$ Moreover, we have the following claims:

$0 \le \alpha _ { j } \le 2 ^ { B } - 1 \le K - 1$ for any $j = 0 , \ldots , p - 1$ . This means that $\alpha \in \mathcal { A } = [ 0 , K - 1 ] ^ { p }$

• The open interval that contains α<sub>j</sub> is $I _ { \alpha _ { j } }$ , which assigns the sign condition $\sigma _ { \alpha _ { j } }$ , and therefore

• the value of $\begin{array} { r } { \ell _ { \alpha } ( x _ { j , b } ) = P _ { x _ { j , b } , \sigma _ { \alpha _ { j } } } ( \alpha , \theta ) = \frac { 1 } { 2 } \mathrm { b i t } _ { b } ( \alpha _ { j } ) = \frac { 1 } { 2 } y _ { j , b } . } \end{array}$

Finally, choose the real-valued thresholds $\begin{array} { r } { \tau _ { j , b } = \frac { 1 } { 4 } } \end{array}$ for all $j = 0 , \ldots , p - 1$ and $b = 0 , \ldots , B - 1$ , we have

$$
\mathbb { I } ( \ell _ { \alpha } ( x _ { j , b } ) \ge \tau _ { j , b } ) = y _ { j , b }
$$

Therefore, we proved that for any sign pattern $y \in \{ 0 , 1 \} ^ { p B }$ , there exists α such that

$$
\mathbb { I } ( \ell _ { \alpha } ( x _ { 1 } ) \ge \tau _ { 1 } ) , \dots , ( \mathbb { I } ( \ell _ { \alpha } ( x _ { N } ) \ge \tau _ { N } ) = y ,
$$

meaning that $\mathcal { L }$ can shatter the set of problem instances $x _ { j , b } ( \mathrm { f o r } j = 0 , \dots , p - 1$ and $b = 0 , \ldots , B - 1 )$ with $\begin{array} { r } { \tau _ { j , b } = \frac { 1 } { 4 } } \end{array}$ witness the shattering. Since $p B = \Omega ( p \log \bar { T _ { f } } )$ , we claim that Pdim $\mathsf { \Omega } _ { 1 } ( \mathcal { L } ) = \Omega ( p \log T _ { f } )$ □

Proof of Lemma 5.3. Again, our goal is to construct $\begin{array} { r } { N = \Omega \left( p d \log \left( 1 + \frac { M } { d } \right) \right) } \end{array}$ problem instances $x _ { 1 } , \ldots , x _ { N }$ and real-valued thresholds $\tau _ { 1 } , \ldots , \tau _ { N }$ such that

$$
| \{ ( \mathbb { I } ( \ell _ { \alpha } ( x _ { 1 } ) \geq \tau _ { 1 } ) , \dots , ( \mathbb { I } ( \ell _ { \alpha } ( x _ { N } ) \geq \tau _ { N } ) \mid \alpha \in \mathcal { A } \} | = 2 ^ { N } .
$$

Let $s \triangleq$ min $\begin{array} { r } { \{ M , d \} , a \triangleq \lfloor \frac { M } { s } \rfloor , \rho \triangleq M - a s \in \{ 0 , \dots , s - 1 \} } \end{array}$ . We then define

$$
m _ { i } \triangleq \left\{ { \begin{array} { l l } { a + 1 , } & { 1 \leq i \leq \rho , } \\ { a , } & { \rho < i \leq s , } \\ { 0 } & { s < i \leq d . } \end{array} } \right.
$$

Then $\begin{array} { r } { \sum _ { i = 1 } ^ { d } m _ { i } = M } \end{array}$ . We then define

$$
R _ { M , d } \triangleq \prod _ { i = 1 } ^ { d } ( m _ { i } + 1 ) , \quad S _ { m , d } \triangleq \prod _ { i = 1 } ^ { d } ( 2 m _ { i } + 1 ) .
$$

Roughly speaking, $R _ { M , d }$ is the number of sign regions in which none of the boundary polynomials vanishes, while $S _ { M , d }$ is the total number of realized sign patterns, including patterns with boundary equalities.

Step 1: Defining the shared boundary functions. For every coordinate $i = 1 , \ldots , d$ and every $q = 0 , \ldots , m _ { i } - 1$ , we define the boundary functions $h _ { i , q } ( \alpha , \theta )$ for all problem instances as

$$
h _ { i , q } ( \alpha , \theta ) \triangleq \theta _ { i } - \left( q + \frac { 1 } { 2 } \right) .
$$

By construction, there are exactl $\begin{array} { r } { \sum _ { i = 1 } ^ { d } m _ { i } = M _ { f } } \end{array}$ such boundary afine polynomials, and do not depend on α. We define the parameter box-like region as $\begin{array} { r } { \Theta \triangleq \prod _ { i = 1 } ^ { d } [ - 1 , m _ { i } + 1 ] } \end{array}$ . For a fixed coordinate i, the ordered thresholds ${ \begin{array} { c } { { \frac { 1 } { 2 } } , { \frac { 3 } { 2 } } , \dots , m _ { i } - { \frac { 1 } { 2 } } } \end{array} }$ can produce at most $m _ { i } + 1$ non-zero sign patterns, and $m _ { i }$ zero-containing sign patterns, one for each threshold equality. Therefore, the one-dimensional family $\{ h _ { i , q } \} _ { q = 0 } ^ { \bar { m } _ { i } - 1 }$ realizes exactly $2 m _ { i } + 1$ sign patterns on the $i ^ { t h }$ coordinate $\theta _ { i }$ of $\theta \in \Theta$

Note that the ordinates $\theta _ { i }$ are independent: any choice of one realizable coordinate-wise sign pattern is obtained by choosing the coordinate $\theta _ { i }$ separately. Therefore, the full boundary family $\{ h _ { i , q } ( \alpha , \theta ) \} _ { i , q }$ realize exactly $\begin{array} { r } { S _ { M , d } = \prod _ { i = 1 } ^ { d } ( 2 m _ { i } + 1 ) } \end{array}$ sign pattern when varying $\theta \in \Theta$

Moreover, among these patterns, there are exactly $m _ { i } + 1$ coordinate-wise choices that contain no zero for coordinate i. Therefore, the full product regions containing no boundary equality are $\begin{array} { r } { R _ { M , d } = \prod _ { i = 1 } ^ { d } ( m _ { i } + 1 ) } \end{array}$ . Each such region corresponds to a distinct integer vector $u = ( u _ { 1 } , \dotsc , u _ { d } )$ , where $u _ { i } \in \{ 0 , \ldots , m _ { i } \}$ . Thus, every one of these $R _ { M , d }$ regions is non-empty and belongs to Θ.

Step 2: Define the problem instances via the piecewise functions. For convenience, define $B \triangleq \lfloor \log _ { 2 } R _ { M , d } \rfloor$ and $K \triangleq 2 ^ { B }$ Because $K \le R _ { M , d }$ , we may select K distinct zero-free grid regions and enumerate the as $C _ { 0 } , \dots , C _ { K - 1 }$ . Let $\sigma _ { r }$ be the boundary sign pattern corresponding to $C _ { r } ,$ and let $\mathcal { A } = [ 0 , K \stackrel { \smile } { - } 1 ] ^ { p }$ be the box-like hyperparameter regions. We then define a finite problem instance sets $\mathcal { X } : = \{ x _ { j , b } : j \in \{ 1 , \dots , \bar { p } \} , b \in \{ 0 , \dots , B - 1 \} \}$ , and therefore $| c X | = p B$ . For $r \in \{ 0 , \ldots , K - 1 \}$ let $\mathbf { b i t } _ { b } ( r ) \in \{ 0 , 1 \}$ denote the $b ^ { t h }$ binary bit of r.

We formally construct the problem instance $x _ { j , b }$ as follows. On the selected sign pattern $\sigma _ { r } .$ , we define the piece function $P _ { x _ { j , b } , \sigma _ { r } } ( \alpha , \theta )$ as

$$
P _ { x _ { j , b } , \sigma _ { r } } ( \alpha , \theta ) \triangleq ( \alpha _ { j } - r ) ^ { 2 } + \frac { 1 } { 2 }  { \mathrm { b i t } } _ { b } ( r ) , \quad r = 0 , \ldots , K - 1 .
$$

For all other realized sign patterns $\sigma ,$ including all boundary patterns and all unselected zero-free patterns, we simply define their corresponding piece functions as

$$
P _ { x _ { j , b } , \sigma } ( \alpha , \theta ) \triangleq 2 .
$$

By construction, each problem instances $x _ { j , b }$ have the function $f _ { j , b }$ that admits piecewise polynomial structure with complexity $( M , S _ { M , d } , 2 )$

Step 3: Establish the pseudo-dimension lower-bound. We now show that the constructed problem instances $x _ { j , b }$ can be shattered by ${ \mathcal { L } } ,$ and there are appropriate real-valued thresholds $\tau _ { j , b }$ that witness the shattering.

Consider an arbitrary labeling $y = ( y _ { j , b } ) _ { j , b } \in \{ 0 , 1 \} ^ { p B }$ . For any $j = 1 , \dotsc , p ,$ we define

$$
r _ { j } \triangleq \sum _ { b = 0 } ^ { B - 1 } y _ { j , b } 2 ^ { b } \in \{ 0 , \ldots , 2 ^ { B } - 1 \} = \{ 0 , \ldots , K - 1 \} ,
$$

and choose the hyperparameter $\alpha = ( \alpha _ { 1 } , \ldots , \alpha _ { p } )$ such as $\alpha _ { j } \triangleq r _ { j }$ . Then $\alpha \in { \mathcal { A } } .$

Consider a problem instance $x _ { j , b }$ . Since $C _ { r _ { j } }$ is not empty, there exists $\theta \in C _ { r _ { i } } \cap \Theta$ , and on this region, we have

$$
P _ { x _ { j , b } , \sigma _ { r _ { j } } } ( \alpha , \theta ) = ( r _ { j } - r _ { j } ) ^ { 2 } + \frac { 1 } { 2 } \mathsf { b i t } _ { b } ( r _ { j } ) = \frac { 1 } { 2 } y _ { j , b } .
$$

For any other selected region where $s \neq r _ { j }$ , we have

$$
P _ { x _ { j , b } , \sigma _ { s } } ( \alpha , \theta ) = ( r _ { j } - s ) ^ { 2 } + \frac { 1 } { 2 } \mathtt { b i t } _ { b } ( s ) \ge 1 .
$$

The inequality comes from the fact that $r _ { j }$ , s are integer, meaning $( r _ { j } - s ) ^ { 2 } \geq 1$ if $r _ { j } \neq s$ . This implies that

$$
\ell _ { \boldsymbol { \alpha } } ( x _ { j , b } ) = \operatorname* { m i n } _ { \boldsymbol { \theta } \in \Theta } f ( x _ { j , b } , \boldsymbol { \alpha } , \boldsymbol { \theta } ) = \frac { 1 } { 2 } y _ { j , b } .
$$

Finally, choose $\begin{array} { r } { \tau _ { j , b } = \frac { 1 } { 4 } } \end{array}$ for all $j \in \{ 1 , \ldots , p \}$ and $b \in \{ 0 , \ldots , B - 1 \}$ , we have

$$
\mathbb { I } \left( \ell _ { \alpha } ( x _ { j , b } ) \ge \tau _ { j , b } \right) = y _ { j , b } .
$$

Therefore, L can shatter the set $\{ x _ { j , b } \} _ { j , b }$ of $p B$ problem instances with $\{ \tau _ { j , b } \} _ { j , b }$ witnesses the shattering. Therefore, we claim that Pdim $\begin{array} { r } { \left( \mathcal { L } \right) \geq p B = p \big \lfloor \log _ { 2 } \dot { R _ { M , d } } \big \rfloor . } \end{array}$

the task now is to show that log $\begin{array} { r } { R _ { M , d } = \Omega \left( d \log \left( 1 + \frac { M } { d } \right) \right) } \end{array}$ . We consider two cases:

• I $\mathrm { ~ f ~ } M \leq d ,$ then $R _ { M , d } = 2 ^ { M }$ . Since $\log ( 1 + t ) \leq t { \mathrm { f o r } } t \geq 0$ , we have $d \log ( 1 + { \frac { M } { d } } ) \leq M$ , whereas log $M _ { M , d } = M$ log 2.   
This means log $\begin{array} { r } { R _ { M , d } = \Omega \left( d \log \left( 1 + \frac { M } { d } \right) \right) } \end{array}$ holds true.

• If $M > d ,$ then $s = d ,$ and $a = \lfloor M / d \rfloor \geq 1$ . By definition, we have $R _ { M , d } \geq ( a + 1 ) ^ { d }$ . Let $x = M / d \geq 1$ , we claim that $a + 1 \geq { \sqrt { 1 + x } } .$ . To see that, ${ \mathrm { i f ~ } } 1 \leq x < 2 .$ , we have $a + 1 = 2 \geq { \sqrt { 1 + x } } . \operatorname { I f } x \geq 2$ , then $a + 1 > x \geq { \sqrt { 1 + x } }$ . Therefore

$$
\log R _ { M , d } \geq d \log ( a + 1 ) \geq { \frac { d } { 2 } } \log ( 1 + x ) = { \frac { d } { 2 } } \log \left( 1 + { \frac { M } { d } } \right) .
$$

This also implies log $\begin{array} { r } { R _ { M , d } = \Omega \left( d \log \left( 1 + \frac { M } { d } \right) \right) } \end{array}$ holds true.

Therefore, we claim that Pdim $\begin{array} { r } { \mathrm { \Lambda } _ { 1 } ( \mathcal { L } ) = \Omega \left( p d \log \left( 1 + \frac { M _ { f } } { d } \right) \right) } \end{array}$

## D Additional Results and Proofs for Section 6

In this section, we will formally present the proof for Theorem 6.1. Again, we then provide a discussion on the improvement of our proposed result compared to prior work by Le et al. (Le, Nguyen, and Nguyen 2026b).

Theorem 6.1 (restated). Under Assumption 1, let $\mathcal { L } _ { \nu a l } = \{ \ell _ { \alpha } ^ { \nu a l } : \mathcal { X } \to [ - H , H ] \mid \alpha \in \mathcal { A } \} , \Delta _ { f , g } = \operatorname* { m a x } \{ 1 , \Delta _ { f } , \Delta _ { g } \}$ , and $M _ { \mathrm { t o t } } = M _ { f } + M _ { g } + T _ { f } + d .$

$$
\mathrm { P d i m } ( \mathcal { L } _ { \nu a l } ) = \mathcal { O } \left( p \log ( T _ { f } T _ { g } ) + p d ^ { 2 } \log \left( 2 + M _ { \mathrm { t o t } } \Delta _ { f , g } \right) \right) .
$$

Proof. Fixed a problem instance $x \in \mathcal { X }$ and a real-valued threshold $\tau \in \mathbb { R }$ , our goal is to express $\mathbb { I } ( \ell _ { \alpha } ^ { \mathrm { v a l } } ( x ) \geq \tau )$ as a boolean combination of of polynomial FOL to which Theorem 4.2 applies. Recall that

$$
\mathcal { S } ( x , \alpha ) = \arg \operatorname* { m i n } _ { \theta \in \Theta } f _ { x } ( \alpha , \theta ) , \mathrm { ~ a n d ~ } \ell _ { \alpha } ^ { \mathrm { v a l } } = \operatorname* { i n f } _ { \theta \in \mathcal { S } ( x , \alpha ) } g _ { x } ( \alpha , \theta ) .
$$

We will proceed with the following steps.

Step 1: Piecewise polynomial representation. For the given fixed problem instances, let $\Sigma _ { x } ^ { f } \subset \{ - 1 , 0 , 1 \} ^ { M _ { f } }$ , and $\Sigma _ { x } ^ { g } \subset$ $\{ - \bar { 1 } , 0 , 1 \} ^ { M _ { g } }$ be the sets of of sign conditions indexing the training and validation pieces, respectively. By Assumption 1, we have $| \Sigma _ { x } ^ { f } | \leq T _ { f }$ and $| \Sigma _ { x } ^ { g } | \le T _ { g }$ . For every $\boldsymbol { \sigma } \in \Sigma _ { x } ^ { f }$ , let $\bar { F } _ { x , \sigma }$ be the corresponding training piece polynomial. For every $\gamma \in \Sigma _ { x } ^ { g }$ let $G _ { x , \gamma }$ be the corresponding validation piece polynomial. We then define the region predicates as

$$
\mathrm { W h e r e } _ { x , \sigma } ^ { f } ( \alpha , \theta ) = \bigwedge _ { m = 1 } ^ { M _ { f } } [ \mathrm { s i g n } ( h _ { x , m } ^ { \prime } ( \alpha , \theta ) = \sigma _ { m } ] ,
$$

and

$$
\mathrm { W h e r e } _ { x , \gamma } ^ { g } ( \alpha , \theta ) = \bigwedge _ { m = 1 } ^ { M _ { g } } [ \mathrm { s i g n } ( h _ { x , m } ^ { g } ( \alpha , \theta ) = \gamma _ { m } ] .
$$

Thus, whenever Where $\operatorname { \mathrm { ? } } _ { x , \sigma } ^ { f } ( \alpha , \theta )$ holds, we have

$$
f _ { x } ( \alpha , \theta ) = F _ { x , \sigma } ( \alpha , \theta ) ,
$$

and similarly, whenever Wher $\scriptstyle { \frac { f } { c _ { x , \sigma } } } ( \alpha , \theta )$ holds, we have

$$
g _ { x } ( \alpha , \theta ) = G _ { x , \gamma } ( \alpha , \theta ) .
$$

Moreover, we also define the parameter-feasibility checking condition as

$$
\mathrm { I n } _ { \Theta } ( u ) = \bigwedge _ { r = 1 } ^ { d } [ u _ { r } - \theta _ { \mathrm { m i n } } \geq 0 \land \theta _ { \mathrm { m a x } } - u _ { r } \geq 0 ] ,
$$

which is true exactly when $u \in \Theta$

Step 2: Certifying that a candidate is a training minimizer. Fixed $\sigma \in \Sigma _ { x } ^ { f }$ . For a candidate solution θ and a competing solution u, we define

$$
\mathrm { C o m p a r e } _ { x , \sigma } ( \alpha , \theta , u ) \triangleq \neg \mathrm { I n } _ { \Theta } ( u ) \vee \bigwedge _ { \rho \in \Sigma _ { x } ^ { f } } \left[ \neg \mathrm { W h e r e } _ { x , \rho } ^ { f } ( \alpha , u ) \vee F _ { x , \sigma } ( \alpha , \theta ) \leq F _ { x , \rho } ( \alpha , u ) \right] .
$$

Let’s elaborate the predicate above closely. First, suppose that $u \in \Theta$ . Note that there is exactly one sign conditions $\boldsymbol { \rho } \in \Sigma _ { x } ^ { f }$ is active at the point $( \bar { \alpha } , u ) \in \mathcal { A } \times \Theta$ . For that active $\rho ,$ the predicate requires

$$
F _ { x , \sigma } ( \alpha , \theta ) \leq F _ { x , \rho } ( \alpha , u ) = f _ { x } ( \alpha , u ) .
$$

For all other inactive $\rho ,$ the corresponding implication is automatically true. Therefore, provided in that θ belong to the training corresponding to $\sigma ,$ , we have $( \forall u \in \mathbb { R } ^ { d } ) \bar { \mathbf { C } } \mathrm { o m p a r e } _ { x , \sigma } ( \alpha , \theta , u )$ is equivalent to

$$
f _ { x } ( \alpha , \theta ) \leq f _ { x } ( \alpha , u ) { \mathrm { ~ f o r ~ a l l ~ } } u \in \Theta .
$$

Therefore

$$
\theta \in S ( x , \alpha ) \Leftrightarrow \mathrm { I n } _ { \Theta } ( \theta ) \land \mathrm { W h e r e } _ { x , \sigma } ^ { f } ( \alpha , \theta ) \land ( \forall u \in \mathbb { R } ^ { d } ) \mathrm { C o m p a r e } _ { x , \sigma } ( \alpha , \theta , u ) ,
$$

for the active training sign condition $\sigma .$

Step 3: Encoding a bad training minimizer. For each pair $( \sigma , \gamma ) \in \Sigma _ { x } ^ { f } \times \Sigma _ { x } ^ { g }$ , we define $\Phi _ { x , \tau , \sigma , \gamma } ( \alpha )$ as

$$
\begin{array} { r } { ( \exists \theta \in \mathbb { R } ^ { d } ) \left[ \mathrm { I n } _ { \Theta } ( \theta ) \wedge \mathrm { W h e r e } _ { x , \sigma } ^ { f } ( \alpha , \theta ) \wedge \mathrm { W h e r e } _ { x , \gamma } ^ { g } ( \alpha , \theta ) \wedge G _ { x , \gamma } ( \alpha , \theta ) < \tau \wedge ( \forall u \in \mathbb { R } ^ { d } ) \mathrm { C o m p a r e } _ { x , \sigma } ( \alpha , \theta , u ) \right] . } \end{array}
$$

Because the candidate conditions do not depend on $u ,$ the formula above can be equivalently be written in the form with the two ordered blocks $( \exists \theta \in \mathbb { R } ^ { d } ) ( \forall u \in \mathbb { R } ^ { d } )$ . This implies that every branch has $K = 2$ and $d _ { 1 } = d _ { 2 } = d .$ Step 4. Recovering the validation-loss threshold event. Since $s ( x , \alpha ) \neq \emptyset$ , we have $\begin{array} { r } { \operatorname* { i n f } _ { \theta \in S ( x , \alpha ) } g _ { x } ( \alpha , \theta ) < \tau } \end{array}$ if and only if there exists $\theta \in { \mathcal { S } } ( x , \alpha )$ such that $g _ { x } ( \alpha , \theta ) < \tau$ . Importantly, this does not require the infimum to be attained. If the infimum is strictly below $\tau ,$ the definition of the infimum guarantees the existence of an element with value below $\tau .$ It follows that

$$
\ell _ { \alpha } ^ { \mathrm { v a l } } ( x ) < \tau \Leftrightarrow \bigvee _ { \sigma \in \Sigma _ { g } ^ { f } , \gamma \in \Sigma _ { x } ^ { g } } \Phi _ { x , \tau , \sigma , \gamma } ( \alpha ) .
$$

This implies that the condition $\mathbb { I } ( \ell _ { \alpha } ^ { \mathrm { e v a l } } ( x ) \geq \tau )$ is a NOR-type Boolean combination of at most $L \leq T _ { f } T _ { g }$ polynomial branches, as required by Theorem 4.2.

Step 5. Counting the atomic polynomials and applying Theorem 4.2. Finally, note that each branch contains at most $\bar { M _ { \mathrm { t o t } } } = 2 M _ { f } + \bar { M } _ { g } + T _ { f } + 4 \bar { d } ^ { - } + \mathrm { i }$ distinct atomic polynomial predicate, and each atomic predicate has the degree at most $\Delta _ { f , g } = \operatorname* { m a x } \{ 1 , \Delta _ { f } ^ { - } , \Delta _ { g } \}$ . Substituting into Theorem 4.2 we have the final conclusion. □

Remark 5 (Comparison to prior work). Compared to (Le, Nguyen, and Nguyen 2026b), Theorem 6.1, which obtained the upper bound of

$$
\mathcal { O } ( p d ^ { 2 } \log ( M _ { f } + T _ { F } + M _ { g } + T _ { g } + d ) + p ^ { 2 } d ^ { 2 } \log \Delta _ { f , g } ) ,
$$

using standard quantifier elimination technique, our Theorem 6.1 improves both algebraic and combinatorial dependencies. In particular, our technique using nested block elimination (Theorem 4.2) reduces the degree-dependent term from $\dot { p } ^ { 2 } d ^ { 2 }$ log $\Delta _ { f , g }$ to $p d ^ { 2 } \log \Delta _ { f , g } .$ Moreover, by treating the $T _ { f } T _ { g }$ possible pairs of active training and validation pieces as separate logical branches, the dependence on $T _ { g }$ decreases from $p d ^ { 2 }$ log $T _ { g }$ to just p log $T _ { g } .$ . The training-piece parameter $T _ { f }$ remains in $M _ { t o t }$ because certifying lower-level optimality requires comparison against every possible training piece.

# E Additional results and Omitted proofs for Section 7

## E.1 Omitted proof for Section 7

In this section, we present the detailed proof for Theorem 7.1.

Theorem 7.1 (restated). Consider the class of validation loss functions $\mathcal { L } _ { \nu a l } = \{ \ell _ { \alpha } ^ { \nu a l } : \mathcal { X }  [ - H , H ] \mid \alpha \in \mathcal { A } \}$ induced by the Weighted Group Lasso objectives. Assuming that $\Theta = \mathbb { R } ^ { d }$ and ${ \mathcal { A } } \subset ( 0 , \infty ) ^ { \dot { p } }$ , then the pseudo-dimension is bounded by $\mathrm { P d i m } ( \bar { \mathcal { L } } _ { \nu a l } ) = \mathcal { O } ( \bar { p ( } d + p ) ^ { 2 } \log p )$ .

Proof. Assume under standard unconstrained formulation, that is $\Theta = \mathbb { R } ^ { d }$ and $\mathcal { A } \subset ( 0 , \infty ) ^ { p }$ . For $v = ( v _ { 1 } , \ldots , v _ { p } ) \in \mathbb { R } ^ { d }$ and $\nu \in \mathbb { R } ^ { p }$ , we define

$$
\operatorname { N o r m } ( v , \nu ) = \sum _ { i = 1 } ^ { p } \left[ \nu _ { i } \geq 0 \land \nu _ { i } ^ { 2 } = \sum _ { j = 1 } ^ { d _ { i } } v _ { i , j } ^ { 2 } \right] .
$$

In other words, the term Norm $( u , \nu )$ holds exactly when $\nu _ { i } = \| v _ { i } \| ^ { 2 }$ for every i. We then define the polynomial lifting of the training objective as

$$
\tilde { f } _ { x } ( \alpha , v , \nu ) = \| A \nu - b \| _ { 2 } ^ { 2 } + \sum _ { i = 1 } ^ { p } \alpha _ { i } \nu _ { i } .
$$

Given a problem instance x and a real-valued threshold $\tau \in \mathbb { R }$ , consider the polynomial FOL formula

$$
\Phi _ { x , \tau } ( \alpha ) \triangleq ( \forall \theta \in \mathbb { R } ^ { d } ) ( { \vec { \Omega } } ( z , \nu ^ { \theta } , \nu ^ { z } ) \in \mathbb { R } ^ { d + 2 p } ) \left[ g _ { x } ( \theta ) \geq \tau \vee \left( \operatorname { N o r m } ( \theta , \nu ^ { \theta } ) \wedge \operatorname { N o r m } ( z , \nu ^ { z } ) \wedge { \vec { J } } _ { x } ( \alpha , z , \nu ^ { z } ) < { \tilde { J } } _ { x } ( \alpha , \theta , \nu ^ { \theta } ) \right) \right] .
$$

In words, this formula states that every θ either has a validation loss at least τ, or admits another point z which strictly smaller training loss. Therefore

$$
\Phi _ { x , \tau } ( \alpha ) \Leftrightarrow \mathrm { e v e r y } \theta \in S ( x , \alpha ) \ \mathrm { s a t i s f i e s } \ g _ { x } ( \alpha ) \geq \tau .
$$

Indeed, a training minimizer cannot satisfy the second disjunct, while every non-minimizer admits a strictly better point because $s ( x , \alpha ) \neq \emptyset$ . Consequently,

$$
\Phi _ { x , \tau } ( \alpha ) \Leftrightarrow \operatorname* { i n f } _ { \theta \in S ( x , \alpha ) } g _ { x } ( \theta ) \geq \tau ,
$$

and therefore

$$
\begin{array} { r } { \mathbb { I } ( \ell _ { \alpha } ^ { \mathrm { v a l } } ( x ) \geq \tau ) = \Phi _ { x , \tau } ( \alpha ) . } \end{array}
$$

This one has one branch and two quantified blocks, with $L = 1 , K = 2 , d _ { 1 } = d ,$ and $d _ { 2 } = d + 2 p$ . Substituting to Theorem 4.2, we have the final conclusion. □

Remark 6 (Comparison with prior work). Le et al. (Le, Nguyen, and Nguyen 2026b) (Theorem 8.1) obtained the pseudodimension bound $( \mathcal { O } ( p ^ { 3 } d + p ^ { 2 } \dot { d } ^ { 2 } ) )$ for data-driven Weighted Group Lasso using standard quantifier elimination. Because the (p) groups are nonempty and (e.g., $\textstyle { \mathcal { O } } ( \sum _ { i = 1 } ^ { p } d _ { i } = d ) )$ , we necessarily have $( p \leq d ) ,$ , and their bound simplifies to $( e . g . , \mathcal { O } ( p ^ { 2 } d ^ { 2 } ) ) .$ In contrast, Theorem 7.1 gives $\mathcal { O } ( p ( d + p ) ^ { 2 } \log ( 2 p ) ) = \mathcal { O } ( p d ^ { 2 } \log ( 2 p ) ) \}$ ). Thus, our result replaces the quadratic dependence on the number ofgroups in the dominant term by an essentially linear dependence, improving the previous bound by afactor of order (that $\begin{array} { r } { i s , \ \frac { \stackrel { \smile } { p } } { \log ( 2 p ) } ) } \end{array}$ as p grows. Both analyses convert the group norms into polynomial form using auxiliary variables; the improvement therefore does not arise from a diferent representation of the Weighted Group Lasso objective. Rather, itfollows from the nested block-elimination argument in Theorem 4.2, which avoids the inflated algebraic dependence introduced by standard quantifier elimination. Forfixed $p ,$ both bounds retain the same $( d ^ { 2 } )$ dependence, whereas the improvement becomes increasingly significant when the number ofgroups grows.

## E.2 Other Applications

We note that our main contribution is the an novel general framework to establish statistical learning guarantees for a class of data-driven algorithms design problems which admits piecewise polynomials structure, or even more general, the structure that can be converted into the form of Theorem 4.2. To further showcase the applicability of our general frameworks, in this section we will provide other applications.

Data-driven Cost-sensitive SVM. Classification errors frequently have heterogeneous costs across classes or subpopulations, motivating cost-sensitive extensions of support vector machines. Rather than fixing these costs manually, we treat the groupspecific penalty vector as a data-driven algorithm configuration learned across problem instances. The hinge objective is piecewise polynomial: its polynomial representation changes whenever a training or validation example crosses its afine margin boundary. Consequently, cost-sensitive SVM provides a direct application of our connected-sign-cell framework, complementary to Weighted Group Lasso, which instead illustrates algebraic lifting.

Fix integers $n , m , p , d \geq 1 . \mathrm { ~ A ~ }$ problem instance $\bar { x ^ { \mathrm { ~ } } } = ( D _ { x } ^ { \mathrm { t r } } , D _ { x } ^ { \mathrm { v a l } } )$ contains a training set $D _ { x } ^ { \mathrm { t r } } = \{ ( a _ { x , j } , y _ { x , j } , c _ { x , j } ) \} _ { j = 1 } ^ { n }$ and a validation set $D _ { x } ^ { \mathrm { v a l } } = \{ ( a _ { x , k } ^ { \prime } , y _ { x , k } ^ { \prime } ) \} _ { k = 1 } ^ { m }$ . Here $a _ { x , j } , a _ { x , k } ^ { \prime } \in \mathbb { R } ^ { d }$ $y _ { x , j } , y _ { x , k } ^ { \prime } \in \{ - 1 , 1 \}$ , and $c _ { x , j } \in \{ 1 , \ldots , p \}$ identifies the cost group of training example j. Let $\mathcal { A } = [ \underline { { C } } , \overline { { C } } ] ^ { p } \subset ( 0 , \infty ) ^ { p }$ be the domain of group-specific penalty parameters, and let $\Theta = \ \mathbf { \bar { [ } } - \hat { B , } B \mathbf { ] } ^ { d }$ be the model-parameter domain. For a fixed regularization coeficient $\lambda > 0$ , define

$$
f _ { x } ( \alpha , w ) = \frac { \lambda } { 2 } \| w \| _ { 2 } ^ { 2 } + \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \alpha _ { c _ { x , j } } \left[ 1 - y _ { x , j } a _ { x , j } ^ { \top } w \right] _ { + } ,
$$

where $[ r ] _ { + } = \operatorname* { m a x } \{ 0 , r \}$ . We use the ordinary validation hinge loss

$$
g _ { x } ( w ) = \frac { 1 } { m } \sum _ { k = 1 } ^ { m } \left[ 1 - { y _ { x , k } ^ { \prime } } { a _ { x , k } ^ { \prime } } ^ { \top } w \right] _ { + } .
$$

The induced optimistic validation loss is $\begin{array} { r } { \ell _ { \alpha } ^ { \mathrm { s v m } } ( x ) = \operatorname* { i n f } _ { w \in S ( x , \alpha ) } g _ { x } ( w ) } \end{array}$ , where $\begin{array} { r } { S ( x , \alpha ) = \arg \operatorname* { m i n } _ { w \in \Theta } f _ { x } ( \alpha , w ) } \end{array}$ . Because Θ is compact and $f _ { x }$ is continuous, $S ( x , \alpha )$ is nonempty. Moreover, the quadratic regularizer makes $f _ { x } ( \alpha , \cdot )$ strongly convex, so the minimizer is unique. The general framework, however, does not require uniqueness.The box Θ can be chosen suficiently large that it does not modify the ordinary unconstrained SVM solution. Indeed, if $\bar { \mathbf { \omega } } _ { w } ^ { - }$ is the unconstrained minimizer, then

$$
\frac { \lambda } { 2 } \| w ^ { \star } \| _ { 2 } ^ { 2 } \leq f _ { x } ( \alpha , w ^ { \star } ) \leq f _ { x } ( \alpha , 0 ) \leq \overline { { C } } .
$$

Therefore $\| w ^ { \star } \| _ { 2 } \leq \sqrt { \frac { 2 \overline { { C } } } { \lambda } }$ , so it sufices to take $B \geq \sqrt { \frac { 2 \overline { { C } } } { \lambda } }$

For each training example, define the afine margin polynomial $h _ { x , j } ^ { f } ( w ) = 1 - y _ { x , j } a _ { x , j } ^ { \top } w , \quad j = 1 , \ldots , n$ . Let $T _ { \mathrm { t r } }$ be a uniform upper bound, over $x ,$ on the number of sign conditions of $\{ h _ { x , 1 } ^ { f } , \ldots , h _ { x , n } ^ { f } \}$ realized on Θ. Similarly, define $h _ { x , k } ^ { g } ( w ) =$ $1 - y _ { x , k } ^ { \prime } { a _ { x , k } ^ { \prime } } ^ { \top } w , \quad k = 1 , \ldots , m$ , and let $T _ { \mathrm { v a l } }$ uniformly bound the number of realized validation sign conditions. We then have the following result, which establishes generalization guarantee for the problem of data-driven cost-sensitive SVM.

Corollary E.1 (Data-driven cost-sensitive SVM). Let ${ \mathcal { L } } _ { s \nu m } = \{ \ell _ { \alpha } ^ { s \nu m } : { \mathcal { X } } \to \mathbb { R } \mid \alpha \in { \mathcal { A } } \}$ . We then have $P d i m ( \mathcal { L } _ { s \nu m } ) =$ $O \left( p ( n + m ) + p d ^ { 2 } \left( n + \log ( d + m + 2 ) \right) \right)$

Proof of Corollary E.1. Fix an instance x. The training objective has the n afine boundary polynomials $h _ { x , 1 } ^ { f } , \ldots , h _ { x , n } ^ { f }$ . For a realized sign condition $\sigma \in \{ - 1 , 0 , 1 \} ^ { n }$ , we define $I _ { + } ( \sigma ) = \{ j \in \{ 1 , \dots , n \} : \sigma _ { j } = 1 \}$ . On the realization of $\sigma ,$ the hinge terms indexed by $j \notin I _ { + } ( \sigma )$ vanish, while the remaining hinge terms equal their afine arguments. Therefore, the active training piece is

$$
F _ { x , \sigma } ( \alpha , w ) = \frac { \lambda } { 2 } \| w \| _ { 2 } ^ { 2 } + \frac { 1 } { n } \sum _ { j \in I _ { + } ( \sigma ) } \alpha _ { c _ { x , j } } \left( 1 - y _ { x , j } a _ { x , j } ^ { \top } w \right) .
$$

This is a polynomial of total degree at most two in $( \alpha , w )$ . In particular, the only mixed terms have the form $\alpha _ { c _ { x , i } } w _ { r }$ , which are bilinear. It follows that $f _ { x }$ has piecewise-polynomial complexity $( M _ { f } , T _ { f } , \dot { \Delta _ { f } } ) = ( n , T _ { \mathrm { t r } } , 2 )$ , where $T _ { \mathrm { t r } } \leq 3 ^ { \stackrel {  } { n } }$ . Similarly, fo the validation objective, we conclude that $g _ { x }$ has the piecewise polynomial complexity $( M _ { g } , T _ { g } , \Delta _ { g } ) = ( m , T _ { \mathrm { v a l } } , 1 )$ , where $T _ { \mathrm { v a l } } \leq 3 ^ { m }$ . Besides, we note that $\Delta _ { f , g } = \mathrm { m a x } \{ 1 , \Delta _ { f } , \Delta _ { g } \} = 2$ , and $\dot { M _ { \mathrm { t o t } } } = M _ { f } + \dot { M } _ { g } + \dot { T } _ { f } + \ddot { d } = \ddot { n } + \stackrel { \sim } { m } + \dot { T _ { \mathrm { t r } } } + d$ . Substituting to Theorem 6.1, we have the final conclusion. □

Multi-penalty ridge regression Multi-parameter ridge regularization uses several penalties to promote diferent structural properties of the fitted solution. Let

$$
f _ { x } ( \alpha , \theta ) = \| A \theta - b \| _ { 2 } ^ { 2 } + \sum _ { i = 1 } ^ { p } \alpha _ { i } \| D _ { i } \theta \| _ { 2 } ^ { 2 } ,
$$

and let $g _ { x } ( \alpha , \theta ) = \lVert A ^ { \prime } \theta - b ^ { \prime } \rVert _ { 2 } ^ { 2 }$ , where $\alpha \in \mathcal { A } = [ \alpha _ { \operatorname* { m i n } } , \alpha _ { \operatorname* { m a x } } ] ^ { p }$ and $\theta \in \Theta = [ \theta _ { \operatorname* { m i n } } , \theta _ { \operatorname* { m a x } } ] ^ { d }$ are box like hyperparameters and parameters domain. Let $\mathcal { L } _ { \mathrm { r i d g e } }$ denote the corresponding optimistic validation-loss class. We then have the following result, which establishes the pseudo-dimension upper bound for the problem of tuning regularization parameters in multi-penalty ridge regression.

Corollary E.2 (Pseudo-dimension for ridge regression). We have Pdim $( { \mathcal { L } } _ { \mathrm { r i d g e } } ) = { \mathcal { O } } ( p d ^ { 2 } \log d ) )$ .

ProofofCorollary E.2. It is clearly that the complexity of the training and validation objectives are $( M _ { f } , T _ { f } , \delta _ { f } ) = ( 0 , 1 , 3 )$ and $\left( { \hat { M } } _ { g } , T _ { g } , \Delta _ { g } \right) = \left( 0 , 1 , 2 \right)$ ), respectively. Therefore applying Theorem 6.1 gives us the final conclusion. □