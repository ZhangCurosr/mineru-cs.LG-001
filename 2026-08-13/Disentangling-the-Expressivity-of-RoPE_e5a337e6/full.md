# Disentangling the Expressivity of RoPE

Selim Jerad<sup>1</sup> Anej Svete<sup>2</sup> Jiaoda Li<sup>2</sup> Ryan Cotterell<sup>2</sup>

Toyota Technological Institute at Chicago<sup>1</sup>

ETH Zürich<sup>2</sup>

sjerad@ttic.edu

{anej.svete, jiaoda.li, ryan.cotterell}@ethz.ch

## Abstract

Two accounts recur in explanations of the success of rotary position embeddings (RoPE). Expressivity studies associate periodic position information with modular predicates, whereas mechanistic and long-context studies emphasize positional anchors and local offsets. We formalize both accounts for fully uniform, finite-precision soft-attention transformers. We find that, if every rotary component is periodic, RoPE transformers recognize exactly the languages definable in past temporal logic with modular predicates. Conventional RoPE is different: The rotations it computes never repeat. This yields a precision-dependent bounded simulation of fixed-offset lookback operators, rather than an all-length modular characterization. Controlled experiments match this separation: Constructed periodic schedules length-generalize on modular languages, while conventional RoPE behaves more like a bounded locality bias and can impair tasks requiring positioninvariant access to distant context. Altogether, our findings shed light on RoPE transformers, bringing theoretical expressivity characterizations closer to models used in practice.

## 1 Introduction

In the absence of positional information, self-attention—the core operation of transformers (Vaswani et al., 2017)—is permutation-equivariant. Positional encodings (PEs) help distinguish symbol positions. The original transformer employs sinusoidal PEs (SiPE), which add position-dependent sine and cosine features to symbol representations. However, models using SiPE often generalize poorly to sequences longer than those seen during training (Rosendahl et al., 2019; Neishi & Yoshinaga, 2019; Press et al., 2022). Motivated by this limitation, RoPE was proposed as an alternative that encodes relative positions within attention: It rotates query and key vectors as a function of position, making their dot product depend on relative displacement (Su et al., 2023). RoPE has since become the de facto PE scheme in transformer-based LMs (Bai et al., 2023; Grattafiori et al., 2024; Team et al., 2024; Olmo et al., 2025; Yang et al., 2025a; DeepSeek-AI et al., 2026, inter alia).

Despite its widespread use, the theoretical contributions of RoPE remain unclear. Existing explanations fall broadly into two themes. Theoretical expressivity work links the periodicity of sinusoidal functions to modular predicates in formal logic (Yang et al., 2024; 2025b). More practically oriented analyses find that the apparent length generalization ability of RoPE can be superficial: RoPE may extrapolate to longer sequences without enabling effective use of distant context (Kazemnejad et al., 2023; Men et al., 2024; Gelberg et al., 2025; Du et al., 2026). A complementary, more mechanistic line of work instead emphasizes its role as a positional anchor and its bias toward fixed local offsets (Barbero et al., 2025a;b).

These perspectives prima facie appear incompatible. We reconcile them within the fully uniform fixed-precision transformer framework. We show that component-periodic RoPE $( \mathrm { R o P E _ { P } } )$ is equivalent to past temporal logic with modular predicates, LTL[P, MOD]. Conventional RoPE, in contrast, realizes a non-periodic schedule. The two variants differ only in the angles they rotate by. RoPE<sub>P</sub> rotates the key and query subspaces by rational multiples of $2 \pi ,$ so each rotary component cycles through finitely many phases and repeats with a fixed period, which a fixed-precision model can realize by a lookup table indexed by the position modulo that period. Conventional RoPE instead rotates by frequencies whose phases are incommensurate with 2π and therefore never repeat (§2.4). Its relative scores can simulate the look-back of k steps, only while a precision-dependent score gap is maintained. This connects RoPE to local attention (Li & Cotterell, 2026) and gives a formal account of fixed-offset heads observed in practice (Barbero et al., 2025a). Fig. 1 summarizes the landscape of fully uniform finite-precision transformers.

![](images/7f449c1e8fec6c3e5315e15a9f573dac4fbb47a4b8ccd13864cf6ef3e28f94e7.jpg)  
Figure 1: The expressivity landscape of fully uniform finite-precision transformers. The red box shows our contributions: The connection of $\mathrm { R o P E } _ { \mathrm { P } } ^ { \mathrm { ^ { . } } }$ and $\mathrm { S i P E _ { P } }$ transformers to LTL[P, MOD], which is incomparable with LTL[S] (capturing rightmost hard-attention NoPE transformers). Their intersection contains $\begin{array} { r } { \dot { \bf L T L } [ \dot { \bf P } ] . } \end{array}$ , corresponding to soft-attention NoPE transformers.

We vet our results experimentally by training transformers on controlled formal languages. We find that constructed periodic RoPE schedules length-generalize on languages requiring modular predicates. In contrast, no conventional base value generalizes on $( { \check { a } } a ) ^ { * } { \mathrm { ~ o r ~ } } ( a b ) ^ { * } ,$ suggesting that changing the base does not recover the modular construction. Across the same sweep, the mean longest-perfect length of conventional RoPE outperforms NoPE at six of seven bases on an $\mathbf { L T L } [ \breve { \mathbf { Y } } ]$ language, but underperforms at every base on both an LTL[P] language and a language requiring both P and Y. These results support the locality account while suggesting that the resulting local bias can interfere with long-distance conditioning.

## 2 Preliminaries

An alphabet Σ is a finite, non-empty set of symbols and $\Sigma ^ { * }$ denotes the set of all strings over Σ. With | · |, we denote the length of a string—in particular, we write $| w _ { 1 } \cdot \cdot \cdot w _ { N } | = \breve { N }$ We write <sub>ε</sub> for the empty string, the unique string of length 0. A formal language L over Σ is a subset of $\Sigma ^ { * }$

## 2.1 Linear Temporal Logic

Regular languages are languages that can be described by regular expressions; see $\ S \mathrm { A } . 3$ for details. Subclasses of regular languages can equivalently be described with different formalisms such as automata and logic. Past work (Yang et al., 2024; Li & Cotterell, 2025; Jerad et al., 2025) links finite-precision transformers with subregular classes of languages. We use linear temporal logic (LTL) as our primary descriptive formalism.

Given $\boldsymbol { w } = \boldsymbol { w } _ { 1 } \cdot \cdot \cdot \boldsymbol { w } _ { N }$ , LTL formulas are evaluated at a position i; a formula accepts w when it holds at the readout position $N + 1$ . The atomic predicate $\pi _ { a } ( i )$ tests whether the symbol $w _ { i }$ at position i matches a. Furthermore, LTL leverages temporal operators to evaluate context-dependent properties at positions. The operators we need are the following:

$$
w , i \left| = \mathbf { P } \psi \iff \exists j < i : w , j \right| = \psi , \qquad w , i \left| = \mathbf { Y } \psi \iff i > 1 \land w , i - 1 \right| = \psi .\tag{1}
$$

For $k \geq 1$ , we write $\mathbf { Y } ^ { k } \boldsymbol { \psi }$ for k-fold composition of $\mathbf { Y } ;$ equivalently, $\boldsymbol { w } , i \left| = \mathbf { Y } ^ { k } \boldsymbol { \psi } \right.$ if and only if $i > k$ and $w , i - k \Vdash \psi$ . We write LTL[P] and $\mathbf { L T L } [ \mathbf { P } , \dot { \mathbf { Y } } ]$ for the fragments generated from these operators, atomic predicates, and Boolean connectives. The detailed LTL semantics and equivalent first-order, automata, and algebraic formalisms appear in $\ S \mathrm { A }$ . A third operator, $\mathbf { S } \ ( ^ { \prime \prime } \mathrm { s i n c e ^ { \prime \prime } } )$ , subsumes both P and $\breve { \mathbf { Y } }$ (Gabbay et al., 1980), so LTL[S] is all of LTL—exactly the star-free languages (Kamp, 1968; McNaughton & Papert, 1971). We use first-order logic over the linear order $<$ interchangeably with LTL, via $\mathbf { L T L } [ \mathbf { S } ] = \mathbf { F O } [ < ]$ (Kamp, 1968) and $\mathbf { L T L } [ \mathbf { P } ] = \mathbf { P F O } ^ { 2 } [ < ]$ , the past fragment of the two-variable logic $\mathbf { F O } ^ { 2 } [ < ]$ (Li & Cotterell, 2025); §A.2 gives the formal definitions.

Modular predicates. The unary predicate $\mathsf { M O D } _ { m } ^ { r } \left( i \right)$ holds exactly when $i \equiv r { \pmod { m } }$ . We write LTL[P, MOD] for LTL[P] augmented with all such predicates. Adding modular predicates preserves the $\mathbf { L T L - F O [ < ] }$ correspondence: $\mathbf { L T L } [ \bar { \mathbf { P } } , \mathsf { M O D } ] = \mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ (Thm. 3.3).

## 2.2 Transformers

A consistent takeaway of theoretical work is that transformers’ computational abilities heavily depend on the modeling assumptions one makes (Hao et al., 2022; Jerad et al., 2025; Svete et al., 2026). Understanding the minimal viable definition of a transformer is particularly useful, as it facilitates the analysis of the fundamental components of every transformer. To this end, we adopt the fully uniform fixed-precision model of Li & Cotterell (2025). Uniformity requires all strings be processed by a model defined by a single set of parameters, in contrast to having a separate model for each length.

Finite precision. Transformer parameters and activations belong to a finite set $\mathbb { F } \ \stackrel { \mathrm { d e f } } { = }$ $\{ f _ { 1 } , \ldots , { \bar { f } } _ { K } \} ,$ , corresponding to real-world transformers $( \mathrm { e . g . , 3 2 }$ -bit floating-point arithmetic). All computations are performed rounded to $\mathbb { F } ,$ with ∞ and $- \infty$ in F which describe overflow. We denote by $f _ { \mathrm { l a r g e } }$ the smallest positive element of $\mathbb { F }$ such that exp $\left( - f _ { \mathrm { l a r g e } } \right)$ rounds to 0. This permits exact hard selection whenever the winning score exceeds every alternative by at least f<sub>large</sub>.

Transformers as string classifiers. We study transformers as string classifiers. An input to a transformer is a string $\boldsymbol { w } \in \Sigma ^ { * }$ augmented with the beginning-of-string BOS and endof-string EOS symbols, BOSwEOS. We set ${ \overline { { \Sigma } } } { \stackrel { \mathrm { d e f } } { = } } \Sigma \cup$ {BOS, EOS}. An initial embedding layer maps symbols in $\overline { { \Sigma } }$ to distinct elements of $\mathbb { F } ^ { D } .$ , where $D \in \mathbb { N }$ is the hidden state size. The embedding layer is followed by a stack of L transformer layers, each composed of an attention layer, a feedforward network, and layer-normalization. For a layer ℓ and position $i , x _ { i } ^ { \ell } \in \mathbb { F } ^ { D }$ is the contextual representation of symbol $w _ { i }$ after being processed by layer ℓ. We denote by $\lambda \colon ( \overline { { \Sigma } } ) ^ { * } \to ( \mathbf { \mathbb { F } } ^ { D } \bar { ) ^ { * } }$ the function that maps an input BOSwEOS to contextual representations through the embedding layer and the $\dot { L }$ transformer layers. For a string w and string position $i ,$ we denote by $\bar { \lambda ( \boldsymbol { w } ) _ { i } } \overset { \cdot } { \in } \mathbb { F } ^ { D }$ the final i’th contextual representation $x _ { i } ^ { L }$ To enable language recognition, λ is paired with an output classification layer $\mathcal { F } \colon (  { \mathbb { F } } ^ { D } ) \ \stackrel { \cdot } { \to } \qquad $ {0, 1} acting on $\lambda ( w ) _ { \mathrm { E O S } }$ for an input w. Given a precision regime $\mathbb { F } ,$ , the tuple ${ \mathsf { T } } { \overset { \underset { \mathrm { d e f } } { } } { = } } ( \lambda , { \mathcal { F } } )$ is an F-transformer and defines a language $\mathbb { L } ( \mathbb { T } ) \stackrel { \mathrm { d e f } } { = } \{ { \boldsymbol w } | { \mathcal F } ( \lambda ( { \boldsymbol w } ) _ { \mathrm { E O S } } ) = 1 \}$

Soft attention. We assume soft attention, which corresponds to standard implementations. We note a minute but important practical detail: For numerical stability, soft attention is usually computed by subtracting from $s _ { i , j } -$ the attention score between the query $\pmb q _ { i }$ and the key $k _ { j }$ —the maximum score. We define the normalized score $\alpha _ { i , j }$ as follows:

$$
\pmb { \alpha } _ { i , j } = \frac { \mathrm { e x p } \left( S _ { i , j } - \operatorname* { m a x } _ { k < i } S _ { i , k } \right) } { \sum _ { l = 0 } ^ { i - 1 } \mathrm { e x p } \left( S _ { i , l } - \operatorname* { m a x } _ { k < i } S _ { i , k } \right) } .\tag{2}
$$

These NoPE transformers (SMAT) recognize exactly $\mathbf { L T L } [ \mathbf { P } ] = \mathbf { P F O } ^ { 2 } [ < ]$ (Li & Cotterell,2025).

## 2.3 Rotary Position Embeddings

RoPE (Su et al., 2023) partitions the query’s and key’s D-dimensional space into twodimensional subspaces and rotates each by an angle that depends on the absolute position i. Concretely, let

$$
\begin{array} { r } { R _ { \Theta , i } \stackrel { \mathrm { d e f } } { = } \mathrm { d i a g } \left( \mathbf { R o t } _ { \theta _ { 1 } , i } , \ldots , \mathbf { R o t } _ { \theta _ { D / 2 } , i } \right) , \qquad \mathbf { R o t } _ { \theta , i } \stackrel { \mathrm { d e f } } { = } \left( \begin{array} { l l } { \cos \left( \theta i \right) } & { - \sin \left( \theta i \right) } \\ { \sin \left( \theta i \right) } & { \cos \left( \theta i \right) } \end{array} \right) } \end{array}\tag{3}
$$

where $R _ { \Theta , i }$ is a block-diagonal matrix and $\Theta \in [ 0 , 2 \pi ] ^ { D / 2 }$ is a tuple of angles assigned to the different two-by-two rotation matrices in $R _ { \Theta , i } ,$ which we call the schedule. When the schedule is implicit, we drop it from the subscript and write, $\mathrm { e . g . } , R _ { i } ;$ we restore it whenever several schedules are in play. In conventional implementations, the angular frequencies are parameterized as $\dot { \theta _ { d } } \dot { = } \dot { \beta } ^ { - 2 ( d - 1 ) / D } .$ , where $\beta > 0$ is called the base; the original RoPE construction uses $\beta = 1 0 ^ { 4 }$ (Su et al., 2023).

At position $i ,$ queries and keys become

$$
q _ { i } = R _ { \Theta , i } \left( Q x _ { i } + b _ { q } \right) , \qquad k _ { i } = R _ { \Theta , i } \left( K x _ { i } + b _ { k } \right) .\tag{4}
$$

For exact rotations and zero biases, orthogonality gives the relative-position identity

$$
\begin{array} { r } { k _ { j } ^ { \intercal } q _ { i } = ( R _ { \Theta , j } K x _ { j } ) ^ { \intercal } R _ { \Theta , i } Q x _ { i } = x _ { j } ^ { \intercal } K ^ { \intercal } ( R _ { \Theta , j } ) ^ { \intercal } R _ { \Theta , i } Q x _ { i } , = x _ { j } ^ { \intercal } K ^ { \intercal } R _ { \Theta , i - j } Q x _ { i } , } \end{array}\tag{5}
$$

making the score depend on the distance $i - j$

Ideal and realized rotations. The rotation above is ideal: It is evaluated in exact arithmetic, and it stores real numbers. A fixed-precision model computes something else. It stores a schedule $\widehat { \Theta }$ and evaluates the rotation with the finite-precision arithmetic. We write $\widehat { R } _ { \widehat { \Theta } , i } \in$ $\mathbb { F } ^ { D \times D }$ for the realized matrix in finite-precision arithmetic at position $i ,$ and $\widehat { R } _ { \widehat { \Theta } , i } ^ { ( d ) } \in \mathbb { F } ^ { 2 \times 2 }$ for its $d ^ { \mathrm { t h } }$ block. The hat marks the realized map as opposed to the ideal one $i \mapsto R _ { \Theta , i }$ . The distinction matters because rounding need not preserve the structure of the ideal map; our periodicity definitions concern the realized map.

Custom schedules and unrotated subspaces. To establish lower bounds for RoPE transformers, we permit a custom schedule—a bespoke set of angles Θ chosen for the target language. Custom schedules are standard in expressivity literature (e.g., Chiang et al., 2023; Yang et al., 2024; 2025b). We also allow unrotated subspaces with ${ \theta _ { d } } ^ { \sim } = 0 ;$ these subsume partial-RoPE variants (Barbero et al., 2025b; Yang et al., 2025c; Khan et al., 2026) and retain the LTL[P] computation available to NoPE, while separate rotary dimensions supply modular or fixed-offset information.

## 2.4 Component-Periodic and Non-Periodic RoPE

Intuitively, PEs with periodicity behave similarly to modular positional predicates in logic (Chiang et al., 2023; Yang & Chiang, 2024; Yang et al., 2025b). We formalize this intuition for finite-precision RoPE, with the proofs in §C.1. Throughout, ${ \widehat { R } } _ { i }$ and $\widehat { R } _ { i } ^ { ( d ) }$ are the realized matrix and its $d ^ { \mathrm { t h } }$ block.

Definition 2.1 (Component-periodic and non-periodic RoPE). A realized RoPE schedule $\widehat { \Theta }$ is component-periodic if, for every $d ,$ there is an integer $m _ { d } \geq 1$ such that

$$
\widehat { R } _ { i + m _ { d } } ^ { ( d ) } = \widehat { R } _ { i } ^ { ( d ) } \qquad f o r a l l i \in \mathbb { N } .\tag{6}
$$

It is non-periodic if this condition fails: Some component has no positive integer period.

Because D is fixed, component-periodicity is equivalent to periodicity of the full map $i \mapsto { \widehat { R } } _ { i }$ In particular, M <sup>def</sup> = lcm $\left( m _ { 1 } , \ldots , m _ { D / 2 } \right)$ satisfies $\widehat { R } _ { i + M } = \widehat { R } _ { i }$ for every i, with unrotated blocks having period 1. We denote by SMAT[RoP $\mathrm { \Delta E _ { P } ] }$ and SMAT[RoPE ] the class of languages recognized by fixed-precision component-periodic and non-periodic RoPE transformers, respectively.

We show that periodic schedules can be expressed via finitely many modular predicates:

Proposition 2.1 (Component-periodicity via MOD). Ifa realized RoPE schedule $\widehat { \Theta }$ is componentperiodic, then its image is finite and, for every matrix coordinate and realized value, the positions attaining that value are definable by unary modular predicates.

A standard class of periodic schedules is the set of rational multiples of π (Yang et al., 2024).

Proposition 2.2 (Rational-angle functions are periodic). $I f \theta _ { d } / \left( 2 \pi \right) = a _ { d } / m _ { d } \in \mathbb { Q } ,$ , then $i \mapsto \mathbf { R o t } _ { \theta _ { d } , i }$ has period $m _ { d }$

Intuitively, adding $m _ { d }$ to i changes the phase by $2 \pi a _ { d }$

Prop. 2.2 concerns the ideal map, and it may not transfer to finite precision. The position i grows without bound as the string does, so neither i nor the phase $\theta _ { d } i$ stays representable in F at all lengths. An implementation that forms the phase and then rounds therefore computes a map that need not agree with $\mathbf { R o t } _ { \theta _ { d } , i } ,$ , and that realized map need not repeat. Rational angles alone thus do not make a schedule component-periodic in the sense of Def. 2.1. We resort to realizing the same rational-angle function without forming the product $\theta _ { d } i .$ For $\theta _ { d } = 2 \pi a _ { d } / m _ { d } ,$ we precompute the finite table

$$
T _ { d } \left( r \right) \stackrel { \mathrm { d e f } } { = } \mathrm { r o u n d } _ { \mathbb { F } } \left( { \bf R o t } _ { \theta _ { d } , r } \right) , \qquad 0 \leq r < m _ { d } ,\tag{7}
$$

and define the realized block directly by $\widehat { \pmb { R } } _ { i } ^ { ( d ) } = T _ { d } \left( i \mathrm { m o d } m _ { d } \right)$ . The table has $m _ { d }$ entries and the counter has $m _ { d }$ states, and both depend on the schedule but not on the input length, so the construction stays inside the fully uniform fixed-precision model. The realized map then satisfies $\widehat { R } _ { i + m _ { d } } ^ { ( d ) } = \widehat { R } _ { i } ^ { ( d ) }$ for every i by construction. Similar cyclic lookups appear in practice. Huo (2026) precomputes one period of P-RoPE and indexes it by position modulo the period. Wang et al. (2024) round each RoPE wavelength to the nearest integer, so that every component repeats after a whole number of positions. Both were introduced to improve length generalization, and both are component-periodic in the sense of Def. 2.1, so the characterization we prove next applies to them directly.

## 3 A Characterization of SMAT[RoPE<sub>P</sub>]

This section describes the equivalence of SMAT[RoPE ] with LTL $[ \mathbf { P } , \mathsf { M O D } ] = \mathbf { P } \mathbf { F O } ^ { 2 } [ < , \mathsf { M O D } ]$ The next two subsections sketch the intuition for the lower and upper bounds; full proofs are in $\ S C$

Theorem 3.1. S ${ \bf M A T } [ \mathrm { R o P E } _ { \mathrm { P } } ] = { \bf L T L } [ { \bf P } , { \sf M O D } ] = { \bf P F O } ^ { 2 } [ < , { \sf M O D } ] .$

## 3.1 From SMAT[RoPE ] to $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$

Let M be the common period to all the rotation matrices in the block diagonal matrix (§2.4). Then, every realized entry depends only on i mod M, so equality to any realized value is a finite disjunction of predicates $\mathsf { M O D } _ { M } ^ { r } \left( i \right)$ (Prop. 2.1). The fixed-precision simulation of all remaining transformer operations by $\mathbf { P F O } ^ { 2 } [ < ]$ (Li & Cotterell, 2025) then gives $\mathbf { S M A T } [ \mathrm { R o P E } _ { \mathrm { P } } ] \subseteq \mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$

Lemma 3.1. Any F-transformer with RoPE<sub>P</sub> can be simulated by $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ . Hence, $\mathbf { S M A T } [ \mathrm { R o P E } _ { \mathrm { P } } ] \subseteq \mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ] = \mathbf { L T L } [ \mathbf { P } , \mathsf { M O D } ]$

RoPE as a unary predicate. Although RoPE scores encode pairwise distances, it does not require binary primitives; unary ones suffice. To see why, consider the binary modular predicate

$$
\mathsf { M O D } _ { m } ^ { r } ( i , j ) = \top \iff ( j - i ) \equiv r \quad ( \mathrm { m o d } m ) ,\tag{8}
$$

which models the relative position dependency of RoPE by passing the distance between two positions through a periodic function. We show in $\tilde { \ S C . 2 }$ that the binary modular predicate can be expressed by a Boolean combination of unary modular predicates.

Proposition 3.1. Binary modular predicates can be expressed with unary modular predicates and Boolean logic.

## 3.2 From LTL[P, MOD] to SMAT[RoPE<sub>P</sub>]

For the reverse inclusion, F-transformers with $\mathrm { R o P E _ { P } }$ can first implement Boolean connectives and P in unrotated dimensions via structural induction (Li & Cotterell, 2025). For each modular atom ${ \mathsf { M O D } } _ { m } ^ { r } \left( i \right)$ , a period-m rotary pair uses the lookup table $T _ { m } \left( s \right) = \mathbf { R o t } _ { 2 \pi / m , s }$ for $0 \leq s < m$ and returns $\mathbf { \hat { \Pi } } _ { T _ { m } }$ (i mod m). An attention head compares the returned phase with the fixed BOS phase. A constant non-BOS baseline separates the matching residue from every nonmatching residue, and stable softmax turns the comparison into an exact Boolean value. Thus $\mathbf { L T L } [ \mathbf { \tilde { P } } , \mathsf { M O D } ] \subseteq \mathbf { S M A T } [ \mathbf { R o P E } _ { \mathrm { P } } ]$

Lemma 3.2. For every LTL[P, MOD] formula, there is an F-transformer with $\mathrm { R o P E _ { P } }$ that simulates theformula. Hence $\mathbf { L } \mathbf { \check { T } L } [ \mathbf { P } , \mathbf { M } 0 \mathbf { \check { \mathrm { D } } } ] \subseteq \mathbf { \check { S } M A T } [ \mathbf { R o P E } _ { \mathrm { P } } ]$

Connection to attention sinks. The use of BOS as a fixed anchor connects the tightness construction to attention sinks, where language models allocate disproportionate attention to initial tokens (Xiao et al., 2024). One account of why sinks are useful is that they prevent contextual representations from over-mixing and keep them sufficiently distinct (Barbero et al., 2025b). Our construction offers a complementary reason: The anchor is what makes unary modular predicates computable, because the query’s phase must be compared against a key whose own phase does not move. This suggests that sinks in RoPE transformers serve in part to expose positional phase, and we note that Gemma 7B, which uses RoPE, does concentrate attention on BOS (Barbero et al., 2025a).

## 3.3 Periodic Sinusoidal Positional Encodings

We find absolute sinusoidal PEs (Vaswani et al., 2017) to be equivalent to $\mathrm { R o P E } _ { \mathrm { P } }$ . As for $\mathrm { R o P E } _ { \mathrm { P } } ,$ , we denote by $\mathrm { S i P E _ { P } }$ the class of periodic sinusoidal encodings such that the realized sinusoidal position-to-embedding map has a finite integer period as in Def. 2.1. We similarly define $\operatorname { S i P E } _ { \mathrm { N P } }$

Theorem 3.2. SMA $\Gamma [ \mathrm { S i P E } _ { \mathrm { P } } ] = \mathbf { L } \mathbf { T } \mathbf { L } \big [ \mathbf { P } , \mathsf { M O D } \big ] = \mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ] .$

The proof (§C.4) again expresses the finite periodic image with modular predicates and uses custom periodic features to recover each modular atom. This shows that the key factor for the corresponding all-length logical class is the periodicity of the encoding; both the absolute and relative encodings are unified under modular predicates.

## 3.4 Characterizing LTL[P, MOD]

We have established that $\mathbf { S M A T } [ \mathrm { R o P E } _ { \mathrm { P } } ]$ and $\mathbf { S M A T } [ \mathrm { S i P E _ { P } } ]$ can both be exactly characterized by LTL[P, MOD]. To better grasp the expressive power of ${ \mathrm { R o P E } } _ { \mathrm { P } }$ and $\mathrm { S i P E _ { P } } ,$ , we provide additional characterizations of LTL[P, MOD]. Building on Dartois & Paperman (2013); Li & Cotterell (2025), we prove the following equivalences<sup>1</sup> in §B.

Theorem 3.3. Let $\mathbb { L } \subseteq \Sigma ^ { * }$ be a regular language, M be its syntactic monoid, and A be the minimal DFA accepting it. The following assertions are equivalent:

(i) L is a left-deterministic modular polynomial

(ii) $\mathbb { M } \in \mathbf { \Theta } \dot { \mathbf { Q R } }$

(iii) $\exists k \in \mathbb { N }$ such that $\boldsymbol { \mathcal { A } } _ { k }$ is partially ordered

(iv) $\mathbb { L } = \mathbb { L } ( \phi )$ for some $\mathbf { P F O } ^ { 2 } [ < ,$ MOD] formula ϕ

(v) $\mathbb { L } = \mathbb { L } ( \dot { \psi } )$ for some LTL[P, MOD] formula ψ

<table><tr><td>Language</td><td>Defining property</td><td>Class</td><td>Account</td></tr><tr><td> $( a a ) ^ { * }$ </td><td>even length</td><td>LTL[P, MOD]; not star-free</td><td>modular</td></tr><tr><td> $( a b ) ^ { * }$ </td><td>ab repeated</td><td>LTL[P, MOD]; not LTL[P]</td><td>modular</td></tr><tr><td> ${ \dot { a } } \Sigma ^ { * }$ </td><td>starts with a</td><td>LTL[P]</td><td>locality</td></tr><tr><td> $\Sigma ^ { * } a$ </td><td>ends with a</td><td> $\mathbf { L T L } [ \mathbf { Y } ] ; \mathrm { n o t } \ \mathbf { L T L } [ \mathbf { P } ]$ </td><td>locality</td></tr><tr><td> $\Sigma ^ { * } a b \Sigma ^ { * }$ </td><td>contains ab</td><td> $\mathbf { L T L } [ \mathbf { { P } } , \mathbf { { Y } } ] ;$  locally testable</td><td>locality</td></tr></table>

Table 1: The formal languages we train on, each over an alphabet Σ with $| \Sigma | > 1$ . The last column records which account the language probes: the modular account of §3 or the locality account of §4. §E.1 gives the full definitions.

The characterization provides novel tools for (in)expressibility statements about $\mathbf { L T L } [ \mathbf { P } , \mathsf { M O D } ]$ We can first show via $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ that LTL[P, MOD] can leverage modular predicates to recognize languages with a periodic behavior where a substring is repeated indefinitely.

Proposition 3.2. For every nonempty word $w _ { 1 } \cdots w _ { m } ,$ the language $\left( w _ { 1 } \cdot \cdot \cdot w _ { m } \right) ^ { * }$ belongs to $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$

Proof. It is defined by MO $\mathsf { \Lambda } _ { m } ^ { 0 } \wedge \forall i \wedge _ { 0 \leq j < m } \left( \mathsf { M } 0 \mathsf { D } _ { m } ^ { j } \left( i \right) \Rightarrow \pi _ { w _ { j } } \left( i \right) \right)$ , which is in $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$

Importantly, Prop. 3.2 implies that languages such as (ab)<sup>∗</sup> and $( a a ) ^ { * }$ belong to $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ leading to the following conclusions.

LTL[P, MOD] and LTL[S] are incomparable. In other words, neither class contains the other. In one direction, $( a \dot { a } ) ^ { * }$ is not star-free (Yang et al., 2024) and so lies outside LTL[S], yet Prop. 3.2 places it in LTL[P, MOD]. In the other direction, LTL[S] can recognize the locally testable languages (Zalcstein, 1972)—those that require checking that a finite sequence of symbols occur at consecutive positions. Written in ${ \bf L } { \hat { \bf T } } { \bf L } ,$ this needs the Y operator: $\sum ^ { \bullet } a b \Sigma ^ { * }$ is defined by $\mathbf { P } \left( \pi _ { b } \land \mathbf { Y } \pi _ { a } \right)$ . The only temporal operator of LTL[P, MOD] is P, which cannot step to an adjacent position, and we show in §B.1 that $\mathbf { L T L } [ \mathbf { P } , \mathsf { M O D } ]$ contains no such language.

LTL[P, MOD] cannot perform modular counting. Modular predicates count positions, not symbols. Adding them to $\mathbf { L T L } [ \mathbf { P } ]$ therefore does not let it decide whether the number of occurrences of a symbol is divisible by an integer. The canonical example is PARITY, the set of bitstrings with an even number of 1s. We prove in §B.1 that PARITY is not $\mathbf { L T L } [ \mathbf { P } , \mathsf { M O D } ]$

## 3.5 Testing the Modular Account

We train transformer classifiers on the languages in Tab. 1, which also records the account each language probes. Training strings have length at most 40; test strings have lengths 41–500. Every configuration is run over 5 seeds and 3 learning rates, and every conventionalschedule configuration is fully rotated. Besides accuracy, we report the longest perfect length $N ^ { * } \colon$ the largest N for which accuracy is 100% at every tested length through N. §E gives the architecture and optimization settings and reports per-run means and standard deviations. Tab. 2 compares NoPE with the period-targeting variants, while Fig. 2 reports the conventional non-periodic controls. In both variants, RoPE is applied on all dimensions.

<table><tr><td></td><td></td><td colspan="2">NoPE</td><td colspan="2"> $\mathrm { R o P E _ { P } }$ </td><td colspan="2"> $\mathrm { S i P E _ { P } }$ </td></tr><tr><td>Language</td><td>∈ Class</td><td>Acc.</td><td> $N ^ { * }$ </td><td> $_ \mathrm { A c c . }$ </td><td> $N ^ { * }$ </td><td>Acc.</td><td> $N ^ { * }$ </td></tr><tr><td> $( a a ) ^ { * }$ </td><td>LTL[P, MOD]</td><td>0.50</td><td>41</td><td>1.00</td><td>500</td><td>0.50</td><td>41</td></tr><tr><td> $( a b ) ^ { * }$ </td><td> $\mathbf { L T L } \big [ \mathbf { P } , \mathsf { M O D } \big ]$ </td><td>0.96</td><td>96</td><td>1.00</td><td>500</td><td>0.50</td><td>44</td></tr></table>

Table 2: Periodic-construction experiments. Maximum accuracy and longest perfect length over the run grid achieved by different transformer variants.

![](images/480acad7401c9cc006a30dfeeae175df320653ae4d528481e3d0ef54a3d14a64.jpg)  
Figure 2: Mean longest perfect length $N ^ { * }$ for conventional non-periodic encodings on the modular languages. Markers are means over the run grid and whiskers are ±1 standard deviation. The two reference lines are best-run values for NoPE and $\mathrm { R o P E _ { P } }$ on these tasks. Neither non-periodic encoding generalizes perfectly at any tested base.

$\mathrm { R o P E _ { P } }$ generalizes perfectly on $\mathbf { L T L } [ \mathbf { P } , \mathsf { M O D } ]$ . The period-targeting $\mathrm { R o P E _ { P } }$ model achieves accuracy 1.00 and $\dot { N } ^ { * } = 5 \mathrm { 0 0 }$ on both $( a b ) ^ { * }$ and $( a a ) ^ { * }$ , matching the modular-predicate construction.

$\mathrm { S i P E _ { P } }$ struggles on LTL[P, MOD]. Although $\mathrm { S i P E _ { P } }$ has the same existential expressivity, the trained model does not learn either modular language. The theorem guarantees a parameterization, not that optimization will find it; one possible explanation is that absolute sinusoidal features in the residual stream are more easily overwritten.

Conventional rotations do not learn the constructed modular behavior. Neither $\mathrm { R o P E _ { N P } }$ nor SiPE generalizes perfectly on the two periodic languages. The full sweep in Fig. 2 reaches the same conclusion at every tested base across twenty-four orders of magnitude, separating the engineered periodic behavior from conventional frequencies.

## 4 A Practical Account of Non-Periodic RoPE

We now analyze RoPE’s role implementing fixed-offset heads (Barbero et al., 2025a)— referencing positions at a specific displacement from the query position. This account applies naturally to conventional ${ \mathrm { R o P E } } _ { \mathrm { N P . } }$ , but does not capture its asymptotic behavior. Rather, it relies on a finite-precision range over which RoPE can create a reliable fixed-offset head, partly explaining its empirical advantage over SiPE (Su et al., 2023). The proof for the bounded non-periodic RoPE simulations is in §D.

## 4.1 Conventional RoPE Is Non-Periodic

We first pinpoint why conventional RoPE is non-periodic. Namely, irrational rotations, those that conventional RoPE leverages, imply non-periodicity.

Proposition 4.1 (Specialization of Walters 1982, Thm 1.8). $f g / ( 2 \pi ) \not \in \mathbb { Q } ,$ then $i \mapsto$ round<sub>F</sub> $\left( \mathbf { R o t } _ { g , i } \right)$ is non-periodic. Hence, conventional RoPE, which contains $g = 1$ , is non-periodic under output-rounded ideal-phase evaluation.

## 4.2 Fixed-Offset Attention up to a Length Bound

We now proceed to show that with irrational rotations, $\mathrm { R o P E } _ { \mathrm { N P } }$ transformers can implement fixed-offset attention heads.

Fix an offset $k \geq 1$ . With exact arithmetic, a frequency $g / \left( 2 \pi \right) \not \in \mathbb { Q }$ yields a k-offset head whose score is

$$
S _ { i , j } ^ { g , C , k } = C \cos \left( g \left( i - k - j \right) \right) , \qquad 0 \leq j < i ,\tag{9}
$$

for some $C > 0$ (Barbero et al., 2025a). At positions $i \geq k + 1$ , the target $j = i - k$ uniquely maximizes Eq. (9) with score C. For the uniqueness to hold under finite precision, the target must remain ahead by at least $f _ { \mathrm { l a r g e } }$ after rounding in the score computation.

We define the k-offset head’s certified range as the maximum length at which it scores its target position at least $f _ { \mathrm { l a r g e } }$ more than any non-target position:

$$
\begin{array} { r } { N _ { \operatorname* { m a x } } ^ { ( k ) } \left( g , C , \mathbb { F } \right) \overset { \mathrm { d e f } } { = } \operatorname* { s u p } \left. L \in \mathbb { N } : \widehat { S } _ { i , i - k } ^ { g , C , k , \mathbb { F } } - \widehat { S } _ { i , j } ^ { g , C , k , \mathbb { F } } \geq f _ { \mathrm { l a r g e } } \overset { \mathrm { f o r } \mathrm { a l l } k + 1 \leq i \leq L + 1 , } { 0 \leq j < i , j \neq i - k } \right. . } \end{array}\tag{10}
$$

Under exact evaluation of Eq. (9) through length $L \geq k ,$ the sufficient gap is $C \delta _ { L } \left( g \right)$ , where $\delta _ { L } \left( g \right) \stackrel { \mathrm { d e f } } { = } 1 - \mathrm { m a x } _ { 1 \leq n \leq L }$ cos (gn). Thus $C \delta _ { L } \left( g \right) \geq f _ { \mathrm { l a r g e } }$ certifies $N _ { \mathrm { m a x } } ^ { ( k ) } \left( g , C , \mathbb { F } \right) \ge L$

## 4.3 From Fixed-Offset Heads to $\mathbf { Y } ^ { k }$

If the certified range is $L ,$ stable softmax makes position $i - k ^ { \prime } s$ weight exactly 1 for every query position $i \in \left\{ k + 1 , \ldots L + 1 \right\}$ . Unrotated dimensions independently simulate LTL[P] and provide the boundary test for the first k positions.

Corollary 4.1. Fix $g , C ,$ and F, and let $L \in \mathbb { N }$ . Write SMAT[RoPE ]-transformers for softattention $\mathrm { \bar { R o P E } _ { N P } }$ transformers with unrotated subspaces, and let L-simulation be as in Def. D.1.

(i) For every offset $k \geq 1$ with $k \le L \le N _ { \operatorname* { m a x } } ^ { ( k ) } \left( g , C , \mathbb { F } \right)$ , such a transformer L-simulates the fixed-offset operator $\boldsymbol { r } ^ { k }$

(ii) If $1 \le L \le N _ { \operatorname* { m a x } } ^ { ( 1 ) } \left( g , C , \mathbb { F } \right)$ , it L-simulates every LTL[P, Y]-definable formula. More generally, for a finite set of offsets $K \subseteq \mathbb { N } _ { > 1 }$ with max $K \le L \le \operatorname* { m i n } _ { k \in K } N _ { \operatorname* { m a x } } ^ { ( k ) } \left( g , C , \mathbb { F } \right)$ , it L-simulates every formula built from atomic predicates, Boolean connectives, P, and the operators $\{ Y ^ { k } : k \in K \}$

(iii) Moreover, for every $L \in \mathbb { N }$ and every finite offset set K there exist a rotation angle g with $g / \left( 2 \pi \right) \bar { \notin } \mathbb { Q } ,$ , scales C, and a finite-precision regime F satisfying the hypothesis of (ii).

The connection to $\mathbf { L T L } [ \mathbf { P } , \mathbf { Y } ]$ links the result to the extra power of local attention (Li & Cotterell, 2026) and resembles the finite-precision implementation of n-gram heads (Svete & Cotterell, 2024; Svete et al., 2024).

## 4.4 Testing the Locality Account

We evaluate whether the bounded fixed-offset mechanism appears in length generalization, under the protocol of §3.5. RoPE is again applied on all dimensions.

![](images/ec7eba2405b3bba3f67f1edfba045ea4388420a9f071e98daa601332df75af88.jpg)

![](images/75d532d73be762f4ad7d5f130f96bd3e1581fa5e06400e89be83450a99d3fd0a.jpg)

![](images/da46921eed1e70be63f7a2ed0d84b01fdb6ce23e2d7a9bbb8657437de9c241cd.jpg)  
Figure 3: Mean longest perfect length $N ^ { * }$ . Whiskers are ±1 standard deviation over the run grid and the grey band is the corresponding NoPE interval.

The standard base is not uniformly best. On each single-operator language, the conventional base $\beta = 1 0 ^ { 4 }$ gives the lowest mean $N ^ { * }$ among the seven tested bases: 45.5 on $\Sigma ^ { * } a$ and 146.9 on $a \Sigma ^ { * }$ . Every tested base on either side of $1 0 ^ { 4 }$ improves on these two values. This mirrors the extrapolation experiments of Liu et al. (2024), who find $1 0 ^ { 4 }$ to be the worst tested base after fine-tuning and report improvements from both smaller and larger bases. The pattern is not universal, as, on the combined P–Y language, $1 0 ^ { 1 } 2$ performs worse. This suggests that the optimal base is task-dependent.

RoPE improves on a Y language. On $\Sigma ^ { * } a ,$ RoPE ’s mean $N ^ { * }$ exceeds NoPE’s at six of seven bases (Fig. 3). The certified-range result offers a mechanism: Relative scores can isolate the preceding position while their finite-precision gap remains large enough. As length grows, more competing offsets appear, increasing the chance of a numerical collision and making the behavior brittle.

RoPE can disrupt P reasoning. On aΣ<sup>∗</sup>, the NoPE mean is 470.5, whereas the RoPE means range from 146.9 to 402.9. On $\Sigma ^ { * } a b \Sigma ^ { * }$ , the NoPE mean is 92.9, whereas the RoPE<sub>NP</sub> means range from 46.1 to 61.9. This suggests a tradeoff: Relative displacement can help retrieve a nearby symbol while disrupting the nearly position-invariant aggregation that supports conditioning on distant information. Conventional RoPE therefore behaves more like a bounded source of Y access than a monotonic expressivity improvement over NoPE. We conjecture that partial-RoPE (Barbero et al., 2025a; Khan et al., 2026) and NoPE-RoPE hybrid attention (Yang et al., 2025c) variants should empirically improve on long-context P reasoning while retaining the locality bias of $\mathrm { R o P E _ { N P } }$

## 5 Discussion

Two regimes. For component-periodic schedules, RoPE yields precisely LTL[P, MOD] over arbitrary input lengths, enabling modular reasoning over input positions. The conventional RoPE<sub>NP</sub> behaves qualitatively differently: Relative scores can directly identify any fixed offset $k ,$ but only while finite precision preserves a sufficient score gap, yielding bounded $\mathbf { Y } ^ { k }$ operators and a bounded simulation of $\mathbf { L T L } [ \mathbf { P } , \mathbf { Y } ]$ . The two accounts therefore differ both in their schedule assumption and in whether the claim is uniform over all lengths. With unrotated subspaces, RoPE also retains the P reasoning of NoPE—checking whether a symbol occurs anywhere in the context.

Limitations. Our periodic lower bound uses a repeated finite table and a tailored exact value set, and does not imply that standard frequencies or optimization discover modular behavior. Empirical results, however, suggest that strong modular behavior often appears. The locality result of §4 is bounded and is not an all-length characterization of RoPE .

Architectural implications. The comparison suggests reserving unrotated dimensions when a model must combine global P reasoning with periodic or local relative-position information. It also supports treating rotation structure as an explicit architectural choice, consistent with work questioning broad long-context claims for RoPE (Du et al., 2026). Neither account gives RoPE general state-tracking or regular language recognition abilities (Liu et al., 2023b), such as those required by Flip-Flop language modeling (Liu et al., 2023a). Alternative state-tracking mechanisms include rational transductors (Mohri, 2026) and transformer–linear-RNN hybrids (Merrill et al., 2026b;a). A remaining theoretical question is how the known depth hierarchy for RoPE transformers (Yang et al., 2025b) differs between exact periodic and bounded-local regimes.

## Acknowledgments

We are grateful to William Merrill and Andy Yang for valuable comments on this work. Selim Jerad thanks Makoto Yamada and the MLDS Unit at the Okinawa Institute of Science and Technology for hosting him while writing this paper. Anej Svete and Jiaoda Li are

supported by an ETH AI Center Doctoral Fellowship. We used generative AI to improve our writing. Every modification introduced by generative AI was carefully reviewed by the authors, who take full responsibility for it.

## References

Jorge Almeida. Pseudovarieties of semigroups. Asian-European Journal of Mathematics, May 2025. ISSN 1793-7183. doi: 10.1142/s1793557125400078. URL http://dx.doi.org/10. 1142/S1793557125400078.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report, 2023. URL https://arxiv.org/abs/2309.16609.

Federico Barbero, Alex Vitvitskyi, Christos Perivolaropoulos, Razvan Pascanu, and Petar Veliˇckovi´c. Round and round we go! what makes rotary positional encodings useful?, 2025a. URL https://arxiv.org/abs/2410.06205.

Federico Barbero, Álvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Michael Bronstein, Petar Veliˇckovi´c, and Razvan Pascanu. Why do llms attend to the first token?, 2025b. URL https://arxiv.org/abs/2504.02732.

Janusz Antoni Brzozowski and Faith Ellen. Languages of R-trivial monoids. Journal of Computer and System Sciences, 20(1):32–49, 1980. ISSN 0022-0000. doi: https://doi.org/ 10.1016/0022-0000(80)90003-3. URL https://www.sciencedirect.com/science/article/ pii/0022000080900033.

David Chiang, Peter Cholak, and Anand Pillay. Tighter bounds on the expressivity of transformer encoders, 2023. URL https://arxiv.org/abs/2301.10743.

Luc Dartois and Charles Paperman. Two-variable first order logic with modular predicates over words. In Natacha Portier and Thomas Wilke (eds.), 30th International Symposium on Theoretical Aspects of Computer Science (STACS 2013), volume 20 of Leibniz International Proceedings in Informatics (LIPIcs), pp. 329–340, Dagstuhl, Germany, 2013. Schloss Dagstuhl – Leibniz-Zentrum für Informatik. ISBN 978-3-939897-50-7. doi: 10.4230/LIPIcs.STACS. 2013.329. URL https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.STACS. 2013.329.

Luc Dartois and Charles Paperman. Adding modular predicates to first-order fragments, 2015. URL https://arxiv.org/abs/1401.6576.

DeepSeek-AI, Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chengyu Hou, Chenhao Xu, Chenze Shao, Chong Ruan, Conner Sun, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Donghao Li, Dongjie Ji, Erhang Li, Fang Wei, Fangyun Lin, Fangzhou Yuan, Feiyu Xia, Fucong Dai, Guangbo Hao, Guanting Chen, Guoai Cao, Guolai Meng, Guowei Li, Han Yu, Han Zhang, Hanwei Xu, Hao Li, Haofen Liang, Haoling Zhang, Haoming Luo, Haoran Wei, Haotian Yuan, Haowei Zhang, Haowen Luo, Haoyu Chen, Haozhe Ji, Hengqing Zhang, Honghui Ding, Hongxuan Tang, Huanqi Cao, Huazuo Gao, Hui Qu, Hui Zeng, J Yang, JQ Zhu, Jia Luo, Jia Song, Jia Yu, Jialiang Huang, Jialu Cai, Jian Liang, Jiangting Zhou, Jiasheng Ye, Jiashi Li, Jiaxin Xu, Jiewen Hu, Jieyu Yang, Jin Chen, Jin Yan, Jingchang Chen, Jingli Zhou, Jingting Xiang, Jingyang Yuan, Jingyuan Cheng, Jingzi Zhou, Jinhua Zhu, Jiping Yu, Joseph Sun, Jun Ran, Junguang Jiang, Junjie Qiu, Junlong Li, Junmin Zheng, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Kexing Zhou, Kezhao Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Wang, Leyi Xia, Li Zhang, Liang Zhao, Lihua Guo, Lingxiao Luo, Linwang

Ma, Linyan Zhu, Litong Wang, Liyu Cai, Liyue Zhang, Longhao Chen, MS Di, MY Xu, Max Mei, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Mingxu Zhou, Minmin Han, Ning Wang, Panpan Huang, Panpan Wang, Peixin Cong, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qingyang Li, Qinyu Chen, Qiushi Du, Qiwei Jiang, Rui Tian, Ruifan Xu, Ruijie Lu, Ruiling Xu, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runqian Chen, Runqiu Yin, Runxin Xu, Ruomeng Shen, Ruoyu Zhang, Ruyi Chen, SH Liu, Shanghao Lu, Shangmian Sun, Shangyan Zhou, Shanhuang Chen, Shaofei Cai, Shaoheng Nie, Shaoqing Wu, Shaoyuan Chen, Shengding Hu, Shengyu Liu, Shiqiang Hu, Shirong Ma, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, Shuying Yu, Songyang Zhou, Tao Ni, Tao Yun, Tian Jin, Tian Pei, Tian Ye, Tianle Lin, Tianran Ji, Tianyi Cui, Tianyuan Yue, Tingting Yu, Tun Wang, W Zhang, WL Xiao, Wangding Zeng, Wei An, Weilin Zhao, Wen Liu, Wenfeng Liang, Wenjie Pang, Wenjing Luo, Wenjing Yao, Wenjun Gao, Wenkai Yang, Wenlve Huang, Wenqing Hou, Wentao Zhang, Wenting Ma, Xi Gao, Xiang He, Xiangwen Wang, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaokang Zhang, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingchen Liu, Xingkai Yu, Xingyou Li, Xinyu Yang, Xinyu Zhang, Xu Chen, Xuanyu Wang, Xuecheng Su, Xueyin Chen, Xuheng Lin, Xuwei Fu, YC Yan, YQ Wang, YW Ma, Yanfeng Luo, Yang Zhang, Yanhong Xu, Yanru Ma, Yanwen Huang, Yao Li, Yao Li, Yao Xu, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Qian, Yi Shao, Yi Yu, Yichao Zhang, Yifan Ding, Yifan Shi, Yijia Wu, Yiliang Xiong, Yiling Ma, Ying He, Ying Tang, Ying Zhou, Yingjia Luo, Yinmin Zhong, Yishi Piao, Yisong Wang, Yixiang Zhang, Yixiao Chen, Yixuan Tan, Yixuan Wei, Yiyang Ma, Yiyuan Liu, Yonglun Yang, Yongqiang Guo, Yongtong Wu, Yu Wu, YuKun Li, Yuan Cheng, Yuan Ou, Yuanfan Xu, Yuanhao Li, Yuduan Wang, Yuehan Yang, Yuer Xu, Yuhan Wu, Yuhao Meng, Yuheng Zou, Yukun Zha, Yunfan Xiong, Yupeng Chen, Yuping Lin, Yuqian Cao, Yuqian Wang, Yushun Zhang, Yuting Yan, Yutong Lin, Yuxian Gu, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuxuan Zhou, Yuyang Zhou, Yuzhen Huang, ZF Wu, Zehao Wang, Zehua Zhao, Zehui Ren, Zekai Zhang, Zhangli Sha, Zhe Fu, Zhe Ju, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zheren Gao, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhixian Huang, Zhixuan Chen, Zhiyu Wu, Zhizhou Ren, Zhongyu Wu, Zhuoshu Li, Zhuping Zhang, Zian Xu, Zihao Wang, Zihua Qu, Zihui Gu, Zijia Zhu, Zilin Li, Zipeng Zhang, Ziwei Xie, Ziyi Gao, Ziyi Wan, Zizheng Pan, and Zongqing Yao. Deepseek-v4: Towards highly efficient million-token context intelligence, 2026. URL https://arxiv.org/abs/2606.19348.

Grégoire Delétang, Anian Ruoss, Jordi Grau-Moya, Tim Genewein, Li Kevin Wenliang, Elliot Catt, Chris Cundy, Marcus Hutter, Shane Legg, Joel Veness, and Pedro A. Ortega. Neural networks and the chomsky hierarchy, 2023. URL https://arxiv.org/abs/2207.02098.

Yufeng Du, Phillip Harris, Minyang Tian, Eliu A Huerta, Srikanth Ronanki, Subendhu Rongali, Aram Galstyan, and Hao Peng. Rope distinguishes neither positions nor tokens in long contexts, provably, 2026. URL https://arxiv.org/abs/2605.15514.

Dov Gabbay, Amir Pnueli, Saharon Shelah, and Jonathan Stavi. On the temporal analysis of fairness. In Proceedings of the 7th ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages, POPL ’80, pp. 163–173, New York, NY, USA, 1980. Association for Computing Machinery. ISBN 0897910117. doi: 10.1145/567446.567462. URL https: //doi.org/10.1145/567446.567462.

Yoav Gelberg, Koshi Eguchi, Takuya Akiba, and Edoardo Cetin. Extending the context of pretrained llms by dropping their positional embeddings, 2025. URL https://arxiv.org/ abs/2512.12167.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bob bie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer,

Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Ra parthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Syd ney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vítor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Chang han Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul,

Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

Yiding Hao, Dana Angluin, and Robert Frank. Formal language recognition by hard attention transformers: Perspectives from circuit complexity. Transactions of the Association for Computational Linguistics, 10:800–810, 2022. doi: 10.1162/tacl\_a\_00490. URL https: //aclanthology.org/2022.tacl-1.46/.

Simin Huo. Periodic rope for infinite context llms, 2026. URL https://arxiv.org/abs/2605. 27980.

Selim Jerad, Anej Svete, Jiaoda Li, and Ryan Cotterell. Unique hard attention: A tale of two sides. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 2: Short Papers), pp. 977–996, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-252-7. doi: 10.18653/v1/2025.acl-short.76. URL https://aclanthology.org/2025.acl-short.76/.

Hans Kamp. Tense Logic and the Theory of Linear Order. PhD thesis, UCLA, 1968. URL https://www.proquest.com/openview/408039eb4ed228dc4cba3fe7e1774163/ 1?pq-origsite=gscholar&cbl=18750&diss=y.

Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, and Siva Reddy. The impact of positional encoding on length generalization in transformers, 2023. URL https://arxiv.org/abs/2305.19466.

Mohammad Aflah Khan, Krishna P. Gummadi, Manish Gupta, and Abhilasha Ravichander. Fractional rotation, full potential? investigating performance and convergence of partial rope, 2026. URL https://arxiv.org/abs/2603.11611.

Jiaoda Li and Ryan Cotterell. Characterizing the expressivity of fixed-precision transformer language models, 2025. URL https://arxiv.org/abs/2505.23623.

Jiaoda Li and Ryan Cotterell. Characterizing the expressivity of local attention in transformers, 2026. URL https://arxiv.org/abs/2605.00768.

Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Exposing attention glitches with flip-flop language modeling, 2023a. URL https://arxiv.org/abs/ 2306.00946.

Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Transformers learn shortcuts to automata. In The Eleventh International Conference on Learning Representations, 2023b. URL https://openreview.net/forum?id=De4FYqjFueZ.

Xiaoran Liu, Hang Yan, Shuo Zhang, Chenxin An, Xipeng Qiu, and Dahua Lin. Scaling laws of rope-based extrapolation, 2024. URL https://arxiv.org/abs/2310.05209.

Robert McNaughton and Seymour Papert. Counter-Free Automata. M.I.T. Press research monographs. M.I.T. Press, 1971. ISBN 9780262130769. URL https://mitpress.mit.edu/ 9780262130769/counter-free-automata/.

Xin Men, Mingyu Xu, Bingning Wang, Qingyu Zhang, Hongyu Lin, Xianpei Han, and Weipeng Chen. Base of rope bounds context length, 2024. URL https://arxiv.org/abs/ 2405.14591.

William Merrill, Hongjian Jiang, Yanhong Li, Anthony Lin, and Ashish Sabharwal. Why are linear rnns more parallelizable?, 2026a. URL https://arxiv.org/abs/2603.03612.

William Merrill, Yanhong Li, Tyler Romero, Anej Svete, Caia Costello, Pradeep Dasigi, Dirk Groeneveld, David Heineman, Bailey Kuehl, Nathan Lambert, Chuan Li, Kyle Lo, Saumya Malik, DJ Matusz, Benjamin Minixhofer, Jacob Morrison, Luca Soldaini, Finbarr Timbers, Pete Walsh, Noah A. Smith, Hannaneh Hajishirzi, and Ashish Sabharwal. Olmo hybrid: From theory to practice and back, 2026b. URL https://arxiv.org/abs/2604.03444.

Mehryar Mohri. Rational transductors, 2026. URL https://arxiv.org/abs/2602.07599.

Masato Neishi and Naoki Yoshinaga. On the relation between position information and sentence length in neural machine translation. In Mohit Bansal and Aline Villavicencio (eds.), Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL), pp. 328–338, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/K19-1031. URL https://aclanthology.org/K19-1031/.

Team Olmo, :, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan Lambert, Pete Walsh, Pradeep Dasigi, Robert Berry, Saumya Malik, Saurabh Shah, Scott Geng, Shane Arora, Shashank Gupta, Taira Anderson, Teng Xiao, Tyler Murray, Tyler Romero, Victoria Graf, Akari Asai, Akshita Bhagia, Alexander Wettig, Alisa Liu, Aman Rangapur, Chloe Anastasiades, Costa Huang, Dustin Schwenk, Harsh Trivedi, Ian Magnusson, Jaron Lochner, Jiacheng Liu, Lester James V. Miranda, Maarten Sap, Malia Morgan, Michael Schmitz, Michal Guerquin, Michael Wilson, Regan Huff, Ronan Le Bras, Rui Xin, Rulin Shao, Sam Skjonsberg, Shannon Zejiang Shen, Shuyue Stella Li, Tucker Wilde, Valentina Pyatkin, Will Merrill, Yapei Chang, Yuling Gu, Zhiyuan Zeng, Ashish Sabharwal, Luke Zettlemoyer, Pang Wei Koh, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi. Olmo 3, 2025. URL https://arxiv.org/abs/2512.13961.

Ofir Press, Noah Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=R8sQPpGCv0.

Jan Rosendahl, Viet Anh Khoa Tran, Weiyue Wang, and Hermann Ney. Analysis of positional encodings for neural machine translation. In Jan Niehues, Rolando Cattoni, Sebastian Stüker, Matteo Negri, Marco Turchi, Thanh-Le Ha, Elizabeth Salesky, Ramon Sanabria,

Loic Barrault, Lucia Specia, and Marcello Federico (eds.), Proceedings ofthe 16th International Conference on Spoken Language Translation, Hong Kong, November 2-3 2019. Association for Computational Linguistics. URL https://aclanthology.org/2019.iwslt-1.20/.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2023. URL https://arxiv.org/ abs/2104.09864.

Anej Svete and Ryan Cotterell. Transformers can represent n-gram language models. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), Proceedings ofthe 2024 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 6845–6881, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.381. URL https://aclanthology.org/2024.naacl-long.381/.

Anej Svete, Nadav Borenstein, Mike Zhou, Isabelle Augenstein, and Ryan Cotterell. Can transformers learn n-gram language models? In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 9851–9867, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.550. URL https: //aclanthology.org/2024.emnlp-main.550/.

Anej Svete, William Merrill, Ryan Cotterell, and Ashish Sabharwal. Revisiting padded transformer expressivity: Which architectural choices matter and which don’t. In Fortythird International Conference on Machine Learning, 2026. URL https://openreview.net/ forum?id=nBuL6HywFX.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, Léonard Hussenot, Pier Giuseppe Sessa, Aakanksha Chowdhery, Adam Roberts, Aditya Barua, Alex Botev, Alex Castro-Ros, Ambrose Slone, Amélie Héliou, Andrea Tacchetti, Anna Bulanova, Antonia Paterson, Beth Tsai, Bobak Shahriari, Charline Le Lan, Christopher A. Choquette-Choo, Clément Crepy, Daniel Cer, Daphne Ippolito, David Reid, Elena Buchatskaya, Eric Ni, Eric Noland, Geng Yan, George Tucker, George-Christian Muraru, Grigory Rozhdestvenskiy, Henryk Michalewski, Ian Tenney, Ivan Grishchenko, Jacob Austin, James Keeling, Jane Labanowski, Jean-Baptiste Lespiau, Jeff Stanway, Jenny Brennan, Jeremy Chen, Johan Ferret, Justin Chiu, Justin Mao-Jones, Katherine Lee, Kathy Yu, Katie Millican, Lars Lowe Sjoesund, Lisa Lee, Lucas Dixon, Machel Reid, Maciej Mikuła, Mateo Wirth, Michael Sharman, Nikolai Chinaev, Nithum Thain, Olivier Bachem, Oscar Chang, Oscar Wahltinez, Paige Bailey, Paul Michel, Petko Yotov, Rahma Chaabouni, Ramona Comanescu, Reena Jana, Rohan Anil, Ross McIlroy, Ruibo Liu, Ryan Mullins, Samuel L Smith, Sebastian Borgeaud, Sertan Girgin, Sholto Douglas, Shree Pandya, Siamak Shakeri, Soham De, Ted Klimenko, Tom Hennigan, Vlad Feinberg, Wojciech Stokowiec, Yu hui Chen, Zafarali Ahmed, Zhitao Gong, Tris Warkentin, Ludovic Peran, Minh Giang, Clément Farabet, Oriol Vinyals, Jeff Dean, Koray Kavukcuoglu, Demis Hassabis, Zoubin Ghahramani, Douglas Eck, Joelle Barral, Fernando Pereira, Eli Collins, Armand Joulin, Noah Fiedel, Evan Senter, Alek Andreev, and Kathleen Kenealy. Gemma: Open models based on gemini research and technology, 2024. URL https://arxiv.org/abs/2403.08295.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need, 2017. URL https://arxiv.org/abs/1706.03762.

P. Walters. An Introduction to Ergodic Theory. Graduate texts in mathematics. Springer-Verlag, 1982. ISBN 9783540905998. URL https://books.google.com/books?id=GCH\_wAEACAAJ.

Suyuchen Wang, Ivan Kobyzev, Peng Lu, Mehdi Rezagholizadeh, and Bang Liu. Resonance RoPE: Improving context length generalization of large language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Findings of the Association for

Computational Linguistics: ACL 2024, pp. 586–598, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.32. URL https://aclanthology.org/2024.findings-acl.32/.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks, 2024. URL https://arxiv.org/abs/2309. 17453.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025a. URL https://arxiv.org/abs/2505.09388.

Andy Yang and David Chiang. Counting like transformers: Compiling temporal counting logic into softmax transformers. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=FmhPg4UJ9K.

Andy Yang, David Chiang, and Dana Angluin. Masked hard-attention transformers recognize exactly the star-free languages, 2024. URL https://arxiv.org/abs/2310.13897.

Andy Yang, Michaël Cadilhac, and David Chiang. Knee-deep in c-rasp: A transformer depth hierarchy, 2025b. URL https://arxiv.org/abs/2506.16055.

Bowen Yang, Bharat Venkitesh, Dwaraknath Gnaneshwar Talupuru, Hangyu Lin, David Cairuz, Phil Blunsom, and Acyr Locatelli. Rope to nope and back again: A new hybrid attention strategy. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, pp. 64133– 64157. Curran Associates, Inc., 2025c. URL https://proceedings.neurips.cc/paper\_ files/paper/2025/file/5c9ab393551b7a39b4c02d88fe5e7e69-Paper-Conference.pdf.

Yechezkel Zalcstein. Locally testable languages. Journal ofComputer and System Sciences, 6 (2):151–167, 1972. ISSN 0022-0000. doi: https://doi.org/10.1016/S0022-0000(72)80020-5. URL https://www.sciencedirect.com/science/article/pii/S0022000072800205.

## Appendix Contents

A Preliminaries: Formal Language Theory . . . . . . . . . . . . 18   
A.1 Linear Temporal Logic . . . . . . . . . . . . . . . 18   
A.2 First-Order Logic . . . . . . . . 19   
A.3 Regular Expressions . . . . . . . 20   
A.4 Automata . . . . . . . . . . 21   
A.5 The Syntactic Monoid . . . . . . . . 22   
A.6 Equivalences between Formalisms . . . . . . . 23   
B Proofs: Characterizations of LTL[P, MOD] . . . . . 23   
B.1 Inexpressibility Results . . . . . . . . . . . . . 25   
C Proofs: Periodic RoPE. . . . . . 26   
C.1 Component-Periodic Schedules . . . . . . 26   
C.2 Upper Bounding Periodic RoPE . . . . . . . . . . . . . . . . 27   
C.3 Lower Bounding Periodic RoPE . . . . . 28   
C.4 Characterizing Periodic SiPE . . . . . . 30   
D Proofs: Non-Periodic RoPE . . . . . . . . . . . . . . . . 30   
E Experiments . . . . . . . 32   
E.1 Languages . . . . . . . 32   
E.2 Experimental Setup . . . . . . 32   
E.3 Periodic-Construction Settings . . . . . 32   
E.4 Conventional Non-Periodic Settings . . . . . . . . . . . . . . . . . . . . 32   
E.5 Detailed Results . . . . . 32

## A Preliminaries: Formal Language Theory

We introduced formal languages and linear temporal logic in $\ S 2$ . We now provide the detailed LTL semantics, and introduce the tools that provide a richer understanding of fragments of LTL. We summarize in Tab. 3 known characterizations of relevant fragments of LTL. In §B, we give a proof of the exact characterization of LTL[P, MOD] in terms of polynomials, first-order logic, automata and the syntactic monoid. Our proofs are based on similar characterizations of LTL[P] and LTL[P, F, MOD] (Li & Cotterell, 2025; Dartois & Paperman, 2013).

## A.1 Linear Temporal Logic

The full semantics for LTL are the following.

$$
\bullet w , i  = \pi _ { a } \iff w _ { i } = a ;
$$

$$
\bullet w , i  - \psi _ { 1 } \vee \psi _ { 2 } \Longleftrightarrow w , i  = \psi _ { 1 } \vee w , i  = \psi _ { 2 } ;
$$

$$
\bullet w , i \left| = \psi _ { 1 } \wedge \psi _ { 2 } \iff w , i \right| = \psi _ { 1 } \wedge w , i \left| = \psi _ { 2 } ; \right.
$$

$$
\bullet w , i  = \lnot \psi \iff w , i  \neq \psi ;right.
$$

$$
\bullet w , i \models \mathbf { P } \psi \iff \exists j < i : w , j \models \psi ;
$$

$$
\bullet w , i  = \mathbf { F } \dot { \boldsymbol { \psi } } \iff \exists j > i : w , j   = \dot { \boldsymbol { \psi } } ;
$$

$w , i \in \psi _ { 1 } \mathbf { S } \psi _ { 2 } \iff \exists j < i : w , j \models \psi .$ <sub>2</sub> and w, $k \models \psi _ { 1 }$ for all k with $j < k < i ;$

$w , i \nearrow \dot { \psi } _ { 1 } \dot { \bf U } \psi _ { 2 } \iff \ddot { \exists } j > i : w , j \ne \dot { \psi } _ { 2 }$ and $w , k \models \psi _ { 1 }$ for all k with $\ : i < k < j ; \ :$

$$
\bullet w , i  = \dot { \mathbf { Y } } \psi \iff w , i - 1  = \psi \mathrm { f o r } i > 1 ; 
$$

To define string acceptance, we denote by $N + 1$ a position outside of the string and define

$$
w \Vdash \psi \Leftrightarrow w , N + 1 \bigm | = \psi .\tag{11}
$$

For a set of temporal operators ${ \mathcal { O } } ,$ we denote by $\mathbf { L T L } [ \mathcal { O } ]$ the corresponding fragment of LTL. We say a formula ψ is a $\mathbf { L T L } [ \mathcal { O } ]$ formula if it can be written in LTL with only the Boolean connectives and operators from O. S subsumes the left-context-dependent operators P and Y (Gabbay et al., 1980).

## Example:

The language $a \Sigma ^ { * }$ of strings starting with a specific symbol a can be described by a LTL[P] formula:

$$
\mathbf { P } \left( \pi _ { a } \wedge \neg \mathbf { P } ^ { \top } \right)\tag{12}
$$

The language $\Sigma ^ { * } a$ of strings ending with a specific symbol a can be described by a LTL[Y] formula:

$$
\mathbf { Y } \pi _ { a }\tag{13}
$$

The language $a \Sigma ^ { * } a$ can be described by a $\mathbf { L T L } [ \mathbf { S } ]$ definable formula because $\mathbf { s }$ can express both P and $\breve { \mathbf { Y } } .$

LTL has become a key tool in characterizing the expressivity of fixed-precision transformers. Softmax transformers, leftmost hard-attention transformers and average hard-attention transformers are equivalent to LTL[P] (Li & Cotterell, 2025; Jerad et $\mathsf { a l . , } 2 0 2 5 ) .$ , while rightmost hard-attention transformers are equivalent to full $\mathbf { L T L } [ \mathbf { \dot { S } } ]$ (Yang et al., 2024).

As formalized in §2, LTL can be extended with modular predicates, denoted by $\mathbf { L T L } [ { \mathcal { O } } _ { \cdot }$ , MOD] for a set of operators O.

## A.2 First-Order Logic

First-order logic $( \mathbf { F O } [ < ] )$ can also be used to characterize subregular classes. $\mathbf { F O } [ < ] { ' } \mathbf { s }$ building blocks are atomic formulas, denoted by the unary predicate $\pi _ { a }$ for all $a \in \Sigma$ and the binary predicate $<$ which determines the order between string positions. We have $\pi _ { a } ( i ) = \dot { \top } \dot { \mathbf i }$ f and only if $w _ { i } = a$ . Building on these atomic formulas, $\mathbf { F O } | { < } |$ operates on variables, which denote positions of the input string. In $\mathbf { F O } [ < ]$ , a variable is either free or bounded by a quantifier. For example, in the existential quantification $\exists i < j , i$ is a bounded variable while j is a free variable. $\mathbf { F O } [ < ]$ formulas are then constructed inductively as follows.

(1) Every atomic formula is an $\mathbf { F O } [ < ]$ formula;

(2) A Boolean combination of $\mathbf { F O } [ < ]$ formulas is an $\mathbf { F O } | { < } |$ formula;

(3) If $\phi ( i \ldots )$ is an $\mathbf { F O } [ < ]$ formula, then so is $\exists i \colon \phi ( i \ldots ) ^ { 2 }$

Let ϕ be a formula without any free variables. We say $w \not = \phi$ if ϕ is satisfied when evaluating it on w. $\phi$ induces a language, which we denote by ${ \mathbb L } ( \phi ) \stackrel { \mathrm { d e f } } { = } \{ { \pmb w } \in \Sigma ^ { * } \mid { \pmb w } \mid = \phi \}$

Fragments of $\mathbf { F O } [ < ]$ : We denote by $\mathbf { F O } ^ { 2 } [ < ]$ the fragment of $\mathbf { F O } | { < } |$ restricted to using at most two distinct variables in any sub-formula. We denote by $\mathbf { P F O } ^ { \bar { 2 } } [ < ]$ the past fragment of $\mathbf { F O } ^ { 2 } [ < ]$ in which formulas are restricted so that whenever a free variable i is present, every quantifier must be of the form $\exists j : j < i \mathrm { o r } \forall j : j < i .$ . In other words, only past positions relative to the free variable are accessed. $\mathbf { P F O } ^ { 2 } [ < ]$ formulas can be inductively written as follows.

(1) Every atomic formula is a $\mathbf { P F O } ^ { 2 } [ < ]$ formula;

(2) A Boolean combination of $\mathbf { P F O } ^ { 2 } [ < ]$ formulas is a $\mathbf { P F O } ^ { 2 } [ < ]$ formula if it does not contain more than two variables;

(3) If $\phi ( i , j )$ is a $\mathbf { P F O } ^ { 2 } [ < ]$ formula with two free variables $i , j , \exists i < j \colon \phi ( i , j )$ and $\exists j <$ i : $\phi ( i , j )$ are $\mathbf { P F O } ^ { 2 } [ < ]$ formulas;

(4) If $\phi ( i )$ is a $\mathbf { P F O } ^ { 2 } [ < ]$ formula with one free variable $i , \exists i < j \colon \phi ( i )$ is a $\mathbf { P F O } ^ { 2 } [ < ]$ formula;

(5) If $\phi ( i )$ is a $\mathbf { P F O } ^ { 2 } [ < ]$ formula with one free variable i, ∃i : $\phi ( i )$ is a $\mathbf { P F O } ^ { 2 } [ < ]$ formula.

Over finite strings, $\mathbf { P F O } ^ { 2 } [ < ]$ and $\mathbf { L T L } [ \mathbf { P } ]$ define the same languages (Li & Cotterell, 2025), while $\mathbf { F O } [ < ]$ is equivalent to the full star-free class, which is LTL[S] (Kamp, 1968; Mc-Naughton $\dot { \& }$ Papert, 1971; Gabbay et al., 1980). We therefore use LTL[P] in the main narrative and retain $\mathbf { P F O } ^ { 2 } [ < ]$ when invoking first-order, automata-theoretic, or algebraic characterizations.

Modular predicates. Akin to $\mathbf { L T L } , \mathbf { F O } [ < ]$ can be extended with unary modular predicates, albeit with a slight modification to its LTL counterpart. In $\mathbf { L T L } ,$ formulas are evaluated at position $N + 1 .$ , meaning LTL can robustly make statements about the residue class of $N ,$ the length of the string. Because $\mathbf { F O } [ < ]$ formulas are evaluated by ranging variables over the entire string, such an operator cannot be directly implemented with unary predicates. Therefore, 0-ary predicates $\mathsf { \tilde { M } O D } _ { m } ^ { r }$ are needed, where $\mathsf { M O D } _ { m } ^ { r ^ { \star } }$ is true if and only if the length of the input string N is congruent to r modulo m.

Adding these modular predicates to first-order fragments gives $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ] , \mathbf { F O } ^ { 2 } [ < , \mathsf { M O D } ]$ and $\mathbf { F O } [ < , \mathsf { M O D } ] ;$ as shown in §3.4, LTL[P, MOD] and $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ define the same class of languages.

## A.3 Regular Expressions

Regular expressions are a declarative formalism for describing languages, defined recursively: (1) ε and each $w \in \Sigma$ is a regular expression; (2) If α and $\beta$ are regular expressions, so are the union $\alpha + \beta ,$ concatenation $\alpha \beta ,$ complement $\alpha ^ { \mathrm { { C } } }$ and the Kleene closure $\alpha ^ { * } , \ \mathrm { A }$ language is regular if it can be described by a regular expression. A regular language is star-free if it can be described by a regular expression without the Kleene operator.

Definition A.1. A monomial over an alphabet Σ is a language described via a regular expression of the form:

$$
\Sigma _ { 0 } ^ { * } w _ { 1 } \Sigma _ { 1 } ^ { * } \dots w _ { N } \Sigma _ { N } ^ { * }\tag{14}
$$

Where $\Sigma _ { 0 } , \Sigma _ { 1 } \dots \subseteq \Sigma$ and $w _ { 1 } , w _ { 2 } \dots \in \Sigma$

Definition A.2. A left-deterministic monomial is a monomial where for every $0 < i \leq N ,$ $w _ { i } \not \in \Sigma _ { i - 1 } .$ right-deterministic monomial is a monomial wherefor every $0 < i \leq N , w _ { i } \notin \Sigma _ { i }$ An unambiguous monomial is a monomial that is either left-deterministic or right-deterministic.

## Example:

The language $a \Sigma ^ { * }$ of strings starting with the symbol a is a left-deterministic monomial, since the marker a does not appear in the preceding empty block $( a \notin \Sigma _ { 0 } )$

Symmetrically, the language $\Sigma ^ { * } a$ of strings ending with the symbol a is a rightdeterministic monomial. Both $a \Sigma ^ { * }$ and $\Sigma ^ { * } a$ are therefore unambiguous monomials.

Their intersection $a \Sigma ^ { * } \cap \Sigma ^ { * } a = a \Sigma ^ { * } a$ is an unambiguous monomial.

A polynomial is a finite union of monomials. A left-deterministic (resp., right-deterministic, unambiguous) polynomial is a finite union of left-deterministic (resp., right-deterministic, unambiguous) monomials.

To adapt our analysis to RoPE transformers, we now introduce modular polynomials (Dartois & Paperman, 2013).

Definition A.3. A monomial over Σ is modular with respect to some integer d $i f i t$ can be written as:

$$
\left( \Sigma _ { 0 , 0 } \Sigma _ { 0 , 1 } \dots \Sigma _ { 0 , d - 1 } \right) ^ { * } w _ { 1 } \left( \Sigma _ { 1 , 0 } \Sigma _ { 1 , 1 } \dots \Sigma _ { 1 , d - 1 } \right) ^ { * } \dots w _ { N } \left( \Sigma _ { N , 0 } \Sigma _ { N , 1 } \dots \Sigma _ { N , d - 1 } \right) ^ { * }\tag{15}
$$

Where:

• Each marker $w _ { i }$ occurs at a position congruent to some $r _ { i }$ modulo $d ;$

• Each $\Sigma _ { i , j }$ is some subset of Σ;

• For every i and every position p lying in the i’th block, the symbol at position p belongs to $\Sigma _ { i , p m o d } \bar { _ { d } }$

Note that the standard monomial is recovered with $d = 1$

Definition A.4. A left-deterministic modular monomial is a modular monomial such thatfor every $0 < i \leq N , w _ { i } \ \dot { \not \in } \ \Sigma _ { i - 1 , r _ { i } } .$

As before, we similarly define unambiguous modular monomials, unambiguous modular polynomials and left-deterministic modular polynomials.

## A.4 Automata

The standard formalism for (sub)regular languages are finite automata—machines that transition within finitely many states.

Definition A.5. A semiautomaton A is a 3-tuple $\left( \Sigma , Q , \delta \right)$ where Σ is an alphabet, Q is a finite set of states and $\delta \colon Q \times \Sigma  Q$ is a transition function. We further define an initialized semiautomaton as a semiautomaton with an initial state.

Definition A.6. A deterministicfinite automaton (DFA) A is a 5-tuple $\left( \Sigma , { \cal Q } , q _ { \iota } , F , \delta \right)$ where $\left( \Sigma , Q , \delta \right)$ is a semiautomaton, $q _ { \iota } \in \dot { Q }$ is an initial state, and $F \subseteq Q$ is a set of final states.

We now introduce the automata that characterize the class of languages described by $\mathbf { P F O } ^ { 2 } [ < ]$ . Intuitively, because quantifiers in $\mathbf { P F O } ^ { 2 } [ < ]$ can only look backward, they can only accumulate information about the past. We now introduce a class of automata that mirror this behavior: Their states are partially ordered and transitions may only move upward in this order. Consequently, reading a symbol can only refine what the automaton knows about the prefix read so far—no information can be discarded, and no state can be revisited.

Definition A.7. Let $\delta ^ { * } \colon Q \times \Sigma ^ { * } \to Q$ be the transitive closure of δ, defined as

$$
\delta ^ { * } ( q , w ) = \delta ( q , w ) , f o r w \in \Sigma\tag{16a}
$$

$$
\delta ^ { * } ( q , w _ { 1 } \cdot \cdot \cdot w _ { N } ) = \delta ( \delta ^ { * } ( q , w _ { 1 } \cdot \cdot \cdot w _ { N - 1 } ) , w _ { N } )\tag{16b}
$$

with $\delta ^ { * } ( q , \varepsilon ) = q .$ for any $q \in Q .$ A partially ordered DFA (PODFA) is a DFA $\mathcal { A } = ( \Sigma , Q , q _ { l } , F , \delta )$ where there is a partial order relation ⪯ on Q defined as $q \preceq p i f$ and only $i f \delta ^ { * } ( q , w ) = p f o r$ some string $\boldsymbol { w } \in \Sigma ^ { * }$

As an example, the language of strings that contain the symbol a over any alphabet can be modeled by the following PODFA.

![](images/1d298a1da119640e6847fead9f06bab91c0c7e8b44812a4fac0f34ba315067fc.jpg)

Figure 4: PODFA for $\Sigma ^ { * } a \Sigma ^ { * }$ over $\Sigma = \{ a , b \}$ . The partial order $q _ { 0 } { \preceq } q _ { 1 }$ holds since $q _ { 1 }$ is reachable from q<sub>0</sub> but not vice versa.

To link automata with modular predicates, we now introduce the k-automaton of some given automaton. In first-order logic with modular predicates, strings can be evaluated to the same value up to a modulus. For some modulus $\cdot _ { k , \cdot }$ we therefore consider k-automata where strings are read as blocks of k symbols.

Definition A.8. Let $\boldsymbol { \mathcal { A } } \ = \ ( \Sigma , Q , q _ { \iota } , F , \delta )$ be a DFA. Let k be an integer. We define $\boldsymbol { \mathcal { A } } _ { k }$ <sup>def</sup> = $( \Sigma ^ { k } , Q , q _ { l } , F , \delta ^ { k } )$ as the k-automaton of A where:

$$
\bullet \ \Sigma ^ { k } \overset { \mathrm { d e f } } { = } \{ \pmb { w } \ | \ | \pmb { w } | = k a n d \ \pmb { w } \in \Sigma ^ { * } \}
$$

$$
\bullet \ \delta ^ { k } ( q , w _ { 1 } w _ { 2 } \ldots w _ { k } ) \overset { \mathrm { d e f } } { = } \delta ( \ldots \delta ( \delta ( q , w _ { 1 } ) , w _ { 2 } ) , w _ { k } ) .
$$

We denote by $\mathrm { P O D F A } _ { k }$ the set of automata such that their k-automaton are PODFAs. As an example, $( a \tilde { b } ) ^ { * }$ is the language of an automaton A in $\mathrm { P O D F A } _ { 2 } ,$ as seen in Fig. 5.

![](images/cbea034485a0c5fb32f363dbead9a113050856682f234665d47750b49e92aaf2.jpg)  
Figure 5: Minimal automata for $( a b ) ^ { * }$ and its corresponding 2-automaton.

## A.5 The Syntactic Monoid

The syntactic monoid is a tool that partitions $\Sigma ^ { * }$ into classes of strings, where each class contains strings that all contain the same syntactic information with respect to a given regular language. We now formalize the syntactic monoid of a language.

Definition A.9. A monoid M is a set equipped with a binary operation and an identity element.

For instance, the free monoid is the set $\Sigma ^ { * }$ equipped with the concatenation operation and the empty string ε as identity.

Definition A.10. A monoid M is in R ifand only iffor all $w _ { 1 } , w _ { 2 } , w _ { 3 } \in \mathbb { M } , w _ { 1 } w _ { 2 } w _ { 3 } = w _ { 1 }$ implies $w _ { 1 } w _ { 2 } = w _ { 1 }$ . A monoid M is in L if and only if for all $w _ { 1 } , w _ { 2 } , w _ { 3 } \in \mathbb { M }$ $w _ { 3 } w _ { 2 } w _ { 1 } = w _ { 1 }$ implies $w _ { 2 } w _ { 1 } = w _ { 1 }$

Definition A.11. The syntactic congruence $\preceq _ { \mathbb { I } }$ is the equivalence relation on $\Sigma ^ { * }$ given the language L such that for all x, $\pmb { y } \in \Sigma ^ { * }$ , we have $x { \preceq } _ { \mathbb { L } } y$ ifand only if:

$$
s x z \in \mathbb { L } \iff s y z \in \mathbb { L } \forall s , z \in \Sigma ^ { * }\tag{17}
$$

Given a language, the syntactic congruence partitions strings of $\Sigma ^ { * }$ into classes such that they are syntactically equivalent with respect to the language’s automaton.

Example:

Consider the language PARITY (introduced in $\ S 3 . 4 )$ and its syntactic congruence $\preceq _ { \mathbb { L } } . \preceq _ { \mathbb { L } }$ then has exactly two congruence classes, one for strings with an even numbers of 1s (which we denote by $\mathcal { C } _ { \mathrm { e v e n } } \breve { ) }$ and another for strings with an odd number of 1s (which we denote by $\mathcal { C } _ { \mathrm { o d d } } )$ . Example strings in each class are:

$$
\begin{array} { c } { 0 1 1 , 1 1 1 1 , 0 1 0 1 \in \mathcal { C } _ { \mathrm { e v e n } } } \\ { 1 , 1 0 0 1 1 , 1 1 0 1 \in \mathcal { C } _ { \mathrm { o d d } } } \end{array}
$$

Definition A.12. Let M be thefree monoid, L be a regular language and $\preceq _ { \mathbb { I } }$ its syntactic congruence. The syntactic monoid of L is the quotient monoid $\mathbb { M } / { \overset { \sim } { \ b { \preceq } } } _ { \mathbb { I } }$

Interestingly, the syntactic monoid of a regular language is isomorphic to the transition monoid of the minimal automaton accepting the language.

Finally, we now extend the monoid class R to account for modularity.

Definition A.13. Let M be the syntactic monoid of a regular language L with associated automaton A. Let ${ \mathbb M } _ { k }$ be the syntactic monoid of the language associated with $\bar { \boldsymbol { A } } _ { k }$ , the k-automaton. Then, M is in QR if and only if there exists an integer k such that ${ \mathbb M } _ { k }$ is in R.

## A.6 Equivalences between Formalisms

To conclude this section, we summarize in Tab. 3 the known characterizations of $\mathbf { F O } ^ { 2 } [ < ,$ , MOD] and $\mathbf { P F O } ^ { 2 } [ < ]$ , and introduce the analogous characterization of $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ which we prove in §B.
<table><tr><td>First-order logic</td><td>LTL</td><td>Regular expression</td><td>Syntactic Monoid</td><td>Automata</td><td>Note</td></tr><tr><td> $\mathbf { F O } ^ { 2 } [ < , \mathsf { M O D } ]$ </td><td> $\mathbf { L T L } [ \mathbf { P } , \mathbf { F } , \mathsf { M O D } ]$ </td><td>Unambiguous modular</td><td>QDA</td><td> $2 \mathrm { - P O D F A } _ { k }$ </td><td>Dartois &amp; Paperman (2013)</td></tr><tr><td> $\mathbf { P F O } ^ { 2 } [ < ]$ </td><td>LTL[P]</td><td>Left- deterministic</td><td>R</td><td>PODFA</td><td>Brzozowski &amp; Ellen (1980) and Li &amp;</td></tr><tr><td> $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ </td><td>LTL[P, MOD]</td><td>Left- deterministic modular</td><td>QR</td><td> $\mathrm { P O D F A } _ { k }$ </td><td>Cotterell (2025) §B</td></tr></table>

Table 3: Characterizations of fragments of first-order logic. The equivalences in the last row are novel.

## B Proofs: Characterizations of LTL[P, MOD]

In this section, we provide a rich characterization of $\mathbf { L T L } [ \mathbf { P } , \mathsf { M O D } ]$ via the following theorem.

Theorem 3.3. Let $\mathbb { L } \subseteq \Sigma ^ { * }$ be a regular language, M be its syntactic monoid, and A be the minimal DFA accepting it. The following assertions are equivalent:

(i) L is a left-deterministic modular polynomial

(ii) $\mathbb { M } \in \mathbf { \Theta } \dot { \mathbf { Q R } }$

(iii) $\exists k \in \mathbb { N }$ such that $\boldsymbol { \mathcal { A } } _ { k }$ is partially ordered

(iv) $\mathbb { L } = \mathbb { L } ( \phi )$ for some $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ ] formula $\phi$

(v) $\mathbb { L } = \mathbb { L } ( \dot { \psi } )$ for some $\mathbf { L T L } [ \dot { \mathbf { P } } , \mathsf { M O D } ]$ formula ψ

Most of the following lemmata are adaptations from equivalences proven in prior work (Dartois & Paperman, 2013; 2015; Li & Cotterell, 2025) applied on different fragments of first-order logic.

Lemma B.1. A language L has its syntactic monoid in QR if and only if there exists a k such that L’s automaton belongs to $\mathrm { P O D F A } _ { k }$

Proof. By definition, a language L has its syntactic monoid in QR if and only if there exists a k such that its k-automaton $\textstyle { \mathcal { A } } _ { k }$ has its syntactic monoid in R. The equivalence between the transition monoid of $\boldsymbol { \mathcal { A } } _ { k }$ being in R and $\boldsymbol { \mathcal { A } } _ { k }$ being a PODFA follows from Brzozowski $\&$ Ellen (1980). Therefore, L’s automaton belongs to $\mathrm { P O D F A } _ { k }$ for some k. ■

Lemma B.2. $I f \phi$ is a $\mathbf { P F O } ^ { 2 } [ < .$ , MOD] formula, then the syntactic monoid of ϕ’s language is in QR.   
If a language’s syntactic monoid is in QR, it is the language of some $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ formula.

Proof. The regular languages with a syntactic monoid in R are exactly the $\mathbf { P F O } ^ { 2 } [ < ]$ languages (Li & Cotterell, 2025). Moreover, for any logic characterized by a local variety V, augmenting the logic with modular predicates yields a logic characterized by QV (Dartois & Paperman, $2 0 1 5 )$ . Since $\mathbf { P F O } ^ { 2 } [ < ]$ is characterized by R and R is local (Almeida, 2025), augmenting $\mathbf { P F O } ^ { 2 } [ < ]$ with modular predicates gives the class characterized by QR. ■

Lemma B.3. $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ and $\mathbf { L T L } [ \mathbf { P } ,$ , MOD] characterize the same class of languages.

Proof. The equivalence without modular predicates has already been proven via structural induction (Li & Cotterell, 2025). Adding the same unary predicate MOD to both logics therefore preserves the equivalence. It remains to show that unary modular predicates in $\mathbf { L T L } [ \mathbf { P } , \mathsf { M O D } ]$ can express 0-ary predicates in $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$

Formulas in $\mathbf { L T L } [ \mathbf { P } ]$ are evaluated at the string position $N + 1$ . Moreover, $N$ is congruent to r modulo m if and only if $N + 1$ is congruent to $r + 1$ modulo m. Therefore, evaluating in LTL[P, MOD] the unary modular predicate MOD<sup>r+1</sup><sub>m</sub> at $N + 1$ is equivalent to evaluating the 0-ary predicate $\mathsf { M O D } _ { m } ^ { r }$ in $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ ■

To prove the equivalence between $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ and left-deterministic modular polynomials, we consider languages over enriched alphabets (Dartois & Paperman, 2013).

We denote by $\mathsf { P F O } ^ { 2 } [ < , \mathsf { M O D } _ { d } ]$ the restriction of $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ with congruences modulo $d .$ For any $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ formula, there exists an integer d such that the formula is also P $\mathsf { F O } ^ { 2 } [ < , \mathsf { M O D } _ { d } ]$ -definable—by for example setting d as the least common multiplier of all the moduli used.

Definition B.1 (Enriched alphabets; Dartois & Paperman 2013). The set $\Sigma _ { d } \overset { \mathrm { d e f } } { = } \Sigma \times ( \mathbb { Z } / d \mathbb { Z } )$ is the enriched alphabet $o f \Sigma ,$ , where $\mathbb { Z } / d \mathbb { Z } \overset { \mathrm { d e f } } { = } \left\{ 0 , 1 , \ldots , d - 1 \right\}$ . The projection $\mathsf { p r o j } \colon \Sigma _ { d } ^ { * } \to \Sigma ^ { * }$ removes the $\left( \mathbb { Z } / d \mathbb { Z } \right)$ component of every symbol $( a , i ) \in \Sigma _ { d }$ of strings in $\Sigma _ { d } ^ { * }$

Definition B.2 (Well-formed words; Dartois & Paperman 2013). Given some alphabet $\Sigma ,$ the enriched word $( w _ { 0 } , i _ { 0 } ) \dots ( w _ { N } , i _ { N } )$ is well-formed $\dot { i } f i _ { j } \equiv j$ (mod d) for all $0 \leq  \dot { \} } j \leq N$ . We denote by $K \subset \Sigma _ { d } ^ { * }$ the set of well-formed words.

Note that on $K ,$ , the projection operator proj is a bijective application. Indeed, every word $w = w _ { 0 } \cdot \cdot \cdot w _ { N } \in \Sigma ^ { * }$ has a unique preimage in K: the word $( \mathrm { { w } } _ { 0 } , 0$ mod $d ) \cdots ( w _ { N } ,$ N mod $d )$ obtained by annotating each position j with j mod $d .$ Surjectivity is immediate, and injectivity holds because distinct well-formed words over $\Sigma _ { d }$ differ in at least one symbol and therefore project to distinct words.

We can now describe $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ with enriched alphabets $\cdot ^ { 3 }$

Proposition B.1. $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } _ { d } ] \left( \Sigma ^ { * } \right) = \mathsf { p r o j } ( \mathbf { P F O } ^ { 2 } [ < ] ( \Sigma _ { d } ^ { * } ) \cap K )$

Proof. The statement has been shown to be true for $\mathbf { F O } ^ { 2 } [ < , \mathsf { M O D } ]$ (Dartois & Paperman, 2013). The result does not depend on how the binary predicate $<$ < is used, and therefore it also holds for $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ . Modular predicates of the form $\mathsf { M O D } _ { d } ^ { r }$ can be computed by disjuncting over all atomic formulas of the form $\pi _ { ( w , r ) }$ in $\mathbf { P F O } ^ { 2 } [ < ] ( \Sigma _ { d } ^ { * } ) \cap K$ for all $w \in \Sigma$ . For the converse, formulas over exclusively well-formed words can be equivalently described by formulas over words in $\Sigma ^ { * }$ with modular predicates. For instance, $\pi _ { ( w , r ) }$ in $\mathbf { \dot { P } F O } ^ { 2 } [ < ] ( \Sigma _ { d } ^ { * } ) \mathbf { \dot { \cap } } K$ can be written as $\pi _ { w } \wedge \mathsf { M O D } _ { d } ^ { r }$ in $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } _ { d } ] ( \Sigma ^ { * } )$ ■

We can now prove the following lemma.

Lemma B.4. A language is a left-deterministic modular polynomial if and only $i f i t$ can be described by $a \mathrm { \bf ~ P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ formula.

Proof. Recall that for any $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ formula describing a language L, there exists a $\mathsf { P F O } ^ { 2 } [ < , \mathsf { M O D } _ { d } ]$ formula, for some integer d, that also describes L. By Prop. B.1, L is therefore the projection of the well-formed words of a language $\mathbb { L } ^ { \prime }$ definable by a $\mathbf { P F O } ^ { 2 } [ < ] ( \Sigma _ { d } ^ { * } )$ formula. It remains to be shown via this projection that L can be equivalently described with a left-deterministic polynomial.

<sup>3</sup>We may write $\mathbf { P F O } ^ { 2 } [ < ] ( \Sigma ^ { * } )$ to specify that the logic is being used over the alphabet $\Sigma .$

Because $\mathbb { L } ^ { \prime }$ is definable by a $\mathbf { P F O } ^ { 2 } [ < ] ( \Sigma _ { d } ^ { * } )$ formula, it is a disjoint union of left-deterministic monomials over $\Sigma _ { d }$ (Li & Cotterell, 2025). Recall that over well-formed strings, proj is a bijection. Therefore, proj preserves disjoint unions. It thus suffices to show that $\mathsf { p r o j }$ maps each left-deterministic monomial over $\Sigma _ { d }$ to a left-deterministic modular monomial over $\scriptstyle \sum ,$ and conversely.

Fix an enriched monomial $M = \Xi _ { 0 } ^ { * } \bar { w } _ { 1 } \Xi _ { 1 } ^ { * } \cdot . . \bar { w } _ { N } \Xi _ { N } ^ { * }$ over $\Sigma _ { d } ,$ where $\Xi _ { i } \subseteq \Sigma \times ( \mathbb { Z } / d \mathbb { Z } )$ and $\overline { { \boldsymbol { w } } } _ { i } = \left( w _ { i } , r _ { i } \right)$ for all $0 \leq i \leq N ,$ , and set

$$
\Sigma _ { i , j } \stackrel { \mathrm { d e f } } { = } \{ w \mid ( w , j ) \in \Xi _ { i } \} .
$$

We claim that the projection of the well-formed words of M is exactly the modular monomial determined, in the sense of Def. A.3, by the markers $w _ { i } ,$ the residues $r _ { i } ,$ and the subsets $\Sigma _ { i , j } .$

Forward (⇒). Let $\begin{array} { r c l } { w } & { = } & { \mathsf { p r o j } ( \overline { { w } } ) } \end{array}$ for a well-formed $\begin{array} { l l l } { { \overline { { { w } } } } } & { { \in } } & { { M , } } \end{array}$ factored as $\begin{array} { r l } { \overline { { \mathbf { w } } } } & { { } = } \end{array}$ u<sub>0</sub> w<sub>1</sub> $\overline { { u } } _ { 1 } \ldots \overline { { w _ { N } } } \overline { { u } } _ { N }$ with $\bar { u } _ { i } \in \Xi _ { i } ^ { * }$ . Projecting yields $u = u _ { 0 } w _ { 1 } u _ { 1 } \_ \_ w _ { N } u _ { N }$ . Well-formedness places $\overline { { w _ { i } } } = \left( w _ { i } , r _ { i } \right)$ ) at a position ${ \mathrm { 1 } } \equiv r _ { i }$ (mod $d )$ , so each w stands at residue $r _ { i }$ in u. For any position p inside the block $u _ { i } ,$ its enriched letter is (w, p mod $d ) \in \Xi _ { i }$ , hence $w \in \Sigma _ { i , p }$ <sub>mod d</sub>. Thus u satisfies both conditions of Def. A.3 and lies in the modular monomial.

Reverse (⇐). Conversely, any u in that modular monomial admits a factorization meeting the same two conditions; tagging each position p of u with $p$ mod d produces the unique well-formed preimage u with $\mathsf { p r o j } ( \bar { u } ) \bar { = } u ,$ , and those conditions place u in M. No rotation or partial-block bookkeeping arises, since Def. A.3 is stated over global positions and already absorbs each block’s residue offset.

Finally, left-determinism transfers in both directions: for every $0 < i \leq N _ { \cdot }$

$$
\overline { { { w } _ { i } } } \notin \Xi _ { i - 1 } \iff ( w _ { i } , r _ { i } ) \notin \Xi _ { i - 1 } \iff w _ { i } \notin \Sigma _ { i - 1 , r _ { i } } ,
$$

the right-hand side being exactly the condition of Def. A.4. Hence M is left-deterministic if and only if its projected modular monomial is left-deterministic, and proj restricts to a bijection between left-deterministic monomials over $\Sigma _ { d }$ and left-deterministic modular monomials over Σ. Applying this correspondence to each term of the disjoint union describing $\mathbb { L } ^ { \prime } ,$ together with Prop. B.1, shows that L is in $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ if and only if it is a left-deterministic modular polynomial. ■

## B.1 Inexpressibility Results

Proposition B.2. The locally testable language $\Sigma ^ { * } a b \Sigma ^ { * }$ with $| \Sigma | > 2$ is not in $\mathbf { L T L } [ \mathbf { P } , \mathsf { M O D } ] =$ $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ] = \mathbf { S M A T } [ \mathrm { R o P E } _ { \mathrm { P } } ] ^ { 4 }$

Proof. Consider the minimal DFA A for $\Sigma ^ { * } a b \Sigma ^ { * }$ with the alphabet $\Sigma = \{ a , b , c \}$ , illustrated in Fig. 6.

![](images/5b98b0792d185f54c2e07f009db0dd2a93afc6c5318c6b2f3c19a35146a26280.jpg)  
Figure 6: Minimal DFA for $\Sigma ^ { * } a b \Sigma ^ { * }$ over $\Sigma = \{ a , b , c \}$

$\mathcal { A } _ { 1 } = \mathcal { A }$ can be seen to not be partially ordered. Consider some integer $k > 1$ and the k-automaton $\boldsymbol { \mathcal { A } } _ { k }$ of ${ \mathcal { A } } .$ Let ${ \boldsymbol { w } } _ { a } \in { \Sigma } ^ { k }$ be a string of k times the symbol $a ,$ and ${ \boldsymbol { w } } _ { c } \in { \Sigma } ^ { k }$ a string

<sup>4</sup>Note that over the two-letter alphabet $\Sigma ~ = ~ \{ a , b \} , \ \Sigma ^ { * } a b \Sigma ^ { * }$ is in $\mathbf { P F O } ^ { 2 } [ < ]$ and thus also in $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$

of k times the symbol c. We have that $\delta ^ { k } ( q _ { 0 } , w _ { a } ) = q _ { 1 }$ and $\delta ^ { k } ( q _ { 1 } , w _ { c } ) = q _ { 0 }$ . Therefore, there is no partial order on the states of $\boldsymbol { \mathcal { A } } _ { k }$ as $q _ { 0 }$ and $q _ { 1 }$ are mutually reachable. Thus, for any integer $k , A _ { k }$ can never be partially ordered, and $\dot { \Sigma } ^ { * } a b \Sigma ^ { * } \not \in \mathrm { \bf ~ P F O } ^ { 2 } [ < , \mathsf { M O D } ] \mathrm { \bf ~ f o r } | \Sigma | > 2$ ■

Proposition B.3. PARITY is not in $\mathbf { L T L } [ \mathbf { P } , \mathsf { M O D } ] = \mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ] = \mathbf { S M A T } [ \mathrm { R o P E } _ { \mathrm { P } } ]$

Proof. Consider the minimal DFA A for PARITY, illustrated in Fig. 7.

![](images/1a3dc2edcc37a3736608377c18cae8bed3a1217be4e59f6d2e8e39483451face.jpg)  
Figure 7: Minimal DFA for PARITY over $\Sigma = \{ 0 , 1 \}$

$\mathcal { A } _ { 1 } = \mathcal { A }$ can be seen to not be partially ordered. Consider some integer $k > 1$ and the k-automaton $\boldsymbol { \mathcal { A } } _ { k }$ of A. Let w be a string of length k with an odd number of 1s. We have that $\delta ^ { k } ( q _ { 0 } , w ) = q _ { 1 }$ and $\delta ^ { k } ( q _ { 1 } , w ) = q _ { 0 }$ . Therefore, there is no partial order on the states of $\boldsymbol { \mathcal { A } } _ { k }$ as $q _ { 0 }$ and $q _ { 1 }$ are mutually reachable. Thus, for any integer $k , \lrcorner A _ { k }$ can never be partially ordered, and PARITY̸∈ $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ ■

## C Proofs: Periodic RoPE

This section supplies the full proofs for periodic RoPE and for both inclusions summarized in $\ S 3 \colon$ : first $\mathbf { S M A T } [ \mathrm { R o P E } _ { \mathrm { P } } ] \subseteq \mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ , and then $\mathbf { L T L } [ \mathbf { P } , \mathsf { M O D } ] \subseteq \mathbf { S M A T } [ \mathbf { R o P E } _ { \mathrm { P } } ]$ . In all the following proofs, we follow the integer indexing convention for arrays: For $\boldsymbol { v } \in \mathbb { F } ^ { D }$ and a finite set of integers $\{ i _ { 1 } , \cdot \cdot \cdot i _ { n } \} , v [ i _ { 1 } , \cdot \cdot \cdot i _ { n } ] \in \mathbb { F } ^ { n }$ is the vector of elements in v indexed by $i _ { 1 } , \cdots i _ { n }$

## C.1 Component-Periodic Schedules

Proposition 2.1 (Component-periodicity via MOD). If a realized RoPE schedule $\widehat { \Theta }$ is componentperiodic, then its image is finite and, for every matrix coordinate and realized value, the positions attaining that value are definable by unary modular predicates.

Proof. Let $m _ { 1 } , \ldots , m _ { D / 2 }$ be component periods and $M \stackrel { \mathrm { d e f } } { = }$ lcm $\left( m _ { 1 } , \ldots , m _ { D / 2 } \right)$ . Then $\widehat { R } _ { i + M } =$ ${ \widehat { R } } _ { i } ,$ so the image is contained in $\left\{ \widehat { R } _ { 0 } , \ldots , \widehat { R } _ { M - 1 } \right\}$ . For a matrix coordinate c and realized value v, define ${ \cal A } _ { c , v } \stackrel { \mathrm { d e f } } { = } \left\{ a \in \{ 0 , \ldots , M - 1 \} : \widehat { R } _ { a } [ c ] = v \right\}$ . Then

$$
\widehat { \pmb { R } } _ { i } [ c ] = \boldsymbol { v } \quad \Longleftrightarrow \quad \bigvee _ { a \in A _ { c , v } } \mathsf { M } 0 \mathsf { D } _ { M } ^ { a } \left( i \right) .\tag{18}
$$

Thus every entry-value relation is definable with unary modular predicates.

Proposition 2.2 (Rational-angle functions are periodic). $I f \theta _ { d } / \left( 2 \pi \right) = a _ { d } / m _ { d } \in \mathbb { Q } ,$ , then $i \mapsto \mathbf { R o t } _ { \theta _ { d } , i }$ has period $m _ { d }$

Proof. We consider the mapping for cosine, as the sine analysis is the same. For some $i \in \mathbb { N } ,$ we have:

$$
\cos \left( \theta _ { d } i \right) = \cos \left( \frac { 2 \pi \theta _ { d } } { 2 \pi } i \right)\tag{19a}
$$

$$
= \cos \left( 2 \pi \frac { a _ { d } } { m _ { d } } i \right)\tag{19b}
$$

$$
= \cos \left( 2 \pi { \frac { a _ { d } } { m _ { d } } } i + 2 a _ { d } \pi \right)\tag{19c}
$$

$$
= \cos \left( 2 \pi { \frac { a _ { d } } { m _ { d } } } ( i + m _ { d } ) \right)\tag{19d}
$$

$$
= \cos { \left( \theta _ { d } ( i + m _ { d } ) \right) }\tag{19e}
$$

Therefore, cos $( \theta _ { d } i )$ is periodic with integer period $m _ { d }$

## C.2 Upper Bounding Periodic RoPE

Lemma 3.1. Any F-transformer with $\mathrm { R o P E _ { P } }$ can be simulated by $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ . Hence, SMAT[Ro $\mathtt { P E } _ { \mathtt { P } } \mathtt { I } \subseteq \mathtt { P F O } ^ { 2 } [ < , \mathsf { M O D } ] = \mathtt { L T L } [ \mathtt { P } , \mathsf { M O D } ]$

Proof. We formalize what it means for $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ to simulate a transformer.

Definition C.1 (Li & Cotterell 2025, Definition B.1.). A function $\lambda \colon ( \overline { { \Sigma } } ) ^ { * } \to ( \mathbb { F } ^ { D } ) ^ { * }$ is said to be simulated by $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ if for every dimension $d \in [ D ]$ and every floating-point value $f \in \mathbb { F } ,$ there exists a single-variable formula $\phi$ in $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ such that for every position $i \in [ N ] .$

$$
\lambda ( \pmb { w } ) _ { d , i } = f \iff \pmb { w } , i \neq \phi\tag{20}
$$

Similarly, we can define the simulation of a matrix-valuedfunction via a two-variable $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ formula.

We prove that $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ ] can simulate any RoPE<sub>P</sub> transformer. First, each realized matrix entry can be simulated by the formulas from Prop. 2.1.

A realized rotation matrix ${ \widehat { R } } _ { i }$ can be seen as a function mapping positions i to elements in $\mathbb { F } ^ { D \times D }$ . For every matrix coordinate $c \in [ D \times D ]$ and floating-point value $f \in \mathbb { F } .$ , we define a $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ formula $\phi _ { c , f }$ such that:

$$
\widehat { \pmb { R } } _ { i } [ c ] = f \iff w , i \ v { S } \ v { S } \ v { S } \Vdash { \phi } _ { c , f }\tag{21}
$$

0-elements of the matrix: For the 0-elements of ${ \widehat { R } } _ { i } ,$ we create constant formulas $\phi _ { \intercal } \overset { \mathrm { d e f } } { = } \intercal$ and $\phi _ { \perp } \ { \stackrel { \mathrm { d e f } } { = } } \ \perp$ . If the floating-point value f is $0 ,$ we assign the constant formula $\phi _ { \top }$ . Otherwise, we assign $\phi _ { \perp }$

Position-dependent elements of the matrix. For every coordinate c and realized value $v ,$ the formula $\mathsf { V } _ { a \in A _ { c , v } } \mathsf { M O D } _ { M } ^ { a } \left( i \right)$ is true exactly where ${ \widehat { R } } _ { i } [ c ] = v$ (Prop. 2.1). This covers every block entry without recovering an ideal angle from its finite-precision realization.

Therefore, we can simulate the rotation matrices with $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ . Omitting the periodic rotations, all other components of a fixed precision soft-attention transformer can be simulated by $\mathbf { P F O } ^ { 2 } [ < ]$ (Li & Cotterell, 2025). Finally, as the elements of the rotation matrices belong to a finite set (due to Prop. 2.1), the rotation matrices belong themselves to some finite set. Therefore, the rotation of the keys and queries is a mapping between two finite sets. Any mapping between finite sets can be simulated via a disjunction over the finitely-many inputs in the domain that lead to some output in the image (Chiang et al., 2023; Li & Cotterell, 2025). This concludes the proof.

Proposition 3.1. Binary modular predicates can be expressed with unary modular predicates and Boolean logic.

Proof. We will show that $\mathsf { M O D } _ { m } ^ { r } ( i , j )$ can be written as the following finite disjunction.

$$
\mathsf { M O D } _ { m } ^ { r } ( i , j ) = \bigvee _ { a = 0 } ^ { m - 1 } ( \mathsf { M O D } _ { m } ^ { a } ( i ) \wedge \mathsf { M O D } _ { m } ^ { ( a + r ) \bmod m } ( j ) )\tag{22}
$$

We denote by $i _ { 0 }$ and $j _ { 0 }$ the residues such that $i \equiv i _ { 0 }$ (mod m) and $j \equiv j _ { 0 }$ (mod m).

$\mathsf { M O D } _ { m } ^ { r } ( i , j ) = \top$ if and only $\operatorname { i f } \left( j - i \right) \equiv r$ (mod m). Equivalently, $( j _ { 0 } - i _ { 0 } ) \equiv r ~ ( \mathrm { m o d } ~ m ) _ { \cdot }$ , i.e. $j _ { 0 } \equiv r + i _ { 0 }$ (mod m). Therefore, $\mathsf { M O D } _ { m } ^ { r } ( i , \dot { j } ) = \top$ if and only if there exist residues $i _ { 0 }$ and $j _ { 0 }$ such that $1 ) \ : i \equiv i _ { 0 } { \pmod { m } } 2 ) \ : j \equiv j _ { 0 }$ (mod m) with $j _ { 0 } \equiv i _ { 0 } + r$ (mod m). This statement is exactly the right hand side of the given equation. ■

## C.3 Lower Bounding Periodic RoPE

On finite precision. Before proving the lower bound $\mathbf { L T L } \mathbf { \left[ P , M O D \right] } \subseteq \mathbf { S M A T } \mathbf { \left[ R o P E _ { P } \right] } ,$ , we recall that our result holds for some finite-precision regime F. In other words, we design a tailored finite set of values F that works for our constructions. This assumption is implicitly made in related work (Yang et al., 2024). This allows us to assume that the finitely many values our constructions manipulate are exactly representable, i.e. are present in F. For each modulus m, we specify the required roots of unity directly as an m-entry rotation table and include its finitely many entries in F. A cyclic counter or modular index return the corresponding table entry. This establishes an exact result in the abstract expressivity model; it does not assert that standard floating-point hardware represents $2 \pi / m ,$ the corresponding roots of unity, or the unbounded product $2 \pi i / m$ exactly. The experiments in $\ S 3 . 5$ test a practical finite-precision implementation empirically.

Lemma 3.2. For every LTL[P, MOD]formula, there is an F-transformer with $\mathrm { R o P E _ { P } }$ that simulates theformula. Hence $\mathbf { L } \mathbf { \check { T } L } \mathbf { [ P , \check { M } O D ] } \subseteq \mathbf { \check { S } M A T } \mathbf { [ R o P E _ { P } ] }$

Proof. We formalize what it means for SMAT[RoPE ] to simulate a LTL[P, MOD] formula.

Definition C.2 (Li & Cotterell 2025, Definition B.9.). A LTL[P, MOD] formula ψ is simulated by a function λ : $( \overline { { \Sigma } } ) ^ { * } \to ( \mathbb { F } ^ { D } ) ^ { * }$ if there exists a coordinate $d \in [ D ]$ such that for every input string w $\ b { \in } ( \overline { { \Sigma } } ) ^ { * }$ and string position $i \in [ N ]$

$$
\lambda ( \pmb { w } ) _ { d , i } = \left\{ \begin{array} { l l } { 1 i f w , i \Vdash \emptyset } \\ { 0 o t h e r w i s e } \end{array} \right.\tag{23}
$$

We proceed by structural induction on the LTL[P, MOD] formula.

Induction on LTL[P] without MOD. Any LTL[P] formula can be simulated by a transformer (without PEs) via structural induction on its constituents (Li & Cotterell, 2025). To lift this to $\begin{array} { r } { \dot { \bf S } { \bf M } { \bf A } { \bf T } [ { \bf R o } { \mathrm { P E } } _ { \mathrm { P } } ] . } \end{array}$ , we allocate dedicated dimensions for the Boolean connectives and P—the sole temporal operator of LTL[P]—and set their rotation angles to 0 (identity matrices), so that RoPE has no effect on those dimensions. We then apply Li & Cotterell’s (2025) simulation directly, yielding $\mathbf { L T L } [ \mathbf { P } ] \subseteq \mathbf { S M A T } [ \mathbf { R o P E } _ { \mathrm { P } } ]$

Simulating modular predicates. It remains to realize each modular atom. Consider the formula $\mathsf { M O D } _ { m } ^ { r } \left( i \right)$ . We isolate the case $i = 1$ , where ${ \mathsf { M O D } } _ { m } ^ { r } ( 1 )$ is then a compile-time constant. The first position can be flagged by performing uniform attention over the strict left context and checking the count is exactly 1, which we assume is in F. We can then store ${ \mathsf { M O D } } _ { m } ^ { r } ( 1 )$ at that position. Moreover, the case $m = 1$ is the constant-true predicate, so we proceed to the $m \geq 2 , i \geq 2 \mathrm { c a s e }$

We first allocate one rotary pair at dimensions $d , d + 1$ with the exact period-m lookup table (see §2.4)

$$
\begin{array} { r l } & { T _ { m } \left( s \right) \stackrel { \mathrm { d e f } } { = } \mathbf { R o t } _ { 2 \pi / m , s } \quad \left( 0 \leq s < m \right) , \qquad \widehat { R } _ { i } ^ { \left( d \right) } \stackrel { \mathrm { d e f } } { = } T _ { m } \left( i \bmod m \right) , } \\ & { \qquad \delta _ { m } \stackrel { \mathrm { d e f } } { = } \underset { 1 \leq s < m } { \operatorname* { m i n } } \left( 1 - \cos \left( 2 \pi s / m \right) \right) > 0 , \quad \quad b _ { m } \stackrel { \mathrm { d e f } } { = } 1 - \delta _ { m } / 2 . } \end{array}\tag{24}
$$

We also allocate one unrotated pair of dimensions $d ^ { \prime } , d ^ { \prime } + 1$ . All other dimensions are unused $( \mathrm { i . e . }$ , manually set to 0 in keys and queries) in our construction.

Query. Before the rotations, we set the elements of the query projection to positionindependent constants. These constants are implemented by setting the query weight matrix Q to a zero-matrix and encoding the desired values in the query bias vector $\boldsymbol { b } _ { \boldsymbol { q } } ,$ so that $Q x _ { i } ^ { \ell } + b _ { q } = b _ { q }$ for all inputs $x _ { i } ^ { \ell }$

$$
\begin{array} { r l } & { \quad ( Q x _ { i } ^ { \ell } ) [ d , d + 1 ] = \binom { \cos { \left( 2 \pi r / m \right) } } { - \sin { \left( 2 \pi r / m \right) } } } \\ & { \quad ( Q x _ { i } ^ { \ell } ) [ d ^ { \prime } , d ^ { \prime } + 1 ] = \binom { 1 } { 0 } } \end{array}
$$

The post-rotation query dimensions then become:

$$
\begin{array} { r l } & { q _ { i } ^ { \ell } [ d , d + 1 ] = \left( \cos \left( \frac { 2 \pi i } { m } \right) \cos \left( \frac { 2 \pi r } { m } \right) + \sin ( \frac { 2 \pi i } { m } ) \sin \left( \frac { 2 \pi r } { m } \right) \right) } \\ & { q _ { i } ^ { \ell } [ d , d + 1 ] = \left( \cos \left( \frac { 2 \pi i } { m } \right) \cos \left( \frac { 2 \pi r } { m } \right) - \cos \left( \frac { 2 \pi i } { m } \right) \sin \left( \frac { 2 \pi r } { m } \right) \right) } \\ & { q _ { i } ^ { \ell } [ d ^ { \prime } , d ^ { \prime } + 1 ] = \binom { 1 } { 0 } } \end{array}
$$

Key. We set the pre-rotation dimensions of the key vector as follows.

$$
\begin{array} { r l } & { ( K x _ { j } ^ { \ell } ) [ d , d + 1 ] = \binom { C \mathbb { 1 } \left\{ w _ { j } = \mathrm { B O S } \right\} } { 0 } } \\ & { ( K x _ { j } ^ { \ell } ) [ d ^ { \prime } , d ^ { \prime } + 1 ] = \binom { C b _ { m } ( 1 - \mathbb { 1 } \left\{ w _ { j } = \mathrm { B O S } \right\} ) } { 0 } } \end{array}
$$

Where C is a positive constant in F defined later, and $\mathbb { 1 } \{ w _ { j } = \mathtt { B O S } \}$ can be computed via a feedforward network.

Note that because BOS correspond to position 0, the dimensions $d , d + 1$ are unaffected by rotation: At BOS the rotation is an identity mapping, at a symbol position the vector is a 0-vector. Moreover, the dimensions $d ^ { \prime } , d ^ { \prime } \overset { } { + } 1$ are by construction unrotated.

Score. The query–key computation score yields:

$$
( \pmb { q } _ { i } ^ { \ell } ) ^ { \top } \pmb { k } _ { j } ^ { \ell } = \left\{ \begin{array} { l l } { C \cos \left( 2 \pi \left( i - r \right) / m \right) , } & { \mathrm { i f } w _ { j } = \mathrm { B O S } } \\ { C b _ { m } , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{25}
$$

Notice that cos $\left( 2 \pi \left( i - r \right) / m \right)$ is maximized when $i \equiv r { \pmod { m } }$

Modular predicate condition satisfied. If $i \equiv r$ (mod m), the score with BOS is C and the scores with non-BOS positions are all $\overline { { C ( 1 } } - \delta _ { m } / 2 )$ . Therefore, the largest score is C by a margin of $C \delta _ { m } / 2$

Modular predicate condition not satisfied. Suppose $i \not \equiv r$ (mod m). Then cos $( 2 \pi ( i - r ) / m ) \leq$ $1 - \delta _ { m }$ by definition of $\delta _ { m } ,$ so the score with BOS satisfies C cos $( 2 \pi ( i - r ) / m ) \leq C ( 1 - \delta _ { m } )$ Every non-BOS position still scores exactly $C b _ { m } = C ( 1 - \delta _ { m } / 2 )$ , independently of j. Hence every non-BOS position now beats BOS, by a margin of at least

$$
C ( 1 - \delta _ { m } / 2 ) - C ( 1 - \delta _ { m } ) = C \delta _ { m } / 2 .\tag{26}
$$

So, in either case $( i \equiv r$ (mod m) or not), whichever side wins—BOS or the (tied) non-BOS positions—does so by a margin of at least $C \delta _ { m } / 2$

Choosing C and extracting the predicate. We define the value vector to be 1 at BOS and 0 at every other position, so that the attention output at position i is exactly $\alpha _ { i , \mathrm { B O S } }$ . Selecting $C \geq 2 f _ { \mathrm { l a r g e } } / \dot { \delta } _ { m }$ makes the margin $C \delta _ { m } / 2$ established above at least $f _ { \mathrm { l a r g e } }$ in both cases. By stable softmax (§2), subtracting the maximum score from every score leaves an exponent of 0 at every position attaining the maximum and an exponent at most $- f _ { \mathrm { l a r g e } }$ —hence rounding to $\mathrm { e x p } ( - f _ { \mathrm { l a r g e } } ) = 0 .$ —at every position on the losing side of the margin.

• If $i \equiv r$ (mod m): BOS uniquely attains the maximum score, so $\pmb { \alpha } _ { i , \mathrm { B O S } } = 1$ and $\alpha _ { i , j } = 0$ for every non-BOS j. The attention output is 1.

• If $i \not \equiv r$ (mod m): Every non-BOS position attains the (tied) maximum score, so $\pmb { \alpha } _ { i , \mathrm { B O S } } = 0 .$ , while the non-BOS weights $\alpha _ { i , j }$ absorb the remaining mass. The attention output is still $\alpha _ { i , \mathrm { B O S } } = 0 .$ , regardless of how stable softmax splits weight among the tied non-BOS positions.

The attention output then equals $\mathbb { 1 } \left\{ \mathsf { M O D } _ { m } ^ { r } ( i ) \right\}$ : The rotary pair $d , d + 1$ supplies the positiondependent score against BOS, while the unrotated pair $\bar { d } ^ { \prime } , d ^ { \prime } + \mathrm { \ i }$ supplies the constant non-BOS baseline $C b _ { m }$ that this score is compared against, and together they correctly compute $\mathsf { M O D } _ { m } ^ { r } ( i )$ .

Completing the induction. A formula contains only finitely many modular atoms, so it requires finitely many rotary pairs and constants. Choose F to contain the entries of their finite lookup tables and all displayed intermediate values. This completes the induction. ■

## C.4 Characterizing Periodic SiPE

Theorem 3.2. SM $\mathrm { \bf { I A T } } [ \mathrm { \cal { S i P E } } _ { \mathrm { \bf { P } } } ] = \mathrm { \bf { I T L } } [ \mathrm { \bf { P } } , \mathsf { \sf { M O D } } ] = \mathrm { \bf { P F O } } ^ { 2 } [ < , \mathsf { \sf { M O D } } ] .$

Proof. We now show both directions of the equivalence.

Forward (⇒). Every non-PE component of a $\mathbf { S M A T } [ \mathrm { S i P E _ { P } } ]$ transformer is in $\mathbf { P F O } ^ { 2 } [ < ]$ (Li & Cotterell, 2025). Periodic sinusoidal position encodings can also be expressed via unary modular predicates by Prop. 2.1, yielding the inclusion in $\mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$

Reverse (⇐). NoPE transformers, which are subsumed by SMAT[SiPE ] transformers, can express $\mathbf { P F O } ^ { 2 } [ < ]$ . Simulating modular predicates can be done by processing sinusoidal position encodings with feedforward networks (Yang et al., 2024) using an appropriate schedule and pre-computed lookup table (§2.4), leading to the reverse direction. Note that Yang et al.’s (2024) modular predicate simulation does not rely on the attention type. ■

## D Proofs: Non-Periodic RoPE

Because Cor. 4.1 certifies the simulation only up to a finite length, we first record the length-bounded variant of Def. C.2 that it uses.

Definition D.1 (Bounded-length simulation). Let $L \in \mathbb { N }$ . A formula ψ is L-simulated by a function $\lambda \colon ( \overline { { \Sigma } } ) ^ { * } \to ( \mathbb { F } ^ { D } ) ^ { * }$ if there exists a coordinate d $\in [ D ]$ such that for every input BOSwEOS with $| w | \le \dot { L }$ and every position $i \in [ | w | + 1 ]$

$$
\lambda \left( \mathrm { B O S } w \mathrm { E O S } \right) _ { d , i } = \left\{ \begin{array} { l l } { { 1 } } & { { i f w , i \left| = \psi \right. } } \\ { { 0 } } & { { o t h e r w i s e . } } \end{array} \right.\tag{27}
$$

Taking $L \to \infty$ recovers Def. C.2.

Throughout, positions are indexed so that BOS occupies position 0, the string occupies positions $1 , \ldots , | w |$ , and EOS occupies position $| \boldsymbol { w } | + 1$ , which is the readout position. Hence $| { \boldsymbol w } | \leq L$ makes every query position at most $\dot { L } + 1$ , which is exactly the range quantified over in Eq. (10).

Corollary 4.1. Fix $g , C ,$ and $\mathbb { F } ,$ and let $L \in \mathbb { N }$ . Write SMAT[RoPE<sub>NP</sub>]-transformers for softattention RoPE transformers with unrotated subspaces, and let L-simulation be as in Def. D.1.

(i) For every offset $k \geq 1$ with $k \le L \le N _ { \operatorname* { m a x } } ^ { ( k ) } \left( g , C , \mathbb { F } \right)$ , such a transformer L-simulates the fixed-offset operator $\boldsymbol { r } ^ { k }$

(ii) $I f 1 \le L \le N _ { \mathrm { m a x } } ^ { ( 1 ) } \left( g , C , \mathbb { F } \right)$ , it L-simulates every LTL[P, Y]-definable formula. More generally, for a finite set of offsets $K \subseteq \mathbb { N } _ { \geq 1 }$ with max $K \le L \le \operatorname* { m i n } _ { k \in K } N _ { \operatorname* { m a x } } ^ { ( k ) } \left( g , C , \mathbb { F } \right)$ , it L-simulates every formula built from atomic predicates, Boolean connectives, P, and the operators $\{ Y ^ { k } : k \in K \}$

(iii) Moreover, for every $L \in \mathbb { N }$ and every finite offset set K there exist a rotation angle g with $g / \left( 2 \pi \right) \check { \notin } \mathbb { Q } ,$ , scales C, and a finite-precision regime F satisfying the hypothesis of (ii).

Proof. Allocate unrotated dimensions to the standard structural-induction simulation of LTL[P] (Li & Cotterell, 2025). Fix $k \geq 1$ . To simulate a $\mathbf { Y } ^ { k } \boldsymbol { \psi }$ subformula directly, use one rotary pair $d , d + 1$ in a dedicated attention head, with query bias C (cos $( k g ) , \dot { - } \sin { ( k g ) } \rangle$ and key bias $( 1 , 0 )$ at those two coordinates. In this head we set Q and K to the zero matrix and set every coordinate of $b _ { q }$ and $\boldsymbol { b } _ { k }$ outside the pair $d , d + 1$ to 0. The score is therefore exactly the contribution of that pair: The unrotated dimensions carrying the LTL[P] simulation, and every other rotary pair, contribute 0 to it. In particular the biases are the head’s only source of query and key content, so the score is position-dependent but content-independent, as the fixed-offset semantics of $\mathbf { Y } ^ { k }$ requires.

After rotation at query position i and key position $j ,$ exact arithmetic gives

$$
\begin{array} { r } { { S _ { i , j } ^ { g , C , k } = C \cos \left( g \left( i - k - j \right) \right) , } } \end{array}\tag{28}
$$

which is Eq. (9). Under the model’s actual arithmetic, denote the realized score by $\widehat { s } _ { i , j } ^ { g , C , k , \mathbb { F } }$ as in Eq. (10). Store the truth value of ψ in the value vector at each string position and store 0 at BOS.

We prove (i). Fix $k \le L \le N _ { \mathrm { m a x } } ^ { ( k ) } \left( g , C , \mathbb { F } \right)$ . At every $k + 1 \leq i \leq L + 1$ , position $i - k$ beats every other key by at least $f _ { \mathrm { l a r g e } } ,$ by the definition of $N _ { \mathrm { m a x } } ^ { ( k ) } ;$ note that the competing keys include BOS at $j = 0$ , whose rotation matrix is the identity, so its key is $( \cos 0 , \sin 0 ) = ( 1 , 0 )$ Stable softmax therefore assigns weight 1 to $i - k$ and 0 elsewhere, so the head returns the truth value of ψ exactly k positions earlier. Since $i \geq k + 1$ , the attended position $i - k$ is a string position, not BOS. At the boundary $i \leq k ,$ the semantics require $\boldsymbol { \Upsilon } ^ { k } \boldsymbol { \psi }$ to be false. The unrotated construction simulates the gate $\mathbf { P } ^ { k } \top$ , which holds exactly when $i > k -$ indeed P⊤ holds at i iff $i > 1$ , and inductively $\mathbf { P } ^ { k } \top = \mathbf { P } \left( \mathbf { P } ^ { k - 1 } \top \right)$ holds at i iff some $j < i \mathrm { h a s } j > k - 1$ i.e. iff $i > k ; \mathsf { a }$ feedforward Boolean conjunction with this gate therefore suppresses the arbitrary attention output at those boundary positions, mapping it to 0 regardless of its value. This L-simulates $\mathbf { Y } ^ { k } .$ , proving (i).

For (ii), every LTL[P, Y] formula is built from atomic predicates, Boolean connectives, P, and Y, so it needs only the offset $k = 1 \colon$ : one layer per Y occurrence, applying (i) with $k = 1 ,$ which is available because $L \leq N _ { \operatorname* { m a x } } ^ { ( 1 ) } \left( g , C , \mathbb { F } \right)$ by hypothesis. Together with the unrotated P and Boolean constructions this completes the structural induction. For a finite offset set K, give each $k \in K$ its own head and rotary pair; the hypothesis $L \leq \operatorname* { m i n } _ { k \in K } N _ { \operatorname* { m a x } } ^ { ( k ) } \left( g , C , \mathbb { F } \right)$ makes (i) available for every $k \in K$ simultaneously, and max $K \leq L$ makes each application well posed.

For (iii), fix any finite L and finite collection of offsets, and choose $g / \left( 2 \pi \right) \notin \mathbb { Q }$ . Then $\delta _ { \cal L } \left( g \right) > 0$ . Choose the scales C and a finite-precision regime jointly so that $C \delta _ { L } \left( g \right) \geq f _ { \mathrm { l a r g e } }$ for every corresponding head and all parameters and intermediate sine, cosine, product, and score values used by the constructions through positions $L + 1$ are represented exactly. This is possible in the abstract finite-value model because only finitely many offsets and nonzero values are required, while the regime can retain a separate underflow cell around zero. On that finite range the ideal identities are evaluated exactly, so Eq. (10) gives $N _ { \mathrm { m a x } } ^ { ( k ) } \left( g , C , \mathbb { F } \right) \ge L$ for every selected k. ■

## E Experiments

We supplement the periodic and conventional non-periodic experiments in §§ 3.5 and 4.4 with additional details.

## E.1 Languages

We first introduce the different languages we train our transformers on.

$( a b ) ^ { * }$ : the language of strings in which the substring ab is repeated arbitrarily many times. It is star-free, inside $\mathbf { L T L } [ \mathbf { P } , \mathsf { M O D } ] = \mathbf { P F O } ^ { 2 } [ < , \mathsf { M O D } ]$ and outside of $\mathbf { P F O } ^ { 2 } [ < ]$ . It is also the Dyck language (well-balanced parentheses) with 1 pair of parentheses, limited to depth 1.

$( a a ) ^ { * }$ : the language of strings of even length. It is a canonical example of a non-starfree, LTL[P, MOD] language.

$a \Sigma ^ { * } ;$ the language of strings starting with a designated symbol a over some alphabet Σ such that $| \Sigma | > 1$ . It is the simplest language of the class $\mathbf { P } \mathbf { F } \mathbf { O } ^ { 2 } [ < ] = \mathbf { L } \mathbf { T } \mathbf { L } [ \mathbf { \bar { P } } ]$

$\Sigma ^ { * } a \colon$ the language of strings ending with a designated symbol a over some alphabet Σ such that $| \Sigma | > 1$ . It belongs in LTL[F] and $\check { \mathbf { L T L } } [ \mathbf { Y } ]$

$\Sigma ^ { * } a b \Sigma ^ { * }$ : the language of strings containing the contiguous substring ab. It is locally testable and definable in $\mathbf { L T L } \mathbf { \Big [ P , Y \Big ] }$

## E.2 Experimental Setup

Our experimental setup follows Delétang et al. (2023); Li & Cotterell (2025). Our transformers use soft attention and strict future masking, with 5 layers, model dimension set to 64, and 8 attention heads. Training strings are of length up to 40, while test strings range from length 41 to 500. The model is trained for 1,000,000 steps with a batch size of 128. For evaluation, we generate 512 samples per test length. Each experiment is run with 5 different random seeds and 3 learning rates $( \hat { 1 } \times 1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 4 } , 5 \times \hat { 1 0 } ^ { - 4 } )$

## E.3 Periodic-Construction Settings

The periodic experiments in Tab. 2 compare the period-targeting $\mathrm { R o P E _ { P } }$ and $\mathrm { S i P E _ { P } }$ variants with NoPE; the conventional fully rotated controls appear in Fig. 2. The component-periodic scheduler is explicitly written out in the code implementation.

## E.4 Conventional Non-Periodic Settings

Recall in RoPE’s initial presentation that the base value is set to $1 0 ^ { 4 }$ . Prior extrapolation experiments find that this standard choice can be the worst tested base, with improvements on both sides of $1 0 ^ { 4 }$ (Liu et al., 2024). To test whether our conclusions depend on this choice, we therefore sweep bases {10 $^ { - 1 2 } , 1 0 ^ { - 8 } , 1 0 ^ { - 4 } , 1 , 1 0 ^ { 4 } , 1 0 ^ { 8 } , 1 0 ^ { 1 2 } \}$ . Fig. 3 reports every fully rotated $\mathrm { R o P E _ { N P } }$ configuration and compares it with NoPE. Fig. 2 reports fully rotated ${ \mathrm { R o } } { \check { \mathrm { P E } } } _ { \mathrm { N P } }$ and $\mathrm { S i P E _ { N P } }$ configurations across the same sweep for $( a a ) ^ { * }$ and $( a b ) ^ { * }$

## E.5 Detailed Results

Tabs. 4 and 5 report the complete aggregate statistics for the tasks and fully rotated settings shown in Figs. 2 and 3. Each row aggregates 5 seeds and 3 learning rates.

Table 4: Detailed results for the fully rotated conventional base sweep on the modular languages in Fig. 2.
<table><tr><td>Language</td><td>PE</td><td>Base</td><td colspan="3">Accuracy</td><td colspan="3"> $N ^ { * }$ </td></tr><tr><td></td><td></td><td></td><td>Max</td><td>Mean</td><td>Std.</td><td>Max</td><td>Mean</td><td>Std.</td></tr><tr><td> $( a a ) ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { - 1 2 }$ </td><td>0.522</td><td>0.489</td><td>0.0204</td><td>53.0</td><td>41.3</td><td>11.6</td></tr><tr><td> $( a a ) ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { - 8 }$ </td><td>0.500</td><td>0.483</td><td>0.0069</td><td>65.0</td><td>50.1</td><td>14.9</td></tr><tr><td> $( a a ) ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { - 4 }$ </td><td>0.513</td><td>0.501</td><td>0.0061</td><td>57.0</td><td>49.5</td><td>3.1</td></tr><tr><td> $( a a ) ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td>1</td><td>0.509</td><td>0.494</td><td>0.0076</td><td>47.0</td><td>34.3</td><td>17.2</td></tr><tr><td> $( a a ) ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { 4 }$ </td><td>0.507</td><td>0.489</td><td>0.0117</td><td>48.0</td><td>26.3</td><td>21.5</td></tr><tr><td> $( a a ) ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { 8 }$ </td><td>0.515</td><td>0.490</td><td>0.0146</td><td>46.0</td><td>22.7</td><td>21.3</td></tr><tr><td> $( a a ) ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { 1 2 }$ </td><td>0.517</td><td>0.494</td><td>0.0095</td><td>49.0</td><td>23.3</td><td>21.9</td></tr><tr><td> $( a a ) ^ { * }$ </td><td> $\operatorname { S i P E } _ { \mathrm { N P } }$ </td><td> $1 0 ^ { - 1 2 }$ </td><td>0.520</td><td>0.504</td><td>0.0117</td><td>41.0</td><td>16.4</td><td>20.1</td></tr><tr><td> $( a a ) ^ { * }$ </td><td> $\operatorname { S i P E } _ { \mathrm { N P } }$ </td><td> $1 0 ^ { - 8 }$ </td><td>0.654</td><td>0.619</td><td>0.0218</td><td>88.0</td><td>71.3</td><td>11.9</td></tr><tr><td> $( a a ) ^ { * }$ </td><td> $\operatorname { S i P E } _ { \mathrm { N P } }$ </td><td> $1 0 ^ { - 4 }$ </td><td>0.657</td><td>0.623</td><td>0.0273</td><td>90.0</td><td>75.7</td><td>6.3</td></tr><tr><td> $( a a ) ^ { * }$ </td><td> $\operatorname { S i P E } _ { \mathrm { N P } }$ </td><td> $^ 1$ </td><td>0.509</td><td>0.496</td><td>0.0089</td><td>43.0</td><td>8.5</td><td>16.9</td></tr><tr><td> $( a a ) ^ { * }$ </td><td> $\operatorname { S i P E } _ { \mathrm { N P } }$ </td><td> $1 0 ^ { 4 }$ </td><td>0.522</td><td>0.503</td><td>0.0078</td><td>41.0</td><td>2.7</td><td>10.2</td></tr><tr><td> $( a a ) ^ { * }$ </td><td> $\operatorname { S i P E } _ { \mathrm { N P } }$ </td><td> $1 0 ^ { 8 }$ </td><td>0.517</td><td>0.500</td><td>0.0049</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td> $( a a ) ^ { * }$ </td><td> $\operatorname { S i P E } _ { \mathrm { N P } }$ </td><td> $1 0 ^ { 1 2 }$ </td><td>0.515</td><td>0.497</td><td>0.0107</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td> $( a b ) ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { - 1 2 }$ </td><td>0.572</td><td>0.530</td><td>0.0168</td><td>98.0</td><td>65.1</td><td>13.0</td></tr><tr><td> $( a b ) ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { - 8 }$ </td><td>0.520</td><td>0.510</td><td>0.0057</td><td>58.0</td><td>48.9</td><td>5.2</td></tr><tr><td> $( a b ) ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { - 4 }$ </td><td>0.530</td><td>0.514</td><td>0.0060</td><td>68.0</td><td>52.8</td><td>5.6</td></tr><tr><td> $( a b ) ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $^ 1$ </td><td>0.522</td><td>0.509</td><td>0.0041</td><td>56.0</td><td>48.1</td><td>3.1</td></tr><tr><td> $( a b ) ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { 4 }$ </td><td>0.740</td><td>0.560</td><td>0.0602</td><td>96.0</td><td>66.0</td><td>15.7</td></tr><tr><td> $( a b ) ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { 8 }$ </td><td>0.817</td><td>0.566</td><td>0.0896</td><td>98.0</td><td>57.1</td><td>14.7</td></tr><tr><td> $( a b ) ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { 1 2 }$ </td><td>0.703</td><td>0.524</td><td>0.0484</td><td>72.0</td><td>50.1</td><td>6.5</td></tr><tr><td> $( a b ) ^ { * }$ </td><td> $\operatorname { S i P E } _ { \mathrm { N P } }$ </td><td> $1 0 ^ { - 1 2 }$ </td><td>0.515</td><td>0.504</td><td>0.0050</td><td>42.0</td><td>16.8</td><td>20.6</td></tr><tr><td> $( a b ) ^ { * }$ </td><td> $\operatorname { S i P E } _ { \mathrm { N P } }$ </td><td> $1 0 ^ { - 8 }$ </td><td>0.648</td><td>0.536</td><td>0.0416</td><td>54.0</td><td>41.9</td><td>16.7</td></tr><tr><td> $( a b ) ^ { * }$ </td><td> $\operatorname { S i P E } _ { \mathrm { N P } }$ </td><td> $1 0 ^ { - 4 }$ </td><td>0.547</td><td>0.523</td><td>0.0145</td><td>70.0</td><td>52.9</td><td>7.6</td></tr><tr><td> $( a b ) ^ { * }$ </td><td> $\operatorname { S i P E } _ { \mathrm { N P } }$ </td><td>1</td><td>0.515</td><td>0.505</td><td>0.0045</td><td>48.0</td><td>23.1</td><td>21.6</td></tr><tr><td> $( a b ) ^ { * }$ </td><td> $\operatorname { S i P E } _ { \mathrm { N P } }$ </td><td> $1 0 ^ { 4 }$ </td><td>0.515</td><td>0.503</td><td>0.0044</td><td>46.0</td><td>17.1</td><td>20.9</td></tr><tr><td> $( a b ) ^ { * }$ </td><td> $\operatorname { S i P E } _ { \mathrm { N P } }$ </td><td> $1 0 ^ { 8 }$ </td><td>0.559</td><td>0.514</td><td>0.0142</td><td>52.0</td><td>26.3</td><td>21.6</td></tr><tr><td> $( a b ) ^ { * }$ </td><td> $\operatorname { S i P E } _ { \mathrm { N P } }$ </td><td> $1 0 ^ { 1 2 }$ </td><td>0.586</td><td>0.514</td><td>0.0204</td><td>52.0</td><td>24.1</td><td>22.8</td></tr></table>

Table 5: Detailed results for NoPE and the fully rotated conventional RoPE base sweep in Fig. 3.
<table><tr><td>Language</td><td>PE</td><td>Base</td><td colspan="3">Accuracy</td><td colspan="3"> $N ^ { * }$ </td></tr><tr><td></td><td></td><td></td><td>Max</td><td>Mean</td><td>Std.</td><td>Max</td><td>Mean</td><td>Std.</td></tr><tr><td> $\Sigma ^ { * } a$ </td><td>NoPE</td><td></td><td>0.604</td><td>0.564</td><td>0.0237</td><td>60.0</td><td>50.8</td><td>3.9</td></tr><tr><td> $\Sigma ^ { * } a$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { - 1 2 }$ </td><td>0.741</td><td>0.709</td><td>0.0181</td><td>149.0</td><td>95.0</td><td>26.6</td></tr><tr><td> $\Sigma ^ { * } a$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { - 8 }$ </td><td>0.927</td><td>0.829</td><td>0.0898</td><td>116.0</td><td>81.3</td><td>16.8</td></tr><tr><td> $\Sigma ^ { * } a$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { - 4 }$ </td><td>0.913</td><td>0.785</td><td>0.0538</td><td>224.0</td><td>99.8</td><td>46.8</td></tr><tr><td> $\Sigma ^ { * } a$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td>1</td><td>0.753</td><td>0.649</td><td>0.0483</td><td>82.0</td><td>54.8</td><td>9.2</td></tr><tr><td> $\Sigma ^ { * } a$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { 4 }$ </td><td>0.555</td><td>0.524</td><td>0.0152</td><td>51.0</td><td>45.5</td><td>2.3</td></tr><tr><td> $\Sigma ^ { * } a$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { 8 }$ </td><td>0.890</td><td>0.733</td><td>0.1003</td><td>112.0</td><td>65.7</td><td>14.6</td></tr><tr><td> $\Sigma ^ { * } a$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { 1 2 }$ </td><td>0.685</td><td>0.640</td><td>0.0305</td><td>70.0</td><td>58.9</td><td>4.8</td></tr><tr><td> $a \Sigma ^ { * }$ </td><td> $\mathrm { N o P E }$ </td><td></td><td>1.000</td><td>0.989</td><td>0.0413</td><td>500.0</td><td>470.5</td><td>82.0</td></tr><tr><td> $a \Sigma ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { - 1 2 }$ </td><td>1.000</td><td>0.977</td><td>0.0253</td><td>500.0</td><td>199.5</td><td>99.1</td></tr><tr><td> $a \Sigma ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { - 8 }$ </td><td>1.000</td><td>0.927</td><td>0.1033</td><td>500.0</td><td>166.7</td><td>101.4</td></tr><tr><td> $a \Sigma ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { - 4 }$ </td><td>1.000</td><td>0.961</td><td>0.0515</td><td>500.0</td><td>199.1</td><td>95.4</td></tr><tr><td> $a \Sigma ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $^ 1$ </td><td>1.000</td><td>0.953</td><td>0.1005</td><td>500.0</td><td>204.9</td><td>109.5</td></tr><tr><td> $a \Sigma ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { 4 }$ </td><td>1.000</td><td>0.847</td><td>0.1449</td><td>184.0</td><td>146.9</td><td>30.2</td></tr><tr><td> $a \Sigma ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { 8 }$ </td><td>1.000</td><td>0.928</td><td>0.0864</td><td>500.0</td><td>293.3</td><td>121.4</td></tr><tr><td> $a \Sigma ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { 1 2 }$ </td><td>1.000</td><td>0.988</td><td>0.0254</td><td>500.0</td><td>402.9</td><td>125.1</td></tr><tr><td> $\Sigma ^ { * } a b \Sigma ^ { * }$ </td><td> $\mathrm { N o P E }$ </td><td></td><td>0.874</td><td>0.672</td><td>0.0829</td><td>200.0</td><td>92.9</td><td>32.6</td></tr><tr><td> $\Sigma ^ { * } a b \Sigma ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { - 1 2 }$ </td><td>0.770</td><td>0.625</td><td>0.0697</td><td>103.0</td><td>61.9</td><td>16.8</td></tr><tr><td> $\Sigma ^ { * } a b \Sigma ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { - 8 }$ </td><td>0.753</td><td>0.604</td><td>0.0812</td><td>76.0</td><td>58.7</td><td>9.8</td></tr><tr><td> $\Sigma ^ { * } a b \Sigma ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { - 4 }$ </td><td>0.740</td><td>0.606</td><td>0.0556</td><td>73.0</td><td>55.2</td><td>7.4</td></tr><tr><td> $\Sigma ^ { * } a b \Sigma ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td>1</td><td>0.647</td><td>0.624</td><td>0.0221</td><td>51.0</td><td>46.6</td><td>1.5</td></tr><tr><td> $\Sigma ^ { * } a b \Sigma ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { 4 }$ </td><td>0.612</td><td>0.572</td><td>0.0250</td><td>53.0</td><td>49.8</td><td>3.5</td></tr><tr><td> $\Sigma ^ { * } a b \Sigma ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { 8 }$ </td><td>0.624</td><td>0.594</td><td>0.0183</td><td>53.0</td><td>47.2</td><td>2.6</td></tr><tr><td> $\Sigma ^ { * } a b \Sigma ^ { * }$ </td><td> $\mathrm { R o P E _ { N P } }$ </td><td> $1 0 ^ { 1 2 }$ </td><td>0.658</td><td>0.637</td><td>0.0102</td><td>50.0</td><td>46.1</td><td>2.2</td></tr></table>