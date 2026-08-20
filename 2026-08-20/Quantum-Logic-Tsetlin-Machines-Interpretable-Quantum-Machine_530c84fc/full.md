# Quantum-Logic Tsetlin Machines: Interpretable Quantum Machine Learning with Commuting Projector Clauses

Krishna Bhatia

QuantumAI Lab, Fractal Analytics, Mumbai, India

Abstract—Tsetlin Machines learn propositional clauses over Boolean literals using teams of finite-state automata. This paper introduces a quantum-logic analogue, the Quantum-Logic Tsetlin Machine (QL-TM), in which literals are quantum propositions represented by projection operators and clauses are conjunctions inside commuting measurement contexts. The activation of a clause on a quantum state is the Born probability of the corresponding joint projector. This gives a hybrid, interpretable quantum machine-learning model: the automata remain classical, but the learned rules are quantum propositions over states rather than post-measurement bitstrings. We prove that the model reduces to an ordinary Boolean Tsetlin Machine in diagonal contexts, and that Pauli projector clauses recover stabilizer and syndrome semantics. Experiments on Bell states, phase-flip syndrome states, randomized stabilizer-context tasks, mixed literal pools, and finiteshot noisy simulations show that QL projector clauses attain 1.000 accuracy in the exact/noiseless controlled settings, while diagonal or wrong-context baselines lose the relevant phase or syndrome information and remain far below the true context. In 16-class random stabilizer tasks, the true QL context attains 1.000 accuracy for five- and six-qubit systems, while wrong QL contexts remain near the 0.0625 chance level. Mixed-pool and context-budget ablations further show that the learner recovers all true context generators in the presence of distractor projectors, and that accuracy degrades predictably when true quantum propositions are removed. The contribution is not a claim of quantum advantage, but a controlled bridge between Tsetlin clause learning, quantum logic, and interpretable quantum machine learning.

Index Terms—Tsetlin Machine, quantum machine learning, quantum logic, projector clauses, stabilizer formalism, interpretable learning.

## I. INTRODUCTION

Tsetlin Machines (TMs) are logic-based machine-learning models in which classification is expressed through propositional clauses learned by Tsetlin automata [1]. Their appeal is that the learned model is not a dense numerical representation, but a set of human-readable rules such as conjunctions over Boolean literals. This makes TMs attractive in settings where interpretability, discrete learning, and hardware-near implementation matter.

Quantum machine learning (QML), in contrast, usually represents information through quantum states, measurements, kernels, or variational circuits [2]–[5]. These models can exploit quantum state spaces, but their learned objects are often less directly logical. This raises a natural question: if a classical TM learns Boolean propositions, what should a TM-like model learn when the input is a quantum state?

A direct “fully quantum” Tsetlin Machine is difficult to define. Classical TM learning uses irreversible reward and penalty updates, while closed quantum dynamics is unitary unless measurement, feedback, or coupling to an environment is introduced. We therefore take a conservative route: we quantize the logic of the TM rather than the automata memory. In the quantum-logic view of Birkhoff and von Neumann, propositions about a quantum system correspond to closed subspaces of Hilbert space, equivalently projectors [6], [7]. This suggests replacing Boolean literals by projector literals.

The resulting model, the Quantum-Logic Tsetlin Machine (QL-TM), is a hybrid QML model. Its inputs are quantum states, its literals are projective measurement propositions, and its clauses are commuting conjunctions of such propositions. The Tsetlin automata remain classical finite-state learners that include or exclude literals. Fig. 1 summarizes the model. The model is not a quantum Turing machine [8]; the acronym QL TM is used here only for “Quantum-Logic Tsetlin Machine.”

Contributions. This paper makes four contributions:

1) It defines TM clauses over quantum projectors, with Born-probability activations and an explicit commutingcontext restriction.

2) It proves a classical-reduction property: when all projectors are diagonal in a common computational basis, QL-TM clauses reduce exactly to Boolean TM clauses.

3) It connects Pauli-projector clauses to stabilizer and syndrome rules, giving a physical interpretation of learned clauses.

4) It provides controlled experiments showing contextual expressivity, randomized stabilizer generalization, recovery of true projectors among distractors, predictable degradation under context-budget removal, and graceful degradation under finite-shot noise.

The goal is deliberately modest but foundational: define a mathematically controlled quantum-logic generalization of Tsetlin clause learning and test whether the resulting clauses recover physically meaningful quantum propositions.

## II. BACKGROUND AND RELATED WORK

## A. Tsetlin Machines

A classical TM learns a set of clauses over Boolean literals x and ¬x [1]. Each clause is a conjunction of included literals. Teams of Tsetlin automata decide whether each literal is included or excluded, and Type I/Type II feedback drives clauses toward discriminative propositional patterns. The learned representation is therefore symbolic: a classifier can be inspected as a weighted collection of logical rules. Classical TM software ecosystems such as TMU provide scalable implementations and variants [9]; our experimental implementation is independent and minimal because the literal semantics here are quantum projectors rather than ordinary Boolean variables.

![](images/58a048eb2ab39d3a6e5af5b6f617c62e16b9f26525dde780f88d9151ee440fb9.jpg)  
Fig. 1. QL-TM pipeline. A quantum state is evaluated through a library of commuting projector literals. Automata select literals into clauses, each clause produces a Born-probability activation, and class scores are formed from clause votes.

## B. Quantum Logic and Projectors

In quantum mechanics, a proposition such as “observable A has value in set $S ^ { \ast }$ is represented by a projector onto the corresponding subspace. The set of all such propositions is not generally Boolean: projectors need not commute, and the lattice of quantum propositions is generally non-distributive [6], [7]. This is exactly where a QL-TM must differ from a classical TM. An arbitrary conjunction of non-commuting quantum propositions is not a simple Boolean AND. We therefore restrict each clause to a commuting context, where products of projectors are again projectors and the usual conjunction semantics are well defined.

## C. Pauli Projectors and Stabilizers

For a Pauli observable g with eigenvalues ±1, the positive and negative projector literals are

$$
P _ { g } ^ { \pm } = \frac { I \pm g } { 2 } .\tag{1}
$$

If $g _ { 1 } , \ldots , g _ { k }$ commute, the product

$$
C _ { s } = \prod _ { i = 1 } ^ { k } { P _ { g _ { i } } ^ { s _ { i } } } , \quad s _ { i } \in \{ + , - \} ,\tag{2}
$$

projects onto a joint eigenspace. This is the same semantic structure used by stabilizer codes, syndrome measurements, and quantum error correction [10]–[14]. A learned QL-TM clause can therefore be read as a stabilizer-like proposition: it identifies a syndrome pattern, not merely a numerical feature vector.

## D. Relation to Existing QML Models and Research Gap

Variational classifiers and quantum kernels learn continuous circuit/readout parameters or implicit similarities rather than explicit logical propositions [2]–[5]. Our question is different: can TM clause semantics be lifted from Boolean events to quantum propositions while preserving an interpretable include/exclude learner? The baselines therefore separate measurement-context information, projector-feature expressivity, and clause recovery. We do not claim predictive superiority over VQCs or quantum kernels; end-to-end comparisons on less structured stateclassification tasks are future work. Prior quantum-adjacent TM work also does not, to our knowledge, define TM literals themselves as Hilbert-space projectors. In QL-TM, the automata remain classical while the learned clauses become explicit commuting quantum propositions.

## III. QUANTUM-LOGIC TSETLIN MACHINE

Let H be a finite-dimensional Hilbert space and let $\rho$ be a density operator on H. A quantum proposition is represented by a projector $P = P ^ { 2 } = P ^ { \dagger }$ . Its complement is $P ^ { \perp } = I - P .$

Definition 1 (Projector literal). Given a base projector $P _ { i } ,$ a QL-TM literal is either $P _ { i } ,$ its complement $I - P _ { i }$ , or the excluded action ∅.

Definition 2 (Commuting context). A context is a set of projectors $\mathcal { C } = \{ P _ { 1 } , \ldots , P _ { m } \}$ such that $[ P _ { i } , P _ { j } ] = 0$ for all $i , j .$

Definition 3 (Projector clause). A projector clause is a product

$$
C _ { j } = \prod _ { i \in S _ { j } } L _ { j i } , \qquad L _ { j i } \in \{ P _ { i } , I - P _ { i } \} ,\tag{3}
$$

where all selected literals belong to a common commuting context. Its activation on input state $\rho$ is

$$
c _ { j } ( \rho ) = \mathrm { T r } ( \rho C _ { j } ) \in [ 0 , 1 ] .\tag{4}
$$

Thus the binary output of a classical Boolean clause becomes a Born probability. In an exact simulation, $c _ { j } ( \rho )$ can be used directly as a soft clause vote. In an experimental implementation, $c _ { j } ( \rho )$ can be estimated by repeated measurements in the clause context.

For multiclass classification, class $y$ has a signed set of clauses, and the score is

$$
S _ { y } ( \rho ) = \sum _ { j } w _ { y j } c _ { j } ( \rho ) , \qquad w _ { y j } \in \{ - 1 , + 1 \} ,\tag{5}
$$

with prediction $\hat { y } = \arg \operatorname* { m a x } _ { y } S _ { y } ( \rho )$ . In the prototype used here, each class is represented by one positive clause, which is sufficient for the stabilizer-style tasks considered below.

## A. Automata Learning Interface

For each class $y ,$ clause j, and signed projector literal $L _ { i } ,$ the learner stores an automaton $a _ { y j i } \in \{ 1 , \ldots , 2 N \}$ and includes $L _ { i }$ iff $a _ { y j i } > N$ . Signed literals $P _ { g } ^ { + }$ and $P _ { g } ^ { - }$ are represented separately, with conflict resolution preventing both signs of the same observable from remaining in one clause. Each literal has soft truth value

$$
p _ { i } ( \rho ) = \mathrm { T r } ( \rho L _ { i } ) \in [ 0 , 1 ] ,\tag{6}
$$

and the included set $S y j = \{ i : a _ { y j i } > N \}$ defines $C _ { y j } =$ $\prod _ { i \in S _ { y i } } L _ { i } .$

The formal clause activation is the joint Born probability $c _ { y j } ( \rho ) = \mathrm { T r } ( \rho C _ { y j } )$ . The minimal learner uses

$$
\tilde { c } _ { y j } ( \rho ) = \prod _ { i \in S _ { y j } } \mathrm { T r } ( \rho L _ { i } )\tag{7}
$$

only as a lightweight training surrogate. In general $\begin{array} { r } { \operatorname { T r } ( \rho \prod _ { i } L _ { i } ) \neq \prod _ { i } \operatorname { T r } ( \rho L _ { i } ) } \end{array}$ , even for commuting projectors, so no independence assumption is made. For the exact stabilizereigenstate tasks central to our representational claims, relevant signed stabilizer projectors have probabilities in {0, 1}; hence the surrogate and joint activation induce the same binary clause ordering. This equivalence is not guaranteed for finite shot, noisy, or general mixed states, where direct joint-context estimation is a natural extension.

Algorithm 1 states the update used in the experiments. Target class clauses receive Type-I-like feedback: high-probability literals are rewarded by moving their automata toward the include region. Non-target clauses that fire receive Type-II-like feedback: the learner searches for a low-probability literal that suppresses the non-target clause and rewards inclusion of that suppressing literal. Same-observable sign conflicts are resolved after each update.

a) Relation to canonical TM feedback.: The prototype is intentionally TM-style rather than a complete reproduction of production score-gated TM feedback. Target updates depend locally on projector probabilities, non-target suppression is triggered by $\tilde { c } _ { y j }$ , and aggregate class scores are used for prediction. We therefore use “Type-I-like” and “Type-IIlike” deliberately. The semantic change is precise: Boolean satisfaction $x _ { i } \in \{ 0 , 1 \}$ is replaced by projector satisfaction $p _ { i } ( \rho ) ~ = ~ \mathrm { T r } ( \rho L _ { i } ) ~ \in ~ [ 0 , 1 ]$ , while discrete include/exclude automata are retained.

```latex
Algorithm 1 Minimal QL-TM training rule
Require: $\mathcal { D } = \{ ( \rho , y ) \}$ ; literals {L } by context; states $a _ { y j i } ;$ threshold $\tau ;$
epochs E
Ensure: Learned literal sets $S _ { y j }$
1: Initialize automata near the include/exclude boundary.
2: for $e = 1 , \ldots , E$ do
3: for all $( \rho , y ) \in \mathcal { D }$ do
4: Estimate $p _ { i } = \mathrm { T r } ( \rho L _ { i } )$ in the selected contexts.
5: for all classes $y ^ { \prime }$ and clauses j do
6: $S _ { y ^ { \prime } j } \gets \{ i : a _ { y ^ { \prime } j i } > N \}$ , restricted to one context.
7: $\begin{array} { r } { \tilde { c } _ { y ^ { \prime } j } \gets \Pi _ { i \in S _ { y ^ { \prime } j } } p _ { i } } \end{array}$
8: end for
9: for all target clauses $j$ and candidate literals $L _ { i }$ do
10: If $p _ { i } \ \geq \ \tau ,$ move $a _ { y j i }$ toward Include; optionally penalize
low $- p _ { i }$ included literals.
11: end for
12: for all $y ^ { \prime } \ne y$ and firing clauses $j$ with $\tilde { c } _ { y ^ { \prime } j } > \tau$ do
13: Move a compatible low-probability suppressor $a _ { y ^ { \prime } j q }$ toward
Include.
14: end for
15: Resolve $P _ { g } ^ { + } / P _ { g } ^ { - }$ conflicts by retaining the stronger inclusion state.
16: end for
17: end for
18: Predict with $\begin{array} { r } { c _ { y j } ( \rho ) = \mathrm { T r } ( \rho C _ { y j } ) , S _ { y } ( \rho ) = \sum _ { j } w _ { y j } c _ { y j } ( \rho ) , } \end{array}$ , and $\hat { y } =$
arg max<sub>y</sub> $S _ { y } ( \boldsymbol { \rho } ) .$
```

The experiments use ProjectorClauseMiner, a transparent margin-based clause learner, and TinyProjectorTsetlin, the minimal automata prototype above. TinyProjectorTsetlin is not a drop-in replacement for production TM libraries; it tests whether TM-style automata can recover meaningful quantum-logic clauses. Quantum mechanics supplies states and measurement propositions, while classical automata perform clause search; commuting-context clauses can in principle be estimated from compatible measurement counts.

## B. Classical Reduction

Theorem 1 (Diagonal classical reduction). Suppose all base projectors $P _ { i }$ are diagonal in the computational basis and correspond to events $x _ { i } ~ = ~ 1$ for Boolean variables $x _ { i } \in { \mathsf { \Gamma } }$ $\{ 0 , 1 \}$ . For any computational-basis state $| x \rangle$ , every QL-TM clause activation equals the value ofthe corresponding Boolean conjunction:

$$
\mathrm { T r } ( | x \rangle \langle x | C _ { j } ) = \bigwedge _ { i \in S _ { j } } \ell _ { i } ( x ) ,\tag{8}
$$

where $\ell _ { i } ( x )$ is either $x _ { i } \ o r \ \lnot x _ { i }$

Proof. Since all projectors are diagonal, they commute and act as indicator functions on computational-basis states. A product of selected projectors is therefore the product of the corresponding indicators, which is exactly Boolean AND. Complements $I - P _ { i }$ act as negated literals. Hence the QL-TM score reduces to a classical TM score over Boolean literals.

This theorem is important for positioning: QL-TM does not discard the classical TM. It extends its clause semantics from Boolean events to commuting quantum propositions.

![](images/586e31d8fd762b94b9b4bebb8e7520a9ba39dca6eecdf785de8ccb802579cd82.jpg)  
Fig. 2. Minimal QL-TM update used in the experiments. Classical finite-state include/exclude automata are retained while Boolean literal truth values are replaced by projector Born probabilities within commuting contexts.

## C. Pauli-Stabilizer Clause Semantics

Proposition 1 (Stabilizer clause). Let $g _ { 1 } , \ldots , g _ { k }$ be commuting Pauli operators and let $s _ { i } \in \{ + 1 , - 1 \}$ . The clause

$$
C _ { s } = \prod _ { i = 1 } ^ { k } { \frac { I + s _ { i } g _ { i } } { 2 } }\tag{9}
$$

projects onto the joint eigenspace with eigenvalue $s _ { i }$ for each g<sub>i</sub>.

Proof. Each factor is the spectral projector onto the $s _ { i }$ eigenspace of $g _ { i }$ . Because the $g _ { i }$ commute, their spectral projectors commute. The product is therefore the projector onto the intersection of these eigenspaces. □

## IV. EXPERIMENTAL DESIGN

## A. Research Questions

The experiments answer five questions. First, do QL projector contexts give information unavailable to diagonal classical features? Second, can a QL-TM recover known physical clauses such as Bell-state and syndrome propositions? Third, does the result generalize to randomized stabilizer contexts rather than hand-picked examples? Fourth, if the learner is given a mixed pool containing true, wrong, diagonal, and random distractor literals, does it select the true quantum-context generators? Fifth, if some true generators are removed entirely, does performance degrade according to the remaining stabilizer information?

## B. Tasks

Bell task. The four classes are ${ | \Phi ^ { + } \rangle } , { | \Phi ^ { - } \rangle } , { | \Psi ^ { + } \rangle }$ , and $| \Psi ^ { - } \rangle$ The informative QL context contains ZZ and XX projectors. A diagonal baseline only sees Z-basis projectors and cannot distinguish the phase labels.

Phase-flip syndrome task. The four classes are no phase error and single-qubit $Z _ { 1 } , \ Z _ { 2 }$ , or $Z _ { 3 }$ errors on a GHZ-like state. The informative QL context contains $X _ { 1 } X _ { 2 }$ and $X _ { 2 } X _ { 3 }$ projectors. A diagonal $Z$ context is deliberately insufficient.

Random stabilizer tasks. We generate randomized n-qubit, k-generator stabilizer-style tasks. A full-rank binary parity matrix defines k commuting Pauli generators after local basis maps $Z \mapsto X , Y , Z$ . Each syndrome pattern gives one class, so k = 4 yields 16 classes with chance accuracy $1 / 1 6 = 0 . 0 6 2 5$ We compare the true QL context, a diagonal Z context, and a wrong non-diagonal QL context.

Mixed literal-pool task. We form a larger candidate pool containing true context literals, wrong-context literals, diagonal Z literals, and random Pauli distractors. This tests whether success depends on handing the learner only the correct context.

Context-budget ablation. To test whether QL-TM succeeds only because a complete true context is supplied, we also consider partial-context tasks. For a k-generator stabilizer problem, only $b \leq k$ true generators are made available. If only b true generators are present, the available projector features can distinguish at most $2 ^ { b }$ syndrome groups out of $2 ^ { k }$ classes. In a balanced task, the expected accuracy therefore follows the ladder $2 ^ { b - k }$ . This ablation checks whether performance degrades predictably when physically necessary quantum propositions are removed, rather than remaining perfect merely because the feature space is enlarged.

NOTATION AND EXPERIMENTAL QUANTITIES.  
TABLE I
<table><tr><td>Symbol/quantity</td><td>Meaning</td></tr><tr><td> $P _ { g } ^ { \pm }$ </td><td>Pauli projector  $( I \pm g ) / 2$ </td></tr><tr><td> $\overset { \mathcal { Y } } { C _ { j } }$ </td><td>Commuting projector clause</td></tr><tr><td> $c _ { j } ( \rho )$ </td><td>Clause activation  $\operatorname { T r } ( \rho C _ { j } )$ </td></tr><tr><td> $k$ </td><td>Number of stabilizer generators/classes  $2 ^ { k }$ </td></tr><tr><td>True coverage</td><td>Fraction of true generators selected</td></tr><tr><td>Selected lit.</td><td>Average literals per learned class clause</td></tr><tr><td>Wrong QL</td><td>Non-diagonal projectors in the wrong basis</td></tr></table>

## C. Noise and Sampling

For a pure state $\rho ,$ depolarizing noise is simulated as

$$
\rho _ { p } = ( 1 - p ) \rho + p I / d ,\tag{10}
$$

where $d \ = \ 2 ^ { n }$ . We also simulate finite-shot estimates by binomial sampling of Born probabilities, readout noise by flipping binary outcomes with probability r, and coherent local perturbations by applying small random single-qubit rotations before measurement.

## D. Baselines and Metrics

We report four models: ProjectorClauseMiner, TinyProjectorTsetlin, a nearest-prototype baseline, and a dependency-free ridge one-vs-rest classifier. The baselines separate projectorfeature expressivity from a specific clause learner. We report accuracy, standard deviation across seeds/tasks, average literals per clause, true-generator coverage, and distractor literals selected.

a) Statistical reporting.: Means and standard deviations are reported over the seeds/tasks in Table II. We do not claim significance for small baseline differences; the principal claims concern large context separations, true-generator recovery, and the predicted $2 ^ { b - k }$ context-budget trend. Formal paired significance testing is left to larger future studies.

Table I summarizes the notation used in the experiments and analysis. Exact/noiseless rows use Born probabilities directly. Finite-shot rows replace each probability by an empirical frequency from repeated Bernoulli trials, matching the way a commuting Pauli projector would be estimated from measurement counts.

b) Hyperparameter policy.: Hyperparameters in Table II were fixed within experiment families and were not tuned against the reported test sets. We use $N = 1 6 ,$ a common $\tau = 0 . 5 5 ,$ , and short fixed epoch budgets because the goal is controlled clause recovery rather than optimized predictive performance. A systematic sensitivity analysis over $N , \tau ,$ , clause multiplicity, and feedback schedules remains future work.

Consequently, the reported results should be interpreted as evidence for the proposed clause semantics under a fixed proto type configuration, not as evidence that these hyperparameters are optimal. A systematic sensitivity analysis over $N , \tau ,$ clause multiplicity, and feedback schedules remains future work.

Contextual projector features  
![](images/0537fb455227d6afd9d4c90d7e005ee21ed2822576a07f455f0c0a0b865663f2.jpg)  
Fig. 3. Context advantage for TinyProjectorTsetlin. Non-diagonal QL contexts solve the tasks, while diagonal Z-only contexts lose the relevant phase/syndrome information.

## V. RESULTS

## A. Contextual Expressivity

Table III and Fig. 3 show the basic separation. In the exact/noiseless setting, QL projector contexts attain 1.000 mean accuracy on the Bell and phase-flip tasks across all four evaluated models. Diagonal Z-only contexts collapse: Bell accuracy is about 0.50 because the diagonal context cannot distinguish the + and − Bell phases, and phase-flip accuracy is about 0.20–0.24 because the relevant syndrome is in an $X$ context.

The learned clauses are also physically meaningful. For the Bell task, both clause learners recover

$$
\begin{array} { c c } { { | \Phi ^ { + } \rangle : P _ { Z Z } ^ { + } \wedge P _ { X X } ^ { + } , ~ } } & { { | \Phi ^ { - } \rangle : P _ { Z Z } ^ { + } \wedge P _ { X X } ^ { - } , } } \\ { { | \Psi ^ { + } \rangle : P _ { Z Z } ^ { - } \wedge P _ { X X } ^ { + } , ~ } } & { { | \Psi ^ { - } \rangle : P _ { Z Z } ^ { - } \wedge P _ { X X } ^ { - } . } } \end{array}
$$

For phase-flip syndromes, the recovered clauses are

$$
\begin{array} { r l r } & { } & { \mathrm { N o n e : ~ } P _ { X _ { 1 } X _ { 2 } } ^ { + } \wedge P _ { X _ { 2 } X _ { 3 } } ^ { + } , \qquad Z _ { 1 } : P _ { X _ { 1 } X _ { 2 } } ^ { - } \wedge P _ { X _ { 2 } X _ { 3 } } ^ { + } , } \\ & { } & { Z _ { 2 } : P _ { X _ { 1 } X _ { 2 } } ^ { - } \wedge P _ { X _ { 2 } X _ { 3 } } ^ { - } , \qquad Z _ { 3 } : P _ { X _ { 1 } X _ { 2 } } ^ { + } \wedge P _ { X _ { 2 } X _ { 3 } } ^ { - } . } \end{array}
$$

The corresponding diagonal clauses either merge Bell phase pairs or degenerate to TRUE for all phase-flip classes.

## B. Randomized Stabilizer Generalization

Fig. 4 and Table IV show the stronger randomized task. For $k = 4$ , the classifier must distinguish 16 syndrome classes. The true QL context achieves 1.000 accuracy for $n = 5$ and $n = 6 ,$ , while wrong QL contexts remain near chance. Diagonal Z contexts sometimes extract weak accidental information but remain far below the true context.

The representative random task clauses have the form

$$
\begin{array} { r l } { s _ { \sigma _ { 1 } \sigma _ { 2 } \sigma _ { 3 } \sigma _ { 4 } } : } & { { } P _ { g _ { 1 } } ^ { \sigma _ { 1 } } \wedge P _ { g _ { 2 } } ^ { \sigma _ { 2 } } \wedge P _ { g _ { 3 } } ^ { \sigma _ { 3 } } \wedge P _ { g _ { 4 } } ^ { \sigma _ { 4 } } , } \end{array}\tag{11}
$$

which is exactly the expected stabilizer-syndrome proposition. This supports the claim that QL-TM is learning quantumcontext clauses, not memorizing Bell-state labels.

TABLE II  
EXPERIMENTAL PROTOCOL AND HYPERPARAMETERS. REPORTED STANDARD DEVIATIONS ARE COMPUTED OVER THE SEEDS AND RANDOM TASKS USED BY THE CORRESPONDING EXPERIMENT.  
```csv
Quantity Value
Train/test split Random stratified 70/30 train/test split using the package routine train_test_split; fixed seed per run
Bell/phase-flip context experiment 80 samples per class, 10 seeds for feature-baseline results
Random stabilizer experiment n $\in \{ \bar { 5 , 6 } \} , \mathsf { \dot { k } } = 4$ generators, 16 classes, 6 random tasks, 3 seeds, 50 samples per class
Mixed literal-pool experiment n ∈ $\{ 5 , 6 \} , k = 4 ,$ 6 random tasks, 3 seeds, 60 samples per class
Context-budget ablation n ∈ $\{ 5 , 6 \} , k = 4 ,$ 6 random tasks, 3 seeds, 60 samples per class; all subsets for each true-generator budget b ∈ {0, 1, 2, 3, 4}
Finite-shot/noise grid Depolarizing p ∈ {0, 0.05, 0.1, 0.2, 0.4, 0.6}; shots ∈ {exact, 16, 32, 64, 128, 256}
Additional noise Coherent rotation scale ∈ {0, 0.1, 0.2, 0.3}; readout noise ∈ {0, 0.02, 0.05}
ProjectorClauseMiner Minimum margin 0.03; one positive clause per class in reported stabilizer experiments
TinyProjectorTsetlin Automaton states parameter ${ \dot { N } } = 1 6 ;$ threshold $\tau = 0 . 5 5 ; $ epochs for context/noise runs and 10 epochs for random/mixed stabilizer runs
Mixed-pool composition True-context literals, wrong-context literals, diagonal Z literals, and 32 random Pauli-observable distractors, each with ± projectors
Wrong QL context Non-diagonal Pauli-projector context generated in an incompatible/random basis, excluding the true stabilizer generators
Implementation NumPy, pandas, matplotlib; no dependency on TMU or pyTsetlinMachine
```

TABLE III  
CONTEXT ADVANTAGE ACROSS MODELS. VALUES ARE MEAN ± STANDARD DEVIATION.
<table><tr><td>Model</td><td>Bell QL</td><td>Bell Z</td><td>Phase QL</td><td>Phase Z</td></tr><tr><td>ProjectorClauseMiner</td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 5 0 0 { \scriptstyle \pm 0 . 0 5 5 }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 2 3 9 { \pm } 0 . 0 2 8$ </td></tr><tr><td>TinyProjectorTsetlin</td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 5 0 0 { \scriptstyle \pm 0 . 0 5 5 }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 2 3 9 { \pm } 0 . 0 2 8$ </td></tr><tr><td>RidgeOneVsRest</td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 4 4 9 { \pm } 0 . 0 3 2$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 2 0 1 { \scriptstyle \pm 0 . 0 2 0 }$ </td></tr><tr><td>NearestPrototype</td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 5 0 0 { \scriptstyle \pm 0 . 0 5 5 }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 2 3 9 { \pm } 0 . 0 2 8$ </td></tr></table>

![](images/5e695ae73f591eed82425bb98c89aed79713f589463144bb0fa434a2dff3c6ee.jpg)  
Fig. 4. Random 16-class stabilizer tasks using TinyProjectorTsetlin. The correct QL context solves the task; wrong non-diagonal contexts remain at the chance level.

## C. Mixed Literal Pools and Pruning

The mixed-pool experiment asks whether the learner can handle a candidate library that contains true, wrong, diagonal, and random projector literals. Table V shows that the mixed pool retains perfect accuracy for both $n = 5$ and n = 6 while achieving full true-generator coverage. When the true literals are removed, performance becomes unstable and substantially lower. Wrong contexts remain near chance.

The mixed-pool learner sometimes retains redundant distractors in addition to the four true literals. Fig. 5 reports a small post-hoc pruning ablation: when redundant nontrue literals are removed while retaining the true generators, clauses compress back to the $k \ = \ 4$ stabilizer form with unchanged exact/noiseless accuracy. This suggests that explicit sparsification should be part of a future full-scale QL-TM implementation.

RANDOM STABILIZER RESULTS FOR TINYPROJECTORTSETLIN, EXACT/NOISELESS SETTING, $k = 4 { \mathrm { ~ G E N E R A T O R S } } .$  
TABLE IV
<table><tr><td>n</td><td>Feature set</td><td>Accuracy</td><td>Lit./clause</td></tr><tr><td>5</td><td>True QL</td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>4.00</td></tr><tr><td>5</td><td>Diagonal Z</td><td> $0 . 1 5 0 { \scriptstyle \pm 0 . 0 5 2 }$ </td><td>2.50</td></tr><tr><td>5</td><td>Wrong QL</td><td> $0 . 0 6 2 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td>0.00</td></tr><tr><td>6</td><td>True QL</td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>4.00</td></tr><tr><td>6</td><td>Diagonal Z</td><td> $0 . 1 3 0 { \scriptstyle \pm 0 . 0 6 4 }$ </td><td>1.17</td></tr><tr><td>6</td><td>Wrong QL</td><td> $0 . 0 6 2 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td>0.00</td></tr></table>

TABLE V

MIXED-POOL RESULTS FOR TINYPROJECTORTSETLIN, EXACT/NOISELESS SETTING, $k = 4$ . COVERAGE IS THE FRACTION OF TRUE GENERATORS SELECTED.
<table><tr><td>n</td><td>Feature pool</td><td>Accuracy</td><td>Selected lit.</td><td>Coverage</td></tr><tr><td>5</td><td>True only</td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>4.00</td><td>1.00</td></tr><tr><td>5</td><td>Mixed pool</td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>7.00</td><td>1.00</td></tr><tr><td>5</td><td>Mixed w/o true</td><td> $0 . 4 1 9 { \pm } 0 . 4 2 7$ </td><td>3.00</td><td>0.00</td></tr><tr><td>5</td><td>Diagonal Z</td><td> $0 . 1 8 6 { \pm } 0 . 1 5 5$ </td><td>2.00</td><td>0.00</td></tr><tr><td>5</td><td>Wrong QL</td><td> $0 . 0 4 9 { \pm } 0 . 0 1 1$ </td><td>0.00</td><td>0.00</td></tr><tr><td>6</td><td>True only</td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>4.00</td><td>1.00</td></tr><tr><td>6</td><td>Mixed pool</td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>7.83</td><td>1.00</td></tr><tr><td>6</td><td>Mixed w/o true</td><td> $0 . 3 5 1 { \scriptstyle \pm 0 . 3 3 1 }$ </td><td>3.83</td><td>0.00</td></tr><tr><td>6</td><td>Diagonal Z</td><td> $0 . 1 7 9 { \scriptstyle \pm 0 . 0 6 7 }$ </td><td>3.33</td><td>0.00</td></tr><tr><td>6</td><td>Wrong QL</td><td> $0 . 0 4 9 { \pm } 0 . 0 1 1$ </td><td>0.00</td><td>0.00</td></tr></table>

The next ablation asks the complementary question: if some true generators are removed entirely rather than merely hidden among distractors, does performance degrade according to the remaining syndrome information?

## D. Context-Budget Ablation

The mixed-pool experiment shows that QL-TM can recover the true generators when they are present among distractors. A stricter question is whether performance degrades when some of the true quantum propositions are absent. Table VI reports the context-budget ablation for $k = 4 .$ , where only b of the four true stabilizer generators are available. The expected separability ladder is $2 ^ { b - 4 } \colon$ : with $b = 0$ the task is at the 16-class chance level, while $b = 4$ restores full separability.

![](images/eda2d90bc70af3919e43a12f838c68607de10aac2147b707ddae834af155e5d3.jpg)  
Fig. 5. Post-hoc mixed-pool pruning. Mixed-pool clauses include all true stabilizer generators plus redundant distractors; pruning compresses clauses back to four literals while preserving accuracy in the exact/noiseless task.

TABLE VI  
CONTEXT-BUDGET ABLATION FOR TINYPROJECTORTSETLIN, EXACT/NOISELESS SETTING, $k = 4 .$ . VALUES ARE MEAN ± STANDARD DEVIATION OVER 6 RANDOM TASKS, 3 SEEDS, AND ALL TRUE-GENERATOR SUBSETS AT EACH BUDGET.
<table><tr><td>Available true generators  $b / 4$ </td><td>Expected</td><td> $n = 5 { \mathrm { ~ a c c . } }$ </td><td>n = 6 acc.</td></tr><tr><td>0/4</td><td>0.0625</td><td>0.049±0.011</td><td>0.049±0.011</td></tr><tr><td>1/4</td><td>0.1250</td><td>0.113±0.013</td><td>0.113±0.013</td></tr><tr><td>2/4</td><td>0.2500</td><td>0.240±0.023</td><td>0.240±0.023</td></tr><tr><td>3/4</td><td>0.5000</td><td>0.495±0.027</td><td>0.495±0.027</td></tr><tr><td>4/4</td><td>1.0000</td><td>1.000±0.000</td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td></tr></table>

The observed accuracies closely track the predicted $2 ^ { b - k }$ separability ladder: near chance with no true generator, approximately 0.125, 0.25, and 0.5 with one, two, and three generators, and 1.000 with all four. Because this hierarchy is determined by the available stabilizer information before learning, it provides controlled evidence that performance is governed by physically meaningful projector propositions rather than raw feature dimensionality.

## E. Finite-Shot and Noise Robustness

The robustness grid varies depolarizing noise, finite shots, coherent local rotations, and readout noise. The most severe plotted slice uses depolarizing noise $p = 0 . 6$ , rotation scale 0.2, and readout noise 0.02. In this setting, TinyProjectorTsetlin still obtains 0.973 accuracy for Bell states at only 16 shots and 0.944 for phase-flip syndromes; by 64 shots both tasks recover 1.000 accuracy. Fig. 6 shows this behavior. The reason exact depolarizing noise is mild for these tasks is that it shrinks confidence without changing the stabilizer syndrome ordering; degradation appears when finite-shot and readout uncertainty make the estimated projector probabilities ambiguous.

![](images/b6c7dff0d15350730be470b12087747c24a77f575e2550ec182a88307fb23a89.jpg)  
Fig. 6. Finite-shot robustness under a harsh setting: depolarizing noise $p = 0 . 6$ coherent rotation scale 0.2, and readout noise 0.02. Accuracy degrades mainly at very low shot counts.

## VI. DISCUSSION

Three scoped conclusions follow. First, the diagonalreduction theorem identifies ordinary Boolean TM semantics as a special case, while the Bell, phase-flip, and randomized stabilizer experiments show that non-diagonal commuting contexts expose phase/syndrome information absent from diagonal features. Second, learned clauses have direct jointeigenspace interpretations; mixed-pool experiments recover all true generators among distractors, although redundant literals motivate native sparsification. Third, the context-budget results follow the externally predicted $2 ^ { b - k }$ hierarchy as true generators are removed, providing the strongest evidence that performance tracks quantum-context information rather than feature count.

The scope remains deliberately narrow. The tasks are Bell/GHZ/stabilizer based because their correct projector propositions are analytically known; they do not establish superiority over VQCs, quantum kernels, or other end-to-end QML models. Likewise, the product-of-marginals quantity is only a training surrogate; exact joint-context learning is not yet validated beyond the deterministic stabilizer setting.

## A. Measurement and Limitations

Because each clause lies in one commuting context, its literals can in principle be estimated from compatible measurements, suggesting a future measurement-cost objective based on contexts and shots. The present study, however, is entirely simulated and does not model hardware calibration drift or device-specific measurement constraints. TinyProjectorTsetlin is a minimal prototype rather than a full score-gated production TM, no task-specific hyperparameter sweep or formal significance-testing campaign is reported, and less structured state families remain unevaluated. Future work should therefore study direct joint-projector feedback, context selection, multiclause learning, native sparsity, hardware-derived data, and comparisons on broader QML tasks.

## VII. CONCLUSION

We introduced QL-TM, a hybrid quantum-state classifier whose Tsetlin-style literals are projectors and whose clauses are restricted to commuting contexts. The framework reduces exactly to Boolean TM clause semantics in diagonal contexts and yields stabilizer/syndrome semantics for Pauli projectors. Controlled experiments demonstrate contextual expressivity, randomized stabilizer generalization, true-literal recovery from mixed pools, the predicted context-budget degradation, and finite-shot robustness. These results establish a foundation for quantum-logic Tsetlin learning without claiming quantum advantage or state-of-the-art end-to-end QML performance.

## REPRODUCIBILITY NOTE

Source code, raw results, figures, and learned clauses are available at

https://github.com/FraQTech/quantum-logic-tsetlin-machines. Table II records the data-generation settings, seeds, sample counts, and fixed hyperparameters used in the reported experiments.

## REFERENCES

[1] O.-C. Granmo, “The tsetlin machine: A game theoretic bandit driven approach to optimal pattern recognition with propositional logic,” IEEE Access, vol. 6, pp. 78 387–78 407, 2018.

[2] J. Biamonte, P. Wittek, N. Pancotti, P. Rebentrost, N. Wiebe, and S. Lloyd, “Quantum machine learning,” Nature, vol. 549, pp. 195–202, 2017.

[3] M. Schuld and N. Killoran, “Quantum machine learning in feature hilbert spaces,” Physical Review Letters, vol. 122, no. 4, p. 040504, 2019.

[4] V. Havlícek, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala, J. M.ˇ Chow, and J. M. Gambetta, “Supervised learning with quantum-enhanced feature spaces,” Nature, vol. 567, pp. 209–212, 2019.

[5] M. Cerezo, A. Arrasmith, R. Babbush, S. C. Benjamin, S. Endo, K. Fujii, J. R. McClean, K. Mitarai, X. Yuan, L. Cincio, and P. J. Coles, “Variational quantum algorithms,” Nature Reviews Physics, vol. 3, pp. 625–644, 2021.

[6] G. Birkhoff and J. von Neumann, “The logic of quantum mechanics,” Annals of Mathematics, vol. 37, no. 4, pp. 823–843, 1936.

[7] A. Wilce, “Quantum logic and probability theory,” The Stanford Encyclo pedia of Philosophy, 2021, available: https://plato.stanford.edu/entries/qt quantlog/.

[8] D. Deutsch, “Quantum theory, the church–turing principle and the universal quantum computer,” Proceedings ofthe Royal Society ofLondon. A. Mathematical and Physical Sciences, vol. 400, no. 1818, pp. 97–117, 1985.

[9] CAIR, “Tsetlin machine unified (tmu): One codebase to rule them all,” GitHub repository, 2025, available: https://github.com/cair/tmu.

[10] M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum Information, 10th ed. Cambridge University Press, 2010.

[11] D. Gottesman, “Stabilizer codes and quantum error correction,” Ph.D. dissertation, California Institute of Technology, 1997, arXiv:quantph/9705052.

[12] S. Aaronson and D. Gottesman, “Improved simulation of stabilizer circuits,” Physical Review A, vol. 70, no. 5, p. 052328, 2004.

[13] A. R. Calderbank and P. W. Shor, “Good quantum error-correcting codes exist,” Physical Review A, vol. 54, no. 2, pp. 1098–1105, 1996.

[14] A. M. Steane, “Error correcting codes in quantum theory,” Physical Review Letters, vol. 77, no. 5, pp. 793–797, 1996.