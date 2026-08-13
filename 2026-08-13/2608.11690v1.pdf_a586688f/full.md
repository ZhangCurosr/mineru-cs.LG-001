# Drift and Dependence: Layer-wise Information-Theoretic Bounds for Replay-Based Continual Learning

Tieliang Gong, Zhongbo Zhang, Wen Wen and Yong-Jin Liu, Senior Member, IEEE

## Abstract

Continual learning must absorb new tasks without erasing old ones, and replay—mixing a small buffer of past examples into current training—is among the most effective remedies for catastrophic forgetting. Yet its generalization behavior is shaped by two coupled effects that existing analyses fold into a single hypothesis-level quantity: finite memory replaces each past distribution with an empirical proxy, and repeated reuse couples the buffer, the current data, and the final hypothesis through a shared optimization trajectory. We develop a layer-wise information-theoretic framework that separates these effects at every depth. Our main result decomposes the expected generalization gap into a replay-induced representation drift and an optimization-dependence term, the latter further resolved into stability, plasticity, interaction, and residual-coupling components. Two refinements make the framework operational. A Wasserstein relaxation of the drift term, valid under support mismatch, yields a depth-dependent drift–sensitivity trade-off whose minimizer identifies which interior layer to stabilize. An SGLD instantiation of the optimization term reduces it to a trajectory-level log-determinant budget, exposing a curvature-aware gradient-alignment statistic that serves as an online diagnostic of task-wise forgetting. Controlled and benchmark experiments confirm the predicted memory scaling, the interior funnel, and the alignment signal’s link to forgetting.

## Index Terms

Continual learning, experience replay, information theory, Wasserstein distance, stochastic gradient Langevin dynamics.

## I. INTRODUCTION

Continual learning (CL) studies how a model can incrementally learn a sequence of tasks while retaining knowledge acquired from previous ones [1]–[3]. A central obstacle in CL is catastrophic forgetting: when the model adapts to a new task, its performance on earlier tasks may deteriorate significantly [4], [5]. In recent years, extensive efforts have been made to mitigate catastrophic forgetting [6], [7]. Among them, replay-based methods [8]–[10] have emerged as one of the most successful practical mechanisms, where the learner stores a small buffer of past examples and mixes them with the current task during training.

Despite this success, the generalization behavior of replay-based CL remains insufficiently understood since replay changes the statistical object being optimized. The learner never minimizes risk against the past-task populations. It minimizes risk against finite proxies of them, and does so repeatedly as the model continues to evolve. This substitution acts in two channels First, replacing a population by a finite stored subset produces a proxy bias, say, the empirical objective on the buffer is a biased stand-in for the population objective, and the bias does not disappear merely because the data are i.i.d. Second, reusing the same stored samples across many updates produces a dependence coupling among the old samples, the current samples, and the final hypothesis, since all three are tied together through a shared optimization trajectory. Note that neither channel presupposes that the model learns a representation, and both are already present when replay is performed directly on raw inputs.

What a deep network adds is not a third channel but a change in how these two channels behave with depth, and it does so asymmetrically. Because the buffer influences later tasks only through a learned feature map, and because that map is fitted to the particular stored exemplars, the proxy bias is no longer fixed: it is generated and reshaped at every layer, and a discrepancy that is negligible in input space can be amplified by the sub-network above it. The dependence coupling, by contrast, is set by the algorithm’s reuse of data and survives even when no representation is learned. The learned representation therefore turns the proxy bias into a depth-dependent object while leaving the coupling channel essentially intact. As the boundary case of our bound make precise, the proxy-bias channel vanishes at the input layer and grows with depth, whereas the coupling channel survives even there.

Existing theory does not separate these channels. Analyses based on replay losses, trade-off parameters, or domain-adaptation arguments [11], [12] blend the intrinsic difficulty of learning from the full stream with the approximation cost of compressing it into a buffer, so the finite capacity loss and the reuse-induced dependence never surface as distinct quantities. Classic complexity bounds fare no better: their capacity terms grow far faster than the sample size, leaving them numerically vacuous in precisely the overparameterized regime of interest. Stability and convergence analyses under smoothness or Lipschitz assumptions [13], [14] take the opposite tack—they track optimization dynamics closely but say little about how buffer size, sequence length, depth, and algorithmic dependence jointly shape the gap. The information-theoretic framework comes closest: it relates the learned parameters to the data through mutual information and its refinements [15]–[17] and recent work carries this analysis across the layers of a deep network [18]. This is our starting point, but replay changes the objects the framework must measure, so it cannot be applied off the shelf. The relevant quantities are no longer parameter–data dependencies but buffer-induced representation distributions, and the coupling created by reuse and by sampling without replacement is intrinsic to the problem rather than a nuisance term: it must be built into the analysis from the outset. What makes a layer-wise treatment unavoidable is the asymmetry between the two channels. Because the finite-memory channel is reshaped at every layer while the coupling channel is not, a black-box, hypothesis-level analysis averages over exactly the depth structure the buffer induces, collapsing the very quantity we set out to resolve—the depth at which the buffer’s finite-sample bias is least amplified by the network above it. We therefore introduce two layer-wise objects: per-task replay centroids, which make the finite-memory discrepancy well defined at every depth, and a decoupled reference, which isolates the dependence due to reuse. Together these let finite-memory bias and optimization coupling sit in a single decomposition—something hypothesis-level information-theoretic bounds cannot express.

There are two difficulties in carrying this out rigorously. The first is circularity: the feature map at a given depth is learned online and shifts with whichever examples enter the buffer, so the map we use to measure replay discrepancy is shaped by the very buffer we are measuring. The second is accumulation: the dependence entering a new task is not generated fresh but inherited from earlier tasks and built up along the training trajectory. Both must be handled jointly and at an arbitrary depth, not just at the input or output. This is what the paper’s three constructions provide: a replay centroid that makes the finite-memory discrepancy well-defined at every layer and separates it from the dependence term; a Wasserstein relaxation of that discrepancy for the regime where its information-theoretic form is ill-posed under support mismatch; and a trajectory-leve decomposition that, for a concrete noisy optimizer, splits the accumulated dependence into per-step contributions.

At a high level, our analysis separates replay generalization into two coupled effects: a replay-induced drift term, arising from the finite-memory approximation of past task distributions, and an optimization-dependence term, arising from the reuse of replayed samples through shared downstream parameters. The former quantifies the mismatch between the replay buffer and the corresponding past-task population in representation space, while the latter captures how strongly the learned suffix parameters encode information about both old and current task representations. This decomposition cleanly distinguishes the statistical cost of memory compression from the dependence created by repeated optimization. We then further refine the drift branch with a Wasserstein relaxation when the KL-based form becomes uninformative under support mismatch, and instantiate the dependence branch for stochastic gradient Langevin dynamics (SGLD). The resulting quantities are intended as analytic diagnostics and control signals for replay-based training, rather than as a replacement for existing continual learning algorithms.

In summary, our contributions are as follows:

• We develop a layer-wise information-theoretic bound for replay-based continual learning that explicitly separates replayinduced drift from optimization dependence. The bound further clarifies how old-task dependence, current-task dependence, their interaction, and residual coupling jointly shape the expected generalization gap.

• When the KL-based drift term becomes vacuous under empirical-population support mismatch, we derive a Wasserstein relaxation that yields a depth-dependent drift–sensitivity trade-off and identifies an interior generalization-funnel layer, indicating which layer to stabilize.

• We instantiate the optimization-dependence branch with SGLD and obtain a trajectory-level log-determinant budget. Thi leads to sensitivity-aware gradient-alignment diagnostics and a local replay–current mixing rule.

• We validate the theory through controlled and benchmark experiments, confirming the predicted finite-memory scaling, funnel behavior, and alignment patterns in replay dynamics.

## II. RELATED WORK

## A. Replay-based continual learning

Catastrophic forgetting under sequential training has long been recognized as a central obstacle for neural networks [4], [19], and modern CL methods are commonly grouped into replay-based, regularization-based, and architectural/parameterisolation approaches [2], [20], [21]. Regularization methods constrain updates through parameter-importance penalties such as Fisher information and synapse-importance surrogates [1], [22], [23], while architectural strategies expand or partition parameters across tasks [24], [25]. Replay remains a strong and competitive option because it directly counteracts forgetting by revisiting stored examples, through exemplar rehearsal [9], generic experience replay [8], and generative replay [26]. Modern replay systems differ mainly in how they allocate memory and select exemplars: a typical choice is reservoir sampling [27], while more informed strategies prioritize examples that maximize future utility or reduce interference [28], and further work corrects the representation drift and class imbalance that erode replay in online CL [29]. Replay also connects to constrainedoptimization views that explicitly control gradient interference [30], including GEM [31], its averaged variant A-GEM [32], and orthogonal projection [33]. These methods align with the view taken here, that replay replaces past populations with finite proxy distributions while evolving representations cause the buffer to shift within feature space. Our goal is different from any single update rule: we begin from a layer-wise decomposition of replay generalization that separates distributional drift from optimization coupling, and that explains when replay acts as a stable rehearsal mechanism and when it becomes a biased or strongly coupled proxy for the past.

## B. Information-theoretic analysis

Recent theoretical work studies forgetting, task order, and generalization in sequential learning through statistical or optimization lenses. Closest to us are information-theoretic and statistical analyses of forgetting [34], [35], which operate in linear or overparameterized regimes and characterize how task similarity and ordering affect generalization. For the replay setting in particular, information-theoretic bounds have been derived from the dependence between the hypothesis and the memory buffer [36], separating an ideal full-data risk from a memory-compression cost. Because these bounds are read at the level of the whole hypothesis, their finite-memory term stays a raw-sample compression cost that does not resolve the representation-level drift or the depth at which the gap arises. Our contribution is different in kind: we retain learned hierarchical representations and expose layer-dependent quantities $( \mathcal { K } ^ { ( l ) } , \mathcal { S } ^ { ( l ) } , \mathcal { P } ^ { ( l ) } , \mathcal { R } ^ { ( l ) } )$ that such hypothesis- or input-layer analysis cannot see. We do not claim a tighter constant; we claim a finer-grained decomposition. More broadly, the information-theoretic framework relates learned parameters to data via mutual information [15] and its conditional refinements [16], [17], and has been specialized to the trajectory of noisy iterative algorithms such as SGLD [37], [38]; the dependence of generalization on network depth has also been studied directly [18], [39]. Our main theorem combines these perspectives for replay: it keeps the information-theoretic dependence structure, but introduces replay centroids and layer-wise old/new task representations so that finite replay bias and optimization coupling appear in the same decomposition.

## C. Geometry and gradient interaction

Wasserstein and optimal-transport arguments provide finite notions of mismatch when KL-type divergences are unstable under support shift, both as a general geometric tool [40] and in domain-adaptation and generalization bounds that control risk through a transport discrepancy [41]–[43]; these works, however, remain confined to single-task learning. We use this geometry only to relax the drift branch of our main decomposition. Conversely, our SGLD analysis targets the optimization branch and yields gradient-moment diagnostics. This distinguishes our alignment quantities from Euclidean projection rules in GEM/OGD-style methods: here alignment is derived as a bound-driven, sensitivity-aware diagnostic of optimization coupling rather than introduced as a standalone projection heuristic.

## III. PROBLEM SETUP AND LAYER-WISE REPLAY DISTRIBUTIONS

Notations. Throughout this paper, we denote random variables by uppercase letters (e.g., X), their realizations by lowercase letters $( \mathrm { e } . \mathrm { g } . , x )$ , and their domains by calligraphic letters $( \mathrm { e . g . , } \mathcal { X } )$ . The distribution of X is denoted $P _ { X }$ . Expectation, variance, and covariance are written E[X], $\operatorname { V a r } ( X )$ , and $\operatorname { C o v } ( X )$ , respectively. For conditional distributions, we write $P _ { X \mid Y = y } .$ We use the standard information-theoretic quantities: entropy $H ( X )$ , mutual information $I ( X ; Y )$ , Kullback-Leibler (KL) divergence $D _ { \mathrm { K L } } ( P \Vert Q )$ , and the $p \textmd { - }$ Wasserstein distance $\mathcal { W } _ { p }$ between probability measures. For vectors, $\lVert v \rVert$ denotes the Euclidean $( \ell _ { 2 } )$ norm, and for matrices, $\| A \| _ { \mathrm { o p } }$ denotes the operator norm. We write $[ T ] \triangleq \{ 1 , \dots , T \}$

## A. Replay-Based Continual Learning

We consider supervised classification with input space X, label space $\mathcal { V }$ and joint space $\mathcal { Z } = \mathcal { X } \times \mathcal { Y }$ . A sequence of $T$ tasks $D ^ { 1 : T } = \{ D ^ { 1 } , \ldots , \bar { D ^ { T } } \}$ arrives incrementally. For each $t \in [ T ]$ , task t provides a dataset $D ^ { t } = \{ Z _ { i } ^ { t } \} _ { i = 1 } ^ { n } \in \mathcal { Z } ^ { n }$ , consisting of n i.i.d. samples drawn from an unknown task distribution $\mathcal { D } _ { t }$ on Z. We do not impose restrictions on the relationships among $\{ \mathcal { D } _ { t } \} _ { t = 1 } ^ { T } .$ , permitting arbitrary distributional heterogeneity across tasks. Moreover, the datasets $\{ D ^ { t } \} _ { t = 1 } ^ { T }$ are assumed independent across tasks. At the beginning of task t, the learner receives a fresh dataset $D ^ { t }$ and a replay buffer $\mathbf { \bar { \mathcal { M } } ^ { 1 : t - 1 } } = \{ M ^ { i } \} _ { i = 1 } ^ { t - 1 }$ holding exemplars from previous tasks. For each past task $i \in [ t - 1 ]$ , the buffer stores a subset $M ^ { i } = \{ Z _ { i } ^ { i } \} _ { j \in \mathbb { Z } _ { i } } \subset D ^ { i }$ , where the index set $\mathcal { T } _ { i } \subset [ n ]$ has fixed size $| { \mathcal { T } } _ { i } | = m$ and is drawn uniformly without replacement from [n]. We focus on the memory-limited regime $m \ll n$ . Importantly, a past task never recurs as a full dataset: once task i is completed, it can be revisited only through the m exemplars retained in $M ^ { i }$

Let W denote the hypothesis parameter space, and let $\ell : \mathcal { W } \times \mathcal { Z } \to \mathbb { R } _ { + }$ be a fixed nonnegative loss function. After receiving task t, training uses the union of the current dataset and all replayed exemplars. We define the empirical risk for $W \in { \mathcal { W } }$ as

$$
\hat { \mathcal { L } } ^ { ( t ) } ( W ) \ = \ \sum _ { i = 1 } ^ { t - 1 } \frac { 1 } { m } \sum _ { j \in \mathcal { T } _ { i } } \ell ( W , Z _ { j } ^ { i } ) \ + \ \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \ell ( W , Z _ { j } ^ { t } ) ,\tag{1}
$$

with the population risk

$$
\mathcal { L } ^ { ( t ) } ( W ) = \sum _ { i = 1 } ^ { t } \mathbb { E } _ { Z \sim \mathcal { D } _ { i } } \big [ \ell ( W , Z ) \big ] .\tag{2}
$$

Since performance is evaluated after training over all tasks, we write $\mathcal { L } ( W ) \equiv \mathcal { L } ^ { ( T ) } ( W )$ and $\hat { \mathcal { L } } ( W ) \equiv \hat { \mathcal { L } } ^ { ( T ) } ( W )$ , and define the expected generalization error of replay-based continual learning as

$$
\mathrm { g e n } _ { W } \ \triangleq \ \mathbb { E } _ { W , D ^ { 1 : T } , \{ \mathbb { Z } _ { i } \} _ { i < T } } \big [ \mathcal { L } ( W ) - \hat { \mathcal { L } } ( W ) \big ] ,\tag{3}
$$

where the expectation is over the task data, the algorithm randomness, and the replay-sampling procedure. Here $W$ is not a fixed parameter but the random output of the learning algorithm: its law is induced jointly by the stream $D ^ { 1 : T }$ , the random buffer construction, and any stochasticity of the optimizer.

## B. Layer-wise Replay Distributions

Layer-wise representations. We analyze a depth-L network $f _ { W } : \mathcal { X } \to \mathbb { R } ^ { K }$ with weights $W = ( W _ { 1 } , \ldots , W _ { L } )$ , where $W _ { l } \in \mathbb { R } ^ { d _ { l } \times d _ { l - 1 } }$ . Setting $a _ { 0 } ( x ) = x$ , each layer defines the intermediate representation

$$
a _ { l } ( x ) = \phi _ { l } ( W _ { l } a _ { l - 1 } ( x ) ) , \qquad l \in [ L ] ,\tag{4}
$$

for an element-wise activation $\phi _ { l } .$ , and the output is $\hat { y } = f _ { W } ( x ) = a _ { L } ( x )$ . We write $A _ { l } = a _ { l } ( \boldsymbol { X } )$ for the layer-l representation of a random input $X ,$ , and $A _ { j , l } ^ { t } = a _ { l } ( X _ { j } ^ { t } )$ for the representation of the j-th example of task t. We use $W _ { 1 : l } = ( W _ { 1 } , \ldots , W _ { l } )$ for the bottom (feature) sub-network up to layer l and $W _ { l + 1 : L }$ for the remaining suffix.

Split-layer distributions. To analyze generalization at an arbitrary depth, we fix a split layer $l \in \{ 0 , \ldots , L \}$ and condition on the learned bottom parameters $W _ { 1 : l }$ , with the conventions that $W _ { 1 : 0 }$ and $W _ { L + 1 : I }$ are empty. We unify the data source of each task as $S ^ { i }$ , where $S ^ { i } = M ^ { i }$ for an old task $i < T$ and $S ^ { i } = D ^ { T }$ for the current task $i = T$ , and write $S \triangleq \textstyle \bigcup _ { i = 1 } ^ { T } S ^ { i }$ for the full training set at time T. Replay generalization is governed by how three laws of the representation–label pair $( A _ { l } , Y )$ relate to one another, which we introduce in turn.

(i) Population law. $P _ { A _ { l } , Y | i , W _ { 1 : l } }$ is the population distribution of $( A _ { l } , Y )$ obtained by pushing $( X , Y ) \sim { \mathcal { D } } _ { i }$ through the learned features. Here, “pushing through $W _ { 1 : l } '$ acts on the input only: $X \mapsto A _ { l } = a _ { l } ( X )$ , while the label Y is carried over unchanged. This is the target the learner would like to control but never observes directly.

(ii) Empirical proxy. $\hat { P } _ { A _ { l } , Y | S ^ { i } , W _ { 1 : } }$ is the uniform distribution over the representation-label pairs induced by the finite sample $S ^ { i }$ under the learned features $W _ { 1 : l }$ . This is the actual objective used in empirical risk minimization.

(iii) Replay centroid. To quantify the statistical bias introduced by the finite buffer, we average over the randomness of buffer construction and define, for $i < T ,$

$$
Q _ { A _ { l } , Y | i , W _ { 1 : l } } \ \triangleq \ \mathbb { E } _ { S ^ { i } } \left[ \hat { P } _ { A _ { l } , Y | S ^ { i } , W _ { 1 : l } } \ \Big | \ W _ { 1 : l } \right] .\tag{5}
$$

The subscript $W _ { 1 : l }$ and the outer expectation are not redundant: the inner object $\hat { P } _ { A _ { l } , Y | S ^ { i } , W _ { 1 : l } }$ is the proxy for a single realized buffer $S ^ { i }$ , whereas the outer $\mathbb { E } _ { S ^ { i } } [ \cdot ~ \vert ~ W _ { 1 : l } ]$ averages over which m exemplars happened to be stored, holding the features at their learned value. Equivalently, $Q _ { A _ { l } , Y | i , W _ { 1 : l } }$ is the conditional distribution of a random training example from task i given the learned representation parameters. For old tasks, in general, one has $Q _ { A _ { l } , Y | i , W _ { 1 : l } } \neq P _ { A _ { l } , Y | i , W _ { 1 : l } }$ . The cause is a representation-induced selection effect: although the raw exemplars are drawn i.i.d. from $\mathcal { D } _ { i } \mathrm { - } \mathrm { s } _ { 0 }$ that at the input layer a uniformly chosen stored example already carries the task marginal—the feature map $W _ { 1 : l }$ is itself fitted to the particular realized buffer. Conditioning on this data-dependent $W _ { 1 : l }$ ties the stored representations to the very sample that shaped them, shifting their averaged law away from the population.

(iv) Stacked training representations. Collecting the layer-l representations of the entire training set gives

$$
\mathbf { U } ^ { ( l ) } \triangleq \left\{ ( a _ { l } ( x ) , y ) : ( x , y ) \in S \right\} \equiv \big ( \mathbf { U } _ { \mathrm { o l d } } ^ { ( l ) } , \mathbf { U } _ { \mathrm { n e w } } ^ { ( l ) } \big ) ,\tag{6}
$$

where $\mathbf { U } _ { \mathrm { o l d } } ^ { ( l ) }$ gathers the representations of the buffered exemplars $\mathsf { U } _ { i < T } M ^ { i }$ and $\mathbf { U } _ { \mathrm { n e w } } ^ { ( l ) }$ those of the current task $D ^ { T }$ . We emphasize that $\mathbf { U } _ { \mathrm { o l d } } ^ { ( l ) }$ is a collection of representations, not of raw exemplars: the stored inputs are frozen once written to the buffer, but their images $a _ { l } ( \cdot )$ shift as the shared features $W _ { 1 : l }$ are learned. This is exactly why replayed data can carry information about the current optimization even though the buffer content is fixed.

(v) Decoupled reference. Let $P _ { \mathbf { U } ^ { ( l ) } | W _ { 1 : l } }$ denote the joint law of this stacked sequence. Because all coordinates share the data-dependent map $W _ { 1 : l }$ and the buffer is drawn without replacement, the entries of $\mathbf { U } ^ { ( l ) }$ are statistically dependent. To isolate this coupling, we compare against an idealized Decoupled Reference in which every coordinate is drawn independently, with old-task coordinates from their centroid and current-task coordinates from the population:

$$
\widetilde { Q } _ { { \bf U } ^ { ( l ) } | W _ { 1 : l } } \triangleq \left( \bigotimes _ { i = 1 } ^ { T - 1 } ( Q _ { A _ { l } , Y | i , W _ { 1 : l } } ) ^ { \otimes m } \right) \otimes \big ( P _ { A _ { l } , Y | T , W _ { 1 : l } } \big ) ^ { \otimes n } .\tag{7}
$$

The tensor exponents record the sample budgets: $( \cdot ) ^ { \otimes m }$ places m i.i.d. copies for each of the $T - 1$ past tasks (one per stored exemplar), and $( \cdot ) ^ { \otimes n }$ places n i.i.d. copies for the current task. The reference is therefore an i.i.d. process across the stored and current coordinates: it preserves every coordinate’s marginal but strips away the cross-sample coupling created by non-replacement sampling and by the shared feature map.

## IV. A LAYER-WISE DECOMPOSITION OF REPLAY GENERALIZATION

This section contains the backbone result of the paper. Theorem IV.1 is the only place where the full replay generalization gap is decomposed. The remainder of the paper revisits its two components separately: the next subsection presents the decomposition itself, Section IV-B relaxes the drift branch geometrically when KL control becomes vacuous, and Section V upper bounds the optimization branch for a concrete noisy optimizer.

## A. Main Decomposition: The Synergy-Drift Bound

The classical stability–plasticity framework is useful [21], but it does not by itself isolate the replay-specific bias induced by finite memory nor the dependence created by repeatedly optimizing on replayed and current samples. The next theorem addresses both issues simultaneously: it isolates replay-induced drift and then decomposes the remaining optimization complexity into stability, plasticity, interaction, and residual coupling.

Theorem IV.1 (Hierarchical Synergy–Drift Bound). Let W be the output of a learner minimizing the empirical risk $\hat { \mathcal { L } } ( W )$ Assume the loss function $\ell ( W , Z )$ is σ-subgaussian for all $Z \in { \mathcal { Z } }$ and $W \in { \mathcal { W } }$ . For any split layer $l \in \{ 0 , \ldots , L \}$ , we have

$$
\left. \mathrm { g e n } _ { W } \right. \leq \underbrace { \left( T - 1 \right) \sqrt { 2 \sigma ^ { 2 } { \mathcal K } ^ { \left( l \right) } } } _ { \mathrm { R e p l a y - c e n t r o i d ~ D r i f t } } + \underbrace { \sqrt { \frac { 2 \sigma ^ { 2 } } { N _ { \mathrm { e f f } } } } \Bigl ( { S } ^ { \left( l \right) } + { \mathcal P } ^ { \left( l \right) } - { \mathcal R } ^ { \left( l \right) } + { \mathcal C } ^ { \left( l \right) } \Bigr ) } _ { \mathrm { O p t i m i z a t i o n ~ V a r i a n c e } } ,
$$

where $N _ { \mathrm { e f f } } \triangleq \frac { 1 } { ( T - 1 ) / m + 1 / n }$ is the effective sample size, $\begin{array} { r } { K ^ { ( l ) } \triangleq \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } \mathbb { E } _ { W _ { 1 : l } } [ D _ { \mathrm { K L } } ( Q _ { A _ { l } , Y | i , W _ { 1 : l } } \Vert P _ { A _ { l } , Y | i , W _ { 1 : l } } ) ] , \ \mathscr { C } ^ { ( l ) } \triangleq } \end{array}$ $\mathbb { E } _ { W _ { 1 : l } } [ D _ { \mathrm { K L } } ( P _ { \mathbf { U } ^ { ( l ) } | W _ { 1 : l } } \| \widetilde { Q } _ { \mathbf { U } ^ { ( l ) } | W _ { 1 : l } } ) ]$

Theorem IV.1 splits replay generalization into two parts. The replay-centroid drift term captures the bias induced by finite memory: at layer l, the mismatch between the replay centroid $Q _ { A _ { l } , Y | i , W _ { 1 : l } }$ and the corresponding population distribution $P _ { A _ { l } , Y | i , W _ { 1 : l } } . \mathrm { ~ A ~ }$ key feature of this term is that it generally does not vanish as training proceeds, even though the raw exemplars are drawn i.i.d. from the task. The i.i.d. sampling guarantees only that the stored data match the population in input space, which is why the drift is zero at the input layer. In feature space the situation differs, because the map $W _ { 1 : l }$ is itself fitted to the particular stored buffer. Conditioning on this buffer-dependent map couples the stored representations to the same samples that shaped the map, and this selection effect displaces their averaged law away from the population. The second term measures the optimization dependence under the actual training representations.

Its prefactor $\sqrt { 2 \sigma ^ { 2 } / N _ { \mathrm { e f f } } }$ is set by the effective sample size $\begin{array} { r } { N _ { \mathrm { e f f } } = \frac { 1 } { ( T - 1 ) / m + 1 / n } } \end{array}$ for the replay-current empirical objective, and the variance term scales as $1 / \sqrt { N _ { \mathrm { e f f } } }$ . When replay is effectively unlimited $( m  \infty )$ , the old-task contribution $( T - 1 ) / m$ vanishes and the current-task dependence recovers the usual $\mathcal { O } ( 1 / \sqrt { n } )$ rate. In contrast, when the current-task size grows under a fixed replay budget $( n  \infty ) , N _ { \mathrm { e f f } }$ approaches the finite limit $m / ( T - 1 )$ , so the variance term does not vanish but settles at a finite floor of order $\sqrt { ( T - 1 ) / m }$ , which we call the finite-memory variance floor. The variance term should not be considered a measure of forgetting. It bounds the train-population discrepancy, whereas forgetting is the degradation of old-task population risk, carried by the drift term $\kappa ^ { ( l ) }$ , which does not vanish during training, together with the cross-task dependence accumulated along the optimization trajectory (Corollary V.2).

The optimization term further admits a stability–plasticity–synergy (SPS) decomposition. Here $S ^ { ( l ) } \triangleq I ( \mathbf { U } _ { \mathrm { o l d } } ^ { ( l ) } ; W _ { l + 1 : L } \mid W _ { 1 : l } )$ measures how much information the suffix weights retain about replayed old-task representations; this is the stability term, the learner’s information-theoretic memory of past tasks. Correspondingly, $\mathcal { P } ^ { ( l ) } \triangleq I ( \mathbf { U } _ { \mathrm { n e w } } ^ { ( l ) } ; W _ { l + 1 : L } \ | \ W _ { 1 : l } )$ measures the dependence on the current task; this is the plasticity term, the extent to which the suffix encodes and adapts to the new task. The interaction term $\mathscr { R } ^ { ( l ) } \triangleq I ( \mathbf { U } _ { \mathrm { o l d } } ^ { ( l ) } ; \mathbf { U } _ { \mathrm { n e w } } ^ { ( l ) } ; W _ { l + 1 : L } \mid \dot { W } _ { 1 : l } )$ is signed: positive values tighten the bound and correspond to beneficial sharing between old and current representations, whereas negative values enlarge the bound and correspond to interference. The theorem does not assert that $\mathcal { R } ^ { ( \bar { l } ) }$ is always nonnegative. Finally, $\mathscr { C } ^ { ( l ) }$ measures the residual dependence between the actual replay sequence and an idealized independent reference. This term accounts for statistical coupling introduced by sampling without replacement and by the shared learned representation map $W _ { 1 : l }$

Corollary IV.2 (Input-Layer Bound). At the input layer $l = 0 ,$ , the hierarchical generalization bound simplifies to:

$$
| \mathrm { g e n } _ { W } | \leq \sqrt { \frac { 2 \sigma ^ { 2 } } { N _ { \mathrm { e f f } } } I \big ( S ; W \big ) } ,
$$

where $S$ is the original (X, Y ) pair in the replay buffer and the current task dataset.

Specifically, at the input layer, $\mathbf { U } ^ { ( 0 ) }$ represents the original data pair. From Theorem IV.1, we have $I ( S ; W ) = I ( \mathbf { U } _ { \mathrm { o l d } } ^ { ( 0 ) } ; W ) +$ $I ( \bar { \mathbf { U } _ { \mathrm { n e w } } ^ { ( 0 ) } } ; W ) ^ { - } - I ( \mathbf { U } _ { \mathrm { o l d } } ^ { ( 0 ) } ; \mathbf { U } _ { \mathrm { n e w } } ^ { ( 0 ) } ; W )$ . The replay centroid $Q _ { X , Y \mid i , W }$ coincides with the population $P _ { X , Y \mid i , W }$ and the training sequence is block-wise i.i.d., therefore $\mathcal { K } ^ { ( 0 ) } = 0$ and $\mathcal { C } ^ { ( 0 ) } = 0 ,$ . In the special case $T = 1$ , replay disappears: the empirical and population risks reduce to their standard single-task form, $N _ { \mathrm { e f f } } = n$ , and the bound recovers the result of [15]. When $m = n$ the buffer retains every past example and S consists of $T n$ i.i.d. samples; the bound then matches the classical on-average stability bound [16] for $T n$ samples. Corollary IV.2 thus extends the standard mutual-information framework to replay-based continual learning. The CL structure therefore does not arise at the input layer itself. Instead, replay-specific effects emerge through the finite-memory factor $N _ { \mathrm { e f f } }$ , the representation-level drift term $\kappa ^ { ( i ) }$ , and the interaction terms appearing for $l > 0$ which become relevant only when multiple tasks are learned under memory constraints.

Then, when considering $l = L$ , we obtain a bound dominated by pure drift:

Corollary IV.3 (Output-Layer Bound). At the output layer $l = L ,$ the hierarchical generalization bound simplifies to:

$$
| \mathrm { g e n } _ { W } | \leq ( T - 1 ) \sqrt { 2 \sigma ^ { 2 } { \mathcal K } ^ { ( L ) } } + \sqrt { \frac { 2 \sigma ^ { 2 } } { N _ { \mathrm { e f f } } } { \mathcal C } ^ { ( L ) } } ,
$$

$$
\begin{array} { r } { w h e r e \ K ^ { ( L ) } = \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } \mathbb { E } _ { W } \big [ D _ { \mathrm { K L } } \big ( Q _ { A _ { L } , Y | i , W } \big | | P _ { A _ { L } , Y | i , W } \big ) \big ] , \ \mathcal { C } ^ { ( L ) } = \mathbb { E } _ { W } \big [ D _ { \mathrm { K L } } \big ( P _ { \Psi ^ { ( L ) } | W } \big | | \widetilde { Q } _ { \Psi ^ { ( L ) } | W } \big ) \big ] . } \end{array}
$$

In particular, Corollary IV.3 can be viewed as a degenerate limit of Theorem IV.1. At the output layer, there is no trainable mapping downstream of the representation, hence the stability–plasticity–synergy tradeoff over $W _ { l + 1 : L }$ disappears and ${ \mathcal { S } } ^ { ( L ) } =$ $\mathcal { P } ^ { ( \bar { L } ) } \stackrel { - } { = } \mathcal { R } ^ { ( L ) } = 0$ , leaving generalization solely governed by the replay-centroid drift $\mathcal { K } ^ { ( L ) }$ and dependence divergence $\mathcal { C } ^ { ( L ) }$ . Read together with Corollary IV.2, this highlights a depth-wise shift of the dominant error: from parameter-information complexity at shallow splits to representational mismatch at deep ones. This shift is directly actionable. Because neither end of the network balances the two costs, the bound predicts that stabilization techniques such as feature distillation or partial freezing [44] should target the interior layer that minimizes the drift–sensitivity product, rather than being applied uniformly across layers. Figure 3 locates this layer empirically, and the feature-distillation probe confirms that stabilizing the interior basin outperforms stabilizing either end.

## B. Geometric Relaxation of the Drift Term

Theorem IV.1 identifies replay-induced drift as one of the two sources of replay generalization error. Its KL form is natural from the information-theoretic proof, but it is not always the most informative geometry for replay. When an old task is represented by finitely many stored examples while the corresponding population representation distribution is continuous, KL-based drift can become vacuous under support mismatch even if the two measures remain geometrically close. The next result therefore revisits only the drift branch of Theorem IV.1: it keeps the layer-wise representation viewpoint but replaces KL by Wasserstein geometry.

Theorem IV.4 (Hierarchical Wasserstein Bound). Assume the loss function $\ell ( \cdot , y )$ is ρ -Lipschitz and the activation functions ϕ<sub>l</sub> are ρ<sub>l</sub>-Lipschitz. Then, at time T, we have:

$$
\left. \mathrm { g e n } _ { W } \right. \leq \operatorname* { m i n } _ { l \in \{ 0 , \ldots , L \} } \mathbb { E } \biggl [ \bar { \rho } _ { l } ( W ) \cdot \sum _ { i = 1 } ^ { T } \mathcal { W } _ { 1 } \Bigl ( \hat { P } _ { A _ { l } , Y | S ^ { i } , W _ { 1 : l } } , P _ { A _ { l } , Y | i , W _ { 1 : l } } \Bigr ) \biggr ] ,
$$

where $\begin{array} { r } { \bar { \rho } _ { l } ( W ) : = \rho _ { 0 } \left( 1 \vee \prod _ { h = l + 1 } ^ { L } \rho _ { h } \| W _ { h } \| _ { \mathrm { o p } } \right) } \end{array}$

Theorem IV.4 relaxes the drift component of the generalization bound into a geometric form, complementing rather than replacing the full stability-plasticity-interaction (SPS) decomposition of the optimization term in Theorem IV.1. The Wasserstein distance $\mathcal { W } _ { 1 }$ quantifies the distribution mismatch between replay-induced representation–label pairs $( A _ { l } , Y )$ and their task-i population counterparts, while $\bar { \rho } _ { l } ( W )$ measures how strongly the suffix network amplifies this mismatch. This relaxation admits a finite, geometrically interpretable bound even under support mismatch, at the cost of losing the explicit structural decomposition of the optimization term.

The layer minimization reveals a depth-dependent trade-off. At shallow layers, representations tend to be shared across tasks, keeping $\mathcal { W } _ { 1 }$ small, but any residual mismatch is amplified by a long suffix network, inflating $\bar { \rho } _ { l } ( W )$ . At deeper layers, this amplification shrinks while representations become increasingly task-specific, enlarging the drift. The optimal layer $l ^ { \star }$ is where these two competing effects are balanced, yielding the tightest bound — a phenomenon we formalize as the “generalization funnel layer” in the following.

Corollary IV.5 (Generalization Funnel Layer). Define a Generalization Funnel Layer as any minimizer of the geometric upper-bound proxy:

$$
l ^ { \star } \in \underset { l \in \{ 0 , \ldots , L \} } { \arg \operatorname* { m i n } } \mathbb { E } \biggl [ \rho _ { 0 } \bigl ( \prod _ { j = l + 1 } ^ { L } \rho _ { j } \| W _ { j } \| _ { \mathrm { o p } } \bigr ) \cdot \sum _ { i = 1 } ^ { T } \mathcal { W } _ { 1 } \bigl ( \hat { P } _ { A _ { l } , Y | S ^ { i } , W _ { 1 : l } } , P _ { A _ { l } , Y | i , W _ { 1 : l } } \bigr ) \Bigg ] .
$$

Corollary IV.5 does not claim that every architecture possesses a unique, intrinsic funnel layer. Rather, it identifies a principled criterion: minimizing the product of suffix sensitivity and representation drift for selecting which layer to be stabilized. When drift increases with depth while suffix sensitivity decreases, this product attains its minimum at an intermediate layer, providing a bound-level rationale for intermediate-layer stabilization strategies such as feature distillation or partial freezing [45], [46].

Remark IV.6 (General layer families). The factor $\bar { \rho } _ { l } ( W )$ enters Theorem IV.4 only as an upper bound on the Lipschitz constant of the suffix map $a _ { l } \mapsto { \hat { y } }$ , and its proof uses nothing about the layers beyond the fact that Lipschitz constants compose. The product $\begin{array} { r } { \prod _ { h > l } \rho _ { h } \| W _ { h } \| _ { \mathrm { o p } } } \end{array}$ is thus the closed form of this constant for the feedforward layers of (4); for any other family of Lipschitz layers the same bound holds with $\rho _ { h } \| W _ { h } \| _ { \mathrm { o p } }$ replaced by the constant $\kappa _ { h } : = \mathrm { L i p } ( g _ { W _ { h } } )$ of the layer-h map $g _ { W _ { h } } : a _ { h - 1 } \mapsto a _ { h }$ , giving $\begin{array} { r } { \bar { \rho } _ { l } ( W ) = \rho _ { 0 } \big ( 1 \vee \prod _ { h > l } \kappa _ { h } \big ) } \end{array}$ . A residual block $g _ { W _ { h } } ( u ) = P _ { h } u + F _ { h } ( u )$ , with shortcut $P _ { h }$ (the identity when the dimensions agree) and branch $F _ { h } .$ , satisfies

$$
\begin{array} { r } { \kappa _ { h } \leq \| P _ { h } \| _ { \mathrm { o p } } + \mathrm { L i p } ( F _ { h } ) , \qquad \mathrm { L i p } ( F _ { h } ) \leq \prod _ { k } \rho _ { h , k } \| W _ { h , k } \| _ { \mathrm { o p } } , } \end{array}
$$

the product running over the layers of the branch $F _ { h }$ and any post-addition activation contributing only its own (unit, for ReLU) Lipschitz factor. The residual networks used in our experiments are therefore instances of Theorem IV.4 and Corollary IV.5: only the per-layer constant $\kappa _ { h }$ changes, and the drift–sensitivity trade-off is read from the same product.

## V. OPERATIONALIZING THE OPTIMIZATION TERM: AN SGLD INSTANTIATION

Theorem IV.1 leaves the optimization branch $\mathcal { S } ^ { ( l ) } + \mathcal { P } ^ { ( l ) } - \mathcal { R } ^ { ( l ) } + \mathcal { C } ^ { ( l ) }$ in an optimizer-agnostic form. This section makes it concrete for one explicit noisy optimizer, stochastic gradient Langevin dynamics (SGLD) [47]. The result is specific to this branch and to SGLD: it bounds the optimization complexity, complementing the geometric treatment of the drift term in Section IV-B.

## A. Algorithm-Based Hierarchical Bounds

For analytical convenience, we condition on the prefix parameters $W _ { 1 : l }$ (the bottom l layers) and analyze the generalization performance of the suffix parameters $\Theta = W _ { l + 1 : L }$ . Conducting this analysis for an arbitrary split layer l produces a unified layer-wise framework. Let $\{ W _ { r } \} _ { r = 0 } ^ { R }$ denote the training trajectory of the SGLD-based CL algorithm for current task $T ,$ where $W _ { 0 }$ is initialized from the optimized parameters of task $T - 1$ . To derive a generalization bound applicable to an arbitrary layer l, we fix $W _ { 1 : l }$ and update only the remaining parameters $\Theta _ { r } ^ { ( T ) } : = \bar { W } _ { l + 1 : L , r } \in \mathbb { R } ^ { d _ { l + 1 : L } }$ . At each iteration $r = 1 , \cdots , R ,$ we sample a replay mini-batch $\dot { B } _ { r } ^ { \mathrm { o l d } }$ of size $b _ { \mathrm { o l d } }$ from $\scriptstyle { \mathcal { M } } ^ { 1 : T - 1 }$ and a current-task mini-batch $B _ { r } ^ { \mathrm { n e w } }$ of size $b _ { \mathrm { n e w } }$ from $D ^ { T }$ . Define the corresponding stochastic gradient descent directions $\begin{array} { r } { G _ { r } ^ { \mathrm { o l d } } : = - \frac { 1 } { b _ { \mathrm { o l d } } } \sum _ { Z \in B _ { r } ^ { \mathrm { o l d } } } \nabla \Theta \ell ( ( W _ { 1 : l } , \Theta _ { r - 1 } ) , Z ) } \end{array}$ and $\begin{array} { r } { G _ { r } ^ { \mathrm { n e w } } : = - \frac { 1 } { b _ { \mathrm { n e w } } } \sum _ { Z \in B _ { r } ^ { \mathrm { n e w } } } \nabla _ { \Theta } \ell ( ( W _ { 1 : l } , \Theta _ { r - 1 } ) , Z ) } \end{array}$ . The update rule at iteration r can then be formalized by

$$
\Theta _ { r } ^ { ( T ) } = \Theta _ { r - 1 } ^ { ( T ) } + \eta _ { r } G _ { r } + N _ { r } , \qquad N _ { r } \sim \mathcal { N } ( 0 , \tau _ { r } ^ { 2 } I _ { d _ { l + 1 : L } } ) ,
$$

where the gradient mixture is $G _ { r } = \lambda _ { \mathrm { o l d } } G _ { r } ^ { \mathrm { o l d } } + \lambda _ { \mathrm { n e w } } G _ { r } ^ { \mathrm { n e w } }$ with weights $\begin{array} { r } { \lambda _ { \mathrm { o l d } } = \frac { T - 1 } { T } } \end{array}$ and $\begin{array} { r } { \lambda _ { \mathrm { n e w } } = \frac { 1 } { T } } \end{array}$ . These weights point along the gradient of $\hat { \mathcal { L } } ( W )$ , whose overall scale is absorbed into the learning rate $\eta _ { r }$ , so the update shares the minimizer of ${ \hat { \mathcal { L } } } ( W )$ . The term $N _ { r }$ is the isotropic Gaussian noise injected at each step.

Notably, in traditional single-task SGLD analysis, the initialization $W _ { 0 }$ is typically assumed to be independent of the current dataset. However, this independence generally fails in CL. At task T, the initialization $\Theta _ { 0 } ^ { ( T ) }$ is inherited from the previous task $\Theta _ { 0 } ^ { ( T ) } = \Theta _ { R } ^ { ( T - 1 ) }$ . Consequently, it may already encode information about the current representation $\mathbf { U } _ { T } ^ { ( l ) } = \big ( \mathbf { U } _ { \mathrm { o l d } } ^ { ( l ) ^ { \ast } } , \mathbf { U } _ { \mathrm { n e w } } ^ { ( l ) } \big )$ especially when $\mathbf { U } _ { \mathrm { o l d } } ^ { ( l ) }$ is constructed by replaying past data. To accurately characterize the optimization at time T, we must disentangle the inherited information carried by $\mathbf { \bar { \Theta } } _ { 0 } ^ { ( T ) }$ from the new information created during task-T optimization, and then further decompose the latter along the SGLD trajectory.

Proposition V.1 (Heritage). Fix a layer l and condition on $W _ { 1 : l } .$ At task $T ,$ we have

$$
\begin{array} { r l } & { I ( \mathbf { U } _ { T } ^ { ( l ) } ; \boldsymbol { \Theta } _ { R } ^ { ( T ) } | W _ { 1 : l } ) \leq I ( \mathbf { U } _ { T } ^ { ( l ) } ; \boldsymbol { \Theta } _ { 0 } ^ { ( T ) } | W _ { 1 : l } ) + I ( \mathbf { U } _ { T } ^ { ( l ) } ; \boldsymbol { \Theta } _ { R } ^ { ( T ) } | \boldsymbol { \Theta } _ { 0 } ^ { ( T ) } , W _ { 1 : l } ) } \\ & { \qquad = : H _ { T } ^ { ( l ) } + \Delta _ { T } ^ { ( l ) } . } \end{array}
$$

This proposition ties the SGLD trajectory back to the main decomposition. By the interaction identity, the optimization complexity in Theorem IV.1 is a single conditional mutual information, $\mathbf { \mathcal { S } } ^ { ( l ) } + \mathbf { \dot { \mathcal { P } } } ^ { ( l ) } - \mathbf { \mathcal { R } } ^ { ( l ) } = I ( \mathbf { U } _ { T } ^ { ( l ) } ; \mathbf { \dot { W } } _ { l + 1 : L } \mathbf { \dot { \Omega } } | \mathbf { \Omega } { W } _ { 1 : l } ) = \mathbf { \Omega }$ $I ( \mathbf { U } _ { T } ^ { \overline { { ( l ) } } } ; \Theta _ { R } ^ { \overline { { ( T ) } } } \mid W _ { 1 : l } )$ , where $\Theta _ { R } ^ { ( T ) }$ is the task- $- T$ terminal suffix $W _ { l + 1 : L }$ . Proposition V.1 splits it into a heritage term $\dot { H } _ { T } ^ { ( l ) }$ the dependence already present in the inherited initialization, and an increment $\Delta _ { T } ^ { ( l ) }$ , the dependence injected by task-T optimization. Crucially, because $\Theta _ { 0 } ^ { ( t + 1 ) } = \Theta _ { R } ^ { ( t ) }$ , the heritage term is not generated afresh at task $T$ but carries forward the dependent build up over all previous tasks. Corollary V.2 makes this accumulation precise as a bound by past increments.

Corollary V.2 (Cumulative information budget). Fix a layer l and condition on $W _ { 1 : l }$ . For any $T \geq 2 ,$ the heritage term admits the following cumulative-budget upper bound:

$$
H _ { T } ^ { ( l ) } = I ( \mathbf { U } _ { T } ^ { ( l ) } ; \Theta _ { 0 } ^ { ( T ) } \mid W _ { 1 : l } ) \leq \sum _ { t = 1 } ^ { T - 1 } \Delta _ { t } ^ { ( l ) } ,
$$

where $\Delta _ { t } ^ { ( l ) } : = I ( \mathbf { U } _ { t } ^ { ( l ) } ; \Theta _ { R } ^ { ( t ) } \mid \Theta _ { 0 } ^ { ( t ) } , W _ { 1 : l } )$ is the within-task information increment in Proposition V.1 applied to task t.

Corollary V.2 reveals that $H _ { T } ^ { ( l ) }$ can be upper bounded by the cumulative sum of past increments. Therefore, to control the cross-task heritage $H _ { T } ^ { ( l ) }$ , it suffices to control the sequence of within-task increments $\{ \Delta _ { t } ^ { ( l ) } \} _ { t < T }$ . This insight reduces the problem of bounding the total complexity to controlling the within-task increment $\Delta _ { t } ^ { ( l ) }$ . Thus, our analysis now focuses on deriving a sharp, algorithm-dependent bound for general single-task update $\Delta _ { t } ^ { ( l ) }$ , which we instantiate for SGLD in the next theorem.

Theorem V.3 (Task-Wise SGLD-Based Bound). Fix a split layer $l \in \{ 0 , \ldots , L - 1 \}$ and condition on the base parameters $W _ { 1 : l } ,$ so that the suffix network remains nonempty. Assume the loss is σ-subgaussian as in Theorem IV.1. For task t trained by SGLD for R<sub>t</sub> steps, we have

$$
\left. \mathbf { g } \mathbf { e n } _ { W ^ { ( t ) } } \right. \leq ( t - 1 ) \sqrt { 2 \sigma ^ { 2 } K _ { t } ^ { ( l ) } } + \sqrt  \frac { 2 \sigma ^ { 2 } } { N _ { \mathrm { e f f } , t } } \Bigl ( \mathcal { C } _ { t } ^ { ( l ) } + \sum _ { s = 1 } ^ { t } \sum _ { r = 1 } ^ { R _ { s } } \mathbb { E } \left[ \frac { 1 } { 2 } \log \operatorname* { d e t } \Bigl ( I + \frac { \eta _ { s , r } ^ { 2 } } { \tau _ { s , r } ^ { 2 } } M _ { s , r } ^ { ( l ) } \Bigr ) \right] \Bigr ) ,
$$

where $\begin{array} { r } { M _ { s , r } ^ { ( l ) } = \mathbb { E } \left[ G _ { s , r } G _ { s , r } ^ { \top } \middle | \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } \right] } \end{array}$ , and $\kappa _ { t } ^ { ( l ) } , \mathcal { C } _ { t } ^ { ( l ) }$ , and $N _ { \mathrm { e f f } , t }$ denote the corresponding quantities of Theorem IV.1 with the task horizon T replaced by t.

Theorem V.3 characterizes the generalization dynamics of SGLD-based CL algorithms from a hierarchical perspective. It operationalizes only the optimization branch: it reduces the suffix information complexity to a cumulative trajectory budget of log-determinants. This budget is governed by the ratio of learning rate to injected noise $\dot { \eta } _ { s , r } ^ { 2 } / \tau _ { s , \tau } ^ { 2 }$ and the conditional second moment matrix $\boldsymbol { M _ { s , r } ^ { ( l ) } }$ . Unlike standard SGLD bounds that depend only on gradient covariance [37], $\boldsymbol { M _ { s , r } ^ { ( l ) } }$ explicitly captures the systematic direction of the gradient update. The exact budget is high-dimensional; the next subsection expands it to firs order to expose what about replay drives it and to obtain measurable diagnostics.

## B. Gradient-Moment Diagnostics of Task Interaction

While Theorem V.3 bounds the optimization term through a cumulative log-determinant budget, the mechanism by which replay affects this budget remains implicit. We therefore perform a second-order expansion that separates per-step cost into two orthogonal forces: (i) Optimization Instability, driven by gradient covariance; and (ii) Interaction Cost, driven by the alignment of old-task and new-task mean gradients in a sensitivity-aware metric. The resulting alignment quantities are motivated proxies for task interaction in the SGLD model. This reveals the physical meaning of “Synergy” in $\mathrm { { s G L D } } { \cdot }$ it is not merely a reduction in variance, but a constructive interaction of mean descent directions in a curvature-induced metric.

Setup. Throughout this subsection, we fix a split layer l and condition on the current $( \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } )$ . For each task s and step r, let $\alpha _ { s , r } : = \eta _ { s , r } ^ { 2 } / \tau _ { s , i } ^ { 2 }$ be the signal-to-noise ratio. Recall that the second moment matrix $M _ { s , r } ^ { ( l ) }$ is defined over the gradient mixture $G _ { s , r } = \lambda _ { s , \mathrm { o l d } } G _ { s , r } ^ { \mathrm { o l d } } + \lambda _ { s , \mathrm { n e w } } G _ { s , r } ^ { \mathrm { n e w } }$

Lemma V.4 (Second-moment split). Define the conditional mean and covariance

$$
\mu _ { s , r } ^ { ( l ) } : = \mathbb { E } [ G _ { s , r } \ | \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } ] , V _ { s , r } ^ { ( l ) } : = \mathrm { C o v } ( G _ { s , r } \ | \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } ) .
$$

Then

$$
M _ { s , r } ^ { ( l ) } = V _ { s , r } ^ { ( l ) } + \mu _ { s , r } ^ { ( l ) } \mu _ { s , r } ^ { ( l ) \top } .
$$

Lemma V.4 separates stochasticity from signal: the covariance term $V _ { s , r } ^ { ( l ) }$ quantifies gradient instability when learning task s, while the rank-one term $\mu _ { s , r } ^ { ( l ) } \mu _ { s , r } ^ { ( l ) \dag }$ encodes the systematic descent direction, representing the task interaction.

Proposition V.5 (Instability–mean decomposition). Let $A _ { s , r } ^ { ( l ) } : = I + \alpha _ { s , r } V _ { s , r } ^ { ( l ) } \succ 0$ with $\alpha _ { s , r } = \eta _ { s , r } ^ { 2 } / \tau _ { s , r } ^ { 2 }$ . Then

$$
\frac { 1 } { 2 } \log \operatorname* { d e t } \bigl ( I + \alpha _ { s , r } M _ { s , r } ^ { ( l ) } \bigr ) = \underbrace { \frac { 1 } { 2 } \log \operatorname* { d e t } ( A _ { s , r } ^ { ( l ) } ) } _ { i n s t a b i l i t y } + \underbrace { \frac { 1 } { 2 } \log \Bigl ( 1 + \alpha _ { s , r } \mu _ { s , r } ^ { ( l ) \top } \bigl ( A _ { s , r } ^ { ( l ) } \bigr ) ^ { - 1 } \mu _ { s , r } ^ { ( l ) } \Bigr ) } _ { i n t e r a c t i o n \ c o s t } .
$$

Proposition V.5 splits each step’s budget into two sources. The instability term is governed by the gradient covariance $V _ { s , r } ^ { ( l ) }$ a large $V _ { s , r } ^ { ( l ) }$ marks a sharp region of the landscape, which in CL typically means the replay gradients are inconsistent with the current parameters, resulting in catastrophic forgetting. The interaction cost is the mean-gradient energy, which captures the task alignment but is filtered by $( A _ { s , r } ^ { ( l ) } ) ^ { - 1 } \mathrm { ; }$ ; this matrix suppresses the mean gradient $\mu _ { s , r } ^ { ( l ) }$ in directions where gradients are most unstable. This filtering has a consequence worth stating: when the instability term $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ log det $\left( A _ { s , r } ^ { ( l ) } \right)$ is large, the filter shrinks and the second term falls, yet the reduction is not a gain. It implies that the algorithm discards usable gradient signal along sharp directions rather than making progress, remaining stuck in a sharp region. Consequently, synergistic interaction requires that the two tasks’ gradients align within flat, stable directions, which the Euclidean inner product does not distinguish. The next definition supplies a metric that does.

Definition V.6 (Sensitivity metric). Define the sensitivity metric

$$
H _ { s , r } ^ { ( l ) } : = \left( A _ { s , r } ^ { ( l ) } \right) ^ { - 1 } \succsim 0 .
$$

For vectors $u , v \neq 0 .$ , define the $H _ { s , r } ^ { ( l ) }$ -inner product and norm by $\langle u , v \rangle _ { H } : = u ^ { \top } H _ { s , r } ^ { ( l ) }$ v and $\| u \| _ { H } : = \sqrt { \langle u , u \rangle _ { H } }$ , and the induced cosine

$$
\displaystyle \cos _ { H } ( u , v ) : = \frac { \langle u , v \rangle _ { H } } { \| u \| _ { H } \| v \| _ { H } } \in [ - 1 , 1 ] .
$$

Geometrically, $H _ { s , r } ^ { ( l ) }$ induces a stability-aware Mahalanobis metric: it discounts directions of high curvature while prioritizing flat subspaces. This is consistent with the intuition above.

Corollary V.7 (Alignment of old/new gradients). Let $\mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } : = \mathbb { E } [ G _ { s , r } ^ { \mathrm { o l d } } \ | \ \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } ] , \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } : = \mathbb { E } [ G _ { s , r } ^ { \mathrm { n e w } } \ | \ \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } ] ,$ , so that $\mu _ { s , r } ^ { ( l ) } = \lambda _ { s , \mathrm { o l d } } \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } + \lambda _ { s , \mathrm { n e w } } \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) }$ . Then under the sensitivity metric $H _ { s , r } ^ { ( l ) }$ in Definition V.6,

$$
\mu _ { s , r } ^ { ( l ) \top } H _ { s , r } ^ { ( l ) } \mu _ { s , r } ^ { ( l ) } = \lambda _ { s , \mathrm { o l d } } ^ { 2 } \Big \| \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } \Big \| _ { H } ^ { 2 } + \lambda _ { s , \mathrm { n e w } } ^ { 2 } \Big \| \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } \Big \| _ { H } ^ { 2 } + 2 \lambda _ { s , \mathrm { o l d } } \lambda _ { s , \mathrm { n e w } } \Big \| \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } \Big \| _ { H } \Big \| \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } \Big \| _ { H } \mathrm { c o s } _ { H } \big ( \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } , \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } \big ) ,
$$

where the information-geometric alignment

$$
\begin{array} { r } { \mathrm { A l i g n } = \cos _ { H } \Big ( \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } , \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } \Big ) . } \end{array}
$$

Corollary V.7 gives the interaction term $\mathcal { R } ^ { ( l ) }$ of Theorem IV.1 to a concrete geometric reading: whether old and new task gradients align determines whether the per-step budget is small or large. When the two gradients align within the stable subspaces identified by H (constructive synergy), they reinforce a common descent direction, the mean-gradient energy $\| \mu \| _ { H }$ is spent on progress, and the budget stays low. When they conflict (interference), the updates fight rather than reinforce: $\| \mu \| _ { H }$ and the instability factor $A _ { s , r } ^ { ( l ) }$ remain large, the optimization fails to settle, and the cumulative budget surges. Alignment and interference are thus the two regimes that drive the generalization cost down or up, respectively.

This sensitivity-aware alignment is related to the gradient-projection family—GEM, A-GEM, and orthogonal projection [31]–[33]: both address the conflict between old- and current-task gradients, but ours differs in three concrete ways. First, the gradients compared here are the replay and current mini-batch gradients of the running optimizer, not gradients stored on a fixed past subspace. Second, alignment is read in the sensitivity metric $H _ { s , r } ^ { ( l ) } = ( I + \overline { { \alpha } } _ { s , r } \overline { { V _ { s , r } ^ { ( l ) } } } ) ^ { - 1 }$ rather than the Euclidean inner product, so unstable directions are discounted instead of all coordinates being weighted equally. Third, the quantity is diagnostic and feeds the soft, per-step mixing rule of Corollary V.9, not a hard orthogonality constraint on the update. The alignment cosine is therefore a reading of the bound rather than a projection imposed on training.

## C. Interpretable Diagnostics

We conclude our theoretical analysis by translating the trajectory budget into practical scalar signals and a local sensitivity weighted mixing rule. First, a first-order relaxation of the budget yields decomposable diagnostics:

Corollary V.8 (Scalar diagnostic decomposition). Conditioning on $( \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } )$ , the per-step trajectory budget admits the scalar upper bound

$$
\sum _ { s = 1 } ^ { t } \sum _ { r = 1 } ^ { R _ { s } } \mathbb { E } \left[ \frac { 1 } { 2 } \log \operatorname* { d e t } \left( I + \alpha _ { s , r } M _ { s , r } ^ { ( l ) } \right) \right] \leq \frac { 1 } { 2 } \sum _ { s = 1 } ^ { t } \sum _ { r = 1 } ^ { R _ { s } } \alpha _ { s , r } \mathbb { E } \Bigl [ \mathrm { t r } \bigl ( V _ { s , r } ^ { ( l ) } \bigr ) + \| \mu _ { s , r } ^ { ( l ) } \| ^ { 2 } \Bigr ] .
$$

Moreover, the two scalar components admit the explicit decompositions t $\begin{array} { r } { \cdot \bigl ( V _ { s , r } ^ { ( l ) } \bigr ) = \lambda _ { s , \mathrm { o l d } } ^ { 2 } \mathrm { t r } \bigl ( \Sigma _ { s , r } ^ { \mathrm { o l d } , ( l ) } \bigr ) + \lambda _ { s , \mathrm { n e w } } ^ { 2 } \mathrm { t r } \bigl ( \Sigma _ { s , r } ^ { \mathrm { n e w } , ( l ) } \bigr ) } \end{array}$ and $\begin{array} { r } { \| \mu _ { s , r } ^ { ( l ) } \| ^ { 2 } = \lambda _ { s , \mathrm { o l d } } ^ { 2 } \| \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } \| ^ { 2 } + \lambda _ { s , \mathrm { n e w } } ^ { 2 } \| \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } \| ^ { 2 } + 2 \lambda _ { s , \mathrm { o l d } } \lambda _ { s , \mathrm { n e w } } \langle \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } , \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } \rangle . } \end{array}$

Corollary V.8 yields three monitorable scalar quantities. t $\cdot ( \Sigma _ { s , r } ^ { \mathrm { o l d } , ( l ) } )$ measures gradient inconsistency within the replay buffer under the current representation, with elevated values serving as an early indicator of catastrophic forgetting when replay no longer provides stable optimization anchors. $\mathrm { t r } ( \Sigma _ { s , r } ^ { \mathrm { n e w } , ( l ) } )$ indicates the intrinsic stochasticity of the new task gradients. $\langle \mu _ { s , r } ^ { \mathrm { o l d } , ( \tilde { l } ) } , \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } \rangle$ directly quantifies cross-task alignment, distinguishing positive transfer from destructive interference.

Log-Det Budget  
![](images/ebd83fa9d4d014a09259fdcb9cd84a2975701f11dafda1b1a5f297e62255f23a.jpg)  
(a) MNIST+

Alignment cos<sub>H</sub>  
![](images/f2141e1a8d0bfc8dc95fdace17c466594f0ad6f13da4a164557a743f81f6e79b.jpg)  
(b) MNIST+

Forgetting Dynamics  
![](images/d5ef84c89f43ca5903a2e7a46f86019b44f6d7741a0174b4bf3143fcde3fbbba.jpg)  
(c) MNIST+

Statistics over 20 seeds  
![](images/d1536d706783358a94b739a3aaaf14ca6b5fdf45fae6afe988d03d1162e80dcf.jpg)  
(d) Alignment Signal cos<sub>H</sub>

![](images/931d2c604de3a82b7001164cd2858465681d719808ae1a377babaa6433486370.jpg)  
(e) MNIST-

![](images/37a809065c5ba8d4a5f3a5558203992404eea15905536105417cfa8cd59aac79.jpg)  
(f) MNIST-

![](images/e969e47515660fe0b2a804497b1a1452f7a98656b20ac48fe5715a3ced10d29d.jpg)  
(g) MNIST- 100×y-axis

![](images/f7dcd40cdef2ed5a37784c91181c75039fcb1a4c37678c278946d29b36de4096.jpg)  
(h) Budget ratio under $\lambda ^ { \star }$  
Fig. 1: Decoupling Generalization Dynamics: Synergy vs. Interference on MNIST. Vertical dashed lines indicate task transitions. Top (Rotate-MNIST): Gradient alignment (cos<sub>H</sub>) stays well above its interference-regime level (1b), accompanied by stable retention (1c) and a low trajectory budget (1a). Bottom (Flip-MNIST): Persistent anti-alignment $( \cos _ { H } \approx - 1$ , 1f) coincides with a sharp growth of the trajectory budget (1e) and the onset of catastrophic forgetting (1g). (1d) cos<sub>H</sub> reliably distinguishes synergy from interference. (1h) The one-step mixing coefficient $\lambda ^ { \star }$ (Corollary V.9) reduces the local per-step budget ratio relative to fixed mixing, with a larger effect under interference; we present $\lambda ^ { \star }$ only as a local diagnostic/control signal, not as a global replay policy.

Corollary V.9 (Sensitivity-weighted local mixing). Fix any sensitivity metric $H _ { s , r } ^ { ( l ) } .$ . For $\lambda \in [ 0 , 1 ]$ , consider the mixed vector $\mu _ { s , r } ^ { ( l ) } ( \lambda ) : = \lambda \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } + ( 1 - \lambda ) \bar { \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } }$ and its metric energy $\| \mu _ { s , r } ^ { ( l ) } ( \lambda ) \| _ { H } ^ { 2 }$ . Then $\| \mu _ { s , r } ^ { ( l ) } ( \lambda ) \| _ { H } ^ { 2 }$ is minimized over $\lambda \in [ 0 , 1 ]$ by

$$
\lambda ^ { \star } = \mathrm { c l i p } _ { [ 0 , 1 ] } \left( \frac { \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } ^ { \top } H ( \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } - \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } ) } { ( \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } - \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } ) ^ { \top } H ( \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } - \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } ) } \right) ,
$$

with the convention ${ \lambda ^ { \star } = 0 \ i f \ \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } = \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } }$

Corollary V.9 derives a local mixing coefficient $\lambda ^ { \star }$ that minimizes the H-metric energy of the mixed mean update. It should therefore be interpreted as a one-step geometric diagnostic/control rule rather than as a universal globally optimal replay policy. Without an explicit current-task progress constraint, minimizing this local energy need not optimize the overal continual-learning objective. Under task alignment, the energy is relatively insensitive to the mixing ratio, permitting flexible interpolation between replay and current-task gradients. Under conflict, $\lambda ^ { \star }$ identifies the mixture that least excites the sensitive directions encoded in $H ;$ replay-heavy values should therefore be read as warning signals of severe interference rather than as standalone prescriptions.

## VI. EXPERIMENTS

In this section, we empirically evaluate the predictions made by Theorems IV.1, IV.4 and V.3 together with their corollaries. Our goal is to determine whether the bounds and diagnostics derived in Sections IV–V retain predictive power across (i) controlled settings in which ground-truth distributional objects are accessible, and (ii) standard replay benchmarks in which these objects must be estimated from finite samples.

## A. Experimental Details

a) Datasets, architectures, and replay methods: Controlled experiments use a synthetic Gaussian-mixture stream and a binary MNIST stream optimized by SGLD, both of which permit closed-form or large-sample estimation of the replay centroid $Q _ { A _ { l } , Y | i , W _ { 1 : l } }$ and of the task population $P _ { A _ { l } , Y | i , W _ { 1 : l } }$ . Standard-benchmark experiments use Split-CIFAR-100 (10 disjoint 10- class tasks) and Split-TinyImageNet (20 disjoint 10-class tasks). Three replay methods are evaluated as instantiations of the framework: experience replay (ER) [8], DER++ [48], and iCaRL [9]. All benchmark configurations use buffer capacity 2,000, 20 epochs per task, mini-batch size 64, replay batch size 64, and SGD with learning rate 0.03, momentum 0.9, and weight decay $5 \times 1 0 ^ { - 4 }$ . For every (benchmark, method) pair, we report 2 class orderings × 3 $\times 3$ random seeds = 6 independent runs.

![](images/aab9a7c0a793007650120c7465d1d3bd170bcd955180ee3375c0b67569dc7fb3.jpg)  
(a) Gap scaling

![](images/729aa8976b6a91228f59c53d564caccbe6e88ffbe87e215160db140e48c932bc.jpg)  
(b) plateau scaling  
Fig. 2: Scaling laws of the generalization gap for fixed m or n.

b) Depth family and the suffix amplification factor: The funnel study of Section IV-B additionally trains a family of residual networks, ResNet-{10, 18, 26, 34}, on Split-CIFAR-100 and Split-TinyImageNet with both ER and DER++, spanning 6 to 18 representation cuts; the 12 (method, dataset, depth) cells with ER and DER++ enter the analysis below. Theory and experiment here concern a single quantity, the Lipschitz constant of the suffix map $a _ { \ell } \mapsto { \hat { y } }$ , which sets how strongly the network above layer ℓ amplifies a representation mismatch. Theorem IV.4 bounds it in closed form by $\bar { \rho } _ { \ell } ( W )$ , a bound covered by Remark IV.6 once $\| W _ { h } \| _ { \mathrm { o p } }$ is read as the per-block Lipschitz constant. Because that worst-case product is many orders of magnitude loose in deep networks, we additionally estimate, for a depth-resolved location, a data-dependent surrogate for the same Lipschitz constant, the on-distribution Jacobian norm of $a _ { \ell } \mapsto { \hat { y } } ,$ which is a lower bound on it.

c) Performance and diagnostic observables: Let $A _ { t , i }$ denote the test accuracy on task i evaluated immediately after training on task t completes. The standard class-incremental metrics are

$$
\mathrm { A A } _ { t } \triangleq \frac { 1 } { t } \sum _ { i \leq t } A _ { t , i } , \quad \quad \mathrm { A F } _ { t } \triangleq \frac { 1 } { t - 1 } \sum _ { i < t } \left( \operatorname* { m a x } _ { s < t } A _ { s , i } - A _ { t , i } \right) , \quad t \geq 2 .
$$

Here $\mathrm { A A } _ { t }$ is the average task accuracy, and $\mathrm { A F } _ { t }$ represents the average forgetting over all previously learned tasks.

After each task transition t → t+1 we additionally compute, for every prior task $i \leq t ,$ the diagnostic quantities: the sensitivityaware gradient alignment cos (i, t) derived from the SGLD analysis of Section V, the layer-ℓ drift estimate $D _ { \ell } ( i , t )$ from the geometric relaxation of Section IV-B, the suffix sensitivity $\bar { \rho } _ { \ell } ( W _ { t } )$ from Theorem $\mathrm { I V . 4 } ,$ and the gradient-covariance traces $\mathrm { t r } \big ( \Sigma _ { \mathrm { o l d } } ( i , t ) \big )$  and $\mathrm { t r } \big ( \Sigma _ { \mathrm { n e w } } ( t ) \big )$ . The central response variable is the pairwise forgetting

$$
\Delta F _ { i , t } \triangleq A _ { t , i } - A _ { t + 1 , i } .
$$

d) Statistical methodology: Two choices make the benchmark analysis robust to the small-sample artifacts that affect transition-level studies. First, the unit of analysis is the (old-task, new-task) pair rather than the task boundary: each Split TinyImageNet run contributes 190 pairs and each Split-CIFAR-100 run 45, for 3,690 pairwise observations in total, which is two orders of magnitude more than a one-point-per-transition analysis. Second, because any quantity that drifts monotonically with task index would appear spuriously predictive, for each diagnostic we report the task-controlled partial Pearson correlation with pairwise forgetting after partialling out the new-task index and the old-task age. To respect the dependence structure of the data, the partial correlation is computed within each run (over its 45 or 190 pairs); the reported 95% bootstrap interval and sign consistency are then taken over the six runs of each cell, so the interval reflects between-run variability rather than treating non-independent pairs as independent samples. We compare each partial correlation against two naive monotone baselines: the raw correlation of forgetting with task index and with old-task age. Unless stated otherwise, the sensitivity metric $H _ { s , r } ^ { ( l ) }$ is used in its diagonal approximation, so that all diagnostics are computed in $O ( d )$ time and require no $d \times d$ matrix.

## B. Controlled Test of the Effective-Sample-Size Term

This subsection isolates the optimization-variance branch of Theorem IV.1, in which the dependence on memory size m, current-task size n, and sequence length $T$ enters through the effective sample size $\begin{array} { r } { N _ { \mathrm { e f f } } = \frac { 1 } { ( T - 1 ) / m + 1 / n } } \end{array}$ . Two scaling predictions in m and n follow directly from the prefactor $\sqrt { 2 \sigma ^ { 2 } / N _ { \mathrm { e f f } } }$ in the bound. First, for fixed $( m , T )$ , increasing the current-task size n does not drive the variance term to zero but instead yields a finite plateau $\begin{array} { r } { \operatorname* { l i m } _ { n \to \infty } N _ { \mathrm { e f f } } = \frac { m } { T - 1 } } \end{array}$ , which we refer to as the finite-memory variance floor. Second, in the regime $m \ll n , N _ { \mathrm { e f f } } \approx m / ( T - 1 )$ , so the variance term scales as $m ^ { - 1 / 2 }$ in the buffer size. The empirical gap reported below is the task-averaged discrepancy, equal to the total-risk gap of Theorem IV.1 divided by the fixed task count $T .$

![](images/ba45bb9ce80f9a31657d762bb2f2d167fea3e883c345d2a83138033e13d91de4.jpg)

![](images/86b866d04c95a0defe90cbb8ff9fa4e0ebb5b9569ced812331a7896b1707034a.jpg)  
(a) Variance proxy vs. gap over the (m, n) grid  
(b) T-sweep: gap rises while the variance proxy falls  
Fig. 3: The bound is loose in constant but order-correct, and its rate prefactor separates cleanly from its information content.

We construct a T-task Gaussian-mixture stream in which each task $\mathcal { D } _ { i }$ defines a binary classification problem between two isotropic Gaussians with task-specific means and a shared covariance. The learner is a linear-Gaussian model, for which the population risk $L ( W )$ and the empirical risk $\hat { L } ( W )$ are available in closed form, as are all KL terms appearing in Theorem IV.1. The empirical generalization gap $| L ( W ) - \hat { L } ( W ) |$ is averaged over $K = 2 0 0$ independent replications of the buffer construction and the current-task draw. Figure 2a plots the empirical gap as a function of $n$ for $T = 8$ and three values of $m \in \{ 4 0 , 8 0 , 1 6 0 , 3 2 0 \}$ . Each curve flattens as n grows, and the asymptote agrees with the prediction: doubling n in the saturated regime produces a change in the gap that is statistically indistinguishable from zero, while doubling m shifts the asymptote downward by a factor close to $\sqrt { 2 }$ . Figure 2b reports the same data in the log-log regime, together with an ordinary-least-squares fit of log gap versus log m. The fitted slope is $\widehat { \beta } = - 0 . 4 4 3 \pm 0 . 0 2 7$ (95% bootstrap confidence interval $\left[ - 0 . 4 9 8 , - 0 . 3 9 1 \right]$ $K = 2 0 0$ replications per point), so the interval lies just above $- 1 / 2 ,$ , and the estimate drifts systematically from $- 0 . 4 5 4 \ \mathrm { t o } \ - 0 . 4 2 3$ as the tail threshold used to define the plateau grows. The prediction under test is the $m ^ { - 1 / 2 }$ scaling of the variance prefactor $\sqrt { 2 \sigma ^ { 2 } / N _ { \mathrm { e f f } } }$ in isolation. However, what we fit is the full empirical gap, not the prefactor in isolation: it consists of the variance prefactor $\sqrt { 2 \sigma ^ { 2 } / N _ { \mathrm { e f f } } }$ multiplied by the square root of the information content $\mathbf { \dot { \boldsymbol { S } } } ^ { ( \ell ) } + \mathcal { P } ^ { ( \ell ) } - \mathcal { R } ^ { ( \ell ) } + \mathcal { C } ^ { ( \ell ) }$ , with a residual drift contribution that persists at finite $n .$ Any mild m-dependence of that information content shifts the composite exponent of the gap off the prefactor’ $; - 1 / 2 ;$ the measured value sitting slightly above $- 1 / 2$ and stable across tail thresholds, is consistent with such a small perturbation.

The closed-form setting lets us test a sharper claim: that the bound tracks the gap as its controlling quantities vary, not merely that it attains the correct asymptote. Writing the task-averaged variance proxy as $1 / \sqrt { N _ { \mathrm { e f f } } }$ , we first sweep the memory and sample budgets on a $4 \times 4$ grid $( m \in \{ 4 0 , 8 0 , 1 6 0 , 3 2 0 \}$ , n ∈ {500, 1000, 2000, 5000}, $T = 8 )$ . Across the 16 cells the proxy and the measured gap are not merely rank-correlated (Spearman $\rho = 0 . 9 4$ at the cell level) but essentially linear: a straight-line fit of the gap against $1 / \sqrt { N _ { \mathrm { e f f } } }$ gives $R ^ { 2 } = 0 . 9 7$ (Pearson $r = 0 . 9 8 )$ , so the proxy reproduces the functional form of the $( m , n )$ dependence, not only its ordering. The variance branch is thus quantitatively faithful in $( m , n )$ up to its loose absolute constant. A complementary T-sweep (fixed $m { = } 1 6 0 , n { = } 1 0 0 0 , T \in \{ 2 , 4 , 8 , 1 6 \}$ , 20 seeds) isolates the $N _ { \mathrm { e f f } }$ prefactor from the information content of the bound. We stress at the outset that the variance prefactor $\sqrt { 2 \sigma ^ { 2 } / N _ { \mathrm { e f f } } } = \sqrt { 2 \sigma ^ { 2 } \big ( ( T - 1 ) / m + 1 / n \big ) }$ grows with $T ,$ , as $\sqrt { T / m }$ in the total-risk gap; the quantity plotted in Fig. 3b is the task-averaged proxy $\frac { 1 } { T } \sqrt { 2 \sigma ^ { 2 } / N _ { \mathrm { e f f } } } \sim 1 / \sqrt { T m }$ , which decreases only because the fixed $1 / T$ normalization (the reported gap is the total-risk gap of Theorem IV.1 divided by T) rescales every term alike. In either convention the prefactor’s T-growth is at most $\sqrt { T } .$ , whereas the measured task-averaged gap rises from 0.0045 at $T { = } 2$ to 0.0156 at T=16—faster than the prefactor and in the opposite direction to its task-averaged proxy. The prefactor alone therefore cannot account for the $T \cdot$ -dependence. The growth is carried by the bound’s information content: the replay-centroid drift $\kappa ^ { ( \ell ) }$ , which enters the drift term with an explicit $( T - 1 )$ factor, and the optimization information $S ^ { ( \ell ) } \overset { ^ { \cdot } } { + } \bar { \mathcal { P } } ^ { ( \ell ) } - \mathcal { R } ^ { ( \ell ) }$ , whose heritage component is non-decreasing in the number of tasks by Corollary V.2. The T-sweep thus separates the finite-memory rate prefactor $N _ { \mathrm { e f f } }$ (validated by the $( m , n )$ grid above) from the information content that accumulates with T.

## C. The Drift–Sensitivity Trade-off and the Empirical Funnel

We now turn to the geometric drift component developed in Section IV-B. This subsection examines whether such a funnel is empirically observable, and whether its location is consistent across controlled and standard-benchmark settings.

![](images/3c0403049d4358f0ec9edf010f4a8fb3516633ef734f9945dba48013c92d6119.jpg)  
(a) Controlled Gaussian (closed-form populations)

![](images/d278e56101c6b8f4a97d3efd93dded2c096417e976f13d95da139045def66bc5.jpg)

![](images/5747b08adde3a88638df44573137bc6d88622ce2f6ae8085dd5f94d9ddc2dea0.jpg)  
(b) Funnel location moves deeper with network depth

![](images/19274cf4934299854e36d0c5b68e7c895e611808992333a54f234fbc2ef303bd.jpg)  
(c) ResNet-{10, 18, 26, 34}: drift, suffix amplification, and their product across depth  
Fig. 4: The drift–sensitivity trade-off and the empirical funnel. (a) On the controlled Gaussian setting with closed-form populations, the product is minimized at an interior layer $( \ell ^ { \star } = 2 )$ . (b) The same trade-off holds in deep residual networks, where suffix amplification is measured on-distribution, with the minimum appearing in a broad interior region at every depth. (c) This interior region moves systematically deeper as the network depth increases.

Three quantities are tracked at each layer index ℓ over the course of training. The drift $D _ { \ell }$ is estimated as a sliced Wasserstein-1 distance between the buffer-induced and the population-induced empirical distributions of $( A _ { \ell } , Y )$ , averaged over old tasks. Suffix sensitivity is calculated in two ways. The first option is the worst-case factor $\bar { \rho } _ { \ell } ( W )$ from Theorem IV.4. We construct this factor by taking the product of layer-wise spectral norms across layers $\ell + 1$ through L, and every operator norm is computed via power iteration. This is a certified Lipschitz bound that we use for the controlled Gaussian setting in Fig. 4a. The second option relies on the on-distribution Jacobian norm of the suffix map $a _ { \ell } \mapsto { \hat { y } } .$ . This data-driven quantity acts as a lower bound for the Lipschitz constant. We apply it to find the funnel in the deep residual networks of Fig. 4c, since the first method produces results that are too loose to be useful. We next compute the product proxy ${ \mathrm { G e n } } _ { \ell } = D _ { \ell }$ · (suffix sensitivity) for each layer, and the empirical funnel $\hat { \ell } ^ { \star }$ is determined by taking the argmin of this proxy.

Figure 4a reports the three curves on the Controlled Gaussian, where ground-truth populations $P _ { A _ { \ell } , Y | i , W _ { 1 : \ell } }$ are accessible. The drift $D _ { \ell }$ decreases first and then increases, while $\bar { \rho } _ { \ell } ( W )$ falls; their product is convex with minimizer concentrated at $\ell ^ { \star } = 2$ on a four-layer architecture, and the two ends $\ell \in \{ 0 , L \}$ are never selected. In low dimensions, where the worst-case factor $\bar { \rho } _ { \ell }$ has small dynamic range, the predicted interior minimum is directly observable.

Figure 4c extends the construction to deep residual networks—ResNet-{10, 18, 26, 34}—across both replay methods and both benchmarks, for 12 (method, dataset, depth) cells. Two findings hold uniformly. First, the drift forms a pronounced “bathtub”: $D _ { \ell }$ is large at both the input and the final representation and small across a broad interior, so the drift–amplification product is minimized strictly inside the network in every one of the 12 cells. Second, the interior minimizer moves systematically deeper as the network deepens: its absolute index increases with $L ,$ and the Spearman’s $\rho$ between funnel index and depth falls between 0.8 and 1.0 across all three method–dataset pairs (Fig. 4b). Because the product is flat across a broad basin where values within 20% of the minimum span 30–47% of the interior, we report the funnel as an interior region rather than a single sharply-resolved layer.

A feature-distillation probe that stabilizes individual blocks, consistent with Corollary IV.5, confirms that stabilizing an interior block improves over the no-distillation baseline and over stabilizing either end, but the two interior blocks nearest the basin are not statistically separated under our seed budget—as expected of a minimizer lying in a flat interior region. On real benchmarks the funnel proxy is the weakest of our diagnostics (Table I): its partial correlation with forgetting is significant and correctly signed only for ER (−0.37 and −0.30), and is weak or absent for DER++ and iCaRL. We therefore present the funnel as a structural result about the drift–sensitivity trade-off in controlled and depth-resolved settings, and not on the same footing as the alignment diagnostic of Section VI-D, whose benchmark transfer is both stronger and consistent across the gradient-replay methods.

TABLE I: Task-controlled partial Pearson correlation between theory-derived diagnostics and pairwise forgetting $\Delta F _ { i , t }$ , after partialling out the new-task index and the old-task age. Each cell aggregates 6 runs (2 class orders × 3 seeds); <sup>†</sup> marks a 95% bootstrap CI that excludes zero. The last two columns are naive monotone baselines (raw correlation of forgetting with task index / old-task age). Negative values are theory-consistent: alignment of old/new gradients reduces forgetting. “–” denotes a quantity not recorded for iCaRL under our diagnostic instrumentation, whose distillation-based objective does not expose the same replay/current gradient split as ER and DER++.
<table><tr><td>Benchmark / Method</td><td> $\mathrm { c o s } _ { H }$ </td><td> $\langle \mu _ { \mathrm { o l d } } , \mu _ { \mathrm { n e w } } \rangle$ </td><td> $\operatorname { t r } ( \Sigma _ { \mathrm { o l d } } )$ </td><td>funnel  $D _ { \ell } \bar { \rho } _ { \ell }$ </td><td>task-idx (null)</td><td>old-age (null)</td></tr><tr><td>S-CIFAR100 / ER</td><td>-0.948†</td><td>-0.915†</td><td>-0.840†</td><td>-0.372†</td><td>-0.29</td><td>-0.62</td></tr><tr><td>S-CIFAR100 / DER++</td><td>-0.794†</td><td>-0.561†</td><td>-0.185</td><td>-0.043</td><td>-0.04</td><td>-0.48</td></tr><tr><td>S-CIFAR100 / iCaRL</td><td>+0.076</td><td>+0.062</td><td></td><td>-0.088</td><td>-0.43</td><td>-0.11</td></tr><tr><td>S-TinyImageNet / ER</td><td>-0.796†</td><td>-0.281†</td><td>-0.261†</td><td>-0.296†</td><td>-0.23</td><td>-0.44</td></tr><tr><td>S-TinyImageNet / DER++</td><td>-0.645†</td><td>-0.230†</td><td>-0.098</td><td>-0.127†</td><td>-0.13</td><td>-0.41</td></tr><tr><td>S-TinyImageNet / iCaRL</td><td>+0.132†</td><td>-0.156†</td><td></td><td>+0.074†</td><td>-0.28</td><td>-0.10</td></tr></table>

## D. SGLD Optimization-Branch Diagnostics

Theorem V.3 reduces the optimization branch of Theorem IV.1 to a trajectory-level log-determinant budget, and Lemma V.4 and Proposition V.5 decompose the per-step cost into a covariance-driven instability term and a mean-driven interaction cost governed by the sensitivity-aware alignment cos . We evaluate these diagnostics at two levels: first on a controlled MNIST stream where the gradient moments admit clean estimation and the mechanism can be examined directly, and then on standard replay benchmarks where every quantity must be estimated under the noise of practical pipelines.

1) Controlled MNIST, the mechanism in isolation: On the controlled MNIST stream the gradient moments are estimable, so the decomposition of Proposition V.5 can be examined mechanism by mechanism (Fig. 1). In the positive-transfer regime (Rotate-MNIST), the per-run mean alignment $\mathrm { c o s } _ { H }$ sits well above its interference-regime level (≈ −0.5 versus ≈ −0.85 on the box-plots of Fig. 1(d)), the per-step interaction cost and the gradient covariance remain small, and retention is stable. In the interference regime (Flip-MNIST), cos<sub>H</sub> collapses to strongly negative values (per-run mean ≈ −0.85), the log-determinant trajectory budget grows sharply, and forgetting rises. The alignment signal cleanly separates the two regimes across 20 seed (Fig. 1(d)), consistent with Corollary V.7 read as a statement about relative alignment: the synergy regime sustains a markedly higher (less negative) cos than the interference regime. We read the corollary’s sign prediction in this relative sense, since under the diagonal-H approximation the per-run mean $\mathrm { c o s } _ { H }$ is negative in both regimes; it is their separation, not an absolute sign, that the corollary ties to constructive versus destructive interaction and that predicts forgetting. We note one diagnostic caveat. Under severe interference the model collapses toward chance, so once train and test risks reach the same entropy floor the terminal forgetting metric saturates and becomes uninformative; the trajectory budget and $\mathrm { c o s } _ { H }$ remain informative precisely in this regime, which is why we treat them as the primary diagnostics rather than terminal forgetting alone.

a) The curvature-aware metric is not the Euclidean inner product: A natural objection is that cos merely re-expresses the Euclidean gradient cosine $\mathrm { c o s } _ { E }$ used by orthogonal-projection methods (GEM/OGD). The controlled streams refute this directly: because both quantities are computed at every step from the same replay and current-task gradients, the comparison is not a post-hoc re-instrumentation. On Rotate/Flip-MNIST, cos separates the synergy and interference regimes more sharpl than cos (standardized between-regime mean difference 3.8 vs. 3.0) and predicts per-run forgetting more strongly (−0.89 vs. −0.84); the same ordering holds on the synthetic Gaussian stream (cos separation 2.9 vs. 2.0, forgetting correlation −0.64 vs. −0.55). These gaps are statistically reliable rather than artifacts of the seed budget: a paired bootstrap over runs places the cos − cos difference above zero with a 95% CI that excludes zero for both the regime separation (∆d CI [0.7, 1.4 on MNIST) and the forgetting correlation (∆|r| CI [0.02, 0.09] on MNIST and [0.01, 0.16] on the synthetic stream). The improvement is modest but consistent and significant, just as the theory predicts. The curvature reweighting $H = ( I + \alpha V ) ^ { - 1 }$ discounts alignment along unstable directions (Proposition V.5). Consequently, cos is an intrinsic measure derived from ou bound, not merely a hidden variant of the Euclidean OGD cosine.

2) Standard benchmarks: do the diagnostics track forgetting?: We now test the central empirical claim of the optimization branch: that the abstract interaction term $\mathcal { R } ^ { ( l ) }$ , made operational through the alignment $\mathrm { c o s } _ { H }$ of Corollary V.7, tracks forgetting once estimated on standard replay benchmarks. Table I and Fig. 6 report the task-controlled partial correlations of the diagnostic with pairwise forgetting across all six (benchmark, method) cells.

a) Alignment predicts forgetting for gradient-replay methods: For the two methods whose updates are literally a mixture of replay and current-task gradients—ER and DER++—the alignment term is a strong and stable predictor of forgetting on both benchmarks. The partial correlation of cos with pairwise forgetting is −0.948 (ER) and −0.794 (DER++) on Split-CIFAR-100, and −0.796 (ER) and −0.645 (DER++) on Split-TinyImageNet, with confidence intervals that exclude zero and the same sign in all six runs of every cell. The sign is exactly the one the theory predicts: when old- and new-task gradients align in the stability-aware metric H, forgetting is small; when they anti-align, forgetting is large. Crucially, this is not an artifact of generic task-order drift. The task-controlled cos<sub>H</sub> signal is several times stronger than the two naive monotone baselines (the raw correlation of forgetting with task index never exceeds 0.29 in magnitude, and with old-task age never exceeds 0.62), confirming that the predictive content survives after the dominant monotone trends are removed. The raw alignment inner product $\langle \mu _ { \mathrm { o l d } } , \mu _ { \mathrm { n e w } } \rangle$ and, for ER, the replay-gradient inconsistency $\mathrm { t r } ( \Sigma _ { \mathrm { o l d } } )$ (−0.840 on Split-CIFAR-100) are theory-consistent as well, while the geometric funnel proxy transfers more weakly but with the correct sign for ER on both benchmarks (−0.372 and −0.296).

![](images/ece4b7a154a0406776839ab454d99569e943a55f67e91377b86b0086c944b4c9.jpg)  
(a) Regime separation: cos<sub>H</sub> vs. cos<sub>E</sub>

![](images/bf0d323b4d0b4875ddbccf038f59003d49b3684e4711c0b146383614ef1e508b.jpg)  
(b) Forgetting prediction: cos<sub>H</sub> vs. cos<sub>E</sub>

Fig. 5: The sensitivity-aware alignment cos<sub>H</sub> improves on the Euclidean gradient cosine $\mathrm { c o s } _ { E }$ (the OGD/GEM quantity), both logged from identical gradients. (a) cos<sub>H</sub> separates synergy from interference more sharply (larger between-regime effect size). (b) cos<sub>H</sub> is a stronger predictor of per-run forgetting. Rotate/Flip-MNIST; the synthetic Gaussian stream shows the same ordering.  
![](images/4eeddd5726199be81b8b18d88d3ed81067e5b24bbd5a81a35951c7ced623b4a0.jpg)  
Fig. 6: Task-controlled partial Pearson correlation of cos with pairwise forgetting on the six (benchmark, method) cells, with 95% bootstrap intervals. Blue: significant and theory-consistent (CI excludes zero, negative sign); orange: weak or signinconsistent (iCaRL). Grey ticks are the two naive monotone baselines (raw correlation of forgetting with task index and with old-task age). For ER and DER++ the alignment signal is several times stronger than either monotone baseline, on both benchmarks.

b) A scope boundary, stated honestly: The iCaRL columns are included deliberately as a boundary of the framework rather than omitted. iCaRL classifies by nearest-mean-of-exemplars and is trained with knowledge distillation, so its forgetting mechanism departs from the replay-plus-current gradient mixture that the SGLD model of Section V assumes. Consistently, cos is no longer a reliable predictor for iCaRL: its CI includes zero on Split-CIFAR-100 and the sign even reverses on Split-TinyImageNet. The raw alignment $\langle \mu _ { \mathrm { o l d } } , \mu _ { \mathrm { n e w } } \rangle$ still retains the theory-consistent negative sign after partialling (−0.156 on Split-TinyImageNet), which is what one would expect if only the part of iCaRL’s update that resembles a gradient step carries the alignment signal. This is the honest reading of the framework: the gradient-alignment diagnostic is most faithful precisely for methods whose optimization matches the modeled dynamics, and weaker for exemplar-classifier methods—a delimitation the theory itself anticipates, rather than a failure to be hidden

c) Correlation with actual old-task test error: Finally, the geometric branch also transfers to actual test error rather than only to the forgetting increment. Merging the diagnostic logs with the accuracy logs on Split-CIFAR-100 with iCaRL, the layer-wise drift and the funnel proxy correlate with current old-task test error at +0.262 (95% CI [0.132, 0.392]) and +0.257 ([0.128, 0.385]) respectively. Taken together, the benchmark study addresses its core motivating question of whether the quantities appearing in the bound retain meaning when estimated under realistic conditions. It delivers affirmative conclusions for the alignment term and gradient-replay methods, while clearly defining the limitations of the diagnostics

## VII. LIMITATIONS

We do not claim that the bounds are numerically tight for deep networks: as with all mutual-information generalization bounds, the absolute multiplicative constants are loose. What our experiments validate is sharper and, we argue, more useful— three falsifiable predictions about the structure of the gap. First, the predicted functional form: the variance floor scales as $m ^ { - 1 / 2 }$ , the measured exponent is −0.443 (CI [−0.498, −0.391]), and the bound is order-correct beyond this asymptote: across a memory–sample grid, its variance proxy ranks the measured gap with Spearman 0.94. Second, the predicted geometry: the drift–sensitivity trade-off, with suffix sensitivity measured on-distribution, is minimized strictly inside the network in all 12 (method, dataset, depth) cells we examine, and the interior region moves systematically deeper as depth grows; we claim an interior region and its depth trend, located empirically rather than certified by the worst-case Lipschitz factor, which is itself too loose to localize a funnel in deep networks. Third, the predicted monotone relationship: the alignment term that the bound makes responsible for the interaction component negatively tracks forgetting on real benchmarks, with partial correlations from −0.65 to −0.95 for gradient-replay methods. A bound can be loose in its constants yet correct in the dependencies it predicts; our experiments target the latter.

## VIII. CONCLUSION

We developed a layer-wise information-theoretic theory of replay generalization in continual learning. The central message is that replay generalization is governed by two coupled mechanisms: a replay-induced representation drift created by finite memory, and an optimization-complexity term created by shared downstream parameters. The main Synergy–Drift theorem isolates these two mechanisms in a single decomposition. The Wasserstein result revisits the first branch when KL-based drift is ill-posed under support mismatch, while the SGLD result instantiates the second branch through trajectory-level gradient moments. Together, these results provide a common language for studying replay bias, task interaction, and layer-wise sensitivity in continual learning.

## REFERENCES

[1] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska et al., “Overcoming catastrophic forgetting in neural networks,” Proceedings of the national academy of sciences, vol. 114, no. 13, pp. 3521–3526, 2017.

[2] G. I. Parisi, R. Kemker, J. L. Part, C. Kanan, and S. Wermter, “Continual lifelong learning with neural networks: A review,” Neural networks, vol. 113, pp. 54–71, 2019.

[3] M. De Lange, R. Aljundi, M. Masana, S. Parisot, X. Jia, A. Leonardis, G. Slabaugh, and T. Tuytelaars, “A continual learning survey: Defying forgetting in classification tasks,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 7, pp. 3366–3385, 2021.

[4] M. McCloskey and N. J. Cohen, “Catastrophic interference in connectionist networks: The sequential learning problem,” in Psychology of learning and motivation. Elsevier, 1989, vol. 24, pp. 109–165.

[5] I. J. Goodfellow, M. Mirza, D. Xiao, A. Courville, and Y. Bengio, “An empirical investigation of catastrophic forgetting in gradient-based neura networks,” ArXiv Preprint ArXiv:1312.6211, 2013.

[6] T.-C. Kao, K. Jensen, G. van de Ven, A. Bernacchia, and G. Hennequin, “Natural continual learning: success is a journey, not (just) a destination,” Advances in Neural Information Processing Systems, vol. 34, pp. 28 067–28 079, 2021.

[7] Y.-S. Liang and W.-J. Li, “Adaptive plasticity improvement for continual learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 7816–7825.

[8] D. Rolnick, A. Ahuja, J. Schwarz, T. Lillicrap, and G. Wayne, “Experience replay for continual learning,” Advances in neural information processing systems, vol. 32, 2019.

[9] S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert, “icarl: Incremental classifier and representation learning,” in Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2017, pp. 2001–2010.

[10] Y. Guo, B. Liu, and D. Zhao, “Online continual learning through mutual information maximization,” in International Conference on Machine Learning. PMLR, 2022, pp. 8109–8126.

[11] M. Riemer, I. Cases, R. Ajemian, M. Liu, I. Rish, Y. Tu, and G. Tesauro, “Learning to learn without forgetting by maximizing transfer and minimizing interference,” in International Conference on Learning Representations, 2019.

[12] H. Shi and H. Wang, “A unified approach to domain incremental learning with memory: Theory and algorithm,” Advances in Neural Information Processing Systems, vol. 36, pp. 15 027–15 059, 2023.

[13] Y. Li and E.-J. van Kampen, “Stability analysis for incremental adaptive dynamic programming with approximation errors,” Journal of Aerospace Engineering, vol. 37, no. 1, p. 04023097, 2024.

[14] J. Knoblauch, H. Husain, and T. Diethe, “Optimal continual learning has perfect memory and is np-hard,” in International Conference on Machine Learning. PMLR, 2020, pp. 5327–5337.

[15] A. Xu and M. Raginsky, “Information-theoretic analysis of generalization capability of learning algorithms,” Advances in neural information processing systems, vol. 30, 2017.

[16] T. Steinke and L. Zakynthinou, “Reasoning about generalization via conditional mutual information,” in Conference on Learning Theory. PMLR, 2020, pp. 3437–3452.

[17] M. Haghifam, J. Negrea, A. Khisti, D. M. Roy, and G. K. Dziugaite, “Sharpened generalization bounds based on conditional mutual information and an application to noisy, iterative algorithms,” Advances in Neural Information Processing Systems, vol. 33, pp. 9925–9935, 2020.

[18] H. He, C. L. Yu, and Z. Goldfeld, “Hierarchical generalization bounds for deep neural networks,” in 2024 IEEE International Symposium on Information Theory (ISIT). IEEE, 2024, pp. 2688–2693.

[19] R. M. French, “Catastrophic forgetting in connectionist networks,” Trends in cognitive sciences, vol. 3, no. 4, pp. 128–135, 1999.

[20] G. M. van de Ven, N. Soures, and D. Kudithipudi, “Continual learning and catastrophic forgetting,” arXiv preprint arXiv:2403.05175, 2024.

[21] L. Wang, X. Zhang, H. Su, and J. Zhu, “A comprehensive survey of continual learning: Theory, method and application,” IEEE transactions on pattern analysis and machine intelligence, vol. 46, no. 8, pp. 5362–5383, 2024.

[22] F. Zenke, B. Poole, and S. Ganguli, “Continual learning through synaptic intelligence,” in International conference on machine learning. PMLR, 2017, pp. 3987–3995.

[23] R. Aljundi, F. Babiloni, M. Elhoseiny, M. Rohrbach, and T. Tuytelaars, “Memory aware synapses: Learning what (not) to forget,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 139–154.

[24] A. A. Rusu, N. C. Rabinowitz, G. Desjardins, H. Soyer, J. Kirkpatrick, K. Kavukcuoglu, R. Pascanu, and R. Hadsell, “Progressive neural networks,” arXiv preprint arXiv:1606.04671, 2016.

[25] A. Mallya and S. Lazebnik, “Packnet: Adding multiple tasks to a single network by iterative pruning,” in Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2018, pp. 7765–7773.

[26] H. Shin, J. K. Lee, J. Kim, and J. Kim, “Continual learning with deep generative replay,” Advances in neural information processing systems, vol. 30, 2017.

[27] A. Chrysakis and M.-F. Moens, “Online continual learning from imbalanced data,” in International Conference on Machine Learning. PMLR, 2020, pp. 1952–1961.

[28] S. Sun, D. Calandriello, H. Hu, A. Li, and M. Titsias, “Information-theoretic online memory selection for continual learning,” arXiv preprint arXiv:2204.04763, 2022.

[29] L. Caccia, R. Aljundi, N. Asadi, T. Tuytelaars, J. Pineau, and E. Belilovsky, “New insights on reducing abrupt representation change in online continual learning,” in International Conference on Learning Representations, 2022.

[30] R. Aljundi, M. Lin, B. Goujaud, and Y. Bengio, “Gradient based sample selection for online continual learning,” Advances in neural information processing systems, vol. 32, 2019.

[31] D. Lopez-Paz and M. Ranzato, “Gradient episodic memory for continual learning,” Advances in neural information processing systems, vol. 30, 2017.

[32] A. Chaudhry, M. Ranzato, M. Rohrbach, and M. Elhoseiny, “Efficient lifelong learning with a-gem,” arXiv preprint arXiv:1812.00420, 2018.

[33] G. Saha, I. Garg, and K. Roy, “Gradient projection memory for continual learning,” arXiv preprint arXiv:2103.09762, 2021.

[34] S. Lin, P. Ju, Y. Liang, and N. Shroff, “Theory on forgetting and generalization of continual learning,” in International Conference on Machine Learning. PMLR, 2023, pp. 21 078–21 100.

[35] M. Ding, K. Ji, D. Wang, and J. Xu, “Understanding forgetting in continual learning with linear regression,” arXiv preprint arXiv:2405.17583, 2024.

[36] W. Wen, T. Gong, Z. Gao, Y. Zhang, W. Zhang, and Y.-J. Liu, “Information-theoretic generalization bounds of replay-based continual learning,” 2026 [Online]. Available: https://arxiv.org/abs/2507.12043

[37] J. Negrea, M. Haghifam, G. K. Dziugaite, A. Khisti, and D. M. Roy, “Information-theoretic generalization bounds for sgld via data-dependent estimates,” Advances in Neural Information Processing Systems, vol. 32, 2019.

[38] Y. Dong, T. Gong, H. Chen, and C. Li, “Understanding the generalization ability of deep learning algorithms: a kernelized renyi’s entropy perspective,”´ arXiv preprint arXiv:2305.01143, 2023.

[39] S. Sonoda, Y. Hashimoto, I. Ishikawa, and M. Ikeda, “Why and when deep is better than shallow: An implementation-agnostic state-transition view of depth supremacy,” 2025. [Online]. Available: https://arxiv.org/abs/2505.15064

[40] G. Peyre, M. Cuturi´ et al., “Computational optimal transport: With applications to data science,” Foundations and Trends® in Machine Learning, vol. 11, no. 5-6, pp. 355–607, 2019.

[41] I. Redko, A. Habrard, and M. Sebban, “Theoretical analysis of domain adaptation with optimal transport,” in Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, 2017, pp. 737–753.

[42] A. T. Lopez and V. Jog, “Generalization error bounds using wasserstein distances,” in 2018 ieee information theory workshop (ITW). IEEE, 2018, pp. 1–5.

[43] H. Wang, M. Diaz, J. C. S. Santos Filho, and F. P. Calmon, “An information-theoretic view of generalization via wasserstein distance,” in 2019 IEEE international symposium on information theory (ISIT). IEEE, 2019, pp. 577–581.

[44] A. Douillard, M. Cord, C. Ollion, T. Robert, and E. Valle, “Podnet: Pooled outputs distillation for small-tasks incremental learning,” in European Conference on Computer Vision. Springer, 2020, pp. 86–102.

[45] Y. Shen, S. Dasgupta, and S. Navlakha, “Reducing catastrophic forgetting with associative learning: a lesson from fruit flies,” Neural Computation, vol. 35, no. 11, pp. 1797–1819, 2023.

[46] A. Sorrenti, G. Bellitto, F. P. Salanitri, M. Pennisi, C. Spampinato, and S. Palazzo, “Selective freezing for efficient continual learning,” in 2023 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW). IEEE Computer Society, 2023, pp. 3542–3551.

[47] Q. Chen, C. Shui, L. Han, and M. Marchand, “On the stability-plasticity dilemma in continual meta-learning: Theory and algorithm,” Advances in Neural Information Processing Systems, vol. 36, pp. 27 414–27 468, 2023.

[48] P. Buzzega, M. Boschini, A. Porrello, D. Abati, and S. Calderara, “Dark experience for general continual learning: a strong, simple baseline,” Advances in Neural Information Processing Systems, vol. 33, pp. 15 920–15 930, 2020.

[49] R. M. Gray, Entropy and information theory. Springer Science & Business Media, 2011.

[50] H. Harutyunyan, M. Raginsky, G. Ver Steeg, and A. Galstyan, “Information-theoretic generalization bounds for black-box learning algorithms,” Advances in Neural Information Processing Systems, vol. 34, pp. 24 670–24 682, 2021.

[51] Y. Dong, T. Gong, H. Chen, Z. He, M. Li, S. Song, and C. Li, “Towards generalization beyond pointwise learning: A unified information-theoretic perspective,” in Forty-first International Conference on Machine Learning, 2024.

# APPENDIX A PREREQUISITE DEFINITIONS AND LEMMAS

Definition A.1 (σ-sub-Gaussian). A random variable X is σ-sub-Gaussian if for any λ, ln $\mathbb { E } [ e ^ { \lambda ( X - \mathbb { E } X ) } ] \le \lambda ^ { 2 } \sigma ^ { 2 } / 2$

Definition A.2 (Kullback-Leibler Divergence). Let P and Q be probability measures on a measurable space $( \mathcal { X } , \Sigma )$ . If $P$ is absolutely continuous with respect to $Q$ (denoted $P \ll Q )$ , the Kullback-Leibler (KL) divergence from $P$ to Q is defined as: $\begin{array} { r } { D _ { \mathrm { K L } } ( P \| Q ) \triangleq \int _ { \mathcal { X } } \log \left( \frac { \mathrm { d } P } { \mathrm { d } Q } \right) \mathrm { d } P , } \end{array}$ where $\textstyle { \frac { \mathrm { d } P } { \mathrm { d } Q } }$ denotes the Radon-Nikodym derivative.

Definition A.3 (Mutual Information). Let X and Y be random variables taking values in measurable spaces X and $\mathcal { V } ,$ respectively. Let $P _ { X , Y }$ denote their joint distribution on the product space $\mathcal { X } \times \mathcal { V }$ , and let $P _ { X }$ and $P _ { Y }$ denote the corresponding marginal distributions. The mutual information between X and Y is defined as the KL divergence between the joint distribution and the product of the marginals: $I ( X ; Y ) \triangleq D _ { \mathrm { K L } } ( P _ { X , Y } \Vert P _ { X } \otimes P _ { Y } )$

Lemma A.4 (Interaction information identity). For any random variables $A , B , C , D ,$

$$
I ( A , B ; C \mid D ) = I ( A ; C \mid D ) + I ( B ; C \mid D ) - I ( A ; B ; C \mid D ) .
$$

Definition A.5 (Wasserstein Distance). Let P and Q be probability measures on $\mathcal { X } \subseteq \mathbb { R } ^ { k }$ . The Wasserstein-p distance $( p \ge 1 )$ is defined as: $\begin{array} { r } { \mathcal { W } _ { p } ( P , Q ) \stackrel { \Delta } { = } \left( \operatorname* { i n f } _ { \gamma \in \Gamma ( P , Q ) } \mathbb { E } _ { ( x , x ^ { \prime } ) \sim \gamma } \left[ \| x - x ^ { \prime } \| ^ { p } \right] \right) ^ { 1 / p } } \end{array}$ , where $\Gamma ( P , Q )$ is the set of all joint distributions on $\mathcal { X } \times \mathcal { X }$ with marginals $P$ and $Q ,$ , and $\| \cdot \|$ denotes the Euclidean norm.

Lemma A.6 (Donsker-Varadhan formula(Theorem 5.2.1 in [49])). Let P and $Q$ be probability measures on X such that $P \ll Q .$ . Then:

$$
D _ { \mathrm { K L } } ( P \Vert Q ) = \operatorname* { s u p } _ { f \in L _ { \infty } ( \mathcal { X } ) } \left\{ \mathbb { E } _ { P } [ f ] - \log \mathbb { E } _ { Q } \left[ e ^ { f } \right] \right\} ,
$$

where the supremum is taken over the set of all bounded measurable functions $f : \mathcal { X } \to \mathbb { R } . \ I f X$ is σ-sub-Gaussian under $Q ,$ then

$$
| \mathbb { E } _ { P } [ X ] - \mathbb { E } _ { Q } [ X ] | \leq \sigma { \sqrt { 2 D _ { \mathrm { K L } } ( P \| Q ) } } .
$$

Lemma A.7 (Lemma 1 in [50]). Let $( X , Y ) \sim P _ { X Y }$ and let $\bar { Y }$ be an independent copy of Y such tha $( X , { \bar { Y } } ) \sim P _ { X } \otimes P _ { Y }$ If the function $f ( X , { \bar { Y } } )$ is σ-sub-gaussian with respect to the distribution $P _ { X } \otimes P _ { Y }$ , then:

$$
\left| \mathbb { E } [ f ( X , Y ) ] - \mathbb { E } [ f ( X , { \bar { Y } } ) ] \right| \leq { \sqrt { 2 \sigma ^ { 2 } I ( X ; Y ) } } .
$$

Lemma A.8 (Kantorovich-Rubinstein Duality). Let P and Q be probability measures defined on a metric space $( \mathcal { X } , c )$ . Then, the 1-Wasserstein distance is given by

$$
{ \mathcal W } _ { 1 } ( P , Q ) = \operatorname* { s u p } _ { f \in \mathrm { L i p } _ { 1 } } \left\{ \int _ { \mathcal X } f { \mathrm { d } } P - \int _ { \mathcal X } f { \mathrm { d } } Q \right\} ,
$$

where Lip denotes the set of 1-Lipschitz functions with respect to the metric c, i.e., $| f ( x ) - f ( x ^ { \prime } ) | \leq c ( x , x ^ { \prime } )$ for any $f \in \operatorname { L i p } _ { 1 }$ and $x , x ^ { \prime } \in { \mathcal { X } }$

Lemma A.9 (Lemma A.11 in [51]). Let $X \sim { \mathcal { N } } ( 0 , \Sigma )$ and let Y be any zero-mean random vector with ${ \mathrm { C o v } } [ Y ] = \Sigma .$ . Then $h ( Y ) \leq h ( X )$

Lemma A.10 (Entropy upper bound under second-moment constraint). Let $X \in \mathbb { R } ^ { d }$ be any random vector with finite second moment matrix $M : = \mathbb { E } [ X X ^ { \top } ] \in \mathbb { R } ^ { d \times d }$ . Then the differential entropy satisfies

$$
h ( X ) \leq { \frac { 1 } { 2 } } \log { \Big ( } ( 2 \pi e ) ^ { d } \operatorname* { d e t } ( M ) { \Big ) } .
$$

Proof. Let $\mu : = \mathbb { E } [ X ]$ and ${ \widetilde { X } } : = X - \mu$ . Since entropy is translation-invariant, $h ( X ) = h ( { \widetilde { X } } )$ . Let $\Sigma : = \operatorname { C o v } ( X ) = \mathbb { E } [ \widetilde { X } \widetilde { X } ^ { \top } ]$ By Lemma A.9, we have

$$
h ( \widetilde { X } ) \leq \frac { 1 } { 2 } \log \Big ( ( 2 \pi e ) ^ { d } \operatorname* { d e t } ( \Sigma ) \Big ) .
$$

Next note that

$$
M - \Sigma = \mathbb { E } [ X X ^ { \top } ] - \mathbb { E } [ \widetilde { X } \widetilde { X } ^ { \top } ] = \mu \mu ^ { \top } \succeq 0 ,
$$

$$
h ( X ) = h ( \widetilde X ) \le \frac 1 2 \log \Big ( ( 2 \pi e ) ^ { d } \operatorname* { d e t } ( \Sigma ) \Big ) \le \frac 1 2 \log \Big ( ( 2 \pi e ) ^ { d } \operatorname* { d e t } ( M ) \Big ) .
$$

hence $M \succeq \Sigma \succeq 0$ . Since determinant is monotone under the Loewner order on PSD matrices, det $( M ) \geq \operatorname* { d e t } ( \Sigma )$ , so

## APPENDIX B OMITTED PROOFS

## A. Proof of Theorem IV.1

Theorem IV.1 (Restate). Let W be the output of a learner minimizing the empirical risk $\hat { \mathcal { L } } ( W )$ . Assume the loss function $\ell ( W , Z )$ is σ-subgaussian for all $Z \in { \mathcal { Z } }$ and $W \in { \mathcal { W } }$ . For any split layer $l \in \{ 0 , \ldots , L \}$ , we have

$$
\left. \mathrm { g e n } _ { W } \right. \leq \underbrace { \left( T - 1 \right) \sqrt { 2 \sigma ^ { 2 } K ^ { \left( l \right) } } } _ { \mathrm { R e p l a y - c e n t r o i d ~ D r i f t } } + \underbrace { \sqrt { \frac { 2 \sigma ^ { 2 } } { N _ { \mathrm { e f f } } } \left( S ^ { \left( l \right) } + \mathcal { P } ^ { \left( l \right) } - \mathcal { R } ^ { \left( l \right) } + \mathcal { C } ^ { \left( l \right) } \right) } } _ { \mathrm { O p t i m i z a t i o n ~ V a r i a n c e } } ,
$$

where $\begin{array} { r } { N _ { \mathrm { e f f } } \triangleq \frac { 1 } { ( T - 1 ) / m + 1 / n } } \end{array}$ is the effective sample size and $\begin{array} { r } { \mathcal { K } ^ { ( l ) } \triangleq \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } \mathbb { E } _ { W _ { 1 : l } } \left[ D _ { \mathrm { K L } } \left( Q _ { A _ { l } , Y | i , W _ { 1 : l } } \big | \big | P _ { A _ { l } , Y | i , W _ { 1 : l } } \right) \right] } \end{array}$ , and $\mathcal { C } ^ { ( l ) } \triangleq \mathbb { E } _ { W _ { 1 : l } } \left[ D _ { \mathrm { K L } } \left( P _ { \mathbf { U } ^ { ( l ) } | W _ { 1 : l } } \big | \big | \widetilde { Q } _ { \mathbf { U } ^ { ( l ) } | W _ { 1 : l } } \right) \right]$

Proof. Fix a layer $l \in \{ 0 , \ldots , L \}$ . Recall that $a _ { 0 } ( x ) = x$ and, for $h \in \{ 1 , \ldots , L \} , a _ { h } ( x ) = \phi _ { h } ( W _ { h } a _ { h - 1 } ( x ) )$ . We also adopt the boundary convention that $g _ { W _ { L + 1 : L } }$ is the identity map when $l = L$ . To make the layer-l representation parametrization unambiguous, we explicitly decompose the network at layer l. Define the top sub-network mapping from the layer-l representation to logits as

$$
g _ { W _ { l + 1 : L } } : \mathbb { R } ^ { d _ { l } }  \mathbb { R } ^ { K } , \qquad f _ { W } ( x ) = g _ { W _ { l + 1 : L } } ( a _ { l } ( x ) ) .
$$

Accordingly, define the loss as a function of the layer-l representation–label pair:

$$
F _ { W } ( t , y ) \triangleq \ell \big ( g _ { W _ { l + 1 : L } } ( t ) , y \big ) , \qquad \mathrm { s o ~ t h a t } \qquad \ell ( W , ( X , Y ) ) = F _ { W } ( a _ { l } ( X ) , Y ) .
$$

By definition,

$$
\mathrm { g e n } _ { W } = \mathbb { E } _ { W , D ^ { 1 : T } } [ \mathcal { L } ( W ) - \hat { \mathcal { L } } ( W ) ] .
$$

Expand $\mathcal { L } ( W )$ and $\hat { \mathcal { L } } ( W )$ :

$$
\mathcal { L } ( W ) = \sum _ { t = 1 } ^ { T } \mathbb { E } _ { Z \sim \mathcal { D } _ { t } } [ \ell ( W , Z ) ] , \qquad \hat { \mathcal { L } } ( W ) = \sum _ { i = 1 } ^ { T - 1 } \frac { 1 } { m } \sum _ { j \in \mathcal { Z } _ { i } } \ell ( W , Z _ { j } ^ { i } ) + \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \ell ( W , Z _ { j } ^ { T } ) .
$$

Therefore,

$$
\mathcal { L } ( W ) - \hat { \mathcal { L } } ( W ) = \sum _ { i = 1 } ^ { T - 1 } \left( \mathbb { E } _ { Z \sim \mathcal { D } _ { i } } [ \ell ( W , Z ) ] - \frac { 1 } { m } \sum _ { j \in \mathcal { T } _ { i } } \ell ( W , Z _ { j } ^ { i } ) \right) \ + \ \left( \mathbb { E } _ { Z \sim \mathcal { D } _ { T } } [ \ell ( W , Z ) ] - \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \ell ( W , Z _ { j } ^ { T } ) \right) \ .\tag{8}
$$

For each old task $i \in [ T - 1 ]$ , define the layer-l population risk and replay empirical risk

$$
L _ { i } ( W ) \triangleq \mathbb { E } _ { ( A _ { l } , Y ) \sim P _ { A _ { l } , Y \mid i , W _ { 1 : l } } } [ F _ { W } ( A _ { l } , Y ) ] , \qquad \widehat { L } _ { i } ( W ) \triangleq \frac { 1 } { m } \sum _ { j \in \mathcal { T } _ { i } } F _ { W } ( A _ { j , l } ^ { i } , Y _ { j } ^ { i } ) .
$$

Then the old-task block in (8) can be written as

$$
\begin{array} { r l } & { \Delta _ { \mathrm { o l d } } \triangleq \displaystyle \sum _ { i = 1 } ^ { T - 1 } \left( \mathbb { E } _ { Z \sim \mathcal { D } _ { i } } [ \ell ( W , Z ) ] - \frac { 1 } { m } \sum _ { j \in \mathcal { T } _ { i } } \ell ( W , Z _ { j } ^ { i } ) \right) } \\ & { \qquad = \displaystyle \sum _ { i = 1 } ^ { T - 1 } \Big ( L _ { i } ( W ) - \widehat { L } _ { i } ( W ) \Big ) . } \end{array}\tag{9}
$$

For each old task $i \in [ T - 1 ]$ , recall the task-wise replay proxy $\hat { P } _ { A _ { l } , Y \mid S ^ { i } , W _ { 1 : l } }$ and its conditional centroid law

$$
\begin{array} { r } { Q _ { A _ { l } , Y | i , W _ { 1 : l } } \ \triangleq \ \mathbb { E } \left[ \hat { P } _ { A _ { l } , Y | S ^ { i } , W _ { 1 : l } } \ \Big | \ W _ { 1 : l } \right] . } \end{array}\tag{10}
$$

Add and subtract $\mathbb { E } _ { ( A _ { l } , Y ) \sim Q _ { A _ { l } , Y \mid i , W _ { 1 : l } } } [ F _ { W } ( A _ { l } , Y ) ]$ inside each $L _ { i } ( W ) - \widehat { L } _ { i } ( W )$ in (9) to obtain

$$
\Delta _ { \mathrm { o l d } } = \underbrace { \sum _ { i = 1 } ^ { T - 1 } \left( \mathbb { E } _ { P _ { A _ { l } , Y \mid i } , w _ { 1 : l } } \left[ F _ { W } \right] - \mathbb { E } _ { Q _ { A _ { l } } , Y \mid i , W _ { 1 : l } } \left[ F _ { W } \right] \right) } _ { \triangleq \mathrm { ~ \Delta _ d ~ e n t ~ } } + \underbrace { \sum _ { i = 1 } ^ { T - 1 } \left( \mathbb { E } _ { Q _ { A _ { l } , Y \mid i } , w _ { 1 : l } } \left[ F _ { W } \right] - \widehat { L } _ { i } ( W ) \right) } _ { \triangleq \mathrm { ~ \Delta _ { o l d , v a r } ^ { ( l ) } ~ } } .\tag{11}
$$

For the new task block in (8), define

$$
\Delta _ { \mathrm { n e w , v a r } } ^ { ( l ) } \triangleq \left( \mathbb { E } _ { Z \sim \mathcal { D } _ { T } } [ \ell ( W , Z ) ] - \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \ell ( W , Z _ { j } ^ { T } ) \right) = \left( \mathbb { E } _ { P _ { A _ { l } , Y | T , W _ { 1 : l } } } [ F _ { W } ( A _ { l } , Y ) ] - \frac { 1 } { n } \sum _ { j = 1 } ^ { n } F _ { W } ( A _ { j , l } ^ { T } , Y _ { j } ^ { T } ) \right) ,
$$

where $P _ { A _ { l } , Y | T , W _ { 1 : l } }$ is the law of $( a _ { l } ( X ) , Y )$ under $Z = ( X , Y ) \sim { \mathcal { D } } _ { T }$ . Combining (8) and (11) yields the exact decomposition

$$
\begin{array} { r } { \mathcal { L } ( W ) - \hat { \mathcal { L } } ( W ) = \Delta _ { \mathrm { d r i f t } } ^ { ( l ) } + \Delta _ { \mathrm { o l d , v a r } } ^ { ( l ) } + \Delta _ { \mathrm { n e w , v a r } } ^ { ( l ) } . } \end{array}\tag{12}
$$

Let $\Delta _ { \mathrm { v a r } } ^ { ( l ) } \triangleq \Delta _ { \mathrm { o l d , v a r } } ^ { ( l ) } + \Delta _ { \mathrm { n e w , v a r } } ^ { ( l ) }$ . Taking expectation and using $| a + b | \leq | a | + | b |$ gives

$$
\left. \mathrm { g e n } _ { W } \right. = \left. \mathbb { E } [ \mathcal { L } ( W ) - \hat { \mathcal { L } } ( W ) ] \right. \leq \mathbb { E } \left[ \big \vert \Delta _ { \mathrm { d r i f t } } ^ { ( l ) } \big \vert \right] + \left. \mathbb { E } \left[ \Delta _ { \mathrm { v a r } } ^ { ( l ) } \right] \right. .\tag{13}
$$

Fix $i \in [ T - 1 ]$ and condition on $W _ { 1 : l } .$ . Let $Q _ { i } \triangleq P _ { A _ { l } , Y | i , W _ { 1 : l } }$ and $\tilde { Q } _ { i } \triangleq Q _ { A _ { l } , Y | i , W _ { 1 : l } } .$ . If the loss function $\ell ( W , ( X , Y ) )$ is σ-sub-Gaussian for all $W \in { \mathcal { W } }$ and $Z \in { \mathcal { Z } } ,$ it follows that for any fixed W, the random variable $F _ { W } ( A _ { l } , Y )$ is σ-subgaussian under $Q _ { i }$ (and also under $\tilde { Q } _ { i } )$ . By Lemma A.6, we can obtain

$$
\left| \mathbb { E } _ { Q _ { i } } [ F _ { W } ( A _ { l } , Y ) ] - \mathbb { E } _ { \tilde { Q } _ { i } } [ F _ { W } ( A _ { l } , Y ) ] \right| \leq \sqrt { 2 \sigma ^ { 2 } D _ { \mathrm { K L } } ( \tilde { Q } _ { i } \| Q _ { i } ) } .
$$

Substitute into $\Delta _ { \mathrm { d r i f t } } ^ { ( l ) }$

$$
| \Delta _ { \mathrm { d r i f t } } ^ { ( l ) } | \leq \sum _ { i = 1 } ^ { T - 1 } \sqrt { 2 \sigma ^ { 2 } D _ { \mathrm { K L } } \mathopen { } \mathclose \bgroup ( Q _ { A _ { l } , Y | i , W _ { 1 : l } } \aftergroup \egroup  P _ { A _ { l } , Y | i , W _ { 1 : l } } \aftergroup \egroup ) } .
$$

Take the expectation and use Jensen and Cauchy–Schwarz’s inequality,

$$
\begin{array} { r l } & { \mathbb { E } \big [ | \Delta _ { \mathrm { d r i f t } } ^ { ( l ) } | \big ] \leq \sqrt { ( T - 1 ) \displaystyle \sum _ { i = 1 } ^ { T - 1 } \mathbb { E } \big [ 2 \sigma ^ { 2 } D _ { \mathrm { K L } } \big ( Q _ { A _ { l } , Y | i , W _ { 1 : l } } ~ \big \| ~ P _ { A _ { l } , Y | i , W _ { 1 : l } } \big ) \big ] } } \\ & { \qquad = ( T - 1 ) \sqrt { 2 \sigma ^ { 2 } \cdot \frac { 1 } { T - 1 } \displaystyle \sum _ { i = 1 } ^ { T - 1 } \mathbb { E } \big [ D _ { \mathrm { K L } } \big ( Q _ { A _ { l } , Y | i , W _ { 1 : l } } ~ \big \| ~ P _ { A _ { l } , Y | i , W _ { 1 : l } } \big ) \big ] } . } \end{array}\tag{14}
$$

For each fixed $W _ { 1 : l }$ and each i, take expectation over $W _ { 1 : l }$ and define

$$
{ \bf \nabla } { \bf K } ^ { ( l ) } \triangleq \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } \mathbb { E } \big [ D _ { \mathrm { K L } } \big ( Q _ { A _ { l } , Y | i , W _ { 1 : l } } \ \big \| \ P _ { A _ { l } , Y | i , W _ { 1 : l } } \big ) \big ] .
$$

Combining with (14) that

$$
\mathbb { E } \big [ | \Delta _ { \mathrm { d r i f t } } ^ { ( l ) } | \big ] \leq ( T - 1 ) \sqrt { 2 \sigma ^ { 2 } \mathcal { K } ^ { ( l ) } } .\tag{15}
$$

Next, we bound $| \mathbb { E } [ \Delta _ { \mathrm { v a r } } ^ { ( l ) } ] |$ . Recall the weights

$$
a \triangleq { \frac { 1 } { m } } \quad { \mathrm { ( e a c h ~ o l d ~ r e p l a y ~ s a m p l e ) } } , \qquad b \triangleq { \frac { 1 } { n } } \quad { \mathrm { ( e a c h ~ n e w - t a s k ~ s a m p l e ) } } .
$$

Let $N = ( T - 1 ) m + n$ . Enumerate all representation–label pairs used in the variance term as a single sequence

$$
U _ { 1 } ^ { ( l ) } , \ldots , U _ { N } ^ { ( l ) } , \quad \mathrm { w h e r e } \quad \{ U _ { k } ^ { ( l ) } \} _ { k = 1 } ^ { ( T - 1 ) m } = \mathbf { U } _ { \mathrm { o l d } } ^ { ( l ) } , \quad \{ U _ { k } ^ { ( l ) } \} _ { k = ( T - 1 ) m + 1 } ^ { N } = \mathbf { U } _ { \mathrm { n e w } } ^ { ( l ) } .
$$

Define the corresponding weights

$$
\begin{array} { r } { w _ { k } \triangleq \left\{ \begin{array} { l l } { a , } & { 1 \leq k \leq ( T - 1 ) m , } \\ { b , } & { ( T - 1 ) m < k \leq N . } \end{array} \right. } \end{array}
$$

For each index k, define its centering marginal (given $W _ { 1 : l } )$ as

$$
\begin{array} { r } { Q _ { k } ( \cdot ) \triangleq P _ { U ^ { ( l ) } | \mathbf { s } \mathbf { r } \mathbf { c } ( k ) , W _ { 1 : l } } = \left\{ \begin{array} { l l } { Q _ { A _ { l } , Y | i , W _ { 1 : l } } , } & { \mathbf { s } \mathbf { r } \mathbf { c } ( k ) = i \in [ T - 1 ] , } \\ { P _ { A _ { l } , Y | T , W _ { 1 : l } } , } & { \mathbf { s } \mathbf { r } \mathbf { c } ( k ) = \mathrm { n e w } . } \end{array} \right. } \end{array}
$$

where $\mathsf { s r c } ( k ) \in \{ 1 , \dots , T - 1 , \mathrm { n e w } \}$ indicates the task source of $U _ { k } ^ { ( l ) }$ : for $k \leq ( T - 1 ) m , \mathsf { s r c } ( k ) = i \mathrm { i f } \ U _ { k } ^ { ( l ) }$ is a replayed pair from old task i, and for $k > ( T - 1 ) m , { \mathsf { s r c } } ( k ) = { \mathrm { n e w } }$

Define the random centering functional ${ \bar { F } } _ { k } ( W ) \triangleq \mathbb { E } _ { U \sim Q _ { k } } [ F _ { W } ( U ) ]$ , which is a random variable because it still depends on the top parameters $W _ { l + 1 : L }$ through $F _ { W }$ . With this notation, the variance term can be written as

$$
\Delta _ { \mathrm { v a r } } ^ { ( l ) } = \sum _ { k = 1 } ^ { N } w _ { k } \Big ( \bar { F } _ { k } ( W ) - F _ { W } ( U _ { k } ^ { ( l ) } ) \Big ) .\tag{16}
$$

Introduce the filtration $\mathcal { F } _ { 0 } \subset \mathcal { F } _ { 1 } \subset \cdots \subset \mathcal { F } _ { N }$ with

$$
\begin{array} { r } { \mathcal { F } _ { k } \triangleq \sigma \big ( W _ { 1 : l } , U _ { 1 } ^ { ( l ) } , \dots , U _ { k } ^ { ( l ) } \big ) , \qquad k = 0 , 1 , \dots , N , } \end{array}
$$

(where $\mathcal { F } _ { 0 } = \sigma ( W _ { 1 : l } ) )$ .

Fix any $k \in [ N ]$ and condition on $\mathcal { F } _ { k - 1 }$ . Define the conditional joint law

$$
\mu _ { k } \triangleq P _ { W _ { l + 1 : L } , U _ { k } ^ { ( l ) } | \mathcal { F } _ { k - 1 } } .
$$

Define the corresponding reference conditional law

$$
\nu _ { k } ^ { \star } \triangleq { P } _ { W _ { l + 1 : L } | \mathcal { F } _ { k - 1 } } \otimes Q _ { k } .\tag{17}
$$

Taking expectation of (16) and using the tower property and the definition of $\Delta _ { \mathrm { v a r } } ^ { ( l ) }$ , we have

$$
\mathbb { E } \big [ \Delta _ { \mathrm { v a r } } ^ { ( l ) } \big ] = \sum _ { k = 1 } ^ { N } w _ { k } \mathbb { E } \Big [ \mathbb { E } \big [ \bar { F } _ { k } ( W ) \mid \mathcal { F } _ { k - 1 } \big ] - \mathbb { E } \big [ F _ { W } ( U _ { k } ^ { ( l ) } ) \mid \mathcal { F } _ { k - 1 } \big ] \Big ] ,\tag{18}
$$

Now we identify the two conditional expectations in (18) with expectations under $\nu _ { k } ^ { \star }$ and $\mu _ { k }$ . By definition $\bar { F } _ { k } ( W ) =$ $\mathbb { E } _ { U \sim Q _ { k } } [ F _ { W } ( U ) ] , { \mathcal { F } } _ { k - 1 }$ averages over the remaining randomness in $W _ { l + 1 : L }$ . Using $\nu _ { k } ^ { \star }$ in (17),

$$
\operatorname { \mathbb { E } } \left[ { \bar { F } } _ { k } ( W ) \mid { \mathcal { F } } _ { k - 1 } \right] = \operatorname { \mathbb { E } } _ { \nu _ { k } ^ { \star } } \left[ F _ { W } ( U ) \mid { \mathcal { F } } _ { k - 1 } \right] ,\tag{19}
$$

and explicitly

$$
\mathbb { E } _ { \nu _ { k } ^ { \star } } [ F _ { W } ( U ) \mid { \mathcal F } _ { k - 1 } ] = \int \left( \int F _ { ( W _ { 1 : l } , w ) } ( u ) Q _ { k } ( d u ) \right) P _ { W _ { l + 1 : L } \mid { \mathcal F } _ { k - 1 } } ( d w ) ,
$$

which is indeed $\mathcal { F } _ { k - 1 }$ -measurable. By definition of $\mu _ { k }$ in (17),

$$
\mathbb { E } \Big [ F _ { W } ( U _ { k } ^ { ( l ) } ) \mid \mathcal { F } _ { k - 1 } \Big ] = \mathbb { E } _ { \mu _ { k } } \Big [ F _ { W } ( U _ { k } ^ { ( l ) } ) \mid \mathcal { F } _ { k - 1 } \Big ] .\tag{20}
$$

Substituting (19) and (20) into (18) gives the exact bridge identity,

$$
\mathbb { E } [ \Delta _ { \mathrm { v a r } } ^ { ( l ) } ] = \sum _ { k = 1 } ^ { N } w _ { k } \mathbb { E } \left[ \mathbb { E } _ { \nu _ { k } ^ { \star } } \left[ F _ { W } ( U ) \mid \mathcal { F } _ { k - 1 } \right] - \mathbb { E } _ { \mu _ { k } } \left[ F _ { W } ( U _ { k } ^ { ( l ) } ) \mid \mathcal { F } _ { k - 1 } \right] \right] .\tag{21}
$$

Applying $| \mathbb { E } [ X ] | \leq \mathbb { E } [ | X | ]$ and the triangle inequality gives

$$
\left| \mathbb { E } \left[ \Delta _ { \mathrm { v a r } } ^ { ( l ) } \right] \right| \leq \sum _ { k = 1 } ^ { N } w _ { k } \mathbb { E } \left[ \left| \mathbb { E } _ { \nu _ { k } ^ { \star } } \big [ F _ { W } ( U ) \mid \mathcal { F } _ { k - 1 } \big ] - \mathbb { E } _ { \mu _ { k } } \big [ F _ { W } ( U _ { k } ^ { ( l ) } ) \mid \mathcal { F } _ { k - 1 } \big ] \right| \right] .\tag{22}
$$

Under the assumption that the loss is σ-subgaussian, for any fixed $W$ the random variable $F _ { W } ( U )$ is σ-subgaussian under $Q _ { k }$ Hence, conditioning on $\mathcal { F } _ { k - 1 }$ , the same subgaussian proxy holds under $\nu _ { k } ^ { \star }$ , because it is a mixture over $W _ { l + 1 : L }$ with the same $\sigma .$ . Therefore Lemma A.6 applied conditionally on $\mathcal { F } _ { k - 1 }$ yields

$$
\begin{array} { r } { \Big | \mathbb { E } _ { \nu _ { k } ^ { * } } \big [ F _ { W } ( U ) \mid \mathcal { F } _ { k - 1 } \big ] - \mathbb { E } _ { \mu _ { k } } \big [ F _ { W } ( U _ { k } ^ { ( l ) } ) \mid \mathcal { F } _ { k - 1 } \big ] \Big | \leq \sqrt { 2 \sigma ^ { 2 } D _ { \mathrm { K L } } ( \mu _ { k } \| \nu _ { k } ^ { \star } ) } . } \end{array}\tag{23}
$$

Let $\nu _ { k } \triangleq P _ { W _ { l + 1 : L } | \mathcal { F } _ { k - 1 } } \otimes P _ { U _ { k } ^ { ( l ) } | \mathcal { F } _ { k - 1 } }$ be the product of the true marginals under $\mathcal { F } _ { k - 1 }$ . Then by the KL chain rule,

$$
\begin{array} { r l } {  { D _ { \mathrm { K L } } ( \mu _ { k } \| \nu _ { k } ^ { \star } ) = D _ { \mathrm { K L } } ( \mu _ { k } \| \nu _ { k } ) + \mathbb { E } _ { \mu _ { k } } [ \log \frac { d \nu _ { k } } { d \nu _ { k } ^ { \star } } ( W _ { l + 1 : L } , U _ { k } ^ { ( l ) } ) ] } } \\ & { = I ( U _ { k } ^ { ( l ) } ; W _ { l + 1 : L } \mid \mathcal { F } _ { k - 1 } ) + D _ { \mathrm { K L } } \Big ( P _ { U _ { k } ^ { ( l ) } | \mathcal { F } _ { k - 1 } } \mid \mid Q _ { k } \Big ) . } \end{array}\tag{24}
$$

Define

$$
I _ { k } \triangleq I \big ( U _ { k } ^ { ( l ) } ; W _ { l + 1 : L } \mid \mathcal { F } _ { k - 1 } \big ) , \qquad M _ { k } \triangleq D _ { \mathrm { K L } } \Big ( P _ { U _ { k } ^ { ( l ) } | \mathcal { F } _ { k - 1 } } \ \big \| \ Q _ { k } \Big ) .
$$

Then by (22)–(24),

$$
\Big | \mathbb { E } _ { \nu _ { k } ^ { \star } } \big [ F _ { W } ( U ) \mid \mathcal { F } _ { k - 1 } \big ] - \mathbb { E } _ { \mu _ { k } } \big [ F _ { W } ( U _ { k } ^ { ( l ) } ) \mid \mathcal { F } _ { k - 1 } \big ] \Big | \leq \sqrt { 2 \sigma ^ { 2 } ( I _ { k } + M _ { k } ) } .\tag{25}
$$

Substituting (25) into (22) yields

$$
\left| \mathbb { E } \big [ \Delta _ { \mathrm { v a r } } ^ { ( l ) } \big ] \right| \leq \sqrt { 2 \sigma ^ { 2 } } \mathbb { E } \left[ \sum _ { k = 1 } ^ { N } w _ { k } \sqrt { I _ { k } + M _ { k } } \right] .\tag{26}
$$

Applying Cauchy–Schwarz to the inner weighted sum gives

$$
\sum _ { k = 1 } ^ { N } w _ { k } \sqrt { I _ { k } + M _ { k } } \leq \sqrt { \sum _ { k = 1 } ^ { N } w _ { k } ^ { 2 } } \cdot \sqrt { \sum _ { k = 1 } ^ { N } ( I _ { k } + M _ { k } ) } .\tag{27}
$$

Using Jensen’s inequality $\mathbb { E } [ { \sqrt { X } } ] \leq { \sqrt { \mathbb { E } [ X ] } }$ and noting,

$$
\sum _ { k = 1 } ^ { N } w _ { k } ^ { 2 } = ( T - 1 ) m \bigg ( \frac { 1 } { m } \bigg ) ^ { 2 } + n \bigg ( \frac { 1 } { n } \bigg ) ^ { 2 } = \bigg ( \frac { T - 1 } { m } + \frac { 1 } { n } \bigg ) = \frac { 1 } { N _ { \mathrm { e f f } } } ,
$$

we obtain

$$
\left| \mathbb { E } \big [ \Delta _ { \mathrm { v a r } } ^ { ( l ) } \big ] \right| \leq \sqrt { \frac { 2 \sigma ^ { 2 } } { N _ { \mathrm { e f f } } } } \sqrt { \mathbb { E } \left[ \sum _ { k = 1 } ^ { N } I _ { k } \right] + \mathbb { E } \left[ \sum _ { k = 1 } ^ { N } M _ { k } \right] } .\tag{28}
$$

By the chain rule for mutual information and Lemma A.4,

$$
\mathbb { E } \left[ \sum _ { k = 1 } ^ { N } I _ { k } \right] = I ( \mathbf { U } ^ { ( l ) } ; W _ { l + 1 : L } \mid W _ { 1 : l } ) = \mathcal { S } ^ { ( l ) } + \mathcal { P } ^ { ( l ) } - \mathcal { R } ^ { ( l ) } .
$$

Moreover, by the chain rule for KL under the task-wise block product centering $\widetilde Q _ { \mathbf { U } ^ { ( l ) } \left| W _ { 1 : l } \right. } = \bigotimes _ { k = 1 } ^ { N } Q _ { k }$ , we have

$$
\mathbb { E } \left[ \sum _ { k = 1 } ^ { N } M _ { k } \right] = \mathbb { E } \left[ D _ { \mathrm { K L } } \Big ( P _ { \mathbf { U } ^ { ( l ) } | W _ { 1 : l } } \ \big \| \ \widetilde { Q } _ { \mathbf { U } ^ { ( l ) } | W _ { 1 : l } } \Big ) \right] = \mathcal { C } ^ { ( l ) } .
$$

Combining (28), we can get

$$
\left| \mathbb { E } \big [ \Delta _ { \mathrm { v a r } } ^ { ( l ) } \big ] \right| \leq \sqrt { \frac { 2 \sigma ^ { 2 } } { N _ { \mathrm { e f f } } } \Big ( S ^ { ( l ) } + \mathcal { P } ^ { ( l ) } - \mathcal { R } ^ { ( l ) } + \mathcal { C } ^ { ( l ) } \Big ) } .\tag{29}
$$

Finally, substituting (15) and (29) into (13) completes the proof.

B. Proof of Corollary IV.2

Corollary IV.2 (Restate). At the input layer $l = 0 ,$ , the hierarchical generalization bound simplifies to:

$$
| \mathrm { g e n } _ { W } | \leq \sqrt { \frac { 2 \sigma ^ { 2 } } { N _ { \mathrm { e f f } } } I \big ( S ; W \big ) } ,
$$

where $S$ is the original $( X , Y )$ pair in the replay buffer and the current task dataset.

Proof. The proof adopts the same framework as Theorem IV.1. $\mathbf { A } \mathbf { t } \ l = 0 ,$ , the representation $A _ { 0 } = X$ and $W _ { 1 : 0 }$ is empty. We use the same notation $F _ { W } ( Z ) = \ell ( W , Z )$ for $Z = ( X , Y ) \in { \mathcal { Z } }$ . By definition,

$$
\mathrm { g e n } _ { W } = \mathbb { E } _ { W , D ^ { 1 : T } } \big [ \mathcal { L } ( W ) - \hat { \mathcal { L } } ( W ) \big ] ,
$$

where $\mathcal { L }$ and $\hat { \mathcal { L } }$ are population and empirical risks. Expanding $\mathcal { L } ( W )$ and $\hat { \mathcal { L } } ( W )$ as in (8) yields the exact old and new split

$$
\mathcal { L } ( W ) - \hat { \mathcal { L } } ( W ) = \sum _ { i = 1 } ^ { T - 1 } \left( \mathbb { E } _ { Z \sim \mathcal { D } _ { i } } [ F _ { W } ( Z ) ] - \frac { 1 } { m } \sum _ { j \in \mathcal { T } _ { i } } F _ { W } ( Z _ { j } ^ { i } ) \right) + \left( \mathbb { E } _ { Z \sim \mathcal { D } _ { T } } [ F _ { W } ( Z ) ] - \frac { 1 } { n } \sum _ { j = 1 } ^ { n } F _ { W } ( Z _ { j } ^ { T } ) \right) .\tag{30}
$$

Recall $P _ { A _ { l } , Y | i , W _ { 1 : l } }$ denote the conditional law of the representation–label pair $( A _ { l } , Y )$ induced by $( X , Y ) \sim { \mathcal { D } } _ { i }$ under the current representation parameters $W _ { 1 : l }$ . Therefore, when $l = 0 .$ , for each old task $i \in [ T - 1 ]$ , the input-layer population law is $P _ { X , Y \mid i } \equiv \mathcal { D } _ { i }$ . Similarly, the replay proxy $\begin{array} { r } { \hat { P } _ { A _ { 0 } , Y | S ^ { i } } = P _ { S ^ { i } } = \frac { 1 } { m } \sum _ { j \in \mathbb { Z } _ { i } } \delta _ { Z _ { j } ^ { i } } ( \cdot ) } \end{array}$ , which is uniform on the buffer subset. Define its unconditional centroid law at $l = 0$ by

$$
Q _ { A _ { 0 } , Y | i } \triangleq \mathbb { E } \left[ \hat { P } _ { S ^ { i } } \right] .\tag{31}
$$

Using the same add-and-subtract step as (11) (with $W _ { 1 : 0 }$ void), we obtain the exact decomposition

$$
\begin{array} { r } { \mathcal { L } ( W ) - \hat { \mathcal { L } } ( W ) = \Delta _ { \mathrm { d r i f t } } ^ { ( 0 ) } + \Delta _ { \mathrm { v a r } } ^ { ( 0 ) } , } \end{array}\tag{32}
$$

where

$$
\Delta _ { \mathrm { d r i f t } } ^ { ( 0 ) } \triangleq \sum _ { i = 1 } ^ { T - 1 } \Big ( \mathbb { E } _ { Z \sim \mathcal { D } _ { i } } [ F _ { W } ( Z ) ] - \mathbb { E } _ { Z \sim Q _ { A _ { 0 } , Y | i } } [ F _ { W } ( Z ) ] \Big ) ,\tag{33}
$$

$$
\Delta _ { \mathrm { v a r } } ^ { ( 0 ) } \triangleq \sum _ { i = 1 } ^ { T - 1 } \Big ( \mathbb { E } _ { Z \sim Q _ { A _ { 0 } , Y \mid i } } [ F _ { W } ( Z ) ] - \frac { 1 } { m } \sum _ { j \in \mathbb { Z } _ { i } } F _ { W } ( Z _ { j } ^ { i } ) \Big ) + \Big ( \mathbb { E } _ { Z \sim \mathcal { D } _ { T } } [ F _ { W } ( Z ) ] - \frac { 1 } { n } \sum _ { j = 1 } ^ { n } F _ { W } ( Z _ { j } ^ { T } ) \Big ) .\tag{34}
$$

At the input layer, the replay centroid coincides with the task distribution: indeed, by symmetry of uniform subsampling from i.i.d. samples, a uniformly chosen element from the random subset $M ^ { i }$ has marginal law $\mathcal { D } _ { i }$ , hence $Q _ { A _ { 0 } , Y | i } = D _ { i } = P _ { X , Y | i } .$ Therefore, $\Delta _ { \mathrm { d r i f t } } ^ { ( 0 ) } = 0$ in (33), and

$$
\begin{array} { r } { \left. \mathrm { g e n } _ { W } \right. = \left. \mathbb { E } [ \mathcal { L } ( W ) - \hat { \mathcal { L } } ( W ) ] \right. = \left. \mathbb { E } [ \Delta _ { \mathrm { v a r } } ^ { ( 0 ) } ] \right. . } \end{array}\tag{35}
$$

Let $N = ( T - 1 ) m + n$ . Enumerate all examples used in (34) as a single sequence $U _ { 1 } ^ { ( 0 ) } , \dots , U _ { N } ^ { ( 0 ) } \in \mathcal { Z }$ , where the first $( T - 1 ) m$ terms come from the replay buffers and the last n terms come from the current task dataset. Define the weights

$$
a \triangleq \frac { 1 } { m } , \qquad b \triangleq \frac { 1 } { n } , \qquad w _ { k } \triangleq \left\{ a , \quad 1 \leq k \leq ( T - 1 ) m , \right. \qquad 
$$

and let $\mathsf { s r c } ( k ) \in \{ 1 , \ldots , T - 1$ , new} denote the source task of $U _ { k } ^ { ( 0 ) }$ . For each $k ,$ let

$$
\begin{array} { r } { Q _ { k } ( \cdot ) \triangleq P _ { U ^ { ( 0 ) } | \mathsf { s r c } ( k ) } = \left\{ \mathscr { D } _ { i } , \quad \mathsf { s r c } ( k ) = i \in [ T - 1 ] , \right. } \\ { \left. \mathscr { D } _ { T } , \quad \mathsf { s r c } ( k ) = \mathrm { n e w } , \right. \qquad } \end{array}
$$

Define the random centering functional ${ \bar { F } } _ { k } ( W ) \triangleq \mathbb { E } _ { U \sim Q _ { k } } [ F _ { W } ( U ) ]$ , with this notation, the variance term can be written as

$$
\Delta _ { \mathrm { v a r } } ^ { ( 0 ) } = \sum _ { k = 1 } ^ { N } w _ { k } \Bigl ( \bar { F } _ { k } ( W ) - F _ { W } ( U _ { k } ^ { ( 0 ) } ) \Bigr ) .\tag{36}
$$

Define the filtration

$$
\mathcal { F } _ { k } \triangleq \sigma \big ( U _ { 1 } ^ { ( 0 ) } , \ldots , U _ { k } ^ { ( 0 ) } \big ) , \qquad k = 0 , 1 , \ldots , N .
$$

For each $k ,$ let the joint law be $\mu _ { k } \triangleq P _ { W , U _ { k } ^ { ( 0 ) } | \mathcal { F } _ { k } }$ and the corresponding reference conditional law $\nu _ { k } ^ { \star } \triangleq { P _ { W | \mathcal { F } _ { k - 1 } } } \otimes Q _ { k }$ −1 Taking expectation of (36) and using the tower property and the definition of $\Delta _ { \mathrm { v a r } } ^ { ( 0 ) }$ , we have

$$
\mathbb { E } \big [ \Delta _ { \mathrm { v a r } } ^ { ( 0 ) } \big ] = \sum _ { k = 1 } ^ { N } w _ { k } \mathbb { E } \Big [ \mathbb { E } \big [ \bar { F } _ { k } ( W ) \mid \mathcal { F } _ { k - 1 } \big ] - \mathbb { E } \big [ F _ { W } ( U _ { k } ^ { ( 0 ) } ) \mid \mathcal { F } _ { k - 1 } \big ] \Big ] ,\tag{37}
$$

Now we identify the two conditional expectations in (37) with expectations under $\nu _ { k } ^ { \star }$ and $\mu _ { k }$ . By definition $\bar { F } _ { k } ( W ) =$ $\mathbb { E } _ { U \sim Q _ { k } } [ F _ { W } ( U ) ] , \mathcal { F } _ { k - 1 }$ averages over the remaining randomness in W. Using $\nu _ { k } ^ { \star }$

$$
\operatorname { \mathbb { E } } \left[ { \bar { F } } _ { k } ( W ) \mid { \mathcal { F } } _ { k - 1 } \right] = \operatorname { \mathbb { E } } _ { \nu _ { k } ^ { \star } } \left[ F _ { W } ( U ) \mid { \mathcal { F } } _ { k - 1 } \right] ,\tag{38}
$$

and explicitly

$$
\mathbb { E } _ { \nu _ { k } ^ { \star } } [ F _ { W } ( U ) \mid \mathcal { F } _ { k - 1 } ] = \int \left( \int F _ { w } ( u ) Q _ { k } ( d u ) \right) P _ { W | \mathcal { F } _ { k - 1 } } ( d w ) ,
$$

which is indeed $\mathcal { F } _ { k - 1 }$ -measurable. By definition of $\mu _ { k }$ ,

$$
\mathbb { E } \Big [ F _ { W } ( U _ { k } ^ { ( 0 ) } ) \mid \mathcal { F } _ { k - 1 } \Big ] = \mathbb { E } _ { \mu _ { k } } \Big [ F _ { W } ( U _ { k } ^ { ( 0 ) } ) \mid \mathcal { F } _ { k - 1 } \Big ] .\tag{39}
$$

Substituting (38) and (39) into (37) gives the exact bridge identity,

$$
\mathbb { E } [ \Delta _ { \mathrm { v a r } } ^ { ( 0 ) } ] = \sum _ { k = 1 } ^ { N } w _ { k } \mathbb { E } \left[ \mathbb { E } _ { \nu _ { k } ^ { \star } } \left[ F _ { W } ( U ) \mid \mathcal { F } _ { k - 1 } \right] - \mathbb { E } _ { \mu _ { k } } \left[ F _ { W } ( U _ { k } ^ { ( 0 ) } ) \mid \mathcal { F } _ { k - 1 } \right] \right] .\tag{40}
$$

Applying $| \mathbb { E } [ X ] | \leq \mathbb { E } [ | X | ]$ and the triangle inequality gives

$$
\Big | \mathbb { E } \big [ \Delta _ { \mathrm { v a r } } ^ { ( 0 ) } \big ] \Big | \leq \sum _ { k = 1 } ^ { N } w _ { k } \mathbb { E } \Big [ \Big | \mathbb { E } _ { \nu _ { k } ^ { \star } } \big [ F _ { W } ( U ) \big | \mathcal { F } _ { k - 1 } \big ] - \mathbb { E } _ { \mu _ { k } } \big [ F _ { W } ( U _ { k } ^ { ( 0 ) } ) \big | \mathcal { F } _ { k - 1 } \big ] \Big | \Big ] .\tag{41}
$$

Under the assumption that the loss is σ-subgaussian, for any fixed W the random variable $F _ { W } ( U )$ is σ-subgaussian under $Q _ { k }$ Hence, conditioning on $\mathcal { F } _ { k - 1 }$ , the same subgaussian proxy holds under $\nu _ { k } ^ { \star }$ , because it is a mixture over $W$ with the same σ. Therefore Lemma A.6 applied conditionally on $\mathcal { F } _ { k - 1 }$ yields

$$
\begin{array} { r } { \left| \mathbb { E } _ { \nu _ { k } ^ { \star } } \big [ F _ { W } ( U ) \mid \mathcal { F } _ { k - 1 } \big ] - \mathbb { E } _ { \mu _ { k } \big [ F _ { W } ( U _ { k } ^ { ( 0 ) } ) \mid \mathcal { F } _ { k - 1 } \big ] } \right| \leq \sqrt { 2 \sigma ^ { 2 } D _ { \mathrm { K L } } ( \mu _ { k } \| \nu _ { k } ^ { \star } ) } . } \end{array}\tag{42}
$$

Let $\nu _ { k } \triangleq P _ { W | \mathcal { F } _ { k - 1 } } \otimes P _ { U _ { k } ^ { ( 0 ) } | \mathcal { F } _ { k - 1 } }$ be the product of the true marginals under $\mathcal { F } _ { k - 1 }$ . Then by the KL chain rule,

$$
D _ { \mathrm { K L } } ( \mu _ { k } \| \nu _ { k } ^ { \star } ) = I \big ( U _ { k } ^ { ( 0 ) } ; W \mid \mathcal { F } _ { k - 1 } \big ) + D _ { \mathrm { K L } } \Big ( P _ { U _ { k } ^ { ( 0 ) } | \mathcal { F } _ { k - 1 } } \ \big \| \ Q _ { k } \Big ) ,\tag{43}
$$

Define

$$
{ \cal I } _ { k } \triangleq { \cal I } \big ( { \cal U } _ { k } ^ { ( 0 ) } ; { \cal W } \mid { \mathcal F } _ { k - 1 } \big ) , \qquad { \cal M } _ { k } \triangleq { \cal D } _ { \mathrm { K L } } \Big ( { \cal P } _ { { \cal U } _ { k } ^ { ( 0 ) } | { \mathcal F } _ { k - 1 } } \ \parallel \ Q _ { k } \Big ) .
$$

Substituting into (41) and following exactly the Cauchy–Schwarz + Jensen aggregation in (26)–(28) yields

$$
\left| \mathbb { E } \big [ \Delta _ { \mathrm { v a r } } ^ { ( 0 ) } \big ] \right| \leq \sqrt { \frac { 2 \sigma ^ { 2 } } { N _ { \mathrm { e f f } } } } \sqrt { \mathbb { E } \left[ \sum _ { k = 1 } ^ { N } I _ { k } \right] } + \mathbb { E } \left[ \sum _ { k = 1 } ^ { N } M _ { k } \right] .\tag{44}
$$

By the mutual information chain rule,

$$
\mathbb { E } \left[ \sum _ { k = 1 } ^ { N } I _ { k } \right] = I ( S ; W ) .
$$

Moreover, by the chain rule for KL under the task-wise block product centering $\widetilde Q _ { \mathbf { U } ^ { ( 0 ) } } \triangleq \left( \bigotimes _ { i = 1 } ^ { T - 1 } ( \mathcal { D } _ { i } ) ^ { \otimes m } \right) \otimes ( \mathcal { D } _ { T } ) ^ { \otimes n }$ , we have

$$
\mathbb { E } \left[ \sum _ { k = 1 } ^ { N } M _ { k } \right] = D _ { \mathrm { K L } } \big ( P _ { S } \ \lVert \ \widetilde { Q } _ { \mathbf { U } ^ { ( 0 ) } } \big ) .
$$

At $l = 0 .$ , the sequence S is block-wise i.i.d. by construction (each task dataset is i.i.d., tasks are mutually independent, and each buffer contributes m distinct coordinates from an i.i.d. task dataset); hence $P _ { S } = \widetilde { Q } _ { \mathbf { U } ^ { ( 0 ) } }$ and the above KL term is zero. Combining with (35) and (44) yields

$$
| \mathrm { g e n } _ { W } | \leq \sqrt { \frac { 2 \sigma ^ { 2 } } { N _ { \mathrm { e f f } } } I ( S ; W ) } .
$$

## C. Proof of Corollary IV.3

Corollary IV.3 (Restate). At the output layer $l = L$ , the hierarchical generalization bound simplifies to:

$$
| \mathrm { g e n } _ { W } | \leq ( T - 1 ) \sqrt { 2 \sigma ^ { 2 } { \mathcal K } ^ { ( L ) } } + \sqrt { \frac { 2 \sigma ^ { 2 } } { N _ { \mathrm { e f f } } } { \mathcal C } ^ { ( L ) } } ,
$$

where $\begin{array} { r } { { \cal K } ^ { ( L ) } = \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } \mathbb { E } _ { W } \big [ D _ { \mathrm { K L } } \big ( Q _ { A _ { L } , Y | i , W } \big | \big | P _ { A _ { L } , Y | i , W } \big ) \big ] , \ { \mathcal C } ^ { ( L ) } = \mathbb { E } _ { W } \big [ D _ { \mathrm { K L } } \big ( P _ { \mathbf { U } ^ { ( L ) } | W } \big | \big | \tilde { Q } _ { \mathbf { U } ^ { ( L ) } | W } \big ) \big ] . } \end{array}$

Proof. We follow the proof structure of Theorem IV.1 and highlight the simplifications at $l = L . \ \mathrm { A t } \ l = L$ , the representation is the network output $A _ { L } = f _ { W } ( X )$ . The top subnetwork above layer L is empty, so we may regard $g _ { W _ { L + 1 : L } }$ as the identity map and write the loss as a function of the output–label pair:

$$
F _ { W } ( t , y ) \triangleq \ell ( t , y ) , \qquad \mathrm { s o ~ t h a t } \qquad \ell ( W , ( X , Y ) ) = F _ { W } ( A _ { L } , Y ) .
$$

All layer-L distributions below are understood as conditional laws given the full parameter $W \ = \ W _ { 1 : L }$ . As in (8), the population–empirical gap admits the exact split into old and new tasks. Introducing the layer-L population law $P _ { A _ { L } , Y | i , W _ { 1 : L } }$ of $( A _ { L } , Y )$ under $( X , Y ) \sim { \mathcal { D } } _ { i }$ and the replay proxy $\hat { P } _ { A _ { L } , Y \mid S ^ { i } , W _ { 1 : L } }$ which is uniform over $\{ ( A _ { j , L } ^ { i } , Y _ { j } ^ { i } ) : j \in \mathcal { T } _ { i } \}$ , recall the replay centroid

$$
Q _ { A _ { L } , Y | i , W _ { 1 : L } } = \mathbb { E } \left[ \hat { P } _ { A _ { L } , Y | S ^ { i } , W _ { 1 : L } } \middle | W \right] .\tag{45}
$$

Adding and subtracting $\mathbb { E } _ { Q _ { A _ { L } , Y | i , W _ { 1 : L } } } [ F _ { W } ( A _ { L } , Y ) ]$ inside each old-task gap yields an exact decomposition

$$
\begin{array} { r } { \mathcal { L } ( W ) - \hat { \mathcal { L } } ( W ) = \Delta _ { \mathrm { d r i f t } } ^ { ( L ) } + \Delta _ { \mathrm { v a r } } ^ { ( L ) } , } \end{array}\tag{46}
$$

where

$$
\Delta _ { \mathrm { d r i f t } } ^ { ( L ) } \triangleq \sum _ { i = 1 } ^ { T - 1 } \Big ( \mathbb { E } _ { Z \sim \mathcal { D } _ { i } } [ F _ { W } ( Z ) ] - \mathbb { E } _ { Z \sim Q _ { A _ { L } , Y | i , W _ { 1 : L } } } [ F _ { W } ( Z ) ] \Big ) ,\tag{47}
$$

$$
\Delta _ { \mathrm { v a r } } ^ { ( L ) } \triangleq \sum _ { i = 1 } ^ { T - 1 } \Big ( \mathbb { E } _ { Z \sim Q _ { A _ { L } , Y | i , W _ { 1 : L } } } [ F _ { W } ( Z ) ] - \frac { 1 } { m } \sum _ { j \in \mathcal { Z } _ { i } } F _ { W } ( Z _ { j } ^ { i } ) \Big ) + \Big ( \mathbb { E } _ { Z \sim \mathcal { D } _ { T } } [ F _ { W } ( Z ) ] - \frac { 1 } { n } \sum _ { j = 1 } ^ { n } F _ { W } ( Z _ { j } ^ { T } ) \Big ) .\tag{48}
$$

By definition,

$$
\begin{array} { r } { | \mathrm { g e n } _ { W } | = \left| \mathbb { E } [ \mathcal { L } ( W ) - \hat { \mathcal { L } } ( W ) ] \right| \leq \mathbb { E } \big [ | \Delta _ { \mathrm { d r i f t } } ^ { ( L ) } | \big ] + | \mathbb { E } \big [ \Delta _ { \mathrm { v a r } } ^ { ( L ) } \big ] | . } \end{array}\tag{49}
$$

Fix an old task $i \in [ T - 1 ]$ and condition on W. Since $F _ { W } ( A _ { L } , Y )$ is σ-subgaussian under both $P _ { A _ { L } , Y | i , W _ { 1 : L } }$ and $Q _ { A _ { L } , Y | i , W _ { 1 : L } } ,$ by Lemma A.6, we can obtain

$$
\begin{array} { r } { \left| \mathbb { E } _ { P _ { A _ { L } , Y | i , W _ { 1 : L } } } [ F _ { W } ] - \mathbb { E } _ { Q _ { A _ { L } , Y | i , W _ { 1 : L } } } [ F _ { W } ] \right| \leq \sqrt { 2 \sigma ^ { 2 } D _ { \mathrm { K L } } \Bigl ( Q _ { A _ { L } , Y | i , W _ { 1 : L } } \ \big \| \ P _ { A _ { L } , Y | i , W _ { 1 : L } } \Bigr ) } . } \end{array}
$$

Averaging over i and applying Cauchy–Schwarz exactly as in the main proof gives

$$
\mathbb { E } \big [ | \Delta _ { \mathrm { d r i f t } } ^ { ( L ) } | \big ] \leq ( T - 1 ) \sqrt { 2 \sigma ^ { 2 } \mathcal { K } ^ { ( L ) } } ,\tag{50}
$$

with $\mathcal { K } ^ { ( L ) }$ as stated. Let $N = ( T - 1 ) m + n$ and enumerate the layer-L training pairs as a single sequence $U _ { 1 } ^ { ( L ) } , \dots , U _ { N } ^ { ( L ) }$ with weights

$$
a \triangleq \frac { 1 } { m } , \qquad b \triangleq \frac { 1 } { n } , \qquad w _ { k } \triangleq \left\{ a , \quad 1 \leq k \leq ( T - 1 ) m , \right. \qquad 
$$

and let $\mathsf { s r c } ( k ) \in \{ 1 , \dots , T - 1 , \mathrm { n e w } \}$ denote the source task of $U _ { k } ^ { ( L ) }$ . For each k, let

$$
\begin{array} { r } { Q _ { k } ( \cdot ) \triangleq P _ { U ^ { ( L ) } | \mathrm { s r c } ( k ) , W } = \left\{ \begin{array} { l l } { Q _ { A _ { L } , Y | i , W _ { 1 : L } } , } & { \mathrm { s r c } ( k ) = i \in [ T - 1 ] , } \\ { P _ { A _ { L } , Y | T , W _ { 1 : L } } , } & { \mathrm { s r c } ( k ) = \mathrm { n e w } . } \end{array} \right. } \end{array}
$$

With this notation, the variance term can be written as

$$
\Delta _ { \mathrm { v a r } } ^ { ( L ) } = \sum _ { k = 1 } ^ { N } w _ { k } \Big ( \mathbb { E } _ { U \sim Q _ { k } } [ F _ { W } ( U ) ] - F _ { W } ( U _ { k } ^ { ( L ) } ) \Big ) .\tag{51}
$$

Define the filtration $( \mathcal { F } _ { 0 } = \sigma ( W ) )$

$$
\begin{array} { r } { \mathcal { F } _ { k } \triangleq \sigma \big ( W , U _ { 1 } ^ { ( L ) } , \ldots , U _ { k } ^ { ( L ) } \big ) , \qquad k = 0 , 1 , \ldots , N . } \end{array}
$$

For each k, define the true conditional law of the k-th coordinate

$$
\mu _ { k } \triangleq P _ { U _ { k } ^ { \left( L \right) } | \mathcal { F } _ { k - 1 } } .
$$

Since at $l = L$ the “top parameter” $W _ { L + 1 : L }$ is empty (a deterministic constant), the reference law $\nu _ { k } ^ { \star }$ reduces to the centering marginal:

$$
\nu _ { k } ^ { \star } \triangleq Q _ { k } .
$$

Taking expectation of (51) and using the tower property, we have

$$
\mathbb { E } \big [ \Delta _ { \mathrm { v a r } } ^ { ( L ) } \big ] = \sum _ { k = 1 } ^ { N } w _ { k } \mathbb { E } \Big [ \mathbb { E } \big [ F _ { W } ( U ) \mid \mathcal { F } _ { k - 1 } \big ] - \mathbb { E } \big [ F _ { W } ( U _ { k } ^ { ( L ) } ) \mid \mathcal { F } _ { k - 1 } \big ] \Big ] ,\tag{52}
$$

Since W is $\mathcal { F } _ { k - 1 }$ -measurable, $\mathfrak { L } \big [ F _ { W } ( U ) \mid \mathcal { F } _ { k - 1 } \big ] = \mathbb { E } _ { \nu _ { k } ^ { \star } } \left[ F _ { W } ( U ) \right]$ . By definition of $\mu _ { k } , \operatorname { \mathbb { E } } [ F _ { W } ( U _ { k } ^ { ( L ) } ) \mid { \mathcal { F } } _ { k - 1 } ] = \operatorname { \mathbb { E } } _ { \mu _ { k } } [ F _ { W } ( U _ { k } ^ { ( L ) } ) \mid$ | $\mathcal { F } _ { k - 1 } ]$ . Substituting into (52) gives the bridge identity,

$$
\mathbb { E } [ \Delta _ { \mathrm { v a r } } ^ { ( L ) } ] = \sum _ { k = 1 } ^ { N } w _ { k } \mathbb { E } \left[ \mathbb { E } _ { \nu _ { k } ^ { \star } } \left[ F _ { W } ( U ) \right] - \mathbb { E } _ { \mu _ { k } } \left[ F _ { W } ( U _ { k } ^ { ( L ) } ) \mid \mathcal { F } _ { k - 1 } \right] \right] .\tag{53}
$$

Applying $| \mathbb { E } [ X ] | \leq \mathbb { E } [ | X | ]$ and the triangle inequality gives

$$
\left| \mathbb { E } \big [ \Delta _ { \mathrm { v a r } } ^ { ( L ) } \big ] \right| \leq \sum _ { k = 1 } ^ { N } w _ { k } \mathbb { E } \Big [ \Big | \mathbb { E } _ { \nu _ { k } ^ { \star } } \big [ F _ { W } ( U ) \big ] - \mathbb { E } _ { \mu _ { k } } \big [ F _ { W } ( U _ { k } ^ { ( L ) } ) \mid \mathcal { F } _ { k - 1 } \big ] \Big | \Big ] .\tag{54}
$$

By Lemma A.6, we can obtain

$$
\left| \mathbb { E } _ { \nu _ { k } ^ { \star } } [ F _ { W } ( U ) ] - \mathbb { E } _ { \mu _ { k } } [ F _ { W } ( U _ { k } ^ { ( L ) } ) \mid { \mathcal F } _ { k - 1 } ] \right| \le \sqrt { 2 \sigma ^ { 2 } D _ { \mathrm { K L } } ( \mu _ { k } \| Q _ { k } ) } .
$$

Aggregating over k with Cauchy–Schwarz and Jensen gives

$$
\left| \mathbb { E } [ \Delta _ { \mathrm { v a r } } ^ { ( L ) } ] \right| \leq \sqrt { \frac { 2 \sigma ^ { 2 } } { N _ { \mathrm { e f f } } } } \sqrt { \mathbb { E } \left[ \sum _ { k = 1 } ^ { N } D _ { \mathrm { K L } } ( \mu _ { k } \| Q _ { k } ) \right] } .
$$

Finally, by the chain rule for KL under the product reference $\widetilde { Q } _ { \mathbf { U } ^ { ( L ) } | W } = \bigotimes _ { k = 1 } ^ { N } Q _ { k }$ , we have

$$
\mathbb { E } \left[ \sum _ { k = 1 } ^ { N } D _ { \mathrm { K L } } \big ( \mu _ { k } \| Q _ { k } \big ) \right] = \mathbb { E } \Big [ D _ { \mathrm { K L } } \Big ( P _ { \mathbf { U } ^ { ( L ) } | W } \left\| \ \widetilde { Q } _ { \mathbf { U } ^ { ( L ) } | W } \right) \Big ] = \mathcal { C } ^ { ( L ) } .
$$

Hence

$$
\left| \mathbb { E } [ \Delta _ { \mathrm { v a r } } ^ { ( L ) } ] \right| \leq \sqrt { \frac { 2 \sigma ^ { 2 } } { N _ { \mathrm { e f f } } } } \mathcal { C } ^ { ( L ) } .
$$

Then ${ \cal S } ^ { ( L ) } , \mathcal { P } ^ { ( L ) } , \mathcal { R } ^ { ( L ) }$ are mutual/interaction information involving $W _ { L + 1 : L }$ conditioned on $W _ { 1 : L } = W$ . Since $W _ { L + 1 : L }$ is deterministic (empty), all these quantities are zero:

$$
\mathcal { S } ^ { ( L ) } = \mathcal { P } ^ { ( L ) } = \mathcal { R } ^ { ( L ) } = 0 .
$$

Combining the drift and variance bounds completes the proof.

## D. Proof of Theorem IV.4

Theorem IV.4 (Restate). Assume the loss function $\ell ( \cdot , y )$ is $\rho _ { 0 }$ -Lipschitz and the activation functions $\phi _ { l }$ are ρ -Lipschitz. Then, at time T, we have:

$$
\left. \mathrm { g e n } _ { W } \right. \leq \operatorname* { m i n } _ { l \in \{ 0 , \ldots , L \} } \mathbb { E } \left[ \bar { \rho } _ { l } ( W ) \cdot \sum _ { i = 1 } ^ { T } \mathcal { W } _ { 1 } \Bigl ( \hat { P } _ { A _ { l } , Y | S ^ { i } , W _ { 1 : l } } , P _ { A _ { l } , Y | i , W _ { 1 : l } } \Bigr ) \right] ,
$$

where $\begin{array} { r } { \bar { \rho } _ { l } ( W ) : = \rho _ { 0 } \left( 1 \vee \prod _ { h = l + 1 } ^ { L } \rho _ { h } \| W _ { h } \| _ { \mathrm { o p } } \right) } \end{array}$

Proof. Fix an arbitrary $l \in \{ 0 , 1 , \ldots , L \}$ . Define the upper sub-network mapping from layer l to the output: $G _ { W _ { l + 1 : L } } ( a ) : =$ $g _ { W _ { L } } \circ g _ { W _ { L - 1 } } \circ \cdot \cdot \cdot \circ g _ { W _ { l + 1 } } ( a )$ , and define the re-parameterized loss on $( a , y ) \in \mathcal { A } _ { l } \times \mathcal { Y } \colon f _ { W } ^ { ( l ) } ( a , y ) : = \ell \bigl ( G _ { W _ { l + 1 : L } } ( a ) , y \bigr )$ . We first bound the Lipschitz constant of $f _ { W } ^ { ( l ) }$ . First, for any layer $h \in \{ l + 1 , \ldots , L \}$ and any $u , u ^ { \prime } ,$

$$
\begin{array} { r l } & { \| g _ { W _ { h } } ( u ) - g _ { W _ { h } } ( u ^ { \prime } ) \| _ { 2 } = \| \phi _ { h } ( W _ { h } u ) - \phi _ { h } ( W _ { h } u ^ { \prime } ) \| _ { 2 } } \\ & { \qquad \leq \rho _ { h } \| W _ { h } u - W _ { h } u ^ { \prime } \| _ { 2 } } \\ & { \qquad \leq \rho _ { h } \| W _ { h } \| _ { \mathrm { o p } } \| u - u ^ { \prime } \| _ { 2 } . } \end{array}
$$

Hence $g _ { W _ { h } }$ is $\rho _ { h } \| W _ { h } \| _ { \mathrm { o p } } .$ -Lipschitz. Next, define for integers $p \leq q$ the partial composition $G _ { p : q } : = g _ { W _ { q } } \circ g _ { W _ { q - 1 } } \circ \cdot \cdot \cdot \circ g _ { W _ { p } }$ Then for any $u , u ^ { \prime }$

$$
\begin{array} { r l } { \Vert G _ { p : q } ( u ) - G _ { p : q } ( u ^ { \prime } ) \Vert _ { 2 } = \Vert g _ { W _ { q } } ( G _ { p : q - 1 } ( u ) ) - g _ { W _ { q } } ( G _ { p : q - 1 } ( u ^ { \prime } ) ) \Vert _ { 2 } } & { } \\ { \leq \mathrm { L i p } ( g _ { W _ { q } } ) \cdot \Vert G _ { p : q - 1 } ( u ) - G _ { p : q - 1 } ( u ^ { \prime } ) \Vert _ { 2 } } & { } \\ { \leq \left( \rho _ { q } \Vert W _ { q } \Vert _ { \mathrm { o p } } \right) \cdot \mathrm { L i p } ( G _ { p : q - 1 } ) \cdot \Vert u - u ^ { \prime } \Vert _ { 2 } } & { } \\ { \leq ( \rho _ { q } \Vert W _ { q } \Vert _ { \mathrm { o p } } ) \displaystyle \left( \displaystyle \prod _ { h = p } ^ { q - 1 } \rho _ { h } \Vert W _ { h } \Vert _ { \mathrm { o p } } \right) \Vert u - u ^ { \prime } \Vert _ { 2 } , } & { } \end{array}
$$

Taking $p = l + 1$ and $q = L$ gives

$$
\| G _ { W _ { l + 1 : L } } ( a ) - G _ { W _ { l + 1 : L } } ( a ^ { \prime } ) \| _ { 2 } \leq \left( \prod _ { h = l + 1 } ^ { L } \rho _ { h } \| W _ { h } \| _ { \mathrm { o p } } \right) \| a - a ^ { \prime } \| _ { 2 } .
$$

Denote $\begin{array} { r } { \alpha _ { l } ( W ) : = \prod _ { h = l + 1 } ^ { L } \rho _ { h } \| W _ { h } \| _ { \mathrm { o p } } } \end{array}$ for short.

Using that $\ell ( \hat { y } , y )$ is $\rho _ { 0 } \mathbf { - } \mathbf { } \mathbf { } \mathbf { } \mathbf { I }$ Lipschitz in $( \hat { y } , y )$ under the Euclidean norm, for any $( a , y ) , ( a ^ { \prime } , y ^ { \prime } )$ we have

$$
\begin{array} { r l } & { | f _ { W , l } ( a , y ) - f _ { W , l } ( a ^ { \prime } , y ^ { \prime } ) | = | \ell ( G _ { W _ { l + 1 : L } } ( a ) , y ) - \ell ( G _ { W _ { l + 1 : L } } ( a ^ { \prime } ) , y ^ { \prime } ) | } \\ & { \qquad \leq \rho _ { 0 } \sqrt { \| G _ { W _ { l + 1 : L } } ( a ) - G _ { W _ { l + 1 : L } } ( a ^ { \prime } ) \| _ { 2 } ^ { 2 } + \| y - y ^ { \prime } \| _ { 2 } ^ { 2 } } } \\ & { \qquad \leq \rho _ { 0 } \sqrt { \alpha _ { l } ( W ) ^ { 2 } \| a - a ^ { \prime } \| _ { 2 } ^ { 2 } + \| y - y ^ { \prime } \| _ { 2 } ^ { 2 } } . } \end{array}
$$

If $\alpha _ { l } ( W ) \leq 1$ , then $\alpha _ { l } ( W ) ^ { 2 } \lVert a - a ^ { \prime } \rVert _ { 2 } ^ { 2 } + \lVert y - y ^ { \prime } \rVert _ { 2 } ^ { 2 } \leq \lVert a - a ^ { \prime } \rVert _ { 2 } ^ { 2 } + \lVert y - y ^ { \prime } \rVert _ { 2 } ^ { 2 } . \mathrm { ~ I f ~ } \alpha _ { l } ( W ) \geq 1$ , then $\alpha _ { l } ( W ) ^ { 2 } \| a - a ^ { \prime } \| _ { 2 } ^ { 2 } + \| y - y ^ { \prime } \| _ { 2 } ^ { 2 } \leq$ $\alpha _ { l } ( W ) ^ { 2 } ( \| a - a ^ { \prime } \| _ { 2 } ^ { 2 } + \| y - y ^ { \prime } \| _ { 2 } ^ { 2 } )$ . In both cases,

$$
\begin{array} { r } { \sqrt { \alpha _ { l } ( W ) ^ { 2 } \| a - a ^ { \prime } \| _ { 2 } ^ { 2 } + \| y - y ^ { \prime } \| _ { 2 } ^ { 2 } } \leq ( 1 \vee \alpha _ { l } ( W ) ) \sqrt { \| a - a ^ { \prime } \| _ { 2 } ^ { 2 } + \| y - y ^ { \prime } \| _ { 2 } ^ { 2 } } = ( 1 \vee \alpha _ { l } ( W ) ) d \big ( ( a , y ) , ( a ^ { \prime } , y ^ { \prime } ) \big ) . } \end{array}
$$

Therefore,

$$
\begin{array} { r } { | f _ { W , l } ( a , y ) - f _ { W , l } ( a ^ { \prime } , y ^ { \prime } ) | \leq \rho _ { 0 } ( 1 \vee \alpha _ { l } ( W ) ) d \big ( ( a , y ) , ( a ^ { \prime } , y ^ { \prime } ) \big ) = \bar { \rho } _ { l } ( W ) d \big ( ( a , y ) , ( a ^ { \prime } , y ^ { \prime } ) \big ) , } \end{array}
$$

which shows $\mathrm { L i p } ( f _ { W , l } ) \le \bar { \rho } _ { l } ( W )$ with $\bar { \rho } _ { l } ( W )$ defined

$$
\bar { \rho } _ { l } ( W ) : = \rho _ { 0 } \left( 1 \vee \prod _ { h = l + 1 } ^ { L } \rho _ { h } \| W _ { h } \| _ { \mathrm { o p } } \right) .\tag{55}
$$

For each task $i \in [ T ]$ , by definition of $P _ { A _ { l } , Y | i , W _ { 1 : l } }$ we have

$$
\begin{array} { r } { \mathbb { E } _ { Z \sim \mathcal { D } _ { i } } \ell ( W , Z ) = \mathbb { E } _ { ( A _ { l } , Y ) \sim P _ { A _ { l } , Y \mid i , W _ { 1 : l } } } f _ { W , l } ( A _ { l } , Y ) . } \end{array}
$$

Moreover, by definition of $\hat { P } _ { A _ { l } , Y \mid S ^ { i } , W _ { 1 : l } }$

$$
\begin{array} { r } { \mathbb { E } _ { U \sim \hat { P } _ { A _ { l } , Y | S ^ { i } , W _ { 1 : l } } } f _ { W , l } ( U ) = \left\{ \begin{array} { l l } { \frac { 1 } { m } \sum _ { j \in \mathcal { T } _ { i } } f _ { W , l } ( A _ { j , l } ^ { i } , Y _ { j } ^ { i } ) = \frac { 1 } { m } \sum _ { j \in \mathcal { T } _ { i } } \ell ( W , Z _ { j } ^ { i } ) , } & { i \in [ T - 1 ] , } \\ { \frac { 1 } { n } \sum _ { j = 1 } ^ { n } f _ { W , l } ( A _ { j , l } ^ { T } , Y _ { j } ^ { T } ) = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \ell ( W , Z _ { j } ^ { T } ) , } & { i = T . } \end{array} \right. } \end{array}
$$

Substituting these identities into population $\mathcal { L } ( W )$ and empirical risks $\hat { \mathcal { L } } ( W )$ yields, for the fixed realization,

$$
\mathcal { L } ( W ) - \hat { \mathcal { L } } ( W ) = \sum _ { i = 1 } ^ { T } \left( \mathbb { E } _ { P _ { A _ { l } , Y \mid i , W _ { 1 : l } } } \left[ f _ { W , l } \right] - \mathbb { E } _ { \hat { P } _ { A _ { l } , Y \mid S ^ { i } , W _ { 1 : l } } } \left[ f _ { W , l } \right] \right) .
$$

By Kantorovich–Rubinstein Lemma A.8 for the 1-Wasserstein distance, for any Lipschitz function $f$ on $( \mathcal { A } _ { l } \times \mathcal { Y } , d )$

$$
\left| \mathbb { E } _ { \mu } f - \mathbb { E } _ { \nu } f \right| \leq \operatorname { L i p } ( f ) \mathcal { W } _ { 1 } ( \mu , \nu ) .
$$

Applying this with $f = f _ { W , l } , \mu = P _ { A _ { l } , Y | i , W _ { 1 : l } }$ , and $\nu = \hat { P } _ { A _ { l } , Y | S ^ { i } , W _ { 1 : l } }$ , and using $\mathrm { L i p } ( f _ { W , l } ) \le \bar { \rho } _ { l } ( W )$ from the previous steps, we obtain for each $i \in [ T ]$

$$
\begin{array} { r } { \left| \mathbb { E } _ { P _ { A _ { l } , Y | i } , w _ { 1 : l } } f _ { W , l } - \mathbb { E } _ { \hat { P } _ { A _ { l } , Y | S ^ { i } , W _ { 1 : l } } } f _ { W , l } \right| \le \bar { \rho } _ { l } ( W ) \mathcal { W } _ { 1 } \Bigl ( \hat { P } _ { A _ { l } , Y | S ^ { i } , W _ { 1 : l } } , P _ { A _ { l } , Y | i , W _ { 1 : l } } \Bigr ) . } \end{array}
$$

Hence, by the triangle inequality,

$$
\left| \mathcal { L } ( W ) - \hat { \mathcal { L } } ( W ) \right| \leq \bar { \rho } _ { l } ( W ) \cdot \sum _ { i = 1 } ^ { T } \mathcal { W } _ { 1 } \Bigl ( \hat { P } _ { A _ { l } , Y | S ^ { i } , W _ { 1 : l } } , P _ { A _ { l } , Y | i , W _ { 1 : l } } \Bigr ) .
$$

Now take expectation over all randomness, including the generally dependent training sequence law $P _ { \mathbf { U } ^ { ( l ) } | W _ { 1 : l } } \colon$

$$
\left| \operatorname { g c n } _ { W } \right| = \left| \mathbb { E } \left[ \mathcal { L } ( W ) - \hat { \mathcal { L } } ( W ) \right] \right| \leq \mathbb { E } \left| \mathcal { L } ( W ) - \hat { \mathcal { L } } ( W ) \right| \leq \mathbb { E } \left[ \bar { \rho } _ { l } ( W ) \cdot \sum _ { i = 1 } ^ { T } \mathcal { W } _ { 1 } \Bigl ( \hat { P } _ { A _ { l } , Y | S ^ { i } , W _ { 1 : l } } , P _ { A _ { l } , Y | i , W _ { 1 : l } } \Bigr ) \right] .\tag{56}
$$

Since the bound (56) holds for every $l \in \{ 0 , \ldots , L \}$ , taking the minimum over $l$ gives,

$$
\left| \mathrm { g e n } _ { W } \right| \leq \operatorname* { m i n } _ { l \in \{ 0 , \ldots , L \} } \mathbb { E } \left[ \bar { \rho } _ { l } ( W ) \cdot \sum _ { i = 1 } ^ { T } \mathcal { W } _ { 1 } \Bigl ( \hat { P } _ { A _ { l } , Y | S ^ { i } , W _ { 1 : l } } , P _ { A _ { l } , Y | i , W _ { 1 : l } } \Bigr ) \right] .\tag{57}
$$

## E. Proof of Proposition V.1

Proposition V.1 (Restate). Fix a layer l and condition on $W _ { 1 : l } . \ \mathrm { A t }$ task $T ,$ we have

$$
\begin{array} { r l } & { I ( \mathbf { U } _ { T } ^ { ( l ) } ; \boldsymbol { \Theta } _ { R } ^ { ( T ) } | W _ { 1 : l } ) \leq I ( \mathbf { U } _ { T } ^ { ( l ) } ; \boldsymbol { \Theta } _ { 0 } ^ { ( T ) } | W _ { 1 : l } ) + I ( \mathbf { U } _ { T } ^ { ( l ) } ; \boldsymbol { \Theta } _ { R } ^ { ( T ) } | \boldsymbol { \Theta } _ { 0 } ^ { ( T ) } , W _ { 1 : l } ) } \\ & { \qquad = : H _ { T } ^ { ( l ) } + \Delta _ { T } ^ { ( l ) } . } \end{array}
$$

Proof. Condition on $W _ { 1 : l }$ and suppress it in the notation. By data processing, $I ( \mathbf { U } _ { T } ^ { ( l ) } ; \Theta _ { R } ^ { ( T ) } ) \leq I ( \mathbf { U } _ { T } ^ { ( l ) } ; ( \Theta _ { 0 } ^ { ( T ) } , \Theta _ { R } ^ { ( T ) } ) )$ . Applying the chain rule for mutual information yields

$$
{ \cal I } ( { \bf U } _ { T } ^ { ( l ) } ; ( \Theta _ { 0 } ^ { ( T ) } , \Theta _ { R } ^ { ( T ) } ) ) ~ = ~ { \cal I } ( { \bf U } _ { T } ^ { ( l ) } ; \Theta _ { 0 } ^ { ( T ) } ) + { \cal I } ( { \bf U } _ { T } ^ { ( l ) } ; \Theta _ { R } ^ { ( T ) } \mid \Theta _ { 0 } ^ { ( T ) } ) ,
$$

which proves the Proposition.

## F. Proof of Corollary V.2

Corollary V.2 (Restate). Fix a layer l and condition on $W _ { 1 : l }$ . For any $T \geq 2 .$ , the heritage term admits the following cumulativebudget upper bound:

$$
H _ { T } ^ { ( l ) } = I ( \mathbf { U } _ { T } ^ { ( l ) } ; \Theta _ { 0 } ^ { ( T ) } \mid W _ { 1 : l } ) \leq \sum _ { t = 1 } ^ { T - 1 } \Delta _ { t } ^ { ( l ) } ,
$$

where $\Delta _ { t } ^ { ( l ) } : = I ( \mathbf { U } _ { t } ^ { ( l ) } ; \Theta _ { R } ^ { ( t ) } \mid \Theta _ { 0 } ^ { ( t ) } , W _ { 1 : l } )$ is the within-task information increment in Proposition V.1 applied to task t.

Proof. Throughout the proof we implicitly condition on $W _ { 1 : l }$ to simplify notation. By the recursion $\Theta _ { 0 } ^ { ( T ) } = \Theta _ { R } ^ { ( T - 1 ) }$ , we have $\big . H _ { T } ^ { ( l ) } = I ( \mathbf { U } _ { T } ^ { ( l ) } ; \boldsymbol { \Theta } _ { R } ^ { ( T - 1 ) } )$ . The task-t training sequence $\mathbf { U } _ { t } ^ { ( l ) }$ is generated by sampling from (i) the current-task dataset and (ii) a replay memory $\mathcal { M } ^ { 1 : t - 1 }$ built from the past, and this sampling procedure does not depend on $\Theta _ { R } ^ { ( t - 1 ) }$ once $\mathcal { M } ^ { 1 : t - 1 }$ is fixed. Then,

$$
\mathbf { U } _ { t } ^ { ( l ) } \perp \Theta _ { R } ^ { ( t - 1 ) } \mid \mathcal { M } ^ { 1 : t - 1 } .
$$

Therefore, at task $T ,$ we have $\mathbf { U } _ { T } ^ { ( l ) } \perp \Theta _ { R } ^ { ( T - 1 ) } \mid \mathcal { M } ^ { 1 : T - 1 }$ , which implies the Markov chain

$$
{ \bf U } _ { T } ^ { ( l ) } \to { \mathcal { M } } ^ { 1 : T - 1 } \to \Theta _ { R } ^ { ( T - 1 ) } .
$$

By the data processing inequality,

$$
H _ { T } ^ { ( l ) } = I ( \mathbf { U } _ { T } ^ { ( l ) } ; \Theta _ { R } ^ { ( T - 1 ) } ) \leq I ( \mathcal { M } ^ { 1 : T - 1 } ; \Theta _ { R } ^ { ( T - 1 ) } ) .\tag{58}
$$

Moreover, since $\mathcal { M } ^ { 1 : T - 1 }$ is measurable with respect to $\mathbf { U } _ { 1 : T - 1 } ^ { ( l ) }$ , we have Markov chains again,

$$
\Theta _ { R } ^ { ( T - 1 ) } \to \mathbf { U } _ { 1 : T - 1 } ^ { ( l ) } \to \mathcal { M } ^ { 1 : T - 1 } .
$$

By the data processing inequality,

$$
I ( \mathcal { M } ^ { 1 : T - 1 } ; \Theta _ { R } ^ { ( T - 1 ) } ) \le I ( \mathbf { U } _ { 1 : T - 1 } ^ { ( l ) } ; \Theta _ { R } ^ { ( T - 1 ) } ) .\tag{59}
$$

For each $t \geq 1$ , define $J _ { t } : = I ( \mathbf { U } _ { 1 : t } ^ { ( l ) } ; \boldsymbol { \Theta } _ { R } ^ { ( t ) } )$ . We claim that for every $t \geq 1$

$$
J _ { t } \ \le \ J _ { t - 1 } + \Delta _ { t } ^ { ( l ) } .\tag{60}
$$

Next, our goal is to prove (60). Based on the monotonicity of mutual information since $\Theta _ { R } ^ { ( t ) }$ is a component of $( \Theta _ { 0 } ^ { ( t ) } , \Theta _ { R } ^ { ( t ) } )$ , we have

$$
I ( \mathbf { U } _ { 1 : t } ^ { ( l ) } ; \Theta _ { R } ^ { ( t ) } ) \leq I ( \mathbf { U } _ { 1 : t } ^ { ( l ) } ; \Theta _ { 0 } ^ { ( t ) } , \Theta _ { R } ^ { ( t ) } ) .\tag{61}
$$

Apply the chain rule to the RHS:

$$
I ( \mathbf { U } _ { 1 : t } ^ { ( l ) } ; \Theta _ { 0 } ^ { ( t ) } , \Theta _ { R } ^ { ( t ) } ) = I ( \mathbf { U } _ { 1 : t } ^ { ( l ) } ; \Theta _ { 0 } ^ { ( t ) } ) + I ( \mathbf { U } _ { 1 : t } ^ { ( l ) } ; \Theta _ { R } ^ { ( t ) } \mid \Theta _ { 0 } ^ { ( t ) } ) .\tag{62}
$$

We handle the two terms in (62) separately.

Since $\Theta _ { 0 } ^ { ( t ) } = \Theta _ { R } ^ { ( t - 1 ) }$ is fully determined by the past training procedure, conditioning on $\mathbf { U } _ { 1 : t - 1 } ^ { ( l ) }$ leaves no additional dependence on $\mathbf { U } _ { t } ^ { ( l ) }$ , thus:

$$
I ( \mathbf { U } _ { t } ^ { ( l ) } ; \Theta _ { 0 } ^ { ( t ) } \mid \mathbf { U } _ { 1 : t - 1 } ^ { ( l ) } ) = 0 .
$$

Then, by the chain rule,

$$
\begin{array} { r l } & { I ( \mathbf { U } _ { 1 : t } ^ { ( l ) } ; \Theta _ { 0 } ^ { ( t ) } ) = I ( \mathbf { U } _ { 1 : t - 1 } ^ { ( l ) } ; \Theta _ { 0 } ^ { ( t ) } ) + I ( \mathbf { U } _ { t } ^ { ( l ) } ; \Theta _ { 0 } ^ { ( t ) } \mid \mathbf { U } _ { 1 : t - 1 } ^ { ( l ) } ) } \\ & { \qquad = I ( \mathbf { U } _ { 1 : t - 1 } ^ { ( l ) } ; \Theta _ { 0 } ^ { ( t ) } ) = I ( \mathbf { U } _ { 1 : t - 1 } ^ { ( l ) } ; \Theta _ { R } ^ { ( t - 1 ) } ) = J _ { t - 1 } . } \end{array}\tag{63}
$$

Decompose $\mathbf { U } _ { 1 : t } ^ { ( l ) } = ( \mathbf { U } _ { 1 : t - 1 } ^ { ( l ) } , \mathbf { U } _ { t } ^ { ( l ) } )$ and apply the chain rule:

$$
I ( \mathbf { U } _ { 1 : t } ^ { ( l ) } ; \Theta _ { R } ^ { ( t ) } \mid \Theta _ { 0 } ^ { ( t ) } ) = I ( \mathbf { U } _ { t } ^ { ( l ) } ; \Theta _ { R } ^ { ( t ) } \mid \Theta _ { 0 } ^ { ( t ) } ) + I ( \mathbf { U } _ { 1 : t - 1 } ^ { ( l ) } ; \Theta _ { R } ^ { ( t ) } \mid \Theta _ { 0 } ^ { ( t ) } , \mathbf { U } _ { t } ^ { ( l ) } ) .\tag{64}
$$

For each t, the final parameter $\Theta _ { R } ^ { ( t ) }$ is conditionally independent of the past training sequences $\mathbf { U } _ { 1 : t - 1 } ^ { ( l ) }$ given the initialization $\Theta _ { 0 } ^ { ( t ) }$ and the current training sequence $\mathbf { U } _ { t } ^ { ( l ) }$ :

$$
\Theta _ { R } ^ { ( t ) } \perp { \bf U } _ { 1 : t - 1 } ^ { ( l ) } \mid ( \Theta _ { 0 } ^ { ( t ) } , { \bf U } _ { t } ^ { ( l ) } , W _ { 1 : l } ) .
$$

Therefore the second term $I ( \mathbf { U } _ { 1 : t - 1 } ^ { ( l ) } ; \boldsymbol { \Theta } _ { R } ^ { ( t ) } \mid \boldsymbol { \Theta } _ { 0 } ^ { ( t ) } , \mathbf { U } _ { t } ^ { ( l ) } )$ on the RHS of (64) is zero, hence

$$
I ( \mathbf { U } _ { 1 : t } ^ { ( l ) } ; \boldsymbol { \Theta } _ { R } ^ { ( t ) } \mid \boldsymbol { \Theta } _ { 0 } ^ { ( t ) } ) = I ( \mathbf { U } _ { t } ^ { ( l ) } ; \boldsymbol { \Theta } _ { R } ^ { ( t ) } \mid \boldsymbol { \Theta } _ { 0 } ^ { ( t ) } ) = \Delta _ { t } ^ { ( l ) } .\tag{65}
$$

Plugging (63) and (65) into (62), and then using (61), yields

$$
J _ { t } = I ( \mathbf { U } _ { 1 : t } ^ { ( l ) } ; \Theta _ { R } ^ { ( t ) } ) \leq I ( \mathbf { U } _ { 1 : t } ^ { ( l ) } ; \Theta _ { 0 } ^ { ( t ) } , \Theta _ { R } ^ { ( t ) } ) = J _ { t - 1 } + \Delta _ { t } ^ { ( l ) } ,
$$

which establishes (60). Iterating (60) from $t = 1$ to $t = T - 1$ gives

$$
J _ { T - 1 } \leq J _ { 0 } + \sum _ { t = 1 } ^ { T - 1 } \Delta _ { t } ^ { ( l ) } .
$$

Because parameters are randomly initialized before network training, therefore $\Theta _ { 0 } ^ { ( 1 ) }$ is independent of $\mathbf { U } _ { 1 } ^ { ( l ) }$ , which implies $J _ { 0 } = I ( \bar { \mathbf { U } } _ { 1 : 0 } ^ { ( l ) } ; \boldsymbol { \Theta } _ { R } ^ { ( 0 ) } ) = 0$ , we obtain

$$
J _ { T - 1 } \leq \sum _ { t = 1 } ^ { T - 1 } \Delta _ { t } ^ { ( l ) } .\tag{66}
$$

From (58)–(59) we have

$$
H _ { T } ^ { ( l ) } \leq I ( \mathbf { U } _ { 1 : T - 1 } ^ { ( l ) } ; \Theta _ { R } ^ { ( T - 1 ) } ) = J _ { T - 1 } .
$$

Combining with (66) yields the desired result:

$$
H _ { T } ^ { ( l ) } \leq \sum _ { t = 1 } ^ { T - 1 } \Delta _ { t } ^ { ( l ) } .
$$

## G. Proof of Theorem V.3

Theorem V.3 (Restate). Fix a split layer $l \in \{ 0 , \ldots , L - 1 \}$ and condition on the base parameters $W _ { 1 : l } ,$ so that the suffix network remains nonempty. Assume the loss is σ-subgaussian as in Theorem IV.1. For task t trained by SGLD for $R _ { t }$ steps, we have

$$
\left. \mathrm { g e n } _ { W ^ { ( t ) } } \right. \leq ( t - 1 ) \sqrt { 2 \sigma ^ { 2 } { \mathcal K } _ { t } ^ { ( l ) } } + \sqrt { \frac { 2 \sigma ^ { 2 } } { N _ { \mathrm { e f f } , t } } } \Bigl ( { \mathcal C } _ { t } ^ { ( l ) } + \sum _ { s = 1 } ^ { t } \sum _ { r = 1 } ^ { R _ { s } } { \mathbb E } \left[ \frac 1 2 \log \operatorname* { d e t } \Bigl ( I + \frac { \eta _ { s , r } ^ { 2 } } { \tau _ { s , r } ^ { 2 } } M _ { s , r } ^ { ( l ) } \Bigr ) \right] \Bigr ) ,
$$

where $M _ { s , r } ^ { ( l ) } = \mathbb { E } \big [ G _ { s , r } G _ { s , r } ^ { \top } \big | \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } \big ]$ , and $\kappa _ { t } ^ { ( l ) } , \mathcal { C } _ { t } ^ { ( l ) }$ , and $N _ { \mathrm { e f f } , t }$ denote the corresponding quantities of Theorem IV.1 with the task horizon T replaced by t.

Proof. Throughout the proof we implicitly fix the split layer l and condition on $W _ { 1 : l }$ to simplify notation. Apply Theorem IV.1 to the t-task scenario by substituting t for T in the statement of Theorem IV.1. Then for any fixed split layer l,

$$
\left. \mathrm { g e n } _ { W ^ { ( t ) } } \right. \leq ( t - 1 ) \sqrt { 2 \sigma ^ { 2 } { K _ { t } ^ { ( l ) } } } + \sqrt { \frac { 2 \sigma ^ { 2 } } { N _ { \mathrm { e f f } , t } } \Bigl ( S _ { t } ^ { ( l ) } + \mathcal { P } _ { t } ^ { ( l ) } - \mathcal { R } _ { t } ^ { ( l ) } + \mathcal { C } _ { t } ^ { ( l ) } \Bigr ) } .\tag{67}
$$

By the interaction-information identity used in Section V-A, the SPS combination equals a conditional mutual information between the task-t training sequence and the final upper parameters:

$$
\mathscr { S } _ { t } ^ { ( l ) } + \mathscr { P } _ { t } ^ { ( l ) } - \mathscr { R } _ { t } ^ { ( l ) } = I \Big ( \mathbf { U } _ { t } ^ { ( l ) } ; \boldsymbol { \Theta } _ { R _ { t } } ^ { ( t ) } \mid W _ { 1 : l } \Big ) .\tag{68}
$$

$$
\Delta _ { s } ^ { ( l ) } : = { \cal I } \left( { \bf U } _ { s } ^ { ( l ) } ; \Theta _ { R _ { s } } ^ { ( s ) } \mid \Theta _ { 0 } ^ { ( s ) } , W _ { 1 : l } \right) ,
$$

Thus, it suffices to upper bound $I \left( \mathbf { U } _ { t } ^ { ( l ) } ; \Theta _ { R _ { t } } ^ { ( t ) } \mid W _ { 1 : l } \right)$ by a cumulative, algorithmic quantity. For each task index $s \geq 1$ , define the within-task increment

and the heritage term

$$
H _ { s } ^ { ( l ) } : = I \left( \mathbf { U } _ { s } ^ { ( l ) } ; \Theta _ { 0 } ^ { ( s ) } \mid W _ { 1 : l } \right) , \qquad \mathrm { w h e r e ~ } \Theta _ { 0 } ^ { ( s ) } = \Theta _ { R _ { s - 1 } } ^ { ( s - 1 ) } \mathrm { ~ f o r ~ } s \ge 2 .
$$

Applying Proposition V.1 at task s we have decomposition,

$$
\begin{array} { r } { I \Big ( \mathbf { U } _ { s } ^ { ( l ) } ; \boldsymbol { \Theta } _ { R _ { s } } ^ { ( s ) } \mid W _ { 1 : l } \Big ) \leq H _ { s } ^ { ( l ) } + \Delta _ { s } ^ { ( l ) } . } \end{array}\tag{69}
$$

Applying Corollary V.2, we can obtain a sum-to-t recursion,

$$
{ \cal I } \Big ( { \bf U } _ { t } ^ { ( l ) } ; \Theta _ { R _ { t } } ^ { ( t ) } \mid W _ { 1 : l } \Big ) \le \sum _ { s = 1 } ^ { t - 1 } \Delta _ { s } ^ { ( l ) } + \Delta _ { t } ^ { ( l ) } = \sum _ { s = 1 } ^ { t } \Delta _ { s } ^ { ( l ) } .\tag{70}
$$

Fix a task $s \in \{ 1 , \ldots , t \}$ . Let $\Theta _ { 0 : R _ { s } } ^ { ( s ) } : = ( \Theta _ { 0 } ^ { ( s ) } , \Theta _ { 1 } ^ { ( s ) } , \dots , \Theta _ { R _ { s } } ^ { ( s ) } )$ denote the full within-task trajectory. Since $\Theta _ { R _ { s } } ^ { ( s ) }$ is a coordinate of $\Theta _ { 0 : R _ { s } } ^ { ( s ) }$ , by monotonicity of mutual information,

$$
I \Big ( \mathbf { U } _ { s } ^ { ( l ) } ; \boldsymbol { \Theta } _ { R _ { s } } ^ { ( s ) } \big | \boldsymbol { \Theta } _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) \leq I \Big ( \mathbf { U } _ { s } ^ { ( l ) } ; \boldsymbol { \Theta } _ { 0 : R _ { s } } ^ { ( s ) } \big | \boldsymbol { \Theta } _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) .
$$

Apply the chain rule:

$$
I \Big ( \mathbf { U } _ { s } ^ { ( l ) } ; \boldsymbol { \Theta } _ { 0 : R _ { s } } ^ { ( s ) } \big | \boldsymbol { \Theta } _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) = \sum _ { r = 1 } ^ { R _ { s } } I \Big ( \mathbf { U } _ { s } ^ { ( l ) } ; \boldsymbol { \Theta } _ { r } ^ { ( s ) } \big | \boldsymbol { \Theta } _ { 0 : r - 1 } ^ { ( s ) } , \boldsymbol { \Theta } _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) ,
$$

where there is no $r = 0$ term because $\Theta _ { 0 } ^ { ( s ) }$ is already conditioned on. For each $r \geq 1$ , write the conditional mutual information as a differential entropy difference:

$$
\begin{array} { r } { I \Big ( \mathbf { U } _ { s } ^ { ( l ) } ; \boldsymbol { \Theta } _ { r } ^ { ( s ) } \big \vert \ \boldsymbol { \Theta } _ { 0 : r - 1 } ^ { ( s ) } , \boldsymbol { \Theta } _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) = h \Big ( \boldsymbol { \Theta } _ { r } ^ { ( s ) } \big \vert \ \boldsymbol { \Theta } _ { 0 : r - 1 } ^ { ( s ) } , \boldsymbol { \Theta } _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) - h \Big ( \boldsymbol { \Theta } _ { r } ^ { ( s ) } \big \vert \ \mathbf { U } _ { s } ^ { ( l ) } , \boldsymbol { \Theta } _ { 0 : r - 1 } ^ { ( s ) } , \boldsymbol { \Theta } _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) . } \end{array}
$$

First, conditioning reduces entropy, hence

$$
h \Big ( \Theta _ { r } ^ { ( s ) } \big | \Theta _ { 0 : r - 1 } ^ { ( s ) } , \Theta _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) \leq h \Big ( \Theta _ { r } ^ { ( s ) } \big | \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } \Big ) .
$$

Second, by the Markov structure of the update rule, given $( \mathbf { U } _ { s } ^ { ( l ) } , \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } )$ , the next iterate $\Theta _ { r } ^ { ( s ) }$ is generated using only the freshly sampled minibatch $B _ { s , r }$ (sampled from $\mathbf { U } _ { s } ^ { ( l ) } )$ and the fresh Gaussian noise $N _ { s , r } .$ , so it is conditionally independent of $\Theta _ { 0 : r - 2 } ^ { ( s ) }$ . Therefore

$$
h \Big ( \Theta _ { r } ^ { ( s ) } \big \vert \mathbf { U } _ { s } ^ { ( l ) } , \Theta _ { 0 : r - 1 } ^ { ( s ) } , \Theta _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) = h \Big ( \Theta _ { r } ^ { ( s ) } \big \vert \mathbf { U } _ { s } ^ { ( l ) } , \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } \Big ) .
$$

Combining these two displays gives, for each $r ,$

$$
I \Big ( \mathbf { U } _ { s } ^ { ( l ) } ; \boldsymbol { \Theta } _ { r } ^ { ( s ) } \big | \Theta _ { 0 : r - 1 } ^ { ( s ) } , \Theta _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) \leq I \Big ( \mathbf { U } _ { s } ^ { ( l ) } ; \boldsymbol { \Theta } _ { r } ^ { ( s ) } \big | \Theta _ { r - 1 } ^ { ( s ) } , \Theta _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) .
$$

Summing over r yields

$$
\Delta _ { s } ^ { ( l ) } = I \Big ( \mathbf { U } _ { s } ^ { ( l ) } ; \Theta _ { R _ { s } } ^ { ( s ) } \big | \Theta _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) \leq \sum _ { r = 1 } ^ { R _ { s } } I \Big ( \mathbf { U } _ { s } ^ { ( l ) } ; \Theta _ { r } ^ { ( s ) } \big | \Theta _ { r - 1 } ^ { ( s ) } , \Theta _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) .
$$

Now fix r and apply data processing from the full data $\mathbf { U } _ { s } ^ { ( l ) }$ to the sampled minibatch $B _ { s , r } \colon$ given $( \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } )$ , the algorithm draws $B _ { s , r }$ by randomized sampling from $\mathbf { U } _ { s } ^ { ( l ) }$ , and the update mapping from $B _ { s , i }$ <sub>r</sub> to $\Theta _ { r } ^ { ( s ) }$ depends on $\mathbf { U } _ { s } ^ { ( l ) }$ only through $B _ { s , r }$ . Thus we have the Markov chain

$$
{ \bf U } _ { s } ^ { ( l ) } \longrightarrow B _ { s , r } \longrightarrow \Theta _ { r } ^ { ( s ) } \quad \mathrm { c o n d i t i o n e d ~ o n ~ } ( \Theta _ { r - 1 } ^ { ( s ) } , \Theta _ { 0 } ^ { ( s ) } , W _ { 1 : l } ) ,
$$

and hence

$$
I \Big ( \mathbf { U } _ { s } ^ { ( l ) } ; \boldsymbol { \Theta } _ { r } ^ { ( s ) } \bigm | \boldsymbol { \Theta } _ { r - 1 } ^ { ( s ) } , \boldsymbol { \Theta } _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) \leq I \Big ( B _ { s , r } ; \boldsymbol { \Theta } _ { r } ^ { ( s ) } \bigm | \boldsymbol { \Theta } _ { r - 1 } ^ { ( s ) } , \boldsymbol { \Theta } _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) .
$$

Therefore,

$$
\Delta _ { s } ^ { ( l ) } \leq \sum _ { r = 1 } ^ { R _ { s } } I \Big ( B _ { s , r } ; \Theta _ { r } ^ { ( s ) } \big | \Theta _ { r - 1 } ^ { ( s ) } , \Theta _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) .\tag{71}
$$

It remains to upper bound each per-step term. Using the SGLD update $\Theta _ { r } ^ { ( s ) } = \Theta _ { r - 1 } ^ { ( s ) } + \eta _ { s , r } G _ { s , r } + N _ { s , \ i }$ <sub>r</sub> with $N _ { s , r } \sim \mathcal { N } ( 0 , \tau _ { s , r } ^ { 2 } I )$ translation invariance of mutual information gives

$$
I \Big ( B _ { s , r } ; \Theta _ { r } ^ { ( s ) } \mid \Theta _ { r - 1 } ^ { ( s ) } , \Theta _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) = I \Big ( B _ { s , r } ; \eta _ { s , r } G _ { s , r } + \tau _ { s , r } \mid \Theta _ { r - 1 } ^ { ( s ) } , \Theta _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) .
$$

Write this as a differential entropy difference:

$$
I \Big ( B _ { s , r } ; \eta _ { s , r } G _ { s , r } + \tau _ { s , r } \big | \Theta _ { r - 1 } ^ { ( s ) } , \Theta _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) = h \Big ( \eta _ { s , r } G _ { s , r } + \tau _ { s , r } \big | \Theta _ { r - 1 } ^ { ( s ) } , \Theta _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) - h ( N _ { s , r } ) ,
$$

because $G _ { s , r }$ is a deterministic function of $( B _ { s , r } , \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } )$ and $N _ { s , \prime }$ is independent of $B _ { s , \prime }$ <sub>r</sub> and Gaussian. Let $Y _ { s , r } : =$ $\eta _ { s , r } G _ { s , r } + N _ { s , r }$ and denote $d : = d _ { l + 1 : L }$ . By the Lemma A.10 bound applied conditionally,

$$
h \Bigl ( Y _ { s , r } \big | \Theta _ { r - 1 } ^ { ( s ) } , \Theta _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Bigr ) \leq \frac { 1 } { 2 } \log \Bigl ( ( 2 \pi e ) ^ { d } \operatorname* { d e t } \Big ( \mathbb { E } \bigl [ Y _ { s , r } Y _ { s , r } ^ { \top } \big | \Theta _ { r - 1 } ^ { ( s ) } , \Theta _ { 0 } ^ { ( s ) } , W _ { 1 : l } \bigr ] \Bigr ) \Bigr ) .
$$

Compute the conditional second moment using independence of $N _ { s , \eta }$ <sub>r</sub> and $G _ { s , r }$ given $( \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } )$ and $\mathbb { E } [ N _ { s , r } ] = 0 \colon$

$$
\begin{array} { r l } & { \mathbb { E } \big [ Y _ { s , r } Y _ { s , r } ^ { \top } \big | \Theta _ { r - 1 } ^ { ( s ) } , \Theta _ { 0 } ^ { ( s ) } , W _ { 1 : l } \big ] = \eta _ { s , r } ^ { 2 } \mathbb { E } \big [ G _ { s , r } G _ { s , r } ^ { \top } \big | \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } \big ] + \mathbb { E } [ N _ { s , r } N _ { s , r } ^ { \top } ] } \\ & { \qquad = \eta _ { s , r } ^ { 2 } M _ { s , r } ^ { ( l ) } + \tau _ { s , r } ^ { 2 } I . } \end{array}
$$

Also $\begin{array} { r } { h ( N _ { s , r } ) = \frac { 1 } { 2 } \log \left( ( 2 \pi e ) ^ { d } \operatorname* { d e t } ( \tau _ { s , r } ^ { 2 } I ) \right) } \end{array}$ since $N _ { s , r } \sim \mathcal { N } ( 0 , \tau _ { s , r } ^ { 2 } I )$ . Subtracting these two entropies yields

$$
I \Big ( B _ { s , r } ; \Theta _ { r } ^ { ( s ) } \big | \Theta _ { r - 1 } ^ { ( s ) } , \Theta _ { 0 } ^ { ( s ) } , W _ { 1 : l } \Big ) \leq \frac { 1 } { 2 } \log \operatorname* { d e t } \Big ( I + \frac { \eta _ { s , r } ^ { 2 } } { \tau _ { s , r } ^ { 2 } } M _ { s , r } ^ { ( l ) } \Big )
$$

Taking expectation and summing over r in (71), we obtain

$$
\Delta _ { s } ^ { ( l ) } \leq \sum _ { r = 1 } ^ { R _ { s } } \mathbb { E } \Bigg [ \frac { 1 } { 2 } \log \operatorname* { d e t } \Big ( I + \frac { \eta _ { s , r } ^ { 2 } } { \tau _ { s , r } ^ { 2 } } M _ { s , r } ^ { ( l ) } \Big ) \Bigg ] .
$$

This holds for each $s ,$ and hence

$$
I \Bigl ( \mathbf { U } _ { t } ^ { ( l ) } ; \Theta _ { R _ { t } } ^ { ( t ) } \left| W _ { 1 : l } \right. \Bigr ) \leq \sum _ { s = 1 } ^ { t } \Delta _ { s } ^ { ( l ) } \leq \sum _ { s = 1 } ^ { t } \sum _ { r = 1 } ^ { R _ { s } } \mathbb { E } \left[ \frac { 1 } { 2 } \log \operatorname* { d e t } \Bigl ( I + \frac { \eta _ { s , r } ^ { 2 } } { \tau _ { s , r } ^ { 2 } } M _ { s , r } ^ { ( l ) } \Bigr ) \right] .\tag{72}
$$

Substitute (68) and (72) into (67) to obtain

$$
\left. \mathbf { g } \mathbf { e n } _ { W ^ { ( t ) } } \right. \leq ( t - 1 ) \sqrt { 2 \sigma ^ { 2 } { K _ { t } ^ { ( l ) } } } + \sqrt { \frac { 2 \sigma ^ { 2 } } { N _ { \mathrm { e f f } , t } } \Bigl ( \mathcal { C } _ { t } ^ { ( l ) } + \sum _ { s = 1 } ^ { t } \sum _ { r = 1 } ^ { R _ { s } } \mathbb { E } \Bigl [ \frac { 1 } { 2 } \log \operatorname* { d e t } \Bigl ( I + \frac { \eta _ { s , r } ^ { 2 } } { \tau _ { s , r } ^ { 2 } } M _ { s , r } ^ { ( l ) } \Bigr ) \Bigr ] } \Bigr ) .
$$

This completes the proof.

## H. Proof of Proposition V.5

Proposition V.5 (Restate). Let $A _ { s , r } ^ { ( l ) } : = I + \alpha _ { s , r } V _ { s , r } ^ { ( l ) } \succ 0$ with $\alpha _ { s , r } = \eta _ { s , r } ^ { 2 } / \tau _ { s , r } ^ { 2 }$ . Then

$$
\frac { 1 } { 2 } \log \operatorname* { d e t } \bigl ( I + \alpha _ { s , r } M _ { s , r } ^ { ( l ) } \bigr ) = \underbrace { \frac { 1 } { 2 } \log \operatorname* { d e t } ( A _ { s , r } ^ { ( l ) } ) } _ { \mathrm { i n s t a b i l i t y } } + \underbrace { \frac { 1 } { 2 } \log \Bigl ( 1 + \alpha _ { s , r } \mu _ { s , r } ^ { ( l ) \top } \bigl ( A _ { s , r } ^ { ( l ) } \bigr ) ^ { - 1 } \mu _ { s , r } ^ { ( l ) } \Bigr ) } _ { \mathrm { i n t e r a c t i o n ~ c o s t } } .
$$

Proof. We condition on $( \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } )$ throughout. By definition,

$$
M _ { s , r } ^ { ( l ) } = \mathbb { E } [ G _ { s , r } G _ { s , r } ^ { \top } ] \quad \mathrm { w i t h } \quad G _ { s , r } = \lambda _ { s , \mathrm { o l d } } G _ { s , r } ^ { \mathrm { o l d } } + \lambda _ { s , \mathrm { n e w } } G _ { s , r } ^ { \mathrm { n e w } } .
$$

Since the old and new minibatches are sampled independently given the state, $G _ { s , r } ^ { \mathrm { o l d } }$ and $G _ { s , r } ^ { \mathrm { n e w } }$ are conditionally independent, hence

$$
\mathrm { C o v } ( G _ { s , r } ) = \lambda _ { s , \mathrm { o l d } } ^ { 2 } \mathrm { C o v } ( G _ { s , r } ^ { \mathrm { o l d } } ) + \lambda _ { s , \mathrm { n e w } } ^ { 2 } \mathrm { C o v } ( G _ { s , r } ^ { \mathrm { n e w } } ) = V _ { s , r } ^ { ( l ) } .
$$

Also $\begin{array} { r } { \mathbb { E } [ G _ { s , r } ] = \lambda _ { s , \mathrm { o l d } } \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } + \lambda _ { s , \mathrm { n e w } } \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } = \mu _ { s , r } ^ { ( l ) } } \end{array}$ . Using the identity $\mathbb { E } [ X X ^ { \top } ] = \operatorname { C o v } ( X ) + \mathbb { E } [ X ] \mathbb { E } [ X ] ^ { \top }$ , we can obtain

$$
I + \alpha _ { s , r } M _ { s , r } ^ { ( l ) } = I + \alpha _ { s , r } V _ { s , r } ^ { ( l ) } + \alpha _ { s , r } \mu _ { s , r } ^ { ( l ) } \mu _ { s , r } ^ { ( l ) \top } = A _ { s , r } ^ { ( l ) } + u u ^ { \top } , \quad u : = \sqrt { \alpha _ { s , r } } \mu _ { s , r } ^ { ( l ) } .
$$

Since $A _ { s , r } ^ { ( l ) } \succ 0$ , the matrix determinant lemma det $( I + A + u u ^ { \top } ) = \operatorname* { d e t } ( I + A ) \left( 1 + u ^ { \top } ( I + A ) ^ { - 1 } u \right)$ gives

$$
\operatorname* { d e t } \bigl ( A _ { s , r } ^ { ( l ) } + u u ^ { \top } \bigr ) = \operatorname* { d e t } \bigl ( A _ { s , r } ^ { ( l ) } \bigr ) \Bigl ( 1 + u ^ { \top } \bigl ( A _ { s , r } ^ { ( l ) } \bigr ) ^ { - 1 } u \Bigr ) = \operatorname* { d e t } \bigl ( A _ { s , r } ^ { ( l ) } \bigr ) \Bigl ( 1 + \alpha _ { s , r } \mu _ { s , r } ^ { ( l ) \top } \bigl ( A _ { s , r } ^ { ( l ) } \bigr ) ^ { - 1 } \mu _ { s , r } ^ { ( l ) } \Bigr ) .
$$

Taking $\textstyle { \frac { 1 } { 2 } }$ log of both sides completes the proof.

## I. Proof of Corollary V.7

Corollary V.7 (Restate). Let $\mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } : = \mathbb { E } [ G _ { s , r } ^ { \mathrm { o l d } } \ | \ \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } ] , \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } : = \mathbb { E } [ G _ { s , r } ^ { \mathrm { n e w } } \ | \ \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } ]$ , so that $\mu _ { s , r } ^ { ( l ) } = \lambda _ { s , \mathrm { o l d } } \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } +$ $\lambda _ { s , \mathrm { n e w } } \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) }$ . Then under the sensitivity metric $H _ { s , r } ^ { ( l ) }$ in Definition V.6,

$$
\mu _ { s , r } ^ { ( l ) \top } H _ { s , r } ^ { ( l ) } \mu _ { s , r } ^ { ( l ) } = \lambda _ { s , \mathrm { o d d } } ^ { 2 } \Big \| \mu _ { s , r } ^ { \mathrm { o d d } } ( l ) \Big \| _ { H } ^ { 2 } + \lambda _ { s , \mathrm { n e c e } } ^ { 2 } \Big \| \mu _ { s , r } ^ { \mathrm { n e c e } , ( l ) } \Big \| _ { H } ^ { 2 } + 2 \lambda _ { s , \mathrm { o d d } } \lambda _ { s , \mathrm { n e c e } } \Big \| \mu _ { s , r } ^ { \mathrm { o d d } , ( l ) } \Big \| _ { H } \Big \| \mu _ { s , r } ^ { \mathrm { n e c e } , ( l ) } \Big \| _ { H } \mathrm { c o s } _ { H } \Big ( \mu _ { s , r } ^ { \mathrm { o d d } , ( l ) } , \mu _ { s , r } ^ { \mathrm { n e c e } , ( l ) } \Big ) .
$$

where the information-geometric alignment

$$
\begin{array} { r } { \mathrm { A l i g n } = \cos _ { H } \left( \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } , \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } \right) . } \end{array}
$$

Proof. Since $\mu _ { s , r } ^ { ( l ) } = \lambda _ { s , \mathrm { o l d } } \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } + \lambda _ { s , \mathrm { n e w } } \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) }$ , then

$$
\begin{array} { r l } & { \mu _ { s , r } ^ { ( l ) } \mathcal { \mathsf { T } } _ { \boldsymbol { s } , \mathsf { p } } ^ { ( l ) } ( \boldsymbol { \mathsf { a } } _ { s , r } ^ ) = \left\| \lambda _ { s , \mathrm { o l d } } \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } + \lambda _ { s , \mathrm { n e v } } \mu _ { s , r } ^ { \mathrm { a e v } , ( l ) } \right\| _ { H } ^ { 2 } } \\ & { \qquad = \lambda _ { s , \mathrm { o l d } } ^ { 2 } \| \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } \| _ { H } ^ { 2 } + \lambda _ { s , \mathrm { n e v } } ^ , \left\| \mu _ { s , r } ^ { \mathrm { a e v } , ( l ) } \right\| _ { H } ^ { 2 } + 2 \lambda _ { s , \mathrm { o l d } } \lambda _ { s , \mathrm { n e v } } \langle \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } , \mu _ { s , r } ^ { \mathrm { a e v } , ( l ) } \rangle _ { H } } \\ & { \qquad = \lambda _ { s , \mathrm { o l d } } ^ { 2 } \| \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } \| _ { H } ^ { 2 } + \lambda _ { s , \mathrm { n e v } } ^ , \left\| \mu _ { s , r } ^ { \mathrm { a e v } , ( l ) } \right\| _ { H } ^ { 2 } + 2 \lambda _ { s , \mathrm { o l d } } \lambda _ { s , \mathrm { n e v } } \left\| \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } \right\| _ { H } \| \mu _ { s , r } ^ { \mathrm { a e v } , ( l ) } \| _ { H } \mathrm { c o s } _ { H } \left( \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } , \mu _ { s , r } ^ { \mathrm { a e v } , ( l ) } \right) . } \end{array}
$$

This completes the proof.

## J. Proof of Corollary V.8

Corollary V.8 (Restate). Conditioning on $( \Theta _ { r - 1 } ^ { ( s ) } , W _ { 1 : l } )$ , the per-step trajectory budget admits the scalar upper bound

$$
\sum _ { s = 1 } ^ { t } \sum _ { r = 1 } ^ { R _ { s } } \mathbb { E } \bigg [ \frac { 1 } { 2 } \log \operatorname* { d e t } \Big ( I + \alpha _ { s , r } M _ { s , r } ^ { ( l ) } \Big ) \bigg ] \leq \frac { 1 } { 2 } \sum _ { s = 1 } ^ { t } \sum _ { r = 1 } ^ { R _ { s } } \alpha _ { s , r } \mathbb { E } \Big [ \mathrm { t r } \big ( V _ { s , r } ^ { ( l ) } \big ) + \| \mu _ { s , r } ^ { ( l ) } \| ^ { 2 } \Big ] .
$$

Moreover, the two scalar components admit the explicit decompositions $\mathrm { t r } \big ( V _ { s , r } ^ { ( l ) } \big ) = \lambda _ { s , \mathrm { o l d } } ^ { 2 } \mathrm { t r } \big ( \Sigma _ { s , r } ^ { \mathrm { o l d } , ( l ) } \big ) + \lambda _ { s , \mathrm { n e w } } ^ { 2 } \mathrm { t r } \big ( \Sigma _ { s , r } ^ { \mathrm { n e w } , ( l ) } \big )$ , and $\begin{array} { r } { \| \mu _ { s , r } ^ { ( l ) } \| ^ { 2 } = \lambda _ { s , \mathrm { o l d } } ^ { 2 } \| \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } \| ^ { 2 } + \lambda _ { s , \mathrm { n e w } } ^ { 2 } \| \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } \| ^ { 2 } + 2 \lambda _ { s , \mathrm { o l d } } \lambda _ { s , \mathrm { n e w } } \langle \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } , \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } \rangle . } \end{array}$

Proof. Let $B \succeq 0$ have eigenvalues $\{ \lambda _ { i } \} _ { i = 1 } ^ { d } \subset \mathbb { R } _ { + }$ . Then log de $\begin{array} { r } { ( I + B ) = \sum _ { i = 1 } ^ { d } \log ( 1 + \lambda _ { i } ) } \end{array}$ and $\begin{array} { r } { \operatorname { t r } ( B ) = \sum _ { i = 1 } ^ { d } \lambda _ { i } } \end{array}$

Since log $\lceil 1 + x \rceil \leq x$ for all $x \geq 0 .$ , summing over i yields log det $( I + B ) \leq \operatorname { t r } ( B )$

Apply to $B = \alpha _ { s , r } M _ { s , r } ^ { ( l ) } \succeq 0 \colon$

$$
\frac { 1 } { 2 } \log \mathrm { d e t } ( I + \alpha _ { s , r } M _ { s , r } ^ { ( l ) } ) \leq \frac { 1 } { 2 } \alpha _ { s , r } \mathrm { t r } ( M _ { s , r } ^ { ( l ) } ) .
$$

Taking expectation and summing over $( s , r )$ gives $\begin{array} { r l } { } & { { \frac { 1 } { 2 } } \sum _ { s = 1 } ^ { t } \sum _ { r = 1 } ^ { R _ { s } } \alpha _ { s , r } \mathbb { E } \big [ \mathrm { t r } ( M _ { s , r } ^ { ( l ) } ) \big ] } \end{array}$

By the identity $\mathbb { E } [ X X ^ { \top } ] = \operatorname { C o v } ( X ) + \mathbb { E } [ X ] \mathbb { E } [ X ] ^ { \top }$ , we have $M _ { s , r } ^ { ( l ) ^ { \smash { \mathstrut } } } = V _ { s , r } ^ { ( l ) ^ { \smash { \mathstrut } } } + \mu _ { s , r } ^ { ( l ) } \mu _ { s , r } ^ { ( l ) ^ { \smash { \mathstrut } } }$ , hence $\mathrm { t r } ( { \cal M } _ { s , r } ^ { ( l ) } ) = \mathrm { t r } ( V _ { s , r } ^ { ( l ) } ) +$ $\mathrm { t r } ( \mu _ { s , r } ^ { ( l ) } \mu _ { s , r } ^ { ( l ) \top } ) = \mathrm { t r } ( \bar { V _ { s , r } ^ { ( l ) } } ) + \bar { \| \mu _ { s , r } ^ { ( l ) } \| ^ { 2 } }$ , which yields $\begin{array} { r l } { } & { { } \frac { 1 } { 2 } \sum _ { s = 1 } ^ { t } \sum _ { r = 1 } ^ { R _ { s } } \alpha _ { s , r } \mathbb { E } \Bigl [ \mathrm { t r } \bigl ( V _ { s , r } ^ { ( l ) } \bigr ) + \bigl \| \mu _ { s , r } ^ { ( l ) } \bigr \| ^ { 2 } \Bigr ] } \end{array}$

Finally, applying linearity of the trace to the definition of $V _ { s , r } ^ { ( l ) }$ , we obtain

$$
\mathrm { t r } \big ( V _ { s , r } ^ { ( l ) } \big ) = \lambda _ { s , \mathrm { o l d } } ^ { 2 } \mathrm { t r } \big ( \Sigma _ { s , r } ^ { \mathrm { o l d } , ( l ) } \big ) + \lambda _ { s , \mathrm { n e w } } ^ { 2 } \mathrm { t r } \big ( \Sigma _ { s , r } ^ { \mathrm { n e w } , ( l ) } \big ) .
$$

The second identity follows by expanding the squared Euclidean norm

$$
\begin{array} { r } { \| \mu _ { s , r } ^ { ( l ) } \| ^ { 2 } = \lambda _ { s , \mathrm { o l d } } ^ { 2 } \| \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } \| ^ { 2 } + \lambda _ { s , \mathrm { n e w } } ^ { 2 } \| \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } \| ^ { 2 } + 2 \lambda _ { s , \mathrm { o l d } } \lambda _ { s , \mathrm { n e w } } \langle \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } , \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } \rangle . } \end{array}
$$

## K. Proof of Corollary V.9

Corollary V.9 (Restate). Fix any sensitivity metric $H _ { s , r } ^ { ( l ) }$ . For $\lambda \in [ 0 , 1 ]$ , consider the mixed vector $\mu _ { s , r } ^ { ( l ) } ( \lambda ) : = \lambda \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } + ( 1 -$ $\lambda ) \mu _ { s , r } ^ { \mathrm { n e w } , ( \tilde { l } ) }$ and its metric energy $\| \mu _ { s , r } ^ { ( l ) } ( \lambda ) \| _ { H } ^ { 2 }$ . Then $\| \mu _ { s , r } ^ { ( l ) } ( \lambda ) \| _ { H } ^ { 2 }$ is minimized over $\lambda \in [ 0 , 1 ]$ by

$$
\lambda ^ { \star } = \mathrm { c l i p } _ { [ 0 , 1 ] } \left( \frac { \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } { } ^ { \top } H ( \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } - \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } ) } { ( \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } - \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } ) ^ { \top } H ( \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } - \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } ) } \right) ,
$$

with the convention ${ { \lambda } ^ { \star } } = 0$ if $\mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } = \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) }$

Proof. Let $d : = \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } - \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) }$ . Then $\mu ( \lambda ) = \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } + \lambda d$ and

$$
\| \mu ( \lambda ) \| _ { H } ^ { 2 } = ( \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } + \lambda d ) ^ { \top } H ( \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } + \lambda d ) = \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } { } ^ { \top } H \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } + 2 \lambda \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } { } ^ { \top } H d + \lambda ^ { 2 } d ^ { \top } H d .
$$

If $\mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } = \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) }$ , then $d = 0$ and $\Vert \mu ( \lambda ) \Vert _ { H } ^ { 2 } \equiv { \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } } ^ { \top } H \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) }$ is constant; we define ${ \bf \Phi } \lambda ^ { \star } = 0 . \mathrm { I f } \ \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } \neq \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) }$ , then $d \neq 0$ and since $H \succ 0$ , we have $d ^ { \top } H d > 0$ . Therefore the scalar function $\varphi ( \lambda ) : = \| \mu ( \lambda ) \| _ { H } ^ { 2 }$ is a strictly convex quadratic. Differentiate:

$$
\varphi ^ { \prime } ( \lambda ) = 2 \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } { } ^ { \top } H d + 2 \lambda d ^ { \top } H d .
$$

Setting $\varphi ^ { \prime } ( \lambda ) = 0$ yields the unique unconstrained minimizer

$$
\lambda _ { \mathrm { u n c } } ^ { \star } = - \frac { \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } ^ { \top } H d } { d ^ { \top } H d } = \frac { \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } ^ { \top } H \big ( \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } - \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } \big ) } { \big ( \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } - \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } \big ) ^ { \top } H \big ( \mu _ { s , r } ^ { \mathrm { o l d } , ( l ) } - \mu _ { s , r } ^ { \mathrm { n e w } , ( l ) } \big ) } .
$$

The constrained minimizer over $\lambda \in [ 0 , 1 ]$ is the Euclidean projection of ${ \lambda _ { \mathrm { u n c } } ^ { \star } }$ onto $[ 0 , 1 ] , \mathrm { { i . e . , } } \lambda ^ { \star } = \mathrm { { c l i p } } _ { [ 0 , 1 ] } ( \lambda _ { \mathrm { { u n c } } } ^ { \star } )$ . The proof is complete. □

## APPENDIX C

DETAILS OF THE SYNTHETIC GAUSSIAN EXPERIMENTS

We construct a sequence of $T$ binary classification tasks $\{ \mathcal { T } _ { t } \} _ { t = 0 } ^ { T - 1 }$ in $\mathbb { R } ^ { d }$ . Each task is a balanced two-class Gaussian mixture:

$$
\begin{array} { c } { { x \mid ( y = 0 , T _ { t } ) \sim \mathcal { N } ( \mu _ { t } - s w _ { t } , I _ { d } ) , } } \\ { { x \mid ( y = 1 , T _ { t } ) \sim \mathcal { N } ( \mu _ { t } + s w _ { t } , I _ { d } ) , } } \end{array}
$$

where $w _ { t }$ is a unit direction controlling the class-discriminative axis, $\mu _ { t }$ is a task-specific mean shift, and $s > 0$ is the class separation. To control task relatedness, we restrict $\{ w _ { t } \}$ to a random 2D subspace and vary its evolution across tasks. We consider two modes: (i) positive synergy, where $w _ { t }$ rotates smoothly by a fixed angle increment $\Delta$ so adjacent tasks remain aligned; (ii) negative synergy, where $w _ { t }$ alternates sign $( w _ { t + 1 } = - w _ { t } )$ , which is equivalent to a label-flip across tasks and induces strong interference. We control cross-task drift via $\mu _ { t } = C \cdot \xi _ { t }$ with $\xi _ { t } \sim \mathcal { N } ( 0 , I _ { d } )$ . For each task $\mathcal { T } _ { t } ,$ we sample an i.i.d. training set $D _ { t }$ of size n and an i.i.d. test set $D _ { t } ^ { \mathrm { t e s t } }$ of size $n _ { \mathrm { t e s t } }$

After completing task t, we store exactly m examples from its training set into a per-task buffer $B _ { t }$ . The total replay memory thus scales as $O ( T m )$ . When training on task $t \geq 1$ , each update step samples a mini-batch $B ^ { \mathrm { n e w } }$ from the current task and a replay mini-batch $\dot { B } ^ { \mathrm { o l d } }$ from replay buffer $\cup _ { i < t } B _ { i }$ . We minimize the mixture loss

$$
\ell _ { t } ( \theta ) = \lambda _ { \mathrm { n e w } } ( t ) { \widehat { \mathcal { L } } } ( B ^ { \mathrm { n e w } } ; \theta ) + \lambda _ { \mathrm { o l d } } ( t ) { \widehat { \mathcal { L } } } ( B ^ { \mathrm { o l d } } ; \theta ) , \qquad \lambda _ { \mathrm { n e w } } ( t ) = { \frac { 1 } { t + 1 } } , \ \lambda _ { \mathrm { o l d } } ( t ) = { \frac { t } { t + 1 } } .
$$

We use SGD with fixed learning rate and batch size. After finishing all $T$ tasks, we report the test risk

$$
L ( \theta ) = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \widehat { \mathcal { L } } ( D _ { t } ^ { \mathrm { t e s t } } ; \theta ) ,
$$

and the empirical risk (old tasks via replay buffers, the last task via its training set)

$$
\widehat { L } ( \theta ) = \frac { T - 1 } { T } \widehat { \mathcal { L } } \Big ( \bigcup _ { t = 0 } ^ { T - 2 } \mathcal { B } _ { t } ; \theta \Big ) + \frac { 1 } { T } \widehat { \mathcal { L } } ( D _ { T - 1 } ; \theta ) .
$$

The generalization gap is ${ \mathrm { g a p } } = L ( \theta ) - { \widehat { L } } ( \theta )$ . We report this task-averaged gap; it equals the total-risk gap of Theorem IV.1 divided by the task count T. Because T is fixed in this experiment, the constant $1 / T$ rescales the gap but leaves the predicted $m ^ { - 1 / 2 }$ scaling and the location of the finite-memory plateau unchanged. Unless stated otherwise, all curves are averaged over K independent runs; we report 95% confidence intervals for the mean (Student-t).