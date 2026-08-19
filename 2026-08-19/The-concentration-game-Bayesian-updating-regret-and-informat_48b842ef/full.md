# The concentration game: Bayesian updating, regret, and information

Akshay Balsubramani

akshay@vac.bio

## Abstract

We give a two-player zero-sum repeated game between a learner and nature whose value identity generates Bayesian updating and an exact accounting of exponential-weights regret at once, and supplies the comparator-class variational form that a wide class of concentration phenomena share. The terminal payof is the most a comparator can gain at fixed relative entropy from the prior, and the one-step constraint is an information budget on nature’s move under the learner’s mixed action. With the learner’s move otherwise unrestricted, Gibbs/Bayes weights emerge as its unique Bellman equalizer — the mixed action that makes the per-round loss independent of which direction nature moves — with log-partition functions playing the role of value functions. The regret decomposes exactly into three parts: a per-round information loss reflecting the variation in observed outcomes, an additive retempering drift that accounts exactly for any change of measurement scale between rounds, and the information the comparator carries relative to the prior. The variance and bounded-range proxies that drive standard regret bounds are looser relaxations of this decomposition, which holds generally and governs them all. Both players’ strategies are read of from the decomposition term by term, and repeated play yields an information-theoretic ledger of self-play in place of the usual quadratic-variation surrogate. The same comparator-class geometry accounts for the classical large-deviation bounds, and methods across bandits, posterior sampling, aggregation, and boosting are specializations of the one regret decomposition

## 1 Introduction

We develop the idea that Bayesian updating and exponential-weights regret are two faces of one identity: the value of a two-player zero-sum game played over several (T) rounds. Here we call it the concentration game, because a wide class of concentration phenomena against comparators draws its form from that same identity. This unifies several research communities’ separate formalisms for a cluster of interrelated information accounting problems. Online learning derives exponential-weights regret from a learning-rate tradeof; Bayesian inference derives the posterior from a likelihood-ratio update; concentration of measure and large-deviation theory derive tail bounds from a moment-generating-function inequality; repeated game theory derives equilibrium convergence from no-regret dynamics; and boosting derives margin guarantees from a weighted-error potential.

The concentration game’s value identity accounts for all of these. Bayes/exponential-weights updating is its equilibrium strategy, the moment-generating function is its per-round loss, and the log-partition function is its value. Repeated-game equilibria, Thompson sampling, log-pooling, and AdaBoost are specializations of it, and concentration of measure and Sanov large deviations draw their comparator-class variational form from it. The game is sequential by nature, because the phenomena it unifies are progressive: each round, belief is updated and the information budget that drives concentration is used up. Game 1.1 below states the basic framework. Retempering, budget allocation, and partial information are then layered onto the same rounds in Sections 4–7.

## 1.1 The game and the central identity

Fix a strictly positive prior π on a finite set of actions $[ K ] : = \{ 1 , \dots , K \}$ , a comparator complexity budget $\Gamma \geq 0 ,$ and a schedule $( \eta _ { t } , q _ { t } ) _ { t = 1 } ^ { T }$ of measurement scales and per-round information budgets, fixed in advance or chosen adaptively from history. A comparator is a fixed mixture over the actions against which the learner’s cumulative loss is compared, and Γ caps its relative entropy from the prior. The state entering round t is the cumulative centered score $\begin{array} { r } { S _ { t - 1 } : = \sum _ { s < t } z _ { s } } \end{array}$ of nature’s past moves $( S _ { 0 } = 0 ;$ the moves $z _ { s }$ are set out in the game box below). This has zero mean under the mixture p<sub>t</sub> the learner plays that round, $\langle p _ { t } , z _ { t } \rangle = 0 { : }$ nature’s move redistributes among the actions under the learner’s own weighting, without shifting them all together, so a loss shared equally by every action is invisible to the game (Section 2). Each round the learner sets a mixture over the actions, nature then moves the running state $S _ { t }$ under a single one-step budget, and one potential — the log-partition function — controls the whole run (Section 3). Nature’s one-round budget is measured by the rescaled centered cumulant generating function (RCGF) $\begin{array} { r } { Q _ { \eta } ( p , z ) : = \eta ^ { - 2 } \log \sum _ { i } p ( i ) e ^ { \eta z ( i ) } } \end{array}$ of its centered move z under the learner’s mixture p; the unrescaled $\begin{array} { r } { \eta ^ { 2 } Q _ { \eta } = \log \mathbb { E } _ { p } [ e ^ { \eta z } ] } \end{array}$ is the centered CGF, and the $\eta ^ { - 2 }$ factor makes the small-scale limit half the variance (Section 2). The RCGF measures how large nature’s move is at the scale η, and the scale sets which feature of the move that size responds to: the variance of z at small $\eta ,$ its worst tail at large η.

```latex
Game 1.1 (The concentration game). Fix a strictly positive prior $\pi \in \Delta ( [ K ] )$ , a comparator complexity budget
$\Gamma \geq 0 ,$ and a schedule $( \eta _ { t } , q _ { t } ) _ { t = 1 } ^ { \check { T } }$ of measurement scales and per-round RCGF budgets (predictable: $( \eta _ { t } , q _ { t } )$ may
depend on $z _ { 1 } , \ldots , z _ { t - 1 } )$ . Start from $S _ { 0 } = 0 .$ For each round $t = 1 , \ldots , T$
1. Learner picks a mixture $p _ { t } \in \Delta ( [ K ] )$ over the actions;
2. Nature, observing $p _ { t }$ , picks a centered $z _ { t } \in \mathbb { R } ^ { K }$ with $\langle p _ { t } , z _ { t } \rangle = 0$ and $Q _ { \eta _ { t } } ( p _ { t } , z _ { t } ) \leq q _ { t } ;$
3. State advances by $S _ { t } : = S _ { t - 1 } + z _ { t } .$
The terminal payof to nature is $L _ { \Gamma } ( S _ { T } )$ , the support function of a relative-entropy ball around the prior (Section 2.2):
$L _ { \Gamma } ( S ) : = \operatorname* { s u p } \{ \langle \nu , S \rangle : \nu \in \Delta ( [ K ] ) , \operatorname { K L } ( \nu \| \pi ) \leq \Gamma \}$
```

$$
p _ { t }
$$

$$
p _ { t } = G _ { \eta _ { t } } ( S _ { t - 1 } )
$$

$$
\begin{array} { r } { p _ { t } \overline { { ( i ) } } \propto \pi ( i ) e ^ { \eta _ { t } S _ { t - 1 } ( i ) } } \end{array}
$$

$$
\begin{array} { r } { \bar { G _ { \eta } } ( \dot { S } ) \colon \dot { i } \mapsto \pi ( i ) e ^ { \eta S ( i ) } / \sum _ { i } \bar { \pi ( j ) } e ^ { \eta S ( j ) } } \end{array}
$$

$$
\begin{array} { r } { \eta { \bf \bar { \Phi } } > 0 , } \end{array}
$$

$$
S ( i ) : = \eta ^ { - 1 } \log \bigl ( p ( i ) / \pi ( i ) \bigr )
$$

$$
\pi ( i ) e ^ { \eta S ( i ) } = p ( i )
$$

$$
G _ { \eta } ( S ) = p .
$$

The cumulative regret against any comparator $\begin{array} { r } { \nu \in \Delta ( [ K ] ) \mathrm { i s } R _ { T } ^ { c } ( \nu ) : = \sum _ { t = 1 } ^ { T } \langle p _ { t } - \nu , c _ { t } \rangle } \end{array}$ , the learner’s cumulative loss minus the comparator’s, where $c _ { t }$ is the round-t loss in the standard sign convention $( z _ { t } = \langle p _ { t } , c _ { t } \rangle \mathbf { 1 } - c _ { t }$ is the centered version, so that $R _ { T } ^ { c } ( \nu ) = \langle \nu , S _ { T } \rangle )$ .

The main identity (Theorem 4.3, Section 4) is that this cumulative regret decomposes exactly into three structurally distinct terms:

$$
R _ { T } ^ { c } ( \nu ) = \sum _ { t = 1 } ^ { T } \eta _ { t } Q _ { t } ( c ) + D _ { T } + B _ { T } ( \nu )\tag{1.1}
$$

where the leading sum is the intrinsic-time loss [7], built from the round-t centered RCGF $Q _ { t } ( c ) \geq 0$ (the intrinsictime density) weighted by the scale $\eta _ { t }$ . Intrinsic time measures the run’s length in information. Rounds elapsed do not enter. Write $\begin{array} { r } { W _ { \eta } ( \bar { S } ) : = \dot { \eta } ^ { - 1 } \log \sum _ { i } \pi ( i ) e ^ { \eta S ( i ) } } \end{array}$ for the scale-η log-partition of the state, the potential of Section 3. The retempering drift $\begin{array} { r } { D _ { T } = \sum _ { t = 1 } ^ { T - 1 } \bigl ( W _ { \eta _ { t + 1 } } ( S _ { t } ) - W _ { \eta _ { t } } ( S _ { t } ) \bigr ) } \end{array}$ is the exact loss from changing the measurement scale—retuning the inverse temperature—between rounds, each summand the change in $W _ { \eta }$ at a fixed state, so $D _ { T } = 0$ at constant scale. The terminal relative-entropy transport $B _ { T } ( \nu ) = \eta _ { T } ^ { - 1 } \bigl ( \mathrm { K L } ( \nu \| \bar { \pi ) } - \mathrm { K L } \dot { ( } \nu \| \rho _ { T , \eta _ { T } } ) \bigr )$ , with $\rho _ { T , \eta _ { T } } : = G _ { \eta _ { T } } ( S _ { T } )$ the terminal Gibbs posterior, is the comparator information carried from the prior π to that posterior. The diference of the two relative entropies collapses to a single expectation, $\eta _ { T } B _ { T } ( \nu ) = \mathbb { E } _ { \nu } \left[ \log ( \rho _ { T , \eta _ { T } } / \pi ) \right]$ , the log-likelihood ratio of posterior to prior averaged under the comparator: how far the run has moved the belief in the comparator’s own direction (Section 5.5). Taken together the three terms are the run’s ledger: a closed account of the regret, one entry per mechanism, which sums to it exactly and leaves no remainder. The word is used in that sense throughout, and each later setting has its own ledger, the same three entries evaluated on whatever score sequence that setting scores. Because (1.1) is an identity rather than an inequality, both the learner’s and nature’s strategies are read of term by term: the learner plays the Bellman equalizer (Gibbs at every round), and nature saturates the round budget $Q _ { t } ( c ) = q _ { t }$ in the worst comparator direction. Figure 2 follows one realized run, with the three terms accumulating along it.

The same basic structure then survives changing the suficient statistic (the score sequence the learner actually applies Bayes to, which need not be the raw loss), the comparator class, the observation structure, the geometry, or the action rule (Appendix B). Each modification changes surface features while the Bellman backbone stays fixed

The tilt is the exponential-weights / Bayes update (multiplicative weights, Hedge), and equally the entropic Followthe-Regularized-Leader (FTRL) iterate arg max $\dot { \iota } _ { p } \{ \langle p , S _ { t - 1 } \rangle - \eta _ { t } ^ { - 1 } \mathrm { K L } ( p \| \pi ) \}$ on the cumulative centered scores, the two coinciding by the Gibbs variational identity (2.6). Section 3 derives it as the game’s Bellman equalizer: the play that makes the round’s loss a deterministic function of nature’s budget alone, whichever feasible move nature selects. Equalizing is weaker than round-value minimaxity, since the minimax play minimizes the round’s worst-case value where the equalizer flattens the loss across nature’s replies, and the two need not agree (Section 3.4).

Of the four ingredients, only the prior π is a modeling decision, encoding the prior on action quality. The measurement scale $\eta _ { t }$ gives “the size of a move” its meaning, dual to π through the exponential-family identification $p \propto \pi e ^ { \eta S }$ ; it is an inverse temperature, large η cold and concentrating the tilt, small η hot and difuse. The constraint $Q _ { \eta _ { t } } ( p _ { t } , z _ { t } ) \leq q _ { t }$ is an information budget on nature’s one-round move: by the identity $\eta ^ { 2 } Q _ { \eta } ( p , z ) = \mathrm { K L } ( p \| p ^ { + } )$ (with $p ^ { + } ( i ) \propto p ( i ) e ^ { \eta z ( i ) }$ the within-round update), $q _ { t }$ bounds the prior-to-posterior relative-entropy transfer at scale $\eta _ { t }$ . And Γ is a matching budget on the comparator class: $L _ { \Gamma } ( S _ { T } )$ is the largest terminal gain any distribution within relative entropy Γ of the prior can extract. An uncapped nature would play unstructured noise; the cap is the only constraint the game imposes (Section 2 reads it as a coincidence budget).

![](images/c731f431ca0bb1e4d1606c4952be063f1feecba6fcb4ed6fb1276add25e78de3.jpg)  
Figure 1: The concentration game and its specializations. A two-player zero-sum game (center) has Bayes / exponentialweights updating as its Bellman equilibrium and the log-partition function as its value; its regret decomposes exactly as $\begin{array} { r } { R _ { T } ^ { c } ( \nu ) = \sum _ { t } \eta _ { t } Q _ { t } ( c ) + D _ { T } + B _ { T } ( \nu ) } \end{array}$ — an intrinsic-time loss, a retempering drift, and terminal relative-entropy transport (Theorem 4.3, terms as in the text). Each surrounding family’s regret decomposition is recovered by specializing the prior, the measurement scale, the per-round budget, the comparator class, and the feedback structure—one paragraph of bookkeeping around the same equilibrium; the sharp rate in each family still needs a family-specific bound on the estimated per-round RCGF.

The schedule and the horizon are both data of the game, given to both players and optimized by neither, until Section 4 promotes the scale to a choice the learner makes (Game 4.1). Fixing T changes nothing in the accounting: the decomposition of Theorem 4.3 holds with equality at every prefix, so stopping at any $t \leq T$ leaves the same three entries, evaluated at horizon t, and Section 5.5 quantifies the anytime-versus-fixed-horizon diference.

## 1.2 Main results

The main result is that the decomposition (1.1) holds with equality at every horizon, for every comparator, and for every predictable schedule: the Bellman equalizer composes across rounds, and each of the three terms is a sum of per-round contributions (Theorem 4.3). The drift is the closed-form loss from changing measurement scale between rounds, its per-step form a prior-to-posterior relative entropy (Proposition C.1); at a fixed scale it vanishes and the decomposition reduces to the entropic mix-loss account carried by the log-partition potential, classical in its inequality-relaxed form.

The variance and bounded-range quantities that drive standard regret bounds are derived relaxations of that per-round primitive. The RCGF tends to ${ \scriptstyle { \frac { 1 } { 2 } } } \mathrm { V a r } _ { p } ( z )$ as η ↓ 0 (Corollary 2.5) and at fixed scale is bounded above by a variance proxy and by a range proxy, neither refining the other (Section 3.3). Substituting either for the centered RCGF yields a looser regret upper bound, and quadratic-variation bounds appear as its small-η limit.

Under a single cumulative budget V on nature’s total information, in place of the per-round caps, the horizon drops out of the value. At constant scale it equals $\Gamma / \eta + \eta V$ from a finite horizon on, so the square-root envelope $2 \sqrt { \Gamma V }$ is attained and the shortfall the one-round game exhibits belongs to that single round (Proposition 4.2). Fixing the horizon therefore concedes nothing to nature. A known horizon instead settles the scale, and it releases the slack an anytime guarantee holds in reserve for the times it is not consulted — the drawdown of the comparator’s transport trajectory, how far it has ever fallen below its current level (Section 5.5). Those two savings separate an anytime-valid confidence sequence from a fixed-horizon interval read once.

The same identity gives the regret decomposition of a wide family of methods (Figure 1): Gibbs conditioning,

![](images/8fb83e4c8b68fc3846e68ed25f291c64c3ffca1dfd3e2bd549a67616604ab898.jpg)

b  
![](images/f9c7166ed179144242a7ccbcc5bd4bccf1421309f7100b17b056328c05d60ab3.jpg)  
Figure 2: One trajectory of the concentration game, and its information decomposition. A representative realized path, with a continuum of experts on the line $x \in \mathbb { R }$ under a Gaussian reference π and a schedule $( \eta _ { t } , q _ { t } )$ that grants nature more of its per-round budget in the early rounds. (a) Each round the learner plays the Bellman-equalizer Gibbs tilt $p _ { t } = G _ { \eta _ { t } } ( S _ { t - 1 } ) \propto \pi e ^ { \eta _ { t } S _ { t - 1 } }$ (faint densities), whose mean $m _ { t } = \langle p _ { t } , x \rangle$ is the realized walk (blue); nature answers with a centered increment $z _ { t } .$ . The whiskers mark how far nature may move the state in one round under the RCGF budget $q _ { t } .$ This reach scales as $\sqrt { { q } _ { t } }$ (the centered RCGF is second order in the step, so in the small-step regime the budget-saturating move obeys $\delta \approx { \sqrt { 2 q } } ;$ exactly, $\delta = \eta ^ { - 1 } \mathrm { a r c o s h } ( e ^ { \eta ^ { 2 } q } )$ for two symmetric experts, Proposition A.6). As the budget is used up—most of it early—the reach shrinks and the state concentrates. $( b )$ The comparator regret decomposes at every horizon, $R _ { t } ^ { c } ( \nu ) = P _ { t } + D _ { t } + B _ { t } ( \nu )$ with loss $\begin{array} { r } { P _ { t } = \sum _ { s < t } \eta _ { s } Q _ { s } } \end{array}$ , starting from the origin $( R _ { 0 } ^ { c } = 0 ;$ before any round every term is zero). The solid curve is the total. The dotted waterfall at horizons $t _ { 1 }$ and $t _ { 2 }$ splits it into the intrinsic-time loss $P _ { t }$ (up), the retempering drift $D _ { t } \leq 0 -$ —the loss from changing scale between rounds, a gain to the learner here (down)—and the terminal relative-entropy transport $B _ { t } ( \nu ) = \left[ \mathrm { K L } ( \nu \| \pi ) - \mathrm { K L } ( \nu \| \rho _ { t , \eta _ { t } } ) \right] / \eta _ { t }$ <sub>t</sub> (up), with $\rho _ { t , \eta _ { t } }$ the horizon-t Gibbs posterior. The comparator is $\nu = \rho _ { t _ { 2 } , \eta _ { t _ { 2 } } }$ , the posterior the game settles on.

Sanov’s theorem, low-noise luckiness, confidence sets, repeated-game equilibria, bandits, feedback graphs, contextual bandits, Thompson sampling, and AdaBoost (Sections 5–9), and logarithmic pooling (Appendix A.12). Each is obtained by specializing the prior, the measurement scale, the per-round budget, the comparator class, and the feedback structure — one paragraph of bookkeeping at the level of the decomposition. The contribution is the accounting; the rates themselves are not new, and the sharp rate in each family still needs a family-specific bound on the per-round RCGF. Converting a decomposition into a sharp rate, such as the minimax bandit $O ( \sqrt { K T } )$ dependence, is settled by the estimated per-round RCGF, itself a closed form in the true losses and observation geometry (Proposition A.15). That closed form recovers the classical variance- and range-based envelopes as small-scale relaxations the worst case never improves on. The self-play ledger of Section 6 instead reads the per-round loss of the realized path, giving equilibrium rates faster than the $1 / \dot { \sqrt { T } }$ worst case whenever intrinsic time is sublinear

The specific contribution is that these separate literatures share one primitive — the terminal relative-entropy-ball payof under a one-scale RCGF constraint — from which each draws its comparator-class variational form. Earlier work gets variance surrogates because it starts downstream; starting at the RCGF recovers those surrogates as relaxations and makes the identities visible.

## 2 Coordinates, the terminal payof, and the per-round primitive

The game runs on two primitives — the terminal payof $L _ { \Gamma }$ and the per-round RCGF — both expressed in one set of centered coordinates. We work on a finite set $[ K ] = \{ 1 , \ldots , K \}$ with a strictly positive prior $\pi \in \Delta ( [ K ] )$ ). The learner’s move each round is an unrestricted mixed action $p _ { t } \in \Delta ( [ K ] )$ —the game constrains nature alone—recorded in exponential-family coordinates as the Bayes/exponential-weights update of a composite score sequence $c _ { t } \in \mathbb { R } ^ { K }$ :

$$
p _ { t } ( i ) = \frac { \pi ( i ) e ^ { - \eta _ { t } C _ { t - 1 } ( i ) } } { \sum _ { j = 1 } ^ { K } \pi ( j ) e ^ { - \eta _ { t } C _ { t - 1 } ( j ) } } , \qquad C _ { t } ( i ) : = \sum _ { s = 1 } ^ { t } c _ { s } ( i ) , \qquad C _ { 0 } \equiv 0\tag{2.1}
$$

The correspondence is with the cumulative score: distinct interior plays come from distinct score classes, while $c _ { t } =$ $C _ { t } - C _ { t - 1 }$ is pinned, again up to an additive constant, only once the next play and the next scale are given as well. Only the product $\eta _ { t } C _ { t - 1 }$ enters $p _ { t }$ , so $\eta _ { t }$ is a choice of scale that couples the move to nature’s information budget. This same form is the game’s Bellman equalizer (Section 3), the play the analysis singles out. The scores may be literal losses, predictable ofsets, importance-weighted estimates, optimistic residuals, signed margins, or any statistic to which the learner applies Bayes—the game is defined relative to the chosen suficient statistic, with no ambient “true” loss behind it.

This costs no generality: (2.1) is a bijection from $\mathbb { R } ^ { K } / \mathbb { R } \mathbf { 1 }$ onto the interior of $\Delta ( [ K ] )$ , so it reparameterizes arbitrary play instead of restricting it, and its inverse reads the state as the round’s posterior-to-prior log-ratio at the round’s own scale (Appendix A.1.1).

## 2.1 Centered excess-score coordinates

For round $t ,$ define the learner mean $\mu _ { t } : = \langle p _ { t } , c _ { t } \rangle$ , the centered move $z _ { t } ( i ) : = \mu _ { t } - c _ { t } ( i )$ , and the intrinsic state $\textstyle S _ { t } : = \sum _ { s = 1 } ^ { t } z _ { s }$ with $S _ { 0 } : = 0$ . Each move is centered under the learner’s play: $\langle p _ { t } , z _ { t } \rangle = 0$

Proposition 2.1 (Centered-coordinate form of Bayes updating). For every $t \geq 1$ and $i \in [ K ]$

$$
S _ { t - 1 } ( i ) = \sum _ { s = 1 } ^ { t - 1 } \mu _ { s } - C _ { t - 1 } ( i ) , \qquad p _ { t } ( i ) = \frac { \pi ( i ) e ^ { \eta _ { t } S _ { t - 1 } ( i ) } } { \sum _ { j = 1 } ^ { K } \pi ( j ) e ^ { \eta _ { t } S _ { t - 1 } ( j ) } }\tag{2.2}
$$

Moreover, $f o r$ every comparator $\begin{array} { r } { \nu \in \Delta ( [ K ] ) , R _ { T } ^ { c } ( \nu ) : = \sum _ { t = 1 } ^ { T } \bigl ( \langle p _ { t } , c _ { t } \rangle - \langle \nu , c _ { t } \rangle \bigr ) = \langle \nu , S _ { T } \rangle = \sum _ { t = 1 } ^ { T } \langle \nu , z _ { t } \rangle . } \end{array}$

Only relative performance matters. Replacing $c _ { t }$ by $c _ { t } + b _ { t } \mathbf { 1 }$ leaves every $z _ { t }$ unchanged: centered coordinates quotient out the global-bias direction from the start, and the game lives on the orthogonal complement of 1.

## 2.2 The terminal concentration functional

The game ends with nature gaining whatever the best admissible comparator can extract from the final state. The one remaining modeling choice is which comparators count as admissible. We focus on a relative-entropy ball: a comparator may reweight the actions within an information budget Γ of the prior.

For $\Gamma \geq 0$ and a state $S \in \mathbb { R } ^ { K }$ , define

$$
L _ { \Gamma } ( S ) : = \operatorname* { s u p } \left\{ \langle \nu , S \rangle : \nu \in \Delta ( [ K ] ) , \operatorname { K L } ( \nu \| \pi ) \leq \Gamma \right\}\tag{2.3}
$$

This number can be read in multiple ways.

1. Benchmark: $\langle \nu , S _ { T } \rangle$ is the cumulative regret against the comparator ν (Proposition 2.1), so $L _ { \Gamma } ( S _ { T } )$ is the worst regret over the admissible class.

2. Support function: it is how far the relative-entropy ball $\{ \nu : \mathrm { K L } ( \nu \| \pi ) \leq \Gamma \}$ extends in the direction S.

3. Statistical displacement: it is the largest mean shift attained by a change of measure costing Γ in relative entropy, the quantity Chernof and Sanov bounds compute (Section 5).

At $\Gamma = 0$ no reweighting is permitted and $L _ { 0 } ( S ) = \langle \pi , S \rangle$ , the prior mean; once Γ is large enough for the ball to fill the simplex, $L _ { \Gamma } ( S ) = \operatorname* { m a x } _ { i } S ( i )$ , the best single action (Proposition $_ { \mathrm { A . 4 ) } }$ . In between, Γ interpolates between the two.

Fix $\eta > 0 .$ . The dual description of $L _ { \Gamma }$ rests on two quantities built from the prior and the state: the Gibbs tilt $\rho _ { \eta , S }$ of π toward S at scale $\eta ,$ and the associated scale-η log-partition $W _ { \eta } ( S )$ . For $S \in \mathbb { R } ^ { K }$ define

$$
[ \rho _ { \eta , S } ] _ { i } : = \frac { \pi ( i ) e ^ { \eta S ( i ) } } { \sum _ { j } \pi ( j ) e ^ { \eta S ( j ) } } , \qquad W _ { \eta } ( S ) : = \frac { 1 } { \eta } \log \sum _ { i = 1 } ^ { K } \pi ( i ) e ^ { \eta S ( i ) }\tag{2.4}
$$

Proposition 2.2 (Dual form of the terminal concentration functional). For every $S \in \mathbb { R } ^ { K }$ and $\Gamma \geq 0$

$$
L _ { \Gamma } ( S ) = \operatorname* { i n f } _ { \eta > 0 } \left\{ { \frac { \Gamma } { \eta } } + W _ { \eta } ( S ) \right\}\tag{2.5}
$$

Equivalently, the Gibbs variational identity gives

$$
W _ { \eta } ( S ) = \operatorname * { s u p } _ { \nu \in \Delta ( [ K ] ) } \left\{ \langle \nu , S \rangle - \frac { 1 } { \eta } \mathrm { K L } ( \nu \| \pi ) \right\}\tag{2.6}
$$

and the optimizer is $q = \rho _ { \eta , S }$

The function $W _ { \eta } ( S )$ is the scaled log-moment generating function of the score S under the prior π, while $L _ { \Gamma } ( S )$ says how much terminal mass a comparator of complexity budget Γ can concentrate in the direction S. Identity (2.5) is the Cramér–Chernof mechanism written as an equality. With $\Lambda ( \eta ) : = \eta W _ { \eta } ( S ) =$ log $\mathbb { E } _ { i \sim \pi } \lceil e ^ { \eta S ( i ) } \rceil$ the cumulant generating function of the state under the prior and $\Lambda ^ { * }$ its Legendre conjugate, (2.5) reads ${ \cal L } _ { \Gamma } ( S ) = \mathrm { i n f } _ { \eta > 0 } ( \bar { \Gamma } + \Lambda ( \eta ) ) / \eta = \operatorname* { s u p } \{ s \in$ $\mathbb { R } : \Lambda ^ { * } ( s ) \le \Gamma \}$ , the inverse Cramér transform: the largest displacement attainable at rate Γ. A classical tail bound performs the same optimization over η and keeps an inequality; here the optimized value is equal to the benchmark. Any looseness downstream therefore comes from committing to one scale in advance — the fixed η need not be the minimizer at the realized state — while the mechanism itself contributes none, a distinction the fixed-scale analysis of Section 3 makes quantitative.

## 2.3 Robust Bayes, maximum entropy, and exponential-family structure

The optimizer $\rho _ { \eta , S }$ of the Gibbs variational identity has a second characterization, as the maximum-entropy distribution matching the exposed moment—the one feature of a distribution ν that a linear terminal payof can see, its mean score $\langle \nu , S \rangle$ . That second characterization is the robust-Bayes reading of the terminal functional.

Proposition 2.3 (Robust Bayes / maximum entropy). Fix $S \in \mathbb { R } ^ { K }$ and $\eta > 0 .$ . Then:

(a) $\rho _ { \eta , S }$ uniquely maximizes $\langle \nu , S \rangle - \eta ^ { - 1 } \mathrm { K L } ( \nu \| \pi )$ over $\nu \in \Delta ( [ K ] )$

(b) $I f m _ { \eta } : = \langle \rho _ { \eta , S } , S \rangle$ , then $\rho _ { \eta , S }$ is the unique minimizer of $\mathrm { K L } ( \nu \| \pi )$ subject to the linear moment constraint $\langle \nu , S \rangle = m _ { \eta } .$

Thus the posterior is simultaneously a best response, a Bellman equalizer, and the maximum-entropy distribution consistent with the currently exposed suficient statistic. The exponential family appears because the concentration game is linear in the state, so the Gibbs variational identity $\left( 2 . 6 \right)$ applies at every scale, and varying η traces out a onedimensional exponential family in which the intrinsic-time density $Q _ { t } ( c )$ is the one-step analogue of the Fisher information (Appendix A.1.2).

## 2.4 The one-scale centered RCGF

For $p \in \Delta ( [ K ] )$ and a centered move $z \in \mathbb { R } ^ { K }$ with $\langle p , z \rangle = 0$ , define

$$
Q _ { \eta } ( p , z ) : = \frac { 1 } { \eta ^ { 2 } } \log \sum _ { i = 1 } ^ { K } p ( i ) e ^ { \eta z ( i ) }\tag{2.7}
$$

Equivalently, in loss coordinates,

$$
Q _ { t } ( c ) = \frac { 1 } { \eta _ { t } ^ { 2 } } \psi _ { p _ { t } } ( - \eta _ { t } ; c _ { t } ) , \qquad \psi _ { p } ( \lambda ; c ) : = \log \sum _ { i } p ( i ) e ^ { \lambda ( c ( i ) - \langle p , c \rangle ) }\tag{2.8}
$$

The argument of $Q _ { t }$ names the loss sequence it is evaluated on, while the play and scale are those of the round: $Q _ { t } ( c )$ abbreviates $Q _ { \eta _ { t } } ( p _ { t } , z _ { t } )$ , and later sections write $Q _ { t } ( g )$ and $Q _ { t } ( \ell )$ for the same round-t functional on other score sequences. Here $\eta ^ { 2 } Q _ { \eta } ( p , z ) = \log \mathbb { E } _ { p } [ e ^ { \eta z } ]$ is the centered cumulant generating function (CGF) of nature’s score z under the learner’s mixed action p at scale η. The quantity carried through the paper is not that CGF but its $\eta ^ { - 2 }$ rescaling $Q _ { \eta } ,$ the normalization under which $Q _ { \eta } \to { \textstyle \frac { 1 } { 2 } } \mathrm { V a r } _ { p } ( z )$ as $\eta \downarrow 0 ;$ it is called the rescaled centered cumulant generating function, abbreviated RCGF, or equivalently the intrinsic-time density. The distinction is not cosmetic: the two difer by a factor that itself depends on the scale, so a statement about $Q _ { \eta }$ across scales is not a statement about the CGF across scales. “CGF” below always means the unrescaled $\eta ^ { 2 } Q _ { \eta }$ . It is the round’s primitive information-theoretic quantity in the framework: it measures how big nature’s move is, at the learner’s chosen scale $\eta ,$ in units that make the Bellman update an algebraic identity and the per-round constraint $Q _ { \eta } ( p , z ) \leq q$ caps how concentrated nature’s centered tilt can be at that scale. It is not a variance surrogate: at small η it reduces to half the variance $\begin{array} { r } { \frac { 1 } { 2 } \mathrm { V a r } _ { p } ( z ) } \end{array}$ (Corollary 2.5), at large η it captures the worst-tail behavior of z under p, and for any η it is the unique non-negative quantity for which $W _ { \eta } ( S + z ) - W _ { \eta } ( S ) = \eta Q _ { \eta } ( \rho _ { \eta , S } , z )$ holds (Theorem 3.2). Bounding $Q _ { \eta } ( p , z ) \leq q$ therefore caps nature’s coincidence budget at the chosen scale, and is the only constraint the game requires.

Three readings of the same functional — a coincidence budget over the moments $\mathbb { E } _ { p } [ z ^ { n } ]$ , the constraint half of a robust-Bayes min-max value, and a prior-to-posterior relative-entropy increment — are developed in Appendix A.1.3. There the moment expansion also exhibits variance $( n = 2 )$ and range $( n = \infty )$ as truncations, which is why they appear later as derived relaxations (Section 3.3). The same primitive recurs at every level of the framework: nature’s one-round move (here), the comparator’s tilt of the prior (Proposition 2.2), the bandit estimator’s importance-weighted score (Theorem A.14), and the Thompson-sampled posterior’s likelihood ratio (Section 8.2).

The centered RCGF also admits an integral representation as a weighted average of tilted variances.

Proposition 2.4 (Tilted-variance representation). Let $p _ { s } ^ { ( \eta , z ) } ( i ) \propto p ( i ) e ^ { s \eta z ( i ) } f o r s \in [ 0 , 1 ] .$ . Then

$$
Q _ { \eta } ( p , z ) = \int _ { 0 } ^ { 1 } ( 1 - s ) \mathrm { V a r } _ { i \sim p _ { s } ^ { ( \eta , z ) } } ( z ( i ) ) d s\tag{2.9}
$$

Consequently $Q _ { \eta } ( p , z ) \ge 0 ,$ , with equality if z is p-a.s. constant.

The tilt family $p _ { s } ^ { ( \eta , z ) }$ interpolates from the learner’s play p at $s = 0$ to the within-round update at $s = 1$ , and the representation makes the non-negativity of $\dot { Q } _ { \eta }$ manifest and exhibits both the small-η variance limit and the bounded-range bound as immediate corollaries.

Corollary 2.5 (Variance and bounded-range relaxations). $I f \eta \downarrow 0$ then $\begin{array} { r } { Q _ { \eta } ( p , z ) = \frac { 1 } { 2 } \mathrm { V a r } _ { p } ( z ) + o ( 1 ) . \ I f z ( i ) \in [ a , b ] } \end{array}$ for all i, then $\begin{array} { r } { Q _ { \eta } ( p , z ) \le ( b - a ) ^ { 2 } / 8 . } \end{array}$

The same representation also controls how the primitive responds to a perturbation of the mixed action. Contaminating the play with a point mass moves the centered RCGF by an influence that is bounded below by a linear term and above by the exponential of the largest centered score: linear in the depth of the score range, exponential in its height (Corollary A.1, Appendix A.1). An estimate of the round’s centered RCGF from draws of p therefore has bounded per-draw influence whenever the centered scores are bounded, with the bound set by the endpoints of their range and the scale alone; the number of experts does not enter. The primitive is estimable at moderate scale in exactly this sense.

## 3 The fixed-scale concentration game

The fixed-scale game holds the measurement scale η fixed across all rounds. A single log-partition potential then governs the entire run: its one-step increment is exactly the centered RCGF, and the transport identity, the dual regularized form (Appendix A.2.3), and the variance and range relaxations all read of that one potential. The exact one-round value closes the fixed-scale picture before the scale is allowed to vary in Section 4.

## 3.1 Nature’s feasible set and the Bellman equalizer

The fixed-scale game confines nature’s per-round move to a one-round information budget on the centered score. For $p \in \Delta ( [ K ] ) , q \geq 0 ,$ , and $\eta > 0 ,$ , set

$$
{ \mathcal { Z } } _ { \eta , q } ( p ) : = \left\{ z \in \mathbb { R } ^ { K } : \langle p , z \rangle = 0 , Q _ { \eta } ( p , z ) \leq q \right\}\tag{3.1}
$$

Game 3.1 (Fixed-scale concentration game). Game 1.1 with one datum frozen: the scale is held constant, $\eta _ { t } \equiv \eta ,$ confining nature’s round-t move to $\mathcal { Z } _ { \eta , q _ { t } } ( p _ { t } )$ . Move order, state, and the terminal payof $L _ { \Gamma } ( S _ { T } )$ at time $T + 1$ are unchanged.

A single potential runs the game: the log-partition function $\begin{array} { r } { W _ { \eta } ( S ) = \eta ^ { - 1 } \log \sum _ { i } \pi ( i ) e ^ { \eta S ( i ) } } \end{array}$ , the entropic counterpart of a drifting-game potential — the potential-function device of boosting-style min-max analyses [45]. When the learner plays the Gibbs weights $p = G _ { \eta } ( S ) = \nabla W _ { \eta } ( S )$ , one round changes the potential by the amount $W _ { \eta } ( S + z ) - W _ { \eta } ( S ) =$ $\eta Q _ { \eta } ( p , z )$ the round’s centered RCGF and nothing more—which nature’s budget caps at ηq . The equalizer simplifies the solution: the round’s loss is capped by nature’s budget whatever the state and whichever feasible direction nature moves, so the rounds do not interact and the potential telescopes from $W _ { \eta } ( 0 ) = 0$ to the closed form of Theorem 3.2. A sequential game is otherwise solved by a backward recursion from the horizon; against the equalizer that recursion collapses to a forward sum. The game is constrained: there is no entropy penalty in the primitive description, and regularization appears only after dualizing the terminal comparator class, which turns the game into the entropic FTRL update on the cumulative scores (Appendix A.2.3).

The value-to-go defined next is an admissible Bellman relaxation: it dominates the game’s value from every state, and at the equalizer play its round decrement is nonnegative on the feasible set and vanishes at nature’s budget-saturating reply. Verifying it therefore takes one round’s inequality, which every round inherits, where computing the value outright would take the backward recursion (Appendix A.2.1).

Theorem 3.2 (Bellman relaxation and equalizer). Define

$$
U _ { t } ( S ) : = \frac { \Gamma } { \eta } + W _ { \eta } ( S ) + \eta \sum _ { s = t } ^ { T } q _ { s } , \qquad 1 \le t \le T + 1\tag{3.2}
$$

Then $U _ { t }$ is an admissible Bellman relaxation of Game 3.1. In particular, $\mathrm { V a l } _ { t } ^ { \eta , \Gamma } ( S ) \leq U _ { t } ( S )$ , and the Bellman equalizer is the Gibbs distribution $p = \rho _ { \eta , S } = \nabla W _ { \eta } ( S )$ . Moreover, for that choice of p and any feasible z,

$$
W _ { \eta } ( S + z ) - W _ { \eta } ( S ) = \eta Q _ { \eta } ( p , z ) \leq \eta q _ { t }\tag{3.3}
$$

The inequality $\eta Q _ { \eta } ( p , z ) \leq \eta q _ { t }$ in (3.3) is by hypothesis (nature’s per-round budget); the equality $W _ { \eta } ( S + z ) -$ $W _ { \eta } ( S ) = \eta Q _ { \eta } ( p , z )$ is exact and is the content of the theorem. At nature’s best response $Q _ { \eta } ( p , z ) = q _ { t }$ the increment inequality is tight, so the Bellman relaxation $U _ { t }$ closes at the Bellman increment. The relaxation is an equality against the game value only when the committed scale also happens to be the dual optimizer of $L _ { \Gamma }$ at the realized terminal state, which the learner cannot guarantee in advance (Appendix A.2.1). In particular, the Bellman bound $\Gamma / \eta + \eta V$ on the cumulative budget regret is strict in general: the exact minimax value can be smaller, since equalizing is weaker than round-value minimaxity and the closed form is a consequence of equalizing.

Nature’s best response to the equalizer follows from the same increment. It is the centered move that saturates the budget, $Q _ { \eta } ( p , z ) = q _ { t }$ , and aligns with the worst comparator direction: by the Gibbs variational identity (Propositions 2.2 and 2.3) that comparator is the Gibbs tilt $\rho _ { \eta , S }$ of the prior by $\eta S ,$ , and the saturating z is the exponential-family tilt of p toward it. This is the robust-Bayes picture of Section 2.3 read from nature’s side: the worst comparator is a posterior at inverse temperature η over the comparator class, and the conditional law the equalizer tracks runs along the same exponential family.

Corollary 3.3 (Bellman bound on the budgeted fixed-scale game). If nature is constrained only by a cumulative budget $\textstyle \sum _ { t = 1 } ^ { T } q _ { t } \leq V$ , then for every $\eta > 0$

$$
\mathrm { V a l } _ { 1 } ^ { \eta , \Gamma } ( 0 ) \leq \frac { \Gamma } { \eta } + \eta V , \qquad s o \qquad \operatorname* { i n f } _ { \eta > 0 } \mathrm { V a l } _ { 1 } ^ { \eta , \Gamma } ( 0 ) \leq \operatorname* { i n f } _ { \eta > 0 } \left[ \frac { \Gamma } { \eta } + \eta V \right] = 2 \sqrt { \Gamma V }\tag{3.4}
$$

the infimum attained at $\eta = \sqrt { \Gamma / V }$ by $A M { - } G M \left( \Gamma / \eta + \eta V \geq 2 \sqrt { \Gamma V } \right) .$ . At the horizon $T = 1$ the same statement reads $\Gamma / \eta + \eta q ,$ , minimized at $\eta = { \sqrt { \Gamma / q } } ,$ , where it equals $2 { \sqrt { \Gamma q } } .$ a sub-Gaussian concentration profile in the budget pair $( \Gamma , q )$

The first inequality is Theorem 3.2 summed across rounds; the second line records the standard AM–GM optimization. The bound $2 \sqrt { \Gamma V }$ is the Bellman upper envelope; the exact minimax value can be strictly smaller (cf. the remark following Theorem 3.2), by an amount computed in closed form for the two-expert instance (Proposition $\mathrm { A } . 6 ,$ Appendix A.3). That slack is a short-horizon efect: under the cumulative budget the two sides meet from a finite horizon on, so (3.4) holds with equality at constant scale (Proposition 4.2).

## 3.2 One-step transport identity

The fixed-scale value bound of Theorem 3.2 came from telescoping the potential; tracking the same telescoping against a fixed comparator ν instead gives a per-round accounting. It is the constant-scale case of (1.1), in which the retempering drift vanishes and only the intrinsic-time loss and the terminal transport remain; the general form, with the scale free to move, is Theorem 4.3.

Proposition 3.4 (One-step transport identity). Fix $\eta ~ > ~ 0 , ~ p ~ \in ~ \Delta ( [ K ] )$ , and centered z. Define $p ^ { + } ( i ) : =$ $\begin{array} { r } { p ( i ) \overline { { e } } ^ { \eta z ( i ) } / \sum _ { j } p ( j ) e ^ { \eta z ( j ) } } \end{array}$ . Then for every comparator $\nu \in \Delta ( [ K ] )$

$$
\langle \nu , z \rangle = \eta Q _ { \eta } ( p , z ) + \frac { \mathrm { K L } ( \nu \| p ) - \mathrm { K L } ( \nu \| p ^ { + } ) } { \eta }\tag{3.5}
$$

Summing over a fixed-scale trajectory gives

$$
\langle \nu , S _ { T } \rangle = \eta \sum _ { t = 1 } ^ { T } Q _ { \eta } ( p _ { t } , z _ { t } ) + \frac { \operatorname { K L } ( \nu \| \pi ) - \operatorname { K L } ( \nu \| p _ { T + 1 } ) } { \eta }\tag{3.6}
$$

This is the fixed-scale information balance. The immediate loss is the one-scale centered RCGF $\eta Q _ { \eta } ;$ the remainder is transported as terminal relative entropy. Read (3.5) from the left: the comparator’s gain on the round is the learner’s information loss, plus the ground the comparator gains on the learner’s belief, measured as the drop in relative entropy from the comparator to it.

## 3.3 Variance and range as upper bounds on the centered RCGF

The centered RCGF has the tilted-variance representation of Proposition 2.4, and its two classical proxies are read of from it directly. Retained as pointwise upper bounds they are

$$
\begin{array} { r l r l } & { Q _ { \eta _ { t } } ( p _ { t } , c ) \le ( e - 2 ) \operatorname { V a r } _ { p _ { t } } ( c ) } & & { ( \mathrm { f o r } \eta _ { t } \le 1 \mathrm { a n d } c \in [ 0 , 1 ] ^ { K } ) , } \\ & { Q _ { \eta _ { t } } ( p _ { t } , c ) \le \operatorname { r a n g e } ( c ) ^ { 2 } / 8 } \end{array}\tag{3.7}
$$

the first the non-asymptotic entropy-version second-order bound [15] (its sharper $\eta _ { t } \downarrow 0$ form is the variance-limi equality $\begin{array} { r } { Q _ { \eta _ { t } } \to \frac { 1 } { 2 } } \end{array}$ Var of Corollary 2.5), the second Hoefding’s lemma. Replacing the exact $Q _ { t }$ in the regret decomposition (4.8) by either proxy gives a looser upper bound on regret, recovering the classical second-order and Hoefding-style envelopes. Each trades information about $c \boldsymbol { s }$ tail shape under $p _ { t }$ for an easier-to-evaluate quantity, and the inequality is strict whenever c is non-Gaussian under $p _ { t } .$ . The two proxies do not refine each other: as constraint sets on nature, $\{ ( e - 2 ) \mathrm { V a r } _ { p  t } ( c ) \leq \beta \}$ and $\{ { \mathrm { r a n g e } } ( c ) ^ { 2 } \leq 8 \beta \}$ both sit inside $\{ Q _ { \eta _ { t } } ( p _ { t } , c ) \leq \beta \}$ , but neither contains the other, and which is smaller depends on how concentrated $p$ is (Appendix $\phantom { - } 8 . 2 . 2 )$ . Retaining the remainder turns the range relaxation into an identity: at a fixed rate and bounded range the classical Hedge bound is the range-relaxed game plus an explici nonnegative slack, the terminal posterior mismatch together with the cumulative gap between the Hoefding proxy and the true RCGF increment (Appendix A.2.2).

The learner chooses $\eta _ { t }$ before seeing $c _ { t }$ and observes only $Q _ { \eta } ( p _ { t } , z _ { t } )$ at that single scale, leaving the rest of the profile $\eta \mapsto Q _ { \eta } ( p _ { t } , z _ { t } )$ unobserved. Diferent inverse temperatures reveal diferent structure — variance-like at small $\eta ,$ tail-like at large $\eta - s 0$ intrinsic time is algorithm-dependent, and the RCGF constraint set (A.9) moves with $\eta _ { t }$ where the range constraint is scale-free. The tradeof the schedule must equalize is visible in one instance: for a binary centered move on $\{ - 1 , + 1 \}$ under $\begin{array} { r } { p _ { t } = ( \frac { 1 } { 2 } , \frac { 1 } { 2 } ) , Q _ { t } ( \eta ) = \log \cosh ( \eta ) / \eta ^ { 2 } } \end{array}$ falls from $\frac { 1 } { 2 }$ at $\eta  0 ^ { + }$ to 0 as $\eta  \infty$ even as the round’s intrinsic-time loss $\eta Q _ { t } = \log \cosh ( \eta ) / \eta$ rises.

## 3.4 The exact one-round value

The Bellman certificate of Corollary 3.3 is an upper bound, and it is strict already at a single round. The one-round value is known exactly on two slices — two experts at every comparator budget (Proposition $\phantom { + } \Lambda . 6 ) ,$ and every $K$ at a comparator budget large enough that the relative-entropy ball is the whole simplex (Proposition A.5) — and on both slices it sits strictly below the Bellman envelope $\Gamma / \eta + \eta q$ . The Gibbs play equalizes at every state but need not be minimax: under a non-uniform prior the two part already in the first round, and from a skewed state the exact minimax play over-tilts relative to the Gibbs iterate, so no elementary closed form survives past $T = 1$ . Appendix A.3 develops the one-round game in full: the nature-side certificate, the two exact values, a pointwise alignment identity confining the Bellman gap to a single bounded terminal functional, and the exact two-round computation.

## 4 Changing scale: retempering, allocation, and active constraints

The fixed-scale game is clean because one Bellman potential governs all rounds. Variable scale means switching Bellman potentials. The loss from switching them is the drift term.

## 4.1 From per-round caps to a cumulative budget

The scale changes roles here: data of Game 1.1, it becomes the learner’s own move, selected round by round with the play pinned to the Gibbs parameterization of (2.1), so the scale is the learner’s only remaining choice. Alongside that promotion, nature’s dificulty is capped once over the whole run instead of round by round, which lets it choose where to concentrate.

Game 4.1 (The self-tuned concentration game ${ \mathcal { G } } ^ { \mathrm { s t } } ( \Gamma , V ) )$ . Game 1.1 with two parts changed. Learner’s move: the learner also selects the scale $\eta _ { t } > 0$ each round (predictably, from $c _ { 1 } , \ldots , c _ { t - 1 } ) ,$ , its play pinned to the Gibbs parameterization $p _ { t } ( i ) \propto \pi ( i ) e ^ { - \eta _ { t } C _ { t - 1 } ( i ) }$ of (2.1). Nature’s constraint: the per-round caps $q _ { t }$ are replaced by a single budget over the run,

$$
\sum _ { t = 1 } ^ { T } Q _ { t } ( c ) \leq V \qquad { \mathrm { ( i n t r i n s i c - t i m e ~ b u d g e t ) } } ,\tag{4.1}
$$

$$
\begin{array} { r } { \mathrm { K L } ( \nu \| \pi ) \leq \Gamma \qquad \mathrm { ( c o m p a r a t o r b u d g e t , f o r } \nu \in \Delta ( [ K ] ) ) , } \end{array}\tag{4.2}
$$

with π of full support and payof $\begin{array} { r } { R _ { T } ^ { c } ( \nu ) = \sum _ { t = 1 } ^ { T } \langle p _ { t } - \nu , c _ { t } \rangle } \end{array}$ to nature. The payof rule is unchanged: $R _ { T } ^ { c } ( \nu ) =$ $\langle \nu , S _ { T } \rangle$ (Proposition 2.1), so its supremum over the comparator budget (4.2) is the terminal payof $L _ { \Gamma } ( S _ { T } )$ . The game’s minimax value is

$$
{ \mathrm { V a l } } { \big ( } \mathcal { G } ^ { \mathrm { s t } } ( \Gamma , V ) { \big ) } = \operatorname* { i n f } _ { ( \eta _ { t } ) { \mathrm { p r e d i c t a b l e } } \ } \operatorname* { s u p } _ { \nu : { \mathrm { K L } } ( \nu \| \pi ) \le \Gamma } R _ { T } ^ { c } ( \nu )\tag{4.3}
$$

Written round-by-round, Game 4.1 is the alternating min-max in $\begin{array} { r } { \dot { \mathbf { \varphi } } _ { \eta _ { 1 } , p _ { 1 } } \operatorname* { s u p } _ { c _ { 1 } } \cdot \cdot \cdot \operatorname* { i n f } _ { \eta _ { T } , p _ { T } } \operatorname* { s u p } _ { c _ { T } } \operatorname* { s u p } _ { \nu } \sum _ { t } \left. p _ { t } - \nu , c _ { t } \right. } \end{array}$ over predictable $( \eta _ { t } , p _ { t } )$ , per-round moves constrained by an allocation $\left( q _ { t } \right)$ summing to at most $V ,$ and a terminal comparator-selection step inside the budget Γ.

The fixed-scale analysis already bounds this chain with no horizon dependence. At any constant scale $\eta _ { t } \equiv \eta$ the drift vanishes; at the Gibbs response the round’s potential increment is exactly $\eta _ { t } Q _ { t }$ , capped at $\eta q _ { t }$ by the budget (Theorem 3.2); and the terminal transport costs at most $\Gamma / \eta$ . Hence

$$
{ \mathrm { V a l } } { \big ( } \mathcal { G } ^ { \mathrm { s t } } ( \Gamma , V ) { \big ) } \ \leq \ \operatorname* { i n f } _ { \eta > 0 } { \Big \{ } \frac { \Gamma } { \eta } + \eta V { \Big \} } = 2 { \sqrt { \Gamma V } }\tag{4.4}
$$

a bound depending only on the budget pair (Γ, V ): doubling $T$ at fixed V leaves it unchanged. The global constraint $\textstyle \sum _ { t } Q _ { t } \leq V$ lets nature allocate dificulty unevenly across rounds; a per-round cap $Q _ { t } \le \beta$ is the case $V = T \beta _ { \mathrm { { \scriptsize : } } }$ , and the per-round maximum $Q _ { T } ^ { * } ( c ) = \operatorname* { m a x } _ { t } Q _ { t } ( c )$ appears in Theorem 4.4 as an edge term for budget concentrated in one round.

The value is horizon-free in the same sense from a finite horizon on, so fixing T alongside the budgets constrains nature less than it appears to.

Proposition 4.2 (Horizon-freeness of the constant-scale value). Fix $\eta ~ > ~ 0 , \ \Gamma ~ \in ~ ( 0 , \Gamma _ { \mathrm { m a x } } )$ with $\Gamma _ { \mathrm { m a x } } : =$ $\operatorname* { m a x } _ { i } \log ( 1 / \pi ( i ) )$ the budget at which the relative-entropy ballfills the simplex, and $V > 0$ , and write $\mathrm { V a l } _ { T } ^ { \eta } ( \Gamma , V ) f o r$ the value of Game 4.1 from $S _ { 0 } = 0$ at the constant scale $\eta _ { t } \equiv \eta$ . Then there is afinite $T _ { 0 } ( \eta , V , \Gamma )$ with

$$
\mathrm { V a l } _ { T } ^ { \eta } ( \Gamma , V ) = \frac { \Gamma } \eta + \eta V \qquad f o r e \nu e r y T \ge T _ { 0 }\tag{4.5}
$$

The mechanism is that nature must do two things at once: exhaust the budget, and finish on the sphere where the Gibbs posterior meets the worst comparator. One round couples them rigidly and the value falls short by the mismatch; a second round decouples them, and from there on both can be met together. Appendix A.4 carries the chain characterization this rests on, together with the terminal alignment functional $\begin{array} { r } { A _ { \Gamma } ( p ) : = \operatorname* { s u p } _ { \nu : \mathrm { K L } ( \nu \| \pi ) \le \Gamma } \left[ \mathrm { K L } ( \nu \| \pi ) - \mathrm { K L } ( \nu \| p ) \right] } \end{array}$ that it optimizes. It also carries the closed form for the two-step gate, the numerical margin by which a varying scale improves on a constant one, and the unbounded-adaptation example that forces the clip at $\eta _ { t } = 1$ in the schedule below.

Past $T _ { 0 }$ the envelope $2 \sqrt { \Gamma V }$ is the constant-scale value exactly, with $T _ { 0 } = 2$ in every instance computed at $K \leq 5 ,$ uniform and skewed priors alike, and $T _ { 0 } = 3$ only within a few percent of saturation. The strictness at $T = 1$ is thus a one-round phenomenon and does not persist. What remains open at $T \geq 2$ is the per-round-capped game, whose reachable set still grows with the horizon through $V = T \beta$ (Section 11).

The constant-scale argument supplies no guarantee for a learner that must set its scale online, without using $V ;$ the decomposition proved next quantifies that plug-in

## 4.2 The exact variable-temperature decomposition

Let $\rho _ { t , \eta } ( i ) \propto \pi ( i ) e ^ { - \eta C _ { t } ( i ) }$ . Define

$$
A _ { t } ( \eta ) : = - \frac { 1 } { \eta } \log \sum _ { i } \pi ( i ) e ^ { - \eta C _ { t } ( i ) } , \qquad D _ { T } : = \sum _ { t = 1 } ^ { T - 1 } \bigl ( A _ { t } ( \eta _ { t } ) - A _ { t } ( \eta _ { t + 1 } ) \bigr )\tag{4.6}
$$

Equivalently, in centered coordinates, $\begin{array} { r } { D _ { T } = \sum _ { t = 1 } ^ { T - 1 } \bigl ( W _ { \eta _ { t + 1 } } ( S _ { t } ) - W _ { \eta _ { t } } ( S _ { t } ) \bigr ) } \end{array}$

The drift $D _ { T }$ is the algebraic loss from switching potentials between rounds. Telescoping the per-round transport identity along the trajectory and grouping the switching points produces three independent terms: a centered RCGF per round (from each measurement-scale-fixed step), a drift (from round-to-round retempering), and a terminal relativeentropy transport (from comparing the comparator with the final posterior). Each term is a clean Bregman / relative-entropy gap; no quadratic-variation surrogate is needed. The decomposition and its sign rule are stated in [7]; the game coordinates here, and the drift’s continuous-time limit (Proposition C.1), are the present development. The sign rule $D _ { T } \leq 0$ for non-increasing schedules follows from $A _ { t } ( \eta )$ being nonincreasing in η at fixed $C _ { t }$ . This is visible from the variational form $\begin{array} { r } { A _ { t } ( \eta ) = \mathrm { { m i n } } _ { \nu \in \Delta ( [ K ] ) } \{ \langle \nu , C _ { t } \rangle + \eta ^ { - 1 } \mathrm { { K L } } ( \nu \| \pi ) \} } \end{array}$ }: increasing η relaxes the relative-entropy penalty, so the minimum value cannot increase.

Theorem 4.3 (Exact variable-temperature decomposition). For the prior-retempered update (2.1), define

$$
Q _ { t } ( c ) = \frac { 1 } { \eta _ { t } ^ { 2 } } \psi _ { p _ { t } } ( - \eta _ { t } ; c _ { t } ) , \qquad V _ { T } ( c ) : = \sum _ { t = 1 } ^ { T } Q _ { t } ( c ) , \qquad B _ { T } ( \nu ) : = \frac { \mathrm { K L } ( \nu \| \pi ) - \mathrm { K L } ( \nu \| \rho _ { T , \eta _ { T } } ) } { \eta _ { T } }\tag{4.7}
$$

Then for every comparator $\nu \in \Delta ( [ K ] )$

$$
R _ { T } ^ { c } ( \nu ) = \sum _ { t = 1 } ^ { T } \eta _ { t } Q _ { t } ( c ) + D _ { T } + B _ { T } ( \nu )\tag{4.8}
$$

$I f \left( \eta _ { t } \right)$ is nonincreasing, then $D _ { T } \leq 0 .$

This identity replaces the backward induction for the Gibbs family: the realized regret factors, pathwise and for every predictable schedule, into three terms that are then bounded separately, with no recursive Bellman equation to solve.

At fixed temperature the sequential structure collapses to a single log-partition: $D _ { T } = 0 ,$ , and the cumulative mix loss $A _ { T } ( \eta _ { T } )$ —the cumulative loss of the Bayesian/Hedge mixture forecaster [25]—is the terminal log-partition function, the negative log Bayes marginal likelihood of the data under prior π and inverse temperature η.

The identity (4.8) reads two ways. The primal reading is a path-information accounting, $\begin{array} { r } { \sum _ { t } \eta _ { t } Q _ { t } , D _ { T } , } \end{array}$ , and $B _ { T } ( \nu )$ the predictable book-keeping quantities incurred along the realized trajectory (Figure 2). The dual reading evaluates the same quantities inside the Bellman relaxation of Game 4.1, where at each round the Gibbs equalizer and nature’s budget-saturating move form a saddle point of the relaxed one-step value worth $\eta _ { t } q _ { t }$ (Theorem 3.2), with $W _ { \eta }$ acting as both primal potential and dual value function. Section 4.4 takes the symmetric dual from the other side; the retempering drift’s continuous-time Itô form subsumes the two as the traversal orders of one exact potential (Appendix C).

## 4.3 The second-order equalizer and nature’s allocation

The constant-scale slice fixes the benchmark and requires no assumption on the schedule: each constant η guarantees $R _ { T } ^ { c } ( \nu ) \leq \Gamma / \eta + \eta V$ , and the oracle scale $\eta ^ { \star } ( V ) = \sqrt { \Gamma / V }$ equalizes the two costs at the AM–GM value $2 \sqrt { \Gamma V }$ of (4.4). The oracle scale consumes one input the online learner does not have, the total intrinsic time. The second-order equalizer is the plug-in rule that removes that input, substituting the running total $V _ { t - 1 } ( c )$ —the part of the intrinsic time already revealed—for $V$ in the oracle scale: fix $\Gamma > 0$ and $C > 0$ and let

$$
\eta _ { t } = \left\{ { 1 , \atop } { V _ { t - 1 } ( c ) = 0 , }  \right.\tag{4.9}
$$

The clip at 1 handles the cold start, and the schedule is computable from realized data with no knowledge of the horizon or of $V .$

Neither of the two properties of (4.9) is assumed; both are derived. First, $V _ { t - 1 } ( c )$ is nondecreasing in t and $v \mapsto$ min $\{ 1 , C \sqrt { \Gamma / v } \}$ is nonincreasing, so the scale is nonincreasing along every loss sequence—in temperature terms, the schedule warms monotonically. Second, the sign rule of Theorem 4.3 then makes the retempering drift a gain to the learner, $D _ { T } \leq 0$ , so the regret is bounded by the intrinsic-time loss plus the transport. The schedule earns its name by pinning the ratio of marginal loss to marginal transport at $2 C ^ { 2 }$ along the whole path, equal round by round at $C = 1 / \sqrt { 2 }$ (Appendix A.4.5).

The intrinsic-time loss is then a Riemann sum of a falling scale, evaluated at each interval’s left endpoint, so it sits above the corresponding integral and exceeds it by at most the largest single increment $Q _ { T } ^ { * } ( c )$ . That bracketing is the two-sided envelope below, whose two sides difer by the discretization cost of sum against integral. $V _ { T } ( c )$ is the cumulative centered RCGF along the realized trajectory, which is a diferent quantity from a sum of variances. The small-η approximation $\begin{array} { r } { V _ { T } \approx \frac { 1 } { 2 } \sum _ { t } \operatorname { V a r } _ { p _ { t } } ( c _ { t } ) } \end{array}$ recovers classical second-order rates. Away from that approximation, when nature plays heavy-tailed centered moves, the variance-relaxed game is loose while the exact RCGF schedule remains tight.

Theorem 4.4 (Two-sided square-root envelope). Let $Q _ { T } ^ { * } ( c ) : = \operatorname* { m a x } _ { 1 \leq t \leq T } Q _ { t } ( c )$ . Under (4.9),

$$
2 C \sqrt { \Gamma V _ { T } ( c ) } - C ^ { 2 } \Gamma \ \leq \ \sum _ { t = 1 } ^ { T } \eta _ { t } Q _ { t } ( c ) \ \leq \ Q _ { T } ^ { * } ( c ) + 2 C \sqrt { \Gamma V _ { T } ( c ) }\tag{4.10}
$$

Consequently, i $\begin{array} { r } { f \mathrm { K L } ( \nu \| \pi ) \leq \Gamma , } \end{array}$

$$
R _ { T } ^ { c } ( \nu ) \leq \Gamma + Q _ { T } ^ { \ast } ( c ) + ( 2 C + C ^ { - 1 } ) \sqrt { \Gamma V _ { T } ( c ) }\tag{4.11}
$$

At $C = 1 / \sqrt { 2 }$ the leading coeficient is $2 C + C ^ { - 1 } = 2 { \sqrt { 2 } } .$ every feasible sequence of Game 4.1 incurs

$$
R _ { T } ^ { c } ( \nu ) \leq 2 \sqrt { 2 } \sqrt { \Gamma V _ { T } ( c ) } + O \big ( \Gamma + Q _ { T } ^ { * } ( c ) \big ) .
$$

The unit coeficient on $Q _ { T } ^ { * } ( c )$ meets the necessity floor of Lemma 4.5 with equality; the same envelope, with an additional $C ^ { 2 } \Gamma$ slack on the upper side, is established in [7]. This envelope is the bound form of the exact identity (4.8): the left and right sides sandwich the cumulative loss $\textstyle \sum _ { t } \eta _ { t } Q _ { t } ( c )$ that the identity isolates, and the upper side, added to $D _ { T } \leq 0$ and the terminal transport $B _ { T } ( \nu ) \leq \Gamma / \eta _ { T }$ , yields (4.11). The gap between the two sides—the edge term $Q _ { T } ^ { * } ( c )$ and the $C ^ { 2 } \Gamma$ initialization overhead—is the discretization cost identified above. Against the known-V benchmark (4.4), the constant oracle scale attains $2 \sqrt { \Gamma V }$ using $V$ as an input; the plug-in schedule attains the same square-root rate pathwise, at the realized $V _ { T } ( c ) \leq V$ and without $V$ as an input, up to the edge terms. A regularized variant $\eta _ { t } = C \sqrt { \Gamma / ( V _ { t - 1 } ( c ) + \Gamma ) }$ replaces the clipping at $\eta _ { t } = 1$ by the ofset Γ and satisfies the same envelope with $V _ { T } ( c )$ replaced by $V _ { T } ( c ) + \Gamma$ (Appendix D.3). It is the form used in the self-play ledger of Section 6.

Nature’s side of the schedule is the allocation problem max $\{ \sum _ { t } \eta _ { t } Q _ { t } : Q _ { t } \ge 0 , \sum _ { t } Q _ { t } \le V \}$ . Against a precommitted nonincreasing sequence this is a linear program solved by front-loading, filling the largest-scale early rounds first. Against the self-tuned schedule (4.9) the Riemann-sum bracketing makes the loss allocation-insensitive up to the edge terms — the equalizer property in time — so nature’s remaining leverage sits exactly at the edges, placing its mass inside the clipped cold start or concentrating in a single round. Both attacks are quantified by the one-round witnesses below.

Lemma 4.5 (Edge-term necessity). Both edge terms in Theorem 4.4 are forced by discrete efects, each by a one-round witness: the initialization constant $C ^ { 2 } \Gamma$ is attained exactly, and the single-jump witness shows the $Q _ { T } ^ { \star }$ coeficient cannot fall below 1.

## 4.4 The active-constraint game and pressure targets

A dual sequential game reverses the round order and drops nature’s cap: nature moves first, unconstrained, and the learner answers by choosing the scale that makes the one-step constraint active — holding with equality — at a pressure target of its choosing.

The target is the round’s loss itself. For a centered move that is not p-a.s. constant and any target $b \in$ $( 0 , \operatorname* { m a x } _ { i \in \mathrm { s u p p } ( p ) } z ( i ) )$ , there is a unique scale $\eta > 0$ solving log $\begin{array} { r } { \sum _ { i } p ( i ) e ^ { \eta z ( i ) } = \eta b } \end{array}$ (Proposition $\mathrm { A } . 7 ) ;$ the left side is $\eta ^ { 2 } Q _ { \eta } ( p , z )$ by the definition (2.7), so the equation asks for $b = \eta Q _ { \eta } ( p , z )$ and the ratio $b / \eta$ is the round's centered RCGE exactly. The learner names the loss it will take this round, and the equation returns the scale that delivers it.

Game A.8 is Game 1.1 with three parts changed: nature reveals its centered move first, no cap is imposed, and the learner answers with the target-induced scale, its update pinned to the Gibbs iterate $p _ { t + 1 } ( i ) \propto p _ { t } ( i ) e ^ { \eta _ { t } z _ { t } ( i ) }$ . The terminal payof is unchanged, the comparator-worst loss along any trajectory again $L _ { \Gamma } ( S _ { T } )$ (Proposition 4.7).

Theorem 4.6 (Exact active-constraint identity). Under Game $A . 8 ,$ for every comparator $\nu \in \Delta ( [ K ] )$ ,

$$
\sum _ { t = 1 } ^ { T } \eta _ { t } \big ( \langle \nu , z _ { t } \rangle - b _ { t } \big ) = \mathrm { K L } ( \nu \| p _ { 1 } ) - \mathrm { K L } ( \nu \| p _ { T + 1 } )\tag{4.12}
$$

Equivalently, with $\kappa _ { t } : = b _ { t } / \eta _ { t }$

$$
\sum _ { t = 1 } ^ { T } \eta _ { t } \left. \nu , z _ { t } \right. = \sum _ { t = 1 } ^ { T } \kappa _ { t } \eta _ { t } ^ { 2 } + \mathrm { K L } ( \nu \| p _ { 1 } ) - \mathrm { K L } ( \nu \| p _ { T + 1 } )\tag{4.13}
$$

Moreover, $\begin{array} { r } { \kappa _ { t } = \frac { 1 } { 2 } \mathrm { V a r } _ { p _ { t } } ( z _ { t } ) + O ( \eta _ { t } ) a s \eta _ { t } \to 0 . } \end{array}$ , so the usual quadratic-variation penalties are the small-η limit of the exact pressure-target line search

The small-scale expansion of $\kappa _ { t }$ is Corollary 2.5 read in these coordinates, since $\kappa _ { t }$ is the round’s centered RCGF.

Two dual games therefore share the one-step transport identity. In Game A, the retempered second-order game, the learner commits to $\eta _ { t }$ first and benefits from warming $( D _ { T } \leq 0$ for nonincreasing $\eta _ { t } ) .$ In Game B, the pressure game, nature reveals its move first and the learner solves a local line search, favored by cooling. The pressure target $b _ { t }$ is the operational variable and the scale $\eta _ { t }$ the dual variable that opens up to satisfy it, which puts boosting (pressure = per-round margin) and concentration (pressure = per-round RCGF) on the same footing. The two readings exchange which quantity is primal and which the Lagrange multiplier, leaving the equilibrium unchanged.

Proposition 4.7 (Terminal collapse of the variable-temperature ledger). Run the variable-temperature game (Game 1.1) with any predictable schedule $( \eta _ { t } ) _ { t = 1 } ^ { T }$ and let $\textstyle S _ { T } = \sum _ { t = 1 } ^ { T } z _ { t }$ be the realized terminal score, $\rho _ { T , \eta _ { T } } ( i ) \propto \pi ( i ) e ^ { \eta _ { T } S _ { T } ( i ) }$ the terminal posterior. With $\begin{array} { r } { \sum _ { t } \eta _ { t } Q _ { t } , D _ { T } , } \end{array}$ , and $B _ { T } ( \nu )$ as in (4.8),

$$
\sum _ { t = 1 } ^ { T } \eta _ { t } Q _ { t } + D _ { T } = W _ { \eta _ { T } } ( S _ { T } )\tag{4.14}
$$

and consequently the comparator-worst regret depends on the realized state alone,

$$
\operatorname* { s u p } _ { \nu : \mathrm { K L } ( \nu \| \pi ) \leq \Gamma } R _ { T } ^ { c } ( \nu ) = L _ { \Gamma } ( S _ { T } )\tag{4.15}
$$

The three horizon terms are therefore not independent: against the worst comparator they telescope onto a function of the realized state alone, so $D _ { T }$ is pinned by the state and the losses, and the schedule enters the worst-case regret only through which terminal state it lets nature reach. The same collapse holds for the pressure game, so Games A and B give the same comparator-worst regret along any common score trajectory and the horizon comparison $\mathrm { { V a l } ( B ) \leq V a l ( A ) }$ reduces to reachability.

## 5 Comparator classes, concentration, and luckiness

Changing the comparator class, the suficient statistic the learner scores, or the terminal event turns the same game into quantile and tracking regret, side information and optimism, event-restricted concentration (Gibbs conditioning, Sanov, Chernof, Cramér), stochastic luckiness (guarantees that sharpen on easy loss sequences), and confidence sets. These are all terminal slices packaged by the terminal payof $L _ { \Gamma }$ .

## 5.1 Changing comparator classes

Quantile regret is the simplest comparator change. If π is uniform and $\nu _ { A }$ is uniform on a set $A \subseteq [ K ]$ , then $\mathrm { K L } ( \nu _ { A } \| \pi ) =$ $\log ( K / | A | )$ ). Every PAC-Bayes/concentration statement immediately yields the corresponding fixed-budget quantile statement by choosing A as a hindsight set of good experts.

A simple dyadic meta-controller makes the guarantee simultaneous over all quantiles: one worker copy of the second order game per budget $\Gamma _ { j } = 2 ^ { j }$ , aggregated by a logarithmic-budget controller playing the mixture of worker predictions. This adds a controller term of O(log log log K) plus a controller intrinsic-time term $O ( \sqrt { \Gamma ^ { \mathrm { c t l } } V _ { T } ^ { \mathrm { c t l } } } )$ in the controller’s own budget and intrinsic time. The dyadic grid reaching the saturated budget has $O ( \log \log K )$ workers, and the additive cost is the comparator complexity of a uniform prior over them, under the envelope of Theorem 4.4 run at the controller level.

A dynamic comparator path $\nu _ { 1 } , \dots , \nu _ { T }$ is handled by the same calculus:

$$
R _ { T } ^ { c , \mathrm { d y n } } ( \nu _ { 1 : T } ) = D _ { T } + B _ { T } ( \nu _ { T } ) + \sum _ { t = 1 } ^ { T } \eta _ { t } Q _ { t } ( c ) + \sum _ { t = 1 } ^ { T - 1 } \left. \nu _ { t + 1 } - \nu _ { t } , C _ { t } \right.\tag{5.1}
$$

Only the last term is new. It is exact before being bounded, so switching/tracking regret enters as a comparator-path transport term, controlled for losses in $[ 0 , 1 ]$ by the path’s total variation, $\begin{array} { r } { | \sum _ { t = 1 } ^ { T - 1 } \langle \nu _ { t + 1 } - \nu _ { t } , C _ { t } \rangle | \leq 2 \sum _ { t = 1 } ^ { T - 1 } t \operatorname { T V } ( \nu _ { t + 1 } , \nu _ { t } ) | } \end{array}$ under the convention $\begin{array} { r } { \mathrm { T V } ( p , q ) = \frac { 1 } { 2 } \| p - q \| _ { 1 } } \end{array}$ <sub>1</sub>.

## 5.2 The suficient statistic as a design choice

The concentration game is defined relative to whatever suficient statistic the learner scores, with no ambient loss vector behind it; varying that choice recovers side information, optimism, specialist experts, sparsity, and bounded-influence robustness as special cases of one game. If the learner applies Bayes to raw losses, optimistic residuals, importance-weighted

estimates, signed margins, one-sided shortfalls, or bounded-influence transforms, then the resulting state, intrinsic time, and move constraint are genuinely diferent.

Optimism scores a residual against a predictable baseline. Suppose the original losses are $\ell _ { t }$ and a predictable baseline $m _ { t }$ is available. Setting $c _ { t } = \ell _ { t } - m _ { t }$ yields

$$
\sum _ { t = 1 } ^ { T } \left. p _ { t } - \nu , \ell _ { t } \right. = D _ { T } + B _ { T } ( \nu ) + \sum _ { t = 1 } ^ { T } \eta _ { t } Q _ { t } ( \ell - m ) + \sum _ { t = 1 } ^ { T } \left. p _ { t } - \nu , m _ { t } \right.\tag{5.2}
$$

Only the residual sequence $\ell _ { t } - m _ { t }$ contributes to the intrinsic-time loss; the final term records the predictable mismatch.   
Slow temporal drift, optimistic game play, model-based predictions, and martingale compensators are all instances.

Six further choices follow the same template, each entering the entropic FTRL update as the cumulative statistic $C _ { t - 1 } ( i )$ of a per-round score, and each with its own consequence: a confidence-weighted score $\begin{array} { r } { \sum _ { s < t } \beta _ { s } ( i ) \ell _ { s } ( i ) } \end{array}$ gives soft specialists, the framework for confidence-rated experts [16] and for the side-information device of some adaptive analyses. A predictable ofset gives variance corrections; the excess-loss positive part $e _ { t } ^ { + } ( i ) : = ( \langle p _ { t } , c _ { t } \rangle - c _ { t } ( i ) ) _ { + }$ <sub>+</sub> seeks sparsity, with $\begin{array} { r } { R _ { T } ^ { c } ( \nu ) \leq \sum _ { t } \left. \bar { \nu } , e _ { t } ^ { + } \right. } \end{array}$ and range $( e _ { t } ^ { + } ) \leq \mathrm { r a n g e } ( e _ { t } )$ ; and a Winsorization or Catoni transform [14] caps the loss on heavy tails. In each case $p _ { t }$ is the max-entropy distribution given the chosen statistic, $Q _ { t }$ is the RCGF of the new observation in the corresponding exponential family, and a mismatch term records exactly how the game on the raw losses difers from the reduced game on the transformed ones. Heavy-tailed scores need the bounded-influence transforms. A raw move with unbounded coordinates can send the one-round RCGF to +∞, and running the same game on a truncated score restores a finite round while leaving every identity intact. The truncation is then carried as an exact additive bias in the comparator’s gain. Table 3 collects them; specialist aggregation also scales, combining an ensemble’s predictions on unlabeled data through a minimax game of the same kind [9, 10]. The framework absorbs each as a suficient-statistic choice, and no separate algorithm is needed.

## 5.3 Event-restricted terminal games, Gibbs conditioning, and Sanov

The terminal support problem may restrict comparators to an event $A \subseteq [ K ]$ , where the standard form uses a relativeentropy ball.

Proposition 5.1 (Event-restricted terminal game). Fix an event $A \subseteq \left\lceil K \right\rceil$ and $\eta > 0 .$ . Then

$$
\operatorname* { s u p } _ { \nu \in \Delta ( [ K ] ) , \operatorname { s u p p } ( \nu ) \subseteq A } \left\{ \langle \nu , S \rangle - \frac { 1 } { \eta } \mathrm { K L } ( \nu \| \pi ) \right\} = \frac { 1 } { \eta } \log \sum _ { i \in A } \pi ( i ) e ^ { \eta S ( i ) }\tag{5.3}
$$

Equivalently,

$$
\frac { 1 } { \eta } \log \sum _ { i \in A } \pi ( i ) e ^ { \eta S ( i ) } = \frac { \log \pi ( A ) } { \eta } + \frac { 1 } { \eta } \log \mathbb { E } _ { i \sim \pi ( \cdot | A ) } e ^ { \eta S ( i ) }\tag{5.4}
$$

The optimizer is the Gibbs tilt of the conditioned prior π $( \cdot \mid A )$

This is the discrete Gibbs-conditioning principle in game form — conditioned on an event, inference proceeds from the prior conditioned on that event: the event cost $\mathrm { i } s - \log \pi ( A )$ , and conditional concentration inside the event is handled by the same log-partition as before.

The same comparator-class geometry underlies the classical large-deviation bounds. On the simplex of types (empirical distributions) $\Delta ( [ K ] )$ , a closed set E has empirical-measure exponent in $\mathrm { f } _ { \nu \in E } \operatorname { K L } ( \nu \| p )$ , the Lagrange dual of the same comparator-class variational problem that $L _ { \Gamma }$ solves (Proposition 2.2); when E is a face $\{ \nu : \operatorname { s u p p } ( \nu ) \subseteq A \}$ the dual has the closed form of Proposition 5.1. Chernof, Cramér, Gibbs conditioning, and Sanov are then each a comparator-class concentration game, sharing one derivation that difers only in where it is entered (Table 1; for the standard treatment see [26]). Fixing the triple $( \pi , S , A )$ , the comparator-class dual supplies the variational value, the closed form of Proposition 5.1 evaluates it on a face, and optimizing the scale converts it into the rate function — one argument, proved once in the two propositions, and the whole of the game’s contribution to every row. Only the closing step from rate function to probability statement difers across rows, and each is classical: exponential Markov after tilting for Chernof and Cramér, the conditional-limit theorem for the I-projection, and n-scaling with closure and interior conditions for Sanov (Appendix A.5).

## 5.4 Comparator-centered luckiness

A luckiness guarantee is one whose regret bound adapts to the dificulty of the realized loss sequence, sharpening on low-noise instances below the worst-case rate. For losses in [0, 1], the exact intrinsic increment can be upper bounded by

<table><tr><td>Classical bound Triple</td><td> $( \pi , S , A )$ </td><td>Game quantity</td><td>Classical step, taken as given</td></tr><tr><td>Chernoff, iid sum  ${ \bar { X } } _ { n }$ </td><td> $\pi = P , S = n c , A =$   $\{ i : c ( i ) \geq t \}$ </td><td>variational tilt  $\begin{array} { r } { \eta ^ { - 1 } \log \sum _ { i } P ( i ) e ^ { \eta n c ( i ) } } \end{array}$   $= W _ { \eta } ( n c ) \left( { \mathrm { P r o p . ~ } } 2 . 2 \right)$ </td><td>Markov / exponential tilting  $\Rightarrow$   $\operatorname* { P r } ( { \bar { X } } _ { n } \geq t ) \dot { \leq } e ^ { - n I ( t ) }$ </td></tr><tr><td>Cramér large de- same viations</td><td> $( \pi , S , A )$ </td><td>Lagrange dual  $\begin{array} { r } { \operatorname* { i n f } _ { \nu : \mathbb { E } _ { \nu } X \geq t } \operatorname { K L } ( \nu \| P ) = } \end{array}$   $I ( t )$  (Prop. 2.2)</td><td>tilted-law representation, exponen- tial Markov</td></tr><tr><td>conditional limit supp ⊆ A</td><td>I-projection / Gibbs conditioning on</td><td> $\eta \downarrow 0$  optimizer = I-projection of π onto conditional-limit theorem A (Prop. 5.1)</td><td></td></tr><tr><td>Finite-state Sanov</td><td>types  $\Delta ( [ K ] ) , \pi = P ,$  E closed</td><td>Lagrange dual  $\mathrm { i n f } _ { \nu \in E } \mathrm { K L } ( \nu \| P )$  (Prop. 2.2)</td><td>n-scaling, closure/interior condi- tions</td></tr></table>

Table 1: Classical bounds as one comparator-class terminal game. Each fixes the triple $( \pi , S , A ) ;$ ; the comparator-class dual of Proposition 2.2 supplies the variational form, which takes the closed form of Proposition 5.1 on a face; the remaining ingredient is a standard large-deviation step. The shared ingredient is a Gibbs / I-projection of the conditioned base measure. Appendix A.5 carries each row from the game to the classical statement, closing step included.

a comparator-centered second moment.

Proposition 5.2 (Comparator-centered second-order envelope). Assume $c _ { t } ( i ) \in [ 0 , 1 ]$ and $\eta _ { t } \in ( 0 , 1 ] .$ . Define $\Psi _ { t } ( \nu ) : =$ $\begin{array} { r } { \sum _ { i = 1 } ^ { K } p _ { t } ( i ) ( c _ { t } ( i ) - \langle \nu , c _ { t } \rangle ) ^ { 2 } . } \end{array}$ . Then

$$
Q _ { t } ( c ) \leq ( e - 2 ) \Psi _ { t } ( \nu )\tag{5.5}
$$

Hence for a nonincreasing schedule $\begin{array} { r } { ( \eta _ { t } ) , R _ { T } ^ { c } ( \nu ) \leq \eta _ { T } ^ { - 1 } \mathrm { K L } ( \nu \| \pi ) + ( e - 2 ) \sum _ { t = 1 } ^ { T } \eta _ { t } \Psi _ { t } ( \nu ) . } \end{array}$

For i.i.d. losses $c _ { 1 } , c _ { 2 } , \ldots$ in $[ 0 , 1 ] ^ { K }$ with mean $\mu = \mathbb { E } c _ { 1 } ,$ a comparator $\nu \in \Delta ( [ K ] )$ satisfies a low-noise condition with constant $\kappa _ { \nu } \geq 0$ when $\mathbb { E } [ ( c _ { t } ( i ) - \langle \nu , c _ { t } \rangle ) ^ { 2 } ] \le \kappa _ { \nu } ( \mu ( i ) - \langle \nu , \mu \rangle )$ for every $i \in [ K ]$

Theorem 5.3 (Fixed-rate stochastic luckiness). Assume the low-noise condition above with constant $\kappa _ { \nu } ,$ and run fixed-rate Hedge on the composite losses with $\eta \in ( 0 , 1 ] . \ I f \left( e - 2 \right) \kappa _ { \nu } \eta < 1$ , then

$$
\mathbb { E } R _ { T } ^ { c } ( \nu ) \leq \frac { \mathrm { K L } ( \nu \| \pi ) } { \eta ( 1 - ( e - 2 ) \kappa _ { \nu } \eta ) }\tag{5.6}
$$

In particular, choosing $\eta = \mathrm { m i n } \{ 1 , [ 2 ( e - 2 ) \kappa _ { \nu } ] ^ { - 1 } \}$ gives a constant-in-T expected regret bound: $\mathbb { E } R _ { T } ^ { c } ( \nu ) \leq 2 ( 1 + 2 ( e -$ $2 ) \kappa _ { \nu } ) \mathrm { K L } ( \nu \| \pi )$

The same phenomenon survives the second-order schedule: under (4.9),

$$
\mathbb { E } R _ { T } ^ { c } ( \nu ) \leq 2 \Gamma + 2 \mathbb { E } [ Q _ { T } ^ { * } ( c ) ] + ( 2 C + C ^ { - 1 } ) ^ { 2 } ( e - 2 ) \kappa _ { \nu } \Gamma\tag{5.7}
$$

so hard and easy sequences are not diferent theorems but diferent realized fixed points of the same game. The low-noise condition matches the standard point-mass condition at $\nu = \delta _ { k ^ { * } }$ , and is satisfiable with a finite constant only when the comparator ties the minimal mean $\mu ^ { * } = \operatorname* { m i n } _ { j } \mu ( j )$ , so Theorem 5.3 applies on the minimal-mean face.

A comparator-mean monotonicity carries the guarantee past that restriction. Every comparator above the minimal mean is beaten by a linear margin and so bounded a fortiori, and one that ties it inherits the best expert’s constant-in-T bound (Proposition A.9). For finite K with a strict of-face mean gap the remaining tied-minimum face is handled separately (Appendix A.6.2). The no-gap regime, a continuum of experts or means accumulating at $\mu ^ { * }$ , remains open.

## 5.5 Confidence regions and confidence trajectories

Because the terminal term is a relative-entropy transport, exponential-family slices of the game trace out level sets of the comparator’s terminal information transport. For a parametric family $\{ \nu _ { \theta } : \theta \in \Theta \}$ define

$$
\Lambda _ { T } ( \theta ) : = \mathrm { K L } ( \nu _ { \theta } \| \pi ) - \mathrm { K L } ( \nu _ { \theta } \| \rho _ { T , \eta _ { T } } ) = \mathbb { E } _ { \nu _ { \theta } } \left[ \log \frac { \rho _ { T , \eta _ { T } } } { \pi } \right]\tag{5.8}
$$

the amount the data path has moved toward $\nu _ { \theta }$ in relative entropy, and in the exact regret identity $\begin{array} { r } { \langle \nu _ { \theta } , S _ { T } \rangle = \sum _ { t } \eta _ { t } Q _ { t } ( c ) + } \end{array}$ $D _ { T } + \Lambda _ { T } ( \theta ) / \eta _ { T }$ it is literally the leftover description cost of θ after processing the sequence. Higher $\Lambda _ { T }$ marks comparators the data favors over the prior, so the data-supported region is $\mathcal { C } _ { T } ( \Gamma ) : = \{ \theta : \Lambda _ { T } ( \theta ) \geq - \Gamma \}$ , the comparators not yet rejected at log-Bayes-factor level Γ. At point masses this is the standard posterior-tail confidence set, and for a general exponential family it is a super-level set of $\Lambda _ { T }$ in the log-partition geometry, which need not be convex. Where the prior posterior ratio is an e-process — a nonnegative process with expectation at most one at every stopping time — inverting $\mathrm { P R } _ { T } ( \theta ) : = \pi ( \theta ) / \rho _ { T , \eta _ { T } } ( \theta )$ yields anytime-valid sets $\mathcal { C } _ { T } ^ { \mathrm { a n y } } ( \alpha ) : = \{ \theta : \operatorname* { s u p } _ { t \leq T } \operatorname { P R } _ { t } ( \theta ) < 1 / \alpha \}$ . The hypotheses under which the ratio is an e-process, and the cost of a post-hoc reading of such sets, are not pursued here.

Reading these sets at a horizon fixed in advance separates two terms that an anytime construction incurs together. Both invert the same statistic and difer only in which of its values is consulted: $\mathcal { C } _ { T } ^ { \mathrm { a n y } } ( \alpha )$ takes the running supremum, where the fixed-time set $\mathfrak { C } _ { T } ^ { \mathrm { f i x } } ( \alpha ) : = \{ \theta : \mathrm { P R } _ { T } ( \theta ) < 1 / \alpha \}$ takes the endpoint. Writing ${ \mathrm { D D } } _ { T } ( \theta ) : = \Lambda _ { T } ( \theta ) - { \mathrm { i n f } } _ { t \leq T } \Lambda _ { t } ( \theta ) \geq 0$ for the drawdown of the transport trajectory,

$$
\mathcal { C } _ { T } ^ { \mathrm { a n y } } ( \alpha ) = \left\{ \theta : \Lambda _ { T } ( \theta ) > \mathrm { D D } _ { T } ( \theta ) - \log ( 1 / \alpha ) \right\}\tag{5.9}
$$

the fixed-time set read at the tightened level $\log ( 1 / \alpha ) - \mathrm { D D } _ { T } ( \theta )$ . The budget an anytime set holds back for the times it is not consulted is the drawdown, comparator by comparator, and it is read of the realized path without being estimated. Retuning the scale decides whether it keeps growing with the horizon: at a scale held fixed, the drawdown tracks the e-process’s accumulated log-moment deficit, which is linear in $T ;$ retuned to each horizon it stays flat (Appendix $\mathsf { A } . 6 . 3 )$

The other fixed-horizon saving sits in the scale. Proposition 4.2 licenses taking it: the constant scale $\eta = \sqrt { \Gamma / V }$ attains the value from the finite horizon $T _ { 0 } \mathrm { { o n } , }$ , and $T _ { 0 } = 2$ at the comparator budgets of interest. A learner told the horizon in advance therefore pre-commits that scale and incurs no drift, $D _ { T } = 0$ . One that is not must instead run the plug-in schedule of Theorem $4 . 4 ,$ whose leading constant is $2 \sqrt { 2 }$ in place of $2 ,$ or mix over the scale at the Jensen surplus of Section 11.3. Mixing is therefore wider than a fixed-horizon interval read once, at both its own best tuning and the conventional one, and the extra width is visible twice: once directly, and once as coverage the sequence never uses (Appendix $\mathsf { A } . 6 . 3 )$ . The residual conservatism common to all three is the exponential-moment step itself, which the game accounts for in $A _ { \Gamma }$ and no choice of horizon removes. Fixing the horizon therefore concedes nothing to nature, by Proposition 4.2; it recovers the scale.

## 6 Repeated games, equilibria, and the information ledger

Every setting so far has one learner facing nature. Let each player of a repeated game run the concentration game instead, each with its own prior, its own schedule, and its own ledger. The distance from equilibrium is then not merely bounded by those ledgers but equal to a sum of their entries: the duality gap in a zero-sum game (Theorem 6.2), and the distance to coarse correlated equilibrium in a general one (Theorem 6.3). Two things follow from having the equality rather than a bound. The rate is read of the play that actually happened, so it improves automatically on the sequences where the players settle, and costs nothing on the sequences where they do not. And the classical guarantees are recovered by relaxing the entries one at a time, so what each classical step discards is visible as a specific term. No quadratic-variation surrogate enters at any point; the standard variance bounds are the small-scale limit of what follows.

## 6.1 Repeated zero-sum play and intrinsic-time regret matching

Consider a repeated zero-sum matrix game with row-loss matrix $M \in [ 0 , 1 ] ^ { K \times L }$ . If the column player plays $\omega _ { t } \in \Delta ( [ L ] )$ the row-loss vector is $\ell _ { t } ( i ) = ( M \omega _ { t } ) _ { i }$ , and the centered regret vector is $g _ { t } ( i ) : = \langle p _ { t } , \ell _ { t } \rangle - \ell _ { t } ( i )$ with $\langle p _ { t } , g _ { t } \rangle = 0$ Running the retempered concentration game on $\left( g _ { t } \right)$ in place of the raw losses is intrinsic-time regret matching, the strategy used throughout this section.

A superscript or argument on a ledger entry names the sequence that entry is computed on. So $Q _ { t } ( g ) , D _ { T } ^ { g }$ , and $B _ { T } ^ { g } ( \nu )$ are the three entries of Theorem 4.3 — per-round centered RCGF, retempering drift, and terminal relative-entropy transport — evaluated on the centered regret vectors $\left( g _ { t } \right)$ in place of the raw losses, and $\begin{array} { r } { V _ { T } ( g ) : = \sum _ { t = 1 } ^ { T } Q _ { t } ( g ) } \end{array}$ is the corresponding intrinsic time.

Nothing about the ledger changes under this substitution, and the square-root envelope of Theorem 4.4 comes with it.

Theorem 6.1 (Intrinsic-time regret matching). If the learner runs the retempered concentration game on the centered regret vectors $g _ { t } ,$ then for every comparator $\nu \in \Delta ( [ K ] )$ ,

$$
\sum _ { t = 1 } ^ { T } \bigl ( \langle p _ { t } , \ell _ { t } \rangle - \langle \nu , \ell _ { t } \rangle \bigr ) = \sum _ { t = 1 } ^ { T } \eta _ { t } Q _ { t } ( g ) + D _ { T } ^ { g } + B _ { T } ^ { g } ( \nu )\tag{6.1}
$$

In particular, $f o r$ each pure action i and budget $\begin{array} { r } { \Gamma \geq - \log \pi ( i ) , \sum _ { t = 1 } ^ { T } ( \langle p _ { t } , \ell _ { t } \rangle - \ell _ { t } ( i ) ) \leq \Gamma + Q _ { T } ^ { * } ( g ) + ( 2 C + } \end{array}$ $C ^ { - 1 } ) \sqrt { \Gamma V _ { T } ( g ) }$

The schedule needs no knowledge of the loss scale. Scaling all loss vectors by a constant $a > 0$ scales $Q _ { t } ( g )$ and $V _ { T } ( g )$ by $a ^ { 2 }$ and the learning rate by $1 / a$ , so the schedule form is scale-free in the unclipped regime (Appendix D.5).

Several candidate opponent models are accommodated in the prior, before play begins. Pooling them multiplicatively in prior space is ordinary exponential response to the averaged opponent model, and the worst case over poolings is a robust center; running the game on the residuals against that center converges quickly whenever one candidate is close to the opponent’s actual strategy (Appendix A.7.1).

## 6.2 An exact information ledger: two-player duality gap

Now let both players run intrinsic-time regret matching, with respective priors $\pi ^ { \mathrm { r o w } } \in \Delta ( [ K ] )$ and $\pi ^ { \mathrm { c o l } } \in \Delta ( [ L ] )$ and schedules $( \eta _ { t } ^ { \mathrm { r o w } } )$ and $( \eta _ { t } ^ { \mathrm { c o l } } )$ . Each keeps its own centered regret vectors, centered under its own play: the row player’s are $g _ { t } ^ { \operatorname { r o w } } ( i ) : = \langle p _ { t } , M \omega _ { t } \rangle - ( M \omega _ { t } ) _ { i }$ , and the column player’s are $\begin{array} { r } { g _ { t } ^ { \mathrm { c o l } } ( j ) : = ( M ^ { \top } p _ { t } ) _ { j } - \left. \omega _ { t } , M ^ { \top } p _ { t } \right. } \end{array}$ , in the zero-sum convention in which the column player’s loss vector $\mathrm { i } s - M ^ { \top } p _ { t }$

How far the averaged play sits from the game’s value is then the two players’ ledgers added together, entry by entry, with nothing left over.

Theorem 6.2 (Exact information ledger of self-play). With

$$
P _ { T } ^ { \mathrm { r o w } } : = \sum _ { t = 1 } ^ { T } \eta _ { t } ^ { \mathrm { r o w } } Q _ { t } ^ { \mathrm { r o w } } ( g ) , \qquad P _ { T } ^ { \mathrm { c o l } } : = \sum _ { t = 1 } ^ { T } \eta _ { t } ^ { \mathrm { c o l } } Q _ { t } ^ { \mathrm { c o l } } ( g ) ,
$$

let $D _ { T } ^ { \mathrm { r o w } } , D _ { T } ^ { \mathrm { c o l } }$ be the corresponding retempering drifts and $B _ { T } ^ { \mathrm { r o w } } ( \nu ) , B _ { T } ^ { \mathrm { c o l } } ( \nu ^ { \prime } )$ the terminal relative-entropy transports. Then for every comparator pair $( \nu , \nu ^ { \prime } ) \in \Delta ( [ K ] ) \times \Delta ( [ L ] )$

$$
\bar { p } _ { T } ^ { \top } M \nu ^ { \prime } - \nu ^ { \top } M \bar { \omega } _ { T } = \frac { 1 } { T } \Big ( P _ { T } ^ { \mathrm { r o w } } + P _ { T } ^ { \mathrm { c o l } } + D _ { T } ^ { \mathrm { r o w } } + D _ { T } ^ { \mathrm { c o l } } + B _ { T } ^ { \mathrm { r o w } } ( \nu ) + B _ { T } ^ { \mathrm { c o l } } ( \nu ^ { \prime } ) \Big )\tag{6.2}
$$

Suppose further that the complexity budgets admit every pure deviation, $\begin{array} { r l r } { \Gamma ^ { \mathrm { r o w } } } & { { } \ge } & { \operatorname* { m a x } _ { i } \log ( 1 / \pi ^ { \mathrm { r o w } } ( i ) ) } \end{array}$ and $\begin{array} { r } { \Gamma ^ { \mathrm { { c o l } } } \ge \dot { \mathrm { \ m a x } } _ { j } \log ( 1 / \pi ^ { \mathrm { { c o l } } } ( j ) ) } \end{array}$ . Specializing the right-hand side to maximizing comparators inside those budgets then yields the duality gap

$$
\operatorname* { m a x } _ { \omega ^ { \prime } } \bar { p } _ { T } ^ { \top } M \omega ^ { \prime } - \operatorname* { m i n } _ { p ^ { \prime } } p ^ { \prime \top } M \bar { \omega } _ { T } = \frac { P _ { T } ^ { \mathrm { r o w } } + D _ { T } ^ { \mathrm { r o w } } + B _ { T } ^ { \mathrm { r o w } , \star } + P _ { T } ^ { \mathrm { c o l } } + D _ { T } ^ { \mathrm { c o l } } + B _ { T } ^ { \mathrm { c o l } , \star } } { T }\tag{6.3}
$$

where $B _ { T } ^ { \mathrm { r o w , \star } } : = \operatorname* { s u p } \{ B _ { T } ^ { \mathrm { r o w } } ( \nu ) : \mathrm { K L } ( \nu \| \pi ^ { \mathrm { r o w } } ) \leq \Gamma ^ { \mathrm { r o w } } \}$ and analogously for the column player.

Every term on the right of (6.3) is an RCGF or a relative entropy drawn from one of the two players’ own ledgers, and none is a variance. Under smaller budgets the same right-hand side is exact for the relative-entropy-restricted deviation classes, and the unrestricted duality gap on the left may exceed it. The classical self-play gap bound [15] and its quadraticvariation refinements are relaxations of it: setting $C = 1 / \sqrt { 2 }$ and applying the variance bound $Q _ { t } \leq ( e - 2 ) \mathrm { V a r } _ { p _ { t } } ( g _ { t } )$ (Proposition 5.2) to each player’s loss recovers the standard $O ( \sqrt { \log K / T } )$ self-play rate as the small-η limit.

The exact form adds a rate that responds to the play. The horizon-growing part of (6.3) is governed by the loss $\begin{array} { r } { P _ { T } ^ { ( k ) } = \sum _ { t } \eta _ { t } ^ { ( k ) } Q _ { t } ^ { ( k ) } } \end{array}$ and the transport $B _ { T } ^ { ( k ) , \star } \le \Gamma ^ { ( k ) } / \eta _ { T } ^ { ( k ) }$ , both of order $\sqrt { \Gamma ^ { ( k ) } V _ { T } ^ { ( k ) } }$ in the cumulative intrinsic time $\begin{array} { r } { V _ { T } ^ { ( k ) } : = \sum _ { t } Q _ { t } ^ { ( k ) } } \end{array}$ under the optimal schedule for a complexity budget $\Gamma ^ { ( k ) }$ ; the drift is a gain. Sublinear $V _ { T } ^ { ( k ) }$ in either player therefore produces a strictly faster duality-gap rate: at $V _ { T } ^ { ( k ) } = { \cal { O } } ( \log T )$ the exact gap is $O ( { \sqrt { \log T } } / T )$ in expectation, strictly faster than the $\Theta ( 1 / \sqrt { T } )$ worst case and with no hidden logarithmic factors.

Two scope conditions bound that reading. The equality holds for the present family, so it constrains what that family realizes and is not a minimax lower bound over all algorithms; an update rule outside it can realize a smaller gap on a given sequence, and the identity neither predicts nor precludes this. And within the family a gap below the loss’s lower envelope is available only through the drift gain, which leaves it undecided whether $V _ { T } ^ { ( k ) } = \Theta ( T )$ forces a $\Theta ( 1 / \sqrt { T } )$ gap at $C = 1 / { \sqrt { 2 } } \left( \operatorname { A p p e n d i x } \mathrm { A } . 7 . 3 \right)$

## 6.3 General-sum play and equilibrium consequences

Beyond two players the target is a correlated equilibrium notion, and the one-player game composes in the standard way. External regret at most $\varepsilon _ { m } T$ for every player of a finite normal-form game makes the empirical distribution of play an ε-coarse correlated equilibrium (CCE). Swap regret at that level makes it an ε-correlated equilibrium [33], at $\varepsilon = \operatorname* { m a x } _ { m } \varepsilon _ { m }$ in both cases (Propositions A.11 and A.12).

Those are one-way implications, and measuring the distance directly makes them quantitative. For a joint distribution $\sigma \in \Delta ( \prod _ { k } [ K _ { k } ] )$ on action profiles, define the CCE-deviation distance

$$
d ( \sigma , { \mathrm { C C E } } ) : = \operatorname* { m a x } _ { k } \operatorname* { s u p } _ { \nu _ { k } \in \Delta ( [ K _ { k } ] ) } \mathbb { E } _ { a \sim \sigma } \left[ u _ { k } ( \nu _ { k } , \pmb { a } _ { - k } ) - u _ { k } ( \pmb { a } ) \right]\tag{6.4}
$$

where $\begin{array} { r } { u _ { k } : \prod _ { k ^ { \prime } } [ K _ { k ^ { \prime } } ] \to \mathbb { R } } \end{array}$ is player k’s utility and we adopt the convention $u _ { k } = - c _ { k }$ for cost-minimizing players. This is the most any single player could gain by deviating on its own; $\sigma \in \mathrm { C C E }$ if $d ( \sigma , \mathrm { C C E } ) \leq 0$ (the gain can be strictly negative when every unilateral deviation strictly loses), and σ is an ε-CCE if $d ( \sigma , \mathrm { C C E } ) \leq \varepsilon \left[ 1 1 , 3 3 \right]$

Evaluated at the empirical joint $\begin{array} { r } { \sigma _ { T } : = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \otimes _ { k } p _ { t } ^ { ( k ) } } \end{array}$ of the players’ mixed actions, that distance equals the largest of the players’ ledgers.

Theorem 6.3 (Exact per-player external-regret ledger and CCE distance). For a finite game, let each player k run intrinsictime regret matching with prior $\pi ^ { ( k ) }$ , schedule $( \eta _ { t } ^ { ( k ) } )$ , and comparator budget $\Gamma ^ { ( k ) } \geq \operatorname* { m a x } _ { i } \log ( 1 / \pi ^ { ( k ) } ( i ) ) .$ , so that every pure deviation lies in the comparator class. Each player’s external regret admits the exact decomposition (Theorem 4.3) $R _ { T } ^ { ( k ) } ( \nu ) =$ $P _ { T } ^ { ( k ) } + D _ { T } ^ { ( k ) } + B _ { T } ^ { ( k ) } ( \nu )$ into RCGF losses, retempering drift, and terminal relative-entropy transport. The corresponding worst-case-over-comparator sum is $P _ { T } ^ { ( k ) } + D _ { T } ^ { ( k ) } + B _ { T } ^ { ( k ) , \star }$ , with $B _ { T } ^ { ( k ) , \star } : = \operatorname* { s u p } \{ B _ { T } ^ { ( k ) } ( \nu _ { k } ) : \mathrm { K L } ( \nu _ { k } \| \pi ^ { ( k ) } ) \leq \Gamma ^ { ( k ) } \}$ . Then

$$
d ( \sigma _ { T } , \mathrm { C C E } ) = \frac { 1 } { T } \operatorname* { m a x } _ { k } \Big ( P _ { T } ^ { ( k ) } + D _ { T } ^ { ( k ) } + B _ { T } ^ { ( k ) , \star } \Big )\tag{6.5}
$$

Both sides are exact, so for the mixed-action joint the one-way implication above becomes an identity. With a smaller $\Gamma ^ { ( k ) }$ the right-hand side governs only the relative-entropy-restricted deviation class $\{ \nu _ { k } : \mathrm { K L } ( \nu _ { k } \| \pi ^ { ( k ) } ) \leq \Gamma ^ { ( k ) } \}$ . If instead the players sample pure profiles $\mathbf { } \mathbf { a } _ { t }$ and $\sigma _ { T }$ is read as $\begin{array} { r } { \frac { 1 } { T } \sum _ { t } \delta _ { { \bf a } _ { 1 } } } \end{array}$ , the same ledger holds in expectation up to an additive $O ( \sqrt { \log ( \sum _ { k } K _ { k } ) / T } )$ from the play martingales, which are mean-zero at each fixed deviation.

The identity turns a rate question into a question about whether the play settles. Under the self-tuned schedule $\eta _ { t } ^ { ( k ) } = C \sqrt { \Gamma ^ { ( k ) } / ( V _ { t - 1 } ^ { ( k ) } + \Gamma ^ { ( k ) } ) }$ , a trajectory converging to a strict pure Nash profile freezes the scale at a positive value and decays the per-round RCGF geometrically, with no assumption on the rate of approach. Each player’s realized intrinsic time is then bounded, $V _ { T } ^ { ( k ) } = { \cal { O } } ( 1 )$ , and hence $d ( \sigma _ { T } , \mathrm { C C E } ) = O ( 1 / T )$ — strictly faster than the classica $\Theta ( 1 / \sqrt { T } )$ (Theorem A.13, Appendix A.7).

That statement is conditional on convergence to a strict pure profile, which a finite potential or congestion game need not produce: fixed-step multiplicative weights admits limit cycles and chaos there [51]. And while the schedule’s negative feedback — growing $V ^ { ( k ) }$ forces η downward, toward the small-step replicator-dynamics limit — heuristically self-stabilizes, whether $V _ { T } ^ { ( k ) }$ stays sublinear for every such game is a convergence-of-dynamics question the schedule does not settle.

The link between stabilization and the improved rate runs in one direction only. The CCE distance is a per-player maximum, exactly the worst single player’s unilateral deviation gain, and $V _ { T } ^ { ( k ) }$ governs every horizon-growing term of that player’s regret, so max $V _ { T } ^ { ( k ) } = o ( T )$ implies $d ( \sigma _ { T } , \mathrm { C C E } ) = o ( T ^ { - 1 / 2 } )$ . Since $Q _ { t } ^ { ( k ) }$ vanishes exactly as player k’s centered regret vector becomes constant on the support of its last-iterate play (Proposition 2.4), that hypothesis is a last-iterate stabilization statement. The play itself settles, beyond its time average, either by concentrating on a single action or by the regrets flattening at an interior equilibrium. The converse is unavailable from the ledger, since the residual coeficient $2 C - C ^ { - 1 }$ is exactly zero at the headline $C = 1 / \sqrt { 2 }$ (Appendix A.7.3).

Which regime a game falls into is therefore settled by the realized intrinsic time alone. On potential and congestion instances converging to a pure equilibrium, plain Gibbs stabilizes and (6.5) gives the instance-adaptive rate. On interiorequilibrium games such as zero-sum polymatrix games, plain Gibbs need not stabilize: a cyclic loss sequence holds the per-round centered RCGF away from zero, giving $V _ { T } ^ { ( k ) } = \Theta ( T )$ and the classical $O ( 1 / \sqrt { T } )$ , whose small-η variance relaxation recovers the bound of [15]. The guaranteed $O ( 1 / T )$ improvement there needs an optimistic predictable schedule [22] outside the Gibbs family, and unlike the sum of regrets, whose loss-movement telescopes across players under such a schedule $[ 6 6 ] ,$ the CCE maximum admits no cross-player cancellation. Appendix A.7 works the canonical instances through the ledger; in two-player zero-sum, $d ( \sigma _ { T } , \mathrm { C C E } )$ is the larger of the two exact deviation gains, at most the Nash duality gap of (6.3).

## 7 Partial information: bandits, graphs, context, and robust scores

In the full-information game of Sections 2–6 the learner sees nature’s entire loss vector $c _ { t } ,$ , and its regret decomposes exactly into intrinsic-time loss, retempering drift, and terminal relative-entropy transport. Partial information changes one thing (Game 7.1): the learner no longer observes $c _ { t }$ , only the coordinate it samples, and must commit both a play distribution $p _ { t }$ and a possibly distinct sampling law $\mu _ { t }$ before nature moves. It then runs the same intrinsic-time game on an estimated score $\widehat { c } _ { t } ;$ the only new terms are the bandit-specific play and estimation martingales, an explicit exploration cost, and any predictable estimation bias, which separate cleanly from that game (Theorem A.14).

Game 7.1 (The bandit concentration game). Game 1.1’s data — a strictly positive prior $\pi \in \Delta ( [ K ] )$ , a comparator budget $\Gamma \geq 0 ,$ and per-round RCGF budgets $( q _ { t } ) -$ with the state, the moves, and the terminal payof changed. The state is the cumulative estimated centered score $\begin{array} { r } { \widehat { S } _ { t } : = \sum _ { s < t } \widehat { z } _ { s } } \end{array}$ (from $\widehat { S } _ { 0 } = 0 )$ , and the learner selects its own scale as in Game 4.1. Each round, after observing the context $x _ { t }$ in the contextual case, the learner picks $\eta _ { t } > 0$ predictably, plays the Gibbs tilt $p _ { t } = G _ { \eta _ { t } } ( \widehat { S } _ { t - 1 } )$ of the estimated state, and picks a sampling distribution $\mu _ { t } \in \Delta ( [ K ] )$ , which may difer from $p _ { t }$ for exploration. Nature then commits the true loss $\boldsymbol { c } _ { t } \in \mathbb { R } ^ { K }$ , unseen by the learner and held to a per-round budget on the estimated centered RCGF; the learner samples $A _ { t } \sim \mu _ { t } ,$ observes only $c _ { t } ( A _ { t } )$ , forms an estimate $\widehat { c } _ { t }$ of $c _ { t } ,$ and advances the state by the centered estimated score $\widehat { z } _ { t } : = \langle p _ { t } , \widehat { c } _ { t } \rangle \mathbf { 1 } - \widehat { c } _ { t }$ . The payof to nature is the sampled composite-loss regret $\begin{array} { r } { \mathrm { R e g } _ { T } ^ { c } ( u ) : = \sum _ { t = 1 } ^ { T } c _ { t } ( A _ { t } ) - \langle u , C _ { T } \rangle } \end{array}$ , its supremum over the comparator ball $\{ u \in \Delta ( [ K ] ) : \mathrm { K L } ( u \| \pi ) \leq \Gamma \}$ replacing the terminal payof $L _ { \Gamma } ( S _ { T } )$ of Game 1.1.

That budget is imposed on the learner’s estimated centered RCGF $\widehat { Q } _ { t } : = Q _ { \eta _ { t } } ( p _ { t } , \widehat { z } _ { t } )$ : since $c _ { t }$ is committed before the arm is drawn, the constraint is on the conditional mean $\mathbb { E } [ \widehat { Q } _ { t } \mid \mathcal { F } _ { t - 1 } ] \leq q _ { t }$ , the analogue of the full-information cap. Because that expected estimated RCGF is an explicit closed form in the true losses and geometry $\mu _ { t } ,$ it pins a feasible set for the true $c _ { t }$ once $\mu _ { t }$ is fixed.

## 7.1 Generic bandit decomposition

For a comparator $u \in \Delta ( [ K ] )$ the sampled composite-loss regret decomposes pathwise (Theorem A.14, Appendix A.8) as

$$
\mathrm { R e g } _ { T } ^ { c } ( u ) = M _ { T } ^ { \mathrm { p l a y } } + \Xi _ { T } - M _ { T } ^ { \mathrm { e s t } } ( u ) + \mathrm { B i a s } _ { T } ( u ) + \widehat { D } _ { T } + \widehat { B } _ { T } ( u ) + \sum _ { t = 1 } ^ { T } \eta _ { t } \widehat { Q } _ { t }
$$

into the full-information game on estimated scores (the last three terms) plus an observation penalty of play/estimation martingales, exploration cost $\begin{array} { r } { \Xi _ { T } = \sum _ { t } \left. \mu _ { t } - p _ { t } , c _ { t } \right. } \end{array}$ , and predictable bias.

The standard bandit templates are all estimator-specific instances of this game: explicit-exploration inverse-propensity scoring (IPS; bias vanishes, and $\Xi _ { T }$ records the exploration cost), implicit-exploration EXP3-IX, predictable-ofset control variates, and feedback graphs. Graph topology enters only through the observation geometry $p _ { t } ( a ) / o _ { t } ( a )$ , with $o _ { t } ( a )$ the probability that arm a is observed on the round; for undirected graphs that geometry is bounded by the independence number, the size of the largest set of mutually non-adjacent vertices [3]. Each is a choice of estimated score $\widehat { c } _ { t }$ fed to the otherwise-unchanged game (Appendix A.8).

## 7.2 An identity for the estimated RCGF

The one term this leaves implicit, the estimated per-round RCGF $\widehat { Q } _ { t }$ , is not to be bounded but has a closed form. Conditioned on the past its only randomness is the drawn arm, so $\begin{array} { r } { \mathbb E [ \widehat { Q } _ { t } \mid \mathcal F _ { t - 1 } ] = \eta _ { t } ^ { - 2 } \sum _ { a } \mu _ { t } ( a ) \log \sum _ { i } p _ { t } ( i ) e ^ { \eta _ { t } \widehat z _ { t } ^ { ( a ) } ( i ) } } \end{array}$ written entirely in the true losses and the observation geometry $\mu _ { t } ,$ , with small-scale limit the propensity-inflated variance $\begin{array} { r } { \frac { 1 } { 2 } \sum _ { a } \frac { p _ { t } ( a ) ( 1 - p _ { t } ( a ) ) } { \mu _ { t } ( a ) } c _ { t } ( a ) ^ { 2 } } \end{array}$ (Proposition A.15, Appendix A.8).

Taking expectations of (A.33) annihilates the two martingales, leaves the exploration term and predictable bias as functions of the true loss, and leaves the estimated intrinsic time $\textstyle \sum _ { t } \eta _ { t } \mathbb { E } [ \widehat { Q } _ { t } ]$ in the closed form just given; the estimated drift and transport are the same prior-transport ledger as the full-information game. Partial information therefore introduces no new obstruction: the exact minimax value is the min-max of an explicit function of the true losses and observation geometry, and solving it in closed form for general $K$ is the full-information exact-value problem itself (solved for $K = 2$ over one round by Proposition $\mathsf { A } . 6 )$ . The worst-case slice $\mu _ { t } = p _ { t }$ caps the per-round value at $K / 2 ,$ recovering the $O ( \sqrt { K T \log K } )$ exponential-weights rate as the small-scale relaxation; closing the remaining $\sqrt { \log K }$ factor remains open (Section 11).

## 7.3 Contextual wrappers and identification scores

Only the estimated score fed to the otherwise-unchanged game changes across the remaining partial-information settings: the proper contextual-bandit wrapper chooses $\widehat { c } _ { t }$ by lifting a policy class, and the testing-exponent scores choose it as a log-likelihood-ratio.

Treating a finite policy class as experts and lifting it by duplication runs the full-information game on importanceweighted losses, at a comparator complexity equal to the aggregate prior mass of a designated policy’s loss-identical copies. The resulting wrapper bound holds with exact constants, and its sharp rate closes on the estimated per-round RCGF exactly as the plain-bandit rate does (Proposition A.16, Corollary A.17).

Scoring a log-likelihood ratio instead identifies a latent model, and the game’s own quantities are the classical testing ones. The Chernof testing coeficient [17] is minus the per-observation Bellman potential, so the Gibbs equalizer is the Chernof tilt and optimizing the scale is Chernof information. The pairwise testing exponent is then the edge-restricted multi-way coincidence radius of the arm-induced laws [8], equal to the MAP error exponent for identifying the hypothesis from repeated pulls of an arm (Theorem A.18, Appendix A.8).

## 8 Thompson sampling and prior-posterior-ratio martingales

Randomizing the learner’s play exposes two martingale readings of the same game. The first is the sampling martingale of Thompson sampling: posterior sampling attains the deterministic concentration-game value in expectation, and the entire efect of sampling is a single mean-zero martingale on the realized path. The second is the prior-posterior-ratio martingale, which recasts the game’s own Bayes update as the test supermartingale of anytime-valid inference. Their step-by-step derivations are collected in Appendix D.7.

## 8.1 Thompson sampling as the same game plus a martingale

If the learner samples $I _ { t } \sim p _ { t }$ and plays that expert, the sampled composite-loss regret $\begin{array} { r } { \widehat { R } _ { T } ^ { c } ( \nu ) : = \sum _ { t = 1 } ^ { T } c _ { t } ( I _ { t } ) - \langle \nu , C _ { T } \rangle } \end{array}$ decomposes exactly as

$$
\widehat { R } _ { T } ^ { c } ( \nu ) = M _ { T } ^ { \mathrm { s a m } } + D _ { T } + B _ { T } ( \nu ) + \sum _ { t = 1 } ^ { T } \eta _ { t } Q _ { t } ( c )\tag{8.1}
$$

with $\begin{array} { r } { M _ { T } ^ { \mathrm { s a m } } : = \sum _ { t = 1 } ^ { T } ( c _ { t } ( I _ { t } ) - \langle p _ { t } , c _ { t } \rangle ) } \end{array}$ a martingale diference: Thompson sampling incurs the same deterministic concentration-game value in expectation as exponential weights, plus a mean-zero martingale whose per-realization size is a genuine path cost. Taking $\mathbb { E } [ \cdot \mid \mathcal { F } _ { 0 } ]$ and $\mathbb { E } [ M _ { T } ^ { \mathrm { s a m } } ] = 0$

$$
\mathbb { E } \big [ \widehat { R } _ { T } ^ { c } ( \nu ) \big ] = D _ { T } + B _ { T } ( \nu ) + \sum _ { t = 1 } ^ { T } \eta _ { t } Q _ { t } ( c ) = R _ { T } ^ { c } ( \nu )\tag{8.2}
$$

for every comparator and predictable loss sequence—the sampling step contributes nothing in expectation. Randomizing and then scoring in expectation therefore cancel exactly, pinning down an expected sampled concentration game (Game A.19) whose saddle point is posterior sampling (Theorem A.20, Appendix A.9). This is a value identity: $w _ { t } = p _ { t }$ attains the deterministic value in expectation as the unique equalizer at value $\Gamma / \eta + \eta \sum _ { t } q _ { t }$ , while any $w _ { t } \neq p _ { t }$ over-drives the Gibbs increment and strictly loses.

The resolution is exact in expectation: per realization the sampled regret fluctuates by the mean-zero martingale $M _ { T } ^ { \mathrm { s a m } }$ whose typical magnitude—the square root of the cumulative running variance $\begin{array} { r } { V _ { T } ^ { \mathrm { s a m } } : = \sum _ { t } \mathrm { V a r } _ { p _ { t } } ( c _ { t } ) } \end{array}$ —is a genuine cost on the realized path that the construction leaves in place. No scoring rule removes it: there is no fluctuation-free realized posterior-sampling game, and the sharpest high-probability control uses a diferent scale from the in-expectation optimum (Proposition A.21). A certificate for its size comes from Ville’s inequality applied to the exponential supermartingale on the variance clock $V _ { t } ^ { \mathrm { s a m } }$ , which bounds $M _ { t } ^ { \mathrm { s a m } }$ uniformly in time. At a constant scale the prior-posterior ratio of Section 8.2 is the companion e-process under the usual hypotheses, and inverting it gives the anytime-valid comparator sets of Section 5.5 (Appendix D.7; an exact decomposition of the slack in Ville’s inequality is obtained in concurrent work [23]).

The sampling martingale concentrates at the standard rates — a high-probability $\sqrt { 2 V _ { T } ^ { \mathrm { s a m } } \log ( 1 / \delta ) }$ , an anytime bound via Ville’s inequality, and a $O ( \sqrt { V _ { T } ^ { \mathrm { s a m } } \log \log V _ { T } ^ { \mathrm { s a m } } } )$ law of the iterated logarithm (Appendix D.7). In the worst case its cost matches the exponential-weights regret up to the iterated-logarithm factor, a genuine cost of comparable size; in the low-noise regime it vanishes in expectation, while the regret is $O ( \Gamma )$ .

## 8.2 The prior-posterior-ratio martingale

The second martingale reading turns the game’s own update into a test statistic: track how far a fixed comparator ν sits from the running posterior in relative entropy, and each Bayes step moves that distance by exactly the comparator’s per-round margin.

Writing $\begin{array} { r } { m _ { t } ( \eta _ { t } ) : = - \eta _ { t } ^ { - 1 } \log \sum _ { i } p _ { t } ( i ) e ^ { - \eta _ { t } c _ { t } ( i ) } } \end{array}$ for the round’s mix loss, an elementary expansion of $\log ( p _ { t + 1 } / p _ { t } )$ for the update $p _ { t + 1 } ( i ) \propto p _ { t } ( i ) e ^ { - \eta _ { t } c _ { t } ( i ) }$ (Appendix D.7) gives the transport identity

$$
\mathrm { K L } ( \nu \| p _ { t + 1 } ) - \mathrm { K L } ( \nu \| p _ { t } ) = \eta _ { t } ( \langle \nu , c _ { t } \rangle - m _ { t } ( \eta _ { t } ) )\tag{8.3}
$$

for every fixed comparator ν. Read the right-hand side as the comparator’s margin: the relative entropy from ν to the posterior falls by η<sub>t</sub> $( m _ { t } - \langle \nu , c _ { t } \rangle )$ on every round the comparator’s loss beats the mix. Summing over the run, $\begin{array} { r } { \mathrm { K L } ( \nu \| \pi ) - \mathrm { K L } ( \nu \| p _ { T + 1 } ) = \sum _ { t } \eta _ { t } ( m _ { t } - \langle \nu , c _ { t } \rangle ) } \end{array}$ , and nonnegativity of the terminal relative entropy caps the comparator’s total accumulated margin at $\mathrm { K L } ( \nu \| \pi ) \leq \Gamma$ — the same information budget that identifies the comparator elsewhere in the game.

Sampling turns this deterministic ledger into a test. Under Thompson sampling with $I _ { t } \sim p _ { t }$ , the log-weight of the sampled expert moves by log $p _ { t + 1 } ( I _ { t } ) - \log p _ { t } ( I _ { t } ) = - \eta _ { t } ( c _ { t } ( I _ { t } ) - m _ { t } )$ , whose conditional mean is the nonpositive drift $- \eta _ { t } \delta _ { t } ( c )$ ; the excess over that drift is the prior-posterior-ratio (PPR) martingale, connecting the concentration game to anytime-valid inference [56].

This martingale is central to a dual, testing-facing reading of the game. The predictable play $p _ { t } = G _ { \eta _ { t } } ( S _ { t - 1 } )$ is the dual variable in a Lagrangian–martingale duality, the game’s min<sub>p</sub> max<sub>z</sub> matching the min-over-predictable / max-overprocess shape of safe anytime-valid testing [56, 63]. The prior-posterior-ratio supermartingale is the process the learner bets against, and the Gibbs play the equalizing bet that fixes the round’s loss whichever direction nature moves. In the time-uniform regime the concentration-game identity itself reads as the capital process of a bettor against the forecasts — the Skeptic of game-theoretic probability. Such a guarantee must control the maximum over the process instead of its endpoint, and the ledger quantifies that diference exactly: the two readings of the same supermartingale difer by the drawdown of the comparator’s transport trajectory (Section 5.5).

## 9 Boosting and the pressure game

Boosting is exactly about the pressure game on signed margins.

Let $g _ { t } ( i ) ~ = ~ y _ { i } h _ { t } ( x _ { i } ) ~ \in ~ [ - 1 , 1 ]$ be the signed margin score of weak learner $h _ { t }$ on example $i ,$ and $M _ { T } ( i ) ~ =$ $\textstyle \sum _ { t = 1 } ^ { T } \alpha _ { t } g _ { t } ( i )$ the cumulative weighted margin. The roles reverse relative to the expert game: the booster plays the learner’s role, with the $N$ training examples as its actions and the uniform weighting $u _ { N }$ as its prior; the weak learner supplies nature’s move $g _ { t } ;$ and a comparator is a reweighting of the examples. The ε-quantile margin below corresponds to the comparator uniform on the hardest $\lceil \varepsilon N \rceil$ examples.

## 9.1 From regret to the exponential-loss terminal game

Running the concentration game on example scores $\ell _ { t } ( i ) = ( 1 + y _ { i } h _ { t } ( x _ { i } ) ) / 2$ with round-t edge $\gamma _ { t } : = \frac { 1 } { 2 } \left. p _ { t } , g _ { t } \right.$ , the standard boosting conventions give, for any posterior $\begin{array} { r } { \mathrm { ~ , ~ } R _ { T } ^ { \ell } ( \nu ) = \frac { T } { 2 } ( 2 \bar { \gamma } _ { T } - \langle \nu , m \rangle ) } \end{array}$ , i.e. $\langle \nu , m \rangle = 2 \bar { \gamma } _ { T } - ( 2 / T ) R _ { T } ^ { \ell } ( \nu )$ with $\bar { \gamma } _ { T }$ the average edge and $m _ { i } = T ^ { - 1 } \textstyle \sum _ { t } y _ { i } h _ { t } ( x _ { i } )$ the unit-weight average margin $( M _ { T }$ is its α<sub>t</sub>-weighted, unaveraged counterpart). Hence any regret bound $R _ { T } ^ { \ell } ( \nu ) \leq U _ { T } ( \nu )$ implies $\langle \nu , m \rangle \geq 2 \bar { \gamma } _ { T } - 2 U _ { T } ( \nu ) / T$

In particular the ε-quantile margin is controlled by comparator complexity $\log ( 1 / \varepsilon )$ and the intrinsic time of the example-loss sequence: a comparator uniform on the $\lceil \varepsilon N \rceil$ examples of smallest margin has $\mathrm { K L } ( \nu \| u _ { N } ) \leq \log ( 1 / \varepsilon )$ The envelope (4.11) at $\Gamma = \log ( 1 / \varepsilon )$ and $C = 1 / \sqrt { 2 }$ then lets that margin fall short of twice the average edge by at most $2 \sqrt { \log ( 1 / \varepsilon ) / T }$ plus additive $O ( \log ( 1 / \varepsilon ) / T )$ ) edge terms in the worst case, and by less whenever the realized intrinsic time is smaller (Appendix A.10.1).

The exponential training loss also admits a direct mixed-coincidence identity, exhibiting it as an exact terminal game. For $\begin{array} { r } { L _ { T } ^ { \mathrm { e x p } } : = N ^ { - 1 } \sum _ { i = 1 } ^ { N } { e ^ { - M _ { T } ( i ) } } , \mathrm { i f } p _ { t + 1 } ( i ) \propto p _ { t } ( i ) e ^ { - \alpha _ { t } g _ { t } ( i ) } } \end{array}$ with $p _ { 1 } = u _ { N }$ , then for every comparator $\nu \in \Delta ( [ N ] )$ ,

$$
- \log L _ { T } ^ { \mathrm { e x p } } + \mathrm { K L } ( \nu \| p _ { T + 1 } ) = \mathrm { K L } ( \nu \| u _ { N } ) + \langle \nu , M _ { T } \rangle\tag{9.1}
$$

Equivalently, applying the Donsker–Varadhan variational form $\mathrm { t o } - M _ { T } $

$$
- \log L _ { T } ^ { \mathrm { e x p } } = \operatorname* { i n f } _ { \nu \in \Delta ( [ N ] ) } \big \{ \langle \nu , M _ { T } \rangle + \mathrm { K L } ( \nu \| u _ { N } ) \big \} ,
$$

the infimum attained at $\nu = p _ { T + 1 }$ , where $\mathrm { K L } ( \nu | | p _ { T + 1 } ) = 0 \mathrm { i n } ( 9 . 1 ) ,$ . Training error, margin tails, and quantile margins are therefore controlled by the same terminal concentration functional.

For the margin-tail bound: for every $\theta \in \mathbb { R }$ , the fraction of examples with normalized margin at most θ satisfies $\begin{array} { r }  E _ { T } ( \theta ) \leq \exp ( \theta \bar { A } _ { T } ) \cdot L _ { T } ^ { \mathrm { e x p } } = \exp \bigl ( \theta A _ { T } - \operatorname* { i n f } _ { \nu } \{ \langle \nu , M _ { T } \rangle + \mathrm { K L } ( \nu \| u _ { N } ) \bar { \} \} \bigr ) } \end{array}$ , where $A _ { T } = \textstyle \sum _ { t } \alpha _ { t }$ (by the Markov/Chernof step $\mathbf { 1 } \{ M _ { T } ( i ) \leq \theta A _ { T } \} \leq e ^ { - M _ { T } ( i ) + \theta A _ { T } } ) .$

## 9.2 Pressure-target boosting and AdaBoost

If $\alpha _ { t }$ is chosen by the local pressure rule $\begin{array} { r } { - \alpha _ { t } ^ { - 1 } \log { \sum _ { i } p _ { t } ( i ) e ^ { - \alpha _ { t } g _ { t } ( i ) } } = a _ { t } } \end{array}$ , then

$$
- \log L _ { T } ^ { \mathrm { e x p } } = \sum _ { t = 1 } ^ { T } \alpha _ { t } a _ { t } , \qquad \sum _ { t = 1 } ^ { T } \alpha _ { t } ( \langle \nu , g _ { t } \rangle - a _ { t } ) = \mathrm { K L } ( \nu \| p _ { T + 1 } ) - \mathrm { K L } ( \nu \| u _ { N } )\tag{9.2}
$$

The pressure target is an exact exponential-loss descent statement.

For binary weak hypotheses $g _ { t } ( i ) \in \{ - 1 , + 1 \}$ the coeficient minimizing the one-step normalizer log $\begin{array} { r } { \sum _ { i } p _ { t } ( i ) e ^ { - \alpha g _ { t } ( i ) } } \end{array}$ — the unrescaled CGF at scale α — is the classical AdaBoost weight $\begin{array} { r } { \alpha _ { t } ^ { \star } = \frac { 1 } { 2 } \log \frac { 1 - \varepsilon _ { t } } { \varepsilon _ { \star } } ( \varepsilon _ { t } = \operatorname* { P r } _ { i \sim p _ { t } } \{ g _ { t } ( i ) = - 1 \} ) } \end{array}$ the binary special case of the pressure game. It makes the pressure identity evaluate the exact exponential-loss ledger − log $\begin{array} { r } { L _ { T } ^ { \mathrm { e x p } } = \sum _ { t } - \frac { 1 } { 2 } \log ( 1 - 4 \gamma _ { t } ^ { 2 } ) } \end{array}$ , the Gaussian-channel information of the per-round mean margin. The convexity relaxation $- \log ( 1 - x ) \geq x$ then recovers the classical training-error rate ${ \widehat { \mathrm { e r r } } } _ { T } \leq e ^ { - 2 \gamma ^ { 2 } T }$ as a one-line corollary, sharper because it holds for adaptive non-uniform edges (Proposition A.22, Appendix A.10).

The exact ledger (A.48) is strictly stronger than $e ^ { - 2 \gamma ^ { 2 } T }$ : it requires no uniform lower bound on the edge, and the per-round $\mathrm { g a p } - { \textstyle \frac { 1 } { 2 } } \log ( 1 - 4 \gamma _ { t } ^ { 2 } ) - 2 \gamma _ { t } ^ { 2 } = 4 \gamma _ { t } ^ { 4 } + O ( \gamma _ { t } ^ { 6 } )$ is exactly what the classical bound discards.

A one-sided shortfall statistic reweights toward the examples still below target. Replacing the signed score $g _ { t } ( i )$ by $( a _ { t } - g _ { t } ( i ) )$ <sub>+</sub> produces sparse example weights: examples already above target receive zero cost. This is the boosting analogue of the positive-part suficient-statistic reduction—a change of statistic, leaving the game itself untouched.

## 10 Related work

The minimax view of online learning as a repeated game is classical [1, 15, 64, 65], and the multiplicative-weights update on the entropic side of it is itself a meta-algorithm whose disparate instances share a single exponential potential [5]. The drifting-games framework [45, 60] defines a potential, lets the adversary’s feasible set depend on it, and computes the value by backward induction, reformulating Hedge, bandits, and boosting in minimax terms. The fixed-scale concentration game is the entropic version of that program: the state potential is the log-partition function and the one-step Bellman increment is the centered RCGF $\eta _ { t } Q _ { t } .$ , where that program chooses a quadratic penalty externally and the variance and range games are its relaxations. The efective state is $( p _ { t } , V _ { t - 1 } ) _ { : }$ , and Theorem 4.3 solves the backward induction in closed form without numerical recursion. The algorithmic ingredients treated here — entropic FTRL regularizers, relative-entropy-anchored mirror maps, the second-order rate schedule, drifting-game potentials — are downstream consequences of that constrained min-max value, where a design-first account would install them as upstream choices. Under the RCGF choice the variable-temperature drift has a clean exponential-family relative-entropy interpretation, where the dual-feasibility / mirror-map choice yields no immediate Bregman expression, which leaves the RCGF as the upstream quantity inside the entropic class. For repeated play, the equilibrium consequences are the standard external- and swap-regret ones [11, 33], derived here in the present notation. The same no-regret dynamics drive practical equilibrium computation: counterfactual regret minimization solves large imperfect-information games through time-averaged play [12], and optimistic dynamics additionally secure last-iterate convergence in zero-sum games [22].

The framework closest in spirit is the game-theoretic probability of Shafer and Vovk [64, 65], which casts the laws of large numbers, the central limit theorem, the law of the iterated logarithm, and large deviations as outcomes of a perfect-information game built around a nonnegative-(super)martingale capital process. The distinction is in the primitive: there a Skeptic bets against a forecaster and its wealth multiplies unless the asserted event holds [63], and that line solves for a capital-maximizing bet. Here the primitive is the comparator-class regret value of an aggregating learner, whose value function is the log-partition and whose solution is a Gibbs equalizing equilibrium. The two programs are dual, meeting where the prior-posterior-ratio martingale of Section 8.2 is a Skeptic-style capital process: in the time-uniform regime the log-partition value and the testing supermartingale describe the same process.

The paper sits at the intersection of adaptive online learning, concentration of measure, and robust Bayes. For adaptive regret it connects with AdaHedge [25], NormalHedge [16], Squint [42], coin-betting and parameter-free methods [21, 50], data-dependent adaptive-geometry methods [27], and the sequential-complexity program that measures adversarial dificulty by a martingale-indexed capacity [53, 54]. For concentration it connects with PAC-Bayes and the thermodynamics of learning [4, 14] and with the time-uniform / e-process viewpoint [36, 38, 56, 70], whose safe anytime-valid constructions the prior-posterior-ratio identity makes explicit at the level of the game. Closest of all is the line that converts regret into concentration directly: the regret of universal portfolio yields concentration inequalities and confidence sequences tight to the leading constant [49], and any e-process whose betting regret is sublinear is log-optimal for a composite alternative [71]. That conversion is one-directional — a regret guarantee yields a tail bound, and the conversion factor between them remains implicit — and the decomposition here states that factor: the comparator transport $B _ { T } ( \nu )$ is the tail statement’s contribution, and the per-round loss $\eta _ { t } Q _ { t }$ the sequence’s own. The log-optimality those results establish is the extremal property the reverse information projection selects [43], which is the terminal transport of (1.1) read as a projection onto the comparator class. The same conversion runs outside the tail-bound setting, in generalization games [44] and in sequential likelihood mixing [40]. Each constructs a game to prove a statistical statement; here the statistical statements and the regret are two readings of one value identity, so the construction step only names a comparator class and a per-round budget, and manufactures no bound. On the adaptive-regret side, the equality-form analyses of adaptive FTRL [47] decompose regret into a comparator-regularizer term and a per-round stability term, the two roles played here by $B _ { T } ( \nu )$ and $\eta _ { t } Q _ { t }$ <sub>t</sub>. The scale enters there as a regularizer the analyst chooses and here as nature’s information budget, which is why the retempering drift surfaces as a third term with a closed form instead of being absorbed into the schedule. For bandits it connects with EXP3, EXP3-IX, Tsallis-INF, and feedback-graph algorithms [3, 6, 41, 57, 73]; for Thompson sampling, with the information-directed tradition [56, 58, 59]; and for stochastic low-noise / fast rates, with mixability-based theory [67]. The classical tail inequalities that the terminal payof accounts for exactly—Markov, Chebyshev, and Chernof—have themselves recently been sharpened by randomization and exchangeability into never-worse, typically strictly tighter forms [55]. In the time-uniform regime the analogous sharpening is exact: in concurrent and independent work, [23] decompose the slack in Ville’s inequality for a nonnegative supermartingale into explicit overshoot, predictable-loss, and residual-survival terms, the anytime-valid counterpart to the terminal accounting isolated here. For dynamic / shifting comparators, the foundational work is [35, 37]; the comparator-path identity (5.1) fits inside that program as the pathwise statement that those papers’ inequality forms approximate.

The parallel Fenchel-game program for convex optimization [69] runs the other way, converting a static optimization into a game where the concentration game converts a sequential inferential procedure into one. A further parallel is the natural-gradient variational-Bayes program [39], which derives a broad range of learning algorithms from a single Bayesian regularized update primitive, where here the corresponding primitive is instead a constrained min-max game value. For boosting, the main classical reference is [31, 61]; the pressure-target perspective in Section 9 aligns with the smoothed-boosting line [62] and the entropy-regularized boosting tradition.

A second adjacent line is the literature on budgeted adversaries, where an exogenous budget — on switches, cumulative loss variation, or adversarial strength relative to a stochastic baseline — constrains nature’s per-round move and regret scales in that budget in place of $T$ alone [2]. Here the round-budget allocation is instead derived from the RCGF/logpartition Bellman structure: $\textstyle \sum _ { t } Q _ { t } \leq V$ is a suficient statistic of nature’s feasibility under a fixed potential, and front-loading is forced by the nonincreasing scale schedule. That literature’s budget is the cumulative intrinsic time $V _ { T } ( c )$ and its regret bounds are the variance relaxation of the present identity, with the same $\sqrt { \Gamma V _ { T } }$ leading constant up to relaxation slack.

## 11 Discussion

With the game made explicit, the online-learning and concentration primitives treated here share one structure: within the relative-entropy setting each is a constrained min-max value with the same Bellman equalizer, the per-round constraint the centered RCGF and the comparator regularizer relative entropy to a prior. The variable-temperature decomposition (4.8) makes this operational, both players’ strategies read of term by term, and repeated play yields an exact informationtheoretic ledger of self-play in place of the usual quadratic-variation surrogate. The variance and bounded-range quantities of standard regret theory enter only as small-η relaxations of that identity, and the derivation does not run in the reverse direction. The gain is conceptual, and no new algorithm comes with it; the limitation is the restriction to entropic geometry, the complementary non-entropic case being the province of the Fenchel-game program [69].

<table><tr><td>Game</td><td>Scale set by</td><td>Nature&#x27;s constraint Move order</td><td></td><td></td><td>State score Terminal payoff</td></tr><tr><td>Game 1.1 (base game)</td><td>schedule</td><td>per-round cap qt</td><td>learner first</td><td>true</td><td> $L _ { \Gamma } ( S _ { T } )$ </td></tr><tr><td>Game 3.1 (fixed scale)</td><td>frozen</td><td>per-round cap qt</td><td>learner first</td><td>true</td><td> $L _ { \Gamma } ( S _ { T } )$ </td></tr><tr><td>Game 4.1 (self-tuned)</td><td>learner</td><td>cumulative V</td><td>learner first</td><td>true</td><td> $L _ { \Gamma } ( S _ { T } )$ </td></tr><tr><td>Game A.8 (pressure)</td><td></td><td>learner, via bt none (made active)</td><td>nature first</td><td>true</td><td> $L _ { \Gamma } ( S _ { T } )$ </td></tr><tr><td>Game 7.1 (bandit)</td><td>learner</td><td>estimated-RCGF cap</td><td>learner first</td><td>estimated</td><td>sampled-regret sup</td></tr><tr><td>Game A.19 (expected sampled)</td><td>) frozen</td><td>per-round cap qt</td><td>learner first</td><td>true</td><td>expected sampled regret</td></tr></table>

Table 2: Every boxed variant is Game 1.1 with at most three parts changed. Columns are the parts that vary across the variants — who sets the measurement scale, what constrains nature’s move, the order of moves within a round, which score drives the state, and the terminal payof; the base-game row gives Game 1.1’s own values.

## 11.1 Common structure across settings; the scaling-time question

Every setting treated above instantiates the same abstract game, difering only in the observation structure (full, bandit, feedback graph, contextual), the action space (finite, continuous, policy class), and the suficient statistic (raw loss, residual, estimated, signed margin). The RCGF constraint always takes the same form $Q _ { t } ( \cdot ) \leq \beta _ { t } ;$ what changes is the loss sequence the RCGF is evaluated on, and how that sequence relates to the true losses. The same economy holds at the level of the protocol, where Table 2 compares the boxed games on the parts that vary.

The historical scaling-time question [30] asks for quantile regret controlled by an internal variance-like process in place of horizon and expert count alone. We settle the fixed-budget PAC-Bayes version for the exact cumulant intrinsic time, with a logarithmic meta-controller giving simultaneous quantile guarantees. Simultaneous adaptivity in both comparator complexity and intrinsic time, for a single copy and up to only mild lower-order terms, remains open: NormalHedge together with contextual-bandit model-selection lower bounds [30, 46] show that self-variance, quantiles, and parameterfreeness cannot all be demanded at once. In the face of these impossibility results, the concentration game makes the tradeofs visible.

## 11.2 The identities in neighboring fields

Several of the framework’s identities read cleanly in the languages of neighboring areas: the retempering drift as a summation by parts of the log-partition’s η-derivative, whose Itô form turns the committed and reactive games into dual stochastic-control problems on the information manifold. The Bellman potential as a negative free energy at inverse temperature η with external field S. In the bandit reading, the observation-noise terms of (A.33) are the axis along which the algorithms trade of against one another. EXP3-IX makes the estimation bias small and the martingale term slightly larger; Tsallis-INF makes the entropic information small and the sample-dependent drift larger (Appendix A.11.1).

The one-round $( T = 1 )$ specialization of the fixed-scale game is the classical redundancy–capacity saddle point $[ 1 8 , 3 4 , 4 8 ] ;$ : a channel’s input distribution is nature’s comparator prior, the log-partition value $W _ { \eta }$ is the mixture codelength, and the Gibbs equalizer $\rho _ { \eta , S } = \nabla W _ { \eta } ( S )$ is the capacity-achieving input, with capacity the saddle-point value ${ \mathsf { C } } =$ min<sub>q</sub> max<sub>θ</sub> $\begin{array} { r } { \mathrm { K L } ( P _ { \theta } \| q ) = \operatorname* { m a x } _ { p _ { 0 } } I ( p _ { 0 } ) } \end{array}$ . When an external process pins the input prior, leaving no freedom to equalize, the shortfall from capacity is the same terminal relative-entropy transport the horizon ledger isolates in $B _ { T } ( \nu )$ , now read across channel inputs. It is computable term by term as a Bregman divergence generated by the log-partition potential (Appendix A.11.1).

An algorithm gains from better-behaved losses exactly the higher-cumulant tail $\kappa _ { \geq 3 }$ that the variance and range proxies discard, and the relaxation remainder — the heavy-tailed moves the game admits that a proxy forbids — has the closed form of Proposition A.23. A change of suficient statistic $z \mapsto \phi ( z )$ leaves every identity intact and shrinks that remainder insofar as it shrinks $\{ \kappa _ { n } ( \phi ( z ) ) \} _ { n \geq 3 }$ . Bounded-influence transforms and optimism are both strategies to reduce the residual cumulants

The Bellman equalizer is also the maximum-entropy distribution consistent with the exposed suficient statistic (Proposition 2.3), and the game-theoretic reading derives both properties from one minimax argument on the RCGFconstrained feasible set. The coincidence holds well beyond the entropic case: for any strictly convex Legendre-type potential W with conjugate entropy −Ψ, ∇W(S) is simultaneously the Bellman equalizer, since the budget-sphere increment is constant and nature is indiferent across directions, and the maximum-(−Ψ)-entropy distribution. Both therefore hold across the regular exponential/Bregman family. The Shore–Johnson axioms then single out the centered RCGF member, pinning the inference rule uniquely; deriving that invariance from the equalizer postulate alone is beyond the present scope.

## 11.3 Exact values and rates

These readings sharpen what remains open, and one quantity is the residual behind most of it: the exact value. The one round is solved on two slices, each strictly below the Bellman bound — two experts at every comparator budget (Proposition $_ { \mathrm { A } . 6 ) }$ , and every $K$ at a saturated one (Proposition A.5) — and their common refinement is missing. So is any $K$ over $T \geq 2$ rounds under per-round caps, beyond the edge terms $O ( \Gamma + Q _ { T } ^ { * } )$ . Under a cumulative budget the multi-round question closes at constant scale from the finite horizon $T _ { 0 }$ on (Proposition 4.2), and varying the scale lowers the value by under a tenth of a percent at the budgets computed (Appendix $\mathrm { A . 4 . 4 } $ . The per-round-capped game remains open, its budget $V = T \beta$ growing with the horizon. Even at $K = 2$ with a uniform prior, where the terminal payof is an exact support function, nature’s reachable set solves a state-dependent recursion whose optimum over-tilts the Gibbs iterate, and no elementary closed form survives past $T = 1$ (Appendix A.11.2). The same quantity settles several others. The partial-information value is the same function with the estimated per-round RCGF of Proposition A.15 in place of the true one. The pressure and second-order games difer at the horizon only through which terminal state each lets nature reach, their drifts agreeing by Proposition 4.7. But the loss couples the rounds through the path-dependent $p _ { t } ,$ so the optimal schedule is again this control problem and a matching lower bound for the surrogate is unknown.

Simultaneous adaptation is separate. A single fixed-budget run is valid for every comparator simpler than its budget but not adaptive to it, losing a factor $\sqrt { \Gamma _ { \mathrm { m a x } } / \Gamma }$ against true complexity Γ. A dyadic meta-controller restores adaptivity at an additive O(log log log $K )$ plus the controller’s own intrinsic-time term, and a single copy attaining the ideal $\sqrt { \Gamma V _ { T } }$ with no additive term for all complexities at once is ruled out by the model-selection lower bound [30, 46]. Enlarging the class to expert–scale pairs under a log-scale prior over $\eta -$ the continuous-mixture device of [42] and the parameter-free family [50] — does achieve simultaneous adaptivity at a strictly-lower-order O(log log $V _ { T } )$ overhead, leaving only the leading constant. The exact value above also decides whether the mixture’s per-round Jensen surplus preserves the envelope constant $2 \sqrt { 2 }$

Correlated-equilibrium rates rest on a diferent quantity. An improved CCE rate for a structured game is equivalent to worst-player last-iterate stabilization, max<sub>k</sub> $V _ { T } ^ { ( k ) } = o ( T )$ ; the stabilizing sub-case is settled for potential, congestion, and polymatrix games under Gibbs-equalizer dynamics (Theorem $\mathrm { A } . 1 3 )$ , where the interior-equilibrium sub-case fails for plain Gibbs and needs an optimistic schedule. An exact growth-rate law for $V _ { T } ^ { ( k ) }$ , and a matching lower bound on the non-stabilizing polymatrix class, remain open.

# Appendices to “The concentration game: Bayesian updating, regret, and information”

## A Further results

Results whose full detail the main text relegates to the appendix are stated and discussed here, grouped by source section;   
the proofs of the numbered statements, the body’s and this appendix’s alike, are collected in Appendix D.

## A.1 Further results from Section 2

## A.1.1 The exponential-family parameterization costs no generality

Fix $\eta _ { t } > 0$ and a strictly positive prior. Adding b1 to a cumulative score multiplies every numerator and the denominator alike by $e ^ { - \eta _ { t } b }$ , so two scores give the same $p _ { t }$ in (2.1) exactly when they difer by a constant vector; and every interior $p _ { t } -$ positive weight on every action — is realized, by $C _ { t - 1 } ( i ) = - \eta _ { t } ^ { - 1 } \log \bigl ( p _ { t } ( i ) / \pi ( i ) \bigr )$ . Hence (2.1) is a bijection from $\mathbb { R } ^ { K } / \mathbb { R } \mathbf { 1 }$ onto the interior of $\Delta ( [ K ] )$ : it reparameterizes arbitrary play instead of restricting it. Read in the centered coordinates of Section 2 the same inverse says that the state is the round’s posterior-to-prior log-ratio at the round’s own scale, $S _ { t - 1 } = \eta _ { t } ^ { - 1 } \log ( p _ { t } / \pi )$ up to an additive constant, and doubling the scale exactly halves the state needed for the same tilt. The per-round move is its increment: at a constant scale $z _ { t } = \eta ^ { - 1 } \log ( p _ { t + 1 } / p _ { t } )$ , the log-ratio of consecutive posteriors, again up to a constant.

## A.1.2 The exponential family traced by the inverse temperature

The family of posteriors parameterized by inverse temperature forms a one-dimensional exponential family

$$
\mathcal { E } ( \pi , C _ { t - 1 } ) = \left\{ q _ { \eta } ( i ) = \frac { \pi ( i ) e ^ { - \eta C _ { t - 1 } ( i ) } } { \sum _ { j } \pi ( j ) e ^ { - \eta C _ { t - 1 } ( j ) } } : \eta > 0 \right\}\tag{A.1}
$$

with natural parameter $\eta ,$ suficient statistic $C _ { t - 1 }$ , log-partition $\begin{array} { r } { \Lambda ( \eta ) = \log \sum _ { i } \pi ( i ) e ^ { - \eta C _ { t - 1 } ( i ) } = - \eta A _ { t - 1 } ( \eta ) } \end{array}$ , and Fisher information $\Lambda ^ { \prime \prime } ( \eta ) = \operatorname { V a r } _ { q _ { \eta } } ( C _ { t - 1 } )$ . The intrinsic-time density $Q _ { t } ( c )$ is the one-step analogue of that Fisher information—the curvature cost of incorporating $c _ { t }$ at the current scale—whose running accumulation is the pathwise intrinsic-time clock adaptive Bayes tracks [7]. In the SafeBayes tradition $[ 3 2 ] , p _ { t }$ is the maximum-entropy distribution typical of the observed average loss, and $Q _ { t } ( c )$ measures how much the new observation forces the learner away from the prior in relative entropy.

## A.1.3 The moment expansion behind the RCGF

The choice of the RCGF as the budgeted quantity descends from the moment expansion of the MGF:

$$
\mathbb { E } _ { p } [ e ^ { \eta z } ] = \sum _ { n \ge 0 } \frac { \eta ^ { n } } { n ! } \mathbb { E } _ { p } [ z ^ { n } ] = \sum _ { n \ge 0 } \frac { \eta ^ { n } } { n ! } \sum _ { i } p ( i ) z ( i ) ^ { n }\tag{A.2}
$$

where each moment $\begin{array} { r } { \mathbb { E } _ { p } [ z ^ { n } ] = \sum _ { i } p ( i ) z ( i ) ^ { n } } \end{array}$ is the coincidence rate of the coincidence-budget reading. The centered log-MGF $\eta ^ { 2 } Q _ { \eta } ( p , z )$ packages these rates of all orders into a single scale-aware suficient statistic, so the cap privileges no moment order. Variance $( n = 2 )$ and range $( n = \infty )$ are individual moment-bounds obtained by truncating (A.2), which is why the variance and range games appear as derived relaxations of the primitive (Section 3.3).

The centered RCGF admits three readings, each at its own rescaling. As a coincidence budget, $\eta ^ { 2 } Q _ { \eta } ( p , z ) = \log \mathbb { E } _ { p } [ e ^ { \eta z } ]$ is the logged generating function of the moments $\mathbb { E } _ { p } [ z ^ { n } ]$ , summed at scale η; each moment is the z-weighted rate at which n draws coincide on a symbol of $p \left[ 8 \right]$ . As a min-max value, $\eta Q _ { \eta } ( p , z ) + \eta ^ { - 1 } \mathrm { K L } ( \nu \| p )$ is the robust-Bayes value against nature’s tilt z (Propositions 2.3, 2.2): the centered RCGF is the part of the round’s value coming from the constraint on nature’s move, the relative-entropy term the part from the comparator class. And as a relative-entropy increment, the forward identity $\eta ^ { 2 } Q _ { \eta } ( p , z ) = \mathrm { K L } ( p \| p ^ { + } )$ makes $\eta Q _ { \eta }$ the round’s prior-to-posterior information gain divided by η, with cumulative form $\begin{array} { r } { \sum _ { t } \eta _ { t } Q _ { t } = \sum _ { t } \mathrm { K L } ( p _ { t } \| p _ { t + 1 } ) / \eta _ { t } } \end{array}$ . These are faces of one functional, each useful in a diferent proof.

## A.1.4 Influence of the centered CGF under a contaminated play

The tilted-variance representation of Proposition 2.4 also controls how the primitive responds to a perturbation of the mixed action, with the centering recomputed at the perturbed action.

Corollary A.1 (Influence of the centered CGF). Fix $c \in \mathbb { R } ^ { K }$ , a strictly positive $p \in \Delta ( [ K ] )$ , and $\eta > 0 ;$ write $z : = c - \langle p , c \rangle$ componentwise, and let $p _ { 1 } : = p _ { 1 } ^ { ( \eta , z ) }$ be the endpoint of the tilt path of Proposition 2.4. For $j \in [ K ]$ let $\delta _ { j }$ be the point mass on $j$ and $p _ { \epsilon } : = ( 1 - \epsilon ) p + \epsilon \delta _ { j }$ . Then

$$
\frac { d } { d \epsilon } \Big | _ { \epsilon = 0 ^ { + } } \eta ^ { 2 } Q _ { \eta } \big ( p _ { \epsilon } , c - \langle p _ { \epsilon } , c \rangle \big ) = \xi _ { \eta } ( j ) : = \frac { p _ { 1 } ( j ) } { p ( j ) } - 1 - \eta z ( j )\tag{A.3}
$$

with $\mathbb { E } _ { j \sim p } [ \xi _ { \eta } ] = 0$ and

$$
\operatorname { V a r } _ { j \sim p } ( \xi _ { \eta } ) = \chi ^ { 2 } ( p _ { 1 } \| p ) - 2 \eta \left. p _ { 1 } , z \right. + \eta ^ { 2 } \operatorname { V a r } _ { p } ( z ) , \qquad \chi ^ { 2 } ( p _ { 1 } \| p ) : = \sum _ { j } p ( j ) \Big ( \frac { p _ { 1 } ( j ) } { p ( j ) } - 1 \Big ) ^ { 2 }\tag{A.4}
$$

$I f z ( i ) \in [ a , b ]$ for all i, then $- ( 1 + \eta b ) \le \xi _ { \eta } ( j ) \le e ^ { \eta b } - 1 - \eta a$ for every j.

The body reads of the two-sided bracket. The influence is bounded below by the linear term alone and above by the exponential of the largest centered score, so it is linear in the depth of the score range and exponential in its height, and the number of experts does not enter.

## A.2 Further results from Section 3

## A.2.1 Why the value-to-go is a certificate

The value-to-go $\begin{array} { r } { U _ { t } ( S ) = \Gamma / \eta + W _ { \eta } ( S ) + \eta \sum _ { s > t } q _ { s } } \end{array}$ of (3.2) is an admissible Bellman relaxation: writing $\mathrm { V a l } _ { t } ^ { \eta , \Gamma } ( S )$ for the value of Game 3.1 from state $S$ with rounds $t , \ldots , T$ still to play, it satisfies $U _ { t } ( S ) \geq \mathrm { V a l } _ { t } ^ { \eta , \Gamma } ( S )$ . At the equalizer play the round decrement $U _ { t } ( S ) - U _ { t + 1 } ( S + z ) = \eta q _ { t } - \eta Q _ { \eta } ( p , z )$ is nonnegative on the feasible set and vanishes at nature’s budget-saturating reply, so the Bayes normalizer $W _ { \eta }$ is the Bellman potential of the constrained game. Those two properties make $U _ { t }$ a certificate. At the horizon $U _ { T + 1 } ( S ) = \Gamma / \eta + W _ { \eta } ( S )$ already dominates the terminal payof, since (2.5) writes $L _ { \Gamma } ( S )$ as an infimum over scales of which that quantity is one member; and a value-to-go that starts above the payof and never rises along a feasible round stays above the game value at every earlier round. Verifying the relaxation therefore requires only one round’s inequality, which every round inherits; computing the value outright would require the recursion.

The bound $\mathrm { V a l } _ { t } ^ { \eta , \Gamma } ( S ) \le U _ { t } ( S )$ becomes an equality only when, in addition, the fixed scale η coincides with the dual optimizer of $L _ { \Gamma }$ at the realized terminal state — a condition the learner generally cannot guarantee in advance. Identity (2.5) locates that gap: at the horizon the relaxation yields $\Gamma / \eta + W _ { \eta } ( S _ { T } )$ , where the game yields $L _ { \Gamma } ( S _ { T } ) =$ $\mathrm { i n f } _ { \eta ^ { \prime } > 0 } \{ \Gamma / \eta ^ { \prime } + W _ { \eta ^ { \prime } } ( S _ { T } ) \}$ , so a scale committed before $S _ { T }$ is seen loses exactly how far that infimum sits below the committed scale’s value.

## A.2.2 The two proxies do not refine each other

At $p = ( 1 / 2 , 1 / 2 )$ and $c = ( 0 , 1 )$ the variance proxy $( e - 2 ) \mathrm { V a r } _ { p } ( c ) \approx 0 .$ 18 exceeds the range proxy rang $\mathrm { { s } ^ { 2 } / 8 = 0 . 1 2 5 ; }$ at concentrated p, where $\mathrm { V a r } _ { p } ( c )$ collapses but range(c) remains 1, the order reverses. Viewed as constraint sets on nature, $\{ ( e - 2 ) \mathrm { V a r } _ { p _ { t } } ( c ) \leq \bar { \beta } \}$ and $\{ { \mathrm { r a n g e } } ( c ) ^ { 2 } \leq 8 \beta \}$ are both contained in $\{ Q _ { \eta _ { t } } ( p _ { t } , c ) \leq \beta \}$ by (3.7), and neither contains the other.

At fixed rate η and under a bounded range $c _ { t } ( i ) \in [ a _ { t } , b _ { t } ]$ , the classical Hedge bound becomes an exact identity once the relaxation remainder is retained:

$$
R _ { T } ^ { c } ( \nu ) = \frac { \mathrm { K L } ( \nu | | \pi ) } { \eta } + \frac { \eta } { 8 } \sum _ { t = 1 } ^ { T } ( b _ { t } - a _ { t } ) ^ { 2 } - \Delta _ { T } ^ { \mathrm { c l a s s } } ( \nu , \eta )\tag{A.5}
$$

where $\begin{array} { r } { \Delta _ { T } ^ { \mathrm { c l a s s } } ( \nu , \eta ) = \eta ^ { - 1 } \mathrm { K L } ( \nu \| p _ { T + 1 } ) + \eta \sum _ { t = 1 } ^ { T } \big [ ( b _ { t } - a _ { t } ) ^ { 2 } / 8 - Q _ { t } ( c ) \big ] \geq 0 } \end{array}$ . The first piece is terminal posterior mismatch; the second is the cumulative gap between the Hoefding proxy and the true RCGF increment.

## A.2.3 The Fenchel dual: from constrained to regularized

The same game has a dual regularized description. For the regularizer $R _ { \eta } ( p ) = \eta ^ { - 1 } \mathrm { K L } ( p \| \pi )$ on $\Delta ( [ K ] )$ , the FTRL update

$$
p _ { t } = \operatorname * { a r g m i n } _ { p \in \Delta ( [ K ] ) } \left\{ \langle p , C _ { t - 1 } \rangle + \frac { 1 } { \eta _ { t } } \mathrm { K L } ( p \| \pi ) \right\}\tag{A.6}
$$

yields $p _ { t } ( i ) \propto \pi ( i ) e ^ { - \eta _ { t } C _ { t - 1 } ( i ) }$ . The Fenchel conjugate is the scale-η log-partition

$$
R _ { \eta } ^ { * } ( \ell ) : = \operatorname* { s u p } _ { p \in \Delta ( [ K ] ) } \left\{ \langle p , \ell \rangle - \eta ^ { - 1 } \mathrm { K L } ( p \| \pi ) \right\} = \eta ^ { - 1 } \log \sum _ { i = 1 } ^ { K } \pi ( i ) e ^ { \eta \ell ( i ) }\tag{A.7}
$$

The global constraint set under the prior is

$$
\mathcal { ( \mathrm { g l o b a l } } _ { } ( \eta , r ) = \left\{ \ell \in \mathbb { R } ^ { K } : \eta ^ { - 1 } \log \sum _ { i } \pi ( i ) e ^ { \eta \ell ( i ) } \leq r \right\}\tag{A.8}
$$

which constrains the log-partition function of ℓ under the prior at inverse temperature η, independently of the learner’s current distribution $p _ { t }$

The global constraint (A.8) is the constraint set matched to the prior-anchored FTRL update. But the one-round identity (3.5) operates locally, via the centered mixability gap—the expected-minus-mix-loss diference $\langle p _ { t } , c _ { t } \rangle - m _ { t } ( \eta _ { t } ) =$ $\eta _ { t } Q _ { t } ( c ) -$ computed relative to the current distribution $p _ { t }$ . This motivates a distribution-dependent constraint

Given $p _ { t } \in \Delta ( [ K ] ) , \eta _ { t } > 0$ , and budget $\beta > 0$ , define

$$
\mathcal { C } _ { t } ^ { \mathrm { R C G F } } ( p _ { t } , \eta _ { t } , \beta ) : = \left\{ c \in \mathbb { R } ^ { K } : \frac { 1 } { \eta _ { t } ^ { 2 } } \psi _ { p _ { t } } ( - \eta _ { t } ; c ) \leq \beta \right\}\tag{A.9}
$$

Proposition A.2 (Geometry of the local RCGF constraint). For fixed $( p _ { t } , \eta _ { t } , \beta )$

(a) The set $\mathcal { C } _ { t } ^ { \mathrm { R C G F } }$ is closed and convex (sublevel set of a log-sum-exp).

(b) $I f c \in \mathcal { C } _ { t } ^ { \mathrm { R C G F } }$ , then $c + b \mathbf { 1 } \in \mathcal { C } _ { t } ^ { \mathrm { R C G F } }$ for all $b \in \mathbb { R }$

(c) All constant vectors lie in $\mathcal { C } _ { t } ^ { \mathrm { R C G F } }$

(d) $\mathcal { C } _ { t } ^ { \mathrm { R C G F } }$ is dependent on $p _ { t } ,$ but not directly on π.

Property (b) is the game-theoretic statement that only relative expert performance matters.

The global constraint $\left( \mathsf { A } . 8 \right)$ uses the prior and cumulative loss as base measure. The local constraint (A.9) uses the current posterior and the one-round loss. They are related because $p _ { t }$ is the FTRL output applied to $C _ { t - 1 }$ . The local constraint is the conditional version: it constrains the one-round loss given the current state.

## A.3 The exact one-round value

The Bellman bound of Corollary 3.3 is strict at the single round treated in this section. It is an upper bound because it fixes the learner’s side of in $\mathrm { f } _ { p } \ \mathrm { s u p } _ { z } \mathrm { : }$ : any particular play, evaluated against nature’s best reply, is an upper bound on the value. The complementary certificate fixes nature’s side. Since nature’s feasible set $\mathcal { Z } _ { \eta , q } ( p )$ moves with $p ,$ nature’s strategy is a response rule $\zeta$ assigning a feasible move to each learner action, and every such rule bounds the value from below.

Proposition A.3 (Nature-side lower bound). Fix $\eta > 0 , q \ge 0 , \Gamma \ge 0$ and a state S, write $\mathrm { V a l _ { 1 } } ( S ; \eta , q , \Gamma )$ for the one-round value from S (so that the $\mathrm { V a l _ { 1 } } ( \eta , q , \Gamma )$ of Propositions A.5 and A.6 is $\mathrm { V a l } _ { 1 } ( 0 ; \eta , q , \Gamma ) )$ , and let $\zeta : \Delta ( [ K ] ) \to \mathbb { R } ^ { K }$ satisfy $\zeta ( p ) \in \mathcal { Z } _ { \eta , q } ( p )$ for every $p \in \Delta ( [ K ] )$ . Then

$$
\operatorname* { i n f } _ { p \in \Delta ( [ K ] ) } L _ { \Gamma } \big ( S + \zeta ( p ) \big ) \ \leq \ \mathrm { V a l } _ { 1 } ( S ; \eta , q , \Gamma ) : = \operatorname* { i n f } _ { p \in \Delta ( [ K ] ) } \operatorname* { s u p } _ { z \in { \mathscr Z } _ { \eta , q } ( p ) } L _ { \Gamma } ( S + z )\tag{A.10}
$$

with equality whenever $\zeta ( p )$ attains the inner supremum at every $p .$ Fixing instead a single comparator ν with $\mathrm { K L } ( \nu \| \pi ) \leq \Gamma$ and replacing $L _ { \Gamma }$ by $\langle \nu , \cdot \rangle$ weakens (A.10) further but keeps it valid.

The two certificates are not symmetric in dificulty. Playing Gibbs and reading of the potential increment gives the closed-form upper bound $\Gamma / \eta + \eta q$ with no case analysis, whereas a response rule must be specified at every $p .$ . The rule that closes the gap attacks the action the learner has left least protected, and it is exactly optimal once the comparator budget is large enough to reach every action — the regime in which the budget stops mattering at all.

Proposition A.4 (Saturated-budget reduction to a parameter-free game). Let $\Gamma _ { \mathrm { m a x } } : = \mathrm { m a x } ,$ $\log ( 1 / \pi ( i ) )$ (so $\Gamma _ { \mathrm { m a x } } =$ log K for the uniform prior). For every $\Gamma \geq \Gamma _ { \mathrm { m a x } }$ the relative-entropy ball $\{ \nu : \mathrm { K L } ( \nu \| \pi ) \leq \Gamma \}$ is the whole simplex $\Delta ( [ K ] )$ , and

$$
L _ { \Gamma } ( S ) = \operatorname* { m a x } _ { i } S ( i ) \qquad ( \Gamma \geq \Gamma _ { \operatorname* { m a x } } )\tag{A.11}
$$

Consequently, for every $\eta > 0 ,$ , horizon T, and budget profile $\left( q _ { t } \right)$ , the value $\mathrm { V a l } _ { 1 } ^ { \eta , \Gamma } ( 0 )$ is independent of Γ on $[ \Gamma _ { \mathrm { m a x } } , \infty )$ and equals the minimax value of the RCGF-budgeted game with terminal payofmax<sub>i</sub> $S _ { T } ( i )$ : the comparator-complexity dependence of the value saturates at $\Gamma = \Gamma _ { \mathrm { m a x } } .$

At $T = 1$ that parameter-free game is solved in closed form.

Proposition A.5 (Exact one-round value at a saturated comparator budget). Let $\Gamma \geq \Gamma _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { i } \log ( 1 / \pi ( i ) )$ , so that $L _ { \Gamma } = \operatorname* { m a x } _ { i } ( \cdot )$ by Proposition A.4. For $\eta > 0 , q > 0 , T = 1$ and $S _ { 0 } = 0 ,$ , the value of the fixed-scale game is

$$
\mathrm { V a l } _ { 1 } ( \eta , q , \Gamma ) = a _ { K } ( \eta , q )\tag{A.12}
$$

where $a _ { K } ( \eta , q )$ is the unique $a > 0$ solving

$$
\frac { 1 } { K } e ^ { \eta a } + \frac { K - 1 } { K } e ^ { - \eta a / ( K - 1 ) } = e ^ { \eta ^ { 2 } q }\tag{A.13}
$$

The value is attained by the uniform learner play $p = u _ { K }$ and by nature’s move placing $a _ { K }$ on one action and $- a _ { K } / ( K - 1 )$ on the rest. It does not depend on π.

At $K = 2$ Equation (A.13) is cosh $\begin{array} { r } { \mathbf { \nabla } _ { \mathfrak { l } } ( \eta a ) = e ^ { \eta ^ { 2 } q } . } \end{array}$ , so $a _ { 2 } = \eta ^ { - 1 }$ arcosh(e<sup>η2q</sup>) and (A.12) reproduces Proposition A.6 at $g ( \Gamma ) = 1$ . The general-K value is again strictly below the Bellman envelope: for the uniform prior at $\eta = 1 , q = 0 . 1$ $K = 5$ it is 0.8178 against the tightest envelope $\Gamma _ { \mathrm { m a x } } / \eta + \eta q = \log 5 + 0 . 1 \approx 1$ .7094 available in this regime. Two features of (A.13) carry over from the two-expert case: the budget enters only through $e ^ { \eta ^ { 2 } { q } }$ , and $a _ { K } = \sqrt { 2 ( K - 1 ) q } + O ( q )$ as $q \downarrow 0 .$ , the small-step regime in which the per-round loss is second order in nature’s step. Saturating the comparator budget removes π and Γ from the answer; for $\Gamma < \Gamma _ { \mathrm { m a x } }$ the optimal play moves of the uniform toward the prior — already visible at $K = 2$ , where it sits strictly between the two — and no comparable closed form is known beyond $K = 2$ (Section 11).

For a comparator budget below saturation, the two-expert instance is solved exactly. Fix $\begin{array} { r } { K = 2 , \pi = ( \frac { 1 } { 2 } , \frac { 1 } { 2 } ) } \end{array}$ , and a single round from $S = 0 .$ Define the comparator concentration coeficient $g ( \Gamma ) \in [ 0 , 1 ]$ as the unique $g \in [ 0 , 1 ]$ solving

$$
\begin{array} { r } { \frac { 1 } { 2 } \big [ ( 1 + g ) \log ( 1 + g ) + ( 1 - g ) \log ( 1 - g ) \big ] = \Gamma } \end{array}\tag{A.14}
$$

capped at $g ( \Gamma ) = 1$ once $\Gamma \geq \log 2 .$ The left-hand side is the relative entropy of $\textstyle \left( { \frac { 1 + g } { 2 } } , { \frac { 1 - g } { 2 } } \right)$ from the uniform prior, so $g ( \Gamma )$ measures how far a comparator of budget Γ can move of the uniform along a single coordinate. The limits are $g ( \Gamma ) \sim \sqrt { 2 \Gamma }$ as $\Gamma  0 ^ { + }$ and $g ( \log 2 ) = 1$

Proposition A.6 (Exact two-expert one-round value). For $\begin{array} { r } { K = 2 , \pi = ( \frac { 1 } { 2 } , \frac { 1 } { 2 } ) , \eta > 0 , q \geq 0 , \Gamma \geq 0 , } \end{array}$ , and $S = 0 ,$ , the one-round game has value

$$
\operatorname { V a l } _ { 1 } ( \eta , q , \Gamma ) = g ( \Gamma ) \frac { \operatorname { a r c o s h } \bigl ( e ^ { \eta ^ { 2 } q } \bigr ) } { \eta }\tag{A.15}
$$

attained by the learner’s Gibbs play $\begin{array} { r } { p = \left( \frac { 1 } { 2 } , \frac { 1 } { 2 } \right) } \end{array}$ ) and nature’s budget-saturating move $z ^ { \star } = ( \delta ^ { \star } , - \delta ^ { \star } ) , \delta ^ { \star } = \eta ^ { - 1 }$ arcosh $\left( e ^ { \eta ^ { 2 } \boldsymbol { q } } \right)$ For $\Gamma > 0$ and $q > 0$ this value satisfies thefixed-scale Bellman bound

$$
\mathrm { V a l } _ { 1 } ( \eta , q , \Gamma ) \le \frac { \Gamma } { \eta } + \eta q\tag{A.16}
$$

with strict inequality at everyfinite $\eta > 0$ when $\Gamma \geq \log 2 ;$ when $\Gamma < \log 2$ the inequality is strict except at the single scale $\begin{array} { r } { \eta ^ { 2 } q = - \frac { 1 } { 2 } \log \bigl ( 1 - \dot { g } ( \Gamma ) ^ { 2 } \bigr ) } \end{array}$ , where equality holds. At the Bellman-optimal scale $\eta _ { \mathrm { B } } = \sqrt { \Gamma / q }$ the exact value is afixed multiple of the Bellman envelope $2 { \sqrt { \Gamma q } } ,$

$$
\mathrm { { V a l } } _ { 1 } ( \eta _ { \mathrm { { B } } } , q , \Gamma ) = \kappa ( \Gamma ) \cdot 2 \sqrt { \Gamma q } \qquad w i t h \qquad \kappa ( \Gamma ) : = \frac { g ( \Gamma ) \mathrm { { a r c o s h } } ( e ^ { \Gamma } ) } { 2 \Gamma } \in ( 0 , 1 )\tag{A.17}
$$

where the gap factor $\kappa ( \Gamma )$ depends on Γ alone (not on $q ) ,$ is decreasing in Γ, and tends to 1 as $\Gamma  0 ^ { + }$

A $\operatorname { t } \eta = 1 , q = 0 . 1 , \Gamma = \log 2$ this gives $\delta ^ { \star } = \mathrm { a r c o s h } ( e ^ { 0 . 1 } ) \approx 0 . 4 5 4 7 , g ( \log 2 ) = 1$ , so $\mathrm { V a l } _ { 1 } \approx 0 . 4 5 4 7$ against the Bellman bound log $2 + 0 . 1 \approx 0 . 7 9 3 1$ . At the optimal scale the gap factor is $\kappa ( \log 2 ) \approx 0 . 9 5 \colon$ the envelope $2 \sqrt { \Gamma q }$ overshoots the truth by about 5% for the full simplex, and by more as the comparator budget grows $( \kappa ( 1 ) \approx 0 . 8 3$ $\kappa ( 2 ) \approx 0 . 6 7 )$ , while becoming exact in the low-complexity limit $\Gamma  0 ^ { + }$ . For $\Gamma \geq \log 2$ the gap shrinks to 0 as $\eta  \infty$ (the bound is asymptotically tight at large scale), whereas for $\Gamma < \log$ 2 the gap is smallest at a finite scale near $\eta _ { \mathrm { B } }$ and grows like $( 1 - g ( \Gamma ) ) \eta q \mathrm { a s } \eta \to \infty$ . Carrying the same instance to a second round shows where the multi-round dificulty enters: from the skewed post-round-1 state the exact minimax play over-tilts relative to the Gibbs iterate, so the single-round product form is lost (the computation appears below). Two features of this instance keep it hard, and dropping either one settles it: nature is capped round by round, and the learner may play of the Gibbs trajectory. Replace the per-round caps by one cumulative budget and hold the learner to Gibbs at a constant scale, as Game 4.1 does, and the multi-round value i closed form from a finite horizon on (Proposition 4.2).

One further structural fact localizes the gap. At every fixed scale the Bellman bound $L _ { \Gamma } ( S ) \le \Gamma / \eta + W _ { \eta } ( S )$ of Proposition 2.2 sharpens to an exact pointwise identity whose slack carries the whole obstruction:

$$
L _ { \Gamma } ( S ) = W _ { \eta } ( S ) + \frac { 1 } { \eta } A _ { \Gamma } ( \rho _ { \eta , S } ) , \qquad A _ { \Gamma } ( p ) : = \operatorname* { s u p } _ { \substack { \nu : \mathrm { K L } ( \nu \| \pi ) \le \Gamma } } \big [ \mathrm { K L } ( \nu \| \pi ) - \mathrm { K L } ( \nu \| p ) \big ] \le \Gamma\tag{A.18}
$$

a consequence of the Gibbs variational identity (2.6): the algebra $\mathrm { K L } ( \nu \| \rho _ { \eta , S } ) = \mathrm { K L } ( \nu \| \pi ) - \eta \left. \nu , S \right. + \eta W _ { \eta } ( S )$ rearranges to $\langle \nu , S \rangle = W _ { \eta } ( S ) + \eta ^ { - 1 } [ \mathrm { K L } ( \nu \Vert \pi ) - \mathrm { K L } ( \nu \Vert \rho _ { \eta , S } ) ]$ , and taking the supremum over the relative-entropy ball isolates $A _ { \Gamma } ( \rho _ { \eta , S } )$ . The slack in the Bellman bound is thus exactly $( \Gamma - A _ { \Gamma } ( \rho _ { \eta , S } ) ) / \eta \geq 0 \mathrm { ~ }$ , vanishing when the Gibbs posterior $\rho _ { \eta , S }$ meets the worst comparator on the relative-entropy sphere $\{ { \mathrm { K L } } ( \nu \Vert \pi ) = \Gamma \}$ , and the identity confines the whole gap to the single bounded terminal functional $A _ { \Gamma } ( \rho _ { \eta , S _ { T } } )$ . For $T = 1$ from $S _ { 0 } = 0$ under a uniform prior the learner’s Gibbs play $\rho _ { \eta , 0 } = \pi$ is minimax and the alignment is closed form: at $\begin{array} { r } { K = 2 , \pi = ( \frac { 1 } { 2 } , \frac { 1 } { 2 } ) } \end{array}$ , nature’s saturating move gives $A _ { \Gamma } ( \rho _ { \eta , S _ { 1 } } ) = g ( \Gamma )$ arcosh $\iota ( e ^ { \eta ^ { 2 } q } ) - \eta ^ { 2 } q$ , recovering Proposition A.6. That statement requires the prior to be uniform. Under a non-uniform prior the Gibbs play at $S _ { 0 } = 0 \mathrm { i } \mathrm { s } \pi$ itself, which concedes the response rule of Proposition A.5 a payof governed by $\mathrm { m i n } _ { i } \pi ( i ) < 1 / K$ and so is strictly beaten by the uniform play: at $\Gamma \ge \Gamma _ { \mathrm { m a x } } , K = 3 , \eta = 1$ $q = 0 . 1$ and $\pi = ( 0 . 6 , 0 . 3 , 0 . 1 )$ the Gibbs play concedes 1.1381 against the value $a _ { 3 } = 0 . 6 1 0 8$ . The Gibbs play is the equalizer at every state; whether it is also minimax depends on the prior and on the comparator budget, and the two can part already at $T = 1$ . For $T \geq 2$ the minimax learner over-tilts relative to the fixed-scale Gibbs iterate, so the telescoped loss $\begin{array} { r } { W _ { \eta } ( S _ { T } ) = \eta \sum _ { t } Q _ { t } } \end{array}$ no longer holds at the optimal play and the terminal functional ceases to factor across rounds. Numerically $( \eta = 1$ , per-round budget $q = 0 . 1 , \Gamma \geq$ log 2) the exact value at $T = 2$ equals 0.6486, strictly below both product forms $2 \eta ^ { - 1 }$ arcosh $( e ^ { \eta ^ { 2 } q } ) = 0 . 9 0 9 4$ and $\eta ^ { - 1 }$ arcosh $( e ^ { 2 \eta ^ { 2 } q } ) = 0 . 6 5 3 7$ , so no elementary closed form in $( \eta , q , \Gamma , T )$ survives.

The mechanics of the same instance $( \eta = 1$ , per-round budget $q _ { t } = 0 . 1 , \Gamma = \log 2 )$ make the obstruction concrete. Round 1 is the one-round game of Proposition A.6: from $p _ { 1 } ~ = ~ \left( \frac { 1 } { 2 } , \frac { 1 } { 2 } \right)$ nature saturates with the symmetric move $z _ { 1 } = ( \delta ^ { \star } , - \delta ^ { \star } )$ , whose centered RCGF $Q _ { 1 } = \eta ^ { - 2 }$ log cosh $( \eta \delta ^ { \star } ) = q _ { 1 }$ fixes $\delta ^ { \star } = \mathrm { a r c o s h } ( e ^ { 0 . 1 } ) \approx 0 . 4 5 \dot { 4 } 7$ , and the state moves to the skewed $S _ { 1 } = ( \delta ^ { \star } , - \delta ^ { \star } )$ . At round 2 the Gibbs play is no longer uniform, $p _ { 2 } = G _ { \eta } ( S _ { 1 } ) \approx ( 0 . 7 1 3 , 0 . 2 8 7 )$ ), so nature’s budget-saturating move—and the worst comparator direction it aligns with—becomes state-dependent: the symmetric arcosh form of round 1 gives way to the inverse of the Bernoull $\mathfrak { i } ( p _ { 2 } )$ centered RCGF, an asymmetric $z _ { 2 } \approx ( 0 . 3 1 0 , - 0 . 7 7 1 )$ (still centered, $\langle p _ { 2 } , z _ { 2 } \rangle = 0 )$ saturating the same budget $Q _ { 2 } = q _ { 2 } = 0 . 1$ . The intrinsic time accumulates to $V _ { 2 } = Q _ { 1 } + Q _ { 2 } = 0 . 2$ and the intrinsic-time loss to $\begin{array} { r } { \sum _ { t } \eta _ { t } Q _ { t } = 0 . 2 ; } \end{array}$ the two increments match only because the budgets do; the moves that saturate them difer. From the skewed state the exact minimax play parts from the Gibbs trajectory—the play that exactly minimizes the terminal payof $L _ { \Gamma } ( S _ { 2 } )$ over-tilts relative to $p _ { 2 } ,$ , so the Gibbs trajectory upper-bounds the value. Pinning the exact value down remains open (Section 11).

## A.4 Further results from Section 4

## A.4.1 The pressure game and its active scale

Proposition A.7 (Existence and uniqueness of the active scale). Fix $p \in \Delta ( [ K ] )$ and centered $z \in \mathbb { R } ^ { K }$ that is not $p { - } a . s .$ constant. For every target $b \in ( 0 , \operatorname* { m a x } _ { i \in \mathrm { s u p p } ( p ) } z ( i ) )$ there exists a unique $\eta > 0$ such that

$$
\log \sum _ { i = 1 } ^ { K } p ( i ) e ^ { \eta z ( i ) } = \eta b\tag{A.19}
$$

Game A.8 (Active-constraint pressure game). Game 1.1 with three parts changed. Move order: nature reveals its centered move $z _ { t } \left( \langle p _ { t } , z _ { t } \right. = 0 )$ first, and the learner responds. Nature’s constraint: no cap is imposed; instead the learner chooses a pressure target $b _ { t } \in ( 0 , \operatorname* { m a x } _ { i } z _ { t } ( i ) )$ and solves (A.19) for the unique $\eta _ { t } > 0$ that makes the one-step constraint active at $b _ { t } .$ . Learner’s move: the update is pinned to the Gibbs iterate $p _ { t + 1 } ( i ) \propto p _ { t } ( i ) e ^ { \eta _ { t } z _ { t } ( i ) }$ . The terminal

## A.4.2 The chain characterization behind horizon-freeness

Under a cumulative budget the whole run reduces to a walk of the Gibbs iterate: the transport identity $\eta ^ { 2 } Q _ { \eta } ( p , z ) =$ $\mathrm { K L } ( p \| p ^ { + } )$ makes each round’s loss the relative entropy between consecutive iterates. Nature’s play is then a chain $p _ { 1 } = \pi , p _ { 2 } , . . . , p _ { T + 1 }$ in $\Delta ( [ K ] )$ whose total length, measured by these per-step relative entropies, is capped by the budget. With $A _ { \Gamma }$ as in Section 4 — the terminal functional of the alignment identity (A.18) — and $p _ { t } = G _ { \eta } ( S _ { t - 1 } )$ , the value has the closed chain form below; it is established alongside Proposition 4.2 in Appendix D.3.

$$
\operatorname { V a l } _ { T } ^ { \eta } ( \Gamma , V ) = \frac { 1 } { \eta } \operatorname { s u p } \Bigl \{ \sum _ { t = 1 } ^ { T } \operatorname { K L } ( p _ { t } \| p _ { t + 1 } ) + A _ { \Gamma } ( p _ { T + 1 } ) \Bigr \}\tag{A.20}
$$

the supremum over chains $p _ { 1 } = \pi , \ldots , p _ { T + 1 }$ of full support with $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \mathrm { K L } ( p _ { t } \| p _ { t + 1 } ) \le \eta ^ { 2 } V } \end{array}$ . It equals the Bellman value $\Gamma / \eta + \eta V$ if and only if some such chain uses the budget in full and ends on the comparator sphere $\{ p : \mathrm { K L } ( p \| \pi ) = \Gamma \}$ The chain totals reaching that sphere in T steps form $\{ \mathrm { K L } ( \pi \| p ) : \mathrm { K L } ( p \| \pi ) = \Gamma \}$ at $T = 1$ and an interval $\lbrack c _ { T } , \infty )$ for $T \geq 2$ , with $c _ { T }$ nonincreasing in $T$ and $c _ { T } \to 0$ . That shrinking interval produces the finite $T _ { 0 }$ of (4.5).

The two requirements are coupled rigidly at one round: the single move that lands on the sphere also fixes the amount used at $\mathrm { K L } ( \pi \| p _ { 2 } )$ , the relative entropy from the prior to the comparator taken in the reverse direction, and the value falls short by that mismatch. A second round decouples them, since the intermediate point can be placed to use any amount at or above $c _ { 2 }$

## A.4.3 The two-step gate in closed form

At the Bellman-optimal scale $\eta = \sqrt { \Gamma / V }$ the gate of Proposition 4.2 reads $\Gamma \geq c _ { 2 } ,$ , and at $K = 2$ under a uniform prior $c _ { 2 }$ is closed form. In the tilt coordinate, with the sphere at w = arctanh g(Γ) (the comparator concentration coeficient $g ( \Gamma ) \in [ 0 , 1 ]$ of Proposition A.6), a two-step chain to it costs log cosh w − tanh $( w _ { 1 } ) ( w - w _ { 1 } )$ , stationary where $\begin{array} { r } { \frac 1 2 \sinh ( 2 w _ { 1 } ) + \dot { w _ { 1 } } = w ; } \end{array}$ that left side rises strictly from 0, so the root $w _ { 1 }$ is unique and

$$
c _ { 2 } = \log \cosh w - \sinh ^ { 2 } ( w _ { 1 } )\tag{A.21}
$$

Both ends of the range follow. As $\Gamma \downarrow 0$ the root tends to $w / 2$ and $c _ { 2 } ~  ~ w ^ { 2 } / 4 ~  ~ \Gamma / 2$ , so the gate holds and $T _ { 0 } = 2$ . As $\Gamma \uparrow \Gamma _ { \mathrm { m a x } } = \log 2$ the comparator sphere recedes toward the boundary of the simplex. Then $w \to \infty$ and sinh $\begin{array} { r } { \mathrm { \Pi } ^ { ^ 2 } ( w _ { 1 } ) \sim w - w _ { 1 } , { \mathrm s o } c _ { 2 } \sim \frac { 1 } { 2 } } \end{array}$ log w diverges while Γ stays below log 2: the gate fails and $T _ { 0 } > 2 .$ . The regimes meet at $\Gamma = 0 . 6 4 7 9 = 0 . 9 3 5 \Gamma _ { \mathrm { m a x } }$ , where $T _ { 0 }$ steps from 2 to 3. The gap factor $\kappa ( \Gamma )$ of Proposition A.6 does not persist: measured at that same scale, the $T = 1$ value is 0.06% below $2 \sqrt { \Gamma V }$ at $\Gamma = 0 . 1$ and $2 . 0 \%$ below at $\Gamma = 1 / 2$ , and $0 \%$ below at every $T \geq T _ { 0 }$

## A.4.4 What a varying scale is worth

Letting the scale vary lowers the value by very little. At $K = 2$ and $T = 2$ the minimax value over predictable schedules can be computed without discretizing nature at all. Its reply maximizes a convex function of its move over an interval, so the maximum sits at an endpoint, and the two endpoints solve a scalar equation. That computation puts the best predictable schedule below $2 \sqrt { \Gamma V }$ by a relative $6 \times 1 0 ^ { - 4 } \mathrm { a t } \Gamma = 1 / 2$ , and by under $1 0 ^ { - 6 } \mathrm { a t } \Gamma = 1 / 5$ . Both figures survive local refinement of nature’s opening move, and the first grows slightly as the scale grid is refined, so they bound from below what a schedule is worth. Retuning the scale therefore does lower the value, by less than a tenth of a percent at the budgets computed, and (4.4) over-estimates the value of Game 4.1 by no more than that.

Adaptation nevertheless has to be bounded to be safe. A rule placing the equalizer at the current state, $\eta _ { t } \propto 1 / \| S _ { t - 1 } \| _ { \infty } ,$ hands nature the scale: an opening move of size ϵ forces a scale of order $1 / \epsilon ,$ , at which nature converts its remaining budget to regret at rate $\eta _ { t } ,$ so the value grows without bound as $\epsilon \downarrow 0$ . Clipping the scale at $\eta _ { t } = 1$ , as the schedule (4.9) does, rules out exactly this.

## A.4.5 The intrinsic-time loss as a Riemann sum of a falling scale

Writing $\phi ( v ) : = \operatorname* { m i n } \{ 1 , C \sqrt { \Gamma / v } \}$ for the per-unit scale (with $\phi ( 0 ) : = 1 )$ , the schedule (4.9) evaluates round t at the left endpoint, $\eta _ { t } = \phi ( V _ { t - 1 } )$ , so

$$
\sum _ { t = 1 } ^ { T } \eta _ { t } Q _ { t } ( c ) = \sum _ { t = 1 } ^ { T } \phi ( V _ { t - 1 } ) \left( V _ { t } - V _ { t - 1 } \right) ,
$$

the left-endpoint sum of the nonincreasing ϕ against the increments of the intrinsic time. A nonincreasing scale evaluated at each interval’s left endpoint overestimates on every interval, so the sum lies above the integral $\int _ { 0 } ^ { V _ { T } } \dot { \phi } ( v )$ dv—which equals $2 C \sqrt { \Gamma V _ { T } } - C ^ { 2 } \Gamma$ once $V _ { T } \geq C ^ { 2 } \Gamma$ —and exceeds it by at most the largest single increment, $Q _ { T } ^ { * } ( c )$ . That bracketing is Theorem 4.4, whose two sides difer by the discretization cost of sum against integral, lower-order whenever no single round carries a macroscopic share of $V _ { T } ( c )$

The name of the schedule is earned in the unclipped regime: the marginal loss per unit of intrinsic time is the scale $\eta _ { t } = C \sqrt { \Gamma / V _ { t - 1 } } ;$ the transport bound $\Gamma / \eta _ { t }$ at that scale is $C ^ { - 1 } \sqrt { \Gamma V _ { t - 1 } }$ , so the marginal transport term is $\begin{array} { r } { \frac { d } { d v } \left[ C ^ { - 1 } \sqrt { \Gamma v } \right] = \frac { 1 } { 2 } C ^ { - 1 } \sqrt { \Gamma / v } \mathrm { a t } v = V _ { t - 1 } } \end{array}$ . The schedule keeps their ratio pinned at $2 C ^ { 2 }$ along the whole path—equal round by round at $C = 1 / \sqrt { 2 }$

## A.5 Classical bounds from the game

Each row of Table 1 fixes a triple $( \pi , S , A )$ and enters the shared three-step argument of Section 5.3 at a diferent point: the comparator-class dual (Proposition 2.2) evaluates the variational form, Proposition 5.1 closes it on a face, and the scale optimization converts the value into a relative-entropy rate. Each row still owes the classical literature its closing step. The derivations below carry each row from the game to the classical statement on the finite alphabet, closing step included; throughout, $X _ { 1 } , \ldots , X _ { n }$ are i.i.d. with law $P$ of full support on $[ K ] ,$ , and $c : [ K ] \to$ R is a score with $\mathbb { E } _ { P } [ c ] < t < \operatorname* { m a x } _ { i } c ( i )$

## A.5.1 Chernof

Take $\pi = P$ and state $c ,$ so that the scale-λ log-partition of the game is the cumulant generating function $\Lambda ( \lambda ) : =$ $\lambda W _ { \lambda } ( c ) = \log \mathbb { E } _ { P } [ e ^ { \lambda c } ]$ , and write $\begin{array} { r } { \bar { X } _ { n } : = n ^ { - 1 } { \bar { \sum _ { i = 1 } ^ { n } } } c ( X _ { j } ) } \end{array}$ . The closing step is exponential Markov. For every $\lambda \geq 0$ monotonicity of x $\mapsto e ^ { n \lambda x }$ and independence give

$$
\operatorname* { P r } [ { \bar { X } } _ { n } \geq t ] \ \leq \ e ^ { - n \lambda t } \operatorname { \mathbb { E } } _ { P } \left[ e ^ { \lambda \sum _ { j } c ( X _ { j } ) } \right] = e ^ { - n \lambda t } \operatorname { \mathbb { E } } _ { P } \left[ e ^ { \lambda c } \right] ^ { n } = e ^ { - n ( \lambda t - \Lambda ( \lambda ) ) }\tag{A.22}
$$

and optimizing the exponent over $\lambda \geq 0$ yields $\operatorname* { P r } [ \bar { X } _ { n } \geq t ] \leq e ^ { - n \Lambda ^ { \ast } ( t ) }$ with $\begin{array} { r } { \Lambda ^ { * } ( t ) = \operatorname* { s u p } _ { \lambda \geq 0 } \{ \lambda t - \Lambda ( \lambda ) \} } \end{array}$ . The game supplies the meaning of the exponent: by the Lagrange duality of Proposition 2.2,

$$
\Lambda ^ { * } ( t ) = \operatorname* { i n f } \left\{ \operatorname { K L } ( \nu \| P ) : \nu \in \Delta ( [ K ] ) , \langle \nu , c \rangle \geq t \right\} = : I ( t )\tag{A.23}
$$

the least comparator complexity that moves the score mean to $t -$ the same relative entropy the terminal payof $L _ { \Gamma }$ requires of a comparator, read at the mean-t face in place of a relative-entropy sphere.

## A.5.2 Cramér

The Chernof bound is one side of the large-deviation statement; the tilted-law argument matches it from below, so the exponent $I ( t )$ of (A.23) is exact. Since t is interior, the supremum defining $\Lambda ^ { * } ( t )$ is attained at a finite $\lambda ^ { \star }$ with $\Lambda ^ { \prime } ( \lambda ^ { \star } ) = t$ and the attaining tilt is the game’s Gibbs law at scale $\lambda ^ { \star } \colon P _ { \lambda ^ { \star } } ( i ) \propto P ( i ) e ^ { \lambda ^ { \star } c ( i ) }$ , with mean E $P _ { \lambda ^ { \star } } \left[ c \right] = \Lambda ^ { \prime } ( \lambda ^ { \star } ) = t$ and

$$
\mathrm { K L } ( P _ { \lambda ^ { \star } } | | P ) = \lambda ^ { \star } t - \Lambda ( \lambda ^ { \star } ) = \Lambda ^ { * } ( t )\tag{A.24}
$$

Fix $\varepsilon > 0$ and let $A _ { n } : = \left\{ | \bar { X } _ { n } - t | < \varepsilon \right\}$ . Changing measure from $P ^ { n }$ to $P _ { \lambda ^ { \star } } ^ { n }$

$$
\begin{array} { r } { \operatorname* { P r } [ \bar { X } _ { n } > t - \varepsilon ] \ \ge \ P ^ { n } ( A _ { n } ) = \mathbb { E } _ { P _ { \lambda ^ { \star } } ^ { n } } \Big [ { \mathbf 1 } _ { A _ { n } } e ^ { - \lambda ^ { \star } } \sum _ { j } c ( X _ { j } ) + n \Lambda ( \lambda ^ { \star } ) \Big ] \ \ge \ e ^ { - n ( \lambda ^ { \star } ( t + \varepsilon ) - \Lambda ( \lambda ^ { \star } ) ) } \ P _ { \lambda ^ { \star } } ^ { n } ( A _ { n } ) , } \end{array}
$$

and the law of large numbers under the tilt gives $P _ { \lambda ^ { \star } } ^ { n } ( A _ { n } ) \to 1$ , so lim in $\mathrm { f } _ { n } n ^ { - 1 } \log \mathrm { P r } [ \bar { X } _ { n } > t - \varepsilon ] \geq - \Lambda ^ { \ast } ( t ) - \lambda ^ { \star } \varepsilon$ Letting $\varepsilon \downarrow 0$ and combining with (A.22), the exponent is exactly the rate function at every interior continuity point,

$$
\operatorname* { l i m } _ { n  \infty } - \frac { 1 } { n } \log \operatorname* { P r } [ \bar { X } _ { n } \geq t ] = I ( t ) = \Lambda ^ { \ast } ( t )\tag{A.25}
$$

The two expressions for the rate are the two sides of the game’s duality: $\Lambda ^ { * }$ is the scale-side (Legendre) reading and I the comparator-side (relative-entropy) reading of the same terminal variational problem.

## A.5.3 Gibbs conditioning and the conditional limit

On a face the game’s optimizer is exact at every finite $n ,$ with no limit taken. Fix $A \subseteq [ K ]$ with $P ( A ) > 0$ and condition on $\{ X _ { j } \in A$ for all $j \leq n \}$ : the joint law factorizes, so

$$
\operatorname* { P r } [ X _ { 1 } = a \mid X _ { j } \in A \forall j \leq n ] = { \frac { P ( a ) \mathbf { 1 } \{ a \in A \} } { P ( A ) } } = P ( a \mid A )\tag{A.26}
$$

the conditioned prior itself — which is precisely the I-projection of $P$ onto the face $\{ \nu : \operatorname { s u p p } ( \nu ) \subseteq A \}$ , at relative entropy $\operatorname { K L } ( P ( \cdot \mid A ) \| P ) = - \log P ( A )$ , the event cost that Proposition 5.1 isolates and that the game’s terminal functional imposes. For a general closed convex constraint set $E \subseteq \Delta ( [ K ] )$ with $P \notin E$ and $\mathrm { i n f } _ { \nu \in E } \mathrm { K L } ( \nu \| P )$ attained at the I-projection $\nu ^ { \star }$ (unique by strict convexity), the conditional limit theorem replaces the exact identity by a limit: conditionally on the empirical type $\hat { \nu } _ { n }$ lying in $E ,$ the law of $X _ { 1 }$ converges to $\nu ^ { \star }$ . The mechanism is the Pythagorean inequality of the I-projection [20], $\mathrm { K L } ( \nu \| P ) \geq \mathrm { K L } ( \nu \| \nu ^ { \star } ) + \mathrm { K L } ( \nu ^ { \star } \| P )$ for every $\nu \in E$ , which places any type in $E$ at distance δ from $\nu ^ { \star }$ at exponent $\mathrm { K L } ( \nu ^ { \star } \| P ) + \delta ;$ Sanov’s bound below then makes such types conditionally negligible, and averaging types near $\nu ^ { \star }$ yields the limit [26]. The game reads the conditional limit as its $\eta \downarrow 0$ equilibrium: the optimizer of (5.3) is the Gibbs tilt of the conditioned prior, and as the scale vanishes the tilt relaxes onto the I-projection itself.

## A.5.4 Finite-state Sanov

The closing step is the method of types. For a type ν of denominator n (an empirical distribution of n points), every sequence $x ^ { n }$ of type ν has probability $\begin{array} { r } { \dot { P } ^ { n } ( x ^ { n } ) = \ddot { \prod _ { i } } P ( i ) ^ { n \nu ( i ) } = e ^ { - n ( \mathrm { K L } ( \nu | | P ) + \dot { H ( \nu ) } ) } } \end{array}$ , with $\begin{array} { r } { H ( \nu ) : = - \sum _ { i } \nu ( i ) } \end{array}$ log $\nu ( i )$ the Shannon entropy, and the type class $T _ { n } ( \nu )$ contains ${ \binom { n } { n \nu ( 1 ) , \dots , n \nu ( K ) } } \leq e ^ { n H ( \nu ) }$ sequences, at least $( n + 1 ) ^ { - \ l { K } } e ^ { \ l { n } \ l { H } ( \ l { \nu } ) }$ of them; hence

$$
\begin{array} { r } { ( n + 1 ) ^ { - K } e ^ { - n \mathrm { K L } ( \nu \| P ) } \leq P ^ { n } \big [ \hat { \nu } _ { n } = \nu \big ] \leq e ^ { - n \mathrm { K L } ( \nu \| P ) } } \end{array}\tag{A.27}
$$

Since there are at most $( n + 1 ) ^ { K }$ types of denominator n, summing the upper bound over the types in a closed set $E$ and taking the best lower-bound type near any interior $\nu \in E$ gives

$$
- \operatorname* { i n f } _ { \nu \in \mathrm { i n t } , E } \mathrm { K L } ( \nu \| P ) \ \leq \ \operatorname* { l i m i n f } _ { n } ~ \frac { 1 } { n } \log P ^ { n } \big [ \hat { \nu } _ { n } \in E \big ] \ \leq \ \operatorname* { l i m s u p } _ { n } ~ \frac { 1 } { n } \log P ^ { n } \big [ \hat { \nu } _ { n } \in E \big ] \ \leq \ - \operatorname* { i n f } _ { \nu \in E } \mathrm { K L } ( \nu \| P )\tag{A.28}
$$

the finite-state Sanov theorem. The game’s contribution is again the exponent’s variational form: on the simplex of types, $\mathrm { i n f } _ { \nu \in E } \mathrm { K L } ( \nu \| P )$ is the Lagrange dual of the comparator-class problem that $L _ { \Gamma }$ solves (Proposition 2.2), and when $E$ is a face it collapses to the closed form of Proposition 5.1. The polynomial factor $( n + 1 ) ^ { \pm K }$ is absorbed by the exponential scale.

## A.6 Further results from Section 5

## A.6.1 The suficient-statistic catalogue

Each row enters the update (A.6) as the cumulative statistic $C _ { t - 1 } ( i )$ of a per-round score. Here $\psi _ { \tau }$ is Winsorization at level $\tau ,$ so that $| \psi _ { \tau } | \leq \tau , \psi _ { \mathrm { C a t } }$ is Catoni’s influence function, and $u _ { t }$ is a predictable per-round ofset $( c _ { t } = \ell _ { t } + u _ { t } )$ , whose mismatch term $\begin{array} { r } { \sum _ { t } \left. \nu - p _ { t } , u _ { t } \right. } \end{array}$ records exactly how the game on $\ell _ { t }$ difers from the reduced game on $c _ { t }$
<table><tr><td>Algorithm</td><td>Sufficient statistic  $C _ { t - 1 } ( i )$ </td><td>Consequence</td></tr><tr><td>Vanilla Hedge</td><td> $\textstyle \sum _ { s < t } \ell _ { s } ( i )$ </td><td>Raw cumulative loss</td></tr><tr><td>Optimistic Hedge</td><td> $\begin{array} { r } { \sum _ { s < t } ( \ell _ { s } ( i ) - m _ { s } ( i ) ) } \end{array}$ </td><td>Only residuals contribute</td></tr><tr><td>Compensated</td><td> $\begin{array} { r } { \sum _ { s < t } ( \ell _ { s } ( i ) + u _ { s } ( i ) ) } \end{array}$ </td><td>Variance corrections included</td></tr><tr><td>Soft specialists</td><td> $\begin{array} { r } { \sum _ { s < t } \beta _ { s } ( i ) \ell _ { s } ( i ) } \end{array}$ </td><td>Confidence-rated experts</td></tr><tr><td>One-sided statistics</td><td> $\begin{array} { r l } { \sum _ { s < t } ( \langle p _ { s } , c _ { s } \rangle - c _ { s } ( i ) ) _ { + } } & { { } } \end{array}$ </td><td>Only adversarial coordinates</td></tr><tr><td>Winsorized scores</td><td> $\begin{array} { r } { \sum _ { s < t } \psi _ { \tau } ( z _ { s } ( i ) ) } \end{array}$ </td><td>Finite loss on heavy tails</td></tr><tr><td>Catoni transform</td><td> $\begin{array} { r l } { \sum _ { s < t } \alpha _ { s } ^ { - 1 } \psi _ { \mathrm { C a t } } ( \alpha _ { s } z _ { s } ( i ) ) } & { { } } \end{array}$ </td><td>Bounded influence, exact bias</td></tr></table>

Table 3: Suficient-statistic choices and their consequences. The game is defined relative to whichever statistic the learner scores; changing it changes the state, the intrinsic time, and the move constraint, and leaves every identity intact.

## A.6.2 Comparators on and of the minimal-mean face

Proposition A.9 (Difuse comparators reduce to the best expert). Runfixed-rate Hedge on i.i.d. composite losses $c _ { t } \in [ 0 , 1 ] ^ { K }$ with mean $\mu ,$ and let $\mu ^ { * } = \mathrm { m i n } _ { i } \mu ( i )$ and $k ^ { \ast } \in$ arg min ${ } _ { \cdot i } \mu ( i )$ . For every comparator ν $\in \Delta ( [ K ] )$

$$
\mathbb { E } R _ { T } ^ { c } ( \nu ) = \mathbb { E } R _ { T } ^ { c } ( \delta _ { k ^ { * } } ) - T \big ( \langle \nu , \mu \rangle - \mu ^ { * } \big ) \ \leq \ \mathbb { E } R _ { T } ^ { c } ( \delta _ { k ^ { * } } )\tag{A.29}
$$

Consequently, whenever some minimal-mean expert $k ^ { * }$ satisfies the point-mass low-noise condition with a finite constant $\kappa _ { k ^ { * } } .$ the choice $\eta = \operatorname* { m i n } \{ 1 , [ 2 ( e - 2 ) \kappa _ { k ^ { * } } ] ^ { - 1 } \}$ gives a constant-in-T bound for every ν,

$$
\mathbb { E } R _ { T } ^ { c } ( \nu ) \ \leq \ \mathbb { E } R _ { T } ^ { c } ( \delta _ { k ^ { * } } ) \ \leq \ 2 \big ( 1 + 2 ( e - 2 ) \kappa _ { k ^ { * } } \big ) \big ( - \log \pi ( k ^ { * } ) \big )\tag{A.30}
$$

The one comparator class left uncovered by the point-mass mechanism is a tied minimum carried by experts with distinct losses, at which no minimal-mean expert satisfies the point-mass condition with a finite constant.

The restriction to that face is sharp, and comes directly from the definition. If the comparator-centered low-noise condition holds with a finite constant, then any expert in supp(ν) sharing the posterior-average mean loss must have $c _ { t } ( i ) = \langle \nu , c _ { t } \rangle$ almost surely; taking $i \in \arg \operatorname* { m i n } _ { j } \mu ( j )$ then forces $\langle \nu , \mu \rangle = \mu ^ { * }$ , so the condition is restrictive for a difuse comparator.

Expanding the square in Proposition 5.2 about the algorithm mean gives the exact identity $\Psi _ { t } ( \nu ) = \operatorname { V a r } _ { p _ { t } } ( c _ { t } ) +$ $\langle p _ { t } - \nu , c _ { t } \rangle ^ { 2 }$ , so the comparator enters the comparator-centered second moment only through the instantaneous regret $\langle p _ { t } - \nu , c _ { t } \rangle$ . A comparator with $\langle \nu , \mu \rangle > \mu ^ { * }$ is beaten by a linear margin $T ( \langle \nu , \mu \rangle - \mu ^ { * } )$ under (A.29), so a constant regret bound holds a fortiori; one with $\langle \nu , \mu \rangle = \mu ^ { * }$ ties the best expert in mean and inherits its guarantee whenever that expert meets the point-mass condition, which is the mechanism Theorem 5.3 uses. For finite K a diferent argument handles the tied-minimum face whenever the of-face mean gap is strict. If every expert outside the minimal-mean face has mean at least $\mu ^ { * } + \Delta$ for some $\Delta > 0$ , fixed-rate Hedge puts geometrically decaying weight on those experts, so the excess over a face comparator sums to $\mathbb { E } R _ { T } ^ { c } ( \nu ) = O ( 1 )$ uniformly in T.

## A.6.3 Drawdown and mixture width, numerically

Since in $\dot { \bar { \iota } } _ { t \leq T } \Lambda _ { t } \leq \Lambda _ { 0 } = 0 ,$ , every path has $\mathrm { D D } _ { T } \geq \Lambda _ { T } = - \log \mathrm { P R } _ { T } $ , and under the true parameter the mean of that lower bound, E[− log $\mathrm { P R } _ { T } ]$ , is the accumulated log-moment deficit of the e-process: linear in T for independent increments, and equal to $\bar { T } \eta ^ { 2 } \sigma ^ { 2 } \dot { / } 2$ for a score of sub-Gaussian scale σ at a fixed $\eta .$ The two regimes separate numerically. Take a standard normal score at $\alpha = 0 . 0 5$ , so the budget is log $( 1 / \alpha ) = 3 . 0 0$ nats. With η held at its $T = 1 0 ^ { 3 }$ tuning (the η solving $T \eta ^ { 2 } \sigma ^ { 2 } / 2 = \log ( 1 / \alpha )$ there), the deficit is linear in the horizon and the drawdown follows it. The averages are 0.74, 3.86 and 30.9 nats at $\dot { T } = 1 0 ^ { 2 } , 1 0 ^ { 3 }$ and $1 0 ^ { 4 }$ , each the linear deficit at that horizon plus a running-minimum overshoot of less than one nat. Retuned to each horizon instead, $T \eta ^ { 2 } \sigma ^ { 2 } / 2$ is held at $\log ( 1 / \alpha )$ throughout, and the same quantity stays near 3.8 nats at all three.

The width comparison is read at the same setting: a standard normal score at $\alpha = 0 . 0 5$ , three sets read once at $T = 1 0 ^ { 3 }$ , each against a nominal two-sided miss rate of $1 0 ^ { - 1 }$ . The fixed-horizon interval is the reference, missing at $1 . 4 \times 1 0 ^ { - 2 }$ . A normal-mixture sequence at the mixture’s own best tuning is 24.0% wider and misses at $2 . 4 \times 1 0 ^ { - 3 } ;$ ; under the conventional unit tuning it is 46.8% wider and misses at $3 . 3 \times 1 0 ^ { - 4 }$ . The three miss rates are the Gaussian endpoint tails of the three radii, so the shortfall against the nominal level measures the same conservatism the extra width does. Both widths are read of closed forms: the mixture boundary $[ \tau ^ { - 2 } ( 1 + \tau ^ { 2 } \sigma ^ { 2 } T ) ( 2 \log ( 1 / \alpha ) + \log ( 1 + \tau ^ { 2 } \sigma ^ { 2 } T ) ) ] ^ { 1 / 2 }$ at mixing scale τ, against the endpoint radius $\sigma \sqrt { 2 T \log ( 1 / \alpha ) }$ . Their ratio depends on $( \tau , T )$ only through $\tau ^ { 2 } \sigma ^ { 2 } T$ , which is why the best-tuned figure carries no horizon.

## A.7 Further results from Section 6

## A.7.1 Opponent models and the robust center

Opponent models pool multiplicatively. $\mathrm { I f } \omega _ { t , 1 } , \dots , \omega _ { t , W } \in \Delta ( [ L ] )$ are candidate opponent models with response priors $\pi _ { t , w } ( i ) \propto e ^ { - \eta ( M \omega _ { t , w } ) _ { i } }$ , then each log $\pi _ { t , w } ( i )$ is afine in $( M \omega _ { t , w } ) _ { \ i }$ <sub>i</sub> up to an i-independent normalizer, so the geometric pool at nonnegative weights $\begin{array} { r } { \alpha _ { t , w } \mathrm { i s } \prod _ { w } \pi _ { t , w } ( i ) ^ { \alpha _ { t , w } } \propto \exp ( - \eta \sum _ { w } \alpha _ { t , w } ( M \omega _ { t , w } ) _ { i } ) } \end{array}$ . Geometric pooling in prior space is ordinary exponential response to the averaged opponent model, and the worst case over weightings gives a robust center. The duality below is stated for strategy priors in $[ 7 ] ;$ it is recorded here in the game’s coordinates

Proposition A.10 (Robust center over several priors). Given positive priors $\pi _ { 1 } , \ldots , \pi _ { W } \in \Delta ( [ K ] )$ , define $C _ { \alpha } ( \pi _ { 1 : W } ) : =$ $\begin{array} { r } { - \log \sum _ { i = 1 } ^ { K } \prod _ { w = 1 } ^ { W } \pi _ { w } ( i ) ^ { \alpha _ { w } } } \end{array}$ and $\begin{array} { r } { C ( \pi _ { 1 : W } ) : = \operatorname* { m a x } _ { \alpha \in \Delta ( [ W ] ) } C _ { \alpha } ( \pi _ { 1 : W } ) } \end{array}$ . Then

$$
C ( \pi _ { 1 : W } ) = \operatorname* { m i n } _ { \substack { p \in \Delta ( [ K ] ) w \in [ W ] } } \mathrm { K L } ( p \| \pi _ { w } )\tag{A.31}
$$

and the optimizer is the geometric mixture induced by a maximizing α.

In a repeated game, using the robust center as a baseline and running optimistic Hedge on the residuals yields regret controlled by the intrinsic time of the residual sequence. If one of the W models is close to the opponent’s actual strategy, the residuals are small. Convergence is then fast.

## A.7.2 The regret-to-equilibrium conversions

Proposition A.11 (External regret implies CCE). If each player in afinite normal-form game satisfies external regret at most $\varepsilon _ { m } T - r e g r e t$ against the best single action in hindsight — then the empirical distribution of play is an ε-coarse correlated equilibrium (CCE) [11, 33]. Here $\varepsilon = \operatorname* { m a x } _ { m } \varepsilon _ { m }$

Proposition A.12 (Swap regret implies CE). If each player satisfies swap regret at most $\varepsilon _ { m } T - r e g r e t$ against the bestfixed remapping of its own actions — then the empirical distribution of play is an ε-correlated equilibrium [33].

## A.7.3 Why the ledger supplies no converse

The forward implication max<sub>k</sub> $V _ { T } ^ { ( k ) } = o ( T ) \Rightarrow d ( \sigma _ { T } , \mathrm { C C E } ) = o ( T ^ { - 1 / 2 } )$ holds because $\begin{array} { r } { V _ { T } ^ { ( k ) } = \sum _ { t } Q _ { t } ^ { ( k ) } } \end{array}$ governs every horizon-growing term of $R _ { T } ^ { ( k ) } = P _ { T } ^ { ( k ) } + D _ { T } ^ { ( k ) } + B _ { T } ^ { ( k ) , \star }$ : the terminal transport obeys $B _ { T } ^ { ( k ) , \star } \leq \Gamma ^ { ( k ) } / \eta _ { T } ^ { ( k ) } =$ $O ( \sqrt { \Gamma ^ { ( k ) } V _ { T } ^ { ( k ) } } )$ , and the drift is a gain

The reverse implication is not available. A lower bound on $R _ { T } ^ { ( k ) }$ in terms of $V _ { T } ^ { ( k ) }$ must ofset the envelope’s lower side $2 C \sqrt { \Gamma ^ { ( k ) } V _ { T } ^ { ( k ) } } - C ^ { 2 } \Gamma ^ { ( k ) }$ against the drift gain $D _ { T } ^ { ( k ) }$ , whose magnitude is itself controlled only by the transport scale $\Gamma ^ { ( k ) } / \eta _ { T } ^ { ( k ) } \le C ^ { - 1 } \sqrt { \Gamma ^ { ( k ) } V _ { T } ^ { ( k ) } }$ . The residual coeficient $2 C - C ^ { - 1 }$ is exactly zero at the headline $C = 1 / { \sqrt { 2 } } .$ So in the setting of Theorem 6.3 under the self-tuned schedule, whether $d ( \sigma _ { T } , \mathrm { C C E } ) = o ( T ^ { - 1 / 2 } )$ forces max<sub>k</sub> $V _ { T } ^ { ( k ) } = o ( T )$ is undecided, and deciding it needs a lower bound the envelope cannot supply at that constant.

Theorem A.13 (Bounded intrinsic time at strict equilibria). Consider intrinsic-time regret-matching self-play (Theorem 6.1) in a finite game under the self-tuned schedule $\eta _ { t } ^ { ( k ) } = C \sqrt { \Gamma ^ { ( k ) } / ( V _ { t - 1 } ^ { ( k ) } + \Gamma ^ { ( k ) } ) }$ , and write $\begin{array} { r } { B \ : = \ \operatorname* { m a x } _ { k } \operatorname* { m a x } _ { \pmb { a } } | c _ { k } ( \pmb { a } ) | } \end{array}$ for the loss range. Suppose the realized trajectory converges to a strict pure Nash profile $a ^ { \star }$ with stability gap $\Delta : = \ :$ min<sub>k</sub> min $\not = a _ { k } ^ { \star } \left[ u _ { k } \big ( a _ { k } ^ { \star } , a _ { - k } ^ { \star } \big ) - u _ { k } \big ( i , a _ { - k } ^ { \star } \big ) \right] > 0$ (cost convention $u _ { k } = - c _ { k } \colon a _ { k } ^ { \star }$ strictly beats every deviation by at least $\Delta$ when opponents $p l a y a _ { - k } ^ { \star } ) ,$ , with no assumption on the rate of approach. Then for every player k the of-equilibrium mass $1 - p _ { t } ^ { ( k ) } ( a _ { k } ^ { \star } )$ and the per-round RCGF $Q _ { t } ^ { ( k ) }$ decay geometrically, the schedule freezes at a positive scale $\eta _ { t } ^ { ( k ) }  \eta _ { \infty } ^ { ( k ) } \in ( 0 , C ]$ and the realized intrinsic time is bounded,

$$
V _ { T } ^ { ( k ) } = \sum _ { t = 1 } ^ { T } Q _ { t } ^ { ( k ) } = O ( 1 )\tag{A.32}
$$

so that $d ( \sigma _ { T } , \mathrm { C C E } ) = O ( 1 / T )$ in the ledger (6.5).

A few canonical instances make the ledger (6.5) concrete.

• Two-player zero-sum (exact Nash gap under admissible budgets). If $u _ { 2 } = - u _ { 1 } = : M$ and $\Gamma _ { \mathrm { r o w } } \geq \operatorname* { m a x } _ { i } ( - \log \pi _ { \mathrm { r o w } } ( i ) )$ $\Gamma _ { \mathrm { c o l } } \geq \operatorname* { m a x } _ { j } ( - \log \pi _ { \mathrm { c o l } } ( j ) )$ (every pure strategy is in the comparator class), then $d ( \sigma _ { T } , \mathrm { C C E } )$ equals the larger of the two players’ exact unilateral deviation gains (each player’s own exact-regret sum $P _ { T } ^ { ( k ) } + D _ { T } ^ { ( k ) } + B _ { T } ^ { ( k ) , \star }$ divided by T). These two gains sum to the classical Nash duality gap ma $\mathrm { x } _ { \omega ^ { \prime } } \bar { p } _ { T } ^ { \top } M \omega ^ { \prime } - \operatorname* { m i n } _ { p ^ { \prime } } p ^ { \prime \top }$ Mω¯<sub>T</sub> of (6.3), so $d ( \sigma _ { T } , \mathrm { C C E } )$ is at most that gap, with equality exactly when one player’s deviation gain vanishes. With smaller comparator budgets the same identity bounds a relative-entropy-restricted deviation class.

• Random general-sum $2 \times 2 .$ For utilities drawn $\operatorname { U n i f } ( [ - 1 , 1 ] )$ and the schedule $\eta _ { t } ^ { ( k ) } = \sqrt { ( \log 2 ) / ( V _ { t - 1 } ^ { ( k ) } + \log 2 ) }$ , the linear-intrinsic-time regime $V _ { T } ^ { ( k ) } = \Theta ( T )$ arises on the instances whose only equilibrium is interior, where (6.5) reads $d ( \sigma _ { T } , \mathrm { C C E } ) = { \cal O } ( T ^ { - 1 / 2 } )$ with the leading constant governed by the realized $V _ { T } ^ { ( k ) }$ . Instances admitting a pure equilibrium to which self-play converges instead have bounded intrinsic time and fall under the potential/congestion case below.

• Potential / congestion games. On traces where self-play converges to a strict pure Nash profile, the per-round centered RCGF collapses geometrically as $p _ { t } ^ { ( k ) }$ concentrates and the self-tuned schedule freezes at a positive scale. The realized intrinsic time is therefore in fact bounded, $V _ { T } ^ { ( k ) } = { \cal { O } } ( 1 )$ (Theorem $\mathrm { A } . 1 3 ) _ { \mathrm { i } }$ , yielding $d ( \sigma _ { T } , \mathrm { C C E } ) = O ( 1 / T )$ — sharper than the $O ( { \sqrt { \log T } } / T )$ a merely logarithmic intrinsic time would give. Whether a finite potential game can be forced to keep $V _ { T } ^ { ( k ) }$ super-logarithmic under this schedule, via non-convergent multiplicative-weights dynamics [51], is a convergence-of-dynamics question this paper does not settle.

• Adversarial linear-growth witness. Conversely, if a player’s losses are constructed to keep $p _ { t } ^ { ( k ) }$ far from any equalizer (e.g. a cyclic loss sequence), then $V _ { T } ^ { ( k ) } = \Theta ( T )$ . Along that trajectory the ledger is an equality, so its bound is the classical $O ( 1 / \sqrt { T } )$ ; whether the realized rate matches it is undecided.

The instances separate through the realized intrinsic time alone: the ledger reproduces each player’s exact unilateral deviation gain, so $d ( \sigma _ { T } , \mathrm { C C E } )$ is one exploitability in place of the Nash duality gap, and $V _ { T } ^ { ( k ) }$ bounded versus linear is exactly the $O ( 1 / T )$ versus $O ( 1 / \sqrt { T } )$ dichotomy.

## A.8 Further results from Section 7

Theorem A.14 (Generic pathwise bandit decomposition). For a comparator $u \in \Delta ( [ K ] )$ , the sampled composite-loss regret decomposes pathwise as follows.

$$
\mathrm { R e g } _ { T } ^ { c } ( u ) = M _ { T } ^ { \mathrm { p l a y } } + \Xi _ { T } - M _ { T } ^ { \mathrm { e s t } } ( u ) + \mathrm { B i a s } _ { T } ( u ) + \widehat { D } _ { T } + \widehat { B } _ { T } ( u ) + \sum _ { t = 1 } ^ { T } \eta _ { t } \widehat { Q } _ { t }\tag{A.33}
$$

Here $\begin{array} { r } { M _ { T } ^ { \mathrm { p l a y } } : = \sum _ { t } ( c _ { t } ( A _ { t } ) - \langle \mu _ { t } , c _ { t } \rangle ) } \end{array}$ (play martingale), $\begin{array} { r } { \Xi _ { T } : = \sum _ { t } \left. \mu _ { t } - p _ { t } , c _ { t } \right. } \end{array}$ (exploration cost), $M _ { T } ^ { \mathrm { e s t } } ( u ) : =$ $\begin{array} { r } { \sum _ { t } \left. p _ { t } - u , \widehat { c } _ { t } - \bar { c } _ { t } \right. } \end{array}$ (estimation martingale, with $\bar { \boldsymbol { c } } _ { t } : = \mathbb { E } [ \widehat { \boldsymbol { c } } _ { t } \mid \mathcal { F } _ { t - 1 } ]$ the predictable conditional mean of the estimated score), Bias $\begin{array} { r } { \mathbf { \Pi } _ { T } ( u ) : = \sum _ { t } \left. p _ { t } - u , c _ { t } - \bar { c } _ { t } \right. } \end{array}$ (predictable bias), and the last three terms are the full-information game on estimated scores. Both $M _ { T } ^ { \mathrm { p l a y } }$ and $M _ { T } ^ { \mathrm { e s t } } ( u )$ are martingales.

The one term the bandit decomposition (A.33) appears to leave implicit — the estimated per-round RCGF $\widehat { Q } _ { t } =$ $Q _ { \eta _ { t } } ( p _ { t } , \widehat { z } _ { t } )$ on the estimated scores — is not a quantity to be bounded but has an exact closed form. Conditioned on the past its only randomness is the drawn arm $A _ { t } , \mathsf { a } K$ -valued variable. Its conditional expectation is therefore a finite explicit sum.

Proposition A.15 (Exact per-round value in the true losses and observation geometry). Let $A _ { t } \sim \mu _ { t }$ and let $\widehat { z } _ { t }$ be any estimated centered score that is a deterministic function of the drawn arm and its feedback. Write ${ \widehat { z } } _ { t } ^ { ( a ) } f o r$ its realized value on $\{ A _ { t } = a \}$ . Then

$$
\mathbb { E } \left[ \widehat { Q } _ { t } \mid \mathcal { F } _ { t - 1 } \right] = \eta _ { t } ^ { - 2 } \sum _ { a } \mu _ { t } ( a ) \log \sum _ { i } p _ { t } ( i ) e ^ { \eta _ { t } \widehat { z } _ { t } ^ { ( a ) } ( i ) }\tag{A.34}
$$

For plain-bandit inverse-propensity scores $\widehat { c } _ { t } ( i ) = c _ { t } ( i ) \mathbf { 1 } \{ A _ { t } = i \} / \mu _ { t } ( i )$ this collapses coordinatewise to

$$
\mathbb { E } \big [ \widehat { Q } _ { t } \mid \mathcal { F } _ { t - 1 } \big ] = \frac { \langle p _ { t } , c _ { t } \rangle } { \eta _ { t } } + \eta _ { t } ^ { - 2 } \sum _ { a } \mu _ { t } ( a ) \log \Big ( 1 - p _ { t } ( a ) + p _ { t } ( a ) e ^ { - \eta _ { t } c _ { t } ( a ) / \mu _ { t } ( a ) } \Big )\tag{A.35}
$$

exact at every scale $\eta _ { t }$ and written entirely in the true losses $c _ { t }$ and the observation geometry $\mu _ { t }$ . Its small-scale limit is th observation-inflated variance,

$$
\mathbb { E } [ \widehat { Q } _ { t } \mid \mathcal { F } _ { t - 1 } ] = \frac { 1 } { 2 } \sum _ { a } \frac { p _ { t } ( a ) \big ( 1 - p _ { t } ( a ) \big ) } { \mu _ { t } ( a ) } c _ { t } ( a ) ^ { 2 } + O ( \eta _ { t } ) , \qquad \mathbb { E } \big [ \mathrm { V a r } _ { p _ { t } } ( \widehat { c } _ { t } ) \mid \mathcal { F } _ { t - 1 } \big ] = \sum _ { a } \frac { p _ { t } ( a ) \big ( 1 - p _ { t } ( a ) \big ) } { \mu _ { t } ( a ) } c _ { t } ( a ) ^ { 2 }\tag{A.36}
$$

the geometry entering through the propensity inflation $1 / \mu _ { t } ( a ) ,$ ;for afeedback graph the samefinite sum holds with the graph-IX scores, the geometry entering through the observation probabilities $o _ { t } ( a )$

Proposition A.16 (Proper-wrapper duplication identity). Let Π be afinite policy class and lift it to an expert class E with $p r i o r \tilde { \pi } \in \Delta ( \mathcal { E } )$ . A designated policy $\pi ^ { \star } \in \Pi$ is represented by a group $G \subseteq { \mathcal { E } }$ of loss-identical copies (two copies of $\cdot _ { \pi ^ { \star } }$ induce the same action law $\pi ^ { \star } ( \cdot \mid x _ { t } )$ in every context, hence ${ \widehat { c } } _ { t } ( e ) = { \widehat { c } } _ { t } ( \pi ^ { \star } )$ for all $e \in G$ and all t). Write $\begin{array} { r } { \tilde { \pi } ( G ) : = \sum _ { e \in G } \tilde { \pi } ( e ) } \end{array}$ for the aggregate prior mass. Running the full-information game (Section 1.1) over E at scale η on the estimated expert losses, the lowest-complexity comparator supported on G is $\nu ^ { \star } ( e ) = \tilde { \pi } ( e ) / \tilde { \pi } ( G )$ on G. The comparator complexity of competing against $\pi ^ { \star }$ is then exactly

$$
\operatorname* { m i n } _ { \nu : \operatorname* { s u p p } ( \nu ) \subseteq G } \mathrm { K L } ( \nu \| \tilde { \pi } ) = \mathrm { K L } ( \nu ^ { \star } \| \tilde { \pi } ) = - \log \tilde { \pi } ( G ) = \log \frac { 1 } { \tilde { \pi } ( G ) }\tag{A.37}
$$

Consequently, by the transport identity (3.6) applied in the lifted game, the comparator-complexity term of the policy-expert decomposition is

$$
\widehat { B } _ { T } ^ { \Pi } ( \pi ^ { \star } ) = \frac { \log \frac { 1 } { \widetilde { \pi } ( G ) } - \mathrm { K L } ( \nu ^ { \star } \| \widehat { \rho } _ { T + 1 } ) } { \eta }\tag{A.38}
$$

with $\widehat { \rho } _ { T + 1 }$ the terminal posterior. The leading term $\eta ^ { - 1 } \log { \frac { 1 } { \tilde { \pi } ( G ) } }$ is exact.

Corollary A.17 (The comparator-complexity reduction as a mass log-ratio). $H \pi ^ { \star }$ carries original mass $\beta _ { 0 } = \tilde { \pi } _ { 0 } ( \pi ^ { \star } )$ and, after duplication, aggregate mass $\beta = { \tilde { \pi } } ( G )$ , the reduction in comparator complexity is

$$
\frac { 1 } { \eta } \Bigl ( \log \frac { 1 } { \beta _ { 0 } } - \log \frac { 1 } { \beta } \Bigr ) = \frac { 1 } { \eta } \log \frac { \beta } { \beta _ { 0 } }\tag{A.39}
$$

the log-ratio of duplicated to original mass. Driving $\beta  1$ sends the designated policy’s comparator complexity to 0.

This is the same pooling phenomenon that runs through the framework: loss-identical experts coincide on every round, so their prior masses pool additively and the relative-entropy comparator complexity sees only the aggregate mass. Assembling this into the bandit decomposition of Theorem A.14 gives the wrapper with matching constants: with sampling $\begin{array} { r } { \mu _ { t } = ( 1 - \gamma ) \bar { p } _ { t } + \gamma u _ { [ A ] } , \bar { p } _ { t } ( a ) = \sum _ { e } p _ { t } ( e ) \pi _ { e } ( a \mid x _ { t } ) } \end{array}$ , and unbiased IPS losses $\widehat { c } _ { t } ( a ) = c _ { t } ( a ) \mathbf { 1 } \{ A _ { t } = a \} / \mu _ { t } ( a )$ , one has Bias<sub>T</sub> ≡ 0. The exploration cost obeys $\begin{array} { r } { | \Xi _ { T } | \le \gamma T } \end{array}$ (since $\mu _ { t } - \bar { p } _ { t } = \gamma ( u _ { [ A ] } - \bar { p } _ { t } )$ and losses lie in $[ 0 , 1 ] ;$ the inner product $\left. u _ { [ A ] } - \bar { p } _ { t } , c _ { t } \right.$ carries either sign); the IPS scores obey ${ \widehat { c } } _ { t } ( a ) \leq A / \gamma ;$ and $\widehat { B } _ { T } ^ { \Pi } ( \pi ^ { \star } )$ is (A.38). Taking expectations removes the play and estimation martingales, and at constant scale $\widehat { D } _ { T } = 0$ . This leaves

$$
\mathbb { E } \big [ \mathrm { R e g } _ { T } ^ { c } ( \pi ^ { \star } ) \big ] \ \leq \ \gamma T \ + \ \frac { 1 } { \eta } \log \frac { 1 } { \tilde { \pi } ( G ) } \ + \ \sum _ { t = 1 } ^ { T } \eta _ { t } \mathbb { E } \big [ \widehat { Q } _ { t } ^ { \Pi } \big ]\tag{A.40}
$$

in which the two costs the reduction introduces — the explicit exploration term $\gamma T$ and the $A / \gamma$ IPS rescaling inside the estimated intrinsic time $\begin{array} { r } { \sum _ { t } \eta _ { t } \widehat { Q } _ { t } ^ { \Pi } - \mathrm { c a r r y } } \end{array}$ exact constants, and the comparator complexity is the exact duplicationadjusted log $\frac { 1 } { { \tilde { \pi } } ( G ) }$ in place of log $\lvert \Pi \rvert$ . As with the plain-bandit case, the sharp rate closes once the estimated per-round RCGF $\widehat { Q } _ { t } ^ { \Pi }$ is evaluated by its exact closed form (Proposition A.15).

Theorem A.18 (Pairwise testing exponents are the concentration game’s Bellman potential). With $S , P _ { j } ^ { a }$ , and $\eta = s$ as above, for every $s \in ( 0 , 1 )$

(a) (Exact identity.) The Chernof coeficient is minus the game’s per-observation Bellman potential of $S ,$

$$
C _ { s } ( P _ { i } ^ { a } , P _ { j } ^ { a } ) = - s W _ { s } ( S ) = - \log \mathbb { E } _ { P _ { j } ^ { a } } \big [ e ^ { s S } \big ] = - \log Z ( s , 1 - s )\tag{A.41}
$$

where $Z ( s , 1 - s ) = \textstyle { \int } ( d P _ { i } ^ { a } ) ^ { s } ( d P _ { j } ^ { a } ) ^ { 1 - s }$ is the two-way mixed-coincidence partition function.

(b) (Equilibrium play is the Chernof tilt.) The game’s Gibbs equalizer at $( S , s )$ is the geometric-mean tilt of the two hypotheses,

$$
G _ { s } ( S ) = \rho _ { s , S } = \nabla W _ { s } ( S ) \propto ( d P _ { i } ^ { a } ) ^ { s } ( d P _ { j } ^ { a } ) ^ { 1 - s } = : R _ { s }\tag{A.42}
$$

Equivalently $R _ { s }$ is the relative-entropy barycenter, and by the mixed-coincidence identity [8], $C _ { s } ( P _ { i } ^ { a } , P _ { i } ^ { a } ) =$ mi $\begin{array} { r } { \mathfrak { n } _ { p } \big \lbrace s \mathrm { K L } ( p \| P _ { i } ^ { a } ) + ( 1 - s ) \mathrm { K L } ( p \| P _ { j } ^ { a } ) \big \rbrace = s \mathrm { K L } ( R _ { s } \| P _ { i } ^ { a } ) + ( 1 - s ) \mathrm { K L } ( R _ { s } \| P _ { j } ^ { a } ) } \end{array}$ , attained at $p = R _ { s }$

(c) (Scale optimization is Chernof information.) The tilt optimum is the game’s scale optimization, m $\begin{array} { r } { \operatorname { 1 a x } _ { s \in [ 0 , 1 ] } C _ { s } ( P _ { i } ^ { a } , P _ { j } ^ { a } ) = } \end{array}$ $- \operatorname* { m i n } _ { s } s W _ { s } ( S )$ , and at the optimizer $s ^ { \star }$ the score is neutral in nature’s direction, $\mathbb { E } _ { R _ { s } \star } [ S ] = { \dot { 0 } } ,$ from which the balance

$$
\operatorname* { m a x } _ { \mathbf { s } } C _ { s } ( P _ { i } ^ { a } , P _ { j } ^ { a } ) = \mathrm { K L } ( R _ { s ^ { \star } } \| P _ { i } ^ { a } ) = \mathrm { K L } ( R _ { s ^ { \star } } \| P _ { j } ^ { a } )\tag{A.43}
$$

Consequently the pairwise testing exponent $\Gamma ( a ) = \mathrm { m i n } _ { i \neq j } \mathrm { m a x } _ { s \in [ 0 , 1 ] } C _ { s } ( P _ { i } ^ { a } , P _ { j } ^ { a } )$ is the edge-restricted multi-way coincidence radius of the arm-induced laws $\{ P _ { i } ^ { a } \} _ { i \in [ W ] }$ , and equals the $M A P$ error exponent for identifying the latent hypothesis from repeated pulls of arm $\begin{array} { r } { a , - \operatorname* { l i m } _ { n \to \infty } n ^ { - 1 } \log { \dot { P } _ { e , n } ( a ) } = \Gamma ( a ) \left[ \delta \right] . } \end{array}$

$R _ { s }$ is the log-linear pool of $P _ { i } ^ { a }$ and $P _ { j } ^ { a }$ , and $C _ { s }$ is the two-way mixed-coincidence divergence of [8]; at the optimizer $s ^ { \star }$ the neutrality $\mathbb { E } _ { R _ { s } \star } [ S ] = 0$ is the same equalizing condition the Bellman equalizer enforces. The geometric mean that drives the Bellman game — the Gibbs til $\rho _ { \eta , S } \propto \pi e ^ { \eta S }$ , equilibrium play and gradient of the log-partition — is, on a log-likelihood-ratio score, exactly the Chernof tilt $R _ { s } \propto p _ { i } ^ { s } p _ { j } ^ { 1 - s }$ whose normalizer is the testing divergence. The game’s scale optimization is the Chernof-information tilt optimization, and identification dificulty is the reciprocal scale of the game’s terminal concentration. An arm with large $\Gamma ( a )$ is one whose two closest induced hypotheses have small geometric-mean coincidence $Z ( s ^ { \star } , 1 - s ^ { \star } )$ , i.e. whose concentration game drives the posterior onto the truth fastest, so $\Gamma ( a )$ is an identification score. The remaining modeling step — turning per-pull exponents $\Gamma ( a )$ into a finite-horizon pure-exploration policy with matching sample complexity — is a sequential allocation problem on top of this identity, and adds no further fact about the geometric mean.

Most of the existing algorithmic templates for bandits reduce to estimator-specific games in this framework.

• Explicit-exploration IPS: With $\mu _ { t } = ( 1 - \gamma _ { t } ) p _ { t } + \gamma _ { t } u _ { K }$ and standard inverse-propensity-score (IPS) estimation, the bias vanishes and $\Xi _ { T }$ records the exploration cost.

• Implicit exploration / EXP3-IX: With $\mu _ { t } = p _ { t }$ and estimator $\widehat { c } _ { t } ^ { \mathrm { I X } } ( a ) = u _ { t } ( a ) + \ell _ { t } ( A _ { t } ) \mathbf { 1 } \{ A _ { t } = a \} / ( p _ { t } ( a ) + \gamma _ { t } ) ,$ the conditional mean is $c _ { t } - d _ { t } ^ { \mathrm { I X } }$ where $d _ { t } ^ { \mathrm { I X } } ( a ) = \gamma _ { t } \ell _ { t } ( a ) / ( p _ { t } ( a ) \dot { + } \gamma _ { t } )$ . Total bias is bounded by $\begin{array} { r } { \sum _ { t } K \gamma _ { t } / ( 1 { + } K \gamma _ { t } ) } \end{array}$

• Predictable ofsets: If the estimate uses a control variate or optimistic baseline $m _ { t }$ , only the estimated intrinsic time and the jump term change; the variational front end does not.

• Feedback graphs: For a directed feedback graph $G _ { t } ~ = ~ ( V , E _ { t } )$ , let $o _ { t } ( a )$ be the probability that arm a is observed. The graph-IX estimator $\widehat { c } _ { t } ^ { \mathrm { G - I X } } ( a ) = u _ { t } ( a ) + \ell _ { t } ( a ) \mathbf { 1 } \{ A _ { t } \ \in \ N _ { t } ^ { \mathrm { i n } } ( a ) \} / ( o _ { t } ( a ) + \gamma _ { t } )$ has predictable bias $d _ { t } ^ { \mathrm { G - I X } } ( a ) = \gamma _ { t } \ell _ { t } ( a ) / ( o _ { t } ( a ) + \gamma _ { t } )$ . Both the bias and the natural quadratic surrogates are controlled by the observation ratios $p _ { t } ( a ) / o _ { t } ( a )$ . For undirected graphs these ratios are bounded by the independence number $\alpha ( G _ { t } )$ [3]. Graph topology enters the game only through the observation geometry, leaving the Bellman calculus itself untouched.

Recent feedback-graph best-of-both-worlds algorithms [57] combine explicit exploration with fast stochastic rates. The exact identity explains the mechanism term by term: the cost of playing the exploration distribution $\mu _ { t }$ instead of $p _ { t }$ is recorded in $\Xi _ { T }$ , the graph-IX correction in $\mathrm { B i a s } _ { T } ^ { G - \mathrm { I X } } ( u )$ , and once stochastic gaps make observability ratios large for good actions, the realized intrinsic time stops accumulating.

## A.9 Further results from Section 8

Game A.19 (Expected sampled concentration game). Game 1.1 at a fixed scale $\eta _ { t } \equiv \eta ,$ , with the learner’s move and   
the terminal payof changed. Start from $S _ { 0 } = 0$ . For each round $t = 1 , \ldots , T \colon$   
1. Learner commits a predictable sampling law $w _ { t } \in \Delta ( [ K ] )$ over the experts;   
2. Learner draws an expert $I _ { t } \sim w _ { t }$ and plays it;   
3. Nature, observing $w _ { t }$ but not the drawn $I _ { t } ,$ , picks a centered $z _ { t } \in \mathbb { R } ^ { K }$ with $\langle w _ { t } , z _ { t } \rangle = 0$ and $Q _ { \eta } ( w _ { t } , z _ { t } ) \leq q _ { t } ;$   
4. State advances by $S _ { t } : = S _ { t - 1 } + z _ { t }$   
Nature’s move is thus the move of Game 1.1 with the sampling law in the mixed action’s place. The terminal payof to   
nature is the expected sampled budgeted regret at the committed scale,   
ΦT (w1:T ) := Eh X ct(It) − ⟨wt, ct⟩i + X Wη(St) − Wη(St−1) + T T Γ (A.44)   
t=1 t=1 η   
whose first term is the expected play excess and whose remaining terms are the committed-scale value-to-go $\Gamma / \eta +$   
$W _ { \eta } ( S _ { T } )$ measured from the start.

Theorem A.20 (Posterior sampling is the exact saddle point of Game A.19). For any prior π, budgets (Γ, V ) with $\textstyle \sum _ { t } q _ { t } \leq V ,$ and fixed scale η:

(i) The expected play excess vanishes for every predictable sampling law: $\mathbb { E } [ c _ { t } ( I _ { t } ) - \langle w _ { t } , c _ { t } \rangle \ | \ \mathcal { F } _ { t - 1 } ] = 0$ whenever $I _ { t } \mid \mathcal { F } _ { t - 1 } \sim w _ { t } ,$ , and in particular for the Gibbs sampler $w _ { t } = p _ { t } = \rho _ { \eta , S _ { t - } }$ <sub>1</sub> .

(ii) Posterior sampling is the unique equalizer: with $w _ { t } = p _ { t } .$ , the round increment $W _ { \eta } ( S _ { t } ) - W _ { \eta } ( S _ { t - 1 } ) = \eta Q _ { \eta } ( p _ { t } , z _ { t } )$ is independent of nature’s direction and equals ηq<sub>t</sub> at the budget, so

$$
\mathrm { V a l } ( G a m e A . I 9 ) = \Phi _ { T } ( p _ { 1 : T } ) = \frac { \Gamma } { \eta } + \eta \sum _ { t = 1 } ^ { T } q _ { t }\tag{A.45}
$$

attained simultaneously by the Gibbs sampler (min over $w _ { 1 : T } )$ and by nature saturating each budget in the worst-comparator direction (max), with no order-of-play gap.

(iii) Any sampling law $w _ { t } \neq p _ { t }$ strictly loses: sup Φ $\dot { \boldsymbol { \cdot } } \boldsymbol { T } > \boldsymbol { \Gamma } / \eta + \eta \sum _ { t } q _ { t }$ whenever some $q _ { t } > 0 ,$ so $w _ { t } = p _ { t }$ is then the exact and unique minimizer; in the degenerate case $q _ { t } \equiv 0$ nature is confined to $z _ { t } = 0$ and every sampling law attains the same value $\Gamma / \eta .$

Posterior sampling therefore attains exactly the deterministic concentration-game value in expectation. The strict loss at $w _ { t } \neq p _ { t }$ has a simple source: the value-to-go increment $W _ { \eta } ( S _ { t } ) - W _ { \eta } ( S _ { t - 1 } ) = \eta Q _ { \eta } ( \rho _ { \eta , S _ { t - 1 } } , z _ { t } )$ is η times the centered RCGF of $z _ { t }$ under the Gibbs law $\rho _ { \eta , S _ { t - 1 } }$ , a function of the state alone, whereas nature is constrained only by $Q _ { \eta } ( w _ { t } , z _ { t } ) \leq q _ { t }$ . The two forms then difer, so nature can satisfy its own budget while over-driving the Gibbs increment.

Proposition A.21 (The sampling fluctuation is intrinsic; no fluctuation-free realized posterior-sampling game). Fix any modification of Game A.19 whose payof is an arbitrary measurable function $\Phi ( w _ { 1 : T } , z _ { 1 : T } , I _ { 1 : T } )$ of the realized transcript. The learner draws $I _ { t } \sim w _ { t }$

(i) Exact residual identity. Subtracting (8.2) from (8.1) gives, for every comparator ν simultaneously and every predictable loss sequence,

$$
\widehat { R } _ { T } ^ { c } ( \nu ) - R _ { T } ^ { c } ( \nu ) = M _ { T } ^ { \mathrm { s a m } } , \qquad M _ { T } ^ { \mathrm { s a m } } = \sum _ { t = 1 } ^ { T } \bigl ( c _ { t } ( I _ { t } ) - \langle p _ { t } , c _ { t } \rangle \bigr )\tag{A.46}
$$

so at $w _ { t } = p _ { t }$ the residual $M _ { T } ^ { \mathrm { s a m } }$ is a mean-zero martingale.

(ii) Impossibility of a fluctuation-free realized game. No payof that genuinely depends on the sampled play can be made constant over the sampling randomness (Appendix A.9). A payof built from $c _ { t } ( I _ { t } )$ leaves, on a.e. path, an irreducible per-round variance $\mathrm { V a r } _ { p _ { t } } ( c _ { t } ) > 0$ that no scoring rule removes.

(iii) The high-probability route has a diferent optimizer. Sharpening the sampling law to $w _ { t } ^ { ( \beta ) } \propto p _ { t } ^ { \beta }$ leaves the full-information posterior path unchanged and, by Theorem A.20, leaves the expected payof stationary at $\beta = 1$ But the sampling variance $\begin{array} { r } { V _ { T } ^ { \mathrm { s a m } } ( \beta ) : = \sum _ { t } \mathrm { V a r } _ { w _ { \cdot } ^ { ( \beta ) } } ( c _ { t } ) } \end{array}$ is not stationary there, so, to leading order in a Gaussian approximation of the realized payof, the minimizer of any upper quantile is a law with $\beta ^ { * } \neq 1$ , which posterior sampling is not (Appendix A.9).

At $w _ { t } = p _ { t }$ the realized payof is thus the deterministic minimax value plus a single additive martingale — the same one for every comparator, and for any scoring choice. Were that payof constant over the sampling randomness, it could not depend on $I _ { 1 : T }$ at all and would reduce to the deterministic game, in which the learner plays the mixture $p _ { t }$ and does not sample.

## A.10 Further results from Section 9

## A.10.1 The quantile-margin shortfall in full

Under the second-order schedule (4.9) with $\Gamma = \log ( 1 / \varepsilon )$ and $C = 1 / { \sqrt { 2 } } ,$ the envelope (4.11) lets the ε-quantile margin fall short of twice the average edge $2 \bar { \gamma } _ { T }$ by at most

$$
\begin{array} { r } { \frac { 2 } { T } \Big ( 2 \sqrt { 2 } \sqrt { \log ( 1 / \varepsilon ) V _ { T } ( \ell ) } + \frac { 3 } { 2 } \log ( 1 / \varepsilon ) + ( 1 + \frac { 1 } { \sqrt { 2 } } ) Q _ { T } ^ { * } ( \ell ) \Big ) . } \end{array}
$$

The range bound $Q _ { t } ( \ell ) \leq 1 / 8$ for losses in $[ 0 , 1 ]$ caps $V _ { T } ( \ell )$ at $T / 8 ,$ so the worst-case shortfall is at most $2 \sqrt { \log ( 1 / \varepsilon ) / T }$ plus the additive $O ( \log ( 1 / \varepsilon ) / T )$ edge terms.

The pressure-target identity follows by telescoping the exponential-loss normalizer $\begin{array} { r } { Z _ { t } : = N ^ { - 1 } \sum _ { i } e ^ { - M _ { t } ( i ) } } \end{array}$ , which starts at $Z _ { 0 } = 1$ and ends at $Z _ { T } = L _ { T } ^ { \mathrm { e x p } }$ . Under the local pressure rule of Section 9 each ratio is $Z _ { t } / Z _ { t - 1 } = \left. p _ { t } , e ^ { - \alpha _ { t } g _ { t } } \right. =$ $e ^ { - \alpha _ { t } { \boldsymbol { a } } _ { t } }$ <sup>t</sup> , so $\begin{array} { r } { Z _ { T } = \prod _ { t } e ^ { - \alpha _ { t } a _ { t } } } \end{array}$ and − log $\begin{array} { r } { L _ { T } ^ { \mathrm { e x p } } = \sum _ { t } \alpha _ { t } a _ { t } } \end{array}$ . Combining with the general identity (9.1) and summing the oneround transport identity against ν yields the comparator statement of (9.2). The binary case, where the pressure-optimal $\alpha _ { t }$ is the classical AdaBoost coeficient $\frac { 1 } { 2 } \log ( ( 1 - \varepsilon _ { t } ) / \varepsilon _ { t } )$ with $\begin{array} { r } { \varepsilon _ { t } = \frac { 1 } { 2 } - \gamma _ { t } . } \end{array}$ , is Proposition A.22.

Proposition A.22 (AdaBoost rate from the pressure ledger). For binary weak hypotheses $g _ { t } \in \{ - 1 , + 1 \} ^ { N }$ , take the one-step normalizer minimizer $\begin{array} { r } { \alpha _ { t } ^ { \star } = \frac { 1 } { 2 } \log \frac { 1 - \varepsilon _ { t } } { \varepsilon _ { t } } } \end{array}$ , which is exactly the normalizer-minimizing (equivalently pressure-maximizing) choice $\alpha _ { t } ^ { \star } = \arg \operatorname* { m i n } _ { \alpha }$ log $\langle p _ { t } , e ^ { - \alpha g _ { t } } \rangle$ . With mean margin $\langle p _ { t } , g _ { t } \rangle = 2 \gamma _ { t }$ , the pressure identity (9.2) evaluates the per-round loss in closed form,

$$
\begin{array} { r } { \alpha _ { t } ^ { \star } a _ { t } = - \log \left. p _ { t } , e ^ { - \alpha _ { t } ^ { \star } g _ { t } } \right. = - \frac { 1 } { 2 } \log \left( 1 - 4 \gamma _ { t } ^ { 2 } \right) } \end{array}\tag{A.47}
$$

the Gaussian-channel information at correlation $2 \gamma _ { t } .$ . Summing (9.2) gives the exact exponential-loss ledger

$$
- \log L _ { T } ^ { \mathrm { e x p } } = \sum _ { t = 1 } ^ { T } \alpha _ { t } ^ { \star } a _ { t } = \sum _ { t = 1 } ^ { T } - \frac { 1 } { 2 } \log \bigl ( 1 - 4 \gamma _ { t } ^ { 2 } \bigr )\tag{A.48}
$$

Applying the convexity relaxation $- \log ( 1 - x ) \geq x a t x = 4 \gamma _ { t } ^ { 2 }$ (the same relaxation by which variance and range sit above the centered RCGF) recovers the classical rate as a one-line corollary. Under a uniform edge $\gamma _ { t } \geq \gamma > 0 .$

$$
- \log L _ { T } ^ { \mathrm { e x p } } \ \ge \ 2 \sum _ { t = 1 } ^ { T } \gamma _ { t } ^ { 2 } \ \ge \ 2 \gamma ^ { 2 } T , \qquad \widehat { \mathrm { e r r } } _ { T } = \langle u _ { N } , \mathbf { 1 } \big \{ M _ { T } \le 0 \big \} \rangle \le L _ { T } ^ { \mathrm { e x p } } \le e ^ { - 2 \gamma ^ { 2 } T }\tag{A.49}
$$

the last step being the $\theta = 0$ case of the margin-tail bound above.

## A.11 Further results from Section 11

## A.11.1 The identities read in neighboring languages

The retempering drift is a summation by parts of the log-partition’s η-derivative: since $- \eta \partial _ { \eta } A _ { t } ( \eta ) = \eta ^ { - 1 } \mathrm { K L } ( \rho _ { t , \eta } | | \pi )$ (Proposition C.1), $\begin{array} { r } { D _ { T } = \sum _ { t = 1 } ^ { T - 1 } \bigl ( A _ { t } ( \eta _ { t } ) - A _ { t } ( \eta _ { t + 1 } ) \bigr ) } \end{array}$ is a discrete reading of the path integral $- \int \partial _ { \eta } A _ { t } \ : d \eta _ { t }$ of prior-toposterior relative entropy over the inverse-temperature schedule. Its continuous-time Itô form—Theorem C.2, of which (C.5) is the deterministic reduction—turns the committed Game A and reactive Game B into dual stochastic-control problems on the information manifold.

The Bellman potential $W _ { \eta } ( S )$ is the negative free energy at inverse temperature η with external field $S ,$ so the boosting identity $\begin{array} { r } { \sum _ { t } \alpha _ { t } a _ { t } = - \log L _ { T } ^ { \exp } } \end{array}$ commits its one-step increment round by round, holding pathwise instead of in expectation. The horizon quantities $R _ { T } ^ { \bar { c } } ( \nu ) , P _ { T } , D _ { T } , B _ { T } ( \nu ) , V _ { T } , Q _ { T } ^ { * }$ index the axes along which the game relaxes—rate schedule, comparator complexity, suficient-statistic choice, observation structure, per-round budget, and retempering pattern—so the game is summarized by a small vector of quantities. And bandit algorithms trade the observation-noise terms of (A.33) against one another: EXP3-IX minimizes estimation bias at small cost to the martingale, while Tsallis-INF minimizes entropic information at the cost of more sample-dependent drift.

For an input prior $p _ { 0 }$ that an external process pins, with no freedom to equalize, the operating-information deficit of the redundancy–capacity reading is the exact Bregman divergence generated by the log-partition potential,

$$
\mathsf { C } - I ( p _ { 0 } ) = \mathrm { K L } \big ( m _ { p _ { 0 } } \big | \big | q ^ { \star } \big ) + \sum _ { \theta } p _ { 0 } ( \theta ) \big [ \mathsf { C } - \mathrm { K L } ( P _ { \theta } \| q ^ { \star } ) \big ]\tag{A.50}
$$

between the operating output mixture $\begin{array} { r } { m _ { p _ { 0 } } = \sum _ { \theta } p _ { 0 } ( \theta ) P _ { \theta } } \end{array}$ and the capacity-achieving output $q ^ { \star } \left[ 2 0 \right]$ . The second sum is nonnegative and vanishes exactly when $p _ { 0 }$ is supported on the capacity-active inputs, those with KL $( P _ { \theta } \| q ^ { \star } ) = \mathsf { C } ;$ ; there the deficit is the Bregman divergence alone.

## A.11.2 The exact-value residual at two experts

The two solved slices of the one-round value are $g ( \Gamma ) \eta ^ { - 1 } \mathrm { a r c o s h } ( e ^ { \eta ^ { 2 } q } )$ for two experts at every comparator budget (Proposition A.6) and the root $a _ { K } ( \eta , q )$ of (A.13) for every K at a saturated budget (Proposition A.5). For $K = 2$ with $\pi = \textstyle { \bigl ( } { \frac { 1 } { 2 } } , { \frac { 1 } { 2 } } { \bigr ) }$ the terminal payof is the exact support function $\begin{array} { r } { L _ { \Gamma } ( S ) = \frac { 1 } { 2 } ( S _ { 1 } + S _ { 2 } ) \bar { + } \frac { 1 } { 2 } | S _ { 1 } - S _ { 2 } | g ( \Gamma ) } \end{array}$ , maximized over nature’s reachable states; that set solves a state-dependent recursion whose optimum over-tilts the Gibbs iterate, so Proposition A.6 only upper-bounds the value, with no elementary closed form for $T \geq 2$ . The supremum is finite only once $T$ is fixed, since the per-round centered RCGF is second order in nature’s step.

## A.12 Logarithmic pooling as the same coincidence game

The same geometric-mean identity applies when the simplex is over probabilistic experts. If expert $e \in [ N ]$ reports a predictive distribution $\pi _ { t , e }$ on an outcome space $Y _ { i }$ , and the learner pools the reports geometrically at weights $\alpha _ { t }$

$$
p _ { t } ( y ) = \frac { \prod _ { e = 1 } ^ { N } \pi _ { t , e } ( y ) ^ { \alpha _ { t , e } } } { \sum _ { z \in Y } \prod _ { e = 1 } ^ { N } \pi _ { t , e } ( z ) ^ { \alpha _ { t , e } } }\tag{A.51}
$$

then for the realized outcome $y _ { t }$

$$
- \log p _ { t } ( y _ { t } ) = \sum _ { e = 1 } ^ { N } \alpha _ { t , e } ( - \log \pi _ { t , e } ( y _ { t } ) ) - C _ { \alpha _ { t } } ( \pi _ { t , 1 : N } )\tag{A.52}
$$

where $\begin{array} { r } { C _ { \alpha _ { t } } ( \pi _ { t , 1 : N } ) : = - \log \sum _ { y \in Y } \prod _ { e = 1 } ^ { N } \pi _ { t , e } ( y ) ^ { \alpha _ { t , e } } } \end{array}$ is the coincidence discount. Thus online logarithmic pooling is the same mixed-coincidence game on a diferent simplex: the learner’s log loss is the weighted expert log loss minus a nonnegative coincidence discount.

Any online convex optimization guarantee for the meta-loss $\begin{array} { r } { g _ { t } ( \alpha ) : = - \sum _ { e } \alpha _ { e } \log \pi _ { t , e } ( y _ { t } ) + \log \sum _ { y } \prod _ { e } \pi _ { t , e } ( y ) ^ { \alpha _ { e } } } \end{array}$ yields an online logarithmic-pooling guarantee whose benchmark is stronger than the usual “best convex combination of expert log losses” because of the explicit coincidence discount. The pooling weights $\alpha _ { t }$ can themselves be learned by the same entropic game.

## A.13 Exact relaxation remainders

The relaxation remainders of the RCGF–variance–range hierarchy of Section 3.3, discussed in Section 11.2, are given here in exact form.

Proposition A.23 (Exact relaxation remainders). Let $\kappa _ { n } ( z )$ denote the n-th cumulant of the centered score z under $p$ $( \kappa _ { 2 } = \mathrm { V a r } _ { p } ( z ) )$ , equivalently the centered n-fold coincidence rate of (A.2). Then

$$
Q _ { \eta } ( p , z ) - { \textstyle \frac { 1 } { 2 } } \mathrm { V a r } _ { p } ( z ) = \sum _ { n > 3 } \displaystyle \frac { \eta ^ { n - 2 } } { n ! } \kappa _ { n } ( z ) = \displaystyle \frac { \eta } { 6 } \kappa _ { 3 } ( z ) + \displaystyle \frac { \eta ^ { 2 } } { 2 4 } \kappa _ { 4 } ( z ) + { \cal O } ( \eta ^ { 3 } )\tag{A.53}
$$

and, when $z ( i ) \in [ a , b ]$

$$
{ \frac { \mathrm { r a n g e } ( z ) ^ { 2 } } { 8 } } - Q _ { \eta } ( p , z ) = \Delta \ \geq 0\tag{A.54}
$$

the exact nonnegative Hoefding slack of (A.5). Writing $\Phi _ { t } ^ { ( 1 ) } = { \textstyle \frac { 1 } { 2 } } \mathrm { V a r } _ { p _ { t } } ( z _ { t } )$ and $\Phi _ { t } ^ { ( 2 ) } = \mathrm { r a n g e } ( z _ { t } ) ^ { 2 } / 8$ for the two proxy potentials, the algorithm’s exact gain along a trajectory, from using the true per-round RCGF where proxy j would be used, is $\begin{array} { r l } { \sum _ { t } \eta _ { t } \big ( \Phi _ { t } ^ { ( j ) } - Q _ { t } \big ) } & { { } } \end{array}$ , whichfor the range proxy is $\Delta _ { T } ^ { \mathrm { c l a s s } } o f \left( \mathrm { A } . 5 \right)$ and for the variance proxy $\begin{array} { r } { i s - \sum _ { t } \eta _ { t } \sum _ { n \geq 3 } \eta _ { t } ^ { n - 2 } \kappa _ { n } ( z _ { t } ) / n ! . } \end{array}$

## B Other geometries and domains

In the main development the simplex geometry was fixed: the prior π lives on $[ K ]$ , the comparator class is a relativeentropy ball, and the regularizer is $\eta ^ { - 1 } \mathrm { K L } ( \cdot \| \pi )$ ). The same one-round equilibrium (Theorem 3.2) and the same exact decomposition (Theorem 4.3) survive when the geometry is changed: weighting the entropy per-expert, replacing the prior by a continuum measure, or moving from a finite simplex to a compact convex set in $\bar { \mathbb { R } ^ { d } }$ . In each case the recipe is the same: rewrite the prior anchor and the divergence in the new geometry; the Bellman equalizer is still a Gibbs-type best response; the per-round loss is still a centered RCGF; and the concentration-game bookkeeping persists.

## B.1 Weighted entropy and multiscale experts

Let $\sigma _ { 1 } , \dotsc , \sigma _ { K } \ > \ 0$ and define the weighted entropy geometry $\begin{array} { r } { F _ { \sigma } ( p ) : = \sum _ { i = 1 } ^ { K } \sigma _ { i } p ( i ) \log p ( i ) } \end{array}$ and the weighted divergence $\begin{array} { r } { D _ { \sigma } ( \nu \| p ) : = \sum _ { i = 1 } ^ { K } \sigma _ { i } ( \nu ( i ) \log ( \nu ( i ) / p ( i ) ) - \nu ( i ) + p ( i ) ) } \end{array}$ . The weighted-entropy mirror-descent step $\widetilde { p } _ { t + 1 } ( i ) : = p _ { t } ( i ) e ^ { - \eta _ { t } \ell _ { t } ( i ) / \sigma _ { i } } , \widetilde { p _ { t + 1 } } ( i ) \propto \widetilde { p } _ { t + 1 } ( i ) e ^ { - \lambda _ { t } / \sigma _ { i } } \mathrm { o b e y s }$ the exact one-step identity

$$
\langle p _ { t } , \ell _ { t } \rangle - \langle \nu , \ell _ { t } \rangle = \frac { D _ { \sigma } ( \nu \| p _ { t } ) - D _ { \sigma } ( \nu \| p _ { t + 1 } ) } { \eta _ { t } } + \frac { D _ { \sigma } ( p _ { t } \| p _ { t + 1 } ) } { \eta _ { t } }\tag{B.1}
$$

The same concentration bookkeeping persists after replacing relative entropy by the geometry dictated by the expert scales. Multiscale entropy methods [13, 52] are therefore the same game under a diferent transport geometry; the map $F _ { \sigma }$ is the mirror map introduced with the multiscale experts problem [13], where regret against each expert scales with that expert’s own loss range. Conditional refinements of the same weighted entropy, summed along a tree filtration, drive the entropic-regularization approach to metrical task systems [19], a competitive-analysis setting in which the comparator is dynamic and the learner’s own transport is part of the cost.

The connection to the mixed coincidence identity is direct: the per-expert scales $\sigma _ { i }$ play the role of mixture weights, and $D _ { \sigma }$ replaces relative entropy as the transport divergence.

## B.2 Continuum priors and forgetting mixtures

If $\pi _ { \chi }$ is the exponential-weights prior obtained from a forgetting rate $\chi \in ( 0 , 1 )$ , then geometric pooling over χ produces a mixture of exponential memory kernels:

$$
\rho _ { t } ( i ) \propto \exp \left( - \eta \sum _ { s = 1 } ^ { t - 1 } k _ { t } ( t - 1 - s ) \ell _ { s } ( i ) \right) , \qquad k _ { t } ( u ) = \int \chi ^ { u } \alpha _ { t } ( d \chi )\tag{B.2}
$$

A prior over forgetting rates is therefore exactly a prior over memory kernels. This is the continuum version of mixed coincidence, and it brings hyperparameter averaging inside the same concentration geometry, where it would otherwise sit as an outer model-selection layer.

More generally, if each θ indexes an expert class or a hyperparameter setting and $\pi _ { \theta }$ is the action distribution that would be used under that setting, then the pooled distribution $\begin{array} { r } { p _ { \alpha } ^ { \star } \propto \exp ( \int _ { \Theta } \log \pi _ { \theta } ( x ) \alpha ( d \theta ) ) } \end{array}$ is the exact geometric mixture across the hyperparameter continuum.

The same pooling yields a multi-prior PAC-Bayes penalty. Any PAC-Bayes variational formula whose dependence on the prior enters only through the term $\mathrm { K L } ( \nu \| \pi )$ admits an exact multi-prior version with penalty $\begin{array} { r } { \sum _ { w } \alpha _ { w } \mathrm { K L } ( \nu \| \pi _ { w } ) - } \end{array}$ $C _ { \alpha } ( \pi _ { 1 : W } )$ , where $C _ { \alpha } ( \pi _ { 1 : W } ) = - \log Z _ { \alpha }$ is the coincidence term. This is potentially useful for domain adaptation, meta-learning, or transfer settings with several plausible source priors.

## B.3 Continuous-action online convex optimization

The same exact chain passes from sums to integrals. On a compact convex set $S \subset \mathbb { R } ^ { d }$ with prior measure π, define $\begin{array} { r } { p _ { t } ( d x ) \propto e ^ { - \eta _ { t } F _ { t - 1 } ( x ) } \pi ( d x ) , F _ { t } ( x ) : = \sum _ { s = 1 } ^ { t } f _ { s } ( x ) , x _ { t } : = \int _ { S } x p _ { t } ( d x ) } \end{array}$ . Then the density regret against any posterior measure $\nu \ll$ π satisfies

$$
R _ { T } ^ { \mathrm { { d e n s } } } ( \nu ) = D _ { T } ^ { \mathrm { o c o } } + B _ { T } ^ { \mathrm { o c o } } ( \nu ) + \sum _ { t = 1 } ^ { T } \eta _ { t } Q _ { t } ^ { \mathrm { { o c o } } }\tag{B.3}
$$

Convexity of $f _ { t }$ turns density regret into ordinary OCO regret for the barycenters $x _ { t }$

For a point comparator $x ^ { \star } \in$ arg min $_ { x \in S } F _ { T } ( x )$ interior to $S ,$ and with π the uniform (normalized-Lebesgue) measure on the full-dimensional $S ,$ , the geometric-shrinking measure $\nu _ { \varepsilon , x ^ { \star } }$ uniform on $( 1 - \varepsilon ^ { 1 / d } ) x ^ { \star } + \varepsilon ^ { 1 / d } S$ satisfies $\mathrm { K L } ( \nu _ { \varepsilon , x ^ { \star } } \| \pi ) = \log ( 1 / \varepsilon )$ and $\mathrm { R e g } _ { T } ( x ^ { \star } \bar { ) } \leq \log ( 1 / \varepsilon ) + Q _ { T } ^ { \ast , \bar { \partial } \mathrm { c o } } + ( 2 C + \bar { C } ^ { - 1 } ) \sqrt { V _ { T } ^ { \mathrm { o c o } } \log ( 1 / \varepsilon ) } + \dot { T } \varepsilon ^ { 1 / d }$ . With $\varepsilon = T ^ { - d }$ and $C = 1$ this gives $\mathrm { R e g } _ { T } ( x ^ { \star } ) \leq d \log T + Q _ { T } ^ { * , \mathrm { o c o } } + 3 \sqrt { d V _ { T } ^ { \mathrm { o c o } } \log T } + 1$ . Since $Q _ { t } ^ { \mathrm { o c o } } \leq 1 / 8$ for $f _ { t } \in [ 0 , 1 ] , V _ { T } ^ { \mathrm { o c o } } \leq T / 8$ and $\mathrm { R e g } _ { T } ( x ^ { \star } ) = O ( \sqrt { d T \log T } )$ . This is primarily of structural interest; standard projection-based and dual-averaging methods achieve dimension-independent rates by using $L _ { 2 }$ geometry instead $[ 7 2 , 7 4 ]$

## C Continuous-time limit and Itô-level treatment of the retempering drift

The retempering drift has a deterministic continuous limit whose integrand is a path integral of relative entropy.

Proposition C.1 (Deterministic continuous limit of the retempering drift). Fix the realized cumulative-loss trajectory and the prior π. For every horizon index t and every $\eta > 0 ,$ , the mix loss (4.6) satisfies the exact derivative identity

$$
\partial _ { \eta } A _ { t } ( \eta ) = - \frac { 1 } { \eta ^ { 2 } } \operatorname { K L } \bigl ( \rho _ { t , \eta } \parallel \pi \bigr ) \qquad e q u inu a l e n t l y \qquad - \eta \partial _ { \eta } A _ { t } ( \eta ) = \frac { 1 } { \eta } \operatorname { K L } \bigl ( \rho _ { t , \eta } \parallel \pi \bigr )\tag{C.1}
$$

Consequently $A _ { t } ( \cdot )$ is nonincreasing in $\eta ,$ and for any schedule $( \eta _ { t } )$ the drift admits the summation-by-parts expansion

$$
D _ { T } = \sum _ { t = 1 } ^ { T - 1 } \bigl ( A _ { t } ( \eta _ { t } ) - A _ { t } ( \eta _ { t + 1 } ) \bigr ) = \sum _ { t = 1 } ^ { T - 1 } \frac { \mathrm { K L } ( \rho _ { t , \eta _ { t } } \| \pi ) } { \eta _ { t } ^ { 2 } } \Delta \eta _ { t } + R _ { T } , \qquad \Delta \eta _ { t } : = \eta _ { t + 1 } - \eta _ { t + 1 } .\tag{C.2}
$$

with the explicit second-order remainder bound

$$
| R _ { T } | \leq \frac { 1 } { 2 } \sum _ { t = 1 } ^ { T - 1 } ( \Delta \eta _ { t } ) ^ { 2 } \operatorname* { s u p } _ { \xi \in [ \eta _ { t } \land \eta _ { t + 1 } , \eta _ { t } \lor \eta _ { t + 1 } ] } \left| \partial _ { \eta } ^ { 2 } A _ { t } ( \xi ) \right|\tag{C.3}
$$

If in addition the schedule is sampled from a fixed $C ^ { 2 }$ control $\eta \colon [ 0 , 1 ] \to ( 0 , \infty ) a s \eta _ { t } = \eta ( t / T )$ , and the realized cumulative loss stabilizes in rescaled time $s = t / T$ (a bounded $C ^ { 1 }$ field $\bar { C } \colon [ \bar { 0 } , 1 ] \to \bar { \mathbb { R } } ^ { K }$ with $C _ { t } = \bar { C } ( t / T ) )$ , then $\begin{array} { r } { \operatorname* { s u p } _ { t , T } | \partial _ { \eta } ^ { 2 } A _ { t } | < \infty , } \end{array}$ the remainder is $R _ { T } = O ( 1 / T )$ , and

$$
D _ { T } \ { \xrightarrow { } } \ \int _ { } ^ { } \ - \int _ { 0 } ^ { 1 } \partial _ { \eta } A \ d \eta = \int _ { 0 } ^ { 1 } { \frac { \operatorname { K L } \bigl ( \rho _ { { \bar { C } } ( s ) , \eta ( s ) } \bigr \| \ \pi \bigr ) } { \eta ( s ) ^ { 2 } } } \ \eta ^ { \prime } ( s ) d s\tag{C.4}
$$

a Riemann–Stieltjes integral of the prior-to-posterior relative entropy over the controlled inverse-temperature path.

Identity (C.1) makes the sign rule transparent: the retempering loss per unit of inverse temperature is exactly the rescaled prior-to-posterior relative entropy $\eta ^ { - 2 } \mathrm { K L } ( \rho _ { t , \eta } | | \pi )$ , so warming $( \Delta \eta _ { t } \le 0 )$ is always favorable, and (C.4) exhibits $D _ { T }$ as the relative-entropy work integrated along the inverse-temperature path. $\mathsf { A } s \ \eta \ \downarrow$ 0 the integrand reduces to ${ \scriptstyle \frac { 1 } { 2 } } \mathrm { V a r } _ { \pi } ( \bar { C } ( s ) )$ , the same $\frac { 1 } { 2 } \cdot$ Var small-η limit that governs the second-order schedule (4.9). The optimal-schedule problem then has a deterministic optimal-control reading: along the realized trajectory, minimizing the schedule-dependent part of the regret identity (4.8) over a $C ^ { 2 }$ control $\eta ( \cdot )$ is the optimal-control (Bolza) problem

$$
\operatorname* { m i n } _ { \eta ( \cdot ) } \int _ { 0 } ^ { 1 } \eta ( s ) \dot { V } ( s ) d s + \int _ { 0 } ^ { 1 } \frac { \mathrm { K L } \bigl ( \rho _ { \bar { C } ( s ) , \eta ( s ) } \| \pi \bigr ) } { \eta ( s ) ^ { 2 } } \eta ^ { \prime } ( s ) d s\tag{C.5}
$$

with $\dot { V } ( s )$ the rate of intrinsic-time accumulation, $\eta ( s )$ the controlled parameter, and the relative entropy the running cost; its small-η stationarity recovers $\eta ^ { \star } \approx \sqrt { \Gamma / V }$

Theorem C.2 (Itô-level form of the retempering–drift duality). On the scale–score manifold $\{ ( \eta , S ) : \eta > 0 , S \in \mathbb { R } ^ { K } \}$ the log-partition potential $W _ { \eta } ( S ) = \eta ^ { - 1 } \log \left. \pi , \bar { e } ^ { \eta S } \right.$ has the exact diferential

$$
d W = { \frac { \operatorname { K L } ( \rho _ { \eta , S } \| \pi ) } { \eta ^ { 2 } } } d \eta + \langle \rho _ { \eta , S } , d S \rangle\tag{C.6}
$$

with gradient $\nabla _ { S } W _ { \eta } ( S ) \ = \ \rho _ { \eta , S }$ the Gibbs law $( 2 . 4 ) ,$ scale derivative $\partial _ { \eta } W _ { \eta } ( S ) ~ = ~ \eta ^ { - 2 } \mathrm { K L } ( \rho _ { \eta , S } \| \pi )$ , and Hessian $\nabla _ { S } ^ { 2 } \bar { W _ { \eta } } ( S ) = \eta \mathrm { C o v } _ { \rho _ { \eta , S } } ^ { - } \succeq 0$ (the tilted-variance representation (2.9)). Consequently:

(i) The one-form (C.6) is closed, hence exact on the simply connected manifold, and the terminal collapse (4.14) is its endpoint evaluation, $\begin{array} { r } { \sum _ { t } \eta _ { t } Q _ { t } + D _ { T } = W _ { \eta _ { T } } ( S _ { T } ) } \end{array}$

(ii) For $S$ a continuous $\mathbb { R } ^ { K }$ -semimartingale and η a strictly positive continuous predictable finite-variation process, Itô’s formula for the smooth W collapses to the exact three-leg identity

$$
W _ { \eta _ { a } } ( S _ { u } ) - W _ { \eta _ { 0 } } ( S _ { 0 } ) = \int _ { 0 } ^ { u } \left. \rho _ { \eta _ { s } , S _ { s } } , d S _ { s } \right. + \frac { 1 } { 2 } \int _ { 0 } ^ { u } \eta _ { s } \ \mathrm { t r } ( \mathrm { C o v } _ { \rho _ { \eta _ { s } , S _ { s } } } d [ S ^ { c } ] _ { s } ) + \int _ { 0 } ^ { u } \frac { \mathrm { K L } ( \rho _ { \eta _ { s } , S _ { s } } \Vert \pi ) } { \eta _ { s } ^ { 2 } } d \eta _ { s }\tag{C.7}
$$

whose first leg vanishes along the game’s centered trajectory — there $\langle p _ { t } , z _ { t } \rangle = 0$ and the equalizer play is $p _ { t } = \rho _ { \eta _ { t } , S _ { t - 1 } }$ — and whose second is the intrinsic-time accumulation,

$$
\int _ { 0 } ^ { u } \left. \rho _ { \eta _ { s } , S _ { s } } , d S _ { s } \right. = 0 , \qquad \frac 1 2 \int _ { 0 } ^ { u } \eta _ { s } \mathrm { t r } \big ( \mathrm { C o v } _ { \rho _ { \eta _ { s } } , S _ { s } } d [ S ^ { c } ] _ { s } \big ) = \int _ { 0 } ^ { u } \eta _ { s } d V _ { s }\tag{C.8}
$$

(iii) On a one-step budget $\{ Q _ { \eta } ( p , z ) \leq q \}$ the increment $W _ { \eta } ( S + z ) - W _ { \eta } ( S )$ is independent of nature’s direction, equal to ηq at the budget, if and only $i f p = \rho _ { \eta , S } \ : ( 3 . 3 )$

So (C.7) is the continuous form of the terminal collapse (4.14); the sampling martingale $M ^ { \mathrm { s a m } }$ of (8.1) is not a leg of dW, but the separate additive term in $\widehat { R } _ { T } ^ { c } ( \nu ) = M _ { T } ^ { \mathrm { s a m } } + \langle \nu , S _ { T } \rangle$ . The two Stieltjes legs identify Game A with the predictable control problem mi $\begin{array} { r } { \boldsymbol { 1 } _ { \eta ( \cdot ) } \mathbb { E } \left[ \int \eta \boldsymbol { d } V + \int \eta ^ { - 2 } \mathrm { K L } ( \boldsymbol { \hat { \rho } } _ { \eta , S } \| \boldsymbol { \pi } ) \boldsymbol { d \eta } \right] } \end{array}$ , whose variance-frozen deterministic reduction is the Bolza problem in (C.5).

Exactness is path independence: $W _ { \eta _ { T } } ( S _ { T } ) - W _ { \eta _ { 0 } } ( S _ { 0 } )$ depends only on the endpoints, and the loss leg of fixed-η increments and the drift leg of fixed-S scale switches telescope to that endpoint. Game $\mathrm { A } ,$ in which the scale is committed before the move, and Game $\mathrm { B } ,$ in which it is chosen reactively (Section 4.4), are therefore the two traversal orders of the same exact form and share the identical terminal value. Continuity and finite variation of $\eta$ make the quadratic variation [η] and, by Kunita–Watanabe, the covariation $[ \eta , S ]$ vanish; that drops every scale-side Itô correction and leaves the drift leg a pure Riemann–Stieltjes integral against dη, the continuous form of (C.4). Because the learner enters the loss leg only through the integrand $\nabla _ { S } W = \rho _ { \eta , S } ,$ , the equalizer is to play the gradient of the potential, and the telescoping is the fundamental theorem of calculus for that potential, with no backward recursion. Game B is the Lagrangian dual of the predictable control problem, with the pressure target the multiplier that opens η to activate the one-step constraint. The deterministic continuous limit is therefore rigorous, and the measure-theoretic stochastic interface is the exactness of the one-form $d W$ on the scale–score manifold: Theorem C.2 carries the retempering-drift duality to the Itô level.

## D Proofs

The proofs are grouped by the section their statements serve. The potential-games stabilization of Section 6 is proved for strict-equilibrium-convergent traces (Theorem A.13), with only the convergence of the dynamics itself left open there. The two-sided envelope (Theorem 4.4) brackets the per-round-RCGF loss isolated by the identity (4.8): its upper inequality is derived in full below, and its lower inequality is the AM–GM/online-equalizer bound $\begin{array} { r } { \sum _ { t } \eta _ { t } Q _ { t } \geq 2 C \sqrt { \Gamma V _ { T } } - C ^ { 2 } \Gamma } \end{array}$ (round-by-round equalizing of $\Gamma / \eta + \eta V \geq 2 \sqrt { \Gamma V }$ , Section 4), attained with equality at the one-round initialization witness of Lemma 4.5. The envelope gap is exactly the necessary edge terms $Q _ { T } ^ { \star } ( c ) + C ^ { 2 } \Gamma$ of that lemma, and is no artifact of the analysis. The almost-sure iterated-logarithm statement of Section 8 norms $M _ { T } ^ { \mathrm { s a m } }$ by the exact law-of-the iterated-logarithm scale $\sqrt { 2 V _ { T } ^ { \mathrm { s a m } } }$ log log $\overline { { V _ { T } ^ { \mathrm { s a m } } } }$ . Its fixed-time and anytime forms are derived directly, and the a.s. form is the standard method-of-mixtures bound, valid once the mixing density is supported on the sub-gamma range $\lambda \in ( 0 , 3 )$ on which the exponential process is a supermartingale.

## D.1 Proofs from Section 2

Proof of Proposition 2.1. The centered move $z _ { s } ( i ) = \langle p _ { s } , c _ { s } \rangle - c _ { s } ( i )$ telescopes: $\begin{array} { r } { \sum _ { s = 1 } ^ { t - 1 } z _ { s } ( i ) = \sum _ { s = 1 } ^ { t - 1 } \mu _ { s } - C _ { t - 1 } ( i ) } \end{array}$ which is the first display. For the Bayes form, divide (2.1) numerator and denominator by exp $\begin{array} { r } { \left( - \eta _ { t } \sum _ { s = 1 } ^ { t - 1 } \mu _ { s } \right) } \end{array}$ (a constant in i) to replace $- \eta _ { t } C _ { t - 1 } ( i )$ by $\eta _ { t } S _ { t - 1 } ( i )$ . For the regret identity, $\langle p _ { s } - \nu , c _ { s } \rangle = \langle \nu , \mu _ { s } { \bf 1 } - c _ { s } \rangle = \langle \nu , z _ { s } \rangle$ since $\langle \nu , \mathbf { 1 } \rangle = 1 ;$ summing over s and using $\textstyle \sum _ { s } z _ { s } = S _ { T }$ gives the result. □

Proof of Proposition 2.2. The Gibbs variational identity (2.6) is standard: the supremum on its right-hand side is attained at $q = \rho _ { \eta , S }$ by direct diferentiation, with optimum value $W _ { \eta } ( S )$ . For (2.5), the constrained primal $L _ { \Gamma } ( S ) = \operatorname* { s u p } \{ \langle \nu , S \rangle$ $\nu \in \Delta ( [ K ] ) , \operatorname { K L } ( \nu \| \pi ) \leq \Gamma \}$ is a convex maximization with Slater’s condition (taking $q = \pi$ gives $\mathrm { K L } = 0 < \Gamma$ for $\Gamma > 0 )$ . Writing the Lagrangian $L ( \nu , \lambda ) = \langle \nu , S \rangle - \lambda ( \mathrm { K L } ( \nu \| \pi ) - \Gamma )$ for $\lambda \geq 0$ , strong duality gives

$$
L _ { \Gamma } ( S ) = \operatorname* { i n f } _ { \lambda \geq 0 } \operatorname* { s u p } _ { \nu \in \Delta ( [ K ] ) } L ( \nu , \lambda ) = \operatorname* { i n f } _ { \lambda \geq 0 } \Bigl \{ \lambda \Gamma + \operatorname* { s u p } _ { \nu \in \Delta ( [ K ] ) } \bigl ( \langle \nu , S \rangle - \lambda \mathrm { K L } ( \nu \| \pi ) \bigr ) \Bigr \} .
$$

Setting $\eta : = 1 / \lambda$ (and observing that $\lambda = 0$ gives +∞ unless $S = 0 ,$ , so the infimum is over $\eta > 0 )$ , the inner supremum equals $W _ { \eta } ( S )$ by (2.6), yielding $\begin{array} { r } { L _ { \Gamma } ( S ) = \operatorname* { i n f } _ { \eta > 0 } \{ \Gamma / \eta + W _ { \eta } ( S ) \} } \end{array}$ . The infimum formula holds for every S and $\Gamma \geq 0$ When the comparator constraint is active at the optimum—Γ strictly below li $\operatorname { n } _ { \eta \to \infty } \mathrm { K L } ( \rho _ { \eta , S } \| \pi )$ , the maximal relative entropy reachable by tilting along S—it is attained at the unique finite η solving $\eta ^ { 2 } \partial _ { \eta } W _ { \eta } ( S ) = \mathrm { K L } ( \rho _ { \eta , S } \| \pi ) = \Gamma$ . For constant S (every tilt leaves $\rho _ { \eta , S } = \pi )$ or Γ at or above that limit it is attained as $\eta  \infty$ with value max $_ i S ( i )$ , and for $\Gamma = 0 \mathrm { a s } \eta  0 ^ { + }$ with value $\langle \pi , S \rangle$ □

Proof of Proposition 2.3. (a) The functional $\nu \mapsto \langle \nu , S \rangle - \eta ^ { - 1 } \mathrm { K L } ( \nu \Vert \pi )$ is strictly concave in ν (because KL $( \cdot \| \pi )$ is strictly convex on the open simplex), and $\rho _ { \eta , S }$ has positive entries, so the unique stationary point on the relative interior of $\Delta ( [ K ] )$ is the unique maximizer. Direct diferentiation of $\nu ( i ) \log ( \nu ( i ) / \pi ( i ) ) - \eta \nu ( i ) S ( i )$ subject to $\begin{array} { r } { \sum _ { i } \nu ( i ) = 1 } \end{array}$ gives $\nu ( i ) \propto \pi ( i ) e ^ { \eta S ( i ) }$ , which is $\rho _ { \eta , S } . \left( \mathrm { b } \right)$ The constrained problem min $\mathrm { K L } ( \nu \| \pi )$ subject to $\langle \nu , S \rangle = m _ { \eta }$ has Lagrangian $L ( \nu , \lambda ) = \mathrm { K L } ( \nu \| \pi ) + \lambda ( \langle \nu , \dot { S } \rangle - m _ { \eta } )$ . Stationarity $\partial L / \partial \nu ( i ) = \log ( \nu ( i ) / \pi ( i ) ) + \lambda S ( i ) = 0$ gives $\nu ( i ) \propto \pi ( i ) e ^ { - \lambda S ( i ) }$ The constraint $\langle \nu , S \rangle = m _ { \eta } = \langle \rho _ { \eta , S } , S \rangle$ then forces $\lambda = - \eta , s _ { 0 } \nu = \rho _ { \eta , S } .$ . Strict convexity of KL on the simplex makes the minimizer unique. □

Proof of Proposition 2.4. Let $\begin{array} { r } { \psi ( s ) : = \log \sum _ { i } p ( i ) e ^ { s \eta z ( i ) } } \end{array}$ . Then $\begin{array} { r } { \psi ( 0 ) = 0 , \psi ^ { \prime } ( s ) = \eta \sum _ { i } z ( i ) p _ { s } ^ { ( \eta , z ) } ( i ) } \end{array}$ , and the centering $\langle p , z \rangle = 0$ gives $\psi ^ { \prime } ( 0 ) = 0$ . Diferentiating once more, $\psi ^ { \prime \prime } ( s ) = \eta ^ { 2 } \mathrm { V a r } _ { p _ { \mathrm { e } } ^ { ( \eta , z ) } } ( z )$ . Taylor’s theorem with integral remainder at $\begin{array} { r } { s = 0 \mathrm { y i e l d s } \psi ( 1 ) = \int _ { 0 } ^ { 1 } ( 1 - s ) \psi ^ { \prime \prime } ( s ) d s = \eta ^ { 2 } \int _ { 0 } ^ { 1 } ( 1 - s ) \mathrm { V a r } _ { p _ { \ast } ^ { ( \eta , z ) } } ( z ) } \end{array}$ ds, and dividing by $\eta ^ { 2 }$ produces (2.9). Nonnegativity is immediate from the integrand being nonnegative; equality forces $\mathrm { V a r } _ { p _ { s } ^ { ( \eta , z ) } } ( z ) = 0$ for almost every $s \in ( 0 , 1 )$ , which (continuity in s) forces z to be p-a.s. constant. □

Proof of Corollary 2.5. By Proposition $\begin{array} { r } { 2 . 4 , Q _ { \eta } ( p , z ) = \int _ { 0 } ^ { 1 } ( 1 - s ) \mathrm { V a r } _ { n ^ { ( \eta , z ) } } ( z ) d s } \end{array}$ . As $\eta \downarrow 0 ,$ the tilted measure $p _ { s } ^ { ( \eta , z ) } \to p$ uniformly in $s \in [ 0 , 1 ]$ , so the integrand converges pointwise to $\begin{array} { r } { \big ( 1 - s \big ) \mathrm { V a r } _ { p } ( z ) } \end{array}$ , and dominated convergence (the integrand is bounded by ra $\mathrm { | g e } ( z ) ^ { 2 } / 4 )$ yields $\begin{array} { r } { Q _ { \eta } ( p , z )  \int _ { 0 } ^ { 1 } ( 1 - s ) d s \cdot \mathrm { V a r } _ { p } ( z ) = \frac { 1 } { 2 } \mathrm { V a r } _ { p } ( z ) } \end{array}$ . For the range bound, when $z ( i ) \in [ a , b ]$ , Hoefding’s lemma gives $\operatorname { V a r } _ { p _ { s } ^ { ( \eta , z ) } } ( z ) \leq ( b - a ) ^ { 2 } / 4$ , and integrating $\begin{array} { r } { \int _ { 0 } ^ { 1 } ( 1 - s ) d s = 1 / 2 } \end{array}$ delivers $Q _ { \eta } ( p , z ) \leq ( b - a ) ^ { 2 } / 8 .$ □

Proof of Corollary A.1. Write $\begin{array} { r } { F ( q ) : = \log \sum _ { i } q ( i ) e ^ { \eta c ( i ) } - \eta \left. q , c \right. } \end{array}$ for $q \in \Delta ( [ K ] )$ ; expanding the centered exponent shows $F ( q ) = \eta ^ { 2 } Q _ { \eta } ( q , c - \langle q , c \rangle )$ , the functional of the statement. Both terms of $F ( p _ { \epsilon } )$ are diferentiable in ϵ at 0:

$$
\frac { d } { d \epsilon } \Big \vert _ { 0 + } \log \sum _ { i } p _ { \epsilon } ( i ) e ^ { \eta \epsilon ( i ) } = \frac { e ^ { \eta \epsilon ( j ) } - \sum _ { i } p ( i ) e ^ { \eta \epsilon ( i ) } } { \sum _ { i } p ( i ) e ^ { \eta \epsilon ( i ) } } , \qquad \frac { d } { d \epsilon } \Big \vert _ { 0 + } \eta \left. p _ { \epsilon } , c \right. = \eta \big ( c ( j ) - \left. p , c \right. \big ) = \eta z ( j )
$$

Multiplying numerator and denominator of the first display by $e ^ { - \eta \left. p , c \right. }$ turns it into $e ^ { \eta z ( j ) } / \sum _ { i } p ( i ) e ^ { \eta z ( i ) } - 1 =$ $p _ { 1 } ( j ) / p ( j ) - 1$ , and subtracting the second display gives (A.3). The mean is $\begin{array} { r } { \sum _ { i } p ( j ) \xi _ { \eta } ( j ) = 1 - 1 - \eta \left. p , z \right. = 0 } \end{array}$ . With $r ( j ) : = p _ { 1 } ( j ) / p ( j )$ , the constant shift drops from the variance, so $\mathrm { V a r } _ { p } ( \xi _ { \eta } ) \stackrel { } { = } \mathrm { V a r } _ { p } ( r - \eta z ) = \mathrm { V a r } _ { p } ( r ) + \eta ^ { 2 } \mathrm { V a r } _ { p } ( z ) -$ $2 \eta \mathrm { C o v } _ { p } ( r , z )$ ; here $\operatorname { V a r } _ { p } ( r ) = \chi ^ { 2 } ( p _ { 1 } \| p )$ since $\mathbb { E } _ { p } [ r ] = 1$ , and $\mathrm { C o v } _ { p } ( r , z ) = \mathbb { E } _ { p } [ r z ] = \langle p _ { 1 } , z \rangle$ since $\langle p , z \rangle = 0 .$ . For the bounds, $\mathbb { E } _ { p } [ e ^ { \eta z } ] \geq e ^ { \eta \langle \bar { p , { z } } \rangle } = 1$ by Jensen’s inequality, so $0 < r ( j ) \bar { \leq } e ^ { \eta z ( j ) } \leq \bar { e ^ { \eta b } }$ , and with $z ( j ) \in [ a , b ]$ the two-sided bound on $\xi _ { \eta } ( j ) = r ( j ) - 1 - \eta z ( j )$ follows. □

## D.2 Proofs from Section 3

Proof of Theorem 3.2. The Bellman increment identity $W _ { \eta } ( S + z ) - W _ { \eta } ( S ) = \eta Q _ { \eta } ( \rho _ { \eta , S } , z )$ is direct: by (2.7),

$$
\eta ( W _ { \eta } ( S + z ) - W _ { \eta } ( S ) ) = \log \frac { \sum _ { i } \pi ( i ) e ^ { \eta ( S ( i ) + z ( i ) ) } } { \sum _ { i } \pi ( i ) e ^ { \eta S ( i ) } } = \log \sum _ { i } [ \rho _ { \eta , S } ] _ { i } e ^ { \eta z ( i ) } = \eta ^ { 2 } Q _ { \eta } ( \rho _ { \eta , S } , z ) .
$$

Under the budget $Q _ { \eta } ( p , z ) \leq q _ { t }$ this is bounded by ηq , giving (3.3). For the relaxation property, define $U _ { t }$ as in (3.2) and check that under the learner’s choice $p = \rho _ { \eta , S } \colon U _ { T } ( S + z ) \le U _ { T + 1 } ( S + z ) + \eta q _ { T } = \Gamma / \eta + W _ { \eta } ( S + z ) + \eta q _ { T } \le$ $\Gamma / \eta + W _ { \eta } ( S ) + \eta ( q _ { T - 1 } + q _ { T } ) = U _ { T - 1 } ( S )$ for the previous round (after re-indexing), and the same step iterates back to $\begin{array} { r } { U _ { 1 } ( 0 ) = \Gamma / \eta + W _ { \eta } ( 0 ) + \eta \sum _ { s = 1 } ^ { T } q _ { s } = \Gamma / \eta + \eta \sum _ { s } q _ { s } } \end{array}$ (since $\begin{array} { r } { W _ { \eta } ( 0 ) = \eta ^ { - 1 } \log \sum _ { i } \pi ( i ) = 0 ) } \end{array}$ . The terminal payof is bounded by $L _ { \Gamma } ( S _ { T } ) \le \Gamma / \eta + W _ { \eta } ( S _ { T } ) = U _ { T + 1 } ( S _ { T } )$ by Proposition 2.2, so $\mathrm { V a l } _ { t } ^ { \eta , \Gamma } ( S ) \leq U _ { t } ( S )$ at every node, and the equalizer is the Gibbs distribution by the saturating choice in $W _ { \eta } ( S + z ) - W _ { \eta } ( S )$ □

Proof of Corollary 3.3. By Theorem 3.2, $\begin{array} { r } { \mathrm { V a l } _ { 1 } ^ { \eta , \Gamma } ( 0 ) \le U _ { 1 } ( 0 ) = \Gamma / \eta + \eta \sum _ { t = 1 } ^ { T } q _ { t } \le \Gamma / \eta + \eta V } \end{array}$ . The AM–GM inequality $\Gamma / \eta + \eta V \geq 2 \sqrt { \Gamma V }$ for $\eta , \Gamma , V > 0$ is standard and tight at $\eta = \sqrt { \Gamma / V }$ □

Proof of Proposition 3.4. Fix $\nu \in \Delta ( [ K ] )$ . A direct computation, using (2.7), gives log $: p ^ { + } ( i ) - \log p ( i ) = \eta z ( i ) - $ log $\begin{array} { r } { \sum _ { j } p ( j ) e ^ { \eta z ( j ) } = \eta ( z ( i ) - \eta Q _ { \eta } ( p , z ) ) } \end{array}$ . Multiplying by $\nu ( i )$ and summing,

$$
\mathrm { K L } ( \nu \| p ) - \mathrm { K L } ( \nu \| p ^ { + } ) = \sum _ { i } \nu ( i ) \log \frac { p ^ { + } ( i ) } { p ( i ) } = \eta \left. \nu , z \right. - \eta ^ { 2 } Q _ { \eta } ( p , z ) ,
$$

which rearranges to (3.5). The cumulative form (3.6) is the telescoping sum: $\begin{array} { r } { \sum _ { t = 1 } ^ { T } ( \mathrm { K L } ( \nu \| p _ { t } ) - \mathrm { K L } ( \nu \| p _ { t + 1 } ) ) = } \end{array}$ $\mathrm { K L } ( \nu | | \pi ) - \mathrm { K L } ( \nu | | p _ { T + 1 } )$ at fixed scale. □

Proof of Proposition A.2. (a) The function $\begin{array} { r } { c \mapsto \eta _ { t } ^ { - 2 } \psi _ { p _ { t } } ( - \eta _ { t } ; c ) = \eta _ { t } ^ { - 2 } \log \sum _ { i } p _ { t } ( i ) e ^ { - \eta _ { t } ( c ( i ) - \langle p _ { t } , c \rangle ) } } \end{array}$ is a log-sum-exp in c, hence convex; its sublevel set is closed and convex. (b) Adding b1 to c leaves $c ( i ) - \langle p _ { t } , c \rangle = c ( i ) - \langle p _ { t } , c \rangle + b - b \langle p _ { t } , \mathbf { 1 } \rangle =$ $c ( i ) - \langle p _ { t } , c \rangle$ unchanged, so $\psi _ { p _ { t } }$ is invariant. (c) Constant vectors $c = b \mathbf { 1 }$ have $\psi _ { p _ { t } } ( - \eta _ { t } ; b { \bf 1 } ) = 0$ , so they lie in every sublevel set. (d) The defining expression depends only on $p _ { t } , \eta _ { t } , \beta _ { \mathrm { ~ \scriptsize ~ { ~ \cdot ~ } ~ } }$ and $c ;$ the prior π does not appear. □

Proof of Proposition A.3. For every $p \in \Delta ( [ K ] )$ the move $\zeta ( p )$ is feasible, so $\begin{array} { r } { L _ { \Gamma } ( S + \zeta ( p ) ) \le \operatorname* { s u p } _ { z \in \mathcal { Z } _ { \eta , q } ( p ) } L _ { \Gamma } ( S + z ) } \end{array}$ Taking the infimum over p on both sides preserves the inequality and gives $( \mathrm { A . 1 0 } ) . \mathrm { I f } \ \zeta ( p )$ attains the inner supremum at every $p ,$ the two sides agree pointwise in $p$ and hence after the infimum. The final claim holds because $\langle \nu , S + z \rangle \leq$ $L _ { \Gamma } ( S + z )$ for every ν in the comparator ball. □

Proof of Proposition $A . 4 . \mathrm { \ K L } ( \cdot \| \pi )$ is convex on $\Delta ( [ K ] )$ , so it attains its simplex maximum at a vertex, where its value is max<sub>i</sub> $\log ( 1 / \pi ( i ) ) = \Gamma _ { \mathrm { m a x } }$ . For $\Gamma \geq \Gamma _ { \mathrm { m a x } }$ the constraint $\mathrm { K L } ( \nu \| \pi ) \leq \Gamma$ is therefore vacuous, the supremum in (2.3)

runs over all of $\Delta ( [ K ] )$ , and it is attained at a point mass on an arg max $S ( i )$ , giving (A.11). The terminal payof then does not involve Γ, and neither does the value of the game it terminates. □

Proof of Proposition A.5. Throughout, $L _ { \Gamma } = \operatorname* { m a x } _ { i } ( \cdot )$ by Proposition A.4, and $S _ { 0 } = 0 ,$ , so nature’s payof at the move z is $\operatorname* { m a x } _ { i } z ( i )$ . Write $F ( a , m ) : = \bar { m } e ^ { \eta a } + \left( 1 - m \right) e ^ { \bar { - } \bar { \eta } \dot { m a } / ( 1 - \bar { m } ) }$ for $a \geq 0$ and $m \in ( 0 , 1 )$ , and note $F ( 0 , m ) = 1$ Existence and uniqueness of $a _ { K }$ . The left side of (A.13) is $F ( a , 1 / K )$ , which equals 1 at $a = 0 ,$ , has $\partial _ { a } F ( a , m ) =$ $m \eta \big ( e ^ { \eta a } - e ^ { - \eta m a / ( 1 - m ) } \big ) > \bar { 0 }$ for $a > 0$ at every $\begin{array} { r } { m \in ( 0 , 1 ) { \stackrel { } { - } } \operatorname { a t } \dot { m } = 1 / K , \frac { \eta } { K } \left( \bar { e } ^ { \eta a } - e ^ { - \eta a / ( K - 1 ) } \right) - } \end{array}$ and diverges as $a  \infty ;$ since $e ^ { \eta ^ { 2 } q } > 1$ for $q > 0$ , there is exactly one root $\iota _ { K } > 0$

Upper bound. Let the learner play the uniform u<sub>K</sub>, so that nature’s constraints are $\begin{array} { r } { \sum _ { i } z ( i ) = 0 } \end{array}$ and $\begin{array} { r } { \frac { 1 } { K } \sum _ { i } e ^ { \eta z ( i ) } \leq e ^ { \eta ^ { 2 } q } } \end{array}$ Fix a coordinate j and consider moves with $z ( j ) = a > 0$ . The remaining coordinates satisfy $\begin{array} { r } { \sum _ { i \neq j } z ( i ) = - a _ { 1 } } \end{array}$ , and by convexity of exp the sum $\textstyle \sum _ { i \neq j } e ^ { \eta z ( i ) }$ is minimized subject to that constraint at the common value $z ( i ) = - a / ( K - 1 )$ where it equals $( K - 1 ) e ^ { - \eta a / ( K - 1 ) }$ . Feasibility of $z ( j ) = a$ therefore forces $F ( a , 1 / K ) \leq e ^ { \eta ^ { 2 } q } , s o a \leq a _ { K }$ by the monotonicity just established. The move placing a<sub>K</sub> on one coordinate and $- a _ { K } / ( K - 1 )$ on the others meets both constraints with equality, so $\operatorname* { s u p } _ { z \in { \mathcal { Z } } _ { \eta , q } ( u _ { K } ) }$ max<sub>i</sub> $z ( i ) = a _ { K }$ and $\mathrm { V a l } _ { 1 } \leq a _ { K }$

Monotonicity of $F$ in m. Fix $a > 0$ and set $v : = \eta a / ( 1 - m ) > 0$ , so that the exponent of the second term is $- \eta m a / ( 1 - m ) = - v m$ and $\eta a = v ( 1 - m )$ . Diferentiating,

$$
\partial _ { m } F ( a , m ) = e ^ { \eta a } - e ^ { - v m } - ( 1 - m ) e ^ { - v m } \frac { \eta a } { ( 1 - m ) ^ { 2 } } = e ^ { v ( 1 - m ) } - e ^ { - v m } \Big ( 1 + v \Big ) = e ^ { - v m } \big ( e ^ { v } - 1 - v \big ) > 0 .
$$

Hence $F ( \cdot , m )$ is strictly increasing in a and $F ( a , \cdot )$ strictly increasing in $m ,$ so the root $a ( m )$ of $F ( a , m ) = e ^ { \eta ^ { 2 } q }$ is strictly decreasing in m.

Lower bound. If $p ( j ) = 0$ for some $j ,$ the move z equal to λ on coordinate $j$ and 0 elsewhere satisfies $\langle p , z \rangle = 0$ and $\begin{array} { r } { \sum _ { i } p ( i ) e ^ { \eta z ( i ) } = 1 } \end{array}$ for every $\lambda > 0 ,$ , so nature’s payof $\begin{array} { r } { \operatorname* { m a x } _ { i } z ( i ) = \lambda } \end{array}$ is unbounded and such p is never optimal. For p in the interior, define the response rule ζ: pick $j \in$ arg min $p ( i )$ , set $m : = p ( j ) \in ( 0 , 1 / K ]$ , and let $\zeta ( p )$ place $a ( m )$ on coordinate j and $- m a ( m ) / ( 1 - m )$ on every other coordinate. Then $\begin{array} { r } { \langle p , \zeta ( p ) \rangle = m a ( m ) - ( 1 - m ) \frac { m a ( m ) } { 1 - m } = 0 } \end{array}$ and $\begin{array} { r } { \sum _ { i } p ( i ) e ^ { \eta \zeta ( p ) ( i ) } = F ( a ( m ) , m ) = e ^ { \eta ^ { 2 } q } , \ L { s o } \zeta ( p ) \in \mathcal { Z } _ { \eta , q } ( p ) } \end{array}$ with the budget saturated, and max $\zeta ( p ) ( i ) = a ( m )$ Since $m = \mathrm { m i n } _ { i } p ( i ) \leq 1 / K$ and $a ( \cdot )$ is decreasing, $a ( m ) \geq a ( 1 / K ) = a _ { K }$ . Proposition $\mathsf { A } . 3$ then gives Val<sub>1</sub> ≥ $\begin{array} { r } { \operatorname* { i n f } _ { p } a ( \operatorname* { m i n } _ { i } p ( i ) ) = a _ { K } } \end{array}$ . Combining the two bounds gives (A.12), attained at $p = u _ { K }$ . Neither bound refers to π, which enters only through the hypothesis $\Gamma \geq \Gamma _ { \mathrm { m a x } } .$

Small-budget expansion. Expanding (A.13) at $a = ~ 0$ gives $\begin{array} { l } { { F ( a , 1 / K ) = 1 + { \frac { ( \eta a ) ^ { 2 } } { 2 ( K - 1 ) } } + { \cal { O } } \bigl ( ( \eta a ) ^ { 3 } \bigr ) } } \end{array}$ and $e ^ { \eta ^ { 2 } q } = $ $1 + \eta ^ { 2 } q + O ( q ^ { 2 } )$ , from which $a _ { K } = \sqrt { 2 ( K - 1 ) q } + O ( q )$ □

Proof of Proposition A.6. The value is bracketed from the two sides as in Proposition A.5. For the upper bound let the learner play $\begin{array} { r } { p = \pi = ( \frac { 1 } { 2 } , \frac { 1 } { 2 } ) } \end{array}$ , which at $S = 0$ is also the Gibbs equalizer of Theorem 3.2, and evaluate nature’s best reply, max<sub>z</sub> $L _ { \Gamma } ( z )$ over $z \in \mathcal { Z } _ { \eta , q } ( p )$ . Centering $\langle p , z \rangle = 0 \mathrm { f o r c e s } z = ( \delta , - \delta )$ , and $Q _ { \eta } ( p , z ) = \eta ^ { - 2 }$ log cosh $( \eta \delta )$ is even and strictly increasing in $| \delta |$ , so the budget binds at nature’s best response: log cosh $\iota ( \eta \delta ) = \eta ^ { 2 } q , \mathrm { i . e . } \delta ^ { \star } = \eta ^ { - 1 }$ arcosh $. ( e ^ { \eta ^ { 2 } q } )$ For the terminal payof, $\langle \nu , ( \delta , - \delta ) \rangle = \delta ( 2 \nu _ { 1 } - 1 )$ , so for $\delta > 0$ the supremum defining $L _ { \Gamma }$ drives $\nu _ { 1 }$ to the boundary of the relative-entropy ball. With π uniform, $\mathrm { K L } ( \nu | | \pi ) = \log 2 - h ( \nu _ { 1 } )$ with h the binary entropy, so the ball is $\begin{array} { r } { \{ \nu _ { 1 } : \log 2 - h ( \nu _ { 1 } ) \le \Gamma \} = \big [ \frac { 1 - g ( \Gamma ) } { 2 } , \frac { 1 + g ( \Gamma ) } { 2 } \big ] } \end{array}$ with $g ( \Gamma )$ as in (A.14). The maximizer is $\begin{array} { r } { \nu _ { 1 } = \frac { 1 + g ( \Gamma ) } { 2 } } \end{array}$ , giving ${ \cal L } _ { \Gamma } \big ( ( \delta , - \delta ) \big ) = \delta g ( \Gamma )$ ; the move $\delta < 0$ gives the equal value $| \delta | g ( \Gamma )$ by symmetry of π. Evaluating at $\delta ^ { \star }$ yields the upper bound $\mathrm { V a l _ { 1 } } ^ { \cdot } \le g ( \Gamma ) \delta ^ { \star }$

For the matching lower bound, apply Proposition A.3 with the response rule of Proposition A.5: against p with $\begin{array} { r } { m : = \operatorname* { m i n } _ { i } p ( i ) \leq \frac { 1 } { 2 } } \end{array}$ attained at coordinate j, nature plays a(m) on j and $- m a ( m ) / ( 1 - m )$ on the other coordinate, where $a ( m )$ solves $F ( a , m ) = e ^ { \eta ^ { 2 } q }$ (a boundary p again concedes an unbounded payof). Writing $b : = m a ( m ) / ( 1 - m ) \in$ $[ 0 , a ( m ) ]$ ], the terminal support function $L _ { \Gamma }$ evaluates to

$$
L _ { \Gamma } \big ( \zeta ( p ) \big ) = \frac { a ( m ) - b } { 2 } + \frac { a ( m ) + b } { 2 } g ( \Gamma ) \geq a ( m ) g ( \Gamma ) ,
$$

the inequality because $g ( \Gamma ) \ \leq \ 1$ and $b \ \leq \ a ( m )$ . Since $a ( \cdot )$ is decreasing and $m \le \frac 1 2 , a ( m ) \ge a ( \frac 1 2 ) = \delta ^ { \star }$ , so $\mathrm { V a l } _ { 1 } \geq g ( \Gamma ) \delta ^ { \star }$ and (A.15) follows. Theorem 3.2 is not what makes the Gibbs play optimal here: it supplies the relaxation $U _ { t } ,$ , and the equalizer it identifies is minimax at $S = 0$ under a uniform prior but not under a general one (Proposition A.5). For (A.16), Corollary 3.3 gives $\mathrm { V a l } _ { 1 } \le \Gamma / \eta + \eta q ;$ writing $t : = \eta ^ { 2 } q > 0$ , the gap equals ${ \boldsymbol { \eta } } ^ { - 1 } { \boldsymbol { f } } ( t )$ with $f ( t ) : =$ $\Gamma + t - g ( \Gamma ) \mathrm { a r c o s h } ( e ^ { t } )$ by (A.15). Here $f ( 0 ) = \Gamma > 0$ and $f ^ { \prime } ( t ) = 1 - g ( \Gamma ) / \sqrt { 1 - e ^ { - 2 t } }$ increases from −∞ (as $t \to 0 ^ { + } )$ to $1 - g ( \Gamma )$ (as $t \to \infty )$ . If $\Gamma \geq \log 2$ then $g ( \Gamma ) = 1$ , so $f ^ { \prime } < 0$ throughout and $f$ decreases strictly to its infimum $\Gamma - \log 2$ , approached only as $t \to \infty$ (since arcosh $( e ^ { t } ) = t + \log 2 + o ( 1 ) )$ ; hence $f ( t ) > \Gamma - \log 2 \geq 0$ at every finite t and the bound is strict. If Γ < log 2 then $g ( \Gamma ) < 1$ , so $f ^ { \prime }$ vanishes exactly once, at the interior minimizer $\begin{array} { r } { t _ { \star } = - \frac { 1 } { 2 } \log \bigl ( 1 - g ( \Gamma ) ^ { 2 } \bigr ) } \end{array}$ , where $e ^ { t _ { \star } } = ( 1 - g ( \Gamma ) ^ { 2 } ) ^ { - 1 / 2 }$ and arcosh $\begin{array} { r } { ( e ^ { t _ { \star } } ) = \frac { 1 } { 2 } \log \frac { 1 + g ( \Gamma ) } { 1 - g ( \Gamma ) } } \end{array}$ ; substituting and using the definition $\begin{array} { r } { \Gamma = \frac 1 2 ( 1 + g ( \Gamma ) ) \log ( 1 + g ( \Gamma ) ) + \frac 1 2 ( 1 - g ( \Gamma ) ) \log ( 1 - g ( \Gamma ) ) } \end{array}$ of $g ( \Gamma )$ via (A.14) gives $f ( t _ { \star } ) = 0$ . Thus $f ( t ) \geq 0$ with equality exactly at $t = t _ { \star } , s o \left( \mathrm { A } . 1 6 \right)$ is strict except at the single scale $\eta ^ { 2 } q = t _ { \star }$

For (A.17), set $\eta _ { \mathrm { B } } = \sqrt { \Gamma / q }$ so $\eta _ { \mathrm { R } } ^ { 2 } q = \Gamma$ and $2 \sqrt { \Gamma q } = 2 \Gamma / \eta _ { \mathrm { B } }$ . Then (A.15) gives $\mathrm { V a l _ { 1 } } = \eta _ { \mathrm { B } } ^ { - 1 } g ( \Gamma )$ arcosh $( e ^ { \Gamma } )$ , and dividing by $2 \Gamma / \eta _ { \mathrm { B } }$ gives the ratio $g ( \Gamma )$ arcosh $( e ^ { \Gamma } ) / ( 2 \Gamma ) = \kappa ( \Gamma )$ , independent of $q . \mathrm { A t } \eta = \eta _ { \mathrm { B } }$ the scale is $t = \Gamma$ , which is not the equality scale $( t _ { \star } > \Gamma$ for $\Gamma < \log 2 _ { \mathrm { : } }$ , and the bound is strict for $\Gamma \geq \log 2 )$ , so $f ( \Gamma ) > 0$ and hence $\kappa ( \Gamma ) < 1 ;$ and $\kappa ( \Gamma )  1$ as $\Gamma  0 ^ { + }$ follows from $g ( \Gamma ) \sim \sqrt { 2 \Gamma }$ and arcosh $( e ^ { \Gamma } ) \sim \sqrt { 2 \Gamma }$ □

## D.3 Proofs from Section 4

Proof of Theorem 4.3. Apply Proposition 3.4 round-by-round, taking $\eta = \eta _ { t } , p = p _ { t } = \rho _ { t - 1 , \eta _ { t } } , z = - c _ { t } + \langle p _ { t } , c _ { t } \rangle \mathbb { 1 }$ 1 (the centered version), and $p ^ { + } = \widetilde { p } _ { t + 1 } = \rho _ { t , \eta _ { t } }$ . Summing,

$$
R _ { T } ^ { c } ( \nu ) = \sum _ { t = 1 } ^ { T } \eta _ { t } Q _ { t } ( c ) + \sum _ { t = 1 } ^ { T } \frac { \mathrm { K L } ( \nu \| \rho _ { t - 1 , \eta _ { t } } ) - \mathrm { K L } ( \nu \| \rho _ { t , \eta _ { t } } ) } { \eta _ { t } } .
$$

The relative-entropy telescope is not directly summable because the second argument uses $\eta _ { t }$ at time t but $\eta _ { t + 1 }$ would be needed at time $t + 1$ . Write $\begin{array} { r } { A _ { t } ( \eta ) : = - \eta ^ { - 1 } \log \sum _ { i } \pi ( i ) e ^ { - \eta C _ { t } ( i ) } } \end{array}$ for the scaled log-partition at horizon t. Expanding $\begin{array} { r } { \mathrm { K L } ( \nu \| \rho _ { t , \eta } ) = \mathrm { K L } ( \nu \| \pi ) + \eta \left. \nu , C _ { t } \right. + \log \sum _ { i } \pi ( j ) e ^ { - \eta C _ { t } ( j ) } } \end{array}$ and rearranging yields the retempering identity

$$
\eta ^ { - 1 } \bigl ( \mathrm { K L } ( \nu \| \pi ) - \mathrm { K L } ( \nu \| \rho _ { t , \eta } ) \bigr ) = A _ { t } ( \eta ) - \langle \nu , C _ { t } \rangle\tag{D.1}
$$

Applying (D.1) at $( t - 1 , \eta _ { t } )$ and $( t , \eta _ { t } )$ and subtracting gives $\eta _ { t } ^ { - 1 } ( \mathrm { K L } ( \nu \| \rho _ { t - 1 , \eta _ { t } } ) - \mathrm { K L } ( \nu \| \rho _ { t , \eta _ { t } } ) ) = ( A _ { t } ( \eta _ { t } ) -$ $A _ { t - 1 } ( \eta _ { t } ) ) - \langle \nu , c _ { t } \rangle$ . Summing in $t ,$ the A-diferences telescope with retempering: $\begin{array} { r } { \sum _ { t = 1 } ^ { T } ( A _ { t } ( \eta _ { t } ) - A _ { t - 1 } ( \eta _ { t } ) ) \ = } \end{array}$ $\begin{array} { r }  A _ { T } ( \eta _ { T } ) - A _ { 0 } ( \eta _ { 1 } ) + \sum _ { t = 1 } ^ { T - 1 } ( A _ { t } ( \eta _ { t } ) - A _ { t } ( \eta _ { t + 1 } ) ) = A _ { T } ( \eta _ { T } ) + D _ { T }  \end{array}$ , using $A _ { 0 } ( \eta _ { 1 } ) = - \eta _ { 1 } ^ { - 1 } \log { \sum _ { i } \pi ( i ) } = 0 . \mathrm { ~ A ~ }$ final use of (D.1) at $( T , \eta _ { T } )$ converts $A _ { T } ( \eta _ { T } ) - \langle \nu , C _ { T } \rangle = B _ { T } ( \nu )$ . Finally, $D _ { T } \leq 0$ for nonincreasing schedules because $A _ { t } ( \eta )$ is nonincreasing in $\eta ,$ visible from the variational form $\begin{array} { r } { A _ { t } ( \eta ) = \operatorname* { m i n } _ { \nu \in \Delta ( [ K ] ) } \{ \langle \nu , C _ { t } \rangle + \eta ^ { - 1 } \mathrm { K L } ( \nu \| \pi ) \} } \end{array}$ (minimizing over a Lagrangian whose constraint relaxes as $\eta ^ { - 1 }$ grows). □

Proof of Proposition 4.2. At constant scale $D _ { T } = 0 ,$ so Proposition 4.7 gives sup $\nu { : \mathrm { K L } } ( \nu | | \pi ) { \leq } \Gamma ^ { \mathit { R } _ { T } ^ { c } ( \nu ) } = { L } _ { \Gamma } ( S _ { T } )$ and $\textstyle \sum _ { t = 1 } ^ { T } \eta Q _ { t } = W _ { \eta } ( S _ { T } )$ , and (A.18) rewrites the first as $\begin{array} { r } { L _ { \Gamma } ( S _ { T } ) = W _ { \eta } ( S _ { T } ) + \eta ^ { - 1 } A _ { \Gamma } ( \rho _ { \eta , S _ { T } } ) } \end{array}$ . Successive Gibbs iterates satisfy $p _ { t + 1 } ( i ) = G _ { \eta } ( S _ { t - 1 } + z _ { t } ) ( i ) \propto p _ { t } ( i ) e ^ { \eta z _ { t } ( i ) }$ , so the transport identity $\eta ^ { 2 } Q _ { \eta } ( p , z ) = \mathrm { K L } ( p \| p ^ { + } )$ reads $\eta ^ { 2 } Q _ { t } =$ $\mathrm { K L } ( p _ { t } \| p _ { t + 1 } )$ , with $p _ { 1 } = G _ { \eta } ( 0 ) = \pi$ and $p _ { T + 1 } = \rho _ { \eta , S _ { 1 } }$ . Substituting both into $L _ { \Gamma } ( S _ { T } ) = W _ { \eta } ( S _ { T } ) + \eta ^ { - 1 } A _ { \Gamma } ( \rho _ { \eta , S _ { T } } ) =$ $\begin{array} { r l } { \sum _ { t } \eta Q _ { t } + \eta ^ { - 1 } A _ { \Gamma } ( p _ { T + 1 } ) } \end{array}$ gives the bracket of (A.20), and the budget $\begin{array} { r } { \sum _ { t } Q _ { t } \leq V \mathrm { i s } \sum _ { t } \mathrm { K L } ( p _ { t } \| \dot { p } _ { t + 1 } ) \leq \eta ^ { 2 } V } \end{array}$ . Conversely every chain of full-support distributions is nature’s play for a feasible move sequence, realized by the centered moves $z _ { t } : = \eta ^ { - 1 } \lceil \log ( p _ { t + 1 } / p _ { t } ) - \langle p _ { t } , \log ( p _ { t + 1 } / p _ { t } ) \rangle \mathbf { 1 } \rceil$ . This proves (A.20).

Both bracket terms are separately capped, the first at $\eta ^ { 2 } V$ by the budget and the second at Γ by (A.18), so the supremum is at most $\eta ^ { 2 } V + \Gamma$ . Equality forces both caps. For the second, $A _ { \Gamma } ( p ) = \Gamma$ holds if and only if $\mathrm { K L } ( p \Vert \pi ) = \Gamma$ : taking $\nu = p$ gives $A _ { \Gamma } ( p ) \geq \mathrm { K L } ( p \Vert \pi )$ , while a maximizer ν attaining Γ must have $\mathrm { K L } ( \nu \| \pi ) = \Gamma$ and $\mathrm { K L } ( \nu \| p ) = 0$ , hence $p = \nu .$ This is the stated criterion

For the chain totals, $W _ { \eta }$ is convex with $\nabla W _ { \eta } = G _ { \eta } ,$ and centering makes each round’s increment a Bregman divergence of $W _ { \eta } ,$

$$
\mathrm { K L } ( p _ { t } \| p _ { t + 1 } ) = \eta \big [ W _ { \eta } ( S _ { t } ) - W _ { \eta } ( S _ { t - 1 } ) - \langle G _ { \eta } ( S _ { t - 1 } ) , S _ { t } - S _ { t - 1 } \rangle \big ]
$$

whose linear term is $\langle p _ { t } , z _ { t } \rangle = 0 .$ . Summing, a chain accumulates $\eta W _ { \eta } ( S _ { T } ) \mathrm { . }$ : the increment of $W _ { \eta }$ along the path, less a left-endpoint Riemann sum of $\boldsymbol { \int } \left. \boldsymbol { \nabla } W _ { \eta } , d \boldsymbol { S } \right.$ that centering annihilates term by term. The endpoint distribution does not pin this, because $S _ { T }$ still ranges over the level set $\{ S : G _ { \eta } ( S ) = p _ { T + 1 } \} = \{ S ^ { \dagger } + c \mathbf { 1 } : c \in \mathbb { R } \}$ , along which $W _ { \eta }$ takes every value $W _ { \eta } ( S ^ { \dagger } ) + c .$

At $T = 1$ the single move is centered under $p _ { 1 } = \pi$ , which forces $\langle \pi , S _ { 1 } \rangle = 0$ and so selects one point of that level set, leaving the one value $\mathrm { K L } ( \pi \| p _ { 2 } )$ ; this gives the stated $T = 1$ set. At $T \geq 2$ at least one interior point of the chain is free, and it ranges over a connected set. With $p _ { T + 1 }$ fixed the total depends continuously on that point, attains a minimum $c _ { T }$ , and grows without bound as $| z _ { 1 } |$ does, so by the intermediate value theorem the achievable totals sweep the interval $\lbrack c _ { T } , \infty )$ . Appending a null move $z _ { T + 1 } = 0$ , which is feasible and contributes nothing, embeds every T-step chain in a $( T { + } 1 )$ -step one, so $c _ { T }$ is nonincreasing; and $c _ { T } \to 0 ,$ since following the path in $T$ steps of size $O ( 1 / T )$ accumulates

$O ( 1 / T )$ in total by the local quadraticity of the divergence. Taking $T _ { 0 }$ with $c _ { T _ { 0 } } \leq \eta ^ { 2 } V$ , which exists because $c _ { T }  0 ,$ nature uses the budget in full on a chain ending anywhere on the sphere, and (4.5) follows. □

Proof of Theorem 4.4. Set $\textstyle V _ { t } : = \sum _ { s = 1 } ^ { t } Q _ { s } .$

The schedule $( 4 . 9 )$ is nonincreasing, since $V _ { t - 1 }$ is nondecreasing in t, so $D _ { T } \leq 0$ by Theorem 4.3. For the intrinsic-time loss, write $\phi ( v ) : = \operatorname* { m i n } \{ 1 , C \sqrt { \Gamma / v } \}$ with $\phi ( 0 ) : = 1$ , so that $\eta _ { t } = \phi ( V _ { t - 1 } )$ and the loss $\begin{array} { r } { \sum _ { t } \eta _ { t } Q _ { t } = \sum _ { t } \phi ( V _ { t - 1 } ) ( V _ { t } - } \end{array}$ $V _ { t - 1 } )$ is the left-endpoint sum of the nonincreasing ϕ against the increments of $V _ { t } .$ Per round,

$$
\phi ( V _ { t - 1 } ) Q _ { t } - \int _ { V _ { t - 1 } } ^ { V _ { t } } \phi ( v ) d v = \int _ { V _ { t - 1 } } ^ { V _ { t } } \left( \phi ( V _ { t - 1 } ) - \phi ( v ) \right) d v \leq Q _ { t } \left( \phi ( V _ { t - 1 } ) - \phi ( V _ { t } ) \right) ,
$$

and the brackets are nonnegative and telescope, so

$$
\sum _ { t } \eta _ { t } Q _ { t } \ \leq \ \int _ { 0 } ^ { V _ { T } } \phi ( v ) d v + Q _ { T } ^ { \star } ( c ) \left( \phi ( 0 ) - \phi ( V _ { T } ) \right) \ \leq \ \int _ { 0 } ^ { V _ { T } } \phi ( v ) d v + Q _ { T } ^ { \star } ( c ) .
$$

The integral is $V _ { T }$ for $V _ { T } \leq C ^ { 2 } \Gamma$ and $2 C \sqrt { \Gamma V _ { T } } - C ^ { 2 } \Gamma$ for $V _ { T } \geq C ^ { 2 } \Gamma _ { }$ ; in both cases it is at most $2 C \sqrt { \Gamma V _ { T } }$ . Hence

$$
\sum _ { t } \eta _ { t } Q _ { t } \ \leq \ 2 C \sqrt { \Gamma V _ { T } ( c ) } + Q _ { T } ^ { \star } ( c ) ,
$$

the stated upper side of (4.10). For the terminal term, $\eta _ { T } \ge \operatorname* { m i n } \{ 1 , C \sqrt { \Gamma / V _ { T } } \}$ gives $\Gamma / \eta _ { T } \le \Gamma + C ^ { - 1 } \sqrt { \Gamma V _ { T } ( c ) }$ Adding the loss bound, $D _ { T } \leq 0$ , and the terminal term yields $R _ { T } ^ { c } ( \nu ) \leq \Gamma + Q _ { T } ^ { \star } ( c ) + ( 2 C + C ^ { - 1 } ) \sqrt { \Gamma V _ { T } ( c ) }$ , the stated envelope (4.11).

For the matching lower envelope, the clipped phase $T _ { \mathrm { c } }$ is the initial segment $\{ 1 , \ldots , t _ { 0 } \}$ , since $V _ { t - 1 }$ is nondecreasing in t; there $\eta _ { t } = 1$ , so $\textstyle \sum _ { t \in T _ { \mathrm { c } } } \eta _ { t } Q _ { t } = V _ { t _ { 0 } }$ . On the unclipped phase $T _ { \mathrm { u } } = \{ t _ { 0 } + 1 , . . . , T \} , \eta _ { t } = C \sqrt { \Gamma / V _ { t - 1 } }$ with $V _ { t - 1 } \geq V _ { t _ { 0 } } > 0 ;$ ; because $v \mapsto v ^ { - 1 / 2 }$ is decreasing, its value at the left endpoint dominates the interval, $Q _ { t } / \sqrt { V _ { t - 1 } } \geq$ $\int _ { V _ { t - 1 } } ^ { V _ { t } } v ^ { - 1 / 2 } d v$ , and summing telescopes to $\begin{array} { r } { \sum _ { t \in T _ { \mathrm { u } } } Q _ { t } / \sqrt { V _ { t - 1 } } \ge \int _ { V _ { t _ { 0 } } } ^ { V _ { T } } v ^ { - 1 / 2 } d v = 2 ( \sqrt { V _ { T } } - \sqrt { V _ { t _ { 0 } } } ) } \end{array}$ , from which $\begin{array} { r } { \sum _ { t \in T _ { \mathrm { u } } } \eta _ { t } Q _ { t } \ge 2 C \sqrt { \Gamma V _ { T } } - 2 C \sqrt { \Gamma V _ { t _ { 0 } } } . } \end{array}$ Adding the clipped phase and completing the square,

$$
\sum _ { t = 1 } ^ { T } \eta _ { t } Q _ { t } \ge V _ { t _ { 0 } } - 2 C \sqrt { \Gamma V _ { t _ { 0 } } } + 2 C \sqrt { \Gamma V _ { T } } = \left( \sqrt { V _ { t _ { 0 } } } - C \sqrt { \Gamma } \right) ^ { 2 } - C ^ { 2 } \Gamma + 2 C \sqrt { \Gamma V _ { T } } \ge 2 C \sqrt { \Gamma V _ { T } } - C ^ { 2 } \Gamma ,
$$

the lower inequality of (4.10); the constant is met with equality at the one-round initialization witness of Lemma 4.5 $( T _ { \mathrm { u } } = \emptyset , V _ { t _ { 0 } } = C ^ { 2 } \Gamma )$ . The matching one-round witnesses for the edge terms $C ^ { 2 } \Gamma$ (initialization overhead) and $Q _ { T } ^ { \star }$ (single-jump term) are recorded in Lemma 4.5: a single round with $V _ { 0 } = 0$ and $q _ { 1 } = C ^ { 2 } \Gamma$ attains $C ^ { 2 } \Gamma$ exactly, and a single round with $V _ { 0 } = 0$ and $q _ { 1 } = q \gg \Gamma$ contributes q while $2 C { \sqrt { \Gamma q } } = o ( q )$ , forcing the $Q _ { T } ^ { \star }$ overhead. □

Proof of Lemma 4.5. The $C ^ { 2 } \Gamma$ term is an initialization overhead caused by clipping $\eta _ { 1 }$ at 1: when $V _ { 0 } = 0$ the schedule sets $\eta _ { 1 } = 1$ , and even a single round with $q _ { 1 } = C ^ { 2 } \Gamma$ yields $\eta _ { 1 } q _ { 1 } = C ^ { 2 } \Gamma$ , while $\sqrt { \Gamma V _ { 1 } } = \sqrt { \Gamma \cdot C ^ { 2 } \Gamma } = C \Gamma ,$ so the lower envelope $2 C \sqrt { \Gamma V _ { 1 } } - C ^ { 2 } \bar { \Gamma } = C ^ { 2 } \Gamma$ equals the realized loss and the $C ^ { 2 } \Gamma$ constant is attained exactly in this one-round witness. The $Q _ { T } ^ { \star }$ term is forced because a single large jump $Q _ { 1 } = q \gg \Gamma$ gives loss $\eta _ { 1 } Q _ { 1 } = q ( \mathrm { s i n c e } \ : \eta _ { 1 } = 1 )$ while $2 C { \sqrt { \Gamma q } } = o ( q )$ as $q / \Gamma \to \infty _ { \mathrm { : } }$ , so the $Q _ { T } ^ { \star }$ overhead cannot be avoided when a small fraction of the total intrinsic time concentrates in a single round. Both witnesses are one-round sequences, establishing the necessity of both edge terms.

Proof of Proposition A.7. Let $\begin{array} { r } { \Phi ( \eta ) : = \eta ^ { - 1 } \log \sum _ { i } p ( i ) e ^ { \eta z ( i ) } } \end{array}$ for $\eta > 0 _ { \mathrm { { ; } } }$ , and let $p _ { \eta } ( i ) \propto p ( i ) e ^ { \eta z ( i ) }$ denote the tilted measure. L’Hôpital gives $\Phi ( 0 ^ { + } ) = \left. p , z \right.$ (which equals 0 when z is centered, the case treated by the proposition; the analysis applies in general after subtracting $\langle p , z \rangle$ from both sides of $\left( \mathsf { A } . 1 9 \right) )$ ). For non-degenerate z (not p-a.s. constant), Φ is strictly increasing on $( 0 , \infty )$ with li $\begin{array} { r } { \mathrm { n } _ { \eta  0 ^ { + } } \Phi ( \eta ) = \langle p , z \rangle } \end{array}$ and $\begin{array} { r } { \operatorname* { l i m } _ { \eta \to \infty } \Phi ( \eta ) = \operatorname* { m a x } _ { i \in \mathrm { s u p p } ( p ) } z ( i ) } \end{array}$ (coordinates with $p ( i ) = 0$ contribute nothing to $\mathbb { E } _ { p } [ e ^ { \eta z } ]$ , so the limit is the maximum over the support of p). The right-hand side b is constant in η; the equation $\Phi ( \eta ) = b$ therefore admits a unique solution in $( 0 , \infty )$ for any $b \in ( \langle p , z \rangle$ ， $\operatorname* { m a x } _ { i \in \mathrm { s u p p } ( p ) } z ( i ) )$ , by intermediatevalue and strict monotonicity. Strict monotonicity follows from the derivative $\begin{array} { r } { \Phi ^ { \prime } ( \eta ) = \eta ^ { - \bar { 2 } } \bigl ( \eta \mathbb { E } _ { p _ { \eta } } [ z ] - \log \mathbb { E } _ { p } [ e ^ { \eta z } ] \bigr ) } \end{array}$ which is positive whenever z is not p-a.s. constant. (Equivalently, $\eta ^ { 2 } \Phi ^ { \prime } ( \eta ) = \eta \mathbb { E } _ { p _ { \eta } } [ z ] - \log \bar { \mathbb { E } } _ { p } [ e ^ { \bar { \eta } \bar { z } } ]$ vanishes at $\eta = 0$ and has positive derivative η $\mathrm { \Delta } \mathrm { V a r } _ { p _ { \eta } } ( z )$ in η, by direct diferentiation of the log-MGF and its first moment.) □

Proof of Theorem 4.6. By the active-scale Equation (A.19), log $\begin{array} { r } { \sum _ { i } p _ { t } ( i ) e ^ { \eta _ { t } z _ { t } ( i ) } = \eta _ { t } b _ { t } } \end{array}$ . The one-step transport identity (Proposition 3.4 with $\eta = \eta _ { t } , p = p _ { t } , z = z _ { t } , p ^ { + } = p _ { t + 1 } )$ gives η<sub>t</sub> $\begin{array} { r } { \langle \nu , z _ { t } \rangle = \eta _ { t } ^ { 2 } Q _ { \eta _ { t } } ( p _ { t } , z _ { t } ) + \mathrm { K L } ( \nu \| p _ { t } ) - \mathrm { K L } ( \nu \| p _ { t + 1 } ) = } \end{array}$ $\eta _ { t } b _ { t } + \mathrm { K L } ( \nu \| p _ { t } ) - \mathrm { K L } ( \nu \| p _ { t + 1 } ) .$ , so

$$
\mathrm { K L } ( \nu \| p _ { t } ) - \mathrm { K L } ( \nu \| p _ { t + 1 } ) = \eta _ { t } \left. \nu , z _ { t } \right. - \log \sum _ { i } p _ { t } ( i ) e ^ { \eta _ { t } z _ { t } ( i ) } = \eta _ { t } \left. \nu , z _ { t } \right. - \eta _ { t } b _ { t } = \eta _ { t } ( \left. \nu , z _ { t } \right. - b _ { t } ) .
$$

Summing over $t = 1 , \dots , T$ telescopes to (4.12). For $( 4 . 1 3 ) , \kappa _ { t } : = b _ { t } / \eta _ { t }$ is well-defined by Proposition A.7, and substitution gives η<sub>t</sub> $\langle \nu , z _ { t } \rangle = \eta _ { t } b _ { t } + \eta _ { t } ( \langle \nu , z _ { t } \rangle - b _ { t } ) = \kappa _ { t } \eta _ { t } ^ { 2 } + \eta _ { t } ( \langle \nu , z _ { t } \rangle - b _ { t } )$ , then summing and applying (4.12). The small-scale expansion $\begin{array} { r } { \kappa _ { t } = \frac { 1 } { 2 } \mathrm { V a r } _ { p _ { t } } ( z _ { t } ) + O ( \eta _ { t } ) } \end{array}$ comes from the second-order Taylor expansion of $\begin{array} { r } { \eta \Phi ( \eta ) = \log \sum _ { i } p _ { t } ( i ) e ^ { \eta z _ { t } ( i ) } } \end{array}$ at $\eta = 0$ , which gives $\begin{array} { r } { \Phi ( \eta ) = \eta \left. p _ { t } , z _ { t } \right. + \frac { 1 } { 2 } \eta ^ { 2 } \mathrm { V a r } _ { p _ { t } } ( z _ { t } ) + { \cal O } ( \eta ^ { 3 } ) } \end{array}$ , and dividing by η at $\eta = \eta _ { t }$ produces $b _ { t } / \eta _ { t } =$ ${ \textstyle \frac { 1 } { 2 } } \mathrm { V a r } _ { p _ { t } } ( z _ { t } ) + { \cal O } ( \eta _ { t } )$ in the centered case. □

Proof of Proposition C.1. Write $\begin{array} { r } { Z _ { t } ( \eta ) = \sum _ { i } \pi ( i ) e ^ { - \eta C _ { t } ( i ) } } \end{array}$ and $\Lambda _ { t } ( \eta ) = \log Z _ { t } ( \eta )$ , so $A _ { t } ( \eta ) = - \Lambda _ { t } ( \eta ) / \eta$ . Diferentiating, $\Lambda _ { t } ^ { \prime } ( \eta ) = - \left. \rho _ { t , \eta } , C _ { t } \right.$ , hence $\partial _ { \eta } A _ { t } = - \big ( \eta \Lambda _ { t } ^ { \prime } - \Lambda _ { t } \big ) / \eta ^ { 2 } = \big ( \eta \langle \rho _ { t , \eta } , C _ { t } \rangle + \Lambda _ { t } \big ) / \eta ^ { 2 }$ . For the relative entropy, log $\begin{array} { r } { \left( \rho _ { t , \eta } ( i ) / \pi ( i ) \right) = - \eta C _ { t } ( i ) - \Lambda _ { t } ( \eta ) } \end{array}$ , so $\mathrm { K L } ( \rho _ { t , \eta } | | \pi ) = - \eta \left. \rho _ { t , \eta } , C _ { t } \right. - \Lambda _ { t } ( \eta )$ ; substituting $\Lambda _ { t } = - \mathrm { K L } - \eta \left. \rho _ { t , \eta } , C _ { t } \right.$ gives $\partial _ { \eta } A _ { t } = - \mathrm { K L } ( \rho _ { t , \eta } | | \pi ) / \eta ^ { 2 }$ , which is (C.1). Since ${ \mathrm { K L } } \geq 0 , A _ { t }$ is nonincreasing. Applying Taylor’s theorem with Lagrange remainder to $\eta \mapsto A _ { t } ( \eta )$ on each bracket $\begin{array} { r } { [ \eta _ { t } , \eta _ { t + 1 } ] , A _ { t } ( \eta _ { t + 1 } ) = A _ { t } ( \eta _ { t } ) + \partial _ { \eta } A _ { t } ( \eta _ { t } ) \Delta \eta _ { t } + \frac { 1 } { 2 } \partial _ { \eta } ^ { 2 } A _ { t } ( \xi _ { t } ) ( \Delta \eta _ { t } ) ^ { 2 } } \end{array}$ for some $\xi _ { t }$ between η and $\eta _ { t + 1 } ;$ rearranging and summing yields (C.2) with the bound (C.3). Under the stated regularity, $C _ { t } = \bar { C } ( t / T )$ is uniformly bounded, so $\partial _ { \eta } ^ { 2 } A _ { t }$ is bounded uniformly in $t , T$ on the compact range of η; since $\Delta \eta _ { t } = \eta ^ { \prime } ( t / T ) / \dot { T } + O ( 1 / T ^ { 2 } ) , \sum _ { t } ( \Delta \eta _ { t } ) ^ { 2 } = O ( \mathrm { i } / T )$ , from which $R _ { T } = O ( 1 / T )$ , and the leading sum in (C.2) is a Riemann sum of the integrand of (C.4), converging to it; the equality wit $\textstyle 1 - \int \partial _ { \eta } A d \eta \operatorname { i s } \left( \mathbf { C } . 1 \right)$ □

Proof of Proposition 4.7. For the predictable Gibbs play $p _ { t } = \rho _ { \eta _ { t } , S _ { t - 1 } }$ and $S _ { t } = S _ { t - 1 } + z _ { t } ,$ the one-step transport identity (3.3) reads $\eta _ { t } Q _ { t } = W _ { \eta _ { t } } ( S _ { t } ) - W _ { \eta _ { t } } ( S _ { t - 1 } )$ , while the centered form of the drift in (4.6) is $\begin{array} { r } { D _ { T } = \sum _ { t = 1 } ^ { T - 1 } ( W _ { \eta _ { t + 1 } } ( S _ { t } ) - } \end{array}$ $W _ { \eta _ { t } } ( S _ { t } ) )$ . Adding the two sums,

$$
\sum _ { t = 1 } ^ { T } \eta _ { t } Q _ { t } + D _ { T } = \sum _ { t = 1 } ^ { T } \bigl ( W _ { \eta _ { t } } ( S _ { t } ) - W _ { \eta _ { t } } ( S _ { t - 1 } ) \bigr ) + \sum _ { t = 1 } ^ { T - 1 } \bigl ( W _ { \eta _ { t + 1 } } ( S _ { t } ) - W _ { \eta _ { t } } ( S _ { t } ) \bigr )
$$

and collecting the contributions at each interior state $S _ { t } ~ \mathrm { ( 1 ~ \leq ~ } t ~ \leq ~ T - 1 \mathrm { ) }$ gives $W _ { \eta _ { t } } ( S _ { t } ) \ : - \ : W _ { \eta _ { t + 1 } } ( S _ { t } ) \ : +$ $W _ { \eta _ { t + 1 } } ( S _ { t } ) \ : - \ : W _ { \eta _ { t } } ( S _ { t } ) = 0$ . Only the endpoint terms survive: $- W _ { \eta _ { 1 } } ( S _ { 0 } )$ and $+ W _ { \eta _ { T } } ( S _ { T } ) ;$ ; since $S _ { 0 } ~ = ~ 0$ and $\begin{array} { r } { W _ { \eta } ( 0 ) = \eta ^ { - 1 } \log \sum _ { i } \pi ( i ) = 0 , } \end{array}$ , this is $( 4 . 1 4 )$ . For (4.15), write log $\rho _ { T , \eta _ { T } } ( i ) = \eta _ { T } S _ { T } ( i ) +$ log $\pi ( i ) - \eta _ { T } W _ { \eta _ { T } } ( S _ { T } )$ , so the comparator transport in (4.7) is $B _ { T } ( \nu ) = \eta _ { T } ^ { - 1 } \left. \nu , \log \rho _ { T , \eta _ { T } } - \log \bar { \pi } \right. = \left. \nu , S _ { T } \right. - W _ { \eta _ { T } } ( S _ { T } )$ . Taking the supremum over $\{ \nu : \mathrm { K L } ( \nu \| \pi ) \leq \Gamma \}$ and using the definition of $L _ { \Gamma }$ gives sup<sub>ν</sub> $B _ { T } ( \nu ) = L _ { \Gamma } ( S _ { T } ) - W _ { \eta _ { T } } ( S _ { T } ) _ { }$ ; substituting this and (4.14) into (4.8) gives sup ${ } _ { \nu } R _ { T } ^ { c } ( \nu ) = W _ { \eta _ { T } } ( S _ { T } ) + L _ { \Gamma } ( S _ { T } ) - W _ { \eta _ { T } } ( S _ { T } ) = L _ { \Gamma } ( S _ { T } )$ □

Proof of Theorem C.2. The gradient and scale-derivative identities give the two coeficients of dW in (C.6); since $\partial _ { \eta } \rho _ { \eta , S } =$ $\nabla _ { S } ( \partial _ { \eta } { W _ { \eta } ( S ) } )$ the form is closed, hence exact on the simply connected domain. (i) Path independence of an exact form is immediate, and evaluating the increment at fixed η recovers $\textstyle \sum _ { t } \eta _ { t } Q _ { i }$ by (3.3) while the fixed-S scale switches sum to $D _ { T } , s 0 \ ( 4 . 1 4 )$ is the endpoint value. (ii) Because η is continuous and of finite variation, $[ \eta ] = [ \eta , S ] = 0 ,$ so the only second-order Itô term is the S-quadratic one, $\begin{array} { r } { { \frac { 1 } { 2 } } \operatorname { t r } ( \nabla _ { S } ^ { 2 } W d [ S ^ { c } ] ) = { \frac { 1 } { 2 } } \eta \operatorname { t r } ( \operatorname { C o v } _ { \rho _ { \eta , S } } d [ S ^ { c } ] ) } \end{array}$ by the Hessian identity; the stochastic integral $\textstyle \int \langle \nu , d S \rangle$ vanishes identically on the game’s centered state process; the second-order term is the intrinsic-time accumulation $\textstyle \int \eta d V$ . (iii) The direction independence at $p = \rho _ { \eta , S } \mathrm { i } s ( 3 . 3 ) ;$ ; any other p misaligns the budget form $Q _ { \eta } ( p , \cdot )$ with the gradient and strictly raises the achievable increment. The two Stieltjes legs reduce to (C.5) in the small-η variance limit. □

## D.4 Proofs from Section 5

Proof of Proposition 5.1. Fix an event $A \subseteq [ K ]$ and $\eta > 0$ . For $\nu \in \Delta ( [ K ] )$ with $\operatorname { s u p p } ( \nu ) \subseteq A$ , decompose the relative-entropy cost from the full prior π through the conditional $\pi ( \cdot \mid A )$

$$
\operatorname { K L } ( \nu \| \pi ) = \sum _ { i \in A } \nu ( i ) \log { \frac { \nu ( i ) } { \pi ( i \mid A ) \pi ( A ) } } = \operatorname { K L } ( \nu \| \pi ( \cdot \mid A ) ) - \log \pi ( A )\tag{D.2}
$$

Consequently, for any such $\nu ,$

$$
\langle \nu , S \rangle - \frac { 1 } { \eta } \mathrm { K L } ( \nu \| \pi ) = \langle \nu , S \rangle - \frac { 1 } { \eta } \mathrm { K L } ( \nu \| \pi ( \cdot \ | \ A ) ) + \frac { 1 } { \eta } \log \pi ( A ) .
$$

Taking the supremum over ν with $\operatorname { s u p p } ( \nu ) \subseteq A$ reduces to the classical Gibbs variational identity on the simplex $\Delta ( A )$ with base measure $\pi ( \cdot \mid A )$

$$
\operatorname* { s u p } _ { \nu \in \Delta ( A ) } \left\{ \langle \nu , S \rangle - \frac { 1 } { \eta } \mathrm { K L } ( \nu \| \pi ( \cdot \mid A ) ) \right\} = \frac { 1 } { \eta } \log \sum _ { i \in A } \pi ( i \mid A ) e ^ { \eta S ( i ) } .
$$

Adding $\eta ^ { - 1 } \log \pi ( A )$ and using $\pi ( i \mid A ) \pi ( A ) = \pi ( i )$ for $i \in A$ gives

$$
\operatorname* { s u p } _ { \nu \colon \mathrm { s u p p } ( \nu ) \subseteq A } \left\{ \langle \nu , S \rangle - \frac { 1 } { \eta } \mathrm { K L } ( \nu \| \pi ) \right\} = \frac { 1 } { \eta } \log \sum _ { i \in A } \pi ( i ) e ^ { \eta S ( i ) }\tag{D.3}
$$

which is (5.3). The conditional decomposition (D.2) also yields the advertised split (5.4):

$$
\frac { 1 } { \eta } \log \sum _ { i \in A } \pi ( i ) e ^ { \eta S ( i ) } = \frac { \log \pi ( A ) } { \eta } + \frac { 1 } { \eta } \log \mathbb { E } _ { i \sim \pi ( \cdot | A ) } \left[ e ^ { \eta S ( i ) } \right] .
$$

The optimizing ν is the Gibbs tilt of $\pi ( \cdot \mid A )$ by the score $\eta S ,$ , by uniqueness of the inner Gibbs optimizer.

Proof of Proposition 5.2. Let $u _ { t } ( i ) : = c _ { t } ( i ) - \langle p _ { t } , c _ { t } \rangle$ , so that $\langle p _ { t } , u _ { t } \rangle = 0$ and $Q _ { t } ( c ) = \eta _ { t } ^ { - 2 }$ log $\begin{array} { r } { \sum _ { i } p _ { t } ( i ) e ^ { - \eta _ { t } u _ { t } ( i ) } } \end{array}$ . Under $c _ { t } ( i ) \in [ 0 , 1 ]$ and $\eta _ { t } \in ( 0 , 1 ]$ , one has $- \eta _ { t } u _ { t } ( i ) \in [ - \eta _ { t } , \eta _ { t } ] \subseteq [ - 1 , 1 ]$ . The elementary inequality $\bar { e } ^ { x } \le 1 + x + ( e - 2 ) x ^ { 2 }$ valid for every $x \in [ - 1 , 1 ]$ , is Lemma A.1 of [15]; a direct proof notes that $g ( x ) : = e ^ { x } - 1 - x - ( e - 2 ) x ^ { 2 }$ satisfies $g ( - 1 ) = 1 / e - 1 + 1 - ( \stackrel { . } { e } - 2 ) = 1 / e - e + 2 < 0 , g ( 0 ) = 0 , g ( 1 ) = 0$ , and $g ^ { \prime \prime }$ has a single sign change on $[ - 1 , 1 ]$ , so $g \le 0$ throughout. Taking expectation under $p _ { t }$ and using $\langle p _ { t } , u _ { t } \rangle = 0 ,$

$$
\begin{array} { r } { \mathbb { E } _ { p _ { t } } e ^ { - \eta _ { t } u _ { t } } \le 1 + ( e - 2 ) \eta _ { t } ^ { 2 } \mathbb { E } _ { p _ { t } } u _ { t } ^ { 2 } = 1 + ( e - 2 ) \eta _ { t } ^ { 2 } \operatorname { V a r } _ { p _ { t } } ( c _ { t } ) . } \end{array}
$$

Since $\begin{array} { r } { \log ( 1 + z ) \leq z \mathrm { ~ f o r ~ } z \geq 0 , \eta _ { t } ^ { 2 } Q _ { t } ( c ) = \log \mathbb { E } _ { p _ { t } } e ^ { - \eta _ { t } u _ { t } } \leq ( e - 2 ) \eta _ { t } ^ { 2 } \mathrm { V a r } _ { p _ { t } } ( c _ { t } ) , \mathrm { s o } Q _ { t } ( c ) \leq ( e - 2 ) \mathrm { V a r } _ { p _ { t } } ( c _ { t } ) } \end{array}$ . Finally, for any comparator $\nu \in \Delta ( [ K ] )$

$$
\operatorname { V a r } _ { p _ { t } } ( c _ { t } ) = \operatorname* { m i n } _ { a \in \mathbb { R } } \sum _ { i } p _ { t } ( i ) ( c _ { t } ( i ) - a ) ^ { 2 } \leq \sum _ { i } p _ { t } ( i ) ( c _ { t } ( i ) - \langle \nu , c _ { t } \rangle ) ^ { 2 } = \Psi _ { t } ( \nu ) ,
$$

which yields (5.5). The regret consequence is immediate: summing the fixed-rate telescope $R _ { T } ^ { c } ( \nu ) \leq \eta _ { T } ^ { - 1 } \mathrm { K L } ( \nu \| \pi ) +$ $\textstyle \sum _ { t } \eta _ { t } Q _ { t } ( c )$ (valid for nonincreasing schedules by Theorem 4.3) and substituting the per-round envelope $Q _ { t } ( c ) \ \leq$ $( e - 2 ) \Psi _ { t } ( \nu )$ gives the stated bound. □

Proof of Theorem 5.3. Assume $c _ { 1 } , c _ { 2 } , \ldots$ . are i.i.d. vectors in $[ 0 , 1 ] ^ { K }$ with mean $\mu = \mathbb { E } c _ { 1 }$ , and that the comparator ν satisfies the low-noise condition with constant $\kappa _ { \nu } .$ . Because $p _ { t } \mathrm { i s } \mathcal F _ { t - 1 }$ -measurable and $c _ { t }$ is independent of $\mathcal { F } _ { t - 1 }$

$$
\mathbb { E } [ \Psi _ { t } ( \nu ) \mid \mathcal { F } _ { t - 1 } ] = \sum _ { i } p _ { t } ( i ) \mathbb { E } [ ( c _ { t } ( i ) - \langle \nu , c _ { t } \rangle ) ^ { 2 } ] \leq \kappa _ { \nu } \sum _ { i } p _ { t } ( i ) ( \mu ( i ) - \langle \nu , \mu \rangle ) = \kappa _ { \nu } \big ( \langle p _ { t } , \mu \rangle - \langle \nu , \mu \rangle \big )\tag{D.4}
$$

using the low-noise hypothesis coordinatewise. In the same filtration, $\mathbb { E } [ \langle p _ { t } - \nu , c _ { t } \rangle \mid \mathcal { F } _ { t - 1 } ] = \langle p _ { t } - \nu , \mu \rangle$ , so summing over t and taking full expectation,

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } [ \Psi _ { t } ( \nu ) ] \leq \kappa _ { \nu } \sum _ { t = 1 } ^ { T } \mathbb { E } [ \langle p _ { t } - \nu , c _ { t } \rangle ] = \kappa _ { \nu } \mathbb { E } R _ { T } ^ { c } ( \nu )\tag{D.5}
$$

At fixed scale $\eta _ { t } \equiv \eta \in ( 0 , 1 ]$ , the fixed-scale exact identity (Theorem 4.3 with $D _ { T } ~ = ~ 0 )$ and the terminal bound $\mathrm { K L } ( \nu \| p _ { T + 1 } ) \geq 0$ give

$$
R _ { T } ^ { c } ( \nu ) \leq \frac { \mathrm { K L } ( \nu \| \pi ) } { \eta } + \eta \sum _ { t = 1 } ^ { T } Q _ { t } ( c )\tag{D.6}
$$

Combining (D.6) with the comparator-centered envelope $Q _ { t } ( c ) \leq ( e - 2 ) \Psi _ { t } ( \nu )$ (Proposition 5.2) and taking expectation,

$$
\begin{array} { r l r } {  { \mathbb { E } R _ { T } ^ { c } ( \nu ) \le \frac { \mathrm { K L } ( \nu \| \pi ) } { \eta } + ( e - 2 ) \eta \sum _ { t = 1 } ^ { T } \mathbb { E } [ \Psi _ { t } ( \nu ) ] } } \\ & { } & { \le \frac { \mathrm { K L } ( \nu \| \pi ) } { \eta } + ( e - 2 ) \kappa _ { \nu } \eta \mathbb { E } R _ { T } ^ { c } ( \nu ) } \end{array}\tag{D.7}
$$

where the second step uses (D.5). If $( e - 2 ) \kappa _ { \nu } \eta < 1$ , rearrangement of (D.7) gives

$$
\mathbb { E } R _ { T } ^ { c } ( \nu ) \leq \frac { \mathrm { K L } ( \nu \| \pi ) } { \eta \big ( 1 - ( e - 2 ) \kappa _ { \nu } \eta \big ) } ,
$$

which is (5.6). With $\eta = \mathrm { m i n } \{ 1 , [ 2 ( e - 2 ) \kappa _ { \nu } ] ^ { - 1 } \}$ one has $( e - 2 ) \kappa _ { \nu } \eta \leq 1 / 2 , \ s \circ 1 - ( e - 2 ) \kappa _ { \nu } \eta \geq 1 / 2 ,$ , and therefore

$$
\mathbb { E } R _ { T } ^ { c } ( \nu ) \leq \frac { 2 \mathrm { K L } ( \nu \| \pi ) } { \eta } \leq 2 \big ( 1 + 2 ( e - 2 ) \kappa _ { \nu } \big ) \mathrm { K L } ( \nu \| \pi ) ,
$$

proving the constant-in-T corollary.

Proof of Proposition A.9. Because $p _ { t }$ is $\mathcal { F } _ { t - 1 }$ -measurable and $c _ { t }$ is independent of $\mathcal { F } _ { t - 1 }$ with mean $\mu ,$ the per-round identity $\mathbb { E } [ \langle p _ { t } - \nu , c _ { t } \rangle \mid \mathcal { F } _ { t - 1 } ] = \langle p _ { t } - \nu , \mu \rangle$ used in (D.5) gives, on summing and taking full expectation, $\mathbb { E } R _ { T } ^ { c } ( \nu ) =$ $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \left. \mathbb { E } p _ { t } - \nu , \mu \right. } \end{array}$ . Subtracting the same expression at $\nu = \delta _ { k ^ { * } }$ ∗ cancels the common algorithm term $\textstyle \sum _ { t } \left. { \mathbb { E } } p _ { t } , \mu \right.$ and leaves the exact identity

$$
\mathbb { E } R _ { T } ^ { c } ( \nu ) - \mathbb { E } R _ { T } ^ { c } ( \delta _ { k ^ { * } } ) = T \left. \delta _ { k ^ { * } } - \nu , \mu \right. = T \left( \mu ^ { * } - \left. \nu , \mu \right. \right)
$$

which is nonpositive because $\langle \nu , \mu \rangle \geq \operatorname* { m i n } _ { i } \mu ( i ) = \mu ^ { * }$ ; this is (A.29). Under the stated hypothesis Theorem 5.3 applies at $\nu = \delta _ { k ^ { * } }$ with KL $( \delta _ { k ^ { * } } \| \pi ) = - \log \pi ( k ^ { * } )$ , and combining its constant-in-T bound with (A.29) gives (A.30) for every ν. □

Proof of (5.7). Under the predictable schedule (4.9), the variable-temperature exact identity (Theorem 4.3) and the nonpositivity of $D _ { T }$ under nonincreasing η<sub>t</sub> give

$$
\mathbb { E } R _ { T } ^ { c } ( \nu ) \leq \mathbb { E } \left[ \frac { \Gamma } { \eta _ { T } } + \sum _ { t = 1 } ^ { T } \eta _ { t } Q _ { t } ( c ) \right] .
$$

The two-sided envelope of Theorem 4.4 then yields E $\begin{array} { r } { { \cal { R } } _ { T } ^ { c } ( \nu ) \leq \Gamma + \mathbb { E } [ Q _ { T } ^ { * } ( c ) ] + ( 2 C + C ^ { - 1 } ) \sqrt { \Gamma \mathbb { E } V _ { T } ( c ) } } \end{array}$ , where we used E $\sqrt { V _ { T } ( c ) } \leq \sqrt { \mathbb { E } V _ { T } ( c ) }$ by concavity. Applying Proposition 5.2 pathwise, E $\begin{array} { r } { V _ { T } ( c ) = \sum _ { t } \mathbb { E } Q _ { t } ( c ) \leq ( e - 1 ) } \end{array}$ $\begin{array} { r } { 2 ) \sum _ { t } \mathbb { E } \Psi _ { t } ( \nu ) \leq ( e - \dot { 2 } ) \kappa _ { \nu } \mathbb { E } R _ { T } ^ { c } ( \nu ) } \end{array}$ by (D.5). Writing $X : = \mathbb { E } R _ { T } ^ { c } ( \nu )$ and $A : = \Gamma + \mathbb { E } [ Q _ { T } ^ { * } ( c ) ] , B : = ( 2 C + C ^ { - 1 } )$ $\lambda : = ( e - 2 ) \kappa _ { \nu } ,$ that bound becomes $X \le A + B \sqrt { \lambda X \Gamma }$ . Writing $b : = B ^ { 2 } \lambda \Gamma$ , solving the quadratic in $\sqrt { X }$ gives $\begin{array} { r } { X \le \frac { \mathrm { i } } { 2 } \big ( 2 A + b + \sqrt { b ^ { 2 } + 4 A b } \big ) } \end{array}$ , and $\sqrt { b ^ { 2 } + 4 A b } \leq b + 2 A$ then gives $X \leq 2 A + b$ . Substituting back gives

$$
\begin{array} { r } { \mathbb { E } R _ { T } ^ { c } ( \nu ) \leq 2 \Gamma + 2 \mathbb { E } [ Q _ { T } ^ { * } ( c ) ] + ( 2 C + C ^ { - 1 } ) ^ { 2 } ( e - 2 ) \kappa _ { \nu } \Gamma , } \end{array}
$$

which is (5.7).

## D.5 Proofs from Section 6

Proof of Proposition A.10. By definition $\begin{array} { r } { C _ { \alpha } ( { \pi _ { 1 : W } } ) = - \log \sum _ { i } \prod _ { w } \pi _ { w } ( i ) ^ { \alpha _ { w } } } \end{array}$ . For any $p \in \Delta ( [ K ] )$

$$
- \log \sum _ { i } \prod _ { w } \pi _ { w } ( i ) ^ { \alpha _ { w } } = - \log \sum _ { i } p ( i ) \prod _ { w } \left( \frac { \pi _ { w } ( i ) } { p ( i ) } \right) ^ { \alpha _ { w } } \leq - \sum _ { i } p ( i ) \sum _ { w } \alpha _ { w } \log \frac { \pi _ { w } ( i ) } { p ( i ) } = \sum _ { w } \alpha _ { w } \mathrm { K L } ( p | | \pi _ { w } ) ,
$$

where the first equality uses $\textstyle \sum _ { w } \alpha _ { w } \ = \ 1$ to write $\begin{array} { r } { p ( i ) \ = \ \prod _ { w } p ( i ) ^ { \alpha _ { w } } } \end{array}$ , and the inequality is Jensen applied to the concave log. The first inequality is sharpest, attained by $\begin{array} { r } { p _ { \alpha } ^ { \star } ( i ) \propto \prod _ { w } \pi _ { w } ( i ) ^ { \alpha _ { w } } } \end{array}$ , giving $\begin{array} { r } { C _ { \alpha } ( \pi _ { 1 : W } ) = \sum _ { w } \alpha _ { w } \mathrm { K L } ( p _ { \alpha } ^ { \star } | | \pi _ { w } ) } \end{array}$ Now apply Sion’s minimax theorem to the bilinear function $\begin{array} { r } { ( \alpha , p ) \mapsto \sum _ { w } \alpha _ { w } \mathrm { K L } ( p \| \pi _ { w } ) } \end{array}$ on the compact convex sets $\Delta ( [ W ] ) \times \Delta ( [ K ] )$ :

$$
C ( \pi _ { 1 : W } ) = \operatorname* { m a x } _ { \alpha } C _ { \alpha } = \operatorname* { m a x } _ { \alpha } \operatorname* { m i n } _ { p } \sum _ { \omega } \alpha _ { w } \mathrm { K L } ( p | | \pi _ { w } ) = \operatorname* { m i n } _ { p } \operatorname* { m a x } _ { \alpha } \sum _ { \omega } \alpha _ { w } \mathrm { K L } ( p | | \pi _ { w } ) = \operatorname* { m i n } _ { p } \operatorname* { m a x } _ { \omega } \mathrm { K L } ( p | | \pi _ { w } ) ,
$$

where the last equality is because α 7→ $\textstyle \sum _ { w } \alpha _ { w } x _ { w }$ on the simplex attains its maximum at a vertex. The optimizer $p ^ { \star } = p _ { \alpha } ^ { \star } ,$ is the geometric mixture induced by a maximizing $\alpha ^ { \star }$ □

Proof of Theorem 6.1. Apply Theorem 4.3 to the centered regret-vector sequence $\left( g _ { t } \right)$ with prior π and schedule $( \eta _ { t } )$ treating $g _ { t }$ as the loss vector in centered coordinates: $\langle p _ { t } - \nu , \ell _ { t } \rangle = \langle \nu , g _ { t } \rangle$ since $g _ { t } ( i ) ~ = ~ \langle p _ { t } , \ell _ { t } \rangle - \ell _ { t } ( i )$ implies $\langle \nu , g _ { t } \rangle = \langle p _ { t } , \ell _ { t } \rangle - \langle \nu , \ell _ { t } \rangle$ . Summing produces (6.1) with $Q _ { t } ( g ) , D _ { T } ^ { g } , B _ { T } ^ { g } ( \nu )$ defined as in Theorem 4.3 but on the centered regret sequence. Specializing to the comparator $\nu = \delta _ { i }$ with $\bar { \Gamma _ { } } \geq - \log \pi ( i )$ gives $B _ { T } ^ { g } ( \delta _ { i } ) \ \leq \ \Gamma / \eta _ { T }$ , and combining with Theorem 4.4 (and $D _ { T } ^ { g } \leq 0$ under the second-order schedule) yields the displayed inequality. Scale equivariance: replacing $\ell _ { t }$ by $a \ell _ { t }$ scales g<sub>t</sub> by $^ { a , }$ , hence $z _ { t } = - g _ { t }$ centered also by a; then $Q _ { t } ( a g ; p , \eta ) = a ^ { 2 } Q _ { t } ( g ; p , a \eta )$ and the schedule $\eta _ { t } = C \sqrt { \Gamma / V _ { t - 1 } ( g ) }$ with $V _ { t - 1 } ( g )$ scaling by $a ^ { 2 }$ becomes $\eta _ { t } / a ,$ so each loss $\eta _ { t } Q _ { t }$ and the cumulative $\textstyle \sum _ { t } \eta _ { t } Q _ { i }$ <sub>t</sub> scale by $^ { a , }$ matching the degree-one homogeneity of the regret. The play sequence and the schedule’s form are scale-free. □

Proof of Theorem 6.2. Both players run intrinsic-time regret matching, so Theorem 6.1 applies to each. For the row player, whose round-t loss vector is $M \omega _ { t }$ , the identity reads

$$
\sum _ { t = 1 } ^ { T } \left. p _ { t } - \nu , M \omega _ { t } \right. = P _ { T } ^ { \mathrm { r o w } } + D _ { T } ^ { \mathrm { r o w } } + B _ { T } ^ { \mathrm { r o w } } ( \nu ) ,
$$

and the left-hand side equals $\begin{array} { r } { \sum _ { t } p _ { t } ^ { \top } M \omega _ { t } - \nu ^ { \top } M \sum _ { t } \omega _ { t } = T \big ( \overline { { p _ { t } ^ { \top } M \omega _ { t } } } \big ) - T \nu ^ { \top } M \bar { \omega } _ { T } } \end{array}$ . For the column player, whose round-t loss vector $\mathrm { i } s - M ^ { \top } p _ { t }$ (the maximizer’s loss),

$$
\sum _ { t = 1 } ^ { T } \left. \omega _ { t } - \nu ^ { \prime } , - M ^ { \top } p _ { t } \right. = P _ { T } ^ { \mathrm { c o l } } + D _ { T } ^ { \mathrm { c o l } } + B _ { T } ^ { \mathrm { c o l } } ( \nu ^ { \prime } ) ,
$$

whose left-hand side equals $\bar { p } _ { T } ^ { \top } M \nu ^ { \prime } \cdot T - T \big ( \overline { { p _ { t } ^ { \top } M \omega _ { t } } } \big )$ . Adding the two identities cancels the common $T p _ { t } ^ { \top } M \omega _ { t }$ term, leaving

$$
T \big ( \hat { p } _ { T } ^ { \top } M \nu ^ { \prime } - \nu ^ { \top } M \hat { \omega } _ { T } \big ) = P _ { T } ^ { \mathrm { r o w } } + P _ { T } ^ { \mathrm { c o l } } + D _ { T } ^ { \mathrm { r o w } } + D _ { T } ^ { \mathrm { c o l } } + B _ { T } ^ { \mathrm { r o w } } ( \nu ) + B _ { T } ^ { \mathrm { c o l } } ( \nu ^ { \prime } ) ,
$$

which is (6.2) after dividing by $T .$ Both outer optimizations are linear in their comparator, so arg ma $\mathrm { x } _ { \omega ^ { \prime } } \bar { p } _ { T } ^ { \top } M \omega ^ { \prime }$ and arg min<sub>p</sub>′ $p ^ { \prime \top } M \bar { \omega } _ { T }$ are attained at pure strategies, of comparator complexity $\log ( 1 / \pi ^ { \operatorname { c o l } } ( j ) )$ ) and $\log ( 1 / \pi ^ { \mathrm { r o w } } ( i ) )$ respectively. The budget hypotheses place those pure strategies inside the respective comparator classes, so $\dot { B } _ { T } ^ { \mathrm { r o w } } ( \nu ) = B _ { T } ^ { \mathrm { r o w , \star } }$ and $B _ { T } ^ { \mathrm { c o l } } ( \nu ^ { \prime } ) = B _ { T } ^ { \mathrm { c o l } , \star }$ at the specialized comparators, yielding the duality-gap form (6.3). No quadratic-variation surrogate enters, so the equality is exact. Under smaller budgets only $B _ { T } ^ { \mathrm { r o w } } ( \nu ) \leq B _ { T } ^ { \mathrm { r o w , \star } }$ and $B _ { T } ^ { \mathrm { c o l } } ( \nu ^ { \prime } ) \leq B _ { T } ^ { \mathrm { c o l , \star } }$ survive, and (6.3) holds with its left-hand side restricted to the two comparator classes. □

Proof of Proposition A.11. This is the standard external-regret-to-CCE reduction [11, 33]: averaging player m’s regret bound under the empirical joint $\sigma _ { T }$ yields the $\varepsilon _ { m } \mathrm { - C C E }$ inequality directly, and taking the maximum over players gives $\varepsilon = \operatorname* { m a x } _ { m } \varepsilon _ { m }$ □

Proof of Proposition A.12. This is the standard reduction from external to swap regret [11, 33], applied to the empirical joint one swap function at a time. □

Proof of Theorem 6.3. For each player $k ,$ apply Theorem 6.1 to that player’s centered regret sequence with prior $\pi ^ { ( k ) }$ and schedule $( \eta _ { t } ^ { ( k ) } )$ , obtaining the exact decomposition $R _ { T } ^ { ( k ) } = P _ { T } ^ { ( k ) } \hat { + } \hat { D _ { T } } ^ { ( k ) } + B _ { T } ^ { ( k ) } ( \nu _ { k } )$ for each comparator $\nu _ { k }$ in player k’s simplex. For the mixed-action joint $\begin{array} { r } { \sigma _ { T } = \frac { 1 } { T } \sum _ { t } \otimes _ { k } p _ { t } ^ { ( k ) } } \end{array}$ , player k’s deviation gain in (6.4) against comparator $\nu _ { k }$ is $\begin{array} { r l } { \frac { 1 } { T } \sum _ { t } \bigl ( \left. p _ { t } ^ { ( k ) } , \ell _ { t } ^ { ( k ) } \right. - \left. \nu _ { k } , \ell _ { t } ^ { ( k ) } \right. \bigr ) } & { { } } \end{array}$ with $\ell _ { t } ^ { ( k ) }$ that player’s expected loss vector under the opponents’ round-t mixtures, which is $R _ { T } ^ { ( k ) } ( \nu _ { k } ) / T ;$ ; hence $\begin{array} { r } { d ( \sigma _ { T } , \mathrm { C C E } ) = \frac { 1 } { T } } \end{array}$ max<sub>k</sub> $\begin{array} { r } { \operatorname* { s u p } _ { \nu _ { k } \in \Delta ( [ K _ { k } ] ) } R _ { T } ^ { ( k ) } ( \nu _ { k } ) } \end{array}$ exactly, and Proposition A.11 is its $\varepsilon -$ form. Each $R _ { T } ^ { ( k ) } ( \cdot )$ is linear in the comparator, so its supremum over the simplex is attained at a pure deviation $\delta _ { i } ,$ whose comparator complexity is $\mathrm { K L } ( \delta _ { i } \bar { | | } \pi ^ { ( k ) } ) = \log ( 1 / \bar { \pi } ^ { ( k ) } ( i ) )$ . The hypothesis $\Gamma ^ { ( k ) } \geq \operatorname* { m a x } _ { i } \log ( 1 / \pi ^ { ( k ) } ( i ) )$ places every such vertex inside the budget, so the budget-restricted supremum coincides with the simplex supremum and $\begin{array} { r } { \operatorname* { s u p } _ { \nu _ { k } }  { R _ { T } } ^ { ( k ) } ( \nu _ { k } ) = P _ { T } ^ { ( k ) } +  { D _ { T } ^ { ( k ) } } +  { B _ { T } ^ { ( k ) , \star } } } \end{array}$ . Taking the maximum over k produces (6.5). Without that hypothesis the two suprema difer and only the restricted class is governed: the budget-restricted right-hand side can fall strictly below $d ( \sigma _ { T } , \mathrm { C C E } )$ whenever the maximizing pure deviation is excluded. The right-hand side is exact: each $P _ { T } ^ { ( k ) } , D _ { T } ^ { ( k ) } , B _ { T } ^ { ( k ) , \prime }$ is an information-theoretic quantity drawn from player k’s separate concentration game; no quadratic-variation surrogate is invoked. □

The proof tracks each deviation’s log-odds in place of the of-equilibrium mass itself. An opponent playing of $a ^ { \star }$ shifts a player’s per-round loss gap by an amount proportional to its deviation probability, and in the retempered update that shift acts on the drift of the log-odds and leaves their level alone. An additive recursion on the mass itself, $\delta _ { t + 1 } \leq e ^ { - \eta _ { t } ^ { ( k ) } \Delta } \delta _ { t } + \varepsilon _ { t }$ overstates the coupling: under it a persistent vanishing drive $\varepsilon _ { t } = \Theta ( 1 / t )$ appears to unbound the intrinsic time, and only a summability hypothesis on the opponents’ approach would rule that out. In log-odds coordinates no such hypothesis arises.

Proof of Theorem A.13. Fix a player k and write $\delta _ { t } : = 1 - p _ { t } ^ { ( k ) } ( a _ { k } ^ { \star } )$ for its of-equilibrium mass, $\begin{array} { r } { \varepsilon _ { t } : = \sum _ { k ^ { \prime } \neq k } ( 1 - } \end{array}$ $p _ { t } ^ { ( k ^ { \prime } ) } ( a _ { k ^ { \prime } } ^ { \star } ) )$ for its opponents’ deviation mass, $\ell _ { t } ^ { ( k ) } ( i ) : = \mathbb { E } [ c _ { k } ( i , a _ { - k } ) ]$ for its round loss under the opponents’ play, $\begin{array} { r } { C _ { t } ^ { ( k ) } : = \sum _ { s < t } \ell _ { s } ^ { ( k ) } } \end{array}$ for the cumulative loss, and $g _ { t } ^ { ( k ) }$ for the centered score, with $\mathrm { r a n g e } ( g _ { t } ^ { ( k ) } ) \le 2 B$

Density bound: $g _ { t } ^ { ( k ) }$ is centered under $p _ { t } ^ { ( k ) }$ , which places mass $1 - \delta _ { t }$ on $a _ { k } ^ { \star }$ , so $\operatorname { V a r } _ { p _ { t } ^ { ( k ) } } ( g _ { t } ^ { ( k ) } ) = O ( \delta _ { t } )$ ; the tilted-variance representation $\begin{array} { r } { Q _ { \eta } ( p , z ) = \int _ { 0 } ^ { 1 } ( 1 - s ) \mathrm { V a r } _ { p _ { s } } ( z ) } \end{array}$ ds of Proposition 2.4 carries this to the per-round centered RCGF at every $\eta _ { t } ^ { ( k ) } \leq C ,$ , since $p _ { s } ( i ) / p ( i ) = e ^ { s \eta z ( i ) } / \mathbb { E } _ { p } [ e ^ { s \eta z } ] \le e ^ { s \eta }$ <sup>η</sup> <sup>maxi</sup> <sup>z(i)</sup> byJensen’s inequality, which with range $( g _ { t } ^ { ( k ) } ) \leq 2 B$ gives $\operatorname { V a r } _ { p _ { s } } ( g _ { t } ^ { ( k ) } ) \leq e ^ { 2 B C } \operatorname { V a r } _ { p _ { t } ^ { ( k ) } } ( g _ { t } ^ { ( k ) } )$ and hence $Q _ { t } ^ { ( k ) } \leq { \textstyle \frac { 1 } { 2 } } e ^ { 2 B C } \mathrm { { V a r } } _ { p _ { t } ^ { ( k ) } } ( g _ { t } ^ { ( k ) } ) \leq c \delta _ { t }$ for a constant $c = c ( B , C )$ . Scale floor: the per-round RCGF is bounded, $Q _ { t } ^ { ( k ) } \leq \mathrm { r a n g e } ( g _ { t } ^ { ( k ) } ) ^ { 2 } / 8 \leq B ^ { 2 } / 2$ (Hoefding, Corollary 2.5), so $V _ { t - 1 } ^ { ( k ) } \le B ^ { 2 } t / 2$ and

$$
\eta _ { t } ^ { ( k ) } = C \sqrt { \frac { \Gamma ^ { ( k ) } } { V _ { t - 1 } ^ { ( k ) } + \Gamma ^ { ( k ) } } } \geq \frac { c _ { 1 } } { \sqrt { t } } , \qquad c _ { 1 } : = C \sqrt { \frac { \Gamma ^ { ( k ) } } { B ^ { 2 } / 2 + \Gamma ^ { ( k ) } } }\tag{D.8}
$$

on every trajectory — the floor uses only that the per-round RCGF is bounded. Gap floor: convergence to $a ^ { \star }$ gives a finite time τ with $\varepsilon _ { t } \leq \Delta / ( 2 c _ { M } )$ for all $t \geq \tau$ , where $c _ { M } : = 4 B$ bounds the efect of an opponent deviation on a loss gap. Since the opponents’ joint law difers from the point mass at $a _ { - k } ^ { \star }$ in total variation by at most $\varepsilon _ { t } ,$ , every deviation $i \neq a _ { k } ^ { \star }$ and every $t \geq \tau$ satisfy

$$
\ell _ { t } ^ { ( k ) } ( i ) - \ell _ { t } ^ { ( k ) } ( a _ { k } ^ { \star } ) \geq \Delta - c _ { M } \varepsilon _ { t } \geq \Delta / 2\tag{D.9}
$$

Now track the log-odds. For each deviation $i \neq a _ { k } ^ { \star }$ the retempered update gives

$$
y _ { t } ^ { ( k , i ) } : = \log \frac { p _ { t } ^ { ( k ) } ( a _ { k } ^ { \star } ) } { p _ { t } ^ { ( k ) } ( i ) } = \eta _ { t } ^ { ( k ) } \bigl ( C _ { t - 1 } ^ { ( k ) } ( i ) - C _ { t - 1 } ^ { ( k ) } ( a _ { k } ^ { \star } ) \bigr ) + \log \frac { \pi ^ { ( k ) } ( a _ { k } ^ { \star } ) } { \pi ^ { ( k ) } ( i ) }
$$

Split the cumulative gap at τ: each summand with $s < \tau$ is at least $- 2 B$ (bounded scores) and each with $s \geq \tau$ is at least $\Delta / 2$ by (D.9), so $\begin{array} { r } { C _ { t - 1 } ^ { ( k ) } ( i ) - C _ { t - 1 } ^ { ( k ) } ( a _ { k } ^ { \star } ) \geq \frac { \Delta } { \gamma } ( t - \tau ) - 2 B \tau } \end{array}$ . With the scale floor (D.8) and $\kappa : = | \log ( \pi ^ { ( k ) } ( a _ { k } ^ { \star } ) / \pi ^ { ( k ) } ( i ) ) |$

$$
y _ { t } ^ { ( k , i ) } \geq \frac { c _ { 1 } } { \sqrt { t } } \Big ( \frac { \Delta } { 2 } ( t - \tau ) - 2 B \tau \Big ) - \kappa = \frac { c _ { 1 } \Delta } { 2 } \sqrt { t } - O ( 1 )
$$

Hence $\begin{array} { r } { \delta _ { t } = \sum _ { i \neq a _ { k } ^ { \star } } p _ { t } ^ { ( k ) } ( i ) \leq \sum _ { i } e ^ { - y _ { t } ^ { ( k , i ) } } \leq ( K _ { k } - 1 ) e ^ { O ( 1 ) } e ^ { - ( c _ { 1 } \Delta / 2 ) \sqrt { t } } , } \end{array}$ , a stretched exponential, and the density bound gives

$$
V _ { \infty } ^ { ( k ) } = \sum _ { t } Q _ { t } ^ { ( k ) } \ \leq \ c \sum _ { t } \delta _ { t } \ < \ \infty
$$

which is $( \mathsf { A } . 3 2 ) \mathrm { - } \mathsf { a }$ convergent series, which is stronger than ${ \cal O } ( \log T )$ . With $V ^ { ( k ) }$ bounded the schedule freezes at $\eta _ { \infty } ^ { ( k ) } = C \sqrt { \Gamma ^ { ( k ) } / ( V _ { \infty } ^ { ( k ) } + \Gamma ^ { ( k ) } ) > 0 } ;$ then $\eta _ { t } ^ { ( k ) } \geq \eta _ { \infty } ^ { ( k ) }$ makes the log-odds grow linearly, $y _ { t } ^ { ( k , i ) } \geq \eta _ { \infty } ^ { ( k ) } \frac { \Delta } { 2 } ( t - \tau ) - O ( 1 )$ so $\delta _ { t } = O \left( e ^ { - \eta _ { \infty } ^ { ( k ) } ( \Delta / 2 ) t } \right)$ and $Q _ { t } ^ { ( k ) } \leq c \delta _ { t }$ decay geometrically. Finally, each term of player k’s exact-regret sum in the ledger (6.5) is $O ( 1 )$ : the loss obeys $P _ { T } ^ { ( k ) } \leq C V _ { T } ^ { ( k ) }$ since $\eta _ { t } ^ { ( k ) } \leq C$ , the drift is a gain under the nonincreasing schedule, and the transport obeys $B _ { T } ^ { ( k ) , \star } \leq \bar { \Gamma ^ { ( k ) } } / \eta _ { T } ^ { ( k ) } \leq \bar { \Gamma ^ { ( k ) } } / \eta _ { \infty } ^ { ( k ) }$ ; hence $d ( \sigma _ { T } , \mathrm { C C E } ) = O ( 1 / T )$ □

The argument is not circular. The scale floor (D.8) holds for every trajectory, so the $\sqrt { t }$ growth of the log-odds — and thus the summability of $\delta _ { t } -$ is derived without presupposing that $V ^ { ( k ) }$ is bounded. The worst case $V _ { t } ^ { ( k ) } = \Theta ( t )$ is self-refuting: it forces $\eta _ { t } ^ { ( k ) } = \Theta ( 1 / \sqrt { t } )$ , log-odds $\Theta ( { \sqrt { t } } )$ , and $\delta _ { t } = O ( e ^ { - ( c _ { 1 } \Delta / 2 ) \sqrt { t } } )$ , from which $\begin{array} { r } { V _ { \infty } ^ { ( k ) } = \sum _ { t } Q _ { t } ^ { ( k ) } < \infty } \end{array}$ contradicting $V _ { t } ^ { ( k ) } = \Theta ( t )$ . The mechanism is specific to the self-tuned schedule: it dilutes the learning rate at most as fast as $1 / { \sqrt { t } } ,$ , while the strict gap accumulates an advantage linear in $t ,$ and the linear term wins on every trajectory. A fixed rate would give an exactly geometric decay, and sustaining a slow, unbounded- $. V$ branch would need a rate decaying faster than $1 / { \sqrt { t } } ,$ , which this schedule never produces. The bound is structural: shrinking the strict gap $\Delta$ lengthens the transient by the factor $1 / \Delta$ without unbounding $V _ { T } ^ { ( k ) }$ . One level up remains open: whether the coupled dynamics-and-schedule system converges to a strict profile at all, for a given class of games, or instead cycles or settles on an interior equilibrium.

## D.6 Proofs from Section 7

Proof of Theorem A.14. Write $\bar { c } _ { t } : = \mathbb { E } [ \widehat { c } _ { t } \ | \ \mathcal { F } _ { t - 1 } ]$ for the predictable conditional mean of the estimator. The sampled composite-loss regret regroups around the sampling and posterior distributions:

$$
c _ { t } ( A _ { t } ) - \langle u , c _ { t } \rangle = \big ( c _ { t } ( A _ { t } ) - \langle \mu _ { t } , c _ { t } \rangle \big ) + \langle \mu _ { t } - p _ { t } , c _ { t } \rangle + \langle p _ { t } - u , c _ { t } \rangle .
$$

For the last summand, insert and subtract $\widehat { c } _ { t }$ and then $\bar { c } _ { t } \mathrm { : }$

$$
\langle p _ { t } - u , c _ { t } \rangle = \langle p _ { t } - u , \widehat c _ { t } \rangle - \langle p _ { t } - u , \widehat c _ { t } - \bar { c } _ { t } \rangle + \langle p _ { t } - u , c _ { t } - \bar { c } _ { t } \rangle .
$$

Summing over t and naming the observation terms,

$$
\mathrm { R e g } _ { T } ^ { c } ( u ) = M _ { T } ^ { \mathrm { p l a y } } + \Xi _ { T } - M _ { T } ^ { \mathrm { e s t } } ( u ) + \mathrm { B i a s } _ { T } ( u ) + \sum _ { t = 1 } ^ { T } \langle p _ { t } - u , \widehat c _ { t } \rangle .
$$

The remaining sum is the full-information regret on the estimated scores $\widehat { c } _ { t }$ , so Theorem 4.3 applied to that sequence gives $\begin{array} { r } { \sum _ { t } \left. p _ { t } \stackrel { - } { - } u , \widehat c _ { t } \right. = \widehat { D } _ { T } + \widehat { B } _ { T } ( u ) + \sum _ { t } \eta _ { t } \widehat { Q } _ { t } } \end{array}$ , which is (A.33). The martingale property of $M _ { T } ^ { \mathrm { p l a y } }$ follows from $\mathbb { E } [ c _ { t } ( A _ { t } ) \mid \mathcal { F } _ { t - 1 } ] = \langle \mu _ { t } , c _ { t } \rangle$ (since $A _ { t } \sim \mu _ { t } )$ , and that of $M _ { T } ^ { \mathrm { e s t } } ( u )$ from $\mathbb { E } [ \widehat { c } _ { t } - \bar { c } _ { t } \ | \ \mathcal { F } _ { t - 1 } ] = 0$ by the definition of $\bar { c } _ { t } ;$ the regrouping itself is a pathwise identity independent of these expectations. □

Proof of Proposition A.15. Conditioned on $\mathcal { F } _ { t - 1 }$ the estimator depends only on $A _ { t } ~ \sim ~ \mu _ { t }$ , so $\begin{array} { r l r l } { \mathbb E [ \widehat { Q } _ { t } } & { { } | } & { \mathcal F _ { t - 1 } ] } & { = } \end{array}$ $\begin{array} { r } { \sum _ { a } \mu _ { t } ( a ) Q _ { \eta _ { t } } ( p _ { t } , \widehat { z } _ { t } ^ { ( a ) } ) } \end{array}$ , and inserting the centered-RCGF definition from (2.7) gives $\left( \mathrm { A } . 3 4 \right)$ . For inverse-propensity scores, on $\left\{ A _ { t } = a \right\}$ only coordinate a is nonzero, so $\begin{array} { r } { \sum _ { i } p _ { t } ( i ) e ^ { - \eta _ { t } \widehat { c } _ { t } ( i ) } = 1 - p _ { t } \overline { { ( a ) } } + p _ { t } ( a ) e ^ { - \eta _ { t } c _ { t } ( a ) / \mu _ { t } ( a ) } } \end{array}$ while the centering term $\langle p _ { t } , \widehat { c } _ { t } \rangle$ has conditional mean $\langle p _ { t } , c _ { t } \rangle ; ( \mathrm { A } . 3 5 )$ follows. Expanding the logarithm to second order in η yields (A.36), and the exact variance identity is a direct second-moment computation. □

Proof of Corollary A.17. By Proposition A.16 the comparator complexity of competing against $\pi ^ { \star }$ is $\eta ^ { - 1 } \log ( 1 / \tilde { \pi } ( G ) )$ in the aggregate mass ${ \tilde { \pi } } ( G )$ of its loss-identical copies. Evaluating that expression at $\tilde { \pi } ( G ) = \beta _ { 0 }$ before duplication and at ${ \tilde { \pi } } ( G ) = \beta$ after, and subtracting, gives (A.39). □

Proof of Proposition A.16. The map $\nu \mapsto \mathrm { K L } ( \nu \| \tilde { \pi } )$ on $\Delta ( G )$ is minimized by the I-projection of π˜ onto the coordinate subset $G ,$ namely the renormalized prior $\nu ^ { \star } ( e ) = \tilde { \pi } ( e ) / \tilde { \pi } ( G )$ ; its value is $\begin{array} { r } { \sum _ { e \in G } \frac { \tilde { \pi } ( e ) } { \tilde { \pi } ( G ) } \log \frac { 1 } { \tilde { \pi } ( G ) } = - \log \tilde { \pi } ( G ) , \mathrm { g i v } ^ { - } } \end{array}$ ing (A.37). Because the experts in $G$ are loss-identical, $\left. \nu , \widehat { S } _ { T } \right.$ is the same for every $\nu \in \Delta ( G )$ and equals the cumulative estimated score of $\pi ^ { \star }$ ; the benchmark is therefore unchanged by replacing the designated policy with the lowest-complexity representative $\nu ^ { \star }$ . Substituting $\nu ^ { \star }$ and $\mathrm { K L } ( \nu ^ { \star } \| \tilde { \pi } ) = - \log \tilde { \pi } ( G )$ into (3.6) yields (A.38). □

Proof of Theorem A.18. (a) Writing $\begin{array} { r } { p _ { i } = d P _ { i } ^ { a } / d \mu , p _ { j } = d P _ { j } ^ { a } / d \mu , Z ( s , 1 - s ) = \int p _ { i } ^ { s } p _ { j } ^ { 1 - s } d \mu = \int p _ { j } \left( p _ { i } / p _ { j } \right) ^ { s } d \mu = } \end{array}$ $\mathbb { E } _ { P _ { i } ^ { a } } [ e ^ { s S } ] ,$ , and by the definition of $W _ { \eta }$ in Theorem 3.2 with prior $P _ { j } ^ { a }$ and score $S , s W _ { s } ( S ) =$ log $\mathbb { E } _ { P _ { i } ^ { a } } [ e ^ { s S } ] ~ =$ log $Z ( s , 1 - s ) ; ( \mathrm { A } . 4 1 )$ follows. (b) The Gibbs variational identity (2.6) gives $\rho _ { s , S } ( x ) \propto p _ { j } ( x ) e ^ { s S ( x ) } = p _ { i } ( x ) ^ { s } p _ { j } ( x ) ^ { 1 - s }$ which is $R _ { s } ;$ the relative-entropy-barycenter form is the $W = 2$ mixed-coincidence identity [8]. (c) By $( \mathsf { A } . 4 1 ) , C _ { s } =$ $- \log \mathbb { E } _ { P _ { i } ^ { a } } [ e ^ { s S } ] \ = : \ - \Lambda ( s )$ with Λ convex, so max $\begin{array} { r } { \mathrm { ~  ~ \psi ~ } _ { s } C _ { s } ~ = ~ - \operatorname* { m i n } _ { s } \Lambda ( s ) } \end{array}$ and $\Lambda ^ { \prime } ( s ^ { \star } ) ~ = ~ \mathbb { E } _ { R _ { s ^ { \star } } } [ S ] ~ = ~ 0$ . From (b), $\mathrm { K L } ( R \Vert \dot { P _ { i } ^ { a } } ) - \mathrm { K L } ( R \Vert P _ { i } ^ { a } ) = \mathbb { E } _ { R } [ \log ( p _ { j } / p _ { i } ) ] = - \mathbb { E } _ { R } [ S ]$ , which vanishes at $s ^ { \star } ;$ hence both equal their $s ^ { \star }$ -weighted average max<sub>s</sub> $C _ { s }$ , giving (A.43). The final claim is the definition of the edge-restricted multi-way coincidence radius and its MAP-error-exponent characterization [8] □

## D.7 Proofs from Section 8

Proof of (8.1). Let $I _ { t } \sim p _ { t }$ conditional on $\mathcal { F } _ { t - 1 }$ and set $\begin{array} { r } { M _ { T } ^ { \mathrm { s a m } } : = \sum _ { t = 1 } ^ { T } ( c _ { t } ( I _ { t } ) - \langle p _ { t } , c _ { t } \rangle ) } \end{array}$ . Then

$$
\widehat { R } _ { T } ^ { c } ( \nu ) = \sum _ { t = 1 } ^ { T } c _ { t } ( I _ { t } ) - \langle \nu , C _ { T } \rangle = \underbrace { \sum _ { t = 1 } ^ { T } ( c _ { t } ( I _ { t } ) - \langle p _ { t } , c _ { t } \rangle ) } _ { M _ { T } ^ { \mathrm { s a m } } } + \sum _ { t = 1 } ^ { T } \langle p _ { t } - \nu , c _ { t } \rangle
$$

Applying Theorem 4.3 to the second summand (full information, so $\hat { c } _ { t } = c _ { t } )$ gives exactly (8.1). The martingale-diference property $\mathbb { E } [ c _ { t } ( I _ { t } ) - \langle p _ { t } , c _ { t } \rangle \ | \ \mathcal { F } _ { t - 1 } ] = 0$ is immediate because $p _ { t } \mathrm { i } s \mathcal F _ { t - 1 }$ -measurable and $I _ { t } \mid \mathcal { F } _ { t - 1 } \sim p _ { t }$

For the high-probability bound, apply Freedman’s inequality. Each increment $X _ { t } : = c _ { t } ( I _ { t } ) - \langle p _ { t } , c _ { t } \rangle$ satisfies $| X _ { t } | \le 1$ (since $c _ { t } ( i ) \in [ 0 , 1 ] )$ , with conditional variance $\operatorname { V a r } ( X _ { t } \mid { \mathcal { F } } _ { t - 1 } ) = \operatorname { V a r } _ { p _ { t } } ( c _ { t } )$ , so the cumulative conditional variance is the variance clock $\begin{array} { r } { V _ { T } ^ { \mathrm { s a m } } = \sum _ { t } \mathrm { V a r } _ { p _ { t } } ( c _ { t } ) } \end{array}$ . Freedman’s inequality for bounded martingale diferences ([29], see also [28, Thm. 1]) gives, for every $\delta \in ( 0 , 1 )$

$$
\operatorname* { P r } \left[ M _ { T } ^ { \mathrm { s a m } } \ge \sqrt { 2 V _ { T } ^ { \mathrm { s a m } } \log ( 1 / \delta ) } + \frac 1 3 \log ( 1 / \delta ) \right] \le \delta .
$$

Since $c _ { t } ( i ) \in [ 0 , 1 ]$ implies $\mathrm { V a r } _ { p _ { t } } ( c _ { t } ) \leq 1 / 4$ , one has $V _ { T } ^ { \mathrm { s a m } } \leq T / 4$

For the anytime bound, use Ville’s inequality. Define, for each $\lambda \in ( 0 , 3 )$ , the nonnegative supermartingale

$$
E _ { t } ^ { ( \lambda ) } : = \exp ( \lambda M _ { t } ^ { \mathrm { s a m } } - \psi ( \lambda ) V _ { t } ^ { \mathrm { s a m } } ) , \qquad \psi ( \lambda ) : = \frac { \lambda ^ { 2 } / 2 } { 1 - \lambda / 3 } ,
$$

whose supermartingale property is the standard Bernstein-type bound $\mathbb { E } [ e ^ { \lambda X _ { t } } ~ \vert ~ \mathcal { F } _ { t - 1 } ] \le \exp ( \psi ( \lambda ) \mathrm { V a r } _ { p _ { t } } ( c _ { t } ) )$ for $| X _ { t } | \le 1$ . Ville’s maximal inequality [68] yields, for every $\delta \in ( 0 , 1 )$

$$
\operatorname* { P r } \left[ \exists t \geq 1 : ~ M _ { t } ^ { \mathrm { s a m } } \geq \frac { \psi ( \lambda ) V _ { t } ^ { \mathrm { s a m } } + \log ( 1 / \delta ) } { \lambda } \right] \leq \delta ,
$$

and optimizing over λ (or taking a geometric union over a dyadic grid of λ) produces the anytime envelope $M _ { t } ^ { \mathrm { s a m } } \lesssim$ $\sqrt { V _ { t } ^ { \mathrm { s a m } } \log ( \log ( V _ { t } ^ { \mathrm { s a m } } ) / \delta ) } + \log ( 1 / \delta )$ simultaneously in t.

The law-of-the-iterated-logarithm refinement follows from an almost-sure argument: integrate over λ with a Gamma $( 1 / 2 , 1 / 2 )$ mixture restricted to the sub-gamma range $\lambda \in \mathsf { ( 0 , 3 ) }$ on which each $E _ { t } ^ { ( \lambda ) }$ E(^) is a supermartingale. Define

$$
\bar { E } _ { t } : = \int _ { 0 } ^ { 3 } E _ { t } ^ { ( \lambda ) } \frac { \lambda ^ { - 1 / 2 } e ^ { - \lambda / 2 } } { \sqrt { 2 \pi } } \mathrm { d } \lambda ,
$$

the method of mixtures of [24] (the iterated-logarithm-optimal tilt $\lambda \asymp \sqrt { 2 \log \log V _ { t } ^ { \mathrm { s a m } } / V _ { t } ^ { \mathrm { s a m } } }$ lies interior to (0, 3) for large $V _ { t } ^ { \mathrm { s a m } }$ , so the truncation does not afect the rate). Because each $E _ { t } ^ { ( \lambda ) }$ is a supermartingale and the mixture preserves nonnegativity, $\bar { E } _ { t }$ is a nonnegative supermartingale with $\bar { E } _ { 0 } \leq 1$ . By a Gaussian-tail calculation ([24, §9.4]),

$$
\log \bar { E } _ { t } \geq \frac { ( M _ { t } ^ { \mathrm { s a m } } ) ^ { 2 } } { 2 ( V _ { t } ^ { \mathrm { s a m } } + \log \log V _ { t } ^ { \mathrm { s a m } } ) } - \frac 1 2 \log ( V _ { t } ^ { \mathrm { s a m } } + \log \log V _ { t } ^ { \mathrm { s a m } } ) + O ( 1 ) ,
$$

and Ville’s inequality gives s $\log _ { t } \bar { E } _ { t } < \infty$ almost surely. Combining the two displays,

$$
\operatorname* { l i m } _ { T \to \infty } \operatorname* { s u p } _ { \sqrt { 2 V _ { T } ^ { \mathrm { s a m } } \log \log V _ { T } ^ { \mathrm { s a m } } } } \leq 1 \quad \mathrm { a . s . }
$$

which implies the LIL statement $M _ { T } ^ { \mathrm { s a m } } = O \big ( \sqrt { V _ { T } ^ { \mathrm { s a m } } \log \log V _ { T } ^ { \mathrm { s a m } } } \big )$ a.s. for large $T .$

Proof of (8.3). Consider the local Bayesian update $p _ { t + 1 } ( i ) \propto p _ { t } ( i ) e ^ { - \eta _ { t } c _ { t } ( i ) }$ with one-round mix loss

$$
m _ { t } ( \eta _ { t } ) : = - \frac { 1 } { \eta _ { t } } \log \sum _ { i } p _ { t } ( i ) e ^ { - \eta _ { t } c _ { t } ( i ) } .
$$

Direct computation gives the pointwise identity

$$
\log \frac { p _ { t + 1 } ( i ) } { p _ { t } ( i ) } = - \eta _ { t } c _ { t } ( i ) - \log \sum _ { j } p _ { t } ( j ) e ^ { - \eta _ { t } c _ { t } ( j ) } = - \eta _ { t } \big ( c _ { t } ( i ) - m _ { t } ( \eta _ { t } ) \big ) .
$$

Multiplying by $\nu ( i )$ and summing,

$$
\sum _ { i } \nu ( i ) \log \frac { p _ { t + 1 } ( i ) } { p _ { t } ( i ) } = - \eta _ { t } \bigl ( \langle \nu , c _ { t } \rangle - m _ { t } ( \eta _ { t } ) \bigr ) ,
$$

and the left-hand side is $\mathrm { K L } ( \nu | | p _ { t } ) - \mathrm { K L } ( \nu | | p _ { t + 1 } )$ by the definition of relative entropy. Negating both sides,

$$
\mathrm { K L } ( \nu \| p _ { t + 1 } ) - \mathrm { K L } ( \nu \| p _ { t } ) = \eta _ { t } \big ( \langle \nu , c _ { t } \rangle - m _ { t } ( \eta _ { t } ) \big ) ,
$$

which is (8.3). Summing the per-round increments telescopes the left-hand side to $\mathrm { K L } ( \nu \| p _ { T + 1 } ) - \mathrm { K L } ( \nu \| \pi )$ , so

$$
\mathrm { K L } ( \nu \| \pi ) - \mathrm { K L } ( \nu \| p _ { T + 1 } ) = \sum _ { t = 1 } ^ { T } \eta _ { t } \big ( m _ { t } ( \eta _ { t } ) - \langle \nu , c _ { t } \rangle \big ) ,
$$

and the nonnegativity $\mathrm { K L } ( \nu \| p _ { T + 1 } ) \ge 0$ gives the information-budget bound $\begin{array} { r } { \sum _ { t } \eta _ { t } ( m _ { t } - \langle \nu , c _ { t } \rangle ) \leq \mathrm { K L } ( \nu \| \pi ) \leq \Gamma } \end{array}$ Under Thompson sampling with $I _ { t } \sim p _ { t }$ , the posterior log-weight of the sampled expert evolves as log $p _ { t + 1 } ( I _ { t } ) \gets$ log $p _ { t } ( I _ { t } ) = - \eta _ { t } ( c _ { t } ( I _ { t } ) - m _ { t } ( \eta _ { t } ) )$ ), whose conditional expectation given $\mathcal { F } _ { t - 1 } \mathrm { ~ i s ~ } - \eta _ { t } \delta _ { t } ( c )$ with $\delta _ { t } ( c ) : = \langle p _ { t } , c _ { t } \rangle -$ $m _ { t } ( \eta _ { t } ) \geq 0$ (nonnegativity follows fromJensen’s inequality applied to the log-exp). The residual log $p _ { t + 1 } ( I _ { t } ) - \log p _ { t } ( I _ { t } ) +$ $\eta _ { t } \delta _ { t } ( c )$ is therefore a martingale-diference sequence. The prior-posterior ratio $\mathrm { P R } _ { t } ( \theta ) : = \pi ( \theta ) / \rho _ { t , \eta } ( \theta )$ at a constant scale η inherits the multiplicative version of this identity: log ${ \mathrm { P R } } _ { t } ( \theta ) - \log { \mathrm { P R } } _ { t - 1 } ( \theta ) = \eta ( c _ { t } ( \theta ) - m _ { t } )$ , so the exponentiated ratio is an e-process under the usual hypotheses. □

Proof of Theorem A.20. Part (i) is the martingale-diference property: $w _ { t }$ is F<sub>t−1</sub>-measurable and $I _ { t } \ | \ \mathcal { F } _ { t - 1 } \sim w _ { t } .$ , so $\mathbb { E } [ c _ { t } ( I _ { t } ) \mid \mathcal { F } _ { t - 1 } ] = \langle w _ { t } , c _ { t } \rangle ; \mathrm { a t } w _ { t } = p _ { t }$ , summing gives $\mathbb { E } [ M _ { T } ^ { \mathrm { s a m } } ] = 0$ and (8.2). Part (ii) is the equalizer identity (3.3): at $p = \rho _ { \eta , S } = \nabla W _ { \eta } ( S ) , W _ { \eta } ( S + z ) - W _ { \eta } ( S ) = \eta Q _ { \eta } ( p , z )$ for every feasible z, which $\mathrm { i s } \leq \eta q$ with equality at saturation and independent of the direction of z. The first term of (A.44) vanishes by (i) and the value-to-go terms telescope to $\Gamma / \eta + \eta \sum _ { t } q _ { t }$ , attained by nature’s budget-saturating tilt in the comparator-Gibbs direction, with the equalizer making nature indiferent across directions. For part (iii), the value-to-go increment $W _ { \eta } ( S _ { t - 1 } + z ) - W _ { \eta } ( S _ { t - 1 } ) = \eta Q _ { \eta } ( \rho _ { \eta , S _ { t - 1 } } , z )$ depends on the state and on z alone (the play-excess term of (A.44) vanishes in expectation for every $w _ { t }$ by part (i)). Nature’s efective per-round objective is therefore η times the centered RCGF of z under the Gibbs law $\rho _ { \eta , S _ { t - 1 } } -$ the same maximizer, since $\eta > 0 -$ subject to the feasibility $Q _ { \eta } ( w _ { t } , z ) \leq q _ { t }$ . When $w _ { t } = p _ { t } = \rho _ { \eta , S _ { t - 1 } }$ the feasible set is exactly the sublevel set of that objective, so the supremum is ηq<sub>t</sub>, attained on the whole budget sphere (the equalizer of (ii)). When $w _ { t } \neq p _ { t }$ the forms $Q _ { \eta } ( w _ { t } , \cdot )$ and $Q _ { \eta } ( p _ { t } , \cdot )$ difer, so a feasible direction has $Q _ { \eta } ( p _ { t } , z ) > q _ { t }$ , from which the increment exceeds $\eta q _ { t }$ and sup $_ z \Phi _ { T } > \Gamma / \eta + \eta \sum _ { t } q _ { t } ;$ the Gibbs sampler alone attains the value and is the unique minimizer.

Proof of Proposition A.21. Part (i) is immediate: subtracting (8.2) from (8.1) cancels every deterministic term and leaves the single martingale $M _ { T } ^ { \mathrm { s a m } }$ , which is mean-zero and free of ν by construction. For part (ii), suppose the realized payof Φ were constant over the sampling randomness. On any round where $p _ { t }$ places positive mass on two experts $i \neq j$ with $c _ { t } ( i ) \neq c _ { t } ( j )$ , the two transcripts that difer only in $I _ { t } \in \{ i , j \}$ each have positive probability and must give equal $\Phi ;$ ranging over all such rounds and pairs forces Φ to be independent of $I _ { 1 : T }$ . Hence Φ does not depend on the sampled play at all: it is the deterministic game, in which the learner plays the mixture $p _ { t }$ without sampling. Any payof that genuinely depends on $c _ { t } ( I _ { t } )$ therefore leaves, on a.e. path, the per-round variance $\mathrm { V a r } _ { p _ { t } } ( c _ { t } ) > 0 .$ . For part (iii), sharpen the sampling law to $w _ { t } ^ { ( \beta ) } \propto p _ { t } ^ { \beta } ;$ ; the full-information posterior path is unchanged. By Theorem A.20 the expected payof is stationary at $\beta = 1$ , but the sampling variance $\begin{array} { r } { V _ { T } ^ { \mathrm { s a m } } ( \beta ) = \sum _ { t } \operatorname { V a r } _ { w _ { t } ^ { ( \beta ) } } ( c _ { t } ) } \end{array}$ is not: $\begin{array} { r } { \left. \frac { d } { d \beta } V _ { T } ^ { \mathrm { s a m } } ( \beta ) \right| _ { \beta = 1 } \neq 0 } \end{array}$ generically. Since, under the martingale central-limit approximation, any upper- $( 1 - \delta )$ -quantile of the realized payof is value $+ z _ { \delta } \sqrt { V _ { T } ^ { \mathrm { s a m } } ( \beta ) } + o ( \cdot )$ with $z _ { \delta }$ the Gaussian quantile, its minimizer is a law with $\beta ^ { * } \neq 1$ , moving in whichever direction lowers $V _ { T } ^ { \mathrm { s a m } } ( \beta )$ ; this is a leading-order statement under that approximation, and falls short of an exact quantile identity. □

## D.8 Proofs from Section 9

Proof of (9.1). Let $p _ { 1 } = u _ { N }$ be the uniform distribution on $[ N ]$ , and let $p _ { t + 1 } ( i ) \propto p _ { t } ( i ) e ^ { - \alpha _ { t } g _ { t } ( i ) }$ for $t = 1 , \ldots , T .$ . By induction on $t ,$

$$
p _ { t + 1 } ( i ) = \frac { u _ { N } ( i ) e ^ { - M _ { t } ( i ) } } { Z _ { t } } , \qquad Z _ { t } : = \sum _ { j } u _ { N } ( j ) e ^ { - M _ { t } ( j ) } = N ^ { - 1 } \sum _ { j } e ^ { - M _ { t } ( j ) } .
$$

In particular, $Z _ { T } = L _ { T } ^ { \mathrm { e x p } }$ , and log $p _ { T + 1 } ( i ) = \log u _ { N } ( i ) - M _ { T } ( i ) - \log L _ { T } ^ { \exp }$ . Take expectation under any comparator $\nu \in \Delta ( [ N ] )$

$$
\sum _ { i } \nu ( i ) \log p _ { T + 1 } ( i ) = \sum _ { i } \nu ( i ) \log u _ { N } ( i ) - \langle \nu , M _ { T } \rangle - \log L _ { T } ^ { \mathrm { e x p } }
$$

Rewriting the left-hand side via relative entropy,

$$
\sum _ { i } \nu ( i ) \log p _ { T + 1 } ( i ) = \sum _ { i } \nu ( i ) \log u _ { N } ( i ) - \left( \mathrm { K L } ( \nu \| p _ { T + 1 } ) - \mathrm { K L } ( \nu \| u _ { N } ) \right) ,
$$

so that $- \big ( \mathrm { K L } ( \nu \| p _ { T + 1 } ) - \mathrm { K L } ( \nu \| u _ { N } ) \big ) = - \left. \nu , M _ { T } \right. - \log L _ { T } ^ { \mathrm { e x p } }$ . Rearranging,

$$
- \log L _ { T } ^ { \mathrm { e x p } } + \mathrm { K L } ( \nu \| p _ { T + 1 } ) = \mathrm { K L } ( \nu \| u _ { N } ) + \left. \nu , M _ { T } \right. ,
$$

which is (9.1). Taking the infimum over ν and using in $\dot { \bar { \mathbf { \rho } } } _ { \nu } \operatorname { K L } ( \nu \| p _ { T + 1 } ) \ = \ 0$ at $\nu ~ = ~ p _ { T + 1 }$ immediately gives the Donsker–Varadhan dual form $- \log L _ { T } ^ { \mathrm { e x p } } \ = \ \operatorname * { i n f } _ { \nu } \{ \langle \nu , M _ { T } \rangle + \mathrm { K L } ( \nu \| u _ { N } ) \}$ , and the margin-tail bound $E _ { T } ( \theta ) \ \leq$ $e ^ { \theta A _ { T } - \operatorname* { i n f } _ { \nu } \left\{ \langle \nu , M _ { T } \rangle + \mathrm { K L } ( \nu \| u _ { N } ) \right\} }$ follows by a standard Chernof step: for the fraction of examples with $M _ { T } ( i ) / A _ { T } \leq \theta ,$

$$
\begin{array} { r } { E _ { T } ( \theta ) \le \mathbb { E } _ { i \sim u _ { N } } \Big [ e ^ { \theta A _ { T } - M _ { T } ( i ) } \Big ] = e ^ { \theta A _ { T } } L _ { T } ^ { \mathrm { e x p } } . } \end{array}
$$

Proof of Proposition A.22. For binary $g _ { t } , \ \left. p _ { t } , e ^ { - \alpha g _ { t } } \right. \ = \ \varepsilon _ { t } e ^ { \alpha } \ + \ ( 1 \ - \ \varepsilon _ { t } ) e ^ { - \alpha }$ is minimized at $\begin{array} { r c l } { \alpha _ { t } ^ { \star } } & { = } & { \frac { 1 } { 2 } \log \frac { 1 - \varepsilon _ { t } } { \varepsilon _ { t } } } \end{array}$ with value $2 \sqrt { \varepsilon _ { t } ( 1 - \varepsilon _ { t } ) } = \sqrt { 1 - ( 1 - 2 \varepsilon _ { t } ) ^ { 2 } } = \sqrt { 1 - 4 \gamma _ { t } ^ { 2 } }$ , using $2 \gamma _ { t } ~ = ~ 1 - 2 \varepsilon _ { t } ;$ the pressure rule sets $a _ { t } = - \alpha _ { t } ^ { - 1 } \log \left. p _ { t } , e ^ { - \alpha _ { t } g _ { t } } \right.$ , so $\alpha _ { t } ^ { \star } a _ { t } = - \log \sqrt { 1 - 4 \gamma _ { t } ^ { 2 } }$ , which is (A.47). Equation (A.48) is (9.2) summed over $t ,$ and (A.49) follows from $- \log ( 1 - x ) \geq x$ □

Proof of (A.52). Let $\begin{array} { r } { p _ { t } ( y ) = \left[ \prod _ { e = 1 } ^ { N } \pi _ { t , e } ( y ) ^ { \alpha _ { t , e } } \right] / Z _ { t } \mathrm { w i t h } Z _ { t } : = \sum _ { y \in Y } \prod _ { e = 1 } ^ { N } \pi _ { t , e } ( y ) ^ { \alpha _ { t , e } } . } \end{array}$ Taking − log of $p _ { t } ( y _ { t } )$

$$
- \log p _ { t } ( y _ { t } ) = - \sum _ { e = 1 } ^ { N } \alpha _ { t , e } \log \pi _ { t , e } ( y _ { t } ) + \log Z _ { t } = \sum _ { e = 1 } ^ { N } \alpha _ { t , e } \big ( - \log \pi _ { t , e } ( y _ { t } ) \big ) - C _ { \alpha _ { t } } ( \pi _ { t , 1 : N } ) ,
$$

where $C _ { \alpha _ { t } } ( \pi _ { t , 1 : N } ) : = - \log Z _ { t }$ . This is (A.52).

When $\begin{array} { l } { \sum _ { e } \alpha _ { t , e } } & { = ~ 1 } \end{array}$ , the AM–GM (or Jensen applied to log) inequality gives, for each y, $\begin{array} { r l } { \prod _ { e } \pi _ { t , e } ( y ) ^ { \alpha _ { t , e } } } & { { } \leq } \end{array}$ $\textstyle \sum _ { e } \alpha _ { t , e } \pi _ { t , e } ( y )$ , so $Z _ { t } \leq 1$ and $C _ { \alpha _ { t } } \geq 0$ . In particular, the pooled log loss is bounded above by the weighted expert log loss and below by the same minus the coincidence discount; the discount is zero if and only if all non-null experts assign the same probability at every outcome (identical predictive distributions), and agreement at $y _ { t }$ alone does not sufice.

Proof of Proposition A.23. Dividing log $\begin{array} { r } { \mathbb { E } _ { p } e ^ { \eta z } \ = \ \sum _ { n > 2 } \eta ^ { n } \kappa _ { n } ( z ) / n ! } \end{array}$ by $\eta ^ { 2 }$ gives $\begin{array} { r l r } { Q _ { \eta } ( p , z ) } & { { } = } & { \sum _ { n > 2 } \eta ^ { n - 2 } \kappa _ { n } ( z ) / n ! ; } \end{array}$ subtracting the $n = 2$ term is (A.53). Equation $\mathrm { ( A . } 5 4 )$ is Hoefding’s lemma written as an exact slack, i.e. the per-round summand of $\Delta _ { T } ^ { \mathrm { c l a s s } } \mathrm { i n } \left( \mathrm { A } . 5 \right)$ . The trajectory statement is the round-wise sum of the two. □

## Acknowledgments

Large language models were used in producing this work, for which the author is solely responsible.

## References

[1] Jacob Abernethy, Peter L. Bartlett, Alexander Rakhlin, and Ambuj Tewari. Optimal strategies and minimax lower bounds for online convex games. In Proceedings of the 21st Annual Conference on Learning Theory (COLT), pages 415–424, 2008.

[2] Jacob D. Abernethy and Manfred K. Warmuth. Repeated games against budgeted adversaries. In Advances in Neural Information Processing Systems 23 (NIPS 2010), 2010.

[3] Noga Alon, Nicolò Cesa-Bianchi, Ofer Dekel, and Tomer Koren. Online learning with feedback graphs: Beyond bandits. In Proceedings of the 28th Conference on Learning Theory (COLT), volume 40 of Proceedings of Machine Learning Research, pages 23–35, 2015.

[4] Pierre Alquier, James Ridgway, and Nicolas Chopin. On the properties of variational approximations of Gibbs posteriors. Journal of Machine Learning Research, 17(236):1–41, 2016.

[5] Sanjeev Arora, Elad Hazan, and Satyen Kale. The multiplicative weights update method: a meta-algorithm and applications. Theory of Computing, 8(6):121–164, 2012. doi: 10.4086/toc.2012.v008a006.

[6] Peter Auer, Nicolò Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. The nonstochastic multiarmed bandit problem. SIAMJournal on Computing, 32(1):48–77, 2002. doi: 10.1137/S0097539701398375.

[7] Akshay Balsubramani. Adaptive Bayes exactly tracks information over intrinsic time. arXiv preprint arXiv:2607.08789, 2026.

[8] Akshay Balsubramani. Information from coincidences. arXiv preprint arXiv:2606.25042, 2026.

[9] Akshay Balsubramani and Yoav Freund. Scalable semi-supervised aggregation of classifiers. In Advances in Neural Information Processing Systems (NIPS), 2015.

[10] Akshay Balsubramani and Yoav S Freund. Optimal binary classifier aggregation for general losses. In Advances in Neural Information Processing Systems, pages 5032–5039, 2016.

[11] Avrim Blum and Yishay Mansour. From external to internal regret. Journal of Machine Learning Research, 8:1307–1324, 2007. doi: 10.5555/1314498.1314543.

[12] Noam Brown and Tuomas Sandholm. Solving imperfect-information games via discounted regret minimization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 1829–1836, 2019. doi: 10.1609/aaai. v33i01.33011829.

[13] Sébastien Bubeck, Nikhil R Devanur, Zhiyi Huang, and Rad Niazadeh. Multi-scale online learning: Theory and applications to online auctions and pricing. Journal of Machine Learning Research, 20(62):1–37, 2019.

[14] Olivier Catoni. PAC-Bayesian Supervised Classification: The Thermodynamics of Statistical Learning, volume 56 of Institute of Mathematical Statistics Lecture Notes–Monograph Series. Institute of Mathematical Statistics, Beachwood, OH, 2007. doi: 10.1214/074921707000000391.

[15] Nicolo Cesa-Bianchi and Gábor Lugosi. Prediction, Learning, and Games. Cambridge University Press, 2006. doi: 10.1017/CBO9780511546921.

[16] Kamalika Chaudhuri, Yoav Freund, and Daniel Hsu. A parameter-free hedging algorithm. In Advances in Neural Information Processing Systems (NeurIPS), 2009.

[17] Herman Chernof. A measure of asymptotic eficiency for tests of a hypothesis based on the sum of observations. Annals of Mathematical Statistics, 23(4):493–507, 1952. doi: 10.1214/aoms/1177729330.

[18] B.S. Clarke and A.R. Barron. Jefreys’ prior is asymptotically least favorable under entropy risk. Journal of Statistical Planning and Inference, 41:37–60, 1994. doi: 10.1016/0378-3758(94)90153-8.

[19] Christian Coester and James R. Lee. Pure entropic regularization for metrical task systems. In Proceedings of the 32nd Annual Conference on Learning Theory (COLT), volume 99 of Proceedings of Machine Learning Research, pages 835–848, 2019.

[20] I. Csiszár. I-divergence geometry of probability distributions and minimization problems. Annals of Probability, 3(1): 146–158, 1975. doi: 10.1214/aop/1176996454.

[21] Ashok Cutkosky. Combining online learning guarantees. In Conference on Learning Theory (COLT), 2019.

[22] Constantinos Daskalakis, Andrew Ilyas, Vasilis Syrgkanis, and Haoyang Zeng. Training GANs with optimism. In International Conference on Learning Representations (ICLR), 2018.

[23] Victor H. de la Peña and Michael J. Klass. The exact Ville identity: From the absorbing case to the general law with an application to E-values. arXiv preprint arXiv:2607.04620, 2026.

[24] Victor H. de la Peña, Tze Leung Lai, and Qi-Man Shao. Self-Normalized Processes: Limit Theory and Statistical Applications. Probability and its Applications. Springer, Berlin, Heidelberg, 2009. doi: 10.1007/978-3-540-85636-8.

[25] Steven de Rooij, Tim van Erven, Peter D. Grünwald, and Wouter M. Koolen. Follow the leader if you can, hedge if you must. Journal of Machine Learning Research, 15:1281–1316, 2014. doi: 10.5555/2627435.2638576.

[26] Amir Dembo and Ofer Zeitouni. Large Deviations Techniques and Applications. Stochastic Modelling and Applied Probability. Springer Berlin Heidelberg, 2nd, corrected reprint edition, 2010. doi: 10.1007/978-3-642-03311-7.

[27] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12:2121–2159, 2011.

[28] Xiequan Fan, Ion Grama, and Quansheng Liu. Hoefding’s inequality for supermartingales. Stochastic Processes and their Applications, 122(10):3545–3559, 2012. doi: 10.1016/j.spa.2012.06.009.

[29] David A. Freedman. On tail probabilities for martingales. Annals of Probability, 3(1):100–118, 1975. doi: 10.1214/ aop/1176996452.

[30] Yoav Freund. Open problem: Second order regret bounds based on scaling time. In Conference on Learning Theory (COLT), 2016.

[31] Yoav Freund and Robert E. Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. Journal of Computer and System Sciences, 55(1):119–139, 1997. doi: 10.1006/jcss.1997.1504.

[32] Peter D. Grünwald. The safe bayesian: Learning the learning rate via the mixability gap. In Algorithmic Learning Theory (ALT 2012), volume 7568 of Lecture Notes in Computer Science, pages 169–183. Springer, 2012. doi: 10.1007/ 978-3-642-34106-9\_16.

[33] Sergiu Hart and Andreu Mas-Colell. A simple adaptive procedure leading to correlated equilibrium. Econometrica, 68 (5):1127–1150, 2000. doi: 10.1111/1468-0262.00153.

[34] David Haussler and Manfred Opper. Mutual information, metric entropy and cumulative relative entropy risk. The Annals of Statistics, 25(6):2451–2492, 1997. doi: 10.1214/aos/1030741081.

[35] Mark Herbster and Manfred K. Warmuth. Tracking the best expert. Machine Learning, 32(2):151–178, 1998. doi: 10.1023/a:1007424614876.

[36] Steven R. Howard, Aaditya Ramdas, Jon McAulife, and Jasjeet Sekhon. Time-uniform, nonparametric, nonasymptotic confidence sequences. The Annals of Statistics, 49(2):1055–1080, 2021. doi: 10.1214/20-AOS1991.

[37] Adam Kalai and Santosh Vempala. Eficient algorithms for online decision problems. Journal of Computer and System Sciences, 71(3):291–307, 2005. doi: 10.1016/j.jcss.2004.10.016.

[38] Emilie Kaufmann and Wouter M. Koolen. Mixture martingales revisited with applications to sequential tests and confidence intervals. Journal of Machine Learning Research, 22(246):1–44, 2021.

[39] Mohammad Emtiyaz Khan and Håvard Rue. The bayesian learning rule. Journal of Machine Learning Research, 24 (281):1–46, 2023.

[40] Johannes Kirschner, Andreas Krause, Michele Meziu, and Mojmir Mutny. Confidence estimation via sequential likelihood mixing, 2025.

[41] Tomáš Kocák, Gergely Neu, Michal Valko, and Rémi Munos. Eficient learning by implicit exploration in bandit problems with side observations. In Advances in Neural Information Processing Systems (NeurIPS), 2014.

[42] Wouter M. Koolen and Tim van Erven. Second-order quantile methods for experts and combinatorial games. In Conference on Learning Theory (COLT), 2015.

[43] Martin Larsson, Aaditya Ramdas, and Johannes Ruf. The numeraire e-variable and reverse information projection, 2024.

[44] Gábor Lugosi and Gergely Neu. Online-to-PAC conversions: generalization bounds via regret analysis, 2023.

[45] Haipeng Luo and Robert E. Schapire. A drifting-games analysis for online learning and applications to boosting. In Advances in Neural Information Processing Systems (NeurIPS) 27, pages 1368–1376, 2014.

[46] Teodor V. Marinov and Julian Zimmert. The pareto frontier of model selection for general contextual bandits. In Advances in Neural Information Processing Systems (NeurIPS) 34, 2021.

[47] H. Brendan McMahan. A survey of algorithms and analysis for adaptive online learning. Journal of Machine Learning Research, 18, 2017.

[48] Neri Merhav and Meir Feder. Universal prediction. IEEE Transactions on Information Theory, 44(6):2124–2147, 1998. doi: 10.1109/18.720534.

[49] Francesco Orabona and Kwang-Sung Jun. Tight concentrations and confidence sequences from the regret of universal portfolio. IEEE Transactions on Information Theory, 70(1):436–455, 2024. doi: 10.1109/TIT.2023.3330187.

[50] Francesco Orabona and Dávid Pál. Coin betting and parameter-free online learning. Advances in Neural Information Processing Systems (NeurIPS), 2016.

[51] Gerasimos Palaiopanos, Ioannis Panageas, and Georgios Piliouras. Multiplicative weights update with constant step-size in congestion games: Convergence, limit cycles and chaos. In Advances in Neural Information Processing Systems (NeurIPS) 30, pages 5872–5882, 2017.

[52] Muriel F. Pérez-Ortiz and Wouter M. Koolen. Luckiness in multiscale online learning. In Advances in Neural Information Processing Systems (NeurIPS) 35, 2022. doi: 10.52202/068431-1824.

[53] Alexander Rakhlin and Karthik Sridharan. Optimization, learning, and games with predictable sequences. In Advances in Neural Information Processing Systems (NeurIPS) 26, pages 3066–3074, 2013.

[54] Alexander Rakhlin and Karthik Sridharan. Online non-parametric regression. In Proceedings of the 27th Annual Conference on Learning Theory (COLT), volume 35, pages 1232–1264, 2014.

[55] Aaditya Ramdas and Tudor Manole. Randomized and exchangeable improvements of markov’s, chebyshev’s and chernof’s inequalities. Statistical Science, 41(1):121–142, 2026.

[56] Aaditya Ramdas, Peter Grünwald, Vladimir Vovk, and Glenn Shafer. Game-theoretic statistics and safe anytime-valid inference. Statistical Science, 38(4):576–601, 2023. doi: 10.1214/23-STS894.

[57] Chloé Rouyer, Dirk van der Hoeven, Nicolò Cesa-Bianchi, and Yevgeny Seldin. A near-optimal best-of-both-worlds algorithm for online learning with feedback graphs. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

[58] Daniel Russo and Benjamin Van Roy. Learning to optimize via posterior sampling. Mathematics of Operations Research, 39(4):1221–1243, 2014. doi: 10.1287/moor.2014.0650.

[59] Daniel Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, and Zheng Wen. A tutorial on thompson sampling. Foundations and Trends in Machine Learning, 11(1):1–96, 2018. doi: 10.1561/9781680834710.

[60] Robert E. Schapire. Drifting games. Machine Learning, 43(3):265–291, 2001. doi: 10.1023/a:1010800213066

[61] Robert E. Schapire and Yoav Freund. Boosting: Foundations and Algorithms. MIT Press, 2012. doi: 10.7551/mitpress/ 8291.001.0001.

[62] Rocco A. Servedio. Smooth boosting and learning with malicious noise. Journal of Machine Learning Research, 4: 633–648, 2003.

[63] Glenn Shafer. Testing by betting: a strategy for statistical and scientific communication. Journal of the Royal Statistical Society: Series A, 184(2):407–431, 2021. doi: 10.1111/rssa.12647.

[64] Glenn Shafer and Vladimir Vovk. Probability and finance: It’s only a game! John Wiley & Sons, 2001. doi: 10.1002/ 0471249696.

[65] Glenn Shafer and Vladimir Vovk. Game-Theoretic Foundations for Probability and Finance. Wiley, 2019. doi: 10.1002/9781118548035.

[66] V. Syrgkanis, A. Agarwal, H. Luo, and R. E. Schapire. Fast convergence of regularized learning in games. In Advances in Neural Information Processing Systems, pages 2989–2997, 2015.

[67] Tim van Erven, Peter D. Grünwald, Nishant A. Mehta, Mark D. Reid, and Robert C. Williamson. Fast rates in statistical and online learning. Journal of Machine Learning Research, 16:1793–1861, 2015.

[68] Jean Ville. Étude critique de la notion de collectif. Number 218 in Thèses de l’entre-deux-guerres. Gauthier-Villars, Paris, 1939.

[69] Jun-Kun Wang, Jacob Abernethy, and Kfir Y. Levy. No-regret dynamics in the fenchel game: A unified framework for algorithmic convex optimization. Mathematical Programming, 205:203–268, 2024. doi: 10.1007/s10107-023-01976-y.

[70] Ian Waudby-Smith and Aaditya Ramdas. Estimating means of bounded random variables by betting. Journal of the Royal Statistical Society: Series B, 86(1):1–27, 2024. doi: 10.1093/jrsssb/qkad009.

[71] Ian Waudby-Smith, Ricardo Sandoval, and Michael I. Jordan. Universal log-optimality for general classes of e-processes and sequential hypothesis tests, 2025.

[72] Lin Xiao. Dual averaging methods for regularized stochastic learning and online optimization. Journal of Machine Learning Research, 11:2543–2596, 2010. doi: 10.5555/1756006.1953017.

[73] J. Zimmert and Y. Seldin. Tsallis-inf: An optimal algorithm for stochastic and adversarial bandits. Journal of Machine Learning Research, 22(28):1–49, 2021.

[74] Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In International Conference on Machine Learning (ICML), 2003.