# DCM Bandits: Multiplayer Information Asymmetric Cascading Bandits for Multiple Clicks

Andy Wang<sup>∗</sup> Department of Computer Science University of California, Los Angeles Los Angeles, USA dxwang@ucla.edu

Charlton Shih<sup>∗</sup> Department of Computer Science University of California, Los Angeles Los Angeles, USA charltonshih645@g.ucla.edu

William Chang Department of Mathematics University of California, Los Angeles Los Angeles, USA chang314@g.ucla.edu

Abstract—In this work, we extend the Dependent Click Model (DCM) Bandits to a multiplayer information-asymmetric setting, where multiple agents interact with a shared ranked list and may observe multiple clicks per session, introducing new challenges for selection strategies. We study asymmetry in (1) actions and (2) rewards, providing sublinear regret guarantees for three settings where at least one asymmetry is present. Establishing matching information-theoretic lower bounds for these settings is left as an open problem. We further show that for small termination probabilities, the termination ranking need not be known, improving on prior single-agent results. Experiments confirm that our algorithms perform well across asymmetric environments, and highlight the critical role of feedback structure—full versus first-click—in coordinating exploration and minimizing regret.

Index Terms—multi-armed bandits, cascading bandits, multiplayer learning, information asymmetry, dependent click model, online learning

## I. INTRODUCTION

The cascade model [1] and its dependent-click variant (DCM) [2], [3] are now standard frameworks for ranking under bandit feedback. In DCM, a user examines a ranked list, may click on multiple items, and terminates the session at any point governed by slot-specific satisfaction probabilities. The literature has produced regret-optimal single-agent algorithms [1], [3]–[6] and extensions to combinatorial actions [7], [8], user fatigue [9], exposure bias [10], and adversarial robustness [11].

Modern recommender systems, however, are rarely singleagent: industrial deployments involve multiple modules or stakeholders that jointly determine what is shown, each with partial state and partial feedback [12], [13]. This motivates a decentralized view of cascading bandits in which several players contribute distinct components of the joint ranking and observe different parts of the feedback. The multiplayer bandit literature [14]–[19] addresses coordination under such constraints, but assumes simple per-action rewards; the clickmodel literature, conversely, assumes a single decision-maker. The intersection has, to our knowledge, not been studied.

a) Our Contribution.: We introduce multiplayer cascading bandits with multi-click feedback and develop algorithms for three settings of information asymmetry:

• Problem A (action asymmetry). mCascadeUCB: a decentralized UCB algorithm in which players jointly select the top-K items by upper confidence bound and update shared statistics from observed cascade feedback.

• Problem B (reward asymmetry). mCascadeUCB-Intervals-Ranking: an elimination algorithm in which separated confidence intervals implicitly signal suboptimality through action choices. A round-robin variant removes assumptions required in [3] when termination probabilities are small.

• Problem C (both). mMDSEE-TopK: a phased explorethen-commit algorithm requiring no communication, extending [16], [17] to multi-slot cascade feedback.

We prove sublinear regret for all three algorithms and validate them empirically. To our knowledge this is the first systematic study of decentralized multi-click cascading bandits. Sections II–IV present the model, algorithms with regret analyses, and experiments; Section V discusses related work.

## II. PROBLEM STATEMENTS

## A. Single-Agent Cascading Bandits

We first review the classical single-agent cascading bandit model, which captures user interaction with ranked lists under the cascade hypothesis: users scan items sequentially and click the first attractive item, if any.

Let E be a ground set of L items. In each round $t = 1 , \dots , T$ , the agent recommends an ordered list $A _ { t } \ =$ $( e _ { 1 } , \dots , e _ { K } )$ of K distinct items from E. Each item $e \in E$ has an unknown attraction probability $w ( e ) \in [ 0 , 1 ]$ , representing the probability that a user clicks on the item when examined. Attraction probabilities are assumed independent and stationary.

The agent receives reward 1 if any item is clicked and 0 otherwise. The objective is to learn the optimal ranking $e ^ { \star } = \quad$ $( e _ { 1 } ^ { \star } , \ldots , e _ { K } ^ { \star } )$ maximizing the probability that at least one item is clicked:

$$
e ^ { \star } = \arg \operatorname* { m a x } _ { A \in \Pi _ { K } ( E ) } \left( 1 - \prod _ { k = 1 } ^ { K } ( 1 - w ( e _ { k } ) ) \right) ,\tag{1}
$$

where $\Pi _ { K } ( E )$ denotes the set of ordered K-subsets of E.

## B. Multi-Agent Cascading Bandits

We now consider a decentralized multi-agent extension with M players. Each player $i \in \{ 1 , \ldots , M \}$ has a local action set $E ^ { i }$ with $| E ^ { i } | = L$

In each round, player i selects K local items from $E ^ { i }$ The players’ choices combine to form a joint ranking $A _ { t } =$ $( { \pmb a } _ { 1 } , \dots , { \pmb a } _ { K } )$ , where each joint item $\mathbf { { a } } _ { k } \ = \ ( a _ { k } ^ { 1 } , . . . , a _ { k } ^ { M } )$ contains one action from each player. Thus player i controls the i-th coordinate $a _ { k } ^ { i } \in E ^ { i }$ of each joint item.

This setting captures two key features: (i) local autonomy, where each player controls only their own marginal choice from $E ^ { i } ;$ and (ii) global coordination, where the joint ranking emerges from the simultaneous choices of all players.

Feedback remains partial as in the cascade model: agents observe the clicked joint item (if any), which reveals that earlier items were unattractive while later items remain unobserved. The reward is shared across players and equals 1 if any item is clicked.

To maintain decentralization, agents may agree on a protocol before learning begins but cannot communicate during execution. The joint action space is large: there are $L ^ { M }$ joint items and $\frac { ( L ^ { \stackrel { N } { } } ) ! } { ( L ^ { M } - K ) ! }$ possible rankings of size $K .$

We consider three information structures:

a) Problem A: Action asymmetry.: Players cannot observe other players’ actions, but all players observe the same click feedback.

b) Problem B: Reward asymmetry.: Players observe each other’s actions but receive independent (i.i.d.) click feedback.

c) Problem C: Action and reward asymmetry.: Players receive i.i.d. click feedback and cannot observe the actions of other players.

## C. Multi-Agent Cascading Bandits with Multiple Clicks

We extend the model by introducing slot-dependent termination probabilities. In addition to attraction probabilities, each slot j has a termination probability $v _ { j }$ with $1 \leq j \leq K$

When a user clicks an item at position $j ,$ the session terminates with probability $v _ { j } ;$ otherwise the user continues examining later items. The termination probability is independent of the item and the round.

As before, feedback is partial: items before the click are observed to be unattractive, the clicked item is attractive, and later items remain unobserved. Under this model the optimal ranking becomes

$$
e ^ { \star } = \arg \operatorname* { m a x } _ { A \in \Pi _ { K } ( E ) } \left( 1 - \prod _ { j = 1 } ^ { K } ( 1 - v _ { j } w ( e _ { j } ) ) \right) .\tag{2}
$$

The termination probabilities allow multiple clicks within a session, increasing the amount of feedback per round while also introducing additional uncertainty for learning and coordination.

Algorithm 1 mCascadeUCB-A   
Require: M players with action sets $E ^ { i }$ of size $L ,$ horizon   
T, list size $K$   
1: $\mathbf { E }  E ^ { 1 } \times \cdot \cdot \cdot \times E ^ { M }$   
2: Initialize $\widehat { w } ( \mathbf { e } ) \gets 0 , n ( \mathbf { e } ) \gets 0$ for all $\mathbf { e } \in \mathbf { E }$   
3: for $t = 1$ to T do   
4: Compute $\operatorname { U C B } _ { t } ( \mathbf { e } )$ for all $\mathbf { e } \in \mathbf { E }$   
5: Select top-K items $( \mathbf { e } _ { 1 } , \ldots , \mathbf { e } _ { K } )$ by $\mathrm { U C B } _ { t }$   
6: Players extract their components and observe click $C _ { t }$   
7: for $j = 1$ to min $\{ C _ { t } , K \}$ do   
8: Update wb and n for $\mathbf { e } _ { j }$   
9: end for   
10: end for

## III. MAIN RESULTS

## A. Problem A: Information Asymmetry in Actions

We now consider Problem A, where players observe identical rewards but cannot observe one another’s actions. The main challenge is that players must infer which joint item was played in order to correctly update empirical statistics. Otherwise, miscoordination may occur, where different players update different joint items despite observing the same reward.

Because all players observe identical clicks, coordination can be achieved through a predetermined exploration phase. Specifically, during the first $K _ { \operatorname* { m a x } } ~ = ~ K _ { 1 } \cdot \cdot \cdot K _ { M }$ rounds players follow a fixed joint schedule that explores all joint items. This ensures that all players obtain identical statistics for every joint item.

After this phase, players compute UCB indices for each joint item:

$$
\mathrm { U C B } _ { t } ( e ) = \left\{ \begin{array} { l l } { \infty } & { \mathrm { i f ~ } n _ { t - 1 } ( e ) = 0 , } \\ { \widehat { w } _ { n _ { t - 1 } ( e ) } ( e ) + c _ { n _ { t - 1 } ( e ) } } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{3}
$$

Since all players share identical estimates, they select the same joint item at each round by choosing the K items with the largest UCB values. Ties are broken using a fixed lexicographic order on $\mathbb { R } ^ { M }$ from [16].

Definition 1. We say $( x _ { 1 } , \ldots , x _ { M } ) < ( y _ { 1 } , \ldots , y _ { M } )$ if there exists n such that $x _ { i } = y _ { i }$ for all $i < n$ and $x _ { n } < y _ { n }$

Coordination then follows by induction: if players select the same actions up to time t − 1, identical UCB statistics imply they select the same joint action at time t.

Unlike the classical cascade model, we allow multiple clicks and update all observed prefixes up to min $( C _ { t } , K )$ . Because all players maintain identical statistics, no communication is required to maintain coordination. The procedure is given in Algorithm 1.

Theorem 2. If each player uses mCascadeUCB-A under Problem A, the expected T-step regret satisfies

$$
R _ { T } \leq \sum _ { e = K + 1 } ^ { L ^ { M } } \frac { 1 2 } { \Delta _ { e , K } } \log T + \frac { \pi ^ { 2 } } { 3 } L ^ { M } .
$$

This matches the regret order of the single-agent cascading UCB algorithm. The exponential dependence on $L ^ { M }$ is unavoidable for the algorithm as stated: combinatorial cascading bandits over factored action sets [7], [8] obtain polynomial regret only when reward is a known function (e.g. disjunctive) of per-item attractions, whereas we make no parametric assumption tying w(e) to per-player marginals $w ^ { i } ( a ^ { i } ) -$ coordinates may interact arbitrarily, as with complementary product bundles. Combining a known link function with our coordination mechanism to obtain polynomial-in-L regret is a natural extension that we leave to future work.

## B. Problem B: Information Asymmetry in Rewards

We now consider Problem B, where players observe the same actions but receive independent (i.i.d.) click realizations. As a result, players maintain different reward estimates and UCB indices for the same joint item, which may lead to coordination failures. The cascading feedback structure further complicates learning since items receive different numbers of observations across players.

To address this, we use confidence intervals with both upper and lower bounds. For each joint item e, players compute

$$
\begin{array} { r } { \mathrm { U C B } _ { t } ( e ) = \widehat { w } _ { n _ { t - 1 } ( e ) } + c _ { t - 1 , n _ { t - 1 } ( e ) } , } \\ { \mathrm { L C B } _ { t } ( e ) = \widehat { w } _ { n _ { t - 1 } ( e ) } - c _ { t - 1 , n _ { t - 1 } ( e ) } . } \end{array}
$$

Standard concentration inequalities (Lemma 9) imply that the true click probability lies in $( \mathrm { L C B } _ { t } ( e ) , \mathrm { U C B } _ { t } ( e ) )$ with high probability. If $\mathrm { U C B } _ { t } ( e ) < \mathrm { L C B } _ { t } ( e ^ { \prime } )$ , the algorithm can safely eliminate e.

a) mCascadeUCB-Intervals-Ranking.: The algorithm proceeds in K phases, identifying one item per phase. At phase r, the candidate set is $\mathcal { D } = E \backslash$ R, where R contains items selected in earlier phases.

Players cycle through items in D using a shared ordering and repeatedly place the current candidate in the first slot. Because players observe independent clicks, their confidence intervals may differ. Coordination is achieved through implicit signaling: if a player detects that $\mathrm { U C B } _ { t } ( e ) < \mathrm { L C B } _ { t } ( e ^ { \prime } )$ , they sabotage e by deviating from the scheduled marginal action. This observable deviation signals that e should be removed from D.

As intervals shrink, inferior items are eliminated until a single item remains, which is appended to R. Repeating this process for K phases identifies the full ranking.

Example 3. Let $K = 2$ with two players. A joint item $( i , j )$ denotes player 1 choosing i and player 2 choosing j. Suppose the candidate cycle is $( 1 , 1 )  ( 1 , 2 )  ( 2 , 1 )  ( 1 , 1 ) $ · · · . The corresponding placements are

$$
\begin{array} { r l } & { t = 1 \colon ( ( 1 , 1 ) , ( 1 , 2 ) ) , } \\ & { t = 2 \colon ( ( 1 , 2 ) , ( 2 , 1 ) ) , } \\ & { t = 3 \colon ( ( 2 , 1 ) , ( 1 , 1 ) ) , \ldots } \end{array}
$$

If a player detects $\mathrm { U C B } _ { t } ( ( 1 , 2 ) ) < \mathrm { L C B } _ { t } ( ( 1 , 1 ) )$ , they sabotage round t = 2 by playing ((1, 1), (2, 1)) instead of the scheduled $( ( 1 , 2 ) , ( 2 , 1 ) )$ . Other players detect the deviation and remove (1, 2) from the cycle, leaving $( 1 , 1 )  ( 2 , 1 ) $ $( 1 , 1 ) \to \cdots .$

Algorithm 2 mCascadeUCB-Intervals-Ranking-multiple   
Require: M players, item set E, horizon T, target size K   
1: ${ \mathcal { R } } \gets \emptyset ;$ initialize means and counts   
2: for $r = 1$ to $K - | \mathcal { R } |$ do   
3: $\mathcal { D }  E \backslash$ R; set cycle pointer $c \gets 0$   
4: while $| \mathcal { D } | > 1$ do   
5: Place items $\mathcal { D } [ c ] , \mathcal { D } [ c + 1 ] , \ldots , \mathcal { D } [ c + K - 1 ] \left( \bmod \right) \mathcal { D } | )$   
in slots $1 , \ldots , K$   
6: Compute $\mathrm { U C B } _ { t } , \mathrm { L C B } _ { t }$ for $e \in \mathcal { D }$   
7: if $\exists e \in \mathcal { D }$ with $\mathrm { U C B } _ { t } ( e ) ~ < ~ \mathrm { m a x } _ { e ^ { \prime } \in \mathcal D } \mathrm { L C B } _ { t } ( e ^ { \prime } )$   
then   
8: Sabotage scheduled placement by deviating; signal   
elimination of e   
9: end if   
10: Observe cascade; for each slot $k \_ \leq$ min $( C _ { t } , K )$   
update estimate of the item placed there   
11: Remove dominated items from $\mathcal { D } ;$ advance $c \gets c + 1$   
12: end while   
13: Append remaining item to R   
14: end for   
15: Recommend R ordered by empirical means

This procedure achieves gap-dependent regret ${ \cal O } ( \log T )$ Note that obtaining sublinear regret requires recovering not only the top-K set but also their correct ordering. Sabotage signaling is unambiguous because the round-robin schedule is deterministic and common knowledge: every player can compute the scheduled joint action from $( t , \mathcal { R } , \mathcal { D } )$ alone, so any deviation is a signal rather than noise. The only remaining failure mode—a player eliminating an item whose true attraction is higher than the eliminator’s—is controlled by the confidence radius of Lemma 9 and contributes only an $O ( L ^ { M } / T )$ additive term to the regret, handled in the proof of Theorem 4.

Theorem 4. Let $e _ { 1 } ^ { \star } , \ldots , e _ { K } ^ { \star }$ denote the top-K joint items (in decreasing order of attraction probability), and for each $r \in \{ 1 , \ldots , K \}$ let $\mu _ { r } ^ { \star } = w ( e _ { r } ^ { \star } )$ and $\Delta _ { e _ { r } ^ { \star } , e } = \mu _ { r } ^ { \star } - w ( e )$ for any suboptimal e in the r-th elimination phase. The regret of mCascadeUCB-Intervals-Ranking under Problem B satisfies

$$
R _ { T } \leq \sum _ { r = 1 } ^ { K } \sum _ { e : w ( e ) < \mu _ { r } ^ { \star } } \frac { ( 4 + 4 \sqrt { 2 } ) ^ { 2 } \log T } { \Delta _ { e _ { r } ^ { \star } , e } ^ { 2 } } .
$$

b) Round-Robin Ranking with Multiple Placements.: The previous algorithm only uses feedback from the first slot. When termination probabilities are small, additional slots can provide useful observations. The algorithm mCascadeUCB-Intervals-Ranking-multiple (Algorithm 2) exploits this by cycling through candidates and placing K items each round.

If the upper confidence bound of an item falls below another item’s lower confidence bound, a player sabotages the scheduled placement, signaling elimination. An item e placed in slot k is observed if the cascade reaches that slot. The observation probability for slot k is

$$
p _ { k } ( e ) = \prod _ { j = 1 } ^ { k - 1 } \left[ ( 1 - w ( e _ { j } ) ) + w ( e _ { j } ) v _ { j } \right] .
$$

Define the minimum slot-survival probability

$$
p _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { 1 \leq k \leq K } \operatorname* { m i n } _ { ( e _ { 1 } , \ldots , e _ { k - 1 } ) } \prod _ { j = 1 } ^ { k - 1 } \Big [ ( 1 - w ( e _ { j } ) ) + w ( e _ { j } ) v _ { j } \Big ] .
$$

Lemma 5. If a joint item e is placed in slot $j > 1 \ a$ total of M times, then with probability at least $1 - { \textstyle { \frac { 1 } { T } } }$ it receives at least $\left( 1 - { \sqrt { \frac { \log T } { 2 p _ { \operatorname* { m i n } } ^ { 2 } M } } } \right) M p _ { \operatorname* { m i n } }$ observations. In particular, the bound is non-vacuous whenever

$$
M \geq \frac { \log T } { 2 p _ { \operatorname* { m i n } } ^ { 2 } } ,
$$

which is the regime in which the multi-placement strategy yields strictly more information than first-slot-only exploration.

Theorem 6. The regret of mCascadeUCB-Intervals-Ranking-multiple satisfies

$$
R _ { T } = O \left( \sum _ { e } L ^ { M } \left( \frac { B + \sqrt { B ^ { 2 } + 4 A \cdot R _ { e } } } { 2 A } \right) ^ { 2 } \right) ,
$$

where $A = 1 + ( K - 1 ) p _ { \operatorname* { m i n } } , \ B = ( K - 1 ) \sqrt { \frac { \log T } { 2 } } ,$ and $\begin{array} { r } { R _ { e } = \frac { 8 \log T } { \Delta _ { e } ^ { 2 } } + 1 } \end{array}$

When $p _ { \mathrm { m i n } } \approx 1 \ ( \mathrm { i . e }$ ., termination probabilities are small), the bound approaches the classical cascading-bandit regret $O ( \log T / \Delta ^ { 2 } )$ [1], [3].

## C. Problem C: Information Asymmetry in Both

We now consider Problem C, which combines the difficulties of Problems A and B: players cannot observe one another’s actions and they receive asymmetric (i.i.d.) click feedback. As a result, mCascadeUCB-Intervals from Problem B cannot coordinate eliminations without action observability, and the approach for Problem A fails because players no longer share identical reward statistics.

To handle both asymmetries, we adapt the MDSEE framework [16] to the multi-agent cascade setting. Our goal is to identify the top-K joint items using a phased explore-thencommit strategy. The resulting algorithm, mMDSEE-TopK, alternates between exploration and exploitation phases according to a schedule $F ( \lambda )$ (default $F ( \lambda ) = \lambda )$ ), where λ is the phase index.

In the exploration phase, the algorithm spends $L \cdot F ( \lambda )$ rounds uniformly sampling items via a fixed round-robin schedule: at each round t it recommends a K-list so that each item appears in each slot sufficiently often. The user returns a click position $C _ { t } \in \{ 1 , \ldots , K , \infty \}$ . For each observed position $k \leq C _ { t }$ , the algorithm updates $n _ { t } ( e )$ and $\widehat { w } _ { t } ( e )$ for the item in slot $a _ { t } ^ { k }$

Algorithm 3 mMDSEE-TopK   
Require: Item set E of size $L ,$ target size $K ,$ horizon $\overline { { T } } ,$   
schedule $F ( \lambda ) = \lambda$   
Ensure: Top-K items identification   
1: Initialize $\hat { w } _ { 0 } ( e ) \gets 0 , n _ { 0 } ( e ) \gets 0$ for all $e \in E$   
2: $\lambda  1 ; \quad t  1$   
3: while $t < T$ do   
4: Exploration: For $F ( \lambda ) \cdot L$ rounds, cycle through items   
with a sliding window of size K   
5: for $j = 1$ to L do   
6: Recommend $[ e _ { j } , e _ { ( j + 1 ) }$ mod $\iota , \cdot \cdot \cdot , e _ { ( j + K - 1 ) }$ <sub>mod</sub> <sub>L</sub> ]   
7: Observe click $C _ { t }$ and update estimates   
8: $t \gets t + 1$   
9: end for   
10: Exploitation: Let $\mathcal { R } _ { t }$ be the top-K items by $\hat { w } _ { t } ( e )$   
11: $T _ { \mathrm { e x p l o i t } }  2 ^ { r } - t$ (with $2 ^ { r } > t )$   
12: for $j = 1$ to $T _ { \mathrm { e x p l o i t } }$ do   
13: Recommend $\mathcal { R } _ { t } ,$ observe feedback   
14: $t \gets t + 1$   
15: end for   
16: $\lambda  \lambda + 1$   
17: end while

After exploration, the exploitation phase forms a ranking $\mathcal { R } _ { t }$ of the K items with largest empirical means $\widehat { w } _ { t } ( e )$ and repeatedly recommends $\mathcal { R } _ { t }$ until the next power-of-two boundary $2 ^ { \tau }$ (smallest τ with $2 ^ { \tau } > t )$ . The phase index λ is then incremented. We only update estimates using exploration rounds: under asymmetric rewards and unobservable joint actions, feedback during exploitation cannot be reliably attributed to specific joint items.

The algorithm is given in Algorithm 3.

a) Regret with First-Slot Feedback.: We first analyze the special case where players observe only whether the first slot terminates.

Theorem 7 (First-Slot Feedback). Consider Algorithm 3. Suppose that $\begin{array} { r } { \varepsilon < \frac 1 2 \operatorname* { m i n } _ { 1 \leq i \leq K } \operatorname* { m i n } _ { e } \Delta _ { i + 1 , i } , } \end{array}$ and let $F ( \lambda ) = \lambda$ Then the cumulative regret after T rounds satisfies

$$
\begin{array} { l } { \displaystyle { R _ { T } = O \Big ( L ^ { M } \log \log T \log T } } \\ { \displaystyle { + \frac { M L ^ { M } \sqrt \pi } { \sqrt { 2 c } \varepsilon } e ^ { \frac 1 { 8 c \varepsilon ^ { 2 } } } \left[ 1 + \mathrm { e r f } \left( \frac 1 { \sqrt { 8 c } \varepsilon } \right) \right] \Big ) . } } \end{array}
$$

Here, $\Delta _ { i + 1 , i }$ denotes the gap between the $( i + 1 )$ -th and i-th best joint items. Choosing $\begin{array} { r } { \varepsilon = \frac 1 2 \operatorname* { m i n } _ { 1 \leq i \leq K } \Delta _ { i + 1 , i } } \end{array}$ is sufficient to separate optimal from suboptimal items and to recover the correct ordering within the top-K.

The first term reflects exploration across phases; since phase lengths grow exponentially, the effective sample size increases rapidly and error probabilities decay quickly. The second term accounts for mis-commitment: the probability that players commit to an incorrect joint item due to estimation error. We use $F ( \lambda ) = \lambda$ to obtain sharper asymptotic bounds than [16].

b) Regret with Multiple-Slot Feedback.: We next consider the case where players use feedback from all observed slots up to K. The algorithm is unchanged; only the exploration updates incorporate additional observed joint items, improving estimation without altering coordination.

Theorem 8 (Multi-Slot Feedback). Under the same setting and notation as Theorem 7, let $c ~ = ~ 1 + ( K - 1 ) ( 1 ~ -$ $\frac { \log T } { 2 p _ { \operatorname* { m i n } } ^ { 2 } F ( T ) } ) p _ { \operatorname* { m i n } } ,$ , with $\alpha = 1 + ( K - 1 ) p _ { \mathrm { m i n } } . \ I f \ F ( \lambda ) = \lambda ,$

$$
\begin{array} { l } { { \displaystyle R _ { T } = O \bigg ( \log T \log \log T } } \\ { { \displaystyle \qquad + \frac { M L ^ { M } \sqrt \pi } { \sqrt { 2 c } \varepsilon \sqrt { \alpha } } e ^ { \frac { 1 } { 8 c \alpha \varepsilon ^ { 2 } } } \left[ 1 + \mathrm { e r f } \left( \frac { 1 } { \sqrt { 8 c \alpha } \varepsilon } \right) \right] \bigg ) . } } \end{array}
$$

The factor α captures the benefit of multi-slot feedback: observing more slots increases the effective number of observations per round by $\alpha = 1 + ( K - 1 ) p _ { \operatorname* { m i n } } .$ , tightening concentration and reducing regret. When $p _ { \mathrm { m i n } } \ = \ 0 .$ , no additional slots are observed and we recover Theorem 7 with $c = 1$ . As $p _ { \operatorname* { m i n } }  1$ , we obtain $c \approx K$ , corresponding to nearly K effective observations per round.

## IV. EXPERIMENTS

We evaluate our algorithms under the DCM cascade simulator: at slot k with item $e ,$ the user clicks with probability $w ( e ) ;$ ; if a click occurs, the session terminates with probability $v _ { k }$ . The terminating click position $C _ { t } \ ( \mathrm { o r \propto } )$ is the observed feedback. Reward is $\mathbb { I } \{ C _ { t } ~ \neq ~ \infty \}$ and instantaneous regret is $\begin{array} { r } { \left[ 1 - \prod _ { k } ( 1 - v _ { k } w _ { k } ^ { \star } ) \right] - \left[ 1 - \prod _ { k } ( 1 - v _ { k } w ( e _ { k } ^ { t } ) ) \right] } \end{array}$ , where $w _ { 1 } ^ { \star } , \ldots , w _ { K } ^ { \star }$ are the top-K attractions paired with the largest $v _ { k }$ values (optimal by the rearrangement inequality).

a) Baselines.: To verify that joint-arm UCB is not artificially inflating measured regret, we compare against an independent per-player UCB baseline in which each player runs a single-agent UCB over its local item set $E ^ { i } .$ , ignoring joint structure: player i at slot k updates the marginal estimate $\hat { w } ^ { i } ( a ^ { i } )$ from the joint feedback. We do not include a separate centralized oracle because under Problem A the decentralized mCascadeUCB-A already maintains identical statistics to a centralized UCB on joint arms; mCascadeUCB-A therefore serves simultaneously as our algorithm and as the centralized benchmark.

b) Setup.: Attraction probabilities are drawn $w ( e ) \sim$ Uniform[0.1, 0.9] and fixed across seeds. We average over 5 random seeds; shaded regions show ±2 standard deviations. The instance is $L = 3 , K = 2 , M = 3 \ : ( 2 7$ joint arms) with horizon $T = 5 \times 1 0 ^ { 4 }$ . We sweep two termination regimes: low $( v \in [ 0 . 1 5 , 0 . 2 5 ] )$ and high $( v \in [ 0 . 8 5 , 0 . 9 5 ] )$ ).

## A. Cumulative Regret

Figures 1 and 2 report cumulative regret.

Synthetic, low termination (v [0.15, 0.25]), L = 3, K= 2, M = 3  
![](images/9d44feb201c524439f758d1a3f13cd754b26572f9cf1bb4b5ff06a0487c56c81.jpg)  
Fig. 1. Cumulative regret under low termination probability $v \in [ 0 . 1 5 , 0 . 2 5 ]$ Coordinated methods cluster tightly; mMDSEE-TopK with full feedback benefits from observations in deeper slots.

Synthetic, high termination (v [0.85, 0.95]), L = 3, K= 2, M = 3  
![](images/257aecea0f7f745a39c4c8c88002c5e6cf24a6327c907e5a644c4b227c4f09ff.jpg)  
Fig. 2. Cumulative regret under high termination probability $v \in [ 0 . 8 5 , 0 . 9 5 ]$ mCascadeUCB-A is best; first-slot mMDSEE-TopK outperforms full-slot here because deeper-slot observations are rare. Independent per-player UCB plateaus at ∼ 6× the regret of coordinated methods.

a) $H i g h$ termination $( v ~ \in ~ [ 0 . 8 5 , 0 . 9 5 ] ) .$ : When users frequently terminate after the first click (Fig. 2), most click feedback comes from slot 1. mCascadeUCB-A is the most data-efficient $( \approx ~ 5 8 0$ at $T { = } 5 \times 1 0 ^ { 4 } )$ . The first-slot variant of mMDSEE-TopK (≈ 810) actually outperforms its fullslot counterpart (≈ 1300): when later-slot observations are rare and noisy, restricting updates to slot 1 reduces variance. Independent per-player UCB plateaus around 3,500, an order of magnitude worse.

b) Low termination $( v \in [ 0 . 1 5 , 0 . 2 5 ] ) .$ : When cascades typically reach both slots (Fig. 1), full-feedback algorithms exploit every observation. mMDSEE-TopK with full feedback edges out its first-slot counterpart, consistent with the $\alpha =$ $1 + ( K - 1 ) p _ { \operatorname* { m i n } }$ scaling predicted by Theorem $\mathbf { \boldsymbol { 8 } ; }$ the gap is modest at $K = 2$ but is expected to widen as K grows. Independent per-player UCB again pays a multiplicative penalty for ignoring joint structure.

## B. Discussion

Three insights emerge:

Multi-slot feedback matters when $p _ { \mathrm { m i n } }$ is large. The $\alpha =$ $1 + ( K - 1 ) p _ { \operatorname* { m i n } }$ factor predicted by Theorem 8 matches the observed gap between full- and first-slot variants at low termination.

First-slot feedback can win at high termination. When most clicks terminate at slot 1, full-slot updates inject noise rather than information. This is consistent with our analysis: the $( K -$ $1 ) p _ { \mathrm { m i n } }$ improvement collapses to zero as $p _ { \operatorname* { m i n } }  0$

Coordination beats independence even with asymmetric rewards. Independent UCB consistently lags coordinated methods across both termination regimes, showing that the multiplayer asymmetric setting is genuinely harder than M parallel single-agent problems and that joint-arm coordination—even without communication, as in mMDSEE-TopK—provides real value.

a) Computational cost.: All algorithms have per-round cost dominated by the top-K selection over $L ^ { M }$ joint arms, i.e. ${ \cal O } ( L ^ { M } \log { L ^ { \dot { M } } } )$ . mMDSEE-TopK requires no inter-player communication; mCascadeUCB-Intervals-Ranking requires action observability but no protocol messages. A single $T { = } 5 \times 1 0 ^ { 4 }$ run completes in a few seconds on a standard laptop.

## V. RELATED WORK

a) Click-model bandits.: The cascade model [1] and its many extensions—combinatorial actions [7], contextual settings [5], Thompson sampling [6], scalability improvements [4], and clustering-based methods [20]—form the algorithmic foundation for learning to rank under bandit feedback. The dependent click model [2], [3] extends single-click cascades to multi-click sessions and is the closest single-agent precursor to our work; recent variants incorporate fatigue [9], exposure bias [10], RL-style ranking [21], and adversarial robustness [11]. All of these assume a single decision-maker.

b) Decentralized and multiplayer bandits.: A parallel line of work studies multiple agents learning simultaneously under communication or observation constraints [14], [15], [22]–[26]. Most relevant here are decentralized learning with information asymmetry and no communication [16]–[19] and federated/combinatorial multi-agent formulations [8], which provide the techniques we extend to multi-click feedback. Real-world recommender deployments [12], [13], [27]–[29] and bandit models with structured feedback [30], [31] further motivate the decentralized cascading setting. To our knowledge no prior work has combined multi-click cascade feedback with multi-agent decentralized learning under information asymmetry.

## VI. CONCLUSION

We study decentralized multi-click cascading bandits with private observations and information asymmetry. Building on the Dependent Click Model and multiplayer bandit theory, we design algorithms that handle action asymmetry, reward asymmetry, and their combination. Our methods extend UCBbased approaches and MDSEE-style exploration to the multiagent cascade setting, achieving sublinear regret under both first-slot and multi-slot feedback.

Our theoretical analysis captures the role of multiple clicks and slot termination probabilities. Empirical results confirm that leveraging multi-slot feedback significantly improves performance when termination probabilities are small, and that coordinated decentralized learning consistently outperforms independent per-player baselines.

a) Open problems.: Two questions remain open. First, we prove no matching information-theoretic lower bound for any of Problems $\mathrm { \mathbf { A } \mathrm { - } \mathbf { C } ; }$ establishing such bounds (or identifying a regime where our upper bounds are tight) is the natural next step. Second, the worst-case $L ^ { M }$ dependence reflects the absence of factored structure in our model; identifying mild parametric assumptions (e.g. generalized linear link functions over player features) under which polynomial-in-L regret is attainable would substantially improve scalability.

## ACKNOWLEDGMENT

The authors would like to thank the reviewers for their valuable feedback.

## REFERENCES

[1] B. Kveton, C. Szepesvari, Z. Wen, and A. Ashkan, “Cascading bandits: Learning to rank in the cascade model,” in Proceedings of the 32nd International Conference on Machine Learning, vol. 37 of Proceedings of Machine Learning Research, pp. 767–776, PMLR, 2015.

[2] F. Guo, C. Liu, and Y. M. Wang, “Efficient multiple-click models in web search,” in Proceedings of the Second ACM International Conference on Web Search and Data Mining, pp. 124–131, 2009.

[3] S. Katariya, B. Kveton, C. Szepesvari, and Z. Wen, “Dcm bandits:´ Learning to rank with multiple clicks,” in Proceedings of the 33rd International Conference on Machine Learning (ICML), pp. 1215–1224, PMLR, 2016.

[4] S. Zong, H. Ni, K. Sung, N. R. Ke, Z. Wen, and B. Kveton, “Cascading bandits for large-scale recommendation problems,” 2016.

[5] S. Li, B. Wang, S. Zhang, and W. Chen, “Contextual combinatorial cascading bandits,” in Proceedings of the 33rd International Conference on Machine Learning (ICML), pp. 1245–1253, PMLR, 2016.

[6] Z. Zhong, W. C. Cheung, and V. Y. F. Tan, “Thompson sampling algorithms for cascading bandits,” Journal of Machine Learning Research, vol. 22, no. 218, pp. 1–66, 2021.

[7] B. Kveton, Z. Wen, A. Ashkan, and C. Szepesvari, “Combinatorial cascading bandits,” Advances in Neural Information Processing Systems, vol. 28, 2015.

[8] F. Fourati, M.-S. Alouini, and V. Aggarwal, “Federated combinatorial multi-agent multi-armed bandits,” 2024.

[9] J. Cao, W. Sun, Z.-J. M. Shen, and M. Ettl, “Fatigue-aware bandits for dependent click models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, pp. 3341–3348, 2020.

[10] M. Mansoury, B. Mobasher, and H. van Hoof, “Mitigating exposure bias in online learning to rank recommendation: A novel reward model for cascading bandits,” in Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, pp. 1638– 1648, 2024.

[11] J. Xie, C. Chen, Z. Wang, and S. Li, “Cascading bandits robust to adversarial corruptions,” 2025.

[12] K. Zou and A. Sun, “A survey of real-world recommender systems: Challenges, constraints, and industrial perspectives,” 2025.

[13] Y. Wang, L. Tao, and X. X. Zhang, “Recommending for a multisided marketplace: A multi-objective hierarchical approach,” Marketing Science, vol. 44, no. 1, pp. 1–29, 2025.

[14] E. Boursier and V. Perchet, “A survey on multiplayer bandits,” Journal of Machine Learning Research, vol. 23, no. 1, pp. 1–47, 2022.

[15] S. Branzei and Y. Peres, “Multiplayer bandit learning, from competition to cooperation,” in Proceedings of the 34th Conference on Learning Theory, vol. 134 of PMLR, pp. 679–723, 2021.

[16] W. Chang, M. Jafarnia-Jahromi, and R. Jain, “Online learning for cooperative multi-player multi-armed bandits,” in 2022 IEEE 61st Conference on Decision and Control (CDC), pp. 7248–7253, IEEE, 2022.

[17] W. Chang and Y. Lu, “Optimal cooperative multiplayer learning bandits with noisy rewards and no communication,” arXiv preprint arXiv:2311.06210, 2023.

[18] W. Chang and Y. Lu, “Multiplayer information asymmetric contextual bandits,” 2025.

[19] W. Chang and A. Karthik, “Multiplayer information asymmetric bandits in metric spaces,” 2025.

[20] S. Li and S. Zhang, “Online clustering of contextual cascading bandits,” in Proceedings of the 32nd AAAI Conference on Artificial Intelligence, 2018.

[21] Y. Du, R. Srikant, and W. Chen, “Cascading reinforcement learning,” in Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024.

[22] B. Awerbuch and R. D. Kleinberg, “Competitive collaborative learning,” in Proceedings of the 18th Annual Conference on Learning Theory (COLT), vol. 3559 of Lecture Notes in Computer Science, pp. 233–248, Springer, 2005.

[23] P. Landgren, V. Srivastava, and N. E. Leonard, “Distributed cooperative decision-making in multiarmed bandits: Frequentist and bayesian algorithms,” in Proceedings of the 55th IEEE Conference on Decision and Control (CDC), pp. 167–172, IEEE, 2016.

[24] N. Cesa-Bianchi, C. Gentile, Y. Mansour, and A. Minora, “Delay and cooperation in nonstochastic bandits,” in Proceedings of the 29th Annual Conference on Learning Theory, vol. 49 of Proceedings of Machine Learning Research, pp. 605–622, PMLR, 2016.

[25] P. Alatur, K. Y. Levy, and A. Krause, “Multi-player bandits: The adversarial case,” Journal of Machine Learning Research, vol. 21, pp. 1– 23, 2020.

[26] G. Lugosi and A. Mehrabian, “Multiplayer bandits without observing collision information,” Mathematics of Operations Research, 2021.

[27] B. Loepp, “Multi-list interfaces for recommender systems: survey and future directions,” Frontiers in big Data, vol. 6, p. 1239705, 2023.

[28] A. Fletcher, P. L. Ormosi, and R. Savani, “Recommender systems and supplier competition on platforms,” Journal of Competition Law & Economics, vol. 19, no. 3, pp. 397–426, 2023.

[29] R. Demsyn-Jones, “Measurement and applications of position bias in a marketplace search engine,” 2022.

[30] N. Alon, N. Cesa-Bianchi, C. Gentile, S. Mannor, Y. Mansour, and O. Shamir, “Nonstochastic multi-armed bandits with graph-structured feedback,” Theoretical Computer Science, 2014.

[31] J. Mo and H. Xie, “A multi-player mab approach for distributed selection problems,” in Advances in Knowledge Discovery and Data Mining (PAKDD), Springer, 2023.

## APPENDIX

Lemma 9. Assume that $X _ { i } ~ - ~ \mu$ are independent, $\sigma \cdot$ subgaussian random variables. Then for any $\varepsilon \geq 0$

$$
\mathbb { P } ( \hat { \mu } \geq \mu + \varepsilon ) \leq \exp \left( - \frac { n \varepsilon ^ { 2 } } { 2 \sigma ^ { 2 } } \right) , \quad \mathbb { P } ( \hat { \mu } \leq \mu - \varepsilon ) \leq \exp \left( - \frac { n \varepsilon ^ { 2 } } { 2 \sigma ^ { 2 } } \right)
$$

where $\begin{array} { r } { \hat { \mu } = \frac { 1 } { n } \sum _ { t = 1 } ^ { n } X _ { t } . } \end{array}$

Proof. Suppose item e was placed in slot k for the i-th time, and let $p _ { i } ^ { k }$ be the probability that this item was observed. Let $X _ { i }$ be the Bernoulli random variable for the clicks for the i-th placement. Then $X _ { i } - p _ { i } ^ { k }$ is a sequence of mean-0 independent 1 -subgaussian random variables, so by Lemma 9 we have

$$
\mathrm { P r } \biggl [ \frac { \sum _ { i } X _ { i } } { M } < \frac { \sum _ { i } p _ { i } ^ { k } } { M } - \epsilon \biggr ] \leq \exp \bigl ( - 2 M \epsilon ^ { 2 } \bigr ) .
$$

Setting $\begin{array} { r } { \epsilon = \frac { \alpha \sum _ { i } { p _ { i } ^ { k } } } { M } } \end{array}$ for some small $\alpha > 0$ , we conclude that

$$
\operatorname* { P r } \left[ \frac { \sum _ { i } X _ { i } } { M } < ( 1 - \alpha ) \frac { \sum _ { i } p _ { i } ^ { k } } { M } \right] \leq \exp \left( - 2 \alpha ^ { 2 } p _ { \operatorname* { m i n } } ^ { 2 } M \right) .
$$

Setting the right-hand side to be upper bounded by $1 / T \colon$

$$
\exp \bigl ( - 2 \alpha ^ { 2 } p _ { \mathrm { m i n } } ^ { 2 } M \bigr ) = \frac { 1 } { T } \quad \Longrightarrow \quad \alpha = \sqrt { \frac { \log T } { 2 p _ { \mathrm { m i n } } ^ { 2 } M } } .
$$

Therefore, with probability at least $\textstyle 1 - { \frac { 1 } { T } }$

#{observations of e in slot $k \} > \left( 1 - { \sqrt { \frac { \log T } { 2 p _ { \operatorname* { m i n } } ^ { 2 } M } } } \right) M p _ { \operatorname* { m i n } } .$

Proof. A necessary condition for item e to still be in the candidate set is that $\mathrm { U C B } ( e ) > \mathrm { L C B } ( e ^ { \star } )$ . This implies

$$
\hat { \mu } _ { e } + \sqrt { \frac { 4 \log T } { n _ { e } ( t ) } } > \hat { \mu } _ { e ^ { \star } } - \sqrt { \frac { 4 \log T } { n _ { e ^ { \star } } ( t ) } } ,\tag{4}
$$

$$
\Longrightarrow ~ \mu _ { e } + 2 { \sqrt { \frac { 4 \log T } { n _ { e } ( t ) } } } > \mu ^ { * } - 2 { \sqrt { \frac { 4 \log T } { n _ { e } ( t ) } } } ~ ( { \mathrm { g o o d ~ e v e n t } } ) ,\tag{5}
$$

$$
\implies 2 \sqrt { \frac { 4 \log T } { n _ { e } ( t ) } } > \Delta _ { e ^ { \star } , e } - 2 \sqrt { \frac { 4 \log T } { n _ { e ^ { \star } } ( t ) } } .\tag{6}
$$

Since items in the candidate set are pulled in a round-robin fashion, we have $| n _ { e } ( t ) - n _ { e ^ { \star } } ( t ) | \leq 1$ . This implies that when $n _ { e } ( t ) \geq 2$ , we also have $n _ { e ^ { \star } } ( t ) \geq n _ { e } ( t ) / 2$ . Plugging this in:

$$
2 \sqrt { \frac { 4 \log T } { n _ { e } ( t ) } } > \Delta _ { e ^ { \star } , e } - 2 \sqrt { \frac { 4 \log T } { n _ { e } ( t ) / 2 } } ,\tag{7}
$$

$$
\sqrt { \frac { \log T } { n _ { e } ( t ) } } \left( 4 + 4 \sqrt { 2 } \right) > \Delta _ { e ^ { \star } , e } ,\tag{8}
$$

$$
\implies n _ { e } ( t ) < \frac { ( 4 + 4 \sqrt { 2 } ) ^ { 2 } \log T } { \Delta _ { e ^ { \star } , e } ^ { 2 } } .\tag{9}
$$

Now suppose the optimal joint items are $e _ { 1 } ^ { \star } , \ldots , e _ { K } ^ { \star }$ , where $e _ { r } ^ { \star }$ is the item selected in the r-th elimination phase. The total number of pulls across all suboptimal items is bounded as

$$
\tau < \sum _ { r = 1 } ^ { K } \sum _ { e : w ( e ) < \mu _ { r } ^ { \star } } \frac { ( 4 + 4 \sqrt { 2 } ) ^ { 2 } \log T } { \Delta _ { e _ { r } ^ { \star } , e } ^ { 2 } } .
$$

<sup>,</sup>Under the good event, the regret from $t = \tau \mathrm { ~ t o ~ } T$ is zero, since only top items are pulled from that point onward.

Proof. By the reasoning similar to Theorem 4, item e needs only $\begin{array} { r } { N _ { \mathrm { e l i m } } = \frac { 8 \log T } { \Delta _ { r } ^ { 2 } } + \breve { 1 } } \end{array}$ observations before it is eliminated (the additional +1 accounts for the sabotage round). Since the first slot always gets observed, the total observations satisfy

$$
\begin{array} { r l r } {  { Y _ { \mathrm { t o t a l } } = M + \sum _ { r = 2 } ^ { K } \# \{ \mathrm { o b s ~ i n ~ s l o t ~ } r \} } } \\ & { } & { > M + ( K - 1 ) \Big ( 1 - \sqrt { \frac { \log T } { 2 p _ { \operatorname* { m i n } } ^ { 2 } M } } \Big ) M p _ { \operatorname* { m i n } } . } \end{array}
$$

We require $Y _ { \mathrm { t o t a l } } \ge N _ { \mathrm { e l i m } }$ . Let $A : = 1 + ( K - 1 ) p _ { \operatorname* { m i n } } ,$ $B : = ( K - 1 ) \sqrt { \textstyle { \frac { \log T } { 2 } } }$ , and $\begin{array} { r } { R : = \frac { 8 \log T } { \Delta _ { K } ^ { 2 } } + 1 } \end{array}$ . Then $A M -$ $B \sqrt { M } - R \geq 0$ . Setting $x : = \sqrt { M } \geq \overset { } { 0 }$ yields the quadratic $A x ^ { 2 } - B x - R \geq 0 .$ , solved by

$$
x \geq \frac { B + \sqrt { B ^ { 2 } + 4 A R } } { 2 A } .
$$

Squaring gives $\begin{array} { r } { M \ge { { \left( \frac { { B + \sqrt { { B ^ { 2 } } + 4 A R } } } { { 2 { A } } } \right) } ^ { 2 } } } \end{array}$

The instantaneous regret satisfies $\begin{array} { r } { R _ { t } \ \le \ \sum _ { k = 1 } ^ { K } [ w ( e _ { k } ^ { * } ) \ - \ } \end{array}$ $w ( e _ { k } ^ { t } ) ]$ , so the total regret is $\begin{array} { r c l } { { R _ { T } } } & { { \le } } & { { \sum _ { e \mathrm { s u b o p t i m a l } } \Delta _ { K } ~ . } } \end{array}$ E[#placements of e], yielding the stated bound. 厂

Proof. Split the total regret into exploration and commitment phases: $R _ { T } = R _ { T , E } + R _ { T , C } .$

Exploration term. Let $\nu _ { t } ( e )$ be the number of times item e has been placed first up to time t. Since exploration occurs at $t \stackrel { . } { = } 2 ^ { 0 } , 2 ^ { 1 } , \ldots , \stackrel { . } { 2 } ^ { \lfloor \log _ { 2 } T \rfloor }$ , we have $\nu _ { t } ( e ) \ \leq$ $F ( \lfloor \log _ { 2 } T \rfloor ) \lceil \log _ { 2 } T \rceil$ . Each pull incurs at most L regret, so

$$
R _ { T , E } \leq L ^ { M } F ( \lfloor \log _ { 2 } T \rfloor ) \lceil \log _ { 2 } T \rceil .
$$

Commitment term. Let $\varepsilon ~ < ~ { \frac { 1 } { 2 } }$ min $1 \leq i \leq K$ min $\ L _ { \rho } \Delta _ { i + 1 , i }$ Define the good event for player i at time $t \colon G _ { t } ^ { i } ( e ) \ =$ $\{ | \widehat { w } _ { n _ { t } ( e ) } ^ { i } ( e ) - w ( e ) | < \varepsilon \}$ . If $\begin{array} { r } { G _ { t } = \bigcap _ { i } G _ { t } ^ { i } ( e ) } \end{array}$ holds, all players choose the optimal ranking, so $R _ { t } = 0$ . Hence

$$
\begin{array} { l } { \displaystyle R _ { T , C } = \sum _ { t = 1 } ^ { T } [ \mathbb { E } [ R _ { t } \mid G _ { t } ] P ( G _ { t } ) + \mathbb { E } [ R _ { t } \mid G _ { t } ^ { c } ] P ( G _ { t } ^ { c } ) ] } \\ { \displaystyle \leq \sum _ { t = 1 } ^ { T } P ( G _ { t } ^ { c } ) . } \end{array}
$$

By De Morgan and a union bound, $\begin{array} { r l } { R _ { T , C } } & { { } \leq } \end{array}$ $\begin{array} { r } { M \sum _ { t = 1 } ^ { T } \sum _ { e \in E } P ( | \widehat { w } _ { n _ { t } ( e ) } ( e ) ~ - ~ w ( e ) | ~ \geq ~ \varepsilon ) } \end{array}$ . Hoeffding’s inequality gives $\begin{array} { r } { P ( | \widehat { w } _ { n _ { t } ( e ) } ( e ) ~ - ~ w ( e ) | \quad \geq ~ \varepsilon ) \quad \leq } \end{array}$ $2 \exp ( - 2 n _ { t } ( e ) \varepsilon ^ { 2 } )$

Since $F ( \lambda ) \ = \ \lambda$ , the total number of explorations up to phase $\lambda ~ = ~ \log _ { 2 } t$ is $n _ { t } ( e ) = 1 + 2 + \cdot \cdot \cdot + F ( \lambda )$ = $\Omega ( ( \log _ { 2 } t ) ^ { 2 } / 2 )$ . Therefore $F _ { 0 } ( t ) : = n _ { t } ( e ) / \ln t \geq c \ln t$ for a constant $c \ > \ 0 .$ . The exponent becomes $t ^ { - 2 F _ { 0 } ( t ) \varepsilon ^ { 2 } } \leq$ $t ^ { - 2 c \varepsilon ^ { 2 } \ln t } = \exp ( - 2 c \varepsilon ^ { 2 } ( \ln t ) ^ { 2 } )$

Define the comparison integral $\begin{array} { r l } { I ( c , \varepsilon ) } & { { } = } \end{array}$ $\textstyle \int _ { 1 } ^ { \infty } e ^ { - 2 c \varepsilon ^ { 2 } ( \ln x ) ^ { 2 } }$ dx. Substituting u = ln x and completing the square:

$$
\begin{array} { l } { I ( c , \varepsilon ) = e ^ { \frac { 1 } { 8 c \varepsilon ^ { 2 } } } \int _ { - \frac { 1 } { 4 c \varepsilon ^ { 2 } } } ^ { \infty } e ^ { - 2 c \varepsilon ^ { 2 } v ^ { 2 } } d v } \\ { \quad \quad = \frac { \sqrt { \pi } } { 2 \sqrt { 2 c } \varepsilon } e ^ { \frac { 1 } { 8 c \varepsilon ^ { 2 } } } \left[ 1 + \mathrm { e r f } \left( \frac { 1 } { \sqrt { 8 c } \varepsilon } \right) \right] . } \end{array}
$$

Since the summand is positive and decreasing, $\begin{array} { r } { \sum _ { t = 1 } ^ { \infty } \exp ( - 2 c \varepsilon ^ { 2 } ( \ln t ) ^ { 2 } ) \le 1 + \bar { I ( c , \varepsilon ) } } \end{array}$ , hence

$$
\begin{array} { r } { R _ { T , C } \lesssim \displaystyle \frac { M L ^ { M } \sqrt { \pi } } { \sqrt { 2 c } \varepsilon } \exp \left( \frac { 1 } { 8 c \varepsilon ^ { 2 } } \right) } \\ { \times \left[ 1 + \mathrm { e r f } \left( \displaystyle \frac { 1 } { \sqrt { 8 c } \varepsilon } \right) \right] . } \end{array}
$$

This completes the proof.

Proof. We follow the proof of Theorem 7 with minor modifications. Under multiple-slot feedback, each time e is placed we obtain additional observations from deeper slots. The effective observation mass is

$$
\tilde { n } _ { t } ( e ) = n _ { t } ( e ) + ( K - 1 ) \left( 1 - \sqrt { \frac { \log T } { 2 p _ { \operatorname* { m i n } } ^ { 2 } n _ { t } ( e ) } } \right) n _ { t } ( e ) p _ { \operatorname* { m i n } } .
$$

Concentration bound. Hoeffding’s inequality applied to the empirical estimate gives $\begin{array} { r } { \operatorname* { P r } ( | \widehat { w } _ { \tilde { n } _ { t } ( e ) } ( e ) - w ( e ) | \geq \varepsilon ) \leq } \end{array}$ $2 \exp ( - 2 \tilde { n } _ { t } ( e ) \varepsilon ^ { 2 } )$

Lower bound on effective observations. Since $n _ { t } ( e ) \ \geq$ $F _ { 0 } ( t ) \log _ { 2 } t$ , dropping lower-order terms, we simplify the dominant scaling as $\tilde { n } _ { t } ( e ) = \Theta ( F _ { 0 } ( t ) \log t \cdot [ 1 + ( K - 1 ) p _ { \mathrm { m i n } } ] )$

Regret bound. The commitment regret becomes

$$
\begin{array} { c l } { \displaystyle R _ { T , C } \leq 2 M | E | \sum _ { t = 1 } ^ { T } \exp \Big ( - 2 F _ { 0 } ( t ) \log ( t ) } \\ { \times ( 1 + ( K - 1 ) p _ { \operatorname* { m i n } } ) \varepsilon ^ { 2 } \Big ) . } \end{array}
$$

Using $\alpha ~ = ~ 1 + ( K - 1 ) p _ { \operatorname* { m i n } }$ , this rewrites to $R _ { T , C } ~ \leq$ $\begin{array} { r } { 2 M \bar { L ^ { M } } \sum _ { t = 1 } ^ { \infty } t ^ { - 2 \alpha \bar { F } _ { 0 } ( t ) \varepsilon ^ { 2 } } } \end{array}$ . Multi-slot feedback increases the effective observation count by α, yielding

$$
\begin{array} { r } { R _ { T , C } \lesssim \frac { M L ^ { M } \sqrt { \pi } } { \sqrt { 2 c } \varepsilon \sqrt { \alpha } } \exp \left( \frac { 1 } { 8 c \alpha \varepsilon ^ { 2 } } \right) } \\ { \times \left[ 1 + \mathrm { e r f } \left( \frac { 1 } { \sqrt { 8 c \alpha } \varepsilon } \right) \right] . } \end{array}
$$

□