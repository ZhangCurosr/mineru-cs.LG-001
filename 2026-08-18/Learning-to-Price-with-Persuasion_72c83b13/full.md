# Learning to Price with Persuasion

Maria-Florina Balcan<sup>1</sup>, Tejas Pagare<sup>1</sup>, and Karan Singh<sup>1</sup>

<sup>1</sup>Carnegie Mellon University, Pittsburgh, USA

ninamf@cs.cmu.edu, tpagare@andrew.cmu.edu, karansingh@cmu.edu

## Abstract

Motivated by modern marketplaces, where the platform or the seller routinely gathers detailed user profiles, we study a novel learning theoretic model that simultaneously involves information and mechanism design. Specifically, we consider the economic setting recently introduced by Bergemann et al. (2022), where in addition to the menu of quality-price pairs, the seller ofers information on the value of the match between product quality and buyer’s taste via a signaling scheme. We relax the assumption that the seller knows the buyers’ belief about the distribution of tastes and study the sample requirements of designing a revenue maximizing scheme. We consider both the batch setting where we have access to data from a set of i.i.d. buyers and an online demand query model where we observe the buyers’ behaviors to seller’s schemes. Despite the apparent non-convexity of the problem, we also give the first FPTAS to compute a scheme that maximizes the revenue within an arbitrarily small additive loss, which was left open by Bergemann et al. (2022). Overall, this brings a new learning perspective in asymmetric economic settings where buyers and sellers know diferent types of information.

## 1 Introduction

A cornerstone of microeconomic theory is information asymmetry. Economic agents participating in a market might have latent preferences (or motives) invisible to the market maker. The foundational works of Spence (1973); Akerlof (1970) detail signaling mechanisms by which participants can partially overcome this information asymmetry, and market failures that result from the unmitigated presence of these circumstances. However, in recent years, the presence of monolithic algorithmic marketplaces has altered this picture. Such marketplaces routinely gather detailed user profiles, consisting of both demographic data and the history of past purchases. For example, recently, a shopper was surprised to discover that Kroger, a not particularly tech-savy retail giant in the US, had a 62-page shopping profile on them, including details like pet ownership and travel history (Kravitz, 2025). As a result, such digital platforms can forecast the personalized value a product or service holds for a given buyer much more accurately than individual buyers themselves. As another example, Google, as an advertising marketplace serving a trillion impressions every month, can arguably estimate the conversion rate associated with displaying a specific advertisement in a specific spot far more accurately than individual publishers.

We consider one of the most foundational problems, namely, monopolistic screening (Mussa and Rosen, 1978; Maskin and Riley, 1984), albeit in the presence of such an informationally advantaged seller. In the classical problem, a monopolist aims to maximize the revenue gathered from a pool of buyers who have a certain distribution of latent tastes, by ofering a menu of quality-price pairs. Products of higher quality also cost the monopolist more. By ofering a menu of varied qualities at diferent price points, the seller can simultaneously appeal to disparate buyers whose utility for a given product is a function of both their latent taste and the product’s quality. In the new-age model, proposed by Bergemann et al. (2022) (also see Bergemann et al. (2026b)), the key distinction is that the seller can forecast the utility that a buyer with a certain taste associates with diferent product qualities, far better than the buyers themselves. A notable constraint on the seller is that in spite of knowing the buyer’s utility precisely she can not engage in perfect price discrimination (driving the consumer surplus to zero), because existing regulations exclude explicit price discrimination in most markets (Mansoor, 2026). Thus, in addition to designing a public menu of quality-price pairs, the seller commits to releasing partial information about the value of buyer-quality matches via a signaling scheme of their choice allowing an indirect price discrimination. In practice, one can think of such a signal as a personalized recommendation to the buyer indicating what product (quality) she should purchase (or in a softer manner through search rankings). Recommendation algorithms in this context play the role of a commitment device. For example, Amazon labels certain products, which are neither the ones with the highest rating nor the highest number of reviews, as Amazon’s choice. Such personalized recommendations can be viewed as signals to the buyer.

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Value distribution</td><td rowspan=1 colspan=1>Sample Complexity / Regret</td></tr><tr><td rowspan=1 colspan=1>Value samples</td><td rowspan=1 colspan=1>Arbitrary</td><td rowspan=1 colspan=1> $\widetilde { \mathcal { O } } ( 1 / \varepsilon ^ { 3 } )$ </td></tr><tr><td rowspan=1 colspan=1>Demand queries</td><td rowspan=1 colspan=1>Discrete</td><td rowspan=1 colspan=1> $\overline { { \mathcal { O } ( 1 ) } }$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Bounded pdf</td><td rowspan=1 colspan=1> $\widetilde { \mathcal { O } } ( 1 / \varepsilon ^ { 2 } )$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Lipschitz pdf</td><td rowspan=1 colspan=1> $\widetilde { \mathcal { O } } ( 1 / \varepsilon )$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Analytic pdf</td><td rowspan=1 colspan=1> $\overline { { \mathcal { O } ( \mathrm { p o l y l o g } ~ 1 / \varepsilon ) } }$ </td></tr><tr><td rowspan=1 colspan=1>Joint learning</td><td rowspan=1 colspan=1>Arbitrary</td><td rowspan=1 colspan=1> $\widetilde { \mathcal { O } } ( T ^ { 3 / 4 } )$ </td></tr></table>

Figure 1: Summary of our results. For value samples and demand queries, we list the sample requirements to produce an ε-optimal menu along with the associated signaling scheme. The last row lists the regret in a T-horizon game.

To design a revenue maximizing menu and signaling scheme in the above context, it is typically assumed that the seller knows the precise distribution of the buyer’s tastes. This is implausible given the diversity of products sold on modern marketplaces, and in this work, we relax this assumption. Instead, we consider explicit learning algorithms that dictate how such knowledge may be acquired via samples and past interactions, with explicit bounds on sample and compute requirements. Concretely, we make the following contributions (also partly summarized in Figure 1).

1. We give a learning algorithm that when given $\widetilde { \mathcal { O } } ( 1 / \varepsilon ^ { 3 } )$ i.i.d. samples from the buyers’ taste (or as we will call it going forward, value) distribution, produces a signaling scheme and menu that achieves a revenue within ε of the maximum.

2. The problem of jointly designing a menu and a signaling scheme is non-convex, as noted in Bergemann et al. (2022). Despite this, we give the first FPTAS that computes a solution in polynomial time with revenue within an arbitrarily small additive loss of the optimum.

3. We also study a demand query model, where the seller can observe how the buyers behave in presence of the menus and signaling schemes she designs. In this interactive setting, we obtain a constant sample complexity for discrete value distributions and improve to $\widetilde { \mathcal { O } } ( 1 / \varepsilon ^ { 2 } )$ samples for continuous distributions with bounded pdfs. Further smoothness assumptions result in smaller sample requirements; for example, polylogarithmic for analytic pdfs.

4. Finally, we give regret upper bounds for a model in which both the seller and the buyer population jointly learn in an online setting based on the past realizations of values. In this setting, the seller has to design schemes that are somewhat robust to the buyers’ beliefs, which are incompletely specified.

## 1.1 Related Work

In economics, the nonlinear pricing problem introduced in Mussa and Rosen (1978) and further studied in Maskin and Riley (1984) characterizes the optimal quality, price menus for a revenue maximizing monopolist seller facing privately informed random buyer or a population of buyer with known value distribution. See Wilson (1993) for a broader survey and Bolton and Dewatripont (2004) for the simplified analysis for computing optimal menu.The information design literature in which a informationally advantaged sender influences receiver via communication was initially popularized by a cheap talk model Crawford and Sobel (1982). This was then formalized with a commitment power to the sender on distribution of messages as Bayesian Persuasion in Kamenica and Gentzkow (2011). See Bergemann and Morris (2019); Kamenica (2019) for a survey treatment. This was then translated to the linear bayesian persuasion setting Gentzkow and Kamenica (2016); Dworczak and Martini (2019) where the senders’ utility is solely characterized by the expected posterior value. We build on Bergemann et al. (2026b), who prove that the seller-optimal joint design always takes a monotone partitional form with finitely many signals and menu items. See also Bergemann et al. (2026a) for more developments on the joint mechanism and information design problem.

The intersection of machine learning and economics is now a flourishing area of research. The use of sample complexity theory for mechanism design was initiated by Balcan et al. (2005, 2008). It was later studied by many others including Cole and Roughgarden (2014); Roughgarden and Schrijvers (2016) for single-item auction and Morgenstern and Roughgarden (2016); Balcan et al. (2016); Mohri and Medina (2016) for parametrized auction classes. Online learning with unknown demand distribution for posted pricing was studied in Kleinberg and Leighton (2003), see also Balcan and Beyhaghi (2024) and Balcan et al. (2025) for other pricing mechanisms. Online learning in Bayesian Persuasion, where the sender faces a receiver with unknown type, which characterizes the receiver’s utility function, was initiated in Castiglioni et al. (2020) and further developed in Babichenko et al. (2022); Zu et al. (2025); Castiglioni et al. (2021). Recent developments include Bacchiocchi et al. (2024) which further considers that the sender does not know the common prior, Zu et al. (2025) studies the sender interaction with a population of receivers who also does not know the common prior. See also Balcan et al. (2026a,b) who consider learning in the presence of additional side-information which is public to both the sender and receiver.

A notable work on the learning-theoretic interface of the joint information design and pricing problem is Agrawal et al. (2023), which studies a variation of the posted pricing problem in which the seller reveals information about the uncertain quality of a good. However, the mechanism formats, the posted pricing in the former, and the menu pricing for us, are fundamentally incomparable. For instance, the ratio of optimal revenues for posted pricing for a single quality product vs. menu pricing can be vanishingly small. In addition, the former assumes that the seller knows the (exogenous) quality distribution but not the buyers’ demand distribution, whereas in our case, the seller can produce any quality and influence the demand/type of the buyer through carefully constructed signals.

## 2 Problem Setup and Preliminaries

We consider the setting from Bergemann et al. (2022), where a monopolist seller can manufacture goods of arbitrary (positive) quality $q$ in the range $\mathcal { Q } = [ q , \overline { { q } } ]$ at a cost $c ( q )$ , which is assumed to be convex and non-decreasing. The buyer’s value $V$ is distributed according to a common prior on $\nu = [ 0 , 1 ]$ with the CDF $F ( v ) : = P ( V \leq v )$ . Following Mussa and Rosen (1978), a buyer with value v gains a utility of $v \cdot q - p$ upon purchasing a product of quality q at price $p .$ Buyers in this setup are Bayesian utility-maximizing agents, who are ex ante unaware of their own value in contrast to Mussa and Rosen (1978).

The seller commits to a public menu and a signaling scheme; the latter is sometimes termed an information structure or a Blackwell experiment (Blackwell, 1953). The public menu $\mathcal { M } = \{ ( q _ { i } , p _ { i } ) : i \in I \}$ is composed of a collection of quality-price tuples for some index set I, with qualities restricted to $\mathcal { Q }$ and prices at most q so that buyers still find the menu attractive. $\mathrm { B y }$ an argument mirroring the revelation principle, sometimes termed an obedience argument, we can generically assume that the signal space S as being in a one-to-one correspondence with the menu and, therefore, just as numerous. Intuitively, each menu item can be labeled by the signal for which it is the best response – if there is no such signal, it can be removed altogether – and signals inducing the same menu item as the best response can be merged. Formally, the signaling scheme $\mathcal { T } : = ( S , \{ \pi ( \cdot | v ) \} _ { v \in \mathcal { V } } )$ encodes a signal space $\mathcal { S } \subseteq \mathcal { Q }$ and a conditional distribution over signals for every possible value $\pi ( \cdot | v ) \in \Delta ( \mathcal { S } )$ . To fully specify the timeline, upon seeing a buyer with value $v ,$ the seller samples a signal $s \sim \pi ( \cdot | v )$ , as promised upon commitment. The buyer, upon observing a signal $s ,$ forms a posterior mean $\begin{array} { r } { \bar { v } _ { \pi , F } ( s ) = [ \int _ { 0 } ^ { 1 } v \pi ( s | v ) \mathrm { d } F ( v ) ] / [ \int _ { 0 } ^ { 1 } \pi ( s | v ) \mathrm { d } F ( v ) ] } \end{array}$ ], and rationally chooses a quality $( \mathsf { q } ( s ) , \mathsf { p } ( s ) ) \in$ arg max $\cdot ( q , p ) { \in } \mathcal { M } ^ { \overline { { \upsilon } } _ { \pi , F } ( s ) \cdot q - p }$ , breaking ties in the favor of the seller. This gives the seller a revenue of $\mathsf { p } ( s ) - c ( \mathsf { q } ( s ) )$ which is the profit minus the production cost. The seller thus aims to find an incentive-compatible direct menu $\{ ( \mathsf { q } ( s ) , \mathsf { p } ( s ) ) : s \in S \}$ and a signaling scheme $\mathcal { T }$ to maximize the expected revenue which corresponds to the following program where the first constraints enforce individual rationality (IR) and the second constraints enforce incentive compatibility (IC) which ensures that menu item $( \mathsf { q } ( s ) , \mathsf { p } ( s ) )$ is a best-response for the buyer upon receiving signal s.

$$
\begin{array} { r l } & { \underset { S \subseteq Q , \ \left\{ ( q ( s ) , \mathfrak { p } ) \colon s \in S \right\} } { \operatorname* { m a x } } \ \underset { s \in S } { \sum } \ \underset { 0 } { \sum } \int _ { 0 } ^ { 1 } ( \mathfrak { p } ( s ) - c ( \mathfrak { q } ( s ) ) ) \pi ( s | v ) \mathrm { d } F ( v ) } \\ & { \overset { S \subseteq Q , \ \{ ( q ( s ) , \mathfrak { p } ) \colon s \in S \} } { \overbrace { s \sigma ( s ) \colon v \in V ) } } \ \underset { s \in S } { \sum } \ \underset { ( s ) \ \cdot } { \sum } \ q ( s ) - \mathfrak { p } ( s ) \geq 0 \ \forall s \in S \quad \mathrm { ( I R ) } \quad \mathrm { a n d } } \\ & { \qquad \quad \overline { { v } } _ { \pi , F } ( s ) \cdot \mathfrak { q } ( s ) - \mathfrak { p } ( s ) \geq \bar { v } _ { \pi , F } ( s ) \cdot \mathfrak { q } ( s ^ { \prime } ) - \mathfrak { p } ( s ^ { \prime } ) \ \forall s , s ^ { \prime } \in S \quad \mathrm { ( I C ) } . } \end{array}\tag{1}
$$

A key result from Bergemann et al. (2022) that we will use states that the resulting menu has only a finite number of items, even if the underlying value distribution is continuous. In contrast, in Mussa and Rosen (1978), where there is no information provision, this number can be unbounded.

Definition 1. A signaling scheme $\mathcal { T } : = ( S , \{ \pi ( \cdot | v ) \} _ { v \in \mathcal { V } } )$ ) belongs to the set of monotone partitional information structures $\mathcal { T } _ { \mathsf { p a r t } }$ , if there exists an $I \in \mathbb { Z } _ { > 0 }$ and $0 = t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot < t _ { I } = 1$ such that upon receiving any signal i each buyer only knows that their value corresponds to the quantile in $[ t _ { i - 1 } , t _ { i } )$ with respect to the distribution of values.

Theorem 1 (Bergemann et al. (2026b)). The optimal menu has as most $K : = \overline { { q } } / q$ items. Furthermore, the optimal information structure is monotone partitional with at most K distinct signals.

## 3 Sample Complexity with Value Samples

In this section, we consider the setting in which the seller can draw i.i.d. samples from the value distribution $F ,$ , and uses these in Algorithm 1 to compute a near-optimal menu and signaling scheme.

We explain the Algorithm 1 with the four key ingredients that we use, the last three of which are novel in this setting. First, we solve the best monotone partitional signaling scheme and menu with empirica distribution, by solving the optimization program from Bergemann et al. (2022). In fact, in Section 4, we will give an algorithm to do this step in polynomial time. This gives the quantile partitions of values corresponding to diferent signals, as in Definition 1. Second, to actually implement such a signaling scheme on the true distribution, we translate this description into the value space. There is some subtlety to this and we describe this in detail in Section 3.1. Third, the buyer’s response, i.e., the item they pick from the menu, is dictated by their signal-conditioned posterior mean, which is diferent on the true and the empirical distributions. Posting the empirically determined menu as is can result in a constant revenue loss. To fix this, we discount the prices in a linear-additive manner in Line 5 so that a buyer does not defect to a menu item cheaper than what was intended for them in the empirical world, since a price corresponding to a higher signal also gets a larger discount. Multiplicative discounting of the form $\tilde { p } _ { k } = ( 1 - \rho ) p _ { k }$ is a common tool to repair incentive-compatibility constraints (e.g., see Lemma 9 in Cai and Velegkas (2021)), but this would lead to a sample complexity of $\widetilde { \mathcal { O } } ( 1 / \varepsilon ^ { 5 } )$ in our setting. The linear-additive discount crucially relies on the menu being bounded in size (Li et al., 2025), which a multiplicative discount does not take advantage of. Finally, the signal-conditioned posterior means can be arbitrarily far of under $F$ and $\widehat F$ for signals with low marginal probability. So, in truth, we bundle all such signals, termed bad, into a null signal, and only attempt to repair incentive compatibility for good signals. The finitude of the menu once again means that the revenue loss due to this bundling of (individually) rare signals can not be too high.

From now on, we will use $R e v ^ { * } ( F )$ to denote the optimal revenue associated with a distribution $F ,$ and $R e v ( { \pmb x } ; F )$ to denote the revenue associated with a menu-signaling-scheme pair x on the same distribution. The distribution $F$ in fact plays two roles in determining the revenue (see Equation (1)): it appears in the objective dictating the probability with which signals are generated, and it is used to update the buyer’s belief to arrive at posterior means. In more elaborate tri-variate form, for the analysis, we will let $R e v ( { \pmb x } ; G , F )$ be the revenue when signals are generated according to $F _ { ; }$ , yet buyers update their beliefs starting from the prior G. Thus, ${ \mathit { R e v } } ( { \pmb x } ; { \cal F } ) = { \mathit { R e v } } ( { \pmb x } ; { \cal F } , { \cal F } )$

Algorithm 1: Empirical Revenue Maximization (ERM) via Samples   
Input: value distribution ${ \overline { { F , } } }$ sample budget $n ,$ cutof $\tau ,$ discount $\rho .$   
1 Construct the empirical distribution $\begin{array} { r } { \widehat { F } ( v ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { 1 } _ { v _ { i } \leq v } } \end{array}$ by drawing n i.i.d. samples from $F .$   
2 Let $( t _ { k } : k \in [ K ] )$ be the quantile description of a near best monotone partitional signaling scheme for   
the distribution ${ \hat { F } } ,$ with the direct menu $( q _ { k } , p _ { k } : k \in [ K ] )$ , up to ε<sub>OPT</sub> error.   
3 Optionally, $\mathrm { i f ~ } \varepsilon _ { \mathrm { O P T } } > 0 .$ run the procedure in Section A.4 to obtain a direct menu with non-decreasing   
$( p _ { k } - c ( q _ { k } ) : k \in [ K ] )$ of equal or greater revenue.   
4 Use the quantiles $( t _ { k } : k \in [ K ] )$ to generate an explicit value-space description of the form   
$( \pi ( k \mid v ) : \forall v \in \mathcal { V } , k \in [ K ] )$ with respect to ${ \widehat { F } } ,$ as detailed in Section 3.1.   
5 Designate $z _ { \tau } = \{ k \in [ K ] : t _ { k } - t _ { k - 1 } < \tau \}$ as the set of bad signals.   
6 Modify the prices as $\tilde { p } _ { k } = p _ { k } - k \rho$ for all $k \in [ K ] .$   
7 Design a modified signaling scheme that clumps all bad signals into a new null signal as   
${ \tilde { \pi } } ( s \mid v ) = { \left\{ \begin{array} { l l } { \pi ( s \mid v ) } & { { \mathrm { ~ i f ~ } } s \not \in z _ { \tau } , } \\ { \sum _ { k \in z _ { \tau } } \pi ( k \mid v ) } & { { \mathrm { ~ i f ~ } } s = { \mathrm { n u l l } } } \end{array} \right. }$ l   
Output: $\tilde { \pmb { x } } _ { \hat { F } }$ encoding $( ( q _ { k } , \tilde { p } _ { k } ) : k \in [ K ] )$ and $( { \tilde { \pi } } ( s \mid v ) : v \in \mathcal { V } , s \in [ K ] \cup \{ \mathtt { n u l l } \} )$

Theorem 2. For any $\varepsilon , \delta > 0$ , there exists a setting of $n , \tau , \rho$ and $\varepsilon _ { O P T } = 0$ satisfying $\tau = \widetilde { \mathcal { O } } ( \varepsilon / \overline { { q } } K )$ and $\rho = \widetilde { \mathcal { O } } ( \varepsilon / K )$ such that upon observing $\begin{array} { r } { n = \tilde { \mathcal { O } } \left( \frac { K ^ { 3 } \overline { { q } } ^ { 3 } } { \varepsilon ^ { 3 } } \log \frac { 1 } { \delta } \right) } \end{array}$ i.i.d. samples from $F ,$ , with probability at least $1 - \delta ,$ Algorithm 1 produces a menu-signaling-scheme pair $\tilde { \pmb { x } } _ { \hat { F } }$ satisfying

$$
R e v ( \tilde { x } _ { \hat { F } } ; F ) \geq R e v ^ { \ast } ( F ) - \varepsilon .
$$

Further, for optimization error $\varepsilon _ { O P T } > 0$ in line ${ \it 2 } ,$ the revenue is reduced by an additive $\varepsilon _ { O P T }$ term.

If we constrain our menu to a single item, we can improve the sample complexity to $1 / \varepsilon ^ { 2 }$ . In fact, Bergemann et al. (2022) provide suficient conditions under which such a menu may be optimal within the wider unrestricted class. We preview the key points in the analysis in Section 3.2, where we also highlight the dificulty of the multi-item case, while complete proofs are given in Section A.

Theorem 3. For any $\varepsilon , \delta > 0 _ { ; }$ , there exists an algorithm that draws $\begin{array} { r } { \tilde { \mathcal { O } } \left( \frac { \overline { { q } } ^ { 2 } } { \varepsilon ^ { 2 } } \log \frac { 1 } { \delta } \right) } \end{array}$ i.i.d. samples from $F ,$ and with probability at least $1 - \delta _ { i }$ , produces a signaling scheme and a menu containing a single item with revenue within ε of the optimal revenue generated by single-item menus.

## 3.1 Value-space Description of Quantile-based Signals

Recall that the buyer chooses a menu item using the prior $F$ which is unknown to the seller. In this section, given a quantile description of a monotone partitional signaling scheme $( t _ { k } : k \in [ K ] )$ associated with a distribution ${ \widehat F } .$ , we give a recipe to convert this to a value-space description: given a specific $v ,$ what is the corresponding distribution over signals? This matters because we are transplanting a signaling scheme designed for $\widehat F$ to $F _ { ; }$ , and only a value-space description is invariant to changes in the prior distribution. For a continuous distribution ${ \widehat F } .$ , this is straightforward as noted in Bergemann et al. (2022); namely, all values in the range $[ \widehat { F } ^ { - 1 } ( t _ { k - 1 } ) , \widehat { F } ^ { - 1 } ( t _ { k } ) )$ are assigned the signal k deterministically, where $\widehat F ^ { - 1 } ( t ) = \operatorname* { i n f } \{ v : t \leq \widehat F ( v ) \}$ is the generalized inverse (or quantile) function. <sup>1</sup>

The translation for discrete (or mixed) distributions is more delicate and requires randomized signaling. For all k such that $\widehat { F } ^ { - 1 } ( t _ { k } ) = \widehat { F } ^ { - 1 } \dot { ( t _ { k - 1 } ) } = v _ { 0 }$ (say), we choose the conditional law $\pi ( k | v ) = ( t _ { k } - t _ { k - 1 } ) / ( \widehat { F } ( v _ { 0 } ) -$ $\widehat { F } ( v _ { 0 } ^ { - } ) )$ for $v = v _ { 0 }$ and 0 for all other $v ,$ where $\widehat { F } ( v _ { 0 } ^ { - } )$ is the left limit of $\widehat F$ at $v _ { 0 }$ . Intuitively, only a part of the total mass $\widehat { F } ( v _ { 0 } ) - \widehat { F } ( v _ { 0 } ^ { - } )$ of $v _ { 0 }$ is assigned to the signal k. For all k such that $\widehat F ^ { - 1 } ( t _ { k } ) = v _ { 1 }$ and $\widehat { F } ^ { - 1 } ( t _ { k - 1 } ) = v _ { 0 }$ with $v _ { 0 } \neq v _ { 1 }$ , the conditional probability of signal k is

$$
\pi ( k | v ) = \mathbf { 1 } \{ x = v _ { 0 } \} \frac { \widehat { F } ( v _ { 0 } ) - t _ { k - 1 } } { \widehat { F } ( v _ { 0 } ) - \widehat { F } ( v _ { 0 } ^ { - } ) } + \mathbf { 1 } \{ x \in ( v _ { 0 } , v _ { 1 } ) \} + \mathbf { 1 } \{ x = v _ { 1 } \} \frac { t _ { k } - \widehat { F } ( v _ { 1 } ^ { - } ) } { \widehat { F } ( v _ { 1 } ) - \widehat { F } ( v _ { 1 } ^ { - } ) } .
$$

The key property of this translation is that the marginal probability of generating signal k under $\widehat F$ remains $t _ { k } - t _ { k - 1 }$ and $\begin{array} { r } { \mathbb { E } _ { \widehat { F } } [ v \vert k ] = \int _ { t _ { k - 1 } } ^ { t _ { k } } \widehat { F } ^ { - 1 } ( x ) d x / ( t _ { k } - t _ { k - 1 } ) } \end{array}$ by an immediate application of integration by parts, exactly as intended in the quantile space.

## 3.2 Key Ingredients in the Analysis

The main point of care is that for the same signaling scheme π, the posterior means conditioned on the same signal $\overline { { v } } _ { \pi , F } ( k )$ and $\overline { { v } } _ { \pi , \widehat { F } } ( k )$ can be diferent. This can break IC and IR constraints, resulting in a receiver of signal k buying a diferent item than intended or nothing at all, netting $O ( 1 )$ revenue loss. To fix this, we discount the prices. Naively, one might hope to discount the price by $q _ { k } | \overline { { v } } _ { \pi , F } ( k ) - \overline { { v } } _ { \pi , \widehat { F } } ( k )$ | for item k. Such conditional means can be arbitrarily far apart for rare signals; one way to see this is that the probability of signal’s occurrence appears in the denominator of these quantities. This by itself might not seem bad because the contribution to the net revenue for a signal k is also proportional to the probability of its occurrence. This reasoning indeed gets $\widetilde { \mathcal { O } } ( 1 / \varepsilon ^ { 2 } )$ sample complexity against the benchmark of single-item menus (see Theorem 3).

But for a multi-item menu, that is, the general case, this mode of analysis breaks down because discounting $p _ { k }$ by itself can break incentive-compatibility for other types (recipients of other signals). So, any discount ofered on p<sub>k</sub> must propagate through higher prices too, to ensure that other types do not defect to cheaper products (see Lemma 3). Thus, the impact of discounting the price by $\Delta$ at some signal k is not localized, and afects the net revenue by $O ( K \Delta )$ . This is why we designate rare signals as bad signals, and only hope to repair incentive-compatibility constraints on good signals, where the conditional means can be estimated well, and hence, where limited discounting is suficient.

The above framework gets us a sample complexity of $\widetilde { \mathcal { O } } ( 1 / \varepsilon ^ { 4 } )$ , when paired with standard uniform deviation arguments (e.g., the DKW inequality Dvoretzky et al. (1956); Massart (1990)). Such uniform arguments are needed because the boundaries of our signals are data-dependent, since we optimize over these. To improve upon this, we in fact prove a variance-aware uniform deviation inequality that says the error in conditional means scales as $1 / \sqrt { \nu n }$ (upto lower order terms), instead of $1 / \nu \sqrt { n }$ , where $\nu$ is the frequency of the signal’s occurrence (see Lemma 2). Our results imply a variance-aware DKW inequality whose existence has been posed as an open question in Guo et al. (2019), but in fact we show that such results can be derived with relative ease from known relative-deviation-style uniform convergence bounds for VC classes (Haussler, 1992; Li et al., 2001; Bartlett et al., 2005; Maurer and Pontil, 2009).

## 4 Additive FPTAS for Revenue Maximization

As usual for an FPTAS, the key is to formulate a dynamic program on suitably small state space. What makes this possible is the fact that while scanning values from left to right, or quantiles from bottom to top, with the intention of pooling these, it is enough to remember just the quality assigned to the last pool (instead of all qualities). This relies on the structure of incentive compatibility constraints in the setting of Mussa and Rosen (1978), as we detail next. The theorem as stated, and proved in Section D, only holds for discrete distributions. But the generalization to arbitrary distributions is $f r e e ,$ given sampling access, because the empirical distribution is discrete, and we can plug this algorithm into line 2 of Algorithm 1.

Algorithm 2: Dynamic Programming (DP) for Optimal Menu and Signaling Scheme   
Input: CDF F on $[ 0 , 1 ] ;$ quantile grid $\pmb { t } = \left( t _ { 0 } , t _ { 1 } , \ldots , t _ { n _ { t } } \right)$ with $t _ { 0 } = 0 , t _ { i } = i \varepsilon ;$ quality grid   
$\mathcal { E } _ { q } = \{ q _ { 0 } , q _ { 1 } , . . . , q _ { n _ { q } } \}$ with $q _ { 0 } = 0 , q _ { j } = \underline { { q } } + j \varepsilon ;$ max. menu size K and $A ( t )$   
1 $\mathtt { R e v } [ 0 , 0 , 0 ] : = 0$ and $\mathtt { R e v } [ k , r , j ] : = - \infty$ for all $( k , r , j ) \neq ( 0 , 0 , 0 ) / /$ Initialization   
2 for $k = 1 , \ldots , K ; r = 1 , \ldots , m ; j = 1 , \ldots , n _ { q }$ do   
3 $\begin{array} { r } { \mathtt { R e v } [ k , r , j ] : = \operatorname* { m a x } _ { 0 \le \ell < r , \ 0 \le j ^ { \prime } \le j } \left\{ \mathtt { R e v } [ k - \mathtt { i } , \ell , j ^ { \prime } ] + \mathcal { U } [ \ell , r , j ^ { \prime } , j ] \right\} ; } \end{array}$   
4 $\begin{array} { r } { \operatorname { p t r } [ k , r , j ] : = \arg \operatorname* { m a x } _ { 0 \leq \ell < r , \ : 0 \leq j ^ { \prime } \leq j } \left\{ \operatorname { R e v } [ k - 1 , \ell , j ^ { \prime } ] + \mathcal { U } [ \ell , r , j ^ { \prime } , j ] \right\} ; } \end{array}$   
5 $\begin{array} { r } { ( k ^ { \star } , j ^ { \star } ) : = \arg \operatorname* { m a x } _ { 1 \leq k \leq K , 1 \leq j \leq n _ { q } } \mathrm { R e v } [ k , m , j ] , r _ { k ^ { \star } } : = n _ { t } , j _ { k ^ { \star } } : = j ^ { \star } , r _ { 0 } : = 0 ; } \end{array}$   
6 for $k = k ^ { \star } - 1$ to 1 do   
${  \begin{array} { l l l } { { \boldsymbol { \tau } } } & { { \lfloor \ ( r _ { k } , j _ { k } ) : = \mathtt { p t r } [ k + 1 , r _ { k + 1 } , j _ { k + 1 } ] ; } } \end{array}  }$   
8 for $k = 1$ to $k ^ { \star }$ do   
9 $\begin{array} { r } { p _ { k } : = \mu _ { F } ( r _ { k - 1 } , r _ { k } ) q _ { j _ { k } } - \sum _ { \ell = 1 } ^ { k - 1 } \bigl ( \mu _ { F } ( r _ { \ell } , r _ { \ell + 1 } ) - \mu _ { F } ( r _ { \ell - 1 } , r _ { \ell } ) \bigr ) q _ { j _ { \ell } } ; } \end{array}$   
Output: Menu $\{ ( p _ { k } , q _ { j _ { k } } ) \} _ { k = 1 } ^ { k ^ { \star } }$ and signaling scheme $\mathcal { T } = ( k ^ { \star } , ( t _ { r _ { k } } : k \in [ k ^ { \star } ] ) )$

Theorem 4 (Additive-FPTAS). For any $\varepsilon _ { 0 \mathsf { P T } } > 0 .$ , given a discrete distribution with support size $N _ { ; }$ Algorithm 2 runs in po $\mathsf { l y } ( K , N , \overline { { q } } , 1 / \varepsilon _ { 0 \mathsf { P T } } )$ time and outputs a menu $\{ ( p _ { k } , q _ { j _ { k } } ) \} _ { k = 1 } ^ { k ^ { \star } }$ and a monotone paritional signaling scheme $\mathcal { T } = ( k ^ { \star } , ( t _ { r _ { k } } : k \in [ k ^ { \star } ] ) )$ ) with revenue exceeding $R e v ^ { \star } ( F ) \bar { - } \varepsilon _ { 0 \mathsf { P T } }$ by setting the resolution of grid as $\varepsilon = ( \varepsilon _ { 0 \mathsf { P T } } ) ^ { 2 } / ( K ( 6 \overline { { q } } + 1 + \overline { { q } } ) ) ^ { 2 }$

We start by simplifying the optimization problem (Equation (1)). It is well known that the participation constraint for the lowest type binds and incentive compatibility constraints for all other types bind with the immediately lower type (Maskin and Riley, 1984). Moreover, the constraints imply monotonicity of the qualities with respect to the types (or values). That is $\bar { v } _ { i } \cdot q _ { i } - p _ { i } = \bar { v } _ { i } \cdot q _ { i - 1 } - p _ { i - 1 }$ with $q _ { 0 } = p _ { 0 } = 0$ and $q _ { i } \geq q _ { i - 1 } \forall i$ . This gives us the so called payment formula $\begin{array} { r } { p _ { i } = \bar { v } _ { i } q _ { i } - \sum _ { j = 1 } ^ { \imath - 1 } ( \bar { v } _ { j + 1 } - \bar { v } _ { j } ) q _ { j } } \end{array}$ . Thus, we can write the optimization problem in the quantile space, which we use in the dynamic programming formulation, as

$$
\begin{array} { l } { { \mathrm { L } _ { F } ^ { s } : \displaystyle \operatorname* { m a x } _ { \big ( ( I , t ) , ( q _ { i } ) _ { i \in [ I ] } \big ) } \sum _ { i = 1 } ^ { I } \bar { v } _ { i } \big ( 1 - t _ { i - 1 } \big ) \big ( q _ { i } - q _ { i - 1 } \big ) - \big ( t _ { i } - t _ { i - 1 } \big ) c \big ( q _ { i } \big ) } } \\ { { \mathrm { } q _ { i } \geq q _ { i - 1 } \forall i \in \{ 1 , \dots , I \} \mathrm { ~ w i t h ~ } q _ { 0 } = 0 , \quad t _ { 0 } = 1 - t _ { I } = 0 , \mathrm { ~ a n d ~ } \bar { v } _ { i } = \frac { f _ { t _ { i - 1 } } ^ { t _ { i } } F ^ { - 1 } ( t ) \mathrm { d } t } { t _ { i } - t _ { i - 1 } } } } \end{array}
$$

Let $\mathcal { E } _ { q } = \{ q _ { 0 } , q _ { 1 } , . . . , q _ { n _ { q } } \}$ with $q _ { 0 } = 0$ be the quality grid, where $q _ { 0 }$ denotes the outside option and $q _ { j } : = \underline { { { q } } } + j \varepsilon$ for $j \in \{ 1 , \ldots , n _ { q } \}$ , with $\begin{array} { r } { n _ { q } = \lceil \frac { \overline { { q } } - \underline { { q } } } { \varepsilon } \rceil } \end{array}$ , and $\mathcal { E } _ { t } = \{ t _ { 0 } , \ldots , t _ { n _ { t } } \}$ with $t _ { 0 } = 1 - t _ { n { t } } = 0$ be the quantile grid with $t _ { i } = i \varepsilon$ and $n _ { t } = 1 / \varepsilon \in \mathbb { Z } _ { > 0 }$ . Define $\begin{array} { r } { A ( t ) = \int _ { 0 } ^ { t } F ^ { - 1 } ( s ) } \end{array}$ ds $\forall t \in [ 0 , 1 ]$ . For indices $0 \leq \ell < r \leq m .$ define $\begin{array} { r } { \mu _ { F } ( \ell , r ) : = \frac { A ( t _ { r } ) - A ( t _ { \ell } ) } { t _ { r } - t _ { \ell } } } \end{array}$ the posterior mean of the quantile interval $\left( t _ { \ell } , t _ { r } \right]$ under $F .$ For the dynamic programming we store $\mathsf { R e v } [ k , r , j ]$ which denotes the maximum revenue from partitioning $( 0 , t _ { r } ]$ into k nonempty quantile grid intervals with the last interval assigned quality $q _ { j } \in \mathcal { E } _ { q }$ . Moreover, we store $\mathcal { U } [ \ell , r , j ^ { \prime } , j ]$ for quality indices $0 \leq j ^ { \prime } \leq j \leq n _ { q }$ which we define as the transition utility

$$
\mathcal { U } [ \ell , r , j ^ { \prime } , j ] : = \mu _ { F } ( \ell , r ) ( 1 - t _ { \ell } ) ( q _ { j } - q _ { j ^ { \prime } } ) - c ( q _ { j } ) ( t _ { r } - t _ { \ell } ) ,
$$

i.e. the revenue contribution from assigning quality $q _ { j }$ to the interval $( t _ { \ell } , t _ { r } ]$ , given that the preceding interval was assigned quality $q _ { j ^ { \prime } }$ . The dynamic programming recursion immediately follows where one needs to maintain the monotonicity of quality being assigned.

## 5 Sample Complexity with Demand Queries

We consider a setting where the seller deploys signaling schemes and menus of their choice, and observes how a buyer with a randomly drawn value responds to it. Despite the stochasticity in values, we will see

```latex
Algorithm 3: Empirical Revenue Maximization via Demand Queries
Input: query budget $n ,$ cutof $\tau ,$ discount $\rho .$
1 Fix a menu ${ \mathcal { M } } = { \bar { \{ } }  ( q , p ( q ) : = ( q - \underline { { q } } ) ^ { 2 } / 2 ( { \overline { { q } } } - \underline { { q } } ) ) : q \in [ \underline { { q } } , \overline { { q } } ] \}$
2 for $j = 1 , \dots n$ do
3 Choose a signaling scheme with two signals $\pi _ { j } ( 0 | v ) = 1 - \pi _ { j } ( 1 | v ) = v ^ { j - 1 }$
4 Observe the generated signal $s _ { j }$ and the purchased quality $q _ { j }$ when the menu M was coupled with
a signaling scheme $\pi _ { j } .$
5 Let $q _ { j } ^ { \prime } = ( q _ { j } - \underline { { q } } ) / ( \overline { { q } } - \underline { { q } } )$ be the normalized quali $\mathrm { 5 y . }$
$m _ { j } = \left\{ \begin{array} { l l } { q _ { j } ^ { \prime } m _ { j - 1 } } \\ { m _ { 1 } - q _ { j } ^ { \prime } ( 1 - m _ { j - 1 } ) } \end{array} \right.$ if $s _ { j } = 0 ,$
6 With the convention that $m _ { 0 } = 1$ , set
if $s _ { j } = 1 .$
7 if the value distribution is discrete then
8 For a discrete value distribution with support $\{ x _ { 1 } , \ldots . x _ { n + 1 } \}$ of size $n + 1$ , calculate the pmf
$p \in \Delta ( [ n ] )$ by solving the linear system $V p = m$ , where $\pmb { m } = [ m _ { 0 } , m _ { 1 } , . . . m _ { n } ] ^ { \top }$ and
$\dot { V } \in \mathbb { R } ^ { \dot { ( n + 1 ) } \times \dot { ( n + 1 ) } }$ is a Vandermonde matrix with $V _ { i , j } = x _ { j } ^ { i }$
9 Let $\widehat F$ be the corresponding CDF.
10 else
11 Let $\textstyle L _ { j } ( x ) = \sum _ { i = 0 } ^ { j } a _ { j i } x ^ { i }$ be the expansion of the $j ^ { t h }$ Legendre polynomial shifted to $[ 0 , 1 ] .$
12 Calculate $\begin{array} { r } { \alpha _ { j } = \sum _ { i = 0 } ^ { j } a _ { j i } ( 1 - m _ { i + 1 } / ( i + 1 ) ) } \end{array}$ and $\begin{array} { r } { \tilde { F } _ { j } ( x ) = \sum _ { i = 0 } ^ { j } \alpha _ { i } L _ { i } ( x ) } \end{array}$ , which is the degree-j
Legendre projection of the CDF.
13 Compute the de la Vallee Poussin mean $\begin{array} { r } { \hat { F } _ { \lceil n / 2 \rceil } ( x ) = \frac { 1 } { \lceil n / 2 \rceil } \sum _ { i = \lceil n / 2 \rceil } ^ { 2 \lceil n / 2 \rceil - 1 } \tilde { F } _ { i } ( x ) } \end{array}$
14 Let $\hat { F }$ be the smallest non-negative non-decreasing upper envelope of $\hat { F } _ { \lceil n / 2 \rceil }$
15 Let $( t _ { k } : k \in [ K ] )$ be the quantile description of the best monotone partitional signaling scheme for the
distribution $\hat { F } _ { ; }$ , with the direct menu $( q _ { k } , p _ { k } : k \in [ K ] )$
16 Use the quantiles $( t _ { k } : k \in [ K ] )$ to generate an explicit value-space description of the form
$( \pi ( k \mid v ) : \forall v \in \mathcal { V } , k \in [ K ] )$ with respect to ${ \widehat { F } } ,$ as detailed in Section 3.1.
17 Designate $z _ { \tau } = \{ i \in [ I ] : t _ { i } - t _ { i - 1 } < \tau \}$ as the set of bad signals.
18 Modify the prices as $\tilde { p } _ { k } = p _ { k } - k \rho$ for all $k \in [ K ]$
19 Design a modified signaling scheme that clumps all bad signals into a new null signal as
${ \tilde { \pi } } ( s \mid v ) = { \left\{ \begin{array} { l l } { \pi _ { k } ( s \mid v ) } & { { \mathrm { ~ i f ~ } } s \not \in z _ { \tau } , } \\ { \sum _ { k \in z _ { \tau } } \pi _ { k } ( k \mid v ) } & { { \mathrm { ~ i f ~ } } s = { \mathrm { n u l } } { \mathrm { . } } } \end{array} \right. }$ l
Output: $\tilde { \pmb { x } } _ { \hat { F } }$ encoding $( ( q _ { k } , \tilde { p } _ { k } ) : k \in [ K ] )$ and $( { \tilde { \pi } } ( s \mid v ) : v \in \mathcal { V } , s \in [ K ] \cup \{ \mathrm { n u 1 1 } \} ) .$
```

that it is possible to extract the moments of the value distribution exactly in this setting, without any error. We couple this with results from approximation theory (Trefethen, 2019) to estimate the underlying value distribution. We define the setting first and then give the algorithm.

Definition 2. A demand query accepts as input a signaling scheme $( S , \{ \pi ( \cdot | v ) \} _ { v \in \mathcal { V } } )$ and a menu $( ( p ( z ) , q ( z ) ) _ { z \in Z } )$ for some index set Z and outputs the signal $s \sim \pi ( \cdot | v )$ made available to the buyer along with the item $( q , p )$ selected by the buyer, where the value of the buyer v is drawn randomly from $F$

## 5.1 Approximating Discrete Distributions

The main observation is that in this model we can exactly simulate bounded linear functionals of the pmf (or the pdf) corresponding to the common prior F. We specifically chose moments. Interestingly, this makes our queries non-adaptive.

Theorem 5. For any discrete distribution F with support size $n + 1$ , when Algorithm 3 is run for n queries, with $\tau = \rho = 0$ , it returns a revenue-maximizing menu and signaling scheme pair $\tilde { \pmb { x } } _ { \widehat { F } }$ that attains the optimal revenue, that is, $R e v ( \tilde { { \bf x } } _ { \widehat { F } } ) = R e v ^ { { \ast } } ( F )$

We first begin with the following moment recovery lemma.

Lemma 1. For any $j \le n , m _ { j } = \mathbb { E } _ { F } [ v ^ { j } ]$ as recovered on line 6 of Algorithm ${ \mathcal { B } } .$

Proof. Upon observing any signal s, the quality of the item chosen by the buyer is arg $\begin{array} { r } { \operatorname* { m a x } _ { q \in [ q , \overline { { q } } ] } q \overline { { v } } _ { \pi , F } ( s ) - p ( q ) } \end{array}$ First-order conditions dictate that $q _ { j } ^ { \prime } = \overline { { v } } _ { \pi , F } ( s )$ , thus we recover the posterior mean exactly. Now, in any round $n ,$ we have $\begin{array} { r } { \overline { { v } } _ { \pi , F } ( 0 ) = \frac { \mathbb { E } _ { F } [ v ^ { k } ] } { \mathbb { E } _ { F } [ v ^ { k - 1 } ] } } \end{array}$ and $\begin{array} { r } { \overline { { v } } _ { \pi , F } ( 1 ) = \frac { \mathbb { E } _ { F } [ v ] - \mathbb { E } _ { F } [ v ^ { k } ] } { 1 - \mathbb { E } _ { F } [ v ^ { k - 1 } ] } } \end{array}$ . Thus, inductively, we have that $m _ { j } = \mathbb { E } _ { F } [ v ^ { j } ]$ .

Given a suficient number of moments, Vandermonde inversion guarantees exact pmf recovery.

Proof of Theorem 5. Using Lemma 1 we can see that $\begin{array} { r } { \mathbb { E } _ { F } [ v ^ { j } ] = \sum _ { i = 1 } ^ { n + 1 } p _ { i } x _ { i } ^ { j } = V _ { j } ^ { \top } p , } \end{array}$ , where $V _ { j }$ is the $j ^ { t h }$ row of the Vandermonde matrix defined on line 8 of Algorithm 3. Hence, the true pmf satisfies $V p = m$ . Because the Vandermonde matrix is invertible as long as $x _ { i } \neq x _ { j }$ for all $i < j$ , the pmf reconstructed by the algorithm is exact, and no revenue is lost. □

## 5.2 Approximating the CDF for Continuous Distributions

The continuous case is trickier. We would like to approximate the CDF F via orthogonal polynomial families in the $L _ { \infty }$ sense, that is, in the Kolmogorov metric. We instead settle on the best $L _ { 2 }$ approximation, since those can be computed via inner products on the Lebesgue measure; the latter restriction rules out well-loved classes like the Chebyshev family (Cheney and Light, 2009). However, Legendre projections are known to have suboptimal $L _ { \infty }$ approximation guarantees, so we use de la Vallee Poussin mean of Legendre projections which in efect stabilizes the latter, and uses twice the number of moments to give near-optimal guarantees matching the best $L _ { \infty }$ approximation à la Jackon’s theorem (Mastroianni et al., 2008). A final hurdle is that moments of a distribution naturally correspond to inner products of the moment curve with the $\mathrm { p d f } ,$ and we would like to compute inner products with the CDF. This gap can be bridged by an application of integration by parts. We supply the details in the Appendix B.

$$
\begin{array} { r } { \int _ { 0 } ^ { 1 } x ^ { j } F ( x ) d x = 1 - \frac { 1 } { j + 1 } \int _ { 0 } ^ { 1 } x ^ { j + 1 } f ( x ) d x . } \end{array}
$$

Theorem 6. For any $r \in \mathbb { Z } _ { > 0 }$ and any continuous distribution F where the $r ^ { t h }$ derivative of the $C D F$ is bounded as $B ^ { r }$ , for some constant $B > 0$ , when Algorithm 3 is run for $\begin{array} { r } { n = \mathcal { O } \big ( \frac { \overline { { q } } ^ { 2 / r } K ^ { 2 / r } B } { \varepsilon ^ { 2 / r } } \big ) } \end{array}$ queries and suitable $\tau , \rho ,$ it produces a menu-signaling-scheme pair $\tilde { \pmb { x } } _ { \hat { F } }$ satisfying

$$
R e v ( \tilde { x } _ { \hat { F } } ; F ) \geq R e v ^ { \ast } ( F ) - \varepsilon .
$$

As an immediate corollary, for bounded and Lipschitz pdfs we need $1 / \varepsilon ^ { 2 }$ and $1 / \varepsilon$ queries, respectively. For analytic functions, we choose $r = \log ( 1 / \varepsilon )$ to get polylogarithmic in $1 / \varepsilon$ queries.

## 6 Regret under Joint Learning

Followed by Zu et al. (2025), we consider a more general setting where the seller interacts with a buyer on each day, commits to a menu and signaling scheme, and observes the purchase decision and the value of the buyer at the end of the day. More importantly, we allow the case when the population of buyers are learning, perhaps exogenously through reviews of the item on the platform. This setting thus subsumes the previous settings, however, the analysis closely mirrors the earlier one once we introduce the following relaxed behavioral assumption. Since the buyers are learning, the seller asks, what does the belief the buyer who arrives on that day hold? If the seller knew the belief of the buyer, he could solve the ofline problem. Since he does not know the buyer’s belief precisely, he robustifies the menu and signaling scheme which performs well for a set of beliefs, which the seller iteratively updates knowing that buyers are learning. This idea is captured in the following definition, where the seller confidently assumes that the arriving buyer’s belief on day t lies in the set $\mathcal { C } ( \hat { F } _ { t } )$ which can simply be $\{ F ^ { \star } \}$ , the unknown common belief, $B _ { 0 }$ the sellers prior belief about buyers’ belief or simply $\hat { F } _ { t } ^ { \phantom { \dagger } } .$ the empirical distribution on day t.

Algorithm 4: Online Empirical Menu & Information Design   
Input: $\tau _ { t } , \varepsilon _ { t } , h ( \tau _ { t } , \varepsilon _ { t } )$ for $t = 1 , \dots , T$   
1 Initialize $\hat { F } _ { 1 } \in B _ { 0 } ;$   
2 for $t = 1 , \dots , T$ do   
3 Call Algorithm 1 with the distribution $\hat { F } _ { t } ,$ cutof $\tau _ { t }$ and discount $\rho _ { t }$ to obtain $\tilde { \mathbfit { x } } _ { \hat { F } _ { t } }$   
4 Execute $\tilde { \mathbf { \ b { x } } } _ { \hat { F } _ { t } }$ and observe $v _ { t }$ and update $\begin{array} { r } { \hat { F } _ { t + 1 } ( v ) = \frac { 1 } { t } \sum _ { s = 1 } ^ { t } \mathbf { 1 } _ { \{ v _ { s } \leq v \} } ; } \end{array}$

Definition 3. We say that an algorithm which executes $\tilde { \mathbf { x } } _ { t }$ in each round is $( \beta , \kappa )$ -robustly revenue eficient, if there exists data-dependent sets $\mathcal { C } _ { t }$ such that we have (i) coverage i.e. $P r \left( \cap _ { t = 1 } ^ { T } \mathcal { C } _ { t } \ni F ^ { \star } \right) \geq 1 - \beta$ and (ii) robustness, i.e. for $\mathcal { H } _ { t }$ the set of all possible histories up to time t, we need for all $t \in [ T ]$ and for all $h _ { t } \in \mathcal { H } _ { t } ,$ conditioned on the history $h _ { t }$ that $\operatorname { s u p } _ { G \in { \mathcal { C } } ( { \hat { F } } _ { t } ) }$ max $\mathrm { ~ , ~ } R e v ( x ; \hat { F } _ { t } , \hat { F } _ { t } ) - R e v ( \tilde { x } _ { t } ; G , \hat { F } _ { t } ) \le \kappa _ { t }$

Assumption 1. For a $( \beta , \kappa )$ -robustly revenue eficient algorithm w.r.t. sets $\mathcal { C } _ { t } ,$ , the buyers’ prior $F _ { t } ^ { b }$ at round t is such that $P r ( \mathcal C _ { t } \ni F _ { t } ^ { b } ) = 1 \ \forall \ t \in [ T ]$ . The seller knows the set $B _ { 0 }$ s.t. $P r ( B _ { 0 } \ni F ^ { \star } ) = 1$

Although the Definition 3 allows for arbitrary sets ${ \mathcal { C } } ,$ one is interested in sets which are statistically achievable by the buyer. We propose Algorithm 4, where the seller chooses parameter sequence based on the assumed sets C. Throughout the following results we take the sets $\begin{array} { r } { \mathcal { C } _ { t } ( \varepsilon _ { t } ) : = \{ G \in \Delta ( \mathcal { V } ) : \operatorname* { s u p } _ { v } | G ( v ) - \hat { F } _ { t } ( v ) | \leq } \end{array}$ ${ { \varepsilon } _ { t } } \}$ for all $t \in [ T ]$ , which are $\varepsilon _ { t } > 0$ sized balls centered at $\hat { F } _ { t }$ in the Kolmogorov metric. First, we show the space parameters $\beta ,$ κ for which the algorithm is robustly revenue eficient.

Theorem 7. The Algorithm $\it 4$ with any cutof sequence $\{ \tau _ { t } \} _ { t = 1 } ^ { T }$ and for any sequence $\{ \varepsilon _ { t } \} _ { t = 1 } ^ { T }$ with $\varepsilon _ { t } \in ( 0 , 1 ]$ and discount sequence given by $\rho _ { t } ~ = ~ 4 \bar { q } \varepsilon _ { t } / \tau _ { t }$ is $( \beta , \kappa ) \ – r o b u s t l y$ revenue eficient for sets $\mathcal { C } _ { t } ( \varepsilon _ { t } )$ for all $\begin{array} { r } { \beta \geq \sum _ { t \in [ T ] } 2 \exp \left( - t \varepsilon _ { t } ^ { 2 } / 2 \right) } \end{array}$ and $\kappa _ { t } = K \bar { q } ( 4 \varepsilon _ { t } / \tau _ { t } + \tau _ { t } )$

We measure the performance of the algorithm through cumulative regret defined as Re $\begin{array} { r } { \mathbf { g } : = \sum _ { t = 1 } ^ { T } \mathrm { R e v } ^ { \star } ( F ^ { \star } ) - } \end{array}$ $\mathtt { r e v } _ { t } ( \pmb { x } _ { t } ; F _ { t } ^ { b } )$ where $\mathtt { r e v } _ { t } ( \pmb { x } _ { t } ; F _ { t } ^ { b } )$ denotes the revenue generated by seller on day t upon committing to $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ when the arriving buyer hold belief $F _ { t } ^ { b }$

Theorem 8. Algorithm $\it 4$ with cutof $\tau _ { t } = \sqrt { \varepsilon _ { t } }$ and discount $\rho _ { t } = 4 \bar { q } \sqrt { \varepsilon _ { t } }$ , for the sets $\mathcal { C } _ { t } ( \varepsilon _ { t } )$ with $\varepsilon _ { t } ~ =$ $\sqrt { 3 \log T / t }$ gives the cumulative regret $R e g \le \mathcal { O } \left( K \bar { q } ( \log T ) ^ { 1 / 4 } T ^ { 3 / 4 } \right)$ with probability at least $1 - 3 T ^ { - 1 / 2 }$

In Section $\mathrm { C } ,$ we prove the above results and also show that there are (more intricate) statistically plausible sets $\mathcal { C } _ { t }$ for which we obtain a $\tilde { \mathcal { O } } ( T ^ { 2 / 3 } )$ regret upper bound.

## 7 Conclusion

We study a novel learning problem motivated by the fact that sellers and algorithmic platforms today have access to vast user trails that they can use to forecast the personalized value of niche products for a specific user. We give learning algorithms that reflect how a seller might use past experience to craft the best menu of products and prices, along with a recommendation for individual customers on what products to buy. With access to samples of values, we achieved a sample complexity of ordert $1 / \varepsilon ^ { 3 }$ . We then showed that this can be improved significantly in a model where the seller can observe the buyers’ behaviors to carefully chosen menu and recommendations, under some circumstances making the sample complexity near-constant. Finally, we give an eficient algorithm to compute a near-optimal menu and signaling scheme, despite the underlying problem being non-convex. Our work brings a learning-theoretic perspective to the interface between information and (traditional) mechanism design.

Acknowledgements TP is supported by the Balas Fellowship Award and Ph.D. funding from the Tepper School of Business. We thank Rattana Pukdee, Kiriaki Fragkia and Andreas Kalavas for helpful comments.

## References

Agrawal, S., Feng, Y., and Tang, W. (2023). Dynamic pricing and learning with bayesian persuasion. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA. Curran Associates Inc.

Akerlof, G. A. (1970). The market for “lemons”: Quality uncertainty and the market mechanism. The Quarterly Journal of Economics, 84(3):488–500.

Anthony, M. and Bartlett, P. L. (1999). Neural Network Learning: Theoretical Foundations. Cambridge University Press.

Babichenko, Y., Talgam-Cohen, I., Xu, H., and Zabarnyi, K. (2022). Regret-minimizing bayesian persuasion. Games and Economic Behavior, 136:226–248.

Bacchiocchi, F., Bollini, M., Castiglioni, M., Marchesi, A., and Gatti, N. (2024). Online bayesian persuasion without a clue. In Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS ’24, Red Hook, NY, USA. Curran Associates Inc.

Balcan, M. F., Bernasconi, M., Castiglioni, M., Celli, A., Harris, K., and Wu, S. (2026a). Nearly-optimal bandit learning in stackelberg games with side information. In The Fourteenth International Conference on Learning Representations.

Balcan, M. F. and Beyhaghi, H. (2024). New guarantees for learning revenue maximizing menus of lotteries and two-part tarifs. Transactions on Machine Learning Research.

Balcan, M.-F., Blum, A., Hartline, J., and Mansour, Y. (2005). Mechanism design via machine learning. In 46th Annual IEEE Symposium on Foundations of Computer Science (FOCS’05), pages 605–614.

Balcan, M.-F., Blum, A., Hartline, J. D., and Mansour, Y. (2008). Reducing mechanism design to algorithm design via machine learning. Journal of Computer and System Sciences, 74(8):1245–1270.

Balcan, M. F., Fragkia, K., and Harris, K. (2026b). Learning in structured stackelberg games. In Forty-third International Conference on Machine Learning.

Balcan, M.-F., Sandholm, T., and Vitercik, E. (2016). Sample complexity of automated mechanism design. In Proceedings of the 30th International Conference on Neural Information Processing Systems, NIPS’16, page 2091–2099, Red Hook, NY, USA. Curran Associates Inc.

Balcan, M.-F., Sandholm, T., and Vitercik, E. (2025). Generalization guarantees for multi-item profit maximization: Pricing, auctions, and randomized mechanisms. Operations Research, 73(2):648–663.

Bartlett, P. L., Bousquet, O., and Mendelson, S. (2005). Local rademacher complexities. The Annals of Statistics, 33(4):1497–1537.

Bergemann, D., Heumann, T., and Morris, S. (2022). Screening with persuasion. arXiv preprint arXiv:2212.03360v1.

Bergemann, D., Heumann, T., and Morris, S. (2026a). Information design and mechanism design: An integrated framework. arXiv preprint arXiv:2601.17267.

Bergemann, D., Heumann, T., and Morris, S. (2026b). Screening with persuasion. Journal of Political Economy, 134(2):570–625.

Bergemann, D. and Morris, S. (2019). Information design: A unified perspective. Journal of Economic Literature, 57(1):44–95.

Blackwell, D. (1953). Equivalent comparisons of experiments. The Annals of Mathematical Statistics, 24(2):265–272.

Bolton, P. and Dewatripont, M. (2004). Contract Theory. Contract Theory. MIT Press.

Cai, Y. and Velegkas, G. (2021). How to Sell Information Optimally: An Algorithmic Study. In Lee, J. R., editor, 12th Innovations in Theoretical Computer Science Conference (ITCS 2021), volume 185 of Leibniz International Proceedings in Informatics (LIPIcs), pages 81:1–81:20, Dagstuhl, Germany. Schloss Dagstuhl – Leibniz-Zentrum für Informatik.

Castiglioni, M., Celli, A., Marchesi, A., and Gatti, N. (2020). Online bayesian persuasion. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA. Curran Associates Inc.

Castiglioni, M., Marchesi, A., Celli, A., and Gatti, N. (2021). Multi-receiver online bayesian persuasion. In Meila, M. and Zhang, T., editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 1314–1323. PMLR.

Cheney, E. and Light, W. (2009). A Course in Approximation Theory. Graduate studies in mathematics. American Mathematical Society.

Cole, R. and Roughgarden, T. (2014). The sample complexity of revenue maximization. In Proceedings of the Forty-Sixth Annual ACM Symposium on Theory of Computing, STOC ’14, page 243–252, New York, NY, USA. Association for Computing Machinery.

Crawford, V. P. and Sobel, J. (1982). Strategic information transmission. Econometrica, 50(6):1431–1451.

Dvoretzky, A., Kiefer, J., and Wolfowitz, J. (1956). Asymptotic minimax character of the sample distribution function and of the classical multinomial estimator. The Annals of Mathematical Statistics, 27(3):642–669.

Dworczak, P. and Martini, G. (2019). The simple economics of optimal persuasion. Journal of Political Economy, 127(5):1993–2048.

Gentzkow, M. and Kamenica, E. (2016). A rothschild-stiglitz approach to bayesian persuasion. American Economic Review, 106(5):597–601.

Goldberg, P. and Jerrum, M. (1993). Bounding the vapnik-chervonenkis dimension of concept classes parameterized by real numbers. In Proceedings of the Sixth Annual Conference on Computational Learning Theory, COLT ’93, page 361–369, New York, NY, USA. Association for Computing Machinery.

Guo, C., Huang, Z., and Zhang, X. (2019). Settling the sample complexity of single-parameter revenue maximization. In Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing, STOC 2019, page 662–673, New York, NY, USA. Association for Computing Machinery.

Haussler, D. (1992). Decision theoretic generalizations of the pac model for neural net and other learning applications. Information and Computation, 100(1):78–150.

Kamenica, E. (2019). Bayesian persuasion and information design. Annual Review of Economics, 11:pp. 249–272.

Kamenica, E. and Gentzkow, M. (2011). Bayesian persuasion. American Economic Review, 101(6):2590–2615.

Kleinberg, R. and Leighton, T. (2003). The value of knowing a demand curve: bounds on regret for online posted-price auctions. In 44th Annual IEEE Symposium on Foundations of Computer Science, 2003. Proceedings., pages 594–605.

Kravitz, D. (2025). Inside kroger’s secret shopper profiles: Why you may be paying more than your neighbors. Consumer Reports.

Li, A., Ravi, R., Singh, K., Yi, Z., and Zhang, W. (2025). How to sell high-dimensional data optimally. arXiv preprint arXiv:2510.15214.

Li, Y., Long, P. M., and Srinivasan, A. (2001). Improved bounds on the sample complexity of learning. Journal of Computer and System Sciences, 62(3):516–527.

Mansoor, S. (2026). Maryland becomes first state to ban surveillance pricing in grocery stores. The Guardian.

Maskin, E. and Riley, J. (1984). Monopoly with incomplete information. The RAND Journal of Economics, 15(2):171–196.

Massart, P. (1990). The tight constant in the dvoretzky-kiefer-wolfowitz inequality. The Annals of Probability, 18(3):1269–1283.

Mastroianni, G., Themistoclakis, W., et al. (2008). De la vallée poussin means and jackson’s theorem. Acta Scientiarum Mathematicarum, 74(1-2):147–170.

Maurer, A. and Pontil, M. (2009). Empirical bernstein bounds and sample-variance penalization. In COLT 2009 - The 22nd Conference on Learning Theory, Montreal, Quebec, Canada, June 18-21, 2009.

Mohri, M. and Medina, A. M. (2016). Learning algorithms for second-price auctions with reserve. Journal of Machine Learning Research, 17(74):1–25.

Morgenstern, J. and Roughgarden, T. (2016). Learning simple auctions. In Feldman, V., Rakhlin, A., and Shamir, O., editors, 29th Annual Conference on Learning Theory, volume 49 of Proceedings of Machine Learning Research, pages 1298–1318, Columbia University, New York, New York, USA. PMLR.

Mussa, M. and Rosen, S. (1978). Monopoly and product quality. Journal of Economic Theory, 18(2):301–317.

Roughgarden, T. and Schrijvers, O. (2016). Ironing in the dark. In Proceedings of the 2016 ACM Conference on Economics and Computation, EC ’16, page 1–18, New York, NY, USA. Association for Computing Machinery.

Spence, M. (1973). Job market signaling. The Quarterly Journal of Economics, 87(3):355–374.

Trefethen, L. N. (2019). Approximation Theory and Approximation Practice, Extended Edition. Society for Industrial and Applied Mathematics, Philadelphia, PA.

van der Vaart, A. and Wellner, J. (2023). Weak Convergence and Empirical Processes: With Applications to Statistics. Springer Series in Statistics. Springer.

Wilson, R. (1993). Nonlinear Pricing. Oxford University Press.

Zu, Y., Iyer, K., and Xu, H. (2025). Learning to persuade on the fly: Robustness against ignorance. Operations Research, 73(1):194–208.

## Appendix

In Section A, we complete the proof for sample complexity with value samples. Section B supplies the proofs for sample complexity of demand queries. In Section C and Section D, we complete the proofs for the joint-learning-based regret setting and prove the correctness of the FPTAS, respectively. Section E provides formal proofs for the signaling transformation discussed in Section 3.1.

## A Sample Complexity with Value Samples

## A.1 Proof of Theorem 2

We decompose the error as

$$
R e v ^ { * } ( F ) - R e v ( \tilde { x } _ { \hat { F } } ; F ) = \underbrace { R e v ^ { * } ( F ) - R e v ^ { * } ( \hat { F } ) } _ { \mathrm { T e r m 1 } } + \underbrace { R e v ^ { * } ( \hat { F } ) - R e v ( \tilde { x } _ { \hat { F } } ; F , F ) } _ { \mathrm { T e r m 2 } } .
$$

Throughout the analysis, we condition on the event in Lemma 2, which holds with high probability.

Lemma 2. There exists a universal constant C such that for any $\delta > 0 , n > 1$ and distribution F, with probability $1 - \delta _ { i }$ , for all monotone paritional signaling scheme π with K signals, we have that

$$
\begin{array} { r l } & { \left| \underset { \pi \circ F } { \operatorname* { P r } } ( k ) - \underset { \pi \circ \hat { F } } { \operatorname* { P r } } ( k ) \right| \leq C \left( \sqrt { \operatorname* { m i n } \{ \underset { \pi \circ F } { \operatorname* { P r } } ( k ) , \underset { \pi \circ \hat { F } } { \operatorname* { P r } } ( k ) \} } \frac { \log ( n / \delta ) } { n } + \frac { ( \log n / \delta ) } { n } \right) , } \\ & { \left| \mathbb { E } _ { F } [ v | k ] - \mathbb { E } _ { \widehat { F } } [ v | k ] \right| \leq C \left( \sqrt { \frac { \log \left( n / \delta \right) } { n \operatorname* { m a x } \{ \operatorname* { P r } _ { \pi \circ F } ( k ) , \mathrm { P r } _ { \pi \circ \widehat { F } } ( k ) \} } } + \frac { \log \left( n / \delta \right) } { n \operatorname* { m a x } \{ \mathrm { P r } _ { \pi \circ F } ( k ) , \mathrm { P r } _ { \pi \circ \widehat { F } } ( k ) \} } \right) , } \end{array}\tag{2}
$$

for all $k \in [ K ]$ , where $\widehat F$ is a n-point empirical distribution sampled from F.

First, we upper bound Term 1. Let $\pmb { x } _ { F }$ be the optimal pair of menu $( ( q _ { k } , p _ { k } ) : k \in [ K ] )$ ) and monotone partitional signaling scheme $( t _ { k } : k \in [ K ] )$ under $F .$ . Let $z _ { \tau } = \{ k \in [ K ] : t _ { k } - t _ { k - 1 } < \tau \}$ . Define $\tilde { { \bf x } } _ { F }$ with signaling scheme π˜ modified so that all signals in $z _ { \tau }$ are mapped to a null signal:

$$
{ \tilde { \pi } } ( s \mid v ) = { \left\{ \begin{array} { l l } { \pi ( s \mid v ) } & { { \mathrm { i f ~ } } s \not \in z _ { \tau } , } \\ { \sum _ { k \in z _ { \tau } } \pi ( k \mid v ) } & { { \mathrm { i f ~ } } s = { \mathrm { n u l l } } , } \end{array} \right. }
$$

and the prices discounted as $\tilde { p } _ { k } = p _ { k } - k \rho$ . Then we have

$$
\begin{array} { r l } & { \mathtt { T e r m 1 } : = R e v ^ { * } ( F ) - R e v ( \tilde { x } _ { F } ; \hat { F } , \hat { F } ) + R e v ( \tilde { x } _ { F } ; \hat { F } , \hat { F } ) - R e v ^ { * } ( \hat { F } ) } \\ & { \qquad \le R e v ^ { * } ( F ) - R e v ( \tilde { x } _ { F } ; \hat { F } , \hat { F } ) } \end{array}
$$

where we appeal to the fact that $R e v ^ { * } ( \hat { F } )$ is the optimal revenue for the empirical distribution ${ \widehat F } .$ . Now, we use Lemma 3, which places an upper bound of ${ \tilde { \mathcal { O } } } ( { \overline { { q } } } K / n ^ { 1 / 3 } )$ on Term 1, by choosing $\rho = \widetilde { \Theta } ( \overline { { q } } / n ^ { 1 / 3 } )$ and $\tau = \widetilde { \Theta } ( 1 / n ^ { 1 / 3 } )$

Lemma 3. Consider any two value distributions $F , { \widehat { F } }$ such that for all monotone partitional signaling schemes π with K signals, we have for all $k \in [ K ]$ that

$$
\left| \operatorname* { P r } _ { \pi \circ F } ( k ) - \operatorname* { P r } _ { \pi \circ \hat { F } } ( k ) \right| \leq { \varepsilon _ { 0 } } , \ a n d \ \left| \mathbb { E } _ { F } [ v | k ] - \mathbb { E } _ { \hat { F } } [ v | k ] \right| \leq \frac { { \varepsilon _ { 1 } } } { \operatorname* { P r } _ { \pi \circ F } ( k ) } + \frac { { \varepsilon _ { 2 } } } { \sqrt { \operatorname* { P r } _ { \pi \circ F } ( k ) } } .
$$

Let $\scriptstyle { \mathbf { \boldsymbol { x } } } _ { F }$ be $\varepsilon _ { O P T }$ -suboptimal solution for distribution F comprising of an incentive-compatible menu and monotone partitional signaling scheme, for which $p _ { k } - c ( q _ { k } )$ is in non-decreasing in k. Further, let $\tilde { \boldsymbol { x } } _ { F }$ be a

modification of $\scriptstyle { \mathbf { \boldsymbol { x } } } _ { F }$ with all signals with marginal probability smaller than τ (with respect to $F )$ mapped to a null signal and price modifications $\tilde { p } _ { k } = p _ { k } - k \rho$ , where $\rho = 2 \overline { { q } } ( \varepsilon _ { 1 } / \tau + \varepsilon _ { 2 } / \sqrt { \tau } )$ $p _ { k }$ is the corresponding price for signal k in $\scriptstyle { \mathbf { \boldsymbol { x } } } _ { F }$ . Then, we have

$$
R e v ( { \pmb x } _ { F } ; F , F ) - R e v ( \tilde { { \pmb x } } _ { F } , \widehat { F } , \widehat { F } ) \leq \overline { { q } } K ( \tau + 2 ( \varepsilon _ { 1 } / \tau + \varepsilon _ { 2 } / \sqrt { \tau } ) + \varepsilon _ { 0 } ) .
$$

Similarly, for Term2, let $\boldsymbol { x } _ { \widehat { F } }$ be an $\varepsilon _ { O P T ^ { - \mathrm { { s u b o p t i m a l } } } }$ pair of menu $( ( q _ { k } , p _ { k } ) : k \in [ K ] )$ and monotone partitional signaling scheme $( t _ { k } : k \in [ K ] )$ this time under ${ \widehat { F } } .$ . Let $\tilde { z } _ { \tau } = \{ k \in [ K ] : t _ { k } - t _ { k - 1 } < \tau \}$ . Let $\tilde { \pmb { x } } _ { \widehat { F } }$ be the analogous modification where all signals in $\tilde { z } _ { \tau }$ are mapped to a null signal, and prices are linear-additively discounted by $\rho$ at all indices. Now, applying Lemma 3, with the roles of $F$ and $\hat { F }$ reversed, we get

$$
\begin{array} { r l } & { \mathtt { T e r m 2 : } = R e v ^ { * } ( \widehat { F } ) - R e v ^ { * } ( x _ { \widehat { F } } ; \widehat { F } , \widehat { F } ) + R e v ( x _ { \widehat { F } } ; \widehat { F } , \widehat { F } ) - R e v ^ { * } ( { \tilde { x } _ { \widehat { F } } } ; F , F ) } \\ & { \qquad \le \varepsilon _ { \mathrm { O P T } } + R e v ( x _ { \widehat { F } } ; \widehat { F } , \widehat { F } ) - R e v ( { \tilde { x } _ { F } } ; \widehat { F } , \widehat { F } ) } \\ & { \qquad \le \varepsilon _ { \mathrm { O P T } } + \tilde { \cal O } ( \overline { { q } } K / n ^ { 1 / 3 } ) . } \end{array}
$$

## A.2 Proof of Lemma 3

Since $\scriptstyle { \mathbf { \boldsymbol { x } } } _ { F }$ encodes an incentive-compatible menu $( ( q _ { k } , p _ { k } ) : k \in [ K ] )$ under the distribution $F ,$ we know that for all $k \in [ K ]$ that

$$
\begin{array} { r l } & { q _ { k } \cdot \mathbb { E } _ { \hat { F } } [ v | k ] - p _ { k } \geq q _ { k } \cdot \mathbb { E } _ { F } [ v | k ] - p _ { k } - \overline { { q } } \left( \frac { \varepsilon _ { 1 } } { \operatorname* { P r } _ { \pi \circ F } ( k ) } + \frac { \varepsilon _ { 2 } } { \sqrt { \operatorname* { P r } _ { \pi \circ F } ( k ) } } \right) } \\ & { \qquad \geq \operatorname* { m a x } \{ 0 , \operatorname* { m a x } \{ q _ { j } \cdot \mathbb { E } _ { F } [ v | k ] - p _ { j } \} \} - \overline { { q } } \left( \frac { \varepsilon _ { 1 } } { \operatorname* { P r } _ { \pi \circ F } ( k ) } + \frac { \varepsilon _ { 2 } } { \sqrt { \operatorname* { P r } _ { \pi \circ F } ( k ) } } \right) } \\ & { \qquad \geq \operatorname* { m a x } \{ 0 , \operatorname* { m a x } \{ q _ { j } \cdot \mathbb { E } _ { \hat { F } } [ v | k ] - p _ { j } \} \} - 2 \overline { { q } } \left( \frac { \varepsilon _ { 1 } } { \operatorname* { P r } _ { \pi \circ F } ( k ) } + \frac { \varepsilon _ { 2 } } { \sqrt { \operatorname* { P r } _ { \pi \circ F } ( k ) } } \right) } \end{array}
$$

Now by implementing the modified prices $\begin{array} { r } { \tilde { p } _ { k } = p _ { k } - k \rho . } \end{array}$ , it is clear that participation constraints are upheld for $\widehat F$ for all (good) signals in not $z _ { \tau }$ . We do not claim to uphold incentive compatibility under ${ \widehat { F } } ,$ only that any buyer receiving signal k does not purchase an item indexed by a lower signal, as long as $k \notin z _ { \tau }$ . To observe this, note for all $j < k$ that

$$
\begin{array} { r l } & { q _ { k } \cdot \mathbb { E } _ { \hat { F } } [ v | k ] - \widetilde { p } _ { k } = q _ { k } \cdot \mathbb { E } _ { \hat { F } } [ v | k ] - p _ { k } + k \rho } \\ & { \qquad \geq q _ { j } \cdot \mathbb { E } _ { \hat { F } } [ v | k ] - p _ { j } + k \rho - \rho } \\ & { \qquad = q _ { j } \cdot \mathbb { E } _ { \hat { F } } [ v | k ] - \widetilde { p } _ { j } + ( k - j ) \rho - \rho } \\ & { \qquad \geq q _ { j } \cdot \mathbb { E } _ { \hat { F } } [ v | k ] - \widetilde { p } _ { j } . } \end{array}
$$

Thus, each receipt of a good signal $k \notin z _ { \tau }$ agrees to pay at least $p _ { k } - K \rho$ and generates a margin at least $p _ { k } - c ( q _ { k } ) - K \rho$ for the buyer under the distribution $\widehat F .$ . Thus, we have

$$
R e v ( { \pmb x } _ { F } ; F , F ) - R e v ( { \tilde { \pmb x } } _ { F } , \widehat F , F ) \leq K ( \rho + \overline { { q } } \tau ) ,
$$

even assuming that in the worst case those assigned the null signal refuse to purchase anything. Finally, using the observations that the marginal probability of signals is similar under $F$ and ${ \widehat { F } } ;$

$$
R e v ( \tilde { x } _ { F } ; \widehat { F } , F ) - R e v ( \tilde { x } _ { F } , \widehat { F } , \widehat { F } ) \le \overline { q } K \varepsilon _ { 0 } .
$$

Combining the last two inequalities concludes the proof.

## A.3 Proof of Lemma 2

Our plan is to appeal to variance-sensitive versions of uniform convergence bounds for function classes with bounded pseudo-dimension. We start with a result of Maurer and Pontil (2009), which is stated in terms of $L _ { \infty }$ covering number on n-sample distributions, a term we will quickly rid ourselves of. (See van der Vaart and Wellner (2023) for the definition.)

Theorem 9 (Maurer and Pontil (2009)). Let X be a random variable with values in a set X with distribution F and let F be a class of hypotheses $f : \mathcal { X } \to [ 0 , 1 ]$ . Fix $\delta \in ( 0 , 1 ) , n \geq 1 6$ and set

$$
\mathcal { M } ( n ) = 1 0 \mathcal { N } _ { \infty } ( 1 / n , \mathcal { F } , 2 n ) .
$$

Then with probability at least $1 - \delta$ in the random vector $\mathbf { X } = ( X _ { 1 } , \ldots , X _ { n } ) \sim \mu ^ { n }$ we have

$$
\mathbb { E } _ { \boldsymbol { x } \sim F } [ f ( \boldsymbol { x } ) ] - \mathbb { E } _ { \boldsymbol { x } \sim \hat { \boldsymbol { F } } } [ f ( \boldsymbol { x } ) ] \le \sqrt { \frac { 1 8 V _ { n } ( f , \mathbf { X } ) \ln ( \mathcal { M } ( n ) / \delta ) } { n } } + \frac { 1 5 \ln ( \mathcal { M } ( n ) / \delta ) } { n - 1 } , \forall f \in \mathcal { F } ,
$$

where $\begin{array} { r } { V _ { n } ( f , \mathbf { X } ) = \sum _ { i < j } ( f ( x _ { i } ) - f ( x _ { j } ) ) ^ { 2 } / n ( n - 1 ) } \end{array}$ is the sample variance, and $\widehat F$ is the empirical distribution on the dataset X.

For real-valued function classes with bounded pseudo-dimension, the above statement can be stated in terms of pseudo-dimension.

Definition 4. A function class $\mathcal { F } : \mathcal { X }  \mathbb { R }$ has pseudo-dimension d if d is the maximum number for which there is d-sized set $S = \{ x _ { 1 } , \ldots , x _ { d } \} \subseteq \mathcal { X }$ and real numbers $r _ { 1 } , \ldots , r _ { d }$ such that for each $b \in \{ 0 , 1 \} ^ { d }$ there is a function $f _ { b } \in \mathcal { F }$ with sign $\left( f _ { b } ( x _ { i } ) - r _ { i } \right) = b _ { i } \forall i \in [ d ]$

Theorem 10 (Anthony and Bartlett (1999)). Let F be a set of real functions from a domain X to the bounded interval $[ 0 , B ]$ . Let $\epsilon > 0$ and suppose that the pseudo-dimension of F is d. Then

$$
\mathcal { N } _ { \infty } ( \epsilon , F , m ) \leq \sum _ { i = 1 } ^ { d } { \binom { m } { i } } \left( \frac { B } { \epsilon } \right) ^ { i } ,
$$

which is less than $( e m B / ( \epsilon d ) ) ^ { d } ~ f o r ~ m \ge d .$

Now, we perform three steps at once: one, we compose the last two results; two, we apply the results to the class $\mathcal { F } ^ { \prime } = \{ 1 - f : f \in \mathcal { F } \}$ in addition to the original class; three, we note for any $f : \mathcal { X } \to [ 0 , 1 ]$ that $\begin{array} { r } { V _ { n } ( f , \mathbf { X } ) = \frac { n } { n - 1 } ( \mathbb { E } _ { { x } \sim \widehat { F } } [ { \dot { f } } ( x ) ^ { 2 } ] - \mathbb { E } _ { { x } \sim \widehat { F } } [ f ( x ) ] ^ { 2 } ) \leq 2 \mathbb { E } _ { { x } \sim \widehat { F } } [ f ( x ) ] } \end{array}$ ]. As a result, we get that there is a universal constant C such that for any function class $\mathcal { F } : \mathcal { X }  [ 0 , 1 ]$ with pseudo-dimension $d ,$ with probability $1 - \delta .$ we have for all $f \in { \mathcal { F } }$ that

$$
\left| \mathbb { E } _ { F } [ f ( x ) ] - \mathbb { E } _ { \widehat { F } } [ f ( x ) ] \right| \leq C \left( \sqrt { \frac { \mathbb { E } _ { \widehat { F } } [ f ( x ) ] d \log ( n / \delta ) } { n } } + \frac { d \log ( n / \delta ) } { n } \right) .
$$

Let $\Delta : = | \mathbb { E } _ { \hat { F } } [ f ( x ) - \mathbb { E } _ { F } [ f ( x ) ] |$ | and $b : = d \log ( n / \delta ) / n$ The above claim can be restated as $\Delta \ \leq$ $C \sqrt { ( \mathbb { E } _ { F } [ f ( x ) ] + \Delta ) b } + C ^ { \prime } b$ for some constants $C , C ^ { \prime }$ . Using ${ \sqrt { a + b } } \leq { \sqrt { a } } + { \sqrt { b } }$ and then applying AM-GM as $\begin{array} { r } { C \sqrt { \Delta b } \leq \frac { 1 } { 2 } \Delta + \frac { C ^ { 2 } } { 2 } b } \end{array}$ gives us $\Delta \le 2 C \sqrt { \mathbb { E } _ { F } [ f ( x ) ] \cdot b } + ( C ^ { 2 } + 2 C ^ { \prime } ) b$ . Thus, whenever the above inequality holds, we also have for a diferent universal constant $C ^ { \prime }$ that

$$
\left| \mathbb { E } _ { F } [ f ( x ) ] - \mathbb { E } _ { \widehat { F } } [ f ( x ) ] \right| \leq C ^ { \prime } \left( \sqrt { \frac { \mathbb { E } _ { F } [ f ( x ) ] d \log ( n / \delta ) } { n } } + \frac { d \log ( n / \delta ) } { n } \right) .
$$

Let G be the class of function on [0, 1] of the form $g _ { u , v , \xi _ { u } , \xi _ { v } } ( x ) : = \xi _ { u } \mathbf { 1 } \{ x = u \} + \mathbf { 1 } \{ u < x < v \} + \xi _ { v } \mathbf { 1 } \{ x = v \}$ where $0 \leq \xi _ { u } , \xi _ { v } \leq 1 , \ u \leq v \in [ 0 , 1 ]$ . Let ${ \mathcal { F } } = \{ x g ( x ) : g \in { \mathcal { G } } \}$ . We wish to bound the pseudo-dimension of these classes. Recall that for any class ${ \mathcal F } ,$ the pseudo-dimension is precisely the VC dimension of $\{ ( x , t ) \mapsto \mathbf { 1 } \{ f ( x ) \geq t \} : f \in { \mathcal { F } } \}$ . We use the following result from Goldberg and Jerrum (1993) on VC dimension of classes involving real numbers to conclude that the pseudo-dimension of both classes is at most a constant, by observing that membership for the subgraph sets of such function classes can be computed by a constant sized algebraic circuit that permits usual arithmetric operations and (in)equalities.

Theorem 11 (Goldberg and Jerrum (1993)). Let $\{ { \mathcal { C } } _ { k , n } : k , n \in \mathbb { Z } _ { > 0 } \}$ be a family of concept classes where concepts in $\mathcal { C } _ { k , n }$ and inputs are represented by k and n real values, respectively. Further, let the test for membership of an instance x in a concept C in $\mathcal { C } _ { k , n }$ consist of an algorithm $\mathcal { A } _ { k , n }$ taking $k + n$ real inputs representing $C$ and $x ,$ whose runtime is $t = t ( k , n )$ , and which returns the truth value $a \in C$ . The algorithm $\mathcal { A } _ { k , n }$ is allowed to perform conditional jumps (conditioned on equality and inequality of real values) and execute the standard arithmetic operations on real numbers $( + , - , \times , / )$ in constant time. Then the $V C$ dimension of $\mathcal { C } _ { k , n }$ is at most $O ( k t )$

Note that for any monotone partitional signaling scheme $\pi ,$ as detailed in Section 3.1, $\operatorname* { P r } _ { \pi \circ F } [ k ]$ and $\mathbb { E } _ { \pi \circ F } [ v \mathbf { 1 }$ {signal k is observed}] can be realized as expectations of specific members in $\mathcal { G }$ and ${ \mathcal F } .$ , respectively. Thus, we already have with high probability for a suitable universal constant $C ^ { \prime \prime }$ that

$$
\begin{array} { r l } & { \operatorname* { m a x } \left\{ \left| \underset { \pi \circ F } { \operatorname* { P r } } ( k ) - \underset { \pi \circ \hat { F } } { \operatorname* { P r } } ( k ) \right| , \left| \mathbb { E } _ { \pi \circ F } [ v \mathbf { 1 } \{ k \mathrm { ~ i s ~ o b s e r v e d } \} ] - \underset { \pi \circ \hat { F } } { \operatorname* { P r } } [ v \mathbf { 1 } \{ k \mathrm { ~ i s ~ o b s e r v e d } \} ] \right| \right\} } \\ & { \leq C ^ { \prime \prime } \left( \sqrt { \operatorname* { m i n } \{ \underset { \pi \circ F } { \operatorname* { P r } } ( k ) , \underset { \pi \circ \hat { F } } { \operatorname* { P r } } ( k ) \} } \frac { \log \left( n / \delta \right) } { n } + \frac { \log \left( n / \delta \right) } { n } \right) . } \end{array}
$$

To conclude the statement concerning the conditional means, we observe the following elementary inequality for positive reals: $\begin{array} { r } { \left| \frac { A ^ { \prime } } { B ^ { \prime } } - \frac { A } { B } \right| \le \frac { | A ^ { \prime } - A | } { B } + \frac { A ^ { \prime } } { B ^ { \prime } } \frac { | B ^ { \prime } - B | } { B } } \end{array}$ and substitute the appropriate quantities from the last inequality, while noting that the condition mean, i.e., $A ^ { \prime } / B ^ { \prime }$ , can at most be one.

## A.4 Ensuring Monotonic Margins

Let us consider that we are given a monotone partitional scheme $( t _ { k } : k \in [ K ] )$ and an associated incentive compatible direct menu $( ( q _ { k } , p _ { k } ) : k \in [ K ] )$ for a value distribution $\widehat F$ , which might not be revenue maximizing pair. A key condition used in previous proofs are that the seller’s margins $p _ { k } - c ( q _ { k } )$ are non-decreasing in $k .$ This is true by default for optimal incentive-compatible direct menus with respect to ${ \widehat { F } } ;$ our approach here will give an alternative proof of this. However, if we use the approximately optimal menu from Section 4, this may not be true. To placate this worry, we now provide an algorithm that given an incentive-compatible menu converts it into another incentive compatible menu where margins are non-decreasing in the index, while weakly increasing the revenue. Thus, this condition can always be ensured algorithmically.

The modification works as follows: first, we calculate the margins $\pi _ { k } = p _ { k } - c ( q _ { k } )$ for all signals. Next, we only retain menu items in the set $S = \{ k : \pi _ { k } \geq \operatorname* { m a x } _ { j < k } \pi _ { j } \}$ and delete the rest. This results in an indirect menu. $\mathrm { S o }$ , we calculate what a buyer observing signal k purchases in this smaller menu (with respect to the prior $\widehat { F } )$ and designate that as their quality-item pair in a new direct menu.

Proposition 1. Given any incentive-compatible direct menu, the above procedure produces another incentive-compatible direct menu with non-decreasing margins. In addition, during this transformation, the revenue weakly increases.

Proof. The incentive compatibility holds because of the relabeling in the last step. Notice that the item corresponding to smallest k is retained by definition, and hence participation for all (signal) types is upheld. Let $\sigma ( k )$ be the item a buyer observing signal k picks in this smaller menu S. It is well known incentive-compatibility for one-dimensional (signal) types ensures that $q _ { \sigma ( k ) }$ is non-decreasing (Maskin and Riley, 1984), and hence, by construction, $\pi _ { \sigma ( k ) }$ follows the same pattern. As for preserving the revenue, this is immediate for any signal k for which k belongs to $S ,$ because a buyer receiving this signal will still pick the same item, which in turn has not been deleted. If k is not in $S ,$ let $k ^ { - }$ be the largest index picked in $S$ that is smaller than $k ,$ and $k ^ { + }$ be the smallest index picked in S that is larger than k. Since incentive-compatibility constraints only enforce on neighboring types even for suboptimal menus (Maskin and Riley, 1984), $\sigma ( k )$ is either $k ^ { - } \mathrm { ~ o r ~ } k ^ { + }$ . Because $k$ is not $S ,$ we know $\pi _ { k ^ { - } } \geq \pi _ { k }$ , and similarly, $\pi _ { k ^ { + } } \geq \pi _ { k }$ . So, without knowing the precise details of what signal k picks, we know that the margin (and hence the net revenue) is weakly greater. □

## A.5 Proof of Theorem 3

We consider $\mathcal { X } _ { \mathbf { s } } = \{ ( ( I , \pmb { w } , \pmb { \xi } ) , ( p , q ) )$ with $I = 2$ and thus $w = \{ 0 = w _ { 0 } \leq w _ { 1 } \leq w _ { 2 } = 1 \} \} , \xi = \{ 0 = \xi _ { 0 } \leq$ $\xi _ { 1 } \le \xi _ { 2 } = 1 \} \}$ . Thus we can equivalently write $\mathcal { X } _ { \mathsf { s } } = \{ ( w _ { 1 } , \xi _ { 1 } , ( p , q ) ) \}$ . Here the information structure simply consists of two signals $s _ { \mathrm { H } }$ and $s _ { \mathrm { L } }$ where

$$
\pi ( s _ { \mathrm { { L } } } | v ) = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } v < w _ { 1 } } \\ { \xi _ { 1 } } & { { \mathrm { i f ~ } } v = w _ { 1 } } \end{array} \right. } , \quad \pi ( s _ { \mathrm { { H } } } | v ) = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } v > w _ { 1 } } \\ { 1 - \xi _ { 1 } } & { { \mathrm { i f ~ } } v = w _ { 1 } } \end{array} \right. }
$$

Let $\mu _ { F } ( w _ { 1 } , \xi _ { 1 } )$ be the posterior mean under signal $s _ { \mathrm { H } } .$ , which is explicitly given as

$$
\mu _ { F } ( w _ { 1 } , \xi _ { 1 } ) = \frac { \mathbb { E } _ { F } [ ( 1 - \xi _ { 1 } ) v \mathbf { 1 } \{ v = w _ { 1 } \} + v \mathbf { 1 } \{ v > w _ { 1 } \} ] } { \mathbb { E } _ { F } [ ( 1 - \xi _ { 1 } ) \mathbf { 1 } \{ v = w _ { 1 } \} + \mathbf { 1 } \{ v > w _ { 1 } \} ] } .
$$

The optimization problem can be written as

$$
\mathbf { L } _ { F } ^ { ( 1 ) } : \operatorname* { m a x } _ { p \in \mathbb { R } , q \in [ q , \bar { q } ] } \underbrace { \mathbb { E } _ { F } [ ( 1 - \xi _ { 1 } ) \mathbf { 1 } \{ v = w _ { 1 } \} } _ { \mathrm { w } _ { 1 } , \xi _ { 1 } \in [ 0 , 1 ] \mathrm { ~ p r o b a b i l i t y ~ o f ~ r e a l i z i n g ~ s i g n a l ~ } s _ { \mathrm { R } } : \mathrm { ~ P r } ( s _ { \mathrm { R } } | w _ { 1 } , \xi _ { 1 } ) } \cdot \underbrace { ( p - c ( q ) ) } _ { \mathrm { r e v e n u e } } \cdot \underbrace { \mathbb { I } \{ \mu _ { F } ( w _ { 1 } , \xi _ { 1 } ) \cdot q - p \geq 0 \} } _ { \mathrm { p a r t i c i p a t i o n ~ o f ~ t h e ~ b u y e r } } .
$$

Given N samples $v _ { 1 } , \ldots , v _ { N } \stackrel { \mathrm { i i d } } { \sim } F .$ , let $\textstyle { \hat { F } } ( v ) = | \{ v _ { i } : v _ { i } \leq v \} | / n = \sum _ { i = 1 } ^ { n } \mathbb { I } \{ v _ { i } \leq v \} / n$ and let $( w , \xi , ( p , q ) )$ and $( \hat { w } , \hat { \xi } , ( \hat { p } , \hat { q } ) )$ be the solution of $\mathtt { L } _ { F } ^ { ( 1 ) }$ and $\mathtt { L } _ { \hat { F } } ^ { ( 1 ) }$ respectively.

Define

$$
\mathcal { G } = \left\{ g _ { w , \xi } ( v ) = ( 1 - \xi ) \mathbf { 1 } \{ v = w \} + \mathbf { 1 } \{ v > w \} \ \forall \xi \in [ 0 , 1 ] , \ w , v \in [ 0 , 1 ] \right\}
$$

We have with probability at least $1 - \delta$ , the following from the analysis of section A.3

$$
\operatorname* { s u p } _ { \xi \in [ 0 , 1 ] , w \in [ 0 , 1 ] } \left| \mathbb { E } _ { v \sim F } [ g _ { w , \xi } ( v ) ] - \mathbb { E } _ { v \sim \hat { F } } [ g _ { w , \xi } ( v ) ] \leq \varepsilon \right.
$$

for $\begin{array} { r } { \varepsilon = C \sqrt { \frac { 1 } { 2 n } \ln \left( \frac { 1 } { \delta } \right) } } \end{array}$ for some constant $C > 0$ . Using this, a simple application of triangle inequality gives for all $\xi \in [ 0 , 1 ]$ and $w \in [ 0 , 1 ]$ with probability at least $1 - \delta$ , the following

$$
| \mu _ { F } ( w , \xi ) - \mu _ { \hat { F } } ( w , \xi ) | \leq \operatorname* { m i n } \{ \rho _ { w , \xi } , \hat { \rho } _ { w , \xi } \} ,
$$

where $\hat { \rho } _ { w , \xi } = \varepsilon / \mathbb { E } _ { v \sim \hat { F } } [ g _ { w , \xi } ( v ) ]$ and $\rho _ { w , \xi } = \varepsilon / \mathbb { E } _ { v \sim F } [ g _ { w , \xi } ( v ) ]$

Thus, $\mu _ { \hat { F } } ( \hat { w } , \hat { \xi } ) \cdot \hat { q } - \hat { p } \geq 0$ implies $\mu _ { F } ( \hat { w } , \hat { \xi } ) \cdot \hat { q } - \hat { p } ^ { \prime } \geq 0$ where $\hat { p } ^ { \prime } = \hat { p } - \hat { \rho } _ { \hat { w } , \hat { \xi } } \overline { { q } }$

Similarly, $\mu _ { F } ( w , \xi ) \cdot q - p \geq 0$ implies $\mu _ { \hat { F } } ( w , \xi ) \cdot q - p ^ { \prime } \ge 0$ for $p ^ { \prime } = p - \rho _ { w , \xi } \overline { { q } }$

This simply means that $( w , \xi , ( p ^ { \prime } , q ) )$ satisfies the IR conditions under F<sup>ˆ</sup> and similarly $( \hat { w } , \hat { \xi } , ( \hat { p } ^ { \prime } , \hat { q } ) )$ satisfies the IR conditions under F. Now, observe that

$$
\begin{array} { r l } & { \mathtt { R e v } ( \hat { w } , \hat { \xi } , ( \hat { p } ^ { \prime } , \hat { q } ) ; F ) = \mathbb { E } _ { v \sim F } [ g _ { \hat { w } , \hat { \xi } } ( v ) ] \cdot ( \hat { p } ^ { \prime } - c ( \hat { q } ) ) } \\ & { \qquad \ge \left( \mathbb { E } _ { v \sim \hat { F } } [ g _ { \hat { w } , \hat { \xi } } ( v ) ] - \varepsilon \right) \cdot \Big ( \hat { p } - \rho _ { \hat { w } , \hat { \xi } } \overline { { q } } - c ( \hat { q } ) \Big ) } \\ & { \qquad \ge \left( \mathbb { E } _ { v \sim \hat { F } } [ g _ { \hat { w } , \hat { \xi } } ( v ) ] \right) \cdot ( \hat { p } - c ( \hat { q } ) ) - \mathbb { E } _ { v \sim \hat { F } } [ g _ { \hat { w } , \hat { \xi } } ( v ) ] \cdot \rho _ { \hat { w } , \hat { \xi } } \overline { { q } } - \varepsilon \overline { { q } } } \end{array}
$$

$$
= \mathsf { R e v } ( \hat { w } , \hat { \xi } , ( \hat { p } , \hat { q } ) ; \hat { F } ) - 2 \varepsilon \overline { { q } }
$$

Using the feasibility of $( w , \xi , ( p ^ { \prime } , q ) )$ for $\mathtt { L } _ { \hat { F } }$ , we get

$$
\begin{array} { r l } & { \mathrm { R e v } ( \hat { w } , \hat { \xi } , ( \hat { p } ^ { \prime } , \hat { q } ) ; F ) \ge \mathrm { R e v } ( \hat { w } , \hat { \xi } , ( \hat { p } , \hat { q } ) ; \hat { F } ) - 2 \varepsilon \overline { { q } } } \\ & { \qquad \ge \mathrm { R e v } ( w , \xi , ( p ^ { \prime } , q ) ; \hat { F } ) - 2 \varepsilon \overline { { q } } } \\ & { \qquad = \mathbb { E } _ { v \sim \hat { F } } [ g _ { w , \xi } ( v ) ] \cdot ( p ^ { \prime } - c ( q ) ) - 2 \varepsilon \overline { { q } } } \\ & { \qquad \ge ( \mathbb { E } _ { v \sim F } [ g _ { w , \xi } ( v ) ] - \varepsilon ) \cdot ( p - \rho _ { w , \xi } \overline { { q } } - c ( q ) ) - 2 \varepsilon \overline { { q } } } \\ & { \qquad \ge \mathbb { E } _ { v \sim F } [ g _ { w , \xi } ( v ) ] \cdot ( p - c ( q ) ) - \varepsilon \overline { { q } } - \mathbb { E } _ { v \sim F } [ g _ { w , \xi } ( v ) ] \cdot \rho _ { w , \xi } \overline { { q } } - 2 \varepsilon \overline { { q } } } \\ & { \qquad \ge \mathrm { R e v } ( w , \xi , ( p , q ) ; F ) - 4 \varepsilon \overline { { q } } } \end{array}
$$

Thus with probability at least 1 − δ for ε additive error we need $\begin{array} { r } { \mathcal { O } \big ( \frac { \overline { { q } } ^ { 2 } } { \varepsilon ^ { 2 } } \log \frac { 1 } { \delta } \big ) } \end{array}$ samples.

## B Sample Compelxity with Demand Queries

## B.1 Proof of Theorem 6

We already know from Lemma 1 that $m _ { j }$ captures the $j ^ { t h }$ moment exactly. Let $\begin{array} { r } { \tilde { F } _ { j } = \sum _ { i = 0 } ^ { j } \alpha _ { i } L _ { i } ( x ) } \end{array}$ be the degree-j Legendre projection of F. Since $L _ { j } '$ s are a family of orthogonal polynomials under the usual measure, we know

$$
\begin{array} { l } { \displaystyle \alpha _ { j } = \int _ { 0 } ^ { 1 } F ( x ) L _ { j } ( x ) d x = \sum _ { i = 0 } ^ { j } a _ { j i } \int _ { 0 } ^ { 1 } F ( x ) x ^ { i } d x } \\ { \displaystyle = \sum _ { i = 0 } ^ { j } a _ { j i } \left( 1 - \int _ { 0 } ^ { 1 } \frac { x ^ { i + 1 } } { ( i + 1 ) } f ( x ) d x \right) = \sum _ { i = 0 } ^ { j } a _ { j i } \left( 1 - \frac { m _ { i + 1 } } { i + 1 } \right) , } \end{array}
$$

as claimed. For the de la Vallee Poussin mean, which are sufix averages of higher-degree Legendre projections, using Theorem 2.1 from Mastroianni et al. (2008), with $\alpha = \beta = \delta = \gamma = \nu = 0$ and $p = \infty$ substituted in their statement, for a constant C that changes neither with n nor $F ,$ we have

$$
\operatorname* { m a x } _ { v \in [ 0 , 1 ] } | F ( v ) - \hat { F } _ { \lceil n / 2 \rceil } ( x ) | \leq \Delta : = \frac { C B ^ { r } } { ( n / 2 ) ^ { r } } .
$$

Now it follows that $\begin{array} { r } { \operatorname* { m a x } _ { v \in [ 0 , 1 ] } | F ( v ) - \widehat { F } ( v ) | \leq \Delta } \end{array}$ , because forcing the CDF estimate to be non-negative and non-decreasing can only (weakly) decrease the $L _ { \infty }$ distance. Note that in this section, both F and $\widehat F$ are continuous, the former by assumption, and the latter because it is a polynomial. For continuous distributions, signals are simple partitions of the value space, as explained in Section 3.1. Consider any generic signal k such that $\pi ( k | v ) = 1 \{ v \in [ v _ { 0 } , v _ { 1 } ) \}$ . Now, we have that

$$
\begin{array} { c } { \displaystyle { \operatorname* { P r } _ { \pi o F } [ k ] = F ( v _ { 1 } ) - F ( v _ { 0 } ) , } } \\ { \displaystyle { \mathbb { E } _ { F } [ v \mathbf { 1 } \{ v \in [ v _ { 0 } , v _ { 1 } ) \} ] = \int _ { v _ { 0 } } ^ { v _ { 1 } } v f ( v ) d v = v _ { 1 } F ( v _ { 1 } ) - v _ { 0 } F ( v _ { 0 } ) - \int _ { v _ { 0 } } ^ { v _ { 1 } } F ( v ) d v . } } \end{array}
$$

Thus, we can transfer $L _ { \infty }$ approximation guarantees from F to probability of occurrence of signals, and the corresponding posterior means. Concretely, we have

$$
\left| \operatorname* { P r } _ { \pi \circ F } ( k ) - \operatorname* { P r } _ { \pi \circ \hat { F } } ( k ) \right| \leq \Delta , \mathrm { ~ a n d ~ } \left| \mathbb { E } _ { F } [ v | k ] - \mathbb { E } _ { \hat { F } } [ v | k ] \right| \leq \frac { \Delta } { \operatorname* { m a x } \{ \operatorname* { P r } _ { \pi \circ F } ( k ) , \operatorname* { P r } _ { \pi \circ \hat { F } } ( k ) \} } ,
$$

where the last display uses the inequality that $\begin{array} { r } { \left| { \frac { A ^ { \prime } } { B ^ { \prime } } } - { \frac { A } { B } } \right| \leq { \frac { | A ^ { \prime } - A | } { B } } + { \frac { A ^ { \prime } } { B ^ { \prime } } } { \frac { | B ^ { \prime } - B | } { B } } } \end{array}$

From here on, we essentially retrace the proof of Theorem 2. We decompose the error as

$$
R e v ^ { * } ( F ) - R e v ( \tilde { x } _ { \hat { F } } ; F ) = \underbrace { R e v ^ { * } ( F ) - R e v ^ { * } ( \hat { F } ) } _ { \mathrm { T e r m 1 } } + \underbrace { R e v ^ { * } ( \hat { F } ) - R e v ( \tilde { x } _ { \hat { F } } ; F , F ) } _ { \mathrm { T e r m 2 } } .
$$

To upper bound Term 1, let $\scriptstyle { \mathbf { \boldsymbol { x } } } _ { F }$ be the optimal pair of menu $( ( q _ { k } , p _ { k } ) : k \in [ K ] )$ and monotone partitional signaling scheme $( t _ { k } : k \in [ K ] )$ under $F$ . Let $z _ { \tau } = \{ k \in [ K ] : t _ { k } - t _ { k - 1 } < \tau \}$ . Define $\tilde { { \pmb x } } _ { F }$ with signaling scheme $\tilde { \pi }$ modified so that all signals in $z _ { \tau }$ are mapped to a null signal, and the prices discounted as $\tilde { p } _ { k } = p _ { k } - k \rho$ Then, we have that

$$
\begin{array} { r l } & { \mathtt { T e r m 1 } : = R e v ^ { * } ( F ) - R e v ( \tilde { x } _ { F } ; \hat { F } , \hat { F } ) + R e v ( \tilde { x } _ { F } ; \hat { F } , \hat { F } ) - R e v ^ { * } ( \hat { F } ) } \\ & { \qquad \le R e v ^ { * } ( F ) - R e v ( \tilde { x } _ { F } ; \hat { F } , \hat { F } ) } \end{array}
$$

where we once again appeal to the fact that $\mathit { R e v } ^ { * } ( \hat { \boldsymbol { F } } )$ is the optimal revenue for the empirical distribution $\widehat F$ $\mathrm { N o w } .$ , we use Lemma 3, which places an upper bound of $\mathcal { O } ( \overline { { q } } K \sqrt { \Delta } )$ on Term 1, by choosing $\rho = 4 \overline { { q } } \sqrt { \Delta }$ and $\tau = \sqrt { \Delta }$

Similarly, for Term2, let $\boldsymbol { x } _ { \widehat { F } }$ be the optimal pair of menu $( ( q _ { k } , p _ { k } ) : k \in [ K ] )$ and monotone partitional signaling scheme $( t _ { k } : k \in [ K ] )$ this time under $\widehat F$ . Let $\tilde { z } _ { \tau } = \{ k \in [ K ] : t _ { k } - t _ { k - 1 } < \tau \}$ . Let $\tilde { \ b { x } } _ { \hat { F } }$ be the analogous modification where all signals in $\tilde { z } _ { \tau }$ are mapped to a null signal, and prices are linear-additively discounted by $\rho$ at all indices. Now, applying Lemma 3, with the roles of $F$ and $\widehat F$ reversed, we get

$$
\begin{array} { r l } & { \mathtt { T e r m 2 : } = R e v ^ { * } ( \widehat { F } ) - R e v ^ { * } ( x _ { \widehat { F } } ; \widehat { F } , \widehat { F } ) + R e v ( x _ { \widehat { F } } ; \widehat { F } , \widehat { F } ) - R e v ^ { * } ( \widehat { x } _ { \widehat { F } } ; F , F ) } \\ & { \qquad \le R e v ( x _ { \widehat { F } } ; \widehat { F } , \widehat { F } ) - R e v ( \widehat { x } _ { F } ; \widehat { F } , \widehat { F } ) = \mathcal { O } ( \overline { { q } } K \sqrt { \Delta } ) . } \end{array}
$$

Now, we solve for n that makes the net revenue loss ε.

## C Regret under Joint Learning

Define the following quantity,

$$
\mathtt { G A P } ( x , F , \mathcal { C } ( F ) ) : = \operatorname* { s u p } _ { G \in \mathcal { C } ( F ) } \left( \underbrace { R e v ( x _ { F } ^ { \star } ; F , F ) } _ { \mathrm { o p t i m a l ~ r e v e n u e ~ u n d e r ~ t h e } } - \underbrace { R e v ( x ; G , F ) } _ { \mathrm { \normalfont ~ \mathrm { w h e n u e r ~ h o l d s ~ p r i o r ~ } } G } \right) .
$$

For any CDF F over $[ 0 , { \overline { { v } } } ]$ with $\overline { { v } } \leq 1$ , let the solution of $\mathtt { L } _ { F }$ be denoted by $\pmb { x } _ { F } = ( ( K , \{ t _ { i } \} _ { i = 0 } ^ { K } ) , ( p _ { i } , q _ { i } ) _ { i = 1 } ^ { K } )$ where $0 = t _ { 0 } < . . . < t _ { K } = 1$ is the quantile partition of the motonone partitional information structure, and $( p _ { i } , q _ { i } )$ is the direct menu item corresponding to the quantile interval $\left( t _ { k - 1 } , t _ { k } \right]$ . For $\tau \in [ 0 , 1 ]$ and $\rho \geq 0$ define the transformation $\tilde { \pmb { x } } _ { F } = \mathsf { T r } ( \pmb { x } _ { F } ; \tau , \rho )$ where $\tilde { \pmb { x } } _ { F }$ consists of the following modified signaling scheme in the value-space for $z _ { \tau } = \{ k \in [ K ] : t _ { k } - t _ { k - 1 } < \tau \}$

$$
\tilde { \pi } ( s  { | } v ) = \left\{ \begin{array} { l l } { { \pi } ( s _ { k }  { | } v ) } & { \mathrm { i f } s = s _ { k } , \ k \notin z _ { \tau } , \ \forall v \in [ 0 , \overline { { v } } ] } \\ { \sum _ { k \in z _ { \tau } } { \pi } ( s _ { k }  { | } v ) } & { \mathrm { i f } s = \mathrm { n u 1 1 } , \ \forall v \in [ 0 , \overline { { v } } ] } \\ { 1 } & { \mathrm { i f } s = \mathrm { n u 1 1 } , \ \forall v \in ( \overline { { v } } , 1 ] } \end{array} \right.
$$

where $( \pi ( k \mid v ) : \forall v \in [ 0 , \overline { { v } } ] , k \in [ K ] )$ is the value-space description w.r.t F from Section 3.1 with a modified menu $( \tilde { p } _ { i } , q _ { i } ) _ { i = 0 } ^ { K }$ where $\tilde { p } _ { i } = p _ { i } - i \rho \forall i \in [ K ]$

Lemma 4. Let F be a CDF over $\nu = [ 0 , \overline { { v } } ]$ with $\overline { { v } } \leq 1$ and let $\pmb { x } _ { F } = ( ( K , \{ t _ { i } \} _ { i = 0 } ^ { K } ) , ( p _ { i } , q _ { i } ) _ { i = 1 } ^ { K } )$ be the solution of $L _ { F }$ . For each CDF G, define $\mathbb { E } _ { G } [ v | k ]$ the posterior mean of signal $s _ { k }$ (which corresponds to the quantile interval $\left( t _ { k - 1 } , t _ { k } \right]$ of F) under G. Let $\mathcal { C } ( F )$ be the family of $C D F s ~ G$ on [0, 1] such that we have $\begin{array} { r } { | \mathbb { E } _ { F } [ v | k ] - \mathbb { E } _ { G } [ v | k ] | \le \rho / \bar { q } \forall k \in [ K ] \backslash z _ { \tau } \ f o r \ z _ { \tau } = \{ k \in [ K ] : t _ { k } - t _ { k - 1 } < \tau \} } \end{array}$ . Then $f o r \tilde { \pmb { x } } _ { F } = \mathsf { T r } ( \pmb { x } _ { F } ; \tau , \rho )$ we have $G A P ( \tilde { x } _ { F } , F , { \mathcal { C } } ( F ) ) \leq K ( \rho + \bar { q } \tau )$

Proof of Lemma $\it 4 .$ Since $( p _ { k } , q _ { k } )$ is a solution of $\operatorname { L } _ { F } .$ , it satisfies individual rationality (IR) under $F ,$ , i.e. $E _ { F } [ v | k ] \cdot q _ { k } - p _ { k } \geq 0$ for all $k \in [ K ]$ . For any $k \in [ K ] \backslash z _ { \tau }$ , since $| \mathbb { E } _ { F } [ v | k ] - \mathbb { E } _ { G } [ v | k ] | \le \rho / { \overline { { q } } }$ , we have under G that IR for all $k \in [ K ] \backslash z _ { \tau }$ with the modified prices.

Furthermore, the incentive compatibility constraint under F gives, for all $k , j \in [ K ]$

$$
\begin{array} { r } { \mathbb { E } _ { F } [ v | k ] \cdot q _ { k } - p _ { k } \ge \mathbb { E } _ { F } [ v | k ] \cdot q _ { j } - p _ { j } . } \end{array}\tag{3}
$$

Observe that for all $k \in [ K ]$ we have that

$$
\begin{array} { r l } & { q _ { k } \cdot \mathbb { E } _ { G } [ v | k ] - p _ { k } \geq q _ { k } \cdot \mathbb { E } _ { F } [ v | k ] - p _ { k } - \rho } \\ & { \qquad \geq \operatorname* { m a x } \{ 0 , \underset { j } { \operatorname* { m a x } } \{ q _ { j } \cdot \mathbb { E } _ { F } [ v | k ] - p _ { j } \} \} - \rho } \\ & { \qquad \geq \operatorname* { m a x } \{ 0 , \underset { j } { \operatorname* { m a x } } \{ q _ { j } \cdot \mathbb { E } _ { G } [ v | k ] - p _ { j } \} \} - 2 \rho } \end{array}
$$

Now observe that for all $j < k$ that

$$
\begin{array} { r l } & { q _ { k } \cdot \mathbb { E } _ { G } [ v | k ] - \tilde { p } _ { k } = q _ { k } \cdot \mathbb { E } _ { G } [ v | k ] - p _ { k } + k \rho } \\ & { \qquad \geq q _ { j } \cdot \mathbb { E } _ { G } [ v | k ] - p _ { j } + k \rho - \rho } \\ & { \qquad = q _ { j } \cdot \mathbb { E } _ { G } [ v | k ] - \tilde { p } _ { j } + ( k - j ) \rho - \rho } \\ & { \qquad \geq q _ { j } \cdot \mathbb { E } _ { G } [ v | k ] - \tilde { p } _ { j } . } \end{array}
$$

Thus ${ \tt G A P } ( \tilde { x } _ { F } , F , C ( F ) ) \leq K ( \rho + \bar { q } \tau )$ using the same argument as in Lemma A.2.

Definition 5. Let G denote the class of functions $g _ { u , v , \xi _ { u } , \xi _ { v } } : [ 0 , \overline { { v } } ] \to [ 0 , 1 ]$ of the form

$$
g _ { u , v , \xi _ { v } , \xi _ { v } } ( x ) : = \xi _ { u } \mathbf { 1 } \{ x = u \} + \mathbf { 1 } \{ u < x < v \} + \xi _ { v } \mathbf { 1 } \{ x = v \} , \quad u \leq v \in [ 0 , \overline { { v } } ] , \quad \xi _ { u } , \xi _ { v } \in [ 0 , 1 ] .
$$

Lemma 5. Let $F , G \in \Delta ( [ 0 , 1 ] )$ be CDFs with $\begin{array} { r } { \operatorname* { s u p } _ { v } | F ( v ) - G ( v ) | \leq \varepsilon } \end{array}$ . For any $g \in { \mathcal { G } }$ such that $\mathbb { E } _ { F } [ g ] , \mathbb { E } _ { G } [ g ] >$ 0, we have

$$
\left| \frac { \mathbb { E } _ { F } [ x g ] } { \mathbb { E } _ { F } [ g ] } - \frac { \mathbb { E } _ { G } [ x g ] } { \mathbb { E } _ { G } [ g ] } \right| \leq 4 \varepsilon \operatorname* { m i n } \left\{ \frac { 1 } { \mathbb { E } _ { F } [ g ] } , \frac { 1 } { \mathbb { E } _ { G } [ g ] } \right\} .
$$

Proof. By triangle inequality

$$
\left| \frac { \mathbb { E } _ { F } [ x g ] } { \mathbb { E } _ { F } [ g ] } - \frac { \mathbb { E } _ { G } [ x g ] } { \mathbb { E } _ { G } [ g ] } \right| \leq \frac { | \mathbb { E } _ { F } [ x g ] - \mathbb { E } _ { G } [ x g ] | } { \mathbb { E } _ { F } [ g ] } + \frac { \mathbb { E } _ { G } [ x g ] } { \mathbb { E } _ { F } [ g ] } \cdot \frac { | \mathbb { E } _ { F } [ g ] - \mathbb { E } _ { G } [ g ] | } { \mathbb { E } _ { G } [ g ] } .
$$

Since $\mathbb { E } _ { G } [ x g ] / \mathbb { E } _ { G } [ g ] \le 1$ as $x \in [ 0 , 1 ]$ , it sufices to bound $\left| \mathbb { E } _ { F } [ h ] - \mathbb { E } _ { G } [ h ] \right|$ for $h \in \{ g , x g \}$ . Observe that

$$
| \mathbb { E } _ { F } [ h ] - \mathbb { E } _ { G } [ h ] | \leq d _ { \mathrm { T V } } ( h ) \operatorname* { s u p } _ { v } | F ( v ) - G ( v ) | \leq \varepsilon d _ { \mathrm { T V } } ( h )
$$

where $\begin{array} { r } { d _ { \mathrm { T V } } ( h ) = \operatorname* { s u p } _ { n } \operatorname* { s u p } _ { 0 = x _ { 0 } < x _ { 1 } < \ldots < x _ { n } = 1 } \sum _ { i = 1 } ^ { n } \left| h ( x _ { i } ) - h ( x _ { i - 1 } ) \right| } \end{array}$ is the total variation of the function h on $[ 0 , 1 ]$ , this is so since

$$
\mathbb { E } _ { F } [ h ] - \mathbb { E } _ { G } [ h ] = \int _ { 0 } ^ { 1 } h \mathrm { d } ( F - G ) = - \int _ { 0 } ^ { 1 } ( F - G ) \mathrm { d } h .
$$

Now $g \in { \mathcal { G } }$ has the total variation at most 2, and $x \cdot g$ has the total variation at most 2 since $x \leq 1$ . Hence $| \mathbb { E } _ { F } [ g ] - \mathbb { E } _ { G } [ g ] | \le 2 \varepsilon$ and $| \mathbb { E } _ { F } [ x g ] - \mathbb { E } _ { G } [ x g ] | \le 2 \overline { { v } } \varepsilon$ . Substituting gives us the bound. Swapping F and G yields the same bound as $\mathbb { E } _ { G } [ g ]$ in the denominator, and taking the minimum proves the claim. □

Theorem 12. Algorithm 4 with cutof τ and discount $\rho _ { t } = 4 \bar { q } \varepsilon _ { t } / \tau _ { t } \ i s \ ( \beta , \kappa )$ -robustly revenue eficient for sets ${ \mathcal { C } } ( { \hat { F } } _ { t } ) = \left\{ G \in \Delta ( \mathcal { V } ) : \operatorname* { s u p } _ { v } \left| G ( v ) - { \hat { F } } _ { t } ( v ) \right| \leq \varepsilon _ { t } \right\}$ for $\begin{array} { r } { \beta \geq \sum _ { t \in [ T ] } 2 \exp \left( - t \varepsilon _ { t } ^ { 2 } / 2 \right) } \end{array}$ and $\kappa _ { t } = K \bar { q } ( 4 \varepsilon _ { t } / \tau _ { t } + \tau _ { t } )$ where $\nu = [ 0 , 1 ]$ for any sequence $\{ \varepsilon _ { t } \} _ { t = T } , \{ \tau _ { t } \} _ { t = 1 } ^ { T }$ with $\tau _ { t } , \varepsilon _ { t } \in ( 0 , 1 ] \ \forall t \in [ T ]$

Proof of Theorem 12. Consider $G \in { \mathcal { C } } ( { \hat { F } } _ { t } )$ , then we have by definition su $\mathrm { p } _ { v \in \mathcal { V } } | G ( v ) - \hat { F } _ { t } ( v ) | \leq \varepsilon _ { t }$

From Lemma 5, we have $\forall G \in \mathcal { C } ( \hat { F } _ { t } )$ the following for all $[ u , v ] \subseteq \mathcal { V }$

$$
\left| \frac { \mathbb { E } _ { F } [ x g ] } { \mathbb { E } _ { F } [ g ] } - \frac { \mathbb { E } _ { G } [ x g ] } { \mathbb { E } _ { G } [ g ] } \right| \leq 4 \varepsilon \operatorname* { m i n } \left\{ \frac { 1 } { \mathbb { E } _ { F } [ g ] } , \frac { 1 } { \mathbb { E } _ { G } [ g ] } \right\} .
$$

Let $\pmb { x } _ { \hat { F } } = \big ( ( I , \pmb { t } ) , ( p _ { i } , q _ { i } ) _ { i \in [ I ] } \big ) \big )$ be the solution of $\mathtt { L } _ { \hat { F } _ { t } }$ and let $z _ { \tau _ { t } } = \{ i \in [ I ] : t _ { i } - t _ { i - 1 } < \tau _ { t } \} \subseteq [ I ]$ . Then from above we have $\forall G \in \mathcal { C } ( \hat { F } _ { t } )$ the following

$$
\left| \mathbb { E } _ { G } [ v | k ] - \mathbb { E } _ { \hat { F } _ { t } } [ v | k ] \right| \le 4 \varepsilon _ { t } / \tau _ { t } = : \rho _ { t } / \bar { q } \quad \forall \ i \in [ I ] \backslash z _ { \tau } .
$$

For $\tilde { x } _ { \hat { F } _ { t } } = \mathsf { T r } ( \pmb { x } _ { \hat { F } _ { t } } ; \tau _ { t } , \rho _ { t } )$ using Lemma 4, we get

$$
\mathsf { G A P } \big ( \tilde { x } _ { \hat { F } _ { t } } , \hat { F } _ { t } , \mathcal { C } ( \hat { F } _ { t } ) \big ) \le K \bar { q } ( 4 \varepsilon _ { t } / \tau _ { t } + \tau _ { t } ) = : \kappa _ { t } .
$$

Finally,

$$
\begin{array} { r l } { \operatorname* { P r } _ { F ^ { \star } } \left( \cap _ { t = 1 } ^ { T } \mathcal { C } ( \hat { F } _ { t } ) \check { \varphi } F ^ { \star } \right) = \operatorname* { P r } _ { F ^ { \star } } \left( \bigcup _ { t = 1 } ^ { T } \mathcal { C } ( \hat { F } _ { t } ) ^ { c } \ni F ^ { \star } \right) } & { } \\ & { \leq \displaystyle \sum _ { t \in [ T ] } \operatorname* { P r } _ { F ^ { \star } } \left( \mathcal { C } ( \hat { F } _ { t } ) ^ { c } \ni F ^ { \star } \right) } \\ & { = \displaystyle \sum _ { t \in [ T ] } \operatorname* { P r } _ { F ^ { \star } } \left( \operatorname* { s u p } _ { v } | F ^ { \star } ( v ) - \hat { F } _ { t } ( v ) | > \epsilon _ { t } \right) } \\ & { \leq \displaystyle \sum _ { t \in [ T ] } 2 \exp \left( - 2 t \epsilon _ { t } ^ { 2 } \right) , } \end{array}
$$

where the last inequality follows from the Dvoretzky–Kiefer–Wolfowitz (DKW) Dvoretzky et al. (1956); Massart (1990) inequality given as follows

$$
\operatorname* { P r } \left( \operatorname* { s u p } _ { v \in \mathbb { R } } | F ( v ) - \hat { F } ( v ) | > \epsilon _ { n } \right) \le 2 \exp \left( - 2 n \epsilon _ { n } ^ { 2 } \right) .
$$

Theorem 13. Algorithm 4 with cutof $\tau _ { t } = \sqrt { \varepsilon _ { t } }$ and discount ${ \rho } _ { t } ~ = 4 \bar { q } \sqrt { \varepsilon _ { t } } .$ , for $\mathcal { C } ( \hat { F } _ { t } ) = \{ G \in \Delta ( \nu )$ $\begin{array} { r } { \operatorname* { s u p } _ { v } | G ( v ) - \hat { F } _ { t } ( v ) | \leq \varepsilon _ { t } \} } \end{array}$ with $\varepsilon _ { t } = \sqrt { 3 \log T / t }$ gives the following cumulative regret $R e g \le \mathcal { O } \left( K \bar { q } ( \log T ) ^ { 1 / 4 } T ^ { 3 / 4 } \right)$ with probability at least $1 - 3 T ^ { - 1 / 2 }$

Proof of Theorem 13. Define the event $\mathcal { E } : = \cap _ { t = 1 } ^ { T } \{ F ^ { \star } \in \mathcal { C } ( \hat { F } _ { t } ) \}$ . Let $\pmb { x } _ { \hat { F } } = \big ( ( I , \pmb { t } ) , ( p _ { i } , q _ { i } ) _ { i \in [ I ] } \big ) \big )$ be the solution of $\mathrm { L } _ { \hat { F } _ { t } }$ and let $z _ { \tau _ { t } } = \{ i \in [ I ] : t _ { i } - t _ { i - 1 } < \tau _ { t } \} \subseteq [ I ]$ . Then recall that Algorithm 4 uses $\tilde { x } _ { \hat { F } _ { t } } = \mathsf { T r } ( \pmb { x } _ { \hat { F } _ { t } } ; \tau _ { t } , \rho _ { t } )$ Then for $\varepsilon _ { t } = \sqrt { 3 \log T / t }$ and $\tau _ { t } = \sqrt { \varepsilon _ { t } } \forall t \in [ T ]$ , Theorem 12 gives $( \beta , \kappa )$ -robustly revenue eficiency for $\begin{array} { r } { \beta = \sum _ { t = 1 } ^ { T } 2 \exp ( - t \varepsilon _ { t } ^ { 2 } / 2 ) = \sum _ { t = 1 } ^ { T } 2 T ^ { - 3 / 2 } = 2 T ^ { - 1 / 2 } } \end{array}$ and $\kappa _ { t } = K \bar { q } ( 4 \varepsilon _ { t } / \tau _ { t } + \tau _ { t } ) = 5 K \bar { q } \sqrt { \varepsilon _ { t } }$ . This implies $\operatorname* { P r } ( { \mathcal { E } } ) \geq 1 - 2 T ^ { - 1 / 2 }$

Then $\forall G \in { \mathcal { C } } ( { \hat { F } } _ { t } )$ we have the following from Lemma 5

$$
\left| \mathbb { E } _ { G } [ v | k ] - \mathbb { E } _ { \hat { F } _ { t } } [ v | k ] \right| \le 4 \varepsilon _ { t } / \tau _ { t } = : \rho _ { t } / \bar { q } \quad \forall \ i \in [ I ] \backslash z _ { \tau } ,
$$

and using Lemma 4, we get

$$
\mathtt { G A P } ( \tilde { x } _ { \hat { F } _ { t } } , \hat { F } _ { t } , \mathcal { C } ( \hat { F } _ { t } ) ) \le K \bar { q } ( 4 \varepsilon _ { t } / \tau _ { t } + \tau _ { t } ) = \kappa _ { t } .
$$

Regret upper bound: On $\mathcal { E } , \ \operatorname* { s u p } _ { v } | F ^ { \star } ( v ) - \hat { F } _ { t } ( v ) | \leq \varepsilon _ { t } \ \forall t \in [ T ]$ . Also $F _ { t } ^ { b } \in \mathcal { C } ( \hat { F } _ { t } )$ by Assumption 1. Decompose the $t ^ { \mathrm { t h } } { \mathrm { - r e g r e } }$ t term as follows

$$
\begin{array} { r l } & { \mathsf { 0 P T } ( F ^ { \star } ) - R e v ( \tilde { x } _ { \hat { F } _ { t } } ; F _ { t } ^ { b } , F ^ { \star } ) = \underbrace { \left[ \mathsf { 0 P T } ( F ^ { \star } ) - \mathsf { 0 P T } ( \hat { F } _ { t } ) \right] } _ { \mathrm { ( A ) } } + \underbrace { \left[ \mathsf { 0 P T } ( \hat { F } _ { t } ) - R e v ( \tilde { x } _ { \hat { F } _ { t } } ; F _ { t } ^ { b } , \hat { F } _ { t } ) \right] } _ { \mathrm { ( B ) } } } \\ & { \qquad + \underbrace { \left[ R e v ( \tilde { x } _ { \hat { F } _ { t } } ; F _ { t } ^ { b } , \hat { F } _ { t } ) - R e v ( \tilde { x } _ { \hat { F } _ { t } } ; F _ { t } ^ { b } , F ^ { \star } ) \right] } _ { \mathrm { ( C ) } } . } \end{array}
$$

Since $F _ { t } ^ { b } \in \mathcal { C } ( \hat { F } _ { t } )$ we get $( \mathrm { B } ) \leq \mathtt { G A P } ( \tilde { x } _ { \hat { F } _ { t } } , \hat { F } _ { t } , \mathcal { C } ( \hat { F } _ { t } ) ) \leq \kappa _ { t } .$

For (C) the buyer’s decision is the same, so it can be upper bounded by q¯sup $\begin{array} { r } { | F ^ { \star } ( v ) - \hat { F } _ { t } ( v ) | \sum _ { k = 1 } ^ { K } d _ { \mathrm { T V } } ( g _ { k } ) \leq } \end{array}$ $2 \bar { q } \varepsilon _ { t } K$ since g has 2 jumps of size at most 1, where $d _ { \mathrm { T V } } ( \cdot )$ is the total variation distance.

Define $\tilde { \pmb { x } } _ { F ^ { \star } } = \mathsf { T r } ( \pmb { x } _ { F ^ { \star } } ; \tau _ { t } , \rho _ { t } )$ for $\pmb { x } _ { F ^ { \star } } = ( ( K ^ { \star } , { \pmb { t } } ^ { \star } ) , ( p _ { i } ^ { \star } , q _ { i } ^ { \star } ) _ { i = 1 } ^ { K } )$ solution of $\mathbb { L } _ { F ^ { \star } }$ . Then $\forall G \in { \mathcal { C } } ( F ^ { \star } )$ we have the following from Lemma 5

$$
\begin{array} { r } { | \mathbb { E } _ { G } [ v | k ] - \mathbb { E } _ { F ^ { \star } } [ v | k ] | \leq 4 \varepsilon _ { t } / \tau _ { t } = \rho _ { t } / \bar { q } \quad \forall \ k \in [ K ^ { \star } ] : t _ { k } ^ { \star } - t _ { k - 1 } ^ { \star } \geq \tau _ { t } , } \end{array}
$$

and using Lemma 4, we get

$$
\mathtt { G A P } \big ( \tilde { \pmb { x } } _ { F ^ { \star } } , F ^ { \star } , \mathcal { C } ( F ^ { \star } ) \big ) \le K \bar { q } ( 4 \varepsilon _ { t } / \tau _ { t } + \tau _ { t } ) = \kappa _ { t } .
$$

Recall that under E, we also have $\hat { F } _ { t } \in \mathcal { C } ( F ^ { \star } ) \ \forall t \in [ T ]$ . For (A) we perform the following decomposition

$$
\begin{array} { r l } & { \mathrm { ( A ) } \leq \underbrace { \left[ 0 \mathrm { P T } ( F ^ { \star } ) - R e v ( \tilde { x } _ { F ^ { \star } } ; \hat { F } _ { t } , F ^ { \star } ) \right] } _ { \leq \mathrm { G a b } ( \tilde { x } _ { F ^ { \star } } , F ^ { \star } , \mathcal { C } ( F ^ { \star } ) ) \leq \kappa _ { t } } + \underbrace { \left[ R e v ( \tilde { x } _ { F ^ { \star } } ; \hat { F } _ { t } , F ^ { \star } ) - R e v ( \tilde { x } _ { F ^ { \star } } ; \hat { F } _ { t } , \hat { F } _ { t } ) \right] } _ { \leq 2 \bar { q } \varepsilon _ { t } K \mathrm { ~ u s i n g ~ s a m e ~ T V ~ a r g u m e n t ~ a s ~ ( C ) ~ } } } \\ & { + \quad \underbrace { \left[ R e v ( \tilde { x } _ { F ^ { \star } } ; \hat { F } _ { t } , \hat { F } _ { t } ) - 0 \mathrm { P T } ( \hat { F } _ { t } ) \right] } _ { \leq 0 \mathrm { ~ d u e ~ t o ~ s u b o p t i m a l i t y ~ o f ~ } \tilde { x } _ { F ^ { \star } } \mathrm { ~ u n d e r ~ } \hat { F } _ { t } } } \\ & { \leq \kappa _ { t } + 2 \bar { q } \varepsilon _ { t } K . } \end{array}
$$

Combining we get overall

$$
\begin{array} { r } { 0 \mathrm { P T } ( F ^ { \star } ) - R e v ( \tilde { x } _ { \hat { F } _ { t } } ; F _ { t } ^ { b } , F ^ { \star } ) \leq 2 \kappa _ { t } + 4 \bar { q } \varepsilon _ { t } K . } \end{array}
$$

Finally, define $Y _ { t } : = \mathbf { r e v } _ { t } ( \tilde { x } _ { \hat { E } } ; F _ { t } ^ { b } ) - R e v ( \tilde { x } _ { \hat { E } } ; F _ { t } ^ { b } , F ^ { \star } )$ . Since $\tilde { \pmb { x } } _ { \hat { F } _ { t } }$ and $F _ { t } ^ { b }$ are H -measurable and $v _ { t } | \mathcal { H } _ { t } \sim$ $F ^ { \star }$ , we have $\mathbb { E } [ Y _ { t } | \mathcal { H } _ { t } ] = 0$ and $| \dot { Y } _ { t } | \le \bar { q }$ where $\ddot { \mathcal { H } } _ { t }$ denotes the σ-algebra of all the histories upto time t. By Azuma-Hoefding, we get

$$
\operatorname* { P r } \left( \sum _ { t = 1 } ^ { T } ( - Y _ { t } ) \geq z \bar { q } \right) \leq \exp \left( - \frac { z ^ { 2 } } { 2 T } \right) .
$$

Setting $z = { \sqrt { T \log T } }$ gives $T ^ { - 1 / 2 }$ upper bound.

Thus overall on the event $\mathcal E \cap \{ \sum _ { t } ( - Y _ { t } ) \leq z \bar { q } \}$ , which now holds with probabili $\mathrm { t y } \geq 1 - 2 T ^ { - 1 / 2 } - T ^ { - 1 / 2 } =$ $1 - 3 T ^ { - 1 / 2 }$ <sup>2</sup>, we get

$$
\begin{array} { r l } & { \mathsf { R e g } = \displaystyle \sum _ { t = 1 } ^ { T } \left( 0 \mathsf { P T } \left( F ^ { \star } \right) - \mathbf { r } \mathbf { e } \mathbf { v } _ { t } \right) = \displaystyle \sum _ { t = 1 } ^ { T } \left( 0 \mathsf { P T } \left( F ^ { \star } \right) - R e v ( \tilde { x } _ { \hat { F } _ { t } } ; F _ { t } ^ { b } , F ^ { \star } ) \right) + \displaystyle \sum _ { t = 1 } ^ { T } ( - Y _ { t } ) } \\ & { \qquad \le \displaystyle \sum _ { t = 1 } ^ { T } ( 2 \kappa _ { t } + 4 \bar { q } \varepsilon _ { t } K ) + \bar { q } \sqrt { T \log T } . } \end{array}
$$

Substituting $\tau _ { t } = \sqrt { \varepsilon _ { t } }$ gives us $\kappa _ { t } = K \bar { q } ( 4 \varepsilon _ { t } / \tau _ { t } + \tau _ { t } ) = 5 K \bar { q } \sqrt { \varepsilon _ { t } }$ and since $\varepsilon _ { t } = \sqrt { 3 \log T / t }$ , we have $\sqrt { \varepsilon _ { t } } = ( 3 \log T ) ^ { 1 / 4 } t ^ { - 1 / 4 }$ and $\varepsilon _ { t } = ( 3 \log T ) ^ { 1 / 2 } t ^ { - 1 / 2 }$ . Using $\textstyle \sum _ { t = 1 } ^ { T } t ^ { - 1 / 4 } \leq { \frac { 4 } { 3 } } T ^ { 3 / 4 }$ and $\textstyle \sum _ { t = 1 } ^ { T } t ^ { - 1 / 2 } \leq 2 T ^ { 1 / 2 }$ we get the leading term as follows

$$
\sum _ { t = 1 } ^ { T } \sqrt { \varepsilon _ { t } } \leq \frac { 4 } { 3 } ( 3 \log T ) ^ { 1 / 4 } T ^ { 3 / 4 } = \mathcal { O } \left( ( \log T ) ^ { 1 / 4 } T ^ { 3 / 4 } \right)
$$

Finally we have

$$
\mathsf { R e g } \leq \mathcal { O } \left( K \bar { q } ( \log T ) ^ { 1 / 4 } T ^ { 3 / 4 } \right) .
$$

Consider the following set for which we show the $\mathcal { O } ( T ^ { 2 / 3 } )$ regret guaranty.

$$
\mathcal { C } ^ { \prime } ( \hat { F } _ { t } ) = \left\{ G \in \Delta ( \mathcal { V } ) : \left| \frac { \mathbb { E } _ { \hat { F } _ { t } } [ x g ] } { \mathbb { E } _ { \hat { F } _ { t } } [ g ] } - \frac { \mathbb { E } _ { G } [ x g ] } { \mathbb { E } _ { G } [ g ] } \right| \leq \varepsilon _ { t } \sqrt { \frac { 1 } { \operatorname* { m a x } \{ \mathbb { E } _ { \hat { F } _ { t } } [ g ] , \mathbb { E } _ { G } [ g ] \} } } + \varepsilon _ { t } ^ { 2 } \frac { 1 } { \operatorname* { m a x } \{ \mathbb { E } _ { \hat { F } _ { t } } [ g ] , \mathbb { E } _ { G } [ g ] \} } \right\} .
$$

Theorem 14. Algorithm 4 with cutof $\tau _ { t }$ and discount $\rho _ { t } = \bar { q } ( \varepsilon _ { t } / \sqrt { \tau _ { t } } + \varepsilon _ { t } ^ { 2 } / \tau _ { t } )$ is $( \beta , \kappa )$ -robustly revenue eficient for the sets $\mathcal { C } ^ { \prime } ( \hat { F } _ { t } )$ for $\begin{array} { r } { \beta \geq \sum _ { t \in \left[ T \right] } t \exp \left( - t \varepsilon _ { t } ^ { 2 } / C \right) } \end{array}$ and $\kappa _ { t } = K \bar { q } ( \varepsilon _ { t } / \sqrt { \tau _ { t } } + \varepsilon _ { t } ^ { 2 } / \tau _ { t } + \tau _ { t } )$ where $\nu = [ 0 , 1 ]$ for any sequence $\{ \varepsilon _ { t } \} _ { t = T } , \{ \tau _ { t } \} _ { t = 1 } ^ { T }$ with $\tau _ { t } , \varepsilon _ { t } \in ( 0 , 1 ] \ \forall t \in [ T ]$

Proof of Theorem 14. Consider $G \in { \mathcal { C } } ^ { \prime } ( { \hat { F } } _ { t } )$ . Let $\pmb { x } _ { \hat { F } } = \big ( ( I , \pmb { t } ) , ( p _ { i } , q _ { i } ) _ { i \in [ I ] } \big ) \big )$ be the solution of $\mathtt { L } _ { \hat { F } _ { t } }$ . Then we have $\forall G \in { \mathcal { C } } ^ { \prime } ( { \hat { F } } _ { t } )$ the following by definition

$$
\left| \mathbb { E } _ { G } [ v | k ] - \mathbb { E } _ { \hat { F } _ { t } } [ v | k ] \right| \le \varepsilon _ { t } / \sqrt { \tau _ { t } } + \varepsilon _ { t } ^ { 2 } / \tau _ { t } = : \rho _ { t } / \bar { q } \forall \ i \in [ I ] \backslash z _ { \tau } .
$$

For $\tilde { x } _ { \hat { F } _ { t } } = \mathsf { T r } ( \pmb { x } _ { \hat { F } _ { t } } ; \tau _ { t } , \rho _ { t } )$ using Lemma 4, we get

$$
\mathtt { G A P } \big ( \tilde { x } _ { \hat { F } _ { t } } , \hat { F } _ { t } , \mathcal { C } ^ { \prime } ( \hat { F } _ { t } ) \big ) \le K \bar { q } ( \varepsilon _ { t } / \sqrt { \tau _ { t } } + \varepsilon _ { t } ^ { 2 } / \tau _ { t } + \tau _ { t } ) = : \kappa _ { t } .
$$

Finally,

$$
\begin{array} { r l r } {  { \operatorname* { P r } _ { F ^ { \star } } ( \cap _ { t = 1 } ^ { T } \mathcal { C } ^ { \prime } ( \hat { F } _ { t } ) \prec  { F } ^ { \star } ) = \operatorname* { P r } _ { F ^ { \star } } ( \cup _ { t = 1 } ^ { T } \mathcal { C } ^ { \prime } ( \hat { F } _ { t } ) ^ { c } \ni F ^ { \star } ) } } \\ & { } & { \quad \le \displaystyle \sum _ { t \in [ T ] } \operatorname* { P r } _ { F ^ { \star } } ( \mathcal { C } ^ { \prime } ( \hat { F } _ { t } ) ^ { c } \ni F ^ { \star } ) } \\ & { } & { \quad \le \displaystyle \sum _ { t \in [ T ] } t \exp ( - t \varepsilon _ { t } ^ { 2 } / C ) , \qquad } \end{array}
$$

where the last inequality follows from Lemma 2 for some constant C.

Theorem 15. Algorithm 4 with cutof $\tau _ { t } = \varepsilon _ { t } ^ { 1 / 3 }$ and discount $\rho _ { t } = \bar { q } ( \varepsilon _ { t } / \sqrt { \tau _ { t } } + \varepsilon _ { t } ^ { 2 } / \tau _ { t } )$ , for $\mathcal { C } ^ { \prime } ( \hat { F } _ { t } )$ with $\varepsilon _ { t } = \sqrt { 5 \log T / ( 2 t ) }$ gives the following cumulative regret $R e g \le \mathcal { O } \left( K \bar { q } ( \log T ) ^ { 1 / 3 } T ^ { 2 / 3 } \right)$ with probability at least $1 - \mathcal { O } ( T ^ { - 1 / 2 } )$ .

Proof of Theorem 15. Define the event $\begin{array} { r } { \mathcal E : = \cap _ { t = 1 } ^ { T } \{ F ^ { \star } \in { \mathcal C } ( \hat { F } _ { t } ) : = \{ G \in \Delta ( [ 0 , 1 ] ) : \operatorname* { m a x } _ { v } | G ( v ) - \hat { F } _ { t } ( v ) | \leq \xi _ { t } \} \} } \end{array}$ and $\mathcal { E } ^ { \prime } : = \cap _ { t = 1 } ^ { T } \{ F ^ { \star } \in \mathcal { C } ^ { \prime } ( \hat { F } _ { t } ) \}$ . Let $\pmb { x } _ { \hat { F } } = \big ( ( I , \pmb { t } ) , ( p _ { i } , q _ { i } ) _ { i \in [ I ] } \big ) \big )$ be the solution of $\mathtt { L } _ { \hat { F } _ { t } }$ and let $z _ { \tau _ { t } } = \{ i \in [ I ]$ $t _ { i } - t _ { i - 1 } < \tau _ { t } \} \subseteq [ I ]$ . Then recall that Algorithm 4 uses $\tilde { x } _ { \hat { F } _ { t } } = \mathsf { T r } ( \mathbf { x } _ { \hat { F } _ { t } } ; \tau _ { t } , \rho _ { t } )$ . Then for $\varepsilon _ { t } = \sqrt { 5 \log T / ( 2 t ) }$ and $\tau _ { t } = \varepsilon _ { t } ^ { 1 / 3 }$ , Theorem 14 gives $( \beta , \kappa )$ -robustly revenue eficiency for $\begin{array} { r } { \beta = \sum _ { t \in \left[ T \right] } t \exp \left( - t \varepsilon _ { t } / C \right) = \mathcal { O } ( T ^ { - 1 / 2 } ) } \end{array}$ and $\kappa _ { t } = K \bar { q } ( \varepsilon _ { t } / \sqrt { \tau _ { t } } + \varepsilon _ { t } ^ { 2 } / \tau _ { t } + \tau _ { t } ) = \mathcal { O } ( K \overline { { q } } \varepsilon _ { t } ^ { 1 / 3 } )$ . This gives $\operatorname* { P r } ( \mathcal { E } ^ { \prime } ) \geq 1 - \mathcal { O } ( T ^ { - 1 / 2 } )$

Also, we have $\forall G \in { \mathcal { C } } ^ { \prime } ( { \hat { F } } _ { t } )$ the following by definition

$$
\left| \mathbb { E } _ { G } [ v | k ] - \mathbb { E } _ { \hat { F } _ { t } } [ v | k ] \right| \le \varepsilon _ { t } / \sqrt { \tau _ { t } } + \varepsilon _ { t } ^ { 2 } / \tau _ { t } = \rho _ { t } / \bar { q } \forall \ i \in [ I ] \backslash z _ { \tau } .
$$

For $\tilde { x } _ { \hat { F } _ { t } }$ using Lemma 4, we get

$$
\mathtt { G A P } \big ( \tilde { x } _ { \hat { F } _ { t } } , \hat { F } _ { t } , \mathcal { C } ^ { \prime } ( \hat { F } _ { t } ) \big ) \le K \bar { q } ( \varepsilon _ { t } / \sqrt { \tau _ { t } } + \varepsilon _ { t } ^ { 2 } / \tau _ { t } + \tau _ { t } ) = \kappa _ { t } .
$$

Using DKW we also have

$$
1 - \operatorname* { P r } ( \mathcal { E } ) \leq \sum _ { t \in [ T ] } 2 \exp ( - 2 t \xi _ { t } ^ { 2 } ) .
$$

For $\xi _ { t } = \sqrt { \frac { \log ( 2 T ^ { 3 / 2 } ) } { 2 t } }$ we get $\operatorname* { P r } ( \mathcal { E } ) \geq 1 - \mathcal { O } ( T ^ { - 1 / 2 } )$ and therefore $\operatorname* { P r } ( \mathcal { E } \cap \mathcal { E } ^ { \prime } ) \geq 1 - \mathcal { O } ( T ^ { - 1 / 2 } )$

Regret upper bound: $F _ { t } ^ { b } \in \mathcal { C } ( \hat { F } _ { t } )$ by Assumption 1. Under the event $\mathcal { E } \cap \mathcal { E } ^ { \prime }$ decompose the $t ^ { \mathrm { t h } }$ -regret term as follows

$$
\begin{array} { r l } & { \mathsf { 0 P T } ( F ^ { \star } ) - R e v ( \tilde { x } _ { \hat { F } _ { t } } ; F _ { t } ^ { b } , F ^ { \star } ) = \underbrace { \left[ \mathsf { 0 P T } ( F ^ { \star } ) - \mathsf { 0 P T } ( \hat { F } _ { t } ) \right] } _ { \mathrm { ( A ) } } + \underbrace { \left[ \mathsf { 0 P T } ( \hat { F } _ { t } ) - R e v ( \tilde { x } _ { \hat { F } _ { t } } ; F _ { t } ^ { b } , \hat { F } _ { t } ) \right] } _ { \mathrm { ( B ) } } } \\ & { \qquad + \underbrace { \left[ R e v ( \tilde { x } _ { \hat { F } _ { t } } ; F _ { t } ^ { b } , \hat { F } _ { t } ) - R e v ( \tilde { x } _ { \hat { F } _ { t } } ; F _ { t } ^ { b } , F ^ { \star } ) \right] } _ { \mathrm { ( C ) } } . } \end{array}
$$

Since $F _ { t } ^ { b } \in \mathcal { C } ^ { \prime } ( \hat { F } _ { t } )$ , we get $( \mathrm { B } ) \le \mathtt { G A P } ( \tilde { x } _ { \hat { F } _ { t } } , \hat { F } _ { t } , \mathcal { C } ^ { \prime } ( \hat { F } _ { t } ) ) \le \kappa _ { t }$

For (C) the buyer’s decision is the same, so it can be upper bounded by q¯sup<sub>v</sub> $\begin{array} { r } { | F ^ { \star } ( v ) - \hat { F } _ { t } ( v ) | \sum _ { k = 1 } ^ { K } d _ { \mathrm { T V } } ( g _ { k } ) \leq } \end{array}$ $2 \bar { q } \xi _ { t } K$ since g has atmost 2 jumps. 9

Define $\tilde { \pmb { x } } _ { F ^ { \star } } = \mathsf { T r } ( \pmb { x } _ { F ^ { \star } } ; \tau _ { t } , \rho _ { t } )$ for $\pmb { x } _ { F ^ { \star } } = ( ( K ^ { \star } , { \pmb { t } } ^ { \star } ) , ( p _ { i } ^ { \star } , q _ { i } ^ { \star } ) _ { i = 1 } ^ { K } )$ solution of $\operatorname { L } _ { F ^ { \star } }$ . Then $\forall G \in { \mathcal { C } } ^ { \prime } ( F ^ { \star } )$ we have the following by definition

$$
\begin{array} { r } { | \mathbb { E } _ { G } [ v | k ] - \mathbb { E } _ { F ^ { \star } } [ v | k ] | \leq \rho _ { t } / \bar { q } \quad \forall \ k \in [ K ^ { \star } ] : t _ { k } ^ { \star } - t _ { k - 1 } ^ { \star } \geq \tau _ { t } , } \end{array}
$$

and using Lemma 4, we get

$$
\mathtt { G A P } ( { \tilde { x } } _ { F ^ { \star } } , F ^ { \star } , { \mathcal { C } } ^ { \prime } ( F ^ { \star } ) ) \leq \kappa _ { t } .
$$

Recall that under $\mathcal { E } \cap \mathcal { E } ^ { \prime }$ , we also have $\hat { F } _ { t } \in \mathcal { C } ^ { \prime } ( F ^ { \star } ) \ \forall t \in [ T ]$ . For (A) we perform the following decomposition

$$
\begin{array} { r l } & { \mathrm { ( A ) } \leq \underbrace { \left[ 0 \mathrm { P T } ( F ^ { \star } ) - R e v ( \tilde { x } _ { F ^ { \star } } ; \hat { F } _ { t } , F ^ { \star } ) \right] } _ { \leq \mathrm { G a p } ( \tilde { x } _ { F ^ { \star } } , F ^ { \star } , \mathcal { C } ^ { \prime } ( F ^ { \star } ) ) \leq \kappa _ { t } } + \underbrace { \left[ R e v ( \tilde { x } _ { F ^ { \star } } ; \hat { F } _ { t } , F ^ { \star } ) - R e v ( \tilde { x } _ { F ^ { \star } } ; \hat { F } _ { t } , \hat { F } _ { t } ) \right] } _ { \leq 2 \bar { q } \xi _ { t } K \mathrm { ~ u s i n g ~ s a m e ~ T V ~ a r g u m e n t ~ a s ~ ( C ) ~ } } } \\ & { + \quad \underbrace { \left[ R e v ( \tilde { x } _ { F ^ { \star } } ; \hat { F } _ { t } , \hat { F } _ { t } ) - 0 \mathrm { P T } ( \hat { F } _ { t } ) \right] } _ { \leq 0 \mathrm { ~ d u e ~ t o ~ s u b o p t i m a l i t y ~ o f ~ } \tilde { x } _ { F ^ { \star } } \mathrm { ~ u n d e r ~ } \hat { F } _ { t } } } \\ & { \leq \kappa _ { t } + 2 \bar { q } \xi _ { t } K . } \end{array}
$$

Combining we get overall

$$
\tt O P T  ( F ^ { \star } ) - R e v ( \tilde { x } _ { \hat { F } _ { t } } ; F _ { t } ^ { b } , F ^ { \star } ) \leq 2 \kappa _ { t } + 4 \bar { q } \xi _ { t } K .
$$

Finally, define $Y _ { t } : = \mathbf { r e v } _ { t } ( \tilde { x } _ { \hat { F } } ; F _ { t } ^ { b } ) - R e v ( \tilde { x } _ { \hat { F } _ { \ast } } ; F _ { t } ^ { b } , F ^ { \star } )$ . Since $\tilde { \pmb { x } } _ { \hat { F } _ { t } }$ and $F _ { t } ^ { b }$ are $\mathcal { H } _ { t } { \mathrm { - m e a s u r a b l e } }$ and $v _ { t } | \mathcal { H } _ { t } \sim$ $F ^ { \star }$ , we have $\mathbb { E } [ Y _ { t } | \mathcal { H } _ { t } ] = 0$ and $| \dot { Y } _ { t } | \le \bar { q }$ where $\mathcal { \hat { H } } _ { t }$ denotes the σ-algebra of all the histories upto time t. By Azuma-Hoefding, we $\mathrm { g e t }$

$$
\operatorname* { P r } \left( \sum _ { t = 1 } ^ { T } ( - Y _ { t } ) \geq z \bar { q } \right) \leq \exp \left( - \frac { z ^ { 2 } } { 2 T } \right) .
$$

Setting $z = \sqrt { T \log T }$ gives $T ^ { - 1 / 2 }$ upper bound.

Thus overall on the event $\mathcal E \cap \mathcal E ^ { \prime } \cap \{ \sum _ { t } ( - Y _ { t } ) \leq z \bar { q } \}$ , which now holds with probability $\geq 1 - \mathcal { O } ( T ^ { - 1 / 2 } )$ we get

$$
\begin{array} { l } { \displaystyle \mathtt { R e g } = \sum _ { t = 1 } ^ { T } \bigl ( 0 \mathrm { P T } \bigl ( F ^ { \star } \bigr ) - \mathbf { r } \mathbf { e } \mathbf { v } _ { t } \bigr ) = \sum _ { t = 1 } ^ { T } \bigl ( 0 \mathrm { P T } \bigl ( F ^ { \star } \bigr ) - R e v \bigl ( \tilde { x } _ { \hat { F } _ { t } } ; F _ { t } ^ { b } , F ^ { \star } \bigr ) \bigr ) + \sum _ { t = 1 } ^ { T } ( - Y _ { t } ) } \\ { \displaystyle \leq \sum _ { t = 1 } ^ { T } \bigl ( 2 \kappa _ { t } + 4 \bar { q } \xi _ { t } K \bigr ) + \bar { q } \sqrt { T \log T } . } \end{array}
$$

Substituting $\tau _ { t } = \varepsilon _ { t } ^ { 1 / 3 }$ gives us $\kappa _ { t } = K \bar { q } ( \varepsilon _ { t } / \sqrt { \tau _ { t } } + \varepsilon _ { t } ^ { 2 } / \tau _ { t } + \tau _ { t } ) = \mathcal { O } ( K \bar { q } \varepsilon _ { t } ^ { 1 / 3 } )$ and since $\varepsilon _ { t } = \sqrt { 5 C \log T / ( 2 t ) }$ we have $\varepsilon _ { t } ^ { 1 / 3 } = ( 5 C \log T / ( 2 t ) ) ^ { 2 / 3 }$ and $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \xi _ { t } = \mathcal { O } ( \sqrt { 3 T \log T } ) } \end{array}$ , which gives us

$$
\mathsf { R e g } \leq \mathcal { O } \left( K \bar { q } ( \log T ) ^ { 1 / 3 } T ^ { 2 / 3 } \right)
$$

## D Proofs for the FPTAS

Lemma 6. Consider any quantile interval $[ t _ { 1 } , t _ { 2 } ]$ with $0 \leq t _ { 1 } < t _ { 2 } \leq 1$ , and define $\begin{array} { r } { \tilde { t } _ { 1 } : = \varepsilon \lfloor \frac { t _ { 1 } } { \varepsilon } \rfloor , \tilde { t } _ { 2 } : = \varepsilon \lfloor \frac { t _ { 2 } } { \varepsilon } \rfloor } \end{array}$ Then $| ( t _ { 2 } - t _ { 1 } ) - ( \tilde { t } _ { 2 } - \tilde { t } _ { 1 } ) | \leq \varepsilon$ . Moreover, let $\mu ( t _ { 1 } , t _ { 2 } ) = \frac { \int _ { t _ { 1 } } ^ { t _ { 2 } } Q ( u ) \mathrm { d } u } { t _ { 2 } - t _ { 1 } }$ and define $\mu ( \tilde { t } _ { 1 } , \tilde { t } _ { 2 } ) = \frac { \int _ { \tilde { t } _ { 1 } } ^ { t _ { 2 } } Q ( u ) \mathrm { d } u } { \tilde { t } _ { 2 } - \tilde { t } _ { 1 } }$ and as $Q ( \tilde { t } _ { 1 } )$ whenever $\tilde { t } _ { 1 } = \tilde { t } _ { 2 }$ . Then

$$
| \mu ( t _ { 1 } , t _ { 2 } ) - \mu ( \tilde { t } _ { 1 } , \tilde { t } _ { 2 } ) | \leq \frac { 3 \varepsilon } { t _ { 2 } - t _ { 1 } } .
$$

Proof of Lemma 6. By definition, $\tilde { t } _ { 1 } \leq t _ { 1 }$ and $t _ { 1 } - \varepsilon \le \tilde { t } _ { 1 }$ and similarly, $\tilde { t } _ { 2 } \le t _ { 2 }$ and $t _ { 2 } - \varepsilon \le \tilde { t } _ { 2 }$ . Thus   
$t _ { 2 } - t _ { 1 } \le \tilde { t } _ { 2 } + \varepsilon - \tilde { t } _ { 1 }$ and $t _ { 2 } - t _ { 1 } \geq \tilde { t } _ { 2 } - \tilde { t } _ { 1 } - \varepsilon$ . Thus $| t _ { 2 } - t _ { 1 } - ( \tilde { t } _ { 2 } - \tilde { t } _ { 1 } ) | \leq \varepsilon$   
Thus if $\tilde { t } _ { 2 } - \tilde { t } _ { 1 } = 0$ , then $t _ { 2 } - t _ { 1 } < \varepsilon$ , then the inequality immediately follows

$$
| \mu ( t _ { 2 } , t _ { 1 } ) - Q ( \tilde { t } _ { 1 } ) | \leq 1 \leq 3 \varepsilon / ( t _ { 2 } - t _ { 1 } ) .
$$

Otherwise for $\tilde { t } _ { 2 } > \tilde { t } _ { 1 }$ we have

$$
\left| \int _ { t _ { 1 } } ^ { t _ { 2 } } Q ( u ) \mathrm { d } u - \int _ { \tilde { t } _ { 1 } } ^ { \tilde { t } _ { 2 } } Q ( u ) \mathrm { d } u \right| \leq \left| \int _ { t _ { 1 } } ^ { \tilde { t } _ { 1 } } Q ( u ) \mathrm { d } u \right| + \left| \int _ { t _ { 2 } } ^ { \tilde { t } _ { 2 } } Q ( u ) \mathrm { d } u \right| \leq 2 \varepsilon
$$

which gives

$$
\begin{array} { r l } & { \displaystyle | \mu ( t _ { 1 } , t _ { 2 } ) - \mu ( \tilde { t } _ { 1 } , \tilde { t } _ { 1 } ) | \leq \frac { \left| \int _ { t _ { 1 } } ^ { t _ { 2 } } Q ( u ) \mathrm { d } u - \int _ { \tilde { t } _ { 1 } } ^ { \tilde { t } _ { 2 } } Q ( u ) \mathrm { d } u \right| } { t _ { 2 } - t _ { 1 } } + \int _ { \tilde { t } _ { 1 } } ^ { \tilde { t } _ { 2 } } Q ( u ) \mathrm { d } u \left| \frac { 1 } { t _ { 2 } - t _ { 1 } } - \frac { 1 } { \tilde { t } _ { 2 } - \tilde { t } _ { 1 } } \right| } \\ & { \qquad \leq \frac { 2 \varepsilon } { t _ { 1 } - t _ { 2 } } + \int _ { \tilde { t } _ { 1 } } ^ { \tilde { t } _ { 2 } } Q ( u ) \mathrm { d } u \left| \frac { 1 } { t _ { 1 } - t _ { 2 } } - \frac { 1 } { \tilde { t } _ { 1 } - \tilde { t } _ { 2 } } \right| } \\ & { \qquad \leq \frac { 2 \varepsilon } { t _ { 2 } - t _ { 1 } } + ( \tilde { t } _ { 2 } - \tilde { t } _ { 1 } ) \frac { \varepsilon } { ( t _ { 2 } - t _ { 1 } ) ( \tilde { t } _ { 2 } - \tilde { t } _ { 1 } ) } } \\ & { \qquad \leq 3 \varepsilon / \tau } \end{array}
$$

Theorem 16 (Correctness of Algorithm 2). For $L _ { F } ^ { s }$ restricted to quantiles in $\mathcal { E } _ { t }$ and qualities in $\mathcal { E } _ { q }$ , the Algorithm 2 outputs an optimal menu $\{ ( p _ { k } , q _ { j _ { k } } ) \} _ { k = 1 } ^ { k ^ { \star } }$ and a valid information structure $\boldsymbol { \mathcal { T } } = ( \boldsymbol { k } ^ { \star } , t )$

Proof. For $k \in [ K ] , r \in [ n _ { t } ]$ and $j \in [ n _ { q } ]$ , define

$$
V ( k , r , j ) : = \operatorname* { m a x } \left\{ \sum _ { i = 1 } ^ { k } \mathcal { U } [ \ell _ { i - 1 } , \ell _ { i } , j _ { i - 1 } , j _ { i } ] \Big | 0 = \ell _ { 0 } < . . . < \ell _ { k } = r , 0 = j _ { 0 } \right\}
$$

which denotes the maximum revenue that can be obtain from partitioning $( 0 , t _ { r } ]$ into k nonempty quantile grid intervals with the last interval assigned quality $q _ { j }$ . Define $V ( 0 , 0 , 0 ) = 0$ and $V ( 0 , r , j ) = - \infty \forall ( r , j ) \neq ( 0 , 0 )$ We claim Rev $\cdot [ k , r , j ] = V ( k , r , j ) \ \forall \ k , r , j .$

We will show this by an induction over k.

Base case is when $k = 0$ , by initialization we have equality.

Inductive step assume that Rev $[ k - 1 , \cdot , \cdot ] = V ( k - 1 , \cdot , \cdot )$ . The DP recursion gives us

$$
\mathtt { R e v } [ k , r , j ] = \operatorname* { m a x } _ { 0 \le \ell < r } \{ \mathtt { R e v } [ k - 1 , \ell , j ^ { \prime } ] + \mathcal { U } [ \ell , r , j ^ { \prime } , j ] \} = \operatorname* { m a x } _ { 0 \le \ell < r } \{ V ( k - 1 , \ell , j ^ { \prime } ) + \mathcal { U } [ \ell , r , j ^ { \prime } , j ] \}
$$

We will first show that Rev $\cdot [ k , r , j ] \leq V ( k , r , j ) \ \forall r , j$ . To do this, fix any $0 \leq \ell < r$ and $0 \leq j ^ { \prime } \leq j$ and let $0 = \ell _ { 0 } < . . . < \ell _ { k - 1 } = \ell$ be a partition of intervals $\left( k - 1 \right)$ with quality assignment $0 = j _ { 0 } \le . . . \le j _ { k - 1 } = j ^ { \prime }$ where the quantile interval indexed by $[ \ell _ { k - 2 } , \ell _ { k - 1 } ]$ is assigned quality indexed $j _ { k - 1 }$ and so on. Appending $( \ell , r ]$ at the end of the quantile interval gives us a valid valid partition as $\ell < r$ and assigning quality $q _ { j }$ to this interval is valid since it maintains monotonicity, such a scheme gives us revenue of $V ( k - 1 , \ell , j ^ { \prime } ) + \mathcal { U } [ \bar { \ell } , r , j ^ { \prime } , j ]$ By feasibility we have $V ( k - 1 , \ell , j ^ { \prime } ) + \mathcal { U } [ \ell , r , j ^ { \prime } , j ] \leq V ( k , r , j ) \ \forall 0 \leq \ell < r , 0 \leq j ^ { \prime } \leq j$ . Taking maximum on all such $\ell , j ^ { \prime }$ gives us Rev $\lceil k , r , j \rceil \leq V ( k , r , j ) \forall r , j$

Now we show that $V ( k , r , j ) \leq \mathtt { R e v } [ k , r , j ] \ \forall r , j$ . Consider the k partition $0 \leq \ell _ { 0 } < \ldots < \ell _ { k } = r$ and the quality assignment of $0 = j _ { 0 } \leq . . . \leq j _ { k } = j$ that obtains the value $V ( k , r , j )$ . Set $\ell = \ell _ { k - 1 }$ and $j ^ { \prime } = j _ { k - 1 }$ . Then the partition $0 = \ell _ { 0 } < \dots < \ell$ is a valid $\left( k - 1 \right)$ partition with feasible quality assingment $0 = j _ { 0 } \le \ldots \le j ^ { \prime }$ giving us $\begin{array} { r } { \sum _ { i = 1 } ^ { k - 1 } \mathcal { U } [ \ell _ { i - 1 } , \ell _ { i } , j _ { i - 1 } , j _ { i } ] \leq V ( k - 1 , \ell , j ^ { \prime } ) } \end{array}$ . Finally, by definition, we have

$$
V ( k , r , j ) = \sum _ { i = 1 } ^ { k - 1 } \mathcal { U } [ \ell _ { i - 1 } , \ell _ { i } , j _ { i - 1 } , j _ { i } ] + \mathcal { U } [ \ell , r , j ^ { \prime } , j ] \leq V ( k - 1 , \ell , j ^ { \prime } ) + \mathcal { U } [ \ell , r , j ^ { \prime } , j ] \leq \mathsf { R e v } [ k , r , j ]
$$

where the last inequality follows from the DP update.

Optimality of the partitions and quality assignment follows from the greedy backward pass, and optimality and IC of the menu follow from the payment formula.

Theorem 17. Let $\pmb { x } = \left( ( I , \pmb { t } ) , ( p _ { i } , q _ { i } ) _ { i \in [ I ] } \right)$ be the solution of $L _ { F }$ . Then the DP output $\pmb { x } ^ { \prime } = \big ( ( I ^ { \prime } , t ^ { \prime } ) , ( p _ { i } ^ { \prime } , q _ { i } ^ { \prime } ) _ { i \in [ I ] } \big )$ is such that $R e v ( { \pmb x } ^ { \prime } ; F ) \geq R e v ( { \pmb x } ; F ) - \varepsilon _ { 0 \mathsf { P T } }$ where the optimization error is given as $\varepsilon _ { 0 \mathsf { P T } } = I ( 6 \overline { { q } } + 1 + \overline { { q } } ) \sqrt { \varepsilon }$

Proof of Theorem 17. We use the following decomposition

$$
R e v ( { \bf x } ; F ) - R e v ( { \bf x } ^ { \prime } ; F ) = R e v ( { \bf x } ; F ) - R e v ( \tilde { \bf x } ; F ) + \underbrace { R e v ( \tilde { \bf x } ; F ) - R e v ( { \bf x } ^ { \prime } ; F ) } _ { \le 0 ~ \mathrm { ( b y ~ c o r r e c t n e s s ~ o f ~ n P ) } }
$$

where $\tilde { \boldsymbol { x } } \gets$ modification(x) such that for $\pmb { \tilde { x } } = ( ( I , \tilde { \pmb { t } } ) , ( \tilde { p } _ { i } , \tilde { q } _ { i } ) _ { i \in [ I ] } )$ where each component of t<sup>˜</sup> and each $\tilde { q } _ { i }$ lie on the grid $\mathcal { E } _ { t }$ and $\mathcal { E } _ { q }$ respectively, with ${ \tilde { p } } _ { i }$ defined later. In particular let $\begin{array} { r } { \tilde { q } _ { i } = \varepsilon \lfloor \frac { q _ { i } } { \varepsilon } \rfloor } \end{array}$ and $\begin{array} { r } { \tilde { t } _ { i } = \varepsilon \lfloor \frac { t _ { i } } { \varepsilon } \rfloor } \end{array}$ . Thus the monotonicity of $\tilde { q } _ { i }$ is preserved. Based on the modification, we have $\tilde { q } _ { i } \le q _ { i }$ and $q _ { i } - \varepsilon \le \tilde { q } _ { i }$ and similarly, $\tilde { t } _ { i } \leq t _ { i }$ and $t _ { i } - \varepsilon \le \tilde { t } _ { i }$ . Let $z _ { \tau } = \{ i \in [ I ] : t _ { i - 1 } - t _ { i } < \tau \}$

Consider $i \not \in z _ { \tau }$ . Then using Lemma 6 we get

$$
\begin{array} { r } { \tilde { q } _ { i } \cdot \mu ( \tilde { t } _ { i - 1 } , \tilde { t } _ { i } ) - p _ { i } \geq q _ { i } \cdot \mu ( t _ { i - 1 } , t _ { i } ) - p _ { i } - 3 \overline { { q } } \varepsilon / ( t _ { i - 1 } - t _ { i } ) - \varepsilon } \end{array}
$$

$$
\begin{array} { r l } & { \geq q _ { i } \cdot \mu ( t _ { i - 1 } , t _ { i } ) - p _ { i } - 3 \overline { { q } } \varepsilon / \tau - \varepsilon } \\ & { \geq \operatorname* { m a x } \{ 0 , \underset { j } { \operatorname* { m a x } } \{ q _ { j } \cdot \mu ( t _ { i - 1 } , t _ { i } ) - p _ { j } \} \} - 3 \overline { { q } } \varepsilon / \tau - \varepsilon } \\ & { \geq \operatorname* { m a x } \{ 0 , \underset { j } { \operatorname* { m a x } } \{ q _ { j } \cdot \mu ( \tilde { t } _ { i - 1 } , \tilde { t } _ { i } ) - p _ { j } \} \} - 6 \overline { { q } } \varepsilon / \tau - \varepsilon } \\ & { \geq \operatorname* { m a x } \{ 0 , \underset { j } { \operatorname* { m a x } } \{ \tilde { q } _ { j } \cdot \mu ( \tilde { t } _ { i - 1 } , \tilde { t } _ { i } ) - p _ { j } \} \} - 6 \overline { { q } } \varepsilon / \tau - \varepsilon } \end{array}
$$

Let $\tilde { p } _ { i } = p _ { i } - i \rho$ for $\rho = 6 \overline { { q } } \varepsilon / \tau + \varepsilon$ . Then observe that for all $j < i$

$$
\begin{array} { r l } & { \mu ( \tilde { t } _ { i - 1 } , \tilde { t } _ { i } ) \cdot \tilde { q } _ { i } - \tilde { p } _ { i } = \mu ( \tilde { t } _ { i - 1 } , \tilde { t } _ { i } ) \cdot \tilde { q } _ { i } - p _ { i } + i \rho } \\ & { \qquad \geq \mu ( \tilde { t } _ { i - 1 } , \tilde { t } _ { i } ) \cdot \tilde { q } _ { j } - p _ { j } + i \rho - \rho } \\ & { \qquad = \mu ( \tilde { t } _ { i - 1 } , \tilde { t } _ { i } ) \cdot \tilde { q } _ { j } - \tilde { p } _ { j } + ( i - j ) \rho - \rho } \\ & { \qquad \geq \mu ( \tilde { t } _ { i - 1 } , \tilde { t } _ { i } ) \cdot \tilde { q } _ { j } - \tilde { p } _ { j } } \end{array}
$$

Therefore, buyer upon receiving signal $i \not \in$ z<sub>τ</sub> agrees to pay at least $p _ { i } - I \rho ,$ generating a margin of $p _ { i } - c ( q _ { i } ) - I \rho .$ therefore,

$$
R e v ( { \pmb x } ; F ) - R e v ( \tilde { { \pmb x } } ; F ) \leq I ( \rho + \overline { { q } } \tau )
$$

where for upper bound we assume the worst case that buyers with signal $i \in z _ { \tau }$ do not purchase anything. Setting $\tau = { \sqrt { \varepsilon } }$ , we get

$$
R e v ( { \pmb x } ; F ) - R e v ( { \pmb x } ^ { \prime } ; F ) \leq I ( 6 \overline { { q } } + 1 + \overline { { q } } ) \sqrt { \varepsilon }
$$

Theorem 18. For discrete F with $\mathcal { O } ( m )$ atoms, Algorithm 2 has a runtime complexity of $\mathcal { O } \left( K \cdot m ^ { 2 } \cdot n _ { q } ^ { 2 } \right)$ which is $\mathcal { O } ( 1 / \varepsilon ^ { 4 } )$ for $m = \mathcal { O } ( 1 / \varepsilon )$ , and has memory complexity $\mathcal { O } ( K \cdot m \cdot n _ { q } )$

Proof of Thm 18. Complexity. Algorithm 2 preprocesses $A ( \cdot )$ at all $m + 1$ grid points in $\mathcal { O } ( m + N )$ time, so that each $\mu _ { F } ( \ell , r )$ is evaluated in $\mathcal { O } ( 1 )$ . The DP table has $K \cdot m \cdot n _ { q }$ entries where filling each entry requires maximizing over $O ( m \cdot n _ { q } )$ pairs $( \ell , j ^ { \prime } )$ , giving a total forward-pass runtime complexity of $\mathcal { O } \left( K \cdot m ^ { 2 } \cdot n _ { q } ^ { 2 } \right)$ The backward pass runs in $O ( K )$ with price computation in $O ( K ^ { 2 } )$ , and the signaling scheme in $\dot { O } ( K \cdot N )$ , all dominated by the forward pass. The grid $\{ t _ { 0 } , \ldots , t _ { m } \}$ can be chosen freely, choosing m points with maximum spacing $\Delta : = \mathrm { m a x } _ { i } ( t _ { i } - t _ { i - 1 } ) = \mathcal { O } ( \varepsilon )$ requires $m = \mathcal { O } ( 1 / \varepsilon )$ . With $n _ { q } = \mathcal { O } ( ( \overline { { q } } - \underline { { q } } ) / \varepsilon )$ , the total time is thus $\mathcal { O } ( K / \varepsilon ^ { 4 } )$ which polynomial in $1 / \varepsilon$ for fixed $K , \overline { { q } } ,$ and q.

Memory complexity is $\mathcal { O } ( K \cdot m \cdot n _ { q } )$ for the Rev and ptr tables.

Theorem 19 (Additive-FPTAS). For any $\varepsilon _ { 0 \mathsf { P T } } > 0$ , given a discrete distribution with support size $N _ { ; }$ Algorithm 2 runs in $\mathsf { p o l y } ( K , N , \overline { { q } } , 1 / \varepsilon _ { 0 \mathsf { P T } } )$ time and outputs a menu $\{ ( p _ { k } , q _ { j _ { k } } ) \} _ { k = 1 } ^ { k ^ { \star } }$ and a monotone paritional signaling scheme $\mathcal { T } = ( k ^ { \star } , ( t _ { r _ { k } } : k \in [ k ^ { \star } ] ) )$ ) with revenue exceeding $R e v ^ { \star } ( F ) - \varepsilon _ { 0 \mathsf { P T } }$ by setting the resolution of the grid as $\varepsilon = ( \varepsilon _ { 0 \mathsf { P T } } ) ^ { 2 } / ( K ( 6 \overline { { q } } + 1 + \overline { { q } } ) ) ^ { 2 }$

Proof. The proof follows by a combination of earlier results, namely, Theorems 16, 17, and 18. □

## E Equivalent Representations of Signaling Schemes

Lemma 7. Consider the following equivalent representation of the signaling scheme.

1. Quantile space representation: $\boldsymbol { S } = \{ r _ { i } \} _ { i = 1 } ^ { [ I ] }$ with the signaling scheme as $\pi ( r _ { i } | u ) = \mathbf { 1 } \{ u \in [ t _ { i - 1 } , t _ { i } ] \}$ for all $u \sim U n i f [ 0 , 1 ]$ where $t _ { 0 } = 1 - t _ { I } = 0$ with $t _ { i - 1 } \leq t _ { i }$ and $t _ { i } \in [ 0 , 1 ] \ \forall \ i \in [ I ]$

2. Value space representation: ${ \mathcal { S } } = \{ s _ { i } \} _ { i = 1 } ^ { [ I ] } w _ { 0 } \leq w _ { 1 } \leq \cdots \leq w _ { I } , \xi _ { i } \in [ 0 , 1 ]$ with $\xi _ { 0 } = 0 , \xi _ { I } = 1$ , and $\xi _ { i - 1 } \leq \xi _ { i }$ whenever $w _ { i - 1 } = w _ { i }$ , with the signaling scheme as

$$
\pi ( s _ { i } | v ) = { \left\{ \begin{array} { l l } { ( 1 - \xi _ { i - 1 } ) \mathbf { 1 } \{ v = w _ { i - 1 } \} + \mathbf { 1 } \{ w _ { i - 1 } < v < w _ { i } \} + \xi _ { i } \mathbf { 1 } \{ v = w _ { i } \} } & { { \mathrm { ~ } } i f { \mathrm { ~ } } w _ { i - 1 } < w _ { i } , } \\ { ( \xi _ { i } - \xi _ { i - 1 } ) \mathbf { 1 } \{ v = w _ { i - 1 } \} } & { { \mathrm { ~ } } i f { \mathrm { ~ } } w _ { i - 1 } = w _ { i } . } \end{array} \right. }
$$

for all $v \sim F$

Then for $t _ { i } = ( 1 - \xi _ { i } ) F ( w _ { i } ^ { - } ) + \xi _ { i } F ( w _ { i } )$ , the posterior mean under signal $s _ { i }$ and signal $r _ { i }$ is equal to $\textstyle \int _ { t _ { i - 1 } } ^ { t _ { i } } Q ( t ) \mathrm { d } t / ( t _ { i } - t _ { i - 1 } )$ where $Q ( t ) = \operatorname* { i n f } \{ v : F ( v ) \geq t \}$

Proof of Lemma 7. In the value space, it is a little messy and can be written as $\pi ( r _ { i } | v ) = \lambda ( [ F ( v ^ { - } ) , F ( v ) ] \cap$ $[ t _ { i - 1 } , t _ { i } ] ) / ( F ( v ) - F ( v ^ { - } ) )$ , where $\lambda ( \cdot )$ captures the length of the interval, concretely, a Lebesgue measure. The posterior expected value under $r _ { i }$ is $\mathtt { N / D }$ where

$$
\mathbb { N } = \int _ { 0 } ^ { 1 } Q ( u ) \mathbf { 1 } \{ u \in [ t _ { i - 1 } , t _ { i } ] \} \mathrm { d } u = \int _ { t _ { i - 1 } } ^ { t _ { i } } Q ( u ) \mathrm { d } u
$$

where we used $v = Q ( u )$ and

$$
\mathbb { D } = \int _ { 0 } ^ { 1 } \mathbf { 1 } \{ u \in [ t _ { i - 1 } , t _ { i } ] \} \mathrm { d } u = t _ { i } - t _ { i - 1 }
$$

Consider first the case when $w _ { i - 1 } < w _ { i }$ , this implies $F$ is continuous and strictly increasing on $( w _ { i - 1 } , w _ { i } )$ Then the posterior expected value under $s _ { i }$ is $\mathbb { N } ^ { \prime } / \mathbb { D } ^ { \prime }$ where

$$
\mathfrak { N ^ { \prime } } = \int _ { 0 } ^ { v } v \mathrm { d } F ( v ) = w _ { i - 1 } ( 1 - \xi _ { i - 1 } ) ( F ( w _ { i - 1 } ) - F ( w _ { i - 1 } ^ { - } ) ) + \int _ { ( w _ { i - 1 } , w _ { i } ) } v \mathrm { d } F ( v ) + w _ { i } \xi _ { i } ( F ( w _ { i } ) - F ( w _ { i } ^ { - } ) )
$$

Note by the definition of $Q ( \cdot )$ we have $Q ( t ) = w _ { i - 1 } \forall t \in [ t _ { i - 1 } , F ( w _ { i - 1 } ) ]$ and $Q ( t ) = w _ { i } \forall t \in [ F ( w _ { i } ^ { - } ) , t _ { i } ]$ Therefore $\begin{array} { r } { \int _ { ( w _ { i - 1 } , w _ { i } ) } v \mathrm { d } F ( v ) = \int _ { F ( w _ { i - 1 } ) } ^ { F ( w _ { i } ^ { - } ) } Q ( t ) \mathrm { d } t } \end{array}$ , substituting this in $\mathbb { N } ^ { \prime }$ and writing $\xi _ { i }$ in terms of $t _ { i }$ gives us $\mathbb { N } ^ { \prime } = \mathbb { N }$ . Also,

$$
\begin{array} { r l } { \displaystyle \mathbb { D } ^ { \prime } = \int _ { 0 } ^ { \overline { { v } } } \mathrm { d } F ( v ) = ( 1 - \xi _ { i - 1 } ) ( F ( w _ { i - 1 } ) - F ( w _ { i - 1 } ^ { - } ) ) + \int _ { ( w _ { i - 1 } , w _ { i } ) } \mathrm { d } F ( v ) + \xi _ { i } ( F ( w _ { i } ) - F ( w _ { i } ^ { - } ) ) } \\ { \displaystyle \qquad = ( 1 - \xi _ { i - 1 } ) ( F ( w _ { i - 1 } ) - F ( w _ { i - 1 } ^ { - } ) ) + ( F ( w _ { i } ^ { - } ) - F ( w _ { i - 1 } ) + \xi _ { i } ( F ( w _ { i } ) - F ( w _ { i } ^ { - } ) ) } \\ { \displaystyle \qquad = t _ { i } - t _ { i - 1 } = \mathbb { D } } \end{array}
$$

For the case when $w _ { i - 1 } = w _ { i } , t _ { i - 1 } , t _ { i } \in [ F ( w _ { i } ^ { - } ) , F ( w ) ]$ thus $Q ( t ) = w _ { i } \ \forall t \in [ t _ { i - 1 } , t _ { i } ]$ . Thus $\begin{array} { r l } { \int _ { 0 } ^ { \bar { v } } v \mathrm { d } F ( v ) = } & { { } } \end{array}$ $w _ { i } ( \xi _ { i } - \xi _ { i - 1 } ) ( F ( w _ { i } ) - F ( w _ { i } ^ { - } ) )$ and $\begin{array} { r } { \int _ { 0 } ^ { \bar { v } } \mathrm { d } F ( v ) = ( \xi _ { i } - \xi _ { i - 1 } ) ( F ( w _ { i } ) - F ( w _ { i } ^ { - } ) ) } \end{array}$ ), which writing $\xi _ { i }$ in terms of $t _ { i }$ gives the desired result. □