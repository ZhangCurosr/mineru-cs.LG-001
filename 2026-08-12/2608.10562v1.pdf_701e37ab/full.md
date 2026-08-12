# MARCO: CLICK-INTENT DECOMPOSITION FOR CALIBRATED ADS CONVERSION PREDICTION

Shiwen Shen<sup>1</sup>, Xiru Huang<sup>1</sup>, Liang Luo<sup>1</sup>, Jianbo Sun<sup>1</sup>, He Lyu<sup>1</sup>, Zihang Fu<sup>1</sup>, Ivonne Xu<sup>1</sup>, Zhizhuo Li<sup>1</sup>, Zhengyu Zhang<sup>1</sup>, Pei-Ju Sung<sup>1</sup>, Yunmiao Wang<sup>1</sup>, Zixuan Wang<sup>1</sup>, Zhengli Zhao<sup>1</sup>, Qiang Jin<sup>1</sup>, Mike Jermann<sup>1</sup>, Mingda Li<sup>1</sup>, Yang Xiao<sup>1</sup>, Bhavana Challa<sup>1</sup>, Brooke Bian<sup>1</sup>, Yang Li<sup>1</sup>, Ashish Chamoli<sup>1</sup>, Bibek Bhusal<sup>1</sup>, Danning Di<sup>1</sup>, Yuan Jin<sup>1</sup>, Meet Raval<sup>1</sup>, Zhiwen Chen<sup>1</sup>, Boyao Sun<sup>1</sup>, Shuguang Wang<sup>1</sup>, Yunlong He<sup>1</sup>, Yantao Yao<sup>1</sup>, Sagar Chordia<sup>1</sup>, Wenlin Chen<sup>1</sup>, Santanu Kolay<sup>1</sup>, Qin Huang<sup>1</sup>, and Ellie Wen<sup>1</sup>

<sup>1</sup>Meta AI, Menlo Park, California, USA

## ABSTRACT

Not all clicks are equal. Industrial ads ranking decouples conversion probability into click-through rate (CTR) and post-click conversion rate (CVR), yet treats every click as the same event. In reality, users provide a free, self-generated signal of intent through their physical UI interactions. Different click types on the same ad exhibit a 4 difference in actual conversion rates. By conflating these signals, the standard CVR model systematically under-predicts high-intent clicks and over-predicts low-intent ones, which is a severe bias masked by near-perfect aggregate calibration.

We propose MARCO (Multi-intent Ads Ranking Composition Optimization), a framework that resolves this bias by decomposing each click by intent. Using the logged click type as a free behavioral label, MARCO trains per-intent CVR heads on homogeneous populations, and at serving time composes their per-intent CVR estimates under a predicted distribution over intents. Theoretically, we prove that decomposition never raises population risk, give the exact headroom under squared loss and non-negativity under the deployed loss, and show through a routing-efficiency dial how much of it reaches serving. Because the population-optimal score is unchanged, any gain is a finite-capacity estimation and calibration effect that we validated both offline and online. For deployment at scale, we further cast multi-impression, multi-click attribution as credit assignment with a bias-variance tradeoff analogous to RL return estimation, showing last-impression, first-click attribution is the low bias, low-variance, deterministic choice under production constraints, and derive three consistency conditions enforced end-to-end at scale.

Deployed at binary intent granularity, MARCO corrects per-intent calibration to approximately 100%, lifts conversions per click by +2.80%, and drives +0.98% cumulative improvement in topline metrics.

Keywords computational advertising, conversion rate prediction, click-through rate prediction, model calibration, click-intent decomposition, prediction composition, credit attribution

## 1 Introduction

Industrial ads ranking systems estimate impression conversion probability by factorizing it into a click-through rate (CTR) and a post-click conversion rate (CVR) as an exact marginal factorization [1, 2]:

$$
\Phi _ { \mathrm { s t d } } ( x ) = \lambda ( x ) \cdot \mu ( x ) ,\tag{1}
$$

where x denotes impression-time features, $\lambda ( x ) : = P ( \operatorname { c l i c k } \mid \operatorname* { i m p } , x )$ is the CTR, $\mu ( x ) : = P ( \mathrm { c o n v } \mid \mathrm { c l i c k } , x )$ is the CVR, and $\Phi _ { \mathrm { s t d } } ( x ) : = P ( \mathrm { c o n v } \mid \mathrm { i m p } , x )$ is the conversion score used for auction ranking. Each component is estimated by a dedicated machine learning model—a compositional approach that has served as the industry standard for over a decade.

The primary operational challenge lies in CVR estimation: predicting conversion rates across a mixed click population is difficult due to substantial variance across interaction types. High-intent actions convert at elevated rates, whereas low-intent actions convert far less frequently. Compressing this heterogeneous click mixture into a single scalar prediction introduces group-conditional bias that no finite-capacity model can resolve. Conversely, partitioning clicks into homogeneous intent strata improves estimation efficiency, leveraging foundational principles of post-stratified estimation [3, 4]. Unlike classical post-stratification, the intent stratum in ad ranking is a post-impression outcome unobserved during scoring. We address this by predicting the intent distribution at inference time and routing per-stratum estimates through the predicted distribution

Standard interaction logs already contain the signal required to decouple user intent. Modern social ads feature distinct UI surfaces: call-to-action (CTA) taps navigate off-platform, whereas social interactions (e.g., likes, comments) reflect lightweight on-platform engagement. Conventional architectures collapse these behaviors into a single click label, leaving a zero-cost supervision signal unexploited. MARCO instead leverages logged click types as free supervision for intent. Empirically, high-intent CTA taps convert at nearly 4 the rate of social actions—a gap single-head models cannot capture.

Ignoring this signal forces a single CVR head to fit distinct conversion funnels, causing systematic subgroup miscalibra tion: under-predicting high-intent traffic while over-predicting low-intent traffic. This directly degrades ranking, as inflated low-intent scores displace higher-converting candidates in the auction. Because these opposing errors cancel in aggregate, standard monitoring tools report healthy overall calibration while masking severe internal distortion. We term this silent failure mode click-intent heterogeneity and formalize its observability limit in Section 2.

While prior literature addresses adjacent modeling challenges, no existing framework restructures the CTR-CVR composition formula to isolate heterogeneous click populations. Vertical funnel methods [1, 2, 5, 6, 7] decompose the user journey after the click. Multi-task learning and mixture-of-experts (MoE) architectures [8, 9, 10] optimize routing across shared experts, yet ultimately blend representations into a single prediction over the mixed click population. Expert gating modifies architectural capacity rather than the supervised target, leaving the underlying label distribution unaltered for each task. Post-hoc calibration techniques [11, 12, 13, 14] cannot condition on click intent, as the specific interaction type is a post-impression event unobservable at inference time. While intent models [15] enrich feature representations, they leave the underlying prediction composition unchanged.

We propose MARCO (Multi-intent Ads Ranking Composition Optimization), a framework that resolves click-intent heterogeneity through structural intent decomposition. MARCO partitions click events into intent-differentiated subcategories, estimating dedicated CTR and CVR predictions over more homogeneous sub-populations. At training time, MARCO uses the logged click type as a free supervision label; at serving time, it predicts the latent intent distribution across categories. By reformulating the prediction composition, MARCO eliminates group-conditional miscalibration while remaining orthogonal to and fully composable with existing CVR architectures. Our contributions are:

1. Problem Formulation & Observability Limit. We identify click-intent heterogeneity as a root cause of systematic miscalibration that conventional architectures cannot eliminate, and prove that a single impression-time prediction is inherently incapable of achieving per-intent calibration (§2).

2. Free Supervision & Mechanism Isolation. We leverage logged click types as zero-cost behavioral supervision labels. Through controlled ablations, we demonstrate that single-output architectures consistently fail to eliminate per-intent miscalibration, even when augmented with auxiliary intent tasks, historical features, or Mixture-of-Experts (MoE) gating. Conversely, exposing per-intent outputs via output-layer decomposition eliminates over 99% of calibration error with zero annotation cost and negligible parameter overhead (§6).

3. Production System & Industrial Impact. We articulate the end-to-end system design, including a credit-attribution design and the cross-pipeline consistency conditions the system must satisfy (§5). Deployed at scale across Meta’s ad platforms under binary intent granularity, MARCO increases conversions per click by +2.80% in a production holdback and delivers a +0.98% cumulative topline metric lift across successive launches (§6).

4. Theoretical Foundation & Routing Efficiency. We formalize intent decomposition as a population-level class enlargement, proving weak dominance over pooled models and establishing generic strictly positive headroom ∆ . Furthermore, we introduce the routing-efficiency parameter η to quantify the fraction of theoretical headroom realized at inference (§3).

The remainder of this paper is organized as follows. Section 2 formalizes the click-intent heterogeneity problem. Section 3 establishes the theoretical framing and headroom analysis for MARCO. Section 4 details the proposed MARCO model, while Section 5 describes the credit attribution design and end-to-end system architecture. Section 6 presents our offline evaluations and online A/B testing results. All mathematical proofs are deferred to Appendix A.

![](images/90d1a2521eae86f0df0a8ed22f331c02f20272ae8a91e3cd992c9d70afdbe2e4.jpg)  
Figure 1: The MARCO ads-ranking system. Gray marks the unchanged baseline and teal marks what MARCO adds.

## 2 Click-Intent Heterogeneity in Ads Ranking

Recall from Eq. 1 that conventional ads ranking scores an impression via $\Phi _ { \mathrm { s t d } } ( x ) = \lambda ( x ) \cdot \mu ( x )$ , relying on a single pooled conversion head $\mu ( x )$ across all click types. To capture this intent variation, we partition the click space into K mutually exclusive and collectively exhaustive intent categories covering all click types, indexed by $k \in \{ 1 , \ldots , K \}$ grouped by historical click-type conversion likelihood. Intent routing relies strictly on logged interaction types rather than conversion outcomes, eliminating target leakage.

For impression features x, we define the per-intent CTR $\lambda _ { k } ( x ) : = P ( \mathrm { c l i c k } _ { k } \mid \mathrm { i m p } , x )$ , per-intent CVR $\mu _ { k } ( x ) : =$ P(conv click<sub>k</sub>, x), and intent distribution $\pi _ { k } ( x ) : = P ( k \mid$ click, x). By the law of total probability, the overall impression conversion probability estimated by MARCO is:

$$
\Phi _ { \mathrm { M A R C O } } ( x ) = \sum _ { k = 1 } ^ { K } \lambda _ { k } ( x ) \mu _ { k } ( x ) .\tag{2}
$$

While $\Phi _ { \mathrm { M A R C O } } ( x )$ mathematically generalizes to arbitrary K, our evaluation and deployment focus on the binary instantiation $( K = 2 )$ , capturing the dominant contrast between off-platform CTA taps and on-platform social engagements. Throughout the remainder ofthis paper, we drop the explicitfeature argument (x)for brevity whenever feature conditioning is clearfrom context.

We evaluate prediction quality using the calibration ratio (predicted over observed conversion rate). Because a pooled head µ outputs a single scalar across heterogeneous clicks, opposing errors from under-predicting high-intent traffic and over-predicting low-intent traffic cancel in aggregate, masking severe subgroup bias behind a deceptively healthy overall ratio.

Existing ads ranking architectures (the gray baseline path in Figure 1) are structurally blind to per-intent bias across every ranking stage:

1. Attribution System: Clicks and conversions are joined without distinguishing interaction types, generating pooled training targets.

2. Model Training: CTR and CVR heads optimize a single loss over mixed click traffic, ignoring intent differentiation.

3. Calibration Service: Calibration relies strictly on impression-time features, leaving unobserved post-impression intent miscalibrated beneath aggregate cancellation.

4. Prediction Composition: Pooled estimates are composed via Eq. 1, propagating single-click bias into auction scores.

This limitation represents a structural observability barrier rather than a model capacity constraint. As long as a ranking system outputs a single scalar prediction µ prior to interaction observation, no added capacity, architectural gating, or post-hoc recalibration can achieve simultaneous group-conditional calibration. We formalize this fundamental limit in Proposition 1.

Proposition 1 (Per-intent calibration is unattainable by a single output before the click occurs) Before the click intent is observed, under a proper scoring rule and a fixed impression-timefeature set, no model emitting a single CVR value µ can be simultaneously calibratedfor all intent categories whenever the per-intent $C V R s \mu _ { k }$ differ. It equals at most one $\mu _ { k }$

Section 3 formalizes this mechanism as a class-enlargement headroom, proving it is non-negative and generically strictly positive. Section 6 directly quantifies this headroom on held-out benchmarks.

## 3 Theoretical Analysis

We frame intent decomposition as a population-level class enlargement $( { \mathcal { F } } \subseteq { \mathcal { G } } )$ and establish three key theoretical guarantees:

(1) Weak Dominance: The intent-decomposed class never yields higher risk than the pooled class at the population optimum (Theorem 1(i)).

(2) Genericity: Strict improvement over pooled models occurs almost surely across parameter space (Proposition 4).

(3) Routing Monotonicity: Expanding CTR model capacity provably realizes a larger fraction of the theoretical headroom (Proposition 7).

These guarantees characterize the best achievable risk within each hypothesis class. Because the pooled and decomposed scores coincide at the true Bayes optimum $( \Phi _ { \mathrm { M A R C O } } = \Phi _ { \mathrm { s t c } }$ , Lemma 1), any realized gain is strictly a finite-capacity estimation and calibration effect.

Class Enlargement and Headroom $( \Delta _ { \mathcal { F } } )$ . Let $\mathcal { F }$ denote the hypothesis class of a single pooled conversion head, and let $\begin{array} { r } { \mathcal { G } = \{ g = \sum _ { k = 1 } ^ { K } \pi _ { k } f _ { k } : f _ { k } \in \mathcal { F } \} } \end{array}$ denote the intent-decomposed class. We define the theoretical headroom opened by decomposition as the risk gap between the best-in-class pooled model and the best-in-class decomposed model:

$$
\Delta _ { \mathcal { F } } : = \operatorname* { i n f } _ { f \in \mathcal { F } } \mathcal { R } ( f ) - \operatorname* { i n f } _ { g \in \mathcal { G } } \mathcal { R } ( g )\tag{3}
$$

Because ${ \mathcal { F } } \subseteq { \mathcal { G } }$ , weak dominance $( \Delta \ v { r } ) \stackrel { } { = } 0 )$ holds universally regardless of backbone architecture, feature representations, or loss function (Theorem 1(i)). Under squared loss, $\Delta { _ { \mathcal { F } } }$ resolves to an exact $L ^ { 2 }$ projection energy $\| P _ { \mathcal { G } } \mu - P _ { \mathcal { F } } \mu \| ^ { 2 } \ge 0$ (Proposition 3). Under the deployed Normalized Entropy metric, while non-linearity prevents a closed-form expression, weak dominance $( \Delta \ v { r } \geq 0 )$ holds unconditionally.

Furthermore, Proposition 4 establishes that parameter configurations yielding zero headroom $( \Delta \ v { r } _ { \mathcal { F } } = 0 )$ form a set of Lebesgue measure zero. Thus, strict headroom $( \Delta \ v { r } > 0 )$ is generic, i.e., decomposition strictly improves population risk whenever per-intent conversion rates differ. In Section 6, we directly measure this headroom on held-out production traffic at $K = 2$

Routing Efficiency and CTR Capacity. The theoretical headroom $\Delta _ { \mathcal { F } }$ is a CVR-side quantity. How much of this potential reaches inference depends on the intent distribution $\hat { \pi } _ { k } = \hat { \lambda } _ { k } / \hat { \lambda }$ predicted by the CTR model. We formalize this transfer via the routing efficiency:

$$
\hat { \eta } : = 1 - \frac { \varepsilon _ { \mathrm { r o u t e } } } { \Delta _ { \mathcal { F } } }\tag{4}
$$

where $\varepsilon _ { \mathrm { r o u t e } }$ represents the heterogeneity-weighted routing cost incurred by imperfect intent predictions (Lemma 11). The routing cost obeys $\varepsilon _ { \mathrm { r o u t e } } \leq \mathbb { E } [ \tilde { \lVert \boldsymbol { \pi } } - \dot { \boldsymbol { \pi } } \rVert ^ { 2 } s ^ { 2 } ]$ , where $\begin{array} { r } { s ^ { 2 } \stackrel { \smile } { = } \sum _ { k } ( \mu _ { k } - \mu ) ^ { 2 } } \end{array}$ measures intent variance. This cost vanishes where intent is homogeneous $( s = 0 )$ and concentrates strictly where per-intent conversion rates diverge.

In Proposition 7, we prove that expanding CTR model capacity expands the realizable routing class $\Pi ^ { ( c ) }$ , monotonically reducing routing error $\varepsilon _ { \mathrm { r o u t e } } ^ { * }$ and increasing serving efficiency ηˆ. Consequently, larger CTR models unlock a higher fraction of the CVR headroom—a theoretical property directly validated by our offline capacity sweeps in Section 6.

![](images/75a5c2b1641f2e971da9df53e755778bc6d3a8f6cf90b4ead4c6c530c1a6bbde.jpg)  
Figure 2: Model architecture: Standard vs. MARCO.

## 4 The MARCO Model

Guided by our theoretical framing, MARCO introduces a minimal architectural modification to capture per-intent estimation gains while remaining strictly constrained in parameter and latency budgets. Rather than predicting a single scalar CVR, MARCO estimates per-intent CTR and CVR vectors and composes the total conversion probability Φ<sub>MARCO</sub> via the law of total probability as formulated in Eq. 2. Estimating each $\mu _ { k }$ on a homogeneous click subpopulation eliminates the label conflation that causes systematic group-conditional bias. In our production platform, we instantiate MARCO at K = 2.

Free Behavioral Supervision. MARCO leverages logged user interaction types as zero-cost supervision targets. Rather than requiring manual annotation, the intent label for each converted click is derived directly from physical UI behavior, serving as a behavioral pretext signal for latent intent. At training time, each per-intent CVR head $\mu _ { k }$ is trained exclusively on clicks belonging to intent category k (attributed via Section 5). At serving time, because the true click intent is unobserved prior to user interaction, MARCO evaluates all per-intent CVR heads and weights them by the predicted intent distribution $\hat { \pi } _ { k } = \hat { \lambda } _ { k } / \hat { \lambda }$ , computing the expected CVR over predicted latent intents.

Vector-Head Architecture. As shown in Figure 2, MARCO implements this decomposition via multi-output task heads without altering the underlying feature representations or model backbones. Shared embedding layers and deep neural network (DNN) representations remain intact for both CTR and CVR models. Alongside the baseline scalar projection, each model appends a K-dimensional vector head $( \mathbb { R } ^ { d } \to \mathbb { R } ^ { K }$ , where d denotes the final backbone representation dimension). The CTR head emits the complete per-intent click rate vector $( \lambda _ { 1 } , \ldots , \lambda _ { K } )$ , while the CVR head emits the corresponding conversion rate vector $( \mu _ { 1 } , \ldots , \mu _ { K } )$ . The original scalar output is retained strictly as an operational baseline for automated fallback (Section 5.3).

Parameter Efficiency & Operational Overhead. Constructing K separate dedicated models would significantly inflate parameter footprint, memory consumption, and serving latency, causing severe model fragmentation [16]. By expanding scalar projection heads into K-vector heads within existing backbones, MARCO adds fewer than $1 0 ^ { - 7 } \%$ parameters relative to total model capacity. Because representation layers and embedding lookups are shared, serving latency remains entirely unaffected while successfully restoring per-intent calibration.

System Integration. MARCO operates within an end-to-end industrial ranking architecture (Figure 1), where data logging, credit attribution, model training, online serving, and real-time calibration are managed by independent infrastructure components. For the composite score in Eq. 2 to remain semantically coherent, all pipeline stages must maintain strict agreement on intent category assignments—a system requirement we formalize and enforce in Section 5.

Table 1: Attribution design space and its qualitative rationale: bias, variance, real-time feasibility, and RL creditassignment analogs.
<table><tr><td></td><td></td><td>Imp. Click Bias Var.</td><td></td><td></td><td>Feasibility</td><td>RL analog</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>First-touch First First High Low Deterministic TD(0)</td><td></td></tr><tr><td>Time-decay Wtd. Wtd. Low Med. Req. full hist. GAE(λ)</td><td></td><td></td><td></td><td></td><td></td><td>Last-touch Last Last Low High Req. buffering Monte Carlo</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>MARCO Last First Low Low Deterministic -</td><td></td></tr></table>

## 5 Attribution and System Design

Deploying the decomposed composition Eq. 2 in a distributed industrial ranking environment requires all logging, training, serving, and calibration pipelines to maintain strict agreement on which interaction defines the credited click intent for every impression. This section formalizes the credit attribution rationale (§5.1), establishes the cross-pipeline consistency invariants (§5.2), and details the production infrastructure built to enforce them at scale (§5.3).

## 5.1 Credit Assignment Framework

In conversion-optimized advertising, a user’s pre-conversion journey spans multiple impressions and multi-touch interaction sequences. We model this trajectory as an ordered history H of n impressions, where each impression imp logged at render time $\tau _ { i }$ contains a multiset $C _ { i }$ of $m _ { i } \geq 0$ within-impression clicks logged at times $t _ { i , j } \mathrm { : }$

$$
H = \left[ \left( \operatorname* { i m p } _ { i } , \tau _ { i } , C _ { i } \right) \right] _ { i = 1 } ^ { n } \longrightarrow \mathrm { c o n v } , \quad \mathrm { w h e r e } ~ C _ { i } = \big \{ \left( \mathrm { c l i c k } _ { i , j } , t _ { i , j } \right) \big \} _ { j = 1 } ^ { m _ { i } }\tag{5}
$$

with $\tau _ { 1 } < \tau _ { 2 } < \cdots < \tau _ { n }$ and $\tau _ { i } \leq t _ { i , 1 } \leq t _ { i , 2 } \leq \cdot \cdot \cdot \leq t _ { i , m _ { i } }$ . Because a single rendered ad unit can generate a sequence of heterogeneous interactions $( \mathrm { e . g . }$ , a social like at $t _ { i , 1 }$ followed by a CTA tap at $t _ { i , 3 } ) _ { \mathrm { : } }$ , credit attribution must assign two non-negative weight vectors: a cross-impression weight $w _ { i }$ and a within-impression click weight $v _ { i , j }$ . The credited conversion outcome is distributed across training tuples via the joint weight $w _ { i } \cdot v _ { i , j } ,$ yielding the per-impression training target decomposition:

$$
{ \bar { \Phi } } ( H ) = \sum _ { i = 1 } ^ { n } w _ { i } \sum _ { j = 1 } ^ { m _ { i } } v _ { i , j } \cdot P ( { \mathrm { c o n v } } \mid { \mathrm { i m p } } _ { i } , { \mathrm { c l i c k } } _ { i , j } )\tag{6}
$$

enforcing $\textstyle \sum _ { i } w _ { i } = 1$ and $\textstyle \sum _ { j } v _ { i , j } = 1$ to guarantee total conversion mass.

Bias-Variance Tradeoff & RL Analog. Selecting $( w _ { i } , v _ { i , j } )$ represents a structural bias-variance tradeoff analogous to temporal-difference return estimation in reinforcement learning. As summarized in Table 1, impressions and clicks serve distinct causal roles:

• Impression Axis (Bias Control): An impression is the external stimulus presented to the user. Weighting towards the last impression $( w _ { n } = 1 )$ isolates the causally proximal context that directly triggered conversion, minimizing context attribution bias (analogous to Monte Carlo return estimation).

• Click Axis (Variance Control): A click sequence within a fixed impression represents a stochastic exploration process. Selecting the first click $( v _ { i , 1 } = 1 )$ captures the user’s immediate intent signature before subsequent taps inject behavioral noise, minimizing intent estimation variance (analogous to TD(0) temporal-difference learning).

Production Feasibility Constraints. MARCO’s credit-assignment policy balances streaming feasibility on the click axis with context proximity on the impression axis. The consistency conditions of Section 5.2 mandate that credited intent be deterministic and instantly resolvable without event buffering, ruling out last-click and time-decay rules in favor of first-click attribution. This choice carries minimal empirical risk: across the vast majority of impressions, initial and subsequent clicks share the same intent category, ensuring high intent alignment regardless of the click-axis rule. On the impression axis, we select the last impression to minimize context attribution bias, as it represents the causally proximal stimulus. The resulting last-impression, first-click policy is both statistically robust and instantly resolvable at serving time.

## 5.2 Cross-Pipeline Consistency Conditions

Because CTR and CVR training and calibration operate as independent distributed services, all four pipelines must resolve to the exact same credited click intent $\kappa ( i ) \in \{ 1 , \ldots , \dot { K } \}$ for every impression $i \in I .$ . Training consistency requires $\kappa _ { \mathrm { C T R } } ^ { \mathrm { t r a i n } } ( i ) = \kappa _ { \mathrm { C V R } } ^ { \mathrm { t r a i n } } ( i )$ , calibration consistency requires $\kappa _ { \mathrm { C T R } } ^ { \mathrm { c a l i } } ( i ) = \kappa _ { \mathrm { C V R } } ^ { \mathrm { c a l i } } ( i )$ , and the end-to-end invariant requires all four to agree as $\kappa _ { \mathrm { C T R } } ^ { \mathrm { t r a i n } } ( i ) = \kappa _ { \mathrm { C V R } } ^ { \mathrm { t r a i n } } ( i ) = \kappa _ { \mathrm { C T R } } ^ { \mathrm { c a l i } } ( i ) = \kappa _ { \mathrm { C V R } } ^ { \mathrm { c a l i } } ( i )$ . The first two align the training and calibration stages internally, but they do not link training to calibration. The shared single-source dedup cache of Section 5.3 supplies that link and enforces the invariant by construction.

## 5.3 Production System Engineering

To enforce global consistency across distributed streaming environments (Figure 1), MARCO deploys two core infrastructure services: a Click Metadata Table recording event-level payloads and a low-latency, durable key-value Deduplication Cache.

Concurrent Click Attribution. When interaction events arrive concurrently across distributed edge servers, the engine executes an atomicfirst-write-wins check on the Deduplication Cache. The earliest click event within the last attributed impression is registered as the credited intent $\kappa ( i )$ , while subsequent clicks are marked as deduplicated. This guarantees key uniqueness and write-atomicity under heavy write concurrency.

Schema-Agnostic Conversion Attribution. Offsite conversions arrive via asynchronous pipelines with heterogeneous user identifier schemas. When a conversion event triggers a lookup, the attribution system queries the Deduplication Cache using whichever identifier is present. Because all identifier paths resolve to the unified underlying cache entry established during click processing, all downstream consumers (CTR/CVR trainers and calibrators) observe the identical intent label, fulfilling consistency conditions of Section 5.2.

Composition-Level Fallback Guardrail. To protect auction delivery against model dilution or upstream prediction outages in concurrent A/B testing environments, MARCO incorporates a zero-latency fallback operator:

$$
\Phi = \Phi _ { \mathrm { M A R C O } } + \mathbb { I } ( \Phi _ { \mathrm { M A R C O } } = 0 ) \cdot \Phi _ { \mathrm { s t d } }\tag{7}
$$

If experimental dilution causes intent-specific predictions to vanish $( \mathbf { e . g . , \Phi _ { M A R C O } = 0 } )$ , the system reverts to the standard scalar score $\Phi _ { \mathrm { s t d } } = \lambda \mu$ , guaranteeing no-worse-than-baseline robustness.

Cold-Start Protocol. MARCO addresses two operational cold-start scenarios:

1. Unseen Click Types: New UI interaction features default to the low-intent group, a low-risk strategy since high-intent actions represent established, high-conversion navigation paths.

2. Unwarmed Calibration Services: To resolve the circular dependency where the real-time calibrator requires prediction traffic before MARCO goes live, we deploy a two-phase dummy composition: $\Phi _ { \mathrm { p h a s e 1 } } = \Phi _ { \mathrm { s t d } } { \bf \bar { \Psi } } + 0$ $\Phi _ { \mathrm { M A R C O } }$ . In Phase 1, scores remain identical to production baseline while intent-specific heads generate shadow predictions to warm up the calibration service. Once calibration variance converges (typically 2–3 days), Phase 2 activates the full MARCO composition (Eq. 7) with zero delivery perturbation.

## 6 Experiments

We evaluate MARCO across both offline log benchmarks and large-scale online A/B experiments on Meta’s production advertising platforms. All experiments instantiate the framework at binary intent granularity $( K = 2 )$ , capturing the primary contrast based on conversion likelihood. Online experiments span multi-week evaluation windows covering hundreds of millions of users with user-level randomization. All reported online gains are statistically significant at $p < 0 . 0 5$ under 95% confidence intervals. Automated fallback (§5.3) triggers on $< 0 . 1 \%$ of traffic, ensuring near-complete treatment exposure. Our empirical evaluation addresses four primary Research Questions:

• RQ1 Per-Intent Calibration: Does the observability barrier of Proposition 1 hold in practice? Do progressively stronger single-output methods such as auxiliary tasks, historical features, and MoE gating remain unable to reduce per-intent miscalibration, while a model that exposes per-intent outputs, such as MARCO, eliminates it?

• RQ2 Composed Post-Impression NE: Does MARCO improve end-to-end post-impression conversion prediction, and do empirical results align with theoretical headroom $( \Delta _ { \mathcal { F } } )$ and routing efficiency (ηˆ)?

• RQ3 Pre-Impression Funnel Recall: Does restoring per-intent calibration improve candidate retrieval fidelity in early ranking stages?

• RQ4 Online Business Impact: What is the topline business impact in live auctions, and how does score separation alter candidate ranking?

## 6.1 Offline Evaluation

## 6.1.1 Post-click: per-intent calibration (RQ1)

We compare MARCO against four progressively stronger paradigms for utilizing intent signals short of structural decomposition:

1. Baseline (Pooled CVR): An ESMM-style vertical funnel model [1, 2] fitting a single conversion head over all clicks.

2. + Auxiliary Intent Tasks: Appends auxiliary task heads to predict per-intent distribution, testing whether auxiliary representation learning resolves pooled head miscalibration.

3. + Historical Intent Feature: Feeds historical intent distributions as impression-time input features—the strongest intent signal observable prior to interaction (Proposition 1).

4. + MoE on Historical Intent: Deploys a Mixture-of-Experts (MoE) architecture gated on historical intent features, testing whether architectural gating eliminates subgroup bias.

Table 2: Per-intent calibration-error reduction relative to the pooled baseline (%), higher is better. Single-output architectures are bounded by Proposition 1, whereas MARCO eliminates over 99% of per-intent error.
<table><tr><td>Model</td><td>High intent</td><td>Low intent</td></tr><tr><td>Baseline (pooled)</td><td>0%</td><td>0%</td></tr><tr><td>+ Auxiliary intent task</td><td>0%</td><td>0%</td></tr><tr><td>+ Historical intent feature</td><td>1%</td><td>3%</td></tr><tr><td>+ MoE on historical intent</td><td>2%</td><td>8%</td></tr><tr><td>MARCO (decomposition)</td><td> $> { \bf 9 9 \% }$ </td><td> $> { \bf 9 9 \% }$ </td></tr></table>

Table 2 presents the evaluation ladder. Extra capacity through the auxiliary task leaves per-intent calibration unchanged (0%). Historical intent features and MoE gating yield marginal improvements $( \leq 8 \% )$ ). Because these baselines emit a single output $\mu$ for a mixed click population, Proposition 1 imposes a strict observational upper bound: a single prediction cannot simultaneously match distinct conditional targets $\mu _ { k }$ . MARCO removes essentially all of the per-intent calibration error, more than an order of magnitude above the nearest alternative, so the effect is unambiguous.

This evaluation ladder focuses on single-output architectures to empirically validate the observability limit in Proposition 1. Because these models emit a single prediction prior to interaction, their calibration ceiling remains fixed regardless of added tasks, historical features, or MoE gating. While any architecture with per-intent output heads can bypass this barrier, MARCO provides the minimal, zero-latency design needed at industrial scale.

## 6.1.2 Composed Post-impression NE (RQ2)

The post-impression evaluation assesses the end-to-end conversion prediction Φ across the global impression space, composing CTR and CVR scores via Eq. 1 for the baseline and Eq. 2 for MARCO. We evaluate offline performance on production data [17] across a 7-day evaluation window. Table 3 reports post-impression Normalized Entropy gains across a comprehensive evaluation matrix, systematically varying CTR backbone capacity (Base, Large 5 , and an Oracle router evaluating intent routing under ground-truth $P ( k { \bar { \ } }$ click) while using predicted click rates), composition architecture (MARCO vs CTR-split-only), and calibration adjustment strategies (None, CVR-only, CTR-only, and joint CTR & CVR).

Table 3 reveals four key findings directly validating our theoretical framework:

1. RQ2.1 Empirical Weak Dominance: MARCO strictly outperforms the pooled baseline in post-impression NE across all model capacities and calibration settings in Table $3 ( \mathrm { \bar { e } . g . , + 0 . 0 5 4 \mathrm { \% } }$ pre-calibration under Base CTR). This universal, strictly positive lift empirically validates Weak Dominance $( \Delta \varepsilon \geq 0$ , Theorem 1(i)), confirming that structural intent decomposition expands the hypothesis class without inflating population risk.

2. RQ2.2 Calibration Service Synergy: Pre-calibration gains jump by over 4 when passing through the online calibration service (+0.054% to +0.223% for Base CTR under joint CTR & CVR calibration). Isolating individual calibration components reveals that CVR-only calibration accounts for the majority of this lift (+0.181%), whereas CTR-only calibration contributes +0.097%. Unmixing per-intent outputs allows CVR calibrators to adjust each intent head against its homogeneous sub-population, resolving group-conditional errors that aggregate calibrators are structurally blind to (Proposition 1).

Table 3: Composed-model post-impression NE gain across evaluation settings (%), higher is better, with 95% confidence intervals reported against standard composition.
<table><tr><td>Related RQ</td><td>Composition Logic</td><td>Calibration Adjustment</td><td>NE Gain (%) [95% CI]</td></tr><tr><td colspan="4">Ablation Analysis (Base Model)</td></tr><tr><td>2.1</td><td>MARCO</td><td>None</td><td>+0.054 [+0.043, +0.067]</td></tr><tr><td>2.1, 2.2</td><td>MARCO</td><td>CTR &amp; CVR</td><td>+0.223 [+0.210, +0.235]</td></tr><tr><td>2.2.1</td><td>MARCO</td><td>CVR only</td><td>+0.181 [+0.168, +0.194]</td></tr><tr><td>2.2.2</td><td>MARCO</td><td>CTR only</td><td>+0.097 [+0.088, +0.110]</td></tr><tr><td>2.3</td><td>CTR-split-only</td><td>None</td><td>+0.002 [-0.020, +0.007]</td></tr><tr><td colspan="4">Capacity Scaling &amp; Theoretical Bound</td></tr><tr><td>2.4</td><td>MARCO (Large CTR, 5×)</td><td>None</td><td>+0.092  $[ + 0 . 0 6 7 , + 0 . 1 1 2 ]$ </td></tr><tr><td>2.4</td><td>MARCO (Oracle Router)</td><td>None</td><td>+2.268  $\left[ + 2 . 2 3 5 , + 2 . 3 0 2 \right]$ </td></tr></table>

3. RQ2.3 Estimation Coupling Bias $( \varepsilon _ { \mathrm { c o u p l e } } ) { : }$ The CTR-split-only ablation yields a statistically negligible precalibration gain $( + 0 . 0 0 2 \% [ - 0 . 0 2 0 \% , + 0 . { \dot { 0 } } 0 7 \% ] )$ . Because $\sum _ { k } \lambda _ { k } \overset { \cdot } { = } \lambda$ (Lemma 9), splitting the CTR head without CVR-side decomposition provides zero CVR headroom while introducing estimation coupling bias $\varepsilon _ { \mathrm { c o u p l e } }$ . This confirms that CVR-side structural intent decomposition is strictly required to capture post-impression gains.

4. RQ2.4 Routing Efficiency Expansion (ηˆ): Expanding CTR model capacity from 1 (Base) to 5 (Large) increases the pre-calibration composed NE gain from +0.054% to +0.092% in Table 3, demonstrating that scaling intent prediction capacity directly unlocks higher post-impression accuracy. When intent routing is driven by an Oracle router $( { \widehat { \pi } } = \pi )$ , the NE gain reaches $+ 2 . { \overset { \cdot } { 2 } } 6 8 \% \ [ + 2 . 2 3 5 \% , + 2 . 3 0 { \overset { \cdot } { 2 } } \% ]$ , establishing the empirical upper bound for realizable theoretical headroom $\Delta { \_ }$ . Benchmarking pre-calibration gains against Oracle headroom $( \Delta _ { \mathcal { F } } = + 2 . 2 6 8 \% )$ confirms that scaling CTR capacity increases realized routing efficiency ηˆ from 2.38% (Base) to 4.06% (Large). This capacity progression directly validates Proposition 7: expanding CTR capacity enlarges the realizable routing class $\Pi ^ { ( c ) }$ , systematically reducing the heterogeneity-weighted routing cost $\varepsilon _ { \mathrm { r o u t e } }$ and raising serving efficiency ηˆ toward 100%

## 6.1.3 Pre-Impression Funnel Recall (RQ3)

The pre-impression stage governs early-stage candidate retrieval, deciding which ad candidates survive to full auction scoring. We evaluate candidate retrieval fidelity on offline production logs across a 3-day evaluation window by truncating candidate sets at an early-funnel scale of $\mathcal { O } ( 1 0 ^ { 2 } )$ . Retrieval quality is measured using value-weighted recall (the ratio of value delivered by the truncated candidate list to the ideal unconstrained list). MARCO achieves a +0.38% lift in value-weighted candidate retrieval recall over the pooled baseline. Restoring per-intent score calibration at inference improves rank-order fidelity early in the funnel, ensuring higher-quality candidates advance to final auction scoring.

## 6.2 Online Business Impact (RQ4)

MARCO has been incrementally deployed across multiple ad surfaces, conversion types, and traffic segments on Meta’s social media platforms.

## 6.2.1 Cumulative Production Impact

MARCO was deployed through several successive production launches. Each launch was conducted as an independent experiment on a disjoint traffic segment, measured against its own concurrent user-randomized holdback. Every launch yielded positive gains, collectively covering a substantial fraction of Meta’s ad traffic. Evaluated globally across all combined launch segments, MARCO delivered a +0.98% cumulative lift in topline metrics (95% CI: $[ + 0 . 8 6 \% , + 1 . 1 2 \% ] )$ ). The confidence interval is a bootstrap percentile interval computed across launches, reflecting performance stability across distinct rollout segments.

## 6.2.2 Detailed Holdback Analysis

To evaluate MARCO’s production performance under live auction dynamics, we conducted a multi-week, userrandomized holdback experiment on a primary advertising surface, comparing the proposed intent-decomposed composition against the standard single-click baseline. MARCO achieved a +2.80% lift in conversions per click with a 95% confidence interval of $[ + 2 . 5 0 \% , + 3 . 1 0 \% ]$ , confirming substantial end-to-end performance gains in live production traffic.

![](images/a5b442c2eaa71f22b361cd4722590e61409f840b9cf9390b28f433e3ccab189a.jpg)  
Figure 3: Cumulative Distribution Function (CDF) of predicted conversion rates (eCVR) at the pre-impression stage, illustrating score separation under MARCO.

The online conversion rate improvement (+2.80%) significantly exceeds the offline impression-space NE gain (+0.223%). This divergence is expected and attributable to two fundamental structural mechanisms:

1. Probability Space Compression: Offline NE evaluates the global impression space, where the served score $\Phi = \lambda \mu$ scales on-click conversion rates µ by the baseline click-through rate λ. Under squared-loss score headroom analysi (Proposition 6), impression-level risk gaps scale proportionally to $\lambda ^ { 2 }$ . Because click-through rates are small $( \lambda \ll 1 )$ offline impression-space metrics undergo geometric compression relative to raw on-click conversion gains.

2. Auction Reallocation Dynamics: Modern real-time ad auctions operate as competitive winner-take-all mechanisms. By eliminating group-conditional miscalibration, MARCO achieves superior score separation across intent strata (Section 6.2.3). This enhanced score fidelity allows high-intent candidate ads to systematically win auctions while suppressing low-converting traffic, dynamically reallocating delivery budget toward higher-converting opportunities and amplifying calibrated scoring gains into non-linear online business lifts.

## 6.2.3 Prediction Distribution Analysis

Figure 3 illustrates the mechanism driving online auction gains. For impressions prone to low-intent clicks, MARCO depresses expected conversion scores, causing the treatment CDF to rise faster in the low-eCVR regime $( < 0 . 0 4 )$ Conversely, for impressions prone to high-intent CTA taps, MARCO amplifies scores, shifting the treatment CDF below the baseline in the high-eCVR regime (> 0.08). This score separation provides the auction mechanism with high-fidelity conversion signals, systematically suppressing low-converting candidates and winning auctions for high-intent traffic.

## 7 Discussion and Conclusion

In this section, we address privacy and ethical considerations, demonstrate the generalizability of intent decomposition to adjacent domains, and outline future technical directions.

Ethical Considerations and Advertiser Alignment. MARCO’s design adheres strictly to platform privacy standards and ethical AI practices. First, regarding data privacy, MARCO relies exclusively on physical UI interactions already logged in standard interaction pipelines, requiring no new sensitive user profiling or off-platform tracking. Second, to ensure statistical fairness across advertisers, restoring subgroup calibration corrects systematic under-prediction on high-intent traffic and over-prediction on low-intent traffic, eliminating unfair auction penalties for advertisers optimizing for high-converting direct actions. Finally, MARCO remains objective-agnostic, dynamically routing candidate value based on advertiser-defined campaign goals rather than hard-coding a preference for direct sales ove brand engagement.

Broad Generalizability and Diagnostic Framework. The principle of structural decomposition extends beyond clickintent prediction. Whenever a single supervised target aggregates events with structurally heterogeneous downstream success rates, single-output models inherit subgroup bias masked by aggregate calibration. Representative applications include app install campaigns (pooling organic-attributed and ad-driven installs with distinct 7-day retention rates), e-commerce funnels (merging quick-add interactions from product grids with considered-adds from detailed review pages), and video recommendation platforms (combining lightweight preview plays with intentional full-screen watch sessions).

Industrial practitioners can evaluate whether a target domain warrants intent decomposition prior to model engineering. By evaluating the impression-weighted Kullback-Leibler (KL) divergence spread across candidate interaction strata (following Lemma 3 and Lemma 11), one can quantify the exact theoretical headroom $\Delta { _ { \mathcal { F } } }$ opened by decomposition directly from historical interaction logs.

Limitations and Future Directions. While MARCO achieves substantial production impact at binary intent granularity $( K = 2 )$ , theoretical headroom $\Delta { _ { \mathcal { F } } }$ increases monotonically with finer intent partitions. We outline three key vectors for future research:

1. Taxonomy Scaling (Depth & Breadth): Expanding K can proceed along two dimensions: depth (partitioning CTA clicks into sub-navigation types) and breadth (incorporating post-impression interaction signals such as dwell time, video watch percentage, and organic engagement). Because Weak Dominance holds for arbitrary partitions (Theorem 1(i)), finer taxonomies never degrade population risk.

2. Routing Efficiency Leverage (ηˆ): Realized online gain is governed by routing efficiency $\hat { \eta } = 1 - \varepsilon _ { \mathrm { r o u t e } } / \Delta _ { \mathcal { F } }$ Because routing error concentrates where per-intent conversion rates diverge (Lemma 11), scaling the capacity of the CTR intent predictor—rather than the CVR model—is the highest-leverage vector to close the remaining headroom gap.

3. Sparse-Stratum Calibration Infrastructure: As K increases, interaction volume is divided across finer intent buckets. Rare interaction categories risk calibration variance due to sample sparsity. Developing dynamic calibrators capable of robust real-time convergence under long-tailed intent distributions remains an important system engineering challenge.

Conventional ads ranking architectures conflate distinct user behaviors into a single click label, imposing a structural observability barrier that no model capacity can resolve. MARCO demonstrates that decoupling predictions by intent via zero-cost behavioral supervision restores subgroup calibration, resolves auction score distortions, and delivers substantial topline growth at industrial scale.

## References

[1] Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Xiaoqiang Zhu, and Kun Gai. Entire space multi-task model: An effective approach for estimating post-click conversion rate. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval, SIGIR ’18, pages 1137–1140. ACM, 2018.

[2] Hong Wen, Jing Zhang, Yuan Wang, Fuyu Lv, Wentian Bao, Quan Lin, and Keping Yang. Entire space multi-task modeling via post-click behavior decomposition for conversion rate prediction. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’20, pages 2377–2386. ACM, 2020.

[3] D. Holt and T. M. F. Smith. Post stratification. Journal of the Royal Statistical Society: Series A, 142(1):33–46, 1979.

[4] Roderick J. A. Little. Post-stratification: A modeler’s perspective. Journal ofthe American Statistical Association, 88(423):1001–1012, 1993.

[5] Hao Wang, Tai-Wei Chang, Tianqiao Liu, Jianmin Huang, Zhichao Chen, Chao Yu, Ruopeng Li, and Wei Chu. Escm2: Entire space counterfactual multi-task model for post-click conversion rate estimation. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’22, pages 363–372. ACM, 2022.

[6] Feng Zhu, Mingjie Zhong, Xinxing Yang, Longfei Li, Lu Yu, Tiehua Zhang, Jun Zhou, Chaochao Chen, Fei Wu, Guanfeng Liu, and Yan Wang. Dcmt: A direct entire-space causal multi-task framework for post-click conversion estimation. In 2023 IEEE 39th International Conference on Data Engineering (ICDE), pages 3113–3125. IEEE, 2023.

[7] Dongbo Xi, Zhen Chen, Peng Yan, Yinger Zhang, Yongchun Zhu, Fuzhen Zhuang, and Yu Chen. Modeling the sequential dependence among audience multi-step conversions with multi-task learning in targeted display advertising. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, KDD ’21, pages 3745–3755. ACM, 2021.

[8] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H. Chi. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings ofthe 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’18, pages 1930–1939. ACM, 2018.

[9] Hongyan Tang, Junning Liu, Ming Zhao, and Xudong Gong. Progressive layered extraction (ple): A novel multitask learning (mtl) model for personalized recommendations. In Fourteenth ACM Conference on Recommender Systems, RecSys ’20, pages 269–278. ACM, 2020.

[10] Jianxin Chang, Chenbin Zhang, Yiqun Hui, Dewei Leng, Yanan Niu, Yang Song, and Kun Gai. Pepnet: Parameter and embedding personalized network for infusing with personalized prior information. In Proceedings ofthe 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’23, pages 3795–3804. ACM, 2023.

[11] John C. Platt. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In Alexander J. Smola, Peter Bartlett, Bernhard Schölkopf, and Dale Schuurmans, editors, Advances in Large Margin Classifiers, pages 61–74. MIT Press, 1999.

[12] Bianca Zadrozny and Charles Elkan. Transforming classifier scores into accurate multiclass probability estimates. In Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining, KDD02, pages 694–699. ACM, 2002.

[13] Yewen Fan, Nian Si, and Kun Zhang. Calibration matters: Tackling maximization bias in large-scale advertising recommendation systems. In The Eleventh International Conference on Learning Representations, 2023.

[14] Jia-Qi Yang, De-Chuan Zhan, and Le Gan. Beyond probability partitions: Calibrating neural networks with semantic aware grouping. In Advances in Neural Information Processing Systems 36, NeurIPS 2023, pages 58448–58460. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2023.

[15] Sejoon Oh, Moumita Bhattacharya, Yesu Feng, and Sudarshan Lamkhede. IntentRec: Predicting user session intent with hierarchical multi-task learning. arXiv preprint arXiv:2408.05353, 2024.

[16] Liang Luo, Yuxin Chen, Zhengyu Zhang, Mengyue Hang, Andrew Gu, Buyun Zhang, Boyang Liu, Chen Chen, Fan Yang, Feifan Gu, Huayu Li, Jade Nie, Jiayi Xu, Jiyan Yang, Jongsoo Park, Laming Chen, Longhao Jin, Qin Huang, Shali Jiang, Shiwen Shen, Shuaiwen Wang, Siyang Yuan, Tongyi Tang, Weilin Zhang, Xi Liu, Xiaohan Wei, Yuchen Hao, Xiaozhen Xia, Yasmine Badr, Zeliang Chen, Chengze Fan, Dong Liang, Qianru Li, Sihan Zeng, Wenjun Wang, Yunlong He, Yinbin Ma, Maxim Naumov, Yantao Yao, Wenlin Chen, and Ellie Dingqiao Wen. Meta lattice: Model space redesign for cost-effective industry-scale ads recommendations. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1, KDD ’26, pages 2335–2346. Association for Computing Machinery, 2026.

[17] Mahanth Kumar Beeraka, Chen Chen, Yining Lu, Briac Marcatte, Weikun Lyu, Brooke Bian, Enriko Aryanto, Ellie Wen, Mohamed A. Radwan, Tianshan Cui, Wenjing Lu, Mohsen Malmir, and Yang Li. Closing the online-offline gap: A scalable framework for composed model evaluation. In Proceedings of the Nineteenth ACM Conference on Recommender Systems, RecSys ’25, pages 923–926. ACM, 2025.

[18] David G Luenberger. Optimization by vector space methods. John Wiley & Sons, Nashville, TN, January 1997.

[19] A. Banerjee, X. Guo, and H. Wang. On the optimality of conditional expectation as a bregman predictor. IEEE Transactions on Information Theory, 51(7):2664–2669, 2005.

[20] Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation. Journal ofthe American Statistical Association, 102(477):359–378, March 2007.

[21] A. Philip Dawid. The well-calibrated bayesian. Journal of the American Statistical Association, 77(379):605–610, 1982.

[22] Boris Mityagin. The zero set of a real analytic function. arXiv preprint arXiv:1512.07276, 2015.

[23] Thomas P. Minka. Bayesian model averaging is not model combination. Technical report, MIT Media Lab, 2000. Technical note.

[24] Kristine Monteith, James L. Carroll, Kevin Seppi, and Tony Martinez. Turning Bayesian model averaging into Bayesian model combination. In The 2011 International Joint Conference on Neural Networks (IJCNN), pages 2657–2663, 2011.

[25] Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. Neural Networks, 2(5):359–366, January 1989.

## A Derivation of the Headroom and Weak Dominance

This appendix gives the full statements and proofs behind the theory section. We prove the observability barrier, the Bayes equivalence, the excess-risk identity, weak dominance in any function class, the exact squared-loss headroom, the genericity of strict improvement, the score bridge, and the signed decomposition of the realized gain. The squared-loss results carry a closed form. The normalized-entropy results keep only the loss-free parts, for the reason given in the mixture-of-experts lemma (Lemma 6).

## A.1 Setup and notation

We condition on a click $C = 1$ unless the impression level is named. Let X be impression-time features with impression distribution $\rho .$ Let $k \in \{ 1 , \ldots , K \}$ index the click intent over K disjoint categories, observed at training from the logged click type. Let $Y \in \{ 0 , 1 \}$ be the conversion. Define

$$
\begin{array} { r } { \mu _ { k } = \operatorname { \mathbb { E } } [ Y \mid X = x , K = k ] , \quad \pi _ { k } = P ( K = k \mid X = x , C = 1 ) , } \\ { \mu = \operatorname { \mathbb { E } } [ Y \mid X = x , C = 1 ] . } \end{array}\tag{8}
$$

The law of total probability gives the exact identity $\begin{array} { r } { \mu = \sum _ { k = 1 } ^ { K } { \pi _ { k } \mu _ { k } } } \end{array}$ . This identity assumes no homogeneity of clicks. Let $\lambda = \dot { P ( C = 1 \mid X = x ) }$ Pbe the click-through rate and $\lambda _ { k } = \lambda \pi _ { k }$ the per-intent click rate, so $\sum _ { k } \lambda _ { k } ^ { - } = \lambda$ and $\pi _ { k } = \lambda _ { k } / \lambda .$

Let $\mathcal { F }$ be the hypothesis class of a single pooled conversion head. The intent-decomposed class is

$$
{ \mathcal { G } } = { \Big \{ } g = \sum _ { k = 1 } ^ { K } \pi _ { k } f _ { k } : f _ { k } \in { \mathcal { F } } { \Big \} } .\tag{9}
$$

MARCO serves the impression-level score $\begin{array} { r } { \Phi _ { \mathrm { M A R C O } } = \sum _ { k } \lambda _ { k } \mu _ { k } } \end{array}$ . The standard model serves $\Phi _ { \mathrm { s t d } } = \lambda \mu$

## A.2 Bayes equivalence

Lemma 1 (Bayes equivalence) At the true conditionals $\Phi _ { \mathrm { M A R C O } } = \Phi _ { \mathrm { s t d } }$

$$
\begin{array} { r } { \mathbf { P r o o f 1 } \ U s e \lambda _ { k } = \lambda \pi _ { k } \ a n d \sum _ { k } \pi _ { k } = 1 . \ T h e n \ \Phi _ { \mathrm { M A R C O } } = \sum _ { k } \lambda \pi _ { k } \mu _ { k } = \lambda \sum _ { k } \pi _ { k } \mu _ { k } = \lambda \mu = \Phi _ { \mathrm { s t d } } . } \end{array}
$$

At the population level the two scores coincide. The Bayes-optimal impression score is unchanged by the decomposition. Any deployed gain is therefore a finite-capacity estimation and calibration effect. The click-rate decomposition adds no independent factor. It supplies the routing $\pi _ { k } = \lambda _ { k } / \lambda$ and the total $\lambda = \sum _ { k } \lambda _ { k }$ . The served conversion factor is $\sum _ { k } \pi _ { k } \mu _ { k }$ , an element of $\mathcal { G }$ whose routing is produced by the click-rate head. The analysis reduces to the single conversion head of (9).

## A.3 Excess risk

We first record the squared-loss identity, then the cross-entropy identity.

Squared loss uses the risk $\mathcal { R } ( f ) = \mathbb { E } [ ( Y - f ) ^ { 2 } ]$ . Write $\langle \cdot , \cdot \rangle$ and $\lVert \cdot \rVert$ for the $L ^ { 2 } ( \rho )$ inner product and norm. he identity below is the standard $L ^ { 2 } ( \rho )$ projection and bias-variance decomposition [18].

Lemma 2 (Squared-loss excess risk) For any $f , \mathcal { R } ( f ) = \mathcal { R } ( \mu ) + \| f - \mu \| ^ { 2 }$ . The infimum of $\mathcal { R }$ over a closed set is attained at the $L ^ { 2 } ( \rho )$ projection of µ onto that set.

Proof 2 Expand

$$
\begin{array} { r l } & { { \mathcal { R } } ( f ) = \operatorname { \mathbb { E } } [ ( Y - \mu ) ^ { 2 } ] } \\ & { \quad \quad + \ 2 \operatorname { \mathbb { E } } [ ( Y - \mu ) ( \mu - f ) ] } \\ & { \quad \quad + \ \operatorname { \mathbb { E } } [ ( \mu - f ) ^ { 2 } ] . } \end{array}\tag{10}
$$

The cross term vanishes because $\mathbb { E } [ Y - \mu \mid X ] = 0$ . The first term is $\textstyle { \mathcal { R } } ( \mu )$ . The last term is $\| f - \mu \| ^ { 2 }$ . Minimizing $\| f - \mu \| ^ { 2 }$ over a closed set is projection.

Cross-entropy uses the risk $\mathcal { R } ( f ) = \mathbb { E } [ - Y$ log $f - ( 1 - Y ) \log ( 1 - f ) ]$ for f valued in $( 0 , 1 )$ . Normalized entropy is $\mathrm { N E } ( f ) = \mathcal { R } ( f ) / H ( \bar { y } )$ with $\bar { y } = \mathbb { E } [ Y ]$ and H the binary entropy. $H ( \bar { y } )$ is a fixed positive constant, so NE and cross-entropy share minimizers and orderings. Every statement below applies to both.

Lemma 3 (Cross-entropy excess-risk identity) For any $( 0 , 1 )$ -valued $f ,$

$$
\mathcal { R } ( f ) = \mathcal { R } ( \mu ) + \mathbb { E } _ { X } \left[ \mathrm { K L } ( \mathrm { B e r n } ( \mu ) \| \mathrm { B e r n } ( f ) ) \right] .\tag{11}
$$

Hence $\mathcal { R } ( f ) \geq \mathcal { R } ( \mu )$ with equality iff $f = \mu$ almost everywhere.

Proof 3 Cross-entropy is the Bregman divergence generated by $\psi ( p ) = p \log p + ( 1 - p ) \log ( 1 - p ) \ : [ I 9$ , 20]. Write $D _ { \psi } ( y , f ) = \psi ( y ) - \overset { \cdot } { \psi } ( f ) - \psi ^ { \prime } ( \overset { \cdot } { f } ) ( y - f )$ . Then ${ \tilde { \mathbb { E } } } [ D _ { \psi } ( Y , f ) \mid { \dot { X } } ] { \tilde { = } } { \mathbb { E } } [ { \tilde { \psi } } ( Y ) ^ { \circ } \mid X ] - { \tilde { \psi ( f ) } } - \psi ^ { \prime } ( { \dot { f } } ) ( \mu - f )$ because $\mathbb { E } [ { \dot { Y } } \mid X ] = \mu$ . Subtracting the same expression at $f = \mu$ leaves $\operatorname { K L } ( \operatorname { B e r n } ( \mu ) \mid$ Bern(f)). Take the expectation over $\dot { X . }$ . Nonnegativity and the equality case follow from strict convexity of ψ.

The excess risk under cross-entropy is an averaged Kullback-Leibler divergence between the true per-impression conversion rate and the prediction. Under the classical calibration–refinement decomposition of a proper scoring rule [21], this gap splits into a calibration term, which measures how far predictions sit from the conditional conversion rate they are grouped with, and a refinement term, which measures the residual heterogeneity within those groups. A per-intent calibration step drives the calibration term to zero within each intent, while decomposition attacks the refinement term by splitting the pooled group into homogeneous intent slices. The log-loss gap and per-intent calibration are thus the same object measured on each intent slice.

## A.4 Observability barrier

Proof 4 (Proof of Proposition 1) Fix x and condition on a click. Being calibrated on intent category k means the emitted value equals that category’s conditional mean, $\mu = \mu _ { k } .$ . Simultaneous calibration on all K categories thus requires $\mu = \mu _ { k } f o r$ every k, which is impossible whenever the $\mu _ { k }$ are not all equal, since a single scalar equals at most one element ofa set ofdistinct reals; the samefact gives the at-most-one claim. Under a strictly proper scoring rule the risk-minimizing single output is the pooled conditional mean $\textstyle \mathbb { E } [ Y \mid X { = } x , C { = } 1 ] = \sum _ { k }$ π<sub>k</sub>µ<sub>k</sub> (Lemmas 2 and 3), which is aggregate-calibrated yet, as a convex combination of the $\mu _ { k }$ , coincides with at most one of them.

## A.5 Inclusion

Lemma 4 (Inclusion) ${ \mathcal { F } } \subseteq { \mathcal { G } }$

Proof 5 Given $f \in { \mathcal { F } } ,$ , set $f _ { k } \equiv f$ for all k in (9). Then $\textstyle \sum _ { k } \pi _ { k } f = f$ because $\textstyle \sum _ { k } \pi _ { k } = 1$ . This is set-theoretic and uses neither the loss nor any structure on ${ \mathcal F } .$

Inclusion gives the headroom used throughout. Define

$$
\Delta _ { \mathcal { F } } = \operatorname* { i n f } _ { f \in \mathcal { F } } \mathcal { R } ( f ) - \operatorname* { i n f } _ { g \in \mathcal { G } } \mathcal { R } ( g ) ,\tag{12}
$$

the risk gap between the best-in-class pooled and decomposed models. Weak dominance $( \Delta \ v { r } _ { \mathcal { F } } \geq 0 )$ is the special case of the responsibility-gated dominance theorem (Theorem 1(i)) at the supervised gate $\pi _ { k } = P ( K { = } k \mid X { = } x , C { = } 1 )$ since ${ \mathcal { F } } \subseteq { \mathcal { G } }$ , the infimum over the superset can only decrease. It holds for squared loss and for cross-entropy, needs no linear structure and no capacity assumption, so the decomposed objective is never worse than the pooled objective in any function class. This is why the improvement is structural rather than a capacity effect.

## A.6 Exact squared-loss headroom

Take $\mathcal { F }$ to be a finite-dimensional linear subspace of $L ^ { 2 } ( \rho )$ with basis $\{ e _ { 1 } , \ldots , e _ { D } \}$ . This is the finite-capacity model for a single scalar head.

Lemma 5 ( is a subspace) $\mathcal { G } = \mathrm { s p a n } \{ \pi _ { k } e _ { j } : 1 \leq k \leq K , 1 \leq j \leq D \}$ , a subspace ofdimension at most $K D ,$ , and ${ \mathcal { F } } \subseteq { \mathcal { G } }$ . Moreover $\mathcal G = \mathcal F i f f \pi _ { k } e _ { j } \in \mathcal F \mathring { f } o \iota$ all k and j. Equivalently $\mathcal { G } \supset \mathcal { F }$ iff there exist k and j with $\pi _ { k } e _ { j } \notin \mathcal { F }$

Proof 6 Write $\begin{array} { r } { f _ { k } = \sum _ { j } a _ { k j } e _ { j } } \end{array}$ . Then $\begin{array} { r } { g = \sum _ { k , i } a _ { k j } ( \pi _ { k } e _ { j } ) } \end{array}$ , so $\mathcal { G } = \operatorname { s p a n } \{ \pi _ { k } e _ { j } \}$ . This is a subspace. The inclusion lemma gives ${ \mathcal { F } } \subseteq { \mathcal { G } } . H$ every $\pi _ { k } e _ { j } \in \mathcal { F }$ then ${ \mathcal { G } } \subseteq { \mathcal { F } }$ , so $\mathcal { G } = \mathcal { F } .$ . Conversely $\mathcal { G } = \mathcal { F }$ forces each generator $\pi _ { k } e _ { j } \in \mathcal { F }$

Proposition 2 (Non-constant routing is necessary but not sufficient) Constant $\pi _ { k }$ gives $\pi _ { k } e _ { j } \in { \mathcal { F } } ,$ , so non-constant routing is necessaryfor ${ \mathcal { G } } \supset { \mathcal { F } } .$ . It is not sufficient. Let ρ have atoms $x _ { 1 } , x _ { 2 } , x _ { 3 }$ and let = span $\left[ { \bf { 1 } } _ { x _ { 1 } } , { \bf { 1 } } _ { x _ { 2 } } \right\}$ . Let $\pi _ { 1 }$ take distinct values on $x _ { 1 }$ and x and let $\pi _ { 2 } = 1 - \pi _ { 1 }$ . Multiplication by $\pi _ { k }$ is diagonal, so $\pi _ { k } e _ { j } \in \mathcal { F } a n d \mathcal { G } = \mathcal { F }$ . The correct condition is the one in the subspace lemma.

Proposition 3 (Exact headroom) Let $P _ { \mathcal { F } } , P _ { \mathcal { G } }$ be the $L ^ { 2 } ( \rho )$ projections onto ${ \mathcal { F } } , { \mathcal { G } } .$ . Then

$$
\begin{array} { r l } & { \Delta _ { \mathcal { F } } = \underset { f \in \mathcal { F } } { \operatorname* { i n f } } \ \mathcal { R } ( f ) - \underset { g \in \mathcal { G } } { \operatorname* { i n f } } \ \mathcal { R } ( g ) } \\ & { \quad \quad = \| \mu - P _ { \mathcal { F } } \mu \| ^ { 2 } - \| \mu - P _ { \mathcal { G } } \mu \| ^ { 2 } } \\ & { \quad \quad = \| P _ { \mathcal { G } } \mu - P _ { \mathcal { F } } \mu \| ^ { 2 } \ \geq 0 , } \end{array}\tag{13}
$$

with $\Delta _ { \mathcal { F } } = 0 \ i f f _ { \mathscr { G } } \mu = P _ { \mathcal { F } } \mu .$

Proof 7 By the squared-loss excess-risk lemma the two infima are $\mathcal { R } ( \mu ) + \| \mu - P _ { \mathcal { F } } \mu \| ^ { 2 }$ and $\mathcal { R } ( \mu ) + \| \mu - P g \mu \| ^ { 2 }$ . Since ${ \mathcal { F } } \subseteq { \mathcal { G } }$ we have $\bar { P } _ { \mathcal { F } } = P _ { \mathcal { F } } P _ { \mathcal { G } }$ . The Pythagorean identity on nested subspaces (the classical projection theorem $I I { \& } I )$ gives $\| \mu - P _ { \mathcal F } \mu \| ^ { 2 } = \| \mu - \tilde { P } _ { \mathcal G } \mu \| ^ { 2 } \dot { + } \| \tilde { P _ { \mathcal G } } \mu - P _ { \mathcal F } \mu \| ^ { 2 }$

No assumption that $\mu _ { k } \in \mathcal { F } \mathrm { o r } \mu \in \mathcal { F }$ is used. The headroom is the energy of $\mu$ along the intent-modulated directions in $\mathcal { G }$ that are orthogonal to $\mathcal { F }$

Lemma 6 (No closed form under normalized entropy) Under cross-entropy the class $\mathcal { F }$ cannot be a linear subspace. A finite-dimensional subspace contains functions valued outside (0, 1) on a set of positive probability, where crossentropy is infinite. Modeling in logit space with $f = \sigma ( z )$ makes the decomposed predictor $\scriptstyle \sum _ { k } \pi _ { k } \sigma ( \dot { z _ { k } } )$ a mixture of experts, because $\begin{array} { r } { \sigma ( \sum _ { k } \pi _ { k } \bar { z } _ { k } ) \ne \overleftarrow { \sum _ { k } } \pi _ { k } \sigma ( z _ { k } ) } \end{array}$ . So is not a subspace: there are no orthogonal projections and no Pythagorean identity, hence no closed-form projection expression for the cross-entropy headroom $\Delta { _ { \mathcal { F } } }$ analogous to the squared-lossform. The cross-entropy risk itselfremains well defined andfinite; only its best-in-class gap lacks a closed form. Inclusion and weak dominance $( \Delta _ { \mathcal { F } } \geq 0 )$ still hold because they are set-theoretic. We therefore measure $\Delta _ { \mathcal { F } }$ empirically as the held-out NE gap between the best pooled and the best decomposed model ofthe same architecture and capacity.

## A.7 Genericity of strict improvement

Lemma 7 (Zero set of a real-analytic function) $I f h : U \to \mathbb { R }$ is real-analytic on a connected open $U \subseteq \mathbb { R } ^ { m }$ and $h \not \equiv 0 ,$ , then $\{ h = 0 \}$ has Lebesgue measure zero $I 2 2 J .$

Proposition 4 (Genericity, squared loss) Let $\Theta \subseteq \mathbb { R } ^ { m }$ be open and connected. Fix $\rho$ and the routing π, hence $\mathcal { F } , \bar { \mathcal { G } } , P _ { \mathcal { F } } , P _ { \mathcal { G } }$ . Assume

(A1) $\theta \mapsto \mu _ { \theta }$ is real-analytic into $L ^ { 2 } ( \rho )$ , with a local $L ^ { 2 } ( \rho )$ envelope dominating $\theta \mapsto \mu _ { \theta }$ , so Hilbert-valued analyticity holds.

(A2) there is $\theta _ { 0 }$ with $P g \mu _ { \theta _ { 0 } } \ne P _ { \mathcal { F } } \mu _ { \theta _ { 0 } }$

Then $\{ \theta : \Delta _ { \mathcal { F } } ( \theta ) = 0 \}$ has Lebesgue measure zero. Under any prior absolutely continuous with respect to Lebesgue measure, $\Delta _ { \mathcal { F } } > 0$ almost surely.

Proof 8 $L e t r ( \theta ) = \| P \varsigma \mu _ { \theta } - P \mathcal { F } \mu _ { \theta } \| ^ { 2 } .$ . Since π and ρ arefixed, $P _ { \mathcal { F } }$ and $P _ { \mathcal { G } }$ arefixed bounded linear operators. $B y$ $( A I ) \ : \theta \mapsto ( P _ { \mathcal { G } } - P _ { \mathcal { F } } ) \ddot { \mu } _ { \theta }$ is real-analytic into $L ^ { 2 } ( \rho )$ . Composing with the continuous bilinear inner product makes r real-analytic. By $( A 2 ) , r ( \theta _ { 0 } ) > 0 ,$ , so $r \not \equiv 0 .$ The analytic zero-set lemma makes $\{ r = 0 \}$ null. The exact-headroom proposition gives $\Delta \_ = r .$

Corollary 1 (Affine instance) Let be affine, $K \ = \ 2 ,$ , let $\mu _ { H } , \mu _ { L }$ be affine, and let $\pi _ { H }$ be affine. From $\mu ~ =$ $\mu _ { L } + \pi _ { H } ( \mu _ { H } - \mu _ { L } )$ , a product oftwo affinefunctions is affine iffafactor is constant. So $\mu \in \mathcal { F } \ : i f f \pi _ { H }$ is constant or $\mu _ { H } - \mu _ { L }$ is constant. That is a finite union of proper affine subspaces of parameter space, hence null. Any θ outside it witnesses (A2). No analytic machinery is needed here.

Proposition 5 (Genericity, cross-entropy) Parametrize the conditionals by θ in a connected open $\Theta \subseteq \mathbb { R } ^ { m }$ . Suppose $\theta \mapsto \operatorname* { i n f } _ { f \in { \mathcal { F } } } { \mathcal { R } } _ { \theta } ( f )$ and $\theta \mapsto \operatorname* { i n f } _ { g \in { \mathcal { G } } } \mathcal { R } _ { \theta } ( g )$ are real-analytic, which holdsfor a smooth link, a strictly convex loss, and a unique interior minimizer with non-singular Hessian by the implicit function theorem. Suppose $\Delta _ { \mathcal { F } } ( \theta _ { 0 } ) > 0 f o t$ some $\theta _ { 0 } .$ . Then $\{ \theta : \Delta _ { \mathcal { F } } ( \theta ) = 0 \}$ has Lebesgue measure zero.

Proof 9 $\begin{array} { r } { \Delta _ { \mathcal { F } } ( \theta ) = \operatorname* { i n f } _ { \mathcal { F } } \mathcal { R } _ { \theta } - \operatorname* { i n f } _ { \mathcal { G } } \mathcal { R } _ { \theta } } \end{array}$ is real-analytic by hypothesis and non-zero at $\theta _ { 0 }$ . The analytic zero-set lemma makes its zero set null. The passage from a null set to probability zero is absolute continuity. The content is the analyticity of the two infima, which for cross-entropy is a genuine implicit-function argument rather than a projection identity.

Lemma 8 (Genericity is not a deployment guarantee) Genericity rules out the knife-edge in which the pooled class alreadyfits $\mu$ as well as the decomposed class does. The deployed system is a single unknown truth $\theta ^ { * }$ , which can lie on the null set. Structured configurations such as coarse intent buckets, symmetries, or sparsity are exactly where it might. The operative claim for deployment is the measured headroom, not the prior.

## A.8 Responsibility-gated dominance

Theorem 1 (Responsibility-gated dominance) Let $\mathcal { F }$ be any single-expert hypothesis class. Let the gate $\pi =$ $( \pi _ { 1 } , \ldots , \pi _ { K } )$ with $\pi _ { k } \ = \ P ( K { = } k \ | \ X { = } x , C { = } 1 )$ be a supervised, input-dependent responsibility over an intent $K$ observed at training, with $\textstyle \sum _ { k } \pi _ { k } = 1$ , and define $\begin{array} { r } { \mathcal { G } = \{ \bar { \sum _ { k } } \pi _ { k } f _ { k } : f _ { k } \in \mathcal { F } \} } \end{array}$ . Thenfor any risk $\mathcal { R } .$

(i) no worse: in $\mathrm { f } _ { \mathcal { G } } \mathscr { R } \leq \operatorname* { i n f } _ { \mathcal { F } } \mathscr { R }$

(ii) generically better: if the truth is parametrized by θ in a connected open set with $\theta \mapsto \operatorname* { i n f } _ { \mathcal { F } } \mathcal { R } _ { \theta }$ , inf $\scriptstyle { \mathcal { R } } _ { \theta }$ real-analytic and in $\therefore x < \operatorname* { i n f } _ { \mathcal { F } } \mathcal { R }$ at one $\theta _ { 0 } ,$ , then $\{ \theta : \operatorname* { i n f } _ { \mathcal { G } } \mathcal { R } = \operatorname* { i n f } _ { \mathcal { F } } \mathcal { \bar { R } } \}$ is Lebesgue-null.

(iii) not BMA: the gate is a fixed per-example responsibility, not a posterior over models, so the minimizer $\begin{array} { r } { g ^ { \star } = \sum _ { k } \pi _ { k } f _ { k } ^ { \star } } \end{array}$ does not collapse to a single expert as $n  \infty$ whenever π is non-degenerate and the $\mu _ { k }$ differ.

Proof 10 (Proof sketch) (i) Setting $f _ { k } \equiv f$ gives $\textstyle \sum _ { k } \pi _ { k } f = f$ since $\textstyle \sum _ { k } \pi _ { k } = 1$ , so ${ \mathcal { F } } \subseteq { \mathcal { G } } ,$ , and the infimum over a superset can only decrease. (ii) Let $\begin{array} { r } { r ( \theta ) = \operatorname* { i n f } _ { \mathcal { F } } \overline { { \mathscr { R } } } _ { \theta } ^ { \ \ast } - \operatorname* { i n f } _ { \mathcal { G } } \mathscr { R } _ { \theta } \geq 0 , } \end{array}$ , real-analytic by hypothesis and positive at $\theta _ { 0 } ,$ so $r \not \equiv 0 ;$ , and by Mityagin’s theorem its zero set is Lebesgue-null. Under squared loss $\dot { r } ( \theta ) = \| ( P \mathcal { G } - P _ { \mathcal { F } } ) \mu _ { \theta } \| ^ { 2 }$ is analytic directly. (iii) Bayesian model averaging weights $\overset { \cdot } { w } _ { k } = P ( M _ { k } \mid D _ { n } )$ concentrate on one model by posterior consistency, so $\begin{array} { r } { \sum _ { k } w _ { k } f _ { k } \to f _ { k ^ { \star } } } \end{array}$ , which is selection, not combination. Here $\pi _ { k }$ is the data-generating responsibility, constant in n and varying in x, so $g ^ { \star }$ stays a genuine input-dependent mixture.

Part (iii) records the standard mixture-of-experts versus Bayesian model averaging distinction [23, 24]. A fitted input-dependent gate combines rather than selects, whereas BMA weights concentrate on a single model, so BMA does not enlarge the class. We include it for completeness and do not claim it as novel, since combination rather than selection is the defining property of a mixture of experts. What is distinctive to MARCO is not combination versus selection but that the gate is a supervised observed responsibility read from the logged click type, which lets each expert be exposed and separately calibrated, a property that sits outside both latent-gate mixtures of experts and Bayesian model averaging. Parts (i) and (ii) are the weak-dominance and genericity results, here stated for a general supervised responsibility gate.

## A.9 Realized gain is a signed decomposition

$\Delta { _ { \mathcal { F } } }$ is a population best-in-class quantity. The deployed model uses estimated routing ${ \widehat { \pi } } ,$ , finite-sample heads, and a stratified training objective. We express the realized gain as an exact telescoping identity through reference predictors: the best-in-class pooled model $P _ { \mathcal { F } } \mu$ with risk inf $\mathcal { F R }$ , the best-in-class decomposed model $P _ { \mathcal { G } } \mu$ with risk $\operatorname { i n f } _ { \mathcal { G } } \mathcal { R } .$ , the stratified predictor $\begin{array} { r } { g _ { \mathrm { s t r a t } } = \sum _ { k } \pi _ { k } P _ { \mathcal { F } } \mu _ { k } \in \mathcal { G } } \end{array}$ that fits each head on its own intent slice and routes by the true $\pi _ { \mathrm { : } }$ , and the deployed model $\widehat g$ with estimated heads and routing π. Writing $\widehat { f }$ for the fitted pooled model,

$$
\begin{array} { r l } & { \mathcal { R } ( \widehat { f } ) - \mathcal { R } ( \widehat { g } ) = \Delta _ { \mathcal { F } } + \underbrace { \big ( \mathcal { R } ( \widehat { f } ) - \operatorname* { i n f } _ { \mathcal { F } } \mathcal { R } \big ) } _ { \mathrm { e s t } _ { \mathcal { F } } \geq 0 } } \\ & { \qquad - \underbrace { \big ( \mathcal { R } ( g _ { \mathrm { s t r a t } } ) - \operatorname* { i n f } _ { \mathcal { G } } \mathcal { R } \big ) } _ { \varepsilon _ { \mathrm { c o u p l e } } \geq 0 } - \underbrace { \big ( \mathcal { R } ( \widehat { g } ) - \mathcal { R } ( g _ { \mathrm { s t r a t } } ) \big ) } _ { \varepsilon _ { \mathrm { r o u t e } } } . } \end{array}\tag{14}
$$

The identity is exact by cancellation, for any choice of reference predictors. Three terms carry a fixed sign by construction: $\Delta \_ \geq 0$ is weak dominance, est $_ { \perp } \geq 0$ is the pooled estimation error, and $\varepsilon _ { \mathrm { c o u p l e } } ~ \geq ~ 0$ because $g _ { \mathrm { s t r a t } } \in \mathcal { G }$ cannot beat the joint minimizer $P _ { \mathcal { G } } \mu$ , whose heads solve a coupled least-squares problem rather than the per-slice projections. It is the price of training heads separately rather than jointly (Assumption 1). The remaining term $\bar { \varepsilon } _ { \mathrm { r o u t e } } = \bar { \mathcal { R } } ( \widehat { \mathfrak { g } } ) - \mathcal { R } ( g _ { \mathrm { s t r a t } } )$ collects the finite-sample head error and the estimated-routing error; its routing component is controlled by the heterogeneity-weighted cost of Lemma 11, $\varepsilon _ { \mathrm { r o u t e } } \leq \mathbb { E } [ \| \widehat { \pi } - \pi \| ^ { 2 } s ^ { 2 } ]$ , and is nonnegative in expectation when the heads and routing are consistent, vanishing as routing becomes exact or intents become homogeneous. The realized gain is thus $\Delta _ { \mathcal { F } }$ inflated by the pooled model’s own estimation error and deflated by the coupling and routing penalties, so it can exceed $\Delta { _ { \mathcal { F } } }$ or turn negative; we report the measured headroom empirically. In the deployed system the decomposed heads share the backbone and add fewer than $1 0 ^ { - 7 0 7 }$ of parameters, so both $\mathrm { e s t } _ { \mathcal { F } }$ and the estimation part of $\varepsilon _ { \mathrm { r o u t e } }$ are small rather than the K-fold blow-up of K independent models, which is why the realized gains are positive rather than variance-swamped.

Assumption 1 (Objective) Weak dominance and the headroom refer to minimizing the composite risk $\textstyle { \mathcal { R } } ( g )$ over ${ \mathcal { G } } .$ Training per-intent heads separately yields a suboptimal member of  that is not guaranteed to beat the pooled optimum. Carry $\varepsilon _ { \mathrm { c o u p l e } }$ explicitly or reframe training as joint minimization $o f \sum _ { k } \pi _ { k } f _ { k }$

## A.10 Bridging to the served score

Lemma 9 (Score bridge) The conversion decomposition supplies the per-intent targets $\mu _ { k }$ that enlarge the class and create the headroom, and the click-rate decomposition supplies the routing $\pi _ { k } = \lambda _ { k } / \lambda .$ . Neither head aloneforms the composition $\sum _ { k } \lambda _ { k } \mu _ { k }$ , so both are required by construction. There is no separate click-rate approximation headroom, because $\sum _ { k } \overline { { \lambda _ { k } } } = \dot { \lambda }$

Proposition 6 (Score headroom, squared loss) Work at the impression level with impression distribution $\rho _ { \mathrm { i m p } } .$ Suppose both models share the same CTR estimate ${ \widehat { \lambda } } = \lambda ,$ , so only the conversionfactor differs. Under squared loss on the score,

$$
\begin{array} { r l } { \ } & { \underset { \mathcal { F } _ { \mu } } { \operatorname* { i n f } } \ \mathcal { R } _ { \mathrm { s c o r e } } - \underset { \mathcal { G } _ { \mu } } { \operatorname* { i n f } } \ \mathcal { R } _ { \mathrm { s c o r e } } } \\ & { \quad \ = \| P _ { \mathcal { G } _ { \mu } } \mu - P _ { \mathcal { F } _ { \mu } } \mu \| _ { \lambda ^ { 2 } \rho _ { \mathrm { i m p } } } ^ { 2 } . } \end{array}\tag{15}
$$

The score-level headroom equals the conversion-head headroom ofthe exact-headroom proposition, computed in the click-magnitude-weighted measure $\lambda ^ { 2 } \rho _ { \mathrm { i m p } }$ , with routing $\pi _ { k } = \lambda _ { k } / \lambda$ supplied by the CTR head.

Proof 11 With $\widehat { \lambda } = \lambda$ the score predictor is $\lambda \widehat { \mu } f o r c$ conversion predictor ${ \widehat { \mu } } ,$ and the score target is $\lambda \mu$ . The squared bscore-risk excess is $\mathbb { E } _ { \rho _ { \mathrm { i m p } } } [ ( \lambda \widehat { \mu } - \lambda \mu ) ^ { 2 } ] = \mathbb { E } _ { \rho _ { \mathrm { i m p } } } ^ { \widehat { \mu } ^ { * } } [ \lambda ^ { 2 } ( \widehat { \mu } - \mu ) ^ { 2 } ] = \widehat { \| \mu } - \mu \| _ { \lambda ^ { 2 } \rho _ { \mathrm { i m p } } } ^ { 2 }$ . Minimize over ${ \mathcal { F } } _ { \mu }$ and over ${ \mathcal G } _ { \mu }$ and bsubtract. Apply the exact-headroom proposition in $L ^ { 2 } ( \lambda ^ { 2 } \rho _ { \mathrm { i m p } } )$

Since λ is small, the score-level headroom is scaled down by $\lambda ^ { 2 }$ relative to the on-click conversion headroom. This is why impression-space gains look modest even when the on-click miscalibration is large. The CTR head enters only through the routing and the shared magnitude λ. There is no separate CTR approximation headroom because $\begin{array} { r } { \sum _ { k } \lambda _ { k } = \bar { \lambda } . } \end{array}$

Corollary 2 (Score bridge, cross-entropy) Under cross-entropy, with ${ \widehat { \lambda } } = \lambda ,$ the served MARCO score $i s \lambda$ $( \sum _ { k } \pi _ { k } \mu _ { k } )$ , an element of $\lambda \cdot { \mathcal { G } } ,$ , and the standard score is an element $o f \lambda \cdot \mathcal { F }$ . Since ${ \mathcal { F } } \subseteq { \mathcal { G } }$ gives $\lambda { \mathcal { F } } \subseteq \lambda { \mathcal { G } }$ weak dominance carries to the score and $\Delta _ { \Phi } \geq 0 .$ . There is no closed form. Report the impression-space NE gap directly.

Proof 12 $\lambda { \mathcal { F } } \subseteq \lambda { \mathcal { G } }$ by the inclusion lemma scaled pointwise by λ. Take the infimum over the superset. The absence of a closed form is the mixture-of-experts lemma (Lemma 6).

## A.11 Scope of the claims

Lemma 10 (Vanishing at infinite capacity) For nested classes whose union is dense, inf $\tau \mathcal { R } \to \mathcal { R } ( \mu )$ . Since ${ \mathcal { F } } \subseteq { \mathcal { G } }$ $\operatorname { i n f } _ { \mathcal { G } } \mathcal { R }$ is squeezed to $\textstyle { \mathcal { R } } ( \mu )$ as well, so $\Delta \mathcal { F }  0$ . The headroom is a strictly finite-capacity phenomenon. This is consistent with the Bayes equivalence ofthe pooled and decomposed scores at the population level.

Proof 13 Let ${ \mathcal { F } } _ { 1 } \subseteq { \mathcal { F } } _ { 2 } \subseteq \cdots$ be the nested classes with $\overline { { \bigcup _ { n } \mathcal { F } _ { n } } } ~ = ~ L ^ { 2 } ( \rho )$ , and $\mathcal { G } _ { n } \supseteq \mathcal { F } _ { n }$ the corresponding decomposed classes. $B y$ the excess-risk identities (Lemmas 2 and 3), in $\mathrm { f } _ { \mathcal { F } _ { n } } \mathcal { R } - \mathcal { R } ( \mu )$ equals the approximation error $o f \mu b y { \mathcal { F } } _ { n }$ , which decreases to 0 by density (the standard universal-approximation/denseness property [25]). Since $\begin{array} { r } { \mathcal { \dot { R } } \dot { ( \mu ) } \dot { \operatorname { \mathcal { R } } } \dot { \operatorname { \mathcal { R } } } \dot { \operatorname { \mathcal { R } } } \dot { \operatorname { \mathcal { R } } } \dot { \operatorname { \mathcal { R } } } \dot { \operatorname { \mathcal { R } } } \dot { \operatorname { \mathcal { R } } } \dot { \operatorname { \mathcal { R } } } } \end{array}$ , the middle term is squeezed to $\textstyle { \mathcal { R } } ( \mu )$ , so $\begin{array} { r } { \bar { \Delta } _ { \mathcal { F } _ { n } } = \operatorname* { i n f } _ { \mathcal { F } _ { n } } \mathcal { R } - \operatorname* { i n f } _ { \mathcal { G } _ { n } } \mathcal { R } \to 0 . } \end{array}$

## A.12 Routing realizes the headroom

Lemma 11 (Routing cost is heterogeneity-weighted) The routing cost obeys $\varepsilon _ { \mathrm { r o u t e } } \leq \mathbb { E } [ \| \widehat { \pi } - \pi \| ^ { 2 } s ^ { 2 } ]$ with $s ^ { 2 } = { }$ $\textstyle \sum _ { k } ( \mu _ { k } - \mu ) ^ { 2 }$ . It vanishes where the per-intent rates are homogeneous $( s \ : = \ : 0 )$ , regardless of routing error, and concentrates where the intents disagree.

Proof 14 The served conversion factor is $\begin{array} { r } { \widehat { g } = \sum _ { k } \widehat { \pi } _ { k } \mu _ { k } = \mu + \delta } \end{array}$ with $\begin{array} { r } { \delta = \sum _ { k } ( \widehat { \pi } _ { k } - \pi _ { k } ) \mu _ { k } } \end{array}$ . Since $\widehat { \pi }$ and $\pi$ are distributions, $\begin{array} { r } { \sum _ { k } ( \widehat { \pi } _ { k } - \pi _ { k } ) = 0 } \end{array}$ b, so subtracting $\begin{array} { r } { \mu \sum _ { k } ^ { \setminus } ( \widehat { \pi } _ { k } - \pi _ { k } ) = 0 g i \nu e s \delta = \sum _ { k } ( \widehat { \pi } _ { k } - \pi _ { k } ) ( \mu _ { k } - \mu ) } \end{array}$ b. Cauchy-Schwarz over k gives $\begin{array} { r } { | \delta | \leq \left. \widehat { \pi } - \pi \right. . } \end{array}$ s with $\begin{array} { r } { s ^ { 2 } = \sum _ { k } ( \mu _ { k } - \ddot { \mu } ) ^ { 2 } } \end{array}$ . Squaring and taking $\mathbb { E } _ { \rho }$ byields the bound. When $s = 0$ every $\mu _ { k } = \mu , s o \ \delta = 0 .$

Proposition 7 (Routing monotonicity) As CTR capacity c grows, the best-in-class routing cost $\varepsilon _ { \mathrm { r o u t e } } ^ { \star } ( c )$ is nonincreasing and the routing efficiency $\eta ^ { \star } ( c ) = 1 - \varepsilon _ { \mathrm { r o u t e } } ^ { \star } ( c ) / \Delta _ { \mathcal { F } }$ is non-decreasing.

Proof 15 $L e t \Pi ^ { \left( c \right) }$ be the routings realizable at CTR capacity $c ,$ nested so $c \leq c ^ { \prime }$ implies $\Pi ^ { ( c ) } \subseteq \Pi ^ { ( c ^ { \prime } ) }$ . The best-in-class routing cost $\begin{array} { r } { \varepsilon _ { \mathrm { r o u t e } } ^ { \star } ( c ) = \operatorname* { i n f } _ { \pi ^ { \prime } \in \Pi ^ { ( c ) } } \mathbb { E } \big [ ( \sum _ { k } ( \pi _ { k } ^ { \prime } - \pi _ { k } ) \mu _ { k } ) ^ { 2 } \big ] } \end{array}$ is the infimum of a fixed nonnegative functional over a growing set, hence non-increasing in c. With $\Delta \mathcal { F } > 0 . \hbar$ xed, $\eta ^ { \star } ( c ) = 1 - \varepsilon _ { \mathrm { r o u t e } } ^ { \star } ( c ) / \Delta _ { \mathcal { F } }$ is non-decreasing and $\bar { \Delta } \mathcal { F } \eta ^ { \star } ( c )$ is non-decreasing. The served cost adds a nonnegative estimation-excess term over thisfloor, so the served efficiency inherits the monotonicity only under an estimation-negligible or a bias-dominates-variance condition. Under cross-entropy the same infimum argument applies to the exactfunctional $\pi ^ { \prime } \mapsto \mathbb { E } [ \mathrm { K L } ( \mathrm { B e r n } ( \mu ) \lVert \mathrm { B e r n } ( \sum _ { k } \pi _ { k } ^ { \prime } \mu _ { k } ) ) ]$ , so thefloor monotonicity holds with no approximation.