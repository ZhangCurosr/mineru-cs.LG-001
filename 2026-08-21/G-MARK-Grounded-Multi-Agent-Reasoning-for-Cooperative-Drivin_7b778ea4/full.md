# G-MARK: Grounded Multi-Agent Reasoning for Cooperative Driving via Knowledge Graphs

Bhavya Gupta<sup>1</sup>, Onat Gungor<sup>2</sup>, Tajana Rosing<sup>1</sup>

<sup>1</sup>University of California, San Diego, CA, USA

<sup>2</sup>West Virginia University, Morgantown, WV, USA

{b5gupta, tajana}@ucsd.edu, onat.gungor@wvu.edu

Abstract—Autonomous driving systems must operate under partial observability, where safety-critical objects may be occluded or visible only to neighboring connected vehicles. Vehicle-tovehicle cooperation can reduce this uncertainty, but existing cooperative driving methods often compress multi-agent evidence into latent features or hidden multimodal states. As a result, they obscure which agent observed each object, whether the object is visible to the ego vehicle, and how conflicting evidence affects downstream decisions. We propose G-MARK, a grounded multi-agent reasoning framework that converts cooperative objectcentric observations into explicit provenance-aware knowledge graphs (KGs). The resulting KGs preserve object hypotheses together with their source attribution, ego-versus-partner visibility, uncertainty, conflicts, spatial relations, and planning-relevant context. G-MARK then derives a shared feature representation from these KGs, enabling lightweight task heads to support object reasoning, motion prediction, control selection, and trajectory forecasting. Compared with the state-of-the-art baseline, G-MARK improves occlusion reasoning accuracy by 42.2%, reduces control-selection error by 13.1%, and achieves comparable trajectory-planning accuracy with a 25.6× smaller structured communication payload. Our code is available at this repository.

Index Terms—Cooperative Driving, Multi-Agent Knowledge Graphs, Vehicle-to-Vehicle Communication, Motion Planning

## I. INTRODUCTION

Autonomous driving systems must make safety-critical decisions from incomplete and viewpoint-dependent observations [1]. Objects that are crucial for planning may be occluded, outside the ego vehicle’s field of view, or observable only by a connected partner vehicle [2]. Vehicle-to-vehicle cooperation can mitigate this partial observability by sharing complementary scene evidence across agents [3]. However, when the ego vehicle must reason about an object it has not directly observed, the system should retain more than the object hypothesis itself. It should also preserve its provenance: which agent observed the object, how reliable the observation is, and why the object matters for downstream planning [4].

Existing cooperative perception methods improve autonomous driving perception through early fusion, late fusion, and feature-level fusion [5]–[7]. These methods primarily aim to aggregate multi-agent observations into more accurate perception outputs, often by compressing shared evidence into latent feature maps or final detections. However, this compression can discard the evidence structure needed for downstream reasoning. Hence, downstream modules cannot infer which agent supported an object hypothesis, whether the ego vehicle directly observed the object, or how much uncertainty remains in the object estimate.

Recent language-based and vision-language driving systems study high-level driving reasoning through visual question answering, explanation generation, behavior planning, and language-conditioned control [8]–[11]. Cooperative driving systems such as V2V-LLM [12] and V2V-GoT [13] extend this paradigm to multi-agent reasoning. Yet, their intermediate states are often encoded as prompts, generated text, free-form rationales, or hidden multimodal features rather than structured object-level cooperative evidence. This limits traceability for safety-critical decisions that depend on objects outside the ego vehicle’s direct view but observed by a connected partner.

We propose G-MARK, a grounded multi-agent reasoning framework that represents cooperative driving scenes as provenance-aware knowledge graphs. Operating above the perception layer, G-MARK consumes processed multi-agent observations, associates compatible observations into shared object hypotheses, and preserves weak or unmatched observations as uncertain candidates rather than discarding them. The resulting graph explicitly stores source-agent support, egoversus-partner visibility, uncertainty, cross-agent disagreement, spatial relations, motion cues, and planning context. KGbased task heads then query this shared KG to support object grounding, occlusion reasoning, hidden-object discovery, motion prediction, control selection, and trajectory forecasting.

We evaluate G-MARK on V2V-GoT-QA, a cooperative driving benchmark spanning perception, prediction, and planning tasks. Compared with V2V-GoT [13], G-MARK improves occlusion reasoning by 42.2%, hidden-object discovery by 12.3%, and control-selection error by 13.1%. For future trajectory forecasting, G-MARK reaches near-parity in average L2 error while using approximately 25.6× lower structured communication. Our contributions are summarized as follows:

• We introduce a provenance-aware cooperative KG that preserves object hypotheses, source-agent support, visibility, uncertainty, disagreement, spatial relations, motion cues, and planning context.

• We propose task-conditioned delayed evidence fusion, which retains weak partner-only observations so down stream tasks can decide which evidence is relevant.

• We show that our approach can support object reasoning, motion prediction, control selection, and future trajectory forecasting with compact structured communication.

TABLE I  
COMPARISON OF COOPERATIVE EVIDENCE EXPOSED BY RELATED METHOD FAMILIES.
<table><tr><td>Capability</td><td>Cooperative Perception [6] / V2X Fusion [7]</td><td>Language / VLM Driving Reasoning [8]</td><td>Structured Scene and KG-based Reasoning [14]</td><td>Cooperative Language-Based Reasoning [13]</td><td>G-MARK (Ours)</td></tr><tr><td>Multi-agent evidence</td><td></td><td>△</td><td></td><td>√</td><td></td></tr><tr><td>Object provenance</td><td></td><td></td><td>△</td><td>△</td><td></td></tr><tr><td>Ego-versus-partner visibility</td><td></td><td>△</td><td></td><td>△</td><td></td></tr><tr><td>Uncertainty / conflict</td><td></td><td>△</td><td>△</td><td>△</td><td></td></tr><tr><td>Planning relevance</td><td>△</td><td></td><td>△</td><td></td><td></td></tr><tr><td>Reusable across tasks</td><td></td><td></td><td>△</td><td>√</td><td></td></tr><tr><td>Comm. efficiency</td><td>△</td><td>△</td><td></td><td>△</td><td></td></tr></table>

✓ Explicitly modeled

△ Partially exposed

## II. RELATED WORK

Prior work relevant to G-MARK spans cooperative perception and V2X fusion [5]–[7], language- and VLM-based driving reasoning [8]–[11], structured scene and KG-based reasoning [14], [15], and cooperative language-based reasoning [12], [13], [16], [17]. G-MARK targets the gap between these research families by constructing an explicit cooperative evidence representation from perception outputs. This representation preserves source support, ego-versus-partner visibility, uncertainty, disagreement, and planning context before downstream task inference. These properties are important because cooperative driving decisions depend not only on which objects are present, but also on who observed them, whether they are visible to the ego vehicle, and why they affect planning [4]. Table I compares how existing method families expose these evidence variables for downstream reasoning.

Cooperative Perception and V2X Fusion. Cooperative perception improves scene understanding beyond a single ego view point by sharing raw observations, object predictions, learned features, BEV representations, or sparse spatial messages across agents. Representative systems include Where2Comm [5], V2X ViT [7], and CoBEVT [6]. These methods provide strong perception backbones, but their fused outputs rarely expose source support, ego-versus-partner visibility, uncertainty, or disagreement as reusable object-level evidence.

Language and VLM-based Driving Reasoning. Language, vision-language, and multimodal large language models have recently been used for driving question answering, explanation, planning, and control [8]–[11]. These methods broaden autonomous-driving evaluation from perception accuracy to reasoning-oriented tasks, including object grounding, scene understanding, decision explanation, and planning. However, most are ego-centric or non-cooperative, motivating a layer that preserves source-agent support and cross-agent disagreement.

Structured Scene and KG-based Reasoning. Structured scene representations and knowledge graphs encode entities, attributes, and relations in an explicit form for visual reasoning and decision making [15]. In autonomous driving, systems such as DriveLM [8] and KLDrive [14] show that structured scene facts can support interpretable, planning-oriented reasoning. However, existing driving KG approaches primarily focus on ego-centric scenes, language-grounded QA, or high-level facts rather than cooperative V2V evidence. G-MARK extends this direction with a per-scene cooperative KG that stores object hypotheses, source-agent support, visibility, uncertainty, disagreement, and planning relevance.

Cooperative Language-Based Reasoning. Recent cooperative language-based methods extend language reasoning to multi-agent driving. LangCoop uses language for compact collaborative communication [16], CoLMDriver studies LLMbased negotiation [17], V2V-LLM formulates vehicle-to-vehicle reasoning as multimodal QA [12], and V2V-GoT organizes perception, prediction, and planning questions through graphof-thought reasoning [13]. These systems are closest to G-MARK in task scope, but their intermediate states are typically language messages, generated answers, reasoning traces, or hidden multimodal activations rather than explicit object-level evidence. G-MARK instead uses a provenance-aware KG, allowing downstream predictions to be traced to source agents, visibility facts, uncertainty, and supporting observations.

## III. G-MARK FRAMEWORK

Fig. 1 provides an overview of G-MARK, a cooperative reasoning layer that operates on top of a cooperative perception pipeline. Given processed multi-agent scene artifacts, including object detections or tracks, agent poses, confidence scores, and planning context, G-MARK constructs a provenanceaware KG for downstream reasoning. Importantly, G-MARK does not consume raw camera or LiDAR streams directly; instead, it abstracts already-processed cooperative perception evidence into a structured graph representation. The central design principle of G-MARK is to postpone irreversible fusion until the downstream reasoning task is specified. Rather than merging all agent observations into a single fused object list, G-MARK represents observing agents, local observations, and object hypotheses as linked graph entities. This structure allows different reasoning heads, such as occlusion reasoning and motion prediction, to query the same cooperative evidence while emphasizing different task-relevant factors.

![](images/76a7384771d134c6c69866583aacb83dae813d70c3496ab656edaa5baa0f6837.jpg)  
Fig. 1. Overview of the G-MARK cooperative KG reasoning framework.

G-MARK proceeds in four stages. First, each CAV’s processed perception output is instantiated as a local evidence KG containing an agent node, observation nodes, and objecthypothesis nodes before cross-agent association. Second, egoside conservative association links compatible local evidence across CAVs into shared object hypotheses, attaches observation nodes as support, and retains weak unmatched candidates. Third, each hypothesis is enriched with provenance, agentrelative visibility, uncertainty, disagreement, spatial relations, motion cues, and planning relevance. Finally, KG-based task heads query the enriched cooperative KG for object-level reasoning, motion prediction, and trajectory planning.

## A. Cooperative KG Generation

G-MARK represents each CAV’s processed perception output, including local observations and V2V-communicated object hypotheses, as local KG evidence and assembles it into an initial cooperative KG, instead of collapsing it into a flat list of fused objects [14], [15]. A flat object list can indicate that an object exists, but it does not preserve who observed it, whether it is ego-visible or partner-only, how reliable the supporting evidence is, or whether the object should influence planning. These factors are especially important under occlusion and partial observability, where safety-relevant evidence may be available only through another connected agent [7], [18], [19].

To preserve the evidence structure behind cooperative perception outputs, each local evidence graph separates agent nodes, observation nodes, and object-hypothesis nodes. An agent node stores the identity, pose, timestamp, and ego-versus partner role of the CAV that produced the local evidence. An observation node represents one agent-local object observation $o _ { t , j } ^ { ( i ) }$ , while an object-hypothesis node represents an object candidate from communicated CAV evidence or a shared hypothesis formed after cross-agent association. This separation preserves the source vehicle, supporting observation, and egoversus-partner visibility of each hypothesis. For example, if a partner observes a pedestrian hidden from the ego vehicle, the agent node identifies the partner, the observation node stores the local detection, and the hypothesis node represents the shared belief that the pedestrian exists.

Let $\mathcal { A } _ { t } = \{ a _ { 1 } , . . . , a _ { N _ { t } } \}$ denote the set of connected agents available at timestamp t, where $N _ { t }$ is the number of agents at that timestamp. Each agent $a _ { i } \in \mathcal A _ { t }$ reports a set of $M _ { i , t }$ local object observations:

$$
\mathcal { O } _ { t } ^ { ( i ) } = \{ o _ { t , 1 } ^ { ( i ) } , \ldots , o _ { t , M _ { i , t } } ^ { ( i ) } \} .
$$

Here, $M _ { i , t }$ is the number of local observations from agent $a _ { i }$ at time t, and $j = 1 , \dots , M _ { i , t }$ indexes them. Each observation is represented as

$$
o _ { t , j } ^ { ( i ) } = \left( \mathbf { x } _ { t , j } ^ { ( i ) } , \tau _ { t , j } ^ { ( i ) } , c _ { t , j } ^ { ( i ) } , a _ { i } , t , \pmb { \eta } _ { t , j } ^ { ( i ) } \right) ,
$$

where $j$ denotes an individual observation, $\mathbf { x } _ { t , j } ^ { ( i ) }$ is the object position or bounding box in a shared coordinate frame, $\tau _ { t , j } ^ { ( i ) }$ is the semantic object type, such as car, truck, or pedestrian, $c _ { t , j } ^ { ( i ) }$ is the detection confidence, and $a _ { i }$ records the source agent. The auxiliary vector $\eta _ { t , j } ^ { ( i ) }$ stores optional observation-level attributes, such as velocity, track identity, source flags, or visibility labels. Graph-level quantities such as hypothesis-level provenance, uncertainty, disagreement, and planning relevance are finalized after association. Each local observation $o _ { t , j } ^ { ( i ) }$ is instantiated as an observation node $v _ { t , i , j } ^ { O } \in \mathcal { V } _ { t } ^ { O }$ . One agent node $v _ { t , i } ^ { A } \in \mathcal { V } _ { t } ^ { A }$ is created for each connected vehicle $a _ { i } \in \mathcal A _ { t }$ . Each observation node $v _ { t , i , j } ^ { O }$ is linked to the agent node $v _ { t , i } ^ { A }$ that generated it, making the source of every observation explicit in the KG.

The CAV-local evidence is then assembled into an initial cooperative KG:

$$
\mathcal { G } _ { t } = ( \nu _ { t } , \mathcal { E } _ { t } ) ,
$$

with node set

$$
\mathcal { V } _ { t } = \mathcal { V } _ { t } ^ { A } \cup \mathcal { V } _ { t } ^ { O } \cup \mathcal { V } _ { t } ^ { H } .
$$

Here, $\mathcal { V } _ { t } ^ { A } , \mathcal { V } _ { t } ^ { O }$ , and $\mathcal { V } _ { t } ^ { H }$ denote agent nodes, observation nodes, and object-hypothesI removedis nodes, respectively.

The relation set $\mathcal { E } _ { t }$ uses typed relations for source attribution, hypothesis support, spatial context, planning context, and evidence consistency. The source relation connects an agent node to each observation node it produces, while the support relation connects an observation node to the object-hypothesis node it contributes to. Spatial relations, such as front-of, behind, left-of, right-of, and near, are computed in the ego coordinate frame. Planning-context relations, such as near-trajectory and path-relevant, use distances to the ego vehicle’s planned or predicted trajectory. Evidence-consistency relations include cooperatively-supported, indicating support from multiple agents or from a partner agent, and low-conflict, indicating geometrically consistent support. This typed relation design follows standard entity–relation KG representations [15], while provenance is stored explicitly through the support set and provenance record [20].

An object-hypothesis node $h \in \mathcal { V } _ { t } ^ { H }$ is associated with a support set of compatible observation nodes,

$$
\begin{array} { r } { S ( h ) \subseteq \mathcal { V } _ { t } ^ { O } . } \end{array}
$$

Thus, $S ( h )$ contains observation nodes, each corresponding to an original local observation $o _ { t , j } ^ { ( i ) }$ . It is represented as

$$
h = ( \mathbf { x } _ { h } , \tau _ { h } , c _ { h } , \pi _ { h } , S ( h ) , u _ { h } , \gamma _ { h } , r _ { h } ) ,
$$

where $\left( \mathbf { x } _ { h } , \tau _ { h } , c _ { h } \right)$ denote fused location, semantic type, and confidence; $( \pi _ { h } , S ( h ) )$ store supporting agents and observations; and $( u _ { h } , \gamma _ { h } , r _ { h } )$ capture uncertainty, disagreement, and planning relevance.

The provenance record is

$$
\pi _ { h } = \{ ( a _ { i } , v _ { t , i , j } ^ { O } ) : v _ { t , i , j } ^ { O } \in S ( h ) , v _ { t , i , j } ^ { O } \mathrm { ~ i s ~ p r o d u c e d ~ b y ~ } a _ { i } \} .
$$

This record keeps each object hypothesis traceable to its supporting agents and observations [20]. The KG exposes cooperative evidence variables that are often lost after immedi ate fusion. Source support is computed from the number and identity of agents in $\pi _ { h }$ . Ego-relative visibility is represented as an agent-specific relation,

$$
\begin{array} { r } { \mathrm { v i s } ( h , a _ { i } ) \in \lbrace \mathrm { v i s i b l e , o c c l u d e d , u n c e r t a i n } \rbrace , } \end{array}
$$

rather than a global label. Downstream heads use this relation to distinguish ego-visible hypotheses from partner-only or occluded hypotheses, especially for invisible-object retrieval, occlusion reasoning, and planning-aware object selection. Uncertainty $u _ { h }$ captures weak confidence, candidate status, and limited support; disagreement $\gamma _ { h }$ captures inconsistent observations; and planning relevance $r _ { h }$ measures proximity to the ego vehicle’s future path.

The output of this stage is an initial cooperative KG $\mathcal { G } _ { t }$ containing agent nodes, observation nodes, object-hypothesis nodes, typed relations, and traceable evidence variables. The following stages resolve compatible candidate hypotheses through conservative association and enrich each hypothesis with provenance, visibility, uncertainty, disagreement, and planning relevance.

## B. Conservative Association

After local evidence is assembled into the cooperative $\mathrm { K G } ,$ G-MARK associates compatible candidate hypotheses across CAVs into shared cooperative object hypotheses. Observation nodes are attached as hypothesis support, while unmatched observations are retained as weak candidates. This avoids overmerging nearby but distinct objects while preserving partneronly evidence for occlusion and planning.

Following standard tracking and data-association practice, we use a deterministic gating rule to decide which local evidence pairs are eligible for association [21]. For observation-level support attachment, let $v _ { p } ^ { O } , v _ { q } ^ { O } \in \bar { \mathcal { V } } _ { t } ^ { O }$ denote two observation nodes, with semantic types $\tau _ { p } , \tau _ { q }$ and positions $\mathbf { x } _ { p } , \mathbf { x } _ { q }$ inherited from their corresponding local observations. We define the association gate as

$$
\delta _ { t } ( v _ { p } ^ { O } , v _ { q } ^ { O } ) = { \bf 1 } \left[ \mathrm { t y p e } \_ { \bf m a t c h } ( \tau _ { p } , \tau _ { q } ) \wedge d ( { \bf x } _ { p } , { \bf x } _ { q } ) < \epsilon _ { \tau _ { p } , \tau _ { q } } \right] ,
$$

where $d ( \cdot , \cdot )$ is the distance between observation centers and $\epsilon _ { \tau _ { p } , \tau _ { q } }$ is a class-dependent association threshold. The function type match $( \tau _ { p } , \tau _ { q } )$ returns true only when the two observations have the same semantic class. We use exact semantic-class matching as a conservative design choice because false crossclass merges can create unsafe object hypotheses. For example, nearby vehicle observations can be associated, but a pedestrian and a vehicle remain separate even if they are geometrically close.

For gated pairs, G-MARK assigns an interpretable distancebased affinity score,

$$
\alpha _ { t , p q } = \left\{ \begin{array} { l l } { 1 - \frac { d ( \mathbf { x } _ { p } , \mathbf { x } _ { q } ) } { \epsilon _ { \tau _ { p } , \tau _ { q } } } , } & { \mathrm { i f } ~ \delta _ { t } ( v _ { p } ^ { O } , v _ { q } ^ { O } ) = 1 , } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

The affinity score ranks gated pairs by spatial agreement, with higher values assigned to closer pairs. The gating criterion follows standard data-association practice [21], while the normalized distance score is an interpretable heuristic used in G-MARK. When an unmatched observation node is promoted to a candidate hypothesis, its representative state inherits the observation’s semantic type and position, so the same semantic and geometric compatibility rule is used for later candidate-hypothesis merging. The association graph induced by $\delta _ { t } ( v _ { p } ^ { O } , v _ { q } ^ { \dot { O } } ) = 1$ determines which observation nodes attach as support evidence. For candidate hypotheses, the same semantic and geometric rule is applied to their representative states to identify candidates eligible for merging. The fused position $\mathbf { x } _ { h }$ is computed as a confidence-weighted average of the observation positions in $S ( h )$

TABLE II  
COOPERATIVE CONTEXT ENRICHMENT CATEGORIES IN G-MARK.
<table><tr><td>Category</td><td>Role in downstream reasoning</td></tr><tr><td>Provenance and</td><td>Stores source agents, support</td></tr><tr><td>support</td><td>counts, and ego/partner/cooperative status to trace object evidence.</td></tr><tr><td>Agent visibility</td><td>Stores ego-relative and agent-relative visibility for occlusion and hidden-object</td></tr><tr><td>Uncertainty and reliability</td><td>reasoning. Marks weak, ambiguous, or candidate hypotheses so they are</td></tr><tr><td>Cross-agent</td><td>not treated as fully reliable. Records geometric or semantic</td></tr><tr><td>disagreement</td><td>inconsistency among agents to</td></tr><tr><td>Planning relevance</td><td>expose conflicting evidence. Stores trajectory proximity, motion cues, and control context.</td></tr></table>

For each object hypothesis, G-MARK summarizes confidence using a bounded noisy-or-style aggregation [22]:

$$
c _ { h } = 1 - \prod _ { v ^ { O } \in S ( h ) } ( 1 - c ( v ^ { O } ) ) ,
$$

where $c ( v ^ { O } )$ is the confidence stored by observation node $v ^ { O }$ . We use $c _ { h }$ as a bounded confidence summary rather than as a calibrated probabilistic fusion model. Unlike immediate fusion, the association output is not only a fused object location. For every hypothesis, G-MARK stores the support set $S ( h )$ provenance record $\pi _ { h } .$ , confidence summary $c _ { h } .$ , candidate status, uncertainty, and disagreement attributes. If an observation node does not pass the association gate with any other observation node, G-MARK retains it as a candidate hypothesis with lower support and higher uncertainty. Candidate retention is a safety-oriented design choice: partner-only or partially observed objects remain available for downstream inference, but are not treated as fully reliable detections.

This module outputs well-supported hypotheses with attached observation evidence, merged compatible candidates, and retained weak candidates. The next module enriches them with provenance, visibility, uncertainty, disagreement, and planning relevance.

## C. Cooperative Context Enrichment

After hypothesis formation, G-MARK derives context features for each object hypothesis from support, visibility, relative geometry, and ego motion context. Unlike association, this stage describes how the belief was formed, who can observe it, how reliable it is, whether agents disagree, and whether it is relevant to ego motion. We group these enrichments into five categories, summarized in Table II. Together, these enrichments let downstream modules reason over object state, supporting evidence, and ego-motion relevance [8].

These enrichments are needed because the same object can matter differently across tasks: partner-only objects support hidden-object discovery, uncertainty affects prediction confidence, and path proximity affects control and forecasting.

TABLE III KG-BASED TASK-HEAD FAMILIES IN G-MARK.
<table><tr><td>Head family</td><td>Role in downstream reasoning</td></tr><tr><td>Selection and visibility reasoning</td><td>Retrieves visible, occluded, hidden, or planning-relevant objects using KG evidence.</td></tr><tr><td>Motion and interaction prediction</td><td>Predicts object or agent behavior from state, motion, support, uncertainty, and path-relative</td></tr><tr><td>Control and trajectory planning</td><td>geometry. Predicts controls or future</td></tr><tr><td></td><td>waypoints from risk, planning relevance, geometry, and trajectory context.</td></tr></table>

Formally, for each object hypothesis h, G-MARK augments the node with an enrichment vector

$$
\phi ( h ) = [ \phi _ { \mathrm { p r o v } } ( h ) , \phi _ { \mathrm { v i s } } ( h ) , \phi _ { \mathrm { u n c } } ( h ) , \phi _ { \mathrm { d i s } } ( h ) , \phi _ { \mathrm { p l a n } } ( h ) ] ,
$$

where $\phi _ { \mathrm { p r o v } } ( h ) , ~ \phi _ { \mathrm { u n c } } ( h )$ , and $\phi _ { \mathrm { d i s } } ( h )$ summarize support, provenance, confidence, and association consistency from the conservative association stage. The components $\phi _ { \mathrm { v i s } } ( h )$ and $\phi _ { \mathrm { p l a n } } ( h )$ encode ego-relative visibility and distance to the ego vehicle’s planned or predicted trajectory. The final KG-derived feature vector $\psi ( h )$ concatenates object-state attributes, such as position, semantic type, confidence, and motion cues, with the enrichment vector ϕ(h).

The output is an enriched cooperative KG in which downstream modules can query each hypothesis through ψ(h) without recomputing evidence from raw observations.

## D. Unified KG Task Heads

The enriched KG provides a shared evidence state for task heads with different output formats. As shown in Table III, we group them into object selection and visibility reasoning, motion and interaction prediction, and control and trajectory planning, following the standard perception-prediction-planning decomposition [23]. Each head operates on KG-derived features $\psi ( h )$ that combine object state, provenance, visibility, uncertainty, disagreement, spatial relations, and planning-context attributes.

For a task family k, we write the inference head as

$$
y _ { k } = R _ { k } \big ( \mathcal { G } _ { t } , \{ \psi ( h ) : h \in \mathcal { V } _ { t } ^ { H } \} \big ) ,
$$

where $\mathcal { G } _ { t }$ is the enriched cooperative KG, ψ(h) is the KGderived feature vector for hypothesis $h , R _ { k }$ is the task head, and $y _ { k }$ is the structured task output. Depending on the task, $R _ { k }$ is implemented as either a deterministic retrieval/ranking function or a lightweight learned predictor over $\psi ( h )$ . Learned heads support motion regression, scene-level prediction, control selection, and trajectory forecasting without heavy multimodal inference, while preserving evidence traceability. Table III summarizes how the same enriched KG supports different task families through a common feature interface. This design decouples representation construction from task inference: G-MARK builds one provenance-aware cooperative KG per scene, and each head selects the evidence needed for its output. In our benchmark instantiation, these heads cover object grounding, visibility and occlusion reasoning, planning-aware object selection, motion prediction, control selection, and future trajectory forecasting.

TABLE IV  
MAIN RESULTS ON V2V-GOT-QA. G-MARK IMPROVES MOST STRONGLY ON TASKS THAT REQUIRE EXPLICIT COOPERATIVE EVIDENCE, INCLUDINGOCCLUSION REASONING AND HIDDEN-OBJECT DISCOVERY.
<table><tr><td>Task</td><td>Task family</td><td>Metric</td><td>G-MARK</td><td>V2V-GoT [13]</td><td>Improvement</td></tr><tr><td>Notable Objects (Q1)</td><td>Visible-object grounding</td><td>F1@0.5m ↑</td><td>0.586</td><td>0.525</td><td>+11.6%</td></tr><tr><td>Occluding Objects (Q2)</td><td>Occlusion reasoning</td><td>F1@0.5m ↑</td><td>0.428</td><td>0.301</td><td>+42.2%</td></tr><tr><td>Invisible Objects (Q3)</td><td>Hidden-object discovery</td><td>F1@0.5m ↑</td><td>0.494</td><td>0.440</td><td>+12.3%</td></tr><tr><td>Planning Awareness (Q4)</td><td>Planning-relevance reasoning</td><td>F1@0.5m ↑</td><td>0.614</td><td>0.608</td><td>+0.9%</td></tr><tr><td>Object Motion (Q5)</td><td>Object motion prediction</td><td>L2 avg. ↓</td><td>3.822</td><td>8.050</td><td>+52.5%</td></tr><tr><td>Agent Motion (Q6)</td><td>Agent motion prediction</td><td>Accuracy ↑</td><td>0.905</td><td>0.874</td><td>+3.5%</td></tr><tr><td>Object Motion (Q7)</td><td>Object motion prediction</td><td>L2 avg. ↓</td><td>3.822</td><td>7.610</td><td>+49.8%</td></tr><tr><td>Control Settings (Q8)</td><td>Control selection</td><td>Action L1 ↓</td><td>0.076</td><td>0.088</td><td>+13.1%</td></tr><tr><td>Future Trajectory (Q9)</td><td>Future trajectory forecasting</td><td>L2 avg. ↓</td><td>2.710</td><td>2.620</td><td>-3.4%</td></tr></table>

## IV. EXPERIMENTAL ANALYSIS

## A. Experimental Setup

Benchmarks. V2V4Real provides synchronized real-world V2V scenes with connected vehicles, object annotations, agent poses, timestamps, and processed perception artifacts. V2V-GoT-QA extends this setting with typed object-selection, visibility-reasoning, motion-prediction, control-selection, and trajectory-forecasting tasks. We use the official splits, with approximately 110K training and 31K validation questions, and evaluate on the validation split using the same task definitions and reference answers as V2V-GoT [13].

Evaluation metrics. Object selection and visibility reasoning tasks use localization-aware precision, recall, and F1-score, where predictions must match ground-truth coordinates within the benchmark threshold. Motion and trajectory outputs use L2 error, agent motion uses binary accuracy, and control selection uses normalized action error. Higher is better for precision, recall, F1, and accuracy; lower is better for L2 and action error.

Baselines. Our primary baseline is V2V-GoT [13], the closest recent cooperative reasoning method for this benchmark. V2V-GoT extends V2V-LLM-style cooperative multimodal QA [12] by incorporating graph-of-thought reasoning for perception, prediction, and planning questions. It reports strong performance on future trajectory forecasting and other high-level cooperative driving tasks. We therefore compare G-MARK with the reported V2V-GoT results across all benchmark tasks, including object reasoning, motion prediction, control selection, and trajectory forecasting.

Communication efficiency is an important requirement for cooperative driving, where agents must share sufficient scene evidence without incurring excessive bandwidth overhead. For the communication–accuracy analysis, we compare G-MARK with the future-trajectory forecasting results of No Fusion, AttFuse [3], V2X-ViT [7], CoBEVT [6], V2V-LLM [12], and

V2V-GoT [13]. We restrict this analysis to future trajectory forecasting because this task directly links cooperative communication cost to downstream planning accuracy.

These comparisons evaluate whether explicit cooperative evidence improves visibility-sensitive reasoning, generalizes across multiple task families, and offers a communication-efficient alternative to language-mediated cooperative reasoning.

Implementation details. All experiments use the KG construction pipeline over processed cooperative perception artifacts, including object boxes, confidence scores, agent poses, trajectory context, and scene metadata. After KG construction, G-MARK exports a shared KG-derived feature bank with three typed views: object retrieval for Q1–Q4, motion regression for Q5/Q7/Q9, and scene-action prediction for Q6/Q8.

Object retrieval uses a shared logistic regression ranker with task one-hot features and thresholds selected from the training split. Motion prediction uses regularized regression heads, with a shared Q5/Q7 object-motion head and a separate Q9 trajectory head. Action prediction uses lightweight classifiers for Q6 and separate speed and steering heads for Q8. All heads are trained on the official training split, with validation labels used only for evaluation; experiments run on a Linux machine with 8 CPU cores and 24 GB RAM.

## B. Results

Explicit cooperative evidence improves visibility-sensitive reasoning. Table IV compares G-MARK with V2V-GoT [13], a state-of-the-art cooperative reasoning baseline for V2V-GoT-QA, across all task categories. We report Object Motion Q5 and Q7 separately because they are distinct benchmark entries, although they share the same output family. The strong gains on these entries likely reflect the structure of the output: objectmotion prediction is a numeric regression problem, and G-MARK directly exposes object state, recent motion cues, and path-relative geometry as compact KG-derived features. This representation is well suited to motion regression because ψ(h) directly encodes numeric object state, recent motion cues, and path-relative geometry. Language-mediated reasoning baselines must infer these quantities less directly before producing a numeric output.

![](images/99e1580c0f98e8970484bad05d30fcf94f1429d584014e07acbd6c5d122396f0.jpg)  
Fig. 2. Planning accuracy versus communication cost.

G-MARK improves over V2V-GoT on eight of the nine benchmark entries, with the largest gains on tasks that require explicit cooperative evidence. Occlusion reasoning improves by 42.2%, hidden-object discovery by 12.3%, and object motion prediction by 49.8%. These gains support task-conditioned delayed evidence fusion: preserving provenance, visibility, uncertainty, disagreement, and candidate status helps each task head query relevant evidence rather than relying on a single collapsed object state. The shared KG also supports prediction and control, with object motion benefiting from KG-derived motion cues and path-relative geometry, and control-selection error reduced by 13.1%. Future trajectory forecasting shows a different pattern: G-MARK reaches near-parity with V2V-GoT, with an average L2 error of 2.710m compared with 2.620m. This suggests that the KG-conditioned representation captures useful planning context, but longer-horizon forecasting remains sensitive to accumulated speed, steering, and curvature errors.

G-MARK approaches planning accuracy with substantially lower communication. We use future trajectory forecasting because it tests whether communication preserves planning-relevant evidence, including nearby objects, occluded or partner-only hazards, and path-relative interactions. Trajec tory error therefore measures useful scene information retained per unit of communication. Fig. 2 compares G-MARK with three categories of cooperative planning baselines: a noncooperative baseline (No Fusion), intermediate-fusion baselines including AttFuse [3], V2X-ViT [7], and CoBEVT [6], and language-mediated cooperative reasoning baselines including V2V-LLM [12] and V2V-GoT [13]. No Fusion has high trajectory error, while intermediate-fusion and language-mediated methods improve accuracy at roughly 0.4008–0.4068 MB per sample. G-MARK achieves a different operating point: it uses only 0.0159 MB per sample, giving about 25.2× lower communication than intermediate-fusion methods and 25.6× lower communication than V2V-GoT. This reduction comes from transmitting compact KG evidence rather than dense feature-level or language-mediated payloads. Although V2V-GoT achieves the lowest trajectory error, G-MARK attains comparable forecasting accuracy with far lower communication and explicit evidence traceability.

![](images/1076f0e050916493975574dbb4d10b5ac1009a813a04ce4b41a9c64634506270.jpg)  
Fig. 3. End-to-end latency breakdown of G-MARK by task.

![](images/15f4ebad99c1f94bab05e4c66198d78b5dad703394bcb6297de70bfdd1213e35.jpg)

Control Settings, Action L1 ↓  
![](images/8daeb0d44d63386ace206a9c2eb36b767f2f782ee09a19f443611540b187aef6.jpg)  
Fig. 4. Diagnostic ablations on evidence-sensitive tasks.

Structured KG reasoning adds minimal CPU-only overhead. Fig. 3 shows representative per-sample CPU runtimes after upstream perception has produced processed scene artifacts. We group runtime into scene loading, KG/task solving, and other overhead for graph construction and feature extraction.

The reasoning head is lightweight: task-solver time remains below 1.4 ms per sample and below 1 ms for six of the eight task types. Current-frame tasks require 10.3–14.2 ms total CPU latency, while temporal motion tasks require 32.4– 33.4 ms due to previous-frame context loading. Most runtime comes from data access and scene preparation rather than KG reasoning, indicating that the proposed reasoning layer is not the primary bottleneck. Recent VLM-based cooperative driving systems report end-to-end latencies on the order of hundreds of milliseconds [24], [25]. Although not a direct system-level comparison, G-MARK requires only about 4–6% of these reported latencies for current-frame tasks, while preserving millisecond-scale CPU reasoning.

## C. Ablation Study

We conduct construction-level ablations to evaluate which KG evidence components contribute to downstream reasoning. The task heads are kept fixed, while the KG evidence is selectively modified before inference. Fig. 4 summarizes the results for hidden-object discovery and control selection.

Partner evidence is essential for hidden-object discovery. As shown in Fig. 4, removing partner observations drops Invisible Objects F1 from 0.494 to 0.000. This confirms that hidden-object discovery depends on cooperative evidence outside the ego vehicle’s direct view; an ego-only representation cannot recover objects visible only to connected vehicles.

Provenance affects both reasoning and control. Removing provenance reduces Invisible Objects F1 from 0.494 to 0.396 and increases Control Settings Action L1 from 0.076 to 0.152, showing that source-agent support is useful for downstream decisions.

KG-derived structure helps beyond object geometry. Re placing the KG with an unstructured object-level representation reduces Invisible Objects F1 from 0.494 to 0.443 and increases Control Settings Action L1 from 0.076 to 0.089. These results suggest that visibility, uncertainty, disagreement, and path-relevance features provide useful structure for evidencesensitive cooperative reasoning.

## V. CONCLUSION

We presented G-MARK, a grounded multi-agent reasoning framework that maps processed cooperative perception evidence into a provenance-aware KG for object reasoning, prediction, control selection, and trajectory forecasting. On V2V-GoT-QA, G-MARK improves evidence-sensitive tasks such as occlusion reasoning and hidden-object discovery, while achieving near-parity in future trajectory forecasting with 25.6× lower structured communication and millisecond-scale CPU overhead. These results demonstrate the potential of provenance-aware cooperative KGs as compact, traceable, and communicationefficient reasoning layers for cooperative driving.

## ACKNOWLEDGEMENTS

This work has been funded in part by NSF, with award numbers #2112665, #2112167, #2003279, #2120019, #2211386, #2052809, #1911095 and in part by PRISM and CoCoSys, centers in JUMP 2.0, an SRC program sponsored by DARPA.

## REFERENCES

[1] Z. Zhang and J. F. Fisac, “Safe occlusion-aware autonomous driving via game-theoretic active perception,” arXiv preprint arXiv:2105.08169, 2021.

[2] C. Zhang et al., “Occlusion-aware planning for autonomous driving with vehicle-to-everything communication,” IEEE Transactions on Intelligent Vehicles, vol. 9, no. 1, pp. 1229–1242, 2023.

[3] R. Xu, H. Xiang, X. Xia, X. Han, J. Li, and J. Ma, “Opv2v: An open benchmark dataset and fusion pipeline for perception with vehicle-tovehicle communication,” in 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022, pp. 2583–2589.

[4] A. Kuznietsov et al., “Explainable ai for safe and trustworthy autonomous driving: A systematic review,” IEEE Transactions on Intelligent Trans portation Systems, vol. 25, no. 12, pp. 19 342–19 364, 2024.

[5] Y. Hu et al., “Where2comm: Communication-efficient collaborative perception via spatial confidence maps,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 4874–4886.

[6] R. Xu, Z. Tu, H. Xiang, W. Shao, B. Zhou, and J. Ma, “Cobevt: Coop erative bird’s eye view semantic segmentation with sparse transformers,” in Conference on Robot Learning. PMLR, 2023, pp. 989–1000.

[7] R. Xu, H. Xiang, Z. Tu, X. Xia, M.-H. Yang, and J. Ma, “V2x-vit: Vehicle-to-everything cooperative perception with vision transformer,” in European Conference on Computer Vision. Springer, 2022, pp. 107–124.

[8] C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, J. Beisswenger, P. Luo, A. Geiger, and H. Li, “DriveLM: Driving with graph visual question answering,” in Proceedings of the European Conference on Computer Vision (ECCV). Springer, 2024, pp. 256–274.

[9] Z. Xu, Y. Zhang, E. Xie, Z. Zhao, Y. Guo, K.-Y. K. Wong, Z. Li, and H. Zhao, “DriveGPT4: Interpretable end-to-end autonomous driving via large language model,” IEEE Robotics and Automation Letters, 2024, also available as arXiv:2310.01412.

[10] H. Shao et al., “LMDrive: Closed-loop end-to-end driving with large language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[11] X. Tian, J. Gu, B. Li, Y. Liu, Y. Wang, Z. Zhao, K. Zhan, P. Jia, X. Lang, and H. Zhao, “DriveVLM: The convergence of autonomous driving and large vision-language models,” in Proceedings of the Conference on Robot Learning (CoRL), 2024.

[12] H.-k. Chiu et al., “V2V-LLM: Vehicle-to-vehicle cooperative autonomous driving with multi-modal large language models,” 2025, accepted to ICRA 2026.

[13] ——, “V2V-GoT: Vehicle-to-vehicle cooperative autonomous driving with multimodal large language models and graph-of-thoughts,” 2025, accepted to ICRA 2026.

[14] Y. Tian, J. Zhang, Z. Wang, X. Ren, X. Yu, O. Gungor, and T. Rosing, “Kldrive: Fine-grained 3d scene reasoning for autonomous driving based on knowledge graph,” arXiv preprint arXiv:2603.21029, 2026.

[15] A. Hogan et al., “Knowledge graphs,” ACM Computing Surveys, vol. 54, no. 4, pp. 1–37, 2021.

[16] X. Gao et al., “Langcoop: Collaborative driving with language,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2025.

[17] C. Liu et al., “Colmdriver: Llm-based negotiation benefits cooperative autonomous driving,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

[18] R. Xu et al., “V2v4real: A real-world large-scale dataset for vehicleto-vehicle cooperative perception,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 13 712–13 722.

[19] Y. Li, D. Ma, Z. An, Z. Wang, Y. Zhong, S. Chen, and C. Feng, “V2X-Sim: Multi-agent collaborative perception dataset and benchmark for autonomous driving,” IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 10 914–10 921, 2022.

[20] P. Groth and L. Moreau, “PROV-Overview: An Overview of the PROV Family of Documents,” W3C Working Group Note, Apr. 2013, world Wide Web Consortium (W3C). [Online]. Available: https://www.w3.org/TR/prov-overview/

[21] Y. Bar-Shalom and T. E. Fortmann, Tracking and Data Association. Academic Press, 1988.

[22] D. Koller and N. Friedman, Probabilistic Graphical Models: Principles and Techniques. MIT Press, 2009.

[23] E. Yurtsever, J. Lambert, A. Carballo, and K. Takeda, “A survey of autonomous driving: Common practices and emerging technologies,” IEEE Access, vol. 8, pp. 58 443–58 469, 2020.

[24] J. You et al., “V2x-vlm: End-to-end v2x cooperative autonomous driving through large vision-language models,” arXiv preprint arXiv:2408.09251, 2024.

[25] J. You, P. Li, Z. Jiang, Z. Huang, R. Gan, H. Shi, and B. Ran, “V2x-realm: Vision-language model-based robust end-to-end cooperative autonomous driving with adaptive long-tail modeling,” arXiv preprint arXiv:2506.21041, 2025.