# Computational Measurement of Team-Process Phase Dynamics in Collaborative Virtual Reality

Qing Huang, Jianing Zhang, and Pooja Pol

## Abstract

Collaborative virtual reality (VR) environments make team communication observable as it unfolds, but conventional transcript analyses often summarize entire trials or divide them into fixed temporal windows. Such approaches can obscure changes in team communication and coordination over time. This article presents a computational framework for detecting and interpreting dynamic team-process phases from timestamped dialogue in a collaborative VR game. The framework uses late chunking to generate context-aware transcript representations, aggregates them into temporal chunks, and applies penalized Gaussian-kernel change-point detection to identify semantic transitions in team communication. After boundary detection, term frequency–inverse document frequency (TF–IDF), non-negative matrix factorization (NMF), and representative transcript segments provide structured evidence for phase interpretation. A locally deployed large language model (LLM) uses in-context learning to generate initial interpretations that are subsequently reviewed by humans. Independently recorded interaction logs are then aligned with the detected phases to examine corresponding task-action patterns. The evaluation compares representations, pooling strategies, segmentation methods, parameter settings, reviewed phase interpretations, and phase-aligned interaction profiles. The results show that the framework identifies coherent and interpretable phase structures while preserving traceability to the underlying transcript evidence. The correspondence between transcript-derived phases and interaction behavior further supports their relevance for analyzing collaborative activity. The framework therefore offers a transparent and transferable approach for studying temporal changes in teamwork from timestamped transcripts across collaborative task settings.

## Index Terms

virtual reality collaboration, team communication, team process, change point detection, late chunking, large language models, in-context learning

## I. INTRODUCTION

OLLABORATIVE virtual reality (VR) training and teamwork platforms provide shared task environments in which communication, coordination, and interaction can be examined as they unfold [1], [2], [3]. Because speech, task actions, and system events can be recorded as temporally aligned traces, VR also offers a useful setting for computational teamwork research [4], [5]. Team members must establish shared understanding, coordinate their actions, monitor the task situation, and resolve communication problems during collaboration [6], [7], [8]. The key analytical challenge is therefore to identify how these communicative and coordination processes change over time rather than merely summarizing the overall amount of interaction [1], [9].

Team-process research describes teamwork as a sequence of changing activities, including task interpretation, shared reference formation, action coordination, and problem response [6], [7]. Recent temporal perspectives similarly emphasize that team processes and emergent states should be examined as evolving trajectories rather than static aggregates [9], [10], [11], [12]. However, raw transcripts do not directly reveal where one collaborative phase ends and another begins. Trial-level summaries remove temporal structure, while fixed temporal windows may split coherent episodes or combine intervals with different communicative functions. This motivates methods that detect variable-length semantic phases from changes in the communication trajectory [13].

A further challenge is that short VR utterances often depend on surrounding discourse. Expressions such as “there,” “the green one,” or “wait” cannot be interpreted reliably without preceding dialogue and task context. Representing each segment independently may therefore remove information needed to distinguish its local function, which motivates context-preserving approaches such as late chunking [14], [15]. Phase interpretation must also remain traceable to the underlying evidence. Rather than asking an LLM to summarize an entire trial, structured phase-specific evidence can be used to generate candidate interpretations that are subsequently reviewed by humans [16], [17], [18], [19].

This article presents a computational framework for detecting and interpreting semantic phases from timestamped team transcripts. As shown in Fig. 1, the framework combines late-chunked contextual embeddings, temporal chunk pooling, kernel based change-point (CP) detection, structured evidence extraction, LLM-assisted interpretation with in-context learning, human review, and phase-aligned interaction analysis [4], [14], [20]. Semantic boundaries are determined before interpretation so that the labeling process cannot influence their placement. The evaluation examines contextual representation, pooling strategy, segmentation method, and parameter sensitivity, and then assesses the resulting phases through reviewed phase-process labels, non-negative matrix factorization (NMF) topic evidence, and aligned task-action indicators. The framework provides a traceable approach for studying how team communication and coordination develop during collaborative activity.

## II. RELATED WORK AND RESEARCH MOTIVATION

Research on teamwork emphasizes that team behavior should be analyzed as an evolving process rather than as a static aggregate score [6], [7]. This view is increasingly important in digital, multimodal, and virtual settings, where emergent states, coordination patterns, and resilience can change over multiple timescales [4], [5], [8], [9], [10], [11], [12], [21], [22]. For human–machine systems, training feedback and after-action review require interpretable information about when and how these processes change.

Existing computational analyses often operate either at the transcript-segment level or at the trial level. Segment-level labels can be too local to describe sustained team-process episodes, whereas trial-level summaries collapse meaningful transitions. The needed intermediate unit is a semantically coherent phase that is longer than a single transcript segment, shorter than a full task episode, and variable in duration according to the team’s communication dynamics.

Collaborative VR dialogue differs from well-formed documents because it contains short turns, deictic expressions, simultaneous player perspectives, task-specific object references, and local repair sequences. Late chunking is relevant because it encodes a longer context first and then derives local segment representations, allowing short segments to inherit discourse information [14], [15]. For boundary estimation, kernel change-point detection is suitable because it evaluates changes in the distribution of a semantic trajectory rather than relying only on adjacent local similarity [13], [20], [23].

Interpretation remains separate from boundary detection. A boundary set does not specify what communicative function a phase serves. Recent work on in-context learning and LLM-assisted communication analysis suggests a useful division of labor in which computational methods extract structured evidence, LLMs convert that evidence into reviewable candidate labels, and human reviewers assess whether the labels are supported by the evidence [16], [17], [18], [19], [24], [25]. The present study follows this evidence-first logic and treats LLM use as LLM-assisted evidence-grounded annotation within a human-reviewed workflow.

## III. RESEARCH QUESTIONS AND CLAIM BOUNDARIES

The evaluation follows four questions aligned with the measurement pipeline. RQ1 examines whether late chunking improves semantic separation relative to standard segment embeddings. RQ2 examines whether CP-kernel segmentation produces highe semantic separation than fixed-width and random-boundary reference partitions. RQ3 examines sensitivity to the embedding model, chunk pooling, kernel bandwidth, and change-point penalty. RQ4 examines whether reviewed phase-process labels can be assigned to the detected phases, whether these labels are supported by interpreted NMF topic evidence, and whether the phases can be aligned with VR interaction logs for cross-session comparisons of communication patterns and task-action profiles.

Claim boundary. The paper evaluates transcript-semantic structure, reviewed interpretability, and alignment with taskaction and performance measures. The detected phases are analytic units for examining team communication and coordination in collaborative VR rather than direct measurements of latent psychological states. External validation through trainer ratings, participant self-reports, and independent behavioral coding remains future work.

## IV. DATA AND PREPROCESSING

## A. VR Game Context and Transcript Data

The study was conducted in a collaborative VR training game designed around communication and coordination rather than individual dexterity. Participants entered a shared virtual environment as visually similar avatars and had to move task-relevant resources to target locations under time pressure. The task required team members to identify player-specific attributes, establish mutual visibility, localize resources, coordinate object manipulation, and repair misunderstandings during the ongoing activity. Each team completed two VR-game trials. Between trials, the group left VR for a facilitated reflection and strategy discussion, then returned for a second attempt. This paired design makes it possible to compare an initial encounter with the task and a later attempt after explicit team reflection. All participants provided written informed consent before participation, including consent for the recording and analysis of audio, transcript, and VR interaction data.

The data consist of timestamped German speech transcripts and timestamped VR interaction logs from these trials. Color terms in this article refer only to VR-game properties, such as object colors or color-coded player roles, and should not be interpreted as demographic, personal, or social descriptors. The dataset contains 15 teams, each of which completed two trials, resulting in 30 team-session transcripts. Player-level audio was transcribed with a Whisper-based automatic speech recognition (ASR) pipeline [26]. A transcript segment is the minimal textual unit used in the analysis and consists of one timestamped ASR output segment from a player-level audio recording. Because each participant had a separate recording channel, the player identifier was inherited from the source channel rather than inferred through diarization. The ith segment is

$$
\sigma _ { i } = ( s _ { i } , e _ { i } , c _ { i } , x _ { i } ) ,\tag{1}
$$

![](images/02f1bdfd6a7b6230fe545721164e89838b5ae9c2bdfaaae29d8a04640ddde39c.jpg)  
Fig. 1. Workflow for semantic phase detection, evidence extraction, and phase interpretation.

where $s _ { i }$ and $e _ { i }$ denote the start and end timestamps, $c _ { i }$ denotes the player identifier, and $x _ { i }$ denotes the transcript text. A team trial is represented as the ordered sequence

$$
{ \cal S } = \{ \sigma _ { 1 } , \sigma _ { 2 } , \ldots , \sigma _ { N } \} ,\tag{2}
$$

where N is the number of usable timestamped ASR segments.

Empty, non-speech, and unusable ASR fragments were removed, and silent intervals were not converted into pseudo-segments. Transcript text and timestamp alignment were manually checked. Because the focus of this study is semantic phase detection from timestamped transcripts rather than speech-to-text performance, ASR quality is not evaluated as a separate research objective, and the effect of transcription errors on the downstream results is not examined here.

## B. Contextual Segment Embeddings and Temporal Chunks

Three analysis units are used in the framework. A transcript segment is the timestamped automatic speech recognition (ASR) unit defined above. A contextual segment embedding represents the semantic content of that segment together with its surrounding discourse. A temporal chunk aggregates nearby contextual segment embeddings for change-point detection. Transcript segments retain the original textual evidence, while temporal chunks provide a more stable representation for boundary estimation.

For late chunking, neighboring transcript segments are concatenated in temporal order into a transcript block up to the encoder context limit. Let

$$
T ^ { ( b ) } = ( \tau _ { 1 } , \tau _ { 2 } , \dots , \tau _ { L } )\tag{3}
$$

denote the token sequence of transcript block b, where $\tau _ { \ell }$ is the ℓth token and L is the number of tokens in the block. A long-context encoder $f _ { \theta }$ maps this sequence to contextual token representations.

$$
( \vartheta _ { 1 } , \vartheta _ { 2 } , \ldots , \vartheta _ { L } ) = f _ { \theta } ( T ^ { ( b ) } ) .\tag{4}
$$

Here, $\vartheta _ { \ell }$ denotes the contextual representation of token $\tau _ { \ell } ,$ and $\theta$ denotes the parameters of the embedding model. The embedding of transcript segment $\sigma _ { i }$ is obtained by pooling the contextual token representations aligned with that segment.

$$
\begin{array} { r l } & { \mathbf { x } _ { i } = \mathrm { M e a n P o o l } _ { \mathrm { t o k e n } } \big ( \{ \pmb { \vartheta } _ { \ell } \mid \tau _ { \ell } \in \sigma _ { i } \} \big ) , } \\ & { \tilde { \mathbf { x } } _ { i } = \cfrac { \mathbf { x } _ { i } } { \| \mathbf { x } _ { i } \| _ { 2 } } . } \end{array}\tag{5}
$$

In these expressions, $\mathbf { x } _ { i }$ is the pooled embedding of transcript segment $\sigma _ { i } ,$ , and $\tilde { \mathbf { x } } _ { i }$ is its $L _ { 2 }$ -normalized form. Token representations are combined using mean pooling. Normalization places all segment embeddings on a comparable scale for cosine-based evaluation and kernel-based distance computation.

When a transcript exceeds the encoder context limit, it is divided into overlapping transcript blocks. If a segment appears in more than one block, its resulting embeddings are combined before normalization. This procedure preserves broader conversational context while maintaining alignment with the original ASR timestamps. The standard embedding condition processes each transcript segment independently and serves as the comparison condition for evaluating the effect of late chunking.

Change-point detection is not applied directly to individual transcript segments. These segments are often short, unevenly distributed over time, and semantically ambiguous without local context. Direct segmentation at this level would therefore be sensitive to local transcription and timing fluctuations. Instead, nearby contextual segment embeddings are aggregated into temporal chunks of width $\Delta .$

$$
\begin{array} { r l } & { I _ { k } = [ t _ { 0 } + ( k - 1 ) \Delta , ~ t _ { 0 } + k \Delta ) , } \\ & { C _ { k } = \{ \sigma _ { i } : s _ { i } \in I _ { k } \} , } \\ & { \mathbf { u } _ { k } = \mathrm { P o o l } _ { \mathrm { c h u n k } } \big ( \{ \tilde { \mathbf { x } } _ { i } ~ | ~ \sigma _ { i } \in C _ { k } \} \big ) , } \\ & { \mathbf { z } _ { k } = \frac { \mathbf { u } _ { k } } { \| \mathbf { u } _ { k } \| _ { 2 } } , \qquad \| \mathbf { u } _ { k } \| _ { 2 } > 0 . } \end{array}\tag{6}
$$

Here, $I _ { k }$ denotes the kth temporal interval, $t _ { 0 }$ is the trial start time, and $\Delta$ is the temporal chunk width. The set $C _ { k }$ contains all transcript segments whose start timestamps fall within $I _ { k } .$ . The vector $\mathbf { u } _ { k }$ is the pooled chunk representation and $\mathbf { z } _ { k }$ is its L -normalized form. Thus, after mean, maximum, minimum, or mean–maximum pooling, each nonzero chunk embedding is divided by its Euclidean norm so that its length equals one. This normalization is applied consistently across all chunk-pooling strategies before kernel computation. The ordered sequence

$$
\mathbf { Z } = \{ \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , \ldots , \mathbf { z } _ { M } \}\tag{7}
$$

forms the semantic trajectory used for boundary estimation, where M is the number of non-empty temporal chunks. Chunks without usable transcript segments are excluded rather than imputed. After the boundaries are detected, phase interpretation and human review return to the original transcript segments. Temporal chunking therefore stabilizes boundary estimation while preserving access to the underlying textual evidence.

## V. METHOD

For each trial, the computation follows the steps below.

1) Construct the ordered transcript sequence ${ \cal S } = \{ \sigma _ { 1 } , \ldots , \sigma _ { N } \}$ from player-level ASR segments.

2) Compute one normalized embedding $\tilde { \mathbf { x } } _ { i }$ for each transcript segment using either standard segment embedding or latechunked contextual embedding.

3) Aggregate the segment embeddings into temporal chunk vectors $\mathbf { Z } = \{ \mathbf { z } _ { 1 } , \dots , \mathbf { z } _ { M } \}$ using the selected pooling rule.

4) Compute the Gaussian kernel matrix over $\mathbf { Z }$ and the interval costs $\widehat { C } ( s , e )$ for admissible chunk intervals.

5) Estimate the CP-kernel boundaries $\widehat { \tau }$ by minimizing the penalized objective with exact dynamic programming.

6) Map the detected phase intervals back to the original transcript segments.

7) Convert each fixed phase interval into an evidence packet for LLM-assisted interpretation, human review, and phase-aligned interaction analysis.

## A. Gaussian-Kernel Change-Point Objective

Each trial is treated as an ordered trajectory of chunk vectors $\mathbf { z } _ { k }$ . The objective is to identify internally coherent intervals separated by changes in the embedding distribution. A Gaussian radial basis function (RBF) kernel is therefore used to compare chunks.

$$
k ( \mathbf { z } _ { i } , \mathbf { z } _ { j } ) = \exp \left( - \gamma \| \mathbf { z } _ { i } - \mathbf { z } _ { j } \| _ { 2 } ^ { 2 } \right) .\tag{8}
$$

Here $\gamma$ controls the sensitivity of the Gaussian kernel to distances between chunk embeddings and $\| \cdot \| _ { 2 }$ is the Euclidean norm. The Gaussian kernel captures changes in the geometry of the embedding distribution, including shifts in location, dispersion, and local density. Such changes can reflect transitions from exploratory discussion to focused coordination even when task vocabulary remains similar.

For a candidate interval [s, e], the empirical kernel cost is

$$
\begin{array} { l } { { \displaystyle { \widehat C ( s , e ) = \sum _ { t = s } ^ { e } k ( \mathbf { z } _ { t } , \mathbf { z } _ { t } ) } } } \\ { { \displaystyle ~ - \frac { 1 } { e - s + 1 } \sum _ { i = s } ^ { e } \sum _ { j = s } ^ { e } k ( \mathbf { z } _ { i } , \mathbf { z } _ { j } ) . } } \end{array}\tag{9}
$$

Here $\widehat C ( s , e )$ is the cost of representing chunks s through e as one phase, and $e - s + 1$ is the number of chunks in the interval. A coherent interval has a lower cost because the within-interval kernel similarities are high. An interval that spans a semantic shift is more heterogeneous and therefore receives a higher cost.

The estimated change points minimize the penalized objective

$$
\begin{array} { l } { { \displaystyle { \hat { \pmb { \tau } } = \arg \operatorname* { m i n } _ { \pmb { \tau } } J ( \pmb { \tau } ) , } } } \\ { { \displaystyle J ( \pmb { \tau } ) = \sum _ { r = 1 } ^ { K + 1 } \hat { C } ( \tau _ { r - 1 } + 1 , \tau _ { r } ) + \beta K , } } \\ { { \displaystyle ~ \pmb { \tau } : ~ 0 = \tau _ { 0 } < \dots < \tau _ { K } < \tau _ { K + 1 } = M . } } \end{array}\tag{10}
$$

In this objective, $\widehat { \tau }$ is the estimated boundary sequence, $J ( \tau )$ is the total segmentation cost, $K$ is the number of detected change points, $K + 1$ is the number of resulting phases, $\tau _ { r }$ is the end chunk of phase $r , M$ is the number of temporal chunks in the trial, and $\beta$ penalizes additional boundaries. A minimum phase length in chunks prevents uninterpretable micro-phases. The default kernel scale is derived from the median pairwise squared distance,

$$
\gamma _ { 0 } = \frac { 1 } { d _ { \mathrm { m e d } } } ,\tag{11}
$$

where $d _ { \mathrm { m e d } }$ is the median of the nonzero pairwise squared distances between chunk embeddings. Sensitivity analysis uses

$$
\gamma = \alpha \gamma _ { 0 } ,\tag{12}
$$

where $\alpha$ is the gamma multiplier reported in the sensitivity plots.

Sensitivity analysis uses late-chunked $\begin{array} { r } { \dot { 2 } \dot { \mathrm { 1 n a - v } } 2 - \mathrm { d e } . } \end{array}$ , mean pooling, $\alpha = 1 , \beta = 1$ , and $m _ { \operatorname* { m i n } } = 1$ as the reference configuration. Embedding model, embedding mode, pooling rule, gamma multiplier, and penalty are varied separately. Both α and $\beta$ are evaluated over $\{ 0 . 0 1 , 0 . 0 5 , 0 . 1 , 0 . 5 , 1 , 5 , 1 0 \}$ . For each condition, boundaries are recomputed and evaluated with the same semantic-separation score. LLM labels and interaction logs are not used for parameter selection.

In the reported experiments, the penalized objective in (10) is solved by exact dynamic programming over the ordered chunk sequence. Let $F ( t )$ denote the minimum segmentation cost for the prefix ending at chunk t. With minimum phase length $m _ { \mathrm { m i n } }$ and admissible predecessor set $\mathcal { A } ( t ) = \{ s : 0 \leq s < t , \ t - s \geq m _ { \operatorname* { m i n } } \}$ , the Bellman recursion is

$$
\begin{array} { r l } & { F ( t ) = \underset { s \in \mathcal { A } ( t ) } { \mathrm { m i n } } \Big [ F ( s ) + \widehat { C } ( s + 1 , t ) } \\ & { \qquad + \beta \mathbb { I } ( s > 0 ) \Big ] , } \\ & { F ( 0 ) = 0 . } \end{array}\tag{13}
$$

Here $\mathbb { I } ( s > 0 )$ denotes an indicator that equals one only when the candidate segment does not start at the beginning of the trial. Thus, the first phase is penalty-free, whereas every additional boundary contributes $\beta$ to the objective. For each endpoint $t ,$ the algorithm evaluates all admissible predecessor indices s, stores the best predecessor, and reconstructs the optimal boundary sequence by backtracking from $t = M$ . This exact dynamic-programming formulation is feasible because temporal chunking keeps the number of analysis units per trial moderate.

## B. Segmentation Methods and Evaluation Metric

The empirical comparison considers three segmentation methods applied to the same semantic chunk trajectory. The proposed method is CP-kernel segmentation, which minimizes the penalized kernel objective in (10) and places boundaries where the semantic distribution of the chunk sequence changes. It is therefore the only method that uses the semantic objective directly for boundary placement.

The first reference method is fixed-width segmentation. It uses the same number of phases as the CP-kernel solution, $P = K + 1$ , but replaces adaptive boundary placement with an equal-width partition.

$$
\tau _ { r } ^ { \mathrm { f w } } = \left\lfloor { \frac { r M } { P } } \right\rfloor , \qquad r = 1 , \ldots , P - 1 ,\tag{14}
$$

where $\tau _ { r } ^ { \mathrm { f w } }$ is the rth fixed-width boundary, M is the number of temporal chunks, and $P$ is the number of phases. This baseline tests whether a regular temporal grid is sufficient to recover interpretable semantic phases.

The second reference method is random-boundary segmentation. It preserves the same number of change points and the same minimum phase-length constraint as the CP-kernel solution, but it discards the semantic objective used for boundary placement. A random boundary sequence satisfies

$$
0 = \tau _ { 0 } ^ { \mathrm { r a n d } } < \tau _ { 1 } ^ { \mathrm { r a n d } } < \cdot \cdot \cdot < \tau _ { P - 1 } ^ { \mathrm { r a n d } } < \tau _ { P } ^ { \mathrm { r a n d } } = M ,\tag{15}
$$

and every resulting phase must satisfy

$$
\tau _ { r } ^ { \mathrm { r a n d } } - \tau _ { r - 1 } ^ { \mathrm { r a n d } } \geq m _ { \mathrm { m i n } } , \quad \quad r = 1 , \ldots , P .\tag{16}
$$

Here, $m _ { \mathrm { m i n } }$ is the minimum allowed phase length in chunks. For each session, 1000 admissible random boundary sequences are sampled under this constraint using a fixed pseudo-random generator with seed 42. Each random segmentation is matched to the CP-kernel output in both the number of change points and the minimum phase-length constraint. The random-reference result is summarized by the mean semantic-separation score across the 1000 repetitions together with an empirical 95% confidence interval. Using the same seeded random-number generator makes the baseline reproducible for identical inputs. This baseline therefore functions as a chance-level control that preserves segmentation granularity while removing optimized boundary placement.

These three methods are compared with the same semantic-separation score. Because boundary estimation is performed on temporal chunks, semantic separation is evaluated primarily on the chunk embeddings used by the segmentation procedure. Let $C _ { p }$ denote the set of temporal chunks assigned to phase $p .$ For phases containing at least two chunks, within-phase similarity is the average pairwise cosine similarity among chunk embeddings in $C _ { p }$

$$
W _ { p } ^ { \mathrm { c h u n k } } = \frac { 2 } { | C _ { p } | ( | C _ { p } | - 1 ) } \sum _ { \stackrel { i , j \in C _ { p } } { i < j } } \cos ( \mathbf { z } _ { i } , \mathbf { z } _ { j } ) ,\tag{17}
$$

where $W _ { p } ^ { \mathrm { c h u n k } }$ is the within-phase chunk similarity for phase $p$ and cos $( \mathbf { z } _ { i } , \mathbf { z } _ { j } )$ is the cosine similarity between the normalized chunk embeddings.

Between-phase similarity compares the chunks in $C _ { p }$ with all chunks outside that phase.

$$
B _ { p } ^ { \mathrm { c h u n k } } = \frac { 1 } { | C _ { p } | | C _ {  p } | } \sum _ { i \in C _ { p } } \sum _ { j \in C _ {  p } } \cos ( { \bf z } _ { i } , { \bf z } _ { j } ) ,\tag{18}
$$

where $C _ { \lnot p }$ denotes all temporal chunks in the same session that are not assigned to phase $p .$ The corresponding chunk-level phase separation score is

$$
\begin{array} { r } { D _ { p } ^ { \mathrm { c h u n k } } = W _ { p } ^ { \mathrm { c h u n k } } - B _ { p } ^ { \mathrm { c h u n k } } . } \end{array}\tag{19}
$$

A larger value indicates that chunks within the same phase are more semantically similar to one another than to chunks outside that phase.

If a detected phase contains only one temporal chunk, the pairwise term in $W _ { p } ^ { \mathrm { c h u n k } }$ is undefined. In this case, the score is computed from the original transcript-segment embeddings assigned to that phase rather than discarding the session. Let $S _ { p }$ denote the set of transcript segments mapped to phase $p .$ . The fallback within-phase and between-phase similarities are

$$
W _ { p } ^ { \mathrm { s e g } } = \frac { 2 } { | S _ { p } | ( | S _ { p } | - 1 ) } \sum _ { \stackrel { i , j \in S _ { p } } { i < j } } \cos ( \tilde { \bf x } _ { i } , \tilde { \bf x } _ { j } ) ,\tag{20}
$$

$$
B _ { p } ^ { \mathrm { s e g } } = \frac { 1 } { \left| S _ { p } \right| \left| S _ { \mathbf { \bar { \Phi } } p } \right| } \sum _ { i \in S _ { p } } \sum _ { j \in S _ { \mathbf { \bar { \Phi } } p } } \cos ( \tilde { \mathbf { x } } _ { i } , \tilde { \mathbf { x } } _ { j } ) ,\tag{21}
$$

with

$$
\begin{array} { r } { D _ { p } ^ { \mathrm { s e g } } = W _ { p } ^ { \mathrm { s e g } } - B _ { p } ^ { \mathrm { s e g } } . } \end{array}\tag{22}
$$

The final phase-level score is therefore

$$
D _ { p } = \left\{ { D } _ { p } ^ { \mathrm { c h u n k } } , \quad \left| C _ { p } \right| \geq 2 , \right.\tag{23}
$$

The session score is the mean of all valid phase-level scores. If a session collapses to a single detected phase, its score is set to zero because no between-phase contrast exists. For the reference methods, the phase count is matched to the CP-kernel solution for the same embedding generation method. If a valid reference segmentation cannot be constructed under this constraint, the session score is also set to zero.

## C. Evidence Extraction and Reviewed Interpretation

After boundaries are fixed, the pipeline returns to transcript evidence inside each phase. This step is deliberately separated from boundary estimation. Evidence extraction and labeling do not move, add, or remove change points. TF–IDF terms and NMF topics summarize recurrent lexical patterns, while representative transcript segments preserve direct qualitative evidence. The NMF model provides topic weights and topic-specific lexical evidence. The labels reported later are interpreted NMF topic labels, not raw lists of high-weight topic terms. For phase $p ,$ the phase-term matrix is approximated as

$$
\begin{array} { c } { { \mathbf { V } ^ { \left( p \right) } \approx \mathbf { W } ^ { \left( p \right) } \mathbf { H } ^ { \left( p \right) } , } } \\ { { \mathbf { W } ^ { \left( p \right) } , \mathbf { H } ^ { \left( p \right) } \geq 0 , } } \end{array}\tag{24}
$$

where $\mathbf { V } ^ { ( p ) }$ is the TF–IDF-weighted matrix, $\mathbf { W } ^ { ( p ) }$ contains segment-topic weights, and $\mathbf { H } ^ { \left( p \right) }$ contains topic-term weights.   
High-weight terms and representative segments form the evidence base for phase interpretation.

A locally deployed qwen3.5:9b model converts each structured evidence packet into an initial phase interpretation. The LLM serves as an annotation assistant rather than an autonomous evaluator, reducing the need to formulate every label from scratch. In-context examples provide the VR task context, expected abstraction level, and previous-phase information, thereb producing more consistent, task-grounded labels than unconstrained summarization would.

The interpretation record contains a predefined phase-process label for cross-session comparison and interpreted NMF topic labels as supporting semantic evidence. The original transcript evidence is in German, while the reported labels are provided in English for readability. The LLM generates an initial phase-process label and interpreted NMF topic labels from the structured evidence packet. Human review then verifies whether these candidate labels are supported by the phase-specific transcript evidence and task context. Reviewers may retain, revise, or replace the proposed labels, but they do not modify the detected boundaries. The reported outputs are therefore human-reviewed labels from an LLM-assisted annotation process that reduces the need to formulate every label from scratch.

Figure 2 shows the prompt template used for the initial LLM-assisted phase interpretation. The prompt combines task-level context with phase-specific evidence, including TF–IDF terms, NMF topic evidence, representative utterances, and information about the preceding phase. It explicitly instructs the model to interpret the complete detected phase rather than individual utterances and to leave the previously detected phase boundaries unchanged. The team-process function is selected from the predefined codebook, while the phase-specific focus provides a more concrete description of the communication occurring within the phase. Requiring a short evidence-based justification and multiple candidate labels makes the initial interpretation easier to inspect during subsequent human review.

![](images/793084f279eeddbad26f3119dd2df95399ea36298d50e0a2677e2890e9a1a0ba.jpg)  
Fig. 2. Context-grounded prompt template for LLM-assisted phase labeling using task context and phase-specific transcript evidence.

## VI. RESULTS

The results examine representation choice, pooling strategy, parameter sensitivity, segmentation baselines, detected phase structure, and evidence-grounded interpretation. All reported analyses use 30-second temporal chunks. In each sensitivity analysis, only the parameter under examination is varied, while all other parameters remain fixed at their final selected values.

## A. Representation Choice and the Effect of Late Context

Figure 3 compares late-chunked and standard embeddings across the tested models using mean pooling and the same semantic-separation metric. Across all tested models, late chunking produces higher semantic separation than standard embedding, indicating that contextualization has a stronger influence on the representation than model choice alone. Within the late-chunked condition, jina-v2-de achieves the highest semantic separation in both trials. These results show that preserving conversational context is the primary factor, while selecting a language-appropriate embedding model provides an additional benefit for downstream phase detection.

![](images/31d09e99630ea6a2c518a2cd0d08e5c0deecdb444dc4f010d41d01452668e5ce.jpg)  
Fig. 3. Semantic separation across embedding models and embedding modes using mean pooling.

## B. Pooling Strategy and Chunk-Trajectory Stability

Figure 4 compares chunk-pooling strategies under the late-chunked jina-v2-de representation using the same downstream change-point kernel configuration. Mean pooling produces the highest semantic separation in both trials. A temporal chunk can contain several complementary transcript segments that jointly express the local communication function. Mean pooling captures the central semantic pattern of this exchange, whereas max and min pooling can overemphasize individual segments and mean–max pooling can retain part of this sensitivity. Because temporal pooling is used only for boundary estimation and interpretation returns to the original transcript segments, mean pooling stabilizes the semantic trajectory without allowing a single segment to dominate it.

![](images/c1a9085d01c557685a9b80bfe74cbe1a6e2612848868ef99f6bff839bc5d4ae5.jpg)  
Fig. 4. Semantic separation across temporal chunk-pooling strategies using late-chunked jina-v2-de embeddings.

## C. Sensitivity to the Gaussian-Kernel Scale and Change-Point Penalty

Figure 5 presents the sensitivity of semantic separation and mean phase count to the gamma multiplier and change-point penalty under late-chunked jina-v2-de with mean pooling. Very small gamma multipliers make the Gaussian kernel less sensitive to embedding differences, causing chunks to appear more similar and reducing the number of detected transitions. Larger gamma multipliers make local semantic differences more visible and increase semantic separation, but overly large values can fragment transcripts into many short phases. The penalty controls the opposing tendency. Small penalties permit more boundaries and can increase apparent semantic separation, whereas large penalties suppress boundaries and can collapse the phase structure. The final setting of α = 1 and $\beta = 1$ provides clear semantic separation without excessive fragmentation or phase collapse. Both trials exhibit the same overall trade-off.

![](images/ab9f7c375734371119711d8ba4a8018737941cc2b5180c32d09c8b5a5d813965.jpg)  
Fig. 5. Semantic separation and mean number of detected phases across gamma multipliers and change-point penalties.

## D. Baseline Comparison of Adaptive and Reference Boundaries

Under the final configuration using late-chunked jina-v2-de, mean pooling, a gamma multiplier of α = 1, and a penalty of $\beta = 1$ , Fig. 6 compares change-point kernel segmentation with fixed-width and random-boundary references using the same semantic-separation metric. For each embedding method, the phase count produced by change-point kernel segmentation define the target number of phases for the reference methods, ensuring that all methods are compared under matched segmentation granularity. Change-point kernel segmentation achieves the highest semantic separation under late chunking, indicating greater internal semantic separation under the evaluation metric used here when contextual representation and adaptive boundary placement are combined. Fixed-width and random-boundary references preserve the same number of phases but do not use semantic changes to determine boundary locations.

## E. Phase Count, Timelines, and Interaction Profiles

The combined phase-structure and interaction results reveal descriptive differences in teamwork between the two trials. A shown in Fig. 7, Trial 1 is dominated by four-phase solutions, with such solutions observed in 11 of the 15 sessions, wherea three-phase solutions are most frequent in Trial 2, occurring in seven sessions.

The timelines in Figs. 8 and 9 show the temporal order and relative duration of the detected phases for each team. Colors indicate ordinal phase positions from Phase 1 to Phase 5 within each session. The color-coded positions visualize the sequence of semantic changes and do not imply that phases with the same number have the same communicative or coordination meaning across teams. The specific phase meanings are examined in Section VII. The team-specific boundary locations further show that the detected phases do not follow a fixed temporal template. Together, the phase-count and timeline results show that

![](images/2104adf2f037f39c980720bec86778829a20371adb70d3d209f65b2f72e8293e.jpg)  
Fig. 6. Semantic separation across segmentation methods and embedding modes.

![](images/1b59d83a85b3a7370e996e655183bae97b7f8c9a707def847762cf29550682e1.jpg)  
Fig. 7. Distribution of detected phase counts across Trials 1 and 2.

Trial 2 sessions more often contain fewer major changes in the semantic organization of team communication. Combined with the stronger task engagement observed at earlier ordinal phase positions in the interaction profiles, this pattern is consistent with more stable collaboration in Trial 2 after the initial task experience and interim reflection.

![](images/efd17363829c1e60fa261afe130b6c1cb174cc4c37d818ade77a8afa0d8a14d2.jpg)  
Fig. 8. Normalized detected-phase timelines for Trial 1 sessions.

To examine whether the transcript-derived phase structure corresponds to task behavior, independently recorded interaction events are aligned with the detected phase intervals after segmentation. These events are not used to estimate the phase boundaries.

![](images/9296dda3a9f03762be2667ac566021cfed0fc62810ac4156410ce95629ae0c8f.jpg)  
Fig. 9. Normalized detected-phase timelines for Trial 2 sessions.

For phase p in session i, the manipulation count is defined as

$$
M _ { i p } = G _ { i p } + D _ { i p } ,\tag{25}
$$

where $G _ { i p }$ and $D _ { i p }$ denote the numbers of object-grab and object-drop events assigned to that phase.

The manipulation rate is

$$
R _ { i p } ^ { \mathrm { m a n i p } } = \frac { M _ { i p } } { d _ { i p } } ,\tag{26}
$$

where $d _ { i p }$ is the phase duration in minutes. This measure represents task-action intensity.

The score rate is

$$
R _ { i p } ^ { \mathrm { s c o r e } } = \frac { Q _ { i p } } { d _ { i p } } ,\tag{27}
$$

where $Q _ { i p }$ is the number of scoring events in the phase. This measure represents scoring progress per minute.

Score per manipulation is

$$
E _ { i p } = \frac { Q _ { i p } } { M _ { i p } } , \qquad M _ { i p } > 0 .\tag{28}
$$

This measure represents how effectively object manipulations translate into successful scoring outcomes. When no manipulation event occurs within a phase, $E _ { i p }$ is treated as undefined rather than assigned a value of zero.

The interaction profiles in Fig. 10 indicate stronger task engagement at earlier ordinal phase positions in Trial 2. Manipulation rate rises quickly and peaks in Phase 3, whereas Trial 1 shows a more gradual increase across the commonly observed phase positions. Score rate follows a similar pattern. In Trial 2, the score rate increases sharply by Phase 2 and remains comparatively high through Phases 3 and 4, whereas the score rate in Trial 1 increases more gradually. Score per manipulation also reaches a relatively high and stable level at earlier ordinal phase positions in Trial 2, suggesting that teams convert object manipulation into successful outcomes more consistently in the earlier detected phases.

Taken together, the three measures suggest stronger task engagement, scoring progress, and more stable manipulation efficiency at earlier ordinal phase positions in Trial 2. This pattern is consistent with teams becoming familiar with the virtual environment, task procedure, and available objects during the first trial and applying that experience more effectively in the second.

The correspondence between the transcript-derived phase structure in Figs. 7–9 and the independently recorded interaction profiles in Fig. 10 provides complementary behavioral evidence for the relevance of the detected phases. Because phase boundaries are estimated exclusively from transcript semantics and interaction measures are aligned only afterward, the observed correspondence indicates that the transcript-derived phases are associated with differences in collaborative task activity.

![](images/ca85eb75f5bcb575b6b12b78aaff1f0b2b26def8ee6f2617fff1ab236456542c.jpg)  
Fig. 10. Phase-aligned interaction measures by ordinal phase position in Trials 1 and 2.

## VII. EVIDENCE-GROUNDED PHASE INTERPRETATION

After CP-kernel segmentation, each fixed phase is interpreted using its time span, TF–IDF terms, NMF topic evidence, representative German transcript segments, and previous-phase context. A locally deployed LLM generates an initial English interpretation that is subsequently reviewed by humans. Each phase receives a reviewed phase-process label and interpreted NMF topic labels, while boundary placement remains unaffected by the interpretation step.

Table I summarizes the most frequent phase-process labels by ordinal position and trial. Tables II and III show the phase sequences and supporting NMF topic evidence for all 15 teams in Trial 1 and Trial 2, allowing within-team changes across trials to be examined.

In Trial 1, Object/Resource Coordination and Player-Attribute Assignment are most frequent in the early phases, while Spatial Orientation and Localization becomes more prominent in Phases 3 and 4. At the aggregate level, this distribution is consistent with a shift from establishing player roles, identifying resources, and clarifying task requirements toward spatially grounded execution, and it corresponds to the gradual increase in manipulation and scoring shown in Fig. 10.

In Trial 2, Player-Attribute Assignment is most frequent in Phase 1, followed by Mutual Visibility Check and Spatial Orientation and Localization in Phases 2 and 3. Together with the stronger task engagement observed at earlier ordinal phase positions in Fig. 10, this pattern suggests that role establishment, shared visibility, and spatial relations are more prominent in the earlier detected phases of Trial 2 following the initial task experience and interim reflection.

Phase 5 is excluded from the cross-trial interpretation because it occurs in only one Trial 1 session and two Trial 2 sessions. The example sequences also show that teams do not follow a single fixed trajectory and that the same phase-process label can be supported by different NMF topic evidence.

## VIII. DISCUSSION, LIMITATIONS, CONCLUSION, AND ETHICS

This study presents a phase-centered framework for detecting and interpreting dynamic team-process phases from times tamped collaborative VR dialogue. It combines context-aware transcript representations, adaptive semantic segmentation, evidence-grounded LLM-assisted interpretation, and phase-aligned interaction analysis. Because boundaries are fixed before interpretation, the assigned labels cannot influence their placement. Among the tested configurations, the configuration combining late-chunked jina-v2-de, mean pooling, and CP-kernel segmentation achieves the highest semantic separation. The main contribution is the detection of variable-length semantic phases without relying on fixed temporal windows or trial level summaries. Each phase remains traceable to the original transcript segments, TF–IDF terms, and NMF topic evidence, allowing the resulting labels to be reviewed against their underlying evidence. The LLM supports this process by generating initial candidate labels, while final interpretations remain subject to human review.

TABLE I  
MOST FREQUENT REVIEWED PHASE-PROCESS LABELS AT EACH ORDINAL PHASE POSITION IN TRIALS 1 AND 2.
<table><tr><td>Phase position</td><td>Trial 1 labels</td><td>Trial 2 labels</td></tr><tr><td>Phase 1</td><td>Object/Resource Coordination (5/15), Player-Attribute Assignment (4/15)</td><td>Player-Attribute Assignment (6/15), Player Identity Grounding (3/15)</td></tr><tr><td>Phase 2</td><td>Player-Attribute Assignment (4/15), Object/Resource Coordination (3/15)</td><td>Mutual Visibility Check (5/15), Player-Attribute Assignment (5/15)</td></tr><tr><td>Phase 3</td><td>Spatial Orientation and Localization (8/15), Player-Attribute Assignment (3/15) Mutual Visibility Check (4/14), Spatial Orientation and Localization (4/14)</td><td></td></tr><tr><td>Phase 4</td><td>Spatial Orientation and Localization (5/12), Player-Attribute Assignment (4/12) Spatial Orientation and Localization (4/7)</td><td></td></tr><tr><td>Phase 5</td><td>Player Identity Grounding (1/1)</td><td>Spatial Orientation and Localization (1/2), Object/Resource Coordination (1/2)</td></tr></table>

TABLE II  
REVIEWED TEAM-PROCESS FUNCTIONS AND SUPPORTING NMF TOPIC LABELS ACROSS DETECTED PHASES IN TRIAL 1.
<table><tr><td>Index Phase 1</td><td></td><td>Phase 2</td><td>Phase 3</td><td>Phase 4</td><td>Phase 5</td></tr><tr><td rowspan="2">ID01</td><td rowspan="2">Player-Attribute Assignment</td><td rowspan="2">Spatial Orientation and Localization</td><td rowspan="2">Mutual Visibility Check</td><td rowspan="2">Spatial Orientation and Localization</td><td rowspan="2"></td></tr><tr><td>NMF: T1: Resource identification T2: Color iden- NMF: T1: Numerical Sequence Coordination T2: NMF: T1: Targeted visibility checks T2: Visual NMF: T1: Spatial Orientation via Gestures T2:</td></tr><tr><td rowspan="2">ID02</td><td rowspan="2">tity verification Plaver-Attribute Assignment</td><td rowspan="2">Strategic and Empathic Approach T3: Spatial Re- perception verification orientation Player-Attribute Assignment</td><td rowspan="2">Player-Attribute Assignment</td><td rowspan="2">Resource color identification</td><td rowspan="2"></td></tr><tr><td>Plaver-Attribute Assignment</td></tr><tr><td rowspan="2"></td><td rowspan="2">source color identification T3: Visual Perception identity declaration</td><td rowspan="2"></td><td rowspan="2">identification</td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td>NMF: T1: Visual filter identification T2: Re- NMF: T1: Movement Synchronization T2: Color NMF: T1: Avatar identification T2: Avatar visual NMF: T1: Resource color identification T2: Re- source Color Identification</td></tr><tr><td rowspan="2">ID03</td><td rowspan="2">Calibration Object / Resource Coordination</td><td rowspan="2">Player-Attribute Assignment</td><td rowspan="2">Spatial Orientation and Localization</td><td rowspan="2">Spatial Orientation and Localization</td><td rowspan="2"></td></tr><tr><td></td></tr><tr><td rowspan="2"></td><td rowspan="2">Tower construction planning</td><td rowspan="2">detection and confirmation T3: Spatial resource Resource Identification localization</td><td rowspan="2"></td><td rowspan="2">source visibility verification T3: Resource color</td><td rowspan="2"></td></tr><tr><td>identification</td></tr><tr><td rowspan="2">ID04</td><td rowspan="2">Player-Attribute Assignment count verification</td><td rowspan="2">Object / Resource Coordination</td><td rowspan="2">Spatial Orientation and Localization</td><td rowspan="2">Player-Attribute Assignment</td><td rowspan="2">Player Identity Grounding NMF: T1: Task procedure inquiry T2: Resource NMF: T1: Task Execution Breakdown T2: Object NMF: T1: Shared Visual Perception T2: Repeated NMF: T1: Green Resource Identification T2: NMF: T1: Avatar color identification T2: Col</td></tr><tr><td>Failed resource retrieval T3: Player identification identity verification T3: Directives for green gue</td></tr><tr><td rowspan="2">ID05</td><td>Action Instruction and Turn Coordination</td><td>manipulation attempts Player Identity Grounding</td><td>Rejection of Incorrect Cues</td><td>queries</td><td rowspan="2">location</td></tr><tr><td></td><td>NMF: T1: Action Synchronization T2: Frag- NMF: T1: Execution Speed Regulation T2: Visual NMF: T1: Resource color identification T2: Turn- NMF: T1: Visual Visibility Checks T2: Visual</td><td></td><td>Spatial Orientation and Localization</td><td>Mutual Visibility Check</td></tr><tr><td rowspan="3">ID06</td><td rowspan="3">tress T4: Personal resource identification</td><td>mented Attention Signals T3: Expressions of Dis- Grounding Struggle</td><td></td><td>taking regulation</td><td>Grounding T3: Visual resource identification</td></tr><tr><td>Player Identity Grounding Object / Resource Coordination</td><td></td><td>Player Identity Grounding</td><td rowspan="2"></td></tr><tr><td>of Surprise T3: Collective Tower Construction</td><td>NMF: T1: Identity verification T2: Expressions NMF: T1: Resource Path Selection T2: Identity NMF: T1: Color-based Identity Declaration T2:</td><td></td></tr><tr><td rowspan="3">ID07</td><td rowspan="3">Object / Resource Coordination</td><td>verification via color Repair and Re-grounding</td><td>Mutual Visibility Establishment</td><td>Spatial Orientation and Localization</td><td></td></tr><tr><td></td><td>NMF: T1: Spatial Orientation T2: Dice placement NMF: T1: Waiting for external cues T2: Exclu- NMF: T1: Resource location queries T2: Vi-</td><td></td><td></td></tr><tr><td></td><td></td><td>sual Grounding Confirmation T3: Visual resource</td><td></td></tr><tr><td rowspan="3">ID08</td><td rowspan="3">Object / Resource Coordination NMF: T1: Spatial orientation and identification NMF: T1: Color-location mapping T2: Resource NMF: T1: Visual property verification T2: Color- NMF: T1: Resource color identification T2: Spa-</td><td>Spatial Orientation and Localization</td><td>Player-Attribute Assignment identification</td><td>Mutual Visibility Check</td><td></td></tr><tr><td>T2: Resource Availability Negotiation location identification</td><td>based resource identification</td><td></td><td></td></tr><tr><td>Mutual Visibility Check Task-Constraint Grounding</td><td></td><td>ification</td><td>tial location verification T3: Visual Contact Ver-</td></tr><tr><td rowspan="3">ID09</td><td rowspan="3">NMF: T1: Expressions of shock T2: Realization NMF: T1: Resource selection negotiation T2: NMF: T1: Spatial orientation of resources T2: NMF: T1: Ownership identification T2: Member</td><td></td><td></td><td>Object / Resource Coordination Object / Resource Coordination</td><td></td></tr><tr><td></td><td>and acknowledgment T3: Visual Presence Verifi- Graspability Verification T3: Avatar Identity Ver- Positive performance acknowledgment</td><td></td><td>location verification T3: Resource Inventory Dec-</td></tr><tr><td>Object / Resource Coordination Repair and Re-grounding</td><td>ification T4: Repeated Rejection Signals</td><td>laration T4: Graspability Verification</td><td></td></tr><tr><td rowspan="3">ID10</td><td rowspan="3">NMF: T1: Object Location Grounding T2: Con- NMF: T1: Exit and Visibility Loss T2: Expres- NMF: T1: Purple resource location search T2: NMF: T1: Color resource identification T2: Iden- tact establishment</td><td></td><td></td><td>Spatial Orientation and Localization</td><td>Player-Attribute Assignment</td></tr><tr><td>sions of Confusion</td><td></td><td>Color assignment verification</td><td>tity verification queries T3: Spatial Proximity</td></tr><tr><td>Player-Attribute Assignment</td><td></td><td>Coordination</td><td></td></tr><tr><td rowspan="3">ID11</td><td rowspan="3">NMF: T1: Permission assignment by color T2: NMF: T1: Resource color assignment T2: Re- NMF: T1: Confirming purple resource assign-</td><td>Player-Attribute Assignment</td><td></td><td>Player-Attribute Assignment</td><td></td></tr><tr><td>Seeking opinions and clarification T3: Resource source identification queries</td><td></td><td></td><td></td></tr><tr><td>identification T4: Failed resource identification</td><td>ment T2: Verbal color grounding</td><td></td><td></td></tr><tr><td rowspan="3">ID12</td><td rowspan="3">NMF: T1: Resource identification T2: Spatial NMF: T1: Avatar color identification T2: Color NMF: T1: Resource assignment queries T2: NMF: T1: Resource location queries T2: Re- convergence coordination</td><td>Player Identity Grounding Player Identity Grounding</td><td></td><td>Spatial Orientation and Localization</td><td>Spatial Orientation and Localization</td></tr><tr><td>identity verification</td><td></td><td></td><td></td></tr><tr><td></td><td>Movement coordination</td><td>source attribute verification T3: Resource iden-</td><td></td></tr><tr><td rowspan="3">ID13</td><td rowspan="3">Player Identity Grounding</td><td>Player-Attribute Assignment</td><td></td><td></td><td>tification T4: Spatial Reorientation</td></tr><tr><td>NMF: T1: Blue resource identification T2: Mu- NMF: T1: Resource Identity Confirmation T2: NMF: T1: Player Identity Identification T2: Red NMF: T1: Avatar Identity Confirmation T2: Ac-</td><td></td><td>Spatial Orientation and Localization</td><td>Player-Attribute Assignment</td></tr><tr><td>Identity declaration T3: Identity verification resource identification</td><td></td><td></td><td></td></tr><tr><td rowspan="3">ID14</td><td rowspan="3">Object / Resource Coordination</td><td>tual Orientation Struggle queries</td><td></td><td></td><td>tion Synchronization T3: Color-based role assign-</td></tr><tr><td>Object / Resource Coordination</td><td></td><td>Object / Resource Coordination</td><td>ment Spatial Orientation and Localization</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3"></td><td rowspan="3">pair T4: Visual Perception Verification ID15 Mutual Visibility Check</td><td>resource selection T3: Resource identification re- Resource Possession Confirmation</td><td></td><td>NMF: T1: Spatial location verification T2: Visual NMF: T1: Resource identification by color T2: NMF: T1: Action Synchronization T2: Explicit NMF: T1: Grounding orange resource locations Waving Signaling T3: Team member identifica- T2: Spatial Resource Identification</td><td></td></tr><tr><td></td><td>tion</td><td></td><td></td></tr><tr><td>Mutual Visibility Check NMF: T1: Visual perception confirmation T2: NMF: T1: Visual Grounding Verification T2: NMF: T1: Spatial location verification T2: Red NMF: T1: Resource identification T2: Commit-</td><td>Spatial Orientation and Localization</td><td>Spatial Orientation and Localization</td><td></td></tr></table>

The detected trajectories show how team communication shifts across grounding, orientation, coordination, and task execution. Independently recorded interaction and performance logs are aligned only after segmentation, allowing the transcript-derived phases to be examined in relation to corresponding task activity. This may support future after-action review and phase-specific feedback in collaborative training.

Several limitations remain. Semantic separation is an internal evaluation metric because boundary detection and evaluation use the same embedding representation. In contrast, the independently recorded interaction and performance logs provide complementary behavioral evidence that is aligned only after segmentation. The reviewed labels should nevertheless be understood as evidence-grounded interpretations rather than direct measurements of latent psychological states or replacements for independent behavioral coding. Future work should compare the detected phases with trainer ratings, participant self-reports, independent behavioral annotations, physiological signals, and broader performance outcomes.

TABLE III  
REVIEWED TEAM-PROCESS FUNCTIONS AND SUPPORTING NMF TOPIC LABELS ACROSS DETECTED PHASES IN TRIAL 2.
<table><tr><td>Index Phase 1</td><td></td><td>Phase 3</td><td>Phase 4</td><td>Phase 5</td></tr><tr><td rowspan="2">ID01</td><td>Player Identity Grounding Mutual Visibility Check NMF: T1: Avatar color identification T2: Identity NMF: T1: Visual Perception Grounding T2: Re- NMF: T1: Visual confirmation of yellow resource</td><td>Mutual Visibility Check</td><td></td><td></td></tr><tr><td>declaration</td><td>source identification via color</td><td>T2: Resource location confirmation T3: Visual grounding statements</td><td></td></tr><tr><td rowspan="2">ID02</td><td>Spatial Orientation and Localization Player-Attribute Assignment NMF: T1: Target location inquiry T2: Spatial NMF: T1: Avatar Identity Verification T2: Avatar NMF: T1: Color identity confirmation T2: Visual</td><td>Player Identity Grounding</td><td></td><td></td></tr><tr><td>resource identification Player-Attribute Assignment</td><td>Identity Verification Player-Attribute Assignment</td><td>resource identification Mutual Visibility Check</td><td></td></tr><tr><td rowspan="2"></td><td>reference confirmation Color Identification</td><td></td><td>Spatial Orientation and Localization NMF: T1: Shared Visual Perception T2: Spatial NMF: T1: Shared Visual Perception T2: Visual NMF: T1: Visual Contact Establishment T2: Non- NMF: T1: Resource color identification T2: NMF: T1: Spatial resource orientation T2: Ge verbal visual signaling T3: Spatial orientation via Avatar Movement Coordination T3: Interactional tural signaling for attention T3: Red containe</td><td>Spatial Orientation and Localization</td></tr><tr><td></td><td>color cues</td><td>tion</td><td>Acknowledgment T4: Spatial resource identifica- identification</td></tr><tr><td rowspan="2">ID04</td><td>Plaver-Attribute Assignment Mutual Visibility Check NMF: T1: Resource identification and confirma- NMF: T1: Resource Identification T2: Task exe- NMF: T1: Resource identification T2: Green Re-</td><td>Mutual Visibility Check</td><td></td><td></td></tr><tr><td>tion T2: Resource verification T3: Visual Ground- cution feedback T3: Resource identification fail- source Identification T3: Visual Grounding Veri- ing and Orientation T4: Visual perspective shar- ure T4: Visual Occlusion Reporting</td><td></td><td></td><td></td></tr><tr><td rowspan="2">ID05</td><td>ing</td><td>fication</td><td></td><td></td></tr><tr><td>Spatial Orientation and Localization NMF: T1: pink gate identification T2: Yellow NMF: T1: Visual Presence Verification T2: Re- NMF: T1: Mutual Visual Grounding T2: Non-</td><td>Mutual Visibility Check</td><td>Mutual Visibility Check</td><td></td></tr><tr><td rowspan="2">ID06</td><td>object identification Mutual Visibility Check</td><td>quests for Pauses Mutual Visibility Check</td><td>verbal gesture signaling Object / Resource Coordination</td><td></td></tr><tr><td>Spatial Orientation and Identification T3: Vi- Task execution feedback</td><td></td><td>NMF: T1: Resource location identification T2: NMF: T1: Visual Grounding Confirmation T2: NMF: T1: Resource need inquiry T2: Resource NMF: T1: Resource identification T2: Identity identifier verification Grounding via Name</td><td>Object / Resource Coordination</td></tr><tr><td rowspan="2">ID07</td><td>sual Contact Establishment T4: Visual Grounding Confirmation Mutual Visibility Check</td><td></td><td></td><td></td></tr><tr><td>NMF: T1: Mutual Visibility Verification T2: Spa- NMF: T1: Visual Grounding T2: Spatial bound- tial resource localization T3: Resource acquisition ary definition</td><td>Spatial Orientation and Localization</td><td></td><td></td></tr><tr><td rowspan="2">ID08</td><td>confirmation Player-Attribute Atssignment</td><td></td><td>Player Identity Grounding</td><td></td></tr><tr><td>identity declaration T3: Spatial orientation check Resource assignment confirmation</td><td>Player-Attribute Assignment</td><td>NMF: T1: pink resource identification T2: Color NMF: T1: orange Resource Identification T2: NMF: T1: Green resource identification T2: Vi- NMF: T1: Resource location assignment T2: Spa- NMF: T1: Resource color assignment T2: Re</td><td>Spatial Orientation and Localization Object / Resource Coordination</td></tr><tr><td rowspan="2">ID09</td><td>T4: Blue Zügel Resource Identification</td><td></td><td>sual Grounding Confirmation T3: Resource at- tial alignment invitation tribute identification T4: Resource assignment</td><td>source acquisition confirmation T3: Movemen</td></tr><tr><td>Mutual Visibility Check Spatial Orientation and Localization NMF: T1: Visual Perception Verification T2: Blue NMF: T1: Resource location queries T2: Visual NMF: T1: Resource color identification T2: Vi- NMF: T1: Color identity confirmation T2: Visual</td><td></td><td>verification Spatial Orientation and Localization Mutual Visibility Check</td><td>Coordination</td></tr><tr><td rowspan="2">ID10</td><td>cube identification</td><td></td><td>sual resource orientation</td><td></td></tr><tr><td>Player-Attribute Assignment NMF: T1: Spatial resource orientation T2: Color- NMF: T1: Resource identification T2: Visual Ref- NMF: T1: Identity and Color Mapping T2: Color NMF: T1: Resource assignment by color T2:</td><td>observation reporting Spatial Orientation and Localization</td><td>Player Identity Grounding Player-Attribute Assignment</td><td>Presence Verification</td></tr><tr><td rowspan="2">ID11</td><td>based Identity Verification</td><td>erence Grounding</td><td>identity confirmation T3: Green resource identi- Color-based resource identification</td><td></td></tr><tr><td>Player-Attribute Assignment Player-Attribute Assignment</td><td></td><td>fication T4: Avatar identification</td><td></td></tr><tr><td rowspan="2"></td><td>NMF: T1: Visual object identification T2: Re- NMF: T1: Player Identity Verification T2: Role NMF: T1: Resource color identification T2: Re- NMF: T1: Resource color identification T2: Blue</td><td></td><td>Spatial Orientation and Localization</td><td>Spatial Orientation and Localization</td></tr><tr><td>source color identification identification via color</td><td></td><td>source holder identification</td><td></td></tr><tr><td rowspan="2">ID12</td><td>Player Identity Grounding</td><td></td><td>identification</td><td>container identification T3: Resource location</td></tr><tr><td>NMF: T1: Avatar identification T2: Green re- NMF: T1: Resource location queries T2: Identity NMF: T1: Resource location queries T2: Re-</td><td>Spatial Orientation and Localization</td><td>Spatial Orientation and Localization</td><td></td></tr><tr><td rowspan="2">source identification ID13</td><td></td><td></td><td></td><td></td></tr><tr><td>Player-Attribute Atssignment Player-Attribute Assignment</td><td>confirmation T3: Resource color identification</td><td>source location identification</td><td></td></tr><tr><td rowspan="2"></td><td>NMF: T1: Avatar identification by color T2: NMF: T1: Resource identification via color T2: NMF: T1: Blue task completion confirmation T2:</td><td></td><td>Player-Attribute Assignment</td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">ID14</td><td>Color identity identification</td><td>Temporal Coordination T3: Grounding Purple Movement synchronization</td><td></td><td></td></tr><tr><td>Player Identity Grounding Mutual Visibility Check</td><td>Player T4: Personal resource identification</td><td></td><td></td></tr><tr><td rowspan="2"></td><td>NMF: T1: Resource identification T2: Visual re- NMF: T1: Visual Grounding T2: Team Identity NMF: T1: pink Resource Identification T2: Mem- NMF: T1: Grounding orange gate location T2:</td><td></td><td>Object / Resource Coordination Spatial Orientation and Localization</td><td></td></tr><tr><td>source identification</td><td></td><td></td><td></td></tr><tr><td rowspan="2">ID15</td><td></td><td>Identification T3: Spatial Identity Verification T4: ber location queries</td><td>Location confirmation</td><td></td></tr><tr><td>Spatial Orientation and Localization</td><td>Visual resource identification</td><td>Spatial Orientation and Localization</td><td></td></tr><tr><td rowspan="2"></td><td></td><td>Spatial Orientation and Localization</td><td></td><td></td></tr><tr><td>source identification queries T3: Visual resource orientation localization</td><td>NMF: T1: Visual Object Verification T2: Re- NMF: T1: Spatial orientation T2: Spatial resource NMF: T1: Resource acquisition confirmation T2:</td><td>Resource color identification</td><td></td></tr></table>

The workflow also raises data-protection requirements because audio, transcripts, timestamps, and speaker information may contain identifiable data. Relevant safeguards include anonymization, data minimization, secure storage, local processing where possible, and continued human oversight of LLM-assisted interpretation. The framework is intended for research and training analysis rather than for ranking or diagnosing individual participants.

In conclusion, the framework transforms timestamped VR dialogue into interpretable semantic phase trajectories that remain traceable to transcript evidence and can be examined alongside task-action profiles. It provides a reproducible basis for analyzing how team communication and coordination change over time in collaborative human–machine systems.

## CONFLICT OF INTEREST

The authors declare that they have no conflicts of interest.

## REFERENCES

[1] D. Sparks, R. Begum, F. Aqlan, J. Saleem, and M. DeCaro, “Exploring team dynamics in virtual reality environments,” IISE Transactions on Occupational Ergonomics and Human Factors, pp. 1–15, 2025.

[2] J. Wolfartsberger, J. Zenisek, and N. Wild, “Supporting teamwork in industrial virtual reality applications,” Procedia Manufacturing, vol. 42, pp. 2–7, 2020.

[3] M. Lehmann, J. Mikulasch, H. Poimann, J. Backhaus, S. Konig, and T. M ¨ uhling, “Training and assessing teamwork in interprofessional virtual reality-¨ based simulation using the teamstepps framework: Protocol for a randomized pre-post intervention study,” JMIR Research Protocols, vol. 14, p. e68705, 2025.

[4] B. Goldberg, R. Spain, A. Sinatra, K. Brawner, and R. Sottilare, “Towards a multimodal data-driven framework for adaptive coaching in collaborative simulation-based training,” in Workshop on Artificial Intelligence in Support of Guided Experiential Learning, 2024, metadata transcribed from uploaded manuscript.

[5] N. Lehmann-Willenbrock and H. Hung, “A multimodal social signal processing approach to team interactions,” Organizational Research Methods, vol. 27, no. 3, pp. 477–515, 2024.

[6] M. A. Marks, J. E. Mathieu, and S. J. Zaccaro, “A temporally based framework and taxonomy of team processes,” Academy of Management Review, vol. 26, no. 3, pp. 356–376, 2001.

[7] N. J. Cooke, J. C. Gorman, C. W. Myers, and J. L. Duran, “Interactive team cognition,” Cognitive Science, vol. 37, no. 2, pp. 255–285, 2013.

[8] J. N. Lane, P. M. Leonardi, N. S. Contractor, and L. A. DeChurch, “Teams in the digital workplace: Technology’s role for communication, collaboration, and performance,” Small Group Research, vol. 55, no. 1, pp. 139–183, 2024.

[9] F. Klonek, M. Twemlow, M. Tims, and S. K. Parker, “It’s about time! understanding the dynamic team process-performance relationship using micro and macroscale time lenses,” Group & Organization Management, vol. 50, no. 5–6, pp. 1660–1702, 2025.

[10] E. Georganta, C. S. Burke, S. Merk, and F. Mann, “Understanding how team process-sequences emerge over time and their relationship to team performance,” Team Performance Management, vol. 27, no. 3/4, pp. 159–174, 2021.

[11] J. Li and C.-R. Li, “Understanding the change trajectories of team transition and action processes over time: A regulatory focus perspective,” Journal of Organizational Behavior, vol. 46, pp. 850–866, 2025.

[12] B. Fyhn, V. Schei, and T. E. Sverdrup, “Taking the emergent in team emergent states seriously: A review and preview,” Human Resource Management Review, vol. 33, p. 100928, 2023.

[13] J. L. Harrison, S. A. Jain, T. Dunbar, J. C. Gorman, and S. Varma, “Toward automated detection of phase changes in team collaboration,” in Proceedings of the 44th Annual Conference of the Cognitive Science Society, 2022, pp. 2357–2363.

[14] M. G”unther, I. Mohr, D. J. Williams, B. Wang, and H. Xiao, “Late chunking: Contextual chunk embeddings using long-context embedding models,” arXiv preprint arXiv:2409.04701, 2024.

[15] C. Merola and J. Singh, “Reconstructing context: Evaluating advanced chunking strategies for retrieval-augmented generation,” in Knowledge-Enhanced Information Retrieval: Second International Workshop, KEIR 2025, Lucca, Italy, April 10, 2025, Revised Selected Papers. Berlin, Heidelberg: Springer-Verlag, 2025, pp. 3–18.

[16] Q. Dong, L. Li, D. Dai, C. Zheng, J. Ma, R. Li, H. Xia, J. Xu, Z. Wu, B. Chang, X. Sun, L. Li, and Z. Sui, “A survey on in-context learning,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, Eds. Miami, Florida, USA: Association for Computational Linguistics, Nov. 2024, pp. 1107–1128.

[17] S. Wanna, N. Solovyev, R. Barron, M. E. Eren, M. Bhattarai, K. Ø. Rasmussen, and B. S. Alexandrov, “Topictag: Automatic annotation of NMF topic models using chain of thought and prompt tuning with LLMs,” in Proceedings of the ACM Symposium on Document Engineering 2024, ser. DocEng ’24. Association for Computing Machinery, 2024.

[18] T. Khandelwal, “Using llm-based approaches to enhance and automate topic labeling,” arXiv preprint arXiv:2502.18469, 2025.

[19] R. Spain, W. Min, V. Kumaran, J. Pande, J. Saville, and J. Lester, “Applying large language models to enhance dialogue and communication analysis for adaptive team training,” International Journal of Artificial Intelligence in Education, vol. 35, pp. 2534–2568, 2025.

[20] M. Jia and J. Diaz-Rodriguez, “Unsupervised text segmentation via kernel change-point detection on sentence embeddings,” arXiv preprint arXiv:2601.18788, 2026.

[21] D. A. P. Grimm, J. C. Gorman, N. J. Cooke, M. Demir, and N. J. McNeese, “Dynamical measurement of team resilience,” Journal of Cognitive Engineering and Decision Making, vol. 17, no. 4, pp. 351–382, 2023.

[22] C. Peifer, A. Pollak, O. Flak, A. Pyszka, M. A. Nisar, M. T. Irshad, M. Grzegorzek, B. Kordyaka, and B. Kozusznik, “The symphony of team flow in˙ virtual teams: Using artificial intelligence for its recognition and promotion,” Frontiers in Psychology, vol. 12, p. 697093, 2021.

[23] M. A. Hearst, “Texttiling: Segmenting text into multi-paragraph subtopic passages,” Computational Linguistics, vol. 23, no. 1, pp. 33–64, 1997.

[24] K.-W. Chang, M.-H. Hsu, S.-W. Li, and H.-y. Lee, “Exploring in-context learning of textless speech language model for speech classification tasks,” in Proceedings of Interspeech 2024, 2024, pp. 4713–4717.

[25] A. Raut, P. Paromita, S. Begerowski, S. Bell, and T. Chaspari, “Assessing the feasibility of Large Language Models for detecting micro-behaviors in team interactions during space missions,” in Interspeech, 2025, pp. 5453–5457.

[26] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proceedings of the 40th International Conference on Machine Learning, vol. 202, 2023, pp. 28 492–28 518.