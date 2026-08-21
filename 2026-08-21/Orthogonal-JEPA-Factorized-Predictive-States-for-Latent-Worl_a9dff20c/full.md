# Orthogonal JEPA: Factorized Predictive States for Latent World Models

Taoyong Cui Pheng-Ann Heng Wanli Ouyang

The Chinese University of Hong Kong (CUHK)

## Abstract

World models construct latent states that support prediction, planning, and reasoning about an underlying system. Joint-embedding predictive architectures (JEPAs) ofer a direct way to learn such states by predicting targets in representation space instead of reconstructing every detail of the observation. Standard JEPAs, however, organize all predictable content through one target embedding and one prediction pathway. In complex systems, this monolithic state can allocate redundant capacity to dominant signals while providing weak or conflicting gradients to less dominant predictive structure.

We introduce Orthogonal JEPA, a latent world-modeling framework based on orthogonal predictive factorization. Learned basis matrices analyze each target state into multiple components, and a dedicated prediction branch estimates each component from a shared context representation. Predictive regression preserves the factor magnitudes required for state synthesis, an orthogonality objective discourages repeated directions, factor-activity regularization maintains variation in projected targets, and online variance regularization discourages coordinate-wise encoder collapse. Predicted components are synthesized into a complete latent state that can be used by a readout, decoder, planner, or autoregressive rollout. The same predictive-state mechanism applies when the target is temporally future, spatially hidden, or another partial observation of the same system. Experiments on controlled vision, single-cell transcriptomics, longitudinal health records, continuous control, and molecular dynamics evaluate representation quality, forecasting, planning, and long-horizon stability.

## 1 Introduction

World models learn internal states that allow an agent or scientific model to anticipate unobserved consequences of an observed situation. Early latent world models compressed observations and learned recurrent dynamics for control [1]; more recent systems have demonstrated planning across diverse control domains using learned latent dynamics [2]. Across these formulations, the central object is not the raw observation itself but a state that retains information needed to predict how the modeled system can change.

This predictive-state perspective extends beyond action-conditioned control. A latent state may summarize a physical configuration, a molecular trajectory, a patient history, a partially observed scene, or a cellular profile. Its target may be a future state, a hidden spatial region, another view, or a more complete observation of the same system. We use latent world model in this operational sense: a model that constructs a latent state from context and uses it to predict another state of the same underlying system. The definition does not require pixel generation or an explicit simulator, but it does require a meaningful context–target relation.

JEPAs provide a natural mechanism for learning such states. A context encoder summarizes what is observed, a target encoder defines the state to be predicted, and a predictor maps between the two in representation space. Latent prediction can focus on shared, predictable structure without reconstructing noise, local texture, or other raw-space details that may be irrelevant to downstream use. Reconstruction may overemphasize high-variance directions that are weakly aligned with perceptual usefulness [3], whereas the learned target encoder determines the abstraction level of a JEPA target. This view also connects latent prediction to energy-based learning [4], learned similarity metrics [5, 6], masked image prediction [7], and video representation learning [8].

The usual JEPA formulation nevertheless represents the requested state through one target embedding and one prediction pathway. Complex systems commonly combine local and global structure, multiple interacting entities, and changes at diferent scales. When these signals share a monolithic target, easily predicted or high-variance structure may dominate optimization, multiple latent directions may serve similar roles, and weaker predictive structure may receive conflicting gradients. This motivates treating latent-state design as a capacity-allocation problem.

## Can a JEPA organize a predictable world state through multiple structured prediction factors rather than one monolithic prediction path?

We answer this question with Orthogonal JEPA. A collection of learned basis matrices analyzes each target state into multiple components. Each component is assigned a prediction branch, while within-factor and cross-factor orthogonality objectives allocate diferent target-space directions to diferent branches. Factor-activity regularization maintains variation in projected targets, and an online variance term discourages coordinate-wise collapse of the trainable encoder. The predicted components can be synthesized into a complete latent state for downstream readout, planning, or autoregressive rollout.

The method separates the predictive-state mechanism from the geometry of the observation. Each application chooses an adapter, a context–target construction, structural descriptors, and an encoder architecture. The orthogonal factorization, branch prediction, state synthesis, and core regularizers remain shared. Temporal conditioning produces conventional future-state world models; masking and partial observation apply the same mechanism to spatial and biological systems.

Our contributions are:

• We formulate Orthogonal JEPA as a common predictive-state mechanism for latent world models with domain-specific context–target interfaces.

• We introduce orthogonal predictive factorization, which analyzes a latent target state into multiple learned components and predicts each with a dedicated branch.

• We combine factorized prediction with state synthesis, orthogonality, factor-activity regularization, and online variance regularization.

• We evaluate the same core mechanism across visual, cellular, clinical, control, and molecular systems.

Table 1: Context–target definitions and uses of the predicted latent state.
<table><tr><td>System</td><td>Observed context</td><td>Latent target</td><td>State use</td></tr><tr><td>Vision</td><td>Visible image patches</td><td>Hidden patch states</td><td>Structured readout</td></tr><tr><td>Single cell</td><td>Masked expression profile</td><td>Complete cell state</td><td>Clustering, perturbation prediction</td></tr><tr><td>Health</td><td>Patient history</td><td>Future health state</td><td>Clinical-event forecasting</td></tr><tr><td>Control</td><td>Current state and candidate actions</td><td>Future control state</td><td>Planning</td></tr><tr><td>Molecules</td><td>Current atomic configuration</td><td>Future molecular state</td><td>Autoregressive rollout</td></tr></table>

## 2 Orthogonal JEPA

## 2.1 Predictive-State Interface

Let δ index a domain and let $x \sim \mathcal { D } _ { \delta }$ be a raw observation. A domain adapter $\mathcal { A } _ { \delta }$ maps x to content tokens $H = \{ h _ { i } \} _ { i \in \Omega }$ and optional structural descriptors ${ \cal S } = \{ s _ { i } \} _ { i \in \Omega }$ . A descriptor may encode a patch coordinate, time stamp, entity identity, or another target specification. A view sampler $\nu _ { \delta }$ selects context indices $C \subset \Omega$ and target indices $T \subset \Omega$

$$
x \stackrel { \mathcal { A } _ { \delta } } { \longrightarrow } ( H , S ) \stackrel { \mathcal { V } _ { \delta } } { \longrightarrow } ( H _ { C } , S _ { C } , T , S _ { T } ) .\tag{1}
$$

An online encoder produces a context representation $z _ { c } = f _ { \theta } ( H _ { C } , S _ { C } )$ . A target encoder produces $z _ { t } = f _ { \bar { \theta } } ( H , S ) _ { t } \in \mathbb { R } ^ { d }$ for every $t \in T$ . The target parameters are updated by exponential moving average,

$$
\bar { \theta }  m \bar { \theta } + ( 1 - m ) \theta , \qquad 0 \leq m < 1 ,\tag{2}
$$

and receive no gradients. The formulation includes token-valued targets and the single-vector case $| T | = 1$

Table 1 shows how the same predictive-state interface is instantiated in the evaluated systems.

## 2.2 Orthogonal Predictive Factorization

We introduce K learned basis matrices $B _ { k } \in \mathbb { R } ^ { d \times r }$ and choose r such that $K r = d .$ . The stop-gradient target state is analyzed as

$$
\widetilde { \boldsymbol { z } } _ { t } = \mathrm { s g } ( \boldsymbol { z } _ { t } ) , \qquad \boldsymbol { z } _ { t } ^ { ( k ) } = \boldsymbol { B } _ { k } ^ { \top } \widetilde { \boldsymbol { z } } _ { t } , \qquad k = 1 , \ldots , K .\tag{3}
$$

Only the target-encoder output is stopped. The basis matrices remain trainable and receive gradients from the predictive, orthogonality, and factor-activity terms.

Each factor has a predictor $q _ { k }$ that maps the shared context representation and, when required, target descriptor $s _ { t }$ into the corresponding r-dimensional coordinate space:

$$
\widehat { z } _ { t } ^ { ( k ) } = q _ { k } ( z _ { c } , s _ { t } ) .\tag{4}
$$

The factor predictions are concatenated as

$$
\widehat { \boldsymbol { u } } _ { t } = [ ( \widehat { \boldsymbol { z } } _ { t } ^ { ( 1 ) } ) ^ { \top } , \ldots , ( \widehat { \boldsymbol { z } } _ { t } ^ { ( K ) } ) ^ { \top } ] ^ { \top } \in \mathbb { R } ^ { d } .
$$

Let ${ \boldsymbol { B } } = [ B _ { 1 } , \dots , B _ { K } ] \in \mathbb { R } ^ { d \times d }$ . The complete predicted state is synthesized through the Moore– Penrose pseudoinverse of the analysis map:

$$
\widehat { z } _ { t } = ( B ^ { \top } ) ^ { \dagger } \widehat { u } _ { t } .\tag{5}
$$

When B is orthogonal, $( B ^ { \top } ) ^ { \dagger } = B$ , and $\begin{array} { r } { \widehat { z } _ { t } = \sum _ { k } B _ { k } \widehat { z } _ { t } ^ { ( k ) } } \end{array}$

To preserve both direction and magnitude, we directly regress each factor:

$$
\mathcal { L } _ { \mathrm { p r e d } } = \frac { 1 } { K | T | r } \sum _ { t \in T } \sum _ { k = 1 } ^ { K } \left\| \widehat { z } _ { t } ^ { ( k ) } - z _ { t } ^ { ( k ) } \right\| _ { 2 } ^ { 2 } .\tag{6}
$$

The basis columns are encouraged to be orthonormal within a factor and orthogonal across factors:

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { o r t h } } = \sum _ { k = 1 } ^ { K } \left\| \boldsymbol { B } _ { k } ^ { \top } \boldsymbol { B } _ { k } - I _ { r } \right\| _ { F } ^ { 2 } } \\ { \displaystyle \qquad + \sum _ { 1 \leq i < j \leq K } \left\| \boldsymbol { B } _ { i } ^ { \top } \boldsymbol { B } _ { j } \right\| _ { F } ^ { 2 } . } \end{array}\tag{7}
$$

This objective is related to redundancy-reduction methods [9, 10], but acts on learned predictive target coordinates rather than only on embedding statistics.

Proposition 1 (Exact orthogonal decomposition). If $B _ { i } ^ { \top } B _ { j } = 0 f o r i \neq j , B _ { k } ^ { \top } B _ { k } = I _ { r }$ , and $K r = d $ then B is orthogonal and, for every $z \in \mathbb { R } ^ { d }$ 2

$$
\| B ^ { \top } z \| _ { 2 } ^ { 2 } = \sum _ { k = 1 } ^ { K } \| B _ { k } ^ { \top } z \| _ { 2 } ^ { 2 } = \| z \| _ { 2 } ^ { 2 } , \qquad z = \sum _ { k = 1 } ^ { K } B _ { k } B _ { k } ^ { \top } z .\tag{8}
$$

Proof. The assumptions give $B ^ { \top } B = B B ^ { \top } = I _ { d }$ . The norm identity follows by blockwise expansion of $B ^ { \top } z$ , and the second identity follows from $z = B B ^ { \top } z$ . □

The proposition describes the exact-orthogonality limit of the training objective. Geometric separation alone does not imply statistical independence or semantic disentanglement; these properties require empirical evaluation.

## 2.3 Factor Activity and Encoder Variance

Orthogonality does not ensure that every projected target varies across samples. For a mini-batch and all sampled targets, let $\sigma _ { k , j } ^ { \mathrm { f a c } }$ be the empirical standard deviation of coordinate $j$ in $z _ { t } ^ { ( k ) }$ . Let $\sigma _ { j } ^ { \mathrm { e n c } }$ be the empirical standard deviation of coordinate j in the online context representation $z _ { c } .$ . For token-valued contexts, this statistic is computed over all valid context tokens. We define

$$
\mathcal { L } _ { \mathrm { f a c } } = \frac { 1 } { K r } \sum _ { k = 1 } ^ { K } \sum _ { j = 1 } ^ { r } \operatorname* { m a x } ( 0 , \gamma _ { \mathrm { f a c } } - \sigma _ { k , j } ^ { \mathrm { f a c } } ) ,
$$

$$
\mathcal { L } _ { \mathrm { e n c } } = \frac { 1 } { d } \sum _ { j = 1 } ^ { d } \operatorname* { m a x } ( 0 , \gamma _ { \mathrm { e n c } } - \sigma _ { j } ^ { \mathrm { e n c } } ) ,\tag{9}
$$

where standard deviations are computed as $\sqrt { \mathrm { V a r } + \epsilon }$ . The first term discourages inactive projected coordinates; the second supplies a direct coordinate-wise variance gradient to the online encoder.

The core objective is

$$
{ \mathcal { L } } _ { \mathrm { O J E P A } } = { \mathcal { L } } _ { \mathrm { p r e d } } + \lambda _ { \mathrm { o r t h } } { \mathcal { L } } _ { \mathrm { o r t h } } + \lambda _ { \mathrm { f a c } } { \mathcal { L } } _ { \mathrm { f a c } } + \lambda _ { \mathrm { e n c } } { \mathcal { L } } _ { \mathrm { e n c } } .\tag{10}
$$

When an application retains a domain-standard auxiliary pretraining loss, it is added as $\mathcal { L } _ { \mathrm { t o t a l } } =$ $\mathcal { L } _ { \mathrm { O J E P A } } + \lambda _ { \mathrm { a u x } } \mathcal { L } _ { \mathrm { a u x } }$ and held fixed between direct comparisons.

## 2.4 Downstream and Predictive Use

For ordinary downstream evaluation, the online encoder supplies reusable features and a task-specific readout $R _ { \delta , \tau }$

$$
h _ { \delta , \tau } ( x ) = R _ { \delta , \tau } \Big ( f _ { \theta } ^ { \delta } ( \mathcal { A } _ { \delta } ( x ) ) \Big ) .\tag{11}
$$

The EMA encoder, factor bases, and prediction heads are not required for this use. Predictive applications instead retain Equation (5); the synthesized state is consumed by a decoder, planner, or subsequent rollout step.

## 3 Experiments

We evaluate Orthogonal JEPA in five systems spanning hidden-state completion, temporal forecasting, planning, and autoregressive rollout. In controlled comparisons, the adapter, encoder family, context–target sampler, data split, optimization budget, and downstream readout are held fixed; the primary change is the monolithic versus factorized JEPA target. Domain-standard auxiliary terms, when used, are held fixed.

## 3.1 Predictive-State Quality and Readout

## 3.1.1 Controlled Visual Binding

This task asks whether a frozen state retains both where a controlled visual change occurs and which operation occurred, including support–operation combinations withheld from readout training. DINOv3 [13] and SigLIP2 [14] provide the visual backbones. The frozen baseline uses the original checkpoint. Standard JEPA and Orthogonal JEPA start from identical copies and use blockmasked patch prediction with two-dimensional position descriptors. They share data, masks, optimization settings, and readout architecture.

For each source image and controlled intervention, patch-token diferences form an innovation field. Reconstructed MuJoCo scenes [12] are evaluated with image-disjoint splits, leave-one-cell-out readout, and injective Hungarian alignment [11]. Known-grid and learned-grid conditions difer only in whether the support–operation taxonomy is supplied or estimated from observed cell labels. We report injective held-out-cell accuracy (INJ), collapse rate at threshold 0.5 (Coll.), and grid recovery (Rec.), averaged over ten seeds.

Table 2: Controlled visual binding. Standard JEPA and Orthogonal JEPA use the same learnedgrid readout.
<table><tr><td>Backbone</td><td>Encoder pretraining</td><td>Grid supervision</td><td>INJ↑</td><td>Coll. ↓</td><td>Rec. ↑</td></tr><tr><td>DINOv3</td><td>Frozen checkpoint</td><td>Known grid</td><td>.452</td><td>.667</td><td></td></tr><tr><td>DINOv3</td><td>Frozen checkpoint</td><td>Learned grid</td><td>.569</td><td>.433</td><td>.643</td></tr><tr><td>DINOv3</td><td>Standard JEPA</td><td>Learned grid</td><td>.572</td><td>.426</td><td>.645</td></tr><tr><td>DINOv3</td><td>ORTHOGONAL JEPA</td><td>Learned grid</td><td>.581</td><td>.417</td><td>.659</td></tr><tr><td>SigLIP2</td><td>Frozen checkpoint</td><td>Known grid</td><td>.476</td><td>.656</td><td></td></tr><tr><td>SigLIP2</td><td>Frozen checkpoint</td><td>Learned grid</td><td>.484</td><td>.511</td><td>.676</td></tr><tr><td>SigLIP2</td><td>Standard JEPA</td><td>Learned grid</td><td>.483</td><td>.514</td><td>.679</td></tr><tr><td>SigLIP2</td><td>ORTHOGONAL JEPA</td><td>Learned grid</td><td>.490</td><td>.503</td><td>.688</td></tr></table>

With the learned-grid readout fixed, Orthogonal JEPA increases INJ and recovery while reducing the reported collapse metric for both visual backbones.

## 3.1.2 Single-Cell Transcriptomics

Each cell is tokenized by gene identity and discretized expression value using an scGPT backbone [16]. Following Cell-JEPA [15], a student receives masked expression values and predicts the cell-level state produced by an EMA teacher from the unmasked cell. Both JEPA conditions retain the same masked-gene reconstruction term; standard Cell-JEPA uses a monolithic latent target, while Orthogonal JEPA factorizes that target.

Models are pretrained on approximately 800,000 human kidney cells and evaluated on PBMC-10K clustering under finetuned and zero-shot settings. Perturbation response is evaluated on Adamson [17] and Norman [18]. AvgBIO measures clustering quality, and Pearson correlation measures absolute post-perturbation expression prediction.

Table 3: Single-cell clustering and perturbation-response evaluation.
<table><tr><td>Model</td><td>JEPA</td><td>Orthogonal factors</td><td>AvgBIO ↑</td><td>PBMC finetuned PBMC zero-shot AvgBIO ↑</td><td>Norman Pearson ↑</td><td>Adamson Pearson ↑</td></tr><tr><td>scGPT</td><td>No</td><td>No</td><td>0.7531</td><td>0.5288</td><td>0.631</td><td>0.905</td></tr><tr><td>Cell-JEPA</td><td>Yes</td><td>No</td><td>0.7830</td><td>0.7194</td><td>0.787</td><td>0.937</td></tr><tr><td>ORTHOGONAL JEPA</td><td>Yes</td><td>Yes</td><td>0.8001</td><td>0.7452</td><td>0.798</td><td>0.942</td></tr></table>

## 3.1.3 Latent Health-State Forecasting

Each patient state combines time-ordered clinical events, demographic context, and molecular or biochemical variables. A context encoder represents patient history, and an EMA target encoder represents a future health state. A shared decoder maps the predicted state to risks for more than 1,000 future clinical events. Mean area under the precision–recall curve (PRAUC) across the event vocabulary is the primary metric.

Table 4: Prediction across more than 1,000 future clinical events.
<table><tr><td>Model</td><td>Model family</td><td>JEPA</td><td>Orthogonal factors</td><td>Mean PRAUC ↑</td></tr><tr><td>Random Forest</td><td>Classical ML</td><td>No</td><td>No</td><td>0.602</td></tr><tr><td>XGBoost</td><td>Classical ML</td><td>No</td><td>No</td><td>0.667</td></tr><tr><td>Qwen2.5-0.5B</td><td>General-purpose language model</td><td>No</td><td>No</td><td>0.648</td></tr><tr><td>Qwen3-0.8B</td><td>General-purpose language model</td><td>No</td><td>No</td><td>0.651</td></tr><tr><td>Prophet</td><td>Autoregressive disease trajectory</td><td>No</td><td>No</td><td>0.680</td></tr><tr><td>Delphi</td><td>Autoregressive disease trajectory</td><td>No</td><td>No</td><td>0.689</td></tr><tr><td>Cross-attention fusion</td><td>Multimodal fusion</td><td>No</td><td>No</td><td>0.702</td></tr><tr><td>Standard JEPA</td><td>Latent health-state JEPA</td><td>Yes</td><td>No</td><td>0.711</td></tr><tr><td>ORTHOGONAL JEPA</td><td>Orthogonal health-state JEPA</td><td>Yes</td><td>Yes</td><td>0.718</td></tr></table>

## 3.2 Latent Dynamics and Rollout Stability

## 3.2.1 Continuous Control

We evaluate state-based control on Walker2d-v5, HalfCheetah-v5, and InvertedPendulum-v5. Both JEPA conditions are trained ofline on 500 random-action trajectories using the same two-layer MLP backbone, latent width, and optimization budget. Planning uses the cross-entropy method with 200 candidates, 20 elites, five iterations, and an eight-step horizon. Results are mean ± standard deviation over three seeds.

Table 5: CEM planning return on state-based MuJoCo control tasks.
<table><tr><td>Model</td><td></td><td></td><td>Walker2d ↑ HalfCheetah ↑ InvertedPendulum ↑</td></tr><tr><td>Standard JEPA</td><td> $4 . 9 \pm 1 2 . 6$ </td><td> $- 1 1 . 2 \pm 0 . 8$ </td><td> $1 8 . 1 \pm 2 . 3$ </td></tr><tr><td>ORTHOGONAL JEPA</td><td> $4 5 . 1 \pm 1 1 . 2$ </td><td> $- 8 . 5 \pm 0 . 6$ </td><td> $3 0 . 6 \pm 3 . 8$ </td></tr></table>

## 3.2.2 Force-Free Molecular Dynamics

Each molecular state contains atomic positions, velocities, species, and the simulation cell. A TrajCast-style equivariant forecaster [19] is evaluated on liquid water, crystalline α-quartz, gas-phase paracetamol, and benzene. TrajCast-JEPA denotes the monolithic JEPA-pretrained condition. Orthogonal JEPA retains the same data, O(3)-equivariant backbone, prediction intervals, supervised adaptation, and rollout protocol. Factorization operates only among compatible irreduciblerepresentation channels.

We report one-step displacement MAE and median final-position RMSD after 100 free autoregressive steps across five trained seeds.

Table 6: Force-free molecular forecasting. MAE is one-step displacement error; RMSD is finalposition error after 100 autoregressive steps, both in <sup>˚</sup>A.
<table><tr><td rowspan="3">Model</td><td colspan="2">Water</td><td colspan="2">Quartz</td><td colspan="2">Paracetamol</td><td colspan="2">Benzene</td></tr><tr><td>MAE↓</td><td>RMSD ↓</td><td>MAE↓</td><td>RMSD↓</td><td>MAE↓</td><td>RMSD ↓</td><td>MAE↓</td><td>RMSD ↓</td></tr><tr><td>Scratch</td><td>0.00387</td><td>3.331</td><td>0.01080</td><td>2.089</td><td>0.00901</td><td>3.155</td><td> $2 . 5 9 \times 1 0 ^ { - 5 }$ </td><td>0.0958</td></tr><tr><td>TrajCast-JEPA</td><td>0.00452</td><td>2.536</td><td>0.01043</td><td>1.912</td><td>0.00777</td><td>1.868</td><td> $2 . 1 0 \times 1 0 ^ { - 5 }$ </td><td>0.0701</td></tr><tr><td>ORTHOGONAL JEPA</td><td>0.00376</td><td>2.459</td><td>0.01011</td><td>1.877</td><td>0.00765</td><td>1.846</td><td> $2 . 0 5 \times 1 0 ^ { - 5 }$ </td><td>0.0699</td></tr></table>

## 4 Discussion and Limitations

The experiments test whether an orthogonality-regularized multi-branch target improves predictivestate learning across several observation geometries and uses. Orthogonality is geometric: it separates basis directions but does not by itself imply statistical independence, causal modularity, or semantic disentanglement. Factor interpretation therefore requires additional probes or ground-truth factor benchmarks.

The variance regularizers constrain marginal coordinate variation but do not guarantee full-rank covariance. Future work may combine the current objective with covariance or spectral regularization. State synthesis also depends on the conditioning of B; monitoring singular values is important when synthesized states are repeatedly fed into a rollout. Finally, the present predictor is deterministic and does not explicitly represent multimodal futures.

The current experiments span partial observation, forecasting, planning, and rollout, but do not cover every world-model regime. Pixel-based closed-loop control, stochastic futures, continuous physical fields, and tasks with known causal factors would provide complementary tests.

## 5 Conclusion

Orthogonal JEPA replaces a monolithic JEPA target with multiple learned predictive components. A common context representation feeds factor-specific predictors; orthogonality and activity regularizers organize the target coordinates; and the predicted components can be synthesized into a complete latent state. The same mechanism supports frozen readout, clinical forecasting, control planning, and molecular rollout. Across the reported comparisons, the factorized target improves the corresponding task metrics, motivating orthogonal predictive factorization as a reusable design for latent world models.

## References

[1] D. Ha and J. Schmidhuber. Recurrent world models facilitate policy evolution. In Advances in Neural Information Processing Systems, volume 31, 2018.

[2] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap. Mastering diverse control tasks through world models. Nature, 640:647–653, 2025.

[3] R. Balestriero and Y. LeCun. How learning by reconstruction produces uninformative features for perception. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 2566–2585, 2024.

[4] Y. LeCun, S. Chopra, R. Hadsell, M. Ranzato, and F.-J. Huang. A tutorial on energy-based learning. In Predicting Structured Data. MIT Press, 2006.

[5] S. Chopra, R. Hadsell, and Y. LeCun. Learning a similarity metric discriminatively, with application to face verification. In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, volume 1, pages 539–546, 2005.

[6] R. Hadsell, S. Chopra, and Y. LeCun. Dimensionality reduction by learning an invariant mapping. In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, volume 2, pages 1735–1742, 2006.

[7] M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[8] A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, and N. Ballas. Revisiting feature prediction for learning visual representations from video. Transactions on Machine Learning Research, 2024.

[9] J. Zbontar, L. Jing, I. Misra, Y. LeCun, and S. Denys. Barlow Twins: Self-supervised learning via redundancy reduction. In Proceedings of the 38th International Conference on Machine Learning, 2021.

[10] A. Bardes, J. Ponce, and Y. LeCun. VICReg: Variance-invariance-covariance regularization for self-supervised learning. In International Conference on Learning Representations, 2022.

[11] H. W. Kuhn. The Hungarian method for the assignment problem. Naval Research Logistics Quarterly, 2(1–2):83–97, 1955.

[12] E. Todorov, T. Erez, and Y. Tassa. MuJoCo: A physics engine for model-based control. In IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012.

[13] O. Sim´eoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, et al. DINOv3. arXiv preprint arXiv:2508.10104, 2025.

[14] M. Tschannen, A. Gritsenko, X. Wang, M. F. Naeem, I. Alabdulmohsin, N. Parthasarathy, T. Evans, et al. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

[15] A. ElSheikh, R.-X. Wang, W. Wu, Y. Wen, P. Dibaeinia, J. Y. Zhang, J. Y.-C. Hu, et al. Cell-JEPA: Latent representation learning for single-cell transcriptomics. arXiv preprint arXiv:2602.02093, 2026.

[16] H. Cui, C. Wang, H. Maan, K. Pang, F. Luo, N. Duan, and B. Wang. scGPT: Toward building a foundation model for single-cell multi-omics using generative AI. Nature Methods, 21(8):1470–1480, 2024.

[17] B. Adamson, T. M. Norman, M. Jost, M. Y. Cho, J. K. Nu˜nez, Y. Chen, J. E. Villalta, et al. A multiplexed single-cell CRISPR screening platform enables systematic dissection of the unfolded protein response. Cell, 167(7):1867–1882.e21, 2016.

[18] T. M. Norman, M. A. Horlbeck, J. M. Replogle, A. X. Ge, A. Xu, M. Jost, L. A. Gilbert, and J. S. Weissman. Exploring genetic interaction manifolds constructed from rich single-cell phenotypes. Science, 365(6455):786–793, 2019.

[19] F. L. Thiemann, T. Resch¨utzegger, M. Esposito, T. Taddese, J. D. Olarte-Plata, and F. Martelli. Force-free molecular dynamics through autoregressive equivariant networks. Nature Machine Intelligence, 8:764–776, 2026.