# Incremental Evaluation and Training in Relational Deep Learning

Jakub Peleška Czech Technical University in Prague jakub.peleska@cvut.cz

Gustav Šír Czech Technical University in Prague gustav.sir@cvut.cz

## Abstract

Relational Deep Learning (RDL) models multi-tabular databases as temporal heterogeneous graphs to enable end-to-end representation learning. However, prevailing RDL evaluation practices rely on static, single-episode dataset snapshots, overlooking the continuous, time-evolving nature of real-world databases. Consequently, current RDL benchmarks fail to capture how model performance changes as new data accumulates over time. To address this limitation, we introduce an incremental, multi-episode evaluation and training paradigm to assess and improve the temporal robustness and adaptability of state-of-the-art RDL models. Using established large-scale datasets, we examine data evolution and model training dynamics, demonstrating that temporal concept drifts occur in the majority of predictive tasks. We present multiple incremental training regimes for fine-tuning the models and demonstrate that transfer learning is both feasible and highly effective in the RDL setting. Alongside a new temporal evaluation metric that prioritizes near-future accuracy, we show that our incrementally fine-tuned models consistently outperform the standard, expensive, from-scratch trained baselines.

## 1 Introduction

Relational databases (RDBs) are the foundational infrastructure for data storage across a vast majority of modern industries [6, 1, 38]. Despite their ubiquity and immense predictive value, applying traditional machine learning (ML) to RDBs presents significant challenges. The inherently heterogeneous, dynamically structured nature of relational data—spanning multiple interconnected tables with varying schemas—resists the fixed-size feature vector representations required by classical ML algorithms. Consequently, practitioners are often forced into laborious manual feature engineering or lossy table joins that discard important structural semantics [11]. To address these bottlenecks, Relational Deep Learning (RDL) [41, 11] has recently emerged as a transformative ML paradigm. By interpreting an entire relational database as a heterogeneous graph, RDL leverages Graph Neural Networks (GNNs) [35, 2] alongside deep tabular representation learning [13, 30] to enable end-to-end predictive modeling directly on the native structures of the RDBs (Figure 1).

While RDL successfully eliminates the need for manual feature flattening, bringing ML significantly closer to the raw, heterogeneous reality of relational data, it currently neglects a foundational dimension of real-world RDBs: their continuous temporal evolution. Indeed, databases are not static repositories; they dynamically expand and shift over time, causing models trained on historical snapshots to become incrementally obsolete. Despite this dynamic reality, current RDL evaluation practices remain focused on static models. Given the structural complexity of RDBs, which prevents classic i.i.d. sample-based predictions, predictive tasks in RDL are defined by augmenting the schema with a dedicated “task table” that specifies target entities and labels. To avoid temporal leakage, existing frameworks [33, 29] correctly use timestamps to perform chronological train–test splits [11, 32]. However, this snapshot-based evaluation assesses models in a single, isolated episode, fundamentally failing to capture how model performance degrades over time, e.g., due to temporal concept drift. Characterizing this temporal stability and evaluating how models adapt to new increments of relational data then remains an unaddressed prerequisite for advancing toward robust, real-world RDL foundation models [32, 36].

![](images/7eceb701dfbaf6201b2e34ed1c432c883f0e96190f4d048f948a84e10a993705.jpg)  
Figure 1: Incremental RDL workflow with (a) append-only data updates, (b) induced relational graph increments $\Delta G _ { i }$ on top of historical context $G _ { t - 1 }$ , and (c) episode-wise fine-tuning from $\theta _ { i - 1 } \tan \theta _ { i }$

Contributions. To address the gap between static benchmarking and the dynamic reality of relational data, we introduce an incremental, multi-episodic evaluation and training paradigm, visualized in Figure 1. Tailored for robust assessment of RDL models, our approach simulates the real-world life-cycle of database growth by transitioning RDL benchmarking from static snapshots to continuous, horizon-shifting evaluation. Leveraging state-of-the-art models and large-scale datasets [33, 29, 15], we empirically demonstrate that temporal concept drift is a prevalent factor in standard RDL tasks. To combat model degradation, we define and benchmark multiple incremental fine-tuning strategies, showing that knowledge transfer is highly effective and computationally superior to from-scratch retraining in the continuous RDL setting. Finally, we propose a suitable exponential decay-based evaluation that prioritizes near-future predictive accuracy, closely aligning benchmark evaluation with practical operational priorities.

## 2 Related Work

RDL Benchmarks and Frameworks Standardized ecosystems have accelerated RDL research. RelBench [33, 15] formalized predictive tasks with chronological splits to prevent temporal leakage [22]. Concurrently, ReDeLEx [29] expanded the evaluation landscape across a large set of diverse databases to analyze structural impacts on model performance. While these frameworks provide essential building blocks for static testing, they lack the multi-episodic evaluation necessary to measure continuous model degradation and real-world lifecycle dynamics.

Architectural Advances and Foundation Models To capture the extreme heterogeneity of RDBs, many recent methods integrate tabular representation learning [25, 30], LLMs [39], and graph transformers [9, 24]. This progress naturally steers the field toward RDL foundation models [37, 10, 12, 19, 36, 32], supported by task-agnostic pretraining [31] and synthetic data generation [20, 23]. While these foundational architectures are largely designed for static deployment, the overarching shift towards foundation models and transfer learning strongly overlaps with our proposed temporal fine-tuning, forming a prerequisite for maintaining their robustness over time.

Temporal Graph Representation Learning Relational data evolution conceptually intersects with Temporal Graph Representation Learning [21]; nevertheless, traditional temporal GNNs (e.g., TGN [34], TGAT [8]) model high-frequency dynamic systems to predict immediate interactions. In contrast, our incremental RDL paradigm does not treat databases as stateful time-series systems. Instead, we focus on macro-level volumetric growth and long-term temporal concept drift to assess model obsolescence over extended horizons.

Continual Learning and Concept Drift By addressing model obsolescence, our approach conceptually bridges RDL with Continual Learning (CL) [28] and concept drift adaptation [26]. Standard CL and Continual Graph Learning [42] mitigate catastrophic forgetting primarily in homogeneous structures. RDL, however, involves highly heterogeneous, multi-tabular data increments. Our work provides the first formal benchmark to evaluate CL strategies directly within the dynamically evolving environment of real-world databases.

## 3 Relational Deep Learning

RDL is a machine learning paradigm built on interpreting relational databases as heterogeneous graphs. This perspective enables the direct application of deep learning techniques—specifically GNNs alongside tabular representation learning—on the native structures of RDBs. While the underlying idea was explored in earlier work [7, 41], it was recently formalized under the RDL umbrella [11] to consolidate these efforts into a cohesive research field, detailed below.

## 3.1 Graph Representation

Formally, a relational database R is defined as a finite set of relations $R _ { 1 } , R _ { 2 } , \ldots , R _ { n }$ . Each relation R consists of a heading (signature) $R _ { / n }$ , formed by the set of its attributes, and a body, formed by the tuples of attribute values $t _ { i } = ( a _ { 1 } , a _ { 2 } , \ldots , a _ { n } )$ , commonly represented as a table $T _ { R }$ . Additionally, the relation $R$ can be defined extensionally by the unordered set of its tuples: $R = \{ t _ { 1 } , t _ { 2 } , \ldots , t _ { m } \}$ , corresponding to the rows (or entities) of the table $T _ { R }$

Practically, the attributes forming the rows of each table $T _ { R }$ are categorized into four functional groups: (i) primary key (PK), a minimal set of attributes $P K _ { R }$ that uniquely identifies each entity, such that for any two distinct tuples $t _ { a } , t _ { b } \in R , t _ { a } [ P K _ { R } ] \neq t _ { b } [ P K _ { R } ]$ ; (ii) foreign key (FK), a set of attributes in a relation $R _ { i }$ that references the primary key of another relation $R _ { i } ,$ establishing a structural dependency ensuring referential integrity $\forall t \in R _ { i } , t [ F K _ { R _ { i }  R _ { i } } ] \in \{ t ^ { \prime } [ { P } \bar { K } _ { R _ { i } } ] \ | \ t ^ { \prime } \in \bar { R _ { j } } \}$ (iii) feature attributes, containing characteristic information about the entity; and, optionally, (iv) a timestamp, indicating when the row was created or last modified.

The relational entity graph of a database R is a temporal heterogeneous graph defined as $G =$ $( \nu , \mathcal { E } , \phi , \psi , \tau )$ . The set of nodes V is defined as the union of all tuples from each relation: $\mathcal { V } = \{ v _ { i , j } \ \vert$ $R _ { i } \in \mathcal { R } , t _ { j } \in R _ { i } \}$ , inferring $v _ { i , j } \simeq t _ { j } \in R _ { i }$ . Consequently, the set of edges $\mathcal { E }$ is defined by the PK-FK relationships: $\mathcal { E } = \breve { \{ }  ( v _ { i , k } ^ { \prime \prime \prime } , v _ { j , l } \breve { ) } \mid t _ { k } \in R _ { i } , t _ { l } \ \mathring { \in } \ R _ { j } , \breve { t } _ { k } [ F K _ { R _ { j } } ] = t _ { l } [ P K _ { R _ { j } } ] \vee t _ { l } [ F K _ { R _ { i } } ] =$ $t _ { k } [ P K _ { R _ { i } } ] \big \}$ (Figure 1b).

To capture the heterogeneity of the database, the function $\phi : \mathcal { V } \to \mathcal { T } ^ { v }$ serves as a node mapping. The set of node types $\mathcal { T } ^ { v }$ corresponds directly to the set of relations within R, meaning ϕ maps a node $v _ { i , j }$ strictly to its original source relation $R _ { i }$ . Similarly, $\psi : { \mathcal { E } } \to { \mathcal { T } } ^ { e }$ is an edge mapping function, where the set of edge types $\mathcal { T } ^ { e }$ corresponds to the specific pairs of PK and FK attributes (or their reversed counterparts) that generated the edge. Finally, the time mapping function $\tau : \mathcal { V } \to \mathrm { T }$ assigns a timestamp to each node such that $\tau ( v ) = \bar { \mathrm { T } } _ { v }$ . If a temporal attribute is not available for a given row, a zero timestamp is assigned by default.

## 3.2 Neural Methods

Building upon the graph representation G, RDL models typically follow a four-stage pipeline: (i) a table-level attribute encoder initializes node embedding matrices $h _ { v } ^ { ( 0 ) } \in \mathbb { R } ^ { d ^ { ( 0 ) } }$ <sup>×n</sup> by encoding each feature attribute $A _ { 1 } , \ldots , A _ { n }$ of the node’s source relation $\phi ( v )$ based on its semantic data type; (ii) a table-level tabular model employs tabular architectures [4, 18] to refine these embeddings into $h _ { v } ^ { ( l ) } \in \mathbb { R } ^ { d ^ { ( l ) } \times n }$ , optionally compressing the matrix into a single vector $h _ { v } ^ { ( l ) } \in \mathbb { R } ^ { d _ { \phi ( v ) } ^ { ( l ) } }$ ; (iii) a graph neural model applies message-passing based on the defined edge types $\mathcal { T } ^ { e }$ , using standard GNNs [16, 35, 2] for vectors or custom schemes [30, 5] for attribute-matrices; and (iv) a task-specific model head maps the final node embeddings to predictions, typically via simple MLP layers. This RDL architectural blueprint is illustrated in Figure 1c.

## 3.3 Predictive Tasks

Predictive tasks in RDL are formalized by constructing a dedicated “task table $\because T _ { t a s k }$ that extends the existing relational database R [11, 22]. A task table $T _ { t a s k }$ consists of tuples defining the individual prediction instances, abstracted as $( v , y , t )$ . Here, $v \in \mathcal V$ is the target entity (node) identified via a foreign key reference $T _ { t a s k } [ F K ] , y \in \mathcal { Y }$ is the ground-truth target label to be predicted $( \mathbf { e } . \mathbf { g } . , y \in \mathbb { R }$ for regression), and $t \in \mathrm { T }$ is the anchor time, specifying the moment at which the prediction is made.

![](images/5a9454520221f4c4c70375ce33f06ec5734f9deee71a97c34e8fa1cace6fd871.jpg)  
Figure 2: Left: a relational task table is chronologically split by anchor times, with a leakage barrier (red) separating validation from training context, further split into respective target task windows of size $\Delta w .$ , induced by the predictive queries $Q ( t _ { i } )$ . Right: evaluation proceeds over increments of length $\Delta I ,$ shifting the train/validation horizon $t _ { v a l } ^ { ( i ) }$ and instantiating the queries at successive anchors, while extracting context $G _ { \leq t }$ and future targets from the newly revealed task windows.

The objective of an RDL model $f _ { \theta }$ is to predict the target label $\hat { y } = f _ { \theta } ( G _ { v , \leq t } ) \approx y$ , where $G _ { v , \leq t }$ is the relational subgraph defining the computational context for entity v, strictly filtered to only include information available at or before anchor time t. Depending on the role of t, these tasks are categorized into static and temporal formulations.

Static Tasks. Static tasks involve predicting missing values or properties within a database viewed as a single, fixed snapshot in time. The temporal dimension is effectively ignored, reducing instances to pairs $( v , y )$ . The objective simplifies to learning a function $\hat { y } = f _ { \theta } ( G _ { v } )$ , utilizing the entire available graph G without risk of temporal information leakage. In these cases, the target label y belongs to a feature column that already structurally exists within the database but is masked out.

Temporal Tasks. Conversely, temporal tasks (or “forecasting tasks”) predict future behaviors or events that have not yet occurred. Here, the anchor time t is crucial, as the label y represents an aggregation or occurrence in a future window $\left( t , t + \Delta w \right] \left( \mathbf { e . g . } \right.$ , total sales over the next 30 days). The set of task labels is formed as $Y = \{ Q _ { \Delta w } ( t ) \ : | \ : t \in \mathrm { T } _ { t a s k } \}$ , where Q is a future-pointing task query <sup>1</sup> with a fixed window size $\Delta w .$ , and $\dot { \mathrm { T } _ { t a s k } }$ is a set of task anchor timestamps (see Figure 2). To prevent temporal data leakage, the model must strictly condition its representations on past data, requiring the graph to be temporally masked to yield $G _ { \leq t } ^ { \cdot } = ( \nu _ { \leq t } , \mathcal { E } _ { \leq t } )$ , where $\mathcal { V } _ { \leq t } = \{ \bar { u ^ { \cdot } } \in \mathcal { V } | \tau ( u ) \overset { \cdot } { \leq } t \}$

Data Splits. The strategy for partitioning data varies fundamentally between the tasks. For static tasks, test instances $( v , y )$ can be split randomly. Temporal tasks, however, necessitate strict chrono logical evaluation to prevent backward data leakage (Figure 2). These tasks further define preset validation and test timestamps: the training set contains instances where $t < t _ { v a l }$ , the validation set where $t _ { v a l } \leq t < t _ { t e s t }$ , and the test set where $t \geq t _ { t e s t }$ . This ensures the model is evaluated exclusively on unseen future time periods.

## 4 Incremental Evaluation Paradigm

Standard evaluation of models on relational databases relies on static dataset snapshots, failing to capture the dynamic evolution of real-world data where temporal increments dictate both target creation and predictive tasks. To standardize the measurement of model robustness against undetected data shifts and temporal concept drifts, we propose an incremental evaluation paradigm that simulates the real-world deployment lifecycle (Figure 1). This enables researchers to assess how well RDL models adapt to continuously growing databases over time.

## 4.1 Multi-Episodic Relational Tasks

Expanding upon the single-episode temporal tasks defined by future-pointing queries $Q _ { \Delta w } ( t )$ (Section 3.3), we introduce a fixed time increment $\Delta I \ge \Delta $ w representing the temporal stride between consecutive model updates (Figure 2), where an episode constitutes a batch of new target data arriving over this increment.

To transition to multi-episodic setting, we evaluate and update models incrementally over a sequence of monotonically increasing evaluation horizons: $T _ { \Delta I } = \{ t _ { i } \ | \ t _ { i } = t _ { i - 1 } + \Delta I , \quad t _ { i } \stackrel { . } { \leq } t _ { v a l } , \quad \bar { i } \geq 1 \}$

At each episode $i ,$ new training examples generated by $Q _ { \Delta w } ( t )$ from newly available anchor times $t \in ( t _ { i - 1 } , t _ { i } ]$ are revealed up to the anchor time $t _ { i } .$ This interval may contain multiple base task windows $\Delta w$ . Once evaluated, the true labels become available, and the increment is appended to the historical training data. This strict append-only growth (Figure 1a) prevents temporal leakage, ensuring the model never accesses information beyond its current validation horizon.

## 4.2 Incremental Training Regimes

As the prediction horizon shifts from episode i − 1 to episode i (Figure 2), a new data increment becomes available. The training set correspondingly expands to include all data up to the new validation timestamp $t _ { v a l } ^ { ( i ) }$ . The validation and test datasets shift forward chronologically, maintaining their structural role but evaluating the model on the newly established horizon. Because an increment can span multiple task windows, these shifted splits dynamically adapt to test the model on the exact block of newly revealed target data.

To evaluate how effectively models assimilate these increments, we benchmark four training regimes:

1. From Scratch (Baseline): Trainedfrom scratch using all available data up to $t _ { v a l } ^ { ( i ) }$ , serving as a (computationally expensive) baseline that follows standard single-episode RDL training.

2. Finetune (Cumulative): Retains weights from episode $i - 1$ and fine-tunes on the entire historical dataset up to $t _ { v a l } ^ { ( i ) }$

3. Finetune (Incremental): Updates the episode i − 1 model exclusively on the newest data increment to maximize training speed and immediate responsiveness.

4. Finetune (Upsampled): A hybrid strategy fine-tuning on a distribution that upsamples the most recent increment while retaining historical data to mitigate catastrophic forgetting. This is parameterized by a probability governing how often new increment data is drawn.

## 4.3 Time-Weighted Evaluation Metrics

In multi-episodic settings, near-future predictions are typically more operationally critical than longterm forecasts. Standard evaluations, however, weight all future samples equally, which can mask a model’s responsiveness to immediate temporal shifts. To prioritize near-future accuracy, we introduce a time-weighted exponential decay applicable to any standard metric (e.g., AUROC, MAE).

Given N future evaluation windows ordered by temporal distance from the training horizon (where $i = 1$ is the immediate next window) and a decay rate $\alpha \in [ 0 , 1 ]$ , we assign a normalized weight $W _ { s , i }$ to each sample s in window i:

$$
W _ { s , i } = \frac { ( 1 - \alpha ) ^ { i - 1 } } { \sum _ { j = 1 } ^ { N } | S _ { j } | ( 1 - \alpha ) ^ { j - 1 } }\tag{1}
$$

where $| S _ { j } |$ is the number of samples in window $j .$ This reduces to a uniform mean when $\alpha = 0$ while $\alpha  1$ strictly isolates performance on the immediate next increment.

## 5 Experiments

We evaluate our multi-episodic paradigm using 12 tasks (both regression and binary classification) defined on 4 different databases from the established RelBench [33] benchmark, which currently serves as the standard for evaluating RDL methods [24, 40]. Through these experiments, we (i)

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Task</td><td colspan="3">Original Data Splits</td><td colspan="2">Total N of ∆</td><td colspan="2">Avg. Samples</td><td colspan="2">Span in days</td></tr><tr><td>Train</td><td>Val</td><td>Test</td><td>∆w</td><td>∆I</td><td>∆w</td><td>∆I</td><td>∆w</td><td>∆I</td></tr><tr><td rowspan="3">rel-avito</td><td>ad-ctr</td><td>5.1k</td><td>1.8k</td><td>1.8k</td><td>5</td><td>4</td><td>1.7k</td><td>2.3k</td><td>4</td><td>6</td></tr><tr><td>user-clicks</td><td>59.5k</td><td>21.2k</td><td>48.0k</td><td>5</td><td>4</td><td>20.2k</td><td>26.9k</td><td>4</td><td>6</td></tr><tr><td>user-visits</td><td>86.6k</td><td>30.0k</td><td>36.1k</td><td>5</td><td>4</td><td>29.2k</td><td>38.9k</td><td>4</td><td>6</td></tr><tr><td rowspan="3">rel-f1</td><td>driver-position</td><td>7.5k</td><td>499</td><td>760</td><td>310</td><td>13</td><td>28</td><td>663</td><td>77</td><td>1812</td></tr><tr><td>driver-dnf</td><td>11.4k</td><td>566</td><td>702</td><td>475</td><td>13</td><td>27</td><td>998</td><td>48</td><td>1814</td></tr><tr><td>driver-top3</td><td>1.4k</td><td>588</td><td>726</td><td>116</td><td>4</td><td>23</td><td>647</td><td>60</td><td>1928</td></tr><tr><td rowspan="3">rel-stack</td><td>post-votes</td><td>2.45M</td><td>156k</td><td>161k</td><td>48</td><td>20</td><td>55.5k</td><td>137k</td><td>91</td><td>225</td></tr><tr><td>user-badge</td><td>3.39M</td><td>247k</td><td>255k</td><td>42</td><td>19</td><td>88.6k</td><td>202k</td><td>91</td><td>207</td></tr><tr><td>user-engagement</td><td>1.36M</td><td>85.8k</td><td>88.1k</td><td>48</td><td>20</td><td>30.8k</td><td>76.1k</td><td>91</td><td>225</td></tr><tr><td rowspan="3">rel-trial</td><td>site-success</td><td>151k</td><td>19.7k</td><td>22.6k</td><td>21</td><td>9</td><td>8.6k</td><td>21.4k</td><td>365</td><td>912</td></tr><tr><td>study-adverse</td><td>43.3k</td><td>3.6k</td><td>3.1k</td><td>21</td><td>9</td><td>2.3k</td><td>5.9k</td><td>365</td><td>912</td></tr><tr><td>study-outcome</td><td>12.0k</td><td>960</td><td>825</td><td>21</td><td>9</td><td>648</td><td>1.6k</td><td>365</td><td>912</td></tr></table>

Table 1: Temporal and evaluation statistics across datasets and tasks [33] used in the experiments. We report the original split sizes, alongside the total number, average sample count, and temporal span (in days) for both the evaluation windows $( \Delta w )$ and increments (∆I). Large sample counts are abbreviated (k = thousands, M = millions).

analyze the temporal data distribution, (ii) confirm the presence of temporal concept drift, (iii) evaluate the effectiveness of knowledge transfer via incremental fine-tuning, and (iv) test the time-weighted metrics prioritizing near-future accuracy.

## 5.1 Incremental Data Analysis

Expanding upon the single-episode temporal tasks defined by future-pointing queries $Q _ { \Delta w } ( t _ { i } )$ (Section 3.3 and Figure 2), our multi-episodic paradigm incrementally evaluates models using a fixed time increment ∆I (Section 4.1).

Table 1 details the temporal increments $( \Delta w , \Delta I )$ and the original split sizes. Notably, we set the time increment $\Delta I$ based on the temporal span of the original validation split, $[ t _ { v a l } , t _ { t e s t } )$ , facilitating seamless comparison with the original single-episode setting [33]. The table also indicates the number of available training episodes. We note that, because the final two increments correspond to the original validation and test splits and are excluded from training, tasks on the rel-avito dataset (due to short temporal coverage) and the rel-f1 driver-top3 task (due to the chosen increment span) yield only two training episodes.

To detail the temporal data distribution, Figure $3 \mathrm { a } ^ { 2 }$ presents new sample arrivals per time window ∆w alongside cumulative task sizes. From the evolution of task sizes, we identify two principal growth patterns: roughly linear (rel-avito ad-ctr, rel-f1 driver-position) and exponential (rel-stack user-engagement, rel-trial study-adverse). Figure 3b illustrates the evolving cumulative target-value distributions. Notably, with the exception of rel-avito ad-ctr (which spans a brief interval), all remaining tasks exhibit clear evidence of temporal concept drift.

## 5.2 Experimental Setup

We implement our incremental evaluation and training framework using PyTorch, PyTorch Lightning, and PyTorch Geometric, building upon the RelBench [33] and REDELEX [29] libraries.

Model Architecture and Initialization. We employ a standard RDL benchmark architecture [33, 9]: tabular ResNet [14] node encoders (via PyTorch Frame [17]) paired with GraphSAGE [16]. The core GNN uses 2 message-passing layers (hidden dimensionality 128, sum aggregation), followed by task-specific prediction heads with batch normalization.

![](images/566b1e2617bcb616cb2bed41f2a467758b88746d9bce9d457ae3e7a05f19c054.jpg)  
(a) Number of new samples arriving in each time window $\Delta w ,$ together with the cumulative task size, highlighting differences in growth across successive windows in different datasets.

![](images/17995339b68622fd0177379b175a38140c51aa4a1adaeea14a2b09612907dba8.jpg)  
(b) Cumulative value of target y over the full range off the task windows $\Delta w .$ . Mean value with standard deviation for regression tasks, positive class rate for the binary classification tasks.  
Figure 3: Visual analysis of data distribution over time windows $\Delta w$ from anchor timestamps $t _ { i }$

Graph Construction and Sampling. Following the RDL blueprint, databases are processed into temporal heterogeneous PK-FK graphs (Section 3). To handle large scales, we use mini-batch training with uniform temporal neighbor sampling, restricting context to edges and nodes valid at or prior to the anchor timestamp to strictly prevent temporal leakage (Figure 2). We sample a maximum of 32 neighbors for the first GNN layer and 16 for the second (decaying by a factor of two). For the Finetune (Upsampled) regime, a composed data loader samples mini-batches equally (50% probability) from the newest data increment and the historical data (Figure 1c).

Training and Optimization. Models are optimized via Adam (learning rate 0.001) with a batch size of 128. During each episodic increment, models train for a maximum of 2000 steps, artificially limited to 100 batches per epoch to ensure frequent validation checks. All configurations are evaluated across 5 random seeds, reporting the mean and standard deviation. Further compute, runtime, and orchestration details can be found in Appendix A.1.

![](images/6191d01aca6ed4666a94a02c1a4aa5b91ef777178f43d2eef180834603a2e71c.jpg)

![](images/4ccbe38ae65a415278efd1bf8aa64444931a2f9c764610ba96afb8ad96084846.jpg)

![](images/61a4bb64db6f810320a9590edeab0c3f0d5a061307fc87d175365e608ab7b760.jpg)

![](images/8a9da7575c5515807fb0473c13da732536a3d7b0866472d4d498bcab25e2aa9f.jpg)  
Figure 4: Performance of the From Scratch models trained in the multi-episodic setting, reported on the original validation and test splits. Metrics are Mean Absolute Error (MAE; lower indicates better performance) for regression tasks and AUROC (higher indicates better performance) for classification.

## 5.3 Investigating Temporal Concept Drift

To confirm that the observed distribution shifts constitute temporal concept drift, we incrementally train baseline From Scratch models and assess them on the original validation and test splits. Concretely, given anchor times $T _ { \Delta I } = \{ t _ { i } \mid t _ { i } = t _ { i - 1 } + \Delta I , t _ { i } \leq t _ { v a l } \} , ^ { 3 }$ each model $M _ { i }$ is trained up to $t _ { i } ,$ with the best checkpoint selected via validation on the subsequent increment (up to $t _ { i + 1 } )$

![](images/af1dd1a4661fa92f649ccceca1c4c4b45901b4773d4e1b621c5664ec9686301e.jpg)  
(a) Top model metrics after complete training.  
(b) Model metrics after the first 100 training steps.  
Figure 5: Performance of different model training regimes in the multi-episodic setting, reporting the validation score for the subsequent increment. Metrics are, again, MAE (lower indicates better performance) and AUROC (higher indicates better performance).

Figure 4 then substantiates the presence of concept drift through a general trend of performance degradation over time across tasks. A minor exception to that trend is rel-f1 driver-position, which exhibits a substantial drop in performance, likely attributable to its relatively small dataset size. Nevertheless, the overall empirical validation of the drift confirms the original motivation for the RDL focus on temporal dynamics.

## 5.4 Evaluating Fine-tuning Regimes

Our multi-episodic setting enables RDL methods that exploit incrementally expanding data. Figure 5a compares the top performances of our three fine-tuning regimes (Section 4.2) against a From Scratch baseline. The fine-tuning regimes consistently match or outperform the baseline, an effect most pronounced in the rel-stack user-engagement and rel-trial study-adverse tasks.

To further illustrate knowledge transfer capabilities, Figure 5b shows model performance after just 100 training steps, resembling a Few-Shot Learning scenario. Notably, pretrained models quickly achieve performance on par with or superior to a fully trainedfrom scratch baseline, underscoring the computational efficiency and predictive potential of continuously fine-tuned models.

## 5.5 Decayed Evaluation Metrics

To account for the operational prioritization of near-future predictions, Table 2 applies our temporally decayed metrics (Section 4.3) to the regression tasks. A decay of 0.0 yields a standard uniform average, while 0.9 places almost all weight on the immediate next window. Notably, except for driver-position, varying the decay factor preserves the optimal training regime rankings. Nevertheless, absolute metric values shift in distinct patterns: post-votes and ad-ctr exhibit minimal variation, study-adverse shows a consistent performance drop, and driver-position improves at α = 0.3—a variation we suspect reflects differing rates of localized temporal volatility across datasets, with a detailed analysis left for future work.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Task</td><td rowspan="2">Decay</td><td colspan="4">Training Regime - MAE ↓</td></tr><tr><td>FT Cuml.</td><td>FT Incr.</td><td>FT Upspl.</td><td>Scratch</td></tr><tr><td rowspan="3">rel-avito</td><td rowspan="3">ad-ctr</td><td>0.0</td><td>0.037</td><td>0.037</td><td>0.039</td><td>0.039</td></tr><tr><td>0.3</td><td>0.037</td><td>0.037</td><td>0.039</td><td>0.038</td></tr><tr><td>0.9</td><td>0.035</td><td>0.036</td><td>0.037</td><td>0.037</td></tr><tr><td rowspan="3">rel-f1</td><td rowspan="3">driver-position</td><td>0.0</td><td>3.256</td><td>3.408</td><td>3.278</td><td>3.106</td></tr><tr><td>0.3</td><td>2.858</td><td>2.922</td><td>3.058</td><td>2.827</td></tr><tr><td>0.9</td><td>3.466</td><td>3.293</td><td>3.569</td><td>3.358</td></tr><tr><td rowspan="3">rel-stack</td><td rowspan="3">post-votes</td><td>0.0</td><td>0.061</td><td>0.062</td><td>0.062</td><td>0.066</td></tr><tr><td>0.3</td><td>0.061</td><td>0.062</td><td>0.062</td><td>0.065</td></tr><tr><td>0.9</td><td>0.059</td><td>0.060</td><td>0.060</td><td>0.063</td></tr><tr><td rowspan="6">rel-trial</td><td rowspan="3">site-success</td><td>0.0</td><td>0.428</td><td>0.395</td><td>0.419</td><td>0.393</td></tr><tr><td>0.3</td><td>0.431</td><td>0.396</td><td>0.420</td><td>0.392</td></tr><tr><td>0.9</td><td>0.440</td><td>0.399</td><td>0.421</td><td>0.389</td></tr><tr><td rowspan="3">study-adverse</td><td>0.0</td><td>46.369</td><td>45.996</td><td>46.545</td><td>48.143</td></tr><tr><td>0.3</td><td>46.565</td><td>46.163</td><td>46.819</td><td>48.281</td></tr><tr><td>0.9</td><td>47.239</td><td>46.735</td><td>47.760</td><td>48.755</td></tr></table>

Table 2: Performance of individual training regimes, trained on the complete training split, assessed using the temporally decayed MAE over the combined original validation and test splits.

## 6 Conclusion

In this work, we addressed a major shortcoming in standard Relational Deep Learning (RDL) evaluation: the reliance on static dataset snapshots, which fundamentally misaligns with the continuous, dynamic evolution of real-world databases. By formalizing an incremental, multi-episodic evaluation and training paradigm, we provide a framework that accurately reflects the real-world lifecycle of database growth and model deployment.

Our empirical analysis across large-scale relational datasets establishes several core findings. First, we validated the widespread prevalence of temporal concept drift in standard RDL tasks, highlighting why snapshot-based testing is insufficient. Furthermore, we showed that incremental fine-tuning strategies successfully mitigate this degradation. Transferring knowledge from previous training episodes consistently matched or exceeded the performance of expensive from-scratch retraining, often adapting within just a few optimization steps. Finally, introducing a time-weighted exponential decay metric allowed for a more realistic assessment of a model’s utility by properly prioritizing near-future predictive accuracy.

Altogether, the proposed paradigm and associated training regimes lay the groundwork for transitioning RDL research from isolated benchmarks toward practical, continually learning systems.

Limitations. While our experimental setup currently focuses on append-only database growth, real-world systems often involve data updates and deletions (CRUD operations) [6]. Although our multi-episodic paradigm can in principle accommodate these mutations—such as by dynamically updating historical node states or utilizing temporal edges [21]—doing so introduces complex attribute dependencies that require further formal investigation. Furthermore, while we establish foundational fine-tuning regimes, mitigating catastrophic forgetting over extended horizons remains a challenge [42]. Although the incremental fine-tuning offers a highly scalable alternative as cumulative data grows, optimally balancing this computational efficiency with long-term knowledge retention may require integrating more advanced continual learning techniques, such as dynamic replay buffers or weight regularization [28]. Finally, extending this continuous evaluation to emerging monolithic RDL foundation models [37, 19, 32, 36] remains an open challenge to fully understand their specific resilience to temporal concept drift.

## References

[1] R. Agrawal, A. Somani, and Y. Xu. Storage and querying of e-commerce data. In VLDB, volume 1, pages 149–158, 2001.

[2] S. Brody, U. Alon, and E. Yahav. How attentive are graph attention networks? In International Conference on Learning Representations, 2022.

[3] D. D. Chamberlin and R. F. Boyce. SEQUEL: A structured English query language. In Proceedings of the 1974 ACM SIGFIDET (now SIGMOD) workshop on Data description, access and control, SIGFIDET ’74, pages 249–264, New York, NY, USA, 1974. Association for Computing Machinery.

[4] K.-Y. Chen, P.-H. Chiang, H.-R. Chou, T.-W. Chen, and D. T.-H. Chang. Trompt: towards a better deep neural network for tabular data. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023.

[5] T. Chen, C. Kanatsoulis, and J. Leskovec. RelGNN: Composite message passing for relational deep learning. In Forty-second International Conference on Machine Learning, 2025.

[6] E. F. Codd. A relational model of data for large shared data banks. Commun. ACM, 13(6):377–387, June 1970.

[7] M. Cvitkovic. Supervised learning on relational databases with graph neural networks. arXiv preprint arXiv:2002.02046, 2020.

[8] C. R. Da Xu, E. Korpeoglu, S. Kumar, and P. Awasthi. Inductive representation learning on temporal graphs. In International Conference on Learning Representations, 2020.

[9] V. P. Dwivedi, S. Jaladi, Y. Shen, F. Lopez, C. I. Kanatsoulis, R. Puri, M. Fey, and J. Leskovec. Relational graph transformer. In Temporal Graph Learning Workshop @ KDD 2025, 2025.

[10] V. P. Dwivedi, C. Kanatsoulis, S. Huang, and J. Leskovec. Relational deep learning: Challenges, foundations and next-generation architectures. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, KDD ’25, page 5999–6009, New York, NY, USA, 2025. Association for Computing Machinery.

[11] M. Fey, W. Hu, K. Huang, J. E. Lenssen, R. Ranjan, J. Robinson, R. Ying, J. You, and J. Leskovec. Position: Relational deep learning - graph representation learning on relational databases. In Forty-first International Conference on Machine Learning, 2024.

[12] M. Fey, V. Kocijan, F. Lopez, J. E. Lenssen, and J. Leskovec. KumoRFM: A Foundation Model for In-Context Learning on Relational Data. May 2025.

[13] Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko. Revisiting deep learning models for tabular data. In Proceedings ofthe 35th International Conference on Neural Information Processing Systems, NIPS ’21, pages 18932–18943, Red Hook, NY, USA, 2021. Curran Associates Inc.

[14] Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko. Revisiting deep learning models for tabular data. In Proceedings ofthe 35th International Conference on Neural Information Processing Systems, NIPS ’21, pages 18932–18943, Red Hook, NY, USA, 2021. Curran Associates Inc.

[15] J. Gu, R. Ranjan, C. Kanatsoulis, H. Tang, M. Jurkovic, V. Hudovernik, M. Znidar, P. Chaturvedi, P. Shroff, F. Li, and J. Leskovec. RelBench v2: A Large-Scale Benchmark and Repository for Relational Data, Feb. 2026. arXiv:2602.12606 [cs].

[16] W. Hamilton, Z. Ying, and J. Leskovec. Inductive Representation Learning on Large Graphs. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

[17] W. Hu, Y. Yuan, Z. Zhang, A. Nitta, K. Cao, V. Kocijan, J. Leskovec, and M. Fey. Pytorch frame: A modular framework for multi-modal tabular learning. arXiv preprint arXiv:2404.00776, 2024.

[18] X. Hu, W. Tang, C.-K. Hsieh, and S. Shi. Tabtransformer: Tabular data modeling using contextual embeddings. arXiv preprint arXiv:2012.06678, 2020.

[19] V. Hudovernik, F. López, V. Kocijan, A. Nitta, J. E. Lenssen, J. Leskovec, and M. Fey. KumoRFM-2: Scaling Foundation Models for Relational Learning, Apr. 2026. arXiv:2604.12596 [cs].

[20] M. Jurkovic, V. Hudovernik, and E. Štrumbelj. Syntherela: A benchmark for synthetic relational database generation. In Will Synthetic Data Finally Solve the Data Access Problem?, 2025.

[21] S. M. Kazemi, R. Goel, K. Jain, I. Robertson, H. Schmid, S. Srinivasan, S. Kieffer, Y. Chen, and P. Poupart. Representation learning for dynamic graphs: A survey. The Journal of Machine Learning Research, 21(1):2648–2718, 2020.

[22] V. Kocijan, J. Sunil, J. E. Lenssen, V. Deb, X. Xe, F. R. Gomez, M. Fey, and J. Leskovec. Predictive Query Language: A Domain-Specific Language for Predictive Modeling on Relational Databases, Feb. 2026. arXiv:2602.09572 [cs].

[23] V. Kothapalli, R. Ranjan, V. Hudovernik, V. P. Dwivedi, J. Hoffart, C. Guestrin, and J. Leskovec. PluRel: Synthetic Data unlocks Scaling Laws for Relational Foundation Models, Feb. 2026. arXiv:2602.04029 [cs].

[24] D. Lachi, M. Mohammadi, J. Meyer, V. Arora, T. Palczewski, and E. L. Dyer. RGP: A Cross-Attention based Graph Transformer for Relational Deep Learning. Oct. 2025.

[25] V. Lachi, A. Longa, B. Bevilacqua, B. Lepri, A. Passerini, and B. Ribeiro. Boosting Relational Deep Learning with Pretrained Tabular Models, Apr. 2025. arXiv:2504.04934 [cs].

[26] J. Lu, A. Liu, F. Dong, F. Gu, J. Gama, and G. Zhang. Learning under concept drift: A review. IEEE transactions on knowledge and data engineering, 31(12):2346–2363, 2018.

[27] P. Moritz, R. Nishihara, S. Wang, A. Tumanov, R. Liaw, E. Liang, M. Elibol, Z. Yang, W. Paul, M. I. Jordan, and I. Stoica. Ray: A distributed framework for emerging ai applications, 2018.

[28] G. I. Parisi, R. Kemker, J. L. Part, C. Kanan, and S. Wermter. Continual lifelong learning with neural networks: A review. Neural networks, 113:54–71, 2019.

[29] J. Peleška and G. Šír. Redelex: A framework for relational deep learning exploration. In Machine Learning and Knowledge Discovery in Databases., pages 438–456. Springer Nature Switzerland, 2025.

[30] J. Peleška and G. Šír. Tabular transformers meet relational databases. ACM Trans. Intell. Syst. Technol., 16(5), Sept. 2025.

[31] J. Peleška and G. Šír. Task-Agnostic Contrastive Pretraining for Relational Deep Learning, June 2025. arXiv:2506.22530 [cs].

[32] R. Ranjan, V. Hudovernik, M. Znidar, C. Kanatsoulis, R. Upendra, M. Mohammadi, J. Meyer, T. Palczewski, C. Guestrin, and J. Leskovec. Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data, Oct. 2025. arXiv:2510.06377 [cs].

[33] J. Robinson, R. Ranjan, W. Hu, K. Huang, J. Han, A. Dobles, M. Fey, J. E. Lenssen, Y. Yuan, Z. Zhang, X. He, and J. Leskovec. Relbench: A benchmark for deep learning on relational databases. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024.

[34] E. Rossi, B. Chamberlain, F. Frasca, D. Eynard, F. Monti, and M. Bronstein. Temporal graph networks for deep learning on dynamic graphs. arXiv preprint arXiv:2006.10637, 2020.

[35] P. Velickovi ˇ c, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio. Graph attention´ networks. In International Conference on Learning Representations, 2018.

[36] Y. Wang, X. Wang, Q. Gan, M. Wang, Q. Yang, D. Wipf, and M. Zhang. Griffin: Towards a Graph-Centric Relational Database Foundation Model, May 2025. arXiv:2505.05568 [cs].

[37] J. Wehrstein, C. Binnig, F. Ozcan, S. Vasudevan, Y. Gan, and Y. Wang. Towards foundation database models. In Proceedings ofthe 15th Annual Conference on Innovative Data Systems Research (CIDR), 2025.

[38] J. White. PubMed 2.0. Medical Reference Services Quarterly, 39(4):382–387, Oct. 2020.

[39] F. Wu, V. P. Dwivedi, and J. Leskovec. Large language models are good relational learners. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 7835–7854, 2025.

[40] J. Yin, P. Huo, B. Zhu, H. Yan, S. Wang, S. Pan, and C. Zhang. Rel-MOSS: Towards Imbalanced Relational Deep Learning on Relational Databases, Mar. 2026. arXiv:2603.07916 [cs].

[41] L. Zahradník, J. Neumann, and G. Šír. A deep learning blueprint for relational databases. In NeurIPS 2023 Second Table Representation Learning Workshop, 2023.

[42] X. Zhang, D. Song, and D. Tao. Continual graph learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.

## A Technical appendices and supplementary material

This appendix provides additional implementation details, full data-distribution analyses, and extended experimental results that complement the main text.

## A.1 Detailed Experimental Setup and Compute Resources

We expand the experimental overview from Section 5.2 by summarizing the implementation stack and the main architectural, data-loading, and training choices. All models are implemented in PyTorch using PyTorch Lightning and PyTorch Geometric, and integrate with the RelBench [33] and REDELEX [29] libraries.

Model Architecture and Initialization. As formalized in the source scripts, the core architecture relies on a HeterogeneousSAGE module. The relational databases are processed into temporal heterogeneous graphs following the primary key-foreign key (PK-FK) relationships. To initialize the node features from the raw database attributes, we utilize HeteroEncoder built upon PyTorch Frame [17], leveraging tabular architectures such as ResNet [14] to produce initial node embedding matrices of dimensionality 128.

Crucially, temporal dynamics are injected via a HeteroTemporalEncoder. For a given prediction instance, this module computes relative time encodings based on the anchor time of the target entity (Section 3.3). These temporal embeddings are directly added to the initial node features before being passed to the GNN. The graph message passing is handled by a 2-layer HeteroGraphSAGE using sum aggregation. Finally, a single-layer MLP with batch normalization serves as the task-specific prediction head. All components are trained end-to-end to jointly learn the tabular representations and the graph structure.

Graph Construction and Sampling. Given the large scale of the datasets, we utilize mini-batch training with a temporal neighbor sampling strategy. Specifically, we sample a maximum of 32 neighbors for the first GNN layer and 16 neighbors for the second layer (decaying by a factor of two per layer). To strictly prevent temporal leakage, we use a uniform temporal sampling strategy that only considers edges and nodes valid at or prior to the anchor timestamp (Section 3.3). For the Finetune (Upsampled) training regime (Section 4.2), we construct a specialized data loader that samples mini-batches from the newest data increment and the historical data with an equal 50% probability.

Training and Optimization. The models are optimized using the Adam optimizer with an initial learning rate of 0.001. We use a uniform batch size of 128 for all tasks. During each episodic increment (Section 4), models are trained for a maximum of 2000 optimization steps. We artificially limit the training epochs to 100 batches per epoch to ensure frequent validation checks and early stopping.

All experiments were orchestrated using Ray [27] to manage the distributed execution of the incremental training regimes (Section 4.2) across different time splits. Trials were automatically scheduled on available GPU resources, most commonly on the Tesla V100-SXM2-32GB. To ensure computational feasibility and simulate realistic production constraints, a strict maximum time limit of 2 hours was enforced for each individual incremental training step. Model checkpointing was handled by PyTorch Lightning, saving the best model weights based on the validation metric at each epoch.

## A.2 Data Analysis

Figures 6 and 7 present the full temporal distribution analysis for all 12 evaluated tasks. Whereas the main text focused on selected illustrative cases, these complete results confirm that the same qualitative patterns described there also hold more generally. In Figure 6, we see that tasks defined on a shared dataset exhibit similar distributions of samples across time windows (Section 4.1), including the categorization into the ‘linear’ and ‘exponential’ categories. In contrast, Figure 7 shows that the target value distributions vary considerably between tasks, with pronounced distributional shifts that are particularly evident for the tasks derived from the rel-f1 dataset.

![](images/a2ca259feff031d521e85dfa32244e697a38210d868592db548bfa78ca296520.jpg)  
Figure 6: Number of new samples from individual time windows ∆w and the cumulative size of the task over time.

![](images/b4de5179cc92e6d19852b7f8ea4d8ea676a8618df2a01a8047021438b1983db0.jpg)  
Figure 7: Cumulative value of target y over the full range off the task windows ∆w. Mean value with standard deviation for regression tasks, positive class rate for the binary classification tasks.

## A.3 Extended Results

This section reports the complete set of performance metrics for all evaluated tasks, extending the illustrative examples provided in the main text.

Figure 8 presents the performance of the From Scratch baseline (Section 4.2) across all increments (Section 4). With the exception of rel-f1 driver-dnf and rel-trial site-success, all tasks display a clear performance improvement for episodes nearer to the validation horizon, underscoring the susceptibility of static models to temporal concept drift.

Figures 9 and 10 contrast the different proposed incremental training regimes (Section 4.2). The results consistently indicate that our fine-tuning strategies substantially enhance performance in the few-shot scenario shown in Figure 10, highlighting the potential of knowledge transfer between the episodes, and frequently also improve fully trained models, with the effect being especially pronounced for tasks defined on the rel-stack database.

Tables 3 and 4 broaden the time-weighted evaluation from the main text (Section 4.3) by reporting standard deviations over 5 independent runs for regression and classification tasks, respectively. Notably, Table 4 highlights the consistently superior performance of the fine-tuning regimes, with the sole exception of rel-trial study-outcome, which remains challenging, as traditional tabular models can outperform RDL [33] on this task.

![](images/04dac47bbc0176f01fc7bc8d7f9d07171f4b8b7731d41a5328bdc48bd3f95fe5.jpg)  
Figure 8: Performance of the From Scratch models trained in the multi-episodic setting, reported on the original validation and test splits. Metrics are Mean Absolute Error (MAE; lower indicates better performance) for regression tasks and AUROC (higher indicates better performance) for binary classification.

![](images/71cad760a90dd5ee9359556c691d5b47c01745191f990a8231cb96e8e7d43c52.jpg)  
Figure 9: Top performance of different model training regimes in the multi-episodic setting, reporting the validation score for the subsequent increment. Metrics are Mean Absolute Error (MAE; lower indicates better performance) for regression tasks and AUROC (higher indicates better performance) for binary classification.

![](images/551aa931f4dccb507bb617cd3a25e8cabed2872b0d10fe951d681f497042d724.jpg)  
Figure 10: Performance of different model training regimes in the multi-episodic setting after first 100 training steps, reporting the validation score for the subsequent increment. Metrics are Mean Absolute Error (MAE; lower indicates better performance) for regression tasks and AUROC (higher indicates better performance) for binary classification.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Task</td><td rowspan="2">Decay</td><td colspan="4">Training Regime</td></tr><tr><td> $\mathrm { { F T } C u m l . }$ </td><td>FT Incr.</td><td> $\mathrm { F T } \mathrm { U p s p l . }$ </td><td>Scratch</td></tr><tr><td rowspan="3">rel-avito</td><td rowspan="3">ad-ctr</td><td>0.0</td><td> ${ \bf 0 . 0 3 7 \pm 0 . 0 0 1 }$ </td><td> ${ \bf 0 . 0 3 7 \pm 0 . 0 0 1 }$ </td><td> $0 . 0 3 9 \pm 0 . 0 0 2$ </td><td> $0 . 0 3 9 \pm 0 . 0 0 2$ </td></tr><tr><td>0.3</td><td> ${ \bf 0 . 0 3 7 \pm 0 . 0 0 1 }$ </td><td> ${ \bf 0 . 0 3 7 \pm 0 . 0 0 1 }$ </td><td> $0 . 0 3 9 \pm 0 . 0 0 2$ </td><td> $0 . 0 3 8 \pm 0 . 0 0 2$ </td></tr><tr><td>0.9</td><td> ${ \bf 0 . 0 3 5 \pm 0 . 0 0 1 }$ </td><td> $0 . 0 3 6 \pm 0 . 0 0 1$ </td><td> $0 . 0 3 7 \pm 0 . 0 0 2$ </td><td> $0 . 0 3 7 \pm 0 . 0 0 2$ </td></tr><tr><td rowspan="3">rel-f1</td><td rowspan="3">driver-position</td><td>0.0</td><td> $3 . 2 5 6 \pm 0 . 0 5 6$ </td><td> $3 . 4 0 8 \pm 0 . 0 5 3$ </td><td> $3 . 2 7 8 \pm 0 . 1 0 4$ </td><td> ${ \bf 3 . 1 0 6 \pm 0 . 1 2 5 }$ </td></tr><tr><td>0.3</td><td> $2 . 8 5 8 \pm 0 . 0 5 5$ </td><td> $2 . 9 2 2 \pm 0 . 0 2 1$ </td><td> $3 . 0 5 8 \pm 0 . 1 5 3$ </td><td> ${ \bf 2 . 8 2 7 \pm 0 . 1 1 6 }$ </td></tr><tr><td>0.9</td><td> $3 . 4 6 6 \pm 0 . 1 3 5$ </td><td> ${ \bf 3 . 2 9 3 \pm 0 . 1 0 7 }$ </td><td> $3 . 5 6 9 \pm 0 . 3 1 6$ </td><td> $3 . 3 5 8 \pm 0 . 2 4 7$ </td></tr><tr><td rowspan="3">rel-stack</td><td rowspan="3">k post-votes</td><td>0.0</td><td> $\mathbf { 0 . 0 6 1 } \pm 0 . 0 0 0$ </td><td> $0 . 0 6 2 \pm 0 . 0 0 1$ </td><td> $0 . 0 6 2 \pm 0 . 0 0 1$ </td><td> $0 . 0 6 6 \pm 0 . 0 0 2$ </td></tr><tr><td>0.3</td><td> $\mathbf { 0 . 0 6 1 } \pm 0 . 0 0 1$ </td><td> $0 . 0 6 2 \pm 0 . 0 0 1$ </td><td> $0 . 0 6 2 \pm 0 . 0 0 1$ </td><td> $0 . 0 6 5 \pm 0 . 0 0 2$ </td></tr><tr><td>0.9</td><td> $\mathbf { 0 . 0 5 9 \pm 0 . 0 0 1 }$ </td><td> $0 . 0 6 0 \pm 0 . 0 0 1$ </td><td> $0 . 0 6 0 \pm 0 . 0 0 1$ </td><td> $0 . 0 6 3 \pm 0 . 0 0 2$ </td></tr><tr><td rowspan="6">rel-trial</td><td rowspan="3">site-success</td><td>0.0</td><td> $0 . 4 2 8 \pm 0 . 0 1 7$ </td><td> $0 . 3 9 5 \pm 0 . 0 0 3$ </td><td> $0 . 4 1 9 \pm 0 . 0 0 8$ </td><td> ${ \bf 0 . 3 9 3 \pm 0 . 0 1 3 }$ </td></tr><tr><td>0.3</td><td> $0 . 4 3 1 \pm 0 . 0 1 6$ </td><td> $0 . 3 9 6 \pm 0 . 0 0 3$ </td><td> $0 . 4 2 0 \pm 0 . 0 0 6$ </td><td> ${ \bf 0 . 3 9 2 \pm 0 . 0 1 1 }$ </td></tr><tr><td>0.9</td><td> $0 . 4 4 0 \pm 0 . 0 1 4$ </td><td> $0 . 3 9 9 \pm 0 . 0 0 5$ </td><td> $0 . 4 2 1 \pm 0 . 0 0 6$ </td><td> ${ \bf 0 . 3 8 9 \pm 0 . 0 0 7 }$ </td></tr><tr><td rowspan="3">study-adverse</td><td>0.0</td><td> $4 6 . 3 6 9 \pm 0 . 1 0 4$ </td><td> $\mathbf { 4 5 . 9 9 6 \pm 0 . 3 2 1 }$ </td><td> $4 6 . 5 4 5 \pm 0 . 3 0 4$ </td><td> $\overline { { 4 8 . 1 4 3 \pm 0 . 1 8 3 } }$ </td></tr><tr><td>0.3</td><td> $4 6 . 5 6 5 \pm 0 . 0 9 2$ </td><td> ${ \bf 4 6 . 1 6 3 \pm 0 . 3 0 0 }$ </td><td> $4 6 . 8 1 9 \pm 0 . 2 9 9$ </td><td> $4 8 . 2 8 1 \pm 0 . 1 7 5$ </td></tr><tr><td>0.9</td><td> $4 7 . 2 3 9 \pm 0 . 1 2 0$ </td><td> ${ \bf 4 6 . 7 3 5 \pm 0 . 2 6 7 }$ </td><td> $4 7 . 7 6 0 \pm 0 . 3 2 7$ </td><td> $4 8 . 7 5 5 \pm 0 . 1 7 0$ </td></tr></table>

Table 3: Performance of individual training regimes on regression tasks, trained on the complete training split, assessed using temporally decayed Mean Absolute Error with standard deviations (lower indicates better performance) over the combined original validation and test splits.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Task</td><td rowspan="2">Decay</td><td colspan="4">Training Regime</td></tr><tr><td>FT Cuml.</td><td>FT Incr.</td><td>FT Upspl.</td><td>Scratch</td></tr><tr><td rowspan="6">rel-avito</td><td rowspan="2">user-clicks</td><td>0.0</td><td> ${ \bf 0 . 6 5 9 \pm 0 . 0 0 5 }$ </td><td> $\mathbf { 0 . 6 5 9 \pm 0 . 0 0 6 }$ </td><td> $0 . 6 5 6 \pm 0 . 0 1 1$ </td><td> $0 . 6 4 8 \pm 0 . 0 1 3$ </td></tr><tr><td>0.3</td><td> $\mathbf { 0 . 6 5 7 \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 6 5 7 \pm 0 . 0 0 6 }$ </td><td> $0 . 6 5 4 \pm 0 . 0 1 0$ </td><td> $0 . 6 4 6 \pm 0 . 0 1 2$ </td></tr><tr><td rowspan="2"></td><td>0.9</td><td> $0 . 6 4 5 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 6 4 6 \pm 0 . 0 0 4 }$ </td><td> $0 . 6 4 1 \pm 0 . 0 0 7$ </td><td> $0 . 6 3 7 \pm 0 . 0 0 6$ </td></tr><tr><td>0.0</td><td> $0 . 6 6 5 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 6 6 6 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 6 6 6 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 6 6 6 \pm 0 . 0 0 3 }$ </td></tr><tr><td rowspan="2">user-visits</td><td>0.3</td><td> $0 . 6 6 7 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 6 6 9 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 6 6 9 \pm 0 . 0 0 1 }$ </td><td> $0 . 6 6 8 \pm 0 . 0 0 4$ </td></tr><tr><td>0.9</td><td> $0 . 6 7 6 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 6 7 8 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 6 7 8 \pm 0 . 0 0 1 }$ </td><td> $0 . 6 7 7 \pm 0 . 0 0 4$ </td></tr><tr><td rowspan="6">rel-f1</td><td rowspan="2">driver-dnf</td><td>0.0</td><td> $0 . 7 6 3 \pm 0 . 0 1 6$ </td><td> $\mathbf { 0 . 8 1 8 \pm 0 . 0 0 1 }$ </td><td> $0 . 7 7 6 \pm 0 . 0 0 8$ </td><td> $0 . 8 1 4 \pm 0 . 0 1 1$ </td></tr><tr><td>0.3</td><td> $0 . 7 3 7 \pm 0 . 0 3 4$ </td><td> ${ \bf 0 . 7 5 9 \pm 0 . 0 1 5 }$ </td><td> $0 . 7 4 5 \pm 0 . 0 2 4$ </td><td> $0 . 7 5 5 \pm 0 . 0 1 6$ </td></tr><tr><td rowspan="2"></td><td>0.9</td><td> $0 . 6 9 5 \pm 0 . 0 7 4$ </td><td> $\mathbf { 0 . 7 1 6 \pm 0 . 0 0 4 }$ </td><td> $0 . 6 8 7 \pm 0 . 0 5 3$ </td><td> $0 . 6 7 0 \pm 0 . 0 2 5$ </td></tr><tr><td>0.0</td><td> $\overline { { 0 . 8 9 2 \pm 0 . 0 0 6 } }$ </td><td> $\mathbf { \overline { { 0 . 8 9 8 \pm 0 . 0 0 3 } } }$ </td><td> $\overline { { 0 . 8 4 1 \pm 0 . 0 0 9 } }$ </td><td> $\overline { { 0 . 8 6 7 \pm 0 . 0 1 6 } }$ </td></tr><tr><td rowspan="2">driver-top3</td><td>0.3</td><td> ${ \bf 0 . 7 7 9 \pm 0 . 0 0 7 }$ </td><td> ${ \bf 0 . 7 7 9 \pm 0 . 0 0 9 }$ </td><td> $0 . 7 2 5 \pm 0 . 0 2 9$ </td><td> $0 . 7 5 4 \pm 0 . 0 2 0$ </td></tr><tr><td>0.9</td><td> $\mathbf { 0 . 6 8 9 \pm 0 . 0 1 8 }$ </td><td> $0 . 6 7 1 \pm 0 . 0 1 6$ </td><td> $0 . 6 5 9 \pm 0 . 0 5 3$ </td><td> $0 . 6 5 3 \pm 0 . 0 4 8$ </td></tr><tr><td rowspan="6">rel-stack</td><td rowspan="3">user-badge</td><td>0.0</td><td> $0 . 8 8 9 \pm 0 . 0 0 0$ </td><td> $\mathbf { 0 . 8 9 0 \pm 0 . 0 0 1 }$ </td><td> $0 . 8 8 9 \pm 0 . 0 0 1$ </td><td> $0 . 8 8 2 \pm 0 . 0 0 0$ </td></tr><tr><td>0.3</td><td> $0 . 8 9 0 \pm 0 . 0 0 0$ </td><td> $\mathbf { 0 . 8 9 1 \pm 0 . 0 0 1 }$ </td><td> $0 . 8 9 0 \pm 0 . 0 0 1$ </td><td> $0 . 8 8 3 \pm 0 . 0 0 1$ </td></tr><tr><td>0.9</td><td> $\mathbf { 0 . 8 9 4 \mathop { \pm } 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 8 9 4 \pm 0 . 0 0 1 }$ </td><td> $0 . 8 9 3 \pm 0 . 0 0 1$ </td><td> $0 . 8 8 7 \pm 0 . 0 0 1$ </td></tr><tr><td rowspan="3">user-engagement</td><td>0.0</td><td> $\mathbf { 0 . 9 0 2 \pm 0 . 0 0 1 }$ </td><td> $\overline { { 0 . 9 0 1 \pm 0 . 0 0 1 } }$ </td><td> $\overline { { 0 . 9 0 1 \pm 0 . 0 0 1 } }$ </td><td> $0 . 8 9 6 \pm 0 . 0 0 1$ </td></tr><tr><td>0.3</td><td> $\mathbf { 0 . 9 0 2 \pm 0 . 0 0 1 }$ </td><td> $0 . 9 0 1 \pm 0 . 0 0 1$ </td><td> $0 . 9 0 1 \pm 0 . 0 0 1$ </td><td> $0 . 8 9 6 \pm 0 . 0 0 1$ </td></tr><tr><td>0.9</td><td> $\mathbf { 0 . 9 0 1 \pm 0 . 0 0 1 }$ </td><td> $0 . 9 0 0 \pm 0 . 0 0 1$ </td><td> $0 . 8 9 9 \pm 0 . 0 0 1$ </td><td> $0 . 8 9 5 \pm 0 . 0 0 2$ </td></tr><tr><td rowspan="3">rel-trial</td><td rowspan="3">study-outcome</td><td>0.0</td><td> $0 . 6 3 3 \pm 0 . 0 0 6$ </td><td> $0 . 6 5 3 \pm 0 . 0 0 5$ </td><td> $0 . 6 3 8 \pm 0 . 0 2 4$ </td><td> $\mathbf { 0 . 6 5 7 \pm 0 . 0 0 4 }$ </td></tr><tr><td>0.3</td><td> $0 . 6 3 1 \pm 0 . 0 0 6$ </td><td> $0 . 6 5 1 \pm 0 . 0 0 5$ </td><td> $0 . 6 3 5 \pm 0 . 0 2 4$ </td><td> ${ \bf 0 . 6 5 6 \pm 0 . 0 0 4 }$ </td></tr><tr><td>0.9</td><td> $0 . 6 2 6 \pm 0 . 0 0 7$ </td><td> $0 . 6 4 6 \pm 0 . 0 0 4$ </td><td> $0 . 6 2 6 \pm 0 . 0 2 4$ </td><td> $\mathbf { 0 . 6 5 1 \pm 0 . 0 0 5 }$ </td></tr></table>

Table 4: Performance of individual training regimes, trained on the complete training split, assessed using temporally decayed AUROC with standard deviations (higher indicates better performance) over the combined original validation and test splits.