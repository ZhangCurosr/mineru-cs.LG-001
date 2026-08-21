# End-to-end Early Classification of Time Series in Non-Stationary Environments

Aurélien Renault aurelien.renault@orange.com Orange Research & AgroParisTech Paris, France

Antoine Cornuéjols antoine.cornuejols@agroparistech.fr AgroParisTech UMR MIA-Paris Palaiseau, France

Alexis Bondu   
alexis.bondu@orange.com   
Orange Research   
Paris, France

## Abstract

Early Classification of Time Series (ECTS) requires making accurate decisions as early as possible in inherently online and evolving environments. Yet, most existing methods assume stationarity and rely on separable designs, where classification and triggering are optimized independently—an assumption that fundamentally limits their adaptability under drift. In this work, we challenge this paradigm and study ECTS under non-stationary conditions. We provide the first systematic comparison between separable and endto-end approaches across controlled drifting scenarios. Building on Reinforcement Learning, we introduce DQeND, a unified architecture that jointly learns representation, classification, and triggering decisions, while remaining directly comparable to state-of-the-art separable baselines. Across a wide range of drifts, DQeND demonstrates strong robustness across various non-stationary scenarios, consistently outperforming separable baselines. An ablation study further highlights that jointly updating representation and decision modules is critical to these gains. Overall, our results indicate that end-to-end learning can ofer improved adaptation capabilities for ECTS in dynamic environments, and motivate further investigation of alternatives to separable designs.

## CCS Concepts

• Computing methodologies → Temporal reasoning; Online learning settings; Reinforcement learning.

## Keywords

Time Series, Online Learning, Reinforcement Learning

Aurélien Renault, Alexis Bondu, Antoine Cornuéjols, and Vincent Lemaire. 2026. End-to-end Early Classification of Time Series in Non-Stationary Environments . In . ACM, New York, NY, USA, 10 pages. https://doi.org/10. 1145/nnnnnnn.nnnnnnn

Vincent Lemaire vincent.lemaire@orange.com Orange Research Paris, France

## 1 Motivation

Early Classification of Time Series (ECTS) has emerged as a critical paradigm for applications where timely decision-making is as important as predictive accuracy [2, 10, 21]. By allowing a model to issue a classification before observing the full time series, ECTS systems inherently operate under a trade-of between earliness and accuracy. While substantial progress has been made in designing accurate early classifiers, most existing approaches are developed and evaluated under the implicit assumption of a stationary datagenerating process. However, this assumption rarely holds in practical deployments, where the process generating successive time series can be subject to distribution shifts and concept drifts. Despite the growing maturity of the ECTS literature, the impact of non-stationarity remains largely unexplored.

For instance, detecting toxic behavior on social media is subject to strong temporal constraints, as harmful content should be identified and filtered as early as possible. Moreover, the nature of such content can evolve with current events, e.g. during election periods, requiring early detection systems to continuously adapt to shifting definitions of what should be flagged.

One specificity of ECTS is that it can be viewed as a dual decision process: (i) the classification task, which aims to predict the class label based on partial observations, and (ii) the trigger (or stopping) task, which determines the optimal time at which a prediction should be emitted. The vast majority of existing methods adopt a separable design, where these two components are implemented independently and optimized one after another [10, 21]. Such a decomposition ofers practical advantages, including flexibility, interpretability, and ease of optimization. However, it also implicitly assumes that classification and stopping decisions can be optimized independently.

This assumption becomes particularly limiting when the underlying time series distribution evolves over time. In such conditions, both the classification boundaries and the optimal stopping time may evolve in a coupled manner. A separable framework, by design, may struggle to capture this joint adaptation, as each component is typically optimized in a sequential fashion, e.g., encoder → classifier → trigger, preventing upstream modules from benefiting from feedback provided by downstream decisions. In contrast, end-to-end architectures alleviate these unidirectional optimization dependencies by enabling the learning signal, namely the incurred cost, to propagate throughout the entire decision pipeline. Yet, this potential benefit of end-to-end learning for ECTS under distribution drift has yet to be investigated. Indeed, on one hand, existing work on end-to-end ECTS are limited in scope: they are often evaluated under stationary conditions and rarely benchmarked against strong separable baselines. On the other hand, existing work on nonstationarity remains in the separable scope and mainly focuses on adapting the triggering mechanism under changing decision costs, while assuming a fixed classification model [23]. Consequently, a question stands out: does jointly optimizing classification and stopping decisions improve robustness and adaptability when the underlying time series distribution itself evolves? And can we identify where the diferences with separable methods, if any, manifest themselves?

In this work, we provide a systematic investigation of separable and end-to-end ECTS architectures under controlled nonstationarities. Our contributions are threefold:

• We conduct an extensive empirical comparison between separable and end-to-end paradigms, explicitly analyzing the regimes in which joint optimization becomes beneficial.

• Building on recent work on Reinforcement Learning (RL)- based ECTS [18, 22], we propose an end-to-end framework that jointly optimizes classification and triggering decisions, while being directly comparable to state-of-the-art separable baselines.

• We design controlled experimental protocols involving covariate shifts and concept drifts, demonstrating that endto-end optimization improves robustness and adaptation capabilities across a wide range of evolving environments.

Together, our results provide new insights into the design ofadaptive ECTS systems and highlight the importance ofjointly learning prediction and stopping decisions in dynamic environments. Experiments of this paper are fully reproducible and the source code can be found at https://anonymous.4open.science/r/ects\_concept\_drift-8177

The remainder of this paper is organized as follows. Section 2 provides background on ECTS. Section 3 reviews related ECTS works. Section 4 describes the proposed RL-based end-to-end ECTS pipeline. Section 5 describes the experimental protocol, including the evaluation metrics, synthetic datasets, and the controlled cost drift scenarios. Finally, Section 6 presents and discusses the experimental results for several types of non-stationarities, and Section 7 concludes and discusses future work.

## 2 Background

## 2.1 Problem statement

In the ECTS problem, an input time series of size � is observed progressively. At time $t \leq T _ { i }$ only a prefix $\mathbf { x } _ { t } ~ = ~ \langle x _ { 1 } , \ldots , x _ { t } \rangle$ is available, where $\{ x _ { i } \} _ { ( 1 \leq i \leq t ) }$ denotes the measurements (possibly multivariate). The full series x<sub>�</sub> belongs to an unknown class $y \in { \mathcal { Y } } .$ The task is to output a prediction $\hat { y } \in \mathcal { Y }$ for the class of the incoming time series at some time $\hat { t } \in [ 1 , T ]$ before the deadline �.

ECTS explicitly addresses a trade-of between predictive accuracy and earliness. The prediction correctness is measured by a misclassification cost $C _ { m } ( \hat { y } | y )$ , and time pressure is modeled by a delay cost $C _ { d } ( t )$ , which is assumed to be positive and typically non-decreasing over time. Formally, we consider: $C _ { m } : y \times y \to$ R, $C _ { d } : \mathbb { R } ^ { + }  \mathbb { R }$

Let � ∈ S be some ECTS function belonging to a class offunctions $s ,$ whose output at time � when receiving x<sub>�</sub> is:

$$
s ( \mathbf { x } _ { t } ) = { \left\{ \begin{array} { l l } { \emptyset } & { { \mathrm { ~ i f ~ e x t r a ~ m e a s u r e s ~ a r e ~ q u e r i e d } } ; } \\ { ( { \hat { y } } _ { t } , \ t ) } & { { \mathrm { ~ i f ~ p r e d i c t i o n ~ i s ~ t r i g g e r e d ~ a t ~ } } t < T ; } \end{array} \right. }\tag{1}
$$

Specifically, when the deadline is reached, i.e. $t = T _ { \cdot }$ , the model is forced to return a prediction: $s ( \mathbf x _ { T } ) = ( \hat { y } _ { T } , T )$ In practice, ECTS is an online optimization problem: at each time �, the function $s ( \mathbf { x } _ { t } )$ must decide whether to predict or wait for more data, based only on the prefix $\mathbf { X } _ { t } .$ . The � function induces a stopping time �ˆ on each incoming series that is the first time step for which � returns a non-null prediction. It can be defined as:

$$
\hat { t } = \operatorname* { i n f } \{ t \mid s ( x _ { t } ) \neq 0 \}\tag{2}
$$

Once triggered, the remaining observations $\left\{ \mathbf { x } _ { t } \right\} _ { ( \hat { t } < t \leq T ) }$ are not considered and prediction process is stopped. The goal is to choose �ˆ such that the incurred, time-dependent, loss $\mathcal { L } ( s ( \mathbf { x } _ { \hat { t } } ) , y ) = \mathbb { C } _ { m } ( \hat { y } _ { \hat { t } } | y ) +$ $\mathrm { C } _ { d } ( \hat { t } )$ is as small as possible.

From a machine learning perspective, the goal is to find a function $s \in S$ that best optimizes the time-dependent loss function $\mathcal { L } ,$ minimizing the true risk over the data-generating process $\mathbb { P } ( X \times \mathcal { Y } )$ [21]:

$$
\underset { s \in S } { \arg \operatorname* { m i n } } \mathbb { E } _ { ( \mathbf { x } _ { T } , y ) \sim \mathbb { P } _ { ( X \times y ) } } \left[ \mathcal { L } ( s ( \mathbf { x } _ { \hat { t } } ) , y ) \right]\tag{3}
$$

In order to solve the ECTS problem, a training set is made available that is composed of � labeled time series, denoted by $( \mathbf { x } _ { T } ^ { i } , y ^ { i } ) _ { i \in [ 0 , N ] } \in ( X \times y )$ , where each series $\mathbf { x } _ { T } = \langle x _ { 1 } , \dots , x _ { T } \rangle$ is complete and of the same size $T ,$ and associated with its label $y \in \mathcal { Y }$

## 2.2 Online ECTS with drift

The standard ECTS formulation typically assumes that the datagenerating process observed at training time remains unchanged at deployment. This implicitly entails that both the input distribution and the underlying predictive relationship are stationary. In practice, this assumption is often violated: the actual conditions encountered in production may difer from those seen during training and can evolve over time. Such variations may afect the input distribution (covariate shift), or the conditional relationship between inputs and labels (concept drift). Those may occur either within the duration of a single time series (i.e., along the time index $t \in [ 0 , T ] )$ or across longer time scales, for instance due to seasonal or operational changes.

To explicitly account for these diferent temporal dimensions, we distinguish between the time index of a given time series and the deployment timeline. Let $u \in \mathbb { N } ^ { + }$ index successive incoming labeled time series $\{ ( { \bf x } _ { T } , y ) ^ { u } \} _ { ( 1 \leq u \leq U ) }$ observed during deployment, $( { \bf x } _ { T } , y ) ^ { u }$ is generated by some distribution $\mathcal { D } _ { u }$ over $\chi _ { \times } y$

In the general case, the deployment process is characterized by several sources of uncertainty: (i) the distribution $\mathcal { D } _ { u }$ is unknown to the learner at deployment; (ii) it can be non-stationary, i.e., drifting over deployment time �.

At deployment, an ECTS system repeatedly processes incoming time series drawn from the evolving sequence of distributions $\left\{ \mathcal { D } _ { u } \right\} _ { u \geq 1 }$ . The deployment performance of some ECTS function � ∈ S is then measured by:

$$
R _ { U } ( s ) = \sum _ { u = 1 } ^ { U } \mathbb { E } _ { ( \mathbf { x } _ { T } , y ) \sim \mathcal { D } _ { u } } \big [ \mathcal { L } ( s ( \mathbf { x } _ { \hat { t } } ) , y ) \big ] ,\tag{4}
$$

where the expectation is taken with respect to the current (and potentially drifting) data-generating process, as well as any additional sources of randomness.

The remainder of the paper instantiates and compares diferent controlled forms of non-stationarity, encompassing data and concept drifts and evaluates ECTS methods’ robustness across this wide range of dynamic scenarios.

## 3 Related work

The ECTS literature can be broadly divided into end-to-end and separable approaches. End-to-end methods jointly learn a classifier and a triggering mechanism. While flexible, they are classifierspecific and often dificult to compare consistently across studies. In contrast, separable approaches decouple classification and decision triggering. This modular design dominates the literature and has been extensively evaluated [2, 10, 21].

## 3.1 Separable ECTS

Classification module. A common design choice in separable ECTS systems consists in discretizing the temporal axis and learning a collection of classifiers, each specialized for a given timestamp or subset of timestamps. This strategy enables the direct reuse of state-of-the-art Time Series Classification algorithms, while ensuring that predictions can be produced at diferent stages of the sequence. Among recent advances, the MiniROCKET pipeline [7] has emerged as a particularly strong choice, combining high predic tive performance with excellent scalability. Its efectiveness stems from the use of a large number of fixed random convolutional kernels, coupled with simple yet informative aggregation functions, yielding rich representations at a very low computational cost. As such, MiniROCKET-based classifiers are frequently adopted as backbone models in modern separable ECTS pipelines [3, 22, 23].

Trigger module. On the decision side, a widely used baseline is the Proba Threshold strategy, which triggers a prediction as soon as the maximum posterior class probability exceeds a predefined threshold. Despite its simplicity, this approach has proven to be competitive across a variety of datasets and cost configurations, and is therefore often used as a reference method [21].

Beyond such myopic strategies, most high-performing trigger models fall into the class ofnon-myopic approaches, which explicitly account for the potential evolution of future observations when making decisions. The Economy family of methods [1, 4] follows this principle by triggering a decision when the expected cost at the current time step is minimal among all anticipated future costs. Similarly, Calimera [3] adopts an anticipation-based framework, leveraging regression models learned via backward induction to estimate future costs and guide the triggering decision.

Recently, RL has been introduced as a flexible alternative for modeling the triggering process. In particular, Alert [22] learns a triggering policy, based on a state space composed by handcrafted features, obtained by probing the ensemble of calibrated MiniROCKET estimators. In addition to strong empirical performance, this approach is inherently well-suited to online adaptation and can naturally incorporate feedback from a deployment phase. For those reasons, the present work primarily builds upon this trigger model and investigates its integration within an end-to-end architecture, where the classifier is directly informed by the triggering signal.

## 3.2 End-to-end ECTS

Historically, the earliest works that can be considered end-to-end ECTS consist of a classifier-specific stopping criterion, directly integrated within a base Time Series Classification algorithm. A seminal example is the work of [28], which determines a unique minimum prediction length such that the predictions of a 1-nearest neighbor classifier remain the same as those obtained using a complete time series. In a similar vein, the EDSC algorithm [29] builds upon shapelet-based classification by associating each shapelet with a temporal penalty, thereby implicitly favoring patterns that occur earlier in the time series.

More recent advances in deep learning have enabled tighter integration between representation learning, classification, and decision timing. In particular, ELECTS [24] proposes an LSTM-based architecture, augmented by a stopping head that determines the probability of triggering. It is trained with a dedicated loss function that explicitly balances classification accuracy and earliness.

In parallel, the growing popularity of RL has led to a number of approaches that formulate ECTS as a sequential decision-making problem. Earlier work by [17, 18] introduced the use of Deep Q-Networks (DQN) for ECTS, modeling the triggering task as a valuebased decision process. EARLIEST [11, 12] relies on a policy gradient method to learn a stopping policy jointly with a LSTM-based encoder and classifier. These RL-based methods ofer a natural framework to capture the trade-of between earliness and accuracy, while enabling flexible adaptation to complex and potentially non-stationary environments.

## 4 DQeND: End-to-end RL for ECTS

In this section, the design choices for the proposed end-to-end ECTS pipeline, called DQeND, are described. In contrast to traditional separable approaches, where the diferent components are optimized independently, our architecture integrates them into a single decision process that can be trained and adapted holistically. The pipeline consists of two main modules: (i) an encoder that transforms raw time series prefixes into informative latent representations, and (ii) a RL–based decision module that simultaneously determines when to stop and which class to predict.

To enable fair comparisons with state-of-the-art separable methods and isolate the impact of end-to-end learning from architectural diferences, we build our pipeline upon both ROCKET-type and Alert frameworks [6, 7, 22]. The global DQeND pipeline is illustrated in Figure 1.

## 4.1 Architecture

4.1.1 ELM Encoder. The design of the encoder module is guided by the objective of combining strong representational capacity with computational eficiency in an online setting. To this end, we draw inspiration from the principles underlying the widely adopted, ROCKET family of methods [6, 7], whose features are computed via a large number of random convolutions. More broadly, the idea of generating a large hidden representation of the input data using random weights originates from the Extreme Learning Machine (ELM) literature [13, 27].

![](images/e783fbed4bbd2ff910911e2143f18a8bb150c932cbd1c7531b078091a14eef75.jpg)  
Figure 1: DQeND pipeline

At a high level, ELM-based approaches are single-layer neural networks. They rely on projecting input data into a high-dimensional feature space using a large set of randomly initialized and fixed parameters, and subsequently learning a lightweight mapping on top of this representation. Formally, the output of some ELM network with � hidden nodes can be defined as:

$$
f _ { L } ( \mathbf { x } ) = \sum _ { i = 1 } ^ { L } \beta _ { i } h _ { i } ( \mathbf { x } )\tag{5}
$$

where $\beta = [ \beta _ { 1 } , \cdots , \beta _ { L } ]$ is the learned weight matrix, and $\mathbf { h } ( \mathbf { x } ) =$ $[ h _ { 1 } ( { \bf x } ) , \cdot \cdot \cdot , h _ { L } ( { \bf x } ) ]$ the fixed random features for the �-th hidden node.

This paradigm has been shown to yield competitive performance across a wide range of time series classification tasks [15, 20], while maintaining a favorable computational profile due to the limited number of trainable parameters. Such properties are particularly desirable in online learning scenarios, where frequent updates must be performed under eficiency constraints.

Combining the ELM principles with the ROCKET-type of time series transformations, the proposed encoder implements a randomized feature extractor based on a single one-dimensional convolutional layer with frozen weights. Multiple kernel sizes � are employed in parallel in order to capture temporal patterns at different scales, thereby enriching the expressiveness of the resulting representation. The convolutional outputs are then aggregated through a mean pooling operation, producing a fixed-size embed ding regardless of the input length. In particular, feature $h _ { i }$ can be obtained through the following operations:

$$
h _ { i } ( \mathbf { x } _ { t } ) = \mathrm { A v g P o o l } ( \mathrm { C o n v 1 D } _ { k } ( \mathbf { x } _ { t } ) )\tag{6}
$$

The convolutional filters are initialized by sampling their weights from a standard normal distribution and remain unchanged throughout training and deployment, ensuring a stable and inexpensive feature extraction process. To complement this representation, we additionally learn an explicit embedding of the time step, which is concatenated to the ELM-based features. This allows the downstream decision modules to incorporate temporal position information alongside the extracted signal patterns, a key aspect for early classification tasks where the decision process is inherently time-dependent.

4.1.2 DQN Decision. The decision-making module is designed to remain directly comparable to strong separable baselines while enabling full end-to-end optimization. To this end, we adopt the same underlying architecture as the Alert framework [22], namely a Deep Q-Network (DQN)-based decision model [5, 26]. This choice ensures that any observed diferences in performance can be attributed to the representation and training paradigm, rather than to architectural discrepancies in the decision process itself.

In contrast to separable approaches, Alert in particular, that typically rely on handcrafted features or classifier-derived confidence scores, the proposed module operates directly on the learned representation produced by the encoder. More precisely, the input state $s _ { t }$ is given by the embedding output at time �, allowing the decision module to leverage rich, task-adaptive features without additional engineering.

The action space is extended so as to jointly model the stopping decision and the class prediction. Specifically, we define $\mathcal { A } =$ $\{ ^ { \ast } w a i t ^ { \ast }$ , “predict class $\vec { 1 ^ { \circ } } , \cdots$ , “predict class $K ^ { \mathfrak { s } } \}$ , thereby unifying the triggering and classification tasks within a single decision process. At each time step, the selected action follows the standard greedy policy induced by the Q-function:

$$
a _ { t } = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } Q ( s _ { t } , a )\tag{7}
$$

The prediction mechanism is derived directly from the chosen action. If the model selects a prediction action, the corresponding class is immediately output. A specific handling is introduced for the terminal time step�: ifthe selected action remains “wait”, the model is forced to emit a prediction, corresponding to the class associated with the highest Q-value among the non-waiting actions. Formally, the predicted label is defined as:

$$
\hat { y } _ { t } ( a _ { t } ) = \left\{ \begin{array} { l l } { 0 } & { \mathrm { i f ~ } a _ { t } = \mathrm { ' } w a i t ^ { \mathrm { * } } \mathrm { ~ a n d ~ } t < T } \\ { \begin{array} { l l } { \mathrm { a r g ~ m a x } } & { Q ( s _ { t } , a ) } \\ { a \in \mathcal { R } \backslash \{ ^ { \circ } w a i t ^ { \mathrm { * } } \} } & { \mathrm { i f ~ } a _ { t } = \mathrm { ' } w a i t ^ { \mathrm { * } } \mathrm { ~ a n d ~ } t = T } \\ { a _ { t } } & { \mathrm { i f ~ } a _ { t } \neq \mathrm { ' } w a i t ^ { \mathrm { * } } } \end{array} } \end{array} \right.\tag{8}
$$

## 4.2 Training (ofline)

For stability reasons, learning is performed ofline in a batch setting, where the model is trained on a fixed collection of fully labeled time series without requiring active exploration during training. In particular, the overall training procedure is kept close to that of Alert [22].

The RL objective is defined through a reward function that captures the trade-of between earliness and accuracy. At each time step, the reward is expressed as the temporal variation of the delay cost when the agent decides to wait, and additionally incorporates the misclassification cost when a prediction is made. Formally, the reward is defined as:

$$
r ( t ) = \left\{ \begin{array} { l l } { - \Delta C _ { d } ( t ) } & { \mathrm { i f } a _ { t } = ^ { \mathrm { { \normalsize ~ \alpha \omega \omega \omega \omega \omega \omega } } } w a i t ^ { \mathrm { { \tiny ~ \cdot ~ } } } } \\ { - C _ { m } ( \hat { y } _ { t } | y ) - \Delta C _ { d } ( t ) } & { \mathrm { i f } a _ { t } \neq ^ { \mathrm { { \tiny ~ \cdot ~ } } } w a i t ^ { \mathrm { { \tiny ~ \cdot ~ } } } } \end{array} \right.\tag{9}
$$

where $- \Delta \Sigma _ { d } ( t ) = \Sigma _ { d } ( t ) - \Sigma _ { d } ( t - 1 )$ , with ${ \mathrm C } _ { d } ( 0 ) = 0$

The gradient ofthe DQN objective, corresponding to the Bellman error, is propagated through the entire DQeND pipeline, updating all parameters associated with the shaded arrows in Figure 1. Finally, the model selection procedure is kept identical to that of [22].

## 4.3 Deployment (online)

The proposed framework naturally extends to an online adaptation setting, where both the encoder and the decision modules are jointly updated based on the feedback induced by the decisions. In contrast to separable approaches, this joint update enables the representation itself to evolve in response to non-stationary conditions, allowing for a tighter coupling between feature extraction and decisionmaking.

At deployment time, each observed interaction is stored in a replay bufer. The pipeline is then updated by sampling mini-batches of transitions from this bufer, following standard DQN optimization procedures.

When moving from ofline training to online adaptation, exploration is explicitly reintroduced in order to account for potential distribution shifts. Concretely, we adopt an �-greedy strategy, where with probability � a random action is selected, and with probability $1 - \epsilon$ the greedy action, i.e. the one with maximum �-value, is chosen. In the exploration phase, actions are sampled uniformly from the action space, ensuring equal probability across all possible decisions $( \mathrm { e . g . , 1 / 3 }$ for each action in the binary classification case). In all experiments, the exploration rate is fixed to $\epsilon = 5 \%$ . Details on all the hyperparameters’ values can be found in Table 1.

## 5 Experimental Protocol

This section describes the experimental protocol used to evaluate the proposed online ECTS methods under non-stationarities. We detail the evaluation metrics, the data generation process, the drifts scenarios, and the evaluated methods. All these elements are fixed prior to the analysis of the results presented in Section 6.

## 5.1 Evaluation Metrics

All online methods are evaluated using an eval-then-update (prequential) protocol [8]. At each deployment step �, a time series and its associated deployment-time costs are first used to evaluate the current model. The observed loss is then used to update the models according to its update strategy.

Table 1: DQeND hyperparameters
<table><tr><td rowspan="4">Encoder</td><td>kernel sizes</td><td>{3, 6, 9, 12}</td></tr><tr><td>nb. of kernels</td><td>{128, 64, 32, 16}</td></tr><tr><td>embedding dim.</td><td>64</td></tr><tr><td>time embedding dim.</td><td>2</td></tr><tr><td rowspan="3">Decision</td><td>nb. hidden layers</td><td>1</td></tr><tr><td>hidden dim.</td><td>64</td></tr><tr><td>activation function</td><td>tanh</td></tr><tr><td rowspan="3">Training</td><td>learning rate</td><td>1e-3</td></tr><tr><td>nb. epochs</td><td>100</td></tr><tr><td>batch size</td><td>256</td></tr><tr><td rowspan="3">Deployment</td><td>learning rate</td><td>1e-4</td></tr><tr><td>exploration rate €</td><td>0.05</td></tr><tr><td>batch size</td><td>64</td></tr></table>

To study the impact of diferent trade-ofs between misclassification and delay costs, we use the standard weighted formulation of the average cost, along with the cost functions definitions used by [1, 3, 21]:

$$
\begin{array} { c } { { \displaystyle { \cal A } \nu g C o s t _ { \alpha } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } C _ { m } ( \hat { y } _ { i } \mid y _ { i } ) + C _ { d } ( \hat { t } _ { i } ) } } \\ { { = \displaystyle { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \alpha \mathbb { 1 } ( \hat { y } \neq y ) + ( 1 - \alpha ) \frac { t } { T } } } } \end{array}
$$

where $\alpha \in [ 0 , 1 ]$ controls the relative importance of accuracy and earliness. Throughout this work, as in [16, 19], we fix $\alpha = 0 . 8$

The evaluation relies on a set of three metrics derived from the standard AvgCost, in order to capture complementary aspects of performance under non-stationary conditions:

• First, we report the cumulative cost, defined as the total sum of incurred costs over all deployment examples.

• Second, we consider the hold-out AvgCost drift, computed on a disjoint evaluation set that is subject to the same drift conditions as the current deployment step. Unlike the raw cumulative deployment cost, which may exhibit high variance due to transient efects or small batches, this metric provides a more stable estimate of performance under the current time series distribution.

• Finally, we report the hold-out AvgCost original, evaluated on a separate hold-out set drawn from the original (training-time) distribution. This complementary metric assesses the extent to which a method preserves performance on the base concept, thereby ofering insight into its ability to balance adaptation and retention; often referred to as the stability–plasticity trade-of [14, 25].

## 5.2 Data Generation

Studying non-stationarity in ECTS requires benchmarks where the dificulty of the task and the nature of the drift can be precisely controlled. In particular, the ability to independently manipulate factors such as noise intensity, temporal location of discriminative patterns, or class structure is essential. The MNIST-1D [9] dataset naturally fulfills these requirements through its synthetic generation process, making it particularly well-suited for the controlled drift scenarios considered in this work. The MNIST-1D dataset is a ten-class classification problem, which has been binarized in this work, based on univariate time series of fixed length � = 100. Each generated time series contains a class-specific pattern representing the shape of a digit, embedded at a random position. Training phase is conducted using 1,000 examples (700 for training and the 300 left to compute the hold-out AvgCost original); at each deployment step �, 1,500 new (drifted) examples are generated, among which 500 are used for updates and 1,000 to compute the hold-out AvgCost drift.

## 5.3 Drift Scenarios

To evaluate the robustness ofonline ECTS methods to non-stationarities, we consider controlled variations of input data and concept. Nonstationary patterns occur across deployment steps �, while the time index � ∈ [0, � ] remains associated with the progression within an individual time series.

We first distinguish two temporal patterns of variation:

• Incremental (I) drifts occur when the drifted variable(s) evolve smoothly during deployment.

• Abrupt (A) drifts occur when the drifted variable(s) suddenly switch from the training regime to a new deployment regime.

We consider three variables over which to intervene in order to generate drifts:

• Template Location (loc) defines the location of the true class pattern within a time series.

• Gaussian Noise (noise) defines the scale of the Gaussian noise to be applied over a time series.

• Class shufling (class) defines the set of sub-classes characterizing the two binary (meta-)classes.

Combining these two dimensions yields four experimental scenarios:

(1) I\_loc: the class patterns are first located at the very beginning of the time series; it progressively shifts so that it reaches the end of the series at the end of deployment.

![](images/4227320147f23818ded0be2eba4738958e4c1d865ada53390f4d99e6eba6ec46.jpg)  
Figure 2: The I\_loc scenario.

(2) I\_noise: the scale of the iid Gaussian noise becomes progressively greater, following a log spaced schedule from 0.01 to 1.

![](images/dcce1c69def43c973b152f7f5dec608e78a3708eaec48d1636c8d0224b7bc3f9.jpg)  
Figure 3: The I\_noise scenario.

(3) A\_loc: the class patterns are first located within the first quarter of the series; after five deployment steps, they are shifted to the last quarter.

![](images/f4318222855a2451206405ddc477b2170061f01a58a6a0b6b048ae27ea4ff06f.jpg)  
Figure 4: The A\_loc scenario.

(4) A\_class: the set of sub-classes defining the two main classes is completely resampled, with a 5% probability of occurring at each deployment step.

![](images/f348d244c97c9585a974fb1bbb1d8358da0445d015365bdc4147b38eda62cbb7.jpg)  
Figure 5: The A\_class scenario.

## 5.4 Evaluated Methods

Separable baselines. As highlighted in Section 4, Alert constitutes the comparison reference for the proposed end-to-end method. Thus, all the separable models described in this section are based on: (i) a classifier that shares the same architecture as the ELM encoder described in Section 4.1.1, followed by a linear classification layer and (ii) a trigger which is Alert, as described in the original paper [22], including the handcrafted input space. In what follows, several ways of adapting separable baselines to online environments, depending on the addressed drift type, are presented.

## • Common baselines:

(i) bronze: both the classifier and the trigger model remain frozen during deployment.

(ii) silver: both the classifier and the trigger model are being trained on the concept on which they will be evaluated. Act as an upper bound, not realistic in practice.

## • Incremental drifts:

(i) online/freeze: the classifier is updated online, after each deployment step, the trigger model remains frozen.

(ii) online/online: both the classifier and the trigger model are updated online, after each deployment step. The data batch is split in two so that each module is updated based on one half of the batch.

## • Abrupt drifts:

(i) retrain/freeze: the classifier is retrained from scratch, at a given frequency, the trigger model remains frozen.

(ii) retrain/retrain: both the classifier and the trigger model are retrained from scratch, at a given frequency. The data batch is split in two so that each module is trained on the basis of one half of the batch.

End-to-end. The following end-to-end methods have been adapted so that they may be updated online: ELECTS [24] and EARLIEST [11].

## 5.5 Implementation details

The deployment phase consists of � = 100 steps of 500 examples each. For the retrain baselines, the retraining frequency has been set to 20. For both ELECTS and EARLIEST algorithms, training phase is conducted over 500 epochs, the deployment batch size has been set to 64 and the learning rate to 5e-3. All experiments have been run on CPU, with up to five threads to perform cross-validation, when needed. Results that follow consist of the mean taken over five diferent random seeds. For all other details, we make the code available at: https://anonymous.4open.science/r/ects\_concept\_drift-8177

## 6 Results

In this section, we analyze the ECTS online performances under the two types of drifts presented in Section 5.

## 6.1 Incremental drifts

Incremental Location (I\_loc). This first experiment investigates a gradual shift of discriminative patterns toward later positions in the deployment time series.

As shown in Figure 6a, end-to-end approaches clearly outperform separable baselines, consistently achieving the lowest cumulative costs (top panel). In particular, DQeND ranks first overall and, together with ELECTS, best preserves performance on the original concept (bottom panel), demonstrating a strong balance between adaptation and retention. In addition, the qualitative analysis in Figure 7 (top panel) shows that end-to-end models efectively adapt their triggering policies by progressively delaying predictions, illustrated by the increasing light-blue earliness curve of each box, as the discriminative patterns shift later in the series.

In contrast, both Alert-based separable methods degrade signifi cantly as informative patterns move later in the sequences (middle panel, right), highlighting the limitations of decoupled classifica tion and triggering under temporal drift. Freezing the Alert trigger further leads to unstable behavior throughout deployment, whereas updating both classifier and trigger slightly improves stability but still fails to improve global performance.

Incremental Noise (I\_noise). This second experiment evaluates robustness under a gradual increase of additive Gaussian noise, progressively degrading the signal-to-noise ratio and the reliability of discriminative patterns.

As displayed in Figure 6b, end-to-end approaches again achieve the best cumulative costs, with DQeND ranking first, closely followed by ELECTS. Despite similar overall performance, analysing Figure 7, bottom panel, indicates that the two methods exhibit distinct adaptation behaviors. ELECTS follows a conservative strategy, initially delaying predictions to preserve accuracy while the signal remains informative, before progressively shifting toward earlier decisions as noise becomes dominant. In contrast, DQeND rapidly converges toward an early-prediction regime once accuracy approaches random performance levels (explaining the plateau shape in Figure 6b, middle panel), efectively minimizing delay costs when waiting no longer provides useful information. This diference in the adaptation strategy is illustrated by a vertical shift of the earli ness curve of both algorithms.

![](images/f8d07ea5d3865569491a27cd967deaf405ffc1d87b33183ecebac078577ea24d.jpg)  
(a) I\_loc1

![](images/945fa6a6e1df97be6479c88c8de2105622dffb774e7505b633a5a75f189d0fd5.jpg)  
(b) I\_noise1  
Figure 6: Incremental drifts, deployment’s metrics, averaged over five seeds. From top to bottom: the raw cumulative cost; the hold-out AvgCost, obtained on unseen drifted data; the hold-out AvgCost original, obtained using the fixed test set from training phase. UPPERCASE approaches refers to endto-end methods while lowercase indicates separable ones.

Regarding separable approaches, the same trends as in the incremental location drift experiment can be observed. In particular, the frozen Alert trigger again produces unstable behavior, whereas updating both classifier and trigger significantly improves robustness. The online/online variant achieves performance close to DQeND under low to moderate noise levels (Figure 6b, middle panel, left), although the end-to-end model remains slightly more robust, suggesting that unified architectures better exploit weak residual signal under mild perturbations.

## 6.2 Abrupt drifts

Abrupt Location (A\_loc). This experiment evaluates an abrupt drift scenario in which the discriminative pattern is suddenly shifted from the beginning to the end of the time series.

As shown in Figure 8a, top-middle panel, the RL-based approaches adapt most efectively to this sudden change. Both DQeND and EARLIEST rapidly shift from early to late predictions, successfully adjusting their triggering policies to the new pattern location. RLbased methods also best preserve performance on the original concept (Figure 8a, bottom), illustrating their ability to balance adaptation and retention. In contrast, ELECTS struggles to substantially delay its decisions at deployment, with trigger points rarely extend ing beyond half of the sequence. This limited shift highlights a lack of plasticity, as the model remains biased toward its initial regime and fails to fully adapt to the new optimal decision timing.

![](images/385abd4590c720e47976506de253f6b80652008b2f476ca7426a2e5804bbc828.jpg)  
Figure 7: Hold-out AvgCost drift: detailed performances. The figures display decomposition of the hold-out AvgCost drift into both earliness, i.e. the average proportion of the time series used, and error rate, during the deployment phase.

For the Alert-based separable baselines, the importance of online adaptation becomes even more evident. Freezing the trigger model can lead to catastrophic failures, as the decision policy becomes completely misaligned with the shifted distribution. Updating the trigger substantially improves stability and eventually recovers appropriate triggering behavior, although adaptation remains slower than with end-to-end approaches.

Sub-classes shufling (A\_class). This final experiment considers a highly non-stationary setting where the composition of metaclasses is randomly reshufled at irregular intervals, directly altering the underlying class structure rather than the time series distribu tion.

In Figure 8b, ELECTS achieves the best overall performance, consistently recovering after each drift event. The RL-based approaches, DQeND and EARLIEST, closely follow and remain competitive. Noticeably, RL-based approaches sufer from a stronger constraint: unlike ELECTS and separable baselines, they only observe the data up to the estimated trigger time rather than the full time series. This makes adaptation to abrupt-type drifts substantially more challenging, a strong exploration rate being thus critical to eficiently recover. A more detailed analysis via Figure 9, bottom panel, nevertheless reveals a limitation of DQeND in this regime. Under repeated abrupt drifts, its triggering policy progressively collapses toward systematically early predictions, represented by the light blue earliness curve, which keeps decreasing. This highlights the instability of the fully RL-based pipeline, which likely results from the accumulation of policy updates across successive distribution shifts. In contrast, ELECTS and EARLIEST display more stable earliness profiles (almost flat earliness curve) throughout deployment, reflecting a more conservative adaptation strategy.

Within this setting, separable baselines achieve performance comparable to the RL-based approaches, although being retrained at fixed, predetermined intervals. Interestingly, updating the trigger model brings only limited improvements over keeping it frozen.

![](images/885ff56ff7f69b48fb7fb7a356a9530e7e6c79ea9290685f4f900504d13078aa.jpg)  
(a) A\_loc1

![](images/62b967499283709471c26f4d29b1575d40a17aff67b970b989bab7ffc1eab269.jpg)  
(b) A\_class1  
Figure 8: Abrupt drifts, deployment’s metrics, averaged over five seeds. From top to bottom: the raw cumulative cost; the hold-out AvgCost, obtained on unseen drifted data; the holdout AvgCost original, obtained using the fixed test set from training phase. UPPERCASE approaches refers to end-to-end methods while lowercase indicates separable ones.

![](images/1e35519c2d01faaebefe84dacf360ab20958e8b2985e51ce1b0b282f2bfcc6bd.jpg)  
Figure 9: Hold-out AvgCost drift: detailed performances. The figures display decomposition of the hold-out AvgCost drift into both earliness, i.e. the average proportion of the time series used, and error rate, during the deployment phase.

## 6.3 Ablation studies

To further assess the benefits of the proposed end-to-end design, we conduct an ablation study in which the encoder and decision modules of DQeND are alternatively frozen during adaptation. This allows isolating the respective contributions of representation learning and decision-making in the presented non-stationary settings.

The results, reported in Table 2, show that jointly updating both modules consistently yields the best performance across most scenarios. In contrast, partially frozen variants lead to noticeable degradations, confirming the importance of coordinated adaptation. Among the two, freezing the decision module results in the most significant performance drop. This observation is further supported by the fact that the decision-making module accounts for most of the learnable parameters in the proposed architecture.

Interestingly, freezing the encoder has a more limited impact in the abrupt drift settings considered in our experiments. This suggests that a well-trained representation can remain suficiently informative, provided that the decision module is allowed to adapt. Such behavior is intuitive in the A\_class scenario, where the input distribution itself remains unchanged, despite shifts in class composition. In the A\_loc setting, although the temporal position of the discriminative pattern changes, the nature and scale of the signal remain consistent, and the drift range remains limited, enabling the decision module to adjust its policy without requiring an update of the underlying representation.

Table 2: (a): The mean percentage diference between the cumulative cost of full updated DQeND and the partially frozen version, over five random seeds. (b): The mean of the median percentage diference of the hold-out AvgCost drift of the full updated DQeND and partially frozen version, over five random seeds. Negative figures indicate degraded performance compared to original DQeND.
<table><tr><td></td><td colspan="2">freeze encoder</td><td colspan="2">freeze decision</td></tr><tr><td>I_loc</td><td>(a) -32.42%</td><td>(b) -36.44%</td><td>(a) -36.61%</td><td>(b) -31.87%</td></tr><tr><td>I_noise</td><td>-11.29%</td><td>-4.04%</td><td>-19.28%</td><td>-6.63%</td></tr><tr><td>A_loc</td><td>1.38%</td><td>-0.34%</td><td>-43.52%</td><td>-45.30%</td></tr><tr><td>A_class</td><td>1.32%</td><td>2.14%</td><td>-18.18%</td><td>-4.39%</td></tr></table>

## 7 Conclusion

In this paper, we investigated the relevance of end-to-end learning for Early Classification of Time Series under evolving time series distributions. To address this question, we conducted extensive experiments within a controlled synthetic drift framework, systematically comparing end-to-end approaches against the widely used separable paradigm. To this end, we proposed DQeND, a Reinforcement Learning–based end-to-end ECTS architecture designed to remain closely comparable to state-of-the-art separable baselines. Across a broad range of gradual and abrupt non-stationarities, endto-end approaches consistently demonstrated stronger robustness and adaptation capabilities than separable competitors, whose behavior often remained unstable and limited under drift. In particular, DQeND achieved strong online performance while preserving adapted behaviors on the original training concept, highlighting its ability to balance adaptation and retention in dynamic environments. More broadly, our results emphasize the benefits of jointly optimizing representation learning and triggering decisions in the context of ECTS.

Future work could include a more rigorous theoretical analysis of the stability–plasticity trade-of in ECTS, as well as the study of additional forms of non-stationarity, such as drifting cost functions or evolving class priors. Finally, beyond the controlled experimental setting considered in this work, the proposed framework provides a practical foundation for deployment in more complex real-world streaming environments.

## A Variability report

In this section, the evaluation metrics displayed in Section 6 are detailed in Tables 3, 4, 5 and variability is reported.

Table 3: The mean cumulative cost and standard deviation, over five random seeds. Bold value indicates the best average performing method.
<table><tr><td></td><td>I_loc</td><td>I_noise</td><td>A_loc</td><td>A_class</td></tr><tr><td>Alert(online/freeze)</td><td> $\overline { { 2 9 . 0 2 \pm 4 . 0 8 } }$ </td><td> $4 6 . 1 4 \pm 0 . 3$ </td><td> $\overline { { 3 1 . 0 6 \pm 1 1 } }$ </td><td> $\overline { { 3 2 . 1 5 \pm 1 . 2 7 } }$ </td></tr><tr><td>Alert(online/online)</td><td> $3 1 . 8 7 \pm 5 . 4 9$ </td><td> $3 7 . 9 2 \pm 0 . 8 3$ </td><td> $2 3 . 3 2 \pm 1 . 8 8$ </td><td> $3 0 . 8 6 \pm 0 . 2$ </td></tr><tr><td>DQeND</td><td> $1 6 . 1 8 \pm 3 . 4 0$ </td><td> $3 6 . 2 \pm 1 . 0 4$ </td><td> ${ \bf 1 8 . 8 6 \pm 0 . 5 5 }$ </td><td> $3 2 . 9 8 \pm 3 . 8 8$ </td></tr><tr><td>EARLIEST</td><td> $2 0 . 5 3 \pm 6 . 5 4$ </td><td> $4 3 \pm 3 . 3 7$ </td><td> $1 9 . 9 0 \pm 0 . 7 9$ </td><td> $3 2 . 4 4 \pm 5 . 8 8$ </td></tr><tr><td>ELECTS</td><td> $1 7 . 3 3 \pm 1 . 8 1$ </td><td> $3 8 . 6 3 \pm 1 . 9 8 $ </td><td> $2 9 . 9 8 \pm 1 2 . 0 6$ </td><td> ${ \bf 2 5 . 7 9 \pm 3 . 5 2 }$ </td></tr></table>

Table 4: The mean hold-out AvgCost drift and standard deviation, over all the deployment steps and five random seeds. Bold value indicates the best average performing method.
<table><tr><td></td><td>I_loc</td><td>I_noise</td><td>A_loc</td><td>A_class</td></tr><tr><td>Alert(online/freeze)</td><td> $\overline { { 0 . 3 \pm 0 . 0 4 } }$ </td><td> $\overline { { 0 . 4 6 \pm 0 . 0 0 3 } }$ </td><td> $\overline { { 0 . 3 1 \pm 0 . 1 1 } }$ </td><td> $\overline { { 0 . 3 2 \pm 0 . 0 1 } }$ </td></tr><tr><td>Alert(online/online)</td><td> $0 . 3 1 \pm 0 . 0 7$ </td><td> $0 . 3 8 \pm 0 . 0 0 9$ </td><td> $0 . 2 3 \pm 0 . 0 2$ </td><td> $0 . 3 1 \pm 0 . 0 0 2$ </td></tr><tr><td>DQeND</td><td> $\mathbf { 0 . 1 6 \pm 0 . 0 3 }$ </td><td> $\mathbf { 0 . 3 6 \pm 0 . 0 1 }$ </td><td> ${ \bf 0 . 1 9 \pm 0 . 0 1 }$ </td><td> $0 . 3 3 \pm 0 . 0 4$ </td></tr><tr><td>EARLIEST</td><td> $0 . 2 \pm 0 . 0 6$ </td><td> $0 . 4 3 \pm 0 . 0 3$ </td><td> $0 . 2 \pm 0 . 0 1$ </td><td> $0 . 3 2 \pm 0 . 0 6$ </td></tr><tr><td>ELECTS</td><td> $0 . 1 7 \pm 0 . 0 2$ </td><td> $0 . 3 9 \pm 0 . 0 2$ </td><td> $0 . 3 \pm 0 . 1 2$ </td><td> ${ \bf 0 . 2 6 \pm 0 . 0 4 }$ </td></tr></table>

Table 5: The mean hold-out AvgCost original and standard deviation, over all the deployment steps and five random seeds. Bold value indicates the best average performing method.
<table><tr><td></td><td>I_loc</td><td>I_noise</td><td>A_loc</td><td>A_class</td></tr><tr><td>Alert(online/freeze)</td><td> $\overline { { 0 . 3 1 \pm 0 . 0 2 } }$ </td><td> $\overline { { 0 . 3 1 \pm 0 . 0 4 } }$ </td><td> $\overline { { 0 . 2 1 \pm 0 . 0 4 } }$ </td><td> $\overline { { { \bf 0 . 3 7 \pm 0 . 0 3 } } }$ </td></tr><tr><td>Alert(online/online)</td><td> $0 . 3 3 \pm 0 . 0 6$ </td><td> $0 . 3 3 \pm 0 . 0 2$ </td><td> $0 . 1 8 \pm 0 . 0 2$ </td><td> $0 . 3 7 \pm 0 . 0 0 3$ </td></tr><tr><td>DQeND</td><td> $\mathbf { 0 . 0 9 \pm 0 . 0 4 }$ </td><td> $\mathbf { 0 . 3 1 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 1 1 \pm 0 . 0 2 }$ </td><td> $0 . 4 \pm 0 . 0 4$ </td></tr><tr><td>EARLIEST</td><td> $0 . 2 7 \pm 0 . 0 7$ </td><td> $0 . 3 6 \pm 0 . 0 1$ </td><td> $0 . 1 1 \pm 0 . 0 7$ </td><td> $0 . 5 1 \pm 0 . 0 1$ </td></tr><tr><td>ELECTS</td><td> $0 . 1 4 \pm 0 . 0 7$ </td><td> $0 . 3 3 \pm 0 . 0 1$ </td><td> $0 . 1 9 \pm 0 . 1$ </td><td> $0 . 4 8 \pm 0 . 0 2$ </td></tr></table>

## B Runtime analysis

Computation times are reported in Table 6. Overall, ELECTS and EARLIEST are the fastest approaches, both during training and deployment. Unlike Alert and DQeND, which rely on convolutionbased encoders, these methods are LSTM-based and therefore benefit from the CPU-only experimental setup used in this work. In practice, convolutional approaches would likely reduce this gap when leveraging GPU acceleration. For Alert-based methods indeed, most of the training time is spent fitting the ELM-based classifier rather than the trigger module itself, confirming the relatively low computational overhead of the RL trigger. In contrast, DQeND exhibits the highest training and update times.

More broadly, the deployment times of DQeND remain signif icantly longer. This is consistent with the known sample ineficiency of value-based RL algorithms, whose repeated updates and replay-based optimization can constitute computational limitations in online settings.

Table 6: Computation times (in s.). The deployment are displayed on a per batch basis. The total time accounts for the all 100 deployment steps. The fit time of the Alert-based methods are decomposed over (classifier fit time, trigger fit time).

<table><tr><td></td><td>Training Fit</td><td>Deployment (per batch) Inference Update</td><td>Total ↓</td></tr><tr><td>ELECTS</td><td>31.89</td><td>0.026 0.05</td><td>39.22</td></tr><tr><td>EARLIEST</td><td>119.25</td><td>0.23 0.09</td><td>138.98</td></tr><tr><td>Alert (online/online)</td><td>(78.02, 46.55)</td><td>2.39 2.45</td><td>612.75</td></tr><tr><td>Alert (retrain/retrain)</td><td>(78.02, 46.55)</td><td>2.71 3.59</td><td>759.86</td></tr><tr><td>DQeND</td><td>248.74</td><td>2.90 5.75</td><td>1124.31</td></tr></table>

## References

[1] Youssef Achenchabe, Alexis Bondu, Antoine Cornuéjols, and Asma Dachraoui. 2021. Early classification of time series: Cost-based optimization criterion and algorithms. Machine Learning 110, 6 (2021), 1481–1504.

[2] Charilaos Akasiadis, Evgenios Kladis, Petro-Foti Kamberi, Evangelos Miche lioudakis, Elias Alevizos, and Alexander Artikis. 2024. A Framework to Evaluate Early Time-Series Classification Algorithms.. In EDBT. 623–635.

[3] Jakub Michal Bilski and Agnieszka Jastrzebska. 2023. CALIMERA: A new early time series classification method. Information Processing & Management 60, 5 (2023), 103465.

[4] Asma Dachraoui, Alexis Bondu, and Antoine Cornuéjols. 2015. Early classification of time series as a non myopic sequential decision making problem. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2015, Porto, Portugal, September 7-11, 2015, Proceedings, Part I 15. Springer, 433–447.

[5] Peter Dayan and CJCH Watkins. 1992. Q-learning. Machine learning 8, 3 (1992), 279–292.

[6] Angus Dempster, François Petitjean, and Geofrey I Webb. 2019. ROCKET: excep tionally fast and accurate time series classification using random convolutional kernels. arXiv preprint arXiv:1910.13051 (2019).

[7] Angus Dempster, Daniel F Schmidt, and Geofrey I Webb. 2021. Minirocket: A very fast (almost) deterministic transform for time series classification. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining. 248–257.

[8] João Gama, Raquel Sebastião, and Pedro Pereira Rodrigues. 2009. Issues in evaluation of stream learning algorithms. In SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’09). Association for Computing Machinery, 329–338.

[9] Sam Greydanus and Dmitry Kobak. 2020. Scaling down deep learning with mnist-1d. arXiv preprint arXiv:2011.14439 (2020).

[10] Ashish Gupta, Hari Prabhat Gupta, Bhaskar Biswas, and Tanima Dutta. 2020. Approaches and applications of early classification of time series: A review. IEEE Transactions on Artificial Intelligence 1, 1 (2020), 47–61.

[11] Thomas Hartvigsen, Cansu Sen, Xiangnan Kong, and Elke Rundensteiner. 2019. Adaptive-halting policy network for early classification. In Proceedings ofthe 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 101–110.

[12] Thomas Hartvigsen, Cansu Sen, Xiangnan Kong, and Elke Rundensteiner. 2020. Recurrent halting chain for early multi-label classification. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1382–1392.

[13] Guang-Bin Huang, Qin-Yu Zhu, and Chee-Kheong Siew. 2006. Extreme learning machine: theory and applications. Neurocomputing 70, 1-3 (2006), 489–501.

[14] Dongwan Kim and Bohyung Han. 2023. On the stability-plasticity dilemma of class-incremental learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 20196–20204.

[15] Huaping Liu, Lianzhi Yu, Wen Wang, and Fuchun Sun. 2016. Extreme learning machine for time sequence classification. Neurocomputing 174 (2016), 322–330.

[16] Junwei Lv, Yuqi Chu, Jun Hu, Peipei Li, and Xuegang Hu. 2023. Second-order Confidence Network for Early Classification of Time Series. ACM Transactions on Intelligent Systems and Technology (2023).

[17] Coralie Martinez, Guillaume Perrin, Emmanuel Ramasso, and Michèle Rombaut. 2018. A deep reinforcement learning approach for early classification of time series. In 2018 26th European Signal Processing Conference (EUSIPCO). IEEE, 2030– 2034.

[18] Coralie Martinez, Emmanuel Ramasso, Guillaume Perrin, and Michèle Rombaut. 2020. Adaptive early classification of temporal sequences using deep reinforcement learning. Knowledge-Based Systems 190 (2020), 105290.

[19] Usue Mori, Alexander Mendiburu, Eamonn Keogh, and Jose A Lozano. 2017. Reliable early classification of time series based on discriminating the classes over time. Data mining and knowledge discovery 31 (2017), 233–263.

[20] Jin-Man Park and Jong-Hwan Kim. 2017. Online recurrent extreme learning machine and its application to time-series prediction. In 2017 International Joint Conference on Neural Networks (IJCNN). IEEE, 1983–1990.

[21] Aurélien Renault, Alexis Bondu, Antoine Cornuéjols, and Vincent Lemaire. [n. d.]. Early Classification of Time Series: A Survey and Benchmark. Transactions on Machine Learning Research ([n. d.]).

[22] Aurélien Renault, Alexis Bondu, Antoine Cornuéjols, and Vincent Lemaire. 2025. Deep Reinforcement Learning based Triggering Function for Early Classifiers of Time Series. arXiv preprint arXiv:2502.06584 (2025).

[23] Aurélien Renault, Alexis Bondu, Antoine Cornuéjols, and Vincent Lemaire. 2026. Early Classification ofTime Series in Non-Stationary Cost Regimes. arXiv preprint arXiv:2602.00918 (2026).

[24] Marc Rußwurm, Nicolas Courty, Rémi Emonet, Sébastien Lefèvre, Devis Tuia, and Romain Tavenard. 2023. End-to-end learned early classification of time series for in-season crop type mapping. ISPRS Journal ofPhotogrammetry and Remote Sensing 196 (2023), 445–456.

[25] Miquel Serra-Perello and Alberto Ortiz. 2025. Incremental Learning Methodologies for Addressing Catastrophic Forgetting: Analysis and Experimental Evaluation. Journal ofArtificial Intelligence Research 83 (2025).

[26] Richard S Sutton and Andrew G Barto. 2018. Reinforcement learning: An introduction. MIT press.

[27] Jian Wang, Siyuan Lu, Shui-Hua Wang, and Yu-Dong Zhang. 2022. A review on extreme learning machine. Multimedia Tools and Applications 81, 29 (2022), 41611–41660.

[28] Zhengzheng Xing, Jian Pei, and S Yu Philip. 2009. Early Prediction on Time Series: A Nearest Neighbor Approach.. In IJCAI. Citeseer, 1297–1302.

[29] Zhengzheng Xing, Jian Pei, Philip S Yu, and Ke Wang. 2011. Extracting interpretable features for early classification on time series. In Proceedings ofthe 2011 SIAM international conference on data mining. SIAM, 247–258.