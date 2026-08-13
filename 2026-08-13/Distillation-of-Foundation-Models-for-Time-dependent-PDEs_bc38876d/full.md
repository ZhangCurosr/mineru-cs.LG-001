# Distillation of Foundation Models for Time-dependent PDEs

Daniel Musekamp<sup>1∗</sup> Boshra Ariguib<sup>1,</sup> <sup>2</sup> Andrei Manolache<sup>1,3</sup> Mathias Niepert<sup>1</sup>

<sup>1</sup>University of Stuttgart <sup>2</sup>University of Cologne <sup>3</sup>Bitdefender

## Abstract

Foundation models for time-dependent partial differential equations (PDEs) are trained on large and diverse collections of physical systems and can generalize effectively to new downstream tasks. After fine-tuning on only a few trajectories from a target domain, they can achieve strong accuracy in low-data regimes. However, these models are typically large and computationally intensive, limiting their usefulness as fast surrogates for numerical solvers. We propose Teacher Rollout Extension (TREX), a knowledge distillation framework that transfers the predictive capability of a pretrained foundation model into a compact and efficient student. Starting from a fine-tuned teacher, TREX augments limited downstream data by generating long synthetic trajectories through teacher rollouts, optionally with periodic noise injection. This procedure samples from the teacher-induced rollout distribution without requiring explicit knowledge of the initial-condition distribution, while exposing the student to long-horizon states and local recovery behavior around states encountered during autoregressive prediction. The student can further incorporate task-specific inductive biases, such as equivariance, that the teacher does not necessarily enforce. We evaluate TREX on multiple PDE benchmarks. The resulting students can match or surpass the teacher’s accuracy while reducing the number of parameters by several orders of magnitude and achieving more than an order-of-magnitude speedup in inference.

## 1 Introduction

Spatiotemporal physical processes such as fluid dynamics or cell growth can be described using partial differential equations (PDEs). Hence, solving PDEs is integral to science and engineering, for example, in astrophysics, design optimization, or weather prediction. Since solving PDEs for large systems and nonlinear dynamics using numerical solvers can be slow, recent research has investigated applying neural networks as surrogate models to learn solutions to PDEs. Compared with classical solvers, such approaches are differentiable out of the box and can incorporate experimental data, thereby enabling the modeling of effects not included in the simulation model. One core problem with neural PDE surrogates is that they require a training dataset, which is often generated using an expensive numerical solver, thereby making the surrogate modeling potentially uneconomical.

As a remedy, foundation models for PDEs have been introduced (McCabe et al., 2024; Herde et al., 2024; McCabe et al., 2025). By pretraining a surrogate on a large set of diverse physics, foundation models can transfer knowledge to downstream tasks with fewer training samples. On the downside, their inference is computationally expensive, as they require more parameters to generalize well. Slow inference prohibits their use in applications that have real-time requirements, such as planning and control tasks in manufacturing or energy systems (Tian et al., 2025; Janssens & Meyers, 2024). Furthermore, the used architectures are mostly large transformers that lack inductive biases, since these may vary from problem to problem. For example, many PDEs are locally equivariant to rotations and translations, and globally as well if the boundary conditions are periodic. Including these equivariances in the architecture can make models more effective and data-efficient (Zhdanov et al., 2024; Helwig et al., 2023).

![](images/7bcfd13d97179c35e08b5474bb97bb65b1bf86c39f9b3f6f0c213c7fd085a3cc.jpg)  
Figure 1: Foundation models are pretrained on a variety of PDE prediction tasks first (1), enabling them to adapt to a new task with only a few trajectories (2). To overcome their high inference time, we propose to distill the foundation model into a smaller and potentially equivariant student by extending the training trajectories using noisy rollouts of the teacher (3).

Therefore, we propose using knowledge distillation (KD) to reconcile the trade-off between fast inference and strong generalization (Fig. 1). Specifically, the pretrained foundation model is first fine-tuned on a small set of trajectories in the target domain. Afterward, the foundation model is used as the teacher to amend the available data with additional information. To our knowledge, we present the first systematic study of distilling general pretrained PDE foundation models into compact downstream surrogate models. Most prior work relies on specialized architectures or assumes ful knowledge of the initial state distribution (Zhang et al., 2025; Li et al., 2025). However, access to the downstream initial-condition distribution is not always available. This is particularly relevant when data originates from experiments or partially observed systems, where only a limited set of trajectories is given.

Hence, we introduce Teacher Rollout Extension (TREX), a distillation technique that avoids the need to sample new initial conditions. Starting from the few available downstream trajectories, TREX generates long teacher rollouts and trains the student on the resulting teacher-labeled trajectories. This exposes the student to the teacher-induced rollout distribution, i.e., the distribution of states encountered during autoregressive prediction, rather than only the short trajectories observed in the downstream data. Periodic noise injection can further broaden local state-space coverage and teach the student how the teacher evolves from perturbed states. To transfer the teacher’s knowledge to the student model, we add these teacher-generated trajectories to the student dataset and train on a mixture of ground-truth and teacher-labeled transitions.

To overcome the aforementioned lack of inductive biases in foundation models, we propose distilling the teacher into a student with task-appropriate inductive biases. For example, we use a tensorized FNO (Kossaifi et al., 2024) as the student. In addition to being equivariant to input translations, FNOs also exhibit inductive biases such as excluding high-frequency modes that are prone to noise.

We investigate the effectiveness of our method through experiments with the Poseidon (Herde et al., 2024) and Walrus (McCabe et al., 2025) foundation models across a range of fluid dynamics tasks, demonstrating that TREX can achieve teacher accuracy while drastically reducing parameter count and inference time. Overall, our work provides the following contributions:

1. To our knowledge, we present the first systematic study of distilling general pretrained PDE foundation models into compact downstream surrogate models.

2. We introduce TREX, a novel distillation method that generates supervision data using noisy teacher rollouts, in some cases surpassing the teacher’s accuracy.

3. We show that by distilling the foundation model into a small, lightweight student architecture with suitable inductive biases, we can reduce inference time by more than an order of magnitude while respecting physical properties such as equivariances.

## 2 Problem Definition

We consider time-dependent PDEs on a spatial domain X and time interval $[ 0 , T ]$ , with solution $\mathbf { u } : [ 0 , T ] \times \mathcal { X }  \mathbb { R } ^ { \hat { N _ { c } } }$

$$
\partial _ { t } \mathbf { u } = F \left( t , \mathbf { x } , \mathbf { u } , \partial _ { \mathbf { x } } \mathbf { u } , \partial _ { \mathbf { x } \times } \mathbf { u } , \ldots \right) , \qquad \left( t , \mathbf { x } \right) \in [ 0 , T ] \times \mathcal { X } ,\tag{1}
$$

$$
\mathbf { u } ( 0 , \mathbf { x } ) = \mathbf { u } ^ { 0 } ( \mathbf { x } ) , \qquad \mathbf { x } \in \mathcal { X } ,\tag{2}
$$

$$
\mathcal { B } [ \mathbf { u } ] ( t , \mathbf { x } ) = 0 , \qquad &  ( t , \mathbf { x } ) \in [ 0 , T ] \times \partial \mathcal { X } .\tag{3}
$$

Here, F defines the dynamics, $\mathbf { u } ^ { 0 }$ is the initial condition, and $\boldsymbol { B }$ specifies the boundary condition. A task corresponds to solving this initial-value problem for ${ \bf u } ^ { 0 } \sim p _ { 0 }$ . For learning, we use discretized solution trajectories $( \mathbf { u } ^ { 0 } , \mathbf { \breve { u } } ^ { 1 } , \dots , \mathbf { u } ^ { N _ { T } - 1 } )$ , where each state $\mathbf { u } ^ { i } \in \mathbb { R } ^ { N _ { x } \times N _ { c } }$ contains the solution on $N _ { x }$ spatial grid points and $N _ { c }$ channels. A neural PDE surrogate then learns a discrete time-stepper $\mathcal { G }$ and is rolled out autoregressively $\mathbf { u } ^ { i + 1 } \approx \mathcal { G } ( \mathbf { u } ^ { i } )$ . For models that require multiple context frames, we treat the complete context window as the autoregressive state. Hence, each transition shifts thi context by one frame and appends the newly predicted state.

## 3 Related Work

Knowledge distillation (Hinton et al., 2015; Gou et al., 2021) aims to provide the student with additional information from the teacher network. For classification tasks, the class logits can be transferred to the student (Ba & Caruana, 2014; Hinton et al., 2015). For general tasks, the student can be trained to have feature representations close to those of the teacher (Romero et al., 2015). Relational KD (Park et al., 2019) trains the student by comparing the relationship between the student’s feature representations of different inputs and the teacher’s, i.e., pushing the student to have the same distance between inputs as the teacher. Thus, the teacher and student do not need the same feature space representation. Similarly, Lin et al. (2025) introduce a region-aware attention mechanism that reconstructs student patches into representations more closely aligned with the teacher’s view, even when the two models have substantially different architectures. Lastly, the student can also be trained to match the teacher’s derivatives. Srinivas & Fleuret (2018) train a student by matching the teacher’s Jacobian matrix. Amin et al. (2025) apply a similar approach to distill a foundation model for molecules into a smaller student model by matching the teacher’s energy-prediction Hessian. In addition to providing additional information about the encountered training examples, it is also possible to artificially generate new training instances using the teacher model. For example, Lopes et al. (2017) introduce data-free distillation, which aims to generate new samples that match the statistics of a dataset for which only those statistics are known.

Distillation has also been applied to PDEs or spatiotemporal predictions in general. Li et al. (2025) distill a special teacher architecture with branches for low- and high-frequency features into the student using separate losses for each branch. Wan et al. (2025) propose SINO, which first learns a fine-step surrogate from a small number of trajectories and subsequently distills it into a coarse-step FNO using a larger teacher-generated dataset. The synthetic trajectories are generated from newly sampled initial conditions, whereas TREX targets settings in which no such initial-condition sampler is available and instead expands the available downstream trajectories through teacher rollouts. Zhang et al. (2025) present Omnifluids. Omnifluids first trains a high-resolution model on a PDE with varying parameters using a finite-difference loss for a known equation, similar to PINO (Li et al., 2024). The teacher is distilled into a lower-resolution model with the same architecture by matching the teacher’s predictions on samples from the $\mathrm { I C } ,$ , assuming the teacher’s parameter distribution is known. In contrast, we consider general foundation models pretrained on large datasets, and do not assume knowledge of the PDE or input distributions.

PDE Foundation Models. In recent years, a number of PDE foundation models have been presented. Most use large vision transformers as their backbone (McCabe et al., 2024; Liu et al., 2024; Herde et al., 2024; McCabe et al., 2025). The earliest example of a PDE foundation model for nonlinear problems is MPP (McCabe et al., 2024), which also uses axial attention to reduce the computational load. Since then, model and dataset sizes have been increasing. Poseidon (Herde et al., 2024) is based on a SwinV2 (Liu et al., 2022a) vision transformer trained to predict the future state with the step size as a model input. Hence, it can be employed both for direct and autoregressive prediction. Walrus (McCabe et al., 2025) is trained on The Well data (Ohana et al., 2024) and can solve 2D and 3D tasks. While using a vision transformer, it also takes multiple time steps as inputs to infer the underlying physics from the observed trajectory segment. Zhou et al. (2025) present Unisolver, which provides the vision transformer backbone with additional information about the PDE, such as the domain geometry or the physical parameters. Most notably, the equation symbols are embedded using an LLM. In contrast, PDEformer-2 (Ye et al., 2025) uses a graph-based transformer, and DPOT (Hao et al., 2024) is based on Fourier attention layers. Shen et al. (2024) propose to use a pretrained LLM backbone to propagate in time. Additionally, they also embed additional inputs using an LLM similar to Unisolver. For more related work, see Sec. D.

## 4 Knowledge Distillation of PDE Foundation Models

We are given a small training set of trajectories u from a downstream target domain

$$
\begin{array} { r } { \mathcal { D } _ { \mathrm { G T } } = \{ ( \mathbf { u } _ { n } ^ { 0 } , . . . , \mathbf { u } _ { n } ^ { N _ { T } - 1 } ) \} _ { n = 1 } ^ { N } , } \end{array}\tag{4}
$$

where each trajectory consists of $N _ { T }$ states. Our goal is to build an accurate and efficient surrogate model by infusing knowledge from a foundation model (FM) used as the teacher $\tau$ , that autoregressively predicts the next state $\mathbf { z } ^ { t + 1 } = \mathcal { T } ( \mathbf { z } ^ { t } )$ . Since the FM was pretrained on a large set of PDE problems, it can provide a lower error on the downstream task than a model trained from scratch. In the first step, the FM is fine-tuned on the available downstream task data. Afterward, we distill the teacher’s knowledge into the student model S. A natural distillation strategy is to sample new initial conditions u<sub>0</sub> ∼ p<sub>0</sub>, roll out the teacher, and train the student on the resulting teacher-labeled pairs (Zhang et al., 2025). However, in downstream applications, the initial condition distribution $p _ { 0 }$ may be unknown, for example, for data coming from experimental sources. In other cases, it may be difficult to sample directly from this distribution. Lastly, for FMs that require multiple context frames, sampling an initial condition alone is insufficient to initialize the teacher, since generating a valid context requires additional simulator calls.

## 4.1 Knowledge Distillation using Teacher Rollout Extension

We propose Teacher Rollout Extension (TREX), a distillation procedure that trains the student not only on the available ground truth downstream trajectories but also on the state distribution induced by the fine-tuned teacher during autoregressive prediction. This distinction is important for PDE surrogates: at test time, the model is repeatedly applied to its own previous predictions, and therefore encounters states that need not lie exactly on the finite set of observed trajectories.

TREX targets this autoregressive state distribution directly. It generates additional training trajectories by rolling out the teacher from the available ground truth states. For simplicity, we state the following occupancy measure for the single-frame case used in our main Poseidon experiments. For multi-frame models, the same construction applies by replacing each state with its corresponding context window. Let $\delta _ { \mathbf { u } _ { n } ^ { 0 } }$ be the Dirac measure concentrated at ${ \bf u } _ { n } ^ { 0 }$ , and let

$$
\nu _ { 0 } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \delta _ { \mathbf { u } _ { n } ^ { 0 } }\tag{5}
$$

be the empirical distribution over the available downstream initial states. A standard one-step training objective only supervises the model on transitions drawn from the short observed ground truth trajectories. In contrast, TREX constructs an extended teacher-induced state occupancy measure

$$
\rho _ { 7 } ^ { K } = \frac { 1 } { K } \sum _ { t = 0 } ^ { K - 1 } ( T ^ { t } ) _ { \# } \nu _ { 0 } ,
$$

where $( \mathcal T ^ { t } ) _ { \# } \nu _ { 0 }$ denotes the pushforward of $\nu _ { 0 }$ under t autoregressive teacher steps and $K$ is the rollout horizon. Hence, the occupancy measure describes the state distribution induced by rolling out the teacher from the selected set of ground-truth samples. This deterministic occupancy measure motivates TREX by describing the additional states made available through teacher rollout extension. In practice, TREX further perturbs this rollout distribution through periodic noise injection, as described below.

![](images/64ea7673b90ff4cf14ab04a82d36fa8e570d0db7812044a041845c0e307baa09.jpg)  
Figure 2: Long rollouts can increase state-space coverage in dynamical systems. Two trajectories of the Lorenz system (Lorenz, 1963) visit similar regions of state space over time (a). Starting from a short observed trajectory (b), extending the rollout generates additional states from the long-time trajectory distribution (c).

Thus, TREX does not require access to the true initial-condition distribution. Instead, it reuses the few ground-truth trajectories as anchors and expands them through the dynamics learned by the foundation model. This perspective connects TREX to the long-time behavior of dynamical systems. Many dissipative PDEs and chaotic systems possess recurrent long-time structure, such as attractors or invariant measures, so that trajectories initialized from different states eventually visit similar regions of the state space. If the teacher dynamics admit an invariant measure $\mu _ { T }$ satisfying $\tau _ { \# } \mu _ { T } = \mu _ { T }$ , then, under additional ergodicity or mixing assumptions, sufficiently long teacher rollouts may approach this long-time distribution. More generally, rollout extension increases coverage of states reachable under the teacher dynamics from the available downstream trajectories. The ergodic Lorenz attractor example in Fig. 2 illustrates this effect: although trajectories start from different initial conditions, their long rollouts repeatedly visit a shared region of the state space. TREX exploits the same principle for PDE foundation models by turning a small number of ground-truth trajectories into a much broader set of teacher-labeled states.

## 4.2 Noisy Teacher Rollouts

Deterministic rollouts sample states on the teacher trajectory, but may provide limited coverage of nearby off-trajectory states. Additionally, it can be beneficial to include data points near typical rollout states in order to teach the student how the dynamics evolve from perturbed inputs. Hence, when appropriate, we inject noise periodically during the teacher rollout:

$$
\tilde { \mathbf { z } } ^ { t } = \mathbf { z } ^ { t } + m _ { t } \boldsymbol { \xi } ^ { t } , \qquad \mathbf { z } ^ { t + 1 } = \mathcal { T } \left( \tilde { \mathbf { z } } ^ { t } \right) , \qquad m _ { t } = \mathbf { 1 } \{ t > 0 \land t \equiv 0 \mod T _ { \mathrm { N o i s e } } \} ,\tag{6}
$$

$$
( { \pmb \xi } ^ { t } ) _ { c } ( { \bf x } ) \sim \mathcal { N } ( 0 , \sigma ^ { 2 } s _ { c } ^ { 2 } ) .\tag{7}
$$

Here, $\xi ^ { t }$ is Gaussian noise applied independently at each spatial location and channel, $s _ { c }$ denotes the global standard deviation of channel c computed over the dataset, and σ is a fixed scaling factor. The noise is not applied at every step, but at intervals of length $T _ { \mathrm { N o i s e } }$ , so that we obtain unperturbed subtrajectories for training and allow the teacher rollout to return toward states typical of its unperturbed rollout distribution after a perturbation. Adding noise during training is a common strategy for autoregressive neural PDE solvers (Sanchez-Gonzalez et al., 2018; Pfaff et al., 2020; Takamoto et al., 2023). In these approaches, the training inputs may be perturbed while the target trajectory remains unchanged. In TREX, by contrast, the teacher is explicitly evaluated after perturbation, so that the synthetic target reflects the teacher dynamics starting from the perturbed state.

By performing noisy rollouts from the initial conditions of the available ground-truth trajectories, we obtain the dataset $\mathcal { D } _ { \mathrm { T R E X } }$ . It consists of teacher-generated sub-trajectories between consecutive noise injections. A segment starting at a noise-injection step $t _ { i }$ is of the form

$$
\left( \tilde { \mathbf { z } } ^ { t _ { i } } , \mathbf { z } ^ { t _ { i } + 1 } , \ldots , \mathbf { z } ^ { t _ { i } + T _ { \mathrm { N o i s e } } } \right) .
$$

These segments are subsequently split into sub-trajectories of the desired student rollout length.

The final student objective combines ground-truth supervision with the teacher rollout distillation:

$$
\begin{array} { r } { \mathcal { L } _ { S } = \mathcal { L } _ { \mathrm { G T } } + \lambda _ { \mathrm { T R E X } } \mathcal { L } _ { \mathrm { T R E X } } . } \end{array}
$$

For both ground-truth and teacher-generated sub-trajectories, we train the student using a k-step autoregressive rollout loss. For TREX data,

$$
\mathcal { L } _ { \mathrm { T R E X } } = \frac { 1 } { \left| \mathcal { D } _ { \mathrm { T R E X } } \right| } \sum _ { ( \mathbf { z } ^ { i } , . . . , \mathbf { z } ^ { i + k } ) \in \mathcal { D } _ { \mathrm { T R E X } } } \frac { 1 } { k } \sum _ { t = i } ^ { i + k - 1 } \ell \left( \mathbf { z } ^ { t + 1 } , S ( \hat { \mathbf { u } } ^ { t } ) \right) , \quad \hat { \mathbf { u } } ^ { i } = \mathbf { z } ^ { i } , \quad \hat { \mathbf { u } } ^ { t + 1 } = S ( \hat { \mathbf { u } } ^ { t } ) .
$$

The training sub-trajectories $\left( \mathbf { z } ^ { i } , \ldots , \mathbf { z } ^ { i + k } \right)$ are obtained by splitting the noise-free teacher-generated segments into sequences of the desired rollout length. $\mathcal { L } _ { \mathrm { G T } }$ is defined analogously on sub-trajectories extracted from the available ground-truth trajectories. As TREX merely provides additional data points, it could be combined with other methods, such as feature-based distillation techniques, in the future.

## 4.3 Enforcing Physical Constraints

Foundation models for PDEs are usually constructed using large transformer architectures (Sec. 3). This makes them flexible enough to model many different systems, but it also means that they often do not explicitly encode the symmetries of a specific downstream PDE. Once the downstream task is fixed, however, such a structure may be known and can be built into the student. We focus on equivariance as one such constraint. Let G be a group of transformations acting on the state space, and let $\tau _ { g } u$ denote the transformed state for $g \in G$ . A model M is equivariant to G if

$$
\mathcal { M } ( \tau _ { g } u ) = \tau _ { g } \mathcal { M } ( u ) , \qquad \forall g \in G .
$$

For a PDE surrogate, this means that transforming the input state and then predicting the next state gives the same result as predicting first and then transforming the output. Many physical systems satisfy such symmetries. For example, on a periodic domain, the solution operator of a translation-invariant PDE satisfies $\Phi _ { \Delta t } ( \tau _ { a } u ) = \tau _ { a } \Phi _ { \Delta t } ( u )$ , where $\tau _ { a }$ denotes a spatial shift. This provides a complementary benefit to distillation. The teacher contributes knowledge acquired during large-scale pretraining, while the student can impose task-specific structure known for the PDE. In low-data settings, this is particularly useful, as the student does not need to infer symmetry solely from the few available trajectories. The student can therefore learn from the teacher-generated data while restricting the learned transition map to functions that better match the downstream physics. Architectural constraints can also act as regularizers. For example, a student’s architecture may suppress unresolved high-frequency components, enforce locality, preserve conservation laws, or respect boundary conditions. Such constraints can reduce noisy artifacts and improve rollout stability.

## 5 Experiments

For our main experiments, we use the Poseidon-L model (Herde et al., 2024) as the teacher. For simplicity, we always use the autoregressive evaluation of Poseidon (instead of the lead-time prediction mode). The teacher is distilled into a lightweight TFNO model (Kossaifi et al., 2024, 2025), which represents FNO’s (Li et al., 2021) weight matrix using a Tucker factorization to reduce the number of parameters but also improve generalization through regularization effects. While the teacher is trained with the original n-to-n training protocol, the student is trained autoregressively with 2-step sub-trajectories. More model and training details are listed in Sec. F. For TREX, the teacher is rolled out for 100 steps starting from the initial condition, where noise is injected after every 10 steps. The relative noise standard deviation is set to $\sigma = 1$ . As distillation baselines, we compare the original Poseidon-L teacher and the student trained only with the available ground truth data. Additionally, we add relational KD (Park et al., 2019) as a feature-based KD method, using Gaussian sketching to reduce the dimensionality to make the pairwise distances computationally feasible. Lastly, we add a baseline with full knowledge of the true initial condition distribution (IC-KD) similar to Zhang et al. (2025). We use a total batch size of 32 and adjust the number of epochs so that the total number of batch updates is equal between all methods (25k when using only ground truth). The training times are listed in the appendix (Tab. 7). TREX training is only slightly slower than pure ground truth training (e.g., increased training time from 272 to 291 min for 32 trajectories). For more details on the KD methods, see Sec. G.

![](images/bb436bff7bf1e2a3912ffaaa7a4ef3ca70acfdecf89863cdba94b1c487cef365.jpg)  
Figure 3: The median relative $L ^ { 1 }$ error for Poseidon downstream experiments. The distillation methods are compared to the teacher’s accuracy over the number of trajectories in the training set N. TREX matches or surpasses the teacher across the selected downstream settings.

Table 1: Inference time, GPU memory, and number of parameters for the Poseidon-L teacher and the TFNO student. Inference and memory are computed for a batch size of 64.
<table><tr><td rowspan="2">Model</td><td colspan="2">1-step</td><td colspan="2">Full rollout</td><td rowspan="2"># Param.</td></tr><tr><td>Time</td><td>Memory</td><td>Time</td><td>Memory</td></tr><tr><td>Poseidon-L (lead-time)</td><td>270.9 ms</td><td>3569 MB</td><td>1908.2 ms</td><td>10325 MB</td><td>628.6M</td></tr><tr><td>Poseidon-L (autoreg.)</td><td>269.4 ms</td><td>3605 MB</td><td>1887.7 ms</td><td>3749 MB</td><td>628.6M</td></tr><tr><td>TFNO (student)</td><td>22.1 ms</td><td>1059 MB</td><td>152.1 ms</td><td>1171 MB</td><td>174.4K</td></tr></table>

As downstream tasks, we use compressible Euler and incompressible Navier-Stokes equations from Herde et al. (2024). Sec. E lists the used PDEs in detail. Since our goal is specifically to study whether useful knowledge from a foundation model can be transferred into a compact student, we focus on spatiotemporal Poseidon tasks for which the fine-tuned teacher outperforms the TFNO trained only on the available downstream data, as determined on the validation set. The trajectories contain 8 frames each. We report the original median relative $L ^ { 1 }$ error from Poseidon (Herde et al., 2024), which normalizes the $\bar { L } ^ { 1 }$ loss for each quantity of interest (QOI) frame-wise:

$$
L _ { \mathrm { Q O I } } ^ { 1 } ( \mathbf { u } , \hat { \mathbf { u } } ) = \frac { \sum _ { \mathbf { x } } \sum _ { c _ { \mathrm { Q O I } } } \left| \hat { \mathbf { u } } ( t _ { \mathrm { l a s t } } , \mathbf { x } , c _ { \mathrm { Q O I } } ) - \mathbf { u } ( t _ { \mathrm { l a s t } } , \mathbf { x } , c _ { \mathrm { Q O I } } ) \right| } { \sum _ { \mathbf { x } } \sum _ { c _ { \mathrm { Q O I } } } \left| \mathbf { u } ( t _ { \mathrm { l a s t } } , \mathbf { x } , c _ { \mathrm { Q O I } } ) \right| } .
$$

For completeness, we also provide the results over all time steps in Tab. 6. For the incompressible Navier-Stokes tasks, the QOI is the velocity vector. For compressible Euler (CE-RPUI), the mean of the median of all QOI errors is reported. We report the mean across three seeds for all main results, using different training trajectories for each seed as well.

## 5.1 Main Results

Fig. 3 shows the main results for the Poseidon distillation. TREX matches or surpasses its teacher network across the selected low-data settings, which we will further investigate in the next section. IC-KD is also able to match the teacher on all investigated settings; however, it requires the full IC distribution to be known, which is not the case for experimental settings and might also be expensive (for example, for NS-BB, the IC is actually an already evolved state (Sec. E)). While relational KD can also slightly improve the student, we were not able to match the teacher in settings where the teacher outperformed ground truth learning. A reason might be the difficulty of meaningful distance measures in the large-dimensional feature fields encountered in the spatial neural PDE solver latent space. Interestingly, we found that the TFNO is an exceptionally well-performing model for the considered tasks, even capable of outperforming the pretrained Poseidon in certain settings without KD. In these cases, TREX can perform worse than only using the ground truth since the added teacher-generated data is moving the student away from the better solution it would have found using only ground truth. For the other student models considered below, we did not find similar results. Thus, distillation should only be applied if the foundation model was found to generalize well to the task at hand using a held-out validation set. Fig. 4c) shows the error growth over time for the NS-PwC task, comparing the TFNO student with Poseidon. While the student has a larger error at the beginning, it has a more stable long-term rollout and has a slower error growth. Interestingly, this behavior is seen across multiple datasets (Fig. 9). While surpassing the teacher’s accuracy, the student needs 3604 times fewer parameters (Tab. 1). The inference time is reduced by over an order of magnitude, and the memory consumption for a forward pass by a factor of 3.2 (compared to autoregressive Poseidon rollouts).

![](images/b92fe60fe10b446c6b213ba7e3a5a44bf96b2a38e3515bec0d6d55902190fe16.jpg)

![](images/fb464f1c604ce7773c376bfb746105eccd0f3765c7ff65f6590c9d01db83caff.jpg)

![](images/7e9948a3293ae05f02049c593037f9e65da4fbe761873fb530a943df57a1e6bb.jpg)  
Figure 4: Ablation study on NS-SVS (four trajectories). The effect of the rollout is relatively constant over a wide range of hyperparameters (a). Injecting noise also reduces the error, particularly at relatively high noise amplitudes (b). In (c), the error over time is shown for NS-PwC (8 trajectories). Time steps $t > 7$ show temporal extrapolation beyond the training time horizon. While the TREXtrained student has a higher one-step error than the teacher, the student has a more stable rollout.

## 5.2 Ablations

To investigate why TREX can generate a student that outperforms its teacher, we conduct an ablation study. To this end, we select the NS-PwC-4 case and fully remove the groundtruth trajectories, training the student only on teacher-generated data (Tab. 2). Even without the ground truth data, TREX still outperforms its teacher. Removing noise injections during the rollout reduces the gap, but not fully. The phenomenon that distillation can improve student performance has been observed in the general KD literature (Mobahi et al., 2020; Park et al., 2019). It can be explained by regularization effects, which can even occur when distilling the same model into the same architecture (Zhang et al., 2019; Mobahi et al., 2020).

Table 2: Ablation study on NS-PwC (four trajectories). Shown are the effects of removing the noise injection during the teacher rollout, as well as removing ground truth trajectories fully from the student training.
<table><tr><td>Variant</td><td>Median rel.  $L ^ { 1 }$ </td></tr><tr><td>TREX</td><td> $0 . 0 5 6 \pm 0 . 0 0 3$ </td></tr><tr><td>w/o GT</td><td> $0 . 0 6 0 \pm 0 . 0 0 2$ </td></tr><tr><td>w/o GT &amp; w/o Noise</td><td> $0 . 0 6 4 \pm 0 . 0 0 6$ </td></tr><tr><td>Teacher</td><td> $0 . 0 7 0 \pm 0 . 0 1 1$ </td></tr></table>

Next to the core algorithm choices, we also investigate the influence of the included hyperparameters. Fig. 4a) shows the effect of the rollout length. The algorithm is quite robust to the choice of this hyperparameter. The performance only decreases slowly for very long rollouts. The effect of the strength of the injected noise is shown in Fig. 4b). Interestingly, choosing the noise quite high at a level of one time the channel-wise standard deviation $s _ { c }$ produces the best results. Fig. 8 shows an example of an extended noisy rollout (appendix). While the state appears very noisy after the noise injection, the teacher rollout effectively reduces the noise level in subsequent states. We also provide an experiment with other student architectures. We use a modern UNet (Gupta & Brandstetter, 2023)

and a standard FNO (Li et al., 2021). As shown in Fig. 5, TREX is also able to let other student architectures match the teacher’s accuracy for all considered numbers of trajectories. While the TFNO was able to improve over the teacher using only ground truth for a higher number of trajectories in the main experiments (Fig. 3), this was not the case for the UNet or even a standard FNO, and was only possible through distillation.

![](images/375217084ca6db5ac46feb6040b67d254932d071f1fbbfbb0dc3150c9a66b9ba.jpg)  
Figure 5: Ablation on the student architecture (NS-PwC). Shown are the results for a UNet and FNO trained with TREX (full) and only ground truth data (dashed).

![](images/c2789e310869f2a320f5cb933df824835803975bef13b4e90e33238039a2451a.jpg)  
Figure 6: Example TREX rollout of the Poseidon teacher on the NS-SVS task $( v _ { x }$ channel). Each row represents a segment of the complete rollout, starting with the state to which the noise was applied.

Table 3: Distillation of Walrus (McCabe et al., 2025) on the Kolmogorov Flow dataset. TREX improves long-rollout accuracy while reducing inference time by more than an order of magnitude.
<table><tr><td></td><td> $\mathrm { V R M S E } _ { \mathrm { O n e - S t e p } }$ </td><td> $\mathrm { V R M S E } _ { T \in [ 1 : 2 0 ] }$ </td><td> $\mathrm { V R M S E } _ { T \in [ 2 1 : 6 0 ] }$ </td><td>Inference Time in s</td></tr><tr><td>Walrus</td><td>0.012</td><td>0.214</td><td>0.713</td><td>21.425</td></tr><tr><td>Only GT</td><td>0.050</td><td>0.355</td><td>0.972</td><td>0.401</td></tr><tr><td>TREX</td><td>0.016</td><td>0.123</td><td>0.6859</td><td>0.414</td></tr></table>

## 5.2.1 Teacher Model

We also experiment with other teacher models. Walrus (McCabe et al., 2025) is a larger vision transformer (1.3B parameters) that uses the last six states as context for next state prediction. This setup is crucially different since it does not allow IC-KD, as we would need to run the simulator to get the initial sequence to start the rollout. To also go beyond the rather short trajectories provided by the Poseidon datasets, we test distillation on the 2D Kolmogorov Flow dataset from Li et al. (2022) with a Reynolds number of 5000. The trajectories have 501 time steps, where the first 100 are discarded (Li et al., 2022). For the student, we select a higher capacity TFNO with 64 modes and a feature dimension of 80, trained for only 10k epochs with batch size 64. Similar to the teacher, the student is using six input steps as the context. For the ground truth only baseline, we found a single step as input to work better. We fine-tune Walrus on a single trajectory and report the VRMSE metric (Eq. (30)) for different rollout intervals. The student trained with TREX can also successfully distill the multi-step teacher foundation model (Tab. 3).

## 5.2.2 Equivariance

To investigate the effect of the explicit equivariance built into the student architecture, we test how the models behave under spatial shifts on the periodic grid. We apply random 2D rolls to the input fields, which corresponds to the translation symmetry of the underlying periodic domain. An equivariant model should produce the same prediction up to this same roll. We evaluate this in two ways: first, as a direct equivariance error, and second, through shifted autoregressive rollouts.

For the direct equivariance metric, let $a = ( d _ { x } , d _ { y } )$ denote a sampled 2D periodic translation, and let $\tau _ { a }$ be the corresponding shift operator. We compare the prediction on the shifted input with the shifted prediction on the original input, $\| f ( \tau _ { a } x ) - \tau _ { a } f ( x ) \| _ { 1 }$ , using the same median relative $L ^ { 1 }$ normalization as in the main evaluation. As shown in Tab. 4, the student has substantially lower equivariance error than the teacher across all datasets. This is expected from the student architecture, but it is still important: distillation transfers the teacher’s predictive behavior into a model class that enforces a symmetry not explicitly respected by the teacher.

Table 4: Translation equivariance results. Left: median relative $L ^ { 1 }$ equivariance error, computed using the unperturbed predictions as ground truth. Right: final-step relative $L ^ { 1 }$ errors on original and shifted OOD rollouts.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">Equivariance error</td><td colspan="4">Final-step rollout error</td></tr><tr><td>Teacher</td><td>Student</td><td>Poseidon</td><td>Poseidon shift</td><td>TFNO</td><td>TFNO shift</td></tr><tr><td>NS-BB</td><td> $0 . 0 2 6 \pm 0 . 0 1 3$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 1$ </td><td> $0 . 0 6 5 1 \pm 0 . 0 2 8 6$ </td><td> $0 . 0 6 7 6 \pm 0 . 0 2 0 3$ </td><td> $0 . 0 5 0 4 \pm 0 . 0 0 8 0$ </td><td> $0 . 0 5 0 4 \pm 0 . 0 0 8 0$ </td></tr><tr><td>CE-RPUI</td><td> $0 . 1 0 1 \pm 0 . 0 0 6$ </td><td> $0 . 0 0 1 \pm 0 . 0 0 1$ </td><td> $0 . 4 1 8 9 \pm 0 . 0 7 6 7$ </td><td> $0 . 4 1 0 8 \pm 0 . 0 8 5 6$ </td><td> $0 . 3 5 4 9 \pm 0 . 0 3 4 4$ </td><td> $0 . 3 5 4 9 \pm 0 . 0 3 4 4$ </td></tr><tr><td>NS-PwC</td><td> $0 . 0 2 3 \pm 0 . 0 0 6$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 1$ </td><td> $0 . 1 0 9 5 \pm 0 . 0 4 2 0$ </td><td> $0 . 1 0 4 1 \pm 0 . 0 3 8 5$ </td><td> $0 . 0 7 4 4 \pm 0 . 0 2 0 6$ </td><td> $0 . 0 7 4 4 \pm 0 . 0 2 0 6$ </td></tr><tr><td>NS-SVS</td><td> $0 . 1 1 0 \pm 0 . 0 0 5$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 1$ </td><td> $0 . 0 8 2 0 \pm 0 . 0 0 8 3$ </td><td> $0 . 2 3 9 8 \pm 0 . 0 0 8 6$ </td><td> $0 . 0 7 5 5 { \scriptstyle \pm 0 . 0 0 7 2 }$ </td><td> $0 . 0 7 5 5 { \scriptstyle \pm 0 . 0 0 7 2 }$ </td></tr></table>

![](images/ae393d746e9086e794f38a1510c91059c391a6532197e504867b6ae8c843ae5d.jpg)  
Figure 7: Final-step shifted-rollout comparison on NS-SVS. The teacher shows a visibly larger error with respect to the shifted ground truth (GT), while the student prediction remains much closer and the corresponding difference is substantially smaller.

We further evaluate the effect of equivariance in autoregressive rollouts for temporal extrapolation by applying the same spatial shift to the initial condition and comparing the final prediction to the shifted ground truth. The results in Tab. 4 show that the shifted rollout error of the student remains essentially unchanged, while the teacher can degrade substantially, most clearly on NS-SVS. Since NS-SVS also has the largest teacher equivariance error, we show a qualitative example in Fig. 7. In this case, the teacher produces visibly worse shifted rollouts, while the student remains consistent with the translated dynamics. We include the entire rollout trajectory in Appendix Fig. 10. These results suggest that equivariance can be a useful property of the student, especially in autoregressive simulations where small symmetry errors may accumulate over long rollouts. More broadly, it is encouraging that TREX can use a non-equivariant teacher to train a compact student that preserves the teacher’s predictive behavior while enforcing the symmetry of the downstream task.

## 6 Conclusion

We introduce TREX, a simple and effective knowledge distillation framework for PDE foundation models. By extending limited downstream data through long and noisy teacher rollouts, TREX enables the student to match the teacher’s long-horizon dynamics, thereby providing valuable additional information. Our results demonstrate that distillation in the foundation model regime is not only feasible but highly effective: compact student models can match the accuracy of large teachers while achieving substantial improvements in inference speed and memory efficiency. Additionally, the student can incorporate inductive biases to faithfully reproduce prior knowledge about the system, such as equivariances. Overall, TREX bridges the gap between the strong generalization capabilities of large foundation models and the efficiency requirements of practical surrogate modeling. In the future, TREX could be combined with other distillation techniques to also provide more information on the influence of other factors such as physical parameters to the student.

## Acknowledgments and Disclosure of Funding

Funded by the German Federal Ministry of Research, Technology and Space (BMFTR) as part of the InnoPhase project (funding code: 02NUK078). Additionally, we acknowledge the support of Deutsche Forschungsgemeinschaft (DFG) under funding code 569019417 and the support of the Stuttgart Center for Simulation Science (SimTech). Andrei Manolache acknowledges funding by the EU Horizon project ELIAS (No. 101120237). The authors thank the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for supporting Daniel Musekamp, Andrei Manolache, and Mathias Niepert.

## References

Ishan Amin, Sanjeev Raja, and Aditi S Krishnapriyan. Towards fast, specialized machine learning force fields: Distilling foundation models via energy hessians. In The Thirteenth International Conference on Learning Representations, 2025.

Lei J Ba and Rich Caruana. Do deep nets really need to be deep? Advances in neural information processing systems, 27, 2014.

X Cheng, Y He, Y Yang, X Xue, S Cheng, D Giles, X Tang, and Y Hu. Learning chaos in a linear way. In International Conference on Learning Representations, pp. 55276–55318. ICLR, 2025.

Jianping Gou, Baosheng Yu, Stephen J Maybank, and Dacheng Tao. Knowledge distillation: A survey. International journal ofcomputer vision, 129(6):1789–1819, 2021.

Jayesh K. Gupta and Johannes Brandstetter. Towards multi-spatiotemporal-scale generalized PDE modeling. Trans. Mach. Learn. Res., 2023.

Zhongkai Hao, Chang Su, Songming Liu, Julius Berner, Chengyang Ying, Hang Su, Anima Anandkumar, Jian Song, and Jun Zhu. Dpot: auto-regressive denoising operator transformer for large-scale pde pre-training. In Proceedings of the 41st International Conference on Machine Learning, pp. 17616–17635, 2024.

Jacob Helwig, Xuan Zhang, Cong Fu, Jerry Kurtin, Stephan Wojtowytsch, and Shuiwang Ji. Group equivariant fourier neural operators for partial differential equations. In International Conference on Machine Learning, pp. 12907–12930. PMLR, 2023.

Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.

Maximilian Herde, Bogdan Raonic, Tobias Rohner, Roger Käppeli, Roberto Molinaro, Emmanuel ´ de Bézenac, and Siddhartha Mishra. Poseidon: Efficient foundation models for pdes, 2024. URL https://arxiv.org/abs/2405.19101.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

David Holzmüller, Viktor Zaverkin, Johannes Kästner, and Ingo Steinwart. A framework and benchmark for deep batch active learning for regression. Journal of Machine Learning Research, 24(164):1–81, 2023.

Nick Janssens and Johan Meyers. Towards real-time optimal control of wind farms using large-eddy simulations. Wind Energy Science, 9(1):65–95, 2024.

William B Johnson, Joram Lindenstrauss, et al. Extensions of lipschitz mappings into a hilbert space. Contemporary mathematics, 26(189-206):1, 1984.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun (eds.), 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015.

Jean Kossaifi, Nikola Borislavov Kovachki, Kamyar Azizzadenesheli, and Anima Anandkumar. Multi-grid tensorized fourier neural operator for high- resolution pdes. Trans. Mach. Learn. Res., 2024.

Jean Kossaifi, Nikola Kovachki, Zongyi Li, David Pitt, Miguel Liu-Schiaffini, Robert Joseph George, Boris Bonev, Kamyar Azizzadenesheli, Julius Berner, Valentin Duruisseaux, and Anima Anandku mar. A library for learning neural operators. arXiv preprint arXiv:2412.10354, 2025.

Yuqi Li, Chuanguang Yang, Hansheng Zeng, Zeyu Dong, Zhulin An, Yongjun Xu, Yingli Tian, and Hao Wu. Frequency-aligned knowledge distillation for lightweight spatiotemporal forecasting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7262–7272, 2025.

Zongyi Li, Nikola Borislavov Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew M. Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021, 2021.

Zongyi Li, Miguel Liu-Schiaffini, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Learning chaotic dynamics in dissipative systems. Advances in Neural Information Processing Systems, 35:16768–16781, 2022.

Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin, Haoxuan Chen, Burigede Liu, Kamyar Azizzadenesheli, and Anima Anandkumar. Physics-informed neural operator for learning partial differential equations. ACM/IMS Journal ofData Science, 1(3):1–27, 2024.

Jhe-Hao Lin, Yi Yao, Chan-Feng Hsu, Hong-Xia Xie, Hong-Han Shuai, and Wen-Huang Cheng. Perspective-aware teaching: Adapting knowledge for heterogeneous distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4178–4187, 2025.

Yuxuan Liu, Jingmin Sun, Xinjie He, Griffin Pinney, Zecheng Zhang, and Hayden Schaeffer. Prose-fd: A multimodal pde foundation model for learning multiple operators for forecasting fluid dynamics. arXiv preprint arXiv:2409.09811, 2024.

Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, et al. Swin transformer v2: Scaling up capacity and resolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12009–12019, 2022a.

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022. IEEE, 2022b.

Raphael Gontijo Lopes, Stefano Fenu, and Thad Starner. Data-free knowledge distillation for deep neural networks. arXiv preprint arXiv:1710.07535, 2017.

EN Lorenz. Deterministic nonperiodic flow. J. Atmos. Sci., 20:130–141, 1963.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019, 2019.

Michael McCabe, Bruno Régaldo-Saint Blancard, Liam Parker, Ruben Ohana, Miles Cranmer, Alberto Bietti, Michael Eickenberg, Siavash Golkar, Geraud Krawezik, Francois Lanusse, Mariel Pettee, Tiberiu Tesileanu, Kyunghyun Cho, and Shirley Ho. Multiple physics pretraining for spatiotemporal surrogate models. In Advances in Neural Information Processing Systems, volume 37, 2024.

Michael McCabe, Payel Mukhopadhyay, Tanya Marwah, Bruno Regaldo-Saint Blancard, Francois Rozet, Cristiana Diaconu, Lucas Meyer, Kaze WK Wong, Hadi Sotoudeh, Alberto Bietti, et al. Walrus: A cross-domain foundation model for continuum dynamics. arXiv preprint arXiv:2511.15684, 2025.

Hossein Mobahi, Mehrdad Farajtabar, and Peter Bartlett. Self-distillation amplifies regularization in hilbert space. Advances in Neural Information Processing Systems, 33:3351–3361, 2020.

Ruben Ohana, Michael McCabe, Lucas Meyer, Rudy Morel, Fruzsina J Agocs, Miguel Beneitez, Marsha Berger, Blakesley Burkhart, Stuart B Dalziel, Drummond B Fielding, et al. The well: a large-scale collection of diverse physics simulations for machine learning. Advances in Neural Information Processing Systems, 37:44989–45037, 2024.

Wonpyo Park, Dongju Kim, Yan Lu, and Minsu Cho. Relational knowledge distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3967–3976, 2019.

Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, and Peter Battaglia. Learning mesh-based simulation with graph networks. In International conference on learning representations, 2020.

Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio. Fitnets: Hints for thin deep nets. In Yoshua Bengio and Yann LeCun (eds.), International Conference on Learning Representations, 2015.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, pp. 234–241. Springer, 2015.

Alvaro Sanchez-Gonzalez, Nicolas Heess, Jost Tobias Springenberg, Josh Merel, Martin Riedmiller, Raia Hadsell, and Peter Battaglia. Graph networks as learnable physics engines for inference and control. In International conference on machine learning, pp. 4470–4479. PMLR, 2018.

Yair Schiff, Zhong Yi Wan, Jeffrey B Parker, Stephan Hoyer, Volodymyr Kuleshov, Fei Sha, and Leonardo Zepeda-Núñez. Dyslim: Dynamics stable learning by invariant measure for chaotic systems. In Forty-first International Conference on Machine Learning, 2024.

Junhong Shen, Tanya Marwah, and Ameet Talwalkar. UPS: efficiently building foundation models for PDE solving via cross-modal adaptation. Trans. Mach. Learn. Res., 2024.

Suraj Srinivas and François Fleuret. Knowledge transfer with jacobian matching. In International conference on machine learning, pp. 4723–4731. PMLR, 2018.

Makoto Takamoto, Francesco Alesiani, and Mathias Niepert. Learning neural pde solvers with parameter-guided channel attention. In International Conference on Machine Learning, pp. 33448– 33467. PMLR, 2023.

Mingxuan Tian, Haochen Mu, Tao Liu, Mengjiao Li, Donghong Ding, and Jianping Zhao. Physicsinformed machine learning-based real-time long-horizon temperature fields prediction in metallic additive manufacturing. Communications Engineering, 4(1):168, 2025.

Han Wan, Rui Zhang, and Hao Sun. Spectral-inspired neural operator for data-efficient pde simulation in physics-agnostic regimes. arXiv preprint arXiv:2505.21573, 2025.

Zhanhong Ye, Zining Liu, Bingyang Wu, Hongjie Jiang, Leheng Chen, Minyan Zhang, Xiang Huang, Qinghe Meng Zou, Hongsheng Liu, Bin Dong, et al. Pdeformer-2: A versatile foundation model for two-dimensional partial differential equations. arXiv preprint arXiv:2507.15409, 2025.

Linfeng Zhang, Jiebo Song, Anni Gao, Jingwei Chen, Chenglong Bao, and Kaisheng Ma. Be your own teacher: Improve the performance of convolutional neural networks via self distillation. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 3713–3722, 2019.

Rui Zhang, Qi Meng, Han Wan, Yang Liu, Zhi-Ming Ma, and Hao Sun. Omnifluids: Unified physics pre-trained modeling of fluid dynamics. arXiv preprint arXiv:2506.10862, 2025.

Maksim Zhdanov, David Ruhe, Maurice Weiler, Ana Lucic, Johannes Brandstetter, and Patrick Forré. Clifford-steerable convolutional neural networks. In International Conference on Machine Learning, pp. 61203–61228. PMLR, 2024.

Hang Zhou, Yuezhou Ma, Haixu Wu, Haowen Wang, and Mingsheng Long. Unisolver: Pdeconditional transformers towards universal neural PDE solvers. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, Proceedings of Machine Learning Research. PMLR, 2025.

## A Limitations

While TREX provides an effective framework for distilling PDE foundation models, it comes with several limitations. First, TREX relies on the quality of the fine-tuned teacher. Since the method samples from the teacher-induced rollout distribution, any systematic bias or instability in the teacher’s long-horizon dynamics may be transferred to the student. In particular, if the teacher converges to an incorrect or distorted attractor, TREX may reinforce this behavior. Although mixing teachergenerated data with ground-truth trajectories mitigates this issue, it does not fully eliminate it. Second, rollout extension is most useful when long trajectories continue to explore informative regions of state space. Its benefit may therefore be limited for systems that rapidly collapse to a trivial fixed point, exhibit strongly transient dynamics, or contain multiple disconnected long-time regimes that are not represented by the available trajectories. Moreover, the Gaussian perturbations used by TREX are not guaranteed to satisfy all physical constraints of the underlying PDE and should therefore be viewed as a state-space augmentation strategy rather than as physically valid perturbations.

## B Broader Impact

This work aims to advance machine learning methods for scientific simulation and time-dependent PDEs by making foundation-model-based surrogates more efficient and easier to deploy. Faster surrogate models could reduce the computational cost of simulation workflows and make such tools more accessible in scientific and engineering applications. We do not expect broad negative societal impacts from the method itself.

## C Compute Resources

The experiments were performed on a cluster with 10 NVIDIA GeForce RTX 4090 GPUs and on a larger compute cluster with 32 NVIDIA A100 GPUs and 32 NVIDIA L40 GPUs. Each individual training run used a single GPU. The reported inference-time and memory measurements for Poseidon experiments were collected for a batch size of 64, as described in the main text.

## D Continued Related Work

Several authors have investigated explicitly including attractor dynamics for neural PDE solvers. Li et al. (2022) include a loss term to enforce that the model always converges to the attractor on artificial data drawn uniformly from a fixed ball around zero. DySLIM (Schiff et al., 2024) enforces that the learned model follows the invariant measure by minimizing the Maximum Mean Discrepancy between the data distribution and the learned attractor. Cheng et al. (2025) proposes a Poincaré Flow Neural Network, which is trained by dividing the data into the contracting and measure-invariant phases that can occur in a dissipative chaotic system. Depending on the phase, a loss is applied to enforce the contracting or invariant behavior.

## E Datasets

We mainly use the datasets provided by Herde et al. (2024). All datasets, besides Kolmogorov Flow, contain 8 time steps, which end at $\dot { T } = 0 . 7$ and use periodic boundary conditions. The spatial resolution is 128x128 for a grid of $[ 0 , 1 ] \times [ 0 , 1 ]$ . The Poseidon problems have 120 validation and 240 test trajectories.

## E.1 NS-PwC

The Navier-Stokes piecewise constants task uses the incompressible Navier-Stokes equations:

$$
\partial _ { t } { \mathbf u } + ( \mathbf u \cdot \nabla ) { \mathbf u } + \nabla p = \nu \Delta { \mathbf u } , \nabla \cdot { \mathbf u } = 0 .\tag{8}
$$

For NS-PwC, the initial state is given as a set of constants on a grid of 10 cells in each dimension:

$$
\omega _ { 0 } ( x , y ) = c _ { i , j } \quad \mathrm { f o r } ( x , y ) \in [ x _ { i - 1 } , x _ { i } ] \times [ y _ { j - 1 } , y _ { j } ] , \qquad i , j = 1 , \dots , p .\tag{9}
$$

with $\begin{array} { r } { x _ { i } = y _ { i } = \frac { i } { p } } \end{array}$ , for $i = 0 , 1 , 2 , \dotsc , p$ and the constant square values drawn from $c _ { i , j } \sim \mathcal { U } _ { [ - 1 , 1 ] }$ Here, ω refers to the vorticity that can be transformed into the velocity in the x and y directions using the incompressibility condition.

## E.2 NS-BB

In Fourier space, the Brownian bridge initial conditions are initialized as

$$
W ( \boldsymbol { x } ) = \sum _ { | k | _ { \infty } \leq N } \frac { 1 } { \| k \| _ { 2 } ^ { 3 / 2 } } \sum _ { m , n , \ell \in \{ 0 , 1 \} } \alpha _ { k } ^ { ( m n \ell ) } \operatorname { s c } _ { m } ( \boldsymbol { x } ) \mathrm { { s c } } _ { n } ( \boldsymbol { x } ) \mathrm { { s c } } _ { \ell } ( \boldsymbol { x } ) ,\tag{10}
$$

where

$$
\operatorname { s c } _ { i } ( x ) = { \left\{ \begin{array} { l l } { \sin ( x ) , } & { { \mathrm { f o r ~ } } i = 0 , } \\ { \cos ( x ) , } & { { \mathrm { f o r ~ } } i = 1 } \end{array} \right. } .\tag{11}
$$

$\mathbf { W } ( \mathbf { x } )$ does not directly give $u _ { 0 }$ . Instead, it defines the state at $\mathrm { t } = - 0 . 5 ,$ , and the initial conditions are derived by evolving the Navier-Stokes equations until t=0. The random variables are sampled from $\alpha _ { k } ^ { ( m n \ell ) } \sim \mathcal { U } _ { [ - 1 , 1 ] }$

## E.3 NS-SVS

The next Navier-Stokes task uses Sinusoidal Vortex Sheet (SVS) initial conditions, which also use a vorticity formulation

$$
\omega _ { 0 } ^ { \rho } = \phi _ { \rho } * \omega _ { 0 } ,\tag{12}
$$

where $\omega _ { 0 }$ is defined as

$$
\omega _ { 0 } ( x ) = \delta ( x - \Gamma ) - \int _ {  { \mathbb { T } } ^ { 2 } } d \Gamma ,\tag{13}
$$

and

$$
\phi _ { \rho } ( x ) = \rho ^ { - 2 } \psi \biggl ( \frac { \| x \| } { \rho } \biggr ) .\tag{14}
$$

$\psi ( r )$ is defined as

$$
\psi ( r ) = { \frac { 8 0 } { 7 \pi } } \left[ ( r + 1 ) _ { + } ^ { 3 } - 4 \left( r + { \frac { 1 } { 2 } } \right) _ { + } ^ { 3 } + 6 r _ { + } ^ { 3 } - 4 \left( r - { \frac { 1 } { 2 } } \right) _ { + } ^ { 3 } + ( r - 1 ) _ { + } ^ { 3 } \right] ,\tag{15}
$$

and Γ as

$$
\Gamma = \left\{ ( x , y ) \in \mathbb { T } ^ { 2 } \biggm | y = \frac { 1 } { 2 } + 0 . 2 \sin ( 2 \pi x ) + \sum _ { i = 1 } ^ { p } \alpha _ { i } \sin \big ( 2 \pi ( x + \beta _ { i } ) \big ) \right\} .\tag{16}
$$

The variable parameters are drawn from

$$
\alpha _ { i } \sim \mathcal { U } _ { [ 0 , 0 . 0 0 3 1 2 5 ] } , \quad \beta _ { i } \sim \mathcal { U } _ { [ 0 , 1 ] } ,\tag{17}
$$

and $\rho$ and p are fixed to $\rho = \textstyle { \frac { 5 } { 1 2 8 } }$ and $p = 1 0$

## E.4 CE-RPUI

The Compressible Euler PDE is given as

$$
u _ { t } + \nabla \cdot F ( u ) = 0 ,\tag{18}
$$

where u and $F ( u )$ are defined as

$$
u = \left[ \rho \mathbf { v } \right] , \quad F ( u ) = \left[ \rho \mathbf { v } \otimes \mathbf { v } + p \mathbf { I } \right] .\tag{19}
$$

For the CE-RPUI problem, the four channels are drawn as constant values for each region of the input grid. The input division sets $D _ { i , j }$ are computed using sinusoidal functions $\sigma _ { x } ( x , y )$ and $\sigma _ { y } ( x , y )$ which are defined as

$$
\sigma _ { x } ( x , y ) = \sum _ { i , j = 1 } ^ { p } \alpha _ { x , i , j } \sin \Bigl ( 2 \pi \bigl ( i + 2 p ^ { 2 } \bigr ) x + \bigl ( j + 2 p ^ { 2 } \bigr ) y + \beta _ { x , i , j } \Bigr ) ,\tag{20}
$$

and similarly

$$
\sigma _ { y } ( x , y ) = \sum _ { i , j = 1 } ^ { p } \alpha _ { y , i , j } \sin \Bigl ( 2 \pi \bigl ( i + 2 p ^ { 2 } \bigr ) x + \bigl ( j + 2 p ^ { 2 } \bigr ) y + \beta _ { y , i , j } \Bigr ) .\tag{21}
$$

The coefficients $\alpha _ { k , i , j } \beta _ { k , i , j }$ are drawn from uniform distributions:

$$
\alpha _ { k , i , j } \sim \mathcal { U } [ - 0 . 0 1 , 0 . 0 1 ] , \qquad \beta _ { k , i , j } \sim \mathcal { U } [ 0 , 1 ] .\tag{22}
$$

Afterward, the input space regions are defined as

$$
D _ { i , j } = \left\{ ( x , y ) \in \mathbb { T } ^ { 2 } \left| \begin{array} { l l } { x _ { \operatorname* { m i n } } \leq \{ x + \sigma _ { x } ( x , y ) + 1 \} < x _ { \operatorname* { m a x } } , } \\ { y _ { \operatorname* { m i n } } \leq \{ y + \sigma _ { y } ( x , y ) + 1 \} < y _ { \operatorname* { m a x } } } \end{array} \right. \right\} ,\tag{23}
$$

where $\{ x \} : = x - \lfloor \left\lfloor x \right\rfloor \operatorname { s g n } ( x )$ and

$$
x _ { \operatorname* { m i n } } = \frac { i } { p + 1 } , \quad x _ { \operatorname* { m a x } } = \frac { i + 1 } { p + 1 } , \quad y _ { \operatorname* { m i n } } = \frac { j } { p + 1 } , \quad y _ { \operatorname* { m a x } } = \frac { j + 1 } { p + 1 } .\tag{24}
$$

In the end, the initial condition is defined using the regions $D _ { i , j }$

$$
\left( \rho , v _ { x } , v _ { y } , p \right) \big | _ { t = 0 } = \left( \rho _ { i , j } , v _ { i , j } ^ { x } , v _ { i , j } ^ { y } , p _ { i , j } \right) \quad { \mathrm { i n ~ } } D _ { i , j } ,\tag{25}
$$

where the region-wise parameters are drawn from

$$
\rho _ { i , j } \sim \mathcal { U } [ 1 , 3 ] , \quad v _ { i , j } ^ { x } \sim \mathcal { U } [ - 1 0 , 1 0 ] , \quad v _ { i , j } ^ { y } \sim \mathcal { U } [ - 1 0 , 1 0 ] , \quad p _ { i , j } \sim \mathcal { U } [ 5 , 7 ] .\tag{26}
$$

## E.5 Kolmogorov Flow

For the ablations, we use the Kolmogorov flow dataset (Li et al., 2022) at a Reynolds number (Re) of 5000. The PDE can be described as

$$
\frac { \partial u } { \partial t } = - u \cdot \nabla u - \nabla p + \frac { 1 } { R e } \Delta u + \sin ( 4 y ) \hat { x } , \qquad ( x , y , t ) \in [ 0 , 2 \pi ] \times [ 0 , 2 \pi ] \times [ 0 , \infty )\tag{27}
$$

$$
\nabla \cdot u = 0 ,
$$

$$
( x , y , t ) \in [ 0 , 2 \pi ] \times [ 0 , 2 \pi ] \times [ 0 , \infty )\tag{28}
$$

$$
u ( \cdot , 0 ) = u _ { 0 } ,
$$

$$
( x , y ) \in [ 0 , 2 \pi ] \times [ 0 , 2 \pi ] .\tag{29}
$$

We transform the PDE into velocity form so that it can be included in Walrus’ channel mapping. The resolution is 128x128. The trajectories are 501 time steps long, where the first 100 are discarded. We use 48 trajectories for validation and test each.

## F Model and Training Details

All student models are trained to predict the difference, i.e., $\hat { \mathbf { u } } ^ { t + 1 } = \mathcal { S } ( \mathbf { u } ^ { t } ) = \mathcal { M } ( \mathbf { u } ^ { t } ) + \mathbf { u } ^ { t }$ . The learning rate is set to $1 0 ^ { - 3 }$ initially and scaled down to $1 0 ^ { - 6 }$ using cosine annealing. We use the standard Adam optimizer (Kingma & Ba, 2015) and a batch size of 32. The experiments were performed on NVIDIA GeForce RTX 4090 GPUs (one GPU per run). For the main results, we report the test error for the model checkpoint with the best error on the validation set for the ground-truth only and relational KD baselines, since we encountered overfitting for these methods. For TREX and IC-KD, we report the last model checkpoint to be fair to the Teacher baseline, which also did not use early-stopping. We use the original training loss from Poseidon over the batch B:

$$
L _ { \mathrm { t r a i n } } ^ { 1 } ( \theta ) = \frac { 1 } { N _ { \mathrm { Q O I } } } \sum _ { \mathrm { Q O I } } \frac { \displaystyle \sum _ { b \in \mathcal { B } } \sum _ { t } \sum _ { \textbf { x } _ { c _ { \mathrm { Q O I } } } } \left. \hat { \mathbf { u } } _ { \theta , b } \left( t , \mathbf { x } , c _ { \mathrm { Q O I } } \right) - \mathbf { u } _ { b } \left( t , \mathbf { x } , c _ { \mathrm { Q O I } } \right) \right. } { \displaystyle \sum _ { b \in \mathcal { B } } \sum _ { t } \sum _ { \textbf { x } _ { c _ { \mathrm { Q O I } } } } \left. \mathbf { u } _ { b } \left( t , \mathbf { x } , c _ { \mathrm { Q O I } } \right) \right. + \varepsilon _ { L _ { 1 } } } .
$$

TFNO. We use the Tucker factorized FNO (Kossaifi et al., 2024) version with the implementation provided by Kossaifi et al. (2025). The factorization reduces the number of parameters with a compression factor of 0.1. The network uses 32 modes per dimension, 32 hidden channels, three layers, and a GELU (Hendrycks & Gimpel, 2016) activation function.

U-Net. For the student model ablations, we use the modern U-Net (Ronneberger et al., 2015) as provided by Gupta & Brandstetter (2023). The U-Net has 4 levels of resolution, increasing the number of channels per level by 1, 2, 2, and 4, starting at 16 hidden channels after the embedding, resulting in 9,161,332 parameters. For U-Net, we found it advantageous to train with a 5-step error for the model ablation experiment.

FNO. We also use a traditional FNO (Li et al., 2021) without factorization for the model ablations. The model uses the same number of modes and channels as the TFNO, resulting in 5,042,836 parameters. The FNO uses 4 layers.

Poseidon. As the teacher, we use the Poseidon-L model (Herde et al., 2024). Poseidon-L was pretrained on six different datasets containing Navier-Stokes and compressible Euler equations with different sets of initial conditions. The model uses a SwinV2 transformer as its backbone, a multiresolution vision transformer. The architecture operates on a latent representation with embedding dimension $C = 1 9 2$ . For each hierarchical level i, the model employs $t _ { i } = 8$ blocks, resulting in a total of 629M trainable parameters. The network consists of $L = 4$ levels, connected via $L - 1$ downsampling and upsampling operations to progressively refine spatial resolution. The number of attention heads per level is defined as:

$$
[ h _ { 1 } , h _ { 2 } , h _ { 3 } , h _ { 4 } ] = [ 3 , 6 , 1 2 , 2 4 ] ,
$$

allowing increased representation capacity at coarser scales. In addition, each level incorporates two ConvNeXt layers (Liu et al., 2022b). The model operates on input patches of size $p = 4$ and a window size of $M = 1 6$

Training is performed using the AdamW optimizer (Loshchilov & Hutter, 2019) with cosine learning rate decay. The main learning rate is initialized as $\eta _ { b } = 5 \cdot 1 0 ^ { - 5 }$ . A weight decay of $1 0 ^ { - 6 }$ is applied for regularization. The batch size is set to 16, and the model is trained for 200 epochs without early stopping. Gradient norms are clipped at a maximum value of 5 to ensure numerical stability.

Walrus. For Walrus (McCabe et al., 2025) fine-tuning, we used only 50k sample presentations to the model (batch size 4). We used global normalization and followed the original training procedure otherwise.

For the distillation, we started 50 rollouts from different random sub-sequences in the trajectory of length 100. We found noise during Walrus rollout harmful, likely since adding noise to a multi-frame input does not just create a different state, but also suggests different physics to the model if it learned to infer it from its inputs. More importantly, it was crucial to deactivate the jitter added in Walrus inference, as this creates stochastic predictions and hence noisy labels. Walrus uses the VRMSE metric:

$$
\mathrm { V R M S E } _ { t _ { 1 } : t _ { 2 } } \left( \widehat { \mathbf { u } } , \mathbf { u } \right) = \frac { 1 } { t _ { 2 } - t _ { 1 } + 1 } \sum _ { t = t _ { 1 } } ^ { t _ { 2 } } \sqrt { \frac { \left. \left| \widehat { \mathbf { u } } _ { t } - \mathbf { u } _ { t } \right| ^ { 2 } \right. } { \left. \left| \mathbf { u } _ { t } - \overline { { \mathbf { u } } } _ { t } \right| ^ { 2 } \right. + \varepsilon } } .\tag{30}
$$

We use the normalize mean absolute error as the loss function

$$
\mathcal { L } _ { \mathrm { W } } ( \mathbf { u } , \hat { \mathbf { u } } ) = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \left| \mathrm { N o r m } _ { \Delta } \left( \hat { \mathbf { u } } ^ { t + 1 } - \mathbf { u } ^ { t } \right) - \mathrm { N o r m } _ { \Delta } \left( \mathbf { u } ^ { t + 1 } - \mathbf { u } ^ { t } \right) \right| _ { 1 } .
$$

The normalization is performed with the global mean and standard deviation of the transition differences.

## G Knowledge Distillation Details

In this section, we provide additional results on the KD techniques used in this paper.

TREX For TREX, we use 100 rollout steps starting from the initial condition by default. The noise is applied after 10 steps, with a scaling factor of $\sigma = 1$ . The TREX loss weight is set to 1 as well. The noisy rollouts are regenerated every 1000 epochs.

IC-KD For the IC-KD baseline, we use a buffer of teacher-generated trajectories that we regenerate every 10 epochs, with the size of the buffer determined by the number of ground truth trajectories available. The teacher-generated rollouts are then also divided into sub-trajectories for n-step training. Note that for the Brownian bridge task, the initial conditions are computationally intensive to compute because they are evolved from an earlier state.

$$
\mathcal { L } _ { \mathrm { I C } } = \mathbb { E } _ { \mathbf { u } _ { 0 } \sim p _ { 0 } } \left[ \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \ell \left( S ( \mathbf { u } ^ { t } ) , \mathcal { T } ( \mathbf { u } ^ { t } ) \right) \right] , \qquad \mathbf { u } ^ { t + 1 } = \mathcal { T } ( \mathbf { u } ^ { t } )
$$

Relational KD Relational KD (Park et al., 2019) aims to make the distance between an input pair $i , j$ from a mini-batch in the student feature space close to the distance in the feature space:

$$
\mathcal { L } _ { \mathrm { R K D } } = \sum _ { i < j } \ell _ { \delta } \Big ( d ( \mathbf { t } _ { i } , \mathbf { t } _ { j } ) , d ( \mathbf { s } _ { i } , \mathbf { s } _ { j } ) \Big ) ,\tag{31}
$$

where t and s are the teacher and student features, respectively. The loss is the Huber distance

$$
\ell _ { \delta } ( x , y ) = { \left\{ \begin{array} { l l } { { \frac { 1 } { 2 } } ( x - y ) ^ { 2 } } & { { \mathrm { i f ~ } } | x - y | \leq 1 } \\ { | x - y | - { \frac { 1 } { 2 } } } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. } .\tag{32}
$$

Since the feature spaces of the spatial neural network are very large, we use Gaussian sketching, which uses random Gaussian matrices R to reduce the input features to a lower dimension for the teacher

$$
\begin{array} { r } { \mathbf { t } _ { i } = \mathbf { R } _ { T } \tilde { \mathbf { t } } _ { i } , \qquad ( \mathbf { R } _ { T } ) _ { a b } \sim \mathcal { N } \bigl ( 0 , \frac 1 k \bigr ) , \qquad \mathbf { R } _ { T } \in \mathbb { R } ^ { k \times d _ { T } } , } \end{array}\tag{33}
$$

and equivalently, the student model

$$
\begin{array} { r } { \mathbf { s } _ { i } = \mathbf { R } _ { S } \tilde { \mathbf { s } } _ { i } , \qquad ( \mathbf { R } _ { S } ) _ { a b } \sim \mathcal { N } \big ( 0 , \frac { 1 } { k } \big ) , \qquad \mathbf { R } _ { S } \in \mathbb { R } ^ { k \times d _ { S } } . } \end{array}\tag{34}
$$

It can be shown that the error in the distances $d ( \mathbf { t } _ { i } , \mathbf { t } _ { j } )$ introduced by the Gaussian sketch is bounded (Holzmüller et al., 2023; Johnson et al., 1984). We use k = 512 in all experiments

The loss $\mathcal { L } _ { \mathrm { R K D } }$ is added to the main, ground truth loss with a scaling factor of 0.001. We use the last-layer features for both the student and the teacher.

## H Further Results

Tab. 5 shows the main results for the median relative $L ^ { 1 }$ metric with the 95% confidence interval of the mean. For completeness, we also show the relative $L ^ { 1 }$ error for all time steps (Tab. 6). Additionally, we report the training durations in Tab. 7. Fig. 8 shows a full example rollout for the NS-SVS task and Fig. 9 shows the error growth for all 4 considered tasks.

Table 5: Median rel. $L ^ { 1 }$ metric of the last time step. Entries show the mean with its 95% confidence interval calculated over three seeds.
<table><tr><td>Method</td><td> $\overline { { N = 4 } }$ </td><td> $N = 8$ </td><td> $N = 1 6$ </td><td> $\overline { { N = 3 2 } }$ </td></tr><tr><td colspan="5">CE-RPUI</td></tr><tr><td>TREX</td><td> $\mathbf { \overline { { 0 . 4 2 1 \pm 0 . 0 1 5 } } }$ </td><td> $\mathbf { \overline { { 0 . 2 9 5 \pm 0 . 1 8 2 } } }$ </td><td> $\mathbf { 0 . 1 9 7 \pm 0 . 0 3 8 }$ </td><td> $\overline { { 0 . 1 4 4 \pm 0 . 0 1 0 } }$ </td></tr><tr><td>IC-KD</td><td> $0 . 4 4 0 \pm 0 . 0 7 7$ </td><td> $0 . 3 0 4 \pm 0 . 1 6 7$ </td><td> $0 . 2 2 3 \pm 0 . 1 0 5$ </td><td> $0 . 1 4 5 \pm 0 . 0 0 6$ </td></tr><tr><td>Relational KD</td><td> $0 . 6 0 8 \pm 0 . 0 9 9$ </td><td> $0 . 4 4 6 \pm 0 . 0 6 8$ </td><td> $0 . 2 9 6 \pm 0 . 0 3 1$ </td><td> $0 . 1 9 6 \pm 0 . 0 2 0$ </td></tr><tr><td>Only GT</td><td> $0 . 6 0 8 \pm 0 . 0 8 6$ </td><td> $0 . 4 5 8 \pm 0 . 0 9 2$ </td><td> $0 . 2 9 6 \pm 0 . 0 1 1$ </td><td> $0 . 1 9 9 \pm 0 . 0 1 8$ </td></tr><tr><td>Teacher</td><td> $0 . 4 4 6 \pm 0 . 0 7 4$ </td><td> $0 . 3 1 0 \pm 0 . 1 5 7$ </td><td> $0 . 2 2 3 \pm 0 . 1 0 8$ </td><td> $0 . 1 4 3 \pm 0 . 0 0 9$ </td></tr><tr><td colspan="5"> $\overline { { { \mathrm { N S } } { - } { S } { \mathrm { V S } } } }$ </td></tr><tr><td>TREX</td><td> $\mathbf { \overline { { 0 . 0 4 3 \pm 0 . 0 3 7 } } }$ </td><td> $\mathbf { \overline { { 0 . 0 2 1 } } } \pm \mathbf { 0 . 0 0 5 }$ </td><td> $\mathbf { \overline { { 0 . 0 1 4 } } } \pm \mathbf { 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 0 1 1 \pm 0 . 0 0 2 }$ </td></tr><tr><td>IC-KD</td><td> $0 . 0 4 5 \pm 0 . 0 3 2$ </td><td> $0 . 0 2 5 \pm 0 . 0 0 5$ </td><td> $0 . 0 1 8 \pm 0 . 0 0 3$ </td><td> $0 . 0 1 5 \pm 0 . 0 0 4$ </td></tr><tr><td>Relational KD</td><td> $0 . 1 7 8 \pm 0 . 0 4 9$ </td><td> $0 . 0 5 0 \pm 0 . 0 2 0$ </td><td> $0 . 0 2 1 \pm 0 . 0 0 4$ </td><td> $0 . 0 1 1 \pm 0 . 0 0 1$ </td></tr><tr><td>Only GT</td><td> $0 . 1 7 1 \pm 0 . 0 5 3$ </td><td> $0 . 0 6 0 \pm 0 . 0 2 1$ </td><td> $0 . 0 2 3 \pm 0 . 0 0 5$ </td><td> $0 . 0 1 1 \pm 0 . 0 0 1$ </td></tr><tr><td>Teacher</td><td> $0 . 0 4 3 \pm 0 . 0 3 0$ </td><td></td><td> $0 . 0 1 8 \pm 0 . 0 0 5$ </td><td> $0 . 0 1 5 \pm 0 . 0 0 3$ </td></tr><tr><td colspan="5"> $\frac { 0 . 0 2 4 \pm 0 . 0 0 5 } { \mathrm { N S - P w C } }$ </td></tr><tr><td>TREX</td><td> $\mathbf { \overline { { 0 . 0 5 6 \pm 0 . 0 0 3 } } }$ </td><td> $\mathbf { 0 . 0 4 2 \pm 0 . 0 1 7 }$ </td><td> $\overline { { 0 . 0 3 2 \pm 0 . 0 1 2 } }$ </td><td> $\overline { { 0 . 0 2 1 \pm 0 . 0 0 3 } }$ </td></tr><tr><td>IC-KD</td><td> $0 . 0 6 7 \pm 0 . 0 1 2$ </td><td> $0 . 0 5 3 \pm 0 . 0 3 9$ </td><td> $0 . 0 4 2 \pm 0 . 0 3 8$ </td><td> $0 . 0 2 9 \pm 0 . 0 0 7$ </td></tr><tr><td>Relational KD</td><td> $0 . 1 7 5 \pm 0 . 0 1 4$ </td><td> $0 . 0 5 9 \pm 0 . 0 1 1$ </td><td> $\mathbf { 0 . 0 2 7 \pm 0 . 0 0 3 }$ </td><td> $0 . 0 1 3 \pm 0 . 0 0 1$ </td></tr><tr><td>Only GT</td><td> $0 . 2 8 2 \pm 0 . 0 8 1$ </td><td> $0 . 0 6 8 \pm 0 . 0 1 1$ </td><td> $0 . 0 2 9 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 0 1 3 \pm 0 . 0 0 1 }$ </td></tr><tr><td>Teacher</td><td> $0 . 0 7 0 \pm 0 . 0 1 1$ </td><td> $\frac { 0 . 0 5 5 \pm 0 . 0 4 0 } { \mathrm { N S - B B } }$ </td><td> $0 . 0 4 4 \pm 0 . 0 3 5$ </td><td> $0 . 0 3 1 \pm 0 . 0 0 5$ </td></tr><tr><td colspan="5"></td></tr><tr><td>TREX</td><td> $\mathbf { 0 . 0 3 6 \pm 0 . 0 0 7 }$ </td><td> $\mathbf { 0 . 0 2 6 \pm 0 . 0 0 7 }$ </td><td> $\overline { { 0 . 0 2 3 \pm 0 . 0 0 6 } }$ </td><td> $\overline { { 0 . 0 1 6 \pm 0 . 0 0 8 } }$ </td></tr><tr><td>IC-KD</td><td> $0 . 0 4 1 \pm 0 . 0 1 0$ </td><td> $0 . 0 3 4 \pm 0 . 0 0 4$ </td><td> $0 . 0 3 1 \pm 0 . 0 0 5$ </td><td> $0 . 0 2 3 \pm 0 . 0 0 9$ </td></tr><tr><td>Relational KD</td><td> $0 . 1 1 1 \pm 0 . 0 2 2$ </td><td> $0 . 0 3 4 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 0 1 5 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 0 0 7 \pm 0 . 0 0 2 }$ </td></tr><tr><td>Only GT</td><td> $0 . 1 7 8 \pm 0 . 0 7 7$ </td><td> $0 . 0 4 1 \pm 0 . 0 1 5$ </td><td> $0 . 0 1 5 \pm 0 . 0 0 3$ </td><td> $0 . 0 0 7 \pm 0 . 0 0 1$ </td></tr><tr><td>Teacher</td><td> $0 . 0 4 6 \pm 0 . 0 1 0$ </td><td> $0 . 0 3 9 \pm 0 . 0 0 5$ </td><td> $0 . 0 3 3 \pm 0 . 0 0 3$ </td><td> $0 . 0 2 5 \pm 0 . 0 0 8$ </td></tr></table>

Table 6: Median rel. $L ^ { 1 }$ metric over all timesteps. Entries show the mean with its 95% confidence interval calculated over three seeds.
<table><tr><td>Method</td><td> $\overline { { N = 4 } }$ </td><td> $N = 8$ </td><td> $N = 1 6$ </td><td> $\overline { { N = 3 2 } }$ </td></tr><tr><td colspan="5"> $\overline { { \mathrm { C E - R P U I } } }$ </td></tr><tr><td>TREX</td><td> $\mathbf { 0 . 2 7 7 \pm 0 . 0 2 7 }$ </td><td> $\mathbf { 0 . 1 8 2 \pm 0 . 0 6 5 }$ </td><td> $\mathbf { 0 . 1 2 7 \pm 0 . 0 1 9 }$ </td><td> $\overline { { 0 . 0 9 5 \pm 0 . 0 0 8 } }$ </td></tr><tr><td>IC-KD</td><td> $0 . 2 8 1 \pm 0 . 0 3 5$ </td><td> $0 . 1 8 4 \pm 0 . 0 8 0$ </td><td> $0 . 1 3 5 \pm 0 . 0 5 5$ </td><td> $0 . 0 9 3 \pm 0 . 0 0 5$ </td></tr><tr><td>Relational KD</td><td> $0 . 4 7 1 \pm 0 . 0 7 4$ </td><td> $0 . 3 1 6 \pm 0 . 0 4 2$ </td><td> $0 . 1 9 7 \pm 0 . 0 2 5$ </td><td> $0 . 1 2 9 \pm 0 . 0 1 4$ </td></tr><tr><td>Only GT</td><td> $0 . 4 6 6 \pm 0 . 0 6 7$ </td><td> $0 . 3 2 7 \pm 0 . 0 4 9$ </td><td> $0 . 2 0 1 \pm 0 . 0 1 1$ </td><td> $0 . 1 3 0 \pm 0 . 0 0 5$ </td></tr><tr><td>Teacher</td><td> $0 . 2 8 2 \pm 0 . 0 3 8$ </td><td> $0 . 1 8 5 \pm 0 . 0 7 8$ </td><td> $0 . 1 2 9 \pm 0 . 0 5 5$ </td><td> $0 . 0 8 4 \pm 0 . 0 0 5$ </td></tr><tr><td colspan="5"> $\overline { { { \bf N S } - { \bf S V } { \bf S } } }$ </td></tr><tr><td>TREX</td><td> $\overline { { 0 . 0 2 0 \pm 0 . 0 1 4 } }$ </td><td> $\mathbf { \overline { { 0 . 0 0 9 \pm 0 . 0 0 2 } } }$ </td><td> $\mathbf { \overline { { 0 . 0 0 6 \pm 0 . 0 0 1 } } }$ </td><td> $\overline { { 0 . 0 0 4 \pm 0 . 0 0 1 } }$ </td></tr><tr><td>IC-KD</td><td> $0 . 0 2 0 \pm 0 . 0 0 9$ </td><td> $0 . 0 1 1 \pm 0 . 0 0 2$ </td><td> $0 . 0 0 8 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 6 \pm 0 . 0 0 1$ </td></tr><tr><td>Relational KD</td><td> $0 . 0 7 0 \pm 0 . 0 2 3$ </td><td> $0 . 0 1 9 \pm 0 . 0 0 9$ </td><td> $0 . 0 0 8 \pm 0 . 0 0 2$ </td><td> $0 . 0 0 4 \pm 0 . 0 0 1$ </td></tr><tr><td>Only GT</td><td> $0 . 0 6 8 \pm 0 . 0 2 4$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 9$ </td><td> $0 . 0 0 8 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 0 0 4 } \pm 0 . 0 \mathbf { 0 1 }$ </td></tr><tr><td>Teacher</td><td> $0 . 0 2 0 \pm 0 . 0 0 8$ </td><td> $0 . 0 1 1 \pm 0 . 0 0 2$ </td><td> $0 . 0 0 8 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 6 \pm 0 . 0 0 1$ </td></tr><tr><td colspan="5"> $\overline { { { \mathrm { N S - P w C } } } }$ </td></tr><tr><td>TREX</td><td> $\mathbf { 0 . 0 3 2 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 0 2 4 } \pm 0 . 0 0 7$ </td><td> $\overline { { 0 . 0 1 9 \pm 0 . 0 0 4 } }$ </td><td>0.014 ± 0.002</td></tr><tr><td>IC-KD</td><td> $0 . 0 3 7 \pm 0 . 0 0 9$ </td><td> $0 . 0 2 8 \pm 0 . 0 1 6$ </td><td> $0 . 0 2 3 \pm 0 . 0 1 4$ </td><td> $0 . 0 1 7 \pm 0 . 0 0 3$ </td></tr><tr><td>Relational KD</td><td> $0 . 1 0 5 \pm 0 . 0 1 1$ </td><td> $0 . 0 3 7 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 0 1 8 \pm 0 . 0 0 2 }$ </td><td> $0 . 0 1 0 \pm 0 . 0 0 1$ </td></tr><tr><td>Only GT</td><td> $0 . 1 7 2 \pm 0 . 0 4 8$ </td><td> $0 . 0 4 3 \pm 0 . 0 0 8$ </td><td> $0 . 0 1 9 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 0 1 0 \pm 0 . 0 0 1 }$ </td></tr><tr><td>Teacher</td><td> $0 . 0 3 9 \pm 0 . 0 1 1$ </td><td> $0 . 0 2 9 \pm 0 . 0 1 6$ </td><td> $0 . 0 2 4 \pm 0 . 0 1 4$ </td><td> $0 . 0 1 8 \pm 0 . 0 0 2$ </td></tr><tr><td colspan="5">NS-BB</td></tr><tr><td>TREX</td><td> $\mathbf { 0 . 0 2 2 \pm 0 . 0 0 5 }$ </td><td> $\mathbf { 0 . 0 1 6 \pm 0 . 0 0 5 }$ </td><td> $\overline { { 0 . 0 1 3 \pm 0 . 0 0 2 } }$ </td><td> $\overline { { 0 . 0 0 9 \pm 0 . 0 0 3 } }$ </td></tr><tr><td>IC-KD</td><td> $0 . 0 2 3 \pm 0 . 0 0 6$ </td><td> $0 . 0 1 9 \pm 0 . 0 0 4$ </td><td> $0 . 0 1 7 \pm 0 . 0 0 1$ </td><td> $0 . 0 1 2 \pm 0 . 0 0 4$ </td></tr><tr><td>Relational KD</td><td> $0 . 0 6 7 \pm 0 . 0 1 2$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 4$ </td><td> $\mathbf { 0 . 0 1 0 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 0 0 5 \pm 0 . 0 0 1 }$ </td></tr><tr><td>Only GT</td><td> $0 . 1 0 8 \pm 0 . 0 5 0$ </td><td> $0 . 0 2 6 \pm 0 . 0 0 9$ </td><td> $0 . 0 1 0 \pm 0 . 0 0 2$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 1$ </td></tr><tr><td>Teacher</td><td> $0 . 0 2 6 \pm 0 . 0 0 7$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 2$ </td><td> $0 . 0 1 8 \pm 0 . 0 0 1$ </td><td> $0 . 0 1 4 \pm 0 . 0 0 4$ </td></tr></table>

Table 7: Average training duration in minutes on NS-PwC.
<table><tr><td>Method</td><td> $N = 4$ </td><td> $N = 8$ </td><td> $N = 1 6$ </td><td> $N = 3 2$ </td></tr><tr><td>TREX</td><td> $4 7 . 1 \pm 1 . 9$ </td><td> $9 9 . 8 \pm 3 . 2 $ </td><td> $1 4 7 . 4 \pm 1 . 4$ </td><td> $2 9 1 . 4 \pm 4 . 6$ </td></tr><tr><td>IC-KD</td><td> $8 5 . 2 \pm 2 . 3$ </td><td> $1 1 7 . 7 \pm 4 . 5$ </td><td> $1 9 3 . 5 \pm 1 . 4$ </td><td> $3 3 8 . 7 \pm 6 . 1$ </td></tr><tr><td>Relational KD</td><td> $3 8 . 2 \pm 1 . 4$ </td><td> $7 2 . 9 \pm 2 . 7$ </td><td> $1 4 3 . 1 \pm 3 . 8$ </td><td> $2 8 0 . 2 \pm 9 . 8$ </td></tr><tr><td>Only Ground Truth</td><td> $3 7 . 3 \pm 0 . 4$ </td><td> $7 0 . 2 \pm 0 . 4$ </td><td> $1 3 8 . 4 \pm 4 . 3$ </td><td> $2 7 1 . 5 \pm 7 . 8$ </td></tr></table>

t in Segment  
![](images/aabb0277d27036f52bd0e04f13c9f1bdc3560ca352715905b6d4ef03d47c215b.jpg)  
Figure 8: Example trajectory (velocity-x channel) generated using the teacher for σ = 1 on NS-SVS. After each sub-trajectory (row), the noise is applied.

![](images/3d2967eb1e56d6429ad417a09312d849cd10845da25619f8a1646787ce6af5d2.jpg)  
Figure 9: Error over time for the student trained with TREX (8 trajectories). The shaded area shows the 95% confidence interval of the mean (3 seeds). The dashed line shows the time horizon seen during training.

## H.1 Self-Distillation

We also perform a self-distillation experiment to further examine the influence of the teacher pretraining. We use the TFNO trained only on ground-truth samples as the teacher for distillation into a freshly initialized model of the same architecture. The results in Tab. 8 show that, while self-distillation has a small positive regularizing effect, the main advantage comes from distilling the pretrained, higher-capacity foundation model.

Table 8: Self-distillation experiment using TREX on CE-RPUI (4 trajectories).
<table><tr><td>Method</td><td>Median rel. L¹</td></tr><tr><td>Only GT</td><td>0.608</td></tr><tr><td>TREX (self-distillation)</td><td>0.585</td></tr><tr><td>TREX (from Poseidon)</td><td>0.421</td></tr></table>

## H.2 Equivariance

As a complement to the final-step comparison in Fig. 7, Fig. 10 shows the complete shifted rollout on NS-SVS. On the unshifted rollout, the teacher initially appears slightly closer to the ground truth than the student. After applying the spatial shift, however, the teacher prediction quickly degrades, starting around steps 3/4, while the student remains consistent throughout the rollout.

![](images/049676c3deb576099d183f6a13f5b99ca878e1ef4a1779cbf5cb26e36d8e953a.jpg)  
Figure 10: Full shifted-rollout comparison on NS-SVS. Each row shows one autoregressive step. The teacher accumulates larger errors under the spatial shift, while the equivariant student remains consistent with the shifted ground truth throughout the rollout.