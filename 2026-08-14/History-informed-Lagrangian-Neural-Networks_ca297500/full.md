# History-informed Lagrangian Neural Networks

Tianshuo Zhang , Xianglei Xing <sup>⋆</sup>, Wenzhe Zhai , Jia Gao , and He Cao

College of Intelligent Systems Science and Engineering, Harbin Engineering University, Harbin 150001, China

Abstract. Forecasting the long-horizon evolution of mechanical systems from position-only observations is a pivotal yet dificult task, as hidden velocities and trajectory-specific physical properties must be inferred simultaneously. Although physics-guided neural networks like Lagrangian Neural Networks (LNNs) guarantee physical plausibility, they generally require complete state inputs and lack adaptability to changing system parameters. To break these limitations, we introduce Historyinformed Lagrangian Neural Networks (HiLNN). Grounded in the insight that temporal position sequences implicitly encode underlying dynamics, HiLNN employs a recurrent encoder to extract a latent context from history. This context not only reconstructs the unobserved initial velocity but also adaptively modulates the mass matrix, potential energy, and damping coeficients of a structured Lagrangian system. By leveraging a diferentiable RK4 rollout scheme, the entire pipeline is optimized end-toend under multi-step trajectory supervision and energy-consistency regularization. Empirical evaluations across conservative, dissipative, and heterogeneous variable-parameter systems show that HiLNN delivers superior long-term prediction accuracy and maintains precise energy profiles compared to state-of-the-art baselines. The source code is publicly available at https://github.com/yingtian22/History-informed-LNN.

Keywords: physical forecasting · Lagrangian neural networks · partial observations · system identification · structured dynamics

## 1 Introduction

Long-horizon forecasting of mechanical systems is fundamental to physical reasoning, robotics, control, and scientific modeling [1,15,17,9,12]. A reliable model should remain stable and physically plausible during extended open-loop rollouts, where small errors can recursively accumulate and produce unstable trajectories or inconsistent physical quantities such as velocity and energy.

Recent neural dynamics models learn physical evolution from data through discrete-time transitions or continuous-time vector fields [15,3,16,13,10]. However, they often treat dynamics as black-box mappings without explicit mechanical structure, making long-term rollouts prone to error accumulation and physical inconsistency [17,7].

![](images/a8e036f3de730ed4bb7fb743cbb7fa654a905f5d125806031605c3264bf1c910.jpg)  
Fig. 1. Overview of HiLNN.

Physics-guided neural dynamics address this limitation by incorporating mechanical priors. Hamiltonian and Lagrangian Neural Networks derive dynamics from learned energy functions or Lagrangians, improving interpretability and rollout stability [7,4,14,6,22]. Recent extensions further handle constrained, controlled, or dissipative systems through explicit structure and energy-dissipation mechanisms [14,22,18,5]. However, most Lagrangian models require full initial states, especially velocity, and learn a fixed global dynamics function, limiting their use under position-only observations and trajectory-varying dynamics.

Our key observation is that position history contains cues about missing states and trajectory-specific properties, consistent with delay-coordinate reconstruction and recent studies on partial-observation dynamics learning [19,2]. Although velocity and physical parameters are unobserved, temporal position evolution reflects local motion trends and hidden dynamical variations. We therefore infer a compact latent context from history, enabling structured dynamics to adapt to each trajectory beyond current-position or finite-diference estimates.

To this end, we propose HiLNN, a history-informed Lagrangian framework for position-only mechanical forecasting. Given a coordinate history, HiLNN uses a recurrent encoder to infer a latent dynamical context, which estimates the missing initial velocity and conditions context-dependent mass, potential, and optional damping terms. Starting from the inferred state, HiLNN performs differentiable RK4 rollout and jointly trains the encoder and structured dynamics module.

We evaluate HiLNN on conservative, dissipative, and variable-parameter pendulum systems. Results show that HiLNN improves rollout accuracy and physical consistency over representative baselines, including LNN, HNN, Neural ODE, and MLP predictors. Our contributions are summarized as follows:

– We study position-only long-horizon mechanical forecasting, where hidden velocity and trajectory-specific dynamics are inferred from history.

– We propose HiLNN, a history-informed Lagrangian framework that conditions mass, potential, and optional damping on latent context.

– We show that HiLNN produces more accurate and physically consistent rollouts on conservative, dissipative, and variable-parameter systems.

## 2 Related Work

## 2.1 Neural Dynamics Learning

Neural networks, such as recurrent models and Neural ODEs [3,16,11], are widely used to model continuous or discrete dynamical systems. Despite their flexibility in predicting future evolution via numerical integration, these black-box models lack explicit mechanical structure, energy consistency, or physical constraints. Consequently, they often sufer from accumulated rollout errors and physically implausible predictions over long horizons. To address this, our work builds on structured physical dynamics and introduces history-dependent context to improve long-horizon forecasting under incomplete observations.

## 2.2 Physics-Guided Neural Dynamics

Physics-guided neural dynamics embed mechanical structures to improve prediction accuracy. Representative examples include Hamiltonian Neural Networks for energy conservation [7] and Lagrangian Neural Networks using the Euler– Lagrange equation [4]. Further extensions introduce mass matrices, dissipation, or domain priors to enhance interpretability [14,21]. However, these methods typically learn a single global model and require complete state observations. In contrast, our work conditions structured Lagrangian dynamics on a historyinferred latent context, adapting to partially observed and trajectory-dependent systems.

## 2.3 Dynamics Learning from Partial Observations

Physical dynamics are frequently observed through incomplete measurements, omitting key variables like velocity or damping. Existing approaches employ recurrent encoders, latent state-space models, or neural ODE variants to infer hidden states from observation histories [8,16,20]. Although efective, their latent dynamics remain black-box and lack explicit mechanical structure. Conversely, our method infers a latent mechanical context from position-only history to condition a Lagrangian dynamics model, enabling structured, long-horizon prediction under partial observability and trajectory-dependent variations.

## 3 Method

## 3.1 Problem Formulation

We study long-horizon forecasting of mechanical systems from position-only observations. Let $q _ { t } \in \mathbb { R } ^ { d }$ be the generalized coordinate and $\dot { q } _ { t } \in \mathbb { R } ^ { d }$ be the corresponding velocity, forming the physical state $x _ { t } = ( q _ { t } , \dot { q } _ { t } )$ . Unlike standard

Lagrangian Neural Networks that usually assume access to the full initial state, we consider a partially observed setting where only a history of positions is available:

$$
\mathcal { H } _ { t } ^ { q } = \{ q _ { t - L + 1 } , q _ { t - L + 2 } , . ~ . ~ . ~ , q _ { t } \} .\tag{1}
$$

Given $\mathcal { H } _ { t } ^ { q }$ , the goal is to predict the future state trajectory

$$
\hat { \mathcal { X } } _ { t + 1 : t + H } = \{ ( \hat { q } _ { t + 1 } , \hat { \dot { q } } _ { t + 1 } ) , \dots , ( \hat { q } _ { t + H } , \hat { \dot { q } } _ { t + H } ) \}\tag{2}
$$

over an H-step open-loop rollout.

This setting requires inferring both the missing initial velocity and trajectoryspecific dynamics from position history. HiLNN addresses this by using a historyinformed context to condition a structured Lagrangian dynamics model.

## 3.2 Overview of HiLNN

HiLNN combines history-based state inference and structured Lagrangian dynamics. As shown in Fig. 1, the position history $\mathcal { H } _ { t } ^ { q }$ is encoded into a latent context

$$
z _ { t } = \mathrm { E n c } _ { \theta } ( \mathcal { H } _ { t } ^ { q } ) ,\tag{3}
$$

which summarizes trajectory-specific dynamical information. The context is used by a velocity head to infer the missing initial velocity $\hat { \dot { q } } _ { t }$ , forming

$$
\hat { x } _ { t } = ( q _ { t } , \hat { \dot { q } } _ { t } ) .\tag{4}
$$

It also conditions the Lagrangian module through context-dependent mass, potential, and optional damping terms. Starting from $\boldsymbol { \hat { x } } _ { t } .$ , HiLNN performs an openloop RK4 rollout for H steps, with $z _ { t }$ kept fixed as the inferred context of the current trajectory.

Unlike conventional LNNs that require known velocities and use a fixed global dynamics model, HiLNN infers both hidden velocity and trajectory-specific context from history.

## 3.3 History-Informed Context Encoder

The history encoder maps the position history to a compact latent context. Given

$$
\mathcal { H } _ { t } ^ { q } = \{ q _ { t - L + 1 } , q _ { t - L + 2 } , . ~ . ~ . ~ , q _ { t } \} ,\tag{5}
$$

we encode the sequence with a recurrent network:

$$
h _ { t } = \mathrm { G R U } _ { \theta } ( \mathcal { H } _ { t } ^ { q } ) ,\tag{6}
$$

where $h _ { t }$ is the final hidden representation. It is projected to the latent context:

$$
z _ { t } = \mathrm { M L P } _ { z } ( h _ { t } ) .\tag{7}
$$

The context $z _ { t }$ captures information unavailable from the current position alone, including motion trends, hidden velocity cues, and parameter variations. It is shared by the velocity head and the Lagrangian module, and remains fixed during rollout.

## 3.4 Context-Conditioned Lagrangian Dynamics

To preserve mechanical structure, HiLNN models dynamics with a contextconditioned Lagrangian. Given $x = ( q , \dot { q } )$ and context $z _ { t } ,$ , we define

$$
\mathcal { L } _ { \theta } ( \boldsymbol { q } , \dot { \boldsymbol { q } } , z _ { t } ) = T _ { \theta } ( \boldsymbol { q } , \dot { \boldsymbol { q } } , z _ { t } ) - V _ { \phi } ( \boldsymbol { q } , z _ { t } ) ,\tag{8}
$$

where $T _ { \theta }$ and $V _ { \phi }$ denote the kinetic and potential energy terms, respectively. The kinetic energy is parameterized by a positive mass matrix:

$$
T _ { \theta } ( q , \dot { q } , z _ { t } ) = \frac { 1 } { 2 } \dot { q } ^ { \top } M _ { \theta } ( q , z _ { t } ) \dot { q } ,\tag{9}
$$

and the potential energy is predicted by a neural network conditioned on both the coordinate and the inferred context:

$$
V _ { \phi } ( q , z _ { t } ) = \mathrm { M L P } _ { V } ( [ q , z _ { t } ] ) .\tag{10}
$$

The mass term is constrained positive. For one-dimensional systems, we use

$$
M _ { \theta } ( q , z _ { t } ) = \mathrm { s o f t p l u s } \left( \mathrm { M L P } _ { M } ( [ q , z _ { t } ] ) \right) + \epsilon ,\tag{11}
$$

where ϵ ensures numerical stability. For multi-dimensional systems, we use a positive diagonal mass matrix with element-wise softplus.

The resulting acceleration is derived from the Euler–Lagrange equation:

$$
\frac { d } { d t } \frac { \partial \mathcal { L } _ { \theta } } { \partial \dot { q } } - \frac { \partial \mathcal { L } _ { \theta } } { \partial q } = Q ,\tag{12}
$$

where $Q$ is the generalized non-conservative force. We set $Q = 0$ for conservative systems and optionally use context-conditioned damping for dissipative systems:

$$
Q = - D _ { \psi } ( q , z _ { t } ) \dot { q } ,\tag{13}
$$

where $D _ { \psi } ( q , z _ { t } )$ is constrained to be non-negative.

In practice, the acceleration can be obtained by solving the Euler–Lagrange system:

$$
{ \ddot { q } } = A ^ { - 1 } \left( { \frac { \partial { \mathcal { L } } _ { \theta } } { \partial q } } - B { \dot { q } } + Q \right) ,\tag{14}
$$

where

$$
A = \frac { \partial ^ { 2 } \mathcal { L } _ { \theta } } { \partial \dot { q } ^ { 2 } } , \qquad B = \frac { \partial ^ { 2 } \mathcal { L } _ { \theta } } { \partial q \partial \dot { q } } .\tag{15}
$$

Thus, dynamics are computed from a structured energy-based formulation rather than direct acceleration regression. Conditioning mass, potential, and damping on $z _ { t }$ allows HiLNN to adapt to trajectory-dependent dynamics while retaining mechanical structure.

## 3.5 Initial State Inference and Diferentiable RK4 Rollout

Since the initial velocity is unobserved, HiLNN infers it from the last position and context:

$$
\hat { \dot { q } } _ { t } = \mathrm { M L P } _ { v } ( [ q _ { t } , z _ { t } ] ) ,\tag{16}
$$

where $[ q _ { t } , z _ { t } ]$ denotes concatenation. The initial state is then

$$
\hat { x } _ { t } = ( q _ { t } , \hat { \dot { q } } _ { t } ) .\tag{17}
$$

Starting from $\hat { x } _ { t } ,$ the model performs open-loop prediction using the contextconditioned dynamics. Let

$$
\frac { d x } { d t } = f _ { \theta } ( x , z _ { t } ) = \left[ \begin{array} { c } { { \dot { q } } } \\ { { \ddot { q } \theta ( q , \dot { q } , z _ { t } ) } } \end{array} \right] ,\tag{18}
$$

where ${ \ddot { q } } \theta$ is derived from the context-conditioned Euler–Lagrange dynamics. We use a fourth-order Runge–Kutta integrator to advance the state:

$$
\begin{array} { r } { \hat { x } _ { t + k + 1 } = \operatorname { R K } 4 \big ( f _ { \theta } , \hat { x } _ { t + k } , z _ { t } , \varDelta t \big ) , \quad k = 0 , \ldots , H - 1 . } \end{array}\tag{19}
$$

The rollout is fully diferentiable, with $z _ { t }$ fixed while the state is recursively updated by the learned dynamics.

## 3.6 Training Objective

HiLNN is trained end-to-end with multi-step rollout supervision. Given predicted trajectory $\hat { \mathcal { X } } _ { t + 1 : t + H }$ and ground truth $\mathscr { X } _ { t + 1 : t + H }$ , the rollout loss is

$$
\mathcal { L } _ { \mathrm { r o l l } } = \frac { 1 } { H } \sum _ { k = 1 } ^ { H } \left( \lambda _ { q } \| \hat { q } _ { t + k } - q _ { t + k } \| _ { 2 } ^ { 2 } + \lambda _ { \dot { q } } \| \hat { \dot { q } } _ { t + k } - \dot { q } _ { t + k } \| _ { 2 } ^ { 2 } \right) ,\tag{20}
$$

where $\lambda _ { q }$ and $\lambda _ { \dot { q } }$ balance position and velocity prediction errors.

When ground-truth velocity at time t is available, we additionally supervise the velocity head:

$$
\begin{array} { r } { \mathcal { L } _ { v 0 } = \| \hat { \dot { q } } _ { t } - \dot { q } _ { t } \| _ { 2 } ^ { 2 } . } \end{array}\tag{21}
$$

This encourages a physically meaningful rollout initialization.

We further add energy regularization for physical consistency. Let $E ( q , \dot { q } )$ denote mechanical energy:

$$
\mathcal { L } _ { E } = \frac { 1 } { H } \sum _ { k = 1 } ^ { H } \left. E ( \hat { q } _ { t + k } , \hat { { q } } _ { t + k } ) - E ( q _ { t + k } , \dot { q } _ { t + k } ) \right. _ { 2 } ^ { 2 } .\tag{22}
$$

The final training objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { r o l l } } + \lambda _ { v 0 } \mathcal { L } _ { v 0 } + \lambda _ { E } \mathcal { L } _ { E } ,\tag{23}
$$

where $\lambda _ { v 0 }$ and $\lambda _ { E }$ control initial velocity supervision and energy consistency. All losses are computed after diferentiable RK4 rollout, enabling joint optimization of the encoder, velocity head, and Lagrangian dynamics.

## 3.7 Implementation Details

Unless otherwise specified, we use history length $L = 8 ,$ , prediction horizon $H =$ 32, and time interval $\varDelta t = 0 . 0 5$ . The encoder is a one-layer GRU with hidden dimension 64, followed by an MLP that outputs a 32-dimensional context. The velocity head is a lightweight MLP taking $[ q _ { t } , z _ { t } ]$ as input.

The context-conditioned mass and potential networks are MLPs with hidden dimension 128 and Tanh activations. Mass positivity is enforced by softplus with $\epsilon = 1 0 ^ { - 3 }$ . Training and evaluation use RK4, and gradients are backpropagated through the full rollout without detaching intermediate states.

We optimize the model using Adam with learning rate $1 0 ^ { - 3 }$ , batch size 256, gradient clipping of 1.0, and early stopping with patience 15 over at most 50 epochs. The default loss weights are

$$
\lambda _ { q } = 1 . 0 , \quad \lambda _ { \dot { q } } = 0 . 1 , \quad \lambda _ { v 0 } = 0 . 1 , \quad \lambda _ { E } = 0 . 0 1 .\tag{24}
$$

All results are evaluated under open-loop rollout using only observed history.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We evaluate HiLNN on three pendulum-based systems: a fixed conservative pendulum, a fixed damped pendulum, and a variable-parameter pendulum with trajectory-dependent dynamics. For all datasets, the model observes only a position history of length $L = 8$ and predicts $H = 3 2$ future steps with $\varDelta t = 0 . 0 5$ . Details are summarized in Table 1.

Table 1. Dataset setup with $L = 8$ $H = 3 2$ , and $\varDelta t = 0 . 0 5$
<table><tr><td>Dataset</td><td>Type</td><td>Train/Val/Test Test win. Params</td><td></td><td>Property</td></tr><tr><td></td><td>Pendulum Fixed conservative</td><td>1000/200/200</td><td>32200 fixed  $l { = } 1 , m { = } 1 , c { = } 0$ </td><td>Conservative</td></tr><tr><td>Damped</td><td>Fixed dissipative</td><td>1000/200/200</td><td>32200 fixed c=0.15</td><td>Decay</td></tr><tr><td>Variable</td><td>Variable params.</td><td>3000/500/500</td><td>80500 sampled  $l , m , c$ </td><td>Heterogeneous</td></tr></table>

Evaluation protocol. All methods follow the same position-only forecasting protocol. At test time, each model observes only the position history and performs open-loop rollout without future ground truth. We report MSE, MAE, Final MSE@32, and Energy MSE to measure trajectory accuracy, long-horizon error accumulation, and physical consistency.

Baselines. We compare HiLNN with representative dynamics models, including LNN [4], HNN [7], Neural ODE [3], MLP-based predictors, and an LNNmultistep variant trained with rollout supervision. These baselines cover both black-box and physics-guided dynamics, enabling direct comparison with the proposed history-informed Lagrangian formulation.

![](images/b16ab51592c874ab2e365dc502b43d5b4a52fd2c496175f9f5005a72421b4649.jpg)  
Fig. 2. Qualitative rollout comparison. Left: ground truth and HiLNN on three datasets (a–c). Right: ground truth and baseline predictions on the fixed pendulum (d–f).

Implementation details. HiLNN uses a one-layer GRU to encode position history into a latent context, which is shared by the initial velocity head and the context-conditioned Lagrangian module. The mass and potential terms are modeled by MLPs, with mass positivity enforced by softplus and a small constant. Unless otherwise specified, training and evaluation use RK4 with gradients propagated through the full rollout horizon. The training configuration is shown in Table 2.

Table 2. Training configuration of HiLNN.
<table><tr><td>Item</td><td>Value Item</td><td></td><td>Value</td></tr><tr><td>History length L</td><td>8</td><td>Prediction horizon H</td><td>32</td></tr><tr><td>Time step ∆t</td><td>0.05</td><td>Encoder</td><td>GRU</td></tr><tr><td>GRU hidden dim.</td><td>64</td><td>Context dim.  $d _ { z }$ </td><td>32</td></tr><tr><td>GRU layers</td><td>1</td><td>Lagrangian MLP hidden</td><td>128</td></tr><tr><td>Mass positivity ε</td><td>10⁻3</td><td>Integrator</td><td>RK4</td></tr><tr><td>Detach between steps</td><td>False</td><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td>10⁻³</td><td>Batch size</td><td>256</td></tr><tr><td>Max epochs</td><td>50</td><td>Early-stop patience</td><td>15</td></tr><tr><td>Gradient clip norm</td><td>1.0</td><td>Loss weights</td><td> $\lambda _ { q } / \lambda _ { \dot { q } } / \lambda _ { v 0 } / \lambda _ { E } = 1 . 0 / 0 . 1 / 0 . 1 / 0 . 0 1$ </td></tr></table>

## 4.2 Main Quantitative Results

Fig. 2 qualitatively compares 32-step open-loop rollouts: panels (a–c) show groundtruth and HiLNN trajectories on the fixed, damped, and variable pendulum datasets, while panels (d–f) contrast LNN, LNN-multistep, and HiLNN on the fixed pendulum, with HiLNN tracking the ground truth most closely. Table 3 reports the overall comparison on the three pendulum systems. HiLNN achieves the best performance across conservative, dissipative, and variable-parameter settings. On the standard pendulum, it reduces the average MSE from 0.279 of

LNN to 0.103, while also obtaining the lowest Final MSE@32 and Energy MSE, showing improved prediction accuracy and long-horizon physical consistency.

The advantage is more pronounced on damped and variable-parameter systems. For the damped pendulum, HiLNN achieves an average MSE of 4.28×10<sup>−3</sup> and reduces Energy MSE from 0.987 to 0.107 compared with LNN, indicating that the context-conditioned damping term captures dissipative dynamics. For the variable-parameter pendulum, HiLNN reduces MSE from 0.894 of LNNmultistep to 0.061 and Final MSE@32 from 2.295 to 0.193, suggesting that the latent context efectively adapts the structured dynamics to trajectory-dependent physical variations.

Overall, these results show that a fixed global Lagrangian model is insuficient under position-only and heterogeneous dynamics. By inferring the missing initial velocity and conditioning Lagrangian components on history, HiLNN provides more accurate and physically coherent long-horizon forecasts.

Table 3. Full quantitative results on the three pendulum systems.
<table><tr><td>Dataset</td><td>Method</td><td>MSE↓</td><td>MAE ↓ Final MSE@32 ↓ Energy MSE ↓</td><td></td><td></td></tr><tr><td rowspan="6">Pendulum</td><td>LNN [4]</td><td>0.279</td><td>0.116</td><td>0.483</td><td>4.222</td></tr><tr><td>LNN [4]-multistep</td><td>0.415</td><td>0.238</td><td>0.857</td><td>5.906</td></tr><tr><td>HNN [7]</td><td>1.646</td><td>0.826</td><td>2.740</td><td>17.512</td></tr><tr><td>Neurai ODE [3]</td><td>0.580</td><td>0.458</td><td>1.355</td><td>9.603</td></tr><tr><td>MLP-one-step</td><td>0.261</td><td>0.156</td><td>0.564</td><td>3.695</td></tr><tr><td>HiLNN</td><td>0.103</td><td>0.109</td><td>0.271</td><td>1.705</td></tr><tr><td rowspan="3">Damped</td><td>LNN [4]</td><td>0.037</td><td>0.104</td><td>0.149</td><td>0.987</td></tr><tr><td>LNN [4]-multistep</td><td>0.042</td><td>0.124</td><td>0.089</td><td>1.338</td></tr><tr><td>HiLNN</td><td>4.28× 10 -3</td><td>0.036</td><td>0.018</td><td>0.107</td></tr><tr><td rowspan="3">Variable</td><td>LNN [4]</td><td>0.960</td><td>0.536</td><td>2.483</td><td>19.516</td></tr><tr><td>LNN-multistep [4]</td><td>0.894</td><td>0.522</td><td>2.295</td><td>26.944</td></tr><tr><td>HiLNN</td><td>0.061</td><td>0.113</td><td>0.193</td><td>1.759</td></tr></table>

## 4.3 Long-Horizon Rollout Analysis

To further analyze error accumulation, we report step-wise rollout errors over the 32-step open-loop horizon. As shown in Fig. 3, HiLNN consistently achieves lower errors on all three systems, especially at medium and long horizons. This shows that the history-informed context improves prediction accuracy and mitigates recursive error growth in open-loop forecasting.

Table 4 reports detailed rollout errors on the conservative pendulum. Although LNN has the lowest one-step error, its error grows quickly over longer horizons. HiLNN achieves lower errors from Step 8 onward and reduces the final-step error from 0.483 to 0.271, showing that velocity inference and historyconditioned Lagrangian dynamics improve long-horizon stability.

![](images/67586b954d5581c08aeff6530f495793e0e969ce67fa3c34d093bcb45449f87d.jpg)

![](images/455c1775e54fd31a65d1acd79f59de584d8bb60aa399705873a76559c31fb2cb.jpg)

![](images/3724a0a2ed6b15a9ab25b69dac6b65e3650acdc4b6739f91f5cea64858864c5e.jpg)

Fig. 3. Step-wise rollout error on the three pendulum systems. HiLNN achieves lower state MSE than LNN and LNN-multistep over the 32-step open-loop prediction horizon.  
![](images/03e9596f4f929a58896def0718968bba5700346c73a5ee4021f0a6fe619c4797.jpg)

![](images/79e935084881624644f0483caace2d0e4e6a183c5c2fe6b433d8b5588b4bc238.jpg)

![](images/aebabe7a14a8ebff299161f9cc7f217f7a18f602da4592cdd252b647baa3392e.jpg)  
Fig. 4. Energy behavior on the damped, standard, and variable-parameter pendulum systems.

Table 4. Rollout error summary on the pendulum system. Values report state MSE at selected forecast steps.
<table><tr><td>Method</td><td>Step 1 Step 8 Step 16 Step 32</td></tr><tr><td></td><td>0.112 0.299 0.483</td></tr><tr><td>LNN [4]  $2 . 1 7 \times 1 0 ^ { - 3 }$ </td><td> $3 . 5 5 \times 1 0 ^ { - 3 }$  0.162 0.402</td></tr><tr><td>LNN [4]-multistep 0.024</td><td>0.857 0.415 1.848 2.740 1.646</td></tr><tr><td>HNN [7] Neural ODE [3]</td><td>0.897</td></tr><tr><td> $8 . 5 1 \times 1 0 ^ { - 3 }$  0.252</td><td>0.504 1.355 0.580 0.103</td></tr><tr><td>HiLNN 0.061</td><td>0.039 0.072 0.271</td></tr></table>

The damped pendulum results in Table 5 further show HiLNN’s advantage under dissipative dynamics. Although LNN has a very small Step-1 error, its error grows rapidly at later steps. HiLNN achieves the lowest errors at Steps 8, 16, and 32, reducing the final-step error from 0.149 to 0.018. This indicates that context-conditioned damping better captures long-term energy decay.

The improvement is most evident in the variable-parameter setting, as shown in Table 6. Since trajectories follow diferent physical parameters, a single global Lagrangian model struggles with long-horizon prediction. HiLNN achieves the best performance at all selected steps and reduces the Step-32 error from 2.483/2.295 to 0.193 compared with LNN/LNN-multistep. This indicates that the latent context captures trajectory-dependent dynamics and enables adaptation across heterogeneous systems.

Table 5. Rollout error summary on the damped pendulum system. Values report state MSE at selected forecast steps.
<table><tr><td>Method</td><td>Step 1</td><td>Step 8</td><td>Step 16</td><td>Step 32</td><td>Avg.</td></tr><tr><td>LNN [4]</td><td> $2 . 0 8 \times 1 0 ^ { - 4 }$ </td><td> $8 . 6 0 \times 1 0 ^ { - 3 }$ </td><td>0.016</td><td>0.149</td><td>0.037</td></tr><tr><td>LNN [4]-multistep</td><td> $6 . 6 2 \times 1 0 ^ { - 3 }$ </td><td>0.027</td><td>0.039</td><td>0.089</td><td>0.042</td></tr><tr><td>HiLNN</td><td> $2 . 6 8 \times 1 0 ^ { - 3 }$ </td><td> $1 . 3 6 \times 1 0 ^ { - 3 }$ </td><td> $2 . 2 1 \times 1 0 ^ { - 3 }$ </td><td>0.018</td><td> $4 . 2 8 \times 1 0 ^ { - 3 }$ </td></tr></table>

Table 6. Rollout error summary on the variable-parameter pendulum system. Values report state MSE at selected forecast steps.
<table><tr><td>Method</td><td>Step 1 Step 8 Step 16 Step 32</td></tr><tr><td></td><td>0.717 2.483</td></tr><tr><td>0.301 0.960</td><td>LNN [4]  $6 . 1 2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>LNN [4]-multistep  $7 . 5 6 \times 1 0 ^ { - 3 }$  HiLNN</td><td>0.350 0.685</td></tr><tr><td> $5 . 9 2 \times 1 0 ^ { - 3 }$ </td><td>2.295 0.894</td></tr><tr><td>0.016</td><td>0.042 0.193 0.061</td></tr></table>

Overall, these analyses show that baseline models may achieve competitive short-term prediction but sufer from rapid error accumulation in long open-loop rollouts. HiLNN stabilizes error growth by using history to infer both the missing initial state and trajectory-specific dynamical context.

## 4.4 Trajectory and Physical Consistency

Beyond aggregate metrics, we further examine trajectory and energy behavior. Fig. 5 shows that HiLNN closely tracks the ground-truth position q and velocity ˙q over 32 steps, whereas baselines show larger phase shifts or amplitude errors, consistent with Sec. 4.3. For physical consistency, Table 7 shows that HiLNN obtains the lowest energy errors at Steps 16 and 32 on the damped pendulum, reducing Test Energy MSE from 0.987 to 0.107 and mean absolute energy error from 0.686 to 0.167. These results suggest that history-informed context improves initialization and rollout dynamics, while context-conditioned damping better captures dissipative energy decay. These results indicate that history-informed context improves state initialization and dynamical evolution, while context-conditioned damping helps capture dissipative energy decay.

The variable-parameter setting further evaluates physical consistency under trajectory-dependent parameters. As shown in Table 7, LNN and LNN-multistep accumulate large energy errors, with Test Energy MSE values of 19.516 and 26.944. In contrast, HiLNN reduces Test Energy MSE to 1.759 and Step-32 energy error from 26.111 to 3.646 compared with LNN. Although not best at Step 1, HiLNN achieves much lower later-step errors, showing that the learned context improves long-term physical consistency under heterogeneous dynamics.

Overall, these results show that HiLNN improves both trajectory accuracy and physical consistency. By following dissipative energy decay and reducing long-horizon energy drift in parameter-varying systems, HiLNN validates the benefit of combining history-informed context with structured Lagrangian dynamics.

Table 7. Energy behavior summary on damped and variable-parameter pendulum systems. For the damped system, the true energy drop is approximately 1.24 over 32 steps on average.
<table><tr><td>Dataset</td><td>Method</td><td>E-MSE@1</td><td>E-MSE@16</td><td>E-MSE@32</td><td>Mean |err|</td><td>Test E-MSE</td></tr><tr><td rowspan="3">Damped</td><td>LNN [4]</td><td> $5 . 3 1 \times 1 0 ^ { - 3 }$ </td><td>0.800</td><td>2.607</td><td>0.686</td><td>0.987</td></tr><tr><td>LNN [4]-multistep</td><td>0.225</td><td>1.277</td><td>2.825</td><td>0.730</td><td>1.338</td></tr><tr><td>HiLNN</td><td>0.085</td><td>0.044</td><td>0.565</td><td>0.167</td><td>0.107</td></tr><tr><td rowspan="3">Variable</td><td>LNN [4]</td><td>0.386</td><td>31.072</td><td>26.111</td><td>2.004</td><td>19.516</td></tr><tr><td>LNN [4]-multistep</td><td>0.462</td><td>44.162</td><td>31.178</td><td>2.073</td><td>26.944</td></tr><tr><td>HiLNN</td><td>0.530</td><td>1.675</td><td>3.646</td><td>0.645</td><td>1.759</td></tr></table>

![](images/d08e4c318677f304ee70e3f2dff5d981667adb33b2e903778b99ae3f7bcc05f7.jpg)  
Fig. 5. Representative trajectory predictions on the three pendulum systems. HiLNN closely follows the ground-truth position q and velocity ˙q over the 32-step prediction horizon.

## 4.5 Ablation Studies

We conduct ablations on energy regularization, rollout training, initial velocity supervision, and history length.

Unless otherwise specified, all ablations are performed on the pendulum setting. Table 8 summarizes the efects of energy regularization, rollout design, and initial velocity supervision. For energy regularization, removing $\lambda _ { E }$ weakens physical consistency, yielding an MSE of 0.118 and an Energy MSE of 3.147, while a small weight improves the accuracy–energy trade-of. Specifically, $\lambda _ { E } =$ 0.01 gives the best overall MSE and Energy MSE, whereas $\lambda _ { E } = 0 . 0 0 1$ achieves the lowest Final MSE@32 but higher Energy MSE. A larger weight, $\lambda _ { E } = 0 . 0 5 ,$ hurts both accuracy and energy consistency, indicating over-regularization; thus, we use $\lambda _ { E } = 0 . 0 1$ by default. For rollout training, the Euler variant with detached states, HiLNN v1a, performs poorly, with an MSE of 0.449 and Final MSE@32 of 1.025. Using RK4 with full backpropagation through time reduces them to 0.118 and 0.261, respectively, and also outperforms the LNN baseline under the same RK4 protocol. This confirms the benefit of diferentiable high-order integration and full-horizon gradient propagation. Finally, removing the auxiliary initial velocity loss keeps rollout MSE similar but increases the initial velocity error from 0.115 to 0.245, showing that $\mathcal { L } _ { v 0 }$ mainly improves the accuracy and interpretability of the inferred initial state. We retain this loss to provide a more physical initialization for open-loop rollout.

Table 8. Ablation summary of HiLNN on the pendulum system, covering energy regularization, rollout training design, and initial velocity supervision. “–” indicates that the metric is not applicable to the corresponding ablation.
<table><tr><td>Ablation</td><td>Setting</td><td>MSE↓</td><td>Final MSE@32 ↓ Energy MSE ↓ Init. Vel. MSE ↓</td><td></td><td></td></tr><tr><td rowspan="4">Energy weight</td><td> $\lambda _ { E } = 0$ </td><td>0.118</td><td>0.261</td><td>3.147</td><td></td></tr><tr><td> $\lambda _ { E } = 0 . 0 0 1$ </td><td>0.116</td><td>0.253</td><td>2.680</td><td></td></tr><tr><td> $\lambda _ { E } = 0 . 0 1$ </td><td>0.103</td><td>0.271</td><td>1.705</td><td></td></tr><tr><td> $\lambda _ { E } = 0 . 0 5$ </td><td>0.261</td><td>0.495</td><td>4.041</td><td></td></tr><tr><td rowspan="3">Rollout design</td><td>HiLNN v1a, Euler, detach</td><td>0.449</td><td>1.025</td><td>9.070</td><td>一</td></tr><tr><td>HiLNN v1b, RK4, full BPTT</td><td>0.118</td><td>0.261</td><td>3.147</td><td>1</td></tr><tr><td>LNN [4], RK4</td><td>0.279</td><td>0.482</td><td>4.222</td><td>一</td></tr><tr><td rowspan="2">Init. velocity</td><td> $\lambda _ { v 0 } = 0 . 1$ </td><td>0.118</td><td>0.261</td><td>3.147</td><td>0.115</td></tr><tr><td> $\lambda _ { v 0 } = 0$ </td><td>0.117</td><td>0.262</td><td>3.318</td><td>0.245</td></tr></table>

## 5 Conclusion

In this paper, we proposed HiLNN, a history-informed Lagrangian framework for long-horizon mechanical forecasting from position-only observations. By inferring a latent context from observed history, HiLNN estimates the missing initial velocity and adaptively conditions the mass, potential, and damping terms of a structured Lagrangian model. Combined with diferentiable RK4 rollout and multi-step supervision, HiLNN preserves mechanical structure while improving long-horizon stability. Experiments on conservative, dissipative, and variable-parameter pendulum systems show that HiLNN achieves more accurate and physically consistent predictions than representative black-box and physicsguided baselines.

## Acknowledgements

The work is supported by the National Natural Science Foundation of China under Grant No. 62076078, the Fundamental Research Funds for the Central Universities under Grant No. 3072024LJ0403, and the CAAI-Huawei MindSpore Open Fund under Grant No. CAAI–XSJLJJ–2020–033A.

## References

1. Battaglia, P.W., Pascanu, R., Lai, M., Rezende, D.J., Kavukcuoglu, K.: Interaction networks for learning about objects, relations and physics. In: Advances in Neural Information Processing Systems. vol. 29, pp. 4502–4510 (2016)

2. Buisson-Fenet, M., Morgenthaler, V., Trimpe, S., Di Meglio, F.: Recognition models to learn dynamics from partial observations with neural odes. Transactions on Machine Learning Research (2023)

3. Chen, R.T.Q., Rubanova, Y., Bettencourt, J., Duvenaud, D.K.: Neural ordinary diferential equations. In: Advances in Neural Information Processing Systems. vol. 31, pp. 6572–6583 (2018)

4. Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., Ho, S.: Lagrangian neural networks. In: ICLR 2020 Workshop on Integration of Deep Neural Models and Diferential Equations (2020)

5. Desai, S.A., Mattheakis, M., Sondak, D., Protopapas, P., Roberts, S.J.: Port-hamiltonian neural networks for learning explicit timedependent dynamical systems. Physical Review E 104(3), 034312 (2021). https://doi.org/10.1103/PhysRevE.104.034312

6. Finzi, M., Wang, K.A., Wilson, A.G.: Simplifying hamiltonian and lagrangian neural networks via explicit constraints. In: Advances in Neural Information Processing Systems. vol. 33, pp. 13880–13889 (2020)

7. Greydanus, S., Dzamba, M., Yosinski, J.: Hamiltonian neural networks. In: Advances in Neural Information Processing Systems. vol. 32, pp. 15353–15363 (2019)

8. Krishnan, R.G., Shalit, U., Sontag, D.: Deep kalman filters. arXiv preprint arXiv:1511.05121 (2015)

9. Li, M., Xiao, Y., Xing, X.: Trajectory prediction methods based on analytical mechanics and graph neural networks. CAAI Transactions on Intelligent Systems 20(6), 1355–1365 (2025). https://doi.org/10.11992/tis.202501020

10. Li, Y., Ding, K., Yang, C., Chen, S.Y., Tian, Y.: Distilling time series foundation models for eficient forecasting. In: IEEE International Conference on Acoustics, Speech and Signal Processing (2026)

11. Li, Y., Ding, K., Yang, C., Wang, H., Wang, H., Duan, H., Liu, J., Tian, Y.: Ddtime: Dataset distillation with spectral alignment and information bottleneck for time-series forecasting. arXiv preprint arXiv:2511.16715 (2025)

12. Li, Y., Dong, J., Liu, J., Koniusz, P., Zeng, H., Yang, C., Liu, J., Tian, Y., Huang, T., Wu, H.: Evolving multimodal models for physical dynamics: A multiobjective neuroevolution approach. IEEE Transactions on Evolutionary Computation (2026). https://doi.org/10.1109/TEVC.2026.3698641

13. Li, Y., Yang, C., Zeng, H., Dong, Z., An, Z., Xu, Y., Tian, Y., Wu, H.: Frequencyaligned knowledge distillation for lightweight spatiotemporal forecasting. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7262–7272 (2025)

14. Lutter, M., Ritter, C., Peters, J.: Deep lagrangian networks: Using physics as model prior for deep learning. In: International Conference on Learning Representations (2019)

15. Nagabandi, A., Kahn, G., Fearing, R.S., Levine, S.: Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning. In: 2018 IEEE International Conference on Robotics and Automation. pp. 7559–7566 (2018). https://doi.org/10.1109/ICRA.2018.8463189

16. Rubanova, Y., Chen, R.T.Q., Duvenaud, D.: Latent odes for irregularly-sampled time series. In: Advances in Neural Information Processing Systems. vol. 32 (2019)

17. Sanchez-Gonzalez, A., Godwin, J., Pfaf, T., Ying, R., Leskovec, J., Battaglia, P.W.: Learning to simulate complex physics with graph networks. In: Proceedings of the 37th International Conference on Machine Learning. vol. 119, pp. 8459–8468 (2020)

18. Sosanya, A., Greydanus, S.: Dissipative hamiltonian neural networks: Learning dissipative and conservative dynamics separately. arXiv preprint arXiv:2201.10085 (2022)

19. Takens, F.: Detecting strange attractors in turbulence. In: Dynamical Systems and Turbulence, Warwick 1980, Lecture Notes in Mathematics, vol. 898, pp. 366–381. Springer, Berlin, Heidelberg (1981). https://doi.org/10.1007/BFb0091924

20. Yildiz, C., Heinonen, M., Lahdesmaki, H.: Ode2vae: Deep generative second order odes with bayesian neural networks. In: Advances in Neural Information Processing Systems. vol. 32 (2019)

21. Zhang, T., Zhai, W., Yann, R., Gao, J., Cao, H., Xing, X.: Floating-body hydrodynamic neural networks. arXiv preprint arXiv:2509.13783 (2025)

22. Zhong, Y.D., Dey, B., Chakraborty, A.: Symplectic ode-net: Learning hamiltonian dynamics with control. In: International Conference on Learning Representations (2020)