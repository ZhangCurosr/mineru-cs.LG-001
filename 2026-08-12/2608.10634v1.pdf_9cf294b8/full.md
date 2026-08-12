# IADD-TR: Intervention-Aware Dynamics Decoupling with Targeted Regularization for Model-Based Reinforcement Learning

Zefeng Liang<sup>§</sup>, Jie Qiao<sup>§</sup>, Ruichu Cai, Senoir Member, IEEE, Weilin Chen, and Zhifeng Hao, Member, IEEE,

This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

Abstract—Model-based reinforcement learning (MBRL), which learns environment dynamics to generate synthetic experience, is a promising approach to sample-efficient decision making. Numerous methods have been developed to improve dynamics prediction and policy optimization for MBRL through uncertainty estimation, model regularization, and conservative value learning. However, these methods typically treat the transition model and critic as monolithic predictors, overlooking the policyinduced data bias. Consequently, action can become entangled with environmental evolution, while uneven action coverage may distort the counterfactual value estimates used for policy improvement. To address this, we propose IADD-TR, a unified framework combining Intervention-Aware Dynamics Decoupling (IADD) and Targeted Regularization (TR). IADD factorizes transitions into an action-intervention stage and an action-free natural evolution stage, using a zero-action anchor to resolve the non-uniqueness of this two-stage factorization for robust generalization. Its latent and state-aligned components are identifiable up to an invertible within-block transformation and pointwise, respectively. For policy learning, we derive TR from the efficient influence function of a replay-state policy-gradient functional. TR augments the critic with an action-density-scaled residual correction and optimizes a targeted loss, yielding doubly robust policy-gradient estimation when either the critic or the replay action density is consistently specified. Extensive experiments on five MuJoCo tasks show that IADD-TR achieves competitive returns with improved sample efficiency.

Index Terms—Model-based reinforcement learning, causal inference, confounding bias, dynamics modeling, actor-critic methods, policy optimization, sample efficiency, targeted regularization.

## I. INTRODUCTION

The pursuit of sample-efficient and generalizable decisionmaking agents has long been a central challenge in reinforcement learning (RL) [1]. Model-based reinforcement learning (MBRL) addresses this challenge by learning environment dynamics to predict the consequences of actions and then using the learned model to generate simulated experience without repeatedly querying the original environment. This capability is particularly valuable when real-world interactions are costly or difficult to obtain, such as in high-fidelity simulators, physical robotics, and human-involved decision-making environments. These advantages have fueled substantial interest in MBRL [2]–[8] and have led to advanced methods such as Model-based Policy Optimization (MBPO) [9], which builds on Dyna-style learning [10] with a learned dynamics model and a Soft Actor-Critic (SAC) agent [11].

![](images/de91054204c280a1fed435ebe3c06bd8fdd9ce0207ec8cebcf3c008efd373e6b.jpg)  
Fig. 1. An example of confounding bias in a car-driving scenario, together with an illustration of intervention-aware dynamics decoupling and counterfactual reasoning for the Q-function.

Despite these successes, two interrelated challenges remain central to MBRL: (1) learning an accurate dynamics model from limited agent-collected data, and (2) learning an effective policy from imperfect model-generated data. At its core, both dynamics model learning and policy optimization rely on estimating complex mappings from data collected by the agent. For model learning, MBRL seeks to learn a one-step transition model $p ( s _ { t + 1 } | s _ { t } , a _ { t } )$ ; for policy learning, actor-critic methods often learn a value function such as $Q _ { \omega } ( s _ { t } , a _ { t } )$ . These formulations, which turn complex dynamical and decisionmaking processes into direct function approximation problems, provide a convenient simplification and work well in many settings. However, they can become error-prone when interactions among states, actions, and subsequent transitions cause policy-generated data to contain confounding bias. Consider the example in Fig. 1(a), in which a self-driving car is approaching a destination. Along the observed trajectories, the policy accelerates primarily near the starting region and brakes as the car approaches the destination. As a result, these policy-generated data would create spurious correlations such that a larger acceleration is correlated with smaller subsequent positions and a smaller acceleration is correlated with larger subsequent positions. While the model might perform well in this specific scenario, it would fail to generalize. This is because the position acts as a confounder, influencing both the action and the resulting state.

This confounding bias corrupts the learning process in two critical ways. First, for model learning, the model p may learn spurious correlations from limited agent-collected data. For instance, it may wrongly associate deceleration with large positional gains, leading to poor generalization in new scenarios. Second, for policy learning, Q-value estimation can suffer from selection bias because the critic is trained only on the actions observed in the agent’s experience. As illustrated in Fig. 1(b) and (d), this state-dependent action coverage makes it difficult to evaluate counterfactual actions, such as the outcome of accelerating at a larger position, thereby hindering the agent’s ability to find a truly optimal policy.

Various approaches have been proposed to mitigate these challenges in model and policy learning. For model learning, techniques such as model ensembles [12], Lipschitz constraints [13], [14], and distribution matching methods [15], [16] have been used to improve the robustness of one-step transition estimation. However, these approaches still treat dynamics learning as a monolithic, black-box mapping problem. Similarly, for policy learning, methods from offline reinforcement learning address selection bias by either constraining the policy to the data distribution [17] or learning conservative Qfunctions [18]. While effective at preventing catastrophic failures, these approaches do not directly address the confounding mechanism that induces biased dynamics and value estimates.

In this work, we focus on confounding bias arising from limited interactive data within the actor-critic framework for model-based reinforcement learning [9], [19]. The preceding analysis suggests that confounding bias is not merely a defect of the collected data but is also amplified by algorithmic choices inherited from monolithic black-box modeling. Such choices can lead to model misspecification and biased estimates from biased data [20]. Instead of treating the dynamics model and the critic as generic functions to be approximated, we deconstruct both learning targets to explicitly account for and mitigate confounding bias.

First, for model learning, we begin with the insight from causal inference that an agent’s action is fundamentally a causal intervention in many environments. For instance, as shown in Fig. 1(c), the action of acceleration is an intervention on the velocity over a short time interval. This perspective motivates our Intervention-Aware Dynamics Decoupling method, which reconstructs the transition process by aligning it with the underlying causal process present in many environments: (1) the action’s immediate intervention, and (2) the environment’s subsequent natural evolution. To model this, we use an intermediate state to capture the direct consequences of the intervention. The environment then transitions naturally from this intermediate state. Crucially, this two-stage process can be anchored in many environments by a reference zero action, namely an action that applies no direct intervention and leaves the intermediate observable state unchanged. By leveraging this zero-action condition, we can learn the natural evolution component and then separate the effect of nonzero interventions from the subsequent environment dynamics. This causal decomposition helps prevent the dynamics model from explaining action-induced changes as natural evolution, thereby reducing confounding bias and improving sample efficiency.

Second, for policy learning in a general actor-critic framework for MBRL, the Q-function is commonly treated as a black-box approximation learned via Bellman error minimization. Learning this approximation is difficult from confounded data, and more importantly, it is not the ultimate target of policy optimization. In its role as a critic, the Q-function should provide the information needed for an accurate and less biased estimate of the policy gradient. The standard Bellman-error objective can therefore be mismatched with the final goal of improving the actor through its gradient. Motivated by targeted learning in causal inference [21], we introduce Targeted Regularization (TR) to align critic learning more directly with the policy-gradient target. Rather than only minimizing the prediction error of the Q-function, our approach regularizes the critic so that it produces a debiased and efficient estimate of the target parameter, guiding the policy update in a more reliable direction.

This work makes four key contributions toward addressing confounding bias in MBRL:

1) We propose an intervention-aware dynamics decoupling model that separates action-induced intervention effects from the natural evolution of the environment.

2) We develop a zero-action-based identification strategy for the two-stage decoupling, which anchors the naturalevolution component and enables the effect of nonzero interventions to be separated from subsequent environment dynamics.

3) We introduce targeted regularization for Q-function learning to reduce the impact of confounding bias on policy-gradient estimation.

4) We provide theoretical and empirical validation demonstrating the effectiveness of the proposed method.

## II. RELATED WORK

Our work is related to several established fields, including model-based reinforcement learning (MBRL), offline reinforcement learning, policy evaluation, and causal inference. The core challenge in MBRL is learning an accurate dynamics model and an effective policy from limited, agentcollected data [7]. Early work focused on local linear models [5], [22] or Gaussian processes [4], but in more complex environments, higher capacity models are required, such as neural networks [23]. To further combat model error from limited data, a primary strategy is to use probabilistic models or ensemble methods to control uncertainty [12], [24]–[26]. This uncertainty is then used to regularize policy updates, preventing the agent from exploiting areas where the model is inaccurate, as seen in Dyna-style algorithms like MBPO [9]. However, these approaches still treat the dynamics as a monolithic, black-box mapping, leaving them vulnerable to learning spurious correlations when the collected data contains confounding bias.

The confounding bias issue is also closely related to the challenge of out-of-distribution (OOD) actions in offline reinforcement learning (offline RL) [27]. In offline RL, agents learn from a fixed dataset, facing the OOD problem when evaluating OOD actions of the Q-function [17]. To mitigate this, common methods either constrain the learned policy to remain close to the data-generating behavior policy [17] or learn a conservative Q-function that explicitly penalizes OOD actions [18], [28], [29]. Model-based offline methods, such as MOPO [30] and MOREL [31], also adopt a conservative stance, using model uncertainty to construct a pessimistic MDP that penalizes policy learning in risky regions. While effective at preventing catastrophic failures, these approaches are inherently conservative. Our approach, in contrast, aims to debias the learning process itself.

For policy learning, our work addresses the confounding bias that corrupts the critic’s estimates. This concern with the reliability of value estimates from biased data is also related to off-policy evaluation (OPE), which aims to accurately estimate a policy’s value under distributional shift [32]. The OPE field has long bridged RL and causal inference by treating the evaluation of a policy as a counterfactual quantity estimation problem [33], [34]. Typical methods include Direct Methods (DM) (analogous to G-computation [35]), which have low variance but suffer from high bias if the model is misspecified, and Importance Sampling (IS) [36] (analogous to inverse propensity scoring [37]), which is unbiased but suffers from high variance [38]. This trade-off led to Doubly Robust (DR) estimators [39]–[41] and the targeted learning estimator [42], combining DM and IS to reduce both bias and variance.

While sharing this causal perspective, we re-evaluate the goal of the critic. Unlike OPE, which focuses on estimating policy value optimally, we argue that in the actor-critic framework, the critic should guide the actor by estimating a value function (such as the Q-function) precisely for estimating an unbiased and efficient policy gradient, which is the primary optimization target. Inspired by Targeted Learning and Targeted Maximum Likelihood Estimation (TMLE) [21], [43], we introduce targeted regularization. This approach optimizes the critic to be an efficient estimator for the policy gradient via the efficient influence function (EIF) of the target parameter, thereby correcting the bias induced by confounding.

## III. BACKGROUND

In this section, we introduce the basic notation for modelbased reinforcement learning, off-policy replay data, and the policy-gradient target used throughout the paper.

We consider a Markov decision process (MDP) represented by the tuple $( S , \mathcal { A } , p , r , \gamma , \rho _ { 0 } )$ . Here, S denotes the state space, and $\mathcal { A }$ denotes the action space; for vector-valued states, we write $s _ { t } ~ \in ~ \mathbb { R } ^ { d _ { s } }$ , where $d _ { s }$ denotes the state dimension. The transition kernel $p ( s ^ { \prime } \mid s , a )$ describes the environment dynamics, defining the transition probability to state $s ^ { \prime }$ given state s and action $a ; r ( s , a )$ is the reward function; $\gamma \in ( 0 , 1 )$ is the discount factor; and $\rho _ { 0 }$ is the initial state distribution. The goal of reinforcement learning is to learn a policy $\pi ( a \mid s )$ that maximizes the expected cumulative discounted reward:

$$
\eta [ \pi ] = \mathbb { E } _ { \tau \sim \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t } , a _ { t } ) \right] ,\tag{1}
$$

where the trajectory $\tau ~ = ~ ( s _ { 0 } , a _ { 0 } , s _ { 1 } , \dots )$ is generated by rolling out the policy in the environment, with $s _ { 0 } \sim \rho _ { 0 }$ and $a _ { t } \sim \pi ( \cdot \mid s _ { t } )$ . The Q-function represents $\eta [ \pi ]$ conditioned on a specific state-action pair and is given by

$$
Q ^ { \pi _ { \theta } } ( s _ { t } , a _ { t } ) = \mathbb { E } _ { \tau \sim \pi _ { \theta } } \left[ \sum _ { k = t } ^ { \infty } \gamma ^ { k - t } r ( s _ { k } , a _ { k } ) \Bigg | s _ { t } , a _ { t } \right] .\tag{2}
$$

In off-policy actor-critic learning, the replay buffer contains transitions collected by historical behavior policies. We denote the replay distribution by D and its state marginal by $\mathcal { D } _ { S }$ . For a replay action a generated by its associated behavior policy $\pi _ { \beta } ( \cdot \mid s )$ , we log the density

$$
e _ { \pi _ { \beta } } = \pi _ { \beta } ( a \mid s ) .\tag{3}
$$

Accordingly, a replay transition is written as $( s , a , e _ { \pi _ { \beta } } , r , s ^ { \prime } ) \sim$ D. When computing its Bellman target, we sample $a ^ { \prime } \sim \pi _ { \theta } ( \cdot \ |$ $s ^ { \prime } )$ from the current policy and evaluate the density of that sampled action as

$$
e _ { \pi _ { \theta } } = \pi _ { \theta } ( a ^ { \prime } \mid s ^ { \prime } ) .\tag{4}
$$

Thus, $e _ { \pi _ { \beta } }$ is the logged density of the replay action under its behavior policy, whereas $e _ { \pi _ { \theta } }$ is the density of the action under the current policy.

To optimize the policy using the learned model, in this work, we focus on the Dyna-style MBRL [10] with a soft actor-critic agent [11], which is widely adopted in state-of-the-art methods such as MBPO [9]. One of our central objects of interest is the policy-gradient target, which serves as the foundation for policy optimization via the policy gradient theorem [44]. We denote this target by $\psi _ { \mathrm { p g } }$ , which is expressed as

$$
\begin{array} { r } { \psi _ { \mathrm { p g } } = \nabla _ { \theta } \eta [ \pi _ { \theta } ] = \mathbb { E } _ { \underset { a \sim \pi _ { \theta } ( \cdot | s ) } { s \sim d ^ { \pi _ { \theta } } } } \left[ \nabla _ { \theta } \log \pi _ { \theta } ( a \mid s ) \cdot Q ^ { \pi _ { \theta } } ( s , a ) \right] , } \end{array}\tag{5}
$$

where $d ^ { \pi _ { \theta } } ( s )$ is the discounted state visitation distribution under $\pi _ { \theta } .$ , and $Q ^ { \pi _ { \theta } } ( s , a )$ is the action-value function. In this paradigm, the actor and critic are typically implemented as neural networks for estimating the policy $\pi _ { \theta } ( a \mid s )$ with parameter θ and the value function $Q _ { \omega } ( s , a )$ with parameter ω, respectively.

## IV. METHODOLOGY

In this section, we describe the Intervention-Aware Dynamics Decoupling with Targeted Regularization (IADD-TR) framework in the context of model-based reinforcement learning. As illustrated in Fig. 2(a), IADD-TR follows an iterative model-based training pipeline. The agent first interacts with the environment to collect real transitions, which are stored in the environment replay buffer and used for both dynamics model learning and policy optimization. The learned IADD dynamics model then generates synthetic transitions, which are stored in the model replay buffer and further used to augment policy learning. During policy learning, targeted regularization is incorporated into the critic update to mitigate selection bias in value estimation. Overall, the proposed framework is designed to address two sources of error that arise during training: model learning bias and policy learning bias.

![](images/e93ade0ce9a9f7619030e334aee44994257d9be0314510399855e7e216e94f4b.jpg)  
Fig. 2. Illustration of the proposed IADD-TR method. The framework consists of two key components: IADD and targeted regularization (TR). (a) In the overall training pipeline, the agent interacts with the environment to collect real transitions, which are used to train the dynamics model and to update the policy. The learned model further generates synthetic transitions for model-based policy optimization. (b) IADD adopts a two-stage dynamics model, which decomposes the transition prediction process into an intermediate-stage prediction and a final reward/next-state prediction. (c) TR modifies the critic learning obiective by introducing an adiusted critic and a targeted regularization term, improving the robustness of value estimation under imperfect model rollouts.

## A. Addressing Model Learning Bias

A key challenge in dynamics model learning is that actioninduced changes and environment-induced changes are often entangled in the collected transition data, which can introduce confounding bias into transition prediction. To address this issue, we do not model the transition as a monolithic mapping from $\left( { { s _ { t } } , { a _ { t } } } \right)$ to $s _ { t + 1 }$ . Instead, as illustrated in Fig. 2(b), we explicitly decompose each one-step transition into an actionintervention stage and a natural natural evolution stage. This intervention-aware decomposition makes the action effect explicit, thereby reducing its confounding with natural evolution during dynamics learning.

Formally, we introduce an intermediate representation $\tilde { s } _ { t } =$ $( \tilde { s } _ { t , o } , \tilde { s } _ { t , l } )$ to characterize the post-intervention outcome induced by the action. The variable $\tilde { s } _ { t , o } \in \mathbb { R } ^ { d _ { s } }$ denotes the observable post-intervention state. It is designed to have the same dimensionality and semantic structure as the original state $s _ { t } ,$ $\mathrm { i } . \mathrm { e } . , d _ { o } = d _ { s }$ , and is designed to capture the direct causal effect of the action. Moreover, to retain additional action-dependent information needed by the subsequent evolution model, we introduce an auxiliary latent component $\tilde { s } _ { t , l } \in \mathbb { R } ^ { d _ { l } }$ . Together, $\tilde { s } _ { t , o }$ and $\tilde { s } _ { t , l }$ form the post-intervention representation used for predicting the next state. We write the action-intervention stage as the deterministic structural mechanism

$$
\tilde { s } _ { t , o } = p _ { \mathrm { a c t } , o } ( s _ { t } , a _ { t } ) , \qquad \tilde { s } _ { t , l } = p _ { \mathrm { a c t } , l } ( s _ { t } , a _ { t } , \varepsilon _ { t } ) ,\tag{6}
$$

where $p _ { \mathrm { a c t } , o }$ and $p _ { \mathrm { a c t } , l }$ are the observable and latent components of $p _ { \mathrm { a c t } } .$ , respectively, and $\varepsilon _ { t }$ denotes exogenous noise. Thus, the mechanism is deterministic given $\left( { { s _ { t } } , { a _ { t } } , { \varepsilon _ { t } } } \right)$ , while the randomness in $\varepsilon _ { t }$ induces a conditional density for $\tilde { s } _ { t , l }$ We assume that this latent conditional density factorizes as

$$
p _ { l } \big ( \tilde { s } _ { t , l } \ | \ s _ { t } , a _ { t } \big ) = \prod _ { i = 1 } ^ { d _ { l } } p _ { l , i } \big ( \tilde { s } _ { t , l , i } \ | \ s _ { t } , a _ { t } \big ) ,\tag{7}
$$

where $\tilde { s } _ { t , l } = ( \tilde { s } _ { t , l , 1 } , \ldots , \tilde { s } _ { t , l , d _ { l } } )$ . The natural evolution stage is represented by the deterministic mechanism $p _ { \mathrm { e n v } }$ , and the resulting one-step transition model is the composition

$$
s _ { t + 1 } = p _ { \mathrm { e n v } } ( p _ { \mathrm { a c t } , o } ( s _ { t } , a _ { t } ) , p _ { \mathrm { a c t } , l } ( s _ { t } , a _ { t } , \varepsilon _ { t } ) ) .\tag{8}
$$

Note that the implementation also includes the rewardprediction head shown in Fig. 2(b).

While this two-stage formulation separates the transition, the intermediate representation $( \tilde { s } _ { t , o } , \tilde { s } _ { t , l } )$ is strictly latent and not directly observed in the collected data. As a result, there is no guarantee that the learned intermediate representation corresponds to the intended causal post-intervention state, which can in turn compromise generalization performance.

To anchor the intermediate representation in physical reality and study its identifiability, we introduce a reference zero action $a _ { 0 }$ . Intuitively, a zero action represents the absence of any direct physical intervention. Under $a _ { 0 } .$ , the observable state should remain unchanged immediately after the action-intervention stage. Formally, we define the zero-actionanchored IADD model class by the condition $p _ { \mathrm { a c t } , o } ( s _ { t } , a _ { 0 } ) =$ $s _ { t } .$ This zero-action condition aligns the observable postintervention coordinates with the original state space on the zero-action support. Where zero-action data cover the relevant reachable support, the same alignment extends to intermediate representations reached under nonzero actions.

This equality is enforced by construction rather than through an auxiliary loss. Whenever the input action equals $a _ { 0 } .$ , the action-intervention stage directly copies $s _ { t }$ into the observable coordinates of its intermediate output, so that $\tilde { s } _ { t , o } = s _ { t }$ exactly. The two stages are otherwise trained jointly using the standard one-step prediction objective

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { m o d e l } } = \mathbb { E } _ { ( s _ { t } , a _ { t } , s _ { t + 1 } ) \sim \mathcal { D } } \left[ \left. \hat { s } _ { t + 1 } - s _ { t + 1 } \right. _ { 2 } ^ { 2 } \right] . } \end{array}\tag{9}
$$

Thus, the exact zero-action equality is part of the IADD model-class definition and introduces neither an additional identifiability assumption nor a weighting hyperparameter in the learning objective.

With this anchor, we investigate the identifiability of the post-intervention representations by examining the uniqueness of the learned variables. To formalize this, let $( p _ { \mathrm { a c t } } , p _ { \mathrm { e n v } } , \tilde { s } _ { t , o } , \tilde { s } _ { t , l } )$ denote the true parameterization of the mechanisms and intermediate variables. We then consider an observationally equivalent alternative, denoted with bars $( \bar { p } _ { \mathrm { a c t } } , \bar { p } _ { \mathrm { e n v } } , \bar { s } _ { t , o } , \bar { s } _ { t , l } )$ . Identifiability asks how strictly the equality of the observed transition distributions constrains the relationship between these two intermediate representations under the same marginal distribution. Since the observable component is grounded by the state of zero-action, we expect an exact pointwise recovery, such that $\bar { s } _ { t , o } = \tilde { s } _ { t , o } .$ . Conversely, for the unobserved latent component, we seek identifiability up to a permutation and component-wise invertible transformations.

The following theorem formalizes this pointwise recovery guarantee in the population limit.

Theorem 1. (Pointwise Identifiability of $\tilde { s } _ { t , o } . )$ Consider two parameterizations in the zero-action-anchored IADD model class that induce the same conditional one-step transition distribution in Eq. (8) for every reachable state-action context. Suppose the following condition holds:

• A1 (Injectivity and support inverse): The deterministic natural evolution mechanisms $p _ { \mathrm { e n v } }$ and $\bar { p } _ { \mathrm { e n v } }$ are injective on their reachable intermediate supports, admit continuous inverses on their common reachable nextstate support, and are twice continuously differentiable on these supports.

Then the observable post-intervention state is pointwise identifiable for every reachable $( s _ { t } , a _ { t } )$ whose conditional next-state support is covered by zero-action contexts:

$$
\begin{array} { r } { \bar { s } _ { t , o } ( s _ { t } , a _ { t } ) = \tilde { s } _ { t , o } ( s _ { t } , a _ { t } ) . } \end{array}
$$

The theorem shows that the hard zero-action anchor makes the observable post-intervention state pointwise identifiable wherever zero-action data cover the relevant reachable support. In population terms, this requires sufficiently broad zeroaction support. With finite data, it motivates collecting a sufficiently large and diverse set of zero-action transitions to approximate that coverage. Intuitively, zero-action transitions fix the observable coordinates, and the invertibility in A1 extends this calibration to nonzero-action states within the covered support.

We next consider the latent component. Once Theorem 1 has fixed the observable component pointwise, the remaining task is to identify the latent coordinates. The following theorem uses the factorized action-intervention distribution and its variation across state-action contexts to establish componentwise identifiability.

Theorem 2. (Component-wise Identifiability of $\tilde { s } _ { t , l } . )$ On the zero-action-covered reachable support where Theorem 1 establishes pointwise identifiability, suppose the factorized action-intervention model in Eq. (7) holds. For every fixed reachable value $s _ { o } ^ { \prime }$ of the observable post-intervention state, suppose further that:

• B1 (Regularity): The conditional latent component densities are positive and twice continuously differentiable on connected supports that locally contain product neighborhoods.

• B2 (Sufficient variability): State-action contexts producing that observable state induce sufficiently diverse variations in the conditional latent distributions.

The precise full-rank variability condition is given in $A p \cdot$ pendix D. Then there exist a permutation $\sigma _ { l , s _ { o } ^ { \prime } }$ and onedimensional diffeomorphisms $h _ { 1 , s _ { o } ^ { \prime } } , \ldots , h _ { d _ { l } , s _ { o } ^ { \prime } }$ such that

$$
\begin{array} { r } { \bar { s } _ { t , l , j } = h _ { j , s _ { o } ^ { \prime } } \Big ( \tilde { s } _ { t , l , \sigma _ { l , s _ { o } ^ { \prime } } ( j ) } \Big ) , \qquad j = 1 , \dotsc , d _ { l } . } \end{array}\tag{10}
$$

Once Theorem 1 has identified the observable component, component-wise identification of the latent component requires no additional zero-action information. It instead follows from B1 and B2, which are standard conditions in auxiliaryvariable nonlinear ICA [45]. B1 imposes smoothness and support regularity, while B2 requires enough variation across state-action contexts to distinguish the latent factors. Under these conditions, the latent coordinates are identifiable up to permutation and one-dimensional invertible transformations.

Together, the two theorems identify the observable component exactly and the latent component componentwise. The proofs are given in Appendices C and D.

## B. Addressing Policy Learning Bias

Beyond model-learning bias, confounding can also induce policy-learning bias. To see this, in the actor-critic framework, the critic $Q _ { \omega } ( s , a )$ learns from a replay buffer

$( s , a , e _ { \pi _ { \beta } } , r , s ^ { \prime } ) \sim \mathcal { D }$ collected by historical behavior policies, and is trained by minimizing the temporal-difference (TD) loss:

$$
\mathcal { L } _ { \mathrm { T D } } ( \omega ) = \mathbb { E } _ { \mathcal { D } } \left[ \left( Q _ { \omega } ( s , a ) - y \right) ^ { 2 } \right] ,\tag{11}
$$

where

$$
y = r + \gamma \mathbb { E } _ { a ^ { \prime } \sim \pi _ { \theta } ( \cdot | s ^ { \prime } ) } \bigl [ Q _ { \omega ^ { - } } ( s ^ { \prime } , a ^ { \prime } ) \bigr ]\tag{12}
$$

is the Bellman target. Because replay actions are sampled under the behavior density $\pi _ { \beta } ( a \mid s )$ , the collected distribution can be biased in the sense that some actions can be observed more frequently than others for the same state. As a result, the standard TD update forces the critic to fit high-density actions more accurately while leaving the value estimates for rare or counterfactual actions highly erroneous. This uneven accuracy is critical because the critic guides the actor through the policy gradient, which requires reliable estimates for counterfactual actions—the actions most affected by confounding bias. To formally investigate the nature of this bias, we evaluate the policy-gradient target directly rather than focusing solely on the Bellman loss. We formulate this target with respect to the replay state marginal $\mathcal { D } _ { S }$ as follows:

$$
\begin{array} { c } { { \psi _ { \mathrm { p g } } ^ { \mathcal { D } } = \mathbb { E } _ { \mathbf { \Phi } _ { a \sim \mathcal { D } _ { S } } } \left[ Q ^ { \pi _ { \theta } } ( s , a ) g _ { \theta } ( s , a ) \right] , } } \\ { { a \sim \pi _ { \theta } ( \cdot | s ) } } \\ { { g _ { \theta } ( s , a ) = \nabla _ { \theta } \log \pi _ { \theta } ( a \mid s ) . } } \end{array}\tag{13}
$$

In practice, the actor update uses the learned critic $Q _ { \omega }$ rather than the ideal action-value function $Q ^ { \pi _ { \theta } }$ , yielding the empirical plug-in estimate:

$$
\begin{array} { r } { \widehat { \psi } _ { \mathrm { p g } } ^ { D } = \mathbb { E } _ { \underset { a \sim \pi _ { \theta } ( \cdot | s ) } { s \sim \mathcal { D } _ { S } } } \left[ Q _ { \omega } ( s , a ) g _ { \theta } ( s , a ) \right] . } \end{array}\tag{14}
$$

The gap $\widehat { \psi } _ { \mathrm { p g } } ^ { D } - \psi _ { \mathrm { p g } } ^ { D }$ characterizes the policy learning bias. To rigorously trace this gap, we adopt a causal and semiparametric perspective, analyzing the first-order sensitivity of the target parameter $\psi _ { \mathrm { p g } } ^ { \mathcal { D } }$ under the replay distribution D. This sensitivity is mathematically represented by the efficient influence function (EIF). Let $\dot { \varphi } _ { \mathrm { p g } } ^ { \mathcal { D } }$ denote the EIF of $\psi _ { \mathrm { p g } } ^ { \mathcal { D } }$ . Under standard regularity conditions, the leading first-order behavior of the bias satisfies:

$$
\sqrt { n } \{ \widehat { \psi } _ { \mathrm { p g } } ^ { D } - \psi _ { \mathrm { p g } } ^ { D } \} = \frac { 1 } { \sqrt { n } } \sum _ { i = 1 } ^ { n } \varphi _ { \mathrm { p g } } ^ { D } ( o _ { i } ) + o _ { P } ( 1 ) ,\tag{15}
$$

where $\begin{array} { r c l } { o _ { i } } & { = } & { \left( s _ { i } , a _ { i } , y _ { i } ^ { \pi _ { \theta } } \right) } \end{array}$ represents replay samples augmented with the ideal one-step Bellman variable $\begin{array} { r l } { y ^ { \pi _ { \theta } } } & { { } = } \end{array}$ $r + \gamma Q ^ { \pi _ { \theta } } ( s ^ { \prime } , a ^ { \prime } )$ . The explicit formulation of the EIF formally reveals how the data-generating mechanism dictates this bias, as detailed in the following theorem.

Theorem 3. Consider the replay-state policy-gradient functional $\psi _ { \mathrm { p g } } ^ { \mathcal { D } }$ in Eq. (13), where the state marginal is $\mathcal { D } _ { S }$ and the replay action density is $\pi _ { \beta } ( a \mid s )$ . Treating π<sub>θ</sub> and g<sub>θ</sub> as fixed when taking the pathwise derivative, under the nonparametric replay model for $( s , a , y ^ { \pi _ { \theta } } )$ , the efficient influence function is

$$
\begin{array} { r l r } & { } & { \varphi _ { \mathrm { p g } } ^ { \mathcal { D } } ( o ) = \mathbb { E } _ { \tilde { a } \sim \pi _ { \theta } ( \cdot | s ) } \Bigg [ \Bigg ( \frac { \delta ( a - \tilde { a } ) } { \pi _ { \beta } ( \tilde { a } \mid s ) } ( y ^ { \pi _ { \theta } } - Q ^ { \pi _ { \theta } } ( s , \tilde { a } ) ) } \\ & { } & { \qquad + Q ^ { \pi _ { \theta } } ( s , \tilde { a } ) \Bigg ) g _ { \theta } ( s , \tilde { a } ) \Bigg ] - \psi _ { \mathrm { p g } } ^ { \mathcal { D } } , } \end{array}\tag{16}
$$

where $\delta ( a - \tilde { a } )$ denotes the Dirac delta centered at the replay action a.

Theorem 3 formally uncovers the source of the policy learning bias, where the inverse-probability term $1 / \pi _ { \beta } ( { \tilde { a } } \mid s )$ in Eq. (16) amplifies the critic’s residual error for actions rarely observed in the replay buffer. Since standard TD learning (Eq. (11)) is blind to this gradient-specific sensitivity, it cannot eliminate the bias on its own.

To mitigate this issue, motivated by targeted learning in the causal inference literature [21], we develop a targeted regularization framework. By constructing an adjusted critic that targets the EIF score equation, we can systematically neutralize the magnified errors on low-propensity actions, effectively resolving the policy learning bias at its source. As illustrated in Fig. 2(c), we construct an adjusted critic that augments the baseline critic $Q _ { \omega }$ with a density-scaled residual term:

$$
Q _ { \omega , \xi } ^ { \mathrm { a d j } } ( s , a , e _ { \pi _ { \beta } } ) = Q _ { \omega } ( s , a ) + \frac { \epsilon _ { \xi } ( a ) } { e _ { \pi _ { \beta } } } ,\tag{17}
$$

where $e _ { \pi _ { \beta } }$ is the paired action-selection score defined in Eq. (3), and $\epsilon _ { \xi } ( a )$ is a learnable residual correction parameterized by $\xi .$ . The same adjusted form is used for the target network:

$$
y ^ { \mathrm { a d j } } = r + \gamma Q _ { \omega ^ { - } , \xi ^ { - } } ^ { \mathrm { a d j } } ( s ^ { \prime } , a ^ { \prime } , e _ { \pi _ { \theta } } ) ,\tag{18}
$$

where $a ^ { \prime }$ is the target action and $e _ { \pi _ { \theta } }$ is its paired actionselection score from $\pi _ { \boldsymbol { \theta } } .$

We learn the residual correction $\epsilon _ { \xi } ,$ while keeping the baseline critic parameters ω fixed, by minimizing the targeted regularization loss

$$
\begin{array} { r } { \mathcal { R } _ { \mathrm { T R } } ( \xi ) = \mathbb { E } _ { \mathcal { D } } \left[ \left( Q _ { \omega , \xi } ^ { \mathrm { a d j } } ( s , a , e _ { \pi _ { \beta } } ) - y ^ { \mathrm { a d j } } \right) ^ { 2 } \right] . } \end{array}\tag{19}
$$

The overall critic objective combines the standard TD loss and the targeted regularization term:

$$
\mathcal { L } _ { \mathrm { I A D D - T R } } ( \omega , \boldsymbol { \xi } ) = \mathcal { L } _ { \mathrm { T D } } ( \omega ) + \beta \mathcal { R } _ { \mathrm { T R } } ( \boldsymbol { \xi } ) ,\tag{20}
$$

where $\beta > 0$ controls the strength of the targeted regularization. We use the adjusted critic in the policy-gradient plug-in estimator

$$
\widehat { \psi } _ { \mathrm { p g } } ^ { D , \mathrm { T R } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { \widetilde { a } \sim \pi _ { \boldsymbol { \theta } } ( \cdot | \boldsymbol { s } _ { i } ) } \left[ Q _ { \omega , \xi } ^ { \mathrm { a d j } } ( \boldsymbol { s } _ { i } , \widetilde { \boldsymbol { a } } ) g _ { \boldsymbol { \theta } } ( \boldsymbol { s } _ { i } , \widetilde { \boldsymbol { a } } ) \right] .\tag{21}
$$

The following result summarizes the statistical role of this estimator; its empirical EIF construction, regularity conditions, and proof are deferred to Appendix F.

Theorem 4. (Double robustness of targeted policy-gradient estimation.) Under the overlap, moment, and empiricalprocess conditions stated in Appendix $F ,$ and provided that the fluctuation update asymptotically solves the corresponding EIF score equation, the estimator in Eq. (21) is consistent if either the targeted critic is consistent for $Q ^ { \pi _ { \theta } }$ or the replay action-density estimate is consistentfor the true replay density. In either regime,

$$
\left. \widehat { \psi } _ { \mathrm { p g } } ^ { D , \mathrm { T R } } - \psi _ { \mathrm { p g } } ^ { D } \right. = o _ { P } ( 1 ) .\tag{22}
$$

Here double robustness refers to the critic and replay action density; the fluctuation parameter is only a device for enforcing the targeted score equation. Because the behaviorpolicy density is logged with each replay action in our setting, the replay-density branch does not require fitting an additional propensity model. This property makes the actor update less sensitive to critic misspecification under uneven replay-action coverage.

## V. EXPERIMENTS

In this section, we evaluate the proposed method on standard MuJoCo continuous-control benchmarks [46] and use two controlled diagnostics to test the mechanisms behind its model and policy-learning components. The experiments are organized around four questions: Q1. Does IADD-TR improve sample efficiency and final control performance relative to representative model-free and model-based baselines? Q2. Do the two-stage dynamics model and targeted regularization contribute distinct and complementary performance gains? Q3. Does the zero-action-anchored model recover the simulatordefined post-intervention state without sacrificing next-state prediction accuracy? Q4. Does the targeted critic yield a policy-gradient estimate that is more directionally accurate relative to a finite-sample Monte Carlo (MC) reference?

## A. Experiment Settings

We conduct experiments on five widely used Mu-JoCo continuous-control benchmarks: HalfCheetah, Hopper, Walker2d, Ant, and Humanoid. All benchmark runs use Mu-JoCo 2.0.0. These environments cover locomotion tasks with different levels of contact dynamics, control complexity, and long-horizon stability requirements. To isolate the identifiability mechanism of the two-stage model, we additionally construct a controlled synthetic dynamics environment in which the simulator explicitly generates and records the observable post-intervention state. The toy system has state $\boldsymbol { s } _ { t } = ( x _ { t } , v _ { t } )$ and scalar action $a _ { t } ;$ the action first changes velocity, producing $\tilde { s } _ { t , o } ,$ and an action-independent damped-oscillator ODE then maps $\tilde { s } _ { t , o }$ to $s _ { t + 1 }$ . The construction satisfies the exact zero-action anchor $\tilde { s } _ { t , o } ~ = ~ s _ { t }$ when $a _ { t } ~ = ~ 0$ . The simulatorrecorded $\tilde { s } _ { t , o }$ is used only for evaluation, not for fitting the recovery model; the full environment definition and protocol are given in Appendix B.

We compare our method with both model-free and modelbased reinforcement learning baselines. For model-free reinforcement learning, we use SAC [11] and PPO [47]. SAC is an off-policy actor-critic method that optimizes an entropy-regularized objective, while PPO is a widely used on-policy policy-gradient method based on clipped policy updates. For model-based reinforcement learning, we consider SLBO [48] and MBPO [9]. SLBO performs model rollouts from the initial-state distribution, whereas MBPO generates short-horizon model rollouts from states visited by previous policies.

Zero-action collection is enabled only for IADD-TR; SAC, PPO, SLBO, and MBPO retain their original action-sampling procedures. At each IADD-TR environment interaction, with probability $\lambda _ { 0 } ~ = ~ 0 . 1$ , we replace the action proposed by SAC with the all-zero vector $a _ { 0 } \ = \ \mathbf { 0 }$ before executing it;

otherwise, the SAC action is retained. The resulting real zeroaction transitions, including their observed rewards and next states, are used only to train a separate zero-action dynamics model. This model then generates zero-action transitions used exclusively for IADD dynamics-model training, where they support the anchor $\tilde { s } _ { t , o } ~ = ~ s _ { t }$ . Neither the real nor modelgenerated zero-action transitions are added to the actor or critic replay buffers, including the replay data used for TR. Because action replacement occurs within the scheduled interactions, rather than through additional data collection, all methods use the same real-environment interaction budget. Thus, the zeroaction mechanism neither alters the baseline replay buffers nor gives IADD-TR access to additional real transitions; it is evaluated as an internal component of the proposed method.

For IADD-TR, we fix the two additional settings introduced by the proposed components across all MuJoCo environments: the real zero-action collection ratio is set to 0.1, and the targeted-regularization weight is set to $\beta = 1$ . The remaining training protocol is kept aligned with MBPO, including the epoch length, model-rollout schedule, learning rates, and evaluation frequency; the detailed settings are reported in Appendix A. For all MuJoCo methods, we report results across five independent runs with different random seeds. We evaluate the policy once per epoch, corresponding to every 1,000 environment steps, and report the mean return with standard deviation. The controlled synthetic dynamics study uses a fixed three-member ensemble to isolate the effect of the zero-action ratio; its complete protocol is reported in Appendix B.

## B. Results

End-to-End Control Performance (Q1). Fig. 3 tests whether the proposed model and critic designs translate into more effective policy learning under a fixed real-environment interaction budget. IADD-TR attains the highest or competitive returns across all five tasks while improving more rapidly than SAC, PPO, and SLBO. Relative to the strongest modelbased comparator, MBPO, the gain is clearest on HalfCheetah, Walker2d, Ant, and Humanoid; on Hopper, IADD-TR improves faster early in training and remains competitive at convergence.

The advantage is particularly pronounced on Ant and Humanoid, where the higher-dimensional dynamics make policy learning more sensitive to errors in model-generated data and critic estimates. These results establish the end-to-end consequence of the proposed design: IADD-TR preserves the sample-efficiency benefit of short model rollouts while producing stronger policy performance on most tasks. Because aggregate return cannot identify which component produces the gain, Q2–Q4 separately examine component contributions and their intended mechanisms.

Complementary Contributions of IADD and TR (Q2). The ablation in Fig. 4 separates the two design claims: IADD modifies how model-generated transitions are constructed, whereas TR modifies how the critic is used to estimate the policy gradient. We compare MBPO, MBPO+TR, IADD, and the full IADD+TR method under the same real-environment interaction budget, training schedule, and random seeds. Zeroaction collection is used only by the IADD variants, so each comparison preserves the sampling procedure associated with the corresponding model design.

![](images/da79b3f12aa9b7f708912171c4ac776ba3e9c6604396335d799cd77fd7166af6.jpg)

![](images/991e6b984ed3b943e9facfa112a1d8912647ad00325af71a95b1f69ea6834152.jpg)

![](images/f6f11e1c3e7f503437114364b017bd8711129089a6fd3fb0a14c81cf179b9522.jpg)

![](images/c8530a7ff57e2b1b6b60392e4fccfde509b4d1ae6a24145b547122942d4acaeb.jpg)

![](images/7e5378996c557a3b6f356df3c6b43a45c662b2de55a6f59a79398029c2314acd.jpg)  
IADD-TR MBPO SLBO PPO SAC

Fig. 3. Performance comparison on MuJoCo continuous-control benchmarks. We compare the proposed method (IADD-TR) with two model-free baselines (SAC and PPO) and two model-based baselines (SLBO and MBPO). Solid curves denote the mean return over five random seeds, and the shaded regions indicate one standard deviation.  
![](images/320748b84ac02f23db7a218735f2a9f961ca9b0bd8bf84143f747302c37a3f3d.jpg)

![](images/683c906d93592ee7f296289853bb4c0799f30b25960b83faa1ff2201ca66885a.jpg)

![](images/9b3dbd4e2fe50a1657663cf78b8aad5108f9ec94972b402e6823390fa652f270.jpg)

![](images/3840ef8f17265a0e677a5884e20ab6529d7049f7f56fba59535ad29f3675e57f.jpg)

![](images/86d04f0be4bdbc1dff7ec8182a9510c4d5850fee46e7a88fd543808aa217df3b.jpg)  
Fig. 4. Ablation results on MuJoCo continuous-control benchmarks. We compare MBPO, MBPO with targeted regularization (MBPO+TR), the proposed IADD, and the full method that combines both components (IADD+TR). Solid curves denote the mean return over five random seeds, and shaded regions indicate one standard deviation.

Adding either IADD or TR generally improves over MBPO, showing that neither component is useful only in the presence of the other. Their effects are also task dependent. IADD provides the clearer individual gain on HalfCheetah, whereas MBPO+TR is particularly strong on Hopper. On Walker2d and Ant, combining the components produces the clearest improvement in learning speed and final return. On Humanoid, all three enhanced variants substantially improve the learning trajectory relative to MBPO, with IADD+TR maintaining strong performance throughout training.

The consistent performance of the combined method supports the intended division of labor: the model component improves the construction of synthetic experience, and TR improves the critic signal used by the actor. The return curves establish that the components are complementary, but they do not by themselves verify these internal mechanisms. Q3 therefore tests the model representation directly, while Q4 evaluates the policy-gradient estimate directly.

Validation of the Zero-Action Model Design (Q3). Q3 tests the model-specific claim that the hard zero-action definition guides the learned decomposition toward the intended physical intermediate state, rather than merely fitting $s _ { t + 1 }$ through an arbitrary hidden representation. We use the controlled synthetic dynamics environment detailed in Appendix B, in which the simulator records the postintervention state $\tilde { s } _ { t , o }$ before an action-independent natural evolution produces $s _ { t + 1 }$ . This recorded state is never used as a nonzero-action training target. The observable-state MSE and action-effect correlation therefore measure recovery of the intended intermediate coordinate. Held-out next-state MSE checks whether this recovery compromises the model’s predictive role. We vary the zero-action ratio from 0 to 0.40 with 100,000 training transitions and 10,000 held-out nonzeroaction transitions. Figure 5 reports the completed sweep for random seed 1.

![](images/920b7cd83cc8e21e6ec77c5bb4908af3613bc1ace03efe30e6f1f27bc833941e.jpg)

![](images/b7b01d15e237a6e0e11dff1d065ecce5dbd9d0ea1c3ca153945db66c98097580.jpg)

![](images/d02ba5f8d2f481e267b0678c091ff2af6e11b7798e4576b92a06f470c1bd31e2.jpg)  
Fig. 5. Controlled synthetic dynamics study. The panels show the observable post-intervention-state MSE (left), action-effect Pearson correlation (middle), and held-out next-state MSE (right) over zero-action ratios from 0 to 0.4. Each curve reports the training run evaluated on 10,000 held-out nonzero-action transitions.

Increasing zero-action coverage substantially improved the two representation-sensitive metrics in the reported run. From $\lambda _ { 0 } ~ = ~ 0 ~ t o ~ \lambda _ { 0 } ~ = ~ 0 . 4 0$ , the post-intervention-state MSE decreased from 4.043 to 0.0109, a 99.7% reduction. Over the same range, the action-effect correlation increased from 0.346 to 0.973. The MSE had already reached 0.379 at $\lambda _ { 0 } = 0 . 1 0$ and then decreased at every higher ratio. Thus, the anchor changes what the intermediate representation encodes, not only the final transition fit.

The held-out next-state MSE was $2 . 3 2 5 \times 1 0 ^ { - 4 }$ without zero-action samples and improved to $2 . 1 5 6 \times 1 0 ^ { - 4 }$ at $\lambda _ { 0 } =$ 0.10. It remained between $2 . 3 2 5 \times 1 0 ^ { - 4 }$ and $2 . 4 3 4 \times 1 0 ^ { - 4 }$ at the three higher ratios. Together, these results support the proposed model design within this run: zero-action coverage steers the observable block toward the simulator-defined postintervention state while preserving low transition error.

Validation of the Targeted Policy-Gradient Estimator (Q4). Q4 directly tests the estimator-specific claim behind TR: the targeted critic should produce a policy-gradient direction closer to the long-horizon return signal used to evaluate the policy. We conduct a paired diagnostic on Hopper using independently trained SAC actors frozen at six checkpoints from 50 to 300 epochs. At each checkpoint, critics with and without TR are trained from identical initializations using the same 50,000 replay transitions, minibatch order, and stochastic target actions. Their score-function gradients are evaluated on identical held-out state–initial-action pairs and accumulated over 20 critic-training epochs, isolating the effect of the targeted correction from actor, data, and optimization variation.

We form a finite-sample MC reference on the same state– initial-action pairs using 32 trajectories per state and a horizon of 500. The MC, TR, and No-TR estimates share the same actions and policy-score vectors and differ only in the return or critic value used to weight those scores. Cosine similarity with this MC reference therefore measures directional accuracy without conflating it with gradient scale.

![](images/db339465b47b8c7c88305ecaa8ff45f0cafdc80da50e3ef54c01159eb012fbf4.jpg)  
Fig. 6. Policy-gradient alignment diagnostic on Hopper. We independently train a SAC agent and freeze its actor checkpoints after 50, 100, 150, 200, 250, and 300 training epochs; these are not actors from the main IADD-TR experiments. At each checkpoint, we train paired critics with and without TR from identical initializations on the same 50,000 replay transitions. The critic-induced score-function gradients are evaluated on the same fixed state– initial-action pairs after every critic update and accumulated over 20 critictraining epochs. The finite-sample MC reference uses 256 held-out states, 32 trajectories per state, and a horizon of 500. Curves report the mean over five paired random seeds, and shaded bands indicate mean ± one standard error of the mean (s.e.m.). Higher cosine similarity indicates closer directional agreement with the MC policy-gradient reference.

Fig. 6 shows that TR achieves higher mean cosine similarity than No-TR at every actor checkpoint. The improvement is largest at epoch 50, where similarity increases from 0.728 to 0.771, and remains positive at epoch 300, increasing from 0.918 to 0.928. Across the six checkpoints, the paired gain ranges from 0.009 to 0.044. The larger gains at earlier and intermediate checkpoints are consistent with TR being most useful when the baseline critic provides a less accurate update direction.

Q4 therefore supports the policy-learning mechanism proposed in the paper: after targeted correction, the critic-induced policy-gradient estimate is more closely aligned with the MC reference. Together with the return improvements in Q2, this result links the estimator-level correction to more reliable actor updates. The MC reference is itself a finite-sample rollout estimate, and cosine similarity evaluates direction rather than magnitude; consequently, this diagnostic does not claim exact gradient recovery or guarantee monotonic return improvement.

## VI. CONCLUSION

This paper presents IADD-TR, a framework for reducing model-learning and policy-learning bias in model-based reinforcement learning. IADD decomposes transition dynamics into an action-intervention stage and a natural evolution stage without direct action input. A zero-action anchor aligns the observable component of the learned intermediate representation with the physical state space. TR further improves policy learning by explicitly addressing the mismatch between replay-based critic learning and policy-gradient optimization. Across five MuJoCo continuous-control tasks, IADD-TR performed competitively against representative model-free and model-based baselines and learned faster on several tasks. These results suggest that separating intervention effects from natural evolution and improving the alignment between critic learning and policy optimization can effectively mitigate learning bias in model-based reinforcement learning. The applicability of IADD-TR is currently bounded by the identifiability assumptions and the availability of a meaningful zero-action anchor.

## APPENDIX A IMPLEMENTATION DETAILS

We summarize the implementation and training settings used in the MuJoCo experiments in Tables I and II. The rollout-length schedules and the remaining model-based training settings follow MBPO, while the three additional settings introduced by IADD-TR are fixed across all environments. For a schedule denoted by $x  y$ over epochs $a  b ,$ the value at epoch e is the thresholded linear function min $\{ \operatorname* { m a x } [ x + ( e -$ $a ) ( y - x ) / ( b - a ) , x ] , y \}$

TABLE I  
COMMON IMPLEMENTATION SETTINGS AND PROPOSED-METHOD HYPERPARAMETERS FOR THE MUJOCO EXPERIMENTS.
<table><tr><td rowspan=1 colspan=1>Setting</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>MuJoCo version</td><td rowspan=1 colspan=1>2.0.0</td></tr><tr><td rowspan=1 colspan=1>Environment steps per epoch</td><td rowspan=1 colspan=1>1000</td></tr><tr><td rowspan=1 colspan=1>Model rollouts per environment step</td><td rowspan=1 colspan=1>400</td></tr><tr><td rowspan=1 colspan=1>Ensemble size</td><td rowspan=1 colspan=1>7</td></tr><tr><td rowspan=1 colspan=1>Model learning rate</td><td rowspan=1 colspan=1> $\overline { { 1 \times 1 0 ^ { - 3 } } }$ </td></tr><tr><td rowspan=1 colspan=1>Agent learning rate</td><td rowspan=1 colspan=1> $\overline { { 3 \times 1 0 ^ { - 4 } } }$ </td></tr><tr><td rowspan=1 colspan=1>Zero-action collection ratio</td><td rowspan=1 colspan=1> $\overline { { \lambda _ { 0 } = 0 . 1 } }$ </td></tr><tr><td rowspan=1 colspan=1>Targeted-regularization weight</td><td rowspan=1 colspan=1> $\overline { { \beta = 1 } }$ </td></tr><tr><td rowspan=1 colspan=1>Random seeds</td><td rowspan=1 colspan=1>5</td></tr></table>

TABLE II  
ENVIRONMENT-SPECIFIC HYPERPARAMETER SETTINGS FOR THE MUJOCO EXPERIMENTS.
<table><tr><td rowspan=1 colspan=1>Env.</td><td rowspan=1 colspan=1>Epochs</td><td rowspan=1 colspan=1>Updates perstep</td><td rowspan=1 colspan=1>Model horizon</td><td rowspan=1 colspan=1>Network</td></tr><tr><td rowspan=1 colspan=1>HalfCheetah</td><td rowspan=1 colspan=1>90</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>MLP4 × 200</td></tr><tr><td rowspan=1 colspan=1>Walker2d</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>MLP4 × 200</td></tr><tr><td rowspan=1 colspan=1>Ant</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>1 → 25 over epochs20 → 100</td><td rowspan=1 colspan=1>MLP4 × 200</td></tr><tr><td rowspan=1 colspan=1>Hopper</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>1 → 15 over epochs20 → 100</td><td rowspan=1 colspan=1>MLP4 × 200</td></tr><tr><td rowspan=1 colspan=1>Humanoid</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>1 → 25 over epochs20 → 300</td><td rowspan=1 colspan=1>MLP4 × 400</td></tr></table>

## APPENDIX B

## CONTROLLED SYNTHETIC DYNAMICS STUDY

We construct a controlled synthetic dynamics environment using a two-stage ordinary differential equation (ODE) system to evaluate whether zero-action coverage enables recovery of the simulator-defined observable post-intervention state $\tilde { s } _ { t , o } .$ The state is two-dimensional, $s _ { t } ~ = ~ ( x _ { t } , v _ { t } ) \in \mathbb { R } ^ { 2 }$ , and the action is scalar, $a _ { t } \in [ - 1 , 1 ]$ . Each transition is generated by two stages:

$$
s _ { t } , a _ { t } \to \tilde { s } _ { t , o } \to s _ { t + 1 } ,\tag{23}
$$

where $\boldsymbol { \tilde { s } } _ { t , o } ~ = ~ ( \boldsymbol { \tilde { x } } _ { t } , \boldsymbol { \tilde { v } } _ { t } )$ is the simulator-recorded observable post-intervention state. The first stage is an instantaneous intervention on velocity,

$$
\tilde { x } _ { t } = x _ { t } , \qquad \tilde { v } _ { t } = v _ { t } + \kappa _ { a } \operatorname { t a n h } ( a _ { t } ) ,\tag{24}
$$

with intervention scale $\kappa _ { a } = 0 . 8 .$ . Therefore, the zero-action anchor holds exactly: when $a _ { t } = 0 , \tilde { s } _ { t , o } = s _ { t }$

The second stage is an action-independent natural evolution. Starting from $\tilde { s } _ { t , o } ,$ the system follows a damped oscillator ODE,

$$
\frac { d x } { d t _ { c } } = v , \qquad \frac { d v } { d t _ { c } } = - \Omega ^ { 2 } x - \kappa _ { d } v ,\tag{25}
$$

where $t _ { c }$ denotes continuous time, $\Omega = 1 . 2$ is the natural frequency, and $\kappa _ { d } ~ = ~ 0 . 2 5$ is the damping coefficient. We integrate this ODE over $\Delta t = 0 . 1 5$ using fixed-step fourthorder Runge–Kutta (RK4) with eight substeps, giving the passive flow map $F _ { 0 }$ and the next state $s _ { t + 1 } ~ = ~ F _ { 0 } ( \tilde { s } _ { t , o } )$ Initial states are sampled uniformly from $x _ { t } ~ \in ~ [ - 2 , 2 ]$ and $v _ { t } \in [ - 1 . 5 , 1 . 5 ]$ . Ordinary replay actions are sampled from the state-correlated behavior policy $a _ { t } = \operatorname { t a n h } ( 1 . 1 x _ { t } - 0 . 7 v _ { t } + \zeta _ { t } )$ where $\zeta _ { t } \sim \mathcal { N } ( 0 , 0 . 2 5 ^ { 2 } )$ , and then clipped to the action range. Consequently, ordinary replay tuples mix intervention effects with passive evolution.

For each zero-action ratio $\lambda _ { 0 } \in \{ 0 , 0 . 1 0 , 0 . 2 0 , 0 . 3 0 , 0 . 4 0 \}$ we keep the total training size fixed at $N ~ = ~ 1 0 0 { , } 0 0 0$ The training set contains $\lfloor \lambda _ { 0 } N \rfloor$ exact zero-action transitions and $N - \lfloor \lambda _ { 0 } N \rfloor$ ordinary transitions. The observed training next states contain independent Gaussian noise with standard deviation 0.02. Evaluation uses $M = 1 0 \small { , } 0 0 0$ clean held-out transitions. All ratios use the same evaluation set and nested ordinary and zero-action data pools, so the mixture ratio is the only changing factor. The reported sweep uses random seed 1, which determines the model initialization and generated data

## TABLE III

RESULTS OF THE CONTROLLED SYNTHETIC DYNAMICS STUDY FOR RANDOM SEED 1 WITH 100,000 TRAINING TRANSITIONS AND 10,000 CLEAN HELD-OUT TRANSITIONS PER RATIO. LOWER MSE AND HIGHER PEARSON CORRELATION ARE BETTER.
<table><tr><td rowspan=1 colspan=1> $\overline { { \lambda _ { 0 } } }$ </td><td rowspan=1 colspan=1> $\overline { { \mathrm { M S E } _ { \mathrm { m i d } } } }$ ↓</td><td rowspan=1 colspan=1>Effect Pearson↑</td><td rowspan=1 colspan=1> $\overline { { \mathrm { M S E } _ { \mathrm { n e x t } } \downarrow } }$ </td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>4.0435</td><td rowspan=1 colspan=1>0.3455</td><td rowspan=1 colspan=1>2.3250 × 10−4</td></tr><tr><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.3794</td><td rowspan=1 colspan=1>0.7178</td><td rowspan=1 colspan=1>2.1564 × 10</td></tr><tr><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.2378</td><td rowspan=1 colspan=1>0.6039</td><td rowspan=1 colspan=1> $2 . 3 2 5 2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td rowspan=1 colspan=1>0.30</td><td rowspan=1 colspan=1>0.03725</td><td rowspan=1 colspan=1>0.8876</td><td rowspan=1 colspan=1> $\overline { { 2 . 4 1 6 8 \times 1 0 ^ { - 4 } } }$ </td></tr><tr><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.01090</td><td rowspan=1 colspan=1>0.9735</td><td rowspan=1 colspan=1> $\overline { { 2 . 4 3 4 4 \times 1 0 ^ { - 4 } } }$ </td></tr></table>

pools. The simulator records $\tilde { s } _ { t , o } ,$ but it is never provided as a training target on nonzero-action samples.

Figure 5 and Table III report this single completed training run. The 10,000 held-out transitions are evaluation observations rather than independent training repetitions, so no crossseed uncertainty is claimed.

We train a three-member ensemble of two-stage models for 100 epochs. Each stage contains two fully connected layers with hidden width four and a Swish activation. The actionintervention model produces two-dimensional observable and auxiliary latent blocks,

$$
\begin{array} { r } { p _ { \mathrm { a c t } } \big ( \boldsymbol { s } _ { t } , \boldsymbol { a } _ { t } \big ) = \left( \widehat { \widetilde { s } } _ { t , o } , \widehat { \widetilde { s } } _ { t , l } \right) \in \mathbb { R } ^ { 2 } \times \mathbb { R } ^ { 2 } , } \end{array}\tag{26}
$$

giving a four-dimensional intermediate representation. The natural evolution model receives this complete representation and directly predicts the next state $s _ { t + 1 }$

All inputs and next-state targets are standardized using the training set. For each ensemble member k, the observable intermediate output is defined by

$$
\begin{array} { r } { \widehat { \widetilde { s } } _ { i , o } ^ { ( k ) } = \left\{ \begin{array} { l l } { s _ { i } , } & { a _ { i } = 0 , } \\ { p _ { \mathrm { a c t } , o } ^ { ( k ) } ( s _ { i } , a _ { i } ) , } & { a _ { i } \neq 0 . } \end{array} \right. } \end{array}\tag{27}
$$

The latent block $\hat { \tilde { s } } _ { i , l } ^ { ( k ) }$ remains learned for both zero and nonzero actions. The action-intervention and natural evolution models are jointly optimized using Adam with a learning rate of $1 0 ^ { - 3 }$ , a batch size of 256, and the Gaussian negative loglikelihood objective $\widehat { \mathcal { L } } = \widehat { \mathcal { L } } _ { \mathrm { N L L } }$ . Thus, zero-action samples fix the observable input to the natural evolution stage through the hard model definition, without an auxiliary intermediate-state loss or an anchoring weight.

We evaluate the observable post-intervention recovery error in physical state coordinates,

$$
\mathrm { M S E } _ { \mathrm { m i d } } = \frac { 1 } { K M d _ { o } } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { M } \left. \widehat { \widetilde { s } } _ { i , o } ^ { ( k ) } - \widetilde { s } _ { i , o } \right. _ { 2 } ^ { 2 } ,\tag{28}
$$

the Pearson correlation between the ensemble-mean recovered action effect $\widehat { \tilde { s } } _ { i , o } - s _ { i }$ and the true effect $\tilde { s } _ { i , o } - s _ { i } ,$ , and the heldout next-state MSE.

Table III shows progressively lower intermediate-state MSE as the zero-action ratio increases. From $\lambda _ { 0 } = 0 \mathrm { \ t o \ } \lambda _ { 0 } = 0 . 4 0 .$ the post-intervention-state MSE decreases by 99.7%, while the effect correlation increases from 0.3455 to 0.9735. Because the model copies the observable intermediate output from the current state at $a = 0 ,$ , its zero-action error is at numerical precision. This hard-constraint error holds for every ratio and is therefore not reported as a learned metric.

The next-state MSE reaches its minimum of $2 . 1 5 6 4 \times 1 0 ^ { - 4 }$ at $\lambda _ { 0 } ~ = ~ 0 . 1 0$ and remains below $2 . 4 4 \times 1 0 ^ { - 4 }$ at every ratio. Thus, the reported run shows improved observablestate recovery while maintaining low one-step prediction error. Because changing $\lambda _ { 0 }$ also changes the number of nonzeroaction training transitions, the sweep does not isolate the cause of the small differences among nonzero ratios.

## APPENDIX C PROOF OF THEOREM 1

Proof. For each reachable context $\textbf { u } = ~ ( s , a )$ , let $\nu _ { \mathbf { u } }$ and $\bar { \nu } _ { \mathbf { u } }$ denote the conditional distributions of the complete intermediate representations under the two parameterizations. Observational equivalence gives

$$
( p _ { \mathrm { e n v } } ) _ { \# } \nu _ { \mathbf { u } } = ( \bar { p } _ { \mathrm { e n v } } ) _ { \# } \bar { \nu } _ { \mathbf { u } }\tag{29}
$$

for every u, where $( \cdot ) _ { \# }$ denotes the pushforward of a probability distribution.

By A1, both natural evolution mechanisms have inverses on the common reachable next-state support. Hence they induce the context-independent change of intermediate coordinates

$$
{ \cal H } : = \bar { p } _ { \mathrm { e n v } } ^ { - 1 } \circ p _ { \mathrm { e n v } } .\tag{30}
$$

Applying $\bar { p } _ { \mathrm { e n v } } ^ { - 1 }$ to Eq. (29) yields

$$
\begin{array} { r } { H _ { \# } \nu _ { \mathbf { u } } = \bar { \nu } _ { \mathbf { u } } . } \end{array}\tag{31}
$$

Let $\pi _ { o }$ denote projection onto the observable coordinate. Take any reachable intermediate representation z whose image $y ~ = ~ p _ { \mathrm { e n v } } ( \mathbf { z } )$ lies in the conditional next-state support of some zero-action context $( s , a _ { 0 } )$ . This is precisely the zeroaction-covered portion of the reachable support specified in the theorem. Observational equivalence makes this conditional support the same under both parameterizations. Because the natural evolution mechanisms are one-to-one and have continuous inverses on their reachable ranges, z belongs to the intermediate support under $( s , a _ { 0 } )$ , while $H ( \mathbf { z } )$ belongs to the corresponding barred support. The defining zero-action identities of the two parameterizations then give $\pi _ { o } ( \mathbf { z } ) = s$ and $\pi _ { o } ( H ( \mathbf { z } ) ) = s$ . Therefore,

$$
\pi _ { o } ( H ( \mathbf { z } ) ) = \pi _ { o } ( \mathbf { z } ) .\tag{32}
$$

For an arbitrary reachable $\mathbf { u } = \left( s _ { t } , a _ { t } \right)$ whose conditional next-state support is covered by zero-action contexts, $\nu _ { \mathbf { u } }$ is concentrated on intermediate representations whose observable coordinate is the deterministic value $p _ { \mathrm { a c t } , o } ( s _ { t } , a _ { t } )$ . Similarly, $\bar { \nu } _ { \mathbf { u } }$ has the deterministic observable coordinate $\bar { p } _ { \mathrm { a c t } , o } ( s _ { t } , a _ { t } )$ Combining Eqs. (31) and (32) shows that these two point masses coincide. Thus

$$
\bar { p } _ { \mathrm { a c t } , o } ( s _ { t } , a _ { t } ) = p _ { \mathrm { a c t } , o } ( s _ { t } , a _ { t } ) ,
$$

which proves pointwise identifiability on the zero-actioncovered reachable support.

## APPENDIX D PROOF OF THEOREM 2

Fix a reachable observable post-intervention state $s _ { o } ^ { \prime }$ and define its state-action context fiber by

$$
\mathcal { U } ( s _ { o } ^ { \prime } ) : = \{ \mathbf { u } = ( s , a ) : p _ { \mathrm { a c t } , o } ( s , a ) = s _ { o } ^ { \prime } \} .\tag{33}
$$

Because both parameterizations belong to the factorized model class in Eq. (7), their latent conditional densities already satisfy

$$
\begin{array} { r l r } & { } & { { p } _ { l } ( \tilde { s } _ { t , l } \mid { \bf u } ) = \displaystyle \prod _ { j = 1 } ^ { d _ { l } } p _ { l , j } ( \tilde { s } _ { t , l , j } \mid { \bf u } ) , } \\ & { } & { \bar { p } _ { l } ( \bar { s } _ { t , l } \mid { \bf u } ) = \displaystyle \prod _ { j = 1 } ^ { d _ { l } } \bar { p } _ { l , j } ( \bar { s } _ { t , l , j } \mid { \bf u } ) . } \end{array}
$$

Thus, factorization is part of the model definition rather than an additional identifiability assumption. The additional conditions used in the theorem are:

• B1 (Regularity of conditional latent densities): The component densities are positive and twice continuously differentiable on connected latent supports that locally contain product neighborhoods.

• B2 (Sufficient variability within an observable fiber): Fix $\mathbf { u } _ { 0 } \in \overline { { \mathfrak { U } ( s _ { o } ^ { \prime } ) } }$ and define

$$
\bar { c } _ { j } ( x , \mathbf { u } ) : = \log \bar { p } _ { l , j } ( x \mid \mathbf { u } ) - \log \bar { p } _ { l , j } ( x \mid \mathbf { u } _ { 0 } ) .
$$

For every reachable $\bar { s } _ { t , l } .$ , there exist 2d<sub>l</sub> contexts ${ \mathbf { u } } _ { 1 } , \ldots , { \mathbf { u } } _ { 2 d _ { l } } \in { \mathcal { U } } ( s _ { o } ^ { \prime } )$ such that the matrix with columns

$$
\begin{array} { r } { \mathbf { v } ( \bar { s } _ { t , l } , \mathbf { u } _ { k } ) = \left[ \begin{array} { c } { \partial _ { x } ^ { 2 } \bar { c } _ { 1 } ( \bar { s } _ { t , l , 1 } , \mathbf { u } _ { k } ) } \\ { \vdots } \\ { \partial _ { x } ^ { 2 } \bar { c } _ { d _ { l } } ( \bar { s } _ { t , l , d _ { l } } , \mathbf { u } _ { k } ) } \\ { \partial _ { x } \bar { c } _ { 1 } ( \bar { s } _ { t , l , 1 } , \mathbf { u } _ { k } ) } \\ { \vdots } \\ { \partial _ { x } \bar { c } _ { d _ { l } } ( \bar { s } _ { t , l , d _ { l } } , \mathbf { u } _ { k } ) } \end{array} \right] \in \mathbb { R } ^ { 2 d _ { l } } } \end{array}
$$

has full rank.

Proof. Fix a reachable observable post-intervention state $s _ { o } ^ { \prime } .$ By A1, the support-level coordinate change $H ~ = ~ { \bar { p } } _ { \mathrm { e n v } } ^ { - 1 } ~ \circ$ p<sub>env</sub> and its inverse are twice continuously differentiable. Theorem 1 gives $\pi _ { o } \circ H \ : = \ : \pi _ { o }$ on the reachable support. Consequently, H maps each latent fiber bijectively onto the corresponding barred fiber and has the form

$$
H ( s _ { o } ^ { \prime } , \tilde { s } _ { t , l } ) = \big ( s _ { o } ^ { \prime } , H _ { l , s _ { o } ^ { \prime } } ( \tilde { s } _ { t , l } ) \big ) .
$$

Both $H _ { l , s _ { o } ^ { \prime } }$ and its inverse are twice continuously differentiable. Differentiating their composition gives

$$
J _ { H _ { l , s _ { o } ^ { \prime } } ^ { - 1 } } \left( H _ { l , s _ { o } ^ { \prime } } ( \tilde { s } _ { t , l } ) \right) J _ { H _ { l , s _ { o } ^ { \prime } } } ( \tilde { s } _ { t , l } ) = I _ { d _ { l } } ,
$$

so ${ \ J } _ { H _ { l , s _ { \alpha } ^ { \prime } } }$ is nonsingular. Thus the support inverses in A1 induce the latent-coordinate diffeomorphism

$$
\bar { s } _ { t , l } = H _ { l , s _ { o } ^ { \prime } } ( \tilde { s } _ { t , l } ) .
$$

For notational simplicity within this fixed observable fiber, write $H = H _ { l , s _ { o } ^ { \prime } }$ and $H = \left( H _ { 1 } , \ldots , H _ { d _ { l } } \right)$

For every $\mathbf { u } \in \mathcal { U } ( s _ { o } ^ { \prime } )$ , let $P _ { l } ^ { \mathbf { u } }$ and $\bar { P } _ { l } ^ { \mathbf { u } }$ denote the two conditional latent distributions. Because the observable coordinate is fixed at $s _ { o } ^ { \prime } ,$ restricting Eq. (31) to this fiber gives

$$
\begin{array} { r } { H _ { \# } P _ { l } ^ { \mathbf { u } } = \bar { P } _ { l } ^ { \mathbf { u } } . } \end{array}
$$

The change-of-variables formula therefore yields

$$
p _ { l } ( \tilde { s } _ { t , l } \mid \mathbf { u } ) = \bar { p } _ { l } ( H ( \tilde { s } _ { t , l } ) \mid \mathbf { u } ) \mid \mathrm { d e t } J _ { H } ( \tilde { s } _ { t , l } ) \mid .\tag{34}
$$

Let

$$
c _ { j } ( x , \mathbf { u } ) : = \log p _ { l , j } ( x \mid \mathbf { u } ) - \log p _ { l , j } ( x \mid \mathbf { u } _ { 0 } ) ,
$$

and define $\bar { c } _ { j }$ as in B2. Evaluating the logarithm of Eq. (34) at u and $\mathbf { u } _ { 0 }$ and subtracting cancels the Jacobian term. The component factorizations in Eq. (7) then yield

$$
\sum _ { j = 1 } ^ { d _ { l } } c _ { j } \big ( \tilde { s } _ { t , l , j } , \mathbf { u } \big ) = \sum _ { m = 1 } ^ { d _ { l } } \bar { c } _ { m } \big ( H _ { m } \big ( \tilde { s } _ { t , l } \big ) , \mathbf { u } \big ) .\tag{35}
$$

Fix two distinct latent coordinate indices $r , k \in \{ 1 , \ldots , d _ { l } \}$ Applying $\partial ^ { 2 } / ( \partial \tilde { s } _ { t , l , r } \partial \tilde { s } _ { t , l , k } )$ to Eq. (35) gives

$$
\begin{array} { l } { { \displaystyle 0 = \sum _ { m = 1 } ^ { d _ { l } } \Big [ \partial _ { x } ^ { 2 } \bar { c } _ { m } ( H _ { m } ( \tilde { s } _ { t , l } ) , { \bf u } ) \partial _ { \tilde { s } _ { t , l , r } } H _ { m } ( \tilde { s } _ { t , l } ) \partial _ { \tilde { s } _ { t , l , k } } H _ { m } ( \tilde { s } _ { t , l } ) } } \\ { { \displaystyle ~ + \partial _ { x } \bar { c } _ { m } ( H _ { m } ( \tilde { s } _ { t , l } ) , { \bf u } ) \partial _ { \tilde { s } _ { t , l , r } \tilde { s } _ { t , l , k } } ^ { 2 } H _ { m } ( \tilde { s } _ { t , l } ) \Big ] . } } \end{array}\tag{36}
$$

For fixed $\tilde { s } _ { t , l } , r , k$ , the derivatives of H do not depend on the context u. Evaluating Eq. (36) at the $2 d _ { l }$ contexts supplied by B2 gives a full-rank homogeneous linear system. Its unique solution is

$$
\begin{array} { r } { \partial _ { \tilde { s } _ { t , l , r } } H _ { m } ( \tilde { s } _ { t , l } ) \partial _ { \tilde { s } _ { t , l , k } } H _ { m } ( \tilde { s } _ { t , l } ) = 0 , } \\ { \partial _ { \tilde { s } _ { t , l , r } \tilde { s } _ { t , l , k } } ^ { 2 } H _ { m } ( \tilde { s } _ { t , l } ) = 0 . } \end{array}\tag{37}
$$

for every m and every pair $r \neq k$

Equation (37) shows that each output coordinate $H _ { m }$ can depend locally on at most one input coordinate. Because H is a diffeomorphism, its Jacobian is nonsingular; consequently, every row and every column of $J _ { H }$ contains exactly one nonzero entry. This assignment defines a permutation $\sigma _ { l , s _ { o } ^ { \prime } }$ . The connected-product-support condition in B1 prevents the assignment from switching within the latent support, so integration along each assigned coordinate yields a onedimensional diffeomorphism $h _ { m , s _ { o } ^ { \prime } }$ satisfying

$$
H _ { m } ( \tilde { s } _ { t , l } ) = h _ { m , s _ { o } ^ { \prime } } \Big ( \tilde { s } _ { t , l , \sigma _ { l , s _ { o } ^ { \prime } } ( m ) } \Big ) .
$$

This proves Eq. (10).

## APPENDIX E

## PROOF OF THEOREM 3

Proof. Let $O \ = \ ( S , A , Y ^ { \pi _ { \theta } } )$ denote the augmented replay sample introduced before Theorem 3, with $S ~ \sim ~ \mathcal { D } _ { S }$ and $A \mid S \sim \pi _ { \beta } ( \cdot \mid S )$ . Bellman consistency gives

$$
\mathbb { E } _ { \mathcal { D } } \left[ Y ^ { \pi _ { \theta } } \mid S = s , A = a \right] = Q ^ { \pi _ { \theta } } ( s , a ) .\tag{38}
$$

Consider an arbitrary regular parametric submodel $\{ \mathcal { D } _ { \varepsilon } \}$ through the replay distribution D, with score ℓ(O). Along this path, define

$$
Q _ { \varepsilon } ( s , a ) : = \mathbb { E } _ { \mathcal { D } _ { \varepsilon } } \left[ Y ^ { \pi _ { \theta } } \mid S = s , A = a \right]\tag{39}
$$

and

$$
\psi _ { \varepsilon } = \mathbb { E } _ { S \sim \mathcal { D } _ { \varepsilon , S } } \left[ \mathbb { E } _ { \tilde { a } \sim \pi _ { \theta } ( \cdot | S ) } \left[ Q _ { \varepsilon } ( S , \tilde { a } ) g _ { \theta } ( S , \tilde { a } ) \right] \right] .\tag{40}
$$

At $\varepsilon = 0 , Q _ { 0 } = Q ^ { \pi _ { \theta } }$ and $\psi _ { 0 } = \psi _ { \mathrm { p g } } ^ { \mathcal { D } }$ . We hold $\pi _ { \theta }$ and $g _ { \theta }$ fixed throughout the pathwise derivative.

The derivative of the conditional Bellman mean is

$$
\begin{array} { l } { \displaystyle \frac { d } { d \varepsilon } Q _ { \varepsilon } ( s , a ) \bigg \vert _ { \varepsilon = 0 } = \mathbb { E } _ { \mathcal { D } } \left[ \left( Y ^ { \pi _ { \theta } } - Q ^ { \pi _ { \theta } } ( s , a ) \right) \right. } \\ { \left. \times \ell ( O ) \mid S = s , A = a \right] . } \end{array}\tag{41}
$$

After changing the action measure from $\pi _ { \beta }$ to $\pi _ { \theta } .$ , its contribution to the derivative of $\psi _ { \varepsilon }$ becomes

$$
\mathbb { E } _ { \mathcal { D } } \left[ \frac { \pi _ { \theta } ( A \mid S ) } { \pi _ { \beta } ( A \mid S ) } \big ( Y ^ { \pi _ { \theta } } - Q ^ { \pi _ { \theta } } ( S , A ) \big ) g _ { \theta } ( S , A ) \ell ( O ) \right] .\tag{42}
$$

The state-marginal contribution is

$$
\mathbb { E } _ { \mathcal { D } } \left[ \left( \mathbb { E } _ { { \tilde { a } } \sim \pi _ { \theta } ( \cdot \vert S ) } \left[ Q ^ { \pi _ { \theta } } ( S , { \tilde { a } } ) g _ { \theta } ( S , { \tilde { a } } ) \right] - \psi _ { \mathrm { p g } } ^ { \mathcal { D } } \right) \ell ( O ) \right] .\tag{43}
$$

Perturbing the behavior-policy density $\pi _ { \beta } ( a \mid s )$ contributes no additional term because the target functional integrates actions under the fixed policy $\pi _ { \theta }$ . Combining the two contributions gives

$$
\left. \frac { d } { d \varepsilon } \psi _ { \varepsilon } \right| _ { \varepsilon = 0 } = \mathbb { E } _ { \mathcal { D } } \left[ \varphi _ { \mathrm { p g } } ^ { \mathcal { D } } ( O ) \ell ( O ) \right] ,\tag{44}
$$

where $\varphi _ { \mathrm { p g } } ^ { \mathcal { D } }$ is exactly Eq. (16). Its expectation under D is zero because the residual has conditional mean zero and the plugin term is centered by $\psi _ { \mathrm { p g } } ^ { \mathcal { D } }$ . Thus it is an influence function. Since the replay distribution D is modeled nonparametrically, its tangent space is $L _ { 0 } ^ { 2 } ( \mathcal { D } )$ , so the influence function is efficient.

## APPENDIX F PROOF OF THEOREM 4

We analyze the targeted correction action by action, following the functional targeted-regularization construction for continuous treatments [49]. Let $O \ = \ ( S , A , Y ^ { \pi _ { \theta } } )$ denote a replay observation, with $S \sim { \mathcal { D } } _ { S } , A \mid S \sim e _ { 0 } ( \cdot \mid S )$ , and

$$
m _ { 0 } ( s , a ) : = \mathbb { E } [ Y ^ { \pi _ { \theta } } \mid S = s , A = a ] = Q ^ { \pi _ { \theta } } ( s , a ) .\tag{45}
$$

The action-wise target is the marginal conditional-mean curve

$$
\mu _ { 0 } ( a ) : = \mathbb { E } _ { S \sim \mathcal { D } _ { S } } \left[ m _ { 0 } ( S , a ) \right] .\tag{46}
$$

This target is the mean action value at action $^ { a ; }$ no contrast with a reference action is taken.

Let ${ \widehat { m } } ( s , a ) = Q _ { \omega } ( s , a )$ be a preliminary critic and let ${ \widehat { e } } ( a \mid$ $s )$ be a candidate replay action-density estimate. For an actionwise correction $\epsilon ( a )$ , define the adjusted conditional mean

$$
m _ { \epsilon } ( s , a ) = { \widehat m } ( s , a ) + \frac { \epsilon ( a ) } { { \widehat e } ( a \mid s ) } .\tag{47}
$$

This is the population counterpart of the adjusted critic in Eq. (17). Its population targeted risk is

$$
\mathcal { R } _ { \infty } ( \epsilon ) = \mathbb { E } \left[ \left\{ Y ^ { \pi _ { \theta } } - \widehat { m } ( S , A ) - \frac { \epsilon ( A ) } { \widehat { e } ( A \mid S ) } \right\} ^ { 2 } \right] .\tag{48}
$$

The argument uses the following conditions. First, $e _ { 0 }$ and $\widehat { e }$ are positive and bounded away from zero on the relevant compact action support. Second, the conditional Bellman noise has mean zero and finite second moment, and the critic, inverse action densities, and correction functions are square integrable. Third, the action-wise correction class can approximate the population minimizer below, and the empirical targeted risk converges uniformly to Eq. (48). The fitted correction is an $o _ { P } ( 1 )$ approximate empirical-risk minimizer. For continuous actions, these conditions are imposed in $L ^ { 2 } ( \mathcal { D } _ { A } )$ over the replay action marginal, as in functional targeted regularization. Finally, either the preliminary critic is consistent for $m _ { 0 }$ , or eb is consistent for $e _ { 0 }$ . When a practical bootstrapped Bellman target is used in place of $Y ^ { \pi _ { \theta } }$ , its contribution to the targeted risk is additionally assumed to be $o _ { P } ( 1 )$

Proof. Because the correction is indexed by the observed action, the first-order condition for Eq. (48) can be evaluated action by action. For every action value a in the relevant support, the population minimizer satisfies

$$
\mathbb { E } \left[ \left. \frac { Y ^ { \pi _ { \theta } } - \widehat { m } ( S , A ) - \epsilon ^ { * } ( A ) / \widehat { e } ( A \mid S ) } { \widehat { e } ( A \mid S ) } \right. A = a \right] = 0 .\tag{49}
$$

Consequently,

$$
\epsilon ^ { * } ( a ) = { \frac { \mathbb { E } [ \{ Y ^ { \pi _ { \theta } } - { \widehat { m } } ( S , A ) \} / { \widehat { e } } ( A \mid S ) | A = a ] } { \mathbb { E } [ { \widehat { e } } ( A \mid S ) ^ { - 2 } | A = a ] } } .\tag{50}
$$

Uniform convergence, sufficient approximation capacity, and approximate empirical-risk minimization give

$$
\| \epsilon _ { \xi } - \epsilon ^ { * } \| _ { L ^ { 2 } ( \mathcal { D } _ { A } ) } = o _ { P } \bigl ( 1 \bigr ) .\tag{51}
$$

It remains to show that the adjusted conditional-mean curve is doubly robust. Let $p _ { A } ( a )$ denote the replay action marginal density. Since

$$
\begin{array} { r } { p _ { A } ( a ) = \mathbb { E } _ { S \sim \mathcal { D } _ { S } } \left[ e _ { 0 } ( a \mid S ) \right] , } \end{array}
$$

conditioning on $A = a$ in Eq. (50) yields

$$
\epsilon ^ { * } ( a ) = \frac { \mathbb { E } _ { S } \left[ \frac { e _ { 0 } ( a \mid S ) } { \widehat { e } ( a \mid S ) } \{ m _ { 0 } ( S , a ) - \widehat { m } ( S , a ) \} \right] } { \mathbb { E } _ { S } \left[ \frac { e _ { 0 } ( a \mid S ) } { \widehat { e } ( a \mid S ) ^ { 2 } } \right] } .\tag{52}
$$

Define the population curve induced by this correction as

$$
\mu ^ { * } ( a ) = \mathbb { E } _ { S } \left[ { \widehat { m } } ( S , a ) + { \frac { \epsilon ^ { * } ( a ) } { { \widehat { e } } ( a \mid S ) } } \right] .\tag{53}
$$

Writing $r _ { a } ( S ) = m _ { 0 } ( S , a ) - \widehat { m } ( S , a )$ and substituting Eq. (52) into Eq. (53) gives

$$
\begin{array} { r l } & { \mu ^ { * } ( a ) - \mu _ { 0 } ( a ) = - \operatorname { \mathbb { E } } _ { S } [ r _ { a } ( S ) ] } \\ & { \quad \quad + \frac { \operatorname { \mathbb { E } } _ { S } [ \widehat { e } ( a \mid S ) ^ { - 1 } ] \operatorname { \mathbb { E } } _ { S } \left[ \frac { e _ { 0 } ( a \mid S ) } { \widehat { e } ( a \mid S ) } r _ { a } ( S ) \right] } { \operatorname { \mathbb { E } } _ { S } \left[ \frac { e _ { 0 } ( a \mid S ) } { \widehat { e } ( a \mid S ) ^ { 2 } } \right] } . } \end{array}\tag{54}
$$

If the preliminary critic is correct, then $r _ { a } ( S ) = 0 .$ , and both terms on the right-hand side of Eq. (54) vanish. If the replay action density is correct, then $\widehat { e } = e _ { 0 }$ , and Eq. (52) reduces to

$$
\epsilon ^ { * } ( a ) = \frac { \mathbb { E } _ { S } [ r _ { a } ( S ) ] } { \mathbb { E } _ { S } [ e _ { 0 } ( a \mid S ) ^ { - 1 } ] } .\tag{55}
$$

The correction term in $\operatorname { E q . }$ (53) then equals $\mathbb { E } _ { S } [ r _ { a } ( S ) ]$ , so $\mu ^ { * } ( a ) = \mu _ { 0 } ( a )$ even when the preliminary critic is misspecified. Thus the population adjusted curve is correct if either the critic or the replay action density is correct.

For the fitted correction, define

$$
{ \widehat { \mu } } ^ { \mathrm { T R } } ( a ) = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \left[ Q _ { \omega } ( s _ { i } , a ) + { \frac { \epsilon _ { \xi } ( a ) } { { \widehat { e } } ( a \mid s _ { i } ) } } \right] .\tag{56}
$$

Equation (51), overlap, and uniform convergence of the state averages imply that $\widehat { \mu } ^ { \mathrm { T R } }$ converges to $\mu ^ { * }$ in $L ^ { 2 } ( \mathcal { D } _ { A } )$ . Combining this convergence with Eq. (54) gives

$$
\begin{array} { r } { \left. \widehat { \mu } ^ { \mathrm { T R } } - \mu _ { 0 } \right. _ { L ^ { 2 } ( \mathcal { D } _ { A } ) } = o _ { P } ( 1 ) } \end{array}\tag{57}
$$

whenever either $\widehat { m }$ is consistent for $m _ { 0 }$ or $\widehat { e }$ is consistent for $e _ { 0 }$ . This proves double robustness of the action-wise marginal conditional-mean curve.

## REFERENCES

[1] R. S. Sutton, A. G. Barto et al., Introduction to reinforcement learning. MIT press Cambridge, 1998, vol. 135.

[2] R. S. Sutton, “Integrated architectures for learning, planning, and reacting based on approximating dynamic programming,” in Machine learning proceedings 1990. Elsevier, 1990, pp. 216–224.

[3] F.-M. Luo, T. Xu, H. Lai, X.-H. Chen, W. Zhang, and Y. Yu, “A survey on model-based reinforcement learning,” Science China Information Sciences, vol. 67, no. 2, p. 121101, 2024.

[4] M. Deisenroth and C. E. Rasmussen, “Pilco: A model-based and data-efficient approach to policy search,” in Proceedings of the 28th International Conference on machine learning (ICML-11), 2011, pp. 465–472.

[5] S. Levine and V. Koltun, “Guided policy search,” in International conference on machine learning. PMLR, 2013, pp. 1–9.

[6] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to control: Learning behaviors by latent imagination,” in International Conference on Learning Representations, 2020.

[7] T. M. Moerland, J. Broekens, A. Plaat, C. M. Jonker et al., “Modelbased reinforcement learning: A survey,” Foundations and Trends® in Machine Learning, vol. 16, no. 1, pp. 1–118, 2023.

[8] J. Schrittwieser, I. Antonoglou, T. Hubert, K. Simonyan, L. Sifre, S. Schmitt, A. Guez, E. Lockhart, D. Hassabis, T. Graepel et al., “Mastering atari, go, chess and shogi by planning with a learned model,” Nature, vol. 588, no. 7839, pp. 604–609, 2020.

[9] M. Janner, J. Fu, M. Zhang, and S. Levine, “When to trust your model: Model-based policy optimization,” Advances in neural information processing systems, vol. 32, 2019.

[10] R. S. Sutton, “Dyna, an integrated architecture for learning, planning, and reacting,” ACM Sigart Bulletin, vol. 2, no. 4, pp. 160–163, 1991.

[11] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor,” in International conference on machine learning. PMLR, 2018, pp. 1861–1870.

[12] T. Kurutach, I. Clavera, Y. Duan, A. Tamar, and P. Abbeel, “Modelensemble trust-region policy optimization,” in International Conference on Learning Representations, 2018.

[13] A. Venkatraman, M. Hebert, and J. Bagnell, “Improving multi-step prediction of learned time series models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 29, no. 1, 2015.

[14] K. Asadi, D. Misra, and M. Littman, “Lipschitz continuity in modelbased reinforcement learning,” in International conference on machine learning. PMLR, 2018, pp. 264–273.

[15] X.-H. Chen, Y. Yu, Z. Zhu, Z. Yu, C. Zhenjun, C. Wang, Y. Wu, R.- J. Qin, H. Wu, R. Ding et al., “Adversarial counterfactual environment model learning,” Advances in Neural Information Processing Systems, vol. 36, pp. 70 654–70 706, 2023.

[16] J. Ho and S. Ermon, “Generative adversarial imitation learning,” Advances in neural information processing systems, vol. 29, 2016.

[17] S. Fujimoto, D. Meger, and D. Precup, “Off-policy deep reinforcement learning without exploration,” in International conference on machine learning. PMLR, 2019, pp. 2052–2062.

[18] A. Kumar, A. Zhou, G. Tucker, and S. Levine, “Conservative q-learning for offline reinforcement learning,” Advances in neural information processing systems, vol. 33, pp. 1179–1191, 2020.

[19] B. Frauenknecht, A. Eisele, D. Subhasish, F. Solowjow, and S. Trimpe, “Trust the model where it trusts itself - model-based actor-critic with uncertainty-aware rollout adaption,” in Forty-first International Conference on Machine Learning, 2024.

[20] M. A. Hernan, Causal Inference: What If, J. M. Robins, Ed. Boca Raton: Taylor & Francis, 2024.

[21] M. J. Van der Laan, S. Rose et al., Targeted learning: causal inference for observational and experimental data. Springer, 2011, vol. 4.

[22] S. Gu, T. Lillicrap, I. Sutskever, and S. Levine, “Continuous deep qlearning with model-based acceleration,” in International conference on machine learning. PMLR, 2016, pp. 2829–2838.

[23] A. Nagabandi, G. Kahn, R. S. Fearing, and S. Levine, “Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning,” in 2018 IEEE international conference on robotics and automation (ICRA). IEEE, 2018, pp. 7559–7566.

[24] Y. Gal, R. McAllister, and C. E. Rasmussen, “Improving pilco with bayesian neural network dynamics models,” in Data-efficient machine learning workshop, ICML, vol. 4, no. 34, 2016, p. 25.

[25] J. Buckman, D. Hafner, G. Tucker, E. Brevdo, and H. Lee, “Sampleefficient reinforcement learning with stochastic ensemble value expansion,” Advances in neural information processing systems, vol. 31, 2018.

[26] K. Chua, R. Calandra, R. McAllister, and S. Levine, “Deep reinforcement learning in a handful of trials using probabilistic dynamics models,” Advances in neural information processing systems, vol. 31, 2018.

[27] S. Levine, A. Kumar, G. Tucker, and J. Fu, “Offline reinforcement learning: Tutorial, review, and perspectives on open problems,” arXiv preprint arXiv:2005.01643, 2020.

[28] G. An, S. Moon, J.-H. Kim, and H. O. Song, “Uncertainty-based offline reinforcement learning with diversified q-ensemble,” Advances in neural information processing systems, vol. 34, pp. 7436–7447, 2021.

[29] C.-A. Cheng, T. Xie, N. Jiang, and A. Agarwal, “Adversarially trained actor critic for offline reinforcement learning,” in International Conference on Machine Learning. PMLR, 2022, pp. 3852–3878.

[30] T. Yu, G. Thomas, L. Yu, S. Ermon, J. Y. Zou, S. Levine, C. Finn, and T. Ma, “Mopo: Model-based offline policy optimization,” Advances in Neural Information Processing Systems, vol. 33, pp. 14 129–14 142, 2020.

[31] R. Kidambi, A. Rajeswaran, P. Netrapalli, and T. Joachims, “Morel: Model-based offline reinforcement learning,” Advances in neural information processing systems, vol. 33, pp. 21 810–21 823, 2020.

[32] M. Uehara, C. Shi, and N. Kallus, “A review of off-policy evaluation in reinforcement learning,” Statistical Science, 2025.

[33] S. A. Murphy, M. J. van der Laan, J. M. Robins, and C. P. P. R. Group, “Marginal mean models for dynamic regimes,” Journal of the American Statistical Association, vol. 96, no. 456, pp. 1410–1423, 2001.

[34] M. Dud´ık, J. Langford, and L. Li, “Doubly robust policy evaluation and learning,” in Proceedings of the 28th International Conference on International Conference on Machine Learning, 2011, pp. 1097–1104.

[35] J. M. Robins, S. Greenland, and F.-C. Hu, “Estimation of the causal effect of a time-varying exposure on the marginal mean of a repeated binary outcome,” Journal of the American Statistical Association, vol. 94, no. 447, pp. 687–700, 1999.

[36] D. Precup, R. S. Sutton, and S. P. Singh, “Eligibility traces for off-policy policy evaluation,” in Proceedings of the Seventeenth International Conference on Machine Learning, ser. ICML ’00, 2000, p. 759–766.

[37] P. R. Rosenbaum and D. B. Rubin, “The central role of the propensity score in observational studies for causal effects,” Biometrika, vol. 70, no. 1, pp. 41–55, 1983.

[38] M. Farajtabar, Y. Chow, and M. Ghavamzadeh, “More robust doubly robust off-policy evaluation,” in International Conference on Machine Learning. PMLR, 2018, pp. 1447–1456.

[39] J. M. Robins, A. Rotnitzky, and L. P. Zhao, “Estimation of regression coefficients when some regressors are not always observed,” Journal of the American statistical Association, vol. 89, no. 427, pp. 846–866, 1994.

[40] N. Jiang and L. Li, “Doubly robust off-policy value evaluation for reinforcement learning,” in International conference on machine learning. PMLR, 2016, pp. 652–661.

[41] P. Thomas and E. Brunskill, “Data-efficient off-policy policy evaluation for reinforcement learning,” in International conference on machine learning. PMLR, 2016, pp. 2139–2148.

[42] A. Bibaut, I. Malenica, N. Vlassis, and M. Van Der Laan, “More efficient off-policy evaluation through regularized targeted learning,” in

International Conference on Machine Learning. PMLR, 2019, pp. 654– 663.

[43] M. J. van der Laan and D. Rubin, “Targeted maximum likelihood learning,” The International Journal of Biostatistics, vol. 2, no. 1, 2006.

[44] R. S. Sutton, D. McAllester, S. Singh, and Y. Mansour, “Policy gradient methods for reinforcement learning with function approximation,” Advances in neural information processing systems, vol. 12, 1999.

[45] A. Hyvarinen, H. Sasaki, and R. Turner, “Nonlinear ica using auxiliary variables and generalized contrastive learning,” in The 22nd international conference on artificial intelligence and statistics. PMLR, 2019, pp. 859–868.

[46] E. Todorov, T. Erez, and Y. Tassa, “Mujoco: A physics engine for modelbased control,” in 2012 IEEE/RSJ international conference on intelligent robots and systems. IEEE, 2012, pp. 5026–5033.

[47] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[48] Y. Luo, H. Xu, Y. Li, Y. Tian, T. Darrell, and T. Ma, “Algorithmic framework for model-based deep reinforcement learning with theoretical guarantees,” in International Conference on Learning Representations, 2019.

[49] L. Nie, M. Ye, Q. Liu, and D. Nicolae, “Vcnet and functional targeted regularization for learning causal effects of continuous treatments,” arXiv preprint arXiv:2103.07861, 2021.