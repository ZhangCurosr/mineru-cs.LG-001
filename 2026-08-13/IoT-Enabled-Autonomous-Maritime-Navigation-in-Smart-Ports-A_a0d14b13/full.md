# IoT-Enabled Autonomous Maritime Navigation in Smart Ports: A Curriculum-Guided Shared Policy Learning Framework

Yuqing Lin, Rangya Zhang, Kum Fai Yuen

Abstract—As smart port infrastructures increasingly rely on autonomous maritime devices enabled by the Internet of Things (IoT), ensuring reliable onboard navigation intelligence has become a critical challenge for safe and scalable operations in congested waterways. This paper investigates onboard autonomous navigation for such IoT devices under partial observability and dense traffic conditions. A curriculum-guided reinforcement learning framework with a shared recurrent policy is developed to enhance temporal reasoning, deployment scalability, and robustness of edge-level decision-making. Centralized training is adopted as an offline design-time strategy, while all navigation actions are executed fully onboard, consistent with IoT edge intelligence paradigms. Extensive simulations in multiple realistic port environments demonstrate that the proposed approach improves navigation reliability, collision avoidance, and training stability compared with standard baseline methods, and generalizes effectively to previously unseen high-density scenarios. The results indicate that curriculum-guided shared learning provides a practical solution for scalable deployment of IoTenabled autonomous maritime devices in smart port operations.

Index Terms—IoT-enabled autonomous navigation, Edge intelligence, Curriculum-guided reinforcement learning, Scalable deployment, Safety-critical navigation

## I. INTRODUCTION AND RELATED WORK

surface vehicles for inspection, logistics, and traffic support in modern smart ports. Major global hubs such as Singapore, Los Angeles, and Rotterdam have actively invested in datadriven port management, IoT-based vessel monitoring, and autonomous navigation trials. Representative initiatives include Singapore’s autonomous navigation programs integrated with the national Vessel Traffic Information System, Los Angeles data-driven berth optimization and vessel traffic management platforms, and Rotterdam’s IoT-enabled vessel monitoring and autonomous vessel testing facilities [1]–[3]. In such IoTenabled port environments, autonomous surface vehicles are expected to operate safely and reliably in congested waterways while relying primarily on onboard sensing and decision intelligence.

A fundamental challenge in this context lies in the design of robust onboard navigation intelligence that can function under partial observability, dense traffic, and dynamic environmental disturbances, without assuming continuous communication or global situational awareness. Reliable collision avoidance and regulation-compliant navigation are therefore critical capabilities for the large-scale deployment of IoT-enabled autonomous maritime devices in smart port operations.

A. Onboard Decision Intelligence for Autonomous Maritime IoT Devices

Deep Reinforcement Learning (DRL) has emerged as a powerful paradigm for onboard sequential decision-making in dynamic and uncertain environments, and has been increasingly adopted in intelligent IoT-enabled autonomous systems for real-time control and decision intelligence [4], [5]. In maritime autonomous navigation, DRL has been applied to a variety of tasks including trajectory tracking, heading control, disturbance rejection, and collision avoidance [6]–[10]. Early applications primarily focused on single-vessel scenarios or simplified interaction settings, demonstrating the feasibility of learning-based controllers as alternatives to classical control methods such as Proportional–Integral–Derivative or Model Predictive Control [11]–[13]. Actor–Critic methods, including A2C and A3C, have further enabled stable learning under stochastic disturbances and partial observability [6], [14], [15].

As autonomous maritime IoT devices are increasingly deployed in congested and dynamic port environments, reliable onboard decision intelligence becomes critical for safe and scalable operation [16], particularly when navigation must handle complex multi-vessel encounters governed by international regulations such as the International Regulations for Preventing Collisions at Sea (COLREGs) [17], [18]. Many existing approaches embed COLREGs compliance through handcrafted reward terms or heuristic penalties [19]–[21], which may constrain adaptability when onboard agents are exposed to diverse traffic patterns and environmental conditions. Recent works have explored more structured reward formulations and attention-based architectures to improve robustness and interpretability [22], [23], yet generalization to realistic, highdensity port operations remains challenging.

To address partial observability and improve temporal reasoning, DRL-based edge intelligence has been widely explored in IoT-enabled autonomous systems for sequential decisionmaking under uncertainty [24]. In particular, recurrent neural networks such as Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) have been integrated into DRL policies to capture temporal dependencies and infer latent environmental states. Recurrent variants of Proximal Policy Optimization (PPO) have demonstrated improved stability and memory-aware decision-making in sequential navigation tasks [25], [26]. In maritime contexts, recurrent PPO-based approaches have shown promising performance in regulationaware navigation and collision avoidance under dense traffic conditions [27]–[29]. These studies indicate that shared recurrent policy structures can support robust onboard navigation intelligence across diverse deployment scenarios.

## B. Curriculum Learning for Robust Onboard Navigation

Curriculum Learning (CL) improves sample efficiency, training stability, and generalization by gradually increasing task difficulty during learning [30], [31]. Common strategies include designer-specified curricula [32], self-paced progression [33], and automatically generated curricula such as reverse curriculum learning [34]. CL has been widely adopted in robotics and navigation to mitigate sparse rewards and accelerate convergence in high-dimensional environments.

In intelligent IoT-enabled robotic and autonomous systems, curriculum-based training has shown strong potential in improving robustness and adaptability under real-world uncertainty [35]. In maritime and marine robotics, similar curriculum-based approaches have demonstrated effectiveness in enhancing DRL-driven trajectory planning, obstacle avoidance under environmental disturbances, and regulation-aware navigation behaviors [36]–[38]. These studies suggest that progressive exposure to increasingly complex navigation scenarios enables onboard agents to acquire safer and more conservative behaviors. However, CL remains underexplored for autonomous surface vessels operating in congested port environments, particularly in conjunction with recurrent policy learning and explicit regulatory constraints. Moreover, existing approaches often rely on heuristic reward shaping rather than embedding navigation regulations directly into the environment and learning process.

## C. Contributions and Scope of This Work

Motivated by the above limitations, this paper focuses on enhancing the onboard decision intelligence of IoT-enabled autonomous surface vehicles through curriculum-guided policy learning. Rather than proposing a new reinforcement learning algorithm, we aim to develop a practical and scalable training framework for robust deployment in smart port environments. The main contributions of this work are summarized as follows:

• A shared recurrent policy learning framework for IoTenabled edge devices, enabling robust onboard decisionmaking under partial observability and dense traffic conditions without relying on continuous cloud communication.

• A curriculum-guided training strategy that progressively exposes the policy to increasing encounter complexity, supporting stable acquisition of safety- and regulationaware navigation behaviors.

• An integrated treatment of safety-critical navigation rules, embedding COLREGs as hard regulatory constraints within the IoT environment and reward formulation to ensure compliant edge-level operations.

• Extensive evaluation across multiple geographically distinct smart port scenarios, demonstrating the deployment scalability and generalization of the shared policy under diverse traffic densities and disturbances.

The remainder of this paper is organized as follows. Section II introduces the proposed curriculum-guided recurrent policy learning framework, including observation, action, and reward design. Section III presents the simulation environments and experimental setup. Section IV reports experimental results and analysis, and Section V concludes the paper with future research directions.

## II. METHODOLOGY

## A. Problem Formulation

We consider onboard autonomous navigation for IoTenabled autonomous surface vehicles operating in congested port environments under partial observability. Each device relies solely on local sensing and onboard decision intelligence to avoid collisions and reach designated targets while complying with navigation regulations. The navigation task is formulated as a POMDP [39]:

$$
\mathcal { P } = \langle \boldsymbol { S } , \boldsymbol { \mathcal { A } } , \boldsymbol { \Omega } , \mathcal { T } , \mathcal { O } , \mathcal { R } , \gamma \rangle ,\tag{1}
$$

where $s$ denotes the set of true environmental states, $\mathcal { A }$ the set of possible actions, Ω the set of possible observations, $\mathcal { T } ( s ^ { \prime } | s , a )$ the transition function, $\mathcal { O } ( o | s , a )$ the observation function, $\mathcal { R } ( s , a )$ the reward function, and $\gamma$ the discount factor.

The objective is to learn a stochastic policy $\pi _ { \boldsymbol { \theta } } ( a | \boldsymbol { o } )$ that maps partial observations to control actions and maximizes the expected discounted return [40]:

$$
\mathbb { E } _ { \pi _ { \theta } } \left[ \sum _ { { t = 0 } } ^ { \infty } \gamma ^ { t } \mathcal { R } ( s _ { t } , a _ { t } ) \right] ,\tag{2}
$$

where $\pi _ { \boldsymbol { \theta } } ( a | \boldsymbol { o } )$ denotes the policy distribution over actions given observation $o , \mathcal { R } ( s _ { t } , a _ { t } )$ is the reward obtained at time $t ,$ and $\gamma \in [ 0 , 1 )$ balances short- and long-term rewards. The expectation is taken over trajectories generated by following $\pi _ { \theta }$

In realistic port operations, autonomous maritime IoT devices operate under uncertainty and partial observability due to limited sensor range, occlusions, limited communication bandwidth, and environmental disturbances such as currents and wind [25], [41]. Each device relies on onboard sensors $( \mathrm { e . g . }$ radar, LiDAR, proximity sensors) without access to the full global state; real-time synchronization is typically hindered by limited communication bandwidth in remote or congested IoT environments. This motivates the POMDP formulation as interaction patterns must be inferred from local observations.

To improve robustness under partial observability, we adopt a shared recurrent policy learning approach, in which a single policy network is reused across identical devices deployed in shared environments. Reuse of policies enables efficient model deployment at scale while maintaining fully onboard execution. Recurrent LSTM units are integrated into the policy to capture temporal dependencies and infer latent environmental states from observation histories.

## B. Shared Recurrent PPO Architecture

To enable robust onboard collision avoidance for IoTenabled autonomous surface vehicles operating in congested port environments, we adopt PPO as the underlying learning framework due to its stability and effectiveness in continuous control tasks. A shared recurrent policy is trained offline using experience collected from multiple autonomous surface vehicles operating in parallel simulation environments, and subsequently deployed onboard each device for independent execution. This shared-policy design facilitates efficient model reuse across identical platforms and improves training stability under diverse initial conditions and encounter scenarios. CL is employed during training to progressively increase task complexity, enhancing learning efficiency and policy generalization.

To address partial observability arising from limited sensor range and occlusions caused by surrounding vessels and obstacles, LSTM units are embedded within both the actor and critic networks. This recurrent design enables the policy to capture temporal dependencies and infer latent environmental states from observation histories, which is consistent with the POMDP formulation of the navigation task.

The overall architecture, illustrated in Fig. 1, integrates recurrent policy learning with fully onboard execution to handle dynamic and partially observable maritime environments. During deployment, each autonomous surface vehicle operates solely based on local observations, including radarbased detection of static and dynamic obstacles (buoys and other vessels), relative coordinates to the navigation target, normalized distances to environmental boundaries, relative positions of nearby vessels, and ego-motion variables. The continuous action space consists of commanded yaw rate and forward surge velocity.

Observations are processed sequentially through an LSTM layer to maintain an internal hidden state over time. The LSTM output is passed through fully connected layers to generate the parameters of a Gaussian action distribution, from which continuous control actions are sampled. This architecture enables temporal reasoning and improves decision robustness under uncertainty. During training, experience trajectories from multiple simulated vehicles are aggregated to update the shared policy parameters. Both actor and critic networks share the same recurrent backbone, consisting of an LSTM layer followed by two linear layers, with the actor outputting Gaussian policy parameters and the critic estimating the state value function. PPO with a clipped surrogate objective is adopted to ensure stable policy updates, while Generalized Advantage Estimation is used to reduce variance in advantage computation.

By combining a shared recurrent policy with curriculumguided training, the proposed framework supports robust onboard decision-making under partial observability and dense traffic conditions, while remaining scalable for deployment across multiple IoT-enabled autonomous surface vehicles operating in realistic port environments.

![](images/86d220f254d1e754e3d6a3952c191b9ddc31590b9515898f469491c7dc558340.jpg)  
Fig. 1: Framework of the proposed onboard recurrent policy learning approach for IoT-enabled autonomous surface vehicles. The policy is trained offline and executed fully onboard using local observations.

## C. Observation and Action Space

Onboard autonomous navigation for IoT-enabled autonomous surface vehicles requires informative yet partial perception of surrounding traffic and obstacles under realistic sensor constraints. The observation space is designed to capture both geometric layout and dynamic environmental information while accounting for limitations such as finite radar range and occlusion. Radar-based perception discretizes the sensing field into angular sectors, with each sector encoding the normalized distance and bearing to the nearest detected object. All quantities are expressed in the body-fixed frame to ensure rotational invariance and normalized according to sensor and velocity limits. Temporal consistency across observations is preserved through the use of an LSTM-based policy, which is essential for decision-making under POMDP conditions.

To structure onboard situational awareness, an anisotropic elliptical ship domain with direction-dependent radii $R _ { \mathrm { f o r e } } , R _ { \mathrm { a f t } } , R _ { \mathrm { p o r t } } , R _ { \mathrm { s t a r b } }$ is adopted to reflect varying clearance priorities in different directions (Fig. 2) [42], [43]. This domain serves as a perception and safety abstraction that adapts with vessel orientation and supports regulation-aware risk assessment compatible with COLREGs-based reward design. Within this representation, each device perceives its own kinematic state, navigation goal, surrounding traffic vessels with heterogeneous control behaviors, and static obstacles. The resulting observation vector is defined as:

![](images/3f20feca513464e505cf6dbfd939d8b8f0805049657227f903a0e473ee86cd3f.jpg)  
Fig. 2: Ship perception model and onboard observation structure.

$$
\begin{array} { r l } & { \underbrace { \left[ \underbrace { x _ { o } , z _ { o } , v _ { o x } , v _ { o z } } _ { \mathrm { O w n ~ s h i p } } , \underbrace { x _ { t } , z _ { t } } _ { \mathrm { G o a l } } , \underbrace { x _ { a _ { 1 } } , z _ { a _ { 1 } } , x _ { u _ { 1 } } , z _ { u _ { 1 } } , . . . , } _ { \mathrm { S u r r o u n d i n g ~ v e s s e l s } } , \underbrace { x _ { \cdot } , \tilde { y } _ { \cdot } , \tilde { y } _ { \cdot } , \tilde { y } _ { \cdot } } _ { \mathrm { G o a l } } , \underbrace { x _ { \cdot } , \tilde { y } _ { \cdot } , \tilde { y } _ { \cdot } } _ { \mathrm { S u r r o u n d i n g ~ v e s s e l s } } , \right. } _ { \mathrm { R a d a r ~ o b s t a c l e s } } } \end{array}\tag{3}
$$

where all positions and velocities are expressed relative to the own ship, and $( d _ { i } , \theta _ { i } )$ denote radar-derived obstacle ranges and bearings.

For motion control, a simplified three-degree-of-freedom (3- DOF) kinematic model in the horizontal plane is adopted, including surge $u ,$ sway $v ,$ and yaw rate r, while neglecting vertical-plane dynamics to reduce modeling complexity [44]. The body-fixed reference frame is centered at the vessel’s center of mass G, with heading angle ψ defined relative to the global frame (Fig. 3). The continuous two-dimensional action space is defined as:

$$
[ \mathrm { a n g u l a r } _ { z } , \ \mathrm { l i n e a r } _ { x } ] = [ r , \ u ]\tag{4}
$$

where linear<sub>x</sub> corresponds to the commanded surge velocity and angular to the yaw rate. This high-level action abstraction decouples decision-making from low-level actuation details such as propeller thrust and rudder dynamics, enabling efficient onboard policy learning and robust generalization under environmental disturbances. The resulting control interface supports a range of navigation behaviors, including trajectory tracking, maneuvering in constrained waterways, and regulation-compliant collision avoidance.

## D. Reward Shaping Strategy

To support robust onboard decision-making for IoT-enabled autonomous surface vehicles operating in congested port environments, we design a composite reward that balances navigation efficiency, safety preservation, and regulation-aware risk mitigation. Rather than encoding explicit maneuvering rules, the reward function provides continuous guidance signals derived from geometric relationships, proximity risk, and encounter context, enabling effective learning under partial observability.

![](images/16ee03d4e25823eb3eb75a13641cb8367c7daf9bb50b0368960b17af15ef6a38.jpg)  
Fig. 3: 3-DOF kinematic model and onboard action space.

The overall reward consists of four dense components, heading alignment $r _ { \mathrm { y a w } }$ , distance progress $r _ { \mathrm { d i s t } }$ , regulationaware interaction $r _ { \mathrm { r e g } } ,$ , and proximity-based safety penalty r<sub>prox</sub>, supplemented by sparse terminal rewards.

a) Heading alignment: This term encourages progress toward the navigation target by aligning the vessel’s heading with the desired bearing:

$$
r _ { \mathrm { y a w } } = \cos ( \psi - \psi _ { d } ) ,\tag{5}
$$

where $\psi$ denotes the current heading and $\psi _ { d }$ the bearing toward the target.

b) Distance progress: To promote efficient navigation, distance progress is rewarded as:

$$
r _ { \mathrm { d i s t } } = \lambda _ { \mathrm { d i s t } } \cdot ( \mathrm { D i s t } _ { t - 1 } - \mathrm { D i s t } _ { t } ) ,\tag{6}
$$

where $\lambda _ { \mathrm { d i s t } }$ is a scaling coefficient and Dist<sub>t</sub> denotes the normalized distance to the target at time t.

c) Regulation-aware interaction reward: To incorporate maritime navigation regulations as part of onboard risk assessment [45], [46], a regulation-aware interaction reward $r _ { \mathrm { r e g } }$ is activated when surrounding traffic vessels enter the anisotropic ship domain of the own vessel (Fig. 2). The ship domain is parameterized by direction-dependent radii $R _ { \mathrm { f o r e } } , R _ { \mathrm { a f t } } , R _ { \mathrm { p o r t } } ,$ and $R _ { \mathrm { s t a r b } } .$ , providing a continuous representation of encounter proximity and relative geometry.

Encounter contexts are categorized based on relative bearing and domain location (Fig. 4), and each context is associated with a signed shaping signal that reflects relative collision risk and navigational responsibility:

• Head-on encounters emphasize lateral separation.

• Crossing encounters differentiate between yielding and stand-on responsibilities.

• Overtaking encounters emphasize longitudinal clearance.

• Parallel navigation emphasizes safe lateral spacing.

![](images/54c71a46fb45fedaf920602526deaa68113f8df3bdbd23cd5ca42335677bd4c1.jpg)  
Fig. 4: Regulation-aware encounter sectors in the anisotropic ship domain.

The total regulation-aware reward aggregates contributions from all surrounding traffic vessels:

$$
\begin{array} { r l } { \displaystyle r _ { \mathrm { r e g } } = \sum _ { j \in \mathcal { T } } \left( \mathcal { T } _ { \mathrm { h e a d - o n } } ( j ) f _ { \mathrm { h e a d - o n } } + \mathcal { T } _ { \mathrm { c r o s s - g i v e } } ( j ) f _ { \mathrm { c r o s s - g i v e } } \right. } & { } \\ { \displaystyle \left. + \mathcal { T } _ { \mathrm { o v e r t a k e } } ( j ) f _ { \mathrm { o v e r t a k e } } + \mathcal { T } _ { \mathrm { c r o s s - s t a n d } } ( j ) f _ { \mathrm { c r o s s - s t a n d } } \right. } & { } \\ { \displaystyle \left. + \mathcal { T } _ { \mathrm { p a r a l l e l } } ( j ) f _ { \mathrm { p a r a l l e l } } \right) , } \end{array}\tag{7}
$$

where T denotes the set of surrounding traffic vessels detected within sensing range.

d) Proximity-based safety penalty: To discourage unsafe proximity, a continuous penalty is applied when the distance d to a nearby vessel falls below a minimum safety threshold d<sub>min</sub>:

$$
r _ { \mathrm { p r o x } } = \left\{ \begin{array} { l l } { - \lambda _ { \mathrm { p r o x } } \left( 1 - \frac { d } { d _ { \mathrm { m i n } } } \right) , } & { d < d _ { \mathrm { m i n } } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{8}
$$

e) Terminal rewards: Sparse terminal rewards reflect task completion and safety outcomes:

• Successful arrival at the navigation target: +1.0.

• Collision or violation of navigable boundaries: −1.0.

• Episode timeout without reaching the target: −0.6.

f) Final reward: The per-step reward is computed as:

$$
r _ { t } = r _ { \mathrm { y a w } } + r _ { \mathrm { d i s t } } + r _ { \mathrm { r e g } } + r _ { \mathrm { p r o x } } .\tag{9}
$$

This reward formulation integrates geometric guidance, safety margins, and regulation-aware risk cues into a unified onboard shaping strategy, enabling sample-efficient learning and robust navigation under dense traffic and partial observability without relying on explicit rule execution.

## E. CL Strategy

Training autonomous surface vehicles for reliable onboard navigation in congested port environments with static obstacles and dynamic surrounding traffic presents challenges in convergence, safety preservation, and generalization. To address these issues, we adopt a hybrid CL strategy that combines designerspecified task staging [30] with self-paced progression driven by policy performance [33]. Task difficulty is gradually increased by adjusting environmental and operational conditions, and policy advancement occurs only when predefined proficiency criteria are satisfied. This structured learning process improves sample efficiency, convergence stability, and robustness for onboard decision-making under partial observability.

Each curriculum stage varies along three key dimensions:

• Deployment Scale: Training begins with a single autonomous device operating independently, and progressively increases the number of simultaneously deployed devices. This allows the shared policy to mature under increasing traffic density while preserving fully onboard decision-making without explicit coordination.

• Traffic Interaction Complexity: Early stages involve simple or limited surrounding traffic. Later stages introduce heterogeneous autonomous surface vehicles executing randomized regulation-consistent behaviors (e.g., headon, overtaking, crossing, and parallel encounters), increasing encounter diversity and improving robustness to unseen traffic patterns.

• Environmental Complexity: Static obstacle density, navigational constraints, and environmental disturbances (e.g., wind and current) are gradually increased to better approximate real-world operating conditions and enhance adaptability.

Table I summarizes the curriculum structure. Stage 1 (Port of Los Angeles) focuses on isolated interaction learning under a deterministic traffic configuration. Stage 2 (Port of Singapore) introduces moderate traffic variability through randomized encounter scenarios. Stage 3 (Port of Rotterdam) exposes the policy to dense and heterogeneous traffic conditions to promote generalization under realistic operational complexity. Curriculum progression is triggered by either a fixed number of training episodes or performance thresholds (e.g., success rate ≥ 80% over a moving evaluation window).

This staged training strategy enables stable policy devel opment under safety-critical and partially observable conditions, while providing a systematic framework for evaluating onboard navigation performance across increasingly realistic and complex port environments.

## III. SIMULATION ENVIRONMENT

## A. Platform Overview

The simulation framework is implemented using Unity ML-Agents, which provides a flexible 3D physics-based environment for training and evaluating onboard decision policies. The Unity PhysX engine enables realistic vessel dynamics, collision handling, and motion modeling, while high-fidelity rendering supports accurate scenario representation.

Each autonomous surface device is equipped with simulated maritime sensing capabilities, including radar-like proximity sensing for obstacle and traffic detection, lidar-style range measurements for spatial awareness, and relative goal-position inputs for navigation. These complementary sensing modalities allow the onboard policy to perceive local surroundings and make informed navigation decisions under dynamic traffic and environmental uncertainty. Vessel motion is modeled using a simplified differential thrust scheme, where the policy outputs continuous commands for angular velocity and forward propulsion that are mapped to left and right thruster forces within physical constraints. The vessel model incorporates inertia, turning delay, and velocity-dependent dynamics, making navigation and collision avoidance non-trivial. Boundary interactions and buoyancy effects are handled by the underlying physics engine.

<table><tr><td colspan="5">Stage 1: Port of Los Angeles</td><td colspan="5">Stage 2: Port of Singapore</td><td colspan="5">Stage 3: Port of Rotterdam</td></tr><tr><td>USVs</td><td></td><td>ASVs Encounter Obstacles Types</td><td></td><td>Weather</td><td>USVs</td><td>ASVs Encounter</td><td>Types</td><td>Obstacles</td><td>Weather</td><td>USVs</td><td>ASVs Encounter Types</td><td></td><td>Obstacles</td><td>Weather</td></tr><tr><td>1</td><td>0</td><td>None</td><td>Sparse</td><td>None</td><td>2</td><td>2</td><td>Crossing / Over- taking / Parallel (random)</td><td>Head-on / Moderate</td><td>None</td><td>4+</td><td>4+ ized)</td><td>Mixed en- counters (random-</td><td>Dense</td><td>Wind + Waves</td></tr><tr><td>1</td><td>1</td><td>Head-on / Crossing 1 Over- taking 1 Parallel</td><td>Sparse</td><td>None</td><td>2</td><td>4 1</td><td>Head-on / Crossing Over- taking 1 Parallel (random)</td><td>Dense</td><td>Waves</td><td></td><td></td><td></td><td></td><td></td></tr></table>

TABLE I: Curriculum stages with progressively increasing difficulty and randomized ASV encounters per episode. Stage 1 corresponds to the Port of Los Angeles (open and sparse traffic), Stage 2 to the Port of Singapore (semi-structured and high-traffic), and Stage 3 to the Port of Rotterdam (hybrid, complex transitions).

A shared recurrent PPO policy with LSTM memory is used to enable sequential decision-making under partial observability. The shared policy allows consistent deployment across multiple autonomous devices while preserving fully onboard execution without requiring explicit coordination, supporting robust navigation across diverse operational scenarios.

## B. Scenario Design

To enhance generalization and real-world robustness for onboard autonomous navigation, we adopt a staged curriculum learning (CL) framework across three representative port environments of increasing operational complexity: Los Angeles (open water, low traffic), Singapore Pasir Panjang (semistructured, dense traffic), and Rotterdam (hybrid layout with complex transitions). Task difficulty is progressively increased across stages, allowing the onboard policy to mature in handling collision avoidance, regulation-aware navigation, and adaptive behavior under diverse maritime conditions. Progression between stages is governed by either a fixed episode budget or performance thresholds.

a) Stage 1: LA Port (Open, Low-Traffic Waters): A simplified offshore environment inspired by the Port of Los Angeles with low traffic density and large maneuvering space (Fig. 5). A single surrounding traffic vessel executes a deterministic regulation-consistent encounter (e.g., head-on, crossing, overtaking, or parallel), enabling the autonomous device to develop stable navigation and energy-efficient maneuvering in a controlled setting.

![](images/ca05416dc13a4276cb468f478d5a8808d35cb435175f29b49fe8b6168846bce6.jpg)  
(a) LA port waters from OpenSeaMap.

![](images/35fa5aa968e92d7a99f5b7a58fe3c4c7b8ec1ae68e1adfbdf7725fa3a878b7fd.jpg)  
(b) Simulation training environment based on LA port layout.

Fig. 5: Mapping real-world maritime structure to simulation at the port of Los Angeles.  
![](images/d23d641efa3a7aeeabcdd93e2c92701cba59994fdabdb152d016db0a2a94d41d.jpg)

![](images/c7a13c34205853021a3aa5a702d408055c31a54acbb80f05970f3bd64b9be110.jpg)  
(a) Singapore port waters from OpenSeaMap.  
(b) Simulation training environment based on Singapore port layout.  
Fig. 6: Mapping real-world maritime structure to simulation at the Port of Singapore.

b) Stage 2: Singapore Port (Semi-Structured, Dense Traffic): A constrained fairway environment based on the Pasir Panjang Container Terminal (Fig. 6), featuring narrow channels and increased traffic density. Two surrounding traffic vessels with randomized regulation-consistent behaviors are introduced, exposing the onboard policy to more complex interaction patterns under congested navigational constraints.

c) Stage 3: Rotterdam Port (Hybrid, Complex Transitions): A hybrid environment modeled after the Port of

![](images/edc8c053bb9797f407cb21da691d7efc0ed4ef3302ae1cb7e31284722dcebf27.jpg)  
(a) Rotterdam port waters from OpenSeaMap.

![](images/e8dd552f4346b1ea223104398371bcfcfffbad759cda73097cc94f004000e299.jpg)  
(b) Simulation training environment based on Rotterdam port layout.  
Fig. 7: Mapping real-world maritime structure to simulation at the port of Rotterdam.

Rotterdam (Fig. 7) containing both narrow channels and open basins. Multiple autonomous surface devices operate concurrently within dense and heterogeneous traffic, where surrounding vessels follow randomized regulation-consistent encounter patterns. This stage introduces complex traffic merging, environmental disturbances, and long-horizon decision challenges, promoting robust policy generalization under realistic operational conditions.

Table I summarizes the curriculum structure. Across stages, scripted surrounding vessels generate diverse and realistic encounter conditions, enabling the onboard policy to progressively adapt from low-risk scenarios to dense and uncertain traffic environments. This geography-informed curriculum design supports scalable and transferable onboard navigation suitable for deployment in diverse real-world port settings.

## C. Training Process and Parameters

The training framework in Figure 8 employs a hybrid CL strategy to incrementally expose agents to progressively challenging multi-vessel navigation scenarios. Three stages of increasing complexity are used: (1) open, low-traffic waters in Los Angeles, (2) semi-structured, high-traffic port approaches in Singapore, and (3) hybrid port layouts in Rotterdam with narrow waterways, high vessel density, complex traffic transitions, and environmental disturbances (wind and waves). A curriculum scheduler monitors success rates and advances to the next stage once proficiency thresholds are met, ensuring stable skill acquisition.

At the beginning of each episode, a curriculum scheduler selects the training environment within a Unity-based maritime simulator containing static obstacles, dynamic surrounding vessels following regulation-consistent trajectories, and stochastic environmental disturbances. Each autonomous surface device receives multimodal observations, including relative goal position, radar-like proximity sensing, and lidarbased range measurements. Observations are normalized and temporally stacked to form a sequential state representation for onboard decision-making.

The proposed policy follows a shared recurrent PPO architecture, consisting of a multilayer perceptron encoder and an LSTM module to capture temporal dependencies under partial observability. The actor head outputs Gaussian-distributed steering and propulsion commands, while the critic head estimates state values for policy optimization. A single shared policy is trained and then consistently deployed across multiple autonomous devices, enabling scalable onboard execution without requiring explicit coordination.

![](images/591b3107d9cff78a5b0828c2dda4c28a25be558fb716b36f2a0762e1dd5d1421.jpg)  
Fig. 8: Training process of curriculum-guided shared recurrent PPO.

Training proceeds iteratively through parallel trajectory collection, reward evaluation using a composite shaping function (navigation efficiency, regulation-aware safety, and collision avoidance), and PPO updates based on generalized advantage estimation and a clipped surrogate objective. Performance indicators such as navigation success rate, collision frequency, and regulation-compliance score determine curriculum progression, allowing the policy to gradually adapt to increasingly complex operational conditions. Training is executed asynchronously using the PPO trainer in Unity ML-Agents, and key hyperparameters are summarized in Table II.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Batch size</td><td>2048 steps</td></tr><tr><td>Buffer size</td><td>40960</td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Discount factor (γ)</td><td>0.99</td></tr><tr><td>GAE lambda</td><td>0.95</td></tr><tr><td>PPO clip range</td><td>0.2</td></tr><tr><td>LSTM hidden size</td><td>128</td></tr><tr><td>Number of environments</td><td>8 parallel simulations</td></tr><tr><td>Total training steps</td><td>5 million</td></tr></table>

TABLE II: Key PPO hyperparameters used during training with Unity ML-Agents.

## IV. EXPERIMENTS AND RESULTS

## A. Algorithm Comparison

All experiments are run on a workstation with an NVIDIA RTX 4080 GPU and Intel i7 CPU. Each training run lasts 5M interaction steps (6–8 hours), using Unity ML-Agents with 8 parallel environments. The curriculum progresses through three port scenarios: Stage 1 (Los Angeles) features openwater navigation with static obstacles; Stage 2 (Singapore) introduces congested harbor traffic with multiple COLREGscompliant vessels; Stage 3 (Rotterdam) adds hybrid layouts with wind, waves, dense traffic, and complex docking. Agents advance to the next stage upon achieving an 80% success rate over 100 evaluation episodes.

We compare the proposed shared recurrent PPO framework against three representative learning paradigms: (1) DDPG [47], a deterministic off-policy method without curriculum or recurrence; (2) SAC [48], an entropy-regularized off-policy method without memory; and (3) standard PPO, trained without curriculum learning or recurrence. These baselines evaluate how different learning paradigms influence onboard navigation reliability and scalability for IoT-enabled autonomous devices.

To ensure fair comparison, all algorithms share identical observation, action, reward formulation, and network capacity. Differences arise only from learning paradigm (on/off-policy), recurrent memory, and curriculum scheduling. The objective is not to outperform algorithmic state-of-the-art, but to validate the effectiveness of a deployable shared onboard policy framework under realistic maritime IoT conditions.

As shown in Fig. 9, the proposed PPO demonstrates more stable convergence and higher final performance under complex traffic conditions, validating the effectiveness of shared recurrent policy learning for onboard navigation. SAC learns rapidly in early stages but plateaus in complex scenarios due to the absence of memory and curriculum exposure. Standard PPO converges slower and achieves lower final performance, while DDPG exhibits unstable learning under partial observability.

Table III confirms these trends across metrics. In Stage 1, all methods perform well, though SR-PPO achieves the highest success (96.2%) and lowest collisions (2.1%). In Stage 2, traffic complexity widens the gap: SR-PPO maintains strong performance with 90.5% success and 4.0% collisions, slightly ahead of SAC and clearly more stable than PPO and DDPG. In Stage 3, under dense interactions and disturbances, SR-PPO continues to demonstrate robust behavior with 85.6% success and 6.3% collisions. SAC and PPO degrade under increasing complexity, while DDPG shows the weakest performance. Generalization scores follow a similar trend, with SR-PPO consistently achieving strong performance across stages.

Therefore, curriculum learning and recurrent policy design improve the robustness, safety, and reliability of onboard navigation under partial observability. While SAC remains competitive in simpler environments, the absence of recurrence and curriculum limits its performance in complex scenarios. The results highlight the effectiveness of shared onboard policy learning for scalable and reliable autonomous navigation in IoT-enabled maritime environments.

## B. Ablation Study Results

We evaluate the contribution of each core component through ablation studies, selectively removing or modifying elements of the proposed framework. Figure 10 summarizes performance across navigation effectiveness, safety, and regulation compliance. The first six configurations are trained using the full curriculum and evaluated in Stage 3, while the last three are trained and evaluated solely within Stage 1, Stage 2, or Stage 3 to analyze the effect of curriculum progression.

![](images/2a090c15fd4ff5f71b9c3236dab386b44940b6db18436b9ff117d189d305cf1e.jpg)  
Fig. 9: Convergence comparison of onboard navigation policies under curriculum training.

![](images/1a45f795c2b16a005559409e99438d61db2a1a9477da6316e5789c2ec23afc16.jpg)  
Fig. 10: Ablation study of key design choices. Metrics for the first six configurations are averaged in Stage 3. The last three configurations are trained and evaluated independently in Stage 1, 2, or 3 to examine CL benefits.

Success Rate (%): The proposed method achieves the highest overall success rate, improving by approximately 10– 15% compared to Individual-policy PPO and No-Curriculum PPO. Reward-ablation variants (COLREGs-only or Distanceonly) show moderate degradation (6–12%), while Stage 3- only training experiences over 20% reduction, indicating the importance of shared policy learning and staged curriculum exposure for reliable onboard navigation.

Collision Rate (%): The proposed model maintains the lowest collision frequency. Individual-policy PPO increases collisions by approximately 40%, while No-Curriculum PPO significantly increases risk. Training directly in Stage 3 without curriculum results in substantially higher collision rates, demonstrating reduced stability under complex traffic conditions.

<table><tr><td>Algorithm</td><td>Success Rate (%)</td><td>Collision Rate (%)</td><td>Min. Passing Dist. (m)</td><td>Episode Dura- tion (steps)</td><td>Generalization Score (%)</td></tr><tr><td colspan="6">Stage 1: Los Angeles (Open Water)</td></tr><tr><td>Proposed PPO</td><td> $9 6 . 2 \pm 2 . 1 $ </td><td> $2 . 1 \pm 1 . 0$ </td><td> $2 . 2 \pm 0 . 4$ </td><td> $8 2 . 4 \pm 6 . 7$ </td><td> $9 1 . 3 \pm 3 . 2 $ </td></tr><tr><td>SAC</td><td> $9 4 . 1 \pm 2 . 8 $ </td><td> $2 . 3 \pm 1 . 4$ </td><td> $2 . 2 \pm 0 . 4$ </td><td> $8 0 . 5 \pm 7 . 1 $ </td><td> $8 9 . 4 \pm 4 . 2 $ </td></tr><tr><td>DDPG</td><td> $8 7 . 9 \pm 4 . 5$ </td><td> $8 . 9 \pm 3 . 1$ </td><td> $2 . 5 \pm 0 . 7$ </td><td> $1 0 2 . 3 \pm 1 0 . 1 $ </td><td> $7 6 . 2 \pm 6 . 3$ </td></tr><tr><td>PPO</td><td> $9 0 . 6 \pm 3 . 8$ </td><td> $6 . 4 \pm 2 . 9$ </td><td> $2 . 8 \pm 0 . 5$ </td><td> $9 3 . 7 \pm 7 . 9$ </td><td> $8 1 . 1 \pm 4 . 6$ </td></tr><tr><td colspan="6">Stage 2: Singapore (High Traffic)</td></tr><tr><td>Proposed PPO</td><td> $9 0 . 5 \pm 3 . 3 $ </td><td> $4 . 0 \pm 1 . 8$ </td><td> $2 . 8 \pm 0 . 3$ </td><td> $9 6 . 7 \pm 7 . 5$ </td><td> $8 8 . 9 \pm 3 . 9$ </td></tr><tr><td>SAC</td><td> $8 7 . 3 \pm 4 . 1$ </td><td> $4 . 2 \pm 3 . 1$ </td><td> $2 . 7 \pm 0 . 5$ </td><td> $1 0 0 . 2 \pm 8 . 6$ </td><td> $8 0 . 6 \pm 5 . 5$ </td></tr><tr><td>DDPG</td><td> $6 8 . 5 \pm 5 . 7 $ </td><td> $2 1 . 3 \pm 6 . 4$ </td><td> $2 . 1 \pm 0 . 6$ </td><td> $1 2 6 . 1 \pm 1 1 . 8$ </td><td> $6 0 . 9 \pm 7 . 3$ </td></tr><tr><td>PPO</td><td> $8 3 . 7 \pm 4 . 6$ </td><td> $1 2 . 1 \pm 4 . 3$ </td><td> $2 . 5 \pm 0 . 5$ </td><td> $1 0 8 . 5 \pm 8 . 6 $ </td><td> $7 4 . 4 \pm 6 . 1$ </td></tr><tr><td colspan="6">Stage 3: Rotterdam (Hybrid Conditions)</td></tr><tr><td>Proposed PPO</td><td> $8 5 . 6 \pm 4 . 0$ </td><td> $6 . 3 \pm 2 . 2$ </td><td> $3 . 5 \pm 0 . 4$ </td><td> $1 0 9 . 9 \pm 9 . 1 $ </td><td> $8 5 . 2 \pm 4 . 7$ </td></tr><tr><td>SAC</td><td> $7 8 . 2 \pm 6 . 2$ </td><td> $1 8 . 1 \pm 5 . 4$ </td><td> $2 . 5 \pm 0 . 6$ </td><td> $1 2 1 . 8 \pm 1 0 . 4$ </td><td> $6 9 . 8 \pm 7 . 9$ </td></tr><tr><td>DDPG</td><td> $5 5 . 7 \pm 7 . 1$ </td><td> $3 4 . 5 \pm 7 . 3$ </td><td> $1 . 9 \pm 0 . 5$ </td><td> $1 3 9 . 6 \pm 1 2 . 7$ </td><td> $4 9 . 3 \pm 8 . 4$ </td></tr><tr><td>PPO</td><td> $7 0 . 2 \pm 5 . 7$ </td><td> $1 6 . 8 \pm 4 . 7$ </td><td> $2 . 4 \pm 0 . 5$ </td><td> $1 2 3 . 1 \pm 9 . 4$ </td><td> $6 6 . 7 \pm 7 . 4$ </td></tr></table>

TABLE III: Performance comparison of onboard navigation policies across three IoT-enabled port environments.

Min. Passing Distance (m): The proposed method maintains the largest safety margins, approximately 25% higher than reward-ablation variants. No-Curriculum and Stage-isolated models tend to generate more aggressive trajectories with reduced safety buffers.

Episode Duration: The proposed policy completes navigation tasks efficiently while maintaining safety. Individualpolicy PPO and Distance-only variants require up to 15% longer durations, whereas Stage 3-only training shows the slowest convergence (exceeding 120 steps), indicating unstable decision-making under complex conditions.

Regulation Compliance:

• Head-On Right Turn Rate (Rule 14): Proposed method achieves substantially higher compliance compared to Stage 3-only training.

• Overtake Clearance Rate (Rule 13): Composite reward and proposed method maintain stable clearance margins, while Distance-only shows noticeable degradation.

• Crossing Right Yield Rate (Rule 15): Proposed approach maintains consistently higher compliance than non-composite reward variants and Stage 3-only training.

• Parallel Safe Distance Rate: Highest safety margins are maintained by proposed method and composite reward models, while other variants show reduced separation distances.

Taken together, the combination of shared recurrent policy learning, staged curriculum training, and composite reward design improves the robustness, safety, and reliability of onboard navigation under partial observability. These results further support the effectiveness of shared onboard policy learning for scalable and dependable autonomous navigation in IoTenabled maritime environments.

## C. Behavioral Analysis and Generalization Insights

To evaluate the effect of CL and the impact of the LSTMbased shared recurrent PPO architecture on agent behavior, we visualize representative trajectories from the Stage 1 (1vs4) LA port scenario (Fig. 11). The USV navigates from the Start Zone (bottom-left, red dashed box) to the End Zone (top-center, red dashed box) under four COLREGs-defined encounters. Trajectory colors indicate interaction types, with the light blue vessel as the ego USV (Own Ship), white vessels as ASVs (Target Ships), white dashed arrows as ASV planned paths, and solid colored lines as USV motions. Only one start–end configuration is shown for clarity; actual experiments randomize both.

The pink line shows the baseline without interference. The orange trajectory (crossing, Rule 15) depicts the OS yielding to starboard when the TS is detected on its right front quarter, adjusting speed/heading to pass astern. The green trajectory (overtaking, Rule 13) shows the OS approaching from behind (< 22.5<sup>◦</sup>) and maneuvering around the TS with a safe passing side. The blue trajectory (head-on, Rule 14) involves both vessels altering course to starboard at a 5–10<sup>◦</sup> approach. The purple trajectory (parallel) maintains lateral distance with minimal deviation. Stage 1’s low-traffic setting allows gradual acquisition of fundamental collision-avoidance behaviors, where parallel interactions cause minimal deviation, headon and crossing require larger adjustments, and overtaking produces the most substantial detours.

Stage 2 increases complexity with two busy terminals, multiple buoys, stationary workboats, and dynamic ASVs selecting two distinct COLREGs encounters per episode. In 2vs2 (Fig. 12a), USVs generally maintain smooth paths, with overtaking resolved by starboard course changes and parallel encounters showing minimal deviation. Head-on maneuvers involve early right turns, and crossing cases prompt gradual mid-path adjustments. In 2vs4 (Fig. 12b), higher density leads to greater baseline deviations. Overtaking requires larger lateral shifts, crossing triggers earlier turns, head-on cases still resolve via rightward avoidance but with larger detours, and parallel encounters sometimes drift due to cumulative avoidance.

![](images/c44c88b25b5c72dd3c6d22ac8e15426cb1c70058193d23600277fcee316353b8.jpg)  
Fig. 11: Representative onboard navigation trajectories under four COLREGs encounter conditions in the Stage 1 LA port scenario. Pink: baseline without interaction; light blue: ego USV; white: target vessels (ASVs); dashed arrows: ASV planned paths; solid lines: USV trajectories.

Comparisons show that traffic density consistently increases path deviation and prompts earlier avoidance in 2vs4. Overtaking produces the largest offsets, crossing is resolved earlier in denser traffic, and parallel/head-on remain structured but less precise.

Stage 3 (4vs4, 6vs6, 12vs12) in Rotterdam adds narrow waterways, dense multi-vessel interactions, and wave/wind disturbances. In 4vs4 (Fig. 13a), all four encounter types show early avoidance, with deviations shaped by narrow-channel constraints. In 6vs6 (Fig. 13b), pre-emptive avoidance persists but trajectories align closer to baseline, indicating adaptability. Parallel nearly matches the baseline, and head-on/overtaking are smoother with maintained clearance. The 12vs12 case (Fig. 13c) tests generalization in highly congested, disturbed waters. Agents anticipate conflicts early, maintain separation bands in multi-crossings, and preserve regulation compliance with reduced oscillations. Shared recurrent PPO with CL enables coordinated, temporally aware navigation robust to density and disturbances.

Building on trajectory insights, aggregated metrics (Fig. 14) assess robustness, safety, and rule compliance across stages. Stage 1 yields >96% success with <2.5% collisions and nearperfect regulation compliance. Stage 2 sustains ∼90% success in 2vs2 but drops to 87.3% in 2vs4, with increased collisions and longer episodes. Rule compliance also declines slightly under higher density. Stage 3 shows moderate success drop (83.5% → 80.1%) but improved minimum passing distances and consistent COLREGs adherence, reflecting stable, safe behavior under maximum complexity.

![](images/ab88b6089ebf45434658c00efab81e539e3d9a663b0fd645d70e20250faa8af5.jpg)

(a) 2vs2 scenario with moderate traffic, illustrating stable onboard navigation and COLREGs-compliant maneuvers.  
![](images/ff2b511a59784ddee40bf68075d29c4c400fe73885c991dc3ebe37a5570746bc.jpg)  
(b) 2vs4 dense-traffic scenario illustrating robust onboard navigation under dynamic multi-vessel interactions.  
Fig. 12: Stage 2 onboard navigation trajectories under varying traffic densities, demonstrating stable COLREGs-compliant behavior and adaptive path adjustment.

These observations provide several practical insights: (1) staged training in realistic port simulations supports progressive learning of safe navigation behaviors prior to deployment; (2) quantitative COLREGs-related metrics can serve as useful indicators for evaluating navigation reliability; (3) performance degradation under high-density traffic highlights the importance of adaptive safety margins and conservative maneuvering; and (4) curriculum-guided training combined with behavioral evaluation can improve the robustness and reliability of autonomous maritime navigation in complex operational environments.

## V. CONCLUSION

This paper presented a curriculum-guided DRL framework for reliable onboard navigation of IoT-enabled maritime autonomous surface vehicles in complex port environments. The proposed approach is built on a shared recurrent PPO architecture under a centralized training and decentralized execution paradigm. By incorporating temporal memory through LSTM and staged curriculum progression, the learned policy achieves stable, regulation-aware navigation under varying traffic densities and environmental disturbances.

Behavioral analysis shows that the learned policy develops consistent navigation strategies, including early conflict anticipation, smooth overtaking, structured head-on avoidance, and stable separation in multi-vessel interactions. These behaviors persist in progressively complex port scenarios and under environmental disturbances, indicating that the policy captures transferable navigation patterns rather than scenario-specific behaviors. Experimental results demonstrate stable performance across stages, maintaining high navigation success, low collision frequency, and consistent regulation-compliant behavior under increasing complexity.

![](images/3f832ce2bb1b00dc42588e359f8e6acd303fd7ca3630c686fd5baeb9445b9502.jpg)

(a) 4vs4 scenario in the Rotterdam port, showing stable onboard navigation in narrow-channel conditions.  
![](images/464f21add75aee8f766a3ad3afa5083a3a5668fd6cddcbb4bd16bf8e3170a02a.jpg)

(b) 6vs6 scenario with increased multi-vessel interactions and environmental disturbances, illustrating adaptive onboard navigation behavior.  
![](images/bf475c70b9358b0b6cd004df0258051d56881558298ecb585f2e1d3a388f4e70.jpg)  
(c) 12vs12 high-density scenario demonstrating robust onboard navigation under complex and dynamic traffic conditions.

Fig. 13: Stage 3 onboard navigation trajectories in the Rotterdam port. Narrow channels, dense interactions, and environmental disturbances require early avoidance and conservative navigation behavior.  
![](images/8dce291b584e56262b9b5a06084cd239d1ebe5fbb5eafa80b83357288c6bae06.jpg)  
Fig. 14: Heatmap of performance metrics across CL stages.

The results highlight the effectiveness of shared recurrent policy learning combined with curriculum-guided training to improve the robustness, safety, and reliability of onboard navigation under partial observability. The use of quantitative regulation-related metrics provides interpretable indicators for evaluating navigation behavior, while staged simulation environments enable systematic assessment of policy robustness across varying operational conditions.

Future work will explore improving robustness through richer multi-modal sensing, including vision and radar fusion [49], expanding scenario diversity to improve generalization across broader operational conditions [50], [51], and investigating limited communication strategies for improved coordination in dense traffic [52], [53]. Incorporating prior knowledge or imitation learning [54], [55] can further improve training efficiency and safety in unfamiliar scenarios. In addition, hardware-in-the-loop and real-world validation will be investigated to assess real-time performance and deployment robustness, further bridging the gap between simulation and practical autonomous maritime systems.

## REFERENCES

[1] Maritime and Port Authority of Singapore, “Maritime digitalisation playbook,” 2025, accessed: 2025-01. [Online]. Available: https://www.mpa.gov.sg/maritime-singapore/innovation-and-r-d/ maritime-digitalisation-playbook

[2] Port of Los Angeles, “Port of los angeles strategic plan 2018–2022,” 2018, accessed: 2025-07-17. [Online]. Available: https://kentico.portoflosangeles.org/getmedia/ 6ac20c8e-f574-44a8-b28e-da8683b41cf6/Strategic-Plan-2018-2022

[3] Port of Rotterdam, “Artificial intelligence in the port,” 2025, accessed: 2025-01. [Online]. Available: https://www.portofrotterdam. com/en/port-future/innovation/artificial-intelligence-port

[4] Z. Chang, S. Liu, X. Xiong, Z. Cai, and G. Tu, “A survey of recent advances in edge-computing-powered artificial intelligence of things,” IEEE Internet of Things Journal, vol. 8, no. 18, pp. 13 849–13 875, 2021.

[5] X. Wei, J. Zhao, L. Zhou, and Y. Qian, “Broad reinforcement learning for supporting fast autonomous iot,” IEEE Internet of Things Journal, vol. 7, no. 8, pp. 7010–7020, 2020.

[6] G. Wen, C. P. Chen, S. S. Ge, H. Yang, and X. Liu, “Optimized adaptive nonlinear tracking control using actor–critic reinforcement learning strategy,” IEEE transactions on industrial informatics, vol. 15, no. 9, pp. 4969–4977, 2019.

[7] J. Duan, Y. Guan, S. E. Li, Y. Ren, Q. Sun, and B. Cheng, “Distributional soft actor-critic: Off-policy reinforcement learning for addressing value estimation errors,” IEEE transactions on neural networks and learning systems, vol. 33, no. 11, pp. 6584–6598, 2021.

[8] Y. Zeng, G. Liang, Q. Liu, E. Rodriguez, J. Pou, H. Jie, X. Liu, X. Zhang, J. Kotturu, and A. Gupta, “Multi-agent soft actor-critic aided active disturbance rejection control of dc solid-state transformer,” IEEE Transactions on Industrial Electronics, vol. 72, no. 1, pp. 492–503, 2024.

[9] Y. Zhao, F. Han, D. Han, X. Peng, W. Zhao, and G. Xia, “A port water navigation solution based on priority sampling sac: Taking yantai port environment as an example,” Robotics and Autonomous Systems, vol. 188, p. 104956, 2025.

[10] S. Hao, W. Guan, Z. Cui, and Z. Xi, “Intelligent navigation system for unmanned surface vessel based on rrt\* and sac,” in 2023 IEEE International Conference on Unmanned Systems (ICUS). IEEE, 2023, pp. 707–712.

[11] C. Bao, P. Wang, R. He, and G. Tang, “Autonomous trajectory planning method for hypersonic vehicles in glide phase based on ddpg algorithm,” Proceedings of the Institution of Mechanical Engineers, Part G: Journal of Aerospace Engineering, vol. 237, no. 8, pp. 1855–1867, 2023.

[12] Y. Du, X. Zhang, Z. Cao, S. Wang, J. Liang, F. Zhang, and J. Tang, “An optimized path planning method for coastal ships based on improved ddpg and dp,” Journal of Advanced Transportation, vol. 2021, no. 1, p. 7765130, 2021.

[13] J. Xue, M. He, J. Chen, B. Dong, and Y. Zheng, “Improved ddpg based on enhancing decision evaluation for path planning in high-density environments,” Expert Systems with Applications, vol. 279, p. 127378, 2025.

[14] X. Tang, Y. Yang, T. Liu, X. Lin, K. Yang, and S. Li, “Path planning and tracking control for parking via soft actor-critic under non-ideal scenarios,” IEEE/CAA Journal of Automatica Sinica, vol. 11, no. 1, pp. 181–195, 2023.

[15] Z. Ye, D. Zhang, Z.-G. Wu, and H. Yan, “A3c-based intelligent eventtriggering control of networked nonlinear unmanned marine vehicles subject to hybrid attacks,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 8, pp. 12 921–12 934, 2021.

[16] X. Liu, J. Yu, J. Wang, and Y. Gao, “Resource allocation with edge computing in iot networks via machine learning,” IEEE Internet of Things Journal, vol. 7, no. 4, pp. 3415–3426, 2020.

[17] Y. Niu, F. Zhu, M. Wei, Y. Du, and P. Zhai, “A multi-ship collision avoidance algorithm using data-driven multi-agent deep reinforcement learning,” Journal of marine science and engineering, vol. 11, no. 11, p. 2101, 2023.

[18] Z. Cui, W. Guan, X. Zhang, and C. Zhang, “Autonomous navigation decision-making method for a smart marine surface vessel based on an improved soft actor–critic algorithm,” Journal of Marine Science and Engineering, vol. 11, no. 8, p. 1554, 2023.

[19] S. Guo, X. Zhang, Y. Du, Y. Zheng, and Z. Cao, “Path planning of coastal ships based on optimized dqn reward function,” Journal of Marine Science and Engineering, vol. 9, no. 2, p. 210, 2021.

[20] W. Wang, X. Luo, Y. Li, and S. Xie, “Unmanned surface vessel obstacle avoidance with prior knowledge-based reward shaping,” Concurrency and Computation: Practice and Experience, vol. 33, no. 9, p. e6110, 2021.

[21] X. Lin, P. Szenher, Y. Huang, and B. Englot, “Distributional reinforcement learning based integrated decision making and control for autonomous surface vehicles,” IEEE Robotics and Automation Letters, 2024.

[22] Z. Cui, W. Guan, X. Zhang, and G. Zhang, “Autonomous collision avoidance decision-making method for usv based on atl-td3 algorithm,” Ocean Engineering, vol. 312, p. 119297, 2024.

[23] X. Sun, G. Li, Z. Liu, L. Zhang, H. Yu, D. Song, and Q. J. Wu, “Path planning algorithm for unmanned surface vessels based on colregs and meta-td3,” Ocean Engineering, vol. 334, p. 121580, 2025.

[24] F. Wu, Y. Chen, X. Chen, W. Fan, and Y. Liu, “An adaptive dual prediction scheme based on edge intelligence,” IEEE Internet of Things Journal, vol. 7, no. 10, pp. 9481–9493, 2020.

[25] K. Zheng, X. Zhang, C. Wang, M. Zhang, and H. Cui, “A partially observable multi-ship collision avoidance decision-making model based on deep reinforcement learning,” Ocean & Coastal Management, vol. 242, p. 106689, 2023.

[26] Z. Rongcai, X. Hongwei, and Y. Kexin, “Autonomous collision avoidance system in a multi-ship environment based on proximal policy optimization method,” Ocean Engineering, vol. 272, p. 113779, 2023.

[27] R. Zhang, X. Qin, M. Pan, S. Li, and H. Shen, “Adaptive temporal reinforcement learning for mapping complex maritime environmental state spaces in autonomous ship navigation,” Journal of Marine Science and Engineering, vol. 13, no. 3, p. 514, 2025.

[28] Z. Cui, W. Guan, and X. Zhang, “A collision avoidance decision-making method for multiple marine autonomous surface ships based on p3dl-ppo algorithm,” Journal of Marine Science and Technology, vol. 30, no. 1, pp. 205–223, 2025.

[29] R. Sawada, K. Sato, and T. Majima, “Automatic ship collision avoidance using deep reinforcement learning with lstm in continuous action spaces,” Journal of Marine Science and Technology, vol. 26, no. 2, pp. 509–524, 2021.

[30] S. Narvekar, B. Peng, M. Leonetti, J. Sinapov, M. E. Taylor, and P. Stone, “Curriculum learning for reinforcement learning domains: A framework and survey,” Journal of Machine Learning Research, vol. 21, no. 181, pp. 1–50, 2020.

[31] R. Portelas, L. Denoyer, A. Dumur, S. Lamprier, and A. Pere, “Teacher algorithms for curriculum learning of deep rl in continuous environments,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 13 002–13 015.

[32] T. Matiisen, A. Oliver, T. Cohen, and J. Schulman, “Teacher–student curriculum learning,” IEEE transactions on neural networks and learning systems, vol. 31, no. 9, pp. 3732–3740, 2019.

[33] Z. Ren, D. Dong, H. Li, and C. Chen, “Self-paced prioritized curriculum learning with coverage penalty in deep reinforcement learning,” IEEE transactions on neural networks and learning systems, vol. 29, no. 6, pp. 2216–2226, 2018.

[34] C. Florensa et al., “Reverse curriculum generation for reinforcement learning,” arXiv preprint arXiv:1707. backward, 2017.

[35] P. Huang, L. Zeng, X. Chen, K. Luo, Z. Zhou, and S. Yu, “Edge robotics: Edge-computing-accelerated multirobot simultaneous localization and mapping,” IEEE Internet of Things Journal, vol. 9, no. 15, pp. 14 087– 14 102, 2022.

[36] Ł. Marchel, R. Kot, P. Szymak, and P. Piskur, “Model-based auv path planning using curriculum learning and deep reinforcement learning on a simplified electronic navigation chart,” Applied Sciences, vol. 15, no. 11, p. 6081, 2025.

[37] S. T. Havenstrøm, A. Rasheed, and O. San, “Deep reinforcement learning controller for 3d path following and collision avoidance by autonomous underwater vehicles,” Frontiers in Robotics and AI, vol. 7, p. 566037, 2021.

[38] Y. Weng, J. Pajarinen, R. Akrour, T. Matsuda, J. Peters, and T. Maki, “Reinforcement learning based underwater wireless optical communication alignment for autonomous underwater vehicles,” IEEE Journal of Oceanic Engineering, vol. 47, no. 4, pp. 1231–1245, 2022.

[39] M. T. Spaan, “Partially observable markov decision processes,” in Reinforcement learning: State-of-the-art. Springer, 2012, pp. 387–414.

[40] H. Kurniawati, “Partially observable markov decision processes and robotics,” Annual Review ofControl, Robotics, and Autonomous Systems, vol. 5, no. 1, pp. 253–277, 2022.

[41] Y. Ding and H. Zhu, “Risk-sensitive markov decision processes of usv trajectory planning with time-limited budget,” Sensors, vol. 23, no. 18, p. 7846, 2023.

[42] R. Szlapczynski and J. Szlapczynska, “Review of ship safety domains: Models and applications,” Ocean Engineering, vol. 145, pp. 277–289, 2017.

[43] N. Wang, “An intelligent spatial collision risk based on the quaternion ship domain,” The Journal of Navigation, vol. 63, no. 4, pp. 733–749, 2010.

[44] T. I. Fossen, Handbook of Marine Craft Hydrodynamics and Motion Control. John Wiley & Sons, 2011.

[45] W. Naeem, G. W. Irwin, and A. Yang, “Colregs-based collision avoidance strategies for unmanned surface vehicles,” Mechatronics, vol. 22, no. 6, pp. 669–678, 2012.

[46] Y. Kuwata, M. T. Wolf, D. Zarzhitsky, and T. L. Huntsberger, “Safe maritime autonomous navigation with colregs, using velocity obstacles,” IEEE Journal ofOceanic Engineering, vol. 39, no. 1, pp. 110–119, 2013.

[47] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, “Continuous control with deep reinforcement learning,” arXiv preprint arXiv:1509.02971, 2015.

[48] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor,” International Conference on Machine Learning (ICML), 2018.

[49] B. R. Kiran, I. Sobh, V. Talpaert, P. Mannion, A. A. Al Sallab, S. Yogamani, and P. Perez, “Deep reinforcement learning for autonomous´ driving: A survey,” IEEE transactions on intelligent transportation systems, vol. 23, no. 6, pp. 4909–4926, 2021.

[50] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, “Domain randomization for transferring deep neural networks from simulation to the real world,” in 2017 IEEE/RSJ international conference on intelligent robots and systems (IROS). IEEE, 2017, pp. 23–30.

[51] X. B. Peng, M. Andrychowicz, W. Zaremba, and P. Abbeel, “Sim-to-real transfer of robotic control with dynamics randomization,” in 2018 IEEE international conference on robotics and automation (ICRA). IEEE, 2018, pp. 3803–3810.

[52] J. Foerster, I. A. Assael, N. De Freitas, and S. Whiteson, “Learning to communicate with deep multi-agent reinforcement learning,” Advances in neural information processing systems, vol. 29, 2016.

[53] C. Zhu, M. Dastani, and S. Wang, “A survey of multi-agent deep reinforcement learning with communication,” Autonomous Agents and Multi-Agent Systems, vol. 38, no. 1, p. 4, 2024.

[54] A. Hussein, M. M. Gaber, E. Elyan, and C. Jayne, “Imitation learning: A survey of learning methods,” ACM Computing Surveys (CSUR), vol. 50, no. 2, pp. 1–35, 2017.

[55] M. Zhou, J. Luo, J. Villella, Y. Yang, D. Rusu, J. Miao, W. Zhang, M. Alban, I. Fadakar, Z. Chen et al., “Smarts: Scalable multi-agent reinforcement learning training school for autonomous driving,” arXiv preprint arXiv:2010.09776, 2020.