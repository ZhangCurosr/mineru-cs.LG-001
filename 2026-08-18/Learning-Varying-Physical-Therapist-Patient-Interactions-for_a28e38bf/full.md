# Learning Varying Physical Therapist-Patient Interactions for Robot-mediated Upper Limb Task-Specific Training

Jia Quan Loh, Vincent Crocher, Marlena Klaic, Denny Oetomo and Ying Tan

Abstract—Upper extremity motor function recovery is positively linked to Task-Specific Training (TST) and sufficient therapy dosage. Rehabilitation robots can increase TST dosage via controlled, repetitive treatment and free therapists to simultaneously manage other patients, but it has yet to demonstrate significant benefits over conventional treatment. This is potentially linked to inaccurate robotic representation of personalised physical therapist-patient interaction and lack of practice variability during TST. Hence, we advocate for robotic interventions that preserve the personalised physical therapist-patient interactions when delivering TST for patients across varying practise conditions. We propose a Learning-from-Demonstration framework using Task-Parameterised Gaussian Mixture Models (TPGMM) to learn personalised physical therapist-patient interaction in Task-Specific exercises, mapping patient joint kinematics to therapist-applied torques using few demonstrations. The model is generalised to reconstruct therapist torques in new task variations. The framework was evaluated on physical interactions from 14 mock “therapist-patient” pairs over three tasks of increasing complexity, each with six variations. A benchmark comparison against a Look-Up Table was conducted. The results show both methods reproducing interactions in unseen task variations that deviate slightly from the actual interaction, with TPGMM slightly outperforming LUT. Both methods reproduced interactions that gets increasingly closer to the actual interaction as task complexity increases.

Index Terms—Task-Specific Training, Neuro-rehabilitation, Physical therapist-patient interaction, Robot-mediated rehabilitation, Learning-from-Demonstration

## I. INTRODUCTION

The improvement of upper extremity motor function is positively correlated with the inclusion of functionally orientated therapy or Task-Specific Training (TST) [1], [2] and sufficient therapy dosage [3], [4]. Incorporating robotic interventions into TST enables a robot to deliver controlled, repetitive treatment to a patient, increasing the training dosage beyond what conventional one-to-one therapy allows, while freeing the therapist to manage multiple patients simultaneously [5].

However, robotic delivery of TST faces challenges: robots must understand the task and environment a patient is practising in, and adapt their role as a “therapist” when interacting with the patient [6]. The challenge is compounded when robots are expected to deliver TST in highly dynamic environments analogous to real world settings [7], where numerous variables (e.g. different task positions, movement speeds, and objects) can be introduced in TST.

These challenges prevent current robot-patient interaction models from accurately modelling what therapists really do during TST [6]—specifically, the desired movements, forces, and communication of intent therapist use when practising a task with a patient. This may explain why reviews over 24 clinical studies [8], [9] found robotic TST no substantially better than conventional TST, despite robots’ ability to deliver larger training doses.

Of these challenges, we focus on the robotic delivery of the physical forces exchanged between therapist and patient in TST, forming the basis for robots to physically interact with patients. Rehabilitation robots are often programmed with predefined physical Human–Robot Interaction (pHRI) schemes (e.g. support, resist, guide, assist-as-needed [10], [11]), designed to replicate evidence-based therapist interventions that promote motor recovery after stroke. However, rehabilitation literature has increasingly advocated for personalised interventions catered towards the specific neurological characteristics and impairments of a patient [12]. Consequently, as patients demonstrate varying degrees of impairment, targeted interventions may fall outside the capabilities of the robot’s in-built pHRI schemes. This creates a need for therapists to easily design new pHRI customised to each patients in-situ if patient needs are beyond what is pre-programmed in a robot [13].

To address this need, an experimental rehabilitation approach proposed around 2016 [14] but recently coined Robotmediated physical Human-Human Interaction (RHHI) [15] was proposed. The therapist works through a robot to influence patient movement, and the robot directly captures the personalised interaction for future reproduction, removing the need to select predefined controllers to imitate the interaction. Pilot studies showed feasible reproduction of physical therapistpatient interaction [13], and recently, improved patient recovery in lower limb joints over conventional therapy [16].

While current RHHI approaches allows for directly capturing and imitating personalised therapist actions, its efficacy for robot-assisted TST when variations are incorporated remain unknown. Clinical studies reviewed in [9] report a lack of practice variability (only four out of 17 studies report variability). Cognitive science theory and motor learning studies investigating the Variability of Practice hypothesis [26]–[28] emphasise the importance of including varying task contexts (e.g. different reaching and grasping points, varying movement speeds) to improve motor skill learning. Lack of practice variability leads to movement-specific learning that does not induce the neuro-plastic changes necessary for skill transfer to similar tasks or varied settings [29]. Capturing the required physical interaction does not enable robots to understand its application across varying conditions; therefore, constant therapist presence is potentially required to redemonstrate the interaction.

TABLE I: Learning Therapist Interaction Profiles from Demonstration— Summary of literature.
<table><tr><td>Ref.</td><td>RHHI Configuration</td><td>Input</td><td>Learning Method</td><td>Desired Output</td><td>Variation</td></tr><tr><td>[14]</td><td>T-R-P</td><td>Measured end-effector position</td><td>GMM</td><td>End-effector force</td><td>x</td></tr><tr><td>[17]</td><td>T-R-R-P</td><td>Measured end-effector force</td><td>GMM-based non-linear system</td><td>End-effector force derivative</td><td>x</td></tr><tr><td>[18]</td><td>T-R-P</td><td>Measured end-effector position</td><td>GMM-based Impedance model</td><td>End-effector force</td><td>x</td></tr><tr><td>[19]</td><td>T-R-R-P</td><td>Measured patient end-effector kinematics</td><td>GMM</td><td>Patient end-effector position</td><td>x</td></tr><tr><td>[20]</td><td>T-R-P</td><td>Measured end-effector kinematics</td><td>Potential Field Function</td><td>End-effector force</td><td>x</td></tr><tr><td>[13]</td><td>T-R-P</td><td>Measured joint positions</td><td>Look-Up Table</td><td>Joint torques</td><td>x</td></tr><tr><td>[21]</td><td>T-P</td><td>Measured joint kinematics</td><td>GMM-based Impedance model</td><td>Joint torques</td><td>x</td></tr><tr><td>[22]</td><td>T-R-P</td><td>Measured joint kinematics</td><td>Look-Up Table</td><td>Joint torques</td><td>x</td></tr><tr><td>[23]</td><td>T-R-P</td><td>Time</td><td>DMP</td><td>End-effector positions</td><td>√</td></tr><tr><td>[24]</td><td>T-R-P</td><td>Time</td><td>DMP</td><td>Joint positions</td><td> $\checkmark$ </td></tr><tr><td>[25]</td><td>T-P</td><td>Time</td><td>Task-Parameterised GMM</td><td>Joint positions</td><td> $\checkmark$ </td></tr></table>

T: Therapist, R: Robot, P: Patient, GMM: Gaussian Mixture Models, DMP: Dynamic Movement Primitives

Motivated by this limitation, we believe that to achieve practical RHHI for variable robot-assisted TST, the latent features of the physical interactions applied by the therapist along a patient’s movement could be learned, from a few demonstrations, and the learned interaction generalised to variations of the task. In such scenario, a clinician could demonstrate a task for a given patient-task and further let the patient use the robot for structured mass repetition [6] over varying task conditions.

Learning-from-Demonstration (LfD) provides a framework for modelling pHRI in varying environments using data from few demonstrations [30], which could be suitable for modelling such physical therapist-patient interaction. Literature on learning physical therapist-patient interactions using LfD, is summarised in Table I. From the table, the extent of methods that focus on learning physical forces in RHHI [13], [14], [17]–[22] when the task conditions are varied remains unexplored, with approaches only “reproducing” previously seen condition. Generalising demonstrations in upper limb rehabilitation to unseen variations were approached [23]–[25], though restricted to learning only the movement of the task.

Lauretti et al., for example, used Dynamic Movement Primitives (DMP) to learn task and joint space movement of healthy and impaired subjects in robot-assisted TST [23], [24]. They achieved generalisation to different conditions with a 100% task completion rate for different task positions and posture. However, the similarity of patient’s movement features along with the representation of the intended therapist-patient forces by the robot were not quantified and evaluated.

Recently, we proposed to use DMP and Task-Parameterised GMMs (TPGMM) to learn joint movements across different task variations, and compared the ability of both models to preserve the movement features demonstrated by healthy individuals [25]. It shows that TPGMM preserves the intended motion features better than DMPs in unseen task variations. However, this study lacks a robotic implementation and evaluation of the intended therapist-patient forces.

Building on this literature, in this work, we propose to learn the physical therapist-patient interaction and to generalise it to unseen task variations. We first collect physical therapist-patient interactions comprised of patient joint kinematics and therapist-applied joint torques, across variations of task-specific exercises. We propose a LfD framework, to learn and generalise these interactions across task variations, as a therapist–patient–task-specific interaction model. Finally, we evaluate this method against a Look Up Table (LUT) approach proposed by Luciani et al. [22] in an experiment with physiotherapy students.

## II. FRAMEWORK FOR LEARNING PHYSICAL PATIENT–THERAPIST INTERACTION

This section presents the proposed framework for learning physical therapist–patient interactions during task-specific rehabilitation. The therapist–patient interaction is first mathematically formulated using patient joint kinematics and therapistapplied joint torques. A Task-Parameterized Gaussian Mixture Model (TPGMM)-based learning framework is then developed to capture and generalize therapist interactions across varying task conditions. Finally, a conventional look-up table (LUT) method is described as the benchmark for evaluating the proposed approach.

We define the physical interaction between a therapist and a patient by the interaction vector $\gamma _ { t } ,$ given by

$$
\gamma _ { t } = [ \pmb q _ { t } , \pmb { \dot { q } } _ { t } , \pmb { \tau } _ { t } ] \ \in \mathbb { R } ^ { 3 \cdot D } , \ t \in [ 0 , T _ { n } ] ,\tag{1}
$$

where $\mathbf { \psi } _ { q _ { t } } , \mathbf { \psi } _ { \dot { q } _ { t } } ,$ , and $\boldsymbol { \tau } _ { t } ~ \in ~ \mathbb { R } ^ { D }$ denote the patient joint positions, joint velocities, and therapist-applied joint torques, respectively, for a D-DOF musculoskeletal model at time t.

A therapist–patient pair, indexed by $a \in [ 1 , \ldots , A ]$ , performs task $e \in [ 1 , \ldots , E ]$ under each task variation $v \in [ 1 , \ldots , V ]$ Each variation is repeated R times, indexed by $r \in [ 1 , \ldots , R ]$ Each trial is uniquely identified by the index $\boldsymbol { n } = \{ a , e , v , r \}$ and has a duration of $T _ { n }$ seconds.

The interaction vector $\gamma _ { t }$ is sampled at a fixed sampling period $\Delta t ,$ , producing the discrete interaction vector $\gamma _ { n , j } ,$ where $j$ denotes the sample index within trial n. Consequently, trial n consists of $J _ { n } = T _ { n } / \Delta t$ interaction samples collected over its duration $T _ { n }$ . The sequence of interaction vectors from trial n forms the interaction profile $\Gamma _ { n } .$ given by

$$
\Gamma _ { n } = [ \pmb q _ { n } , \pmb { \dot { q } } _ { n } , \pmb { \tau } _ { n } ] \in \mathbb { R } ^ { J _ { n } \times 3 \cdot D } ,\tag{2}
$$

where $\mathbf { } q _ { n } , { \dot { q } } _ { n } .$ , and $\tau _ { n } \in \mathbb { R } ^ { J _ { n } \times D }$ denote the patient joint position, joint velocity, and therapist-applied joint torque profiles, respectively, over the $J _ { n }$ samples of trial $n .$

For a given therapist–patient pair a performing task e, demonstrations are collected over all task variations and repetitions, resulting in $N = V \cdot R$ trials. The corresponding interaction profiles are stacked vertically to form the task demonstration set Γ, containing a total of $J ~ = ~ \sum _ { n = 1 } ^ { N } J _ { n }$ samples, given by

$$
\mathbf { \Gamma } = \left[ \begin{array} { l } { \Gamma _ { 1 } } \\ { \Gamma _ { 2 } } \\ { \vdots } \\ { \Gamma _ { N } } \end{array} \right] = [ \pmb { q } , \dot { \pmb { q } } , \tau ] \in \mathbb { R } ^ { J \times 3 \cdot D } ,\tag{3}
$$

where $q , { \dot { q } } ,$ and $\boldsymbol { \tau } \in \mathbb { R } ^ { J \times D }$ denote the patient joint position, joint velocity, and therapist-applied joint torque profiles from all N trials stacked vertically.

Following previous studies [13], [21], [22], we seek to learn, for each therapist–patient pair a and task e, a therapist–patient– task-specific interaction model $g _ { a , e }$ that maps the patient’s joint kinematics $( q , \dot { q } )$ to the therapist-applied joint torques τ . A Learning-from-Demonstration (LfD) approach is employed to learn an approximation $\hat { g } _ { a , e }$ of the unknown interaction model $g _ { a , e } .$ . The estimated therapist-applied joint torques are approximated by

$$
\hat { \pmb { \tau } } = \hat { g } _ { a , e } ( { \pmb q } , \pmb { \dot { q } } ) .\tag{4}
$$

## A. Task-Parameterised Gaussian Mixture Models

In this work, we employ a Task-Parameterised Gaussian Mixture Model (TPGMM) to approximate the interaction model $\hat { g } _ { a , e } .$ TPGMM was selected because of its ability to generalise demonstrations across varying task conditions while preserving the key characteristics of the demonstrated interactions. In our previous work, TPGMM was shown to preserve the intended movement features more effectively than Dynamic Movement Primitives (DMPs) when reproducing unseen task variations, particularly as task complexity increases [25]. TPGMM extends the standard Gaussian Mixture Model (GMM) by introducing a fixed number of task-dependent reference frames $( P )$ for each demonstration. Each demonstration is first transformed with respect to these reference frames before a GMM is learned in each frame.

1) Frame Transformation: Each demonstration is assumed to consist of movements between a fixed number of $P$ task points (e.g., start, via, and end points). A reference frame is assigned to each task point, resulting in $P$ reference frames for every interaction profile $\Gamma _ { n }$

Following Calinon’s formulation [31], the $p ^ { t h }$ reference frame of $\Gamma _ { n }$ is defined by a rotation matrix $\bar { A } _ { n } ^ { ( p ) } \in \mathbb { R } ^ { D \times D }$ and a displacement vector $\mathbf { \bar { b } } _ { n } ^ { ( p ) } \in \mathbb { R } ^ { D }$ with respect to a known inertial frame. The interaction profile $\Gamma _ { n }$ is subsequently transformed into each reference frame.

Specifically, the $j ^ { t h }$ interaction vector $\gamma _ { n , j }$ is transformed into the $p ^ { t h }$ reference frame to obtain $\gamma _ { n , j } ^ { ( p ) }$ , given by

$$
\gamma _ { n , j } ^ { ( p ) } = ( A _ { n } ^ { ( p ) } ) ^ { - 1 } ( \gamma _ { n , j } - b _ { n } ^ { ( p ) } ) , \forall j \in [ 0 , 1 , \ldots , J _ { n } ] .\tag{5}
$$

Applying this transformation to every interaction vector in trial n yields the transformed interaction profile $\Gamma _ { n } ^ { \left( p \right) }$ . The transformed interaction profiles from all N trials are then stacked to form the demonstration set in the $p ^ { t h }$ reference frame,

$$
\begin{array} { r } { \mathbf { T } ^ { \left( p \right) } = \left[ \begin{array} { l } { \Gamma _ { 1 } ^ { \left( p \right) } } \\ { \Gamma _ { 2 } ^ { \left( p \right) } } \\ { \vdots } \\ { \Gamma _ { N } ^ { \left( p \right) } } \end{array} \right] , \forall p \in [ 1 , 2 , \dotsc , P ] . } \end{array}\tag{6}
$$

Note: Unlike the conventional implementation of TPGMM in a three-dimensional Euclidean space, we apply TPGMM in the patient’s D-DOF joint space [25]. Consequently, the reference frame parameters are given by

$$
A _ { n } ^ { ( p ) } = I _ { D \times D } \ a n d \ b _ { n } ^ { ( p ) } = [ \pmb { q } _ { n , j _ { p } } , \pmb { \dot { q } } _ { n , j _ { p } } , 0 _ { D } ] ,\tag{7}
$$

where $\mathbf { \boldsymbol { q } } _ { n , j _ { p } }$ and $\dot { \pmb q } _ { n , j _ { p } }$ denote the joint positions and joint velocities, respectively, at sample $j _ { p }$ of trial $n ,$ corresponding to the instant when the patient reaches the $p ^ { t h }$ task point. Since the interaction is represented in the patient’s joint space, no rotational or scaling transformation is tested, i.e., $\bar { A } _ { n } ^ { ( p ) } = I _ { D \times D }$ . Furthermore, the joint torque component is not translated between reference frames (represented by $0 _ { D }$ in ${ \pmb b } _ { n } ^ { ( p ) } )$ , ensuring that demonstrations with identical joint kinematic configurations in different reference frames share the same joint torque profile.

2) Modelling: A TPGMM models the demonstrations using a mixture of K Gaussian kernels, denoted by $\{ \pi _ { k } , \bar { \mathcal { N } } ( \pmb { \mu } _ { k } , \Sigma _ { k } ) \} _ { k = 1 } ^ { K }$ . The $k ^ { t h }$ Gaussian kernel is characterised by a mixture weight $\pi _ { k }$ and a Gaussian distribution $\mathcal { N } ( \mu _ { k } , \dot { \Sigma _ { k } } )$ , where $\pmb { \mu _ { k } } \in \mathbb { R } ^ { D }$ and $\Sigma _ { k } \in \mathbb { R } ^ { D \times D }$ denote the mean vector and covariance matrix, respectively.

To account for task variations, each Gaussian distribution is parameterised by the $P$ reference frames through a set of sub-Gaussian distributions, $\{ \mathcal { N } ( \boldsymbol { \mu } _ { k } ^ { ( p ) } , \boldsymbol { \Sigma } _ { k } ^ { ( p ) } ) \} _ { p = 1 } ^ { P }$ , where $\mu _ { k } ^ { \left( p \right) }$ and $\Sigma _ { k } ^ { \left( p \right) }$ denote the mean vector and covariance matrix of the $k ^ { t h }$ Gaussian kernel expressed in the $p ^ { t h }$ reference frame. Consequently, the complete TPGMM is represented by

$$
\{ \pi _ { k } , \{ \mathcal { N } ( { \pmb \mu } _ { k } ^ { ( p ) } , { \Sigma } _ { k } ^ { ( p ) } ) \} _ { p = 1 } ^ { P } \} _ { k = 1 } ^ { K } .
$$

The optimal number of Gaussian kernels, K, is determined using the Bayesian Information Criterion (BIC) [32]. For each reference frame p, a Gaussian Mixture Model (GMM), $\{ \pi _ { k } , \mathcal { N } ( \pmb { \mu } _ { k } ^ { ( p ) } , \Sigma _ { k } ^ { ( p ) } ) \} _ { k = 1 } ^ { K }$ , is learned from the transformed demonstration set $\mathbf { \Gamma } \Gamma ^ { \left( p \right) }$ using the Expectation–Maximisation (EM) algorithm. The EM algorithm estimates the mixture parameters by maximising the likelihood of the observed data $\boldsymbol { \Gamma } ^ { \left( p \right) }$ under the corresponding GMM [31], for all $p \in$ $[ 1 , 2 , \ldots , P ]$

3) Regression: To predict therapist-applied joint torques for a new task variation, a new set of reference frames is first defined by $\begin{array} { r } { \pmb { b } _ { n e w } ^ { ( p ) } , } \end{array}$ , corresponding to the required joint kinematics at the new task points. The learned GMM in each reference frame, $\{ \pi _ { k } , \mathcal { N } ( \pmb { \mu } _ { k } ^ { ( p ) } , \Sigma _ { k } ^ { ( p ) } ) \} _ { k = 1 } ^ { K } ,$ , is then conditioned on the new reference frames to obtain

$$
\{ \pi _ { k } , \mathcal { N } ( \hat { \pmb { \mu } } _ { k } ^ { ( p ) } , \hat { \Sigma } _ { k } ^ { ( p ) } ) \} _ { k = 1 } ^ { K } ,
$$

given by

$$
\hat { \pmb { \mu } } _ { k } ^ { ( p ) } = \pmb { \mu } _ { k } ^ { ( p ) } + \pmb { b } _ { n e w } ^ { ( p ) } , \hat { \Sigma } _ { k } ^ { ( p ) } = \Sigma _ { k } ^ { ( p ) } .\tag{8}
$$

Note: Equation (8) is a simplified form of the conditioning proposed by Calinon [31], since the transformation matrix in the joint space is the identity matrix, i.e., $A _ { n e w } ^ { ( p ) } = I .$

The conditioned Gaussian distributions from all P reference frames are subsequently combined using the product of Gaussians to obtain the reduced GMM,

$$
\Theta = \{ \pi _ { k } , \mathcal { N } ( \hat { \pmb { \mu } } _ { k } , \hat { \Sigma } _ { k } ) \} _ { k = 1 } ^ { K } ,
$$

where

$$
\hat { \pmb { \mu } } _ { k } = \hat { \Sigma } _ { k } \sum _ { p = 1 } ^ { P } \hat { \Sigma } _ { k } ^ { ( p ) ^ { - 1 } } \hat { \pmb { \mu } } _ { k } ^ { ( p ) } , ~ \hat { \Sigma } _ { k } = \left( \sum _ { p = 1 } ^ { P } \hat { \Sigma } _ { k } ^ { ( p ) ^ { - 1 } } \right) ^ { - 1 } .\tag{9}
$$

Finally, Gaussian Mixture Regression (GMR) is performed on the reduced GMM Θ to approximate the learned interaction model $\hat { g } _ { a , e } .$ . Given a new patient joint kinematic input $( { \pmb q } _ { j _ { \mathrm { t e s t } } } , \dot { \pmb q } _ { j _ { \mathrm { t e s t } } } )$ , the corresponding therapist-applied joint torques are estimated as

$$
\hat { \tau } _ { j _ { \mathrm { t e s t } } } = \hat { g } _ { a , e } ( \pmb { q } _ { j _ { \mathrm { t e s t } } } , \pmb { \dot { q } } _ { j _ { \mathrm { t e s t } } } ) .\tag{10}
$$

## B. Look-Up Table

The Look-Up Table (LUT) method proposed by Luciani et al. [22] is implemented as the benchmark for evaluating the proposed TPGMM framework. Unlike TPGMM, which learns a statistical model to generalise therapist–patient interactions across varying task conditions, the LUT directly stores demonstrated interactions and predicts therapist-applied joint torques by retrieving the demonstrated interaction with the most similar joint kinematics.

Given the task demonstration set $\Gamma = [ q , \dot { q } , \tau ]$ , comprising all patient joint position, joint velocity, and therapist-applied joint torque samples from therapist–patient pair a performing task $e ,$ a LUT is constructed.

Given a new patient joint kinematic input $( { \pmb q } _ { j _ { \mathrm { t e s t } } } , \dot { \pmb q } _ { j _ { \mathrm { t e s t } } } )$ a closest-vector search is performed over Γ to identify the demonstration sample with the smallest Euclidean distance to the input. The therapist-applied joint torque associated with the nearest neighbour is then selected as the estimated therapistapplied joint torque, $\hat { \tau } _ { j _ { \mathrm { t e s t } } }$

To reduce rapid variations in the estimated torque caused by switching between neighbouring LUT entries, $\hat { \tau } _ { j _ { \mathrm { t e s t } } }$ is subsequently filtered using a low-pass filter with a cut-off frequency of 3 Hz [22].

## III. EXPERIMENTAL METHODS

This section describes the experimental methodology used to evaluate the proposed therapist–patient interaction learning framework. The experimental protocol, participant recruitment, data acquisition system, and data processing pipeline are first presented. The training, validation, and generalisation procedures are then described, followed by the outcome measures and statistical analyses used to compare the proposed TPGMM framework with the benchmark LUT approach.

## A. Experimental Protocol

1) Experimental Procedure: Pairs of participants were recruited for the study to role play as both mock patient and therapist. In each half-session, one participant is instructed to act out three “patient” personas each performing one Taskspecific exercise. The other participant interacts with the “patient” based on their clinical expertise as the “therapist”. At the end of each half-session, the two participants exchange roles.

Each task consists of a series of sub-tasks performed under specific variations. The persona diagnosis and sub-tasks are defined in Table II. Each task has two parameters for variations, one with two levels, one with three levels, as described in Table II. Each variation of each task is repeated for four trials. An illustration of the tasks is given in Fig 1.

![](images/d0930cac6749d9787f43393e3ecb5e1a48214a91e6e9e6f801b5d3b5c744aaa9.jpg)  
Fig. 1: Task-specific exercises performed in the study. TP refers to task points where the joint kinematics are extracted from for the task frames. 1., 2. and 3. refers to each patient persona.

A therapist–patient pair $a \in \{ 1 , \ldots , A \}$ , demonstrates three task $e \in \{ 1 , 2 , 3 \}$ for six variations $v \in \{ 1 , 2 , \ldots , 6 \}$ with four repetitions, $r \in \{ 1 , 2 , \ldots , 4 \}$ each.

TABLE II: Patient personas (P#) with associated task descriptions.
<table><tr><td rowspan=20 colspan=1>PlP2P3</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=5>Patient History                                Task Description                                  Sub-tasks</td></tr><tr><td rowspan=1 colspan=1>FMA-UE</td><td rowspan=1 colspan=1>8/66</td><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Reaching towards a cup and return</td><td rowspan=2 colspan=1>1.2Move wrist from home (TP0) to cup (TP1)Move wrist from cup (TP1) to home (TP2)</td></tr><tr><td rowspan=1 colspan=1>MMSE</td><td rowspan=1 colspan=1>28/30</td><td rowspan=1 colspan=1>Variations</td><td rowspan=1 colspan=1>Cup Distance    Close</td></tr><tr><td rowspan=1 colspan=1>Diagnosis</td><td rowspan=2 colspan=1>Dense right hemiparesisLeft middle cerebral artery stroke(6 weeks)</td><td rowspan=3 colspan=1></td><td rowspan=4 colspan=2>FarShoulder Rotation External rotationNo rotationInternal rotation</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Presentation</td><td rowspan=1 colspan=1>Severe weakness in right upperextremityCannot initiate active grasp andvoluntary movement againstgravity</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FMA-UE</td><td rowspan=1 colspan=1>32/66</td><td rowspan=1 colspan=1>Task</td><td rowspan=4 colspan=2>Using utensils to transport food    21 22 23Move wrist from home (TP0) to holdspoon (TP1)Spoon Orientation UprightMove spoon (TP1) to rice bowl (TP2) andscoopSidewaysMove spoon (TP2) to plate (TP3) andfinish pouring2-4Move spoon (TP3) to home (TP4)</td></tr><tr><td rowspan=1 colspan=1>MMSE</td><td rowspan=1 colspan=1>30/30</td><td rowspan=3 colspan=1>Variations</td></tr><tr><td rowspan=2 colspan=1>Diagnosis</td><td rowspan=2 colspan=1>Post-ischaemic stroke (3 months)Moderate motor return</td></tr><tr><td rowspan=1 colspan=1>Plate DistanceClose</td></tr><tr><td rowspan=2 colspan=1>Presentation</td><td rowspan=1 colspan=1>Right-sided weaknessCan initiate shoulder and elbowmovement</td><td rowspan=3 colspan=3>MiddleFar</td></tr><tr><td rowspan=1 colspan=1>Limited wrist extension and finemotor control</td></tr><tr><td></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FMA-UE</td><td rowspan=1 colspan=1>56/66</td><td rowspan=1 colspan=1>Task</td><td rowspan=4 colspan=2>Pour water from a jug              2-1Move wrist from home (TP0) to hold jug(TP1)Jug Volume     Half Full       2 23 24Move jug (TP1) to rice bowl (TP2) andfinish pouring set amountFullMove jug (TP2) and release at original jugposition (TP3)Bowl DistanceCloseMove wrist (TP3) to home (TP4)</td></tr><tr><td rowspan=1 colspan=1>MMSE</td><td rowspan=1 colspan=1>30/30</td><td rowspan=3 colspan=1>Variations</td></tr><tr><td rowspan=1 colspan=1>Diagnosis</td><td rowspan=1 colspan=1>Mild right hemiparesispost-lacunar infarct</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Shoulder hitching and synergisticmovements</td></tr><tr><td rowspan=1 colspan=1>Presentation</td><td rowspan=1 colspan=1>Mild coordination deficits andslight delay in fine controlElevated shoulder compensationnoted</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>MiddleFar</td></tr></table>

2) Participant Recruitment: Participants who are third-year physiotherapy students that had underwent clinical placements in neuro-rehabilitation or sports therapy, were recruited from the Department of Physiotherapy, University of Melbourne. All procedures were approved by the Melbourne Health Human Research Ethics Committee (#31992), and written informed consent was obtained prior the investigations.

## B. Data Collection

In order to obtain force interaction measures during therapist-patient interaction, a wearable sensor system to record the wrenches exerted by the therapist on the patient’s upper body was first designed. The overall system hardware design is illustrated in Fig. 2.

The upper-body joint positions of the patient, $\mathbf { q } _ { n , j } \in$ $\mathbb { R } ^ { D = 1 0 }$ , is measured using the XSENS Awinda motion capture system (XSENS, Netherlands). The upper body (pelvis to wrist) is here modelled as a four-link, 10 DOFs, serial manipulator after XSENS’s MVN Biomechanical Model [33].

The interaction wrenches between the patient upper-body and the therapist hands are measured by three 6-DOF RFT80- 6A01 force-torque sensors (Robotous, Seongnam, South Korea) attached to custom rigid cuffs and placed on the shoulder, upper-arm and forearm (see Fig. 2). For a given sample j, of trial n, each measured wrench is denoted $\boldsymbol { w } _ { i , n , j } \in \mathbb { R } ^ { 6 }$ and $\{ w _ { i , n , j } \} _ { i = 1 , 2 , 3 }$ represents the wrenches sample of all three force sensors on the upper-body.

$\mathbf { q } _ { n , j }$ and $\{ w _ { i , n , j } \} _ { i = 1 , 2 , 3 }$ are streamed into a Python application to synchronise both data on a common time frame and then resampled at $1 0 0 H z , \Delta t = 0 . 0 1 s$ . Each trial n provides joint positions and wrenches profile $\pmb q _ { n }$ and $\{ { \pmb w } _ { i , n } \} _ { i = 1 , 2 , 3 } .$

## C. Data Processing

1) Approximating Patient Joint Kinematics and Therapistcontributed Joint Torque : $\mathbf { q } _ { n }$ and $\{ { w _ { i , n } } \} _ { i = 1 , 2 , 3 }$ are filtered with a fourth-order low-pass filter cut-off at $1 0 H z$

Two forms of biases were identified and subtracted from $\{ w _ { i , j } \} _ { i = 1 , 2 , 3 }$ . Wrench bias $\{ { \pmb w } _ { i , \mathrm { b i a s } } \} _ { i = 1 , 2 , 3 }$ is assumed as a result of the gradual deformation of the wearer’s soft tissue. $\{ { \pmb w } _ { i , \mathrm { b i a s } } \} _ { i = 1 , 2 , 3 }$ is collected with the wearer held stationary and averaged over five seconds before the start of each variation. Gravity bias $\{ { w _ { i , \mathrm { g r a v i t y } } } \} _ { i = 1 , 2 , 3 }$ resulting from the mass of the 3D-printed cuffs are computed using the orientation of the IMU trackers and the weight of the cuffs.

From Eq. 2, an interaction profile was defined as $\Gamma _ { n } ~ =$ $[ q _ { n } , \dot { q } _ { n } , \tau _ { n } ] . \mathrm { ~ } \dot { q } _ { n }$ is approximated numerically using the forward difference of $\mathbf { } q _ { n } .$ . The resulting derivative is filtered further using a fourth-order low-pass filter cut-off at 1.5Hz to remove the residual high frequency component amplified when taking the derivative.

![](images/a389f9697401099888037ce3986a7bb21cd99198e8797165aac79ad5f0a49df8.jpg)  
Fig. 2: 1. Construction of TPU braces - PETG cuffs sensorised to measure force-torque interactions and placement of sensors on the wearer’s body. Sensors holders house the IMU trackers on the right upper and forearm, with the force-torque sensors fixed on top. 2. Tight attachments on the wearer’s body using buckle-velcro straps and kinesiology tapes. Foam padding is layered underneath the brace to increase surface contact with the skin. 3. Common interactions “therapist” apply on the “patient” when constrained to only interact with the sensorised cuffs (highlighted in red), identified during the experiment.

$\tau _ { i , n }$ denoting the resultant joint torque profile from each $i ^ { t h }$ interaction point, is computed using the $\bar { \mathbf { \chi } } _ { i ^ { t h } }$ force sensor’s Jacobian $H _ { i , n } ,$ , and summed to obtain $\mathbf { \Delta } \tau _ { n } ,$ given as

$$
{ \pmb \tau } _ { i , n } = ( H _ { i , n } ) ^ { T } { \pmb w } _ { i , n } , \ { \pmb \tau } _ { n } = \sum _ { i = 1 } ^ { 3 } { \pmb \tau } _ { i , n } \ .\tag{11}
$$

2) Segmentation and Alignment: Each $\Gamma _ { n }$ is segmented by the sub-task phases described in Table II via the sub-movement onset at the specified task points (TP in Fig. 1), referencing the exact time stamps of the video recording in the experiment. Joint kinematics at TPs are extracted to provide the reference frames $ { \boldsymbol { b } } _ { n } ^ { ( p ) }$ for Section II’s learning framework.

To account for variability in interaction speed and duration across repetitions of a variation, we performed temporal alignment between trials of a variation. For all trials in each therapist-patient-task-variation $\Gamma _ { \zeta = \{ a , e , v \} } = \Gamma _ { n = \{ \zeta , r \} }$ , ∀r ∈ $[ 1 , \ldots , 4 ] $ , the joint kinematics in each trial are aligned to the same number of samples $J _ { \zeta }$ using Dynamic Time Warping (DTW) [34], and the subsequent warping paths used to extract the corresponding joint torques. A random trial in $\mathbf { \Gamma } \Gamma _ { \zeta }$ is chosen as the reference signal. All variations of a therapist-patienttask $\mathbf { r } ~ = ~ \mathbf { r } _ { \zeta = \{ a , e , v \} } , ~ \forall v ~ \in ~ [ 1 , \ldots , 6 ]$ after alignment are normalised to a common [0, 1] time scale.

## D. Generalisation (Variation Split) and Validation (Monte Carlo Sampling)

From the six task variations per persona, we selected two variations as unseen variations to evaluate the generalisation capability of the algorithms. The therapist-patient-task-specific interaction model $g _ { a , e }$ was trained on four variations, and generalised to the two unseen variations. To ensure sufficient coverage of the interaction space for training, only sets of two unseen variations differing by at least one level in both parameters were considered. This constraint yielded six valid train-generalisation combinations per $\{ a , e \}$

We validated the performance of LfD when reconstructing interaction for seen variations by randomly holding out one trial from each of the four training variations and evaluate $g _ { a , e }$ against these held-out trials. We perform four Monte Carlo samplings for each combination to ensure each trial in the training set appears in the validation set once.

We trained a separate $g _ { a , e }$ model across all combinationssamplings and evaluated it against the held-out trials and unseen variations, ensuring complete coverage of trainvalidation-generalisation combinations.

## E. Outcome Measures

The variability of therapist behaviour in a variation $\pmb { \tau } _ { \zeta = \{ a , e , v \} } = \pmb { \tau } _ { n = \{ \zeta , r \} } , \ \forall r \in [ 1 , \ldots , 4 ]$ is quantified by the central behaviour (e.g. mean) and the behaviour boundaries of the therapist (e.g. 95% confidence interval) [13]. Using joint kinematic profiles $\mathbf { \delta } _ { q _ { n } } , \dot { \mathbf { q } } _ { n }$ from trials that were held out or in the unseen variations, the reconstructed torque profile, $\hat { \pmb { \tau } } _ { n } = \hat { g } _ { a , e } ( { \pmb q } _ { n } , \pmb { \dot { q } } _ { n } )$ is consistent to the therapist’s intended interaction in that variation if $\scriptstyle { { \hat { \tau } } _ { n } }$ lies within $\tau _ { \zeta }$

However, given the limited number of repetitions of a variation, we opt for a non-parametric version of the interpretation used by Just [13]. The central behaviour in each $\tau _ { \zeta }$ is instead represented by the therapist’s mid-range sample $m _ { \zeta , j }$ at each $j ^ { t h }$ sample and peak torque $m _ { \zeta , \mathrm { p e a k } }$ , given as

$$
m _ { \zeta , j } = \frac { \underset { r = 1 , 2 , 3 , 4 } { \operatorname* { m a x } } \{ \tau _ { n , j } \} + \underset { r = 1 , 2 , 3 , 4 } { \operatorname* { m i n } } \{ \tau _ { n , j } \} } { 2 } ,\tag{12}
$$

$$
m _ { \zeta , \mathrm { p e a k } } = \frac { \displaystyle { \operatorname* { m a x } } _ { r = 1 , 2 , 3 , 4 } \{ \pmb { \tau } _ { n , \mathrm { p e a k } } \} + \operatorname* { m i n } _ { r = 1 , 2 , 3 , 4 } \{ \pmb { \tau } _ { n , \mathrm { p e a k } } \} } { 2 } \ .\tag{13}
$$

The reconstructed torques $\scriptstyle { { \hat { \tau } } _ { n } }$ , are evaluated against the variability of therapist behaviour using these outcome measures:

1) Normalised average reconstruction error $\epsilon _ { n }$

$$
\epsilon _ { n } = \frac { 1 } { J _ { \zeta } } \sum _ { j = 0 } ^ { J _ { \zeta } } \frac { \lvert \hat { \pmb { \tau } } _ { n , j } - \pmb { m } _ { \zeta , j } \rvert } { \operatorname* { m a x } _ { r = 1 , 2 , 3 , 4 } \{ \pmb { \tau } _ { n , j } \} - \pmb { m } _ { \zeta , j } } ;\tag{14}
$$

2) Normalised peak torque difference $\Delta \tau _ { n , \mathrm { p e a k } }$

$$
\Delta \pmb { \tau } _ { n , \mathrm { p e a k } } = \frac { \left| \hat { \pmb { \tau } } _ { n , \mathrm { p e a k } } - \pmb { m } _ { \zeta , \mathrm { p e a k } } \right| } { \underset { r = 1 , 2 , 3 , 4 } { \operatorname* { m a x } } \left\{ \pmb { \tau } _ { n , \mathrm { p e a k } } \right\} - \pmb { m } _ { \zeta , \mathrm { p e a k } } } \ a n d\tag{15}
$$

3) Coverage $C _ { n } \ ( \% )$

$$
\begin{array} { r l r } { c _ { n , j } } & { = } & { \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f } \ \hat { \pmb { \tau } } _ { n , j } \in [ \displaystyle \operatorname* { m i n } _ { r = 1 , 2 , 3 , 4 } , \displaystyle \operatorname* { m a x } _ { r = 1 , 2 , 3 , 4 } ] \{ \pmb { \tau } _ { n , j } \} , } \\ { 0 } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{16}
$$

$$
\begin{array} { r c l } { \displaystyle C _ { n } } & { = } & { \displaystyle \frac { 1 } { J _ { \zeta } } \sum _ { j = 0 } ^ { J _ { \zeta } } c _ { n , j } ~ . } \end{array}\tag{17}
$$

We interpret $\epsilon _ { n }$ and $\Delta \tau _ { n , \mathrm { p e a k } }$ as how far away on average $\scriptstyle { { \hat { \tau } } _ { n } }$ are from the central behaviour of the therapist in each variation, normalised by the behaviour boundaries. Values below 1 indicate that $\scriptstyle { \hat { \tau } } _ { n }$ stays within the boundaries of $\tau _ { \zeta } .$ , producing physical inputs consistent with the intended therapist interaction under the seen / unseen variation. We interpret $C _ { n }$ as how long on average $\scriptstyle { \hat { \tau } } _ { n }$ stays within $\tau _ { \zeta }$ as a percentage of the normalised time.

## F. Statistical Analysis

Outcome measures values were averaged across the ten different DOFs and subsequently all samples of all trainvalidation-generalisation combinations, resulting in one sample per candidate method per patient persona per outcome measure for each participant in both the validation (seen variations) and generalisation (unseen variations) cases. Outliers, outside of 1.5IQR, were removed.

After testing distributions for normality, each outcome measure was analysed with a two-way repeated-measures ANOVA testing the effect of the patient personas, candidate reconstruction methods and interaction. The Greenhouse-Geisser correction was applied where appropriate based on Mauchly’s test of sphericity. For each outcome measure, the validation and generalisation cases were analysed separately, leading to six separate ANOVAs. Where appropriate, post-hoc analysis were performed to test the independent effects of personas and candidate methods (Holm correction applied to reduce false negatives risk).

## IV. RESULTS

This section presents the experimental results of the proposed therapist–patient interaction learning framework. The reconstruction performance of the proposed TPGMM framework is evaluated and compared with the benchmark LUT approach for both seen and unseen task variations using the defined outcome measures and statistical analyses.

## A. Participants and Exclusion

Twenty four participants without upper-limb injury or known neurological injury $( \#$ of pairs: 12, Age: $2 5 . 2 \pm 0 . 5 ,$ Height: $1 6 9 . 4 \pm { 1 . 8 }$ cm, Gender: 20f) were recruited for the study. All participants were instructed to use their right hand when acting out the three patient personas.

For the first five pairs, “therapists” were allowed to freely interact with their “patient”. However, sensor engagement was neglected during the majority of the session, resulting in minimal recorded interactions. Hence, participants after the fifth pair were instructed to restrict their interaction to the 3D-printed cuffs, ensuring that interaction is fully captured by the force sensors. Participants were given time to adjust their interaction with the “patient” before the start of each variation.

Data from the first five pairs were therefore discarded, leaving data from 14 participants for the full analysis. Data from participant #20 during the second variation in the third patient persona $\Gamma _ { \zeta = \{ 2 0 , 3 , 2 \} }$ encountered a communication error that corrupted two of four trials. Hence, outcome measures obtained from training or test samples from $\Gamma _ { \zeta = \{ 2 0 , 3 , 2 \} }$ were excluded from the analysis.

## B. Visualisation for $\tau _ { \zeta }$ and $\scriptstyle { { \hat { \tau } } _ { n } }$

A representative example of the variability of the therapist applied torque for a given variation and the four repetitions $( \tau _ { \zeta = \{ 1 8 , 2 , 1 \} } )$ is shown in Fig. 3 (Left). The corresponding torque reconstructions using TPGMM and LUT approaches $( \hat { \tau } _ { n = \{ 1 8 , 2 , 1 , 3 \} } )$ are shown in Fig. 3 (Right). $\tau _ { \zeta }$ across patient personas-variations and participants for the shoulder flexionextension is shown in Fig. S1 in the Supplementary Material.

TABLE III: Top: Repeated-measures ANOVA for Effects of Persona, Method and Interaction. Bottom: Holm-corrected pairwise comparisons between independent within-subject factor levels. (Val: Seen Variations, Gen: Unseen Variations)
<table><tr><td colspan="2"></td><td colspan="10">RM-ANOVA</td></tr><tr><td></td><td></td><td></td><td>Persona</td><td></td><td></td><td></td><td></td><td>Method</td><td></td><td></td><td></td><td>Persona * Method</td><td></td><td></td></tr><tr><td rowspan="3">Measure Case</td><td></td><td>SS df1 df2</td><td>MS</td><td>F</td><td>P-val</td><td>SS</td><td>df1 df2</td><td>MS</td><td>F</td><td>P-val</td><td>SS</td><td>df1 df2 MS</td><td>F</td><td>P-val</td></tr><tr><td>Val</td><td>0.348 2</td><td>26</td><td>0.174</td><td>15.590.0***</td><td>0.133</td><td>1 13</td><td>0.133</td><td>7.623 0.016*</td><td>0.003</td><td>2 26</td><td>0.002</td><td>0.162</td><td>0.766</td></tr><tr><td>ε Gen</td><td>0.612</td><td>2 26</td><td>0.306</td><td>7.965</td><td>0.002**</td><td>0.164 1</td><td>13</td><td>0.164</td><td>8.134 0.014*</td><td>0.013</td><td>2 26</td><td>0.006</td><td>0.609</td><td>0.551</td></tr><tr><td rowspan="2"> $\Delta \tau _ { p e a k }$ </td><td>Val</td><td>1.032 2</td><td>26</td><td>0.516</td><td>15.069 0.000***</td><td>5.694</td><td>1 13</td><td>5.694</td><td>74.0260.000***</td><td></td><td>0.667 2</td><td>26</td><td>0.333</td><td>11.752 0.000***</td><td></td></tr><tr><td>Gen</td><td>0.265</td><td>2 26</td><td>0.133</td><td>1.95</td><td>0.163</td><td>2.01 1</td><td>13</td><td>2.01</td><td>24.514 0.000***</td><td>0.592</td><td>26 2</td><td>0.296</td><td></td><td>8.785 0.005**</td></tr><tr><td rowspan="2">C(%)</td><td>Val</td><td>311.725 2</td><td>26</td><td>155.863</td><td>18.237 0.000***</td><td>88.262</td><td>1</td><td>13 88.262</td><td>4.683</td><td>0.05</td><td>15.794</td><td>2 26</td><td>7.897</td><td>0.914</td><td>0.413</td></tr><tr><td>Gen</td><td>1200.068</td><td>2 26</td><td>600.034</td><td>18.429 0.000***</td><td>295.573</td><td>1</td><td>13</td><td></td><td>295.573 17.296 0.001**</td><td>23.504</td><td>2 26</td><td>11.752</td><td>1.508</td><td>0.24</td></tr></table>

Post-Hoc
<table><tr><td colspan="2" rowspan="2"></td><td colspan="2">Persona 1-2</td><td rowspan="2"></td><td colspan="2">Persona 1-3</td><td rowspan="2"></td><td colspan="2" rowspan="2">Persona 2-3</td><td rowspan="2"></td><td rowspan="2">LUT-TPGMM</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Measure Case</td><td></td><td>T</td><td>df</td><td>P-val</td><td>T -5.599</td><td>df</td><td>P-val</td><td>T</td><td>df P-val</td><td>T df</td><td>P-val</td></tr><tr><td>Val</td><td></td><td>-4.757 13</td><td>0.001**</td><td></td><td>13</td><td>0.000***</td><td>0.605</td><td>13 0.555</td><td>-2.761</td><td>13 0.016*</td></tr><tr><td rowspan="2">ε  $\Delta \tau _ { p e a k }$ </td><td>Gen</td><td>3.902 13</td><td></td><td>0.005**</td><td>3.189</td><td>13 0.014*</td><td></td><td>-0.373 13 0.715</td><td></td><td>2.852 13 0.014*</td><td></td></tr><tr><td>Val</td><td>-4.386 13</td><td></td><td>0.001**</td><td>-4.708</td><td></td><td>13 0.001**</td><td></td><td>-0.489 13 0.633</td><td>-8.604 13</td><td>0.000***</td></tr><tr><td rowspan="2">C(%)</td><td>Gen</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>-4.951 13 0.000***</td><td></td></tr><tr><td>Val</td><td></td><td></td><td>6.084130.000***</td><td></td><td></td><td>4.306130.002**</td><td></td><td>-0.937 13 0.366</td><td></td><td></td></tr><tr><td rowspan="2"></td><td>Gen</td><td>-6.243 13 0.000***</td><td></td><td></td><td></td><td>-4.706 13 0.001**</td><td></td><td></td><td>0.163 13 0.873</td><td>-4.159 13 0.001**</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

<sup>∗</sup>p < 0.05, <sup>∗∗</sup>p < 0.01, <sup>∗∗∗</sup>p < 0.001.

![](images/4032d1b488c9b96a76857549ad8fbd7ed542aef6ddf8a1b9d4b93a077366c09d.jpg)

A.  
![](images/c6b1bfd88a7b33768b73ea5059dd44095e386defc01ddb0fd493f48601305a33.jpg)

![](images/fcb62560abf997053fafa8667b9bae427ad0517f3790aabd4b7f7eb283da6d15.jpg)

B.  
![](images/5f261ce6081a6e8f50aeb7a538a873466f34a94dbdcedc05c51a04a178bacf8d.jpg)

![](images/e5a5a8f11c86530ad66810f39b15ac71d1ee7ac976387832cc06974ab7d512a8.jpg)  
time (s)

![](images/f12ee963ff5f179214a3c1236f5c2c9354fd2116d66b2851a2d2826820d06491.jpg)  
time (s)

![](images/e8fcea27c309b25770a885713b7006c8fd09891eff35b3c40680079b955affdc.jpg)

![](images/eb88b48ab30a0d9a3958f169689ec3d27096fc14a0307540e024307ffaf4c83c.jpg)  
time (s)  
Fig. 3: A: Variability of therapist behaviour, $\tau _ { \zeta = \{ 1 8 , 2 , 1 \} }$ in the trunk, shoulder and elbow. The dotted line represents the mid range profile $m _ { \zeta = \{ 1 8 , 2 , 1 \} }$ and the shaded region represents the behaviour boundaries. B: Reconstructed torques from TPGMM (TPGMM) and vector search (LUT) for $ \tau _ { n = \{ 1 8 , 2 , 1 , 3 \} }$

![](images/b42bebf7ff32933e234aa857603505206f6dca5343223af408e56dc1c104fad4.jpg)  
Fig. 4: Distribution of outcome measures for both candidate methods between the validation (white background) and generalisation (grey background) cases across patient personas. The therapist’s own variability of behaviour is shown as reference.

The distribution of the three outcome measures for both candidate methods between the validation and generalisation cases across patient personas are presented on Fig. 4.

All outcome measures in both cases are normally distributed $( R ^ { 2 } \mathrm { ~  ~ { ~ > ~ } ~ } 9 0 )$ , while ε and $\Delta \tau _ { p e a k }$ in the validation case appear approximately normal. The deviation from normality of ε and $\Delta \tau _ { p e a k }$ is traced to a single participant’s atypical performance in Persona 2 and 3 for TPGMM. We retained this participant to avoid removing potentially representative samples that reflect genuine population variability. Residuals of each outcome measure for both cases are shown in Fig. S2 in the Supplementary Material.

Outcomes of the repeated-measure ANOVAs and associated post-hoc tests are presented in Table III. The results of the ANOVA revealed a significant effect of patient personas on all cases-outcome measures except for $\Delta \tau _ { \mathrm { p e a k } }$ in the generalisation case. The choice of candidate methods also significantly affect all cases-outcome measures except for C in the validation case. The interaction effect between Persona and Method was not statistically significant for any outcome measure except for $\Delta \tau _ { \mathrm { p e a k } }$ in both cases, hence the post-hoc pairwise comparisons were not conducted.

1) Validation Performance: Both methods reconstructed therapist-intended interactions effectively, with both method’s ε distributed below or around 1 and C distributed around 68%. However, TPGMM demonstrated poor fidelity in capturing finer interaction details compared to vector search, evidenced by its $\Delta \tau _ { \mathrm { p e a k } }$ distributed around 1.2. The choice of method produced a small but significant effect on all outcome measures except for C in the validation case, showing vector search’s performance over TPGMM.

2) Generalisation Performance: Both method’s ε are distributed above 1 but remained close to the performance boundary, whereas C are distributed around 52 − 56% across patient personas for both methods. indicating reconstructions trained with minimal training data slightly deviates from the therapist-intended behaviour for unseen variations. TPGMM demonstrated a small but significant advantage over vector search for ε and C, albeit with the same low fidelity in capturing finer interaction details.

3) Effects of Patient Persona: Subsequent post-hocs for patient persona comparisons revealed significant effects when comparing between Persona 1 to Persona 2 and 3, but no significant effects between Persona 2 and 3. The results complies with the task complexity design of each patient persona, whereby Persona 1’s task is simpler, and Persona 2 and 3’s tasks are almost equivalent

The increase in task complexity produced opposing effects on method performance between cases. In the validation case, increased complexity from Persona 1 to Persona 2 and 3 significantly degraded performance (ε ↑, C ↓). Conversely, in the generalisation case, increased complexity significantly improved performance (ε ↓, C ↑).

In this work, we conducted a preliminary study that extends previous works [13], [22] on learning physical therapist-patient interaction in upper limb rehabilitation from simple two-point movement to more complex varying task-specific exercises. We learn the relation between patient movement and physical therapist actions using [22]’s vector search algorithm as a benchmark compared against the proposed TPGMM. Both methods are trained on a subset of variations of each task, validated against the same variations with unseen patient movement and generalised against unseen variations.

Our results demonstrate that both candidate methods reconstructed interactions that stay within the variability of therapist behaviour for seen variations, whereby the vector search algorithm has a slight advantage over TPGMM. In unseen variations, both candidate methods reconstructed interactions that slightly deviated from the variability of therapist behaviour, while under minimal demonstration constraints. TPGMM has a small but significant advantage over the vector search algorithm when reconstructing interaction for unseen variations. However, it should be acknowledged that TPGMM is unable to capture the finer interaction details such as peak torques, of which warrant further investigation.

## A. Reduction in Interaction Coverage and Training Samples

Here, we also validated the use of the vector search algorithm [22] to learn and replicate physical therapist-patient interaction for more complex movement under various conditions, albeit only when reconstructing interaction for variations that is within the training dataset. The difference in performance between TPGMM and the vector search algorithm are also significantly small in the unseen cases.

Several caveats is noted on the performance of the vector search. The selected variations cover 66% of the interaction workspace, whereby samples used for training cover 75% of the seen variations, resulting in 12 training samples out of 24 demonstrations. We suspect the consistency of the therapist’s own behaviour across the variations of the task contribute to the performance of the vector search. Hence, further works should explore the effects of reducing variation coverage and training samples, with the aim of reducing the number of demonstrations that the therapist has to provide in the session.

## B. Correlation Between Task Complexity and LfD Performance

A significant relation between task complexity and the performance of LfD when generalising for unseen variations was observed. In [25], it was noted that TPGMM’s performance over DMPs improves as the task complexity increases, albeit the authors did not analyse the effects of task complexity. Here, we showed that the performance of both candidate methods improves significantly as task complexity increases when generalising for unseen variations. However, this comes at the expense of the performance in the seen variations. While a definite explanation cannot be given due to differences in tasks performed and instructed patient diagnosis, this suggests a trade-off between task complexity and therapist behaviour reproduction in seen and new variations.

## C. Limitations and Future works

In the absence of a system that captures full body interaction between two humans, we constrained interaction to three points of contact (clavicle, upper arm, forearm) to simplify system design. We acknowledge that the constraint significantly alters the interaction performed by the participants compared to conventional therapy sessions. During the pilot study (with ten participants), “therapists” were free to interact anywhere, and some interactions were localised to the joints (shoulder, elbow, wrist) which are not sensorised. After enforcing the constraint on interaction points, a similar report was obtained from the remaining participants in that they would preferably support the elbow for reaching motions rather than the upper arm, the wrist for elbow pro-supination, and the fingertips for fine motor control. Six participants report that they would distribute a larger contact surface along a more severe patient’s body, even necessitating the presence of two therapists to ensure patient movement stability.

Physiotherapy students were recruited as they were readily available and had not yet developed clinical habits that could bias their engagement with the system developed in Fig. 2. While meaningful and detailed personas were provided, lack of clinical experience could lead to differing interpretation of patient personas between participants, leading to larger interaction variability than in actual practice.

The LfD frameworks were also evaluated offline on patient kinematics that is unconstrained by any robotic system. In an actual application scenario, the presence of the robot and its physical constraint may affect the interaction of the clinicians in a different way. We also note that the real-time efficacy of the framework with human-in-the loop has yet to be validated, yet TPGMM training lasted between 2 - 4 minutes on a standard CPU depending on task complexity.

These limitations prevents claiming any direct applicability of this findings to actual neuro-rehabilitation settings, nevertheless, the results suggests the suitability of the proposed methods for such scenarios. The next step would be to integrate this framework with an actual rehabilitation robotic system and conduct pilot studies to evaluate its efficacy for rehabilitation post-stroke.

## REFERENCES

[1] C.-Y. Lee and T.-H. Howe, “Effectiveness of activity-based task-oriented training on upper extremity recovery for adults with stroke: a systematic review,” Am. J. Occup. Ther., vol. 78, no. 2, p. 7802180070, 2024.

[2] C. E. Lang and R. L. Birkenmeier, “Upper-extremity task-specific training after stroke or disability,” Bethesda: AOTA Pr, 2013.

[3] E. J. Schneider, N. A. Lannin, L. Ada, and J. Schmidt, “Increasing the amount of usual rehabilitation improves activity after stroke: a systematic review,” J. Physiother., vol. 62, no. 4, pp. 182–187, 2016.

[4] C. E. Lang, K. R. Lohse, and R. L. Birkenmeier, “Dose and timing in neurorehabilitation: prescribing motor therapy after stroke,” Curr. Opin. Neurol., vol. 28, no. 6, pp. 549–555, 2015.

[5] V. Crocher, K. Brock, J. Simondson, M. Klaic, and M. P. Galea, “Robotic task specific training for upper limb neurorehabilitation: a mixed methods feasibility trial reporting achievable dose,” Disabil. Rehabil., vol. 47, no. 9, pp. 2349–2357, 2025.

[6] M. J. Johnson, M. Mohan, and R. Mendonca, “Therapist-patient interactions in task-oriented stroke therapy can guide robot-patient interactions,” Int. J. Soc. Robot., vol. 14, no. 6, pp. 1527–1546, 2022.

[7] A. Chrungoo, P. Shirsat, and M. J. Johnson, “Towards perception driven robot-assisted task-oriented therapy,” in 2015 International Conference on Rehabilitation Robotics (ICORR), 2015, pp. 660–665.

[8] Y.-H. Cho, M.-H. Lee, and Y.-J. Cha, “Effects of robot-assisted taskoriented training on upper limb function and activities of daily living in patients with stroke: A systematic review,” J. Int. Med. Res., vol. 54, no. 5, p. 03000605261446490, 2026.

[9] S. G. Rozevink, J. M. Hijmans, K. A. Horstink, and C. K. van der Sluis, “Effectiveness of task-specific training using assistive devices and taskspecific usual care on upper limb performance after stroke: a systematic review and meta-analysis,” Disabil. Rehabil.: Assist. Technol., vol. 18, no. 7, pp. 1245–1258, 2023.

[10] S. Dalla Gasperina, L. Roveda, A. Pedrocchi, F. Braghin, and M. Gandolla, “Review on patient-cooperative control strategies for upper-limb rehabilitation exoskeletons,” Front. Robot. AI, vol. 8, p. 745018, 2021.

[11] A. Basteris, S. M. Nijenhuis, A. H. Stienen, J. H. Buurke, G. B. Prange, and F. Amirabdollahian, “Training modalities in robot-mediated upper limb rehabilitation in stroke: a framework for classification based on a systematic review,” J. NeuroEng. Rehabil., vol. 11, no. 1, p. 111, 2014.

[12] S. Micera, M. Caleo, C. Chisari, F. C. Hummel, and A. Pedrocchi, “Advanced neurotechnologies for the restoration of motor function,” Neuron, vol. 105, no. 4, pp. 604–620, 2020.

[13] F. Just, “Fusing conventional and robotic rehabilitation therapy,” Ph.D. dissertation, ETH Zurich, 2020.

[14] M. Maaref, A. Rezazadeh, K. Shamaei, and M. Tavakoli, “A gaussian mixture framework for co-operative rehabilitation therapy in assistive impedance-based tasks,” IEEE J. Sel. Top. Signal Process., vol. 10, no. 5, pp. 904–913, 2016.

[15] L. Vianello, M. Short, J. Manczurowsky, E. B. Kuc¸¨ uktabak, F. Di Tom-¨ maso, A. Noccaro, L. Bandini, S. Clark, A. Fiorenza, F. Lunardini et al., “Robot-mediated physical human–human interaction in rehabilitation: A position paper,” IEEE Rev. Biomed. Eng., 2025.

[16] E. B. Kuc¸¨ uktabak, M. R. Short, L. Vianello, D. Ludvig, L. Hargrove,¨ K. Lynch, and J. Pons, “Therapist-exoskeleton-patient interaction for gait therapy,” Sci. Robot., vol. 11, no. 115, p. eadz9628, 2026.

[17] C. Mart´ınez and M. Tavakoli, “Learning and robotic imitation of therapist’s motion and force for post-disability rehabilitation,” in 2017 IEEE International Conference on Systems, Man, and Cybernetics (SMC). IEEE, 2017, pp. 2225–2230.

[18] J. Fong and M. Tavakoli, “Kinesthetic teaching of a therapist’s behavior to a rehabilitation robot,” in 2018 International Symposium on Medical Robotics (ISMR). IEEE, 2018, pp. 1–6.

[19] C. M. Martinez, J. Fong, S. F. Atashzar, and M. Tavakoli, “Semiautonomous robot-assisted cooperative therapy exercises for a therapist’s interaction with a patient,” in 2019 IEEE Global Conference on Signal and Information Processing (GlobalSIP), 2019, pp. 1–5.

[20] M. Najafi, C. Rossa, K. Adams, and M. Tavakoli, “Using potential field function with a velocity field controller to learn and reproduce the therapist’s assistance in robot-assisted rehabilitation,” IEEE/ASME Trans. Mechatron., vol. 25, no. 3, pp. 1622–1633, 2020.

[21] S. M. R. Sorkhabadi, M. Smith, R. Khodmbashi, R. Lopez, M. Raasch, T. Maruyama, C. Kwasnica, and W. Zhang, “Learning post-stroke gait training strategies by modeling patient-therapist interaction,” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 31, pp. 1687–1696, 2023.

[22] B. Luciani, M. Sommerhalder, M. Gandolla, P. Wolf, F. Braghin, and R. Riener, “Therapists’ force-profile teach-and-mimic approach for upper-limb rehabilitation exoskeletons,” IEEE Trans. Med. Robot. Bionics, vol. 6, no. 4, pp. 1658–1665, 2024.

[23] C. Lauretti, F. Cordella, E. Guglielmelli, and L. Zollo, “Learning by demonstration for planning activities of daily living in rehabilitation and assistive robotics,” IEEE Robot. Autom. Lett., vol. 2, no. 3, pp. 1375–1382, 2017.

[24] C. Lauretti, F. Cordella, A. L. Ciancio, E. Trigili, J. M. Catalan, F. J. Badesa, S. Crea, S. M. Pagliara, S. Sterzi, N. Vitiello et al., “Learning by demonstration for motion planning of upper-limb exoskeletons,” Front. Neurorobotics, vol. 12, p. 5, 2018.

[25] J. Q. Loh, V. Crocher, D. Oetomo, and Y. Tan, “Can learning from demonstration approaches encode and generalise human movements for neurorehabilitation?” in 2025 International Conference On Rehabilitation Robotics (ICORR), 2025, pp. 1–7.

[26] L. Raviv, G. Lupyan, and S. C. Green, “How variability shapes learning and generalization,” Trends Cogn. Sci., vol. 26, no. 6, pp. 462–483, 2022.

[27] C. H. Shea and R. M. Kohl, “Composition of practice: Influence on the retention of motor skills,” Res. Q. Exerc. Sport, vol. 62, no. 2, pp. 187–195, 1991.

[28] ——, “Specificity and variability of practice,” Res. Q. Exerc. Sport, vol. 61, no. 2, pp. 169–177, 1990.

[29] A. Pollock, S. E. Farmer, M. C. Brady, P. Langhorne, G. E. Mead, J. Mehrholz, and F. Van Wijck, “Interventions for improving upper limb function after stroke,” Cochrane Database Syst. Rev., no. 11, 2014.

[30] H. Ravichandar, A. S. Polydoros, S. Chernova, and A. Billard, “Recent advances in robot learning from demonstration,” Annu. Rev. Control Robot. Auton. Syst., vol. 3, no. 1, pp. 297–330, 2020.

[31] S. Calinon, “A tutorial on task-parameterized movement learning and retrieval,” Intell. Serv. Robot., vol. 9, no. 1, pp. 1–29, 2016.

[32] T. Hastie, R. Tibshirani, J. Friedman, and J. Franklin, “The elements of statistical learning: data mining, inference and prediction,” Math. Intell., vol. 27, no. 2, pp. 83–85, 2005.

[34] S. Sempena, N. U. Maulidevi, and P. R. Aryan, “Human action recognition using dynamic time warping,” in Proceedings of the 2011 International Conference on Electrical Engineering and Informatics, 2011, pp. 1–5.

[33] XSENS, “Mvn anatonomical model,” MVN User Manual, 2025. [Online]. Available: https://go.unimelb.edu.au/ff32