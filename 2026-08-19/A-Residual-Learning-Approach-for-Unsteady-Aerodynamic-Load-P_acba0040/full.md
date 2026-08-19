This is a preprint of the following article   
Sanghi, D. and Cesnik, C. E. S., A Residual Learning Approach for Unsteady Aerodynamic Load Prediction, International Forum on Aeroelasticity and Structural Dynamics, Go¨ettingen, Germany, June 2026. The published article to appear may difer from this preprint.

# A Residual Learning Approach for Unsteady Aerodynamic Load Prediction

Divya Sanghi<sup>1,\*</sup> and Carlos E. S. Cesnik<sup>1,\*</sup> <sup>1</sup>University of Michigan, Ann Arbor, MI, 48109

## Abstract

This paper investigates the feasibility of using residual learning to improve unsteady aerodynamic load prediction for aeroelastic applications. The machine learning technique selected for the study is the long short-term memory (LSTM) neural network, which is used for its suitability for sequential data with aerodynamic memory efects. The approach is investigated for the NLR 7301 airfoil benchmark using high-fidelity CFD lift data for prescribed pitch and plunge motions in the transonic flow regime in the presence of shock motion. An analytical unsteady aerodynamic model based on the Wagner function is used as a physics-based baseline, and the neural network is trained to learn the diference between the CFD lift coeficient and the Wagner prediction. The residual model is compared with a direct neural-network model trained to predict the CFD lift coeficient. The comparison includes feature and normalization studies, external benchmark cases, and leaveone-out and leave-family-out generalization tests across a range of sinusoidal and non-sinusoidal motions. The residual model performs best when its inputs align with the Wagner formulation variables, generally giving lower error and more consistent performance across training runs, though the direct model remains more accurate for some high-frequency cases. The residual model also generalizes better in the leave-one-out and leave-family-out tests, with a smaller increase in error than the direct model when entire motion families are withheld from training. Overall, the results indicate that residual learning shows promise as a modular approach for augmenting classical loworder aerodynamic theories, especially when the physics baseline removes a structured part of the aerodynamic response and leaves a lower-variance correction for the neural network to learn.

## 1 Introduction

Accurate prediction of unsteady aerodynamic loads is essential for aeroelastic analysis, flutter assessment, and control-oriented modeling. The unsteady aerodynamic response depends not only on the instantaneous kinematics, but also on the reduced frequency and prior motion history. Classical thin-airfoil theories [1, 2], such as Theodorsen’s formulation and the Wagner function, decompose the attached-flow response into circulatory and noncirculatory contributions. These models provide physically interpretable low-order baselines, but they are derived under linear, incompressible, attached-flow assumptions. Their accuracy is therefore limited when compressibility efects, shock motion, viscous separation, dynamic-stall efects, or complex three-dimensional geometry become important. High-fidelity CFD can represent these efects more accurately, but at substantially higher computational cost, motivating reduced-order models that retain the relevant unsteady aerodynamic efects at a lower cost.

Neural-network-based ROMs approximate the input–output behavior of an aerodynamic system directly from data and avoid repeated solution of the governing equations during online prediction. For unsteady aerodynamic loads, this input–output relation is inherently sequential: the lift at a given time depends not only on the current pitch, plunge, flow conditions, and motion derivatives, but also on the prior motion history. Recurrent neural networks (RNNs) are therefore widely used for unsteady aerodynamic modeling because their hidden states provide a data-driven mechanism for carrying information from previous time steps into the current prediction. Prior studies have used recurrent multilayer perceptrons, radial-basis-function networks, neuro-fuzzy architectures, multi-kernel neural networks, and connected neural networks to predict aerodynamic coeficients, dynamic-stall loads, limit-cycle-oscillation behavior, and pitch–plunge responses at varying flow conditions [3, 4, 5, 6, 7, 8, 9, 10], with a comprehensive review provided by Kou and Zhang [11]. Other studies have combined neural networks with proper orthogonal decomposition (POD) or surrogate modeling techniques to predict distributed pressure or aerodynamic-field quantities rather than only integral coeficients [12, 13, 14, 15]. However, conventional gradient-trained RNNs can have dificulty learning long-term dependencies because gradients may decay over repeated time steps [16]. Long short-term memory (LSTM) networks address this issue through gated cell states that retain sequence information over many time steps [17]. Because LSTMs have been shown to preserve amplitude, phase, and history-dependent load behavior in unsteady aerodynamic ROMs [15], they are adopted in the present work.

The NLR 7301 airfoil has served as a benchmark test case for data-driven unsteady aerodynamic modeling. Winter and Breitsamter used radial-basis-function neural networks to model unsteady aerodynamic loads for NLR 7301 pitch–plunge responses [6]. Wang et al. investigated techniques for improving neural-network aerodynamic ROMs for the same benchmark [18] and later developed multivariate recurrent models for both scalar lift histories and pressure-distribution predictions [15]. The Wang et al. dataset spans flow conditions from Mach 0.65 to 0.75 and includes single-harmonic, amplitude-modulated, Gaussian-pulse, and noise-perturbed pitch–plunge motion cases [15]. In those studies, the neural-network ROMs were trained to predict the full CFD lift coeficient directly, with the later multivariate RNN study also reconstructing pressure distributions [18, 15]. As a result, the learned mapping is asked to represent quasi-steady lift, unsteady phase lag, wake memory, and compressibility-related trends within a single target. This combined target can be dificult to generalize when the test motion difers from the training motion family. Moreover, the ROM remains purely data-driven: the physics is present only implicitly in the CFD training data, and no aerodynamic governing equation is explicitly retained in the learned model.

Physics-guided residual and layered learning strategies have been investigated to reduce the burden of learning the full high-fidelity response directly. Kou and Zhang introduced layered ROMs in which a fitted linear aerodynamic surrogate is corrected by a nonlinear neural-network model for nonlinear aerodynamic and aeroelastic prediction [19]. They later proposed a multi-fidelity framework in which an Euler-CFD baseline is corrected using data-driven residuals to approximate high-fidelity RANS responses for airfoils [20]. He and Schumann analyzed a physics-based deep recurrent residual neural-network model applied to fixed-wing flight-dynamics modeling [21]. Other multi-fidelity neural-network approaches have fused low- and high-fidelity aerodynamic data for coeficient prediction [22] and aerodynamic shape optimization [23]. Residual learning has also been used for aerodynamic coeficient prediction [24], DeepONet-based multi-fidelity ROM correction [25], and flight-load prediction with neural-network residual kriging [26]. More recent neural network augmentation studies include increased-order modeling of aerodynamic nonlinearities [27] and control-oriented aerodynamic-coeficient modeling for distributed-propulsion vehicles [28].

In previous residual-learning studies, the baseline has typically been a fitted linear or statespace model, or a lower-fidelity CFD model. Whether an analytical low-fidelity model can provide an efective residual baseline for unsteady aerodynamic load prediction across a broad range of motion types remains insuficiently studied. The Wagner function provides a convenient physicsbased baseline because it requires only the prescribed airfoil kinematics and flow speed, without additional fitting data or CFD solves, although its applicability is limited by its linear, incompressible and attached-flow assumptions. In addition, the robustness of physics-based residual learning under motion-type distribution shifts has not been systematically quantified, particularly when individual cases or entire motion families are withheld from training. The objective of this paper is therefore to quantify how a residual target based on an analytical Wagner indicial-function baseline changes the generalization behavior of an LSTM aerodynamic ROM relative to direct c prediction for the NLR 7301 airfoil. The evaluation spans multiple pitch–plunge input families, including single-harmonic, amplitude-modulated harmonic, Gaussian-pulse, and noise-perturbed harmonic motions, using leave-one-out, leave-family-out, and external benchmark tests. By com bining analytical unsteady aerodynamic theory with a data-driven residual correction, this study evaluates a practical methodology for improving unsteady aerodynamic predictions while preserving physical interpretability and computational eficiency for future aeroelastic, control-oriented, and multi-fidelity modeling applications.

The remainder of the paper is organized as follows. Section 2 defines the NLR 7301 airfoil benchmark test case, training and testing data, and the Wagner baseline. Section 3 presents the direct and residual model formulations, training and testing procedures, and error metrics used in the study. Section 4 presents the results and Sec. 5 summarizes the main conclusions and future work.

## 2 Benchmark Problem and Physics Baseline

This section describes the aerodynamic test case, the training and external benchmark data, and the analytical baseline used throughout the study.

## 2.1 NLR 7301 Airfoil CFD Data

This work considers the two-dimensional NLR 7301 airfoil benchmark test case used in prior aerodynamic ROM studies [6, 18, 15]. The high-fidelity data used here are taken from the CFD database generated by Wang et al. [15] for a pitching and plunging airfoil using the in-house CFD code xflow [29]. The simulations were conducted at a Reynolds number of $2 . 1 \times 1 0 ^ { 6 }$ over Mach numbers from 0.65 to 0.75. This Mach range includes transonic flow cases in which moving shocks appear over the airfoil (as shown in Fig. 1), increasing the complexity of the unsteady load prediction problem. The dataset used in this study comprises 48 CFD time histories for training and validation, together with three external benchmark time histories for testing.

![](images/65ecef3fa26d6ce96b1f99f515f56948aa75b7d0fecbbcfb6168563880d6bc64.jpg)  
Figure 1: Snapshots of the NLR 7301 airfoil showing transonic shock motion from the Wang et al. high-fidelity CFD dataset [15].

Table 1: Training motion families in the high-fidelity CFD dataset.
<table><tr><td>Family</td><td>Cases</td><td>Mach numbers</td><td>Frequency content</td></tr><tr><td>Single harmonic</td><td>24</td><td>0.65, 0.70, 0.75</td><td> $k \in [ 0 . 0 8 8 , 3 . 0 0 7 ]$ </td></tr><tr><td>Amplitude- modulated harmonic</td><td>6</td><td>0.65, 0.67, 0.69, 0.71, 0.73, 0.75 Components about</td><td> $k \approx 2 . 1 6$ </td></tr><tr><td>Gaussian pulse</td><td>6</td><td>0.65, 0.67, 0.69, 0.71, 0.73, 0.75 Approx.</td><td> $k \in [ 0 . 6 3 , 3 . 0 2 ]$ </td></tr><tr><td>Noise-perturbed</td><td>12</td><td>0.65, 0.67, 0.69, 0.71, 0.73, 0.75 Approx.</td><td> $k \in [ 0 . 0 3 , 3 . 6 5 ]$ </td></tr></table>

The available input histories include the prescribed pitch angle $\alpha ( t )$ , plunge displacement $h ( t )$ Mach number $M ,$ and the corresponding derivatives $\dot { \alpha } ( t )$ and $\dot { h } ( t )$ . The pitch and plunge motion inputs are analytically defined, and the first-order derivatives are provided directly with the CFD data. Second-order derivatives, $\ddot { \alpha } ( t )$ and $\ddot { h } ( t )$ , when required as model inputs, are estimated from the recorded position histories using second-order central finite diferences. The available aerodynamic outputs include integral and distributed aerodynamic quantities, but the present study focuses only on lift coeficient prediction, $c _ { l } ( t )$

The reduced frequency used throughout this paper is defined as

$$
k = \frac { 2 \omega b } { U } ,\tag{1}
$$

where ω is the angular oscillation frequency, b is the semi-chord, 2b is the full chord, and U is the free-stream speed. This convention is consistent with the study in [15]. The time coordinate in the raw CFD files is the nondimensional time, defined as

$$
\tau = t _ { \mathrm { x f l o w } } = \frac { U t _ { \mathrm { p h y s } } } { 2 b } ,\tag{2}
$$

where $t _ { \mathrm { p h y s } }$ is physical time. For this dataset, the semi-chord is $b = 0 . 0 9 \mathrm { m }$ (full chord $2 b = 0 . 1 8 \mathrm { m } )$ ， and $c = 3 4 0 . 2 9 4 \mathrm { { m } / \mathrm { { s } } }$ is the speed of sound used for dimensional conversion. If dimensional time is needed for interpretation, $t _ { \mathrm { p h y s } } = 2 b \tau / U = 2 b \tau / ( M c )$ . All training, validation, and external testing are performed on the original τ-indexed solver samples, without conversion to physical seconds or resampling to a common time grid. The nondimensional time step varies across motion cases, and all data splits are made by complete motion case to avoid leakage of aerodynamic history information across subsets.

## 2.2 Training and External Benchmark Cases

The training dataset contains 48 cases spanning four prescribed-motion families, summarized in Table 1. The single-harmonic cases provide controlled frequency and Mach variation over a grid of k values, while the three non-sinusoidal families introduce amplitude modulation, transient pulse behavior, and noise-perturbed harmonic samples [15].

The physical character of each family determines its role in the generalization studies. Representative examples of each family are shown in Figs. 2–5. The single-harmonic cases are narrowspectrum sinusoidal inputs in pitch and plunge at fixed frequency and Mach, and they serve as the primary family for leave-one-out and leave-family-out comparisons. The amplitude-modulated harmonic cases impose a low-frequency amplitude envelope onto a sinusoidal carrier near $k \approx 2 . 1 6$ . The resulting pitch and plunge histories have a semi-periodic behavior in which the instantaneous amplitude varies slowly relative to the oscillation period, producing a narrowband but non-stationary excitation. The Gaussian pulse cases concentrate excitation energy in the middle of the simulation window, with pitch and plunge driven by a chirp-like modulated pulse. The transient ramp-up and ramp-down phases require the aerodynamic model to capture initial transient and decay behavior that sinusoidal training data do not represent. The noise-perturbed harmonic cases are constructed by superimposing noise onto a single-harmonic base, yielding non-periodic inputs whose frequency content spans the full k range used in training. For each Mach number, the two variants use the same pitch input but opposite sign for the plunge inputs, so $\alpha _ { 2 } ( t ) = \alpha _ { 1 } ( t )$ and $h _ { 2 } ( t ) = - h _ { 1 } ( t )$

![](images/536bd5c47316d10fb67c72eeb5b8f69a15b7b69f9d27223577822805b2a77d22.jpg)  
(a) Excitation

![](images/59fe54840e6babf3da2e146139c245d8548dd115feab30b36a648349f23c67ad.jpg)  
(b) Lift coeficient

Figure 2: Representative single-harmonic training case.  
![](images/43da7bb3a4cda515da85c7a6399dbde452d03f0d4e3f768812f1ed76f108bb8c.jpg)  
(a) Excitation

![](images/ef44a341229d1bff4e1cce2144df029c861cf390f5b21fd077e3bdfa9c8470ed.jpg)  
(b) Lift coeficient  
Figure 3: Representative amplitude-modulated harmonic training case.

In addition to the training data, three external benchmark cases are used as the verification set. Table 2 gives the Mach number and reduced frequency for all three cases, and their prescribed kinematics and CFD lift-coeficient time histories are shown in Figs. 6–8. These benchmark cases are not used for fitting and are retained as complete time histories in the same $\tau = t _ { \mathrm { x f l o w } }$ convention as the training cases. The three cases cover distinct kinematic regimes: pure pitch at high reduced frequency, pure plunge at moderate reduced frequency, and coupled pitch–plunge motion at low reduced frequency.

![](images/363460f1b5886755c71e154ce6ee135b5ca36152b0644442a5dce1f32fa334c0.jpg)  
(a) Excitation

![](images/abab23a9de7eb87f193f6fe08a464eedb22ca9c99b79552b1bdb78617d8bdd3a.jpg)  
(b) Lift coeficient  
Figure 4: Representative Gaussian-pulse training case.

![](images/679fe36c040d243981c69830209d648942c1e805683da5f5d5743c15171f4e13.jpg)  
(a) Excitation

![](images/6dc732c7e40d2aa0aa8b5bada8eae84980de13e0899260b9f75c9a50c99c335b.jpg)  
(b) Lift coeficient  
Figure 5: Representative noise-perturbed harmonic training case.

Table 2: External benchmark cases used for testing.
<table><tr><td>Case M</td></tr><tr><td>Pitch only 0.705</td></tr><tr><td>3.01</td></tr><tr><td>Plunge only 0.720 2.00 Coupled pitch-plunge 0.750 0.30</td></tr></table>

![](images/ee014d0376710c5e5024cf81e4bf656720cd2ac5bdc39d4a8eac02489176bfed.jpg)  
(a) Excitation

![](images/a9cce62cf1a98ed080390ff7c723183097395e287d74f16cfb32f13571347108.jpg)  
(b) Lift coeficient  
Figure 6: Pitch-only external benchmark case.

## 2.3 Wagner Baseline

This subsection defines the analytical aerodynamic baseline used in the residual learning framework. The Wagner function was selected because it provides a physics-based unsteady lift prediction from the prescribed pitch–plunge kinematics and free-stream conditions alone, without requiring additional CFD computations. It decomposes the unsteady lift for a thin airfoil undergoing pitch and plunge motion in incompressible flow into a noncirculatory added mass term and a circulatory term obtained from a Duhamel convolution. The Wagner indicial response equations are written in terms of the nondimensional time

$$
s = \frac { U t _ { \mathrm { p h y s } } } { b } ,\tag{3}
$$

where U is the free-stream speed and b is the airfoil semi-chord. Since the CFD histories are stored in the full-chord time coordinate of Eq. (2), the Wagner function is evaluated using

$$
s = 2 \tau .\tag{4}
$$

The CFD plunge displacement is positive upward while the Wagner formulation uses the thin-airfoil sign convention, so the CFD data plunge variable is reversed in sign before evaluating the baseline. The efective downwash at the three-quarter-chord point is

$$
\begin{array} { r } { w _ { 3 / 4 } = - \Big [ \dot { h } + U \alpha + b \big ( \frac { 1 } { 2 } - a \big ) \dot { \alpha } \Big ] , } \end{array}\tag{5}
$$

![](images/0a4f49f35c6b82d9398bd16174e25476bdc68b0cd2e079754ed7656a8d6a6d02.jpg)  
(a) Excitation

![](images/5fb43b053bc460e025f49fe95e20deb4e6637fc235601d575ff80d2346818a1f.jpg)  
(b) Lift coeficient  
Figure 7: Plunge-only external benchmark case.

![](images/3c3fadf5344bd1ae99738a9699e85336a6520382827478a1888d6491c32aa2e3.jpg)  
(a) Excitation

![](images/5285ab0fa2a53a48f5df6ee7e4a241e8cb32506367dbe010c98d01f03c3b739b.jpg)  
(b) Lift coeficient  
Figure 8: Coupled pitch–plunge external benchmark case.

where h denotes the plunge displacement and α is the pitch angle. Using the above, the lift is given by

$$
\begin{array} { r l } & { L = \underbrace { \pi \rho b ^ { 2 } \Big ( \ddot { h } + U \dot { \alpha } - b a \ddot { \alpha } \Big ) } _ { \mathrm { n o n c i r c u l a t o r y } } } \\ & { - \rho U b \underbrace { \Bigg [ 2 \pi w _ { 3 / 4 } ( s ) \phi ( 0 ) + 2 \pi \int _ { 0 } ^ { s } \frac { \mathrm { d } \phi } { \mathrm { d } \sigma } w _ { 3 / 4 } ( s - \sigma ) \mathrm { d } \sigma \Bigg ] } _ { \mathrm { c i r c u l a t o r y } } , } \end{array}\tag{6}
$$

where $\rho$ is the air density and $\phi$ is the Wagner indicial function, given by

$$
\phi ( \sigma ) = 1 - 0 . 2 0 3 e ^ { - 0 . 0 7 2 \sigma } - 0 . 2 3 6 e ^ { - 0 . 2 6 1 \sigma } - 0 . 0 6 e ^ { - 0 . 8 \sigma } ,\tag{7}
$$

The lift coeficient is normalized by the dynamic pressure and full chord,

$$
c _ { l } = \frac { L } { \frac { 1 } { 2 } \rho U ^ { 2 } ( 2 b ) } .\tag{8}
$$

In this study, three variants of this analytical baseline are defined and compared to select the most suitable physics-based baseline for residual learning: quasi-steady, linear unsteady, and nonlinear unsteady. The quasi-steady baseline keeps the noncirculatory term from Eq. (6) but replaces the circulatory convolution by the instantaneous steady value 2π $w _ { 3 / 4 }$ . The linear Wagner baseline uses Eqs. (6)–(7). The nonlinear Wagner baseline replaces $w _ { 3 / 4 }$ in the circulatory term by

$$
w _ { 3 / 4 } ^ { \mathrm { e f f } } = w _ { 3 / 4 } \left[ 1 - 0 . 1 \left( \frac { 2 \pi w _ { 3 / 4 } } { U } \right) ^ { 2 } \right] .\tag{9}
$$

All Wagner baselines are evaluated on the same τ grid and motion histories as the CFD data. The raw Wagner lift coeficient is denoted by $c _ { l , \mathrm { W } } ^ { 0 } ( t )$ , which is calculated using incompressible assumptions and does not capture the steady lift ofset caused by finite Mach number, airfoil shape, and nonzero mean angle of attack. Therefore, Eq. (10) defines a scalar static correction before the unsteady residual is calculated. For each motion case, this correction is

$$
\begin{array} { r l } & { \Delta c _ { l , \mathrm { s t a t i c } } = \mathrm { m e a n } ( c _ { l , \mathrm { C F D } } ) - \mathrm { m e a n } ( c _ { l , \mathrm { W } } ^ { 0 } ) , } \\ & { ~ c _ { l , \mathrm { W } } ( t ) = c _ { l , \mathrm { W } } ^ { 0 } ( t ) + \Delta c _ { l , \mathrm { s t a t i c } } . } \end{array}\tag{10}
$$

The term $c _ { l , \mathrm { W } } ( t )$ denotes the Wagner baseline after the static correction. This residual target is then the remaining time-dependent diference between the CFD lift and the lift obtained with the static-corrected Wagner baseline. In this study, the available CFD history is used to define the scalar ofset in Eq. (10) for each benchmark case, and the static correction was also fitted from motion and flow parameters, including Mach number, mean and amplitude of the prescribed kinematics, reduced frequency, and the mean raw Wagner lift. This fit captured the available ofsets suficiently for this study, but in future applications, $\Delta c _ { l , \mathrm { s t a t i c } }$ could be supplied by a static ROM trained from static CFD runs, allowing the unsteady residual ROM to focus only on the unsteady correction.

Table 3 and Fig. 9 compare the candidate analytical baselines on the three external benchmark cases. Table 3 reports the NRMSE values, while Fig. 9 shows the corresponding CFD and baseline time histories. The quasi-steady baseline has the largest errors because it lacks the circulatory memory and phase lag contributions. The linear and nonlinear unsteady predictions are nearly identical for these cases, indicating that the motions remain within a regime where the linear unsteady response dominates and the nonlinear amplitude correction has no significant efect. Linear unsteady Wagner is therefore selected as the physics-based baseline for all subsequent residual learning studies.

Table 3: Analytical baseline NRMSE for external benchmark cases.
<table><tr><td>Case</td><td>M</td><td>k</td><td>Quasi- steady</td><td>Linear unsteady</td><td>Nonlinear unsteady</td></tr><tr><td>Pitch only</td><td>0.705</td><td>3.01</td><td>0.964</td><td>0.658</td><td>0.655</td></tr><tr><td>Plunge only</td><td>0.720</td><td>2.00</td><td>0.377</td><td>0.240</td><td>0.260</td></tr><tr><td>Coupled pitch-plunge</td><td>0.750</td><td>0.30</td><td>0.160</td><td>0.114</td><td>0.116</td></tr><tr><td>Mean</td><td></td><td></td><td>0.500</td><td>0.337</td><td>0.344</td></tr></table>

## 3 Residual Learning Methodology

This section defines the residual-learning formulation, describes the direct model and the residual model based on the analytical baseline (linear unsteady Wagner), and specifies the training, evaluation, and feature-selection procedures used for all generalization studies.

## 3.1 Residual-Learning Model Formulation

In this study, a general residual-learning framework is proposed which decomposes the desired aerodynamic output into a physics-based baseline and one or more residual modules:

$$
F _ { \mathrm { t o t a l } } = F _ { \mathrm { b a s e l i n e } } ^ { ( f _ { 0 } ) } + \sum _ { i = 1 } ^ { N } g _ { i } \Delta F _ { i } ^ { ( f _ { i } ) } , \qquad 0 \leq g _ { i } \leq 1 .\tag{11}
$$

Here, $\Delta F _ { i } ^ { ( f _ { i } ) }$ is a learned residual correction associated with a chosen physical efect or fidelity level, and $g _ { i }$ is an optional activation or blending factor. Although Eq. (11) defines a general residuallearning framework, the present work evaluates only the single-module unsteady lift case, with one LSTM correction, fixed $g _ { 1 } = 1$ , and the linear Wagner model as the baseline for $c _ { l }$ . Broader multi-module applications will be explored in future work.

The notation below reports the model output at a generic time t, but the implemented sequences are given in the nondimensional coordinate τ defined in Eq. (2). Each LSTM operates on the input sequence history available up to the current sample. The direct model prediction is obtained from the input sequence as

$$
c _ { l , \mathrm { d i r e c t } } ( t ) = \mathrm { L S T M } _ { \mathrm { d i r e c t } } \left( \pmb { x } _ { \mathrm { d i r e c t } } ( t ) \right) ,\tag{12}
$$

with the selected direct input set

$$
\begin{array} { r } { \pmb { x } _ { \mathrm { d i r e c t } } ( t ) = \left[ \alpha ( t ) \quad \dot { \alpha } ( t ) \quad \dot { h } ( t ) \quad M \right] ^ { T } . } \end{array}\tag{13}
$$

The residual model learns the diference between the CFD lift and the Wagner baseline,

$$
\Delta c _ { l } ( t ) = c _ { l , \mathrm { C F D } } ( t ) - c _ { l , \mathrm { W } } ( t ) ,\tag{14}
$$

![](images/bb0f98f9e2fc311132267b24bdb2682161f327cc85a5c772ae143d1ae275ee26.jpg)  
(a) Pitch only

![](images/e5ee5b6b7d9c4b9ea6aaee2453bb2e3ddd0b292e9da35ec57711962ad17d8f5a.jpg)  
(b) Plunge only

![](images/fa22d4c1c919cf4141799c6fd11fea15f3fbdde59266e47ba1950a0f3aea5003.jpg)  
(c) Coupled pitch–plunge  
Figure 9: Quasi-steady, linear Wagner, and nonlinear Wagner comparison on external benchmark cases, correspond ing to NRMSE values in Table 3.

Table 4: Selected LSTM architecture and training settings.
<table><tr><td>Parameter</td><td>Value</td><td>Note</td></tr><tr><td>Architecture</td><td>1-layer LSTM</td><td>Consistent with the prior NLR 7301 LSTM setup [15].</td></tr><tr><td>Hidden size</td><td>250</td><td>Larger hidden size did not improve the priority cases.</td></tr><tr><td>Optimizer</td><td>Adam</td><td>Learning rate  $1 0 ^ { - 3 }$ </td></tr><tr><td></td><td></td><td>Epoch handling 1000, restore-best Selected using validation reconstructed-cl RMSE.</td></tr><tr><td>Loss</td><td>MSE</td><td>Applied to the selected training target for each model.</td></tr></table>

where this residual is the training target for the residual LSTM. The residual LSTM output is

$$
\Delta c _ { l , \mathrm { p r e d } } ( t ) = \mathrm { L S T M } _ { \mathrm { r e s } } \left( \pmb { x } _ { \mathrm { r e s } } ( t ) \right) .\tag{15}
$$

The final residual-model lift prediction is obtained by adding the predicted residual correction to the Wagner baseline,

$$
c _ { l , \mathrm { r e s } } ( t ) = c _ { l , \mathrm { W } } ( t ) + \Delta c _ { l , \mathrm { p r e d } } ( t ) .\tag{16}
$$

The selected residual input set uses derivative and acceleration features along with Mach number,

$$
\begin{array} { r } { \pmb { x } _ { \mathrm { r e s } } ( t ) = \left[ \dot { h } ( t ) \quad \ddot { h } ( t ) \quad \alpha ( t ) \quad \dot { \alpha } ( t ) \quad \ddot { \alpha } ( t ) \quad \ddot { \alpha } ( t ) \quad M \right] ^ { T } . } \end{array}\tag{17}
$$

The feature and normalization checks used to select these direct and residual input sets are discussed in Sec. 3.3.

## 3.2 Training Procedure and Error Metrics

Both direct and residual models use the same LSTM architecture. The network contains one LSTM layer with sequence output, a fully connected layer, and a regression output layer. The selected architecture and training settings are summarized in Table 4. All training, validation, and testing are performed on the original τ-indexed solver samples, without conversion to physical time or interpolation to a common time step. All splits are made by complete motion case rather than by time sample, and the external benchmark cases are never used for training or validation. For residual models, reported errors are always computed after reconstructing $c _ { l }$ using Eq. (16) and the best epoch is selected with the lowest validation reconstructed- $- c _ { l }$ RMSE. Since LSTM training can depend on random initialization, direct and residual models are trained with five independent random seeds for all the studies in this paper.

Two error metrics are used to evaluate the reconstructed lift histories. Let $N _ { e }$ be the number of samples in the evaluation window, and let $c _ { l , \mathrm { p r e d } }$ denote either $c _ { l , \mathrm { d i r e c t } }$ or $c _ { l , \mathrm { r e s } } .$ . The root-meansquare error is

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { N _ { e } } \sum _ { n = 1 } ^ { N _ { e } } \left( c _ { l , \mathrm { C F D } } ( t _ { n } ) - c _ { l , \mathrm { p r e d } } ( t _ { n } ) \right) ^ { 2 } } ,\tag{18}
$$

and the normalized RMSE is computed using the CFD lift range over the same samples,

$$
\mathrm { N R M S E } = \frac { \mathrm { R M S E } } { \operatorname* { m a x } ( c _ { l , \mathrm { C F D } } ) - \operatorname* { m i n } ( c _ { l , \mathrm { C F D } } ) } .\tag{19}
$$

Table 5: Residual-model external-benchmark NRMSE for selected feature-set and target-normalization cases.
<table><tr><td>Feature set</td><td>Normalization</td><td>Pitch-only</td><td>Plunge-only</td><td>Coupled</td><td>Mean</td></tr><tr><td> $\alpha , \dot { \alpha } , \dot { h } , M$ </td><td>X-only</td><td>0.124</td><td>0.035</td><td>0.091</td><td>0.083</td></tr><tr><td> ${ \dot { h } } , { \ddot { h } } , \alpha , { \dot { \alpha } } , { \ddot { \alpha } }$ </td><td>X-only</td><td>0.063</td><td>0.043</td><td>0.089</td><td>0.065</td></tr><tr><td> $\dot { h } , \ddot { h } , \alpha , \dot { \alpha } , \ddot { \alpha } , M$ </td><td>X/Y</td><td>0.078</td><td>0.028</td><td>0.075</td><td>0.060</td></tr><tr><td> ${ \dot { h } } , { \ddot { h } } , \alpha , \dot { \alpha } , \ddot { \alpha } , M$ </td><td>X-only</td><td>0.058</td><td>0.036</td><td>0.039</td><td>0.044</td></tr><tr><td> $\alpha , h , \dot { \alpha } , \dot { h } , \ddot { \alpha } , \ddot { h } , M$ </td><td>X-only</td><td>0.066</td><td>0.030</td><td>0.128</td><td>0.075</td></tr></table>

Table 6: Direct-model external-benchmark NRMSE for selected feature sets.

<table><tr><td>Feature set</td><td>Pitch-only</td><td>Plunge-only</td><td>Coupled</td><td>Mean</td></tr><tr><td> $\alpha , h , \dot { \alpha } , \dot { h }$ </td><td>0.070</td><td>0.042</td><td>0.122</td><td>0.078</td></tr><tr><td> $\alpha , \dot { \alpha } , \dot { h } , M$ </td><td>0.041</td><td>0.049</td><td>0.072</td><td>0.054</td></tr><tr><td> $\alpha , h , \dot { \alpha } , h , M$ </td><td>0.063</td><td>0.021</td><td>0.106</td><td>0.063</td></tr><tr><td> $\dot { h } , \ddot { h } , \alpha , \dot { \alpha } , \ddot { \alpha } , M$ </td><td>0.123</td><td>0.026</td><td>0.164</td><td>0.104</td></tr><tr><td> $\alpha , h , \dot { \alpha } , \dot { h } , \ddot { \alpha } , \ddot { h } , M$ </td><td>0.050</td><td>0.021</td><td>0.105</td><td>0.059</td></tr></table>

In Eq. (19), the maximum and minimum are taken over the same evaluation window used in Eq. (18). Unless otherwise stated, the evaluation window is the final 60% of each time history so that zero-state startup transients do not dominate the comparison.

## 3.3 Feature Selection

The feature and normalization study identifies the input representation and target scaling used for the final direct and residual models. This check is needed because the direct model learns the full lift coeficient, while the residual model learns the remaining correction after subtracting the Wagner baseline. These two diferent targets can favor diferent kinematic variables and scaling choices. The entries in Tables 5 and 6 are NRMSE values on the three external benchmark cases, and the final column is the mean across those cases. In Table 5, X-only denotes input normalization only, while $\mathrm { X } / \mathrm { Y }$ denotes normalization of both the inputs and the residual target. Table 5 shows that the best residual model uses the derivative and acceleration feature set $[ \dot { h } , \ddot { h } , \alpha , \dot { \alpha } , \ddot { \alpha } , M ] ^ { T }$ with input-only normalization. Adding Mach to the derivative and acceleration features improves the result across all cases, and not normalizing the residual target reduces the mean NRMSE from 0.060 to 0.044 for the same feature set. Table 6 reports the corresponding feature study for the direct model. The direct model performs best with the compact rate-based input set $[ \alpha , \dot { \alpha } , \dot { h } , M ] ^ { T }$ and a normalized output. Adding second-derivative features raises the mean NRMSE from 0.054 to 0.104. This indicates that the direct and residual models benefit from diferent representations. The residual model benefits from variables aligned with the Wagner function, while the direct model is more sensitive to noisy acceleration inputs. Similarly, the selected direct model benefits from using normalization for total $c _ { l } ,$ whereas the residual model benefits by normalizing the input only since it preserves the physical scale of $c _ { l , \mathrm { C F D } } - c _ { l , \mathrm { W } }$

Table 7: External-test NRMSE over five random seeds.
<table><tr><td>Model</td><td>Pitch-only</td><td>Plunge-only</td><td>Coupled</td><td>Mean</td></tr><tr><td>Direct</td><td> $0 . 0 5 2 \pm 0 . 0 2 0$ </td><td> $0 . 0 5 6 \pm 0 . 0 1 1$ </td><td> $0 . 0 8 6 \pm 0 . 0 2 4$ </td><td> $0 . 0 6 5 \pm 0 . 0 0 9$ </td></tr><tr><td>Residual</td><td> $0 . 0 7 0 \pm 0 . 0 1 6$ </td><td> $0 . 0 2 6 \pm 0 . 0 1 0$ </td><td> $0 . 0 6 6 \pm 0 . 0 1 0$ </td><td> $0 . 0 5 4 \pm 0 . 0 0 7$ </td></tr></table>

![](images/f96d93dccb115587666e3475004396bdefadeee838f079a8fe930f2322c098ba.jpg)  
Figure 10: External-test NRMSE comparison across random seeds.

## 4 Results

This section presents the external benchmark and generalization results. It first evaluates the direct and residual models on the external benchmark cases and then examines leave-one-out and leave-family-out generalization across the four motion families.

## 4.1 External Benchmark Performance

The external benchmark study evaluates the selected direct and residual models on three test cases that are not used during training or validation. A random-seed sensitivity analysis is used to check whether the comparison is stable across diferent random initializations. Table 7 reports the mean and standard deviation of the NRMSE over five random seeds for each model and case. The residual model gives a lower mean error over the three cases, with $0 . 0 5 4 \pm 0 . 0 0 7$ compared with $0 . 0 6 5 { \pm } 0 . 0 0 9$ for the direct model. However, it is also important to note that the case-wise behavior is more informative than the mean values. The direct model remains better for the pitch-only case, while the residual model is better for the plunge-only and coupled pitch-plunge cases.

Figure 10 shows the same external benchmark errors across the five random seeds. The bars show means over the five random seeds and the dots show individual random-seed results. The residual model has smaller scatter across random seeds on the plunge-only and coupled cases. It also shows that the pitch-only case is where the direct model remains preferable because, for this external pure-pitch case, the Wagner baseline does not reduce the phase and amplitude mismatch enough to make the residual target easier than direct lift prediction.

The time-history comparisons provide more insight into the diferent performance of the direct and residual models across the three cases. The time histories are shown for the median seed in each case, but similar trends are observed across all seeds. Figure 11 compares the reconstructed lift, residual correction, and pointwise prediction error for the pitch-only external benchmark case. The time-history comparison shows that both models follow the lift oscillation, but the direct mode gives a smaller NRMSE since the residual model needs to learn both the phase and amplitude diference between the baseline and the CFD. Figure 12 shows the plunge-only external benchmark case. The reconstructed lift histories show closer agreement for the residual model. This is the regime where the Wagner baseline removes a useful part of the memory and phase behavior, leaving the LSTM to learn a smaller correction. Next, Fig. 13 shows the coupled pitch–plunge external benchmark case. The residual prediction follows the CFD lift more closely than the direct model, and the pointwise error is smaller over most of the evaluation window. This result suggests that the Wagner baseline provides useful phase and memory structure for the low-frequency coupled motion, even though the baseline alone does not fully match the CFD response. Overall, both models are less accurate than in the plunge-only case. This is likely because the coupled case has a longer duration and a coarser nondimensional time-step, making the LSTM evaluation less accurate for this case.

![](images/d5fd07365238a7121acd6f51994a7e940de5006d38cf19c38d1a52622f6c5fa6.jpg)  
(a) Lift coeficient

![](images/74edb8e193d45c1b6b6279a4c5a94e9870b500d6c4c5a964b4d0da632863e686.jpg)  
(b) Residual correction

![](images/e03cfd987df33b6cd325a01ed1a02e1def7dc925a1f0d51d2947a593b13eb10b.jpg)  
(c) Prediction error  
Figure 11: Pitch-only external benchmark test case comparison.

![](images/ed1c44fbd8d690e6d86f605ecf19787fe5a8ccfb62b4be4d16b487b3c60edb8f.jpg)  
(a) Lift coeficient

![](images/d81878198cbe57a2f40bbf7a07ee87c6a9d503a901372183ce091efcbbde5db2.jpg)  
(b) Residual correction

![](images/ea8e1f19ba591735dba62ca8dc3d1dcf9207655cef4d70f81da08c24924dfeb4.jpg)  
(c) Prediction error  
Figure 12: Plunge-only external benchmark test case comparison.

![](images/01c3eb3d54570f53f6e428e53e8edf38c0ad9a8e4744f1d21b01ce34899a951c.jpg)  
(a) Lift coeficient

![](images/42f19077a7a9c15d0f5570aa1b01462ddaeb1dab0d6846b2c97be6c956c5668e.jpg)  
(b) Residual correction

![](images/767bc21d65dceb196260ee2fc8ceb024e47a0ff46e9419fd384b10e710a30f5e.jpg)  
(c) Prediction error  
Figure 13: Coupled pitch–plunge external benchmark test case comparison.

![](images/f1618510ef8cf0b52cee31184a0763f0afe98596087a7ca6491198a1b5c05a0d.jpg)

Figure 14: Single-harmonic leave-one-out NRMSE by holdout case.  
![](images/e565d10edfb9eca750756f82f08700fec1ece8cc2d848c5e6717ba1de9635801.jpg)  
Figure 15: Amplitude-modulated leave-one-out NRMSE by holdout case.

## 4.2 Leave-One-Out Generalization

The leave-one-out study removes one case at a time from a motion family and trains on the remaining cases. Test cases from the same family remain in the training set, so this study measures in-family interpolation to an unseen motion input. A random-seed sensitivity analysis was also conducted for this generalization study. Since the trends were similar across seeds, the results in this section are shown for one representative seed case.

Figure 14 shows the single-harmonic leave-one-out NRMSE by holdout case. These results help identify how the direct and residual models behave as reduced frequency and Mach number vary across the single-harmonic family. The residual model gives lower NRMSE for most of the singleharmonic holdouts. The largest gains occur mostly at higher reduced frequency and higher Mach number within the single-harmonic leave-one-out setting, where nearby training cases allow the LSTM to refine the phase structure supplied by the Wagner baseline. The direct model is more competitive for lower-frequency cases and outperforms the residual model. Next, Fig. 15 shows the amplitude-modulated harmonic leave-one-out results where the direct and residual errors are compared for each amplitude-modulated holdout case. The residual model has the lower family mean, although the direct model remains competitive at the lowest Mach number. This indicates that the residual formulation improves prediction for nonstationary harmonic inputs as well, but sparse Mach coverage within this family impacts the individual fold results. Gaussian-pulse leaveone-out results are shown in Fig. 16; these results evaluate whether the residual advantage remains for transient inputs as well. The residual model gives lower errors for all Gaussian-pulse holdouts in this representative run. This suggests that the Wagner baseline still captures a useful part of the attached-flow response and leaves the LSTM a smaller correction to learn, even when the input is a short transient pulse rather than a sustained oscillation. Next, Fig. 17 shows the noise-perturbed harmonic leave-one-out results. The figure separates the two plunge-sign variants and compares the Mach-number trend within each variant. The residual model gives lower NRMSE for every holdout in this family. This result indicates that the residual learning is not limited to unperturbed sinusoidal inputs and remains efective when the input contains broadband perturbations.

![](images/bcd06d7b4a9e0accdf0eb837f97a17399da621451901f38b4f30086294731891.jpg)

Figure 16: Gaussian-pulse leave-one-out NRMSE by holdout case.  
![](images/4fd9b87e51159134ccf806074a99e94e2d093078746fc290860f21a237776432.jpg)  
Figure 17: Noise-perturbed leave-one-out NRMSE by holdout case.

The results for the leave-one-out are summarized in Table 8 by motion family. The table reports the family-mean NRMSE and the number of holdouts for which the residual model is better than the direct model. The residual model has a lower mean for all four families. The amplitude-modulated family has fewer residual-favorable holdouts, but its residual-model mean remains lower. The table therefore supports the conclusion that the residual advantage extends beyond the sinusoidal training family. Overall, across all four motion families, the residual model lowered the familymean NRMSE, confirming that the linear unsteady Wagner baseline supplied a useful attached-flow response and left the LSTM a smaller correction to learn for in-family interpolation. The direct model remained competitive only at the lowest reduced frequencies and Mach numbers, where the Wagner baseline ofered less benefit.

Table 8: Leave-one-out generalization summary by motion family.
<table><tr><td>Motion family</td><td></td><td>Direct mean Residual mean Residual lower</td><td></td></tr><tr><td>Single harmonic</td><td>0.057</td><td>0.045</td><td>19/24</td></tr><tr><td>Amplitude-modulated harmonic</td><td>0.028</td><td>0.022</td><td>3/5</td></tr><tr><td>Gaussian pulse</td><td>0.018</td><td>0.010</td><td>5/5</td></tr><tr><td>Noise-perturbed harmonic</td><td>0.027</td><td>0.015</td><td>10/10</td></tr></table>

## 4.3 Leave-Family-Out Generalization

The final generalization study is the leave-family-out study, where an entire motion family is removed from both training and validation. This is a more challenging test than leave-one-out because the trained model has not seen any examples from the withheld family. The family holdout cases are used to evaluate generalization across motion type, while the external benchmark cases are evaluated with the same trained model. A random-seed sensitivity analysis was also conducted for this study. Similar to Sec. 4.2, the plots in this section are shown for one representative randomseed case. Since the Gaussian-pulse trend was not consistent across random-seed runs, Table 9 also compares the best and worst runs, defined by the residual improvement in family-mean NRMSE.

Figure 18 shows the single-harmonic leave-family-out result. These results show that removing all single-harmonic cases from training mainly afects the direct model. The residual-model mean remains close to its leave-one-out value, indicating that the Wagner baseline supplies a transferable sinusoidal response structure while the direct model must learn the sinusoidal behavior only from non-sinusoidal examples. Next, Fig. 19 shows the amplitude-modulated harmonic leave-family-out result. For this motion family as well, the residual model has a lower error for every holdout case. This indicates that the residual formulation remains useful for slowly varying harmonic amplitudes, even when the training set contains no amplitude-modulated examples. Next, Fig. 20 shows the noise-perturbed harmonic leave-family-out result separated by the positive- and negative-plunge testing cases. The residual model gives lower NRMSE for all noise-perturbed holdouts as well. This indicates that residual learning is not limited to clean sinusoidal inputs and remains efective when the withheld family contains broadband perturbations.

Finally, Fig. 21 shows the Gaussian-pulse leave-family-out result. In the representative run shown here, the residual model has lower NRMSE for the Gaussian-pulse holdouts. However, the random-seed sensitivity analysis comparison in Table 9 shows that this is the most sensitive leavefamily-out case, and the model behavior can change across random-seed runs. Table 9 summarizes the leave-family-out results using the best and worst random-seed runs for each motion family. Best and worst are based on the reduction in family-mean NRMSE from the direct model to the residual model. The final column reports the number of family holdout cases, out of the family total, for which the residual model has the lower NRMSE. The table shows that the residual model remains better in family-mean NRMSE for the single-harmonic, amplitude-modulated, and noiseperturbed families even in the worst runs. The Gaussian-pulse family is the exception. The best run gives lower residual-model error for all holdouts, while the worst run favors the direct model for all holdouts. This indicates that the residual advantage is strongest when the Wagner baseline removes a transferable part of the response, but it can weaken for transient cases where the withheld motion difers more strongly from the remaining training families.

![](images/5f57de54008ba00115222a16527a5cb32d36540ec4c41139d6f7fda55c3658cf.jpg)  
(a) Family holdout cases

![](images/34dcb91b4bf51d2e140255a06b5602db5e78dd3b74354fd76774d2a203ac689b.jpg)  
(b) External test cases  
Figure 18: Single-harmonic leave-family-out NRMSE.

## 5 Concluding Remarks

This paper evaluated a residual learning framework for unsteady lift prediction on the NLR 7301 airfoil in the transonic flow regime in the presence of shock motion. The residual framework uses the Wagner indicial function as an analytical baseline function. The study compared a direct LSTM trained on total CFD lift with a residual LSTM trained on the diference between the CFD lift and a mean-aligned Wagner prediction. The comparison used external benchmark tests, leave-one-out studies, and leave-family-out studies across single-harmonic, amplitude-modulated harmonic, Gaussian-pulse, and noise-perturbed harmonic motion families.

The baseline comparison and feature-selection study show that the direct and residual models favor diferent inputs and scaling choices. The linear Wagner model was selected because it performed nearly the same as the nonlinear Wagner variant and better than the quasi-steady baseline for the external benchmark cases. The residual model performed best with derivative and acceleration features and input-only normalization. The direct model performed best with a compact rate-based feature set and a normalized total-lift target. This diference indicates that the residual model is not simply a direct model with a redefined target. It is a lower-variance correction problem with its own useful feature representation. With these selected direct and residual models, the external benchmark results show that the residual model gives a lower mean NRMSE over five random seeds and smaller average scatter across random seeds than the direct model. However, the improvement is case dependent since the residual model performed better for the plunge-only and coupled pitch-plunge cases, where the Wagner gives a useful amplitude and phase baseline.

![](images/6807dab659fa27633dee8a0d25a09e742a23266fadf83c04dd6e96d6f94dfee2.jpg)  
Family holdout case (varying Mach number)

(a) Family holdout cases  
![](images/24f4906552861ae428c153d5a73b56fbcd070afa983e622c12fcd6570d7afbec.jpg)  
(b) External test cases  
Figure 19: Amplitude-modulated leave-family-out NRMSE.

Table 9: Leave-family-out generalization summary for random-seed runs.
<table><tr><td>Motion family</td><td>Run</td><td>Direct mean Residual mean Residual lower</td><td></td></tr><tr><td rowspan="2">Single harmonic</td><td>Best</td><td>0.091</td><td>0.046 19/24</td></tr><tr><td>Worst</td><td>0.085</td><td>0.046 19/24</td></tr><tr><td rowspan="2">Amplitude-modulated harmonic</td><td>Best</td><td>0.045</td><td>0.032 6/6 6/6</td></tr><tr><td>Worst</td><td>0.040</td><td>0.032</td></tr><tr><td rowspan="2">Gaussian pulse</td><td>Best</td><td>0.027</td><td>0.019 6/6</td></tr><tr><td>Worst</td><td>0.019</td><td>0.024 0/6</td></tr><tr><td rowspan="2">Noise-perturbed harmonic</td><td>Best</td><td>0.042</td><td>0.022 12/12</td></tr><tr><td>Worst</td><td>0.040</td><td>0.025 12/12</td></tr></table>

![](images/8fd0ea3405088878fb23736e21f527eab5e562ca15f6e8f9f1e222e58bd05703.jpg)  
(a) Family holdout cases

![](images/dd26983d08ba4bf0097482dd8da70d4b65e3c6f876dc9113b69fa705d96558a9.jpg)  
(b) External test cases  
Figure 20: Noise-perturbed leave-family-out NRMSE.

![](images/9b6a57ecd9e46611b19b463165a44879b2f0f487a9ae8d04ed1e4e0de2d1f9ef.jpg)  
(a) Family holdout cases

![](images/c730296164a1f2ab60cfff78e180bd2a7ab803492356438b8924c26cb318353a.jpg)  
(b) External test cases  
Figure 21: Gaussian-pulse leave-family-out NRMSE.

The direct model remains better for the high-frequency pitch-only case, where the Wagner baseline does not simplify the residual target enough to improve the final reconstructed lift prediction.

The leave-one-out results show that the residual advantage is not limited to sinusoidal inputs. The residual model improves the family-mean NRMSE for all four motion families and gives lower NRMSE for most holdouts in the single-harmonic, Gaussian-pulse, and noise-perturbed families. The leave-family-out results provide a stronger generalization test because an entire motion family is removed from training. In these tests, the residual model is nearly unchanged when all singleharmonic cases are removed, while the direct model degrades substantially. The residual model also generalizes well to the amplitude-modulated and noise-perturbed family holdouts. The Gaussianpulse family is the exception and the most sensitive case, suggesting that the residual advantage depends on how well the remaining training families represent the withheld transient response.

Overall, the results support residual learning as a useful way to augment classical unsteady aerodynamic models with data-driven corrections. The approach is most efective when the physics baseline removes a structured part of the aerodynamic response and leaves the neural network to learn a smaller unsteady correction. Future work should examine Mach-number interpolation and extrapolation more systematically, replace the present scalar static correction with a static ROM, and evaluate alternative baselines such as compressibility-corrected Wagner and other lowfidelity unsteady solvers. The framework should also be evaluated for other aerodynamic coeficients and additional test cases, including three-dimensional wings and configurations beyond the twodimensional airfoil study.

## References

[1] Theodorsen, T. (1949). General Theory of Aerodynamic Instability and the Mechanism of Flutter. Tech. Rep. TR-496, NACA.

[2] Bisplinghof, R. L. and Ashley, H. (1962). Principles of Aeroelasticity. New York: Wiley & sons.

[3] Faller, W. E. and Schreck, S. J. (1995). Real-Time Prediction of Unsteady Aerodynamics: Application for Aircraft Control and Manoeuvrability Enhancement. IEEE Transactions on Neural Networks, 6(6), 1461–1468.

[4] Suresh, S., Omkar, S. N., Mani, V., et al. (2003). Lift Coeficient Prediction at High Angle of Attack Using Recurrent Neural Network. Aerospace Science and Technology, 7(8), 595–602.

[5] Zhang, W., Wang, B., Ye, Z., et al. (2012). Eficient Method for Limit Cycle Flutter Analysis Based on Nonlinear Aerodynamic Reduced-Order Models. AIAA Journal, 50(5), 1019–1028.

[6] Winter, M. and Breitsamter, C. (2014). Reduced-Order Modeling of Unsteady Aerodynamic Loads Using Radial Basis Function Neural Networks. In Deutscher Luft- und Raumfahrtkongress. Augsburg, Germany.

[7] Mannarino, A. and Mantegazza, P. (2014). Nonlinear Aeroelastic Reduced Order Modeling by Recurrent Neural Networks. Journal of Fluids and Structures, 48, 103–121.

[8] Ignatyev, D. I. and Khrabrov, A. N. (2015). Neural Network Modeling of Unsteady Aerodynamic Characteristics at High Angles of Attack. Aerospace Science and Technology, 41, 106–115.

[9] Kou, J. and Zhang, W. (2017). Multi-Kernel Neural Networks for Nonlinear Unsteady Aerodynamic Reduced-Order Modeling. Aerospace Science and Technology, 67, 309–326.

[10] Winter, M. and Breitsamter, C. (2018). Nonlinear Identification via Connected Neural Networks for Unsteady Aerodynamic Analysis. Aerospace Science and Technology, 77, 802–818.

[11] Kou, J. and Zhang, W. (2021). Data-Driven Modeling for Unsteady Aerodynamics and Aeroelasticity. Progress in Aerospace Sciences, 125, 100725.

[12] Park, K. H., Jun, S. O., Baek, S. M., et al. (2013). Reduced-Order Model with an Artificial Neural Network for Aerostructural Design Optimization. Journal of Aircraft, 50(4), 1106–1116.

[13] Lindhorst, K., Haupt, M. C., and Horst, P. (2014). Eficient Surrogate Modelling of Nonlinear Aerodynamics in Aerostructural Coupling Schemes. AIAA Journal, 52(9), 1952–1966.

[14] Winter, M. and Breitsamter, C. (2016). Eficient Unsteady Aerodynamic Loads Prediction Based on Nonlinear System Identification and Proper Orthogonal Decomposition. Journal of Fluids and Structures, 67, 1–21.

[15] Wang, Q., Cesnik, C. E. S., and Fidkowski, K. J. (2020). Multivariate Recurrent Neural Network Models for Scalar and Distribution Predictions in Unsteady Aerodynamics. In AIAA SciTech 2020 Forum. Orlando, FL.

[16] Bengio, Y., Simard, P., and Frasconi, P. (1994). Learning Long-Term Dependencies with Gradient Descent Is Dificult. IEEE Transactions on Neural Networks, 5(2), 157–166.

[17] Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780.

[18] Wang, Q., Medeiros, R. R., Cesnik, C. E. S., et al. (2019). Techniques for Improving Neural Network-Based Aerodynamics Reduced-Order Models. In AIAA SciTech 2019 Forum.

[19] Kou, J. and Zhang, W. (2017). Layered Reduced-Order Models for Nonlinear Aerodynamics and Aeroelasticity. Journal of Fluids and Structures, 68, 174–193.

[20] Kou, J. and Zhang, W. (2019). Multi-Fidelity Modeling Framework for Nonlinear Unsteady Aerodynamics of Airfoils. Applied Mathematical Modelling, 76, 832–855. doi:10.1016/j.apm. 2019.06.034.

[21] He, Y. and Schumann, J. (2020). A Framework for the Analysis of Deep Neural Networks in Autonomous Aerospace Applications Using Bayesian Statistics. In Workshop on Assured Autonomous Systems.

[22] He, L., Qian, W., Zhao, T., et al. (2020). Multi-Fidelity Aerodynamic Data Fusion with a Deep Neural Network Modeling Method. Entropy, 22(9), 1022.

[23] Zhang, X., Xie, F., Ji, T., et al. (2021). Multi-Fidelity Deep Neural Network Surrogate Model for Aerodynamic Shape Optimization. Computer Methods in Applied Mechanics and Engineering, 373, 113485.

[24] Lee, D. H., Lee, D., Han, S., et al. (2023). Deep Residual Neural Network for Predicting Aerodynamic Coeficient Changes with Ablation. Aerospace Science and Technology, 136, 108207.

[25] Demo, N., Tezzele, M., and Rozza, G. (2023). A DeepONet Multi-Fidelity Approach for Residual Learning in Reduced Order Modeling. Advanced Modeling and Simulation in Engineering Sciences, 10(1), 12.

[26] Yan, Q., Wan, Z., and Yang, C. (2023). Flight Load Calculation Using Neural Network Residual Kriging. Aerospace, 10(7), 599.

[27] Feldwisch, J. M. (2025). Increased-Order Model for Unsteady Aerodynamic Nonlinearities Using Neural Networks. Journal of Aircraft. doi:10.2514/1.C038330.

[28] Dong, Z., Liu, B., Da, X., et al. (2025). Control-Oriented Six-Degree-of-Freedom Modeling of Distributed Propulsion Vehicle. The Aeronautical Journal, 129(1339), 2460–2479.

[29] Fidkowski, K. J. and Roe, P. L. (2010). An Entropy Adjoint Approach to Mesh Refinement. SIAM Journal on Scientific Computing, 32(3), 1261–1287. doi:10.1137/090759057.

## Copyright statement

The authors confirm that they, and/or their company or organisation, hold copyright on all of the original material included in this paper. The authors also confirm that they have obtained permission from the copyright holder of any third-party material included in this paper to publish it as part of their paper. The authors confirm that they give permission, or have obtained permission from the copyright holder of this paper, for the publication and public distribution of this paper as part of the IFASD 2026 proceedings or as individual of-prints from the proceedings.