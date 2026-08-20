# PROCESS OPTIMIZATION UNDER UNCERTAINTY FOR IMPROVING THE BOND QUALITY OF POLYMER FILAMENTS IN FUSED FILAMENT FABRICATION

Berkcan Kapusuzoglu<sup>†,1</sup>, Matthew Sato<sup>1</sup>, Sankaran Mahadevan<sup>1</sup>, Paul Witherell<sup>2</sup>

<sup>1</sup>Department of Civil and Environmental Engineering, Vanderbilt University, Nashville, TN 37235, USA

<sup>2</sup>Systems Integration Division, Engineering Laboratory,

National Institute of Standards and Technology (NIST), Gaithersburg, MD 20899, USA

## ABSTRACT

This paper develops a computational framework to optimize the process parameters such that the bond quality between extruded polymer filaments is maximized in fused filament fabrication (FFF). A transient heat transfer analysis providing an estimate of the temperature profile of the filaments is coupled with a sintering neck growth model to assess the bond quality that occurs at the interfaces between adjacent filaments. Predicting the variability in the FFF process is essential for achieving proactive quality control of the manufactured part; however, the models used to predict the variability are affected by assumptions and approximations. This paper systematically quantifies the uncertainty in the bond quality model prediction due to various sources of uncertainty, both aleatory and epistemic, and includes the uncertainty and the model discrepancy in the process parameter optimization. Variance-based sensitivity analysis based on Sobol’ indices is used to quantify the relative contributions of the different uncertainty sources to the uncertainty in the bond quality. A Gaussian process (GP) surrogate model is constructed to compute and include the model discrepancy within the optimization. Physical experiments are conducted for calibration and validation of the physics model, and also for validation of the optimum solution. The results show that the proposed formulation for process parameter optimization under uncertainty results in high bond quality between adjoining filaments of the FFF product.

Keywords additive manufacturing · fused filament fabrication · process optimization · uncertainty quantification · heat transfer model · sintering model · model uncertainty · surrogate modeling

## 1 Introduction

Fused filament fabrication (FFF), an extrusion-based deposition technique, is a widely used additive manufacturing (AM) process. Among other rapid prototyping technologies, FFF is popular due to its low cost, easy operation, and suitability for complex geometries. FFF is the process of joining materials, usually layer upon layer, by extruding a molten filament through a heated nozzle at a controlled rate and a gantry, which moves in the horizontal plane in a predefined pattern onto a build plate or onto other filaments that move in the vertical direction. The build plate supporting the extruded polymer filaments is typically set at a controlled temperature.

The material cools down, solidifies and bonds with the surrounding filaments as it is deposited. The bond formation process in FFF, as illustrated in Fig. 1 is mainly affected by the thermal energy of the extruded material. The bond quality is mainly driven by wetting, also known as the neck growth phenomenon, molecular diffusion, and randomization when the interface temperature is above the critical sintering temperature (Fig. 1). The neck growth between adjacent filaments within a layer may be termed as intra-layer bonding, and the similar neck growth evolution that occurs between two successive layers may be called as inter-layer bonding. As an additional measure, inter-layer adhesion of the printed parts can be used to analyze the quality of inter-layer bonding. There are some studies using fracture-mechanics-based methods, which characterize the fracture resistance and the inter-layer adhesion of FFF 3D printed materials [1].

The bond quality between adjacent filaments and layers strongly affects the mechanical properties of FFF-produced parts. The correlation between better neck radius and overall part strength has been studied by Sun et al. [2] where it is shown that that better bonding between filaments results in greater mechanical strength. The temperature history at the interfaces between filaments has a direct impact on the bond quality and plays an important role in predicting FFF specimen strength since it affects the neck growth, molecular diffusion and thermal stresses. Thus, it is important to determine the evolution of temperature in the interfaces between filaments in order to predict the bond formation, and therefore the mechanical properties of the manufactured part. Yardimci et al. [3] and Yardimci and Güçeri [4] presented numerical heat transfer models for fused filament fabrication of ceramics. They modeled the cooling process of a filament due to convection with the environment and compared different build patterns without taking contacts with adjoining filaments into account. Thomas and Rodríguez [5] modeled the fracture strength of the FFF part by developing a transient 2D analysis using a finite element method. They assumed the cross section of the filaments to be rectangular and neglected contact resistances. Li et al. [6] proposed a 1D analytical transient heat transfer model coupled with a lumped capacity method, considering elliptical filaments with a semi-infinite filament length, which means that temperature is uniform across the cross-section, and varies along the length of the filaments. Sun et al. [2] compared the heat transfer models developed by Li et al. [6] and Thomas and Rodríguez [5] to the temperature of the bottom-most filament using thermocouples. They found that the model proposed by Li et al. [6] underestimated conduction, and the model developed by Thomas and Rodríguez [5] underestimated convection. They assessed the bond quality both experimentally and using a sintering neck growth model, and concluded that the sintering phenomenon had a significant effect on bond strength development only for a short time since the temperature profile of the filaments remained below the critical sintering temperature after a very short time. Costa et al. [7] analyzed the mechanisms that had the largest effect on the cooling process of filaments. They found convection with the environment, and conduction with adjacent filaments were the most significant mechanisms. Costa et al. [8] proposed a transient heat transfer model that considers the physical contacts between a filament and its adjacent filaments or the build plate. Current models in the literature do not include the humidity in the environment, the effect of filament age and percentage of humidity in the filament.

![](images/98ec4177cddb059de3665f1f36df2f4a1245514a526517ee9ba9c95c8175942e.jpg)  
(1)(

![](images/f0529aa8bf529adbd1fd1e93da92f0d623faa51f6d8fa3d873e7e8eee7c13b41.jpg)  
(2) ))

![](images/38cbaa5f00b14af797ea8839bf77bddbeabbae8abba7f6d0262a71423569214e.jpg)  
(3)  
Figure 1: The bond formation process between two filaments: (1) initial surface contact; (2) wetting or neck growth; (3) molecular diffusion and randomization across the cross-section of two FFF extruded filaments

The sintering process, describing the neck growth between filaments (stage 2 in Fig. 1), is defined as the coalescence of particles under the action of surface tension [9]. A Newtonian sintering model for polymers was initially developed by Frenkel et al. [10] to predict the rate of polymer sintering. Pokluda et al. [9] developed a closed-form equation to predict the neck growth between two spherical particles based on the work balance of viscous dissipation and surface tension. Bellehumeur et al. [11] applied the model proposed by Pokluda et al. [9] to FFF for predicting the sintering between adjacent filaments as a nonlinear function of time, temperature-dependent material properties, and an initial spherical particle radius. Gurrala et al. [12] extended the study done by Bellehumeur et al. [11] by considering cylindrical filamen geometry rather spherical particles.

The models used to predict the bond quality are affected by various sources of uncertainty and error. This affects process optimization and process control decisions. Therefore, it is critical to identify and quantify the effects of the uncertainty and error sources in order to improve the overall quality of FFF products. Many previous studies on process optimization in AM have employed a design of experiments-based trial-and-error approach to determine the effect of different process parameter settings on the quantity of interest such as part strength, residual stress, surface roughness etc. [2, 13, 14, 15]. The experiment-based approach is problem-specific, and cannot be used as a general-purpose process design method for different materials and part geometries. Therefore, there is a need to use a physics model-based approach that optimizes the process parameters for better overall bond quality without running multiple economically expensive physical experiments, and also includes the uncertainty in the neck growth prediction.

The focus of this work is to develop a framework for optimizing process parameters under uncertainty that maximizes the filament bond quality of FFF printed parts. The overall bond length is the decision criterion we use for optimization;

greater overall bond lengths signify lesser void area (thus better bonding) in an average sense. We only consider stage 1 and 2 shown in Fig. 1 to predict the overall bond length using the sintering neck growth model shown in Section 2.2. Stage 3 is not considered in the bond formation modeling. Other alternative criteria could also be used within this framework, such as reducing the void area. Note that the focus of this paper is not to improve the process, but to make optimum use of the existing process. In this context, the proposed methodology leverages the transient heat transfer model and sintering neck growth model for cylindrical filaments proposed by Costa et al. [7] and Gurrala et al. [12], respectively. It quantifies the uncertainty in the coupled multi-physics model prediction, and determines the optimal process parameters through model-based optimization under uncertainty to improve the bond quality at each layer to overcome poor intra-layer and inter-layer bonding. Monte Carlo simulation (MCS) is used to propagate uncertain quantities through the coupled physics-based models by generating random samples from the design space. Optimization under uncertainty requires the evaluation of the uncertainty in the output for each iteration of the optimization algorithm. In other words, the MCS is nested within the optimization step, making the optimization under uncertainty very expensive. Therefore, a surrogate-based optimization framework [16] is employed, where an inexpensive surrogate of the original physics model is used to find the optimum process parameters.

In summary, the contributions of this paper are as follows:

1. Development of a Bayesian methodology to quantify the uncertainty in the neck growth prediction.

2. Construction of a GP surrogate model for efficient computation of model error, in order to incorporate model error in process parameter optimization.

3. Evaluation of the relative contributions of various uncertainty sources to the uncertainty in the model output.

4. Development of a computational framework for process parameter optimization under uncertainty, in order to maximize the bond quality between extruded polymer filaments in FFF.

The outline of the rest of this paper is as follows. Section 2 provides background information on physics-based models in FFF. Section 3 presents the basic framework of Bayesian model calibration, variance-based sensitivity analysis, surrogate modeling, and develops the proposed optimization under uncertainty methodology that optimizes the process parameters of FFF for maximizing the overall bond quality of the part. The proposed methodology is illustrated for a numerical example and is validated with physical experiments in Section 4. Concluding remarks are provided in Section 5.

## 2 Background

## 2.1 Heat transfer analysis

In FFF, all the filaments are subjected to the same heat transfer mechanism, but with different boundary conditions depending on the part geometry and deposition sequence (Fig. 2). The heat transfer model needs to be run for each interface since the neck growth predictions between each filament require the temperature evolution as an input. The time-dependent temperature profiles of filaments can be determined by performing a one-dimensional transient heat transfer analysis developed by Costa et al. [8]. The reason why a low-fidelity model has been used instead of a 2D or 3D high-fidelity model is that the low-fidelity model is much cheaper to run. Moreover, the low-fidelity model does well at higher temperatures, where most of the neck growth occurs.

![](images/fd186b868e3fd6a6108fd689146df0e53dff3ddc9ab92c7d0fc9d5619442e3a1.jpg)  
Figure 2: Example of a filament deposition sequence

The deposition of filaments is modeled gradually by joining elementary lengths that are associated with a given deposition time. The mathematical energy balance for an elementary length dx can be written as:

$$
\begin{array} { c } { \displaystyle { - k A \frac { \partial T _ { m } ( x , t ) } { \partial x } - h _ { c o n v } A _ { m } ^ { c o n v } \left( T _ { m } ( x , t ) - T _ { S } \right) - \sum _ { i = 1 } ^ { n } h _ { i } A _ { m } ^ { i } \left( T _ { m } ( x , t ) - T _ { m } ^ { i } \right) = \alpha A \frac { \partial T _ { m } ( x , t ) } { \partial t } d x - \frac { \partial T _ { m } ( x , t ) } { \partial x } \mathrm { ~ d ~ } _ { i } } } \\ { \displaystyle { A \left[ k \frac { \partial T _ { m } ( x , t ) } { \partial x } + \frac { \partial \left( k \frac { \partial T _ { m } ( x , t ) } { \partial x } \right) } { \partial x } d x \right] _ { \mathrm { ~ a ~ x ~ } } } } \end{array}\tag{1}
$$

where k, $h _ { c o n v } , h _ { i }$ and $\alpha = \rho C$ are the material properties of the polymer and assumed to be temperature-independent. The parameter k is the thermal conductivity, $h _ { c o n v }$ is the convective heat transfer coefficient, $h _ { i }$ represents the contact heat transfer coefficient, $\rho$ and C are the density and specific heat capacity of the material, $T _ { m }$ is the temperature at cross-section ${ \mathit { z } } _ { \mathrm { c u t } }$ of the m-th filament $( m \in { \bar { \{ 1 , . . . , N } }  \}$ , where N is the total number of deposited filaments) at deposition time instant t, $T _ { m } ^ { i }$ represents the temperature of the adjacent filament or build plate at contact $i ( i \in \{ 1 , . . . , n \}$ where n is the total number of contact surfaces of a filament including the contact with the build plate) of the m-th filament or the build plate, $T _ { S }$ is the surrounding environment temperature, and A represents the cross-section area of a filament. $A _ { m } ^ { c o n v }$ is the area of the m-th filament that is in contact with the environment, $A _ { m } ^ { i }$ is the area of contact i for the m-th filament as shown in Fig. 3, and they are defined as:

$$
A _ { m } ^ { c o n v } = P \left( 1 - \sum _ { i = 1 } ^ { n } \omega _ { m } ^ { i } \lambda _ { i } \right) d x ,\tag{2}
$$

$$
A _ { m } ^ { i } = P \omega _ { m } ^ { i } \lambda _ { i } d x ,\tag{3}
$$

where $P$ is the filament perimeter, $\lambda _ { i }$ is the fraction of P that is in contact with another filament or with the build plate, and $\omega _ { m } ^ { i }$ is a variable, which equals unity if the m-th filament has the i-th contact, and zero otherwise.

![](images/b3ebaeb20bfacab15c6342c924e9f18e8a18c1f2821cdb575b08bd48da99eed3.jpg)  
(a)

![](images/c07f2e3ef24ba09c941109927df2a7cfdfa3a8cedaf9a5be9fe4746bb0a579d4.jpg)  
<sup>(b)</sup> Figure 3: Possible contact areas of a filament in the: (a) first layer and (b) remaining layers

Axial and radial heat conduction can be neglected due to the low thermal conductivity of polymers and small filament radius [17]. Humidity in the environment, the effect of filament age and percentage of humidity in the filament were not considered in the heat transfer analysis. Thus, after these assumptions the energy equation becomes:

$$
\frac { \partial T _ { m } ( x , t ) } { \partial t } = - \frac { P } { \alpha A } \left[ h _ { c o n v } \left( 1 - \sum _ { i = 1 } ^ { n } \omega _ { m } ^ { i } \lambda _ { i } \right) \left( T _ { m } ( x , t ) - T _ { B P } \right) + \sum _ { i = 1 } ^ { n } h _ { i } \omega _ { m } ^ { i } \lambda _ { i } \left( T _ { m } ( x , t ) - T _ { m } ^ { i } \right) \right] .\tag{4}
$$

The analytical solution of Eq. (4) can be obtained using the characteristic polynomial method [18]:

$$
T _ { m } ( x , t ) = \phi _ { 1 } \exp \left[ \frac { P \chi \left( \omega _ { m } ^ { 1 } , . . . , \omega _ { m } ^ { n } \right) } { \alpha A } \left( t - t _ { m } ( x ) \right) \right] + \psi \left( \omega _ { m } ^ { 1 } , . . . , \omega _ { m } ^ { n } \right) ,\tag{5}
$$

where $\phi _ { 1 } = T _ { m } ( t _ { m } ( x ) ) - \psi \bigl ( \omega _ { m } ^ { 1 } , . . . , \omega _ { m } ^ { n } \bigr ) , T _ { m } ( t _ { m } ( x ) )$ is the temperature of the m-th filament at instant $t _ { m } ( x )$ at which an elementary length x of the m-th filament is deposited and starts to cool down or contact with an adjacent filament or the build plate. The functions that are influenced by the contacts χ and $\psi$ are defined as:

$$
\chi \left( \omega _ { m } ^ { 1 } , . . . , \omega _ { m } ^ { n } \right) = h _ { c o n v } \left( 1 - \sum _ { i = 1 } ^ { n } \omega _ { m } ^ { i } \lambda _ { i } \right) + \sum _ { i = 1 } ^ { n } \omega _ { m } ^ { i } h _ { i } \lambda _ { i } ,\tag{6}
$$

$$
\psi \left( \omega _ { m } ^ { 1 } , . . . , \omega _ { m } ^ { n } \right) = \frac { h _ { c o n v } \left( 1 - \sum _ { i = 1 } ^ { n } \omega _ { m } ^ { i } \lambda _ { i } \right) T _ { E } + \sum _ { i = 1 } ^ { n } \omega _ { m } ^ { i } h _ { i } \lambda _ { i } T _ { m } ^ { i } } { \chi \left( \omega _ { m } ^ { 1 } , . . . , \omega _ { m } ^ { n } \right) } .\tag{7}
$$

## 2.2 Sintering neck growth model between filaments

In this paper, we consider a sintering neck growth model to assess the bond quality that occurs at the interfaces between adjacent filaments. Note that bond length measurements are at the macro-scale; therefore our computations have also been at this scale. The bonding is quantified by measuring the length of the perimeter that is shared by adjacent filaments. A Newtonian sintering model was initially developed by Frenkel [10]. Pokluda et al. [9] developed a closed-form equation to predict the bond length (half of which is neck radius) $y$ as shown in Fig. 1.

Bellehumeuer et al. [11] applied the model proposed by Pokluda et al. [9] to FFF to predict the neck growth between adjacent filaments as a nonlinear function of time t, temperature-dependent surface tension $\Gamma ,$ viscosity of the ABS material $\eta ,$ and an initial radius $a _ { 0 }$ of the extruded filament before the sintering process. In this work, the Newtonian sintering model illustrated in Eq. (8) is used to predict the intra-layer bond lengths of the FFF product.

$$
\frac { d \theta } { d t } = \frac { \Gamma } { a _ { 0 } \eta } \frac { 2 ^ { - 5 / 3 } \cos \theta \sin \theta ( 2 - \cos \theta ) ^ { 1 / 3 } } { ( 1 - \cos \theta ) ( 1 + \cos \theta ) ^ { 1 / 3 } }\tag{8}
$$

where $\theta = \sin ^ { - 1 } y / a$ is the bond angle.

The bonding model (Eq. (8)) assumes that the filaments are spheres. In order to predict the bond lengths between two cylindrical filaments, a model developed by Gurrala et al. [12] given in Eq. (9) is used:

$$
\frac { d \theta } { d t } = \frac { \Gamma } { 3 \sqrt { \pi } a _ { 0 } \eta } \frac { [ ( \pi - \theta ) \mathrm { c o s } \theta + \mathrm { s i n } \theta ] [ \pi - \theta + \mathrm { s i n } \theta \mathrm { c o s } \theta ] ^ { 1 / 2 } } { ( \pi - \theta ) ^ { 2 } \mathrm { s i n } ^ { 2 } \theta }\tag{9}
$$

## 3 Proposed Methodology

The proposed methodology for process parameter optimization under uncertainty consists of the following steps:

1. Uncertainty quantification in FFF product bond quality

2. Probabilistic sensitivity analysis

3. Surrogate modeling of physics model discrepancy

4. Process parameter optimization under uncertainty

5. Physical experiments for model calibration and validation

The following subsections describe these steps in detail.

## 3.1 Uncertainty quantification in FFF

In this subsection, several uncertainty sources in FFF are identified and methods to quantify them are discussed.

## 3.1.1 Uncertainty sources

The heat transfer and sintering neck growth models used in this paper have their own model inputs, parameters, and errors. Some of these model inputs, parameters, and errors are deterministic while others are uncertain. The uncertainty sources can be aleatory (natural variability) or epistemic (lack of knowledge). The input values used in the heat transfer model (the thickness, width and length of filaments, printer nozzle temperature and extrusion speed) may not be the same as the actual value, thus introducing uncertainty regarding the input to the bond length model. Of the above parameters, the printer nozzle temperature is assumed to vary across printed parts. The temperature of the filaments immediately after being extruded was found to be significantly lower than the specified printer nozzle temperature. This variation in the temperature of the filament as it leaves the nozzle tip is included in the heat transfer model. Moreover, the heat transfer model is also affected by model parameter uncertainty, which is considered epistemic uncertainty (i.e., they have fixed values which are unknown), such as density, specific heat capacity, convective heat transfer coefficient, and fractions of filament perimeter that is in contact with another filament or with the build plate. Thus, the uncertainty in the printer nozzle temperature and model parameters of the heat transfer model introduces uncertainty in the temperature history of the interfaces. The uncertainty in the output of the heat transfer model further propagates to the output quantity of interest through the sintering neck growth model. In addition, the uncertainty regarding the model parameters of the sintering model (such as surface tension, and material viscosity) introduce additional uncertainties in the quantity of interest. Both models also have errors since they have various assumptions, and are not perfect representations of the actual physics. Thus, the bond length predictions have uncertainty due to the propagation of the effects of these different uncertainty sources. The unknown model parameters and errors can be estimated using the experimental data, which contain measurement noise/observation error, through Bayesian model calibration.

## 3.1.2 Model calibration under uncertainty

Consider a single physical quantity of interest y predicted by a physics model that maps input variables x and model parameters $\theta _ { m }$ to the numerical model output ${ \bf y } _ { m }$ :

$$
\mathbf { y } _ { m } = \mathbf { G } \big ( \mathbf { x } ; \pmb { \theta } _ { m } ( \mathbf { x } ) \big ) .\tag{10}
$$

Let $n _ { D }$ be the number of collected observation data $\mathbf { y } _ { D }$ from experiments with input variables $\mathbf { x } ^ { ( 1 ) } , . . . , \mathbf { x } ^ { ( n _ { D } ) }$ , where $\mathbf { x } ^ { ( i ) }$ are the input variables for the i-th experiment. The difference between observations $\mathbf { y } _ { D }$ and the true response of the system $\mathbf { y } _ { \mathrm { t r u e } }$ is attributed to observation error $\epsilon _ { \mathrm { o b s } } .$ , which is often treated as a zero-mean Gaussian random variable with variance $\sigma _ { \mathrm { o b s } } ^ { 2 }$ . As discussed in Section 3.1.1, the model predictions are affected by model errors due to missing physics or approximations. Therefore, a model discrepancy term $\pmb { \delta } ( \mathbf { x } )$ as a function of model inputs (one of the main features of the Bayesian calibration framework developed by Kennedy and O’Hagan [19]) can be introduced as shown in Fig. 4 to capture the disagreement between $\mathbf { y } _ { \mathrm { t r u e } }$ and ${ \bf y } _ { m }$ . Thus, the true system response can be described as

$$
\mathbf { y } _ { \mathrm { t r u e } } ( \mathbf { x } ) = \mathbf { y } _ { D } ( \mathbf { x } ) + \epsilon _ { \mathrm { o b s } } ( \mathbf { x } ) = \mathbf { y } _ { m } ( \mathbf { x } ) + \delta ( \mathbf { x } ) = \mathbf { G } \big ( \mathbf { x } ; \theta _ { m } ( \mathbf { x } ) \big ) + \delta ( \mathbf { x } ) .\tag{11}
$$

![](images/106bd861223eee1802306cfd11b7b8f4ec283505d7cb80cd9d7ba6e18088ed0f.jpg)  
Figure 4: Relating model output to observation data

Input variables x are measurable quantities and chosen by the experimenter. These can be considered deterministic or stochastic with known probability distributions due to natural variability (aleatory) or measurement error. Whereas, model parameters are uncertain due to lack of knowledge (epistemic) since $\pmb { \theta } _ { m }$ take some unknown deterministic values during the experiment. The purpose of Bayesian model calibration is to use observation data $\mathbf { y } _ { D }$ to estimate the posterior distributions of $\theta _ { m }$ and other unknown quantities such as parameters of observation error and the discrepancy term. Kennedy and O’Hagan [19] employ a probabilistic relationship between the predictions and observations, which incorporates both model parameters and a discrepancy function. Note that the discrepancy function is not observable from the observation data, since the true values of model parameters are unknown. The model discrepancy function is treated as a Gaussian process (GP). The hyperparameters of the GP (including the coefficients of the trend function) can be estimated along with physics model parameters using a Bayesian approach. However, in the presence of insufficient amount of experimental data and non-informative prior knowledge about the uncertainty sources in the engineering system, it may be difficult to distinguish between the effects of the model parameters and model discrepancy; this problem is referred to as non-identifiability [20, 21] when the number of parameters and hyperparameters that need to be estimated becomes large when the model discrepancy term is treated as a GP.

Two strategies are pursued here for improving the identifiability: (1) performing sensitivity analysis to identify the most important physics model parameters as described in Section 3.2, and (2) ignoring the discrepancy term during the calibration step and building a surrogate model for the discrepancy (i.e., the difference between the calibrated model prediction and actual system response), as described in Section 3.3.

First, the physics model parameters that have the most significant contribution to the uncertainty in the model output during the entire printing process are calibrated without including the model discrepancy term. The joint posterior distribution of $\pmb { \theta } _ { m }$ can be computed using Bayes’ theorem:

$$
\pi ( \pmb { \theta } _ { m } | \mathbf { y } _ { D } ) = \frac { L ( \mathbf { y } _ { D } | \pmb { \theta } _ { m } ; \mathbf { x } ^ { ( 1 ) } , . . . , \mathbf { x } ^ { ( n _ { D } ) } ) \pi ( \pmb { \theta } _ { m } ) } { \int _ { \Omega _ { \pmb { \theta } _ { m } } } L ( \mathbf { y } _ { D } | \pmb { \theta } _ { m } ; \mathbf { x } ^ { ( 1 ) } , . . . , \mathbf { x } ^ { ( n _ { D } ) } ) \pi ( \pmb { \theta } _ { m } ) d \pmb { \theta } _ { m } }\tag{12}
$$

where $\Omega _ { \theta _ { m } }$ is the domain of $\theta _ { m } , \pi ( \theta _ { m } )$ is the joint prior distribution of ${ \pmb \theta } _ { m } , L ( { \bf y } _ { D } | { \pmb \theta } _ { m } ; { \bf x } ^ { ( 1 ) } , . . . , { \bf x } ^ { ( n _ { D } ) } )$ is the likelihood function, and the observation data is denoted as $\mathbf { y } _ { D } = [ \mathbf { y } _ { D } ^ { ( 1 ) } , . . . , \mathbf { y } _ { D } ^ { ( n _ { D } ) } ] \in \mathbb { R } ^ { n _ { D } }$

Bayesian model calibration is often performed using Markov chain Monte Carlo (MCMC) sampling algorithms (such as Metropolis-Hastings [22], Gibbs [23], or slice sampling [24]) since the integral in the denominator of Eq. (12) makes numerical integration intractable for increasing dimension of calibration quantities [25]. The Metropolis-Hastings algorithm is used in this paper.

Next, a GP surrogate model is constructed as discussed in Section 3.3 for the model discrepancy. A set of additional experiments can be conducted to obtain training data for building a surrogate model such as GP model in order to estimate the model discrepancy at any input value x. (Note that two sets of experimental data are used: the first set for calibrating the physics model parameters and the second set for building the surrogate model of model discrepancy. This two-step approach was possible in this study since the experiments were inexpensive and fast; if it is not possible to conduct two sets of experiments, then simultaneous calibration of physics model parameters and GP hyperparameters will need to be pursued, with appropriate assumptions and sensitivity analysis to reduce the number of calibration quantities and achieve identifiability). The training data of the GP model for the model discrepancy can be evaluated for different input values of experimental tests and realizations of observation errors by comparing model predictions against experimental data $\mathbf { y } _ { D } ( \mathbf { x } )$

$$
\delta ( \mathbf { x } ) = \mathbf { y } _ { D } ( \mathbf { x } ) + \epsilon _ { \mathrm { o b s } } - \mathbf { y } _ { m } ( \mathbf { x } ) .\tag{13}
$$

The GP model $( \delta _ { G P } ( \mathbf { x } ) )$ for the model discrepancy captures the combined contribution of model form and measurement error for a given bond length. Thus, the corrected prediction of the physics-based model $\mathbf { y } _ { \mathrm { p r e d } }$ can be written as

$$
\mathbf { y } _ { \mathrm { p r e d } } ( \mathbf { x } ) = \mathbf { y } _ { m } ( \mathbf { x } ) + \delta _ { G P } ( \mathbf { x } ) .\tag{14}
$$

Two cases with different inputs and model functions are considered in this paper, namely cases A and B. The inputs to the sintering neck growth model, temperature profiles of the extruded filaments, are predicted using the heat transfer model in case A, whereas, measured temperature data from experiments are used as the inputs to the neck growth model in case B. $\mathbf { G } \left( \mathbf { x } ; \pmb { \theta } _ { m } ( \mathbf { x } ) \right)$ represents (a) the coupled physics-based heat transfer and sintering neck growth models for case A and (b) sintering neck growth model, which can be evaluated using a numerical technique such as 4th order Runge-Kutta method, for case B (see Fig. 5). In these cases, the physics model is inexpensive to evaluate, thus a surrogate model has not been built for the model $\mathbf { G } \left( \mathbf { x } ; \pmb { \theta } _ { m } ( \mathbf { x } ) \right)$ .

![](images/fcc53a4a20b1d8a0ad2b1f04bbcd57902c2c57c12906dd1d55a014f800ebe855.jpg)  
<sup>(a) (b)</sup> Figure 5: Flowchart of the simulation models for (a) case A and (b) case B

## 3.2 Probabilistic sensitivity analysis

A probabilistic sensitivity analysis, commonly referred to as global sensitivity analysis (GSA) is used to assess the relative contribution of each uncertainty source towards the uncertainty of the model output (bond length in this case). Model inputs or parameters with negligible contribution can be fixed at their mean values in order to reduce the number of stochastic variables. Variance-based GSA, using Sobol’ indices, is adopted in this paper, as briefly described below.

## 3.2.1 Variance-based global sensitivity analysis

Consider a real integrable deterministic one-to-one system response function $\mathrm { Y } = f ( \mathbf { X } )$ , where $\mathbf { X } = \{ \mathrm { X } _ { 1 } , . . . , \mathrm { X } _ { k } \}$ are mutually independent model inputs, $f ( \cdot )$ is the computational model and Y is the model output. As shown by Sobol [26], the variance of Y can be decomposed as

$$
V ( \mathbf { y } ) = \sum _ { i } ^ { k } V _ { i } + \sum _ { i _ { 1 } } ^ { k } \sum _ { i _ { 2 } = i _ { 1 } + 1 } ^ { k } V _ { i _ { 1 } i _ { 2 } } + \sum _ { i _ { 1 } } ^ { k } \sum _ { i _ { 2 } = i _ { 1 } + 1 } ^ { k } \sum _ { i _ { 3 } = i _ { 2 } + 1 } ^ { k } V _ { i _ { 1 } i _ { 2 } i _ { 3 } } + \ldots + V _ { 1 2 \ldots k }\tag{15}
$$

where $V _ { i }$ is the variance of $\mathrm { Y }$ due to $\mathrm { X } _ { \mathrm { i } }$ alone, and $V _ { i _ { 1 } \dots i _ { p } } ( p \geq 2 )$ indicates the variance of $\mathrm { Y }$ due to $\{ \mathrm { X } _ { \mathrm { i _ { 1 } } } , . . . , \mathrm { X } _ { \mathrm { i _ { p } } } \}$ The Sobol indices are defined by dividing both sides of Eq. (15) with $V ( \mathrm { Y } )$

$$
1 = \sum _ { i } ^ { k } S _ { i } + \sum _ { i _ { 1 } } ^ { k } \sum _ { i _ { 2 } = i _ { 1 } + 1 } ^ { k } S _ { i _ { 1 } i _ { 2 } } + \sum _ { i _ { 1 } } ^ { k } \sum _ { i _ { 2 } = i _ { 1 } + 1 } ^ { k } \sum _ { i _ { 3 } = i _ { 2 } + 1 } ^ { k } S _ { i _ { 1 } i _ { 2 } i _ { 3 } } + \ldots + S _ { 1 2 \ldots k }\tag{16}
$$

where $S _ { i }$ is the first-order or main effects index that assesses the contribution of $\mathrm { X } _ { \mathrm { i } }$ individually to the variance of the output Y without considering interactions with other inputs. The higher-order indices $S _ { i _ { 1 } \dots i _ { p } } ( p \geq 2 )$ in Eq. (16) measure the contributions of the interactions among $\{ \mathrm { X } _ { \mathrm { i _ { 1 } } } , . . . , \mathrm { X } _ { \mathrm { i _ { p } } } \}$

In other words, the evaluation of $S _ { i }$ is as follows:

$$
S _ { i } = \frac { V _ { i } } { V ( \Upsilon ) } = \frac { V _ { X _ { i } } ( E _ { \mathbf { X } _ { - i } } ( \Upsilon | \mathrm { X } _ { i } ) ) } { V ( \Upsilon ) }\tag{17}
$$

where $\mathbf { X } _ { - i }$ are all the model inputs other than $\mathrm { X } _ { i }$

The overall contribution of $\mathrm { X } _ { \mathrm { i } }$ considering an individual input and its interactions with all other inputs is measured by the total effects index $S _ { i } ^ { T }$

$$
S _ { i } ^ { T } = 1 - \frac { V _ { - i } } { V ( \Upsilon ) } = \frac { V _ { { \bf X } _ { - i } } ( E _ { { \bf x } _ { i } } ( \Upsilon | { \bf X } _ { - i } ) ) } { V ( \Upsilon ) } .\tag{18}
$$

The calculation of first-order and total effects indices requires a deterministic function. Furthermore, the total effects index $S _ { i } ^ { T }$ is only meaningful for uncorrelated model inputs [27]. Whereas, the first-order index $S _ { i }$ can be calculated for both correlated and uncorrelated model inputs [27].

The computation of $S _ { i }$ analytically is nontrivial since $E _ { \mathbf { X } _ { - i } } ( \cdot )$ requires multi-dimensional integrals. Using Monte Carlo simulation (MCS) to measure $S _ { i }$ is also expensive because calculation of the numerator of $\breve { S } _ { i }$ requires a double-loop MCS, where the inner loop $E _ { \mathbf { X } _ { - i } } ( \mathrm { Y } | \mathrm { X } _ { i } )$ computes the mean value of $\mathrm { Y }$ using $n _ { 1 }$ random samples of ${ \bf X } _ { - i }$ and the outer loop computes $V _ { X _ { i } } ( \mathrm { { \bar { E } } } _ { \mathbf { X } _ { - i } } ( \mathrm { { \bar { Y } } } | \mathrm { { X } } _ { i } ) )$ by iterating the inner loop $n _ { 2 }$ times for different values of $\mathrm { X } _ { i } .$ . Moreover, the computation of $V ( \dot { \mathrm { Y } } )$ requires additional $n _ { 3 }$ MCS iterations. The total computational cost of the first-order and total effect indices (i.e. the number of function evaluations, $N _ { f } )$ is approximately $N _ { f } = k N ^ { 2 } + N$ , where $N = n _ { 1 } = n _ { 2 } = n _ { 3 }$ . Thus, the calculation of indices become unaffordable for a large number of inputs and expensive models since the required number of random samples are of the order greater than 1000 in many practical applications.

The sampling-based method modularized global sensitivity analysis (MGSA) proposed by Li and Mahadevan [28], which has a computational cost that is not proportional to the model input dimension, can directly estimate the first-order Sobol’ index from Monte Carlo samples with a single-loop instead of the double-loop MCS. The first-order Sobol’ index can be computed by dividing the input variable $X _ { i }$ into equally probable intervals $\phi = \phi ^ { 1 } , . . . , \phi ^ { M }$

$$
S _ { i } = 1 - \frac { E _ { \phi } ( V _ { \phi ^ { p } } ( Y ) ) } { V ( Y ) } , p = 1 , . . . , M\tag{19}
$$

where $V _ { \phi ^ { p } } ( Y )$ is the variance of $Y$ when $X _ { i }$ in the subspace $\phi ^ { p } , V ( Y )$ represents the variance of the system response $Y$ . The main advantages of this efficient data-driven method based on the concept of stratified sampling [29] are that the computational cost is not proportional to the number of model inputs and the physics/computational model does not have to be available since the input-output samples are enough to compute the first-order Sobol index. Further advances in this direction based on importance sampling and kernel functions have been reported by DeCarlo et al. [30].

## 3.3 Gaussian process (GP) surrogate modeling

In this paper, a GP surrogate model is constructed to estimate the model discrepancy at unobserved inputs. The GP surrogate model properly accounts for statistical dependence between the outputs at different input values, and provides the output at a given input value as a normal distribution with an expected value and a standard deviation.

In the GP model, the response at prediction point u, G(u) is described by:

$$
G ( \mathbf { u } ) = \mathbf { h } ( \mathbf { u } ) ^ { T } \beta + Z ( \mathbf { u } )\tag{20}
$$

where $\mathbf { h } ( \cdot )$ is the trend of the model, $\beta$ is the vector of trend coefficients, and $Z ( \cdot )$ is a zero-mean stationary Gaussian process which describes the deviation of the model from the trend. The covariance between the outputs $Z ( \cdot )$ of the GP surrogate at points a and b is defined as:

$$
\mathrm { C o v } [ Z ( { \bf a } ) , Z ( { \bf b } ) ] = \sigma _ { Z } ^ { 2 } R ( { \bf a } , { \bf b } )\tag{21}
$$

where $\sigma _ { Z } ^ { 2 }$ is the process variance and $R ( \cdot , \cdot )$ is the correlation function. A squared exponential function with separated length scale parameters $l _ { i }$ for each input dimension has often been used in the literature:

$$
R ( \mathbf { a } , \mathbf { b } ) = \exp \left[ - \sum _ { i = 1 } ^ { \mathrm { M } } \frac { \left( a _ { i } - b _ { i } \right) ^ { 2 } } { l _ { i } } \right]\tag{22}
$$

The outputs of the GP model are the mean prediction $\mu _ { G } ( \cdot )$ and the variance of the prediction $\sigma _ { G } ^ { 2 } ( \cdot )$ , defined as:

$$
\mu _ { G } ( \mathbf { u } ) = \mathbf { h } ( \mathbf { u } ) ^ { T } { \boldsymbol \beta } + \mathbf { r } ( \mathbf { u } ) ^ { T } \mathbf { R } ^ { - 1 } ( \mathbf { g } - \mathbf { F } { \boldsymbol \beta } )\tag{23}
$$

$$
\sigma _ { G } ^ { 2 } ( \mathbf { u } ) = \sigma _ { Z } ^ { 2 } - \mathbf { A } \left[ \begin{array} { l l } { \mathbf { 0 } } & { \mathbf { F } ^ { T } } \\ { \mathbf { F } } & { \mathbf { R } } \end{array} \right] ^ { - 1 } \mathbf { A } ^ { T }\tag{24}
$$

where $\mathbf { r } ( \mathbf { u } )$ is a vector containing the covariance between u and each of the training points $\{ x _ { 1 } , x _ { 2 } , . . . , x _ { n } \} , i \in$ $\{ 1 , . . . , n \}$ , R is an $n \times n$ matrix containing the correlation between each pair of training points, ${ \bf R } ( x _ { i } , ~ x _ { j } ) = { }$ $\dot { \mathrm { C o v } } [ Z ( \dot { x _ { i } } ) , Z ( x _ { j } ) ] ; \mathbf { g }$ is the vector of original physics model outputs at each of the training points, F is a $n \times q$ matrix with rows ${ \bf h } ( { \bf u } _ { i } ) ^ { T }$ , and $\mathbf { A } = \left[ \mathbf { h } ( \mathbf { u } ) ^ { T } \mathbf { \Omega } \mathbf { r } ( \mathbf { u } ) ^ { \top } \right]$

The trained GP surrogate model is used to correct the model prediction, which feeds into the process parameter optimization calculations as presented in Section 3.4.

## 3.4 Process optimization under uncertainty

The optimal design point that satisfies design criteria and a specified level of reliability is of great interest in many engineering applications. In this paper, the focus is on selecting the optimum values of process parameters that maximize the bond lengths between filaments and between layers. The AM process optimization under uncertainty can be pursued in two directions: (1) reliability-based design optimization (RBDO) [31], and (2) robust design optimization (RDO) [32]. In RBDO, the decision variables are optimized to either maximize or achieve a desired target level of reliability (i.e., probability of satisfying a desired threshold of performance or quality). In RDO, the decision variables are optimized such that the variability of the objective function is minimized, and the constraints are satisfied within specified uncertainty bounds. The approach of robustness-based design optimization (RDO) is used to maximize the overall bond quality. The robustness of the objective function can be achieved by simultaneously optimizing the mean and variance; thus, this is a bi-objective problem. Monte Carlo sampling is used to compute the mean and variance of the objective function (the mean bond length of a layer) in the probabilistic optimization process. An efficient sampling-based method Latin hypercube sampling is used to simulate the uncertain parameters.

The robust design optimization problem can be formulated as follows:

$$
\begin{array} { r l } { \underset { \mathbf { d } \in \mathbb { R } ^ { n _ { d } } } { \mathrm { m i n i m i z e } } } & { \big \{ E ( y _ { o } ( \mathbf { d } ) ) , V ( y _ { o } ( \mathbf { d } ) ) \big \} ; } \\ { \mathrm { s u b j e c t ~ t o } } & { \big \{ E ( y _ { c } ( \mathbf { d } ) ) , V ( y _ { c } ( \mathbf { d } ) ) \big \} \leq 0 , c = 1 , 2 , . . . , n _ { c } ; } \\ & { d _ { i , \mathrm { l b } } \leq d _ { i } \leq d _ { i , \mathrm { u b } } , i = 1 , 2 , . . . , n _ { v } , } \end{array}\tag{25}
$$

where the objective function and constraints are expressed as functions of expectation $E ( \cdot )$ and variance $V ( \cdot )$ of objective function $y _ { o }$ and constraints $y _ { c } ,$ , respectively; the vector of design variables d can be either deterministic parameters or mean/standard deviation of the design parameters, $n _ { c }$ and $n _ { v }$ are the number of constraints and design variables, respectively. The i-th design variable $d _ { i }$ is constrained by its lower bound $d _ { i ,  { \mathrm { l b } } }$ and upper bound $d _ { i , \mathrm { u b } }$ . The RDO problem becomes bi-objective by adding the variance of the performance function to the expected value of the objective function, and a weighted sum method can be used to assign proportional weights for the aggregation of the two objectives according to their importance [33]. The aggregation formulation is given by Eq. (26).

$$
\begin{array} { r l } { \underset { \mathbf { x } \in \mathbb { R } ^ { n _ { x } } } { \mathrm { m i n i m i z e } } } & { w _ { 1 } E \big ( y _ { o } ( \mathbf { d } ) \big ) + w _ { 2 } V \big ( y _ { o } ( \mathbf { d } ) \big ) ; } \\ { \mathrm { s u b j e c t ~ t o } } & { \mathbf { h } ( \mathbf { d } ) = 0 , } \\ & { \mathbf { g } ( \mathbf { d } ) \le 0 , } \\ & { d _ { i , \mathrm { l b } } \le d _ { i } \le d _ { i , \mathrm { u b } } , \ i = 1 , 2 , . . . , n _ { v } , } \end{array}\tag{26}
$$

where $w _ { 1 } , w _ { 2 } > 0$ are the weighting coefficients representing the relative importance of each objective function, and $\mathbf { h } ( \mathbf { d } )$ and $\mathbf { g } ( \mathbf { d } )$ are the vectors of equality and inequality constraints respectively.

The nested bi-objective robustness-based design optimization (RDO) problem can be converted into a single objective formulation, using a weighted sum approach, as:

$$
\begin{array} { r l } { \underset { \mathbf { x } \in \mathbb { R } ^ { n _ { x } } } { \mathrm { m i n i m i z e } } } & { w _ { 1 } \mathrm { F } _ { 1 } + w _ { 2 } \mathrm { F } _ { 2 } } \\ { \mathrm { s u b j e c t } \mathrm { t o } } & { \mathbf { x } _ { \mathrm { l b } } \leq \mathbf { x } \leq \mathbf { x } _ { \mathrm { u b } } } \end{array}\tag{27}
$$

where weighting coefficients $w _ { 1 } , w _ { 2 } > 0$ represent the relative importance of two objectives. $\mathrm { F } _ { 1 } = - \mu _ { \pmb { \mu } _ { \mathrm { B L , i } } } + \sigma _ { \pmb { \mu } _ { \mathrm { B L , i } } }$ represents the mean and standard deviation of the average bond length predictions $\mu _ { \mathrm { B I } }$ for layer i, and $\mathrm { F _ { 2 } } = \mu _ { \sigma _ { \mathrm { B L , i } } } +$ $\sigma _ { \sigma _ { \mathrm { B L , i } } }$ represents the mean and standard deviation of the standard deviation of the bond length predictions $\sigma _ { \mathrm { B L } }$ for layer $i , i \in \{ 1 , . . . , M \}$ . M is the total number of layers, x are the design variables (printer nozzle temperature and printer extrusion speed for each layer $i ) .$ , and ${ \bf x } _ { \mathrm { l b } } \le { \bf x } \le { \bf x } _ { \mathrm { u b } }$ represents the lower and upper bounds for the design variables. The weighted sum approach is a convex combination of two different objectives, $\mathrm { { \bar { F } _ { 1 } } }$ and $\mathrm { F _ { 2 } } .$ . The solution of the optimization problem approximates the Pareto front by changing the weights of each objective. The Pareto front maps the relation between these two objective functions. The negative of the mean value of mean bond lengths at each layer is minimized while minimizing the deviation of mean bond lengths at each layer with the use of function $\mathrm { F _ { 1 } }$ . The function $\mathrm { F _ { 2 } , }$ which is a convex combination of the mean and standard deviation of the deviation of bond lengths at each layer, is minimized simultaneously with $\mathrm { F _ { 1 } }$ . In other words, the overall bond quality (the mean value of bond length for a part) is maximized, while minimizing the variations in the quantity of interest (bond lengths between filaments) using the functions $\mathrm { F _ { 1 } }$ and $\mathrm { F _ { 2 } }$ respectively.

## 3.5 Experimental work

A commercial material, Ultimaker Black ABS, was used in the experiments. A unidirectional and aligned building strategy was adopted at a specified printer nozzle temperature and extrusion speed. Two different options for the deposition sequence of the filaments were considered, as shown in Fig. 6. In Fig. 6a, the filaments are sequenced from left to right in all the layers. In Fig. 6b, the filaments are sequenced from left to right in odd numbered layers and from right to left in even numbered layers. The filament numbers in the two figures correspond to the two deposition sequence options.

![](images/2765cb91dabe1056841a0b53b7e8b2e00dbbbd8a7b8bcac85c77fd6b424ba2e6.jpg)  
<sup>(a) (b)</sup> Figure 6: Deposition sequence of unidirectional 90 filaments: (a) from left to right for all the layers and (b) from left to right in odd numbered layers and from right to left in even numbered layers

Multiple rectangular-shaped specimens were produced with the same geometry but different combinations of process parameter values. For each specimen, the temperature distribution at the top of each layer during deposition was monitored using an infrared thermography camera. Thermal images were recorded with a specified frequency until all filaments were deposited. The neck growth between the filaments and the total void area of the parts were identified at a specified cross-section with the use of microscopy images processed through the ImageJ software [34]. The statistical properties of the neck growth along the length of the specimens were constant. Therefore, all specimens were sectioned at the midpoint to analyze the mesostructural feature of interest only at that cross-section.

## 4 Numerical results and discussion

The experimental setup used to build rectangular acrylonitrile butadiene styrene (ABS) amorphous polymer specimens of length 35 mm, width 12 mm, and thickness 4.2 mm is shown in Fig. 7. The specimens were created on an Ultimaker 2 extended+ printer, which is within an enclosure to reduce the part variability; a commercial material, Ultimaker Black ABS, was used. Air flow was not considered because we enclosed the printer to prevent the occurrence of airflow. All parts were printed through a nozzle with 0.8 mm diameter. The build plate temperature was constant and set to 110<sup>◦</sup>C and the environment temperature is assumed to be ${ 7 0 } ^ { \circ } \mathbf { C }$ . The extrusion rate and vertical position of the nozzle were adjusted by the printer to be able to produce each filament with 0.8 mm width and 0.7 mm height.

The surface temperature profiles of extruded filaments were monitored using an infrared thermography camera as shown in Fig. 7. The extrusion of the next layer prevents the camera from monitoring the temperature profiles of the previous layers. Due to the inability to obtain temperature data of the filaments below the top layer, we could only predict the quality of the intra-layer bonding using temperature profiles of extruded filaments within that layer. Thermal images were recorded with a frequency of 10 Hz until the deposition of all filaments was completed.

![](images/48cff11a636fb7844504ea1361fb75e97a0426f703062ac59a766799c10066f2.jpg)  
Figure 7: The experimental setup

All specimens used in this study are produced with unidirectional filaments to enhance the effects of process parameters on the bond quality between adjoining filaments. Each filament of the rectangular part is deposited at a specified printer nozzle temperature $T _ { n }$ and extrusion speed $v _ { p } .$ . The temperature evolution of the interfaces, and the neck growth between the filaments (the mesostructural feature of interest), are predicted at $z _ { \mathrm { c u t } } = L / 2$ as shown in Fig. 6. The process parameters and material properties used in this work are presented in Table 1. The specific heat capacity and density of the material are calibrated together as a single term $\alpha = \rho C$ , where $\rho$ and C are density $\mathrm { ( k g / m ^ { 3 } ) }$ and specific heat capacity $\left( \mathrm { J } / \mathrm { k g } ^ { \circ } \mathrm { C } \right)$ respectively. The analysis assumes temperature dependent material properties such as material viscosity η and surface tension Γ. The surface tension of ABS P400 at $2 4 0 ^ { \circ } \mathrm { C }$ is 0.029 N/m as reported by Bellehumeur et al. [11] with a temperature dependence $\Delta \Gamma / \Delta T = - \gamma \mathrm { \bf N / m } \cdot \mathrm { \bf K }$ , where the neck growth model parameter $\gamma =$ $0 . 0 0 3 4 5$ . The temperature dependent material viscosity η is given by $\eta = \eta _ { r } \exp [ - \beta ( T - T _ { r } ) ]$ ], where the material viscosity at the reference temperature $( T _ { r } = 2 4 0 ^ { \circ } \mathrm { C } ) \eta _ { r }$ is 5100 Pa · s, β is a model parameter that is selected as 0.056 by Sun et al. [2], and T is the temperature of the material at a given time instance.

Table 1: Process parameters and material properties
<table><tr><td>Property</td><td>Value</td></tr><tr><td>Printer nozzle temperature  ${ \mathcal { C } } { \mathrm { { } } }$ </td><td>240</td></tr><tr><td>Build plate temperature  ${ \mathcal { C } } { \mathrm { { } } }$ </td><td>110</td></tr><tr><td>Printer extrusion speed (m/s)</td><td>0.042</td></tr><tr><td>Filament length (m)</td><td>0.035</td></tr><tr><td>Filament width (m)</td><td>0.0008</td></tr><tr><td>Filament thickness (m)</td><td>0.0007</td></tr><tr><td>Fraction of filament&#x27;s perimeter for all contacts</td><td>0.15</td></tr><tr><td>Convective heat transfer coefficient  $\mathrm { ( W / m ^ { 2 } \ ^ { \circ } C ) }$ </td><td>86</td></tr><tr><td>Conductive heat transfer coefficient between filaments  $\mathrm { ( W / m ^ { 2 } \ ^ { \circ } C ) }$ </td><td>200</td></tr><tr><td>Conductive heat transfer coefficient between filament and build plate  $\mathrm { ( W / m ^ { 2 } \ ^ { \circ } C ) }$ </td><td>86</td></tr><tr><td>Thermal conductivity  $\left( \mathbf { W } / \mathbf { m } ^ { \circ } \mathbf { C } \right)$ </td><td>0.15</td></tr><tr><td> $\alpha \ : ( \mathrm { J / m ^ { 3 } \ : ^ { \circ } C ) }$ </td><td> $1 . 1 9 6 \times 1 0 ^ { 6 }$ </td></tr></table>

## 4.1 Prediction of the cooling of filaments

A typical IR image of the temperature profile for the first layer of a part printed using $( T _ { n } , v _ { p } ) = ( 2 4 0 ^ { \circ } \mathbf { C } , 0 . 0 4 2$ m/s) is shown in Fig. 8. The interface temperature is monitored at corresponding locations between filaments. The experimental temperature profile was used to assess the validity of the heat transfer model in order to be able to predict the neck growth accurately using the heat transfer model predictions. The temperature of the filaments immediately after being extruded onto the build plate or onto another filament was found to be significantly lower $( 2 0 ^ { \circ } \mathrm { C }$ to $5 0 ^ { \circ } \dot { \mathbf { C } } )$ than the specified printer nozzle temperature. At the upper temperature limit of the printer the filaments were extruded at temperatures approximately $4 0 – 5 0 ^ { \circ } \mathrm { C }$ less than the set nozzle temperature. The enclosure was not a precisely controlled environmental chamber with quantitative measurements of temperature, humidity, and airflow. Therefore, these effects might be influencing the difference between the printer setting and observed temperature. This variation in the temperature of the filament as it leaves the nozzle tip is considered as a bias term in the heat transfer model.

![](images/2b015423bb8acffb94a6cb0372cb68dc1a1dcfae631a281ec1586f453f2f11cb.jpg)  
Figure 8: Top view temperature profile of the first layer

![](images/e995f23341f2ccc0c319ec4b8c70b68d34bb524abdb2cc07fc5364df75b6629f.jpg)  
(a)

![](images/1f522f99777cbc1b57494bb4e3149330b6545511ef33f6434656d9ea1c9fcbf3.jpg)  
(b)  
Figure 9: Temperature evolution of 90 filaments at $z _ { \mathrm { c u t } } = L / 2$ (a) with $L = 0 . 0 3 5$ m shown in Fig. 6a and (b) with $L \stackrel { - } { = } 0 . 2 1$ m shown in Fig. 6b along deposition time

The temperature evolution of all the filaments illustrated in Fig. 6a and Fig. 6b at $z _ { \mathrm { c u t } } = 0 . 0 1 7 5$ m along deposition time is shown in Fig. 9a and Fig. 9b respectively. The length to diameter ratio of filaments has a significant effect on the cooling process. The time it takes for the printer to extrude a single filament increases as the lengths of the filaments get longer. This results in a faster cooling process, and consequently a smaller amount of heat transfer between each filament. Moreover, extruding each layer’s first filament at the same x-coordinate results in a more homogeneous part quality as the temperature difference between the filaments extruded on top of the filaments below is approximately the same. Whereas, in Fig. 6b, the temperature difference between the 1st and 30th filaments is much greater than 15th and 16th filaments, resulting in a staggered temperature evolution and part quality. Therefore, the build strategy shown in Fig. 6a is used for further analysis.

The Newtonian sintering model is coupled with the heat transfer model to predict the bond lengths between filaments. The one-dimensional transient heat transfer model predictions, and observed temperature data for the interface between filaments 1 and 2 are compared in Fig. 10. The model predictions are in general agreement with the measured data. The measured temperature data and model predictions show a similar trend in the initial stage due to enhanced convection at higher temperatures, but at temperatures below 130<sup>◦</sup>C the model prediction deviates away from the measurement data. However, the inaccuracy of the model at lower temperatures is not relevant for neck growth predictions since the neck growth process occurs at higher temperatures [11].

The neck radius predictions for each case are corrected with the model discrepancy (estimated by the surrogate model) at given input values. The measured neck radius and the corrected neck growth predictions corresponding to case A (heat transfer model predictions as the input to the sintering neck growth model) and case B (observed temperature profile as the input to the sintering neck growth model) are demonstrated in Fig. 10 for $( T _ { n } , v _ { p } ) = ( 2 4 0 ^ { \circ } \mathbf { C } , 0 . 0 4 2$ m/s). The corrected neck radius predictions are in agreement with the measured data around $1 3 \dot { 0 } ^ { \circ } \mathrm { C }$ since the model parameters are calibrated using the neck radius predictions when the interface temperature is at $1 3 0 ^ { \circ } \mathrm { C }$ . Thus, case A is used for further analysis when the temperature data is not available.

![](images/1dfabb44f962effeb4201cdd34518fe62194275c4848c179c2a45a053dfec9a0.jpg)  
Figure 10: Experimental temperature profiles compared with model predictions for the interface between filaments 1 and 2 at $\tilde { z } _ { \mathrm { c u t } } = 0 . 0 1 7 5$ m and the neck growth predictions using the heat transfer model predictions (case A) and experimental temperature data (case B)

α  
![](images/edc326470265f14c4ed0822e3521d49f46151f2036604ccfdc8db0dc2212105b.jpg)  
(a)

![](images/9d8e548c94c16d2f6d1aedb703bacd514c7d73ef91c992787f9ccc7fc307e00d.jpg)

![](images/0b4c73189a1536eb83291a9d0853328ad61e768abb277e97dc67c0b948665ad5.jpg)  
(c)

)  
![](images/2569695186735ea4815a4f4f3d592f0fbdc1f78e837ad74c2fea98ea4d4a9531.jpg)  
(d)  
Figure 11: Sensitivity indices for the bond length between the 1st and 2nd filaments in the first layer

## 4.2 Contribution assessment of each uncertainty source

A sample-based single loop algorithm called MGSA (modularized GSA) proposed by Li and Mahadevan [28] is used to compute the first-order Sobol’ indices. The results from MGSA indicate which parameters’ individual effect have significant contribution to the uncertainty in the coupled heat transfer and neck growth models. A low first-order index implies that the individual effect of the parameter is insignificant; thus, it can be fixed at its mean value. Thus, GSA provides insights on where to focus resources for improving the AM process.

The random variables in the heat transfer model are $\alpha , h _ { c o n v }$ , and $\lambda _ { i } , i \in \{ 1 , 2 , 3 , 4 , 5 \}$ , i.e., the material parameter, convective heat transfer coefficient and fraction of filament’s perimeter that is in contact with other filaments or with the build plate. The random variables in the sintering neck growth model are Γ, $\eta , \beta$ , and $\gamma ,$ i.e., the surface tension, material viscosity values at reference temperature of ${ \bar { 2 } } 4 0 ^ { \circ } \mathrm { C } ,$ , and model parameters of the temperature dependent surface tension and material viscosity respectively.

## 4.2.1 GSA of the bond length model

The coupled heat transfer and sintering neck growth model considers eleven parameters as uncertain, i.e., α, $h _ { c o n v } ,$ $\lambda _ { i } , i \in \bar { \{ 1 , 2 , 3 , 4 , 5 \} } , \Gamma , \eta , \beta$ and $\gamma .$ As discussed earlier, the contributions of various uncertainty sources to the neck growth vary for each layer as well; whereas, these contributions to the neck growth between filaments within a layer remain the same. Therefore, the first-order Sobol’ indices of material parameters are assessed for four different neck growths at four different layers, i.e., the 1st, 20th, 50th and 87th bond formations in the first, second, fourth and sixth layers, respectively. We performed sensitivity analysis for each layer, at several interfaces within each layer. We present the results for only 4 layers (i.e., first, second, fourth and sixth layers) out of a total of 6 layers since the results corresponding to the fifth layer were the same as for the fourth and sixth layers, and the results corresponding to the third layer were the same as for the second layer. The results from GSA for these neck growths are illustrated in Figs. 11, 12, 13, and 14.

α  
![](images/71ceeecc5724627e107d44f1ebdd94d18f4d455f9ce87b96d65746a5be5bf565.jpg)

![](images/7208b93caaf63ade847c0ad76836f46c22d2e6d5b20077d53adbb2c5e2d4d785.jpg)

(a)  
![](images/a5e9d8b3e4c4825b7668e6c726ff52dab3a1dde934aed0e3781898846cd6cc33.jpg)  
(c)

(b)  
![](images/346aa13febb8b10b52f378d44611957d008e844af1c0704866ba66ab0d4e2616.jpg)  
(d)  
Figure 12: Sensitivity indices for the bond length between the 20th and 21st filaments in the second layer

![](images/3568fb8779563f78507307801f6792a599edff7fee6d035e8e17fa127a52a04a.jpg)  
Figure 13: Sensitivity indices for the bond length between the 50th and 51st filaments in the fourth layer

![](images/afd21e07606c989239d0aa7330afe9960c7879aa4ba3f1cbda67ce13edcbbd27.jpg)  
Figure 14: Sensitivity indices for the bond length between the 87th and 88th filaments in the sixth layer

The contributions of $\beta$ increase significantly during the deposition of the specimen, while the contributions of $\alpha , \lambda _ { 1 } , \lambda _ { 2 }$ $\lambda _ { 3 } , \lambda _ { 4 } , \lambda _ { 5 } , h _ { c o n v } , \Gamma ,$ η and $\gamma$ to the variations in the neck growth at given layers are negligible. The sensitivity index of $\beta$ increases as the temperature of the interface between the 1st and 2nd, and the 20th and 21st filaments cools down as illustrated in Figs. 11c and 12c.

The uncertainty grows fast as the deposition time increases or temperature decreases because the influence of uncertainty in the model parameter $\beta$ on the neck growth increases with decreasing temperature. Another reason for the increase in uncertainty as the temperature decreases is that the neck growth model cannot capture the physics accurately when the temperature is below $\mathrm { i } 3 0 ^ { \circ } \mathrm { C }$ as shown in Fig. 10.

Based on the contributions of the various uncertainty sources to the neck growth at each layer of the $\mathrm { p a r t } , \beta$ was found to be dominant for all the layers and interfaces considered; next was α (considering the high contribution of α to the uncertainty in the temperature evolution of filaments) which was found to be significant in some of the layers. All the other parameters are fixed at their mean value to reduce computational effort since their contribution to the uncertainty in the neck growth predictions were negligible compared to the parameters α and $\beta ,$ based on the rigorous sensitivity analysis described in Section 4.2.

## 4.3 Model calibration

In a Bayesian setting, the epistemic uncertainty regarding the model parameters that have significant sensitivity indices can be reduced using experimental data. As discussed in Section 4.2, all material properties and model parameters, except α and $\beta ,$ are fixed at their nominal values. The measurement error is considered to be negligible since the measured bond lengths are precise to seven decimal points. Note that the material property $\alpha ,$ which is required for the heat transfer analysis, is not needed in case B, where experimentally measured temperature profile is the input to the sintering neck growth model instead of the heat transfer model prediction.

The posterior distribution for the sintering neck growth model parameter $\beta$ for case B (using observed temperature data as the input) is illustrated in Fig. 16. For case $\mathbf { A } ,$ where heat transfer model predictions are the input to the sintering neck growth model, the posterior distributions for α and $\beta$ are shown in Fig. 15.

![](images/365506479d40e3f3fe37eddd392b36576344045c05b53f85dfa739cc6170cd2d.jpg)  
(a)

![](images/ee7dd67b6874ba4f8d8e03c1b0c03edecc5a0302ed41151f68004e493a0ab234.jpg)  
(b)

Figure 15: Prior and posterior distributions of the material property α and model parameter $\beta$ considering the heat transfer model predictions as the input to the sintering neck growth model (case A)  
![](images/30f54d892c45d779ebf35b6c378ae46548b25b7033df9ead1a6e503e1382d321.jpg)  
Figure 16: Prior and posterior distributions of the model parameter $\beta$ using experimental temperature profile as the input to the sintering neck growth model (case B)

In order to calibrate these model parameters, 15,000 posterior samples are drawn using MCMC and the initial 5,000 samples are rejected (initial burn-in samples). The last 10,000 samples yield a mean value of 1. $. 1 9 6 \times 1 0 ^ { 6 }$ , and a coefficient of variation of 0.215 for $\alpha = \rho C$ and two different posterior distributions of $\beta$ with mean values of 0.00378 and 0.0193, and coefficient of variations of 0.1154 and 0.05 for case A and B respectively. The mean of the posterior distribution of $\beta$ for case B is close to the value reported in the literature [2]. However, for case A the uncertainty in α has a significant effect on the posterior distribution of $\beta$ by shifting its mean to a smaller value. These values are used in the subsequent analysis (i.e., in the surrogate models for the model errors and optimization under uncertainty).

## 4.4 Surrogate modeling

The surrogate models for the model discrepancy in cases A and B are built using the calibrated model parameters illustrated in Section 4.3. The GP model described in Section 3.3 is built with the training point inputs $[ \bar { T } _ { n } , v _ { p } , x , y ]$ (i.e., nozzle temperature, printer speed, and the location of midpoint of the intra-layer bonds, respectively), and the corresponding intra-layer bond length model discrepancy δ. Then, for a given combination of nozzle temperature and speed, the predicted model discrepancy is used to correct the bond length estimated by the neck growth model.

A series of experiments with different combinations of nozzle temperature and printer speed are used to measure the bond length, thus providing discrepancy data to train and test the above $\mathrm { G P } ^ { \mathrm { \ d } }$ surrogate model. The specimens were sectioned at the midpoint, i.e. $z _ { \mathrm { c u t } } = \dot { L } / 2 = 0 . 0 1 7 5$ m, and their cross-sections were analyzed under a digital microscope. The features of these cross-sections were analyzed using the image processing program ImageJ [34] (Fig. 17).

![](images/15b6bfbb74b437330779c7763e5103e44a02db19eb1e277fbdd86421ee020731.jpg)  
Figure 17: Cross-section view of a sample produced with $T _ { n } = 2 4 0 ^ { \circ } \mathrm { C } ,$ and $v _ { p } = 0 . 0 4 2$ m/s

For a sample that is printed with inputs $( T _ { n } , v _ { p } ) = ( 2 4 0 ^ { \circ } \mathrm { C } , 0 . 0 4 2 \mathrm { m } / \mathrm { s } )$ , the intra-layer bond lengths between adjacent filaments at $z _ { \mathrm { c u t } } = 0 . 0 1 7 5$ m of each layer are given in Table. 2. The numbering of the interfaces is done from left to right for all layers. For example, the label for the interface between 1st and 2nd filaments is 1 and the label for the interface between 89th and 90th filaments is 14.

Table 2: Intra-layer bond length measurements at each layer for $( T _ { n } , v _ { p } ) = ( 2 4 0 ^ { \circ } \mathrm { C } , 0 . 0 4 2 \mathrm { m } / \mathrm { s } )$
<table><tr><td>Layer</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td></tr><tr><td>1</td><td>0.4146</td><td>0.4558</td><td>0.5050</td><td>0.5050</td><td>0.5130</td><td>0.4339</td><td>0.4425</td><td>0.3915</td><td>0.3806</td><td>0.3930</td><td>0.3840</td><td>0.3798</td><td>0.3323</td><td>0.3162</td></tr><tr><td>2</td><td>0.4529</td><td>0.5056</td><td>0.5398</td><td>0.5357</td><td>0.5345</td><td>0.5050</td><td>0.3350</td><td>0.3293</td><td>0.3709</td><td>0.3833</td><td>0.4041</td><td>0.3854</td><td>0.2870</td><td>0.2606</td></tr><tr><td>3</td><td>0.4203</td><td>0.4828</td><td>0.5469</td><td>0.5517</td><td>0.5046</td><td>0.4889</td><td>0.3771</td><td>0.2269</td><td>0.2406</td><td>0.3271</td><td>0.3284</td><td>0.3361</td><td>0.2758</td><td>0.2506</td></tr><tr><td>4</td><td>0.4619</td><td>0.4895</td><td>0.5472</td><td>0.5944</td><td>0.5917</td><td>0.5196</td><td>0.3472</td><td>0.2055</td><td>0.2266</td><td>0.2608</td><td>0.3069</td><td>0.3316</td><td>0.2780</td><td>0.2781</td></tr><tr><td>5</td><td>0.4268</td><td>0.4798</td><td>0.5002</td><td>0.6125</td><td>0.5746</td><td>0.5667</td><td>0.4010</td><td>0.2617</td><td>0.1874</td><td>0.3071</td><td>0.2743</td><td>0.2913</td><td>0.2334</td><td>0.2621</td></tr><tr><td>6</td><td>0.4185</td><td>0.4694</td><td>0.4891</td><td>0.5334</td><td>0.4948</td><td>0.4750</td><td>0.3427</td><td>0.2617</td><td>0.2356</td><td>0.2886</td><td>0.2771</td><td>0.2670</td><td>0.2731</td><td>0.2884</td></tr></table>

Using Latin hypercube sampling, 25 sets of process parameters were generated, and experiments were conducted at these 25 values. The experimental data is divided into 2 sets, namely training set (20 specimens) and testing set (5 specimens). The training set is further subdivided into two subsets for cross-validation; k-fold cross-validation is performed by splitting the training set into 16 sets for model training and 4 sets for cross-validation, and these sets are selected randomly k = 5 different times. The bond length predictions using the observed temperature profiles are different than the ones using the heat transfer model predictions as inputs to the sintering neck growth model. Thus, two different GP models are built to represent the model error associated with these two cases, i.e., the bond length predictions using (A) heat transfer model predictions, and (B) observed temperature data. In each fold of training and cross-validation, the GP models are trained using 16 sets of data and the trained models are cross-validated using the remaining 4 sets of data. The average cross-validation accuracy of the GP models over the 5 folds (random shuffles) is assessed by evaluating the average mean squared error (MSE), which is found to be 0.0018 and 0.0017 for cases A and B respectively.

Further validation of the corrected bond length prediction model is done using testing data, i.e., 5 sets of experiments. The prediction accuracy of the corrected model for the test set is assessed by evaluating the mean squared error (MSE), which is found to be 0.0020 and 0.0019 for cases A and B respectively. MSE of 1% is used as the quantitative criterion for acceptance. The results indicate that the MSE values for both models are less than 1%, thus they are accepted for further analysis. The corrected bond length predictions at each interface based on case A and B are validated by comparing with the bond length measurements at each interface between adjacent filaments of the test set (see Figs. 18 and 19. Note that the points are close to the 45-degree line, showing good agreement between predictions and observations even for individual interfaces.

![](images/87beef5e2b5e447c7b457a2c79c7c10ce0373e2b9f244735ee710b70f89feb65.jpg)  
Figure 18: Validation of the bond length prediction model for case A, where x and y-axes represent the bond length predictions and experimental observations respectively at each interface between adjacent filaments

![](images/9a692e41ff1fc37e70ba8bfdaadfc082e7649c7599db27ca175b6c0334897cdf.jpg)  
Figure 19: Validation of the bond length prediction model for case B, where x and y-axes represent the bond length predictions and experimental observations respectively at each interface between adjacent filaments

## 4.5 Process design optimization

In this optimization, the process parameters (nozzle temperature and extrusion speed) are the design variables. The objective is to optimize the bond quality (indicated by bond length) at each layer of the specimen while satisfying the constraints on the design variables. The lower and upper bounds (LB and UB respectively) for the design variables are shown in Table 3. The lower and upper bounds have the same numerical values for all layers. The upper bound for the printer nozzle temperature was chosen as $2 6 0 ^ { \circ } \mathrm { C } ,$ , since the printer did not allow an extrusion temperature above $2 6 0 ^ { \circ } \mathrm { C }$ The lower bound for the printer nozzle temperature was chosen as $2 1 0 ^ { \circ } \mathrm { C }$ because the quality of the specimens reduced significantly below a nozzle temperature of $\mathrm { \bar { 2 } l 0 ^ { \circ } C }$ . The lower and upper bounds for the printer extrusion speed $v _ { p }$ were chosen as 0.015 and 0.043 m/s, respectively. We observed debonding and geometrical inaccuracies (warping) in the FFF parts that are printed using process parameter combinations above the linear fit. For example, a part printed with $2 2 0 ^ { \circ } \mathrm { C }$ and 40 mm/s resulted in debonding in several layers and interfaces. These experimental data points and a linear fit to these combinations of process parameters that result in poor bonding (bonding frontier) are plotted in Fig. 20. The overall bond quality of a part printed with parameter values above the bonding frontier is poor and delamination is observed. Whereas, the bond quality with parameter values below the bonding frontier is good, and gets better as the distance increases.

Table 3: Lower and upper bounds for the process design variables
<table><tr><td>Design variable</td><td>LB UB</td></tr><tr><td> $T _ { n } \ ( ^ { \circ } \mathrm { C } )$ </td><td>260</td></tr><tr><td> $v _ { p } \ ( \mathrm { m / s } )$ </td><td>210 0.015 0.043</td></tr></table>

![](images/05add104a7faa0adf96edaa2bafe91a9102b60ebd959d0f75e2459466259b3b1.jpg)  
Figure 20: Bonding frontier for Ultimaker 2 Extended +

The methodology proposed in Section 3.4 is implemented here for the part shown in Fig. 6a with $L = 0 . 0 3 5 { \mathrm { m } }$ . The calibrated values of the material property α and model parameter $\beta$ in case A are used in the optimization of the neck growth at each layer. In order to demonstrate the robustness of the proposed formulation, two cases are considered: (I) different printer nozzle temperature and extrusion speed values for each layer, and (II) same printer nozzle temperature and extrusion speed value for all layers. The optimal solutions for these two cases are presented in Table 4.

Table 4: Optimal printer nozzle temperatures (<sup>◦</sup>C) and extrusion velocities (m/s)
<table><tr><td colspan="3">Case I</td><td colspan="2">Case II</td></tr><tr><td>Layer</td><td> $T _ { n }$ </td><td> $v _ { p }$ </td><td> $T _ { n }$ </td><td> $v _ { p }$ </td></tr><tr><td>1</td><td>259.91</td><td>0.0430</td><td>259.30</td><td>0.0372</td></tr><tr><td>2</td><td>259.65</td><td>0.0162</td><td>259.30</td><td>0.0372</td></tr><tr><td>3</td><td>258.63</td><td>0.0150</td><td>259.30</td><td>0.0372</td></tr><tr><td>4</td><td>258.87</td><td>0.0150</td><td>259.30</td><td>0.0372</td></tr><tr><td>5</td><td>258.99</td><td>0.0150</td><td>259.30</td><td>0.0372</td></tr><tr><td>6</td><td>258.99</td><td>0.0150</td><td>259.30</td><td>0.0372</td></tr></table>

The optimal solutions are used to print three specimens for each case. The optimization results are then validated by comparing the bond length model predictions with the bond length measurements at each interface between adjacent filaments. The MSE value of 0.0027 is obtained based on the parts printed with the optimal solutions. Figure 21 shows the validation results for all six specimens, i.e., three specimens for each of Case I and II, printed with the corresponding optimum process parameter solution. The mean and standard deviation of bond lengths of each layer are averaged across the three specimens $( \overline { { \mu } } _ { \mathrm { B L } }$ and $\overline { { \sigma } } _ { \mathrm { B L } } )$ . The standard deviation across the averaged mean of bond lengths of three parts $\sigma _ { \mathrm { p a r t s } }$ are shown in Table 5 together with $\overline { { \mu } } _ { \mathrm { B I } }$ and $\overline { { \sigma } } _ { \mathrm { B L } }$ . It is seen in Table 5 that the variations across these specimens are relatively small. Note that the averaged mean of bond lengths $( \overline { { \mu } } _ { \mathrm { B L } } )$ ) across these three specimens i found to be smaller at the first layer than the other layers for the first case. The likely reasons for this difference are that the calibration/leveling of the build plate can be erroneous, and/or the printer extrusion speed decreases significantly (from 0.0430 m/s to 0.0162 m/s) as the second layer starts printing. These may result in excessive deformation on the first layer due to gravity and/or the weight of the material deposited above the first layer. As the bond quality starts reaching its upper limit (i.e., dimensionless neck radius $y / a = 1 )$ as shown in Fig. 22c, the differences regarding the bond length between the top and bottom layers become negligible. Whereas, in the second case, since the dimensionless neck radius is still relatively less than unity, the decrease in the neck growth in the top few layers is more prominent. This difference can be attributed to the fact that the top layer cools down more than the bottom layers because a larger surface area of the filaments in the top layer is exposed to the environmental temperature. This results in relatively poor bonding in the top layers. As it can be seen in Table 5, the overall bond quality of the part and the bond quality at each layer are significantly better for the first case since the printer nozzle temperature and extrusion speed are optimized at each layer separately.

Two specimens (sample 1 $( T _ { n } = 2 4 0 ^ { \circ } \mathrm { C }$ and $v _ { p } = 0 . 0 4 2 \ : \mathrm { m / s } )$ and sample $2 ( T _ { n } = 2 4 7 ^ { \circ } \mathrm { C }$ and $v _ { p } = 0 . 0 3 7 \ : \mathrm { m / s ) ) }$ that are printed with non-optimal process parameters are used to demonstrate the effect of the proposed methodology. The total void area and the overall mean bond length (BL) at $z _ { \mathrm { c u t } } = 0 . 0 1 7 5$ of case I and II are compared with sample

![](images/faa7677dbd06b7e66c0a27d14293e665fc708952d1564d0e9704e9e0f23241ea.jpg)  
Figure 21: Validation of the optimization results, where x and y-axes represent the bond length predictions and experimental observations respectively at each interface between adjacent filaments of Case I and II, respectively (three specimens printed for each case)

Table 5: Overall average bond length at optimal solutions (all units are in millimeters)
<table><tr><td></td><td colspan="3">Case I</td><td colspan="3"></td></tr><tr><td>Layer</td><td> $\overline { { \mu } } _ { \mathrm { B L } }$ </td><td> $\overline { { \sigma } } _ { \mathrm { B L } }$ </td><td> $\sigma _ { \mathrm { p a r t s } }$ </td><td> $\overline { { \mu } } _ { \mathrm { B L } }$ </td><td>σBL</td><td> $\sigma _ { \mathrm { p a r t s } }$ </td></tr><tr><td>1</td><td>0.56</td><td>0.0458</td><td>0.0479</td><td>0.58</td><td>0.0458</td><td>0.0049</td></tr><tr><td>2</td><td>0.65</td><td>0.0387</td><td>0.0292</td><td>0.59</td><td>0.0385</td><td>0.0053</td></tr><tr><td>3</td><td>0.65</td><td>0.0368</td><td>0.0146</td><td>0.59</td><td>0.0567</td><td>0.0022</td></tr><tr><td>4</td><td>0.64</td><td>0.0460</td><td>0.0086</td><td>0.57</td><td>0.0568</td><td>0.0149</td></tr><tr><td>5</td><td>0.66</td><td>0.0359</td><td>0.0068</td><td>0.55</td><td>0.0602</td><td>0.0129</td></tr><tr><td>6</td><td>0.65</td><td>0.0407</td><td>0.0048</td><td>0.52</td><td>0.0494</td><td>0.0081</td></tr></table>

![](images/0e954ed678176f5f790c85b5f50bb77b0850051da540e2cb9b17bb251fd3dc72.jpg)

![](images/9b674f6ae3a8a7b183d3dc064f39602e23ec2e66454799c5510f7389d0abaa5f.jpg)

![](images/9362d425373c1ac8171668a204c07da82720c8a779f6a6a92675f30d711e0d9d.jpg)

(c)  
![](images/ad0bc3588e897f383775893501f117fd75b6e611beae168283d4e8485adcbb9f.jpg)  
(d)  
Figure 22: Cross-section views of the parts at $z _ { \mathrm { c u t } } = 0 . 0 1 7 5$ m built with non-optimal and optimal process parameters: (a) sample 1, (b) sample 2, (c) case I and (d) case II

1 and sample 2 in Table 6. The cross-section views of these parts are shown in Fig. 22. The overall bond quality of sample 2 represented by total void area (0.54 mm<sup>2</sup>) and overall mean bond length metrics (0.52 mm) is better than the bond quality of sample 1 since the total void area and overall mean bond length of sample 1 are 2.11 mm<sup>2</sup> and 0.39 mm respectively (lower values of total void area and higher values of overall mean bond length imply a better-quality product). The total void area of the parts is identified with the use of microscopy images processed through the ImageJ software [34], combined with a Matlab script to estimate the size of voids. The total void area is the smallest and the overall mean bond length is the largest for case I as expected, thus demonstrating the effectiveness of the proposed optimization methodology.

Table 6: Total void areas and overall mean bond lengths at $z _ { \mathrm { c u t } } = 0 . 0 1 7 5 \ \mathrm { m }$
<table><tr><td>Metric</td><td>Case I</td><td>Case II</td><td>Sample 1</td><td>Sample 2</td></tr><tr><td>Total void area (mm²)</td><td>0.42</td><td>0.47</td><td>2.11</td><td>0.54</td></tr><tr><td>Overall mean BL (mm)</td><td>0.64</td><td>0.57</td><td>0.39</td><td>0.52</td></tr></table>

The coupled heat transfer and sintering neck growth model is directly used in the uncertainty quantification and optimization problems since the original simulation model is not very expensive (∼100 s for one run on Intel<sup>®</sup> Xeon<sup>®</sup> CPU E5-2650 v4@2.20GHz with 64 GB RAM desktop machine).

## 5 Conclusion

This paper developed a formulation for FFF process optimization under uncertainty, using an analytical solution for the transient heat transfer during filament deposition and cooling, and a sintering neck growth model. The neck growth between adjacent filaments is optimized while accounting for various sources of uncertainty and error. Variance-based sensitivity analysis is used to quantify the contribution of each uncertainty source to the variability of the output quantity (bond length). The physics model parameters that have the most significant contribution to the uncertainty in the mode output are calibrated using experimental measurement of bond length. Additional experimental observations are used to build a surrogate model for the physics model discrepancy in predicting the bond length. The surrogate model is used to estimate the model discrepancy for given process parameter values and correct the physics model predictions. The corrected prediction model is used to select the optimal process parameters to maximize the bond quality at each layer. The optimum solution is validated with specimens printed with the optimized process parameters.

The proposed approach helps to replace the trial-and-error experimental approach with a model-based process parameter optimization under uncertainty in FFF. In future work, the proposed framework needs to be extended to online control of process parameters, thus further reducing the variability in the bond quality and other quantities of interest in order to achieve various product quality objectives. Current work did not include the effect of filament age, percentage of humidity in the filament, and ambient air flow and humidity; future work could include these effects. Future work can also explore the generalization capabilities of the proposed method to parts with more filaments, different geometries and sizes.

## Acknowledgment

This study was supported by funds from the National Institute of Standards and Technology under the Smart Manufacturing Data Analytics Project (Cooperative Agreement No. 70NANB18H245). The support is gratefully acknowledged. At Vanderbilt University, valuable discussions with Dr. Pranav Karve and Dr. Paromita Nath, as well as support for experimental work by Garrett Thorne and Rich Tiesing are gratefully acknowledged.

## Disclaimer

Certain commercial equipment, instruments, or materials are identified in this paper in order to specify the experimental procedure adequately. Such identification is not intended to imply recommendation or endorsement by the National Institute of Standards and Technology, nor is it intended to imply that the materials or equipment identified are necessarily the best available for the purpose.

## References

[1] Nahal Aliheidari, Rajasekhar Tripuraneni, Amir Ameli, and Siva Nadimpalli. Fracture resistance measurement of fused deposition modeling 3d printed polymers. Polymer Testing, 60:94–101, 2017.

[2] Q Sun, GM Rizvi, CT Bellehumeur, and P Gu. Effect of processing conditions on the bonding quality of fdm polymer filaments. Rapid Prototyping Journal, 14(2):72–80, 2008.

[3] M Atif Yardimci, Selcuk I Guceri, Mukesh Agarwala, and Stephen C Danforth. Part quality prediction tools for fused deposition processing. In 1996 International Solid Freeform Fabrication Symposium, pages 539–548, 1996.

[4] M Atif Yardimci and Selçuk Güçeri. Conceptual framework for the thermal process modelling of fused deposition. Rapid Prototyping Journal, 2(2):26–31, 1996.

[5] JP Thomas and JF Rodríguez. Modeling the fracture strength between fused-deposition extruded roads. In 2000 International Solid Freeform Fabrication Symposium, pages 16–23, 2000.

[6] Longmei Li, Qian Sun, Celine Bellehumeur, and Peihua Gu. Investigation of bond formation in fdm process. Solid Freeform Fabrication Proceedings,(403), 400407, 2002.

[7] SF Costa, FM Duarte, and JA Covas. Thermal conditions affecting heat transfer in fdm/ffe: a contribution towards the numerical modelling of the process: This paper investigates convection, conduction and radiation phenomena in the filament deposition process. Virtual and Physical Prototyping, 10(1):35–46, 2015.

[8] SF Costa, FM Duarte, and JA Covas. Estimation of filament temperature and adhesion development in fused deposition techniques. Journal of Materials Processing Technology, 245:167–179, 2017.

[9] Ondˇrej Pokluda, Céline T Bellehumeur, and John Vlachopoulos. Modification of frenkel’s model for sintering. AIChE journal, 43(12):3253–3256, 1997.

[10] JJ Frenkel. Viscous flow of crystalline bodies under the action of surface tension. J. phys., 9:385, 1945.

[11] Céline Bellehumeur, Longmei Li, Qian Sun, and Peihua Gu. Modeling of bond formation between polymer filaments in the fused deposition modeling process. Journal of Manufacturing Processes, 6(2):170–178, 2004.

[12] Pavan Kumar Gurrala and Srinivasa Prakash Regalla. Part strength evolution with bonding between filaments in fused deposition modelling: This paper studies how coalescence of filaments contributes to the strength of final fdm part. Virtual and Physical Prototyping, 9(3):141–149, 2014.

[13] PA Kobryn and SL Semiatin. The laser additive manufacture of ti-6al-4v. JOM, 53(9):40–42, 2001.

[14] Jose F Rodriguez, James P Thomas, and John E Renaud. Characterization of the mesostructure of fused-deposition acrylonitrile-butadiene-styrene materials. Rapid Prototyping Journal, 6(3):175–186, 2000.

[15] Harry A Pierson and Bharat Chivukula. Process–property relationships for fused filament fabrication on preexisting polymer substrates. Journal of Manufacturing Science and Engineering, 140(8):084501, 2018.

[16] H-M Gutmann. A radial basis function method for global optimization. Journal of Global Optimization, 19(3):201–227, 2001.

[17] SF Costa, FM Duarte, and JA Covas. Towards modelling of free form extrusion: analytical solution of transient heat transfer. International Journal ofMaterial Forming, 1(1):703–706, 2008.

[18] Richard S Palais and Robert Andrew Palais. Differential equations, mechanics, and computation, volume 51. American Mathematical Soc., 2009.

[19] Marc C Kennedy and Anthony O’Hagan. Bayesian calibration of computer models. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 63(3):425–464, 2001.

[20] Paul D Arendt, Daniel W Apley, and Wei Chen. Quantification of model uncertainty: Calibration, model discrepancy, and identifiability. Journal ofMechanical Design, 134(10):100908, 2012.

[21] You Ling, Joshua Mullins, and Sankaran Mahadevan. Selection of model discrepancy priors in bayesian calibration. Journal ofComputational Physics, 276:665–680, 2014.

[22] W K Hastings. Monte carlo sampling methods using markov chains and their applications. Biometrika, 57(1):97– 109, 1970.

[23] George Casella and Edward I George. Explaining the gibbs sampler. The American Statistician, 46(3):167–174, 1992.

[24] Radford M. Neal. Slice sampling. The Annals ofStatistics, 31(3):705–767, June 2003.

[25] Erich Novak and Henryk Wo´zniakowski. Approximation of infinitely differentiable multivariate functions is intractable. Journal of Complexity, 25(4):398–404, 2009.

[26] Ilya M Sobol. Global sensitivity indices for nonlinear mathematical models and their monte carlo estimates. Mathematics and Computers in Simulation, 55(1-3):271–280, 2001.

[27] Andrea Saltelli and Stefano Tarantola. On the relative importance of input factors in mathematical models: safety assessment for nuclear waste disposal. Journal ofthe American Statistical Association, 97(459):702–709, 2002.

[28] Chenzhao Li and Sankaran Mahadevan. An efficient modularized sample-based method to estimate the first-order sobol’ index. Reliability Engineering & System Safety, 153:110–121, 2016.

[29] Bertrand Iooss and Mathieu Ribatet. Global sensitivity analysis of computer models with functional inputs. Reliability Engineering & System Safety, 94(7):1194–1204, 2009.

[30] Erin C DeCarlo, Sankaran Mahadevan, and Benjamin P Smarslok. Efficient global sensitivity analysis with correlated variables. Structural and Multidisciplinary Optimization, 58(6):2325–2340, 2018.

[31] Xiaoping Du and Wei Chen. Sequential optimization and reliability assessment method for efficient probabilistic design. Journal ofMechanical Design, 126(2):225–233, 2004.

[32] Kais Zaman, Mark McDonald, Sankaran Mahadevan, and Lawrence Green. Robustness-based design optimization under data uncertainty. Structural and Multidisciplinary Optimization, 44(2):183–197, 2011.

[33] Wei Chen, Janet K Allen, Kwok-Leung Tsui, and Farrokh Mistree. A procedure for robust design: minimizing variations caused by noise factors and control factors. Journal ofMechanical Design, 118(4):478–485, 1996.

[34] Caroline A Schneider, Wayne S Rasband, and Kevin W Eliceiri. Nih image to imagej: 25 years of image analysis. Nature methods, 9(7):671, 2012.