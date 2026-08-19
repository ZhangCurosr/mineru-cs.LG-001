# PHYSICS-INFORMED AND HYBRID MACHINE LEARNING IN ADDITIVE MANUFACTURING: APPLICATION TO FUSED FILAMENT FABRICATION \*

Berkcan Kapusuzoglu Department of Civil and Environmental Engineering Vanderbilt University Nashville, TN 37235, USA berkcan.kapusuzoglu@vanderbilt.edu

Sankaran Mahadevan

Department of Civil and Environmental Engineering Vanderbilt University Nashville, TN 37235, USA sankaran.mahadevan@vanderbilt.edu

## ABSTRACT

This paper investigates several physics-informed and hybrid machine learning strategies that incorporate physics knowledge in experimental data-driven deep learning models for predicting the bond quality and porosity of fused filament fabrication (FFF) parts. Three types of strategies are explored to incorporate physics constraints and multi-physics FFF simulation results into a deep neural network (DNN), thus ensuring consistency with physical laws: (1) incorporate physics constraints within the loss function of the DNN, (2) use physics model outputs as additional inputs to the DNN model, and (3) pre-train a DNN model with physics model input-output and then update it with experimental data. These strategies help to enforce a physically consistent relationship between bond quality and tensile strength, thus making porosity predictions physically meaningful. Eight different combinations of the above strategies are investigated. The results show how the combination of multiple strategies produces accurate machine learning models even with limited experimental data.

Keywords Additive manufacturing Fused filament fabrication Deep learning Physics-informed machine learning  Porosity

## 1 Introduction

Achieving the desired material properties and product quality in additive manufacturing (AM) processes has been studied using trial-and-error experiments as well as process models (either physics-based or machine learning (ML) models). In a trial-and-error approach, the AM process is repeated multiple times with different process parameter combinations to achieve the desired microstructure and properties of the manufactured parts; this is expensive and time-consuming. Moreover, this trial-and-error approach needs to be implemented every time a new design needs to be manufactured. Therefore, in recent years, research efforts have focused on model-based methods for optimizing the AM process parameters.

Several physics-based models have been developed depending on the AM process category and the quantity of interest (QoI) [1]. Costa et al. [2] proposed an analytical solution for transient heat transfer during the printing process in fused filament fabrication (FFF). Different models have been proposed in the literature to study polymer sintering [3, 4, 5, 6]. Many of these models are parametric representations of complex physical processes based on various approximations. The parameters of such physics-based models, as well as the model errors need to be calibrated for each AM process using available observation data to reduce the uncertainty in the model predictions [7, 8, 9]. Due to the complex physics of the AM process, a different model is needed for each sub-stage or phenomenon in the manufacturing process in order to accurately predict the QoI. Further, physics models with reasonable fidelity and accuracy require tremendous computational effort; as a result, the use of physics-based modeling in AM of realistic products has been challenging and limited.

Recently, several studies have used the available AM experimental data to build black-box ML models. In addition, with increased computing power, deep learning has become a prominent tool for solving classification and regression problems. In the context of AM, Khanzadeh et al. [10] compared supervised machine learning approaches to classify melt pools to predict porosity. Artificial neural network has been used to predict the geometry of a single bead in wire and arc additive manufacturing from the wire-feed rate and travel speed [11]. Kwon et al. [12] investigated the convolutional neural network (CNN) to predict laser power from melt pool images. Zhang et al. [13] used a CNN model to perform in-process porosity monitoring of laser-based AM processes.

Physics-based models do not require large amounts of data, but are generally limited by their computational complexity or incomplete physics. In contrast, ML models appear promising for complex systems that are not fully understood or represented with simplified relationships, given adequate quality and quantity of data. However, ML models represent the complex physics without taking into account any physical laws and thus can produce results that are inconsistent with physical laws. Karpatne et al. [14] proposed the combined use of physics-based and ML models to achieve more accurate and physically consistent predictions by leveraging the advantages of each method. In order to make ML models consistent with physical laws, Karpatne et al. [14] incorporated physical constraints into the loss function of ML models. Another method that is also implemented by Karpatne et al. [14] to combine physics-based and ML models is to use the physics-based model outputs as additional inputs in an ML model along with other inputs. Jia et al. [15] used synthetic data generated by executing physics-based models for multiple input combinations to pre-train the ML model in order to leverage the knowledge embedded in physics-based models. These general ideas have been explored in multiple engineering applications, such as geoscience, fluid dynamics, and thermodynamics, and this paper investigates these ideas for additive manufacturing.

The goal of this paper is to enhance the experimental data-driven ML models for AM by incorporating the physics knowledge, thus helping the ML model to produce more accurate and physically meaningful results. Three different strategies are explored for this purpose: (1) incorporate physics constraints within the loss function of the deep neural network (DNN), (2) use physics model outputs as additional inputs to the DNN model, and (3) pre-train a DNN model with physics model input-output and then update it with experimental data. Eight different combinations of the above three strategies are explored, and their performance in porosity and bond quality prediction of the FFF-produced parts is examined.

The strategies investigated in this paper help to better capture the physics of the AM process by leveraging physical laws while improving the generalization performance of data-driven AM models. Compared to previous studies, the proposed strategies address two different physical QoIs (neck diameter and porosity) and use multiple physics-based loss functions to enhance the ML model. The main contributions of this work are as follows:

1. Several physics-informed and hybrid machine learning models are developed for porosity prediction of FFF parts using physics constraints, physics-based model, and experimental data, using the three strategies mentioned above.

2. An enhanced physics-based model is developed to account for realistic filament geometry and the change of geometry during the printing process.

3. The proposed models are trained and evaluated using laboratory experiments where FFF parts are printed with varying input conditions, and data is collected to measure the quality characteristics (bond quality, porosity) of the parts.

The remainder of the paper is organized as follows. The proposed methodology is presented in Section 2, followed by the implementation of the methodology for a FFF process in Section 3. Concluding remarks are provided in Section 4.

## 2 Methodology

This section develops the proposed physics-informed machine learning (PIML) approaches to predict the bond formation and mesostructure in FFF-produced parts. The methodology is applicable to any AM process with corresponding data and physical laws for the prediction QoI. The three components of the methodology are: (a) Physics-based models, (b) Experiments, and (c) Construction of PIML models.

## 2.1 Physics-based models

This subsection describes the two coupled multi-physics models (thermal model and polymer sintering model) that predict the bond formation between adjacent filaments and the mesostructure of the printed part. The porosity and bond quality of an FFF part is dependent on the temperature history of filaments. Thus, it is important to predict the temperature evolution of filaments to estimate the final mesostructure of the printed part. The thermal model, based on the work by Costa et al. [2], is used to predict the temperature evolution of filaments considering the material properties, part geometry, and process parameters. The output of the heat transfer model (temperature) is input to the sintering model to predict the porosity and bond quality. A new method is then developed, which considers realistic filament geometry, and allows the filament geometry to change during the printing process, to compute the rate of polymer sintering and the final mesostructure of the printed part using the predicted temperature evolution of each filaments. Thus the mapping from input to output is a multi-physics model, i.e., models of two physical phenomena (heat transfer and sintering) are combined to predict the porosity and bond quality.

![](images/445555b4061fb16507ca33f8cfa47dd3bdac09fc3ca87a49b5a9e9573122c070.jpg)  
Figure 1. Schematic of FFF.

## 2.1.1 FFF temperature modeling

Fused filament fabrication (FFF) (also known as fused deposition modeling or FDM<sup>®</sup>) is an AM technology based on material extrusion for manufacturing polymer and plastic parts. The continuous strand of material is pushed through a heated nozzle and deposited as a molten extruded thin filament onto the build plate in a predefined path to form the part in the desired shape. The schematic of the FFF process is shown in Fig. 1.

In FFF, each filament is subjected to the same heat transfer mechanism with different boundary conditions depending on thermal conditions such as environment, build plate and extrusion temperature, as well as the part geometry, material properties and deposition sequence. Recently, Costa et al. [2] developed an analytical solution for the transient heat transfer during the print process in FFF. The temperature prediction model considers conduction heat transfer with the build plate and adjacent filaments based on the fraction λ of filament perimeter that is in contact, and convection heat transfer with the environment. The details of this heat transfer model are provided in the online supplementary material (refer to online supplementary material).

## 2.1.2 Bond formation modeling

This subsection develops a new method to compute the rate of polymer sintering and the final mesostructure of the printed part. The sintering process is defined as the coalescence of particles, in which two particles of molten polymer form a homogeneous melt, under the action of surface tension [3]. The sintering process for amorphous polymers is driven by the surface tension force since the mechanism is considered a Newtonian viscous flow [16].

A Newtonian sintering model for polymers was initially developed by Frenkel et al. [4] to predict the rate of polymer sintering. Pokluda et al. [3] developed a closed-form equation to predict the bond formation between two spherical particles based on the work balance of viscous dissipation and surface tension. Bellehumeur et al. [17] applied the model proposed by Pokluda et al. [3] to FFF for predicting the sintering between adjacent filaments as a nonlinear function of time, temperature-dependent surface tension Γ(T) and viscosity η(T), and an initial particle radius. The model has limitations related to the mesostructure of the filaments; specifically, the geometry of the filaments is assumed constant during the printing process. Based on the above discussion, in this paper we propose a new sintering model, which considers realistic filament geometry (similar to the one proposed by Garzon et al. [6]) and also accounts for the change in the mesostructure of the filaments during the printing process.

![](images/4841e986170b90e08735b43b7f35707827fb91671ae9c6ee1a972245dc9187c8.jpg)  
Figure 2. Evolution of neck diameter during sintering process.

The sintering process is simulated by considering two symmetrical adjacent filaments. In this work, all filaments are assumed to undergo the same sintering process by neglecting the effect of location. The cross-section geometry of a filament is composed of a rectangle with an initial width $a _ { 0 }$ and two half circles with a radius of $h _ { 0 } / 2$ at initial time $t = 0$ as proposed by Garzon et al. [6]. At $t = 0 ,$ , adjacent filaments have one contact point between them (see Fig. 2). The width of the rectangle $a ( t )$ evolves in time, together with $h ( t )$ such that width of each filament $w = a _ { 0 } ^ { - } + h _ { 0 } = a ( t ) + h ( t ) \cos ( \theta ( t ) )$ stays constant. The layer height $h ( t )$ is assumed to evolve in time based on the experimental observations, in which the average layer height for FFF parts has decreased more than the average width of filaments (see Fig. 3). During the sintering process, the width and length of each filament (w and L) are assumed to be constant. The coalescence between these two half circles forms the sintering angle $\theta ( t )$ and neck diameter $d ( t )$

![](images/a67847fb5fe6465ab046f48fc4c5e4f0f75a3e98cc35ce9fc900f4c826cfa523.jpg)  
Figure 3. Cross-sectional geometry of an FFF part printed with an extrusion temperature $2 4 0 ^ { \circ } \mathrm { C } ,$ speed 42 mm/s, initial layer height 0.7 mm, filament width 0.8 mm, filament length 35 mm, 6 number of layers and 15 filaments per layer.

In order to calculate the change in the filament geometry during the sintering process, the law of conservation of mass is expressed for two adjacent filaments that are assumed to have constant density:

$$
2 L \left( \pi \left( \frac { h _ { 0 } } { 2 } \right) ^ { 2 } + h _ { 0 } a _ { 0 } \right) = 2 L \left( \pi \left( \frac { h ( t ) } { 2 } \right) ^ { 2 } + h ( t ) a ( t ) - 2 \delta A \right) ,\tag{1}
$$

where $2 \delta A = 2 ( h ( t ) / 2 ) ^ { 2 } ( \theta ( t ) - \sin ( 2 \theta ( t ) ) )$ is the area of intersection between two filaments as shown in Fig. 2.

Physics-Informed and Hybrid Machine Learning in Additive Manufacturing: Application to Fused Filament Fabrication

Thus, the evolution of layer height is obtained as

$$
h ( t ) = \frac { - 2 w + \sqrt { Q } } { \pi - 2 \theta + \sin { ( 2 \theta ) } - 4 \cos { ( \theta ) } } ,\tag{2}
$$

where

$$
\begin{array} { r l } & { Q = h _ { 0 } ^ { 2 } \pi ^ { 2 } - 2 h _ { 0 } ^ { 2 } \pi \theta + h _ { 0 } ^ { 2 } \pi \sin { ( 2 \theta ) } - 4 h _ { 0 } ^ { 2 } \pi \cos { ( \theta ) } - 4 h _ { 0 } ^ { 2 } \pi } \\ & { \quad \quad + 8 h _ { 0 } ^ { 2 } \theta - 4 h _ { 0 } ^ { 2 } \sin { ( 2 \theta ) } + 1 6 h _ { 0 } ^ { 2 } \cos { ( \theta ) } + 4 h _ { 0 } \pi w } \\ & { \quad \quad - 8 h _ { 0 } \theta w + 4 h _ { 0 } w \sin { ( 2 \theta ) } - 1 6 h _ { 0 } w \cos { ( \theta ) } + 4 w ^ { 2 } . } \end{array}\tag{3}
$$

The work done by surface tension is defined as

$$
W _ { S } = - \Gamma ( T ) \frac { d S } { d t } ,\tag{4}
$$

where $S$ is the total surface of the filaments that undergo sintering process and is given as

$$
\begin{array} { l } { { S ( t ) = h ( t ) ^ { 4 } \left( - \theta + \displaystyle \frac { \sin { ( 2 \theta ) } } { 2 } \right) + h ( t ) ^ { 2 } \pi + 2 h ( t ) L \pi } } \\ { { - ~ 4 h ( t ) L \theta + 4 h ( t ) a ( t ) + 4 L a ( t ) . } } \end{array}\tag{5}
$$

Thus, applying the chain rule on Eq. (5), the work done by surface tension is obtained as

$$
W _ { S } = \Gamma ( T ) h ( t ) \left( 2 h ( t ) ^ { 3 } \sin ^ { 2 } { ( \theta ) } + 4 L \right) \theta ^ { \prime } .\tag{6}
$$

The work done by the viscous forces for a Newtonian fluid can be expressed as

$$
W _ { \nu } = \int \int \int _ { V } \eta ( T ) \nabla u : ( \nabla u + \nabla { u ^ { T } } ) d V ,\tag{7}
$$

V being the volume of the sintering system, and $\nabla u$ the gradient of velocity and is expressed as

$$
\nabla \boldsymbol { u } = \left[ \begin{array} { c c c } { \dot { \epsilon _ { 1 } } } & { 0 } & { 0 } \\ { 0 } & { \dot { \epsilon _ { 2 } } } & { 0 } \\ { 0 } & { 0 } & { \dot { \epsilon _ { 3 } } } \end{array} \right] .\tag{8}
$$

where $\dot { \epsilon _ { i } }$ is the strain rate in ith direction.

Assuming the deformations in width and length are negligible w.r.t. deformations in height, $\dot { \epsilon _ { 2 } } = \dot { \epsilon }$ is approximated by

$$
\dot { \epsilon } = \frac { \partial u _ { x } ( A ) } { \partial x } \approx \frac { u _ { x } ( A ) - u _ { x } ( O ) } { \overline { { O A } } _ { 0 } } = \frac { \frac { d } { d t } h ( t ) } { h _ { 0 } } .\tag{9}
$$

Consequently, the work done by the viscous forces can be defined as follows

$$
\begin{array} { l } { { \displaystyle W _ { \nu } = \int \int \int _ { V } 2 \eta ( T ) \dot { \epsilon } ^ { 2 } d V } } \\ { { \displaystyle \qquad = \frac { 2 L \eta ( T ) h ( t ) ^ { 2 } ( h _ { 0 } \pi - 4 h _ { 0 } + 4 w ) \sin ^ { 2 } { ( \theta ) } } { h _ { 0 } \cos ^ { 2 } { ( \theta ) } } \theta ^ { \prime \prime } } . }  \end{array}\tag{10}
$$

(11)

The evolution of the sintering angle $\theta ( t )$ is then obtained by equating the work done by surface tension and the viscous forces under the assumption that $\overline { { { \theta } ^ { \prime } } }$ is always positive [3]:

$$
\frac { d \theta ( t ) } { d t } = \frac { \Gamma ( T ) h _ { 0 } \left( h ( t ) ^ { 3 } + \frac { 2 L } { \sin ^ { 2 } \left( \theta ( t ) \right) } \right) \cos ^ { 2 } \left( \theta \right) } { L h ( t ) \eta ( T ) \left( h _ { 0 } \pi - 4 h _ { 0 } + 4 w \right) } .\tag{12}
$$

The evolution of neck diameter $d ( t )$ with time is then computed as

$$
d ( t ) = h ( t ) \sin ( \theta ( t ) ) ,\tag{13}
$$

The porosity can be expressed using the geometry of the mesostructure shown in Fig. 2, where the shaded grey region is the filled area and blue region is the void area. Thus, the evolution of porosity ϕ(t) is calculated as

$$
\phi ( t ) = \frac { h ^ { 2 } \cos \left( \theta \right) - \left( \frac { h } { 2 } \right) ^ { 2 } \left[ \pi - \left( 2 \theta - \sin \left( 2 \theta \right) \right) \right] } { h ^ { 2 } \cos \left( \theta \right) + h a } .\tag{14}
$$

The proposed methodology is an improvement upon the previous work on sintering models by Pokluda et al. [3] and Gurrala et al. [18], because (i) it considers a realistic filament geometry based on our experiments, and (ii) it accounts for changes in the filament geometry during the printing process. This proposed methodology is compared against experimental data in Section 3, and is found to give a smaller error than Gurrala et al.’s model. However, it still makes assumptions about the bonding process and sintering is not the only physical phenomenon that takes place during the bond formation. Thus, deep learning is studied to further enhance the prediction accuracy.

## 2.2 PIML for additive manufacturing

Although physics-based models predicting the temperature evolution, bond formation and mesostructure evolution of FFF parts are based on physical laws, they introduce bias due to incomplete representation of the complex physical process by approximating the reality. In addition, these models contain a significant number of model parameters that need to be calibrated using experimental data [9]. On the other hand, ML models are not aware of physical laws, which may result in physically inconsistent model predictions. However, they can extract complex physical relationships from available data. Thus, physics-based models and ML models can be integrated in an innovative manner to better capture the dynamics of the AM process.

In PIML models, physics knowledge and data are sought to be integrated in a synergistic manner by leveraging the complementary strengths of both models [14]. Thus, the goal is to improve the predictions beyond that of physics-based models or ML models alone by coupling physics-based models with ML models. In the following, three different strategies to combine physics knowledge and ML models are pursued: (1) incorporate physics constraints within the loss function of the DNN, (2) use physics model outputs as additional inputs to the DNN model, and (3) pre-train a DNN model with physics model input-output and then update it with experimental data.

## 2.2.1 Physics-informed loss functions

A direct strategy to improve ML model predictions is by including physics-based loss functions [14]. Consider a PIML model with inputs X and outputs $\hat { Y }$ trained using physical laws that are incorporated as constraints into the loss function:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { D N N } } ( \pmb { Y } , \hat { \pmb { Y } } ) + \sum _ { k = 1 } ^ { M } \lambda _ { \mathrm { p h y , k } } \mathcal { L } _ { \mathrm { p h y } , k } ( \hat { \pmb { Y } } ) ,\tag{15}
$$

where $\mathcal { L } _ { \mathrm { D N N } }$ is the regular training loss of a DNN that evaluates a supervised error (e.g., root mean squared error (RMSE); $\begin{array} { r } { \mathcal { L } _ { \mathrm { D N N } } ( \pmb { Y } , \hat { \pmb { Y } } ) = \sqrt { \sum _ { i = 1 } ^ { n } ( Y _ { i } - \hat { Y } _ { i } ) ^ { 2 } / n ) } } \end{array}$ , which measures how far off the predictions $\hat { Y }$ are from the observations Y for the n training samples, and $\mathcal { L } _ { \mathrm { p h y , k } }$ is the k-th physics-based loss function, whose contribution is controlled by a hyperparameter $\lambda _ { \mathrm { p h y , k } }$ and M is the total number of physics-based loss functions. The inclusion of $\mathcal { L } _ { \mathrm { p h y , k } }$ ensures physically consistent model predictions (the second term of Eq. (15) means physical inconsistency) and can decrease the generalization error even when there is a small amount of training data [14]. In addition, $\mathcal { L } _ { \mathrm { p h y , k } }$ does not require experimental observations; the data obtained from the physics model is used to evaluate physics-based loss functions.

In this work, we enforce five different physics-based loss functions (i.e., five separate physical relationships, $\mathcal { L } _ { \mathrm { p h y } , k } ( \hat { Y } )$ where $k = \{ 1 , 2 , 3 , 4 , 5 \}$ and $\hat { \mathbf { Y } } = ( \hat { Y } _ { 1 } , \hat { Y } _ { 2 } )$ are the overall dimensionless neck diameter (i.e., overall bond quality) and porosity predictions of FFF parts, respectively). These loss functions are defined as follows:

$$
\mathcal { L } _ { \mathrm { p h y } , 1 } ( \hat { Y } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { R e L U } ( - \hat { Y } _ { 1 , \mathrm { i } } ) ,
$$

$$
\mathcal { L } _ { \mathrm { p h y } , 2 } ( \hat { Y } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { R e L U } ( \hat { Y } _ { 1 , \mathrm { i } } - d _ { m a x } ) ,
$$

$$
\mathcal { L } _ { \mathrm { p h y } , 3 } ( \hat { Y } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { R e L U } ( - \hat { Y } _ { 2 , \mathrm { i } } ) ,
$$

$$
\mathcal { L } _ { \mathrm { p h y } , 4 } ( \hat { Y } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { R e L U } ( \hat { Y } _ { 2 , \mathrm { i } } - \phi _ { 0 , \mathrm { i } } ) ,
$$

$$
\mathcal { L } _ { \mathrm { p h y } , 5 } ( \hat { \mathbf { Y } } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { R e L U } ( \Delta _ { \mathrm { i } } ) ,\tag{16}
$$

where the first four loss functions consider the physical violations related to the overall dimensionless neck diameter and porosity across N samples and the fifth loss function represents the physical relationship between the mechanical properties and neck diameter. The physical inconsistencies in the model predictions are evaluated using these physicsbased loss functions. In the first and third loss functions, negative values of neck diameter and porosity are treated as physical violations. The second loss function evaluates physically inconsistent dimensionless neck diameter predictions which are greater than the maximum dimensionless neck diameter $d _ { m a x } = 1$ . The fourth loss function penalizes the model when porosity predictions $\hat { Y } _ { 2 , i }$ are greater than the initial porosity $\phi _ { 0 , i }$ of ith part. This is based on the physics knowledge that the total void area decreases as the sintering process takes place. The fifth physics-based loss function exploits the monotonic relationship between bond quality and tensile strength of FFF-produced parts. This loss function is constructed by computing the difference in the sorted dimensionless neck diameter predictions, $\hat { Y } _ { 1 , \mathrm { s o r t e d } }$ , and dimensionless neck diameter predictions corresponding to sorted tensile strength estimates $( \sigma _ { T S } ( \hat { Y } _ { 2 , \mathrm { s o r t e d } } , \pmb { \xi } ) ) , \hat { Y } _ { 1 , i } ^ { ' } ,$ i.e., $\Delta _ { i } = \hat { Y } _ { 1 , \mathrm { { s o r t e d } , i } } - \hat { Y } _ { 1 , i } ^ { ' }$ . The maximum stress for longitudinal raster orientation, $\sigma _ { T S } ( \hat { Y } _ { 2 } , \pmb { \xi } )$ with $\pmb { \xi } = \{ \sigma _ { 0 1 } , \sigma _ { 0 2 } , C _ { \sigma } \}$ being the material parameters, is computed according to the analytical expression proposed in Garzon et al. [6] using the porosity predictions:

$$
\sigma _ { T S } = \sigma _ { 0 1 } \left[ \exp \left( \big ( 1 - \hat { Y } _ { 2 } \big ) ^ { \mathrm { C } _ { \sigma } \mathrm { n } _ { 1 } } \right) - \hat { Y } _ { 2 } \right] + \sigma _ { 0 2 } ( 1 - \hat { Y } _ { 2 } ) ,\tag{17}
$$

where $n _ { l }$ is the number of layers of FFF parts. The tensile strength is constrained when porosity is equal to 0, i.e., $\sigma _ { T S } = \sigma _ { 0 1 } e + \sigma _ { 0 2 }$ . The overall average neck diameter and tensile strength are positively correlated, and tensile strength increases monotonically with neck diameter. Whereas, porosity and tensile strength are negatively correlated (as are porosity and neck diameter). More specifically, the model predictions $( \hat { Y } _ { 1 , i } , \hat { Y } _ { 2 , i } )$ and $( \hat { Y } _ { 1 , i + 1 } , \hat { Y } _ { 2 , i + 1 } )$ corresponding to ith and $( i + 1 ) !$ th FFF parts can be used to estimate $\sigma _ { T S , i }$ and $\sigma _ { T S , i + 1 }$ . If $\sigma _ { T S , i + 1 }$ is greater than $\sigma _ { T S , i } -$ meaning $( i + 1 )$ )th part has less voids than ith part $( \hat { Y } _ { 2 , i + 1 } < \hat { Y } _ { 2 , i } )$ then $\hat { Y } _ { 1 , i + 1 }$ should be greater than $\hat { Y } _ { 1 , i }$ as well by exploiting a key monotonic physical relationship between porosity and tensile strength of FFF-produced parts. Thus, with the inclusion of these physics-based penalty functions, the neck diameter and porosity predictions are ensured to be physically meaningful.

## 2.2.2 Physics model output as additional ML model input

A physics-based model $f ^ { \mathrm { p h y } }$ $\mathbf { \boldsymbol { X } } \to \hat { \mathbf { \boldsymbol { Y } } } ^ { \mathrm { p h y } }$ can be used to predict the QoI, where $\hat { \mathbf { Y } } ^ { \mathrm { p h y } }$ are predicted estimates of the true response of the system ${ \bf Y } , { \bf A }$ straightforward approach to combine physics-based and ML models is to use physics model output $\hat { \mathbf { Y } } ^ { \mathrm { p h y } }$ (at the experimental inputs) as additional input along with input $X ; { \mathrm { i . e . , } } f { \mathrm { ^ { h y b } : } } X { \mathrm { ^ { h y b } = } }$ $[ { \cal X } , \hat { \cal Y } ^ { \mathrm { p h y } } ]  \hat { \cal Y } ^ { \mathrm { h y b } }$

Adding the physics output as an extra input to the DNN model (which is trained using experimental data) is information fusion, where the physics model is an additional source of information that is consistent with physics (i.e., when the physics model satisfies the constraints mentioned in the previous subsection). The resulting DNN model can be thought of as a hybrid model that uses the experimental data to correct the output of the physics model which is an incomplete representation of the actual physics.

## 2.2.3 Pre-trained PIML model

In AM, especially in the FFF process with not a high-quality printer, parts have significant variability in quality. There is also uncertainty in measurement and lack of data due to the high cost associated with conducting experiments. Thus, data of adequate quality and quantity is important for good quality model predictions in AM.

In order to leverage the complex physical knowledge inherent in the physics-based models, synthetic data can be generated for multiple input combinations using physics-based models. The synthetic data can be used to train a ML model, which is used as the initial model to be updated with experimental data. The transfer of physical knowledge using a pre-trained ML model can prevent poor initialization due to lack of knowledge of initial choice of ML model parameters prior to training. This allows the pre-trained ML model to be fine-tuned even with limited observed data. In addition, it has been shown that using synthetic data from even imperfect physics models with uncalibrated model parameters can still reduce the amount of experimental training data needed [15].

More importantly, the pre-training can use a large amount of training data (with multiple input parameter combinations) over a wide range of values, which is not possible in experiments that could be expensive; as a result, the pre-training may help the eventual ML model to have wider generalization beyond experimental data. This is also an important distinction of the pre-training strategy from the second strategy. Both strategies use the physics model, but in the second strategy, the physics model is only used to provide outputs corresponding to the experimental inputs, whereas in the current pre-training strategy, the physics model is used to provide outputs corresponding to a much larger set of inputs. In the numerical example in Section 3, the pre-training strategy exercises the physics model over 1525 input combinations, whereas the second strategy above only employs the physics model over 39 experimental input combinations. However, the advantage of the pre-training strategy in using a larger input data set (for physics mode runs) compared to the experiments becomes limited if the physics model is computationally expensive.

In this work, the ML model is pre-trained using the outputs of an uncalibrated coupled multi-physics model (i.e., neck diameter and porosity). Further, the transfer of learned physical knowledge is shown to be valuable even when the input parameters of the synthetic data generated are quite different than the experimental observations. Once the ML model is pre-trained, it is fine-tuned using limited experimental observations. This helps to learn a 3D printer-specific physical process faster and with less samples.

The three proposed strategies to predict the QoIs are shown in Fig. 4. Figure 4(a) shows the first method, where the physical knowledge is included through constraints within the loss function of a DNN trained with experimental data. Figure 4(b) shows the second method, where the outputs of the physics model are additional inputs to the DNN model. Figure 4(c) shows the third method, where a DNN model is pre-trained with data generated using the physics-based model and then updated using experimental data. The proposed PIML strategies can be applied to any AM process by leveraging the physical constraints or physics-based models

![](images/17645098dc616a5a802a20662eb8d4409ee941a28727c0f9741201b361081c7f.jpg)  
Figure 4. PIML strategies: (a) incorporate physics constraints within the loss function of the DNN, (b) use physics model outputs as additional inputs to the DNN model, and (c) pre-training a DNN model with physics model input-output and updating it with experimental data.

## 2.2.4 Combination of PIML strategies

Based on the proposed three strategies to incorporate physics knowledge into the ML model, eight separate ML models can be constructed:

1. DNN

5. $\mathrm { D N N } ^ { \mathrm { h y b , \mathcal { L } _ { \mathrm { p h y } } } }$

2. $\mathrm { D N N } ^ { \mathcal { L } _ { \mathrm { p h y } } }$

6. $\mathrm { D N N ^ { u p d , h y b } }$

3. $\mathrm { D N N ^ { h y b } }$

7. $\mathrm { D N N } ^ { \mathrm { u p d , \mathcal { L } _ { \mathrm { p h y } } } }$

4. $\mathrm { D N N ^ { u p d } }$

8. DNN<sup>upd,hyb,L</sup>phy

In model 1, a deep neural network DNN is trained using only experimental data. The inputs X for this basic DNN model are the process parameters, printer extrusion temperature, speed, layer height, filament width, length, number of layers, and number of filaments per layer; and the outputs are overall dimensionless neck diameter and porosity. These inputs and outputs are the same as those used in the physics-based model $f ^ { \mathrm { p h y } }$ . Model $2 \mathrm { ( D N N } ^ { \mathcal { L } _ { \mathrm { p h y } } } \big )$ pursues the first strategy: physical knowledge related to the FFF process is included through constraints within the loss function of the DNN as shown in Eq. (15). Model 3 pursues the second strategy: a hybrid physics-based neural network $\mathrm { D N N ^ { h y b } }$ is trained using the outputs $\hat { \mathbf { Y } } ^ { \mathrm { p h y } }$ of $f ^ { \mathrm { p h y } }$ as extra inputs in addition to X, i.e., $\boldsymbol { X } ^ { \mathrm { h y b } } = [ \boldsymbol { X } , \hat { \boldsymbol { Y } } ^ { \mathrm { p h y } } ]$ . Model $4 ( \mathrm { D N N ^ { u p d } } )$ pursues the third strategy, where the weights and biases (model parameters) of all the layers excluding the input layer of the pre-trained network $f ^ { \mathrm { p r e } }$ (which is trained with the coupled multi-physics model input-output described in Section 2.2.3) are used as initial parameters for the DNN model and these parameters are updated with experimental data.

The architecture of the pre-trained model and the updated models (Model 4, 6, 7, and 8) is the same (except the input layer which changes with different numbers of inputs). The rest of the models (5-8) represent the combinations of the three strategies. Models 5, 6, and 7 each combine any two of the three strategies, whereas model 8 combines all three strategies. Model $5 \left( \mathrm { D N N } ^ { \mathrm { h y b } , \mathcal { L } _ { \mathrm { p h y } } } \right)$ combines the use of physics model outputs as additional inputs and the incorporation of physics constraints $\mathcal { L } _ { \mathrm { p h y } }$ within the loss function of the DNN. Model $\dot { 6 } ( \mathrm { { D N N } u p d , h y b } )$ ) combines second and third strategies, where the optimized model parameters of $f ^ { \mathrm { p r e } }$ are used as the initial values and the outputs $\hat { \mathbf { Y } } ^ { \mathrm { p h y } }$ of $f ^ { \mathrm { p h y } }$ are included as additional inputs. $\mathrm { D N N } ^ { \mathrm { a p d , h y b } }$ has the same number of inputs as $\mathrm { { D N N } ^ { h y b } \ ( i . e . }$ , both include the physics model output as additional input) and uses the optimized model parameters of all the layers excluding the input layer of $f ^ { \mathrm { p r e } }$ as initial parameters before updating with experimental data. Model $7 ( \mathrm { { D N N } u p d } , \mathcal { L } _ { \mathrm { { p h y } } } ^ { \bullet } )$ combines first and third strategies. The model parameters obtained from $f ^ { \mathrm { p r e } }$ are updated using the experimental data by minimizing the augmented loss function shown in Eq. 15. Model $8 ( \mathrm { D N N } ^ { \mathrm { u p d , h y b , \mathcal { L } _ { \mathrm { p h y } } } } )$ combines the use of physics model outputs as additional inputs to the updated DNN mode $\mathrm { D N N } ^ { \mathrm { u p d } }$ (which results in $\mathrm { D N N ^ { u p d , h y b } ) }$ and the physics constraints $\mathcal { L } _ { \mathrm { p h y } }$ are incorporated within the loss function of $\mathrm { D N N ^ { u p d , h y b } }$

## 3 Implementation of PIML to FFF

In this section we demonstrate the implementation of the proposed methodology to FFF-produced parts, and investigate the performance of eight PIML models described in Section 2.2.4. The results show that the proposed PIML models are capable of achieving physically meaningful and accurate model predictions, and require a smaller number of experiments.

## 3.1 Problem setup

First, data is collected from laboratory experiments in order to build the prediction models for the QoI using ML techniques. The shape of the part is conceptualized, a CAD model is visualized, and then sliced in a slicing software using the defined FFF process parameters and printing path.

The print quality depends on the adhesion of the first layer with the build plate [9]. Thus, several measures are needed to ensure proper adhesion. For example, the printing environment is modified by adding an enclosure to the 3D printer to isolate the printing environment from external effects. Kapton tape is used on the glass build plate to enhance the adhesion of Acrylonitrile butadiene styrene (ABS) with the build plate. After these modifications, the part is printed and then measured with appropriate monitoring techniques.

A commercial material Ultimaker Black ABS was extruded from an Ultimaker 2 Extended+ 3D printer to manufacture parts with unidirectionally aligned filaments. Using Latin hypercube sampling, 20 sets of process parameters are generated. The ranges considered for the variables are printer extrusion temperature $T _ { e } \colon ( 2 1 0 ^ { \circ } \mathbf { C } ^ { \bullet } - 2 6 0 ^ { \circ } \mathbf { C } )$ , and extrusion speed $S _ { e } \colon ( 1 5 \ \mathrm { m m / s } - 4 6 \ \mathrm { m m / s } )$ . Since the values of material properties $\boldsymbol { \xi }$ do not affect the outcome of $\mathcal { L } _ { \mathrm { p h y } , 5 } ( \hat { Y } )$ , the values calibrated by Garzon et al. [6] are used. All specimens were sectioned at the midpoint $z _ { \mathrm { c u t } } = L / 2$ (since the statistical properties of QoIs along the length of the specimens were constant) to analyze the mesostructural feature of interest with the use of microscopy images processed through the ImageJ software [19]. The collected experimental data is subsequently used to create DNN prediction models.

## 3.2 Model training and prediction

The eight DNN models were implemented using the Keras package [20] with Tensorflow backend. The pre-trained model, $f ^ { \mathrm { p r e } }$ , is first trained with physics model input-output data consisting of 1525 input parameter combinations over a range of experimental values, i.e., $( 2 1 0 ^ { \circ } C \leq T _ { e } \leq 2 6 0 ^ { \circ } C$ , 15 mm/s $\leq S _ { e } \leq 4 6$ mm/s ), and then updated for different combinations of the proposed strategies using observed data. (Note that in contrast only 39 physical experiments with 20 unique input parameter combinations are available, see Fig. 5). The input data of the training and test sets are normalized prior to the training of the DNN models (the output quantities are dimensionless and between 0 and 1), and the hyperparameters of these models are tuned with grid search $( \lambda _ { \mathrm { p h y } } = 0 . 3 , 0 . 3 , 0 . 1 5 , 0 . 1 5 , 0 . 0 0 8 )$ . Fully-connected

DNN models with 2 hidden layers and 10 neurons in each hidden layer are constructed and the weights of all neurons in these models are uniformly randomly initialized between 0 and 1. L1 and L2 regularizers are used as a penalty term to avoid overfitting. The Rectified Linear Unit (ReLU) activation function and Adam optimizer are used to perform stochastic gradient descent in learning the model parameters.

![](images/e73f49f3c63297f31fcc20f1080d9c4dedcdd34c440ec3576c5e5fb0834ccc36.jpg)  
Figure 5. Design of experiments for physics-based simulations and experiments.

The number of epochs for the convergence of training is approximately the same for each model except $f ^ { \mathrm { p r e } }$ , which converges in 40 epochs. The computation time for training of each model is on average 15 sec using a desktop computer $( \mathrm { I n t e l } ^ { \mathfrak { G } } \mathrm { X e o n } ^ { \mathfrak { G } }$ CPU E5-1660 v4@3.20GHz with 32 GB RAM and GPU NVIDIA Quadro K620 with 2 GB).

## 3.3 Model performance

In order to measure the accuracy of the trained DNN models, the model predictions of the test data are compared with the observed overall neck diameters and porosity. Each model is trained 30 times and compared against the FFF experimental data not used for training (i.e., data from 19 parts) to evaluate the mean and standard deviation of RMSE and physical inconsistency. The effect of the training data size on the RMSE of different models is shown in Table I. Here, the models are trained with 4, 6, 8, 10, and 20 experimental data points (i.e., 20%, 30%, 40%, 50%, and 100% of the available training data). The RMSE values of the coupled-physics model based on the sintering model developed by Gurrala et al. [18] and the new proposed one are 0.173 and 0.112, respectively, which are larger than the ML models due to the approximations used to represent the FFF process and bias in the model. The mean RMSE value of $f ^ { \mathrm { p r e } }$ which is trained with physics model input-output data consisting of 1525 input parameter combinations is 0.362. The RMSE value of $f ^ { \mathrm { p r e } }$ is greater than the RMSE of $f ^ { \mathrm { p h y } }$ because the pre-trained model is an approximation of the physics model, which causes uncertainty and bias. The RMSE of the basic DNN model is about 0.025 when we use all the parts in the training set. The RMSE of the basic DNN model is better than $f ^ { \mathrm { p r e } }$ and $f ^ { \mathrm { p h y } }$ because experimental data is directly used as the training data for the basic DNN model, whereas $f ^ { \mathrm { p r e } }$ and $f ^ { \mathrm { p h y } }$ use the approximate physics model to generate either pre-training data or additional input to the ML model.

The physics-based loss functions allow $\mathrm { D N N } ^ { \mathcal { L } _ { \mathrm { p h y } } }$ and $\mathrm { D N N } ^ { \mathrm { h y b , \mathcal { L } _ { \mathrm { p h y } } } }$ to achieve physically meaningful results with a lower value of average RMSE than $\mathrm { D N N ^ { h y b } }$ and improve the generalization performance. The $\mathrm { \bar { D } N N ^ { \mathrm { h y b } } }$ model performs similar to DNN, which shows that the physics model outputs do not improve the learning process significantly. The $\mathrm { D N N ^ { u p d } }$ and DNN<sup>upd,hyb</sup> models achieve a similar performance improvement as the DNN models that include physics-based loss functions. However, the models without physics constraints produce physically inconsistent results as shown in Table I. The results show that pre-training the PIML model improves the performance, and the improvement is relatively larger as the amount of observed data gets smaller. Additionally, the models that are pre-trained (DNN<sup>upd</sup> and $\mathrm { D N N } ^ { \mathbf { \bar { u p d } } , \mathbf { h } \mathbf { \breve { y } } b } )$ reach a physically more consistent initialization when they are updated with experimental data even without using physics constraints, compared to models that are not pre-trained, DNN and $\mathrm { D N N } ^ { \mathrm { h y b } }$ . The combination of all strategies $( \mathbf { \bar { D } N N } ^ { \mathrm { u p d , h y b } , \mathcal { L _ { \mathrm { p h y } } } } ,$ ) allows the model to get closest to the ground truth.

The prediction accuracy (w.r.t. neck diameter and porosity) of different models trained with 100% of the experimental data is shown in Fig. 6. The x and y-axis represent the physical inconsistency and the mean and standard deviation of RMSE, respectively. Figure 6 shows that models with physics-based loss functions produce physically consistent results. The incorporation of the physics knowledge using either the first or third strategy enables the models 2, 4-8 to generalize to configurations unseen in the training set.

Table I. Effect of different amounts of training data on the RMSE of different ML models
<table><tr><td>Model</td><td>20%</td><td>30%</td><td>40%</td><td>50%</td><td>100%</td><td>Mean Physical Inconsistency</td></tr><tr><td>fphy</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1. DNN</td><td> $0 . 1 0 1 ( \pm 0 . 0 2 2 )$ </td><td> $0 . 0 8 4 ( \pm 0 . 0 1 8 )$ </td><td> $0 . 0 4 4 ( \pm 0 . 0 1 7 )$ </td><td> $0 . 0 2 8 ( \pm 0 . 0 0 4 )$ </td><td>0.112  $0 . 0 2 5 ( \pm 0 . 0 0 5 )$ </td><td>0.000 0.201</td></tr><tr><td> $2 . \mathrm { D N N } ^ { \mathcal { L } _ { \mathrm { p h y } } }$ </td><td> $0 . 0 5 5 ( \pm 0 . 0 1 7 )$ </td><td> $0 . 0 5 0 ( \pm 0 . 0 1 1 )$ </td><td> $0 . 0 2 5 ( \pm 0 . 0 0 5 )$ </td><td> $0 . 0 2 4 ( \pm 0 . 0 0 7 )$ </td><td> $0 . 0 2 1 ( \pm 0 . 0 0 5 )$ </td><td>0.000</td></tr><tr><td>3.  $\mathrm { D N N ^ { h y b } }$ </td><td> $0 . 0 9 8 ( \pm 0 . 0 2 1 )$ </td><td> $0 . 0 7 8 ( \pm 0 . 0 2 4 )$ </td><td> $0 . 0 4 4 ( \pm 0 . 0 1 6 )$ </td><td> $0 . 0 2 8 ( \pm 0 . 0 0 7 )$ </td><td> $0 . 0 2 4 ( \pm 0 . 0 0 5 )$ </td><td>0.195</td></tr><tr><td> $4 . \mathrm { D N N ^ { u p d } }$ </td><td> $0 . 0 5 8 ( \pm 0 . 0 1 2 )$ </td><td> $0 . 0 4 9 ( \pm 0 . 0 1 1 )$ </td><td> $0 . 0 2 6 ( \pm 0 . 0 0 5 )$ </td><td> $0 . 0 2 5 ( \pm 0 . 0 0 5 )$ </td><td> $0 . 0 2 0 ( \pm 0 . 0 0 5 )$ </td><td>0.133</td></tr><tr><td>41  $\bar { 5 } . \ \mathrm { D N N } ^ { \mathrm { h y b , \mathcal { L } _ { \mathrm { p h y } } } }$ </td><td> $0 . 0 5 9 ( \pm 0 . 0 1 8 )$ </td><td> $0 . 0 4 1 ( \pm 0 . 0 1 2 )$ </td><td> $0 . 0 2 6 ( \pm 0 . 0 0 4 )$ </td><td> $0 . 0 2 3 ( \pm 0 . 0 0 4 )$ </td><td> $0 . 0 2 0 ( \pm 0 . 0 0 5 )$ </td><td>0.000</td></tr><tr><td>6.  $\mathrm { D N N ^ { u p d , h y b } }$ </td><td> $0 . 0 5 7 ( \pm 0 . 0 1 4 )$ </td><td> $0 . 0 4 7 ( \pm 0 . 0 1 1 )$ </td><td> $0 . 0 2 6 ( \pm 0 . 0 0 3 )$ </td><td> $0 . 0 2 6 ( \pm 0 . 0 0 5 )$ </td><td> $0 . 0 2 4 ( \pm 0 . 0 0 6 )$ </td><td>0.134</td></tr><tr><td>7.  $\mathrm { D N N } ^ { \mathrm { u p d , } \mathcal { L } _ { \mathrm { p h y } } }$ </td><td> $0 . 0 5 4 ( \pm 0 . 0 1 1 )$ </td><td> $0 . 0 4 5 ( \pm 0 . 0 1 3 )$ </td><td> $0 . 0 2 6 ( \pm 0 . 0 0 3 )$ </td><td> $0 . 0 2 5 ( \pm 0 . 0 0 5 )$ </td><td> $0 . 0 2 1 ( \pm 0 . 0 0 5 )$ </td><td>0.000</td></tr><tr><td>8.  $\mathrm { D N N } _ { \mathrm { N } } ^ { \mathrm { u p d , h y b , \it C _ { \mathrm { p h y } } } }$ </td><td> $0 . 0 5 3 ( \pm 0 . 0 1 3 )$ </td><td> $0 . 0 4 0 ( \pm 0 . 0 1 0 )$ </td><td> $0 . 0 2 5 ( \pm 0 . 0 0 4 )$ </td><td> $0 . 0 2 3 ( \pm 0 . 0 0 4 )$ </td><td> $0 . 0 1 8 ( \pm 0 . 0 0 3 )$ </td><td>0.000</td></tr></table>

![](images/6a1279b2c3ba18d00b9265cc4e3523a84476e43482aeb5f83e3db19eeeb4b75a.jpg)  
Figure 6. Performance and physical inconsistency of proposed models.

In order to further analyze the improvement in model predictions, the predicted dimensionless overall neck diameter and porosity for the test set that comprises 19 FFF parts are visualized in Fig. 7. Porosity predictions of model 1 (DNN) have large physical inconsistency (Fig. 7(a)). For instance, the blue triangle with the lowest porosity prediction has a negative value $( \mathrm { i . e . , } ( \hat { Y } _ { 1 } , \hat { Y } _ { 2 } ) = ( 0 . 5 2 , - 0 . 0 2 5 ) )$ . Model $2 \mathrm { ( D N N } ^ { \mathcal { L } _ { \mathrm { p h y } } } \big )$ predictions are physically consistent due to enforced physics constraints, but model $3 \mathrm { ( D N N ^ { h y b } ) }$ predictions also do not follow a monotonic decreasing relationship. Model 4 and $6 \mathrm { \ ( D N N ^ { u p d } }$ and $\mathrm { D N N ^ { u p d , h y b } ) }$ have some physically inconsistent predictions. Figure 7(b) shows that $\mathrm { D N N ^ { h y b , \mathcal { L } _ { \mathrm { p h y } } } } , \dot { \mathrm { D N N ^ { u p d , \mathcal { L } _ { \mathrm { p h y } } } } }$ and $\mathrm { D N N } ^ { \mathrm { u p d , i y b , } C _ { \mathrm { p h y } } }$ produce physically consistent model predictions, i.e., porosity and neck diameter predictions follow a monotonically decreasing relationship.

## 4 Conclusion

In this work, three strategies for physics-informed machine learning (PIML) are investigated for predicting the quality related metrics of FFF-produced parts. First, a physics-based sintering model is developed to predict the overall average neck diameter and porosity of FFF parts, using the temperature evolution of filaments, material properties, part geometry, and process parameters as inputs. The developed sintering model offers two improvements over existing models: (i) consideration of realistic filament geometry, and (ii) allowing the filament geometry to change during the printing process. Next, several PIML models are developed to predict the bond quality and porosity of FFF parts by leveraging three strategies for incorporating physics knowledge into the DNN model: (1) physics-based loss functions, (2) using the outputs of the coupled multi-physics model as additional inputs to the DNN, and (3) pre-training a DNN with data generated using physics-based model and then updating it with experimental data. The physics-based loss functions exploit the relationship between bond quality and tensile strength of the FFF parts.

![](images/8dba1bb90a5a898daacbcf140f51d77f4e97b8d1429fa4b32abe70c6783380b8.jpg)  
(a)

![](images/10caee52aa6c357fc0212833d41fc3caacb27d5c49e4f215122cd344ce0cd32c.jpg)  
(b)  
Figure 7. Comparison of model prediction with test data: (a) for the first 4 models, and (b) for the last 4 models.

The numerical results show that the incorporation of physics knowledge not only improves the prediction accuracy while producing physically meaningful results, but also allows accurate model predictions even with smaller amounts of experimental data. Thus, the proposed approach helps to fill the physics knowledge gap in the ML model while leveraging the capability of ML to extract complex process-material-geometry relationships in AM, and correcting for the approximation in the physics-based model.

Future work can include experimental data that consists of higher dimensional input, with multiple combinations, in order to further evaluate the performance of the proposed PIML strategies. Future work can also explore the generalization capabilities of the proposed strategies to parts of different geometry, as well as transfer learning to parts manufactured with different 3D printers and materials. Also, in the context of the third strategy, the data produced by experiments and physics models have different levels of credibility; thus the weighting of the two sources of data needs to be investigated in the future.

## Conflict of Interest

On behalf of all authors, the corresponding author states that there is no conflict of interest.

## References

[1] T. DebRoy, W. Zhang, J. Turner, and S.S. Babu. Building digital twins of 3D printing machines. Scripta Materialia, 135:119 – 124, 2017.

[2] S.F. Costa, F.M. Duarte, and J.A. Covas. Estimation of filament temperature and adhesion development in fused deposition techniques. Journal ofMaterials Processing Technology, 245:167 – 179, 2017.

[3] Ondˇrej Pokluda, Céline T. Bellehumeur, and John Vlachopoulos. Modification of Frenkel’s model for sintering. AIChE Journal, 43(12):3253–3256, 1997.

[4] JJ Frenkel. Viscous flow of crystalline bodies under the action of surface tension. J. phys., 9:385, 1945.

[5] R W Hopper. Coalescence of two equal cylinders: exact results for creeping viscous plane flow driven by capillarity. J. Am. Ceram. Soc.; (United States), 67:12, 12 1984.

[6] S. Garzon-Hernandez, D. Garcia-Gonzalez, A. Jérusalem, and A. Arias. Design of FDM 3D printed polymers: An experimental-modelling methodology for the prediction of mechanical properties. Materials & Design, 188:108414, 2020.

[7] You Ling, Joshua Mullins, and Sankaran Mahadevan. Selection of model discrepancy priors in Bayesian calibration. Journal ofComputational Physics, 276:665 – 680, 2014.

[8] Paromita Nath, Zhen Hu, and Sankaran Mahadevan. Uncertainty quantification of grain morphology in laser direct metal deposition. Modelling and Simulation in Materials Science and Engineering, 27(4):044003, apr 2019.

[9] Berkcan Kapusuzoglu, Matthew Sato, Sankaran Mahadevan, and Paul Witherell. Process Optimization under Uncertainty for Improving the Bond Quality of Polymer Filaments in Fused Filament Fabrication. Journal of Manufacturing Science and Engineering, pages 1–46, 08 2020.

[10] Mojtaba Khanzadeh, Sudipta Chowdhury, Mohammad Marufuzzaman, Mark A. Tschopp, and Linkan Bian. Porosity prediction: Supervised-learning of thermal history for direct laser deposition. Journal of Manufacturing Systems, 47:69 – 82, 2018.

[11] Donghong Ding, Zengxi Pan, Dominic Cuiuri, Huijun Li, Stephen van Duin, and Nathan Larkin. Bead modelling and implementation of adaptive mat path in wire and arc additive manufacturing. Robotics and Computer-Integrated Manufacturing, 39:32 – 42, 2016.

[12] Ohyung Kwon, Hyung Giun Kim, Wonrae Kim, Gun-Hee Kim, and Kangil Kim. A convolutional neural network for prediction of laser power using melt-pool images in laser powder bed fusion. IEEE Access, 2020.

[13] Bin Zhang, Shunyu Liu, and Yung C. Shin. In-process monitoring of porosity during laser additive manufacturing process. Additive Manufacturing, 28:497 – 505, 2019.

[14] Anuj Karpatne, William Watkins, Jordan Read, and Vipin Kumar. Physics-guided Neural Networks (PGNN): An Application in Lake Temperature Modeling. arXiv e-prints, page arXiv:1710.11431, October 2017.

[15] Xiaowei Jia, Jared Willard, Anuj Karpatne, Jordan S Read, Jacob A Zwart, Michael Steinbach, and Vipin Kumar. Physics-Guided Machine Learning for Scientific Discovery: An Application in Simulating Lake Temperature Profiles. arXiv e-prints, page arXiv:2001.11086, January 2020.

[16] N. Rosenzweig and M. Narkis. Sintering rheology of amorphous polymers. Polymer Engineering & Science, 21(17):1167–1170, 1981.

[17] Céline Bellehumeur, Longmei Li, Qian Sun, and Peihua Gu. Modeling of Bond Formation Between Polymer Filaments in the Fused Deposition Modeling Process. Journal ofManufacturing Processes, 6(2):170 – 178, 2004.

[18] Pavan Kumar Gurrala and Srinivasa Prakash Regalla. Part strength evolution with bonding between filaments in fused deposition modelling. Virtual and Physical Prototyping, 9(3):141–149, 2014.

[19] Caroline A Schneider, Wayne S Rasband, and Kevin W Eliceiri. NIH Image to ImageJ: 25 years of image analysis. Nature methods, 9(7):671–675, 2012.

[20] Francois Chollet et al. Keras, 2015.