# MULTI-OBJECTIVE OPTIMIZATION UNDER UNCERTAINTY OF PART QUALITY IN FUSED FILAMENT FABRICATION

Berkcan Kapusuzoglu<sup>†,1</sup>, Paromita Nath<sup>1</sup>, Matthew Sato<sup>1</sup>, Sankaran Mahadevan<sup>1</sup>, Paul Witherell<sup>2</sup>

<sup>1</sup>Department of Civil and Environmental Engineering, Vanderbilt University, Nashville, TN 37235, USA <sup>2</sup>Systems Integration Division, Engineering Laboratory,

National Institute of Standards and Technology (NIST), Gaithersburg, MD 20899, USA

## ABSTRACT

This work presents a data-driven methodology for multi-objective optimization under uncertainty of process parameters in the fused filament fabrication (FFF) process. The proposed approach optimizes the process parameters with the objectives of minimizing the geometric inaccuracy and maximizing the filament bond quality of the manufactured part. First, experiments are conducted to collect data pertaining to the part quality. Then, Bayesian neural network (BNN) models are constructed to predict the geometric inaccuracy and bond quality as functions of the process parameters. The BNN model captures the model uncertainty caused by the lack of knowledge about model parameters (neuron weights) and the input variability due to the intrinsic randomness in the input parameters. Using the stochastic predictions from these models, different robustness-based design optimization formulations are investigated, wherein process parameters such as nozzle temperature, nozzle speed, and layer thickness are optimized under uncertainty for different multi-objective scenarios. Epistemic uncertainty in the prediction model and the aleatory uncertainty in the input are considered in the optimization. Finally, Pareto surfaces are constructed to estimate the trade-offs between the objectives. Both the BNN models and the effectiveness of the proposed optimization methodology are validated using actual manufacturing of the parts.

Keywords additive manufacturing · fused filament fabrication · robust design optimization · multi-objective optimization · optimization under uncertainty · process design · Bayesian neural network (BNN) · machine learning

## 1 Introduction

Additive manufacturing (AM) is the layer-wise material deposition process to manufacture parts in the desired shape from a computer aided design (CAD) model [1]. Although AM is a rapidly expanding technology and has shown immense potential in several industries, application of AM has been limited due to variability in the product quality. Thus, improving the product quality and reducing the variability in AM parts is an active area of research. The traditional trial-and-error approach of optimizing process parameters based on physical experiments is expensive and time-consuming. Therefore, recently there is increasing interest in using model-based methods to study AM and optimize the process parameters. Both physics-based models and data-driven models have been developed to better understand the AM process [2].

Developing physics-based models is a major direction in AM research. Several physics-based models have been developed depending on the AM process category and the quantity of interest (QoI) [3]. Since AM is a complicated process, a different model may be required to represent a particular sub-stage or phenomenon in the manufacturing process. Multiple physics-based models at different length and time scales are coupled together to simulate the AM process. Sophisticated commercial software are often used to solve the governing equations, and these tend to be computationally demanding. Thus, physics-based modeling of full-scale AM parts has been challenging [4].

The potential of data-driven models for quality monitoring and control in AM is also being explored by the AM community [5]. Data-driven models have been shown to perform well and generally do not require expert, in-depth knowledge of the complex physics of the AM process [6]. Thus, studies have been conducted on using the available experimental data to build data-driven models of the AM process. Khanzadeh et al. [7] compared supervised machine learning approaches such as Decision Tree, K-Nearest Neighbor, Support Vector Machine, Linear Discriminant Analysis, and Quadratic Discriminant Analysis to classify melt pools for porosity prediction. With the evolution of data collection techniques and instruments, more and more data is now available in the form of images such as thermal images, optical images, and profilometer data, in addition to the data from instruments such as thermocouples, digital calipers, etc. Meanwhile, with the advancement in computer hardware, researchers also have access to increased computing power. This has led to a growing trend of using artificial neural networks (ANN) as the preferred tool for solving classification and regression problems to handle the large amounts of data.

ANN has been used to predict the geometry of a single bead in wire and arc additive manufacturing (WAAM) from the wire-feed rate and travel speed [8]. An ANN-based prediction model was developed to predict strain recovery rates and transformation temperatures of fabricated metal parts for given laser power, laser speed, and hatch spacing of a selective laser melting (SLM) AM process [9]. Ye et al. [10] proposed an adapted deep belief network approach using the plume and spatter signatures obtained during an SLM process to detect defects in the part. Gaikwad et al. [11] built a deep convolutional neural network (CNN) model with in-situ layer-wise optical images as input to predict the quality in terms of statistical features obtained from offline X-ray computed tomography (XCT) of the manufactured part. Kwon et al. [6] adapted CNNs to predict laser power from melt pool images.

However, the introduction of process models (either physics-based or data-driven) introduces several sources of uncertainty such as model parameter uncertainty, input uncertainty, model error, etc., which is then propagated to the QoI predicted by the model [12, 13, 14]. Data-driven models are created with data collected from experiments. The uncertainty in the measurement process from instruments, human error, etc. needs to be considered. It is also important to select the optimal model and tuning parameters of the model (e.g., number of layers and units in a deep neural network), and avoid data overfitting. Often the amount of data available to construct the data-driven model is limited, leading to uncertainty in the model prediction. Thus, it is necessary to consider the model uncertainty for an accurate and reliable prediction model. The Bayesian approach has been extensively used to characterize the epistemic uncertainty in the model. Several studies with respect to physics-based AM models have employed Bayesian strategies for model parameter estimation [15, 16, 13]. However, when the number of parameters to be calibrated is large, use of traditional Bayesian inference for model parameter estimation becomes computationally unaffordable.

Most of the data-driven prediction models in the AM literature have been used for analyzing and understanding the AM process and the relationship between the process parameters and the QoI. Some studies such as Ding et al. [8] used an ANN model to optimize the WAAM process parameters for arriving at the desired bead geometry. However, in AM, several quality characteristics may need to be maximized for a part to be acceptable. For example, a part may need to be geometrically accurate and at the same time have high mechanical strength. Sometimes these objectives are conflicting in nature, e.g., attempting to minimize voids in the part may result in part geometry distortion. Also, since process variability is a major concern and there is model uncertainty in predicted the QoI, it is necessary to optimize the process parameters for multiple objectives while also considering these uncertainty sources.

Based on the above discussion, a data-driven approach for multi-objective optimization under uncertainty of FFF process parameters is developed in this paper, by specifically considering model uncertainty and input variability. The main contributions of this work are as follows: (1) Experiments are conducted in the laboratory to print parts with different process parameter values, and data is collected to measure two quality characteristics of the parts. (2) Data-driven Bayesian neural network (BNN) prediction models (one for each quality characteristic) are constructed by implementing Monte Carlo (MC) dropout in the neural network models to quantify the uncertainty in the model prediction as well as the input variability due to the intrinsic randomness in the process parameters. (3) The trained BNN models are then used to find the optimal values of process parameters, and several multi-objective problem formulations are investigated for robust design optimization (RDO).

The remainder of the paper is organized as follows. The proposed methodology is presented in Section 2, followed by the implementation of the methodology for an FFF process in Section 3. Section 4 provides the concluding remarks.

this section, we present the methodology for process parameter optimization under uncertainty with a focus on the FFF process. However, the proposed methodology is applicable to any AM process with corresponding data and QoI prediction models. The three components of the methodology are: (a) Data collection, (b) Construction of data-driven prediction models including uncertainty, and (c) Multi-objective optimization under uncertainty.

## 2 Methodology

FFF (also known as fused deposition modeling or $\mathrm { F D M } ^ { \mathfrak { ( B ) } } )$ is an AM technology based on material extrusion [1] generally used for manufacturing polymer and plastic parts. A spool of the material in filament form is attached to the AM machine or 3D printer. The continuous strand of material is fed to the heated nozzle which then extrudes the material in a predefined path to form the part in the desired shape. The schematic of the FFF process is shown in Fig. 1. In

![](images/39adbd018c7653a9aeb39fdd58e760432f85cb90164add48f28770ddf5c87b88.jpg)  
Figure 1: Schematic of FFF

## 2.1 Data collection

For a data-driven methodology, the first step is to collect data to build the prediction models for the QoI. Traditionally, data is collected from physical experiments. With the advent of physics-based modeling, some researchers also use data generated by the physics-based model [17]. In that case, the physics-based model prediction should be first validated with experiments to ensure that the training data closely approximates the actual physics of the AM process. In some cases, data available from previous studies or in the public domain might also be used [18, 19, 20, 14]. For this work, we collected data from laboratory experiments to build the prediction models. The shape of the part is conceptualized, and a CAD model is first built and then sliced in a slicing software where the printing path is also defined. The printing instructions thus generated are used to print the part.

Various sensors can be used to monitor the AM process and parts. The most common measurement QoIs are melt pool temperature, part dimension, surface roughness, microstructure, tensile properties, etc. Monitoring can be broadly divided into online and offline monitoring techniques. Online monitoring refers to in situ monitoring of the part during the manufacturing process in order to implement process control. Effective online monitoring is not disruptive to the ongoing process, and is non-destructive to the part such as monitoring of the temperature profile using an infrared (IR) camera, or monitoring of the part geometry using profilometer or optical camera. Offline or ex situ monitoring of the part after it has been produced may be either destructive (such as microstructure characterization with scanning electron microscope) or non-destructive (such as part dimension measurement using calipers). Both contact and non-contact techniques may be used for monitoring. For example, temperature data can be collected through contact thermocouples or non-contact IR cameras. Thus, depending on the QoI and the monitoring mode, the instruments and experimental setup are determined. The collected data is then used to construct the prediction models. In this paper, the QoIs are part thickness and bond quality, both of which are measured offline after the part is manufactured.

## 2.2 Construction of data-driven prediction model

There are several machine learning techniques to build prediction models using data. Commonly used approaches include Decision Tree, K-Nearest Neighbor, Support Vector Machine, Linear Discriminant Analysis, Quadratic Discriminan Analysis, Artifical Neural Network (ANN), Polynomial Chaos Expansion, Radial Basis Function, Gaussian Process Modeling, Random Forest Regression etc. As the amount of data available has increased, use of ANN has gained popularity; several types of ANN are available, such as deep neural network (DNN), recurrent neural network (RNN)

to handle time series data, and convolution neural network (CNN) to handle image data. Machine learning (ML) models appear promising for complex systems that are not fully understood or cannot be represented with simplified physics-based relationships, given adequate quality and quantity of data. Generally, the construction of data-driven ML models does not require in-depth knowledge of the complex physics inherent in the physical process [2]. In this work, adequate experimental data is available to build a deep learning (DL) model based on the observation data, thus we pursue the ML approach. In addition, we quantify the DL model uncertainty by constructing Bayesian neural network (BNN) models to predict the QoIs and use these BNN models to perform optimization under uncertainty. In the next sections, we introduce the underlying mechanisms in the feedforward neural network and Bayesian neural network (BNN).

## 2.2.1 Neural networks

Feedforward neural network is the simplest type of ANN. It has four major components: neuron, activation function, cost function, and optimization. The connections between the units of the feedforward neural networks do not form a cycle. The information moves only forward from the input layer through the hidden layers and to the output layer as shown in Fig. 2. The values of the various input variables of a particular neuron are multiplied by their associated weights, then the sum of the products of the neuron weights and the inputs are calculated at each neuron. The summed value is passed through an activation function that maps the summed value into a fixed range before passing these signals on to the next layer of neurons.

![](images/75db737b504069e10de1aa3ffa99b0b9946b821b69b764ed02248edf9cd1bb6d.jpg)  
Figure 2: A feedforward neural network with n hidden layers

The predictions of the feedforward neural network after a forward propagation are compared against the actual values by defining a loss function, which measures how far off the predictions are form the observations for the training samples. After the forward propagation, backpropagation algorithms are performed to keep track of small perturbations to the weights that affect the error at the output and to distribute this error back through the network layers by computing gradients for each layer using the chain rule. In order to minimize the value of the loss function, necessary adjustments are applied at each iteration to the neuron weights in each layer of the network. These procedures are performed at each iteration until the loss value converges to a stable value.

ANNs are able to learn powerful representations that can map high dimensional data to an array of outputs. However, deterministic neural networks do not allow for estimating the uncertainty in their predictions. Model prediction uncertainty is important to account for in AM. In addition, there is process variability, uncertainty in measurement, and also limited data due to the high cost associated with conducting experiments especially for metal AM. The quantification of model uncertainty using the Bayesian method is discussed next.

## 2.2.2 Bayesian neural network

In the Bayesian context, the distributions of the model parameters (neuron weights w) that are most likely to generate the observed data are inferred. A prior distribution over the neuron weights $p ( \mathbf { w } )$ is defined, and $\mathbf { a }$ likelihood function $p ( \mathbf { Y } | \mathbf { X } , \mathbf { w } )$ is defined to represent the probability of generating the observed data given model parameters. Bayesian neural network (BNN) implementations so far have used Gaussian prior distributions $p ( \mathbf { w } = \bar { \mathcal { N } } ( 0 , \mathcal { T } ) )$ to replace the deterministic network’s weight parameters [21, 22, 23], thus representing the epistemic uncertainty in the DNN model. Following Bayes’ theorem, a posterior distribution over the model parameters given the training set $\{ { \bf X } , { \bf Y } \} = \{ \{ { \bf x } _ { 1 } , . . . , { \bf x } _ { N } \} , \{ \{ { \bf y } _ { 1 } , . . . , { \bf y } _ { N } \} \}$ is defined by

$$
p ( \mathbf { w } | \mathbf { X } , \mathbf { Y } ) = \frac { p ( \mathbf { Y } | \mathbf { X } , \mathbf { w } ) p ( \mathbf { w } ) } { p ( \mathbf { Y } | \mathbf { X } ) } .\tag{1}
$$

After inferring the posterior distributions of $p ( \mathbf { w } )$ , the predictive distribution of the model output for a new input $\mathbf { x } ^ { * }$

$$
p ( { \mathbf { y } ^ { * } | { \mathbf { x } ^ { * } } , \mathbf { X } } , { \mathbf { Y } } ) = \int _ { \Omega } p ( { \mathbf { y } ^ { * } | { \mathbf { x } ^ { * } } , \mathbf { w } } ) p ( { \mathbf { w } | { \mathbf { X } } , \mathbf { Y } } ) d { \mathbf { w } } .\tag{2}
$$

The posterior distribution of model parameters $p ( \mathbf { w } | \mathbf { X } , \mathbf { Y } )$ cannot be evaluated analytically over the whole parameter space Ω due to the highly non-linear behavior in the neural network caused by the non-linear activation functions and their combinations across multiple hidden layers. Thus, it becomes difficult to perform exact analytical inference in BNNs.

Several approximate inference techniques are available to infer the posterior distribution $p ( \mathbf { w } | \mathbf { X } , \mathbf { Y } )$ , such as variational inference [24], Approximate Bayesian Computation [25], Markov Chain Monte Carlo (MCMC) [26], and Particle Filter [27] methods. Markov Chain Monte Carlo (MCMC) can be a more accurate technique than the other methods, but it is computationally too expensive. It would not be affordable to perform MCMC to estimate posterior distributions of the deep neural network model parameters (i.e., weights) since there are usually a large number of parameters in the model. Variational inference (VI) fits a simple and tractable distribution $q _ { \boldsymbol { \theta } } ( \mathbf { w } )$ to the posterior, parametrized by a variational parameter θ [24]. This approximates the intractable problem by optimizing the parameters of $q _ { \boldsymbol { \theta } } ( \mathbf { w } )$ The accuracy of the variational distribution is often measured by the Kullback-Leibler (KL) divergence between the approximate distribution $q _ { \boldsymbol { \theta } } ( \mathbf { w } )$ and the true model posterior $p ( \mathbf { w } | \mathbf { X } , \mathbf { Y } )$

Gal and Ghahramani [28] showed that Monte Carlo (MC) dropout is equivalent to performing approximate VI; the former infers the posterior by performing dropout not only while training a model but also during prediction. In MC dropout, randomly chosen neurons are temporarily removed from the network along with their connections. Next, the gradients of neuron weights are calculated on each smaller neural network and these gradients are then averaged over the training sets to obtain the weights of overall network. The construction of BNNs with MC dropout builds on the concept of dropout as regularization on neural networks. However, in contrast to standard neural networks, MC dropout performs dropout and generates random samples following a Bernoulli distribution for each neuron in the input and hidden layers during prediction. The dropout is applied to the neuron that takes the value 0 with a given dropout probability $p _ { d } .$ . The outputs of the network are predicted using the collection of generated random samples from the posterior predictive distribution and the uncertainty in the prediction is quantified. The computational effort for the construction of BNNs with MC dropout is comparable to the standard neural networks. Moreover, the simplicity of the MC dropout strategy provides an efficient way of Bayesian inference to quantify the model prediction uncertainty with a variety of neural networks, such as DNN, CNN, and RNN.

## 2.2.3 Uncertainty quantification in Bayesian neural network

Various sources of uncertainty can be considered, such as (a) Epistemic uncertainty due to lack of knowledge, and (b) Aleatory uncertainty due to the inherent variability across multiple samples and over space and time. Epistemic uncertainty is caused by insufficient knowledge or information about the model and data. In the case of a DNN model, the values of the model parameters such as neuron weights are estimated from the data. If limited data is available, there is epistemic uncertainty in the model prediction. BNN is used to describe the epistemic uncertainty caused by the model by placing distributions over the network weights.

Aleatory uncertainty is caused by the natural variability in the AM process, leading to variability in the process output QoI. For example, the AM parts printed with the same process parameters may have different geometric dimensions and mechanical properties. The observation noise parameter σ also needs to be tuned. Similar to the procedure in Bayesian model calibration, observation noise can be learned through the BNN model by minimizing the following loss function [29]:

$$
\mathcal { L } _ { B N N } ( \theta ) = \frac { 1 } { N } \sum _ { i } \frac { 1 } { 2 } \hat { \sigma } _ { i } ^ { - 2 } | | \mathbf { y } _ { i } - \hat { \mathbf { y } } _ { i } | | ^ { 2 } + \frac { 1 } { 2 } \mathrm { l o g } \hat { \sigma } _ { i } ^ { - 2 } + \lambda g ( \mathbf { w } )\tag{3}
$$

where N is the number of observations ${ \bf y } _ { i } ; \hat { { \bf y } } _ { i }$ and $\hat { \sigma } _ { i } ^ { 2 }$ are the predictive mean and observation noise corresponding to input indexed by i, the first term represents the norm of residuals for measuring goodness of fit; λ is the weight decay parameter, and $g ( \mathbf { w } )$ represents a regularization function for the weights (or the penalty term which often uses $L _ { 2 }$ regularization). The second regularization term prevents the network from predicting infinite uncertainty, thus zero loss. The neuron weights can be drawn from the approximate posterior wˆ $\sim q ( \mathbf { w } )$ to obtain the model outputs of a BNN denoted as $\mathbf { f } ^ { \hat { w } } ( \mathbf { x } ) = [ \hat { \mathbf { y } } , \hat { \sigma } ^ { 2 } ]$ . The second term of the loss function can be regarded as an uncertainty regularization term and it prevents the network from predicting infinite uncertainty. In order to have a numerically stable network during training, the log variance log $\hat { \sigma } _ { i } ^ { 2 }$ is predicted instead of predicting $\hat { \sigma } _ { i } ^ { 2 }$ . (Sometimes the process parameter settings specified by the designer (such as the nozzle temperature, nozzle speed, layer thickness, etc.) may not be actually realized in manufacturing (this is input uncertainty (epistemic), i.e., the input value specified in the model is different from what is actually in the manufacturing process, and we do not know what the actual value is). However, this type of uncertainty is not considered in this paper).

The predictive mean and variance are estimated by collecting the results of stochastic forward passes through the model. The mean prediction of the model with T MC samples can be approximated by

$$
\mathbb { E } ( \mathbf { y } ) \approx \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \hat { \mathbf { y } } _ { t } ,\tag{4}
$$

and the variance of the prediction is estimated by

$$
\mathrm { V a r } ( \mathbf { y } ) \approx \underbrace { \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \hat { \mathbf { y } } _ { t } ^ { 2 } - \left( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \hat { \mathbf { y } } _ { t } \right) ^ { 2 } } _ { \mathrm { m o d e l u n c e r t a i n t y } } + \underbrace { \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \hat { \sigma } _ { t } ^ { 2 } } _ { \mathrm { n o b s e r v a t i o n } } .\tag{5}
$$

In the next section we discuss how to use the model predictions given by $\operatorname { E q . }$ 4 and Eq. 5 for optimization under uncertainty.

## 2.3 Multi-objective optimization under uncertainty

As mentioned earlier, often in AM it is necessary to optimize the process parameters for multiple objective functions. A generic formulation of deterministic multi-objective optimization for $n _ { o b j }$ objectives may be written as:

$$
\begin{array} { r l } { \underset { \mathbf { x } } { \mathrm { m i n i m i z e } } } & { \left\{ \left. f _ { 1 } ( \mathbf { x } , \mathbf { p } ) , . . . , f _ { n _ { o b j } } ( \mathbf { x } , \mathbf { p } ) \right\} \right. } \\ { \mathrm { s u b j e c t ~ t o } } & { g _ { i } ( \mathbf { x } , \mathbf { p } ) \leq 0 , i = 1 , 2 , . . . , n _ { c o n } } \\ & { \left. \mathbf { l b } _ { \mathbf { x } } \leq \mathbf { x } \leq \mathbf { u b } _ { \mathbf { x } } \right. } \end{array}\tag{6}
$$

where x and p are the design and non-design variables, $g _ { i } , i = 1 , 2 , . . . , n _ { c o n }$ represent the $n _ { c o n }$ deterministic constraints, and $\mathbf { l b _ { x } }$ and $\mathbf { u b } _ { \mathbf { x } }$ are the lower and upper bounds for the design variables x.

## 2.3.1 Formulation of multi-objective optimization

The objective functions in Eq. 6 depend on the intended use of the additively manufactured part. Since optimization involves evaluation of the objective function repeatedly at different process parameter settings, instead of conducting expensive physical experiments the objective functions are evaluated using the data-driven model discussed in Section 2.2. The prediction from the BNN model has a mean given by Eq. 4 and the corresponding variance given by Eq. 5. The conventional deterministic optimization does not consider the uncertainty in the input variables, data and model parameters; the output of the manufacturing process is sensitive to these uncertainty sources. Optimization under uncertainty can be pursued in two directions: (1) robust design optimization (RDO) [30], and (2) reliability-based design optimization (RBDO) [31]. In RDO, both the mean and the variability of the objective function are optimized (since minimizing the variability makes the objective insensitive to variations of the input variables and parameters), and the constraints are satisfied within specified uncertainty bounds. On the other hand, in RBDO, a desired target level of reliability is maximized (i.e., probability of satisfying a desired threshold of performance or quality) by optimizing the decision variables, or a cost function is minimized while satisfying a reliability constraint. In this work we are interested in a robust design to improve the quality of products and processes by optimizing both the mean and variance of the quantities of interest. Therefore, robust design optimization (RDO) is used in this study for the design of the FFF process parameters. (Note also that for well-designed practical systems, the probability of failure would be very low; thus the RBDO formulation would require substantially more function evaluations in comparison to the RDO formulation where the means and variances of the objective function and the constraints can be evaluated with a much smaller number of function evaluations). The robustness of the objective function can be achieved by simultaneously optimizing the mean and variance. Thus RDO even with respect to a single objective QoI becomes a bi-objective optimization problem with two objectives: (a) Optimize the mean of the QoI, and (b) Minimize the variance of the QoI. The resulting bi-objective RDO problem can be approximately solved through a single objective formulation, using a weighted sum approach, i.e., mean and variance terms have weights that reflect the designer’s preference.

Consider the case where the objective is to minimize a function $f ( \mathbf { x } , \mathbf { p } )$ with design variables $\mathbf { x } = [ \mathbf { x } _ { d } , \mathbf { x } _ { \theta } ]$ and non-design variables $\mathbf { p } = [ \mathbf { p } _ { d } , \mathbf { p } _ { \theta } ]$ , where $\mathbf { x } _ { d } , \mathbf { p } _ { d }$ are deterministic variables, and $\mathbf { x } _ { \boldsymbol { \theta } } , \mathbf { p } _ { \boldsymbol { \theta } }$ are stochastic design and non-design variables respectively. Since the prediction from the BNN model is stochastic, the single objective RDO formulation using weighted sum approach can be written as

$$
\begin{array} { r l } { \underset { \mathbf { x } \in \mathbb { R } ^ { n _ { x } } } { \mathrm { m i n i m i z e } } } & { w _ { 1 } \mu _ { f } ( \mathbf { x } , \mathbf { p } ) + w _ { 2 } \sigma _ { f } ( \mathbf { x } , \mathbf { p } ) ; } \\ { \mathrm { s u b j e c t ~ t o } } & { \mathbf { l b _ { g } } + \mathbf { k _ { c } } \sigma _ { \mathbf { g } } \mathbf { g } ( \mathbf { x } , \mathbf { p } ) \le \mathbf { g } ( \mathbf { x } , \mathbf { p } ) \le \mathbf { u b _ { g } } - \mathbf { k _ { c } } \sigma _ { \mathbf { g } } \mathbf { g } ( \mathbf { x } , \mathbf { p } ) , } \\ & { \mathbf { l b _ { x } } \le \mathbf { x } \le \mathbf { u b _ { x } } } \end{array}\tag{7}
$$

where $\mu _ { f } ( \cdot )$ and $\sigma _ { f } ( \cdot )$ are the mean and standard deviation of $f , 0 \leq w _ { 1 } , w _ { 2 } \leq 1$ are the weights representing the relative importance of each objective function, $\mathbb { R } ^ { n _ { x } }$ represents the design space, $\mathbf { l b } _ { \mathbf { x } } , \mathbf { u b } _ { \mathbf { x } }$ represent the lower and upper bounds for the design variables, ${ \bf g } ( { \bf x } , { \bf p } )$ is the vector of inequality constraints, and $\mathbf { l b } _ { \mathbf { g } } , \mathbf { u b } _ { \mathbf { g } }$ represent the lower and upper bounds of the constraints. For stochastic constraints, the feasible region is reduced by $k _ { c } \sigma _ { g }$ in each direction where $\mathbf { k _ { c } }$ is a vector of user-defined constants based on the design requirements and $\sigma _ { \mathbf { g } }$ is the vector of standard deviations of the constraints [30]. For a deterministic constraint, $k _ { c } \sigma _ { g } = 0$ . Note that in this paper, the optimization needs to consider multiple objectives, considering multiple QoIs. The same approach in the above equation can be extended for $n _ { o b j }$ objectives by assigning corresponding weights $w _ { 1 } , w _ { 2 } , \cdot \cdot \cdot , w _ { n _ { o b j } }$ to the objective functions.

The computation of objectives $f _ { j } ( \mathbf { x } , \mathbf { p } ) ( j = 1 , \ldots , n _ { o b j } )$ and constraints ${ \bf g } ( { \bf x } , { \bf p } )$ is affected by uncertainty sources such as input variability and model uncertainty; thus we have stochastic objectives and constraints. The input variability is already present in the experimental data that is used for training the DL model, and further, the model uncertainty (epistemic) is also captured through the BNN approach. The aleatory uncertainty is learned by minimizing the loss function shown in Eq. 3. MC dropout, which is performing dropout during prediction, is used to quantify the epistemic uncertainty in the model prediction. As a result, the output of the BNN model is stochastic, which incorporates both sources of uncertainty mentioned above. The optimization formulation in Eq. 7 properly accounts for the resulting uncertainty. In the next section, we discuss the solution of the optimization problem.

## 2.3.2 Solution of multi-objective optimization

Depending on the design variables and constraints, the optimization problem discussed in Section 2.3 can be solved using a suitable global optimization algorithm. In this paper, we choose to employ Non-dominated Sorting Genetic Algorithm II (NSGA-II) [32].

Non-dominated Sorting Genetic Algorithm II

Since a multi-objective problem is formulated, the solution depends on the weighting coefficients. The trade-off among $n _ { o b j }$ objective functions $f _ { k } , \ k = 1 , 2 , . . . , n _ { o b j }$ can be represented by the Pareto front. The Pareto front is a set of Pareto optimal solutions, which correspond to the solutions for different values of the weighting coefficients for the optimization problem specified in Eq. 7. The NSGA-II algorithm is one of the frequently applied multi-objective optimization evolutionary algorithms for generating the Pareto front [32].

The NSGA-II procedure for finding the Pareto front can be briefly described in following steps:

1. An initial random population is generated and the fitnesses of the individuals are evaluated during several generations;

2. Several Pareto fronts are generated by ranking the population based on the non-dominating sorting criteria (where individuals with the best rank represents the first front, the ones with the second best rank generate the second front and so on);

3. The crowding distance value is assigned to each front once the sorting is completed;

4. The individuals are selected using a binary tournament selection with crowded-comparison operator (if two individuals have the same rank, the individual with greater crowding distance is selected to increase the diversity, otherwise an individual with a better rank is chosen);

5. Binary crossover and polynomial mutation are used to generate a new offspring population combined with the current population;

6. The next generation is set by selection until the population size exceeds the current population; and

7. The above steps are repeated until the stopping condition is met and a set of non-dominated Pareto optimal solutions are obtained.

The Pareto optimal set is the set of all possible Pareto optimal vectors and there is a need for a stopping criterion that evaluates the quality of the Pareto front solutions. There are methods that apply a stopping criterion, such as NSGA-II [32]. Another important concept is the ideal stopping generation criterion, where the solution is as close to the optimal Pareto front as possible in a short amount of generation. The ideal stopping generation is a compromise between the distance to the Pareto optimal front and the cost of generation. Therefore, it is important to be able to determine how good a solution is compared to the optimal one by using an indicator, such as hypervolume indicator [33], and epsilon indicator [34, 35].

## Evaluating performance: hypervolume indicator

Among the variety of performance indicators for genetic algorithm in optimization, the hypervolume indicator has been favored by many researchers to measure the quality of a solution [33, 36]. The hypervolume indicator can capture the closeness of the solutions to the Pareto optimal set and partially the spread of the solutions across the objective space.

The hypervolume indicator, $\mathcal { T } _ { h v } ( \mathcal { A } )$ , computes the volume of the region, H, defined by a set of reference or nadir points and given set of points, $\mathcal { N }$ and A, respectively as

$$
\mathcal { T } _ { h v } ( \boldsymbol { A } ) = \mathrm { V o l u m e } \Bigg ( \bigcup _ { \forall a \in \boldsymbol { A } ; \forall n \in \mathcal { N } } \mathrm { h y p e r c u b e } ( a , n ) \Bigg ) ,\tag{8}
$$

where larger values of $\because T _ { h v } ( \mathcal { A } )$ corresponds to better solutions. The absolute performance of an optimization algorithm is measured using nadir points, which are the worst elements of the Pareto front solutions.

The performance of an algorithm as the evolution proceeds can be tracked by transforming the indicator [34]:

$$
\mathcal { T } _ { h v } ( t ) = \mathcal { T } _ { h v } ( \mathcal { P } _ { t } ) - \mathcal { T } _ { h v } ( \mathcal { P } _ { t - 1 } ) ,\tag{9}
$$

where $\mathcal { P } _ { t - 1 }$ and $\mathcal { P } _ { t }$ are the previous and current non-dominated elements of the local Pareto optimal front.

## 2.3.3 Experimental validation

The subsection discusses two types of validation as part of the methodology: first the prediction model is validated against experimental data, and second, the optimization solution is also validated against experimental data. (Note that there is also cross-validation during the model construction phase, by partitioning the data into training and testing sets, in order to check that the model has the required accuracy, as explained in Section 3.3).

## BNN model validation

Quantitative comparison of model prediction against experimental data is traditionally done using hypothesis testing. In recent model validation literature, several quantitative methods have been pursued, such as classical hypothesis testing, Bayesian hypothesis testing, reliability-based method, area-metric-based method, etc. Any of these methods can be used to quantitatively validate the optimization results [37]. The classical t-test is used in this paper to test the null hypothesis that the mean of observations is equal to the mean of the model prediction. The t-test is based on Student’s t-distribution and the corresponding test statistic t is

$$
t = { \frac { { \overline { { Y } } } _ { D } - \mu _ { m } } { s _ { D } / { \sqrt { n } } } }\tag{10}
$$

where $\overline { { Y } } _ { D }$ is the sample mean of experimental observations, $\mu _ { m }$ is the mean prediction, and $s _ { D }$ is the sample standard deviation. The p-value $( \mathrm { i . e . , } p = 2 F _ { T , n - 1 } ( - | t | )$ , where $F _ { T , n - 1 }$ is the cumulative distribution function (CDF) of a t-distribution with (n-1) degrees of freedom) is compared with the significance level α (usually 0.01 or 0.05). If the p-value is less than or equal to α, then the null hypothesis is rejected, otherwise the null hypothesis is not rejected.

## Optimization solution validation

Since we are using an approximate surrogate model, we perform the second validation to check whether the propagation of surrogate model error through the optimization process has significantly affected the optimum solution. Validation of the optimization methodology can be achieved by comparing the performance of the optimum solution against other randomly selected settings for the decision variables. In this paper, the Pareto optimal solutions obtained using the proposed methodology are validated by conducting actual printing experiments at the optimal process parameter settings and other random settings. The performance comparison is done by comparing the quality objectives achieved in the actual manufactured parts, such as mean bond length, mean part thickness, and their standard deviations. Section 3.4 considers four combinations of these individual objectives. In each case, it is investigated whether the optimum solution gives better part quality than randomly selected process parameter settings.

## 2.4 Summary of methodology

The steps of the proposed methodology for multi-objective optimization of AM process parameters under uncertainty are summarized in Fig. 3. The proposed method consists of five main components: (a) Collection of experimental

data, (b) Construction of data-driven probabilistic prediction models, (c) Validation of the trained BNN models, (d) Multi-objective optimization under uncertainty of process parameters, and (e) Validation of the optimization results.

![](images/e58717c081dfc2ee9236737ce87664bb1a65112517d2c60d9178547e89b66d72.jpg)  
Figure 3: Flowchart of the proposed methodology

The methodology presented in this paper can be generalized for any AM process, as long as experimental data is available to build the data-driven model. In the next section, we demonstrate the effectiveness of the proposed methodology for four different cases with multiple objectives.

## 3 3D printing example

In this section we demonstrate the implementation of the proposed methodology on a part of dimensions 35 mm $\times \ 1 2$ mm × 4.2 mm printed with ABS (acrylonitrile butadiene styrene) using an Ultimaker S5 printer [38]. The objective is to find the optimal process parameters nozzle temperature $( T _ { e } )$ , nozzle speed (V<sub>e</sub>), and layer height (l<sub>t</sub>), i.e., $\mathbf { \boldsymbol { x } } = [ T _ { e } , V _ { e } , l _ { t } ]$ such that both the dimensional accuracy and bond quality of the part are maximized. Since the maximum extrusion volume for the 0.8 mm diameter nozzle used in the experiments is $2 4 \mathrm { m m ^ { 3 } / s }$ , an inequality constraint $g = V _ { e } \times l _ { t } \times L _ { w } \leq 2 4$ mm<sup>3</sup>/s, where $L _ { w } = 0 . 8$ mm is the raster width, is added to the optimization problem.

## 3.1 Data collection from experiments

In this work, we use ABS and modify the printing environment by adding an enclosure to the 3D printer to isolate the printer from environmental effects (see [13, 2, 12, 15] for details). Using Latin hypercube sampling (Fig. 5), 25 sets of process parameters x are generated and three parts are printed at each parameter setting. The ranges considered for the variables are: $T _ { e } : 2 1 5 ^ { \circ } \mathrm { C }$ to $2 8 0 ^ { \circ } \mathrm { C } , V _ { e }$ : 25 mm/s to 45 mm/s, and $l _ { t } \overset { \cdot } { = } \{ 0 . 4 2 , 0 . 6 0 , 0 . 7 0 \}$ mm.

![](images/fb4aef1a98c7f0b1ae685c49fd718e19534922abbe8af10b0dd0449184945db8.jpg)  
Figure 4: Deposition sequence of unidirectional filaments

Since the focus is on maximizing the dimensional accuracy and bond quality, data pertaining to these two quality characteristics are collected. The measure for dimensional accuracy is considered to be the error in part thickness, i.e., the difference between the printed part thickness and the target part thickness of 4.2 mm. As shown in Fig. 6, using a laser displacement sensor Keyence LK-H057 [39] the part thickness is measured at $z = \{ 7 , 1 2 , 1 7 , 2 2 , 2 7 \}$ mm at discrete points along the x-axis. The bond lengths (mesostructural feature of interest) between the filaments were measured at cross-section $z _ { \mathrm { c u t } } = L / 2 = 1 7 . 5$ mm as shown in Fig. 4 with the use of microscopy images processed through the ImageJ software [40].

The bond length measurements at each interface of a part, which is printed with inputs $\left( T _ { e } , V _ { e } , l _ { t } \right) = \left( 2 2 7 ^ { \circ } \mathbf { C } , \right.$ , 41 mm/s, 0.6 mm), are given in Table. 1. The interfaces are numbered from left to right for all layers (e.g., the label for the bonding quality between adjacent filaments at $z _ { \mathrm { c u t } } = 1 7 . 5$ mm of the same part is illustrated in Fig. 7. The 8th and 9th filaments at layers 3, 4, 5, and 6 show delamination. This delamination phenomenon was also observed at the exact same interfaces in the other two sets of samples printed with the same process parameter values. The average of these measured bond lengths gives the mean bond length of the part.

![](images/7122b1bf372ffd55cde3d9177a407e9a10497e9266616e5f68a5fe058d32e7ad.jpg)  
Figure 5: Design of experiments for process parameters

![](images/8511c77f1cf0d1b9f6a374d8ad6020df8dca1a0301a6b4c2098dccdaacc15a37.jpg)  
Figure 6: Thickness measurement of the printed part

interface between filaments 1 and 2 is 1 and the label for the interface between filaments 89 and 90 is 14 in Fig. 4). The Table 1: Bond length measurements (in mm) at each layer for $\left( T _ { e } , V _ { e } , l _ { t } \right) = \left( 2 2 7 ^ { \circ } \mathbf { C } , \right.$ 41 mm/s, 0.6 mm)
<table><tr><td rowspan="2">Layer</td><td colspan="10">Interface</td><td rowspan="2"></td><td colspan="3">13</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>14</td></tr><tr><td>1</td><td>0.383</td><td>0.381</td><td>0.416</td><td>0.469</td><td>0.459</td><td>0.487</td><td>0.472</td><td>0.490</td><td>0.528</td><td>0.525</td><td>0.500</td><td>0.474</td><td>0.452</td><td>0.421</td></tr><tr><td>2</td><td>0.431</td><td>0.444</td><td>0.452</td><td>0.429</td><td>0.396</td><td>0.431</td><td>0.322</td><td>0.360</td><td>0.294</td><td>0.353</td><td>0.363</td><td>0.408</td><td>0.317</td><td>0.342</td></tr><tr><td>3</td><td>0.462</td><td>0.495</td><td>0.434</td><td>0.487</td><td>0.431</td><td>0.345</td><td>0.000</td><td>0.167</td><td>0.101</td><td>0.180</td><td>0.281</td><td>0.347</td><td>0.251</td><td>0.220</td></tr><tr><td>4</td><td>0.424</td><td>0.464</td><td>0.416</td><td>0.444</td><td>0.439</td><td>0.314</td><td>0.170</td><td>0.000</td><td>0.220</td><td>0.261</td><td>0.248</td><td>0.238</td><td>0.210</td><td>0.185</td></tr><tr><td>5</td><td>0.365</td><td>0.441</td><td>0.419</td><td>0.441</td><td>0.431</td><td>0.281</td><td>0.162</td><td>0.000</td><td>0.246</td><td>0.266</td><td>0.243</td><td>0.177</td><td>0.233</td><td>0.200</td></tr><tr><td>6</td><td>0.375</td><td>0.398</td><td>0.467</td><td>0.510</td><td>0.396</td><td>0.347</td><td>0.259</td><td>0.000</td><td>0.254</td><td>0.215</td><td>0.208</td><td>0.157</td><td>0.261</td><td>0.208</td></tr><tr><td>7</td><td>0.340</td><td>0.381</td><td>0.480</td><td>0.449</td><td>0.391</td><td>0.360</td><td>0.287</td><td>0.223</td><td>0.180</td><td>0.223</td><td>0.134</td><td>0.182</td><td>0.195</td><td>0.193</td></tr></table>

The thickness of the part printed at $\left( T _ { e } , V _ { e } , l _ { t } \right) = \left( 2 2 7 ^ { \circ } \mathbf { C } , \right.$ 41 mm/s, 0.6 mm), measured using the laser sensor, is shown in Fig. 8. At each of the five Z-axis points, the measurements are taken along the X-axis to measure the part thickness. It is observed that repeated measurements along X-axis for each Z-axis point show similar values demonstrating the reliability of the measuring system. These discrete point measurements averaged together give the mean thickness of the part.

## 3.2 Model training and prediction

Separate BNNs are constructed for predicting each individual QoI (bond length and thickness in this case). Since the data is collected at many discrete spatial locations, the inputs to the BNNs include the process parameters nozzle temperature, nozzle speed, layer height, and also the spatial locations. The inputs to the bond length model $\mathrm { ( B N N _ { b l } ) }$ are $\left[ T _ { e } , V _ { e } , l _ { t } , x , y \right]$ and the inputs to the part thickness model $\left( \mathrm { B N N } _ { \mathrm { t h } } \right)$ are $[ T _ { e } , \bar { V _ { e } } , l _ { t } , x , z ]$ . The output for $\mathrm { B N N _ { b l } }$ and $\mathrm { B N N } _ { \mathrm { t h } }$ are bond length and part thickness respectively.

The available experimental data related to the bond length model and part thickness model is divided into 2 sets, namely training set (75% of all data) and testing set (25% of all data). The input and output data are normalized prior to the training of the BNN models, and the hyperparameters of these models (e.g., batch size, learning rate, dropout rate etc.) are tuned with grid search. The minimum prediction error during prediction for the bond length model, $\mathrm { B N N } _ { \mathrm { b l } } .$ , is achieved using a dropout rate of 0.1 with 128 batch size, rectified Linear Unit (ReLU) activation function and Adam optimizer with a learning rate of 0.004. The minimum prediction error of $\mathrm { B N N } _ { \mathrm { t h } }$ is obtained using a dropout rate of 0.1 with 128 batch size, sigmoid activation function and Adam optimizer with a learning rate of 0.005.

![](images/f19afdcc9e4ee71f10f3f80f074a880107cadce865cd3ba7480a4cb5fea32122.jpg)  
Figure 7: Cross-section view of a sample with poor bonding

## 3.3 Model performance and cross validation

In order to measure the accuracy of the trained bond length BNN model $\mathrm { ( B N N _ { b l } ) }$ , the model predictions for the testing data subset (i.e., subset of 25 percent of all shuffled data set) are compared with the observed bond lengths at randomly selected interfaces of each printed part in Fig. 9. The horizontal axis represents test combinations with different input parameter combinations and the blue dots denote the bond length measurements at the randomly selected interface from the test data for each printed part. In this figure, the black cross denotes the mean prediction of bond length as predicted by the probabilistic model, the shaded light brown area demonstrates the epistemic uncertainty for one standard deviation away from the mean value, the shaded cyan represents the aleatory uncertainty for one standard deviation away from the mean plus the epistemic uncertainty value and the shaded light blue represents the total uncertainty (aleatoric and epistemic) for one standard deviation away from the mean plus the epistemic plus the aleatory uncertainty value. Thus, the whole shaded area represents the uncertainty bounds for three standard deviations away from the mean predictions.

Figure 10 compares the observed average bond lengths with the predictions using $\mathrm { B N N _ { b l } }$ . The horizontal axis represents different sets of samples, which are printed using the same process parameters $( T _ { e } , V _ { e } , l _ { t } )$ . An overall bond quality metric is obtained by averaging the measured and predicted bond lengths of a part at each interface. As shown in Fig. 10, the model is able to capture the ground truth within half standard deviation.

The observed and mean prediction bond length values at the interfaces of the test data using MC dropout are compared on a 45-degree angle line in Fig. 11. The bond length measurements and predictions are carried out at each interface in a layer. The delamination $( \mathrm { i . e . }$ , zero bond length, which is an outlier) observed in one of the parts is predicted with a non-zero mean bond length. Similar analysis is carried out for part height measurements. Figure 12 shows the comparison between the observation and mean prediction using MC dropout for the part height. Note that the pairs of the observations and the mean predictions are close to the 45-degree line, showing good agreement between predictions and observations for the $\mathrm { B N N _ { b l } }$ model but not that good for the $\mathrm { B N N } _ { \mathrm { t h } }$ model. Further cross-validation of the bond length and part thickness prediction models is done using testing data, i.e., (25% of all shuffled data). The prediction accuracy of both models for the test set is assessed by evaluating the root mean squared error (RMSE); $\mathrm { R M S E } { = } \sqrt { \textstyle \sum _ { i = 1 } ^ { N } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } / N }$ , which is found to be 0.0667 and 0.0826 for $\mathrm { B N N } _ { \mathrm { b l } }$ and $\mathrm { B N N } _ { \mathrm { t h } }$ , respectively. RMSE of 0.1 is used as the quantitative criterion for acceptance. The results indicate that the RMSE values for both models are less than 0.1, thus they are accepted for further analysis.

![](images/55923931fae0e518bd886acdc6c4e628e52d55df024a068343a5962cc9b45175.jpg)  
Figure 8: Part thickness of a sample along Z-axis locations

## 3.4 Multi-objective optimization

The predictions from the BNN models are used in the multi-objective optimization methodology to optimize the process parameters such that the dimensional accuracy and bond quality of the part are maximized. Four different optimization cases under uncertainty are considered:

![](images/28f8cd71cc482a8eb3ab03f91249de307d10518cd61889e5d5a5683dabffb4aa.jpg)

Figure 9: Predicted bond lengths and uncertainty bounds at randomly selected interfaces from the test data and actual bond length measurements (obtained using microscopy images)  
![](images/925b7e2020992ed6c8410a4a4eebd1493b383ba2a1abd41d504ea6f0abea52d6.jpg)  
Figure 10: Average bond length predictions vs. observations of three sets of parts printed with same process parameters  
(1) Case 1, the mean overall bond quality of the part is maximized while minimizing its variance;  
(2) Case 2, the mean geometric accuracy of the part is maximized while minimizing its variance;  
(3) Case 3, the mean overall bond quality and the mean geometric accuracy of the part are maximized; and  
(4) Case 4, the mean overall bond quality and the mean geometric accuracy of the parts are maximized while minimizing the variance in bond quality and geometric accuracy.  
Note that cases 1, 2, and 4 include minimization of variance of the part quality (bond quality and geometric accuracy), thus considering uncertainties. Case 3 minimizes the means of two part quality metrics without considering the variance. The weighted sum approach gives a convex combination of two different objectives in the first three cases. Whereas, Case 4 is a convex combination of four different objectives (i.e., one minus the mean value of dimensionless bond

![](images/9813c70a65dfe33a538823cdef37d1161719cadcf9b06528d4f733ff745e9c78.jpg)  
Figure 11: Comparison of mean dropout prediction and observation of bond lengths

length of the part, $( 1 - \mu _ { \hat { \mathrm { b l } } } )$ , the standard deviation of dimensionless bond length of the part $\sigma _ { \hat { \mathrm { b l } } } .$ , the dimensionles mean and standard deviation of the absolute error between the desired and predicted part thickness $( \mu _ { \mathrm { t h } } , \sigma _ { \mathrm { t h } } ) )$

The QoIs (bond length and part thickness) are dependent on the part layer thickness. For example, intra-layer bond length between interfaces is a function of part layer thickness. The maximum achievable intra-layer bond length equals to layer thickness, thus the scale of bond length values changes with changing layer thicknesses. In order to remove this dependency and to prevent the problem caused by different scales, the mean and standard deviation of the QoIs are non-dimensionalized. The mean and standard deviation of the bond length are non-dimensionalized and scaled to unity by dividing with part layer thickness (i.e., nondimensional bond length = (intra-layer bond length)/(layer thickness)). The predicted mean absolute error in part thickness $( \mu _ { \mathrm { t h } } = \vert \mu _ { \mathrm { t h } _ { \mathrm { p r e d } } } - \mathrm { t h } _ { \mathrm { d e s i r e d } } \vert$ , where $\mu _ { \mathrm { t h } _ { \mathrm { p r e d } } }$ and $\mathrm { \ t h _ { d e s i r e d } }$ are the mean dropout prediction of part thickness and desired part thickness, respectively) and standard deviation of the part thickness $\sigma _ { \mathrm { t h } }$ are also nondimensionalized and scaled to unity as $\mu _ { \mathrm { t h } } = \mu _ { \mathrm { t h } } / \mathrm { t h } _ { \mathrm { d e s i r e d } }$ and $\sigma _ { \mathrm { t h } } = \sigma _ { \mathrm { t h } } / \mathrm { t h _ { d e s i r e d } }$

Based on the information obtained from the layer height measurements and the infrared (IR) thermal camera images during the printing process, no significant difference is observed between the measured and reference values; thus, the epistemic uncertainty in the input is not considered. The model uncertainty (epistemic) is described by placing distributions over the model’s weights. The aleatory variability (intrinsic randomness) in the input, which can be described as noise in the observations, is present in the experimental data used in training the BNN models and is learned by minimizing the loss function shown in Eq. 3. As a result, uncertainty in the prediction model as well as the aleatory uncertainty in the input are considered in the optimization.

The hyperparameters of the NSGA-II optimization algorithm (i.e., population size and the maximum number of generations) are tuned by evaluating the performance indicator (hypervolume) (see Section 2.3.2) of nine different experiments. The experiments are repeated 30 times since the methods are stochastic. The convergence of hypervolume values are monitored for each case. The probability of crossover and mutation are chosen as 0.8 and 0.2, respectively. For illustration purposes, the hypervolume values for Case 3 are compared by changing the population size or maximum number of generations while keeping the other one constant (Fig. 13). The maximum hypervolume value is achieved using a population size of 200 with 30 generations as shown in Fig. 13. The hypervolume values of other cases show a similar trend. Fig. 14 shows in more detail the change in hypervolume values with increasing numbers of generations for a population size of 200 for Case 3. It is found that hypervolume values converge at 30 generations; thus a population size of 200 and 30 generations are used for all the multi-objective optimization cases considered in this paper. The maximum extrusion volume inequality constraint $g$ (Section 3) is implemented using a penalty function. The penalty function gives a fitness disadvantage to the individuals that violate the constraint.

![](images/4818fbba09646d352540c705cdf918cd640dc6703f460d67fa124b929a76d2f7.jpg)  
Figure 12: Comparison of mean dropout prediction and observation of part thickness

## 3.4.1 Case 1

In Case 1, one minus the mean of dimensionless bond quality metric $( 1 - \mu _ { \hat { \mathrm { b l } } } )$ and the standard deviation of the dimensionless bond quality metric $\sigma _ { \hat { \mathrm { b l } } }$ are minimized.

$$
\begin{array} { r l } { \underset { \mathbf { x } \in \mathbb { R } ^ { n } { \operatorname { t } } } { \operatorname* { m i n i m i z e } } } & { w _ { 1 } \left( 1 - \mu _ { \hat { \mathrm { b l } } } ( \mathbf { x } ) \right) + w _ { 2 } \sigma _ { \hat { \mathrm { b l } } } ( \mathbf { x } ) } \\ { \mathrm { s u b j e c t ~ t o } } & { g ( \mathbf { x } ) = 0 . 8 m m \times V _ { e } \times l _ { t } \leq 2 4 m m ^ { 3 } / s } \\ & { 2 1 5 ^ { \circ } \mathbf { C } \leq T _ { e } \leq 2 8 0 ^ { \circ } \mathbf { C } } \\ & { 2 5 \mathrm { m m / s } \leq V _ { e } \leq 4 5 \mathrm { m m / s } } \\ & { l _ { t } \in \{ 0 . 4 2 , 0 . 6 0 , 0 . 7 0 \} \mathrm { m m } } \end{array}\tag{11}
$$

The Pareto front obtained for these two objectives is demonstrated in Fig. 15 with selected design points $\mathbf { A } _ { 1 } , \mathbf { B } _ { 1 }$ , and $\mathrm { C _ { 1 } }$ for experimental validation. The design points represent three design variables, i.e., nozzle temperature $T _ { e } ( ^ { \circ } \mathrm { C } )$ , nozzle speed $V _ { e }$ (mm/s), and layer thickness $l _ { t }$ (mm).

In each generation, several values of the design variables (nozzle temperature, nozzle speed, and layer thickness) are generated and passed to the optimizer to evaluate two objectives. The layer height is restricted to three possible values, 0.42, 0.6, and 0.7 mm, in order to have an integer value for the total number of layers given the desired part thickness of 4.2 mm. A single design can be selected from the optimal designs shown in Fig. 15.

![](images/d848f0ba7a97ccd58ea9354ecc9d3a9583866a17f0efb3f90551bc6d82e79781.jpg)  
Figure 13: Hypervolume values (mean and one standard deviation) of nine different experimental designs of Case 3

![](images/36c2ef97cd90ae434515d398c2bd253fec010ce0bd4f8cccba3b8ff69619779a.jpg)  
Figure 14: Hypervolume values vs. number of generations of Case 3 (Population size 200)

![](images/8adfc76c78b6f6a58ad20446e5c369189c04bda697502712a4b7dd2cb272c9c2.jpg)  
Figure 15: Pareto front of Case 1: $\mathbf { A } _ { 1 } = ( T _ { e } \mathbf { \Psi } ( { ^ \circ } \mathbf { C } ) .$ , V<sub>e</sub> (mm/s), l<sub>t</sub> (mm)) = (217.09, 26.14, 0.42), $\mathbf { B } _ { 1 } = ( 2 4 4 . 5 4 , 2 9 . 5 9$ 0.60), $\mathbf { C } _ { 1 } = ( 2 1 9 . 0 3 , 4 3 . 9 6 , 0 . 4 2 )$

## 3.4.2 Case 2

The dimensionless mean and standard deviation of part thickness error $( \mu _ { \mathrm { t h } } , \sigma _ { \mathrm { t h } } )$ are minimized in Case 2 using the $\mathrm { B N N } _ { \mathrm { t h } }$

$$
\begin{array} { r l } { \underset { \mathbf { x } \in \mathbb { R } ^ { n } } { \mathrm { m i n i m i z e } } } & { w _ { 1 } \mu _ { \mathrm { t h } } ( \mathbf { x } ) + w _ { 2 } \sigma _ { \mathrm { t h } } ( \mathbf { x } ) } \\ { \mathrm { s u b j e c t ~ t o } } & { g ( \mathbf { x } ) = 0 . 8 m m \times V _ { e } \times l _ { t } \leq 2 4 m m ^ { 3 } / s } \\ & { 2 1 5 ^ { \circ } \mathbf { C } \leq T _ { e } \leq 2 8 0 ^ { \circ } \mathbf { C } } \\ & { 2 5 \mathrm { m m / s } \leq V _ { e } \leq 4 5 \mathrm { m m / s } } \\ & { l _ { t } \in \{ 0 . 4 2 , 0 . 6 0 , 0 . 7 0 \} \mathrm { m m } } \end{array}\tag{12}
$$

Similar to Case 1, the Pareto front obtained for these two objectives is shown in Fig. 16 with selected design points $\mathbf { A } _ { 2 }$ ${ \bf B } _ { 2 }$ , and $\mathrm { { C _ { 2 } } }$ for experimental validation.

## 3.4.3 Case 3

In this case, one minus the mean of dimensionless bond quality metric $( 1 - \mu _ { \hat { \mathrm { b l } } } )$ and the dimensionless mean of part thickness error $\mu _ { \mathrm { t } \hat { \mathrm { h } } }$ are minimized by using both models $\mathrm { B N N _ { b l } }$ and $\mathrm { B N N } _ { \mathrm { t h } }$

$$
\begin{array} { r l } { \underset { \mathbf { x } \in \mathbb { R } ^ { n } { \operatorname { t r } } _ { \mathbf { x } } } { \operatorname { m i n i m i z e } } } & { w _ { 1 } \left( 1 - \mu _ { \hat { \mathrm { b l } } } ( \mathbf { x } ) \right) + w _ { 2 } \mu _ { \mathrm { t h } } ( \mathbf { x } ) } \\ { \mathrm { s u b j e c t ~ t o } } & { g ( \mathbf { x } ) = 0 . 8 m m \times V _ { e } \times l _ { t } \leq 2 4 m m ^ { 3 } / s } \\ & { 2 1 5 ^ { \circ } \mathbf { C } \leq T _ { e } \leq 2 8 0 ^ { \circ } \mathbf { C } } \\ & { 2 5 \mathrm { m m / s } \leq V _ { e } \leq 4 5 \mathrm { m m / s } } \\ & { l _ { t } \in \{ 0 . 4 2 , 0 . 6 0 , 0 . 7 0 \} \mathrm { m m } } \end{array}\tag{13}
$$

The Pareto front of these two objectives is shown in Fig. 17 with selected design points $\mathbf { A } _ { 3 } , \mathbf { B } _ { 3 } .$ , and $\mathrm { C _ { 3 } }$ for experimental validation.

## 3.4.4 Case 4

In this case, one minus the mean of the dimensionless bond quality metric $( \mathrm { O b j _ { 1 } } = 1 - \mu _ { \hat { \mathrm { b l } } } )$ , the standard deviation of the dimensionless bond quality metric $( \mathrm { O b j _ { 2 } } = \sigma _ { \hat { \mathrm { b l } } } )$ , and the dimensionless mean and standard deviation of part

![](images/9d3dd64d436cf2a5ce3595ae9247fc5a4b73a81dbcb4bae63521afa9f3ade412.jpg)  
Figure 16: Pareto front of Case 2: $\mathbf { A } _ { 2 } = ( T _ { e }$ (<sup>◦</sup>C), V<sub>e</sub> (mm/s), l<sub>t</sub> (mm)) = (272.17, 33.68, 0.42), $\mathbf { B } _ { 2 } = ( 2 2 3 . 9 2 , 3 1 . 0 2$ 0.42), $\mathbf { C } _ { 2 } = ( 2 5 1 . 4 1 , 3 6 . 6 3 , 0 . 4 2 )$

![](images/0c98f1fca1ea65193276f9b5cad6e35f80c2008d1022bbfcd79e99f964a26f6a.jpg)  
Figure 17: Pareto front of Case 3: $\mathbf { A } _ { 3 } = ( T _ { e }$ (<sup>◦</sup>C), V<sub>e</sub> (mm/s), l<sub>t</sub> (mm)) = (217.02, 26.01, 0.42), $\mathbf { B } _ { 3 } = ( 2 1 7 . 0 5 , 2 7 . 8 4 .$ $0 . { \overset { \_ } { 4 } } 2 ) , \operatorname { C } _ { 3 } = ( 2 7 4 . 0 4 , 3 4 . 2 9 , 0 . 4 2 )$

thickness error $( \mathrm { O b j _ { 3 } } = \mu _ { \mathrm { t h } } \ \& \ \mathrm { O b j _ { 4 } } = \sigma _ { \mathrm { t h } } )$ are minimized by using both models $\mathrm { B N N _ { b l } }$ and $\mathrm { B N N } _ { \mathrm { t h } }$ . In each generation, a new value of the design variables $( T _ { e } , V _ { e } , l _ { t } )$ is generated and passed to the optimizer to evaluate four

objectives.

$$
\begin{array} { l l } { \underset { \mathbf { x } \in \mathbb { R } ^ { n _ { x } } } { \mathrm { m i n i m i z e } } } & { w _ { 1 } \left( 1 - \mu _ { \hat { \mathbf { b } } 1 } ( \mathbf { x } ) \right) + w _ { 2 } \sigma _ { \hat { \mathbf { b } } 1 } ( \mathbf { x } ) + w _ { 3 } \mu _ { \hat { \mathbf { t } } \hat { \mathbf { h } } } ( \mathbf { x } ) + w _ { 4 } \sigma _ { \hat { \mathbf { t } } \hat { \mathbf { h } } } ( \mathbf { x } ) } \\ { \mathrm { s u b j e c t \ t o } } & { g ( \mathbf { x } ) = 0 . 8 m m \times V _ { e } \times l _ { t } \leq 2 4 m m ^ { 3 } / s } \\ & { 2 1 5 ^ { \circ } \mathbf { C } \leq { T _ { e } } \leq 2 8 0 ^ { \circ } \mathbf { C } } \\ & { 2 5 \mathrm { m m } / s \leq V _ { e } \leq 4 5 \mathrm { m m / s } } \\ & { l _ { t } \in \{ 0 . 4 2 , 0 . 6 0 , 0 . 7 0 \} \mathrm { m m } } \end{array}\tag{14}
$$

A plot of the Pareto front is demonstrated in Fig. 18, where $\mathrm { O b j _ { 1 } }$ is displayed on the x-axis, $\mathrm { O b j _ { 2 } }$ on the ${ \mathrm { y - a x i s , O b j _ { 3 } } }$ by the color of the markers, and $\mathrm { O b j _ { 4 } }$ by the size of the markers. A single design can be selected from the optimal designs shown in Fig. 18. The designer can choose a design based on the relative importance of each objective over the others, since slight improvement in one of the objectives may lead to significant degradation in other objectives. From the optimal solutions, three points $\mathbf { A } _ { 4 } , \mathbf { B } _ { 4 } .$ , and $\mathrm { C _ { 4 } }$ are chosen for experimental validation.

![](images/b9328b2fb1c8c15be3050c06b713b87ee1ce09f3dd7e8f497515ed46dac85042.jpg)  
Figure 18: Pareto designs for Case 4: $\mathbf { A } _ { 4 } = ( T _ { e } \mathbf { \Psi } ( { ^ \circ C } )$ , $V _ { e }$ (mm/s), l<sub>t</sub> (mm)) = (217.08, 26.88, 0.42), $\mathbf { B } _ { 4 } = ( 2 7 7 . 9 9 , 3 9 . 4 1$ $0 . { \overset { \vartriangle } { 4 } } 2 ) , \operatorname { C } _ { 4 } = ( 2 2 5 . 3 4 , 3 { \overset { \vartriangle } { 1 } } . 8 4 , 0 . 4 2 )$

The optimization results for Case 4 can be illustrated using the parallel coordinate plots shown in Figs. 19 and 20. The resulting parallel coordinates reflect the Pareto dominance relation between different solutions. In Fig. 19, the color map reflects the values of the first objective $( 1 - \mu _ { \hat { \mathrm { b l } } } )$ . The red solid line represents the solution that results in the worst mean bond quality within the Pareto front. Note that due to the negative correlation between the first and second objectives (heavily conflicting objectives), worst mean bond quality solutions (i.e., red solid lines) couple with the best (lower) standard deviation bond quality. The mean bond quality starts to improve as the color of the solid line changes from red to green. In the same manner as Fig. 19, Fig. 20 also illustrates the optimization results with a color map of the third objective. The red solid lines denote the solutions that yield the poorest mean geometric accuracy within the Pareto front approximations. The mean and standard deviation of the geometric accuracy are partially negatively correlated since there are lines that are parallel between the two axes in parallel coordinates. More specifically, the parallel green solid lines at the bottom of the third and fourth objectives that do not intersect with other lines depict the solutions with moderately good mean and standard deviation of geometric accuracy.

The Pearson correlation coefficient between each design variable and/or objective functions is shown in Fig. 21. A correlation coefficient of ±1 represents a perfect linear relationship. The nozzle temperature $\mathrm { ( D V _ { 1 } ) }$ is negatively correlated with $\mathrm { O b j _ { 1 } } = 1 - \mu _ { \hat { \mathrm { b l } } }$ and $\mathrm { O b j _ { 3 } } = \mu _ { \mathrm { t h } }$ , which means that the mean geometric accuracy and the overall mean bond quality of a part tend to improve with increasing nozzle temperature. Whereas, parts with a greater layer height $\mathrm { ( D V _ { 3 } ) }$ appear to have a degraded mean geometrical accuracy and mean bond quality. The nozzle speed $\mathrm { ( D V _ { 2 } ) }$ has a smaller effect on the objective functions. The correlation coefficient between Obj and Obj is calculated as 0.46 indicating that these two objectives do not have a significant relationship, which is also illustrated in Fig. 24. Furthermore, the parallel plots and the correlation matrix demonstrate the difficulty in choosing the optimal process parameters. Therefore, the Pareto front is beneficial in finding a design that offers a good trade-off between the objectives.

![](images/ce7182bba8dd8134d8a65163dc1d3481696fa65a98d793abeafff6e79b15cc08.jpg)

Figure 19: Parallel coordinate representation of the model dependencies with a color mapping along the first objective $( 1 { \bar { \mathbf { \Gamma } } } _ { - } \mu _ { \hat { \mathbf { b } } 1 } )$  
![](images/e9709291eab415fb41374f5e0a5ecdc3a7afd1adc3ea87c87d561cb00e3d5597.jpg)  
Figure 20: Parallel coordinate representation of the model dependencies with a color mapping along the third objective $( \mu _ { \mathrm { t h } } )$ .

![](images/c1ec7c4b406cc57574d444fb2dabcab5150647c3fd225c6d2b0c81c9801aead8.jpg)  
Figure 21: The correlation coefficient between design variables and objective functions  
The computation time for one optimization is on average ten minutes using a desktop computer (Intel<sup>®</sup> Xeon<sup>®</sup> CPU E5-1660 v4@3.20GHz with 32 GB RAM and GPU NVIDIA Quadro K620 with 2 GB).

## 3.5 Monte Carlo optimization

The Pareto front and direct correlations between objective functions obtained from the NSGA-II algorithm in Section 3.4 are compared with a Monte Carlo sampling (MCS) approach. A relatively large number of Monte Carlo samples (10,000 samples) of design variables from the design space are generated and the objective functions for different cases are predicted using the trained BNN models. Due to computational complexity and difficulty in visualizing the four-dimensional results of Case 4, MCS approach is only employed for bi-objective cases i.e., Cases 1, 2 and 3 (Figs. 22, 23, 24).

The relationship between the two objectives in Case 2 and 3 as shown in Figs. 23 and 24 is highly non-linear. The regions with sharp edges generally represent constraint boundaries. The relationship between two physical quantities, i.e., bond quality and geometry accuracy of a part, shown in Fig. 24 demonstrates the importance of choosing the optimal process parameters. For some combinations of design variables, both physical quantities may have significantly degraded values, which can be avoided by using the proposed methodology. The Pareto front results shown in Figs. 15, 16 and 17 are superimposed on the MCS results and labeled as red crosses in Figs. 22, 23, and 24. The superimposed Pareto fronts show that the optimization results are in agreement with the MCS results.

## 3.6 Experimental validation

Three points are chosen for validation from the Pareto fronts for each of the four multi-objective optimization cases considered above. The predicted values of objectives for each point are validated using classical hypothesis testing. The sample size of experiments for each point is 5. The sample size of model predictions for each point is 100, which is produced using MC dropout. For validation of the prediction models, the t-test, which is based on Student’s t-distribution, is used to test the hypothesis that the mean of the observations is equal to the mean of the model prediction. Since the calculated p-values for all selected design points $( \mathrm { A _ { 1 } } , \mathrm { B _ { 1 } } , \mathrm { C _ { 1 } } , \mathrm { A _ { 2 } } , \mathrm { B _ { 2 } } , \mathrm { C _ { 2 } } , \mathrm { A _ { 3 } } , \mathrm { B _ { 3 } } , \mathrm { C _ { 3 } } , \mathrm { A _ { 4 } } , \mathrm { B _ { 4 } } , \mathrm { C _ { 4 } } )$ are above the significance level $\alpha = 0 . 0 5$ , the predictions at the optimum solutions agree with the experimental observations. This is the validation of the BNN prediction models, which are trained with the previous experimental data. Note that the training data and validation data are separate.

![](images/885486efa79cb2ec5e18dca811557a380c3cd03967adbc4a3b24eb221229879d.jpg)  
Figure 22: Pareto front and MCS results of Case 1

![](images/a645437f1cd5f8276d9ceb950e90f144d3ac6cf389e218436f37d37500d430d0.jpg)  
Figure 23: Pareto front and MCS results of Case 2

Next we also validate the optimization result, by demonstrating that the optimum solution gives better part quality than randomly selected process parameter settings. Parts are printed and measured at eleven other process parameter settings selected at random and compared to the part printed with the optimal process parameters of Case 3. The dimensionless mean thickness error $( \mu _ { \mathrm { t h } } )$ and the dimensionless bond quality metric $( 1 - \mu _ { \hat { \mathrm { b l } } } )$ obtained using the randomly selected process parameters are found to be larger than that of the optimal solution (Fig. 25), thus clearly demonstrating the effectiveness of the proposed methodology.

![](images/da40addbce4a74e9184c5db0c56d87507a88f7c81da07aeac72c84c2bef98c55.jpg)  
Figure 24: Pareto front and MCS results of Case 3

![](images/9ffb2c513ee4968e28bce14fae8811397aebd1e8619a634f4e725919096d97f0.jpg)  
Figure 25: Experimental validation of optimal process design (Case 3)

## 3.7 Summary of results

We find that, using the proposed approach, we were able to manufacture FFF parts with better part quality using the optimization results from the Pareto front obtained through the probabilistic data-driven methodology for process parameter optimization. In this study, we obtain probabilistic estimates of both quantities (bond quality and geometrical accuracy) and include the probabilistic estimates within the process design for optimization under uncertainty. The input variability in the experimental data and the model uncertainty are captured through the probabilistic deep learning approach, BNN.

The probabilistic estimates facilitate different options of optimization under uncertainty: optimize only the mean value of an individual quality metric, optimize both the mean and variance of an individual quality metric, optimize the mean values of multiple quality metrics simultaneously, and optimize both the mean values and variances of multiple quality metrics simultaneously. Except the first option, all other options require multi-objective formulations. It is observed that the mean values of both quality metrics (bond length and part thickness accuracy) are optimized with the smallest layer thickness (i.e., 0.42 mm). This is generally expected in AM processes. However, it is not that straightforward to see a similar trend for other parameters. In general, the optimal solutions for the mean of both quality metrics are achieved with larger nozzle temperature and smaller nozzle speed values. However, there are some cases where some other combinations of nozzle temperature and speed (e.g., small temperature and speed) also result in optimal solutions for these two quantities. Thus, the Pareto front is helpful in finding a design that offers a good trade-off between the objectives.

The prediction models are verified using the cross-validation approach as explained in Section 3.3, and validated against experimental data. The optimization results are validated by printing and measuring the parts corresponding to the points A, B, and C of the Pareto fronts. The results show that the optimum solution using the proposed process optimization framework gives better part quality than randomly selected process parameter settings.

## 4 Conclusion

In this paper, a data-driven machine learning model, is proposed for multi-objective optimization under uncertainty of process parameters in AM. The proposed methodology is demonstrated for AM by optimizing the FFF process parameters considering the objectives of minimizing the error in the part geometry and maximizing the bond quality between the filaments. Several cases with multiple objectives are considered: minimizing the mean and standard deviation of the bond quality metric, minimizing the mean and standard deviation of part thickness error, minimizing the mean part thickness error and mean bond quality metric, and minimizing the means and standard deviations of both part thickness error and bond quality metric. The BNN models were constructed using actual manufacturing data, and are able to quantify the uncertainty in the prediction. Both the BNN models and the optimization solutions are validated with actual manufactured parts, at the optimal and various other settings. The parts manufactured with optimal process parameter settings are observed to result in better part quality than parts manufactured with other non-optimal settings.

In future work, the proposed methodology can be extended to process control in AM. Using the real-time data collected during the manufacturing process, the approach can be extended to make layer-wise control decisions under uncertainty about the AM process such as changing the process parameters for subsequent layers to improve the part by monitoring the quality metrics during manufacturing. Future work can also incorporate the physics knowledge to fill the physics knowledge gap in the machine learning model, and to improve the prediction accuracy by correcting for the approximations in the physics-based models.

## Acknowledgment

This study was supported by funding from the National Institute of Standards and Technology under the Smart Manufacturing Data Analytics Project (Cooperative Agreement No. 70NANB14H036). The support is gratefully acknowledged. The authors also thank Joseph D. Olson, research assistant at Vanderbilt University, for conducting some of the experiments.

## Disclaimer

Certain commercial systems are identified in this paper. Such identification does not imply recommendation or endorsement by NIST; nor does it imply that the products identified are necessarily the best available for the purpose. Further, any opinions, findings, conclusions, or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of NIST or any other supporting U.S. government or corporate organizations.

## References

[1] ASTM Standard. ISO/ASTM 52900:2015 Additive manufacturing - general principles - terminology, 2015.

[2] Berkcan Kapusuzoglu and Sankaran Mahadevan. Physics-informed and hybrid machine learning in additive manufacturing: Application to fused filament fabrication. JOM, 72:4695–4705, 2020.

[3] Tarasankar Debroy, Wei Zhang, J Turner, and Sudarsanam Suresh Babu. Building digital twins of 3d printing machines. Scripta Materialia, 135:119–124, 2017.

[4] Mustafa Megahed, Hans-Wilfried Mindt, Narcisse N’Dri, Hongzhi Duan, and Olivier Desmaison. Metal additivemanufacturing process and residual stress modeling. Integrating Materials and Manufacturing Innovation, 5(1):61–93, 2016.

[5] Felix W Baumann, André Sekulla, Michael Hassler, Benjamin Himpel, and Markus Pfeil. Trends of machine learning in additive manufacturing. International Journal ofRapid Manufacturing, 7(4):310–336, 2018.

[6] Ohyung Kwon, Hyung Giun Kim, Wonrae Kim, Gun-Hee Kim, and Kangil Kim. A convolutional neural network for prediction of laser power using melt-pool images in laser powder bed fusion. IEEE Access, 2020.

[7] Mojtaba Khanzadeh, Sudipta Chowdhury, Mohammad Marufuzzaman, Mark A Tschopp, and Linkan Bian. Porosity prediction: Supervised-learning of thermal history for direct laser deposition. Journal of manufacturing systems, 47:69–82, 2018.

[8] Donghong Ding, Zengxi Pan, Dominic Cuiuri, Huijun Li, Stephen van Duin, and Nathan Larkin. Bead modelling and implementation of adaptive mat path in wire and arc additive manufacturing. Robotics and Computer-Integrated Manufacturing, 39:32–42, 2016.

[9] Mehrshad Mehrpouya, Annamaria Gisario, Atabak Rahimzadeh, Mohammadreza Nematollahi, Keyvan Safaei Baghbaderani, and Mohammad Elahinia. A prediction model for finding the optimal laser parameters in additive manufacturing of niti shape memory alloy. The International Journal ofAdvanced Manufacturing Technology, 105(11):4691–4699, 2019.

[10] Dongsen Ye, Jerry Ying Hsi Fuh, Yingjie Zhang, Geok Soon Hong, and Kunpeng Zhu. In situ monitoring of selective laser melting using plume and spatter signatures by deep belief networks. ISA transactions, 81:96–104, 2018.

[11] Aniruddha Gaikwad, Farhad Imani, Hui Yang, Edward Reutzel, and Prahalada Rao. In situ monitoring of thin-wall build quality in laser powder bed fusion using deep learning. 3, 2019.

[12] Paromita Nath, Zhen Hu, and Sankaran Mahadevan. Uncertainty quantification of grain morphology in laser direct metal deposition. Modelling and Simulation in Materials Science and Engineering, 27(4):044003, 2019.

[13] Berkcan Kapusuzoglu, Matthew Sato, Sankaran Mahadevan, and Paul Witherell. Process optimization under uncertainty for improving the bond quality of polymer filaments in fused filament fabrication. Journal of Manufacturing Science and Engineering, 143:1–46, 08 2020.

[14] Berkcan Kapusuzoglu and Sankaran Mahadevan. Information fusion and machine learning for sensitivity analysis using physics knowledge and experimental data. Reliability Engineering & System Safety, 214:107712, 2021.

[15] Paromita Nath, Joseph D. Olson, Sankaran Mahadevan, and Yung-Tsun Tina Lee. Optimization of fused filament fabrication process parameters under uncertainty to maximize part geometry accuracy. Additive Manufacturing, 35:101331, 2020.

[16] Mohamad Mahmoudi, Gustavo Tapia, Kubra Karayagiz, Brian Franco, Ji Ma, Raymundo Arroyave, Ibrahim Karaman, and Alaa Elwany. Multivariate calibration and experimental validation of a 3D finite element thermal model for laser powder bed fusion metal additive manufacturing. Integrating Materials and Manufacturing Innovation, 7(3):116–135, 2018.

[17] Zhuo Wang, Pengwei Liu, Yaohong Xiao, Xiangyang Cui, Zhen Hu, and Lei Chen. A data-driven approach for process optimization of metallic additive manufacturing under uncertainty. Journal ofManufacturing Science and Engineering, 141(8), 2019.

[18] Lyle Levine, Brandon Lane, Jarred Heigel, Kalman Migler, Mark Stoudt, Thien Phan, Richard Ricker, Maria Strantza, Michael Hill, Fan Zhang, et al. Outcomes and conclusions from the 2018 am-bench measurements, challenge problems, modeling submissions, and conference. Integrating Materials and Manufacturing Innovation, 9(1):1–15, 2020.

[19] Brandon Lane, Jarred Heigel, Richard Ricker, Ivan Zhirnov, Vladimir Khromschenko, Jordan Weaver, Thien Phan, Mark Stoudt, Sergey Mekhontsev, and Lyle Levine. Measurements of melt pool geometry and cooling rates of individual laser traces on in625 bare plates. Integrating Materials and Manufacturing Innovation, pages 1–15, 2020.

[20] Daniel P Cole, Frank Gardea, Todd C Henry, Jonathan E Seppala, Edward J Garboczi, Kalman D Migler, Christopher M Shumeyko, Jeffrey R Westrich, Sara V Orski, and Jeffrey L Gair. Amb2018-03: Benchmark physical property measurements for material extrusion additive manufacturing of polycarbonate. Integrating Materials and Manufacturing Innovation, 9(4):358–375, 2020.

[21] John S Denker and Yann LeCun. Transforming neural-net output levels to probability distributions. In Advances in neural information processing systems, pages 853–859, 1991.

[22] David JC MacKay. A practical bayesian framework for backpropagation networks. Neural Computation, 4(3):448–472, 1992.

[23] R. M. Neal. Bayesian learning for neural networks. Lecture Notes in Statistics, 1996.

[24] Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in neural network. In Proceedings ofthe 32nd International Conference on Machine Learning, volume 37, pages 1613–1622. PMLR, 2015.

[25] Mark A Beaumont, Wenyang Zhang, and David J Balding. Approximate bayesian computation in population genetics. Genetics, 162(4):2025–2035, 2002.

[26] Walter R Gilks. Markov chain monte carlo. Encyclopedia ofBiostatistics, 4, 2005.

[27] M Sanjeev Arulampalam, Simon Maskell, Neil Gordon, and Tim Clapp. A tutorial on particle filters for online nonlinear/non-gaussian bayesian tracking. IEEE Transactions on signal processing, 50(2):174–188, 2002.

[28] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In Proceedings ofThe 33rd International Conference on Machine Learning, volume 48 of Proceedings ofMachine Learning Research, pages 1050–1059, New York, New York, USA, 20–22 Jun 2016. PMLR.

[29] Alex Kendall and Yarin Gal. What uncertainties do we need in bayesian deep learning for computer vision? In Proceedings ofthe 31st International Conference on Neural Information Processing Systems, volume 30 of NIPS’17, page 5580–5590. Curran Associates, Inc., 2017.

[30] Kais Zaman, Mark McDonald, Sankaran Mahadevan, and Lawrence Green. Robustness-based design optimization under data uncertainty. Structural and Multidisciplinary Optimization, 44(2):183–197, 2011.

[31] Xiaoping Du and Wei Chen. Sequential optimization and reliability assessment method for efficient probabilistic design. Journal ofmechanical design, 126(2):225–233, 2004.

[32] Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and TAMT Meyarivan. A fast and elitist multiobjective genetic algorithm: Nsga-ii. IEEE transactions on evolutionary computation, 6(2):182–197, 2002.

[33] Eckart Zitzler, Dimo Brockhoff, and Lothar Thiele. The hypervolume indicator revisited: On the design of pareto-compliant indicators via weighted integration. In International Conference on Evolutionary Multi-Criterion Optimization, pages 862–876. Springer, 2007.

[34] Joshua D Knowles, Lothar Thiele, and Eckart Zitzler. A tutorial on the performance assessment of stochastic multiobjective optimizers. TIK-Report, 214, 2006.

[35] Eckart Zitzler, Lothar Thiele, Marco Laumanns, Carlos M Fonseca, and Viviane Grunert Da Fonseca. Performance assessment of multiobjective optimizers: An analysis and review. IEEE Transactions on evolutionary computation, 7(2):117–132, 2003.

[36] Eckart Zitzler, Kalyanmoy Deb, and Lothar Thiele. Comparison of multiobjective evolutionary algorithms: Empirical results. Evolutionary computation, 8(2):173–195, 2000.

[37] You Ling and Sankaran Mahadevan. Quantitative model validation techniques: New insights. Reliability Engineering & System Safety, 111:217–231, 2013.

[38] Ultimaker S5 Specifications. https://ultimaker.com/3d-printers/ultimaker-s5, accessed: 02/01/2020.

[39] Keyence LK-H057 Specifications. https://www.keyence.com/products/measure/laser-1d/lk-g5000/ models/lk-h057/index.jsp, accessed: 07/29/2019.

[40] Caroline A Schneider, Wayne S Rasband, and Kevin W Eliceiri. Nih image to imagej: 25 years of image analysis. Nature methods, 9(7):671, 2012.