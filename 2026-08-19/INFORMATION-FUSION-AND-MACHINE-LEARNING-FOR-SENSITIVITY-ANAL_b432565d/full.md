# INFORMATION FUSION AND MACHINE LEARNING FOR SENSITIVITY ANALYSIS USING PHYSICS KNOWLEDGE AND EXPERIMENTAL DATA ∗

Berkcan Kapusuzoglu<sup>†</sup> and Sankaran Mahadevan

Department of Civil and Environmental Engineering Vanderbilt University, Nashville, TN 37235, USA berkcan.kapusuzoglu@vanderbilt.edu, sankaran.mahadevan@vanderbilt.edu

## ABSTRACT

When computational models (either physics-based or data-driven) are used for the sensitivity analysis of engineering systems, the sensitivity estimate is affected by the accuracy and uncertainty of the model. This paper considers global sensitivity analysis (GSA) for situations where both a physicsbased model and experimental observations are available, and investigates physics-informed machine learning strategies to effectively combine the two sources of information in order to maximize the accuracy of the sensitivity estimate. Two representative machine learning (ML) techniques are considered, namely, deep neural networks (DNN) and Gaussian process (GP) modeling, and two strategies for incorporating physics knowledge within these techniques are investigated, namely: (i) incorporating loss functions in the ML models to enforce physics constraints, and (ii) pre-training and updating the ML model using simulation and experimental data respectively. Four different models are built for each type (DNN and GP), and the uncertainties in these models are included in the Sobol’ indices computation. The DNN-based models, with many degrees of freedom in terms of model parameters and training options, are found to result in smaller bounds on the sensitivity estimates when compared to the GP-based models. The proposed methods are illustrated for additive manufacturing and lake temperature modeling examples.

Keywords Global sensitivity analysis · Sobol’ index · Deep learning · Physics-informed machine learning · Additive manufacturing · Information fusion

## 1 Introduction

Computational models are often used to analyze the response of an engineering system for a variety of input realizations, since conducting experiments to directly measure the true response for many input realizations is often not affordable. However, the computational model is often an incomplete representation of the complex physical system, thus the system response prediction is affected by model uncertainty. In general, the uncertainty sources affecting system response prediction include (a) epistemic uncertainty due to lack of knowledge (arising from either data or model inadequacies), and (b) aleatory uncertainty due to the inherent variability in the system properties or the external inputs. Global sensitivity analysis (GSA) [1] aims to provide a quantitative assessment of the relative contribution of each uncertainty source to the uncertainty in the model response [2, 3, 4].

Much of the GSA literature has focused on variability in the inputs and their effects on output variability; the extension of GSA to include epistemic uncertainty sources (data, model) is recent and sparse [5, 6, 7, 8, 9]. Model outputs can have uncertainty even for a fixed input when there exists model uncertainty. When the model is computationally expensive, it is often replaced with a surrogate model to facilitate the estimation of Sobol’ indices, since such computation requires many input-output samples from the model; the surrogate model introduces additional uncertainty. Several types of surrogate models are used in the literature, e.g., polynomial chaos expansion (PCE), Gaussian process (GP) regression, neural networks, etc., to train a parametric relationship between the inputs and the outputs. The quality and quantity of the training data affect the accuracy of these surrogate models, which directly affects the uncertainty in the mode output [7, 8, 9]. Thus, it is important to also include the contribution of surrogate model uncertainty to the output uncertainty in GSA. In Le Gratiet et al [7] for example, the Gaussian process surrogate model uncertainty is included in the Sobol’ index estimates using multiple realizations of the GP model prediction, which helps to construct prediction intervals for the Sobol’ index estimates.

Expanding GSA to consider both aleatory and epistemic uncertainty sources is beneficial in supporting resource allocation decisions. If the contribution of epistemic uncertainty is found to be significant, then it may be valuable to collect more data or refine the physics model to reduce the epistemic uncertainty and thus its contribution to the output uncertainty. Several GSA studies have developed auxiliary variable-based approaches to include both aleatory and epistemic uncertainty sources at a single level instead of using nested simulations, thus achieving both computational efficiency and direct ranking of the different sources of uncertainty to support resource allocation decision-making. The auxiliary variable is used to transform one-to-many input-output mapping to one-to-one mapping, thus facilitating the computation of Sobol’ indices for both aleatory and epistemic sources [5]. This idea is expanded in [6] to include several epistemic sources, such as input statistical uncertainty, surrogate model error, physics model discrepancy, and numerical solution error, and to systems with time series inputs and outputs.

Three scenarios of model and data availability can be considered for GSA: (1) use of a physics-based computational model alone, (2) use of available input-output data alone (either from experiments or previous simulations), or (3) use of both physics model and available experimental data. A straightforward model-based approach to estimate Sobol’ indices is to use a double-loop Monte Carlo simulation (MCS) [10]. In order to reduce the cost associated with the double-loop MCS, analytical, spectral and efficient sampling-based methods have been developed. The methodology developed by Sudret [11] approximates the original physics model by a PCE and estimates the Sobol’ indices by using the PCE coefficients. Chen et al. [12] proposed analytical formulas to compute Sobol’ indices using a GP surrogate model with input variables that follow normal or uniform distributions. The improved FAST method [13] combines the classical FAST method [2] with random balanced design [14] for generating samples to evaluate Sobol’ indices.

In some problems, input-output data may be already available instead of having to simulate a physics model expressly for the purpose of GSA. Such data may be available from experiments, or real-world observations, or Markov Chain Monte Carlo (MCMC) sampling during Bayesian model calibration, or MC sampling during reliability analysis, etc. In such cases, data-driven methods have been proposed to directly compute the Sobol’ indices based on available input-output samples instead of simulation runs of the physics model. A GSA method based on ANOVA using factorial design of experiments is developed by Ginot et al. [15]. The proposed method results in same values as the Sobol’ index since the variance decomposition used in the Sobol’ index estimations is same as the one used in the classical ANOVA [16]. In high dimensional problems, even the use of a surrogate model for GSA, which repeatedly executes the code by suppressing some variables and running through the range of other variables, may be computationally demanding since the number of executions of the code increases rapidly with the number of inputs [7, 17, 18, 19]. The computational cost of most sample-based methods is proportional to the number of model inputs. Li and Mahadevan [20] proposed a modularized method, which has a computational cost that is not proportional to the model input dimension, to estimate the first-order Sobol’ indices based on stratification of available input-output samples. DeCarlo et al. [21] proposed an importance sampling approach by introducing weights to different data points to estimate Sobol’ indices from available data using Sobol’ sequences to reduce the number of simulations; this method computes both first-order and higher order indices, and is able to include correlated inputs. Approximations to the joint probability distribution of inputs and outputs such as multivariate Gaussian, Gaussian copula, and Gaussian mixture have recently been found to give rapid estimation of Sobol’ indices [17].

The third scenario is of interest in this paper, where both a physics-based model and some experimental or real-world data are available. One option, if adequate data is available, is to simply build a regression or machine learning (ML) model based on the observation data, and use this model to perform GSA. Multiple recent studies have pursued data-driven ML models in situations where abundant experimental data or real-world observations are available due to advances in modern sensing techniques. Generally, the construction of data-driven ML models does not require in-depth knowledge of the complex physics inherent in the physical process. ML models can learn complex systems using available observations, but the accuracy of these models depends on the quality and quantity of the data. If the available data is sparse, then the complexity of the process may not be fully captured. Further, since purely data-driven ML models do not explicitly consider physical laws, they can produce physically inconsistent results. In such cases, incorporating physics knowledge within ML models may improve the accuracy and efficiency of GSA computations.

Information fusion and machine learning for sensitivity analysis using physics knowledge and experimental data

The combined use of physics-based and ML models has been shown to achieve more accurate and physically consistent predictions by leveraging the advantages of each method [22, 23, 24, 25].

In this work, we incorporate physics knowledge into the ML models to better capture the physics of the process by leveraging physical laws while improving the generalization performance of data-driven models. Two types of strategies are considered for incorporating physics knowledge within ML models: (1) incorporating loss functions in the ML model training to enforce physics constraints, and (2) pre-training the ML model with data generated by the physics model and then updating it with experimental data. Note that the first strategy does not use the physics model but only constraints for the output to obey physical requirements; whereas, the second strategy explicitly uses the physics computational model. Two types of ML models are considered in this paper, namely, Gaussian process (GP) and deep neural network (DNN). These two models are selected in order to represent two different kinds of available ML techniques; the GP model is one of the surrogate models commonly used in uncertainty quantification (UQ) studies, and DNN belongs to the emerging class of deep learning algorithms revolutionizing the field of artificial intelligence, spurred by recent advances in sensing, communication and computational resources. Four different physics-informed machine learning (PIML) models are developed for each type (i.e., GP or DNN) to predict the output quantity of interest (QoI), through combinations of the two strategies. The resulting GSA procedure incorporates the effect of uncertainty in the ML or PIML model, and the various models and strategies are compared in terms of accuracy and uncertainty in the GSA results and their computational demand.

In summary, the contributions of this paper are as follows:

• Physics knowledge and experimental observations are fused in order to maximize the accuracy of sensitivity estimates.

• Two PIML strategies and their combinations are investigated for global sensitivity analysis using both physics knowledge and experimental data.

• Four different models are built for each of GP and DNN, and the uncertainties in these models are included in the Sobol’ indices computation.

• The accuracy, uncertainty and computational effort of different options for the ML and PIML models are evaluated and compared.

The outline of the rest of the paper is as follows. Section 2 provides background information on related methods. Section 3 presents the proposed methodology. Two numerical examples are presented in Section 4 to illustrate the proposed methodology and draw insights on the performance of various PIML strategies and models. Concluding remarks are provided in Section 5.

## 2 Background

This section introduces each of the basic techniques used in developing the proposed methodology, namely variancebased GSA, Gaussian process (GP) surrogate modeling, and deep neural networks (DNN). These techniques are well established with extensive literature, therefore only a brief introduction is given here.

## 2.1 Variance-based GSA

Consider a deterministic real integrable one-to-one system response function $\mathrm { Y } = f ( \mathbf { X } )$ , where $f ( \cdot )$ is the computational model, $\mathbf { X } = \{ \mathrm { X } _ { 1 } , . . . , \mathrm { X } _ { k } \}$ are mutually independent model inputs, and Y is the model output. As shown in [10], the variance of Y can be decomposed as

$$
V ( \mathrm { Y } ) = \sum _ { i } ^ { k } V _ { i } + \sum _ { i _ { 1 } } ^ { k } \sum _ { i _ { 2 } = i _ { 1 } + 1 } ^ { k } V _ { i _ { 1 } i _ { 2 } } + \sum _ { i _ { 1 } } ^ { k } \sum _ { i _ { 2 } = i _ { 1 } + 1 } ^ { k } \sum _ { i _ { 3 } = i _ { 2 } + 1 } ^ { k } V _ { i _ { 1 } i _ { 2 } i _ { 3 } } + \ldots + V _ { 1 2 \ldots k }\tag{1}
$$

where $V _ { i }$ is the variance of Y due to $\mathrm { X } _ { i }$ alone, and $V _ { i _ { 1 } \dots i _ { p } } ( p \geq 2 )$ indicates the variance of $\mathrm { Y }$ caused by the interaction of $\{ \mathrm { X } _ { i _ { 1 } } , . . . , \mathrm { X } _ { i _ { p } } \}$

The Sobol’ indices are defined by dividing both sides of Eq. (1) with $V ( \mathrm { Y } )$

$$
1 = \sum _ { i } ^ { k } S _ { i } + \sum _ { i _ { 1 } } ^ { k } \sum _ { i _ { 2 } = i _ { 1 } + 1 } ^ { k } S _ { i _ { 1 } i _ { 2 } } + \sum _ { i _ { 1 } } ^ { k } \sum _ { i _ { 2 } = i _ { 1 } + 1 } ^ { k } \sum _ { i _ { 3 } = i _ { 2 } + 1 } ^ { k } S _ { i _ { 1 } i _ { 2 } i _ { 3 } } + \ldots + S _ { 1 2 \ldots k }\tag{2}
$$

Information fusion and machine learning for sensitivity analysis using physics knowledge and experimental data

where $S _ { i }$ is the first-order or main effects index that assesses the contribution of $\mathrm { X } _ { i }$ individually to the variance of the output $\mathrm { Y }$ without considering interactions with other inputs. The higher-order indices $S _ { i _ { 1 } . . . i _ { p } } ( p \geq 2 )$ in Eq. (2) measure the contributions of the interactions of $\{ \mathrm { X } _ { i _ { 1 } } , . . . , \mathrm { X } _ { i _ { p } } \}$

The first-order index $S _ { i }$ is defined as follows:

$$
S _ { i } = \frac { V _ { i } } { V ( \Upsilon ) } = \frac { V _ { X _ { i } } ( E _ { \mathbf { X } _ { - i } } ( \Upsilon | \mathrm { X } _ { i } ) ) } { V ( \Upsilon ) }\tag{3}
$$

where ${ \bf X } _ { - i }$ are all the model inputs other than $\mathrm { X } _ { i }$

The overall contribution of $\mathrm { X } _ { i }$ considering an individual input and its interactions with all other inputs is measured by the total effects index $S _ { i } ^ { T }$

$$
S _ { i } ^ { T } = 1 - \frac { V _ { - i } } { V ( \Upsilon ) } = \frac { V _ { \mathbf { X } _ { - i } } ( E _ { \mathbf { x } _ { i } } ( \Upsilon | \mathbf { X } _ { - i } ) ) } { V ( \Upsilon ) } .\tag{4}
$$

The computation of $S _ { i }$ analytically is nontrivial since $E _ { \mathbf { X } _ { - i } } ( \cdot )$ requires multi-dimensional integrals. A basic samplingbased approach is to use double-loop sampling [10]. Several approaches to reduce the computational cost were mentioned in Section 1. One of these approaches of particular relevance to this paper is to replace the original computational model $f ( \cdot )$ by a surrogate model and use this surrogate model in GSA [26, 27, 28, 29, 30]. This approach will be addressed further in Section 3.

## 2.2 Gaussian process surrogate modeling

A Gaussian process (GP) surrogate model (or kriging) approximates a response function $y = G ( \mathbf { x } )$ over the domain of input x as a Gaussian random process with a mean function $m ( \mathbf { x } )$ and a covariance function $k ( \dot { \mathbf { x } } , \dot { \mathbf { x } } ^ { \prime } )$ , which describes the deviation of the model from the trend

$$
\mathbf { y } ( \mathbf { x } ) \sim \mathcal { G P } \big ( m ( \mathbf { x } ) , k ( \mathbf { x } , \mathbf { x } ^ { \prime } ) \big ) .\tag{5}
$$

Given a set of training data $\{ \mathbf { X } _ { T } , \mathbf { Y } _ { T } \}$ and the input $\mathbf { X } _ { P } ,$ , where prediction is desired, the conditional probability distribution of the output $\mathbf { Y } _ { P }$ follows a multivariate Gaussian distribution [31] as

$$
\begin{array} { r l } & { { \bf Y } _ { P } | { \bf X } _ { P } , { \bf X } _ { T } , { \bf Y } _ { T } \sim \mathcal { N } ( \pmb { \mu } , { \Sigma } ) } \\ & { \pmb { \mu } = m ( { \bf X } _ { P } ) + { \Sigma } _ { P T } ( { \Sigma } _ { T T } + \sigma _ { o b s } ^ { 2 } { \pmb I } ) ^ { - 1 } \big ( { \bf Y } _ { T } - m ( { \bf X } _ { T } ) \big ) } \\ & { \Sigma = { \Sigma } _ { P P } - { \Sigma } _ { P T } \big ( { \Sigma } _ { T T } + \sigma _ { o b s } ^ { 2 } { \pmb I } \big ) ^ { - 1 } { \Sigma } _ { P T } ^ { T } } \end{array}\tag{6}
$$

where $\pmb { \mu }$ is the mean vector of the prediction ${ \bf Y } _ { P }$ conditioned on the training data, and Σ is the conditional covariance matrix of ${ \bf Y } _ { P } ; { \Sigma } _ { P T }$ is the covariance matrix between the prediction data $\mathbf { X _ { P } }$ and each of the training points $\left\{ \mathbf { X _ { T } } = \right.$ $x _ { 1 } , x _ { 2 } , . . . , x _ { n } \} ; \Sigma _ { T T }$ is the $n \times n$ covariance matrix of the training data; $\Sigma _ { P P }$ is the unconditional covariance matrix of $\mathbf { Y } _ { P } ; \sigma _ { o b s } ^ { 2 }$ is the variance of observation/measurement error (also called noise variance), and I is the identity matrix.

The mean function $m ( \cdot )$ is often formulated as a polynomial function of inputs. The covariance function $k ( \cdot )$ can be formulated by using a covariance function based on the desired properties (order of continuity, stationary/non-stationary, isotropic/anisotropic). A squared exponential correlation function with separate length scale parameters $l _ { i }$ for each input dimension has often been used in the literature:

$$
k ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \sigma _ { f } ^ { 2 } \exp \left[ - \sum _ { i = 1 } ^ { \mathrm { M } } \frac { ( x _ { i } - x _ { i } ^ { \prime } ) ^ { 2 } } { 2 l _ { i } } \right] + \sigma _ { o b s } ^ { 2 } \delta ( \mathbf { x } , \mathbf { x } ^ { \prime } )\tag{7}
$$

where $\sigma _ { f } ^ { 2 }$ is the signal/process variance and defines the maximum allowable covariance, and $\delta \bf { ( x , x ^ { \prime } ) }$ is the Kronecker delta function.

The hyperparameters of the GP model considering a zero mean function, i.e., $\boldsymbol { \Theta } = \{ l , \sigma _ { f } , \sigma _ { o b s } \}$ , are inferred from the training data. A common method is to maximize the log marginal likelihood function, which is defined as

$$
\log p ( { \mathbf { Y } _ { T } | \mathbf { X } _ { T } ; \Theta } ) = - \frac { 1 } { 2 } { \mathbf { Y } _ { T } ( \Sigma _ { T T } + \sigma _ { o b s } ^ { 2 } I ) ^ { - 1 } } { \mathbf { Y } _ { T } } - \frac { 1 } { 2 } \mathrm { l o g } | \Sigma _ { T T } + \sigma _ { o b s } ^ { 2 } I | - \frac { n } { 2 } \mathrm { l o g } 2 \pi .\tag{8}
$$

## 2.3 Deep neural networks

In recent years, due to the confluence of advanced sensing and imaging techniques, big data processing techniques, enormous computational power and the internet, rapid advances are being made in developing sophisticated data-driven machine learning models, particularly neural networks. A deep neural network (DNN) is composed of multiple hidden layers and has four major components: neuron, activation function, cost function, and optimization. Figure 1 shows a neural network consisting of three inputs, two hidden layers, each having four neurons, and two output neurons. The values of various input variables of a particular neuron are multiplied by their associated weights, then the sum of the products of the neuron weights and the inputs are calculated at each neuron. The summed value is passed through an activation function that maps the summed value to a fixed range before passing these signals on to the next layer of neurons.

![](images/b1ce4a9e9c838051a48d306510dbadb839a8976d85e0777d81e03627b09d8edf.jpg)  
Figure 1: A deep neural network with two hidden layers.

The predictions of the DNN after forward propagation, $\hat { \mathbf Y }$ , are compared against the observations, $\mathbf { Y } _ { o b s } .$ , by defining a loss function (e.g., root mean squared error (RMSE); $\mathcal { L } _ { \mathrm { R M S E } } ( \mathbf { Y } _ { o b s } , \hat { \mathbf { Y } } ) = \sqrt { \sum _ { i = 1 } ^ { n } ( \mathrm { y } _ { o b s , i } - \hat { \mathrm { y } } _ { i } ) ^ { 2 } / n ) }$ , which measures how far off the predictions are from the observations for the n training samples. Backpropagation algorithms are employed to keep track of small perturbations to the weights that affect the error in the output and to distribute this error back through the network layers by computing gradients for each layer using the chain rule. In order to minimize the value of the loss function, necessary adjustments are applied at each iteration to the neuron weights in each layer of the network. These procedures are performed at each iteration until the loss function converges to a stable value.

## 3 Proposed methodology

The proposed methodology for sensitivity analysis, using both physics knowledge and experimental data, consists of the following steps:

1. Identification of PIML strategies

2. Implementation of PIML strategies in ML models

3. Variance quantification in ML model prediction

4. Sobol’ indices computation with ML model prediction variance

The following subsections describe these steps in detail.

## 3.1 Identification of PIML strategies

PIML models seek to incorporate physics knowledge or constraints within the data-driven ML models. When a mechanistic, physics-based model is also available, complementary strengths of both mechanistic and ML models can be leveraged in a synergistic manner [24]. In the latter case, the aim is to improve the predictions beyond that of physics-based models or ML models alone by coupling physics-based models with ML models. Thus two different strategies to combine physics knowledge and ML models can be considered: (1) incorporate physics constraints in the ML models, and (2) pre-train and update the ML models using physics model input-output and experimental data, respectively.

Information fusion and machine learning for sensitivity analysis using physics knowledge and experimental data

## 3.1.1 Strategy 1: Enforcing physics constraints

A direct strategy to enforce physics constraints in ML model predictions is by including the constraints within the loss function used in training the ML model [22]. Thus, while training a PIML model with inputs $\mathbf { X } _ { o b s }$ and outputs $\mathbf { Y } _ { o b s }$ the physical constraints can be incorporated as additional penalty terms in the loss function:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { M L } } + \lambda _ { \mathrm { p h y } } \mathcal { L } _ { \mathrm { p h y } } ( \hat { \mathbf { Y } } ) ,\tag{9}
$$

where ${ \mathcal { L } } _ { \mathrm { M L } }$ is the log marginal likelihood of the data for a GP model:

$$
\mathcal { L } _ { \mathrm { G P } } = \log p ( \mathbf { Y } _ { o b s } | \mathbf { X } _ { o b s } ; \boldsymbol { \Theta } )\tag{10}
$$

with experimental observations $\{ \mathbf { X } _ { o b s } , \mathbf { Y } _ { o b s } \}$ being the training data for the GP model. For a DNN ${ \mathcal { L } } _ { \mathrm { M L } }$ represents training a loss function that evaluates a supervised error, e.g., root mean squared error (RMSE):

$$
\mathcal { L } _ { \mathrm { D N N } } ( \mathbf { Y } _ { o b s } , \hat { \mathbf { Y } } ) = \sqrt { \sum _ { i = 1 } ^ { n } \frac { ( Y _ { o b s , i } - \hat { Y } _ { i } ) ^ { 2 } } { n } }\tag{11}
$$

which measures the accuracy of predictions $\hat { \mathbf Y }$ for n training samples. (Note that for the GP model, the likelihood is maximized, whereas for the DNN model, the RMSE is minimized). The additional physics constraint loss function $\mathcal { L } _ { \mathrm { p h y } }$ in the second term of Eq. (9) is weighted by a hyperparameter $\lambda _ { \mathrm { p h y } } ;$ the value of $\lambda _ { \mathrm { p h y } }$ controls the strength of the physics constraint enforcement. The inclusion of the second term ensures physically consistent model predictions and helps to reduce the generalization error, which is a measure of how accurately a model is able to predict the output QoI for previously unseen data [22].

The physical inconsistencies in the model predictions are evaluated using the physics constraint loss term. The generic forms of these physical relationships can be expressed using the following constraints:

$$
\begin{array} { r } { \mathcal { F } _ { 1 } ( \hat { \mathbf { Y } } , \mathbf { r } ) = 0 , } \\ { \mathcal { F } _ { 2 } ( \hat { \mathbf { Y } } , \mathbf { r } ) \leq 0 . } \end{array}\tag{12}
$$

where Γ denotes other variables or thresholds that define the physics constraint regarding the model output $\hat { \mathbf Y }$ Equation. 12 indicates that the constraints may take the form of equalities or inequalities. These equations can involve algebraic relationships or partial differentials of $\hat { \mathbf Y }$ and/or Γ. The physics-based loss functions for these equations can be defined as:

$$
\mathcal { L } _ { \mathrm { p h y } } ( \hat { \mathbf { Y } } ) = | | \mathcal { F } _ { 1 } ( \hat { \mathbf { Y } } , \mathbf { \Gamma } , \mathbf { \Gamma } ) | | + \mathrm { R e L U } ( \mathcal { F } _ { 2 } ( \hat { \mathbf { Y } } , \mathbf { \Gamma } , \mathbf { \Gamma } ) ) ,\tag{13}
$$

where ReL $\boldsymbol { \mathrm { \Pi } } _ { l } \mathrm { U } ( \mathbf { x } ) = \operatorname* { m a x } ( 0 , \mathbf { x } )$ represents the rectified linear unit function and it acquires value when a threshold is violated in the inequality constraint, i.e., it penalizes the optimization when $\mathcal { F } _ { 2 } > 0 .$ . It can also be used to penalize deviations from a desired physically consistent relationship among multiple outputs $\hat { \mathbf Y }$ [32].

## 3.1.2 Strategy 2: Pre-training and Updating

The ML model output accuracy and uncertainty are dependent on the quality and quantity of the available training data. In some systems, the high cost associated with conducting experiments makes it infeasible to have adequate amount of training data to build purely data-driven models. Thus, it may be desirable to combine the physics-based model and available experimental data in seeking to maximize the accuracy of the sensitivity estimates. When the experiments are expensive, they can only be conducted for a few values of the inputs, whereas it might be possible to run the physics-based model for a larger set of input values. In that case, the simulation data can be used to pre-train an ML model, which is used as the initial model to be updated with experimental observations. Further, training of ML models requires the choice of initial values of the model parameters. The transfer of physical knowledge using a pre-trained ML model can prevent poor initialization due to lack of knowledge regarding the initial choice of ML model parameters prior to training.

Since the pre-training based on the physics-based model can use a large amount of training data (with multiple input parameter combinations) over a wide range of values, the pre-training may also help the eventual ML model to have wider generalization beyond experimental data. In the numerical example in Section 4, the pre-training strategy exercises the physics model over 1310 input combinations, whereas only 39 experiments are available. However, if the physics model is computationally expensive, then the advantage of the pre-training strategy in using a larger input data set (for physics model runs) compared to the experiments becomes limited.

The two proposed strategies to predict the QoI are shown in Fig. 2. Figure 2(a) shows the first method, where the physical knowledge is included through constraints within the loss function of an ML trained with only experimental data. Figure 2(b) shows the second method, where an ML model is first trained with data generated using the physics-based model and then updated using experimental data. Figures 2(c) and 2(d) show the trained ML model prediction $( \hat { Y } )$ for the two proposed strategies, respectively. The proposed PIML strategies can be applied to any physical system by leveraging the related physical constraints or physics-based models.

Training  
![](images/12337aacd885e57e2507a9c4df0b095c97d64e93c0802cabab11b6b5512294a8.jpg)  
(a)

![](images/c84d196768e5dc25e172eaec26bd84da424dd2646afeafe8ae4fb5c3f1f1ac02.jpg)  
(b)

![](images/caa014c7ddc2c64516160d498a7af9955e9dfe8a37bed2ac091daf4c86f6c559.jpg)  
(c)

Prediction  
![](images/71777f054c03b72006707b0472b108b54ec41b2c5d9570b6e860995d34faad12.jpg)  
(d)  
Figure 2: PIML strategies: (a) incorporating physics-based loss functions in the ML models to enforce physics constraints, (b) pre-training an ML model with physics model input-output $( \mathbf { X } _ { \mathrm { p h y } } , \mathbf { Y } _ { \mathrm { p h y } } )$ and updating it with experimental training data $( \mathbf { X } _ { \mathrm { o b s } } , \mathbf { Y } _ { \mathrm { o b s } } )$ , (c) the trained ML model prediction (Y<sup>ˆ</sup> ), (d) updated ML model prediction (Y<sup>ˆ</sup> ).

## 3.2 Implementation of PIML strategies in ML models

Based on the proposed two strategies to incorporate physics knowledge into the ML model, four separate ML model can be constructed for each type of surrogate model considered here (i.e., GP and DNN):

1. GP

5. DNN

2. $\mathrm { G P ^ { \mathcal { L } _ { \mathrm { p h y } } } }$

6. $\mathrm { D N N } ^ { \mathcal { L } _ { \mathrm { p h y } } }$

3. $\mathrm { G P ^ { u p d } }$

7. $\mathrm { D N N ^ { u p d } }$

4. $\mathrm { G P ^ { u p d _ { 1 } , } } { \mathcal { L } } _ { \mathrm { p h y } }$

8. $\mathrm { D N N } ^ { \mathrm { u p d , \mathcal { L } _ { \mathrm { p h y } } } }$

These different models cover the following options: model trained with experimental data alone, models trained with PIML strategies 1 or 2 alone, and models trained with both PIML strategies together. The implementations of PIML strategies 1, 2, and their combination are different for the GP models vs. the DNN models. The following subsections describe how the PIML strategies can be implemented for each of the above models.

## 3.2.1 Implementation of PIML in GP models

In Model 1, denoted as GP, only experimental observations are used for training. The hyperparameters of the GP model (process variance, correlation length scale along each input dimension, and trend function coefficients, and also measurement error variance if unknown) are optimized during training by maximizing the log marginal likelihood function shown in Eq. (8). In calculating the likelihood, the difference between the true response of the system $\mathbf { Y _ { \mathrm { t r u e } } }$ and the observed response $\mathbf { Y } _ { \mathrm { o b s } }$ is attributed to the observation error $\epsilon _ { \mathrm { o b s } }$ , which is often treated as a zero-mean Gaussian random variable with variance $\sigma _ { \mathrm { o b s } } ^ { 2 } .$

Model 2, denoted as $\mathrm { G P ^ { \mathcal { L } _ { \mathrm { p h y } } } }$ , incorporates the first PIML strategy by enforcing physics constraints during the optimiza tion of the GP model hyperparameters. More specifically, the physics constraints are included during the maximization of the log marginal likelihood function (Eq. (8)) while inferring the hyperparameters of the GP model. Thus, the training of Model 2 is achieved by maximizing the function in Eq. (14):

$$
\mathcal { L } _ { \mathrm { G P } } = \log p ( \mathbf { Y } _ { o b s } | \mathbf { X } _ { o b s } ; \boldsymbol { \Theta } ) - \lambda _ { \mathrm { p h y } } \mathcal { L } _ { \mathrm { p h y } } ( \hat { \mathbf { Y } } ) ,\tag{14}
$$

where $\hat { \mathbf Y }$ is the GP model prediction. Note that since ${ \mathcal { L } } _ { \mathrm { G P } }$ is to be maximized, the second term corresponding to the physics constraint has a negative sign. Gaussian process modeling under constraints has been studied in the literature [33, 34, 35, 36, 37]. Veiga et al. [34] developed a framework that incorporates bound, monotonicity and convexity constraints in $\mathrm { G P }$ modeling. Golchi et al. [35] developed a Bayesian approach to $\mathrm { G P }$ modeling that incorporates the monotonicity constraint. The need to obtain the monotonicity information at each of the points in the derivative input set can slow down the computation as the input dimension increases since the size of the covariance matrix depends on the input dimension. This paper does not use the methods described above. Instead, the paper proposes a different method that penalizes violations of the physics constraints by introducing a regularization term in the likelihood function. To the best of our knowledge, this is the first study to apply the penalty approach to the likelihood function of the GP model. Further, the computational effort of the proposed method (i.e., the calculation of the regularization term) does not increase with the problem size since the regularization term does not need the inverse of the covariance matrix.

The current work considers two different approaches for the second PIML strategy. Both approaches use a pre-trained model, obtained using the physics model input-output data. In the first approach, the model parameters are updated using the experimental data; and in the second approach, a discrepancy correction term is added to the pre-trained model. The first step of pre-training using data generated by the physics model can be thought of as similar to a lower fidelity model, and the second step of improving the pre-trained model (either by parameter updating or by adding a discrepancy term) can be thought of as similar to incorporating higher fidelity data (experimental data, in this case) to improve the model. Various multi-fidelity modeling strategies with different combinations of the low- and high-fidelity models have been studied in the literature, such as filtering, fusion, and adaptation [38]. The discrepancy correction approach pursued here adopts the simplest strategy, namely, additive correction, where a model discrepancy term is added to the low-fidelity model [39]:

$$
f _ { H F } ( \mathbf { X } ) = f _ { L F } ( \mathbf { X } ) + \delta ( \mathbf { X } ; \pmb { \theta } _ { \delta } )\tag{15}
$$

where $f _ { H F } ( \cdot )$ and $f _ { L F } ( \cdot )$ are the high and low fidelity models, respectively, X is the input to the model, and $\theta _ { \delta }$ are the parameters of the discrepancy correction term. The correction term can be obtained using any suitable surrogate modeling technique.

Model 3, denoted as $\mathrm { G P ^ { u p d } }$ , pursues the first approach of the second PIML strategy, where a GP model is pre-trained using the coupled multi-physics model input-output and then updated with experimental data. Then, the model parameters of this pre-trained network are updated using the experimental data.

An alternative approach, denoted as ${ \mathrm { G P } } ^ { \mathrm { M F } }$ , pursues the second approach of the second PIML strategy, i.e., it pre-trains a GP surrogate model with data generated from the physics model, then improves the surrogate using experimental data. Consider a physics model $G ( \cdot )$ that maps input variables X and model parameters $\theta _ { m }$ to the numerical model output $\mathbf { Y } _ { m } .$

$$
\mathbf { Y } _ { m } ( \mathbf { X } ) = G \big ( \mathbf { X } ; \theta _ { m } ( \mathbf { X } ) \big ) .\tag{16}
$$

Let $n _ { D }$ be the number of collected observation data $\mathbf { Y _ { \mathrm { o b s } } }$ from experiments with input variable settings $\mathbf { x } ^ { ( 1 ) } , . . . , \mathbf { x } ^ { ( n _ { D } ) }$ where $\mathbf { x } ^ { ( i ) }$ is the input variable setting for the ith experiment. The physics model prediction is inaccurate due to missing physics or due to other approximations. Thus, a model discrepancy term $\delta ( \mathbf { X } )$ as a function of model inputs is introduced to capture the difference between $\mathbf { Y _ { \mathrm { t r u e } } }$ and $\mathbf { Y } _ { m }$ [40]:

$$
\mathbf { Y } _ { \mathrm { t r u e } } ( \mathbf { X } ) = \mathbf { Y } _ { \mathrm { m } } ( \mathbf { X } ) + \delta ( \mathbf { X } ) .\tag{17}
$$

The experimental observations $\mathbf { Y _ { \mathrm { o b s } } }$ can be described in terms of the true system response $\mathbf { Y _ { \mathrm { t r u e } } }$ and the corresponding observation errors $\epsilon _ { \mathrm { o b s } } ( \mathbf { X } )$ as

$$
\mathbf { Y } _ { \mathrm { o b s } } ( \mathbf { X } ) = \mathbf { Y } _ { \mathrm { t r u e } } ( \mathbf { X } ) + \epsilon _ { \mathrm { o b s } } ( \mathbf { X } ) .\tag{18}
$$

Combining Eqs. (17) and (18),

$$
{ \bf Y } _ { \mathrm { o b s } } ( { \bf X } ) - \epsilon _ { \mathrm { o b s } } ( { \bf X } ) = { \bf Y } _ { m } ( { \bf X } ) + \delta ( { \bf X } ) = G \big ( { \bf X } ; ~ \theta _ { m } ( { \bf X } ) \big ) + \delta ( { \bf X } ) .\tag{19}
$$

When the physics model is computationally expensive, it is replaced by a cheaper surrogate model. In Model 3, a GP surrogate model is used to approximate the original physics model. The accuracy of the surrogate model prediction depends on the quality and quantity of the training data generated by the original physics model. The surrogate model error $( \epsilon _ { \delta } ( \mathbf { X } ) )$ can be incorporated as follows:

$$
\mathbf { Y } _ { m } ( \mathbf { X } ) = \hat { \mathbf { Y } } _ { m } ( \mathbf { X } ) + \epsilon _ { \delta } ( \mathbf { X } ) ,\tag{20}
$$

where $\hat { \mathbf { Y } } _ { m }$ is the surrogate model prediction.

A common approach to estimate the discrepancy term $\delta ( \mathbf { X } )$ is the one formulated by Kennedy and O’Hagan [40], which is applicable in the context of Bayesian calibration. In that case, physics model parameters are sought to be calibrated, and a discrepancy term is added in the calibration equation. The discrepancy term can be expressed in multiple ways, such as constant, Gaussian random variable with unknown parameters (either input-dependent or not), or Gaussian process (either stationary or non-stationary) [41]. The hyperparameters of the discrepancy term are then estimated along with the physics model parameters using Bayesian calibration [42].

However, the situation considered here is much simpler. There is no calibration of the physics model parameters here; only the discrepancy term is needed. (In other words, the physics model parameters are already established). In that case, the model discrepancy can be evaluated for different input values of experimental tests and realizations of observation errors as follows:

$$
\delta ( \mathbf { X } ) = \mathbf { Y } _ { \mathrm { o b s } } ( \mathbf { X } ) - \epsilon _ { \mathrm { o b s } } ( \mathbf { X } ) - { \hat { \mathbf { Y } } } _ { m } ( \mathbf { X } ) - \epsilon _ { \delta } ( \mathbf { X } ) .\tag{21}
$$

Moving the surrogate model error $\epsilon _ { \delta } ( \mathbf { X } )$ to the left-hand side, we can express the difference between the actual response and $\bar { \mathbf { G P } }$ model prediction as

$$
\hat { \delta } ( \mathbf { X } ) = \delta ( \mathbf { X } ) + \epsilon _ { \delta } ( \mathbf { X } ) = \mathbf { Y } _ { \mathrm { o b s } } ( \mathbf { X } ) - \epsilon _ { \mathrm { o b s } } ( \mathbf { X } ) - \hat { \mathbf { Y } } _ { m } ( \mathbf { X } ) .\tag{22}
$$

In this work, a second GP model is trained for $\hat { \delta } ( { \mathbf X } )$ in terms of the inputs. Thus two GP models are trained in Model 3. The first GP model is constructed using the physics model input-output data, and predicts $\hat { \mathbf { Y } } _ { m }$ . The second GP model is constructed using the experimental data and the corresponding surrogate model predictions, and predicts $\hat { \pmb \delta }$ (the difference between the surrogate model prediction and actual system response).

The GP model for the model discrepancy captures the combined contribution of measurement error, physics and surrogate model errors for a given prediction. Thus, the predictions of the first GP model (pre-trained) are corrected with the second GP model predictions $( \hat { \delta } )$ representing the model discrepancy term and can be written as

$$
\hat { \mathbf { Y } } ( \mathbf { X } ) = \hat { \mathbf { Y } } _ { m } ( \mathbf { X } ) + \hat { \pmb { \delta } } .\tag{23}
$$

Model 4, denoted as $\mathrm { G P ^ { u p d , } } \mathcal { L } _ { \mathrm { p h y } }$ , combines both PIML strategies for GP, where the optimized model parameters of the pre-trained model based on the physics model input-output are used as the initial values. These model parameters are updated using the experimental data by minimizing the augmented loss function shown in Eq. (14) which consists of both the training loss function and the physics constraint loss terms.

## 3.2.2 Implementation of PIML in DNN models

In Model 5, denoted simply as DNN, a deep neural network is trained using only experimental data. In order to train the model, an optimization algorithm is used to find a set of model parameters (weights and biases) that best map inputs to outputs. The number of epochs, which is the number of complete passes through a batch of training dataset, layers and neurons needs to be optimized to reduce cost and improve model accuracy. The model accuracy is evaluated using a validation dataset not used for training; the number of epochs is gradually increased until the accuracy improvement is insignificant. The accuracy and generalization performance of the DNN are also affected by the dropout rate used in training the model; the dropout concept will be discussed in Section 3.3 below, and the selection of the dropout rate will be discussed in Section 4.

Model 6, denoted as $\mathrm { D N N } ^ { \mathcal { L } _ { \mathrm { p h y } } }$ , extends Model 5 by implementing the first PIML strategy, i.e., physical knowledge related to the physical process is enforced through constraints within the loss function of the DNN, as shown in Eq. (9). The physics-based loss function terms are evaluated for given experimental inputs at every optimization iteration during the training of DNN<sup>Lphy</sup> ; this makes the optimization process slower during the training. In particular, the multipliers in the physics constraint penalty terms $( \lambda _ { \mathrm { p h y } } ^ { \mathrm { D N N } } )$ affect the training speed; a higher value of the multiplier makes the penalty stronger (i.e., the physics constraints more stringent), thus requiring an increased number of iterations to converge to the optimum solution.

Information fusion and machine learning for sensitivity analysis using physics knowledge and experimental data

Model 7, denoted as $\mathrm { D N N ^ { u p d } }$ , pursues the second PIML strategy, where a DNN model is pre-trained using the coupled multi-physics model input-output and then updated with experimental data. The pre-trained model is first trained with physics model input-output data consisting of input combinations over a range of values. Then, the weights and biases (model parameters) of this pre-trained network are updated using the experimental data.

Model 8, denoted as $\mathrm { D N N } ^ { \mathrm { u p d , } \mathcal { L } _ { \mathrm { p h y } } }$ , combines both PIML strategies for DNN, where the optimized model parameters of the pre-trained model based on the physics model input-output are used as the initial values. These model parameters are updated using the experimental data by minimizing the augmented loss function shown in Eq. (11) which consists of both the training loss function and the physics constraint loss terms. Similar to Model 6, the inclusion of physics constraints makes it slower for the optimization to converge to optimal model parameter values.

## 3.3 Variance of GP and DNN prediction

In general, every surrogate model has uncertainty in prediction, whether acknowledged or not. In the GP models, the prediction at a given input is expressed by a normal distribution with a mean and variance. In order to quantify the uncertainty in the GP prediction, we can sample multiple realizations of the Gaussian process. Note that this only captures the variance of the GP prediction, not the bias, which can be evaluated by comparing against validation data.

In the DNN models, the estimates of the model parameters (neuron weights w) have uncertainty, and this uncertainty depends on the available training data. When the neural network parameters are represented using distributions (to reflect the epistemic uncertainty) instead of deterministic values, the model is referred to as a Bayesian neural network (BNN) [43, 44, 45]. In this Bayesian context, the model parameter uncertainty is first described using a prior distribution $p ( \mathbf { w } )$ , and the likelihood function is $p ( \mathbf { Y } | \mathbf { X } , \mathbf { w } )$ . Following Bayes’ theorem, a posterior distribution over the model parameters given the training data $\{ \bar { \mathbf { X } } _ { T } , \dot { \mathbf { Y } } _ { T } \} = \{ \{ \mathbf { x } _ { 1 } , . . . , \bar { \mathbf { x } _ { N } } \} , \{ \{ \mathbf { y } _ { 1 } , . . . , \mathbf { y } _ { N } \} \}$ is defined by

$$
p ( \mathbf { w } | \mathbf { X } , \mathbf { Y } ) = \frac { p ( \mathbf { Y } | \mathbf { X } , \mathbf { w } ) p ( \mathbf { w } ) } { p ( \mathbf { Y } | \mathbf { X } ) } .\tag{24}
$$

In this context, the predictive distribution of the model outputs for a given input $\hat { \mathbf X }$ is given by:

$$
p ( \hat { \mathbf { Y } } | \hat { \mathbf { X } } , \mathbf { X } _ { T } , \mathbf { Y } _ { T } ) = \int _ { \Omega } p ( \hat { \mathbf { Y } } | \hat { \mathbf { X } } , \mathbf { w } ) p ( \mathbf { w } | \mathbf { X } _ { T } , \mathbf { Y } _ { T } ) d \mathbf { w } .\tag{25}
$$

The posterior distribution of model parameters $p ( \mathbf { w } | \mathbf { X } _ { T } , \mathbf { Y } _ { T } )$ is challenging to evaluate over the entire parameter space Ω due to the high dimensionality of Ω in a DNN model, and the highly non-linear behavior caused by the non-linear activation functions and their combinations across multiple hidden layers. Therefore, different approximate inference techniques can be considered to infer the posterior distribution $p ( \mathbf { w } | \mathbf { X } _ { T } , \mathbf { Y } _ { T } )$ [46, 47, 48, 49]. One such approximation is variational inference, which fits a simple and tractable distribution $q _ { \boldsymbol { \theta } } ( \mathbf { w } )$ to the posterior, parametrized by a variational parameter θ [46]. This approximates the intractable problem by optimizing the parameters of $q _ { \boldsymbol { \theta } } ( \mathbf { w } )$ . The quality of the variational inference can be assessed by the Kullback-Leibler (KL) divergence between the approximate distribution $q _ { \boldsymbol { \theta } } ( \mathbf { w } )$ and the true model posterior $p ( \mathbf { w } | \mathbf { X } _ { T } , \mathbf { Y } _ { T } )$ .

The term dropout refers to randomly dropping out neurons (along with their connections) with a given dropout rate during the training phase in a neural network. Dropout is a common regularization approach in neural network training, which prevents over-fitting and reduces generalization error. A Monte Carlo (MC) dropout technique has been developed in recent years in the context of Bayesian neural networks [50], which has been shown to be equivalent to performing approximate variational inference. In MC dropout, dropout is not only applied while training a model but also during prediction. Randomly chosen neurons are temporarily removed from the network along with their connections. Next, the gradients of neuron weights are calculated on a sub-neural network for each training data and these gradients are then averaged over the training sets to obtain the weights for the overall network. A Bayesian neural network with MC dropout generates random samples following a binomial distribution (0 or 1) for each neuron in the input and hidden layers during prediction. The neuron that takes the value 0 is dropped with probability $p _ { d } .$ . The outputs of the network are predicted using the collection of generated random samples from the posterior predictive distribution and the uncertainty in the prediction of a new data is quantified with the trained network. Thus the MC dropout strategy provides an efficient way of Bayesian inference to quantify the model prediction variance, and can be applied to a variety of neural networks, such as feedforward neural networks, convolutional neural networks, and recurrent neural networks [51].

The sensitivity estimate results depend on the dropout rate. The main reason for this is that the model is regularized and it underfits the data as it is over-regularized. Further, both the accuracy and uncertainty of the sensitivity estimates also depend on the number of training epochs, and the architecture of the network. If the model is not fully trained, it will also result in underfitting, leading to larger bias and variance in the prediction.

Information fusion and machine learning for sensitivity analysis using physics knowledge and experimental data

## 3.4 Sobol’ indices computation with model uncertainty

This section discusses the incorporation of ML model prediction variance within the estimation of Sobol’ indices using the GP and DNN models.

When the training data is noise-free, the GP predictions at the training points have zero variance and at other points the variance is non-zero. The prediction at any point is given by a normal distribution with a mean and variance. This prediction uncertainty can be captured by sampling multiple realizations of the GP model, which can then be used in GSA. The model uncertainty pertaining to the GP model is propagated to the Sobol’ index calculations using the following estimator (see [7]):

$$
S _ { i } ^ { \mathrm { G P } } = { \frac { V _ { X ^ { i } } \left( E _ { \mathbf { X } } \left[ \mathbf { y } _ { P } ( \mathbf { X } ) | X ^ { i } \right] \right) } { V \left( \mathbf { y } _ { P } ( \mathbf { X } ) \right) } } \approx { \frac { { \frac { 1 } { m } } \sum _ { k = 1 } ^ { m } \mathbf { y } _ { P } ( \mathbf { X } _ { k } ) \mathbf { y } _ { P } ( X _ { k } ^ { i } ) - { \frac { 1 } { m } } \sum _ { k = 1 } ^ { m } \mathbf { y } _ { P } ( \mathbf { X } _ { k } ) \sum _ { k = 1 } ^ { m } \mathbf { y } _ { P } ( X _ { k } ^ { i } ) } { { \frac { 1 } { m } } \sum _ { k = 1 } ^ { m } \mathbf { y } _ { P } ( \mathbf { X } _ { k } ) ^ { 2 } - \left[ { \frac { 1 } { m } } \sum _ { k = 1 } ^ { m } \mathbf { y } _ { P } ( \mathbf { X } _ { k } ) \right] ^ { 2 } } }\tag{26}
$$

where ${ \bf y } _ { P } ( { \bf X } )$ is a realization of the predictive distribution shown in Eq. (6) trained using experimental data, and $\mathbf { X } _ { k }$ and $X _ { k } ^ { i }$ are the kth samples of the random vectors X and $\mathbf { X } ^ { i }$

The distribution of $S _ { i } ^ { \mathrm { G P } }$ can be computed by sampling $N _ { Z }$ realizations from the Gaussian predictive distribution ${ \bf Y } _ { P } ( { \bf X } )$ numerically using Algorithm 1.

Algorithm 1 Estimation of the distribution of $S _ { i } ^ { \mathrm { G P } }$ using GP models.   
1: Generate two samples ${ \bf X } _ { k }$ and $X _ { k } ^ { i } \left( k = 1 , \ldots , m \right)$ of the random vectors X and $X ^ { i }$   
2: for $p = 1 , 2 , \ldots , N _ { Z }$ do   
3: Sample a realization ${ \bf y } _ { P } ( { \bf x } )$ of ${ \bf Y } _ { P } ( { \bf X } )$ with $\mathbf { x } = \{ ( x _ { k } ) _ { k = 1 , \ldots , m } , ( x _ { k } ^ { i } ) _ { k = 1 , \ldots , m } \}$   
4: Compute $\hat { S } _ { i , p } ^ { \mathrm { G P } }$ using Eq. (26).   
5: end for   
return $( \hat { S } _ { i , p } ^ { \mathrm { G P } } ) _ { p = 1 , 2 , \dots , N _ { Z } } .$

The output of Algorithm $1 ( \hat { S } _ { i , p } ^ { \mathrm { G P } } ) _ { p = 1 , 2 , \ldots , N z }$ is a sample of size $N _ { Z }$ , where m is the number of Monte Carlo samples. Thus, the mean and variance of sensitivity estimates obtained using the GP models are defined as follows, respectively:

$$
\begin{array} { r l } & { \displaystyle \mu _ { S _ { i } ^ { \mathrm { G P } } } = \frac { 1 } { N _ { Z } } \sum _ { p = 1 } ^ { N _ { Z } } \hat { S } _ { i , p } ^ { \mathrm { G P } } , } \\ & { \displaystyle \sigma _ { S _ { i } ^ { \mathrm { G P } } } ^ { 2 } = \frac { 1 } { N _ { Z } } \sum _ { p = 1 } ^ { N _ { Z } } ( \hat { S } _ { i , p } ^ { \mathrm { G P } } - \mu _ { S _ { i } ^ { \mathrm { G P } } } ) ^ { 2 } . } \end{array}\tag{27}
$$

A similar approach can be implemented in the DNN models with the use of MC dropout (with a chosen dropout rate). However, in contrast to the GP models, where we sample from a multivariate normal distribution to quantify the uncertainty in the Sobol’ index estimates, the sampling implementation is different in the DNN models. In the DNN models, we randomly set units of the network to zero and generate predictions using the remaining units of the network as shown in Algorithm 2. The neuron weights can be drawn from the approximate posterior $\hat { \mathbf { w } } \sim q _ { \theta } ( \mathbf { w } )$ to obtain the model outputs of a DNN with MC dropout denoted as $\mathcal { G } ^ { \hat { w } } ( \mathbf { X } ) = \hat { \mathbf { Y } }$ , where $\hat { \mathbf Y }$ is the predictive mean,

The predictive posterior given in Eq. 25 can be defined as follows:

$$
p ( \hat { \mathbf { Y } } | \hat { \mathbf { X } } , \mathbf { X } _ { T } , \mathbf { Y } _ { T } ) = \mathcal { N } ( \hat { \mathbf { Y } } ; \mathcal { G } ^ { \hat { w } } ( \mathbf { X } ) , \sigma ^ { 2 } \mathbf { I } )\tag{28}
$$

where $\sigma ^ { 2 } = ( 2 N \lambda ) / ( ( 1 - p _ { d } ) l ^ { 2 } )$ is the noise term [50], l being the prior length-scale, and λ is the regularization strength used in typical loss functions. Dropout can be interpreted as a variational Bayesian approximation and the minimization objective is defined by [52]

$$
\mathcal { L } ( \theta , p _ { d } ) = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log p ( \hat { \mathbf { Y _ { i } } } | \mathcal { G } ^ { \hat { w } } ( \mathbf { X _ { i } } ) ) + \frac { 1 - p _ { d } } { 2 N } | | \theta | | ^ { 2 }\tag{29}
$$

where θ is the set of distribution’s parameters to be optimized (i.e., weights of the network).

Information fusion and machine learning for sensitivity analysis using physics knowledge and experimental data

The predictive mean and predictive uncertainty are estimated by collecting the results of stochastic forward passes through the model. The mean prediction of the model with $N _ { d }$ samples can be approximated by

$$
\mathbb { E } ( { \mathbf { Y } } ) \approx \frac { 1 } { N _ { d } } \sum _ { t = 1 } ^ { N _ { d } } \mathcal { G } ^ { \hat { w } } ( { \mathbf { X } } ) ,\tag{30}
$$

and the variance of the prediction is estimated by

$$
\mathrm { V a r } ( \mathbf { Y } ) \approx \sigma ^ { 2 } + \frac { 1 } { N _ { d } } \sum _ { t = 1 } ^ { N _ { d } } \mathcal { G } ^ { \hat { w } } ( \mathbf { X } ) ^ { T } \mathcal { G } ^ { \hat { w } } ( \mathbf { X } ) - \mathbb { E } ( \mathbf { Y } ) ^ { T } \mathbb { E } ( \mathbf { Y } ) .\tag{31}
$$

Similar to Eq. (26), the uncertainty in the DNN model can be propagated to the sensitivity calculations using the following estimator:

$$
S _ { i } ^ { \mathrm { { D N N } } } = \frac { \frac { 1 } { m } \sum _ { k = 1 } ^ { m } \mathcal { G } ^ { \hat { w } } ( \mathbf { X } _ { k } ) \mathcal { G } ^ { \hat { w } } ( \mathbf { X } _ { k } ^ { \prime } ) - \frac { 1 } { m } \sum _ { k = 1 } ^ { m } \mathcal { G } ^ { \hat { w } } ( \mathbf { X } _ { k } ) \sum _ { k = 1 } ^ { m } \mathcal { G } ^ { \hat { w } } ( \mathbf { X } _ { k } ^ { \prime } ) } { \frac { 1 } { m } \sum _ { k = 1 } ^ { m } \mathcal { G } ^ { \hat { w } } ( \mathbf { X } _ { k } ) ^ { 2 } - \frac { 1 } { m } \sum _ { k = 1 } ^ { m } ( \mathcal { G } ^ { \hat { w } } ( \mathbf { X } _ { k } ) ) ^ { 2 } } ,\tag{32}
$$

where $\mathcal G ^ { \hat { w } } ( \mathbf X )$ denotes the deep neural network (DNN) with MC dropout and all the other terms have the same definition as Eq. (26). We note that the model uncertainty is propagated to the calculation of the sensitivity estimates by directly using the results based on stochastic forward passes through a dropout-reduced DNN instead of the predictive mean and predictive uncertainty.

```latex
Algorithm 2 Estimation of the distribution of $S _ { m } ^ { \mathrm { D N N } }$ using DNN models with MC dropout.
1: Generate samples ${ \bf X } _ { k }$ and $\mathbf { X } _ { k } ^ { \prime } \left( k = 1 , \ldots , m \right)$ of the random vectors X and $\mathbf { X } ^ { \prime }$
2: for $p = 1 , 2 , \ldots , N _ { d }$ do
Perform a stochastic forward pass through the network $\mathcal { G } ^ { \hat { w } } ( \mathbf { X } )$ using MC dropout and calculate the model
prediction $\hat { \mathbf Y } = \mathcal G ^ { \hat { w } } ( \mathbf X )$
4: Compute $\hat { S } _ { m , p } ^ { \mathrm { D N N } }$ using Eq. (32).
5: end for
return $( \hat { S } _ { m , p } ^ { \mathrm { D N N } } ) _ { p = 1 , 2 , . . . , N _ { d } } .$
The output of Algorithm $2 ~ ( \hat { S } _ { m , p } ^ { \mathrm { D N N } } ) _ { p = 1 , 2 , \ldots , N _ { d } }$ is a sample of size $N _ { d } .$ . Thus, the mean and variance of sensitivity
estimates obtained using DNN models are defined as follows, respectively:
N<sub>d</sub>
µ<sub>S</sub>DNN <sub>m</sub> <sup>1</sup><sub>N</sub> S<sup>ˆDNN</sup><sub>m,p</sub> , (33)
p=1
N<sub>d</sub>
σ N<sub>d</sub> 1 p=1 (S<sup>ˆDNN</sup><sub>m,p</sub> − µ<sub>S</sub>DNN <sub>m</sub> )<sup>2</sup>. Sm,p
In summary, two types of PIML strategies are proposed in this section and implemented in two types of ML models
(GP and DNN), in order to evaluate the accuracy and uncertainty of the sensitivity estimates in GSA. Eight different
PIML models are developed by leveraging the two PIML strategies. The accuracy of these models can be assessed by
comparison against validation data, whereas the variance of the sensitivity estimates can be quantified using Algorithms 1
and 2 and Eqs. 26 to 33. The different models have different training strategies and different numbers of parameters,
both of which will affect the accuracy and uncertainty of the sensitivity estimates. These differences are assessed in
detail in the next section.
4 Numerical illustration
4.1 Illustrative example 1
4.1.1 Problem setup
An additive manufacturing application is used to illustrate the proposed PIML models for GSA and compare their
performance. A fused filament fabrication (FFF) process is considered; commercial material Ultimaker Black Acry
lonitrile butadiene styrene (ABS) is extruded from an Ultimaker 2 Extended+ 3D printer to manufacture parts with
```

unidirectionally aligned filaments, and the porosity of the manufactured part is measured. FFF is a widely used additive manufacturing (AM) process due to its easy operation, low cost, and suitability for complex geometries. As the molten filament is deposited layer upon layer through a nozzle, it cools down, solidifies and bonds with the adjacent filaments. Rectangular specimens of length 35 mm, width 12 mm, and thickness 4.2 mm are manufactured for the ABS amorphous polymer, with constant filament height, width and length (0.7, 0.8, and 35 mm, respectively).

The output QoI is the porosity of the printed part, and the inputs are two process parameters, namely nozzle temperature and speed. The porosity of an FFF part is dependent on the temperature history at the interfaces between filaments. Thus, it is important to predict the temperature evolution of filaments for estimating the final mesostructure of the printed part. The analytical solution proposed by Costa et al. [53] for transient heat transfer during the printing process in FFF is used to predict the temperature evolution of filaments. A physics-based sintering model is developed, which considers realistic filament geometry, and allows the filament geometry to change during the printing process [25]. This model is used to predict the porosity of the FFF part using the temperature evolution of filaments, material properties, part geometry, and process parameters as inputs. Thus the mapping from input to output is a multi-physics model, i.e., models of two physical phenomena (heat transfer and sintering) are combined to predict the porosity given the values of two process parameters: nozzle temperature and nozzle speed.

The statistical properties of the QoI were observed to have negligible variability along the length of the specimens; therefore only the porosity measurements taken at the midpoint cross-section (see Fig. 3) are discussed here. These measurements were based on microscopy images processed through the ImageJ software [54]. Filaments were extruded through a nozzle with 0.8 mm diameter. The build plate temperature was constant and set to $1 1 0 ^ { \circ } \mathrm { C }$ . Using Latin hypercube sampling, 39 sets of process parameters $\dot { \mathbf { X } }$ were generated, and experiments were conducted at these 39 values. The ranges considered for the two process variables were: nozzle temperature T: $( 2 1 0 ^ { \circ } C \cdot 2 6 0 ^ { \circ } C )$ , and nozzle speed S: (15 mm/s - 46 mm/s).

![](images/22c9d8f2b656d69d72dd814f74fc948c084f7281e5ec1062400817f308553298.jpg)  
Figure 3: Cross-sectional geometry of an FFF specimen printed with nozzle temperature $2 4 0 ^ { \circ } \mathrm { C }$ and speed 42 mm/s.

## 4.1.2 Training details of the ML models

The basic ML models, namely Model 1 (for GP) and Model 5 (for DNN) are simply trained with the 39 sets of process inputs (temperature and speed) and output (porosity).

In the training of Model $2 \left( \mathrm { G P } ^ { \mathcal { L } _ { \mathrm { p h y } } } \right)$ and Model $6 ( \mathrm { D N N } ^ { \mathcal { L } _ { \mathrm { p h y } } } )$ , we impose two physics constraints $( \mathrm { i . e . }$ , two separate loss function terms, $\mathcal { L } _ { \mathrm { p h y } , k } ( \hat { \mathbf { Y } } )$ , where $k = \{ 1 , 2 \}$ and $\hat { \mathbf Y }$ is the porosity prediction). The corresponding loss function

Information fusion and machine learning for sensitivity analysis using physics knowledge and experimental data

terms are defined as

$$
\mathcal { L } _ { \mathrm { p h y } , 1 } ( \hat { \mathbf { Y } } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { R e L U } ( - \hat { Y _ { \mathrm { i } } } ) ,
$$

$$
\mathcal { L } _ { \mathrm { p h y } , 2 } ( \hat { \mathbf { Y } } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { R e L U } ( \hat { Y _ { \mathrm { i } } } - \phi _ { 0 , \mathrm { i } } ) ,\tag{34}
$$

considering physics violations related to the porosity in all the N samples. In the first loss function, a negative value of porosity is treated as a physics violation. The second loss function penalizes the model when the predicted final porosity $\hat { Y } _ { i }$ is greater than the initial porosity $\phi _ { 0 , i }$ of the ith part. This is based on the physics knowledge that the total void area decreases as the bond formation takes place. Thus, the porosity predictions are ensured to be physically meaningful with the inclusion of these physics-based penalty terms.

The overall “loss” function of the GP model is

$$
\mathcal { L } _ { \overline { { \mathrm { G P } } } } = \mathcal { L } _ { \mathrm { G P } } - \lambda _ { \mathrm { p h y } , 1 } ^ { \mathrm { G P } } \mathcal { L } _ { \mathrm { p h y } , 1 } ( \hat { \mathbf { Y } } ) - \lambda _ { \mathrm { p h y } , 2 } ^ { \mathrm { G P } } \mathcal { L } _ { \mathrm { p h y } , 2 } ( \hat { \mathbf { Y } } ) .\tag{35}
$$

Note that the GP model parameters are obtained by maximizing the above function.

The overall loss function of the DNN model is

$$
\mathcal { L } _ { \overline { { \mathrm { D N N } } } } = \mathcal { L } _ { \mathrm { D N N } } + \lambda _ { \mathrm { p h y } , 1 } ^ { \mathrm { D N N } } \mathcal { L } _ { \mathrm { p h y } , 1 } ( \hat { \mathbf { Y } } ) + \lambda _ { \mathrm { p h y } , 2 } ^ { \mathrm { D N N } } \mathcal { L } _ { \mathrm { p h y } , 2 } ( \hat { \mathbf { Y } } ) .\tag{36}
$$

Note that the DNN model parameters are obtained by minimizing the above function.

In Model $3 \ : ( \mathrm { G P ^ { u p d } } )$ and Model $7 ~ ( \mathrm { D N N ^ { u p d } } )$ , the ML models are pre-trained using the multi-physics model inputoutput. The pre-trained ML models are then updated using the experimental data. The training data for pre-training consists of 1310 input parameter combinations over a range of experimental values, i.e., $( 2 1 0 ^ { \circ } C \leq T \leq 2 6 0 ^ { \circ } C$ , 15 mm/s $\leq S \leq 4 6 \mathrm { { m i n } / \mathrm { { s } } ) }$ . Note that there are only 39 physical experiments available; this is one of the advantages of the pre-training/updating strategy, where the pre-training can be over a much larger set of input combinations, thus improving the generalization performance of the updated model. The input data are normalized prior to the training of the ML models (i.e., the output quantity porosity is dimensionless and between 0 and 1).

Model 4 $( \mathrm { G P ^ { u p d , } } { \mathcal { L } } _ { \mathrm { p h y } } )$ combines both PIML strategies for GP, and consists of two GP models: (i) the first GP model is trained using the physics model input-output samples consisting of 1310 input parameter combinations; and (ii) the second GP model is built for the discrepancy between the first GP model prediction and the actual system response using the experimental data, by maximizing the function shown in Eq. (35) to optimize the hyperparameters of the second GP model.

Model $8 \mathrm { D N N } \mathrm { u p d } , \mathcal { L } _ { \mathrm { p h y } }$ , which is a combination of the two PIML strategies, uses the DNN model parameters trained using the physics model input-output as the initial values. Then, during the updating phase with the experimental data, these parameters are updated by minimizing the loss function shown in Eq. (36).

The four GP models (Models 1 to 4) were implemented using Python. The optimization of the hyperparameters were performed using the scikit-optimize package. The multipliers of the physics constraint terms of models $2 \left( \mathrm { G P } ^ { \mathcal { L } _ { \mathrm { p h y } } } \right)$ and $\dot { 4 } ( \mathrm { G P ^ { u p d , \mathcal { L } _ { p h y } } ) }$ were chosen as $( \lambda _ { \mathrm { p h y } , 1 } ^ { \mathrm { G P } } , \lambda _ { \mathrm { p h y } , 2 } ^ { \mathrm { G P } } ) = ( 5 0 , \overset { \cdot } { 5 } 0 )$ based on a cross-validation test. The Automatic Relevance Determination (ARD) squared exponential function [31] was used as the covariance function for all the GP models.

The four DNN models (Models 5 to 8) were implemented using the Keras package [55] with Tensorflow in the backend. The hyperparameters of each model were tuned with grid search and the multipliers of the physics constraint terms in Model $\dot { 6 } ( \dot { \mathrm { D N N } } ^ { \mathcal { L } _ { \mathrm { p h y } } } )$ and Model 8 $\ B ( \mathrm { D N N } ^ { \mathrm { u p d } , \mathcal { L } _ { \mathrm { p h y } } } )$ were chosen as $( \lambda _ { \mathrm { p h y } , 1 } ^ { \mathrm { D N N } } , \lambda _ { \mathrm { p h y } , 2 } ^ { \mathrm { p h N } } ) = ( 0 . 0 \dot { 1 } , \dot { 0 } . 0 1 )$ based on a cross-validation test. Fully-connected DNN models with 2 hidden layers and 5 neurons in each hidden layer were constructed. The Rectified Linear Unit (ReLU) activation function and Adam optimizer were used to perform stochastic gradient descent for 300 epochs in learning the model parameters. The dropout rate for the DNN models was chosen to be 0.05, for reasons as explained below.

## 4.1.3 Comparison of computational effort

The computational costs of different models for training and estimation of Sobol’ indices based on 5000 MC samples with a fixed number of experimental data $( n = 3 9 )$ is given in Table. 1. Among the GP models, the time it takes for training as well as computation of Sobol’ indices using Models 2-4 is significantly greater than Model 1. The reason for the difference between the training time of GP and $\breve { \mathrm { G P ^ { u p d } } }$ is the pre-training phase, where a large amount of physics input-output samples used. Whereas, the difference between the training time of GP and $\mathrm { G P ^ { \mathcal { L } _ { \mathrm { p h y } } } }$ is due to the inclusion of physics constraints, which makes it harder for the optimization to find optimal hyperparameters. Interestingly, the training time for the DNN models ranges from 20 to 55 sec on the same desktop computer as used for the GP mode (Intel<sup>®</sup> Xeon<sup>®</sup> CPU E5-1660 v4@3.20GHz with 32 GB RAM and GPU NVIDIA Quadro K620 with 2 GB). Among the DNN models, the reasons for the increased training time of models 6-8 compared to Model 5 are the same as for the GP models. The Sobol’ index estimations based on 5000 samples take approximately 1-2 minutes for the DNN models (models 5-8), whereas the GP predictions take much longer because the covariance matrix needs to be stored and inverted.

Table 1: Computational effort of eight models for training and estimation of first-order and total effect Sobol’ indices using 5000 MC samples with $n = 3 9$ number of observations.
<table><tr><td>Models</td><td>Training [in minutes]</td><td>Sobol’ indices calculation [in minutes]</td></tr><tr><td>1. GP</td><td>1</td><td>3</td></tr><tr><td>2.  $\mathrm { G P ^ { \mathcal { L } _ { \mathrm { p h y } } } }$ </td><td>2</td><td>5</td></tr><tr><td>3. GPupd</td><td>3</td><td>7</td></tr><tr><td>4.  $\mathrm { G P ^ { u p d , } } \mathcal { L } _ { \mathrm { p h y } }$ </td><td>3</td><td>8</td></tr><tr><td>5. DNN</td><td>.3</td><td>1</td></tr><tr><td>6.  $\mathrm { D N N } ^ { \mathcal { L } _ { \mathrm { p h y } } }$ </td><td>.7</td><td>2</td></tr><tr><td> $7 . \mathrm { D N N } ^ { \mathrm { u p d } }$ </td><td>.5</td><td>2</td></tr><tr><td>8.  $\mathrm { D N N } ^ { \mathrm { u p d , } \mathcal { L } _ { \mathrm { p h y } } }$ </td><td>1</td><td>2</td></tr></table>

## 4.1.4 Comparison of accuracy

In order to compare the accuracy of the eight different models, the models are trained with different amounts of experimental observations $( n = ( 5 , 1 0 , 1 5 , 2 0 , 3 0 ) )$ ), and the remaining 9 observations are used to compute the errors; the root mean square error (RMSE) based on the 9 validation samples is reported as the accuracy measure for comparison. The mean and one standard deviation RMSE values for the four GP models and the four DNN models are shown in Tables 2 and 3 respectively, for different values of n. To further validate the accuracy of the models, the data set is divided into two subsets for cross-validation; k-fold cross-validation is performed by splitting the data set into sets for model training and cross-validation, and these sets are selected randomly $k = 1 0$ different times. The average cross-validation accuracy of the models over the 10 folds (random shuffles) is assessed by evaluating the average RMSE (see Table 4).

The results in Tables 2, 3 and 4 show that the use of PIML strategies in DNN models improves the performance, and the improvement is relatively larger as the amount of observed data n gets smaller. For $n { = } 2 0$ and $n { = } 3 0$ , the basic DNN model is as accurate as the DNN models with the PIML strategies; whereas for smaller values of n, the DNN models with the PIML strategies are significantly more accurate. This clearly indicates the benefit of PIML over basic ML for the DNN models. However, the GP models incorporating PIML strategies do not show significant improvement. This is because the GP models have a much smaller number of parameters compared to the DNN models, and the achievable accuracy of any ML model is constrained by the number of model parameters (in addition to model form).

In addition to the increased number of parameters, the DNN models have additional advantages in terms of training epochs and dropout rate that affect the prediction accuracy. As mentioned earlier, the optimum number of training epochs was found to be 300. Regarding dropout rate, the RMSE values (based on 9 validation samples) of DNN models with MC dropout for different dropout rates are shown in Fig. 4. The lowest RMSE values for all four models are obtained between 0.005-0.05. On the other hand, the Sobol’ index estimates were found to be similar within this range of dropout rate, for each of the four DNN models. Therefore, a dropout rate of 0.05 was chosen for the reporting of GSA results, since a higher dropout rate results in smaller sub-networks, therefore a smaller number of parameters and faster training. By comparing Tables 2 and 3, it is seen that at the dropout rate of 0.05, the DNN models are more accurate compared to the GP models.

Table 2: Effect of different amounts of training data on the RMSE of the GP models.
<table><tr><td>Model</td><td> $n = 5$ </td><td> $n = 1 0$ </td><td> $n = 1 5$ </td><td> $n = 2 0$ </td><td> $n = 3 0$ </td></tr><tr><td>1. GP</td><td> $0 . 0 2 0 ( \pm 0 . 0 0 4 )$ </td><td> $0 . 0 2 6 ( \pm 0 . 0 0 5 )$ </td><td> $0 . 0 2 1 ( \pm 0 . 0 0 1 )$ </td><td> $0 . 0 2 1 ( \pm 0 . 0 0 2 )$ </td><td> $0 . 0 2 0 ( \pm 0 . 0 0 1 )$ </td></tr><tr><td>2.  $\mathrm { G P } ^ { \mathcal { L } _ { \mathrm { p h y } } }$ </td><td> $0 . 0 1 9 ( \pm 0 . 0 0 4 )$ </td><td> $0 . 0 2 6 ( \pm 0 . 0 0 4 )$ </td><td> $0 . 0 2 4 ( \pm 0 . 0 0 4 )$ </td><td> $0 . 0 2 1 ( \pm 0 . 0 0 2 )$ </td><td> $0 . 0 1 9 ( \pm 0 . 0 0 1 )$ </td></tr><tr><td>3.  $\mathrm { G P ^ { u p d } }$ </td><td> $0 . 0 2 1 ( \pm 0 . 0 0 3 )$ </td><td> $0 . 0 2 8 ( \pm 0 . 0 0 3 )$ </td><td> $0 . 0 2 0 ( \pm 0 . 0 0 1 )$ </td><td> $0 . 0 2 0 ( \pm 0 . 0 0 1 )$ </td><td> $0 . 0 1 9 ( \pm 0 . 0 0 1 )$ </td></tr><tr><td>4.  $\mathrm { G P ^ { u p d , } } \mathcal { L } _ { \mathrm { p h y } }$ </td><td> $0 . 0 2 1 ( \pm 0 . 0 0 5 )$ </td><td> $0 . 0 2 3 ( \pm 0 . 0 0 3 )$ </td><td> $0 . 0 2 2 ( \pm 0 . 0 0 3 )$ </td><td> $0 . 0 2 0 ( \pm 0 . 0 0 2 )$ </td><td> $0 . 0 1 9 ( \pm 0 . 0 0 1 )$ </td></tr></table>

Table 3: Effect of different amounts of training data on the RMSE of the DNN models.
<table><tr><td>Model</td><td> $n = 5$ </td><td> $n = 1 0$ </td><td> $n = 1 5$ </td><td> $n = 2 0$ </td><td> $n = 3 0$ </td></tr><tr><td>5. DNN</td><td> $0 . 0 2 7 ( \pm 0 . 0 0 7 )$ </td><td> $0 . 0 1 7 ( \pm 0 . 0 0 9 )$ </td><td> $0 . 0 1 9 ( \pm 0 . 0 0 6 )$ </td><td> $0 . 0 1 3 ( \pm 0 . 0 0 3 )$ </td><td> $0 . 0 1 3 ( \pm 0 . 0 0 3 )$ </td></tr><tr><td> $6 . \mathrm { D N N } ^ { \mathcal { L } _ { \mathrm { p h y } } }$ </td><td> $0 . 0 1 7 ( \pm 0 . 0 0 7 )$ </td><td> $0 . 0 1 5 ( \pm 0 . 0 0 8 )$ </td><td> $0 . 0 1 4 ( \pm 0 . 0 0 5 )$ </td><td> $0 . 0 1 4 ( \pm 0 . 0 0 3 )$ </td><td> $0 . 0 1 3 ( \pm 0 . 0 0 2 )$ </td></tr><tr><td>7.  $\mathrm { D N N ^ { u p d } }$ </td><td> $0 . 0 1 4 ( \pm 0 . 0 0 2 )$ </td><td> $0 . 0 0 9 ( \pm 0 . 0 0 2 )$ </td><td> $0 . 0 1 4 ( \pm 0 . 0 0 1 )$ </td><td> $0 . 0 1 3 ( \pm 0 . 0 0 1 )$ </td><td> $0 . 0 1 3 ( \pm 0 . 0 0 1 )$ </td></tr><tr><td>8.  $\mathrm { D N N } ^ { \mathrm { u p d , } \mathcal { L } _ { \mathrm { p h y } } }$ </td><td> $0 . 0 1 4 ( \pm 0 . 0 0 2 )$ </td><td> $0 . 0 0 9 ( \pm 0 . 0 0 1 )$ </td><td> $0 . 0 1 4 ( \pm 0 . 0 0 1 )$ </td><td> $0 . 0 1 4 ( \pm 0 . 0 0 1 )$ </td><td> $0 . 0 1 3 ( \pm 0 . 0 0 1 )$ </td></tr></table>

Table 4: Tenfold cross-validation average RMSE results of GP and DNN models
<table><tr><td colspan="8">Models</td></tr><tr><td></td><td>1. GP</td><td> $2 . { \mathrm { G P } } ^ { \mathcal { L } _ { \mathrm { p h y } } }$ </td><td> $3 . \mathrm { G P ^ { u p d } }$ </td><td> $4 . \mathrm { G P ^ { u p d , } } \mathcal { L } _ { \mathrm { p h y } }$ </td><td>5. DNN</td><td> $6 . \mathrm { D N N } ^ { \mathcal { L } _ { \mathrm { p h y } } }$ </td><td> $7 . \mathrm { D N N ^ { u p d } }$ </td><td> $8 . \ \mathrm { D N N } ^ { \mathrm { u p d } , \mathcal { L } _ { \mathrm { p h y } } }$ </td></tr><tr><td> $n = 3 9$ </td><td>0.021(±0.002)</td><td>0.020(±0.002)</td><td>0.020(±0.001)</td><td>0.019(±0.002)</td><td>0.015(±0.004)</td><td>0.013(±0.004)</td><td>0.013(±0.003)</td><td>0.013(±0.003)</td></tr></table>

![](images/3f7b75b81863a19789e538ce68804454dc006d4d84c603b1e06d7b98abf63327.jpg)  
Figure 4: RMSE values of DNN models with varying dropout rates.

The Sobol’ indices estimates obtained using the two approaches of the second PIML strategy are compared in Figs. 5 and 6. The results show that both approaches converge to similar first-order and total effect sensitivity index estimates. Thus, in the rest of the paper the first approach of the second PIML strategy is used for both GP and DNN models, i.e., $\mathrm { G P ^ { u p d } }$ and DNN<sup>upd</sup>.

## 4.1.5 GSA results using GP models

The Sobol’ index computations with the GP models (1-4) are based on 5000 MC samples and 100 realizations of the Gaussian process. The effect of the number of experimental observations (used to train the GP models) on the first-order Sobol’ index estimates from the four GP models is illustrated in Fig. 7. And the total effect Sobol’ index estimates from the GP models for different numbers of experimental training data are shown in Fig. 8. The mean values of sensitivity estimates based on the GP model predictions are denoted with solid dots at a given number of observations n. The sensitivity results are reported for two input variables, nozzle temperature and nozzle speed; the output quantity of interest is porosity.

The 95% prediction intervals are represented with bars above and below the solid dots for the corresponding model. The bounds in the sensitivity estimates are calculated using the mean predictions based on the 100 realizations of the GP models. As expected, the prediction intervals decrease as increasing amounts of experimental data are used to train the GP models. Further, all four models converge to similar first-order and total effect sensitivity estimates for both printer nozzle temperature and speed. The relative individual contribution (at n=39) of nozzle speed to the variance of the porosity $( \approx 0 . 6 5 )$ is greater than that of the nozzle temperature (≈ 0.25) and their sum is ≈ 0.9. And the sum of their total effect indices (which capture parameter interactions) is slightly above 1.0, indicating that the interaction effect is small.

![](images/64473471e7afe63597e3a2fbeba9ae7cc7657bf7bd75f028c2f85ea1cd697505.jpg)  
(a)

![](images/8cadce59ba8b42e0c9da84d49085f3d90c6575f414edadf0de058a882572c7af.jpg)  
(b)

Figure 5: First-order sensitivity index estimators for (a) nozzle temperature, and (b) nozzle speed, using two different pre-training and updating approaches for GP.  
![](images/13fd5132eb64219ac790b8f6c60fb866f459c083a2db45a17fdf68397ee4622d.jpg)  
(a)

![](images/5c338ad5295a47404baddbfd719fd74f15326a07169b945d424241d401e712f4.jpg)  
(b)  
Figure 6: Total effect sensitivity index estimates for (a) nozzle temperature, and (b) nozzle speed, using two different pre-training and updating approaches for GP.

For further clarity regarding uncertainty, numerical values of the prediction bounds of first-order sensitivity estimates of nozzle temperature obtained using 100 realizations of the GP models are shown in Table 5. The results indicate that prediction intervals obtained using the PIML models 3 and 4 $( \mathrm { G P ^ { u p d } }$ and $\mathrm { G P ^ { u p d , } } \mathcal { L } _ { \mathrm { p h y } } )$ converge to the bounds obtained using 39 number of observations for training the models faster than the first two models. Note that the upper 95% bound for Model $2 \left( \mathrm { G P } ^ { \mathcal { L } _ { \mathrm { p h y } } } \right)$ converges to its final value (0.33) within 10 experimental observations.

Table 5: First-order sensitivity estimate prediction bounds of nozzle temperature for different amounts of experimental training data, using the GP models.
<table><tr><td rowspan="2">Models</td><td colspan="5">Lower 95% confidence limit</td><td colspan="5">Upper 95% confidence limit</td></tr><tr><td>n=5</td><td>n=10</td><td>n=20</td><td>n=30</td><td>n=39</td><td>n=5</td><td>n=10</td><td>n=20</td><td>n=30</td><td>n=39</td></tr><tr><td>1. GP</td><td>0.00</td><td>0.08</td><td>0.13</td><td>0.20</td><td>0.19</td><td>0.37</td><td>0.25</td><td>0.34</td><td>0.35</td><td>0.33</td></tr><tr><td> $2 . \mathrm { G P } ^ { \mathcal { L } _ { \mathrm { p h y } } }$ </td><td>0.00</td><td>0.00</td><td>0.05</td><td>0.16</td><td>0.18</td><td>0.55</td><td>0.33</td><td>0.43</td><td>0.34</td><td>0.33</td></tr><tr><td> $3 . \mathrm { G P ^ { u p d } }$ </td><td>0.03</td><td>0.14</td><td>0.10</td><td>0.16</td><td>0.17</td><td>0.77</td><td>0.33</td><td>0.34</td><td>0.31</td><td>0.29</td></tr><tr><td> $4 . \mathrm { G P ^ { u p d , } } { \mathcal { L } } _ { \mathrm { p h y } }$ </td><td>0.06</td><td>0.11</td><td>0.14</td><td>0.18</td><td>0.20</td><td>0.71</td><td>0.38</td><td>0.34</td><td>0.31</td><td>0.32</td></tr></table>

![](images/b85379466a2d05f3bcbbab9a780f5814a95bf182a8bf17403cb6e77b55b8e9e4.jpg)  
(a)

![](images/e1d079e97abd41bb460ba8c59df4dbe51c493aeae8cd05021b9a24599fec20b2.jpg)  
(b)

Figure 7: First-order sensitivity index estimates for (a) nozzle temperature, and (b) nozzle speed, using the GP models.  
![](images/5906cf7a5fd6e128d1b7cc2bba0f67645be23d8e0d4f862189e582c7ae15fa3d.jpg)  
(a)

![](images/2f684c893b28df667a908d2022cc741c05d4cdf1f4915844b3557fd985d6c196.jpg)  
(b)  
Figure 8: Total effect sensitivity index estimates for (a) nozzle temperature, and (b) nozzle speed, using the GP models.

## 4.1.6 GSA results using DNN models with MC dropout

The Sobol’ index computations with the DNN models (5-8) are based on 5000 MC samples and 100 stochastic forward passes through the networks. The calculated first-order sensitivity estimates of temperature and speed, $S _ { T }$ and $S _ { S }$ respectively, for different numbers of experimental observations (training data) $n = \bar { ( 5 , 1 0 , 2 0 , 3 0 , \bar { 3 } 9 ) }$ are shown in Fig. 9. The distributions of sensitivity estimates are obtained using MC dropout predictions based on 100 stochastic forward passes through the networks for different number of observations. The mean values of sensitivity estimates are represented with solid dots and the 95% bounds are denoted with bars above and below the solid dots for the corresponding model. Similarly, the calculated total effect sensitivity estimates $S _ { T }$ and $S _ { S }$ for different values of n are illustrated in Fig. 10. Similar to the GP results, the difference between the total effect and first-order indices of process inputs is negligible, which indicates that their interaction is not significant.

All DNN models converge to similar first-order and total effect sensitivity estimates for both inputs, and these values are consistent with the results obtained using GP models. For this problem with two inputs and one output, 39 experimental observations appear adequate to train even the basic DNN model to achieve similar performance as the other physics-informed models; in fact, the results are similar even at 20 observations. However, the superior accuracy of the physics-informed models becomes apparent if only a small number of experiments are available, say $n = 5$ or $n = 1 0$ , as shown in Table 3 and discussed earlier in Section 4.1.4.

For further clarity regarding uncertainty, numerical values of the prediction bounds of first-order sensitivity estimates of nozzle temperature obtained using 100 forward passes through the DNN models are given in Table 6. The results show

Information fusion and machine learning for sensitivity analysis using physics knowledge and experimental data

![](images/32ca317f41cd4623f38c9fccf78d8bab86ccd57c18574312a0e42d3eb025d1fe.jpg)  
(a)

![](images/2faa110cad8eb354764a91123ab4b9f9b66ea4f6de36d00996d36672c56f7f47.jpg)  
(b)

Figure 9: First-order sensitivity index estimators for (a) nozzle temperature, and (b) nozzle speed, using DNN models with MC dropout.  
![](images/b2369f3da6cc373a0729823174abacdc22c5a6a098c98ffd75e4431edbd2c5e6.jpg)  
(a)

![](images/3a8e6bd1e9c521e831a311cdc47184a3a4c9f59d4401ebe21463ffc1a4fa7306.jpg)  
(b)  
Figure 10: Total effect sensitivity index estimators for (a) nozzle temperature, and (b) nozzle speed, using DNN models with MC dropout.

that prediction intervals obtained using all models show a similar trend. The 95% bounds for all models are significantly narrower than the ones obtained using the GP models. The uncertainty in the sensitivity estimates due to the DNN models is almost negligible when more than 10 number of observations are used to train the models.

Table 6: First-order sensitivity estimate prediction bounds of nozzle temperature for different amounts of experimental training data, using the DNN models.
<table><tr><td rowspan="2">Models</td><td colspan="5">Lower 95% confidence limit</td><td colspan="5">Upper 95% confidence limit</td></tr><tr><td> $\mathrm { n } { = } 5$ </td><td> $\mathrm { n } { = } 1 0$ </td><td> $\scriptstyle \mathrm { n = 2 0 }$ </td><td> $\scriptstyle \mathrm { n = } 3 0$ </td><td> $\mathrm { n } { = } 3 9$ </td><td> $\mathrm { n } { = } 5$ </td><td> $\mathrm { n } { = } 1 0$ </td><td> $\scriptstyle \mathrm { n = 2 0 }$ </td><td> $\scriptstyle \mathrm { n = } 3 0$ </td><td> $\mathrm { n } { = } 3 9$ </td></tr><tr><td>5. DNN</td><td>0.45</td><td>0.22</td><td>0.17</td><td>0.23</td><td>0.30</td><td>0.50</td><td>0.23</td><td>0.18</td><td>0.25</td><td>0.32</td></tr><tr><td>6. DNNLphy</td><td>0.52</td><td>0.18</td><td>0.15</td><td>0.17</td><td>0.22</td><td>0.59</td><td>0.19</td><td>0.17</td><td>0.18</td><td>0.24</td></tr><tr><td>7. DNNupd</td><td>0.78</td><td>0.30</td><td>0.12</td><td>0.22</td><td>0.25</td><td>0.81</td><td>0.32</td><td>0.15</td><td>0.24</td><td>0.27</td></tr><tr><td> $8 . \ \mathrm { D N N } ^ { \mathrm { u p d } , \mathcal { L } _ { \mathrm { p h y } } }$ </td><td>0.78</td><td>0.29</td><td>0.10</td><td>0.18</td><td>0.22</td><td>0.82</td><td>0.33</td><td>0.12</td><td>0.21</td><td>0.25</td></tr></table>

The 95% prediction intervals (upper limit-lower limit) decrease as increasing amounts of experimental data are used to train the DNN models. However, the prediction intervals obtained using the DNN models are much smaller than the ones obtained using the GP models since the DNN models has more degrees of freedom that can be optimized. For example, for $n = 3 9$ , the prediction interval width is 0.02-0.03 for the DNN models, whereas it is 0.12-0.15 for the

Information fusion and machine learning for sensitivity analysis using physics knowledge and experimental data

GP models. In addition to the larger number of parameters, the number of training epochs, which is the number of complete passes through a batch of training dataset, is optimized for the DNN models, thus maximizing their prediction accuracy. The prediction accuracy is further improved by choosing the appropriate dropout rate as discussed earlier. The number of parameters to be learned reduces with the use of dropout, which helps with regularization and prevents ill-conditioning. Further, the number of training epochs is also observed to affect both the accuracy and uncertainty of the sensitivity estimates. It was found that at a smaller number of epochs, the model was not fully trained resulting in underfitting, leading to larger bias and variance in the prediction. As the number of epochs was increased, both the bias and the variance were reduced.

## 4.2 Illustrative example 2

## 4.2.1 Problem setup

In order to further illustrate the effectiveness of the proposed method and how different combinations of methods affect the accuracy and computational effort, a more complex problem involving eleven random variables is considered. The data set used by Karpatne et al. [22] for lake temperature modeling is considered, at Lake Mendota in Wisconsin, USA.

The overall data for Lake Mendota comprised of 13,543 temperature observations from 30 April 1980 to 02 Nov 2015. The modeling uses used 11 meteorological drivers as input variables. The original physics-based model, referred to as the General Lake Model (GLM), modeled the lake temperature by performing 1-D analysis (along depth) considering a variety of lake variables, and produced a total of 662,781 training samples (input-output). The physical non-linear relationship between temperature, depth and density of water is used to evaluate the physical violations across every consecutive depth-pair and time step. A more detailed description of the physics-based loss function and data can be found in the work by Karpatne et al. [22].

## 4.2.2 Comparison of computational effort

The computational costs of different models for training and estimation of Sobol’ indices based on 5000 MC samples with a fixed number of experimental data (n = 1000) is given in Table. 7. The desktop computer used for the illustrative example 1 (Intel<sup>®</sup> Xeon<sup>®</sup> CPU E5-1660 v4@3.20GHz with 32 GB RAM and GPU NVIDIA Quadro K620 with 2 GB) is used for the computations of Sobol’ indices. Among the GP models, the time it takes for training as well as computation of Sobol’ indices using Models 2-4 is similar to Model 1. The Sobol’ index estimations based on 5000 samples take approximately 1-3 minutes for the DNN models (Models 5-8). The training time and the computation of Sobol’ indices using DNN models are significantly less than Models 1-4, which use the GP models, since the GP predictions require the covariance matrix to be inverted.

Table 7: Computational effort of eight models for training and estimation of first-order and total effect Sobol’ indices using 5000 MC samples with n = 1000 number of temperature observations.
<table><tr><td>Models</td><td>Training [in minutes]</td><td>Sobol’ indices calculation [in minutes]</td></tr><tr><td>1. GP</td><td>37.5</td><td>40.1</td></tr><tr><td>2. GPLphy</td><td>37.9</td><td>41.2</td></tr><tr><td>3. GPupd</td><td>36.8</td><td>39.7</td></tr><tr><td>4. GPupd,Lphy</td><td>36.9</td><td>39.8</td></tr><tr><td>5. DNN</td><td>1.3</td><td>2.8</td></tr><tr><td>6. DNNLphy</td><td>2.7</td><td>5.7</td></tr><tr><td>7. DNNupd</td><td>1.4</td><td>2.8</td></tr><tr><td>8. DNNupd,Lphy</td><td>2.8</td><td>6.1</td></tr></table>

## 4.2.3 GSA results using GP models

The effect of the number of experimental observations on the first-order and total effect Sobol’ index estimates from the four GP models of the lake temperature modeling is illustrated in Figs. 11 and 12. The 95% prediction intervals are represented with bars above and below the solid dots for the corresponding model. The prediction intervals decrease as increasing amounts of experimental data are used to train the GP models. In addition, all four models converge to similar first-order and total effect sensitivity estimates for all input variables.

![](images/3c61644e8ee0512f514cea76eb509b755db95609f4229a66e658f1030bb0db63.jpg)  
Figure 11: First-order sensitivity index estimators for the eleven variables of the lake temperature modeling using the GP models.

## 4.2.4 GSA results using DNN models with MC dropout

The Sobol’ index computations with the DNN models (5-8) are based on 5000 MC samples and 100 stochastic forward passes through the networks. The calculated first-order sensitivity estimates of eleven input variables, for different numbers of experimental observations (training data) $n = ( 1 0 0 , 2 5 0 , 5 0 0 , 7 5 0 , 1 0 0 0 )$ ) are shown in Figs. 13 and 14.

All DNN models converge to similar first-order and total effect sensitivity estimates for all eleven inputs, and these values are consistent with the results obtained using GP models. The total effect sensitivity estimates obtained using the basic GP and DNN models for most of the input variables converge to a slightly different value than the ones obtained using the proposed PIML models. Note that the mean estimates of the PIML models converge more smoothly than the basic ML models.

The results show that prediction intervals obtained using all models show a similar trend. The 95% bounds for all DNN models are significantly narrower than the ones obtained using the GP models. The results indicate that the mean estimates and prediction intervals obtained using the PIML models 4 and 8 (GP<sup>upd,Lphy</sup> and DNN<sup>upd,Lphy</sup> ) converge to the mean estimates and bounds obtained using 1000 number of observations for training the models faster than the other models. Note that the mean estimates for Models 4 and $8 ( \mathrm { G P ^ { u p d , \mathcal { L } _ { p h y } } }$ and $\mathrm { D N N } ^ { \mathrm { u p d , } \mathcal { L } _ { \mathrm { p h y } } } )$ converge to their final values within 250 experimental observations for almost of the variables. The prediction bounds of DNN models do not change significantly as more experimental observations are used to train the models. However, the prediction bounds get smaller as more number of training data is used to train the GP models. Moreover, the prediction bounds of physics-informed models are slightly narrower than the basic GP model. For further clarity regarding uncertainty, numerical values of the prediction bounds of first-order sensitivity estimates of $X _ { 1 }$ (day of year) obtained using 100 realizations of the GP models and 100 forward passes through the DNN models are shown in Table 8.

![](images/f37973145b6552973ed81b03e1d0d84d04ab50bcc83653815c167e3c783ca549.jpg)  
Figure 12: Total effect sensitivity index estimates for the eleven variables of the lake temperature modeling using the GP models.

The results of these numerical examples could be summarized as follows:

• The GP models required more computational effort than the DNN models, both in training and prediction.

• The DNN models were able to achieve higher accuracy and lower uncertainty in prediction, due to the optimization of dropout rate and number of training epochs.

• The physics-informed ML models are able to achieve higher prediction accuracy than the basic ML models, especially when the amount of available experimental data is small.

![](images/bae7e1eb6044d6b56ff08994245e2291bbca57b99641573d66616f5140030062.jpg)

![](images/df70987f34f40ca9bb66c5c4105b12979d897e27f39bdc6f33f702294ad29df3.jpg)

![](images/ab71159a8c9a4dad8bade4748a2ba1624169958c122e18da509b79fec3a57556.jpg)

![](images/36b774ecd18ce91ef56f0483c7826045bfd342f0f386b01bb1d61a52257b5632.jpg)

![](images/ae2c04a99b695cb1c33713a3dbfb6f959e8a8f7a39799a4709b423a09748c4f7.jpg)

![](images/487177e4e3eb24daf78647808d7966d93a18e7877bcc498ec84b70196fe03491.jpg)

![](images/519d3db5c18246e9a91c779c881c4bc2bd6d7f7e39e9623033d4695764e2086b.jpg)

![](images/fb97032ef5cf62515255de0f05dfff0e5f4260183b80cc9eb5baa22063f4cf04.jpg)

![](images/e2a89c50a9db576224bf3b4b3912e58dcbc0248dd343d76937e32fc688dfac1c.jpg)

![](images/3433705bc90a817024a8538b71341325287676ae89686fa9ee18cdd9e433a7f9.jpg)

![](images/d812b69aef5d8ed550d65747680c74bdcf14f9d9ad92d656cf9ffa0820ceb61e.jpg)  
Figure 13: First-order sensitivity index estimators for the eleven variables of the lake temperature modeling using the DNN models.

Table 8: First-order sensitivity estimate prediction bounds of $X _ { 1 }$ (day of year) for different amounts of experimental training data, using the GP and DNN models.
<table><tr><td rowspan="2">Models</td><td colspan="5">Lower 95% confidence limit</td><td colspan="5">Upper 95% confidence limit</td></tr><tr><td>n=100</td><td>n=250</td><td>n=500</td><td>n=750</td><td>n=1000</td><td>n=100</td><td>n=250</td><td>n=500</td><td>n=750</td><td>n=1000</td></tr><tr><td>1. GP</td><td>0.00</td><td>0.27</td><td>0.67</td><td>0.61</td><td>0.24</td><td>0.38</td><td>0.71</td><td>0.82</td><td>0.77</td><td>0.53</td></tr><tr><td> $2 . \mathrm { G P } ^ { \mathcal { L } _ { \mathrm { p h y } } }$ </td><td>0.01</td><td>0.51</td><td>0.60</td><td>0.63</td><td>0.32</td><td>0.36</td><td>0.76</td><td>0.77</td><td>0.77</td><td>0.55</td></tr><tr><td> $3 . \mathrm { G P ^ { u p d } }$ </td><td>0.54</td><td>0.48</td><td>0.70</td><td>0.65</td><td>0.49</td><td>0.82</td><td>0.78</td><td>0.82</td><td>0.80</td><td>0.68</td></tr><tr><td>4.  $\mathrm { G P ^ { u p d , } } \mathcal { L } _ { \mathrm { p h y } }$ </td><td>0.46</td><td>0.40</td><td>0.59</td><td>0.55</td><td>0.42</td><td>0.69</td><td>0.66</td><td>0.69</td><td>0.68</td><td>0.57</td></tr><tr><td>5. DNN</td><td>0.06</td><td>0.14</td><td>0.08</td><td>0.45</td><td>0.16</td><td>0.08</td><td>0.16</td><td>0.12</td><td>0.49</td><td>0.19</td></tr><tr><td>6.  $\mathrm { D N N } ^ { \mathcal { L } _ { \mathrm { p h y } } }$ </td><td>0.06</td><td>0.12</td><td>0.18</td><td>0.30</td><td>0.30</td><td>0.09</td><td>0.15</td><td>0.21</td><td>0.33</td><td>0.33</td></tr><tr><td>7.  $\mathrm { D N N ^ { u p d } }$ </td><td>0.03</td><td>0.29</td><td>0.15</td><td>0.41</td><td>0.37</td><td>0.06</td><td>0.33</td><td>0.20</td><td>0.45</td><td>0.41</td></tr><tr><td>8.  $\mathrm { D N N } ^ { \mathrm { u p d , } \mathcal { L } _ { \mathrm { p h y } } }$ </td><td>0.00</td><td>0.30</td><td>0.19</td><td>0.29</td><td>0.32</td><td>0.02</td><td>0.33</td><td>0.23</td><td>0.34</td><td>0.37</td></tr></table>

Overall, for these numerical examples, the DNN models gave higher accuracy and lower uncertainty in the GSA results than the GP models, and also required less computational effort both in training and prediction. However, the first numerical example consisted of only two inputs and a single output. As the number of inputs and outputs increase with the second numerical example, all the ML models considered above were challenged w.r.t. adequacy of training data, computational effort in training and prediction, and the accuracy and uncertainty of the sensitivity estimates. The results showed that the mean estimates of the PIML Models 4 and 8 $( \mathrm { G P ^ { u p d , } } \mathcal { L } _ { \mathrm { p h y } }$ and $\mathrm { D N N } ^ { \mathrm { u p d , } \mathcal { L } _ { \mathrm { p h y } } } )$ for both numerical examples converge to the mean estimates obtained using the largest number of observations (i.e., 39 and 1000 observations for the first and second numerical example, respectively) for training the models faster than the other models.

![](images/66f14226e421fcf42de6fe4f6ecf20bb01d131876033acc6713ffe30fe15ec06.jpg)  
Figure 14: Total effect sensitivity index estimates for the eleven variables of the lake temperature modeling using the DNN models.

## 5 Conclusion

This paper developed methodologies for information fusion and machine learning for sensitivity analysis using both physics knowledge and experimental data, while accounting for model uncertainty. Variance-based sensitivity analysis is used to quantify the relative contribution of each uncertainty source to the variability of the output quantity. Two types of ML models were considered, namely, GP and DNN models. Several PIML models were developed by leveraging two strategies for incorporating physics knowledge into ML models: (1) incorporating physics constraints within the loss functions used in training the ML models, and (2) pre-training an ML model with simulation data and then updating it with experimental data. The first strategy does not use the physics model, whereas the second strategy does.

The calculation of the Sobol’ indices with the GP model simply uses the proposed estimator (Eq. 26). On the other hand, with respect to the DNN model, we use the Monte Carlo dropout strategy to compute prediction bounds on the Sobol’ indices; previous work in this regard has only considered prediction bounds of the model output. Prediction bounds are computed for the sensitivity index estimates to account for the model uncertainty in the trained models, and the accuracy and computational effort of the various PIML models are compared.

The results show that the application of PIML strategies to both GP and DNN enables accurate Sobol’ index computations even with smaller amounts of experimental data while producing physically meaningful results. Thus, the proposed approach helps to fill the physics knowledge gap in the ML models while estimating the Sobol’ indices, by correcting for the approximation in the physics-based models. The numerical examples show that training the GP models and estimating the Sobol’ indices require more computational effort than the DNN models. The uncertainty regarding the sensitivity estimates obtained using the DNN models is smaller than the results obtained using the GP models. In the numerical examples, the DNN models are found to be more accurate compared to the GP models. The higher accuracy, lower uncertainty, and lower computational effort of the DNN models is attributed to their flexibility in terms of number of parameters, training epochs and dropout rate.

In future work, the proposed PIML approaches need to be tested for problems with a larger number of dimensions both in the input and output, with multiple combinations to further analyze the convergence of Sobol’ index estimates. Future work can also explore the weighting of the two sources of information, since the data produced by physics-based models and experiments have different levels of credibility. The proposed approach for GP models can also be extended by using different kernels with varying smoothness. We note that only one kernel, namely the Gaussian kernel, is used in this paper. In order to have a more complete comparison between the GP and DNN models, different kernels with varying smoothness can be considered for future work.

## References

[1] Andrea Saltelli, Marco Ratto, Terry Andres, Francesca Campolongo, Jessica Cariboni, Debora Gatelli, Michaela Saisana, and Stefano Tarantola. Global sensitivity analysis: the primer. John Wiley & Sons, 2008.

[2] A. Saltelli, S. Tarantola, and K. P.-S. Chan. A quantitative model-independent method for global sensitivity analysis of model output. Technometrics, 41(1):39–56, 1999.

[3] Thierry A. Mara and Stefano Tarantola. Variance-based sensitivity indices for models with dependent inputs. Reliability Engineering & System Safety, 107:115 – 121, 2012. SAMO 2010.

[4] Emanuele Borgonovo and Elmar Plischke. Sensitivity analysis: A review of recent advances. European Journal ofOperational Research, 248(3):869 – 887, 2016.

[5] S. Sankararaman and S. Mahadevan. Separating the contributions of variability and parameter uncertainty in probability distributions. Reliability Engineering & System Safety, 112:187 – 199, 2013.

[6] Chenzhao Li and Sankaran Mahadevan. Relative contributions of aleatory and epistemic uncertainty sources in time series prediction. International Journal ofFatigue, 82:474 – 486, 2016.

[7] Loic Le Gratiet, Claire Cannamela, and Bertrand Iooss. A bayesian approach for global sensitivity analysis of (multifidelity) computer codes. SIAM/ASA Journal on Uncertainty Quantification, 2(1):336–363, 2014.

[8] Jeremy E. Oakley and Anthony O’Hagan. Probabilistic sensitivity analysis of complex models: a bayesian approach. Journal ofthe Royal Statistical Society: Series B (Statistical Methodology), 66(3):751–769, 2004.

[9] Amandine Marrel, Bertrand Iooss, Béatrice Laurent, and Olivier Roustant. Calculations of sobol indices for the gaussian process metamodel. Reliability Engineering & System Safety, 94(3):742 – 751, 2009.

[10] I.M Sobol. Global sensitivity indices for nonlinear mathematical models and their monte carlo estimates. Mathematics and Computers in Simulation, 55(1):271 – 280, 2001. The Second IMACS Seminar on Monte Carlo Methods.

[11] Bruno Sudret. Global sensitivity analysis using polynomial chaos expansions. Reliability Engineering & System Safety, 93(7):964 – 979, 2008. Bayesian Networks in Dependability.

[12] Wei Chen, Ruichen Jin, and Agus Sudjianto. Analytical Variance-Based Global Sensitivity Analysis in Simulation Based Design Under Uncertainty. Journal ofMechanical Design, 127(5):875–886, 12 2004.

[13] S. Tarantola, D. Gatelli, and T.A. Mara. Random balance designs for the estimation of first order global sensitivity indices. Reliability Engineering & System Safety, 91(6):717 – 727, 2006.

[14] F. E. Satterthwaite. Random balance experimentation. Technometrics, 1(2):111–137, 1959.

[15] Vincent Ginot, Sabrina Gaba, Rémy Beaudouin, Franck Aries, and Hervé Monod. Combined use of local and anova-based global sensitivity analyses for the investigation of a stochastic dynamic model: Application to the case study of an individual-based model of a fish population. Ecological Modelling, 193(3):479 – 491, 2006.

[16] G. E. B. Archer, A. Saltelli, and I. M. Sobol. Sensitivity measures, anova-like techniques and the use of bootstrap. Journal ofStatistical Computation and Simulation, 58(2):99–120, 1997.

[17] Zhen Hu and Sankaran Mahadevan. Probability models for data-driven global sensitivity analysis. Reliability Engineering & System Safety, 187:40 – 57, 2019. Sensitivity Analysis of Model Output.

[18] Amandine Marrel, Bertrand Iooss, François Van Dorpe, and Elena Volkova. An efficient methodology for modeling complex computer codes with gaussian processes. Computational Statistics & Data Analysis, 52(10):4731 – 4744, 2008.

[19] A. O’Hagan. Bayesian analysis of computer code outputs: A tutorial. Reliability Engineering & System Safety, 91(10):1290 – 1300, 2006. The Fourth International Conference on Sensitivity Analysis of Model Output (SAMO 2004).

[20] Chenzhao Li and Sankaran Mahadevan. An efficient modularized sample-based method to estimate the first-order sobol index. Reliability Engineering & System Safety, 153:110 – 121, 2016.

[21] Erin C DeCarlo, Sankaran Mahadevan, and Benjamin P Smarslok. Efficient global sensitivity analysis with correlated variables. Structural and Multidisciplinary Optimization, 58(6):2325–2340, 2018.

[22] Anuj Karpatne, William Watkins, Jordan Read, and Vipin Kumar. Physics-guided Neural Networks (PGNN): An Application in Lake Temperature Modeling, 2017.

[23] Xiaowei Jia, Jared Willard, Anuj Karpatne, Jordan S Read, Jacob A Zwart, Michael Steinbach, and Vipin Kumar. Physics-Guided Machine Learning for Scientific Discovery: An Application in Simulating Lake Temperature Profiles, 2020.

[24] Jared Willard, Xiaowei Jia, Shaoming Xu, Michael Steinbach, and Vipin Kumar. Integrating physics-based modeling with machine learning: A survey, 2020.

[25] Berkcan Kapusuzoglu and Sankaran Mahadevan. Physics-informed and hybrid machine learning in additive manufacturing: Application to fused filament fabrication. JOM, pages 1–11, 2020.

[26] Loïc Le Gratiet, Stefano Marelli, and Bruno Sudret. Metamodel-based sensitivity analysis: Polynomial chaos expansions and gaussian processes. Handbook ofUncertainty Quantification, page 1–37, 2015.

[27] Amandine Marrel, Bertrand Iooss, Sébastien Da Veiga, and Mathieu Ribatet. Global sensitivity analysis of stochastic computer models with joint metamodels. Statistics and Computing, 22(3):833–847, 2012.

[28] Dongbin Xiu and George Em Karniadakis. The wiener–askey polynomial chaos for stochastic differential equations. SIAM journal on scientific computing, 24(2):619–644, 2002.

[29] Alexandre Janon, Maelle Nodet, and Clementine Prieur. Uncertainties assessment in global sensitivity indices estimation from metamodels. International Journalfor Uncertainty Quantification, 4(1):21–36, 2014.

[30] Zhen Hu and Xiaoping Du. Mixed Efficient Global Optimization for Time-Dependent Reliability Analysis. Journal ofMechanical Design, 137(5), 05 2015. 051401.

[31] Carl Edward Rasmussen and Christopher KI Williams. Gaussian Processes for Machine Learning, volume 2. MIT press Cambridge, MA, 2006.

[32] N. Muralidhar, M. R. Islam, M. Marwah, A. Karpatne, and N. Ramakrishnan. Incorporating prior domain knowledge into deep neural networks. In 2018 IEEE International Conference on Big Data (Big Data), pages 36–45, 2018.

[33] Areski Cousin, Hassan Maatouk, and Didier Rullière. Kriging of financial term-structures. European Journal of Operational Research, 255(2):631 – 648, 2016.

[34] Sébastien Da Veiga and Amandine Marrel. Gaussian process modeling with inequality constraints. Annales de la Faculté des sciences de Toulouse : Mathématiques, Ser. 6, 21(3):529–555, 2012.

[35] Shirin Golchi, Derek R Bingham, Hugh Chipman, and David A Campbell. Monotone emulation of computer experiments. SIAM/ASA Journal on Uncertainty Quantification, 3(1):370–392, 2015.

[36] Andrés F López-Lopera, François Bachoc, Nicolas Durrande, and Olivier Roustant. Finite-dimensional gaussian approximation with linear inequality constraints. SIAM/ASA Journal on Uncertainty Quantification, 6(3):1224– 1255, 2018.

[37] Jaakko Riihimäki and Aki Vehtari. Gaussian processes with monotonicity information. volume 9 of Proceedings ofMachine Learning Research, pages 645–652, Chia Laguna Resort, Sardinia, Italy, 13–15 May 2010. JMLR Workshop and Conference Proceedings.

[38] Benjamin Peherstorfer, Karen Willcox, and Max Gunzburger. Survey of multifidelity methods in uncertainty propagation, inference, and optimization. Siam Review, 60(3):550–591, 2018.

[39] Ghina N Absi and Sankaran Mahadevan. Multi-fidelity approach to dynamics model calibration. Mechanical Systems and Signal Processing, 68:189–206, 2016.

[40] Marc C. Kennedy and Anthony O’Hagan. Bayesian calibration of computer models. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 63(3):425–464, 2001.

[41] You Ling, Joshua Mullins, and Sankaran Mahadevan. Selection of model discrepancy priors in Bayesian calibration. Journal ofComputational Physics, 276:665 – 680, 2014.

[42] Berkcan Kapusuzoglu, Matthew Sato, Sankaran Mahadevan, and Paul Witherell. Process Optimization under Uncertainty for Improving the Bond Quality of Polymer Filaments in Fused Filament Fabrication. Journal of Manufacturing Science and Engineering, pages 1–46, 08 2020.

[43] John S. Denker and Yann LeCun. Transforming neural-net output levels to probability distributions. In Proceedings ofthe 3rd International Conference on Neural Information Processing Systems, NIPS’90, page 853–859, San Francisco, CA, USA, 1990. Morgan Kaufmann Publishers Inc.

[44] David JC MacKay. A practical bayesian framework for backpropagation networks. Neural Computation, 4(3):448–472, 1992.

[45] R. M. Neal. Bayesian learning for neural networks. Lecture Notes in Statistics, 1996.

[46] Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in neural networks, 2015.

[47] Alex Graves. Practical variational inference for neural networks. In J. Shawe-Taylor, R. S. Zemel, P. L. Bartlett, F. Pereira, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems 24, pages 2348– 2356. Curran Associates, Inc., 2011.

[48] Jose Hernandez-Lobato, Yingzhen Li, Mark Rowland, Thang Bui, Daniel Hernandez-Lobato, and Richard Turner. Black-box alpha divergence minimization. volume 48 of Proceedings of Machine Learning Research, pages 1511–1520. PMLR, 20–22 Jun 2016.

[49] Yarin Gal and Zoubin Ghahramani. Bayesian convolutional neural networks with bernoulli approximate variational inference, 2015.

[50] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48, ICML’16, page 1050–1059. JMLR.org, 2016.

[51] Xiaoge Zhang and Sankaran Mahadevan. Bayesian neural networks for flight trajectory prediction and safety assessment. Decision Support Systems, 131:113246, 2020.

[52] Michael I Jordan, Zoubin Ghahramani, Tommi S Jaakkola, and Lawrence K Saul. An introduction to variationa methods for graphical models. Machine learning, 37(2):183–233, 1999.

[53] S.F. Costa, F.M. Duarte, and J.A. Covas. Estimation of filament temperature and adhesion development in fused deposition techniques. Journal ofMaterials Processing Technology, 245:167 – 179, 2017.

[54] Caroline A Schneider, Wayne S Rasband, and Kevin W Eliceiri. NIH Image to ImageJ: 25 years of image analysis. Nature methods, 9(7):671–675, 2012.

[55] Francois Chollet et al. Keras, 2015.