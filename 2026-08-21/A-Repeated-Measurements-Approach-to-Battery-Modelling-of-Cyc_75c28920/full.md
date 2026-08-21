# A Repeated Measurements Approach to ��� Battery Modelling of Cyclic Aged Data in a Laboratory Environment

M. Cary<sup>1</sup> (Email: m.cary@lboro.ac.uk)

Charles Bokor<sup>2</sup> (Email: charlesbokor@outlook.com)

<sup>1</sup> Department of Aeronautical and Automotive Engineering, Loughborough University, Loughborough, LE11 3TU, UK

<sup>2</sup> School of Engineering, Computing and Mathematics, Oxford Brookes University Oxford OX33 1HX, UK

## Abstract

This document describes the application of a first order linearised nonlinear repeated measurements approach to the analysis of battery cell ageing profiles generated under controlled conditions in a laboratory. The primary advantage of the model is it reflects the obvious structure in the data. Consequently, it is a two-component of variance model: variation within ageing profiles (measurement noise) and variation among ageing profiles (test-to-test or cell-to-cell) variation. Novel regularised iterative generalised least squares parameter identification schemes, with optimal hyper-parameter re-estimation, are used to identify the hierarchical nonlinear model. The training data comprised ��� profiles for 10 cells aged at various constant discharge and charge current cycles at a fixed chamber environmental temperature of 25 [<sup>o</sup>C]. Each cell ��� profile is modelled using a simple power law expression, whereas the variation in ageing parameters is modelled using a single knot cubic B-spline. ��� is accurately predicted to ±0.191% for ��� ∈ [0,20].

Keywords: Repeated Measurements, Longitudinal Data, Maximum Likelihood Estimation, Regularisation

## 1 Introduction

In laboratory studies to assess the cyclic ageing characteristics of battery cells, state of health (���) metrics are taken repeatedly at several instances during the life of the cell [refs]. Typically, for each cell, the ageing conditions such as cell temperature, charge and discharge currents etc. are held constant and each cell is subjected to a predefined cyclic stress. The output from these experiments is not a collection of points but rather a collection of ageing profiles. The shape of the profiles will be similar for all cells of the same type but will vary according to the applied cyclic stress. Formally, these measurements constitute longitudinal data [i]. Longitudinal data is defined as repeated measurements where the observations on a single individual were not (or could not be) randomly assigned to the levels of a treatment of interest.

The presence of repeated observations on an individual requires care in characterising the variation in the experimental data. It is important to explicitly represent two sources of variation: random variation among measurements within a given cell (intra-individual) and random variation among cells (inter-individual). Inferential procedures accommodate these different variance components within the framework of an appropriate hierarchical statistical model. Note, intra-individual variation can be interpreted as measurement noise, whereas inter-individual variation corresponds to cell-to-cell variation [ii].

License: CC-BY 4.0, see https://creativecommons.org/licenses/by/4.0/

## 1.1 Model Formulation

This section describes the hierarchical non-linear model that forms the basis for the inferential procedures applied. Following the presentation by Davidian and Giltinan [iii], the model involves two levels. The first stage describes intra-cell variation within cells and the second inter-cell variation among cells.

## 1.2 Stage 1: Intra-cell Model

The model for the $i ^ { t h }$ cell may be written:

$$
{ \pmb y } _ { i } = f ( { \pmb x } _ { i } , { \pmb \beta } _ { i } ) + { \pmb e } _ { i }\tag{1}
$$

Where $\boldsymbol { y } _ { i } \in \mathbb { R } ^ { n _ { i } }$ is the corresponding response vector, $\pmb { x } _ { i } \in \mathbb { R } ^ { n _ { i } }$ is a vector of stage-1 covariates (independent or regressor variable), $\beta _ { i } \in \mathbb { R } ^ { p }$ is the parameter vector for the function $f ( \pmb { x } _ { i } , \theta _ { i } )$ describing the systematic ageing behaviour of the $i ^ { t h }$ cell. The vector $\pmb { e } _ { i } \in \mathbb { R } ^ { n _ { i } }$ represents the intracell random error and is assumed to be distributed as $\pmb { e } _ { i } = \mathcal { N } \big ( \pmb { 0 } , \sigma ^ { 2 } \pmb { \Sigma } _ { i } ( \pmb { \nu } ) \big )$ , with $\boldsymbol { \Sigma } _ { i } \in \mathbb { R } ^ { ( n _ { i } \times n _ { i } ) }$ . Here $\sigma ^ { 2 }$ is an intra-cell variance scale parameter and $\pmb { \nu } \in \mathbb { R } ^ { q }$ a vector of level-1 covariance model parameters defining heteroscedastic and serial correlation phenomenon. It is assumed $\sigma ^ { 2 }$ and � are common to all cells. Throughout, we will assume $\Sigma _ { i } = { \cal I } _ { n _ { i } } ,$ implying $\pmb { \nu } \in \phi$

## 1.3 Stage 2: Inter-Cell variation

Variation among different cells is accounted for via the following model:

$$
\beta _ { i } = d ( { \pmb a } _ { i } , \theta ) + F { \pmb b } _ { i }\tag{2}
$$

The vector valued function $\pmb { d } ( \pmb { a } _ { i } , \theta ) \in \mathbb { R } ^ { p }$ explains the systematic inter-cell variation associated with the level-2 covariates ${ \pmb a } _ { i } ,$ which represent the ageing conditions applied to the $i ^ { t h }$ cell. In this study, as with most laboratory investigations, the ${ \pmb a } _ { i }$ are assumed to be held constant while the $i ^ { t h }$ is aged. The level-2 random effects, $\pmb { b } _ { i } \in \mathbb { R } ^ { k }$ , are assumed to be distributed as $\pmb { b } _ { i } { \sim } \mathcal { N } ( 0 , \pmb { D } )$ , where $\pmb { D } \in$ $\mathbb { R } ^ { k \times k _ { \mathrm { i s } } }$ there level-2 covariance matrix. The matrix $\pmb { F } \in \mathbb { R } ^ { p \times k }$ is a fixed design matrix and permits elements of $\beta _ { i }$ to be represented as either mixed or fixed effects. It is also assumed $\mathbf { } e _ { i }$ and $\pmb { b } _ { i }$ are independent.

## 1.4 Approximation to the Marginal Density of $y _ { i }$

Maximum likelihood estimation for this model is based on the marginal density of $\mathbf { \Delta } y _ { i } ,$ i.e.:

$$
\begin{array} { r } { p ( { \pmb y } _ { i } ) = \int p ( { \pmb y } _ { i } | { \pmb b } _ { i } ) p ( { \pmb b } _ { i } ) d { \pmb b } _ { i } } \end{array}\tag{3}
$$

Unlike the linear case, this integral does not have a closed-form solution when $f ( \pmb { x } _ { i } , \beta _ { i } )$ is nonlinear in $\pmb { b } _ { i } .$ . Consequently, historically, various approximations have been proposed in the literature to evaluate Equation (3). For example, Pinheiro and Bates [iv] consider four different approximations to the log likelihood, comparing their computational and statistical properties. Here the discussion is restricted to methods involving taking a first-order Taylor series expansion of the model function $f ( \pmb { x } _ { i } , \beta _ { i } )$ about the expected value of the random effects [iii, $\mathsf { v } ,$ vi, vii, viii], $E [ { \pmb b } _ { i } ] = 0$ , a procedure first proposed in the pharmacokinetics literature by Sheiner, Rosenberg and Melmon [ix]. This yields the approximate form:

$$
\pmb { y } _ { i } = f ( \pmb { d } ( \pmb { a } _ { i } , \theta ) , 0 ) + J _ { i } ( \theta , 0 ) \pmb { F } \pmb { b } _ { i } + \sigma \Sigma _ { n _ { i } } ^ { 1 / 2 } \pmb { e } _ { i }\tag{4}
$$

Where $J _ { i } ( \theta , 0 ) \in \mathbb { R } ^ { n _ { i } \times p }$ denotes the matrix of derivatives of $f ( \pmb { x } _ { i } , \beta _ { i } )$ with respect to $\beta _ { i }$ evaluated at $\beta _ { i } = \pmb { d } ( \pmb { a } _ { i } , \theta )$ and $\Sigma _ { n _ { i } } ^ { 1 / 2 }$ is the Cholesky factor for $\textstyle \sum _ { i } [ \times ]$ . The notation highlights the fact that the expansion is taken about $E [ \pmb { b } _ { i } ] = 0$ . Defining the matrices, $\pmb { Z } _ { i } ( \theta , 0 ) = \pmb { J } _ { i } ( \theta , 0 ) \pmb { F }$ and $\pmb { e } _ { i } ^ { * } = \sigma \Sigma _ { n _ { i } } ^ { 1 / 2 } \pmb { e } _ { i } ,$ Equation (4) can be written in the compact form:

$$
\pmb { y } _ { i } = f ( \pmb { d } ( \pmb { a } _ { i } , \theta ) , 0 ) + \pmb { Z } _ { i } ( \theta , 0 ) \pmb { b } _ { i } + \pmb { e } _ { i } ^ { * }\tag{5}
$$

Note in (5), as in the linear case, the intra-cell and inter-cell random terms now enter the model in an additive fashion. The immediate consequence of (5) is the marginal mean and covariance of $y _ { i }$ are readily specified:

$$
E [ { \pmb y } _ { i } ] = f \big ( { \pmb d } ( { \pmb a } _ { i } , { \pmb \theta } ) \big )\tag{a}
$$

(6)

$$
V [ { \pmb y } _ { i } ] = { \pmb Z } _ { i } { \pmb D } ( \pmb \omega ) { \pmb Z } _ { i } ^ { T } + \sigma ^ { 2 } \Sigma _ { i } = { \pmb W } _ { i } ( \pmb \omega , \sigma ^ { 2 } )\tag{b}
$$

Given Equation (6), the marginal negative log likelihood for the full data set comprised of � cells is:

$$
\begin{array} { r } { L ( \boldsymbol { y } _ { i } | \boldsymbol { \theta } , \boldsymbol { \omega } , \boldsymbol { \sigma } ^ { 2 } ) = \sum _ { i = 1 } ^ { M } \left( \frac { n _ { i } } { 2 } \log ( 2 \pi ) + \frac { 1 } { 2 } l o g \| \boldsymbol { W } _ { i } \| + \frac { 1 } { 2 } \Big ( \boldsymbol { y } _ { i } - f \big ( \boldsymbol { d } ( \boldsymbol { a } _ { i } , \boldsymbol { \theta } ) \big ) \Big ) ^ { T } \boldsymbol { W } _ { i } ^ { - 1 } \left( \boldsymbol { y } _ { i } - f \big ( \boldsymbol { d } ( \boldsymbol { a } _ { i } , \boldsymbol { \theta } ) \big ) \right) \right) } \end{array}\tag{7}
$$

The parameters are estimated using a novel regularized iterative generalised least squares (RIGLS) procedure specified in [xi]. Benefits of regularization are discussed at length in [xii]. This is accomplished by introducing a ridge regression term [ii, xii, xiii] term of the form $\lambda \theta ^ { T } \theta$ into (7). When combined with an appropriate information theoretic measure, such as ��� [xiv], the hyperparameter, �, can be optimally determined at each iteration of the optimisation search algorithm using a numerically efficient fixed-point iteration formula derived in [xi]. Here we extend the approach to repeated measurements formulations, yielding the following formula, the derivation for which is outlined in Appendix A.

$$
\begin{array} { r } { \lambda _ { B I C } = \frac { \sigma ^ { 2 } l o g ( N ) t r a c e \left( \lambda A ^ { - 2 } - A ^ { - 1 } \right) } { 2 N \mathbf { q } ^ { T } B A ^ { - 3 } B ^ { T } \mathbf { q } } } \end{array}\tag{8}
$$

Equation (8) is of the form $\lambda = h ( \lambda )$ . Consequently, � appears on both sides of equation (8) and we must result to fixed-point iteration [xv] to estimate it.

Definition: If $h ( \lambda )$ is defined on $[ a , b ]$ and $h ( \lambda ) = \lambda$ for some $\lambda \in [ a , b ]$ , then the function $h ( \lambda )$ is said to have the fixed-point � in $[ a , b ]$

The following theorem guarantees convergence of the algorithm [xv]:

Theorem: Let $h ( \lambda )$ be continuous in $[ a , b ] .$ , and suppose that $h ( \lambda ) \in [ a , b ]$ for all $\lambda \in [ a , b ]$ . Further Let $h ^ { \prime } ( \lambda )$ exist on $( a , b )$ with $| h ^ { \prime } ( \lambda ) | \leq k < 1$ , for all $\lambda \in ( a , b )$ . If $\lambda _ { o } \in [ a , b ]$ , then the sequence defined by $\lambda _ { r } = h ( \lambda _ { r - 1 } ) , r \geq 1$ , will converge to the unique fixed-point � in $[ a , b ]$

The necessary proof can be found in the reference cited. Furthe $\mathsf { \Omega } _ { \mathsf { \Omega } } \mathsf { r } ,$ if the theorem is satisfied a bound for the error involved in using $\lambda _ { r }$ to approximate � is given by:

$$
| \lambda _ { r } - \lambda | \leq k ^ { r } m a x \lbrace \lambda _ { o } - a , b - \lambda _ { o } \rbrace , \forall r \geq 1 .\tag{9}
$$

1.5 Formulae for Confidence and Prediction Intervals Using the Delta Method Confidence intervals for the predictions are of great importance in the primary application of this work $i . e .$ , battery end of life prediction. Consequently, for any prediction we desire to provide a corresponding estimate of precision for that estimate. To do this we utilise the well-known deltamethod [iii], which involves expanding the function $f \big ( \pmb { d } ( \pmb { a } _ { i } , \theta ) \big )$ as a first order Taylor series about the estimate �<sup>̂</sup>.

$$
\hat { y } _ { i j } | \hat { \theta } \sim f ( x _ { i j } , \pmb { d } ( a _ { i } , \hat { \theta } ) ) + \pmb { h } ^ { T } ( x _ { i j } , \pmb { d } ( a _ { i } , \hat { \theta } ) ) \Delta \theta\tag{10}
$$

Where $\begin{array} { r } { h ^ { T } \left( x _ { i j } , d \big ( a _ { i } , \widehat { \theta } \big ) \right) = \partial f \big ( x _ { i j } , \beta _ { i } \big ) / \partial \beta _ { i } \times \partial \beta _ { i } \left( d \big ( a _ { i } , \widehat { \theta } \big ) \right) / \partial \theta = J _ { i } \big ( x _ { i j } , \widehat { \theta } \big ) \xi \in \mathbb { R } ^ { 1 \times q } , } \end{array}$ and $\boldsymbol { \Xi } =$ $\partial { d \big ( } a _ { i } , \widehat { \theta } { \big ) } / \partial \theta$ . Applying the variance operator to (10) yields:

$$
V \big [ \hat { y } _ { i j } \big | \hat { \theta } \big ] \sim \pmb { h } ^ { T } \big [ V \Delta \hat { \theta } \big ] \pmb { h }\tag{11}
$$

It can be shown:

$$
\begin{array} { r } { V \varDelta \hat { \theta } = \pmb { \phi } ^ { - 1 } \Bigl [ \sum _ { i = 1 } ^ { m } \pmb { \Xi } _ { i } ^ { T } \pmb { J } _ { i } ^ { T } \pmb { W } _ { i } ^ { - 1 } \pmb { J } _ { i } \pmb { \Xi } _ { i } \Bigr ] \pmb { \phi } ^ { - 1 } } \end{array}\tag{12}
$$

With $\begin{array} { r } { \phi ^ { - 1 } = \left[ \sum _ { i = 1 } ^ { m } \Xi _ { i } ^ { T } { \cal J } _ { i } ^ { T } { \cal W } _ { i } ^ { - 1 } { \cal J } _ { i } \Xi _ { i } + \lambda I \right] ^ { - 1 } } \end{array}$ . Substituting Equation (12) into Equation (11) yields:

$$
\begin{array} { r } { \left[ \widehat { y } _ { i j } \middle | \widehat { \vartheta } \right] \sim \pmb { h } ^ { T } \pmb { \phi } ^ { - 1 } \left[ \sum _ { i = 1 } ^ { m } \Xi _ { i } ^ { T } J _ { i } ^ { T } \pmb { W } _ { i } ^ { - 1 } J _ { i } \Xi _ { i } \right] \pmb { \phi } ^ { - 1 } \pmb { h } } \end{array}\tag{13}
$$

A $1 0 0 \times ( 1 - \alpha ) \%$ confidence interval may be constructed by appealing to standard asymptotic theory according to:

$$
\begin{array} { r } { \mu _ { i j } { \sim } \hat { y } _ { i j } \pm \sqrt { \chi _ { q } ^ { 2 } ( { \alpha } ) \pmb { h } ^ { T } \pmb { \phi } ^ { - 1 } \big [ \sum _ { i = 1 } ^ { m } \pmb { \Xi } _ { i } ^ { T } \pmb { J } _ { i } ^ { T } \pmb { W } _ { i } ^ { - 1 } \pmb { J } _ { i } \pmb { \Xi } _ { i } \big ] \pmb { \phi } ^ { - 1 } \pmb { h } } } \end{array}\tag{14}
$$

Uncertainty may be regarded as the precision up to which an unknown quantity can be determined from the available observed data. Predicting the value of a future observation is inherently more uncertain because it deals with future unobserved data. Prediction intervals related to predictions of future observations. A prediction interval quantifies the expected range for one or more additional future observations, including uncertainty about model parameter and random measurement variability. Let $\hat { z } _ { i , j }$ denote a future observation. Then a $1 0 0 \times ( 1 - \alpha ) \%$ prediction interval may be constructed using:

$$
\begin{array} { r } { \mu _ { i j } { \sim } \hat { z } _ { i j } \pm \sqrt { \chi _ { q } ^ { 2 } ( { \alpha } ) \big ( h ^ { T } \big [ \Phi ^ { - 1 } \big [ \sum _ { i = 1 } ^ { m } \Xi _ { i } ^ { T } J _ { i } ^ { T } W _ { i } ^ { - 1 } J _ { i } \Xi _ { i } \big ] \Phi ^ { - 1 } + W _ { i } \big ] h \big ) } } \end{array}\tag{15}
$$

## 2 Case Study Description

The relevant properties of the ten cells tested are summarised in Table 1. From beginning of life screening tests conducted on 55 pristine cells, Zülke et al [xvi] reported the mean nominal capacity of these cells as $4 . 7 3 \pm 0 . 0 2 5 ~ [ \mathsf { A h } ]$ . In Zülke’s study, capacity data was obtained from discharge tests conducted at one third C-rate and 25 [<sup>o</sup>C]. These data defined the initial full capacity, $Q _ { 0 } \mathsf { s a y } ,$ for the cells under test. All test cells were selected randomly from this batch.

Table 1: Cell Specification Table
<table><tr><td rowspan=1 colspan=1>Properties</td><td rowspan=1 colspan=1>Specification</td></tr><tr><td rowspan=1 colspan=1>Cathode chemistry</td><td rowspan=1 colspan=1>Nickel Cobalt Aluminum (NCA)</td></tr><tr><td rowspan=1 colspan=1>Anode chemistry</td><td rowspan=1 colspan=1>Graphite/Silicon</td></tr><tr><td rowspan=1 colspan=1>Typical capacity(4.2 V, C/3 discharge)</td><td rowspan=1 colspan=1>4.8 [Ah]</td></tr><tr><td rowspan=1 colspan=1>Typical energy(4.2 V, C/3 discharge)</td><td rowspan=1 colspan=1>17.4 [Wh]</td></tr><tr><td rowspan=1 colspan=1>Nominal voltage</td><td rowspan=1 colspan=1>3.62 [V]</td></tr><tr><td rowspan=1 colspan=1>Lower voltage limit, $\underline { { \mathsf { V } _ { m i n } } }$ </td><td rowspan=1 colspan=1>2.5 [V]</td></tr><tr><td rowspan=1 colspan=1>Upper voltage limit, $\underline { { \mathsf { V } _ { m a x } } }$ </td><td rowspan=1 colspan=1>4.2 [V]</td></tr><tr><td rowspan=1 colspan=1>Energy density</td><td rowspan=1 colspan=1>256 [Wh/kg];717 [Wh/L]</td></tr><tr><td rowspan=1 colspan=1>Standard charging current rate</td><td rowspan=1 colspan=1>C/3</td></tr><tr><td rowspan=1 colspan=1>Maximum charging current rate</td><td rowspan=1 colspan=1>1C</td></tr><tr><td rowspan=1 colspan=1>Standard discharging current rate</td><td rowspan=1 colspan=1>C/5</td></tr><tr><td rowspan=1 colspan=1>Maximum discharging current rate</td><td rowspan=1 colspan=1>2C</td></tr><tr><td rowspan=1 colspan=1>Peak discharging current rate (30s,10s) at 50%SOC and BOL</td><td rowspan=1 colspan=1>42 [A]/54[A]</td></tr><tr><td rowspan=1 colspan=1>Cell weight</td><td rowspan=1 colspan=1>68 [g]</td></tr><tr><td rowspan=1 colspan=1>Form factor</td><td rowspan=1 colspan=1>Cylindrical</td></tr></table>

Figure 1 depicts the test protocol applied each of the 10 cells tested. All data were collected at 25 [<sup>o</sup>C] and a constant discharge current of 2 [A] was applied throughout. In contrast, the applied charge current during cycling was varied according to the test plan defined by Table 2. After every 50 cycles, a check-up test was performed to determine the current cell capacity denoted by $Q _ { k }$ . Capacity loss, $Q _ { l o s s }$ was then calculated according to:

$$
\begin{array} { r } { Q _ { l o s s } = 1 0 0 \times \left[ \frac { Q _ { 0 } - Q _ { k } } { Q _ { 0 } } \right] } \end{array}\tag{16}
$$

![](images/b4cdbdb58c656717f9466ba46bdbc91b609b9ea556dcc33660fdeab09a196275.jpg)  
Figure 1: Cycling Ageing Test Protocol. A check-up capacity test is performed every 50 cycles. All cycles were conducted at 25 [<sup>o</sup>C] and a constant discharge current $o f 2 [ A ] .$ . The procedure was repeated until failure.

Table 2: Full Cycling Ageing Data Test Plan. The charging current is varied from 1 to $5 \ [ A ] .$ . Two cells are tested at each condition.
<table><tr><td rowspan=1 colspan=1>Basytec Channel</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>15</td></tr><tr><td rowspan=1 colspan=1>Cell Type</td><td rowspan=1 colspan=10>Delta/ Samsung 48X</td></tr><tr><td rowspan=1 colspan=1>Cell Serial No.</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>43</td></tr><tr><td rowspan=1 colspan=1>Vmax /V</td><td rowspan=1 colspan=1>4.2</td><td rowspan=1 colspan=1>4.2</td><td rowspan=1 colspan=1>4.2</td><td rowspan=1 colspan=1>4.2</td><td rowspan=1 colspan=1>4.2</td><td rowspan=1 colspan=1>4.2</td><td rowspan=1 colspan=1>4.2</td><td rowspan=1 colspan=1>4.2</td><td rowspan=1 colspan=1>4.2</td><td rowspan=1 colspan=1>4.2</td></tr><tr><td rowspan=1 colspan=1>Vmin /V</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>2.5</td></tr><tr><td rowspan=1 colspan=1>Ich/A</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>Idch /A</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>SOC</td><td rowspan=1 colspan=1>0-100%</td><td rowspan=1 colspan=1>0-100%</td><td rowspan=1 colspan=1>0-100%</td><td rowspan=1 colspan=1>0-100%</td><td rowspan=1 colspan=1>0-100%</td><td rowspan=1 colspan=1>0-100%</td><td rowspan=1 colspan=1>0-100%</td><td rowspan=1 colspan=1>0-100%</td><td rowspan=1 colspan=1>0-100%</td><td rowspan=1 colspan=1>0-100%</td></tr><tr><td rowspan=1 colspan=1>Cycle</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>50</td></tr></table>

## 2.1 Detailed Case Study Model Specification

At level-1 we model the cell capacity loss $( Q _ { l o s s } )$ versus ageing time (�) as a power law of the form:

$$
y _ { i j } = B _ { i } x _ { i j } ^ { \frac { 1 } { A _ { i } } } + e _ { i j } ,
$$

$$
s . t . A _ { i } > 1 , B _ { i } > 0\tag{17}
$$

Where suffix � refers to the cell and � the ageing time. It is assumed the random error term, $e _ { i j }$ , is normally identically and independently distributed, i.e., $e _ { i j } { \sim } \mathcal { N } ( 0 , \sigma ^ { 2 } )$ . Consequently, all data receives the same weight of unity. For this model $\beta _ { i } = [ A _ { i } B _ { i } ] ^ { T }$ . We shall refer to the function $f \left( x _ { i j } , \beta _ { i } \right) =$ $B _ { i } x _ { i j } ^ { 1 / A _ { i } }$ as the power law model (PLM). Figure 2 presents both the raw data for Cell 1 and the corresponding fit afforded by the PLM. These plots are typical for all 10 ten cells tested. Overall, the agreement between the data and the PLM is excellent, although the weighted residual versus predicted capacity fade plot demonstrates some slight systematic variation.

The level-2 model component $\pmb { d } ( \pmb { a } _ { i } , \theta )$ is intended to describe the variation in the $\beta _ { i }$ as the charging current is varied. Figure 3 presents the level-1 fit parameters as a function of charging current, $a _ { i }$ Generally, the variation in the coefficient values for the replicated tests at each $a _ { i }$ is relatively small except for observations at $a _ { i } = 3 [ A ]$ . Here, the variation in coefficient values exhibited by the two cells tested is quite large by comparison with the remainder. Upon inspection of the data, cell 5 was deemed to be an outlier and subsequently excluded from the analysis. Regardless, we assume the following level-2 model:

$$
\begin{array}{c} [ \begin{array} { l l l l l } { A _ { i } } \\ { = [ ^ { [ \phi _ { 1 } ( a _ { i } , k _ { A } ) } } & { \cdots } & { \phi _ { 5 } ( a _ { i } , k _ { A } ) ] } & { 0 } \\ { B _ { i } } \end{array} ] [ \begin{array} { l } { \theta _ { A } } \\ { \theta _ { B } } \end{array} ] + [ \begin{array} { l } { b _ { A , i } } \\ { b _ { B , i } } \end{array} ]  \end{array} ] ~ .\tag{18}
$$

Where the $\left\{ \phi _ { q } ( a _ { i } , k ) \right\} _ { q = 1 } ^ { 5 }$ are the B-spline basis functions for a spline of order 4 with a single knot (�) [xi, xvii]. The order of the B-spline, �, is given by:

$$
m = d + 1\tag{19}
$$

Where � denotes the degree of the interpolating polynomial. Hence, in this case, for the full level-2 model cited $\theta ^ { T } = [ k _ { A } \quad \theta _ { A } ^ { T } \quad k _ { B } \quad \theta _ { B } ^ { T } ] \in \mathbb { R } ^ { 1 2 }$ . In (18) we assume that both $A _ { i }$ and $B _ { i }$ are mixed-effects and therefore $\pmb { F } = \pmb { I } _ { 2 }$ . When there are several level-2 covariates and $\pmb { a } _ { i } ^ { T } = [ \pmb { a } _ { i , 1 } \quad \cdots \quad \pmb { a } _ { i , c } ]$ , the corresponding tensor product B-spline basis [xvii] possess a very large number of terms implying the necessity to collect large amounts of experimental data to estimate �. Under these circumstances, we suggest using a more compact basis such as the hybrid b-splines due to Grove et al [xviii] or for sparse high dimensional data a fully non-parametric approach such as the radial basis functions [ii].

![](images/4e1cb8a6594adbef75e8b700127a63168e276ee891bc268bede00d307ac08381.jpg)

![](images/deb503f1f14ca077f9764f1a1a1dc8c813987e95abd33ad062a8cb246a57eedf.jpg)

![](images/630051b671b6f3367c1afcf5111ae3768b90d198c5afa4e08a2f975daab93838.jpg)

![](images/dbe61f3bfff36bae09549622a80cdda2620c137cbde20c7ee827c740f24ecf51.jpg)  
Figure 2: Raw capacity loss $( Q _ { l o s s } )$ versus ageing time for cell 1, charge current = 1 [A]. The data is well approximated by the power law function $f ( x , \theta ) = B x ^ { ( 1 / } A )$ . The appearance of the regression diagnostic plots could be improved but overall, the PLM is in excellent agreement with the raw data. This is an ordinary least squares fit so all residual weights are unity.

![](images/91efe07c64d4879fd62d82d0e5f2b8c840e7418889cfd3c19ac1acc901c7402e.jpg)  
Figure 3: Variation in the level-1 fit coefficients as the charging current is varied. There appears to be an outlying observation at charging current = 3 [A]

## 2.2 Detailed Model Selection and Identification Procedure

One of the advantages of the linearised analysis approach over alternatives, such as two stage regression [xix, xx, ii, iii], is that complete ageing profiles are not required. With two-stage regression it is assumed sufficient cell-specific data exists to fit individual PLM profiles for each cell, that is estimate the $\beta _ { i } , \forall i = 1 \ldots M$ . The advantage of this with respect to model selection and for estimating starting values for $\theta$ is discussed at length in [ii]. Essentially, if preliminary values for the $\beta _ { i }$ are available then we can treat these estimates as data and model each element alone as a function of the $a _ { i } .$ . This is referred to as univariate fitting. The univariate estimates for $\theta ^ { T } = [ k _ { A } \quad \theta _ { A } ^ { T } \quad k _ { B } \quad \theta _ { B } ^ { T } ]$ then provide starting estimates for the full RIGLS procedure. Since here full ageing profile data is available, we also utilise this approach, simplifying the analysis process considerably.

Figure 4 presents a block diagram describing the full identification process workflow. The process begins by fitting each individual ageing profile for all 10 cells tested. At step 2, the common level-1 variance scale parameter, $\sigma ^ { 2 }$ , is estimated using a pooled analysis procedure described in [ii]. Once estimated, ${ \widehat { \sigma } } ^ { 2 }$ is held fixed throughout. Given the $\beta _ { i } ^ { T } = [ A _ { i } B _ { i } ]$ from step 1, at step 3, we initially treat the $A _ { i }$ and $B _ { i }$ estimates from step 1 as data, and perform regularised nonlinear univariate regressions for each utilising the procedure given in [xi]. This step provides preliminary estimates for $\hat { \theta } ^ { T } = \left[ \hat { k } _ { A } \quad \hat { \theta } _ { A } ^ { T } \quad \hat { k } _ { B } \quad \hat { \theta } _ { B } ^ { T } \right]$

![](images/cd6777d989be4b4f59c0d5886d660447d5942c8ed7001a4ac0b49f032cde90c5.jpg)  
Figure 4: Block diagram depicting the nonlinear repeated measurements model identification procedure

Using the preliminary estimates for ${ \widehat { \theta } } ^ { T }$ from step $^ { 3 , }$ we form the residual vectors, $\mathbf { \boldsymbol { r } } _ { i } = \mathbf { \boldsymbol { y } } _ { i } - \mathbf { \boldsymbol { \mathit { r } } }$ $f \big ( \pmb { d } ( \pmb { a } _ { i } , \theta ) \big )$ , and estimate $\pmb { \omega }$ by minimising (7) for fixed ${ \widehat { \theta } } ^ { T }$ and ${ \widehat { \sigma } } ^ { 2 }$ . For compactness, we denote the collection of � residual vectors as $R ( k _ { A } , \theta _ { A } , k _ { B } , \theta _ { B } )$ . In step 4, we first factor it as $\pmb { D } = \pmb { M } ^ { 2 }$ where $\pmb { M } ( \pmb { \omega } )$ is a real symmetric matrix. As shown in Appendix B this factorisation ensures the positive definiteness of �. At step 5, for fixed $\widehat { \mathbf { \Omega } } _ { \pmb { \omega } }$ and ${ \hat { \sigma } } ^ { 2 }$ , we now form the necessary weight matrices $W ( \widehat { \pmb { \omega } } ) _ { i } ^ { - 1 }$ and identify improved ${ \widehat { \theta } } ^ { T }$ by minimising the regularised form of (7) using the algorithm outlined in Appendix A. Steps 4 and 5 are then repeated until convergence. In our experience convergence occurs after roughly 3-5 iterations.

## 2.3 Automatic Outlier Detection

Prior to step 1, to improve robustness of the overall procedure in practice, we conduct a preliminary fit of cell-specific ageing profiles using the nonlinear FAST-LTS (Least Trimmed Squares) procedure developed by Hawkins and Khan [xxi]. The FAST-LTS method involves minimising the cost function:

$$
\begin{array} { r } { L _ { L T S } = \sum _ { s = 1 } ^ { S } r _ { s } ^ { 2 } } \end{array}\tag{20}
$$

Where $S \in [ 0 . 5 n _ { i } , n _ { i } ]$ . This is a subset selection method with proportion $1 - ( S ) / n _ { i }$ set aside from the fit. Essentially, the $L _ { L T S }$ criterion is the sum of the � smallest ordered squared residuals. The fit involves iteration rather than closed form explicit solutions and may converge to a local rather that a global minimum. Hawkins and Khan propose a hybrid method involving the concept of element sets. Initially, randomly generated elemental sets are used to provide a fast search for the initial estimate of the $\beta _ { i }$ . The elemental estimate minimising (20) is used subsequently as a starting value for a full nonlinear fit to the data – the so-called concentration step. The $\beta _ { i }$ at convergence of the concentration step is the final estimate. Data points corresponding to $r _ { s > s } ^ { 2 }$ are permanently excluded from any subsequent analysis.

## 2.4 Results & Discussion

We begin by considering the level-1 fit before moving onto discussing the efficacy of the full nonlinear repeated measurements model. Figure 5 presents the level-1 PLM fit to the data for cell 8, aged at a charging current of 4 [A]. Again, the fit to the data is excellent although the appearance of the residual diagnostics could be improved. The value of the hyper-parameter at convergence is $\lambda = 0 . 0 0 0 6 7$ yielding a very slight reduction in the effective number of parameters from 2, for the unregularized fit, to $\gamma = 1 . 9 9 7$ . This seems sensible given the apparent smoothness of the response. The nonlinear FAST-LTS algorithm [xxi] was applied to these data to identify outliers prior to finally fitting the model utilising the regularised method [xi]. Note, the FAST-LTS analysis identifies the presence of an aberrant observation at (10.75, 3.80). From inspection of the plot this seems entirely reasonable. This observation has been set aside for the remainder of the analysis.

Similarly, Figure 6 presents data for cell 3. Again, the reported values of λ and � at convergence are consistent with the inherent smoothness of the fit function. In this instance, two outlying observations occur. The first, as for cell 8, occurs early in the life of the cell at (10.75, 4.01). However, the second occurs at (490.5, 19.50) and may indicate the cell is entering into a period of rapid capacity decay which occurs at the end of the life cycle. If so, this observation is not consistent with the PLM formulation. Consequently, this observation should be excluded. In addition, if included, an outlying observation of this nature may be unduly influential on the PLM estimates for this cell; a point discussed by Cook et al [xxii].

![](images/37f9fe341e72595d9727faf931d6fe60249e61459997aaaea80c9afde90ca444.jpg)  
Figure 5: Cell 8, charging $c u r r e n t = 4 \ : [ A ] .$ . Level-1 PLM fit diagnostics. At convergence $( A _ { i } , B _ { i } ) = ( 1 . 3 8 3 , 1 . 1 0 1 1 )$ , the value of the hyper-parameter is $\lambda = 0 . 0 0 0 6 7 ,$ yielding $\gamma = 1 . 9 9 7 .$

The pooled estimate of the level-1 variance scale parameter, $\sigma ^ { 2 } , \mathsf { w a s } 0 . 0 0 7 7 ^ { 2 }$ . This reassuringly small value implies the fit quality for the PLM over all 9 cells was highly accurate.

Figure 7 presents the residuals for the training data for the full nonlinear repeated measurements model. The RIGLS algorithm converged after just 2-iterations. The ���� for the trained model was 0.191 [%]. Clearly, the appearance of this plot could be improved as the residual pattern suggests the model exhibits small systematic errors. The presence of these systematic errors is not considered practically significant in this application as the model still yields a very high level of accuracy. Validation data was not available to test the predictive capability of the model for previously untried configurations.

![](images/8515dc9b8d89d99aff04f79042b15dafaa37a60c5b7d798823836e0a2b1ac5e1.jpg)  
Figure 6: Cell 3, charging $c u r r e n t = 2 \ [ A ] .$ . Level-1 PLM fit diagnostics. At convergence, $( A _ { i } , B _ { i } ) = ( 1 . 3 7 4 , 1 . 1 0 3 4 )$ the value of the hyper-parameter is $\lambda = 0 . 0 0 0 4 8$ , yielding $\gamma = 1 . 9 9 8$

![](images/bd12d095cd3921523663fcea51e7e2771b3ab3be20ba6bb086d0236fab1228cf.jpg)  
Figure 7: Training residuals for the full nonlinear repeated measurements model with cell number as a parameter. The RMSE for these data is 0.191 [%]. At convergence $\lambda = 0 . 0 0 0 2 .$ . The RIGLS algorithm converged after just two iterations.

We now factorise the level-2 covariance matrix as:

$$
\pmb { D } = \pmb { T } ^ { 1 / 2 } \pmb { C } \pmb { T } ^ { 1 / 2 }\tag{21}
$$

Where � is a correlation matrix and $\pmb { T } ^ { 1 / 2 }$ is a diagonal matrix of standard errors. The correlation matrix is:

$$
\pmb { C } = \big [ - _ { 0 . 7 4 3 } ^ { 1 } - 0 . 7 4 3 _ { } \big ]\tag{22}
$$

The relatively high value of the off leading diagonal term indicates the utility of the regularised maximum likelihood approach to parameter identification. In this case, if the $\begin{array} { r l } { [ A _ { i } } & { { } B _ { i } ] } \end{array}$ were assumed

to be independent, then as explained in [iii] the corresponding estimates of � would be upwardly biased. Similarly, the standard error matrix is:

$$
\pmb { T } ^ { 1 / 2 } = d i a g \lbrace 0 . 0 3 7 4 \quad 0 . 0 2 3 3 \rbrace\tag{23}
$$

Again, given �, ��� $B _ { i }$ are of unity order this implies the cell-to-cell variation is quite small in practice. This assertion seems quite in keeping with the individual cell estimates presented in Figure 3. Figure 8 compares the level-1 PLM (L1) and full repeated model (L2) fits to the observed data for cell 2. In addition, 95% confidence and prediction intervals are also plotted. The similarity between the L1 and L2 fits is marked, again implying the full hierarchical nonlinear repeated measurements model is a good facsimile for the training data. The 95% confidence interval for the L2 estimates is exceedingly narrow. To emphasise this point, in Figure 9 we present the same comparisons in the interval [200,250] days. The 95% confidence interval is now clearly visible.

![](images/16072639f565cdbcfcc086c54c31f3ffcff0e02baff0321f6e3dff21a99e7725.jpg)  
Figure 8: Cell - 2, Ageing Current = 1 [A], comparison among fits and interval estimates. L1 denotes the level-1 PLM fit, L2 denotes the repeated measurements model fit, LCI (UCI) denotes the lower (upper) 95% confidence interval and LPI (UPI) denotes the lower (upper) 95% prediction interval.

![](images/f92bdbf878eb59db027486854056f58fa5cc87a43bc31325678cc76a484bda2f.jpg)  
Figure 9: Ageing Current = 1 [A], comparison among fits and interval estimates in the interval [200,250]. L1 denotes the level-1 PLM fit, L2 denotes the repeated measurements model fit, LCI (UCI) denotes the lower (upper) 95% confidence interval and LPI (UPI) denotes the lower (upper) 95% prediction interval.

## 3 Conclusions & Future Work

The nonlinear repeated measurements model presented, based on first order linearisation approximations to the relevant marginal likelihood, accurately reflects the obvious structure in laboratory cyclically aged cell data, accounting for both intra- and inter-cell variation. To our knowledge, this is the first time an analysis of this nature has been attempted for cyclic ageing ��� modelling. The effectiveness, of a novel regularised iterative generalised least squares approach to identification is demonstrated, via a case study, yielding a model with a training ���� of 0.191 [%] for the capacity loss data presented. In addition, the model is fully parametric and so easily interpretable from a scientific perspective. A hierarchical model, with two components of variance, yields realistic confidence and prediction intervals for the capacity loss ��� measure considered.

For the analysis of real-world data, the model structure requires modification. This is because in an electric vehicle (EV) application, for example, the ageing stress applied to the battery pack varies with environmental conditions, charging practices and customer usage. Regardless, in our view, real world data is still repeated measurements. Extensions to time dependent covariate [iii] protocols may address the real-world in-service analysis problem.

## Appendix A – Fixed Point Iteration Scheme Formula

It is possible to optimally select the hyper-parameter at each iteration of the search designed to determine �. We begin by expanding $f \big ( \pmb { d } ( \pmb { a } _ { i } , \theta , \mathbf { 0 } ) \big )$ as a first order Taylor-series with respect to the current estimate at the $s ^ { t h }$ iteration, $\theta _ { s }$

$$
f \big ( d ( a _ { i } , \theta , \mathbf { 0 } ) \big ) \approx f \big ( d ( a _ { i } , \theta _ { s } , \mathbf { 0 } ) \big ) + J _ { i } ( \theta _ { s } , \mathbf { 0 } ) \varOmega _ { i } ( \theta _ { s } , \mathbf { 0 } ) \varDelta \vartheta = f \big ( d ( a _ { i } , \theta _ { s } , \mathbf { 0 } ) \big ) + \psi _ { i } ( \theta _ { s } , \mathbf { 0 } ) \varDelta \theta\tag{A1}
$$

With $\Omega _ { i } ( \theta _ { s } , \mathbf { 0 } ) = \partial \beta _ { i } / \partial \theta _ { s }$ . We now factorise ${ \pmb D } = \sigma ^ { 2 } { \pmb D }$ which implies ${ \pmb W } _ { i } = \sigma ^ { 2 } { \pmb W } _ { i } ( { \pmb \omega } )$ . Using (A1) and defining $\boldsymbol { \psi } = \bigoplus _ { i = 1 } ^ { m } \boldsymbol { \psi } _ { i } , \quad \boldsymbol { W } = \bigoplus _ { i = 1 } ^ { m } \boldsymbol { W } _ { i } ( \boldsymbol { \omega } ) \quad \mathrm { ~ a n d ~ } \quad g = v e c [ g _ { 1 } \quad . . . \quad g _ { m } ] ,$ with $\mathbf { \nabla } \pmb { g } _ { i } = \pmb { y } _ { i } - \mathbf { \nabla }$ $f \big ( d ( a _ { i } , \theta _ { s } , 0 ) \big )$ , the appropriate combined likelihood can be written:

$$
\begin{array} { r l } & { L ( A \theta , \omega , \sigma ^ { 2 } ) = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { m } \{ n _ { i } l o g ( 2 \pi ) + n _ { i } l o g ( \sigma ^ { 2 } ) \} + \frac { 1 } { 2 } \sum _ { i = 1 } ^ { m } l o g \| { \bf W } _ { i } ( \omega ) \| + \frac { 1 } { 2 \sigma ^ { 2 } } ( g - \sigma ^ { 2 } ) ^ { 2 } } \\ & { \psi _ { A \theta } ) ^ { T } { \bf W } _ { i } ( \omega ) ( g - \psi _ { A \vartheta } ) } \end{array}\tag{A2}
$$

Where $\oplus$ is the direct sum operator [xxiii]:

$$
\displaystyle \oplus _ { i = 1 } ^ { m } A _ { i } = { \left[ \begin{array} { l l l } { A _ { 1 } } & { 0 } & { 0 } \\ { 0 } & { \ddots } & { 0 } \\ { 0 } & { 0 } & { A _ { m } } \end{array} \right] } .
$$

Likewise, the ��� operator is defined as [xxiii]:

$$
v e c [ { \pmb a } _ { 1 } \quad . . . \quad { \pmb a } _ { m } ] = \left[ \begin{array} { c } { { \pmb a } _ { 1 } } \\ { { \vdots } } \\ { { \pmb a } _ { m } } \end{array} \right]
$$

By definition, � is positive definite and so can be written:

$$
{ \pmb W } = { \pmb U } ^ { T } { \pmb U }\tag{A3}
$$

Where � is upper triangular. Using standard results on matrix inverses [xxiii], $\pmb { W } ^ { - 1 } = \pmb { U } ^ { - 1 } \pmb { U } ^ { - T }$ and inserting this relation into (A2) and defining $\pmb q = \pmb U ^ { - T } \pmb g$ and $\pmb { { \cal B } } = \pmb { { \cal U } } ^ { - T } \pmb { \mathcal { W } }$ yields:

$$
\begin{array} { r } { L ( A \theta , \omega , \sigma ^ { 2 } ) = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { m } \lbrace n _ { i } l o g ( 2 \pi ) + n _ { i } l o g ( \sigma ^ { 2 } ) \rbrace + \frac { 1 } { 2 } l o g \| \boldsymbol { U } \| ^ { 2 } + \frac { 1 } { 2 \sigma ^ { 2 } } ( \boldsymbol { q } - \boldsymbol { B } \boldsymbol { \varDelta \theta } ) ^ { T } ( \boldsymbol { q } - \boldsymbol { B \varDelta \theta } ) } \end{array}\tag{A4}
$$

We now add a ridge regression term to (A4) to yield:

$$
\begin{array} { r l } & { R ( A \theta , \omega , \sigma ^ { 2 } , \lambda ) = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { m } \{ n _ { i } l o g ( 2 \pi ) + n _ { i } l o g ( \sigma ^ { 2 } ) \} + \frac { 1 } { 2 } l o g \| \boldsymbol { U } \| ^ { 2 } + \frac { 1 } { 2 \sigma ^ { 2 } } ( q - B A \theta ) ^ { T } ( q - B A \theta ) + } \\ & { \frac { \lambda } { 2 \sigma ^ { 2 } } \varDelta \theta ^ { T } \varDelta \theta } \end{array}\tag{A5}
$$

Equation (A5) is linear in ��and thus it immediately follows:

$$
\begin{array} { r } { \varDelta \theta = ( B ^ { T } B + \lambda I ) ^ { - 1 } B ^ { T } \mathbf q } \end{array}\tag{A6}
$$

We now profile the likelihood for $\sigma ^ { 2 }$ by first substituting (A6) into (A4), differentiating the result with respect to $\sigma ^ { 2 }$ , and after some algebra, yield the closed form solution:

$$
\begin{array} { r } { { \boldsymbol { \sigma } } ^ { 2 } = \frac { { \boldsymbol { q } } ^ { T } \left( I - S ( { \boldsymbol { \lambda } } ) \right) ^ { 2 } { \boldsymbol { q } } } { N } } \end{array}\tag{A7}
$$

Where $\textstyle N = \sum _ { i = 1 } ^ { m } n _ { i }$ and $\pmb { S } = \pmb { B } ( \pmb { B } ^ { T } \pmb { B } + \lambda \pmb { I } ) ^ { - 1 } \pmb { B } ^ { T }$ is the so-called smoother matrix. Substituting (A7) into (A4) yields the corresponding profile likelihood:

$$
\begin{array} { r } { L ^ { p } ( \pmb { \omega } , \lambda , \sigma ^ { 2 } ) = \frac { N } { 2 } l o g ( 2 \pi ) + \frac { 1 } { 2 } l o g \| \pmb { U } \| ^ { 2 } + \frac { N } { 2 } l o g \left( \frac { \pmb { q } ^ { T } \left( I - S ( \lambda ) \right) ^ { 2 } \pmb { q } } { N } \right) + \frac { 1 } { 2 } N } \end{array}\tag{A8}
$$

The information theoretic criterion ��� [xiv] is defined as:

$$
B I C = 2 L ^ { p } ( \pmb { \omega } , \lambda , \sigma ^ { 2 } ) + \gamma l o g ( N )\tag{A9}
$$

Where $\gamma = t r a c e ( S ( \lambda ) )$ denotes the effective number of parameters. Differentiating (19) with respect to � yields:

$$
\begin{array} { r } { \frac { \partial B I C } { \partial \lambda } = 2 \frac { \partial L _ { p } } { \partial \lambda } + l o g ( N ) \frac { \partial \gamma } { \partial \lambda } } \end{array}\tag{A10}
$$

Using the following results taken from [xi], and after some algebra, it can be shown:

$$
\frac { \partial \pmb { S } } { \partial \lambda } = - \pmb { B } \pmb { A } ^ { - 2 } \pmb { B } ^ { T }\tag{a}
$$

$$
\begin{array} { r } { \frac { \partial S ^ { 2 } } { \partial \lambda } = - 2 B A ^ { - 2 } B ^ { T } + 2 \lambda B A ^ { - 3 } B ^ { T } } \end{array}\tag{b}
$$

(A11)

$$
\begin{array} { r } { \frac { \partial \gamma } { \partial \lambda } = t r a c e ( \lambda A ^ { - 2 } - A ^ { - 1 } ) } \end{array}\tag{c}
$$

Where $\pmb { A } = ( \pmb { B } ^ { T } \pmb { B } + \lambda \pmb { I } )$ . Substituting (A11) into (A10):

$$
\begin{array} { r } { \frac { \partial B I C } { \partial \lambda } = 2 \lambda \left( \frac { N } { \sigma ( \lambda ) ^ { 2 } } \right) { \pmb q } ^ { T } B A ^ { - 3 } { \pmb B } ^ { T } { \pmb q } + l o g ( N ) t r a c e ( \lambda A ^ { - 2 } - A ^ { - 1 } ) } \end{array}\tag{A12}
$$

Setting (A12) to 0 and solving for � gives the desired result:

$$
\begin{array} { r } { \lambda _ { B I C } = \frac { \sigma ^ { 2 } l o g ( N ) t r a c e \left( \lambda A ^ { - 2 } - A ^ { - 1 } \right) } { 2 N \mathbf { q } ^ { T } B A ^ { - 3 } B ^ { T } \mathbf { q } } } \end{array}\tag{A13}
$$

Appendix B - Ensuring Positive Definite �

Let $\pmb { M } ( \pmb { \omega } ) \in \mathbb { R } ^ { k \times k }$ denote a real symmetric square matrix. An eigen decomposition of � yields:

$$
\pmb { M } = \pmb { P } ^ { - 1 } \pmb { \Lambda } \pmb { P }\tag{B1}
$$

Where � is an appropriately dimensioned matrix of eigenvectors and $\pmb { \Lambda } = d i a g \{ \lambda _ { i } \} _ { i = 1 } ^ { k }$ , where the $\lambda _ { i }$ are the eigenvalues of �. Now assume:

$$
\pmb { D } = \pmb { M } ^ { 2 } ( \pmb { \omega } )\tag{B2}
$$

Applying (B1) to (B2):

$$
\pmb { { \cal D } } = \pmb { { \cal P } } ^ { - 1 } \pmb { \Lambda } \pmb { { \cal P } } \pmb { { \cal P } } ^ { - 1 } \pmb { \Lambda } \pmb { { \cal P } } = \pmb { { \cal P } } ^ { - 1 } \pmb { \Lambda } ^ { 2 } \pmb { { \cal P } }\tag{B3}
$$

Clearly, the eigenvalues of � are $\lambda _ { i } ^ { 2 }$ and therefore cannot be negative ensuring positive definite �.

i M. J. Lindstrom, D. M., Bates, Nonlinear Mixed Effects Models for Repeated Measures Data, Biometrics, 1990, Vol. 46, pp 673-687.

ii Cary, M., A Model Based Methodology for the Calibration of a Port Fuel Injection, Spark-Ignition Engine, PhD Thesis, University of Bradford, 2003.

iii M. Davidian, D. M. Giltinan, Nonlinear Models for Repeated Measurement Data, 1995, Chapman & Hall.

iv J. C. Pinheiro, J. C, D. M. Bates, Approximations to the Log-likelihood function in Nonlinear Mixed-Effects Models, Journal of Computational and Graphical Statistics, 1995, Vol. 4, No. 1, pp. 12-35.

v M. Davidian, D. M. Giltinan, Some General Estimation Methods for Nonlinear Mixed-Effects Models, Journal of Biopharmaceutical Statistics, 1993, Vol. 3, part 1, pp 23-55.

vi M. Davidian, D. M. Giltinan, Analysis of repeated measurement data using the nonlinear mixed effects model, Chemometrics and Intelligent Laboratory Systems, 1993, Vol. 20, pp 1-24.

vii M. J., Lindstrom, D. M., Bates, Nonlinear Mixed Effects Models for Repeated Measures Data, Biometrics, 1990, Vol. 46, pp 673-687.

viii L. B. Sheiner, S. L., Beal, Evaluation of Methods for Estimating Population Pharmacokinetic Parameters. I. Michaelis-Menton Model: Routine Clinical Pharmacokinetic Data, Journal of Pharmacokinetics and Biopharmaceutics, 1980, Vol 8, pp 553-571.

ix L. B. Sheiner, B., Rosenberg, K. l., Melmon, Modeling of Individual Pharmacokinetics for Computer-Aided Drug Dosing, Computers and Biomedical Research, 1972, Vol. 5, pp 441-459.

x R. L., Burden, J. D., Faires, A. C. Reynolds, Numerical Analysis, Wadsworth International Student Edition, Second Edition, 1981.

xi M. Cary, H. Hoster, A. Aragon Zulke, J. Elgie, Regularised Iterative Generalised Least Squares with Optimal Selection of the Hyper-Parameter for Identifying Nonlinear Phenomenological Models, 6th Biennial International Conference on Powertrain Modelling and Control, Testing, Mapping and Calibration, Loughborough University, Sep 2022.

xii C. Goutte, Statistical Learning and Regularisation for Regression – Application to system identification and time series modelling, Ph.D. Thesis, L'Universite' Paris, 1996

xiii A. E., Hoerl, R. W., Kennard, Ridge Regression: biased estimation for non-orthogonal problems, Technometrics, 1970, Vol. 12, pp 55-67.

xiv G. Schwarz, (1978), Estimating the dimension of a model, Annals of Statistics. 6, 461-464.

xv R. L. Burden, J. D.., Faires, A. C. Reynolds, Numerical Analysis, Wadsworth International Student Edition, Second Edition, 1981.

xvi A. Zülke et al., Batter. Supercaps, 4, batt.202100046 (2021)

xvii C. De Boor, A Practical Guide to Splines, Springer-Verlag, First Edition, 1978.

xviii D. M., Grove, D. C. Woods, S. M. Lewis, Multifactor B-Spline Models in Designed Experiments for the Engine Mapping Problem, Journal of Quality Technology, 2004, 35, pp. 380-391.

xix T. Holliday, A J. Lawrance, T. P., Davis, Engine-Mapping Experiments: A Two-Stage Regression Approach, Technometrics, 1998, Vol. 40, pp 120-126.

xx D. W. Rose, M. Cary, R. Sbaschnig, K. M. Ebrahimi, An Engine Mapping Case Study: A 2-Stage Regression Approach, IMECHE International Conference on Statistics and Analytical Methods in Automotive Engineering, London UK, Sept. 24-25, 2002, pp 53 – 80.

xxi D. M. Hawkins, D. M. Khan, A procedure for robust fitting in nonlinear regression, Computational Statistics and Data Analysis, 2009, 53, pp 4500-4507.

xxii R. D. Cook, C. -L. Tsai, B. C. Wei, Bias in nonlinear regression, Biometrika, 1986, 73, pp 615- 623.

xxiii S. R. Searle, Matrix Algebra Useful for Statistics, John Wiley & Sons, First Edition, 1982.