# Regularised Iterative Generalised Least Squares with Optimal Selection of the Hyper-Parameter for Identifying Nonlinear Phenomenological Models

M. Cary<sup>1</sup> (Email: m.cary@lboro.ac.uk)

Charles Bokor<sup>2</sup> (Email: charlesbokor@outlook.com)

<sup>1</sup> Department of Aeronautical and Automotive Engineering, Loughborough University, Loughborough, LE11 3TU, UK

<sup>2</sup> School of Engineering, Computing and Mathematics, Oxford Brookes University, Oxford OX33 1HX, UK

## Abstract

In some fields currently dominated by empirical approaches, such as state of health (SoH) prediction for lithium-ion batteries, phenomenological models motivated by quasi-physical thinking contain parameters to be estimated from experimental data. Often the structure of such models yields fully or partially confounded parameters, which are difficult or even impossible to estimate reliably. To preserve the desired model formulation and simultaneously improve the numerical conditioning for the problem we introduce a ridge regression scheme. An automated method is provided, based on information theoretic measures of model performance, which optimises the ridge regression hyperparameter at each iteration. The formulae presented require fixed point iteration to solve for the hyper-parameter. Given a suitable starting value, analysis demonstrates convergence is very rapid. The optimal hyper-parameter selection mechanism is incorporated within an efficient regularised iterative generalised least squares mechanism, capable of fitting both heteroscedastic and serially correlated data as required. Simulation confirms the efficacy of the overall method.

Keywords: Maximum Likelihood Estimation, Regularisation, Optimal Hyper-Parameter Estimation.

## 1 Preamble

The operation of Lithium-ion batteries creates ageing and degradation, which needs to be measured or predicted to keep the battery safe. Online state of health (���) estimation is a key task of practical automotive battery management systems. Essentially ��� is a metric to estimate the ageing level of batteries, which includes capacity fade and power degradation. Particularly, it is the key metric to determine when a battery should be replaced. Phenomenological models for ��� estimation of Lithium-ion batteries rely heavily on quasi-physical motivations. Often these models are comprised of a power law like term multiplied by gains and Arrhenius-like exponential terms, which act as a form of rate constant [i, ii, iii]. Often the multiplicative nature of the model structure calls into question the apparent identifiability of the model.

A model is identifiable if it is theoretically possible to learn the true values of this model's underlying parameters after obtaining an infinite number of observations from it. To make things concrete let $P = \{ P _ { \theta } \colon \theta \in \theta \}$ denote a statistical model, where the parameter space, �, is finite-dimensional. The model $P$ is identifiable if the mapping $\theta  P _ { \theta }$ is one-to-one; i.e. $P _ { \theta _ { 1 } } = P _ { \theta _ { 2 } } \Rightarrow \theta _ { 1 } = \theta _ { 2 } , \forall \theta _ { 1 } , \theta _ { 2 } \in \theta$ Identifiability of the model, in the sense of the invertibility of the mapping $\theta  P _ { \theta } ,$ , is equivalent to being able to discern the true parameter vector if the model can be observed indefinitely.

Let � be the general nonlinear regression model:

$$
y = f ( x , \theta ) + e ~ , e = \mathcal { N } \big ( 0 , \sigma ^ { 2 } v ^ { 2 } ( f ( x , \theta ) , \delta ) \big )\tag{1}
$$

Note although the error term is assumed Gaussian in nature, it is permitted to take on quite general structures incorporating phenomena such as heteroscedasticity and or serial correlation through the specification of the variance-covariance function $v ^ { 2 } ( f ( x , \theta ) , \delta )$ . Expanding $f ( x , \theta )$ as a first order Taylor series about the current estimates $\theta _ { k } \in \mathbb { R } ^ { p }$ and defining $g = y - f ( x , \theta _ { k } )$ , Equation (1) becomes:

$$
g \approx J ( x , \theta _ { k } ) \varDelta \theta + e\tag{2}
$$

Where $J ( x , \theta _ { k } ) \in \mathbb { R } ^ { N \times p }$ is the model Jacobian matrix defined as $J ( x , \theta _ { k } ) = \partial f ( x , \theta _ { k } ) / \partial \theta$ Identifiability is closely linked with sensitivity analysis $[ \mathsf { i v } , \mathsf { v } , \mathsf { v i } ]$ , and so to the properties of $J ( x , \theta )$ Given that the experimental design is adequate, if the columns of the Jacobian matrix are linearly independent then the parameter estimates can be uniquely estimated. This is equivalent to stating that the matrix $\big ( J ( x , \theta ) ^ { T } J ( x , \theta ) \big )$ is invertible.

Vajda et al [vii] investigated the quantitative identifiability of model parameters for a flow reactor employed to pyrolyze methane at three constant temperatures. Their approach employed principal component analysis [viii] to determine the number of ‘‘large’’ eigenvalues of the Fisher Information Matrix. The elements of the corresponding eigenvectors were used to suggest combinations of parameters that could be used to re-parameterize and simplify the model; the number of large eigenvalues corresponding to the number of estimable parameters in the simplified model. Simplifying the model seems at odds with the underlying physical thinking often used to formulate a Lithium-ion battery ��� model, where certain parameters may account for known ageing mechanisms. Indeed, obtaining estimates of the associated parameters to rank the relative significance of individual mechanisms may be the primary goal of the study.

Vajda’s approach was limited to small scale models, with relatively few parameters. Yao et al [iv] define an approach capable of dealing with larger scale problems. Their example contains up to 50 parameters. The algorithm developed is analogous to calculating the alias matrix in linear regression problems [ix] and relies on the fact that the residual vector is orthogonal to the expectation plain. Each column added to the simplified scaled Jacobian is thus orthogonal to any existing columns. Again, the main concern is that the resulting model is simplified, and the original physical formulation altered.

In situations where the columns of $J ( x , \theta )$ are nearly colinear, the conditioning of the Jacobian is very poor, resulting in very imprecise parameter estimates. In addition, the parameter estimates are very sensitive to the peculiarities of the training data, implying that a small perturbation in the data can lead to a large change in the parameter estimates [x]. One way of converting a problem from ill-posed to well-posed is to introduce a regularisation method. The basic idea is to confine the problem to a restricted set of parameters using a regularisation functional $\mathcal { R } ( \theta )$ . Typically, this is achieved by introducing a constraint of the form $\mathcal { R } ( \theta ) \leq c$ to the usual log likelihood function, $L ( \theta )$ , such that:

$$
a r g \operatorname* { m i n } _ { \theta } \left( L ( \theta ) + { \frac { \lambda } { 2 } } { \mathcal { R } } ( \theta ) \right)\tag{3}
$$

This is reminiscent of the method of Langrage multipliers. The regularisation parameter (�), or hyperparameter, implicitly defines a structure on the possible models by constraining the model. Roughly speaking, low values of � in Equation (3) correspond to high values of �; i.e. a weak constraint. Alternatively, a large value for the hyper-parameter, corresponding to a low value for $c ,$ implies greater importance to $\mathcal { R } ( \theta )$ in the minimisation of the cost function. Thus, a large value for � represents a more severe constraint. Hence, we see the regularised cost minimisation problem represents a trade-off between model fidelity, minimising $L ( \theta )$ , and constraining the model to be a member of a small compact subset, $\mathcal { R } ( \theta )$ . Furthermore, the balance between fitting the data and satisfying the constraint is controlled by the regularisation parameter.

In practice, minimisation of the regularised cost function is carried out iteratively using a gradient based optimisation method. In this paper we are concerned with optimal selection of the hyperparameter at each iteration of this process. In this way, the fidelity versus constraint level trade-off can be handled automatically as the iterative parameter search proceeds. This is accomplished by developing a fixed-point iteration scheme to be carried out at each iteration of the optimisation scheme. These are founded on novel re-estimation formulae presented in this paper.

The paper is organised as follows. In section 2 we introduce the information theoretic model selection measures upon which the optimal hyper-parameter estimation process is founded. Both information theoretic measures are functions of the relevant loglikelihood, and in section 3 we derive the appropriate profiled likelihood necessary for the derivation of the re-estimation formulae for the hyper-parameter. For continuity, section 4 we provide expressions for the derivatives of the profile loglikelihood and model degrees of freedom with respect to the hyper-parameter. These are necessary for the derivation of the re-estimation formulae presented in section 5 and derived in detail in Appendix B. Convergence criterion for the iterative algorithm are also given here. Section 6 details the regularised iterative generalised least squares algorithm utilised for the identification of both the fit and covariance model parameters. An example analysis of simulated data follows in section 7. The application considered is the Martinez-Laserna model for battery state of health estimation. The multiplicative structure of the model yields particularly poor identifiability and regularisation proves essential to obtain reasonable parameter estimates. In section 8, we comment on the convergence rate of the iterative process and its dependency on the initial derivative of the re-estimation function with respect to the hyper-parameter. Finally, in section 9 we state conclusions relating to the efficacy of the procedure.

## 2 Model Selection Measures

Various measures appear in the literature, which permit comparison among competing statistical models [xi, xii, xiii, xiv]. Orr [xv, xvi] provides a method for optimally estimating the hyper-parameter in regularised radial basis function network applications. His approach is based on the well-known generalised cross-validation (���) criterion [xiv]. The main justification for ��� as a model selection criterion is conveniently summarised in their ��� theorem [xiv, xvii, xviii]. As discussed by Eubank [xvii] the GCV theorem can be interpreted as stating that, in certain circumstances, GCV is nearly an unbiased estimator of the prediction risk.

Essentially, Orr presents a re-estimation method for optimally determining � as a new basis function is added to the model. By differentiating the relevant ��� criterion with respect to the hyperparameter, setting the result to zero and solving for � he derives an expression for updating the regularisation parameter as each new basis function is added to the network. As � appears on both sides of this expression, iteration is required to determine the updated value.

There are several issues associated with Orr’s approach. Firstly, it relates to ordinary least squares estimation only. Weighted fitting approaches are not accounted for. Secondly, the measure has units which means that models created in transformed scales cannot be compared. Finally, Orr’s parameter count for the model neglects the covariance parameter; $\sigma ^ { 2 }$ in the least squares case.

In contrast, we base our approach on two popular information theoretic criteria: Sugiura’s small sample Akaike Information Criterion $( A I C _ { c } )$ [xiii] and also Schwarz’s Bayesian Information Criterion [xix] (���). These are defined as:

$$
\begin{array} { r } { A I C _ { c } = 2 L ( \theta ) + \frac { 2 \gamma ( \gamma + 1 ) } { N - \gamma - 1 } } \end{array}\tag{4}
$$

$$
B I C = 2 L ( \theta ) + \gamma l o g ( N )\tag{5}
$$

Where �(�) is the negative loglikelihood, � the number of data points and � the total number of estimable parameters in the model specification. We derive formulae based on both measures and compare their respective performance in later sections. Notice the full version of both ��� and $A I C _ { c } ,$ as given in Equations (4) and (5), are based on the negative loglikelihood, which is unitless. Therefore, these measures permit comparisons among models computed in both transformed and natural unit scales. The situation is analogous to the well know Box-Cox method [xx] for determining an optimal transformation in linear regression problems. In addition, by employing the full likelihood we can account for heteroscedastic and or serially correlated data.

For either measure, the estimable parameter count necessarily includes both fit and covariance model parameters. In addition, the term estimable deserves some clarification, particularly for regularised models. Following Burnham and Anderson [xxi], by this we imply parameters that are uniquely identifiable from the data. One reason a parameter can be non-estimable is due to inherent confounding in the model. Under these circumstances, two or more parameters cannot be uniquely identified from the data. This is obvious in multiplicative structures involving products of coefficients. Such formulations are relatively common in phenomenological models for lithium-ion battery ��� estimation.

3 Approximations to the Log-Likelihood and the Ridge Estimator We begin by assuming Gaussian distributed errors, at the $k ^ { t h }$ iteration, according to $\boldsymbol { e } _ { k } =$ ${ \pmb { \mathcal { N } } } ( \mathbf { 0 } , \sigma ^ { 2 } W )$ . Where the general covariance matrix, $\sigma ^ { 2 } W$ , may incorporate heteroscedastic and or serially correlated errors. Further, let $\theta \in \mathbb { R } ^ { p }$ denote the vector of parameters to be identified and $\pmb { f } ( \pmb { \theta } )$ the model to be fitted. Under these circumstances, the corresponding negative loglikelihood is:

$$
\begin{array} { r } { L = \frac { N } { 2 } l o g ( 2 \pi ) + \frac { N } { 2 } l o g ( \sigma ^ { 2 } ) + \frac { 1 } { 2 } l o g ( \| W \| ) + \frac { 1 } { 2 \sigma ^ { 2 } } \big ( y - f ( \theta ) \big ) ^ { T } { W } ^ { - 1 } \big ( y - f ( \theta ) \big ) } \end{array}\tag{6}
$$

In general, there is no closed form solution for the estimation of � and we employ iterative methods. For the $k ^ { t h }$ iteration, we expand $\pmb { f } ( \theta )$ about $\theta _ { k }$ using a first-order Taylor series. Thus:

$$
\pmb { f } ( \theta ) { \sim } \pmb { f } ( \theta _ { k } ) + J ( \theta _ { k } ) \Delta \theta\tag{7}
$$

Where $J ( \theta _ { k } ) = \partial f ( \theta _ { k } ) / \partial \theta _ { k } \in \mathbb { R } ^ { N \times p }$ . Substituting (7) into (6), dropping the notational dependency on $\theta _ { k }$ and writing $\pmb { g } = \pmb { y } - \pmb { f } ( \theta )$

$$
\begin{array} { r } { L \sim \frac { N } { 2 } l o g ( 2 \pi ) + \frac { N } { 2 } l o g ( \sigma ^ { 2 } ) + \frac { 1 } { 2 } l o g ( \| W \| ) + \frac { 1 } { 2 \sigma ^ { 2 } } ( g - J \varDelta \theta ) ^ { T } { W } ^ { - 1 } ( g - J \varDelta \theta ) } \end{array}\tag{8}
$$

As � is real and positive definite, it may be factored as $W = C ^ { T } C$ , where � is upper triangular. Using standard results on matrix inverses [xxii], it can be shown that $W ^ { - 1 } = C ^ { - 1 } C ^ { - T }$ . Further, as � is triangular, $\begin{array} { r } { l o g ( \| W \| ) = 2 \sum _ { i = 1 } ^ { N } l o g ( c _ { i i } ) } \end{array}$ , where $c _ { i i }$ is the $i ^ { t h }$ term on the leading diagonal of �. Applying these results and writing $\pmb { q } = \pmb { C } ^ { - T } \pmb { g }$ and $Z = C ^ { - T } J .$ , yields:

$$
\begin{array} { r } { L \sim \frac { N } { 2 } l o g ( 2 \pi ) + \frac { N } { 2 } l o g ( \sigma ^ { 2 } ) + \sum _ { i = 1 } ^ { N } l o g ( c _ { i i } ) + \frac { 1 } { 2 \sigma ^ { 2 } } ( \pmb { q } - Z \Delta \pmb { \theta } ) ^ { T } ( \pmb { q } - Z \Delta \pmb { \theta } ) } \end{array}\tag{9}
$$

We now introduce a ridge regression term into (10) to form:

$$
\begin{array} { r l } & { R \sim \frac { N } { 2 } l o g ( 2 \pi ) + \frac { N } { 2 } l o g ( \sigma ^ { 2 } ) + \sum _ { i = 1 } ^ { N } l o g ( c _ { i i } ) + \frac { 1 } { 2 \sigma ^ { 2 } } ( { \boldsymbol q } - { \boldsymbol Z } { \boldsymbol A } { \boldsymbol \theta } ) ^ { T } ( { \boldsymbol q } - { \boldsymbol Z } { \boldsymbol A } { \boldsymbol \theta } ) + \frac { \lambda } { 2 \sigma ^ { 2 } } { \boldsymbol A } { \boldsymbol \theta } ^ { T } { \boldsymbol A } \theta } \end{array}\tag{10}
$$

Differentiating (11) w.r.t. ��, setting the result to zero and solving for �� yields after some algebra:

$$
\Delta \theta = ( Z ^ { T } Z + \lambda I ) ^ { - 1 } Z ^ { T } { \pmb q } = A ^ { - 1 } Z ^ { T } { \pmb q }\tag{11}
$$

Substituting (12) into (10) and factorising:

$$
\begin{array} { r } { L \sim \frac { N } { 2 } l o g ( 2 \pi ) + \frac { N } { 2 } l o g ( \sigma ^ { 2 } ) + \sum _ { i = 1 } ^ { N } l o g ( c _ { i i } ) + \frac { 1 } { 2 \sigma ^ { 2 } } \big . \big . \big . \big . \big ( I - S ( \lambda ) \big ) ^ { 2 } \pmb { q } ^ { T } \big ( I - S ( \lambda ) \big ) ^ { 2 } \big . } \end{array}\tag{12}
$$

Where, $S = Z A ^ { - 1 } Z ^ { T }$ is the so-called smoother matrix. Note � is symmetric, so $S = S ^ { T }$ . Additionally, the effective degrees of freedom associated with the model, �, is given by $\gamma = t r ( S )$ [xv, xxiii]. We use (13) as the basis for optimal re-estimation of the hyper-parameter in section 5. In addition, we can exploit the fact that the likelihood is conditionally linear in the variance scale parameter $\sigma ^ { 2 }$ Consequently, a closed form solution for the estimate of $\sigma ^ { 2 }$ exists, which can be shown to be:

$$
\begin{array} { r } { \sigma ^ { 2 } = \frac { \boldsymbol { q } ^ { T } ( I - S ) ^ { 2 } \boldsymbol { q } } { N } } \end{array}\tag{13}
$$

Substituting Equations (14) into (13) yields the corresponding profile likelihood $\left( L _ { p } \right)$ :

$$
\begin{array} { r } { L _ { p } \sim \frac { N } { 2 } l o g ( 2 \pi ) + \frac { N } { 2 } l o g \left( \frac { q ^ { T } \left( I - S ( \lambda ) \right) ^ { 2 } q } { N } \right) + \sum _ { i = 1 } ^ { N } l o g ( c _ { i i } ) + \frac { N } { 2 } } \end{array}\tag{14}
$$

Equation (14) is used as the basis for the covariance model parameters as detailed in section 6 and for optimally selecting the hyper-parameter as described in section 5.

## 4 Derivatives of the Smoother Matrix, � and the Profile Loglikelihood w.r.t. �

Both information theoretic measures involve the derivative of $L _ { p }$ and $\gamma$ with respect to �. This necessarily involves formulae for the corresponding derivatives of �, ��(�) and $S ^ { 2 }$ . These are derived in appendix A, and only the results stated here for continuity. Hence:

$$
\begin{array} { r } { \frac { \partial S } { \partial \lambda } = - Z A ^ { - 2 } Z ^ { T } } \end{array}\tag{15}
$$

$$
\begin{array} { r } { \frac { \partial S ^ { 2 } } { \partial \lambda } = 2 S \frac { \partial S } { \partial \lambda } = - 2 Z A ^ { - 2 } Z ^ { T } + 2 \lambda Z A ^ { - 3 } Z ^ { T } } \end{array}\tag{16}
$$

Similarly, recalling $\gamma = t r ( S )$ :

$$
\begin{array} { r } { \frac { \partial \gamma } { \partial \lambda } = t r ( \lambda A ( \lambda ) ^ { - 2 } - A ( \lambda ) ^ { - 1 } ) } \end{array}\tag{17}
$$

Finally, using these results, it can be shown after some matrix algebra that the derivative of the loglikelihood, Equation (13), is given by:

$$
\begin{array} { r } { \frac { \partial L _ { p } } { \partial \lambda } = \lambda \left( \frac { N } { \sigma ( \lambda ) ^ { 2 } } \right) \pmb { q } ^ { T } Z A ( \lambda ) ^ { - 3 } Z ^ { T } \pmb { q } } \end{array}\tag{18}
$$

5 Optimal Selection of the Hyper-Parameter Based on $B I C$ and $A I C _ { c }$ In appendix B we derive the following fixed-point iteration formulae based on BIC and $A I C _ { c }$

$$
\begin{array} { r } { \lambda _ { B I C } = \frac { \sigma ^ { 2 } l o g ( N ) t r \left( A ^ { - 1 } - \lambda A ^ { - 2 } \right) } { 2 N \pmb q ^ { T } Z A ^ { - 3 } Z ^ { T } \pmb q } } \end{array}\tag{19}
$$

$$
\begin{array} { r } { \lambda _ { A I C _ { c } } = \frac { \sigma ^ { 2 } \left[ \frac { \left( 2 \gamma + 1 \right) } { \left( N - \gamma - 1 \right) } - \frac { \gamma \left( \gamma + 1 \right) } { \left( N - \gamma - 1 \right) ^ { 2 } } \right] t r \left( A ^ { - 1 } - \lambda A ^ { - 2 } \right) } { N q ^ { T } Z A ^ { - 3 } Z ^ { T } q } } \end{array}\tag{20}
$$

Notice in Equations (19) and (20), for notational compactness, we drop the dependency of $\sigma , \gamma$ and � on �. Regardless, � appears on both sides of either expression. Consequently, we must resort to iterative methods to estimate the optimal �. Furthermore, we use $h ( \lambda )$ to denote the righthand side of (19) or (20). These represent a fixed-point iteration scheme [xxiv], with both equations in the form $h ( \lambda ) = \lambda . \mathsf { A }$ fixed-point is defined by:

Definition: If $h ( \lambda )$ is defined on $[ a , b ]$ and $h ( \lambda ) = \lambda$ for some $\lambda \in [ a , b ]$ , then the function $h ( \lambda )$ is said to have the fixed-point � in [�, �].

The following theorem guarantees convergence of the algorithm:

Theorem: Let $h ( \lambda )$ be continuous in $[ a , b ]$ , and suppose that $h ( \lambda ) \in [ a , b ]$ for all $\lambda \in [ a , b ]$ . Further Let $h ^ { \prime } ( \lambda )$ exist on $( a , b )$ with $| h ^ { \prime } ( \lambda ) | \leq k < 1$ , for all $\lambda \in ( a , b )$ . If $\lambda _ { o } \in [ a , b ]$ , then the sequence defined by $\lambda _ { r } = h ( \lambda _ { r - 1 } ) , r \geq 1$ , will converge to the unique fixed-point � in [�, �].

The necessary proof can be found in the reference cited. Further it follows that if the theorem is satisfied then a bound for the error involved in using � to approximate � is given by:

$$
| \lambda _ { r } - \lambda | \leq k ^ { r } m a x \{ \lambda _ { o } - a , b - \lambda _ { o } \} , \forall r \geq 1 .\tag{21}
$$

The theorem requires $h ^ { \prime } ( \lambda )$ , which we calculate numerically using the 5-point stencil approximation provided in [xxv]. This is accurate to ${ \cal O } ( \Delta \lambda ^ { 4 } )$ . Thus, the fixed-point iteration algorithm is:

1. Generate 6 logarithmically spaced values in the interval $\lambda \in [ 1 0 ^ { - 5 } 1 0 ^ { 0 } ]$ and calculate the corresponding $\vert d h ( \lambda _ { 0 } ) / d \lambda \vert$ . Select the � for which the following conditions are met:

a. $m i n \{ | d h ( \lambda _ { 0 } ) / d \lambda | \} < 1$

b. $h ( \lambda _ { 0 } ) \in [ 1 0 ^ { - 5 } 1 0 ^ { 0 } ]$

c. Denote this value by $\lambda _ { r }$ . Set $r = 0$

2. Increment $r = r + 1$

3. Calculate $\lambda _ { r + 1 }$ using either Equation (19) or (20) as desired.

4. If $( | \lambda _ { r + 1 } - \lambda _ { r } | / \lambda _ { r } \leq 0 . 0 0 0 1 ) o r ( i \geq i _ { m a x } ) \to s t o p$ . Else treating $\lambda _ { r + 1 }$ as $\lambda _ { r }$ return to step 2.

Having provided the re-estimation formulae and associated algorithm for optimal hyper-parameter selection, we now detail the full nonlinear regularised iterative generalised least squares (�����) algorithm, with optimal � selection at each iteration.

## 6 A ����� Algorithm with Optimal Hyper-Parameter Selection

Davidian and Giltinan [xxvi] provide a rationale for preferring iterative generalised least squares over joint maximum likelihood. Their argument is based on the improved robustness to both model misspecification and non-normality. Here, we extend their basic algorithm to the regularised case and for the profiled loglikelihood. The regularised iterative generalised least squares algorithm employed is as follows:

1. Obtain the initial estimates, $\theta _ { r } ,$ by minimising the following nonlinear functional with respect to �. Re-estimate λ at each iteration, by iterating equation (19) or (20) to convergence as required.

$$
\bullet \quad a r g \operatorname* { m i n } _ { \theta } R = { \textstyle { \frac { 1 } { 2 } } } { \big ( } y - f ( \theta ) { \big ) } ^ { T } { \big ( } y - f ( \theta ) { \big ) } + { \textstyle { \frac { \lambda } { 2 } } } \theta ^ { T } \theta
$$

2. Using $\theta _ { r }$ , form the residuals, $d _ { r } = y - f ( \theta _ { r } )$ . Using these, minimise the following functional with respect to �. Denote these estimates as $\delta _ { r + 1 }$

$$
\begin{array} { r l } { \bullet } & { { } a r g \operatorname* { m i n } _ { \delta } \left( L _ { p } = \frac { N } { 2 } l o g \left( \frac { d _ { r } ^ { T } W ^ { - 1 } ( \theta , \delta ) d _ { r } } { N } \right) + \frac { 1 } { 2 } l o g \| W ( \theta , \delta ) \| + \frac { N } { 2 } \ \right) } \end{array}
$$

3. Using $\delta _ { r + 1 }$ construct the matrix $W ^ { - 1 } ( \theta _ { r } , \delta _ { r + 1 } )$ . Update estimates for � by minimising the nonlinear function:

$$
\bullet \quad a r g \operatorname* { m i n } _ { \theta } R = \{ \frac { 1 } { 2 } \big ( y - f ( \theta ) \big ) ^ { T } W ^ { - 1 } ( \theta _ { r } , \delta _ { r + 1 } ) \big ( y - f ( \theta ) \big ) + \frac { \lambda } { 2 } \theta ^ { T } \theta \}
$$

Re-estimate λ at each iteration, by iterating equation (20) or (21) to convergence as required.

Denote the values at convergence by $\theta _ { r + 1 }$

4. Compute $\begin{array} { r } { \exists = 1 - \frac { \Vert \theta _ { r + 1 } \Vert _ { r + 1 } ^ { 2 } } { \Vert \theta _ { r } \Vert _ { r } ^ { 2 } } . \Vert \mathbf { f } \supseteq 0 . 0 0 0 1 } \end{array}$ , then treating $\theta _ { r + 1 }$ as $\theta _ { r }$ return to step 2, else stop.

$$
\begin{array} { r l } { \bullet } & { { } \mathsf { C o m p u t e } \sigma ^ { 2 } = \frac { d _ { r } ^ { T } W ^ { - 1 } ( \theta , \delta ) d _ { r } } { N } . } \end{array}
$$

The algorithm has been implemented in MATLAB and utilises the fmincon function from the MATLAB optimisation toolbox. To work as intended, it is necessary to supply analytical gradients to the fmincon routine, so that the hyper-parameter is calculated only once per iteration, as opposed to �-times if finite differences are utilised. Using standard results on matrix derivatives [xxii], it is shown in Appendix C that:

$$
\begin{array} { r } { \frac { \partial R } { \partial \theta } = J ^ { T } ( \theta ) W ^ { - 1 } ( f ( x , \theta ) - y ) + \lambda \theta } \end{array}\tag{22}
$$

Note, as required at step 1, $W = I \in \mathbb { R } ^ { N \times N }$ . The authors have constructed a MATLAB package to implement the algorithm, which makes use of the strategy and template object oriented behavioural patterns defined by Gamma et al [xxvii].

## 7 Example Analysis

For example, consider the model proposed by Martinez-Laserna et al [i], which attempts to predict capacity fade, $Q _ { l o s s }$

$$
\begin{array} { r } { Q _ { l o s s } = \omega . e x p \left( \frac { \beta _ { 1 } } { T } + \beta _ { 2 } \overline { { S O C } } \right) A h ^ { z } } \end{array}\tag{23}
$$

With �ℎ the accumulated charge throughput, � the cell absolute thermodynamic temperature and ���<sup>̅̅̅̅̅̅</sup> the average state of charge. The parameter vector, $\theta = ( \omega , \beta _ { 1 } , \beta _ { 2 } , z )$ , is to be estimated from the data. The multiplicative structure is common in such models for this application and results in extremely poor identifiability. The Martinez-Laserna model has been fitted to simulated data representing a cyclic ageing profile for a single cell. The advantage of utilising simulated data is estimates can be compared to the known true values, thus demonstrating the efficacy of the algorithms employed. The true model is:

$$
\begin{array} { r } { Q _ { l o s s } = 1 0 0 \times 0 . 4 . e x p \left( \frac { 0 . 0 9 } { \left( T + 2 7 3 . 1 5 \right) } + 0 . 0 2 . S O C \right) . A h ^ { 0 . 5 } } \end{array}\tag{24}
$$

In addition, the simulated data is contaminated with Gaussian heteroscedastic noise of the form $\mathcal { N } ( 0 , \sigma ^ { 2 } . Q _ { l o s s } ^ { 2 \delta } )$ with $\sigma = 0 . 0 5$ and $\delta = 1$ . Again, these quantities are to be estimated from the data. Fifty-one equally spaced points were generated for the interval $A h \in [ 0 , 5 0 ]$ . During fitting, both �ℎ and $Q _ { l o s s }$ are mapped onto the interval [0, 1].

Figure 1 presents the usual diagnostic plots for a regularised ordinary least squares analysis. The hyper-parameter is optimised using the algorithm detailed in section 5 and is based on the $A I C _ { c }$ performance measure. This is equivalent to performing step 1 only of the ����� algorithm of section

6. We refer to this as the regularised ordinary least squares (����) model. The heterogenous nature of the variance is apparent from the residual versus predicted plot, which exhibits a distinct increase in the variance scale with increasing predicted $Q _ { l o s s }$ . The optimal value of the hyper-parameter at convergence is 0.0074.

![](images/2bf7280eb3e3f89bc74fd68a7e5c48c22b79f5d4289ed13af6cbbc2ce981bb8a.jpg)

![](images/8933813bae138d5583e1ba78e460727a0db042c636d42a1bc950b0ecad6e5ebb.jpg)

![](images/7e872a5346443ec9cdd3a67320c9f529fa48fc16eba08c3170c0a6c6565cdbe4.jpg)

![](images/1e34934fa7efff686353b23d71f86ab0f429837d02b70aeb427710c1d271cb73.jpg)  
Figure 1: ROLS analysis for the SoH data. Hyper-parameter optimised using $A I C _ { c }$ formula. The residual versus prediction diagnostic plot (top right) illustrates strong evidence for the presence of heterogeneous variance. There is one outlying observation circled in red. Note, both the independent and dependent data have been mapped to the interval [0, 1]. The optimal value of the hyper-parameter is 0.0074.

Figure 2 presents diagnostic plots for the equivalent ����� fit to the same data. The applied covariance model is of the form $\sigma ^ { 2 } f ( x , \theta ) ^ { 2 \delta }$ . The improvement in the appearance of the weighted residual versus predicted plot is immediately apparent. The estimated value for � is 0.973, which is reassuringly close to the true value of unity. Further, the optimal value of the hyper-parameter is 0.0190. This is over 3 times larger than the ���� value. The increased magnitude implies greater bias and yet there is no obvious evidence of this in the diagnostic plots.

![](images/8615f8a6341f08f00ba7acecb7a4b83e9e3eb301aa9394650e79c1a27eafc48b.jpg)

![](images/3649fba07f0e32a87c8fff1251ade418fe49ff740a9af2caf51ed5fd4cea9ef9.jpg)

![](images/4c97d970f08bf147adf186a0997017b503edf32e5bf7a0488c6de7f4497875f9.jpg)

![](images/0afacc8e3341e1cc8785adbae9738f33ffab4bee8e6c0ab2a41e086b5f821827.jpg)  
Figure 2: RIGLS fit to the data. Hyper-parameter optimised using $A I C _ { c }$ formula. The appearance of the weighted residual versus prediction plot is much improved via application of the appropriate covariance model. Both the independent and dependent data have been mapped to the interval [0, 1]. The optimal value of the hyper-parameter is 0.019.

Finally, Figure 3 presents the diagnostic plots for the ����� algorithm, but in this instance optimal hyper-parameter selection is based on ���. These are almost identical in appearance to Figure 2 for the $A I C _ { c }$ optimal λ-selection case. Here at convergence � = 0.0456.

![](images/d0e479d06353df67a695f9b8561608d2ce39750a89a9e80223233f9f8b208ae6.jpg)

![](images/968e82c74be48e65f6e194f52486676420f653fb27ce70787775c6618fe598e3.jpg)

![](images/eb6bc768e47fa7c391e05051d3b301714a93831dfc3264a00beffc33f59a6389.jpg)

![](images/e861d81b75d75940ed871b800d2507b0d1204fdf143ad12104b32aebd3b86809.jpg)  
Figure 3: RIGLS fit to the data. Hyper-parameter optimised using the BICformula. The appearance of the weighted residual versus prediction plot is much improved via application of the appropriate covariance model Both the independent and dependent data have been mapped to the interval [0, 1]. The optimal value of the hyper-parameter is 0.0456.

To gain some insight into the impact of the hyper-parameter on the estimates, we calculate the condition numbers [xxviii], $\mu _ { 0 }$ and $\mu _ { 1 } ,$ for the matrices $F _ { 0 } = ( J ^ { T } J )$ and $F _ { 1 } = ( J ^ { T } J + \lambda I )$ respectively for all 3 cases. These are presented in Table 1. Recall, the inverse of $F _ { 0 }$ and $F _ { 1 }$ are the respective covariance matrices for the ���� or ����� estimates as appropriate.

Table 1: Condition numbers, optimal hyper-parameter value and degrees of freedom for the various estimates. The ideal condition number is unity. Condition numbers approaching infinity indicate rank degeneracy among the columns/rows of the corresponding matrix.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ROLS</td><td rowspan=1 colspan=1>RIGLS AICC</td><td rowspan=1 colspan=1>RIGLS BIC</td></tr><tr><td rowspan=1 colspan=1>λ</td><td rowspan=1 colspan=1>0.0060</td><td rowspan=1 colspan=1>0.0190</td><td rowspan=1 colspan=1>0.0456</td></tr><tr><td rowspan=1 colspan=1> $\mu _ { o }$ </td><td rowspan=1 colspan=1>3.0853E+19</td><td rowspan=1 colspan=1>3.9489E+18</td><td rowspan=1 colspan=1>2.9766E+18</td></tr><tr><td rowspan=1 colspan=1> $\underline { { \pmb { \mu } _ { 1 } } }$ </td><td rowspan=1 colspan=1>2622.6000</td><td rowspan=1 colspan=1>119.3400</td><td rowspan=1 colspan=1>630.7300</td></tr><tr><td rowspan=1 colspan=1>γ</td><td rowspan=1 colspan=1>1.9854</td><td rowspan=1 colspan=1>1.9906</td><td rowspan=1 colspan=1>1.9931</td></tr></table>

Regardless of the source, the values of $\mu _ { o }$ are extremely large, indicating that $F _ { 0 }$ is nearly rank degenerate in all cases. This demonstrates the very poor identifiability of the model, and the ill-posed nature of the unregularized fit. In contrast, despite the relatively small magnitude of the corresponding optimal hyper-parameter values, the associated $\mu _ { 1 }$ numbers are vastly reduced, with the $A I C _ { c }$ based value exhibiting some superiority. Table 1 also presents the corresponding model degrees of freedom, �. In all cases, despite the relatively small associated hyper-parameter values, the degree of freedom reduction is substantial; from 4 to just under 2.

Table 2 presents parameter estimates and 95% confidence intervals for ����� estimates. The parameter estimates for the $A I C _ { c }$ and ��� are reasonably close to the true values, indicating the induced bias from the hyper-parameter is not large. However, it does though have some practical significance. For example, for the ��� case, the $\beta _ { 1 }$ estimate has the wrong sign. In a parametric regression study of this nature, where quasi-physical thinking is used to develop the model formulation, incorrect estimation of the sign of the coefficient may lead to inappropriate conclusions regarding the nature and interpretation of the underlying true ageing mechanisms.

Also shown in Table 2 are the 95% confidence intervals for the parameter estimates. Given the inherent poor identifiability of the model structure, the confidence intervals are relatively small, particularly for $\beta _ { 1 }$ . Notice the confidence intervals for � and � encompass the true estimate, whereas the intervals for $\beta _ { 1 }$ and $\beta _ { 2 }$ do not. This may be symptomatic of the inherent bias induced in regularised estimates, despite the relatively small values of the hyper-parameter. Also, it is interesting to note, despite the larger hyper-parameter value associated with the ��� ����� estimator, the corresponding confidence intervals are larger. This follows from the larger condition number for the regularised $F _ { 1 }$ matrix, resulting in variance inflation of the estimates, demonstrating that the confidence intervals depend on the parameter estimates as well as the size of the hyper-parameter.

Table 2: Comparison among estimators, parameter values and 95% confidence intervals for the RIGLS estimates.
<table><tr><td rowspan=2 colspan=2></td><td rowspan=1 colspan=3>RIGLS AICc</td><td rowspan=1 colspan=3>RIGLS BIC</td></tr><tr><td rowspan=1 colspan=1>True</td><td rowspan=1 colspan=1>LCI</td><td rowspan=1 colspan=1>Estimate</td><td rowspan=1 colspan=1>UCI</td><td rowspan=1 colspan=1>LCI</td><td rowspan=1 colspan=1>Estimate</td><td rowspan=1 colspan=1>UCI</td></tr><tr><td rowspan=1 colspan=1> $\underline { { \boldsymbol { \omega } } }$ </td><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.392</td><td rowspan=1 colspan=1>0.403</td><td rowspan=1 colspan=1>0.415</td><td rowspan=1 colspan=1>0.308</td><td rowspan=1 colspan=1>0.410</td><td rowspan=1 colspan=1>0.513</td></tr><tr><td rowspan=1 colspan=1> $\beta _ { 1 }$ </td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.017</td><td rowspan=1 colspan=1>0.017</td><td rowspan=1 colspan=1>0.017</td><td rowspan=1 colspan=1>-0.088</td><td rowspan=1 colspan=1>-0.088</td><td rowspan=1 colspan=1>-0.088</td></tr><tr><td rowspan=1 colspan=1> $\beta _ { 2 }$ </td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.054</td><td rowspan=1 colspan=1>0.056</td><td rowspan=1 colspan=1>0.058</td><td rowspan=1 colspan=1>0.088</td><td rowspan=1 colspan=1>0.105</td><td rowspan=1 colspan=1>0.123</td></tr><tr><td rowspan=1 colspan=1> $\vartheta$ </td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.491</td><td rowspan=1 colspan=1>0.507</td><td rowspan=1 colspan=1>0.523</td><td rowspan=1 colspan=1>0.380</td><td rowspan=1 colspan=1>0.536</td><td rowspan=1 colspan=1>0.692</td></tr></table>

## 8 Convergence Analysis of the Optimal Hyper-Parameter Selection Algorithm

Figure 4 presents the output from the first iteration of the algorithm of section 5 versus starting value on the left hand axis, and $\vert d h ( \lambda _ { 0 } ) / d \lambda \vert$ at the same starting value on the righthand axis. For $\lambda \le 0 . 0 0 1$ it appears the algorithm can return large negative numbers and appear wildly unstable in places. This is supported by the theorem of section $^ { 5 , }$ which states that convergence to a unique fixed point requires $| h ^ { \prime } ( \lambda ) | < 1$ in an interval. From the figure, $| h ^ { \prime } ( \lambda ) | \gg 1$ for all $\lambda _ { 0 } < 0 . 0 0 1$ . For $\lambda \in \left[ 0 . 0 1 , 1 0 \right] _ { \scriptscriptstyle }$ $| h ^ { \prime } ( \lambda ) | \ll 1$ and rapid convergence of the resulting sequence would be expected. In terms of the number of iterations required, using (22), with $k = 0 . 2 0$ and $\lambda _ { 0 } = 0 . 1 5$ , for the interval $\lambda \in [ 0 . 1 , 1 ]$ for an error bound of 0.00001:

$$
\begin{array} { r } { r = \frac { l o g ( 0 . 0 0 0 0 1 ) - l o g ( m a x \{ 0 . 0 5 , 0 . 8 5 \} ) } { l o g ( 0 . 2 ) } = 7 . 0 5 2 4 } \end{array}
$$

![](images/cdb2c80bc19b73f93fd07112947bed7e85e79ee15e14aaf0bce561945afc9add.jpg)  
Figure 4: Convergence Analysis. This plot demonstrates that convergence of the optimal selection algorithm depends on starting position. The dotted line denotes the value of $| d h ( \lambda _ { 0 } )$ /��| below which convergence would be expected.

![](images/5885c15d640a8ddd9bd217b98ec8060575941b4f389b994c7d5fc550d87359ac.jpg)  
Figure 5: Expected number of iterations for convergence of the hyper-parameter value with increasing k. In all cases, � ∈ [0.1,1.0] and $\lambda _ { o } = 0 . 1 5$ and $| \lambda _ { r } - \lambda | = 0 . 0 0 0 0 1$

Hence around 7-8 iterations are required for convergence to the desired error bound. Consider Figure 5, which presents the expected number of iterations of the optimal hyper-parameter algorithm for

$k \in [ 0 . 2 , 0 . 9 ]$ , again with $\lambda _ { 0 } = 0 . 1 5$ and $\lambda \in [ 0 . 1 , 1 ] .$ , again for an error bound of 0.00001. As expected, the number of iterations required for convergence grows exponentially with increasing �.

## 9 Conclusions

The optimal hyper-parameter estimation method presented, founded on information theoretic measures, is efficient and converges rapidly given a suitable starting position. Good starting values are any for which $\vert d h ( \lambda _ { 0 } ) / d \lambda \vert$ is less than unity. However, the smaller the initial $\vert d h ( \lambda _ { 0 } ) / d \lambda \vert$ , the more rapid the convergence. The fixed-point iterative formulae presented are suitable for general covariance structures, including heteroscedastic and or serially correlated errors. When implemented as part of a general ����� algorithm scheme, simulation suggests it is possible to preserve quasiphysically motivated structures and return estimates with low bias.

For the simulation study presented, the algorithm performs adequately. ����� employing $A I C _ { c }$ as a basis for optimal hyper-parameter estimation marginally outperforms its ��� counterpart in the simulation study presented. Confidence intervals for parameters are relatively narrow, demonstrating the improved precision of the estimates. This follows from the vastly superior condition numbers for the relevant information matrices. However, the simulation study demonstrates that not all confidence intervals reported include the true values – an indication of the bias inherent in the regularised approach.

## Appendix A – Derivatives of the Smoother Matrix and its Trace

Here, we provide detailed derivations of the key results, equations (16) through (18). All these equations utilise standard results on matrix derivatives [xxii]. To begin with, the derivative of � with respect to � is straightforward:

We begin by noting:

$$
\begin{array} { l } { A = Z ^ { T } Z + \lambda I } \\ { \Rightarrow Z ^ { T } Z = A - \lambda I } \\ { \Rightarrow \frac { \partial A } { \partial \lambda } = I } \end{array}\tag{A1}
$$

Hence:

$$
\begin{array} { r } { \frac { \partial S } { \partial \lambda } = Z \frac { \partial A ^ { - 1 } } { \partial \lambda } Z ^ { T } = - Z A ^ { - 1 } \frac { \partial A } { \partial \lambda } A ^ { - 1 } Z ^ { T } = - Z A ^ { - 2 } Z ^ { T } } \end{array}\tag{A2}
$$

Similarly:

$$
{ \frac { \partial S ^ { 2 } } { \partial \lambda } } = { \frac { \partial } { \partial \lambda } } [ Z A ^ { - 1 } Z ^ { T } Z A ^ { - 1 } Z ^ { T } ] = { \frac { \partial } { \partial \lambda } } [ Z A ^ { - 1 } ( A - \lambda I { \bf \nabla } ) A ^ { - 1 } Z ^ { T } ] = - 2 Z A ^ { - 2 } Z ^ { T } + 2 \lambda Z A ^ { - 3 } Z ^ { T }\tag{A3}
$$

Finally, $\frac { \partial \gamma } { \partial \lambda }$ is derived using:

$$
{ \frac { \partial \gamma } { \partial \lambda } } = { \frac { \partial } { \partial \lambda } } t r ( Z A ^ { - 1 } Z ^ { T } ) = { \frac { \partial } { \partial \lambda } } t r ( A ^ { - 1 } Z ^ { T } Z ) = { \frac { \partial } { \partial \lambda } } t r { \big ( } A ^ { - 1 } ( A - \lambda I ) { \big ) } = t r ( \lambda A ( \lambda ) ^ { - 2 } - A ( \lambda ) ^ { - 1 } )\tag{A4}
$$

Appendix B – Fixed Point Iteration Formulae for Hyper-Parameter Estimation

We begin by writing the ��� definition in terms of the profile likelihood (13).

$$
B I C = 2 L _ { p } + \gamma l o g ( N )\tag{B1}
$$

Differentiating this with respect to � yields:

$$
\begin{array} { r } { \frac { \partial B I C } { \partial \lambda } = 2 \frac { \partial L _ { p } } { \partial \lambda } + l o g ( N ) \frac { \partial \gamma } { \partial \lambda } } \end{array}\tag{B2}
$$

Substituting (18) and (19) into (B2):

$$
\begin{array} { r } { \frac { \partial B I C } { \partial \lambda } = 2 \lambda \left( \frac { N } { \sigma ( \lambda ) ^ { 2 } } \right) \pmb { q } ^ { T } Z A ( \lambda ) ^ { - 3 } Z ^ { T } \pmb { q } + l o g ( N ) t r ( \lambda A ( \lambda ) ^ { - 2 } - A ( \lambda ) ^ { - 1 } ) } \end{array}\tag{B3}
$$

Setting (B3) to 0 and solving for � gives the desired result:

$$
\begin{array} { r } { \lambda _ { B I C } = \frac { \sigma ^ { 2 } l o g ( N ) t r \left( A ^ { - 1 } - \lambda A ^ { - 2 } \right) } { 2 N \pmb q ^ { T } Z A ^ { - 3 } Z ^ { T } \pmb q } } \end{array}\tag{B4}
$$

Similarly, writing $A I C _ { c }$ in terms of the profile likelihood:

$$
\begin{array} { r } { A I C _ { c } = 2 L _ { p } + \frac { 2 \gamma ( \gamma + 1 ) } { N - \gamma - 1 } } \end{array}\tag{B5}
$$

Differentiating (B5) with respect to � and yields:

$$
\begin{array} { r } { \frac { \partial A I C _ { c } } { \partial \lambda } = 2 \frac { \partial L _ { p } } { \partial \lambda } + 2 \left[ \frac { ( 2 \gamma + 1 ) } { ( N - \gamma - 1 ) } - \frac { \gamma ( \gamma + 1 ) } { ( N - \gamma - 1 ) ^ { 2 } } \right] \frac { \partial \gamma } { \partial \lambda } } \end{array}\tag{B6}
$$

Again, setting (B6) to 0 and solving for � yields the desired result:

$$
\begin{array} { r } { \lambda _ { A I C _ { c } } = \frac { \sigma ^ { 2 } \left[ \frac { \left( 2 \gamma + 1 \right) } { \left( N - \gamma - 1 \right) } - \frac { \gamma \left( \gamma + 1 \right) } { \left( N - \gamma - 1 \right) ^ { 2 } } \right] t r \left( A ^ { - 1 } - \lambda A ^ { - 2 } \right) } { N q ^ { T } Z A ^ { - 3 } Z ^ { T } q } } \end{array}\tag{B7}
$$

Appendix C – Derivative of the Regularised Cost Function The appropriate regularised cost function for ridge regression is:

$$
\begin{array} { r } { R = \frac { 1 } { 2 } \big ( y - f ( \theta ) \big ) ^ { T } W ^ { - 1 } ( \theta _ { r } , \delta _ { r + 1 } ) \big ( y - f ( \theta ) \big ) + \frac { \lambda } { 2 } \theta ^ { T } \theta } \end{array}\tag{C1}
$$

Expanding the first term on the righthand side of (C1):

$$
\begin{array} { r } { R = \frac { 1 } { 2 } \big ( y ^ { T } W ^ { - 1 } y - y ^ { T } W ^ { - 1 } f ( \theta ) - f ( \theta ) ^ { T } W ^ { - 1 } y + f ( \theta ) ^ { T } W ^ { - 1 } f ( \theta ) \big ) + \frac { \lambda } { 2 } \theta ^ { T } \theta } \end{array}\tag{C2}
$$

Differentiating (C2) with respect to $\theta$ and collecting terms yields the desired result:

$$
\begin{array} { r } { \frac { \partial R } { \partial \theta } = J ^ { T } ( \theta ) W ^ { - 1 } ( f ( x , \theta ) - y ) + \lambda \theta } \end{array}\tag{C3}
$$

## References

E. Martinez-Laserna, V. I. Herrera, I. Gandiaga, A. Milo, E. Sarasketa-Zabala, H. Gaztanaga, Li\_ion Battery Lifetime Model’s Influence on the Economic Assessment of a Hybrid Electric Bus’s Operation, World Electric Vehicle Journal, 2018, 9, 28.

ii G. Suri, S. Onori, (2016), A control-oriented cycle-life model for hybrid electric vehicle lithium-ion batteries, Energy, 96, 644-653.

iii K. Smith, J. Neubauer, E. Wood, M. Jun, A. Pesaran, Models for Battery Reliability and Lifetime, Applications in Design and Health Management, Advanced Vehicles and Fuels Research, NREL/PR-5400- 58550, Battery Congress, Ann Arbor, MI, Apr 2013.

iv Yao, K. Z., Shaw, B. M., Kou, B., McAuley, K. B., & Bacon, D. W. (2003). Modeling ethylene/butene copolymerization with multi‐site catalysts: parameter estimability and experimental design. Polymer Reaction Engineering, 11(3), 563-588.

v Beck, J. V., Arnold, K. J. (1977). Parameter Estimation in Engineering and Science. Toronto, Canada: John Wiley & Sons.

vi Walter, E. (1982). Identifiability of State Space Models with Applications to Transformation Systems. New York, USA: Springer-Verlag.

vii Vajda, S., Rabitz, H., Walter, E., Lecourtier, Y. (1989). Qualitative and quantitative identifiability analysis of non-linear chemical kinetic models. Chem. Eng. Commun. 83(2):191–219.

viii Jolliffe, I. T., Principal Components Analysis, 2002, Springer Series in Statistics: Springer-Verlag

ix Box, G. E. P., Draper, N. R., Empirical Model-Building and Response Surfaces, John Wiley & Sons, First Edition, 1987.

x Goutte, C., Statistical Learning and Regularisation for Regression – Application to system identification and time series modelling, Ph.D. Thesis, L'Universite' Paris, 1996.

xi Mallows, C., Some Comments on Cp, Technometrics, 1973, Vol. 15, pp. 661 – 675.

xii Allen, D. M., The Prediction Sum of Squares as a Criterion for Selecting Variables, University of Kentucky Technical Report No. 23, 1971.

xiii Sugiura, N., Further analysis of the data by Akaike’s information criterion and the finite corrections, Communications in Statistics, Theory and Methods, 1978, A7, 13-26.

xiv Craven, P., Wahba, G., Smoothing noisy data with spline functions: estimating the correct degree of smoothing by the method of generalized cross-validation, Numerische Mathematik, 1979, Vol. 31, pp 377- 403.

xv Orr, M. J. L., Regularisation in the Selection of Radial Basis Function Centres, Neural Computation, 1995, Vol. 17, pp. 606-623.

xvi Orr, M. J. L., Introduction to Radial Basis Function Networks, Technical Report, Centre for Cognitive Science, University of Edinburgh, 1996.

xvii Eubank, R. L., Nonparametric Regression and Spline Smoothing, Marcel Dekker, Second Edition, 1999.

xviii Golub, G., Heath, M., Wahba, G., Generalized cross-validation as a method for choosing a good ridge parameter, Technometrics, 1979, Vol. 21, pp 215-223.

xix G. Schwarz, (1978), Estimating the dimension of a model, Annals of Statistics. 6, 461-464.

xx Box G. E. P., Cox, D. R., An Analysis of Transformations, Journal of the Royal Statistical Society, 1964, Vol. B26, pp 211-252.

xxi K. P. Burnham, D. R. Anderson, Model Selection and Multimodel Inference, 2002, 2<sup>nd</sup> Edition, Springer-Verlag.

xxii Searle, S. R., Matrix Algebra Useful for Statistics, John Wiley & Sons, First Edition, 1982.

xxiii Cary, M., A Model Based Methodology for the Calibration of a Port Fuel Injection, Spark-Ignition Engine, PhD Thesis, University of Bradford, 2003.

xxiv Burden, R. L., Faires, J. D., Reynolds, A. C., Numerical Analysis, Wadsworth International Student Edition, Second Edition, 1981.

xxv Abramowitz M, Stegun I, (1972), Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Table, Dover Publications

xxvi Davidian, M., Giltinan, D. M., Nonlinear Models for Repeated Measurement Data, Chapman & Hall, First Edition, 1995.

xxvii Gamma, E., Helm, R., Johnson, R., Vlissides, J., Design Patterns, Elements of Reusable Object-Oriented Software, Addison – Wesley, 1995.

xxviii Gill, P. E., Murray, W., Wright, M. H., Practical Optimization, Academic Press, First Edition, 1981.