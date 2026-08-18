# Asymptotics-guided learning and symbolic regression for dispersive resonances

Konstantinos Alexopoulos<sup>∗</sup> Josselin Garnier <sup>†</sup>

## Abstract

We study resonance prediction in dispersive media, formulated as nonlinear spectral problems for volume integral operators. The main idea is to use asymptotic analysis not only as a baseline approximation, but also as a guide for constructing predictive correction models. We learn the residual between asymptotic and reference resonances using features suggested by the subwavelength expansion, including the logarithmic scales specific to two dimensions. The resulting corrections substantially improve single-resonator and dimer predictions, and symbolic regression produces compact formulas for the learned residual. The results show that asymptotic analysis can be used not only to approximate resonances, but also to design the feature space in which data-driven corrections become accurate, low-dimensional, and interpretable.

## 1 Introduction

Resonance phenomena in wave propagation play a central role in a wide range of applications, including photonics, metamaterials, and electromagnetic device design. In dispersive media, the computation of resonant frequencies leads to nonlinear spectral problems associated with volume integral formulations, typically derived from the Lippmann–Schwinger equation [4]. These problems are particularly important in the analysis of subwavelength resonators and highly dispersive systems, where the material parameters depend strongly on frequency.

A prominent example arises in the study of halide perovskite resonators [1], whose dispersive permittivity [15] enables strong light-matter interaction and supports a wide range of applications, including optical sensing [10], photovoltaics [20], and light-emitting devices [24]. More broadly, resonant structures underpin many phenomena in metamaterials, such as negative efective parameters [17, 23], cloaking [7, 16], and bio-inspired optical efects [10, 13, 21, 27]. Accurate and eficient prediction of resonances is therefore essential for both mathematical analysis and design.

From a mathematical perspective, asymptotic analysis provides a powerful framework for approximating resonances in subwavelength regimes. Expansions derived from the spectral properties of volume integral operators yield explicit formulas that capture the leading-order behaviour of resonant frequencies [2, 3]. These approaches have been successfully applied to both single resonators and coupled systems, where interaction efects lead to mode splitting and branch-dependent resonance behaviour. However, their accuracy deteriorates away from the strict asymptotic regime, and higher-order corrections are often dificult to compute explicitly, especially in the presence of coupling or complex geometries.

In parallel, machine learning techniques have become important tools in the physical sciences [5], with applications ranging from operator learning and partial diferential equations [22] to inverse design in photonics [8, 11, 14, 25, 26]. In nanophotonics, deep learning has been widely used for the design and optimization of structures with prescribed optical properties. However, many such approaches learn the full input-output map directly from data and rely on high-capacity models, which may require large datasets and can be dificult to interpret.

In this work, we take a diferent approach. Rather than replacing the asymptotic model, we use it as the leading-order approximation and learn only the residual error between this approximation and accurate reference resonances. The learning task is therefore not to rediscover the full resonance map, but to correct the missing higher-order terms. This residual is modelled using physics-informed features derived from the asymptotic expansion, logarithmic frequency-dependent scales suggested by the two-dimensional subwavelength expansions. This leads to a low-dimensional correction model that remains tied to the underlying mathematical structure.

A second objective of the paper is to obtain interpretable correction formulas. For this purpose, we use symbolic regression, which searches for explicit analytic expressions fitting the data. Symbolic regression has recently received significant attention as a tool for interpretable scientific machine learning, notably through PySR and SymbolicRegression.jl [6]. It has also begun to appear in photonics and optical modelling, including applications to topological photonic bands [9], optical responses of biological and bio-inspired structures [18, 19], and broader discovery and optimization problems in physics and topological photonics [12]. Here, symbolic regression is used not as a black-box discovery method, but as an asymptotically guided tool for extracting compact formulas for the learned residual.

The main contributions of this work are as follows. First, we introduce an asymptotics-guided residual learning framework for nonlinear resonance problems in dispersive media, in which machine learning is used to correct, rather than replace, asymptotic or reduced-order models. Second, we construct physics-informed features based on the known structure of the subwavelength expansion, including logarithmic next-order terms, and show through ablation studies that these features are essential for accurate correction. Third, we use symbolic regression to obtain explicit correction formulas for both a single resonator and a coupled dimer, leading to interpretable residual models. Fourth, we show that assisting symbolic regression with a proxy for the next asymptotic correction improves or stabilizes the learned formulas, supporting the interpretation that the residual is governed by the expected asymptotic scales. Finally, we investigate the efect of noisy training resonances and show that the proposed framework remains robust to moderate noise, especially when the number of training samples is increased.

The paper is organized as follows. In Section 2, we introduce the mathematical setting and formulate the resonance problem using volume integral operators. Section 3 recalls the asymptotic and reduced modelling framework, including the next-order structures that motivate the feature design. In Section 4, we present the residual learning method based on physics-informed Ridge regression. Section 5 introduces the symbolicregression correction and the assisted symbolic-regression strategy. Section 6 presents the numerical experiments for single-resonator and dimer configurations, including ablation studies, assisted symbolic-regression tests, and noise robustness experiments. Finally, Section 7 discusses the implications, limitations, and possible extensions of the approach.

## 2 Mathematical setting

## 2.1 Governing equations, non-dimensionalisation and parametric setting

We consider time-harmonic wave propagation in a two-dimensional setting, governed by the Helmholtz equation

$$
\nabla \cdot \left( \frac { 1 } { \mu ( x ) } \nabla u ( x ) \right) + \omega ^ { 2 } \varepsilon ( x , \omega ) u ( x ) = 0 , \qquad x \in \mathbb { R } ^ { 2 } ,\tag{2.1}
$$

where $\omega \in \mathbb { C }$ denotes the angular frequency, $\varepsilon ( x , \omega )$ is the possibly dispersive and complex-valued permittivity, and $\mu ( x )$ is the permeability. The background medium is homogeneous, with constant parameters $( \varepsilon _ { 0 } , \mu _ { 0 } )$ , and the resonators occupy a bounded domain $D \subset \mathbb { R } ^ { 2 }$ . The material parameters are

$$
\varepsilon ( x , \omega ) = \left\{ \varepsilon _ { r } ( \omega ) , \begin{array} { l l } { x \in D , } \\ { \varepsilon _ { 0 } , } & { x \in \mathbb { R } ^ { 2 } \setminus D , } \end{array} \right. \qquad \mu ( x ) \equiv \mu _ { 0 } .\tag{2.2}
$$

The resonator permittivity is described by a Lorentz-type dispersive law,

$$
\varepsilon _ { r } ( \omega ) = \varepsilon _ { 0 } + \frac { \alpha } { \beta - \omega ^ { 2 } - i \gamma \omega + \eta k _ { 0 } ^ { 2 } } , \qquad k _ { 0 } = \omega \sqrt { \varepsilon _ { 0 } \mu _ { 0 } } ,\tag{2.3}
$$

where the material parameters $\alpha , \beta , \gamma , \eta$ are real, with $\gamma \geq 0$ describing damping. The frequency $\omega \in \mathbb { C }$ is complex in the resonance problem and therefore the background wavenumber $k _ { 0 }$ may also be complex. This model captures both dispersion and damping efects and is commonly used in the study of perovskite-based resonators, as described in [2]. Since $k _ { 0 } ^ { 2 } = \omega ^ { 2 } \varepsilon _ { 0 } \mu _ { 0 }$ , the denominator in (2.3) may equivalently be viewed as a quadratic polynomial in $\omega .$ . We keep the form involving $k _ { 0 } ^ { 2 }$ because it matches the notation used in the asymptotic expansion of the volume integral operator.

In order to avoid dimensional ambiguities, we work throughout with non-dimensional variables. We choose the reference frequency and length scales

$$
\omega _ { * } = \sqrt { \beta } , \qquad L _ { * } = \frac { 1 } { \omega _ { * } \sqrt { \varepsilon _ { 0 } \mu _ { 0 } } } ,\tag{2.4}
$$

and define

$$
\widehat { x } = \frac { x } { L _ { * } } , \qquad \widehat { \omega } = \frac { \omega } { \omega _ { * } } , \qquad \widehat { k } _ { 0 } = L _ { * } k _ { 0 } , \qquad \widehat { \gamma } = \frac { \gamma } { \omega _ { * } } .\tag{2.5}
$$

The eigenvalues of the volume integral operator are scaled by

$$
\widehat { \lambda } _ { j } = \frac { \lambda _ { j } } { L _ { * } ^ { 2 } } .\tag{2.6}
$$

After this non-dimensionalisation, the resonance condition keeps the same form, but all quantities appearing in it are dimensionless. For readability, we drop the hats in the remainder of the paper. Thus $\omega , k _ { 0 } , \gamma _ { : }$ , and $\lambda _ { j }$ should be understood below as non-dimensional quantities, and the numerical constants appearing in the symbolic-regression formulas are dimensionless fitted coeficients.

We now specify the parametric family considered in the numerical experiments. The reference singleresonator geometry is fixed and is taken to be the unit disk $B = \{ x \in \mathbb { R } ^ { 2 } : | x | < 1 \}$ . The physical resonator is obtained by the subwavelength rescaling

$$
D _ { \delta } = \delta B , \qquad 0 < \delta \ll 1 .\tag{2.7}
$$

For the dimer, the reference geometry depends on a dimensionless separation parameter $\kappa > 0$ and is given by $B _ { \kappa } = B _ { 1 } ^ { \kappa } \cup B _ { 2 } ^ { \kappa }$ , where $B _ { 1 } ^ { \kappa }$ and $B _ { 2 } ^ { \kappa }$ are two unit disks whose centres are located at $\left( - 1 - \textstyle { \frac { \kappa } { 2 } } , 0 \right)$ and $\left( 1 + { \frac { \kappa } { 2 } } , 0 \right)$ Thus κ represents the gap between the two disks in the reference configuration, and the scaled dimer is

$$
D _ { \delta , \kappa } = \delta B _ { \kappa } .\tag{2.8}
$$

In the numerical experiments, we denote by $\omega _ { \mathrm { r e f } }$ the reference resonance computed from the high-fidelity numerical model introduced below. In the single-resonator case this is one complex resonance, while in the dimer case the two resonance branches are denoted by $\omega _ { 1 , \mathrm { r e f } }$ and $\omega _ { 2 , \mathrm { r e f } }$

In the learning experiments, the geometry of the reference resonator is fixed. For the single-resonator problem, the only varying parameters are the size parameter δ and the damping parameter $\gamma .$ . Thus the learned correction is constructed over the parametric map

$$
( \delta , \gamma ) \longmapsto \omega _ { \mathrm { r e f } } .
$$

For the dimer, the varying parameters are the size parameter $\delta ,$ the damping parameter $\gamma _ { ; }$ , and the dimensionless separation parameter κ, giving the map

$$
\begin{array} { r } { ( \delta , \gamma , \kappa ) \longmapsto \left( \omega _ { 1 , \mathrm { r e f } } , \omega _ { 2 , \mathrm { r e f } } \right) . } \end{array}
$$

The material parameters $\alpha , \beta , \eta$ are fixed and assumed to be known. The damping parameter $\gamma$ is varied because it directly controls the imaginary part of the dispersive response and therefore the resonance linewidth. The present work should therefore be understood as a residual correction study for a fixed dispersive material family, not as a learning problem over arbitrary materials.

This choice reflects the physical role of the diferent parameters in the Lorentz model. The parameters $\alpha , \beta ,$ and η determine the underlying dispersive material law: α fixes the oscillator strength, $\beta$ fixes the characteristic material frequency, and $\eta$ controls the strength of the spatial-dispersion correction. In the present study, these quantities are treated as known material constants. By contrast, $\gamma$ controls the damping, and therefore directly afects the imaginary part and linewidth of the resonances. It is also a natural parameter through which losses, fabrication variability, or material quality can change. We therefore vary $\gamma$ in order to test whether the learned correction remains stable across diferent damping regimes, while keeping the underlying dispersive material family fixed.

Finally, the scattered field is required to satisfy the outgoing Sommerfeld radiation condition. Namely, if $u ^ { \mathrm { i n c } }$ denotes the incident field and $k _ { 0 } = \omega \sqrt { \varepsilon _ { 0 } \mu _ { 0 } }$ is the background wavenumber, then

$$
\operatorname* { l i m } _ { | x | \to \infty } | x | ^ { \frac { 1 } { 2 } } \left( { \frac { \partial } { \partial | x | } } - i k _ { 0 } \right) \left( u ( x ) - u ^ { \mathrm { i n c } } ( x ) \right) = 0 .\tag{2.9}
$$

## 2.2 Volume integral formulation

Let us introduce the contrast function

$$
\xi ( \omega ) = \mu _ { 0 } \left( \varepsilon _ { r } ( \omega ) - \varepsilon _ { 0 } \right) .\tag{2.10}
$$

The total field u can be represented via the Lippmann–Schwinger equation

$$
u ( x ) = u ^ { \mathrm { i n c } } ( x ) - \omega ^ { 2 } \int _ { D } G ( x , y ; \omega ) \xi ( \omega ) u ( y ) d y ,\tag{2.11}
$$

where $G ( \cdot , \cdot ; \omega )$ is the free-space Green’s function of the background medium. In two dimensions, the Green’s function admits the representation

$$
G ( x , y ; \omega ) = - \frac { 1 } { 2 \pi } \log | x - y | + R ( x , y ; \omega ) ,\tag{2.12}
$$

where R is a smooth remainder term. This logarithmic singularity is responsible for the characteristic logarithmic terms that appear in the subwavelength asymptotics. Defining the volume integral operator

$$
{ \mathcal K } _ { D } ^ { \omega } [ u ] ( x ) = \int _ { D } G ( x , y ; \omega ) u ( y ) d y ,\tag{2.13}
$$

the Lippmann-Schwinger equation can be written in operator form as $u + \omega ^ { 2 } \xi ( \omega ) \mathcal { K } _ { D } ^ { \omega } [ u ] = u ^ { \mathrm { i n c } }$ . In the subwavelength regime considered below, asymptotic approximations of ${ \mathcal { K } } _ { D } ^ { \omega }$ yield reduced resonance conditions that can be solved eficiently.

## 2.3 Resonance problem

Resonances are defined as complex frequencies ω for which the corresponding homogeneous problem admits a non-trivial outgoing solution. Equivalently, they are the values of $\omega \in \mathbb { C }$ for which

$$
u + \omega ^ { 2 } \xi ( \omega ) \mathcal { K } _ { D } ^ { \omega } [ u ] = 0\tag{2.14}
$$

has a non-zero solution. Thus the operator Id $+ \omega ^ { 2 } \xi ( \omega ) \mathcal { K } _ { D } ^ { \omega }$ is not invertible. This gives a nonlinear spectral problem, since both the material contrast and the integral operator depend on the frequency.

If $\lambda _ { j } ( \omega )$ denotes an eigenvalue of ${ \cal { { \cal { K } } } } _ { D } ^ { \omega }$ , then a formal resonance condition is

$$
1 + \omega ^ { 2 } \xi ( \omega ) \lambda _ { j } ( \omega ) = 0 .\tag{2.15}
$$

In the asymptotic regime, one often replaces $\lambda _ { j } ( \omega )$ by an approximation obtained from the leading terms of the volume integral operator. This leads to explicit or low-dimensional approximations of the resonant frequencies.

The corresponding asymptotic prediction is denoted by $\omega _ { \mathrm { a s y m p } }$ , and the learned residual is $\Delta \omega = \omega _ { \mathrm { r e f } } -$ ω<sub>asymp</sub>.

## 2.4 Coupled resonator systems

We also consider systems of multiple resonators. In particular, for a pair of identical resonators, or dimer, we use the scaled configuration $D _ { \delta , \kappa } = \delta B _ { \kappa }$ , where $B _ { \kappa }$ is the reference dimer introduced above. Thus κ controls the separation in the reference geometry, while the physical configuration is obtained by the common scaling factor δ. For notational simplicity, we write D instead of $D _ { \delta , \kappa }$ and B instead of $B _ { \kappa }$ when no confusion is possible.

In this setting, the operator ${ \mathcal { K } } _ { D } ^ { \omega }$ has a natural block structure,

$$
\begin{array} { r } { \mathcal { K } _ { D } ^ { \omega } = \left( \begin{array} { l l } { \mathcal { K } _ { D _ { 1 } D _ { 1 } } ^ { \omega } } & { \mathcal { K } _ { D _ { 1 } D _ { 2 } } ^ { \omega } } \\ { \mathcal { K } _ { D _ { 2 } D _ { 1 } } ^ { \omega } } & { \mathcal { K } _ { D _ { 2 } D _ { 2 } } ^ { \omega } } \end{array} \right) , } \end{array}\tag{2.16}
$$

where the diagonal blocks describe self-interactions and the of-diagonal blocks describe coupling between the two resonators. This coupling splits the leading resonance into two branches, associated with symmetric and antisymmetric modes. The size of the splitting depends on the separation distance $\kappa ,$ as described in [3].

Consequently, the resonance condition (2.15) gives two distinct resonant frequencies, which we denote by

$$
\omega _ { 1 } \quad \mathrm { a n d } \quad \omega _ { 2 } .
$$

Their asymptotic approximations will be denoted by

$$
\omega _ { 1 , \mathrm { a s y m p } } \quad \mathrm { a n d } \quad \omega _ { 2 , \mathrm { a s y m p } }
$$

and the corresponding residuals by

$$
\Delta \omega _ { j } = \omega _ { j , \mathrm { r e f } } - \omega _ { j , \mathrm { a s y m p } } , \qquad j = 1 , 2 .
$$

The accurate approximation of these residuals is more delicate than in the single-resonator case, since they contain both higher-order self-interaction efects and coupling-induced corrections.

## 3 Asymptotic modelling

The resonance condition introduced in Section 2 depends on the spectral properties of the volume integral operator. In the subwavelength regime, these spectral quantities can be approximated by asymptotic expansions or by reduced-order models. Such approximations are computationally eficient and provide explicit insight into the dependence of the resonances on the physical and geometric parameters. However, when inserted into the nonlinear resonance condition, even small errors in the approximation of the operator spectrum may lead to visible errors in the predicted resonant frequencies.

The aim of this section is to recall the asymptotic structure that will later guide the learning procedure. In particular, the logarithmic scaling of the two-dimensional Green’s function leads to characteristic terms involving $\delta , k _ { 0 }$ , and log $\left( \delta k _ { 0 } \right)$ . These terms will play a central role in the construction of physics-informed features and in the interpretation of the symbolic regression formulas.

## 3.1 Integral operator and subwavelength scaling

Let $D \subset \mathbb { R } ^ { 2 }$ denote the resonator domain. The volume integral operator associated with D is (2.13). We consider the regime in which the resonator is a small rescaling of a fixed reference domain B:

$$
D = \delta B , \qquad 0 < \delta \ll 1 .\tag{3.1}
$$

Introducing the rescaled variables $x = \delta \widetilde { x } , y = \delta \widetilde { y }$ , and writing $\widetilde { u } ( \widetilde { y } ) = u ( \delta \widetilde { y } )$ , we obtain

$$
{ \cal K } _ { \cal D } ^ { \omega } [ u ] ( \delta \widetilde x ) = \delta ^ { 2 } \int _ { \cal B } \left( - \frac { 1 } { 2 \pi } \log \delta - \frac { 1 } { 2 \pi } \log | \widetilde x - \widetilde y | + R ( \delta \widetilde x , \delta \widetilde y ; \omega ) \right) \widetilde u ( \widetilde y ) \mathrm { d } \widetilde y .\tag{3.2}
$$

This formula separates the explicit dependence on the size parameter δ from the operator acting on the fixed domain $B .$ In particular, it shows that the two-dimensional subwavelength regime is governed not only by powers of δ, but also by logarithmic factors. Moreover, since the regular part of the Green’s function depends on the frequency through the background wavenumber $k _ { 0 } = \omega \sqrt { \varepsilon _ { 0 } \mu _ { 0 } }$ , higher-order terms involve both δ and $k _ { 0 }$

## 3.2 Asymptotic structure of the spectrum

Let $\lambda ( \delta , \omega )$ denote an eigenvalue of ${ \cal { { \cal { K } } } } _ { D } ^ { \omega }$ . In the subwavelength regime, the leading spectral behaviour is determined by the logarithmic kernel on the reference domain. The corresponding static logarithmic operator is

$$
{ \mathcal K } _ { B } ^ { ( 0 ) } [ u ] ( x ) = - \frac { 1 } { 2 \pi } \int _ { B } \log \left| x - y \right| u ( y ) \mathrm { d } y .\tag{3.3}
$$

It appears naturally in the rescaled expression (3.2). The remaining terms in the Green’s function produce higher-order corrections that depend on the frequency and on the background wavenumber.

Schematically, the relevant eigenvalues may be written as

$$
\lambda ( \delta , \omega ) = \lambda _ { \mathrm { l e a d } } ( \delta ) + \lambda _ { \mathrm { c o r r } } ( \delta , \omega ) ,\tag{3.4}
$$

where $\lambda _ { \mathrm { l e a d } }$ is determined by the static logarithmic operator and the geometry of $B ,$ while $\lambda _ { \mathrm { c o r r } }$ contains the frequency-dependent higher-order contributions.

The next-order structure in two dimensions contains terms of the form

$$
\delta ^ { 2 } k _ { 0 } ^ { 2 } \log ( \delta k _ { 0 } ) .\tag{3.5}
$$

At the operator level, this corresponds schematically to an expansion of the rescaled operator of the form

$$
\mathcal { K } _ { B } ^ { \omega } \simeq \left( \log ( \delta k _ { 0 } ) + \gamma _ { \mathrm { E } } \right) { \mathcal { K } } _ { B } ^ { ( - 1 ) } + { \mathcal { K } } _ { B } ^ { ( 0 ) } + \delta ^ { 2 } k _ { 0 } ^ { 2 } \log ( \delta k _ { 0 } ) { \mathcal { K } } _ { B } ^ { ( 1 ) } + \cdots ,\tag{3.6}
$$

where ${ \ K } _ { B } ^ { ( - 1 ) } , { \ K } _ { B } ^ { ( 0 ) }$ , and ${ \kappa } _ { B } ^ { ( 1 ) }$ are operators on the reference domain $B ,$ and $\gamma _ { \mathrm { E } }$ denotes the Euler–Mascheroni constant. The precise form of these operators depends on the expansion of the Green’s function, but the important point for the present work is the scale of the correction. We refer to [2] for the expressions of these operators.

In the continuous resonance problem, $k _ { 0 }$ denotes the background wavenumber and is generally complex when evaluated at a complex resonance. In the learning features, however, we use a real positive frequency scale extracted from the asymptotic resonance. For the single-resonator case, we define

$$
q _ { \mathrm { a s y m p } } = | \omega _ { \mathrm { a s y m p } } | .\tag{3.7}
$$

For the dimer, the corresponding branch-dependent quantities are

$$
q _ { j } = | \omega _ { j , \mathrm { a s y m p } } | , \qquad j = 1 , 2 .\tag{3.8}
$$

Thus, the logarithmic frequency-dependent features used involve real logarithms of positive quantities. In particular, the feature-level analogues of the next-order scales are

$$
\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log ( \delta q _ { \mathrm { a s y m p } } ) , \qquad \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } ,\tag{3.9}
$$

and similarly with $q _ { j }$ for the dimer branches.

Using

$$
\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log ( \delta q _ { \mathrm { a s y m p } } ) = \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log \delta + \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log q _ { \mathrm { a s y m p } } ,\tag{3.10}
$$

we obtain the independent feature family

$$
\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } , \qquad \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log \delta , \qquad \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log q _ { \mathrm { a s y m p } } .\tag{3.11}
$$

These quantities will be used in the learning framework as physics-informed features and in the assisted symbolicregression experiments. The combined term $\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log ( \bar { \delta } q _ { \mathrm { a s y m p } } )$ is useful for identifying the asymptotic scale, but it is not treated as an independent feature because of the decomposition (3.10).

## 3.3 From operator corrections to resonance residuals

The appearance of the same feature family in the frequency residual can be justified from the resonance condition. For a given branch, define the scalar resonance function

$$
\Phi ( \omega , \delta ) = 1 + \omega ^ { 2 } \xi ( \omega ) \lambda ( \delta , \omega ) ,\tag{3.12}
$$

where $\lambda ( \delta , \omega )$ denotes the corresponding eigenvalue of the volume integral operator. The asymptotic resonance $\omega _ { \mathrm { a s y m p } }$ is obtained by replacing λ with its asymptotic approximation $\lambda _ { \mathrm { a s y m p } } .$ Thus

$$
1 + \omega _ { \mathrm { a s y m p } } ^ { 2 } \xi ( \omega _ { \mathrm { a s y m p } } ) \lambda _ { \mathrm { a s y m p } } ( \delta , \omega _ { \mathrm { a s y m p } } ) = 0 .\tag{3.13}
$$

Writing

$$
\lambda ( \delta , \omega ) = \lambda _ { \mathrm { a s y m p } } ( \delta , \omega ) + \Delta \lambda ( \delta , \omega ) ,\tag{3.14}
$$

and assuming that the resonance is simple, so that

$$
\partial _ { \omega } \Phi _ { \mathrm { a s y m p } } ( \omega _ { \mathrm { a s y m p } } , \delta ) \neq 0 ,\tag{3.15}
$$

the implicit function theorem gives, to first order,

$$
\Delta \omega = \omega _ { \mathrm { r e f } } - \omega _ { \mathrm { a s y m p } } \simeq - \frac { \omega _ { \mathrm { a s y m p } } ^ { 2 } \xi ( \omega _ { \mathrm { a s y m p } } ) \Delta \lambda ( \delta , \omega _ { \mathrm { a s y m p } } ) } { \partial _ { \omega } \Phi _ { \mathrm { a s y m p } } ( \omega _ { \mathrm { a s y m p } } , \delta ) } .\tag{3.16}
$$

Consequently, the frequency residual inherits the scale of the neglected terms in the eigenvalue and operator expansions, up to multiplication by a smooth branch-dependent factor. Since the next-order operator correction contains the two-dimensional frequency-dependent scale

$$
\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log ( \delta q _ { \mathrm { a s y m p } } ) = \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log \delta + \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log q _ { \mathrm { a s y m p } } ,\tag{3.17}
$$

it is natural to include the feature family

$$
\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } , \qquad \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log \delta , \qquad \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log q _ { \mathrm { a s y m p } }\tag{3.18}
$$

in the residual model.

## 3.4 Asymptotic resonance approximation

Substituting an asymptotic approximation of $\lambda ( \delta , \omega )$ into the resonance condition $1 + \omega ^ { 2 } \xi ( \omega ) \lambda ( \delta , \omega ) = 0$ yields an approximation of the resonant frequency. At leading order, one obtains a reduced nonlinear equation of the form $1 + \omega ^ { 2 } \xi ( \omega ) \lambda _ { \mathrm { a s y m p } } ( \delta , \omega ) = 0$ , where $\lambda _ { \mathrm { a s y m p } }$ denotes the chosen asymptotic approximation of the relevant eigenvalue. We denote the resulting resonance by

$$
\omega _ { \mathrm { a s y m p } } .\tag{3.19}
$$

The reference resonance, computed from the more accurate spectral model, is denoted by $\omega _ { \mathrm { r e f } }$

The discrepancy

$$
\Delta \omega = \omega _ { \mathrm { r e f } } - \omega _ { \mathrm { a s y m p } }\tag{3.20}
$$

contains the efect of neglected higher-order terms in the operator expansion, as well as the nonlinear propagation of these errors through the resonance condition. The purpose of the data-driven correction developed below is to approximate this residual while preserving the asymptotic structure of the problem.

## 3.5 Coupled resonators and interaction operators

We now consider the dimer configuration $D _ { \delta , \kappa } = \delta B _ { \kappa }$ , where $B _ { \kappa }$ is the reference dimer made of two unit disks separated by the dimensionless gap κ. For notational simplicity, we again write $D = D _ { 1 } \cup D _ { 2 }$ . In this case, the volume integral operator has the block structure (2.16) where,

$$
K _ { D _ { j } D _ { l } } ^ { \omega } [ u ] ( x ) = \int _ { D _ { l } } G ( x , y ; \omega ) u ( y ) d y , \qquad x \in D _ { j } , \qquad j , l = 1 , 2 .\tag{3.21}
$$

The diagonal blocks describe the self-interaction of each resonator, while the of-diagonal blocks describe the interaction between the two components.

In the subwavelength regime, the leading behaviour is governed by approximately constant modes on each resonator. This leads to a finite-dimensional reduced model, whose entries are averaged interaction terms. A representative form is

$$
\begin{array} { r } { M ( \kappa ) = \left( \begin{array} { l l } { \langle K _ { D _ { 1 } D _ { 1 } } ^ { \omega } \mathbb { 1 } _ { D _ { 1 } } , \mathbb { 1 } _ { D _ { 1 } } \rangle } & { \langle K _ { D _ { 1 } D _ { 2 } } ^ { \omega } \mathbb { 1 } _ { D _ { 2 } } , \mathbb { 1 } _ { D _ { 1 } } \rangle } \\ { \langle K _ { D _ { 2 } D _ { 1 } } ^ { \omega } \mathbb { 1 } _ { D _ { 1 } } , \mathbb { 1 } _ { D _ { 2 } } \rangle } & { \langle K _ { D _ { 2 } D _ { 2 } } ^ { \omega } \mathbb { 1 } _ { D _ { 2 } } , \mathbb { 1 } _ { D _ { 2 } } \rangle } \end{array} \right) . } \end{array}\tag{3.22}
$$

The eigenvalues of this reduced matrix approximate the leading eigenvalues of the full operator. The of-diagonal terms depend on the separation distance κ and induce a splitting into two branches, associated with symmetric and antisymmetric modes. We denote the corresponding asymptotic resonances by

$$
\omega _ { 1 , \mathrm { a s y m p } } , \qquad \omega _ { 2 , \mathrm { a s y m p } } .\tag{3.23}
$$

The corresponding reference resonances are denoted by $\omega _ { 1 , \mathrm { r e f } }$ and $\omega _ { 2 , \mathrm { r e f } }$ , and the residuals are

$$
\Delta \omega _ { j } = \omega _ { j , \mathrm { r e f } } - \omega _ { j , \mathrm { a s y m p } } , \qquad j = 1 , 2 .\tag{3.24}
$$

Compared with the single-resonator case, these residuals contain both self-interaction corrections and couplinginduced errors. This makes the dimer a useful test case for assessing whether the learned correction can capture interaction efects.

## 3.6 Limitations of asymptotic and reduced models

The asymptotic and reduced models described above provide eficient approximations of the resonances, but they have several limitations. First, their accuracy deteriorates as the system moves away from the strict subwavelength regime. Second, the next terms in the expansion, although known structurally, are dificult to compute explicitly in a form that is convenient for numerical prediction. Third, in coupled systems, interaction efects introduce branch-dependent corrections that are not fully captured by the leading reduced model. Finally, the nonlinear dependence of the resonance condition on ω can amplify small spectral errors.

These observations motivate a correction strategy that keeps the asymptotic approximation as the leading model, but learns the residual $\omega _ { \mathrm { r e f } } \mathrm { ~ - ~ } \omega _ { \mathrm { a s y m p } }$ from data. The next section introduces this residual learning framework. The asymptotic structures identified above, in particular the terms involving $\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log ( \delta q _ { \mathrm { a s y m p } } )$ will be used to design the input features and to interpret the symbolic formulas obtained later.

## 4 Learning the asymptotic residual

## 4.1 Residual formulation

The asymptotic and reduced models described in Section 3 provide eficient approximations of the resonant frequencies, but they neglect higher-order terms and interaction efects. Rather than learning the resonances directly, we use these approximations as a baseline and learn only the remaining discrepancy.

Let ω<sub>asymp</sub> denote the resonance predicted by an asymptotic or reduced model, and let $\omega _ { \mathrm { r e f } }$ denote the corresponding reference value. We define the residual

$$
\Delta \omega = \omega _ { \mathrm { r e f } } - \omega _ { \mathrm { a s y m p } } .\tag{4.1}
$$

The learning task is then to approximate the map

$$
\mathcal { F } : X \longmapsto \Delta \omega ,\tag{4.2}
$$

where X denotes a vector of physical, asymptotic, and physics-informed features. Once an approximation $\Delta \omega _ { \mathrm { M I } }$ has been obtained, the corrected resonance is given by

$$
\omega _ { \mathrm { p r e d } } = \omega _ { \mathrm { a s y m p } } + \Delta \omega _ { \mathrm { M L } } .\tag{4.3}
$$

This formulation has two main advantages. First, the leading-order behaviour is already encoded in $\omega _ { \mathrm { a s y m p } } ,$ so that the learning problem concerns only a smaller correction. Second, the residual is expected to inherit the structure of the neglected asymptotic terms. This makes it possible to use simple learning models, provided that the input features are chosen consistently with the underlying asymptotics.

For the dimer, the same construction is applied branch by branch:

$$
\Delta \omega _ { j } = \omega _ { j , \mathrm { r e f } } - \omega _ { j , \mathrm { a s y m p } } , \qquad j = 1 , 2 .\tag{4.4}
$$

The corrected predictions are then

$$
\omega _ { j , \mathrm { p r e d } } = \omega _ { j , \mathrm { a s y m p } } + \Delta \omega _ { j , \mathrm { M L } } , \qquad j = 1 , 2 .\tag{4.5}
$$

## 4.2 Feature design

The choice of features is guided by the asymptotic structure of the problem. We use three types of input quantities.

First, we include the raw physical and geometric parameters, such as the size parameter δ, the damping parameter γ, and, in the dimer case, the separation distance κ. Logarithmic transforms of these parameters are also included when they are relevant to the asymptotic scaling.

Second, we include quantities obtained from the asymptotic or reduced model itself. These include the real and imaginary parts of $\omega _ { \mathrm { a s y m p } }$ , the approximate eigenvalues $\lambda _ { \mathrm { a s y m p } } .$ , and, for the dimer, quantities measuring the splitting between the two branches. These features encode the leading-order spectral information already available from the reduced model.

Third, we include physics-informed features motivated by the next terms in the asymptotic expansion. In the single-resonator case, these include terms such as

$$
\delta ^ { 2 } , \qquad \delta ^ { 2 } \log \delta , \qquad \gamma \delta ^ { 2 } ,\tag{4.6}
$$

together with the frequency-dependent family

$$
\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } , \qquad \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log \delta , \qquad \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log q _ { \mathrm { a s y m p } } ,\tag{4.7}
$$

Here $q _ { \mathrm { a s y m p } }$ is defined by (3.7). For the dimer, the same features are evaluated branchwise using $q _ { j }$ from (3.8). This notation separates the physical background wavenumber $k _ { 0 }$ , which belongs to the resonance problem, from the positive real frequency scale used as an input feature in the learning models. These features reflect the structure of the two-dimensional Green’s function expansion discussed in Section 3. In the dimer case, analogous branch-dependent features are used, evaluated with the corresponding asymptotic wavenumbers, together with interaction features involving the eigenvalue splitting and the separation distance κ.

The purpose of this feature design is not to replace the asymptotic model, but to expose the dominant scales of the missing correction to the learning algorithm. This will also allow us to test, through the ablation study, whether the improvement genuinely comes from physically meaningful structure.

## 4.3 Regression model

As a first correction model, we use Ridge regression. The real and imaginary parts of the residual are learned simultaneously. Given a feature vector $X \in \mathbb { R } ^ { p }$ , we approximate

$$
\Delta \omega \approx { \cal W } X + b ,\tag{4.8}
$$

where $W \in \mathbb { R } ^ { 2 \times p }$ and $b \in \mathbb { R } ^ { 2 }$ . The two components correspond to $\Delta \omega = \left( \operatorname { R e } \Delta \omega , \operatorname { I m } \Delta \omega \right)$ . The parameters are obtained by minimizing the regularized least-squares functional

$$
\operatorname* { m i n } _ { W , b } \sum _ { n = 1 } ^ { N } \left\| W X _ { n } + b - \Delta \omega _ { n } \right\| ^ { 2 } + \alpha _ { \mathrm { { R } } } \| W \| ^ { 2 } ,\tag{4.9}
$$

where $\alpha _ { \mathrm { { R } } } ~ > ~ 0$ is the Ridge regularization parameter. The numerical details of the Ridge implementation, including feature standardization, train-test splits, and the value of the regularization parameter, are given in Appendix A.

The use of a linear model is deliberate. Since the asymptotic approximation already captures the dominant behaviour, and since the features include the expected higher-order scalings, the residual should be well approximated by a low-dimensional combination of these quantities. This also makes the learned correction easier to interpret than a high-capacity black-box model.

The model is trained on samples of the form (features, $\omega _ { \mathrm { a s y m p } } , \omega _ { \mathrm { r e f } } )$ , with target $\omega _ { \mathrm { r e f } } - \omega _ { \mathrm { a s y m p } }$ . Details on the train-test split and preprocessing are given in Appendix A.

## 4.4 Corrected prediction and error measure

Given a trained model, the corrected prediction is

$$
\omega _ { \mathrm { p r e d } } = \omega _ { \mathrm { a s y m p } } + \Delta \omega _ { \mathrm { M L } } .\tag{4.10}
$$

The accuracy is measured by the relative error

$$
E ( \omega ) = \frac { | \omega _ { \mathrm { p r e d } } - \omega _ { \mathrm { r e f } } | } { | \omega _ { \mathrm { r e f } } | } .\tag{4.11}
$$

The same definition is used to evaluate the uncorrected asymptotic model, replacing $\omega _ { \mathrm { p r e d } }$ by $\omega _ { \mathrm { a s y m p } } .$ . This allows a direct comparison between the original asymptotic approximation and the learned correction.

## 4.5 Interpretation

The residual learning framework can be viewed as a data-driven extension of asymptotic modelling. The asymptotic approximation provides the leading-order physics, while the learned residual captures higher-order terms and coupling efects that are dificult to compute explicitly.

The Ridge model provides a stable and accurate correction, but its output is still a fitted linear combination of many features. In the next section, we go one step further and use symbolic regression to search for compact analytic expressions for the residual. This provides a more interpretable correction and makes it possible to compare the learned formulas directly with the expected asymptotic structure.

## 5 Symbolic regression of the residual

The residual learning framework introduced in Section 4 improves the accuracy of the asymptotic approximation while preserving the leading-order physical structure. However, even when a simple linear model is used, the learned correction remains a fitted combination of prescribed features. In order to obtain more explicit and interpretable correction formulas, we complement the regression approach with symbolic regression.

Symbolic regression seeks analytic expressions that approximate a target quantity from data by combining input variables through elementary operations. Unlike standard regression models, which fit coeficients within a fixed functional form, symbolic regression searches over the space of mathematical expressions. This makes it particularly useful in scientific applications, where the goal is not only to predict accurately but also to identify compact formulas that reveal structure in the data. In this work, we use PySR, based on the SymbolicRegression.jl framework, following the approach described in [6].

## 5.1 Symbolic regression target

We apply symbolic regression to the asymptotic residual rather than to the full resonance. For a single resonator, the target is

$$
\Delta \omega = \omega _ { \mathrm { r e f } } - \omega _ { \mathrm { a s y m p } } .\tag{5.1}
$$

For the dimer, the same procedure is applied separately to the two branches:

$$
\Delta \omega _ { j } = \omega _ { j , \mathrm { r e f } } - \omega _ { j , \mathrm { a s y m p } } , \qquad j = 1 , 2 .\tag{5.2}
$$

Since the resonances are complex-valued, we fit the real and imaginary parts separately. Thus, for each configuration, symbolic regression is used to find expressions for Re ∆ω and Im $\Delta \omega$ . This choice is important. Learning the full resonance directly would require the symbolic model to rediscover the leading-order asymptotic behaviour. By learning only the residual, the symbolic search is focused on the missing higher-order correction. This is consistent with the philosophy of the paper: the asymptotic model provides the dominant physical approximation, while data-driven methods are used only to model the discrepancy.

## 5.2 Input variables and asymptotic structure

The input variables used for symbolic regression are chosen from the same physics-informed feature set as in Section 4. For the single-resonator problem, these include the size parameter $\delta ,$ the damping parameter $\gamma ,$ the asymptotic resonance $\omega _ { \mathrm { a s y m p } }$ , and asymptotically motivated combinations such as

$$
\delta ^ { 2 } , \qquad \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } , \qquad \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log q _ { \mathrm { a s y m p } } , \qquad \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log ( \delta ) .\tag{5.3}
$$

For the dimer, the symbolic regression also uses branch-dependent asymptotic quantities, including the approximate eigenvalues and the splitting between the two branches. If $\lambda _ { 1 , \mathrm { a s y m p } }$ and $\lambda _ { \mathrm { 2 , a s y m p } }$ denote the two leading asymptotic eigenvalues, we use in particular the splitting

$$
\Lambda = \left| \lambda _ { 1 , \mathrm { a s y m p } } - \lambda _ { 2 , \mathrm { a s y m p } } \right| .\tag{5.4}
$$

The inclusion of these variables is guided by the asymptotic expansion. In particular, the term $\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log ( \delta q _ { \mathrm { a s y m p } } )$ appears naturally in the next-order correction to the two-dimensional volume integral operator (3.6). Therefore, when symbolic regression selects expressions involving this family of terms, the result can be interpreted as being consistent with the expected asymptotic structure. We emphasize, however, that the symbolic formulas obtained below are data-driven fitted corrections; they should not be interpreted as rigorous derivations of the full higher-order asymptotic expansion.

## 5.3 Discovered symbolic correction formulas

We now report compact formulas obtained by symbolic regression. The formulas in this subsection are datadriven symbolic corrections: they are fitted correction laws for the residual in the sampled parameter regime, not rigorous asymptotic expansions. Their purpose is to expose which combinations of variables are used by the learned correction and to provide explicit analytic approximations of the residual.

All quantities appearing below are non-dimensional, following the convention introduced in Section 2. In particular, $\omega , \gamma , \lambda _ { j }$ , q<sub>asymp</sub>, and $q _ { j }$ denote non-dimensional quantities, and the numerical constants in the symbolic expressions are dimensionless fitted coeficients. The quantities $q _ { \mathrm { a s y m p } }$ and $q _ { j }$ are the positive real frequency scales defined in (3.7) and (3.8). Thus logarithms involving q<sub>asymp</sub> or $q _ { j }$ are real logarithms of positive quantities.

It is important to separate these symbolic formulas from the asymptotic expansion itself. Symbolic regression does not derive the next term in the asymptotic expansion. Rather, it searches for compact data-driven expressions that are compatible with the asymptotically motivated feature scales. Schematically, in the assisted setting one may view the symbolic correction as

$$
\Delta \omega _ { \mathrm { S R } } = \underbrace { \Delta \omega _ { \mathrm { p r o x y } } } _ { \mathrm { m o t i v a t e d ~ b y ~ a s y m p t o t i c s } } + \underbrace { \Delta \omega _ { \mathrm { d a t a } } } _ { \mathrm { d i s c o v e r e d ~ f r o m ~ d a t a } } ,\tag{5.5}
$$

where $\Delta \omega _ { \mathrm { p r o x y } }$ is built from the feature scales suggested by the operator expansion, while $\Delta \omega _ { \mathrm { d a t a } }$ is the remaining symbolic correction fitted from data. In the unassisted symbolic-regression experiments, the whole correction is fitted directly from data; in the assisted experiments, these two contributions are separated explicitly.

## Single resonator

For the single-resonator problem, one compact symbolic correction is obtained as follows. Writing

$$
\omega _ { \mathrm { a s y m p } } = \omega _ { \mathrm { a s y m p } , r } + i \omega _ { \mathrm { a s y m p } , i } ,\tag{5.6}
$$

the corrected resonance is

$$
\omega _ { \mathrm { s y m } } = \omega _ { \mathrm { a s y m p } } + \mathrm { R e } ( \Delta \omega _ { \mathrm { s y m } } ) + i \mathrm { I m } ( \Delta \omega _ { \mathrm { s y m } } ) ,\tag{5.7}
$$

where

$$
\mathrm { R e } ( \Delta \omega _ { \mathrm { s y m } } ) = \delta ^ { 2 } \omega _ { \mathrm { a s y m p } , r } \left( 0 . 0 2 3 9 8 7 6 5 \omega _ { \mathrm { a s y m p } , r } ^ { 2 } + 0 . 0 0 0 1 2 7 5 3 5 9 7 \omega _ { \mathrm { a s y m p } , r } \omega _ { \mathrm { a s y m p } , i } + 0 . 0 0 5 3 1 6 7 3 4 3 \right) ,\tag{5.8}
$$

and

$$
\mathrm { I m } ( { \Delta \omega } _ { \mathrm { s y m } } ) = 0 . 0 2 9 4 7 9 9 7 \omega _ { \mathrm { a s y m p } , i } \left( \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } + \delta ^ { 2 } + \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log q _ { \mathrm { a s y m p } } \right) .\tag{5.9}
$$

The real part is dominated by a $\delta ^ { 2 }$ correction modulated by the leading asymptotic resonance. The imaginary part contains the frequency-dependent structures $\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 }$ and $\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 }$ log q<sub>asymp</sub>, which are consistent with the feature scales motivated by the next-order operator expansion. The factor $\omega _ { \mathrm { a s y m p } , i }$ also reflects that the correction to the imaginary part is tied to the damping of the resonance.

## Dimer

For the dimer, symbolic regression is applied separately to the two resonance branches. Let

$$
\Lambda = \vert \lambda _ { 1 , \mathrm { a s y m p } } - \lambda _ { \mathrm { 2 , a s y m p } } \vert\tag{5.10}
$$

denote the non-dimensional asymptotic splitting of the two leading eigenvalues.

For the first branch, the corrected resonance is

$$
\omega _ { 1 , \mathrm { s y m } } = \omega _ { 1 , \mathrm { a s y m p } } + \mathrm { R e } ( \Delta \omega _ { 1 , \mathrm { s y m } } ) + i \mathrm { I m } ( \Delta \omega _ { 1 , \mathrm { s y m } } ) ,\tag{5.11}
$$

with

$$
\mathrm { R e } ( \Delta \omega _ { 1 , \mathrm { s y m } } ) = - 0 . 0 2 9 8 \delta ^ { 2 } q _ { 1 } ^ { 2 } \left( \Lambda - 0 . 2 8 5 \right) ,\tag{5.12}
$$

and

$$
\mathrm { I m } ( \Delta \omega _ { 1 , \mathrm { s y m } } ) = 0 . 0 3 4 3 4 6 \gamma \delta ^ { 2 } \left( \Lambda - 0 . 2 8 8 4 1 9 1 6 \right) .\tag{5.13}
$$

For the second branch, the corrected resonance is

$$
\omega _ { 2 , \mathrm { s y m } } = \omega _ { 2 , \mathrm { a s y m p } } + \mathrm { R e } ( \Delta \omega _ { 2 , \mathrm { s y m } } ) + i \mathrm { I m } ( \Delta \omega _ { 2 , \mathrm { s y m } } ) ,\tag{5.14}
$$

where

$$
\mathrm { R e } ( \Delta \omega _ { 2 , \mathrm { s y m } } ) = \delta ^ { 2 } q _ { 2 } ^ { 2 } \left( 0 . 0 6 1 4 9 2 5 3 6 - 0 . 3 3 4 0 2 3 1 2 \lambda _ { 2 , \mathrm { a s y m p } } \right) - 1 . 4 5 \times 1 0 ^ { - 6 } ,\tag{5.15}
$$

and

$$
\mathrm { I m } ( \Delta \omega _ { 2 , \mathrm { s y m } } ) = \gamma \delta ^ { 2 } \lambda _ { 2 , \mathrm { a s y m p } } ^ { 2 } \left( \mathrm { R e } ( \omega _ { 2 , \mathrm { a s y m p } } ) - \lambda _ { 1 , \mathrm { a s y m p } } - 0 . 2 6 8 2 4 6 4 7 \right) .\tag{5.16}
$$

Since all quantities are non-dimensional, combinations such as $\mathrm { R e } ( \omega _ { 2 , \mathrm { a s y m p } } ) - \lambda _ { 1 , \mathrm { a s y m p } }$ should be read as combinations of non-dimensional input features selected by symbolic regression.

These expressions reveal two important features. First, the corrections are organized around the expected powers of the asymptotic parameter, especially $\delta ^ { 2 }$ and $\delta ^ { 2 } q _ { j } ^ { 2 }$ . Second, the dimer corrections depend explicitly on branch-dependent spectral quantities, through $\lambda _ { j , \mathrm { a s y m p } }$ and the splitting Λ. This provides numerical evidence that the learned residual contains interaction-dependent structure, rather than only single-resonator self-interaction corrections.

## 5.4 Asymptotically assisted symbolic regression

We also consider two variants in which symbolic regression is explicitly guided by the next-order asymptotic structure. The purpose of these experiments is not to claim that symbolic regression derives the next asymptotic term, but to test whether supplying this structure makes the data-driven correction simpler and more stable.

We first construct a calibrated next-order proxy, denoted by $\Delta \omega _ { \mathrm { p r o x y } }$ , using the feature family

$$
\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } , \qquad \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log \delta , \qquad \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log q _ { \mathrm { a s y m p } } ,\tag{5.17}
$$

and, in the dimer case, the corresponding branch-dependent quantities obtained by replacing $q _ { \mathrm { a s y m p } }$ with $q _ { j }$ . We then compare two assisted strategies.

The first strategy is asymptotic preconditioning. In this case, the proxy is subtracted before applying symbolic regression. The symbolic model is trained on

$$
\omega _ { \mathrm { r e f } } - \left( \omega _ { \mathrm { a s y m p } } + \Delta \omega _ { \mathrm { p r o x y } } \right) .\tag{5.18}
$$

Thus symbolic regression is asked to learn only the residual left after the asymptotically motivated proxy has been removed.

The second strategy is asymptotic feature augmentation. In this case, the symbolic model is still trained on the full residual

$$
\omega _ { \mathrm { r e f } } - \omega _ { \mathrm { a s y m p } } ,\tag{5.19}
$$

but $\Delta \omega _ { \mathrm { p r o x y } }$ is supplied as an additional input feature. This allows symbolic regression to decide how to use the proxy, rather than forcing the proxy to be subtracted beforehand.

These two strategies test diferent ways of injecting asymptotic information into the symbolic search. Asymptotic preconditioning changes the target, while asymptotic feature augmentation changes the input representation. The numerical results in Section 6 show that this assisted symbolic regression improves the accuracy further, for both the single resonator and the dimer. Thus, symbolic regression is most efective when used not as a purely black-box discovery tool, but as a mechanism for extracting compact formulas from an asymptotically informed feature space.

## 6 Numerical experiments

In this section, we evaluate the proposed asymptotics-guided correction framework on single-resonator and dimer configurations. The goal is twofold. First, we quantify the improvement obtained by learning the asymptotic residual with Ridge regression. Second, we assess whether symbolic regression can produce compact formulas that retain the accuracy of the learned correction while improving interpretability.

The reference resonances $\omega _ { \mathrm { r e f } }$ are computed from the accurate nonlinear spectral model, while $\omega _ { \mathrm { a s y m p } }$ denotes the asymptotic or reduced approximation. Given a learned residual $\Delta \omega _ { \mathrm { M L } }$ , the corrected prediction is

$$
\omega _ { \mathrm { p r e d } } = \omega _ { \mathrm { a s y m p } } + \Delta \omega _ { \mathrm { M L } } .\tag{6.1}
$$

For symbolic regression, we similarly write

$$
\omega _ { \mathrm { s y m } } = \omega _ { \mathrm { a s y m p } } + \Delta \omega _ { \mathrm { s y m } } .\tag{6.2}
$$

Accuracy is measured using the relative error

$$
E ( \omega ) = \frac { | \omega _ { \mathrm { p r e d } } - \omega _ { \mathrm { r e f } } | } { | \omega _ { \mathrm { r e f } } | } ,\tag{6.3}
$$

with the obvious replacement of $\omega _ { \mathrm { p r e d } }$ by $\omega _ { \mathrm { a s y m p } } ~ \mathrm { o r } ~ \omega _ { \mathrm { s y m } }$ when evaluating the asymptotic or symbolic models.

## 6.1 Learning protocol and validation strategy

For each configuration, the dataset is split into training and test sets. The in-distribution test corresponds to a random held-out subset of the sampled parameter domain. To assess extrapolation, we also consider outof-distribution tests in which the model is trained away from the most challenging regimes and evaluated on held-out parameter ranges. For the single resonator, we test extrapolation toward larger values of δ and larger damping γ. For the dimer, we test extrapolation toward larger δ and smaller separation distance $\kappa ,$ where interaction efects are stronger.

All pre-processing steps, including feature standardization, are fitted only on the training set and then applied to the test set. The learning target is always the residual $\omega _ { \mathrm { r e f } } - \omega _ { \mathrm { a s y m p } } ,$ not the full resonance. We retain only the physical resonance branch satisfying $\mathrm { R e } \omega > 0$ and Im $\omega < 0$ , which correspond to the outgoing damped resonances considered in this work. In the dimer case, both resonance branches are included in the error averages.

The mean relative error reported in the tables is the average of the relative error over the held-out test set:

$$
\overline { { E } } = \frac { 1 } { N _ { \mathrm { t e s t } } } \sum _ { j = 1 } ^ { N _ { \mathrm { t e s t } } } E _ { j } , \qquad E _ { j } = \frac { | \omega _ { \mathrm { p r e d } , j } - \omega _ { \mathrm { r e f } , j } | } { | \omega _ { \mathrm { r e f } , j } | } .\tag{6.4}
$$

For the dimer, the average is taken over both resonance branches.

<table><tr><td>Validation case</td><td>Asymptotic mean error</td><td>Corrected mean error</td><td>Improvement factor</td></tr><tr><td>Single, in-distribution</td><td> $\overline { { 9 . 0 9 \times 1 0 ^ { - 4 } } }$ </td><td> $\overline { { 2 . 2 2 \times 1 0 ^ { - 7 } } }$ </td><td> $\overline { { 4 . 1 0 \times 1 0 ^ { 3 } } }$ </td></tr><tr><td>Single, large-δ OOD</td><td> $2 . 2 9 \times 1 0 ^ { - 3 }$ </td><td> $7 . 8 9 \times 1 0 ^ { - 7 }$ </td><td> $2 . 9 0 \times 1 0 ^ { 3 }$ </td></tr><tr><td>Single, large-γ OOD</td><td> $9 . 9 4 \times 1 0 ^ { - 4 }$ </td><td> $7 . 9 2 \times 1 0 ^ { - 7 }$ </td><td> $1 . 2 6 \times 1 0 ^ { 3 }$ </td></tr><tr><td>Dimer, in-distribution</td><td> $1 . 2 2 \times 1 0 ^ { - 4 }$ </td><td> $3 . 4 7 \times 1 0 ^ { - 6 }$ </td><td>35.2</td></tr><tr><td>Dimer, large-δ OOD</td><td> $3 . 6 8 \times 1 0 ^ { - 4 }$ </td><td> $7 . 3 5 \times 1 0 ^ { - 5 }$ </td><td>5.01</td></tr><tr><td>Dimer, small-κ OOD</td><td> $1 . 4 7 \times 1 0 ^ { - 4 }$ </td><td> $3 . 8 3 \times 1 0 ^ { - 5 }$ </td><td>3.82</td></tr></table>

Table 1: In-distribution and out-of-distribution validation of the physics-informed residual correction. OOD denotes out-of-distribution. The corrected error corresponds to the Ridge residual model using physics-informed features.

The validation table reports the full physics-informed Ridge correction for each validation setting. Table 2 uses the same in-distribution held-out test sets as the first and fourth rows of Table 1, and compares the full physics-informed Ridge correction with symbolic and assisted symbolic regression.

These values are obtained on independent held-out test sets. They show that the learned residual is highly structured: Ridge regression already gives a substantial correction, while symbolic and assisted symbolic regres sion recover compact formulas with significantly smaller errors.

<table><tr><td>Configuration</td><td>Asymptotic</td><td>Ridge</td><td></td><td>Symbolic Assisted symbolic</td></tr><tr><td>Single resonator</td><td> $\overline { { 9 . 0 9 \times 1 0 ^ { - 4 } } }$ </td><td> $\overline { { 2 . 2 2 \times 1 0 ^ { - 7 } } }$ </td><td> $\overline { { 4 . 7 0 \times 1 0 ^ { - 8 } } }$ </td><td> $\overline { { 1 . 8 9 \times 1 0 ^ { - 1 1 } } }$ </td></tr><tr><td>Dimer</td><td> $1 . 2 2 \times 1 0 ^ { - 4 }$ </td><td> $3 . 4 7 \times 1 0 ^ { - 6 }$ </td><td> $8 . 3 5 \times 1 0 ^ { - 7 }$ </td><td> $9 . 3 0 \times 1 0 ^ { - 1 1 }$ </td></tr></table>

Table 2: Mean relative errors on the in-distribution held-out test sets for the asymptotic approximation, the full physics-informed Ridge correction, symbolic regression, and assisted symbolic regression.

## 6.2 Single resonator

We first consider a single resonator over the parameter ranges described in Appendix A. The baseline approximation is the asymptotic resonance $\omega _ { \mathrm { a s y m p } } ,$ and the learning target is the residual $\Delta \omega = \omega _ { \mathrm { r e f } } - \omega _ { \mathrm { a s y m p } } .$

Figure 1a shows the relative error as a function of δ for the asymptotic approximation and the Ridgecorrected model. As expected, the asymptotic approximation deteriorates as δ increases, reflecting the increasing importance of higher-order terms. The learned correction significantly reduces the error across the full range of δ, showing that the residual is efectively captured by the physics-informed features. The symbolic model further improves the prediction while retaining an explicit analytic form for the residual. In this case, the mean relative error is reduced from approximately $9 . 0 9 \times 1 0 ^ { - 4 }$ for the asymptotic approximation to approximately $4 . 7 0 \times 1 0 ^ { - 8 }$ for the symbolic correction. In Figure 1b, the reconstruction in the complex plane is shown. The corrected predictions are nearly indistinguishable from the reference resonances, indicating that both the real and imaginary parts are accurately recovered.

![](images/bb22d59f050416941007dd8abfdd5bb49e21ab3f3574431db6932ed4ff1dc40e.jpg)  
(a) Relative error versus the size parameter δ.

![](images/bc5cf40b1a41d3081697d4f587a7b72ed4889243854dc5fb1a2e593c80a55e49.jpg)  
(b) Reconstruction of the resonances in the complex plane.  
Figure 1: Single-resonator prediction results. Left: relative error of the asymptotic approximation and of the corrected models as a function of the asymptotic parameter δ. Right: comparison of the predicted and reference resonances in the complex plane. The asymptotic approximation deteriorates as δ increases, while the learned corrections remain close to the reference resonances over the sampled regime.

## 6.3 Dimer configuration

The dimer configuration is not intended as a simple repetition of the single-resonator experiment. It is the first nontrivial test of whether residual learning transfers from isolated self-interaction corrections to interactioninduced spectral corrections. In this case, the reduced model must approximate not only the resonance of each component, but also the coupling between the two resonators and the resulting branch splitting. The learned residual is therefore branch-dependent and depends on the separation parameter κ, in addition to δ and γ.

The convergence of the dimer reference discretization is checked in Appendix A.3, where the reference computation is compared with finer and coarser discretizations.

Figure 2a shows the relative error for the first branch. The reduced asymptotic model captures the leading interaction efect, but it does not fully account for the branch-dependent correction. The Ridge model substantially reduces the error over the range of separations. Then, Figure 2b shows the corresponding result for

the second branch. This branch is more sensitive to coupling efects, and the baseline error is more variable.   
Nevertheless, the learned correction remains efective and consistently improves the reduced approximation.

![](images/9fef1232137532139263126c6378cbb9306dec304468577eb2c326a0cdd48369.jpg)  
(a) First dimer branch: relative error as a function of the separation distance κ.

![](images/8cb38de115e2baf5bfaabc315025334342af3dc2ea0af7fa0a27852aa7706c8e.jpg)  
(b) Second dimer branch: relative error as a function of the separation distance κ.

Figure 2: Dimer prediction errors as functions of the separation distance κ for the two resonance branches. The reduced asymptotic model captures the leading coupling efect, but the Ridge correction significantly improves the approximation for both branches. The second branch exhibits a larger variability, reflecting its stronger sensitivity to interaction efects.

We then evaluate the symbolic-regression formulas for the two dimer branches. Figures 3a and 3b show that the symbolic corrections preserve the accuracy of the learned residual model while providing explicit expressions depending on the asymptotic eigenvalues, the branch splitting, and the expected powers of δ and q<sub>j</sub>.

![](images/b0690537f8bb2f35682ce504ba92a3674b38aa2624897b428b15daa24c0bd605.jpg)  
(a) First dimer branch: error of the compact symbolic-regression correction.

![](images/c73c7650acd62d6eac2315a741d496af93ad4a2f49aa3acc5434a079a7959d3d.jpg)  
(b) Second dimer branch: error of the compact symbolic-regression correction.

Figure 3: Symbolic-regression correction for the two dimer branches as functions of the separation distance κ. The symbolic formulas preserve the accuracy ofthe learned residual correction while providing explicit expressions involving the asymptotic eigenvalues, branch splitting, and the expected powers of δ and q .

The improvement factors are smaller than in the single-resonator case. This is expected: the dimer residual contains several sources of error at once, including the self-interaction error inherited from the single-resonator approximation, the error in the interaction blocks of the reduced operator, and the sensitivity of the two resonance branches to the eigenvalue splitting. Moreover, small changes in κ can modify the coupling strength substantially. The dimer results should therefore be interpreted as a more demanding interaction test rather than as a direct confirmation of the single-resonator behaviour.

We also examine the dimer errors as functions of the size parameter δ. While the separation distance κ controls the strength of the interaction between the two resonators, the parameter δ is the small parameter in the subwavelength asymptotic regime. This provides a complementary view to the previous plots: the κ-dependence describes the coupling efects, whereas the δ-dependence shows how the approximation behaves with respect to the asymptotic limit.

Figure 4a shows the relative errors for both dimer branches as functions of δ. The baseline asymptotic approximation exhibits a clear increase in error as δ grows, consistently with the fact that the asymptotic expansion is most accurate in the small-δ regime. The physics-informed Ridge correction reduces the error across the full range of δ, although the corrected errors display a broader scatter than in the single-resonator case. This is expected, since in the dimer configuration the residual depends not only on $\delta ,$ but also on the separation distance κ, the damping parameter γ and the resonance branch.

The symbolic correction remains systematically below the Ridge correction over the sampled range of δ, showing that the symbolic formulas capture additional structure in the residual. The remaining scatter reflects the fact that the dimer residual is genuinely multi-parameter: samples with similar values of δ may correspond to diferent coupling strengths and therefore to diferent correction magnitudes. In Figure 4b, the joint reconstruction of the two branches in the complex plane is shown. The symbolic predictions align closely with the reference resonances for both branches. Quantitatively, the mean relative error is reduced from approximately $1 . 2 2 \times 1 0 ^ { - 4 }$ for the asymptotic model to approximately $8 . 3 5 \times 1 0 ^ { - 7 }$ for the compact symbolic formulas.

![](images/6901c47c952552d5565180f79b908829d93a1193606152b90a9cb73f77b6d8db.jpg)  
(a) Relative error versus the size parameter δ.

![](images/4e3acba716131ec13365b72a713b0b3631dd39ad8ac4c97342dda4bdb9658286.jpg)  
(b) Reconstruction of both dimer branches in the complex plane.

Figure 4: Dimer prediction results. Left: relative error of the asymptotic approximation and of the corrected models as a function of the asymptotic parameter δ, with both resonance branches included. Right: symbolicregression reconstruction of the two resonance branches in the complex plane. The corrected models reduce the baseline error across the sampled range, while the broader scatter compared with the single-resonator case reflects the additional dependence on the separation distance κ, damping and branch coupling.

## 6.4 Ablation study

We now assess the role of feature design more systematically. The purpose of this ablation study is to determine whether the improvement comes merely from increasing the number of input variables, or from adding variables that reflect the asymptotic structure of the resonance problem.

We compare the following nested feature sets:

• M0: raw parameters only, namely $( \delta , \gamma )$ for the single resonator and $( \delta , \gamma , \kappa )$ for the dimer;

• M1: raw parameters together with logarithmic transforms;

• M2: asymptotic resonance or spectral quantities, such as $\omega _ { \mathrm { a s y m p } } , \lambda _ { \mathrm { a s y m p } }$ , and, for the dimer, the branch splitting;

• M3: algebraic asymptotic scales, such as $\delta ^ { 2 } , \delta ^ { 2 } \log \delta ,$ , and $\gamma \delta ^ { 2 }$ ;

• M4: frequency-dependent logarithmic scales, such as $\delta ^ { 2 } q ^ { 2 } , \delta ^ { 2 } q ^ { 2 }$ log δ, and $\delta ^ { 2 } q ^ { 2 }$ log q.

The results are reported in Table 3. For the single resonator, raw parameters already improve the baseline, reducing the mean relative error from $9 . 0 9 \times 1 0 ^ { - 4 }$ to $1 . 2 8 \times 1 0 ^ { - 4 }$ . Adding logarithmic transforms and the asymptotic resonance further improves the prediction. However, the dominant gain occurs when the algebraic asymptotic scales are introduced: the mean error drops from $8 . 2 3 \times 1 0 ^ { - 6 }$ for M2 to $1 . 8 2 \times 1 0 ^ { - 7 }$ for M3. The full feature set M4, which includes the frequency-dependent logarithmic scales, gives the lowest mean error, $1 . 2 5 \times 1 0 ^ { - 7 }$

The same hierarchy is observed for the dimer. The raw and logarithmic parameter sets M0 and M1 provide moderate improvements, while the inclusion of asymptotic spectral quantities in M2 further reduces the error. The main reduction again occurs when algebraic asymptotic scales are added: the mean error decreases from $2 . 2 1 \times 1 0 ^ { - 5 }$ for M2 to $3 . 6 7 \times 1 0 ^ { - 6 }$ for M3. Adding the frequency-dependent logarithmic scales in M4 gives an additional improvement, reaching $2 . 1 8 \times 1 0 ^ { - 6 }$

This progression shows that the correction is not improved merely because more variables are supplied to the regression model. Rather, the improvement is driven by asymptotically meaningful information. In particular, the sharp decrease from M2 to M3, followed by the additional improvement from M3 to M4, supports the claim that the learned residual is organized by the same scales that appear in the asymptotic expansion.
<table><tr><td>Configuration</td><td>Model</td><td>Features</td><td>Median</td><td> $9 0 \mathrm { t h } \ \mathrm { p e r c . }$ </td></tr><tr><td>Single</td><td>Asymptotic baseline</td><td>0</td><td> $5 . 8 5 \times 1 0 ^ { - 4 }$ </td><td> $\overline { { 2 . 2 8 \times 1 0 ^ { - 3 } } }$ </td></tr><tr><td>Single</td><td>M0: raw</td><td>2</td><td> $1 . 2 6 \times 1 0 ^ { - 4 }$ </td><td> $2 . 1 2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Single</td><td> $M 1 \colon \mathrm { r a w } + \mathrm { l o g s }$ </td><td>4</td><td> $3 . 8 1 \times 1 0 ^ { - 5 }$ </td><td> $6 . 3 8 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Single</td><td>M2: + asymptotic resonance</td><td>6</td><td> $6 . 9 5 \times 1 0 ^ { - 6 }$ </td><td> $1 . 4 9 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Single</td><td>M3: + algebraic scales</td><td>9</td><td> $1 . 4 2 \times 1 0 ^ { - 7 }$ </td><td> $3 . 7 3 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Single</td><td>M4: + frequency-log scales</td><td>12</td><td> $8 . 1 9 \times 1 0 ^ { - 8 }$ </td><td> $2 . 6 1 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Dimer</td><td>Asymptotic baseline</td><td>0</td><td> $5 . 1 6 \times 1 0 ^ { - 5 }$ </td><td> $\overline { { 3 . 4 3 \times 1 0 ^ { - 4 } } }$ </td></tr><tr><td>Dimer</td><td>M0: raw</td><td>3</td><td> $2 . 7 7 \times 1 0 ^ { - 5 }$ </td><td> $8 . 1 9 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Dimer</td><td> $M 1 \colon \mathrm { r a w } + \mathrm { l o g s }$ </td><td>6</td><td> $1 . 9 4 \times 1 0 ^ { - 5 }$ </td><td> $7 . 1 9 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Dimer</td><td>M2: + asymptotic spectrum</td><td>13</td><td> $1 . 2 3 \times 1 0 ^ { - 5 }$ </td><td> $5 . 4 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Dimer</td><td> $M 3 \mathrm { : + a l g e b r a i c s c a l e s }$ </td><td>16</td><td> $2 . 5 9 \times 1 0 ^ { - 6 }$ </td><td> $7 . 7 4 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Dimer</td><td> $M 4 \mathrm { : + f r e q u e n c y - l o g ~ s c a l e s }$ </td><td>22</td><td> $1 . 5 4 \times 1 0 ^ { - 6 }$ </td><td> $4 . 6 8 \times 1 0 ^ { - 6 }$ </td></tr></table>

Table 3: Ablation study for the Ridge residual correction. The table reports the distribution of relative errors on the held-out test sets. The nested feature hierarchy shows that the main improvement is obtained when asymptotic and physics-informed scales are added, rather than merely by increasing the number of input variables.

## 6.5 Stability of the symbolic expressions

Symbolic regression involves a stochastic search over a large space of candidate expressions. Therefore, a single symbolic formula should not be overinterpreted as a unique correction law. To assess the stability of the symbolic structure, we repeat the assisted symbolic-regression procedure over ten independent runs with diferent random seeds. For each run, symbolic regression is applied separately to the real and imaginary parts of the residual. In the dimer case, it is also applied separately to the two resonance branches.

We focus on the two assisted strategies introduced above. In the first one, called asymptotic preconditioning, a next-order proxy is first fitted using the asymptotic feature family $\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } , \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 }$ log $\delta , \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 }$ log q<sub>asymp</sub> and symbolic regression is then applied to the remaining residual. In the second one, called asymptotic feature augmentation, the proxy is supplied as an additional input variable and symbolic regression is free to use it or ignore it.

For each selected expression, we record whether the proxy, the asymptotic logarithmic feature family, and the main remaining symbolic structures appear in the formula. The results are summarized in Table 4. For the single resonator, the counts for symbolic expressions are taken over 20 formulas, corresponding to 10 runs times real and imaginary parts. For the dimer, they are taken over 40 formulas, corresponding to 10 runs times two branches times real and imaginary parts.

The table distinguishes two notions of stability. In the preconditioned setting, the next-order proxy is present by construction. Therefore, the logarithmic feature family $\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 }$ log δ, $\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log q _ { \mathrm { a s y m p } }$ appears in every run. This confirms that the preconditioned symbolic correction consistently incorporates the asymptotically motivated next-order scale. The remaining symbolic correction is then fitted on top of this proxy, and its most stable structure is the $\delta ^ { 2 } \mathrm { - t y p e }$ contribution, which appears in every run for both configurations.

The feature-augmentation setting is more selective, because PySR is given the proxy as an input but is not forced to use it. In this case, the proxy is selected in $1 4 / 2 0$ single-resonator expressions and $1 5 / 4 0$ dimer expressions. The algebraic frequency-dependent scale $\delta ^ { 2 } q ^ { 2 }$ is also selected frequently. The individual logarithmic variables are selected less often, especially $\delta ^ { 2 } q ^ { 2 } \log \delta$ . This indicates that, over the sampled parameter range, symbolic regression may represent part of the logarithmic contribution through the proxy or through simpler algebraic combinations. Thus the logarithmic terms are stable in the asymptotically preconditioned correction, while unconstrained selection within the augmented feature set tends to favor more compact representatives.

<table><tr><td>Strategy</td><td>Feature or structure</td><td>Single</td><td>Dimer</td></tr><tr><td rowspan="4">Feature augmentation</td><td>proxy</td><td>14/20</td><td>15/40</td></tr><tr><td> $\delta ^ { 2 } q ^ { 2 }$ </td><td>14/20</td><td>21/40</td></tr><tr><td> $\delta ^ { 2 } q ^ { 2 } \log \delta$ </td><td>1/20</td><td>3/40</td></tr><tr><td> $\delta ^ { 2 } q ^ { 2 } \log q$ </td><td>6/20</td><td>13/40</td></tr><tr><td rowspan="4">Preconditioning</td><td> $\mathrm { p r o x y }$ </td><td>20/20</td><td>40/40</td></tr><tr><td> $\delta ^ { 2 } q ^ { 2 }$ </td><td>20/20</td><td>40/40</td></tr><tr><td> $\delta ^ { 2 } q ^ { 2 } \log \delta$ </td><td>20/20</td><td>40/40</td></tr><tr><td> $\delta ^ { 2 } q ^ { 2 } \log q$ </td><td>20/20</td><td>40/40</td></tr><tr><td rowspan="4">Remainder structure</td><td> $\overline { { \delta ^ { 2 } \mathrm { - } \mathrm { t y p e } } }$  terms</td><td>20/20</td><td>40/40</td></tr><tr><td> $\gamma \delta ^ { 2 } \mathrm { - t y p e }$  terms</td><td>8/20</td><td>8/40</td></tr><tr><td> $\omega _ { \mathrm { a s y m p } ^ { - } } \mathrm { d e p e n d e n t }$  terms</td><td>12/20</td><td></td></tr><tr><td>spectral/splitting terms</td><td></td><td>15/40, 9/40</td></tr></table>

Table 4: Stability of the assisted symbolic-regression expressions. The counts are computed over ten independent runs, corresponding to 20 symbolic expressions for the single resonator and 40 expressions for the dimer. The table reports how often each feature family or symbolic structure appears in the selected expressions. In the dimer case, the spectral/splitting row reports the frequencies of $\lambda _ { \mathrm { a s y m p } } - d e p e n a$ ent terms and splitting-dependent terms, respectively.

The prediction errors across independent runs are reported in Table 5. The feature-augmentation strategy gives the most stable accuracy, with mean relative errors of order $1 0 ^ { - 7 }$ for the single resonator and $1 0 ^ { - 6 }$ for the dimer. The preconditioned strategy has a much smaller median error than mean error, indicating that most runs perform well but a few runs produce less accurate symbolic remainders. The proxy alone is less accurate, which shows that the asymptotic next-order family captures an important part of the residual but still benefits from a learned symbolic correction.
<table><tr><td>Configuration</td><td>Strategy</td><td>Mean</td><td>Median</td><td>Standard deviation</td></tr><tr><td>Single</td><td>next-order proxy</td><td> $\overline { { 7 . 3 1 \times 1 0 ^ { - 3 } } }$ </td><td> $\overline { { 1 . 8 5 \times 1 0 ^ { - 5 } } }$ </td><td> $\overline { { 1 . 9 7 \times 1 0 ^ { - 2 } } }$ </td></tr><tr><td>Single</td><td>asymptotic preconditioning</td><td> $5 . 0 7 \times 1 0 ^ { - 5 }$ </td><td> $1 . 5 4 \times 1 0 ^ { - 6 }$ </td><td> $1 . 2 2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Single</td><td>feature augmentation</td><td> $1 . 6 6 \times 1 0 ^ { - 7 }$ </td><td> $7 . 5 3 \times 1 0 ^ { - 8 }$ </td><td> $1 . 5 3 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Dimer</td><td>next-order proxy</td><td> $3 . 4 5 \times 1 0 ^ { - 3 }$ </td><td> $9 . 6 8 \times 1 0 ^ { - 5 }$ </td><td> $5 . 2 6 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Dimer</td><td>asymptotic preconditioning</td><td> $1 . 1 0 \times 1 0 ^ { - 4 }$ </td><td> $1 . 8 6 \times 1 0 ^ { - 5 }$ </td><td> $1 . 7 6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Dimer</td><td>feature augmentation</td><td> $4 . 8 7 \times 1 0 ^ { - 6 }$ </td><td> $3 . 1 8 \times 1 0 ^ { - 6 }$ </td><td> $4 . 1 1 \times 1 0 ^ { - 6 }$ </td></tr></table>

Table 5: Prediction errors for the assisted symbolic-regression stability study over ten independent runs. The reported quantities are computed from the mean relative error of each run.

Overall, the stability study supports the interpretation used throughout the paper. The asymptotic logarithmic family is not merely an arbitrary collection of input variables: it provides a stable next-order proxy when the symbolic regression is guided by the asymptotic structure. At the same time, the individual symbolic formulas are not unique, and PySR may choose diferent but comparably accurate algebraic representations across runs. This reinforces the view that symbolic regression should be interpreted here as a tool for discovering compact data-driven correction laws consistent with the asymptotic structure, rather than as a rigorous derivation of the next asymptotic term.

## 6.6 Assisted symbolic regression

Finally, we test the assisted symbolic-regression strategy described in Section 5. The goal is to help symbolic regression by explicitly providing information about the expected next-order asymptotic structure. We first construct an asymptotic proxy for the next-order correction using the feature family $\delta ^ { 2 } \bar { q } _ { \mathrm { a s y m p } } ^ { 2 }$ log $\delta , \delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 }$ log q<sub>asymp</sub>, $\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 }$ . We then compare three strategies: using this proxy alone, asking symbolic regression to learn only the remaining discrepancy after subtracting the proxy, and giving the proxy as an additional input variable to symbolic regression. These correspond respectively to the next-order proxy, asymptotic preconditioning, and asymptotic feature augmentation.

Figure 5 shows the resulting errors as functions of the size parameter δ, for the single resonator and for the dimer. In the single-resonator case, the asymptotic proxy already improves the baseline approximation, reducing the mean relative error from approximately ${ \bar { 9 } } . 0 9 \times { \bar { 1 0 } } ^ { - { \bar { 4 } } }$ to approximately $1 . 8 3 \times 1 0 ^ { - 5 }$ . Symbolic regression on the remaining correction further reduces the error to approximately $1 . 8 9 \times 1 0 ^ { - 1 1 }$ , while giving the proxy as an additional input gives approximately $2 . 2 2 \times 1 0 ^ { - 1 1 }$ . Thus, once the next-order asymptotic structure is made explicit, symbolic regression is able to recover almost all of the remaining residual.

The same behaviour is observed for the dimer, although the error cloud is broader. This is expected because the dimer residual depends not only on δ, but also on the separation distance κ, on the damping parameter, and on the splitting between the two branches. In this case, the asymptotic proxy reduces the mean relative error from approximately $1 . 2 2 \times 1 0 ^ { - 4 }$ to approximately $8 . 2 7 \times 1 0 ^ { - 6 }$ . Symbolic regression on the remaining correction gives approximately $6 . 0 4 \times 1 0 ^ { - 1 0 }$ , while using the proxy as an input feature gives approximately $9 . 3 0 \times 1 0 ^ { - 1 1 }$ The feature-augmentation strategy therefore gives the lowest mean error in the dimer case.

![](images/b7adf60dc6b39bfcc5668b8b09e92246ae3d3449f479dad283246d82649eeef5.jpg)  
(a) Single resonator.

![](images/6c17125b64bc3289e4282777a98cbf301f94c2a1dc75f1dbb396b9f1c30bf67d.jpg)  
(b) Dimer, with both resonance branches included.  
Figure 5: Assisted symbolic regression using a next-order asymptotic proxy. The proxy already improves the asymptotic or reduced approximation, and symbolic regression further reduces the remaining residual by several orders of magnitude. The dimer plot is more dispersed because the correction also depends on the separation distance and on branch splitting.

These results show that symbolic regression is most efective when it is guided by the asymptotic structure. The next-order proxy captures a substantial part of the missing correction, while the symbolic step learns the remaining discrepancy. The comparison between asymptotic preconditioning and asymptotic feature augmentation also shows that the asymptotic information can be injected either through the target or through the input representation. Both strategies perform very well for the single resonator, while feature augmentation is slightly more efective for the dimer. This supports the interpretation that the learned residual is governed by the same scales as the neglected asymptotic terms, even in the presence of coupling and branch splitting.

## 6.7 Robustness with respect to noisy training data

We now investigate the robustness of the correction models with respect to noise in the reference resonances. This experiment is motivated by the fact that, in practical settings, the values of $\omega _ { \mathrm { r e f } }$ may come from numerica solvers or measurements and may therefore contain errors.

We also test the efect of perturbing the reference resonances used for training. Figure 6 shows that the corrections remain stable for moderate noise levels and degrade once the noise becomes comparable to the residual being learned. The dimer is more sensitive, as expected from its branch-dependent coupling corrections. The precise noise model, noise levels, and dataset sizes are given in Appendix A.

A clear transition occurs between $\sigma = 1 0 ^ { - 3 }$ and $\sigma = 1 0 ^ { - 2 }$ . For $\sigma \leq 1 0 ^ { - 3 }$ , the corrected models remain well below the asymptotic error, showing that the residual-learning strategy is robust to moderate perturbations of the training data. At $\sigma = 1 0 ^ { - 2 }$ , the noise dominates the residual and the learned corrections deteriorate substantially. This behaviour is expected: once the noise level is larger than the correction itself, the training target no longer contains enough reliable information to identify the residual accurately.

The single-resonator case is more stable than the dimer case. In the dimer, the residual contains both self-interaction corrections and branch-dependent coupling efects, making the learning problem more sensitive to perturbations. Nevertheless, the same qualitative behaviour is observed in both configurations. The noisetrained symbolic correction follows the same degradation pattern as the Ridge and proxy-based corrections, confirming that symbolic regression can also be retrained from noisy residuals when suficiently large datasets are available. The next-order proxy and the Ridge model augmented with the proxy feature behave similarly, reflecting the fact that both are constrained by the same asymptotic structure.

![](images/fe195936cc68714f68a2f0332c5758cb30633dd2267848ac40c3bf56d4864184.jpg)  
(a) Single resonator.

![](images/3d7996f377e9836b7f234b79df9d9f411121d2e8cbbef5d0177f2815f2a492a9.jpg)  
(b) Dimer.  
Figure 6: Robustness with respect to noisy training resonances. The corrections remain stable for moderate noise levels and deteriorate when the noise becomes comparable to the residual.

## 6.8 Efect of the number of training measurements

We finally investigate how the robustness of the diferent correction strategies depends on the number of available training measurements. For each training-set size, the models are trained under noisy resonances and evaluated on a fixed held-out test set. The same relative noise levels as before are considered, namely $\sigma \in \{ 0 , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } \}$ . This experiment complements the previous noise study by separating the efect of the noise amplitude from the efect of the amount of training data.

Figure 7 shows the behaviour of the Ridge residual correction. Increasing the number of training measurements significantly improves the prediction under noisy data. The improvement is especially visible for the larger noise levels, where averaging over more samples stabilizes the learned residual. For clean data, the error is already close to its limiting value and therefore changes only mildly with the training size.

![](images/bcc1f2ffe3c3e1c8642f19c47d65753f7770962ff905596f1a3c055358749c4e.jpg)

![](images/c340d95374d5d131091ea6a8340545491cd65699344fdbbb9ddc9f0c8b88485c.jpg)  
Figure 7: Efect of the number of training measurements on the Ridge residual correction. The error decreases as the amount of training data increases, especially for noisy measurements.

The corresponding result for the next-order proxy is shown in Figure 8. The same qualitative behaviour is observed: the proxy becomes more stable as the number of measurements increases. This is consistent with the fact that the proxy is calibrated from noisy residual data using asymptotically motivated features. The efect is again most pronounced for the larger noise levels.

Figure 9 reports the result obtained when the next-order proxy is supplied as an additional input to the Ridge model. This assisted correction is also improved by larger training sets. In the dimer case, the gain is particularly clear, reflecting the fact that the coupled problem requires more data to stabilize the branchdependent correction.

Finally, Figure 10 shows the same experiment for symbolic regression, where PySR is retrained independently for each training size and each noise level. Increasing the symbolic-training set improves the rediscovered formulas, especially for $\sigma = 1 0 ^ { - 3 }$ and $\sigma = 1 0 ^ { - 2 }$ . The curves are not expected to be perfectly monotone, since symbolic regression solves a non-convex model-selection problem and each run may discover a diferent analytic expression. Nevertheless, the overall trend shows that larger datasets make symbolic rediscovery more stable under noise.

![](images/becd3773b561e765fe39160c87646fe216615fa8c004bc24f65fd9bf697bb059.jpg)

![](images/b33e6c591587f1181201ce7516cc63c4b3a2be14e89d257bede7bee9439ff557.jpg)

Figure 8: Efect of the number of training measurements on the next-order asymptotic proxy. Increasing the training size improves the stability of the calibrated proxy under noisy measurements.  
![](images/197c3d320e7add0b5cde7e366020b6d095a878d11dd683bac75f81ff5725f792.jpg)

![](images/60889c70063b846dfbb286daf1ab1d5b6ccf5c2e1f843dfdf6c92eb261a4feee.jpg)  
Figure 9: Efect of the number of training measurements on the Ridge correction assisted by the next-order proxy. The proxy feature improves the stability of the learned correction, with a particularly visible gain in the dimer case.

Overall, these experiments confirm that the efect of measurement noise can be mitigated by increasing the number of training samples. Ridge-based corrections benefit from this most directly, while symbolic regression also becomes more stable when more data are available, although its non-convex search may lead to mild non-monotonicity.

## 6.9 Summary

Across both the single-resonator and dimer configurations, the proposed corrections substantially improve the asymptotic or reduced approximations. Ridge regression provides a stable residual model, while symbolic regression yields compact analytic formulas that retain high accuracy and expose the structure of the correction. The ablation studies, now performed for both configurations, show that this improvement is not simply due to adding more variables, but relies on features that reflect the asymptotic structure of the problem. In particular, the logarithmic frequency-dependent features $\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log \delta$ and $\bar { \delta } ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 } \log q _ { \mathrm { a s y m p } }$ play an important role in capturing the next-order behaviour.

The assisted symbolic-regression experiments further confirm this interpretation. For both the single resonator and the dimer, an explicit next-order proxy already reduces the asymptotic error, and symbolic regression then captures the remaining discrepancy. In the dimer case, the error distribution is broader because the correction depends not only on δ, but also on the separation distance κ and on the splitting between the two resonant branches. Nevertheless, the same hierarchy of models is observed, showing that the learned residual remains organized by the expected asymptotic scales.

![](images/9b09e5a9ee18245a5c63f83acec2340384bff974da17d34619cb65c5371c4901.jpg)  
number of symbolic-training measurements number of symbolic-training measurement  
Figure 10: Efect of the number of symbolic-training measurements on PySR rediscovery under noisy data. For each training size and noise level, symbolic regression is rerun from the noisy residuals. Larger training sets improve the robustness of the rediscovered symbolic correction, especially at higher noise levels.

The noise experiments show that the residual-learning framework is robust to moderate perturbations of the training resonances, especially when the number of training measurements is increased. Ridge and proxy-based corrections degrade gradually as the noise level grows, and remain efective as long as the noise is not larger than the residual being learned. The dimer case is more sensitive than the single-resonator case, reflecting the additional dificulty of learning coupled, branch-dependent corrections.

Finally, the noisy-test experiment distinguishes between using a symbolic formula discovered from clean data and rediscovering a symbolic formula from noisy data. In both the single-resonator and dimer cases, the clean symbolic formula performs slightly better. This suggests that PySR is most efective as a tool for discovering clean structural corrections, rather than as a procedure for fitting noisy measurements directly.

## 7 Discussion and perspectives

In this work, we developed an asymptotics-guided learning framework for resonance prediction in dispersive resonator systems. The central point is that asymptotic analysis is useful not only because it provides a leading-order approximation, but also because it identifies the variables and scales that organize the error of this approximation. We therefore use the asymptotic model as a baseline and learn only the residual between this prediction and the numerical reference resonance.

The numerical results show that this residual is highly structured. For both the single resonator and the dimer, physics-informed features derived from the subwavelength expansion substantially improve the prediction compared with raw-parameter learning. In particular, the logarithmic frequency-dependent scales suggested by the two-dimensional operator expansion provide efective features for correcting the missing higher-order terms. The dimer case is more demanding, since the residual also contains branch-dependent interaction efects, but the same asymptotics-guided strategy remains efective.

Symbolic regression adds an interpretable layer to this correction procedure. The formulas obtained should not be interpreted as rigorous asymptotic expansions; they are data-driven correction laws valid in the sampled parameter regime. Their significance is that they recover compact expressions organized around the asymptotic scales, including size-dependent, frequency-dependent, damping-dependent, and branch-splitting terms. The assisted symbolic-regression and stability experiments further support the view that symbolic regression is mos efective when guided by the asymptotic structure rather than used as a purely black-box search.

Overall, the results suggest a general methodology: use asymptotic analysis to identify the structure of the residual, use learning to correct the missing terms, and use symbolic regression to extract compact interpretable formulas. This provides a bridge between asymptotic modelling and data-driven prediction for nonlinear spectral problems in dispersive media.

## A Numerical and learning details

This appendix summarizes the numerical choices used to generate the datasets and reproduce the learning experiments. All computations are implemented in Python. Data handling uses NumPy and pandas, Ridge regression and feature standardization use scikit-learn, and symbolic regression is performed with PySR, interfaced with SymbolicRegression.jl. Random seeds are fixed throughout the experiments.

## A.1 Parameter ranges and dataset construction

All parameter values in this appendix are non-dimensional. For the single-resonator experiments, the size parameter and damping parameter are sampled logarithmically $\delta \sim 1 0 ^ { U ( - 2 , - 1 / 2 ) }$ and $\gamma \stackrel { \cdot } { \sim } 1 0 ^ { U ( - 3 , - 1 ) }$ , where $U ( a , b )$ denotes the uniform distribution on the interval [a, b]. Thus, $\delta \in [ 1 0 ^ { - 2 } , 1 0 ^ { - 1 / 2 } ]$ and $\gamma \in [ 1 0 ^ { - 3 } , 1 0 ^ { - 1 } ]$ In the final single-resonator dataset used for the main plots, the realized range is approximately $\delta \in [ 5 . 2 5 \times$ $1 0 ^ { - 2 } , 3 . 1 6 \times 1 0 ^ { - 1 } ]$ and $\gamma \in [ 1 . 0 0 \times 1 0 ^ { - 3 } , 9 . 9 9 8 \times 1 0 ^ { - 2 } ]$ . The lower end of the realized δ-range is larger than $1 0 ^ { - 2 }$ because samples outside the retained physical branch are discarded.

For the dimer experiments, the size parameter, damping parameter, and separation distance are sampled as $\delta \sim 1 0 ^ { U ( - 2 , - 1 / 2 ) } , \gamma \sim 1 0 ^ { \dot { U } ( - 3 , - 1 ) }$ and $\kappa \sim 1 0 ^ { U ( - 2 , 0 ) }$ . Hence, $\kappa \in [ 1 0 ^ { - 2 } , 1 ]$ . In the final dimer dataset, the realized ranges are approximately $\delta \in [ 1 . 0 1 \times 1 0 ^ { - 2 } , 3 . 1 6 \times 1 0 ^ { - 1 } ] , \ : \ : \forall \in [ 1 . 0 0 \times 1 0 ^ { - 3 } , 9 . 9 5 \times 1 0 ^ { - 2 } ]$ , and $\kappa \in [ 1 . 0 0 \times 1 0 ^ { - 2 } , 9 . 9 9 \times 1 0 ^ { - 1 } ]$

Samples for which the nonlinear resonance solver fails to converge are discarded. We retain only outgoing damped resonances satisfying Re ${ \ ; } \omega > 0$ and Im $\omega < 0$

## A.2 Reference and reduced models

For the single resonator, the asymptotic resonance is computed from the leading asymptotic eigenvalue of the volume integral operator. The reference resonance is computed from a more accurate numerical value of the same eigenvalue.

For the dimer, the reduced and reference eigenvalues are computed using two discretization levels of the two-resonator integral operator. The reduced model uses a coarser discretization, while the reference model uses a finer discretization. In the reported computations, the coarse dimer operator uses 25 sample points per resonator and the reference operator uses 120 sample points per resonator. The two resulting resonances are treated as the two dimer branches.

## A.3 Dimer discretization convergence

For the dimer configuration, the reference resonances are obtained from a discretization of the two-resonator volume integral operator. Since the learned correction is trained against this numerical reference, it is important to check that the reference discretization is suficiently stable. We therefore perform a convergence study with respect to the number N of quadrature points per resonator.

For each value $N \in \{ 2 5 , 5 0 , 8 0 , 1 2 0 , 1 6 0 , 2 0 0 \}$ , we compute the two leading dimer resonances and compare them with the finest discretization $N _ { \mathrm { r e f } } = 2 0 0$ . The reported quantity is $\lvert \omega _ { N } - \omega _ { N _ { \mathrm { r e f } } } \rvert .$ computed over the two branches and over a representative set of dimer configurations sampled from the same parameter ranges as in the main experiments.

<table><tr><td>N</td><td>Mean</td><td>Median</td><td>90th percentile</td></tr><tr><td>25</td><td> $\overline { { 3 . 5 6 \times 1 0 ^ { - 4 } } }$ </td><td> $\overline { { 8 . 2 9 \times 1 0 ^ { - 5 } } }$ </td><td> $\overline { { 1 . 1 8 \times 1 0 ^ { - 3 } } }$ </td></tr><tr><td>50</td><td> $1 . 7 1 \times 1 0 ^ { - 4 }$ </td><td> $3 . 8 8 \times 1 0 ^ { - 5 }$ </td><td> $5 . 7 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>80</td><td> $9 . 6 8 \times 1 0 ^ { - 5 }$ </td><td> $2 . 2 6 \times 1 0 ^ { - 5 }$ </td><td> $3 . 2 1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>120</td><td> $3 . 4 4 \times 1 0 ^ { - 5 }$ </td><td> $6 . 9 7 \times 1 0 ^ { - 6 }$ </td><td> $1 . 1 4 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>160</td><td> $1 . 6 9 \times 1 0 ^ { - 5 }$ </td><td> $3 . 5 3 \times 1 0 ^ { - 6 }$ </td><td> $5 . 5 3 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>200</td><td>0</td><td>0</td><td>0</td></tr></table>

Table 6: Dimer discretization convergence. The errors are measured relative to the finest discretization $N _ { \mathrm { r e f } } =$ 200, using $\lvert \omega _ { N } - \omega _ { N _ { \mathrm { r e f } } } \rvert _ { \mathrm { ; } }$ , and are aggregated over both resonance branches.

The error decreases systematically as N increases. In particular, the mean error drops from approximately $3 . 5 6 \times 1 0 ^ { - 4 }$ at $N = 2 5$ to approximately $1 . 6 9 \times 1 0 ^ { - 5 }$ at $N = 1 6 0$ . This indicates that the reference discretization used in the learning experiments is stable enough for the observed learned corrections not to be merely an artefact of a coarse discretization.

![](images/0c74e9e177ec0b2f6eea31029fee7bb785da47fed30d9136f4de903897f21b07.jpg)  
Figure 11: Dimer discretization convergence as a function of the number N of quadrature points per resonator. The reference discretization is $N _ { \mathrm { r e f } } = 2 0 0$

## A.4 Train-test splits and validation protocols

Unless otherwise stated, models are evaluated using an 80–20 train-test split with fixed random seed. This random split is used for the in-distribution tests. For out-of-distribution validation, the split is performed by holding out a challenging region of parameter space. For the single resonator, we test extrapolation to large δ and large γ. For the dimer, we test extrapolation to large δ and small κ. The holdout threshold is taken at the corresponding 80% quantile, except for the small-κ dimer test, where the lower 20% of κ-values is held out. For a test sample j, the relative error $E _ { j }$ and the reported mean relative error E are given in (6.4). For the dimer, the average is taken over both resonance branches.

For the main prediction figures, the single-resonator dataset contains 10 000 samples, split into 8 000 training samples and 2 000 test samples. The dimer dataset contains 3 000 configurations, split into 2 400 training configurations and 600 test configurations. For the large-data noisy-training experiments, the single-resonator dataset contains 12 500 samples, split into 10 000 training samples and 2 500 test samples. The dimer noisy training experiment uses 5 000 training configurations and 1 250 test configurations.

## A.5 Ridge regression

The Ridge model is trained on the residual

$$
\Delta \omega = \omega _ { \mathrm { r e f } } - \omega _ { \mathrm { a s y m p } } ,
$$

rather than on the full resonance. The real and imaginary parts of the residual are learned simultaneously.

The Ridge objective is

$$
\operatorname* { m i n } _ { W , b } \| Y - X W - b \| _ { 2 } ^ { 2 } + \alpha _ { R } \| W \| _ { 2 } ^ { 2 } ,
$$

with $\alpha _ { R } = 1 0 ^ { - 8 }$ . Before training, all input features are standardized using the empirical mean and standard deviation of the training set. The same transformation is then applied to the test set. We do not tune α<sub>R</sub> by cross-validation; the objective is to test whether a simple and stable linear residual model can exploit the asymptotic feature structure, rather than to optimize a large hyperparameter search. For the calibrated nextorder proxy, we use a smaller Ridge parameter, $\alpha _ { R } = 1 0 ^ { - 1 2 }$ , since this step is used as a least-squares calibration of the candidate asymptotic feature family.

## A.6 Feature sets

The raw feature sets contain logarithmic transforms of the sampled parameters, such as log δ, log γ and, in the dimer case, log κ. The physics-informed feature sets include combinations motivated by the asymptotic scaling, for example $\delta ^ { 2 } , \delta ^ { 2 }$ log $\delta , \gamma \delta ^ { 2 }$

The frequency-dependent next-order terms are motivated by the scale (3.10). In the implementation, the independent logarithmic features are therefore taken to be $\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 }$ log δ, $\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 }$ log q<sub>asymp</sub>, together with $\delta ^ { 2 } q _ { \mathrm { a s y m p } } ^ { 2 }$

In the dimer case, the frequency-dependent features are evaluated branchwise using the corresponding asymptotic resonance. Additional coupling features include the asymptotic eigenvalues and the asymptotic branch splitting.

## A.7 Symbolic regression

Symbolic regression is used to obtain compact closed-form corrections for the residual. Separate formulas are fitted for the real and imaginary parts. In the dimer case, symbolic regression is applied separately to each branch.

For the clean-data PySR runs used to obtain the symbolic formulas reported in the main text, the allowed binary operators are $+ , \quad - , \quad \times$ , and no unary operators are used. The model selection criterion is $\mathrm { P y S R } '$ s best rule. The maximum expression size is set to 20, with 20 populations and 500 iterations. Fixed random seeds are used.

For noisy symbolic rediscovery tests, PySR is rerun with the operator set $+ , \quad - , \quad \times , \quad / ,$ maximum expression size 20, population size 40, deterministic serial execution, and 100 iterations unless otherwise specified. These runs are used only for robustness diagnostics. The compact symbolic formulas reported in the main text are the clean-data formulas.

## A.8 Assisted symbolic regression

In the assisted symbolic-regression experiments, a calibrated next-order proxy is first fitted using the frequencydependent feature family described above. We then compare two procedures:

1. symbolic regression is applied to the remaining residual after subtracting the proxy;

2. the proxy is supplied as an additional input feature to symbolic regression.

These experiments test whether explicit next-order asymptotic information improves the symbolic search.

## A.9 Noise experiments

For the noisy-training experiments, the training resonance is perturbed according to

$$
\omega _ { \mathrm { r e f } } ^ { \mathrm { n o i s y } } = \omega _ { \mathrm { r e f } } + \sigma | \omega _ { \mathrm { r e f } } | \frac { \eta _ { 1 } + i \eta _ { 2 } } { \sqrt { 2 } } ,
$$

where $\eta _ { 1 }$ and $\eta _ { 2 }$ are independent standard Gaussian random variables. The models are trained on

$$
\omega _ { \mathrm { r e f } } ^ { \mathrm { n o i s y } } - \omega _ { \mathrm { a s y m p } } ,
$$

and evaluated on a clean held-out test set.

The noise levels are $\sigma \in \{ 0 , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } \}$ . For the Ridge and proxy-based noise experiments, each noise level is averaged over 20 independent noise realizations. For the symbolic rediscovery experiments, PySR is rerun independently for each noise level.

In the clean-versus-noise-trained symbolic experiment, the test resonances are also perturbed. This experiment compares a symbolic formula discovered from clean data with a symbolic formula rediscovered from noisy training data, both evaluated against noisy test resonances.

## References

[1] Q. A. Akkerman and L. Manna. What defines a halide perovskite? ACS Energy Letters, 5(2):604–610, 2020.

[2] K. Alexopoulos and B. Davies. Asymptotic analysis of subwavelength halide perovskite resonators. Partial Diferential Equations and Applications, 3(4):1–28, 2022.

[3] K. Alexopoulos and B. Davies. A mathematical design strategy for highly dispersive resonator systems. Mathematical Methods in the Applied Sciences, pages 1–26, 2023.

[4] H. Ammari, A. Dabrowski, B. Fitzpatrick, P. Millien, and M. Sini. Subwavelength resonant dielectric nanoparticles with high refractive indices. Mathematical Methods in the Applied Sciences, 42(18):6567– 6579, 2019.

[5] G. Carleo, I. Cirac, K. Cranmer, L. Daudet, M. Schuld, N. Tishby, L. Vogt-Maranto, and L. Zdeborov´a. Machine learning and the physical sciences. Reviews of Modern Physics, 91(4):045002, 2019.

[6] M. Cranmer. Interpretable machine learning for science with pysr and symbolicregression.jl. arXiv preprint arXiv:2305.01582, 2023.

[7] R. V. Craster and S. Guenneau. Acoustic Metamaterials: Negative Refraction, Imaging, Lensing and Cloaking, volume 166 of Springer Series in Materials Science. Springer, London, 2013.

[8] L. Gao, Y. Chai, D. Zibar, and Z. Yu. Deep learning in photonics: Introduction, 2021.

[9] A. Ghorashi, S. Vaidya, Z. Liu, C. Loh, T. Christensen, M. Tegmark, and M. Soljacic. Symbolic learning of topological bands in photonic crystals. ACS Photonics, 2026.

[10] L. Gu, S. Poddar, Y. Lin, Z. Long, D. Zhang, Q. Zhang, L. Shu, X. Qiu, M. Kam, A. Javey, et al. A biomimetic eye with a hemispherical perovskite nanowire array retina. Nature, 581(7808):278–282, 2020.

[11] R. S. Hegde. Deep learning: a new tool for photonic nanostructure design. Nanoscale Advances, 2(3):1007– 1023, 2020.

[12] S. Kim. Novel Approaches to Discovery and Optimization in Physics: Symbolic Regression, Bayesian Optimization, and Topological Photonics. PhD thesis, Massachusetts Institute of Technology, 2023.

[13] G. J. Lee, C. Choi, D.-H. Kim, and Y. M. Song. Bioinspired artificial eyes: optic components, digital cameras, and visual prostheses. Advanced Functional Materials, 28(24):1705202, 2018.

[14] W. Ma, Z. Liu, Z. A. Kudyshev, A. Boltasseva, W. Cai, and Y. Liu. Deep learning for the design of photonic structures. Nature photonics, 15(2):77–90, 2021.

[15] S. Makarov, A. Furasova, E. Tiguntseva, A. Hemmetter, A. Berestennikov, A. Pushkarev, A. Zakhidov, and Y. Kivshar. Halide-perovskite resonant nanophotonics. Advanced Optical Materials, 7(1):1800784, 2019.

[16] G. W. Milton and N. A. Nicorovici. On the cloaking efects associated with anomalous localized resonance. Proceedings of the Royal Society A, 462(2074):3027–3059, 2006.

[17] J. B. Pendry. Negative refraction makes a perfect lens. Physical Review Letters, 85(18):3966, 2000.

[18] J. Sierra-Velez, M. Inchaussandague, D. Skigin, A. Vial, H. H¨olscher, and D. Macias. Symbolic regression: an alternative method to model the optical response of photonic biological and bio-inspired structures. Optics Letters, 49(23):6753–6756, 2024.

[19] J. Sierra-Velez, A. Vial, D. Mac´ıas, M. Inchaussandague, and D. Skigin. Modeling the optical properties of biological structures using symbolic regression. Physical Review E, 112(3):034404, 2025.

[20] H. J. Snaith. Present status and future prospects of perovskite photovoltaics. Nature Materials, 17(5):372– 376, 2018.

[21] J. Sun, B. Bhushan, and J. Tong. Structural coloration in nature. RSC Advances, 3(35):14862–14889, 2013.

[22] A. Tsakyridis, M. Moralis-Pegios, G. Giamougiannis, M. Kirtas, N. Passalis, A. Tefas, and N. Pleros. Photonic neural networks and optics-informed deep learning fundamentals. APL photonics, 9(1), 2024.

[23] V. G. Veselago. The electrodynamics of substances with simultaneously negative values of ϵ and µ. Soviet Physics Uspekhi, 10(4):509–514, 1968.

[24] H. Wang, F. U. Kosasih, H. Yu, G. Zheng, J. Zhang, G. Pozina, Y. Liu, C. Bao, Z. Hu, X. Liu, et al. Perovskite-molecule composite thin films for eficient and stable light-emitting diodes. Nature Communications, 11(1):1–9, 2020.

[25] P. R. Wiecha, A. Arbouet, C. Girard, and O. L. Muskens. Deep learning in nano-photonics: inverse design and beyond. Photonics research, 9(5):B182–B200, 2021.

[26] J. Yun, S. Kim, S. So, M. Kim, and J. Rho. Deep learning for topological photonics. Advances in Physics: X, 7(1):2046156, 2022.

[27] Y. Zhao, Z. Xie, H. Gu, C. Zhu, and Z. Gu. Bio-inspired variable structural color materials. Chemical Society Reviews, 41(8):3297–3317, 2012.