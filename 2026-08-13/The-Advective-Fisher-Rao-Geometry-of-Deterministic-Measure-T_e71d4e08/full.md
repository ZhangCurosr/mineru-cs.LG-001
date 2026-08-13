# The Advective Fisher–Rao Geometry of Deterministic Measure Transport

August 13, 2026

Benjamin Gess<sup>1</sup>, Johannes Müller<sup>2,</sup>

1 Institut für Mathematik, Technische Universität Berlin & Max-Planck-Institut für Mathematik in den Naturwissenschaften, Email: benjamin.gess@tu-berlin.de

2 Institut für Mathematik, Technische Universität Berlin, Email: johannes.christoph.mueller@tu-berlin.de

## Abstract

A novel advective Fisher–Rao metric is introduced for optimization tasks on paths of probability measures governed by the continuity equation. This metric is shown to lead to optimal descent directions.

It is then shown that this metric arises naturally from three diferent perspectives: As the rescaled zero-noise limit of the Fisher–Rao metric on path measures, as the expected value of the second variation of the Freidlin–Wentzell large deviation rate functional, and as the Hessian of the Benamou–Brenier action functional from dynamic optimal transport.

We supplement this geometric construction with computational experiments. Here, we demonstrate empirically that the advective Fisher–Rao metric yields the desired optimal fitting of probability densities, whereas the Gauss–Newton method yields optimal fitting of velocity fields.

MSC2020: 35Q90, 49Q22, 53B12, 60H10, 68T07 Keywords: Flow-based generative models, Fisher–Rao metric, natural gradient, Gauss–Newton method, Wasserstein geometry

## Contents

1 Introduction 2   
1.1 Contributions and main results . 3   
1.2 Related works . . 11   
2 Preliminaries and Setup 13   
2.1 Notation and conventions . . 13   
2.2 Flows, the continuity equation, and path measures 14   
2.3 Information geometry and natural gradients 17   
3 The Advective Fisher–Rao Metric 19   
3.1 The Fisher–Rao metric of path measures of SDEs 19   
3.2 The advective Fisher–Rao metric and its optimality 24   
4 The Advective Fisher–Rao Geometry from Three Perspectives 30   
4.1 Information geometry 30   
4.2 Large deviation theory 33   
4.3 Optimal transport 37   
5 The Advective Fisher–Rao Metric on Paths of Probability Measures 41   
6 The Flat $L ^ { 2 } ( \rho _ { \bullet } ^ { \star } )$ -Geometry 51   
7 Implications for Flow-Based Generative Models 54   
7.1 Computational considerations 55   
7.2 Visualization of update directions 58   
7.3 Computational Experiments 60   
A Carathéodory Theory and Method of Characteristics 66   
B Fisher–Rao Gradient Flows and Geodesics 71

## 1 Introduction

At the heart of a variety of problems lies the task of matching a reference path of probability measures $\rho _ { \bullet } ^ { \star } = ( \rho _ { t } ^ { \star } ) _ { t \in [ 0 , 1 ] }$ , which means that one seeks to solve the problem

$$
\operatorname* { m i n } _ { \substack { \rho \bullet \in \mathrm { A C } _ { T } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) ) } } D ( \rho _ { \bullet } ^ { \star } , \rho _ { \bullet } ) ,\tag{1.1}
$$

where $\mathsf { A C } _ { T } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ denotes the set of absolutely continuous paths of probability measures with respect to the Wasserstein-2 distance and $D$ is some measure of distance. For example, this problem arises in optimal control of densities [AB10] including dynamic optimal transport and its entropic regularization in the form of Schrödinger bridges [BB00, Léo12a]. Recently, such control problems on the space of probability measures have attracted a great deal of attention in generative machine learning in the form of flow-matching and difusion models $[ \mathrm { S D W M G 1 } _ { 5 } .$ $\mathrm { H J A } _ { 2 0 , \mathrm { R B L } ^ { + } 2 2 , \mathrm { L C B H } ^ { + } 2 3 ] }$ as well as for approximate sampling methods [BRU24, $\mathrm { D E H A ^ { + } } _ { 2 4 } , \mathrm { B B R ^ { + } } _ { 2 5 } ]$ , see also Section 1.1.4 for a more detailed discussion of problems of this kind. To approximately solve this problem, it is common to employ gradient-based optimization methods to minimize the discrepancy D. This requires the choice of a Riemannian metric on the search space $\mathsf { A C } _ { T } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ , where the choice of metric has a significant influence on the optimization dynamics. A particularly favorable scenario arises if the gradient flow of D with respect to $g$ drives $\rho _ { \bullet }$ along a geodesic towards the reference path $\rho _ { \bullet } ^ { \star }$ , since it enables a global description of the optimization dynamics as well as fast global exponential convergence rates. In this case, we call the metric $g$ compatible with $D .$ . In practice, the problem (1.1) is typically not solved directly, but reformulated as an optimization problem on the space of velocity fields $v _ { \bullet }$ that control the evolution of the density path $\rho _ { \bullet }$ via (1.2) below. This is justified by the fact that any absolutely continuous curve $\rho _ { \bullet } ^ { v } \in \mathsf { A C } _ { T } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ is induced by a time-dependent velocity field $v _ { \bullet } = ( v _ { t } ) _ { t \in [ 0 , 1 ] }$ via the continuity equation

$$
\partial _ { t } \rho _ { t } ^ { v } + \nabla \cdot ( \rho _ { t } ^ { v } v _ { t } ) = 0 , \qquad \rho _ { 0 } ^ { v } = \rho _ { 0 } .\tag{1.2}
$$

Assuming that the reference path $\rho _ { \bullet } ^ { \star }$ is induced by $v _ { \bullet } ^ { \star } .$ , it is a common approach to consider the least-squares functional

$$
E ( v _ { \bullet } ) = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| v _ { t } ( x ) - v _ { t } ^ { \star } ( x ) \| ^ { 2 } \rho _ { t } ^ { \star } ( \mathrm { d } x ) \mathrm { d } t\tag{1.3}
$$

as a proxy for the original problem (1.1). Importantly, the solution map

$$
\Phi \colon L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } ( \mathbb { R } ^ { d } ) , \quad v _ { \bullet } \mapsto \rho _ { \bullet } ^ { v }
$$

of the continuity equation is neither linear nor injective, nor does it have an injective derivative. As a consequence, optimization in the search space of velocity fields is not equivalent to the space of paths of probability densities. The fact that the ultimate goal is to optimize the path of densities $p _ { \bullet } ^ { v }$ rather than the velocity field $v _ { \bullet }$ raises a natural question:

Is there a Riemannian metric g with respect to which the gradient flow of E drives $\rho _ { \bullet } ^ { v }$ along a geodesic towards the reference path $\rho _ { \bullet } ^ { \star } ?$

This question is answered positively in this work, based on the introduction of a novel geometry $^ { g , }$ the advective Fisher–Rao metric, which is shown to induce optimal gradient descent directions on the level of paths of probability measures. Notably, this metric difers from the $L ^ { 2 }$ geometry induced by (1.3) on velocities, and the latter does not lead to such optimal descent directions.

## 1.1 Contributions and main results

In this section, we first introduce the advective Fisher–Rao metric, which leads to optimal gradient descent directions on the level of paths of probability measures. We discuss how this metric appears as a canonical object from three diferent perspectives, as well as practical implications for the training of generative models.

## 1.1.1 The advective Fisher–Rao metric and its optimality

As laid out in the introduction, we are looking for a metric on the paths of probability measures induced by deterministic advection (1.2), and its corresponding flow

$$
\dot { X } _ { t } ^ { v } = v _ { t } ( X _ { t } ^ { v } ) , \qquad X _ { 0 } \sim \rho _ { 0 } .\tag{1.4}
$$

In the setting of difusive convection with difusivity $\varepsilon > 0$ , that is,

$$
\begin{array} { r } { \mathrm { d } X _ { t } ^ { v , \varepsilon } = v _ { t } ( X _ { t } ^ { v , \varepsilon } ) \mathrm { d } t + \sqrt { \varepsilon } \mathrm { d } W _ { t } , \qquad X _ { 0 } ^ { \varepsilon } \sim \rho _ { 0 } , } \end{array}\tag{1.5}
$$

with the corresponding Fokker-Planck equation

$$
\partial _ { t } \rho _ { t } ^ { v , \varepsilon } + \nabla \cdot ( \rho _ { t } ^ { v , \varepsilon } v _ { t } ) = \frac { \varepsilon } { 2 } \Delta \rho _ { t } ^ { v , \varepsilon } , \qquad \rho _ { 0 } ^ { \varepsilon } = \rho _ { 0 } ,
$$

the Fisher–Rao metric on the space of path measures ofers a canonical choice. Indeed, since for $\varepsilon > 0$ , the path measures $\mathbb { P } ^ { v , \varepsilon } = \mathcal { L } ( X _ { \cdot } ^ { v , \varepsilon } )$ for diferent choices of v are equivalent, the Fisher–Rao metric is well-defined, and induces geodesic descent directions [Ama08]. However, for $\varepsilon = 0$ the path measures are typically singular, and the Fisher–Rao metric is not defined. An important insight of this section is that, in the noisy regime $\varepsilon > 0$ , the pullback of the Fisher–Rao metric on path measures along the solution map $v \mapsto \mathbb { P } ^ { v , \varepsilon }$ induces a natural geometry on the space of velocity fields. When expressed in terms of velocity fields and rescaled appropriately, this pulled-back Fisher–Rao metric admits a nontrivial singular limit as $\varepsilon \to 0$ . This limiting structure motivates the definition of the advective Fisher–Rao metric for deterministic measure-valued flows. We then show that the resulting metric yields gradient descent directions that are optimal at the level of the induced density paths. As a first step towards the definition of a compatible metric in the absence of noise, we give an explicit expression for the pullback of the Fisher–Rao metric along the solution map of the SDE $\left( 1 . 5 \right)$

Theorem (Theorem 3.1). Let $\varepsilon > 0$ be the noise level and let $\mathbb { P } ^ { v , \varepsilon }$ denote the path measure ofthe SDE (1.4). Then, under a Novikov condition on $v _ { \bullet } , u _ { \bullet }$ , and $w _ { \bullet } .$ , the pullback of the Fisher–Rao metric along the map $v _ { \bullet } \mapsto \mathbb { P } ^ { v , \varepsilon }$ is given by

$$
g _ { v _ { \bullet } } ^ { \mathrm { F R } , \varepsilon } ( u _ { \bullet } , w _ { \bullet } ) = \frac { 1 } { \varepsilon } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } u _ { t } ( x ) \cdot w _ { t } ( x ) \rho _ { t } ^ { v , \varepsilon } ( \mathrm { d } x ) \mathrm { d } t .\tag{1.6}
$$

Based on this, we propose an extension of the Fisher–Rao metric to the setting of deterministic dynamics.

Definition (Advective Fisher–Rao metric). Consider a velocity field $v _ { \bullet } \in L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } .$ For $u _ { \bullet } , w _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 }$ we call

$$
g _ { v _ { \bullet } } ^ { \mathrm { A F R } } ( u _ { \bullet } , w _ { \bullet } ) = \int _ { 0 } ^ { T } \int _ { \mathbb { R } ^ { d } } u _ { t } ( x ) \cdot w _ { t } ( x ) \rho _ { t } ^ { v } ( \mathrm { d } x ) \mathrm { d } t ,\tag{1.7}
$$

the advective Fisher–Rao metric, where $\rho _ { \bullet } ^ { v }$ solves the continuity equation (1.2).

We next show that the advective Fisher–Rao metric is indeed compatible with the least-squares energy E in the sense that the gradient flow of E with respect to the metric follows a geodesic in the space of paths of probability densities from any initial condition to the global optimizer.

Theorem (Theorem 3.9, optimality). Consider a velocity field $v _ { \bullet } \in L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 }$ such that $\rho _ { \bullet } ^ { \star } \ll \rho _ { \bullet } ^ { v }$ and assume that the gradient with respect to the advective Fisher–Rao metric $u _ { \bullet } = \nabla ^ { \mathrm { A F R } } E ( v _ { \bullet } )$ exists in $L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } .$ Then, it holds that

$$
\left. \frac { \mathrm { d } } { \mathrm { d } h } \right| _ { h = 0 } \rho _ { \bullet } ^ { v + h u } = \rho _ { \bullet } ^ { v } - \rho _ { \bullet } ^ { \star } .
$$

Informally speaking, this result suggests that, given the existence of the gradient flow, the advective Fisher–Rao gradient flow

$$
\partial _ { \tau } v _ { \bullet } ^ { \tau } = - \nabla ^ { \mathrm { A F R } } E ( v _ { \bullet } ^ { \tau } )
$$

leads to the following dynamics

$$
p _ { \bullet } ^ { v ^ { \tau } } = p _ { \bullet } ^ { \star } + e ^ { - \tau } ( p _ { \bullet } ^ { v ^ { 0 } } - p _ { \bullet } ^ { \star } ) .
$$

This means that the density path $p _ { \bullet } ^ { \tau }$ converges to $p _ { \bullet } ^ { \star }$ at a rate of $O ( e ^ { - \tau } )$ along the linear interpolation, which is known as a mixture geodesic in information geometry [Ama16, AJLS17]. In particular, this shows that the advective Fisher–Rao metric is compatible with E, thereby providing an afirmative answer to the original question.

## 1.1.2 The Advective Fisher–Rao geometry from three perspectives

We next show that the advective Fisher–Rao metric introduced above is a canonical object from three diferent perspectives: (1) as a zero-noise limit of the Fisher–Rao metric on path measures, (2) as the expectation of the second variation of the Freidlin–Wentzell rate functional over the initial distribution, and $\left( 3 \right)$ as the Hessian metric of the Benamou–Brenier action functional.

(1) Information geometry. The advective Fisher–Rao metric arises as a natural extension of the Fisher–Rao metric to the setting of deterministic dynamics as a rescaled zero-noise limit in which the Fisher–Rao metric becomes singular.

Theorem (Theorem $4 { \cdot } 1 )$ . Consider $v _ { \bullet } \in L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 }$ and assume that $\rho _ { 0 } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ Then we obtain the advective Fisher–Rao metric as the rescaled zero-noise limit of the Fisher–Rao metrics on path measures, that is,for any $u _ { \bullet } , w _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 }$ we have

$$
g _ { v \bullet } ^ { \mathrm { A F R } } ( u _ { \bullet } , w _ { \bullet } ) = \operatorname* { l i m } _ { \varepsilon  0 } \varepsilon g _ { v \bullet } ^ { \mathrm { F R } , \varepsilon } ( u _ { \bullet } , w _ { \bullet } ) .
$$

(2) Large deviations. As the metric is obtained as a zero-noise limit, we can also relate it to the Freidlin–Wentzell large deviation principle for small-noise difusions, which describes the concentration of the path measures $\mathbb { P } ^ { v , \varepsilon }$ in the limit $\varepsilon \to 0$ . The speed of the concentration is quantified by the Freidlin–Wentzell functional

$$
I _ { v } ( \boldsymbol { x } _ { \bullet } ) = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \lVert \dot { \boldsymbol { x } } _ { t } - \boldsymbol { v } _ { t } ( \boldsymbol { x } _ { t } ) \rVert _ { 2 } ^ { 2 } \mathrm { d } t .
$$

Theorem (Theorem $4 { \cdot } 5 )$ . Consider $\rho _ { 0 } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ , velocity fields $v _ { \bullet } \in L _ { T } ^ { 2 } C _ { \ b } ^ { 2 }$ and $u _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 } ,$ , and denote the flow induced by $v _ { \bullet }$ starting at x<sub>0</sub> by $\varphi ^ { v } ( x _ { 0 } )$ . Then it holds that

$$
g _ { v _ { \bullet } } ^ { \mathrm { A F R } } ( u _ { \bullet } , u _ { \bullet } ) = \int _ { \mathbb { R } ^ { d } } \frac { \mathrm { d } ^ { 2 } } { \mathrm { d } h ^ { 2 } } \bigg | _ { h = 0 } I _ { v } ( \varphi ^ { v + h u } ) ( x _ { 0 } ) \rho _ { 0 } ( \mathrm { d } x _ { 0 } ) ,
$$

(3) Optimal transport. The advective Fisher–Rao metric operates on velocity fields and is tightly connected to the solution operator of the continuity equation. As such, it is a natural question how it is connected to the optimal transport geometry of the space of probability densities. Recall the Benamou–Brenier formulation of the Wasserstein distance, which states

$$
W _ { 2 } ^ { 2 } ( \mu , \nu ) = 2 \operatorname * { i n f } _ { p _ { \bullet } , j _ { \bullet } } \left\{ \mathcal { A } ( p _ { \bullet } , j _ { \bullet } ) : p _ { 0 } = \mu , p _ { 1 } = \nu , \partial _ { t } p _ { t } + \nabla \cdot j _ { t } = 0 \right\} ,
$$

where the action functional is given by

$$
\mathcal { A } ( p _ { \bullet } , j _ { \bullet } ) = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \frac { \| j _ { t } ( x ) \| _ { 2 } ^ { 2 } } { p _ { t } ( x ) } \mathrm { d } x \mathrm { d } t
$$

Here, the current is given by ${ j _ { t } } = v _ { t } p _ { t }$

Theorem (Theorem 4.7). Assume that the initial distribution $p _ { 0 } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ has a positive density with respect to the Lebesgue measure. For any velocity field $v _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 }$ and $u _ { \bullet } \in C _ { \mathrm { c } } ^ { 1 } ( [ 0 , 1 ] \times \mathbb { R } ^ { d } ; \mathbb { R } ^ { d } )$ , it holds that

$$
g _ { v _ { \bullet } } ^ { \mathrm { A F R } } ( u _ { \bullet } , u _ { \bullet } ) = 2 \operatorname* { l i m } _ { h  0 } \frac { \mathcal { A } ( p _ { \bullet } ^ { h } , j _ { \bullet } ^ { h } ) - \mathcal { A } ( p _ { \bullet } , j _ { \bullet } ) - \delta ^ { + } \mathcal { A } ( p _ { \bullet } , j _ { \bullet } ) [ p _ { \bullet } ^ { h } - p _ { \bullet } , j _ { \bullet } ^ { h } - j _ { \bullet } ] } { h ^ { 2 } } ,\tag{1.8}
$$

$$
w h e r e p _ { \bullet } = p _ { \bullet } ^ { v } , p _ { \bullet } ^ { h } = p _ { \bullet } ^ { v + h u } , j _ { \bullet } = v _ { \bullet } p _ { \bullet } ^ { v } , a n d j _ { \bullet } ^ { h } = ( v _ { \bullet } + h u _ { \bullet } ) p _ { \bullet } ^ { v + h u } .
$$

The result relates the advective Fisher–Rao metric to the Benamou–Brenier action functional. Note that the diference quotient in (1.8) is the diference quotient, which, given the existence of the second variation, diferentiability of the map $\Psi ( v _ { \bullet } ) = ( p _ { \bullet } ^ { v } , p _ { \bullet } ^ { v } v _ { \bullet } )$ , and diferentiability of the curve $( p _ { \bullet } ^ { h } , j _ { \bullet } ^ { h } )$ with respect to $h _ { ; }$ would converge to

$$
\delta ^ { 2 } \mathcal { A } ( \Psi ( v _ { \bullet } ) ) [ \delta \Psi ( v _ { \bullet } ) [ u _ { \bullet } ] , \delta \Psi ( v _ { \bullet } ) [ u _ { \bullet } ] ] .
$$

In this case, the advective Fisher–Rao metric can be interpreted as the pullback of the Hessian metric of the Benamou–Brenier action functional along Ψ.

## 1.1.3 The advective Fisher–Rao metric on paths of probability measures

Having defined the advective Fisher–Rao metric on velocity fields, we now lift this construction to a Riemannian metric on the space of paths of probability measures. We work on the space of regular curves

$$
\operatorname { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) ) = \Big \{ \rho _ { \bullet } \in \operatorname { A C } _ { T } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) ) : \int _ { 0 } ^ { 1 } | \dot { \rho } _ { t } | _ { W _ { 2 } } ^ { 2 } \mathrm { d } t < \infty \Big \} ,\tag{1.9}
$$

where $\begin{array} { r } { \vert \dot { \rho } _ { t } \vert _ { W _ { 2 } } = \operatorname* { l i m } _ { h \to 0 } W _ { 2 } ( \rho _ { t + h } , \rho _ { t } ) h ^ { - 1 } } \end{array}$ is the metric derivative [AGS05]. For any curve $\rho _ { \bullet } \in \mathsf { A C } _ { T } ^ { 2 } ( \mathscr { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ there exists a unique velocity field $v _ { \bullet } = v _ { \bullet } ^ { \rho } \in L _ { T } ^ { 2 } L ^ { 2 } ( \rho _ { \bullet } )$ such that the continuity equation

$$
\begin{array} { r } { \partial _ { t } \rho _ { t } + \nabla \cdot ( \rho _ { t } v _ { t } ) = 0 , } \end{array}\tag{1.10}
$$

holds and $v _ { t } \in \mathcal { T } _ { \rho _ { t } } \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ for almost every t, where

$$
\begin{array} { r } { \mathcal { T } _ { \rho } \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) = \overline { { \left\{ \nabla \phi : \phi \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } ) \right\} } } ^ { L ^ { 2 } ( \rho ) } } \end{array}
$$

denotes the Otto–Wasserstein tangent space. To describe variations of a curve that remain compatible with the advective structure, we consider tangent vectors $\begin{array} { r } { \xi _ { \bullet } = \left. \frac { \mathrm { d } } { \mathrm { d } h } \right| _ { h = 0 } \rho _ { \bullet } ^ { v + h u } } \end{array}$ at $\rho _ { \bullet } = \rho _ { \bullet } ^ { v }$ , which arise from a variation of the velocity field. Such a tangent curve $\xi _ { \bullet }$ satisfies the linearized continuity equation

$$
\partial _ { t } \xi _ { t } + \nabla \cdot ( \xi _ { t } \boldsymbol { v } _ { t } ) = - \nabla \cdot ( \rho _ { t } \boldsymbol { u } _ { t } )\tag{1.11}
$$

in a distributional sense. This leads to the definition of the tangent space

$$
\begin{array} { r } { T _ { \rho _ { \bullet } } \mathsf { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) ) : = \left\{ \xi _ { \bullet } \in T _ { \rho _ { \bullet } } L _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) ) : \begin{array} { l } { \xi _ { t } v _ { t } \in \mathcal { D } ^ { \prime } ( \mathbb { R } ^ { d } ) \mathrm { ~ f o r ~ a l l ~ } t \in [ 0 , 1 ] } \\ { \qquad D _ { t } \xi _ { \bullet } \in T _ { \rho _ { \bullet } } L _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) ) } \end{array} \right\} , } \end{array}
$$

where we refer to

$$
D _ { t } \xi _ { t } : = \partial _ { t } \xi _ { t } + \nabla \cdot ( \xi _ { t } v _ { t } )\tag{1.12}
$$

as the transport derivative of $\xi _ { \bullet }$ along $\rho _ { \bullet }$ , which is to be understood in the distributional sense. With these ingredients, we introduce the advective Fisher–Rao metric on paths of measures.

Definition (Advective Fisher–Rao metric on paths of measures, Definition $5 { \cdot } 1 )$ Consider a path $\rho _ { \bullet } \in \mathsf { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ ) and its velocity $v _ { \bullet } \in T _ { \rho _ { \bullet } } L _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ . For two tangent vectors $\xi _ { \bullet } , \zeta _ { \bullet } \in T _ { \rho _ { \bullet } } \mathrm { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ ), we refer to

$$
g _ { \rho \bullet } ^ { \mathrm { A F R } } ( \xi _ { \bullet } , \xi _ { \bullet } ) = \int _ { 0 } ^ { 1 } g _ { \rho _ { t } } ^ { \mathrm { W O } } ( D _ { t } \xi _ { t } , D _ { t } \xi _ { t } ) \mathrm { d } t
$$

as the advective Fisher–Rao metric.

A natural point of reference is the integrated Wasserstein–Otto metric

$$
g _ { \rho _ { \bullet } } ^ { \mathrm { W O } } ( \xi _ { \bullet } , \xi _ { \bullet } ) = \int _ { 0 } ^ { 1 } g _ { \rho _ { t } } ^ { \mathrm { W O } } ( \xi _ { t } , \xi _ { t } ) \mathrm { d } t = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \bigl \| v _ { t } ^ { \xi } ( x ) \bigr \| ^ { 2 } \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t ,\tag{1.13}
$$

where $v _ { t } ^ { \xi } \in \mathcal { T } _ { \rho _ { t } } \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ is the unique solution of $\xi _ { t } = - \nabla \cdot ( \rho _ { t } v _ { t } ^ { \xi } )$ . In contrast, the advective Fisher–Rao metric involves the transport derivative of $\xi _ { \bullet }$ along the base velocity field and measures the size of the source term that appears when the variation $\xi _ { \bullet }$ is transported along the underlying flow. Geometrically, $D _ { t }$ coincides with the covariant derivative of a flat connection that difers from the Levi-Civita connection of the Wasserstein–Otto metric $[ \mathrm { A y } 2 5 ]$ . The metric defined above is the natural counterpart of the advective Fisher–Rao metric on velocity fields and inherits several fundamental properties:

Compatibility. Lemma $5 { \cdot } 3$ shows that the solution map $v _ { \bullet } \mapsto \rho _ { \bullet } ^ { v }$ of the continuity equation is a Riemannian isometry with respect to the advective Fisher–Rao metrics.

Relation to the Benamou–Brenier action. Theorem $5 { \cdot } 4$ identifies the AFR metric on paths as the second variation of the Benamou–Brenier action functional, establishing a direct connection to optimal transport.

Optimality. Theorem $5 { \cdot } 8$ proves that the gradient flow of the least-squares energy on the velocities with respect to the advective Fisher–Rao metric follows a mixture geodesic toward the reference path, guaranteeing exponential convergence.

## 1.1.4 Areas of application

Here, we discuss diferent problems that are covered by our general framework of control of probability measures with quadratic cost on the velocity fields.

Flow matching In generative modeling, it is the goal to produce approximate samples from an unknown data distribution $\rho _ { \mathrm { d a t a } }$ . Flow matching was introduced in $[ \mathrm { L C B H ^ { + } } _ { 2 3 } ]$ and quickly became one of the most powerful paradigms in generative modeling. It aims to generate approximate samples by transforming sampling from an easy source distribution $\rho _ { \mathrm { s o u r c e } } .$ , for example, a standard Gaussian, along a flow of a velocity field. In flow matching, a neural network is used to parametrize a time-dependent velocity field $v _ { \bullet } ^ { \theta }$ that generates the flow. Then, samples from a source distribution $\rho _ { \mathrm { s o u r c e } }$ are transformed according to

$$
\dot { X } _ { t } = v _ { t } ^ { \theta } ( X _ { t } ) , \quad X _ { 0 } \sim \rho _ { \mathrm { s o u r c e } } .
$$

Denoting the distribution of $X _ { t }$ by $\rho _ { t }$ , it is the goal to achieve $\rho _ { 1 } \approx \rho _ { \mathrm { d a t a } }$ . To this end, one constructs a velocity field $v _ { \bullet } ^ { \star }$ such that

$$
\dot { X } _ { t } ^ { \star } = v _ { t } ^ { \theta } ( X _ { t } ^ { \star } ) , \quad X _ { 0 } ^ { \star } \sim \rho _ { \mathrm { s o u r c e } }
$$

achieves exact interpolation, in the sense that $\rho _ { 1 } ^ { \star } = \rho _ { \mathrm { d a t a } }$ , where $\rho _ { t } ^ { \star }$ denotes the distribution of $X _ { t } ^ { \star }$ . Such a reference field $v _ { \bullet } ^ { \star }$ is neither available in closed form nor unique, and various strategies for the construction and estimation of such velocity fields based on conditional interpolation or ideas from optimal transport have been developed. We refer to $\mathrm { [ L H H ^ { + } } 2 4 ,$ , ABVE25, ${ \mathrm { W S } } 2 5 .$ , PTDN26] for extensive discussions on this aspect of flow-based generative models. Having chosen a reference velocity field $v _ { \bullet } ^ { \star }$ , the network’s parameters are optimized to minimize the flow matching loss

$$
L _ { \mathrm { F M } } ( \theta ) = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| v _ { t } ^ { \theta } ( x ) - v _ { t } ^ { \star } ( x ) \| ^ { 2 } \rho _ { t } ^ { \star } ( \mathrm { d } x ) \mathrm { d } t .
$$

This objective function can be written as $L ( \theta ) = E ( v _ { \bullet } ^ { \theta } )$ for a quadratic energy $E$ on the velocity fields and therefore of the form $( 1 . 3 )$

Score matching Difusion models are arguably one of the most successful generative models [SDWMG15, HJA20, RBL+22]. Here, we focus on score matching, where one considers forward dynamics

$$
\mathrm { d } X _ { t } ^ { \star } = f _ { t } ( X _ { t } ^ { \star } ) + g _ { t } \mathrm { d } W _ { t } , \quad X _ { 0 } ^ { \star } \sim \rho _ { \mathrm { d a t a } }
$$

and we denote the density of $X _ { t } ^ { \star }$ by $\rho _ { t } ^ { \star }$ , where $f _ { t }$ and $g _ { t } > 0$ are specified by the user in a way that $\rho _ { 1 } ^ { \star } \approx \rho _ { \mathrm { s o u r c e } } ,$ see $[ \mathrm { L S K ^ { + } } _ { 2 5 } ]$ . The backwards ODE

$$
\dot { X } _ { t } = f _ { t } ( X _ { t } ) - \frac { g _ { t } ^ { 2 } } { 2 } s _ { t } ^ { \star } ( X _ { t } ) , \quad X _ { 1 } \sim \rho _ { 1 } ^ { \star }
$$

induces exactly the same time marginals as $X _ { t } ^ { \star }$ , meaning that $X _ { t } \sim \rho _ { t } ^ { \star }$ , where the true score function is given by $s _ { t } ^ { \star } = \nabla$ ln $p _ { t } ^ { \star }$ and $p _ { t } ^ { \star }$ denotes the Lebesgue density of $\rho _ { t } ^ { \star } . \ \mathrm { A s } \ s _ { \bullet } ^ { \star }$ is generally not accessible, it is approximated by a neural network $s _ { \bullet } ^ { \theta } \approx s _ { \bullet } ^ { \star }$ and the network’s parameters $\theta$ are optimized to minimize the least-squares loss

$$
L _ { \mathrm { S M } } ( \theta ) = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \alpha _ { t } \int _ { \mathbb { R } ^ { d } } \lVert s _ { t } ^ { \theta } ( x ) - s _ { t } ^ { \star } ( x ) \rVert _ { 2 } ^ { 2 } \rho _ { t } ^ { \star } ( \mathrm { d } x ) \mathrm { d } t ,
$$

where $\alpha _ { t } > 0$ is a weighting factor specified by the user. The backward ODE with the approximate score function is given by

$$
\dot { X } _ { t } ^ { \theta } = f _ { t } ( X _ { t } ) - \frac { g _ { t } ^ { 2 } } { 2 } s _ { t } ^ { \theta } ( X _ { t } ) , \quad X _ { 1 } ^ { \theta } \sim \rho _ { \mathrm { s o u r c e } } .
$$

Denoting the corresponding velocity fields by

$$
v _ { t } ^ { \theta } ( x ) : = f _ { t } ( x ) - \frac { g _ { t } ^ { 2 } } { 2 } s _ { t } ^ { \theta } ( x ) , \quad \mathrm { a n d ~ } v _ { t } ^ { \star } ( x ) : = f _ { t } ( x ) - \frac { g _ { t } ^ { 2 } } { 2 } s _ { t } ^ { \star } ( x ) ,
$$

we can rewrite the loss function according to

$$
L ( \theta ) = 2 \int _ { 0 } ^ { 1 } \frac { \alpha _ { t } } { g _ { t } ^ { 4 } } \int _ { \mathbb { R } ^ { d } } \bigl \| v _ { t } ^ { \theta } ( x ) - v _ { t } ^ { \star } ( x ) \bigr \| _ { 2 } ^ { 2 } \rho _ { t } ^ { \star } ( \mathrm { d } x ) \mathrm { d } t ,
$$

which is a quadratic function of the velocity $v _ { \bullet } ^ { \theta } .$

Imitation learning in linear environments The reasoning above can readily be extended to imitation learning under general linear dynamics

$$
{ \dot { x } } _ { t } = A _ { t } x _ { t } + B _ { t } u _ { t }
$$

where $\boldsymbol { x } ~ \in ~ \mathbb { R } ^ { d }$ is the state and $u \in \mathbb { R } ^ { n }$ the control. For various real-world problems, incorporating expert demonstrations was necessary to achieve high performance [Pom88, RB10, BDTD+16, TCA+23, ZKLF23]. The integration of such prior knowledge is implemented in the framework known as imitation learning or behavioral cloning. Here, one assumes access to an expert policy $\pi _ { \bullet } ^ { \star } \colon [ 0 , 1 ] \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { n }$ , uses a neural network $\pi _ { \bullet } ^ { \theta } \colon [ 0 , 1 ] \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { n }$ , and trains the network’s parameters to minimize the loss

$$
L ( \theta ) = \frac { 1 } { 2 } \int _ { \mathbb { R } ^ { d } } \int _ { 0 } ^ { 1 } \| \pi _ { t } ^ { \theta } ( x ) - \pi _ { t } ^ { \star } ( x ) \| ^ { 2 } \rho _ { t } ^ { \star } ( \mathrm { d } x ) \mathrm { d } t
$$

where $\rho _ { t } ^ { \star }$ denotes the distribution of $x _ { t }$ under the policy $\pi _ { \bullet } ^ { \star }$ . Despite the linear nature of the dynamics, the problem is not solvable via the Riccati equations as it involves a possibly nonlinear expert policy $\pi _ { t } ^ { \star }$ . For example, even in linear model predictive control, the expert policy becomes piecewise afine and intractable to compute for high-dimensional problems, and we refer to $[ \mathrm { R H R } ^ { + } 2 6 $ , Section $7 . 2 ]$ for a survey of imitation learning approaches in this case.

In this setting of linear dynamics, the state dynamics under a policy can be written as

$$
{ \dot { x } } _ { t } = v _ { t } ^ { \theta } ( x _ { t } ) , \quad { \mathrm { w h e r e ~ } } v _ { t } ^ { \theta } ( x ) = A _ { t } x + B _ { t } \pi _ { t } ^ { \theta } ( x ) .
$$

If $B _ { t }$ has full column rank for every $t \in [ 0 , 1 ] .$ , we obtain

$$
L ( \theta ) = \frac { 1 } { 2 } \int _ { \mathbb { R } ^ { d } } \int _ { 0 } ^ { 1 } ( v _ { t } ^ { \theta } - v _ { t } ^ { \star } ) ^ { \top } ( B _ { t } ^ { \top } B _ { t } ) ^ { - 1 } ( v _ { t } ^ { \theta } - v _ { t } ^ { \star } ) \rho _ { t } ^ { \star } ( \mathrm { d } x ) \mathrm { d } t ,
$$

which is again a quadratic objective in the velocity field and hence of the form $\left( 1 . 3 \right)$

Computational experiments To use the developed Riemannian metric for optimization of the velocity fields, we consider the application to the flow matching approach. Here, we discretize the advective Fisher–Rao gradient flow and obtain a natural gradient method, which updates the parameters $\theta$ of the neural network according to

$$
\theta _ { k + 1 } = \theta _ { k } - \eta \cdot F ( \theta _ { k } ) ^ { - 1 } \nabla L _ { \mathrm { F M } } ( \theta _ { k } ) ,\tag{1.14}
$$

where the Fisher-information matrix $F ( \theta )$ is a discretization of the advective Fisher– Rao metric with entries

$$
F ( \theta ) _ { i j } = g _ { v _ { \bullet } ^ { \theta } } ^ { \mathrm { A F R } } ( \partial _ { \theta _ { i } } v _ { \bullet } ^ { \theta } , \partial _ { \theta _ { j } } v _ { \bullet } ^ { \theta } ) = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \partial _ { \theta _ { i } } v ^ { \theta } ( x ) \cdot \partial _ { \theta _ { j } } v ^ { \theta } ( x ) p _ { t } ^ { \theta } ( x ) \mathrm { d } x \mathrm { d } t .
$$

In contrast, the Gauss–Newton method replaces the Fisher-information matrix in the preconditioned gradient scheme (1.14) by

$$
G ( \theta ) _ { i j } = ( \partial _ { \theta _ { i } } v _ { \bullet } ^ { \theta } , \partial _ { \theta _ { j } } v _ { \bullet } ^ { \theta } ) _ { L _ { T } ^ { 2 } L ^ { 2 } ( p _ { \bullet } ^ { \star } ) } = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \partial _ { \theta _ { i } } v ^ { \theta } ( x ) \cdot \partial _ { \theta _ { j } } v ^ { \theta } ( x ) p _ { t } ^ { \star } ( x ) \mathrm { d } x \mathrm { d } t .
$$

Consistent with the theoretical results, we demonstrate empirically that the natural gradient method leads to the desired optimal fitting of the paths of probability densities, while the Gauss–Newton method leads to an optimal fitting of the velocity fields, see Figure 1. Further, we demonstrate that both of these second-order optimizers have the ability to increase the accuracy of flow-based generative models over first-order optimizers. All experiments are presented in Section 7.

## 1.2 Related works

Schrödinger problems and entropy-regularized optimal transport. The objective (1.3) is well known to coincide with the Kullback–Leibler divergence between path measures induced by the underlying SDEs, a correspondence that dates back at least to the classical work of [Fö85]. Minimizing this path-space relative entropy gives rise to $L ^ { 2 }$ stochastic optimal control problems, whose optimal solutions characterize dynamic Schrödinger bridges and thereby the entropic regularization of dynamic optimal transport [BB00]. A large body of work has been devoted to understanding the convergence of these entropy-regularized problems in the smallnoise limit $\varepsilon  0$ . Foundational results qualitatively establish the convergence in the vanishing-noise limit [Mik02, Mik04, Léo12a, CDPS17]. More recently, quantitative convergence rates have been developed through asymptotic expansions of the regularization error, including first-order expansions in [CPT23, EN24, MS25] and second-order expansions in [CT21, CRL+20, Pal24, NZ26]. Closely related works further investigate the convergence of Schrödinger potentials and transport maps in the small-noise regime [GT21, PNW21, NW22, BGN22, CCGT23, CPT23, MS25, LR26]. Our perspective is complementary to this line of research. Rather than studying the asymptotic behavior of the optimal values or minimizers of entropy-regularized optimal transport problems, we investigate the diferential geometry induced by the regularization functional itself. More precisely, we study the Riemannian metric arising from its Hessian on path space and identify its singular limit as the noise level tends to zero, leading to the advective Fisher–Rao geometry for deterministic flows of probability measures.

Stochastic optimal control and sampling The objective (1.3) arises in $L ^ { 2 }$ stochastic optimal control problems. A similar least-squares objective was used in [Ric21] and subsequent works [BRU24] to phrase the training of score-based difusion samplers as a path-space stochastic optimal control problem. A similar paradigm is pursued in [ZC22], which frames sampling from unnormalized densities as a path-space stochastic control problem optimized via path-integral methods, and in [DEDKC25], where control and reward-based fine-tuning are addressed using an adjoint matching regression target. Further, a geometric optimization framework for this ansatz utilizing a trust-region method based on the relative entropy on the space of path measures has been developed in $[ \mathrm { B B R } ^ { + } 2 5 ]$ . Although closely related to our approach in spirit, these methods use a reverse KL divergence as their objective function rather than the forward KL divergence that is commonly used in flow and score matching models. This choice of objective results in exponential geodesics, as opposed to the mixture geodesics that arise in our framework. Further, this approach heavily relies on the presence of noise, as otherwise the KL divergence between path measures of diferent velocity fields will never be finite, necessitating our zero-noise construction of a corresponding geometry for deterministic forward dynamics.

Geometry of flow-based generative models Various works have studied the geometry of certain aspects of flow-based generative models. $[ \mathrm { G L C ^ { + } } _ { 2 3 }$ , VMSL25] argue that the forward and backward dynamics in difusion models should be interpreted through the lens of optimal transport, as they can be interpreted as Wasserstein gradient flows of the KL divergence with respect to the data distribution and the prior distribution, respectively. This observation covers the data generation dynamics, but fails to address the objective function or the training process. To study geometric structures embedded within these models, other recent works have introduced localized or pull-back Riemannian structures. For instance, $[ \mathrm { K H P ^ { + } } _ { 2 6 } ]$ analyzes the geometry induced by the Fisher information matrix along the reverse probability flow ODE to examine trajectory properties, while $[ \mathrm { A D B } 2 5 ]$ leverages learned score functions to construct ambient-space metrics for navigating and exploring the underlying data manifold. Though insightful, these approaches pursue a diferent goal and characterize the static space of data points or sample paths rather than the functional landscape of the objectives themselves. On the side of optimization of flow-based generative models, several second-order heuristics have been proposed to improve their trainability. For example, [LCT21] proposes a low-rank approximation of the Hessian for neural ODEs, which, however, is not derived from a functional perspective and does not provide a geometric interpretation, thereby leading to suboptimal update directions $[ \mathrm { M } Z 2 4 ]$ . Further, [HSL25] proposes a natural gradient method for SDE drifts, which uses the Fisher information matrix as a preconditioner and thereby inherently requires noise in the forward dynamics.

Reinforcement learning and natural policy gradients The Riemannian metric constructed in this work is conceptually related, though with important diferences, to the metric underlying many important successes of reinforcement learning. Here, Kakade proposed an adaptive mixture of the Fisher–Rao metrics on the individual components as a metric on the space of Markov kernels [Kak01]. This geometry has been studied in terms of invariance axiomatics [Leb05, MRA14], has been characterized as isometric to the conditional Fisher–Rao metric on occupancy measures [MM24], and has been shown to arise from the Fisher–Rao metric on path measures [BS03, PVS03, Nag05]. The advective Fisher–Rao metric constructed by us is closely related to the pullback of the Fisher–Rao metric on path measures. However, the advective Fisher–Rao metric introduced here operates in a regime of continuous time and deterministic forward dynamics, requiring a rescaled zero-noise limit, thereby ofering close links to large deviation theory and optimal transport. Furthermore, most reinforcement learning focuses on a diferent objective function from the one considered here, which prevents a characterization as a metric that yields the optimal update directions obtained here.

Fisher–Rao gradient flows and sampling A substantial body of work uses Fisher–Rao gradient flows, or their discretisations, for sampling and achieves $O ( e ^ { - t } )$ convergence, independently of the target, see [LLN19, LSW23, CCH+26] and subsequent work. These works are related in philosophy to our approach in that they propose a Fisher–Rao geometry for the optimization of an entropic objective. However, they consider the Fisher–Rao metric for a fixed-time marginal, whereas our $g ^ { \mathrm { A F R } }$ metric is a global, pathwise tensor that accounts for the temporal coupling induced by the continuity equation.

Wasserstein–Fisher–Rao geometry Our work studies the zero-noise limit of the Fisher–Rao metric on path measures, which we show to converge to the Hessian geometry of the Benamou–Brenier action functional. As such, it can be seen as recovering a geometry induced by a functional appearing in optimal transport as the limiting object of the central metric in information geometry. Thus, we want to contrast our work with the construction of a geometry unifying the Fisher–Rao and Wasserstein geometries, which is known as the Wasserstein–Fisher–Rao geometry or Hellinger–Kantorovich geometry. This construction was motivated by unbalanced optimal transport $\mathrm { [ P R 1 4 ] }$ and allows for mass creation and destruction in addition to the transport of mass, thereby ofering an interpolation between the Wasserstein and Fisher–Rao geometries [LMS18, CPSV18]. The resulting object is a geometry on static probability measures, whereas our construction works on the space of paths of probability measures, thereby capturing the temporal coupling induced by the continuity equation.

## 2 Preliminaries and Setup

This section serves the purpose of introducing the notation and technical assumptions we will use throughout the paper.

## 2.1 Notation and conventions

We denote the d-dimensional Euclidean space by $\mathbb { R } ^ { d }$ , the Euclidean gradient by $\nabla .$ and the divergence of a vector field v by $\nabla \cdot \boldsymbol { v }$ . We denote the space of distributions by $\mathcal { D } ^ { \prime } ( \mathbb { R } ^ { d } )$ , which is the topological dual space to the locally convex vector space $C _ { \mathrm { c } } ^ { \infty } ( \mathbb { R } ^ { d } )$ of compactly supported smooth functions on $\mathbb { R } ^ { d }$ . For the sake of simplified notation, we set $L _ { T } ^ { p } : = L ^ { p } ( [ 0 , 1 ] )$ for $p \in [ 1 , \infty ) \cup \{ + \infty \}$ and for a Banach space $X ,$ , we denote the Bochner–Lebesgue and Sobolev–Bochner–Lebesgue spaces by $L _ { T } ^ { p } ( X ) : = L ^ { p } ( [ 0 , 1 ] ; X )$ and $H _ { T } ^ { 1 } ( X ) : = H ^ { 1 } ( [ 0 , 1 ] ; X )$ , respectivly. The space of continuous paths is $C _ { T } ( \mathbb { R } ^ { d } ) : = \bar { C } ( [ 0 , T ] ; \mathbb { R } ^ { d } )$ . With $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ we denote the space of probability measures on $\mathbb { R } ^ { d }$ with finite second moment

$$
\mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) : = \left\{ \mu \in \mathcal { P } ( \mathbb { R } ^ { d } ) : m _ { 2 } ( \mu ) < \infty \right\} , \qquad m _ { 2 } ( \mu ) : = \int _ { \mathbb { R } ^ { d } } \| x \| ^ { 2 } ~ \mathrm { d } \mu ( x ) .
$$

The Wasserstein-2 distance between $\mu , \nu \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ is given by

$$
W _ { 2 } ^ { 2 } ( \mu , \nu ) : = \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int _ { \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } } \left\| x - y \right\| ^ { 2 } \mathrm { d } \pi ( x , y ) ,
$$

where $\Pi ( \mu , \nu )$ denotes the set of couplings of $\mu$ and $\nu .$ . For a measurable map Φ, the pushforward of a measure $\mu$ is $( \Phi ) _ { \sharp } \mu$ . The dual pairing between a measure $\mu$ and a function f is $\textstyle \langle \mu , f \rangle : = \int f \mathrm { d } \mu$ . For $\alpha \in ( 0 , 1 ]$ , the α-Hölder semi-norm of a function $\phi \colon { \mathbb { R } ^ { d } } \to$ R is given by

$$
[ \phi ] _ { \alpha } : = \operatorname* { s u p } _ { x \neq y } { \frac { | \phi ( x ) - \phi ( y ) | } { \| x - y \| ^ { \alpha } } } .
$$

In the special case $\alpha = 1$ , we denote the Lipschitz semi-norm by

$$
[ \phi ] _ { \mathrm { L i p } } : = \operatorname* { s u p } _ { x \neq y } { \frac { | \phi ( x ) - \phi ( y ) | } { \| x - y \| } } .
$$

For a convex functional $F \colon C \to \mathbb { R } \cup \{ + \infty \}$ we denote the Bregman divergence by

$$
D _ { F } ( x , y ) : = F ( x ) - F ( y ) - \delta ^ { + } F ( y ) [ x - y ] \quad { \mathrm { f o r ~ a l l ~ } } x , y \in { \mathrm { d o m } } ( F ) ,
$$

where for any direction $d = y - x$ for $x , y \in \operatorname { d o m } ( F )$ the right-sided first variation is defined as

$$
\delta ^ { + } F ( x ) [ d ] : = \operatorname* { l i m } _ { h \searrow 0 } { \frac { F ( x + h d ) - F ( x ) } { h } } .
$$

## 2.2 Flows, the continuity equation, and path measures

We consider the SDE

$$
\mathrm { d } X _ { t } = v _ { t } ( X _ { t } ) \mathrm { d } t + \sqrt { \varepsilon } \mathrm { d } W _ { t } ,\tag{2.1}
$$

where $X _ { 0 } \sim p _ { 0 }$ , where $v _ { \bullet }$ is a time-dependent velocity field. The introduction of the noise term $\sqrt { \varepsilon } \mathrm { d } W _ { t }$ allows us to construct a non-degenerate geometry Fisher–Rao metric in the noiseless case, via a rescaled zero-noise limit.

Regularity of velocity fields We work in the classic setting of Carathéodory solutions of ODEs, which is a convenient setting for the existence of solutions to the continuity equation that also covers many neural network architectures. We denote the space of all continuously diferentiable Lipschitz vector fields by

$$
C _ { \mathrm { L i p } } ^ { 1 } ( \mathbb { R } ^ { d } ; \mathbb { R } ^ { d } ) : = \left\{ v \in C ^ { 1 } ( \mathbb { R } ^ { d } ; \mathbb { R } ^ { d } ) : v { \mathrm { ~ i s ~ g l o b a l l y ~ L i p s c h i t z } } \right\} .
$$

This is a Banach space with the norm

$$
\begin{array} { r } { \| v \| _ { C _ { \mathrm { L i p } } ^ { 1 } ( \mathbb { R } ^ { d } ; \mathbb { R } ^ { d } ) } : = \| v ( 0 ) \| + \| D v \| _ { \infty } . } \end{array}
$$

Further, we consider the Bochner-Lebesgue space<sup>1</sup>

$$
L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } : = L ^ { 1 } ( [ 0 , 1 ] ; C _ { \mathrm { L i p } } ^ { 1 } ( \mathbb { R } ^ { d } ; \mathbb { R } ^ { d } ) )\tag{2.2}
$$

of time-dependent velocity fields that are continuously diferentiable and Lipschitz in space, such that

$$
\begin{array} { r } { \left\| v _ { t } ( 0 ) \right\| \leq \alpha _ { t } \quad \mathrm { a n d } \ \left\| v _ { t } ( x ) - v _ { t } ( y ) \right\| \leq \beta _ { t } \| x - y \| } \end{array}
$$

for all $x , y \in \mathbb { R } ^ { d } , t \in [ 0 , 1 ]$ and some $\alpha , \beta \in L _ { T } ^ { 1 }$ . Here, the norm on $L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 }$ is given by

$$
\| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } = \int _ { 0 } ^ { 1 } \| v _ { t } \| _ { C _ { \mathrm { L i p } } ^ { 1 } ( \mathbb { R } ^ { d } ; \mathbb { R } ^ { d } ) } \mathrm { d } t = \| \alpha \| _ { L _ { T } ^ { 1 } } + \| \beta \| _ { L _ { T } ^ { 1 } } .
$$

Note that by the Carathéodory existence theorem, $L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 }$ is a convenient space of velocity fields for the existence of solutions of the ODE and consequently for the existence of solutions to the continuity equation. Indeed, a velocity field $v _ { \bullet } \in L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 }$ induces a unique flow $\varphi _ { \bullet } ^ { v }$ satisfying

$$
\partial _ { t } \varphi _ { t } ^ { v } ( x ) = v _ { t } ( \varphi _ { t } ^ { v } ( x ) ) , \quad \mathrm { a n d } \varphi _ { 0 } ^ { v } ( x ) = x
$$

for all $t \in [ 0 , 1 ] , x \in \mathbb { R } ^ { d }$ , see $\mathrm { [ A G S o 8 }$ , Lemma $8 . 1 . 4 ]$ . Sometimes, we require stronger square-integrability in time and write

$$
L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 } : = L ^ { 2 } ( [ 0 , 1 ] ; C _ { \mathrm { L i p } } ^ { 1 } ( \mathbb { R } ^ { d } ; \mathbb { R } ^ { d } ) ) .
$$

In this case, the functions $\alpha , \beta$ are in $L _ { T } ^ { 2 }$ and equivalent norms are given by

$$
\| \alpha \| _ { L _ { T } ^ { 2 } } + \| \beta \| _ { L _ { T } ^ { 2 } } \quad \mathrm { o r } \quad \sqrt { \| \alpha \| _ { L _ { T } ^ { 2 } } ^ { 2 } + \| \beta \| _ { L _ { T } ^ { 2 } } ^ { 2 } } .
$$

Note that for $v \in L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } .$ , not only the ODE but also the SDE (2.1) has a unique strong solution for any initial condition $X _ { 0 } \sim p _ { 0 }$ , where $p _ { 0 }$ is a probability measure with finite second moments, see [KS14]. In flow-matching and difusion models, one typically uses neural networks $v _ { \bullet } ^ { \theta }$ that take time as an explicit input. If the activation function is continuous and diferentiable with a bounded derivative, so is the resulting velocity if a joint function of space and time, and hence contained in $L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 }$ . The regularity of $L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 }$ is enough to ensure well-posedness of the forward dynamics. However, sometimes, we require square integrability in time and denote the corresponding space by $L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 }$ . For some results, we require stronger spatial regularity and hence introduce the function space This is a Banach space with the norm

$$
\begin{array} { r } { \| v \| _ { C _ { \mathfrak { b } } ^ { 2 } ( \mathbb { R } ^ { d } ; \mathbb { R } ^ { d } ) } : = \| v ( 0 ) \| + \| D v \| _ { \infty } + \| D ^ { 2 } v \| _ { \infty } . } \end{array}
$$

Similarly for $\alpha \in ( 0 , 1 ]$ , we define

$$
C _ { \mathrm { L i p } } ^ { 1 , \alpha } ( \mathbb { R } ^ { d } ; \mathbb { R } ^ { d } ) : = \left\{ v \in C ^ { 1 } ( \mathbb { R } ^ { d } ; \mathbb { R } ^ { d } ) : D v { \mathrm { ~ i s ~ g l o b a l l y ~ b o u n d e d ~ a n d ~ } } \alpha \cdot \mathrm { H \ddot { o } l d e r } \right\} .
$$

with the corresponding norm

$$
\begin{array} { r } { \| v \| _ { C _ { \mathrm { L i p } } ^ { 1 , \alpha } ( \mathbb { R } ^ { d } ; \mathbb { R } ^ { d } ) } : = \| v ( 0 ) \| + \| D v \| _ { \infty } + [ D v ] _ { \alpha } . } \end{array}
$$

The continuity equation It is well-known that for any initial probability measure $p _ { 0 } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ with finite second moments the transported measures $\rho _ { t } : = ( \varphi _ { t } ^ { v } ) _ { \sharp } p _ { 0 } \in$ $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ are a distributional solution to the continuity equation

$$
\begin{array} { r } { \partial _ { t } \rho _ { t } + \nabla \cdot ( v _ { t } \rho _ { t } ) = 0 } \end{array}
$$

meaning that for any test function $\phi \in C _ { \mathrm { c } } ^ { \infty } ( ( 0 , 1 ) \times  { \mathbb { R } } ^ { d } )$ it holds that

$$
\int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \left( \partial _ { t } \phi _ { t } ( x ) + v _ { t } ( x ) \cdot \nabla \phi _ { t } ( x ) \right) \rho _ { t } ( \mathrm { d } x ) .
$$

For a proof, we refer to [AGS08, Lemma 8.1.6.] or [FG21, Lemma $4 { \cdot } 1 . 1 . ]$ . In particular, this shows that $\mathrm { i f } \ p _ { 0 }$ is a measure with positive Lebesgue density, then so is $p _ { t }$ for any $t \in [ 0 , 1 ]$ . The method of characteristics can easily be generalized to arbitrary finite signed measures $\rho _ { 0 }$ . Further, any solution of the continuity equations is given by the method of characteristics, and in particular, solutions of the continuity equation are unique under the regularity conditions on $v _ { \bullet }$ . Finally, we remind the reader of the continuity equation with source term $f _ { \bullet }$ , which is given by

$$
\begin{array} { r } { \partial _ { t } \xi _ { t } + \nabla \cdot ( v _ { t } \xi _ { t } ) = f _ { t } } \end{array}
$$

admits a unique solution for $v _ { \bullet } \in L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 }$ given by the methods of characteristics

$$
\xi _ { t } = ( \varphi _ { t } ^ { v } ) _ { \sharp } \xi _ { 0 } + \int _ { 0 } ^ { t } ( \varphi _ { t , s } ^ { v } ) _ { \sharp } f _ { s } \mathrm { d } s ,
$$

see for example $\left[ \mathrm { A C 1 4 } \right]$

Girsanov’s theorem We call the space $C _ { T } ( \mathbb { R } ^ { d } )$ of continuous paths the $d -$ dimensional path space and refer to probability measures on this space as path measures. It is well known that the $L ^ { 2 } .$ -costs on the level of velocity fields can be related to the Kullback–Leibler divergence between the path measures $\mathbb { P } ^ { v , \varepsilon }$ and $\mathbb { P } ^ { \star , \varepsilon }$ via Girsanov’s theorem. To state a version of Girsanov’s theorem, we consider two path measures $\mathbb { P } ^ { v , \varepsilon }$ and $\mathbb { P } ^ { w , \varepsilon }$ induced by the SDEs

$$
\begin{array} { r } { \mathrm { d } X _ { t } = v _ { t } ( X _ { t } ) \mathrm { d } t + \sqrt { \varepsilon } \mathrm { d } W _ { t } , \quad X _ { 0 } \sim p _ { 0 } } \\ { \mathrm { d } Y _ { t } = w _ { t } ( Y _ { t } ) \mathrm { d } t + \sqrt { \varepsilon } \mathrm { d } W _ { t } , \quad Y _ { 0 } \sim p _ { 0 } , } \end{array}\tag{2.3}
$$

where we assume weak existence and uniqueness in law. We set $u _ { t } ( x ) : = v _ { t } ( x )$ $w _ { t } ( x )$ and assume that Novikov’s condition

$$
\mathbb { E } _ { \mathbb { P } ^ { w , \varepsilon } } \left[ \exp \left( \frac { 1 } { 2 \varepsilon } \int \| u _ { t } ( X _ { t } ) \| _ { 2 } ^ { 2 } \mathrm { d } t \right) \right] < + \infty
$$

holds. Then we have

$$
\frac { \mathrm { d } \mathbb { P } ^ { v , \varepsilon } } { \mathrm { d } \mathbb { P } ^ { w , \varepsilon } } ( x _ { \bullet } ) = \exp \left( \frac { 1 } { \varepsilon } \int _ { 0 } ^ { 1 } u _ { t } ( x _ { t } ) \cdot ( \mathrm { d } x _ { t } - w _ { t } ( x _ { t } ) \mathrm { d } t ) - \frac { 1 } { 2 \varepsilon } \int _ { 0 } ^ { 1 } \| u _ { t } ( x _ { t } ) \| _ { 2 } ^ { 2 } \mathrm { d } t \right) .\tag{2.4}
$$

Note that as under $\mathbb { P } ^ { w , \varepsilon }$ we have $\mathrm { d } x _ { t } - w _ { t } ( x _ { t } ) \mathrm { d } t = \sqrt { \varepsilon } \mathrm { d } W _ { t }$ , the Radon-Nikodym derivative is often written as

$$
\frac { \mathrm { d } \mathbb { P } ^ { v , \varepsilon } } { \mathrm { d } \mathbb { P } ^ { w , \varepsilon } } ( x _ { \bullet } ) = \exp \left( \frac { 1 } { \sqrt { \varepsilon } } \int _ { 0 } ^ { 1 } u _ { t } ( x _ { t } ) \cdot \mathrm { d } W _ { t } - \frac { 1 } { 2 \varepsilon } \int _ { 0 } ^ { 1 } \| u _ { t } ( x _ { t } ) \| _ { 2 } ^ { 2 } \mathrm { d } t \right) .
$$

For a proof, we refer to $[ \mathrm { U Z } 1 3 ]$ for additive and to [NR21] for multiplicative noise, respectively. Girsanov’s theorem yields a direct expression of the relative entropy between the path measures $\mathbb P ^ { v }$ and $\mathbb { P } ^ { w }$ as a least-squares cost of the diference of the drifts $v _ { \bullet }$ and $w _ { \bullet }$ , which is well-known in the field of stochastic control and difusion processes, see e.g. [Fö85, Léo12b]. Recall that for two probability measures P and $\mathbb { Q }$ the relative entropy, Kullback–Leibler divergence or shortly KL-divergence is given by

$$
\begin{array} { r } { D _ { \mathrm { K L } } ( \mathbb { P } , \mathbb { Q } ) = \left\{ \begin{array} { l l } { \mathbb { E } _ { \mathbb { P } } \left[ \ln \left( \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { Q } } \right) \right] , } & { \mathrm { i f ~ } \mathbb { P } \ll \mathbb { Q } , } \\ { + \infty , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{2.5}
$$

Consider two path measures $\mathbb P ^ { v }$ and $\mathbb { P } ^ { w }$ induced by the SDE (2.1). Then it holds that

$$
D _ { \mathrm { K L } } ( \mathbb { P } ^ { w } , \mathbb { P } ^ { v } ) = \frac { 1 } { 2 \varepsilon } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| w _ { t } ( x ) - v _ { t } ( x ) \| _ { 2 } ^ { 2 } \rho _ { t } ^ { w } ( \mathrm { d } x ) \mathrm { d } t ,
$$

For a proof, we refer to $[ \mathrm { L S K ^ { + } } _ { 2 5 } ]$

## 2.3 Information geometry and natural gradients

We have seen that a least-squares objective on velocities can be viewed as a KL-divergence on the level of path measures. The KL-divergence is inherently connected to the negative entropy, which induces a natural Riemannian structure given by the Fisher–Rao metric, which is a central object in information geometry. For a more comprehensive introduction to information geometry, we refer to the extensive monographs $\mathrm { [ A m a 1 6 , A J L S 1 7 ] }$ . Consider a probability measure P and two variations $\bar { \mathbb { P } ^ { h } }$ and $\mathbb { Q } ^ { h }$ of P meaning that $\mathbb { P } ^ { 0 } = \mathbb { Q } ^ { 0 } \overset { \cdot } { = } \mathbb { P }$ , where we assume that $\mathbb { P } ^ { h } , \mathbb { Q } ^ { h } \ll \mathbb { P }$ and that the derivatives $\begin{array} { r } { \dot { \Xi } = \left. \frac { \mathrm { d } } { \mathrm { d } h } \right| _ { h = 0 } \mathbb { P } ^ { h } } \end{array}$ and $\begin{array} { r } { Z = \left. \frac { \mathrm { d } } { \mathrm { d } h } \right| _ { h = 0 } \mathbb { Q } ^ { h } } \end{array}$ exist in a suitable sense. Then, the Fisher–Rao metric is, given the existence of the expectation, given by

$$
g _ { \mathbb { P } } ^ { \mathrm { F R } } ( \Xi , Z ) = \mathbb { E } _ { \mathbb { P } } [  \frac { \mathrm { d } } { \mathrm { d } h } | _ { h = 0 } \ln ( \frac { \mathrm { d } \mathbb { P } ^ { h } } { \mathrm { d } \mathbb { P } } ) \frac { \mathrm { d } } { \mathrm { d } h } | _ { h = 0 } \ln ( \frac { \mathrm { d } \mathbb { Q } ^ { h } } { \mathrm { d } \mathbb { P } } ) ] .\tag{2.6}
$$

A precise meaning to the tangent space of the measures equivalent to $\mathbb { P }$ and in particular the derivatives $\textstyle { \frac { \mathrm { d } } { \mathrm { d } h } } \big | _ { h = 0 } \mathbb { P } ^ { h }$ and $\left. { \frac { \mathrm { d } } { \mathrm { d } h } } \right| _ { h = 0 } \mathbb { Q } ^ { h }$ was given by Pistone and Sempi in their seminal work $\left[ \mathrm { P S 9 5 } \right]$ . They showed that one can endow the space of equivalent measures as a smooth Banach manifold over the Orlicz space $L ^ { \Phi } ( \mathbb { P } )$ where $\Phi ( t ) = e ^ { | t | } - 1$ . In particular, this gives the following characterization of the tangent space

$$
T _ { \mathbb { P } } \mathcal { P } ( \mathbb { P } ) = \left\{ \left. \frac { \mathrm { d } } { \mathrm { d } h } \right| _ { h = 0 } \mathbb { P } ^ { h } : \left. \frac { \mathrm { d } } { \mathrm { d } h } \right| _ { h = 0 } \mathrm { l n } \frac { \mathrm { d } \mathbb { P } ^ { h } } { \mathrm { d } \mathbb { P } } \mathrm { e x i s t s ~ i n ~ } L ^ { \Phi } ( \mathbb { P } ) \right\} ,\tag{2.7}
$$

where $\mathcal { P } ( \mathbb { P } ) = \{ \mathbb { Q } : \mathbb { Q } \ll \mathbb { P }$ and $\mathbb { P } \ll \mathbb { Q } \}$ denotes the set of probability measures equivalent to P. In particular, for such variations $\mathbb { P } ^ { h }$ , the derivative $\frac { \mathrm { d } } { \mathrm { d } h } \big | _ { h = 0 } \mathbb { P } ^ { h }$ exists with respect to the total variation distance and is given by

$$
\left. { \frac { \mathrm { d } } { \mathrm { d } h } } \right| _ { h = 0 } \mathbb { P } ^ { h } = \left( \left. { \frac { \mathrm { d } } { \mathrm { d } h } } \right| _ { h = 0 } \ln { \frac { \mathrm { d } \mathbb { P } ^ { h } } { \mathrm { ~ d } \mathbb { P } } } \right) \mathbb { P } \in T _ { \mathbb { P } } { \mathcal { P } } ( \mathbb { P } ) .
$$

The topology on the tangent space is inherited from the Orlicz space $L ^ { \Phi } ( \mathbb { P } )$ , and the corresponding convergence is known as e-convergence. For a detailed discussion, we refer to the original construction $\left[ \mathrm { P S 9 5 } \right]$ and the more recent textbook $[ \mathrm { A J L S 1 7 } ]$ It is well known that the Fisher–Rao metric is inherently connected to the negative entropy, as it can be regarded as its Hessian. Much of information geometry is devoted to the study of the duality induced by this convex function. In particular, it induces a dually flat structure on the space of probability measures that leads to the following notions of geodesics, which are dual to each other. The m-geodesic and e-geodesic between two probability measures are given by linear interpolation and log-linear interpolation of the densities, respectively, meaning that

$$
\mathbb { P } _ { t } ^ { ( m ) } = ( 1 - t ) \mathbb { P } _ { 0 } + t \mathbb { P } _ { 1 } , \quad \mathrm { a n d } \ \frac { \mathrm { d } \mathbb { P } _ { t } ^ { ( e ) } } { \mathrm { d } \mathbb { Q } } \propto \left( \frac { \mathrm { d } \mathbb { P } _ { 0 } } { \mathrm { d } \mathbb { Q } } \right) ^ { 1 - t } \left( \frac { \mathrm { d } \mathbb { P } _ { 1 } } { \mathrm { d } \mathbb { Q } } \right) ^ { t } ,
$$

where $\mathbb { Q }$ is a reference measure dominating both $\mathbb { P } _ { 0 }$ and $\mathbb { P } _ { 1 }$ . Note that the e-geodesic is independent of the reference measure $\mathbb { Q } .$ The e-geodesic is also known as the exponential geodesic, while the m-geodesic is also known as the mixture geodesic. They are not geodesics with respect to the Fisher–Rao metric, but with respect to the dually flat connections, which are not metric connections.

Theorem 2.1 (Fisher–Rao gradients). ${ \mathit { I f D } } _ { \mathrm { K L } } ( \mathbb { P } ^ { \star } , \mathbb { P } ) < +$ and $D _ { \mathrm { K L } } ( \mathbb { P } , \mathbb { P } ^ { \star } ) < + \infty$ it holds that

$$
\begin{array} { r l } & { \nabla _ { \mathbb { P } } ^ { \mathrm { F R } } D _ { \mathrm { K L } } ( \mathbb { P } ^ { \star } , \mathbb { P } ) = \mathbb { P } - \mathbb { P } ^ { \star } , \quad a n d } \\ & { \nabla _ { \mathbb { P } } ^ { \mathrm { F R } } D _ { \mathrm { K L } } ( \mathbb { P } , \mathbb { P } ^ { \star } ) = \left( \ln \left( \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { P } ^ { \star } } \right) - D _ { \mathrm { K L } } ( \mathbb { P } , \mathbb { P } ^ { \star } ) \right) \mathbb { P } . } \end{array}
$$

This shows that the negative Fisher–Rao gradients offorward and backward $K L$ point into the directions of the m- and e-geodesics connecting $\mathbb { P } t o \mathbb { P } ^ { \star }$ , respectively.

Despite both identities being known in the literature, we provide a short, selfcontained proof in the appendix. These identities were used in various works obtaining $O ( e ^ { - t } )$ convergence guarantees for Fisher–Rao gradient flows [LLN19, $\mathrm { L C T } _ { 2 1 } , \mathrm { L S W } _ { 2 3 } , \mathrm { C H H } ^ { + } { } _ { 2 } 6 ]$ . In contrast to Wasserstein gradient flows, the convergence rate is independent of the target measure $\mathbb { P } ^ { \star }$ , which makes it an attractive choice for optimization and sampling. As the space of probability measures is infinite-dimensional, it is typically not feasible to compute the Fisher–Rao gradient directly on the level of probability measures. Hence, one resorts to discretizations of the space of probability measures, such as parametric families, which leads to the notion of natural gradients. Given a Riemannian manifold $( \mathcal { M } , g )$ , a parametric family $\mathcal { M } _ { \Theta } = \{ p _ { \theta } : \theta \in \mathbb { R } ^ { d _ { \mathfrak { p } } } \} \subseteq \mathcal { M }$ , and a loss function $L \colon \mathbb { R } ^ { d _ { \mathfrak { p } } }  \mathbb { R }$ given by $L ( \theta ) = E ( p _ { \theta } )$ , we call a solution $\delta \theta$ to the linear system

$$
G ( \theta ) \delta \theta = \nabla L ( \theta ) ,
$$

natural gradient, where $G ( \theta ) _ { i j } = g _ { p _ { \theta } } ( \partial _ { \theta _ { i } } p _ { \theta } , \partial _ { \theta _ { j } } p _ { \theta } )$ Here, $G ( \theta )$ is called the Gramian matrix. Natural gradients can also be shown to be the best approximation of the gradient descent dynamics in the manifold $\mathcal { M }$ while staying on the parametric model $\mathcal { M } _ { \Theta }$ . To make this precise, we denote the parametrization map by $P = ( \theta \mapsto$ $p _ { \theta } )$ . Then, for any natural gradient direction δθ it holds that

$$
D P ( \boldsymbol { \theta } ) \delta \boldsymbol { \theta } = \Pi _ { T _ { \boldsymbol { \theta } } , M _ { \Theta } } \nabla ^ { g } E ( p _ { \boldsymbol { \theta } } ) ,\tag{2.8}
$$

where $\Pi _ { T _ { \theta } \mathcal { M } _ { \Theta } }$ denotes the g-orthogonal projection onto the generalized tangent space

$$
T _ { \theta } \mathcal { M } _ { \Theta } = \operatorname { s p a n } \{ \partial _ { \theta _ { i } } p _ { \theta } : i = 1 , \dots , d _ { \mathfrak { p } } \} .
$$

For a proof, we refer to [Ama16] for regular Gramians and to [vOMA23] and $[ \mathrm { M } Z 2 3 ]$ for the degenerate and infinite-dimensional case, respectively. In the specific case of the Fisher–Rao metric, the Gramian is given by the Fisher-information matrix

$$
F ( \theta ) _ { i j } : = g _ { \mathbb { P } _ { \theta } } ^ { \mathrm { F R } } ( \partial _ { \theta _ { i } } \mathbb { P } _ { \theta } , \partial _ { \theta _ { j } } \mathbb { P } _ { \theta } ) = \mathbb { E } _ { \mathbb { P } _ { \theta } } \left[ \partial _ { \theta _ { i } } \ln \left( \frac { \mathrm { d } \mathbb { P } _ { \theta } } { \mathrm { d } \mathbb { Q } } \right) \cdot \partial _ { \theta _ { j } } \ln \left( \frac { \mathrm { d } \mathbb { P } _ { \theta } } { \mathrm { d } \mathbb { Q } } \right) \right] ,
$$

where $\mathbb { Q }$ is a reference measure dominating all $\mathbb { P } _ { \theta }$ . Note that the Fisher-information matrix is independent of the choice of $\mathbb { Q }$

## 3 The Advective Fisher–Rao Metric

In this section, we introduce the advective Fisher–Rao metric, which is the main object of this paper. Our definition is motivated by an explicit expression of the pullback of the Fisher–Rao metric on path measures of SDEs to the space of velocity fields. Further, we show that the advective Fisher–Rao metric maintains an important property of the Fisher–Rao metric, as it is compatible with the least-squares objective $E .$

## 3.1 The Fisher–Rao metric of path measures of SDEs

In the presence of noise

$$
\begin{array} { r } { \mathrm { d } X _ { t } = v _ { t } ( X _ { t } ) \mathrm { d } t + \sqrt { \varepsilon } \mathrm { d } W _ { t } , \quad X _ { 0 } \sim \rho _ { 0 } , } \end{array}\tag{3.1}
$$

the least-squares objective on the velocity fields agrees with the Kullback–Leibler divergence of the path measures. The Kullback–Leibler divergence naturally gives rise to the Fisher–Rao metric, as the Fisher–Rao metric is the local Hessian of the KL-divergence and its gradients of the KL-divergence point into the directions of the m- and e-geodesics, see Theorem 2.1. This motivates us to study the Fisher–Rao metric on the space of path measures of SDEs, and we obtain an explicit expression.

Theorem 3.1 (Fisher–Rao metric on path measures). Consider a velocityfield v<sub>•</sub> such that the SDE (3.1) admits a unique weak solution. Further, consider two velocity fields $u _ { \bullet }$ and $w _ { \bullet }$ and assume that there is $\delta > 0$ such that Novikov’s condition

$$
\mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ \exp \left( \frac { h ^ { 2 } } { 2 \varepsilon } \int _ { 0 } ^ { 1 } \| u _ { t } ( X _ { t } ) \| _ { 2 } ^ { 2 } { \mathrm { d } } t \right) \right] < + \infty \quad a n d\tag{3.2}
$$

$$
\mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ \exp \left( \frac { h ^ { 2 } } { 2 \varepsilon } \int _ { 0 } ^ { 1 } \| w _ { t } ( X _ { t } ) \| _ { 2 } ^ { 2 } \mathrm { d } t \right) \right] < + \infty\tag{3.3}
$$

holdsfor all $| h | < \delta$ and assume that the $S D E \left( 3 . 1 \right)$ admits a unique weak solution for $v _ { \bullet } + h u _ { \bullet }$ and $v _ { \bullet } +$ hw<sub>•</sub> for all $| h | < \delta .$ . Then $\begin{array} { r } { \Xi ^ { u , \varepsilon } = \frac { \mathrm { d } } { \mathrm { d } h } \bar { \Big | } _ { h = 0 } \mathbb { P } ^ { v + h u , \varepsilon } } \end{array}$ and $\begin{array} { r } { Z ^ { w , \varepsilon } \ = \ \frac { \mathrm { d } } { \mathrm { d } h } \big | _ { h = 0 } \mathbb { P } ^ { v + h w , \varepsilon } } \end{array}$ exist as elements in $T _ { \mathbb { P } ^ { v , \varepsilon } } \mathcal { P } ( \mathbb { P } ^ { v , \varepsilon } )$ and the Fisher–Rao metric is given by

$$
g _ { \mathbb { P } ^ { v , \varepsilon } } ^ { \mathrm { F R } } ( \Xi ^ { u , \varepsilon } , Z ^ { w , \varepsilon } ) = \frac { 1 } { \varepsilon } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } u _ { t } ( x ) \cdot w _ { t } ( x ) \rho _ { t } ^ { v , \varepsilon } ( \mathrm { d } x ) \mathrm { d } t ,\tag{3.4}
$$

where $\rho _ { t } ^ { v , \varepsilon }$ denotes the law of $X _ { t }$ under $\mathbb { P } ^ { v , \varepsilon }$

Proof. From the definition (2.7) of the tangent space, the statement that $\Xi ^ { u , \varepsilon }$ and $Z ^ { w , \varepsilon }$ exist as elements in the tangent space $T _ { \mathbb { P } ^ { v , \varepsilon } } \mathcal { P } ( \mathbb { P } ^ { v , \varepsilon } )$ is equivalent to the statement that the logarithmic densities

$$
\ln \left( \frac { \mathrm { d } \mathbb { P } ^ { v + h u , \varepsilon } } { \mathrm { ~ d } \mathbb { P } ^ { v , \varepsilon } } \right) \quad \mathrm { a n d ~ } \ln \left( \frac { \mathrm { d } \mathbb { P } ^ { v + h w , \varepsilon } } { \mathrm { ~ d } \mathbb { P } ^ { v , \varepsilon } } \right)
$$

are diferentiable with respect to h in the Orlicz space $L ^ { \Phi } ( \mathbb { P } ^ { v , \varepsilon } )$ , where $\Phi ( t ) = e ^ { | t | } - 1$ We use Girsanov’s theorem and, in particular, $( 2 . 4 )$ to express the Radon–Nikodym derivative of the path measures and obtain

$$
\ln \left( \frac { \mathrm { d } \mathbb { P } ^ { v + h u , \varepsilon } } { \mathrm { d } \mathbb { P } ^ { v , \varepsilon } } ( x _ { \bullet } ) \right) = h f ( x _ { \bullet } ) - h ^ { 2 } g ( x _ { \bullet } ) ,
$$

where

$$
f ( x _ { \bullet } ) = { \frac { 1 } { \varepsilon } } \int _ { 0 } ^ { 1 } u _ { t } ( x _ { t } ) \cdot ( \mathrm { d } x _ { t } - v _ { t } ( x _ { t } ) \mathrm { d } t ) \quad { \mathrm { a n d ~ } } g ( x _ { \bullet } ) = { \frac { 1 } { 2 \varepsilon } } \int _ { 0 } ^ { 1 } \| u _ { t } ( x _ { t } ) \| _ { 2 } ^ { 2 } \mathrm { d } t
$$

do not depend on $h . \ \operatorname { A s } h f + h ^ { 2 } g$ is quadratic in $h ,$ it sufices to show $f , g \in L ^ { \Phi } ( \mathbb { P } ^ { v , \varepsilon } )$ , meaning that

$$
\mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ e ^ { \alpha | f | } \right] , \mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ e ^ { \beta | g | } \right] < + \infty
$$

for some $\alpha , \beta > 0$ . From $_ { ( 3 . 2 ) }$ we immediately obtain

$$
\mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ e ^ { \beta | g | } \right] = \mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ \exp \left( \frac { \beta } { 2 \varepsilon } \int _ { 0 } ^ { 1 } \| u _ { t } ( X _ { t } ) \| _ { 2 } ^ { 2 } \mathrm { d } t \right) \right] < + \infty
$$

for $\beta < \delta ^ { 2 }$ . To show $\mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } [ e ^ { \alpha | f | } ] < + \infty$ we first bound $\mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } [ e ^ { \alpha f } ]$ . It holds that

$$
\mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ e ^ { \alpha f } \right] = \mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ \exp \left( { \frac { \alpha } { \sqrt { \varepsilon } } } \int _ { 0 } ^ { 1 } u _ { t } ( X _ { t } ) \cdot \mathrm { d } W _ { t } \right) \right]
$$

and therefore, we consider

$$
M _ { t } : = \frac { 1 } { \sqrt { \varepsilon } } \int _ { 0 } ^ { t } u _ { s } ( X _ { s } ) \cdot \mathrm { d } W _ { s } ,
$$

for $t \in [ 0 , 1 ]$ , which is a square-integrable martingale as Novikov’s condition $_ { ( 3 . 2 ) }$ implies

$$
\mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ \int _ { 0 } ^ { 1 } \lVert u _ { t } ( X _ { t } ) \rVert _ { 2 } ^ { 2 } \mathrm { d } t \right] < + \infty .
$$

Consider now the stochastic exponential of $2 \alpha M _ { t } .$ , which is given by

$$
Z _ { t } : = \exp \left( 2 \alpha M _ { t } - 2 \alpha ^ { 2 } \langle M \rangle _ { t } \right) .
$$

If $\alpha < \frac { \delta } { 2 }$ Novikov’s condition $_ { ( 3 . 2 ) }$ ensures

$$
\begin{array} { r l } & { \mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ \exp \left( \displaystyle \frac 1 2 \langle 2 \alpha M \rangle _ { 1 } \right) \right] = \mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ \exp \left( 2 \alpha ^ { 2 } \langle M \rangle _ { 1 } \right) \right] } \\ & { \qquad = \mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ \exp \left( \displaystyle \frac { 2 \alpha ^ { 2 } } { \varepsilon } \int _ { 0 } ^ { 1 } \| u _ { t } ( X _ { t } ) \| _ { 2 } ^ { 2 } \mathrm { d } t \right) \right] < + \infty . } \end{array}
$$

Hence, for $\begin{array} { r } { \alpha < \frac { \delta } { 2 } , } \end{array}$ , the stochastic exponential is a martingale and it holds that

$$
\begin{array} { r l } & { \mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ e ^ { \alpha f } \right] = \mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ e ^ { \alpha M _ { 1 } } \right] } \\ & { \qquad = \mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ Z _ { 1 } ^ { \frac 1 2 } e ^ { \alpha ^ { 2 } \langle M \rangle _ { 1 } } \right] } \\ & { \qquad \le \mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ Z _ { 1 } \right] ^ { \frac 1 2 } \mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ e ^ { 2 \alpha ^ { 2 } \langle M \rangle _ { 1 } } \right] ^ { \frac 1 2 } < + \infty . } \end{array}
$$

To bound $\mathbb { E } _ { \mathbb { P } ^ { v } , \varepsilon [ e ^ { - \alpha f } ] }$ , note that $\tilde { u } = - u$ also satisfies the same Novikov condition $_ { ( 3 . 2 ) }$ and therefore the same argument as above applies to $\tilde { f } = - f$ . Hence, $\mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } [ e ^ { - \alpha f } ] = \mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } [ e ^ { \alpha \tilde { f } } ] < + \infty$ for $\alpha < \frac { \delta } { 2 }$ . Finally, for $\alpha < \frac { \delta } { 2 }$ we have

$$
\begin{array} { r } { \mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ e ^ { \alpha | f | } \right] \le \mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ e ^ { \alpha f } \right] + \mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \left[ e ^ { - \alpha f } \right] < + \infty . } \end{array}
$$

Having established that $f , g \in L ^ { \Phi } ( \mathbb { P } ^ { v , \varepsilon } )$ , we can diferentiate with respect to $h$ and obtain

$$
\frac { \mathrm { d } } { \mathrm { d } h } \bigg \vert _ { h = 0 } \ln \bigg ( \frac { \mathrm { d } \mathbb { P } ^ { v + h u , \varepsilon } } { \mathrm { d } \mathbb { P } ^ { v , \varepsilon } } ( x _ { \bullet } ) \bigg ) = \frac { 1 } { \varepsilon } \int _ { 0 } ^ { 1 } u _ { t } ( x _ { t } ) \cdot ( \mathrm { d } x _ { t } - v _ { t } ( x _ { t } ) \mathrm { d } t )
$$

and analogously

$$
\frac { \mathrm { d } } { \mathrm { d } h } \bigg \vert _ { h = 0 } \ln \bigg ( \frac { \mathrm { d } \mathbb { P } ^ { v + h w , \varepsilon } } { \mathrm { d } \mathbb { P } ^ { v , \varepsilon } } ( x _ { \bullet } ) \bigg ) = \frac { 1 } { \varepsilon } \int _ { 0 } ^ { 1 } w _ { t } ( x _ { t } ) \cdot ( \mathrm { d } x _ { t } - v _ { t } ( x _ { t } ) \mathrm { d } t ) .
$$

Recall that $\varepsilon ^ { - \frac { 1 } { 2 } } ( \mathrm { d } X _ { t } - v _ { t } ( X _ { t } ) \mathrm { d } t ) = \mathrm { d } W _ { t }$ under $\mathbb { P } ^ { v , \varepsilon }$ . Using the Itô isometry, we obtain from the definition (2.6) of the Fisher–Rao metric that

$$
\begin{array} { r l } & { g _ { \mathbb { P } ^ { s , e } } ^ { \mathbb { F } \mathbb { R } } ( \Xi ^ { u , \varepsilon } , Z ^ { w , \varepsilon } ) = \mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \Bigg [ \frac { 1 } { \varepsilon ^ { 2 } } \int _ { 0 } ^ { 1 } u _ { t } ( X _ { t } ) \cdot ( \mathrm { d } X _ { t } - v _ { t } ( X _ { t } ) \mathrm { d } t ) \times } \\ & { \qquad \int _ { 0 } ^ { 1 } w _ { t } ( X _ { t } ) \cdot ( \mathrm { d } X _ { t } - v _ { t } ( X _ { t } ) \mathrm { d } t ) \Bigg ] } \\ & { \qquad = \cfrac { 1 } { \varepsilon } \cdot \mathbb { E } _ { \mathbb { P } ^ { v , \varepsilon } } \Bigg [ \int _ { 0 } ^ { 1 } u _ { t } ( X _ { t } ) \cdot w _ { t } ( X _ { t } ) \mathrm { d } t \Bigg ] } \\ & { \qquad = \cfrac { 1 } { \varepsilon } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } u _ { t } ( x ) \cdot w _ { t } ( x ) \rho _ { t } ^ { v , \varepsilon } ( \mathrm { d } x ) \mathrm { d } t . } \end{array}
$$

Novikov’s condition can readily be replaced by weaker conditions ensuring that Girsanov’s theorem holds. However, if both u and w have at most linear growth, then Novikov’s condition is always satisfied. To show this, we use basic notions of sub-Gaussian random variables and refer to [Ver18] for an extensive introduction to this concept. Recall that a random variable X is called sub-Gaussian if either of the following two equivalent conditions holds

1. There is $\lambda _ { 0 } > 0$ such that $\mathbb { E } [ e ^ { \lambda \| X \| _ { 2 } ^ { 2 } } ] < + \infty$ for all $\lambda < \lambda _ { 0 }$

2. There is $c > 0$ such that $\mathbb { P } ( \| X \| _ { 2 } > t ) \le 2 e ^ { - \frac { t ^ { 2 } } { c } }$

For a Gaussian ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } I )$ one can choose $\lambda _ { 0 } = ( 2 \sigma ^ { 2 } ) ^ { - 1 }$ . Finally, recall that the sum of two sub-Gaussians $X , Y$ with constants $c _ { X } , c _ { Y } > 0$ is again sub-Gaussian.

Lemma 3.2 (Linear growth implies Novikov). Let $v _ { \bullet } , \ u _ { \bullet }$ be velocity fields that satisfy the linear growth condition

$$
\begin{array} { r } { \left\| v _ { t } ( x ) \right\| _ { 2 } \leq a _ { t } ( 1 + \left\| x \right\| _ { 2 } ) a n d \left\| u _ { t } ( x ) \right\| _ { 2 } \leq b _ { t } ( 1 + \left\| x \right\| _ { 2 } ) f o r t \in [ 0 , 1 ] , x \in \mathbb { R } ^ { d } , } \end{array}
$$

for some $a \in L _ { T } ^ { 1 }$ and $b \in L _ { T } ^ { 2 }$ , assume that the SDE $\left( 3 . 1 \right)$ has a unique weak solution and assume that the initial condition is sub-Gaussian. Then, Novikov’s condition $_ { ( 3 . 2 ) }$ holdsfor a suitably small $\delta > 0$

Proof. We need to show that

$$
\left( \int _ { 0 } ^ { 1 } \| u _ { t } ( X _ { t } ) \| ^ { 2 } \mathrm { d } t \right) ^ { \frac { 1 } { 2 } }
$$

is a sub-Gaussian random variable, where $X _ { t }$ is the solution of the SDE (3.1). It holds that

$$
X _ { t } = X _ { 0 } + \int _ { 0 } ^ { t } v _ { s } ( X _ { s } ) \mathrm { d } s + \sqrt { \varepsilon } W _ { t } .
$$

The triangle inequality yields

$$
\begin{array} { l } { \displaystyle \| X _ { t } \| \leq \| X _ { 0 } \| + \int _ { 0 } ^ { t } \| v _ { s } ( X _ { s } ) \| \mathrm { d } s + \sqrt { \varepsilon } \| W _ { t } \| } \\ { \displaystyle \leq \| X _ { 0 } \| + \int _ { 0 } ^ { t } a _ { s } ( 1 + \| X _ { s } \| ) \mathrm { d } s + \sqrt { \varepsilon } \operatorname* { s u p } _ { s \in [ 0 , 1 ] } \| W _ { s } \| . } \end{array}
$$

Gronwall’s inequality yields

$$
\begin{array} { r l } & { \| X _ { t } \| \leq \left( \| X _ { 0 } \| + \| a \| _ { L _ { T } ^ { 1 } } + \sqrt { \varepsilon } \operatorname* { s u p } _ { s \in [ 0 , 1 ] } \| W _ { s } \| \right) e ^ { \| a \| _ { L _ { T } ^ { 1 } } } } \\ & { \qquad \leq c Z , } \end{array}
$$

where $Z = 1 + \| X _ { 0 } \| + { \sqrt { \varepsilon } } \operatorname* { s u p } _ { s \in [ 0 , 1 ] } \| W _ { s } \|$ . Note that $Z$ is sub-Gaussian as $\| X _ { 0 } \|$ is sub-Gaussian by assumption and su $\dot { \mathrm { p } } _ { s \in [ 0 , 1 ] } \lVert \boldsymbol { W } _ { s } \rVert$ is sub-Gaussian as the supremum of a Gaussian process. We estimate

$$
\| u _ { t } ( X _ { t } ) \| \leq b _ { t } ( 1 + \| X _ { t } \| ) \leq b _ { t } ( 1 + c Z )
$$

and consequently

$$
\int _ { 0 } ^ { 1 } \| u _ { t } ( X _ { t } ) \| ^ { 2 } \mathrm { d } t \leq \int _ { 0 } ^ { 1 } b _ { t } ^ { 2 } ( 1 + c Z ) ^ { 2 } \mathrm { d } t \leq 2 \| b \| _ { L _ { T } ^ { 2 } } ^ { 2 } ( 1 + c ^ { 2 } Z ^ { 2 } ) .
$$

As Z is sub-Gaussian, this finishes the proof.

Theorem 3.1 provides the explicit expression

$$
\begin{array} { l } { g _ { v _ { \bullet } } ^ { \mathrm { F R } , \varepsilon } ( u _ { \bullet } , w _ { \bullet } ) = { g _ { \mathbb { P } ^ { v , \varepsilon } } ^ { \mathrm { F R } } } (  \frac { \mathrm { d } } { \mathrm { d } h } | _ { h = 0 } \mathbb { P } ^ { v + h u , \varepsilon } , \frac { \mathrm { d } } { \mathrm { d } h } | _ { h = 0 } \mathbb { P } ^ { v + h w , \varepsilon } ) } \\ { \qquad = \displaystyle \frac { 1 } { \varepsilon } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } u _ { t } ( x ) \cdot w _ { t } ( x ) \rho _ { t } ^ { v , \varepsilon } ( \mathrm { d } x ) \mathrm { d } t } \end{array}\tag{3.5}
$$

of the pullback of the Fisher–Rao metric on path measures of SDEs to the space of velocity fields under strong regularity conditions. This definition, however, is meaningful on a much larger space of velocity fields.

Definition 3.3 (Fisher–Rao metric on velocity fields). Let $v _ { \bullet } \in L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 }$ and let $\rho _ { \bullet } ^ { v , \varepsilon }$ denote the densities of the solution of the SDE (3.1). For two velocity fields $u _ { \bullet } , w _ { \bullet }$ we refer to

$$
g _ { v _ { \bullet } } ^ { \mathrm { F R } , \varepsilon } ( u _ { \bullet } , w _ { \bullet } ) = \frac { 1 } { \varepsilon } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } u _ { t } ( x ) \cdot w _ { t } ( x ) \rho _ { t } ^ { v , \varepsilon } ( \mathrm { d } x ) \mathrm { d } t\tag{3.6}
$$

as the Fisher–Rao metric on the space of velocity fields whenever this integral is well-defined.

Remark 3.4 (Extensions). The expression of the Fisher–Rao metric can readily be extended to the case of multiplicative noise as well as to cases where the initial condition is not fixed. More precisely, in the case of multiplicative noise

$$
\mathrm { d } X _ { t } = v _ { t } ( X _ { t } ) \mathrm { d } t + \sigma _ { t } ( X _ { t } ) \mathrm { d } W _ { t } ,
$$

the Fisher–Rao metric is given by

$$
g _ { \mathbb { P } ^ { \nu } , \sigma } ^ { \mathrm { F R } } ( \Xi ^ { u , \sigma } , Z ^ { w , \sigma } ) = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } u _ { t } ( x ) \cdot \left( \sigma _ { t } ( x ) \sigma _ { t } ( x ) ^ { \top } \right) ^ { - 1 } w _ { t } ( x ) \rho _ { t } ^ { v , \sigma } ( \mathrm { d } x ) \mathrm { d } t .
$$

Here, we assume that $\sigma _ { t } ( x ) ^ { \top } \sigma _ { t } ( x )$ is invertible for all $t \in [ 0 , 1 ]$ and $\boldsymbol { x } \in \mathbb { R } ^ { d }$ and denote the path measure of the process by $\mathbb { P } ^ { v , \sigma }$ as well as the time martingals by $\rho ^ { v , \sigma }$ . In the case of a non-fixed initial condition, the Fisher–Rao metric is given by the sum of the Fisher–Rao metric on path measures with a fixed initial condition and the Fisher–Rao metric on the initial distribution. Both statements follow from a general version of Girsanov’s theorem incorporating multiplicative noise and non-fixed initial conditions, see [Ric21].

## 3.2 The advective Fisher–Rao metric and its optimality

Equation $( 3 . 4 )$ gives an explicit expression for the Fisher–Rao metric on the space of path measures induced by the SDE $\left( 3 . 1 \right)$ that blows up as $\varepsilon \to 0$ . It is natural to extend the Fisher–Rao metric by simply removing the factor of $\varepsilon ^ { - 1 }$ , which yields the following Riemannian metric.

Definition 3.5 (Advective Fisher–Rao metric on velocity fields). Consider a velocity field $v _ { \bullet } \in L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 }$ and denote the solution $\rho _ { \bullet } ^ { v }$ of the continuity equation

$$
\partial _ { t } \rho _ { t } ^ { v } + \nabla \cdot ( \rho _ { t } ^ { v } v _ { t } ) = 0 , \qquad \rho _ { 0 } ^ { v } = \rho _ { 0 } .\tag{3.7}
$$

For two velocity fields $u _ { \bullet } , w _ { \bullet }$ we refer to

$$
g _ { v _ { \bullet } } ^ { \mathrm { A F R } } ( u _ { \bullet } , w _ { \bullet } ) = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } u _ { t } ( x ) \cdot w _ { t } ( x ) \rho _ { t } ^ { v } ( \mathrm { d } x ) \mathrm { d } t\tag{3.8}
$$

as the advective Fisher–Rao (AFR) metric on the space of velocity fields whenever this integral is well-defined.

Remark 3.6 (Existence). Note that if $\rho _ { \bullet } ^ { v }$ has uniformly bounded second moments, the metric is well defined for $u _ { \bullet } , w _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 }$ . This is for example the case if $\rho _ { 0 } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ and $v _ { \bullet } \in L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } ,$ , see [AG12].

Similar to the construction of the Wasserstein–Otto metric in optimal transport theory, we define the tangent space as the completion of the space of continuously diferentiable Lipschitz vector fields with respect to the Fisher–Rao metric. This yields

$$
T _ { v \bullet } ^ { \mathrm { A F R } } L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } = { \overline { { L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 } } } } ^ { g _ { v \bullet } ^ { \mathrm { A F R } } } = L _ { T } ^ { 2 } L ^ { 2 } ( \rho _ { \bullet } ^ { v } )
$$

where

$$
L _ { T } ^ { 2 } L ^ { 2 } ( \rho _ { \bullet } ^ { v } ) : = \left\{ v \colon [ 0 , 1 ] \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { d } : \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \lVert v _ { t } ( x ) \rVert ^ { 2 } \rho _ { t } ^ { v } ( \mathrm { d } x ) \mathrm { d } t < + \infty \right\} .
$$

Note that the Fisher–Rao metric depends on $v _ { \bullet } .$ , unlike the $L ^ { 2 } .$ -metric, where the reference density $\rho _ { \bullet } ^ { \star }$ is used. Recall that we are investigating the energy function

$$
E \colon L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 } \to \mathbb { R } , \quad v _ { \bullet } \mapsto \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| v _ { t } ( x ) - v _ { t } ^ { \star } ( x ) \| _ { 2 } ^ { 2 } \rho _ { t } ^ { \star } ( \mathrm { d } x ) \mathrm { d } t .
$$

To ensure finiteness of the energy we assume $\rho _ { 0 } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ , in which case $\rho _ { t } ^ { \star }$ has finite second moments $m _ { 2 } ( \rho _ { t } ^ { \star } )$ , which are bounded uniformly in time, see for example [AG12] or Lemma $4 { \cdot } 4$ . Note that

$$
\lVert v _ { t } ( x ) - v _ { t } ^ { \star } ( x ) \rVert \leq \lVert v _ { t } - v _ { t } ^ { \star } \rVert _ { C _ { \mathrm { L i p } } ^ { 1 } } ( 1 + \lVert x \rVert )
$$

and we can estimate

$$
\begin{array} { r l } & { \| v _ { t } - v _ { t } ^ { \star } \| _ { L ^ { 2 } ( \rho _ { t } ^ { \star } ) } \leq \| v _ { t } - v _ { t } ^ { \star } \| _ { C _ { \mathrm { L i p } } ^ { 1 } } \| 1 + \| x \| \| _ { L ^ { 2 } ( \rho _ { t } ^ { \star } ) } } \\ & { \qquad \leq \| v _ { t } - v _ { t } ^ { \star } \| _ { C _ { \mathrm { L i p } } ^ { 1 } } ( 1 + \| \| x \| \| _ { L ^ { 2 } ( \rho _ { t } ^ { \star } ) } ) } \\ & { \qquad = \| v _ { t } - v _ { t } ^ { \star } \| _ { C _ { \mathrm { L i p } } ^ { 1 } } ( 1 + \sqrt { m _ { 2 } ( \rho _ { t } ^ { \star } ) } ) . } \\ & { \qquad = c \| v _ { t } - v _ { t } ^ { \star } \| _ { C _ { \mathrm { L i p } } ^ { 1 } } . } \end{array}
$$

This shows the finiteness of the energy E if $v _ { \bullet } , v _ { \bullet } ^ { \star } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 }$ and $p _ { 0 } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ . To understand the dynamics of the advective Fisher–Rao gradient flow, we first derive an explicit expression in the space of velocity fields.

Lemma 3.7. Assume that the initial distribution $p _ { 0 } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ has a positive density and consider $v _ { \bullet } , v _ { \bullet } ^ { \star } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 } .$ Then for any $u _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 }$ it holds that

$$
g _ { v _ { \bullet } } ^ { \mathrm { A F R } } \left( ( v _ { \bullet } - v _ { \bullet } ^ { \star } ) \cdot \frac { p _ { \bullet } ^ { \star } } { p _ { \bullet } ^ { v } } , u _ { \bullet } \right) = \delta E ( v _ { \bullet } ) [ u _ { \bullet } ] .
$$

Proof. First, note that positivity is preserved by the continuity equation. A direct computation yields that for any $u _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 }$ it holds that

$$
\begin{array} { l } { \displaystyle \delta E ( v _ { \bullet } ) [ u _ { \bullet } ] = \operatorname* { l i m } _ { h  0 } \frac { E ( v _ { \bullet } + h u _ { \bullet } ) - E ( v _ { \bullet } ) } { h } } \\ { \displaystyle = \operatorname* { l i m } _ { h  0 } \int _ { 0 } ^ { 1 } \int _ { \mathbb R ^ { d } } ( u _ { t } ( x ) \cdot ( v _ { t } ( x ) - v _ { t } ^ { \star } ( x ) ) + h \| u _ { t } ( x ) \| ^ { 2 } ) p _ { t } ^ { \star } ( x ) \mathrm { d } x \mathrm { d } t } \\ { \displaystyle = \int _ { 0 } ^ { 1 } \int _ { \mathbb R ^ { d } } ( v _ { t } ( x ) - v _ { t } ^ { \star } ( x ) ) \cdot u _ { t } ( x ) p _ { t } ^ { \star } ( x ) \mathrm { d } x \mathrm { d } t } \\ { \displaystyle = g _ { v _ { \bullet } } ^ { \mathrm { A F R } } ( ( v _ { \bullet } - v _ { \bullet } ^ { \star } ) \cdot \frac { p _ { \bullet } ^ { \star } } { p _ { \sigma } ^ { n } } , u _ { \bullet } ) , } \end{array}
$$

which yields the claim.

Note that it is not clear whether

$$
( v _ { \bullet } - v _ { \bullet } ^ { \star } ) \cdot \frac { p _ { \bullet } ^ { \star } } { p _ { \bullet } ^ { v } } \in L _ { T } ^ { 2 } L ^ { 2 } ( p _ { \bullet } ^ { v } ) = T _ { v _ { \bullet } } ^ { \mathrm { A F R } } L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } .
$$

As such, it is not guaranteed that the gradient of the energy functional exists in the tangent space, hence in the classical sense. However, it always exists in the larger space $L _ { T } ^ { 2 } L ^ { 1 } ( p _ { \bullet } ^ { v } )$ , as can be checked readily. Nevertheless, we write $\begin{array} { r } { \nabla ^ { \mathrm { A F R } } E ( v _ { \bullet } ) = ( v _ { \bullet } - v _ { \bullet } ^ { \star } ) \cdot \frac { p _ { \bullet } ^ { \star } } { p _ { \bullet } ^ { v } } } \end{array}$ in slightly informal fashion. To understand how the advective Fisher–Rao gradient flow afects the densities $p _ { \bullet } ^ { v }$ , we use the following general result.

Lemma 3.8 (Derivative of the solution operator). Let $v _ { \bullet } , u _ { \bullet } \in L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 }$ and assume that $\rho _ { 0 }$ has afinite second moment. Then, the limit

$$
\xi _ { t } ^ { u } : = \operatorname* { l i m } _ { h  0 } \frac { \rho _ { t } ^ { v + h u } - \rho _ { t } ^ { v } } h\tag{3.9}
$$

exists for every $t \in [ 0 , 1 ]$ as a weak- limit in $( C _ { \mathrm { L i p } } ^ { 1 } ) ^ { * }$ as well as a weak- limit in $( L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } ) ^ { * }$ . For any $\phi \in C _ { \mathrm { L i p } } ^ { 1 }$ it holds that

$$
\langle \xi _ { t } ^ { u } , \phi \rangle = \int _ { \mathbb { R } ^ { d } } \nabla \phi ( x ) \cdot \psi _ { t } ^ { u } ( ( \varphi _ { t } ^ { v } ) ^ { - 1 } ( x ) ) \rho _ { t } ( \mathrm { d } x ) ,
$$

where $\varphi _ { \bullet } ^ { v }$ denotes theflow induced by $v _ { \bullet }$ and $\psi _ { \bullet } ^ { u }$ is the unique solution to

$$
\partial _ { t } \psi _ { t } ^ { u } ( x ) = D v _ { t } ( \varphi _ { t } ^ { v } ( x ) ) \cdot \psi _ { t } ^ { u } ( x ) + u _ { t } ( \varphi _ { t } ^ { v } ( x ) ) .
$$

Further, $\xi _ { \bullet } ^ { u }$ is continuous as a curve in $( C _ { \mathrm { L i p } } ^ { 1 } ) ^ { * }$ with respect to the weak- topology and is a distributional solution of the continuity equation with source term given by

$$
\begin{array} { r } { \partial _ { t } \xi _ { t } ^ { u } + \nabla \cdot ( \xi _ { t } ^ { u } v _ { t } ) = - \nabla \cdot ( \rho _ { t } ^ { v } u _ { t } ) a n d \xi _ { 0 } = 0 . } \end{array}\tag{3.10}
$$

Proof. First, we show convergence of the diference quotients $\begin{array} { r } { \Delta _ { t } ^ { h } : = \frac { \rho _ { t } ^ { v + h u } - \rho _ { t } ^ { v } } { h } } \end{array}$ for a fixed $t \in [ 0 , 1 ]$ . For this, we take a function $\phi \in C _ { \mathrm { L i p } } ^ { 1 } ( \mathbb { R } ^ { d } )$ , and compute

$$
\begin{array} { r l } & { \langle \Delta _ { t } ^ { h } , \phi \rangle = h ^ { - 1 } \left( \displaystyle \int _ { \mathbb R ^ { d } } \phi ( x ) \rho _ { t } ^ { v + h u } (  { \mathrm { d } } x ) - \displaystyle \int _ { \mathbb R ^ { d } } \phi _ { t } ( x ) \rho _ { t } ^ { v } (  { \mathrm { d } } x ) \right) } \\ & { \qquad = \displaystyle \int _ { \mathbb R ^ { d } } \frac { \phi ( \varphi _ { t } ^ { v + h u } ( x ) ) - \phi ( \varphi _ { t } ^ { v } ( x ) ) } { h } \rho _ { 0 } (  { \mathrm { d } } x ) , } \end{array}\tag{3.11}
$$

where $\varphi _ { \bullet } ^ { v + h u }$ denotes the flow induced by the velocity field $v _ { \bullet } + h u _ { \bullet }$ . By Lemma $\mathrm { A } . 5$ , the pointwise limit is given by

$$
\operatorname* { l i m } _ { h \to 0 } \frac { \phi ( \varphi _ { t } ^ { v + h u } ( x ) ) - \phi ( \varphi _ { t } ^ { v } ( x ) ) } { h } = \nabla \phi ( \varphi _ { t } ^ { v } ( x ) ) \cdot \psi _ { t } ^ { u } ( x )
$$

for all $t \in [ 0 , 1 ] , x \in \mathbb { R } ^ { d }$ . By Lemma $\mathrm { { A . 4 } } .$ , there is a constant $c > 0$ such that

$$
\begin{array} { r } { \| \varphi _ { t } ^ { v + h u } ( x ) - \varphi _ { t } ^ { u } ( x ) \| \leq c ( 1 + \| x \| ) \quad \mathrm { f o r ~ a l l ~ } x \in \mathbb R ^ { d } , | h | \leq 1 . } \end{array}
$$

This yields

$$
\begin{array} { r l } & { \left| \frac { \phi ( \varphi _ { t } ^ { v + h u } ( x ) ) - \phi ( \varphi _ { t } ^ { v } ( x ) ) } { h } \right| \leq \| \phi \| _ { C _ { \mathrm { L i p } } ^ { 1 } } \cdot \frac { \| \varphi _ { t } ^ { v + h u } ( x ) - \varphi _ { t } ^ { u } ( x ) \| } { h } } \\ & { \qquad \leq \| \phi \| _ { C _ { \mathrm { L i p } } ^ { 1 } } \cdot c ( 1 + \| x \| ) . } \end{array}
$$

This provides a dominating function as

$$
\int _ { \mathbb { R } ^ { d } } \left. \frac { \phi ( \varphi _ { t } ^ { v + h u } ( x ) ) - \phi ( \varphi _ { t } ^ { v } ( x ) ) } { h } \right. \rho _ { 0 } ( \mathrm { d } x ) \leq \| \phi \| _ { C _ { \mathrm { L i p } } ^ { 1 } } c ( 1 + m _ { 1 } ( \rho _ { 0 } ) )
$$

and hence, we obtain

$$
\begin{array} { r l } & { \displaystyle \operatorname* { l i m } _ { h \to 0 } \langle \Delta _ { t } ^ { h } , \phi \rangle = \int _ { \mathbb { R } ^ { d } } \nabla \phi ( \varphi _ { t } ^ { v } ( x ) ) \cdot \psi _ { t } ^ { u } ( x ) \rho _ { 0 } ( \mathrm { d } x ) } \\ & { \quad \quad \quad = \int _ { \mathbb { R } ^ { d } } \nabla \phi ( x ) \cdot \psi _ { t } ^ { u } ( ( \varphi _ { t } ^ { v } ) ^ { - 1 } ( x ) ) \rho _ { t } ( \mathrm { d } x ) } \\ & { \quad \quad = : \langle \xi _ { t } ^ { u } , \phi \rangle . } \end{array}
$$

Note that this defines a linear bounded functional $\xi _ { t } ^ { u }$ with respect to ϕ as $\psi _ { t } ^ { u }$ has linear growth with a constant bounded uniformly in $t ,$ see for example Lemma $_ { \mathrm { A } . 2 }$ which ensures

$$
\begin{array} { r l } & { \displaystyle \left. \int _ { \mathbb { R } ^ { d } } \nabla \phi ( \varphi _ { t } ^ { v } ( x ) ) \cdot \psi _ { t } ^ { u } ( x ) \rho _ { 0 } ( \mathrm { d } x ) \right. \leq \displaystyle \int _ { \mathbb { R } ^ { d } } \left. \nabla \phi ( \varphi _ { t } ^ { v } ( x ) ) \cdot \psi _ { t } ^ { u } ( x ) \right. \rho _ { 0 } ( \mathrm { d } x ) } \\ & { \quad \quad \quad \quad \quad \quad \leq \displaystyle \lVert \phi \rVert _ { C _ { \mathrm { L i p } } ^ { 1 } } \int _ { \mathbb { R } ^ { d } } \lVert \psi _ { t } ^ { u } ( x ) \rVert \rho _ { 0 } ( \mathrm { d } x ) } \\ & { \quad \quad \quad \quad \quad \leq \displaystyle \lVert \phi \rVert _ { C _ { \mathrm { L i p } } ^ { 1 } } \int _ { \mathbb { R } ^ { d } } c ( 1 + \lVert x \rVert ) \rho _ { 0 } ( \mathrm { d } x ) } \\ & { \quad \quad \quad \quad \leq c ( 1 + m _ { 1 } ( \rho _ { 0 } ) ) \lVert \phi \rVert _ { C _ { \mathrm { L i p } } ^ { 1 } } . } \end{array}
$$

Note that the argument above also provides

$$
\operatorname* { l i m } _ { h \to 0 } \int _ { 0 } ^ { 1 } \langle \Delta _ { t } ^ { h } , \phi _ { t } \rangle { \mathrm { } } { \mathrm { } } \mathrm { d } t = \int _ { 0 } ^ { 1 } \langle \xi _ { t } ^ { u } , \phi _ { t } \rangle { \mathrm { } } { \mathrm { } } \mathrm { d } t\tag{3.12}
$$

for any $\phi _ { \bullet } \in L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 }$ as again

$$
\begin{array} { r } { | \langle \xi _ { t } ^ { u } , \phi _ { t } \rangle | \leq \| \phi _ { t } \| _ { C _ { \mathrm { L i p } } ^ { 1 } } \cdot c ( 1 + m _ { 1 } ( \rho _ { 0 } ) ) , } \end{array}
$$

which provides a dominating function in time. To see that the limit $\xi _ { \bullet } ^ { u }$ is a distributional solution of (3.10), we first note that for a test function $\phi _ { \bullet } \in C _ { \mathrm { c } } ^ { \infty } ( ( 0 , 1 ) \times$ $\mathbb { R } ^ { d } ; \mathbb { R } ^ { d } )$ the continuity equation ensures

$$
\int _ { 0 } ^ { 1 } \left. \rho _ { t } ^ { v + h u } , \partial _ { t } \phi _ { t } + v _ { t } \cdot \nabla \phi _ { t } + h u _ { t } \cdot \nabla \phi _ { t } \right. \mathrm { d } t = 0
$$

for any $h \in \mathbb { R }$ . Taking the diference quotient, we obtain

$$
\int _ { 0 } ^ { 1 } \left( \left. \rho _ { t } ^ { v + h u } , u _ { t } \cdot \nabla \phi _ { t } \right. + \left. \Delta _ { t } ^ { h } , \partial _ { t } \phi _ { t } + v _ { t } \cdot \nabla \phi _ { t } \right. \right) \mathrm { d } t = 0 .
$$

From (3.12) we obtain

$$
\operatorname* { l i m } _ { h \to 0 } \int _ { 0 } ^ { 1 } \left. \Delta _ { t } ^ { h } , \partial _ { t } \phi _ { t } + v _ { t } \cdot \nabla \phi _ { t } \right. \mathrm { d } t = \int _ { 0 } ^ { 1 } \left. \xi _ { t } ^ { u } , \partial _ { t } \phi _ { t } + v _ { t } \cdot \nabla \phi _ { t } \right. \mathrm { d } t
$$

as $\partial _ { t } \phi _ { \bullet } + v _ { \bullet } \cdot \nabla \phi _ { \bullet } \in L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 }$ . To treat the second term, we use that $p _ { t } ^ { v + h u } \to p _ { t } ^ { v }$ in $W _ { 2 }$ as $h  0$ , see Lemma 4.10 below. Since $W _ { 2 }$ convergence implies convergence when testing with functions with at most quadratic growth, see $[ \mathrm { A G } 1 2 $ , Proposition $3 { \cdot } 4 ]$ , and $u _ { \bullet } \cdot \nabla \phi _ { \bullet } \in L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } ( \mathbb { R } ^ { d } )$ , we can use dominated convergence to obtain

$$
\operatorname* { l i m } _ { h \to 0 } \left. \rho _ { t } ^ { v + h u } , u _ { t } \cdot \nabla \phi _ { t } \right. = \operatorname* { l i m } _ { h \to 0 } \left. \rho _ { t } ^ { v } , u _ { t } \cdot \nabla \phi _ { t } \right.
$$

for a fixed time $t \in [ 0 , 1 ]$ . Finally, we obtain a majorant in time, as for any $| h | \leq 1$ we can estimate

$$
\begin{array} { r l } & { \left| \left. \rho _ { t } ^ { v + h u } , u _ { t } \cdot \nabla \phi _ { t } \right. \right| \leq \displaystyle \int _ { \mathbb R ^ { d } } | u _ { t } ( x ) \cdot \nabla \phi _ { t } ( x ) | \rho _ { t } ^ { v + h u } ( \mathrm { d } x ) } \\ & { \qquad \leq \displaystyle \int _ { \mathbb R ^ { d } } \| u _ { t } ( x ) \| \cdot \| \nabla \phi _ { t } ( x ) \| \rho _ { t } ^ { v + h u } ( \mathrm { d } x ) } \\ & { \qquad \leq \| \nabla \phi _ { t } \| _ { C _ { \mathrm { l i p } } ^ { 1 } } \displaystyle \int _ { \mathbb R ^ { d } } c ( 1 + \| \varphi _ { t } ^ { v + h u } ( x ) \| ) \rho _ { 0 } ( \mathrm { d } x ) } \\ & { \qquad \leq \| \nabla \phi _ { t } \| _ { C _ { \mathrm { l i p } } ^ { 1 } } \displaystyle \int _ { \mathbb R ^ { d } } C ( 1 + \| x \| ) \rho _ { 0 } ( \mathrm { d } x ) } \\ & { \qquad \leq \| \nabla \phi _ { t } \| _ { C _ { \mathrm { l i p } } ^ { 1 } } C ( 1 + m _ { 1 } ( \rho _ { 0 } ) ) , } \end{array}
$$

where we used that the flow $\varphi _ { t } ^ { v + h u }$ has at most linear growth with a constant uniform in time and $| h | \leq 1$ , see Lemma $_ { \mathrm { A } . 2 } ^ { \mathrm { ~ A ~ } . }$ . Overall, we obtain

$$
\int _ { 0 } ^ { 1 } \langle \xi _ { t } ^ { u } , \partial _ { t } \phi _ { t } + v _ { t } \cdot \nabla \phi _ { t } \rangle \mathrm { d } t = - \int _ { 0 } ^ { 1 } \langle \rho _ { t } ^ { v } , u _ { t } \cdot \nabla \phi _ { t } \rangle \mathrm { d } t ,
$$

which shows that $\xi _ { \bullet } ^ { u }$ is a distributional solution to

$$
\begin{array} { r } { \partial _ { t } \xi _ { t } + \nabla \cdot ( \xi _ { t } \boldsymbol { v } _ { t } ) = - \nabla \cdot ( \rho _ { t } \boldsymbol { u } _ { t } ) . } \end{array}
$$

Theorem 3.9 (Fisher–Rao gradient and m-geodesics). Consider $v _ { \bullet } , v _ { \bullet } ^ { \star } \in L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 }$ as well as an initial distribution $\rho _ { 0 } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ . Assume that $\rho _ { \bullet } ^ { \star } \ll \rho _ { \bullet } ^ { v }$ and that $\begin{array} { r } { u _ { \bullet } : = \nabla ^ { \mathrm { F R } } E ( v _ { \bullet } ) = ( v _ { \bullet } - v _ { \bullet } ^ { \star } ) \cdot \frac { \mathrm { d } \rho _ { \bullet } ^ { \star } } { \mathrm { d } \rho _ { \bullet } ^ { v } } \in L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } . } \end{array}$ . Then, it holds that

$$
\left. \frac { \mathrm { d } } { \mathrm { d } h } \right| _ { h = 0 } \rho _ { \bullet } ^ { v + h u } = \rho _ { \bullet } ^ { v } - \rho _ { \bullet } ^ { \star } .\tag{3.13}
$$

Proof. We set $\begin{array} { r } { \xi _ { \bullet } : = \left. \frac { \mathrm { d } } { \mathrm { d } h } \right| _ { h = 0 } \rho _ { \bullet } ^ { v + h u } } \end{array}$ . By Lemma $3 . 8$ we have

$$
\begin{array} { r } { \partial _ { t } \xi _ { t } + \nabla \cdot ( \xi _ { t } v _ { t } ) = - \nabla \cdot ( \rho _ { t } u _ { t } ) . } \end{array}
$$

Compare this to the tangent vector $\zeta _ { \bullet } = p _ { \bullet } ^ { v } - p _ { \bullet } ^ { \star }$ , for which the continuity equation yields

$$
\begin{array} { r l } & { \partial _ { t } \zeta _ { t } = \partial _ { t } \rho _ { t } ^ { v } - \partial _ { t } \rho _ { t } ^ { \star } } \\ & { \qquad = - \nabla \cdot ( \rho _ { t } ^ { v } v _ { t } - \rho _ { t } ^ { \star } v _ { t } ^ { \star } ) } \\ & { \qquad = - \nabla \cdot ( \zeta _ { t } v _ { t } + \rho _ { t } ^ { \star } v _ { t } - \rho _ { t } ^ { \star } v _ { t } ^ { \star } ) } \\ & { \qquad = - \nabla \cdot ( \zeta _ { t } v _ { t } + \rho _ { t } u _ { t } ) . } \end{array}
$$

Note that both $\xi _ { \bullet }$ and $\zeta _ { \bullet }$ are weak- continuous in $( C _ { \mathrm { L i p } } ^ { 1 } ) ^ { * }$ with respect to t and satisfy $\xi _ { 0 } = \zeta _ { 0 } = 0$ . Subtracting the two equations, we obtain

$$
\partial _ { t } ( \xi _ { t } - \zeta _ { t } ) + \nabla \cdot ( v _ { t } ( \xi _ { t } - \zeta _ { t } ) ) = 0
$$

as well as $\xi _ { 0 } - \zeta _ { 0 } = 0$ . By the method of characteristics, 0 is the unique solution of this equation, which yields $\xi _ { \bullet } = \zeta _ { \bullet }$ □

Informally, this result suggests that, given its existence, the advective Fisher–Rao gradient flow

$$
\partial _ { \tau } v _ { \bullet } ^ { \tau } = - \nabla ^ { \mathrm { A F R } } E ( v _ { \bullet } ^ { \tau } )
$$

leads to dynamics of the probability densities $\rho _ { \bullet } ^ { \tau } = \rho _ { \bullet } ^ { v ^ { \tau } }$ , which are given by

$$
\rho _ { \bullet } ^ { \tau } = \rho _ { \bullet } ^ { \star } + e ^ { - \tau } ( \rho _ { \bullet } ^ { 0 } - \rho _ { \bullet } ^ { \star } ) .
$$

This means that the density path $p _ { \bullet } ^ { \tau }$ converges to $p _ { \bullet } ^ { \star }$ at a rate of $O ( e ^ { - \tau } )$ along the linear interpolation, which is known as a mixture geodesic in information geometry [Ama16, AJLS17]. In particular, this shows that the advective Fisher–Rao metric is compatible with E, thereby providing an afirmative answer to the original question.

Remark 3.10 (Extension). Consider the case that the least-squares objective is not with respect to the Euclidean norm, but given by

$$
E ( v _ { \bullet } ) : = \frac 1 2 \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \lVert \sigma _ { t } ( x ) ^ { - 1 } ( v _ { t } ( x ) - v _ { t } ^ { \star } ( x ) ) \rVert _ { 2 } ^ { 2 } \rho _ { t } ^ { \star } ( \mathrm { d } x ) \mathrm { d } t ,
$$

where, we assume that $\sigma _ { t } ( x )$ is invertible for all $t \in [ 0 , 1 ]$ and $\boldsymbol { x } \in \mathbb { R } ^ { d }$ . Recall that for the noisy case with multiplicative noise

$$
\mathrm { d } X _ { t } = v _ { t } ( X _ { t } ) \mathrm { d } t + \sigma _ { t } ( X _ { t } ) \mathrm { d } W _ { t } ,
$$

the Fisher–Rao metric is given by

$$
g _ { \mathbb { R } ^ { \nu } , \sigma } ^ { \mathrm { F R } } ( \Xi ^ { u , \sigma } , Z ^ { w , \sigma } ) = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } u _ { t } ( x ) \cdot \left( \sigma _ { t } ( x ) \sigma _ { t } ( x ) ^ { \top } \right) ^ { - 1 } w _ { t } ( x ) \rho _ { t } ^ { v , \sigma } ( \mathrm { d } x ) \mathrm { d } t ,
$$

where we denote the path measure of the process by $\mathbb { P } ^ { v , \sigma }$ as well as the time marginals by $\rho ^ { v , \sigma }$ . A natural extension of the advective Fisher–Rao metric to a non-Euclidean geometry is now given by

$$
g _ { v _ { \bullet } } ^ { \mathrm { A F R } , \sigma } ( u _ { \bullet } , w _ { \bullet } ) = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } u _ { t } ( x ) \cdot \Big ( \sigma _ { t } ( x ) \sigma _ { t } ( x ) ^ { \top } \Big ) ^ { - 1 } w _ { t } ( x ) \rho _ { t } ^ { v } ( \mathrm { d } x ) \mathrm { d } t .
$$

## 4 The Advective Fisher–Rao Geometry from Three Perspectives

We have motivated the advective Fisher–Rao metric via an explicit expression of the Fisher–Rao metric of path measures of SDEs. Here, we show that the rescaled zero-noise limit of these Fisher–Rao metrics does indeed converge to the advective Fisher–Rao metric. Motivated by this characterization as a zero-noise limit, we connect the advective Fisher–Rao metric to the Freidlin–Wentzell rate functional, which describes the concentration of the path measures of the SDE around the ODE. Finally, we study the relation of the advective Fisher–Rao metric to optimal transport and show that it agrees with the Hessian of the Benamou–Brenier action functional.

## 4.1 Information geometry

A central object in information geometry is the Fisher–Rao metric, which arises naturally from the Hessian of the Kullback– Leibler divergence. We now show that the advective Fisher–Rao metric on velocity fields can be obtained as the rescaled zero-noise limit of the Fisher–Rao metric on path measures.

Theorem 4.1. Consider $v _ { \bullet } \in L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 }$ and assume that $\rho _ { 0 } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ . Then we obtain the advective Fisher–Rao metric as the rescaled zero-noise limit of the Fisher–Rao metrics on path measures, that is,for any $u _ { \bullet } , w _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 }$ we have that

$$
g _ { v \bullet } ^ { \mathrm { A F R } } ( u _ { \bullet } , w _ { \bullet } ) = \operatorname* { l i m } _ { \varepsilon  0 } \varepsilon g _ { v \bullet } ^ { \mathrm { F R } , \varepsilon } ( u _ { \bullet } , w _ { \bullet } ) .
$$

We require some auxiliary results.

Lemma 4.2 (Wasserstein convergence of time marginals). Consider a velocity field $v _ { \bullet }$ , which satisfies the one-sided Lipschitz condition

$$
( x - y ) \cdot ( v _ { t } ( x ) - v _ { t } ( y ) ) \leq \beta _ { t } \| x - y \| ^ { 2 } \quad f o r a l l \ t \in [ 0 , 1 ] , x , y \in \mathbb { R } ^ { d }
$$

for some $\beta \in L _ { \tau } ^ { 1 }$ and assume thatfor every $\varepsilon \geq 0$ the SDE $\left( 3 . 1 \right)$ admits a unique weak solution with path measure $\mathbb { P } ^ { v , \varepsilon }$ and assume that the corresponding time marginals $\rho _ { t } ^ { v , \varepsilon }$ have finite second moments, which are uniformly bounded in $\varepsilon \in [ 0 , \delta ]$ and $t \in [ 0 , 1 ]$ for some $\delta > 0$ . Thenfor all $\varepsilon \in [ 0 , \delta ]$ and $t \in [ 0 , 1 ]$ it holds that

$$
W _ { 2 } ^ { 2 } ( \rho _ { t } ^ { v , \varepsilon } , \rho _ { t } ^ { v } ) \leq \varepsilon d t e ^ { 2 \int _ { 0 } ^ { t } \beta _ { s } \mathrm { d } s } .
$$

Proof. Let us denote the solutions of the SDE $\left( 3 . 1 \right)$ and the corresponding ODE by $X _ { t } ^ { \varepsilon }$ for $\varepsilon > 0$ and $X _ { t }$ for $\varepsilon = 0$ , respectively. Then, $( X _ { t } ^ { \varepsilon } , X _ { t } )$ is a coupling of $\rho _ { t } ^ { v , \bar { \varepsilon } }$ and $\rho _ { t } ^ { v }$ and hence

$$
W _ { 2 } ^ { 2 } ( \rho _ { t } ^ { v , \varepsilon } , \rho _ { t } ^ { v } ) \leq \mathbb { E } \left[ \| X _ { t } ^ { \varepsilon } - X _ { t } \| ^ { 2 } \right] .
$$

We set $Z _ { t } = X _ { t } ^ { \varepsilon } - X _ { t }$ , which satisfies the SDE

$$
\mathrm { d } Z _ { t } = ( v _ { t } ( X _ { t } ^ { \varepsilon } ) - v _ { t } ( X _ { t } ) ) \mathrm { d } t + \sqrt { \varepsilon } \mathrm { d } W _ { t } \quad Z _ { 0 } = 0 .
$$

Using Itô’s formula, we obtain

$$
\| Z _ { t } \| ^ { 2 } = 2 \int _ { 0 } ^ { t } Z _ { s } \cdot ( v _ { s } ( X _ { s } ^ { \varepsilon } ) - v _ { s } ( X _ { s } ) ) \mathrm { d } s + 2 \sqrt { \varepsilon } \int _ { 0 } ^ { t } Z _ { s } \cdot \mathrm { d } W _ { s } + d \varepsilon t .
$$

By assumption, $X _ { t } ^ { \varepsilon }$ and $X _ { t }$ have uniformly bounded second moments and hence $\begin{array} { r } { \mathbb { E } [ \int _ { 0 } ^ { 1 } \lVert Z _ { t } \rVert _ { 2 } ^ { 2 } \mathrm { d } t ] < + \infty } \end{array}$ . This shows that the stochastic integral $\int _ { 0 } ^ { t } Z _ { s } \cdot \mathrm { d } W _ { s }$ is a martingale with expectation zero and hence its expectation vanishes. Using the one-sided Lipschitz continuity of $v _ { \bullet }$ , we obtain

$$
\mathbb { E } \left[ \Vert Z _ { t } \Vert ^ { 2 } \right] \leq 2 \int _ { 0 } ^ { t } \beta _ { s } \mathbb { E } \left[ \Vert Z _ { s } \Vert ^ { 2 } \right] \mathrm { d } s + d \varepsilon t .
$$

Now, by Grönwall’s inequality we obtain

$$
\begin{array} { r } { \mathbb { E } \left[ \| Z _ { t } \| ^ { 2 } \right] \leq d \varepsilon t e ^ { 2 \int _ { 0 } ^ { t } \beta _ { s } \mathtt { d } s } , } \end{array}
$$

which finishes the proof.

Lemma 4.3 (Bounding second moments by Wasserstein-2). Consider twoprobability measures $\mu , \nu \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ with finite second moments $m _ { 2 } ( \mu )$ and $m _ { 2 } ( \nu )$ . Then it holds that

$$
\left| \sqrt { m _ { 2 } ( \mu ) } - \sqrt { m _ { 2 } ( \nu ) } \right| \le W _ { 2 } ( \mu , \nu )
$$

as well as

$$
m _ { 2 } ( \mu ) \leq m _ { 2 } ( \nu ) + 2 \sqrt { m _ { 2 } ( \nu ) } W _ { 2 } ( \mu , \nu ) + W _ { 2 } ^ { 2 } ( \mu , \nu ) .
$$

Proof. It holds that

$$
\| x \| \leq \| y \| + \| x - y \| \quad { \mathrm { f o r ~ a l l ~ } } x , y \in \mathbb { R } ^ { d } .
$$

Hence, if we set $f ( x , y ) : = \| x \| , g ( x , y ) : = \| y \|$ , and $h ( x , y ) : = \| x - y \|$ we obtain for any coupling π of $\mu$ and ν that

$$
\| f \| _ { L ^ { 2 } ( \pi ) } \leq \| g + h \| _ { L ^ { 2 } ( \pi ) } \leq \| g \| _ { L ^ { 2 } ( \pi ) } + \| h \| _ { L ^ { 2 } ( \pi ) } .
$$

Note that $\| f \| _ { L ^ { 2 } ( \pi ) } = \sqrt { m _ { 2 } ( \mu ) } , \| g \| _ { L ^ { 2 } ( \pi ) } = \sqrt { m _ { 2 } ( \nu ) }$ for any coupling π. Taking the infimum over all couplings π yields

$$
\sqrt { m _ { 2 } ( \mu ) } \le \sqrt { m _ { 2 } ( \nu ) } + W _ { 2 } ( \mu , \nu ) .
$$

By symmetry, we also obtain ${ \sqrt { m _ { 2 } ( \nu ) } } \leq { \sqrt { m _ { 2 } ( \mu ) } } + W _ { 2 } ( \mu , \nu )$ and hence the desired result. Finally, the second estimate follows from the first one as

$$
\begin{array} { c } { { m _ { 2 } ( \mu ) - m _ { 2 } ( \nu ) = \left( \sqrt { m _ { 2 } ( \mu ) } - \sqrt { m _ { 2 } ( \nu ) } \right) \left( \sqrt { m _ { 2 } ( \mu ) } + \sqrt { m _ { 2 } ( \nu ) } \right) } } \\ { { \leq W _ { 2 } ( \mu , \nu ) \left( \sqrt { m _ { 2 } ( \mu ) } + \sqrt { m _ { 2 } ( \nu ) } \right) } } \\ { { \leq W _ { 2 } ( \mu , \nu ) \left( \sqrt { m _ { 2 } ( \nu ) } + \sqrt { m _ { 2 } ( \nu ) } + W _ { 2 } ( \mu , \nu ) \right) } } \\ { { = 2 \sqrt { m _ { 2 } ( \nu ) } W _ { 2 } ( \mu , \nu ) + W _ { 2 } ^ { 2 } ( \mu , \nu ) . } } \end{array}
$$

Lemma 4.4 (Growth of Wasserstein-2). Consider a velocity field $v _ { \bullet }$ such that there arefunctions $\alpha , \beta \in L _ { T } ^ { 1 }$ satisfying

$$
\begin{array} { r } { \| v _ { t } ( 0 ) \| \leq \alpha _ { t } , \quad \| v _ { t } ( x ) - v _ { t } ( y ) \| \leq \beta _ { t } \| x - y \| \quad f o r { a l l } { t } \in [ 0 , 1 ] , x , y \in { \mathbb R } ^ { d } , } \end{array}
$$

and consider $\rho _ { 0 } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ , and denote the solution of the continuity equation $\rho _ { \bullet } ^ { v } .$ Then we have

$$
W _ { 2 } ( \rho _ { t } ^ { v } , \rho _ { 0 } ) \leq \| \alpha \| _ { L ^ { 1 } ( [ 0 , t ] ) } e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , t ] ) } } + \sqrt { m _ { 2 } ( \rho _ { 0 } ) } \left( e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , t ] ) } } - 1 \right)
$$

and

$$
\sqrt { m _ { 2 } ( \rho _ { t } ^ { v } ) } \leq \| \alpha \| _ { L ^ { 1 } ( [ 0 , t ] ) } e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , t ] ) } } + \sqrt { m _ { 2 } ( \rho _ { 0 } ) } e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , t ] ) } } .
$$

Proof. By the definition of the Wasserstein distance, we have

$$
W _ { 2 } ^ { 2 } ( \rho _ { t } ^ { v } , \rho _ { 0 } ) \leq \int _ { \mathbb { R } ^ { d } } \lVert \varphi _ { t } ^ { v } ( x ) - x \rVert ^ { 2 } \rho _ { 0 } ( \mathrm { d } x ) = \lVert \varphi _ { t } ^ { v } ( x ) - x \rVert _ { L ^ { 2 } ( \rho _ { 0 } ) } ^ { 2 } .
$$

Note that pointwise $\| \varphi _ { t } ^ { v } ( x ) - x \| \leq A + B \| x \|$ for

$$
A = \| \alpha \| _ { L ^ { 1 } ( [ 0 , t ] ) } e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , t ] ) } } \quad \mathrm { a n d } B = ( e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , t ] ) } } - 1 )
$$

follows from Theorem A.2. Hence, using the triangle inequality, we obtain

$$
\| \varphi _ { t } ^ { v } ( x ) - x \| _ { L ^ { 2 } ( \rho _ { 0 } ) } \leq \| A + B \| x \| \| _ { L ^ { 2 } ( \rho _ { 0 } ) } \leq A + B \| \| x \| \| _ { L ^ { 2 } ( \rho _ { 0 } ) } ,
$$

where $\| \| x \| \| _ { L ^ { 2 } ( \rho _ { 0 } ) } = \sqrt { m _ { 2 } ( \rho _ { 0 } ) }$ . The proof for the bound of the second moment follows analogously. □

Proof of Theorem $4 . 1 \cdot$ . From Definition $3 { \cdot } 3$ we obtain

$$
\varepsilon g _ { \mathbb { P } ^ { v , \varepsilon } } ^ { \mathrm { F R } } ( u , w ) = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } u _ { t } ( x ) \cdot w _ { t } ( x ) \rho _ { t } ^ { v , \varepsilon } ( \mathrm { d } x ) \mathrm { d } t .
$$

First note that for fixed $t \in [ 0 , 1 ]$ we have $\rho _ { t } ^ { v , \varepsilon } \to \rho _ { t } ^ { v }$ in $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ by Lemma $4 { \cdot } 2$ . As the integrand $| u _ { t } ( x ) \cdot w _ { t } ( x ) |$ is continuous and has at most quadratic growth in x it holds that

$$
\operatorname* { l i m } _ { \varepsilon \to 0 } \int _ { \mathbb { R } ^ { d } } u _ { t } ( x ) \cdot w _ { t } ( x ) \rho _ { t } ^ { v , \varepsilon } ( \mathrm { d } x ) = \int _ { \mathbb { R } ^ { d } } u _ { t } ( x ) \cdot w _ { t } ( x ) \rho _ { t } ^ { v } ( \mathrm { d } x ) ,
$$

see for example $[ \mathrm { A G } 1 2 ,$ , Proposition $3 { \cdot } 4 ]$ . Hence, it remains to find an integrable dominating function in time. Note that as $u _ { \bullet } , w _ { \bullet } \in L _ { T } ^ { 2 } ( C _ { \mathrm { L i p } } ^ { 1 } ( \mathbb { R } ^ { d } ) )$ there are functions $a , b \in L _ { T } ^ { 2 }$ such that

$$
\left\| u _ { t } ( x ) \right\| \leq a _ { t } ( 1 + \left\| x \right\| ) \quad { \mathrm { a n d ~ } } \left\| w _ { t } ( x ) \right\| \leq b _ { t } ( 1 + \left\| x \right\| )
$$

for all $t \in [ 0 , 1 ]$ and $\boldsymbol { x } \in \mathbb { R } ^ { d }$ . Hence, we have

$$
\begin{array} { r l } & { \displaystyle \int _ { 0 } ^ { 1 } \int _ { \mathbb R ^ { d } } \lvert u _ { t } ( x ) \cdot w _ { t } ( x ) \rvert \rho _ { t } ^ { v , \varepsilon } ( \mathrm { d } x ) \mathrm { d } t \leq \displaystyle \int _ { 0 } ^ { 1 } a _ { t } b _ { t } \int _ { \mathbb R ^ { d } } \left( 1 + \lVert x \rVert \right) ^ { 2 } \rho _ { t } ^ { v , \varepsilon } ( \mathrm { d } x ) \mathrm { d } t } \\ & { \qquad \leq 2 \displaystyle \int _ { 0 } ^ { 1 } a _ { t } b _ { t } \int _ { \mathbb R ^ { d } } \left( 1 + \lVert x \rVert ^ { 2 } \right) \rho _ { t } ^ { v , \varepsilon } ( \mathrm { d } x ) \mathrm { d } t } \\ & { \qquad = 2 \displaystyle \int _ { 0 } ^ { 1 } a _ { t } b _ { t } \left( 1 + m _ { 2 } ( \rho _ { t } ^ { v , \varepsilon } ) \right) \mathrm { d } t . } \end{array}
$$

Hence, it remains to bound the second moment of $\rho _ { t } ^ { v , \varepsilon }$ . Note that by Lemma $4 { \cdot } 3$

$$
m _ { 2 } ( \rho _ { t } ^ { v , \varepsilon } ) \leq m _ { 2 } ( \rho _ { t } ^ { v } ) + 2 \sqrt { m _ { 2 } ( \rho _ { t } ^ { v } ) } W _ { 2 } ( \rho _ { t } ^ { v , \varepsilon } , \rho _ { t } ^ { v } ) + W _ { 2 } ^ { 2 } ( \rho _ { t } ^ { v , \varepsilon } , \rho _ { t } ^ { v } ) .
$$

Note that by Lemma $4 . 2$ we have $W _ { 2 } ^ { 2 } ( \rho _ { t } ^ { v , \varepsilon } , \rho _ { t } ^ { v } ) \leq c \varepsilon$ and by Lemma $4 { \cdot } 4$ the second moment $m _ { 2 } ( \rho _ { t } ^ { v } )$ can be bounded uniformly in $t \in [ 0 , 1 ]$ . Hence, we obtain the existence of a constant $C > 0$ such that

$$
\int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \lvert u _ { t } ( x ) \cdot w _ { t } ( x ) \rvert \rho _ { t } ^ { v , \varepsilon } ( \mathrm { d } x ) \mathrm { d } t \leq C \int _ { 0 } ^ { 1 } a _ { t } b _ { t } \mathrm { d } t < + \infty .
$$

## 4.2 Large deviation theory

We have shown that the advective Fisher–Rao metric can be obtained as the rescaled zero-noise limit of the Fisher–Rao metric on path measures. The concentration of the path measures in the zero-noise limit admits a large deviation principle with rate functional given by the Freidlin–Wentzell functional $[ \mathrm { F W 9 8 } ]$ . The following result shows that the advective Fisher–Rao metric can be obtained by evaluating the second variation of the Freidlin–Wentzell rate functional, which is given by

$$
I _ { v } ( { \boldsymbol { x } } _ { \bullet } ) = { \frac { 1 } { 2 } } \int _ { 0 } ^ { 1 } \| { \dot { \boldsymbol { x } } } _ { t } - { \boldsymbol { v } } _ { t } ( { \boldsymbol { x } } _ { t } ) \| ^ { 2 } \mathrm { d } t
$$

for a path $x _ { \bullet }$ and a fixed velocity field $v _ { \bullet }$ .

Theorem 4.5 (Connection to Freidlin–Wentzell functional). Consider a velocity field $v _ { \bullet } \in L _ { T } ^ { 2 } C _ { \ b } ^ { 2 }$ and $u _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 }$ and assume that $\rho _ { 0 } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ . Then it holds that

$$
g _ { v _ { \bullet } } ^ { \mathrm { A F R } } ( u _ { \bullet } , u _ { \bullet } ) = \int _ { \mathbb { R } ^ { d } } \delta ^ { 2 } I _ { v } ( \varphi _ { \bullet } ^ { v } ( x _ { 0 } ) ) \left[ \psi _ { \bullet } ^ { u } ( x _ { 0 } ) , \psi _ { \bullet } ^ { u } ( x _ { 0 } ) \right] \rho _ { 0 } ( \mathrm { d } x _ { 0 } ) ,\tag{4.1}
$$

where $\delta ^ { 2 } I _ { v }$ denotes the second variation of $I _ { v }$ and

$$
\psi _ { t } ^ { u } ( x _ { 0 } ) = \left. { \frac { \mathrm { d } } { \mathrm { d } h } } \right| _ { h = 0 } \varphi _ { t } ^ { v + h u } ( x _ { 0 } ) ,
$$

where $\varphi _ { t } ^ { v + h u } ( x _ { 0 } )$ denotes the flow generated by $v _ { \bullet } + h u _ { \bullet }$ starting at x<sub>0</sub>.

We begin by computing the first and second variation of the rate functional.

Lemma 4.6 (First and second variation). Consider a velocity field $v _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 } .$ Then the Freidlin–Wentzellfunctional $I _ { v } \colon H _ { T } ^ { 1 } ( \mathbb { R } ^ { d } ) \to \mathbb { R }$ is Gâteaux diferentiable and thefirst variation is given by

$$
\delta I _ { v } ( x _ { \bullet } ) [ y _ { \bullet } ] = \int _ { 0 } ^ { 1 } ( { \dot { x } } _ { t } - v _ { t } ( x _ { t } ) ) \cdot ( { \dot { y } } _ { t } - D v _ { t } ( x _ { t } ) y _ { t } ) \mathrm { d } t
$$

for all $x _ { \bullet } , y _ { \bullet } \in H _ { T } ^ { 1 } ( \mathbb { R } ^ { d } )$ . If further $v _ { \bullet } \in L _ { T } ^ { 2 } C _ { \ b } ^ { 2 }$ , then $I _ { v }$ is twice Gâteaux diferentiable and the second variation is given by

$$
\begin{array} { l } { { \displaystyle \delta ^ { 2 } I _ { v } ( x _ { \bullet } ) [ y _ { \bullet } , z _ { \bullet } ] = \int _ { 0 } ^ { 1 } ( \dot { y } _ { t } - D v _ { t } ( x _ { t } ) y _ { t } ) \cdot ( \dot { z } _ { t } - D v _ { t } ( x _ { t } ) z _ { t } ) \mathrm { d } t } } \\ { { \displaystyle \qquad - \int _ { 0 } ^ { 1 } ( \dot { x } _ { t } - v _ { t } ( x _ { t } ) ) \cdot D ^ { 2 } v _ { t } ( x _ { t } ) [ y _ { t } , z _ { t } ] \mathrm { d } t } } \end{array}
$$

for all $x _ { \bullet } , y _ { \bullet } , z _ { \bullet } \in H _ { T } ^ { 1 } ( \mathbb { R } ^ { d } ) .$

Proof. To verify the existence of the first variation, it sufices to show the existence of the diference quotients

$$
\frac { I _ { v } ( x _ { \bullet } + h y _ { \bullet } ) - I _ { v } ( x _ { \bullet } ) } { h } = \int _ { 0 } ^ { 1 } \Delta _ { t } ^ { h } \mathrm { d } t ,
$$

where

$$
\begin{array} { l } { \displaystyle \Delta _ { t } ^ { h } = \frac { \| \dot { x } _ { t } + h \dot { y } _ { t } - v _ { t } ( x _ { t } + h y _ { t } ) \| ^ { 2 } - \| \dot { x } _ { t } - v _ { t } ( x _ { t } ) \| ^ { 2 } } { 2 h } } \\ { \displaystyle \quad = \frac { h } { 2 } \| \dot { y } _ { t } \| ^ { 2 } + ( \dot { x } _ { t } - v _ { t } ( x _ { t } + h y _ { t } ) ) \cdot \dot { y } _ { t } } \\ { \displaystyle \qquad + \frac { 1 } { 2 h } \left( \| \dot { x } _ { t } - v _ { t } ( x _ { t } + h y _ { t } ) \| ^ { 2 } - \| \dot { x } _ { t } - v _ { t } ( x _ { t } ) \| ^ { 2 } \right) . } \end{array}
$$

Pointwise, the limit is given by

$$
\operatorname* { l i m } _ { h \to 0 } \Delta _ { t } ^ { h } = ( \dot { x } _ { t } - v _ { t } ( x _ { t } ) ) \cdot ( \dot { y } _ { t } - D v _ { t } ( x _ { t } ) y _ { t } ) .
$$

To construct a dominating function, we use that $H ^ { 1 }$ -paths are continuous, which in particular gives

$$
M _ { x } : = \operatorname* { s u p } _ { t \in [ 0 , 1 ] } \| x _ { t } \| , M _ { y } : = \operatorname* { s u p } _ { t \in [ 0 , 1 ] } \| y _ { t } \| < \infty
$$

and we denote the spatial Lipschitz constant of $v _ { t }$ by $\beta _ { t }$ . For $| h | \leq 1$ we can estimate

$$
| \Delta _ { t } ^ { h } | \leq \| \dot { x } _ { t } \| ^ { 2 } + 2 \| \dot { y } _ { t } \| ^ { 2 } + 2 \| v _ { t } ( 0 ) \| ^ { 2 } + 2 ( M _ { x } ^ { 2 } + M _ { y } ^ { 2 } ) \beta _ { t } ^ { 2 } ,
$$

which is integrable. Hence, by dominated convergence, we obtain

$$
\delta I _ { v } ( x _ { \bullet } ) [ y _ { \bullet } ] = \int _ { 0 } ^ { 1 } ( \dot { x } _ { t } - v _ { t } ( x _ { t } ) ) \cdot ( \dot { y } _ { t } - D v _ { t } ( x _ { t } ) y _ { t } ) \mathrm { d } t .
$$

For the existence of the second variation, we assume $v _ { \bullet } \in L _ { T } ^ { 2 } C _ { \ b } ^ { 2 }$ . We need to guarantee the existence of the limit of the diference quotients

$$
\frac { \delta I _ { v } ( x _ { \bullet } + h y _ { \bullet } ) [ z _ { \bullet } ] - \delta I _ { v } ( x _ { \bullet } ) [ z _ { \bullet } ] } { h } = \int _ { 0 } ^ { 1 } \Gamma _ { t } ^ { h } \mathrm { d } t ,
$$

where

$$
\begin{array} { r l } & { \Gamma _ { t } ^ { h } = \bigg ( \dot { y } _ { t } - \frac { v _ { t } ( x _ { t } + h y _ { t } ) - v _ { t } ( x _ { t } ) } { h } \bigg ) \cdot ( \dot { z } _ { t } - D v _ { t } ( x _ { t } + h y _ { t } ) z _ { t } ) } \\ & { \qquad - ( \dot { x } _ { t } - v _ { t } ( x _ { t } ) ) \cdot \bigg ( \frac { D v _ { t } ( x _ { t } + h y _ { t } ) - D v _ { t } ( x _ { t } ) } { h } \bigg ) z _ { t } . } \end{array}
$$

Pointwise, the limit is given by

$$
\operatorname* { l i m } _ { h  0 } \Gamma _ { t } ^ { h } = ( { \dot { y } } _ { t } - D v _ { t } ( x _ { t } ) y _ { t } ) \cdot ( { \dot { z } } _ { t } - D v _ { t } ( x _ { t } ) z _ { t } ) - ( { \dot { x } } _ { t } - v _ { t } ( x _ { t } ) ) \cdot D ^ { 2 } v _ { t } ( x _ { t } ) [ y _ { t } , z _ { t } ] .
$$

We set $\gamma _ { t } : = \| D ^ { 2 } v _ { t } \| _ { \infty }$ . By multiple applications of the mean value theorem applied to $v _ { t }$ and $D v _ { t }$ , together with Young’s inequality, each term in $\Gamma _ { t } ^ { h }$ can be bounded by quadratic expressions in ${ \dot { x } } _ { t } , { \dot { y } } _ { t } , { \dot { z } } _ { t } , v _ { t } ( 0 ) , \beta _ { t }$ , and $\gamma _ { t }$ . Hence there exists a constant $C = C ( M _ { x } , M _ { y } , M _ { z } ) > 0$ , independent of t and $h ,$ such that for $| h | \leq 1$

$$
| \Gamma _ { t } ^ { h } | \leq C \left( \| \dot { x } _ { t } \| ^ { 2 } + \| \dot { y } _ { t } \| ^ { 2 } + \| \dot { z } _ { t } \| ^ { 2 } + \| v _ { t } ( 0 ) \| ^ { 2 } + \beta _ { t } ^ { 2 } + \gamma _ { t } ^ { 2 } \right) .
$$

Since $\beta _ { \bullet } , \gamma _ { \bullet } \in L ^ { 2 } ( 0 , 1 )$ , the right-hand side is integrable, and hence by dominated convergence, we obtain

$$
\begin{array} { r } { \delta ^ { 2 } I _ { v } ( x _ { \bullet } ) [ y _ { \bullet } , z _ { \bullet } ] = \displaystyle \int _ { 0 } ^ { 1 } ( \dot { y } _ { t } - D v _ { t } ( x _ { t } ) y _ { t } ) \cdot ( \dot { z } _ { t } - D v _ { t } ( x _ { t } ) z _ { t } ) \mathrm { d } t } \\ { - \displaystyle \int _ { 0 } ^ { 1 } ( \dot { x } _ { t } - v _ { t } ( x _ { t } ) ) \cdot D ^ { 2 } v _ { t } ( x _ { t } ) [ y _ { t } , z _ { t } ] \mathrm { d } t . } \end{array}
$$

Proof of Theorem $4 { \cdot } 5$ . Recall that we want to show

$$
\int _ { \mathbb { R } ^ { d } } \delta ^ { 2 } I _ { v } ( x _ { \bullet } ) [ \psi _ { \bullet } ^ { u } ( x _ { 0 } ) , \psi _ { \bullet } ^ { u } ( x _ { 0 } ) ] \rho _ { 0 } ( { \mathrm { d } } x _ { 0 } ) = g _ { v _ { \bullet } } ^ { \mathrm { A F R } } ( u _ { \bullet } , u _ { \bullet } ) .
$$

First, note that we can apply Lemma $4 . 6$ as $v _ { \bullet } \in L _ { T } ^ { 2 } C _ { \ b } ^ { 2 }$ and $\psi _ { \bullet } ( x ) \in H _ { T } ^ { 1 } ( \mathbb { R } ^ { d } )$ . With $x _ { \bullet } = \varphi _ { \bullet } ^ { v } ( x _ { 0 } )$ , we have $\dot { x } _ { t } = v _ { t } ( x _ { t } )$ for almost every $t \in [ 0 , 1 ]$ and hence

$$
\delta ^ { 2 } I _ { v } ( x _ { \bullet } ) [ \psi _ { \bullet } ^ { u } ( x _ { 0 } ) , \psi _ { \bullet } ^ { u } ( x _ { 0 } ) ] = \int _ { 0 } ^ { 1 } \| \dot { y } _ { t } - D v _ { t } ( x _ { t } ) y _ { t } \| ^ { 2 } \mathrm { d } t .
$$

By Theorem $\mathrm { A } . 5$ we obtain for $y _ { t } = \psi _ { t } ^ { u } ( x _ { 0 } )$ that ${ \dot { y } } _ { t } - D v _ { t } ( x _ { t } ) y _ { t } = u _ { t } ( x _ { t } )$ . As $v _ { \bullet } \in L _ { T } ^ { 2 } C _ { \ b } ^ { 2 }$ this ensures $y _ { \bullet } \in H _ { T } ^ { 1 } ( \mathbb { R } ^ { d } )$ and hence we can use the expression of the second variation and obtain

$$
\delta ^ { 2 } I _ { v } ( x _ { \bullet } ) [ \psi _ { \bullet } ^ { u } ( x _ { 0 } ) , \psi _ { \bullet } ^ { u } ( x _ { 0 } ) ] = \int _ { 0 } ^ { 1 } \| u _ { t } ( x _ { t } ) \| ^ { 2 } \mathrm { d } t = \int _ { 0 } ^ { 1 } \| u _ { t } ( \varphi _ { t } ^ { v } ( x _ { 0 } ) ) \| ^ { 2 } \mathrm { d } t .
$$

Taking the expectation with respect to the initial condition $x _ { 0 }$ , we obtain

$$
\begin{array} { r l } { \displaystyle \int _ { \mathbb { R } ^ { d } } \delta ^ { 2 } I _ { v } ( x _ { \bullet } ) [ \psi _ { \bullet } ^ { u } ( x _ { 0 } ) , \psi _ { \bullet } ^ { u } ( x _ { 0 } ) ] \rho _ { 0 } ( \mathrm { d } x _ { 0 } ) = \int _ { \mathbb { R } ^ { d } } \int _ { 0 } ^ { 1 } \| u _ { t } ( \varphi _ { t } ^ { v } ( x _ { 0 } ) ) \| ^ { 2 } { \mathrm { d } } t \rho _ { 0 } ( \mathrm { d } x ) } & { } \\ { \displaystyle } & { = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| u _ { t } ( \varphi _ { t } ^ { v } ( x _ { 0 } ) ) \| ^ { 2 } \rho _ { 0 } ( \mathrm { d } x ) { \mathrm { d } } t } \\ { \displaystyle } & { = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| u _ { t } ( x ) \| ^ { 2 } \rho _ { t } ( \mathrm { d } x ) { \mathrm { d } } t } \\ { \displaystyle } & { = \varrho _ { v _ { \bullet } } ^ { \mathrm { A F R } } ( u _ { \bullet } , u _ { \bullet } ) . } \end{array}
$$

This shows that the advective Fisher–Rao metric is the expected value of the second variation of the Freidlin–Wentzell rate functional. Note, however, that the rate functional includes the base velocity field $v _ { \bullet }$ , which is not diferentiated in the second variation. In the introduction, we stated the theorem as

$$
g _ { v _ { \bullet } } ^ { \mathrm { A F R } } ( u _ { \bullet } , w _ { \bullet } ) = \int _ { \mathbb { R } ^ { d } } \frac { \mathrm { d } ^ { 2 } } { \mathrm { d } h ^ { 2 } } \bigg | _ { h = 0 } I _ { v } ( \varphi ^ { v + h u } ( x _ { 0 } ) ) \rho _ { 0 } ( \mathrm { d } x _ { 0 } ) .
$$

To see that this is equivalent to the statement of Theorem $4 { \cdot } 5 \cdot$ recall that $I _ { v }$ is minimized at $\varphi _ { \bullet } ^ { v } ( x _ { 0 } )$ with value $0 ,$ and hence we have $\delta I _ { v } ( \varphi _ { \bullet } ^ { v } ( x _ { 0 } ) ) = 0$ . Now we obtain

$$
\left. { \frac { \mathrm { d } ^ { 2 } } { \mathrm { d } h ^ { 2 } } } \right| _ { h = 0 } I _ { v } ( \varphi ^ { v + h u } ( x _ { 0 } ) ) = \delta ^ { 2 } I _ { v } ( \varphi _ { \bullet } ^ { v } ( x _ { 0 } ) ) [ \psi _ { \bullet } ^ { u } ( x _ { 0 } ) , \psi _ { \bullet } ^ { u } ( x _ { 0 } ) ] .
$$

## 4.3 Optimal transport

The advective Fisher–Rao metric operates on velocity fields and is tightly connected to the continuity equation, which describes the transport of mass along a flow. As such, it is a natural question how it is connected to the geometry of optimal transport. In this subsection, we relate both the least-squares objective on velocities and the advective Fisher–Rao metric to the action functional appearing in the dynamic formulation of optimal transport. Recall the Benamou–Brenier formulation of the Wasserstein distance

$$
W _ { 2 } ^ { 2 } ( \mu , \nu ) = 2 \operatorname * { i n f } _ { p _ { \bullet } , j _ { \bullet } } \left\{ \mathcal { A } ( p _ { \bullet } , j _ { \bullet } ) : p _ { 0 } = \mu , p _ { 1 } = \nu , \partial _ { t } p _ { t } + \nabla \cdot j _ { t } = 0 \right\} ,
$$

where the action functional in momentum or current form is given by

$$
\mathcal { A } ( p _ { \bullet } , j _ { \bullet } ) = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \frac { \| j _ { t } ( x ) \| _ { 2 } ^ { 2 } } { p _ { t } ( x ) } \mathrm { d } x \mathrm { d } t .
$$

In the context of the Benamou–Brenier formulation, the current is given by ${ j _ { t } } = v _ { t } p _ { t }$ It is known that the action functional is convex, and so is its domain

$$
\mathrm { d o m } ( \mathcal { A } ) = \left\{ ( p _ { \bullet } , j _ { \bullet } ) : p _ { \bullet } \in \mathrm { A C } _ { T } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) ) , j _ { \bullet } \mathrm { m e a s u r a b l e } , \mathcal { A } ( p _ { \bullet } , j _ { \bullet } ) < + \infty \right\} .
$$

Theorem 4.7. Assume that the initial distribution $\rho _ { 0 } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ has a positive density p with respect to the Lebesgue measure. For any velocityfield $v _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 }$ and $u _ { \bullet } \in C _ { \mathrm { c } } ^ { 1 } ( [ 0 , 1 ] \times \mathbb { R } ^ { d } ; \mathbb { R } ^ { d } )$ , it holds that

$$
g _ { v _ { \bullet } } ^ { \mathrm { A F R } } ( u _ { \bullet } , u _ { \bullet } ) = 2 \operatorname* { l i m } _ { h  0 } \frac { \mathcal { A } ( p _ { \bullet } ^ { h } , j _ { \bullet } ^ { h } ) - \mathcal { A } ( p _ { \bullet } , j _ { \bullet } ) - \delta ^ { + } \mathcal { A } ( p _ { \bullet } , j _ { \bullet } ) [ p _ { \bullet } ^ { h } - p _ { \bullet } , j _ { \bullet } ^ { h } - j _ { \bullet } ] } { h ^ { 2 } } ,\tag{4.2}
$$

where $p _ { \bullet } = p _ { \bullet } ^ { v } , p _ { \bullet } ^ { h } = p _ { \bullet } ^ { v + h u } , j _ { \bullet } = v _ { \bullet } p _ { \bullet } ^ { v }$ , and $j _ { \bullet } ^ { h } = ( v _ { \bullet } + h u _ { \bullet } ) p _ { \bullet } ^ { v + h u }$ denote densities of the solution of the continuity equation and the corresponding currents.

In the proof, we will use the following lemma, which is of independent interest as it states that the least-squares objective on velocity fields is the Bregman divergence induced by the action functional.

Lemma 4.8. Consider $( p _ { \bullet } , j _ { \bullet } ) , ( p _ { \bullet } ^ { \prime } , j _ { \bullet } ^ { \prime } ) \in$ dom( ) and assume that $p _ { \bullet } > 0$ and $p _ { \bullet } ^ { \prime } > 0 _ { \ast }$ , and that

$$
\int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| v _ { t } ( x ) \| _ { 2 } ^ { 2 } p _ { t } ^ { \prime } ( x ) \mathrm { d } x \mathrm { d } t \quad a n d \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| v _ { t } ^ { \prime } ( x ) \| _ { 2 } ^ { 2 } p _ { t } ( x ) \mathrm { d } x \mathrm { d } t < + \infty ,\tag{4.3}
$$

where $\begin{array} { r } { v _ { \bullet } = \frac { j _ { \bullet } } { p _ { \bullet } } } \end{array}$ and $\begin{array} { r } { v _ { \bullet } ^ { \prime } = \frac { j _ { \bullet } ^ { \prime } } { p _ { \bullet } ^ { \prime } } } \end{array}$ . Then, the right-sidedfirst variation $\delta ^ { + } \mathcal { A } ( p _ { \bullet } , j _ { \bullet } ) [ p _ { \bullet } ^ { \prime } -$ $p _ { \bullet } , j _ { \bullet } ^ { \prime } - j _ { \bullet } ]$ exists and is given by

$$
\delta ^ { + } \mathscr { A } ( p _ { \bullet } , j _ { \bullet } ) [ p _ { \bullet } ^ { \prime } - p _ { \bullet } , j _ { \bullet } ^ { \prime } - j _ { \bullet } ] = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \left( v _ { t } ^ { \prime } \cdot v _ { t } p _ { t } ^ { \prime } - \| v _ { t } \| _ { 2 } ^ { 2 } ( p _ { t } + p _ { t } ^ { \prime } ) \right) \mathrm { d } x \mathrm { d } t ,
$$

Further, the Bregman divergence induced by the actionfunctional is given by

$$
D _ { A } ( ( p _ { \bullet } , j _ { \bullet } ) , ( p _ { \bullet } ^ { \prime } , j _ { \bullet } ^ { \prime } ) ) = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \lVert v _ { t } ( x ) - v _ { t } ^ { \prime } ( x ) \rVert _ { 2 } ^ { 2 } p _ { t } ( x ) \mathrm { d } x \mathrm { d } t .
$$

Proof. First, we rewrite the action functional as

$$
\mathcal { A } ( p _ { \bullet } , j _ { \bullet } ) = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } f ( p _ { t } ( x ) , j _ { t } ( x ) ) \mathrm { d } x \mathrm { d } t ,
$$

where

$$
f \colon \mathbb { R } _ { > 0 } \times \mathbb { R } ^ { d } \to \mathbb { R } , \quad f ( p , j ) = \frac { \| j \| _ { 2 } ^ { 2 } } { 2 p }
$$

is a convex function. Then, we need to establish the existence of the limit of the diference quotients

$$
\frac { \mathcal { A } ( p _ { \bullet } + h ( p _ { \bullet } ^ { \prime } - p _ { \bullet } ) , j _ { \bullet } + h ( j _ { \bullet } ^ { \prime } - j _ { \bullet } ) ) - \mathcal { A } ( p _ { \bullet } , j _ { \bullet } ) } { h } = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \Delta _ { t } ^ { h } ( x ) \mathrm { d } x \mathrm { d } t ,
$$

where

$$
\Delta _ { t } ^ { h } = \frac { f ( p _ { t } + h ( p _ { t } ^ { \prime } - p _ { t } ) , j _ { t } + h ( j _ { t } ^ { \prime } - j _ { t } ) ) - f ( p _ { t } , j _ { t } ) } { h } .
$$

To compute the pointwise limit of $\Delta _ { t } ( x )$ we use that for $p > 0$ the first derivative of f is given by

$$
\nabla f ( p , j ) = \left( \begin{array} { c } { - \frac { \| j \| _ { 2 } ^ { 2 } } { 2 p ^ { 2 } } } \\ { \frac { j } { p } } \end{array} \right) = \left( \begin{array} { c } { - \frac { \| v \| _ { 2 } ^ { 2 } } { 2 } } \\ { v } \end{array} \right) ,
$$

where $\begin{array} { r } { v = { \frac { j } { p } } } \end{array}$ . Now the chain rule implies

$$
\begin{array} { l } { \displaystyle \operatorname* { l i m } _ { h \to 0 } \Delta _ { t } ^ { h } ( x ) = - ( p _ { t } ^ { \prime } ( x ) - p _ { t } ( x ) ) \cdot \frac { \| v _ { t } ( x ) \| ^ { 2 } } { 2 } + v _ { t } ( x ) \cdot ( p _ { t } ^ { \prime } ( x ) v _ { t } ^ { \prime } ( x ) - p _ { t } ( x ) v _ { t } ( x ) ) , } \\ { \displaystyle \qquad = - ( p _ { t } ^ { \prime } ( x ) + p _ { t } ( x ) ) \cdot \frac { \| v _ { t } ( x ) \| ^ { 2 } } { 2 } + v _ { t } ( x ) \cdot v _ { t } ^ { \prime } ( x ) p _ { t } ^ { \prime } ( x ) , } \\ { \displaystyle \qquad = : \Delta _ { t } ^ { 0 } ( x ) , } \end{array}
$$

which is an integrable function over time and space by assumption $( 4 . 3 )$ . To use the dominated convergence theorem, we first note that by the convexity of $f ,$ the diference quotient $\Delta ^ { \bar { h } }$ is pointwise monotonically decreasing and hence

$$
\Delta _ { t } ^ { h } ( x ) \geq \Delta _ { t } ^ { 0 } ( x ) \quad \mathrm { f o r ~ a l l ~ } t \in [ 0 , 1 ] , x \in \mathbb { R } ^ { d } .
$$

Further, for all $t \in [ 0 , 1 ] , x \in \mathbb { R } ^ { d }$ we have

$$
\Delta _ { t } ^ { h } ( x ) \le \Delta _ { t } ^ { 1 } ( x ) = f ( p _ { t } ^ { \prime } ( x ) , j _ { t } ^ { \prime } ( x ) ) - f ( p _ { t } ( x ) , j _ { t } ( x ) ) ,
$$

which is integrable over $[ 0 , 1 ] \times  { \mathbb { R } } ^ { d }$ as $( p _ { \bullet } , j _ { \bullet } ) , ( p _ { \bullet } ^ { \prime } , j _ { \bullet } ^ { \prime } ) \in \mathrm { d o m } ( { \cal A } )$ . As $\Delta ^ { h }$ is pointwise bounded from above and below by integrable functions , we can apply the dominated convergence theorem and obtain

$$
\delta ^ { + } \mathcal { A } ( p _ { \bullet } , j _ { \bullet } ) [ p _ { \bullet } ^ { \prime } - p _ { \bullet } , j _ { \bullet } ^ { \prime } - j _ { \bullet } ] = \operatorname* { l i m } _ { h \searrow 0 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \Delta _ { t } ^ { h } ( x ) \mathrm { d } x \mathrm { d } t
$$

$$
\begin{array} { r l } & { = \displaystyle \int _ { 0 } ^ { 1 } \int _ { \mathbb R ^ { d } } \operatorname* { l i m } _ { h \searrow 0 } \Delta _ { t } ^ { h } ( x ) \mathrm { d } x \mathrm { d } t } \\ & { = \displaystyle \int _ { 0 } ^ { 1 } \int _ { \mathbb R ^ { d } } v ^ { \prime } \cdot v p ^ { \prime } \mathrm { d } x \mathrm { d } t - \frac 1 2 \int _ { 0 } ^ { 1 } \int _ { \mathbb R ^ { d } } \lVert v \rVert _ { 2 } ^ { 2 } ( p + p ^ { \prime } ) \mathrm { d } x \mathrm { d } t . } \end{array}
$$

Finally, we compute the Bregman divergence

$$
\begin{array} { l } { { \displaystyle { \cal D } _ { \mathcal { A } } ( ( p _ { \bullet } , j _ { \bullet } ) , ( p _ { \bullet } ^ { \prime } , j _ { \bullet } ^ { \prime } ) ) = \mathcal { A } ( p _ { \bullet } , j _ { \bullet } ) - \mathcal { A } ( p _ { \bullet } ^ { \prime } , j _ { \bullet } ^ { \prime } ) - \delta ^ { + } \mathcal { A } ( p _ { \bullet } ^ { \prime } , j _ { \bullet } ^ { \prime } ) [ p _ { \bullet } - p _ { \bullet } ^ { \prime } , j _ { \bullet } - j _ { \bullet } ^ { \prime } ] } } \\ { { \displaystyle \qquad = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \lVert v \rVert _ { 2 ^ { p } } ^ { 2 } \mathrm { d } x \mathrm { d } t - \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \lVert v ^ { \prime } \rVert _ { 2 ^ { p ^ { \prime } } } ^ { 2 } \mathrm { d } x ^ { \prime } \mathrm { d } t } } \\ { { \displaystyle \qquad - \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } v \cdot v ^ { \prime } p \mathrm { d } x \mathrm { d } t + \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \lVert v ^ { \prime } \rVert _ { 2 } ^ { 2 } ( p ^ { \prime } + p ) \mathrm { d } x \mathrm { d } t } } \\ { { \displaystyle \qquad = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \lVert v - v ^ { \prime } \rVert _ { 2 ^ { p } } ^ { 2 } \mathrm { d } x \mathrm { d } t } . } \end{array}
$$

Remark 4.9 (Gâteaux diferentiability of the action). With the notation $( \xi _ { \bullet } , n _ { \bullet } ) =$ $( p _ { \bullet } ^ { \prime } , j _ { \bullet } ^ { \prime } ) - ( p _ { \bullet } , j _ { \bullet } )$ , the expression of the right-sided first variation in Lemma 4.8 becomes

$$
\delta ^ { + } \mathcal { A } ( p _ { \bullet } , j _ { \bullet } ) [ \xi _ { \bullet } , n _ { \bullet } ] = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \left( v _ { t } ( x ) \cdot n _ { t } ( x ) - \frac 1 2 \| v _ { t } ( x ) \| _ { 2 } ^ { 2 } \xi _ { t } ( x ) \right) \mathrm { d } x \mathrm { d } t .
$$

As this expression is linear in $( \xi _ { \bullet } , n _ { \bullet } )$ , this implies that if $( p _ { \bullet } , j _ { \bullet } ) - h ( \xi _ { \bullet } , n _ { \bullet } ) \in$ dom( ) for $h > 0$ small, then also the left-sided diference quotients converge to the same limit. Consequently, this shows the Gâteaux diferentiability in the case that the two-sided diference quotients can be formed.

Further, we require a bound on the Wasserstein-2 distance in terms of the least-squares distance of the velocity fields. An analogous bound under stronger regularity conditions on the velocity fields can be found in [BDD24].

Lemma 4.10. Consider $v _ { \bullet } , w _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 }$ and assume that the initial condition $\rho _ { 0 }$ hasfinite second moments. Thenfor all $\bar { t } \in [ 0$ , 1] it holds that

$$
\begin{array} { r } { W _ { 2 } ( \rho _ { t } ^ { v } , \rho _ { t } ^ { w } ) \leq e ^ { \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \operatorname { L i p } } ^ { 1 } } } \| v _ { \bullet } - w _ { \bullet } \| _ { L _ { T } ^ { 1 } L ^ { 2 } ( \rho _ { \bullet } ^ { w } ) } \leq e ^ { \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \operatorname { L i p } } ^ { 1 } } } \| v _ { \bullet } - w _ { \bullet } \| _ { L _ { T } ^ { 2 } L ^ { 2 } ( \rho _ { \bullet } ^ { w } ) } . } \end{array}
$$

Proof. Let $\varphi ^ { v }$ and $\varphi ^ { w }$ be the flow maps associated with the vector fields $v _ { \bullet }$ and $w _ { \bullet }$ By the definition of the Wasserstein distance, we can bound $W _ { 2 }$ using this specific coupling

$$
W _ { 2 } ( \rho _ { t } ^ { v } , \rho _ { t } ^ { w } ) \leq \| \varphi _ { t } ^ { v } - \varphi _ { t } ^ { w } \| _ { L ^ { 2 } ( \rho _ { 0 } ) } .
$$

We can express the diference between the two flow maps at time t via

$$
\varphi _ { t } ^ { v } ( x ) - \varphi _ { t } ^ { w } ( x ) = \int _ { 0 } ^ { t } \left( v _ { s } ( \varphi _ { s } ^ { v } ( x ) ) - w _ { s } ( \varphi _ { s } ^ { w } ( x ) ) \right) \mathrm { d } s .
$$

Taking the $L ^ { 2 } ( p _ { 0 } )$ norm of both sides and applying the triangle inequality, we obtain

$$
\begin{array} { r l } { \displaystyle \| { \varphi } _ { t } ^ { v } - { \varphi } _ { t } ^ { w } \| _ { L ^ { 2 } ( { \rho } _ { 0 } ) } \leq \int _ { 0 } ^ { t } \| v _ { s } \circ { \varphi } _ { s } ^ { v } - { w } _ { s } \circ { \varphi } _ { s } ^ { w } \| _ { L ^ { 2 } ( { \rho } _ { 0 } ) } \mathrm { d } s . } & { } \\ { \displaystyle \leq \int _ { 0 } ^ { t } \| v _ { s } \circ { \varphi } _ { s } ^ { v } - { v } _ { s } \circ { \varphi } _ { s } ^ { w } \| _ { L ^ { 2 } ( { \rho } _ { 0 } ) } \mathrm { d } s } & { } \\ { \displaystyle } & { ~ + \int _ { 0 } ^ { t } \| v _ { s } \circ { \varphi } _ { s } ^ { w } - { w } _ { s } \circ { \varphi } _ { s } ^ { w } \| _ { L ^ { 2 } ( { \rho } _ { 0 } ) } \mathrm { d } s } \\ { \displaystyle } & { \leq \int _ { 0 } ^ { t } \beta _ { s } \| { \varphi } _ { s } ^ { v } - { \varphi } _ { s } ^ { w } \| _ { L ^ { 2 } ( { \rho } _ { 0 } ) } \mathrm { d } s + \int _ { 0 } ^ { t } \| v _ { s } - { w } _ { s } \| _ { L ^ { 2 } ( { p } _ { s } ^ { w } ) } \mathrm { d } s , } \end{array}
$$

where $\beta _ { s }$ denotes the spatial Lipschitz constant of $v _ { s }$ . Applying Grönwall’s inequality yields

$$
\begin{array} { r l r } {  { \| \varphi _ { t } ^ { v } - \varphi _ { t } ^ { w } \| _ { L ^ { 2 } ( \rho _ { 0 } ) } \leq e ^ { \int _ { 0 } ^ { t } \beta _ { r } \mathrm { d } r } \int _ { 0 } ^ { t } \| v _ { s } - w _ { s } \| _ { L ^ { 2 } ( p _ { s } ^ { w } ) } \mathrm { d } s } } \\ & { } & \\ & { } & { \leq e ^ { \int _ { 0 } ^ { t } \beta _ { r } \mathrm { d } r } \sqrt { t } ( \int _ { 0 } ^ { t } \| v _ { s } - w _ { s } \| _ { L ^ { 2 } ( p _ { s } ^ { w } ) } ^ { 2 } \mathrm { d } s ) ^ { \frac { 1 } { 2 } } } \\ & { } & \\ & { } & { \leq e ^ { \int _ { 0 } ^ { t } \beta _ { r } \mathrm { d } r } \sqrt { t } \| v _ { \bullet } - w _ { \bullet } \| _ { L _ { T } ^ { 2 } L ^ { 2 } ( p _ { \bullet } ^ { w } ) } . } \end{array}
$$

ProofofTheorem $4 { \cdot } 7 \cdot$ . First, we notice that for $v _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 } ,$ , we have $( p _ { \bullet } ^ { v } , v _ { \bullet } p _ { \bullet } ^ { v } ) \in$ dom( ). Indeed, $v _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 }$ implies that $p _ { \bullet } ^ { v }$ has uniformly in time bounded second moments, see, for example, Lemma $4 { \cdot } 4$ , and hence

$$
\mathcal { A } ( p _ { \bullet } ^ { v } , v _ { \bullet } p _ { \bullet } ^ { v } ) = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \lVert v _ { t } ( x ) \rVert _ { 2 } ^ { 2 } p _ { t } ^ { v } ( x ) \mathrm { d } x \mathrm { d } t < + \infty .
$$

Consider now $v _ { \bullet } ^ { h } = v _ { \bullet } + h u _ { \bullet } , p _ { \bullet } ^ { h } = p _ { \bullet } ^ { v + h u }$ and $j _ { \bullet } ^ { h } = j _ { \bullet } ^ { v + h u }$ , then we obtain

$$
\begin{array} { l } { \displaystyle \frac { \mathcal { A } ( p _ { \bullet } ^ { h } , j _ { \bullet } ^ { h } ) - \mathcal { A } ( p _ { \bullet } , j _ { \bullet } ) - \delta ^ { + } \mathcal { A } ( p _ { \bullet } , j _ { \bullet } ) [ \xi _ { \bullet } ^ { h } , n _ { \bullet } ^ { h } ] } { h ^ { 2 } } = h ^ { - 2 } D _ { \mathcal { A } } ( ( p _ { \bullet } ^ { h } , j _ { \bullet } ^ { h } ) , ( p _ { \bullet } , j _ { \bullet } ) ) } \\ { \displaystyle \qquad = \frac { 1 } { 2 h ^ { 2 } } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| v _ { t } ^ { h } ( x ) - v _ { t } ( x ) \| _ { 2 } ^ { 2 } p _ { t } ^ { h } ( x ) \mathrm { d } x \mathrm { d } t } \\ { \displaystyle \qquad = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| u _ { t } ( x ) \| _ { 2 } ^ { 2 } p _ { t } ^ { h } ( x ) \mathrm { d } x \mathrm { d } t . } \end{array}
$$

Note that $p _ { t } ^ { h } \to p _ { t }$ as $h  0$ in $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ at a rate of $O ( h )$ uniformly in time, see for example Lemma 4.10. Together with the linear growth of $u _ { t }$ , this yields that

$$
\operatorname* { l i m } _ { h \to 0 } \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| u _ { t } ( x ) \| _ { 2 } ^ { 2 } p _ { t } ^ { h } ( x ) \mathrm { d } x \mathrm { d } t = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| u _ { t } ( x ) \| _ { 2 } ^ { 2 } p _ { t } ( x ) \mathrm { d } x \mathrm { d } t .
$$

## 5 The Advective Fisher–Rao Metric on Paths of Probability Measures

So far, we have defined the advective Fisher–Rao metric on the space of velocity fields. However, it is the goal to solve the control problem (1.1) on the paths of probability densities. Note that there are multiple velocity fields that can generate the same path of probability densities via the continuity equation. Hence, it is not directly possible to pushforward the geometry onto the space of paths of probability densities. In this section, we will define a version of the advective Fisher–Rao metric on the space of paths of probability densities that is compatible with the advective Fisher–Rao metric on the space of gradient velocity fields. Additionally, we show that it preserves the energy compatibility property and that it arises as the Hessian metric of the contracted Benamou–Brenier action functional. We will construct a Riemannian metric on the space

$$
\begin{array} { r } { \mathsf { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) ) = \left\{ \rho _ { \bullet } \in \mathrm { A C } _ { T } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) ) : \int _ { 0 } ^ { 1 } | \dot { \rho } _ { t } | _ { W _ { 2 } } ^ { 2 } \mathrm { d } t < + \infty \right\} , } \end{array}
$$

where $\begin{array} { r } { \vert \dot { \rho } _ { t } \vert _ { W _ { 2 } } = \operatorname* { l i m } _ { h \to 0 } W _ { 2 } ( \rho _ { t + h } , \rho _ { t } ) h ^ { - 1 } } \end{array}$ is the metric derivative, see [AGS05]. To construct a tangent space for this space of paths, recall the definition of the Wasserstein–Otto tangent space

$$
T _ { \rho } \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) : = \{ - \nabla \cdot ( \rho v ) : v \in L ^ { 2 } ( \rho ) \} \subseteq \mathcal { D } ^ { \prime } ( \mathbb { R } ^ { d } ) .
$$

which dates back to [Ott01]. Note that $v \mapsto - \nabla \cdot ( \rho v )$ is not injective. To remove this redundancy, it is common to consider

$$
\begin{array} { r } { \mathcal { T } _ { \rho } \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) : = \overline { { \left\{ \nabla \phi : \phi \in C _ { \mathrm { c } } ^ { \infty } ( \mathbb { R } ^ { d } ) \right\} } } ^ { L ^ { 2 } ( \rho ) } } \end{array}
$$

and to identify the spaces $\mathcal { T } _ { \rho } \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) \cong T _ { \rho } \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ according to $v \mapsto - \nabla \cdot ( \rho v )$ and to refer to $\mathcal { T } _ { \rho } \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ as the tangent space. However, for us, it is convenient to keep the distinction between those two spaces. The Wasserstein–Otto metric is given by

$$
g _ { \rho } ^ { \mathrm { w o } } ( \xi , \zeta ) = \int _ { \mathbb { R } ^ { d } } \boldsymbol { u } \cdot \boldsymbol { w } \rho ( \mathrm { d } \boldsymbol { x } ) ,
$$

where $\xi = - \nabla \cdot ( \rho v )$ and $\zeta = - \nabla \cdot ( \rho w )$

We define the time-dependent version of the tangent spaces by

$$
T _ { \rho _ { \bullet } } L _ { T } ^ { 2 } ( { \mathcal { P } } _ { 2 } ( \mathbb { R } ^ { d } ) ) : = \left\{ \xi _ { \bullet } : \xi _ { t } \in T _ { \rho _ { t } } { \mathcal { P } } _ { 2 } ( \mathbb { R } ^ { d } ) \mathrm { f o r } \mathrm { e v e r y } t \in [ 0 , 1 ] \right\} ,
$$

and similarly

$$
\begin{array} { r } { \mathcal { T } _ { \rho _ { \bullet } } L _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) ) : = \left\{ v _ { \bullet } \in L _ { T } ^ { 2 } L ^ { 2 } ( \rho _ { \bullet } ) : v _ { t } \in \mathcal { T } _ { \rho _ { t } } \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) \mathrm { ~ f o r ~ e v e r y ~ } t \in [ 0 , 1 ] \right\} } \end{array}
$$

The corresponding time-dependent metric is given by

$$
g _ { \rho \bullet } ^ { \mathrm { W O } } ( \xi _ { \bullet } , \xi _ { \bullet } ) = \int _ { 0 } ^ { 1 } g _ { \rho t } ^ { \mathrm { W O } } ( \xi _ { t } , \xi _ { t } ) \mathrm { d } t .
$$

For a curve $\rho _ { \bullet } \in \mathsf { A C } _ { T } ^ { 2 } \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ , it is well known that one can find a velocity field $v _ { \bullet } ^ { \rho } \in \mathcal { T } _ { \rho } . L _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ such that

$$
\begin{array} { r } { \partial _ { t } \rho _ { t } + \nabla \cdot ( \rho _ { t } v _ { t } ^ { \rho } ) = 0 , } \end{array}
$$

see [AGS05], which we refer to as the velocity of $\rho _ { \bullet }$ . Now, we can proceed with our construction of a metric on the space $\mathbf { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ , which respects the advective nature of the continuity equation.

Definition 5.1 (Advective Fisher–Rao metric on paths of measures). Consider a path $\rho _ { \bullet } \in \mathsf { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ and its velocity $v _ { \bullet } \in T _ { \rho _ { \bullet } } L _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ . Then we define the tangent space of $\mathbf { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ as

$$
\begin{array} { r } { T _ { \rho _ { \bullet } } \mathrm { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) ) : = \left\{ \xi _ { \bullet } \in T _ { \rho _ { \bullet } } L _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) ) : \begin{array} { l } { \xi _ { t } v _ { t } \in \mathcal { D } ^ { \prime } ( \mathbb { R } ^ { d } ) \mathrm { ~ f o r ~ a l l ~ } t \in [ 0 , 1 ] } \\ { \qquad D _ { t } \xi _ { \bullet } \in T _ { \rho _ { \bullet } } L _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) ) } \end{array} \right\} , } \end{array}
$$

where we refer to

$$
D _ { t } \xi _ { t } : = \partial _ { t } \xi _ { t } + \nabla \cdot ( \xi _ { t } v _ { t } )\tag{5.1}
$$

as the transport derivative of $\xi _ { \bullet }$ along $\rho _ { \bullet }$ , which is to be understood in the distributional sense. For two tangent vectors $\xi _ { \bullet } , \zeta _ { \bullet } \in T _ { \rho _ { \bullet } } \mathrm { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ , we refer to

$$
g _ { \rho \bullet } ^ { \mathrm { A F R } } ( \xi _ { \bullet } , \xi _ { \bullet } ) = \int _ { 0 } ^ { 1 } g _ { \rho _ { t } } ^ { \mathrm { W O } } ( D _ { t } \xi _ { t } , D _ { t } \xi _ { t } ) \mathrm { d } t
$$

as the advective Fisher–Rao metric.

It can be shown that the transport derivative $D _ { t }$ is the covariant derivative with respect to a flat connection, which is not the Levi-Civita connection with respect to the Wasserstein–Otto metric [Ay25].

Remark 5.2 (Explicit expression of the advective Fisher–Rao metric). For tangent vectors $\xi _ { \bullet } , \zeta _ { \bullet } \in T _ { \rho _ { \bullet } } \mathrm { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ , the advective Fisher–Rao metric is given by

$$
g _ { \rho _ { \bullet } } ^ { \mathrm { A F R } } ( \xi _ { \bullet } , \zeta _ { \bullet } ) : = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } u _ { t } ^ { \xi } ( x ) \cdot w _ { t } ^ { \zeta } ( x ) \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t ,
$$

where

$$
\begin{array} { r l } & { \partial _ { t } \xi _ { t } + \nabla \cdot ( \xi _ { t } v _ { t } ) = - \nabla \cdot ( \rho _ { t } u _ { t } ^ { \xi } ) \quad \mathrm { a n d } } \\ & { \partial _ { t } \zeta _ { t } + \nabla \cdot ( \zeta _ { t } v _ { t } ) = - \nabla \cdot ( \rho _ { t } w _ { t } ^ { \zeta } ) . } \end{array}\tag{5.2}
$$

The following result shows the compatibility of the Fisher–Rao metric on the space of paths of densities with the Fisher–Rao metric on the space of gradient velocity fields. In particular, it shows that the solution operator of the continuity equation is a Riemannian isometry between the two spaces. Further, it gives a class of non-trivial examples in which the advective Fisher–Rao metric on the space of paths of densities is well-defined.

Lemma 5.3 (Compatibility). Consider gradient velocity fields

$$
v _ { \bullet } = \nabla \phi _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 } , \quad u _ { \bullet } = \nabla \psi _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 } ( \mathbb { R } ^ { d } ) , \quad w _ { \bullet } = \nabla \psi _ { \bullet } ^ { \prime } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 } ( \mathbb { R } ^ { d } )
$$

and denote the solutions of the continuity equation with velocity fields $v _ { \bullet } + h u _ { \bullet }$ and $v _ { \bullet } + h w _ { \bullet }$ by $\rho _ { \bullet } ^ { v + h u } , \rho _ { \bullet } ^ { v + h w } \in \mathsf { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ , respectively, and assume that $\rho _ { 0 } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ . Then

$$
 \xi \bullet : =  \frac { \mathrm { d } } { \mathrm { d } h } | _ { h = 0 } \rho _ { \bullet } ^ { v + h u } \quad a n d  \zeta \bullet : =  \frac { \mathrm { d } } { \mathrm { d } h } | _ { h = 0 } \rho _ { \bullet } ^ { v + h w }
$$

are elements in $T _ { \rho \bullet } \mathsf { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ and it holds that

$$
g _ { \rho _ { \bullet } ^ { v } } ^ { \mathrm { A F R } } ( \xi _ { \bullet } , \zeta _ { \bullet } ) = g _ { v _ { \bullet } } ^ { \mathrm { A F R } } ( u _ { \bullet } , w _ { \bullet } ) .\tag{5.3}
$$

Proof. By Lemma $3 . 8 , \xi _ { t }$ and $\zeta _ { t }$ exist as weak- limits in $( C _ { \mathrm { L i p } } ^ { 1 } ) ^ { * }$ for every $t \in [ 0 , 1 ]$ and satisfy

$$
\begin{array} { r l } & { \partial _ { t } \xi _ { t } + \nabla \cdot ( \xi _ { t } ^ { u } \nabla \phi _ { t } ) = - \nabla \cdot ( \rho _ { t } \boldsymbol { u } _ { t } ) \quad \mathrm { a n d } } \\ & { \partial _ { t } \zeta _ { t } + \nabla \cdot ( \zeta _ { t } ^ { w } \nabla \phi _ { t } ) = - \nabla \cdot ( \rho _ { t } w _ { t } ) . } \end{array}\tag{5.4}
$$

Further, as the second moments of $\rho _ { \bullet } ^ { v }$ are uniformly bounded in time by Lemma 4.4 and $u _ { \bullet } , w _ { \bullet } \in \ L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 } ( \mathbb { R } ^ { d } )$ , we obtain $u _ { \bullet } , w _ { \bullet } \in L _ { T } ^ { 2 } L ^ { 2 } ( \rho _ { \bullet } ^ { v } )$ . To show that $\xi _ { \bullet } \in \mathcal { T } _ { \rho _ { \bullet } } \mathbf { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ , it remains to verify $\xi _ { \bullet } \in T _ { \rho _ { \bullet } } L _ { T } ^ { 2 } \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ . For a fixed time $t \in [ 0 , 1 ]$ , Lemma 3.8 gives

$$
\left. \xi _ { t } , \phi \right. = \int _ { \mathbb { R } ^ { d } } \nabla \phi ( x ) \cdot \psi _ { t } ^ { u } ( ( \varphi _ { t } ^ { v } ) ^ { - 1 } ( x ) ) \rho _ { t } ( \mathrm { d } x )
$$

for all $\phi \in C _ { \mathrm { L i p } } ^ { 1 } ( \mathbb { R } ^ { d } )$ , where for any $x \in \mathbb { R } ^ { d } \psi _ { \bullet } ^ { u } ( x )$ is the unique solution to

$$
\partial _ { t } \psi _ { t } ^ { u } ( x ) = D v _ { t } ( \varphi _ { t } ^ { v } ( x ) ) \psi _ { t } ^ { u } ( x ) + u _ { t } ( \varphi _ { t } ^ { v } ( x ) ) , \quad \psi _ { 0 } ( x ) = 0 .
$$

Hence, in distribution it holds that

$$
\xi _ { t } = - \nabla \cdot ( \rho _ { t } ^ { v } v _ { t } ^ { \xi } ) ,
$$

where $v _ { t } ^ { \xi } ( x ) = \psi _ { t } ^ { u } ( ( \varphi _ { t } ^ { v } ) ^ { - 1 } ( x ) )$ . To show $\xi _ { t } \in T _ { \rho _ { t } } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ , we first show $v _ { \bullet } ^ { \xi } \in$ $L _ { T } ^ { 2 } L ^ { 2 } ( \rho _ { \bullet } ^ { v } )$ . From Lemma A.2 we obtain

$$
\begin{array} { r l r } {  { \| \psi _ { t } ( x ) \| \leq e ^ { \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } \int _ { 0 } ^ { t } \| u _ { s } ( \varphi _ { s } ^ { v } ( x ) ) \| \mathrm { d } s } } \\ & { } & { \leq e ^ { \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } \int _ { 0 } ^ { t } \| u _ { s } \| _ { C _ { \mathrm { L i p } } ^ { 1 } } ( 1 + \| \varphi _ { s } ^ { v } ( x ) \| ) \mathrm { d } s } \\ & { } & { \leq \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } e ^ { 2 \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } \int _ { 0 } ^ { t } \| u _ { s } \| _ { C _ { \mathrm { L i p } } ^ { 1 } } ( 1 + \| x \| ) \mathrm { d } s , } \end{array}
$$

$$
\begin{array} { r l } & { \leq \| u _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } \| v \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } e ^ { 2 \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } ( 1 + \| x \| ) , } \\ & { \leq C ( 1 + \| x \| ) . } \end{array}
$$

Further, by Lemma $\mathrm { A } . 2 \ \varphi _ { t } ^ { v }$ is Lipschitz continuous for every $t \in [ 0 , 1 ]$ with constant bounded by $e ^ { \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } . \mathrm { A s } ( \varphi _ { t } ^ { v } ) ^ { - 1 }$ can be interpreted as the flow of $- v _ { t - s }$ , the map $( \varphi _ { t } ^ { v } ) ^ { - 1 }$ is also Lipschitz continuous with constant $e ^ { \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } }$ for every $t \in [ 0 , 1 ]$ Hence, $v _ { t } ^ { \xi }$ is Lipschitz continuous with a Lipschitz constant bounded uniformly in $t \in [ 0 , 1 ]$ . As $\rho _ { 0 } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ , Lemma 4.4 guarantees that the second moments of $\rho _ { \bullet } ^ { v }$ stay uniformly bounded in $t \in [ 0 , 1 ]$ . Together, this shows $v _ { \bullet } ^ { \xi } \in L _ { T } ^ { 2 } L ^ { 2 } ( \rho _ { \bullet } ^ { v } )$ and by symmetry also $\zeta _ { \bullet } \in T _ { \rho _ { \bullet } ^ { v } } \mathrm { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ ).

Having checked that $\xi _ { \bullet } , \zeta _ { \bullet } \in T _ { \rho _ { \bullet } ^ { v } } \mathrm { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ , it remains to show $( 5 . 3 )$ . Definition 5.1 of the advective Fisher–Rao metric on $\Delta \mathbf { C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ and $( 5 . 4 )$ yield

$$
g _ { \rho _ { \bullet } ^ { v } } ^ { \mathrm { A F R } } ( \xi _ { \bullet } , \zeta _ { \bullet } ) = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } u _ { t } ( x ) \cdot w _ { t } ( x ) \rho _ { t } ^ { v } ( \mathrm { d } x ) \mathrm { d } t .
$$

On the other hand, Definition $3 { \cdot } 5$ of the advective Fisher–Rao metric on velocity fields gives

$$
g _ { v _ { \bullet } } ^ { \mathrm { A F R } } ( u _ { \bullet } , w _ { \bullet } ) = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } u _ { t } ( x ) \cdot w _ { t } ( x ) \rho _ { t } ^ { v } ( \mathrm { d } x ) \mathrm { d } t .
$$

Similar to before, we can relate the advective Fisher–Rao metric to the action functional from the dynamic formulation of optimal transport. To this end, we write the Benamou–Brenier formulation of the Wasserstein distance as

$$
W _ { 2 } ^ { 2 } ( \mu , \nu ) = \operatorname* { i n f } _ { \rho _ { \bullet } \in \mathrm { A C } _ { T } ^ { 2 } ( \mathcal P _ { 2 } ( \mathbb R ^ { d } ) ) } \left\{ \mathcal A ( p _ { \bullet } ) : \rho _ { 0 } = \mu , \rho _ { 1 } = \nu \right\} ,
$$

where the action of a curve is given by

$$
\mathcal { A } ( \rho _ { \bullet } ) = \frac { 1 } { 2 } \operatorname* { i n f } _ { v _ { \bullet } } \left\{ \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \Vert v _ { t } ( x ) \Vert _ { 2 } ^ { 2 } \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t : \partial _ { t } \rho _ { t } + \nabla \cdot ( \rho _ { t } v _ { t } ) = 0 \right\} .
$$

Alternatively, the action functional can be written as

$$
\mathcal { A } ( \rho _ { \bullet } ) = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| v _ { t } ^ { \rho } ( x ) \| _ { 2 } ^ { 2 } \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t ,
$$

where $v _ { \bullet } ^ { \rho } \in T _ { \rho _ { 0 } } L _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ is the velocity of $\rho _ { \bullet }$ meaning that $\begin{array} { r } { \partial _ { t } \rho _ { t } + \nabla \cdot ( \rho _ { t } v _ { t } ^ { \rho } ) = 0 } \end{array}$ Theorem 5.4. Consider two gradient velocity fields $\nabla \phi _ { \bullet } , \nabla \psi _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 } ( \mathbb { R } ^ { d } )$ and set $\nabla \phi _ { \bullet } ^ { h } : = \nabla \phi _ { \bullet } + h \nabla \psi _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 } ( \mathbb { R } ^ { d } )$ and denote the corresponding solutions of the continuity equation by $\rho _ { \bullet } ^ { h } \overset { \cdot } { \in } \mathsf { A C } _ { T } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ . Assume that for h small enough, we have $\rho _ { \bullet } ^ { h } \ll \rho _ { \bullet } ^ { 0 }$ with $\begin{array} { r } { \frac { \mathrm { d } \rho _ { \bullet } ^ { h } } { \mathrm { d } \rho _ { \bullet } ^ { 0 } } \in L _ { T } ^ { \infty } L ^ { \infty } ( \rho _ { \bullet } ^ { 0 } ) } \end{array}$ , and that there is a solution $v _ { \bullet } ^ { h } \in \mathcal { T } _ { \rho _ { \bullet } ^ { 0 } } \mathrm { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ to

$$
\begin{array} { r } { \partial _ { t } ( \rho _ { t } ^ { h } - \rho _ { t } ^ { 0 } ) + \nabla \cdot ( \rho _ { t } ^ { 0 } v _ { t } ^ { h } ) = 0 . } \end{array}
$$

Then, for $\begin{array} { r } { \xi _ { \bullet } : = \left. \frac { \mathrm { d } } { \mathrm { d } h } \right| _ { h = 0 } \rho _ { \bullet } ^ { h } \in T _ { \rho _ { \bullet } ^ { 0 } } \mathrm { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) ) } \end{array}$ it holds that

$$
g _ { \rho _ { \bullet } ^ { 0 } } ^ { \mathrm { A F R } } ( \xi _ { \bullet } , \xi _ { \bullet } ) = 2 \operatorname* { l i m } _ { h  0 } \frac { \mathcal { A } ( \rho _ { \bullet } ^ { h } ) - \mathcal { A } ( \rho _ { \bullet } ^ { 0 } ) - \delta ^ { + } \mathcal { A } ( \rho _ { \bullet } ^ { 0 } ) [ \rho _ { \bullet } ^ { h } - \rho _ { \bullet } ^ { 0 } ] } { h ^ { 2 } } .
$$

We use the following auxiliary result.

Lemma $\pmb { 5 } { \cdot } 5$ (Dual formulation of the action). Consider $\rho _ { \bullet } \in \mathsf { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ . Then, it holds that

$$
\mathcal { A } ( \rho _ { \bullet } ) = \operatorname* { s u p } _ { \phi _ { \bullet } \in C _ { \mathfrak { c } } ^ { \infty } ( ( 0 , 1 ) \times \mathbb { R } ^ { d } ) } - \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \left( \partial _ { t } \phi _ { t } ( x ) + \frac { 1 } { 2 } \| \nabla \phi _ { t } ( x ) \| _ { 2 } ^ { 2 } \right) \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t .
$$

Proof. The action functional is given by

$$
\begin{array} { r } { \mathcal { A } ( \rho _ { \bullet } ) = \operatorname* { i n f } \left\{ \mathcal { E } _ { \rho } ( v _ { \bullet } ) : \partial _ { t } \rho _ { t } + \nabla \cdot ( \rho _ { t } v _ { t } ) = 0 \right\} , } \end{array}
$$

where

$$
\mathcal { E } _ { \rho } ( v _ { \bullet } ) = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \lVert v _ { t } ( x ) \rVert ^ { 2 } \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t .
$$

For a fixed path $\rho _ { \bullet } \in \mathsf { A C } _ { T } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ we consider the Lagrangian of the constrained problem

$$
\begin{array} { r } { \mathcal { L } _ { \rho } \colon L _ { T } ^ { 2 } L ^ { 2 } ( \rho _ { \bullet } ) \times C _ { \mathsf { c } } ^ { \infty } ( ( 0 , 1 ) \times \mathbb { R } ^ { d } ) \to \mathbb { R } , } \end{array}
$$

where

$$
\mathcal { L } _ { \rho } ( v _ { \bullet } , \phi _ { \bullet } ) = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \left( \frac { 1 } { 2 } \| v _ { t } \| _ { 2 } ^ { 2 } - \partial _ { t } \phi _ { t } - v _ { t } \cdot \nabla \phi _ { t } \right) \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t .
$$

Then, it holds that

$$
\begin{array} { r } { \mathcal { A } ( \rho _ { \bullet } ) = \underset { v _ { \bullet } \in L _ { T } ^ { 2 } } { \operatorname* { i n f } } \underset { L ^ { 2 } ( p _ { \bullet } ) } { \operatorname* { s u p } } \underset { \phi \in C _ { \mathsf { c } } ^ { \infty } ( ( 0 , 1 ) \times \mathbb { R } ^ { d } ) } { \operatorname* { s u p } } \mathcal { L } _ { \rho } ( v _ { \bullet } , \phi _ { \bullet } ) } \\ { \geq \underset { \phi \in C _ { \mathsf { c } } ^ { \infty } ( ( 0 , 1 ) \times \mathbb { R } ^ { d } ) } { \operatorname* { s u p } } \underset { v _ { \bullet } \in L _ { T } ^ { 2 } } { \operatorname* { i n f } } \mathcal { L } _ { \rho } ( v _ { \bullet } , \phi _ { \bullet } ) . } \end{array}
$$

Carrying out the inner infimum gives

$$
\mathcal { A } ( \rho _ { \bullet } ) \geq \operatorname* { s u p } _ { \phi \in C _ { \mathfrak { c } } ^ { \infty } ( ( 0 , 1 ) \times \mathbb { R } ^ { d } ) } - \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \left( \partial _ { t } \phi _ { t } + \frac { 1 } { 2 } \Vert \nabla \phi _ { t } \Vert _ { 2 } ^ { 2 } \right) \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t .
$$

Let $v _ { \bullet } ^ { \rho } \in T _ { \rho _ { \bullet } } L _ { T } ^ { 2 } \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ denote the velocity of $\rho _ { \bullet }$ meaning that

$$
\begin{array} { r } { \partial _ { t } \rho _ { t } + \nabla \cdot ( \rho _ { t } v _ { t } ^ { \rho } ) = 0 } \end{array}
$$

holds in the distributional sense. Next, we seek an approximation $\nabla \phi _ { \bullet } ^ { \varepsilon }  v _ { \bullet } ^ { \rho }$ in $L _ { T } ^ { 2 } L ^ { 2 } ( \rho _ { \bullet } )$ , where $\phi _ { \bullet } ^ { \varepsilon } \in C _ { \mathrm { c } } ^ { \infty } ( ( 0 , 1 ) \times \mathbb { R } ^ { d } )$ for $\varepsilon \to 0$ . To see that this exists, it sufices to show that for a velocity field $v _ { \bullet } \in T _ { \rho \bullet } L _ { T } ^ { 2 } { \mathcal { P } } _ { 2 } ( \mathbb { R } ^ { d } )$ , the orthogonality

$$
\int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } v _ { t } \cdot \nabla \phi _ { t } \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t = 0 \quad \mathrm { f o r ~ a l l ~ } \phi _ { \bullet } \in C _ { \mathsf { c } } ^ { \infty } ( ( 0 , 1 ) \times \mathbb { R } ^ { d } )
$$

implies that $v _ { \bullet } = 0$ . To see this, we note that this implies

$$
\int _ { 0 } ^ { 1 } \eta ( t ) \int _ { \mathbb { R } ^ { d } } v _ { t } \cdot \nabla \psi \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t = 0 \quad \mathrm { f o r ~ a l l ~ } \psi \in C _ { \mathsf { c } } ^ { \infty } ( \mathbb { R } ^ { d } ) , \eta \in C _ { \mathsf { c } } ^ { \infty } ( ( 0 , 1 ) ) .
$$

In particular, this implies

$$
\int _ { \mathbb { R } ^ { d } } { v _ { t } \cdot \nabla \psi \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t } \quad \mathrm { f o r ~ a l l ~ } \psi \in C _ { \mathrm { c } } ^ { \infty } ( \mathbb { R } ^ { d } )
$$

for almost every $t \in [ 0 , 1 ]$ . As $v _ { t } \in T _ { \rho _ { t } } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ for almost every $t \in [ 0 , 1 ]$ this implies $v _ { t } = 0$ for almost every $t \in [ 0 , 1 ]$ and hence $v _ { \bullet } = 0$ in $L _ { T } ^ { 2 } L ^ { 2 } ( \rho _ { t } )$ . Now, we proceed and estimate

$$
\begin{array} { r l } & { \underset { \phi \in C _ { \epsilon } ^ { \infty } ( ( 0 , 1 ) \times \mathbb { R } ^ { d } ) } { \operatorname* { s u p } } - \displaystyle \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } ( \partial _ { t } \phi _ { t } + \frac 1 2 \| \nabla \phi _ { t } \| _ { 2 } ^ { 2 } ) \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t } \\ & { \ge - \displaystyle \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } ( \partial _ { t } \phi _ { t } ^ { \varepsilon } + \frac 1 2 \| \nabla \phi _ { t } ^ { \varepsilon } \| _ { 2 } ^ { 2 } ) \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t } \\ & { = \displaystyle \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } ( v _ { t } ^ { \rho } \cdot \nabla \phi _ { t } ^ { \varepsilon } - \frac 1 2 \| \nabla \phi _ { t } ^ { \varepsilon } \| _ { 2 } ^ { 2 } ) \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t } \\ & {  \displaystyle \frac 1 2 \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| v _ { t } ^ { \rho } \| _ { 2 } ^ { 2 } \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t = \mathcal { A } ( \rho _ { \bullet } ) } \end{array}
$$

for $\varepsilon  0 .$ , showing the other inequality.

Lemma 5.6 (First variation of the action). Consider $\rho _ { \bullet } , \sigma _ { \bullet } \in \operatorname { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ , set $\xi _ { t } : = \sigma _ { t } - \rho _ { t }$ , and denote the velocity of $\cdot _ { \rho _ { \bullet } }$ by $v _ { \bullet } ^ { \rho } \in \mathcal { T } _ { \rho _ { \bullet } } \mathrm { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ . Assume now that $\sigma _ { t } \ll \rho _ { t }$ and $\begin{array} { l } { \frac { \mathrm { d } \sigma _ { \bullet } } { \mathrm { d } \rho _ { \bullet } } \in L _ { T } ^ { \infty } L ^ { \infty } ( \rho _ { \bullet } ) } \end{array}$ , as well as the existence of $v _ { \bullet } ^ { \xi } \in L _ { T } ^ { 2 } L ^ { 2 } ( \rho _ { \bullet } )$ such that

$$
\begin{array} { r } { \partial _ { t } \xi _ { t } + \nabla \cdot ( \rho _ { t } v _ { t } ^ { \xi } ) = 0 } \end{array}
$$

holds in the distributional sense. Then it holds that

$$
\delta ^ { + } \mathcal { A } ( \rho _ { \bullet } ) [ \xi _ { \bullet } ] = \int _ { 0 } ^ { 1 } \left( \int _ { \mathbb { R } ^ { d } } v _ { t } ^ { \rho } \cdot v _ { t } ^ { \xi } \rho _ { t } ( \mathrm { d } x ) - \int _ { \mathbb { R } ^ { d } } \frac { 1 } { 2 } \| v _ { t } ^ { \rho } \| ^ { 2 } \xi _ { t } ( \mathrm { d } x ) \right) \mathrm { d } t .\tag{5.5}
$$

Proof. The dual formulation of the action functional gives

$$
\mathcal { A } ( \rho _ { \bullet } ) = \operatorname* { s u p } _ { \phi _ { \bullet } \in C _ { \mathrm { c } } ^ { \infty } ( ( 0 , 1 ) \times \mathbb { R } ^ { d } ) } L _ { \phi } ( \rho _ { \bullet } ) ,
$$

where

$$
L _ { \phi } ( \rho _ { \bullet } ) = - \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \left( \partial _ { t } \phi _ { t } + \frac { 1 } { 2 } \| \nabla \phi _ { t } \| _ { 2 } ^ { 2 } \right) \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t ,
$$

where $L _ { \phi }$ is a linear function of $\rho _ { \bullet }$ . We now consider the following functionals on R given by

$$
a _ { \phi } ( h ) : = L _ { \phi } ( \rho _ { \bullet } + h \xi _ { \bullet } ) ,
$$

which is well defined for $h \in [ 0 ,$ , 1] and can be extended to globally linear functions on R. Further, we set

$$
f ( h ) : = \operatorname* { s u p } _ { \phi \in C _ { \mathrm { c } } ^ { \infty } ( ( 0 , 1 ) \times \mathbb { R } ^ { d } ) } a _ { \phi } ( h ) .
$$

Note that $f ( h ) = \mathcal { A } ( \rho _ { \bullet } + h \xi _ { \bullet } )$ for all $h \in [ 0 , 1 ]$ . It is the goal to compute the derivative $f ^ { \prime } ( 0 )$ , for which we first compute its subgradient, which, as the subgradient of a support function, is given by

$$
\partial f ( 0 ) = \mathrm { c o n v } \Big \{ \operatorname* { l i m } _ { n \to \infty } a _ { \phi ^ { n } } ( 1 ) - a _ { \phi ^ { n } } ( 0 ) : \operatorname* { l i m } _ { n \to \infty } a _ { \phi ^ { n } } ( 0 ) = f ( 0 ) \Big \} ,
$$

see for example [Roc97]. Let us fix such a sequence $( \phi _ { \bullet } ^ { n } ) _ { n \in \mathbb { N } } \subseteq C _ { \mathrm { c } } ^ { \infty } ( ( 0 , 1 ) \times \mathbb { R } ^ { d } )$ then we have

$$
\begin{array} { l l } { \displaystyle a _ { \phi ^ { n } } ( 0 ) - f ( 0 ) = L _ { \phi ^ { n } } ( \rho _ { \bullet } ) - \mathcal { A } ( \rho _ { \bullet } ) } \\ { \displaystyle \quad \quad } & { = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \left( \nabla \phi _ { t } ^ { n } \cdot v _ { t } ^ { \rho } - \frac 1 2 \| \nabla \phi _ { t } ^ { n } \| _ { 2 } ^ { 2 } - \frac 1 2 \| v _ { t } ^ { \rho } \| _ { 2 } ^ { 2 } \right) \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t } \\ { \displaystyle \quad } & { = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| \nabla \phi _ { t } ^ { n } - v _ { t } ^ { \rho } \| _ { 2 } ^ { 2 } \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t \to 0 } \end{array}
$$

for $n  \infty ,$ hence $\nabla \phi _ { \bullet } ^ { n }  v _ { \bullet } ^ { \rho }$ for $n \to \infty$ in $L _ { T } ^ { 2 } L ^ { 2 } ( \rho _ { \bullet } )$ . Now, we obtain

$$
\begin{array} { r l } & { a _ { \phi ^ { n } } ( 1 ) - a _ { \phi ^ { n } } ( 0 ) = - \displaystyle \int _ { 0 } ^ { 1 } \left( \int _ { \mathbb R ^ { d } } \partial _ { t } \phi _ { t } ^ { n } + \frac 1 2 \| \nabla \phi _ { t } ^ { n } \| ^ { 2 } \right) \xi _ { t } ( \mathrm { d } x ) \mathrm { d } t } \\ & { \quad \quad \quad = \displaystyle \int _ { 0 } ^ { 1 } \left( \int _ { \mathbb R ^ { d } } \nabla \phi _ { t } ^ { n } \cdot v _ { t } ^ { \xi } \rho _ { t } ( \mathrm { d } x ) - \int _ { \mathbb R ^ { d } } \frac 1 2 \| \nabla \phi _ { t } ^ { n } \| ^ { 2 } \xi _ { t } ( \mathrm { d } x ) \right) \mathrm { d } t } \end{array}
$$

As $\sigma _ { \bullet }$ is dominated by $\rho _ { \bullet }$ with a globally bounded Radon–Nikodym derivative, we obtain

$$
\operatorname* { l i m } _ { n \to \infty } a _ { \phi ^ { n } } ( 1 ) - a _ { \phi ^ { n } } ( 0 ) = \int _ { 0 } ^ { 1 } \left( \int _ { \mathbb { R } ^ { d } } v _ { t } ^ { \rho } \cdot v _ { t } ^ { \xi } \rho _ { t } ( \mathrm { d } x ) - \int _ { \mathbb { R } ^ { d } } { \frac { 1 } { 2 } } \| v _ { t } ^ { \rho } \| ^ { 2 } \xi _ { t } ( \mathrm { d } x ) \right) \mathrm { d } t
$$

as asserted.

Lemma 5.7 (Bregman divergence of the Benamou–Brenier action functional). Consider two curves $\rho _ { \bullet } , \sigma _ { \bullet } \in \mathrm { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ with velocities $v _ { \bullet } ^ { \rho } \in \mathcal { T } _ { \rho _ { \bullet } } \mathrm { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ and $v _ { \bullet } ^ { \sigma } \in \mathcal { T } _ { \sigma _ { \bullet } } \mathrm { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ , respectively. Assume that $\sigma _ { t } \ \ll \ \rho _ { t }$ and $\textstyle { \frac { \mathrm { d } \sigma _ { \bullet } } { \mathrm { d } \rho _ { \bullet } } } \in$ $L _ { T } ^ { \infty } L ^ { \infty } ( \rho _ { \bullet } )$ , that $v _ { \bullet } ^ { \sigma } \in L _ { T } ^ { 2 } L ^ { 2 } ( \rho _ { \bullet } ) , v _ { \bullet } ^ { \rho } \in L _ { T } ^ { 2 } L ^ { 2 } ( \sigma _ { \bullet } )$ , and that there is a solution $v _ { \bullet } \in \mathcal { T } _ { \sigma _ { \bullet } } \mathrm { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ to

$$
\begin{array} { r } { \partial _ { t } ( \rho _ { t } - \sigma _ { t } ) + \nabla \cdot ( \sigma _ { t } v _ { t } ) = 0 . } \end{array}
$$

Then the Bregman divergence of the action functional is given by

$$
D _ { \mathcal { A } } ( \rho _ { \bullet } , \sigma _ { \bullet } ) = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| v _ { t } ^ { \rho } ( x ) - v _ { t } ^ { \sigma } ( x ) \| _ { 2 } ^ { 2 } \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t .\tag{5.6}
$$

Proof. The Bregman divergence is given by

$$
D _ { A } ( \rho _ { \bullet } , \sigma _ { \bullet } ) = A ( \rho _ { \bullet } ) - A ( \sigma _ { \bullet } ) - \delta ^ { + } A ( \sigma _ { \bullet } ) [ \rho _ { \bullet } - \sigma _ { \bullet } ] .
$$

To use the expression of the first variation from Lemma $5 . 6 ,$ we first note that

$$
- \nabla \cdot ( \sigma _ { t } v _ { t } ) = \partial _ { t } ( \rho _ { t } - \sigma _ { t } ) = - \nabla \cdot ( \rho _ { t } v _ { t } ^ { \rho } - \sigma _ { t } v _ { t } ^ { \sigma } ) ,
$$

which holds in distribution meaning that for any $\psi \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } )$ and almost every $t \in [ 0 , 1 ]$ , we have

$$
\begin{array} { r l r } {  { \int _ { \mathbb { R } ^ { d } } \nabla \psi ( \boldsymbol { x } ) \cdot \nabla v _ { t } ( \boldsymbol { x } ) \sigma _ { t } ( \mathrm { d } \boldsymbol { x } ) = \int _ { \mathbb { R } ^ { d } } \nabla \psi ( \boldsymbol { x } ) \cdot v _ { t } ^ { \rho } ( \boldsymbol { x } ) \rho _ { t } ( \mathrm { d } \boldsymbol { x } ) } } \\ & { } & { \quad - \int _ { \mathbb { R } ^ { d } } \nabla \psi ( \boldsymbol { x } ) \cdot v _ { t } ^ { \sigma } ( \boldsymbol { x } ) \sigma _ { t } ( \mathrm { d } \boldsymbol { x } ) . } \end{array}
$$

By density and as $v _ { t } ^ { \sigma } \in L ^ { 2 } ( \rho _ { t } )$ for almost every $t \in [ 0 , 1 ]$ , we obtain

$$
\begin{array} { r l r } & { } & { \displaystyle \int _ { \mathbb { R } ^ { d } } \boldsymbol { v } _ { t } ^ { \sigma } ( \boldsymbol { x } ) \cdot \boldsymbol { v } _ { t } ( \boldsymbol { x } ) \sigma _ { t } ( \mathrm { d } \boldsymbol { x } ) = \int _ { \mathbb { R } ^ { d } } \boldsymbol { v } _ { t } ^ { \sigma } ( \boldsymbol { x } ) \cdot \boldsymbol { v } _ { t } ^ { \rho } ( \boldsymbol { x } ) \rho _ { t } ( \mathrm { d } \boldsymbol { x } ) } \\ & { } & { \displaystyle - \int _ { \mathbb { R } ^ { d } } \boldsymbol { v } _ { t } ^ { \sigma } ( \boldsymbol { x } ) \cdot \boldsymbol { v } _ { t } ^ { \sigma } ( \boldsymbol { x } ) \sigma _ { t } ( \mathrm { d } \boldsymbol { x } ) } \end{array}
$$

for almost every $t \in [ 0 , 1 ]$ . Now, we use this to compute

$$
\begin{array} { l } { \displaystyle D _ { \mathcal { A } } ( \rho _ { \bullet } , \sigma _ { \bullet } ) = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| \nabla \phi _ { t } ^ { \rho } \| _ { 2 \rho t } ^ { 2 } ( \mathrm { d } x ) \mathrm { d } t - \displaystyle \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| \nabla \phi _ { t } ^ { \sigma } \| _ { 2 } ^ { 2 } \sigma _ { t } ( \mathrm { d } x ) \mathrm { d } t } \\ { \displaystyle \quad \quad - \int _ { 0 } ^ { 1 } \left( \int _ { \mathbb { R } ^ { d } } \nabla \phi _ { t } ^ { \sigma } \cdot \nabla \phi _ { t } \sigma _ { t } ( \mathrm { d } x ) - \displaystyle \frac { 1 } { 2 } \int _ { \mathbb { R } ^ { d } } \| \nabla \phi _ { t } ^ { \sigma } \| _ { 2 } ^ { 2 } ( \rho _ { t } - \sigma _ { t } ) ( \mathrm { d } x ) \right) \mathrm { d } t } \\ { \displaystyle \quad \quad = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| \nabla \phi _ { t } ^ { \rho } \| _ { 2 \rho t } ^ { 2 } ( \mathrm { d } x ) \mathrm { d } t - \displaystyle \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| \nabla \phi _ { t } ^ { \sigma } \| _ { 2 } ^ { 2 } \sigma _ { t } \mathrm { d } x \mathrm { d } t } \\ { \displaystyle \quad \quad - \int _ { 0 } ^ { 1 } \left( \int _ { \mathbb { R } ^ { d } } \nabla \phi _ { t } ^ { \sigma } \cdot \nabla \phi _ { t } ^ { \rho } \rho _ { t } ( \mathrm { d } x ) - \int _ { \mathbb { R } ^ { d } } \| \nabla \phi _ { t } ^ { \sigma } \| _ { 2 } ^ { 2 } \sigma _ { t } ( \mathrm { d } x ) \right) \mathrm { d } t } \end{array}
$$

The Advective Fisher–Rao Metric on Paths of Probability Measures

$$
\begin{array} { r l } & { \quad + \displaystyle \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| \nabla \phi _ { t } ^ { \sigma } \| _ { 2 } ^ { 2 } ( \rho _ { t } - \sigma _ { t } ) ( \mathrm { d } x ) \mathrm { d } t } \\ & { = \displaystyle \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \left( \| \nabla \phi _ { t } ^ { \rho } \| _ { 2 } ^ { 2 } - 2 \nabla \phi _ { t } ^ { \rho } \cdot \nabla \phi _ { t } ^ { \sigma } + \| \nabla \phi _ { t } ^ { \sigma } \| _ { 2 } ^ { 2 } \right) \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t } \\ & { = \displaystyle \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| \nabla \phi _ { t } ^ { \rho } - \nabla \phi _ { t } ^ { \sigma } \| _ { 2 } ^ { 2 } \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t . } \end{array}
$$

Proof of Theorem $5 { \cdot } 4 { \cdot }$ . By definition of the Bregman divergence, we have

$$
\frac { \mathcal { A } ( \rho _ { \bullet } ^ { h } ) - \mathcal { A } ( \rho _ { \bullet } ^ { 0 } ) - \delta \mathcal { A } ( \rho _ { \bullet } ^ { 0 } ) [ \rho _ { \bullet } ^ { h } - \rho _ { \bullet } ^ { 0 } ] } { h ^ { 2 } } = \frac { D _ { \mathcal { A } } ( \rho _ { \bullet } ^ { h } , \rho _ { \bullet } ^ { 0 } ) } { h ^ { 2 } } .
$$

Substituting the explicit formula $_ { ( 5 . 6 ) }$ for the Bregman divergence yields

$$
\begin{array} { r l } & { \frac { D _ { A } ( \rho _ { \bullet } ^ { h } , \rho _ { \bullet } ^ { 0 } ) } { h ^ { 2 } } = \frac { 1 } { 2 h ^ { 2 } } \displaystyle \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| \nabla \phi _ { t } ^ { h } ( x ) - \nabla \phi _ { t } ( x ) \| _ { 2 } ^ { 2 } \rho _ { t } ^ { h } ( \mathrm { d } x ) \mathrm { d } t } \\ & { \quad \quad \quad \quad \quad = \frac { 1 } { 2 h ^ { 2 } } \displaystyle \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| h \nabla \psi _ { t } ( x ) \| _ { 2 } ^ { 2 } \rho _ { t } ^ { h } ( \mathrm { d } x ) \mathrm { d } t } \\ & { \quad \quad \quad \quad \quad = \displaystyle \frac { 1 } { 2 } \displaystyle \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| \nabla \psi _ { t } ( x ) \| _ { 2 } ^ { 2 } \rho _ { t } ^ { h } ( \mathrm { d } x ) \mathrm { d } t . } \end{array}
$$

By standard $W _ { 2 }$ stability of the continuity equation, $\rho _ { t } ^ { h }  \rho _ { t } ^ { 0 }$ in $W _ { 2 }$ as $h  0$ , see for example $\left[ \operatorname { A G 1 2 } \right]$ . Since $\nabla \psi _ { t } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 } ( \mathbb { R } ^ { d } )$ it grows at most linearly. Since $W _ { 2 }$ convergence implies convergence when testing with functions with at most quadratic growth, see $[ \mathrm { A G } 1 2 $ , Proposition $3 { \cdot } 4 ]$ , we obtain pointwise in t convergence of the integrals

$$
\operatorname* { l i m } _ { h \to 0 } \int _ { \mathbb { R } ^ { d } } \lVert \nabla \psi _ { t } ( x ) \rVert _ { 2 } ^ { 2 } \rho _ { t } ^ { h } ( \mathrm { d } x ) = \int _ { \mathbb { R } ^ { d } } \lVert \nabla \psi _ { t } ( x ) \rVert _ { 2 } ^ { 2 } \rho _ { t } ^ { 0 } ( \mathrm { d } x ) .
$$

Finally, since the second moments of $\rho _ { t } ^ { h }$ are uniformly bounded in t and $h ,$ see Lemma $4 { \cdot } 4$ , we can apply the dominated convergence theorem to pass the limit inside the time integral and obtain

$$
\operatorname* { l i m } _ { h \to 0 } \frac { \mathcal { A } ( \rho _ { \bullet } ^ { h } ) - \mathcal { A } ( \rho _ { \bullet } ^ { 0 } ) - \delta \mathcal { A } ( \rho _ { \bullet } ^ { 0 } ) [ \rho _ { \bullet } ^ { h } - \rho _ { \bullet } ^ { 0 } ] } { h ^ { 2 } } = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| \nabla \psi _ { t } ( x ) \| _ { 2 } ^ { 2 } \rho _ { t } ^ { 0 } ( \mathrm { d } x ) \mathrm { d } t .
$$

Further, from Definition $3 { \cdot } 5$ and Lemma $5 { \cdot } 3$ it holds that

$$
g _ { \rho _ { \bullet } ^ { \mathrm { ( } \mathbf { R } ) } } ^ { \mathrm { A F R } } ( \xi _ { \bullet } , \xi _ { \bullet } ) = g _ { \nabla \phi _ { \bullet } } ^ { \mathrm { A F R } } ( \nabla \psi _ { \bullet } , \nabla \psi _ { \bullet } ) = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \lVert \nabla \psi _ { t } ( x ) \rVert _ { 2 } ^ { 2 } \rho _ { t } ^ { 0 } ( \mathrm { d } x ) \mathrm { d } t .
$$

Consider the case that $\rho _ { \bullet } , \rho _ { \bullet } ^ { \star } \in \mathsf { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ . A natural projection of the flow-matching energy to the space of paths of densities is given by

$$
\mathcal E ( \rho _ { \bullet } ) = \frac 1 2 D _ { A } ( \rho _ { \bullet } ^ { \star } , \rho _ { \bullet } ) = \frac 1 2 \int _ { 0 } ^ { 1 } \int _ { \mathbb R ^ { d } } \Vert v _ { t } ^ { \rho } ( x ) - v _ { t } ^ { \star } ( x ) \Vert _ { 2 } ^ { 2 } \rho _ { t } ^ { \star } ( x ) \mathrm { d } x \mathrm { d } t ,
$$

where $v _ { \bullet } ^ { \rho } \in T _ { \rho \bullet } L _ { T } ^ { 2 } { \mathcal { P } } _ { 2 } ( \mathbb { R } ^ { d } )$ and $v _ { \bullet } ^ { \star } \in T _ { \rho _ { \bullet } ^ { \star } } L _ { T } ^ { 2 } \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ are the velocities of $\rho _ { \bullet }$ and $\rho _ { \bullet } ^ { \star }$ respectively, meaning that

$$
\begin{array} { r l } & { \partial _ { t } \rho _ { t } + \nabla \cdot ( \rho _ { t } v _ { t } ^ { \rho } ) = 0 \quad \mathrm { a n d } } \\ & { \partial _ { t } \rho _ { t } ^ { \star } + \nabla \cdot ( \rho _ { t } ^ { \star } v _ { t } ^ { \star } ) = 0 . } \end{array}\tag{5.7}
$$

Also in this case, we obtain a characterization of the gradient in terms of an optimal update direction.

Theorem 5.8 (Optimality). Consider $\rho _ { \bullet } ^ { \star } \in \mathsf { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ and gradient fields $v _ { \bullet } = \nabla \phi _ { \bullet } , u _ { \bullet } = \nabla \psi _ { \bullet } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 } ( \mathbb { R } ^ { d } )$ and denote the solutions to the continuity equation with velocity field v<sub>•</sub> + hu<sub>•</sub> by $\rho _ { \bullet } ^ { v + h u }$ and $\rho _ { 0 } ^ { v } = \rho _ { 0 } ^ { \star }$ and consider the variation $\begin{array} { r } { \xi _ { \bullet } = \left. \frac { \mathrm { d } } { \mathrm { d } h } \right| _ { h = 0 } \rho _ { \bullet } ^ { v + h u } \in T _ { \rho _ { \bullet } } \mathrm { A C } _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) ) } \end{array}$ . Assume that $\rho _ { \bullet } ^ { v } \ll \rho _ { \bullet } ^ { \star }$ and that

$$
( v _ { \bullet } - v _ { \bullet } ^ { \star } ) \cdot \frac { \mathrm { d } \rho _ { \bullet } ^ { \star } } { \mathrm { d } \rho _ { \bullet } ^ { v } } \in L _ { T } ^ { 2 } L ^ { 2 } ( \rho _ { \bullet } ^ { v } ) ,
$$

where $v _ { \bullet } ^ { \star } \in T _ { \rho _ { \bullet } ^ { \star } } L _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ is the velocityfieldfor $\rho _ { \bullet } ^ { \star }$ . Then it holds that

$$
\delta { \mathcal E } ( \rho _ { \bullet } ^ { v } ) [ \xi _ { \bullet } ] = g _ { \rho _ { \bullet } ^ { v } } ^ { \mathrm { A F R } } ( \rho _ { \bullet } ^ { v } - \rho _ { \bullet } ^ { \star } , \xi _ { \bullet } ) .
$$

Proof. Let us again fix $\nabla \psi ^ { \xi } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 } ( \mathbb { R } ^ { d } )$ . The first variation is given by

$$
\begin{array} { l } { { \displaystyle \delta { \mathcal E } ( { \boldsymbol \rho } _ { \bullet } ) [ \xi _ { \bullet } ] = \left. \frac { \mathrm { d } } { \mathrm { d } h } \right. _ { h = 0 } \mathcal E ( { \boldsymbol \rho } _ { \bullet } ^ { v + h u } ) } \ ~ } \\ { { \displaystyle ~ = \left. \frac { \mathrm { d } } { \mathrm { d } h } \right. _ { h = 0 } \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { { \mathbb R } ^ { d } } \lVert ( v _ { t } + h u _ { t } ) ( x ) - v _ { t } ^ { \star } ( x ) \rVert _ { 2 } ^ { 2 } \rho _ { t } ^ { \star } ( \mathrm { d } x ) \mathrm { d } t } \ ~ } \\ { { \displaystyle ~ = \int _ { 0 } ^ { 1 } \int _ { { \mathbb R } ^ { d } } ( v _ { t } ( x ) - v _ { t } ^ { \star } ( x ) ) \cdot u _ { t } ( x ) \rho _ { t } ^ { \star } ( \mathrm { d } x ) \mathrm { d } t } , } \end{array}
$$

where again

$$
\partial _ { t } \xi _ { t } + \nabla \cdot ( \xi _ { t } \boldsymbol { v } _ { t } ) = - \nabla \cdot ( \rho _ { t } \boldsymbol { u } _ { t } )
$$

by Lemma $3 . 8$ . Note that for $\zeta _ { \bullet } = \rho _ { \bullet } ^ { v } - \rho _ { \bullet } ^ { \star }$ we have

$$
\partial _ { t } \zeta _ { t } + \nabla \cdot ( \zeta _ { t } v _ { t } ) = - \nabla \cdot ( \rho _ { t } ^ { \star } ( v _ { t } - v _ { t } ^ { \star } ) ) = - \nabla \cdot ( \rho _ { t } \tilde { w } _ { t } ) ,
$$

where $\begin{array} { r } { \tilde { w } _ { \bullet } = ( v _ { \bullet } - v _ { \bullet } ^ { \star } ) \frac { \mathrm { d } \rho _ { \bullet } ^ { \star } } { \mathrm { d } \rho _ { \bullet } ^ { v } } \in L _ { T } ^ { 2 } L ^ { 2 } ( \rho _ { \bullet } ) } \end{array}$ . Let $w _ { \bullet } \in \mathcal { T } _ { \rho _ { \bullet } ^ { v } } L _ { T } ^ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ be the $L _ { T } ^ { 2 } L ^ { 2 } ( \rho _ { \bullet } )$ projection of $\tilde { w } _ { \bullet }$ <sup>•</sup>to the gradient fields, then we obtain

$$
\partial _ { t } \zeta _ { t } + \nabla \cdot ( \zeta _ { t } \boldsymbol { v } _ { t } ) = - \nabla \cdot ( \rho _ { t } \boldsymbol { w } _ { t } )
$$

The Flat $L ^ { 2 } ( \rho _ { \bullet } ^ { \star } )$ )-Geometry

as well as

$$
\begin{array} { r } { - \nabla \cdot ( \rho _ { t } w _ { t } ) = - \nabla \cdot ( \rho _ { t } ^ { \star } ( v _ { t } - v _ { t } ^ { \star } ) ) . } \end{array}
$$

Definition 5.1 now yields

$$
\begin{array} { r l } & { g _ { \rho _ { \bullet } ^ { v } } ^ { \mathrm { A F R } } ( \zeta _ { \bullet } , \xi _ { \bullet } ) = \displaystyle \int _ { 0 } ^ { 1 } \int _ { \mathbb R ^ { d } } w _ { t } ( x ) \cdot u _ { t } ( x ) \rho _ { t } ^ { v } ( \mathrm { d } x ) \mathrm { d } t } \\ & { \qquad = \displaystyle \int _ { 0 } ^ { 1 } \int _ { \mathbb R ^ { d } } ( v _ { t } ( x ) - v _ { t } ^ { \star } ( x ) ) \cdot u _ { t } ^ { \xi } ( x ) \rho _ { t } ^ { \star } ( \mathrm { d } x ) \mathrm { d } t } \\ & { \qquad = \delta \mathcal { E } ( \rho _ { \bullet } ^ { v } ) [ \xi _ { \bullet } ] , } \end{array}
$$

which shows the assertion.

## 6 The Flat $L ^ { 2 } ( \rho _ { \bullet } ^ { \star } )$ -Geometry

Next, we contrast the advective Fisher–Rao metric with the flat $L ^ { 2 } .$ -metric on velocity fields, which is given by

$$
( u _ { \bullet } , w _ { \bullet } ) _ { L ^ { 2 } ( p _ { \bullet } ^ { \star } ) } = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } u _ { t } ( x ) ^ { \top } w _ { t } ( x ) \rho _ { t } ^ { \star } ( \mathrm { d } x ) \mathrm { d } t .\tag{6.1}
$$

Recall that we are investigating the energy function

$$
E \colon L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 } \to \mathbb { R } , \quad v _ { \bullet } \mapsto \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| v _ { t } ( x ) - v _ { t } ^ { \star } ( x ) \| _ { 2 } ^ { 2 } \rho _ { t } ^ { \star } ( \mathrm { d } x ) \mathrm { d } t .
$$

The $L ^ { 2 }$ -metric agrees with the Hessian of $E ,$ which is a quadratic functional on the velocity fields. Hence, it is well known that the gradient points directly into the direction of the optimal velocity field $v _ { \bullet } ^ { \star }$

Lemma ${ \bf 6 . 1 }$ . The $L ^ { 2 } ( \rho _ { \bullet } ^ { \star } )$ -gradient of the energy $E \colon L ^ { 2 } ( \rho _ { \bullet } ^ { \star } ) \to$ R is given by $v _ { \bullet } - v _ { \bullet } ^ { \star }$

Proof. Note that the energy is given by the squared norm $\begin{array} { r } { E ( v _ { \bullet } ) = \frac { 1 } { 2 } \| v _ { \bullet } - v _ { \bullet } ^ { \star } \| _ { L ^ { 2 } ( \rho _ { \bullet } ^ { \star } ) } ^ { 2 } . } \end{array}$ A direct computation yields that for any $u _ { \bullet } \in L ^ { 2 } ( \rho _ { \bullet } ^ { \star } )$ it holds that

$$
\delta E ( v _ { \bullet } ) [ u _ { \bullet } ] = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } ( v _ { t } ( x ) - v _ { t } ^ { \star } ( x ) ) \cdot u _ { t } ( x ) \rho _ { t } ^ { \star } ( \mathrm { d } x ) { \mathrm d } t = ( v _ { \bullet } - v _ { \bullet } ^ { \star } , u _ { \bullet } ) _ { L ^ { 2 } ( \rho _ { \bullet } ^ { \star } ) } ,
$$

which yields the claim.

The $L ^ { 2 }$ gradient will not lead to a mixture geodesic in the space of paths of probability densities, but we can show that it is close to the mixture geodesic if the energy is small.

Theorem 6.2 $( L ^ { 2 } .$ -gradients and m-geodesics). Consider an initial distribution $\rho _ { 0 } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ , velocityfields $v _ { \bullet } , v _ { \bullet } ^ { \star } \in L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 }$ , the variation $u _ { \bullet } : = v _ { \bullet } - v _ { \bullet } ^ { \star }$ , denote the solution ofthe continuity equation with the velocityfields $v _ { \bullet } , v _ { \bullet } ^ { \star }$ , and $v _ { \bullet } + h u _ { \bullet }$

The Flat $L ^ { 2 } ( \rho _ { \bullet } ^ { \star } )$ -Geometry

by $\rho _ { \bullet } ^ { v } , \rho _ { \bullet } ^ { \star }$ , and $\rho _ { \bullet } ^ { v + h u }$ , and denote the corresponding variation $\begin{array} { r } { \xi _ { \bullet } : = \left. \frac { \mathrm { d } } { \mathrm { d } h } \right| _ { h = 0 } \rho _ { \bullet } ^ { v + h u } } \end{array}$ Then there is a constant $c > 0$ such thatfor all $t \in [ 0 , 1 ]$ it holds that

$$
\| \xi _ { t } - ( \rho _ { t } ^ { v } - \rho _ { t } ^ { \star } ) \| _ { ( C _ { \mathrm { L i p } } ^ { 1 } ) ^ { * } } \leq c E ( v _ { \bullet } ) ^ { \frac { 1 } { 2 } } .
$$

Further, the constant can be chosen to be

$$
c = \sqrt { 2 } \left( 2 e ^ { \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } + \left( \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } + \| v _ { \bullet } ^ { \star } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } \right) e ^ { 2 \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } \right) .
$$

Proof. For any $\phi \in C _ { \mathrm { L i p } } ^ { 1 }$ and $t \in [ 0 , 1 ]$ , Lemma $3 . 8$ yields that

$$
\langle \xi _ { t } ^ { u } , \phi \rangle = \int _ { \mathbb { R } ^ { d } } \nabla \phi ( \varphi _ { t } ^ { v } ( x ) ) \cdot \psi _ { t } ^ { u } ( x ) \rho _ { 0 } ( \mathrm { d } x ) ,
$$

where $\varphi _ { \bullet } ^ { v }$ denotes the flow induced by $v _ { \bullet }$ and $\psi _ { \bullet } ^ { u }$ is the unique solution to

$$
\partial _ { t } \psi _ { t } ^ { u } ( x ) = D v _ { t } ( \varphi _ { t } ^ { v } ( x ) ) \cdot \psi _ { t } ^ { u } ( x ) + u _ { t } ( \varphi _ { t } ^ { v } ( x ) ) , \quad \psi _ { 0 } ^ { v } ( x ) = 0
$$

for all $t \in [ 0 , 1 ]$ and $\boldsymbol { x } \in \mathbb { R } ^ { d }$ . It holds that

$$
\begin{array} { r l } & { \displaystyle \left. \left. \xi _ { t } ^ { u } - ( \rho _ { t } ^ { v } - \rho _ { t } ^ { \star } ) , \phi \right. \right. = \displaystyle \left. \int _ { \mathbb { R } ^ { d } } \nabla \phi ( \varphi _ { t } ^ { v } ) \cdot \psi _ { t } ^ { u } - ( \phi ( \varphi _ { t } ^ { v } ) - \phi ( \varphi _ { t } ^ { \star } ) ) \rho _ { 0 } ( \mathrm { d } x ) \right. } \\ & { \qquad \displaystyle \leq \| \nabla \phi \| _ { \infty } \int _ { \mathbb { R } ^ { d } } ( \| \psi _ { t } ^ { u } \| + \| \varphi _ { t } ^ { v } - \varphi _ { t } ^ { \star } \| ) \rho _ { 0 } ( \mathrm { d } x ) } \\ & { \qquad \displaystyle \leq \| \nabla \phi \| _ { \infty } \left( \int _ { \mathbb { R } ^ { d } } \| \psi _ { t } ^ { u } \| \rho _ { 0 } ( \mathrm { d } x ) + \int _ { \mathbb { R } ^ { d } } \| \varphi _ { t } ^ { v } - \varphi _ { t } ^ { \star } \| \rho _ { 0 } ( \mathrm { d } x ) \right) . } \end{array}
$$

The second term can be estimated analogously to Lemma 4.10, which gives

$$
I _ { 2 } = \| \varphi _ { t } ^ { v } - \varphi _ { t } ^ { \star } \| _ { L ^ { 1 } ( \rho _ { 0 } ) } \leq \| \varphi _ { t } ^ { v } - \varphi _ { t } ^ { \star } \| _ { L ^ { 2 } ( \rho _ { 0 } ) } \leq \sqrt { 2 } e ^ { \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \operatorname { L i p } } ^ { 1 } } } E ( v _ { \bullet } ) ^ { \frac { 1 } { 2 } } .\tag{6.2}
$$

The first term can be estimated according to

$$
\| \psi _ { t } ( x ) \| \leq \int _ { 0 } ^ { t } \| D v _ { s } ( \varphi _ { t } ^ { v } ( x ) ) \cdot \psi _ { s } ^ { u } ( x ) + u _ { s } ( \psi _ { s } ^ { v } ( x ) ) \| \mathrm { d } s .
$$

Grönwall’s inequality yields

$$
\begin{array} { r l } & { \| \psi _ { t } ( x ) \| \leq \exp \left( \displaystyle \int _ { 0 } ^ { t } \| D v _ { r } \| _ { \infty } \mathrm { d } r \right) \displaystyle \int _ { 0 } ^ { t } \| u _ { s } ( \varphi _ { s } ^ { v } ( x ) ) \| \mathrm { d } s } \\ & { \qquad \leq e ^ { \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } \displaystyle \int _ { 0 } ^ { t } \| u _ { s } ( \varphi _ { s } ^ { v } ( x ) ) \| \mathrm { d } s . } \end{array}
$$

The Flat $L ^ { 2 } ( \rho _ { \bullet } ^ { \star } )$ -Geometry

Integration with respect to $\rho _ { 0 }$ yields

$$
\begin{array} { r l } & { I _ { 1 } \le e ^ { \| v _ { * } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } \displaystyle \int _ { 0 } ^ { t } \int _ { \mathbb { R } ^ { d } } \| u _ { s } ( \varphi _ { s } ^ { v } ( x ) ) \| \rho _ { 0 } ( \mathrm { d } x ) \mathrm { d } s . } \\ & { \quad \le e ^ { \| v _ { * } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } \displaystyle \int _ { 0 } ^ { t } \int _ { \mathbb { R } ^ { d } } \| u _ { s } ( x ) \| \rho _ { s } ^ { v } ( \mathrm { d } x ) \mathrm { d } s } \\ & { \quad \le e ^ { \| v _ { * } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } \displaystyle \int _ { 0 } ^ { t } \int _ { \mathbb { R } ^ { d } } \| u _ { s } ( x ) \| \rho _ { s } ^ { \star } ( \mathrm { d } x ) \mathrm { d } s + e ^ { \| v _ { * } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } \displaystyle \int _ { 0 } ^ { t } [ u _ { s } ] _ { \mathrm { L i p } } W _ { 1 } ( \rho _ { s } ^ { v } , \rho _ { s } ^ { \star } ) \mathrm { d } s } \end{array}
$$

The first term can be estimated as follows

$$
\begin{array} { r l r } {  { \int _ { 0 } ^ { t } \int _ { \mathbb { R } ^ { d } } \| u _ { s } \| \rho _ { s } ^ { \star } ( \mathrm { d } x ) \mathrm { d } s \leq \int _ { 0 } ^ { t } ( \int _ { \mathbb { R } ^ { d } } \| u _ { s } \| ^ { 2 } \rho _ { s } ^ { \star } ( \mathrm { d } x ) ) ^ { \frac { 1 } { 2 } } \mathrm { d } s } } \\ & { } & { \leq \sqrt { t } ( \int _ { 0 } ^ { t } \int _ { \mathbb { R } ^ { d } } \| v _ { s } - v _ { s } ^ { \star } \| ^ { 2 } \rho _ { s } ^ { \star } ( \mathrm { d } x ) \mathrm { d } s ) ^ { \frac { 1 } { 2 } } } \\ & { } & { \leq \sqrt { 2 } E ( v _ { \bullet } ) ^ { \frac { 1 } { 2 } } . } \end{array}
$$

To estimate the second component, we bound the Wasserstein distance according to

$$
W _ { 1 } ( \rho _ { s } ^ { v } , \rho _ { s } ^ { \star } ) \leq \| \varphi _ { s } ^ { v } - \varphi _ { s } ^ { \star } \| _ { L ^ { 1 } ( \rho _ { 0 } ) } \leq \sqrt { 2 } e ^ { \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } E ( v _ { \bullet } ) ^ { \frac { 1 } { 2 } } ,
$$

where we used (6.2). This implies

$$
\begin{array} { r l } { \displaystyle \int _ { 0 } ^ { t } [ u _ { s } ] _ { \mathrm { L i p } } W _ { 1 } ( \rho _ { s } ^ { v } , \rho _ { s } ^ { \star } ) \mathrm { d } s \cdot \leq \sqrt { 2 } \displaystyle \int _ { 0 } ^ { t } [ u _ { s } ] _ { \mathrm { L i p } } \mathrm { d } s e ^ { \mathrm { \ i } \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } E ( v _ { \bullet } ) ^ { \frac { 1 } { 2 } } } & { } \\ { \leq \sqrt { 2 } \| v _ { \bullet } - v _ { \bullet } ^ { \star } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } e ^ { \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } E ( v _ { \bullet } ) ^ { \frac { 1 } { 2 } } } & { } \\ { \leq \sqrt { 2 } \left( \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } + \| v _ { \bullet } ^ { \star } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } \right) e ^ { \| v _ { \bullet } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } E ( v _ { \bullet } ) ^ { \frac { 1 } { 2 } } } & { } \end{array}
$$

Combining the bounds yields

$$
I _ { 1 } \leq \sqrt 2 \left( e ^ { \| v _ { * } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } + \left( \| v _ { * } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } + \| v _ { * } ^ { \star } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } \right) e ^ { 2 \| v _ { * } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } \right) E ( v _ { * } ) ^ { \frac { 1 } { 2 } } .
$$

Overall, we obtain

$$
| \langle \xi _ { t } ^ { u } - ( \rho _ { t } ^ { v } - \rho _ { t } ^ { \star } , \phi \rangle | \leq \| \nabla \phi \| _ { \infty } ( I _ { 1 } + I _ { 2 } ) \leq c \| \nabla \phi \| _ { \infty } E ( v _ { \bullet } ) ^ { \frac { 1 } { 2 } } ,
$$

where the constant $c > 0$ is given by

$$
c = \sqrt { 2 } \left( 2 e ^ { \| v \bullet \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } + \left( \| v \bullet \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } + \| v _ { \bullet } ^ { \star } \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } \right) e ^ { 2 \| v \bullet \| _ { L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 } } } \right) .
$$

Theorem $6 . 2$ ensures that the update direction of the $L ^ { 2 }$ -gradient is close to the mixture geodesic if the energy $E ( v _ { \bullet } )$ is small, which will typically be the case late in training. Note that the norm $\| v _ { \bullet } \| _ { L _ { T } ^ { 2 } C _ { \mathrm { L i p } } ^ { 1 } }$ is not controlled by the energy $E ( v _ { \bullet } )$ However, it can be controlled by the Lipschitz constant of the network function. For example, it can be controlled by means of the magnitude of the weights of the network, which is commonly done in practice.

## 7 Implications for Flow-Based Generative Models

In generative modeling, it is the goal to design a sampling procedure that approximately samples from a target distribution $p _ { \mathrm { d a t a } }$ , which is unknown but can be accessed via samples. Most modern approaches first sample $X _ { 0 } \sim p _ { 0 }$ from an easy distribution $p _ { 0 }$ like a Gaussian or uniform distribution and then apply a transformation $X _ { 1 } = \Phi ( X _ { 0 } )$ . Then, it is the task to choose $\Phi$ such that $X _ { 1 }$ is approximately distributed according to $p _ { \mathrm { d a t a } }$ , hence such that $\Phi _ { \sharp } p _ { 0 } \approx p _ { \mathrm { d a t a } }$ . Earlier approaches like variational auto-encoders (VAEs) [KW14], generative adversarial networks (GANs) $[ \mathrm { G P A M ^ { + } 1 4 } ]$ directly parameterize the transformation $\Phi ^ { \theta }$ and optimize the models parameters to minimize $D ( \Phi _ { \sharp } ^ { \theta } p _ { 0 } , p _ { \mathrm { d a t a } } )$ , where D is some distance measure. Recently, a new paradigm has become very popular, where the samples $X _ { 0 }$ from $p _ { 0 }$ are transformed successively into synthetic data $X _ { 1 }$ . Most notably, this paradigm shift has led to the development of many successful models, including normalizing flows $[ \mathrm { D K B 1 5 } ] .$ , difusion models [SDWMG15, HJA20, RBL+22], score and flow matching $[ \mathrm { L C B H ^ { + } } _ { 2 3 } ]$ , and Schrödinger bridges [DBTHD21]. In flow matching, one considers the flow $\varphi _ { t }$ associated to a velocity field $v _ { t }$ given by

$$
\partial _ { t } \varphi _ { t } ( x ) = v _ { t } ( \varphi _ { t } ( x ) ) .
$$

Then, synthetic data is produced by sampling $X _ { 0 } \sim p _ { 0 }$ and transforming the data along the flow until the time $t = 1$ . Hence, one considers $X _ { 1 } = \varphi _ { 1 } ( X _ { 0 } )$ and one hopes to approximate $( \varphi _ { 1 } ) _ { \sharp } p _ { 0 } \approx p _ { \mathrm { d a t a } }$ . The velocity $v _ { \bullet }$ is then parametrized by a neural network $v _ { \bullet } ^ { \theta } \approx v _ { \bullet }$ and the parameters $\theta$ are optimized to minimize the objective

$$
L _ { \mathrm { F M } } ( \theta ) = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| v _ { t } ^ { \theta } ( x ) - v _ { t } ^ { \star } ( x ) \| ^ { 2 } p _ { t } ^ { \star } ( x ) \mathrm { d } x \mathrm { d } t .\tag{7.1}
$$

Here, a reference velocity field $v _ { \bullet } ^ { \star }$ is constructed to perfectly match the data, meaning that $( \varphi _ { 1 } ^ { \star } ) _ { \sharp } p _ { 0 } = p _ { \mathrm { d a t a } }$ , where $\varphi _ { \bullet } ^ { \star }$ denotes the flow induced by $v _ { \bullet } ^ { \star }$ . Various ways to construct such a reference velocity field have been proposed in the literature, such as via conditional interpolation and optimal transport $[ \mathrm { L C B H ^ { + } } _ { 2 3 } ]$ , via the Schrödinger bridge problem $\mathrm { [ D B T H D } _ { 2 1 } ]$ . For more details on such constructions, we refer to the general frameworks for stochastic interpolants [ABVE25, $\mathrm { W S } 2 5 ]$ and the overview articles [LHH+24, HE25, PTDN26]. Note that typically, the velocity field $v _ { \bullet } ^ { \star }$ is not accessible directly, but rather the conditioned version on the initial data $v _ { t } ^ { \star } ( x | X _ { 0 } )$ Similar to flow matching, difusion models aim to approximate a reference velocity of a—possibly stochastic—diferential equation, thereby approximately transporting $p _ { 0 }$ to the data distribution $p _ { \mathrm { d a t a } }$ . They use a forward-backward process where first $p _ { \mathrm { d a t a } }$ is difused by a forward dynamic

$$
\mathrm { d } X _ { t } ^ { \star } = f _ { t } ( X _ { t } ^ { \star } ) \mathrm { d } t + \mathrm { d } W _ { t }
$$

leading to densities $p _ { t } ^ { \star } = \mathcal L ( X _ { t } ^ { \star } )$ , where $f _ { \bullet }$ is a drift term chosen by the practitioner. At some time $T > 0$ , the dynamics are assumed to have approximately equilibrated

to $p _ { T } ^ { \star }$ ≈ $p _ { 0 }$ . It is common to approximate the score function $s _ { t } ^ { \star } = \nabla$ ln $p _ { T - t } ^ { \star }$ with a neural network $s _ { \bullet } ^ { \theta }$ by minimizing the objective

$$
L _ { \mathrm { S M } } ( \theta ) = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| s _ { t } ^ { \theta } ( x ) - s _ { t } ^ { \star } ( x ) \| ^ { 2 } p _ { t } ^ { \star } ( x ) \mathrm { d } x \mathrm { d } t .
$$

The data is then either generated by simulating the probability flow ODE

$$
\dot { x } _ { t } = f _ { t } ( x _ { t } ) - \frac { 1 } { 2 } s _ { t } ^ { \theta } ( x _ { t } )
$$

or the reverse-time SDE

$$
\mathrm { d } X _ { t } = \left( f _ { t } ( X _ { t } ) - \frac { 1 } { 2 } s _ { t } ^ { \theta } ( X _ { t } ) \right) \mathrm { d } t + \mathrm { d } { \tilde { W } } _ { t } ,
$$

which both induce the same marginal distributions as the forward SDE $[ \mathrm { L S K ^ { + } } _ { 2 5 } ]$ A second important paradigm shift, next to the parametrization of the instantaneous change of the data that led to the empirical success of flow-based generative models, lies in the choice of the loss function. Earlier flow-based generative models like normalizing flows use $D ( ( \varphi _ { 1 } ^ { \star } ) _ { \sharp } p _ { 0 } , p _ { \mathrm { d a t a } } )$ , which requires both the simulation of the forward dynamics as well as the diferentiation through the numerical solver of the forward dynamics. In contrast, flow matching and score matching objectives operate on the level of velocity fields. Further, it measures the discrepancy of the velocity fields in terms of the reference densities $p _ { t } ^ { \star }$ , which avoids the costly simulation of the forward dynamics induced by $v _ { \bullet } ^ { \theta } .$ . For this reason, flow matching and score matching are often referred to as simulation $- f r e e$ . The choice of the loss function as a square-based objective has great computational advantages, which have arguably led to the success of flow-based generative models. However, it comes at the cost of introducing a proxy for the eventual goal of matching the data distribution at the terminal time. Indeed, the standard Flow Matching loss acts on an Eulerian level of velocity fields, where the generative performance depends on the densities induced by the trajectories. At the level of velocities, the least-squares loss is simply a quadratic function. However, it is inherently connected to relative entropy at the level of path measures, a concept that is well known in stochastic optimal control theory $[ \mathrm { F o } 8 5 ]$ . This already suggests the existence of a non-trivial geometry underlying the optimization of flow-based generative models, which is not visible at the level of velocity fields. In this work, we reveal the geometry induced by the least-squares objectives of flow-based generative models.

## 7.1 Computational considerations

When training a flow-matching model, we do not have access to the reference velocity field $v _ { \bullet } ^ { \star }$ ; hence, the flow-matching loss

$$
L _ { \mathrm { F M } } ( \theta ) = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \lVert v _ { t } ^ { \theta } ( x ) - v _ { t } ^ { \star } ( x ) \rVert _ { 2 } ^ { 2 } p _ { t } ^ { \star } ( x ) \mathrm { d } x \mathrm { d } t
$$

is not directly accessible. Therefore, it is common to use a conditional flow-matching loss, which is given by

$$
L _ { \mathrm { C F M } } ( \theta ) = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \int _ { \mathbb { R } ^ { d } } \Big \| v _ { t } ^ { \theta } ( x ) - v _ { t } ^ { \star } ( x | x _ { 0 } ) \Big \| ^ { 2 } p _ { t } ( x | x _ { 0 } ) \mathrm { d } x \mathrm { d } x _ { 0 } \mathrm { d } t ,
$$

where $v _ { t } ^ { \star } ( x | x _ { 0 } )$ is a conditional reference velocity field. One of the most common choices for the conditional reference velocity field is given by the velocity field obtained through sample-wise interpolation, which is given by

$$
v _ { t } ^ { \star } ( x | x _ { 0 } ) = \frac { x - x _ { 0 } } { 1 - t } .
$$

We refer to this approach as conditional linear interpolation, where a variety of other methods and frameworks, which use couplings of the source and target distributions, Markov kernels, and stochastic processes, have been suggested $[ \mathrm { L C B H ^ { + } } _ { 2 3 } , \mathrm { A B V E } _ { 2 5 } .$ WS25]. It is well-known that the conditional flow-matching loss difers from the unconditional flow-matching loss only by a constant; hence, it can be used as a surrogate loss for training, as $\nabla L _ { \mathrm { F M } } = \nabla L _ { \mathrm { C F M } }$ . Note that samples from the reference density $p _ { t } ^ { \star }$ are given by conditional interpolation; then we can sample from $p _ { t } ^ { \star }$ by sampling $X _ { 0 }$ and $X _ { 1 }$ from $p _ { 0 }$ and $p _ { 1 }$ , respectively, and setting

$$
X _ { t } = ( 1 - t ) X _ { 0 } + t X _ { 1 } .
$$

The natural gradient method In the context of flow-based generative models, the Fisher-information matrix is given by

$$
F ( \theta ) _ { i j } = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \partial _ { \theta _ { i } } v _ { t } ^ { \theta } ( x ) \cdot \partial _ { \theta _ { j } } v _ { t } ^ { \theta } ( x ) p _ { t } ^ { \theta } ( x ) \mathrm { d } x \mathrm { d } t .\tag{7.2}
$$

Note again the diference from the Gauss–Newton matrix, where the reference density $p _ { \bullet } ^ { \star }$ is used instead of the current density $p _ { \bullet } ^ { \theta }$ . This means that, diferent from the Gauss–Newton method, the natural gradient method requires the simulation of the forward dynamics to compute the current density $p _ { \bullet } ^ { \theta } .$ . The natural gradient method is given by the iteration

$$
\boldsymbol { \theta } _ { k + 1 } = \boldsymbol { \theta } _ { k } - \boldsymbol { \eta } \cdot \boldsymbol { F } ( \boldsymbol { \theta } _ { k } ) ^ { + } \nabla L ( \boldsymbol { \theta } _ { k } ) .\tag{7.3}
$$

The Gauss–Newton method The $L ^ { 2 }$ -geometry of velocity fields leads to the Gauss–Newton method for optimization. It is given by the iteration

$$
\boldsymbol { \theta } _ { k + 1 } = \boldsymbol { \theta } _ { k } - \boldsymbol { \eta } \cdot \boldsymbol { G } ( \boldsymbol { \theta } _ { k } ) ^ { + } \nabla L ( \boldsymbol { \theta } _ { k } ) ,
$$

where the Gauss–Newton matrix is given by

$$
G ( \theta ) _ { i j } = ( \partial _ { \theta _ { i } } v _ { \bullet } ^ { \theta } , \partial _ { \theta _ { j } } v _ { \bullet } ^ { \theta } ) _ { L ^ { 2 } ( p _ { \bullet } ^ { \star } ) } = \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \partial _ { \theta _ { i } } v _ { t } ^ { \theta } ( x ) \cdot \partial _ { \theta _ { j } } v _ { t } ^ { \theta } ( x ) p _ { t } ^ { \star } ( x ) \mathrm { d } x \mathrm { d } t .\tag{7.4}
$$

Estimation of preconditioning matrices Recall the definition of the Gramian matrices $F ( \theta )$ and $G ( \theta )$ in $( 7 . 2 )$ and $( 7 . 4 )$ , respectively. In practice, we need to estimate these matrices from samples. To estimate the Gauss–Newton matrix $G ( \theta )$ we can use the samples $( t _ { k } ^ { \star } , x _ { k } ^ { \star } )$ used for the discretization of the loss, where $t _ { k } ^ { \star }$ is sampled uniformly from [0, 1] and $\ v x _ { k } ^ { \star }$ is sampled from $p _ { t _ { k } ^ { \star } } ^ { \star }$ . Then, we obtain the empirical Gauss–Newton matrix

$$
\hat { G } ( \theta ) _ { i j } = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \partial _ { \theta _ { i } } v _ { t _ { k } ^ { \star } } ^ { \theta } ( x _ { k } ^ { \star } ) \cdot \partial _ { \theta _ { j } } v _ { t _ { k } ^ { \star } } ^ { \theta } ( x _ { k } ^ { \star } ) .
$$

As the Fisher-information matrix uses the current density $p _ { t } ^ { \theta }$ instead of the reference density $p _ { t } ^ { \star }$ , we need to sample from $p _ { t } ^ { \theta }$ to obtain an empirical estimate. To this end, we simulate the forward dynamics of the model to obtain samples $( t _ { k } , x _ { k } )$ , where $t _ { k }$ is sampled uniformly in time and $x _ { k }$ is sampled from $p _ { t } ^ { \theta }$ . Then, we can obtain an empirical estimate of the Fisher information matrix by

$$
\hat { F } ( \theta ) _ { i j } = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \partial _ { \theta _ { i } } v _ { t _ { k } } ^ { \theta } ( x _ { k } ) \cdot \partial _ { \theta _ { j } } v _ { t _ { k } } ^ { \theta } ( x _ { k } ) .
$$

Therefore, the natural gradient method is not simulation-free and has an additional overhead compared to the Gauss–Newton method, which is simulation-free.

Linear algebra For the actual computation, we use standard implementation procedures for preconditioned and natural gradient systems as laid out in [Mar20, $\mathrm { G C D G Z } 2 5 ]$ . Both $G ( \theta )$ and $F ( \theta )$ are positive semi-definite Gram matrices, so their pseudo-inverses can be computed via the Cholesky decomposition. To ensure numerical stability, we apply damping, also known as Tikhonov regularization, which gives

$$
\begin{array} { c } { \left( G ( \theta ) + \lambda I \right) \delta \theta _ { \mathbf { G N } } = \nabla L ( \theta ) , } \\ { \left( F ( \theta ) + \lambda I \right) \delta \theta _ { \mathbf { N G } } = \nabla L ( \theta ) , } \end{array}
$$

where $\lambda > 0$ is a small regularization parameter. The Gauss–Newton Gramian $G ( \theta )$ uses the reference density $p _ { t } ^ { \star }$ , whereas the Fisher information $F ( \theta )$ uses the current density $p _ { t } ^ { \theta } \mathrm { : }$ ; the gradient $\nabla L ( \theta )$ is the same in both cases. Empirically, with $N$ samples, $\begin{array} { r } { \dot { \hat { G } } = \frac { 1 } { N } J _ { \star } ^ { \top } J _ { \star } , \hat { F } = \frac { 1 } { N } J _ { \theta } ^ { \top } J _ { \theta } } \end{array}$ , and $\begin{array} { r } { \nabla L = \frac { 1 } { N } J _ { \star } ^ { \top } r } \end{array}$ , where $J _ { \star }$ and $J _ { \theta }$ are the Jacobians of $v _ { t } ^ { \theta }$ at samples from $p _ { t } ^ { \star }$ and $p _ { t } ^ { \theta }$ , respectively, and $r$ is the residual $v _ { t } ^ { \theta } - v _ { t } ^ { \star }$ . For small parameter dimension $p _ { : }$ , we factorize the damped Gramian via Cholesky and solve directly at $O ( p ^ { 3 } )$ cost. For GN, where both the Gramian and the gradient involve the same Jacobian $J _ { \star }$ , the push-through identity

$$
( J ^ { \top } J + \mu I ) ^ { - 1 } J ^ { \top } = J ^ { \top } ( J J ^ { \top } + \mu I ) ^ { - 1 }
$$

reduces the $p \times p$ system to an $( N d ) \times ( N d )$ system:

$$
\delta \theta _ { \mathbf { G } \mathbf { N } } = J _ { \star } ^ { \top } ( J _ { \star } J _ { \star } ^ { \top } + N \lambda I ) ^ { - 1 } r .
$$

For NG, the Gramian uses $J _ { \theta }$ while the gradient uses $J _ { \star }$ , so the push-through identity does not apply. In this case, we either solve the damped $p \times p$ system directly (Cholesky, $O ( p ^ { 3 } ) )$ when $p$ is small, or apply the Woodbury identity to the Gramian factor alone

$$
\delta \theta _ { \mathrm { N G } } = \frac { 1 } { N \lambda } [ I - J _ { \theta } ^ { \top } ( N \lambda I + J _ { \theta } J _ { \theta } ^ { \top } ) ^ { - 1 } J _ { \theta } ] J _ { \star } ^ { \top } r .
$$

When both $p$ and d are large, we avoid forming the Gramian or the Jacobian explicitly and use conjugate gradient to solve the damped system iteratively via the conjugate gradient method, which only requires Jacobian–vector products, see [Mar20].

## 7.2 Visualization of update directions

Note that the $L ^ { 2 }$ -gradient points directly into the direction of the optimal velocity field. Hence, by the general projection property (2.8) of natural gradients, the Gauss–Newton method leads to an optimal fitting of the velocity fields. We can visualize this in an easy case of matching two Gaussians in one dimension. As a source distribution $p _ { 0 } .$ , we consider a standard Gaussian $\mathcal { N } ( 0 , 1 )$ and as a target distribution $p _ { \mathrm { d a t a } }$ we consider $\mathcal { N } ( 1 , 1 )$ . One great benefit of this simple setting is that we have access to a closed-form expression of the reference velocity field $v ^ { \star }$ obtained through conditional interpolation. First, the reference density $p _ { t } ^ { \star }$ obtained by sample-wise interpolation is the Gaussian $\mathcal { N } ( \mu _ { t } ^ { \star } , \sigma _ { t } ^ { \star } )$ , where

$$
\begin{array} { c } { \mu _ { t } ^ { \star } = ( 1 - t ) \mu _ { 0 } ^ { \star } + t \mu _ { 1 } ^ { \star } , } \\ { ( \sigma _ { t } ^ { \star } ) ^ { 2 } = ( 1 - t ) ^ { 2 } \sigma _ { 0 } ^ { 2 } + t ^ { 2 } \sigma _ { 1 } ^ { 2 } . } \end{array}
$$

Further, the marginal vector field is given by

$$
v _ { t } ^ { \star } ( x ) = ( \mu _ { 1 } - \mu _ { 0 } ) + \frac { t \sigma _ { 1 } ^ { 2 } - ( 1 - t ) \sigma _ { 0 } ^ { 2 } } { ( 1 - t ) ^ { 2 } \sigma _ { 0 } ^ { 2 } + t ^ { 2 } \sigma _ { 1 } ^ { 2 } } \cdot ( x - \mu _ { t } ^ { \star } ) ,
$$

see [BGME25]. To visualize the update directions of the two optimizers, we consider a feedforward, fully connected neural network $v _ { \bullet } ^ { \theta }$ with 2 hidden layers of width 32 and hyperbolic tangent as an activation function. Here, we provide the time as an explicit input to the network, which is a common choice in the literature. We initialize the weights of the network according to a normal distribution and the biases to zero and visualize the $L ^ { 2 } .$ -gradient, which agrees with the diference to the reference velocity field $v _ { \bullet } ^ { \theta } - v _ { \bullet } ^ { \star }$ , see the left plot in Figure 1. We compare this to the pushforward of the Gauss–Newton update direction, which is given by $D P ( \boldsymbol { \theta } ) G ( \boldsymbol { \theta } ) ^ { + } \nabla L ( \boldsymbol { \theta } )$ , see the right plot in Figure 1. The Gauss–Newton method leads to an optimal fitting of the velocity fields, but it is a method that is agnostic of the actual goal of fitting the probability densities induced by the corresponding generative model. This motivates the construction of an optimizer based on the geometry of probability densities. This is contrasted with the natural gradient method. Here, the gradient in the space of velocities is weighted by the density ratio. We compare the push forward $D P ( \theta ) \delta \theta _ { \mathrm { N G } }$ of the natural gradient direction $\delta \theta _ { \mathrm { N G } }$ to this weighted diference, see Figure 1.

![](images/80f1c53d278b6213db52e9f9920efeb4a2960436a14f6700cbb1d16f2c5dde86.jpg)

![](images/2019f43c05fe481d98c594d1d253208b76b9a0bd83cb97b6207b1e11f8c3439c.jpg)

$$
\begin{array} { r } { ( v _ { \bullet } ^ { \theta } - v _ { \bullet } ^ { \star } ) \cdot \frac { p _ { \bullet } ^ { \star } } { p _ { \bullet } ^ { \theta } } } \end{array}
$$

![](images/4c68043fe796edfa9514910431c17be78dd0f5736049c497349e383823a65333.jpg)

![](images/8bb80f9d064f99018d33b19e209010e80906c3bc31b4fef71f5374e25b2204f8.jpg)

$$
p _ { \bullet } ^ { \theta } - p _ { \bullet } ^ { \star }
$$

![](images/581c957f6181a0a0721aad2095a98f038321ed8471c97f0836a60ffe031d724a.jpg)

![](images/ad4f1b6face6bd0ec49f86f78a06a2b4ea38ab5b1d028f38bbb54d2f6fef74c8.jpg)

![](images/64ac9ff01e5eb7a41b38cdc4bc3e2d91262c99925d80b6dba6163f2926692cd0.jpg)  
Figure 1: Shown are the gradient with respect to the $L ^ { 2 }$ metric $v _ { \bullet } - v _ { \bullet } ^ { \star }$ (top left) and and continuity Fisher–Rao metric $( v _ { \bullet } - v _ { \bullet } ^ { \star } ) \cdot \frac { p _ { \bullet } ^ { \star } } { p _ { \bullet } ^ { v } }$ as well as the pushforward corresponding to the update directions of the Gauss–Newton method (top right) and natural gradient method (bottom right); from (2.8) we expect the pushforwards to agree with the gradients up to a projection; in the bottom row, the residual $p _ { \bullet } - p _ { \bullet } ^ { \star }$ in the space of paths of probability densities (left panel) as well as the pushforwards corresponding to the update directions of the Gauss–Newton and the natural gradient method (middle and right panels) are shown; from Equation (2.8) we expect the pushforward of the natural gradient to agree with the residual up to a projection.

## 7.3 Computational Experiments

Here, we conduct a series of computational experiments to compare the performance of the natural gradient and the Gauss–Newton method and the standard first-order optimizers like stochastic gradient descent (SGD) and Adam. The code for these experiments will be made available at the time of publication. We use a standard Gaussian as a source distribution and compare the optimizers on the following target distributions, which are:

1. a 2D Gaussian mixture with two standard uniform modes

2. a 2D Gaussian mixture with a standard uniform and an ill-conditioned mode

3. a half-moon distribution

4. a Gaussian mixture model with eight components

5. and a checkerboard distribution.

For every experiment, we report the learning curves of the diferent optimizers in terms of the conditional flow-matching loss $L _ { \mathrm { C F M } }$ as well as the $W _ { 2 }$ distance to the target distribution, which we estimate from samples. Further, we provide tables of the final accuracy of the optimizers when given the same computational budget.

Discussion Across all five target distributions, the second-order methods Gauss– Newton and natural gradient consistently achieve lower $W _ { 2 }$ distances than the first-order baselines. For all experiments, the training curves for a fixed number of iterations as well as samples produced after training are shown in Figures 2 to 6. Here, multiple initializations are used, and the median and 10% and 90% percentiles are reported in the learning curves. Finally, smoothed medians at the end of training are reported for all experiments in Table 1. We consistently observe that the second-order optimizers in the form of the Gauss–Newton and the natural gradient method provide accuracy surpassing that of first-order methods even when given the same computational budget.

![](images/f6ff656a65c4aaa92dcf36989d8aed55225ab4fab956d8fd0a3ea704f08a7f42.jpg)

![](images/be294b09be6c0d2549814af1e58c860c9ef262297dfbaf8c503791c08afdea66.jpg)

SGD  
![](images/d102c1bd99fac24d27d8517ba63b972bcdc035333eadfc17562fcbf83d1b8086.jpg)

Adam  
![](images/3b9e16baccbec9a14f3d6993047b9be95a92195ac3afe1e303dcb07226b7deba.jpg)

Gauss-Newton  
![](images/df28bbe895760c0268c708a629586f653b25202baa1826d3cceb4ec955bd3602.jpg)

Natural gradient  
![](images/3d4a91ceab2044beeaff977650b22f7b739c79ddfc6b709e76d9b85a8f62584e.jpg)  
Figure 2: The top row shows the loss $L _ { \mathrm { C F M } }$ (left) and the $W _ { 2 }$ distance to the target distribution (right) for the diferent optimization methods; the curves show the median over 7 runs, while the shaded areas show the region between the 10% and $9 0 \%$ percentiles; note that the conditional flow matching loss difers from the pure regression loss by a constant; hence, it cannot be expected to vanish; the second and third rows show scatter plots of generated samples for each optimizer; each panel overlays generated samples (blue) with the target distribution (gold) for visual comparison.

![](images/cff6c00331830ba77aa6407a043d49d55a3574fd6017e4a5b3b4ec57cb5f34bc.jpg)

![](images/c726f8bdb5a421e46f7c58fd693515de83e3faa6a4acc540e8a6334bd418f5ba.jpg)

![](images/57be525c5057c82ebbb483190296684c1948c5308f4ed23e9e24787be8bef072.jpg)

![](images/b9c1805d5b40ea39ef521d98b67cd0152771e9dfead7985c69d272f2d1e8ab52.jpg)

![](images/140f557dbb5f98ff098cef11f7dcd2b1331ea3d6e2673d954ba93b239a5fadf4.jpg)

![](images/36de37081524f455d242e7df3b447cee755e0e2960929f5d7a68f8cfcdd146e5.jpg)

![](images/0939133215521a228d3674e1526939a20e35d61e3c39910fc99d2f62e83559b6.jpg)  
Figure 3: Ill-conditioned GMM: The top row shows the loss $L _ { \mathrm { C F M } }$ (left) and the $W _ { 2 }$ distance to the target distribution (right) for the diferent optimization methods; the curves show the median over 7 runs, while the shaded areas show the region between the 10% and 90% percentiles; note that the conditional flow matching loss difers from the pure regression loss by a constant; hence, it cannot be expected to vanish; the second and third rows show scatter plots of generated samples for each optimizer; each panel overlays generated samples (blue) with the target distribution (gold) for visual comparison.

![](images/eee8b5b5235138333b004abec3b22c4137eb135814eb53512f24a924d3105776.jpg)

![](images/337a8b55707998bda5b17d87b114cc3666a89a4446c4ebbb1b119e1a48224c84.jpg)

![](images/b866c9a1368b275d4b3d565d1e5ef1df0d520355404533184f21d68609af747d.jpg)

![](images/2790e658069deca6e9675f842d9cc3a0b1ae897761c5b1c50e6ddca8bc51af2c.jpg)

![](images/6375411240354a2b92b696783c0fc04fce20b46d30e0bf5aa056c479b1510f41.jpg)

![](images/c0903fe70d4cb259fb7e67e5e6680dacae248352e03223cdee11fd32b4716fb1.jpg)

![](images/3a98549b0cc9a78caa8088e7226d10022294aacb8ec1c2a565f2560a284b2402.jpg)  
Figure 4: Half moons: The top row shows the loss $L _ { \mathrm { C F M } }$ (left) and the $W _ { 2 }$ distance to the target distribution (right) for the diferent optimization methods; the curves show the median over 7 runs, while the shaded areas show the region between the 10% and 90% percentiles; note that the conditional flow matching loss difers from the pure regression loss by a constant; hence, it cannot be expected to vanish; the second and third rows show scatter plots of generated samples for each optimizer; each panel overlays generated samples (blue) with the target distribution (gold) for visual comparison.

![](images/b89011b53df4b26ef7ba66c89a27117af72a724d4236000772c5ff3fd0539430.jpg)

![](images/527ed6484cf9fcbe021b3037821c8e57135a6efa34237e6c7e7189dde64a0e75.jpg)

![](images/11f0c9f86430534de7af5bf822faa55813e50343b3d19dabe096724cb3cda936.jpg)

![](images/bedd18ab8c5da5ed57fa4553ba3e5b3f31c744a737eccd571f6c80682a7d5090.jpg)

![](images/da9f084a08af8e8db6ecf9b080cdb6a7f938eefcab602a2f53d7b7c2dcde03df.jpg)

![](images/aadf74e6f406c2045ffcf1edbe8e19a394c1da26a78fecb289d4794404ce5a1b.jpg)  
Figure 5: Eight-GMM experiment: The top row shows the loss $L _ { \mathrm { C F M } }$ (left) and the $W _ { 2 }$ distance to the target distribution (right) for the diferent optimization methods; the curves show the median over 7 runs, while the shaded areas show the region between the $1 0 \%$ and $9 0 \%$ percentiles; note that the conditional flow matching loss difers from the pure regression loss by a constant; hence, it cannot be expected to vanish; the second and third rows show scatter plots of generated samples for each optimizer; each panel overlays generated samples (blue) with the target distribution (gold) for visual comparison.

![](images/299bb635650e2f9be5fa846b936a6da78d3fda8dafe6408150379c969f9e53c4.jpg)

![](images/5d2a3176b70714bd6cac903da5530b65b17f1c96243a713ca933eac97adfa0fd.jpg)

SGD  
![](images/8574be6cc22b8fbf50ce3f812dde753f34032cb37b0969c8233a61f649b7d58d.jpg)

Adam  
![](images/8b470c8911e60d757f0fa84924a51ac83c0a24270eb3e1e4c8cfd0f347cd3620.jpg)

![](images/8f2067e0f275fb27c5ee047420dbab8d8f08606b906653f941f4a3935e49c671.jpg)

![](images/20307235a78da1f9de7099500280b02adc67ebce90d8010b922739ba5aa03a61.jpg)  
Figure 6: Checkerboard experiment: Top row shows the loss $L _ { \mathrm { C F M } }$ (left) and the $W _ { 2 }$ distance to the target distribution (right) for the diferent optimization methods; the curves show the median over 7 runs, while the shaded areas show the region between the $1 0 \%$ and $9 0 \%$ percentiles; note that the conditional flow matching loss difers from the pure regression loss by a constant; hence, it cannot be expected to vanish; the second and third rows show scatter plots of generated samples for each optimizer; each panel overlays generated samples (blue) with the target distribution (gold) for visual comparison.

<table><tr><td rowspan="2">Problem</td><td rowspan="2">Metric</td><td colspan="6">Optimizer</td></tr><tr><td>SGD</td><td>Adam</td><td>GN</td><td>GN-LS</td><td>NG</td><td>NG-LS</td></tr><tr><td rowspan="2">Isotropic GMM</td><td> $W _ { 2 }$ </td><td> $\phantom { - } 0 . 4 8 1$ </td><td> $\mathbf { 0 . 4 } 8 2$ </td><td>0.273</td><td>0.351</td><td>0.167</td><td>0.403</td></tr><tr><td> $\mathrm { T V }$ </td><td>0.346</td><td>0.340</td><td>0.197</td><td>0.289</td><td>0.191</td><td>0.322</td></tr><tr><td rowspan="2">Anisotropic GMM</td><td> $W _ { 2 }$ </td><td>0.647</td><td>0.507</td><td>0.466</td><td>0.391</td><td>0.135</td><td>0.502</td></tr><tr><td> $\mathrm { T V }$ </td><td>0.589</td><td>0.465</td><td>0.182</td><td>0.443</td><td>0.207</td><td>0.542</td></tr><tr><td rowspan="2">8-GMM</td><td> $W _ { 2 }$ </td><td>0.597</td><td>0.261</td><td>0.309</td><td>0.202</td><td>0.268</td><td>0.240</td></tr><tr><td>TV</td><td>0.891</td><td>0.476</td><td>0.212</td><td>0.463</td><td>0.240</td><td>0.536</td></tr><tr><td rowspan="2">Checkerboard</td><td> $W _ { 2 }$ </td><td>0.352</td><td>0.272</td><td>0.321</td><td>0.206</td><td>0.305</td><td>0.271</td></tr><tr><td>TV</td><td>0.493</td><td>0.422</td><td>0.374</td><td>0.388</td><td> $\mathbf { 0 . 3 8 7 }$ </td><td>0.443</td></tr><tr><td rowspan="2">Moons 2D</td><td> $W _ { 2 }$ </td><td>0.178</td><td>0.141</td><td>0.078</td><td>0.155</td><td>0.083</td><td>0.165</td></tr><tr><td> $\mathrm { T V }$ </td><td>0.690</td><td>0.618</td><td>0.532</td><td>0.681</td><td>0.606</td><td>0.684</td></tr></table>

Table 1: Final $W _ { 2 }$ and TV distances for each optimizer across all problems after a fixed wall-clock budget; best values per metric per problem are shown in bold; LS indicates the use of a linesearch.

## Appendix A Carathéodory Theory and Method of Characteristics

We begin by recalling a classic well-posedness theorem.

Theorem A.1 (Carathéodory existence theorem). Consider a velocity field $v _ { \bullet }$ such that there arefunctions $\alpha , \beta \in L _ { T } ^ { 1 }$ satisfying

$$
\begin{array} { r } { \| v _ { t } ( 0 ) \| \leq \alpha _ { t } , \quad \| v _ { t } ( x ) - v _ { t } ( y ) \| \leq \beta _ { t } \| x - y \| \quad f o r { a l l } { t } \in [ 0 , 1 ] , x , y \in { \mathbb R } ^ { d } . } \end{array}
$$

Then the flow $\varphi _ { t } ^ { v }$ , given by

$$
\partial _ { t } \varphi _ { t } ^ { v } ( x ) = v _ { t } ( \varphi _ { t } ^ { v } ( x ) ) \quad a n d \varphi _ { 0 } ^ { v } ( x ) = x ,\tag{A.1}
$$

for all $x \in \mathbb { R } ^ { d } , t \in [ 0 , 1 ]$ exists and is unique. Further, we have

$$
\| \varphi _ { t } ( x ) - \varphi _ { t } ( y ) \| \leq \| x - y \| e ^ { \int _ { 0 } ^ { t } \beta _ { s } \mathrm { d } s }\tag{A.2}
$$

for all $x , y \in \mathbb { R } ^ { d } , t \in [ 0 , 1 ]$

The proof can be found in [AGS08, Lemma 8.1.4]. Additionally, the following bounds will be useful.

Lemma A.2. Consider the setting of the previous theorem. Then, the following bounds hold:

1. For all $x \in \mathbb { R } ^ { d } , t \in [ 0 , 1 ]$ it holds that

$$
\begin{array} { r } { \| \varphi _ { t } ( x ) - x \| \leq \| \alpha \| _ { L ^ { 1 } ( [ 0 , t ] ) } e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , t ] ) } } + \| x \| \left( e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , t ] ) } } - 1 \right) . } \end{array}\tag{A.3}
$$

2. For all $x \in \mathbb { R } ^ { d } , t \in [ 0 , 1 ]$ it holds that

$$
\begin{array} { r } { \| \varphi _ { t } ( x ) \| \leq \| \alpha \| _ { L ^ { 1 } ( [ 0 , t ] ) } e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , t ] ) } } + \| x \| \cdot e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , t ] ) } } . } \end{array}\tag{A.4}
$$

Proof. First, observe that for any $t \in [ 0 , 1 ]$ and $\boldsymbol { x } \in \mathbb { R } ^ { d }$ it holds that

$$
\| v _ { t } ( x ) \| \leq \| v _ { t } ( y ) - v _ { t } ( 0 ) \| + \| v _ { t } ( 0 ) \| \leq \alpha _ { t } + \beta _ { t } \| y \| .\tag{A.5}
$$

By the definition of the flow, we have $\begin{array} { r } { \varphi _ { t } ( x ) = x + \int _ { 0 } ^ { t } v _ { s } ( \varphi _ { s } ( x ) ) \mathrm { d } s } \end{array}$

2. We start by showing the second statement, for which we take the norm of the integral equation and obtain

$$
\begin{array} { l } { \displaystyle \| \varphi _ { t } ( x ) \| \leq \| x \| + \int _ { 0 } ^ { t } \| v _ { s } ( \varphi _ { s } ( x ) ) \| \mathrm { d } s } \\ { \leq \| x \| + \int _ { 0 } ^ { t } \left( \alpha _ { s } + \beta _ { s } \| \varphi _ { s } ( x ) \| \right) \mathrm { d } s } \\ { = \displaystyle \big ( \| x \| + \| \alpha \| _ { L ^ { 1 } ( [ 0 , t ] ) } \big ) + \int _ { 0 } ^ { t } \beta _ { s } \| \varphi _ { s } ( x ) \| \mathrm { d } s . } \end{array}
$$

Now Grönwall’s inequality yields

$$
\begin{array} { r l } & { \displaystyle \| \varphi _ { t } ( x ) \| \leq \left( \| x \| + \int _ { 0 } ^ { t } \alpha _ { s } d s \right) \exp \left( \int _ { 0 } ^ { t } \beta _ { s } \mathrm { d } s \right) } \\ & { \quad \quad \quad = \left( \| x \| + \| \alpha \| _ { L ^ { 1 } ( [ 0 , t ] ) } \right) e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , t ] ) } } } \\ & { \quad \quad \quad = \| \alpha \| _ { L ^ { 1 } ( [ 0 , t ] ) } e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , t ] ) } } + \| x \| \cdot e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , t ] ) } } . } \end{array}\tag{A.6}
$$

1. To prove the first statement, we estimate

$$
\begin{array} { l } { \displaystyle \| \varphi _ { t } ( x ) - x \| = \left\| \int _ { 0 } ^ { t } { v _ { s } ( \varphi _ { s } ( x ) ) \mathrm { d } s } \right\| } \\ { \displaystyle \qquad \leq \int _ { 0 } ^ { t } \| v _ { s } ( \varphi _ { s } ( x ) ) \| \mathrm { d } s } \\ { \displaystyle \qquad \leq \int _ { 0 } ^ { t } \left( \alpha _ { s } + \beta _ { s } \| \varphi _ { s } ( x ) \| \right) \mathrm { d } s . } \end{array}
$$

Substituting the bound (A.6) for $\| \varphi _ { s } ( x ) \|$ we get

$$
\| \varphi _ { t } ( x ) - x \| \leq \int _ { 0 } ^ { t } \alpha _ { s } \mathrm { d } s + \int _ { 0 } ^ { t } \beta _ { s } \left( \| \alpha \| _ { L ^ { 1 } ( [ 0 , s ] ) } e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , s ] ) } } + \| x \| e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , s ] ) } } \right) \mathrm { d } s .
$$

Since $\| \alpha \| _ { L ^ { 1 } ( [ 0 , s ] ) } \leq \| \alpha \| _ { L ^ { 1 } ( [ 0 , t ] ) }$ for all $s \in [ 0 , t ]$ , we can upper bound the integral by pulling this term out

$$
\| \varphi _ { t } ( x ) - x \| \leq \| \alpha \| _ { L ^ { 1 } ( [ 0 , t ] ) } + \left( \| \alpha \| _ { L ^ { 1 } ( [ 0 , t ] ) } + \| x \| \right) \int _ { 0 } ^ { t } \beta _ { s } e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , s ] ) } } \mathrm { d } s .
$$

For almost every $s \in [ 0 , 1 ]$ the chain rule for absolutely continuous functions gives $\beta _ { s } e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , s ] ) } } = \mathring { \frac { d } { d s } } e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , s ] ) } }$ . Hence, we obtain $e ^ { \| \beta \| _ { L ^ { 1 } ( [ 0 , t ] ) } } - 1$ and consequently

$$
\begin{array} { r l } & { \left. \varphi _ { t } ( x ) - x \right. \leq \left. \alpha \right. _ { L ^ { 1 } ( [ 0 , t ] ) } + \left( \left. \alpha \right. _ { L ^ { 1 } ( [ 0 , t ] ) } + \left. x \right. \right) \left( e ^ { \left. \beta \right. _ { L ^ { 1 } ( [ 0 , t ] ) } } - 1 \right) } \\ & { \qquad = \left. \alpha \right. _ { L ^ { 1 } ( [ 0 , t ] ) } e ^ { \left. \beta \right. _ { L ^ { 1 } ( [ 0 , t ] ) } } + \left. x \right. \left( e ^ { \left. \beta \right. _ { L ^ { 1 } ( [ 0 , t ] ) } } - 1 \right) . } \end{array}
$$

This concludes the proof of the first statement.

Lemma $\mathbf { A . } 3$ (Method of characteristics). Consider a velocity field $v \in L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 }$ and the correspondingflow $\varphi _ { \bullet } ^ { v }$ as well as an initial probability measure $\rho _ { 0 } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ withfinite second moments. Then $\rho _ { t } : = ( \varphi _ { t } ^ { v } ) _ { \sharp } \rho _ { 0 }$ is a distributional solution to the continuity equation

$$
\begin{array} { r } { \partial _ { t } \rho _ { t } + \nabla \cdot ( v _ { t } \rho _ { t } ) = 0 } \end{array}\tag{A.7}
$$

meaning that for any test function $\phi \in C _ { \mathrm { c } } ^ { \infty } ( ( 0 , 1 ) \times  { \mathbb { R } } ^ { d } )$ it holds that

$$
\int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \left( \partial _ { t } \phi _ { t } ( x ) + v _ { t } ( x ) \cdot \nabla \phi _ { t } ( x ) \right) \mathrm { d } \rho _ { t } ( x ) \mathrm { d } t = 0 .\tag{A.8}
$$

The proof can be found in [AGS08, Lemma 8.1.6.] or [FG21, Lemma 4.1.1.].

Lemma ${ \bf A . 4 }$ (Stability of flows). Consider two velocity fields $v _ { \bullet }$ and $u _ { \bullet }$ satisfying

$$
\begin{array} { r l } { \| v _ { t } ( 0 ) \| \le \alpha _ { t } } & { a n d \left\| v _ { t } ( x ) - v _ { t } ( y ) \right\| \le \beta _ { t } \| x - y \| } \\ { \| u _ { t } ( 0 ) \| \le \alpha _ { t } ^ { \prime } } & { a n d \left\| u _ { t } ( x ) - u _ { t } ( y ) \right\| \le \beta _ { t } ^ { \prime } \| x - y \| } \end{array}
$$

and by $\varphi ^ { v + h u }$ we denote theflow induced by the velocity $v + h u$ for $h \in \mathbb { R }$ . Then for any $\delta > 0$ there are constants $c _ { 1 } , c _ { 2 } \geq 0$ such that

$$
\begin{array} { r } { \| \varphi _ { t } ^ { v + h u } ( x ) - \varphi _ { t } ^ { v } ( x ) \| \leq | h | ( c _ { 1 } + c _ { 2 } \| x \| ) \quad f o r a l l \ h \in ( - \delta , \delta ) , x \in \mathbb { R } ^ { d } , t \in [ 0 , 1 ] , } \end{array}
$$

where we can choose

$$
\begin{array} { r l } & { c _ { 1 } = \| \beta ^ { \prime } \| _ { L _ { T } ^ { 1 } } \left( \| \alpha \| _ { L _ { T } ^ { 1 } } + \delta \| \alpha ^ { \prime } \| _ { L _ { T } ^ { 1 } } \right) e ^ { 2 \| \beta \| _ { L _ { T } ^ { 1 } } + \delta \| \beta ^ { \prime } \| _ { L _ { T } ^ { 1 } } } } \\ & { \qquad + \| \alpha ^ { \prime } \| _ { L _ { T } ^ { 1 } } e ^ { \| \beta \| _ { L _ { T } ^ { 1 } } } , \quad a n d } \\ & { c _ { 2 } = \| \beta ^ { \prime } \| _ { L _ { T } ^ { 1 } } e ^ { 2 \| \beta \| _ { L _ { T } ^ { 1 } } + \delta \| \beta ^ { \prime } \| _ { L _ { T } ^ { 1 } } } . } \end{array}
$$

Proof. Assume without loss of generality that $h > 0$ , as the other case is then obtained by symmetry. We fix $\boldsymbol { x } \in \mathbb { R } ^ { d }$ and set $x _ { t } ^ { h } : = \varphi _ { t } ^ { v + h u } ( x )$ . Then, we have

$$
\| x _ { t } ^ { h } - x _ { t } ^ { 0 } \| = \left\| \int _ { 0 } ^ { t } ( v _ { s } ( x _ { s } ^ { h } ) - v _ { s } ( x _ { s } ^ { 0 } ) ) \mathrm { d } s - h \int _ { 0 } ^ { t } u _ { s } ( x _ { s } ^ { h } ) \mathrm { d } s \right\|
$$

$$
\begin{array} { l } { \displaystyle \leq \int _ { 0 } ^ { t } \| v _ { s } ( x _ { s } ^ { h } ) - v _ { s } ( x _ { s } ^ { 0 } ) \| \mathrm { d } s + h \int _ { 0 } ^ { t } \| u _ { s } ( x _ { s } ^ { h } ) \| \mathrm { d } s } \\ { \displaystyle \leq \int _ { 0 } ^ { t } \beta _ { s } \| x _ { s } ^ { h } - x _ { s } ^ { 0 } \| \mathrm { d } s + h \int _ { 0 } ^ { t } \| u _ { s } ( x _ { s } ^ { h } ) \| \mathrm { d } s . } \end{array}
$$

To bound $\| u _ { s } \|$ , we use the notation $v _ { t } ^ { h } : = v + h u$ , which satisfies the boundedness and Lipschitz conditions with $\tilde { \alpha } = \alpha + h \alpha ^ { \prime }$ and $\widetilde { \beta } = \beta + h \beta ^ { \prime }$ and apply Grönwall’s inequality to

$$
\begin{array} { r l r } {  { \| { \boldsymbol x } _ { t } ^ { h } } \| = \| { \boldsymbol x } + \int _ { 0 } ^ { t } v _ { s } ^ { h } ( { \boldsymbol x } _ { s } ^ { h } ) \mathrm { d } s \| }  \\ & { } & { \leq \| { \boldsymbol x } \| + \int _ { 0 } ^ { t } \| v _ { s } ^ { h } ( { \boldsymbol x } _ { s } ^ { h } ) \| \mathrm { d } s } \\ & { } & { \leq \| { \boldsymbol x } \| + \int _ { 0 } ^ { t } ( \tilde { \alpha } _ { s } + \tilde { \beta } _ { s } \| { \boldsymbol x } _ { s } ^ { h } \| ) \mathrm { d } s , } \end{array}
$$

which yields

$$
\begin{array} { r l r } {  { \| x _ { t } ^ { h } \| \leq ( \| x \| + \int _ { 0 } ^ { t } \tilde { \alpha } _ { s } { \mathrm { d } } s ) \cdot \exp ( \int _ { 0 } ^ { t } \tilde { \beta } _ { s } { \mathrm { d } } s ) } } \\ & { } & { \leq ( \| x \| + \| \alpha \| _ { L _ { T } ^ { 1 } } + h \| \alpha ^ { \prime } \| _ { L _ { T } ^ { 1 } } ) \cdot \exp ( \| \beta \| _ { L _ { T } ^ { 1 } } + h \| \beta ^ { \prime } \| _ { L _ { T } ^ { 1 } } ) . } \end{array}
$$

This gives us

$$
\begin{array} { r l } & { \| u _ { s } ( x _ { s } ^ { h } ) \| \leq \alpha _ { s } ^ { \prime } + \beta _ { s } ^ { \prime } \| x _ { s } ^ { h } \| } \\ & { \qquad \leq \alpha _ { s } ^ { \prime } + \beta _ { s } ^ { \prime } \left( \| x \| + \| \alpha \| _ { L _ { T } ^ { 1 } } + h \| \alpha ^ { \prime } \| _ { L _ { T } ^ { 1 } } \right) } \\ & { \qquad \times \exp \left( \| \beta \| _ { L _ { T } ^ { 1 } } + h \| \beta ^ { \prime } \| _ { L _ { T } ^ { 1 } } \right) } \\ & { \qquad = : a _ { s } . } \end{array}
$$

Plugging back in yields

$$
\| x _ { t } ^ { h } - x _ { t } ^ { 0 } \| \leq \int _ { 0 } ^ { t } \beta _ { s } \| x _ { s } ^ { h } - x _ { s } ^ { 0 } \| \mathrm { d } s + h \int _ { 0 } ^ { t } a _ { s } \mathrm { d } s .
$$

An application of Grönwall’s inequality yields the claim.

Lemma $\mathbf { A . } _ { 5 }$ (Sensitivity of ODEs). Consider two velocity fields $v , u \in L _ { T } ^ { 1 } C _ { \mathrm { L i p } } ^ { 1 }$ and denote the flow induced by the velocity $v +$ hu $b y \ \varphi ^ { v + h u } \ f o r \ h \in \mathbb { R }$ . Then, the limit $\begin{array} { r } { \psi _ { t } ^ { u } ( x ) : = \left. \frac { \mathrm { d } } { \mathrm { d } h } \right| _ { h = 0 } \varphi _ { t } ^ { v + h u } ( x ) } \end{array}$ existsfor all $x \in \mathbb { R } ^ { d } , t \in [ 0 , 1 ]$ and is the unique solution to the equation

$$
\partial _ { t } \psi _ { t } ^ { u } ( x ) = D v _ { t } ( \varphi _ { t } ^ { v } ( x ) ) \cdot \psi _ { t } ^ { u } ( x ) + u _ { t } ( \varphi _ { t } ^ { v } ( x ) ) , \quad \psi _ { 0 } ( x ) = 0 .\tag{A.9}
$$

Proof. Let us denote the spatial Lipschitz constants of $v _ { \bullet }$ and $u _ { \bullet }$ by $\beta , \beta ^ { \prime } \in L _ { T } ^ { 1 }$ respectively. We fix the initial condition $\boldsymbol { x } \in \mathbb { R } ^ { d }$ and consider the diference quotients $\begin{array} { r } { \Delta _ { t } ^ { h } : = \frac { \varphi _ { t } ^ { v + h u } ( x ) - \varphi _ { t } ^ { v } ( x ) } { h } } \end{array}$ . Further, let $\psi$ be the unique solution of

$$
\partial _ { t } \psi _ { t } = D v _ { t } ( \varphi _ { t } ^ { v } ( x ) ) \cdot \psi _ { t } + u _ { t } ( \varphi _ { t } ^ { v } ( x ) ) \quad { \mathrm { a n d ~ } } \psi _ { 0 } = 0 .
$$

Note that there is a unique solution, as this is a linear ordinary diferential equation and $D v _ { t } ( \varphi _ { t } ^ { v } ( x ) )$ is bounded by $\beta _ { t }$ , where $\beta \in L _ { T } ^ { 1 }$ . We denote the diference between the diference quotients and the limiting candidate by $e _ { t } ^ { h } : = \Delta _ { t } ^ { h } - \psi _ { t }$ . By using the diferential equations that both $\Delta _ { t } ^ { h }$ as well as $\psi _ { t }$ satisfy, we obtain

$$
\begin{array} { l } { \displaystyle | e _ { t } ^ { h } | | = \left\| \int _ { 0 } ^ { t } \left( \frac { v _ { s } ( x _ { s } ^ { h } ) - v _ { s } ( x _ { s } ^ { 0 } ) } { h } - D v _ { s } ( x _ { s } ^ { 0 } ) \psi _ { s } \right) \mathrm { d } s + \int _ { 0 } ^ { t } ( u _ { s } ( x _ { s } ^ { h } ) - u _ { s } ( x _ { s } ^ { 0 } ) ) \mathrm { d } s \right\| } \\ { \displaystyle \quad \leq \int _ { 0 } ^ { t } \left\| \frac { v _ { s } ( x _ { s } ^ { h } ) - v _ { s } ( x _ { s } ^ { 0 } ) } { h } - D v _ { s } ( x _ { s } ^ { 0 } ) \psi _ { s } \right\| \mathrm { d } s + \int _ { 0 } ^ { t } \| u _ { s } ( x _ { s } ^ { h } ) - u _ { s } ( x _ { s } ^ { 0 } ) \| \mathrm { d } s } \\ { \displaystyle \quad \leq \int _ { 0 } ^ { t } \left\| \frac { v _ { s } ( x _ { s } ^ { h } ) - v _ { s } ( x _ { s } ^ { 0 } ) } { h } - D v _ { s } ( x _ { s } ^ { 0 } ) \psi _ { s } \right\| \mathrm { d } s + \int _ { 0 } ^ { t } \beta _ { s } ^ { \prime } \| x _ { s } ^ { h } - x _ { s } ^ { 0 } \| \mathrm { d } s } \\ { \displaystyle \quad \leq \int _ { 0 } ^ { t } \left\| \frac { v _ { s } ( x _ { s } ^ { h } ) - v _ { s } ( x _ { s } ^ { 0 } ) } { h } - D v _ { s } ( x _ { s } ^ { 0 } ) \psi _ { s } \right\| \mathrm { d } s + h ( c _ { 1 } + c _ { 2 } \| x \| ) \int _ { 0 } ^ { t } \beta _ { s } ^ { \prime } \mathrm { d } s } \end{array}
$$

By Theorem $_ { \mathrm { A } . 2 }$ we can find a compact set $K$ such that $x _ { s } ^ { h } \in K$ for all $h \in ( - \delta , \delta )$ $s \in [ 0 , 1 ]$ . Over $K$ we obtain

$$
\| v _ { s } ( y ) - v _ { s } ( x ) - D v _ { s } ( x ) ( y - x ) \| \leq \omega _ { D v _ { s } } ^ { K } ( \| x - y \| ) \| x - y \|
$$

for $x , y \in K$ , where $\omega _ { D v _ { s } } ^ { K }$ denotes the modulus of continuity. Now, we estimate

$$
\begin{array} { r } { \bigg \| \frac { v _ { s } ( x _ { s } ^ { 0 } ) - v _ { s } ( x _ { s } ^ { 0 } ) } { \hbar } - D v _ { s } ( x _ { s } ^ { 0 } ) \psi _ { s } \bigg \| \leq \bigg \| \frac { D v _ { s } ( x _ { s } ^ { 0 } ) ( x _ { s } ^ { 0 } - x _ { s } ^ { 0 } ) } { \hbar } - D v _ { s } ( x _ { s } ^ { 0 } ) \psi _ { s } \bigg \| } \\ { + \frac { \omega _ { P v _ { s } } ^ { K } ( \| x _ { s } ^ { 0 } - x _ { s } ^ { 0 } \| ) \| x _ { s } ^ { 0 } - x _ { s } ^ { 0 } \| } { \hbar } } \\ { \leq \| D v _ { s } ( x _ { s } ^ { 0 } ) \| \left\| \Delta _ { s } ^ { k } - \psi _ { s } \right\| } \\ { + \frac { \omega _ { P v _ { s } } ^ { K } ( ( c _ { 1 } + c _ { 2 } \| x \| ) \hbar ) ( c _ { 1 } + c _ { 2 } \| x \| ) \hbar } { \hbar } } \\ { = \| D v _ { s } ( x _ { s } ^ { 0 } ) \| \left\| c _ { 1 } ^ { k } \right\| } \\ { + \omega _ { P v _ { s } } ^ { K } ( ( c _ { 1 } + c _ { 2 } \| x \| ) \hbar ) ( c _ { 1 } + c _ { 2 } \| x \| ) } \\ { \leq \| D v _ { s } ( x _ { s } ^ { 0 } ) \| \left\| c _ { 1 } ^ { k } \right\| + \omega _ { P v _ { s } } ^ { K } ( c _ { 2 } \hbar ) . } \end{array}
$$

This gives

$$
\| e _ { t } ^ { h } \| \leq \int _ { 0 } ^ { t } \beta _ { s } \| e _ { s } ^ { h } \| \mathrm { d } s + f ( h ) ,
$$

where

$$
f ( h ) = c _ { 3 } \int _ { 0 } ^ { t } \omega _ { D v _ { s } } ^ { K } ( c _ { 4 } h ) \mathrm { d } s + c _ { 5 } h \int _ { 0 } ^ { t } \beta _ { s } ^ { \prime } \mathrm { d } s .
$$

Grönwall’s inequality yields

$$
\| e _ { t } ^ { h } \| \leq f ( h ) e ^ { \int _ { 0 } ^ { t } \beta _ { s } \mathrm { d } s } ,
$$

hence it remains to show that $\begin{array} { r } { \operatorname* { l i m } _ { h \to 0 } f ( h ) = 0 } \end{array}$ . The second term of $f$ clearly vanishes as $h  0$ . To see that the second term vanishes, we first notice that for any $r > 0$ we obtain

$$
\omega _ { D v _ { s } } ^ { K } ( r ) = \operatorname* { s u p } \{ \| D v _ { s } ( x ) - D v _ { s } ( y ) \| : x , y \in K , \| x - y \| \leq r \} \leq 2 \| D v _ { s } \| _ { \infty } \leq 2 \beta _ { s } ,
$$

which is integrable in time. As $D v _ { s }$ is continuous and K is compact, pointwise in time, we have lim $1 _ { h  0 } \omega _ { D v _ { s } } ^ { K } ( c _ { 4 } h ) = 0$ and by the dominated convergence theorem, it holds that

$$
\operatorname* { l i m } _ { h \to 0 } \int _ { 0 } ^ { t } \omega _ { D v _ { s } } ^ { K } ( c _ { 4 } h ) \mathrm { d } s = 0 ,
$$

which finishes the proof.

## Appendix B Fisher–Rao Gradient Flows and Geodesics

Here, we derive the expressions of the Fisher–Rao gradients of the KL-divergence and reverse KL-divergence. We show that they point into the directions of the $m -$ and e-geodesics, respectively.

Lemma B.1 (m- and e-logarithmic maps). For any two probability measures $\mathbb { P } _ { 0 }$ and $\mathbb { P } _ { 1 }$ we have

$$
\left. \frac { \mathrm { d } } { \mathrm { d } t } \right| _ { t = 0 } \mathbb { P } _ { t } ^ { ( m ) } = \mathbb { P } _ { 1 } - \mathbb { P } _ { 0 } .
$$

If further $\mathbb { P } _ { 0 } \ll \mathbb { P } _ { 1 }$ and $D _ { \mathrm { K L } } ( \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) < + \infty$ , then

$$
\left. \frac { \mathrm { d } } { \mathrm { d } t } \right| _ { t = 0 } \mathbb { P } _ { t } ^ { ( e ) } = \left( D _ { \mathrm { K L } } ( \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) - \ln \left( \frac { \mathrm { d } \mathbb { P } _ { 0 } } { \mathrm { d } \mathbb { P } _ { 1 } } \right) \right) \mathbb { P } _ { 0 } .
$$

Proof. The first statement is obtained by diferentiating the linear interpolation. For the second statement, we fix a reference measure $\mathbb { Q } ,$ set $\begin{array} { r } { \rho _ { t } = \frac { \mathrm { d } \mathbb { P } _ { t } ^ { ( e ) } } { \mathrm { d } \mathbb { Q } } \propto \rho _ { 0 } ^ { 1 - t } \rho _ { 1 } ^ { t } } \end{array}$ , denote the normalization constant by $\begin{array} { r } { \psi _ { t } = \int \rho _ { 0 } ^ { 1 - t } \rho _ { 1 } ^ { t } \ d t } \end{array}$ dx. We note that $\psi _ { 0 } = 1$ and

$$
\frac { \mathrm { d } } { \mathrm { d } t } \rho _ { 0 } ^ { 1 - t } \rho _ { 1 } ^ { t } = \frac { \mathrm { d } } { \mathrm { d } t } \rho _ { 0 } \exp \left( t \ln \left( \frac { \rho _ { 1 } } { \rho _ { 0 } } \right) \right) = \rho _ { 0 } ^ { 1 - t } \rho _ { 1 } ^ { t } \ln \left( \frac { \rho _ { 1 } } { \rho _ { 0 } } \right) .
$$

Further, we have

$$
\left. \frac { \mathrm { d } } { \mathrm { d } t } \right| _ { t = 0 } \psi _ { t } = \int \left. \frac { \mathrm { d } } { \mathrm { d } t } \right| _ { t = 0 } \rho _ { 0 } ^ { 1 - t } \rho _ { 1 } ^ { t } \mathrm { d } x = \int \rho _ { 0 } ^ { 1 - t } \rho _ { 1 } ^ { t } \ln \left( \frac { \rho _ { 1 } } { \rho _ { 0 } } \right) \mathrm { d } x | _ { t = 0 } = - D _ { \mathrm { K L } } ( \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) .
$$

Overall, we compute

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } } { \mathrm { d } t } \rho _ { t } | _ { t = 0 } = \frac { \mathrm { d } } { \mathrm { d } t } \left( \frac { \rho _ { 0 } ^ { 1 - t } \rho _ { 1 } ^ { t } } { \psi _ { t } } \right) | _ { t = 0 } } \\ { = \frac { \psi _ { 0 } \mathrm { d } t } { \mathrm { d } t } \rho _ { 0 } ^ { \mathrm { { 1 - } } t } \rho _ { 1 } ^ { t } | _ { t = 0 } - \rho _ { 0 } \frac { \mathrm { d } } { \mathrm { d } t } \psi _ { t } | _ { t = 0 } } \\ { \displaystyle \qquad \psi _ { 0 } ^ { 2 } } \\ { = \rho _ { 0 } \ln \left( \frac { \rho _ { 1 } } { \rho _ { 0 } } \right) + \rho _ { 0 } D _ { \mathrm { K L } } ( \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) } \\ { \displaystyle \qquad = \left( D _ { \mathrm { K L } } ( \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) - \ln \left( \frac { \rho _ { 0 } } { \rho _ { 1 } } \right) \right) \rho _ { 0 } . } \end{array}
$$

Theorem B.2 (Fisher–Rao gradient flows and geodesics). Consider two probability measures P and $\mathbb { P } ^ { \star }$ ${ \cal I } f D _ { \mathrm { K L } } ( \mathbb { P } ^ { \star } , \mathbb { P } ) < + \infty$ it holds that

$$
\nabla _ { \mathbb { P } } ^ { \mathtt { F R } } D _ { \mathrm { K L } } ( \mathbb { P } ^ { \star } , \mathbb { P } ) = \mathbb { P } - \mathbb { P } ^ { \star } ,\tag{B.1}
$$

in the sense that

$$
\left. \frac { \mathrm { d } } { \mathrm { d } h } \right| _ { h = 0 } D _ { \mathrm { K L } } ( \mathbb { P } ^ { \star } , \mathbb { P } + h \Xi ) = g _ { \mathbb { P } } ^ { \mathrm { F R } } ( \mathbb { P } - \mathbb { P } ^ { \star } , \Xi )
$$

for $a l l \equiv$ such that $D _ { \mathrm { K L } } ( \mathbb { P } ^ { \star } , \mathbb { P } + h \Xi ) < + \infty f o r \ | h |$ small enough. Further, $i f$ $D _ { \mathrm { K L } } ( \mathbb { P } , \mathbb { P } ^ { \star } ) < + \infty$ it holds that

$$
\nabla _ { \mathbb { P } } ^ { \mathtt { F R } } D _ { \mathrm { K L } } ( \mathbb { P } , \mathbb { P } ^ { \star } ) = \left( \ln \left( { \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { P } ^ { \star } } } \right) - D _ { \mathrm { K L } } ( \mathbb { P } , \mathbb { P } ^ { \star } ) \right) \mathbb { P }\tag{B.2}
$$

in the sense that

$$
\left. { \frac { \mathrm { d } } { \mathrm { d } h } } \right| _ { h = 0 } D _ { \mathrm { K L } } ( \mathbb { P } + h \Xi , \mathbb { P } ^ { \star } ) = g _ { \mathbb { P } } ^ { \mathrm { F R } } \left( \left( \ln \left( { \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { P } ^ { \star } } } \right) - D _ { \mathrm { K L } } ( \mathbb { P } , \mathbb { P } ^ { \star } ) \right) \mathbb { P } , \Xi \right)
$$

for $a l l \Xi \ll \mathbb { P }$ such that $D _ { \mathrm { K L } } ( \mathbb { P } + h \Xi , \mathbb { P } ^ { \star } ) < + \infty f o r \left| h \right|$ small enough.

Proof. For the finite-dimensional case, see, for example, [DA25]. Here, we provide the proof for general Polish spaces. To this end, we compute the first variation of $D _ { \mathrm { K L } } ( \mathbb { P } ^ { \star } , \mathbb { P } )$ into a direction $\Xi$ such that $D _ { \mathrm { K L } } ( \mathbb { P } ^ { \star } , \mathbb { P } + h \Xi )$ is finite for $| h |$ suficiently small. Note that the finiteness of the Kullback–Leibler divergence together with the concavity of the logarithm allows us to use the dominated convergence theorem, and we obtain

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } } { \mathrm { d } h } D _ { \mathrm { K L } } ( \mathbb { P } ^ { \star } , \mathbb { P } + h \Xi ) | _ { h = 0 } = \frac { \mathrm { d } } { \mathrm { d } h } \mathbb { E } _ { \mathbb { P } ^ { \star } } \left[ \mathrm { l n } \left( \frac { \mathrm { d } \mathbb { P } ^ { \star } } { \mathrm { d } ( \mathbb { P } + h \Xi ) } \right) \right] \Big | _ { h = 0 } } \\ { \displaystyle = \frac { \mathrm { d } } { \mathrm { d } h } \left( - \mathbb { E } _ { \mathbb { P } ^ { \star } } \left[ \mathrm { l n } \left( \frac { \mathrm { d } ( \mathbb { P } + h \Xi ) } { \mathrm { d } \mathbb { P } ^ { \star } } \right) \right] \right) \Big | _ { h = 0 } } \end{array}
$$

$$
\begin{array} { r l } & { = - \mathbb { E } _ { \mathbb { P } ^ { \star } } \left[ \frac { \mathrm { d } } { \mathrm { d } h } \ln \left( \frac { \mathrm { d } ( \mathbb { P } + h \Xi ) } { \mathrm { d } \mathbb { P } ^ { \star } } \right) \Big | _ { h = 0 , } \right. } \\ & { = - \mathbb { E } _ { \mathbb { P } ^ { \star } } \left[ \frac { \mathrm { d } h \frac { \mathrm { d } ( \mathbb { P } + h \Xi ) } { \mathrm { d } \mathbb { P } ^ { \star } } } { \frac { \mathrm { d } ( \mathbb { P } + h \Xi ) } { \mathrm { d } \mathbb { P } ^ { \star } } } \Big | _ { h = 0 } \right] } \\ & { = - \mathbb { E } _ { \mathbb { P } ^ { \star } } \left[ \frac { \mathrm { d } \Xi } { \frac { \mathrm { d } \mathbb { P } ^ { \star } } { \mathrm { d } \mathbb { P } ^ { \star } } } \right] } \\ & { = - \mathbb { E } _ { \mathbb { P } ^ { \star } } \left[ \frac { \mathrm { d } \Xi } { \frac { \mathrm { d } \mathbb { P } ^ { \star } } { \mathrm { d } \mathbb { P } ^ { \star } } } \right] } \end{array}
$$

Note that $Z = \mathbb { P } - \mathbb { P } ^ { \star }$ is contained in the tangent space and we have

$$
g _ { \mathbb { P } } ^ { \mathrm { F R } } ( \Xi , Z ) = \mathbb { E } _ { \mathbb { P } } \left[ \frac { \mathrm { d } \Xi } { \mathrm { d } \mathbb { P } } \cdot \frac { \mathrm { d } Z } { \mathrm { d } \mathbb { P } } \right] = \mathbb { E } _ { \mathbb { P } } \left[ \frac { \mathrm { d } \Xi } { \mathrm { d } \mathbb { P } } \right] - \mathbb { E } _ { \mathbb { P } } \left[ \frac { \mathrm { d } \Xi } { \mathrm { d } \mathbb { P } } \cdot \frac { \mathrm { d } \mathbb { P } ^ { \star } } { \mathrm { d } \mathbb { P } } \right] = - \mathbb { E } _ { \mathbb { P } ^ { \star } } \left[ \frac { \mathrm { d } \Xi } { \mathrm { d } \mathbb { P } } \right] .
$$

For the second part, we fix Ξ with an $L ^ { 1 } ( \mathbb { P } )$ density such that $D _ { \mathrm { K L } } ( \mathbb { P } + h \Xi , \mathbb { P } ^ { \star } )$ is finite for $| h |$ small enough. As the integrand x ln x is to compute the first variation

$$
\begin{array} { r l } { \frac { \mathrm { d } } { \mathrm { d } h } D _ { \mathrm { K L } } ( \mathbb { P } + h \Xi , \| \mathbf { \varPsi } ^ { * } ) \| _ { h = 0 } = \frac { \mathrm { d } } { \mathrm { d } h } \mathbb { E } _ { \mathbb { P } ^ { * } \mid h \Xi } [ \mathrm { l n } ( \frac { \mathrm { d } ( \mathbb { P } + h \Xi ) } { \mathrm { d } \mathbb { P } ^ { * } } ) ] \Big | _ { h = 0 } } & { } \\ & { = \mathbb { E } _ { \Xi } [ \mathrm { l n } ( \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { P } ^ { * } } ) ] + \mathbb { E } _ { \mathbb { P } } [ \frac { \mathrm { d } } { \mathrm { d } h } \mathrm { ~ ( \frac { d ( \mathbb { P } + h \Xi ) } { \mathrm { d } \mathbb { P } ^ { * } } ) ~ \Big | ~ } _ { h = 0 } } \\ & { = \mathbb { E } _ { \Xi } [ \mathrm { l n } ( \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { P } ^ { * } } ) ] + \mathbb { E } _ { \mathbb { P } } [ \frac { \mathrm { d } h } { \frac { \mathrm { d } h ( \mathbb { P } + h \Xi ) } { \mathrm { d } \mathbb { P } ^ { * } } } \Big | _ { h = 0 } ] } \\ & { = \mathbb { E } _ { \Xi } [ \mathrm { l n } ( \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { P } ^ { * } } ) ] + \mathbb { E } _ { \mathbb { P } } [ \frac { \mathrm { d } \Xi } { \frac { h ( \mathbb { P } + h \Xi ) } { \mathrm { d } \mathbb { P } ^ { * } } } \Big | _ { h = 0 } ] } \\ &  = \mathbb { E } _ { \Xi } [ \mathrm { l n } ( \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { P } ^ { * } } ) ] + \mathbb { E } _ { \mathbb { P } } [ \frac { \mathrm { d } \Xi }  \ \end{array}
$$

Note that $\begin{array} { r } { Z = ( \ln \left( \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { P } ^ { \star } } \right) - D _ { \mathrm { K L } } ( \mathbb { P } , \mathbb { P } ^ { \star } ) ) \mathbb { P } } \end{array}$ is contained in the tangent space, and we have

$$
\begin{array} { r l } & { g _ { \mathbb { P } } ^ { \mathrm { F R } } ( \Xi , Z ) = \mathbb { E } _ { \mathbb { P } } \left[ \frac { \mathrm { d } \Xi } { \mathrm { d } \mathbb { P } } \cdot \left( \ln \left( \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { P } ^ { \star } } \right) - D _ { \mathrm { K L } } ( \mathbb { P } , \mathbb { P } ^ { \star } ) \right) \right] } \\ & { \quad \quad \quad = \mathbb { E } _ { \mathbb { P } } \left[ \frac { \mathrm { d } \Xi } { \mathrm { d } \mathbb { P } } \cdot \ln \left( \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { P } ^ { \star } } \right) \right] - D _ { \mathrm { K L } } ( \mathbb { P } , \mathbb { P } ^ { \star } ) \cdot \mathbb { E } _ { \mathbb { P } } \left[ \frac { \mathrm { d } \Xi } { \mathrm { d } \mathbb { P } } \right] } \\ & { \quad \quad \quad = \mathbb { E } _ { \mathbb { P } } \left[ \frac { \mathrm { d } \Xi } { \mathrm { d } \mathbb { P } } \cdot \ln \left( \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { P } ^ { \star } } \right) \right] . } \end{array}
$$

## Acknowledgements

We thank Guido Montúfar for helpful comments on this manuscript. The first author acknowledges support from the Max Planck Society through the Research Group “Stochastic Analysis in the Sciences (SAiS)". This work was supported in part by ERC grant FluCo (Grant Agreement No. 101088488). Funded by the European Union. Views and opinions expressed are, however, those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## References

[AB10] M. Annunziato and A. Borzi. Optimal control of probability density functions of stochastic processes. Mathematical Modelling and Analysis 15, no. 4, (2010), 393–407.

[ABVE25] M. Albergo, N. M. Boffi, and E. Vanden-Eijnden. Stochastic Interpolants: A Unifying Framework for Flows and Difusions. Journal of Machine Learning Research 26, no. 209, (2025), 1–80. http://jmlr.org/papers/ v26/23-1605.html.

[AC14] L. Ambrosio and G. Crippa. Continuity equations and ODE flows with non-smooth velocity. Proceedings of the Royal Society of Edinburgh Section A: Mathematics 144, no. 6, (2014), 1191–1244.

[ADB25] S. Azeglio and A. Di Bernardo. What’s Inside Your Difusion Model? A Score-Based Riemannian Metric to Explore the Data Manifold. arXiv preprint arXiv:2505.11128 (2025).

[AG12] L. Ambrosio and N. Gigli. A user’s guide to optimal transport. In Modelling and Optimisation of Flows on Networks: Cetraro, Italy 2009, Editors: Benedetto Piccoli, Michel Rascle, 1–155. Springer, 2012.

[AGS05] L. Ambrosio, N. Gigli, and G. Savaré. Gradient Flows: In Metric Spaces and in the Space of Probability Measures. Springer, 2005.

[AGS08] L. Ambrosio, N. Gigli, and G. Savaré. Gradient Flows in Metric Spaces and in the Space ofProbability Measures. Birkhäuser, 2008.

[AJLS17] N. Ay, J. Jost, H. V. Lê, and L. Schwachhöfer. Information Geometry, vol. 64 of Ergebnisse der Mathematik und ihrer Grenzgebiete. 3. Folge. A Series ofModern Surveys in Mathematics [Results in Mathematics and Related Areas. 3rd Series. A Series of Modern Surveys in Mathematics]. Springer, Cham, 2017, xi+407. doi:10.1007/978-3-319-56478-4.

[Ama08] S.-i. Amari. Information geometry and its applications: Convex function and dually flat manifold. In LIX Fall Colloquium on Emerging Trends in Visual Computing, 75–102. Springer, 2008.

[Ama16] S.-i. Amari. Information Geometry and Its Applications, vol. 194 of Applied Mathematical Sciences. Springer, [Tokyo], 2016, xiii+374. doi: 10.1007/978-4-431-55978-8.

[Ay25] N. Ay. Information geometry of the Otto metric. Information geometry 8, no. Suppl 1, (2025), 209–232.

[BB00] J.-D. Benamou and Y. Brenier. A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem. Numerische Mathematik 84, no. 3, (2000), 375–393. doi:10.1007/s002110050002.

[BBR+25] D. Blessing, J. Berner, L. Richter, C. Domingo i Enrich, Y. Du, A. Vahdat, and G. Neumann. Trust Region Constrained Measure Transport in Path Space for Stochastic Optimal Control and Inference 38, (2025), 165462–165510. https: //proceedings.neurips.cc/paper\_files/paper/2025/file/ f1ccd13e51f610ff4a7a08db9a8b458b-Paper-Conference.pdf.

[BDD24] J. Benton, G. Deligiannidis, and A. Doucet. Error Bounds for Flow Matching Methods. Transactions on Machine Learning Research (2024). https://openreview.net/forum?id=uqQPyWFDhY.

[BDTD+16] M. Bojarski, D. Del Testa, D. Dworakowski, B. Firner, B. Flepp, P. Goyal, L. D. Jackel, M. Monfort, U. Muller, J. Zhang, X. Zhang, J. Zhao, and K. Zieba. End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316 (2016).

[BGME25] Q. Bertrand, A. Gagneux, M. Massias, and R. Emonet. On the Closed-Form of Flow Matching: Generalization Does Not Arise from Target Stochasticity. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, eds., Advances in Neural Information Processing Systems, vol. 38, 8522–8549. Curran Associates, Inc., 2025.

[BGN22] E. Bernton, P. Ghosal, and M. Nutz. Entropic optimal transport: Geometry and large deviations. Duke Mathematical Journal 171, no. 16, (2022), 3363– 3400.

[BRU24] J. Berner, L. Richter, and K. Ullrich. An optimal control perspective on difusion-based generative modeling. Transactions on Machine Learning Research (2024). https://openreview.net/forum?id=oYIjw37pTP.

[BS03] J. A. Bagnell and J. G. Schneider. Covariant policy search. In IJCAI, 1019–1024. 2003.

[CCGT23] A. Chiarini, G. Conforti, G. Greco, and L. Tamanini. Gradient estimates for the schrödinger potentials: convergence to the brenier map and quantitative stability. Communications in Partial Diferential Equations 48, no. 6, (2023), 895–943.

[CCH+26] J. A. Carrillo, Y. Chen, D. Z. Huang, J. Huang, and D. Wei. Fisher-Rao Gradient Flow: Geodesic Convexity and Functional Inequalities. SIAM Journal on Mathematical Analysis 58, no. 2, (2026), 1062–1099.

[CDPS17] G. Carlier, V. Duval, G. Peyré, and B. Schmitzer. Convergence of entropic schemes for optimal transport and gradient flows. SIAM Journal on Mathematical Analysis 49, no. 2, (2017), 1385–1418.

[CHH+26] Y. Chen, D. Z. Huang, J. Huang, S. Reich, and A. M. Stuart. Sampling via Gradient Flows in the Space of Probability Measures. Mathematics of Computation (2026). doi:10.1090/mcom/4186.

[CPSV18] L. Chizat, G. Peyré, B. Schmitzer, and F.-X. Vialard. An interpolating distance between optimal transport and fisher–rao metrics. Foundations of Computational Mathematics 18, no. 1, (2018), 1–44.

[CPT23] G. Carlier, P. Pegon, and L. Tamanini. Convergence rate of general entropic optimal transport costs. Calc. Var. Partial Diferential Equations 62, no. 4, (2023), Paper No. 116, 28. doi:10.1007/s00526-023-02455-0.

[CRL+20] L. Chizat, P. Roussillon, F. Léger, F.-X. Vialard, and G. Peyré. Faster wasserstein distance estimation with the sinkhorn divergence. Advances in neural information processing systems 33, (2020), 2257–2269.

[CT21] G. Conforti and L. Tamanini. A formula for the time derivative of the entropic cost and applications. Journal of Functional Analysis 280, no. 11, (2021), 108964.

[DA25] A. Datar and N. Ay. Convergence Properties of Natural Gradient Descent for Minimizing KL Divergence. Transactions on Machine Learning Research (2025). https://openreview.net/forum?id=h6hjjAF5Bj.

[DBTHD21] V. De Bortoli, J. Thornton, J. Heng, and A. Doucet. Difusion Schrödinger Bridge with Applications to Score-Based Generative Modeling. Advances in Neural Information Processing Systems 34, (2021), 17695–17709.

[DEDKC25] C. Domingo-Enrich, M. Drozdzal, B. Karrer, and R. T. Q. Chen. Adjoint Matching: Fine-tuning Flow and Difusion Generative Models with Memoryless Stochastic Optimal Control. In International Conference on Learning Representations. 2025.

[DEHA+24] C. Domingo-Enrich, J. Han, B. Amos, J. Bruna, and R. T. Chen. Stochastic optimal control matching. Advances in Neural Information Processing Systems 37, (2024), 112459–112504.

[DKB15] L. Dinh, D. Krueger, and Y. Bengio. NICE: Non-linear independent components estimation. In International Conference on Learning Representations (ICLR) Workshop Track. 2015.

[EN24] S. Eckstein and M. Nutz. Convergence rates for regularized optimal transport via quantization. Mathematics ofOperations Research 49, no. 2, (2024), 1223–1240.

[FG21] A. Figalli and F. Glaudo. An Invitation to Optimal Transport, Wasserstein Distances, and Gradient Flows, vol. 1. EMS Press Berlin, 2021.

[FW98] M. I. Freidlin and A. D. Wentzell. Random perturbations. In Random perturbations ofdynamical systems, 15–43. Springer, 1998.

[Fö85] H. Föllmer. An entropy approach to the time reversal of difusion processes. In Stochastic Diferential Systems (Marseille-Luminy, 1984), vol. 69 of Lect. Notes Control Inf. Sci., 156–163. Springer, Berlin, 1985. doi:10.1007/ BFb0005070.

[GCDGZ25] A. Guzman-Cordero, F. Dangel, G. Goldshlager, and M. Zeinhofer. Improving Energy Natural Gradient Descent Through Woodbury, Momentum, and Randomization. In Advances in Neural Information Processing Systems. 2025.

[GLC+23] S. Ghimire, J. Liu, A. Comas, D. Hill, A. Masoomi, O. Camps, and J. Dy. Geometry of Score Based Generative Models. arXiv preprint arXiv:2302.04411 (2023).

[GPAM+14] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative Adversarial Nets. Advances in Neural Information Processing Systems (2014).

[GT21] N. Gigli and L. Tamanini. Second order diferentiation formula on rcd\* (k, n) spaces. Journal of the European Mathematical Society 23, no. 5, (2021), 1727–1795.

[HE25] P. Holderrieth and E. Erives. An Introduction to Flow Matching and Difusion Models. arXiv preprint arXiv:2506.02070 (2025).

[HJA20] J. Ho, A. Jain, and P. Abbeel. Denoising difusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 6840–6851. 2020.

[HSL25] A. Hu, H. Smith, and S. Linderman. SING: SDE Inference via Natural Gradients. In Advances in Neural Information Processing Systems. 2025.

[Kak01] S. M. Kakade. A natural policy gradient. Advances in neural information processing systems 14(2001).

[KHP+26] R. Karczewski, M. Heinonen, A. Pouplin, S. Hauberg, and V. K. Garg. The Spacetime of Difusion Models: An Information Geometry Perspective. In International Conference on Learning Representations. 2026.

[KS14] I. Karatzas and S. Shreve. Brownian motion and stochastic calculus. springer, 2014.

[KW14] D. P. Kingma and M. Welling. Auto-Encoding Variational Bayes. International Conference on Learning Representations (2014).

[LCBH+23] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow Matching for Generative Modeling. In The Eleventh International Conference on Learning Representations. 2023.

[LCT21] G.-H. Liu, T. Chen, and E. Theodorou. Second-Order Neural ODE Optimizer. Advances in Neural Information Processing Systems 34, (2021), 25267–25279.

[Leb05] G. Lebanon. Axiomatic geometry of conditional models. IEEE Transactions on Information Theory 51, (2005), 1283–1294.

[Léo12a] C. Léonard. From the Schrödinger problem to the Monge–Kantorovich problem. Journal ofFunctional Analysis 262, no. 4, (2012), 1879–1920.

[Léo12b] C. Léonard. Girsanov Theory Under a Finite Entropy Condition. In Séminaire de Probabilités XLIV, 429–465. Springer, 2012.

[LHH+24] Y. Lipman, M. Havasi, P. Holderrieth, N. Shaul, M. Le, B. Karrer, R. T. Chen, D. Lopez-Paz, H. Ben-Hamu, and I. Gat. Flow Matching Guide and Code. arXiv preprint arXiv:2412.06264 (2024).

[LLN19] Y. Lu, J. Lu, and J. Nolen. Accelerating Langevin Sampling with Birth-Death. arXiv preprint arXiv:1905.09863 (2019).

[LMS18] M. Liero, A. Mielke, and G. Savaré. Optimal entropy-transport problems and a new hellinger–kantorovich distance between positive measures. Inventiones mathematicae 211, no. 3, (2018), 969–1117.

[LR26] P. López-Rivera. A uniform rate of convergence for the entropic potentials in the quadratic euclidean setting. ESAIM: Control, Optimisation and Calculus of Variations 32, (2026), 27.

[LSK+25] C.-H. Lai, Y. Song, D. Kim, Y. Mitsufuji, and S. Ermon. The Principles of Difusion Models. arXiv preprint arXiv:2510.21890 (2025).

[LSW23] Y. Lu, D. Slepčev, and L. Wang. Birth-Death Dynamics for Sampling: Global Convergence, Approximations and Their Asymptotics. Nonlinearity 36, no. 11, (2023), 5731–5772.

[Mar20] J. Martens. New Insights and Perspectives on the Natural Gradient Method. Journal ofMachine Learning Research 21, no. 146, (2020), 1–76.

[Mik02] T. Mikami. Optimal control for absolutely continuous stochastic processes and the mass transportation problem (2002).

[Mik04] T. Mikami. Monge’s problem with a quadratic cost by the zero-noise limit of h-path processes. Probability theory and related fields 129, no. 2, (2004), 245–260.

[MM24] J. Müller and G. Montúfar. Geometry and Convergence of Natural Policy Gradient Methods. Information Geometry 7, no. Suppl 1, (2024), 485–523.

[MRA14] G. Montúfar, J. Rauh, and N. Ay. On the Fisher metric of conditional probability polytopes. Entropy 16, no. 6, (2014), 3207–3233.

[MS25] H. Malamut and M. Sylvestre. Convergence rates of the regularized optimal transport: Disentangling suboptimality and entropy. SIAM Journal on Mathematical Analysis 57, no. 3, (2025), 2533–2558.

[MZ23] J. Müller and M. Zeinhofer. Achieving High Accuracy with PINNs via Energy Natural Gradient Descent. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, eds., Proceedings ofthe 40th International Conference on Machine Learning, vol. 202 of Proceedings of Machine Learning Research, 25471–25485. PMLR, 2023.

[MZ24] J. Müller and M. Zeinhofer. Position: Optimization in SciML Should Employ the Function Space Geometry. Proceedings of the 41st International Conference on Machine Learning 235, (2024), 36705–36722.

[Nag05] H. Nagaoka. The exponential family of Markov chains and its information geometry. In 28th Symposium on Information Theory and Its Applications (SITA2005). 2005.

[NR21] N. Nüsken and L. Richter. Solving High-Dimensional Hamilton–Jacobi–Bellman PDEs Using Neural Networks: Perspectives from the Theory of Controlled Difusions and Measures on Path Space. Partial Diferential Equations and Applications 2, no. 4, (2021), 48.

[NW22] M. Nutz and J. Wiesel. Entropic optimal transport: Convergence of potentials. Probability Theory and Related Fields 184, no. 1, (2022), 401– 424.

[NZ26] M. Nutz and C. Zhong. Entropic regularization of monge’s problem. arXiv preprint arXiv:2604.21578 (2026).

[Ott01] F. Otto. The Geometry of Dissipative Evolution Equations: The Porous Medium Equation. Communications in Partial Diferential Equations 26, no. 1-2, (2001), 101–174.

[Pal24] S. Pal. On the diference between entropic cost and the optimal transport cost. The Annals of Applied Probability 34, no. 1B, (2024), 1003–1028.

[PNW21] A.-A. Pooladian and J. Niles-Weed. Entropic estimation of optimal transport maps. arXiv preprint arXiv:2109.12004 (2021).

[Pom88] D. A. Pomerleau. ALVINN: An autonomous land vehicle in a neural network. In Advances in Neural Information Processing Systems. 1988.

[PR14] B. Piccoli and F. Rossi. Generalized wasserstein distance and its application to transport equations with source. Archive for Rational Mechanics and Analysis 211, no. 1, (2014), 335–358.

[PS95] G. Pistone and C. Sempi. An infinite-dimensional geometric structure on the space of all the probability measures equivalent to a given one. The annals of statistics (1995), 1543–1561.

[PTDN26] E. Pierret, V. Tosel, J. Delon, and A. Newson. Flow Matching for Applied Mathematicians (2026).

[PVS03] J. Peters, S. Vijayakumar, and S. Schaal. Reinforcement learning for humanoid robotics. In Proceedings of the third IEEE-RAS international conference on humanoid robots, 1–20. 2003.

[RB10] S. Ross and J. A. Bagnell. Eficient reductions for imitation learning. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, 661–668. PMLR, 2010.

[RBL+22] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-Resolution Image Synthesis with Latent Difusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10684–10695. 2022.

[RHR+26] R. Reiter, J. Hoffmann, D. Reinhardt, F. Messerer, K. Baumgärtner, S. Sawant, J. Boedecker, M. Diehl, and S. Gros. Synthesis of model predictive control and reinforcement learning: Survey and classification. Annual Reviews in Control 61, (2026), 101045.

[Ric21] L. Richter. Solving High-Dimensional PDEs, Approximation of Path Space Measures and Importance Sampling of Difusions. Ph.D. thesis, BTU Cottbus-Senftenberg, 2021.

[Roc97] R. T. Rockafellar. Convex analysis, vol. 28. Princeton university press, 1997.

[SDWMG15] J. Sohl-Dickstein, E. A. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings of the 32nd International Conference on Machine Learning (ICML), vol. 37, 2256–2265. JMLR.org, 2015.

[TCA+23] S. Teng, L. Chen, Y. Ai, Y. Zhou, Z. Xuanyuan, and X. Hu. Hierarchical interpretable imitation learning for end-to-end autonomous driving. IEEE Transactions on Intelligent Vehicles (2023). doi:10.1109/TIV.2022. 3225340.

[UZ13] A. S. Üstünel and M. Zakai. Transformation of Measure on Wiener Space. Springer Science & Business Media, 2013.

[Ver18] R. Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science, vol. 47. Cambridge University Press, 2018.

[VMSL25] A. Vuong, M. T. McCann, J. E. Santos, and Y. T. Lin. Are We Really Learning the Score Function? Reinterpreting Difusion Models Through Wasserstein Gradient Flow Matching. Transactions on Machine Learning Research (2025). https://openreview.net/forum?id=CzyJqXQRhJ.

[vOMA23] J. van Oostrum, J. Müller, and N. Ay. Invariance Properties of the Natural Gradient in Overparametrised Systems. Information Geometry 6, no. 1, (2023), 51–67. doi:10.1007/s41884-022-00067-9.

[WS25] C. Wald and G. Steidl. Flow Matching: Markov Kernels, Stochastic Processes and Transport Plans. Variational and Information Flows in Machine Learning and Optimal Transport (2025), 185–254.

[ZC22] Q. Zhang and Y. Chen. Path Integral Sampler: a stochastic control approach for sampling. In International Conference on Learning Representations. 2022.

[ZKLF23] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705 (2023). https://arxiv.org/abs/2304.13705.