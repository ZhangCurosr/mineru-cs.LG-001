# Distribution Steering via Sliced Optimal Transport Control

Kaito Ito Department of Information Physics and Computing The University of Tokyo, Tokyo 113-8654, Japan kaito@g.ecc.u-tokyo.ac.jp

Anqi Dong Department of Decision and Control Systems Department of Mathematics and Digital Futures KTH Royal Institute of Technology, 100 44 Stockholm, Sweden anqid@kth.se

## Abstract

Distribution steering seeks feedback laws that drive the state law of a dynamical system between prescribed initial and terminal distributions. Optimal transport provides a natural geometric approach, but its implementation generally requires a transport map or coupling in the full state space. Sliced optimal transport avoids this full-dimensional construction through one-dimensional projections. Yet, the resulting projected maps specify only directional displacements and do not by themselves prescribe a realizable feedback law. To this end, we develop a finite-horizon control framework based on sliced optimal transport. At each sampling instant, a projected optimal transport map defines a directional terminal condition, whose minimum-energy realization yields a randomized single-direction controller. Averaging over projection directions gives a deterministic sliced feedback. For the single-integrator dynamics, the averaged feedback makes the sliced Wasserstein distance to the target non-increasing. For Gaussian endpoint laws, it is afine, preserves Gaussianity, and steers the mean and covariance to their prescribed terminal values. We further identify a law-dependent gain that yields linear decay of the sliced Wasserstein distance together with an explicit characterization of the control energy. We also prove that the randomized controller converges to the averaged sliced flow as the sampling period vanishes. Finally, we extend the construction to linear dynamical systems. Reachability-normalized coordinates allow instantaneous realization of the sliced velocity for uniformly fully actuated systems, while local controllability Gramians provide exact finite-step realization for general controllable systems. Numerical examples illustrate the resulting distributional flows.

Keywords: Optimal transport, sliced Wasserstein distance, distribution steering, feedback control, linear dynamical systems

## 1 Introduction

Distribution steering seeks to drive the probability law of a dynamical system between prescribed distributions through suitable control inputs. Optimal transport (OT) provides a natural geometric approach to this problem, with origins tracing back to Monge [Mon81] and the subsequent relaxation of Kantorovich [Kan42]. Optimal transport has since developed into a broad theory for comparing, interpolating, and transforming probability distributions [RR98, Vil03, Vil09, PC19]. A distinguishing feature of OT is that it accounts explicitly for displacement in the underlying state space and therefore retains geometric information that is absent from pointwise comparisons of probability densities. For quadratic transportation cost, the Wasserstein distance admits the dynamic formulation of Benamou and Brenier [BB00, DSG24], which identifies transport with the minimum kinetic action required to move a density between prescribed endpoint marginals. In this formulation, the density evolves according to the continuity equation and the velocity field plays the role of a control input. This interpretation has motivated a broad line of work on distribution steering, including Schr¨odinger bridge formulations, covariance steering, Gaussian-mixture steering, and optimal-transport-based predictive control [DSG24, SDG25, CGP16a, CGP16b, CGT19, IK23b, IK23a, IK25]. For controllable linear dynamical systems, distribution steering can further be related to quadratic optimal transport through a transformation involving the finite-horizon reachability Gramian [CGP17].

A common feature of these formulations is their reliance on a transport map or coupling in the full state space. Such an object specifies how mass distributed according to the initial law is reassigned to the target law and thereby determines the displacement to be realized by the controller. This construction is natural from the viewpoint of classical optimal transport. It may, however, become demanding when distributions are represented by large sample sets or when the transport object must be recomputed repeatedly within a feedback loop. Entropic regularization and Sinkhorn-type methods have improved computational scalability, while stochastic and sample-based formulations provide further alternatives [PC19, BCC<sup>+</sup>15, DGT25]. Nevertheless, these approaches still require constructing, approximating, or sampling from a transport object in the ambient space.

Sliced OT ofers a diferent route by replacing the ambient transport problem with a family of one-dimensional problems along linear projections. Since one-dimensional OT admits explicit solutions through cumulative distribution functions and quantiles, the corresponding transport maps and discrepancies can be eficiently evaluated from samples. This construction is closely related to the Radon-transform viewpoint, where a multidimensional object is represented by its collection of linear projections [Dea07]. Sliced Wasserstein distances have been used for barycenter computation and statistical estimation [BRPP15, CD14], as well as for image registration, texture synthesis, and color transfer [PKD07, LSM<sup>+</sup>19, SDGP<sup>+</sup>15]. More recently, they have also played an important role in generative modeling [NH24].

Sliced-Wasserstein flows further show that projected optimal transport maps can generate continuous distributional evolutions by averaging projected displacements [LSM<sup>+</sup>19, CS25]. However, a projected transport map specifies only a directional displacement and does not prescribe how this displacement should be realized by a dynamical system over a prescribed finite horizon. This leaves a gap between projection-based transport and finite-horizon distribution steering. To this end, we develop a finite-horizon control framework based on sliced optimal transport by interpreting projected transport maps as directional terminal conditions. At each sampling instant, a one-dimensional transport map specifies a terminal value along a direction, and the minimum-energy control that reaches this value is lifted to the ambient state space. Repeating this construction with randomly selected directions yields a randomized sliced controller, while its spherical average defines the feedback law analyzed in this paper.

We first study the averaged sliced feedback for the single-integrator dynamics. Proposition 1 derives the variation of the sliced Wasserstein distance along the continuity equation and identifies the associated discrepancy field. For Gaussian endpoint laws, Proposition 2 shows that the feedback preserves Gaussianity and reduces the dynamics to the evolution of the mean and covariance. Theorem 1 proves terminal convergence, while Theorem 2 provides a law-dependent gain with linear sliced-Wasserstein decay and explicit control-energy bounds. We then consider the implementable randomized sliced controller obtained by sampling projection directions. Theorem 3 shows that its sampling-time distributions converge to the averaged sliced flow as the sampling period vanishes, establishing the connection between the discrete controller and the continuous feedback. Finally, we extend the construction to linear dynamical systems. For uniformly fully actuated systems, reachability-normalized coordinates allow the sliced velocity field to be realized instantaneously. For general controllable systems, finite-step displacement matching based on local controllability Gramians reproduces the virtual sliced iteration exactly at the sampling instants, as shown in Proposition 6.

The conference version of this work [ID26] considered Gaussian marginals for the singleintegrator dynamics. Herein, we extend the result by treating weak solutions of the continuity equation under weaker regularity assumptions, proving convergence of the randomized controller, and developing realizations for general linear systems.

Projection-based iterative distribution transfer was introduced in [PKD07] and later connected with sliced-Wasserstein gradient dynamics in [Bon13]. These works show that onedimensional optimal transport along projections can generate distributional evolutions without constructing a full-dimensional transport map. Sliced-Wasserstein flows further developed this idea by studying continuous-time evolutions generated by averaged projected displacements, including their long-time behavior [CS25] and critical-point structure [VKM25].

More recently, stochastic slice matching has been studied as a randomized projectionbased transport procedure. In [LMW23], almost-sure convergence to the target distribution is established, while [TBN26] derives nonasymptotic convergence rates for Gaussian targets using random orthonormal-basis updates. In contrast, we develop a finite-horizon control interpretation of sliced transport, where projected optimal transport maps define directional terminal conditions, and their minimum-energy realizations generate feedback laws for distribution steering. Theorem 3 establishes convergence of the randomized controller to the averaged sliced feedback, linking sampled sliced updates with continuous-time distribution steering.

The remainder of the paper is organized as follows. Section 2 reviews quadratic and sliced optimal transport and formulates the distribution-steering problem. Section 3 develops the sliced controller and its averaged feedback for the single integrator, including the Gaussian case, terminal convergence, and energy characterization. Section 4 proves convergence of the randomized controller to the averaged sliced flow. Section 5 extends the construction to uniformly fully actuated linear systems through reachability-normalized coordinates, while Section 6 develops the finite-step realization for general controllable systems. Numerical examples illustrate the resulting distributional evolutions.

Short proofs are included in the main text when they clarify the development, while longer arguments are deferred to the appendices so as not to interrupt the flow of our presentation.

## 2 Preliminaries and Problem Formulation

## 2.1 Optimal mass transport

Let $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { n } )$ denote the space of Borel probability measures on $\mathbb { R } ^ { n }$ with finite second moment. For an absolutely continuous measure, we slightly abuse notation by using the same symbol for the measure and its density when no confusion arises. For $\mu , \nu \in { \mathcal { P } } _ { 2 } ( \mathbb { R } ^ { n } )$ , the Wasserstein-2 distance is defined by

$$
W _ { 2 } ^ { 2 } ( \mu , \nu ) : = \operatorname* { i n f } _ { \gamma \in \Gamma ( \mu , \nu ) } \int _ { \mathbb { R } ^ { n } \times \mathbb { R } ^ { n } } \| x - y \| ^ { 2 } \gamma ( \mathrm { d } x , \mathrm { d } y ) ,\tag{1}
$$

where $\Gamma ( \mu , \nu ) : = \lbrace \gamma \mid \gamma ( \cdot \times \mathbb { R } ^ { n } ) = \mu , \gamma ( \mathbb { R } ^ { n } \times \cdot ) = \nu \rbrace$ . A coupling specifies how mass distributed according to $\mu$ is transported to $\nu ,$ while the objective in (1) records the expected squared displacement [RR98, Vil03]. In one dimension, the Wasserstein distance admits an explicit quantile representation. Let $F _ { \mu }$ and $F _ { \nu }$ denote the cumulative distribution functions of $\mu , \nu \in \mathcal { P } _ { 2 } ( \mathbb { R } )$ , and let

$$
F _ { \mu } ^ { - 1 } ( z ) = \operatorname* { i n f } \{ s \in \mathbb { R } \mid F _ { \mu } ( s ) \geq z \}
$$

with $z \in ( 0 , 1 )$ , denote the generalized inverse. Then,

$$
W _ { 2 } ^ { 2 } ( \mu , \nu ) = \int _ { 0 } ^ { 1 } \left| F _ { \mu } ^ { - 1 } ( z ) - F _ { \nu } ^ { - 1 } ( z ) \right| ^ { 2 } \mathrm { d } z .\tag{2}
$$

When $\mu$ is atomless, the optimal transport map is the monotone rearrangement

$$
T _ { \mu \to \nu } ( s ) = F _ { \nu } ^ { - 1 } ( F _ { \mu } ( s ) ) .
$$

Problem in (1) also admits the Benamou–Brenier (dynamical) formulation [BB00]. In density form, it reads

$$
\operatorname* { i n f } _ { \rho , v } \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { n } } \frac { 1 } { 2 } \rho ( x , t ) \| v ( x , t ) \| ^ { 2 } \mathrm { d } x \mathrm { d } t .
$$

The continuity equation describes conservation of mass, while the objective measures the kinetic energy of the density flow.

Equivalently, following individual trajectories gives the Lagrangian formulation

$$
\operatorname* { i n f } _ { u : \ \dot { x } = u \atop x ( 0 ) \sim \mu , \ \ x ( 1 ) \sim \nu } \int _ { 0 } ^ { 1 } \mathbb { E } \left[ \frac { 1 } { 2 } \| u ( t ) \| ^ { 2 } \right] \mathrm { d } t .
$$

The Eulerian and Lagrangian formulations describe the same transport process from the density and particle viewpoints, respectively. For linear dynamics ${ \dot { x } } ( t ) = A ( t ) x ( t ) + B ( t ) u ( t )$ the same viewpoint leads to an optimal transport problem with prior dynamics [CGP16a, CGP17, CGP21].

## 2.2 Sliced Wasserstein distance

We next recall the projection operator and the associated sliced Wasserstein distance, see also [PC19, BRPP15, Bon13, PBC<sup>+</sup>20].

Definition 1 (Projection). For $\theta \in { \mathcal { S } } ^ { n - 1 } : = \{ \theta \in \mathbb { R } ^ { n } | \| \theta \| = 1 \}$ , let $P _ { \boldsymbol { \theta } } ( x ) : = { \boldsymbol { \theta } } ^ { \top } { \boldsymbol { x } }$ denote the projection onto the direction θ. For a map $T , T _ { \# } \mu$ denotes the pushforward of $\mu \in { \mathcal { P } } _ { 2 } ( \mathbb { R } ^ { n } )$ , and the projected measure $\mu _ { \theta } : = ( P _ { \theta } ) _ { \# } \mu$ is the distribution of $\bar { \theta ^ { \intercal } } X$ when $X \sim \mu$

Definition 2 (Sliced Wasserstein distance). Let $\mu , \nu \in \mathcal P _ { 2 } ( \mathbb { R } ^ { n } )$ , and let σ denote the uniform probability measure on $S ^ { n - 1 }$ . The sliced Wasserstein-2 distance is defined by

$$
S W _ { 2 } ^ { 2 } ( \mu , \nu ) : = \int _ { S ^ { n - 1 } } W _ { 2 } ^ { 2 } ( \mu _ { \theta } , \nu _ { \theta } ) \sigma ( \mathrm { d } \theta ) .
$$

The sliced Wasserstein distance averages the one-dimensional Wasserstein discrepancies of the projected measures. Using the one-dimensional representation in (2), it can equivalently be written as

$$
S W _ { 2 } ^ { 2 } ( \mu , \nu ) = \int _ { S ^ { n - 1 } } \int _ { 0 } ^ { 1 } \left| F _ { \mu _ { \theta } } ^ { - 1 } ( z ) - F _ { \nu _ { \theta } } ^ { - 1 } ( z ) \right| ^ { 2 } \mathrm { d } z \sigma ( \mathrm { d } \theta ) .
$$

Thus, $S W _ { 2 }$ reduces the ambient transport problem to a family of one-dimensional transport problems along linear projections.

## 2.3 Problem formulation

Consider the controlled linear dynamics

$$
{ \dot { x } } ( t ) = A ( t ) x ( t ) + B ( t ) u ( t ) ,\tag{3}
$$

where $A ( t ) \in \mathbb { R } ^ { n \times n } , B ( t ) \in \mathbb { R } ^ { n \times m }$ , and $u ( t ) \in \mathbb { R } ^ { m }$ . For $t \in [ 0 , 1 ]$ , let $\rho _ { t }$ denote the probability law of $x ( t )$ , with initial condition $x ( 0 ) \sim \rho _ { 0 }$ . A law-dependent feedback is called admissible if the corresponding closed-loop dynamics admit a solution on $t \in [ 0 , 1 )$ and $\rho _ { t } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { n } )$ We write

$$
\mathcal { P } _ { 2 , \mathrm { a c } } ( \mathbb { R } ^ { n } ) : = \{ \rho \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { n } ) \ | \ \rho \ll \mu _ { L } ^ { n } \} ,
$$

where $\mu _ { L } ^ { n }$ denotes the Lebesgue measure on $\mathbb { R } ^ { n }$

Problem 1. Given an initial law $\rho _ { 0 } \in \mathcal { P } _ { 2 , \mathrm { a c } } ( \mathbb { R } ^ { n } )$ and a target law $\rho _ { 1 } \in { \mathcal { P } } _ { 2 } ( \mathbb { R } ^ { n } )$ , construct an admissible law-dependent feedback for (3) that steers the state law from $\rho _ { 0 }$ to $\rho _ { 1 }$ using only one-dimensional optimal transport maps between projections of the current and target laws.

The key feature of the proposed approach is that the feedback law is built from onedimensional projected transport maps, without constructing a transport map or coupling in the ambient space $\mathbb { R } ^ { n }$ . The sliced feedback and its convergence analysis are developed for a single integrator in Sections 3 and 4. Its continuous and finite-step realizations for linear systems are developed in Sections 5 and 6, respectively.

# 3 Sliced OT Control for the Single Integrator

We begin with the single-integrator dynamics

$$
{ \dot { x } } ( t ) = u ( t )
$$

where $x ( t ) , u ( t ) \in \mathbb { R } ^ { n }$ , and the initial state has density $\rho _ { 0 }$ . Under a feedback law $u ( t ) =$ $v ( t , x ( t ) )$ , let $\rho ( t , \cdot )$ denote the density of $x ( t )$ . We assume that $\rho$ satisfies the continuity equation in the weak sense, i.e.,

$$
\partial _ { t } \rho ( t , x ) = - \nabla _ { x } \cdot ( \rho ( t , x ) v ( t , x ) ) ,\tag{4}
$$

on $( 0 , 1 ) \times \mathbb { R } ^ { n }$ , with initial condition $\rho ( 0 , \cdot ) \ = \ \rho _ { 0 }$ . More precisely, for every compactly supported smooth test function $\zeta \in C _ { c } ^ { \infty } ( ( 0 , 1 ) \times \mathbb { R } ^ { n } )$ ，

$$
\int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { n } } \bigl ( \partial _ { t } \zeta ( t , x ) + \nabla _ { x } \zeta ( t , x ) ^ { \top } v ( t , x ) \bigr ) \rho ( t , x ) \mathrm { d } x \mathrm { d } t = 0 .\tag{5}
$$

Given a target density $\rho _ { 1 }$ , the objective is to construct a velocity field v that steers $\rho _ { 0 }$ to $\rho _ { 1 }$ . For comparison, the classical minimum-energy solution is determined by the optimal transport map $\mathcal { T } : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ from $\rho _ { 0 }$ to $\rho _ { 1 } \ [ \mathrm { C G P 1 7 } ]$ . Let $\mathcal { T } _ { 0 : t } ( x ) : = ( 1 - t ) x + t \mathcal { T } ( x )$ denote the displacement interpolation. The associated velocity field is $v ( t , x ) = \mathcal { T } \circ \mathcal { T } _ { 0 : t } ^ { - 1 } ( x ) - \mathcal { T } _ { 0 : t } ^ { - 1 } ( x )$ Moreover, $\begin{array} { r } { W _ { 2 } ^ { 2 } ( \rho _ { 0 } , \rho _ { 1 } ) = \int _ { \mathbb { R } ^ { n } } \| x - \mathcal { T } ( x ) \| ^ { 2 } \rho _ { 0 } ( x ) } \end{array}$ dx. Thus, the classical controller requires an optimal transport map in the full state space. Constructing such a map between continuous distributions is generally dificult in high dimensions. This motivates the sliced formulation developed next.

## 3.1 Iterative sliced controller and averaged feedback

For $\theta \in S ^ { n - 1 }$ , define $\rho _ { t , \theta } : = ( P _ { \theta } ) _ { \# } \rho _ { t }$ , and let $\mathcal { T } _ { t } ^ { \theta } : \mathbb { R }  \mathbb { R }$ denote the monotone optimal transport map from $\rho _ { t , \theta }$ to $\rho _ { 1 , \theta } .$ . Since $\rho _ { t }$ admits a density, so does $\rho _ { t , \theta }$ , and $\mathcal { T } _ { t } ^ { \theta }$ is uniquely defined $\rho _ { t , \theta } .$ -almost everywhere. Moreover,

$$
W _ { 2 } ^ { 2 } \big ( \rho _ { t , \theta } , \rho _ { 1 , \theta } \big ) = \int _ { \mathbb { R } } \big | s - \mathcal { T } _ { t } ^ { \theta } ( s ) \big | ^ { 2 } \rho _ { t , \theta } ( s ) \mathrm { d } s .
$$

We first construct a controller that uses one projection direction at each sampling instant. Let $t _ { k } : = k h$ , for $k \in \{ 0 , 1 , \ldots , N \}$ , where $h > 0$ and $N : = 1 / h$ is assumed to be an integer. The control is held constant over each interval $[ t _ { k } , t _ { k + 1 } )$ , so that the single-integrator dynamics becomes

$$
x _ { k + 1 } = x _ { k } + h u _ { k } .
$$

At time $t _ { k }$ , select a direction $\theta _ { k } \in S ^ { n - 1 }$ and let $\mathcal { T } _ { k } ^ { \theta _ { k } } : = \mathcal { T } _ { t _ { k } } ^ { \theta _ { k } }$ . The projected state $s _ { k } : =$ $\theta _ { k } ^ { \top } x _ { k }$ evolves according to

$$
s _ { \ell + 1 } = s _ { \ell } + h u _ { \theta _ { k } , \ell } ,
$$

where $u _ { \theta _ { k } , \ell }$ is the scalar control along $\theta _ { k }$ . To steer $s _ { k }$ to its transported image $\mathcal { T } _ { k } ^ { \theta _ { k } } ( s _ { k } )$ at the terminal time, consider

$$
\operatorname* { m i n } _ { \{ u _ { \theta _ { k } , \ell } \} } \sum _ { \ell = k } ^ { N - 1 } h u _ { \theta _ { k } , \ell } ^ { 2 } \quad \mathrm { s . t . } \quad s _ { \ell + 1 } = s _ { \ell } + h u _ { \theta _ { k } , \ell }\tag{6}
$$

with $s _ { k } = \theta _ { k } ^ { \top } x _ { k }$ and $s _ { N } = \mathcal { T } _ { k } ^ { \theta _ { k } } ( \theta _ { k } ^ { \top } x _ { k } )$ . The minimum-energy control is constant over the remaining time steps $\ell \in \{ k , \ldots , N - 1 \}$ and is given by

$$
u _ { \theta _ { k } , \ell } ^ { * } = - \frac { 1 } { 1 - t _ { k } } \Big ( \theta _ { k } ^ { \top } x _ { k } - \mathcal { T } _ { k } ^ { \theta _ { k } } ( \theta _ { k } ^ { \top } x _ { k } ) \Big ) .
$$

In a receding-horizon implementation, only the first control value is applied. Lifting it from the projected coordinate to $\mathbb { R } ^ { n }$ gives

$$
u _ { k } = - \frac { \theta _ { k } ^ { \top } x _ { k } - \mathcal { T } _ { k } ^ { \theta _ { k } } ( \theta _ { k } ^ { \top } x _ { k } ) } { 1 - t _ { k } } \theta _ { k } .\tag{7}
$$

At the next sampling instant, the projection direction and the one-dimensional transport map are updated. Repeating this procedure gives the iterative sliced controller.

For the analysis below, we allow a general nonnegative gain $\lambda : [ 0 , 1 )  [ 0 , \infty )$ and write the directional controller as

$$
\begin{array} { r } { u _ { k } = - \lambda ( t _ { k } ) \Big ( \theta _ { k } ^ { \top } x _ { k } - \mathcal T _ { k } ^ { \theta _ { k } } ( \theta _ { k } ^ { \top } x _ { k } ) \Big ) \theta _ { k } . } \end{array}\tag{8}
$$

The choice $\lambda ( t ) = ( 1 - t ) ^ { - 1 }$ recovers (7). More general choices will be useful in the analysis of the averaged flow and its control energy.

Averaging the directional update in (8) over $\theta \sim \sigma$ gives the velocity field

$$
v ( t , x ) = - \lambda ( t ) g _ { t } ( x ) ,\tag{9}
$$

where

$$
g _ { t } ( x ) : = \int _ { \mathcal { S } ^ { n - 1 } } \left( \theta ^ { \top } x - \mathcal { T } _ { t } ^ { \theta } ( \theta ^ { \top } x ) \right) \theta \sigma ( \mathrm { d } \theta ) .\tag{10}
$$

We refer to $g _ { t }$ as the sliced discrepancy field and to (9) as the averaged sliced feedback. The iterative controller (8) is its randomized single-direction realization.

Remark 1. The gain in (7) is the reciprocal of the remaining time and therefore increases as $t _ { k }$ approaches the terminal time. Alternatively, the projected endpoint problem may be solved over a fixed receding horizon $\tau > 0$ . The corresponding control is

$$
u _ { k } = - \frac { 1 } { \tau } \Big ( \theta _ { k } ^ { \top } x _ { k } - \mathcal { T } _ { k } ^ { \theta _ { k } } ( \theta _ { k } ^ { \top } x _ { k } ) \Big ) \theta _ { k } ,
$$

and the state update becomes

$$
x _ { k + 1 } = x _ { k } - \frac { h } { \tau } \Big ( \theta _ { k } ^ { \top } x _ { k } - \mathcal { T } _ { k } ^ { \theta _ { k } } ( \theta _ { k } ^ { \top } x _ { k } ) \Big ) \theta _ { k } .
$$

Thus, $h / \tau$ acts as a step size of a directional descent iteration. Applying several mutually orthogonal directions in succession gives a procedure closely related to iterative distribution transfer [PKD07]. Related combinations of optimal transport and model predictive control for discrete distributions appear in [IK23b, IK23a].

## 3.2 Averaged sliced flow and descent of the sliced Wasserstein distance

The averaged sliced feedback is related to the variation of the sliced Wasserstein distance along the continuity equation. In this subsection, let $\rho _ { t } ( \mathrm { d } x ) \ = \ \rho ( t , x ) \mathrm { d } x$ be a narrowly continuous curve in $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { n } )$ satisfying (4) weakly. We assume that the kinetic energy of v is finite on every compact subinterval of [0, 1), namely,

$$
\int _ { 0 } ^ { \bar { t } } \int _ { \mathbb { R } ^ { n } } \| v ( t , x ) \| ^ { 2 } \rho _ { t } ( \mathrm { d } x ) \mathrm { d } t < \infty
$$

for $\bar { t } \in ( 0 , 1 )$

Proposition 1. The function $r ( t ) : = S W _ { 2 } ( \rho _ { t } , \rho _ { 1 } )$ is locally absolutely continuous on $[ 0 , 1 )$ Moreover, for almost every $t \in ( 0 , 1 )$ , we have $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } S W _ { 2 } ^ { 2 } ( \rho _ { t } , \rho _ { 1 } ) = 2 \int _ { \mathbb { R } ^ { n } } g _ { t } ( x ) ^ { \top } v ( t , x ) \rho _ { t } ( \mathrm { d } x ) } \end{array}$

Proof. See Appendix A.1.

For averaged sliced feedback $v ( t , x ) = - \lambda ( t ) g _ { t } ( x )$ , define

$$
D ( t ) : = \int _ { \mathbb { R } ^ { n } } \| g _ { t } ( x ) \| ^ { 2 } \rho _ { t } ( \mathrm { d } x ) .\tag{11}
$$

Proposition 1 then gives

$$
\frac { \mathrm { d } } { \mathrm { d } t } S W _ { 2 } ^ { 2 } ( \rho _ { t } , \rho _ { 1 } ) = - 2 \lambda ( t ) D ( t ) \le 0\tag{12}
$$

for almost every $t \in ( 0 , 1 )$ . Hence, the sliced Wasserstein distance is non-increasing along the closed-loop flow.

Remark 2 (Directional consistency). Let $a _ { t } ^ { \theta } ( x ) : = \theta ^ { \top } x - \mathcal { T } _ { t } ^ { \theta } ( \theta ^ { \top } x )$ denote the sliced displacement at x along θ. Then

$$
\begin{array} { r l } & { r ( t ) ^ { 2 } = \displaystyle \int _ { \mathbb { R } ^ { n } } \int _ { \mathcal { S } ^ { n - 1 } } \vert a _ { t } ^ { \theta } ( x ) \vert ^ { 2 } \sigma ( \mathrm { d } \theta ) \rho _ { t } ( \mathrm { d } x ) , } \\ & { D ( t ) = \displaystyle \int _ { \mathbb { R } ^ { n } } \Big \Vert \int _ { \mathcal { S } ^ { n - 1 } } a _ { t } ^ { \theta } ( x ) \theta \sigma ( \mathrm { d } \theta ) \Big \Vert ^ { 2 } \rho _ { t } ( \mathrm { d } x ) . } \end{array}
$$

Thus, $r ( t ) ^ { 2 }$ measures the total sliced discrepancy, whereas $D ( t )$ measures the part retained $a f -$ ter the lifted directional displacements are averaged over the sphere. By the Cauchy–Schwarz inequality and $\begin{array} { r } { \int _ { S ^ { n - 1 } } \theta \theta ^ { \top } \sigma ( \mathrm { d } \theta ) = I / n } \end{array}$ 2

$$
D ( t ) \leq { \frac { 1 } { n } } r ( t ) ^ { 2 } .\tag{13}
$$

For $r ( t ) > 0$ , define

$$
\chi ( t ) : = \frac { n } { r ( t ) ^ { 2 } } D ( t ) ,\tag{14}
$$

and set $\chi ( t ) = 1$ when $r ( t ) = 0$ . Then $0 \leq \chi ( t ) \leq 1$ . Values of $\chi ( t )$ near one indicate that the directional displacements reinforce one another, while small values indicate substantial cancellation.

## 3.3 Gaussian marginals and afine feedback

We now take $\rho _ { i } = \mathcal { N } ( \cdot \mid m _ { i } , \Sigma _ { i } ) , i = 0 , 1$ , where $\mathcal { N } ( \cdot \mid m , \Sigma )$ denotes the Gaussian density with mean m and covariance Σ, and $\Sigma _ { i } \succ 0$ . Since linear projections preserve Gaussianity, $\rho _ { i , \theta } = \mathcal { N } ( \cdot \mid \theta ^ { \top } m _ { i } , \theta ^ { \top } \Sigma _ { i } \theta )$ for $i = 0 , 1$ . For $\Sigma \succ 0$ , define

$$
\alpha ( \Sigma , \theta ) : = \sqrt { \frac { \theta ^ { \top } \Sigma _ { 1 } \theta } { \theta ^ { \top } \Sigma \theta } } .
$$

The optimal transport map between one-dimensional Gaussian distributions is afine [PC19, Remark 2.31]. Hence, the map from $\rho _ { 0 , \theta }$ to $\rho _ { 1 , \theta }$ is

$$
\begin{array} { r } { T _ { 0 } ^ { \theta } ( \theta ^ { \top } \boldsymbol { x } ) = \theta ^ { \top } m _ { 1 } + \alpha \big ( \Sigma _ { 0 } , \theta \big ) \left( \theta ^ { \top } \boldsymbol { x } - \theta ^ { \top } m _ { 0 } \right) . } \end{array}
$$

Introduce the matrix-valued functions

$$
\begin{array} { c } { { H ( \Sigma ) : = \displaystyle \int _ { S ^ { n - 1 } } \alpha ( \Sigma , \theta ) \theta \theta ^ { \top } \sigma ( \mathrm { d } \theta ) - \frac { 1 } { n } I , } } \\ { { K ( t , \Sigma ) : = \lambda ( t ) H ( \Sigma ) . } } \end{array}
$$

Using the spherical identity $\begin{array} { r } { \int _ { \mathcal { S } ^ { n - 1 } } \theta \theta ^ { \top } \sigma ( \mathrm { d } \theta ) = I / n } \end{array}$ , the averaged sliced feedback at the initial time becomes

$$
v ( 0 , x ) = K ( 0 , \Sigma _ { 0 } ) x - \lambda ( 0 ) \Bigl [ \bigl ( H ( \Sigma _ { 0 } ) + \frac { 1 } { n } I \bigr ) m _ { 0 } - \frac { 1 } { n } m _ { 1 } \Bigr ] .
$$

Thus, the initial feedback is afine in the state. The following proposition shows that this afine structure is preserved by the averaged sliced flow.

Proposition 2 (Gaussian preservation and afine feedback). For continuous λ : $[ 0 , 1 )  [ 0 , \infty )$ , the covariance equation

$$
\dot { \Sigma } ( t ) = K ( t , \Sigma ( t ) ) \Sigma ( t ) + \Sigma ( t ) K ( t , \Sigma ( t ) ) ^ { \top }\tag{15}
$$

with $\Sigma ( 0 ) = \Sigma _ { 0 }$ admits a unique solution satisfying $\Sigma ( t ) \succ 0$ on [0, 1). Define

$$
m ( t ) : = m _ { 1 } + \exp \Big ( - \frac { 1 } { n } \int _ { 0 } ^ { t } \lambda ( s ) \mathrm { d } s \Big ) \big ( m _ { 0 } - m _ { 1 } \big )\tag{16}
$$

and

$$
\eta ( t ) : = - \lambda ( t ) \Bigl [ \bigl ( H ( \Sigma ( t ) ) + \frac { 1 } { n } I \bigr ) m ( t ) - \frac { 1 } { n } m _ { 1 } \Bigr ] .
$$

With initial law $X ( 0 ) \sim { \mathcal { N } } ( m _ { 0 } , \Sigma _ { 0 } )$ , the afine system

$$
{ \dot { X } } ( t ) = K ( t , \Sigma ( t ) ) X ( t ) + \eta ( t )\tag{17}
$$

admits a unique solution on [0, 1) and satisfies $\mathbf { \boldsymbol { X } } ( t ) \sim \mathcal { N } ( \boldsymbol { m } ( t ) , \Sigma ( t ) )$ . Moreover, $i f g _ { t }$ is computed from $\rho _ { t } = \mathcal { N } ( \cdot \mid m ( t ) , \Sigma ( t ) )$ and the target density $\rho _ { 1 }$ , then

$$
- \lambda ( t ) g _ { t } ( x ) = K ( t , \Sigma ( t ) ) x + \eta ( t ) .
$$

Hence, the averaged sliced feedback is $a f f i n e$ and preserves Gaussianity.

Proof. See Appendix A.2.

## 3.4 Terminal convergence and energy characterization

Proposition 2 shows that the averaged sliced feedback preserves Gaussianity. We first establish conditions under which this Gaussian flow reaches the prescribed terminal law. Since a Gaussian law is determined by its mean and covariance, it is enough to study the limits of $m ( t )$ and Σ(t).

Theorem 1 (Terminal convergence). Denote the Gaussian flow in Proposition $\mathcal { Q }$ as $\rho _ { t } =$ $\mathcal { N } ( \cdot | \ : m ( t ) , \Sigma ( t ) )$ . Suppose that

$$
\int _ { 0 } ^ { 1 } \lambda ( t ) \mathrm { d } t = \infty\tag{18}
$$

and that $\Sigma ( t ) \succeq \alpha I$ on [0, 1) for some $\alpha > 0$ . Set

$$
\beta : = n \Big ( \sqrt { \frac { \mathrm { t r ( } \Sigma _ { 1 } ) } { n } } + r ( 0 ) \Big ) ^ { 2 } \mathrm { a n d ~ } \chi _ { * } : = \operatorname* { m i n } \{ 1 , \frac { 2 \alpha \sqrt { \lambda _ { \operatorname* { m i n } } ( \Sigma _ { 1 } ) } } { \beta ^ { 3 / 2 } ( n + 2 ) } \} ,
$$

then we have

$$
r ( t ) \leq r ( 0 ) \exp \Big ( - \frac { \chi _ { * } } { n } \int _ { 0 } ^ { t } \lambda ( s ) \mathrm { d } s \Big ) .\tag{19}
$$

Consequently, $m ( t )  m _ { 1 }$ and $\Sigma ( t )  \Sigma _ { 1 }$ as $t \nearrow 1$

Proof. See Appendix A.4.

The same afine feedback also governs the first two moments of any law with finite second moment. Thus, under the conditions of Theorem 1, the feedback law $u ( t , x ) =$ $K ( t , \Sigma ( t ) ) x + \eta ( t )$ steers the mean and covariance from $( m _ { 0 } , \Sigma _ { 0 } )$ to $( m _ { 1 } , \Sigma _ { 1 } )$ , even when the state law is not Gaussian.

We next characterize the control efort. The following results apply to general averaged sliced flows and do not require Gaussianity. Let $\rho _ { t }$ be generated by $v ( t , x ) = - \lambda ( t ) g _ { t } ( x )$ and satisfy the assumptions of Proposition 1 on every compact subinterval of [0, 1). For $X ( t ) \sim \rho _ { t }$ , write $u ( t ) = v ( t , X ( t ) )$ and, for $0 < \tau < 1$ , define

$$
\mathcal { E } _ { u } ( 0 , \tau ) : = \int _ { 0 } ^ { \tau } \mathbb { E } \big [ \| u ( t ) \| ^ { 2 } \big ] ~ \mathrm { d } t .
$$

Lemma 1 (Control energy along the sliced flow). Suppose that $D ( t ) > 0$ whenever $r ( t ) > 0$ . Then, for almost every $t \in ( 0 , 1 )$

$$
\mathbb { E } \big [ \| u ( t ) \| ^ { 2 } \big ] = \frac { n \dot { r } ( t ) ^ { 2 } } { \chi ( t ) } .\tag{20}
$$

Consequently,

$$
\mathcal { E } _ { u } ( 0 , \tau ) = n \int _ { 0 } ^ { \tau } \frac { \dot { r } ( t ) ^ { 2 } } { \chi ( t ) } \mathrm { d } t .\tag{21}
$$

$I f r ( t ) \to 0$ as $t \nearrow 1$ , then

$$
\mathcal { E } _ { u } ( 0 , 1 ) \geq n S W _ { 2 } ^ { 2 } ( \rho _ { 0 } , \rho _ { 1 } ) ,\tag{22}
$$

where $\begin{array} { r } { \mathcal { E } _ { u } ( 0 , 1 ) : = \operatorname* { l i m } _ { \tau \nearrow 1 } \mathcal { E } _ { u } ( 0 , \tau ) } \end{array}$

Proof. See Appendix A.5.

Lemma 1 separates the rate at which the sliced discrepancy decreases from the loss caused by averaging over projection directions. For a fixed decay rate, a smaller value of $\chi ( t )$ requires greater control efort.

Combining the dissipation identity in (12) with (14) gives

$$
\dot { r } ( t ) = - \frac { \lambda ( t ) \chi ( t ) } { n } r ( t )
$$

whenever $r ( t ) > 0$ . To obtain linear decay over the remaining horizon, define

$$
\lambda _ { \mathrm { S W } } ( t ) : = { \frac { r ( t ) ^ { 2 } } { D ( t ) ( 1 - t ) } } = { \frac { n } { \chi ( t ) ( 1 - t ) } } , \qquad r ( t ) > 0 .\tag{23}
$$

Here, $( 1 - t ) ^ { - 1 }$ accounts for the remaining time, while $n / \chi ( t )$ compensates for directional cancellation.

Theorem 2 (Linear sliced decay and energy bounds). Suppose that the assumptions of Lemma 1 hold for $\lambda ( t ) = \lambda _ { \mathrm { S W } } ( t )$ and that $r ( 0 ) > 0$ . Then, for every $t \in [ 0 , 1 )$

$$
r ( t ) = ( 1 - t ) r ( 0 ) .\tag{24}
$$

The corresponding total control energy is

$$
\mathcal { E } _ { u } ( 0 , 1 ) = n S W _ { 2 } ^ { 2 } ( \rho _ { 0 } , \rho _ { 1 } ) \int _ { 0 } ^ { 1 } \frac { 1 } { \chi ( t ) } \mathrm { d } t .\tag{25}
$$

$I f \chi ( t ) \geq \underline { { \chi } } > 0$ for almost every $t \in ( 0 , 1 )$ , then

$$
n S W _ { 2 } ^ { 2 } ( \rho _ { 0 } , \rho _ { 1 } ) \leq { \mathcal { E } } _ { u } ( 0 , 1 ) \leq \frac { n } { \underline { { \chi } } } S W _ { 2 } ^ { 2 } ( \rho _ { 0 } , \rho _ { 1 } ) .\tag{26}
$$

Proof. See Appendix A.6.

For Gaussian flows, the averaged sliced field cannot vanish before the target law is reached. The gain in (23) is therefore well defined whenever $r ( t ) > 0$

Corollary 1 (Gaussian flow). Assume that the mean and covariance equations of Proposition 2 with $\lambda ( t ) = \lambda _ { \mathrm { S W } } ( t )$ admit a solution $( m ( t ) , \Sigma ( t ) ) \in \mathbb { R } ^ { n } \times \mathbb { S } _ { > 0 } ^ { n } , \ t \in [ 0 , 1 )$ and that the curve $\rho _ { t } = \mathcal { N } ( \cdot \mid m ( t ) , \Sigma ( t ) )$ satisfies the assumptions of Proposition 1. Then $D ( t ) > 0$ whenever $r ( t ) > 0$ and

$$
\lVert m ( t ) - m _ { 1 } \rVert \leq \sqrt { n } ( 1 - t ) r ( 0 ) ,\tag{27}
$$

$$
\| \Sigma ( t ) - \Sigma _ { 1 } \| _ { \mathrm { F } } \leq \sqrt { \frac { 2 } { \mu } } ( 1 - t ) r ( 0 ) ,\tag{28}
$$

where $\mu > 0$ is defined in (56). Moreover, $\Sigma ( t ) \succeq \alpha I$ on [0, 1) for some $\alpha > 0$ , and

$$
n S W _ { 2 } ^ { 2 } ( \rho _ { 0 } , \rho _ { 1 } ) \leq { \mathcal E } _ { u } ( 0 , 1 ) \leq \frac { n } { \chi _ { * } } S W _ { 2 } ^ { 2 } ( \rho _ { 0 } , \rho _ { 1 } ) ,
$$

where $\chi _ { * }$ is defined as in Theorem 1 using this value of α.

Proof. See Appendix A.7.

## 4 Convergence of the Iterative Sliced Controller

We now show that the iterative sliced controller approaches the averaged sliced flow as the sampling period tends to zero. To isolate the approximation due to random slicing, we restrict attention to the single-integrator dynamics.

For a current distribution $\rho$ and a direction $\theta \in S ^ { n - 1 }$ , write

$$
\begin{array} { r } { \displaystyle g _ { \theta } [ \rho ] ( x ) : = \left( \theta ^ { \top } x - \mathcal { T } _ { \rho } ^ { \theta } ( \theta ^ { \top } x ) \right) \theta , } \\ { \displaystyle \bar { g } [ \rho ] ( x ) : = \int _ { S ^ { n - 1 } } g _ { \theta } [ \rho ] ( x ) \sigma ( \mathrm { d } \theta ) , } \end{array}
$$

where $\mathcal { T } _ { \rho } ^ { \theta }$ is the one-dimensional optimal transport map from $\rho _ { \theta }$ to $\rho _ { 1 , \theta }$ . Thus, $g _ { t } = \bar { g } [ \rho _ { t } ]$ when $\rho = \rho _ { t }$

Fix $\bar { t } \in ( 0 , 1 )$ and assume that λ is continuously diferentiable on $[ 0 , \bar { t } ]$ . Set $\Lambda _ { \bar { t } } : =$ $\mathrm { m a x } _ { 0 \leq t \leq \bar { t } } \lambda ( t )$ . Let $x _ { 0 } ~ \sim ~ \rho _ { 0 }$ be defined on a probability space $\left( \Omega _ { 0 } , \mathcal { F } _ { 0 } , \mathbb { P } _ { 0 } \right)$ , write $\mathbb { E } _ { 0 }$ for expectation, and set $\lVert X \rVert _ { L ^ { 2 } ( \Omega _ { 0 } ) } ^ { 2 } : = \mathbb { E } _ { 0 } [ \lVert X \rVert ^ { 2 } ]$ . On a separate probability space, let $\{ \theta _ { k } \} _ { k \ge 0 }$ be an i.i.d. sequence with common distribution σ, independent of $x _ { 0 }$ . For a sampling period $h > 0$ , set $t _ { k } : = k h$ and $K _ { h } : = \lfloor \bar { t } / h \rfloor$

Let $\Theta _ { k }$ denote the sigma-algebra generated by $\big \{ \theta _ { 0 } , \dots , \theta _ { k - 1 } \big \}$ , representing the direction history before time $t _ { k }$ . We write $\rho _ { k } ^ { h } : = \operatorname { L a w } ( x _ { k } ^ { h } \mid \Theta _ { k } )$ for the conditional distribution of $x _ { k } ^ { h } .$ leaving its dependence on $\Theta _ { k }$ implicit. The iterative sliced controller generates

$$
x _ { k + 1 } ^ { h } = x _ { k } ^ { h } - h \lambda ( t _ { k } ) g _ { \theta _ { k } } [ \rho _ { k } ^ { h } ] ( x _ { k } ^ { h } ) ,\tag{29}
$$

with $x _ { 0 } ^ { h } = x _ { 0 }$ . Equivalently, for each realization of the direction sequence,

$$
\rho _ { k + 1 } ^ { h } = \left( \operatorname { I d } - h \lambda ( t _ { k } ) g _ { \theta _ { k } } [ \rho _ { k } ^ { h } ] \right) _ { \# } \rho _ { k } ^ { h } ,\tag{30}
$$

where Id denotes the identity map.

For the averaged flow, we make the following assumption.

Assumption 1 (Existence). There exists a W -continuous curve $\{ \rho _ { t } \} _ { t \in [ 0 , \bar { t } ] } \subset \mathcal { P } _ { 2 , \mathrm { a c } } ( \mathbb { R } ^ { n } )$ and a Borel measurable map $\Phi _ { t } : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ such that $\Phi _ { 0 } = \mathrm { I d }$ and, for every $x \in \mathbb { R } ^ { n }$ , the curve $t \mapsto \Phi _ { t } ( x )$ is absolutely continuous on $[ 0 , \bar { t } ]$ and satisfies, for almost every $t \in ( 0 , \bar { t } )$ and for ρ<sub>0</sub>-almost every x,

$$
\frac { \mathrm { d } } { \mathrm { d } t } \Phi _ { t } ( x ) = - \lambda ( t ) \bar { g } [ \rho _ { t } ] ( \Phi _ { t } ( x ) )
$$

with $\rho _ { t } = ( \Phi _ { t } ) _ { \# } \rho _ { 0 }$

Assumption 1 fixes a reference averaged flow in Lagrangian form. It is a well-posedness assumption for the comparison flow. Set $y ( 0 ) = x ( 0 )$ and define $y ( t ) : = \Phi _ { t } ( x ( 0 ) )$ . Then, $y ( t )$ has law $\rho _ { t }$ and satisfies

$$
\frac { \mathrm { d } } { \mathrm { d } t } y ( t ) = - \lambda ( t ) \bar { g } [ \rho _ { t } ] ( y ( t ) ) .\tag{31}
$$

Our objective is to compare $\rho _ { k } ^ { h }$ with $\rho _ { t _ { k } }$ as $h \searrow 0$

For $\mu \in { \mathcal { P } } _ { 2 } ( \mathbb { R } ^ { n } )$ , write

$$
m _ { 2 } ( \mu ) : = ( \int _ { \mathbb { R } ^ { n } } \| x \| ^ { 2 } \mathrm { d } \mu ( x ) ) ^ { 1 / 2 } .
$$

The moment estimates established below show that the iterative and averaged distributions remain in the class

$$
\begin{array} { r } { \mathcal { K } _ { \bar { t } } : = \left\{ \rho \in \mathcal { P } _ { 2 , \mathrm { a c } } ( \mathbb { R } ^ { n } ) \ \middle \vert \ m _ { 2 } ( \rho ) \leq \big ( m _ { 2 } ( \rho _ { 0 } ) + m _ { 2 } ( \rho _ { 1 } ) \big ) \mathrm { e } ^ { \Lambda _ { \bar { t } } \bar { t } } \right\} . } \end{array}
$$

To compare the two evolutions, we require stability along the conditional laws generated by the iterative sliced controller and along the averaged sliced flow.

Assumption 2 (Stability along the generated flows). There exist constants $h _ { \bar { t } } \in ( 0 , 1 ]$ and $L _ { \bar { t } } <$ ∞ such that $h _ { \bar { t } } \Lambda _ { \bar { t } } < 1$ and

1. for any $h \in ( 0 , h _ { \bar { t } } ] , \ k \in \{ 0 , \ldots , K _ { h } \}$ , and almost every realization of the direction history,

$$
\left\| \bar { g } [ \rho _ { k } ^ { h } ] ( x _ { k } ^ { h } ) - \bar { g } [ \rho _ { t _ { k } } ] ( y ( t _ { k } ) ) \right\| _ { L ^ { 2 } ( \Omega _ { 0 } ) } \leq L _ { \bar { t } } \left\| x _ { k } ^ { h } - y ( t _ { k } ) \right\| _ { L ^ { 2 } ( \Omega _ { 0 } ) } .
$$

2. for any $s , t \in [ 0 , \bar { t } ]$ ，

$$
\| \bar { g } [ \rho _ { s } ] ( y ( s ) ) - \bar { g } [ \rho _ { t } ] ( y ( t ) ) \| _ { L ^ { 2 } ( \Omega _ { 0 } ) } \leq L _ { \bar { t } } \| y ( s ) - y ( t ) \| _ { L ^ { 2 } ( \Omega _ { 0 } ) } .
$$

We are now ready to state the convergence result. The expectation below is taken with respect to the sampled directions.

Theorem 3 (Vanishing-step convergence). Suppose that Assumptions 1 and 2 hold. Then, there exists a constant $C _ { \bar { t } } < \infty$ , independent of h, such that, for every $h \in ( 0 , h _ { \bar { t } } ]$

$$
\mathbb { E } _ { \{ \theta _ { k } \} } \left[ \operatorname* { m a x } _ { 0 \le k \le K _ { h } } W _ { 2 } ^ { 2 } \big ( \rho _ { k } ^ { h } , \rho _ { t _ { k } } \big ) \right] \le C _ { \bar { t } } h ,\tag{32}
$$

$$
\mathbb { E } _ { \{ \theta _ { k } \} } \left[ \operatorname* { m a x } _ { 0 \le k \le K _ { h } } S W _ { 2 } ^ { 2 } \big ( \rho _ { k } ^ { h } , \rho _ { t _ { k } } \big ) \right] \le \frac { C _ { \bar { t } } } { n } h .\tag{33}
$$

For a fixed $t \in [ 0 , \bar { t } ]$ , let $k _ { h } : = \lfloor t / h \rfloor$ . Then, we have

$$
\operatorname* { l i m } _ { h \searrow 0 } \mathbb { E } _ { \{ \theta _ { k } \} } \bigl [ W _ { 2 } ^ { 2 } \bigl ( \rho _ { k _ { h } } ^ { h } , \rho _ { t } \bigr ) \bigr ] = 0 .
$$

Thus, the iterative controller converges to the averaged sliced flow with an $O ( h )$ expected squared Wasserstein error. The proof proceeds through uniform moment estimates, a decomposition of each randomized update into its averaged drift and a centered fluctuation, and a discrete stability argument. We establish these ingredients next and return to the proof of Theorem 3 thereafter.

## 4.1 Estimates for the convergence proof

Throughout this subsection, we retain the notation and assumptions introduced above. In particular, the analysis is carried out on [0, t<sup>¯</sup>], where $\lambda \in C ^ { 1 } ( [ 0 , \bar { t } ] )$ ). We also set

$$
\Lambda _ { \bar { t } } ^ { \prime } : = \operatorname* { m a x } _ { 0 \leq t \leq \bar { t } } | \dot { \lambda } ( t ) | .
$$

The randomized sliced iteration can be viewed as a stochastic Euler approximation of the averaged flow. The random projection direction produces a centered fluctuation, while time discretization introduces a local Euler residual. We first record an abstract estimate for these two efects and then show that it applies to the sliced dynamics.

Proposition 3 (Stochastic Euler estimate). Let H be a Hilbert space and $b : [ 0 , \bar { t } ] \times \mathcal { H } $ H. Suppose that $z _ { 0 } ^ { h } = z ( 0 ) \in \mathcal { H }$ and

$$
\begin{array} { c } { { z _ { k + 1 } ^ { h } = z _ { k } ^ { h } + h b ( t _ { k } , z _ { k } ^ { h } ) + h \xi _ { k + 1 } , } } \\ { { z ( t _ { k + 1 } ) = z ( t _ { k } ) + h b ( t _ { k } , z ( t _ { k } ) ) + r _ { k } . } } \end{array}
$$

Assume that $\| r _ { k } \| _ { \mathcal { H } } \le B h ^ { 2 }$ and, for every $k \in \{ 0 , \ldots , K _ { h } \}$ 2

$$
\| b ( t _ { k } , z _ { k } ^ { h } ) - b ( t _ { k } , z ( t _ { k } ) ) \| _ { \mathcal { H } } \leq L \| z _ { k } ^ { h } - z ( t _ { k } ) \| _ { \mathcal { H } } .\tag{34}
$$

Moreover, let $\{ \xi _ { k + 1 } \}$ be an H-valued martingale diference sequence with respect to $\{ \mathcal { F } _ { k } \}$ satisfying $\mathbb { E } [ \xi _ { k + 1 } \ | \ \mathcal { F } _ { k } ] = 0$ and $\mathbb { E } [ \| \xi _ { k + 1 } \| _ { \mathcal { H } } ^ { 2 } \mid \mathcal { F } _ { k } ] \le \sigma _ { \xi } ^ { 2 }$ . Then there exists $C < \infty$ , independent of $h _ { i }$ , such that

$$
\mathbb { E } \big [ \operatorname* { m a x } _ { 0 \leq k \leq K _ { h } } \| z _ { k } ^ { h } - z ( t _ { k } ) \| _ { \mathcal { H } } ^ { 2 } \big ] \leq C h ,\tag{35}
$$

for every $h \in ( 0 , 1 ]$

Proof. Set $e _ { k } : = z _ { k } ^ { h } - z ( t _ { k } )$ . Since $e _ { 0 } = 0$ , summing the error recursion gives

$$
e _ { j } = h \sum _ { i = 0 } ^ { j - 1 } \bigl ( b ( t _ { i } , z _ { i } ^ { h } ) - b ( t _ { i } , z ( t _ { i } ) ) \bigr ) + h \sum _ { i = 0 } ^ { j - 1 } \xi _ { i + 1 } - \sum _ { i = 0 } ^ { j - 1 } r _ { i } .
$$

For $j \le k$ , the trajectory-wise stability condition (34) and the Cauchy–Schwarz inequality give

$$
\Big \| h \sum _ { i = 0 } ^ { j - 1 } \big ( b ( t _ { i } , z _ { i } ^ { h } ) - b ( t _ { i } , z ( t _ { i } ) ) \big ) \Big \| _ { \mathcal { H } } ^ { 2 } \leq \bar { t } L ^ { 2 } h \sum _ { i = 0 } ^ { k - 1 } \operatorname* { m a x } _ { 0 \leq r \leq i } \| e _ { r } \| _ { \mathcal { H } } ^ { 2 } .
$$

Moreover, $\begin{array} { r } { \| \sum _ { i = 0 } ^ { j - 1 } r _ { i } \| _ { \mathcal { H } } \leq B \bar { t } h } \end{array}$ . Hence,

$$
\operatorname* { m a x } _ { 0 \leq j \leq k } \| e _ { j } \| _ { \mathcal { H } } ^ { 2 } \leq 3 L ^ { 2 } \bar { t } h \sum _ { i = 0 } ^ { k - 1 } \operatorname* { m a x } _ { 0 \leq r \leq i } \| e _ { r } \| _ { \mathcal { H } } ^ { 2 } + 3 \operatorname* { m a x } _ { 0 \leq j \leq k } \left\| h \sum _ { i = 0 } ^ { j - 1 } \xi _ { i + 1 } \right\| _ { \mathcal { H } } ^ { 2 } + 3 B ^ { 2 } \bar { t } ^ { 2 } h ^ { 2 } .\tag{36}
$$

By Doob’s $L ^ { 2 }$ inequality and the martingale-diference property,

$$
\mathbb { E } \Big [ \operatorname* { m a x } _ { 0 \leq j \leq k } \big | \big | h \sum _ { i = 0 } ^ { j - 1 } \xi _ { i + 1 } \big | \big | _ { \mathcal { H } } ^ { 2 } \Big ] \leq 4 \sigma _ { \xi } ^ { 2 } \bar { t } h .\tag{37}
$$

Indeed, the cross terms vanish and

$$
\mathbb { E } \Big [ \big \| h \sum _ { i = 0 } ^ { k - 1 } \xi _ { i + 1 } \big \| _ { \mathcal { H } } ^ { 2 } \Big ] = h ^ { 2 } \sum _ { i = 0 } ^ { k - 1 } \mathbb { E } \big [ \| \xi _ { i + 1 } \| _ { \mathcal { H } } ^ { 2 } \big ] \leq \sigma _ { \xi } ^ { 2 } \bar { t } h .
$$

Set $a _ { k } : = \mathbb { E } [ \operatorname* { m a x } _ { 0 \leq j \leq k } \| e _ { j } \| _ { \mathcal { H } } ^ { 2 } ]$ . Taking expectations in (36) and using (37) gives

$$
a _ { k } \leq 3 L ^ { 2 } \bar { t } h \sum _ { i = 0 } ^ { k - 1 } a _ { i } + 1 2 \sigma _ { \xi } ^ { 2 } \bar { t } h + 3 B ^ { 2 } \bar { t } ^ { 2 } h ^ { 2 } .
$$

The discrete Gronwall inequality proves (35). One may take $C = \mathrm { e } ^ { 3 L ^ { 2 } \bar { t } ^ { 2 } } ( 1 2 \sigma _ { \xi } ^ { 2 } \bar { t } + 3 B ^ { 2 } \bar { t } ^ { 2 } )$ .

We next establish the estimates needed to apply Proposition 3 to the sliced dynamics. For a measurable function $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ , write $\begin{array} { r } { \| \bar { f } \| _ { L _ { \rho } ^ { 2 } } ^ { 2 } : = \bar { \int _ { \mathbb { R } ^ { n } } } \| f ( x ) \| ^ { 2 } \mathrm { d } \rho ( x ) } \end{array}$

Lemma 2 (Norm of a sliced displacement). For every $\rho \in \mathcal { P } _ { 2 , \mathrm { a c } } ( \mathbb { R } ^ { n } )$ and $\theta \in S ^ { n - 1 }$ , we have $\| g _ { \theta } [ \rho ] \| _ { L _ { o } ^ { 2 } } ^ { 2 } = W _ { 2 } ^ { 2 } ( \rho _ { \theta } , \rho _ { 1 , \theta } )$ . Consequently, $\| g _ { \theta } [ \rho ] \| _ { L _ { \rho } ^ { 2 } } \leq m _ { 2 } ( \rho ) + m _ { 2 } ( \rho _ { 1 } )$

Proof. See Appendix B.1.

Lemma 3 (Preservation of absolute continuity). Let $\rho \in \mathcal { P } _ { 2 , \mathrm { a c } } ( \mathbb { R } ^ { n } ) , \ \theta \in \mathcal { S } ^ { n - 1 }$ , and $\gamma \in [ 0 , 1 )$ . Then the map

$$
T _ { \rho , \theta , \gamma } ( x ) : = x - \gamma g _ { \theta } [ \rho ] ( x )\tag{38}
$$

satisfies $( T _ { \rho , \theta , \gamma } ) _ { \# } \rho \in { \mathcal { P } } _ { 2 , \operatorname { a c } } ( \mathbb { R } ^ { n } )$ . Consequently, the laws $\rho _ { k } ^ { h }$ generated by (30) with $h \in ( 0 , h _ { \bar { t } } ]$ are absolutely continuous for every $\textit { k } \leq \textit { K } _ { h }$ , for almost every realization of the direction sequence.

Proof. See Appendix B.2.

Lemma 4 (Uniform moment bounds). For almost every realization of the direction sequence,

$$
m _ { 2 } ( \rho _ { k } ^ { h } ) \leq \bigl ( m _ { 2 } ( \rho _ { 0 } ) + m _ { 2 } ( \rho _ { 1 } ) \bigr ) \mathrm { e } ^ { \Lambda _ { \bar { t } } \bar { t } } , \qquad 0 \leq k \leq K _ { h } .\tag{39}
$$

In addition, suppose that Assumption 1 holds. Then, the same bound holds for $m _ { 2 } ( \rho _ { t } )$ on [0, t<sup>¯</sup>]. Consequently, the iterative and averaged laws remain in ${ \boldsymbol { \kappa } } _ { \bar { t } }$ .

Proof. See Appendix B.3.

The preceding estimates also give uniform control of the directional and averaged sliced fields. Set $M _ { \bar { t } } : = \bigl ( m _ { 2 } ( \rho _ { 0 } ) + m _ { 2 } ( \rho _ { 1 } ) \bigr ) \mathrm { e } ^ { \Lambda _ { \bar { t } } \bar { t } } + m _ { 2 } ( \rho _ { 1 } )$

Lemma 5 (Uniform bounds for the sliced fields). For every $\rho \in { \mathcal { K } } _ { \bar { t } }$ ,

$$
\| \bar { g } [ \rho ] \| _ { L _ { \rho } ^ { 2 } } ^ { 2 } \leq \int _ { S ^ { n - 1 } } \| g _ { \theta } [ \rho ] \| _ { L _ { \rho } ^ { 2 } } ^ { 2 } \sigma ( \mathrm { d } \theta ) \leq M _ { \bar { t } } ^ { 2 } .\tag{40}
$$

Proof. See Appendix B.4.

It remains to verify the stability of the averaged drift and the local consistency of the averaged flow. For $X \in L ^ { 2 } ( \Omega _ { 0 } ; \mathbb { R } ^ { n } )$ , let $\mathbb { P } _ { X }$ denote its law and define

$$
b ( t , X ) : = - \lambda ( t ) \bar { g } [ \mathbb { P } _ { X } ] ( X ) .\tag{41}
$$

Lemma 6 (Stability and local consistency). Suppose that Assumptions 1 and 2 hold. Then, for every $h \in ( 0 , h _ { \bar { t } } ]$ and $k \in \{ 0 , \ldots , K _ { h } \}$

$$
\| b ( t _ { k } , x _ { k } ^ { h } ) - b ( t _ { k } , y ( t _ { k } ) ) \| _ { L ^ { 2 } ( \Omega _ { 0 } ) } \leq \Lambda _ { \bar { t } } L _ { \bar { t } } \| x _ { k } ^ { h } - y ( t _ { k } ) \| _ { L ^ { 2 } ( \Omega _ { 0 } ) }\tag{42}
$$

for almost every realization of the direction history. Moreover, whenever $0 \leq t \leq t + h \leq \bar { t } .$ the averaged flow satisfies

$$
y ( t + h ) = y ( t ) + h b ( t , y ( t ) ) + r _ { t , h } ,\tag{43}
$$

where $\| r _ { t , h } \| _ { L ^ { 2 } ( \Omega _ { 0 } ) } \leq B _ { \bar { t } } h ^ { 2 }$ and $\begin{array} { r } { B _ { \bar { t } } : = \frac { M _ { \bar { t } } } { 2 } \big ( \Lambda _ { \bar { t } } ^ { \prime } + \Lambda _ { \bar { t } } ^ { 2 } L _ { \bar { t } } \big ) } \end{array}$

Proof. See Appendix B.5.

## 4.2 Proof of Theorem 3

We now assemble the preceding estimates. Let $\mathcal { H } : = L ^ { 2 } ( \Omega _ { 0 } ; \mathbb { R } ^ { n } )$ and $\| X \| _ { \mathcal { H } } ^ { 2 } : = \mathbb { E } _ { 0 } [ \| X \| ^ { 2 } ]$ recall the drift b defined in (41), and for $X ~ \in ~ \mathcal { H }$ with $\mathbb { P } _ { X } ~ \in ~ { \cal K } _ { \bar { t } } .$ define $B ( t , X , \theta ) : =$ $- \lambda ( t ) g _ { \theta } [ \mathbb { P } _ { X } ] ( X )$ . By Lemmas 3 and 4, whenever $h \in ( 0 , h _ { \bar { t } } ]$ , the random laws $\rho _ { k } ^ { h }$ belong to ${ \ K } _ { \bar { t } }$ for all $k \leq K _ { h }$ , almost surely with respect to the sampled directions. The averaged laws $\rho _ { t _ { k } }$ also belong to ${ \ K } _ { \bar { t } }$

We first identify the random sliced iteration with the stochastic Euler scheme in Proposition 3. The recursion (29) can be written as $x _ { k + 1 } ^ { h } = x _ { k } ^ { h } + h b ( t _ { k } , x _ { k } ^ { h } ) + h \xi _ { k + 1 }$ , where $\xi _ { k + 1 } : = B ( t _ { k } , x _ { k } ^ { h } , \theta _ { k } ) - b ( t _ { k } , x _ { k } ^ { h } )$ . Since $\theta _ { k }$ is independent of $\Theta _ { k }$ and has distribution $\sigma ,$ $\mathbb { E } _ { \{ \theta _ { k } \} } \big [ \xi _ { k + 1 } \ | \ \Theta _ { k } \big ] = 0$ . Thus, $\{ \xi _ { k + 1 } \}$ is a martingale diference sequence with respect to the filtration generated by the sampled directions.

It remains to bound its conditional second moment. We have

$$
\begin{array} { r } { \| \xi _ { k + 1 } \| _ { \mathcal H } ^ { 2 } \leq \Lambda _ { \bar { t } } ^ { 2 } \left\| g _ { \theta _ { k } } [ \rho _ { k } ^ { h } ] ( x _ { k } ^ { h } ) - \bar { g } [ \rho _ { k } ^ { h } ] ( x _ { k } ^ { h } ) \right\| _ { \mathcal H } ^ { 2 } \leq 2 \Lambda _ { \bar { t } } ^ { 2 } \left\| g _ { \theta _ { k } } [ \rho _ { k } ^ { h } ] ( x _ { k } ^ { h } ) \right\| _ { \mathcal H } ^ { 2 } + 2 \Lambda _ { \bar { t } } ^ { 2 } \left\| \bar { g } [ \rho _ { k } ^ { h } ] ( x _ { k } ^ { h } ) \right\| _ { \mathcal H } ^ { 2 } . } \end{array}
$$

Taking conditional expectation and using Lemma 5 gives

$$
\begin{array} { r } { \mathbb { E } _ { \{ \theta _ { k } \} } \left[ \left\| \xi _ { k + 1 } \right\| _ { \mathcal { H } } ^ { 2 } \mid \Theta _ { k } \right] \le 2 \Lambda _ { \vec { t } } ^ { 2 } \mathbb { E } _ { \{ \theta _ { k } \} } \left[ \left\| g _ { \theta _ { k } } [ \rho _ { k } ^ { h } ] ( x _ { k } ^ { h } ) \right\| _ { \mathcal { H } } ^ { 2 } \mid \Theta _ { k } \right] + 2 \Lambda _ { \vec { t } } ^ { 2 } \left\| \bar { g } [ \rho _ { k } ^ { h } ] ( x _ { k } ^ { h } ) \right\| _ { \mathcal { H } } ^ { 2 } \le 4 \Lambda _ { \vec { t } } ^ { 2 } M _ { \vec { t } } ^ { 2 } . } \end{array}
$$

Hence, the martingale-noise condition in Proposition 3 holds with $\sigma _ { \xi } = 2 \Lambda _ { \bar { t } } M _ { \bar { t } }$

We next verify the deterministic part of the Euler estimate. By Lemma 6, the trajectorywise stability condition (34) holds for $z _ { k } ^ { h } = x _ { k } ^ { h }$ and $z ( t _ { k } ) = y ( t _ { k } )$ with $L = \Lambda _ { \bar { t } } L _ { \bar { t } }$ . Moreover, the trajectory under the averaged controller satisfies

$$
y ( t _ { k + 1 } ) = y ( t _ { k } ) + h b ( t _ { k } , y ( t _ { k } ) ) + r _ { k } , \quad \| r _ { k } \| _ { \mathcal { H } } \leq B _ { \bar { t } } h ^ { 2 } .
$$

Proposition 3, applied with $z _ { k } ^ { h } = x _ { k } ^ { h }$ and $z ( t _ { k } ) = y ( t _ { k } )$ , therefore yields

$$
\begin{array} { r } { \mathbb { E } _ { \{ \theta _ { k } \} } \Big [ \operatorname* { m a x } _ { 0 \le k \le K _ { h } } \| x _ { k } ^ { h } - y ( t _ { k } ) \| _ { L ^ { 2 } ( \Omega _ { 0 } ) } ^ { 2 } \Big ] \le C ( \bar { t } , L , B _ { \bar { t } } , \sigma _ { \xi } ) h . } \end{array}\tag{44}
$$

We now transfer this estimate from the lifted variables to their probability laws. For each fixed realization of the direction sequence, $\ v { x } _ { k } ^ { h }$ has law $\rho _ { k } ^ { h }$ under $\mathbb { P } _ { 0 }$ , while $y ( t _ { k } )$ has law $\rho _ { t _ { k } }$ Hence, their joint law provides a coupling, and therefore

$$
W _ { 2 } ^ { 2 } \big ( \rho _ { k } ^ { h } , \rho _ { t _ { k } } \big ) \leq \| x _ { k } ^ { h } - y ( t _ { k } ) \| _ { L ^ { 2 } ( \Omega _ { 0 } ) } ^ { 2 } .
$$

Taking the maximum over $k$ , then expectation with respect to the sampled directions, and using (44), we obtain

$$
\mathbb { E } _ { \{ \theta _ { k } \} } \left[ \operatorname* { m a x } _ { 0 \leq k \leq K _ { h } } W _ { 2 } ^ { 2 } \big ( \rho _ { k } ^ { h } , \rho _ { t _ { k } } \big ) \right] \leq C ( \bar { t } , L , B _ { \bar { t } } , \sigma ) h
$$

that proves (32). The sliced Wasserstein estimate (33) follows immediately from $S W _ { 2 } ^ { 2 } ( \rho , \eta ) \leq$ $\textstyle { \frac { 1 } { n } } W _ { 2 } ^ { 2 } ( \rho , \eta )$

It remains to establish convergence at an arbitrary fixed time $t \in [ 0 , \bar { t } ]$ . Let $k _ { h } : = \lfloor t / h \rfloor$ Since $| t _ { k _ { h } } - t | \le h$ , the temporal Lipschitz estimate (64) gives

$$
W _ { 2 } ( \rho _ { t _ { k _ { h } } } , \rho _ { t } ) \leq \| y ( t _ { k _ { h } } ) - y ( t ) \| _ { L ^ { 2 } ( \Omega _ { 0 } ) } \leq \Lambda _ { \bar { t } } M _ { \bar { t } } h .
$$

By the triangle inequality,

$$
\begin{array} { r l } & { W _ { 2 } ^ { 2 } \big ( \rho _ { k _ { h } } ^ { h } , \rho _ { t } \big ) \leq \Big ( W _ { 2 } \big ( \rho _ { k _ { h } } ^ { h } , \rho _ { t _ { k _ { h } } } \big ) + W _ { 2 } \big ( \rho _ { t _ { k _ { h } } } , \rho _ { t } \big ) \Big ) ^ { 2 } \leq 2 W _ { 2 } ^ { 2 } \big ( \rho _ { k _ { h } } ^ { h } , \rho _ { t _ { k _ { h } } } \big ) + 2 W _ { 2 } ^ { 2 } \big ( \rho _ { t _ { k _ { h } } } , \rho _ { t } \big ) } \\ & { \qquad \leq \underset { 0 \leq k \leq K _ { h } } { \operatorname* { m a x } } W _ { 2 } ^ { 2 } \big ( \rho _ { k } ^ { h } , \rho _ { t _ { k } } \big ) + 2 W _ { 2 } ^ { 2 } \big ( \rho _ { t _ { k _ { h } } } , \rho _ { t } \big ) . } \end{array}
$$

Taking expectations and using the preceding estimates, we obtain

$$
\mathbb { E } _ { \{ \theta _ { k } \} } \left[ W _ { 2 } ^ { 2 } \big ( \rho _ { k _ { h } } ^ { h } , \rho _ { t } \big ) \right] \leq 2 C ( \bar { t } , L , B _ { \bar { t } } , \sigma _ { \xi } ) h + 2 \Lambda _ { \bar { t } } ^ { 2 } M _ { \bar { t } } ^ { 2 } h ^ { 2 } .
$$

Consequently, lim $h \setminus 0  \mathbb { E } _ { \{ \theta _ { k } \} } \big [ W _ { 2 } ^ { 2 } \big ( \rho _ { \lfloor t / h \rfloor } ^ { h } , \rho _ { t } \big ) \big ] = 0 .$

## 5 Sliced OT Control for Fully Actuated Linear Systems

We now extend the sliced optimal transport controller from the single integrator to a linear time-varying system. The main dificulty is that the control input no longer prescribes the state velocity directly, as the motion is also shaped by the drift and the time-varying input directions. We therefore introduce coordinates that remove the drift and normalize the control authority over the steering horizon.

## 5.1 Drift removal and reachability normalization

Consider the linear system

$$
{ \dot { x } } ( t ) = A ( t ) x ( t ) + B ( t ) u ( t ) .\tag{45}
$$

Assumption 3. The matrices $A ( t )$ and $B ( t )$ are continuous on $[ 0 , 1 ]$ , and $B ( t )$ has full row rank for every $t \in [ 0 , 1 ]$

Let $\Phi ( t , s )$ denote the state-transition matrix associated with $A ( \cdot )$ . To remove the drift, set $F _ { t } : = \Phi ( 1 , t )$ and introduce $y ( t ) : = F _ { t } x ( t )$ . Since $\dot { F } _ { t } = - F _ { t } A ( t )$ , the transformed state satisfies

$$
\begin{array} { r } { \dot { y } ( t ) = F _ { t } B ( t ) u ( t ) . } \end{array}
$$

The reachability Gramian over [0, 1] is

$$
G _ { 1 0 } : = \int _ { 0 } ^ { 1 } \Phi ( 1 , s ) B ( s ) B ( s ) ^ { \top } \Phi ( 1 , s ) ^ { \top } \mathrm { d } s .
$$

Assumption 3 implies that $G _ { 1 0 } \succ 0$ . Let $\begin{array} { r } { L : = G _ { 1 0 } ^ { - 1 / 2 } } \end{array}$ and introduce the reachabilitynormalized state $z ( t ) : = L y ( t ) = L F _ { t } x ( t )$ . Its dynamics are

$$
\dot { z } ( t ) = C _ { t } u ( t ) ,
$$

where $C _ { t } : = L F _ { t } B ( t )$ . Defining $Q _ { t } : = C _ { t } C _ { t } ^ { \top }$ , we have $\begin{array} { r } { \int _ { 0 } ^ { 1 } Q _ { t } \mathrm { d } t = L G _ { 1 0 } L ^ { \top } = I } \end{array}$ . Thus, the transformed coordinates remove the drift and normalize the aggregate control authority over the steering horizon. Under Assumption $3 , C _ { t }$ has full row rank and $Q _ { t } \succ 0$ for every $t \in [ 0 , 1 ]$ . Since $Q _ { t }$ is continuous, there exist constants $q _ { - } > 0$ and $q _ { + } < \infty$ such that

$$
q _ { - } I \preceq Q _ { t } \preceq q _ { + } I\tag{46}
$$

for every $t \in [ 0 , 1 ]$

## 5.2 Feedback realization of the sliced flow

We now construct a feedback law that makes the normalized state $z ( t ) = L F _ { t } x ( t )$ follow the averaged sliced flow of the single integrator. Let $\rho _ { t }$ denote the law of $x ( t )$ . The corresponding law of $z ( t )$ is $\widehat { \rho } _ { t } : = ( L F _ { t } ) _ { \# } \rho _ { t }$ . Since $F _ { 1 } = I$ , the target law in the normalized coordinates is $\widehat { \rho } _ { 1 } = L _ { \# } \rho _ { 1 }$

For each $\theta \in S ^ { n - 1 }$ , let ${ \widehat { T } } _ { t } ^ { \theta }$ be the one-dimensional optimal transport map from $( P _ { \theta } ) _ { \# } { \widehat { \rho } } _ { t }$ to $( P _ { \theta } ) _ { \# } { \widehat { \rho } } _ { 1 }$ . Define the associated sliced discrepancy field by

$$
\widehat { g } _ { t } ( z ) : = \int _ { \mathcal { S } ^ { n - 1 } } \left( \theta ^ { \top } z - \widehat { \mathcal T } _ { t } ^ { \theta } ( \theta ^ { \top } z ) \right) \theta \sigma ( \mathrm { d } \theta ) .
$$

For a continuous gain $\lambda ( t )$ , the desired velocity in the normalized coordinates is $- \lambda ( t ) \widehat { g } _ { t } ( z )$ Since $\dot { z } ( t ) = C _ { t } u ( t )$ , the control must satisfy

$$
\begin{array} { r } { C _ { t } u ( t ) = - \lambda ( t ) \widehat { g } _ { t } ( z ( t ) ) . } \end{array}
$$

Under Assumption $3 , C _ { t }$ has full row rank and $Q _ { t } = C _ { t } C _ { t } ^ { \top }$ is invertible. Thus, $C _ { t } ^ { \top } Q _ { t } ^ { - 1 }$ is a right inverse of $C _ { t }$ , and the minimum-norm control realizing the prescribed velocity is

$$
\boldsymbol { u } ( t ) = - \lambda ( t ) C _ { t } ^ { \top } Q _ { t } ^ { - 1 } \widehat { \boldsymbol { g } } _ { t } \big ( L F _ { t } \boldsymbol { x } ( t ) \big ) .\tag{47}
$$

Substitution into the normalized dynamics gives ${ \dot { z } } ( t ) = - \lambda ( t ) { \widehat { g } } _ { t } ( z ( t ) )$ . Hence, the normalized state follows exactly the averaged sliced flow developed for the single integrator, and its terminal convergence can be transferred to the original coordinates.

Proposition 4. Suppose that Assumption 3 holds and that the conditions of Theorem 1 are satisfied for the transformed initial and target laws. Then, under the controller $( 4 7 )$ $\begin{array} { r } { \operatorname* { l i m } _ { t \nearrow 1 } m ( t ) = m _ { 1 } } \end{array}$ and $\dim _ { t \nearrow 1 } \Sigma ( t ) = \Sigma _ { 1 }$

Proof. Since the normalized state follows ${ \dot { z } } ( t ) = - \lambda ( t ) { \widehat { g } } _ { t } ( z ( t ) )$ , the law of $z ( t )$ follows the averaged sliced flow with target $L _ { \# } \rho _ { 1 }$ . Let $m _ { z } ( t )$ and $\Sigma _ { z } ( t )$ denote the mean and covariance of $z ( t )$ . The transformed target law has mean $L m _ { 1 }$ and covariance $L \Sigma _ { 1 } L ^ { \top }$ . Proposition 2 and Theorem 1 therefore give $\dim _ { t \nearrow 1 } m _ { z } ( t ) = L m _ { 1 }$ . Likewise, we have lim $\smash { _ { t \nearrow 1 } \sum _ { z } ( t ) = L \Sigma _ { 1 } L ^ { \top } }$

The relation $z ( t ) = L F _ { t } x ( t )$ gives $m _ { z } ( t ) = L F _ { t } m ( t )$ Since L and $F _ { t }$ are invertible, $m ( t ) = F _ { t } ^ { - 1 } L ^ { - 1 } m _ { z } ( t )$ Continuity of the state-transition matrix and $F _ { 1 } = I$ imply that $\begin{array} { r } { \operatorname* { l i m } _ { t \nearrow 1 } F _ { t } ^ { - 1 } = I } \end{array}$ . Taking the limit in the preceding identity gives $\begin{array} { r } { \operatorname* { l i m } _ { t \nearrow 1 } m ( t ) = m _ { 1 } } \end{array}$ . The covariances satisfy

$$
\Sigma _ { z } ( t ) = L F _ { t } \Sigma ( t ) F _ { t } ^ { \top } L ^ { \top }
$$

Equivalently, it holds that

$$
\Sigma ( t ) = F _ { t } ^ { - 1 } L ^ { - 1 } \Sigma _ { z } ( t ) L ^ { - \top } F _ { t } ^ { - \top }
$$

Taking the limit and using lim $\mathbf { \nabla } _ { t \mathcal { A } } F _ { t } ^ { - 1 } = I$ gives $\dim _ { t \nearrow 1 } \Sigma ( t ) = \Sigma _ { 1 }$

## 5.3 Energy bounds

The preceding construction establishes terminal convergence through the transformed state $z ( t )$ . We now examine how the energy of this sliced flow is reflected in the control efort required by the original linear system. The relation between the two is governed by the matrices $Q _ { t }$

Define the sliced discrepancy in the transformed coordinates by $S ( t ) : = S W _ { 2 } ^ { 2 } \big ( \widehat { \rho } _ { t } , \widehat { \rho } _ { 1 } \big )$ Equivalently, $S ( t ) = S W _ { 2 } ^ { 2 } \big ( ( L F _ { t } ) _ { \# } \rho _ { t } , L _ { \# } \rho _ { 1 } \big )$ . As in Proposition 1, its derivative along the transformed density flow is

$$
\dot { S } ( t ) = 2 \mathbb { E } \big [ \widehat { g } _ { t } ( \boldsymbol { z } ( t ) ) ^ { \top } \dot { \boldsymbol { z } } ( t ) \big ] .\tag{48}
$$

Under the controller (47), the transformed dynamics satisfy

$$
\dot { z } ( t ) = - \lambda ( t ) \widehat { g } _ { t } ( z ( t ) ) .
$$

Substitution into (48) gives

$$
\begin{array} { r } { \dot { S } ( t ) = - 2 \lambda ( t ) \mathbb { E } \left[ \lVert \widehat { g } _ { t } ( z ( t ) ) \rVert ^ { 2 } \right] . } \end{array}
$$

Hence, $S ( t )$ is non-increasing along the transformed density evolution.

We distinguish the physical input energy from the kinetic energy of the transformed flow defined by

$$
\mathcal { E } _ { z } ( 0 , t ) : = \int _ { 0 } ^ { t } \mathbb { E } \big [ \| \dot { z } ( s ) \| ^ { 2 } \big ] ~ \mathrm { d } s .
$$

Since $z ( t )$ follows the averaged sliced flow, the following proposition compares the physical input energy with the kinetic energy of the transformed flow.

Proposition 5 (Input-energy bounds). Suppose that Assumption 3 holds. Then, under the controller in (47), we have that, for any $t \in [ 0 , 1 ]$ ，

$$
\frac { 1 } { q _ { + } } \mathcal { E } _ { z } ( 0 , t ) \leq \mathcal { E } _ { u } ( 0 , t ) \leq \frac { 1 } { q _ { - } } \mathcal { E } _ { z } ( 0 , t ) .
$$

Proof. Under the controller in (47),

$$
\| u ( t ) \| ^ { 2 } = \lambda ( t ) ^ { 2 } \widehat { g } _ { t } ( z ( t ) ) ^ { \top } Q _ { t } ^ { - 1 } \widehat { g } _ { t } ( z ( t ) ) = \dot { z } ( t ) ^ { \top } Q _ { t } ^ { - 1 } \dot { z } ( t ) .\tag{49}
$$

The bounds in (46) imply $\begin{array} { r } { \frac { 1 } { q _ { + } } \| \xi \| ^ { 2 } \leq \xi ^ { \top } Q _ { t } ^ { - 1 } \xi \leq \frac { 1 } { q _ { - } } \| \xi \| ^ { 2 } } \end{array}$ for every $\xi \in \mathbb { R } ^ { n }$ and $t \in [ 0 , 1 ]$ Applying this inequality to $\xi = \dot { z } ( t )$ and using (49) yields $\begin{array} { r } { \frac { 1 } { q _ { + } } \| \dot { { \boldsymbol z } } ( t ) \| ^ { 2 } \leq \| { \boldsymbol u } ( t ) \| ^ { 2 } \leq \frac { 1 } { q _ { - } } \| \dot { { \boldsymbol z } } ( t ) \| ^ { 2 } } \end{array}$ Taking expectations and integrating over [0, t] gives the desired bounds. □

Proposition 5 shows that the physical control energy is equivalent, up to the constants $q _ { - }$ <sub>−</sub> and $q _ { + }$ , to the kinetic energy of the transformed sliced flow. These constants quantify the distortion introduced when the averaged sliced flow is realized through time-varying input directions of the original system.

## 6 Algorithm for General Controllable Systems

The construction in Section 5 relies on uniform full actuation. When $B ( t )$ has full row rank, the matrix $Q _ { t }$ is uniformly positive definite, and any prescribed velocity in the transformed coordinates can be realized instantaneously. The transformed dynamics can therefore be made to coincide with the averaged sliced flow of the single integrator, leading to terminal convergence and corresponding energy bounds.

For a general controllable system, however, $B ( t )$ need not have full row rank. The matrix $Q _ { t }$ may then be singular, and an arbitrary transformed velocity cannot, in general, be realized at each instant. Thus, the continuous-time feedback construction of Section 5 is no longer directly applicable.

Controllability, nevertheless, ensures that a prescribed displacement can be realized over a finite time interval. This suggests replacing instantaneous velocity matching with finite-step displacement matching. Accordingly, over each interval of a finite partition, we first determine a sliced displacement and then construct a control input that realizes this displacement exactly by the end of the interval.

## 6.1 Finite-step sliced update

Consider a partition $0 = t _ { 0 } < t _ { 1 } < \cdots < t _ { N } = 1$ . We retain the fixed transformation introduced in Section 5. For each interval $J _ { k } = [ t _ { k } , t _ { k + 1 } ]$ , define the local lifted Gramian by $\begin{array} { r } { \mathcal { G } _ { k } = \int _ { t _ { k } } ^ { t _ { k + 1 } } C _ { s } C _ { s } ^ { \top } } \end{array}$ ds.

Assumption 4 (Local controllability). The system is controllable on each interval of the partition, i.e., the controllability Gramian satisfies $\mathcal { G } _ { k } \succ 0$ for every k.

Assumption 4 requires the transformed system to be controllable over each interva $[ t _ { k } , t _ { k + 1 } ]$ . It does not require $C _ { t }$ to have full row rank at every time. Thus, although an arbitrary transformed velocity may not be realizable instantaneously, any prescribed finite displacement can be realized over the interval. For a time-invariant controllable pair $( A , B )$ the condition holds on every interval of positive length.

At time $t _ { k }$ , the transformed state and its law are $z _ { t _ { k } } = L F _ { t _ { k } } x ( t _ { k } )$ and $\widehat { \rho } _ { k } = ( L F _ { t _ { k } } ) _ { \# } \rho _ { t _ { k } }$ respectively. The transformed target law remains fixed and is given by $\widehat { \rho } _ { 1 } = L _ { \# } \rho _ { 1 }$

Let $\Xi _ { k } : \mathbb { R } ^ { n } \to \mathbb { R } ^ { n }$ denote a sliced update map constructed from $\widehat { \rho } _ { k }$ and $\widehat { \rho } _ { 1 }$ . For a direction $\theta _ { k } \in S ^ { n - 1 }$ , one possible choice is

$$
\Xi _ { k } ( z ) = z - \alpha _ { k } \Big ( \theta _ { k } ^ { \top } z - \widehat { \mathcal { T } } _ { k } ^ { \theta _ { k } } ( \theta _ { k } ^ { \top } z ) \Big ) \theta _ { k } ,
$$

where $\alpha _ { k }$ is a step size and $\widehat { T } _ { k } ^ { \theta _ { k } }$ is the one-dimensional optimal transport map between the corresponding projected laws. Alternatively, using the averaged sliced discrepancy $\widehat { g } _ { k }$ , one may take

$$
\Xi _ { k } ( z ) = z - \alpha _ { k } \widehat { g } _ { k } ( z ) .
$$

The desired displacement is $\delta _ { k } ( z ) : = \Xi _ { k } ( z ) - z$ . Given the sampled state $z _ { t _ { k } }$ , we apply over the interval $J _ { k }$ the input

$$
u ( t ) = C _ { t } ^ { \top } \mathcal { G } _ { k } ^ { - 1 } \delta _ { k } ( z _ { t _ { k } } ) .\tag{50}
$$

Equivalently,

$$
u ( t ) = B ( t ) ^ { \top } F _ { t } ^ { \top } L ^ { \top } \mathcal { G } _ { k } ^ { - 1 } \delta _ { k } ( z _ { t _ { k } } )
$$

in original coordinates. Here, $z _ { t _ { k } } = L F _ { t _ { k } } x ( t _ { k } )$ , and the input is applied for $t \in J _ { k }$

Proposition 6 (Finite-step exactness). Suppose that Assumption 4 holds. Under the input (50), the transformed state satisfies $z _ { t _ { k + 1 } } = \Xi _ { k } ( z _ { t _ { k } } )$ . Consequently, $\widehat { \rho } _ { k + 1 } = ( \Xi _ { k } ) _ { \# } \widehat { \rho } _ { k }$

Proof. Over the interval $J _ { k } ,$ the transformed dynamics satisfy $\dot { z } ( t ) = C _ { t } u ( t )$ . Therefore,

$$
z _ { t _ { k + 1 } } = z _ { t _ { k } } + \int _ { t _ { k } } ^ { t _ { k + 1 } } C _ { s } u ( s ) \mathrm { d } s = z _ { t _ { k } } + \int _ { t _ { k } } ^ { t _ { k + 1 } } C _ { s } C _ { s } ^ { \top } \mathrm { d } s \mathcal { G } _ { k } ^ { - 1 } \delta _ { k } ( z _ { t _ { k } } ) = z _ { t _ { k } } + \delta _ { k } ( z _ { t _ { k } } ) = \Xi _ { k } ( z _ { t _ { k } } ) .
$$

The law update is thus $\widehat { \rho } _ { k + 1 } = ( \Xi _ { k } ) _ { \# } \widehat { \rho } _ { k }$

Proposition 6 shows that, at the sampling instants, the transformed law follows exactly the virtual sliced iteration $\widehat { \rho } _ { k + 1 } = ( \Xi _ { k } ) _ { \# } \widehat { \rho } _ { k }$ . Thus, the linear dynamics introduce no additional approximation at these instants. Any approximation arises solely from the choice of the sliced update $\Xi _ { k }$

## 6.2 Algorithmic form

The finite-step construction is summarized in Algorithm 1.

Algorithm 1 Finite-step sliced steering   
Require: Initial law $\rho _ { 0 }$ and target law $\rho _ { 1 }$   
Require: Partition $0 = t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot < t _ { N } = 1$   
1: for $k = 0 , \ldots , N - 1$ do   
2: Compute $z _ { t _ { k } } = L F _ { t _ { k } } x ( t _ { k } )$ and $\widehat { \rho } _ { k } = ( L F _ { t _ { k } } ) _ { \# } \rho _ { t _ { k } }$   
3: Construct sliced update map $\Xi _ { k }$ from $\widehat { \rho } _ { k }$ to $\widehat { \rho } _ { 1 }$   
4: Compute $\begin{array} { r } { \mathcal G _ { k } = \int _ { t _ { k } } ^ { t _ { k + 1 } } L F _ { t } B ( \bar { t } ) B ( t ) ^ { \top } F _ { t } ^ { \top } L ^ { \top } \mathrm { d } t } \end{array}$   
5: Apply, for $t \in [ t _ { k } , t _ { k + 1 } ) , u ( t ) = B ( t ) ^ { \top } F _ { t } ^ { \top } L ^ { \top } \mathcal { G } _ { k } ^ { - 1 } \left( \Xi _ { k } ( z _ { t _ { k } } ) - z _ { t _ { k } } \right)$   
6: end for

This procedure is the finite-step counterpart of the iterative sliced controller for a general controllable linear system. At each sampling instant, projected transport maps are recomputed from the current transformed law, and the resulting sliced displacement is realized exactly over the following interval. The construction is intrinsically tied to a finite partition and should not be interpreted as a continuous-time extension of the fully actuated result. When $B ( t )$ is rank deficient, the matrix $Q _ { t } = C _ { t } C _ { t } ^ { \top }$ is singular. Although the local Gramian $\mathcal { G } _ { k }$ may remain positive definite on every interval, it can become increasingly ill-conditioned as the interval length decreases. Consequently, refining the partition does not, in general, produce a finite-energy continuous-time limit.

## 7 Numerical Experiments

All numerical experiments were implemented in MATLAB R2025b and performed on a MacBook equipped with an Apple M1 Pro chip and 16 GB of RAM. The experiments are designed to illustrate two aspects of the proposed framework: the distribution steering capability of the randomized sliced controller and its realization through linear dynamical systems.

## 7.1 Example 1: Gaussian mixtures

We first illustrate the randomized sliced controller on a non-Gaussian distribution-steering problem in $\mathbb { R } ^ { 2 }$ . The initial and target laws are three-component Gaussian mixtures, $\rho _ { i } =$ $\textstyle \sum _ { j = 1 } ^ { 3 } w _ { i , j } { \mathcal { N } } ( m _ { i , j } , \Sigma _ { i , j } )$ for $i \in \{ 0 , 1 \}$ , with weights $w _ { 0 } = ( 0 . 4 5 , 0 . 3 5 , 0 . 2 0 )$ and $w _ { 1 } = ( 0 . 4 0 , 0 . 3 0 , 0 . 3 0 )$ The component means are given by the columns of

$$
M _ { 0 } = \left[ \begin{array} { c c c } { { - 2 . 8 } } & { { - 0 . 9 } } & { { - 2 . 2 } } \\ { { 0 . 2 } } & { { 2 . 4 } } & { { - 2 . 0 } } \end{array} \right] , \quad M _ { 1 } = \left[ \begin{array} { c c c } { { 1 . 2 } } & { { 3 . 0 } } & { { 1 . 7 } } \\ { { 1 . 6 } } & { { - 0 . 3 } } & { { - 2 . 0 } } \end{array} \right] ,
$$

that is, $M _ { i } = [ m _ { i , 1 } , m _ { i , 2 } , m _ { i , 3 } ]$ , where each $m _ { i , j } \in \mathbb { R } ^ { 2 }$ is a column vector.

The corresponding covariance matrices are

$$
\begin{array} { r l } & { \{ \Sigma _ { 0 , j } \} _ { j = 1 } ^ { 3 } = \left\{ \left[ \begin{array} { l l } { 0 . 3 5 } & { 0 . 1 0 } \\ { 0 . 1 0 } & { 0 . 2 5 } \end{array} \right] , \left[ \begin{array} { l l } { 0 . 2 0 } & { - 0 . 0 6 } \\ { - 0 . 0 6 } & { 0 . 3 5 } \end{array} \right] , \left[ \begin{array} { l l } { 0 . 2 2 } & { 0 } \\ { 0 } & { 0 . 1 8 } \end{array} \right] \right\} , } \\ & { \{ \Sigma _ { 1 , j } \} _ { j = 1 } ^ { 3 } = \left\{ \left[ \begin{array} { l l } { 0 . 2 5 } & { - 0 . 0 5 } \\ { - 0 . 0 5 } & { 0 . 2 0 } \end{array} \right] , \left[ \begin{array} { l l } { 0 . 3 0 } & { 0 . 0 9 } \\ { 0 . 0 9 } & { 0 . 2 5 } \end{array} \right] , \left[ \begin{array} { l l } { 0 . 1 8 } & { 0 } \\ { 0 } & { 0 . 3 2 } \end{array} \right] \right\} . } \end{array}
$$

Both laws are represented by $N = 1 0 ^ { 4 }$ particles, and the target samples are drawn once and fixed throughout the experiment. At iteration $k ,$ a direction $\theta _ { k }$ is sampled uniformly from $S ^ { 1 }$ . The empirical one-dimensional transport map is obtained by projecting the samples onto $\theta _ { k }$ , sorting the projected values, and matching them by rank. Using the remaininghorizon gain $\lambda ( t ) = ( 1 - t ) ^ { - 1 }$ , we use the logarithmically refined grid $t _ { k } = 1 - \mathrm { e } ^ { - 0 . 0 0 8 k }$ for $k = 0 , \ldots , 2 2 5 0$

Figure 1 shows representative particle trajectories and density contours during the controlled evolution. The initial mixture is continuously deformed toward the target law, with mass originating from one component potentially redistributed among several target components.

Figure 2 shows the empirical sliced Wasserstein distance to the target, estimated using 96 projection directions. Its decay is consistent with the dissipation property in Proposition 1. Near the terminal time, the remaining discrepancy is comparable to the empirical discrepancy between two independent samples from the target law.

## 7.2 Example 2: Color-distribution steering through fully actuated dynamics

We next illustrate the realization of sliced distribution steering through fully actuated dynamics in a color-transfer problem. The initial and target laws are extracted from two oil paintings, a classical application of optimal transport in imaging (see, e.g., [PC19, SDGP<sup>+</sup>15]). The initial image is Monet’s Haystack, while the target color distribution is extracted from Water Lilies. This example demonstrates the fully actuated realization developed in Section 5 and verifies the associated energy bounds.

![](images/b3e51935aab7b0ce1fc6248ba49d4443fb95c37b4b8aa8e81e683d225d71558d.jpg)

Figure 1: Randomized sliced steering between two Gaussian mixtures. Thin curves show 120 representative particle trajectories, while contours show the controlled density at t = 0, 0.25, 0.5, 0.75, 1. Mass originating from one initial component may be distributed among several target components.  
![](images/e61487c5699fc25262eb17fb3b0131b900a695d973d27dfb61276a9045870504.jpg)  
Figure 2: Empirical sliced Wasserstein distance to the target over time, estimated using 96 projection directions. The final computed point satisfies $1 - t _ { 2 2 5 0 } \approx 1 . 5 \times 1 0 ^ { - 8 }$ and is identified with the terminal time.

Each pixel is represented by its CIELAB color vector $x = [ L ^ { * } , a ^ { * } , b ^ { * } ] ^ { \top } \in \mathbb { R } ^ { 3 }$ . The spatial locations of pixels are fixed, and only the color distribution evolves. Thus, the problem concerns color-distribution steering rather than spatial deformation of the image. For numerical conditioning, we use the normalized coordinates $x = [ ( L ^ { * } - 5 0 ) / 5 0 , \ a ^ { * } / 1 1 0 , \ b ^ { * } / 1 1 0 ] ^ { \top }$ . Both images are resized so that their largest dimension is 160 pixels. All pixels of the initial image are used as particles, and the target quantiles are computed from the complete set of

Target painting

target-image pixels.

The color particles evolve according to the fully actuated linear system (45), with

$$
A = \left[ \begin{array} { c c c } { { - 0 . 0 8 } } & { { - 0 . 2 2 } } & { { 0 . 0 6 } } \\ { { 0 . 1 8 } } & { { - 0 . 0 6 } } & { { 0 . 0 3 } } \\ { { 0 . 0 2 } } & { { - 0 . 0 5 } } & { { - 0 . 1 0 } } \end{array} \right]
$$

and

$$
B ( t ) = \left[ \begin{array} { c c c } { { 1 . 1 0 + 0 . 1 2 \sin ( 2 \pi t ) } } & { { 0 . 0 5 \cos ( 2 \pi t ) } } & { { 0 } } \\ { { 0 . 0 3 \sin ( 2 \pi t ) } } & { { 0 . 9 5 + 0 . 1 0 \cos ( 2 \pi t ) } } & { { 0 . 0 4 \sin ( 4 \pi t ) } } \\ { { 0 . 0 2 \cos ( 2 \pi t ) } } & { { 0 . 0 3 \sin ( 2 \pi t ) } } & { { 1 . 0 5 + 0 . 0 8 \sin ( 4 \pi t ) \vphantom { \frac { 1 } { \cos ( 2 \pi t ) } } } } \end{array} \right] .
$$

The matrix $B ( t )$ has full row rank on [0, 1], and hence Assumption 3 holds. The reachability Gramian $G _ { 1 0 }$ is evaluated by the trapezoidal rule using 2001 quadrature points. The matrices $L = G _ { 1 0 } ^ { - 1 / 2 }$ and $Q _ { t }$ are then computed as in Section 5, giving the uniform actuation bounds $q _ { - } = 0 . 7 8 1 5$ and $q _ { + } = 1 . 2 8 5 2$

The interval [0, 1] is divided into 60 uniform steps. At each step, the spherical average in the sliced controller is approximated using 32 fixed directions on $S ^ { 2 }$ , distributed approximately uniformly by a Fibonacci lattice. The projected optimal transport maps are obtained by sorting the projected color samples and matching empirical quantiles. The minimum-norm input $u = C _ { t } ^ { \top } Q _ { t } ^ { - 1 } \dot { z }$ then realizes the sliced velocity through the physical dynamics. Since the projection directions are fixed, the implementation is deterministic.

Figure 3 shows the two paintings and their empirical color distributions in the $( a ^ { * } , b ^ { * } )$ plane. The initial palette is concentrated around pale blue, gray, and ochre tones, whereas the target distribution contains stronger purple, green, and pink components. The controller moves the color distribution toward the target while preserving the spatial arrangement of the initial image.

Initial painting  
![](images/9b7408f8c43b639d61c8990bd7d69e4def7b65cf203f60c33586fa327f2a99f6.jpg)

![](images/63154472dfa2b137e774d4e5a9fd88e1a857133eff304cac0429723769c2c713.jpg)

![](images/cccec60827bd1aae7aff883aa6c0eaae200f94b49970d1d167ea982524990313.jpg)  
Figure 3: Initial and target paintings and their empirical color distributions in the $( a ^ { * } , b ^ { * } )$ plane. The initial painting provides the spatial structure and initial color distribution, while the target painting provides the desired color distribution.

Figure 4 shows the resulting evolution at six equally spaced times. The pale palette of the initial painting is gradually replaced by the purple, green, and blue tones of the target. Since the controller acts only on color coordinates, the spatial structure remains unchanged. Over the horizon, the sliced distance to the target decreases from 0.299 to 0.066.

![](images/86ba3f2bccab12590b1954adb263dd2647fba98c9a076f43d435d32972d18288.jpg)  
Figure 4: Color-distribution steering through fully actuated linear dynamics. Snapshots are shown at $t = 0 / 5 , 1 / 5 , \dots , 5 / 5$ . The spatial structure of the initial painting is preserved, while its color distribution is steered toward that of the target painting.

We next verify the energy characterization associated with the reachability normalization. Figure 5 shows the accumulated input energy $\mathcal { E } _ { u } ( 0 , t )$ together with the bounds obtained from the transformed energy $\mathcal { E } _ { z } ( 0 , t )$ . Throughout the horizon,

$$
\frac { \mathcal { E } _ { z } ( 0 , t ) } { q _ { + } } \leq \mathcal { E } _ { u } ( 0 , t ) \leq \frac { \mathcal { E } _ { z } ( 0 , t ) } { q _ { - } } .
$$

Thus, the physical energy required to realize the sliced flow through the original dynamics is controlled by the actuation bounds in (46).

![](images/55afc6a9869440a25f5aca60f3b92768653220bd2d3fb1897ee6fc4b1cde3a10.jpg)  
Figure 5: Accumulated physical input energy and the corresponding lower and upper bounds. The physical energy $\mathcal { E } _ { u } ( 0 , t )$ remains between $\mathcal { E } _ { z } ( 0 , t ) / q _ { + }$ and $\mathcal { E } _ { z } ( 0 , t ) / q .$ <sub>−</sub> throughout the steering horizon.

Finally, we compare the sliced controller with classical optimal transport as an energy benchmark. The exact optimal transport map between the transformed endpoint distributions gives the minimum-energy displacement interpolation. Since its computation requires solving an assignment problem over an $N \times N$ cost matrix, the comparison is performed on a common subsample of $N = 1 5 0 0$ pixels from each image. On this subsample, the physical control energy of the sliced controller is 1.31 times the optimal transport minimum, indicating a moderate loss in energy optimality.

Remark 3 (Computational Complexity). Let N denote the number of particles, $N _ { \mathrm { d i r } }$ the number of projection directions, and K the number of time steps. A sliced update requires $O ( N _ { \mathrm { d i r } } N$ log N) operations and $O ( N _ { \mathrm { d i r } } N )$ memory, leading to $O ( K N _ { \mathrm { d i r } } N$ log N) operations over the full horizon. In contrast, exact transport between equally weighted samples requires a dense $N \times N$ cost matrix and an assignment solve, with $\Theta ( N ^ { 2 } )$ memory and cubic worst-case complexity.

For Figure 6, we use $N _ { \mathrm { d i r } } = 3 2$ and $K = 6 0$ . One exact map at $N = 2 0 0 0$ takes 15.3 seconds, whereas the complete sliced run at the full image resolution $N = 2 0 { , } 4 8 0$ takes 1.1 seconds. At full resolution, the dense cost matrix alone requires approximately 3.4 GB, compared with about 5.2 MB for the projected arrays. These timings compare one exact endpoint map with a complete feedback run and illustrate their diferent scaling behavior.

![](images/fb67bcaecdfaf22535821375aaee87bf0d4df5adccb3ccbee3e6619b2d21de0f.jpg)  
Figure 6: Runtime and memory scaling of sliced and exact optimal transport. Solid lines show measured times for one exact endpoint map and a complete 60-step sliced run. Dashed lines show the storage required by the dense $N \times N$ cost matrix and the $N _ { \mathrm { d i r } } \times N$ projected arrays. Exact transport becomes impractical beyond a few thousand particles, while the sliced controller extends to the full image resolution.

## 8 Conclusion

We developed a finite-horizon framework for distribution steering based on one-dimensional optimal transport along linear projections. A sampled projection direction defines a scalar endpoint-control problem, whose minimum-energy realization yields the iterative sliced controller. For the single-integrator dynamics, its averaged flow decreases the sliced Wasserstein discrepancy, admits a Gaussian specialization, and satisfies an explicit energy characterization. We further proved that the randomized controller converges to the averaged sliced flow as the sampling period vanishes. The construction was extended to linear dynamical systems. Reachability-normalized coordinates enable instantaneous realization of the sliced velocity for uniformly fully actuated systems, while local controllability provides exact finite-step realization for general controllable systems. These results show that projected transport maps can provide a tractable and dynamically realizable alternative to full-dimensional optimal transport. Future directions include adaptive projection selection, partial observations, and stochastic, constrained, or nonlinear dynamics.

## Acknowledgment

This work was supported in part by JSPS KAKENHI Grant Number JP24K17297, JST ASPIRE Grant Number JPMJAP2402, the Swedish Research Council Distinguished Professor Grant 2017-01078, and the Knut and Alice Wallenberg Foundation Wallenberg Scholar Grant.

## References

[AGS08] Luigi Ambrosio, Nicola Gigli, and Giuseppe Savar´e. Gradient Flows: in Metric Spaces and in the Space of Probability Measures. Springer, second edition, 2008.

[BB00] Jean-David Benamou and Yann Brenier. A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem. Numerische Mathematik, 84(3):375–393, 2000.

[BCC<sup>+</sup>15] Jean-David Benamou, Guillaume Carlier, Marco Cuturi, Luca Nenna, and Gabriel Peyr´e. Iterative Bregman projections for regularized transportation problems. SIAM Journal on Scientific Computing, 37(2):A1111–A1138, 2015.

[Bon13] Nicolas Bonnotte. Unidimensional and evolution methods for optimal transportation. PhD thesis, Universit´e Paris Sud-Paris XI; Scuola normale superiore (Pise, Italie), 2013.

[BRPP15] Nicolas Bonneel, Julien Rabin, Gabriel Peyr´e, and Hanspeter Pfister. Sliced and Radon Wasserstein barycenters of measures. Journal of Mathematical Imaging and Vision, 51(1):22–45, 2015.

[CD14] Marco Cuturi and Arnaud Doucet. Fast computation of Wasserstein barycenters. In International Conference on Machine Learning, pages 685–693. PMLR, 2014.

[CGP16a] Yongxin Chen, Tryphon T. Georgiou, and Michele Pavon. Optimal steering of a linear stochastic system to a final probability distribution, Part I. IEEE Transactions on Automatic Control, 61(5):1158–1169, 2016.

[CGP16b] Yongxin Chen, Tryphon T. Georgiou, and Michele Pavon. Optimal steering of a linear stochastic system to a final probability distribution, Part II. IEEE Transactions on Automatic Control, 61(5):1170–1180, 2016.

[CGP17] Yongxin Chen, Tryphon T. Georgiou, and Michele Pavon. Optimal transport over a linear dynamical system. IEEE Transactions on Automatic Control, 62(5):2137–2152, 2017.

[CGP21] Yongxin Chen, Tryphon T. Georgiou, and Michele Pavon. Optimal transport in systems and control. Annual Review of Control, Robotics, and Autonomous Systems, 4(1):89–113, 2021.

[CGT19] Yongxin Chen, Tryphon T. Georgiou, and Allen Tannenbaum. Optimal transport for Gaussian mixture models. IEEE Access, 7:6269–6278, 2019.

[CS25] Giacomo Cozzi and Filippo Santambrogio. Long-time asymptotics of the sliced-Wasserstein flow. SIAM Journal on Imaging Sciences, 18(1):1–19, 2025.

[Dea07] Stanley R. Deans. The Radon Transform and Some of Its Applications. Courier Corporation, 2007.

[DGT25] Anqi Dong, Tryphon T. Georgiou, and Allen Tannenbaum. Data Assimilation for Sign-indefinite Priors: A generalization of Sinkhorn’s algorithm. Automatica, 177:112283, 2025.

[DSG24] Anqi Dong, Arthur Stephanovitch, and Tryphon T. Georgiou. Monge– Kantorovich optimal transport through constrictions and flow-rate constraints. Automatica, 160:111448, 2024.

[ID26] Kaito Ito and Anqi Dong. Sliced Wasserstein steering between Gaussian measures. In 2026 European Control Conference (ECC). IEEE, 2026.

[IK23a] Kaito Ito and Kenji Kashima. Entropic model predictive optimal transport for underactuated linear systems. IEEE Control Systems Letters, 7:2761–2766, 2023.

[IK23b] Kaito Ito and Kenji Kashima. Entropic model predictive optimal transport over dynamical systems. Automatica, 152:110980, 2023.

[IK25] Kaito Ito and Kenji Kashima. Maximum entropy density control of discrete-time linear systems with quadratic cost. IEEE Transactions on Automatic Control, 70(5):3024–3039, 2025.

[Kan42] Leonid V. Kantorovich. On the translocation of masses. In Dokl. Akad. Nauk. USSR (NS), volume 37, pages 199–201, 1942.

[LMW23] Shiying Li, Caroline Moosmueller, and Yongzhe Wang. Measure transfer via stochastic slicing and matching. arXiv preprint arXiv:2307.05705, 2023.

[LSM<sup>+</sup>19] Antoine Liutkus, Umut Simsekli, Szymon Majewski, Alain Durmus, and Fabian-Robert St¨oter. Sliced-Wasserstein flows: Nonparametric generative modeling via optimal transport and difusions. In International Conference on Machine Learning, pages 4104–4113. PMLR, 2019.

[Mon81] Gaspard Monge. M´emoire sur la th´eorie des d´eblais et des remblais. Mem. Math. Phys. Acad. Royale Sci., pages 666–704, 1781.

[NH24] Khai Nguyen and Nhat Ho. Sliced Wasserstein estimation with control variates. In International Conference on Learning Representations, pages 2087– 2106, 2024.

[PBC<sup>+</sup>20] Lois Paulin, Nicolas Bonneel, David Coeurjolly, Jean-Claude Iehl, Antoine Webanck, Mathieu Desbrun, and Victor Ostromoukhov. Sliced optimal transport sampling. ACM Trans. Graph., 39(4):99, 2020.

[PC19] Gabriel Peyr´e and Marco Cuturi. Computational optimal transport: With applications to data science. Foundations and Trends® in Machine Learning, 11(5-6):355–607, 2019.

[PKD07] Fran¸cois Piti´e, Anil C. Kokaram, and Rozenn Dahyot. Automated colour grading using colour distribution transfer. Computer Vision and Image Understanding, 107(1-2):123–137, 2007.

[RR98] Svetlozar T. Rachev and Ludger R¨uschendorf. Mass Transportation Problems: Volume I: Theory. Springer, 1998.

[SDG25] Arthur Stephanovitch, Anqi Dong, and Tryphon T. Georgiou. Optimal transport through a toll station. European Journal of Applied Mathematics, 36(3):613–637, 2025.

[SDGP<sup>+</sup>15] Justin Solomon, Fernando De Goes, Gabriel Peyr´e, Marco Cuturi, Adrian Butscher, Andy Nguyen, Tao Du, and Leonidas Guibas. Convolutional Wasserstein distances: Eficient optimal transportation on geometric domains. ACM Transactions on Graphics (ToG), 34(4):1–11, 2015.

[TBN26] Gauthier Thurin, Claire Boyer, and Kimia Nadjahi. Convergence rates for distribution matching with sliced optimal transport. In Proceedings of the 39th Annual Conference on Learning Theory, pages 6156–6196. PMLR, 2026.

[Tes12] Gerald Teschl. Ordinary Diferential Equations and Dynamical Systems. American Mathematical Soc., 2012.

[Vil03] C´edric Villani. Topics in Optimal Transportation, volume 58. American Mathematical Soc., 2003.

[Vil09] C´edric Villani. Optimal Transport: Old and New, volume 338. Springer, 2009.

[VKM25] Christophe Vauthier, Anna Korba, and Quentin M´erigot. Towards understanding gradient dynamics of the sliced-Wasserstein distance via critical point analysis. In Forty-second International Conference on Machine Learning, 2025.

## A Proofs in Section 3

## A.1 Proof of Proposition 1

Fix $\bar { t } \in ( 0 , 1 )$ and set, for almost every $t \in ( 0 , \bar { t } )$

$$
a ( t ) : = \left( \int _ { \mathbb { R } ^ { n } } \| v ( t , x ) \| ^ { 2 } \rho _ { t } ( \mathrm { d } x ) \right) ^ { 1 / 2 } .
$$

By the continuity equation, the assumed integrability of $v ,$ and [AGS08, Theorem 8.3.1], the curve $t \mapsto \rho _ { t }$ is absolutely continuous in $( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { n } ) , W _ { 2 } )$ on [0, t<sup>¯</sup>], with $\begin{array} { r } { | \rho _ { t } ^ { \prime } | : = \operatorname* { l i m } _ { s  t } W _ { 2 } ( \rho _ { s } , \rho _ { t } ) / | s - } \end{array}$ $t | \leq a ( t )$ for almost every t. Hence,

$$
W _ { 2 } ( \rho _ { s } , \rho _ { t } ) \leq \int _ { s } ^ { t } a ( \tau ) \mathrm { d } \tau , \qquad 0 \leq s \leq t \leq \bar { t } .
$$

Using the triangle inequality for $S W _ { 2 }$ and $S W _ { 2 } ( \mu , \nu ) \leq n ^ { - 1 / 2 } W _ { 2 } ( \mu , \nu )$ , we obtain $| r ( t ) -$ $\begin{array} { r } { r ( s ) | \leq \frac { 1 } { \sqrt { n } } \int _ { s } ^ { t } a ( \tau ) } \end{array}$ dτ . Thus, $r ( t ) = S W _ { 2 } ( \rho _ { t } , \rho _ { 1 } )$ is absolutely continuous on $[ 0 , \bar { t } ]$

Fix $\dot { \theta } \in S ^ { n - 1 }$ and define the projected velocity by

$$
w _ { t , \theta } ( y ) : = \mathbb { E } \big [ \theta ^ { \top } v ( t , X _ { t } ) \big | \theta ^ { \top } X _ { t } = y \big ] , \qquad X _ { t } \sim \rho _ { t } .
$$

Jensen’s inequality gives

$$
\int _ { \mathbb { R } } | w _ { t , \theta } ( y ) | ^ { 2 } \rho _ { t , \theta } ( \mathrm { d } y ) \leq \int _ { \mathbb { R } ^ { n } } | \theta ^ { \top } v ( t , x ) | ^ { 2 } \rho _ { t } ( \mathrm { d } x ) .\tag{51}
$$

For $\varphi \in C _ { c } ^ { \infty } ( ( 0 , 1 ) \times \mathbb { R } )$ , substitute $\zeta ( t , x ) = \varphi ( t , \theta ^ { \top } x )$ into (5). This substitution is admissible by [AGS08, Remark 8.1.1] and shows that $( \rho _ { t , \theta } , w _ { t , \theta } )$ satisfies $\partial _ { t } \rho _ { t , \theta } + \partial _ { y } ( w _ { t , \theta } \rho _ { t , \theta } ) = 0$ weakly. It follows from (51) and [AGS08, Theorem 8.3.1] that $t \mapsto \rho _ { t , \theta }$ is locally absolutely continuous in $( \mathcal { P } _ { 2 } ( \mathbb { R } ) , W _ { 2 } )$ . Since $\rho _ { t }$ admits a density, $\rho _ { t , \theta }$ is absolutely continuous and hence atomless. The optimal transport from $\rho _ { t , \theta }$ $\rho _ { 1 , \theta }$ is therefore induced by the monotone map $\mathcal { T } _ { t } ^ { \theta }$ . By [AGS08, Theorem 8.4.7, Remark 8.4.8], for almost every $t \in ( 0 , \bar { t } )$

$$
\frac { \mathrm { d } } { \mathrm { d } t } W _ { 2 } ^ { 2 } ( \rho _ { t , \theta } , \rho _ { 1 , \theta } ) = 2 \int _ { \mathbb { R } } \left( y - { \mathcal { T } } _ { t } ^ { \theta } ( y ) \right) w _ { t , \theta } ( y ) { \rho } _ { t , \theta } ( \mathrm { d } y ) = 2 \int _ { \mathbb { R } ^ { n } } ( \theta ^ { \top } x - { \mathcal { T } } _ { t } ^ { \theta } ( \theta ^ { \top } x ) ) \theta ^ { \top } v ( t , x ) { \rho } _ { t } ( \mathrm { d } x ) .\tag{52}
$$

It remains to average this identity over the projection directions. Since $t \mapsto \rho _ { t }$ is continuous in $W _ { 2 }$ on $[ 0 , \bar { t } ]$ 2

$$
C _ { \bar { t } } : = \operatorname* { s u p } _ { 0 \leq t \leq \bar { t } } W _ { 2 } ( \rho _ { t } , \rho _ { 1 } ) < \infty .
$$

Linear projections are nonexpansive, and hence $W _ { 2 } ( \rho _ { t , \theta } , \rho _ { 1 , \theta } ) \leq C _ { \bar { t } }$ . By the Cauchy–Schwarz inequality, Eq. (52), and $\begin{array} { r } { \int _ { S ^ { n - 1 } } | \theta ^ { \top } z | ^ { 2 } \sigma ( \mathrm { d } \theta ) = \| z \| ^ { 2 } / n } \end{array}$

$$
\int _ { \mathcal { S } ^ { n - 1 } } \left| \frac { \mathrm { d } } { \mathrm { d } t } W _ { 2 } ^ { 2 } ( \rho _ { t , \theta } , \rho _ { 1 , \theta } ) \right| \sigma ( \mathrm { d } \theta ) \leq \frac { 2 C _ { \bar { t } } } { \sqrt { n } } a ( t ) .
$$

The right-hand side is integrable on $( 0 , \bar { t } )$ . Fubini’s theorem and the fundamental theorem of calculus therefore give, for almost every $t \in ( 0 , \bar { t } )$ 2

$$
\frac { \mathrm { d } } { \mathrm { d } t } S W _ { 2 } ^ { 2 } ( \rho _ { t } , \rho _ { 1 } ) = \int _ { S ^ { n - 1 } } \frac { \mathrm { d } } { \mathrm { d } t } W _ { 2 } ^ { 2 } ( \rho _ { t , \theta } , \rho _ { 1 , \theta } ) \sigma ( \mathrm { d } \theta ) .
$$

Substituting (52) and using (10) yields

$$
\frac { \mathrm { d } } { \mathrm { d } t } S W _ { 2 } ^ { 2 } ( \rho _ { t } , \rho _ { 1 } ) = 2 \int _ { \mathbb { R } ^ { n } } g _ { t } ( x ) ^ { \top } \boldsymbol { v } ( t , x ) \rho _ { t } ( \mathrm { d } x ) .
$$

Since $\bar { t } < 1$ was arbitrary, the result holds locally on $[ 0 , 1 )$

## A.2 Proof of Proposition 2

We first establish that the covariance equation is well-posed on $[ 0 , 1 )$ . Let $\mathbb { S } _ { > 0 } ^ { n }$ denote the cone of real symmetric positive-definite matrices. The vector field associated with (15) is

$$
\begin{array} { r } { F ( t , \Sigma ) : = \lambda ( t ) \big ( H ( \Sigma ) \Sigma + \Sigma H ( \Sigma ) \big ) . } \end{array}
$$

On every compact subset of $\mathbb { S } _ { > 0 } ^ { n }$ , the quantity $\theta ^ { \top } \Sigma \theta$ is bounded uniformly away from zero. It follows that $\alpha ( \Sigma , \theta )$ , and hence $H ( \Sigma )$ , is locally Lipschitz in $\Sigma .$ , uniformly in θ. Since λ is continuous, F is locally Lipschitz in Σ, locally uniformly in time. Therefore, Eq. (15) along with $\Sigma ( 0 ) = \Sigma _ { 0 }$ admits a unique maximal solution on some interval $[ 0 , t _ { \mathrm { m a x } } )$

We next show that the solution cannot leave $\mathbb { S } _ { > 0 } ^ { n }$ in finite time. By the definition of $H$

$$
H ( \Sigma ) + \frac { 1 } { n } I = \int _ { S ^ { n - 1 } } \alpha ( \Sigma , \theta ) \theta \theta ^ { \top } \sigma ( \mathrm { d } \theta ) \succeq 0 ,
$$

and hence $H ( \Sigma ) \succeq - I / n$ . Let $\ell ( t ) : = \lambda _ { \operatorname* { m i n } } ( \Sigma ( t ) )$ . The function ℓ is locally Lipschitz and therefore diferentiable almost everywhere. At every such time, choosing a unit eigenvector $q$ associated with ℓ(t) gives

$$
\dot { \ell } ( t ) = q ^ { \top } \dot { \Sigma } ( t ) q = 2 \lambda ( t ) \ell ( t ) q ^ { \top } H ( \Sigma ( t ) ) q \geq - \frac { 2 } { n } \lambda ( t ) \ell ( t ) .
$$

Gronwall’s inequality yields

$$
\lambda _ { \operatorname* { m i n } } ( \Sigma ( t ) ) \geq \lambda _ { \operatorname* { m i n } } ( \Sigma _ { 0 } ) \exp \left( - { \frac { 2 } { n } } \int _ { 0 } ^ { t } \lambda ( s ) \mathrm { d } s \right) > 0 .\tag{53}
$$

Suppose, by contradiction, that $t _ { \mathrm { m a x } } < 1$ . Since λ is continuous, Eq. (53) gives $\Sigma ( t ) \succeq a _ { * } I$ on $[ 0 , t _ { \mathrm { m a x } } )$ for some $a _ { * } > 0$ . Consequently,

$$
\alpha ( \Sigma ( t ) , \theta ) \leq \sqrt { \frac { \lambda _ { \operatorname* { m a x } } ( \Sigma _ { 1 } ) } { a _ { * } } } , \qquad t < t _ { \operatorname* { m a x } } ,
$$

and $H ( \Sigma ( t ) )$ is uniformly bounded on $[ 0 , t _ { \mathrm { m a x } } )$ . Thus, for some $c _ { * } < \infty$ and $t < t _ { \mathrm { m a x } }$ , we have

$$
\| \dot { \Sigma } ( t ) \| _ { 2 } \leq 2 c _ { * } \lambda ( t ) \| \Sigma ( t ) \| _ { 2 } .
$$

Another application of Gronwall’s inequality shows that $\| \Sigma ( t ) \| _ { 2 } \leq b _ { * } \ \mathrm { o n } \ [ 0 , t _ { \mathrm { m a x } } )$ for some $b _ { * } < \infty$

The solution therefore remains in the compact subset

$$
\left\{ \Sigma \in \mathbb { S } _ { > 0 } ^ { n } \mid a _ { * } I \preceq \Sigma \preceq b _ { * } I \right\} .
$$

This contradicts the continuation theorem for ordinary diferential equations [Tes12, Section 2.6]. Hence, Eq. (15) admits a unique solution satisfying $\Sigma ( t ) \succ 0$ throughout [0, 1).

Since Σ and λ are continuous, so are $K ( t , \Sigma ( t ) )$ and $\eta ( t )$ . The afine equation (17), with initial law $X ( 0 ) \sim \mathcal { N } ( m _ { 0 } , \Sigma _ { 0 } )$ , therefore admits a unique solution on [0, 1). The variationof-constants formula shows that $X ( t )$ remains Gaussian. Its mean satisfies

$$
\dot { m } ( t ) = - \frac { \lambda ( t ) } { n } \big ( m ( t ) - m _ { 1 } \big ) ,\tag{54}
$$

with $m ( 0 ) = m _ { 0 }$ , while its covariance satisfies (15) with initial value $\Sigma _ { 0 }$ . The solution of (54) is (16). Uniqueness of the mean and covariance equations therefore gives

$$
X ( t ) \sim { \mathcal { N } } { \big ( } m ( t ) , \Sigma ( t ) { \big ) } , \qquad t \in [ 0 , 1 ) .
$$

It remains to identify the afine drift with the averaged sliced feedback. For the Gaussian law $\rho _ { t } = \mathcal { N } ( \cdot \mid m ( t ) , \Sigma ( t ) )$ , the projected transport map is

$$
\begin{array} { r } { \mathcal T _ { t } ^ { \theta } ( \theta ^ { \top } \boldsymbol { x } ) = \theta ^ { \top } m _ { 1 } + \alpha \big ( \boldsymbol { \Sigma } ( t ) , \theta \big ) \theta ^ { \top } \big ( \boldsymbol { x } - m ( t ) \big ) . } \end{array}
$$

Substitution into Eq. (10), together with the definitions of H, K, and $\eta ,$ gives

$$
- \lambda ( t ) g _ { t } ( x ) = K ( t , \Sigma ( t ) ) x + \eta ( t ) .
$$

Thus, Eq. (17) coincides with the averaged sliced feedback.

## A.3 A Gaussian coercivity estimate

For $\Sigma \succ 0$ , define

$$
V ( \Sigma ) : = \int _ { S ^ { n - 1 } } \left( { \sqrt { \theta ^ { \top } \Sigma \theta } } - { \sqrt { \theta ^ { \top } \Sigma _ { 1 } \theta } } \right) ^ { 2 } { \sigma } ( \mathrm { d } \theta ) .\tag{55}
$$

Lemma 7 (Gaussian coercivity and directional consistency). Let $\rho _ { t } = \mathcal { N } ( \cdot \mid m ( t ) , \Sigma ( t ) )$ be $a$ Gaussian averaged sliced flow generated by a nonnegative gain λ. Set

$$
\beta : = n \Big ( \sqrt { \frac { \mathrm { t r } ( \Sigma _ { 1 } ) } { n } } + r ( 0 ) \Big ) ^ { 2 }
$$

and

$$
\mu : = \frac { \sqrt { \lambda _ { \operatorname* { m i n } } ( \Sigma _ { 1 } ) } } { \beta ^ { 3 / 2 } n ( n + 2 ) } .\tag{56}
$$

Then, for every $t \in [ 0 , 1 )$ , we have

$$
\Sigma ( t ) \preceq \beta I \ a n d \ V ( \Sigma ( t ) ) \geq \frac { \mu } { 2 } \| \Sigma ( t ) - \Sigma _ { 1 } \| _ { \mathrm { F } } ^ { 2 } .\tag{57}
$$

$H ,$ in addition, $\Sigma ( t ) \succeq \alpha I$ on [0, 1) for some $\alpha > 0$ , then $\chi _ { * } \leq \chi ( t ) \leq 1$ whenever $r ( t ) > 0$ ， where $\chi _ { * }$ is the constant in Theorem 1.

Proof. For the Gaussian law $\rho _ { t }$ , the sliced discrepancy field takes the form $\begin{array} { r } { g _ { t } ( x ) = \frac { 1 } { n } ( m ( t ) - } \end{array}$ $m _ { 1 } \big ) - H ( \Sigma ( t ) ) \big ( x - m ( t ) \big )$ . Consequently,

$$
D ( t ) = \frac { 1 } { n ^ { 2 } } \| m ( t ) - m _ { 1 } \| ^ { 2 } + \mathrm { t r } \big ( H ( \Sigma ( t ) ) \Sigma ( t ) H ( \Sigma ( t ) ) ^ { \top } \big ) ,\tag{58}
$$

$$
r ( t ) ^ { 2 } = \frac { 1 } { n } \| m ( t ) - m _ { 1 } \| ^ { 2 } + V ( \Sigma ( t ) ) .\tag{59}
$$

We first bound the covariance. By Eq. $( 1 2 ) , r ( t ) \leq r ( 0 )$ , and hence $V ( \Sigma ( t ) ) \leq r ( 0 ) ^ { 2 }$ . On the other hand, the Cauchy–Schwarz inequality gives

$$
V ( \Sigma ) \geq \Big ( \sqrt { \frac { \mathrm { t r } ( \Sigma ) } { n } } - \sqrt { \frac { \mathrm { t r } ( \Sigma _ { 1 } ) } { n } } \Big ) ^ { 2 } .
$$

It follows that $\operatorname { t r } ( \Sigma ( t ) ) \leq \beta$ , and therefore $\Sigma ( t ) \preceq \beta I$

We next establish the coercivity of V. For a symmetric matrix $\Delta$ , the first and second directional derivatives of V in the direction $\Delta$ are $D _ { \Delta } V ( \Sigma ) = - \mathrm { t r } \big ( H ( \Sigma ) \Delta \big )$ and

$$
D _ { \Delta } ^ { 2 } V ( \Sigma ) = \int _ { \mathcal { S } ^ { n - 1 } } \frac { \sqrt { \theta ^ { \top } \Sigma _ { 1 } \theta } } { 2 ( \theta ^ { \top } \Sigma \theta ) ^ { 3 / 2 } } \big ( \theta ^ { \top } \Delta \theta \big ) ^ { 2 } \sigma ( \mathrm { d } \theta ) .\tag{60}
$$

On the convex set $\{ \Sigma \succ 0 \mid \Sigma \preceq \beta I \}$ , the coeficient in (60) is bounded below by $\sqrt { \lambda _ { \operatorname* { m i n } } ( \Sigma _ { 1 } ) } / ( 2 \beta ^ { 3 / 2 } )$ Moreover,

$$
\int _ { \mathcal { S } ^ { n - 1 } } ( \theta ^ { \top } \Delta \theta ) ^ { 2 } \sigma ( \mathrm { d } \theta ) = \frac { ( \mathrm { t r } \Delta ) ^ { 2 } + 2 \| \Delta \| _ { \mathrm { F } } ^ { 2 } } { n ( n + 2 ) } \geq \frac { 2 } { n ( n + 2 ) } \| \Delta \| _ { \mathrm { F } } ^ { 2 } .
$$

Thus, V is $\mu { \mathrm { - s t r o n g l y } }$ convex on this set. Since $V ( \Sigma _ { 1 } ) = 0$ and $\nabla V ( \Sigma _ { 1 } ) = 0$ , strong convexity gives the second inequality in (57). It also implies

$$
\begin{array} { r } { \| \boldsymbol { H } ( \Sigma ) \| _ { \mathrm { F } } ^ { 2 } = \| \nabla V ( \Sigma ) \| _ { \mathrm { F } } ^ { 2 } \geq 2 \mu V ( \Sigma ) . } \end{array}\tag{61}
$$

Suppose now that $\Sigma ( t ) \succeq \alpha I$ . From Eq. (61), we have

$$
\mathrm { t r } \big ( H ( \Sigma ( t ) ) \Sigma ( t ) H ( \Sigma ( t ) ) ^ { \top } \big ) \geq 2 \alpha \mu V ( \Sigma ( t ) ) .
$$

Set $a : = \| m ( t ) - m _ { 1 } \| ^ { 2 } / n$ . Using (58) and (59), we obtain

$$
\chi ( t ) = \frac { a + n \mathrm { t r } \big ( H ( \Sigma ( t ) ) \Sigma ( t ) H ( \Sigma ( t ) ) ^ { \top } \big ) } { a + V ( \Sigma ( t ) ) } \geq \frac { a + \frac { 2 \alpha \sqrt { \lambda _ { \operatorname* { m i n } } ( \Sigma _ { 1 } ) } } { \beta ^ { 3 / 2 } ( n + 2 ) } V ( \Sigma ( t ) ) } { a + V ( \Sigma ( t ) ) } \geq \chi _ { * } .
$$

The upper bound $\chi ( t ) \leq 1$ follows from (13).

## A.4 Proof of Theorem 1

Set $f ( t ) : = r ( t ) ^ { 2 }$ . By (12) and (14), for almost every $t \in ( 0 , 1 )$ such that $f ( t ) > 0$

$$
\dot { f } ( t ) = - 2 \lambda ( t ) D ( t ) = - \frac { 2 \lambda ( t ) \chi ( t ) } { n } f ( t ) .
$$

Lemma $7$ gives $\chi ( t ) \geq \chi _ { * }$ . Hence, $\begin{array} { r } { \dot { f } ( t ) \le - \frac { 2 \chi _ { * } } { n } \lambda ( t ) f ( t ) } \end{array}$ . If f reaches zero, its monotonicity makes the desired estimate immediate. Otherwise, Gronwall’s inequality gives

$$
f ( t ) \leq f ( 0 ) \exp \left( - \frac { 2 \chi _ { * } } { n } \int _ { 0 } ^ { t } \lambda ( s ) \mathrm { d } s \right) .
$$

Taking square roots yields (19). In view of (18), we also have $r ( t ) \to 0$ as $t \nearrow 1$ . Finally, Eq. (59) gives

$$
\frac { 1 } { n } \| m ( t ) - m _ { 1 } \| ^ { 2 } \leq r ( t ) ^ { 2 } , \qquad V ( \Sigma ( t ) ) \leq r ( t ) ^ { 2 } .
$$

Therefore, $m ( t )  m _ { 1 }$ . Moreover, by (57), it holds that

$$
\frac { \mu } { 2 } \| \Sigma ( t ) - \Sigma _ { 1 } \| _ { \mathrm F } ^ { 2 } \leq V ( \Sigma ( t ) ) \leq r ( t ) ^ { 2 } ,
$$

and hence $\Sigma ( t )  \Sigma _ { 1 } \ \mathrm { a s } \ t \ Z \ 1$

## A.5 Proof of Lemma 1

By Proposition 1 and (12), r is locally absolutely continuous and $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } r ( t ) ^ { 2 } = - 2 \lambda ( t ) D ( t ) } \end{array}$ for almost every $t \in ( 0 , 1 )$ . Whenever $r ( t ) > 0$ , it follows that

$$
{ \dot { r } } ( t ) = - \lambda ( t ) { \frac { D ( t ) } { r ( t ) } } .\tag{62}
$$

On the other hand, since $\begin{array} { r } { u ( t ) = - \lambda ( t ) g _ { t } ( X ( t ) ) } \end{array}$ with $X ( t ) \sim \rho _ { t }$ , we have

$$
\mathbb { E } \big [ \| u ( t ) \| ^ { 2 } \big ] = \lambda ( t ) ^ { 2 } D ( t ) .
$$

Using $D ( t ) = \chi ( t ) r ( t ) ^ { 2 } / n$ in (62) gives

$$
\mathbb { E } \big [ \| u ( t ) \| ^ { 2 } \big ] = \frac { n \dot { r } ( t ) ^ { 2 } } { \chi ( t ) }
$$

whenever $r ( t ) > 0$

If $r ( t _ { 0 } ) = 0$ for some $t _ { 0 } < 1$ , then the nonnegativity and monotonicity of $r ^ { 2 }$ imply that $r ( t ) = 0$ for all $t \geq t _ { 0 }$ . Moreover, Eq. (13) gives $D ( t ) = 0$ there, and hence both ${ \dot { r } } ( t )$ and $\mathbb { E } [ \| u ( t ) \| ^ { 2 } ]$ vanish for almost every $t \geq t _ { 0 }$ . With the convention $\chi ( t ) = 1$ when $r ( t ) = 0$ Eq. (20) therefore holds for almost every $t \in ( 0 , 1 )$ . Integration over $( 0 , \tau )$ gives (21).

Suppose now that $r ( t ) \to 0 \mathrm { ~ a s ~ } t \nearrow 1$ . Since $0 < \chi ( t ) \leq 1$ whenever $r ( t ) > 0$ , and both sides vanish after r reaches zero, Eq. (21) yields $\begin{array} { r } { \mathcal { E } _ { u } ( 0 , \tau ) \ge n \int _ { 0 } ^ { \tau } \dot { r } ( t ) ^ { 2 } } \end{array}$ dt for every $\tau \in ( 0 , 1 )$ . By the Cauchy–Schwarz inequality,

$$
\int _ { 0 } ^ { \tau } \dot { r } ( t ) ^ { 2 } \mathrm { d } t \geq \frac { 1 } { \tau } { \left( \int _ { 0 } ^ { \tau } \dot { r } ( t ) \mathrm { d } t \right) } ^ { 2 } = \frac { ( r ( \tau ) - r ( 0 ) ) ^ { 2 } } { \tau } .
$$

Letting $\tau \nearrow 1$ gives $\mathcal { E } _ { u } ( 0 , 1 ) \ge n r ( 0 ) ^ { 2 } = n S W _ { 2 } ^ { 2 } ( \rho _ { 0 } , \rho _ { 1 } )$ , which proves (22).

## A.6 Proof of Theorem 2

Let $\tau _ { * } : = \operatorname* { i n f } \{ t \in [ 0 , 1 ) : r ( t ) = 0 \}$ and inf $\emptyset : = 1$ . Since $r ( 0 ) > 0$ , we have $r ( t ) > 0$ on $[ 0 , \tau _ { * } )$ Thus, $\lambda _ { \mathrm { S W } }$ is well defined there, and (62) gives, for almost every $t \in ( 0 , \tau _ { * } )$

$$
\dot { r } ( t ) = - \lambda _ { \mathrm { S W } } ( t ) \frac { D ( t ) } { r ( t ) } = - \frac { r ( t ) } { 1 - t } .
$$

Hence, we have ${ \frac { \mathrm { d } } { \mathrm { d } t } } \left( { \frac { r ( t ) } { 1 - t } } \right) = 0$ for almost every $t \in ( 0 , \tau _ { * } )$ , and therefore $r ( t ) = ( 1 - t ) r ( 0 )$ for $t \in [ 0 , \tau _ { * } )$ . If $\tau _ { * } < 1$ , continuity of r would give $r ( \tau _ { * } ) = 0$ , whereas the preceding identity yields $r ( \tau _ { * } ) = ( 1 - \tau _ { * } ) r ( 0 ) > 0$ . Thus, $\tau _ { * } = 1$ , and (24) holds on [0, 1). Since $\dot { r } ( t ) = - r ( 0 )$ for almost every $t \in ( 0 , 1 )$ , Lemma 1 gives for almost every $t \in ( 0 , 1 )$ ,

$$
\mathbb { E } \left[ \Vert u ( t ) \Vert ^ { 2 } \right] = \frac { n \dot { r } ( t ) ^ { 2 } } { \chi ( t ) } = \frac { n r ( 0 ) ^ { 2 } } { \chi ( t ) } .
$$

Integrating over [0, 1) and using $r ( 0 ) = S W _ { 2 } ( \rho _ { 0 } , \rho _ { 1 } )$ proves (25).

Finally, $\chi ( t ) \leq 1$ gives $\mathcal { E } _ { u } ( 0 , 1 ) \ge n S W _ { 2 } ^ { 2 } ( \rho _ { 0 } , \rho _ { 1 } )$ . If $\chi ( t ) \geq \underline { { \chi } } > 0$ almost everywhere, then $1 / \chi ( t ) \leq 1 / \chi$ , which gives the upper bound in (26).

## A.7 Proof of Corollary 1

We first show that $D ( t ) > 0$ whenever $r ( t ) > 0$ . From (58), if $D ( t ) = 0$ , then $m ( t ) = m _ { 1 }$ and tr $\left( H ( \Sigma ( t ) ) \Sigma ( t ) H ( \Sigma ( t ) ) ^ { \top } \right) = 0$ . Since $\Sigma ( t ) \succ 0$ , this implies $H ( \Sigma ( t ) ) = 0$ . As shown in the proof of Lemma 7, the function V is strictly convex and satisfies $\nabla V = - H$ . Hence, $H ( \Sigma ( t ) ) = 0$ implies $\Sigma ( t ) = \Sigma _ { 1 }$ . Equation (59) then gives $r ( t ) = 0$ . Therefore, $D ( t ) > 0$ whenever $r ( t ) > 0$

Theorem 2 now applies and yields $r ( t ) = ( 1 - t ) r ( 0 )$ . By (59), we have

$$
\| m ( t ) - m _ { 1 } \| \leq \sqrt { n } r ( t ) = \sqrt { n } ( 1 - t ) r ( 0 ) ,
$$

which proves (27). Moreover, $V ( \Sigma ( t ) ) \leq r ( t ) ^ { 2 }$ , and (57) gives

$$
\| \Sigma ( t ) - \Sigma _ { 1 } \| _ { \mathrm { F } } \leq \sqrt { \frac { 2 } { \mu } } r ( t ) = \sqrt { \frac { 2 } { \mu } } ( 1 - t ) r ( 0 ) .
$$

This proves (28) and, in particular, $\Sigma ( t )  \Sigma _ { 1 } \ \mathrm { a s } \ t \ Z \ 1$

Define $\overline { { \Sigma } } ( t ) = \Sigma ( t )$ for $t < 1$ and $\overline { { \Sigma } } ( 1 ) = \Sigma _ { 1 }$ . The map $\overline { { \Sigma } } : [ 0 , 1 ] \to \mathbb { S } _ { > 0 } ^ { n }$ is continuous. Therefore,

$$
\alpha : = \operatorname* { m i n } _ { t \in [ 0 , 1 ] } \lambda _ { \operatorname* { m i n } } \big ( \overline { \Sigma } ( t ) \big ) > 0 ,
$$

and $\Sigma ( t ) \succeq \alpha I$ on $[ 0 , 1 )$ . Lemma 7 then gives $\chi ( t ) \geq \chi .$ whenever $r ( t ) > 0$ . The energy bounds follow from Theorem 2.

## B Proofs in Section 4

## B.1 Proof of Lemma 2

By the definition of $g _ { \theta } [ \rho ]$ and pushforward identity $\rho _ { \theta } = ( P _ { \theta } ) _ { \# } \rho _ { \mathrm { \Omega } }$

$$
\| g _ { \theta } [ \rho ] \| _ { L _ { \rho } ^ { 2 } } ^ { 2 } = \int _ { \mathbb { R } ^ { n } } \left| \theta ^ { \top } x - \mathcal { T } _ { \rho } ^ { \theta } ( \theta ^ { \top } x ) \right| ^ { 2 } \mathrm { d } \rho ( x ) = \int _ { \mathbb { R } } \left| s - \mathcal { T } _ { \rho } ^ { \theta } ( s ) \right| ^ { 2 } \mathrm { d } \rho _ { \theta } ( s ) = W _ { 2 } ^ { 2 } ( \rho _ { \theta } , \rho _ { 1 , \theta } ) ,
$$

where the last equality follows from the optimality of $\mathcal { T } _ { \rho } ^ { \theta }$

For the second claim, the triangle inequality gives

$$
W _ { 2 } ( \rho _ { \theta } , \rho _ { 1 , \theta } ) \leq W _ { 2 } ( \rho _ { \theta } , \delta _ { 0 } ) + W _ { 2 } ( \delta _ { 0 } , \rho _ { 1 , \theta } ) = m _ { 2 } ( \rho _ { \theta } ) + m _ { 2 } ( \rho _ { 1 , \theta } ) \leq m _ { 2 } ( \rho ) + m _ { 2 } ( \rho _ { 1 } ) ,
$$

where $\delta _ { 0 }$ denotes the Dirac measure at 0 and the last inequality follows from $\lvert \theta ^ { \top } x \rvert \leq \lVert x \rVert$

## B.2 Proof of Lemma 3

Fix $\theta \in S ^ { n - 1 }$ and write $x = s \theta + z ,$ , with $s \in \mathbb { R }$ and $z \in \theta ^ { \perp } : = \{ z \in \mathbb { R } ^ { n } \mid \theta ^ { \top } z = 0 \}$ . In these coordinates, the map in (38) takes the form

$$
T _ { \rho , \theta , \gamma } ( s , z ) = ( \phi ( s ) , z ) , \quad \phi ( s ) : = ( 1 - \gamma ) s + \gamma T _ { \rho } ^ { \theta } ( s ) .
$$

Since $\mathcal { T } _ { \rho } ^ { \theta }$ is nondecreasing, we have

$$
\phi ( s _ { 2 } ) - \phi ( s _ { 1 } ) \geq ( 1 - \gamma ) ( s _ { 2 } - s _ { 1 } )
$$

whenever $s _ { 2 } > s _ { 1 }$ . Thus, ϕ is strictly increasing, and its inverse on $\phi ( \mathbb { R } )$ is $( 1 - \gamma ) ^ { - 1 }$ -Lipschitz. In particular,

$$
\mu _ { L } ^ { 1 } \big ( \phi ^ { - 1 } ( N ) \big ) = 0\tag{63}
$$

for every Lebesgue-null set $N \subset \mathbb { R }$

Let $N \subset \mathbb { R } ^ { n }$ satisfy $\mu _ { L } ^ { n } ( N ) = 0$ , and denote its section at $z \in \theta ^ { \perp }$ by $N _ { z } : = \{ s \in \mathbb { R }$ $( s , z ) \in N \}$ . By Fubini’s theorem, $\mu _ { L } ^ { 1 } ( N _ { z } ) = 0$ for almost every $z \in \theta ^ { \perp }$ . For each such $z ,$ the corresponding section of the inverse image is $\left( T _ { \rho , \theta , \gamma } ^ { - 1 } ( N ) \right) _ { z } = \phi ^ { - 1 } ( N _ { z } )$ , which is null by (63). Another application of Fubini’s theorem gives $\mu _ { L } ^ { n } \big ( T _ { \rho , \theta , \gamma } ^ { - 1 } ( N ) \big ) = 0$ . Since $\rho \ll \mu _ { L } ^ { n }$ $( T _ { \rho , \theta , \gamma } ) _ { \# } \rho ( N ) = \rho ( T _ { \rho , \theta , \gamma } ^ { - 1 } ( N ) ) = 0$ , and hence $( T _ { \rho , \theta , \gamma } ) _ { \# } \rho \ll \mu _ { L } ^ { n }$

It remains to verify the second-moment condition. By Minkowski’s inequality and Lemma $2 ,$

$$
\begin{array} { r } { m _ { 2 } \big ( ( T _ { \rho , \theta , \gamma } ) _ { \# } \rho \big ) = \| T _ { \rho , \theta , \gamma } \| _ { L _ { \rho } ^ { 2 } } \leq m _ { 2 } ( \rho ) + \gamma \| g _ { \theta } [ \rho ] \| _ { L _ { \rho } ^ { 2 } } \leq m _ { 2 } ( \rho ) + \gamma \big ( m _ { 2 } ( \rho ) + m _ { 2 } ( \rho _ { 1 } ) \big ) < \infty . } \end{array}
$$

Therefore, $( T _ { \rho , \theta , \gamma } ) _ { \# } \rho \in { \mathcal { P } } _ { 2 , \operatorname { a c } } ( \mathbb { R } ^ { n } )$

Finally, the update (30) has this form with $\gamma = h \lambda ( t _ { k } )$ . Since $h \lambda ( t _ { k } ) \le h _ { \bar { t } } \Lambda _ { \bar { t } } < 1$ , the conclusion follows recursively from $\rho _ { 0 } \in \mathcal { P } _ { 2 , \mathrm { a c } } ( \mathbb { R } ^ { n } )$

## B.3 Proof of Lemma 4

Set $\gamma _ { k } : = h \lambda ( t _ { k } )$ . Conditionally on the direction history, Minkowski’s inequality and Lemma 2 give

$$
m _ { 2 } ( \rho _ { k + 1 } ^ { h } ) \leq m _ { 2 } ( \rho _ { k } ^ { h } ) + \gamma _ { k } \| g _ { \theta _ { k } } [ \rho _ { k } ^ { h } ] \| _ { L _ { \rho _ { k } ^ { h } } ^ { 2 } } \leq ( 1 + \gamma _ { k } ) m _ { 2 } ( \rho _ { k } ^ { h } ) + \gamma _ { k } m _ { 2 } ( \rho _ { 1 } ) .
$$

Writing $a _ { k } : = m _ { 2 } ( \rho _ { k } ^ { h } )$ and $c : = m _ { 2 } ( \rho _ { 1 } )$ , we obtain

$$
a _ { k + 1 } + c \leq ( 1 + \gamma _ { k } ) ( a _ { k } + c ) .
$$

Iteration yields

$$
a _ { k } + c \leq ( a _ { 0 } + c ) \prod _ { i = 0 } ^ { k - 1 } ( 1 + \gamma _ { i } ) \leq ( a _ { 0 } + c ) \exp \left( h \sum _ { i = 0 } ^ { k - 1 } \lambda ( t _ { i } ) \right) \leq \bigl ( m _ { 2 } ( \rho _ { 0 } ) + m _ { 2 } ( \rho _ { 1 } ) \bigr ) \mathrm { e } ^ { \Lambda _ { i } \bar { t } } .
$$

Dropping c from the left-hand side proves (39).

For the averaged flow, Minkowski’s inequality and Lemma 2 give

$$
\| \bar { g } [ \rho _ { t } ] \| _ { L _ { \rho _ { t } } ^ { 2 } } \leq \int _ { S ^ { n - 1 } } \| g _ { \theta } [ \rho _ { t } ] \| _ { L _ { \rho _ { t } } ^ { 2 } } \sigma ( \mathrm { d } \theta ) \leq m _ { 2 } ( \rho _ { t } ) + m _ { 2 } ( \rho _ { 1 } ) .
$$

Using the integral form of (31), we therefore have

$$
m _ { 2 } ( \rho _ { t } ) \leq m _ { 2 } ( \rho _ { 0 } ) + \int _ { 0 } ^ { t } \lambda ( s ) \big ( m _ { 2 } ( \rho _ { s } ) + m _ { 2 } ( \rho _ { 1 } ) \big ) \mathrm { d } s .
$$

Gronwall’s inequality gives

$$
m _ { 2 } ( \rho _ { t } ) \leq \left( m _ { 2 } ( \rho _ { 0 } ) + m _ { 2 } ( \rho _ { 1 } ) \right) \exp \Biggl ( \int _ { 0 } ^ { t } \lambda ( s ) \mathrm { d } s \Biggr ) \leq \left( m _ { 2 } ( \rho _ { 0 } ) + m _ { 2 } ( \rho _ { 1 } ) \right) \mathrm { e } ^ { \Lambda _ { t } \bar { t } } .
$$

Thus, it follows from Lemma 3 that both iterative and averaged laws remain in ${ \ K } _ { \bar { t } }$ on $[ 0 , \bar { t } ]$

## B.4 Proof of Lemma 5

By Lemma 2 and the definition of $\mathcal { K } _ { \bar { t } } , \| g _ { \theta } [ \rho ] \| _ { L _ { \rho } ^ { 2 } } \leq m _ { 2 } ( \rho ) + m _ { 2 } ( \rho _ { 1 } ) \leq M _ { \bar { t } }$ . Since σ is a probability measure,

$$
\int _ { S ^ { n - 1 } } \| g _ { \boldsymbol { \theta } } [ \rho ] \| _ { L _ { \rho } ^ { 2 } } ^ { 2 } \sigma ( \mathrm { d } \boldsymbol { \theta } ) \leq M _ { \bar { t } } ^ { 2 } .
$$

Moreover, Jensen’s inequality and Fubini’s theorem give

$$
\| \bar { g } [ \rho ] \| _ { L _ { \rho } ^ { 2 } } ^ { 2 } = \int _ { \mathbb { R } ^ { n } } \left\| \int _ { S ^ { n - 1 } } g _ { \theta } [ \rho ] ( x ) \sigma ( \mathrm { d } \theta ) \right\| ^ { 2 } \mathrm { d } \rho ( x ) \leq \int _ { S ^ { n - 1 } } \| g _ { \theta } [ \rho ] \| _ { L _ { \rho } ^ { 2 } } ^ { 2 } \sigma ( \mathrm { d } \theta ) .
$$

Combining the two estimates proves (40).

## B.5 Proof of Lemma 6

By Assumption 2, we have

$$
\begin{array} { r l } & { \| b ( t _ { k } , x _ { k } ^ { h } ) - b ( t _ { k } , y ( t _ { k } ) ) \| _ { L ^ { 2 } ( \Omega _ { 0 } ) } = \lambda ( t _ { k } ) \left\| \bar { g } [ \rho _ { k } ^ { h } ] ( x _ { k } ^ { h } ) - \bar { g } [ \rho _ { t _ { k } } ] ( y ( t _ { k } ) ) \right\| _ { L ^ { 2 } ( \Omega _ { 0 } ) } } \\ & { \qquad \leq \Lambda _ { \bar { t } } L _ { \bar { t } } \| x _ { k } ^ { h } - y ( t _ { k } ) \| _ { L ^ { 2 } ( \Omega _ { 0 } ) } . } \end{array}
$$

This proves (42). We next estimate the local truncation error. By Lemmas 4 and 5, it holds that $\| b ( t , y ( t ) ) \| _ { L ^ { 2 } ( \Omega _ { 0 } ) } \leq \Lambda _ { \bar { t } } M _ { \bar { t } }$ for $t \in [ 0 , \bar { t } ]$ . The integral form of (31) therefore gives

$$
\| y ( s ) - y ( t ) \| _ { L ^ { 2 } ( \Omega _ { 0 } ) } \leq \Lambda _ { \bar { t } } M _ { \bar { t } } | s - t |\tag{64}
$$

for $s , t \in [ 0 , \bar { t } ]$

The definition of b, Assumption 2, and (64) yield

$$
\begin{array} { r l } & { \| b ( s , y ( s ) ) - b ( t , y ( t ) ) \| _ { L ^ { 2 } ( \Omega _ { 0 } ) } } \\ & { \le | \lambda ( s ) - \lambda ( t ) | \| \bar { g } [ \rho _ { s } ] ( y ( s ) ) \| _ { L ^ { 2 } ( \Omega _ { 0 } ) } + \lambda ( t ) \| \bar { g } [ \rho _ { s } ] ( y ( s ) ) - \bar { g } [ \rho _ { t } ] ( y ( t ) ) \| _ { L ^ { 2 } ( \Omega _ { 0 } ) } } \\ & { \le M _ { \bar { t } } \bigl ( \Lambda _ { \bar { t } } ^ { \prime } + \Lambda _ { \bar { t } } ^ { 2 } L _ { \bar { t } } \bigr ) | s - t | . } \end{array}
$$

Finally, define $\begin{array} { r } { r _ { t , h } : = \int _ { t } ^ { t + h } \bigl ( b ( s , y ( s ) ) - b ( t , y ( t ) ) \bigr ) } \end{array}$ ds. The integral form of the averaged dynamics gives (43), while

$$
\| r _ { t , h } \| _ { L ^ { 2 } ( \Omega _ { 0 } ) } \leq M _ { \bar { t } } \bigl ( \Lambda _ { \bar { t } } ^ { \prime } + \Lambda _ { \bar { t } } ^ { 2 } L _ { \bar { t } } \bigr ) \int _ { t } ^ { t + h } | s - t | \mathrm { d } s = B _ { \bar { t } } h ^ { 2 } ,
$$

where $\begin{array} { r } { B _ { \bar { t } } : = \frac { M _ { \bar { t } } } { 2 } \big ( \Lambda _ { \bar { t } } ^ { \prime } + \Lambda _ { \bar { t } } ^ { 2 } L _ { \bar { t } } \big ) } \end{array}$