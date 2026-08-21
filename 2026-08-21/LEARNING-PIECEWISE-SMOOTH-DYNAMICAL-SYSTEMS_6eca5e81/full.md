# LEARNING PIECEWISE-SMOOTH DYNAMICAL SYSTEMS

Davide Murari Department of Applied Mathematics and Theoretical Physics University of Cambridge dm2011@cam.ac.uk

Chris Budd OBE Department of Mathematical Sciences University of Bath mascjb@bath.ac.uk

Erik Jansson Department of Applied Mathematics and Theoretical Physics University of Cambridge eoj23@cam.ac.uk

Carola-Bibiane Schönlieb Department of Applied Mathematics and Theoretical Physics University of Cambridge cbs31@cam.ac.uk

## ABSTRACT

Discovering dynamical systems from trajectory data is a central problem in applied mathematics and engineering. Whilst recent advances in machine learning have led to strong progress in data-driven system identification, much less attention has been given to systems with discontinuous dynamics. These systems are nevertheless highly relevant in applications, including climate dynamics and mechanical systems with friction. In this work, we consider the problem of identifying piecewisesmooth dynamical systems directly from trajectory data. Compared with the smooth setting, this requires recovering the governing equations and detecting the switching hyperplanes that separate different dynamical regimes and characterising their behaviour, such as sliding motion. We present a modular framework for discovering such systems by first estimating switching hyperplanes from data and then learning smooth dynamics within each region using geometry-constrained neural networks. The geometry-learning phase is studied from a statistical perspective, analysing the identifiability of the discontinuities and the robustness of the procedure. We also introduce a novel neural network architecture with a prescribed discontinuity set, and provide a theoretical analysis of its approximation properties. The approach is tested on low-dimensional benchmark problems, including dry-friction oscillators and the PP04 climate model for the ice ages.

Keywords Data-driven modelling  Filippov systems  Discontinuous Neural Networks  Scientific machine learning

## 1 Introduction

Push a heavy box across a floor, and it does not move at once, until suddenly, it does. Open a stiff door slowly and the hinge does not turn smoothly but in tiny catches and releases. In both cases, the dynamics do not vary smoothly but jump between distinct regimes, sticking and slipping, as the system passes from one regime to another.

Systems like these are not unusual. We consider piecewise-smooth systems, whose vector fields are smooth within separate regions $\mathcal { P } _ { k }$ of state space but may jump across the switching hyperplanes $\Sigma _ { k , \ell }$ dividing them. Throughout this work, such systems are interpreted in the sense of Filippov, which replaces the discontinuous vector field on a switching hyperplane by a set-valued convexification of its nearby limiting values. We refer to systems equipped with this solution concept as Filippov systems. At a regular switching hyperplane, Filippov solutions may cross the hyperplane or slide along an attracting part of it [2, 14, 16, 20]. Discontinuous differential equations arise in mechanical systems with friction [22], sliding-mode control [15], robotics [1], and the glacial cycles of the Earth’s climate [4, 27, 28].

Often the governing dynamics are unknown and must be recovered from observed trajectories. For piecewise-smooth systems, this requires learning the vector field in each regime, locating the switching hyperplanes, and characterising the behaviour at those hyperplanes. A misplaced switching hyperplane misplaces the discontinuities, regardless of how accurately the local dynamics are learned. Figure 1 illustrates this setting for the PP04 climate model: the discontinuity is confined to an affine plane, and the learned model reproduces reference trajectories that repeatedly cross this plane.

![](images/4ec779a9057785fcfcbb09199663e56c8d2c04e9ddee5440386a31e76ee2dcaf.jpg)  
Figure 1: The PP04 system [4,27] is a three-dimensional climate model of the ice ages. The vector field is discontinuous across the affine plane Σ, shown in yellow, and the cyan markers indicate trajectory crossings of this plane. The left panel shows trajectories of the reference system, while the right panel shows rollouts of the learned discontinuous model from the same initial conditions. The learned model is supplied with the true switching plane. The close agreement between the two trajectory families demonstrates its ability to reproduce the discontinuous dynamics. Further details are provided in Section 5.5.2.

Data-driven modelling of dynamical systems is commonly pursued in two complementary ways. One seeks to recover explicit governing equations from observations, deriving a symbolic approximation of them [3, 10, 12]. A related difficulty is that simple governing equations may only exist in appropriate coordinates. This has motivated autoencoderbased approaches that simultaneously learn latent coordinates and parsimonious dynamics [7, 26]. Another way to model dynamical systems instead learns the time evolution directly from trajectory data using neural or latent dynamical models. This includes recurrent architectures for time-series forecasting [19, 34], neural ordinary differential equations [9], and latent or controlled variants designed for partially observed or irregularly sampled trajectories [23,31]. Universal differential equations further combine mechanistic model components with neural parameterisations of unknown terms [30].

Within scientific machine learning, an important refinement is to constrain the architecture so that the learned model respects the known structure of the dynamics. For Hamiltonian and Lagrangian systems, this has led to Hamiltonian neural networks, Lagrangian neural networks, symplectic networks, and symplectic neural flows, where conservation laws, constraints, or symplecticity are built into the parametrisation [5, 6, 11, 18, 21]. The present work follows the same structure-preserving philosophy but applies it to non-smooth systems. The relevant structure in our setting is the geometry of the switching set and the requirement that learned discontinuities remain confined to it. Existing work has considered learning local dynamics when the switching geometry is known, including friction problems [25]. Here, we additionally seek to recover that geometry from trajectory data.

Main contributions We introduce a geometry-first, two-stage framework for discovering piecewise-smooth dynamical systems from trajectory data. Our main contributions are three.

First, we develop a method for recovering arrangements of affine switching hyperplanes before fitting the vector field. Their sign patterns define interpretable polyhedral regions, see Figure 2, so the inferred switching geometry can be inspected and validated independently of the learned dynamics. We complement this method with a detectability analysis quantifying how jump size, observation noise, and derivative-estimation error affect geometry recovery.

Second, we introduce a geometry-preserving neural architecture whose discontinuities are confined, by construction, to a prescribed hyperplane arrangement. It can therefore learn distinct dynamics in adjacent regions without introducing discontinuities elsewhere. The regression experiments in Section 5.3 demonstrate this distinction directly.

Third, we prove that this architecture can approximate piecewise-smooth vector fields to arbitrary accuracy on compact sets away from the switching interfaces, while preserving the prescribed discontinuity set. Experiments on dry friction oscillators and the PP04 climate model validate both stages of the framework. In the reported baseline comparisons, the proposed models achieve lower rollout and vector-field errors than smooth Neural ODEs while using an order of magnitude fewer trainable parameters.

![](images/57e9720008a18dbb0ec2136e6a1884cb1e72e94b9bb1a854d98380709d3a9ed8.jpg)  
Figure 2: Schematic illustration of the geometry-learning stage. The true switching geometry is an arrangement of affine lines, which partitions the state space into polyhedral regions $\mathcal { P } _ { i }$ . From trajectory data, the algorithm estimates the switching geometry and obtains an approximate arrangement, shown on the right, with corresponding regions $\widetilde { \mathcal { P } } _ { i }$ More details on this experiment are in Appendix E.

Modularity and interpretability of the methodology The two stages of the framework are modular. The geometryestimation stage is driven by derivative approximations from trajectory data, and the derivative-estimation procedure can be adapted to the noise level, sampling rate, and dimension of the problem. In this paper, this stage combines derivative-jump candidate extraction with random sample consensus (RANSAC) hyperplane fitting, followed by total least squares refinement. This stage can be skipped when the switching geometry is known in advance. The dynamicslearning stage can likewise incorporate prior knowledge about the local dynamics. For example, if each smooth piece is known to be affine, one can use a piecewise-affine model. If the local dynamics are Hamiltonian, one can use a Hamiltonian neural network [18] in each region. In our numerical experiments, we do not assume such detailed prior knowledge. Instead, we use a general piecewise-smooth neural vector field.

Outline of the paper This paper is structured as follows: The necessary background on piecewise-smooth dynamical systems is presented in Section 2. Section 3 describes the dataset and the joint learning of geometry and dynamics. Some theoretical aspects of the proposed methodology are considered in Section 4, including an analysis of the hyperplane detectability from noisy data and the approximation properties of the considered network. We include a selection of numerical experiments in Section 5 looking at friction and climate models. These experiments illustrate both the methodology and the theoretical results. Finally, we draw some conclusions from this work.

## 2 Piecewise-smooth dynamical systems

In this work, we use the term piecewise-smooth system to refer to a dynamical system whose vector field may be discontinuous across affine switching hyperplanes $\Sigma _ { k , \ell } \left[ 1 6 \right]$ and is continuously differentiable away from them. Its solutions are understood in the Filippov sense [14, 16]. We now make this setting precise.

We focus on systems with smoothness regions given by polyhedral partitions induced by finite arrangements of affine hyperplanes. More precisely, let $\{ \mathsf { \bar { H } } _ { 1 } , \dots , H _ { R } \}$ be a finite affine hyperplane arrangement of $\mathbf { \mathbb { R } ^ { d } }$ , and let $P = \{ \mathcal { P } _ { 1 } , . . . , \mathcal { P } _ { K } \}$ be the collection of connected components of

$$
\mathbb { R } ^ { d } \backslash \bigcup _ { r = 1 } ^ { R } H _ { r } .
$$

Then each $\mathcal { P } _ { k } , k = 1 , \ldots , K$ , is an open convex cell of the form

$$
\mathcal { P } _ { k } = \{ x \in \mathbb { R } ^ { d } \mid A _ { k } x < b _ { k } \} , \qquad A _ { k } \in \mathbb { R } ^ { m _ { k } \times d } , b _ { k } \in \mathbb { R } ^ { m _ { k } } ,\tag{1}
$$

where the inequality is understood componentwise, and such that $\textstyle \bigcup _ { k = 1 } ^ { K } { \overline { { \mathcal { P } _ { k } } } } = \mathbb { R } ^ { d }$ , where $\overline { { \mathcal { P } _ { k } } }$ denotes the closure of $\mathcal { P } _ { k }$ For every pair of regions sharing a codimension-one boundary, we denote by $\Sigma _ { k , \ell } : = \mathrm { a f f } ( \overline { { \mathcal { P } _ { k } } } \cap \overline { { \mathcal { P } _ { \ell } } } )$ the affine hyperplane containing the facet shared by $\bar { \mathcal { P } } _ { k }$ and $\mathcal { P } _ { \ell }$ . We refer to $\Sigma _ { k , \ell }$ as a switching hyperplane. We consider dynamical systems governed by piecewise ${ } _ { - \mathcal { C } ^ { 1 } }$ vector fields $f : \mathbb { R } ^ { d } \times \mathbb { R } \to \mathbb { R } ^ { d }$ , namely systems of the form

$$
\begin{array} { r } { \dot { x } = f ( x , t ) : = f _ { k } ( x , t ) , \quad x \in \mathcal { P } _ { k } , \quad k = 1 , \dots , K , } \end{array}\tag{2}
$$

where each $f _ { k } : \mathcal P _ { k } \times \mathbb R \to \mathbb R ^ { d }$ is continuously differentiable. Thus, all discontinuities of $f$ are confined to the shared facets, each of which lies in its corresponding switching hyperplane $\Sigma _ { k , \ell }$ . We denote the union of these facets by $\Sigma ,$ that is,

$$
\Sigma : = \mathbb { R } ^ { d } \setminus \bigcup _ { k = 1 } ^ { K } \mathcal { P } _ { k } = \bigcup _ { r = 1 } ^ { R } H _ { r } .
$$

Throughout this work, we assume that each $f _ { k }$ admits a continuously differentiable extension $f _ { k } : \mathbb { R } ^ { d } \times \mathbb { R } \to \mathbb { R } ^ { d }$ , and we use the same notation for the vector field on $\mathcal { P } _ { k }$ and its extension.

The values of $f$ on Σ may be assigned arbitrarily without changing its Filippov regularisation: because the facets have Lebesgue measure zero, the regularisation depends on the neighbouring limiting vector fields rather than on pointwise values assigned on the facets. At regular points of codimension-one facets, this regularisation simplifies, as discussed in more detail below. At intersections of several facets, the regularisation accounts for the limiting vector fields from all adjacent regions. A Filippov solution is an absolutely continuous curve whose derivative belongs to the Filippov regularisation of f almost everywhere [14, 16].

A typical example is the case of two half-spaces separated by a hyperplane

$$
\Sigma : = \left\{ x \in \mathbb { R } ^ { d } : \ s ( x ) = 0 \right\} , \qquad s ( x ) : = a ^ { \top } x + b ,
$$

where $a \in \mathbb { R } ^ { d }$ satisfies $\| a \| _ { 2 } = 1$ and $b \in \mathbb { R }$ , so that

$$
\dot { x } = f ( x , t ) : = \left\{ \begin{array} { l l } { f ^ { + } ( x , t ) , \quad s ( x ) > 0 , } \\ { f ^ { - } ( x , t ) , \quad s ( x ) < 0 . } \end{array} \right.\tag{3}
$$

This single-hyperplane setting describes the local behaviour near a regular codimension-one facet separating two regions of a more general partition. For $x \in \Sigma$ , the normal components of the two limiting vector fields are

$$
\nu ^ { + } ( x , t ) : = a ^ { \top } f ^ { + } ( x , t ) , \qquad \nu ^ { - } ( x , t ) : = a ^ { \top } f ^ { - } ( x , t ) .
$$

If $\nu ^ { + }$ and $\nu ^ { - }$ have the same sign, trajectories locally cross the switching hyperplane. If

$$
\nu ^ { + } ( x , t ) < 0 < \nu ^ { - } ( x , t ) ,
$$

then both limiting vector fields point towards $\Sigma ,$ , and the hyperplane is locally attracting at $( x , t )$ . In the Filippov convention [16], sliding motion on the locally attracting part of the hyperplane Σ is then described by the convex combination

$$
f _ { \Sigma } ( x , t ) = \alpha ( x , t ) f ^ { + } ( x , t ) + ( 1 - \alpha ( x , t ) ) f ^ { - } ( x , t ) ,
$$

where $\alpha \in [ 0 , 1 ]$ is chosen so that the trajectory remains tangent to the hyperplane, i.e.,

$$
\boldsymbol { a } ^ { \top } f _ { \Sigma } ( \boldsymbol { x } , t ) = 0 .
$$

Equivalently,

$$
\alpha ( x , t ) = \frac { \nu ^ { - } ( x , t ) } { \nu ^ { - } ( x , t ) - \nu ^ { + } ( x , t ) } .
$$

This gives the sliding vector field along Σ. The opposite sign configuration,

$$
\nu ^ { - } ( x , t ) < 0 < \nu ^ { + } ( x , t ) ,
$$

corresponds to a locally repelling or escaping part of the hyperplane, where forward solutions starting on Σ may fail to be unique. Figure 3 illustrates the two behaviours that are most important for the learning problem: transversal crossings, which generate localised derivative jumps, and attracting sliding, where trajectories remain on the switching hyperplane and must be propagated with the Filippov sliding vector field $f _ { \Sigma }$

For further background, see [14, 16] and the references therein.

![](images/b2db7bd7857ca89e2928ce608ce4f9c0b70e32ed783a1813e7e9be38f7a873a4.jpg)  
Figure 3: Basic switching behaviours in the toy Filippov system ${ \dot { q } } = p , { \dot { p } } = - q + \mathrm { s i g n } ( 1 - p )$ . The switching hyperplane is $\Sigma = \{ p = 1 \}$ , shown by the horizontal black line. The green segment marks the attracting sliding region $- 1 < q < 1$ . On this segment, the one-sided normal velocities satisfy $- q - 1 < 0 < - q + 1$ , so the Filippov vector field is tangent to Σ and gives $\dot { q } = 1 , \dot { p } = 0$ . The black curves show representative trajectories: one crosses Σ transversally outside the sliding interval, while the other reaches the attracting part of Σ and then slides along it.

## 3 Learning piecewise-smooth dynamical systems

Our approach to learning piecewise-smooth dynamical systems is split into two sequential tasks.

First, we estimate the switching geometry by recovering an arrangement of affine switching hyperplanes. The sign patterns of this arrangement define the approximate polyhedral regions

$$
\widetilde { P } = \{ \widetilde { \mathcal { P } } _ { 1 } , \dots , \widetilde { \mathcal { P } } _ { \widetilde { K } } \} ,
$$

which are used during dynamics learning.

Second, we learn a piecewise-smooth vector field adapted to this recovered partition.

This procedure separates “geometry learning” from “dynamics learning”, and allows inductive biases specific to each task to be introduced independently.

We now describe the core building blocks of our methodology, starting with the dataset, then moving to the geometry and dynamics learning phases.

## 3.1 Dataset

We consider N reference trajectories and their noisy observations, indexed by $n = 1 , \ldots , N$ . The nth trajectory starts from the initial condition $x _ { 0 } ^ { n }$ at time $t _ { 0 } ^ { n }$ , and is sampled at the regularly spaced times

$$
t _ { j } ^ { n } : = t _ { 0 } ^ { n } + j \ h , \qquad j = 0 , \ldots , J ,
$$

with final time $T _ { n } = t _ { , I } ^ { n } = t _ { 0 } ^ { n } + J h$ . For simplicity, all trajectory segments are taken to have the same number of time steps. The extension to segments of non-uniform length is immediate. Let $x ( t ; t _ { 0 } , x _ { 0 } )$ denote the solution satisfying

$$
\dot { x } ( t ; t _ { 0 } , x _ { 0 } ) = f ( x ( t ; t _ { 0 } , x _ { 0 } ) , t ) , \qquad x ( t _ { 0 } ; t _ { 0 } , x _ { 0 } ) = x _ { 0 } .
$$

For the nth trajectory, we write

$$
\begin{array} { r } { x _ { * } ^ { n } ( t ) = x ( t ; t _ { 0 } ^ { n } , x _ { 0 } ^ { n } ) , \qquad t \in [ t _ { 0 } ^ { n } , T _ { n } ] . } \end{array}
$$

We assume that the sampling times are known exactly and are not subject to timing errors. In the idealised data model, the noise-free states $x _ { * } ^ { n } ( t _ { j } ^ { n } )$ are additively perturbed by independent Gaussian observation noise. The observations are thus

$$
y _ { j } ^ { n } : = x _ { * } ^ { n } ( t _ { j } ^ { n } ) + \eta _ { j } ^ { n } , \qquad j = 0 , \ldots , J ,\tag{4}
$$

where $\eta _ { i } ^ { n } \sim \mathsf { N } ( 0 , \sigma ^ { 2 } I _ { d } )$ are independent across both n and $j , I _ { d } \in \mathbb { R } ^ { d \times d }$ denotes the identity matrix, and $\sigma \geq 0$ is the noise level. For numerically generated data, the discrepancy from the exact solution may also contain time-discretisation errors. The analysis in Section 4.1 isolates the effect of Gaussian observation noise. In the numerical experiments, see Section 5, the reference trajectories are generated synthetically from the benchmark systems. When an exact solution is available between switching events, we use it together with event localisation at switching times; otherwise, we use an event-aware numerical solver. The system-specific sampling boxes, time grids, and solver details are given in Section 5.2.

## 3.2 Learning the geometry

There are two natural signals for recovering Σ: (i) trajectory points lying on sliding segments, and (ii) jumps in estimated velocities across consecutive samples. The first signal is present only when sliding occurs and is highly sensitive to noise, which can break collinearity. We therefore base the geometry-learning stage on the second signal: large changes in estimated velocity are treated as candidate discontinuity points.

Implementing velocity-jump-based geometry learning requires choosing a velocity estimator. In principle, our method is independent of this choice, but its performance depends on the properties of the adopted approach. For instance, in the regime of negligible observational noise, we can compute velocity estimates along trajectory n at the points $y _ { j } ^ { n }$ using central finite differences, i.e.,

$$
v _ { h , j } ^ { n } = \frac { y _ { j + 1 } ^ { n } - y _ { j - 1 } ^ { n } } { 2 h } , \qquad j = 1 , \ldots , J - 1 ,\tag{5}
$$

meaning that each trajectory provides $J - 1$ finite-difference estimates.

If noise is present, we can instead employ a noise-mitigating Savitzky–Golay velocity estimator [32, 33], so that

$$
v _ { h , j } ^ { n } = \sum _ { k = - r } ^ { r } s _ { k } y _ { j + k } ^ { n } , \qquad j = r , \ldots , J - r ,\tag{6}
$$

where the coefficients are obtained by fitting a polynomial of degree $p$ to uniformly spaced coordinate indices by linear least squares and therefore do not depend on the observed data. Nonlinear alternatives include total-variation regularised differentiation [8, 33]. Our numerical tests use Savitzky–Golay differentiation as a shared geometry-learning procedure across the experiments because it provides accurate hyperplane recovery at modest computational cost.

We then compute the velocity jump $\Delta v _ { h , j } ^ { n } = v _ { h , j + 1 } ^ { n } - v _ { h , j } ^ { n }$ . Note that Equation (5) and Equation (6) are linear estimators, that is, they are of the form

$$
\boldsymbol { v } _ { h , j } ^ { n } = \sum _ { k = - r } ^ { r } \boldsymbol { w } _ { k } \boldsymbol { y } _ { j + k } ^ { n }\tag{7}
$$

for some cut-off $r \in \mathbb N$ and weights $w _ { k } \in \mathbb { R } , k = - r , . . . , r$ . Linear estimators are inexpensive and easy to analyse, but can be sensitive to observational noise, as illustrated in Section 5.4.

Algorithm overview. Algorithm 1 states the geometry-learning procedure for a single dominant affine switching hyperplane, which is the setting used in the main experiments. Having computed the jumps, we rank all the derivativejump magnitudes globally and retain the largest fraction $\rho \in ( 0 , 1 )$ . For each retained jump interval, we also include neighbouring intervals, since a true crossing may affect several adjacent velocity estimates. The endpoints of these intervals form an overcomplete candidate cloud . This cloud contains points near the switching hyperplane, but also outliers caused by curvature, noise, and derivative-estimation artefacts. We proceed with the hyperplane recovery only if there are enough candidates, i.e., $| \mathcal { C } | \geq \operatorname* { m a x } \{ 3 , m _ { \operatorname* { m i n } } , d \}$ with $m _ { \mathrm { m i n } } = 3 0$ for the single-hyperplane recovery tasks. We therefore fit the hyperplane using random sample consensus (RANSAC) [17], a robust randomised fitting procedure that repeatedly proposes a hyperplane from a minimal subset of points, counts the number of candidate points within a prescribed distance tolerance, and keeps the hyperplane with the largest consensus set. We then refine the selected inliers by total least squares. See Appendix A for more background on RANSAC.

Remark 3.1. This stage requires the observed trajectories to produce velocity jumps that dominate smooth curvature and derivative-estimation error. Transversal crossings and transitions into or out of sliding can provide such a signal, whereas grazing contacts and trajectories confined to one side of the switching hyperplane may not. For example, in $\dot { x } = \mathrm { s i g n } ( y ) , \dot { y } = 0$ , the vector field is discontinuous across $\Sigma = \{ y = 0 \}$ , but every trajectory has constant y and therefore produces no derivative jump. Thus, identifiability requires both a sufficiently large jump and sufficient interaction between the observed trajectories and Σ. We analyse the jump-to-noise balance in Section 4.1 and test it numerically in Section 5.

A simple extension for recovering multiple switching hyperplanes is to apply the procedure sequentially, removing inliers associated with the strongest hyperplane recovered before fitting the next one. This produces an arrangement of recovered affine interfaces, rather than an explicit reconstruction of the polygonal cells, and is intended for wellseparated interfaces. We illustrate this sequential extension and a support-based estimate of the number of hyperplanes in Appendix E.

Algorithm 1 Single-hyperplane recovery   
Require: Warmup trajectories $\{ y _ { j } ^ { n } \} _ { n = 1 , j = 0 } ^ { N , J } \subset \mathbb { R } ^ { d } ;$ step size h; Savitzky–Golay window w and polynomial degree p;   
retained jump fraction $\rho \in ( 0 , 1 )$ and neighbouring interval radius $R ;$ RANSAC tolerance $\varepsilon ;$ RANSAC iterations   
$M ;$ minimum support $m _ { \mathrm { m i n } }$   
Ensure: Unit normal $\bar { \boldsymbol { a } } \in \mathbb { R } ^ { d }$ and offset $b \in$ R defining the recovered switching hyperplane $\Gamma = \{ \boldsymbol { x } \in \mathbb { R } ^ { d } : a ^ { \top } \boldsymbol { x } + b =$   
$0 \}$   
1: Estimate velocities with Savitzky–Golay differentiation:   
$\widehat { v } _ { j } ^ { n } \gets \mathrm { S G } _ { w , p } [ y ^ { n } ] _ { j } ,$ for $n = 1 , \ldots , N$ and $j = 0 , \ldots , J .$   
b2: Compute derivative-jump magnitudes   
$\bar { J } _ { n } ( j )  \| \widehat { v } _ { j + 1 } ^ { n } - \widehat { v } _ { j } ^ { \hat { n } } \| _ { 2 } , \mathrm { f o r } j = 0 , \ldots , J - 1 .$   
3: Rank all jump intervals $( n , j )$ globally by $J _ { n } ( j )$   
4: Set $L  N { \dot { J } } .$   
5: Keep the largest max $\{ 1 , \operatorname { r o u n d } ( \rho L ) \}$ jump intervals.   
6: Add the immediate neighbouring intervals of each retained jump:   
$\mathcal { E }  \{ ( n , \ell ) : \ell \in \bar { \{ j }  - R , \bar { \ldots } , j , \ldots , j + R \} , \ 0 \leq \ell < \bar { J } , \ ( n , j )$ is retained .   
7: Build the candidate cloud from both endpoints of every selected interval:   
${ \mathcal { C } } \gets \{ y _ { \ell } ^ { n } , y _ { \ell + 1 } ^ { n } : ( n , \ell ) \in \mathcal { E } \}$   
8: Remove duplicate points from ${ \dot { C } } .$   
9: Set $m _ { \mathrm { r e q } }  \mathrm { m a x } \{ \bar { 3 } , m _ { \mathrm { m i n } } , d \}$   
10: if $| { \mathcal { C } } | < { \mathrm { \bar { \it m } } } _ { \mathrm { r e q } }$ then   
11: fail.   
12: end if   
13: $( a _ { \star } , b _ { \star } , \mathcal { T } _ { \star } ) \gets \mathrm { R A N S A C } ( \mathcal { C } , \varepsilon , M , m _ { \mathrm { r e q } } ) .$ ▷ For a description of RANSAC see Appendix A.   
14: if RANSAC fails then   
15: fail.   
16: end if   
17: Fit $( a , b )$ by total least squares on $\mathcal { T } _ { \star } .$ ▷ $\mathcal { T } _ { \star }$ is the inlier set of RANSAC.   
18: Normalise $( a , b )$ so that $\mathbf { \hat { \Vert } } a \Vert _ { 2 } = 1$ and fix a deterministic sign convention.   
19: Recompute the refined consensus set   
${ \mathcal { T } }  \{ x \in { \mathcal { C } } : | a ^ { \top } x + b | \leq \varepsilon \}$   
20: if $| \mathcal { T } | < m _ { \mathrm { r e q } }$ then   
21: fail.   
22: end if   
23: return $( a , b )$

## 3.3 Learning the dynamics

Once an approximate geometry has been obtained, we need to approximate the dynamics. In this work, geometry recovery and dynamics learning are performed sequentially: after estimating the polyhedral partition $P ,$ we fix it during the training of the vector field. If the switching geometry is known from modelling considerations, the geometry-learning stage can be skipped, and the same dynamics-learning procedure can be applied directly on the prescribed partition.

We denote the recovered partition by $\widetilde { P } = \{ \widetilde { \mathcal { P } } _ { 1 } , \dots , \widetilde { \mathcal { P } } _ { \widetilde { K } } \}$ , allowing its number of regions $\widetilde { K }$ to differ from that of the target partition $K .$ e e e eIn our experiments, we assume we know the number of hyperplanes. However, in Appendix E, we show that such a number can be detected during the geometry-learning phase. The goal now is to define an approximate vector field of the form

$$
\begin{array} { r } { \dot { x } = \tilde { f } ( x , t ) : = \tilde { f } _ { k } ( x , t ) , \quad x \in \widetilde { \mathcal { P } } _ { k } , \ k = 1 , \dots , \widetilde { K } . } \end{array}
$$

Depending on the available structural knowledge, we choose a class $\mathcal { F } _ { \widetilde { P } } ^ { d }$ for $\tilde { f } .$ We call this class adapted to the partition $\widetilde { P }$ if every element has discontinuities only on the hyperplanes defining the partition ${ \widetilde { P } } .$ This prevents the learned vector e efield from creating spurious switching hyperplanes away from the recovered geometry. For example, if the vector field is known to be autonomous and piecewise affine, we could set

$$
\mathcal { F } _ { \widetilde { P } } ^ { d } = \{ \widetilde { f } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d } \ | \ \widetilde { f } ( x ) = \sum _ { k = 1 } ^ { \widetilde { K } } 1 _ { \widetilde { \mathcal { P } } _ { k } } ( x ) ( B _ { k } x + c _ { k } ) , \qquad B _ { k } \in \mathbb { R } ^ { d \times d } , \quad c _ { k } \in \mathbb { R } ^ { d } \} .
$$

When lacking further structural knowledge on the target dynamics, we model the local maps $\tilde { f } _ { k }$ using sufficiently flexible neural networks. In Section 4.2, we show how to realise such partition-adapted neural vector fields without explicitly evaluating indicator functions.

Once an approximation space is defined, we use the available data to recover a vector field that reproduces the behaviour observed in the training data. To do so, we minimise $\mathcal { L } _ { \mathrm { t r a j } } ( g )$ over $g \in \mathcal { F } _ { \widetilde { P } } ^ { d }$ , where

$$
\mathcal { L } _ { \mathrm { t r a j } } ( g ) = \frac { 1 } { N J d } \sum _ { n = 1 } ^ { N } \sum _ { j = 1 } ^ { J } \left. y _ { j } ^ { n } - \varphi _ { g } ( t _ { j } ^ { n } ; t _ { 0 } ^ { n } , \xi _ { n } ) \right. ^ { 2 } ,\tag{8}
$$

$\varphi _ { g } ( t ; t _ { 0 } , \xi )$ denotes a numerical approximation at time t of the corresponding solution of

$$
\dot { z } ( \tau ) = g ( z ( \tau ) , \tau ) , ~ z ( t _ { 0 } ) = \xi .
$$

To improve the training process, we increase the length of the trajectory segments J gradually during training. We call this curriculum training and provide more details in Section 5. Here we set $\xi _ { n } = y _ { 0 } ^ { n }$ , the noisy observed initial condition. The numerical method $\varphi$ that we consider is described in Section $^ { 5 , }$ and the precise class of neural vector fields that we use is defined in Section 4.

In some numerical experiments, we augment the trajectory-matching loss above with velocity-based auxiliary terms. Let denote the set of trajectory indices $( n , j )$ considered by the auxiliary losses. Let $v _ { h , j } ^ { n }$ denote a velocity estimate computed from the observed trajectory using one of the linear estimators in Equations (5) to (7). We define

$$
\mathcal { L } _ { \mathrm { v f } } ( g ) = \frac { 1 } { d \lvert \mathcal { T } _ { \mathrm { v f } } \rvert } \sum _ { ( n , j ) \in \mathcal { T } _ { \mathrm { v f } } } \left. \left. g ( y _ { j } ^ { n } , t _ { j } ^ { n } ) - v _ { h , j } ^ { n } \right. \right. ^ { 2 } ,\tag{9}
$$

where $\mathcal { T } _ { \mathrm { v f } } \subseteq \mathcal { T }$ denotes the set of samples used for vector-field supervision and $| \mathcal { T } _ { \mathrm { v f } } |$ the number of such samples. Thi set may exclude samples close to the switching set, where velocity estimates are least reliable.

The numerical experiments using these directional regularisers have a single switching hyperplane with unit normal a. For arrangements of several hyperplanes, analogous penalties could be computed for each hyperplane and averaged. Let

$$
\Pi _ { \mathrm { t a n } } = I _ { d } - a a ^ { \top }
$$

be the orthogonal projection onto the tangent space of the hyperplane, and define

$$
\begin{array} { r l r } { \mathcal { Z } _ { \mathrm { n o r } } = \{ ( n , j ) \in \mathbb { Z } : | v _ { h , j } ^ { n } \cdot a | > \tau _ { \mathrm { n o r } } \} , } & { } & { \mathcal { Z } _ { \mathrm { t a n } } = \{ ( n , j ) \in \mathbb { Z } : \| \Pi _ { \mathrm { t a n } } v _ { h , j } ^ { n } \| _ { 2 } > \tau _ { \mathrm { t a n } } \} . } \end{array}
$$

The normal regulariser penalises a predicted normal component that points in the opposite direction to the estimated velocity,

$$
\mathcal { L } _ { \mathrm { n o r } } ( g ) = \frac { 1 } { | \mathcal { Z } _ { \mathrm { n o r } } | } \sum _ { ( n , j ) \in \mathcal { I } _ { \mathrm { n o r } } } \operatorname* { m a x } \left\{ 0 , - \left( g ( y _ { j } ^ { n } , t _ { j } ^ { n } ) \cdot a \right) \mathrm { s i g n } ( v _ { h , j } ^ { n } \cdot a ) \right\} ,\tag{10}
$$

while the tangential regulariser matches the tangential components,

$$
\mathcal { L } _ { \mathrm { t a n } } ( g ) = \frac { 1 } { d \vert \mathcal { Z } _ { \mathrm { t a n } } \vert } \sum _ { ( n , j ) \in \mathcal { Z } _ { \mathrm { t a n } } } \left. \Pi _ { \mathrm { t a n } } g ( y _ { j } ^ { n } , t _ { j } ^ { n } ) - \Pi _ { \mathrm { t a n } } v _ { h , j } ^ { n } \right. ^ { 2 } .\tag{11}
$$

The general objective used in the numerical experiments is therefore

$$
\mathcal { I } ( g ) = \lambda _ { \mathrm { t r a j } } \mathcal { L } _ { \mathrm { t r a j } } ( g ) + \lambda _ { \mathrm { v f } } \mathcal { L } _ { \mathrm { v f } } ( g ) + \lambda _ { \mathrm { n o r } } \mathcal { L } _ { \mathrm { n o r } } ( g ) + \lambda _ { \mathrm { t a n } } \mathcal { L } _ { \mathrm { t a n } } ( g ) ,\tag{12}
$$

with the weights, thresholds, and active terms specified in Section 5.

## 4 Theoretical analysis

We now analyse certain theoretical aspects of the methodology. First, we study how observational noise interplays with the properties of jumps across the switching hyperplane in the context of geometry learning, and then we study the approximation properties of a class of piecewise-smooth neural networks later used in the experiments.

## 4.1 Analysis of geometry learning

In this section, we address some theoretical aspects of the geometry learning described in Section 3.2. Here, we consider the simplified case where $\mathbb { R } ^ { d }$ is separated by one hyperplane, i.e., we suppose that

$$
\Sigma : = \left\{ x \in \mathbb { R } ^ { d } : \ s ( x ) = 0 \right\} , \qquad s ( x ) : = a ^ { \top } x + b ,
$$

where $a \in \mathbb { R } ^ { d } \setminus \{ 0 \}$ and $b \in \mathbb { R } .$ , and the goal is to recover a and b.

Velocity-based discontinuity learning with linear velocity estimation methods may be sensitive to noise. Since the observations are affected by random noise, they are random variables, and consequently so are the velocity estimates. As linear velocity estimators rely on numerical differentiation, they amplify high-frequency components of the data, such as noise. The detection of Σ also relies on locally amplifying the velocity jump. If the noise magnitude is too $\mathrm { \ h i g h } ,$ a linear velocity estimator cannot detect jumps across the boundary. The amplification has a floor that no choice of weights can beat. For an estimator $\begin{array} { r } { v _ { h , j } ^ { n } = \sum _ { k = - r } ^ { r } w _ { k } y _ { j + k } ^ { n } } \end{array}$ applied to data of the form (4), the random part $\scriptstyle \sum _ { k = - r } ^ { r } w _ { k } \bar { \eta _ { j + k } ^ { n } }$ is a linear combination of independent Gaussians, so

$$
v _ { h , j } ^ { n } - \sum _ { k = - r } ^ { r } w _ { k } x _ { * } ^ { n } ( t _ { j + k } ^ { n } ) \sim N \left( 0 , \sigma ^ { 2 } \lVert w \rVert _ { 2 } ^ { 2 } I _ { d } \right) , \qquad \lVert w \rVert _ { 2 } ^ { 2 } = \sum _ { k = - r } ^ { r } w _ { k } ^ { 2 } .
$$

The consistency conditions $\textstyle \sum _ { k } w _ { k } = 0$ and $\begin{array} { r } { \sum _ { k } w _ { k } k h = 1 } \end{array}$ require, by Cauchy–Schwarz,

$$
1 = \bigg | \sum _ { k = - r } ^ { r } w _ { k } k h \bigg | \leq h \sqrt { \sum _ { k = - r } ^ { r } w _ { k } ^ { 2 } } \sqrt { \sum _ { k = - r } ^ { r } k ^ { 2 } } = h \| w \| _ { 2 } \sqrt { S _ { r } } , \qquad S _ { r } = \frac { r ( r + 1 ) ( 2 r + 1 ) } { 3 } ,
$$

so $\| w \| _ { 2 } \geq 1 / ( h \sqrt { S _ { r } } )$ . This floor decreases as the window radius r grows, but increases as the step size h shrinks. It explains why a Savitzky–Golay estimator is more robust to noise than central differences, and also why it will still struggle when σ is too large.

The noise floor on a single velocity estimate has consequences for which jumps across $\Sigma$ are detectable. The next result combines this floor with the smoothing error incurred by the choice of window size r, addressing which jump sizes are detectable. To this end, consider the estimated velocity jump

$$
\Delta v _ { h , j } ^ { n } = v _ { h , j + 1 } ^ { n } - v _ { h , j } ^ { n } = \sum _ { k = - r } ^ { r } w _ { k } y _ { j + 1 + k } ^ { n } - \sum _ { k = - r } ^ { r } w _ { k } y _ { j + k } ^ { n } = \sum _ { k = - r } ^ { r + 1 } c _ { k } y _ { j + k } ^ { n } ,
$$

where

$$
c _ { k } = \left\{ { \begin{array} { l l } { - w _ { - r } \quad { \mathrm { i f ~ } } k = - r , } \\ { w _ { r } \quad { \mathrm { i f ~ } } k = r + 1 , } \\ { w _ { k - 1 } - w _ { k } \quad { \mathrm { e l s e } } . } \end{array} } \right.
$$

If the velocity estimator satisfies the same consistency conditions $\begin{array} { r } { \sum _ { k } w _ { k } = 0 , \sum _ { k } w _ { k } k h = 1 } \end{array}$ , then $\scriptstyle \sum _ { k = - r } ^ { r + 1 } c _ { k } = 0$ and $\begin{array} { r } { \sum _ { k = - r } ^ { r + 1 } c _ { k } k h = 0 , \mathrm { i . e . } } \end{array}$ ., no jump is detected on constant and constant-velocity trajectories. Further, it holds that $\| c \| _ { 2 } \leq 2 \| w \| _ { 2 }$ . We are now ready to state the proposition.

Proposition 4.1. Consider the velocity estimator $\begin{array} { r } { v _ { h , j } ^ { n } \ = \ \sum _ { k = - r } ^ { r } w _ { k } y _ { j + k } ^ { n } } \end{array}$ satisfying the consistency conditions $\begin{array} { r } { \sum _ { k } w _ { k } = 0 , \sum _ { k } w _ { k } k h = 1 } \end{array}$ . Let $f ^ { + }$ and $f ^ { - }$ be smooth and extendable to the whole domain. Assume that a reference trajectory $x _ { * } ^ { n }$ crosses $\Sigma$ transversally at one single time $\tau ,$ and that without loss of generality, $s ( x ) < 0$ before the crossing. Set $p = x _ { * } ^ { n } ( \tau ) \in \Sigma .$ . Denote by $x ^ { + }$ the solution to ${ \dot { x } } = f ^ { + } ( x , t )$ , with initial time τ and starting value $x ^ { + } ( \tau ) = p ,$ and denote by $x ^ { - }$ the trajectory of $f ^ { - }$ agreeing with $x _ { * } ^ { n }$ before the switch. By continuity, $x ^ { - } ( \tau ) = x ^ { + } ( \dot { \tau } ) \dot { = } p .$ The velocity jump is given by

$$
f ^ { + } ( p , \tau ) - f ^ { - } ( p , \tau ) = \kappa e , \quad \kappa > 0 , e \in \mathbb { R } ^ { d } ,
$$

where $\kappa = \| f ^ { + } ( p , \tau ) - f ^ { - } ( p , \tau ) \| _ { 2 }$ and $\| e \| _ { 2 } = 1$ . Assume that the second derivatives of both $x ^ { + }$ and $x ^ { - }$ are uniformly bounded by M. Let

$$
R = \sum _ { k = - r } ^ { r + 1 } c _ { k } \operatorname* { m a x } \left( t _ { j + k } - \tau , 0 \right) , \alpha = M \sum _ { k : t _ { j + k } > \tau } | c _ { k } | ( t _ { j + k } - \tau ) ^ { 2 } , \beta = \frac { M } { 2 } \sum _ { k = - r } ^ { r + 1 } | c _ { k } | ( t _ { j + k } - t _ { j } ) ^ { 2 } .
$$

Consider now two scenarios. In the first, the trajectory jumps in the stencil used to compute $\Delta v _ { h , j } ^ { n } ,$ , and in the second, it follows $x ^ { - }$ throughout. For simplicity, denote the first jump estimate by $\Delta _ { 1 }$ , and the second by $\Delta _ { 2 }$ . Then,

1. Both $\Delta _ { 1 }$ and $\Delta _ { 2 }$ are Gaussian, that is, $\Delta _ { i } \sim \mathsf { N } ( \mu _ { i } , C )$ , with

$$
\begin{array} { r } { C = \sigma ^ { 2 } \| c \| _ { 2 } ^ { 2 } I _ { d } , \mu _ { 1 } - \mu _ { 2 } = \kappa R e + \varrho , } \end{array}
$$

where $\varrho \in \mathbb { R } ^ { d }$ satisfies $\| \varrho \| _ { 2 } \leq \alpha$

2. The expected magnitudes satisfy

$$
\begin{array} { r } { \mathbb { E } [ \| \Delta _ { 1 } \| _ { 2 } ] \ge \kappa | R | - \beta - \alpha , \quad \mathbb { E } [ \| \Delta _ { 2 } \| _ { 2 } ] \le \beta + \sigma \| c \| _ { 2 } \sqrt { d } , } \end{array}
$$

so the expected jump magnitude exceeds the expected smooth magnitude whenever κ $R | > 2 \beta + \alpha + \sigma \| c \| _ { 2 } \sqrt { d } .$

The constant R measures how much of the post-crossing displacement the jump stencil captures. The constants α and $\beta$ are second-derivative Taylor remainders, both controlled by $M \colon \beta$ bounds the spurious jump produced by a smooth trajectory of nonzero acceleration, while α bounds the error in approximating the post-crossing displacement by its leading linear-in-time term. Detectability therefore requires the scaled jump $\kappa | R |$ to dominate $\alpha , \beta ,$ and the noise term $\sigma \| c \| _ { 2 } { \sqrt { d } }$

Proof. As above, the randomness is introduced by the estimator being applied to the observational noise. Due to the linearity, the resulting quantity is also normally distributed, that is,

$$
\sum _ { k = - r } ^ { r + 1 } c _ { k } \eta _ { j + k } ^ { n } \sim \mathsf { N } ( 0 , \sigma ^ { 2 } \| c \| _ { 2 } ^ { 2 } I _ { d } ) .
$$

Consider now $\Delta _ { \mathrm { 2 } } , \mathrm { i . e . }$ , when the stencil contains no jumps. In that case, we Taylor expand and write

$$
x ^ { - } ( t _ { j + k } ) = x ^ { - } ( t _ { j } ) + k h \dot { x } ^ { - } ( t _ { j } ) + r _ { k } ,
$$

where $\begin{array} { r } { r _ { k } = \int _ { t _ { i } } ^ { t _ { j + k } } ( t _ { j + k } - s ) \ddot { x } ^ { - } ( s ) } \end{array}$ ds, so that $\begin{array} { r } { \| r _ { k } \| _ { 2 } \le \frac { M } { 2 } ( t _ { j + k } - t _ { j } ) ^ { 2 } } \end{array}$ . The annihilation property of the velocity jump estimator means that

$$
\mu _ { 2 } = \sum _ { k = - r } ^ { r + 1 } c _ { k } r _ { k } ,
$$

which satisfies

$$
\| \mu _ { 2 } \| _ { 2 } \leq \frac { M } { 2 } \sum _ { k = - r } ^ { r + 1 } | c _ { k } | ( t _ { j + k } - t _ { j } ) ^ { 2 } = \beta .
$$

The case when there is a jump is similar. The deterministic trajectory is $x ^ { - } ( t ) + ( x ^ { + } ( t ) - x ^ { - } ( t ) ) \mathbf { 1 } _ { t > \tau }$ , where $\mathbf { 1 } _ { t > \tau }$ is 1 if $t > \tau$ and 0 elsewhere. Since $x ^ { + } ( t ) - x ^ { - } ( t )$ is smooth, we Taylor expand around τ and obtain

$$
x ^ { + } ( t _ { j + k } ) - x ^ { - } ( t _ { j + k } ) = \kappa e ( t _ { j + k } - \tau ) + \varrho _ { k } ,
$$

where $\begin{array} { r } { \varrho _ { k } = \int _ { \tau } ^ { t _ { j + k } } ( \ddot { x } ^ { + } ( s ) - \ddot { x } ^ { - } ( s ) ) ( t _ { j + k } - s ) } \end{array}$ ds. Since $\| \ddot { x } ^ { + } - \ddot { x } ^ { - } \| _ { 2 } \leq 2 M$ , we see that $\| \varrho _ { k } \| _ { 2 } \le M ( t _ { j + k } - \tau ) ^ { 2 }$ Summing this up, we obtain

$$
\mu _ { 1 } = \mu _ { 2 } + \sum _ { k : t _ { j + k } > \tau } c _ { k } \left[ \kappa e ( t _ { j + k } - \tau ) + \varrho _ { k } \right] = \kappa e R + \sum _ { k : t _ { j + k } > \tau } c _ { k } \varrho _ { k } ,
$$

where we have that

$$
\left\| \sum _ { k : t _ { j + k } > \tau } c _ { k } \varrho _ { k } \right\| _ { 2 } \leq M \sum _ { k : t _ { j + k } > \tau } | c _ { k } | ( t _ { j + k } - \tau ) ^ { 2 } = \alpha .
$$

For the second statement, first note that by the triangle inequality and Jensen’s inequality,

$$
\begin{array} { r } { \mathbb { E } [ \| \Delta _ { 2 } \| _ { 2 } ] \le \| \mu _ { 2 } \| _ { 2 } + \sigma \| c \| _ { 2 } \sqrt { d } \le \beta + \sigma \| c \| _ { 2 } \sqrt { d } . } \end{array}
$$

For the second bound, Jensen’s inequality using the norm function, followed by the reverse triangle inequality, gives

$$
\begin{array} { r } { \mathbb { E } [ \| \Delta _ { 1 } \| _ { 2 } ] \ge \| \mathbb { E } [ \Delta _ { 1 } ] \| _ { 2 } \ge \kappa | R | - \beta - \alpha . } \end{array}
$$

In practice, this means that the gap between the jump magnitude and the noise standard deviation must be sufficiently large to distinguish smooth points from true jumps, and that sensitivity decreases as the dimension grows. We numerically test the balance between the discontinuity gap and the noise magnitude in Section 5.

Remark 4.1. The proposition above is a local signal-detection result. It assumes that the crossing time τ lies in the range of the velocity-jump stencil and gives a condition under which the resulting derivative jump is distinguishable from smooth curvature and observation noise. The subsequent hyperplane-fitting step uses the corresponding neighbouring trajectory points as candidates for Σ. This introduces a separate conditioning issue. Points with small signed distance to Σ need not be tightly localised around the crossing time if the normal crossing velocity is small.

Indeed, let $p = x ( \tau )$ and suppose, for simplicity, that $\Sigma = \{ x : s ( x ) = 0 \}$ is a hyperplane with $s ( x ) = a ^ { \top } x + b$ and $\| a \| _ { 2 } = 1$ . For a nearby time t on either side of the crossing,

$$
s ( x ( t ) ) = ( t - \tau ) a ^ { \top } f ^ { \mathrm { p r e / p o s t } } ( p , \tau ) + \mathcal { O } ( | t - \tau | ^ { 2 } ) .
$$

Thus, the signed-distance signal scales like $| t - \tau | { \boldsymbol { v } } _ { \mathrm { n o r } }$ , where

$$
v _ { \mathrm { n o r } } = \operatorname* { m i n } \left\{ | a ^ { \top } f ^ { \mathrm { p r e } } ( p , \tau ) | , | a ^ { \top } f ^ { \mathrm { p o s t } } ( p , \tau ) | \right\} .
$$

With observations $y ( t ) = x ( t ) + \eta ( t )$ and $\eta ( t ) \sim \mathsf { N } ( 0 , \sigma ^ { 2 } I _ { d } )$ , one has

$$
\begin{array} { r } { s ( y ( t ) ) - s ( x ( t ) ) = a ^ { \top } \eta ( t ) \sim \mathsf { N } ( 0 , \sigma ^ { 2 } ) . } \end{array}
$$

Consequently, the time interval over which the clean signed distance is comparable to the observation noise has a width of order $\sigma / v _ { \mathrm { n o r } }$ . Indeed, this is the regime where $| t - \tau | v _ { \mathrm { n o r } } \approx \sigma$ , so the observed signed distance $s ( y ( t ) )$ is strongly affected by noise and is no longer a reliable proxy for the clean distance to $\Sigma$ . When $v _ { \mathrm { n o r } }$ is small, a larger portion of the trajectory near the crossing is ambiguous from the point of view of geometric localisation. Thus, the jump-to-noise condition above should be read together with a non-grazing condition: the normal crossing velocity must be large enough, relative to the sampling step and observation noise, for the candidate cloud to localise Σ reliably

## 4.2 Universal neural network for piecewise-smooth dynamical systems

Let us fix a polyhedral partition $P = \{ \mathcal { P } _ { 1 } , \ldots , \mathcal { P } _ { K } \}$ of $\mathbb { R } ^ { d }$ induced by a finite affine hyperplane arrangement. We now define a neural class $\mathcal { F } _ { P } ^ { d }$ adapted to this partition. The class has two key properties: its discontinuities are confined to the prescribed arrangement, and it can approximate piecewise-smooth vector fields arbitrarily well almost everywhere on compact sets.

Let ReL $\mathrm { U } ( x ) = \operatorname* { m a x } \{ 0 , x \}$ denote the ReLU function, and $\mathcal { H } ( x )$ the Heaviside step function. We use the convention $\mathcal { H } ( r ) = 1$ for $r \geq 0$ and $\mathcal { H } ( r ) = 0$ for $r < 0$ , applied componentwise to vectors. For a vector $J \in \mathbb { R } ^ { k }$ , we define the nonlinear function $\mathrm { R e L U } _ { J } : \mathbb { R } ^ { k }  \mathbb { R } ^ { k }$ by $\mathrm { R e L U } _ { J } ( x ) = \mathrm { R e L U } ( x ) + J \odot { \mathcal { H } } ( x )$ , as in [13]. For fixed $d _ { 1 } , d _ { 2 } \in \mathbb { N }$ , we introduce the networks

$$
\begin{array} { r l } & { \mathcal { F } _ { \mathrm { R e L U } , \iota } ^ { d _ { 1 } , d _ { 2 } } : = \Big \{ \mathcal { N } : \mathbb { R } ^ { d _ { 1 } } \to \mathbb { R } ^ { d _ { 2 } } \Big | \mathcal { N } = W _ { L } \circ \mathrm { R e L U } _ { J _ { L } } \circ \cdots \circ W _ { 1 } \circ \mathrm { R e L U } _ { J _ { 1 } } \circ W _ { 0 } , } \\ & { \qquad \quad L \in \mathbb { N } , W _ { \ell } : \mathbb { R } ^ { h _ { \ell } } \to \mathbb { R } ^ { h _ { \ell + 1 } } \mathrm { a f f i n e } , \ \ell = 0 , \ldots , L } \\ & { \qquad \quad J _ { \ell } \in \mathbb { R } ^ { h _ { \ell } } , \ \ell = 1 , \ldots , L , \ h _ { 0 } = d _ { 1 } , \ h _ { L + 1 } = d _ { 2 } , \ h _ { 1 } , \ldots , h _ { L } \in \mathbb { N } \Big \} . } \end{array}
$$

We denote by $\mathcal { F } _ { \mathrm { R e L U } } ^ { d _ { 1 } , d _ { 2 } } \subset \mathcal { F } _ { \mathrm { R e L U } . } ^ { d _ { 1 } , d _ { 2 } }$ the subclass for which all jump parameters vanish, i.e., the set of deep ReLU feedforward networks. In this section, we repeatedly use the identities

$$
\begin{array} { r l } { \mathcal { H } ( x ) = [ I _ { d } } & { - I _ { d } ] \operatorname { R e L U } _ { J } \left( \left[ I _ { d } \right] x \right) , \qquad x \in \mathbb { R } ^ { d } , J = \left[ \overset { 1 } { 0 _ { d } } \right] \in \mathbb { R } ^ { 2 d } , } \end{array}\tag{13}
$$

and

$$
x = [ I _ { d } - I _ { d } ] { \mathrm { R e L U } } _ { J } \left( \left[ { \frac { I _ { d } } { - I _ { d } } } \right] x \right) , \qquad x \in \mathbb { R } ^ { d } , J = \left[ { \begin{array} { l } { 0 _ { d } } \\ { 0 _ { d } } \end{array} } \right] \in \mathbb { R } ^ { 2 d } .\tag{14}
$$

In particular, $\mathcal { H } \in \mathcal { F } _ { \mathrm { R e L U _ { \it i } } } ^ { d , d }$ and the identity map belongs to $\mathcal { F } _ { \mathrm { R e L U } } ^ { d , d }$

Lemma 4.1 (Closure under affine combinations and parallel concatenation). For every choice of dimensions, both $\mathcal { F } _ { \mathrm { R e L U _ { j } } } ^ { d _ { 1 } , d _ { 2 } }$ and $\mathcal { F } _ { \mathrm { R e L U } } ^ { d _ { 1 } , d _ { 2 } }$ are closed under multiplication with a scalar, linear combination, and parallel concatenation.

The proof follows similar ideas as in [29, Section 5.1], and it can be found in Appendix B.

Adopting the notation in (1), we set

$$
d _ { P } : = \sum _ { k = 1 } ^ { K } m _ { k } , \qquad A _ { P } : = \left[ \begin{array} { l } { A _ { 1 } } \\ { \vdots } \\ { A _ { K } } \end{array} \right] , \qquad b _ { P } : = \left[ \begin{array} { l } { b _ { 1 } } \\ { \vdots } \\ { b _ { K } } \end{array} \right] , \qquad T _ { P } ( x ) : = b _ { P } - A _ { P } x \in \mathbb { R } ^ { d _ { P } } ,
$$

and define $z _ { P } ( x ) = \mathcal { H } ( T _ { P } ( x ) ) \in \{ 0 , 1 \} ^ { d _ { P } }$

$$
\Gamma _ { P } : = \bigcup _ { \ell = 1 } ^ { d _ { P } } \left\{ x \in \mathbb { R } ^ { d } : ( T _ { P } ( x ) ) _ { \ell } = 0 \right\} .
$$

Thus $z _ { P }$ is locally constant on $\mathbb { R } ^ { d } \backslash \Gamma _ { P }$

Definition 4.1 (Geometry-preserving class associated with $P )$ . For the fixed state dimension d, we define $\mathcal { F } _ { P } ^ { d }$ as the set of maps

$$
( x , t ) \mapsto \Psi \bigl ( x , t , \Phi ( z _ { P } ( x ) ) \bigr )
$$

such that

$$
\Phi \in \mathcal { F } _ { \mathrm { R e L U } _ { J } } ^ { d _ { P } , d _ { 1 } } , \qquad \Psi \in \mathcal { F } _ { \mathrm { R e L U } } ^ { 1 + d + d _ { 1 } , d } , \qquad d _ { 1 } \in \mathbb { N } .
$$

Remark 4.2. Time-independent maps of the form $x \mapsto \Psi ( x , \Phi ( z _ { P } ( x ) ) )$ with $\Psi \in \mathcal { F } _ { \mathrm { R e L U } } ^ { d + d _ { 1 } , d } , \Phi \in \mathcal { F } _ { \mathrm { R e L U } , I } ^ { d _ { P } , d _ { 1 } } , d _ { 1 } \in \mathbb { N } .$ , can also be realised as maps in $\mathcal { F } _ { P } ^ { d }$ . If $\mathcal { N } \in \mathcal { F } _ { P } ^ { d }$ is time-independent, we write ${ \mathcal { N } } ( x )$ and avoid including the variable t for conciseness.

Proposition 4.2 (No new jumps). Every element of $: \mathcal { F } _ { P } ^ { d }$ is continuous on $( \mathbb { R } ^ { d } \setminus \Gamma _ { P } ) \times \mathbb { R } .$

Proof. Let $\mathcal { N } \in \mathcal { F } _ { P } ^ { d } . \mathrm { I f } x _ { 0 } \notin \Gamma _ { P } .$ , then $z _ { P }$ is locally constant near $x _ { 0 }$ . Therefore, near $x _ { 0 } .$ , the map $x \mapsto \Phi ( z _ { P } ( x ) )$ is constant, while $( x , \dot { t } ) \mapsto \Psi ( x , t , \Phi ( z _ { P } ( x ) ) )$ is the composition of continuous maps. Thus $\mathcal { N }$ is continuous at $( x _ { 0 } , t )$ for every $t \in \mathbb { R }$ □

Proposition 4.3 (Linearity of the geometry-preserving class). $\mathcal { F } _ { P } ^ { d }$ is a vector space over $\mathbb { R } .$

The proof follows from the vector space property of $\begin{array} { r l } { \mathcal { F } _ { \mathrm { R e L U } _ { * } } ^ { d _ { 1 } , d _ { 2 } } } & { { } } \end{array}$ . We include it in Appendix $\mathbf { B } .$

Theorem 4.1 (Exact ReLU gating on compact sets). $L e t \Omega \subset \mathbb { R } ^ { d }$ be compact, $T > 0 _ { i }$ , and $R \in \mathcal { F } _ { \mathrm { R e L U } } ^ { d + 1 , m }$ . Then there exists $\widetilde { R } \in \mathcal { F } _ { \mathrm { R e L U } } ^ { d + 2 , m }$ such that

$$
\widetilde { R } ( x , t , s ) = \left\{ \begin{array} { l l } { R ( x , t ) , } & { s = 1 , } \\ { 0 , } & { s = 0 , } \end{array} \right. \quad \forall ( x , t , s ) \in \Omega \times [ 0 , T ] \times \{ 0 , 1 \} .
$$

Since affine maps of the form $( x , t ) \mapsto A x + b \in \mathbb { R } ^ { m }$ belong to $\mathcal { F } _ { \mathrm { R e L U } } ^ { d + 1 , m }$ , the result above applies to affine timeindependent maps as well.

Proof. Since R is continuous, and [0, T] and Ω are compact,

$$
\lambda : = \operatorname* { s u p } _ { \stackrel { x \in \Omega } { t \in [ 0 , T ] } } \| R ( x , t ) \| _ { \infty } < \infty .
$$

We propagate the binary variable s unchanged through the network while computing $R ( x , t )$ in parallel. Since $s \in \mathbf { \bar { \{ 0 , 1 \} } } \mathbf { \bar { \subset } } \left[ 0 , \infty \right)$ , one has ReL $\operatorname { I U } ( s ) = s ,$ , so this can be done using only ReLU layers. At the final stage, define

$$
\tilde { R } ( x , t , s ) : = \mathrm { R e L U } \big ( R ( x , t ) - \lambda ( 1 - s ) 1 _ { m } \big ) - \mathrm { R e L U } \big ( - R ( x , t ) - \lambda ( 1 - s ) 1 _ { m } \big ) .
$$

If $s = 1$ , then

$$
\widetilde { R } ( x , t , 1 ) = \mathrm { R e L U } { \big ( } R ( x , t ) { \big ) } - \mathrm { R e L U } { \big ( } - R ( x , t ) { \big ) } = R ( x , t ) .
$$

$s = 0 ,$ , then for every component one has $| R _ { j } ( x , t ) | \leq \lambda .$ , hence $\widetilde { R } ( x , t , 0 ) = 0$ as desired.

We now show that the networks in $\mathcal { F } _ { P } ^ { d }$ can approximate every piecewise-smooth vector field within an arbitrary accuracy.

Definition 4.2 (Piecewise-smooth vector field on a fixed arrangement). Let $\Omega \subset \mathbb { R } ^ { d }$ be compact and let $P =$ $\{ \mathcal { P } _ { 1 } , \ldots , \mathcal { P } _ { K } \}$ be as above. A function $F : \Omega \times \mathbb { R } \to \mathbb { R } ^ { d }$ is said to be piecewise-smooth with respect to $P$ if for every $\bar { k } \in \{ 1 , \ldots , \bar { K } \}$ there exists a map

$$
F _ { k } : \overline { { \Omega \cap \mathcal { P } _ { k } } } \times \mathbb { R }  \mathbb { R } ^ { d }
$$

which is continuous on $\overline { { \Omega \cap \mathcal { P } _ { k } } } \times \mathbb { R }$ and smooth on $\begin{array} { r } { ( \Omega \cap \mathcal { P } _ { k } ) \times \mathbb { R } , } \end{array}$ and such that

$$
F ( x , t ) = F _ { k } ( x , t ) , \qquad \forall x \in \Omega \cap \mathcal { P } _ { k } , t \in \mathbb { R } .
$$

Theorem 4.2 (Approximation of piecewise-smooth systems without creating new jumps). Let $\Omega \subset \mathbb { R } ^ { d }$ be compact, $T > 0 ,$ , and $P = \mathbf { \bar { \{ \mathcal { P } } _ { 1 } , \dots , \mathcal { P } _ { K } \} }$ a finite family of pairwise disjoint open convex polyhedra with

$$
\Omega \setminus \Gamma _ { P } = \bigcup _ { k = 1 } ^ { K } \left( \Omega \cap \mathcal { P } _ { k } \right) .
$$

Let $F : \Omega \times \mathbb { R }  \mathbb { R } ^ { d }$ be piecewise smooth with respect to $P .$ Then for every $\varepsilon > 0$ , there exists $\mathcal { N } \in \mathcal { F } _ { P } ^ { d }$ such that

$$
\operatorname* { s u p } _ { x \in \Omega \setminus \Gamma _ { P } } \| F ( x , t ) - \mathcal { N } ( x , t ) \| _ { \infty } < \varepsilon .
$$

Proof. For every $k = 1 , \ldots , K$ , we have that $F _ { k } : \overline { { \Omega \cap \mathcal { P } _ { k } } } \times [ 0 , T ]  \mathbb { R } ^ { d }$ is continuous. Since $\overline { { \Omega \cap \mathcal { P } _ { k } } } \times [ 0 , T ]$ is compact, the universal approximation theory of ReLU networks, see [35], ensures that there exists $\widetilde { \Psi } _ { k } \in \mathcal { F } _ { \mathrm { R e L U } } ^ { d + 1 , d }$ such that

$$
\operatorname* { s u p } _ { x \in \overline { { \Omega \cap \mathscr { P } _ { k } } } } \left\| F _ { k } ( x , t ) - \widetilde { \Psi } _ { k } ( x , t ) \right\| _ { \infty } < \varepsilon .
$$

Consider the projection $\Pi _ { k } : \mathbb { R } ^ { d _ { P } }  \mathbb { R } ^ { m _ { k } }$ onto the block corresponding to $\mathcal { P } _ { k }$ , so that

$$
\Pi _ { k } T _ { P } ( x ) = b _ { k } - A _ { k } x .
$$

We define the function $\begin{array} { r } { \chi _ { k } ( z ) = \mathcal { H } \bigl ( \boldsymbol { 1 } _ { m _ { k } } ^ { \top } \Pi _ { k } z - m _ { k } + \frac { 1 } { 2 } \bigr ) } \end{array}$ in $\mathcal { F } _ { \mathrm { R e L U } _ { J } } ^ { d _ { P } , 1 }$ , compare [24, Equation 1]. If $x \notin \Gamma _ { P }$ , then $x \in \mathcal { P } _ { k }$ if and only if all entries of $\Pi _ { k } z _ { P } ( x ) = \mathcal { H } ( b _ { k } - A _ { k } x )$ are equal to 1. Therefore, $1 _ { \mathcal { P } _ { k } } ( x ) = \chi _ { k } ( z _ { P } ( x ) )$ for $x \in \Omega \setminus \Gamma _ { P }$

By Theorem 4.1, there exists a ReLU network $\Psi _ { k } \in \mathcal { F } _ { \mathrm { R e L U } } ^ { d + 2 , d }$ such that

$$
\Psi _ { k } ( x , t , s ) = \left\{ \begin{array} { l l } { \widetilde { \Psi } _ { k } ( x , t ) , } & { s = 1 , } \\ { 0 , } & { s = 0 , } \end{array} \right. \qquad \forall ( x , t , s ) \in \Omega \times [ 0 , T ] \times \{ 0 , 1 \} .
$$

Therefore the map

$$
\mathcal { N } _ { k } ( x , t ) : = \Psi _ { k } \big ( x , t , \chi _ { k } ( z _ { P } ( x ) ) \big )
$$

belongs to $\mathcal { F } _ { P } ^ { d }$ and satisfies

$$
\mathcal { N } _ { k } ( x , t ) = 1 _ { \mathcal { P } _ { k } } ( x ) \widetilde { \Psi } _ { k } ( x , t ) , \qquad \forall ( x , t ) \in ( \Omega \setminus \Gamma _ { P } ) \times [ 0 , T ] .
$$

Since $\mathcal { F } _ { P } ^ { d }$ is a vector space, the sum

$$
\mathcal { N } ( x , t ) : = \sum _ { k = 1 } ^ { K } \mathcal { N } _ { k } ( x , t )
$$

belongs to $\mathcal { F } _ { P } ^ { d }$ . Moreover, for $( x , t ) \in ( \Omega \setminus \Gamma _ { P } ) \times [ 0 , T ]$

$$
\mathcal { N } ( x , t ) = \sum _ { k = 1 } ^ { K } 1 _ { \mathcal { P } _ { k } } ( x ) \widetilde { \Psi } _ { k } ( x , t ) .
$$

For each $x \in \Omega \setminus \Gamma _ { P }$ , there is a unique $\underline { { k } } \in \{ 1 , \ldots , K \}$ such that $x \in \mathcal { P } _ { k }$ . Therefore,

$$
\| \mathcal { N } ( x , t ) - F ( x , t ) \| _ { \infty } = \left\| \widetilde { \Psi } _ { \underline { { k } } } ( x , t ) - F ( x , t ) \right\| _ { \infty } < \varepsilon
$$

as desired.

Remark 4.3 (Exact representation of piecewise-affine vector fields on compacts). Given a partition $P = \{ \mathcal { P } _ { 1 } , \ldots , \mathcal { P } _ { K } \}$ and a compact set $\Omega \subset \mathbb { R } ^ { d }$ as in Theorem 4.2, every piecewise-affine function

$$
x \mapsto \sum _ { k = 1 } ^ { K } 1 _ { \mathcal { P } _ { k } } ( x ) ( W _ { k } x + w _ { k } ) , W _ { 1 } , . . . , W _ { K } \in \mathbb { R } ^ { d \times d } , w _ { 1 } , . . . , w _ { K } \in \mathbb { R } ^ { d }
$$

is the restriction over Ω of a function $\mathcal { N } \in \mathcal { F } _ { P } ^ { d }$ . Such a result follows by picking $\widetilde { \Psi } _ { k } ( x , t ) = W _ { k } x + w _ { k }$ in the proof above.

Remark 4.4 (Extended approximation spaces). Replacing the raw time input by a continuous feature map $\varphi ( t )$ , or augmenting Ψ with other continuous features of x and t, does not enlarge the possible discontinuity set because $z _ { P } ( x )$ remains the only discontinuous input. In the periodically forced experiments, we therefore use maps of the form

$$
( x , t ) \mapsto \Psi \big ( x , \varphi ( t ) , \Phi ( z _ { P } ( x ) ) \big ) ,
$$

where

$$
\Psi \in \mathcal { F } _ { \mathrm { R e L U } } ^ { d + 2 k + d _ { 1 } , d } , \qquad \Phi \in \mathcal { F } _ { \mathrm { R e L U } _ { J } } ^ { d _ { P } , d _ { 1 } } ,
$$

and

$$
\begin{array} { r } { \varphi ( t ) = [ \cos ( \omega _ { 1 } t + \theta _ { 1 } ) \quad \sin ( \omega _ { 1 } t + \theta _ { 1 } ) \quad \ldots \quad \cos ( \omega _ { k } t + \theta _ { k } ) \quad \sin ( \omega _ { k } t + \theta _ { k } ) ] ^ { \intercal } \in \mathbb { R } ^ { 2 k } . } \end{array}
$$

The frequencies $\omega _ { 1 } , \ldots , \omega _ { k } > 0$ may be fixed from prior knowledge or chosen as random Fourier features.

## 5 Numerical experiments

We now test the methodology introduced and analysed above. The experiments address three questions: (1) whether the proposed architecture confines discontinuities to a prescribed partition, (2) whether derivative-jump RANSAC can recover the switching geometry from the training trajectories, and (3) whether the recovered geometry supports accurate trajectory learning. We study the geometry and dynamics learning phases both in isolation, in Section 5.3 and Section 5.4, and when combined, in Section 5.5. We also test the effect of incorporating additional prior knowledge, such as the second-order structure of the target dynamics and the forcing frequency in periodically forced systems. Unless stated otherwise, all experiments based on training a network use the random seed 7.

In the experiments in which we recover the geometry from trajectories, we focus on systems with one dominant affine switching surface. The dry-friction and the climate-based PP04 benchmarks both have a physically meaningful single switch, and this setting isolates the interaction between derivative-jump recovery and dynamics learning without adding the separate task of assigning several recovered facets to regions. The approximation architecture is formulated for general finite polyhedral partitions, and the known-geometry regression experiment already tests a multi-region partition. The more general case is discussed in Appendix E, which gives a synthetic demonstration of sequential RANSAC–TLS recovery for multiple affine interfaces.

The GitHub repository associated with the paper is at https://github.com/davidemurari/ learn-piecewise-smooth.

Benchmark dynamical systems The three benchmark systems are chosen to test different parts of the methodology.

Benchmark A: We first isolate the dynamics-learning architecture on a two-dimensional piecewise-linear oscillator with prescribed switching geometry. The vector field is

$$
F ( x ) = \left[ { \frac { p } { - q - \alpha ( q ) p } } \right] , \qquad \alpha ( q ) = \left\{ { \begin{array} { l l } { 0 . 9 , } & { q < - 1 , } \\ { - 0 . 9 , } & { - 1 \leq q < 1 , } \\ { 0 . 9 , } & { q \geq 1 . } \end{array} } \right.\tag{15}
$$

This example has three smooth regions separated by the two switching lines $q = \pm 1$ , and is used to test whether the proposed architecture can approximate a discontinuous vector field without introducing spurious jumps away from the prescribed partition.

We then test the complete learning pipeline on two families of trajectory-learning problems: dry-friction oscillators and the PP04 system.

Benchmark B: The dry-friction oscillator. This system, also considered in [25] and analysed in [2, 14], models the stick-slip dynamics of a forced mass moving on a surface with friction, and is useful because it combines a simple

switching geometry with non-trivial Filippov behaviour. It allows us to separate three effects: the benefit of knowing the switching geometry, the benefit of imposing the mechanical second-order structure, and the degradation caused by noisy trajectory observations. The system has state $x = ( q , p )$ and stick-slip dynamics given by

$$
\dot { q } = p , \qquad \dot { p } = - q - c p + A \cos ( \omega t ) - \mu \operatorname { s i g n } ( p ) ,\tag{16}
$$

with switching hyperplane $\Sigma = \{ ( q , p ) : p = 0 \}$ . We consider two forced cases with $c = 0 . 1 , \mu = 0 . 5 .$ , and $A = 1$ , but different forcing frequencies as in [25]:

$$
\omega = 0 . 3 \quad ( \mathrm { c a s e ~ 1 } ) , \qquad \omega = 0 . 1 5 \quad ( \mathrm { c a s e ~ 2 } ) .
$$

Benchmark C: The PP04 system. This climate-based system models the periodic behaviour of the ice ages [27,28] and provides a complementary three-dimensional benchmark. In contrast to the dry-friction oscillator, its switching surface is an affine plane not aligned with a coordinate axis, and the system is periodically forced. The state is $x = \overset { \cdot } { ( } V , A , C )$ representing the global ice volume $V ,$ the Antarctic ice cover A and the atmospheric carbon dioxide level ${ \dot { C } } .$ . In this model, carbon dioxide is gradually absorbed by the oceans over a period of about 100k years, leading to global cooling and a growth of the ice sheets. At a critical concentration, the oceans can no longer absorb the carbon dioxide, and it is suddenly released into the atmosphere. This causes a sudden rise in $C$ and rapid global warming, leading to a drop in $V$ and A. The process then repeats, synchronised by the periodic variations in the solar insolation caused by the Milankovitch cycles. With the parameter values used in the experiments, the linear switching function (representing the oceanic stratification linked to the carbon dioxide absorption) is given by:

$$
s ( V , A , C ) = a V - b A + d , \quad \quad a = 0 . 3 , \quad b = 0 . 7 , \quad d = 0 . 2 7 .
$$

The switching surface (corresponding to the critical values of the parameters when the carbon dioxide is suddenly released from the oceans) is the plane in the 3-dimensional phase-space given by:

$$
\Sigma = \{ ( V , A , C ) : s ( V , A , C ) \equiv ( a , - b , 0 ) ^ { T } ( V , A , C ) + d = 0 \} .
$$

Figure 1 visualises this plane in the $( V , A , C )$ phase space and compares representative trajectories of the reference PP04 system with learned-model rollouts from the same initial conditions. Setting the (assumed periodic) solar insolation to $\dot { I _ { 6 5 } } ( t ) = \mu \sin ( \omega t )$ (which varies according to the Milankovitch cycles [27]), the vector field (where time t is expressed in kilo-years) is

$$
\begin{array} { r l } & { \dot { V } = \frac { - x C - y I _ { 6 5 } ( t ) + z - V } { \tau _ { V } } , } \\ & { \dot { A } = \frac { V - A } { \tau _ { A } } , } \\ & { \dot { C } = \frac { \alpha I _ { 6 5 } ( t ) - \beta V + \gamma H ( - s ( V , A , C ) ) + \delta - C } { \tau _ { C } } . } \end{array}\tag{17}
$$

Here the Heaviside function is defined by $H ( z ) = 1 { \mathrm { f o r } } z \geq 0$ and $H ( z ) = 0$ otherwise. The $\gamma$ term is active when $s ( V , A , C ) \leq 0$ , including on the switching surface. In the trajectory-learning experiments all PP04 coefficients are fixed at $x = 1 . 3 , y = 0 . 5 , z = 0 . 8 , \alpha = 0 . 1 5 , \beta = 0 . 5 , \gamma = 0 . { \dot { 7 } } , \delta = \stackrel { \sim } { 0 } 0 . 4 , \tau _ { V } = 1 5 , \tau _ { A } = 1 2 , \tau _ { C } = 5 , \mu = 0 . 4 6 7 , c = 0 . 5 , T _ { A C } = 0 . 4 6 9 .$ , and $\omega = 0 . 1 5 3 2$ . The geometry-recovery diagnostic below instead varies γ while keeping every other coefficient and the switching plane fixed. This example tests whether the derivative-jump geometry-learning stage can accurately recover a non-axis-aligned switching plane for subsequent trajectory learning.

## 5.1 Evaluation metrics

For the different experiments, we isolate different aspects of the learning process. This is done by changing the architecture and its training regimen, as well as considering different evaluation metrics, which we introduce below.

Discontinuity detection diagnostic We define a metric to detect discontinuities and test it on the system in Equation (15). We describe it for this system, but it naturally extends.

For a learned vector field ${ \widehat { F } } ,$ , output component $k ,$ grid point $x ,$ we compute

$$
D _ { h , k } ( x ) = \operatorname* { m a x } _ { v \in V } \left| \widehat { F } _ { k } ( x + h v ) - \widehat { F } _ { k } ( x - h v ) \right| , \ V = \left\{ ( 1 , 0 ) , \ ( 0 , 1 ) , \ \frac { ( 1 , 1 ) } { \sqrt { 2 } } , \ \frac { ( 1 , - 1 ) } { \sqrt { 2 } } \right\} .
$$

For a continuous piecewise-linear function, $D _ { h , k } ( x )$ scales approximately linearly with h, so $D _ { 2 h , k } ( x ) / D _ { h , k } ( x ) \approx 2$ We retain a persistent jump score $P _ { h , k } ( x ) = D _ { h , k } ^ { ' } ( x )$ only where

$$
\frac { D _ { 2 h , k } ( x ) } { D _ { h , k } ( x ) + 1 0 ^ { - 1 2 } } < 1 . 3 5
$$

and $D _ { h , k } ( x )$ exceeds a small amplitude floor. The floor is chosen componentwise as

$$
\operatorname* { m a x } \left\{ 1 0 ^ { - 6 } , 1 0 ^ { - 2 } \mathrm { p e r c e n t i l e } _ { 9 9 } ( D _ { h , k } ) \right\} .
$$

We then define the known-switch band

$$
B _ { \delta } = \{ \vert q + 1 \vert \leq \delta \} \cup \{ \vert q - 1 \vert \leq \delta \} , \qquad \delta = 2 h ,
$$

and compute the persistent-jump mass outside this band. For component k we define

$$
M _ { k } ^ { \mathrm { t o t } } = \sum _ { x _ { j } } { P } _ { h , k } ( x _ { j } ) , \qquad M _ { k } ^ { \mathrm { o f f } } = \sum _ { x _ { j } \notin B _ { \delta } } { P } _ { h , k } ( x _ { j } ) ,
$$

summing over a diagnostic grid. The off-switch fraction is $\rho _ { k } ^ { \mathrm { o f f } } = M _ { k } ^ { \mathrm { o f f } } / M _ { k } ^ { \mathrm { t o t } } , \mathrm { i f } \ M _ { k } ^ { \mathrm { t o t } } > 0 .$

Rollout and vector field metrics Across the examples, the tables report trajectory-level and vector field-level diagnostics. Given a fixed set of N initial conditions, the rollout error is

$$
\mathrm { R M S E } _ { \mathrm { r o l l o u t } } = \sqrt { \frac { 1 } { N ( K + 1 ) } \sum _ { n = 1 } ^ { N } \sum _ { k = 0 } ^ { K } \big \| \hat { x } _ { n } ( t _ { k } ) - x _ { n } ( t _ { k } ) \big \| _ { 2 } ^ { 2 } } ,
$$

and the final-time error is

$$
\mathrm { R M S E } _ { \mathrm { f i n a l } } = \sqrt { \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \left. \hat { x } _ { n } ( t _ { K } ) - x _ { n } ( t _ { K } ) \right. _ { 2 } ^ { 2 } } .
$$

These two quantities measure whether the learnt dynamics extrapolate beyond the short training windows.

We also report vector field errors on a regular grid in the state domain, averaged over forcing phases when the system is non-autonomous:

$$
\mathrm { R M S E } _ { f } = \sqrt { \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \left. \hat { f } ( x _ { m } , t _ { m } ) - f ( x _ { m } , t _ { m } ) \right. _ { 2 } ^ { 2 } } .
$$

The same quantity is also evaluated in a narrow band around the switching set. This banded error is a strict local diagnostic of how well the switching dynamics are captured.

The evaluation protocol uses 128 trajectories of 200 steps for each trajectory-learning case. Non-autonomous evaluation trajectories use independently sampled initial forcing phases. Vector-field diagnostics use an $8 0 \times 8 0$ state grid for dry friction and $\mathrm { ~ a ~ } 2 4 ^ { \hat { 3 } }$ grid for PP04. Each non-autonomous grid is averaged over 16 uniformly spaced forcing phases. The switching band used for the local vector-field metric has half-width 0.05. Rollout metrics for the discontinuous dry-friction and PP04 models use the single-hyperplane Filippov solver described in Appendix D; continuous Neural ODE baselines use RK4.

Quality of the recovered switching geometry When the switching geometry is learned, the tables report the angle and offset errors of the recovered hyperplane. For a learned hyperplane $a ^ { \top } x + b = 0$ and the true hyperplane $a _ { \star } ^ { \top } x + b _ { \star } = 0 .$ both coefficient pairs are post-processed before evaluation. The normals are rescaled to unit Euclidean norm and oriented so that their inner product is non-negative. The offsets are rescaled and, when necessary, sign-flipped with their corresponding normals. Denote the resulting representations by $( \widehat { a } , \widehat { b } )$ and $( \widehat { a } _ { \star } , \widehat { b } _ { \star } )$ . The reported angle error, in degrees, is

$$
\theta = \frac { 1 8 0 } { \pi } \operatorname { a r c c o s } \left( \widehat { a } ^ { \top } \widehat { a } _ { \star } \right) ,
$$

and the offset error is

$$
e _ { b } = \left| \widehat { b } - \widehat { b } _ { \star } \right| .
$$

Both errors are therefore independent of the scaling and sign used to represent either hyperplane. For known-geometry runs, these quantities are zero by construction.

## 5.2 Trajectory data generation

For each discontinuous-model trajectory experiment, we generate two independent sets of trajectories. The main set is used for dynamics learning and is split into 90% training and 10% validation trajectories. Each dry-friction run uses a main set of 1, 000 trajectories, while each PP04 run uses 5, 000. The warm-up set contains the same number of trajectories as the corresponding main set and is used only to recover the switching geometry when that geometry is not supplied. The continuous baselines use the same main trajectory sets and splits, but do not generate a warm-up set.

We sample uniformly the initial states from $[ - 3 , 3 ] \times [ - 2 , 2 ]$ for dry friction and from $[ 0 , 1 ] ^ { 3 }$ for PP04. The initial time $t _ { 0 }$ is sampled uniformly over one forcing period for each trajectory.

The sampled trajectory segment is

$$
\{ x ( t _ { 0 } ) , x ( t _ { 0 } + h ) , \ldots , x ( t _ { 0 } + k h ) \} .
$$

For forced dry friction, reference trajectories are generated by a dedicated event-aware integrator. It applies RK4 within each smooth branch, localises detected crossings of $p = 0$ by bisection, and explicitly handles attractive sticking intervals on the switching manifold. For PP04, the true flow is propagated exactly between events using the matrix exponential and a sinusoidal particular solution, see [27, Section 6.2]. Detected switching-plane crossings are localised by bisection before the remainder of the step is propagated on the new side.

Dry-friction main segments use $h = 0 . 0 1$ and $k = 1 0 ,$ and warm-up segments use $h _ { \mathrm { w a r m } } = 0 . 0 1$ and $k _ { \mathrm { w a r m } } = 5 0$ PP04 main segments use $h = 0 . 0 5$ and $k = 5 0$ , and warm-up segments use $h _ { \mathrm { w a r m } } = 0 . 0 2$ and $k _ { \mathrm { w a r m } } = 1 0 0$

Time supplied to the learned vector field The learned discontinuous networks do not receive raw time as an input. Their smooth branch is evaluated as

$$
\Psi \bigl ( x , \Phi ( z ( x ) ) , \tau ( t ) \bigr )
$$

when switching geometry is present, and as $\Psi ( x , \tau ( t ) )$ for a continuous model. With a known forcing frequency, $\tau ( t ) = ( \cos ( \omega t ) , \sin ( \omega t ) )$ . In the unknown-frequency experiment, $\tau ( t )$ contains 16 fixed random Fourier frequencies and phases, represented by their sine and cosine pairs. The universality result in Theorem 4.2 persists within the family of piecewise-smooth vector fields, depending on t only through $\tau ( t )$

## 5.3 Known-geometry regression and discontinuity support

We isolate the role of the architectural prior in the dynamics-learning problem by considering a two-dimensional piecewise affine vector field for which the switching geometry is known exactly, see Equation (15). Both models are given the true switching code, and we ask whether the learned vector field introduces additional output jumps away from the prescribed switching set.

The switching set is given by:

$$
\Sigma = \{ ( q , p ) \in \mathbb { R } ^ { 2 } : q = - 1 \} \cup \{ ( q , p ) \in \mathbb { R } ^ { 2 } : q = 1 \} ,
$$

which separates the state space into the three open polyhedra

$$
\begin{array} { r } { \mathcal { P } _ { 1 } = \{ q < - 1 \} , \qquad \mathcal { P } _ { 2 } = \{ - 1 < q < 1 \} , \qquad \mathcal { P } _ { 3 } = \{ q > 1 \} . } \end{array}
$$

Equivalently, the implementation uses the binary switching code

$$
z _ { P } ( x ) = \mathcal { H } \left( \left[ \begin{array} { l l } { 1 } & { 0 } \\ { 1 } & { 0 } \end{array} \right] x + \left[ \begin{array} { l } { 1 } \\ { - 1 } \end{array} \right] \right) ,
$$

so that the three regions correspond to the codes (0, 0), (1, 0), and $( 1 , 1 )$ , respectively.

Architectures The first model is the time-independent version of the geometry-preserving class in Definition 4.1,

$$
\begin{array} { r } {  { \mathcal { N } } ( x ) = \Psi \bigl ( x , \Phi ( z _ { P } ( x ) ) \bigr ) , \qquad \Phi \in \mathcal { F } _ { \mathrm { R e L U } _ { J } } ^ { 2 , d _ { 1 } } , \qquad \Psi \in \mathcal { F } _ { \mathrm { R e L U } } ^ { 2 + d _ { 1 } , 2 } . } \end{array}
$$

The two switching hyperplanes are fixed to their true values. We refer to this model as the geometry-preserving architecture.

We also compare against an unstructured jump network. This model keeps the same fixed switching code, but concatenates it directly with the state and passes the result through a jump-ReLU network,

$$
\begin{array} { r } { \widetilde { \mathcal { N } } ( \boldsymbol { x } ) = G \big ( \boldsymbol { x } , \boldsymbol { z } _ { P } ( \boldsymbol { x } ) \big ) , \qquad G \in \mathcal { F } _ { \mathrm { R e L U } _ { J } } ^ { 4 , 2 } . } \end{array}
$$

This baseline therefore has access to the same exact partition information, but it does not use the separated Φ–Ψ parametrisation of Definition 4.1. The widths are chosen so that the two models have comparable size.

![](images/f7c70431137a5bf73fa2bfa58c2ff937b0d4261b39a7ed011308123d9b1a35dc.jpg)

![](images/24c2b3a2ed84279e6cb7df89581502b957c58cd7624d6ef1eda472f89d19eacb.jpg)

![](images/2f5956d0483d32bced2ef48a5c2ecc63b130375c0b687147c611c926ee8b591d.jpg)

![](images/4af6b19ed74700502a000c8e31ed982959538e9cf4a17f10898a6935501df7ba.jpg)  
(a) Geometry-preserving architecture.

![](images/eb6742b95b0c3c38759628608f4104d7873c194860b2ecdfcdc7eca4faa99a94.jpg)

![](images/05808015b6cd64ef0ac1eecdbee24dc8bb7861a42074ff50846e06446e487148.jpg)  
(b) Unstructured jump network.

Figure 4: Qualitative reconstruction of the target vector field from noisy regression data. In each row, the first pane shows the target field, the second panel shows the learned field, and the third panel shows the pointwise vector error $\| { \widehat { f } } - f \| _ { 2 }$ on a logarithmic scale. Both architectures recover the main phase portrait and give small clean-grid errors, but bthe geometry-preserving architecture gives a smoother and more faithful reconstruction of the target partition structure. The displayed state domain is the full training rectangle $[ - 3 , 3 ] \times [ - 4 , 4 ]$ , which contains every point of the diagnostic trajectories.

Training protocol Training data are generated directly from the vector field rather than from finite-difference trajectory estimates

$$
x _ { n } \sim \mathcal { U } \left( [ - 3 , 3 ] \times [ - 4 , 4 ] \right) , ~ y _ { n } = F ( x _ { n } ) + \eta _ { n } , \qquad \eta _ { n } \sim N ( 0 , \sigma ^ { 2 } I _ { 2 } ) , \qquad \sigma = 0 . 0 3 ,
$$

for $n = 1 , \ldots , N$ . The model parameters are fitted by minimising the empirical mean-square regression loss

$$
\mathcal { L } ( \mathcal { N } ) : = \frac { 1 } { 2 N } \sum _ { n = 1 } ^ { N } \left. \mathcal { N } ( x _ { n } ) - y _ { n } \right. _ { 2 } ^ { 2 } .
$$

Each model is trained for 1, 000 epochs on 10, 000 noisy uniformly sampled points and selected using an independent noisy validation set of 2, 000 uniformly sampled points. A separate clean test set contains 6, 000 uniformly sampled points. The geometry-preserving model uses one width-32 hidden layer in Φ and three width-32 hidden layers in Ψ. The unstructured model uses three hidden layers of width 45. The widths are chosen to give comparable model sizes: the geometry-preserving and unstructured models contain 4,482 and 4,592 trainable parameters, respectively. Both are optimised with Adam, weight decay $1 0 ^ { - 6 }$ , gradient-norm clipping at 1, and a one-cycle learning-rate schedule from $\mathrm { { 1 0 ^ { - 4 } t o 5 \cdot 1 0 ^ { - 3 } } }$ and back down by a final division factor of $1 0 ^ { 4 } .$ . The checkpoint with the smallest validation mean-square error is restored for all reported diagnostics.

For the orbit diagnostic, both the target and learned fields are propagated with classical RK4 on the same uniform output grid. An outer RK4 step is recursively bisected into consecutive half-steps whenever the discrete region labels at its proposed endpoints differ.

Results As shown in Figure 4, both models recover the main phase portrait from noisy observations. The geometrypreserving model gives a more faithful field reconstruction and more closely reproduces the diagnostic trajectories throughout the training domain. The persistent-jump diagnostic in Figure 5 shows the central architectural difference. All detected persistent jumps produced by the geometry-preserving model remain confined to the prescribed switching surfaces, whereas the unstructured network also produces persistent jumps away from them. Thus, access to the correct partition alone is not sufficient to preserve its geometry. The constrained architecture enforces the intended support for discontinuities while retaining an accurate approximation of the dynamics.

![](images/6b07125f7e13978cc72dd5fd6509921debbf6a403c7b10af73a413f9ffbd8dbf.jpg)  
Figure 5: Persistent jump support after training on noisy regression targets with $\sigma = 0 . 0 3$ . Blue points lie within a 2h neighbourhood of the prescribed surfaces, and red points lie outside it. The architecture fixes the potential discontinuity surfaces but does not prescribe which output components may jump across them. No persistent jump mass is detected in the geometry-preserving model’s first component, while all detected second-component mass lies on the prescribed surfaces. The unstructured jump network produces persistent support away from those surfaces. Pale cyan lines in the first row mark potential architectural surfaces. The stronger solid guides in the second row mark the target component’s actual switching surfaces.

## 5.4 Trajectory-based recovery of the discontinuities

This subsection focuses on the geometry learning phase. We consider the hyperplane learning procedure described in Algorithms 1 and 2. Figure 6 confirms that the candidate cloud of high-jump points that we work with after derivative estimation is overcomplete. Large derivative jumps may come from true discontinuity crossings, but also from smooth high-curvature motion, endpoint effects in the derivative estimate, or observation noise. Consequently, fitting a hyperplane by total least squares (TLS) to all candidates is unreliable. Instead, we use RANSAC to find a hyperplane supported by a coherent subset of candidates, and then refit it to the inliers using TLS.

Because both estimators receive exactly the same complete three-dimensional candidate array, the comparison isolates the effect of robust consensus selection. RANSAC selects the best-supported raw affine proposal, using the mean inlier distance only to break ties in support, and then applies TLS to the winning consensus. This remains appropriate only when a coherent switching trace is identifiable in the candidate cloud. The comparison shows that robust consensus selection is decisive at mild noise but cannot recover an identifiable plane once the switching trace is not sufficiently distinguishable from the effects of noise.

We also empirically validate the detectability result in Section 4.1. We consider the PP04 geometry-recovery problem, varying the coefficient γ which controls the magnitude of the jump across the discontinuity hyperplane. The observationnoise level remains $\sigma = 0 . 0 5$ . Each success fraction is taken over 100 RANSAC random seeds for one paired trajectory/noise realisation. As shown in Figure 7, recovery becomes reliable once the discontinuity is sufficiently pronounced relative to the observation noise, although the transition is not strictly monotone. We remark that changing γ also changes the trajectories and the resulting candidate cloud. Therefore, this test supports the theoretical analysis, but considers a different dynamical system for each γ.

![](images/d2e531d0343b516e57f55c22a92bc6e8f4f922f80de8d9d62093accb3680882b.jpg)

![](images/348dd27e9debbd32d8534966509f11f284b02e179ec06c6467c15b813af85af3.jpg)

![](images/333a6b50a97e9e18fcaf52a2fad9c07abbfa3c18d3a0536a4232d86d45da5129.jpg)  
Figure 6: PP04 geometry recovery from derivative-jump candidates. Grey points are the overcomplete candidate cloud formed from the endpoints of the globally retained high-derivative-jump intervals after neighbour expansion, and black points are the final RANSAC/TLS inliers. Ordinary TLS and RANSAC receive the same candidate cloud. Green is the true switch, red dashed is ordinary ${ \mathrm { T L S } } ,$ and blue is ${ \mathrm { R A N S A C } }$ followed by a final TLS refinement on its consensus set. The panels correspond to observation-noise levels $\sigma = 0 , 1 0 ^ { - 2 }$ , and $5 \times 1 \mathrm { { 0 ^ { - 2 } } }$ . The clean case shows that ordinary TLS works when the candidate cloud is already concentrated near the switch, the low-noise case shows the benefit of robust consensus selection in a contaminated cloud, and the high-noise case shows the recovery failure regime. At $\sigma = 0 .$ , the TLS and RANSAC angle errors are approximately $0 . 6 0 ^ { \overline { { \circ } } }$ and 0.29◦. $\mathrm { { A t } } \ \sigma = 1 0 ^ { - 2 }$ , TLS fails with error 41 $\cdot 7 ^ { \circ }$ , whereas RANSAC recovers the plane with error $0 . 1 0 ^ { \circ } . \dot { \mathrm { A t } } \sigma = 5 \cdot 1 0 ^ { - 2 }$ , both methods fail, with errors $5 1 . 2 ^ { \circ }$ and 44.7◦.

![](images/492cf71c6f3eb14cd45978df63760016be35f28b2351f92ac9d1db53901d0430.jpg)  
Figure 7: Empirical PP04 switching-plane recovery fraction as the discontinuity coefficient $\gamma$ is varied at fixed observation-noise level $\sigma = 0 . 0 5$ . For each value of $\gamma ,$ the experiment generates 5, 000 trajectories and evaluates 100 RANSAC seeds with 20, 000 proposals per seed. Initial conditions, forcing phases, and standardised observation noise are paired across the values of $\gamma .$ . The two curves report the fraction of seeds satisfying the accurate criterion (angle below $2 ^ { \circ }$ and offset error below 0.02) and the relaxed criterion (angle below $5 ^ { \circ }$ and offset error below 0.05), respectively, against $( \gamma / \tau _ { C } ) / \sigma$

## 5.5 Trajectory-based learning of the geometry and the dynamics

The following experiments evaluate the problem of learning the dynamics from trajectory data after selecting a switching geometry. In some runs, the true switching set is supplied to the model. In the remaining runs, it is recovered from warm-up trajectories and then kept fixed during training. This separation is useful because geometry errors and vector field errors affect the learnt flow in different ways. When we use the trajectory rollout-loss to train our piecewise-smooth neural networks, we weigh the contributions to the MSE loss coming from the two sides of the switch so they are balanced. The balancing is based on the sign of the switching function at the trajectory initial condition.

## 5.5.1 Dry-friction oscillators

The dry-friction examples, see Equation (16), provide a controlled setting in which the true switching set is simple, but the long-time behaviour is sensitive to both the discontinuity and the sliding dynamics. The observed trajectories are perturbed with additive Gaussian noise with $\sigma \in \{ 0 , 1 0 ^ { - 2 } , 5 \cdot 1 0 ^ { - 2 } \}$ . When the second-order structure is enforced, we only parametrise with the network the second component of the vector field, and force the first to be equal to p. For each case and noise level, we train the four combinations assuming the knowledge, or lack thereof, of the switching hyperplane and second-order nature of the dynamics. The forcing frequency is assumed known in the main dry-friction tables. Appendix C.2 shows that replacing the known harmonic features with fixed random Fourier features substantially degrades both local accuracy and temporal extrapolation. All dry-friction models use one hidden layer of width 10 in Φ and three hidden layers of width 10 in Ψ. In the learned-switch runs, we retain the largest $2 \%$ of the derivative-jump scores globally, augment each retained interval with its neighbours within radius 1, and use both interval endpoints as candidates. We fit one switching line using 20, 000 RANSAC proposals with distance tolerance 0.03, followed by total least-squares refinement on the inliers. All models minimise (12). The velocities are estimated with a Savitzky–Golay window of 11 and polynomial degree 3. The loss weights are $\lambda _ { \mathrm { t r a j } } = 1 , \lambda _ { \mathrm { v f } } = 1 0 ^ { - 2 } , \lambda _ { \mathrm { n o r } } = 3 \cdot 1 0 ^ { - 3 }$ , and $\lambda _ { \mathrm { t a n } } = 1 0 ^ { - 2 }$ The velocity loss excludes $| s ( x ) | < 0 . 0 5$ , and the normal and tangential thresholds are $\tau _ { \mathrm { n o r } } = 5 \cdot 1 0 ^ { - 3 }$ and $\tau _ { \mathrm { t a n } } = 1 0 ^ { - 3 }$

Training uses classical RK4, not the Filippov solver. This choice is based on empirical evidence of comparable performance but at a much lower time cost. The rollout horizon follows a curriculum from 2 steps to the full 10-step window. Adam is run for 500 epochs with batch size 128, weight decay $1 0 ^ { - 5 }$ , gradient-norm clipping at 1, and a one-cycle schedule between $5 \cdot 1 \dot { 0 } ^ { - 4 }$ and $5 \cdot 1 0 ^ { - 3 } ~ $ . Full-horizon training and validation losses are measured every five epochs and after the final epoch, and the checkpoint with the smallest validation rollout loss is restored. The Filippov solver, see Appendix D, is used for rollout evaluation of the discontinuous models.

The recovered line is fixed during dynamics learning. In the reported tables, the column “2nd order” indicates whether the model is constrained to respect ${ \dot { q } } = p ,$ so that only the acceleration component is learnt.

Table 1: Dynamics-learning results for dry-friction case 1. Each sub-table corresponds to a different trajectory-noise level.  
(a) Clean data, σ = 0
<table><tr><td>Known switch</td><td>2nd order</td><td>Switch angle</td><td>Switch offset</td><td>Rollout RMSE</td><td>Final RMSE</td><td>VF RMSE</td><td>VF band RMSE</td></tr><tr><td>no</td><td>no</td><td>0.225</td><td> $3 . 7 8 \cdot 1 0 ^ { - 3 }$ </td><td>0.0348</td><td>0.0701</td><td>0.0146</td><td> $3 . 0 0 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td>no</td><td>yes</td><td>0.225</td><td> $3 . 7 8 \cdot 1 0 ^ { - 3 }$ </td><td> $3 . 9 0 \cdot 1 0 ^ { - 3 }$ </td><td> $5 . 1 7 \cdot 1 0 ^ { - 3 }$ </td><td> $6 . 2 7 \cdot 1 0 ^ { - 3 }$ </td><td> $4 . 6 5 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td>yes</td><td>no</td><td>0</td><td>0</td><td>0.0356</td><td> $0 . 0 7 1 9 $ </td><td> $0 . 0 1 4 4$ </td><td> $2 . 8 6 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td>yes</td><td>yes</td><td>0</td><td>0</td><td> $3 . 1 8 \cdot 1 0 ^ { - 3 }$ </td><td> $3 . 9 9 \cdot 1 0 ^ { - 3 }$ </td><td> $6 . 4 1 \cdot 1 0 ^ { - 3 }$ </td><td> $4 . 7 3 \cdot 1 0 ^ { - 3 }$ </td></tr></table>

(b) Mild noise, $\sigma = 1 0 ^ { - 2 }$
<table><tr><td>Known switch</td><td>2nd order</td><td>Switch angle</td><td>Switch offset</td><td>Rollout RMSE</td><td>Final RMSE</td><td>VF RMSE</td><td>VF band RMSE</td></tr><tr><td>no</td><td>no</td><td>0.284</td><td> $4 . 9 2 \cdot 1 0 ^ { - 3 }$ </td><td>0.0882</td><td>0.156</td><td>0.0618</td><td>0.0613</td></tr><tr><td>no</td><td>yes</td><td>0.284</td><td> $4 . 9 2 \cdot 1 0 ^ { - 3 }$ </td><td>0.0150</td><td>0.0235</td><td>0.0202</td><td>0.0231</td></tr><tr><td>yes</td><td>no</td><td>0</td><td>0</td><td>0.0887</td><td>0.157</td><td>0.0623</td><td>0.0620</td></tr><tr><td>yes</td><td>yes</td><td>0</td><td>0</td><td>0.0219</td><td>0.0351</td><td>0.0291</td><td>0.0315</td></tr></table>

(c) High noise, $\sigma = 5 \cdot 1 0 ^ { - 2 }$
<table><tr><td>Known switch</td><td>2nd order</td><td>Switch angle</td><td>Switch offset</td><td>Rollout RMSE</td><td>Final RMSE</td><td>VF RMSE</td><td>VF band RMSE</td></tr><tr><td>no</td><td>no</td><td>1.34</td><td>0.0475</td><td>0.202</td><td>0.331</td><td>0.227</td><td>0.553</td></tr><tr><td>no</td><td>yes</td><td>1.34</td><td>0.0475</td><td>0.0869</td><td>0.129</td><td>0.164</td><td>0.514</td></tr><tr><td>yes</td><td>no</td><td>0</td><td>0</td><td>0.275</td><td>0.446</td><td>0.259</td><td>0.276</td></tr><tr><td>yes</td><td>yes</td><td>0</td><td>0</td><td>0.125</td><td>0.184</td><td>0.193</td><td>0.220</td></tr></table>

Results Table 1 and Table 2 show that the second-order constraint improves rollout and final-state RMSE in every case, noise level, and geometry setting. $\mathrm { A t } ~ \sigma \le 1 0 ^ { - 2 }$ , the recovered switches remain accurate, with angle errors below $0 . 3 0 ^ { \circ }$ and offset errors below $5 \cdot 1 0 ^ { - 3 } . { \mathrm { ~ A t ~ } } \sigma = 5 \cdot 1 0 ^ { - 2 }$ , the angle errors increase to approximately $1 . 3 ^ { \circ }$ and the switching-band errors increase substantially. Known geometry does not uniformly improve rollout error. The learned-switch and known-switch models are optimised independently, and the recovered geometry can act as a slightly different inductive bias. The consistent result is the benefit of second-order structure and the degradation of local switching-band accuracy when high-noise geometry recovery deteriorates.

Table 2: Dynamics-learning results for dry-friction case 2. Each sub-table corresponds to a different trajectory-noise level.  
(a) Clean data, σ = 0
<table><tr><td>Known switch</td><td>2nd order</td><td>Switch angle</td><td>Switch offset</td><td>Rollout RMSE</td><td>Final RMSE</td><td>VF RMSE</td><td>VF band RMSE</td></tr><tr><td>no</td><td>no</td><td>0.194</td><td> $9 . 5 8 \cdot 1 0 ^ { - 4 }$ </td><td>0.0234</td><td>0.0537</td><td>0.0146</td><td> $3 . 1 8 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td>no</td><td>yes</td><td>0.194</td><td> $9 . 5 8 \cdot 1 0 ^ { - 4 }$ </td><td> $2 . 1 6 \cdot 1 0 ^ { - 3 }$ </td><td> $3 . 1 6 \cdot 1 0 ^ { - 3 }$ </td><td> $6 . 2 5 \cdot 1 0 ^ { - 3 }$ </td><td> $4 . 7 0 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td>yes</td><td>no</td><td>0</td><td>0</td><td> $0 . 0 2 3 6$ </td><td> $0 . 0 5 4 3 $ </td><td> $0 . 0 1 4 4$ </td><td> $2 . 9 3 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td>yes</td><td>yes</td><td>0</td><td>0</td><td> $2 . 1 7 \cdot 1 0 ^ { - 3 }$ </td><td> $3 . 4 0 \cdot 1 0 ^ { - 3 }$ </td><td> $5 . 8 9 \cdot 1 0 ^ { - 3 }$ </td><td> $4 . 6 1 \cdot 1 0 ^ { - 3 }$ </td></tr></table>

(b) Mild noise, $\sigma = 1 0 ^ { - 2 }$
<table><tr><td>Known switch</td><td>2nd order</td><td>Switch angle</td><td>Switch offset</td><td>Rollout RMSE</td><td>Final RMSE</td><td>VF RMSE</td><td>VF band RMSE</td></tr><tr><td>no</td><td>no</td><td>0.300</td><td> $2 . 5 4 \cdot 1 0 ^ { - 3 }$ </td><td>0.0748</td><td>0.138</td><td>0.0617</td><td>0.0613</td></tr><tr><td>no</td><td>yes</td><td>0.300</td><td> $2 . 5 4 \cdot 1 0 ^ { - 3 }$ </td><td>0.0214</td><td>0.0343</td><td>0.0286</td><td>0.0311</td></tr><tr><td>yes</td><td>no</td><td>0</td><td>0</td><td>0.0752</td><td>0.139</td><td>0.0622</td><td>0.0618</td></tr><tr><td>yes</td><td>yes</td><td>0</td><td>0</td><td>0.0224</td><td>0.0356</td><td>0.0294</td><td>0.0318</td></tr></table>

(c) High noise, $\sigma = 5 \cdot 1 0 ^ { - 2 }$
<table><tr><td>Known switch</td><td>2nd order</td><td>Switch angle</td><td>Switch offset</td><td>Rollout RMSE</td><td>Final RMSE</td><td>VF RMSE</td><td>VF band RMSE</td></tr><tr><td>no</td><td>no</td><td>1.29</td><td>0.0332</td><td>0.170</td><td>0.293</td><td>0.191</td><td>0.514</td></tr><tr><td>no</td><td>yes</td><td>1.29</td><td>0.0332</td><td>0.131</td><td>0.196</td><td>0.207</td><td>0.484</td></tr><tr><td>yes</td><td>no</td><td>0</td><td>0</td><td>0.200</td><td>0.335</td><td>0.195</td><td>0.180</td></tr><tr><td>yes</td><td>yes</td><td>0</td><td>0</td><td>0.126</td><td>0.186</td><td>0.195</td><td>0.222</td></tr></table>

## 5.5.2 PP04

We use PP04, see Equation (17), as a second trajectory-learning benchmark after the dry-friction examples. The switching geometry must either be supplied to the model or recovered from trajectory data, and the learned vector field is then assessed through both local vector field errors and finite-horizon rollouts.

The forcing frequency ω is assumed known in all PP04 runs. The observed trajectories have initial conditions sampled in $[ 0 , 1 ] ^ { 3 }$ and perturbed by additive Gaussian noise with standard deviation $\sigma \in \mathsf { \bar { \{ 0 , 1 0 ^ { - 2 } \} } }$ . We choose these noise levels because of the analysis in Section 4.1. All PP04 models use one hidden layer of width 11 in Φ and one hidden layer of width 14 in Ψ. In the learned-switch runs, we retain the largest 0.5% of derivative-jump scores globally, augment each retained interval with its neighbours within radius 2, and use both interval endpoints as candidates. We fit one switching plane using 20, 000 RANSAC proposals with distance tolerance $1 0 ^ { - 2 }$ , followed by total least-squares refinement on the inliers.

Dynamics training minimises (12) using classical RK4, with $\lambda _ { \mathrm { t r a j } } = 1 , \lambda _ { \mathrm { v f } } = 1 0 ^ { - 2 }$ , no switch-band exclusion, and $\bar { \lambda _ { \mathrm { n o r } } } = \lambda _ { \mathrm { t a n } } = 0$ . Dynamics-loss velocities use a Savitzky–Golay window of 21 and polynomial degree 3; warm-up geometry recovery instead uses window 51 and degree 2. The rollout curriculum grows from 2 to 50 steps. Adam is run for 500 epochs with batch size 128, weight decay $\overline { { 1 } } 0 ^ { - 5 }$ , gradient-norm clipping at 1, and a one-cycle schedule between $1 0 ^ { - 5 }$ and $1 0 ^ { - 4 }$ . Validation is evaluated every five epochs and after the final epoch, and the best full-horizon validation checkpoint is restored.

The recovered switching plane remains accurate at both noise levels. All four models reproduce the PP04 dynamics closely, although observation noise increases both rollout and local vector-field errors. Supplying the true switching plane reduces every reported error at each noise level, with the largest improvement occurring near the switching surface.

The initial state is shared across panels and selected from 30 deterministic candidates by minimising aggregate rollout error over $0 \leq t \leq 2 5 0$ . Every candidate rollout starts at $t _ { 0 } = 0$ . The figure displays the first 500 time units for readability. This selection is used only for visualisation and does not affect training, checkpoint selection, or the evaluation metrics.

Table 3: PP04 dynamics-learning results with known forcing frequency and Filippov rollout evaluation. Each sub-table fixes the noise level and compares the full learned-geometry pipeline against the known-switch setting.  
(a) σ = 0
<table><tr><td>Known switch</td><td>Switch angle</td><td>Switch offset</td><td>Rollout RMSE</td><td>Final RMSE</td><td>VF RMSE</td><td>VF band RMSE</td></tr><tr><td>no</td><td>0.339 0</td><td> $5 . 1 8 \cdot 1 0 ^ { - 3 }$  0</td><td> $\begin{array} { l } { 2 . 6 6 \cdot 1 0 ^ { - 3 } } \\ { 1 . 4 5 \cdot 1 0 ^ { - 3 } } \end{array}$ </td><td> $2 . 6 7 \cdot 1 0 ^ { - 3 }$   $1 . 7 \cdot 1 0 ^ { - 3 }$ </td><td> $6 . 1 8 \cdot 1 0 ^ { - 3 }$   $1 . 3 \cdot 1 0 ^ { - 3 }$ </td><td>0.0191  $2 . 1 9 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td>yes</td><td colspan="6"> $\overline { { ( \mathbf { b } ) \sigma = 1 0 ^ { - 2 } } }$ </td></tr><tr><td>Known switch</td><td>Switch</td><td>Switch</td><td>Rollout</td><td>Final</td><td>VF</td><td>VF band</td></tr><tr><td>no</td><td>angle 0.458</td><td>offset  $4 . 3 7 \cdot 1 0 ^ { - 3 }$ </td><td>RMSE 0.0111</td><td>RMSE 0.0176</td><td>RMSE  $7 . 9 \cdot 1 0 ^ { - 3 }$ </td><td>RMSE 0.0244</td></tr><tr><td>yes</td><td>0</td><td>0</td><td> $8 . 0 7 \cdot 1 0 ^ { - 3 }$ </td><td>0.0125</td><td> $1 . 6 8 \cdot 1 0 ^ { - 3 }$ </td><td> $2 \cdot 1 0 ^ { - 3 }$ </td></tr></table>

![](images/a0fd3933ec262a76bb24240ae81a0f137b115f3f3ef767fd80d00fef52cf8f3b.jpg)  
Figure 8: PP04 rollouts over $0 \leq t \leq 5 0 0$ from the shared initial state selected by minimum aggregate error across all four models, with $t _ { 0 } = 0 .$ . Each panel corresponds to one noise and switching-geometry setting, and each trace overlays the target trajectory with the learnt trajectory for $V , A , C ,$ , and the switching expression $a V - b A + d .$ The PP04 state itself has three components. The clean runs are nearly indistinguishable from the target, while the noisy runs retain the recurrent structure with small local deviations.

## 5.5.3 Comparison with a smooth Neural ODE

We compare the structured models with a smooth Neural ODE trained on the same trajectory pools, splits, segment lengths, and known-frequency time features. The baseline is a three-hidden-layer width-64 MLP with tanh activations. It is trained for 500 epochs using Adam at the fixed learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 6 }$ , batch size 128 for dry friction and for PP04, and full-horizon RK4 trajectory mean-square error. Validation is measured every five epochs and after the final epoch, and the best validation checkpoint is restored. The baseline has no switching code, geometryrecovery stage, vector-field loss, or directional regulariser. This choice follows from empirically seeing that these additional components did not improve the model performance. The trainable parameter counts are constant across noise levels and across the two dry-friction forcing cases. For dry friction, the structured Disc model has 521 trainable parameters and the smooth NODE baseline has 8705. For PP04, the corresponding counts are 448 and 8899. These totals count only parameters with gradients, not the fixed switching-hyperplane tensors.

Table 4: Comparison with a smooth Neural ODE baseline. “Disc” denotes the learned-geometry discontinuous model and “NODE” the continuous baseline. Case 1 and 2 are dry-friction oscillators. For dry friction, both the discontinuous model and the smooth Neural ODE impose the second-order relation ${ \dot { q } } = p$ and parametrise only p˙. For PP04, both models parametrise all three components of the vector field. “VF band” restricts vector-field error to the switching band. Bold indicates the smaller value within each matched pair. Discontinuous-model rollouts use the Filippov solver, while continuous baselines use RK4.
<table><tr><td>System</td><td>Model</td><td>Rollout RMSE</td><td>Final RMSE</td><td>VF RMSE</td><td>VF band RMSE</td></tr><tr><td>Case  $1 , \sigma = 0$ </td><td>Disc</td><td> $\mathbf { 3 . 9 0 \cdot 1 0 ^ { - 3 } }$ </td><td> $\mathbf { 5 . 1 7 \cdot 1 0 ^ { - 3 } }$ </td><td> $\mathbf { 6 . 2 7 \cdot 1 0 ^ { - 3 } }$ </td><td> $\mathbf { 4 . 6 5 \cdot 1 0 ^ { - 3 } }$ </td></tr><tr><td> $\mathbf { C a s e } 1 , \sigma = 0$ </td><td>NODE</td><td> $3 . 5 3 \cdot 1 0 ^ { - 2 }$ </td><td> $5 . 5 5 \cdot 1 0 ^ { - 2 }$ </td><td> $8 . 6 5 \cdot 1 0 ^ { - 2 }$ </td><td> $4 . 1 7 \cdot 1 0 ^ { - 1 }$ </td></tr><tr><td>Case  $1 , \sigma = 1 0 ^ { - 2 }$ </td><td>Disc</td><td> $\mathbf { 1 . 5 0 \cdot 1 0 ^ { - 2 } }$ </td><td> $\mathbf { 2 . 3 5 \cdot 1 0 ^ { - 2 } }$ </td><td> $\mathbf { 2 . 0 2 \cdot 1 0 ^ { - 2 } }$ </td><td> $\mathbf { 2 . 3 1 \cdot 1 0 ^ { - 2 } }$ </td></tr><tr><td>Case  $1 , \sigma = 1 0 ^ { - 2 }$ </td><td>NODE</td><td> $4 . 8 8 \cdot 1 0 ^ { - 2 }$ </td><td> $7 . 5 1 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 0 3 \cdot 1 0 ^ { - 1 }$ </td><td> $4 . 3 4 \cdot 1 0 ^ { - 1 }$ </td></tr><tr><td>Case  $2 , \sigma = 0$ </td><td>Disc</td><td> $\mathbf { 2 . 1 6 \cdot 1 0 ^ { - 3 } }$ </td><td> $\mathbf { 3 . 1 6 \cdot 1 0 ^ { - 3 } }$ </td><td> $\mathbf { 6 . 2 5 \cdot 1 0 ^ { - 3 } }$ </td><td> $\mathbf { 4 . 7 0 \cdot 1 0 ^ { - 3 } }$ </td></tr><tr><td>Case  $2 , \sigma = 0$ </td><td>NODE</td><td> $2 . 7 9 \cdot 1 0 ^ { - 2 }$ </td><td> $4 . 4 3 \cdot 1 0 ^ { - 2 }$ </td><td> $8 . 6 5 \cdot 1 0 ^ { - 2 }$ </td><td> $4 . 1 7 \cdot 1 0 ^ { - 1 }$ </td></tr><tr><td>Case  $2 , \sigma = 1 0 ^ { - 2 }$ </td><td>Disc</td><td> $\mathbf { 2 . 1 4 \cdot 1 0 ^ { - 2 } }$ </td><td> $\mathbf { 3 . 4 3 \cdot 1 0 ^ { - 2 } }$ </td><td> $\mathbf { 2 . 8 6 \cdot 1 0 ^ { - 2 } }$ </td><td> $\mathbf { 3 . 1 1 \cdot 1 0 ^ { - 2 } }$ </td></tr><tr><td>Case  $2 , \sigma = 1 0 ^ { - 2 }$ </td><td>NODE</td><td> $4 . 5 2 \cdot 1 0 ^ { - 2 }$ </td><td> $6 . 7 5 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 0 3 \cdot 1 0 ^ { - 1 }$ </td><td> $4 . 3 4 \cdot 1 0 ^ { - 1 }$ </td></tr><tr><td> $\mathrm { P P 0 } 4 , \sigma = 0$ </td><td>Disc</td><td> $\mathbf { 2 . 6 6 \cdot 1 0 ^ { - 3 } }$ </td><td> $\mathbf { 2 . 6 7 \cdot 1 0 ^ { - 3 } }$ </td><td> $\mathbf { 6 . 1 8 \cdot 1 0 ^ { - 3 } }$ </td><td> $\mathbf { 1 . 9 1 \cdot 1 0 ^ { - 2 } }$ </td></tr><tr><td> $\mathrm { P P 0 } 4 , \sigma = 0$ </td><td>NODE</td><td> $3 . 2 8 \cdot 1 0 ^ { - 2 }$ </td><td> $6 . 2 4 \cdot 1 0 ^ { - 2 }$ </td><td> $8 . 9 6 \cdot 1 0 ^ { - 3 }$ </td><td> $2 . 7 1 \cdot 1 0 ^ { - 2 }$ </td></tr><tr><td> $\mathrm { P P 0 4 } , \sigma = 1 0 ^ { - 2 }$ </td><td>Disc</td><td> $\mathbf { 1 . 1 1 \cdot 1 0 ^ { - 2 } }$ </td><td> $\mathbf { 1 . 7 6 \cdot 1 0 ^ { - 2 } }$ </td><td> $\mathbf { 7 . 9 \cdot 1 0 ^ { - 3 } }$ </td><td> $\mathbf { 2 . 4 4 \cdot 1 0 ^ { - 2 } }$ </td></tr><tr><td>PP04,  $\sigma = 1 0 ^ { - 2 }$ </td><td>NODE</td><td> $2 . 6 1 \cdot 1 0 ^ { - 2 }$ </td><td> $3 . 3 2 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 0 5 \cdot 1 0 ^ { - 2 }$ </td><td> $3 . 1 1 \cdot 1 0 ^ { - 2 }$ </td></tr></table>

The compact discontinuous model outperforms the substantially larger NODE baseline in every metric at both reported noise levels. Thus the comparison supports strong parameter efficiency and model flexibility for the considered examples.

## 6 Discussion and conclusion

We have proposed a two-stage framework for learning piecewise-smooth dynamical systems from trajectory data. The method first estimates affine switching geometry from derivative jumps, and then learns dynamics adapted to the regions induced by the recovered or prescribed switching set. This separation makes the model interpretable and allows prior knowledge to be incorporated either in the geometry-learning stage or in the local dynamics.

The theoretical analysis provides results for both stages. It derives a local expected detectability condition that shows how observation noise, derivative-estimation error, smoothing, and jump size interact under regularity and transversal crossing assumptions. It also shows that the proposed geometry-preserving neural class can uniformly approximate piecewise-smooth vector fields on compact sets away from the prescribed switching hyperplanes, while confining discontinuities to the prescribed partition.

The experiments support these findings. With known geometry, the proposed architecture avoids spurious off-switch jumps. With the recovered geometry, the learned models produce accurate rollouts when the derivative-jump signal i sufficiently strong relative to the noise. The dry-friction and PP04 tests also show the value of incorporating structural information, such as second-order dynamics or known forcing frequencies. More broadly, the results show that explicitly learning and preserving switching geometry offers a practical route to interpretable models of discontinuous dynamics.

The present study is restricted to fully observed low-dimensional states, affine switching geometry, and a sequential geometry–dynamics pipeline. The Filippov-aware solver is further restricted to a single switching hyperplane.

Future work will consider experimental noisy data, higher-dimensional systems, interacting switches, and non-affine switching surfaces. Another natural direction is to combine the method with encoder–decoder models, so that discontinuous latent dynamics can be learned from videos or other high-dimensional scientific observations.

## Acknowledgements

The authors would like to thank Dr Alice Cicirello for the insightful discussions at the start of this project. EJ was funded by the Knut and Alice Wallenberg Foundation grant 2024.0440. CBS acknowledges support from the Royal

Society Wolfson Fellowship, the EPSRC advanced career fellowship EP/V029428/1, the Wellcome Innovator Awards 215733/Z/19/Z and 221633/Z/20/Z. DM, CB, and CBS acknowledge the support of the EPSRC programme grant EP/V026259/1. DM and CBS acknowledge support from the EPSRC programme grant EP/Y028783/1. All the authors acknowledge support from the EU through the Marie Skłodowska-Curie Actions Staff Exchanges project REMODEL, grant agreement 101131557.

AI use declaration ChatGPT was used to assist in checking mathematical proofs and identifying notation issues and inconsistencies across the manuscript prior to submission. Codex was used to assist with the preparation of the research code, under direct and substantial supervision by the authors. The code is publicly available in the GitHub repository associated with this paper. The authors assume responsibility for all content.

## References

[1] R. C. ARKIN, Behavior-based robotics, Intelligent Robots and Autonomous Agents, MIT Press, Cambridge, MA, May 1998.

[2] B. BROGLIATO, Nonsmooth Mechanics: Models, Dynamics and Control, Communications and Control Engineering, Springer International Publishing, 3rd ed., 2016, https://doi.org/10.1007/978-3-319-28664-8.

[3] S. L. BRUNTON, J. L. PROCTOR, AND J. N. KUTZ, Discovering governing equations from data by sparse identification of nonlinear dynamical systems, Proceedings of the National Academy of Sciences, 113 (2016), pp. 3932–3937, https://doi.org/10.1073/pnas.1517384113.

[4] C. BUDD AND K. S. MORUPISI, Grazing bifurcations and transitions between periodic states ofthe PP04 model for the glacial cycle, IMA Journal of Applied Mathematics, 87 (2022), pp. 462–491, https://doi.org/10. 1093/imamat/hxac013.

[5] P. CAÑIZARES, D. MURARI, C.-B. SCHÖNLIEB, F. SHERRY, AND Z. SHUMAYLOV, Symplectic Neural Flows for Modeling and Discovery, 2024, https://arxiv.org/abs/2412.16787.

[6] E. CELLEDONI, A. LEONE, D. MURARI, AND B. OWREN, Learning Hamiltonians ofconstrained mechanical systems, Journal of Computational and Applied Mathematics, 417 (2023), p. 114608, https://doi.org/10. 1016/j.cam.2022.114608.

[7] K. CHAMPION, B. LUSCH, J. N. KUTZ, AND S. L. BRUNTON, Data-driven discovery of coordinates and governing equations, Proceedings of the National Academy of Sciences, 116 (2019), pp. 22445–22451, https: //doi.org/10.1073/pnas.1906995116.

[8] R. CHARTRAND, Numerical Differentiation of Noisy, Nonsmooth Data, ISRN Applied Mathematics, 2011 (2011), pp. 1–11, https://doi.org/10.5402/2011/164564.

[9] R. T. Q. CHEN, Y. RUBANOVA, J. BETTENCOURT, AND D. DUVENAUD, Neural Ordinary Differential Equations, in Advances in Neural Information Processing Systems, vol. 31, 2018, pp. 6572–6583.

[10] M. CRANMER, Interpretable machine learningfor science with PySR and SymbolicRegression.jl, 2023, https: //arxiv.org/abs/2305.01582.

[11] M. CRANMER, S. GREYDANUS, S. HOYER, P. BATTAGLIA, D. SPERGEL, AND S. HO, Lagrangian Neural Networks, 2020, https://arxiv.org/abs/2003.04630.

[12] M. CRANMER, A. SANCHEZ-GONZALEZ, P. BATTAGLIA, R. XU, K. CRANMER, D. SPERGEL, AND S. HO, Discovering Symbolic Models from Deep Learning with Inductive Biases, in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 17429–17442.

[13] F. DELLA SANTA AND S. PIERACCINI, Discontinuous neural networks and discontinuity learning, Journal of Computational and Applied Mathematics, 419 (2023), p. 114678, https://doi.org/10.1016/j.cam.2022. 114678.

[14] M. DI BERNARDO, C. J. BUDD, A. R. CHAMPNEYS, AND P. KOWALCZYK, Piecewise-smooth Dynamical Systems: Theory and Applications, Applied mathematical sciences; 163, Springer-Verlag, London, 2008, https: //doi.org/10.1007/978-1-84628-708-4.

[15] C. EDWARDS AND S. SPURGEON, Sliding Mode Control: Theory And Applications, CRC Press, Aug. 1998, https://doi.org/10.1201/9781498701822.

[16] A. F. FILIPPOV, Differential Equations with Discontinuous Righthand Sides, vol. 18 of Mathematics and Its Applications (Soviet Series), Kluwer Academic Publishers Group, Dordrecht, 1988. Translated from the Russian.

[17] M. A. FISCHLER AND R. C. BOLLES, Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography, Communications of the ACM, 24 (1981), pp. 381– 395, https://doi.org/10.1145/358669.358692.

[18] S. GREYDANUS, M. DZAMBA, AND J. YOSINSKI, Hamiltonian neural networks, in Advances in Neural Information Processing Systems, vol. 32, 2019, pp. 15353–15363.

[19] S. HOCHREITER AND J. SCHMIDHUBER, Long short-term memory, Neural Computation, 9 (1997), pp. 1735– 1780, https://doi.org/10.1162/neco.1997.9.8.1735.

[20] M. R. JEFFREY, Modeling with Nonsmooth Dynamics, vol. 7 of Frontiers in applied dynamical systems: reviews and tutorials, Springer, United States, Feb. 2020, https://doi.org/10.1007/978-3-030-35987-4.

[21] P. JIN, Z. ZHANG, A. ZHU, Y. TANG, AND G. E. KARNIADAKIS, SympNets: Intrinsic structure-preserving symplectic networks for identifying hamiltonian systems, Neural Networks, 132 (2020), pp. 166–179, https: //doi.org/10.1016/j.neunet.2020.08.017.

[22] D. KARNOPP, Computer simulation of stick-slip friction in mechanical dynamic systems, Journal of Dynamic Systems, Measurement, and Control, 107 (1985), pp. 100–103, https://doi.org/10.1115/1.3140698.

[23] P. KIDGER, J. MORRILL, J. FOSTER, AND T. LYONS, Neural controlled differential equations for irregular time series, in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 6696–6707.

[24] I. KONG, J. CHEN, S. LANGER, AND J. SCHMIDT-HIEBER, On the Expressivity ofDeep Heaviside Networks, Constructive Approximation, (2026), https://doi.org/10.1007/s00365-026-09766-3.

[25] C. LATHOURAKIS AND A. CICIRELLO, Physics enhanced sparse identification of dynamical systems with discontinuous nonlinearities, Nonlinear Dynamics, 112 (2024), pp. 11237–11264, https://doi.org/10.1007/ s11071-024-09652-2.

[26] B. LUSCH, J. N. KUTZ, AND S. L. BRUNTON, Deep learning for universal linear embeddings of nonlinear dynamics, Nature Communications, 9 (2018), p. 4950, https://doi.org/10.1038/s41467-018-07210-0.

[27] K. S. MORUPISI AND C. BUDD, An analysis ofthe periodicallyforced PP04 climate model, using the theory of non-smooth dynamical systems, IMA Journal of Applied Mathematics, 86 (2021), pp. 76–120, https://doi. org/10.1093/imamat/hxaa039.

[28] D. PAILLARD AND F. PARRENIN, The antarctic ice sheet and the triggering of deglaciations, Earth and Planetary Science Letters, 227 (2004), pp. 263–271, https://doi.org/10.1016/j.epsl.2004.08.023.

[29] P. PETERSEN AND J. ZECH, Mathematical theory ofdeep learning, arXiv preprint arXiv:2407.18384, (2024).

[30] C. RACKAUCKAS, Y. MA, J. MARTENSEN, C. WARNER, K. ZUBOV, R. SUPEKAR, D. SKINNER, A. RA-MADHAN, AND A. EDELMAN, Universal differential equationsfor scientific machine learning, 2020, https: //arxiv.org/abs/2001.04385.

[31] Y. RUBANOVA, R. T. Q. CHEN, AND D. DUVENAUD, Latent ordinary differential equations for irregularlysampled time series, in Advances in Neural Information Processing Systems, vol. 32, 2019.

[32] A. SAVITZKY AND M. J. E. GOLAY, Smoothing and differentiation of data by simplified least squares procedures, Analytical Chemistry, 36 (1964), pp. 1627–1639, https://doi.org/10.1021/ac60214a047.

[33] F. VAN BREUGEL, J. N. KUTZ, AND B. W. BRUNTON, Numerical differentiation of noisy data: A unifying multi-objective optimization framework, IEEE Access, 8 (2020), pp. 196865–196877, https://doi.org/10. 1109/access.2020.3034077.

[34] P. R. VLACHAS, W. BYEON, Z. Y. WANLESS, T. P. SAPSIS, AND P. KOUMOUTSAKOS, Data-drivenforecasting ofhigh-dimensional chaotic systems with long short-term memory networks, Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 474 (2018), p. 20170844, https://doi.org/10.1098/ rspa.2017.0844.

[35] D. YAROTSKY, Error bounds for approximations with deep ReLU networks, Neural Networks, 94 (2017), pp. 103–114.

## A Fitting linear models in the presence of outliers

Out of a sample consisting of M total points, with M inliers and $M _ { o }$ outliers, RANSAC is a randomised algorithm that iteratively separates inliers from outliers. It works by randomly selecting d points to determine a (d 1)-dimensional affine subspace of $\mathbb { R } ^ { d }$ . Then, a hyperplane is fitted to these points, and the points sufficiently close (say, within a distance ε) are included in the consensus set. A model is deemed as reasonable if the consensus set is sufficiently large. The algorithm is repeated for a fixed number of iterations, each time keeping the model with the largest consensus set.

The number of inliers selected follows a hypergeometric distribution (as there is no replacement), so the probability r of selecting exactly d inliers is given by

$$
r = \frac { { \binom { M _ { i } } { d } } } { { \binom { M } { d } } } .
$$

The probability of selecting at least one point which is not an inlier is therefore $1 - r ,$ , and the probability of this happening k times is thus $( 1 ^ { - } r ) ^ { k }$ , which is the probability that the RANSAC algorithm has not found a good model in k iterations. The probability that at least one iteration has included only inliers is therefore $p = 1 - ( 1 - r ) ^ { k }$ . Thus, to ensure that RANSAC has reached the desired accuracy in k iterations with probability $p ,$ we must take

$$
k \geq { \frac { \log ( 1 - p ) } { \log ( 1 - r ) } } .
$$

We see that we must estimate r, meaning we need to determine, or at least bound, the number of points included as candidates for the RANSAC algorithm, that is, M and $M _ { i }$

Algorithm 2 describes the hyperplane-fitting step in the geometry learning phase.

Algorithm 2 RANSAC( , ε, M, m<sub>req</sub>)   
Require: Candidate cloud ${ \mathcal { C } } \subset \mathbb { R } ^ { d } $ ; tolerance $\varepsilon ;$ iterations $M ;$ required support $m _ { \mathrm { r e q } } .$   
Ensure: Best-supported hyperplane proposal and its consensus set, or failure.   
1: Initialise the current winner as empty.   
2: for $m = 1 , \ldots , M$ do   
3: Sample d points uniformly without replacement from $\mathcal { C } .$   
4: Construct the hyperplane $\mathbf { \check { T } } _ { m } = \bigl \{ x \in \mathbf { \check { \mathbb { R } } } ^ { d } : a _ { m } ^ { \top } x + b _ { m } = 0 \bigr \}$ through the sampled points.   
5: if the sampled points are degenerate then   
6: continue.   
7: end if   
8: Compute the consensus set ${ \mathcal { T } } _ { m } = \left\{ x \in { \mathcal { C } } : \operatorname { d i s t } ( x , \Gamma _ { m } ) \leq \varepsilon \right\}$   
9: if $| \bar { \mathcal { T } _ { m } } | < m _ { \mathrm { r e q } }$ then   
10: continue.   
11: end if   
12: Compute the mean inlier distance   
$D _ { m } = \frac { 1 } { | \mathcal { T } _ { m } | } \sum _ { x \in \mathcal { T } _ { m } } \mathrm { d i s t } ( x , \Gamma _ { m } ) .$   
13: Replace the winner if $\left( | \mathcal { I } _ { m } | , - D _ { m } \right)$ is lexicographically larger than its current score.   
14: end for   
15: if no winner was found then   
16: fail.   
17: end if   
18: return $( a _ { \star } , b _ { \star } , \mathcal { T } _ { \star } )$

## B Additional proofs for the universality result

ProofofLemma 4.1. The arguments in this proof apply indistinguishably to the elements in $\mathcal { F } _ { \mathrm { R e L U } } ^ { d _ { 1 } , d _ { 2 } }$ and $\mathcal { F } _ { \mathrm { R e L U } _ { J } } ^ { d _ { 1 } , d _ { 2 } }$ . Scalar multiplication and post-composition by affine maps are immediate. For addition and parallel concatenation, one first pads the shallower network with identity layers of the form (14) so that the two networks have the same depth. One may then use block-diagonal affine maps and componentwise activations to run the two networks in parallel. A final affine map gives either the concatenation or the sum. □

Proof of Proposition 4.3. Let $f , g \in \mathcal { F } _ { P } ^ { d }$ . By definition there are dimensions $r _ { f } , r _ { g } \in \mathbb { N } ,$ jump networks $\Phi _ { f } \in \mathcal { F } _ { \mathrm { R e L U } _ { * } } ^ { d _ { P } , r _ { f } }$ and $\Phi _ { g } \in \mathcal { F } _ { \mathrm { R e L U } _ { J } } ^ { d _ { P } , r _ { g } }$ , and continuous networks $\Psi _ { f } \in \mathcal { F } _ { \mathrm { R e L U } } ^ { d + r _ { f } + 1 , d }$ and $\Psi _ { g } \in \mathcal { F } _ { \mathrm { R e L U } } ^ { d + r _ { g } + 1 , d }$ such that

$$
f ( x , t ) = \Psi _ { f } \big ( x , t , \Phi _ { f } ( z _ { P } ( x ) ) \big ) , \qquad g ( x , t ) = \Psi _ { g } \big ( x , t , \Phi _ { g } ( z _ { P } ( x ) ) \big ) .
$$

By Lemma 4.1, the parallel concatenation $\Phi ( z ) : = \left[ \Phi _ { f } ( z ) , \Phi _ { g } ( z ) \right] \in \mathcal { F } _ { \mathrm { R e L U } , I } ^ { d _ { P } , r _ { f } + r _ { g } }$ and the parallel concatenation of $\Psi _ { f }$ and $\Psi _ { g }$ both belong to the corresponding ReLU classes. A final affine map produces any linear combination of $f$ and $g .$ Hence $\mathcal { F } _ { P } ^ { d }$ is a vector space. □

## C Additional experiments for the dry friction system

## C.1 Representative dry-friction rollouts

The aggregate errors in Tables 1 and 2 summarise performance over the evaluation set. Here we show representative long-horizon trajectories to illustrate the qualitative behaviour behind those numbers. We fix the initial condition $( q _ { 0 } , p _ { 0 } ) = ( 0 . 1 , 0 . 1 )$ and $t _ { 0 } = 0$ for all runs. All panels use the second-order model, known forcing frequency, noise level $\dot { \sigma } = \dot { 1 } 0 ^ { - 2 }$ , and the Filippov-aware rollout solver. This noise level is useful for visualisation because it is large enough to make geometry recovery nontrivial, but not so large that the recovered-switch model is dominated by geometry failure.

![](images/d7d28466302e447453984f03eb7a84cb63792f29d5ead430c3f473fde7d311e0.jpg)  
(a) Case 1, recovered switch

![](images/ba198ef2c350d8748adc618dd172d91095490728b1b4bb7a28a7925c1a478f65.jpg)  
(b) Case 1, known switch

![](images/2acefe07d2b369fcf2e9369f7e592683d4e3a199728ec2b01fad1dc6235c6f1c.jpg)  
(c) Case 2, recovered switch

![](images/82f8ea01e8674a8ea6befeeda7c786326cb0cee29c3656688302d8ae83581215.jpg)  
(d) Case 2, known switch  
Figure 9: Representative long-horizon dry-friction rollouts at $\sigma = 1 0 ^ { - 2 }$ . Each panel compares the target trajectory with the learned trajectory for q and $p .$ The left column shows the full pipeline, where the switching line is recovered from warm-up trajectories. The right column isolates the dynamics learning by supplying the true switching line.

The known-switch panels show that, once the switching line is supplied, the second-order model accurately reproduces both the smooth portions and the sliding intervals where $p = 0$ . The recovered-switch panels remain close to the target trajectories, but small discrepancies appear near transitions into and out of sliding. This is consistent with the main tables: the rollout errors remain small in the mild-noise regime, while the banded vector-field errors are more sensitive to small errors in the recovered switching line. Thus, the qualitative degradation in the full pipeline is mainly a local switching-geometry effect rather than a global instability of the learned dynamics.

Table 5: Effect of removing the known forcing frequency in the dry-friction experiments. All rows use the true switching manifold and the second-order model. Known/Unknown refer to known and unknown forcing frequency ω. Bold entries are the lowest between known and unknown frequency.
<table><tr><td rowspan="2">Setting</td><td colspan="2">Rollout RMSE</td><td colspan="2">Final RMSE</td><td colspan="2">VF RMSE</td></tr><tr><td>Known</td><td>Unknown</td><td>Known</td><td>Unknown</td><td>Known</td><td>Unknown</td></tr><tr><td>case  $1 , \sigma = 0$ </td><td> $\mathbf { 3 . 1 8 \cdot 1 0 ^ { - 3 } }$ </td><td> $2 . 1 2 \cdot 1 0 ^ { - 2 }$ </td><td> $\mathbf { 3 . 9 9 \cdot 1 0 ^ { - 3 } }$ </td><td> $3 . 8 6 \cdot 1 0 ^ { - 2 }$ </td><td> $\mathbf { 6 . 4 1 \cdot 1 0 ^ { - 3 } }$ </td><td> $3 . 4 7 \cdot 1 0 ^ { - 2 }$ </td></tr><tr><td>case 1  $, \sigma = 0 . 0 1$ </td><td> $\mathbf { 2 . 1 9 \cdot 1 0 ^ { - 2 } }$ </td><td> $3 . 4 7 \cdot 1 0 ^ { - 2 }$ </td><td> $\mathbf { 3 . 5 1 \cdot 1 0 ^ { - 2 } }$ </td><td> $5 . 2 2 \cdot 1 0 ^ { - 2 }$ </td><td> $\mathbf { 2 . 9 1 \cdot 1 0 ^ { - 2 } }$ </td><td> $5 . 7 4 \cdot 1 0 ^ { - 2 }$ </td></tr><tr><td>case  $1 , \sigma = 0 . 0 5$ </td><td> $\mathbf { 1 . 2 5 \cdot 1 0 ^ { - 1 } }$ </td><td> $1 . 7 8 \cdot 1 0 ^ { - 1 }$ </td><td> $\mathbf { 1 . 8 4 \cdot 1 0 ^ { - 1 } }$ </td><td> $2 . 5 5 \cdot 1 0 ^ { - 1 }$ </td><td> $\mathbf { 1 . 9 3 \cdot 1 0 ^ { - 1 } }$ </td><td> $4 . 1 5 \cdot 1 0 ^ { - 1 }$ </td></tr><tr><td>case 2  $, \sigma = 0$ </td><td> $\mathbf { 2 . 1 7 \cdot 1 0 ^ { - 3 } }$ </td><td> $1 . 6 9 \cdot 1 0 ^ { - 2 }$ </td><td> $\mathbf { 3 . 4 0 \cdot 1 0 ^ { - 3 } }$ </td><td> $2 . 8 3 \cdot 1 0 ^ { - 2 }$ </td><td> $\mathbf { 5 . 8 9 \cdot 1 0 ^ { - 3 } }$ </td><td> $4 . 0 5 \cdot 1 0 ^ { - 2 }$ </td></tr><tr><td>case  $2 , \sigma = 0 . 0 1$ </td><td> $\mathbf { 2 . 2 4 \cdot 1 0 ^ { - 2 } }$ </td><td> $5 . 0 9 \cdot 1 0 ^ { - 2 }$ </td><td> $\mathbf { 3 . 5 6 \cdot 1 0 ^ { - 2 } }$ </td><td> $8 . 0 4 \cdot 1 0 ^ { - 2 }$ </td><td> $\mathbf { 2 . 9 4 \cdot 1 0 ^ { - 2 } }$ </td><td> $7 . 3 9 \cdot 1 0 ^ { - 2 }$ </td></tr><tr><td>case  $2 , \sigma = 0 . 0 5$ </td><td> $\mathbf { 1 . 2 6 \cdot 1 0 ^ { - 1 } }$ </td><td>2.07· 10−¹</td><td> $\mathbf { 1 . 8 6 \cdot 1 0 ^ { - 1 } }$ </td><td> $3 . 1 8 \cdot 1 0 ^ { - 1 }$ </td><td> $\mathbf { 1 . 9 5 \cdot 1 0 ^ { - 1 } }$ </td><td> $4 . 4 7 \cdot 1 0 ^ { - 1 }$ </td></tr></table>

## C.2 Unknown forcing frequency

The dry-friction tables in the main body of the paper assume that the forcing frequency is known. To test the importance of this assumption, we repeat the strongest dry-friction configuration, namely the true switching manifold with the second-order model, but remove the known harmonic time features. The model receives random Fourier time features. The training setup, noise levels, architecture, and Filippov rollout evaluation are otherwise the same as in the main dry-friction experiments.

Table 5 shows that removing the known forcing frequency consistently degrades temporal extrapolation and vector-field recovery. A representative low-noise comparison is shown in Figure 10. We fix the initial condition $( q _ { 0 } , p _ { 0 } ) = ( 0 . 1 , 0 . 1 )$ and $t _ { 0 } = 0$ for both runs.

![](images/8af898a616f7834d1e08acfea45f50d31443e572614eb9d34eced8883bb8e5ab.jpg)  
(a) Known forcing frequency

![](images/81c5503248e07c62a50341b11b46df23bc1d6eafda3d5ef1c3a9f8ea7849f78c.jpg)  
(b) Unknown forcing frequency  
Figure 10: Representative long-horizon trajectories for dry-friction case $2$ with $\sigma = 1 0 ^ { - 2 }$ , true switching geometry, and second-order structure. The left panel uses the known harmonic forcing frequency, while the right panel replaces it with random Fourier time features.

## D Filippov-aware trajectory solver

This appendix describes the event-aware solver used for the rollout evaluations of the discontinuous models. The solver is used for single-hyperplane discontinuous models with switching function $s ( x ) = a ^ { \top } x + b ,$ and with learned one-sided vector fields $\check { f } ^ { + } ( x , \overset { \cdot } { t } )$ and $f ^ { - } ( x , t )$ . The purpose of the solver is to avoid treating a crossing of $\Sigma : = \{ x : s ( x ) = 0 \}$ as an ordinary smooth step, and to evolve trajectories along Σ when the two one-sided fields generate sliding

For a step from $x _ { n } \ { \mathrm { t o } } \ x _ { n + 1 }$ with step size h, we first compute the side $r _ { n } = \mathrm { s i g n } ( s ( x _ { n } ) )$ . A trial step is then taken with a branch-frozen second-order Runge–Kutta update, using $f ^ { + }$ throughout the step if $r _ { n } = + 1$ and $\bar { f } ^ { - }$ throughout the step if $r _ { n } = - 1$ . Let $\tilde { x } _ { n + 1 }$ denote this trial endpoint. A switching event is detected when $s ( x _ { n } ) s ( \tilde { x } _ { n + 1 } ) < 0 .$ . We also trigger event handling when the state is already close to the manifold, $| s ( x _ { n } ) | < \varepsilon$ , and the one-sided normal velocities indicate an attractive sliding region. In the experiments we use a small tolerance ε only to decide when this special handling is needed.

When a crossing is detected, the event time is located by bisection on the interval [0, h]. During this search the same branch-frozen RK2 propagator is used, starting from $x _ { n }$ on the pre-crossing side. After bisection, the event point is

projected back onto the hyperplane,

$$
x _ { \Sigma } = x _ { \mathrm { e v e n t } } - s ( x _ { \mathrm { e v e n t } } ) a / \lVert a \rVert _ { 2 } ^ { 2 } ,
$$

which removes the small numerical residual in the switching function.

The vector field on the switching manifold is defined using the standard Filippov convex combination. The trajectory is then advanced along Σ using small explicit substeps and projected back onto the hyperplane after each substep. If the attractive condition fails, the trajectory exits the manifold on the side indicated by the resulting one-sided normal velocities.

The event-handling procedure is repeated within one macro-step if necessary, up to a fixed small maximum number of events. If no crossing or sliding event is detected, the solver reduces to the branch-frozen RK2 update on the current side. This propagation rule is used for the long-horizon rollout diagnostics reported in the experiments. The current implementation cannot handle multi-surface interactions. The event detector is also not a fully general root-finding procedure. It may miss a crossing if a trajectory touches or crosses Σ an even number of times within a single branch-frozen trial step, so that the endpoint sign does not change. In the systems considered here, the chosen time steps were small enough that this event-detection rule was sufficient in practice. For more details on the implementation, see the GitHub repository https://github.com/davidemurari/learn-piecewise-smooth.

## E Sequential recovery of multiple switching lines

The geometry-recovery routine used in the main experiments is designed to find a single affine switching set from a cloud of candidate points. This appendix records a simple extension of the same idea to a setting with more than one interface. For this example, the data are noisy samples from three non-parallel lines in $\mathbb { R } ^ { 2 }$ , plus a small number of outliers. The purpose is to test whether the RANSAC–TLS step can be applied sequentially.

![](images/cb3df03a67c03289e57f35e9d588b038bb2427bc03e286324b8243681f752270.jpg)  
Figure 11: Toy candidate cloud used in the multi-line recovery test. The coloured points are noisy samples from the three true non-parallel lines, and the grey points are uniform outliers.

Figure 11 shows the toy data. Each coloured cloud is sampled near one true line, with independent Gaussian perturbations. The grey points are unrelated outliers. For a line written as $a ^ { \top } x + b = 0 , \| a \| _ { 2 } = 1$ , we run the same RANSAC routine used in the geometry-learning code. After one line has been recovered, its inliers are removed from the cloud, and the procedure is repeated on the remaining points. We show in Figure 12 the number of inliers of each attempt. This figure demonstrates that when the number of hyperplanes is unknown, the number of inliers can be a viable heuristic for determining it.

Figure 13 illustrates the three passes. The first pass matches $L _ { 2 }$ with angle error $0 . 0 8 5 4 ^ { \circ }$ and offset error $2 . 5 5 \cdot 1 0 ^ { - 4 }$ The second pass matches $L _ { 1 }$ with angle error $0 . { \bar { 1 } } 3 5 ^ { \circ }$ and offset error $1 . 3 \mathrm { { 5 } \cdot 1 0 ^ { - 3 } }$ . The third pass matches $L _ { 3 }$ with angle error $0 . 1 1 1 ^ { \circ }$ and offset error $4 . 8 2 \cdot 1 0 ^ { - 3 }$

This toy problem also shows the main limitation of the sequential strategy. The removal step is greedy. If the first recovered line absorbs points belonging to another nearby line, later passes inherit that mistake. Thus, this is not a complete multi-surface estimator or a full reconstruction method for polygonal cell complexes. It is a useful diagnostic for well-separated affine interfaces, but genuinely interacting or too close switching sets would require a joint model-selection step rather than repeated single-line recovery.

![](images/3e3c542b19c55209564189795ccca946fb629a1783f520ad008fba589e8ab8d4.jpg)

![](images/7d650d9bb898a2558ffe5fa2dab2e7eb981cf9709f69f5ff1326d61fb1f05f08.jpg)  
Figure 12: Number of inliers for each RANSAC attempt, after removing the previous inliers.  
remaining current inliers matched true line recovered line  
Figure 13: Sequential RANSAC–TLS extraction. In each panel, black points are the inliers selected during that pass, while light grey points remain available for later passes; points removed during earlier passes are omitted. The solid black line is the matched true line, and the dashed cyan line is the recovered TLS-refined estimate.