# STAY OR STRAY – A DYNAMICAL SYSTEMS VIEWPOINT OFPOPULARITY BIAS

Sarvesh Shashidhar Centre of Machine Intelligence & Data Science Indian Institute of Technology, Bombay Mumbai, Maharashtra, India ssarvesh@iitb.ac.in

Arpit Agarwal Dept. of Computer Science & Engineering Indian Institute of Technology, Bombay Mumbai, Maharashtra, India aarpit@iitb.ac.in

Karan Bhukar Amazon Music Bengaluru, Karnataka, India bhukar@amazon.com

Lankireddy Prabhat Electrical & Computer Engineering Dept. Georgia Institute of Technology Atlanta, GA, USA plankireddy3@gatech.edu

D. Manjunath Dept. of Electrical Engineering Indian Institute of Technology, Bombay Mumbai, Maharashtra, India dmanju@iitb.ac.in

Tanmay Khandelwal Amazon Music Sunnyvale, California, USA tanmayx@amazon.com

ABSTRACT

Popularity bias in recommendation systems arises when a majority user class generates disproportion ate interaction data, causing the system to increasingly favour it while degrading recommendation quality for niche users. While extensive empirical evidence of popularity bias exists, the dynamics leading to its emergence are not well understood. In this work, we study the coupled evolution of recommender model updates and user engagement through the lens of dynamical systems. We formulate a stochastic process and analyse its asymptotic behaviour through an ordinary differential equation (ODE) framework grounded in two-time-scale stochastic approximation. We characterise the equilibrium points of this dynamical system, and derive conditions under which popularity bias is provably emergent, as well as conditions under which symmetric retention of all user classes is possible. We conduct experiments on synthetic data and real-world production logs derived from a large-scale commercial music recommendation platform to validate our theoretical results.

## 1 Introduction

Modern recommendation systems (RS) have become integral to digital media platforms [1], mediating the interaction between users and items by learning user preferences from observed behaviour. The effectiveness of these systems relies on the accurate modelling of user preferences, typically achieved through online learning on interaction data [2, 3]. In practice, however, user populations are rarely balanced [4, 5]– a dominant user class contributes a disproportionate share of interactions, leading the system to preferentially learn majority preferences at the expense of niche groups. This phenomenon, commonly known as popularity bias [6], has been well documented in the literature [7, 8, 9, 10].

However, the vast majority of existing work examines popularity bias through a static lens, capturing only a snapshot of the bias and studying its consequences without taking into account the underlying mechanisms that generate and amplify it over time [7, 8]. In this work, we take a dynamical systems perspective to study the emergence of popularity bias in recommendation systems and its downstream effects on niche user churn.

We formulate a model where there are two classes of users: a majority (popular) class and a minority (niche) class. The recommender system is an online learner that updates its parameters to better match the user class with its preferred content, while users update their rates of arrival based on preference match.<sup>1</sup> To understand the long-term dynamics of this coupled system, we employ a continuous-time ordinary differential equation (ODE) framework grounded in two-timescale stochastic approximation [11]. Specifically, we model the system as evolving on two different timescales: the recommendation algorithm adapts rapidly with each user interaction, while the user population is slow in adjusting its arrival rate.<sup>2</sup> This two-timescale analysis permits treating the arrival rates as quasi-static relative to model parameter changes. Substituting the optimal model parameters for the current arrival rates yields a simplified ODE over arrival rates alone, enabling closed-form analysis of long-term equilibrium behavior.

We characterise the equilibrium structure and convergence properties of this reduced system. Under a non-collinearity condition on the class-conditional feature means, we show that the four corners of $[ 0 , \dot { 1 } ] ^ { 2 }$ are valid equilibrium points, each admitting a natural interpretation– (1, 1) corresponds to retention of both user classes, (1, 0) and (0, 1) correspond to selective disengagement of one class (i.e., popularity bias), and (0, 0) corresponds to simultaneous disengagement. Our analysis yields three principal theoretical results. First, we prove that the equilibrium (0, 0) is not asymptotically reachable from any interior initial condition, thereby ruling out simultaneous disengagement as a feasible long-run outcome (Theorem 4.1). Second, we derive a threshold $p ^ { * }$ on the fraction of majority users such that when $p > p ^ { * }$ , the system asymptotically converges to (1, 0)—the niche class disengages entirely while the majority class is retained— formally characterising the onset of popularity bias (Theorem 4.2). Third, we identify two complementary regimes that guarantee symmetric retention, i.e., convergence to (1, 1)– a sufficient condition based on the sign of the inner product of the class-conditional means in a covariance-transformed space (Theorem 4.3), and an existence result that identifies a feasible interval of p values ensuring both classes are retained, even when the sufficient condition of Theorem 4.3 does not hold (Theorem 4.4).

We validate our theoretical predictions through simulations on synthetic data, demonstrating that the system trajectories conform to the equilibrium behaviour predicted by our theorems across a broad range of initialisations and parameter configurations. Additionally, we construct a semi-synthetic evaluation using production engagement logs from a large-scale commercial music recommendation platform comprising approximately 410 million user–item interactions. By fitting our model to user embeddings derived from matrix factorisation on this interaction data, we show that the observed platform dynamics—including disproportionate churn among niche-preferring users—are consistent with the conditions identified by our framework. Our theoretical framework coupled with empirical evidence suggests a natural mitigation strategy that focuses on balancing the accuracy of the user classes which is similar in spirit to [13]. Our work provides additional justification for this strategy and we empirically demonstrate its effectiveness in reducing popularity bias.

## 2 Related Work

Our work is closely related to the literature on popularity bias and dynamical system analysis. Extensive prior research has characterised popularity bias [4] and the degradation of novelty, diversity, and user satisfaction in the long-tail [5, 8, 6]. This has resulted in multiple interventions being proposed at the pre-processing [14], in-processing [15, 16] and post-processing [9] stages. These strategies predominantly address bias symptoms rather than the underlying mechanisms. Concurrently, dynamic RS frameworks have explored user preference drift and exposure effects [12, 17]. However, they typically treat user engagement as an exogenous variable, and the co-evolution of user behaviour and model parameters remains largely unexplored. A detailed section on related works has been provided in supplementary Section A.

## 3 Dynamic Model Formulation

Binary user-classes. We consider a model consisting of two user groups– class +1 (majority/popular users) and class 1 (minority/niche users). The items are similarly partitioned into two classes indexed b $\{ - \bar { 1 } , \bar { + } 1 \}$

Assumption 1. The users ofclass $c _ { u } \in \{ - 1 , + 1 \}$ prefer the items belonging to class $c _ { i } \in \{ - 1 , + 1 \} i f c _ { u } = c _ { i } .$ . As a result, the user utility reduced to the bilinearform

$$
U t i l i t y ( u , i ) = c _ { u } \cdot c _ { i }
$$

where $c _ { u }$ and $c _ { i }$ represent the classes of the user and item respectively.

Under assumption 1, user preferences are perfectly aligned with class identity. Consequently, the task of providing recommendations reduces to learning a binary classifier over user features. This simplification makes our analysis mathematically tractable; the extension of the results to a multi-class system follows naturally and has been detailed in Section B in the supplementary material.

Arrival dynamics. We now formalize the notation governing the dynamics. Let $p \in ( 0 , 1 )$ denote the fraction of users belonging to class +1. User arrival into the system is stochastic – at time $t ,$ a class +1 user enters the system with probability $\mathsf { \bar { ( \alpha _ { 1 } ) } } _ { t } \in [ 0 , 1 ]$ , while a niche user enters with probability $( \alpha _ { - 1 } ) _ { t } \in [ 0 , 1 ]$ . Quantities $( \alpha _ { 1 } ) _ { t }$ and $( \alpha _ { - 1 } ) _ { t }$ , also called walk-up rates, combined with user class imbalance gives us the overall probability of the arrival of class +1 users and class 1 users as $p ( \alpha _ { 1 } ) _ { t }$ and $( 1 - p ) ( \alpha _ { - 1 } )$ <sub>t</sub> respectively. With the remaining probability, no user arrives to the system (equivalent to having $y _ { t } = 0 )$ . For notational convenience, we shall define $\alpha _ { t } = ( ( \alpha _ { 1 } ) _ { t } , ( \alpha _ { - 1 } ) _ { t } )$ as a 2-tuple of walk-up rates.

User features. The user arriving at time t is represented using a feature vector $X _ { t } \in \mathbb { R } ^ { d }$ and class label $y _ { t } \in \{ - 1 , + 1 \}$ The users are then classified using an online binary classifier, parameterised by ${ \theta } _ { t } \in \mathbb { R } ^ { d }$ which is learnable. The classifier parameter $\left( \boldsymbol { \theta } _ { t } \right)$ and the walk-up rates of the users $( ( \alpha _ { 1 } ) _ { t } ^ { - } , ( \alpha _ { - 1 } ) _ { t } )$ evolve jointly over time, with the resulting coupled dynamics dictating the long-term user behaviour. The user preferences (which remain static with time) aid in characterizing the extent of class imbalance, i.e., value of p.

Assumption 2. The feature vectors of the users, $X _ { t } \in \mathbb { R } ^ { d }$ , are sampled from class-conditional Gaussian distributions.

$$
X _ { t } | _ { y _ { t } = 1 } \sim { \mathcal { N } } ( \mu _ { 1 } , \Sigma _ { 1 } ) \quad ; \quad X _ { t } | _ { y _ { t } = - 1 } \sim { \mathcal { N } } ( \mu _ { - 1 } , \Sigma _ { - 1 } )
$$

Imposing Gaussian priors on user representations is a well-known technique [18, 19] that allows for tractable analysis. In our case, it allows for a closed-form expression for the optimal classifier threshold $\theta ^ { * }$ going forward. In our experimental analysis in Section 5.5 and supplementary Section J.5, we show that the theoretical trends hold for real-world data even though it is not perfectly Gaussian in nature.

Parameter updates. We now outline the discrete time operation of the RS, which is an online binary classifier. At each time step t, a user with feature vector $X _ { t }$ is classified by the model using the rule $\hat { y } _ { t } = \mathrm { s i g n } \left( \theta _ { t } ^ { \top } \dot { X } _ { t } \right) . \mathrm { I f } \hat { y } _ { t } = y _ { t }$ (correctly classified), then the users get serviced optimally by the RS and they experience an increase in their walk-up rates $( ( \dot { \alpha _ { y _ { t } } } ) _ { t + 1 } > ( \alpha _ { y _ { t } } ) _ { t } )$ , leading to an overall increase in their arrival. On the other hand, a misclassification $( \hat { y } _ { t } \neq y _ { t } )$ results in a decrease in walk-up rates $( ( \alpha _ { y _ { t } } ) _ { t + 1 } < ( \alpha _ { y _ { t } } ) _ { t } )$ . After each update, the model updates its parameters, i.e. $\theta _ { t }$ to ensure lower misclassification rates.

We now derive the continuous-time approximation of the discrete stochastic update mentioned above using ordinary differential equations (ODEs). Let us first characterize the evolution of $\theta _ { t } .$ , which is updated by the RS to minimize the regularised mean-squared error (MSE) defined as follows –

$$
\begin{array} { c } { \displaystyle \ell ( \theta _ { t } ; X _ { t } , y _ { t } ) = \frac 1 2 [ y _ { t } - \theta _ { t } ^ { \top } X _ { t } ] ^ { 2 } + \frac { \lambda } { 2 } \| \theta _ { t } \| ^ { 2 } } \\ { \displaystyle L ( \theta _ { t } , \alpha _ { t } ) = \mathbb { E } \left[ \ell ( \theta _ { t } ; X _ { t } , y _ { t } ) | \theta _ { t } , \alpha _ { t } \right] } \end{array}
$$

While it is natural to use either hinge or log loss for online binary classifiers, recent work [20] has shown that MSE loss also empirically performs well. Additionally, using regularised MSE loss allows for a closed-form expression for $\theta ^ { * }$ . However, we perform experiments with other loss functions in supplementary Section C to show that our theoretical insights are valid beyong the MSE loss.

The value of $\theta _ { t }$ is updated using online gradient descent over $L ( \theta _ { t } )$ , which can be viewed as a discretisation of the gradient flow ODE [11].

$$
\theta _ { t + 1 } = \theta _ { t } - \eta _ { t } \nabla _ { \theta } \left( \ell ( \theta _ { t } ) \right) \quad \equiv \quad \dot { \theta } = - \nabla _ { \theta } \left( L ( \theta , \alpha ) \right)\tag{1}
$$

Now we characterize the update for the walk-up rates, starting with $( \alpha _ { 1 } ) _ { t }$ . The value of $\left( \alpha _ { 1 } \right)$ <sub>t</sub> either increases $( \hat { y } _ { t } = y _ { t } = 1 )$ , decreases $( \hat { y } _ { t } \neq y _ { t } = 1 )$ or doesn’t change $( y _ { t } = - 1 )$ . These three conditions can be combined to give a single expression for updating $\left( \alpha _ { 1 } \right)$ <sub>t</sub>.

$$
( \alpha _ { 1 } ) _ { t + 1 } = ( \alpha _ { 1 } ) _ { t } + \delta _ { t } \left[ \mathrm { s i g n } ( \theta _ { t } ^ { \top } X _ { t } ) \cdot \left( \frac { 1 + y _ { t } } { 2 } \right) \right]
$$

where $\delta _ { t }$ is the time-decaying step-size to ensure asymptotic convergence. Taking conditional expectation over $X _ { t }$ yields the continuous time stochastic update for $( \alpha _ { 1 } ) _ { t }$ <sub>t</sub>.

$$
\begin{array} { r l } & { \frac { 1 } { \delta _ { t } } \mathbb { E } \left[ ( \alpha _ { 1 } ) _ { t + 1 } - ( \alpha _ { 1 } ) _ { t } \right] = \mathbb { E } \left[ \mathrm { s i g n } ( \theta _ { t } ^ { \top } X _ { t } ) \cdot \left( \frac { 1 + y _ { t } } { 2 } \right) \right] } \\ & { = \mathbb { P } [ y _ { t } = 1 ] \left\{ \mathbb { P } _ { X _ { t } | y _ { t } = 1 } [ \theta _ { t } ^ { \top } X _ { t } \geq 0 ] - \mathbb { P } _ { X _ { t } | y _ { t } = 1 } [ \theta _ { t } ^ { \top } X _ { t } < 0 ] \right\} } \\ & { = p ( \alpha _ { 1 } ) _ { t } \left\{ 2 ( \hat { p } _ { 1 } ) _ { t } - 1 \right\} } \end{array}
$$

where $( \hat { p } _ { 1 } ) _ { t } = \mathbb { P } _ { X _ { t } | y _ { t } = 1 } [ \theta _ { t } ^ { \top } X _ { t } \ge 0 ]$ is the True Positive Rate (TPR) of the classifier at time t. Applying projected stochastic approximation yields the ODE for $( \alpha _ { 1 } ) _ { t }$

$$
\dot { \alpha } _ { 1 } = \Gamma \left\{ p \alpha _ { 1 } [ 2 \hat { p } _ { 1 } - 1 ] \right\}\tag{2}
$$

where $\Gamma ( x )$ is a projection operator that restricts x to the compact set [0, 1]. Similarly, the update for $( \alpha _ { - 1 } ) _ { t }$ can also be approximated to a continuous-time ODE.

$$
\begin{array} { r l } & { ( \alpha _ { - 1 } ) _ { t + 1 } = ( \alpha _ { - 1 } ) _ { t } + \delta _ { t } \left[ - \mathrm { s i g n } ( \theta _ { t } ^ { \top } X _ { t } ) \cdot \left( \frac { 1 - y _ { t } } { 2 } \right) \right] } \\ & { \qquad \dot { \alpha } _ { - 1 } = \Gamma \left\{ ( 1 - p ) \alpha _ { - 1 } [ 2 \hat { p } _ { - 1 } - 1 ] \right\} } \end{array}\tag{3}
$$

Where $( \hat { p } _ { - 1 } ) _ { t } = \mathbb { P } _ { X _ { t } \mid y _ { t } = - 1 } [ \theta _ { t } ^ { \top } X _ { t } < 0 ]$ is the True Negative Rate (TNR) of the classifier at time t. Equations 1, 2 & 3 characterize the coupled dynamics of model parameters $( \theta _ { t } )$ and user behaviour $\left( \alpha _ { t } \right)$ .

## 3.1 Two-Time-Scale Stochastic Approximation

In a practical RS, the users (at an aggregate level) have inertia and tolerance [12] towards sub-optimal recommendations, while the parameters of the RS are updated much more frequently. Essentially, the user arrival rates are expected to evolve at a slower time scale when compared to the model parameters.

Assumption 3. Consider the stochastic updates –

$$
\begin{array} { c } { \theta _ { t + 1 } = \theta _ { t } - \eta _ { t } \nabla _ { \theta } \ell \bigl ( \theta _ { t } ; X _ { t } , y _ { t } \bigr ) } \\ { ( \alpha _ { 1 } ) _ { t + 1 } = ( \alpha _ { 1 } ) _ { t } + \delta _ { t } H _ { 1 } ( \theta _ { t } , X _ { t } , y _ { t } ) } \\ { ( \alpha _ { - 1 } ) _ { t + 1 } = ( \alpha _ { - 1 } ) _ { t } + \delta _ { t } H _ { - 1 } ( \theta _ { t } , X _ { t } , y _ { t } ) } \end{array}
$$

where the mean-driftsfor each quantity— $- \nabla _ { \boldsymbol { \theta } } L , \mathbb { E } [ H _ { 1 } ] , \mathbb { E } [ H _ { 2 } ] - \boldsymbol { \iota }$ are Lipschitz. Additionally, the decaying step-sizes— $\eta _ { t } , \delta _ { t } - a r e$ such that $\begin{array} { r } { \sum _ { t = 0 } ^ { \infty } \eta _ { t } = \stackrel { - } { \infty } , \sum _ { t = 0 } ^ { \infty } \delta _ { t } = \stackrel { - } { \infty } , } \end{array}$ , while both $\textstyle \sum _ { t = 0 } ^ { \infty } \eta _ { t } ^ { 2 }$ and $\textstyle \sum _ { t = 0 } ^ { \infty } \delta _ { t } ^ { 2 }$ are finite. Then, we assume that lim $\begin{array} { r } { \mathsf { i } _ { t  \infty } \frac { \delta _ { t } } { \eta _ { t } } = 0 } \end{array}$ i.e. $\theta _ { t }$ evolves faster than $\alpha _ { t }$

Not only does Assumption 3 highlight the relation between user and model updates observed in practice, it also is the foundational idea on which we build our framework. With the appropriate choice of $\delta _ { t }$ and $\eta _ { t }$ (as provided by [11]), we can effectively treat the faster transient $( \alpha _ { t } )$ as quasi-static with respect to the slower transient $( \theta _ { t } )$ . This allows us to find an expression for $\theta ^ { * }$ in terms of $\alpha _ { t }$ . Note that if we flip the time scales, we might end up in a situation where the users update their behaviour faster than the system can learn their preferences, which is not ideal. With $\alpha _ { t } = \alpha$ (static), we write the expression for $\theta ^ { * }$ as follows.

$$
\begin{array} { c } { { \theta ^ { * } ( \alpha ) = F ( \alpha ) = A _ { \alpha } ^ { - 1 } b _ { \alpha } } } \\ { { A _ { \alpha } = p \alpha _ { 1 } ( \Sigma _ { 1 } + \mu _ { 1 } \mu _ { 1 } ^ { \top } ) } } \\ { { + ( 1 - p ) \alpha _ { - 1 } ( \Sigma _ { - 1 } + \mu _ { - 1 } \mu _ { - 1 } ^ { \top } ) + \lambda I } } \\ { { b _ { \alpha } = p \alpha _ { 1 } \mu _ { 1 } - ( 1 - p ) \alpha _ { - 1 } \mu _ { - 1 } } } \end{array}\tag{4}
$$

This expression for $\theta ^ { * }$ is the minimizer of $L ( \theta _ { t } )$ and is found by setting $\nabla _ { \theta } L ( \theta _ { t } ) = 0$ . The derivation of Equation (4) along with a proof for Lipschitzness of $F$ (using first principles) has been deferred to Section E in the supplementary material. Crucially, thanks to Assumption 3, our framework has reduced to a two-ODE system of just the walk-up rates. This elegant reduction using two-time scale separation has been explored by [11], the details of which have been highlighted in Section D in the supplementary material. After treating $\alpha _ { t }$ as static and getting $\theta ^ { * }$ , the framework then uses $\bar { \theta ^ { * } }$ to update $\alpha _ { t }$ . The cycle continues and this gives rise to their coupled evolution.

![](images/adccbee7cb286f11e5877b07c70c55acac860f8cfb0e2c308bf01035d177eef9.jpg)  
Figure 1: Phase plot of the baseline system after 100, 000 time steps with $p > p ^ { * }$ (left) and $p < p ^ { * }$ (right).

## 3.2 Characterizing System Equilibria

Using the two-ODE framework defined so far, we aim to find the limiting behaviour of the users and the system, which essentially boils down to finding the equilibrium points where the system converges to asymptotically. Any point $\alpha ^ { * } = ( \alpha _ { 1 } ^ { * } , \alpha _ { - 1 } ^ { * } ) \in [ 0 , 1 ] ^ { 2 }$ is said to be an equilibrium point of the system if $\dot { \alpha } _ { 1 } | _ { \alpha ^ { * } } = \dot { \alpha } _ { - 1 } | _ { \alpha ^ { * } } = 0 .$ , which also results in $\dot { \theta } | _ { \alpha ^ { * } } = 0$

Prior to identifying the equilibrium points, we use Assumption 2 to characterise the evolution of the walk-up rates in terms of means of the user feature distributions.

Lemma 3.1. For all $\alpha ^ { * } \in [ 0 , 1 ] ^ { 2 }$ and $\theta ^ { * } = F ( \alpha ^ { * } )$ , we see that $\dot { \alpha } _ { 1 } | _ { \alpha ^ { * } } > 0$ and $\dot { \alpha } _ { - 1 } | _ { \alpha ^ { * } } > 0$ are equivalent to having $( \theta ^ { * } ) ^ { \top } \mu _ { 1 } > 0$ and $( \theta ^ { * } ) ^ { \top } \mu _ { - 1 } < 0$ respectively.

Proof sketch. The lemma above follows from the fact that for user class i, the true rate $\hat { p } _ { i }$ depends (by definition) on the value of $\theta ^ { \top } X$ which is a random variable of the Gaussian distribution $\mathcal { N } ( \boldsymbol { \theta } ^ { \top } \boldsymbol { \mu } _ { i } , \boldsymbol { \theta } ^ { \top } \Sigma _ { i } \boldsymbol { \theta } ) ^ { * }$ . Performing simple substitution, as highlighted in detail in supplementary Section F, proves Lemma 3.1. It must be noted that Lemma 3.1 represents the necessary & sufficient conditions for an increase in walk-up rates.

From Lemma 3.1, we see that $\dot { \alpha } _ { 1 } = 0$ is equivalent to $\theta ^ { \top } \mu _ { 1 } = 0$ . Now suppose if the means of the distributions are collinear i.e. $\exists c \in \mathbb { R }$ such that $\mu _ { - 1 } = c \cdot \mu _ { 1 }$ , then $\theta ^ { \top } \mu _ { 1 } = 0$ is equivalent to $\theta ^ { \top } \mu _ { - 1 } = 0$ which implies $\dot { \alpha } _ { - 1 } = 0$ Hence, equilibrium analysis for collinear means becomes trivial, and we choose not to focus on this degenerate case.

Assumption 4. Consider the means– $\mathbf { \nabla } \cdot \mu _ { 1 } , \mu _ { - 1 }$ ofthe distribution ofuserfeatures. Then $\nexists c \in \mathbb { R }$ such that $\mu _ { 1 } = c \cdot \mu _ { - 1 }$

Now we can proceed to find the equilibrium points.

Lemma 3.2. Under Assumptions 1–4, the set of equilibria is given by $\mathcal { E } ~ = ~ \{ ( 0 , 0 ) , ( 1 , x ) , ( y , 1 ) \}$ where $x =$ $\begin{array} { r } { \frac { p \left. \mu _ { - 1 } , \mu _ { 1 } \right. _ { A ^ { - 1 } } } { ( 1 - p ) \| \mu _ { - 1 } \| _ { A ^ { - 1 } } ^ { 2 } } \left. a n d y \right. = \frac { ( 1 - \bar { p } ) \left. \mu _ { - 1 } , \mu _ { 1 } \right. _ { A ^ { - 1 } } } { p \| \mu _ { 1 } \| _ { A ^ { - 1 } } ^ { 2 } } } \end{array}$

Proof sketch The first step is to set Equations 2 and 3 to 0. Next, by checking the conditions of Equations 2 and 3 at corner, edge and internal points, along with using the result from Lemma 3.1, gives us the results in Lemma 3.2. A detailed proof of this has been deferred to Section F in the supplementary material.

## 4 Convergence analysis

Having identified the equilibrium points, we now proceed to define the necessary and sufficient conditions for the system to converge to them. This analysis reveals how initial parameters govern the system’s asymptotic behaviour.

## 4.1 Non-Attainability of Simultaneous User Drop-off

When the RS fails to optimally serve both user classes concurrently, both $\alpha _ { 1 }$ and $\alpha _ { - 1 }$ decrease, causing asymptotic convergence to (0, 0) (simultaneous user drop-off). Characterising this phenomenon indicates which initial conditions are non-ideal for both user classes.

Theorem 4.1. Under Assumptions 1–4 and for $p \in ( 0 , 1 )$ , the RS starting from a point $\alpha \in [ 0 , 1 ] ^ { 2 } \setminus ( 0 , 0 )$ cannot asymptotically converge to (0, 0)

![](images/c07b481088887907af7654e014401d12539f5225203b45ba764149cd51b2a0c1.jpg)  
Figure 2: System simulation with engineering setup (left) and the baseline setup (right).

Intuition For convergence to (0, 0), the RS needs both TPR and TNR are both below 0.5. Thus, its expected performance must be worse than a random guesser for both classes. However, since we update the walk-up rates using $\theta ^ { * }$ that minimises the expected regularised MSE loss, these misalignments are corrected for, making (0, 0) convergence impossible.

Proof Sketch Theorem 4.1 is proven by simultaneously solving $\dot { \alpha } _ { 1 } < 0$ and $\dot { \alpha } _ { - 1 } < 0$ . Lemma 3.1 helps re-write these inequalities as $( \theta ^ { * } ) ^ { \top } \mu _ { 1 } < \bar { 0 }$ and $( \theta ^ { * } ) ^ { \top } \mu _ { - 1 } > 0$ respectively, which violate the Cauchy-Schwarz inequality. This makes the convergence to $( 0 , 0 )$ mathematically impossible; a detailed proof has been provided in supplementary Section G.1.

## 4.2 Emergence of Popularity Bias

Popularity bias emerges when the RS favours one user class over another due to its higher representation. In the framework, this leads to the system converging to either (1, 0) or (0, 1), depending on the favoured class. This section characterises the conditions for the system to converge to (1, 0), and the analysis for convergence to (0, 1) follows symmetrically (refer to supplementary Section G.4).

Theorem 4.2. Consider the RS model under Assumptions 1–4 and the mean vectors– $\mathbf { \nabla } \mu _ { 1 } , \mu _ { - 1 }$ such that $\left. \mu _ { - 1 } , \mu _ { 1 } \right. _ { A _ { \alpha } ^ { - 1 } } >$ 0 where $\langle \cdot \rangle _ { A _ { \alpha } ^ { - 1 } }$ 1 is the inner product admitted by the space transformed by operator $A _ { \alpha } ^ { - 1 }$ . Then, the system trajectory tends towards $( 1 , 0 ) i f -$

$$
p > p ^ { * } = \frac { \alpha _ { - 1 } \left. \mu _ { - 1 } \right. _ { A _ { \alpha } ^ { - 1 } } ^ { 2 } } { \alpha _ { 1 } \langle \mu _ { 1 } , \mu _ { - 1 } \rangle _ { A _ { \alpha } ^ { - 1 } } + \alpha _ { - 1 } \left. \mu _ { - 1 } \right. _ { A _ { \alpha } ^ { - 1 } } ^ { 2 } }\tag{5}
$$

where $\| \cdot \| _ { A _ { \alpha } ^ { - } }$ 1 is the norm induced in the space transformed by $A _ { \alpha } ^ { - 1 }$ . Additionally, the system shall asymptotically converge to (1, 0) if and only if

$$
\frac { \langle \mu _ { - 1 } , \mu _ { 1 } \rangle _ { B _ { \alpha } } } { \langle \mu _ { - 1 } , \mu _ { 1 } \rangle _ { A _ { \alpha } ^ { - 1 } } } - \frac { \| \mu _ { - 1 } \| _ { B _ { \alpha } } ^ { 2 } } { \| \mu _ { - 1 } \| _ { A _ { \alpha } ^ { - 1 } } ^ { 2 } } \in \left[ \frac { 1 } { ( p - 1 ) \alpha _ { - 1 } } , \frac { 1 } { p \alpha _ { 1 } } \right]\tag{6}
$$

where $B _ { \alpha } = A _ { \alpha } ^ { - 1 } \left( \Sigma _ { 1 } + \mu _ { 1 } \mu _ { 1 } ^ { \top } \right) A _ { \alpha } ^ { - 1 }$

Intuition For the RS, $p ^ { * }$ from Equation (5) acts as a critical threshold on $p ;$ exceeding which triggers popularity bias, driving the system towards (1, 0). This implies the RS can tolerate a deterministic level of class imbalance before favouring the dominant class. Notably, $p ^ { * }$ depends on both α and $\theta ^ { * }$ and consequently evolves with time. This is noted in the second statement in Theorem 4.2. Bounding the variance differences as in Equation (6) ensures $p ^ { * }$ remains non-increasing as the system approaches (1, 0), creating an inescapable feedback loop that allows asymptotic convergence.

Proof sketch We derive $p ^ { * }$ in Equation (5) by simultaneously solving $\dot { \alpha } _ { 1 } > 0$ and $\dot { \alpha } _ { - 1 } < 0$ using Lemma 3.1. By restricting to the valid cases that obey the Cauchy-Schwarz inequality, we get the expression from Equation (5), detailed in supplementary Section G.2. To ensure asymptotic convergence, $p ^ { * }$ must be non-increasing with respect to α as it approaches (1, 0). This requires $\begin{array} { r } { \frac { \partial p ^ { * } } { \partial \alpha _ { 1 } } \leq 0 \mathrm { ~ a n d ~ } \frac { \bar { \partial } p ^ { * } } { \partial \alpha _ { - 1 } } \geq 0 } \end{array}$ , which when solved together yield Equation (6). This part of the proof has been elaborated in supplementary Section G.3.

![](images/f653716e51e556283856854908e7861f1d05bbc436046507131c61b1b47f6450.jpg)  
Figure 3: Scatter plot for $p \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$ and $\alpha _ { i } \in [ 0 . 0 1 , 0 . 9 9 ]$ . Each point corresponds to the mean of 100 MC runs, each for 10, 000 time steps.

## 4.3 Conditions for Symmetric Retention

Having formalised the non-ideal conditions, we now analyse the ideal case of symmetric user retention, i.e. convergence towards (1, 1).

Theorem 4.3. Consider a system under Assumptions 1–4 such that $\langle \mu _ { 1 } , \mu _ { - 1 } \rangle _ { N _ { \alpha } ^ { - 1 } } < 0$ where $N _ { \alpha } = p \alpha _ { 1 } \Sigma _ { 1 } + ( 1 -$ $p ) \alpha _ { - 1 } \Sigma _ { - 1 } + \lambda I$ , then the system shall asymptotically converge to $( 1 , 1 )$ as long as $\begin{array} { r } { \operatorname* { s u p } _ { \alpha } \left. \mu _ { 1 } , \mu _ { - 1 } \right. _ { N _ { \alpha } ^ { - 1 } } < 0 . } \end{array}$

Intuition The core idea behind Theorem 4.3 is that “well-separated” means (a negative inner product) enable the RS to accurately identify user distributions and optimally classify both classes. Enforcing $\mathrm { s u p } _ { \alpha } \left. \mu _ { 1 } , \mu _ { - 1 } \right. _ { N _ { \alpha } ^ { - 1 } } < 0$ ensures this separation of the means across all $\alpha \in [ 0 , 1 ] ^ { 2 }$ . As a result, this condition holds as the system evolves towards $( 1 , 1 )$ thus ensuring asymptotic convergence. Additionally, the inner product is now taken in the space transformed by $\mathrm { \tilde { N } } _ { \alpha } ^ { - 1 }$ instead of $A _ { \alpha } ^ { - 1 }$ . The proof sketch highlights that these two inner products are equivalent in our case.

Proof sketch To prove Theorem 4.3, we substitute $\langle \mu _ { 1 } , \mu _ { - 1 } \rangle _ { N _ { \alpha } ^ { - 1 } } < 0$ for all α into the expressions for $\dot { \alpha } _ { 1 }$ and $\dot { \alpha } _ { - 1 } . \mathrm { ~ A ~ }$ pplying Lemma 3.1 then yields $\dot { \alpha } _ { 1 } > 0$ and $\dot { \alpha } _ { - 1 } > 0$ . Using simple substitution for $A _ { \alpha }$ can also show that $\langle \mu _ { 1 } , \mu _ { - 1 } \rangle _ { A _ { \alpha } ^ { - 1 } }$ 1 and $\left. \mu _ { 1 } , \mu _ { - 1 } \right. _ { N _ { \alpha } ^ { - } }$ 1 share the same sign, rendering them equivalent for our case. The detailed proofs of these claims are in supplementary Section G.5.

Special case A special case of Theorem 4.3 arises when user features follow isotropic Gaussians, where $\Sigma _ { 1 } = \sigma _ { 1 } ^ { 2 } I$ and $\Sigma _ { - 1 } = \sigma _ { - 1 } ^ { 2 } I $ . Here, N is a scalar multiple of the identity, meaning $\begin{array} { r } { \langle \mu _ { 1 } , \mu _ { - 1 } \rangle _ { N ^ { - 1 } } = \frac { 1 } { k } \langle \mu _ { 1 } , \mu _ { - 1 } \rangle } \end{array}$ for some $k \in \mathbb { R }$ Thus, ensuring $\langle \mu _ { 1 } , \mu _ { - 1 } \rangle < 0$ in the original space is sufficient to guarantee asymptotic convergence to (1, 1).

Although Theorem 4.3 provides sufficient conditions for symmetric user retention, these conditions are not necessary. Convergence to (1, 1) still occurs even when the mean inner product is positive, subject to stricter constraints formalised below.

Theorem 4.4. Assume the RS model under Assumptions 1-4, initialised at an interior point $\alpha ^ { 0 } = ( \alpha _ { 1 } ^ { 0 } , \alpha _ { - 1 } ^ { 0 } )$ . Suppose that $\left. \mu _ { - 1 } , \mu _ { 1 } \right. _ { A _ { \alpha } ^ { - 1 } } > 0$ for all $\alpha \in [ 0 , 1 ] ^ { 2 }$ such that $\left. \mu _ { - 1 } , \mu _ { 1 } \right. _ { A _ { \alpha } ^ { - 1 } } < \left. \mu _ { - 1 } , \mu _ { - 1 } \right. _ { A _ { \alpha } ^ { - 1 } }$ . Also consider functions $f _ { i } ( \alpha , p ) : = ( \theta _ { \alpha } ^ { * } ) ^ { \top } \mu _ { i }$ and $u _ { i } : = A _ { \alpha } ^ { - 1 } \mu _ { i } f o r i \in \{ 1 , - 1 \}$ . Additionally, ifthefollowing conditions hold –

$$
\begin{array} { r l } & { u _ { i } ^ { \top } ( \alpha _ { i } S _ { i } + \lambda I ) u _ { - i } > - \alpha _ { i } ( u _ { i } ^ { \top } S _ { - i } u _ { i } + \frac { \lambda } { \alpha _ { - i } } \| u _ { i } \| ^ { 2 } ) } \\ & { \| u _ { 1 } \| _ { A _ { \alpha } } ^ { 2 } \geq \left. b _ { \alpha } , S _ { 1 } u _ { 1 } \right. _ { A _ { \alpha } ^ { - 1 } } } \\ & { \| u _ { - 1 } \| _ { A _ { \alpha } } ^ { 2 } \geq - \big \langle b _ { \alpha } , S _ { - 1 } u _ { - 1 } \big \rangle _ { A _ { \alpha } ^ { - 1 } } } \\ & { \big \langle u _ { 1 } , u _ { - 1 } \big \rangle _ { A _ { \alpha } } \leq \operatorname* { m i n } ( - \big \langle b _ { \alpha } , S _ { - 1 } u _ { 1 } \big \rangle _ { A _ { \alpha } ^ { - 1 } } , \big \langle b _ { \alpha } , S _ { 1 } u _ { - 1 } \big \rangle _ { A _ { \alpha } ^ { - 1 } } \big ) } \end{array}
$$

where $S _ { i } = \Sigma _ { i } + \mu _ { i } \mu _ { i } ^ { \top } f o r i \in \{ - 1 , 1 \}$ . Then there exists $p _ { 1 } , p _ { - 1 } \in ( 0 , 1 )$ such that $f _ { 1 } ( \alpha , p _ { 1 } ) = 0$ and $f _ { - 1 } ( \alpha , p _ { - 1 } ) =$ 0, and there exists $p \in ( p _ { 1 } , p _ { - 1 } ) f o r$ which the system converges asymptotically to (1, 1).

Intuition Theorem 4.4 addresses the case where a positive inner product of the means makes it difficult for the RS to distinguish the two classes. We define functions $f _ { 1 } ( \alpha , p )$ and $f _ { - 1 } ( \alpha , p )$ , which Lemma 3.1 identifies as the update expressions for $\alpha _ { 1 }$ and $\alpha _ { - 1 }$ , respectively. The list of conditions mentioned ensures two key results – both functions monotonically increase with $p ,$ and both transition from negative to positive as p goes from $0  1$ . Thus, applying Intermediate Value Theorem (IVT) gives $p _ { 1 } , p _ { - 1 }$ such that $\bar { f _ { 1 } ( \alpha , p _ { 1 } ) = f _ { - 1 } ( \alpha , p _ { - 1 } ) = 0 }$

![](images/e403fe7d1bbe7b33fe61b3dd4d24f3f3fb55b8eab48920ef67eb18e271cf6af4.jpg)  
Figure 4: Phase plot of a setup that was simulated for 10, 000 time steps. The plot highlights a reversal of trend from (1, 1) to (1, 0).

From Lemma 3.1, ensuring both $\dot { \alpha } _ { 1 }$ and $\dot { \alpha } _ { - 1 }$ are positive requires $f _ { 1 } ( \alpha , p ) > 0$ and $f _ { - 1 } ( \alpha , p ) < 0$ , which holds for $p \in ( p _ { 1 } , p _ { - 1 } )$ . Thus, selecting a suitable $p$ ensures asymptotic convergence to (1, 1), despite the positively aligned means.

Proof sketch The formal proof involves multiple steps. First, we establish the monotonicity of $f _ { 1 } , f _ { - 1 }$ by proving their gradients with respect to p are positive. Second, we prove the existence of $p _ { 1 } , p _ { - 1 }$ by showing $f _ { i } ( \alpha , 0 ) < 0$ and $f _ { i } ( \alpha , 1 ) > 0$ for both $i \bar { \in } \{ - 1 , 1 \}$ . Finally, differentiating $f _ { 1 } , f _ { - 1 }$ with respect to α confirms these trends hold across all states, guaranteeing asymptotic convergence to $( 1 , 1 { \bar { ) } }$ . These steps yield the conditions in Theorem 4.4 and the detailed proof is deferred to supplementary Section G.6.

## 4.4 Reversal of Trend

Since $\theta ^ { * }$ and $p ^ { * }$ evolve with $( \alpha _ { 1 } , \alpha _ { - 1 } )$ , it’s possible for the system’s initial trajectory to eventually reverse. Consider a case with $p = ( \alpha _ { 1 } ) _ { 0 } = ( \alpha _ { - 1 } ) _ { 0 } = 0 . 5$ , and the user features distributions $\mu _ { 1 } = [ \bar { 2 } \quad 5 ] , \mu _ { - 1 } = [ - 0 . 5 \quad 3 ] , \Sigma _ { 1 } =$ $5 I , \Sigma _ { - 1 } = 2 I$ . Assuming $\lambda = 1$ , the phase plot (Figure 4) shows the system initially heading towards (1, 1) before reversing to converge at $( 1 , 0 )$ . This is because initially $p ^ { * } = 0 . 5 3 2 > p$ which pushes the system towards (1, 1). However at $( ( \alpha _ { 1 } ) _ { t } , \bar { ( \alpha _ { - 1 } ) _ { t } } ) \approx ( 0 . 6 0 1 , 0 . 5 0 6 ) , p ^ { * }$ drops to $0 . 4 9 < p _ { \mathrm { : } }$ , causing the system to converge to (1, 0) as per Theorem 4.2.

## 5 Experiments

This section focuses on empirically validating the theoretical claims and trends established in Section 4, for which we adopt the following baseline simulation setup –

$$
\begin{array} { r } { p = 0 . 5 ; ( \alpha _ { 1 } ) _ { 0 } = 0 . 5 ; ( \alpha _ { - 1 } ) _ { 0 } = 0 . 5 ; \lambda = 1 } \\ { \mu _ { 1 } = [ 2 , 5 ] ; \mu _ { - 1 } = [ - 0 . 5 , 2 ] ; \Sigma _ { 1 } = 5 I ; \Sigma _ { - 1 } = 3 I } \end{array}
$$

The unbiased initialisation $p = ( \alpha _ { 1 } ) _ { 0 } = ( \alpha _ { - 1 } ) _ { 0 } = 0 . 5$ along with $\lambda = 1$ guarantees a positive definite A. We utilise two-dimensional distributions for visual clarity, selecting means that satisfy Assumption 4 and covariances that inject sufficient noise for both classes. These hyperparameters are varied across experiments to evaluate the different theoretical claims. To account for stochastic user arrivals, we report mean trajectories over 100 Monte Carlo runs.

Simulations were implemented in Python3 using a 64-core Intel Xeon Silver 4216 CPU andfour 48GB NVIDIA RTX A6000 GPUs, complete environment details of which are provided in the README of the code appendix. Finally, because our framework characterises system convergence, we primarily measure the correctness of our claims using system trajectories and final states, explicitly defining alternative metrics for other plots as required.

## 5.1 Verifying Theorem 4.1

To empirically verify Theorem 4.1, scatter plot (Figure 3) of the aggregate shifts in walk-up rates $( \Delta \alpha _ { 1 } , \Delta \alpha _ { - 1 } )$ across varying class imbalances p and α was created. The third quadrant (shaded region) of the plot remains empty, confirming the impossibility of simultaneous decreases in $\alpha _ { 1 }$ and $\alpha _ { - 1 }$ . Additional experiments confirm this finding using phase portraits which deviate away from (0, 0) and scatter plots where terminal states never end up at (0, 0). Perturbation analysis from the origin indicated an increasing $L _ { 2 }$ distance between the current state and the origin with time, verifying the unstable nature of (0, 0). These observations remain invariant to the values of λ and d, as confirmed by ablation studies. The experimental details and plots are present in supplementary Section J.1.

![](images/0b6c5ca9bd4bc2305c1b6aa23d40ba53737e53ea1249e0670d9a40eede85e108.jpg)  
(a) Monotonic nature of $f _ { 1 } , f _ { - 1 } .$  
(b) System trajectory converging at (1, 1).  
Figure 5: The monotonic nature of $f _ { 1 } , f _ { - 1 }$ along with their transition from negative to positive as p goes from $0  1$ (left). Presence of $p _ { 1 } , p _ { - 1 }$ is confirmed and the system trajectory after choosing $p = 0 . 3 \in ( p _ { 1 } , p _ { - 1 } )$ causes the system to converge at (1, 1) (right).

## 5.2 Verifying Theorem 4.2

To validate Theorem 4.2, the baseline configuration (for which $p ^ { * } = 0 . 3 5 )$ was simulated using $p = 0 . 5$ and $p = 0 . 2 5$ The phase portraits in Figure 1 verify that $p = 0 . 5$ converges to $( 1 , 0 )$ unlike $p = 0 . 2 5$ which converges towards (1, 1), thus verifying Theorem 4.2. To verify the monotonic nature of $\boldsymbol { p } ^ { * }$ , a heatmap of $p ^ { * }$ values across different α was created, where the value was monotonically non-increasing as the system approached $( 1 , 0 )$ , as expected. Finally, ablation studies reveal that higher λ or covariance values reduce the value of $p ^ { * }$ since user parameters get over-shadowed in the system dynamics. Detailed explanations and visualisations have been provided in the supplementary material under Section J.2.

## 5.3 Verifying Theorem 4.3

To verify Theorem 4.3, we engineered a specific setup with $\mu _ { - 1 } = [ - 1 , 0 ] , \Sigma _ { 1 } = [ [ 2 . 0 , 0 . 3 ] , [ 0 . 3 , 1 . 0 ] ]$ and $\Sigma _ { - 1 } =$ $[ [ 3 . 0 , - \dot { 0 } . 2 ] , [ - 0 . 2 , 2 . 0 ] ]$ to ensure negative inner product of the means. Under this setup, the system trajectory converges to $( 1 , 1 )$ , providing a stark contrast to the simulation with the baseline configuration as shown in Figure 2. We also verify the case for isotropic Gaussians using the baseline parameters with $\mu _ { 1 } = [ 1 , 0 ]$ and $\mu _ { - 1 } = [ - 1 , 0 ]$ such that $\langle \mu _ { 1 } , \mu _ { - 1 } \rangle < 0 .$ . Simulating the system verifies the sufficiency of this condition for (1, 1) convergence. Subsequent ablations confirm that the behaviour remains consistent across different p, d and λ values. The detailed methodology and plots for these experiments are provided in supplementary Section J.3.

## 5.4 Verifying Theorem 4.4

The key aspect of Theorem 4.4 is the monotonic nature of $f _ { i }$ for $i \in \{ 1 , - 1 \}$ . To verify that, we engineer a setup with $\mu _ { 1 } = [ 2 , 5 ] , \mu _ { - 1 } = [ 1 . 5 , 0 . 8 ] , \Sigma _ { 1 } = [ [ 1 , 0 . 3 ] , [ 0 . 3 , 2 ] ]$ and $\Sigma _ { - 1 } = \mathsf { \bar { \Gamma } } [ [ 2 , - 0 . 2 ] , [ - 0 . 2 , 1 ] ]$ ]. Figure 5 indicates the monotonic increase of $f _ { 1 } , \bar { f } _ { - 1 }$ with respect to p for this setup. It also indicates the change of $f _ { i }$ from negative to positive, implying the existence of $p _ { i }$ for $i \in \{ \bar { 1 } , - 1 \}$ . Additional experiments consist of checking for negative cases, wherein for both $p < p _ { 1 }$ and $p > p _ { - 1 }$ , the system no longer converges to (1, 1). We also plot the gap $\Delta p = p _ { - 1 } - p _ { 1 }$ and indicate its increases as the system moves towards (1, 1), verifying asymptotic convergence. Ablation studies indicate the decrease in $\Delta p$ as λ increases, owing to the domination of system dynamics by λ. The detailed methodology and plots have been deferred to supplementary Section J.4.

## 5.5 Experimenting with Real-World Data

We evaluate the validity of the proposed framework using proprietary interaction logs from a commercial music recommendation platform. The interaction dataset comprises of 410M user–item interactions along with timestamped clicks, ranked exposure positions, and popularity aggregates across time windows. The simulations performed on this data establish & explain the presence of popularity bias using the proposed framework. During pre-processing, items with a global popularity in the top 70-th percentile were classified as popular. Consequently, users were classified into three categories based on the fraction of interaction they have with globally popular items—popular-heavy, nicheheavy & mixed. Popular-heavy and niche-heavy users have 70% of their interaction with popular and niche items respectively.

![](images/a03b1eac47cfb7c2aa62edb4565032c0a3a560926f8e5763fcd70228e739df61.jpg)  
Figure 6: System trajectory for the real-world production logs where the item popularity score is calculated over 14 days.

If a user doesn’t log an interaction for 30 consecutive days, then the user is said to have churned out. Else, they are believed to be retained. We then analyse the interaction data along two dimensions–time and position. The time-based analysis checks user retention using metrics like average number of sessions, average number of searches performed etc. The analysis reveals that close to half of the niche-heavy users churn out while only a quarter of the popular-heavy users get churned out. While performing the position/rank analysis, we observe that churned users in general tend to click on items lower on the recommendation list, highlighting the fact that their preferences on average belong to the category of niche items (supplementary Section J.5).

With the motivation established, we proceed to fit the interaction data into our dynamic system framework. First, the interaction data is fed to a Matrix Factorization model that learns user and item embeddings. Following this, users are classified into popular or niche based on their preference for popular items. Popular users tend to prefer a globally popular item more than 50% of the time. Finally, two separate Gaussian distributions are learned over each user group. This gives us the extent of class imbalance, means and covariance matrices for the distribution of the user features. The system satisfies the conditions required by Theorem 4.2 and we expect popularity bias to manifest, which is confirmed after simulating the system for 100, 000 time steps (Figure 6). This provides concrete evidence that the proposed dynamical system does indeed capture bias present in real-world RS.

These results suggest a natural mitigation strategy based on equalizing error rates across the two classes which is similar in spirit to [13]. We provide experiments in supplementary Section J.6 showing that this strategy is effective in reducing popularity bias.

## 6 Conclusion & Future Work

We presented a dynamical system perspective on popularity bias in recommendation systems and, with the help of continuous-time approximations, provided justification for system behaviour. We provide a lower bound on the extent of class imbalance between the users beyond which popularity bias emerges, causing the niche users to drop off asymptotically. We also provide conditions on system parameters for both user classes to be retained in the system. These findings are then verified by simulations – both on synthetic and real-world data (obtained from a commercial music-recommendation platform). These observations allow for a natural equalised odds mitigation, which has empirically been shown to be effective for the proposed framework.

Future work may extend this framework to generalise over any convex loss function and not just the MSE loss (supplementary Section C). Additionally, relaxation of assumptions on the user feature distributions can lead to more generalised results. More broadly, we believe that the proposed framework can be extended to understand and justify multi-user class dynamics as well (supplementary Section B).

## References

[1] Kuan Zou and Aixin Sun. A survey of real-world recommender systems: Challenges, constraints, and industria perspectives, 2025.

[2] Anastasiia Klimashevskaia, Dietmar Jannach, Mehdi Elahi, and Christoph Trattner. A survey on popularity bias in recommender systems: A. klimashevskaia et al. User Modeling and User-Adapted Interaction, 34(5):1777–1834, 2024.

[3] Gregor Meehan and Johan Pauwels. On inherited popularity bias in cold-start item recommendation. In Proceedings ofthe Nineteenth ACM Conference on Recommender Systems, pages 649–654, 2025.

[4] Yoon-Ju Park and Alexander Tuzhilin. The long tail of recommender systems and how to leverage it. In Proceedings ofthe 2008 ACM Conference on Recommender Systems, pages 11–18, 2008.

[5] Oscar Celma. Music Recommendation and Discovery: The Long Tail, Long Fail, and Long Play in the Digital Music Space. Springer Publishing Company, Incorporated, 1st edition, 2010.

[6] Saumya Bhadani. Biases in recommendation system. In Proceedings of the 15th ACM conference on recommender systems, pages 855–859, 2021.

[7] Marius Kaminskas and Derek Bridge. Diversity, serendipity, novelty, and coverage in recommender systems. ACM Transactions on Interactive Intelligent Systems, 7(1):1–42, 2017.

[8] Joeran Beel, Stefan Langer, Andreas Nürnberger, and Marcel Genzmehr. The impact of demographics (age and gender) and other user-characteristics on evaluating recommender systems. In International conference on theory and practice ofdigital libraries, pages 396–400. Springer, 2013.

[9] Himan Abdollahpouri, Robin Burke, and Bamshad Mobasher. Managing popularity bias in recommender systems with personalized re-ranking. arXiv preprint arXiv:1901.07555, 2019.

[10] Ming He, Changshu Li, Xinlei Hu, Xin Chen, and Jiwen Wang. Mitigating popularity bias in recommendation via counterfactual inference. In International Conference on Database Systems for Advanced Applications, pages 377–388. Springer, 2022.

[11] Vivek S Borkar. Stochastic approximation: a dynamical systems viewpoint, volume 100. Springer, 2008.

[12] Erica Coppolillo, Simone Mungari, Ettore Ritacco, Francesco Fabbri, Marco Minici, Francesco Bonchi, and Giuseppe Manco. Algorithmic drift: A simulation framework to study the effects of recommender systems on user preferences. Information Processing & Management, 62(4):104125, 2025.

[13] Zhe Yu, Joymallya Chakraborty, and Tim Menzies. Fairbalance: How to achieve equalized odds with data pre-processing. IEEE Transactions on Software Engineering, 50(9):2294–2312, 2024.

[14] Mert Gulsoy, Emre Yalcin, and Alper Bilge. Equirate: balanced rating injection approach for popularity bias mitigation in recommender systems. PeerJ Computer Science, 11:e3055, 2025.

[15] Juno Prent and Masoud Mansoury. Correcting popularity bias in recommender systems via item loss equalization. arXiv preprint arXiv:2410.04830, 2024.

[16] Yang Zhang, Fuli Feng, Xiangnan He, Tianxin Wei, Chonggang Song, Guohui Ling, and Yongdong Zhang. Causal intervention for leveraging popularity bias in recommendation. In Proceedings ofthe 44th international ACM SIGIR conference on research and development in information retrieval, pages 11–20, 2021.

[17] Dimitrios Rafailidis and Alexandros Nanopoulos. Modeling users preference dynamics and side information in recommender systems. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 46(6):782–792, 2015.

[18] Andriy Mnih and Russ R Salakhutdinov. Probabilistic matrix factorization. Advances in neural information processing systems, 20, 2007.

[19] Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. Variational autoencoders for collabora tive filtering, 2018.

[20] Sanjeev Arora, Simon S. Du, Wei Hu, Zhiyuan Li, Ruslan Salakhutdinov, and Ruosong Wang. On exact computation with an infinitely wide neural net, 2019.