# Learning Random Geometric Graphs Drawn in Probabilistic Metric Spaces

Dalia Chakrabarty<sup>†</sup>, Kangrui Wang<sup>∗</sup>, Chuqiao Zhang<sup>†</sup>, Ye Liu<sup>‡</sup> ,

∗ Alan Turing Institute

London NW1 2DB

U.K.

kangrui.wang@turing.ac.uk

<sup>†</sup> Department of Mathematics

University of York

York YO1 7DD

U.K.

dalia.chakrabarty@york.ac.uk joe.zhang@brunel.ac.uk

<sup>‡</sup> Howbery Park, Wallingford

OX10 8BA

U.K.

y.liu@hwallingford.com

Abstract: We present a new data-driven learning of a Random Geometric Graph (RGG) of a multivariate dataset, where the graph is drawn in a probabilistic metric space. This graph learning works for generic datasets, irrespective of the type of the observables; their probability distributions; or size of the data. We identify a metric of the space that the graph is drawn in, as a probability distribution of a random variable that we introduce, namely, a variable that represents the disparity between the connectedness of two vertices of the graph, and the correlation between the two random variables that are attached to the respective vertex. It is the closed-form cdf of this disparity variable that we advance as the distance function of the host space of the learnt RGG, such that the edge exists between any two nodes, if this inter-nodal distance falls short of a chosen cutof probability. Drawing the RGG in this probabilistic space leads to the graph being an Soft RGG, such that any edge - if it exists - exists with an identified probability. We forward a simple Rejection Sampling-based technique for learning the probability of any edge. The expected degree distribution of a vertex of this RGG is identified as local, and dependent on the inter-observable correlation matrix. If said correlation matrix is not known, it can be learnt given the data, using its closed-form posterior probability density function, that we forward. We illustrate our graph learning method by learning multiple RGGs of highly multivariate real datasets.

Primary: Random graphs 60-XX; Secondary: Distance in graphs 05C12; Measures of association (correlation, canonical correlation, etc.) 62H20; Probabilistic metric spaces 54E70.

Keywords and phrases: Probabilistic geometric graphs, Probabilistic metric spaces, Distance function on graphs, Bayesian inference using sampling.

## 1. Introduction

Graphs of complex multivariate datasets manifest intuitive illustrations of correlation structures of data, and are of pan-disciplinary interest (Airoldi, 2007; Anandkumar et al., 2012; Bandyopadhyay and Canale, 2016; Benner et al., 2014; Carvalho and West, 2007; Whittaker, 2008).

We present a new method for learning a probabilistic graph of a given multivariate dataset, using the correlation structure of this data, where the graph variable in our consideration is a Random Geometric Graph or RGG (Giles et al., 2016; Penrose, 2003, 2016), that is drawn in probabilistic metric space, (Menger, 1942; Schweizer and Sklar, 1983), s.t. the distance between a pair of nodes, is a probability distribution over positive support. We will show that such an RGG is in fact, a Soft Random Geometric Graph or SRGG (Dettmann, 2018; Penrose, 2016). We recall that the probability for an edge to exist in an SRGG is the “connection function”, that takes as its input, the inter-nodal distance, while in an RGG, an edge exists if the distance between the nodes - that straddle the edge - falls short of a cutof. Said cutof is a probability in the RGG drawn in a probabilistic metric space.

The inter-nodal distance of this RGG is advanced as a metric in the probabilistic metric space that the graph is drawn in, and this distance is used to compute the expected degree distribution of the learnt graph, as local, and dependent on the inter-observable correlation matrix of the given multivariate dataset that comprises values of such observables. We provide a closed-form posterior probability density of this inter-observable correlation matrix. We also provide a closed-form probability of the RGG variable, computed given this correlation matrix. Our graph learning is fast, and is not constrained by the nature of the observables or that of the correlation structure of the data at hand. The size of the data is also not a constraint on our graph learning - we perform an empirical illustration of learning an RGG with a large (∼ 7000) number of nodes.

We start with a preliminary recollection of the basic definitions in Section 2. In Section 3, we present the closed-form density of the inter-observable correlation matrix of the given dataset. Thereafter, we develop the closed-form probability of an edge of the RGG variable (in Section 2.6), and identify the inter-nodal distance of this graph, as a distance function in the probabilistic metric space in which the RGG is drawn. Thus, this distance function is shown to be a probability distribution over a positive support (Section 2.5). This RGG is shown to be an SRGG in Section 2.10, where the connection function of this SRGG is shown to be a function of our identified inter-nodal distance. We compute the expected degree distribution of our learnt RGG in Section 2.8 and subsequently, declare the host probabilistic metric space in which we draw the RGG. The technique for learning the RGG is delineated in Section 2.6. Then in Section 4 we present the Bayesian inference that is undertaken to learn the graph alone - when the correlation of the given dataset is known; else, the graph and this correlation are learnt simultaneously with a Metropolis-with-a-2-block-update. Learning of the RGG of a cuboidally-shaped data is discussed in Section 5. Thereafter, we present an illustration on real data - to learn the large network of diseases that aflict humans, correlated by phenotypes (in Section 6). Empirical illustration on simulated data is included in the Supplement, which includes demonstration of the efect of noise in the data on learning the RGG,

## 2. Preliminaries

In this section we provide the background on concepts that we use in the learning of the graph of a multivariate data set.

## 2.1. RGGs and SRGGs

We seek a graph variable defined on vertex set $V = \{ 1 , 2 , \ldots , p \}$ , with the random variable $X _ { i }$ attached to the i-th node, $\forall i \in V$ and $\forall X _ { i } \in \mathcal { X } \subseteq \mathbb { R }$ . Edges form independently in this graph and there are no self-loops. Let $G _ { i , j } = g _ { i , j } \in$ $\{ 0 , 1 \}$ be the edge variable between the i-th and jth nodes, $i , j \in [ M ]$ , where $[ M ] : = \{ i , j : i < j ; i , j \in V \}$

Definition 2.1. In an RGG, $G _ { i , j }$ ∼Poisson Point Process with its realisation $g _ { i , j } = 1 \iff D ( X _ { i } , X _ { j } ) < \tau , f o r$ r a chosen cutof $\tau \geq 0$ , where distance between the i-th and j-th nodes is $D ( X _ { i } , X _ { j } )$ , with $D : \mathcal { X } \times \mathcal { X } \longrightarrow \mathbb { R } _ { \ge 0 }$

We note that $G _ { i , j }$ is the binary random variable that we assign a value of 1 to, if the i-th and j-th nodes are connected by an edge in the graph; if the edge does not exist, then we set $G _ { i , j } = 0$

Definition 2.2. An SRGG defined on vertex set V is s.t. $\operatorname* { P r } ( G _ { i , j } = 1 )$ is given by the connection function $\phi ( d )$ , where distance function between the i-th and $j -$ th nodes is d. Thus, $\phi : \mathbb { R } _ { > 0 } \longrightarrow [ 0 , 1 ]$ is the connection function of this SRGG, $i . e .$ edge exists between the i-th and $j - t h$ nodes in this SRGG, with probability $\phi ( d )$

## 2.2. Partial correlation matrix

Let the given multivariate dataset D, comprise n observations of each of p observables $X _ { 1 } , X _ { 2 } , \ldots , X _ { p }$ , such that the i-th row of D is: $x _ { 1 } ^ { ( i ) } , x _ { 2 } ^ { ( i ) } , \ldots , x _ { p } ^ { ( i ) }$ , where $i \in \{ 1 , 2 , \ldots , n \}$ . We standardise data on the j-th observable, by the estimated mean and standard deviation of the sample $\{ \bar { x _ { j } ^ { ( 1 ) } } , x _ { j } ^ { ( 2 ) } , \dots , x _ { j } ^ { ( n ) } \}$ , s.t. the dataset comprising n values of each of the standardised observables, is $\mathbf { D } _ { S }$ . (Later in Section 5 we will discuss the general case of $X _ { j }$ a d-dimensional random variable, $\forall j \in \left\{ 1 , \ldots , p \right\} )$ ). Here $j \in \{ 1 , \ldots , p \}$ . Then the inter-observable correlation matrix of the data $\mathbf { D } _ { S }$ is $\Sigma _ { C } ^ { ( S ) } = [ c o r r ( X _ { i } , X _ { j } ) ]$

Definition 2.3. For the precision matrix $\Psi = \left[ \psi _ { i , j } \right] : = \left( \Sigma _ { C } ^ { ( S ) } \right) ^ { - 1 }$ , the matrix of absolute partial correlations is $R = [ \rho _ { i , j } ]$ , with

$$
\rho _ { i , j } = \Big | - \frac { \psi _ { i , j } } { \sqrt { \psi _ { i i } \psi _ { j j } } } \Big | , \quad i \neq j , \ a n d \ \rho _ { i i } = 1 \ f o r \ i = j .\tag{2.1}
$$

Here we want to learn the graph of dataset D, using the known absolute partial correlation $\rho _ { i , j }$ between $X _ { i }$ and $X _ { j } \ \forall i , j \in [ M ]$ . If however, we wish to learn the graph given the known (absolute) correlation $| c o r r ( X _ { i } , X _ { j } ) |$ between this variable pair - we only have to replace $\rho _ { i , j }$ in the discussion below, with |corr $( X _ { i } , X _ { j } ) | , \forall i , j \in [ M ]$ . Keeping this in mind, in the discussions below, we will include the possibilities of graph learning given the correlation or the partial correlation, by referring to “(partial) correlation” throughout, unless otherwise stated.

Below, the developed methodology considers that the (partial) correlation matrix of the considered dataset is known.

## 2.3. RGG drawn in probabilistic metric space: an SRGG

We construct an RGG variable in a probabilistic metric space. This renders any edge of this graph a probability distribution (over positive support). Then the inter-nodal distance is anticipated to be a distribution with positive support, and this distribution will be shown to be conditional on the observation that is the absolute partial correlation $\rho _ { i , j }$ between $X _ { i }$ and $X _ { j }$

To reflect this conditioning, we update the notation for distance between i-th and j-th nodes to $D ( X _ { i } , X _ { j } ; \rho _ { i , j } )$ , which we will sometimes abbreviate to $D _ { i , j }$ We learn edge $G _ { i , j }$ given the value $\rho _ { i , j }$ of the absolute partial correlation variable $R _ { i , j }$ , between $X _ { i }$ and $X _ { j } , i , j \in [ \bar { M } ]$ . Thus, $R _ { i , j } = \rho _ { i , j } \in [ 0 , 1 ]$

In this section we discuss the learning of the graph of dataset D, the (partial) correlation matrix of which we treat as known. As a supplement to this current agenda, later in Section 3, we will discuss how we can learn the correlation matrix of the given dataset by developing the closed-form posterior probability density of the correlation matrix.

In Section 2.10, we will show that by constructing the random graph as an RGG drawn in a probability metric space, the graph is rendered an SRGG. We will prove this using the identified distance function in the space in which we draw the graph, and learn using the (partial) correlation structure of the given multivariate dataset.

## 2.4. Disparity between two variables

We embed the graph variable in a probabilistic metric space (Menger, 1942), where in such a space, to any pair of points, we assign a probability distribution over positive support. Thus, to the i, j-th pair of nodes in the graph, we assign a probability distribution $F _ { S ( X _ { i } , X _ { j } ) } ( s )$ of the disparity variable $S ( X _ { i } , X _ { j } )$ that is introduced in Definition 2.4.

Definition 2.4. The disparity random variable is defined as

$$
S ( X _ { i } , X _ { j } ) \equiv S _ { i , j } : = | G _ { i , j } - R _ { i , j } | .
$$

In our work, disparity $S _ { i , j }$ measures the diference between the known (partial) correlation between $X _ { i } , ~ X _ { j }$ , and the connectedness of the i-th and j-th nodes of the graph, where the absolute partial correlation variable is $R _ { i , j }$ , and the binary edge variable $G _ { i , j }$ represents the connectedness between the i-th and j-th nodes. This holds $\forall i , j \in [ M ]$ . Then the disparity $S _ { i , j } = s \in [ 0 , 1 ]$ 2 $\forall i , j \in [ M ]$

Given the sparse information that we have on the behaviour of the disparity variable $S _ { i , j }$ , we recall (using its value $s : = | g _ { i , j } - \rho _ { i , j } | )$ that

if the given absolute value $\rho _ { i , j }$ of the (partial) correlation between $X _ { i }$ and $X _ { j }$ is high, (i.e. close to 1), then the probability for the edge to exist between the i-th and j-th nodes is high; probability for $G _ { i , j }$ to attain the value 0 then, is low. Thus, the probability for $S _ { i , j }$ to attain a low value is higher than for disparity to attain a higher value.

again, if the given $\rho _ { i , j }$ is close to 0, the probability for $G _ { i , j }$ to be 0 is high, while that for $G _ { i , j } = 1$ is low. Thus, again, we find that probability of $S _ { i , j }$ at a low s is higher than that at a higher s.

Using this intuition, we model the probability density of $S _ { i , j }$ as a decreasing function of $s \in \ [ 0 , 1 ]$ . However, the scale with which this density decreases with s, is an unknown, i.e. this scale is a real random variable. We denote such a squared scale, (or a variance variable) to be $v _ { i , j }$ . There is no available observation to further constrain the model of the joint density of $S _ { i , j }$ and $\begin{array} { r } { \boldsymbol { v } _ { i , j } , } \end{array}$ than what we have motivated above. Hence we model this joint density of the disparity variable $S _ { i , j }$ and the variance variable $v _ { i , j }$ , at $S _ { i , j } = s$ and $v _ { i , j } = v$ as:

$$
\begin{array} { l l l } { f _ { S _ { i , j } , v _ { i , j } } ( s , v ) } & { = } & { \displaystyle K \frac { 1 } { \sqrt { 2 \pi v } } \exp { \left( - \frac { s ^ { 2 } } { 2 v } \right) } , \mathrm { ~ i f ~ } s \in [ 0 , 1 ] ; v \in \mathbb { R } _ { > 0 } } \\ & { = } & { \mathrm { ~ 0 ~ o t h e r w i s e } , } \end{array}
$$

$\forall i , j \in [ M ]$ , where $K > 0$ is a constant that we compute below.

Also, lack of further information motivates our assumption that the prior on the variance variable $v _ { i , j }$ is Uniform[0,1]. The above holds $\forall i , j \in [ M ]$ . For other models of this joint density, Bayesian inference on the edge variable is likely to converge in mean.

Keeping in mind our ulterior aim of learning the edge given the known interobservable correlation, we will need to marginalise out the variance variable from this modelled joint density of variance and disparity, (where the latter disparity bears information on the sought edge).

Since we assume $v \sim U n i f o r m [ 0 , 1 ]$ , the marginal density of the disparity

variable $S _ { i , j }$ is

$$
\begin{array} { l c l } { { f _ { S _ { i , j } } ( s ) } } & { { = } } & { { \displaystyle K \int _ { 0 } ^ { 1 } \frac { 1 } { \sqrt { 2 \pi \upsilon } } \exp \left( - \frac { s ^ { 2 } } { 2 \upsilon } \right) d \upsilon , s \in [ 0 , 1 ] , } } \\ { { } } & { { = } } & { { \displaystyle 0 \mathrm { o t h e r w i s e } , } } \end{array}
$$

$\forall i , j \in [ M ]$ , where $K > 0$ is a global scale. Then given that $s \in [ 0 , 1 ]$ , this marginalisation over the variance parameter gives

$$
\begin{array} { l l l } { { f _ { S _ { i , j } } ( s ) } } & { { = } } & { { \displaystyle \frac { K s \Gamma \left( - \frac 1 2 , \frac { s ^ { 2 } } 2 \right) } { 2 \sqrt \pi } } } \\ { { } } & { { = } } & { { K \left[ \sqrt { \displaystyle \frac { 2 } \pi } \exp \left( - \frac { s ^ { 2 } } 2 \right) + s \mathrm { e r f } \left( \displaystyle \frac { s } { \sqrt 2 } \right) - s \right] , } } \end{array}\tag{2.2}
$$

for $s \in [ 0 , 1 ] ;$ ; else $f _ { S _ { i , j } } ( s ) = 0$ . This holds $\forall i , j \in [ M ]$

Since $f _ { S } ( s )$ is a density, the constant K is s.t

$$
\begin{array} { r c l } { { { \cal K } } } & { { = } } & { { \displaystyle \left[ \int _ { 0 } ^ { 1 } \left( \sqrt { \frac { 2 } { \pi } } \exp \left( - \frac { s ^ { 2 } } { 2 } \right) + s \mathrm { e r f } \left( \frac { s } { \sqrt 2 } \right) - s \right) d s \right] ^ { - 1 } } } \\ { { } } & { { = } } & { { \displaystyle \left[ \mathrm { e r f } \left( \frac { 1 } { \sqrt 2 } \right) - \frac { 1 } { 2 } + \frac { 1 } { \sqrt { 2 \pi e } } \right] ^ { - 1 } } } \end{array}\tag{2.3}
$$

We will show below that when the absolute partial correlation between $X _ { i }$ and $X _ { j }$ is known to be $\rho _ { i , j }$ , the distance between the i-th and j-th points in this host space is $D ( X _ { i } , X _ { j } ; \tilde { \rho _ { i , j } } ) = F _ { S ( X _ { i } , X _ { j } ) } ( s )$ , which is the cdf of the disparity computed at a value $s : = | g _ { i , j } - \rho _ { i , j } |$ . We will compute said cdf by invoking the aforementioned pdf of the disparity presented in Equation 2.2. Then, given the multivariate dataset (that leads to the matrix $\boldsymbol { R } = [ \rho _ { i , j } ]$ of absolute inter-observable (partial) correlations), the learnt random graph variable is ${ \mathcal { G } } _ { S , V } ( R , \tau )$ that is defined on vertex set V s.t. the edge exists between the i-th and j-th nodes $\iff D ( X _ { i } , X _ { j } ; \rho _ { i , j } ) = F _ { S ( X _ { i } , X _ { j } ) } ( s ) \leq \tau$ , for a chosen cutof probability τ .

## 2.5. cdf of $S _ { i , j }$ and distance $D ( X _ { i } , X _ { j } ; \rho _ { i , j } )$

Definition 2.5. The density $f _ { S _ { i , j } } ( s )$ of $S _ { i , j }$ (in Equation $\it { 2 . 2 ) }$ is used to write the cdf $F _ { S _ { i , j } } ( s )$ of $S _ { i , j }$ , computed at the disparity value s:

$$
\begin{array} { l } { { F _ { S _ { i , j } } ( s ) = K \left[ \displaystyle \int _ { 0 } ^ { s } \left( \sqrt { \frac { 2 } { \pi } } \exp \left( - \frac { u ^ { 2 } } { 2 } \right) + u \mathrm { e r f } \left( \frac { u } { \sqrt { 2 } } \right) - u \right) d u \right] } } \\ { { = K \left[ \displaystyle \frac { \left( s ^ { 2 } + 1 \right) } { 2 } \mathrm { e r f } \left( \displaystyle \frac { s } { \sqrt { 2 } } \right) - \displaystyle \frac { s ^ { 2 } } { 2 } + \displaystyle \frac { s } { \sqrt { 2 \pi } } \exp ( - s ^ { 2 } / 2 ) \right] , } } \end{array}\tag{2.4}
$$

where K is defined in Equation 2.3. This holds $\forall i , j \in [ M ]$

Remark 2.1. We see from Equation 2.4 that $F _ { S _ { i , j } } ( 0 ) = 0$ and $F _ { S _ { i , j } } ( 1 ) =$ 1, given the definition of K in Equation 2.3. Also, $F _ { S _ { i , j } } ( s )$ is monotonically increasing with $s \in [ 0 , 1 ] ,$ continuous in $[ 0 , 1 ] ;$ and one-to-one in $[ 0 , 1 ]$ . This holds $\forall i , j , \in [ M ]$

Next we check if $D ( X _ { i } , X _ { j } ; \rho _ { i , j } ) = F _ { S _ { i , j } } ( s )$ is a distance function.

Proposition 2.1. $D ( X _ { i } , X _ { j } ; \rho _ { i , j } ) = F _ { S _ { i , j } } ( s )$ for $s = | g _ { i , j } - \rho _ { i , j } |$ , is the distance between the i-th and $j - t h$ nodes, given that the absolute partial correlation between $X _ { i }$ and $X _ { j }$ is $\rho _ { i , j }$ . Here $F _ { S _ { i , j } } ( s ) = K \left[ ( ( s ^ { 2 } + 1 ) / 2 ) \right]$ erf $( s / \sqrt { 2 } ) - s ^ { 2 } / 2$ $+ ( s / \sqrt { 2 \pi } ) \exp \bigl ( - s ^ { 2 } / 2 \bigr ) \bigr ]$

Proof. We condense the notation $D ( X _ { i } , X _ { j } ; \rho _ { i , j } )$ to $D _ { i , j }$

We prove that $D _ { i , j }$ given by $F _ { S _ { i , j } } ( s )$ that is stated in Equation 2.4 is a distance function by proving that it is non-negative; symmetric; is $0 \quad \Longleftrightarrow$ $X _ { i } = X _ { j } ;$ ; and obeys the triangle rule.

— Non-negative: $D _ { i , j }$ is non-negative. This follows since $D _ { i , j } = F _ { S _ { i , j } } ( s )$ is a $c d f , \forall s \in [ 0 , 1 ] , \forall i , j \in [ M ]$

— Symmetric: $D _ { i , j } = D _ { j , i }$ by definition of $S _ { i , j } : = | G _ { i , j } - \rho _ { i , j } | .$

$D _ { i , j } = F _ { S _ { i , j } } ( s ) = 0 { \mathrm { ~ i f ~ } } s = 0$ , from Equation $2 . 4 , \forall i , j \in [ M ]$

Again, if $D ( s ) = F _ { S _ { i . i } } ( s ) = 0$ , it implies $s = 0$ since $F _ { S _ { i , j } } ( \cdot )$ is one-to-one and monotonic increasing in [0,1]. Therefore, $D ( S _ { i , j } = s ) { \widetilde { } = } 0 \iff s = 0$ $\forall i , j \in [ M ]$

$\{ S _ { i , j } \} _ { i \leq j ; i , j \in V }$ is a set comprising identically distributed variables, each distributed as $F _ { S _ { i , j } } ( \cdot )$ as given in Equation 2.4, $\forall i , j \in [ M ]$ . Then for $D _ { i , j } \ = \ F _ { S _ { i , j } } ( s )$ , the triangle rule trivially holds $\forall s \in [ 0 , 1 ] \colon F _ { S _ { i , j } } ( s ) \ : +$ $F _ { S _ { j , k } } ( s ) \geq { \overline { { F } } } _ { S _ { i , k } } ( s )$ i.e. $D _ { i , j } + D _ { j , k } \geq D _ { i , k } , \forall i , j \in [ M ] , j < k ; j , k \in$ $\{ 1 , \ldots , p \} ; i < k ; i , k \in \{ 1 , \ldots , p \}$

Thus, in the RGG drawn in a probabilistic metric space, the distance function $D ( \cdot , \cdot ; \cdot )$ is the probability distribution of the disparity variable, as defined in Definition 2.5.

## 2.6. Learning the RGG at the given (partial) correlation matrix & probability of the graph

Since we develop the graph learning in the information paradigm that $R = [ \rho _ { i , j } ]$ is known, the disparity variable between $X _ { i }$ and $X _ { j }$ reduces to $S ( X _ { i } , X _ { j } ) =$ $| G _ { i , j } - \rho _ { i , j } | , \forall i , j \in [ M ]$

At the given $\rho _ { i , j }$ , the distance $D _ { i , j }$ between the i-th and j-th nodes is $F _ { S _ { i , j } } ( s )$ with $s = | g _ { i , j } - \rho _ { i , j } |$ |, for $g _ { i , j } \in \{ 0 , 1 \} , \forall i , j \in [ M ]$

Remark 2.2. In the RGG, we expect that $D _ { i , j } < \tau _ { , }$ , would imply $G _ { i , j } ~ = ~ 1$ However, this assignment of a value to $G _ { i , j }$ then raises a concern: there appears a circularity in our assignment of the value $g _ { i , j }$ to the edge variable $G _ { i , j }$ , using the $g _ { i , j }$ -dependent inter-nodal distance $D _ { i , j }$

We avoid any such circularity, by stating that if at the given $\rho _ { i , j }$

$$
D _ { i , j } = F _ { S _ { i , j } } ( s = | g _ { i , j } - \rho _ { i , j } | ) < \tau \implies G _ { i , j } = g _ { i , j } ; ~ e l s e ~ G _ { i , j } = 1 - g _ { i , j }
$$

in the learnt RGG, $\forall i , j \in [ M ]$

Thus, in the RGG ${ \mathcal { G } } _ { S , V } ( R , \tau )$ , we will compute $D _ { i , j } = F _ { S _ { i , j } } ( s = 1 - \rho _ { i , j } )$ and check if this computed distance falls short of a chosen τ, or not. If it does, then the edge between the i-th and $j \mathrm { - t h }$ nodes exists in the RGG; otherwise it does not. At the same time, if this edge exists, we learn the probability with which this edge exists. This then reflects the SRGG signature of the random graph that we learn, in addition to the graph being an RGG (drawn in a probabilstic space). In order to identify the probability with which an edge exists, the following is undertaken.

At the given $\rho _ { i , j }$ , we use Rejection Sampling to learn the edge variable that is straddled by the i-th and j-th nodes of the sought RGG. Such sampling will need to be done from the probability of the edge variable $G _ { i , j }$ given the known value $\rho _ { i , j }$ of the variable $R _ { i , j }$ that represents the absolute partial correlation between $X _ { i }$ and $X _ { j }$ . Here,

$$
f _ { S _ { i , j } } ( s = | g _ { i , j } - \rho _ { i , j } | ) = \mathrm { P r } ( G _ { i , j } = g _ { i , j } | \rho _ { i , j } ) f _ { R _ { i , j } } ( \rho _ { i , j } ) / \int _ { u = 0 } ^ { 1 } f _ { R _ { i , j } } ( u ) d u ,
$$

which reduces to $f _ { S _ { i , j } } ( s = | g _ { i , j } - \rho _ { i , j } | ) = C _ { i , j } \operatorname* { P r } ( G _ { i , j } = g _ { i , j } | \rho _ { i , j } )$ , given that the density of $R _ { i , j }$ is known, and $\rho _ { i , j }$ is known, s.t. $C _ { i , j } : = f _ { R _ { i , j } } ( \rho _ { i , j } ) / [ \int _ { u = 0 } ^ { 1 } f _ { R _ { i , j } } ( u ) d u ]$ is a known constant. Then summing over all values of $G _ { i , j }$ , at the known absolute (partial) correlation $\rho _ { i , j }$ between $X _ { i }$ and $X _ { j }$ , the constant $C _ { i , j } = K [ ( 1 -$ $\rho _ { i , j } )$ erf $\left( ( 1 - \rho _ { i , j } ) / \sqrt { 2 } \right) - ( 1 - \rho _ { i , j } ) + \sqrt { 2 / \pi } \exp ( - ( 1 - \rho _ { i , j } ) ^ { 2 } / 2 ) + \rho _ { i , j } \mathrm { e r f } \left( \rho _ { i , j } / \sqrt { 2 } \right) -$ $\rho _ { i , j } + \sqrt { 2 / \pi } \exp ( - ( \rho _ { i , j } ) ^ { 2 } / 2 ) ]$ where K is defined in Equation 2.3. Thus, $C _ { i , j }$ is known, and hence, $\operatorname* { P r } ( G _ { i , j } = g _ { i , j } | \rho _ { i , j } )$ is known in a closed-form $\mathrm { w a y } , \forall i , j \in [ M ]$

We will draw $n _ { i , j }$ number of samples of the edge variable $G _ { i , j }$ from its probability mass function conditional on the known $\rho _ { i , j }$ . But $\mathrm { P r } ( G _ { i , j } = g _ { i , j } | \rho _ { i , j } ) =$ $f _ { S _ { i , j } } ( s ) / C _ { i , j } = K [ | g _ { i , j } - \rho _ { i , j } | \operatorname { e r f } \left( | g _ { i , j } - \rho _ { i , j } | / \sqrt { 2 } \right) - | g _ { i , j } - \rho _ { i , j } | + \sqrt { 2 / \pi } \exp ( - ( g _ { i , j } - \rho _ { i , j } ) / \sqrt { 2 } ) ]$ $\rho _ { i , j } \bar { ) } ^ { 2 } / 2 ) ] / C _ { i , j }$

It follows that $\begin{array} { r } { \operatorname* { P r } ( G _ { i , j } = g _ { i , j } | \rho _ { i , j } ) = f _ { S _ { i , j } } ( s ) / C _ { i , j } \leq 1 \implies f _ { S _ { i , j } } ( s ) / K \leq } \end{array}$ $C _ { i , j } / K$ , for the positive K. In fact, definitions of $f _ { S _ { i , j } } ( \cdot )$ , K and $C _ { i , j }$ show that $f _ { S _ { i , j } } ( s ) / K < C _ { i , j } / K \forall \rho _ { i , j } \in [ 0 , 1 ] , \forall i , j \in [ M ]$

Hence, in our implementation of Rejection Sampling, we draw $n _ { i , j }$ samples of $G _ { i , j }$ from the proposal density that we choose to be the Bernoulli(0.5) density, and scale this by the constant $C _ { i , j } / K$ , s.t. $C _ { i , j } / K$ envelopes the target $f _ { S _ { i , j } } ( s ) / K = ( | g _ { i , j } - \rho _ { i , j } | \mathrm { e r f } \left( | g _ { i , j } - \rho _ { i , j } | / \sqrt { 2 } \right) - | g _ { i , j } - \rho _ { i , j } | + \sqrt { 2 / \pi } \exp ( - ( g _ { i , j } - \rho _ { i , j } ) / \sqrt { 2 } ) ) ,$ $\rho _ { i , j } ) ^ { 2 } / 2 ) ) / K$

We accept the r-th proposed sample $g _ { i , j } ^ { ( \star ) } \mathrm { ~ i f ~ } f _ { S _ { i , j } } ( | g _ { i , j } ^ { ( \star ) } - \rho _ { i , j } | ) / K \geq ( C _ { i , j } / K ) u$ i.e. if $| g _ { i , j } ^ { ( \star ) } - \rho _ { i , j } | \mathrm { e r f } \left( | g _ { i , j } ^ { ( \star ) } - \rho _ { i , j } | / \sqrt { 2 } \right) - | g _ { i , j } ^ { ( \star ) } - \rho _ { i , j } | + \sqrt { 2 / \pi } \exp ( - ( g _ { i , j } ^ { ( \star ) } - \rho _ { i , j } ) ^ { 2 } / 2 ) \geq 1$ $C _ { i , j } u$ , where u is the value of $U \stackrel { \cdot } { \sim } U n i f o r m [ 0 , 1 ]$ . Then the r-th sample is denoted $g _ { i , j } ^ { ( r ) } = g _ { i , j } ^ { ( \star ) }$ . If on the other hand, $g _ { i , j } ^ { ( \star ) }$ is not accepted, then we set $g _ { i , j } ^ { ( r ) } = 1 - g _ { i , j } ^ { ( \star ) }$

We collate these samples into the set $P _ { i , j } ^ { ( R ) } : = \{ g _ { i , j } ^ { ( 1 ) } , g _ { i , j } ^ { ( 2 ) } , . . . , g _ { i , j } ^ { ( n _ { i , j } ) } \}$ , to define the relative frequency of $G _ { i , j } { = } 1$ in this set $P _ { i , j } ^ { ( R ) }$ as:

$$
w _ { i , j } : = \frac { \displaystyle \sum _ { k = 1 } ^ { n _ { i , j } } g _ { i , j } ^ { ( k ) } } { n _ { i , j } } .
$$

Typically $n _ { i , j } = n , \ \forall i , j \in [ M ]$ and we have used $n \sim 1 0 ^ { 3 }$ to $1 0 ^ { 4 }$ in undertaken applications. Thus, the sampling of $g _ { i , j }$ at the known $\rho _ { i , j }$ is undertaken using Rejection Sampling. The algorithm used to undertake the sampling is presented in Section 4.

Remark 2.3. We note that the relative frequency $w _ { i , j }$ of achieving $G _ { i , j } = 1$ in our sample generated via Rejection Sampling using $f _ { S _ { i , j } } ( s )$ , is an approximation of $\operatorname* { P r } ( G _ { i , j } = 1 | \rho _ { i , j } )$ , i.e.

$$
\begin{array} { r } { w _ { i , j } \approx \operatorname* { P r } ( G _ { i , j } = 1 | \rho _ { i , j } ) . } \end{array}
$$

Then at the chosen $\tau \in [ 0 , 1 ]$ , if $F _ { S _ { i , j } } ( s = 1 - \rho _ { i , j } ) < \tau , G _ { i , j } = 1$ , and this edge between the i-th and j-th nodes exists in ${ \mathcal { G } } _ { S , V } ( R , \tau )$ , with probability $w _ { i , j }$ If $F _ { S _ { i , j } } ( s = 1 - \rho _ { i , j } ) \geq \tau$ , then the edge between the i-th and $j \mathrm { - t h }$ nodes exists with probability 0, i.e. $G _ { i , j } = 0$ then. This holds $\forall i , j \in [ M ]$

Thus, we notice that it is possible for us to learn a random graph with the known (partial) correlation matrix R of the dataset D, without imposing any thresholding on any edge, i.e. at $\tau = 1 . \mathrm { A t }$ the same time, our graph learning can be undertaken while acknowledging the geometric-ness of the graph. In particular, we identify a data-driven optimal τ in an application given a real dataset (Section 7).

To compute the adjacency matrix of the RGG ${ \mathcal { G } } _ { S , V } ( R , \tau )$ , we first compute the indicator function

$$
\mathbf { 1 } _ { H ^ { ( \tau ) } } \big ( F _ { S _ { i , j } } ( 1 - \rho _ { i , j } ) \big ) \big ) , \mathrm { ~ w h e r e ~ } H ^ { ( \tau ) } : = \{ \delta : \delta < \tau , \delta \in [ 0 , 1 ] \} ,\tag{2.5}
$$

at a chosen $\tau \in [ 0 , 1 ]$ . Then the adjacency matrix of the learnt RGG is

$$
\mathbf { A } = [ w _ { i , j } \ \mathbf { 1 } _ { H ^ { ( \tau ) } } ( F _ { S _ { i , j } } ( 1 - \rho _ { i , j } ) ) ] .\tag{2.6}
$$

We define the posterior probability of the RGG of the given dataset D, using the probability of the edges that are accepted at a chosen τ .

Definition 2.6. Recalling that edges form independently in the RGG ${ \mathcal { G } } _ { S , V } ( R , \tau )$ that we seek of the dataset D, and that there are no self-loops in this RGG, the posterior probability of this RGG variable, given the known partial correlation

matrix $R = [ \rho _ { i , j } ]$ of this dataset, is:

$$
\begin{array} { r c l } { \pi ( \mathcal { G } _ { S , V } ( R , \tau ) | R ) } & { = } & { \displaystyle \prod _ { i , j \in [ M ] } \operatorname* { P r } ( G _ { i , j } = 1 | \rho _ { i , j } ) \mathbf { 1 } _ { H ^ { ( \tau ) } } ( F _ { S _ { i , j } } ( 1 - \rho _ { i , j } ) ) } \\ & & { \approx } & { \displaystyle \prod _ { i , j \in [ M ] } w _ { i , j } \mathbf { 1 } _ { H ^ { ( \tau ) } } ( F _ { S _ { i , j } } ( 1 - \rho _ { i , j } ) ) } \\ & & {  } & {  _ { i , j \in [ M ] } } \end{array}\tag{2.7}
$$

where the set ${ \pmb { H } } ^ { ( \tau ) }$ is defined in Equation 2.5.

## 2.7. Probability metric space that the RGG is drawn in

Definition 2.7. In our work, the triple $\{ \boldsymbol { x } , D ( \cdot ) , \Delta \}$ is the probabilistic metric space that the RGG ${ \mathcal { G } } _ { S , V } ( R , \tau )$ is drawn in.

Here the distance function

$$
D ( X _ { i } , X _ { j } ; \rho _ { i , j } ) \equiv D _ { i , j } = F _ { S _ { i , j } } ( s )
$$

is defined in Proposition 2.1 for the cdf $F _ { S _ { i , j } } ( \cdot ) \in \mathcal { F } _ { + } , \forall X _ { i } , X _ { j } \in \mathcal { X }$ , where ${ \mathcal { F } } _ { + }$ is the set of distributions over positive support.

Lastly, for a given $s \in \ [ 0 , 1 ]$ , a triangle function $\Delta$ can be defined as the binary operation:

$$
\Delta ( F _ { S _ { j , k } } ( s ) , F _ { S _ { i , k } } ( s ) ) : = F _ { S _ { j , k } } ( s ) + F _ { S _ { i , k } } ( s ) \geq F _ { S _ { k , j } } ( s ) ,
$$

since $\{ S _ { a , b } \} _ { a , b \in [ M ] }$ is a set of identically distributed random variables. Hence, $\Delta ( \cdot , \cdot )$ is commutative, associative and takes $F _ { S _ { i , i } } ( \cdot ) = 0$ as its identity. Here $S _ { i , j } \equiv S ( X _ { i } , X _ { j } )$ and the above holds $\forall X _ { i } , X _ { j } , X _ { k } \in X . ~ i , j , k \in V$

## 2.8. Degree distribution

Vertex set of RGG ${ \mathcal { G } } _ { S , R } ( V , \tau )$ is $V = \{ 1 , \ldots , p \}$ . We assume that the density at point x is $f ( x )$ . For any $i \in V$ , define (open or close, but bound) Borel measurable ball $B _ { i , a } ,$ centred at the i-th node, with radius a, where $a \in [ 0 , 1 ]$ ， given that radius of a ball in the space of the RGG, is a probability.

Let random variable $N ( B _ { i , a } )$ be the number of elements of vertex set V that lie inside $B _ { i , a } ,$ as connected to the i-th node at the centre of $B _ { i , a }$ . We recall that if $D _ { i , j } = F _ { S _ { i , j } } ( s = 1 - \rho _ { i , j } ) = K [ ( ( ( 1 - \rho _ { i , j } ) ^ { 2 } + 1 ) / 2 ) \operatorname { e r f } \left( ( 1 - \rho _ { i , j } ) / { \sqrt { 2 } } \right) - \sum _ { j = 1 } ^ { J } ( ( 1 - \rho _ { i , j } ) ^ { 2 } + 1 ) ]$ $( 1 - \rho _ { i , j } ) ^ { 2 } / 2 + \left( ( 1 - \rho _ { i , j } ) / \sqrt { 2 \pi } \right) \exp ( - ( 1 - \rho _ { i , j } ) ^ { 2 } / 2 ) ] < \tau , G _ { i , j } = 1$ in our RGG, at the chosen cutof τ. At the given partial correlation matrix R (of the dataset for which this RGG is learnt), the number density of edges that will form between the i-th node at the centre of the ball $B _ { i , a }$ , and a point within distance $x \leq a$ within this ball, is

$$
n _ { i } ^ { ( \pmb { R } ) } ( x ) : = \sum _ { j \in C _ { i } ( x ) } f ( x ) H ( \tau - D _ { i , j } ) , \mathrm { ~ w h e r e ~ }
$$

$$
C _ { i } ( x ) : = \{ j : j \neq i , j \in V , D _ { i , j } \leq x \} .
$$

Here $H ( \cdot , \cdot )$ is a Heaviside function.

Then the expectation of the degree of the i-th node in the leant RGG is:

$$
\begin{array} { r c l } { { \mathbb { E } [ N _ { i } ^ { ( { \pmb R } ) } ] } } & { { = } } & { { \displaystyle \int _ { x = 0 } ^ { a } d x ~ 2 \pi x n _ { i } ^ { ( { \pmb R } ) } ( x ) } } \\ { { } } & { { \equiv } } & { { \lambda _ { a , \tau } ^ { ( i ) } ( { \pmb R } ) \pi a ^ { 2 } } } \end{array}\tag{2.8}
$$

where

$$
\lambda _ { \tau } ^ { ( i ) } ( R ) : = \frac { 2 \int _ { x = 0 } ^ { a } d x \ x n _ { i } ^ { ( R ) } ( x ) } { a ^ { 2 } } ,
$$

s.t.

$$
\mathbb { E } [ N ( B _ { i , a } ) ] = \lambda _ { \tau } ^ { ( i ) } ( { \pmb R } ) \pi a ^ { 2 }
$$

for the given a. $\mathrm { ~ I f ~ } \tau < a$ , there is no contribution to the last integral from $x \in [ \tau , a ] ;$ ; else, the expectation is contributed to by all $x \in [ 0 , a ]$

Thus, the number density of edges within a bound region of radius $^ { a , }$ centred at point $X _ { i }$ in X, is given as the location-dependent $\lambda _ { \tau } ^ { ( i ) } ( R )$ for a given partial correlation matrix $R ,$ at a chosen threshold cutof probability τ.

Even if the density $f ( \cdot )$ is homogeneous, the RGG variable is generated by an inhomogeneous point process. We note that the local intensity of the generative process is a function of the threshold $\tau _ { : }$ , and the correlation matrix of the dataset for which the RGG is learnt.

For a given $X _ { i } ,$ if $X _ { j }$ and $X _ { j ^ { \prime } }$ are s.t. disparity $S _ { i , j } = s$ is less than $S _ { i , j ^ { \prime } } = s ^ { \prime } { . }$ cdf $F _ { S _ { i , j } } ( s )$ is lower than $F _ { S _ { i , i } ^ { \prime } } \bar { ( } s ^ { \prime } )$ . Then $D _ { i , j } < D _ { i , j ^ { \prime } } ,$ s.t. the i-th node is more likely to be connected to the j-th node, than to the ${ j ^ { \prime } } \mathrm { - t h }$ node, at a given $\tau .$ Thus, variables with low values of mutual disparity, could form clusters within the RGG of a the given dataset.

Remark 2.4. $| C _ { i } ( x ) |$ is approximately the same for all points that lie in the bulk of such a cluster - as is $f ( x )$ . Thus, nodes that lie in the bulk of a cluster will share similar values of their degrees. On the other hand, for the i-th node that lies near the edge of a cluster, $| C _ { i } ( x ) |$ is lower than if this node were in the bulk. Then, in general, the degree of a node near the edge of a cluster is lower than that in its bulk. Within a “small” distance inwards from the edge of a cluster, the degree distribution is likely to show an increasing trend, but how quickly this trend shows up, i.e. quantification of the aforesaid “small”ness, is afected by the choice of the cutof τ used in the learning of the RGG of a given dataset.

## 2.9. Choosing the cutof probability τ

Since the learnt graph ${ \mathcal { G } } _ { S , V } ( R , \tau )$ is a random graph-valued variable, we can define its posterior probability $\pi ( \mathcal { G } s , \pmb { V } ( \pmb { R } , \tau ) | \pmb { R } )$ given the partial inter-observable correlation matrix R of the given dataset. This then allows for an organic identification of the optimal cutof, at which the most-robust RGG is realised, where said robustness is to changes in τ: the τ at which the rate of change of the RGG posterior is minimised, i.e. $d ( \pi ( \mathcal { G } _ { S , V } ( \pmb { R } , \tau ) | \pmb { R } ) ) / d \tau$ is minimised, is the optimal cutof that produces the most-robust RGG. Since we typically compute logarithm of the posterior probability of the RGG, the optimal τ is implemented as the cutof, at which the rate of change (with τ) of the logarithm of this posterior probability is minimised.

## 2.10. RGG in probabilistic metric space is an SRGG

We draw an RGG in the probabilistic metric space, and an edge is accepted if the inter-nodal distance falls below a chosen cutof probability τ, where the accepted edge exists with an identified probability.

Proposition 2.2. The RGG ${ \mathcal { G } } _ { S , V } ( R , \tau )$ , drawn in a probabilistic metric space is an SRGG.

Proof. At the known partial correlation matrix ${ \bf { \mathit { R } } } = [ \rho _ { i , j } ]$ , and for $s = | g _ { i , j } - \rrangle$ $\rho _ { i , j } |$ , sampling $G _ { i , j }$ from $f _ { S _ { i , j } } ( s ) , n _ { i , j }$ times, produces the set

$$
P _ { i , j } ^ { ( R } = \{ g _ { i , j } ^ { ( t ) } \} _ { t = 1 } ^ { n _ { i , j } } ,
$$

with $\begin{array} { r } { w _ { i , j } : = \sum _ { t = 1 } ^ { n _ { i , j } } g _ { i , j } ^ { ( t ) } / n _ { i , j } } \end{array}$ . Then edge probability $\mathrm { P r } ( G _ { i , j } = 1 | \rho _ { i , j } ) \approx w _ { i , j } ,$ $\forall i , j , \in [ M ]$

We have seen above that inter-nodal distance $D _ { i , j } \equiv F _ { S _ { i , j } } ( s = 1 - \rho _ { i , j } ) <$ $\tau \implies G _ { i , j } = 1 ;$ ; else $G _ { i , j } = 0$

Then we define the function $\phi : [ 0 , 1 ] \longrightarrow [ 0 , 1 ]$ s.t. at a chosen $\tau \in [ 0 , 1 ]$ , the edge between the i-th and j-th nodes exists with probability

$$
\begin{array} { r } { \phi ( D _ { i , j } ) : = w _ { i , j } \mathbf { 1 } _ { H ^ { ( \tau ) } } ( D _ { i , j } ) , } \end{array}
$$

where

$$
\begin{array} { r } { \pmb { H } ^ { ( \tau ) } : = \{ \delta : \delta < \tau , \delta \in [ 0 , 1 ] \} . } \end{array}
$$

This holds $\forall i , j \in [ M ]$

Thus, in this RGG variable, we have identified the probability with which an edge exists, where said probability is the function $\phi ( \cdot )$ of the distance between the nodes that this edge connects.

An SRGG is a random graph in which the probability with which an edge exists, is a function of the inter-nodal distance.

Hence this RGG drawn in a probabilistic metric space, is an SRGG. □

## 3. Posterior density of the inter-observable correlation matrix

In this section we present a closed-form posterior probability density of the inter-observable correlation matrix, where n observations of each of p observables $X _ { 1 } , X _ { 2 } , \ldots , X _ { p }$ comprise the dataset. Later in Section 5, we discuss learning the RGG for vector-valued observables. We standardise the data on the variable $X _ { i }$ using the mean and standard deviation of the sample comprising the n observations, i.e. the sample $\{ x _ { i } ^ { ( 1 ) } , x _ { i } ^ { ( 2 ) } , \ldots , x _ { i } ^ { ( n ) } \}$ . Such standardisation is undertaken for $\forall i \in \left\{ 1 , \ldots , p \right\}$ . While values of $X _ { 1 } , \ldots , X _ { p }$ comprise the dataset D, data on each of p standardised observables $Z _ { 1 } , \ldots , Z _ { p } ,$ comprise the dataset $\mathbf { D } _ { s }$ . The observable $X _ { m }$ with standardised components, is denoted $Z _ { m }$ , s.t. we can define vector $\pmb { Z } _ { m } = \bigl ( Z _ { m } ^ { ( 1 ) } , Z _ { m } ^ { ( 2 ) } , \ldots , Z _ { m } ^ { ( p ) } \bigr ) ^ { T } , \forall m \in \{ 1 , \ldots , n \}$

Following earlier work, (Gruber and West, 2016; Ni et al., 2017; Wang and West, 2009), we model the vector-valued observable output at any observational instance, with a vector-valued Gaussian Process. This implies that the joint density of the n realisations of this observable, namely $Z _ { 1 } , \ldots , Z _ { n } $ is a matrix Normal density $\mathcal { M N } ( \mathbf { 0 } , \boldsymbol { \Sigma } _ { m } ^ { ( S ) } , \boldsymbol { \Sigma } _ { C } ^ { ( S ) } )$ . Here, the undertaken standardisation implies that the mean matrix of this density is a null matrix, while its two covariance matrices are rendered correlation matrices. These two correlation matrices are: the d × d-dimensional inter-observable matrix $\pmb { \Sigma } _ { m } ^ { ( S ) }$ , the $m , m ^ { \prime } .$ -th element of which is the correlation between $Z _ { m }$ and $Z _ { m ^ { \prime } } , \bar { \forall } { \bar { m } } , m ^ { \prime } \in \{ 1 , \dots , n \}$ ; and the $p \times p$ -dimensional inter-variable correlation matrix $\Sigma _ { C } ^ { ( S ) }$ , the i, j-th element of which is the correlation between $Z _ { i }$ and $Z _ { j } , \forall i , j \in \lbrace 1 , \ldots , p \rbrace$

Theorem 3.1. When the prior on $\Sigma _ { C } ^ { ( S ) }$ is Uniform, the joint posterior probability density of the correlation matrices $\Sigma _ { C } ^ { ( S ) }$ and $\pmb { \Sigma } _ { m } ^ { ( S ) }$ , given the standardised data $\mathbf { D } _ { S }$ , can be marginalised over all $\pmb { \Sigma } _ { m } ^ { ( S ) }$ , to yield the marginal posterior pdf of the inter-observable correlation matrix given data $\mathbf { D } _ { S }$ , as:

$$
\pi ( \Sigma _ { C } ^ { ( S ) } | \mathbf { D } _ { S } ) \propto \Big | \Sigma _ { C } ^ { ( S ) } \Big | ^ { - p / 2 } \Big | \mathbf { D } _ { S } ( \Sigma _ { C } ^ { ( S ) } ) ^ { - 1 } ( \mathbf { D } _ { S } ) ^ { T } \Big | ^ { - \frac { m + 1 } { 2 } } ,
$$

where prior on $\pmb { \Sigma } _ { m } ^ { ( S ) }$ is non-informative: $\pi _ { 0 } ( \Sigma _ { m } ^ { ( S ) } ) = \left| \Sigma _ { m } ^ { ( S ) } \right| ^ { \alpha }$ , with $\alpha = - \frac { d } { 2 } - 1$ and $\Sigma _ { C } ^ { ( S ) }$ is assumed invertible.

The proof follows from marginalising over all $\pmb { \Sigma } _ { m } ^ { ( S ) }$ , using the uninformative prior on this matrix, as per the theorem statement, and recalling that the posterior $p d f \pi ( \Sigma _ { C } ^ { ( S ) } | \mathbf { D } _ { S } )$ is obtained for a Uniform prior on $\Sigma _ { C } ^ { ( S ) }$ , s.t. it is proportional to the likelihood of $\Sigma _ { C } ^ { ( S ) }$ in $\mathbf { D } _ { S }$ , i.e. $\pi ( \pmb { \Sigma } _ { C } ^ { ( S ) } | \mathbf { D } _ { S } ) \propto \mathcal { L } ( \pmb { \Sigma } _ { C } ^ { ( S ) } ; \mathbf { D } _ { S } )$ . Towards the computation of this likelihood, we use the result that $d ( \mathbf { \tilde { { Z } } } _ { m } ^ { ( S ) } ) = | \pmb { Y } | ^ { - ( d + 1 ) } d \pmb { Y }$ (Mathai and Pederzoli, 1997), where $Y : = ( \Sigma _ { m } ^ { ( S ) } ) ^ { - 1 }$ . Then invoking invertibility of $\mathbf D _ { S } ( \pmb { \Sigma } _ { C } ^ { ( S ) } ) ^ { - 1 } ( \mathbf D _ { S } ) ^ { T }$ , the result follows.

The posterior probability density of the correlation matrix $\Sigma _ { C } ^ { ( S ) }$ , given the data $\mathbf { D } _ { S }$ , as stated in Theorem 3.1, is reminiscent of the density of the matrixvalued t-distribution, but diferences exist.

## 4. Inference

Remark 4.1. When the inter-observable correlation matrix $\Sigma _ { C }$ of the given multivariate data D is known, then the probability of the RGG variable computed with the known partial correlation matrix $R = [ \rho _ { i , j } ]$ - which is itself computed using the known $\Sigma _ { C } ~ - ~ i s$ closed-form. We then learn $G _ { i , j }$ (given $\rho _ { i , j } )$ by undertaking Rejecton Sampling from the density of the disparity variable $S _ { i , j } = | G _ { i , j } - \rho _ { i , j } | , \forall i , j \in [ M ]$

In Section 4.1, we present the steps for undertaking Rejection Sampling-based learning of the RGG, given the known correlation matrix of the given data. This approach is referred to as Algorithm 1.

Remark 4.2. When we wish to learn $\Sigma _ { C }$ of data D, and then learn the RGG given this learnt correlation matrix, we undertake MCMC-based Bayesian inference on the RGG and the partial correlation matrix R simultaneously, within a Metropolis-with-2-block-update algorithm. We update R (by updating $\Sigma _ { C } )$ in the first block of an iteration of the undertaken MCMC chain, and then in the second block of the iteration, update the RGG variable using the updated R. However, there is no feedback from this second block to the first block of the subsequent iteration. Thus, this inferential scheme difers from Metropolis-within-Gibbs. Hence we refer to it as Metropolis-with-2-block-update.

The algorithm for the learning of the correlation matrix and the RGG using MCMC, is shown in Algorithm 2. A schematic flow of the undertaken inference is in Figure 1.

In the applications we undertake, we do not possess strong prior information on the unknowns; in such situations, the priors used are retained as weak. This is manifest in high variances used for the prior probability density on any unknown. A graph edge parameter has a prior of $B e r n o u l l i ( 0 . 5 )$ . In applications that are blessed with more information, stronger priors can be used.

## 4.1. Algorithm 1: implementation of Rejection Sampling

At a chosen $\tau \in [ 0 , 1 ]$ , and knowing that the (partial) correlation matrix of the data D is ${ \bf { \mathit { R } } } = [ \rho _ { i , j } ]$ , the following algorithm is used to perform Rejection Sampling to learn the RGG of D.

1. In the k-th trial of Rejection Sampling, we propose the edge between the i-th and j-th nodes as $G _ { i , j } = g _ { i , j } ^ { ( \star , k ) }$ , from a proposal distribution $p ( g _ { i , j } ) =$ Bernoulli(0.5), and scale it with the constant $C _ { i , j } / K$ that envelopes the target function $f _ { S _ { i , j } } ( s = | g _ { i , j } ^ { ( \star , k ) } - \rho _ { i , j } | ) / K$ . (See section 2.6 for details). This is equivalent to performing Rejection Sampling from $\mathrm { P r } ( G _ { i , j } = g _ { i , j } | \rho _ { i , j } ) =$ $f _ { S _ { i , j } } ( s ) / C _ { i , j }$ , using the proposal $p ( g _ { i , j } )$ = Bernoulli(0.5), $\forall \rho _ { i , j } \in [ 0 , 1 ]$ . We undertake this $\forall i , j \in [ M ]$

2. Then $( f _ { S _ { i , j } } ( s = | g _ { i , j } ^ { ( \star , k ) } - \rho _ { i , j } | ) / K ) / ( ( C _ { i , j } / K ) ) ) \ge u$ , implies that we accept the k-th sample as $\bar { g } _ { i , j } ^ { ( k ) } = g _ { i , j } ^ { ( \star , k ) }$ ; else we set $g _ { i , j } ^ { ( k ) } = 1 - g _ { i , j } ^ { ( \star , k ) }$ . Here, $U = u$ where $U \sim \mathrm { U n i f o r m } [ 0 , 1 ]$ . We undertake the Rejection Sampling till we have generated $n _ { i , j }$ samples that we use to populate the set $P _ { i , j } ^ { ( R ) }$ as:

![](images/9da828f6ca804320624eb22f7fb068a5e6dda7d9d24f1b1cc835ee623d0d12cd.jpg)  
Fig 1. Flow depicting inference undertaken, when the correlation matrix of the given dataset is known or to-be-learnt.

$P _ { i , j } ^ { ( R ) } = \{ g _ { i , j } ^ { ( 1 ) } , . . . , g _ { i , j } ^ { ( n _ { i , j } ) } \}$ . The above is undertaken for $\forall i , j \in [ M ]$ , with $n _ { i , j } = n \sim 1 0 ^ { 3 } ~ \mathrm { o r } ~ 1 0 ^ { 4 }$

3. Compute $\begin{array} { r } { w _ { i , j } = \sum _ { k = 1 } ^ { ( n _ { i , j } } g _ { i , j } ^ { ( k ) } / n _ { i , j } , \forall i , j \in [ M ] } \end{array}$ , where $\operatorname* { P r } ( G _ { i , j } = 1 | \rho _ { i , j } )$ ≈ $w _ { i , j }$

4. Compute $F _ { S _ { i , j } } ( s = 1 - \rho _ { i , j } )$ (using Equation 2.4). $F _ { S _ { i , j } } ( s = 1 - \rho _ { i , j } ) <$ $\tau \implies$ edge between the i-th and j-th nodes exists with probability $\mathrm { P r } ( G _ { i , j } = 1 | \rho _ { i , j } ) \approx w _ { i , j }$ . On the other hand, $F _ { S _ { i , j } } ( s = 1 - \rho _ { i , j } ) \geq \tau \implies$ $G _ { i , j } = 0$ . This is undertaken $\forall i , j \in [ M ]$

5. Thus, learning the value of $G _ { i , j } \ \forall i , j , \in \ [ M ]$ , will define the realisation of the RGG variable ${ \mathcal { G } } _ { S , V } ( R , \tau )$ at a chosen τ, given the known partial correlation matrix R (of a given dataset D), for which the edge set is;

$$
E = \{ g _ { i , j } : H ( \tau - F _ { S _ { i , j } } ( 1 - \rho _ { i , j } ) ) , i , j \in [ M ] \} ,
$$

where if the edge exists between the i-th and j-th nodes, it will do so, with probability approximated by $w _ { i , j } , \forall i , j \in [ M ]$

## 4.2. Algorithm 2: implementation of Metropolis-with-2-block-update

Updating of the correlation matrix is performed in the first block, and each edge variable is updated in the second block. For the correlation matrix $\Sigma _ { C } ^ { ( S ) } = [ c _ { i , j } ]$ of the given data D, in the t-th iteration, $C _ { i j }$ is proposed from a Truncated Normal density (TN) that is left truncated at -1 and right truncated at 1, with an experimentally chosen variance $( v _ { i , j } )$ , and the current value $c _ { i , j } ^ { ( t - 1 ) }$ as the proposal mean. When we learn the graph with partial correlations - instead of correlations - we update the partial correlation between $X _ { i }$ and $X _ { j }$ to $\rho _ { i , j } ^ { ( t ) }$ using the value of $C _ { i , j }$ that is updated in the first block of the t-th iteration. Then $G _ { i , j }$ is proposed in the second block of this iteration, from a Bernoulli pmf with parameter $\rho _ { i , j } ^ { ( t ) }$ . We discuss the algorithm below.

1. In the first block of t-th iteration of the MCMC chain, the $i , j \cdot$ -th element of the correlation matrix is proposed as:

$$
C _ { i , j } = c _ { i , j } ^ { \star , t } \sim \mathcal { I N } ( c _ { i , j } ^ { ( t - 1 ) } , v _ { i , j } , - 1 , 1 ) , \ \forall i , j \in [ M ] .
$$

Let the proposed $C _ { i , j }$ value be the $i , j \cdot$ -th element of the proposed interobservable correlation matrix $\Sigma _ { C } ^ { ( \star , t ) }$ , i.e.

$$
\begin{array} { r } { \Sigma _ { C } ^ { ( \star , t ) } = [ c _ { i , j } ^ { \star , t } ] . } \end{array}
$$

2. This proposed covariance matrix may or may not be accepted, (as $\Sigma _ { C } ^ { ( t ) } = $ $[ c _ { i , j } ^ { ( t ) } ] )$ , depending on whether the acceptance ratio leads to acceptance or rejection. Acceptance occurs if

$$
\frac { \pi ( \pmb { \Sigma } _ { C } ^ { ( \star , t ) } | \mathbf { D } _ { S } ) \times \prod _ { i , j \in [ M ] } p d f \mathrm { o f } \ \mathcal { I } \mathcal { N } ( c _ { i , j } ^ { ( t - 1 ) } , v _ { i , j } , - 1 , 1 ) } { \pi ( \pmb { \Sigma } _ { C } ^ { ( t - 1 ) } | \mathbf { D } _ { S } ) \times \prod _ { i , j \in [ M ] } p d f \mathrm { o f } \ \mathcal { T } \mathcal { N } ( c _ { i , j } ^ { ( \star , t ) } , v _ { i , j } , - 1 , 1 ) } \geq u ,
$$

where $U = u$ , with $U \sim U n i f o r m [ 0 , 1 ]$ . Here $\Sigma _ { C } ^ { ( t - 1 ) } = [ c _ { i , j } ^ { ( t - 1 ) } ]$ and posterior probability density of a correlation matrix is $\pi ( \Sigma _ { C } ^ { ( \cdot ) } | \mathbf { D } _ { S } )$ as given in Theorem 3.1. If accepted, we set $\Sigma _ { C } ^ { ( t ) } = \Sigma _ { C } ^ { ( \star , t ) }$ ; else, $\Sigma _ { C } ^ { ( t ) } = \Sigma _ { C } ^ { ( t - 1 ) }$ . Here, $\mathbf { D } _ { S }$ is the dataset that results from standardisation of D.

3. Theorem 3.1 gives the posterior of $\Sigma _ { C } ^ { ( S ) }$ for Uniform $( U n i f o r m [ - 1 , 1 ] )$ prior on $c _ { i , j } , \forall i , j \in \{ 1 , \ldots , p \}$ . Thus the likelihood of $\Sigma _ { C } ^ { ( S ) }$ in the data, is given by this theorem. If in an application, a more informative prior on $\ : c _ { i , j } \ : \ : - \ :$ than Uniform - is available, then such a prior is multiplied with the likelihood, to provide the posterior, that is in turn used in the acceptance ratio.

4. Then using this updated $\Sigma _ { C } ^ { ( S ) }$ matrix, the updated partial correlation matrix $R ^ { ( t ) } = [ \rho _ { i , j } ^ { ( t ) } ]$ is populated using the updated absolute partial correlations $\{ \rho _ { i , j } ^ { ( t ) } \} _ { i , j \in V }$ that are computed using Equation 2.1.

5. Subsequently, in the 2nd block of the t-th iteration, the RGG is updated, using the recently updated absolute partial correlation matrix $\boldsymbol { R } ^ { ( t ) }$ . The proposed edge variable connecting the i-th to j-th vertex is

$$
G _ { i , j } = g _ { i , j } ^ { ( \star , t ) } \sim B e r n o u l l i ( \rho _ { i , j } ^ { ( t ) } ) .
$$

We compute the density of the disparity variable $S _ { i , j } = | G _ { i , j } - \rho _ { i , j } ^ { ( t ) } |$ , at the proposed edge, where this density $f _ { S _ { i , j } } ( s )$ computed at $s = | g _ { i , j } ^ { ( \star , \bar { t } ) } - \rho _ { i , j } ^ { ( t ) } |$ 2 is proprtional to $\mathrm { P r } ( G _ { i , j } = g _ { i , j } ^ { ( \star , t ) } | \rho _ { i , j } ^ { ( t ) } )$ . This holds $\forall i , j \in [ M ]$

6. We accept the proposed edge if the acceptance ratio exceeds or equals u that is the value of $U \sim U n i f o r m [ 0 , 1 ]$ , where acceptance ratio relevant to the 2nd block of the t-th iteration is computed at $R _ { i , j } = \rho _ { i , j } ^ { ( t ) }$ as:

$$
\begin{array} { r l } & { \underset { i , j \in [ M ] } { \prod } \ f _ { S _ { i , j } } ( s = | g _ { i , j } ^ { ( \star , t ) } - \rho _ { i , j } ^ { ( t ) } | ) } \\ & { \underset { i , j \in [ M ] } { \prod } \ f _ { S _ { i , j } } ( s = | g _ { i , j } ^ { ( t - 1 ) } - \rho _ { i , j } ^ { ( t - 1 ) } ) } \\ & { \underset { i , j \in [ M ] } { \prod } \underset { p m f \mathrm { o f } \ B e r n o u l l i ( \rho _ { i , j } ^ { ( t - 1 ) } ) } { \prod } } \\ & { \underset { i , j \in [ M ] } { \prod } \ f m f \mathrm { o f } \ B e r n o u l l i ( \rho _ { i , j } ^ { ( t ) } ) } \end{array}
$$

If accepted, we set $g _ { i , j } ^ { ( t ) } = g _ { i , j } ^ { ( \star , t ) }$ ; else, $g _ { i , j } ^ { ( t ) } = g _ { i , j } ^ { ( t - 1 ) } , \forall i , j \in [ M ]$ . Here the density of the disparity variable is given in Equation 2.2.

7. Prior used on $G _ { i , j }$ is Bernoulli(0.5).

Values of the edge variable $G _ { i , j }$ current in the n post-burnin iterations are collated into the set $P _ { i , j } ^ { ( R ) } , \forall i , j \in [ M ]$ . Then out of the n samples collected in $P _ { i , j } ^ { ( R ) }$ , we compute the fraction $w _ { i , j }$ of samples that have a value 1, where $w _ { i , j }$ estimates the sample mean of $\operatorname* { P r } ( \tilde { G } _ { i , j } = 1 | \rho _ { i j } )$ . The edge between the i-th and j-th nodes then exists with probability $\operatorname* { P r } ( G _ { i , j } = 1 | \rho _ { i , j } ) \mathbf { 1 } _ { H ^ { ( \tau ) } } ( F _ { S _ { i , j } } ( 1 - \rho _ { i , j } ) )$ 2 where the set $\pmb { H } ^ { ( \tau ) } = \{ \delta : \delta < \tau , \delta \in [ 0 , 1 ] \}$ . Thus, the RGG ${ \mathcal { G } } _ { S , V } ( R , \tau )$ is learnt, with the probability of the existent edges estimated.

## 4.3. When learning large networks

When our interest is in learning a graphical model on a vertex set with $| V | =$ $p \gtrsim 2 0$ - as in a large network - the computational cost of making MCMC-based inference on the of-diagonal elements in the upper (or lower) triangle of a $p \times p -$ dimensional correlation matrix, is prohibitive, thereby prohibiting computation of R. Then we use the plugin estimate $\hat { c } _ { i , j }$ of $c _ { i , j }$ , s.t. correlation matrix $\Sigma _ { C } ^ { ( S ) }$ is estimated as $\left[ \hat { c } _ { i , j } \right]$ . Then the partial correlation matrix is estimated as $\scriptstyle { \hat { R } } ,$ using this estimated correlation matrix, as long as $p ~ \lesssim ~ 1 0 0 0$ . Thus, such a network is learnt as the RGG $\mathcal { G } _ { S , V } ( \hat { R } , \tau )$ if $p \lesssim$ 1000. However, if $p > 1 0 0 0$ the cost of inverting the estimated $\Sigma _ { C } ^ { ( S ) }$ - to achieve the corresponding partial correlation matrix - might be desired to be avoided. In that case, we will learn the RGG using the estimated correlation matrix, instead of the partial correlation matrix, i.e. as $G _ { S , V } ( \pmb { \Sigma } _ { C } ^ { ( S ) } , \tau )$ . Then we also reduce the number $N _ { i t e r }$ of samples drawn from the marginal of $G _ { i , j }$ , depending on the value of $p ,$ when undertaking Rejection Sampling, $\forall i , j \in [ M ]$

We have learnt such a network bearing > 5000 nodes, each of which is a human disease, (Section 6). In this network, any pair of such diseases is correlated by the rank of the extent of overlap of corresponding phenotypes.

## 5. Generalisation to learning an RGG of data on vector-valued observables

Let us consider learning the graph of the dataset $\mathbf { D } = \{ \pmb { x } _ { 1 } ^ { ( k ) } , \pmb { x } _ { 2 } ^ { ( k ) } , . . . , \pmb { x } _ { p } ^ { ( k ) } \} _ { k = 1 } ^ { n } ,$ where the observable $X _ { i } ~ = ~ \pmb { x } _ { i } ^ { ( \cdot ) } ~ \in ~ \pmb { \chi } ~ \subseteq ~ \mathbb { R } ^ { d } , ~ \forall i ~ \in ~ \{ 1 , \dots , p \}$ . Here we define $\pmb { X _ { i } } = ( X _ { i , 1 } , \ldots , X _ { i , d } ) ^ { T }$ . It is also possible that the dataset D comprises a varying number of observations of $X _ { i } ,$ distinguished from that of $X _ { j }$ , for $i , j \in \{ 1 , 2 , \dotsc , p \}$ , but here we develop the learning of the graph using the same number (n) of observations of each of the $p$ observables. Consequently, the dataset D is cuboidal in shape. In fact, we standardise the data D to the dataset $\mathbf { D } _ { s }$ , as suggested in Section 3.

Motivated by our RGG learning given scalar observables, we draw the RGG of $\mathbf { D } _ { \varepsilon }$ in the probabilistic metric space $\{ \boldsymbol { x } , D ( \cdot ) , \Delta \}$ , with the distance $D ( \cdot )$ between a pair of distinct nodes, given by the cdf of the disparity between the observables attached to these nodes, conditional on the absolute (partial) correlation between these variables. The triangle function $\Delta$ is defined in Definition 2.7.

This RGG is defined on the vertex set $V ^ { \prime } = \{ ( 1 , 1 ) , ( 1 , 2 ) , \ldots , ( 1 , d ) , ( 2 , 1 )$ $\ldots , ( 2 , d ) , \ldots , ( p , 1 ) , \ldots , ( p , 1 ) , \ldots , ( p , d ) \}$ , where the observable $X _ { i , m }$ is attached to the $( i , m )$ -th node, $\forall i \in \left\{ 1 , \ldots , p \right\}$ and $\forall m \in \{ 1 , \ldots , d \}$ . The learnt RGG is a flat graph, s.t. edges can exist between nodes $( i , m )$ and $( i , m ^ { \prime } )$ , as well as between nodes $( i , m )$ and $( j , m ^ { \prime } )$ for $m < m ^ { \prime } ; m , m ^ { \prime } \in \{ 1 , \ldots , d \}$ and $i , j \in [ M ]$ 2 though self-loops are disallowed in our learnt RGGs.

Definition 5.1. In this exercise of learning the RGG with vector-valued variables attached to respective nodes, we invoke the disparity $S _ { i , j } ^ { ( m , m ^ { \prime } ) }$ between $X _ { i , m }$ and $X _ { j , m ^ { \prime } . }$ , i.e. the disparity between the m-th component $o { \ddot { f } } X _ { i }$ and the m′-th component of $X _ { j } , \forall m , m ^ { \prime } \in [ M ]$ and $i < j ; i , j \{ 1 , \ldots . d \}$ . We invoke the defined disparity between two random variables from above, as: $S _ { i , j } ^ { ( m , m ^ { \prime } ) } = | G _ { i , j } ^ { ( m , m ^ { \prime } ) } - $ $\rho _ { i , j } ^ { ( m , m ^ { \prime } ) } | .$ , and $S _ { i , j } ^ { ( m , m ^ { \prime } ) }$ attains values $\in \ [ 0 , 1 ]$ . Here, the edge variable that connects the $( i , m )$ -th node to the $( j , m ^ { \prime } )$ -th node is $G _ { i , j } ^ { ( m , m ^ { \prime } ) }$ , and $\rho _ { i , j } ^ { ( m , m ^ { \prime } ) }$ is the known, absolute (partial) correlation between the variables attached to these two nodes. Then the distance between these two nodes is $D ( X _ { i , m } , X _ { j , m ^ { \prime } } ) \ =$ $F _ { S _ { i , j } ^ { ( m , m ^ { \prime } ) } } ( s ) \ = \ K [ ( s ^ { 2 } \ + \ 1 ) / 2 \mathrm { e r f } ( s / \sqrt { 2 } ) \ - s ^ { 2 } / 2 \ + \ ( s / \sqrt { 2 \pi } ) \exp \bigl ( - s ^ { 2 } / 2 \bigr ) ]$ , as we have seen in Equation 2.4, with K defined in Equation 2.3. In the RGG of dataset $\mathbf { D } _ { s } , i f D ( X _ { i , m } , X _ { j , m ^ { \prime } } ) < \tau$ , the edge exists between the $( i , m )$ -th node (to which $X _ { i , m }$ is attached), and the $( j , m ^ { \prime } ) – t h$ node (to which $X _ { j , m ^ { \prime } }$ is attached). If $D ( X _ { i , m } , X _ { j , m ^ { \prime } } ) \geq \tau _ { ; }$ , the edge between these two nodes is absent in the RGG.

This way of learning the RGG for vector-valued observables is then subject to knowledge of the correlation between variable $X _ { i , m }$ and variable $X _ { j , m ^ { \prime } }$ . Such correlation learning sets the learning of the RGG in this case, diferent from the learning of the RGG given data on scalar-valued observables.

## 5.1. Learning the correlation between $X _ { i , m }$ and $X _ { j , m ^ { \prime } }$

We now discuss learning the inter-variable given dataset $\mathbf { D } _ { s } ,$ in order to learn its RGG. Here $\mathbf { D } _ { s }$ is built with n observations of $X _ { i , m }$ , for $i \in \{ 1 , \ldots , p \}$ and m $\in \{ 1 , \ldots , d \}$ . Hence, the pd × pd-dimensional

$$
\pmb { \Sigma } _ { C } \otimes \pmb { \Sigma } _ { m } = [ c o r r ( X _ { i , m } , X _ { j , m ^ { \prime } } ) ] ,
$$

where

— the $p \times$ p-dimensional matrix of inter-observable correlations is $\Sigma _ { C } =$ $[ c o r r ( { X } _ { i } , { X } _ { j } ) ]$ ;

— the d × d-dimensional matrix of inter-component correlations is $\Sigma _ { m } =$ $[ c o r r ( { \pmb W } ^ { ( m ) } , { \pmb W } ^ { ( m ^ { \prime } ) } ) ]$

— with W defined as:

$$
\mathbf { \boldsymbol { W } } ^ { ( m ) } = ( X _ { 1 , m } , X _ { 2 , m } , \ldots , X _ { p , m } ) ^ { T } .
$$

Thus, corr $\left( X _ { i , m } , X _ { j , m ^ { \prime } } \right)$ is the $d ( i - 1 ) + m , d ( j - 1 ) + m ^ { \prime } - 1$ th element of $\Sigma _ { C } \otimes \Sigma _ { m }$ We need to know corr $\left( X _ { i , m } , X _ { j , m ^ { \prime } } \right)$ to define the absolute (partial) correlation $\rho _ { i , j } ^ { ( m , m ^ { \prime } ) }$ that allows us to compute the cdf $F _ { S _ { i , j } ^ { ( m , m ^ { \prime } ) } } ( s )$ of the disparity between $X _ { i , m }$ and $X _ { j , m ^ { \prime } }$ , (Equation 2.4), and thereby learn if the edge exists between the nodes that these two variables are attached to, at a chosen τ.

## 5.1.1. Inter-observable correlation matrix

The inter-observable correlation $\Sigma _ { C }$ matrix can be estimated before we start learning the RGG, or could be inferred upon using the posterior probability density that was reported earlier in Section 3. This posterior density is

$$
\pi ( \Sigma _ { C } | \mathbf { D } _ { S } ) \propto \Big | \Sigma _ { C } \Big | ^ { - p / 2 } \Big | \mathbf { D } _ { S } ( \Sigma _ { C } ) ^ { - 1 } ( \mathbf { D } _ { S } ) ^ { T } \Big | ^ { - \frac { n + 1 } { 2 } } .
$$

## 5.1.2. Inter-component correlation matrix

The $m , m ^ { \prime } { \cdot } \mathrm { t h }$ element of the inter-component correlation $\Sigma _ { m }$ can be computed by transforming a statistical distance or divergence - such as the Hellinger distance or Kullbeck-Leibler divergence - between the probability of the RGG variable learnt given the data on $\breve { W } ^ { ( m ) }$ and the probability of the RGG given observations of $W ^ { ( m ^ { \prime } ) }$ . Here, n values of $\mathbf { \boldsymbol { W } } ^ { ( m ) }$ comprise the dataset ${ \mathbf { D } } _ { { \mathbf { W } } ^ { ( m ) } } : =$ $\{ ( \boldsymbol { x } _ { 1 , m } ^ { ( i ) } , \ldots , \boldsymbol { x } _ { p , m } ^ { ( i ) } ) \} _ { i = 1 } ^ { n } , \forall m \in \{ 1 , \ldots , d \}$

Let $G _ { i , j } ^ { ( m ) } = g _ { i , j } ^ { ( m ) } \in \{ 0 , 1 \}$ be the edge between the (i, m)-th node that hosts the m-th component of the i-th observable, and the $( j , m ) \ – \mathrm { t h }$ node, (to which the m-th component of the $j \mathrm { - t h }$ observable is attached). Again, let $\rho _ { i , j } ^ { ( m ) }$ be the absolute (partial) correlation between these two observables. To compute the probability $\pi ( \mathcal { G } _ { S , V } ( \boldsymbol { R } ^ { ( m ) } , \tau ) | \boldsymbol { R } ^ { ( m ) } )$ of the RGG variable $\mathcal { G } _ { S , V } ( R ^ { ( m ) } , \tau )$ , given ${ \mathbf { D } } _ { W ^ { ( m ) } }$ , we recall from Equation 2.8 that we first need to compute the probability with which the edge $G _ { i , j } ^ { ( m ) }$ exists in this RGG. We use Rejection Sampling to draw n samples from the unnormalised density of the disparity between variables $X _ { i , m }$ and $X _ { j , m }$ , where said density is $f _ { S _ { i , j } ( m ) } ( | g _ { i , j } ^ { ( m ) } - \rho _ { i , j } ^ { ( m ) } | ) / K =$ $\begin{array} { r } { | g _ { i , j } ^ { ( m ) } - \rho _ { i , j } ^ { ( m ) } | \operatorname { e r f } \left( | g _ { i , j } ^ { ( m ) } - \rho _ { i , j } ^ { ( m ) } | / \sqrt { 2 } \right) - | g _ { i , j } ^ { ( m ) } - \rho _ { i , j } ^ { ( m ) } | + \sqrt { \frac { 2 } { \pi } } \exp ( - ( g _ { i , j } ^ { ( m ) } - \rho _ { i , j } ^ { ( m ) } ) ^ { 2 } / 2 ) } \end{array}$ where K is defined in Equation 2.3. The fraction $w _ { i , j } ^ { ( m ) }$ of the samples that is a 1, is the approximation of the probability $\operatorname* { P r } ( G _ { i , j } ^ { ( m ) } = 1 | \rho _ { i , j } ^ { ( m ) } )$ . To then acknowledge the efect of the thresholding, we compute the distance $D ( X _ { i , m } , X _ { j , m } ) \ -$ between the nodes to which $X _ { i , m }$ and $X _ { j , m }$ are attached - at $G _ { i , j } ^ { ( m ) } = 1$ . This is the cdf of the disparity $S _ { i , j ^ { ( m ) } }$ between these two random variables, computed at $1 - \rho _ { i , j } ^ { ( m ) }$ , and is given in Equation 2.4. If this cdf falls short of the chosen $\tau ,$ then edge $G _ { i , j } ^ { ( m ) } = 1$ , with probability $w _ { i , j } ^ { ( m ) }$ ; else, there is no edge between the nodes to which $X _ { i , m }$ and $X _ { j , m }$ are attached. Then

$$
\pi ( \mathcal { G } _ { S , V } ( \pmb { R } ^ { ( m ) } , \tau ) | \pmb { R } ^ { ( m ) } ) = \prod _ { i , j \in [ M ] } w _ { i , j } ^ { ( m ) } \mathbf { 1 } _ { \pmb { H } ^ { ( \tau ) } } ( F _ { S _ { i , j } ^ { ( m ) } } ( 1 - \rho _ { i , j } ^ { ( m ) } ) )\tag{5.1}
$$

Similarly, the probability of the RGG variable of the data $\mathbf { D } _ { W ^ { ( m ^ { \prime } ) } }$ is computed. The Hellinger distance (or Kullbeck Leibler divergence):

$$
\delta ( \mathcal { G } _ { S , V } ( \pmb { R } ^ { ( m ) } , \tau ) , \mathcal { G } _ { S , V } ( \pmb { R } ^ { ( m ^ { \prime } ) } , \tau ) )
$$

is computed between $\pi ( \mathcal { G } _ { S , V } ( \pmb { R } ^ { ( m ) } , \tau ) | \pmb { R } ^ { ( m ) } )$ and $\pi ( \mathcal { G } _ { S , V } ( \pmb { R } ^ { ( m ^ { \prime } ) } , \tau ) | \pmb { R } ^ { ( m ^ { \prime } ) } )$ .

Definition 5.2. $\delta ( \mathcal { G } _ { S , V } ( \pmb { R } ^ { ( m ) } , \tau ) , \mathcal { G } _ { S , V } ( \pmb { R } ^ { ( m ^ { \prime } ) } , \tau ) )$ , is computed using the set of probability values that are computed using samples (generated in $n _ { i , j }$ trials, with Rejection Sampling) of the edge variables relevant to each RGG, where we normalise each computed probability by the maximal - out of all computed - probability values. Additionally, the inter-observable correlation matrices of datasets ${ \mathbf { D } } _ { W ^ { ( m ) } }$ and $\mathbf { D } _ { W ^ { ( m ^ { \prime } ) } }$ can be learnt using the closed-form posterior density of the inter-observable correlation matrix discussed in Section 3. Then the $m , m ^ { \prime } { - } t h$ element of the inter-component correlation $\Sigma _ { m }$ is modelled as:

$$
\pmb { \Sigma } _ { m } : = [ \mathrm { e x p } ( - \delta ( \mathcal { G } _ { S , V } ( \pmb { R } ^ { ( m ) } , \tau ) , \mathcal { G } _ { S , V } ( \pmb { R } ^ { ( m ^ { \prime } ) } , \tau ) ) ) ] .\tag{5.2}
$$

## 5.1.3. Correlation between any two components of any two observables

Knowing the d × d-dimensional $\Sigma _ { m }$ and the $p \times p .$ -dimensional $\Sigma _ { C }$ , we compute the correlation between the m-th component of the i-th observable and the m′-th component of the j-th observable, $\forall i , j \in V$ , and ∀m, $m ^ { \prime } \in \{ 1 , \ldots , d \}$ :

$$
\pmb { \Sigma } _ { C } \otimes \pmb { \Sigma } _ { m } = [ c o r r ( X _ { i , m } , X _ { j , m ^ { \prime } } ) ] ,
$$

## 5.2. Learning RGG given inter-observable and inter-component correlation matrices

The RGG that we learn given the cuboidally-shaped dataset $\mathbf { D } _ { s } ,$ , is ${ { g } _ { S , V ^ { \prime } } } ( \pmb { \Sigma } _ { C } \otimes$ $\Sigma _ { m } , \tau )$ . Then for $( i , m ) , ( j , m ^ { \prime } ) \in V ^ { \prime }$ , the edge $G _ { i , j } ^ { ( m , m ^ { \prime } ) }$ between the $( i , m )$ -th node and the $( j , m ^ { \prime } )$ -th node exists, if the inter-nodal distance $D ( X _ { i , m } , X _ { j , m ^ { \prime } } ) =$ $[ \sqrt { 2 / \pi } \exp ( - | 1 - \rho _ { i , j } ^ { ( m , m ^ { \prime } ) } | ^ { 2 } / 2 ) + | 1 - \rho _ { i , j } ^ { ( m , m ^ { \prime } ) } | \operatorname { e r f } ( ( | 1 - \rho _ { i , j } ^ { ( m , m ^ { \prime } ) } | ) / \sqrt { 2 } ) - | 1 - \rho _ { i , j } ^ { ( \overline { { m } } , m ^ { \prime } ) } | ] < 0 .$ τ; else this edge does not exist. As delineated above, we can again estimate the probability with which this edge exists - if at all - using the relative frequency of an edge sample to be 1, amongst samples drawn using Rejection Sampling, from the pdf of the disparity between the variables $X _ { i , m }$ and $X _ { j , m ^ { \prime } }$ . Thus, the RGG is learnt given the absolute correlation $\rho _ { i , j } ^ { ( m , m ^ { \prime } ) }$ between variables $X _ { i , m }$ and $X _ { j , m ^ { \prime } } .$ where $\rho _ { i , j } ^ { ( m , m ^ { \prime } ) }$ is the $d ( i - 1 ) + m , d ( j - 1 ) + m ^ { \prime } \ – \mathrm { t h }$ element of $\Sigma _ { C } \otimes \Sigma _ { m }$ . We might know the inter-observable matrix $\Sigma _ { C }$ , or learn it using its closed-form posterior, as delineated in Section 3. The inter-component matrix $\Sigma _ { m }$ might again be known, or it can be computed as a transformation - as stated in Equation 5.2 - of the distance between the probability of the RGG of the data on the m-th component of all observables, and that of the data on the $m ^ { \prime } .$ -th component of all observables.

## 6. Empirical illustration: learning the human disease-phenotype network

The human disease-phenotype network was learnt by Hoehndorf et al. (2015) (HSG hereon), by considering the similarity parameter for each pair of diseases that is an element of an identified set of diseases in the Human Disease Ontology (DO), that contains information about rare and common diseases, and spans heritable, developmental, infectious and environmental diseases. Here, the “similarity parameter” between one disease and another, is computed using the ranked vectors of “normalised pointwise mutual information” (NMPI) parameters for the two diseases, where the NMPI parameter describes the relevance of a phenotype, to the disease in question. HSG define the NMPI parameter semantically, as the normalised number of co-occurrences of a given phenotype and a disease in the titles and abstracts of 5 million articles in Medline. The diseasedisease pairwise semantic similarity parameters – computed using the degree of overlap in the relevance ranks of phenotypes associated with each disease – result in a similarity matrix, which HSG turn into an inter-disease network based on phenotypes. They choose from the top-ranking 0.5% of inter-disease similarity values. Phenotypes associated with diseases, and corresponding scoring functions (such as the NPMI), exist in the file “doid2hpo-fulltext.txt.gz” at http://aber-owl.net/aber-owl/diseasephe notypes. In fact, at the site http://aber-owl.net/aber- owl/diseasephenotypes/data/, HSG have uploaded all the data that they have used. The file ”doid2hpo-fulltext.txt.gz” available at this site, contains information about $N _ { d i s }$ diseases, and the semantic relevance of each of the $N _ { p h e n o }$ phenotypes to each disease, as quantified by NPMI parameter values, in addition to other scores such as t-scores and z-scores. In this file, $N _ { d i s }$ is 8676 and $N _ { p h e n o }$ is 19323.

![](images/9bbee4a51e043d4272f209e78c2f294a9b53f1e8e4db596133a6f49f67b14f1a.jpg)  
Fig 2. RGG ${ \mathcal G } _ { S , \Sigma _ { C } ^ { ( S t ) } } ( V , 0 . 1 )$ , representing the human disease-phenotype network that we learn using the computed Spearman rank correlation between the rank vectors of a list of phenotypes relevant to a disease, where the phenotype ranking reflects semantic relevance of a phenotype to the disease in question (quantified by HSG as the NPMI parameter in the ${ \bf D } _ { D P h }$ dataset). In our learnt RGG, $\tau = 0 . 1$ . Here the vertex set V has 8676 elements, but nodes with no edges are discarded from this visualised graph, resulting in 6052 diseases (nodes) and 145210 edges that are shown this figure. Diseases identified by HSG, to belong to one of the 19 given disease class, are presented above in the same colour; the colour key identifying these classes, is attached.

In the phenotypic similarity network between diseases that HSG report, diseases are the nodes, and the edge between two nodes exists in this undirected graph, if the similarity between the nodes (diseases) is in the highest-ranking 0.5% of the 38,688,400 similarity values. They remove all self-loops and nodes with a degree of 0. Their network is presented in http://aber-owl.net/aberowl/diseasephenotypes/network/. The “Group Selector” function on their visualisation kit, allows for the identification of 19 clusters in their disease-disease network, with each cluster corresponding to a disease-class. Total number of nodes over their identified 19 clusters, is 5059; number of edges is 65,795; average node degree≈26.2.

HSG’s network then manifests a similarity-structure that is computed using available NPMI parameter values.

Our interest is in learning the disease-disease network as an RGG, with each edge existent at a learnt probability. We perform such learning using the NPMI semantic-relevance data that is made available for each of the $N _ { d i s }$ number of diseases, by HSG; so $N _ { d i s }$ is the norm of the vertex set V of our sought RGG. We refer to this human disease-phenotype data as ${ \bf D } _ { D P h }$ . Using $\mathbf { D } _ { D P h } ,$ we first compute the correlation $S _ { i j }$ between the i-th and j-th diseases in $V ,$ for each of which, information on the ranked (semantic) relevance of each of the $N _ { p h e n o }$ phenotypes exist, in this given dataset. Upon computation of pairwise correlations, the RGG for the data $\mathbf { D } _ { D P h }$ is learnt.

We compute the correlation between the i-th and j-th diseases in the $\mathbf { D } _ { D P h }$ data, $( i , j \in \{ 1 , \dots , N _ { d i s } \} , i < j )$ , in the following way. We rank the NPMI parameter values - that indicate association between the i-th disease and each of the $N _ { p h e n o }$ phenotypes - with phenotypes of highest semantic relevance to the i-th disease, assigned a rank 1. Let the rank vector of phenotypes, by semantic relevance to the i-th disease take the value r and similarly, that for the j-th disease is r<sub>j</sub>. We compute the Spearman rank correlation $s _ { i j } ^ { ( r a n k ) }$ , between vectors r<sub>i</sub> and $\pmb { r _ { \mathscr { j } } } \forall i , j \in \{ 1 , . . . , N _ { d i s } \} ; i < j .$ Spearman rank correlation is preferred to the correlation between vectors of normalised NPMI values, since we intend to correlate the i-th disease with the j-th disease, depending on how relevant a given list of phenotypes is, to each disease, i.e. on the ranked relevance of phenotypes. We learn the network given this correlation, that is itself computed using data $\mathbf { D } _ { D P h }$ (see Section 4.3 on learning large networks).

![](images/f3ef653533dcf31e3236bd5c164e6a2177187554f075b22cc3c1c29045d5ae21.jpg)

![](images/91d19b92aa70dd50f0d64879440ef6a7bccd06507ff73ddc1206343a1ba7d92a.jpg)  
Fig 3. Left: comparison of the relative number of nodes (diseases) that we recover in each of the 19 disease classes (in filled circles joined by solid lines), with the relative class-membership reported by HSG overplotted as open circles threaded by broken lines, Right: our computed ratios of the averaged intra-class to inter-class variance for each of the 19 classes, shown in filled circles; the ROC Area Under Curve values reported by HSG for each class, is overplotted as open circles joined by broken lines. The disease class indices are assigned values 1 to 19; these are the following disease classes respectively: cellular proliferation diseases, integumentary diseases, diseases of the nervous system, genetic diseases, diseases of metabolism, diseases by infectious agents, diseases of mental health, physical disorders, diseases of the reproductive system, of the immune system, of the respiratory system, of the musculoskeletal system, syndromes, gastrointestinal diseases,cardiovascular diseases, urinary diseases, viral infections, thoracic diseases, diseases of the endocrine system.

Remark 6.1. The RGG visualised in Figure 2 is a subnet of the full network learnt as the RGG ${ \mathcal G } _ { S , \Sigma _ { C } ^ { ( S t ) } } ( V , 0 . 1 )$ where $| V | = N _ { d i s }$ , and the inter-observable (Spearman rank) correlation matrix of data $\mathbf { D } _ { D P h }$ is $\Sigma _ { C } ^ { ( S t ) } ~ = ~ [ s _ { i j } ^ { ( r a n k ) } ] , ~ i ~ < ~$ j, $i , j \in \{ 1 , \dots , N _ { d i s } \}$ , s.t. this visualised graph has 6052 number of nodes, each with a non-zero degree, and 145210 edges, so that the average node degree is ≈24. This RGG represents our learning of the disease-phenotype network (Figure 2).

## 6.1. Comparing against earlier work

We use Figure 3 to present comparison of our results to HSG’s, including a comparison between the relative number of nodes i.e. diseases, in each of the 19 disease classes that HSG classify their reported network into and our results (shown in the left panel of this figure). Our learnt RGG tallies very well with the earlier result. The right panel of Figure 3 displays the ratio of intra-class to inter-class variance of each disease-class that we identify; value of the area under the Receiver Operating Characteristic curve (ROCAUC) for each cluster identified by HSG is overplotted, where the ROCAUC value for the i-th cluster can be interpreted as probability that a randomly chosen node is ranked as more likely to be in the i-th class than in the j-th; $i \neq j ; i , j = 1 , \ldots , 1 9$

Thus, our method of learning a large network allows for the learning of the clustering distribution of the large dataset, for which this network has been learnt.

## 7. Empirical illustration: identifying optimal cut-of

In this application we consider data on the in-channel water level at 893 river cross-sections, within the wider Humber region in northern England. These water levels are modelled response of the river system to a collection of 132 simulated storm events, and the Environment Agency, U.K. holds proprietorship rights on this data. In this dataset, all storms are characterised by five hydraulic boundary conditions (or parameters), namely the upstream fluvial flow from Aire, Don, Ouse, Trent and the downstream water level at the mouth of the Humber Estuary. By construction, the dataset includes 132 blocks - each block corresponding to a simulated storm - with a block comprising 893 rows and six columns. s.t. the first five columns of the i-th block are populated by the values of the storm parameters relevant to the i-th storm, while the last column holds values of the in-channel water level, (referred hereafter as “water level”), at each of the 893 river cross-section locations, (referred hereon as “location”), that are considered; $i \in \{ 1 , \ldots , 1 3 2 \}$ . Then the observations in the first five columns of any block are the same across all 893 rows of this block, though the observed water level values vary from one row to another, within this block.

We define the random variable $X _ { i }$ as the spatially-local water level variable within the Humber region, when the i-th storm strikes the region. We denote the j-th observation of variable $X _ { i }$ as $x _ { i , j } ;$ this is the water level in the j-th location within the region, due to the i-th storm striking, for $j \in \{ 1 , \dots , 8 9 3 \}$ We also define the random variable $W _ { j }$ that represents the water level in the j-th location, triggered by a storm event. Then the observed water level in the j-th location, due to the i-th storm is $x _ { i , j }$ , i.e. the i-th observed value of $W _ { j }$ in the available data is $x _ { i , j }$

We learn one RGG on the vertex set $V _ { s t o r m } : = \{ 1 , \dots , 1 3 2 \}$ , in which, the random variable $X _ { i }$ is attached to the i-th node, $\forall i \in V _ { s t o r m } .$ Thus, the dataset that is used to learn this RGG is $\mathbf { D } _ { s t o r m }$ that comprises the observed values of each of $X _ { 1 } , \ldots , X _ { 1 3 2 }$ . The $1 3 2 \times 1 3 2 .$ -dimensional correlation function $\Sigma _ { s t o r m } =$ $[ | c o r r ( X _ { i } , X _ { i ^ { \prime } } ) | ]$ is used to learn this RGG. We refer to this RGG learnt at the cutof τ as $\mathcal { G } _ { S , \Sigma _ { s t o r m } } ( V _ { s t o r m } , \tau )$

We learn a second RGG $\mathcal { G } _ { S , \Sigma _ { l o c } } ( V _ { l o c } , \tau )$ on the vertex set $V _ { l o c } : = \{ 1 , \dots , 8 9 3 \}$ in which the variable $W _ { j }$ is attached to the $j \cdot$ -th node $\forall j \in V _ { l o c } ,$ , where the dataset used to learn this RGG is $\mathbf { D } _ { l o c }$ that holds observations of $W _ { 1 } , \ldots , W _ { 8 9 3 }$ s.t. the $8 9 3 \times 8 9 3$ -dimensional correlation matrix $\pmb { \Sigma } _ { l o c } = [ | c o r r ( W _ { j } , W _ { j ^ { \prime } } ) | ]$

We in fact identify the optimal cutof $\tau ^ { \star }$ in each dataset, by identifying the $\tau$ at which the rate of change of the logarithm of the posterior probability of the RGG variable is minimised, s.t. the resulting RGG is most resilient to changes

in τ . Thus,

$$
\tau ^ { \star } : = \arg \operatorname* { m i n } \left( \frac { d \log \pi ( { \mathcal G } _ { S , \Sigma _ { s t o r m } } ( V _ { s t o r m } , \tau ) | \mathbf { D } _ { s t o r m } ) } { d \tau } \right) .
$$

We find $\tau ^ { \star } = 0 . 0 7 8 0 8$ in learning given dataset $\mathbf { D } _ { s t o r m }$

Again, the identification of $\tau ^ { \star }$ in dataset $\mathbf { D } _ { l o c }$ is yields $\tau ^ { \star } = 0 . 2 6 4 3$

![](images/22ece1b6eb048477587cb6560f1c8a2e1ed8b51590f59a2373808d2a554c7bee.jpg)

![](images/89b09357c1bb3d5057022644d3b7a8683f920ee0455a7797f37e3d0af0ea3a3c.jpg)

![](images/61a75b01b27c56fcc037920b6e1df4072cfc1493a80f1141bc45e02d5f25aaf0.jpg)

![](images/21958d1bc2171db479a7610c429e42381bd357a14b8cfeee1a0ad752fff02054.jpg)

![](images/4309a62fca822184e74dd1590bef44eb3b8d8b6e657696ba0ea6a322f6e3b96a.jpg)

![](images/804cfe0b28019165177ceb8410476d94a62394c0d7a07e60084520f93e4c0d70.jpg)

![](images/b0bde6a6137057e4aac3b1c4deb5dd44899119196d2c27c429493892086fab2d.jpg)

![](images/154c87842fade2f8cce0e82d0950d9f571a87a984ea42e05b1655bffa8ba4594.jpg)  
Fig 4. Block $o f$ two left-most panels: panel on the bottom $l e f t$ of this block displays the logarithm of the posterior of the RGG variable of dataset $\mathbf { D } _ { l o c . }$ , learnt at varying values of the cut-of probability. The slope of this graph posterior (with respectto $\tau )$ is plotted against τ on the bottom right. The vertical line in black marks the $\tau$ value of $\it 0 . 2 6 4 3 .$ , at which this slope appears to be minimised. That $t h i s \ \tau$ value is a minima of the slope, is confirmed by plotting the parametric $f i t$ to the numerically-computed first derivative of the slope function with respect $t o \tau$ (in the top left), while the $f i t$ to the second derivative of the slope function is plotted on the top right of this block. Numerical derivative computation is done via differencing in Python, and fits to the computed data are identified using Num $P y ' s$ polynomial fitting, (in Python). The vertical lines in these panels respectively confirm the approximately zero value of the first derivative of the slope, as well as the positivity of the second derivative - at $\tau = 0 . 2 6 4 3$ . Block $o f$ two right-most panels: the same as the two left panels discussed above, except here results relevant $t o$ the $R G G$ variable of daatset $\mathbf { D } _ { s t o r m }$ are presented. The optimal τ value for the RGG of this data is identified as $\tau ^ { \star } = 0 . 0 7 8 0 8$ , at which the slope of the RGG variable with respect to $\tau$ attains a minima (shown $b y$ vertical line in the bottom right panel of this block); where first derivative of the slope is approximately $\theta ,$ (shown in the top $l e f t ) ;$ and its second derivative is positive, as depicted by the vertical line at τ 0.07808 in the top right panel.

Log posterior of the RGG variables computed at $\tau \in \{ 0 . 0 0 1 , 0 . 0 0 2 , \ldots , 0 . 9 9 9 \}$ }, learnt given the correlation between $X _ { i }$ and $X _ { i ^ { \prime } } , \forall i , i ^ { \prime } \in \{ 1 , . . . , 1 3 2 \}$ , are displayed in Figure 4. To illustrate the minimisation of the function that we refer to as $s l o p e ( \tau ) : = \pi ( \mathcal { G } _ { S , \Sigma _ { s t o r m } } ( V _ { s t o r m } , \tau ) | \Sigma _ { s t o r m } ) / d \tau$ , we check the closeness of the (fit to the data on the) derivative of slope(τ ) (with respect to $\tau )$ to 0, and check if the second derivative is positive. Under these checks, $\tau ^ { \star }$ for dataset $\mathbf { D } _ { s t o r m }$ is noted to be about 0.07808. The two right-most panels of Figure 4 show the same for dataset $\mathbf { D } _ { l o c } \mathrm { ~ - ~ } \mathrm { f o r }$ which the corresponding RGG is defined on an 893-sized vertex set - to suggest an optimal $\tau$ of about 0.2643. As stated above, in the computation of the derivatives, we use diferencing, and fit polynomials (using

NumPy’s polynomial fitting) to the generated data. The usage of such fitting defines our algorithm of identifying $\tau ^ { \star }$

Once the optimal values of τ are learnt for $\mathbf { D } _ { s t o r m }$ and that for $\mathbf { D } _ { l o c } .$ we learn the RGG of each of these datasets, at these identified $\tau ^ { \star }$ . The learnt RGGs are visualised in Figure 5. Each of the 893 locations that we consider in $\mathbf { D } _ { l o c } ,$ is within the channel of one of four rivers (Aire, Don, Ouse and Trent) that flow in the Humber region. Thus, the RGG in which a node is attached to the local water level variable, has four clusters. This is noted in the RGG learnt given data $\mathbf { D } _ { l o c }$

![](images/24d9fdee4994c33ae30cf1c4396a5888c5744b0657798ecb261d95f2809230dc.jpg)  
Fig 5. Left: the learnt RGG $\mathcal { G } s , \mathbf { \Sigma } _ { \mathbf { L } o c } ( V _ { l o c } , \tau ^ { \star } )$ , where $\tau ^ { \star } ~ = ~ 0 . 2 6 4 3$ . The RGG displays four clusters, where the nodes of each cluster are represented in distinct colours. Right: $\mathcal { G } _ { S , \Sigma _ { s t o r m } } ( V _ { s t o r m } , 0 . 0 7 8 0 8 )$ learnt at the optimal τ, given dataset $\mathbf { D } _ { s t o r m } .$ . Nodes that belong to the distinct clusters of this graph, are presented in distinct colours. Membership in a cluster is identified using the clustering function in the Network package in Python.

## 8. Empirical illustration: degree distribution at varying $\tau$

We learn RGGs of a dataset comprising binary information, to demonstrate (1) capacity in the RGG-learning to address data on categorical variables, (as well an variables of mixed type); (2) degree distribution, and he efect of the cutof τ on this. The purpose of this empirical illustration is to showcase such capcity, while our treatment of the available dataset does not ofer any realistic interpretation.

The data $\mathbf { D } _ { u s e r }$ we use, is available in “118379821279745746467.feat”, at: https://snap.stanford.edu/data/ego-Gplus.html (Mcauley and Leskovec, 2014), on the p = 500-dimensional feature vector of $n = 6 9 9$ users of Google+ Circles which was a core facility of the defunct Google+ Social Network, used to categorise “friends’“ circles. The presence of any of the p features for any user, is marked by $\mathrm { ~ a ~ } ^ { 6 6 } 1 ^ { 5 }$ in this dataset, while the absence of the same is marked with $\mathrm { ~ a ~ } ^ { 6 6 } 0 ^ { 9 } .$ . We interpret the p-dimensional vector of 0s and 1s as a vector comprising $p$ observed values of a binary parameter X that characterises a user, where each observation is made at each of p distinct instances. X can only attain values of 0 and 1. Thus, a variable $X _ { i } ~ \in ~ \{ 0 , 1 \}$ is attached to the i-th node of the RGG of dataset $\mathbf { D } _ { u s e r } , \forall i \in V _ { u s e r }$ where the RGG is defined on the vertex set $V _ { u s e r } = \{ 1 , \ldots , n \}$ . Since $X _ { i }$ is categorical, we use the Cramer’s V measure (Sheskin, 2003) to compute corr $( X _ { i } , X _ { j } ) \ \forall i , j \in V _ { u s e r }$ . Then we learn the $n \times n$ -dimensional correlation matrix $\pmb { \Sigma _ { u s e r } } \mathbf { \bar { \Psi } } = [ c o r r ( X _ { i } , X _ { j } ) ]$ and learn the RGG $\mathcal { G } _ { S , \Sigma _ { u s e r } } ( V _ { u s e r } , \tau )$ , at diferent τ values, including at the optimal τ for this dataset, which is identified as discussed in the previous section, to be about 0.363.

Figure 6 presents four RGGs learnt at τ values of 0.1752, 0.3630, 0.6036, 0.7968; the degree distribution of each of the learnt RGGs is presented below the corresponding RGG. In the RGG learnt at the best cut-of of about 0.363, the two highest values of the degree are the uniqiue two degree values of all points marked respectively in green and blue, in the corresponding RGG. We note this RGG to bear two core clusters, with the remaining points - i.e. noncolourised points bearing low degrees - contributing to the rest of the RGG. The colourised points in the RGG learnt at the optimal τ , are the respective members of the two clusters that distinguish this learnt graph. Thus, almost all members nodes of each cluster bear the same degree, which are also the highest values of the degree amongst all nodes in the graph.

At higher values of $\tau ,$ we can identify the emergence of further structures in the clustering distribution of the learnt RGG, compared to the cleaner, bimodal clustering distribution that we spot for the RGG learnt at the optimal τ. At these high τ values, a narrow interval of degree values is identified, s.t. these degrees correspond to nodes close to the edges of the individual clusters. The loglog frequency distribution of degrees in this narrow interval is akin to a straight line with a steep slope. At lower degrees, the log-log degree distribution is noted to be similar to a power-law at τ values similar to the optimal $\tau ;$ however, as $\tau$ increases, increasingly more nodes get pulled into the clusters that manifest increasingly more structure, and the degree distribution of the nodes with low degree values then turn Poisson, than power-law.

## 9. Conclusion

In this paper, we have presented a new learning of a random graph of a given multivariate dataset, as an RGG drawn in a probabilistic metric space, to result in an SRGG. We forward the closed-form probability of such an undirected inhomogeneous graph, conditioned on the correlation structure of the dataset for which the graph is learnt. We give the metric of the space that the RGG is drawn in, as the cdf of the disparity variable that we introduce here. Disparity is defined as the absolute diference between the “connectedness” of two nodes, (where said connectedness is given by the mutual edge), and the absolute (partial) correlation between the random variables that are respectively attached to these nodes. Then at the known inter-observable correlation/partial correlation matrix, the edge between these nodes exists in the RGG, if the inter-nodal distance falls short of a chosen cutof probability τ, i.e. if τ exceeds the cdf of the disparity computed at edge=1.

To learn the RGG of a given dataset, we use Rejection Sampling to construct a set of samples drawn randomly from the closed-form probability of the edge between a pair of nodes, (conditioned on the correlation between the pair of observables that are attached respectively to the nodal pair). In this set of edge samples, the relative frequency for the edge variable to attain the value of 1, is then proportional to this conditional edge probability. Thus, if the cdf of the disparity computed at edge=1 falls short of a chosen cutof probability, the edge exists in the RGG, with probability that is (approximately) given by this computed relative frequency.

![](images/bab36b941b14265db29e884bcb2099edf3c672dd89c70e63c27e54e632127a52.jpg)

![](images/f65ef98ec6c738a58dfdcbd7590897e8000105bdfcb1700aaadd3b29aff2fcbb.jpg)

![](images/247fe56cce6450afc86822ad79daf264987ba1c96b229f11a7352b9568282b3f.jpg)

![](images/6fc93deeb3d4cf7c769d922fb01378d2685f7d22676817c559a1fbef0695a4dc.jpg)

![](images/328034e0f10e4dc3081fb77bb81b74db2ce5f51c970f56faa6d5c0cb506c0ab3.jpg)  
log(degree)

![](images/a93d69d56a0e20e0af90d27e81125ee06bf713fa942499fb0398658c8da4b1d4.jpg)

![](images/e5e1878cbb1efedda89526e51d785c649e39a8d276b7eae2379a3973583832bb.jpg)  
log(degree)

![](images/a5371910136fea128a614606095582b9c06cb94b51a44bf23ef82093bd3ad70a.jpg)  
log(degree)  
Fig 6. Top row: RGGs learnt at τ values of 0.1752, 0.3630, 0.6036, 0.7968, from left to right. Nodes that have a degree of 0, are coloured red in these RGGs. The RGG in the second from left panel, represents the RGG learnt at the optimal τ, given the dataset $\mathbf { D } _ { u s e r } ;$ nodes in the two modal clusters of this graph are coloured green and blue. Bottom row: figure representing the degree distribution of a learnt RGG, is placed below the panel that presents the learnt RGG.

Learning such an RGG in a probabilistic metric space allows for an easy way of illustrating the correlation structure of a dataset, including data comprising information from vector-valued observables, irrespective of whether the observables are numerical or not. We realise that the optimal τ at which the RGG is learnt for a given dataset (Zhang et al., 2025) is also organically realised. The τ value at which the slope of the probability of the RGG is minimised, is identified as the threshold that produces the graph that is the most robust to changes in the threshold. Thus, the optimal τ can be identified because we have advanced the closed-form probability of the RGG, given the inter-observable correlation matrix of the given dataset. In fact, the posterior density of this correlation matrix is closed-form, as we have discussed in Section 3.

While the correlation matrix can be updated using its closed-form posterior given the dataset at hand, each edge of the graph can be sampled from the closed-form edge probability, computed using the known inter-observable correlation matrix of the dataset. Inference in such a situation is undertaken using a Metropolis-with-2-block-update scheme. Alternatively, if the correlation structure of the given dataset is known, then we advocate learning the edges by undertaking Rejection Sampling from the closed-form edge probability, (given

the known correlation).

Although omitted from the paper, and included in the Supplement, our approach potentially allows for acknowledgement of measurement errors of the observables, in learning the RGG. It can be demonstrated that the efect of ignoring existent measurement errors, is to distort the learnt RGG, even in the case of a simple, low-dimensional dataset - as demonstrated in Section 1 of the Supplement.

A useful fallout of our RGG learning is the formulation on an inter-graph distance function in the space of such RGGs. Subsequent to the learning of the graph variable given the correlation structure of each of two multivariate datasets, we can define an inter-graph distance as a statistical distance or divergence. Indeed, it is possible to compute the Hellinger distance (Banerjee et al., 2015; Matusita, 1953) between the probabilities of the corresponding pair of random graph variables, given the respective dataset. We have indicated the usage of such an inter-graph distance in Equation 5.2, and real-world applications of such a distance function have been undertaken (Zhang et al., 2025, 2024). In our implementation of the correlation between the m-th and $m ^ { \prime } .$ -th components of the i-th and $j \cdot$ -th observables, we have used the Hellinger distance between the RGG of the data on the m-th component of all observables, and that of the data on the $m ^ { \prime } { \mathrm { - t h } }$ component. This inter-graph distance in our work is a generic distance that is informed by the probability distribution of the random graph variable, conditional on the inter-variable correlation matrix of the dataset comprising (noisy) data available on mixed variables in general. The inter-graph distance that we propose, is unlike distances based on counts of node-wise diferences (Dekker and Colbert, 2004; Imrich, 2000), or distances defined for certain types of graphs (Bang et al., 2015; Godsil, 1988; Mass, 1987).

## Appendix A: Proof of Theorem 3.1

Proof. Likelihood of correlation matrices $\Sigma _ { C } ^ { ( S ) }$ and $\pmb { \Sigma } _ { m } ^ { ( S ) }$ , given data $\mathbf { D } _ { S }$ is matrix Normal, (see Section 3). By invoking the matrix Normal density, we write the joint posterior of these correlation matrices using this likelihood, and the priors stated in the statement of the theorem. The joint posterior probability density of $\Sigma _ { C } ^ { ( S ) } , \Sigma _ { m } ^ { ( S ) }$ , given data $\mathbf { D } _ { S }$ :

$$
\begin{array} { r c l } { \left[ \boldsymbol { \Sigma } _ { C } ^ { ( S ) } , \boldsymbol { \Sigma } _ { m } ^ { ( S ) } | \mathbf { D } _ { S } \right] } & { \propto } & { \mathcal { L } \left( \boldsymbol { \Sigma } _ { m } ^ { ( S ) } , \boldsymbol { \Sigma } _ { C } ^ { ( S ) } ; \mathbf { D } _ { S } \right) \left[ \boldsymbol { \Sigma } _ { C } ^ { ( S ) } , \boldsymbol { \Sigma } _ { m } ^ { ( S ) } \right] , \quad \mathrm { i . e . } } \\ { \left[ \boldsymbol { \Sigma } _ { C } ^ { ( S ) } , \boldsymbol { \Sigma } _ { m } ^ { ( S ) } | \mathbf { D } _ { S } \right] } & { \propto } & { \displaystyle \frac { 1 } { \left( 2 \pi \right) ^ { \frac { d p } { 2 } } \left| \boldsymbol { \Sigma } _ { C } ^ { ( S ) } \right| ^ { \frac { p } { 2 } } \left| \boldsymbol { \Sigma } _ { m } ^ { ( S ) } \right| ^ { \frac { d } { 2 } } } \times } \\ & & { \displaystyle \exp \left[ - \frac { 1 } { 2 } t r \left\{ ( \boldsymbol { \Sigma } _ { m } ^ { ( S ) } ) ^ { - 1 } ( \mathbf { D } _ { S } ) ( \boldsymbol { \Sigma } _ { C } ^ { ( S ) } ) ^ { - 1 } ( \mathbf { D } _ { S } ) ^ { T } \right\} \right] \left| \boldsymbol { \Sigma } _ { m } ^ { ( S ) } \right| ^ { - \frac { d } { 2 } - 1 } , } \end{array}\tag{9.1}
$$

—using prior on $\pmb { \Sigma } _ { m } ^ { ( S ) }$ to be $\pi _ { 0 } ( \Sigma _ { m } ^ { ( S ) } ) = \left| \Sigma _ { m } ^ { ( S ) } \right| ^ { \alpha }$ where $\alpha = - \frac { d } { 2 } - 1$ , and —using prior on $\Sigma _ { C } ^ { ( S ) }$ to be uniform.

Then marginalising this joint posterior over $\pmb { \Sigma } _ { m } ^ { ( S ) }$ , we get:

$$
\begin{array} { c } { { \displaystyle \pi \left( \mathbf { \boldsymbol { \Sigma } } _ { C } ^ { ( S ) } | \mathbf { D } _ { S } \right) \propto } } \\ { { \displaystyle \frac { 1 } { \left| \mathbf { \boldsymbol { \Sigma } } _ { C } ^ { ( S ) } \right| ^ { \frac { p } { 2 } } } \times \int \frac { \left| \mathbf { \boldsymbol { \Sigma } } _ { m } ^ { ( S ) } \right| ^ { - \frac { d } { 2 } - 1 } } { \left| \mathbf { \boldsymbol { \Sigma } } _ { m } ^ { ( S ) } \right| ^ { \frac { d } { 2 } } } \exp \left[ - \frac { 1 } { 2 } t r \left\{ ( \mathbf { \boldsymbol { \Sigma } } _ { m } ^ { ( S ) } ) ^ { - 1 } \mathbf { D } _ { S } ( \mathbf { \boldsymbol { \Sigma } } _ { C } ^ { ( S ) } ) ^ { - 1 } ( \mathbf { D } _ { S } ) ^ { T } \right\} \right] d ( \mathbf { \boldsymbol { \Sigma } } _ { m } ^ { ( S ) } ) } } \end{array}\tag{9.2}
$$

Here $\Sigma _ { m } ^ { ( S ) } \in \mathcal { M } \subseteq \mathbb { R } ^ { ( d \times d ) }$ . Now,

$$
\begin{array} { r l } & { - \mathrm { \normalfont ~ l e t ~ } \pmb { Y } : = ( \pmb { \Sigma } _ { m } ^ { ( S ) } ) ^ { - 1 } \cdot \mathrm { \normalfont ~ T h e n ~ } d ( \pmb { \Sigma } _ { m } ^ { ( S ) } ) = | \pmb { Y } | ^ { - ( d + 1 ) } d \pmb { Y } \mathrm { \normalfont ~ ( M a t h a i ~ a n d ~ P e d e r z o l i , } } \\ & { ~ \mathrm { \normalfont ~ 1 9 9 7 ) } , } \\ & { - \mathrm { \normalfont ~ l e t ~ } \pmb { V } ^ { - 1 } : = \mathbf { D } _ { S } ( \pmb { \Sigma } _ { C } ^ { ( S ) } ) ^ { - 1 } ( \mathbf { D } _ { S } ) ^ { T } , \Longrightarrow \ t r \left[ ( \pmb { \Sigma } _ { m } ^ { ( S ) } ) ^ { - 1 } \mathbf { D } _ { S } ( \pmb { \Sigma } _ { C } ^ { ( S ) } ) ^ { - 1 } ( \mathbf { D } _ { S } ) ^ { T } \right] \equiv } \\ & { ~ t r \left[ \pmb { V } ^ { - 1 } \pmb { Y } \right] \mathrm { \normalfont ~ ( u s i n g ~ c o m m u t a t i v e n e s s ~ o f ~ t r a c e ) } , } \end{array}
$$

so that in Equation 9.2, we get

$$
\begin{array} { r l r } { \pi ( \boldsymbol { \Sigma } _ { \boldsymbol { C } } ^ { ( { \cal S } ) } | \mathbf { D } _ { \cal S } ) } & { \propto } & { \displaystyle \frac { 1 } { | \boldsymbol { \Sigma } _ { \boldsymbol { C } } ^ { ( { \cal S } ) } | ^ { \frac { p } { 2 } } \mathcal { M } } | | \boldsymbol { Y } | ^ { \frac { d } { 2 } } | \boldsymbol { Y } | ^ { \frac { d } { 2 } + 1 } \times \exp [ - \frac { 1 } { 2 } t r \{ \boldsymbol { V } ^ { - 1 } \boldsymbol { Y } \} ] | \boldsymbol { Y } | ^ { - ( d + 1 ) } d \boldsymbol { Y } . } \end{array}\tag{9.3}
$$

The integral in the RHS of Equation 9.3 represents the unnormalised Wishart pdf $W _ { d } ( V , q )$ , over all values of the random matrix $\mathbf { Y }$ , where the scale matrix and degrees of freedom of this $p d f$ are $V$ and $q = d { + } 1$ respectively, i.e. $q > d - 1$ Thus, integral in the RHS of Equation 9.3 is the integral of the unnormalised pdf of $Y \sim W _ { d } ( V , q )$ , over the full support of $\boldsymbol { Y } \left( \equiv \left( \boldsymbol { \Sigma } _ { \boldsymbol { R } } ^ { ( S ) } \right) ^ { - 1 } \right)$ 2

i.e. the integral in the RHS of Equation 9.3 is the normalisation of this $p d f$ :

$$
\begin{array} { r } { 2 ^ { \frac { q d } { 2 } } \Gamma _ { d } \left( \frac { q } { 2 } \right) | V | ^ { \frac { q } { 2 } } \equiv \qquad } \\ { 2 ^ { \frac { ( d + 1 ) ( d ) } { 2 } } \Gamma _ { d } \left( \frac { d + 1 } { 2 } \right) \Big | \left( \mathbf { D } _ { S } ( \Sigma _ { C } ^ { ( S ) } ) ^ { - 1 } ( \mathbf { D } _ { S } ) ^ { T } \right) ^ { - 1 } \Big | ^ { \frac { d + 1 } { 2 } } , } \end{array}
$$

i.e. integral on RHS of Equation 9.3 is proportional to $\Big | \left( \mathbf { D } _ { S } ( \pmb { \Sigma } _ { C } ^ { ( S ) } ) ^ { - 1 } ( \mathbf { D } _ { S } ) ^ { T } \right) ^ { - 1 } \Big | ^ { \frac { d + 1 } { 2 } }$

$$
\mathrm { i . e . } \ \pi \left( \boldsymbol { \Sigma } _ { C } ^ { ( S ) } | \mathbf { D } _ { S } \right) \propto \frac { 1 } { \left| \boldsymbol { \Sigma } _ { C } ^ { ( S ) } \right| ^ { \frac { p } { 2 } } } \Big | \left( \mathbf { D } _ { S } ( \boldsymbol { \Sigma } _ { C } ^ { ( S ) } ) ^ { - 1 } ( \mathbf { D } _ { S } ) ^ { T } \right) ^ { - 1 } \Big | ^ { \frac { d + 1 } { 2 } }\tag{9.4}
$$

Now, if $\mathbf { D } _ { S } ( \pmb { \Sigma } _ { C } ^ { ( S ) } ) ^ { - 1 } ( \mathbf { D } _ { S } ) ^ { T }$ is invertible,

$$
\Big | \left( \mathbf { D } _ { S } ( \Sigma _ { C } ^ { ( S ) } ) ^ { - 1 } ( \mathbf { D } _ { S } ) ^ { T } \right) ^ { - 1 } \Big | ^ { * } = \Big | \mathbf { D } _ { S } ( \Sigma _ { C } ^ { ( S ) } ) ^ { - 1 } ( \mathbf { D } _ { S } ) ^ { T } \Big | ^ { - * } .
$$

It is given that $\Sigma _ { C } ^ { ( S ) }$ is invertible, i.e. $\left( \Sigma _ { C } ^ { ( S ) } \right) ^ { - 1 }$ exists.

The original dataset is examined to discard rows that are linear transformations of each other, leading to data matrix $\mathbf { D } _ { S }$ , no two rows of which are linear transformations of each other

$\implies \mathbf { D } _ { S } ( \Sigma _ { C } ^ { ( S ) } ) ^ { - 1 } ( \mathbf { D } _ { S } ) ^ { T }$ is positive definite, i.e. $\mathbf { D } _ { S } ( \pmb { \Sigma } _ { C } ^ { ( S ) } ) ^ { - 1 } ( \mathbf { D } _ { S } ) ^ { T }$ is invertible, $\begin{array} { r } { \implies \Big | \left( \mathbf { D } _ { S } ( \Sigma _ { C } ^ { ( S ) } ) ^ { - 1 } ( \mathbf { D } _ { S } ) ^ { T } \right) ^ { - 1 } \Big | ^ { ( d + 1 ) / 2 } = \Big | \mathbf { D } _ { S } ( \Sigma _ { C } ^ { ( S ) } ) ^ { - 1 } ( \mathbf { D } _ { S } ) ^ { T } \Big | ^ { - ( d + 1 ) / 2 } . } \end{array}$

Using this in Equation 9.4:

$$
\pi \left( \pmb { \Sigma } _ { C } ^ { ( S ) } | \mathbf { D } _ { S } \right) \propto \Big | \pmb { \Sigma } _ { C } ^ { ( S ) } \Big | ^ { - p / 2 } \Big | \mathbf { D } _ { S } \big ( \pmb { \Sigma } _ { C } ^ { ( S ) } ) ^ { - 1 } \big ( \mathbf { D } _ { S } \big ) ^ { T } \Big | ^ { - ( d + 1 ) / 2 } .\tag{9.5}
$$

## References

Airoldi, E. M. (2007), “Getting Started in Probabilistic Graphical Models,” PLoS Computational Biology, 3(12), e252.

Anandkumar, A., Tan, V., Huang, F., and Willsky, A. (2012), “High-Dimensional Gaussian Graphical Model Selection: Walk Summability and Local Separation Criterion,” JMLR, 13, 2293–2337.

Bandyopadhyay, D., and Canale, A. (2016), “Sparse Multi-Dimensional Graphical Models: A Unified Bayesian Framework,” Journal of Royal Statistical Society Series C, 65(4), 619–640.

Banerjee, S., Basu, A., Bhattacharya, S., Bose, S., Chakrabarty, D., and Mukherjee, S. (2015), “Minimum distance estimation of Milky Way model parameters and related inference,” SIAM/ASA Journal on Uncertainty Quantification, 3(1), 91–115.

Bang, S., Dubickas, A., Koolen, J. H., and Moulton, V. (2015), “There are only finitely many distance-regular graphs of fixed valency greater than two, Advances in Mathematics, 269(Supplement C), 1–55.

Benner, P., Findeisen, R., Flockerzi, D., Reichl, U., and Sundmacher, K. (2014), Large-Scale Networks in Engineering and Life Sciences, Modeling and Simulation in Science, Engineering and Technology, Switzerland: Springer.

Carvalho, C. M., and West, M. (2007), “Dynamic matrix-variate graphical models,” Bayesian Analysis, 2(1), 69–97. URL: https://doi.org/10.1214/07-BA204

Dekker, A. H., and Colbert, B. D. (2004), Network robustness and graph topology„ in Proceedings of the 27th Australasian conference on Computer science, Volume 26, ACSC ’04, Darlinghurst, Australia, Australia: Australian Computer Society, Inc.,, pp. 359–368.

Dettmann, C. P. (2018), “Isolation and Connectivity in Random Geometric Graphs with Self-similar Intensity Measures,” Journal of Statistical Physics, 672, 679–700.

Giles, A. P., Georgiou, O., , and Dettmann, C. P. (2016), “Connectivity of Soft Random Geometric Graphs,” Journal of Statistical Physics, 162(4), 1068– 1083.

Godsil, C. D. (1988), “There are only finitely many distance-regular graphs of fixed valency greater than two,” Combinatorica, 8(4), 333–343.

Gruber, L., and West, M. (2016), “GPU-Accelerated Bayesian Learning and Forecasting in Simultaneous Graphical Dynamic Linear Models,” Bayesian Analysis, 11(1), 125–149.

Hoehndorf, R., Schofield, P. N., and Gkoutos, G. V. (2015), “Analysis of the human diseasome using phenotype similarity between common, genetic, and infectious diseases,” Scientific Reports, 5(10888). URL: http://dx.doi.org/10.1038/srep10888

Imrich, Wilfried; Klavar, S. (2000), Hamming graphs, Product graphs, Wiley-Interscience, New York: Wiley-Interscience Series in Discrete Mathematics and Optimization.

Mass, C. (1987), “Transportation in graphs and the admittance spectrum,” Discrete Applied Mathematics, 16(1), 31–49.

Mathai, A. M., and Pederzoli, G. (1997), “Some Properties of Matrix-Variate Laplace Transforms and Matrix-Variate Whittaker Functions,” Linear Algebra and its Applications, 253, 209–226.

Matusita, K. (1953), “On the estimation by the minimum distance method,” Annals of the Institute of Statistical Mathematics, 5(1), 59–65.

Mcauley, J., and Leskovec, J. (2014), “Discovering social circles in ego networks,” ACM Trans. Knowl. Discov. Data, 8(1). URL: https://doi.org/10.1145/2556612

Menger, K. (1942), “Statistical metrics.,” Proc. Nat. Acad. Sci. USA, 28 (12), 535–537.

Ni, Y., Stingo, F. C., and Baladandayuthapani, V. (2017), “Sparse Multi-Dimensional Graphical Models: A Unified Bayesian Framework,” Journal of the American Statistical Association, 112(518), 779–793.

Penrose, M. (2003), Random Geometric Graphs, Oxford: Oxford Studies in Probability, OUP.

Penrose, M. D. (2016), “Connectivity of Soft Random Geometric Graphs,” Annals of Applied Probability, 26, 986–1028.

Schweizer, B., and Sklar, A. (1983), Probabilistic Metric Spaces, : North-Holland.

Sheskin, D. (2003), Handbook of Parametric and Nonparametric Statistical Procedures: Third Edition, Boca Raton: Taylor and Francis. URL: https://books.google.co.uk/books?id=oX5GnwEACAAJ

Wang, H., and West, M. (2009), “Bayesian analysis of matrix normal graphical models,” Biometrika, 96, 821–834.

Whittaker, J. (2008), Graphical Models in Applied Multivariate Statistics, Switzerland: Wiley.

Zhang, C., Dantu, S., Mitra, D., and Chakrabarty, D. (2025), “Identifying critical residues of a protein using meaningfully-thresholded Random Geometric Graphs,”.

URL: https://arxiv.org/abs/2506.10015

Zhang, C., Grosan, C., and Chakrabarty, D. (2024), “Individualised recovery trajectories of patients with impeded mobility, using distance between probability distributions of learnt graphs,” Artificial Intelligence in Medicine, .