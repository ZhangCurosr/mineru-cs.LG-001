# Finding the Needle in a Haystack: Test-Time Analog Circuit Representation Adaptation for Bayesian Optimization

Fin Amin, Sounak Dutta, and Paul D. Franzon

Abstract—Bayesian optimization (BO) is a sample-efficient framework for analog circuit topology search, where evaluating each candidate topology can require costly simulation. However, representation-based BO methods typically treat circuit embeddings as fixed after encoder training. This creates a mismatch between representation learning and optimization: embeddings learned to encode or reconstruct circuit structure are not necessarily organized according to the figure of merit (FoM) being optimized. This paper introduces Test-Time Analog Representation Adaptation for Bayesian Optimization (TTARO), an online deep-kernel BO framework that adapts circuit representations throughout the search process. Starting from pretrained circuit embeddings, TTARO jointly learns a nonlinear feature transformation and a Gaussian-process surrogate using the FoM labels of the circuits evaluated so far. Following each new evaluation, TTARO updates the representation and surrogate before selecting the next candidate. We compare TTARO with conventional Gaussian Process-based BO over fixed embeddings and with Deep Kernel Learning (DKL), which learns the representation only from the initial evaluated designs and keeps it fixed throughout the remainder of the search. By continually incorporating newly observed FoM labels into representation learning, TTARO aligns the search space with the optimization objective as BO progresses. In our experiments, TTARO reduces regret AUC by 15.2% on average relative to BO and by 20.7% relative to DKL across 40 encoder/kernel/acquisition settings, outperforming prior art in most settings with reductions as large as 46.7%.

Index Terms—Analog circuit design, Bayesian optimization, representation learning, topology synthesis, electronic design automation, online adaptation.

## I. INTRODUCTION AND MOTIVATION

NALOG circuit topology design remains a central challenge in electronic design automation because the search space is discrete, structured, and expensive to evaluate. A candidate topology must often be decoded, sized, simulated, and checked against design constraints before its quality is known. These costs make sample efficiency essential.

Recent learning-assisted approaches encode circuit topologies into latent spaces and perform optimization over the resulting representations [2]–[4]. This strategy makes topology search compatible with continuous black-box optimization methods such as Bayesian optimization (BO). Adjacent work further reflects the broader shift toward representation-centric analog design, including generative topology models such as AnalogGenie and circuit-level representation learning methods such as Ckt2Vec [5], [6]. However, in latent-space BO frameworks, the learned representation is usually fixed before BO begins. As a result, the optimizer must operate in whatever geometry the encoder provides, even when that geometry is poorly aligned with the target FoM.

![](images/219ed969725c2bb32bc9e2ebd3bbbc3818f2fcaabc709ffab1982155a4d99984.jpg)

![](images/fbb507905383b272307011b82121ff8063f8f827fce8f236ad1e426fbc32c1c2.jpg)

![](images/2eae3bf9860edf5fb70fcaeee387e295b5bf5409d3bf4153e9b3a0bb63409e86.jpg)  
Fig. 1. Two-dimensional UMAP [1] visualization analog topology representations before and after TTARO adaptation, with each candidate colored by its FoM percentile. Before adaptation, high- and low-performing circuits are broadly intermixed throughout the representation space. After TTARO, the FoM exhibits substantially stronger spatial organization, with high-performing candidates concentrated within a more coherent region. This objective-aligned geometry allows the Gaussian-process kernel to assign greater similarity to circuits with comparable performance, improving surrogate generalization and enabling the acquisition function to target promising regions more effectively.

This paper introduces Test-Time Analog Representation Adaptation for Bayesian Optimization (TTARO), an online deep-kernel Bayesian optimization framework that learns an objective-aware transformation of pretrained circuit embeddings as figure-of-merit (FoM) observations accumulate. At each BO iteration, TTARO jointly refits a nonlinear feature transformation and Gaussian-process (GP) surrogate using the circuits evaluated so far, then applies an arbitrary acquisition function to select the next candidate from the circuit bank. We compare TTARO against conventional GP-based BO over fixed embeddings, the celebrated Deep-Kernel Learning technique [7], which learns and freezes the transformation using only the initial evaluated designs, and an oracle fixed representation learned using all available FoM labels.

To investigate if TTARO improves optimization, we perform exhaustive experiments across permutations of multiple circuit encoders, kernels and acquisition functions, using two publicly available circuit benchmark search spaces.

The contributions of this paper are:

• We introduce Test-Time Analog Representation Adaptation, an online deep-kernel Bayesian optimization framework that repeatedly learns an objective-aware transformation of pretrained circuit embeddings as new FoM observations become available. As far as we are aware, no prior art has explored this framework for analog circuit topologies.

• We show that TTARO is a general BO framework compatible with expected improvement, upper confidence bound, and Thompson sampling, as well as linear and radial basis function Gaussian-process kernels.

• We conduct a large-scale evaluation comprising 160 principal configurations across two public circuit-topology benchmarks, eight benchmark–encoder pairings, four representation regimes, and five kernel–acquisition settings, with each configuration evaluated over 20 random seeds.

To the best of our knowledge, this is the most comprehensive evaluation ever performed concerning BO for analog circuit topology search.

## II. BACKGROUND AND RELATED WORK

## A. Analog Circuit Topology Search

Analog circuit design is commonly separated into topology selection and device sizing. Device sizing has received substantial attention because, once the topology is fixed, the remaining design variables can be treated as continuous parameters and optimized with simulation-driven methods. Bayesian optimization has been used in this setting to reduce the number of expensive circuit simulations, including neural-network-assisted BO for analog circuit synthesis [8] and transfer-oriented BO for transistor sizing across circuit designs and technology nodes [9]. Related active-learning methods also treat simulation cost as the primary bottleneck and iteratively select circuit candidates whose evaluation is expected to improve the predictive model [10], [11].

Topology search is more difficult because candidate circuits are discrete, structured, and constrained by electrical validity. One line of work represents op-amp topologies at the behavioral level and searches over a lower-dimensional space. Lu et al. encode op-amp behavioral topologies as directed acyclic graphs, learn continuous embeddings with a variational graph autoencoder, and perform topology search in the embedding space using BO before decoding selected points back to circuit topologies [4]. ATOM extends this direction by defining a designer-comprehensible behavior-level op-amp design space, learning continuous topological representations, and using freeze-thaw BO to allocate simulation effort efficiently across candidate topologies [2]. INTO-OA takes a different route: rather than forcing the topology space into a continuous latent space, it applies a Weisfeiler-Lehman (WL) graph kernel inside a GP surrogate and uses gradient information from the surrogate to identify performance-critical substructures and support interpretable topology refinement [12], [13].

Recent work has also framed circuit topology design as graph or sequence generation. CktGNN represents circuits using a two-level graph neural network over a predefined subgraph basis and introduces the Open Circuit Benchmark for reproducible topology and sizing experiments over operational amplifiers [3]. AnalogGenie broadens the generative setting by building a larger analog-circuit topology dataset and representing circuits as pin-level graphs sequentialized through Eulerian circuits, enabling generation beyond a single op-amp topology family [5]. These methods show that learned circuit representations can support topology search, generation, and benchmarking, but they also create a new modeling question: how should an optimizer use a representation once circuit-level FoM observations begin to accumulate?

![](images/815a6ef916ee70d34279a10ef66abecb859f62333f6b32e493332f0d13b483dd.jpg)  
Fig. 2. Two-dimensional UMAP visualization of the DKL representation for Ckt-Bench-301 using CktGNN embeddings. Circuits within the top 2% of the FoM distribution are highlighted in orange, while all remaining candidates are shown in gray. Although DKL introduces localized performance structure, high- and low-performing circuits remain interspersed throughout much of the representation space. Because this representation is learned once and then frozen during DKL, it cannot incorporate the additional FoM evidence collected during Bayesian optimization. This motivates TTARO, which continually revises the surrogate geometry as new circuit evaluations become available.

## B. Circuit Representation Learning

Circuit representations must preserve both graph structure and circuit-relevant behavior. Generic graph encoders can capture connectivity, but analog circuits also contain directionality, functional substructures, device types, and continuous electrical characteristics. CktGNN addresses this by representing each circuit as a composition of subgraphs from a known basis and applying inner and outer GNNs to encode local circuit motifs and global directed message passing [3]. Related circuit-graph learning work has also studied representation quality for circuit equivalence tasks to detect circuit graph isomorphism efficiently [14]. In op-amp topology optimization, graph autoencoders have been used to map discrete behavioral topologies to continuous representations suitable for surrogate modeling and search [2], [4]. Graph-kernel methods provide an alternative representation route by measuring structural similarity directly in graph space, as in INTO-OA’s WL-kernel surrogate [13].

Other methods focus on making circuit embeddings more faithful to analog-domain semantics. Ckt2Vec argues that onehot and text-based device encodings are limited because they do not reflect continuous electrical behavior; it instead extracts frequency-domain features from device I-V curves and combines these electrical features with graph contrastive learning for circuit-level representation learning [6]. AnalogGenie similarly emphasizes representation fidelity, but from a generative perspective: by modeling device pins rather than whole devices as graph nodes, it avoids ambiguous mappings between generated graphs and valid circuit netlists [5]. Together, these works illustrate that the choice of representation is not a neutral preprocessing step. It determines which notions of circuit similarity are visible to downstream prediction, generation, and optimization models.

![](images/1863f1e7e921ce5d6105c147196b1f3d0f344e7e8328a593fbef67e7258f228f.jpg)  
Fig. 3. Overview of Test-Time Analog Representation Adaptation for Bayesian Optimization (TTARO). Candidate circuit topologies are first mapped by a pretrained encoder to fixed representations. At each BO iteration, TTARO uses the observed FoM data to adapt these representations, which are then used by the GP surrogate to compute acquisition scores and select the next circuit for evaluation. The newly observed FoM is added to the dataset, and the representation adaptation and BO procedure are repeated until the evaluation budget is exhausted. TTARO is agnostic to the choice of circuit encoder, acquisition function, and GP kernel.

## C. Bayesian Optimization in Learned Latent Spaces

BO is attractive for analog design because it explicitly targets expensive black-box objectives with a small evaluation budget. A GP surrogate provides both a predictive mean and uncertainty estimate, while an acquisition function such as expected improvement (EI), upper confidence bound (UCB), or Thompson sampling (TS) chooses the next candidate to evaluate [15]–[17]. In analog design, this framework has been used for device sizing and synthesis tasks where each simulation is costly [8], [9]. When the input is a structured object such as a circuit topology, BO is typically applied after mapping the object into a continuous feature or latent space [3], [4].

The effectiveness of this procedure depends strongly on the surrogate’s coordinate system and kernel. Input warping shows that learning transformations of the input space can make GP-based BO more effective on nonstationary objectives [18]. Deep kernel learning (DKL) generalizes this idea by composing a neural feature map with a GP kernel, allowing the model to learn a representation and kernel jointly through the GP marginal likelihood [7]. These methods motivate learned transformations that both represent structured inputs and improve the surrogate used for sequential decision-making.

## D. Adaptive Representations for Bayesian Optimization

Several BO methods explicitly study how representation learning affects optimization. Deep Kernel Bayesian Optimization applies DKL directly inside BO, learning a deep kernel during optimization so that the GP operates in a feature space better suited to the observed objective [19]. SILBO learns a low-dimensional embedding iteratively using both labeled evaluations and unlabeled candidate points proposed by the acquisition function, addressing the difficulty of highdimensional BO when a single initial projection is inadequate [20]. For structured search spaces, contrastive embedding methods use known structural relationships, such as subtree replacements in grammar-defined objects, to learn continuous spaces in which BO can exploit local similarity more effectively [21]. LOCo identifies collisions in learned latent spaces, where points with very different objective values are mapped too close together, and introduces a regularizer to discourage such collisions during BO [22]. CoBO similarly targets latent-space quality by encouraging correlation between latent-space distances and objective-value differences, with additional weighting around promising regions of the search space [23].

Taken together, these methods show that representation quality during BO has been studied across a range of structured and scientific optimization settings. Deep Kernel Bayesian Optimization evaluates adaptive deep kernels on protein engineering, antibody design, and nanophotonics tasks [19]. SILBO studies iterative embedding learning for high-dimensional BO and hyperparameter optimization [20]. Contrastive embedding and CoBO show that latent-space geometry matters in structured search spaces such as molecule design and symbolic expression optimization [21], [23]. LOCo further shows that collisions in learned latent spaces can degrade BO performance and introduces a regularizer for dynamic embedding based BO [22]. LaMBO couples a denoising autoencoder with a multi-task GP head for latent-space BO in smallmolecule and fluorescent-protein design [24], and DKL has also been used as a BO surrogate for chemical reaction outcome optimization [25]. TTARO brings this representationadaptive view to finite-bank analog topology search, where pretrained circuit embeddings are adapted as FoM labels are observed.

## III. PROBLEM FORMULATION AND SETTING

Recent analog topology-search methods often begin by constructing or enumerating a set of candidate circuit topologies, then mapping those candidates into a representation space suitable for learning or optimization. For example, topology-generation and topology-optimization frameworks define behavior-level or graph-based op-amp design spaces, encode each topology as a DAG or circuit graph, and use learned continuous representations to make the discrete topology space searchable by BO [2], [4]. Benchmark-driven methods such as OCB similarly provide finite banks of valid candidate circuits together with graph-based circuit representations and precomputed performance labels [3]. In this paper, we assume this upstream topology-generation or benchmarkconstruction step has already produced a finite candidate bank.

Let

$$
\mathcal { X } = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { N } \}\tag{1}
$$

denote this library of candidate analog circuit topologies. Each $x _ { i }$ may be viewed as a circuit graph, netlist, or topology-level design instance. Each candidate has an associated FoM

$$
y _ { i } = F ( x _ { i } ) ,\tag{2}
$$

which is unknown to the optimizer until that circuit is evaluated. In a deployed circuit-design loop, this evaluation would correspond to simulation, sizing, or measurement. In our benchmark setting, the evaluation is implemented as a label query: selecting a circuit reveals its stored FoM. Given an evaluation budget $B \ll N$ , the objective is to identify

$$
x ^ { \star } = \underset { x _ { i } \in \mathcal { X } } { \arg \operatorname* { m a x } } F ( x _ { i } )\tag{3}
$$

using as few circuit evaluations as possible.

Each candidate is also associated with a fixed initial representation

$$
h _ { i } \in \mathbb { R } ^ { d _ { h } } ,\tag{4}
$$

computed before BO begins. This representation may come from a circuit-specific encoder, a graph autoencoder, or a structural graph feature map. Its purpose is to provide coordinates in which a surrogate model can compare circuits: circuits that are close in representation space are treated as similar by the kernel, and observations from evaluated circuits are generalized to unevaluated circuits through that geometry. However, these representations are typically trained to encode circuit structure, reconstruct graphs, or capture topology semantics, not to organize circuits according to the specific FoM being optimized in the current run.

At BO iteration t, let

$$
\mathcal { T } _ { t } \subseteq \{ 1 , 2 , \dots , N \}\tag{5}
$$

denote the indices of the circuits evaluated so far. The observed dataset is

$$
\mathcal { D } _ { t } = \{ ( h _ { i } , y _ { i } ) \ | \ i \in \mathcal { I } _ { t } \} .\tag{6}
$$

All remaining candidates are unevaluated but already have representations. The optimizer must therefore use the observed FoMs and the representation geometry to decide which unevaluated circuit should be queried next.

Conventional representation-based BO fits a GP surrogate directly over the fixed representations:

$$
f ( h ) \sim \mathcal { G P } \left( m ( h ) , k ( h , h ^ { \prime } ) \right) .\tag{7}
$$

In this setting, the kernel defines circuit similarity entirely through the original representation space. If the encoder places two circuits nearby because they are structurally similar but their FoMs differ substantially, the surrogate can make misleading predictions; conversely, high-performing circuits that are far apart in the initial space may not share useful statistical strength. TTARO addresses this mismatch by learning a FoMinformed transformation of the fixed input representations throughout the BO process, so that the surrogate geometry becomes increasingly aligned with the objective as circuit evaluations accumulate.

## IV. TEST-TIME ANALOG REPRESENTATION ADAPTATION

TTARO adapts a fixed circuit representation during BO by learning a feature map from the FoM observations collected so far. Algorithm 1 summarizes the full loop: TTARO normalizes the candidate representation bank, initializes an evaluated set, repeatedly fits a deep-kernel surrogate, scores unevaluated candidates, and adds the newly evaluated circuit to the observed dataset. At iteration t, each initial circuit representation is first normalized feature-wise over the candidate bank:

$$
\bar { h } _ { i , q } = \frac { h _ { i , q } - h _ { q } ^ { \operatorname* { m i n } } } { h _ { q } ^ { \operatorname* { m a x } } - h _ { q } ^ { \operatorname* { m i n } } } ,\tag{8}
$$

where $h _ { q } ^ { \operatorname* { m i n } }$ and $h _ { q } ^ { \operatorname* { m a x } }$ are computed over all candidates in $\mathcal { X } .$ TTARO then maps the normalized representation into a taskadapted latent space:

$$
z _ { i } ^ { ( t ) } = \phi _ { \theta _ { t } } ( \bar { h } _ { i } ) , \qquad z _ { i } ^ { ( t ) } \in \mathbb { R } ^ { d _ { z } } .\tag{9}
$$

Here, $\theta _ { t }$ denotes the feature-map parameters learned from the FoM observations available at iteration t.

Algorithm 1 Test-Time Analog Circuit   
Representation Adaptation (TTARO)   
1: Input: Candidate representations $\{ h _ { i } \} _ { i = 1 } ^ { N }$   
2: Input: Initial sample size $n _ { 0 }$ and evaluation budget B   
3: Input: Acquisition function a   
4: Normalize the candidate representation bank   
5: Randomly select an initial index set $\mathcal { I } _ { n _ { 0 } }$ with $| \mathcal { I } _ { n _ { 0 } } | = n _ { 0 }$   
6: Evaluate the circuits in $\mathcal { I } _ { n _ { 0 } }$   
7: Construct the initial dataset $\mathcal { D } _ { n _ { 0 } }$   
8: for $t = n _ { 0 }$ to $B - 1$ do   
9: Initialize the feature map and exact GP   
10: Train the feature map and GP using $\mathcal { D } _ { t }$   
11: Compute acquisition scores for all i $\notin \mathcal { T } _ { t }$   
12: Select $i _ { t + 1 } = \arg \operatorname* { m a x } _ { i \notin \mathcal { T } _ { t } } a _ { t } ( h _ { i } )$   
13: Evaluate circuit $i _ { t + 1 }$ and observe $y _ { i _ { t + 1 } }$   
14: ${ \mathcal { T } } _ { t + 1 } = { \mathcal { T } } _ { t } \cup \left\{ i _ { t + 1 } \right\}$   
15: $\mathcal { D } _ { t + 1 } = \mathcal { D } _ { t } \cup \{ ( \bar { h } _ { i _ { t + 1 } } , y _ { i _ { t + 1 } } ) \}$   
16: end for   
17: return The best evaluated circuit in $\mathcal { T } _ { B }$

![](images/742b4e36b04d842450b4d41463542e4e3495db6b8c3afd6520b48c3b891e03fa.jpg)  
Fig. 4. Two-dimensional UMAP visualization of analog topology representations during TTARO adaptation on Ckt-Bench-301 using CktGNN embeddings. Each candidate is colored by its FoM percentile. Panel subtitles report Top-2% KNN@5, the fraction of five nearest representation-space neighbors of top-2% FoM circuits that are also top-2% circuits. As TTARO progresses, high-FoM candidates become more locally concentrated, indicating stronger objective-aligned representation geometry.

The feature map used in our experiments is a two-layer multilayer perceptron:

$$
\phi _ { \theta _ { t } } ( \bar { h } ) = W _ { 2 } \mathrm { D r o p o u t } \left( \mathrm { R e L U } \left( W _ { 1 } \bar { h } + b _ { 1 } \right) \right) + b _ { 2 } .\tag{10}
$$

The hidden layer has width 128, and the transformed representation has dimension $d _ { z } = 1 6 .$

A GP surrogate is defined over the transformed representations:

$$
f ( h ) \sim \mathcal { G P } \left( m _ { \eta _ { t } } \left( \phi _ { \theta _ { t } } ( \bar { h } ) \right) , k _ { \psi _ { t } } \left( \phi _ { \theta _ { t } } ( \bar { h } ) , \phi _ { \theta _ { t } } ( \bar { h } ^ { \prime } ) \right) \right) .\tag{11}
$$

The experiments use either a scaled linear kernel,

$$
k _ { \psi _ { t } } ^ { \mathrm { l i n e a r } } ( z , z ^ { \prime } ) = \sigma _ { f , t } ^ { 2 } z ^ { \top } z ^ { \prime } ,\tag{12}
$$

or a scaled radial basis function (RBF) kernel,

$$
k _ { \psi _ { t } } ^ { \mathrm { R B F } } ( z , z ^ { \prime } ) = \sigma _ { f , t } ^ { 2 } \exp \left( - \frac { 1 } { 2 } \sum _ { q = 1 } ^ { d _ { z } } \frac { ( z _ { q } - z _ { q } ^ { \prime } ) ^ { 2 } } { \ell _ { t , q } ^ { 2 } } \right) .\tag{13}
$$

The parameters $\psi _ { t }$ include the kernel output scale, noise term, and, for the RBF kernel, the length-scale parameters.

## A. Joint Representation and Surrogate Learning

Lines 9–10 of Algorithm 1 correspond to the surrogatefitting step. At BO iteration t, TTARO trains the featuremap parameters $\theta _ { t }$ together with the GP hyperparameters $\psi _ { t }$ and mean parameters $\eta _ { t }$ using the evaluated dataset $\mathcal { D } _ { t }$ . Let $T _ { t } = \{ i _ { 1 } , \dots , i _ { n _ { t } } \}$ denote the evaluated circuit indices. The transformed training representations are collected as

$$
\mathbf { Z } _ { t } = \left[ z _ { i _ { 1 } } ^ { ( t ) } , z _ { i _ { 2 } } ^ { ( t ) } , \ldots , z _ { i _ { n _ { t } } } ^ { ( t ) } \right] ^ { \top } .\tag{14}
$$

The observed FoMs are standardized to form $\tilde { \mathbf { y } } _ { t }$ . The feature map and GP are fit by minimizing the negative log marginal likelihood:

$$
\mathcal { L } _ { t } = \frac { 1 } { 2 } \tilde { \mathbf { y } } _ { t } ^ { \top } \mathbf { K } _ { t } ^ { - 1 } \tilde { \mathbf { y } } _ { t } + \frac { 1 } { 2 } \log \left| \mathbf { K } _ { t } \right| + \frac { n _ { t } } { 2 } \log 2 \pi .\tag{15}
$$

The covariance matrix $\mathbf { K } _ { t }$ is computed from the transformed representations in $\mathbf { Z } _ { t }$ . Its entries are

$$
{ \bf K } _ { t } [ a , b ] = k _ { \psi _ { t } } \left( z _ { i _ { a } } ^ { ( t ) } , z _ { i _ { b } } ^ { ( t ) } \right) + \sigma _ { n , t } ^ { 2 } { \bf 1 } [ a = b ] .\tag{16}
$$

Training uses only FoM observations from circuits already selected by BO. After fitting, TTARO applies the learned feature map to the full candidate bank, computes the GP posterior over unevaluated circuits, and evaluates the acquisition function to select the next circuit.

## B. Online Representation Adaptation

Lines 11–15 of Algorithm 1 describe the online BO step. After fitting the deep-kernel surrogate, an acquisition function $a _ { t }$ is evaluated for every unevaluated candidate. The next circuit is selected according to

$$
i _ { t + 1 } = \underset { i \notin \mathbb { Z } _ { t } } { \arg \operatorname* { m a x } } a _ { t } ( h _ { i } ) .\tag{17}
$$

The selected circuit is evaluated, and its FoM is added to the observed set:

$$
\begin{array} { r } { \mathcal { T } _ { t + 1 } = \mathcal { T } _ { t } \cup \{ i _ { t + 1 } \} . } \end{array}\tag{18}
$$

After the update in Line 15, the loop returns to Line 9: TTARO retrains the feature map and GP using all FoM observations accumulated so far. The procedure is agnostic to the acquisition function used to score unevaluated candidates.

## C. Distinction from Deep Kernel Learning

The DKL baseline uses the same deep-kernel surrogate structure, with the feature map trained only on the initial evaluated dataset. This produces an initial transformed representation bank

$$
\mathcal { Z } ^ { ( 0 ) } = \left\{ \phi _ { \theta _ { 0 } } ( \bar { h } _ { i } ) \right\} _ { i = 1 } ^ { N } .\tag{19}
$$

This transformed representation bank remains fixed for the rest of the optimization process. Additional FoM observations

![](images/fdb35914ef9b2ac970fbfdbd89d246ce44876a394c7b58910290a9ce27faaf10.jpg)  
Fig. 5. Best observed FoM as a function of BO budget on Ckt-Bench-301 using CktGNN representations and a linear kernel. The acquisition functions are (a) upper confidence bound (UCB), (b) expected improvement (EI), and (c) Thompson sampling (TS). The curves compare GP, GP-oracle, DKL, and TTARO, while the shaded regions indicate one standard deviation across runs.

update the GP posterior, while the representation supplied to the GP stays unchanged.

TTARO learns a sequence of feature maps

$$
\phi _ { \theta _ { 0 } } , \phi _ { \theta _ { 1 } } , . . . , \phi _ { \theta _ { B - 1 } } ,\tag{20}
$$

using the growing set of evaluated circuits. The comparison between DKL and TTARO therefore isolates the effect of updating the representation used by the surrogate during the BO process.

## D. On the Importance of Aligned Kernels

The central assumption behind GP-based BO is that the kernel encodes a useful notion of similarity for the objective being modeled. In this work, the relevant objective is the circuit FoM. The kernel should therefore assign high similarity to circuits with similar FoM values and lower similarity to circuits with substantially different FoM values.

Equations (9)–(13) show how TTARO changes the covariance structure used by the GP. Substituting the learned representation $z _ { i } ^ { ( t ) } = \phi _ { \theta _ { t } } ( \bar { \bar { h } } _ { i } )$ into the linear kernel gives

$$
k _ { \psi _ { t } } ^ { \mathrm { l i n e a r } } ( x _ { i } , x _ { j } ) = \sigma _ { f , t } ^ { 2 } \phi _ { \theta _ { t } } ( \bar { h } _ { i } ) ^ { \top } \phi _ { \theta _ { t } } ( \bar { h } _ { j } ) .\tag{21}
$$

Thus, the linear-kernel similarity between two circuits is the scaled inner product of their TTARO-adapted representations. Updating $\theta _ { t }$ rotates, rescales, and reshapes the feature coordinates that determine this inner product.

For the RBF kernel, the same substitution gives

$$
k _ { \psi _ { t } } ^ { \mathrm { R B F } } ( x _ { i } , x _ { j } ) = \sigma _ { f , t } ^ { 2 } \exp \left( - \frac { 1 } { 2 } \sum _ { q = 1 } ^ { d _ { z } } \frac { \left( \phi _ { \theta _ { t } , q } ( \bar { h } _ { i } ) - \phi _ { \theta _ { t } , q } ( \bar { h } _ { j } ) \right) ^ { 2 } } { \ell _ { t , q } ^ { 2 } } \right)\tag{22}
$$

where $\phi _ { { \theta _ { t } } , q } ( \cdot )$ denotes the qth coordinate of the learned representation. In this case, updating $\theta _ { t }$ changes the pairwise distances that appear inside the exponential. Circuits placed closer together in the adapted space receive larger covariance, while circuits separated along dimensions with small length scales receive smaller covariance.

These pairwise similarities enter the GP through the training covariance matrix. For the linear kernel, Eq. (16) becomes

$$
\mathbf { K } _ { t } ^ { \mathrm { l i n e a r } } [ a , b ] = \sigma _ { f , t } ^ { 2 } \phi _ { \theta _ { t } } ( \bar { h } _ { i _ { a } } ) ^ { \top } \phi _ { \theta _ { t } } ( \bar { h } _ { i _ { b } } ) + \sigma _ { n , t } ^ { 2 } \mathbf { 1 } [ a = b ] .\tag{23}
$$

For the RBF kernel, the corresponding entry is

$$
\begin{array} { l } { { \displaystyle { \bf K } _ { t } ^ { \mathrm { R B F } } [ a , b ] = \sigma _ { f , t } ^ { 2 } \exp \left( - \frac { 1 } { 2 } \sum _ { q = 1 } ^ { d _ { z } } \frac { \Delta _ { a b q , t } ^ { 2 } } { \ell _ { t , q } ^ { 2 } } \right) } } \\ { { \displaystyle ~ + \sigma _ { n , t } ^ { 2 } { \bf 1 } [ a = b ] } , } \end{array}\tag{24}
$$

where

$$
\Delta _ { a b q , t } = \phi _ { \theta _ { t } , q } ( \bar { h } _ { i _ { a } } ) - \phi _ { \theta _ { t } , q } ( \bar { h } _ { i _ { b } } ) .\tag{25}
$$

Equations (23) and (24) make the dependence on $\theta _ { t }$ explicit. The learned feature map determines every off-diagonal entry of $\mathbf { K } _ { t }$ , which means it controls how FoM observations from one evaluated circuit influence the inferred FoM of other evaluated circuits.

The same dependence appears in predictions for unevaluated candidates. For an unevaluated circuit $x _ { j }$ , define the crosscovariance vector

$$
\mathbf { k } _ { t , j } = \left[ k _ { \psi _ { t } } ( x _ { j } , x _ { i _ { 1 } } ) , \ldots , k _ { \psi _ { t } } ( x _ { j } , x _ { i _ { n _ { t } } } ) \right] ^ { \top } .\tag{26}
$$

For the linear kernel, its ath entry is

$$
\mathbf { k } _ { t , j } ^ { \mathrm { l i n e a r } } [ a ] = \sigma _ { f , t } ^ { 2 } \phi _ { \theta _ { t } } ( \bar { h } _ { j } ) ^ { \top } \phi _ { \theta _ { t } } ( \bar { h } _ { i _ { a } } ) .\tag{27}
$$

For the RBF kernel, its ath entry is

$$
\mathbf { k } _ { t , j } ^ { \mathrm { R B F } } [ a ] = \sigma _ { f , t } ^ { 2 } \exp \left( - \frac { 1 } { 2 } \sum _ { q = 1 } ^ { d _ { z } } \frac { \left( \phi _ { \theta _ { t } , q } ( \bar { h } _ { j } ) - \phi _ { \theta _ { t } , q } ( \bar { h } _ { i _ { a } } ) \right) ^ { 2 } } { \ell _ { t , q } ^ { 2 } } \right)\tag{28}
$$

The posterior mean and variance used by the acquisition function are

$$
\mu _ { t } ( x _ { j } ) = m _ { t } ( x _ { j } ) + \mathbf { k } _ { t , j } ^ { \top } \mathbf { K } _ { t } ^ { - 1 } \left( \mathbf { y } _ { t } - \mathbf { m } _ { t } \right) ,\tag{29}
$$

$$
\sigma _ { t } ^ { 2 } ( x _ { j } ) = k _ { \psi _ { t } } ( x _ { j } , x _ { j } ) - \mathbf { k } _ { t , j } ^ { \top } \mathbf { K } _ { t } ^ { - 1 } \mathbf { k } _ { t , j } .\tag{30}
$$

The prediction for $x _ { j }$ is therefore controlled by two learned similarity objects: $\mathbf { K } _ { t }$ , which describes similarity among evaluated circuits, and $\mathbf { k } _ { t , j }$ , which describes similarity between the unevaluated candidate and the evaluated set. Updating $\theta _ { t }$ changes both objects. This directly changes the posterior mean, the posterior variance, and the acquisition values used to select the next circuit. TTARO uses the accumulating FoM labels to continually revise this kernel geometry, making the surrogate increasingly aligned with the objective being optimized.

![](images/5c149878e31fd3edcedfabc272a0490772db4dcb3ce8ae8b735b4a4ead51516d.jpg)  
(a) Ckt-Bench-101.

![](images/02275714263411681c4cc82b5ff01ca77d206a8d79590e1c96e74c7bddc4d029.jpg)  
(b) Ckt-Bench-301.  
Fig. 6. Mean Regret AUC Across Kernel and Acquisition-Function Configurations. Lower values indicate more sample-efficient optimization. Error bars denote the standard error of the mean across encoders.

## V. EXPERIMENTAL SETUP

## A. Benchmark Search Spaces

We evaluate TTARO on the two operational-amplifier topology libraries from the Open Circuit Benchmark (OCB) [3]. Ckt-Bench-101 contains 10,000 valid circuit candidates, while Ckt-Bench-301 contains 50,000 candidates. Each candidate is associated with circuit measurements, including gain, phase margin, bandwidth, validity, and FoM.

We use the default OCB FoM definition so that the reported results remain directly comparable with future work on the same benchmark. The FoM is computed as a weighted combination of normalized gain, phase margin, and bandwidth:

$$
\mathrm { F o M = 1 . 2 { \frac { | G a i n | } { 1 0 0 } } + 1 . 6 { \frac { P M } { - 9 0 } } + 1 0 { \frac { | B W | } { 1 0 ^ { 9 } } } . }\tag{31}
$$

During BO, selecting a candidate reveals its stored FoM. This finite-bank label-query setting allows all methods to be compared using the same candidate set, objective values, and evaluation budget.

## B. Circuit Representations

For each benchmark candidate $x _ { i } .$ TTARO takes a fixed vector representation $h _ { i }$ as input. The reported representation sets are dataset-specific. For Ckt-Bench-101, we evaluate CktGNN, D-VAE, D-VAE-GCN, and WL representations. For Ckt-Bench-301, we evaluate CktGNN, DAGNN, D-VAE, and D-VAE-GCN representations.

CktGNN is the circuit-specific encoder proposed with OCB; it represents a circuit using a predefined subgraph basis and applies a two-level GNN to encode local substructures and global circuit connectivity [3]. DAGNN provides a directedacyclic-graph neural encoder baseline [26]. D-VAE and D-VAE-GCN provide graph autoencoding baselines, with D-VAE-GCN replacing the base message-passing component with graph-convolutional layers [27], [28]. The WL representation is a fixed structural graph feature map rather than a learned neural embedding [12].

The learned encoders produce 66-dimensional vectors, giving 10000 × 66 representation banks on Ckt-Bench-101 and 50000×66 representation banks on Ckt-Bench-301 for the corresponding encoder/dataset pairs. The WL representation used for Ckt-Bench-101 has 1,838 dimensions. This representation set tests TTARO across compact learned circuit embeddings and a substantially higher-dimensional structural feature map.

## C. Compared Methods

We compare methods that differ in how the representation used by the GP surrogate is treated during BO. All methods start from the same pretrained circuit embeddings and operate over the same finite candidate bank.

The first baseline is a fixed-representation Gaussian process (GP)–the technique used by the authors of OCB. In this setting, the original embedding $h _ { i }$ is used directly as the GP input for every candidate $x _ { i } .$ . As new FoM labels are observed, only the GP posterior is updated; the representation itself remains unchanged.

The second baseline is DKL [7]. The DKL feature map is trained using only the initial evaluated set, producing transformed representations $\phi _ { \theta _ { 0 } } ( h _ { i } )$ . For parity, we implement DKL’s feature map identically to TTARO via equation 10. This transformation is then frozen and applied to the full candidate bank for the remainder of the BO run. Subsequent evaluations update the GP posterior, but do not update the feature map.

TTARO uses the same deep-kernel surrogate structure, but keeps representation learning coupled to the online search process. After each new circuit evaluation, TTARO retrains the feature map and GP using all FoM observations collected so far. Thus, the representation used by the surrogate can adapt as the evaluated set grows.

The oracle representation (GP-oracle) is included as an upper-reference condition. It trains the feature map using all FoM labels in the candidate bank before BO begins, including labels that would not be available in a real search. This condition is not a deployable optimization method; it measures how much improvement could be expected if the representation were fully aligned with the objective.

## D. Optimization Protocol and Evaluation Metrics

All methods are evaluated under the same finite-bank BO protocol. For a given benchmark, encoder, acquisition function, and random seed, the methods begin from the same initial evaluated circuit set. At each BO iteration, the surrogate is fit using the FoM labels observed so far, the acquisition function is evaluated over the remaining unevaluated candidates, and the selected candidate’s stored FoM is revealed and added to the observed set. Except for the oracle representation condition, no method uses FoM labels from unevaluated candidates when fitting its representation or surrogate. We evaluate two GP kernels, a linear kernel and an RBF kernel, together with five kernel/acquisition settings: Linear/EI, Linear/TS, Linear/UCB, RBF/EI, and RBF/UCB.

TABLE I  
FULL CONFIGURATION-LEVEL BAYESIAN OPTIMIZATION PERFORMANCE ON CKT-BENCH-101. EACH ROW CORRESPONDS TO ONE ENCODER, KERNEL,AND ACQUISITION-FUNCTION SETTING. REGRET AUC IS REPORTED AS MEAN ± STANDARD ERROR; LOWER IS BETTER. THE BEST AND SECOND-BESTREGRET AUC VALUES WITHIN EACH ROW ARE SHOWN IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td rowspan="2"></td><td rowspan="2">Kernel / Acquisition</td><td colspan="4">Regret AUC (↓)</td><td colspan="2">TTARO vs. GP (%)</td></tr><tr><td>GP</td><td>GP-oracle</td><td>DKL</td><td>TTARO</td><td>Regret AUC Reduction</td><td>FoM Improvement</td></tr><tr><td></td><td>Linear / EI</td><td> $2 9 7 7 0 . 0 \pm 3 5 5 . 2$ </td><td> $3 0 9 9 8 . 6 \pm 2 3 8 9 . 1$ </td><td> $3 0 9 5 5 . 8 \pm 7 0 7 . 8$ </td><td> $\mathbf { 2 4 2 1 6 . 1 \pm 1 3 3 5 . 8 }$ </td><td>+18.7%</td><td>+11.8%</td></tr><tr><td></td><td>Linear / TS</td><td> $2 7 1 3 3 . 8 \pm 1 0 2 9 . 9$ </td><td> $\mathbf { 2 4 2 0 1 . 1 \pm 7 2 5 . 0 }$ </td><td> $2 9 7 4 7 . 8 \pm 7 3 0 . 4$ </td><td> $2 4 2 3 3 . 4 \pm 1 3 0 8 . 2$ </td><td>+10.7%</td><td>+4.5%</td></tr><tr><td>CGN</td><td>Linear / UCB</td><td> $\underline { { 2 7 4 7 2 . 7 } } \pm 7 8 1 . 3$ </td><td> $2 8 5 0 8 . 1 \pm 2 6 7 8 . 4$ </td><td> $3 0 5 6 3 . 4 \pm 1 1 5 4 . 1$ </td><td> $\mathbf { 2 4 1 6 1 . 9 \pm 1 4 3 3 . 6 }$ </td><td>+12.1%</td><td>+12.4%</td></tr><tr><td></td><td>RBF / EI</td><td> $2 7 1 8 9 . 0 \pm 1 2 9 7 . 9$ </td><td> $\mathbf { 2 5 9 4 5 . 1 \pm 1 0 0 7 . 1 }$ </td><td> $3 0 0 5 6 . 9 \pm 5 0 0 . 5$ </td><td> $2 6 3 1 1 . 4 \pm 1 1 1 5 . 7$ </td><td>+3.2%</td><td>+3.3%</td></tr><tr><td></td><td>RBF / UCB</td><td> $\mathbf { 2 4 0 1 3 . 6 \pm 1 4 3 2 . 6 }$ </td><td> $3 0 2 4 6 . 8 \pm 2 8 3 5 . 0$ </td><td> $2 8 9 6 7 . 5 \pm 8 0 4 . 6$ </td><td> $\overline { { 2 5 3 3 3 . 4 \pm 1 1 0 7 . 9 } }$ </td><td>-5.5%</td><td>-1.4%</td></tr><tr><td></td><td>Linear / EI</td><td> $2 8 1 6 0 . 2 \pm 2 0 6 . 1$ </td><td> $2 6 8 1 4 . 7 \pm 5 9 1 . 0$ </td><td> $3 0 7 2 4 . 6 \pm 7 0 6 . 3$ </td><td> $\mathbf { 1 5 4 1 2 . 0 \pm 4 5 5 6 . 1 }$ </td><td>+45.3%</td><td>+26.6%</td></tr><tr><td></td><td>Linear / TS</td><td> $2 9 4 9 8 . 1 \pm 3 6 1 . 6$ </td><td> $2 5 1 0 1 . 0 \pm 1 3 0 5 . 7$ </td><td> $2 9 7 6 6 . 7 \pm 4 9 2 . 2$ </td><td> $\mathbf { 2 4 4 4 5 . 5 \pm 2 7 8 5 . 2 }$ </td><td>+17.1%</td><td>+14.1%</td></tr><tr><td>TM</td><td>Linear / UCB</td><td> $2 8 5 0 2 . 1 \pm 2 4 2 . 4$ </td><td> $\underline { { 2 4 7 9 5 . 8 \pm 1 0 6 9 . 7 } }$ </td><td> $2 9 3 9 4 . 9 \pm 5 5 6 . 0$ </td><td> $\mathbf { 2 3 4 0 0 . 4 \pm 5 3 9 8 . 6 }$ </td><td>+17.9%</td><td>+12.8%</td></tr><tr><td></td><td>RBF / EI</td><td> $3 2 1 2 0 . 2 \pm 2 8 1 . 4$ </td><td> $\underline { { 2 7 4 5 4 . 2 \pm 1 0 0 0 . 2 } }$ </td><td> $2 9 3 8 1 . 5 \pm 5 7 3 . 3$ </td><td> $\mathbf { 2 1 8 6 5 . 6 \pm 4 6 3 4 . 5 }$ </td><td>+31.9%</td><td>+14.8%</td></tr><tr><td></td><td>RBF / UCB</td><td> $3 0 1 3 7 . 7 \pm 6 9 0 . 8$ </td><td> $3 1 9 5 7 . 3 \pm 2 5 5 6 . 4$ </td><td> $3 0 3 9 8 . 8 \pm 5 6 5 . 5$ </td><td> $\mathbf { 2 2 5 1 0 . 5 \pm 4 9 0 5 . 0 }$ </td><td>+25.3%</td><td>+13.5%</td></tr><tr><td></td><td>Linear / EI</td><td> $5 1 4 7 6 . 6 \pm 2 4 7 . 2$ </td><td> $\mathbf { 1 8 3 4 9 . 3 \pm 1 6 7 1 . 7 }$ </td><td> $3 6 9 7 5 . 5 \pm 1 6 7 0 . 0$ </td><td> $\underline { { 2 7 4 4 9 . 0 \pm 1 8 9 2 . 5 } }$ </td><td>+46.7%</td><td>+30.8%</td></tr><tr><td></td><td>Linear / TS</td><td> $4 1 5 2 5 . 1 \pm 1 3 9 5 . 0$ </td><td> $\mathbf { 1 3 8 0 1 . 4 \pm 1 0 6 7 . 8 }$ </td><td> $3 4 7 3 3 . 7 \pm 1 2 3 2 . 3$ </td><td> $\underline { { 2 8 7 0 2 . 0 \pm 1 5 4 8 . 7 } }$ </td><td>+30.9%</td><td>+6.9%</td></tr><tr><td>DVAE</td><td>Linear / UCB</td><td> $4 9 6 7 6 . 5 \pm 2 4 1 . 2$ </td><td> $\mathbf { 1 4 2 9 6 . 1 \pm 1 1 0 2 . 2 }$ </td><td> $3 3 2 5 7 . 2 \pm 1 0 6 1 . 2$ </td><td> $\underline { { 2 9 0 7 1 . 5 \pm 1 2 0 6 . 4 } }$ </td><td>+41.5%</td><td>+10.4%</td></tr><tr><td></td><td>RBF / EI</td><td> $2 9 5 8 3 . 8 \pm 4 3 2 . 8$ </td><td> $\mathbf { 2 1 1 9 3 . 7 \pm 1 2 8 6 . 4 }$ </td><td> $3 1 8 6 7 . 2 \pm 1 0 8 4 . 0$ </td><td> $3 0 1 3 7 . 0 \pm 1 2 2 1 . 2$ </td><td>-1.9%</td><td>+5.7%</td></tr><tr><td></td><td>RBF / UCB</td><td> $\overline { { 2 9 8 6 2 . 4 \pm 9 4 6 . 9 } }$ </td><td> $\mathbf { 1 7 2 7 2 . 9 \pm 2 2 0 0 . 5 }$ </td><td> $2 7 8 2 4 . 2 \pm 1 6 7 8 . 8$ </td><td>29766.0 ± 1778.6</td><td>+0.3%</td><td>+3.1%</td></tr><tr><td></td><td>Linear / EI</td><td> $3 9 3 8 4 . 0 \pm 9 3 4 . 0$ </td><td> $2 9 9 8 4 . 9 \pm 8 9 5 . 6$ </td><td> $3 1 1 8 6 . 6 \pm 4 8 6 . 6$ </td><td> $\mathbf { 2 9 7 6 6 . 5 \pm 4 3 6 . 9 }$ </td><td>+24.4%</td><td>+2.0%</td></tr><tr><td></td><td>Linear / TS</td><td> $3 1 4 0 8 . 0 \pm 6 2 2 . 8$ </td><td> $\mathbf { 2 8 3 3 8 . 1 \pm 3 1 6 . 7 }$ </td><td> $3 1 1 2 0 . 8 \pm 5 2 2 . 0$ </td><td> $2 9 7 2 6 . 3 \pm 9 3 0 . 2$ </td><td>+5.4%</td><td>+2.2%</td></tr><tr><td></td><td>Linear / UCB</td><td> $3 6 6 0 6 . 0 \pm 9 4 0 . 1$ </td><td> $3 2 4 5 7 . 1 \pm 2 4 7 3 . 3$ </td><td> $3 0 5 1 4 . 9 \pm 4 7 5 . 6$ </td><td> $\mathbf { 2 9 2 9 7 . 6 \pm 4 0 2 . 3 }$ </td><td>+20.0%</td><td>+0.3%</td></tr><tr><td>D-GCN</td><td>RBF / EI</td><td> $3 1 1 0 7 . 1 \pm 4 1 5 . 8$ </td><td> $\mathbf { 2 8 5 4 0 . 1 \pm 8 3 6 . 0 }$ </td><td> $\overline { { 3 1 5 6 1 . 5 \pm 5 6 5 . 2 } }$ </td><td> $3 0 9 7 6 . 7 \pm 4 7 3 . 9$ </td><td>+0.4%</td><td>-0.3%</td></tr><tr><td></td><td>RBF / UCB</td><td> $3 1 9 2 5 . 0 \pm 6 2 4 . 3$ </td><td> $3 7 9 0 1 . 6 \pm 3 9 5 0 . 3$ </td><td> $\mathbf { 3 0 7 8 4 . 3 \pm 9 7 6 . 1 }$ </td><td> $3 0 9 2 8 . 8 \pm 5 5 5 . 2$ </td><td>+3.1%</td><td>+0.4%</td></tr><tr><td>Dataset Average All</td><td></td><td> $3 2 8 2 7 . 6 \pm 1 6 4 5 . 5$ </td><td> $\mathbf { 2 5 9 0 7 . 9 \pm 1 4 0 2 . 7 }$ </td><td> $3 0 9 8 9 . 2 \pm 4 5 9 . 3$ </td><td> $2 6 3 8 5 . 6 \pm 8 8 2 . 5$ </td><td>+19.6%</td><td></td></tr></table>

We evaluate performance using best-so-far FoM and regretbased metrics, which are standard in BO and GP bandit optimization [16], [29]. Let

$$
f ^ { \star } = \operatorname* { m a x } _ { i \in \{ 1 , . . . , N \} } y _ { i }\tag{32}
$$

denote the best FoM in the candidate bank, used only for evaluation. After t circuit evaluations, the best observed FoM is

$$
f _ { t } ^ { + } = \operatorname* { m a x } _ { i \in \mathbb { Z } _ { t } } y _ { i } .\tag{33}
$$

The simple regret at iteration t is

$$
r _ { t } = f ^ { \star } - f _ { t } ^ { + } .
$$

circuits are more locally concentrated in the learned representation. This quantity is used only as a diagnostic for the visualization; the primary optimization metrics are regret AUC and final best-so-far FoM.

(34)

To summarize the full optimization trajectory, we report regret AUC over the evaluation budget. For a budget of B evaluations, regret AUC is computed as the discrete area under the simple-regret curve:

Lower regret indicates that a method has found higherperforming circuits earlier in the search.

We also report Top-2% KNN@5 in the representationgeometry visualization (see Fig. 4) to help interpret the UMAP plots. For each circuit whose FoM is in the top 2% of the candidate bank, Top-2% KNN@5 measures the fraction of its five nearest representation-space neighbors that are also top-2% FoM circuits. Higher values indicate that high-performing

$$
\mathrm { A U C } _ { r } = \sum _ { t = 1 } ^ { B - 1 } { \frac { r _ { t } + r _ { t + 1 } } { 2 } } .\tag{35}
$$

Equivalently, this is the trapezoidal approximation to the regret curve when evaluations are spaced one step apart. Lower regret AUC indicates better sample efficiency across the full BO run. We also report final best-so-far FoM to measure the best circuit found by the end of the budget.

For TTARO, regret AUC reduction over GP is reported as

$$
\Delta _ { \mathrm { A U C } } = \frac { \mathrm { A U C } _ { r } ^ { \mathrm { G P } } - \mathrm { A U C } _ { r } ^ { \mathrm { T T A R O } } } { \mathrm { A U C } _ { r } ^ { \mathrm { G P } } } \times 1 0 0 .\tag{36}
$$

Final FoM improvement over GP is reported as

$$
\Delta _ { \mathrm { F o M } } = \frac { \mathrm { F o M _ { \it B } ^ { T T A R O } - F o M _ { \it B } ^ { G P } } } { \mathrm { F o M _ { \it B } ^ { G P } } } \times 1 0 0 .\tag{37}
$$

## VI. RESULTS AND DISCUSSION

Tables I and II summarize the configuration-level regret AUC results. TTARO improves regret AUC relative to fixedrepresentation GP on both Ckt-Bench-101 and Ckt-Bench-301, indicating that online representation adaptation improves the sample efficiency of BO. On Ckt-Bench-101, TTARO reduces the dataset-average regret AUC from 32827.6 to 26385.6, a 19.6% reduction relative to GP. On Ckt-Bench-301, TTARO reduces the dataset-average regret AUC from 19901.1 to 17480.7, a 12.2% reduction. These gains are consistent with the intended role of TTARO: as FoM observations accumulate, the surrogate can update the kernel geometry used to generalize from evaluated to unevaluated circuits.

TABLE II  
FULL CONFIGURATION-LEVEL BAYESIAN OPTIMIZATION PERFORMANCE ON CKT-BENCH-301. EACH ROW CORRESPONDS TO ONE ENCODER, KERNEL,AND ACQUISITION-FUNCTION SETTING. REGRET AUC IS REPORTED AS MEAN ± STANDARD ERROR; LOWER IS BETTER. THE BEST AND SECOND-BESTREGRET AUC VALUES WITHIN EACH ROW ARE SHOWN IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td rowspan="2"></td><td rowspan="2">Kernel / Acquisition</td><td colspan="4">Regret AUC (↓)</td><td colspan="2">TTARO vs. GP (%)</td></tr><tr><td>GP</td><td>GP-oracle</td><td>DKL</td><td>TTARO</td><td>Regret AUC Reduction</td><td>FoM Improvement</td></tr><tr><td rowspan="5">CGN</td><td>Linear / EI</td><td> $2 2 9 2 4 . 1 \pm 3 7 5 . 6$ </td><td> ${ \bf 1 7 6 6 . 6 \pm 3 1 . 7 }$ </td><td> $2 6 9 0 3 . 2 \pm 5 9 3 . 8$ </td><td> $1 5 1 1 4 . 6 \pm 1 1 1 8 . 4$ </td><td>+34.1%</td><td>+27.0%</td></tr><tr><td>Linear / TS</td><td> $1 8 0 3 5 . 6 \pm 9 7 7 . 6$ </td><td> ${ \bf 1 8 4 8 . 9 \pm 5 6 . 5 }$ </td><td> $2 5 6 6 3 . 6 \pm 4 4 4 . 9$ </td><td> $\underline { { 1 4 5 2 6 . 3 \pm 1 0 4 2 . 7 } }$ </td><td>+19.5%</td><td>+3.9%</td></tr><tr><td>Linear / UCB</td><td> $2 0 3 0 3 . 5 \pm 8 4 7 . 0$ </td><td> ${ \bf 1 7 6 6 . 6 \pm 3 1 . 7 }$ </td><td> $2 5 7 8 6 . 7 \pm 6 3 8 . 9$ </td><td> $\underline { { 1 4 4 3 2 . 1 \pm 1 2 3 2 . 0 } }$ </td><td>+28.9%</td><td>+17.0%</td></tr><tr><td>RBF / EI</td><td> $1 7 4 0 9 . 3 \pm 1 2 4 9 . 0$ </td><td> $\mathbf { 3 4 6 9 . 8 \pm 4 7 8 . 4 }$ </td><td> $2 3 1 7 5 . 6 \pm 5 9 9 . 4$ </td><td> $\underline { { 1 4 0 3 7 . 8 \pm 1 1 7 6 . 5 } }$ </td><td>+19.4%</td><td>+5.9%</td></tr><tr><td>RBF / UCB</td><td> $1 6 8 0 8 . 7 \pm 1 0 8 0 . 1$ </td><td> $\underline { { 1 4 1 4 1 . 3 \pm 2 5 3 4 . 5 } }$ </td><td> $2 2 5 0 9 . 6 \pm 6 2 7 . 1$ </td><td> $\mathbf { 1 1 8 5 1 . 8 \pm 1 0 7 8 . 7 }$ </td><td>+29.5%</td><td>+3.9%</td></tr><tr><td rowspan="5">DAGN</td><td>Linear / EI</td><td> $1 2 1 4 4 . 2 \pm 1 2 5 3 . 1$ </td><td> $\mathbf { 1 9 7 7 . 0 \pm 1 5 3 . 5 }$ </td><td> $1 8 7 6 7 . 5 \pm 1 6 3 9 . 5$ </td><td> $\underline { { 1 1 1 8 3 . 7 } } \pm 1 4 7 2 . 5$ </td><td>+7.9%</td><td>-0.0%</td></tr><tr><td>Linear / TS</td><td> $1 2 7 5 9 . 9 \pm 1 5 0 9 . 4$ </td><td> $\mathbf { 1 8 6 0 . 2 \pm 5 2 . 6 }$ </td><td> $2 1 1 1 5 . 5 \pm 1 0 4 2 . 7$ </td><td> $1 2 0 1 6 . 2 \pm 1 4 8 7 . 8$ </td><td>+5.8%</td><td>+2.0%</td></tr><tr><td>Linear / UCB</td><td> $1 2 3 9 7 . 7 \pm 1 4 1 8 . 3$ </td><td> $\mathbf { 2 0 0 3 . 9 \pm 1 6 1 . 2 }$ </td><td> $1 9 6 8 5 . 3 \pm 1 2 8 9 . 4$ </td><td> $9 9 7 8 . 7 \pm 1 1 9 5 . 3$ </td><td>+19.5%</td><td>+1.8%</td></tr><tr><td>RBF / EI</td><td> $1 5 8 7 4 . 3 \pm 1 2 7 0 . 5$ </td><td> $\mathbf { 3 5 0 2 . 8 \pm 5 2 8 . 8 }$ </td><td> $1 9 4 4 1 . 8 \pm 1 4 1 8 . 0$ </td><td> $\underline { { 1 1 7 4 2 . 7 \pm 1 3 1 5 . 5 } }$ </td><td>+26.0%</td><td>+3.9%</td></tr><tr><td>RBF / UCB</td><td> $1 7 1 5 8 . 5 \pm 1 2 3 7 . 5$ </td><td> $1 1 8 5 7 . 9 \pm 2 1 5 8 . 6$ </td><td> $2 0 4 3 6 . 6 \pm 1 1 7 2 . 2$ </td><td> $\mathbf { 1 1 4 3 5 . 3 \pm 1 4 1 3 . 4 }$ </td><td>+33.4%</td><td>+8.0%</td></tr><tr><td rowspan="5">DVAE</td><td>Linear / EI</td><td></td><td> $\mathbf { 2 2 0 9 7 . 7 \pm 5 9 . 6 }$ </td><td> $2 7 8 2 0 . 9 \pm 1 0 5 1 . 0$ </td><td> $2 3 2 1 5 . 3 \pm 4 7 3 . 9$ </td><td>+2.5%</td><td>+5.6%</td></tr><tr><td>Linear / TS</td><td> $2 3 8 1 5 . 5 \pm 1 2 0 . 5$   $2 3 5 3 7 . 2 \pm 2 1 7 . 9$ </td><td> $\mathbf { 1 9 4 6 3 . 5 \pm 8 2 2 . 6 }$ </td><td> $2 5 4 1 4 . 4 \pm 4 9 7 . 4$ </td><td> $2 2 2 4 8 . 8 \pm 7 7 5 . 2$ </td><td>+5.5%</td><td>+11.1%</td></tr><tr><td>Linear / UCB</td><td> $2 4 3 7 8 . 6 \pm 2 0 2 . 2$ </td><td> $\mathbf { 1 7 7 6 1 . 2 \pm 1 0 7 8 . 4 }$ </td><td> $2 6 0 7 9 . 9 \pm 6 8 2 . 7$ </td><td> $2 3 1 0 4 . 9 \pm 3 1 5 . 3$ </td><td>+5.2%</td><td>+8.5%</td></tr><tr><td>RBF / EI</td><td> $2 3 2 5 7 . 0 \pm 4 0 8 . 0$ </td><td> $\mathbf { 2 1 1 9 6 . 8 \pm 9 4 0 . 5 }$ </td><td> $2 3 3 7 5 . 2 \pm 6 0 6 . 5$ </td><td> $2 2 6 3 8 . 4 \pm 8 5 2 . 9$ </td><td>+2.7%</td><td>+2.4%</td></tr><tr><td>RBF / UCB</td><td> $2 2 1 1 4 . 3 \pm 1 0 4 3 . 6$ </td><td> $\underline { { 2 1 6 4 0 . 2 \pm 1 2 2 2 . 5 } }$ </td><td> $2 3 5 0 6 . 1 \pm 6 5 1 . 2$ </td><td> $\mathbf { 2 1 5 8 3 . 1 \pm 8 9 8 . 6 }$ </td><td>+2.4%</td><td>-2.5%</td></tr><tr><td rowspan="4"></td><td>Linear / EI</td><td> $2 3 0 0 6 . 3 \pm 5 5 . 0$ </td><td> $\mathbf { 2 2 5 9 0 . 7 \pm 7 5 . 4 }$ </td><td> $2 4 3 9 9 . 2 \pm 6 4 4 . 1$ </td><td> $2 3 2 1 2 . 5 \pm 1 4 6 . 3$ </td><td>-0.9%</td><td>+0.1%</td></tr><tr><td>Linear / TS</td><td> $2 3 1 3 1 . 8 \pm 8 1 . 1$ </td><td> $\mathbf { 1 7 4 6 8 . 6 \pm 6 8 3 . 2 }$ </td><td> $2 5 7 5 4 . 1 \pm 6 4 4 . 3$ </td><td> $\underline { { 2 1 4 4 2 . 6 \pm 9 4 1 . 6 } }$ </td><td>+7.3%</td><td>+14.0%</td></tr><tr><td>Linear / UCB</td><td> $2 3 1 1 7 . 0 \pm 1 9 5 . 8$ </td><td> $\mathbf { 2 0 1 2 6 . 8 \pm 1 9 5 4 . 5 }$ </td><td> $2 8 1 6 2 . 3 \pm 1 2 9 1 . 0$ </td><td> $\underline { { 2 1 8 9 5 . 1 \pm 7 9 1 . 1 } }$ </td><td>+5.3%</td><td>+13.9%</td></tr><tr><td>RBF / EI</td><td> $2 3 0 0 2 . 8 \pm 1 0 2 . 2$ </td><td> $2 3 0 9 1 . 9 \pm 1 5 1 . 5$ </td><td> $\mathbf { 2 2 1 9 4 . 6 \pm 6 0 9 . 7 }$ </td><td> $2 2 3 4 8 . 7 \pm 5 7 5 . 2$ </td><td>+2.8%</td><td>+2.8%</td></tr><tr><td>DA-CN</td><td>RBF / UCB</td><td> $2 2 8 4 6 . 5 \pm 8 4 . 4$ </td><td> $2 4 6 6 0 . 5 \pm 8 6 6 . 7$ </td><td> $2 2 4 8 7 . 6 \pm 9 1 4 . 4$ </td><td> $\mathbf { 2 1 6 0 5 . 2 \pm 5 5 2 . 8 }$ </td><td>+5.4%</td><td>+22.3%</td></tr><tr><td>Dataset Average All</td><td></td><td> $1 9 9 0 1 . 1 \pm 9 3 6 . 4 $ </td><td> $\mathbf { 1 2 7 1 4 . 6 \pm 2 0 6 1 . 5 }$ </td><td> $2 3 6 3 4 . 0 \pm 6 2 8 . 4$ </td><td> $\underline { { 1 7 4 8 0 . 7 \pm 1 1 4 9 . 2 } }$ </td><td>+12.2%</td><td>+7.2%</td></tr></table>

regret AUC by 0.9%.

The comparison with DKL highlights why online adaptation matters. On Ckt-Bench-101, DKL improves regret AUC relative to GP, while its final best FoM remains slightly lower than GP. On Ckt-Bench-301, DKL degrades substantially: regret AUC increases from 19901.1 for GP to 23634.0, and final best FoM drops from 152.7 to 134.0. This suggests that a representation learned only from the initial evaluated set can be brittle, especially in the larger and more heterogeneous search space. TTARO avoids this degradation, reducing regret AUC by 14.9% relative to DKL on Ckt-Bench-101 and by 26.0% on Ckt-Bench-301.

Table III reports aggregate best-so-far FoM at fixed fractions of the BO budget. This view complements regret AUC by showing how quickly each method finds high-performing circuits during the search. On Ckt-Bench-101, TTARO reaches 228.6 FoM by 20% of the budget, compared with 210.4 for GP and 221.2 for DKL. By 60% of the budget, TTARO reaches 252.8, already exceeding the final GP value of 247.6. On Ckt-Bench-301, TTARO reaches 152.8 by 60% of the budget, matching the final GP value of 152.7 while still having 40% of the evaluation budget remaining. These budgeted FoM trends show that TTARO’s regret-AUC gains correspond to earlier discovery of high-performing circuits, not only lower integrated regret.

The configuration-level results in Tables I and II show that these aggregate gains are broadly distributed across encoders, kernels, and acquisition functions. TTARO improves regret AUC relative to GP in 37 of the 40 evaluated settings. The three exceptions are CktGNN/RBF/UCB and D-VAE/RBF/EI on Ckt-Bench-101, where TTARO increases regret AUC relative to GP by 5.5% and 1.9%, respectively, and D-VAE-GCN/Linear/EI on Ckt-Bench-301, where TTARO increases

The oracle representation condition provides useful context for interpreting the upper end of representation quality. It trains the feature map using all FoM labels in the candidate bank before BO begins, giving the surrogate access to the global objective landscape. This allows the learned representation to place circuits with similar FoM values closer together and separate circuits with substantially different FoM values before any sequential search decisions are made. As a result, the GP kernel begins with a more objective-aligned covariance structure, so observations from evaluated circuits can be transferred more effectively to unevaluated candidates. Since the oracle representation is trained using FoM labels that are unavailable during a real BO run, it is not a deployable optimization method and should be interpreted only as an

TABLE III  
AGGREGATE BEST-OBSERVED FOM ACROSS CKT-BENCH-101 AND CKT-BENCH-301 OVER THE BO EVALUATION BUDGET. MEAN ± STANDARD ERROR IS REPORTED AT 20% INCREMENTS OF THE BUDGET; HIGHER IS BETTER. THE BEST VALUE WITHIN EACH DATASET AND BUDGET CHECKPOINT IS SHOWN IN BOLD, AND THE SECOND-BEST VALUE IS UNDERLINED.
<table><tr><td>Dataset</td><td>Method</td><td>Best FoM @ 20%</td><td>Best FoM @ 40%</td><td>Best FoM @ 60%</td><td>Best FoM @ 80%</td><td>Best FoM @ 100%</td></tr><tr><td>CKk--nch</td><td>GP</td><td> $2 1 0 . 4 \pm 2 . 0$ </td><td> ${ 2 2 5 . 9 \pm 1 . 5 }$ </td><td> $2 3 1 . 8 \pm { 1 . 6 }$ </td><td> $2 3 7 . 6 \pm 1 . 5$ </td><td> $2 4 7 . 6 \pm 1 . 3$ </td></tr><tr><td>101</td><td>GP-oracle</td><td> $\mathbf { 2 3 1 . 2 \pm 2 . 0 }$ </td><td> $\mathbf { 2 4 9 . 0 \pm 2 . 1 }$ </td><td> $\mathbf { 2 5 5 . 0 \pm 2 . 1 }$ </td><td> $\mathbf { 2 6 1 . 9 \pm 2 . 3 }$ </td><td> $\mathbf { 2 7 1 . 0 \pm 2 . 3 }$ </td></tr><tr><td></td><td>DKL</td><td> ${ 2 2 1 . 2 \pm 1 . 4 }$ </td><td> $2 3 4 . 1 \pm 0 . 8$ </td><td> $2 3 8 . 5 \pm 0 . 8$ </td><td> $2 4 1 . 7 \pm 0 . 8$ </td><td> $2 4 3 . 5 \pm 0 . 8$ </td></tr><tr><td></td><td>TTARO</td><td> ${ 2 2 8 . 6 \pm 1 . 6 }$ </td><td> $\underline { { 2 4 5 . 3 } } \pm 1 . 4$ </td><td> ${ 2 5 2 . 8 \pm 1 . 6 }$ </td><td> ${ \underline { { 2 5 8 . 2 } } } \pm 1 . 8$ </td><td> $2 6 4 . 6 \pm 2 . 0$ </td></tr><tr><td></td><td>GP</td><td> $1 2 4 . 7 \pm 0 . 7$ </td><td> $1 3 2 . 3 \pm { 1 . 0 }$ </td><td> $1 4 1 . 7 \pm 1 . 5$ </td><td> $1 4 7 . 2 \pm 1 . 6$ </td><td> $1 5 2 . 7 \pm 1 . 7$ </td></tr><tr><td>Ck--nch</td><td>GP-oracle</td><td> ${ \bf 1 5 3 . 6 \pm 2 . 0 }$ </td><td> $\mathbf { 1 5 7 . 8 \pm 1 . 9 }$ </td><td> $\mathbf { 1 6 1 . 6 \pm 1 . 9 }$ </td><td> $\mathbf { 1 6 7 . 9 \pm 1 . 8 }$ </td><td> $\mathbf { 1 7 1 . 8 \pm 1 . 8 }$ </td></tr><tr><td>301</td><td>DKL</td><td> $1 1 0 . 7 \pm 1 . 1$ </td><td> $1 2 2 . 2 \pm 0 . 9$ </td><td> $1 2 8 . 5 \pm { 1 . 0 }$ </td><td> $1 3 1 . 9 \pm 1 . 1$ </td><td> $1 3 4 . 0 \pm 1 . 2$ </td></tr><tr><td></td><td>TTARO</td><td> $\underline { { 1 2 7 . 0 \pm 0 . 9 } }$ </td><td> ${ \underline { { 1 4 1 . 9 } } } \pm 1 . 5$ </td><td> ${ \underline { { 1 5 2 . 8 } } } \pm 1 . 7$ </td><td> $1 5 8 . 5 \pm { 1 . 8 }$ </td><td> $1 6 3 . 7 \pm { 1 . 8 }$ </td></tr></table>

## upper-reference condition.

Figure 6 summarizes these configuration-level trends by kernel and acquisition setting. On Ckt-Bench-301, TTARO improves over GP in all but one encoder/kernel/acquisition setting, and every kernel/acquisition family improves on average across encoders. The gains are especially important for the linear-kernel settings, where improving the representation directly changes the inner-product geometry used by the surrogate. TTARO also improves the RBF settings, showing that the method is useful when the kernel uses a nonlinear distance-based form. The detailed results also show different trends across kernels. On Ckt-Bench-101, the largest TTARO gains occur in the linear-kernel settings, with average regret AUC reductions of 33.8% for Linear/EI, 16.0% for Linear/TS, and 22.9% for Linear/UCB. The RBF settings are more mixed on Ckt-Bench-101, although TTARO still improves most of them. On Ckt-Bench-301, the average gains are 10.9% for Linear/EI, 9.5% for Linear/TS, 14.7% for Linear/UCB, 12.7% for RBF/EI, and 17.7% for RBF/UCB.

The main limitation of TTARO is computational overhead. Retraining the feature map and GP after each evaluation is more expensive than updating a fixed-representation GP posterior, and this cost may matter for very large candidate banks or very short simulation times. Existing scalable GP and deep-kernel methods provide several paths for reducing this overhead. Structured kernel interpolation reduces the cost of kernel-matrix operations and enables scalable kernel learning [30]; stochastic variational deep-kernel learning supports scalable joint training of neural feature maps and GP kernels [31]; Lanczos-based variance estimation and GPU-accelerated GP inference reduce the cost of posterior uncertainty computation and sampling [32], [33]. Online GP methods further address the sequential-update setting by reusing computations after new observations arrive [34]. These methods suggest that future TTARO implementations can reduce surrogate-update cost through structured kernels, sparse variational approximations, and GPU-accelerated linear algebra. In the analog design regimes targeted here, circuit evaluation is typically the dominant cost, making additional surrogate training acceptable when it reduces the number of poor evaluations.

Overall, the results support the central premise of this work: for representation-based circuit BO, the quality of the surrogate depends strongly on whether the kernel geometry is aligned with the objective being optimized. Fixed GP relies entirely on the initial circuit embedding, and DKL learns a transformation only once from the initial data. TTARO keeps the representation coupled to the online search process. This lets the surrogate revise its notion of circuit similarity as new FoM evidence becomes available, which leads to better acquisition decisions and improved sample efficiency.

## ACKNOWLEDGMENTS

The authors used ChatGPT-5.5 for grammatical refinements, boilerplate code generation and to improve the aesthetic aspects of the figures. We assume full responsibility for all the content in this manuscript. We would like to thank the authors of OCB for making their code and datasets available.

## REFERENCES

[1] L. McInnes, J. Healy, and J. Melville, “Umap: Uniform manifold approximation and projection for dimension reduction,” arXiv preprint arXiv:1802.03426, 2018.

[2] J. Shen, F. Yang, L. Shang, C. Yan, Z. Bi, D. Zhou, and X. Zeng, “ATOM: An automatic topology synthesis framework for operational amplifiers,” IEEE Transactions on Computer-Aided Design ofIntegrated Circuits and Systems, vol. 44, no. 3, pp. 1193–1198, 2025.

[3] Z. Dong, W. Cao, M. Zhang, D. Tao, Y. Chen, and X. Zhang, “CktGNN: Circuit graph neural network for electronic design automation,” in The Eleventh International Conference on Learning Representations (ICLR), 2023. [Online]. Available: https://openreview.net/forum?id= NE2911Kq1sp

[4] J. Lu, L. Lei, F. Yang, L. Shang, and X. Zeng, “Topology optimization of operational amplifier in continuous space via graph embedding,” in 2022 Design, Automation & Test in Europe Conference & Exhibition (DATE), 2022, pp. 142–147.

[5] J. Gao, W. Cao, J. Yang, and X. Zhang, “AnalogGenie: A generative engine for automatic discovery of analog circuit topologies,” in The Thirteenth International Conference on Learning Representations (ICLR), 2025. [Online]. Available: https://openreview.net/forum?id= jCPak79Kev

[6] P. Xu, Y. Li, T. Chen, T.-Y. Ho, and B. Yu, “Ckt2Vec: Efficient electrical encoding for analog circuit representations in vector space,” IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2025.

[7] A. G. Wilson, Z. Hu, R. Salakhutdinov, and E. P. Xing, “Deep kernel learning,” in Proceedings of the 19th International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, A. Gretton and C. C. Robert, Eds., vol. 51. PMLR, 2016, pp. 370–378. [Online]. Available: https://proceedings.mlr.press/ v51/wilson16.html

[8] S. Zhang, W. Lyu, F. Yang, C. Yan, D. Zhou, and X. Zeng, “Bayesian optimization approach for analog circuit synthesis using neural network,” in 2019 Design, Automation & Test in Europe Conference & Exhibition (DATE), 2019, pp. 1463–1468.

[9] W. W. Xing, W. Fan, Z. Liu, Y. Yao, and Y. Hu, “KATO: Knowledge alignment and transfer for transistor sizing of different design and technology,” in Proceedings of the 61st ACM/IEEE Design Automation Conference, ser. DAC ’24. ACM, 2024, pp. 1–6. [Online]. Available: https://doi.org/10.1145/3649329.3657380

[10] T. Guo, D. R. Herber, and J. T. Allison, “Reducing evaluation cost for circuit synthesis using active learning,” in ASME 2018 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference. Quebec City, Canada: American Society of Mechanical Engineers, 2018, paper No. DETC2018-85654, V02AT03A011. [Online]. Available: https://doi.org/ 10.1115/DETC2018-85654

[11] S. Dutta, F. Amin, S. Panda, J. Rabe, Y. Wen, and P. Franzon, “Can an actor-critic optimization framework improve analog design?” 2026. [Online]. Available: https://arxiv.org/abs/2603.24714

[12] B. Weisfeiler and A. Leman, “The reduction of a graph to canonical form and the algebra which appears therein,” nti, Series, vol. 2, no. 9, pp. 12–16, 1968.

[13] J. Shen, F. Yang, L. Shang, Z. Bi, C. Yan, D. Zhou, and X. Zeng, “INTO-OA: Interpretable topology optimization for operational amplifiers,” in 2025 Design, Automation & Test in Europe Conference & Exhibition (DATE), 2025, pp. 1–7.

[14] F. Amin, S. Chatterjee, and P. D. Franzon, “Depthgraphnet: Circuit graph isomorphism detection via siamese-graph neural networks,” in 2023 ACM/IEEE 5th Workshop on Machine Learning for CAD (MLCAD), 2023, pp. 1–6.

[15] D. R. Jones, M. Schonlau, and W. J. Welch, “Efficient global optimization of expensive black-box functions,” Journal of Global optimization, vol. 13, no. 4, pp. 455–492, 1998.

[16] N. Srinivas, A. Krause, S. M. Kakade, and M. W. Seeger, “Information-theoretic regret bounds for gaussian process optimization in the bandit setting,” IEEE Transactions on Information Theory, vol. 58, no. 5, pp. 3250–3265, may 2012. [Online]. Available: https://doi.org/10.1109/TIT.2011.2182033

[17] W. R. Thompson, “On the likelihood that one unknown probability exceeds another in view of the evidence of two samples,” Biometrika, vol. 25, no. 3/4, pp. 285–294, 1933.

[18] J. Snoek, K. Swersky, R. Zemel, and R. Adams, “Input warping for bayesian optimization of non-stationary functions,” in Proceedings of the 31st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, E. P. Xing and T. Jebara,

Eds., vol. 32. PMLR, 2014, pp. 1674–1682. [Online]. Available: https://proceedings.mlr.press/v32/snoek14.html

[19] J. Bowden, J. Song, Y. Chen, Y. Yue, and T. A. Desautels, “Deep kernel bayesian optimization,” Lawrence Livermore National Laboratory, Tech. Rep. LLNL-CONF-819001, Feb. 2021, oSTI ID: 1811769. [Online]. Available: https://www.osti.gov/biblio/1811769

[20] J. Chen, G. Zhu, C. Yuan, and Y. Huang, “Semi-supervised embedding learning for high-dimensional bayesian optimization,” arXiv preprint arXiv:2005.14601, 2020. [Online]. Available: https: //arxiv.org/abs/2005.14601

[21] J. Tingey, C. M. Gilligan-Lee, and Z. Dai, “Contrastive embedding of structured space for bayesian optimization,” in NeurIPS 2021 Workshop on Meta-Learning, 2021. [Online]. Available: https://openreview.net/ forum?id=xFpkJUMS9te

[22] F. Zhang, B. Nord, and Y. Chen, “Learning representation for bayesian optimization with collision-free regularization,” arXiv preprint arXiv:2203.08656, 2022. [Online]. Available: https://arxiv.org/abs/2203. 08656

[23] S. Lee, J. Chu, S. Kim, J. Ko, and H. J. Kim, “Advancing bayesian optimization via learning correlated latent space,” Advances in Neural Information Processing Systems, vol. 36, pp. 48 906–48 917, 2023.

[24] S. Stanton, W. Maddox, N. Gruver, P. Maffettone, E. Delaney, P. Greenside, and A. G. Wilson, “Accelerating bayesian optimization for biological sequence design with denoising autoencoders,” in International conference on machine learning. PMLR, 2022, pp. 20 459–20 478.

[25] S. Singh and J. M. Hernandez-Lobato, “Deep kernel learning for reac-´ tion outcome prediction and optimization,” Communications Chemistry, vol. 7, no. 1, p. 136, 2024.

[26] V. Thost and J. Chen, “Directed acyclic graph neural networks,” arXiv preprint arXiv:2101.07965, 2021.

[27] M. Zhang, S. Jiang, Z. Cui, R. Garnett, and Y. Chen, “D-vae: A variational autoencoder for directed acyclic graphs,” Advances in neural information processing systems, vol. 32, 2019.

[28] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” arXiv preprint arXiv:1609.02907, 2016.

[29] B. Shahriari, K. Swersky, Z. Wang, R. P. Adams, and N. De Freitas, “Taking the human out of the loop: A review of bayesian optimization,” Proceedings of the IEEE, vol. 104, no. 1, pp. 148–175, 2015.

[30] A. Wilson and H. Nickisch, “Kernel interpolation for scalable structured gaussian processes (kiss-gp),” in International conference on machine learning. PMLR, 2015, pp. 1775–1784.

[31] A. G. Wilson, Z. Hu, R. R. Salakhutdinov, and E. P. Xing, “Stochastic variational deep kernel learning,” Advances in neural information processing systems, vol. 29, 2016.

[32] G. Pleiss, J. Gardner, K. Weinberger, and A. G. Wilson, “Constanttime predictive distributions for gaussian processes,” in International Conference on Machine Learning. PMLR, 2018, pp. 4114–4123.

[33] J. Gardner, G. Pleiss, K. Q. Weinberger, D. Bindel, and A. G. Wilson, “Gpytorch: Blackbox matrix-matrix gaussian process inference with gpu acceleration,” Advances in neural information processing systems, vol. 31, 2018.

[34] S. Stanton, W. Maddox, I. Delbridge, and A. G. Wilson, “Kernel interpolation for scalable online gaussian processes,” in International conference on artificial intelligence and statistics. PMLR, 2021, pp. 3133–3141.