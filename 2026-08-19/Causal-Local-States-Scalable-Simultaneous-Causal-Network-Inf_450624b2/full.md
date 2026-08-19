# Causal Local States: Scalable Simultaneous Causal Network Inference and Forecasting for Dynamical Systems

Jonas Braun<sup>1,2</sup>, Fabian Fischbach<sup>3</sup>, Daniel Koglmayr¨ <sup>3,4</sup>, Sebastian Baur<sup>3,4</sup>, and Christoph Rath¨ <sup>1,4,5\*</sup>

<sup>1</sup>Ludwig-Maximilians-Universitat M¨ unchen, Faculty of Physics, Geschwister-Scholl-Platz 1, 80539 Munich, Germany¨ <sup>2</sup>Institut fur Theoretische Physik, Universit ¨ at zu K ¨ oln, Z ¨ ulpicher Strasse 77, 50937 Cologne, Germany¨

<sup>3</sup>Institut fur KI-Sicherheit, Deutsches Zentrum f ¨ ur Luft- und Raumfahrt (DLR), St. Augustin & Ulm, Germany¨ <sup>4</sup>Entrox Systems, Munich, Germany

<sup>5</sup>Institut fur Frontier Materials auf der Erde und im Weltraum, Deutsches Zentrum f¨ ur Luft- und Raumfahrt (DLR),¨ 51170 Cologne, Germany

\* christoph.raeth@dlr.de

## ABSTRACT

Machine learning methods predict many real-world systems with remarkable accuracy, but they are typically treated as black boxes that offer no insight into which interactions drive the dynamics. Causal discovery methods reconstruct the interaction network from observational data, but without regard to whether the inferred structure supports prediction. Existing approaches combining both tasks rely on a single global hyperparameter, such as a causal threshold or a fixed neighborhood size, which cannot recover the structure of heterogeneous systems. Here we introduce causal local states (CLS), a framework that simultaneously infers an approximate Granger-causal interaction network and forecasts the system dynamics. For each node independently, we select the smallest set of neighbors that allows a predictive model to forecast the node near-optimally, and the resulting neighborhoods are then combined for a forecast of the full system. On three benchmarks of increasing difficulty, we achieve reconstruction of the underlying networks with high fidelity and forecasts on par with a model that is supplied with the true network, providing a step toward explainable and scalable forecasting of complex systems.

## Introduction

Many of the systems that shape our lives, such as power grids<sup>1</sup>, the atmosphere<sup>2</sup> or ecosystems<sup>3</sup>, are high-dimensional, nonlinear dynamical systems. Anticipating how these systems evolve is essential for countless decisions, and machine learning methods can forecast complex dynamics with considerable success<sup>4,</sup> <sup>5</sup>. However, these models are typically treated as black boxes: they turn raw, unstructured measurements into forecasts without revealing which of the many variables actually interact. Structuring the raw data into an interaction network during a prediction task helps in multiple ways. It allows treatment of high-dimensional systems where a lot of variables have little causal influence on the variable we want to forecast by selecting only a relevant subset of variables instead of considering the full system. Furthermore, an interaction network provides insights into the underlying system, supporting the decisions made from the data instead of blindly trusting a black-box approach. Causal local states (CLS) achieves exactly that: recovering the interaction network from unstructured raw time series data and using that network to improve the system forecast.

In the following, we approach this problem by using ideas from two different fields. Causal discovery aims to reconstruct the interaction network of a nonlinear dynamical system from observational time series data, either by thresholding directed bivariate causal measures such as Granger causality<sup>6</sup>, transfer entropy (TE)<sup>7</sup>, and convergent cross mapping (CCM)<sup>3</sup>, or by multivariate algorithms that additionally address indirect links and confounding<sup>8,</sup> <sup>9</sup>. By applying these methods, causal graphs can be reconstructed that are interpretable yet are inferred without taking into account their utility for forecasting the system’s evolution. Data-driven forecasting methods, in particular reservoir computing (RC)<sup>10</sup> and its computationally cheaper successor, next-generation reservoir computing (NGRC)<sup>11,</sup> <sup>12</sup>, predict dynamical systems accurately at low training cost. Reservoir computers, however, process all variables of the time series simultaneously. While some structural insight may be recoverable in low-dimensional settings, for high-dimensional and unstructured data the trained model provides no reliable insight into the underlying interaction network or even becomes computationally infeasible to train and use.

In this work, we develop a hybrid method that combines causal discovery and a prediction approach into one unified framework that (i) infers the interpretable interaction network of high-dimensional dynamical systems, in the form of one local neighborhood per variable, and (ii) uses these neighborhoods to produce accurate predictions of the whole system’s temporal evolution. Because the interaction network is inferred explicitly rather than left implicit inside a monolithic model (a single model for the full system), it becomes a direct, inspectable output of the method alongside the forecast: for each variable, the framework reports the small set of other variables its prediction relies on. While we use NGRC as the main predictive model, it is important to state that the causal measure and the model used are exchangeable components. This allows CLS to be adaptable to a wide range of systems.

The conceptual bridge between the two approaches is Granger’s predictive definition of causality<sup>6</sup>. A variable is said to Granger-cause another if its past improves the prediction of that other variable, provided it is conditioned on all other relevant data. Granger causality is thus predictive by construction. While this criterion is usually read as a causality test, it can equall be used to select between different variables. To facilitate prediction, a variable should only be passed to the model if and only if it improves the forecast. We achieve this by a wrapper-based approach. A black-box predictive model is trained and evaluated on candidate subsets of variables. The predictive performance across these subsets is then used to score the variables<sup>13,</sup> <sup>14</sup>. Reconstructing an approximate Granger-causal interaction network therefore reduces to a variable-selection problem solved separately for each node, where variables are selected according to how much they improve the node’s prediction accuracy. This identification of a wrapper-based selection as an operational form of Granger causality is one of the central ideas behind this work. Furthermore, the selection of the right neighbors is not only relevant for the inferred structure but also for the forecast itself, because with finite data, processing irrelevant variables adds computational cost and degrades the model quality<sup>15</sup>.

For high-dimensional systems, knowing the interaction structure is not only desirable for interpretability but can also aid in making a prediction possible in the first place. To avoid the curse of dimensionality, scaling to higher-dimensional systems Pathak et al.<sup>16</sup> introduced a local-states approach, which decomposes a spatiotemporal system into fragments. Each fragment contains a core node that is predicted from a small local neighborhood by its own model. Baur and Räth<sup>17</sup> generalized this decomposition to systems with non-local interactions by constructing the neighborhoods from similarity measures such as cross-correlation and mutual information. Such decompositions make data-driven prediction of large systems feasible in the first place. However, a local state approach requires the data to be structured beforehand such that the interaction structure is known. Combining the previously described wrapper approach to structure the data and then subsequently using a local states approach for prediction allows us to predict a high-dimensional system while structuring the underlying data.

Only a small number of studies in the RC literature connect network inference directly to predictive performance. Srinivasan et al.<sup>18</sup> combine TE with a parallel RC scheme, where links above a causal threshold are retained and the threshold is optimized as a global hyperparameter. Similarly, Chu et al. construct each neighborhood from the top-q entries of a TE matrix and again optimize the neighborhood size q as a single global hyperparameter<sup>19</sup>. These approaches assume that one global parameter, either a threshold or a neighborhood size, is appropriate for every node of the network. For heterogeneous systems this assumption fails, because scores are generally not comparable across subsystems with different dynamics and scales. Consequently, no global threshold or neighborhood size can recover all neighborhoods correctly, which we demonstrate explicitly in Supplementary Note 1. A method in which every neighborhood is inferred independently, by a criterion local to its core node, is therefore still missing. Further, Li et al.<sup>20</sup> propose higher-order Granger reservoir computing, which evaluates candidate coupling configurations by their one-step prediction error and encodes the inferred structure directly into the reservoir architecture. While this approach seems very promising, it is unclear how the initial candidate set to start their algorithm with can be chosen efficiently for high-dimensional systems. With no prior system knowledge, the candidate set must include all candidate neighbors initially, which would explode the search space. CLS avoids this pitfall and allow a simultaneous network inference and prediction without any prior knowledge about the underlying network. The CLS framework also does not require a global threshold, and since the per-node inference is fully parallel across nodes, it scales to high-dimensional systems naturally.

## Results

To establish our framework, we have to review some basic concepts central to CLS. CLS aims to structure completely unstructured data prior to prediction. As the prediction then relies on a local-states approach<sup>16</sup>, the structure CLS aims to infer is the neighborhood matrix. The neighborhood matrix holds the neighborhood information for exactly one core node per row, whereas the core node just defines which specific node is in focus. Then the neighborhood is defined around a core node to contain all nodes that share a directed edge pointing towards this core node. Since each neighborhood is centered on a single core, d neighborhoods cover the full system, and the collection can be stored compactly as a neighborhood matrix (a more strict definition can be found in Methods Eq. 1).

The first step of CLS is to structure the unstructured high-dimensional input into neighborhoods for each core node. The algorithm infers the neighborhood for each core node separately, combines all the individual neighborhoods, and then uses the inferred network to forecast the whole system. The input is a collection of time series with unknown couplings (Fig. 1a) For a given core node, a directed causal measure ranks all other variables as candidate neighbors and produces a small nested set of neighborhood candidates (Fig. 1b). A wrapper model then selects the smallest candidate that predicts the core node near-optimally and prunes redundant members by backward elimination (Fig. 1c), yielding an approximate Granger-causal neighborhood for that node (Fig. 1d). Repeating this independently for every node assembles a global neighborhood matrix (Fig. 1e). This matrix is then used to forecast the full system in parallel: each node is predicted from its own neighborhood alone (Fig. 1f), and the one-step predictions are fed back recursively to generate the multi-step trajectory of the entire system (Fig. 1g).

![](images/5bb43f5ed8931590d6aef4c1cd16f4bec8e61a060a0034b7af21cc4b38bcbf91.jpg)  
Figure 1. Schematic illustration of the causal local states (CLS) framework. a, The input consists of high-dimensional time series data generated by a system with an unknown interaction network. b, A causal measure is used to rank candidate neighbors for a core node and generate neighborhood candidates. c, A wrapper-based optimization procedure selects the optimal neighborhood candidate and subsequently removes redundant features through backward elimination. d, The inference process yields an approximation of the causal neighborhood of the core node. e, After repeating a-d for all core nodes, the inferred local neighborhoods are combined into a global neighborhood matrix. f, The CLS framework performs one-step prediction using the inferred neighborhoods by training a model $g _ { i }$ on only the data present in the neighborhood ${ \mathcal { N } } _ { i } ^ { * } . \ \mathbf { g } ,$ One-step predictions are recursively fed back and concatenated to generate multi-step predictions of the full system dynamics. Due to the independent neighborhood inference and prediction procedure for each node, the CLS framework is naturally parallelizable and scalable to high-dimensional systems.

The justification for predicting a system through such local neighborhoods is the causal Markov condition. It guarantees that the parents of a variable are a sufficient input for predicting it: given its parents, a variable is conditionally independent of all its other non-descendants, so no further variable carries additional predictive information. This is what allows a single d-dimensional prediction problem to be replaced by d small, local ones. Importantly, the condition is not violated by additional, spurious neighbors, because adding non-parents leaves the conditional independence intact. Spurious neighbors thus cost computation and obscure interpretation but do not invalidate the decomposition; CLS nonetheless includes an elimination step to remove them, since smaller neighborhoods improve interpretability and simplify model fitting. One might assume that since a graph that fulfills the causal Markov condition is optimal for prediction, causal discovery algorithms would be preferable in network reconstruction over CLS. We refer the reader to Supplementary Note 2 for a more thorough discussion on this matter.

We validate CLS on three benchmarks of increasing difficulty. On composite Lorenz63–Rössler systems of up to 90 dimensions, an exhaustive evaluation of all candidate neighborhoods confirms that the forecast error is a reliable selection criterion, and CLS cleanly separates the constituent attractors in a regime where a single NGRC, trained on the full system without variable selection, fails. On the 40-dimensional Lorenz96 system, CLS recovers the banded coupling structure and produces forecasts comparable to those of a model supplied with the governing-equation adjacency matrix. Finally, on a higher-order Kuramoto model of the UK power grid, CLS paired with a domain-informed wrapper reconstructs the real network topology almost perfectly and forecasts on par with a model given the true network. In all three cases, the inferred neighborhoods additionally provide insights into the system dynamics. CLS predicts the system and at the same time reports which interactions the prediction is based on.

## Composite Lorenz63–Rössler attractor pairs

To test the CLS framework on a well-tested system with a simple interaction structure, we consider a dynamical system composed of N non-interacting Lorenz63–Rössler attractor pairs (see Methods, Benchmark systems). The two subsystems within each pair and distinct pairs share no coupling. Therefore, the interaction structure of a full 6N-dimensional system decomposes cleanly into 2N independent three-dimensional blocks. A schematic of a single attractor pair is shown in Fig. 2a. We emphasize that, despite this simple block structure, the system poses a non-trivial inference challenge, since a pairwise causal measure applied to time series data will in general return spurious causal relations between variables. In CLS we therefore introduce an additional wrapper-elimination step designed to disentangle true from spurious neighbors (see Methods Causal Local States).

Validation of the wrapper model. We first verify that a wrapper model can in principle identify the correct interaction structure when allowed to evaluate every possible interaction exhaustively. This is done for a system with one attractor pair N = 1, as the space of possible interactions would be combinatorially huge otherwise. For each core node $X _ { i }$ we train a separate NGRC model on each possible neighborhood candidate. Concretely, this means that an NGRC is trained to forecast the core node trajectory exclusively from the data present in the neighborhood (Supplementary Note 3). Subsequently, we generate multiple one-step predictions and calculate the resulting mean squared error (MSE) from the true core node trajectory.

As shown in Fig. 2b, the candidate neighborhoods that contain all true neighbors of the same attractor of the core node (green bars) yield one-step MSEs several orders of magnitude lower than those that contain neither (red bars). Neighborhoods that contain one true neighbor but not the other tend to lie between these extremes (blue and orange bars, one color per neighbor). This monotone separation establishes the predictive loss as a reliable metric for neighborhood quality and motivates its use as the wrapper criterion in CLS. Exhaustive evaluation is, however, combinatorially infeasible beyond very small systems, as there are $2 ^ { 6 N - 1 } - 1$ non-empty neighborhood candidates per core node (31 for N = 1). To restrict the search space before applying the wrapper evaluation, a filter step is introduced in the CLS framework, generating a set of potential neighborhood candidates (described in more detail in Methods, Causal Local States).

Inference and prediction for a single attractor pair. We next apply the full CLS framework to a single Lorenz63-Rössler pair (N = 1) (see Methods, Causal Local States, for a thorough description of CLS). Figure 2c shows the TE matrix computed during the filter step. Even though the block-diagonal structure is clearly visible, the cross entries are non-zero and, in places, comparable in magnitude to the weaker intra-attractor entries. For example, when inferring the neighborhood of $R _ { x }$ , the

a  
![](images/6cfbfffd117b434a1365dc1eb87ecab8c5cccfba1203c3ed3a933ce7be9596d7.jpg)

![](images/6e5498e8a5f148d35d569dce81cd7e898337286b18f2bdc9bba5e5fd08aa023f.jpg)

![](images/09b9d43976a3a947a597413245b10cd849f5028f09391bd78a34a13ad04011da.jpg)

![](images/489f1c438de396354ce7f7a056ea9243d12722ac054a41f8fb80f43623c3b49d.jpg)

![](images/38c618047f7ff9eb4da49dc2ba10cc0d7f901bdea640b3052304427dd6937e9e.jpg)

![](images/f087218c957d98bc0bab09abe273eea292da76c9621412b6bbd364e5b89578cd.jpg)

![](images/d9b5cde04a04174b402fa51a29a277cb92738cc44d2670d77d5cd5c2402e00f8.jpg)

c  
![](images/335fd574b3c93f160d480d4e6f7d6941be562b2ea2bd119a24f59133efdb2aa7.jpg)

d  
![](images/2ec79efceef08af572f03ed744e811028b42fcca1050c42c70fcfeff831de181.jpg)

![](images/a6e9af9cb9a36a50b08f4b2aff37e036cc853db6d74770e7d4a6658c189ad147.jpg)

![](images/f402660074cdf843e5e46048e9d11dd50aeb887a54576e6ec05b19465fd228a4.jpg)

![](images/eebe35543e3f128735c2dbf70b1ef4ffe34f67605f424180428ec1558c0583ef.jpg)

![](images/3fb20ab96cadc57c47475f831d83c90c6369191736b8abad37d7cbdb71c8e6bc.jpg)

![](images/8ca21de67dea172b95ee788a70db9e27180c8aa8ba30fc8e38382d5b9217d394.jpg)

![](images/dfe856c22fcb5cab1532206798175ba0e0209b3c5b4b4f15d85da1c1ed57e595.jpg)

![](images/b318abf9b9eef7182e345b290ea8a144ed58c56488db09e7a1c4c40ce0695a19.jpg)

f  
![](images/e25be087266d907ea28ee634886d957ffa10a67f1538413612012f501bb90184.jpg)

![](images/b3525b3988c38b5e3bf57ea3ec180e754be8a0180da5b7023e38ec3f129cbfb0.jpg)

![](images/1b15f0e262d93cb4cd9c1669e8d8075d72e38df3652f76de84154a7f5a6e4dce.jpg)  
Figure 2. Analysis of composite (non-interacting) Lorenz63–Rössler systems using causal local states (CLS). a, Summary graph of a single Lorenz63–Rössler attractor pair. b, Exhaustive wrapper validation for a single attractor pair. All non-empty neighborhood candidates of a core node are evaluated, and their wrapper mean squared error (MSE) is shown on a logarithmic axis, ordered by decreasing error. Bars are colored by how many of the core node’s two same attractor neighbors the candidate contains: red, neither; blue and orange, exactly one (one color per neighbor, identified in the sub-panel legends); green, both. c-e, CLS analysis for a single attractor pair system. c, Transfer entropy (TE) matrix computed during the filtering step. d, Inferred neighborhood matrices before and after feature elimination. e, Short-term prediction performance of CLS compared with standard next-generation reservoir computing (NGRC). f, Inferred neighborhood matrix after feature elimination for a system of 15 coupled attractor pairs (90 variables), shown for seed 0. g,h, CLS performance across different system sizes $( N \in [ 1 , 1 5 ] )$ , using TE as the filter and NGRC as the wrapper model. Every system size was run for n = 10 seeds, and every seed forecast from 10 disjoint starting points; markers give the mean over seeds and error bars ±1 standard deviation across them. g, Neighborhood inference accuracy: true positive rate (TPR) on the left axis (circles; grey dashed, before elimination; green solid, after) and false positive rate (FPR) on the right axis (squares; grey dash-dotted, before elimination; red solid, after) h, Prediction quality as the mean valid prediction steps (VPS) at ε = 0.3, for CLS (blue circles, solid line) and for a monolithic NGRC trained on the full state vector with linear features only (red squares, dashed line). All experiment parameters can be extracted from Table 1.

TE assigned to its true neighbor $R _ { z }$ is lower than the values assigned to the entirely independent Lorenz features $L _ { x } , L _ { y } , L _ { z } .$ A naive thresholding of the TE matrix would, therefore, either miss true neighbors or admit false ones, depending on the threshold (Supplementary Note 1). The filter step nevertheless performs its intended function of ranking the causal neighbors above the spurious ones for every core node, producing a small set of nested neighborhood candidates that the wrapper can evaluate efficiently. The feature elimination step then refines these candidates and removes the spurious and redundant features. Figure 2d compares the neighborhood matrix obtained directly after the wrapper selects an initial candidate (left) with the matrix obtained after backward feature elimination (right). The initial candidate nearly captures the block-diagonal structure but retains several cross-attractor false positives. Backward elimination subsequently removes exactly these entries. The resulting reduced neighborhood matches the block-diagonal structure exactly. One needs to note that the block-diagonal structure just shows that CLS successfully separates the attractors. According to Takens’ embedding theorem<sup>21</sup>, even dimensions that are not directly connected $( \mathrm { e } . \mathrm { g } . , R _ { y }$ and $R _ { z } )$ share information, so they can contribute to an increase in prediction performance.

Scaling to many attractor pairs. To demonstrate that the per-node, parallel structure of CLS scales to larger systems, we extend the experiment to $N = 1 5$ attractor pairs, yielding a 90-dimensional system. Figure 2f shows a sample inferred neighborhood matrix after wrapper elimination. The block-diagonal ground truth is recovered with high fidelity across all 15 blocks, with only a few false off-block entries. Importantly, after computing the initial ranking using a bivariate causal measure, the inference for each core node is performed independently, so the computational complexity of CLS scales linearly in the number of core nodes when parallelized. We systematically quantify inference quality and prediction performance as a function of system size by sweeping N from 1 to 15. Figure 2g reports the true positive rate (TPR) and false positive rate (FPR) of the inferred neighborhood matrices, both before and after the elimination step. After elimination, the TPR remains above 0.8 across all system sizes tested, while the FPR is suppressed to near zero. Furthermore, Figure 2h shows how these inference results translate into prediction performance, measured by the valid prediction steps (VPS) at error threshold $\varepsilon = 0 . 3 ( \mathrm { E q }$ . 14). Standard NGRC trained on the full state vector is not able to predict the system at all. Moreover, it was not possible to run NGRC with higher-order nonlinearities (as was used to achieve the predictions in Figure 2e), as for a high-dimensional feature space, this requires too much RAM (see Methods, Next-Generation Reservoir Computing). This limitation is lighter in CLS, as CLS uses a local-states approach following Pathak et al.<sup>16</sup>, which keeps the feature space for each individual NGRC instance small and manageable. The prediction performance of CLS saturates at roughly 1000 VPS for $N \gtrsim 7$

These experiments establish two main points. First, the filter and wrapper steps of CLS are complementary. The filter alone leaves cross-attractor false positives, while the wrapper elimination alone would be combinatorially infeasible and computationally expensive. Second, the per-node parallel architecture of CLS allows for treatment of high-dimensional systems.

## Lorenz96 system

The composite attractor pairs in the previous section validate the wrapper criterion against an exhaustive ground truth. However, the subsystems in that benchmark are dynamically independent. We therefore consider the Lorenz96 system<sup>22</sup>, a high-dimensional dynamic system that can exhibit chaotic behavior (see Methods, Benchmark systems).

Structure inference. During the filter step, we compute the TE matrix from the simulated trajectories (Fig. 3a). The resulting estimate reproduces the periodic coupling band of the ground-truth network, favoring the true parents of each core node (Fig. 3a, e). Applying wrapper selection and backward elimination with NGRC as the wrapper model yields the neighborhood matrices shown in Fig. 3c,d. Both recover the underlying coupling pattern: across ten seeds every true parent is retained at both stages $( \mathrm { T P R } = 1 . 0 0 0 \pm 0 . 0 0 0 )$ , while backward elimination lowers the FPR from $0 . 1 8 7 \pm 0 . 0 2 4 \mathrm { ~ t o ~ } 0 . 1 4 6 \pm 0 . 0 0 7$ . The inferred neighborhoods nevertheless contain more nodes than the three parents specified by the governing equations, averaging $9 . 7 5 \pm 0 . 8 7$ members before elimination and $8 . 2 5 \pm 0 . 2 3$ after. These additional nodes are not distributed randomly; they remain close to the core node on the lattice, with 93.2% of the retained false positives lying within a lattice distance of four.

This outcome is consistent with a prediction-oriented selection criterion. Takens’ embedding theorem<sup>21</sup> implies that information relevant to a node’s future state is not restricted to its direct dynamical parents. Nearby variables that are onl indirectly connected can also encode predictive information. Including such nodes can therefore improve forecasts of the core variable, and a criterion based on predictive performance may retain them. Another possibility is that some of the extra connections reflect mild overfitting to the wrapper test set. As shown below, however, the retained nodes do not appear to degrade predictive performance.

Prediction quality. To assess the impact of the inferred structure on forecasting performance, we train three otherwise identical NGRC local-states models that differ only in their neighborhood matrix: the governing-equation adjacency, the inferred matrix before elimination, and the final matrix after elimination. Figure 3f–i compare the resulting short-term forecasts with the true trajectory, while Fig. 3j summarizes the VPT across all nodes. All three models reproduce the short-term dynamics accurately. Averaged over ten seeds, the governing-equation adjacency reaches $5 . 1 8 \pm 0 . 6 7$ Lyapunov times, the inferred matrix before elimination $5 . 4 0 \pm 0 . 6 2$ , and the reduced matrix after elimination $5 . 3 1 \pm 0 . 6 4$ . CLS therefore matches the model supplied with the true adjacency matrix, despite working from a structure inferred entirely from data.

![](images/39298e347817492ad9d9e65f93e157055b18ba5694b45b15ec275878002b06be.jpg)

![](images/73bea93bbf6979e567cc68ac0c7c70d6a8ba4f108978249522305e4a42997946.jpg)

![](images/ae1d878786c99f0dc920cd29eca207c46e2447df02165bccdaa06c087004bf90.jpg)

![](images/0209e99bc18591adfb398dbe5fcb3c09b280dda1e4a074bbaa3d45ab600e09df.jpg)

![](images/e5d5b43e15fdef25e0b6341da306e8498403b2d5bf59c342f50a8125e21711ee.jpg)

![](images/6b487dfcc95962d10d3cdfbda864d3b39cbd42b7bbe9fa157c130dab8456ec4f.jpg)

![](images/c2fac6f51e0f1186318f80ab88fde227002162a856c97a20e21ce2b39af662ba.jpg)  
Figure 3. Causal local states (CLS) analysis of a 40-dimensional Lorenz96 system. $\mathbf { a } ,$ Transfer entropy (TE) matrix computed during the filtering step. b, Ground-truth neighborhood matrix defined by the governing equations. c, Inferred neighborhood matrix prior to feature elimination. d, Final neighborhood matrix after feature elimination. e, Histogram showing the distribution of calculated TE values split between true parents of a node and non-parents. f, Ground-truth system trajectory. g–i, Short-term predictions using the true adjacency matrix (g), the inferred matrix before feature elimination (h), and the inferred matrix after feature elimination (i). j, valid prediction steps (VPS) at $\varepsilon = 0 . 3$ expressed as valid prediction time (VPT) in maximum Lyapunov times for CLS using the three neighborhood matrices. Each of the 40 grid nodes contributes one point, its VPS averaged over 100 forecasts (10 seeds × 10 disjoint launch points). All experiment parameters can be extracted from Table 2.

The interaction structure that yields the best predictions does not have to coincide with the governing-equation graph. The additional nearby connections retained by CLS are absent from the nominal ground-truth network, yet they provide useful predictive information. Methods that treat the governing adjacency as the sole target of inference cannot capture this effect (Supplementary Note 2). By selecting the neighborhoods according to forecast skill rather than causal graph recovery, CLS directly optimizes the structure for predictions.

## UK power grid system

As a final and most challenging benchmark, we apply the CLS framework to a real-world network, a dynamical model for the UK power grid in the form discussed by $\mathrm { L i e t a l . } ^ { 2 0 , 2 3 }$ (see Methods, Benchmark systems).

This benchmark exposes a failure point of polynomial NGRC on oscillator networks. As also reported $\mathrm { i n } ^ { 2 4 }$ and consistent with our own experiments, the standard polynomial NGRC formulation tends to produce divergent predictions on oscillator-type systems. As the CLS framework is model-agnostic, this allows us to sidestep this issue by selecting a wrapper whose feature dictionary is better matched to the system at hand (Supplementary Note 3). We consider two such variants:

• Sine-NGRC. The nonlinear feature vector is built from polynomials of $\mathrm { s i n } ( \theta _ { i } )$ and $\cos ( \theta _ { i } )$ , rather than from the raw phases. This variant ensures that its features are bounded in $( - 1 , 1 )$ , which naturally contains the divergence observed in polynomial NGRC.

• Kuramoto-NGRC. The nonlinear expansion is built around the coupling nonlinearity of the Kuramoto model itself, using interaction terms of the form sin $\left( \theta _ { j } - \theta _ { i } \right)$ for candidate node pairs $( i , j )$ . This variant is more domain-informed and, like Sine-NGRC, has bounded features.

Both variants can serve as wrapper models within the CLS framework. Before calculating $\mathrm { C C M } ^ { 3 }$ to generate the variable ranking, we transform the data and apply CCM on sin(θ) instead of the raw θ values (this is necessary as CCM relies on attractor reconstruction techniques that need a bounded trajectory). As the models transform the data with a sine-transformation during the nonlinear feature expansion, we pass the raw θ values. To improve stability, we do not choose $\theta _ { t + 1 }$ as the training target for timestep $t ,$ but rather the difference from the current state $\Delta \theta _ { t } = \theta _ { t + 1 } - \theta _ { t }$ . The predicted $\theta _ { t + 1 }$ value can then easily be calculated from $\Delta \theta _ { t + }$

We first illustrate the per-node inference procedure on a representative core node (node 51), whose true neighbors in the grid are nodes 34, 50 and 58 (Fig. 4f–i). The bar plot of pairwise causal influence calculated during the filter step (Fig. 4f) shows that several false positives have CCM scores comparable to or even exceeding those of true neighbors, which makes them indistinguishable on the basis of the filter alone (Fig. 4j; Supplementary Note 1). Thus, even though the candidate neighborhood returned by CLS using Kuramoto-NGRC contains all true neighbors, it also includes a substantial number of spurious connections. The wrapper-based backward elimination step $\left( \mathrm { F i g . 4 g } \right)$ then prunes these false positives: the remaining neighborhood almost perfectly matches the true local structure.

We next apply the inference process to every node in the grid and assemble the inferred local neighborhoods into a global neighborhood matrix. Figures 4b–e show the neighborhood matrices inferred with both wrappers before and after backward elimination. Two things stand out: First, for both wrapper variants, the elimination step substantially reduces the FPR while largely preserving the TPR. All values quoted here are means ± standard deviations over the ten seeds. For Sine-NGRC the total FPR drops from $2 . 5 7 \pm 0 . 2 5 \%$ to $1 . 4 7 \pm 0 . 1 6 \%$ with a TPR falling from $8 6 . 4 \pm 2 . 4 \%$ to $8 2 . 2 \pm 2 . 5 \%$ . Since the UK power grid is quite sparse, this is still favorable even though the percentage values might make it seem otherwise (Supplementary Note 2). For Kuramoto-NGRC, in comparison, the FPR drops from $1 0 . 6 2 \pm 0 . 4 9 \%$ to $1 . 8 7 \pm 0 . 1 5 \%$ with the TPR largely preserved $( 9 8 . 1 \pm 0 . 5 \%$ to $9 6 . 1 \pm 1 . 1 \%$ ). With the neighborhood size bound at $d _ { \operatorname* { m a x } } = 3 0$ for Kuramoto-NGRC, the CCM filter places 98.8% of all true neighbors among the $d _ { \mathrm { m a x } }$ highest-ranked candidates. The TPR of $9 8 . 1 \pm 0 . 5 \%$ measured before elimination therefore sits essentially at this filter ceiling, which shows that the wrapper selection discards almost no true neighbors.

For each wrapper variant we run closed-loop forecasts using three different neighborhood matrices: the ground-truth adjacency, the initial (post-filter) inferred matrix, and the final (post-elimination) reduced matrix. The Kuramoto-NGRC predictions remain accurate over a long prediction horizon for all three neighborhood matrices, with both inferred ones yielding errors comparable to those obtained with the true graph. The Sine-NGRC predictions reach a median 601 VPS against Kuramoto-NGRC’s 535. Sine only collapses on its own inferred adjacencies $( 6 0 1  1 4 0 )$ , which suggests that the failure is in the neighborhood selection.

Figure 4k summarizes the observations through the VPS across all nodes computed on the $\sin ( \theta )$ values. Most notably, however, the final prediction quality obtained by CLS using Kuramoto-NGRC is comparable with a local-states approach that used the true underlying network.

![](images/ef74c45b75d8c8cf0a9d3e1d90a9fa071f7867c3f7954f35949c9dd24497be7f.jpg)

b  
![](images/886d7ced1aeaa4f1fd92fd1832bb885f8cc06da7295466447fdebfbe138173f8.jpg)

![](images/c835cb3616651a3dac9e7c925bed1a3f163c62bba75002b65056bd2519f4eb25.jpg)

d  
![](images/0e4784dea15fae08cc40fa7a7b0fa1eed0569271fb3f39150d52c28691f82e6c.jpg)

e  
![](images/267b0ffc49b610b9825128f90379a77a68a7597bad179763143196805e4333e7.jpg)

![](images/679d12012f52bc52e027b32b16de06580048d5f238c6e4f3992b476e219d61f6.jpg)

![](images/4fd79f72462f2783afae3a46bb12c713a840a06fabf9e402a9571c2e5fde911d.jpg)

![](images/489c896557ddcadb0f705bfe4c59fc74f0bb886789dbe2f3bcb79515ab7c1dca.jpg)

![](images/b58ff9ec7c298ba9dd35e7e1b9260a64291488f11b3ba9cf133f41e8234f71b5.jpg)

![](images/8e32a3e5ce7e2716b44aa0baaf680600359f011527be14268c43d38d248cd503.jpg)

![](images/7e801de2e8dc10e14c4ccc582d50ca161be9b8554c52713d03e55ed0c1294db0.jpg)  
Figure 4. Causal local states (CLS) analysis of the Kuramoto model of the UK power grid. ${ \mathbf { a } } ,$ Ground-truth adjacency matrix. b,c, Neighborhood matrices inferred using Sine-NGRC before (b) and after (c) feature elimination. $\mathbf { d , e , }$ Neighborhood matrices inferred using Kuramoto-NGRC before (d) and after (e) feature elimination. f–i, Example neighborhood inference process for a single core node using Kuramoto-NGRC as the wrapper model. f, h, Initial neighborhood identified after the filtering step. f shows that multiple spurious neighbors are selected in the first step because their convergent cross mapping (CCM) score is higher than the true neighbors. $\mathbf { g } , \mathbf { i }$ , Final neighborhood after backward feature elimination. The spurious neighbors are largely removed from the neighborhood. j, Histogram showing the distribution of calculated CCM values split between true parents of a node and non-parents. k, valid prediction steps (VPS) for both wrapper models and all three neighborhood matrices, on sin θ at $\varepsilon = 0 . 3$ . Each of the 120 grid nodes contributes one point, its VPS averaged over 100 forecasts (10 seeds × 10 disjoint launch points). All experiment parameters can be extracted from Table 3.

## Discussion

We have introduced causal local states (CLS), a framework that simultaneously infers the interaction structure of a high dimensional dynamical system and forecasts its evolution. CLS extends the local-states<sup>16</sup> and generalized local-states<sup>17</sup> approaches by removing their reliance on a known network structure or simple correlation-based measures. Where generalized local states use similarity measures to group variables, CLS combines a filter- and wrapper-based approach to approximate each node’s causal neighborhood directly from data. By bringing ideas from causal discovery into the prediction task, CLS moves toward more explainable machine learning.

Across our benchmarks, CLS was able to reconstruct the neighborhoods accurately and provide predictions on par with a local states approach that was provided the true network. For the coupled Lorenz63–Rössler task, CLS was able to separate the different attractors and forecast all of them accurately. Applied to the Lorenz96 system, it recovered the banded interaction structure and matched the forecasting performance of a local-states model supplied with the true adjacency matrix. Paired with a domain-informed Kuramoto-NGRC model as a wrapper, CLS reconstructed an almost exact adjacency matrix for the UK power grid and achieved short-term accuracy comparable to a model given the true network. Nonetheless, this experiment showed that model selection matters for performance, as CLS with a Sine-NGRC wrapper did not generate adequate results on the UK power grid. The independence of causal measure and wrapper model in the CLS framework is also a lever here. By choosing a filter and wrapper that match the characteristics of the data, the performance of the algorithm can be greatly improved.

One further aspect of CLS is worth emphasizing. Each neighborhood is inferred separately from the full system, so both inference and prediction parallelize across nodes and scale to high-dimensional systems.

Despite these promising results, several open challenges remain. Most notably, CLS has not yet been applied to real-world datasets. Potential applications include financial time series, power grid dynamics, or weather forecasting, domains where both predictive performance and interpretability are of high importance.

In its current form, the inference procedure is restricted to identifying causal neighbors. Extending the framework to include the selection of relevant temporal features could provide deeper insights into system dynamics and further improve predictive performance. Incorporating tree-based NGRC models with built-in feature selection capabilities could be a promising direction in this context<sup>25</sup>.

Another challenge concerns hyperparameter optimization. The CLS framework involves multiple local models, each of which may require its own set of hyperparameters. The current approach is to use a shared hyperparameter configuration across all neighborhoods. However, this is unlikely to be optimal. Developing efficient strategies to optimize hyperparameters without exhaustive search remains an open challenge.

CLS predicts high-dimensional dynamical systems at scale while inferring an approximation of their underlying causal structure. The benchmark performance demonstrates the strong potential of CLS as a tool for combining prediction and interpretability in complex systems.

## Methods

One of the core concepts used extensively in causal local states are neighborhoods. Consider a multivariate time series $\{ X _ { t } \} _ { t \in \mathbb { Z } }$ with $X _ { t } = ( X _ { t } ^ { 1 } , \ldots , X _ { t } ^ { d } )$ , generated by a dynamical system whose variables interact. Under the assumption of stationarity, the interaction network can be represented compactly by a summary graph, a directed graph whose nodes correspond to the system variables and which contains an edge $X ^ { j }  X ^ { i }$ if and only if $X ^ { j }$ causally influences $X ^ { i }$ at some time $\log ^ { 2 6 }$ . The summary graph abstracts away the lag structure of causal influence, which matches the scope of CLS: it performs spatial feature selection (which variables matter for a node) rather than temporal feature selection (which time lags matter).

For a core node $X ^ { i }$ , we define its neighborhood as

$$
\mathcal { N } ( X ^ { i } ) = \left( \{ X ^ { i } \} \cup \mathbf { P a } ( X ^ { i } ) , \ : \{ ( X ^ { k } , X ^ { i } ) : X ^ { k } \in \mathbf { P a } ( X ^ { i } ) \} \right) ,\tag{1}
$$

where $\mathrm { P a } ( X ^ { i } )$ denotes the parent set of $X ^ { i }$ in the summary graph, i.e., all variables with a directed edge into $X ^ { i } .$ . The definition is close to Pearl’s notion of a family<sup>27</sup> but distinguishes explicitly between the core node and its parents, and is adapted from the generalized local-states approach<sup>17</sup>.

## Causal Local States

The CLS framework fragments a d-dimensional system into d core nodes, each with a local neighborhood that is inferred directly from the data. Unlike the local-states approach<sup>16</sup>, CLS treats the interaction structure as unknown and recovers it through a three-step procedure: causal filtering, wrapper selection, and backward elimination. The individually inferred neighborhoods are then combined into a global neighborhood matrix and used for prediction. Because the inference is run independently for each core node, the framework is trivially parallelizable and scales to high-dimensional systems. A full schematic is shown in Figure 1 and the pseudocode is provided in Algorithm 1.

Let $\{ X _ { t } \} _ { t \in \mathbb { Z } }$ with $X _ { t } = ( X _ { t } ^ { 1 } , \ldots , X _ { t } ^ { d } )$ be the observed time series, and let $X ^ { j } = \{ X _ { t } ^ { j } \} _ { t \in \mathbb { Z } }$ denote the trajectory of node $j .$ We describe the inference of the neighborhood $\mathcal { N } ^ { * } ( X ^ { i } )$ for a single, fixed core node $X ^ { i }$ , but the procedure is identical for all nodes.

Filter step. An exhaustive search over the $2 ^ { d - 1 }$ possible neighborhoods of a core node is infeasible in high dimensions, so we begin by ranking candidates using a bivariate causal measure. With this step, we restrict the search to a nested sequence of the highest-scoring candidates. For each $j \neq i$ we compute a causal score

$$
C _ { j \to i } : = C ( X ^ { j } \to X ^ { i } ) ,\tag{2}
$$

where $C ( \cdot  \cdot )$ can be any directed causal measure (we use TE and CCM throughout, but the choice is interchangeable). Unlike symmetric statistical measures, a causal score captures directed influence. Reindexing the potential neighboring nodes such that $C _ { ( 1 )  i } \geq C _ { ( 2 )  i } \geq . . .$ . yields a nested family of candidate neighborhoods of increasing size, defined as

$$
\mathcal { N } _ { k } ( X ^ { i } ) = \mathcal { N } _ { k } = \Big ( \{ X ^ { i } \} \cup \{ X ^ { ( 1 ) } , . . . , X ^ { ( k ) } \} , \ \{ ( X ^ { ( \ell ) } , X ^ { i } ) \} _ { \ell = 1 } ^ { k } \Big ) , \qquad 1 \leq k \leq d _ { \operatorname* { m a x } } ,\tag{3}
$$

where $X ^ { ( k ) }$ is the node with the k-th highest score and $d _ { \mathrm { m a x } }$ is a hyperparameter bounding the neighborhood size. The filter step thus reduces the search space from $\mathcal { O } ( 2 ^ { d } )$ to $d _ { \operatorname* { m a x } } + 1$ candidates $\bar { \mathcal { C } _ { i } } = \{ \mathcal { N } _ { k } ( X ^ { i } ) \} _ { k = 1 } ^ { d _ { \operatorname* { m a x } } }$ per core node. Calculating the causal matrix is in practice done prior to the separation between the core nodes and has itself a computational complexity of $\mathcal { O } ( d ^ { 2 } )$ Depending on the wrapper used, $k = 0$ is undefined or the neighborhood containing just the core node itself.

Wrapper selection. Each neighborhood candidate is evaluated by a wrapper model<sup>13</sup>. A wrapper model is a concept from feature selection, where a black-box model is trained and evaluated on a given feature set to assess the feature quality. In CLS, the wrapper is trained on the full data present in the neighborhood but scored only on the prediction of the core node. For the candidate, $\mathcal { N } _ { k }$ we form a feature vector $\mathbf { z } _ { k } = ( X ^ { j } ) _ { X ^ { j } \in V ( \mathcal { N } _ { k } ) }$ , where the core node data $X ^ { i }$ may be excluded depending on the wrapper. Using a wrapper that supports an NGRC inference-like mode is preferable. During inference mode the NGRC model learns the mapping from the neighbors to the core node, which means that the neighbor contributions are not masked by the self-influence of the core node (see supplementary information 3 for a closer look at the self-influence during neighborhood selection). Now we train a model $g _ { k } : \mathbb { R } ^ { d _ { k } }  \mathbb { R }$ to predict the next core state, $\tilde { X } _ { k , t + 1 } ^ { i } = g _ { k } ( \mathbf { z } _ { k , t } )$ . The wrapper test loss $\mathcal { L } _ { k }$ is evaluated on a wrapper test set ${ \mathcal { S } } ^ { \mathrm { w - t e s t } }$ disjoint from the wrapper training set. We control the size of the wrapper test set with the wrapper test fraction parameter, which just sets the fraction of the training data that is held out for wrapper evaluation. This is done so the final test set is completely unseen during the evaluation of the full system prediction. Finally, we select the smallest neighborhood that achieves near-optimal performance using a relative improvement criterion with threshold $\alpha _ { 1 } \in ( 0 , 1 )$ . The selection scans candidates in order of increasing size and a candidate $\mathcal { N } _ { k }$ becomes the new baseline $k ^ { * }$ whenever

$$
\Delta ( k , k ^ { * } ) = \frac { \mathcal { L } _ { k ^ { * } } - \mathcal { L } _ { k } } { \mathcal { L } _ { k ^ { * } } } \geq \alpha _ { 1 } .\tag{4}
$$

Backward elimination. The selected neighborhood $\mathcal { N } _ { k ^ { * } }$ may still contain redundant variables. Starting from $\mathcal { N } = \mathcal { N } _ { k ^ { \ast } }$ with a vertex set $V$ and edge set E and a baseline loss $\mathcal { L } _ { 0 } = \mathcal { L } _ { k ^ { * } }$ , we iteratively remove the least informative neighbor. For each neighbor node of $X ^ { i } , X ^ { j } \in \mathcal { N } , j \neq i ,$ , the reduced neighborhood $\mathcal { N } ^ { ( - j ) } = \stackrel { \cdot } { ( } V \setminus \{ X ^ { j } \} , E \setminus \{ ( X ^ { j } , X ^ { i } ) \} )$ ) is re-evaluated to give a loss $\mathcal { L } _ { - j }$ . The candidate $X ^ { m }$ with $\begin{array} { r } { m = \arg \operatorname* { m i n } _ { j } \mathcal { L } _ { - j } , } \end{array}$ i.e. the node whose removal least degrades prediction, is eliminated only if the relative deterioration stays within a threshold $\alpha _ { 2 } \in ( 0 , 1 )$ ,

$$
\frac { { \mathcal { L } } _ { - m } - { \mathcal { L } } _ { 0 } } { { \mathcal { L } } _ { 0 } } \leq \alpha _ { 2 } .\tag{5}
$$

The baseline is then updated and the step repeated until no further removal satisfies the criterion. The causal Markov condition motivates this procedure: non-descendants of $X ^ { i }$ are conditionally independent of $X ^ { i }$ given its parents, so removing a truly causal neighbor degrades prediction sharply, whereas removing a spurious one does not. The surviving variables define the approximate causal neighborhood $\mathcal { N } ^ { * } ( X ^ { i } )$ . The values of $d _ { \mathrm { m a x } } , \alpha _ { 1 }$ and $\alpha _ { 2 }$ used for each benchmark are listed in Supplementary Tables 1–3.

Full-system prediction. Repeating the inference for all core nodes yields neighborhoods $\{ \mathcal { N } ^ { * } ( X ^ { i } ) \} _ { i = 1 } ^ { d }$ , which we store compactly as a neighborhood matrix

$$
A _ { i j } = \left\{ \begin{array} { l l } { 2 , } & { j = i , } \\ { 1 , } & { X ^ { j } \in V ( \mathcal { N } ^ { * } ( X ^ { i } ) ) , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{6}
$$

Algorithm 1 Causal local states: neighborhood inference for core node $X ^ { i }$   
Require: Time series $\mathbf { X } = ( X ^ { 1 } , \ldots , X ^ { d } )$ , core index i, causal measure C, wrapper loss ${ \mathcal { L } } ,$ bounds $d _ { \mathrm { m a x } }$ , thresholds $\alpha _ { 1 } , \alpha _ { 2 }$   
Ensure: Approximate causal neighborhood $\mathcal { N } ^ { * } ( X ^ { i } )$   
// Filter step   
1: for $j = 1$ to d, j ̸= i do   
2: $C _ { j  i }  C ( X ^ { j }  X ^ { i } )$   
3: end for   
4: Reindex so that $C _ { ( 1 )  i } \geq \cdots \geq C _ { ( d - 1 )  i }$   
5: $\mathcal { C } _ { i } \gets \{ \mathcal { N } _ { k } \} _ { k = 0 } ^ { d _ { \operatorname* { m a x } } }$ via $\operatorname { E q . } \left( 3 \right)$   
// Wrapper selection   
6: $k ^ { * }  0$ ▷ or start at 1 if wrapper requires a minimum size   
7: for $k = 1$ to $d _ { \mathrm { m a x } }$ do   
8: train g<sub>k</sub> on ${ \mathcal { N } } _ { k } ;$ evaluate $\mathcal { L } _ { k }$   
9: if $( \mathcal { L } _ { k ^ { * } } - \mathcal { L } _ { k } ) / \mathcal { L } _ { k ^ { * } } \geq \alpha _ { 1 }$ then   
10: $k ^ { * }  k$   
11: end if   
12: end for   
// Backward elimination   
13: $\mathcal { N }  \mathcal { N } _ { k ^ { \ast } } ; \ \mathcal { L } _ { 0 }  \mathcal { L } ( \mathcal { N } )$   
14: repeat   
15: for all $X ^ { j } \in V ( \mathcal { N } ) , j \neq i$ do   
16: $\mathscr { L } _ { - j }  \mathscr { L } ( \mathcal { N } ^ { ( - j ) } )$   
17: end for   
18: m ← arg min<sub>j</sub> $\ : \mathcal { L } _ { - j } \ :$   
19: if $( \mathcal { L } _ { - m } - \dot { \mathcal { L } } _ { 0 } ) / \dot { \mathcal { L } } _ { 0 } \leq \alpha _ { 2 }$ then   
20: $\mathcal { N }  \mathcal { N } ^ { ( - m ) } ; \mathcal { L } _ { 0 }  \mathcal { L } _ { - m }$   
21: else   
22: break   
23: end if   
24: until no candidate satisfies the removal criterion   
25: return $\mathcal { N } ^ { \ast }  \mathcal { N }$

Using this matrix, the full system is forecast with the local-states approach: a model $g _ { i }$ is trained for each node on its neighborhood alone, one-step predictions are concatenated across nodes, and the result is fed back recursively to generate multi-step forecasts of the full system $( \mathrm { F i g . 1 f , g ) }$ .

## Next-Generation Reservoir Computing

NGRC, introduced by Gauthier et al.<sup>11</sup>, is the predictive model we use throughout this work, both as the wrapper in the inference steps and as the per-node model in the full-system forecast. Unlike classical RC, NGRC has no random recurrent reservoir. Its feature vector is instead built explicitly from the input data through time-delay coordinates and polynomial nonlinearities. This design is motivated by the equivalence between RC and nonlinear vector autoregression<sup>28</sup>, and it reaches comparable forecast skill with less training data and fewer hyperparameters than a classical reservoir.

Let $\mathbf { X } _ { t } \in \mathbb { R } ^ { d }$ be the input state at discrete time t. NGRC forms a feature vector $\mathbb { O } _ { t }$ by concatenating a constant bias c, a linear part, and a nonlinear part,

$$
\begin{array} { r } { \mathbb { O } _ { t } = c \oplus \mathbb { O } _ { \operatorname* { l i n } , t } \oplus \mathbb { O } _ { \operatorname { n l } , t } , } \end{array}\tag{7}
$$

where ⊕ denotes concatenation. The linear part is a time-delay embedding of the input,

$$
\mathbb { O } _ { \mathrm { l i n } , t } = \bigoplus _ { i = 1 } ^ { k } { \mathbf { X } } _ { t - ( i - 1 ) s } \in \mathbb { R } ^ { d k } ,\tag{8}
$$

with k delays spaced s steps apart. The nonlinear part collects the unique monomials of the linear features. For a polynomial order p we write $\mathbb { O } _ { \mathrm { n l } , t } ^ { ( p ) } = \lceil \bigotimes _ { i = 1 } ^ { p } \mathbb { O } _ { \mathrm { l i n } , t } \rceil$ , where ⌈·⌉ keeps one copy of each distinct monomial, and concatenate over a chosen set

of orders $\mathcal { P }$ ,

$$
\mathbb { O } _ { \mathrm { n l } , t } = \bigoplus _ { p \in \mathcal { P } } \mathbb { O } _ { \mathrm { n l } , t } ^ { ( p ) } .\tag{9}
$$

The delays $k ,$ the spacing s, and the order set $\mathcal { P }$ are the hyperparameters of the feature map. As in classical RC, only the linear readout is trained,

$$
\begin{array} { r } { \tilde { \mathbf { Y } } _ { t } = \mathbf { W } _ { \mathrm { o u t } } \mathbb { O } _ { t } , } \end{array}\tag{10}
$$

with $\mathbf { W _ { \mathrm { o u t } } }$ obtained by ridge regression with regularization $\beta .$ Multi-step forecasts are produced in closed loop by feeding the prediction back as the next input. The values of $k , s , \mathcal { P }$ and $\beta$ used for each benchmark, separately for the wrapper and for the local-states forecast, are listed in Supplementary Tables 1–3.

Feature-space size. The nonlinear expansion gives NGRC its accuracy, but it also limits the method on high-dimensional systems as the feature space grows in size combinatorially. A monomial of order p corresponds to drawing p factors from the dk linear components with replacement and without regard to order, so the order-p block contributes $\binom { p + d k - 1 } { p }$ features, and the full nonlinear feature vector has size

$$
\left| \mathbb { O } _ { \mathrm { n l } , t } \right| = \sum _ { p \in \mathcal { P } } { \binom { p + d k - 1 } { p } } .\tag{11}
$$

A monolithic NGRC model over the full state quickly produces feature vectors and design matrices too large to hold in memory.   
Restricting each model to a small neighborhood keeps d small and the full-system forecast tractable.

Inference mode. The wrapper model during network inference ideally does not predict the core node from its own past but only from its candidate neighbors. NGRC supports this directly through an inference mode that learns a mapping between disjoint sets of variables, $\mathbf { X } _ { t } \in \mathbb { R } ^ { q } \mapsto \mathbf { Y } _ { t } \in \mathbb { R } ^ { p }$ , rather than forecasting the input forward. Because each prediction depends only on the current input and its delays, the time steps are independent and can be evaluated in parallel, which is well suited to the many one-step wrapper evaluations of the inference stage. Using inference mode in the wrapper also ensures that a candidate neighbor’s contribution is judged on its own predictive value and is not masked by the self-influence of the core node (Supplementary Note 3).

## Causality measures

The filter step of CLS (Eq. (2)) admits any directed, bivariate causality measure as a scoring function. In principle, the filter step would also admit domain-specific scoring functions (e.g. physical distance), but this was not further researched. We briefly review two common bivariate causality measures that are used in this work.

Transfer entropy. Transfer entropy<sup>7</sup> is a model-free, information-theoretic generalization of Granger causality. It measures the reduction in uncertainty about the next state of X gained from the past of Y, beyond the past of X,

$$
T _ { Y  X } = H \Big ( X _ { n + 1 } \mid X _ { n } ^ { ( k ) } \Big ) - H \Big ( X _ { n + 1 } \mid X _ { n } ^ { ( k ) } , Y _ { n } ^ { ( l ) } \Big ) ,\tag{12}
$$

where $X _ { n } ^ { ( k ) }$ and $Y _ { n } ^ { ( l ) }$ are delay vectors of orders k and l, and H is the Shannon entropy. We estimate it with a histogram-based estimator.

Convergent cross mapping. Convergent cross mapping $( \mathbf { C C M } ) ^ { 3 }$ targets systems in which cause and effect are dynamically inseparable, where Granger-type measures break down. Using Takens’ delay-embedding theorem<sup>21</sup>, it reconstructs the shadow manifold of each variable and tests whether states of X can be recovered from the nearest-neighbor structure of Y. If X causally drives Y, this cross-map skill, calculated as the correlation $\rho _ { X  Y }$ between true and cross-mapped values, increases and converges as the library length grows. Because we only compare causal influences across candidate neighbors, we hold the embedding dimension, delay, and library length fixed and use the cross-map skill at fixed library size as the score. We acknowledge that this is an unusual usage of CCM, as one would usually increase the library length and investigate convergence. However, since the filter step is not interested in causal discovery but rather ranking the variables relative to each other, the usual workflow behind CCM had to be adapted.

## Prediction quality metrics

Two metrics are used to quantify short-term predictive performance throughout this work: the mean squared error (MSE) and the valid prediction steps (VPS) with the associated valid prediction time (VPT).

Mean squared error. Let $\mathbf { X } _ { t } \in \mathbb { R } ^ { d }$ denote the true system state at time step t and $\tilde { \mathbf { X } } _ { t } \in \mathbb { R } ^ { d }$ the corresponding prediction. The MSE over a prediction horizon of $N _ { \mathrm { p r e d } }$ steps starting at $t _ { 0 }$ is

$$
\mathbf { M S E } = \frac { 1 } { N _ { \mathrm { p r e d } } } \sum _ { k = 1 } ^ { N _ { \mathrm { p r e d } } } \left. \tilde { \mathbf { X } } _ { t _ { 0 } + k } - \mathbf { X } _ { t _ { 0 } + k } \right. _ { 2 } ^ { 2 } .\tag{13}
$$

The MSE also serves as the wrapper loss $\mathcal { L }$ in the neighborhood inference steps of CLS: there, it is evaluated on the predicted trajectory of the core node alone $( d = 1 )$ , so that the wrapper score reflects only the quality of the core-node prediction and is unaffected by the other variables in the candidate neighborhood.

Valid prediction steps and valid prediction time. While the MSE averages over a fixed horizon, the VPS measures how long a closed-loop forecast remains close to the true trajectory. Given a threshold $\varepsilon > 0$ , the VPS is defined as the number of time steps until the prediction error exceeds the threshold,

$$
k _ { \nu } = \operatorname* { m i n } \left\{ k \in \left\{ 1 , \ldots , N _ { \mathrm { p r e d } } \right\} \big | \left\| \tilde { \mathbf { X } } _ { t _ { 0 } + k } - \mathbf { X } _ { t _ { 0 } + k } \right\| _ { 2 } > \varepsilon \right\} ,\tag{14}
$$

and is set to $N _ { \mathrm { p r e d } }$ if the error remains below ε for the entire horizon. The corresponding valid prediction time is $t _ { \nu } = k _ { \nu } \Delta t ,$ , with ∆t the sampling interval. Where indicated, we express $t _ { \nu }$ in units of the maximal Lyapunov time $1 / \lambda _ { \operatorname* { m a x } }$ of the system. The threshold ε used for each experiment is reported together with the corresponding results.

## Benchmark systems

We validate CLS on three systems of increasing difficulty. All parameter values are provided in tables 1, 2, 3 in the supplementary information.

Composite Lorenz63–Rössler attractor pairs. The first benchmark is a system composed of N non-interacting Lorenz63– Rössler attractor pairs. Each pair consists of a three-dimensional Lorenz63 subsystem<sup>29</sup>,

$$
\dot { L } _ { x } = \sigma ( L _ { y } - L _ { x } ) \ , \qquad \dot { L } _ { y } = L _ { x } ( \rho - L _ { z } ) - L _ { y } \ , \qquad \dot { L } _ { z } = L _ { x } L _ { y } - \beta L _ { z } \ ,\tag{15}
$$

and a three-dimensional Rössler subsystem with state variables $( R _ { x } , R _ { y } , R _ { z } ) ^ { 3 0 } ,$

$$
\dot { R } _ { x } = - R _ { y } - R _ { z } \ , \qquad \dot { R } _ { y } = R _ { x } + a R _ { y } \ , \qquad \dot { R } _ { z } = b + R _ { z } ( R _ { x } - c ) \ .\tag{16}
$$

The two subsystems within a pair share no dynamical coupling, and distinct pairs are likewise non-interacting, so the full 6N-dimensional system decomposes exactly into 2N independent three-dimensional blocks with a block-diagonal ground-truth interaction network.

Lorenz96. The Lorenz96 (L96) system<sup>22</sup> consists of N variables arranged on a one-dimensional periodic lattice and evolving according to

$$
\dot { x } _ { i } ( t ) = \left( x _ { i + 1 } ( t ) - x _ { i - 2 } ( t ) \right) x _ { i - 1 } ( t ) - x _ { i } ( t ) + F , \qquad i = 1 , \dots , N ,\tag{17}
$$

with cyclic indices $x _ { i + N } = x _ { i }$ and a constant forcing F that controls the degree of chaos. This produces a banded, periodic ground-truth neighborhood matrix. We study a chaotic regime with $N = 4 0$ and $F = 5$ , where we find a largest Lyapunov exponent of $\lambda _ { m a x } \approx 0 . 4 2 ^ { 3 1 }$

Higher-order Kuramoto model of the UK power grid. The third benchmark is a dynamical model of the UK power grid in the form discussed by Li et al.<sup>20,</sup> <sup>23</sup>, based on the topology of the UK high-voltage power transmission grid<sup>32,</sup> <sup>33</sup>, comprising 120 nodes connected by 165 undirected edges. The grid dynamics generalize the Kuramoto model<sup>34</sup> to a higher-order network<sup>35,</sup> <sup>36</sup>,

$$
\dot { \theta _ { i } } ( t ) = \omega _ { i } + \gamma _ { 1 } \sum _ { j } A _ { i j } \left[ \sin ( \theta _ { j } - \theta _ { i } - \alpha ) + \sin \alpha \right] + \gamma _ { 2 } \sum _ { ( j , k ) } B _ { i j k } \left[ \sin ( \theta _ { j } + \theta _ { k } - 2 \theta _ { i } - \beta ) + \sin \beta \right] \quad i = 1 , \dots , N ,\tag{18}
$$

where $\theta _ { i } ( t )$ and $\omega _ { i }$ denote the phase and natural frequency of oscillator i. The coefficients $\gamma _ { 1 }$ and $\gamma _ { 2 }$ control pairwise and three-node interactions, respectively, while $A _ { i j }$ is the adjacency matrix and $B _ { i j k }$ the triple-coupling support tensor encoding the higher-order couplings. For some coupling regimes, such higher-order systems exhibit extremely complex dynamics<sup>35</sup>. The natural frequencies are drawn once per run as $\omega _ { i } = s _ { i } a _ { i } .$ , with absolute values $a _ { i } \sim \mathcal { U } [ \omega _ { \mathrm { a b s , l o w } } , \omega _ { \mathrm { a b s , l o w } } + 1 ]$ and signs $s _ { i } = \pm 1$ with equal probability, independently across oscillators. With $\omega _ { \mathrm { a b s , l o w } } = 0 . 3$ the resulting distribution is symmetric about zero but excludes a band around it, so no oscillator is close to stationary. The entries of $B _ { i j k }$ are either 0 or 1, and there are six higher-order triangles in the grid: $B _ { i j k } = 1$ , for {14,111,112}, {33,30,32}, {49,57,114}, {81,82,117}, {84,87,88} and {100,95,109}. For details of the parameters of the model, see Tab. 3 and Li et al. <sup>20,</sup> <sup>23</sup>.

## Author contributions statement

J. B. developed the algorithm and performed the experimental studies. F. F. and S. B. helped with conceptualization of the initial algorithm and coding. C. R. initiated and supervised the work. All authors interpreted the findings and wrote the manuscript.

## Data and Code Availability

The data and code are available from the corresponding author upon reasonable request.

## Acknowledgments

We acknowledge helpful discussions with Max Mynter and Haochun Ma regarding causality measures.

## Declaration of Interests

The German Aerospace Center has filed a patent application related to this work. C. R., S. B. and D. K. have financial interests as co-founders of Entrox Systems, which is commercializing RC. The remaining authors declare no competing interests.

## References

1. Witthaut, D. et al. Collective nonlinear dynamics and self-organization in decentralized power grids. Rev. Mod. Phys. 94, 015005, DOI: 10.1103/RevModPhys.94.015005 (2022).

2. Ghil, M. & Lucarini, V. The physics of climate variability and climate change. Rev. Mod. Phys. 92, 035002, DOI: 10.1103/RevModPhys.92.035002 (2020).

3. Sugihara, G. et al. Detecting causality in complex ecosystems. Science 338, 496–500, DOI: 10.1126/science.1227079 (2012).

4. Shahi, S., Fenton, F. H. & Cherry, E. M. Prediction of chaotic time series using recurrent neural networks and reservoir computing techniques: A comparative study. Mach. learning with applications 8, 100300 (2022).

5. Kong, L.-W., Fan, H.-W., Grebogi, C. & Lai, Y.-C. Machine learning prediction of critical transition and system collapse. Phys. Rev. Res. 3, 013090 (2021).

6. Granger, C. W. J. Investigating causal relations by econometric models and cross-spectral methods. Econometrica 37, 424–438 (1969).

7. Schreiber, T. Measuring information transfer. Phys. Rev. Lett. 85, 461–464, DOI: 10.1103/PhysRevLett.85.461 (2000).

8. Zanga, A., Ozkirimli, E. & Stella, F. A survey on causal discovery: Theory and practice. Int. J. Approx. Reason. 151, 101–129, DOI: 10.1016/j.ijar.2022.09.004 (2022).

9. Runge, J. Causal network reconstruction from time series: From theoretical assumptions to practical estimation. Chaos: An Interdiscip. J. Nonlinear Sci. 28, 075310, DOI: 10.1063/1.5025050 (2018).

10. Jaeger, H. & Haas, H. Harnessing nonlinearity: Predicting chaotic systems and saving energy in wireless communication. science 304, 78–80 (2004).

11. Gauthier, D. J., Bollt, E., Griffith, A. & Barbosa, W. A. Next generation reservoir computing. Nat. communications 12, 5564, DOI: 10.1038/s41467-021-25801-2 (2021).

12. Barbosa, W. A. & Gauthier, D. J. Learning spatiotemporal chaos using next-generation reservoir computing. Chaos: An Interdiscip. J. Nonlinear Sci. 32, 093137, DOI: 10.1063/5.0098707 (2022).

13. Kohavi, R. & John, G. H. Wrappers for feature subset selection. Artif. Intell. 97, 273–324, DOI: https://doi.org/10.1016/ S0004-3702(97)00043-X (1997).

14. Guyon, I. & Elisseeff, A. An introduction to variable and feature selection. J. Mach. Learn. Res. 3, 1157–1182 (2003).

15. Trunk, G. V. A problem of dimensionality: a simple example. IEEE Transactions on Pattern Analysis Mach. Intell. PAMI-1, 306–307, DOI: 10.1109/TPAMI.1979.4766926 (1979).

16. Pathak, J., Hunt, B., Girvan, M., Lu, Z. & Ott, E. Model-free prediction of large spatiotemporally chaotic systems from data: A reservoir computing approach. Phys. Rev. Lett. 120, 024102, DOI: 10.1103/PhysRevLett.120.024102 (2018).

17. Baur, S. & Räth, C. Predicting high-dimensional heterogeneous time series employing generalized local states. Phys. Rev. Res. 3, DOI: 10.1103/physrevresearch.3.023215 (2021).

18. Srinivasan, K. et al. Parallel machine learning for forecasting the dynamics of complex networks. Phys. Rev. Lett. 128 164101, DOI: 10.1103/PhysRevLett.128.164101 (2022).

19. Chu, K.-J., Akashi, N. & Yamamoto, A. Incorporating coupling knowledge into echo state networks for learning spatiotemporally chaotic dynamics. Chaos: An Interdiscip. J. Nonlinear Sci. 35, 093138, DOI: 10.1063/5.0273343 (2025).

20. Li, X. et al. Higher-order granger reservoir computing: simultaneously achieving scalable complex structures inference and accurate dynamics prediction. Nat. Commun. 15, 2506 (2024).

21. Takens, F. Detecting strange attractors in turbulence. In Rand, D. & Young, L.-S. (eds.) Dynamical Systems and Turbulence, Warwick 1980, 366–381 (Springer Berlin Heidelberg, Berlin, Heidelberg, 1981).

22. Lorenz, E. N. Predictability: A problem partly solved. In Proceedings of the Seminar on Predictability, vol. 1, 1–18 (European Centre for Medium-Range Weather Forecasts, Reading, UK, 1996).

23. Li, X. Higher-order granger reservoir computing: analysis code, DOI: 10.5281/zenodo.10685734 (2024).

24. Zhang, Y. & Cornelius, S. P. Catch-22s of reservoir computing. Phys. Rev. Res. 5, 033213, DOI: 10.1103/PhysRevResearch. 5.033213 (2023).

25. Köglmayr, D., Spahic, M., Flynn, A. & Räth, C. Two-shot learning of multiple strange attractors (2026). 2601.21117.

26. Assaad, C. K., Devijver, E. & Gaussier, E. Entropy-based discovery of summary causal graphs in time series. Entropy 24, 1156, DOI: 10.3390/e24081156 (2022).

27. Pearl, J. Causality (Cambridge University Press, 2009), 2 edn.

28. Bollt, E. On explaining the surprising success of reservoir computing forecaster of chaos? the universal machine learning dynamical system with contrast to var and dmd. Chaos: An Interdiscip. J. Nonlinear Sci. 31, DOI: 10.1063/5.0024890 (2021).

29. Lorenz, E. N. Deterministic nonperiodic flow. J. atmospheric sciences 20, 130–141 (1963).

30. Rössler, O. E. An equation for continuous chaos. Phys. Lett. A 57, 397–398, DOI: 10.1016/0375-9601(76)90101-8 (1976).

31. Marwan, N., Kurths, J. & Foerster, S. Analysing spatially extended high-dimensional dynamics by recurrence plots. Phys. Lett. A 379, 894–900, DOI: https://doi.org/10.1016/j.physleta.2015.01.013 (2015).

32. Simonsen, I., Buzna, L., Peters, K., Bornholdt, S. & Helbing, D. Transient dynamics increasing network vulnerability to cascading failures. Phys. Rev. Lett. 100, 218701, DOI: 10.1103/PhysRevLett.100.218701 (2008).

33. Rohden, M., Sorge, A., Timme, M. & Witthaut, D. Self-organized synchronization in decentralized power grids. Phys. Rev. Lett. 109, 064101, DOI: 10.1103/PhysRevLett.109.064101 (2012).

34. Kuramoto, Y. Self-entrainment of a population of coupled non-linear oscillators. In Araki, H. (ed.) International Symposium on Mathematical Problems in Theoretical Physics, 420–422 (Springer Berlin Heidelberg, Berlin, Heidelberg, 1975).

35. Skardal, P. S. & Arenas, A. Higher order interactions in complex networks of phase oscillators promote abrupt synchro nization switching. Commun. Phys. 3, 218 (2020).

36. Battiston, F. et al. Collective dynamics on higher-order networks. Nat. Rev. Phys. 8, 146–159, DOI: 10.1038/ s42254-025-00916-3 (2026).

## Supplementary Information

## Supplementary Note 1: A global causal threshold cannot reconstruct even a simple network

The approaches of Srinivasan et $\mathrm { a l . } ^ { 1 }$ and Chu et ${ \mathrm { a l . } } ^ { 2 }$ determine every neighborhood of the network with a single global hyperparameter. Srinivasan et al. propose a setup where they calculate a causal score for all system links. Then only those links that exceed a global threshold are retained, with the global threshold being a hyperparameter that can be optimized. Chu et al. follow a similar approach where instead of selecting by absolute causal value, each core node’s neighborhood is built by the $q$ highest-scoring candidates. The neighborhood size can then also be optimized as a global hyperparameter. Here we demonstrate, on the composite Lorenz63–Rössler system, why neither strategy can recover the structure of a heterogeneous system and, thus more generally, why the neighborhood criterion must instead be local to each core node.

The analysis is based on the transfer entropy (TE) matrix of a single attractor pair, reproduced with its values in Fig. 1a; it is the same matrix that is shown as a heatmap in Fig. 2c of the main text. According to the governing equations, the parents of the Rössler coordinate $R _ { x }$ are $R _ { y }$ and $R _ { z }$ . TE recovers $R _ { y } \ ( 1 . 0 3 )$ but scores the second parent $R _ { z }$ at only 0.37, below all three of the entirely independent Lorenz variables $( L _ { x } = 0 . 4 \dot { 6 } , L _ { z } = 0 . 4 3 , L _ { y } = 0 . 4 2 )$ . A global neighborhood size, therefore, cannot separate the attractors: $q = 1$ misses the true neighbor $R _ { z } ,$ while $q = 5 ,$ , the smallest size that includes $R _ { z } ,$ necessarily includes the three spurious Lorenz variables ranked above it (Fig. 1b). No value of $q$ recovers the neighborhood of $R _ { x }$ correctly. While such misclassifications may appear negligible in six dimensions, false positives accumulate as the system dimension grows, occupying an increasing fraction of each local model’s feature vector, degrading prediction performance and obscuring the inferred structure.

A global threshold fails for a similar reason. The two attractors evolve on different scales, so TE values are not comparable across the subsystems. The highest threshold that still captures the true neighbor $R _ { x }$ of the core node $R _ { z }$ is 0.28, the value of that entry; at this threshold the neighborhood of $L _ { x }$ already contains the spurious Rössler variables $R _ { \mathrm { y } } ~ ( 0 . 3 9 )$ and $R _ { x } \left( 0 . 3 9 \right)$ both of which have a higher TE score than $R _ { z }$ in the $R _ { x }$ neighborhood (Fig. 1c). Conversely, any global threshold high enough to ensure a correct neighborhood of $L _ { x }$ produces false negatives for the neighborhood of $R _ { z }$ . No single threshold recovers both neighborhoods correctly.

## Supplementary Note 2: Why a network reconstruction with causal-discovery algorithms is not optimal for prediction

Causal-discovery algorithms aim to recover the true causal graph of a system. They are designed and benchmarked for structural correctness rather than for forecast accuracy, and the structure best suited for prediction, especially from finite data, need not coincide with the graph they return.

Causal discovery optimises structure recovery, not prediction. Constraint-based algorithms such as the PC algorithm<sup>3</sup> and its time-series extension $\mathrm { P C M C I ^ { 4 } }$ decide the presence of each edge with conditional-independence tests carried out at a single global significance level $\alpha ,$ applied uniformly to every variable. Their objective is to recover the true graph, and they are evaluated with structural metrics accounting for every misplaced edge equally, regardless of how much that edge matters for forecasting.

Causal Markov Condition. One might object that the true causal graph is already optimal for prediction. This intuition rests on the causal Markov condition $\cdot ^ { 5 , 6 }$ , which states that in a causal graph every variable X is conditionally independent of it non-descendants $\operatorname { N D } ( X )$ given its parents,

$$
X \ \perp \perp \ ( \mathrm { N D } ( X ) \setminus \mathrm { P a } ( X ) ) | \ \mathrm { P a } ( X ) .\tag{1}
$$

It implies that the parents of a node are a sufficient input for predicting it, and it is what justifies decomposing a d-dimensional forecasting problem into d local neighborhood problems. The true causal parents are therefore an optimal predictor set. But Eq. (1) is a statement about conditional independencies in the infinite-data limit. In a finite or real data setting, a causal discovery algorithm is not optimal for inferring a graph that facilitates prediction. First, the effects on the prediction quality differ greatly between edges. In a finite or real data setting, a causal discovery algorithm is expected to make mistakes. Trying to recover all edges with equal priority might therefore harm predictive performance, and an algorithm that takes predictive performance into account during the network reconstruction is preferable. Second, causal discovery algorithms try to specifically filter out hidden confounder effects, as they are not true causal links. However, while such a link is spurious, it can still carry usable predictive information. The next two paragraphs illustrate those points.

![](images/5c372f22398fa891f3c5a902f941061d27093de3a3789bbb73d7e510661b9c01.jpg)  
Supplementary Figure 1. No global hyperparameter recovers the neighborhoods of the composite Lorenz63–Rössler system. a, The transfer entropy (TE) matrix of a single attractor pair, from which both reconstructions below are read off. Entry $( i , j )$ is the TE from source node j to target node $i ,$ in the convention of Fig. 2c of the main text, which shows the same matrix without values; cell shading is proportional to the entry, and the black outlines mark the two ground-truth blocks. Diagonal entries are not scored. $\mathbf { b } ,$ Global neighborhood size: with $q = 1$ the true neighbor $R _ { z }$ of the core node $R _ { x }$ is missed; with $q = 5 ,$ the smallest size that includes $R _ { z }$ , the three independent Lorenz variables are admitted as well, since TE ranks them above $R _ { z } .$ No q recovers the neighborhood of $R _ { x }$ correctly. c, Global threshold: 0.28 is the highest threshold that still captures the true neighbor $R _ { x }$ of the core node $R _ { z }$ (top); at this value the neighborhood of L (bottom) already contains spurious Rössler variables. No single threshold recovers both neighborhoods without introducing spurious edges in one or false negatives in the other.

Importance of edges for prediction. This paragraph briefly answers the question if all edges are of equal importance for prediction. For this experiment we trained two reservoir-computing local-states models, one supplied with the true network and the other with one selected edge missing from the true network. Then the change in the per-node valid prediction steps (VPS) relative to the full-network baseline is recorded (a node counts as affected if its VPS changes by more than five steps). Figure 2 shows three representative removals. Removing a single edge can sharply degrade the forecast of the nodes around it, and different edges affect very different numbers of nodes and by different degrees. The summary statistics across the full UK power grid are provided in Figure 3, which confirms that predictive importance seems to vary widely across edges. Another instructive case arises in systems whose nodes can synchronize. Consider a cluster of neighbors of a core node i whose states are (generalized-)synchronized such that each of these nodes is a smooth function of any other, $X _ { k } ( t ) = \psi _ { k j } ( X _ { j } ( t ) )$ . Including one representative of the cluster in the neighborhood of i can be highly valuable, as it captures the cluster’s entire influence on the core node. Every further member, however, is (approximately) a deterministic function of a variable already conditioned on so its additional predictive information vanishes. From the standpoint of network reconstruction all cluster edges are equall real and equally worth recovering, whereas for prediction only one of them matters. An algorithm that selects neighborhoods by forecast skill naturally exploits this redundancy structure, whereas classical causal discovery cannot.

Spurious links can improve prediction. A hidden confounder is an unmeasured cause of two or more observed variables, which might introduce spurious links into the inferred network. While the assumption of causal sufficiency would prevent such a situation, it is rarely possible to observe all variables relevant to a real system<sup>6</sup>. A minimal example makes this precise (Fig. 4). Let $H _ { t }$ be a hidden process evolving independently, $H _ { t } \sim \mathcal { N } ( 0 , 1 )$ , and let two observed variables be generated as

$$
\begin{array} { r } { X _ { t } = H _ { t - 1 } + \varepsilon _ { t } ^ { X } , } \\ { Y _ { t } = H _ { t - 2 } + \varepsilon _ { t } ^ { Y } , } \end{array}\tag{2}
$$

(3)

with independent noise $\varepsilon _ { t } ^ { X } , \varepsilon _ { t } ^ { Y } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ . The only true causal links are $H _ { t - 1 } \to X _ { t }$ and $H _ { t - 2 } \to Y _ { t } ;$ there is no direct influence of X on Y. Nevertheless, because $X _ { t }$ and $Y _ { t + 1 }$ are both echoes of the same hidden state $( X _ { t }$ reflects $H _ { t - 1 }$ , and so does $Y _ { t + 1 } = H _ { t - 1 } + \varepsilon _ { t + 1 } ^ { Y } )$ , the two are statistically dependent, $\mathbb { E } [ Y _ { t + 1 } \mid X _ { t } ] \neq \mathbb { E } [ Y _ { t + 1 } ]$ . Using X to predict Y therefore improves the forecast even though the link $X  Y$ is spurious. A causal-discovery algorithm that correctly reports the absence of a direct $X  Y$ edge would discard exactly the information that helps predict $Y .$ . The same effect arises even without hidden variables, through Takens’ embedding theorem, and is visible in the Lorenz96 benchmark of the main text, where causal local states (CLS)

![](images/6b96e49017bcc7f4e1a84d49e1cfcf42c9cc7c4feeea5f8b50fb9e8d6c68a363.jpg)

Edge (4, 5)  
![](images/7c349e0ed1f72a37b20c161b08498493eed5f9f3a68a438cf08097cedf624660.jpg)

![](images/307d22c6b9fff3a19e7313bd307d8d17a0b26bd6872f0f3b4a50611b80b982f9.jpg)

![](images/6ef2e4fb6249c048975ab77bd504a65b4fb86edf080f142de255dc6ea7e4a8aa.jpg)  
Supplementary Figure 2. Edge-importance experiment on the UK power grid with three representative single-edge removals shown. In each case, two reservoir-computing local-state models are trained, one with the true adjacency matrix and one with the selected edge removed; a node is counted as affected if its valid prediction steps (VPS) change by more than a threshold $( \mathrm { V P S } _ { \mathrm { t h } } = 5 )$ . Left: the removed edges and the corresponding affected nodes highlighted on the UK power grid network. Right: node-wise change in VPS for each removal. Some edges affect many nodes, others only a few.

retains nearby non-parent nodes that improve the forecast (Fig. 3).

## Supplementary Note 3: Choice of the wrapper model and self-influence

The wrapper that is used to score candidate neighborhoods can in principle be any predictive model. We compare two different established models: reservoir computing (RC) and next-generation reservoir computing (NGRC). NGRC reaches comparable accuracy at lower training cost and with fewer hyperparameters, which already suits the many wrapper evaluations CLS performs. It is not immediately clear how a wrapper model should be used to quantify neighborhood quality. Depending on the strategy, the time-delayed values of the core node itself are included or excluded from the neighborhood (which we call self-influence).

We compare three sample strategies:

• One-step (RC or NGRC). The full neighborhood is reset to the true values at every step and the model makes a one-step forecast of the core node.

• Multi-step (RC or NGRC). The neighborhood is reset only every $N _ { \mathrm { r e s e t } }$ steps, so the forecast error is allowed to accumulate between resets.

• NGRC inference mode. NGRC is used in inference mode, which learns a map from the neighbors alone to the next core state and therefore excludes the core node’s own past from the feature vector. This means that there is no self-influence in this strategy.

Number of Influenced Nodes  
![](images/51322db78b61b8b8f8b5134f302f27bd6fce520ab84972969dca9be9ff961632.jpg)

![](images/2df1da27e7da3d1ca6db11371f48da10482953755908210ec9422d435bd931a9.jpg)

Supplementary Figure 3. Distribution of the change in valid prediction steps (VPS) and of the number of affected nodes across all removed edges. The importance of an edge for prediction is highly heterogeneous, so that a criterion aimed at structural recovery rather than forecast skill is suboptimal when the goal is prediction.  
![](images/1fe6382f11827227a675ee2738472f8f6227cee4089e71c2ce8473bc7f1816de.jpg)  
Supplementary Figure 4. Hidden confounder inducing a spurious but predictive link. The hidden process H drives X at lag τ = 1 and Y at lag τ = 2. There is no direct causal link $X  Y$ (dotted), yet X is predictive of the future of Y because both are echoes of H. Consequently a spurious X → Y link improves the forecast of Y.

A good wrapper should assign a clearly lower error to neighborhoods that contain the true neighbors than to those built from spurious ones. Figure 5 shows that only NGRC inference mode achieves this cleanly. Our hypothesis is that because in the one-step strategy the core node is part of the feature vector, and because a variable’s own past is usually strongly predictive of it next value, the core node dominates the forecast and masks the contribution of the neighbors. The multi-step strategy mitigates the masking slightly by letting the error accumulate between resets, which improves the separation. To confirm this, we can look at two representative sample predictions. Fig. 6 shows the mean squared error (MSE) of the core node predictions for two different cases: (i) all true neighbors are present in the neighborhood, which is highlighted in green, (ii) no true neighbors are present in the neighborhood, highlighted in red. For a functional wrapper, the prediction on the spurious neighborhood should be nonsensical. As evident from this figure, only the NGRC inference strategy, which has no self-influence, is successful in this test.

Importantly, while excluding the self-influence during the graph reconstruction is desirable, we, nevertheless, do include the core node time delays as local NGRC input during the full system forecast, as at this stage the information about the recent past of the core node becomes important for prediction and the masking effect is no longer relevant. The one-step strategy is also expensive, since resetting the neighborhood at every step forces the reservoir to resynchronize before every prediction. The multi-step strategy reduces the number of re-synchronizations (Fig. 6b), but is still slower than the NGRC inference mode.

These considerations lead us to use NGRC in inference mode as the wrapper throughout this paper and NGRC as the per-node model in the full-system forecast. However, this choice is not fundamental, as the framework stays model-agnostic and NGRC can be swapped out for other prediction models.

## Supplementary Note 4: Simulation and model hyperparameters

For reproducibility, Tables 1–3 collect the full simulation and model hyperparameters used to produce the three benchmark results reported in the main text. For each benchmark we list the parameters of the dynamical system (whose time series data are obtained by integration with the Runge-Kutta method RK4), of the causal measure used in the filter step (TE or convergent cross mapping (CCM)), of the wrapper and prediction models (NGRC and its variants), and of the CLS neighborhood-selection thresholds α<sub>1</sub> (wrapper selection) and $\alpha _ { 2 }$ (backward elimination). The benchmark systems are deterministic, so the random seed affects only the choice of the initial condition. For the UK power grid the seed sets the natural frequencies ω and the initial phases.

![](images/73438dce1d07b2b6850063a77858b7c4352160873bc0cd9a0c85f44a80ce282f.jpg)  
Supplementary Figure 5. Sample core-node predictions for a composite Lorenz63–Rössler system. For the core node $L _ { x } ,$ wrapper models are trained on a neighborhood of the true neighbors $L _ { y } , L _ { z }$ (green) and on a neighborhood of spurious neighbors $R _ { x } , R _ { y } , R _ { z }$ (red), for each of the three strategies. Under the one-step strategy the core node’s own past dominates, so even the spurious neighborhood reproduces the trajectory; under next-generation reservoir computing (NGRC) inference mode the spurious neighborhood fails to track the core trajectory at all.

During the wrapper step, the training data is split up further into a wrapper train and a wrapper test set. This is to ensure that the actual test set used to evaluate the final prediction is unseen during the wrapper step. We denote this split with the wrapper test fraction, which just identifies how much of the original training data is used to evaluate the wrapper performance.

As the objective of the paper is not to showcase the highest performance possible on the benchmarks, there was no deep hyperparameter search conducted. The hyperparameters were handpicked such that the model produced interpretable results (e.g. no divergence of the NGRC prediction).

<table><tr><td></td><td>Exhaustive wrapper validation (Fig. 2b)</td><td>Single attractor pair (Fig. 2c–e)</td><td>Scaling sweep (Fig. 2f–h)</td></tr><tr><td>System</td><td></td><td></td><td></td></tr><tr><td>Attractor pairs N</td><td></td><td>1</td><td>1-15</td></tr><tr><td>Lorenz63 dt;  $\sigma , \rho , \beta$ </td><td></td><td>0.01; 10.0, 28.0, 8/3</td><td></td></tr><tr><td>Rössler dt;  $^ { a , b , c }$ </td><td></td><td>0.05; 0.2, 0.2, 5.7</td><td></td></tr><tr><td>Initial condition, every subsystem</td><td>drawn from  $\mathcal { U } ( - 3 , 3 ) ^ { 3 }$ </td><td>with the seed below</td><td></td></tr><tr><td>Random seed</td><td>42</td><td>42</td><td> $0 , 1 , \ldots , 9$  (ten seeds)</td></tr><tr><td>Data</td><td></td><td></td><td></td></tr><tr><td>Transient steps (discarded)</td><td></td><td>30000</td><td></td></tr><tr><td>Data length (post-transient)</td><td>50000</td><td>50000</td><td>56050</td></tr><tr><td>Train / test steps</td><td>25000 / 25000</td><td>25000 / 25000</td><td>25000 / 31050</td></tr><tr><td>Normalization</td><td></td><td>zero mean, unit variance</td><td></td></tr><tr><td>Transfer entropy (filter)</td><td></td><td></td><td></td></tr><tr><td>Number of bins</td><td></td><td>79</td><td>8</td></tr><tr><td>Time delay τ</td><td></td><td>1</td><td>1</td></tr><tr><td>NGRC (wrapper &amp; local states)</td><td></td><td></td><td></td></tr><tr><td>Delays k, spacing s</td><td>5,1</td><td>3,1</td><td>3,1</td></tr><tr><td>Polynomial orders</td><td>[1,2]</td><td>[1,2,3]</td><td>[1,2,3]</td></tr><tr><td>Ridge regularisation</td><td>10−2</td><td>10−3</td><td>10-3</td></tr><tr><td>Wrapper test fraction</td><td>0.2</td><td>0.4</td><td>0.4</td></tr><tr><td>Bias term</td><td></td><td>yes</td><td></td></tr><tr><td>CLS selection</td><td></td><td></td><td></td></tr><tr><td>α1, α2</td><td></td><td>0.05, 0.05</td><td>0.05, 0.05</td></tr><tr><td>Candidate bound  $d _ { \mathrm { m a x } }$ </td><td></td><td>unbounded  $( = d - 1 = 5 )$ </td><td>10</td></tr><tr><td>Forecast and evaluation</td><td></td><td></td><td></td></tr><tr><td>Forecast horizon  $N _ { \mathrm { p r e d } }$ </td><td></td><td>5000</td><td>3000</td></tr><tr><td>VPT launch points^× warmup</td><td></td><td>10 × 50 steps</td><td></td></tr><tr><td>VPT threshold ε</td><td></td><td>0.3</td><td>0.3</td></tr><tr><td>Monolithic NGRC baseline</td><td></td><td>k=3,s=1, orders [1,2,3]</td><td>k=3,s=1, orders [1]</td></tr><tr><td></td><td></td><td>ridge 10−3</td><td>ridge 10−3</td></tr></table>

Supplementary Table 1. Simulation and model hyperparameters for the composite Lorenz63–Rössler benchmark. NGRC, next-generation reservoir computing; CLS, causal local states; VPT, valid prediction time.

![](images/69c11cdc705e88bc8b1935789211a4a4678df064a5c7e64f2574de2e89e70319.jpg)

![](images/bf5b045796adee2a0bcb93aecb9646f5f1c492d0d25695041ff17ebb2d3972bb.jpg)

![](images/0445865f87cb8e75ce584f6b8714893b489944e5a1310c948c729a29672ad026.jpg)

![](images/fa7a6e0e861a39baddb733949880948cc0c9db0a9f52a62272edbaf56ab4510d.jpg)

![](images/1376bd3b81617250f52d13091d4b6e8414d385271300686a1d72daef172a558a.jpg)  
(a) One-step strategy

![](images/3a9093b813576488a0bad682b58911d83393eaf03800b63b048d0f622f24648d.jpg)

![](images/03a6dc743e967337396dc0a0b0861f226d8d8da80835117ca14bc03b413a5698.jpg)

![](images/1a62d239561afea4f273d91cd4ee526b54c9fd547a90b6c369e2054c2a6e98e5.jpg)

![](images/b0eea38ef4f98695badb7e424558a19618d7a21a9fc8ef251150395cecfd9a67.jpg)

![](images/b27e6beeef4c8dd47431de9c604121dac20cebd3074771f0f820e53c3c0fc093.jpg)

![](images/12779d4b8da5f1ef43fd97420e1c961ee563783f00dc420c93593485e3c93fe8.jpg)  
(b) Multi-step strategy

![](images/d95ef66836d60f305b8e5287c79bd442bc696887031bdc47f16fe3892188082d.jpg)

![](images/5f2854f92447a605c3ea00cd1c2634b170f4b25b6d37115dfff9fe34fb98a313.jpg)

![](images/25a9907d580a9b1e46a4a66a8ae382a9d5a594b729efc056a3f99142c27aa151.jpg)

![](images/07dbfc1006f2b539ddcd4545a30bd5b2b83578a7fead982c95ed69b0c4ad1662.jpg)

![](images/2821f38266f709ec2fce52abc70bcaf8af80f1bb5e1a161959c9e4f4e55bce52.jpg)

![](images/8118407c094b4076d76eb2e0a50aa7e514e2f63432a598d0c228ec70a71895ef.jpg)

![](images/79cd60f689a0cd5a725cb3d748aae885b05f2b0e9a414b4d5afc9dbb6bee619f.jpg)  
(c) NGRC inference mode

Supplementary Figure 6. Core-node mean squared error (MSE) for all 32 candidate neighborhoods of the composite Lorenz63–Rössler system, evaluated for each core node and grouped by how many features come from the same attractor as the core node: none (red), one (orange, the two possible cases distinguished), or both (green). Only next-generation reservoir computing (NGRC) inference mode (c) cleanly separates the neighborhoods that contain the true neighbors from those that do not; the one-step strategy (a) barely separates them because the core node dominates the forecast, and the multi-step strategy (b) improves the separation but does not fully resolve it.

<table><tr><td>Lorenz96 system</td><td></td></tr><tr><td>Dimension N</td><td>40</td></tr><tr><td>Forcing F</td><td>5</td></tr><tr><td>Time step dt</td><td>0.05</td></tr><tr><td>Initial condition</td><td> $x _ { 0 } = F \cdot { \bf 1 } + 1 0 ^ { - 5 } \cdot \mathcal { U } ( 0 , 1 ) ^ { N }$ </td></tr><tr><td>Random seeds</td><td> $0 , 1 , . . . , 9 \left( \mathrm { t e n \ s e e d s } \right)$ </td></tr><tr><td>Transient steps (discarded)</td><td>10000</td></tr><tr><td>Train / test</td><td>20000 / 10150</td></tr><tr><td>Normalization</td><td>zero mean, unit variance</td></tr><tr><td>Transfer entropy (filter)</td><td></td></tr><tr><td>Number of bins</td><td>30</td></tr><tr><td>Time delay τ</td><td>1</td></tr><tr><td>Data used for TE estimation</td><td>train split (20000 steps)</td></tr><tr><td>NGRC (wrapper: selection &amp; elimination)</td><td></td></tr><tr><td>Delays k, spacing s</td><td>3, 1</td></tr><tr><td>Polynomial orders</td><td>[1,2]</td></tr><tr><td>Wrapper test fraction</td><td>0.3</td></tr><tr><td>Ridge regularisation</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>NGRC (local states)</td><td></td></tr><tr><td>Delays k, spacing s</td><td>5,1</td></tr><tr><td>Polynomial orders</td><td>[1,2]</td></tr><tr><td>Ridge regularisation</td><td> $\mathrm { \bar { 1 0 } ^ { - 1 } }$ </td></tr><tr><td>Bias term</td><td> $\mathrm { y e s }$ </td></tr><tr><td>CLS selection</td><td></td></tr><tr><td>α1, α2</td><td>0.03, 0.03</td></tr><tr><td>Candidate bound  $d _ { \mathrm { m a x } }$ </td><td>20</td></tr><tr><td>Evaluation</td><td></td></tr><tr><td>VPT launch points × warmup</td><td>10 × 50 steps</td></tr><tr><td>VPT threshold ε</td><td>0.3</td></tr></table>

Supplementary Table 2. Simulation and model hyperparameters for the Lorenz96 benchmark. TE, transfer entropy; NGRC, next-generation reservoir computing; CLS, causal local states; VPT, valid prediction time.

<table><tr><td></td><td>Sine-NGRC</td><td>Kuramoto-NGRC</td></tr><tr><td>UK power grid</td><td></td><td></td></tr><tr><td>Phase lags α, β</td><td>0.02, 0.06</td><td></td></tr><tr><td>Coupling γ1, γ2</td><td>0.4, 0.4</td><td></td></tr><tr><td>ωabs,low</td><td>0.3</td><td></td></tr><tr><td>Time step dt</td><td> $8 \times 1 0 ^ { - 2 }$ </td><td></td></tr><tr><td>Initial phases</td><td> $\mathcal { U } ( 0 , 2 \pi ) ^ { N } ,$ </td><td>drawn with the seed</td></tr><tr><td>Random seeds</td><td> $0 , 1 , \ldots , 9$ </td><td>(ten seeds)</td></tr><tr><td>Transient steps (discarded)</td><td>1000</td><td></td></tr><tr><td>Train / test / forecast steps</td><td>10000 / 20150 / 2000</td><td></td></tr><tr><td>Normalization</td><td>none (raw phases)</td><td></td></tr><tr><td>Convergent cross mapping</td><td></td><td></td></tr><tr><td>Input series</td><td>sin θ</td><td></td></tr><tr><td>Lag</td><td>1</td><td></td></tr><tr><td>Embedding dimension</td><td> $^ 2$ </td><td></td></tr><tr><td>Training fraction</td><td>0.75</td><td></td></tr><tr><td>Library size</td><td> $L _ { \mathrm { t r a i n } }$  (largest library only)</td><td></td></tr><tr><td>Wrapper and forecast models</td><td></td><td></td></tr><tr><td>Input series</td><td>raw phases θ</td><td></td></tr><tr><td>Regression target</td><td> $\Delta \theta _ { t } = \theta _ { t + 1 } - \theta _ { t }$ </td><td></td></tr><tr><td>Nonlinear features</td><td>polynomials of sin θi, cos θ</td><td> $\sin ( \theta _ { j } - \theta _ { i } )$ </td></tr><tr><td>Delays k, spacing s</td><td>3, 1</td><td>1,1</td></tr><tr><td>Polynomial orders</td><td>[1,2]</td><td></td></tr><tr><td>Ridge (wrapper)</td><td>10−3</td><td>10−3</td></tr><tr><td>Ridge (forecast)</td><td>10⁻3</td><td>10⁻2</td></tr><tr><td>Wrapper test fraction</td><td>0.4</td><td>0.3</td></tr><tr><td>Bias term</td><td>yes</td><td>yes</td></tr><tr><td>CLS selection</td><td></td><td></td></tr><tr><td>α1,α2</td><td>0.05, 0.05</td><td>0.05, 0.05</td></tr><tr><td>Candidate bound  $d _ { \mathrm { m a x } }$ </td><td>10</td><td>30</td></tr><tr><td>Backward elimination</td><td>greedy</td><td>greedy</td></tr><tr><td>Evaluation</td><td></td><td></td></tr><tr><td>VPT threshold ε</td><td>0.3</td><td></td></tr><tr><td>VPS / VPT evaluated on</td><td>sin θi</td><td></td></tr></table>

Supplementary Table 3. Simulation and model hyperparameters for the UK power grid benchmark. The seed fixes both the natural frequencies $\omega _ { i }$ and the initial phases. NGRC, next-generation reservoir computing; CLS, causal local states; VPS, valid prediction steps; VPT, valid prediction time.

## Supplementary References

1. Srinivasan, K. et al. Parallel machine learning for forecasting the dynamics of complex networks. Phys. Rev. Lett. 128, 164101, DOI: 10.1103/PhysRevLett.128.164101 (2022).

2. Chu, K.-J., Akashi, N. & Yamamoto, A. Incorporating coupling knowledge into echo state networks for learning spatiotemporally chaotic dynamics. Chaos: An Interdiscip. J. Nonlinear Sci. 35, 093138, DOI: 10.1063/5.0273343 (2025).

3. Spirtes, P., Glymour, C. & Scheines, R. Causation, Prediction, and Search (MIT Press, Cambridge, MA, 2000), 2nd edn.

4. Runge, J., Nowack, P., Kretschmer, M., Flaxman, S. & Sejdinovic, D. Detecting and quantifying causal associations in large nonlinear time series datasets. Sci. Adv. 5, eaau4996, DOI: 10.1126/sciadv.aau4996 (2019).

5. Neapolitan, R. E. & Jiang, X. Probabilistic methods for financial and marketing informatics (Morgan Kaufmann Publishers, San Francisco, CA, 2007).

6. Runge, J. Causal network reconstruction from time series: From theoretical assumptions to practical estimation. Chaos: An Interdiscip. J. Nonlinear Sci. 28, 075310, DOI: 10.1063/1.5025050 (2018).