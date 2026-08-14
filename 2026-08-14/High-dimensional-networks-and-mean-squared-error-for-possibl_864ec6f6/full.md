# High-dimensional networks and mean squared error for possibly misspecified models

Lourens Waldorp

University of Amsterdam, Nieuwe Achtergracht 129-B, 1018 NP, the Netherlands

## Abstract

To avoid missing important variables and their connections in networks, more and more variables are included in network analysis. Here we show that in a setting with many more parameters than observations (high-dimensional) it is possible to get a conservative (i.e., low false positive rate) estimate of the neighbourhood for each node (which connections are in the network). A neighbourhood is often estimated with a linear model, and this leads to two interesting cases: (i) If the true model is linear, then neighbourhood selection work reasonably well, and (ii) if the true model is nonlinear, then neighbourhood selection requires a penalty for the high dimensions. Here we show the impact of the ridge parameter on the mean squared error, and how this leads to low test variance and hence to neighbourhoods with large numbers of edges. We connect these insights with results from machine learning, where the so-called double descent (when more parameters are included than observations, the mean squared error goes down a second time) has put the traditional view on model selection upside down. Essentially, for adequate neighbourhood selection in models with a large number of parameters, the volume of the model space needs to be included in the penalty. Most neighbourhood selection methods (e.g., Lasso, AIC, BIC) lead to spurious edges (high false positive rate), but we prove that in the high-dimensional setting, minimum description length leads to correct neighbourhood selection or smaller (low false positive rates) in both cases when either the model is correctly or incorrectly assumed linear.

Key words: model selection, double descent, bias-variance trade-of, mean squared error, minimum description length

## 1 Introduction

Often we are concerned that the number of observations in a study is insufficient to warrant proper analysis using networks (Epskamp et al., 2018). In particular, a network contains many possible connections and therefore many parameters to be estimated. For instance, with 20 nodes, there are already 190 parameters for the edges in a network. In the classical statistical framework we would need about 1000 observations to reliably estimate these parameters.

Recent advances have brought several estimation techniques that allow for the so-called high-dimensional situation, where we have more edges than observations (overparameterisation). Examples are the least absolute shrinkage and selection operator (lasso, Tibshirani, 1996) and the ridge estimator (Hoerl and Kennard, 1970; Rao, 1990). The lasso is popular because it simultaneously estimates and selects edges to a node (Waldorp and Haslbeck, 2024). However, the lasso does not have a closed form solution and is discontinuous, which makes standard inference problematic (although there are workarounds, see e.g., B¨uhlmann et al., 2014; Dezeure et al., 2015). In contrast, the ridge estimator does have a closed solution and is continuous. This makes the ridge estimate more amenable to analysis and allows for an analytical approach to determine why generalisation is still possible in the over-parameterised regime. The ridge estimate is constructed by adding independent dimensions to the predictors so that a unique solution can be obtained (Hoerl and Kennard, 1970; Rao, 1990). The solution is biased, but can relatively easily be used for inference (B¨uhlmann, 2013). The ridge estimate can also be used for neighbourhood selection, i.e., using Akaike and Bayes information criteria and minimum description length to obtain a reliable estimate of the neighbours of each node in a network. Here we focus on reliable model selection for high-dimensional linear models used to estimate the neighbourhood of any node in the network, where the true edge relations are either linear or nonlinear.

We use the well-known nodewise regression (neighbourhood selection) framework in graphical modelling (Lauritzen, 1996; Meinshausen and B¨uhlmann, 2006; Jankova and van de Geer, 2017). In nodewise regression, the procedure to find the nodes that are connected is completely local; each node is in turn used as the dependent variable in a regression on all remaining nodes. For each node, the objective is to identify the correct set of non-zero coeficients (neighbours) from the (large) set of remaining nodes. Figure 1 shows three graphs with diferent number of nodes (15, 32 and 320) from the perspective of one node; the objective is to identify five non-zero edges in the space of all other zero edges (sparse solution). When all nodes have been the focus of a regression, all neighbourhoods can be ‘stitched’ together to form the entire graph. For nodewise selection the objective is to select the correct neighbourhood or smaller but not larger (i.e., no false positive edges), because then, in stitching together the entire graph, the number of false positive edges will be high in large graphs.

To achieve this, we use recent work on the impact of estimation in the highdimensional setting on the mean squared error (MSE) in regression. Both in Bartlett et al. (2020) and Hastie et al. (2022) the so-called ”ridgeless” estimator was used (see Section 2) to conclude that, surprisingly, the MSE remains well behaved even in high dimensions. This implies that in principle it is possible to perform reliable neighbourhood selection in the highdimensional setting. Cheema and Sugiyama (2020) investigated the geometry of high-dimensional models relating it to the impact of using a large parameter space on model selection, concluding that when the large parameter space of the overparameterised model is properly taken into account, then model selection should work. Dwivedi et al. (2020) adapted minimum description length, a model selection procedure, to be computationally eficient and reliable for large datasets. Other approaches, related to hypothesis testing, have also been investigated in the high-dimensional setting (e.g., Meinshausen and B¨uhlmann, 2010; B¨uhlmann, 2013; Shah and B¨uhlmann, 2023). However, the selection is based on false positive control by thresholding, which is often resolved by additional assumptions, such as independence for the familywise error rate or false discovery rate (e.g., DasGupta, 2008).

![](images/000ce711bbdeab1ef551478198b067918e6bf3e18e2db5e720cf4815b5f7b723.jpg)  
Fig. 1. Graphs from the perspective of nodewise regression for graph sizes 15, 32 and 320. For any node there are five non-zero coeficients (blue edges), so that each regression needs to identify among the remaining nodes the five non-zero coeficients.

Our contribution builds on these works and mostly on the work by Cheema and Sugiyama (2020). Using linear neighbourhood selection, we leverage results about the impact of high-dimensional ridge estimation on MSE, and use this to explain why overfitting (many parameters in linear models) can unexpectedly result in low MSE, causing problems for nodewise selection. Penalties in the Akaike information criterion (AIC), for instance, are inadequate and mostly select too many edges (overfit). But we prove that the penalty of the minimum description length, as given in Gr¨unwald (2007), in the highdimensional setting leads the correct neighbourhood or smaller, in both cases when the true model for edge relation is linear or nonlinear.

We briefly explain the Gaussian graphical model in Section 2 and discuss the problem of estimation in the high-dimensional setting. We describe that ridge regression leads to a possible choice in obtaining a unique estimate in high dimensions. Then in Section 3 we discuss mean squared error (MSE). We show that the MSE remains low unexpectedly after the interpolation point (where the number of parameters equals the number of observations). This MSE behaviour is directly related to the ridge parameter, and we show that the ridge solution is directly related to constraining the parameter space. To achieve this, we discuss the MSE decomposition in Section 4 in terms of squared bias and variance, also in the case of model misspecification. Then in Section 5 we turn to model selection. Here we describe minimum description length as an example where the geometry of the model space is taken into account as a penalty, and prove that this will lead to correct neighbourhood selection or smaller. In Section 6 we use simulations to confirm the theoretical results of neighbourhood selection.

## 2 Estimation in the Gaussian graphical model

The Gaussian graphical model (GGM) is a Gaussian (normal) multivariate distribution associated with a set of nodes and edges in a network. A node j in the network represents a Gaussian random variable $X _ { j } ,$ , and an edge $( i , j )$ between two nodes i and j in the network represents a conditional dependence relation between the two variables $X _ { i }$ and $X _ { j }$ given all other variables $X _ { k }$ with $k \neq i , j$

The main characterisation in a GGM of conditional dependence is the partial correlation (or partial covariance). That is why a GGM is sometimes referred to as a partial correlation network. This characterisation makes sense because whenever a partial correlation is 0, then the two variables are conditionally independent (Lauritzen, 1996, Proposition 5.2) and so no edge is present in the network; when a partial correlation is $\neq 0$ , an edge is present in the network. So, to construct a network we need to determine the 0 partial correlations. Obtaining the partial correlations can be done by estimating the inverse covariance covariance matrix by (penalised) maximum likelihood (Friedman et al., 2008; Jankova and van de Geer, 2017).

The partial correlations can also be determined from regressions. This is called neighbourhood selection or nodewise selection (Meinshausen and B¨uhlmann, 2006; B¨uhlmann et al., 2014; Maathuis et al., 2018; Waldorp and Marsman, 2021). This is because the regression coeficient is proportional to the partial covariance. Hence, if the partial covariance is 0, then the regression coeficient must also be 0 (see Appendix A for details). Thus, determining the GGM is reduced to a series of regressions, one regression for each node regressed on all $p - 1$ remaining nodes, with the objective of determining the neighbourhood. Then, when all of the neighbourhoods have been determined, they can be put together to make up the entire graph. We therefore discuss a single regression, representing all p regressions for the entire graph, selecting the nodes that are part of the neighbourhood of a node.

## 2.1 Nodewise regression

In nodewise selection, each node in turn is the dependent variable and the remaining nodes are the independent or predictor variables. We call variable

$X _ { i }$ the dependent variable and denote it by $Y$ and the remaining nodes are $X _ { j }$ for $j \neq i$ . In the scenarios we discuss we will be selecting a set of nodes that are relevant to predicting $Y$ . We let J denote the index set for the predictors used in predicting $Y$ (which equals $X _ { i } )$ with a subset of the remaining nodes. The set J never contains the dependent variable $Y = X _ { i }$ (no self loops), so that J is a subset of $\{ 0 , 1 , \dotsc , i - 1 , i + 1 , \dotsc , p \}$ . We assume there is a true subset $S ,$ referred to as the support, such that $\beta _ { j } \neq 0 \mathrm { ~ i f ~ } j \in S$ and $\beta _ { j } = 0$ if $j \not \in S$ . To obtain the estimates of $\beta _ { j }$ from the linear regression, we minimise the least squares function

$$
\operatorname { L S } ( J ) = \sum _ { i = 1 } ^ { n } ( Y _ { i } - { \hat { Y } } _ { J } ) ^ { 2 } \quad { \mathrm { ~ a n d ~ } } \quad { \hat { Y } } _ { J } = \sum _ { j \in J } X _ { j } { \hat { \beta } } _ { j } .
$$

For each model (subset) J we can obtain an estimate of the coeficients corresponding to the variables $x _ { j }$ such that $j \in J$ . By estimating $\beta _ { j }$ we obtain information on the neighbourhood of node i: for any $\beta _ { j } \neq 0$ we draw the edge $( i , j )$ ; and if $\beta _ { j } = 0$ , then there is no edge between i and j. We therefore require estimates of the $\beta _ { j }$ with $j \in J$

We can rewrite the model in matrix algebra, with $Y = ( Y _ { 1 } , Y _ { 2 } , \dots , Y _ { n } )$ the n-dimensional vector, $X = ( X _ { 1 } , X _ { 2 } , \ldots , X _ { p } )$ the $n \times p .$ -dimensional matrix and ${ \boldsymbol { \beta } } = ( \beta _ { 1 } , \beta _ { 2 } , \ldots , \beta _ { p } ) , \boldsymbol { Y } = X { \boldsymbol { \beta } }$ . The least squares function can be written in terms of the Euclidean distance function $| | a | | _ { 2 } = { \sqrt { a _ { 1 } ^ { 2 } + a _ { 2 } ^ { 2 } + \cdots a _ { p } ^ { 2 } } }$ . The least squares function is then $\mathrm { L S } ( J ) = | | Y - X \beta _ { J } | | _ { 2 } ^ { 2 }$ . Minimisation is achieved from the normal equations

$$
X ^ { \top } X \beta = X ^ { \top } Y .\tag{1}
$$

Similar to numbers on the real line R, we require that we can invert $X ^ { \top } X$ so that we obtain our estimate. For numbers on the real line we get for $x b = c ,$ that we can invert x so that $b = c / x$ . Similarly, the inverse $( X ^ { \top } X ) ^ { - 1 }$ has the property that $( X ^ { \top } X ) ^ { - 1 } ( X ^ { \top } X ) = { \dot { I } }$ , where the identity matrix I is such that $I X = X$ . Then our least squares (LS) estimate is

$$
{ \hat { \beta } } ^ { L S } = ( X ^ { \top } X ) ^ { - 1 } X ^ { \top } Y .\tag{2}
$$

It is clear from this derivation that we need the inverse $( X ^ { \top } X ) ^ { - 1 }$ . This assumes that the rank $( \mathrm { i . e . }$ , the number of linearly independent vectors in X) is at least $p .$ A necessary (but not suficient) condition is that $p < n$ . For suficiency we also require that there is no collinearity $( \mathrm { i . e . }$ , the eigenvalues of $X ^ { \top } X$ are all positive). Given these assumptions, in the linear model the estimate $\hat { \beta } ^ { L S }$ is unbiased $( \mathrm { i . e . } \mathbb { E } ( \hat { \beta } ^ { L S } ) = \beta _ { S } )$ and has smallest variance (i.e., var $( { \hat { \beta } } ^ { L S } ) \leq \operatorname { v a r } ( \beta _ { S } )$ for any other estimate $\beta _ { S } )$ . Good references for these results are, for instance, Rao (1990) and Seber and Lee (2012).

## 2.2 Estimation when $p > n$

We now know that in order to obtain an estimate $\hat { \beta }$ we require the inverse of $X ^ { \top } X$ to exist. If $p > n$ then this is not the case. As a consequence, there is no unique solution (Pringle and Rayner, 1971; Searle, 1971; Schott, 1997). We can understand this by considering the eigenvalues (spectrum) of $X ^ { \top } X$ , denoted by $\lambda _ { j }$ . If the $p \times p$ matrix $X ^ { \top } X$ has full rank $p ,$ then it has $p$ eigenvalues $\lambda _ { j } > 0$ (Schott, 1997). If $X ^ { \top } X$ does not have full rank (because in our case $p > n )$ but has rank $k ,$ say, then there are $p - k$ eigenvalues equal to 0. This refers to the idea that a part of the space spanned by $X ^ { \top } X$ projects on the null-space $( \mathrm { i . e . }$ , the space such that $X ^ { \top } X u = 0$ for any vector u $\neq 0 )$ . Hence, there is no unique solution when $X ^ { \top } X$ is $p > n$

One way to obtain a unique solution is to impose constraints on the vector $\beta .$ One of the constraints is to impose an upper bound on the sum of absolute values, i.e., $\textstyle \sum _ { j } | \beta _ { j } | \leq c ,$ , for some $c > 0$ . Then the problem

$$
\operatorname* { m i n } _ { \beta } | | Y - X \beta | | _ { 2 } ^ { 2 } \quad { \mathrm { s u c h ~ t h a t ~ } } \quad \sum _ { j \neq i } | \beta _ { j } | \leq c ,
$$

is called the least absolute shrinkage and selection operator, abbreviated by lasso (Tibshirani, 1996). The lasso has several attractive properties, like selecting the non-zero coeficients and consistency (i.e., converging with $n$ to the true value), given certain strong assumptions (see, $\mathrm { e . g . }$ , Wainwright, 2009; B¨uhlmann and van de Geer, 2011; Hastie et al., 2015). The lasso has no closed form solution and is not amenable to inference (P¨otscher and Leeb, 2009; B¨uhlmann et al., 2014). However, several workarounds have been obtained, like the debiased lasso (van de Geer et al., 2014; Waldorp and Haslbeck, 2024) and the multi sample-split (Meinshausen et al., 2009; Dezeure et al., 2015).

Another constraint that can be imposed on the vector $\beta$ is to upper bound the sum of squares of the parameters. This leads to the problem

$$
\operatorname* { m i n } _ { \beta } | | Y - X \beta | | _ { 2 } ^ { 2 } \quad \mathrm { s u c h \ t h a t } \quad \sum _ { j \neq i } \beta _ { j } ^ { 2 } \leq c .
$$

The solution to this minimisation problem (in terms of the Lagrangian) is given by

$$
\begin{array} { r } { \hat { \beta } = ( X ^ { \top } X + \alpha I _ { p } ) ^ { - 1 } X ^ { \top } Y , } \end{array}\tag{3}
$$

where $\alpha > 0$ is some constant. This is called the ridge estimator (Hoerl and Kennard, 1970). It depends on setting the value $\alpha$ . This solution is equivalent to jointly minimising the least squares function and the sum of squared parameters (Rao and Toutenberg, 1999). Hence we obtain the solution such that $| | \hat { \beta } | | _ { 2 } ^ { 2 } = c$ (Hastie et al., 2001). The solution $\operatorname* { l i m } _ { \alpha \to 0 } { \hat { \beta } }$ is called the minimum norm solution (Ben-Israel and Greville, 2003) or the ridgeless solution (Hastie et al., 2019; Dar et al., 2021). Note that if $n > p , \alpha = 0$ , and there is no multicollinearity, then the ridge estimator $\hat { \beta }$ in (3) reduces to the standard least squares estimator in (2) where X has full rank $p .$

The benefits of ridge regression over the lasso are that (a) the ridge regression has a closed form solution, (b) the ridge estimator is continuous and so allows a sampling distribution and the calculation of standard errors, and (c) the ridge estimator is able to handle multicollinearity (B¨uhlmann, 2013). Property (b) is especially appealing in practice because it allows for direct inference with the ridge estimate (B¨uhlmann, 2013; Waldorp and Haslbeck, 2024). The estimate has a bias $( X ^ { \top } X + \alpha I _ { p } ) ^ { - 1 } X ^ { \top } X - I$ , which shows that if $\alpha  0$ , and there is no collinearity, then the bias reduces to 0. The variance can be computed from which the standard errors are obtained for confidence intervals and p-values (Hoerl and Kennard, 1970; Gruber, 1990). In B¨uhlmann (2013) a strategy is proposed to reduce the bias and obtain more reliable confidence intervals and p-values; this is also discussed in Waldorp and Haslbeck (2024).

The parameter α in (3) can be obtained by k-fold cross-validation. The ridge estimator also has a Bayesian interpretation, where α denotes common variance for the normally distributed prior distribution with independent parameters (Gruber, 1990), but α can also be related to the η-generalised Bayesian approach, where $\alpha = \eta ^ { - 1 }$ (Gr¨unwald and van Ommen, 2017). The ridge estimator is biased (i.e., the expected value of the estimate is not the true estimator), like the Lasso, but the solution is in closed form and is continuous. Hence, inference on the ridge estimator directly is possible (but see B¨uhlmann, 2013, for improvements) and, as we will see shortly, the ridge estimator leads to shrinkage useful for model selection and can be linked to a geometric interpretation in high-dimensional models.

## 2.3 Realistic goals for high-dimensional nodewise graphical models

In the context of graphical models, the objective is to find the correct neighbourhood (nodes that are directly connected to a node) of all nodes in the graph, since then we obtain the correct set of non-zero edges for the entire graph (Meinshausen and B¨uhlmann, 2006). The ideal is, therefore, that with high probability, for any node, the set of estimated and selected non-zero edges $\hat { S }$ is equal to the set of true edges S, i.e., $\mathbb { P } ( \hat { S } = S ) = 1 - \epsilon .$ , with small $\epsilon > 0$ . However, in the high-dimensional setting this requires a strong and untestable assumption that the smallest value $\beta _ { j }$ is larger than a value in the order of $| S | \sqrt { \log ( p ) / n }$ (the beta-min condition, B¨uhlmann, 2013), where |S| denotes the cardinality of the true set of non-zero edges for the node in question. Foregoing this assumption may lead to the case where the set of true edges in the neighbourhood of a node is in the set of estimated edges with high probability, i.e., $\mathbb { P } ( S \subset { \hat { S } } ) = 1 - \epsilon$ (B¨uhlmann and van de Geer, 2011). Consequently, this may lead to an undesirably high false positive rate. If for each node we select only a few additional, spurious edges, then for the entire graph, the set of spurious edges will be proportional to the set of nodes. So, for large graphs, the set of spurious edges will be large. Therefore, a more realistic objective in graphical modelling is to obtain a set of neighbours such that for any node, the estimated set of non-zero edges $\hat { S }$ is in the true set of edges S, i.e., $\mathbb { P } ( \hat { S } \subseteq S ) = 1 - \epsilon$

In order to achieve the goal of low false positive rate in the high-dimensional setting, it is necessary that the large space of possible models is accounted for (Foygel and Drton, 2013; Dwivedi et al., 2020; Cheema and Sugiyama, 2020). Hence, a penalty for allowing such a large space of possible non-zero edges should be part of the neighbourhood selection. We show in Section 5.1 that because of the penalty in minimum description length (MDL), this selection method achieves the goal of obtaining the correct neighbourhood or smaller, but will not lead to a neighbourhood that is too large (no false positive edges). First, we show the impact of the high-dimensional setting on the mean squared error, which leads to the problems in neighbourhood selection.

## 3 Mean-squared error and the bias-variance trade-of

Determining the Gaussian graphical model was reduced to a series of regressions (one for each node) when performing nodewise selection (Meinshausen and B¨uhlmann, 2006). Hence, model selection can be applied to each regression separately. We therefore apply ideas of mean-squared error to a single regression to illustrate the central concepts involved.

Model selection is usually considered as a trade-of between model fit and generalisation to other datasets. Rather than fitting exactly all datapoints of a particular dataset (overfitting), the aim is to find regularity that will allow the model to describe other datasets as well and generalise (Myung and Pitt, 1998; Gr¨unwald et al., 2005; Claeskens and Hjort, 2008). How close the model follows the dataset is related to bias and how well the model generalises to other datasets is related to the variance; the mean-squared error is the sum of the bias squared and the variance (made precise in Section 4, Hastie et al., 2001). We first discuss the intuition of the bias-variance trade-of and how this may fail in the high-dimensional setting, and then in Section 4 we discuss rigorous results about the bias-variance trade-of in ridge regression.

The classical idea of model selection is illustrated in Figure 2. The situation is a linear regression model as described in Section 2 for the GGM. The regression coeficients are estimated by ridge regression for an increasing number of predictors that we put in the model with a subset of the data called the training set (we describe the details of the simulation in Section 6). In Figure 2(b) we see the bias (squared) of the test set (independent data not used for estimation) based on the coeficients of the training set as a function of $p / n$ ， the ratio of the number of parameters in the model and the number of observations (in the training set). Then the mean-squared error (MSE) is obtained in the remaining subset of observations, the test set; and is therefore called test MSE. The correct model is at $p / n \approx 0 . 1 5$ (the dashed vertical line). We see that the bias decreases sharply until the correct value is obtained and then increases again, and similarly for the variance. The squared bias and variance together make up the test MSE (also known as excess risk and prediction error). This is shown in Figure $2 ( \mathrm { a } )$ . Here we see that at the point $\approx 0 . 1 5$ (the correct model), where the test MSE is lowest, the corresponding model should be selected. This is the classic case discussed in for instance Hastie et al. (2001, Chapter 7) and Gr¨unwald et al. (2005, Chapter 1 and Chapter 2). The test MSE is used instead of the MSE based on the training (estimation) data (seen as the dashed green line in Figure $2 ( \mathrm { a } )$ , in sample) because the training MSE is overly optimistic and so remains low (see Efron, 1986; Hastie et al., 2001). This trade-of between bias and variance implies that the complexity of the model cannot be too high because then the model tends to overfit, i.e., fit only the training data and does not generalise well to other data.

![](images/a5f13817b7aa0267a3e4f85f56bb85d9d80f9ca4c81bc8b857b08ae13ff5b588.jpg)  
(a)

![](images/b1dac321a2d2e547cb8765b76fbff50c0738da0dac29b07ebce8ffed6858405e.jpg)  
(b)  
Fig. 2. In (a) the test and train mean squared error (MSE) for a true linear model as a function of $p / n$ , the ratio of the number of parameters and the number of observations. The true model is at $p / n$ ≈ 0.15. In (b) are the squared bias (squares) and variance (circles) for the test set.

In recent years it has appeared that this classical scenario is not all there is to it. There are situations where a model can over fit (select too many non-zero coeficients) and still generalise well. This phenomenon (sometimes called benign overfitting, Bartlett et al., 2020) is seen in Figure $\mathrm { 3 ( a ) }$ . The linear regression coeficients are estimated using the ridge estimator where the number of parameters (predictors) is increased such that the ratio $p / n$ (for the training set) ranges between 0.1 to 2. At $p / n = 0 . 1$ we have 10 observations per parameter and at $p / n = 2$ we have 0.5 observations per parameter. The classical case is seen up to $p / n = 1 \ ( \mathrm { i . e . , } \ p = n )$ , the interpolation point (as in Figure 2). From the interpolation point on, the model is overparameterised $( p > n )$ and has very low training MSE (green dashed line in Figure $\mathrm { 3 ( a ) } )$ . In

![](images/64536c5017a7d5219e66f77631b93d641c4cb2baf56aef5ffe61a688da00a571.jpg)  
(a)

![](images/98435984aa4a438c81165c3884b31f49df368cbdb00e0eaddea49c1cce35c5ba.jpg)  
(b)  
Fig. 3. In (a) the test and train mean squared error (MSE, in logarithm) for a true linear model as a function of $p / n .$ , the ratio of the number of parameters and the number of observations (in the training set). The true model with 5 predictors is at $p / n \approx 0 . 1 5$ , indicated by the vertical grey dashed line. In (b) is the bias (squares) and variance (circles), both in logarithm, for the test set and for the training set (dashed lines).

Figure 3(b), we see that, as expected, the bias decreases after the interpolation point. Unexpectedly, the variance starts to decrease as well, suggesting that the model will generalise well to other datasets. But this seems paradoxical since overparametrisation (and hence overfitting) would suggest that noise is being modeled. But the decreasing variance implies that the overparameterised model is capable of capturing regularities in other datasets, i.e., it will generalise well.

This phenomenon is often referred to as the double descent (Belkin et al., 2019), since there is a second descent after the interpolation point $( p = n )$ The double descent has been investigated thoroughly (e.g., Hastie et al., 2019; Bartlett et al., 2020, 2021) with some results as follows. We assume for these results that all random variables are normally distributed and have independent identically distributed predictors.

(a) In linear regression when the model is approximately correct and $p / n$ is large (overparameterised), the global minimum of the test MSE is obtained at the correct model.

(b) If the model is misspecified and $p / n$ is large (overparameterised), then the global minimum of the test MSE could be obtained at a largely overparameterised (incorrect) model.

The first result (a) implies that we are still able to correctly identify the correct neighbourhood for each node when approximately correctly specified. And the second result (b) implies that when the model is not correctly specified, e.g., nonlinear, it might be the case that in neighbourhood selection far too many edges will be selected because the MSE is lowest in that case. This last result explains why in deep learning (and in machine learning) astounding results have been obtained by hugely overparameterised models. Examples are object and speech recognition and trafic sign classification (see, e.g., Goodfellow et al., 2016).

The implications for neighbourhood selection are that obtaining the correct neighbourhood or smaller requires increasing the ridge parameter strongly, so as to counteract the small MSE that is obtained for large misspecified models. It is in general questionable whether this can be achieved by carefully selecting a ridge parameter (by e.g., cross-validation), or whether a separate penalty is required that counteracts the low MSE obtained in the case of highdimensions. We will see in Section 5.1 that we require a separate penalty in the high-dimensional case.

## 4 Mean-squared error decomposition in bias and variance

We remain within the framework of nodewise selection and focus on a neighbourhood of any node to obtain the non-zero edges, which leads to regressing one node on the remaining nodes.

We define the mean squared error (MSE) for linear models as a quantity that represents the expected loss incurred by estimating with model J instead of the true model S. Let $\hat { Y } _ { S }$ be the prediction based on the true set $S$ of predictors, so that $\beta _ { j } \neq 0$ if $j \in S$ and $\beta _ { j } = 0$ if $j \not \in \ S$ , which we refer to as $\beta _ { S }$ . Furthermore, we assume that the data are generated according to an additive model where errors e are added to $\hat { Y } _ { S }$ , so that

$$
Y = \hat { Y } _ { S } + e = \sum _ { j \in S } X _ { j } \beta _ { j } + e ,\tag{4}
$$

where the errors e are independent and identically distributed with mean 0 and variance $\sigma ^ { 2 }$

Test MSE (excess risk) is defined in terms of a so-called test set, an independent point (or set of points) not encountered before in the so-called training set (Hastie et al., 2001). Let $( X _ { 0 } , Y _ { 0 } ) \in \mathbb { R } ^ { p + 1 }$ be a test data point independent of, but from the same distribution as, all other $( X _ { i } , Y _ { i } ) \in \mathbb { R } ^ { p + 1 }$ With the training data the ridge estimate $\hat { \beta } _ { J }$ with model J is obtained. Then the excess risk (out of sample prediction risk, Hastie et al., 2019) with respect to the true parameter $\beta _ { S }$ , is defined as the squared diference between the predictions obtained with model J and the predictions obtained with the true model $S$ for test data $( X _ { 0 } , Y _ { 0 } )$

$$
R ( J ) = \mathbb { E } \left[ ( Y _ { 0 } - X _ { 0 } ^ { \top } \hat { \beta } _ { J } ) ^ { 2 } - ( Y _ { 0 } - X _ { 0 } ^ { \top } \beta _ { S } ) ^ { 2 } \right] ,\tag{5}
$$

where the expectation is conditional on the training data X, which functions in the ridge estimate $\hat { \beta } _ { J } = ( X ^ { \top } X + \alpha I _ { p } ) ^ { - 1 } X ^ { \top } Y$ from (3). Similar to Hastie

et al. (2019) and Bartlett et al. (2021) we can obtain the excess risk in terms of the squared bias and variance (see Lemma B.1 in Appendix B)

$$
R ( J ) = B ( J ) + V ( J ) ,\tag{6}
$$

with

$$
\begin{array} { r } { B ( J ) = ( { \beta _ { S } } - { \mathbb { E } } \hat { \beta } _ { J } ) ^ { \top } { \Sigma } ( \beta _ { S } - \mathbb { E } \hat { \beta } _ { J } ) , ~ V ( J ) = \mathbb { E } ( \hat { \beta } _ { J } - \mathbb { E } \hat { \beta } _ { J } ) ^ { \top } { \Sigma } ( \hat { \beta } _ { J } - \mathbb { E } \hat { \beta } _ { J } ) , } \end{array}\tag{7}
$$

where $B ( J )$ is called the squared bias and $V ( J )$ is called the variance, and $\Sigma = \mathbb { E } ( X _ { 0 } X _ { 0 } ^ { \top } )$ , which is the same as the covariance matrix of the predictors $\mathbb { E } ( X ^ { \top } X )$ . This is the famous variance and squared bias decomposition (Hastie et al., 2001). The classical idea is that reducing bias will increase variance and vice versa, thus balancing the MSE. Hence, a good model will minimise the MSE (excess risk) and most model selection methods are geared towards this goal.

The training bias $B ( J )$ is 0 for the linear model, when $p < n$ , and the linear model is correct (Hastie et al., 2019). However, for the ridge estimator $\hat { \beta }$ in (3) the bias is non-zero (Hoerl and Kennard, 1970). It turns out that when the in-sample (training) residuals are 0 (i.e., overparameterised, see Appendix I), then still the variance on the test set need not be large (Bartlett et al., 2021). Here we explain how this can be achieved with ridge regression.

To show the impact of ridge regression on the variance and the squared bias (and hence the MSE), we will rewrite the variance and squared bias from $( 7 )$ . To do this we further assume that $\Sigma = \sigma _ { \xi } ^ { 2 } I$ (i.e., is isotropic). We then obtain the test variance (see Lemma B.3)

$$
V ( J ) = \sigma _ { \xi } ^ { 2 } \sigma _ { e } ^ { 2 } \sum _ { j = 1 } ^ { n } \frac { \lambda _ { j } } { ( \lambda _ { j } + \alpha ) ^ { 2 } } ,
$$

where $\lambda _ { j }$ are the eigenvalues of $X ^ { \top } X$ . Hence, by increasing α we see that we are lowering the variance $V ( J )$ , and hence the test MSE. Additionally, we can observe that if the contribution of the eigenvalues $\lambda _ { d + 1 }$ up to $\lambda _ { p } ,$ , for some $d < p ,$ is small, then the variance will not increase, and hence, prediction may still be accurate. This implies that adding more predictors (making p larger) will not necessarily inflate the variance $V ( J )$ much, which is stabilised by the ridge parameter α. This argument is used in Bartlett et al. (2020, 2021) to explain why in some cases machine learning with a large number of parameters predicts well; the noise is distributed among the large number of directions (given by the eigenvectors) so that none of the eigenvalues $\lambda _ { k }$ with $k > d$ will be larger than the ridge parameter α.

Similar to the test variance $V ( J )$ the bias $B ( J )$ can also be reduced to a more amenable form when we assume that $\Sigma = \sigma _ { \xi } ^ { 2 } \dot { I }$ . With this assumption we

obtain (see Lemma B.2)

$$
B ( J ) = \sigma _ { \xi } ^ { 2 } \frac { \gamma _ { 1 } \alpha ^ { 2 } } { ( \lambda _ { 1 } + \alpha ) ^ { 2 } } + \sigma _ { \xi } ^ { 2 } \sum _ { j = 2 } ^ { n } \frac { \alpha ^ { 2 } } { ( \lambda _ { j } + \alpha ) ^ { 2 } } ,\tag{8}
$$

where $\gamma _ { 1 }$ is the single non-zero eigenvalue of $\beta _ { S } \beta _ { S } ^ { \top }$ . This shows diferent behaviour for the squared test bias than the test variance $V ( J )$ : the squared test bias could increase by increasing α.

This analysis shows that the peak in the test MSE for the ridge estimator in Figure $\mathrm { 3 ( a ) }$ is in some sense an artefact of choosing the wrong ridge parameter $\alpha = 0 . 0 0 0 1$ (used in Figure 3). The peak disappears when we use a carefully selected ridge parameter α using k-fold cross-validation. This is shown in Figure H.1 in Appendix H. It can also be seen in Figure H.1 that the lasso also does not show such a peak but the lasso has generally larger MSE than the ridge.

We can now conclude from this analysis that there are two views on how to manage the test variance in the case of high-dimensional data.

(a) We can increase the ridge parameter α directly, reducing the test variance $V ( J )$ , and hence obtaining reasonable test MSE.

(b) we can constrain the size of $| | \beta | | _ { 2 }$ in the ridge estimator, therefore, decreasing the model complexity (volume) of the model. We do this by constraining the ridge estimator with small c such that $| | \beta | | _ { 2 } \leq c$

In both cases $( a )$ and (b) we are either directly or indirectly constraining the total signal $| | \beta | | _ { 2 }$ . In model selection this can be done by either increasing α, equivalently, decreasing $c$ in the ridge constraint $| | \beta | | _ { 2 } \leq c ,$ or by incorporating the complexity (volume) of the model. (The ridge parameter $\alpha$ can also be interpreted in terms of the signal-to-noise (SNR) with similar conclusions relating to SNR, see Appendix D.)

## 4.1 Model misspecification

More generally, a model is defined as a class of probability distributions (in a favourable case in terms of corresponding densities), and the model is correctly specified if the distribution is in the particular class used for optimisation (Myung and Pitt, 1998). In our setting of nodewise regression with Gaussian variables, to determine the neighbourhood of a node, the model class is the set of univariate normal distributions parameterised by the mean $\mu$ and variance $\sigma ^ { 2 }$ , with the conditional mean $\mathbb { E } ( Y \mid x )$ given by a linear function $x \mapsto x ^ { \intercal } \beta .$ so that the model is correctly specified. Model misspecification can then be defined in terms of the model class not containing the correct distribution (Gr¨unwald, 2007; Dar et al., 2021; Kleijn, 2026). Continuing our example of normal distributions, the conditional mean could be given by some nonlinear function $f ,$ say a logistic function $x \mapsto 1 / ( 1 + \exp ( x ^ { \top } \beta ) )$ , while we only consider linear models $x ^ { \bar { \top } } \beta$ for estimation. One interpretation of the linear parameters when the true model is nonlinear, is in terms of a weighted average of derivatives of the conditional mean with respect to the covariate $x _ { j }$ , which reduces to mixtures of treatments in causal analysis (White, 1980; Kunievsky, 2025). The problem with misspecifying the conditional mean may lead to the situation where the residual variance is heterogeneous, and the ‘best’ model in terms of being closest to the true model in Kullback-Leibler divergence, is a mixture of models which is not in the model space (non-convex model space). For more information on this see Appendix B.1 and the seminal work of Gr¨unwald and van Ommen (2017).

For nodewise regression with misspecification, we assume that the model is some nonlinear function $f$ modelled by a linear function $X ^ { \top } \beta$ . If the misspec ification is characterised by an additive term, so that the model is $X ^ { \top } \beta + \xi$ where we do not know about ξ, then the misspecification can be described by the mismatch $\mathbb { E } | | \boldsymbol { \xi } | | ^ { 2 }$ (Hastie et al., 2019; Dar et al., 2021). In our approach we do not assume a linear decoupling, but simply that there is some nonlinear function $f .$ . In that case we obtain an additional term in the excess risk (see Lemma B.4 in Appendix B)

$$
R _ { f } ( J ) = B _ { f } ( J ) + V _ { f } ( J ) + M _ { f } ( J ) ,\tag{9}
$$

where

$$
\begin{array} { r l } & { B _ { f } ( J ) = \mathbb { E } \left[ ( f ( X _ { 0 } , \beta _ { S } ) - \mathbb { E } ( X _ { 0 } ^ { \top } \hat { \beta } _ { J } ) ) ^ { 2 } \right] , } \\ & { V _ { f } ( J ) = \mathbb { E } \left[ ( X _ { 0 } ^ { \top } \hat { \beta } _ { J } - \mathbb { E } ( X _ { 0 } ^ { \top } \hat { \beta } _ { J } ) ) ^ { 2 } \right] , \quad \mathrm { a n d } } \\ & { M _ { f } ( J ) = 2 \mathbb { E } \left[ ( Y _ { 0 } - f ( X _ { 0 } , \beta _ { S } ) ) ( f ( X _ { 0 } , \beta _ { S } ) - X _ { 0 } ^ { \top } \hat { \beta } _ { J } ) \right] . } \end{array}
$$

If we are able to bound the misspecification $| f ( X _ { 0 } , \beta _ { S } ) - X _ { 0 } ^ { \top } \hat { \beta } _ { J } | = O _ { p } ( K )$ we obtain a bound on the misspecification term by the Cauchy-Schwarz inequality: $M _ { f } ( J ) \leq \sigma O ( K )$ (see Proposition B.5). This bound on the misspecification will lead to the possibility of obtaining the correct neighbourhood or smaller with minimum description length in the high-dimensional setting. Additionally, in linear models we can leverage the result that for $L \subseteq J$ , we have that $\mathbb { E } [ ( Y _ { 0 } - X _ { 0 } ^ { \top } \hat { \beta } _ { J } ) ^ { 2 } ] \le \mathbb { E } [ ( Y _ { 0 } - X _ { 0 } ^ { \top } \hat { \beta } _ { L } ) ^ { 2 } ]$ (see also Appendix G). Hence, the misspecification decreases as the ratio $p / n$ increases.

## 5 Nodewise selection in high dimensions

A regression model is represented by a distribution for the residuals (Hastie et al., 2001) and, for the GGM using linear regressions, is identified with a set of nodes in the neighbourhood of a node, indexed by the set J. Increasing the number of included nodes in a nodewise regression is hence identified with diferent distributions, which can then also be indexed by J. Nodewise selection is therefore concerned with distinguishing probability distributions. The Kullback-Leibler divergence (KL, Cover and Thomas, 2006) is often used to distinguish distributions, where a KL of 0 indicates no diference between distributions, and a value larger than 0 indicates diferent distributions (see Appendix E). Both the Akaike information criterion (AIC) and minimum description length (MDL) can be considered as minimising the KL of a hypothesised distribution (our linear regression models) and some generative (unknown) model (Gr¨unwald, 2000; Myung et al., 2006; Claeskens and Hjort, 2008). The AIC estimates the KL and corrects the incurred bias of doing so (see Appendix E). The MDL, on the other hand, applies the idea of encoding the data and hypothesised model to obtain a reasonably approximating model, which we explain here.

In the high-dimensional case nodewise selection is problematic precisely because of the large parameter space. In high dimensions we obtain eventually near-zero training error, even with a linear model (Bartlett et al., 2020, see Appendix I). This is because the parameter space of the model is so vast (i.e., high complexity), that nearly all data points can be accounted for. Conceptually, the volume of the parameter space more or less explodes with increasing dimension and, hence, dominates the regression (see Appendix C and Appendix D). This may lead to selecting too many neighbours (over fit), which, in nodewise selection leads to a graph with many spurious edges. In order to obtain a correct neighbourhood or smaller in such high-dimensional situations, this increased model complexity needs to be taken into account explicitly, especially in the case of misspecification.

Here we discuss the MDL in some detail and show that in the highdimensional setting it leads to no false positive edges with high probability. The reason is that the penalty (model encoding) counteracts the large number of possible edges in a neighbourhood. For comparison, we also show results for the AIC, BIC, the Lasso, and another version of MDL for nodewise selection. We discuss these in Appendices E for the AIC and F an optimised version of MDL. There are excellent books that describe these derivations well. For instance, the AIC and BIC (and other model selection criteria) are described in Claeskens and Hjort (2008) and MDL is described extensively in the book by Gr¨unwald (2007).

## 5.1 Minimum description length

The minimum description length (MDL) is derived from the idea that compression (encoding) of data can be translated into probabilities (Gr¨unwald et al., 2005). MDL is concerned with the length of (prefix or invertible) optimal codes which can be separated into a part where the data are encoded given the model and a part where the model is encoded. The length of the data can be encoded by the log likelihood (Hansen and Yu, 2001), i.e. length $( Y , x ) = - \log f ( Y \mid x , { \hat { \beta } } )$ , where f is the conditional density of Y given the predictors X = x (see Appendix E). Additionally, the model is encoded, called the complexity, which leads to a specific formulation depending on the type of encoding (Gr¨unwald, 2007). This encoding term is the penalty for the possibly large parameter space and will to a certain extent guard against

overfitting.

For normally distributed random variables the MDL can be approximated by (Myung et al., 2006; Gr¨unwald, 2007)

$$
\mathrm { M D L } ( J ) = \underbrace { \frac { n } { 2 } \log \hat { \sigma } _ { J } ^ { 2 } } _ { \mathrm { d a t a \ c o d e \ l e n g t h } } + \underbrace { \frac { 1 } { 2 } \mathrm { t r } ( S _ { J } ) \log n + \log \int _ { B } \sqrt { \operatorname * { d e t } ( F _ { J } ) } d \beta } _ { \mathrm { m o d e l \ c o d e \ l e n g t h } } + o ( 1 ) ,\tag{10}
$$

where tr $\cdot ( S _ { J } )$ is the trace of matrix $S _ { J }$ representing the efective number of parameters for model $^ { J , }$ and $F _ { J }$ is the Fisher information matrix for model $J ,$ which is $F _ { J } = X ^ { \top } X / \sigma ^ { 2 }$ when $n > p ,$ and the true model is exponential family (Gr¨unwald et al., 2005, Chapter 2). The matrix $S _ { J }$ is used to determine the number of efective parameters, and is $S _ { J } = X ( X ^ { \top } X ) ^ { - 1 } X ^ { \top }$ when $n > p ;$ the trace of this matrix is $p$ (this is the same as the rank in this case).

However, for the high-dimensional case $( p > n )$ we find that $X ^ { \top } X$ is singular, implying that det $( X ^ { \top } X ) = 0$ . This is because the rank of $X ^ { \top } X$ is min $\{ n , p \}$ , and because $n < p$ , we have that the rank in this case is $n ,$ while the dimensions are $p \times p$ . Thus, we must have that there are $p - n$ dimensions that are not represented by $X ^ { \top } X ~ ( { \mathrm { i . e . , } } ~ p - n$ vectors project onto the kernel or null space). We can use ridge regression to alleviate the problem by using instead of $X ^ { \top } X$ the matrix from ridge regression $W ( \alpha ) = X ^ { \top } X + \alpha I$ (Hastie et al., 2001). We then find that $S _ { J } = X ( X ^ { \top } X + \dot { \alpha } \dot { I } ) ^ { - 1 } X ^ { \top }$ . This gives the general equation for the number of efective parameters (Hastie et al., 2001, Section 7.6)

$$
d f ( \alpha ) = \operatorname { t r } ( S _ { J } ) = \operatorname { t r } X ( X ^ { \top } X + \alpha I ) ^ { - 1 } X ^ { \top } = \sum _ { j = 1 } ^ { \operatorname* { m i n } \{ p , n \} } { \frac { \lambda _ { j } } { \lambda _ { j } + \alpha } } ,
$$

where $\lambda _ { j }$ are the eigenvalues of $X ^ { \top } X$

To approximate the integral in the penalty encoding the model in (10), we make use of the fact that in linear regression the Fisher information is independent of the parameter $\beta _ { j }$ , and hence the integral represents the open p-dimensional ball $\mathbb { B } ^ { p } ( | | \hat { \beta } | | _ { 2 } )$ over the Fisher information where we used the ridge parameter α (see Lemma D.1 in Appendix D). The p-dimensional ball is centered at $| | \hat { \beta } | | _ { 2 }$ because in the ridge solution we obtain the solution such that $\alpha ( | | \beta | | _ { 2 } ^ { 2 } - c ) = 0$ . We then obtain (Cheema and Sugiyama, 2020; Gr¨unwald, 2007)

$$
\log \int _ { B } { \sqrt { \operatorname* { d e t } ( { \boldsymbol { F } } ) } } d \beta \approx \log \operatorname* { d e t } ( X ^ { \top } X + \alpha I ) + \log \mathbb { B } ^ { p } ( | | \beta | | _ { 2 } ) / \sigma ^ { p } .\tag{11}
$$

The p-dimensional ball in the last term, log $\mathbb { B } ^ { p } ( | | \beta | | _ { 2 } )$ , represents the volume of the parameter space and is discussed in Appendix C. The log of this term will become large, and hence apply a stronger penalty with increasing dimension. This is why increasing the dimension can have positive efects on the generalisation error (variance in the test set). However, in determining the volume in this way, we have assumed that the linear model is correct. Hence, we may expect that when the model is misspecified, the term in the integral encoding the model in (11) may not be suficient to guard against incorrect decisions. Gr¨unwald and van Ommen (2017) show that when the model is misspecified the model space can be non-convex, and the best approximation to the true model may not be in the model space (see Appendix B.1). However, using linear models we can with high probability conclude that MDL as defined in (10) with (11) will be able to guard against overfitting. This is because of the penalty term $\mathbb { B } ^ { p } ( | | \beta | | _ { 2 } )$ , as predicted by Cheema and Sugiyama (2020). We are therefore able to achieve the goal mentioned in Section 2.3, that the selected neighbourhood of edges in a graph $\hat { S }$ with MDL is in the true neighbourhood S, with $d = | S |$

Proposition 5.1 Let S denote the true support for a (possibly nonlinear) data generating process of Y<sub>i</sub> for $i = 1 , \ldots , n$ with n fixed, and let J represent the set of predictors such that $S \subset J$ , where $| S | = d$ and $| J | = p$ . We test linear models $X \beta _ { J }$ for any J (including S) by MDL as given in $( 1 0 )$ with $( 1 1 )$ Let p increase and $p > d$ . If $\mathbb { E } \hat { \sigma } _ { J } ^ { 2 } > \gamma$ , for some $\gamma > 0$ , then with probability at least $\begin{array} { r } { 1 - { \frac { d } { p } } \exp ( - ( p - d ) ) } \end{array}$ it holds that ${ \hat { S } } \subseteq S ,$ ; equivalently, the probability of false positives goes to 0.

A proof can be found in Appendix G.

We refer to MDL in (10) with (11) as MDL. The complexity of MDL has also been used without the integral term (Rissanen, 1986; Gr¨unwald, 2000). This approximation is in that case equivalent to the Bayesian information criterion (BIC) introduced by Schwartz (1978). Hence, we call that version MDL-S.

## 6 Simulations

In order to determine the behaviour of nodewise selection in realistic highdimensional scenarios in graphical models, we perform simulations for a single neighbourhood. This corresponds to the graphs shown in Figure 1, with the perspective of a single node for which we need to determine the connected nodes (neighbourhood) among the nodes that are not connected, of which there are many more.

The data are generated by a Gaussian graphical model. An inverse covariance matrix was generated where each node had 5 neighbours with edge coeficient 1. We then selected a node at random. Data were generated where the total number of observations was fixed at $n = 4 0$ . We vary the graph size so that the number of possible edges to be estimated ranges from $p = 3$ to $p = 3 2 0$ (see Figure 1). So, we obtain a ratio of the number of edges and the number of training observations of $p / n = 0 . 1$ to 2 with $p = 6 4$ and $p / n = 1 0$ for $p = 3 2 0$ . Edges (conditional dependences) are generated either according to a linear model (correct model specification) or a sigmoid function (model misspecification). The regression coeficients for nodewise selection are estimated as a linear model by ridge regression (3) with a training set of $n _ { \mathrm { t r a i n } } = 3 2$ (80% of the total sample size $n = 4 0 )$ . The test set is of size $n _ { \mathrm { t e s t } } = 8$ . We consider the case where we know the true model is linear and we estimate the linear model (case (a) in Section 3), and we misspecify the model because the true model is sigmoidal and we use a linear approximation (case (b) in Section 3). We compare the AIC, AIC-CV (where $\alpha$ is obtained with cross-validation), MDL-S (which is the BIC), MDL-opt (the optimised MDL), MDL, the Lasso (also with α obtained with cross-validation), and finally, the desparsified lasso (Lasso-ds, van de Geer et al., 2014; Dezeure et al., 2015).

The measures we use to evaluate the nodewise selection is the proportion of edges in a neighbourhood that is too low (under fit), correct or too high (over fit) in terms of the estimated edges in a single neighbourhood. Then, under, correct and over fit, respectively, for R simulations are defined as

$$
\operatorname { p r o p o r t i o n } { \mathrm { ~ u n d e r ~ f i t : = } } \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \mathbb { 1 } \{ \hat { S } \subset S \} ,
$$

$$
{ \mathrm { p r o p o r t i o n ~ c o r r e c t ~ f i t : } } = { \frac { 1 } { R } } \sum _ { r = 1 } ^ { R } \mathbb { 1 } \{ { \hat { S } } = S \} , \quad { \mathrm { a n d } }
$$

$$
{ \mathrm { p r o p o r t i o n ~ o v e r ~ f i t } } : = { \frac { 1 } { R } } \sum _ { r = 1 } ^ { R } \mathbb { 1 } \{ \hat { S } \supset S \} ,
$$

where $\hat { S }$ is the set of indices for the estimated non-zero edges (support) and S is the true support. Given our objective of selecting a neighbourhood with high probability which includes correct edges but no spurious connections (see Section 2.3), we want the under and correct fit to be as large as possible and the over fit as small as possible. To look into more detail of over fitting, we also consider the false positive rate, defined as

$$
{ \mathrm { f a s l e ~ p o s i t i v e ~ r a t e : } } = { \frac { 1 } { R } } \sum _ { r = 1 } ^ { R } { \frac { | \hat { S } \cap S ^ { c } | } { | S ^ { c } | } } ,
$$

where $S ^ { c }$ is the complement of the support, the true negatives or zero edges. This measure gives an indication of the rate of selecting too many edges in a neighbourhood in the graph (over fitting).

## 6.1 Results

We start with the impact of possible neighbourhood size $( p / n = 0 . 1$ up to 10) on MSE. Figure 4(a) shows the MSE for a correctly specified linear model, and for a misspecified linear model in Figure 4(b). The test MSE in (a) remains above the optimal MSE at the correct neighbourhood size with $p / n \approx 0 . 1 5$ This implies that standard model selection with MDL-S or the AIC should work in the high-dimensional setting. However, in Figure $4 ( \mathrm { b } )$ , when the model is misspecified, the test MSE drops below the MSE at the optimal solution, indicating that standard neighbourhood selection need not work. Since the MSE is lower at models with a large number of parameters, neighbourhood selection could favour models with a large number of parameters.

![](images/e5a5064c7c0e6a800289fb55a64c7392a88d32eab1c88ba61ef694e54d4dc6a7.jpg)  
(a)

![](images/64ae0a0d0796dd68cdf9fc8aa51157194118690c803bb9923eb71c545e2e13b1.jpg)  
(b)  
Fig. 4. The MSE for correctly specified (a) and misspecified model (b) for a neighbourhood with p edges up to 320, so that $p / n$ ranges between 0.1 and 10. For the true linear model (a) the global minimum of the test MSE remains the correct model with $p = 5$ , approximately at $p / n$ ≈ 0.15. For the misspecified model (b), however, we see that the global minimum has now shifted from the correct neighbourhood size with $p = 5$ and MSE 0.0877 to the model with $p / n = 1 0$ (= 320/32) and MSE 0.0844.

The consequences for neighbourhood selection are shown in Figure 5. The figure shows under fit $( { \hat { S } } \subset S )$ , light blue), correct fit $( { \hat { S } } = S$ , blue) and over fit $( \breve { \hat { S } } \supset S , \mathrm { r e d } )$ in each subfigure. The left column of Figure 5 shows the results for when the data is correctly specified as linear, while the right column shows the results for when the model is misspecified; both are estimated with a linear model. In the bottom row, neighbourhood selection is performed when $p _ { \operatorname* { m a x } } =$ 15, and increases to $p _ { \operatorname* { m a x } } = 3 2$ in the middle and 320 in the top row. When the model is correctly specified (left column) and the dimension is low $( p _ { \mathrm { m a x } } = 1 5 )$ neighbourhood selection in graphical models works reasonably well in that either the correct support is obtained, $\hat { S } = S$ or it is underestimated, i.e., $\hat { S } \subset S$ . Only the AIC and the Lasso show increased false positive rates, i.e., $\hat { S } \supset S$ . Increasing the dimension to $p _ { \mathrm { m a x } } = 3 2 0$ exacerbates the over fit for AIC and Lasso (also Lasso-ds). In contrast, MDL, MDL-opt, MDL-S and AIC-CV appear to remain accurate in the sense that over fit remains low, and over fit is absent in the case of MDL. For MDL this is because of the large penalty on using a large dimensional model space described in Section 5.1, in line with Proposition 5.1.

The pattern of over fit remains relatively similar when the model is misspecified (right column of Figure 5), but the probability of selecting the correct neighbourhood is decreased for all methods. Only MDL does not select too many edges (no false positives), which is in line with theory as discussed in Section 5.1 and Appendix G.

![](images/9adf53cbbc9b03c39a9675a6dbc9a43ae9128b7b5817e2506780367c92d8d708.jpg)

![](images/d8564e5d160972d49ba4bdb8a3f240512163e6aad0f530684f8beed794c7afbe.jpg)

![](images/c4c13d0fca18a20248f2dc11d170c7eafdfb3f77a910e4488153e2bae62d2159.jpg)

![](images/1c68d531dcf4919ac1b6ae642594ff7c97d87fb3d4b63a11def442694cedba4f.jpg)

![](images/d3ea2ebe36fa7611a28c2e5a02bb0b4b99af1f0f4ef136d44d5bf9af64bfb331.jpg)

![](images/f3ac8eddbec2e4cf90e407c16aea1f63147e7763a24b215784c1ba24d0b78d31.jpg)  
Fig. 5. Proportion of decisions: underestimating (light blue), correct (blue), and overestimating (red) the dimension of the model for both the correct (linear) and incorrect (logistic) data generating model for diferent maximum number of predictors $ { p _ { \mathrm { m a x } } } = 1 5 , 3 2$ or 320 in each of the respective rows, such that $p / n = 0 . 4 7$ , 1 and 10, respectively. In both the linear and non-linear model, there are $p = 5$ non-zero coeficients. In the left column the data were generated and estimated with the linear model; in the right column, data were generated with the non-linear model while they were estimated with the linear model.

![](images/ad5765e8c648906df035dcef77cb6cfa94b0398520288f4d1d0da9ada42c4464.jpg)  
(a)

![](images/456efe141e41749621918eff71457f22954ba6722462c9f0faa78a022783f697.jpg)  
(b)  
Fig. 6. False positive rate for each method. In (a) the model is correctly specified as linear and in in (b) the model is misspecified as linear while the conditional mean is a logistic function.

Finally, we consider the false positive rate, the probability of including too many edges in the graph. Figure 6(a) shows the false positive rate for each dimension in the simulation $p / n = 0 . .$ 1 up to 10, when the model is correctly specified as linear. MDL stands out in that no matter the model dimension, the false positive rate remains at 0. This corresponds to the results in the left column of Figure 5. Most other methods have a slightly elevated false positive rate; the AIC tends to lead to a neighbourhood that is too large from around the interpolation point $( p / n = 1 )$ on. When the assumption of a true linear model no longer holds, the false positive rate increases for almost all methods, except Lasso and MDL. Both MDL and Lasso have low false positive rate for any ratio of $p / n$

## 7 Conclusion and discussion

We considered nodewise selection of edges in the Gaussian graphical model when the number of nodes exceeds the number of observations. This is an issue because regular estimation techniques like least squares cannot work in such situations. We used the ridge regression estimate to solve this issue, leading to a biased but closed form estimate. We were then able to show the impact of the ridge parameter on the mean squared error, and concluded that indeed, when the true relation between edges is linear, the MSE is lowest at the correct neighbourhood. We also showed that in the misspecified case, the MSE is codetermined by an additional term in the MSE caused by the misspefication of a nonlinear model by a linear model. In addition, the ridge parameter could be interpreted as the inverse of the signal-to-noise-ratio. In low signalto-noise-ratio situations the ridge parameter will be high, leading to a larger ridge penalty. This penalty was then seen to reduce the parameter variance and, hence, implies that the generalisation can still be reasonable, suggesting generalisation to other samples is possible. This also (partly) explains why some highly overparameterised machine learning techniques are able to predict well, mimicking regularised estimation by restricting the class of functions in empirical risk minimisation (Hastie et al., 2022). Generally, selecting the ridge parameter using cross-validation seems good practice, since it reduced the sensitivity of the MSE and neighbourhood selection improved.

We next applied nodewise selection to ascertain whether it is possible to obtain the correct neighbourhood or smaller, but not larger (i.e., no false positives). We proved that in the high-dimensional setting MDL can achieve this objective of correct or smaller neighbourhood selection with high probability, in both the case when true edge relations are linear or nonlinear. The reason is that it explicitly takes into account the volume of the parameter space to counteract the low MSE at high dimensions. Simulations confirmed these results, and also showed that other neighbourhood selection methods (using AIC, for example) lead to neighbourhoods that are too large. Obviously, a low false positive rate and under fit for MDL, lead to a somewhat elevated false negative rate (too few correct edges). Considering the discussion in Section 2.3, we believe that, even though some edges will not be detected, this is preferable to the case when too many edges are selected. Therefore, we recommend using MDL for nodewise selection in the high-dimensional setting for Gaussian graphical modelling.

While the result that MDL leads to low false positives is geared towards the application of the high-dimensional setting in graphical modelling, it is more general. We used the framework of nodewise regression, and so it is valid within the context of regression. But we used the fact that there is a bound on the misspecification for the conditional mean and that the residual variance is bounded away from 0. Given these settings, the result on MDL can be applied to other regression settings.

## Acknowledgements

The research conducted by L. Waldorp has been supported by a grant from the ERC (ERC project 101053880) and by the “New Science of Mental Disorders”, the Dutch Research Council and the Dutch Ministry of Education, Culture and Science (NWO), grant number 024.004.016.

## References

Bartlett, P. L., Long, P. M., Lugosi, G., and Tsigler, A. (2020). Benign overfitting in linear regression. Proceedings of the National Academy of Sciences, 117(48):30063–30070.

Bartlett, P. L., Montanari, A., and Rakhlin, A. (2021). Deep learning: a statistical viewpoint. arXiv preprint arXiv:2103.09177.

Belkin, M., Hsu, D., Ma, S., and Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias–variance trade-of. Proceedings of the National Academy of Sciences, 116(32):15849–15854.

Ben-Israel, A. and Greville, T. (1974). Generalized inverses: theory and applications. New York: John Wiley and Sons.

Ben-Israel, A. and Greville, T. (2003). Generalized inverses: Theory and applications. New York: Springer.

Bilodeau, M. and Brenner, D. (1999). Theory of multivariate statistics. New York: Springer-Verlag.

B¨uhlmann, P. (2013). Statistical significance in high-dimensional linear models. Bernoulli, 19(4):1212–1242.

B¨uhlmann, P., Kalisch, M., and Meier, L. (2014). High-dimensional statistics with a view toward applications in biology. Annual Review of Statistics and Its Application, 1:255–278.

B¨uhlmann, P. and van de Geer, S. (2011). Statistics for High-Dimensional Data: Methods, Theory and Applications. Springer.

Cheema, P. and Sugiyama, M. (2020). Double descent risk and volume saturation efects: A geometric perspective. arXiv preprint arXiv:2006.04366.

Claeskens, G. and Hjort, N. (2008). Model selection and model averaging. Cambridge University Press, Cambridge.

Cover, T. and Thomas, J. (2006). Elements of information theory. Wiley and Sons, 2nd edition.

Dar, Y., Muthukumar, V., and Baraniuk, R. G. (2021). A farewell to the biasvariance tradeof? an overview of the theory of overparameterized machine learning. arXiv preprint arXiv:2109.02355.

DasGupta, A. (2008). Asymptotic Theory of Statistics and Probability. Springer-Verlag, New York.

Dezeure, R., B¨uhlmann, P., Meier, L., and Meinshausen, N. (2015). Highdimensional inference: confidence intervals, p-values and r-software hdi. Statistical science, pages 533–558.

Dwivedi, R., Singh, C., Yu, B., and Wainwright, M. J. (2020). Revisiting complexity and the bias-variance tradeof. arXiv preprint arXiv:2006.10189.

Efron, B. (1986). How biased is the apparent error rate of a prediction rule? Journal of the American statistical Association, 81(394):461–470.

Epskamp, S., Borsboom, D., and Fried, E. I. (2018). Estimating psychological networks and their accuracy: A tutorial paper. Behavior Research Methods, 50(1):195–212.

Epskamp, S. and Fried, E. I. (2018). A tutorial on regularized partial correla-

tion networks. Psychological methods, 23(4):617.

Foygel, R. and Drton, M. (2013). Bayesian model choice and information criteria in sparse generalized linear models. Technical report, University of Chicago.

Friedman, J., Hastie, T., and Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. Biostatistics, 9(3):432–441.

Giraud, C. (2014). Introduction to high-dimensional statistics, volume 138. CRC Press.

Goodfellow, I., Bengio, Y., Courville, A., and Bengio, Y. (2016). Deep learning, volume 1. MIT press Cambridge.

Gruber, M. (1990). Regression Estimators: A Comparative Study. Academic Press, Boston.

Gr¨unwald, P. (2000). Model selection based on minimum description length. Journal of Mathematical Psychology, 44:133–152.

Gr¨unwald, P. and van Ommen, T. (2017). Inconsistency of bayesian inference for misspecified linear models, and a proposal for repairing it. Bayesian Analysis, 12(4):1069–1103.

Gr¨unwald, P. D. (2007). The minimum description length principle. MIT press.

Gr¨unwald, P. D., Myung, I. J., and Pitt, M. A. (2005). Advances in minimum description length: Theory and applications. MIT press.

Hansen, M. H. and Yu, B. (2001). Model selection and the principle of minimum description length. Journal of the American Statistical Association, 96(454):746–774.

Hastie, T., Montanari, A., Rosset, S., and Tibshirani, R. J. (2019). Surprises in high-dimensional ridgeless least squares interpolation. arXiv preprint arXiv:1903.08560.

Hastie, T., Montanari, A., Rosset, S., and Tibshirani, R. J. (2022). Surprises in high-dimensional ridgeless least squares interpolation. Annals of statistics, 50(2):949.

Hastie, T., Tibshirani, R., and Friedman, J. (2001). The Elements of Statistical Learning. Springer-Verlag, New York.

Hastie, T., Tibshirani, R., and Wainwright, M. (2015). Statistical learning with sparsity: the lasso and generalizations. CRC Press.

Hoerl, A. and Kennard, R. (1970). Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1):55–67.

Jacod, J. and Protter, P. (2004). Probability essentials. Springer-Verlag, New York.

Jankova, J. and van de Geer, S. (2017). Inference for high-dimensional graphical models. In Drton, M., Maathuis, M., Lauritzen, S., and Wainwright, M., editors, Handbook of graphical models. Taylor & Francis Group.

Jones, B. and West, M. (2005). Covariance decomposition in undirected gaussian graphical models. Biometrika, 92(4):79–786.

Kleijn, B. (2026). The frequentist theory of bayesian statistics.

Koller, D. and Friedman, N. (2009). Probabilistic graphical models: Principles

and techniques. MIT Press.

Kullback, S. and Leibler, R. (1951). On information and suficiency. Annals of mathematical statistics, 22:79–86.

Kunievsky, N. (2025). Linear regression in a nonlinear world. arXiv preprint arXiv:2512.13645.

Lauritzen, S. (1996). Graphical Models. Oxford University Press, Oxford.

Maathuis, M., Drton, M., Lauritzen, S., and Wainwright, M. (2018). Handbook of graphical models. CRC Press.

Meinshausen, N. and B¨uhlmann, P. (2006). High-dimensional graphs and variable selection with the lasso. The Annals of Statistics, 34(3):1436–1462.

Meinshausen, N. and B¨uhlmann, P. (2010). Stability selection. Journal of the Royal Statistical Society, Series B, 72(4):1436–1462.

Meinshausen, N., Meier, L., and B¨uhlmann, P. (2009). P-values for highdimensional regression. Journal of the American Statistical Association, 104(488).

Myung, I. and Pitt, M. (1998). Issues in selecting mathematical models of cognition. In Grainger, J. and Jacobs, A., editors, Localist connectionist approaches to human cognition, pages 327–355. Lawrence Erlbaum Associates.

Myung, J. I., Navarro, D. J., and Pitt, M. A. (2006). Model selection by normalized maximum likelihood. Journal of Mathematical Psychology, 50(2):167–179.

P¨otscher, B. M. and Leeb, H. (2009). On the distribution of penalized maximum likelihood estimators: The lasso, scad, and thresholding. Journal of Multivariate Analysis, 100(9):2065–2082.

Pringle, R. and Rayner, A. (1971). Generalized inverse matrices with applications to statistics. Charles Grifin and Company, London.

Rao, C. (1990). Linear statistical inference and its applications. John Wiley and Sons, second edition edition.

Rao, C. and Toutenberg, H. (1999). Linear models: least squares and alternatives. Springer.

Rissanen, J. (1986). Stochastic complexity and modeling. The Annals of Statistics, 14:1080–1100.

Schott, J. R. (1997). Matrix analysis for statistics. New York: John Wiley & Sons.

Schwartz, G. (1978). Estimating the dimension of a model. The Annals of Statistics, 6:461–464.

Searle, S. (1971). Linear Models. John Wiley and Sons, New York.

Seber, G. A. and Lee, A. J. (2012). Linear regression analysis, volume 936. John Wiley & Sons.

Shah, R. D. and B¨uhlmann, P. (2023). Double-estimation-friendly inference for high-dimensional misspecified models. Statistical Science, 38(1):68–91.

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society. Series B (Methodological), 58(1):267–288.

Vaart, A. v. d. (1998). Asymptotic Statistics. New York: Cambridge University Press.

van de Geer, S., B¨uhlmann, P., Ritov, Y., and Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. The Annals of Statistics, 42(3):1166–1202.

Vershynin, R. (2018). High-dimensional probability: An introduction with applications in data science, volume 47. Cambridge University Press.

Vuong, Q. H. (1989). Likelihood ratio tests for model selection and non-nested hypotheses. Econometrica: Journal of the Econometric Society, pages 307– 333.

Wainwright, M. J. (2009). Sharp thresholds for high-dimensional and noisy sparsity recovery using-constrained quadratic programming (lasso). IEEE Transactions on Information Theory, 55(5):2183–2202.

Waldorp, L. and Haslbeck, J. (2024). Network inference with the lasso. Multivariate Behavioral Research, 12(in press):1–20.

Waldorp, L. and Marsman, M. (2021). Relations between networks, regression, partial correlation, and the latent variable model. Multivariate Behavioral Research, pages 1–13.

White, H. (1980). Using least squares to approximate unknown regression functions. International Economic Review, 21(1):149–170.

## Appendix

## A Gaussian graphical model

The Gaussian graphical model (GGM) is characterised by the correspondence between a network of $p$ nodes, labeled $j = 1 , 2 , \dotsc , p$ and a set of random variables $X _ { 1 } , X _ { 2 } , \ldots , X _ { p }$ that are jointly Gaussian (multivariate normally) distributed (Koller and Friedman, 2009). The edges in a GGM correspond to conditional dependence between nodes (given all reamining variables). The Gaussian distribution is completely described by its means $\mu _ { j }$ and covariances $\sigma _ { i j }$ . The inverse of the covariance matrix $\Sigma = ( \sigma _ { i j } , i , j = 1 , 2 , . . . , p )$ , denoted by $\Sigma ^ { - 1 } = \Theta$ , contain the partial covariances. The partial covariance $\theta _ { i j }$ is defined as the covariance between the residuals of nodes $i$ and $j$ when all other variables (not i and $j )$ have been regressed out (Lauritzen, 1996). In a GGM a partial covariance of 0 is equal to conditional independence. We show this by example.

Consider three Gaussian variables with mean 0 and covariance matrix $\Sigma$ We write the density of the three variables as $f _ { 1 2 3 }$ and the conditional density as $f _ { 1 3 | 2 }$ for the joint density of variables $X _ { 1 }$ and $X _ { 3 }$ given $X _ { 2 }$ . We will want to come to the conclusion that the product of the conditional distributions $f _ { 1 | 2 } f _ { 3 | 2 }$ is the same as the conditional distribution $f _ { 1 3 | 2 }$ . To do this we consider an example with covariance and inverse covariance matrix, respectively,

$$
\Sigma = \left( { \begin{array} { c c c } { 3 / 4 } & { - 1 / 2 } & { 1 / 4 } \\ { - 1 / 2 } & { 1 } & { - 1 / 2 } \\ { 1 / 4 } & { - 1 / 2 } & { 3 / 4 } \end{array} } \right) \qquad { \mathrm { a n d } } \qquad \Theta = \left( { \begin{array} { c c c } { 2 } & { 1 } & { 0 } \\ { 1 } & { 2 } & { 1 } \\ { 0 } & { 1 } & { 2 } \end{array} } \right) .
$$

Note the 0 at $\theta _ { 1 3 } = \theta _ { 3 1 }$ such that variables $X _ { 1 }$ and $X _ { 3 }$ are conditionally independent given $X _ { 2 }$ . In terms of the distribution this implies the following. So this implies that we have the density $f _ { 1 3 | 2 }$ . We now compute this from the density of thew corresponding conditional Gaussian distribution. The distribution with $\Theta$ is

$$
f _ { 1 2 3 } ( x ) = \frac { 1 } { \sqrt { ( 2 \pi ) ^ { 3 } } } \sqrt { \operatorname* { d e t } ( \Theta ) } \exp \left( - { \scriptstyle { \frac { 1 } { 2 } } } x ^ { \top } \Theta x \right) ,
$$

where the term in the exponential is

$$
x ^ { \top } \Theta x = 2 x _ { 1 } ^ { 2 } + 2 x _ { 2 } ^ { 2 } + 2 x _ { 3 } ^ { 2 } + 2 x _ { 1 } x _ { 2 } + 2 x _ { 2 } x _ { 3 } .
$$

Note that because $\theta _ { 1 3 } = 0$ there is no term $2 x _ { 1 } x _ { 3 }$ and that is why we can rewrite the density. If we take the density of $X _ { 1 }$ and $X _ { 2 }$ , using the first two rows and columns of $\Theta$ and calling that $\Theta _ { 1 2 }$ , then we obtain

$$
f _ { 1 2 } ( x _ { 1 } , x _ { 2 } ) = \frac { 1 } { 2 \pi } \operatorname * { d e t } ( \Theta _ { 1 2 } ) \exp \left( - \textstyle { \frac { 1 } { 2 } } ( 2 x _ { 1 } ^ { 2 } + 2 x _ { 1 } x _ { 2 } + 2 x _ { 2 } ^ { 2 } ) \right) .
$$

And similarly for the density of $X _ { 2 }$ and $X _ { 3 }$ with $\Theta _ { 2 3 }$ the second and third rows and columns of Θ, gives

$$
f _ { 2 3 } ( x _ { 2 } , x _ { 3 } ) = \frac { 1 } { 2 \pi } \operatorname * { d e t } ( \Theta _ { 2 3 } ) \exp \left( - { \textstyle \frac { 1 } { 2 } } ( 2 x _ { 2 } ^ { 2 } + 2 x _ { 2 } x _ { 3 } + 2 x _ { 3 } ^ { 2 } ) \right) .
$$

The density of only $X _ { 2 }$ , ignoring the other two variables is

$$
f _ { 2 } ( x _ { 2 } ) = \frac { 1 } { \sqrt { 2 \pi } } \Theta _ { 2 2 } \exp \left( - { \textstyle \frac { 1 } { 2 } } 2 x _ { 2 } ^ { 2 } \right) .
$$

Then we find (after some algebra) that

$$
\frac { f _ { 1 2 } f _ { 2 3 } } { f _ { 2 } } = f _ { 1 2 3 } \quad \Longleftrightarrow \quad \frac { f _ { 1 2 } f _ { 2 3 } } { f _ { 2 } f _ { 2 } } = f _ { 1 | 2 } f _ { 3 | 2 } = f _ { 1 3 | 2 } .
$$

And we see that if a conditional covariance $\theta _ { i j } = 0$ , then there is a conditional independence. This is true only for the multivariate normal distribution.

Hence, the GGM is particularly attractive for graphical models because whenever the partial correlation between variables $X _ { i }$ and $X _ { j }$ is $0 ,$ , then $X _ { i }$ and $X _ { j }$ are conditionally independent given all other variables (Lauritzen, 1996, Proposition 5.2). Hence, an edge in the GGM is present if and only if the partial correlation is non-zero. The partial correlation is a function of the inverse covariance matrix and has elements $\theta _ { i j }$ . The partial correlation between $X _ { i }$ and $X _ { j }$ is defined by (Koller and Friedman, 2009; Epskamp and Fried, 2018)

$$
\rho _ { i j | \star } = - \frac { \theta _ { i j } } { \sqrt { \theta _ { i i } \theta _ { j j } } } ,
$$

where ⋆ indicates all remaining variables in the network except i and $j .$ . If $\rho _ { i j | \star } = 0$ , then no edge should be present between nodes i and $j$ in the network. A GGM can be interpreted as a network where a correlation corresponds to a (set of) path(s) between two nodes (Jones and West, 2005).

We can determine whether a partial correlation also by considering the regression coeficients $\beta _ { i j }$ from the regression

$$
X _ { i } = \sum _ { j \neq i } X _ { j } \beta _ { i j } + e _ { i } , \quad
$$

where the sum is over all nodes $j$ that are not i. The reason for the correspondence between the partial correlation and the regression coeficient is that (Lauritzen, 1996)

$$
\beta _ { i j } = - \frac { \theta _ { i j } } { \theta _ { i i } } .
$$

Hence, whenever $\beta _ { i j } = 0$ then also ${ \theta _ { i j } } = 0$ and vice versa. We can, therefore, consider each node in turn and determine the non-zero coeficients of the regressions. Because we have both $\beta _ { i j }$ and $\beta _ { j i }$ we will have to decide to use the and-rule, where both $\beta _ { i j }$ and $\beta _ { j i }$ are non-zero to include the edge $( i , j )$ or the $o r \cdot$ -rule, where either $\beta _ { i j }$ or $\beta _ { j i }$ is non-zero to include the edge (i, j) (Meinshausen and B¨uhlmann, 2006).

## B Excess risk and mean squared error (Section 4)

We defined the excess risk as

$$
\begin{array} { r l } & { R ( J ) = \mathbb { E } \left[ ( Y _ { 0 } - X _ { 0 } ^ { \top } \hat { \beta } _ { J } ) ^ { 2 } - ( Y _ { 0 } - X _ { 0 } ^ { \top } \beta _ { S } ) ^ { 2 } \right] } \\ & { \quad \quad \quad = \mathbb { E } \left[ ( Y _ { 0 } - X _ { 0 } ^ { \top } \hat { \beta } _ { J } ) ^ { 2 } \right] - \mathbb { E } \left[ ( Y _ { 0 } - X _ { 0 } ^ { \top } \beta _ { S } ) ^ { 2 } \right] . } \end{array}
$$

where the expectation is conditional on the training data X. This is equivalent to the risk defined in Hastie et al. (2022) and Bartlett et al. (2021). Then we obtain the following decomposition.

Lemma B.1 (Mean squared error decomposition) Assume that the linear model $( 4 )$ is correct. The excess risk (test mean squared error) can be de-

composed as $R ( J ) = B ( J ) + V ( J )$ in $( \boldsymbol { \theta } )$ with

$$
\begin{array} { r } { B ( J ) = ( \beta _ { S } - \mathbb { E } \hat { \beta } _ { J } ) ^ { \top } \Sigma ( \beta _ { S } - \mathbb { E } \hat { \beta } _ { J } ) , \qquad V ( J ) = \mathbb { E } ( \hat { \beta } _ { J } - \mathbb { E } \hat { \beta } _ { J } ) ^ { \top } \Sigma ( \hat { \beta } _ { J } - \mathbb { E } \hat { \beta } _ { J } ) , } \end{array}
$$

where $B ( J )$ is called the squared bias and $V ( J )$ is called the variance.

Proof Assuming that the linear model is correct, so that $\mathbb { E } Y _ { 0 } - X _ { 0 } ^ { \top } \beta _ { S } = 0$ we get

$$
\begin{array} { r l } & { R ( J ) = \mathbb { E } \left[ ( Y _ { 0 } - X _ { 0 } ^ { \top } \beta _ { S } + X _ { 0 } ^ { \top } ( \beta _ { S } - \hat { \beta } _ { J } ) ) ^ { 2 } \right] - \mathbb { E } \left[ ( Y _ { 0 } - X _ { 0 } ^ { \top } \beta _ { S } ) ^ { 2 } \right] } \\ & { \qquad = \mathbb { E } \left( X _ { 0 } ^ { \top } ( \beta _ { S } - \mathbb { E } \hat { \beta } _ { J } ) - X _ { 0 } ^ { \top } ( \hat { \beta } _ { J } - \mathbb { E } \hat { \beta } _ { J } ) \right) ^ { 2 } } \end{array}
$$

This gives

$$
\begin{array} { r l } & { R ( J ) = \mathbb { E } \left( X _ { 0 } ^ { \top } ( \beta _ { S } - \mathbb { E } \hat { \beta } _ { J } ) \right) ^ { 2 } + \mathbb { E } \left( X _ { 0 } ^ { \top } ( \hat { \beta } _ { J } - \mathbb { E } \hat { \beta } _ { J } ) \right) ^ { 2 } } \\ & { \quad \quad \quad - 2 \mathbb { E } ( \beta _ { S } - \mathbb { E } \hat { \beta } _ { J } ) ^ { \top } X _ { 0 } X _ { 0 } ^ { \top } ( \hat { \beta } _ { J } - \mathbb { E } \hat { \beta } _ { J } ) } \end{array}
$$

Recalling that $( X , Y )$ and $( X _ { 0 } , Y _ { 0 } )$ are independent, we obtain for the cross product

$$
\begin{array} { r } { \mathbb { E } ( \beta _ { S } - \mathbb { E } \hat { \beta } _ { J } ) ^ { \top } X _ { 0 } X _ { 0 } ^ { \top } ( \hat { \beta } _ { J } - \mathbb { E } \hat { \beta } _ { J } ) = ( \beta _ { S } - \mathbb { E } \hat { \beta } _ { J } ) ^ { \top } \Sigma \mathbb { E } ( \hat { \beta } _ { J } - \mathbb { E } \hat { \beta } _ { J } ) = 0 , } \end{array}
$$

because $\mathbb { E } ( \hat { \beta } _ { J } - \mathbb { E } \hat { \beta } _ { J } ) = 0$ , and where $\Sigma = \mathbb { E } ( X _ { 0 } X _ { 0 } ^ { \top } )$ , which is the same as the covariance matrix of the predictors $\mathbb { E } ( X ^ { \top } X )$ . Then we obtain $B ( J )$ and $V ( J )$ in (7) from the two remaining terms. ✷

This is the famous variance and squared bias decomposition (Hastie et al., 2022).

Next, we show the formulation of the bias and variance of the MSE with the ridge estimator $\hat { \beta } = { \cal W } ( \alpha ) X ^ { \top } Y$ , where $W ( \alpha ) = ( X ^ { \top } X + \alpha I )$

Lemma B.2 Assume that the linear model $( 4 )$ is true and that $\Sigma = \sigma _ { \xi } ^ { 2 } I$ is the covariance matrix of the covariates. Then the squared bias is

$$
B ( J ) = \sigma _ { \xi } ^ { 2 } \frac { \gamma _ { 1 } \alpha ^ { 2 } } { ( \lambda _ { 1 } + \alpha ) ^ { 2 } } + \sigma _ { \xi } ^ { 2 } \sum _ { j = 2 } ^ { n } \frac { \alpha ^ { 2 } } { ( \lambda _ { j } + \alpha ) ^ { 2 } } .
$$

Proof Recall that the expectation is conditional on X and $( X _ { 0 } , Y _ { 0 } )$ and $( X , Y )$ are independent. Plugging in the ridge estimator (3) in the bias term and assuming $\Sigma = \sigma _ { \xi } I$ , yields

$$
\begin{array} { r l } & { B ( J ) = \mathbb { E } ( \beta _ { S } - \mathbb { E } \hat { \beta } _ { J } ) ^ { \top } X _ { 0 } ^ { \top } X _ { 0 } ( \beta _ { S } - \mathbb { E } \hat { \beta } _ { J } ) } \\ & { \qquad = \mathrm { t r } ( \beta _ { S } - W ( \alpha ) ^ { - 1 } X ^ { \top } X \beta _ { S } ) ( \beta _ { S } - W ( \alpha ) ^ { - 1 } X ^ { \top } X \beta _ { S } ) ^ { \top } \mathbb { E } ( X _ { 0 } ^ { \top } X _ { 0 } ) } \\ & { \qquad = \mathrm { t r } ( ( I - W ( \alpha ) ^ { - 1 } X ^ { \top } X ) \beta _ { S } ) ( ( I - W ( \alpha ) ^ { - 1 } X ^ { \top } X ) \beta _ { S } ) ^ { \top } \Sigma } \\ & { \qquad = \sigma _ { \xi } ^ { 2 } \mathrm { t r } ( I - W ( \alpha ) ^ { - 1 } X ^ { \top } X ) ^ { 2 } \beta _ { S } \beta _ { S } ^ { \top } . } \end{array}
$$

Let $\lambda _ { j }$ be the eigenvalues of $X ^ { \top } X$ and $\gamma _ { 1 }$ be the only non-zero eigenvalue of $\beta _ { S } \beta _ { S } ^ { \top }$ . Then, noting that $U ^ { \top } U = I$ , we obtain for $W ( \alpha ) = X ^ { \top } X + \alpha I$ the eigenvalue decomposition for $W ( \alpha )$ is $U \Lambda ( \alpha ) U ^ { \top }$ with diagonal matrix $\Lambda ( \alpha )$ with elements $\lambda _ { j } + \alpha$ . The squared inverse $W ( \alpha ) ^ { - 2 }$ is then $U \Lambda ( \alpha ) ^ { - 2 } U ^ { \top }$ with diagonal matrix $\bar { \Lambda } ( \alpha ) ^ { - 2 }$ with elements $1 / ( \lambda _ { j } + \alpha ) ^ { 2 }$ . Plugging this in the above result, we obtain $B ( J )$ . This completes the proof. ✷

Lemma B.3 Assume that the linear model $( 4 )$ is true and that $\Sigma = \sigma _ { \xi } ^ { 2 } I$ is the covariance matrix of the covariates. Then the variance is

$$
V ( J ) = \sigma _ { \xi } ^ { 2 } \sigma _ { e } ^ { 2 } \sum _ { j = 1 } ^ { n } \frac { \lambda _ { j } } { ( \lambda _ { j } + \alpha ) ^ { 2 } } .
$$

Proof Again, we note that the expectation is conditional on X and $( X _ { 0 } , Y _ { 0 } )$ and $( X , Y )$ are independent. Plugging in the ridge estimator (3) in the variance term and assuming $\Sigma = \sigma _ { \xi } I$ , yields

$$
\begin{array} { r l } & { V ( J ) = \mathbb { E } \mathrm { t r } ( \hat { \beta } _ { J } - \mathbb { E } \hat { \beta } _ { J } ) ( \hat { \beta } _ { J } - \mathbb { E } \hat { \beta } _ { J } ) ^ { \top } X _ { 0 } ^ { \top } X _ { 0 } } \\ & { \quad \quad = \mathrm { t r } ( W ( \alpha ) ^ { - 1 } X ^ { \top } Y - W ( \alpha ) ^ { - 1 } X ^ { \top } X \beta _ { S } ) } \\ & { \quad \quad \quad \quad \quad \times \left( W ( \alpha ) ^ { - 1 } X ^ { \top } Y - W ( \alpha ) ^ { - 1 } X ^ { \top } X \beta _ { S } \right) ^ { \top } \mathbb { E } ( X _ { 0 } ^ { \top } X _ { 0 } ) } \\ & { \quad \quad = \mathrm { t r } W ( \alpha ) ^ { - 2 } ( X ^ { \top } ( X \beta _ { S } + e - X \beta _ { S } ) ) ( X ^ { \top } ( X \beta _ { S } + e - X \beta _ { S } ) ) ^ { \top } \Sigma } \\ & { \quad \quad = \sigma _ { \xi } ^ { 2 } \sigma _ { e } ^ { 2 } \mathrm { t r } ( W ( \alpha ) ^ { - 2 } X ^ { \top } X ) , } \end{array}
$$

where we used that the variance of the residuals $e$ is $\sigma _ { e } ^ { 2 } I .$ . Similar to Lemma B.2 we use that the eigenvalue decomposition for $W ( \alpha )$ is $U \Lambda ( \alpha ) U ^ { \top }$ with diagonal matrix $\Lambda ( \alpha )$ with elements $\lambda _ { j } + \alpha$ . The squared inverse $W ( \alpha ) ^ { - 2 }$ is then $U \Lambda ( \alpha ) ^ { - 2 } U ^ { \top }$ with diagonal matrix $\Lambda ( \alpha ) ^ { - 2 }$ with elements $1 / ( \lambda _ { j } + \alpha ) ^ { 2 }$ Plugging this in the above results, we obtain $V ( J )$ . This completes the proof. ✷

## B.1 Model misspecification

Misspecification is defined with respect to a family of models. Let M denote a class of probability distributions that are indexed by parameters $\theta \in \Theta \subseteq $ $\mathbb { R } ^ { p }$ , i.e., $\mathcal { M } = \{ p _ { \theta } : \theta \in \Theta \}$ (we assume that the probability distribution is dominated by Lebesgue measure so that a density $p _ { \theta }$ exists, Jacod and Protter, 2004). An example of a model is a class of normal distributions with mean $\mu$ and variance $\sigma ^ { 2 }$ , where the mean is generated by a nonlinear function, such as $f : \mathbb { R } ^ { p } \times \mathbb { R } ^ { p } $ R such that $x \mapsto 1 / ( 1 + \exp ( - x ^ { \top } \beta ) )$ ; then $\theta = ( \beta , \sigma ^ { 2 } )$ is the $p { + 1 }$ dimensional parameter vector. Furthermore, let $\mathcal { M } _ { \sf I n } \subset \mathcal { M }$ be a strict subset consisting of only those distributions where the conditional mean $\mathbb { E } ( Y \mid x )$ is generated by a linear function $x \mapsto x ^ { \top } \beta$ . We say that a model is misspecified if the correct distribution is not in the class of probability distributions we consider in the optimisation (Vaart, 1998; Gr¨unwald, 2007; Kleijn, 2026), i.e., $p _ { 0 } \notin \mathcal { M } _ { \mathsf { l i n } }$ , where $p _ { 0 } = p _ { \theta _ { 0 } }$ . In our setting, this means that the distribution has a mean that is generated by a nonlinear function, while we use only a linear function for optimisation. In such a case, the linear model $x ^ { \top } { \hat { \beta } }$ can be interpreted as corresponding to a density $p _ { \theta }$ that is closest within $\mathcal { M } _ { \sf I I I }$ to $p _ { 0 }$ in M in terms of Kullback-Leibler divergence (Vuong, 1989; Vaart, 1998). Here we have suppressed the variance parameter $\sigma ^ { 2 }$ of the residuals $Y - f$ where f is the true linear model for the conditional mean. However, seminal work by Gr¨unwald and van Ommen (2017) shows that misspecification using linear models, as we do here, also leads to misspecification of the variance $\sigma ^ { 2 }$ (i.e., it depends on the value of $x )$ , and leads to non-convexity of the model space $\mathcal { M } _ { \sf I I I }$ . This in turn leads to models that can be combinations of linear models (like with variance depending on $x )$ that are in the convex hull of the model space $\mathcal { M } _ { \sf I I I }$ and are closer to the true generating model not in $\mathcal { M } _ { \sf I I I }$ than any model in $\mathcal { M } _ { \sf I I I }$ itself; Gr¨unwald and van Ommen (2017) called this bad misspecification.

We approximate a function f by a linear function $X \beta$ with $\beta \in \mathbb { R } ^ { p }$ and $p > n$ very large.

Lemma B.4 (Mean squared error for misspecification) Assume that the true function $f : \mathbb { R } ^ { p } \times \mathbb { R } ^ { p } \to \mathbb { R }$ , and $Y = f ( X , \beta _ { S } ) + e$ , with $\beta _ { S }$ the true parameter vector in f. We estimate $f ( X , \beta _ { S } )$ by $X \beta _ { J }$ with $| J | = p > n$ . Then the risk under model misspecification is

$$
\begin{array} { r l } & { R _ { f } ( J ) = \mathbb { E } \left[ ( f ( X _ { 0 } , \beta _ { S } ) - \mathbb { E } ( X _ { 0 } ^ { \top } \hat { \beta } _ { J } ) ^ { 2 } \right] + \mathbb { E } \left[ ( f ( X _ { 0 } , \beta _ { S } ) - \mathbb { E } ( X _ { 0 } ^ { \top } \hat { \beta } _ { J } ) ^ { 2 } \right] } \\ & { \phantom { \beta _ { 0 } } + 2 \mathbb { E } \left[ ( Y _ { 0 } - f ( X _ { 0 } , \beta _ { S } ) ) ( f ( X _ { 0 } , \beta _ { S } ) - X _ { 0 } ^ { \top } \hat { \beta } _ { J } ) \right] } \end{array}
$$

which we refer to as $R _ { f } ( J ) = B _ { f } ( J ) + V _ { f } ( J ) + M _ { f } ( J )$

Proof The proof follows Lemma B.1 except that the cross products do not vanish because of misspecification. We have that the excess risk is

$$
R _ { f } ( J ) = \operatorname { \mathbb { E } } \left[ ( Y _ { 0 } - f ( X _ { 0 } , \beta _ { S } ) + f ( X _ { 0 } , \beta _ { S } ) - X _ { 0 } ^ { \top } \hat { \beta } _ { J } ) ) ^ { 2 } \right] - \operatorname { \mathbb { E } } \left[ ( Y _ { 0 } - f ( X _ { 0 } , \beta _ { S } ) ) ^ { 2 } \right] .
$$

Working out the cross products cancels E $[ ( Y _ { 0 } - f ( X _ { 0 } , \beta _ { S } ) ^ { 2 } ]$ , without assuming that the model is correct. What remains is $B _ { f } ( J ) , V _ { f } ( J )$ , similarly obtained as in Lemma B.1, but the cross product that remains is

$$
M _ { f } ( J ) = 2 \mathbb { E } \left[ ( Y _ { 0 } - f ( X _ { 0 } , \beta _ { S } ) ) ( f ( X _ { 0 } , \beta _ { S } ) - X _ { 0 } ^ { \top } \hat { \beta } _ { J } ) \right] .
$$

This was to be proved.

Proposition B.5 (Misspecification bound) Assume the model with additive noise $Y = f ( X , \beta _ { S } ) + e ,$ , where the conditional mean $\mathbb { E } ( Y \mid x )$ is given by the function is $f : \mathbb { R } ^ { p } \times \mathbb { R } ^ { p } \to \mathbb { R }$ and $e = Y - f ( X , \beta _ { S } )$ has mean zero and finite variance $\sigma ^ { 2 }$ . Furthermore, we estimate $f ( X , \beta _ { S } )$ by $X \beta _ { J }$ with $| J | = p > n$ and assume that the approximation error is bounded in probability by $K , i . e .$ $f ( X _ { 0 } , \beta _ { S } ) - X _ { 0 } ^ { \top } \hat { \beta } _ { J } = O _ { p } ( K )$ . Then the misspecification is bounded by

$$
M _ { f } ( J ) = 2 \mathbb { E } \left[ \left| ( Y _ { 0 } - f ( X _ { 0 } , \beta _ { S } ) ) ( f ( X _ { 0 } , \beta _ { S } ) - X _ { 0 } ^ { \top } \hat { \beta } _ { J } ) \right| \right] \leq 2 \sigma O ( K )
$$

Proof By the Cauchy-Schwarz theorem we can bound the expectation of the product, and so,

$$
\mathbb { E } \left[ \big | e \big ( f ( X _ { 0 } , \beta _ { S } ) - X _ { 0 } ^ { \top } \hat { \beta } _ { J } \big ) \big | \right] ^ { 2 } \leq \mathbb { E } ( e ^ { 2 } ) \mathbb { E } \left[ \big ( f ( X _ { 0 } , \beta _ { S } ) - X _ { 0 } ^ { \top } \hat { \beta } _ { J } ) ^ { 2 } \right]
$$

where we used that $Y _ { 0 } - f ( X _ { 0 } , \beta _ { S } ) = e$ . By assumption, e has mean 0 and variance $\sigma ^ { 2 }$ , and so $\mathbb { E } ( e ^ { 2 } ) ~ = ~ \sigma ^ { 2 }$ . For the second term, we assumed that $f ( X _ { 0 } , \beta _ { S } ) - X _ { 0 } ^ { \top } \hat { \beta } _ { J } = O _ { p } ( K )$ , and so

$$
\mathbb { E } \left[ ( f ( X _ { 0 } , \beta _ { S } ) - X _ { 0 } ^ { \top } \hat { \beta } _ { J } ) ^ { 2 } \right] = \mathbb { E } \left[ O _ { p } ( K ) ^ { 2 } \right] = O ( K ^ { 2 } )
$$

Taking the square root of the product gives the required bound.

## C High-dimensional space

For some intuition on high-dimensional spaces, the isotropic Gaussian, i.e., with mean 0 and variances of 1 and covariances of 0, is convenient. Let X be a Gaussian random variable of dimension p with mean 0 and isotropic covariance matrix $\mathbb { E } ( X X ^ { \top } ) = \Sigma$ of dimensions $p \times p$ . We will consider an inequality that shows that with high probability, for large dimension $p$ most of the draws from an isotropic distribution are at ${ \sqrt { p } } ,$ , the radius of the spheroid of the distribution.

We can make this clear by considering the probability of an observation falling within a certain bound of the origin (Vershynin, 2018, Chapter 3). Suppose we are interested in the length (norm) of the vector X in p dimensions, i.e., we want to know the probability of $| | X | | _ { 2 } = { \sqrt { X _ { 1 } ^ { 2 } + X _ { 2 } ^ { 2 } + \cdots X _ { p } ^ { 2 } } }$ being smaller than $\sqrt { p }$ . It is intuitive that the length of X is indeed $\sqrt { p }$ because

$$
\mathbb { E } | | X | | _ { 2 } ^ { 2 } = \mathbb { E } \sum _ { j = 1 } ^ { p } X _ { j } ^ { 2 } = \sum _ { j = 1 } ^ { p } \mathbb { E } X _ { j } ^ { 2 } = p .
$$

So, we should expect many of the values of the square $| | X | | _ { 2 } ^ { 2 }$ to be around ${ \sqrt { p } } .$ Suppose we are considering the probability of X being near $0 , \mathrm { i . e . , } | | X | | _ { 2 } \approx 0$ Then we find that

$$
\mathbb { P } ( | | X | | _ { 2 } - \sqrt { p } | \le t ) \approx \mathbb { P } ( \sqrt { p } \le t ) .
$$

![](images/f97cefd6fe67f5766400dabf0b5b59ae7d40753df785047e2b4279dce77654ca.jpg)  
(a)

![](images/86fd3cd9f34ff0ed548b16cebeab642408fd8d44015107939f09a5d078692593.jpg)  
(b)  
Fig. C.1. In (a) we see that, indeed, the volume of a p-dimensional ball with radius 1 in high dimensions gets smaller and smaller, already are $p = 2 0$ is the volume close to 0. In (b) is the consequence of this, where we see that the fraction of observations from an isotropic distribution near the surface increases exponentially to 1.

$\mathrm { S o } .$ for small values $t ,$ say 1 or 2, the probability can only be high if $p$ is small.

Another way to consider this is by looking at the volume of $\mathrm { ~ a ~ } p \mathrm { . ~ }$ -dimensional ball. The volume of the p-dimensional ball in Euclidean space $\mathbb { R } ^ { p }$ with radius r is (Giraud, 2014, Section 1.2)

$$
\mathbb { B } ^ { p } ( r ) = { \frac { \pi ^ { p / 2 } } { \Gamma ( p / 2 + 1 ) } } r ^ { p } \approx \left( { \frac { 2 \pi e r ^ { 2 } } { p } } \right) ^ { p / 2 } { \frac { 1 } { p \pi } } \quad { \mathrm { f o r ~ l a r g e ~ } } p ,\tag{C.1}
$$

where Γ is the Gamma function $\textstyle \Gamma ( \alpha ) = \int _ { 0 } ^ { \infty } t ^ { \alpha - 1 } e ^ { - t } d t$ and $\alpha > 0$ . Figure $\mathrm { { C . 1 ( a ) } }$ shows that the volume of the ball is close to 0 already when the dimension is about 20. Moreover, Figure C.1(b) shows that most of the points from the Gaussian isotropic distribution will be near the surface of the $p { \cdot }$ -dimensional ball $\mathbb { B } ^ { p } ( { \sqrt { p } } )$ . The figure shows the fraction of the points outside an inner $p \mathrm { - }$ dimensional ball of radius 0.95r, so very close to the ball of radius $^ r .$ The fraction increases to 1 exponentially fast (Giraud, 2014). Already at $p = 3 0$ the fraction in the last 5% of the ball is nearly 80%. This shows that most of the points will be near the surface of the p-dimensional ball.

## D Geometry and metric entropy of Gaussian process (Section 5)

Here we first prove that the the ratio of signal and noise volumes can, for linear models in the high-dimensional setting, be written in terms of the ridge estimator. This also implies that the ridge parameter α can be interpreted as the inverse of the signal to noise ratio.

Lemma D.1 (Volume linear model) We assume the linear model $X \beta$ where X is an $n \times p$ matrix and $\beta$ is a p-dimensional vector, such that $Y = X \beta + e$ and $e$ is Gaussian with mean 0 and variance $\sigma _ { e } ^ { 2 } I$ . The volume ratio of the linear model with ridge regression with respect to the noise e is

$$
\frac { v o l ( X \beta ) } { v o l ( e ) } = \sqrt { \operatorname* { d e t } ( X ^ { \top } X + \alpha I _ { p } ) } ,
$$

where $\alpha ^ { - 1 } = | | \beta | | _ { 2 } ^ { 2 } / \sigma _ { e } ^ { 2 }$

Proof Let $\operatorname { v o l } ( \beta ) = \mathbb { B } ^ { n } ( { \sqrt { n } } | | \beta | | _ { 2 } )$ be the n-dimensional ball with radius $\sqrt { n } | | \beta | | _ { 2 }$ . Compared to the volume of the noise B $( { \sqrt { n \sigma ^ { 2 } } } )$ , we obtain the ratio

$$
{ \frac { \mathrm { v o l } ( X \beta ) } { \mathrm { v o l } ( e ) } } = \sqrt { \operatorname* { d e t } ( X X ^ { \top } ) } { \frac { \mathbb { B } ^ { n } ( \sqrt { n } | | \beta | | _ { 2 } ) } { \mathbb { B } ^ { n } ( \sqrt { n \sigma _ { e } ^ { 2 } } ) } } .
$$

An n-dimensional ball $\mathbb { B } ^ { n } ( r )$ with radius $r$ can be written in terms of the $n -$ dimensional ball with radius 1: $\mathbb { B } ^ { n } ( 1 ) r ^ { n }$ . Hence we find for the ratio of volumes vo $. ( X \beta ) / \mathrm { v o l } ( e )$

$$
\sqrt { \operatorname * { d e t } ( X X ^ { \top } ) } \left( { \frac { | | \beta | | _ { 2 } ^ { 2 } } { \sigma _ { e } ^ { 2 } } } \right) ^ { n / 2 } = \sqrt { \operatorname * { d e t } ( \alpha ^ { - 1 } X X ^ { \top } ) } ,
$$

where $\alpha ^ { - 1 } = | | \beta | | _ { 2 } ^ { 2 } / \sigma _ { e } ^ { 2 }$ is the signal-to-noise-ratio (SNR).

It turns out that when $p > n$ we obtain $X ^ { \top } ( X X ^ { \top } ) Y$ as the so-called minimum norm estimate $\hat { \beta } ^ { M N }$ of $\beta$ (Ben-Israel and Greville, 1974; Bartlett et al., 2021). The minimum-norm solution is related to the ridge estimator as follows

$$
\operatorname* { l i m } _ { \alpha  0 ^ { + } } ( X ^ { \top } X + \alpha I ) ^ { - 1 } X ^ { \top } = X ^ { \top } ( X X ^ { \top } ) ^ { - 1 } .
$$

Intuitively, the minimum-norm solution is obtained for the smallest $\alpha$ that makes the augmented inverse of $X ^ { \top } X$ possible. It is therefore reasonable to substitute for $X X ^ { \top }$ the ridge version $X ^ { \top } X + \alpha I$ , where $\alpha$ is small. Hence, we obtain the volume ratio vo $( ( X ^ { \top } , { \sqrt { \alpha } } I ) ^ { \top } \beta ) / { \mathrm { v o l } } ( e )$ with the ridge version (Cheema and Sugiyama, 2020)

$$
{ \sqrt { \operatorname* { d e t } ( \alpha ^ { - 1 } X X ^ { \top } + I _ { n } ) } } = { \sqrt { \operatorname* { d e t } ( X ^ { \top } X + \alpha I _ { p } ) } } ,
$$

by the Weinstein–Aronszajn identity.

Because $\alpha = 1 / \mathrm { { S N R } } = \sigma _ { e } ^ { 2 } / | | \beta | | _ { 2 } ^ { 2 }$ , we obtain small $\alpha$ , close to the minimumnorm estimate, if we have high signal $| | \beta | | _ { 2 } ^ { 2 }$

Clearly, the tuning parameter α can be expressed as the inverse of the signal-to-noise ratio (SNR). The SNR is defined as the ratio of the Euclidean length of the parameters $( | | \beta | | _ { 2 } ^ { 2 }$ , see Appendix 2.2) and the noise variance $\sigma _ { e } ^ { 2 }$ then $\alpha = 1 / \mathrm { S N R }$ and $\mathrm { S N R } = | | \beta | | _ { 2 } ^ { 2 } / \sigma _ { e } ^ { 2 }$ . The Euclidean length is a representation of the volume for the variables $\beta .$ . So, if we restrict $| | \beta | | _ { 2 } ^ { 2 }$ to be no larger than $c$ (ridge regression), then we are restricting the volume of the possible values that the parameters in $\beta$ can take. This implies that when we increase the volume $( | | \beta | | _ { 2 } ^ { 2 } )$ we reduce the tuning parameter $\alpha _ { \because }$ , and hence penalise the regression less. This can be interpreted as follows: imposing a higher α in ridge regression will lead to decreased variance (this was shown in Section 4). This also implies that the peak at the interpolation point in Figure 3 is an artifact of applying a suboptimal tuning parameter (see Figure H.1, where in the middle row the peak has disappeared upon careful selection of $\alpha )$ . Summarising, we can say the following.

(a) We can increase α directly, reducing the test variance, and hence obtaining reasonable test MSE. This leads indirectly to a lower SNR (because $\alpha = 1 / \mathrm { S N R } )$ ; or

(b) we can constrain the size of $| | \beta | | _ { 2 }$ in the ridge estimator, therefore, decreasing the model complexity (volume) of the model. This will reduce SNR and hence, increase $\alpha ,$ leading to a decrease in the test variance. We do this by constraining the ridge estimator with small $c$ such that $\| \beta \| _ { 2 } \leq c .$

In both cases $( a )$ and (b) we are either directly or indirectly constraining the total signal $| | \beta | | _ { 2 }$ . In model selection this can be done by either increasing α, or equivalently, decreasing c in the ridge constraint $| | \beta | | _ { 2 } \leq c ,$ or by incorporating the complexity (volume) of the model (the n-dimensional ball $\mathbb { B } ^ { n } ( | | \beta | | _ { 2 } )$ , see Appendix D).

## E Akaike information criterion

For the derivation of the Akaike information criterion $\mathrm { ( A I C ) }$ , we follow Claeskens and Hjort (2008). The AIC can be derived from the Kullbeck-Leibler divergence (KL). The KL is a way to quantify the overlap between two distributions. The KL is 0 if the distributions are the same (almost surely) and is $> 0$ if they are not the same. Hence, the AIC aims to minimise the KL across the diferent models.

Let $f$ and $g$ be two densities, where we designate $g$ as the true underlying distribution and we use $f$ as an approximation. In practice we estimate a parameter $\hat { \beta }$ that we plug-in $f ,$ denoted by $f ( \cdot , { \hat { \boldsymbol { \beta } } } )$ . The KL is for $f$ and $g$ defined as (Kullback and Leibler, 1951; Cover and Thomas, 2006)

$$
\mathrm { K L } ( f , g ) = \mathbb { E } _ { Y \mid x } \log g ( Y \mid x ) - \mathbb { E } _ { Y \mid x } \log f ( Y \mid x , \hat { \beta } ) ,
$$

where $\mathbb { E } _ { Y \mid x }$ indicates the expectation over $Y$ conditional on $x$ (the values of the predictors in the nodewise GGM). Across diferent models the term $\mathbb { E } _ { Y \mid x } \log g ( Y \mid x )$ is fixed and so we concentrate on the second term. So, maximising the second term of the KL equals minimising the KL itself. The ex-

pectation of this second term is

$$
L = \mathbb { E } _ { X } \left( \mathbb { E } _ { Y \mid x } \log f ( Y \mid x , { \hat { \beta } } ) \right) .
$$

The AIC tries to estimate this expectation and then select the model with the highest value (and hence the minimal value of KL). This expectation is estimated by the log-likelihood $\begin{array} { r } { { \frac { 1 } { n } } L ( { \hat { \boldsymbol { \beta } } } ) = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } f ( Y _ { i } \mid { \dot { x _ { i } } } , { \hat { \boldsymbol { \beta } } } ) } \end{array}$ . And it can then be shown that this estimate is biased and needs to be corrected, since

$$
\mathbb { E } ( \frac { 1 } { n } L ( \hat { \beta } ) - L ) \approx \frac { p } { n } \mathrm { a n d , h e n c e } \frac { 1 } { n } ( - 2 L ( \hat { \beta } ) + 2 p ) .
$$

Discarding the $1 / n$ leads to the standard AIC.

Here, for the GGM we used the Gaussian distribution for $f .$ . Then (Seber and Lee, 2012)

$$
\sum _ { i = 1 } ^ { n } \log f ( Y _ { i } \mid x _ { i } , \beta ) = \sum _ { i = 1 } ^ { n } - { \frac { 1 } { 2 \sigma ^ { 2 } } } ( Y _ { i } - x _ { i } ^ { \top } \beta ) ^ { 2 } - { \frac { n } { 2 } } \log \sigma ^ { 2 } - { \frac { n } { 2 } } \log 2 \pi .
$$

Plugging in our ridge estimate $\hat { \beta } _ { J }$ for model J (see (3)), multiplying by -2 and ignoring constants, we obtain the AIC

$$
\mathrm { A I C } ( J ) = n \log \hat { \sigma } _ { J } + 2 p _ { J } ,
$$

where $p _ { J }$ is the number of parameters for model J and

$$
\hat { \sigma } _ { J } ^ { 2 } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( y _ { i } - x _ { i } ^ { \top } \hat { \beta } _ { J } ) ^ { 2 }
$$

is plugged in for $\sigma ^ { 2 }$ . The standard AIC has a fixed (independent of the data) value of the ridge parameter α. The AIC-CV has a ridge parameter α that has been obtained with k-fold cross validation.

## F Minimum description length: optimised

The minimum description length (MDL) principle has been extended to the high-dimensional setting of $n \ < \ p$ in another way than in Section 5.1. The principle of approximating the code length for the data given the model and the code length for the model is the same, but in order to account for the high-dimensions (such that $d / n \to \gamma < \infty )$ , the efective degrees of freedom are used in combination with an optimal α for the efective degrees of freedom (Dwivedi et al., 2020). The MDL is then obtained by minimising a function for α that gives the smallest MDL value. The function to be optimised over α is

$$
\mathrm { M D L - o p t } = \frac { n } { 2 } \log \hat { \sigma } ^ { 2 } + \frac { 1 } { 2 \hat { \sigma } ^ { 2 } } | | \hat { \beta } | | _ { 2 } ^ { 2 } + \sum _ { j = 1 } ^ { \operatorname* { m i n } \{ n , d \} } \log \left( 1 + \frac { \lambda _ { j } } { \alpha } \right) ,
$$

where $\lambda _ { j }$ are the eigenvalues of $X ^ { \top } X$ and $\hat { \beta }$ is the ridge estimator with value $\alpha .$ . The second term is included here because of the Bayesian interpretation of MDL in Dwivedi et al. (2020), where this term is entered as a prior for $\beta .$ This version of MDL is referred to as MDL-opt.

## G MDL leads to underfit in high-dimensional settings

Proposition 5.1 Let S denote the true support for a (possibly nonlinear) data generating process of $Y _ { i }$ for $i = 1 , \ldots , n$ with n fixed, and let J represent the set of predictors such that $S \subset J ,$ where $| S | = d$ and $| J | = p$ . We test linear models $X \beta _ { J }$ for any J (including S) by MDL as given in (10) with (11). Let $p$ increase and $p > d .$ . If $\mathbb { E } \hat { \sigma } _ { J } ^ { 2 } > \gamma$ , for some $\gamma > 0$ , then with probability at least $\begin{array} { r } { 1 - { \frac { d } { p } } \exp ( - ( p - d ) ) } \end{array}$ it holds that ${ \hat { S } } \subseteq S ;$ equivalently, the probability of false positives goes to 0.

Proof We first note that

$$
\mathbb { P } ( \hat { S } \subseteq S ) = \mathbb { P } ( \operatorname* { m i n } _ { \boldsymbol { r } } \mathrm { M D L } ( \boldsymbol { J } ) \leq \mathrm { M D L } ( S ) : | \boldsymbol { J } | \geq | S | )
$$

And using the complement rule and exponentiation we can rewrite this as

$$
1 - \mathbb { P } ( \operatorname* { m i n } _ { J } \exp [ \mathrm { M D L } ( S ) - \mathrm { M D L } ( J ) ] < 1 : | J | \geq | S | )
$$

Then we apply Markov’s inequality to show that for $d = | S | \leq | J | = p$

$$
\mathbb { P } ( \exp [ \mathrm { M D L } ( S ) - \mathrm { M D L } ( J ) ] < 1 ) \le \mathbb { E } \left( \exp [ \mathrm { M D L } ( S ) - \mathrm { M D L } ( J ) ] \right)
$$

We start with the first part of the MDL, the log of the empirical variance, defined as (using maximum likelihood) $\begin{array}{c} \begin{array} { r } { \hat { \sigma } _ { J } ^ { 2 } = \hat { \bar { \mathbf { \eta } } } \end{array} \overset { \sim } { \underset { n } { \ln } } | | Y - X ^ { \top } \hat { \beta _ { J } } | | ^ { 2 } } \end{array}$ . For models with $p = | J | \geq | S |$ , we have that the in-sample variance is $\hat { \sigma } _ { J } ^ { 2 } \leq \hat { \sigma } _ { S } ^ { 2 } \mathbb { P } \mathrm { - a . e }$ (Bilodeau and Brenner, 1999), and so $\mathbb { E } \hat { \sigma } _ { J } ^ { 2 } \le \mathbb { E } \hat { \sigma } _ { S } ^ { 2 }$ by monotonicity. Therefore, and because both terms are positive, using Jensen’s inequality for concave functions such that $\mathbb { E } ( \log X ) \leq \log \mathbb { E } ( X )$ , we obtain that

$$
0 \leq \mathbb { E } \log \hat { \sigma } _ { S } ^ { 2 } - \mathbb { E } \log \hat { \sigma } _ { J } ^ { 2 } \leq \log \mathbb { E } \hat { \sigma } _ { S } ^ { 2 } - \log \mathbb { E } \hat { \sigma } _ { J } ^ { 2 } = \log \frac { \mathbb { E } \hat { \sigma } _ { S } ^ { 2 } } { \mathbb { E } \hat { \sigma } _ { J } ^ { 2 } }
$$

The expectation $\mathbb { E } _ { n } ^ { \underline { { 1 } } } | | Y - X ^ { \top } \hat { \beta } _ { J } | | ^ { 2 }$ under the true model is $\begin{array} { r l } { \sigma ^ { 2 } + \mathbb { E } \frac { 1 } { n } | | f ( X , \beta _ { S } ) - } & { { } } \end{array}$ $X ^ { \top } { \hat { \beta } } _ { J } | | ^ { 2 }$ , and similarly for the expectation $\sigma ^ { 2 } + \mathbb { E } \frac { 1 } { n } | | f ( X , \beta _ { S } ) - X ^ { \top } \hat { \beta } _ { J } | | ^ { 2 }$ , which are bounded by the assumption on misspecification in Lemma B.4. Using monotonicity above and the assumption that $\mathbb { E } \hat { \sigma } _ { J } ^ { 2 } > 0$ is bounded away from 0, the ratio above is positive and bounded by $K / \gamma$ , with $| f ( X , \beta _ { S } ) - X ^ { \top } \hat { \beta } _ { S } | < K$

Then we move on to the next part tr $\cdot ( S _ { J } )$ log n of MDL. In the setting of linear regression, we have that the sets of predictors in the design matrix

$X = ( x _ { 1 } , x _ { 2 } , \dots , x _ { p } )$ are nested, so that $S \subseteq J$ for $| J | \geq | S |$ . For a fixed $\alpha > 0$ $n > | S | = d .$ and $| J | = p > n$ , we obtain

$$
\mathrm { t r } ( S _ { S } ) = \sum _ { j = 1 } ^ { d } { \frac { \lambda _ { j } } { \lambda _ { j } + \alpha } } \leq \sum _ { j = 1 } ^ { n } { \frac { \lambda _ { j } } { \lambda _ { j } + \alpha } } = \mathrm { t r } ( S _ { J } ) \leq \sum _ { j = 1 } ^ { d } { \frac { \lambda _ { j } } { \lambda _ { j } + \alpha } } + ( n - d ) { \frac { 1 } { \alpha } }
$$

since $\lambda _ { j } \geq 0$ . Then, for the diference between these two MDL parts we obtain

$$
( \operatorname { t r } ( S _ { S } ) - \operatorname { t r } ( S _ { J } ) ) \log n \leq - ( n - d ) { \frac { 1 } { \alpha } } \log n
$$

For the last part of MDL, we use a result by Cheema and Sugiyama (2020, Theorem 3)

$$
\mathbb { E } \left( \log \operatorname* { d e t } ( X ^ { \top } X + \alpha I ) + \log \mathbb { B } ^ { p } ( \| \beta \| _ { 2 } ) / \sigma ^ { p } \right) \leq \log d + \frac { n } { 2 } \log ( \alpha + 1 ) + \log \mathbb { B } ^ { p } ( 1 )
$$

Because $\mathbb { B } ^ { p } ( 1 ) \leq \mathbb { B } ^ { d } ( 1 )$ , where $d = | S | \leq | J | = p .$ , we obtain that

$$
\log { \frac { \mathbb { B } ^ { d } ( 1 ) } { \mathbb { B } ^ { p } ( 1 ) } } = \log \pi ^ { - ( p - d ) / 2 } \exp ( - ( p - d ) / 2 ) + O ( d / p )
$$

where we used Stirling’s approximation for factorials and the approximation of the Euclidean ball $\mathbb { B } ^ { p } ( 1 )$ given in (C.1).

Putting the three elements together gives the upper bound on the expectation

$$
\log \frac { \mathbb { E } \hat { \sigma } _ { S } ^ { 2 } } { \mathbb { E } \hat { \sigma } _ { J } ^ { 2 } } + \log n ^ { - ( n - d ) / \alpha } + \log \frac { d } { p } + \log \pi ^ { - ( p - d ) / 2 } \exp ( - ( p - d ) / 2 )
$$

Since the first term is bounded by $K / \gamma$ , we observe that the above is dominated by the last term, which goes to 0 as $p$ increases. In particular, we find that

$$
\pi ^ { - ( p - d ) / 2 } \exp ( - ( p - d ) / 2 ) \leq \frac { d } { p } \exp ( - ( p - d ) )
$$

Consequently, we obtain that

$$
\mathbb { E } \left( \exp [ \mathrm { M D L } ( S ) - \mathrm { M D L } ( J ) ] \right) \to 0
$$

And, therefore, as p increases

$$
\mathbb { P } ( \hat { S } \subseteq S ) = \mathbb { P } ( \operatorname* { m i n } _ { J } \mathrm { M D L } ( J ) \leq \mathrm { M D L } ( S ) : | J | \geq | S | ) \leq 1 - { \frac { d } { p } } \exp ( - ( p - d ) )
$$

as was to be shown.

## H Simulation details

In the case where the true model is linear we have $f ( X ) ~ = ~ X \beta$ . In the misspecified case we used instead of the linear model the sigmoid function $f ( X ) = 1 / ( 1 + \exp ( X \beta ) )$ . Again we keep $d = 5$ non-zero coeficients fixed. The coeficients $\beta$ were normalised so that the $L _ { 2 }$ norm was $1 , \mathrm { i . e . , } | | \beta | | _ { 2 } = 1$ The signal-to-noise ratio (SNR) is defined as $| | \beta | | _ { 2 } ^ { 2 } / \sigma ^ { 2 }$ , the ratio of the variance of the signal (coeficients) and the noise variance. Since we fix $| | \beta | | _ { 2 } ^ { 2 } = 1$ , the variance of the noise determines the SNR, set to 2, so that the noise variance is 0.5. The number of generated observations is fixed at $n = 4 0$

For estimation we selected a fraction of 0.80 for training (estimation) of parameters and the remaining 0.20 for testing. We fix $n = 4 0$ observations in total, and so obtain $n _ { \mathrm { t r a i n } } = 3 2$ and $n _ { \mathrm { t e s t } } = 8$ . We fix the dimension of the true model at $d = 5$ and vary the number of parameters $p$ from 3 to 320. Hence we obtain the ratio $p / n$ for training data from 0.09 to 10, where the last setting means we have ten parameters per observation. With fixed $d = 5$ we have that the correct ratio $p / n$ is at $5 / 3 2 = 0 . 1 5 6 2 5$

The ridge estimate is either obtained with parameter $\alpha = 0 . 5$ , or the ridge estimate is obtained with α estimated with 10-fold cross-validation, with the α with the smallest MSE. The Lasso is obtained with parameter α from 10-fold cross-validation. Estimation was performed with the R package glmnet.

## I Residuals in training and test samples

The interpolation point at $p = n$ is the point where it is possible that each datapoint is captured by the model in the training set, also in the linear case. We see this in Figure $\mathrm { I . 1 ( a ) }$ , where the residuals of the training (estimation) data as a function of the predicted values $\hat { Y }$ are plotted. Here, the true number of non-zero coeficients is 5 but in the overparameterised case we use $p = 6 4$ As seen, the training residuals are 0 and so $\begin{array} { r } { \hat { Y } _ { S } = \sum _ { j \in S } X _ { j } \beta } \end{array}$ is the same as $\begin{array} { r } { \hat { Y } _ { J } = \sum _ { j \in J } X _ { j } \hat { \beta } _ { J } } \end{array}$ , where $\hat { \beta } _ { J }$ is the ridge estimate for model J and model J contains all 64 predictors. Because the observations are predicted exactly, it was assumed that the variance of the residuals on a test set would be large. Figure I.1(b) shows that this is not the case. There is some variance in the residuals of the test set but not too much. The noise that ends up in the predictors is “hidden” in the unimportant directions of the predictors because of the ridge parameter (Bartlett et al., 2020).

![](images/3617a1d0b6fa190bd634994de205feec9affa8c084c36532b2a71c346f623437.jpg)  
Fig. H.1. Test MSE (left column) and its decomposition in squared bias and variance. The top row is obtained with a fixed value $\alpha = 0 . 0 0 0 1$ , the middle row is obtained with an $\alpha$ estimated using 10-fold cross-validation, and the bottom row is obtained with the lasso, where also the penalty parameter is obtained with 10-fold cross-validation.

![](images/fc7042570bd3f5cd985635b4b4c692219e760c64d8eab537bc4e9e164518ec11.jpg)  
(a)

![](images/9260095c1764c3da2ed1752758b8d58b2013847f624a7d70827614c2b98c79f7.jpg)  
(b)  
Fig. I.1. Residuals of the training set (a), showing that the model with 64 predictors (while correct is 5) is interpolating the data, fitting exactly each point. The residuals of the test set (b) remain relatively small, indicating that the variance need not explode beyond the interpolation point.