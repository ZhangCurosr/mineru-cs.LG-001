# Regime-Gated Residual Mixture-of-Experts for Cross-Sectional Volatility Forecasting

Junyi Ye School of Computing, Montclair State University Montclair, New Jersey, USA yej@montclair.edu

## Abstract

Financial volatility is regime dependent, yet incorporating regime information into neural networks can also destabilize training. This paper asks where such information should enter a neural crosssectional volatility forecasting model. We study five-day realizedvolatility forecasts for 1,027 U.S. equities using a rolling walkforward evaluation framework in which information, model capacity, hyperparameter tuning, and random seeds are matched across architectures. We propose RG-ResMoE, a regime-gated residual mixture-of-experts architecture in which regime information is used only for expert routing rather than for direct forecasting. The base predictor models volatility from stock features, while a gating network uses regime state variables to route residual corrections. RG-ResMoE consistently outperforms a capacity-matched MLP in both forecasting accuracy and training stability in the main U.S. study. Similar gains are observed on an independent Japanese panel. The integration pathway is decisive: appending the same regime variables directly to the forecasting input degrades both predictive performance and training stability, whereas restricting them to the routing gate improves accuracy and Value-at-Risk calibration. Hard routing consistently underperforms soft routing. The results suggest that, in compact neural volatility forecasting models, the primary value of mixture-of-experts models lies less in increasing model capacity than in controlling how nonstationary regime information influences prediction.

## CCS Concepts

• Computing methodologies → Neural networks; Ensemble methods; • Mathematics of computing → Time series analysis; • Applied computing → Economics.

## Keywords

mixture of experts, residual learning, gated routing, volatility forecasting, training stability

## 1 Introduction

Financial volatility forecasting is central to portfolio allocation, derivative pricing, and risk management. Deep neural networks have substantially improved volatility prediction by learning nonlinear relationships from large cross-sectional equity panels. However, financial markets are inherently nonstationary: the relationship between historical observations and future volatility changes across periods of calm, crisis, and recovery. Regime information is therefore a natural source of additional context for neural forecasting models. The central question, however, is not whether market regimes matter, but how such information should be incorporated into the forecasting process.

Gargi Vijay Borde School of Computing, Montclair State University Montclair, New Jersey, USA bordeg1@montclair.edu

Existing approaches introduce regime information in several ways. The simplest strategy appends regime variables directly to the model input [14, 21]. Regime-switching models instead maintain diferent predictors for diferent market states [8, 9], while mixtureof-experts (MoE) architectures learn observation-dependent routing through a gating network [10]. Although these approaches difer substantially, they typically change the regime representation, routing mechanism, and model architecture simultaneously. As a result, it remains unclear whether performance diferences arise from the regime information itselforfrom where that information enters the model. This distinction is particularly important in financial forecasting, where predictive gains are often small, optimization is sensitive to initialization, and additional model capacity alone does not necessarily improve generalization.

In this paper, we study this question through an informationmatched comparison in which the regime variables, forecasting backbone, parameter budget, hyperparameter selection, and evaluation protocol are all held fixed, while only the integration pathway is varied. We propose RG-ResMoE, a residual mixture-of-experts architecture in which a frozen base network first predicts the volatility level and small zero-initialized experts learn only residual corrections. A soft gate observes both stock features and regime variables to determine how these corrections are combined, while the forecasting networks themselves receive only stock features. This design separates two choices that are often confounded: whether regime information is available, and where it is allowed to influence the prediction.

Experiments on a rolling walk-forward evaluation covering 1,027 U.S. equities from 2018 to 2025 reveal three consistent findings. First, the location of regime information is decisive. Appending the same regime variables directly to the forecasting input consistently reduces forecasting accuracy and substantially increases training instability, whereas using them only for gated residual routing improves both accuracy and robustness. Second, continuous soft routing consistently outperforms hard routing based on learned assignments or hand-crafted market partitions, indicating that smooth state-dependent reweighting is more efective than discrete regime selection. Third, the gains are concentrated in the market conditions where volatility forecasts matter most, improving Value-at-Risk calibration and producing the largest forecasting improvements during periods of elevated market volatility. The same stability ordering is reproduced on an independent Japanese equity panel.

More broadly, our results suggest that the primary value of regime-aware MoE models in compact financial forecasting systems lies less in increasing model capacity than in controlling how nonstationary market information influences the prediction. Taken together, these findings establish a simple design principle: in compact neural volatility forecasters, regime information is most efective when it guides residual routing rather than being treated as ordinary predictive input. We believe this principle extends beyond the proposed RG-ResMoE architecture and provides guidance for incorporating market state information into future neural forecasting models.

## 2 Data and Task

## 2.1 Equity panels

Our main experiments use a U.S. equity panel, with a Japanese panel reserved for cross-market replication. The U.S. panel contains 1,027 common stocks from Yahoo Finance, observed from December 2015 to November 2025. The universe spans all eleven sectors and large-, mid-, and small-cap stocks. Among these stocks, 85% are current S&P 1500 constituents. The panel is restricted to stocks that remain listed through the end of the sample and have at least six years of price history. Among them, 92% cover the full ten-year period.

The replication panel contains 1,552 Japanese stocks from Yahoo Finance over the same period and covers the TSE Prime segment. It is constructed using the same data-processing pipeline and serves as a second-market replication of the main U.S. results.

## 2.2 Inputs and regime state variables

For each stock � on day �, the stock-level input $x _ { i , t }$ contains sixteen features computed from the stock’s own price history. These include realized volatility over the trailing 5, 20, and 60 days, cumulative returns over the trailing 5 and 20 days, the 14-day relative strength index (RSI), and the ten most recent daily log returns. For each walkforward window, normalization parameters are estimated from the training data and then applied to the validation and test data.

The regime state variable $z _ { i , t }$ consists of two volatility-based indicators: market volatility and idiosyncratic volatility. Market volatility is defined as the 20-day rolling volatility of the equal weighted market return $r _ { m , t }$ and is shared by all stocks on a given trading day. Idiosyncratic volatility is defined as the 20-day rolling volatility of the residual return $r _ { i , t } - \beta _ { i , t } r _ { m , t }$ , where $r _ { i , t }$ is the return of stock � on day $t , r _ { m , t }$ is the market return, and $\beta _ { i , t }$ is estimated from the preceding 120 trading days. Market volatility captures the overall level of systematic uncertainty, whereas idiosyncratic volatility measures stock-specific uncertainty after removing broad market movements. Together, these variables provide complementary information about the prevailing market regime. Across experiments, both state variables are fixed, and only their point of integration into the model is varied.

## 2.3 Task

The task is to predict cross-sectional stock volatility. For stock � on day �, the target is the annualized five-day forward realized volatility

$$
y _ { i , t } = \sqrt { 2 5 2 } \mathrm { ~ S t d } \left( r _ { i , t + 1 } , . . . , r _ { i , t + 5 } \right) ,\tag{1}
$$

where $r _ { i , t + k }$ denotes the daily log return of stock � on day $t + k$ Models generate predictions for all stocks on each trading day.

Evaluation follows the rolling walk-forward protocol illustrated in Table 1. Each evaluation window uses a 504-trading-day (two years) development period, with the first 85% used for model training and the final 15% used for validation, followed by a 63-tradingday (one quarter) test period. After each evaluation period, the full window advances by 63 trading days. This procedure produces 30 non-overlapping test windows from April 2018 to October 2025 and approximately 1.9 million out-of-sample forecasts. The evaluation period spans diverse market regimes, including the COVID-19 market crash in 2020, the subsequent recovery in 2021, the broad bear market of 2022, and comparatively calmer periods before and afterward.

Table 1: Rolling walk-forward evaluation protocol.
<table><tr><td>Window</td><td>Development period</td><td>Test period</td></tr><tr><td>1</td><td>2016-04- 2018-03</td><td>2018-04- 2018-06</td></tr><tr><td>2</td><td>2016-07 - 2018-06</td><td>2018-07 - 2018-09</td></tr><tr><td></td><td>*</td><td>•·</td></tr><tr><td></td><td></td><td></td></tr><tr><td>30</td><td>2023-08 - 2025-07</td><td>2025-08 - 2025-10</td></tr></table>

## 3 Model Architecture

All neural network models are built from the same two-hidden-layer MLP block, which serves as the common forecasting component across all architectures. The variants difer in three aspects: the use of regime state variables, the routing mechanism, and whether experts predict full volatility forecasts or residual corrections to a shared base forecast.

## 3.1 Shared MLP block

To keep the forecasting component consistent across models, every neural network is constructed from the same two-hidden-layer MLP block. Given an input vector $q ,$ the block is defined as

$$
\begin{array} { c } { h _ { 1 } = \phi ( W _ { 1 } q + b _ { 1 } ) , } \\ { h _ { 2 } = \phi ( W _ { 2 } h _ { 1 } + b _ { 2 } ) , } \\ { \mathrm { B l o c k } ( q ; \theta ) = W _ { 3 } h _ { 2 } + b _ { 3 } . } \end{array}\tag{2}
$$

Here, $\phi$ denotes the GELU activation. Both hidden layers have width �, and dropout with rate 0.1 is applied after each hidden layer during training. Let � denote the stock-level feature vector and � denote the vector ofregime state variables. The MLP baselines receive either the stock features alone, $q = x ,$ or the concatenated stock and state inputs, $q = ( x , z )$ . In MoE and RG-ResMoE, each forecasting block receives only �, while � can afect the prediction only through the gate.

## 3.2 Standard MoE

A standard mixture-of-experts (MoE) model combines the predictions of multiple experts, each implemented as an MLP block, through a learned gate. Intuitively, diferent experts can specialize in diferent market conditions or stock-level patterns, while the gate determines how much each expert contributes to the final prediction. Given the stock-level feature vector �, each expert produces a volatility forecast, and the gate combines these forecasts as

![](images/6e4bb4899e39b77a90469e25bebf2b06f9333a525613cda4577791f826e57bc9.jpg)  
Figure 1: Architectures of the baseline MLP-L(+�) (left) and proposed RG-ResMoE (right).

$$
\begin{array} { r } { e _ { k } ( x ) = \mathrm { B l o c k } ( x ; \theta _ { k } ) , } \\ { \pi = \mathrm { s o f t m a x } \big ( g ( u ) \big ) , } \end{array}\tag{3}
$$

$$
\hat { y } = \sum _ { k = 1 } ^ { K } \pi _ { k } e _ { k } ( \boldsymbol { x } ) .
$$

Here, $e _ { k } ( x )$ is the volatility forecast produced by the �th MLP expert, $\theta _ { k }$ denotes its parameters, � is the gate network, and $\pi _ { k }$ is the weight assigned to expert �. Each expert receives �, while the gate receives both the stock-level features and the volatility state variables, so $\boldsymbol { u } = ( x , z )$ . The gate therefore provides input-dependent conditional weighting. All experts and the gate are trained jointly.

## 3.3 Regime-gated Residual MoE (RG-ResMoE)

Unlike the standard MoE, in which each expert predicts the full volatility level, RG-ResMoE decomposes the prediction into a base network and expert-specific residual corrections, as illustrated in Figure 1. The base network captures common forecasting patterns, while the experts model conditional deviations from the base predic tion. This decomposition allows the experts to focus on specialized corrections rather than relearning the full forecasting function.

Given the stock-level feature vector �, the base network produces

$$
\hat { y } _ { \mathrm { b a s e } } = \mathrm { B l o c k } ( \boldsymbol { x } ; \boldsymbol { \theta } _ { b } ) ,\tag{4}
$$

where ${ \hat { y } } _ { \mathrm { b a s e } }$ is the base forecast. The �th expert produces a residual correction,

$$
r _ { k } ( x ) = \operatorname { B l o c k } ( x ; \theta _ { k } ) .\tag{5}
$$

The gate then combines the expert corrections as

$$
\begin{array} { c } { { \pi = \displaystyle \mathrm { s o f t m a x } \big ( g ( u ) \big ) , } } \\ { { \displaystyle \hat { y } _ { \mathrm { c o r r e c t i o n } } = \sum _ { k = 1 } ^ { K } \pi _ { k } r _ { k } ( x ) , } } \\ { { \hat { y } = \hat { y } _ { \mathrm { b a s e } } + \hat { y } _ { \mathrm { c o r r e c t i o n } } . } } \end{array}\tag{6}
$$

Here, $\theta _ { b }$ and $\theta _ { k }$ denote the parameters of the base network and the �th expert, respectively, and $\pi _ { k }$ is the weight assigned to expert �. The aggregated residual correction $\hat { y } _ { \mathrm { c o r r e c t i o n } }$ is the soft-weighted sum of the expert outputs. The base network and all experts receive �, while the gate receives $\boldsymbol { u } = ( x , z )$

Table 2: Models and architectural variants considered in the experiments. +� denotes direct input integration of regime variables, while -� removes the regime pathway from RG-ResMoE.

<table><tr><td>Category</td><td>Models</td></tr><tr><td>Main models</td><td>RG-ResMoE</td></tr><tr><td>Regime state variants</td><td>RG-ResMoE (-z); Ridge (+z); MLP-S (+z); MLP-L (+z)</td></tr><tr><td>Soft routing</td><td>Learned soft gate</td></tr><tr><td>Hard routing</td><td>Learned top-1 gate; volatility-quantile assign- ment; market×idiosyncratic volatility split; GICS sectors</td></tr><tr><td>Architecture variants</td><td>Standard MoE; random initialization; RG- ResMoE-K2; RG-ResMoE-K6</td></tr><tr><td>Baselines</td><td>Persistence; GARCH; HAR; Ridge; MLP-S; MLP-L</td></tr></table>

Training proceeds in two stages. The base network is trained first and then frozen. The � = 4 correction experts and the gate are then trained jointly by minimizing

$$
\mathcal { L } = \mathrm { M S E } ( \hat { y } , y ) + \alpha \overline { { \left( \sum _ { k = 1 } ^ { K } \pi _ { k } r _ { k } ( x ) \right) ^ { 2 } } } + \lambda _ { \mathrm { L B } } \sum _ { k = 1 } ^ { K } \left( \bar { \pi } _ { k } - \frac { 1 } { K } \right) ^ { 2 } .\tag{7}
$$

Here, the bar denotes an average over the full training batch. The first term is the prediction loss. The second term penalizes the mag nitude of the aggregate residual correction, encouraging the final prediction to remain close to the frozen base forecast ${ \hat { y } } _ { \mathrm { b a s e } } ( x )$ . The coeficient � controls the strength of this shrinkage. The third term is a load-balancing penalty that discourages the routing weights from concentrating on a small number of experts. The coeficient $\lambda _ { \mathrm { L B } }$ controls how strongly the routing weights are encouraged toward the uniform allocation $1 / K .$ Together, these regularizers keep the expert corrections complementary to the base forecast and reduce the risk of routing collapse. Each correction expert uses a zero-initialized final layer, so $r _ { k } ( x ) = 0$ at initialization and the model initially reproduces the base forecast ${ \hat { y } } _ { \mathrm { b a s e } } ( x )$

## 3.4 Routing strategies

The routing mechanism determines how expert predictions are selected and combined. We consider both soft and hard routing. In soft routing, a gate network produces a probability distribution over the experts, � = softmax(�(�)), and the final prediction is formed as a weighted combination of all expert outputs. This allows multiple experts to contribute to each forecast, with their weights varying continuously across observations.

In hard routing, each observation is assigned to a single expert. The selected expert may be determined either by a learned top-1 gate or by predefined rules based on market characteristics, such as volatility regimes or sector membership. The prediction is produced solely by the selected expert, without averaging across experts. Both routing strategies use the same expert architecture and training budget. They difer only in how expert specialization is enforced, allowing us to isolate the efect of routing independently of model capacity.

## 3.5 Comparison set

Table 2 summarizes the models evaluated in this study. The comparison includes classical statistical baselines, MLP architectures of diferent capacities, MoE and the proposed RG-ResMoE family. Beyond the main RG-ResMoE model, the variants isolate three design dimensions: the use of regime state variables, the routing mechanism, and the RG-ResMoE architecture itself. Since all neural models share the same forecasting block, these comparisons isolate the efect of each design choice while keeping the underlying predictor comparable.

## 4 Evaluation Protocol

## 4.1 Training and hyperparameter tuning

All neural network models use the same training procedure: fullbatch Adam optimization of mean-squared error (MSE), early stopping based on validation loss, and restoration of the best checkpoint. Each reported configuration is trained with 30 random seeds, while hyperparameter sweeps use three seeds. This design allows us to evaluate both average performance and sensitivity to random initialization under a fixed training budget.

Hyperparameters are selected using validation information coeficient (IC), defined as the cross-sectional correlation between predicted and realized volatility. Candidate configurations are filtered using three robustness criteria. The selected configuration (i) lies in the interior of the search grid rather than at its boundary, (ii) achieves a mean validation IC exceeding that of the runner-up by more than its across-seed standard deviation, and (iii) has a validation MSE within one standard deviation of the lowest MSE observed in the sweep.

## 4.2 Baselines

The comparison set includes three classical forecasting baselines: 20-day persistence, pooled HAR [4], and per-stock GARCH(1,1) [3]. Ridge provides a linear baseline using the same stock-level features as the neural networks. MLP-S (small) uses a single MLP block with hidden width �=16, matching an individual block in RG-ResMoE. MLP-L (large) uses the same block structure with a larger hidden width �=44 to match the overall forecasting capacity of the full RG-ResMoE model. To distinguish the efect of using the regime variables as predictor inputs from their use in expert routing, we additionally evaluate Ridge(+�), MLP-S(+�), and MLP-L(+�). These variants append the two regime state variables to the ordinary model inputs and are re-tuned from scratch.

## 4.3 Evaluation metrics

Our primary accuracy metrics are IC, RMSE, and out-of-sample $R ^ { 2 } .$ . We additionally report ICIR, QLIKE, VaR calibration, and the frequency of collapsed seeds to assess ranking consistency, volatility calibration, and training stability.

Ranking performance: Daily cross-sectional Spearman information coeficient (IC) between predicted and realized volatility, averaged over test days, and information coeficient information ratio (ICIR), defined as the mean daily IC divided by its time-series standard deviation.

Level accuracy: Root mean-squared error (RMSE) and out-ofsample $R ^ { 2 } ,$ pooled across all stock-days and computed on the annualized volatility scale.

Quasi-likelihood loss (QLIKE): The volatility loss

$$
\mathrm { Q L I K E } _ { i , t } = \frac { \sigma _ { i , t } ^ { 2 } } { \hat { \sigma } _ { i , t } ^ { 2 } } - \ln \left( \frac { \sigma _ { i , t } ^ { 2 } } { \hat { \sigma } _ { i , t } ^ { 2 } } \right) - 1 ,\tag{8}
$$

averaged across all stock-days [17]. Lower values indicate better variance forecasts.

Value at Risk (VaR) coverage: The fraction oftickers for which the null of correct unconditional coverage is rejected by Kupiec’s test [13] at the 5% significance level. We evaluate both 5% and 1% VaR, using Student-� residual distributions estimated from the corresponding training window.

Training stability: The number of collapsed seeds, defined using a prespecified diagnostic threshold as runs whose mean test QLIKE exceeds 2.0.

## 4.4 Statistical testing

Pairwise Diebold–Mariano-style tests [5] assess whether the forecast performance of two competing models difers significantly. For each test day, we compute the diference in the selected metric between models � and � using predictions on the same stocks, then average this diference across the � random seeds:

$$
d _ { t } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \big [ X _ { A , s } ( t ) - X _ { B , s } ( t ) \big ] .\tag{9}
$$

The resulting series of approximately 1,900 daily diferences is tested using Newey–West standard errors [16] with four lags to account for dependence induced by the overlapping five-day targets. We report the resulting �-values alongside the model performance metrics.

## 5 Main Results

Table 3 summarizes the main forecasting results. Across all pooled models, RG-ResMoE achieves the best overall forecast performance. It obtains the highest IC, lowest RMSE, highest out-of-sample $R ^ { 2 }$ highest ICIR, and lowest QLIKE. Most diferences relative to the competing neural networks are statistically significant.

Table 3: Main forecasting results on the U.S. equity panel. Persistence and GARCH are estimated per stock. All other models are pooled across stocks. Neural network results report mean ± standard deviation over 30 seeds. Shading indicates statistically significant paired diferences from RG-ResMoE: �<0.001 , �<0.01 , and $\scriptstyle { p < 0 . 0 5 }$
<table><tr><td rowspan="2">Training</td><td rowspan="2">Model</td><td colspan="5">Forecast performance</td><td>Stability</td></tr><tr><td>IC↑</td><td>RMSE↓</td><td>R2 ↑</td><td>ICIR ↑</td><td>QLIKE↓</td><td>Collapse ↓</td></tr><tr><td rowspan="2">Per-stock</td><td>Persistence</td><td>0.5018</td><td>0.2570</td><td>0.121</td><td>5.18</td><td>0.799</td><td>一</td></tr><tr><td>GARCH(1,1)</td><td>0.5200</td><td>0.2642</td><td>0.070</td><td>5.48</td><td>1.120</td><td>一</td></tr><tr><td rowspan="6">Pooled</td><td>HAR</td><td>0.5315</td><td>0.2347</td><td>0.266</td><td>5.93</td><td>0.749</td><td>一</td></tr><tr><td>Ridge</td><td>0.5266</td><td>0.2336</td><td>0.273</td><td>5.75</td><td>0.760</td><td></td></tr><tr><td>MLP-S</td><td> $0 . 5 3 9 3 \pm 0 . 0 0 3 4$ </td><td> $0 . 2 3 2 3 \pm 0 . 0 0 1 1$ </td><td> $0 . 2 8 0 \pm 0 . 0 0 7$ </td><td> $6 . 0 0 \pm 0 . 1 5$ </td><td> $7 4 . 3 \pm 3 9 5 . 1$ </td><td>2/30</td></tr><tr><td>MLP-L</td><td> $0 . 5 4 2 1 \pm 0 . 0 0 3 3$ </td><td> $0 . 2 3 2 0 \pm 0 . 0 0 0 9$ </td><td> $0 . 2 8 2 \pm 0 . 0 0 5$ </td><td> $6 . 0 8 \pm 0 . 1 8$ </td><td> $2 . 2 3 \pm 5 . 1 5$ </td><td>3/30</td></tr><tr><td>Standard MoE</td><td> $0 . 5 4 1 3 \pm 0 . 0 0 2 7$ </td><td> $0 . 2 3 5 7 \pm 0 . 0 0 4 9$ </td><td> $0 . 2 5 9 \pm 0 . 0 3 1$ </td><td> $6 . 0 7 \pm 0 . 1 4$ </td><td> $8 , 7 7 5 \pm 1 8 , 2 7 7$ </td><td>24/30</td></tr><tr><td>RG-ResMoE (ours)</td><td> $\mathbf { 0 . 5 4 6 9 } \pm 0 . 0 0 1 2$ </td><td> $\mathbf { 0 . 2 3 0 4 } \pm 0 . 0 0 1 3$ </td><td> $\mathbf { 0 . 2 9 2 \pm 0 . 0 0 8 }$ </td><td> $6 . 1 4 \pm 0 . 0 6$ </td><td> $\mathbf { 0 . 7 3 5 \pm 0 . 0 1 7 }$ </td><td>0/30</td></tr></table>

Table 4: Efect of regime-information integration pathway.
<table><tr><td>Model</td><td>Pathway</td><td>IC↑</td><td>ΔIC ↑</td><td>Collapse ↓</td></tr><tr><td>Ridge</td><td></td><td>0.5266</td><td>ref.</td><td>一</td></tr><tr><td>Ridge (+z)</td><td>input</td><td>0.5243</td><td>-0.0023</td><td>一</td></tr><tr><td>MLP-S</td><td>1</td><td> $0 . 5 3 9 3 \pm 0 . 0 0 3 4$ </td><td>ref.</td><td>2/30</td></tr><tr><td>MLP-S (+z)</td><td>input</td><td> $0 . 5 3 8 3 \pm 0 . 0 0 4 1$ </td><td>-0.0011</td><td>22/30</td></tr><tr><td>MLP-L</td><td></td><td> $0 . 5 4 2 1 \pm 0 . 0 0 3 3$ </td><td>ref.</td><td>3/30</td></tr><tr><td>MLP-L (+z)</td><td>input</td><td> $0 . 5 3 7 8 \pm 0 . 0 0 3 5$ </td><td>-0.0043</td><td>30/30</td></tr><tr><td>RG-ResMoE (-z)</td><td>一</td><td> $0 . 5 4 6 6 \pm 0 . 0 0 1 1$ </td><td>ref.</td><td>1/30</td></tr><tr><td>RG-ResMoE</td><td>gate</td><td> $\mathbf { 0 . 5 4 6 9 } \pm 0 . 0 0 1 2$ </td><td>+0.0004</td><td>0/30</td></tr></table>

Compared with the classical baselines, all pooled neural network models improve forecast accuracy substantially, demonstrating the benefit of learning shared representations across stocks. Within the pooled models, however, simply increasing model capacity provides only modest gains. Although MLP-L has approximately the same overall capacity as RG-ResMoE, it remains consistently worse across every forecast metric. This indicates that the improvement comes primarily from the residual expert architecture rather than from additional parameters.

The optimization results further support this conclusion. Standard MoE shows frequent optimization failures, collapsing in 24 of the 30 random seeds, whereas RG-ResMoE completes all runs successfully without a single collapse. The residual decomposition improves not only forecast accuracy but also training stability.

## 6 Where the Gain Comes From

## 6.1 Where should regime information enter the model?

RG-ResMoE uses the two regime state variables only for expert routing, rather than treating them as ordinary forecasting inputs. This design raises a natural question: does the improvement come from the regime information itself, or from how that information is incorporated into the model?

Input integration versus gate-based integration. Table 4 compares two ways of using the same state variables. In the input pathway, the state variables are appended to the stock-level features used by Ridge, MLP-S, and MLP-L, allowing them to directly influence the forecast. In the gate pathway, the forecasting networks continue to use only the stock-level features, while the state variables are used exclusively by the RG-ResMoE gate to determine the mixture of residual experts.

![](images/95d3ee4ab69f49c8995526f77dd447cb8dfd838b85a0218bff486ef520cfc193.jpg)  
Figure 2: Per-window comparison of input and gate integration across the 30 walk-forward windows. Lines show seedaveraged paired diferences within each evaluation window. Positive values indicate improvement for ΔIC and deterioration for ΔRMSE.

The two integration strategies lead to markedly diferent outcomes. Appending the state variables to the forecasting inputs consistently reduces IC and substantially increases training instability for the MLP baselines. In contrast, using the same variables only for expert routing produces a small improvement in IC while eliminating the remaining collapsed run. These results indicate that the value of regime information lies not in expanding the predictor input, but in adapting how residual corrections are combined. Gate-based integration allows the state variables to modulate the expert weights without directly shifting the forecast produced by the underlying forecasting networks.

Behavior across market regimes. The aggregate results in Table 4 may conceal variation across market conditions. Figure 2 therefore reports window-level diferences between models with and without regime state variables. For the input pathway, the diference is computed as MLP-L(+�) minus MLP-L; for the gate pathway, it is computed as RG-ResMoE minus RG-ResMoE(-�). Positive values favor the model with regime variables for IC, whereas negative values favor it for RMSE. Positive values indicate improvement for ΔIC and deterioration for ΔRMSE.

![](images/d70cf9303ad67b2b7ec876a3e9dc5d157aae6c1f110a7fc6fa3bab90f8ab340c.jpg)

(b) Regime loading  
![](images/49335ccea5b268d1486881dfbce84f51402eabe74bba1276d8d93f126b4ed712.jpg)  
Figure 3: Gate behavior of RG-ResMoE across 30 random seeds. (a) Distribution of the maximum routing weight assigned to any expert for each observation. Uniform routing corresponds to 0.25 and hard routing to 1. (b) Correlations between expert weights and the two regime variables after within-seed sorting. Whiskers denote across-seed standard deviations.

The largest diference between the two pathways appears around the transition from the COVID shock to the subsequent market recovery. During the stress period, the input pathway briefly improves IC, but this gain reverses during the recovery phase. RMSE also increases substantially during the stress period before moving back toward zero.

The gate pathway exhibits much smaller changes over the same period and does not show the pronounced IC reversal observed under direct input integration. This comparison suggests that the efect of using regime variables as forecasting inputs is highly dependent on market conditions. Using the same information through the gate instead produces a more stable response during major regime transitions.

## 6.2 What does the gate learn?

The learned routing remains broadly distributed across experts rather than collapsing to a single component (Figure 3a). The average maximum gate weight remains close to uniform routing, and only a small fraction of observations are assigned almost entirely to one expert. At the same time, expert weights show substantial sensitivity to both market and idiosyncratic volatility, with the most strongly loaded experts typically exhibiting opposite-signed correlations (Figure 3b). Together, these patterns indicate that the gate performs continuous state-dependent reweighting rather than imposing discrete regime assignments.

Table 5: Comparison of soft and hard routing. All variants use the same frozen base and four residual experts. ΔIC and �-values are reported relative to RG-ResMoE.
<table><tr><td>Variant</td><td>Routing</td><td>IC↑</td><td>∆IC↑</td><td>p</td><td>Collapse ↓</td></tr><tr><td>RG-ResMoE</td><td>soft</td><td> $\mathbf { 0 . 5 4 6 9 \pm 0 . 0 0 1 2 }$ </td><td>ref.</td><td></td><td>0/30</td></tr><tr><td>learned top-1</td><td>hard</td><td> $0 . 5 4 4 8 \pm 0 . 0 0 1 8$ </td><td> $- 0 . 0 0 2 1 < 1 0 ^ { - 4 }$ </td><td></td><td>1/30</td></tr><tr><td>volatility quantiles hard</td><td></td><td> $0 . 5 4 4 7 \pm 0 . 0 0 1 5$ </td><td> $- 0 . 0 0 2 2 \ < \ 1 0 ^ { - 4 }$ </td><td></td><td>1/30</td></tr><tr><td>GICS sectors</td><td>hard</td><td> $0 . 5 4 4 5 \pm 0 . 0 0 1 6$ </td><td> $- 0 . 0 0 2 4 \ < \ 1 0 ^ { - 4 }$ </td><td></td><td>1/30</td></tr><tr><td>market × idio split hard</td><td></td><td> $0 . 5 4 3 5 \pm 0 . 0 0 1 6$ </td><td> $- 0 . 0 0 3 4 \ < \ 1 0 ^ { - 4 }$ </td><td></td><td>1/30</td></tr></table>

Table 6: Ablations on expert count and residual design. ΔIC is relative to the main RG-ResMoE model.
<table><tr><td>Variant</td><td>IC↑</td><td>∆IC ↑</td><td>Collapse ↓</td></tr><tr><td>RG-ResMoE-k4 (main)</td><td> $\mathbf { 0 . 5 4 6 9 \pm 0 . 0 0 1 2 }$ </td><td>ref.</td><td>0/30</td></tr><tr><td>RG-ResMoE-K2</td><td> $0 . 5 4 6 6 \pm 0 . 0 0 1 4$ </td><td> $- 0 . 0 0 0 4$ </td><td>0/30</td></tr><tr><td>RG-ResMoE-K6</td><td> $0 . 5 4 6 7 \pm 0 . 0 0 1 2$ </td><td> $- 0 . 0 0 0 2$ </td><td>3/30</td></tr><tr><td>RG-ResMoE (random init)</td><td> $0 . 5 4 3 0 \pm 0 . 0 0 2 1$ </td><td> $- 0 . 0 0 3 9$ </td><td>13/30</td></tr><tr><td>Standard MoE</td><td> $0 . 5 4 1 3 \pm 0 . 0 0 2 7$ </td><td> $- 0 . 0 0 5 6$ </td><td>24/30</td></tr></table>

## 6.3 Is soft routing better than hard routing?

The previous analysis shows that the gate learns a continuous redistribution of expert weights. An immediate question is whether this continuous routing is necessary, or whether comparable performance can be achieved using discrete expert assignments.

Table 5 compares the learned soft gate with four hard-routing alternatives while keeping the residual architecture and the number of experts fixed. The hard variants include a learned top-1 gate, volatility quantile assignment, GICS sectors, and a market-volatility × idiosyncratic-volatility split. All hard-routing variants underperform the soft gate. Their IC ranges from 0.5435 to 0.5448, corresponding to reductions of 0.0021–0.0034 relative to RG-ResMoE, with every comparison significant at $ { p ^ { \mathrm { ~ < ~ } 1 0 ^ { - 4 } } }$ . The performance degradation is consistent across all four routing strategies rather than being attributable to a particular hand-crafted partition. These results indicate that the improvement does not arise from a better definition of market regimes. Instead, allowing multiple experts to contribute simultaneously is consistently more efective than assigning each observation to a single expert.

## 6.4 How many experts are needed?

Two aspects of the expert architecture are examined in Table 6: the number of experts and the residual parameterization. To isolate these efects, the expert-count variants maintain approximately the same overall model capacity by widening individual experts when fewer are used and narrowing them when more experts are introduced.

![](images/c976fe605767cedb2d85abd0d50495ca220c24498cb0d15ac295cd5d61e85d2f.jpg)  
Figure 4: Training stability across random initializations. Each point shows the mean test QLIKE for one training seed. The dashed line marks the collapse threshold (QLIKE = 2), and the numbers on the right report the number of collapsed runs.

The number of experts has only a modest efect on overall forecasting accuracy. Reducing the architecture from four experts to two lowers IC by only 0.0004, indicating that most of the benefit is retained with a smaller mixture. Increasing to six experts provides no further improvement, while substantially reducing training stability. Although the average IC remains similar to the four-expert model, three of thirty runs collapse and QLIKE becomes considerably more variable.

The residual parameterization is substantially more important. Replacing the zero-initialized residual layer with random initialization lowers IC and increases the number of collapsed runs from 0 to 13. Removing the frozen base forecast entirely has an even larger efect. The resulting standard MoE exhibits the lowest IC among the neural mixture models and collapses in 24 of 30 runs.

These results indicate that the performance gain does not arise from a large ensemble of experts. Instead, the principal contribution comes from the residual architecture, which anchors expert learning around a shared base forecast while allowing gated residual corrections to model state-dependent deviations.

## 7 Practical Evaluation

Average forecast accuracy does not fully characterize the practical value of a forecasting model. We therefore evaluate RG-ResMoE from three deployment-oriented perspectives: training reliability, risk calibration, and performance during periods of market stress.

Training reliability. Figure 4 summarizes training reliability. RG-ResMoE and RG-ResMoE-K2 complete all 30 runs without collapse, while RG-ResMoE-K6 and MLP-L each collapse in 3 of 30 runs. Random initialization and standard MoE are substantially less stable, with 13 and 24 collapsed runs, respectively, confirming that the residual design and frozen base are the primary sources of robustness.

Risk calibration. Besides producing accurate forecasts, a volatility model should also provide well-calibrated estimates of downside risk. We therefore convert each five-day volatility forecast into oneday Value-at-Risk (VaR) thresholds, which estimate the loss level expected to be exceeded only with a specified probability (e.g., 5% or 1%).

Table 7: Kupiec VaR coverage rejection rates averaged over 30 seeds. Lower values indicate better calibration. �-values are from paired tests with RG-ResMoE as the reference. n.s. denotes not significant.
<table><tr><td>Slice</td><td>RG-ResMoE</td><td>MLP-L</td><td>MLP-S</td></tr><tr><td>5% VaR, all dates</td><td>26.7%</td><td>30.3%  $( p < 1 0 ^ { - 4 } )$ </td><td> $3 2 . 7 \% ( p \mathrm { = } . 0 0 0 7 )$ </td></tr><tr><td>5% VaR, high-vol</td><td>11.9%</td><td>14.8%  $( p < 1 0 ^ { - 4 } )$ </td><td> $1 6 . 9 \% \left( p \mathrm { = } . 0 0 0 1 \right)$ </td></tr><tr><td>5% VaR, low-vol</td><td>22.9%</td><td>24.7% (p=.003)</td><td> $2 8 . 2 \% ( p = . 0 0 2 )$ </td></tr><tr><td>1% VaR, all dates</td><td>9.1%</td><td> $9 . 9 \% ( p \mathrm { = } . 0 0 1 )$ </td><td> $1 1 . 4 \% \left( p \mathrm { = } . 0 0 0 1 \right)$ </td></tr><tr><td>1% VaR, high-vol</td><td>6.2%</td><td> $7 . 1 \% ( p < 1 0 ^ { - 4 } )$ </td><td> $7 . 2 \% ( p < 1 0 ^ { - 5 } )$ </td></tr><tr><td>1% VaR, low-vol</td><td>5.8%</td><td>5.8% (n.s.)</td><td> $6 . 8 \% ( p \mathrm { = } . 0 5 3 , \mathrm { n } . s . )$ </td></tr></table>

Table 8: RG-ResMoE’s IC advantage over MLP-L across market conditions. Relative gain is the ratio of each subset’s ΔIC to the full-sample ΔIC.
<table><tr><td>Slice</td><td>Days</td><td>∆IC ↑</td><td>Rel. gain ↑</td><td>p</td></tr><tr><td>Full sample</td><td>1,890</td><td>+0.0048</td><td>ref.</td><td> $< 1 0 ^ { - 4 }$ </td></tr><tr><td>Top market-vol decile</td><td>189</td><td>+0.0207</td><td>4.3×</td><td> $< 1 0 ^ { - 4 }$ </td></tr><tr><td>COVID crisis (2020-02–06)</td><td>104</td><td>+0.0322</td><td>6.7×</td><td> $< 1 0 ^ { - 4 }$ </td></tr><tr><td>2022 bear market</td><td>251</td><td>+0.0079</td><td>1.6×</td><td> $< 1 0 ^ { - 4 }$ </td></tr><tr><td>Regime-flip windows (±10d)</td><td>826</td><td>+0.0031</td><td>0.6×</td><td> $< 1 0 ^ { - 4 }$ </td></tr></table>

Table 7 reports the fraction of tickers rejected by Kupiec’s coverage test, where lower values indicate better calibrated VaR estimates. RG-ResMoE consistently achieves the lowest rejection rates across nearly all settings, significantly outperforming both MLP baselines in five of the six regime–VaR combinations. The largest improvements occur during high-volatility periods, when accurate tail-risk estimation is most critical for risk management. The only exception is the 1% low-volatility slice, where all models perform similarly and the diferences are not statistically significant.

Performance under market stress. We finally examine where RG-ResMoE’s ranking advantage is concentrated, particularly during periods of elevated market volatility. For each market condition, Table 8 computes RG-ResMoE’s IC advantage over MLP-L using only the dates in that subset.

The gain is not uniform over time. RG-ResMoE outperforms MLP-L by +0.0048 IC in the full sample, but by +0.0207 IC in the highest-volatility decile and +0.0322 IC during the COVID crisis, corresponding to approximately 4.3× and 6.7× the full-sample gain. The 2022 bear market also amplifies the advantage, although less strongly. By contrast, regime-transition windows do not show a similar amplification. These results suggest that the benefit of residual routing is greatest during sustained periods of elevated market volatility rather than during regime transitions themselves.

## 8 Cross-Market Replication

We evaluate generalization beyond the U.S. market using the Japanese TSE Prime panel. Table 9 reports the core comparison under the same features, walk-forward protocol, hyperparameter selection rules, and thirty random seeds. RG-ResMoE outperforms MLP-L by +0.0043 IC (IC and RMSE $p < 1 0 ^ { - 4 } ; \mathrm { Q L I K E } ~ p = 0 . 0 0 2 )$ . Input concatenation is again unstable. It leaves IC unchanged while worsening RMSE and collapsing in 28 of 30 MLP-L (+�) seeds. The same stability ordering is reproduced on the Japanese market, suggesting that the benefit of residual routing is not market-specific.

Table 9: Replication on the Japanese TSE Prime market.
<table><tr><td>Model</td><td>IC↑</td><td>RMSE↓</td><td>QLIKE↓</td><td>Collapse ↓</td></tr><tr><td>MLP-L</td><td> $0 . 4 8 1 5 \pm 0 . 0 0 3 6$ </td><td> $0 . 1 8 5 1 \pm 0 . 0 0 0 7$ </td><td> $1 1 . 6 \pm 3 4 . 2$ </td><td>6/30</td></tr><tr><td>MLP-L (+z)</td><td> $0 . 4 8 1 5 \pm 0 . 0 0 2 6$ </td><td> $0 . 1 9 0 1 \pm 0 . 0 0 2 5$ </td><td> $4 1 , 2 5 6 \pm 4 0 , 2 9 6$ </td><td>28/30</td></tr><tr><td>MoE</td><td> $0 . 4 8 0 5 \pm 0 . 0 0 2 9$ </td><td> $0 . 1 8 6 2 \pm 0 . 0 0 1 2$ </td><td> $^ { 3 , 0 8 1 \pm 7 , 2 2 7 }$ </td><td>22/30</td></tr><tr><td> $\mathbf { R G - R e s M o E }$ </td><td> $\mathbf { 0 . 4 8 5 8 \pm 0 . 0 0 2 4 }$ </td><td> $\mathbf { 0 . 1 8 4 6 \pm 0 . 0 0 0 4 }$ </td><td> $\mathbf { 0 . 8 0 \pm 0 . 3 3 }$ </td><td>1/30</td></tr></table>

## 9 Related Work

Regime switching in finance. Markov-switching models made regime dependence central to macroeconomic and volatility mod eling [7–9]. Recent neural forecasting methods extend this idea through regime-aware architectures and expert routing [2, 10, 22]. These approaches demonstrate the value of incorporating market regimes, but typically modify the regime representation, routing strategy, and model capacity simultaneously. As a result, the contribution of the integration point remains unclear.

Mixture-of-experts. MoE models originated with Jacobs et al. [12] and have become a standard architecture for scalable conditional computation [6, 19], including recent time-series foundation models [15, 20]. Modern sparse MoE architectures primarily use hard top-� routing to increase model capacity while keeping the computation per input nearly constant, with expert specialization emerging through conditional activation. Soft expert combinations have also been studied to improve optimization and training stability [18]. However, most of this literature treats routing as a mechanism for computational scaling rather than as a modeling choice for incorporating market state information.

Residual learning and stable initialization. Residual networks improve optimization by learning corrections rather than complete transformations [11]. Zero- or near-zero-initialized resid ual branches further stabilize training [23], while warm-starting methods show that remaining close to a stable initialization can improve generalization [1]. Although these principles are well established in deep learning, their combination with gated expert routing has received comparatively little attention, particularly in financial forecasting.

## 10 Conclusion and Limitations

This paper studied how regime information should be incorporated into a neural network for volatility forecasting. The answer is not simply to add regime variables or adopt an MoE architecture. In our setting, the same state variables reduce performance when concatenated to the forecasting network but improve it when used only to gate residual correction experts. Soft residual correction on top of a frozen base forecast consistently outperforms hard routing, richer hand-crafted regime definitions, additional experts, and a standard MoE without a frozen base.

The broader implication is that nonstationary market information is most useful for modulating the forecasting process rather than directly determining the forecast itself. More generally, the results suggest that the value of MoE in compact forecasting models lies less in increasing model capacity than in controlling how auxiliary state information influences the prediction.

This study is limited to volatility forecasting in two equity markets using a compact MLP backbone. Whether the same design principle extends to other financial forecasting tasks, larger sequence models, or alternative forms of market state remains an important direction for future work.

## References

[1] Jordan T Ash and Ryan P Adams. 2020. On warm-starting neural network training. NeurIPS 33 (2020), 3884–3894.

[2] Melike Bildirici and Özgür Ersin. 2014. Modeling Markov switching ARMA-GARCH neural networks models and an application to forecasting stock returns. The Scientific World Journal 2014, 1 (2014), 497941.

[3] Tim Bollerslev. 1986. Generalized autoregressive conditional heteroskedasticity. Journal ofEconometrics 31, 3 (1986), 307–327.

[4] Fulvio Corsi. 2009. A simple approximate long-memory model of realized volatil ity. Journal ofFinancial Econometrics 7, 2 (2009), 174–196.

[5] Francis X Diebold and Roberto S Mariano. 1995. Comparing predictive accuracy. Journal of Business and Economic Statistics 13, 3 (1995), 253–263.

[6] William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Eficient Sparsity. Journal ofMachine Learning Research 23, 120 (2022), 1–39.

[7] Stephen F Gray. 1996. Modeling the conditional distribution of interest rates as a regime-switching process. Journal of financial economics 42, 1 (1996), 27–62.

[8] James D Hamilton. 1989. A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica: Journal of the econometric society (1989), 357–384.

[9] James D Hamilton and Raul Susmel. 1994. Autoregressive conditional heteroskedasticity and changes in regime. Journal of Econometrics 64, 1-2 (1994), 307–333.

[10] Cheng He, Zhenyu Guan, Xijie Liang, Defu Lian, Jiajia Li, Enhong Chen, Patrick PC Lee, Geng Hu, and Zehao Chen. 2026. RAVEN: A Regime-Aware Variable-context Expert Network for Financial Time Series Forecasting. arXiv preprint arXiv:2606.24062 (2026).

[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition. 770–778.

[12] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geofrey E Hinton. 1991. Adaptive mixtures of local experts. Neural computation 3, 1 (1991), 79–87.

[13] Paul H. Kupiec. 1995. Techniques for Verifying the Accuracy ofRisk Measurement Models. The Journal ofDerivatives 3, 2 (1995), 73–84. doi:10.3905/jod.1995.407942

[14] Sangheon Lee and Poongjin Cho. 2025. Graph-based stock volatility forecasting with efective transfer entropy and Hurst-based regime adaptation. Fractal and Fractional 9, 6 (2025), 339.

[15] Xu Liu, Juncheng Liu, Gerald Woo, Taha Aksu, Yuxuan Liang, Roger Zimmermann, Chenghao Liu, Junnan Li, Silvio Savarese, Caiming Xiong, et al. 2025. Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts. In International Conference on Machine Learning. PMLR, 38940–38962.

[16] Whitney K Newey and Kenneth D West. 1987. A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. Econometrica: Journal ofthe Econometric Society (1987), 703–708.

[17] Andrew J Patton. 2011. Volatility forecast comparison using imperfect volatility proxies. Journal ofEconometrics 160, 1 (2011), 246–256.

[18] Joan Puigcerver, Carlos Riquelme Ruiz, Basil Mustafa, and Neil Houlsby. 2024. From sparse to soft mixtures of experts. In International Conference on Learning Representations, Vol. 2024. 28435–28445.

[19] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geofrey Hinton, and Jef Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538 (2017).

[20] Xiaoming Shi, Shiyu Wang, Yuqi Nie, Dianqi Li, Zhou Ye, Qingsong Wen, and Ming Jin. 2025. Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts. In International Conference on Learning Representations. https: //openreview.net/forum?id=e1wDDFmlVu

[21] Chunxia Tian, Roengchai Tansuchat, and Songsak Sriboonchitta. 2026. Regime-Aware Stock Index Forecasting Under Latent Market States: A Hybrid Statistical Learning Framework with Cross-Market Validation. Forecasting 8, 3 (2026), 50.

[22] Zhaojian Yu, Yinghao Wu, Genesis Wang, and Heming Weng. 2024. MIGA: Mixture-of-Experts with Group Aggregation for Stock Market Prediction. arXiv preprint arXiv:2410.02241 (2024).

[23] Hongyi Zhang, Yann N Dauphin, and Tengyu Ma. 2019. Fixup initialization: Residual learning without normalization. arXiv preprint arXiv:1901.09321 (2019).