# Do Time-Series Forecasters Use the Right History: Recoverability, Recovery, and Functional Use of Temporal Delays

Qipeng Qian<sup>1,2</sup> Yuntao Qian<sup>2</sup>

<sup>1</sup>Supcon Technology, Hangzhou, China

<sup>2</sup>College of Artificial Intelligence, Zhejiang University, Hangzhou 310027, China qianqipeng@supcon.com ytqian@zju.edu.cn

## Abstract

Forecast accuracy does not tell us which past inputs produced a prediction. We separate three questions for time-series models with known delay structure: can the true delay be recovered from the observed data, does the model report it, and does the forecast actually use the same history? We first derive input-conditioned recoverability measures that separate intrinsic ambiguity from model error. We then prove that a delay report can become arbitrarily reliable while forecast risk approaches the oracle even though the predictor still uses the wrong lag. This failure also appears in finite samples on the point-delay task: among forecasts with a correct delay report and normalized excess risk within 10% of the oracle, the reported history is functionally unused under our matched masking test in 55.4% of N-HiTS cases and 92.7% of TCN cases. Finally, we show that routing the prediction through the reported history removes off-report bypass paths; a hard one-hot control achieves exact fixed-report alignment. The main conclusion is simple: a good forecast, even with a correct delay report, does not show that the model used the right history.

## 1 Introduction

Time-series models are usually judged by forecast error. That is not enough when the location of useful history matters. Autocorrelated inputs can make nearby lags look similar, output memory can hide a wrong input delay, and a model can predict well while using a different part of the past from the one that generated the output [Bjorklund and Ljung, 2003, Sadler and Kozick, 2006, Ljung, 1999, Seborg et al., 2016, Kay,¨ 1993].

We ask one direct question: does aforecaster use the right history? This requires three separate checks. Recoverability asks whether the observed trajectory contains enough information to distinguish the true delay from alternatives. Recovery/reporting asks whether the fitted model identifies that delay or kernel. Functional use asks whether changing the reported history actually changes the numerical forecast. These are different claims, and we audit them in the following order (the arrows denote audit order, not logical implication):

recoverability −→ recovery/report −→ functional use.

A model should not be blamed for a delay that the data cannot distinguish. Conversely, a correct report need not affect the forecast at all. For example, a model may output “delay 8” while its numerical prediction changes mostly when lag 12 is perturbed. Forecast error alone cannot reveal that mismatch. The central question is therefore the last link: even when a delay is recoverable, correctly reported, and paired with a near-oracle forecast, was that history actually used?

Our contributions follow the same order. First, we measure whether the delay is recoverable before judging the model. The point-delay and finite-kernel settings admit input-conditioned Bayes calibration, while the autoregressive setting uses a profile score that allows a wrong lag to refit the other dynamic parameters. Second, we prove that accurate reporting and near-oracle forecasting can still coexist with use of the wrong lag. We then test this exact joint event, rather than only comparing forecast and structural rankings. Third, we intervene on past inputs to measure use directly and study routing as a controlled fix. A no-bypass factorization removes off-report historical paths, and a hard one-hot control realizes the point-delay guarantee exactly. DelayBench-TS is only the controlled testbed that makes these questions measurable; the main result is the separation between recoverability, reporting, and actual use. The nontrivial point is not merely that a model explanation can be unfaithful. It is that the mismatch can survive after the temporal structure is statistically recoverable, the model reports that structure correctly, and the numerical forecast is already near the oracle. This joint condition removes three easier explanations—ambiguous data, an incorrect report, and a poor predictor—before asking whether the reported history actually participates in prediction.

## 2 Related Work

Delay estimation and recoverability. Classical system identification and newer time-series methods estimate delays, lead–lag structure, or temporal edges directly [Bjorklund and Ljung, 2003, Sadler and ¨ Kozick, 2006, Ljung, 1999, Ma and Huang, 2023, Cattaldo et al., 2024, Heyse et al., 2021, Zhao and Shen, 2024, Runge et al., 2019a,b, Runge, 2020]. Related work also asks whether a time-series structure is identifiable from the observed data [Kuskova et al., 2026b, Tan et al., 2025]. We use the same recoverability principle for an explicit finite delay or kernel decision. When the candidate likelihood is complete, we compute an input-conditioned Bayes limit for the structural decision itself. When autoregressive memory introduces nuisance compensation, we use a profile-separation diagnostic instead of calling it a Bayes oracle. The goal is practical: before interpreting a model’s structural error, decide whether that trajectory actually made the delay distinguishable.

Reports, explanations, and actual use. Readable explanations need not match the computation producing a prediction [Jain and Wallace, 2019, Jacovi and Goldberg, 2020, Ross et al., 2017]. Time-series work tests perturbations, context windows, and predictive masks [Ozyegen et al., 2022, Liu et al., 2024c,b, Kim et al., 2026, Zhang et al., 2025, Koh et al., 2020, Zheng et al., 2026], while autocorrelation can confound naive temporal attribution [Tunyi, 2026]. Forecast-necessity testing asks by edge ablation whether a reported relation is required for prediction [Kuskova et al., 2026a]. Our audit adds instance-level recoverability and an externally specified structural target: after a delay is recoverable, correctly reported, and paired with near-oracle forecasting, is that location actually used? We compare the report with intervention sensitivity and use “functional use” only for controlled predictor sensitivity, not real-world causal identification.

Controlled benchmarks. Synthetic and causal benchmarks provide controlled structure and held-out mechanisms [Tan et al., 2025, Cheng et al., 2024, Mogensen et al., 2024, Stein et al., 2025, Herdeanu et al., 2025, Yi et al., 2026, Zhao et al., 2026]. We use that setting only to answer the ordered question above: what can the trajectory reveal, what did the model report, and what history did the forecast use? This keeps the controlled generator in a supporting role rather than making benchmark breadth the main claim.

## 3 Problem Setup

An instance contains input $U _ { 1 : T }$ , output $Y _ { 1 : T }$ , and a finite candidate set. Let L be the largest candidate lag, $\mathcal { T } _ { L } = \{ L + 1 , . . . , T \} , n _ { \mathrm { e f f } } = T - L$ , and

$$
S _ { d } u = ( u _ { t - d } ) _ { t \in \mathbb { Z } _ { L } } .
$$

Model and oracle are compared only when they use the same observed information and the same valid window. This matters most for autoregressive models, because the first scored output may depend on a previous output. If that boundary value is observed, both model and oracle must condition on the same value. If it is not observed, it cannot be silently given to the oracle. Full information-contract details are in Section A. In the released P2 data, the burn-in boundary value was not preserved, so the repaired profile audit simply drops that first transition and uses

$$
\begin{array} { r } { \mathcal { I } _ { L } ^ { \mathrm { P 2 } } = \{ L + 2 , . . . , T \} , \qquad n _ { \mathrm { P 2 } } = T - L - 1 . } \end{array}\tag{1}
$$

This changes only the theory calibration; learned predictions are unchanged. The repair changes favorable membership for only 2 of 400 trajectories, so it fixes the information mismatch without changing the empirical story.

We study three controlled mechanisms:

$$
\begin{array} { r l } {  { \operatorname { P 1 : } } } & { Y _ { t } = \alpha U _ { t - d } + \epsilon _ { t } , } \\ {  { \operatorname { P 2 : } } } & { Y _ { t } = a Y _ { t - 1 } + b U _ { t - d } + \epsilon _ { t } , } \\ {  { \operatorname { P 3 : } } } & { Y _ { t } = a Y _ { t - 1 } + \displaystyle \sum _ { k = 0 } ^ { L } h _ { k } U _ { t - k } + \epsilon _ { t } , \qquad h \in \mathcal { H } . } \end{array}\tag{2}
$$

P1 is a point delay, P2 adds output memory, and P3 uses a finite library of point, multipath, or distributed kernels. These three cases are not meant to cover every time-series mechanism; they isolate the two main complications we need for the argument: nuisance compensation and distributed delay structure. Models may report the candidate directly or through a fixed readout. Functional use is measured separately by intervening on past inputs and comparing the resulting response profile with the reported delay or kernel.

For a point report r, let $A _ { k }$ denote the magnitude of the forecast response when the declared intervention is applied at historical location k, with the report and any intervention context held fixed. Let $k _ { \mathrm { p e a k } }$ be the response peak under the fixed evaluation tie rule and define

$$
d _ { \mathrm { u s e } } ( r , A ) = | r - k _ { \mathrm { p e a k } } | .
$$

Thus a correct report and a small structural error do not by definition force a small $d _ { \mathrm { u s e } }$ . The binary “unused” test used for the confirmatory P1 analysis is stricter and masking-based; it is defined in the Link-II experiment below. Full generator, loss, oracle, and intervention details are in the appendix.

## 4 Theory

The theory answers the same three questions as the experiments. Link I asks what the data make possible before any neural model is involved. Link II asks whether a correct report and a good forecast are already enough to show that the correct history was used. Link III asks what changes when the prediction is forced to pass through the reported history. The three links are meant to be read in that order. Full statements and proofs for the Link-I identities are in Sections B and C.

## 4.1 When is the delay recoverable?

For the point-delay model and fixed observed input $u ,$ two candidate delays $d , d ^ { \prime }$ differ by

$$
D _ { \mathrm { K L } } ( d \Vert d ^ { \prime } \mid u ) = \frac { \alpha ^ { 2 } } { 2 \sigma ^ { 2 } } \Vert S _ { d } u - S _ { d ^ { \prime } } u \Vert _ { 2 } ^ { 2 } .\tag{3}
$$

Thus recovery is hard when the realized input looks similar at the two candidate lags. This is an inputconditioned statement: two trajectories generated with the same nominal SNR can have very different delay information if their realized histories have different shift geometry. Under a stationary input with variance $\sigma _ { U } ^ { 2 }$ and autocorrelation $\rho _ { U }$

$$
\mathbb { E } _ { U } D _ { \mathrm { K L } } = n _ { \mathrm { e f f } } \frac { \alpha ^ { 2 } \sigma _ { U } ^ { 2 } } { \sigma ^ { 2 } } \left[ 1 - \rho _ { U } ( | d - d ^ { \prime } | ) \right] ,
$$

which makes the role of sequence length, SNR, lag spacing, and autocorrelation explicit. For two equal-prior candidates, the exact Bayes error is

$$
R _ { 0 - 1 } ^ { \star } ( d , d ^ { \prime } ; u ) = \Phi \left( - \sqrt { D _ { \mathrm { K L } } ( d \| d ^ { \prime } \mid u ) / 2 } \right) ,
$$

where $\Phi$ is the standard normal CDF. For more than two candidates we compute the finite-library Bayes risk numerically. This gives a direct answer to the first question: on this realized trajectory, how often would even the best structural decision rule be wrong?

Output memory makes the problem harder because a wrong delay can refit other parameters. For P2, write the true conditional mean in the intercept-augmented form

$$
m _ { t } ^ { \star } = \eta ^ { \star } + a ^ { \star } Y _ { t - 1 } + b ^ { \star } U _ { t - d ^ { \star } } ,
$$

with $\eta ^ { \star } = 0$ in the centered specification, and let $\boldsymbol { x } _ { t , d ^ { \prime } }$ contain the previous output and the wrong-lag input (plus an intercept when used). Here $\mathbb { E } ^ { \star }$ denotes expectation under the true P2 process. Define

$$
Q _ { d ^ { \prime } } = \operatorname * { i n f } _ { \beta ^ { \prime } } \sum _ { t \in \mathcal { T } _ { L } ^ { \mathrm { P } ^ { 2 } } } \mathbb { E } ^ { \star } [ ( m _ { t } ^ { \star } - x _ { t , d ^ { \prime } } ^ { \top } \beta ^ { \prime } ) ^ { 2 } ] , \qquad \kappa _ { \mathrm { A R X } } = \operatorname * { m i n } _ { d ^ { \prime } \neq d ^ { \star } } \frac { Q _ { d ^ { \prime } } } { n _ { \mathrm { P } 2 } ( \sigma ^ { \star } ) ^ { 2 } } .\tag{4}
$$

Small $\kappa _ { \mathrm { A R X } }$ means a wrong delay can imitate the true mean after nuisance refitting. This is different from ordinary lag correlation: the wrong lag is allowed to exploit the previous output and change its gain before we measure how much mismatch remains. The corresponding population profile KL is $Q _ { d ^ { \prime } } / ( 2 ( \sigma ^ { \star } ) ^ { 2 } )$ when the innovation variance is fixed, and

$$
\frac { n _ { \mathrm { P 2 } } } { 2 } \log \biggl ( 1 + \frac { Q _ { d ^ { \prime } } } { n _ { \mathrm { P 2 } } ( \sigma ^ { \star } ) ^ { 2 } } \biggr )
$$

when that variance is also refitted. $\kappa _ { \mathrm { A R X } }$ is therefore a separation score, not an exact composite Bayes risk. For a finite kernel library, exact removal of known output memory gives $\widetilde { { \pmb y } } = { \bf U } { \pmb h } + { \pmb \epsilon } ,$ with ${ \textbf { U } } =$ $[ S _ { 0 } u , \dots , S _ { L } u ]$ , and

$$
D _ { \mathrm { K L } } ( h \| h ^ { \prime } \mid u ) = \frac { 1 } { 2 \sigma ^ { 2 } } \| \mathbf { U } ( h - h ^ { \prime } ) \| _ { 2 } ^ { 2 } .\tag{5}
$$

Define the worst finite-library separation

$$
\gamma ( \mathbf { U } , \mathcal { H } ) = \operatorname* { m i n } _ { h \neq h ^ { \prime } } \frac { \Vert \mathbf { U } ( h - h ^ { \prime } ) \Vert _ { 2 } } { \Vert h - h ^ { \prime } \Vert _ { 2 } } .
$$

Then every candidate pair satisfies

$$
D _ { \mathrm { K L } } ( \boldsymbol { h } \| \boldsymbol { h } ^ { \prime } \mid \boldsymbol { u } ) \ge \frac { \gamma ( \mathbf { U } , \mathcal { H } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \| \boldsymbol { h } - \boldsymbol { h } ^ { \prime } \| _ { 2 } ^ { 2 } .
$$

So kernel recovery depends on whether the realized input separates the candidate mean responses. If two different kernels produce the same conditional mean under the realized input, no structural readout can distinguish them from the output distribution alone. The exact P3 calibration used here is the $a = 0$ special case. In the experiments, favorable instances use Bayes risk for P1/P3 and $\kappa _ { \mathrm { A R X } }$ for P2. The common role of these quantities is simply to separate data ambiguity from model failure. The exact form changes across the three mechanisms, but the question does not. The main favorable subsets use $R _ { \mathrm { B a v e s } } ^ { \star } \leq . 0 5$ for P1/P3 and $\kappa _ { \mathrm { A R X } } \geq$ .20 for P2. For P1, ambiguity comes from two shifted copies of the realized input being too similar. For P2, a wrong lag can also borrow predictive power from the previous output and refit its gain. For P3, two different kernels can become indistinguishable after they are filtered through the realized input. We therefore do not compare raw structural error without first checking the corresponding recoverability quantity. This is the only purpose of Link I: it tells us when a structural mistake is informative about the model rather than about the data.

## 4.2 A correct report and a good forecast still need not mean use

A correct report and a low forecast error still do not tell us which structure produced the forecast. The gap is more general than the point-delay model.

Proposition 4.1 (Trajectory separation can coexist with vanishing average prediction penalty). For each n, consider two equal-prior Gaussian structural candidates $P _ { j , n } = \mathcal { N } ( \mu _ { j , n } , \sigma ^ { 2 } I _ { n } ) , j \in \{ 0 , 1 \}$ , and define

$$
D _ { n } : = \frac { \| \mu _ { 0 , n } - \mu _ { 1 , n } \| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } .
$$

Suppose $D _ { n } \to \infty$ while $D _ { n } / n  0$ . Let the structural report be the MAP decision, but under the true candidate 0 let the numerical predictor use the alternative mean $\widehat { Y } = \mu _ { 1 , n }$ . Then

$$
\operatorname* { P r } _ { 0 } ( \widehat { H } \neq 0 ) = \Phi ( - \sqrt { D _ { n } / 2 } ) \to 0 , \qquad \frac { 1 } { n } \mathbb { E } \| Y - \widehat { Y } \| _ { 2 } ^ { 2 } = \sigma ^ { 2 } \bigg ( 1 + \frac { 2 D _ { n } } { n } \bigg ) \to \sigma ^ { 2 } .
$$

Thus the two structures can become arbitrarily reliably distinguishable at the trajectory level while using the wrong structure costs asymptotically nothing under average forecast MSE. Accurate reporting and near-oracle forecasting therefore do not imply functional alignment when reporting and prediction are uncoupled.

P1 is a direct special case with $\mu _ { j , n } = \alpha S _ { d _ { i } } u ^ { ( n ) }$ . For $\alpha \neq 0$ , whenever the two shift operators differ, any sequence $D _ { n } \to \infty$ with $D _ { n } / n \to 0$ can be realized by scaling the input; on a correct report the predictor can still use $d _ { 1 }$ and have report–use distance $| d _ { 0 } - d _ { 1 } |$ (proof in Section C). A concrete choice is $D _ { n } = n ^ { \beta }$ for any $0 < \beta < 1 :$ : structural decision error then vanishes while the wrong-structure excess MSE is only $2 \sigma ^ { 2 } n ^ { \beta - \mathrm { i } }$ The reason is simple: structural evidence adds over the whole trajectory, whereas forecast MSE averages the mismatch over coordinates. The proposition is an existence result, not a claim that every accurate forecaster fails this way; it shows why even a correct report and an excellent forecast cannot replace a direct use test.

## 4.3 What routing guarantees

Link II motivates a direct coupling test. Let v collect all nonhistorical inputs or context held fixed in the intervention, and let $\mathbf { g } ( r )$ be the historical gate vector induced by report r. Suppose prediction is forced through

$$
F _ { \mathrm { s t r } } ( u , v ; r ) = G \big ( v , r , \mathbf { g } ( r ) \odot u \big ) ,\tag{6}
$$

and G receives no other function of the historical block u.

Proposition 4.2 (No-bypass constraint under routed prediction). For afixed report r, changing only coordi nates whose gate entry $g _ { k } ( r ) = 0$ cannot change theforecast. IfG is $L _ { G }$ -Lipschitz in its gated input, thenfor scalar intervention amplitude δ a single-coordinate intervention obeys

$$
\begin{array} { r } { \| F _ { \mathrm { s t r } } ( \boldsymbol { u } + \delta \boldsymbol { e } _ { k } , \boldsymbol { v } ; \boldsymbol { r } ) - F _ { \mathrm { s t r } } ( \boldsymbol { u } , \boldsymbol { v } ; \boldsymbol { r } ) \| \le L _ { G } | g _ { k } ( \boldsymbol { r } ) | | \delta | , } \end{array}
$$

where $e _ { k }$ is the kth coordinate vector.

Corollary 4.3 (Exact point-delay alignment under nondegenerate one-hot routing). $I f \mathbf { g } ( r ) = e _ { r }$ and the selected coordinate has nonzero intervention response, then every off-report response is zero and the unique response peak is r. The point-delay report–use distance is therefore zero.

The guarantee is deliberately narrow. It does not say that routing makes a wrong report correct, and it does not force the model to use the selected coordinate strongly. It only says that an off-report historical path cannot bypass the gate. This guarantee holds with the report and any input-derived context fixed.

Remark 4.4 (Interventions can change the report). In deployed mode, changing an input may also change the report or a context variable recomputed from that input. Such a response is a report-selection effect, not an ungated bypass.

The theorem also explains why we keep soft and hard routing separate in the experiments. A dense soft gate can still satisfy the factorization, but every candidate may retain nonzero weight, so its argmax is not a one-hot certificate. The hard control is stronger: once the report is fixed, all other historical coordinates are removed from the forward path. Neither statement says that routing will improve forecast MSE; the guarantee concerns the relation between the report and the history available to the predictor.

## 5 Experiments

The experiments mirror the three theoretical links rather than forming a single leaderboard. We first validate each recoverability quantity and ask whether structural errors remain on favorable trajectories. We then test two increasingly strong separations: disagreement between forecast and structural rankings, followed by the instance-level event “correct report + near-oracle forecast + functional non-use.” Finally, routing is used as a mechanism test: if off-report paths are the source of the mismatch, removing those paths should close the measured gap.

We use the same 400 held-out trajectories per task and three training seeds for the main comparison. TCN [Bai et al., 2018] and N-HiTS [Challu et al., 2023] are the primary neural backbones, and each task has a matched statistical reference. All paired comparisons use the same trajectories. Bootstrap resampling uses complete trajectories; when an estimate combines training seeds, trajectories and seeds are resampled hierarchically. The confirmatory Link-II rerun keeps the original architecture, splits, optimizer, and training budget and only saves the signed forecasts needed for the new conditional test. Exact generation, additional models, leakage checks, bootstrap details, and independent replications are in the appendix. The main text keeps only the results needed for the three questions above.

## 5.1 Recoverability and recovery

Before looking at learned models, we first verify the recoverability quantities themselves. Point-delay Bayes calculations agree with Monte Carlo, the repaired P2 profile score agrees with direct optimization to about $1 0 ^ { - 1 5 }$ relative error, and the finite-kernel Bayes ordering matches the reference error (Figure 1 and Table 10). The P2 repair changes favorable membership for only 2 of 400 trajectories and does not change the model ordering. We then restrict attention to favorable cases to ask whether a learned model still fails when the structure is statistically clear.

Table 1: Recovery and forecasting on the common cohort. “Recoverable error” is structural error restricted to cases favorable under the task-specific recoverability measure. Lower is better.
<table><tr><td>Task</td><td>Method</td><td>Structural error</td><td>Recoverable error</td><td>Forecast MSE</td></tr><tr><td rowspan="5">P1</td><td>Gaussian MAP</td><td>.065</td><td>.010</td><td>1.189</td></tr><tr><td>Elastic Net</td><td>.265</td><td>.138</td><td>2.509</td></tr><tr><td>MLP-NARX</td><td>.798</td><td>.798</td><td>1.559</td></tr><tr><td>TCN</td><td>.273</td><td>.176</td><td>1.673</td></tr><tr><td>N-HiTS</td><td>.500</td><td>.440</td><td>1.678</td></tr><tr><td rowspan="6">P2</td><td>Profile ARX</td><td>.118</td><td>.004</td><td>.388</td></tr><tr><td>PEM-ARX</td><td>.248</td><td>.049</td><td>.733</td></tr><tr><td>MLP-NARX</td><td>.784</td><td>.783</td><td>.520</td></tr><tr><td>TCN</td><td>.370</td><td>.250</td><td>.935</td></tr><tr><td>N-HiTS</td><td>.570</td><td>.500</td><td>.540</td></tr><tr><td>Finite-library MLE</td><td></td><td></td><td></td></tr><tr><td rowspan="6">P3</td><td></td><td>.115</td><td>.004</td><td>.929</td></tr><tr><td>Elastic Net</td><td>.853</td><td>.857</td><td>1.622</td></tr><tr><td>MLP-NARX</td><td>.851</td><td>.854</td><td>.903</td></tr><tr><td>TCN</td><td>.353</td><td>.225</td><td>1.068</td></tr><tr><td>N-HiTS</td><td>.714</td><td>.691</td><td>.912</td></tr></table>

Table 1 shows that the matched references are close to the favorable-region limit, while the learned structural readouts remain much worse. The lightweight baselines make the same point from another angle. In P2, MLP-NARX reaches forecast MSE .520, close to N-HiTS (.540) and well below TCN (.935), but its structural error is .784 and its recoverable error is .783. In P3, MLP-NARX again forecasts competitively (.903) while its kernel-candidate error remains .851. A model can therefore fit the numerical target well while learning little about the declared delay. The remaining recovery gap is not claimed to be an architecture-level impossibility: structural error continues to improve with more training data (Appendix Section G.9).

## 5.2 Forecast accuracy does not imply recovery

Forecast ranking can disagree with structural ranking. In both P2 and P3, TCN has lower structural error than N-HiTS, while N-HiTS has lower forecast MSE; all four paired bootstrap intervals exclude zero (Table 2). Thus the model that predicts better is not necessarily the model that recovers the delay better. The same qualitative separation appears in the independent iTransformer release. This is useful evidence, but it is still weaker than the theorem because it compares two metrics separately. The next experiment conditions on good reporting and good forecasting at the same time.

Table 2: Forecast and structural rankings disagree. Differences are first system minus second system with paired trajectory-bootstrap 95% CIs; lower is better, and the final column names the better system.
<table><tr><td></td><td>Task Metric (first — second)</td><td>Difference [95% CI]</td><td>Better</td></tr><tr><td>P2</td><td>Structural: TCN — N-HiTS</td><td>-0.200 [-0.247, -0.155]</td><td>TCN</td></tr><tr><td>P2</td><td>Forecast MSE: N-HiTS — TCN</td><td>-0.395[-0.523, -0.271]</td><td>N-HiTS</td></tr><tr><td>P3</td><td>Structural: TCN – N-HiTS</td><td>-0.362[-0.407, -0.314]</td><td>TCN</td></tr><tr><td>P3</td><td>Forecast MSE: N-HiTS – TCN</td><td>-0.157[-0.249, -0.081] N-HiTS</td><td></td></tr></table>

## 5.3 Correct and near-oracle still does not mean used

The stronger test conditions on reporting and forecasting at the same time. For row i, define normalized excess risk

$$
\mathrm { N E R } _ { i } = \frac { \| \widehat { y } _ { i } - \mu _ { i } ^ { \star } \| _ { 2 } ^ { 2 } / n _ { i } } { \sigma _ { i } ^ { 2 } } , \qquad \mu _ { i } ^ { \star } = \alpha S _ { d _ { i } ^ { \star } } u _ { i } .\tag{7}
$$

Here $n _ { i }$ is the scored forecast length and $\sigma _ { i } ^ { 2 }$ is the innovation variance for row i. The near-oracle threshold NER ≤ .10 was fixed before evaluating the confirmatory results. Since the oracle risk is the noise variance, $1 + \mathrm { N E R }$ is the conditional risk in oracle-noise units; using the clean conditional mean avoids a lucky-noise definition of “near oracle.” We call the report unused if zeroing its ±1-lag window changes the scalar forecast no more than the 95th-percentile effect of 20 disjoint equal-width windows with the closest standardized-input energy. This label is fixed before conditioning on report correctness or NER; validation is in Section G.5. Table 3 directly shows the separation from Theorem 4.1: even when the report is correct and the forecast is

Table 3: Direct Link-II joint failure on 1,200 seed–trajectory rows per backbone. “Correct” means the reported delay is true; “near” means $\mathrm { N E R } \leq . 1 0 ;$ “both” satisfies both conditions; “unused” counts both-rows whose reported window fails the matched masking test. The last column is unused/both.
<table><tr><td>Backbone</td><td>Correct</td><td>NER ≤ .10</td><td>Both</td><td>Unused</td><td>Unused | both [95% CI]</td></tr><tr><td>N-HiTS</td><td>600</td><td>401</td><td>177</td><td>98</td><td>.554 [.417,.679]</td></tr><tr><td>TCN</td><td>873</td><td>395</td><td>248</td><td>230</td><td>.927 [.865,.977]</td></tr></table>

near the oracle, the reported history is often unused. The effect is not marginal: the conditional unused rate is .554 for N-HiTS and .927 for TCN, with both confidence intervals well above zero. Conditioning on both events is essential. Correct reporting rules out the trivial case in which the model simply named the wrong lag, while the NER filter rules out cases where the forecast is too poor for its history use to be informative. The unused label itself is computed before either conditioning step, so the subset is not created by retuning the functional diagnostic. This does not prove the asymptotic proposition empirically; it shows the same joint failure in finite samples.

The conclusion is also insensitive to the NER cutoff: $\tau _ { \mathrm { N E R } } = . 1 0$ was fixed first, and the same saved forecasts are evaluated at three additional thresholds without retraining or cutoff selection. As Table 4 shows,

Table 4: NER-cutoff sensitivity. n counts correct-report rows with $\mathrm { N E R } \le \tau _ { \mathrm { N E R } } ;$ the last column is the unused fraction. Large rates indicate persistent report–use mismatch.
<table><tr><td>Backbone</td><td>NER cutoff τNER</td><td>Both n</td><td>Unused | both [95% CI]</td></tr><tr><td rowspan="5">N-HiTS</td><td>.01</td><td>47</td><td>.553 [.326,.778]</td></tr><tr><td>.05</td><td>118</td><td>.525 [.349,.701]</td></tr><tr><td>.10</td><td>177</td><td>.554 [.417,.679]</td></tr><tr><td>.20</td><td>234</td><td>.538 [.417,.658]</td></tr><tr><td>.01</td><td>78</td><td>.949 [.857,1.000]</td></tr><tr><td rowspan="4">TCN</td><td>.05</td><td>171</td><td>.942 [.865,1.000]</td></tr><tr><td>.10</td><td>248</td><td>.927 [.865,.977]</td></tr><tr><td>.20</td><td>305</td><td>.918 [.872,.958]</td></tr><tr><td></td><td></td><td></td></tr></table>

N-HiTS stays near .53–.55 unused and TCN above .91 across all four thresholds. On the same locked rows, three additive amplitudes and input Jacobians also show large report–use gaps (Appendix Section G.5), so the result is not tied to one NER cutoff or one sensitivity calculation.

## 5.4 Routing closes the report–use gap

We next ask whether coupling the report to the numerical path removes the mismatch. We separate dense soft routing from a hard one-hot control. The soft router predicts through the full gate vector, so its structural object is the whole vector rather than the argmax alone. The hard control uses only the reported argmax delay in the forward pass and is therefore the implementation that matches Theorem 4.3. No routing hyperparameter search is used for this closure experiment. Table 5 shows that routing removes the measured fixed report–use

Table 5: P1 routing mechanism test on the common cohort. Structural error and forecast MSE are lower-isbetter; fixed report–use gap is in lags, with 0 meaning exact peak alignment under the fixed-report intervention.
<table><tr><td>Backbone</td><td>Regime</td><td></td><td>Struct. error ↓ Forecast MSE ↓</td><td>Fixed gap (lags) ↓</td></tr><tr><td rowspan="3">N-HiTS</td><td>Separate</td><td>.500</td><td>1.678</td><td>9.846</td></tr><tr><td>Soft routed</td><td>.493</td><td>1.365</td><td>0</td></tr><tr><td>Hard one-hot</td><td>.492</td><td>1.507</td><td>0</td></tr><tr><td rowspan="3">TCN</td><td>Separate</td><td>.273</td><td>1.673</td><td>13.791</td></tr><tr><td>Soft routed</td><td>.257</td><td>1.311</td><td>0</td></tr><tr><td>Hard one-hot</td><td>.264</td><td>1.393</td><td>0</td></tr></table>

gap in this P1 closure for both backbones. Importantly, the structural errors change little while the measured gap falls from 9.846/13.791 lags to zero. The closure is therefore not explained by a large improvement in report accuracy; it follows the change in how historical information reaches the numerical predictor. The soft router also has the lowest forecast MSE here, but only the hard one-hot route matches the point-delay guarantee in Theorem 4.3; the table should therefore be read as a mechanism comparison, not as a claim that routing must improve forecasting.

Table 6: Numerical check of the hard one-hot guarantee under a fixed report. The theorem-compatible target is off-report maximum = 0 and each fraction = 1; responses use tolerance $1 0 ^ { - 7 }$
<table><tr><td></td><td></td><td></td><td>Backbone Off-report max ↓ Off-report zero frac. ↑ Selected active frac. ↑ Peak = report frac. ↑</td><td></td></tr><tr><td>N-HiTS</td><td>0</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>TCN</td><td>0</td><td>1.000</td><td>1.000</td><td>1.000</td></tr></table>

Table 6 checks the theorem directly. For both backbones, every fixed-report off-support response is zero to 10<sup>−7</sup>, every selected response is nonzero, and every response peak equals the report. This is the numerical counterpart of the one-hot corollary. The result is about alignment, not guaranteed forecast improvement: soft routing has the lowest P1 MSE in this release, while hard routing still improves on the separate baseline. Soft routing also removes the ungated bypass in the tested implementation, but its gate is dense and should not be read as a one-hot delay certificate.

Cross-task results give the same direction. As Table 7 shows, shared routing reduces the report–use gap by about 9.8–13.1 lags across P1–P3, and all three 95% intervals exclude zero. This matters because the exact one-hot guarantee is P1-specific, whereas the empirical coupling benefit is not confined to that task. These cross-task effects remain empirical and are not assigned the hard one-hot guarantee.

For non-point P3 kernels, the same coupling improves full-profile $\ell _ { 2 }$ and cosine scores for both backbones; path recall improves for TCN but not significantly for N-HiTS (Table 9). This full-profile check is kept in the appendix because it supports the routing claim without changing the main point-delay theorem.

Table 7: Cross-task empirical coupling contrast. Entries are shared-gate minus separate-head report–use gaps in lags with trajectory-bootstrap 95% intervals; negative favors routing. This is an empirical contrast, not the hard one-hot guarantee.
<table><tr><td>Task ∆ gap = shared — separate (lags) [95% CI]</td><td></td></tr><tr><td>P1</td><td> $- 1 3 . 1 1 3 [ - 1 3 . 9 8 3 , - 1 2 . 2 1 5 ]$ </td></tr><tr><td>P2</td><td> $- 1 1 . 1 5 4 [ - 1 2 . 0 7 8 , - 1 0 . 1 7 8 ]$ </td></tr><tr><td>P3</td><td> $- 9 . 8 3 5 [ - 1 1 . 8 6 5 , - 7 . 9 6 5 ]$ </td></tr></table>

## 5.5 Does the routing effect transfer?

We also drive the controlled mechanisms with Beijing air-quality and hydraulic input histories. All 12 shared-minus-separate report–use intervals favor coupling:

Table 8: Real-input semi-synthetic routing test. Each entry is the shared-minus-separate report–use gap in lags with a paired 95% CI; negative means routing improves alignment.
<table><tr><td></td><td>Task Source</td><td>TCN</td><td>N-HiTS</td></tr><tr><td>P1</td><td>Beijing</td><td>-11.67[-13.69, -10.01]</td><td> $- 9 . 4 8 \left[ { - 1 2 . 7 6 , - 6 . 7 4 } \right]$ </td></tr><tr><td rowspan="2">P2</td><td>Hydraulic</td><td>-11.57[-14.34, -9.12]</td><td> $- 1 3 . 7 9 [ - 1 5 . 5 6 , - 1 1 . 6 8 ]$ </td></tr><tr><td>Beijing</td><td>-11.81[-13.71, -9.37]</td><td> $- 8 . 1 4 \left[ - 1 0 . 3 5 , - 6 . 2 3 \right]$ </td></tr><tr><td rowspan="2">P3</td><td>Hydraulic</td><td> $- 1 4 . 2 4 [ - 1 6 . 4 6 , - 1 1 . 6 7 ]$ </td><td> $- 7 . 1 7 \left[ - 8 . 6 4 , - 5 . 5 6 \right]$ </td></tr><tr><td>Beijing</td><td> $- 9 . 4 6 \left[ - 1 1 . 1 3 , - 7 . 7 9 \right]$ </td><td> $- 8 . 7 5 \left[ - 1 2 . 0 4 , - 4 . 0 2 \right]$ </td></tr><tr><td></td><td>Hydraulic</td><td> $- 1 4 . 0 9 [ - 1 6 . 4 7 , - 1 1 . 0 9 ]$ </td><td> $- 1 0 . 0 9 [ - 1 2 . 6 2 , - 6 . 9 3 ]$ </td></tr></table>

As Table 8 shows, every one of the 12 backbone–task–source contrasts favors coupling, and all 12 intervals exclude zero. These tests keep the output-side delay mechanism known while changing the input geometry. Together with the stress and physics-inspired checks in Sections G.6 and G.7, they show that the report–use effect is not confined to white or AR(1) inputs. The transfer claim remains narrow: routing improves report–use alignment under these controlled mechanisms and authentic input histories. It does not show that routing always improves forecast accuracy or identify real-world causal structure.

Broader robustness and transfer audits are reported in the appendix.

## 6 Discussion and Conclusion

The paper separates three questions that are often mixed together: can the delay be recovered from the data, did the model report it, and did the forecast actually use it? The main result is that success on the first two questions, even together with a near-oracle forecast, does not answer the third. Routing then provides a controlled closure test: hard one-hot routing removes off-report historical paths exactly, while the softer shared gate shows that the same direction holds empirically beyond the point-delay guarantee.

Whether this mismatch matters depends on the role of the report. If the only goal is in-distribution forecast MSE, exploiting a correlated proxy lag may be harmless. The issue arises when a reported delay is interpreted as evidence about the process or used for sensor selection, diagnosis, or control. In those settings, a correct-looking report that is disconnected from the prediction path can support the wrong downstream conclusion even when the forecast itself is accurate. We therefore do not argue that every forecaster must recover the data-generating delay; we argue that a structural report should not be treated as functionally meaningful without a separate use test.

The scope remains narrow. P2 uses profile separation rather than a composite Bayes oracle, the exact P3 calibration uses the a = 0 special case of the known-memory result, the semi-synthetic tests retain controlled output mechanisms, and the interventions measure predictor sensitivity rather than real-world causality. Extending the same recoverability–report–use audit to multivariate pathways, learned context windows, and large pretrained forecasters is a natural next step. Within the present scope, the conclusion is simple: good forecasting does not show that a model recovered or used the right history, and even a correct delay report is not enough.

## References

Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271, 2018. doi: 10.48550/arXiv. 1803.01271.

Svante Bjorklund and Lennart Ljung. A review of time-delay estimation techniques. In¨ Proceedings ofthe 42nd IEEE Conference on Decision and Control, volume 3, pages 2502–2507, Maui, HI, USA, 2003. IEEE. doi: 10.1109/CDC.2003.1272997.

Marco Cattaldo, Alberto Ferrer, and Ingrid Mage. Variable time delay estimation in continuous industrial˚ processes. Chemometrics and Intelligent Laboratory Systems, 246:105082, 2024. doi: 10.1016/j.chemolab. 2024.105082. URL https://doi.org/10.1016/j.chemolab.2024.105082.

Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max Mergenthaler-Canseco, and Artur Dubrawski. N-hits: Neural hierarchical interpolation for time series forecasting. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 37, pages 6989–6997, 2023. doi: 10.1609/aaai.v37i6.25854.

Song Chen. Beijing multi-site air quality. UCI Machine Learning Repository, 2017. URL https: //doi.org/10.24432/C5RK5G. Dataset.

Yuxiao Cheng, Ziqian Wang, Tingxiong Xiao, Qin Zhong, Jinli Suo, and Kunlun He. Causaltime: Realistically generated time-series for benchmarking of causal discovery. In The Twelfth International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/ paper/2024/hash/0c79d6ed1788653643a1ac67b6ea32a7-Abstract-Conference. html.

Thomas M. Cover and Joy A. Thomas. Elements ofInformation Theory. Wiley-Interscience, Hoboken, NJ, 2 edition, 2006. ISBN 9780471241959. doi: 10.1002/047174882X.

Luc Devroye, Laszl´ o Gy´ orfi, and G˝ abor Lugosi.´ A Probabilistic Theory ofPattern Recognition, volume 31 of Stochastic Modelling and Applied Probability. Springer, New York, 1996. ISBN 9781461207115. doi: 10.1007/978-1-4612-0711-5.

Bradley Efron and Robert J. Tibshirani. An Introduction to the Bootstrap, volume 57 of Monographs on Statistics and Applied Probability. Chapman & Hall, New York, 1993. ISBN 9780412042317.

Nikolai Helwig, Eliseo Pignanelli, and Andreas Schutze. Condition monitoring of hydraulic systems. UCI¨ Machine Learning Repository, 2015. URL https://doi.org/10.24432/C5CW21. Dataset.

Benjamin Herdeanu, Juan Nathaniel, Carla Roesch, Jatan Buch, Gregor Ramien, Johannes Haux, and Pierre Gentine. Causaldynamics: A large-scale benchmark for structural discovery of dynamical causal models. In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 39a3f7dc00e4723a5ca7808b04fc4d4f-Abstract-Datasets\_and\_Benchmarks\_ Track.html. Datasets and Benchmarks Track.

Jolan Heyse, Laurent Sheybani, Serge Vulliemoz, and Pieter van Mierlo. Evaluation of directed causality´ measures and lag estimations in multivariate time-series. Frontiers in Systems Neuroscience, 15:620338, 2021. doi: 10.3389/fnsys.2021.620338.

Alon Jacovi and Yoav Goldberg. Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness? In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4198–4205, 2020. doi: 10.18653/v1/2020.acl-main.386.

Sarthak Jain and Byron C. Wallace. Attention is not explanation. In Proceedings of NAACL-HLT, pages 3543–3556, 2019. doi: 10.18653/v1/N19-1357.

Steven M. Kay. Fundamentals ofStatistical Signal Processing, Volume I: Estimation Theory. Prentice Hall, Upper Saddle River, NJ, 1993. ISBN 9780133457117.

Changhun Kim, Yechan Mun, Hyeongwon Jang, Eunseo Lee, Sangchul Hahn, and Eunho Yang. Delta-XAI: A unified framework for explaining prediction changes in online time series monitoring. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/ forum?id=ZHW5pp5nE5.

Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 5338–5348, 2020. URL https://proceedings.mlr.press/v119/koh20a.html.

Solomon Kullback and Richard A. Leibler. On information and sufficiency. The Annals of Mathematical Statistics, 22(1):79–86, 1951. doi: 10.1214/aoms/1177729694.

Valentina Kuskova, Dmitry Zaytsev, and Michael Coppedge. Beyond coefficients: Forecast-necessity testing for interpretable causal discovery in nonlinear time-series models. arXiv preprint arXiv:2604.18751, 2026a. doi: 10.48550/arXiv.2604.18751. URL https://arxiv.org/abs/2604.18751.

Valentina Kuskova, Dmitry Zaytsev, and Michael Coppedge. When are neural interaction discoveries real? identifiability, recoverability, and a pre-fit diagnostic. arXiv preprint arXiv:2606.08390, 2026b. doi: 10.48550/arXiv.2606.08390. URL https://arxiv.org/abs/2606.08390.

Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. itransformer: Inverted transformers are effective for time series forecasting. In The Twelfth International Conference on Learning Representations, 2024a. URL https://openreview.net/forum?id=JePfAI8fah.

Zichuan Liu, Tianchun Wang, Jimeng Shi, Xu Zheng, Zhuomin Chen, Lei Song, Wenqian Dong, Jayantha Obeysekera, Farhad Shirani, and Dongsheng Luo. TimeX++: Learning time-series explanations with information bottleneck. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 32062–32082. PMLR, 2024b. URL https://proceedings.mlr.press/v235/liu24bl.html.

Zichuan Liu, Yingying Zhang, Tianchun Wang, Zefan Wang, Dongsheng Luo, Mengnan Du, Min Wu, Yi Wang, Chunlin Chen, Lunting Fan, and Qingsong Wen. Explaining time series via contrastive and locally sparse perturbations. In The Twelfth International Conference on Learning Representations, 2024c. URL https://openreview.net/forum?id=qDdSRaOiyb.

Lennart Ljung. System Identification: Theory for the User. Prentice Hall PTR, Upper Saddle River, NJ, 2 edition, 1999. ISBN 9780136566953.

Xin-Yue Ma and Chun-Qing Huang. Data-driven approach for time-delay estimation of industrial processes. ISA Transactions, 137:35–58, 2023. doi: 10.1016/j.isatra.2023.01.028.

Søren Wengel Mogensen, Karin Rathsman, and Per Nilsson. Causal discovery in a complex industrial system: A time series benchmark. In Francesco Locatello and Vanessa Didelez, editors, Proceedings of the Third Conference on Causal Learning and Reasoning, volume 236 of Proceedings of Machine Learning Research, pages 1218–1236. PMLR, 2024. URL https://proceedings.mlr.press/ v236/mogensen24a.html.

Susan A. Murphy and Aad W. van der Vaart. On profile likelihood. Journal of the American Statistical Association, 95(450):449–465, 2000. doi: 10.1080/01621459.2000.10474219.

Ozan Ozyegen, Igor Ilic, and Mucahit Cevik. Evaluation of interpretability methods for multivariate time series forecasting. Applied Intelligence, 52(5):4727–4743, 2022. doi: 10.1007/s10489-021-02662-2.

Christian P. Robert and George Casella. Monte Carlo Statistical Methods. Springer Texts in Statistics. Springer, New York, 2 edition, 2004. ISBN 9780387212395. doi: 10.1007/978-1-4757-4145-2.

Andrew Slavin Ross, Michael C. Hughes, and Finale Doshi-Velez. Right for the right reasons: Training differentiable models by constraining their explanations. arXiv preprint arXiv:1703.03717, 2017.

Jakob Runge. Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets. In Jonas Peters and David Sontag, editors, Proceedings of the 36th Conference on Uncertainty in Artificial Intelligence, volume 124 of Proceedings of Machine Learning Research, pages 1388–1397. PMLR, 2020. URL https://proceedings.mlr.press/v124/runge20a.html.

Jakob Runge, Sebastian Bathiany, Erik Bollt, Gustau Camps-Valls, Dim Coumou, Ethan Deyle, Clark Glymour, Marlene Kretschmer, Miguel D. Mahecha, Jordi Munoz-Mar˜ ´ı, Egbert H. van Nes, Jonas Peters, Rick Quax, Markus Reichstein, Marten Scheffer, Bernhard Scholkopf, Peter Spirtes, George Sugihara, Jie¨ Sun, Kun Zhang, and Jakob Zscheischler. Inferring causation from time series in earth system sciences. Nature Communications, 10:2553, 2019a. doi: 10.1038/s41467-019-10105-3.

Jakob Runge, Peer Nowack, Marlene Kretschmer, Seth Flaxman, and Dino Sejdinovic. Detecting and quantifying causal associations in large nonlinear time series datasets. Science Advances, 5(11):eaau4996, 2019b. doi: 10.1126/sciadv.aau4996.

Brian M. Sadler and Richard J. Kozick. A survey of time delay estimation performance bounds. In 2006 IEEE Sensor Array and Multichannel Signal Processing Workshop Proceedings, pages 282–288, Waltham, MA, USA, 2006. IEEE. doi: 10.1109/SAM.2006.1706138.

Dale E. Seborg, Thomas F. Edgar, Duncan A. Mellichamp, and Francis J. Doyle. Process Dynamics and Control. John Wiley & Sons, Hoboken, NJ, 4 edition, 2016. ISBN 9781119285915.

Gideon Stein, Maha Shadaydeh, Jan Blunk, Niklas Penzel, and Joachim Denzler. Causalrivers: Scaling up benchmarking of causal discovery for real-world time-series. In The Thirteenth International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/ paper/2025/hash/a205fda871b0f6c1e18a7ad7325eb6cf-Abstract-Conference. html. Spotlight.

Mingtian Tan et al. Syntsbench: Rethinking temporal pattern learning in deep learning models for time series. In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ d60361f3d0021c76626b2597957261d6-Abstract-Datasets\_and\_Benchmarks\_ Track.html. Datasets and Benchmarks Track.

Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer Series in Statistics. Springer, New York, 2009. ISBN 9780387790527. doi: 10.1007/b13794.

Amadeo Tunyi. The failures of marginal influence-based attribution methods for global time series explanations. arXiv preprint arXiv:2607.16236, 2026. doi: 10.48550/arXiv.2607.16236. URL https://arxiv.org/abs/2607.16236.

Huiyang Yi, Xiaojian Shen, Yonggang Wu, Duxin Chen, He Wang, and Wenwu Yu. Causalcompass: Evaluating the robustness of time-series causal discovery in misspecified scenarios. arXiv preprint arXiv:2602.07915, 2026. doi: 10.48550/arXiv.2602.07915. URL https://arxiv.org/abs/2602. 07915.

Jiahui Zhang, Zhengyang Zhou, Wenjie Du, and Yang Wang. Enhancing the maximum effective window for long-term time series forecasting. In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://openreview.net/forum?id=Gmwsy7TlFI.

Lifan Zhao and Yanyan Shen. Rethinking channel dependence for multivariate time series forecasting: Learning from leading indicators. In The Twelfth International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ b52b07a239a7afa155ca25cf17a55074-Abstract-Conference.html.

Yuyang Zhao, Lian Xu, Hao Miao, Chenxi Liu, and Hao Xue. Ts-fault: Benchmarking time series forecasters against structural faults. arXiv preprint arXiv:2606.18539, 2026. doi: 10.48550/arXiv.2606.18539.

Xu Zheng, Wei Cheng, Zhuomin Chen, Mo Sha, Jingchao Ni, and Dongsheng Luo. Information bottleneck learning for faithful time series forecasting explanations. arXiv preprint arXiv:2607.28124, 2026. doi: 10.48550/arXiv.2607.28124. URL https://arxiv.org/abs/2607.28124.

## A Detailed Assumptions and Notation

Candidate, input, and prior contract. The candidate index H is independent of the input mechanism in the core synthetic tier: $U \perp H$ . Exact Bayes statements condition on every boundary variable actually used by the candidate likelihood. If an input policy or initial-history law depends on the candidate, the structural prior must be conditioned on the observed variables and the corresponding likelihood contribution included. The released P2 profile calibration does not invoke such a Bayes claim: because its burn-in boundary $Y _ { L }$ was not serialized faithfully, the repaired diagnostic simply excludes the first post-boundary transition and uses Equation (1).

Innovation contract. P1 and P3 assume independent Gaussian innovations. P2 assumes $\epsilon _ { t }$ is independent of the filtration generated by past output and the observed exogenous input. Closed-loop inputs are excluded from the exact P2 theorem unless the controller and likelihood are explicitly modeled.

Boundary history and the P2 repair. The general P2/P3 conditional formulas require the boundary output history used by the recursion to be observed and conditioned upon consistently. The released P2 generator violated that condition only for the first scored transition: burn-in produced a random candidate-dependent $Y _ { L }$ , but serialization replaced that boundary value while the target $Y _ { L + 1 }$ had already been generated from it. We therefore make no same-information claim for that transition. All reported P2 profile calibration in the repaired release starts at $L + 2$ , where $Y _ { L + 1 }$ is observed, and uses $n _ { \mathrm { P 2 } } = T - L - 1$ . The implemented exact P3 cohort has $a = 0$ , so no output-boundary history enters its KL/Bayes computation. For a future nonzero-a exact P3 cohort, the common observed boundary history would have to be conditioned upon explicitly.

Known versus unknown nuisance parameters. A known-parameter Bayes oracle defines a strict sameinformation limit only when those parameters are fixed and included in the observed information set for every evaluated system. If the parameters are hidden or vary within the scenario, the known-parameter result is a labeled lower reference. Profile likelihood is a feasible reference when gain, previous-output carry-over, or variance are unknown. Profile-KL is a population separation diagnostic for a composite incorrect candidate; it is not, by itself, an exact composite-hypothesis Bayes risk.

## B Additional Link-I Statements

The main text keeps only the recoverability formulas needed for the argument. The exact decision and profile statements used by the proofs are collected here.

Proposition B.1 (Conditional delay information). For any d, $d ^ { \prime } \in \mathcal { D }$

$$
D _ { \mathrm { K L } } ( P _ { d } ^ { ( u ) } \| P _ { d ^ { \prime } } ^ { ( u ) } ) = \frac { \alpha ^ { 2 } } { 2 \sigma ^ { 2 } } \left\| S _ { d } u - S _ { d ^ { \prime } } u \right\| _ { 2 } ^ { 2 } .\tag{8}
$$

Corollary B.2 (Average information under stationary input). If U<sub>t</sub> is second-order stationary with variance $\sigma _ { U } ^ { 2 }$ and autocorrelation $\rho _ { U }$ , define $\mathrm { S N R } _ { \mathrm { p u r e } } : = \alpha ^ { 2 } \sigma _ { U } ^ { 2 } / \bar { \sigma ^ { 2 } }$ . Thenfor $\Delta = | d - d ^ { \prime } |$

$$
\mathbb { E } _ { U } [ D _ { \mathrm { K L } } ( P _ { d } ^ { ( U ) } | | P _ { d ^ { \prime } } ^ { ( U ) } ) ] = n _ { \mathrm { e f f } } \mathrm { S N R } _ { \mathrm { p u r e } } [ 1 - \rho _ { U } ( \Delta ) ] .\tag{9}
$$

Proposition B.3 (Exact binary Bayes risk). For two equal-prior candidates with common covariance,

$$
R _ { 0 - 1 } ^ { \star } ( d , d ^ { \prime } ; u ) = \Phi \left( - \sqrt { D _ { \mathrm { K L } } ( P _ { d } ^ { ( u ) } \| P _ { d ^ { \prime } } ^ { ( u ) } ) / 2 } \right) .\tag{10}
$$

Proposition B.4 (Known-parameter ARX delay information). Let $\vartheta ^ { \star }$ denote the true nuisance-parameter tuple, with $\vartheta ^ { \star } = ( \eta ^ { \star } , a ^ { \star } , b ^ { \star } , ( \sigma ^ { \star } ) ^ { 2 } )$ and $\eta ^ { \star } = 0$ in the centered specification. For d, $d ^ { \prime } \in \mathcal { D }$

$$
\begin{array} { l } { { \displaystyle { \cal D } _ { \mathrm { K L } } ( P _ { d , \vartheta ^ { \star } } ^ { ( u ) } | | P _ { d ^ { \prime } , \vartheta ^ { \star } } ^ { ( u ) } ) } } \\ { { \displaystyle ~ = \frac { ( b ^ { \star } ) ^ { 2 } } { 2 ( \sigma ^ { \star } ) ^ { 2 } } \sum _ { t \in { \mathcal T } _ { L } ^ { \mathrm { P 2 } } } ( u _ { t - d } - u _ { t - d ^ { \prime } } ) ^ { 2 } . } } \end{array}\tag{11}
$$

Proposition B.5 (Population profile-KL). Condition on the common observed initial history $h _ { 0 } .$ . Use either centered regressors $\pmb { x } _ { t , d ^ { \prime } } = ( Y _ { t - 1 } , U _ { t - d ^ { \prime } } ) ^ { \top }$ or the intercept-augmented version $\pmb { x } _ { t , d ^ { \prime } } = ( 1 , Y _ { t - 1 } , U _ { t - d ^ { \prime } } ) ^ { \top }$ . If the incorrect candidate uses the true innovation variance $( \sigma ^ { \star } ) ^ { 2 }$ , then

$$
J _ { \mathrm { p r o f } } ^ { \mathrm { f i x e d } ~ s ^ { 2 } } ( d ^ { \star } , d ^ { \prime } ) = \frac { Q _ { d ^ { \prime } } } { 2 ( \sigma ^ { \star } ) ^ { 2 } } .\tag{12}
$$

Ifinstead its single common innovation variance $s ^ { 2 } > 0$ is also optimized, then

$$
J _ { \mathrm { p r o f } } ^ { \mathrm { f r e e } ~ s ^ { 2 } } ( d ^ { \star } , d ^ { \prime } ) = \frac { n _ { \mathrm { P 2 } } } { 2 } \log \left( 1 + \frac { { \cal Q } _ { d ^ { \prime } } } { n _ { \mathrm { P 2 } } ( \sigma ^ { \star } ) ^ { 2 } } \right) .\tag{13}
$$

Proposition B.6 (Kernel separation). For every distinct h, $\pmb { h } ^ { \prime } \in \mathcal { H } .$

$$
D _ { \mathrm { K L } } ( P _ { h } ^ { ( u ) } \| P _ { h ^ { \prime } } ^ { ( u ) } ) \geq \frac { \gamma ( \mathbf { U } , \mathcal { H } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \left\| h - h ^ { \prime } \right\| _ { 2 } ^ { 2 } .\tag{14}
$$

Moreover, $\gamma ( \mathbf { U } , \mathcal { H } ) = 0$ iffat least two candidates generate identical conditional means.

## C Proofs

## C.1 Proof of Theorem B.1

Both candidates are multivariate Gaussian with common covariance $\sigma ^ { 2 } I$ and means $\mu _ { d } ~ = ~ \alpha S _ { d } u$ and $\mu _ { d ^ { \prime } } = \alpha S _ { d ^ { \prime } } u$ . Using the standard Gaussian KL identity [Kullback and Leibler, 1951, Cover and Thomas, 2006] gives

$$
D _ { \mathrm { K L } } ( P _ { d } ^ { ( u ) } | | P _ { d ^ { \prime } } ^ { ( u ) } ) = \frac { 1 } { 2 } ( \mu _ { d } - \mu _ { d ^ { \prime } } ) ^ { \top } ( \sigma ^ { 2 } I ) ^ { - 1 } ( \mu _ { d } - \mu _ { d ^ { \prime } } )\tag{15}
$$

$$
= \frac { \alpha ^ { 2 } } { 2 \sigma ^ { 2 } } \left\| S _ { d } u - S _ { d ^ { \prime } } u \right\| _ { 2 } ^ { 2 } .\tag{16}
$$

## C.2 Proof of Theorem B.2

For each $t \in \mathcal { Z } _ { L }$ , stationarity yields

$$
\begin{array} { r l } & { \mathbb { E } [ ( U _ { t - d } - U _ { t - d ^ { \prime } } ) ^ { 2 } ] = 2 \sigma _ { U } ^ { 2 } - 2 \operatorname { C o v } ( U _ { t - d } , U _ { t - d ^ { \prime } } ) } \\ & { \qquad = 2 \sigma _ { U } ^ { 2 } [ 1 - \rho _ { U } ( | d - d ^ { \prime } | ) ] . } \end{array}\tag{17}
$$

(18)

Summing over $n _ { \mathrm { e f f } }$ positions and substituting into Theorem B.1 gives the result.

## C.3 Proof of Theorem B.3

Let $\Delta _ { M } ^ { 2 } = ( \mu _ { d } - \mu _ { d ^ { \prime } } ) ^ { \top } \Sigma ^ { - 1 } ( \mu _ { d } - \mu _ { d ^ { \prime } } )$ . Under equal priors and common covariance, the optimal linear discriminant has error $\Phi ( - \Delta _ { M } / 2 )$ . Since $D _ { \mathrm { K L } } ( P _ { d } | | P _ { d ^ { \prime } } ) = \Delta _ { M } ^ { 2 } / 2$ , the claimed expression follows. Monotonic inversion of $\Phi$ gives the information threshold.

For a finite candidate set with prior masses $\pi _ { j }$ and conditional densities $p _ { j } ^ { ( u ) }$ , the 0–1 Bayes risk is

$$
R _ { \mathrm { B a y e s } } ^ { \star } ( u ) = 1 - \int \operatorname* { m a x } _ { j } \{ \pi _ { j } p _ { j } ^ { ( u ) } ( y ) \} \mathrm { d } y .\tag{19}
$$

## C.4 Derivation of multicandidate Bayes risk

For any rule $\psi ,$ , the correct-decision probability is

$$
\sum _ { j } \int \mathbf { 1 } \{ \psi ( y ) = j \} \pi _ { j } p _ { j } ^ { ( u ) } ( y ) \mathrm { d } y .\tag{20}
$$

Pointwise maximization is achieved by the MAP rule and equals R max<sub>j</sub> $\pi _ { j } p _ { j } ^ { ( u ) } ( y )$ dy. Taking the complement proves Equation (19).

## C.5 Proof of Theorem B.4

The conditional trajectory KL decomposes over time. At time $t ,$ both candidates have variance $( \sigma ^ { \star } ) ^ { 2 }$ and conditional means

$$
a ^ { \star } Y _ { t - 1 } + b ^ { \star } u _ { t - d } , \qquad a ^ { \star } Y _ { t - 1 } + b ^ { \star } u _ { t - d ^ { \prime } } .\tag{21}
$$

Any common intercept and the shared $a ^ { \star } Y _ { t - 1 }$ term cancel, so the mean difference is deterministic conditional on the observed input and equals $b ^ { \star } ( u _ { t - d } - u _ { t - d ^ { \prime } } )$ . Summing the one-dimensional equal-variance Gaussian KL terms proves the proposition.

## C.6 Proof of Theorem B.5

Using the standard profile-likelihood construction [Murphy and van der Vaart, 2000], for a candidate parameter $\beta ^ { \prime }$ and fixed candidate variance $( \sigma ^ { \star } ) ^ { 2 }$ , the conditional KL is

$$
\frac { 1 } { 2 ( \sigma ^ { \star } ) ^ { 2 } } \sum _ { t } \mathbb { E } ^ { \star } [ ( m _ { t } ^ { \star } - \pmb { x } _ { t , d ^ { \prime } } ^ { \top } \pmb { \beta } ^ { \prime } ) ^ { 2 } ] .\tag{22}
$$

The infimum over $\beta ^ { \prime }$ is the population least-squares residual $Q _ { d ^ { \prime } }$ , giving the first result. Define

$$
\Gamma _ { d ^ { \prime } } : = \sum _ { t \in \mathbb { Z } _ { L } ^ { \mathrm { P 2 } } } \mathbb { E } ^ { \star } [ \pmb { x } _ { t , d ^ { \prime } } \pmb { x } _ { t , d ^ { \prime } } ^ { \top } ] ,
$$

$$
\pmb { c } _ { d ^ { \prime } } : = \sum _ { t \in \mathbb { Z } _ { L } ^ { \mathrm { P 2 } } } \mathbb { E } ^ { \star } [ \pmb { x } _ { t , d ^ { \prime } } m _ { t } ^ { \star } ] ,\tag{23}
$$

$$
\boldsymbol { M } ^ { \star } : = \sum _ { t \in \mathbb { Z } _ { L } ^ { \mathrm { P 2 } } } \mathbb { E } ^ { \star } [ ( \boldsymbol { m } _ { t } ^ { \star } ) ^ { 2 } ] .
$$

Then the quadratic objective is $M ^ { \star } - 2 \beta ^ { \prime \top } c _ { d ^ { \prime } } + \beta ^ { \prime \top } \Gamma _ { d ^ { \prime } } \beta ^ { \prime }$ . Since $\Gamma _ { d ^ { \prime } }$ is positive semidefinite and the second moments are finite, $\pmb { c } _ { d ^ { \prime } } \in \mathrm { R a n g e } ( \Gamma _ { d ^ { \prime } } )$ : if ${ \pmb w } _ { 0 } \in \ker ( \Gamma _ { d ^ { \prime } } )$ , then $\begin{array} { r } { \sum _ { t } \mathbb { E } ^ { \star } [ ( \pmb { w } _ { 0 } ^ { \top } \pmb { x } _ { t , d ^ { \prime } } ) ^ { 2 } ] = 0 } \end{array}$ , hence ${ \pmb w } _ { 0 } ^ { \top } { \pmb x } _ { t , d ^ { \prime } } = 0$ almost surely for every t and therefore ${ \pmb w } _ { 0 } ^ { \top } { \pmb c } _ { d ^ { \prime } } = 0$ . Thus the minimum is

$$
Q _ { d ^ { \prime } } = M ^ { \star } - { \pmb { c } } _ { d ^ { \prime } } ^ { \top } \Gamma _ { d ^ { \prime } } ^ { \dag } { \pmb { c } } _ { d ^ { \prime } } .\tag{24}
$$

Here $\Gamma _ { d ^ { \prime } } ^ { \dagger }$ is the Moore–Penrose pseudoinverse.

If the candidate variance is free, after minimizing over $\beta ^ { \prime }$ the KL equals

$$
\frac { 1 } { 2 } \left[ n _ { \mathrm { P 2 } } \log \frac { s ^ { 2 } } { ( \sigma ^ { \star } ) ^ { 2 } } + \frac { n _ { \mathrm { P 2 } } ( \sigma ^ { \star } ) ^ { 2 } + Q _ { d ^ { \prime } } } { s ^ { 2 } } - n _ { \mathrm { P 2 } } \right] .\tag{25}
$$

Differentiating with respect to $s ^ { 2 }$ gives $s _ { d ^ { \prime } , \mathrm { o p t } } ^ { 2 } = ( \sigma ^ { \star } ) ^ { 2 } + Q _ { d ^ { \prime } } / n _ { \mathrm { P 2 } }$ . Substitution gives the logarithmic expression.

## C.7 Partial-correlation interpretation

In a centered stationary P2 process with nondegenerate residual variances, all moments below are taken under the true stationary law. Define

$$
Z _ { t } = U _ { t - d ^ { \star } } , \quad W _ { t } = U _ { t - d ^ { \prime } } , \quad V _ { t } = Y _ { t - 1 } .\tag{26}
$$

Because $a ^ { \star } V _ { t }$ is already in the incorrect candidate span span $\{ V _ { t } , W _ { t } \}$

$$
\begin{array} { r l } & { \underset { a ^ { \prime } , b ^ { \prime } } { \operatorname* { i n f } } \mathbb { E } ^ { \star } [ ( a ^ { \star } V _ { t } + b ^ { \star } Z _ { t } - a ^ { \prime } V _ { t } - b ^ { \prime } W _ { t } ) ^ { 2 } ] } \\ & { } \\ & { \quad \quad = ( b ^ { \star } ) ^ { 2 } \underset { c _ { 1 } , c _ { 2 } } { \operatorname* { i n f } } \mathbb { E } ^ { \star } [ ( Z _ { t } - c _ { 1 } V _ { t } - c _ { 2 } W _ { t } ) ^ { 2 } ] . } \end{array}\tag{27}
$$

Residualize $Z _ { t }$ and $W _ { t }$ against $V _ { t } ,$ , obtaining $\widetilde { Z } _ { t }$ and $\widetilde { W _ { t } }$ . One-dimensional projection then gives

$$
\frac { Q _ { d ^ { \prime } } } { n _ { \mathrm { P 2 } } } = ( b ^ { \star } ) ^ { 2 } \operatorname { V a r } ( \widetilde { Z } _ { t } ) ( 1 - \rho _ { Z W \cdot V } ^ { 2 } ) .\tag{28}
$$

Here $\rho _ { Z W } . V$ is the partial correlation between $Z _ { t }$ and $W _ { t }$ given $V _ { t } .$ . This separates compensation through the previous output from similarity between the true and incorrect input lags after controlling for that previous output.

## C.8 Proof of Theorem B.6

By Equation (5) and the definition of $\gamma .$

$$
\left\| \mathbf { U } ( h - h ^ { \prime } ) \right\| _ { 2 } \geq \gamma ( \mathbf { U } , \mathcal { H } ) \left\| h - h ^ { \prime } \right\| _ { 2 } .\tag{29}
$$

Squaring and dividing by $2 \sigma ^ { 2 }$ proves the lower bound. Since H is finite, the minimum is attained. It is zero iff a distinct pair has $\mathbf { U } h = \mathbf { U } h ^ { \prime }$

## C.9 Proof of Theorem 4.1

For two equal-prior Gaussians with covariance $\sigma ^ { 2 } I _ { n }$ , the squared Mahalanobis distance is $\Delta _ { M , n } ^ { 2 } = \parallel \mu _ { 0 , n } -$ $\mu _ { 1 , n } \| _ { 2 } ^ { 2 } / \sigma ^ { 2 } = 2 D _ { n }$ . The equal-prior MAP error is therefore

$$
\Phi ( - { \Delta _ { M , n } } / { 2 } ) = \Phi ( - \sqrt { D _ { n } / 2 } ) ,
$$

which tends to zero because $D _ { n } \to \infty$ . Under candidate 0, write $Y = \mu _ { 0 , n } + \epsilon$ with $\epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { n } )$ . If the numerical predictor uses $\mu _ { 1 , n } .$ , then

$$
\begin{array} { l } { \displaystyle \frac { 1 } { n } \mathbb { E } \| Y - \mu _ { 1 , n } \| _ { 2 } ^ { 2 } = \sigma ^ { 2 } + \frac { 1 } { n } \| \mu _ { 0 , n } - \mu _ { 1 , n } \| _ { 2 } ^ { 2 } } \\ { \displaystyle = \sigma ^ { 2 } \left( 1 + \frac { 2 D _ { n } } { n } \right) , } \end{array}
$$

which converges to the oracle risk $\sigma ^ { 2 }$ because $D _ { n } / n \to 0$

For the P1 specialization, $\mu _ { j , n } = \alpha S _ { d _ { j } } u ^ { ( n ) }$ . If $S _ { d _ { 0 } } ~ \neq ~ S _ { d _ { 1 } }$ , choose any $z ^ { ( n ) }$ with $\xi _ { n } : = \parallel ( S _ { d _ { 0 } } -$ $S _ { d _ { 1 } } ) z ^ { ( n ) } \Vert _ { 2 } ^ { 2 } > 0$ and set

$$
u ^ { ( n ) } : = \sqrt { \frac { 2 \sigma ^ { 2 } D _ { n } } { \alpha ^ { 2 } \xi _ { n } } } z ^ { ( n ) } .
$$

Then $\alpha ^ { 2 } \| ( S _ { d _ { 0 } } - S _ { d _ { 1 } } ) u ^ { ( n ) } \| _ { 2 } ^ { 2 } / ( 2 \sigma ^ { 2 } ) = D _ { n }$ exactly. The forecast map $u \mapsto \alpha S _ { d _ { 1 } }$ u remains tied to the wrong lag regardless of the MAP report, so on the event that the report equals $d _ { 0 }$ the report–use distance is $| d _ { 0 } - d _ { 1 } |$ that event has probability tending to one.

## C.10 Proof of Theorem 4.2

The no-bypass claim is immediate from the factorization in Equation (6). Fix r and let the vector perturbation $\Delta u$ be supported only on coordinates k with $g _ { k } ( r ) = 0$ . Then

$$
\begin{array} { r } { \mathbf { g } ( r ) \odot ( u + \Delta u ) = \mathbf { g } ( r ) \odot u , } \end{array}
$$

so the arguments supplied to $G$ are unchanged. For the Lipschitz bound, with the report fixed and scalar intervention amplitude $\delta ,$

$$
\begin{array} { r } { \| \mathbf { g } ( r ) \odot ( u + \delta e _ { k } ) - \mathbf { g } ( r ) \odot u \| = | g _ { k } ( r ) | | \delta | . } \end{array}
$$

The Lipschitz property of G gives the stated bound. A one-hot gate is the special case in which all off-report gate entries are zero. None of these steps lower-bounds sensitivity on the selected coordinates, so nontrivial use requires a separate nondegeneracy condition.

## C.11 Proof of Theorem 4.3

For a one-hot gate $\mathbf { g } ( r ) = e _ { r }$ , Theorem 4.2 gives $A _ { k } = 0$ for every $k \neq r$ under the fixed-report intervention.   
The assumption $A _ { r } > 0$ therefore makes $r$ the unique maximizer of the intervention-response magnitude.   
Hence the peak-based point-delay report–use distance is zero.

## D Bayes Oracle Computation

We use the standard Bayes decision rule for finite candidate classes [Devroye et al., 1996, Tsybakov, 2009]. Let $h _ { 0 }$ denote the realized observed initial history when one is required. For each fixed observed $( u , h _ { 0 } )$ and candidate $j \colon$

1. construct the conditional mean/covariance or sequential innovation likelihood on the common valid window, including any candidate-dependent initial-history contribution when required;

2. compute $q _ { j } ( y ) = \log \pi _ { j } ( u , h _ { 0 } ) + \log p _ { j } ^ { ( u , h _ { 0 } ) } ( y ) ;$

3. use log-sum-exp to obtain posterior probabilities;

4. select the Bayes rule $\psi ^ { \star }$ minimizing posterior expected loss;

5. estimate Bayes risk from B independent draws $H ^ { ( b ) } \sim \pi ( \cdot \mid u , h _ { 0 } )$ and $Y ^ { ( b ) } \sim P _ { H ^ { ( b ) } } ^ { ( u , h _ { 0 } ) }$

For 0–1 loss,

$$
\widehat { R } _ { \mathrm { B a y e s } } ^ { \mathrm { M C } } ( u , h _ { 0 } ) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \mathbf { 1 } \{ \psi ^ { \star } ( Y ^ { ( b ) } ; u , h _ { 0 } ) \neq H ^ { ( b ) } \} .\tag{30}
$$

Writing $\widehat { R } = \widehat { R } _ { \mathrm { B a v e s } } ^ { \mathrm { M C } } ( u , h _ { 0 } )$ , the Monte Carlo standard error is $\sqrt { \widehat { R } ( 1 - \widehat { R } ) / B }$ [Robert and Casella, 2004]. An outer bootstrap over input realizations estimates scenario-average uncertainty [Efron and Tibshirani, 1993].

## E Detailed Synthetic Generator Specification

## E.1 Unified generation order

For each trajectory, the implementation follows the fixed order:

1. sample the input family and generate $U _ { \mathrm { 1 : } T _ { \mathrm { r a w } } }$

2. sample the structural candidate and dynamic nuisance parameters;

3. generate the clean latent process and apply burn-in removal;

4. generate innovations and observation corruption using independent random streams;

5. crop every candidate to the common task window; for the released P2 profile audit, additionally drop the first post-boundary transition and use $\mathcal { T } _ { L } ^ { \mathrm { P 2 } }$ so that every lagged output regressor is actually observed;

6. compute P1/P3 KL/Bayes quantities and the repaired P2 finite-sample profile quantities on their declared windows; exact models with nonzero output memory must condition on any required observed boundary history;

7. assign a difficulty bin and write the complete metadata record;

8. split by complete trajectory and parameter-combination group.

The raw length includes a burn-in chosen from the slowest declared state decay and the maximum candidate lag. Numerical stationarity is checked empirically in addition to analytic stability conditions.

## E.2 Reference configuration

Unless a factor is being scanned, the reference instance uses

$$
T = 5 1 2 , d = 8 , \mathrm { S N R } = 1 0 \mathrm { d B } , \rho _ { U } = 0 . 5 , a = 0 . 6 .\tag{31}
$$

For P3, the reference kernel is a normalized two-path kernel with spacing four and secondary mass 0.3. The reference stress track uses Gaussian innovations and no corruption; each stress experiment changes one declared module.

## E.3 Stable ARX(2) templates

Rather than sampling arbitrary coefficients, the initial release uses a small validated template library. Each pair $( a _ { 1 } , a _ { 2 } )$ is accepted only if the roots of the state-transition polynomial imply a stable recursion. Templates are selected to produce fast monotone, slow monotone, and mildly oscillatory impulse responses. The exact released coefficients and their pole locations are stored in the scenario configuration.

## E.4 Kernel construction

Point, multipath, triangular, exponential, and biphasic kernels are first constructed on support $\{ 0 , \ldots , L \}$ and then normalized. For two-path kernels, with overall kernel gain $b _ { h }$ and secondary-path weight $w \in [ 0 , 1 ]$

$$
h _ { k } = b _ { h } \left[ ( 1 - w ) \mathbf { 1 } \{ k = d _ { 1 } \} + w \mathbf { 1 } \{ k = d _ { 2 } \} \right] .\tag{32}
$$

For exponential kernels with decay constant $\tau _ { h } > 0$

$$
h _ { d + \ell } \propto \exp ( - \ell / \tau _ { h } ) , \qquad \ell \ge 0 .\tag{33}
$$

Matched kernel pairs are accepted only after checking that their parameter distance and realized meanresponse distance fall in the pre-specified range.

## E.5 Noise and corruption modules

Student-t innovations are rescaled to the target variance when the variance exists. Contaminated Gaussian noise uses a stored contamination indicator. Colored innovations are generated before the process recursion and are not confused with output autoregression. Missingness, quantization, and saturation are applied only after a clean latent output has been generated. Both clean and corrupted outputs are released.

## E.6 Physics-inspired modules

Cascade, parallel-path, and sensor-smoothing systems are simulated in state-space form with explicit states. The benchmark label contains both the physical path delays and an optional summary kernel derived from an impulse response. Structural scoring uses the physical labels; functional scoring may additionally use the realized impulse-response kernel. This prevents an effective-lag summary from replacing the declared physical target.

## E.7 Generator validation

Each configuration must pass:

• stability and finite-value checks;

• target versus empirical SNR checks;

• candidate-label independence from input metadata and array shape;

• empirical autocorrelation and spectrum sanity checks;

• impulse-response verification for every mechanism family;

• paired-noise reproducibility and random-seed replay;

• shortcut tests using input-only and metadata-only classifiers.

A scenario family is excluded from the main benchmark if a metadata-only classifier predicts its candidate label above the pre-specified tolerance.

## F Detailed Experimental Sampling Protocol

## F.1 One-factor and interaction sweeps

The main severity study varies one factor from the reference configuration. A fixed set of two-factor interactions is also evaluated for robustness; the exact interaction definitions and parameter ranges are part of the released scenario configuration rather than introduced as additional paper notation. All remaining combinations are sampled from a fixed configuration distribution and are used for robustness rather than exhaustive factorial claims.

## F.2 Difficulty-targeted sampling

For exact-tier instances, a proposal is repeatedly generated until its oracle quantity falls in the target bin. The proposal distribution and maximum rejection count are fixed before benchmark execution. If a bin cannot be populated without extreme parameters, the bin is marked unsupported rather than filled by changing its definition.

## F.3 Minimum reporting units

The statistical unit is a complete trajectory. Training seeds are nested within model configurations and trajectories are nested within scenarios. Scenario-level summaries are reported alongside pooled trajectory metrics to prevent large easy scenario families from dominating the leaderboard.

Held-out cohorts. Diagonal lines are reference equality, not fitted trends; P2 uses a profile-margin ordering rather than a Bayes identity

## G Supplementary Empirical Evidence

## G.1 Main-text validation and secondary comparisons

The following validation figure and secondary numerical results are kept here to keep the main text focused on the central argument.

![](images/c9f6979072df5545aaf7717a0e5d10e43f3bdce0550bbe17c3ca1dadd087701a.jpg)

![](images/ecc530e3c441618973c2f37a5b6d0c7dafb106482aa008cd4556296197e0396e.jpg)

![](images/bba9c97d8123e3d731625c785f3e93bac17217687d5900d9860899534ae46867.jpg)  
Figure 1: Stage-I calibration. P1 empirical reference error tracks analytic Bayes risk; P2 error decreases with nuisance-profile separation; P3 structural error is ordered by numerical Bayes risk. P2 is an ordering diagnostic rather than a Bayes identity.

Table 9: P3 full-kernel effects: shared gate minus separate head. Intervals are trajectory-bootstrap 95% CIs.
<table><tr><td>Backbone Family</td><td></td><td> $\Delta \ell _ { 2 }$ </td><td>∆ cosine</td><td>∆ path recall</td></tr><tr><td>TCN</td><td>Multipath</td><td>-.882[-.938, -.829]</td><td> $+ . 6 7 9 \ [ + . 6 4 0 , + . 7 1 8 ]$ </td><td> $+ . 2 2 0 [ + . 1 7 4 , + . 2 6 3 ]$ </td></tr><tr><td>N-HiTS</td><td>Multipath</td><td>−.464[−.491, −.436] +.297[+.278, +.319]</td><td></td><td> $- . 0 6 2 \ : [ - . 1 3 7 , + . 0 1 3 ]$ </td></tr><tr><td>TCN</td><td>Distributed</td><td>1 −.798 [−.903, −.692] +.542 [+.464, +.616]</td><td></td><td></td></tr><tr><td>N-HiTS</td><td></td><td>Distributed −.484[−.515, −.450] +.265 [+.238, +.288]</td><td></td><td></td></tr></table>

This appendix preserves the audit evidence that is necessary for reproducibility but would obscure the main empirical narrative. It includes exact numerical validation, stress-tier functional results, failed-andcorrected audits, and sample-scaling diagnostics. Each subsection names its release and cohort; the release map in Table 24 prevents cross-release pooling.

## G.2 Numerical validation and frozen references

Table 10 collects the frozen numerical checks and reference errors underlying the compact calibration plot in the main text.

P1’s analytic binary Bayes error and its Monte Carlo estimate agree within the pre-specified three-Monte-Carlo-standard-error tolerance across the calibrated difficulty range. The P1 balanced set contains 480 trajectories, 80 in each of six Bayes-risk bins. Its frozen raw and difficulty-balanced risks coincide by construction: 0.2042 for Gaussian MAP, 0.2021 for cross-correlation, 0.2458 for prewhitened crosscorrelation, and 0.2542 for ridge lag regression.

Table 10: Exact-tier theory-calibration checks. Error is the observed fraction of wrong delay or kernel selections.
<table><tr><td>Task</td><td>Sample</td><td>Axis</td><td>Implementation check</td><td>Reference error</td></tr><tr><td>P1</td><td>480</td><td>Bayes risk</td><td>Analytic/Monte Carlo agreement within 3 MC s.e.</td><td>MAP .204; xcorr .202</td></tr><tr><td>P2</td><td>100</td><td>Corrected profile margin</td><td>Max. closed-form/optimizer rela- tive error 1  $\phantom { + } 1 . 0 3 \times 1 0 ^ { - 1 5 }$ </td><td>2/400 favorable labels changed</td></tr><tr><td>P3</td><td>250</td><td>Observability</td><td>Zero identity and KL lower bound verified</td><td>Bayes .113; MLE .116; ridge .248</td></tr></table>

The repaired P2 profile implementation was independently recomputed on 100 configurations using the corrected window $\hat { T } _ { L } ^ { \hat { \mathrm { P 2 } } }$ . Closed-form linear algebra and iterative optimization agree to maximum relative error $1 . 0 2 9 \times 1 0 ^ { - 1 5 }$ . Rejoining the frozen common-cohort predictions changes favorable membership for 2 of 400 trajectories while leaving the favorable count at 244; profile ARX and PEM-ARX favorable errors remain .004 and .049, N-HiTS changes from .497 to .500, TCN from .251 to .250, and MLP-NARX from .781 to .783. The repair therefore removes the boundary-contract mismatch without changing the empirical ordering. P3 contains 250 scenarios, 50 in each of five observability bins. The finite-library Bayes, finite-library MLE, and ridge-kernel risks are 0.1132, 0.1160, and 0.2480. The release also verifies equal realized means for zero-observability candidates and the declared KL lower bound for separated candidates.

## G.3 Confirmatory P3 Bayes-easy audit

We compute a finite-library multicandidate Bayes risk once for each of the 400 trajectory IDs in the unified P3 cohort and join the resulting membership to every backbone and readout. Monte Carlo uses 2,000 draws by default and 20,000 draws within a prespecified margin of any reported threshold. Table 11 reports the primary $R _ { \mathrm { B a y e s } } ^ { \star } \leq 0 . 0 5$ subset; the same qualitative ordering holds at thresholds 0.01, 0.10, and 0.15. The point-estimate Bayes-easy set contains 230 trajectories for every system and has Jaccard overlap 0.406 with the single trajectory-level observability-easy set. Five trajectories have Monte Carlo intervals that cross the 0.05 threshold (226 are easy by the upper-confidence rule and 231 by the lower-confidence rule), so sensitivity membership is reported separately rather than changed by backbone. This demonstrates why observability is explanatory rather than a complete decision limit.

Table 11: P3 structural error on the numerical Bayes-easy subset. Intervals are 95% hierarchical-bootstrap intervals over trajectories and three training seeds; n counts distinct trajectories.
<table><tr><td>Backbone</td><td>Readout</td><td>Error [95% CI]</td><td>n</td></tr><tr><td>TCN</td><td>frozen probe</td><td>.868 [.832, .899]</td><td>230</td></tr><tr><td>TCN</td><td>separate head</td><td>.225 [.184, .265]</td><td>230</td></tr><tr><td>TCN</td><td>shared gate</td><td>.235 [.196, .278]</td><td>230</td></tr><tr><td>N-HiTS</td><td>frozen probe</td><td>.865 [.833, .896]</td><td>230</td></tr><tr><td>N-HiTS</td><td>separate head</td><td>.691 [.645, .738]</td><td>230</td></tr><tr><td>N-HiTS</td><td>shared gate</td><td>.681 [.635, .726]</td><td>230</td></tr></table>

## G.4 Oracle-normalized forecast context

Tables 12 and 13 separate raw forecast units from oracle normalization. NEFE uses $\mathrm { N E F E } = \mathrm { ( M S E _ { m o d e l } - }$ $\mathrm { M S E _ { o r a c l e } ) / ( M S E _ { m e a n } - M S E _ { o r a c l e } ) }$ . The mean and persistence predictors are computed on the exact fixed test trajectories. P1 and P2 provide stable normalization; the smaller P3 denominator produces wide ratio intervals, so paired raw-MSE differences remain primary there.

Table 12: Raw forecast MSE on the frozen NEFE audit cohort. Oracle, mean, and persistence are direct references; soft-routed, TCN, and N-HiTS are model forecasts. Lower is better.
<table><tr><td>Task</td><td>Oracle</td><td>Mean</td><td>Persistence</td><td>Soft-routed</td><td>TCN-sep.</td><td>N-HiTS-sep.</td></tr><tr><td>P1</td><td>1.323</td><td>1.864</td><td>2.979</td><td>1.548</td><td>1.858</td><td>1.725</td></tr><tr><td>P2</td><td>.290</td><td>1.306</td><td>.505</td><td>.708</td><td>.872</td><td>.544</td></tr><tr><td>P3</td><td>.826</td><td>.992</td><td>1.505</td><td>.931</td><td>.956</td><td>.881</td></tr></table>

Table 13: Oracle-normalized forecast error (NEFE) for learned systems on the same cohort. Lower is better; raw paired MSE remains primary for P3 because its normalization denominator is small.
<table><tr><td>Task</td><td>Soft-routed</td><td>TCN-sep.</td><td>N-HiTS-sep.</td></tr><tr><td>P1</td><td>.415</td><td>.989</td><td>.742</td></tr><tr><td>P2</td><td>.412</td><td>.573</td><td>.250</td></tr><tr><td>P3</td><td>.634</td><td>.785</td><td>.333</td></tr></table>

Table 14: Primary paired M8 ranking contrasts over 1,200 matched trajectories and three training seeds. Differences are listed in the comparison direction; 95% hierarchical-bootstrap intervals resample trajectories and seeds.
<table><tr><td>Task</td><td>Comparison</td><td>Difference [95% CI]</td></tr><tr><td>P1</td><td>TCN — N-HiTS structural error</td><td>-.077  $[ - . 1 0 1 , - . 0 5 3 ]$ </td></tr><tr><td>P1</td><td>N-HiTS – TCN forecast MSE</td><td>-.158[-.222, −.093]</td></tr><tr><td>P2</td><td>TCN — N-HiTS structural error</td><td>-.111  $[ - . 1 4 1 , - . 0 8 1 ]$ </td></tr><tr><td>P2</td><td>N-HiTS – TCN forecast MSE</td><td> $- . 1 7 5 \ [ - . 3 1 1 , - . 0 6 6 ]$ </td></tr><tr><td>P3</td><td>TCN — N-HiTS structural error</td><td> $- . 1 6 7 \ [ - . 1 9 5 , - . 1 3 9 ]$ </td></tr><tr><td>P3</td><td>N-HiTS – TCN forecast MSE</td><td> $- . 1 0 2 [ - . 1 5 3 , - . 0 5 9 ]$ </td></tr></table>

Table 14 is an independent architecture-replication contrast and is not pooled with the unified commoncohort table in the main text. The P2 favorable-error entry in Table 21 is intentionally omitted because the corrected boundary-window membership was not replayed for that independent release; its overall structural error, MSE, and functional gap are unaffected.

## G.5 Functional consistency and negative controls

Functional-use terminology. Throughout the paper, “use” means controlled sensitivity of the fitted numerical predictor to historical input locations under the fixed intervention diagnostics. It is intentionally narrower than a claim of real-world causality or internal mechanistic equivalence. For the binary P1 audit, “unused” is the matched-window masking label defined in the main text: the reported ±1-lag window is zeroed and compared with 20 disjoint equal-width energy-matched control windows. This label is computed before any conditioning on report correctness or NER and is held fixed in the triple-separation analysis. Additive perturbations and Jacobians are separate continuous robustness diagnostics. The target profile is tied to a known mechanism because the exact and physics-inspired tiers expose the declared delay or kernel used by the generator.

Before interpreting learned-model scores, we test the diagnostics on seven predictors with known functional structure: a true-delay oracle, a deliberately shifted delay, a true two-path rule, a similar-error wrong kernel, an output-only rule, a uniform-history rule, and an exact hard one-hot replay. On 400 fresh trajectories per predictor, additive sensitivity and input Jacobians recover nonuniform full profiles with cosine similarity numerically equal to one and zero peak error, while all three diagnostics correctly reject input use for the output-only predictor. Matched donor-block replacement localizes single paths to within 0.85–0.88 lags on average, but recalls only 0.620 of weak secondary paths for the two-path rule and 0.423 for hard one-hot replay. It is therefore retained as a qualitative robustness check, not the primary full-kernel diagnostic; its uniform-history blind spot is likewise explicitly recorded.

The complete exact- and stress-tier diagnostic summaries are reported in Tables 15 and 16.

Table 15: Exact-tier functional consistency. Gaps are mean absolute distances between the reported delay (kernel peak for P3) and the functional response peak; brackets are 95% bootstrap confidence intervals.
<table><tr><td>System</td><td>P1 gap</td><td>P1 unused correct</td><td>P2 gap</td><td>P3 peak gap</td></tr><tr><td>xcorr</td><td>0.290 [0.050, 0.585]</td><td>0.055</td><td>0.240 [0.000, 0.530]</td><td>0.070 [0.010, 0.150]</td></tr><tr><td>profile/finite ref.</td><td>0.320 [0.045, 0.726]</td><td>0.064</td><td>0.470 [0.140, 0.890]</td><td>0.105 [0.005, 0.240]</td></tr><tr><td>lag regression</td><td>4.020 [3.010, 5.095]</td><td>0.394</td><td>0.155 [0.000, 0.415]</td><td>2.135 [1.650, 2.660]</td></tr><tr><td>TCN</td><td>14.930 [13.745, 16.020]</td><td>0.891</td><td>14.285 [13.075, 15.486]</td><td>4.870 [4.415, 5.390]</td></tr><tr><td>N-HiTS</td><td>4.400 [3.390, 5.340]</td><td>0.525</td><td>6.025 [5.040, 7.080]</td><td>2.765 [2.410, 3.125]</td></tr><tr><td>iTransformer</td><td>6.900 [5.640, 8.171]</td><td>0.651</td><td>11.010 [9.539, 12.466]</td><td>2.590 [2.325, 2.840]</td></tr></table>

Table 16: P1 stress-tier functional consistency. Gaps include 95% bootstrap confidence intervals; the unused rate conditions on a correct structural report. Each model–tier cell uses 200 trajectories.
<table><tr><td>System</td><td>Mean report-use gap</td><td>Reported history unused | correct report</td><td></td></tr><tr><td>xcorr</td><td>1.575 [0.905, 2.360]</td><td>0.132</td><td></td></tr><tr><td>profile ARX</td><td>0.925 [0.370, 1.540]</td><td>0.104</td><td></td></tr><tr><td>lag regression</td><td>8.470 [7.250, 9.700]</td><td>0.645</td><td></td></tr><tr><td>TCN</td><td>10.085 [9.050, 11.215]</td><td>0.882</td><td></td></tr><tr><td>N-HiTS</td><td>7.250 [6.090, 8.500]</td><td>0.705</td><td></td></tr><tr><td>iTransformer</td><td>9.360 [7.965, 10.890]</td><td>0.725</td><td></td></tr></table>

The functional conclusions pass the pre-specified stability check at additive amplitudes 0.10, 0.25, and 0.50 standardized input units. P2 and P3 exact-tier confidence intervals are reported in Table 15; complete response kernels, rather than only their peaks, are retained in the release artifacts.

Joint-subset diagnostic robustness. We additionally freeze the exact Link-II subset used in the main joint test (177 N-HiTS and 248 TCN seed–trajectory rows) and recompute continuous functional responses without retraining. The binary “unused” label itself is masking-based, so it does not depend on additive amplitude. The additive and Jacobian diagnostics below instead test whether the same locked rows continue to place their functional response away from the reported delay. Table 17 shows that peak locations are highly stable across amplitudes; additive .25 versus Jacobian report–use gaps have Spearman correlation .982 for N-HiTS and .907 for TCN, with exact peak agreement .989 and .734, respectively. Finite perturbations and local derivatives need not agree exactly, but both show the same substantial report–use mismatch on the identical locked subset.

Table 17: Functional-diagnostic robustness on the fixed correct-and-near-oracle P1 subset. Gap is the mean absolute distance between the reported delay and the response peak; brackets are 95% hierarchical-bootstrap intervals. No Jacobian binary-use threshold is introduced.
<table><tr><td>Backbone</td><td>Diagnostic</td><td>Mean report-use gap [95% CI] Peak = report [95% CI]</td></tr><tr><td rowspan="5">N-HiTS</td><td>Additive .10</td><td>9.05 [7.47,10.74]</td></tr><tr><td>Additive .25 9.10 [7.51,10.80]</td><td>.243 [.178,.306] .243 [.178,.310]</td></tr><tr><td>Additive .50</td><td>8.98 [7.30,10.73] .243 [.181,.310]</td></tr><tr><td>Jacobian</td><td>9.16 [7.60,10.82] .243 [.181,.310]</td></tr><tr><td>Additive .10</td><td>.016 [.000,.038]</td></tr><tr><td rowspan="4">TCN</td><td>Additive .25</td><td>13.15 [11.90,14.45] 13.06 [11.77,14.27]</td><td>.012 [.000,.028]</td></tr><tr><td>Additive .50</td><td>12.71 [11.68,13.75]</td><td>.008 [.000,.023]</td></tr><tr><td>Jacobian</td><td>13.28 [12.09,14.52]</td><td>.028 [.000,.059]</td></tr><tr><td></td><td></td><td></td></tr></table>

Alignment-penalty control. A validation-selected differentiable alignment penalty produced no stable gain: all four paired report–use intervals crossed zero. This negative control narrows the remedy claim to routing predictive information through the reported structure; it does not support the tested auxiliary regularizer.

## G.6 Stress and physics-inspired robustness

The stress tier changes one assumption at a time: P1 Gaussian outliers, a P2 unmodeled second autoregressive term, and P3 sensor smoothing. The physics-inspired tier contains 900 trajectories from cascaded units, parallel paths, and process delay with sensor smoothing. Figure 2 reports structural error rather than transferring exact-tier Bayes claims to misspecified models.

The three stress curves have different shapes and ranges, so exact-tier averages do not summarize robustness. The matched candidate MLE remains strong on the declared physics generators because it receives the correct simulator family and candidate library, whereas correlation rules fail especially on broadened, correlated parallel paths. These results validate the intended robustness tiers; they do not claim industrial realism or a universal oracle.

## G.7 Structural OOD and real-input semi-synthetic transfer

We test two different boundaries. Parameter-combination and candidate-space holdouts ask whether a structural head extrapolates beyond random ID splits; hydraulic and Beijing air-quality histories ask whether conclusions survive realistic input autocorrelation and regimes while the P1–P3 output mechanisms retain known ground truth.

Parameter-combination holdouts expose structural-risk increases hidden by random trajectory splits. Candidate-set and continuous-kernel readouts show mixed extrapolation rather than universal improvement: size-seven candidate sets degrade modestly, N-HiTS improves on a held-out smooth kernel family, and both backbones degrade sharply on held-out sparse kernels (Table 18).

The real-input audit contains 28,800 complete records derived from hydraulic [Helwig et al., 2015], Beijing air-quality [Chen, 2017], and synthetic-control inputs with group-safe or chronological splits. Direct within-source contrasts are reported in Table 8; coupled heads preserve their report–use advantage, while forecast effects remain task dependent.

To separate source shift from marginal difficulty, each supported real-input trajectory is also matched to its nearest synthetic trajectory on the task-specific score. The bootstrap resamples real trajectories, synthetic donors, and training seeds and rematches at every draw. The previously named “calibration residual” was a signed per-instance residual, so it is not reported as an absolute calibration gap. A corrected grouplevel absolute-gap audit is secondary: several source–model intervals cross zero, while P1 hydraulic and selected N-HiTS cells show residual calibration shift. Retained support and matching quality are reported in Table 19. The robust transfer conclusion therefore comes from direct routing contrasts, not from a universa calibration-transfer claim.

Stress data (top): different data problems produce different error curves (a) P1: large noise outliers (b) P2: extra previous-output term  
![](images/6270194a2631151b64ae9acc36a0fd663267afb7940b724dc2dd49cc6def96d4.jpg)

![](images/c781553aa42d799ebb9249aea403a138a69f2e2f3aa5171f4e21803642c39b5c.jpg)

(c) P3: sensor averaging  
![](images/4936033d485fc3b55aca73f2c1891c40d9b63c5853325f56383114bd88730e27.jpg)  
Mechanism-based data (bottom): simulator-matched MLE versus correlation

(d) Several stages in series  
![](images/d60eb17101dbb6c947ceeb96fd9dcdfb70a308f9b45788ac624f6fc6b87b9e83.jpg)

(e) Two parallel paths  
![](images/f74413bc4bb741d8da5321d86e880219ea985794fe1d37b3cfaa3bf8c3a4aad6.jpg)

(f) Sensor averaging  
![](images/272fa0552fb7848d27dea2afbf9bf717f5ba867d4168139bfe125aaef32f5bb7.jpg)  
Figure 2: Mechanism-specific robustness profiles; lower structural error is better. Top: 120 trajectories per severity show distinct degradation under P1 outliers, a P2 unmodeled dynamic term, and P3 sensor smoothing. Bottom: cascades, parallel paths, and sensor dynamics compare correlation rules with a simulator-matched candidate MLE. Empirical zeros indicate no errors in the finite sample, not zero population risk; matched MLE is a generator-aware sanity reference rather than a general-purpose model.

Table 18: Exploratory structural OOD effects relative to the corresponding ID split. Positive error is degradation; negative kernel $\ell _ { 2 }$ is improvement. Intervals are 95% bootstrap intervals over 1,200 trajectories and three seeds.
<table><tr><td>Setting</td><td>Task/backbone</td><td>Metric</td><td>OOD – ID [95% CI]</td></tr><tr><td>Candidate set size 7</td><td>P1</td><td>Error</td><td>+.023 [+.007, +.039]</td></tr><tr><td>Candidate set size 7</td><td>P2</td><td>Error</td><td>+.037 [+.020, +.052]</td></tr><tr><td>Held-out smooth kernel</td><td>N-HiTS</td><td>Kernel l2</td><td>-.101 [−.117, -.085</td></tr><tr><td>Held-out smooth kernel</td><td>TCN</td><td></td><td>Kernel l2 +.007 [−.017, +.027]</td></tr><tr><td>Held-out sparse kernel</td><td>N-HiTS</td><td></td><td>Kernel l2 +.316 [+.302, +.331]</td></tr><tr><td>Held-out sparse kernel</td><td>TCN</td><td></td><td>Kernel l2 +.366 [+.353, +.378]</td></tr></table>

Table 19: Common-support diagnostics for difficulty-matched real-input transfer. Coverage is the retained fraction of real trajectories; mismatch is the mean absolute matched-score difference divided by the synthetic IQR.
<table><tr><td>Task</td><td>Source</td><td>Coverage</td><td>n</td><td>Normalized mismatch</td></tr><tr><td>P1</td><td>Hydraulic</td><td>.755</td><td>151</td><td>.063</td></tr><tr><td>P1</td><td>Beijing</td><td>.905</td><td>181</td><td>.110</td></tr><tr><td>P2</td><td>Hydraulic</td><td>.730</td><td>146</td><td>.007</td></tr><tr><td>P2</td><td>Beijing</td><td>.895</td><td>179</td><td>.002</td></tr><tr><td>P3</td><td>Hydraulic</td><td>.525</td><td>105</td><td>.029</td></tr><tr><td>P3</td><td>Beijing</td><td>.745</td><td>149</td><td>.006</td></tr></table>

## G.8 Failed-and-corrected audit chain

The initial TCN negative-control audit searched only within candidate lags 1–32. This produced approximately 65% median relative control-energy mismatch and was correctly retained as a failed audit. The corrected protocol searches the complete observed history for an equal-width, nonoverlapping, energy-matched window. Using new data and seeds 4444, 5555, and 6666, the maximum median control energy error is 0.215, below the pre-specified 0.25 threshold; the unused-rate seed range is 0.165, below its 0.20 threshold. The three unused-given-correct rates are 0.776, 0.941, and 0.905. Forecast output standard deviation is at least 0.282, and the maximum forecast-MSE to mean-baseline-MSE ratio is 0.953, ruling out a constant or entirely invalid numerical forecast. These checks support the confirmatory TCN result only; they do not justify a universal statement about the model class.

The stress-tier audit similarly retains the initial failed run, the pilot, and the fresh-seed confirmatory result. The confirmatory design uses 120 trajectories per severity point, exact replay difference zero, and yields P1/P2/P3 curve ranges 0.275/0.125/0.533. The numerical distance between the three normalized failure curves is 0.414.

## G.9 Neural sample-scaling diagnostics

The first 900-trajectory neural experiment left N-HiTS and iTransformer near random performance. A 32-sample overfit test subsequently reached 100% memorization in all nine task–model combinations, ruling out label misalignment and a failure of the separate candidate-probability output to receive training signals as explanations. The initial architecture replication therefore uses 16,200 trajectories per task. For iTransformer on P3, favorable-region error decreases from $0 . 6 5 3 \pm 0 . 0 2 5$ at 8,100 samples to $0 . 4 9 5 \pm 0 . 0 3 8$ at 16,200 and $0 . 4 0 6 \pm 0 . 0 1 9$ at 32,400. This experiment is a sample-efficiency diagnostic, not a main leaderboard comparison.

Table 20 shows that, on the fixed unified protocol, all four P2/P3 backbone cells improve significantly in separate-head structural error from 8,192 to 16,384 training trajectories. The reductions are .051 for both TCN tasks, .102 for P2 N-HiTS, and .145 for P3 N-HiTS; all paired 95% intervals exclude zero. Overall structural error nevertheless remains .251–.482 at the largest budget, and P3 favorable-subset error remains .106–.400. The corrected P2 favorable-membership replay was performed for the unified 4,096-sample

Table 20: Full training-size audit used to test whether favorable-subset recovery failures vanish with more data. Architecture, optimization protocol, test cohort, and seeds are fixed. Values are separate-head means; continued improvement from 8k to 16k makes the result budget-conditional rather than a saturation claim.
<table><tr><td>Task</td><td>Backbone</td><td>Train n</td><td>Error</td><td>Favorable error</td><td>MSE</td></tr><tr><td>P2</td><td>N-HiTS</td><td>4,096</td><td>.570</td><td>.500</td><td>.540</td></tr><tr><td>P2</td><td>N-HiTS</td><td>8,192</td><td>.468</td><td>一</td><td>.505</td></tr><tr><td>P2</td><td>N-HiTS</td><td>16,384</td><td>.366</td><td>一</td><td>.512</td></tr><tr><td>P2</td><td>TCN</td><td>4,096</td><td>.370</td><td>.250</td><td>.935</td></tr><tr><td>P2</td><td>TCN</td><td>8,192</td><td>.324</td><td>一</td><td>.833</td></tr><tr><td>P2</td><td>TCN</td><td>16,384</td><td>.273</td><td>一</td><td>.722</td></tr><tr><td>P3</td><td>N-HiTS</td><td>4,096</td><td>.714</td><td>.691</td><td>.912</td></tr><tr><td>P3</td><td>N-HiTS</td><td>8,192</td><td>.627</td><td>.577</td><td>.887</td></tr><tr><td>P3</td><td>N-HiTS</td><td>16,384</td><td>.482</td><td>.400</td><td>.906</td></tr><tr><td>P3</td><td>TCN</td><td>4,096</td><td>.353</td><td>.225</td><td>1.068</td></tr><tr><td>P3</td><td>TCN</td><td>8,192</td><td>.302</td><td>.158</td><td>1.056</td></tr><tr><td>P3</td><td>TCN</td><td>16,384</td><td>.251</td><td>.106</td><td>.974</td></tr></table>

closure cohort; to avoid mixing legacy and repaired memberships, the 8,192/16,384 P2 favorable-error cells are not used for a quantitative favorable-scaling claim. The sample-scaling conclusion therefore remains budget-conditional rather than a saturation or impossibility claim.

Table 21: iTransformer [Liu et al., 2024a] independent architecture replication (1,200 trajectories per task, three seeds). It is not pooled with the unified common-400 cohort. Brackets are 95% intervals.
<table><tr><td>Task</td><td>Error</td><td>Favorable error</td><td>MSE</td><td>Functional gap</td></tr><tr><td>P1</td><td>.332 [.311,.351]</td><td>.268 [.248,.291]</td><td>1.360 [1.249,1.480]</td><td>6.900 [5.640,8.171]</td></tr><tr><td>P2</td><td>.387 [.364,.410]</td><td></td><td>1.032 [.943,1.122]</td><td>11.010 [9.539,12.466]</td></tr><tr><td>P3</td><td>.536 [.516,.556]</td><td>.482 [.454,.511]</td><td>1.015 [.934,1.098]</td><td>2.590 [2.325,2.840]</td></tr></table>

Table 22: Separate-head ranges across the three fixed training seeds. Trajectory-bootstrap intervals quantify test-instance uncertainty; these ranges expose training-seed variability separately.
<table><tr><td>Task</td><td>Backbone</td><td>Error range</td><td>Gap range</td><td>MSE range</td></tr><tr><td>P1</td><td>N-HiTS</td><td>.468-.522</td><td>9.002-11.225</td><td>1.654–1.723</td></tr><tr><td>P1</td><td>TCN</td><td>.255-.307</td><td>13.328-14.215</td><td>1.643-1.697</td></tr><tr><td>P2</td><td>N-HiTS</td><td>.557–.588</td><td>8.495-10.748</td><td>.539–.543</td></tr><tr><td>P2</td><td>TCN</td><td>.352-.398</td><td>11.145-12.400</td><td>.821-1.100</td></tr><tr><td>P3</td><td>N-HiTS</td><td>.695-.728</td><td>1.965-3.000</td><td>.889-.930</td></tr><tr><td>P3</td><td>TCN</td><td>.333-.375</td><td>8.893-12.875</td><td>1.045-1.095</td></tr></table>

Table 21 reports iTransformer only on its independent replication cohort, while Table 22 separates the observed three-seed range from trajectory-level uncertainty.

## G.10 Complete common-cohort model coverage

Table 23 contains every system evaluated on the unified 400-trajectory cohort, including forecast-only baselines with unsupported structural cells.

## G.11 Run accounting

The initial architecture replication comprises 27 independent training runs (three models, three tasks, and three seeds), with zero failed runs and three hyperparameter attempts, below the pre-specified limit of 12.

Table 23: Complete unified common-cohort results. Brackets are trajectory-bootstrap 95% intervals; unsupported structural outputs are dashes. Neural entries average three seeds.
<table><tr><td>Task System</td><td></td><td>Readout</td><td>Structural error</td><td>Favorable error</td><td>Forecast MSE</td></tr><tr><td>P1</td><td>cross correlation</td><td>direct</td><td>.083 [.058,.110]</td><td>.007 [.000,.017]</td><td>1.186 [.933,1.452]</td></tr><tr><td>P1</td><td>elastic net</td><td></td><td>direct lag profile .265 [.223,.307].138 [.098,.178]</td><td></td><td>2.509 [1.959,3.071]</td></tr><tr><td>P1</td><td>gaussian map</td><td>direct</td><td>.065 [.043,.090] .010 [.000,.024]</td><td></td><td>1.189 [.939,1.458]</td></tr><tr><td>P1</td><td>mean predictor</td><td>forecast only</td><td></td><td></td><td>1.695 [1.401,2.017]</td></tr><tr><td>P1</td><td>mlp narx frozen probe</td><td>frozen probe</td><td></td><td>.798[.773,.821].798[.769,.826]</td><td>1.559 [1.294,1.854]</td></tr><tr><td>P1</td><td>nhits</td><td>separate head</td><td>.500 [.460,.538] .440 [.396,.488]</td><td></td><td>1.678 [1.375,2.002]</td></tr><tr><td>P1</td><td>pem arx</td><td>direct system ID</td><td></td><td>.242 [.198,.282].114 [.077,.152]2.291 [1.811,2.839]</td><td></td></tr><tr><td>P1</td><td>persistence predictor</td><td>forecast only</td><td></td><td></td><td>3.015 [2.339,3.809]</td></tr><tr><td>P1</td><td>prewhitened cross correlation direct</td><td></td><td>.175 [.140,.212].051 [.027,.074]</td><td></td><td>1.236 [.964,1.533]</td></tr><tr><td>P1</td><td>tcn</td><td>separate head</td><td></td><td>.273[.238,.306].176[.143,.211]1.673 [1.383,1.973]</td><td></td></tr><tr><td>P2</td><td>elastic net</td><td></td><td>direct lag profile .268 [.225,.312]</td><td></td><td>.816 [.612,1.051]</td></tr><tr><td>P2</td><td>mean predictor</td><td>forecast only</td><td></td><td></td><td>1.374 [1.039,1.769]</td></tr><tr><td>P2</td><td>mlp narx frozen probe</td><td>frozen probe</td><td>.784 [.759,.808] .783 [.731,.832]</td><td></td><td>.520 [.401,.659]</td></tr><tr><td>P2</td><td>nhits</td><td>separate head</td><td>.570 [.532,.607] .500 [.443,.560]</td><td></td><td>.540 [.429,.664]</td></tr><tr><td>P2</td><td>pem arx</td><td>direct system ID</td><td>.248 [.205,.290].049 [.025,.078]</td><td></td><td>.733 [.558,.933]</td></tr><tr><td>P2</td><td>persistence predictor</td><td>forecast only</td><td></td><td></td><td>.569 [.461,.689]</td></tr><tr><td>P2</td><td>profile arx</td><td>direct</td><td>.118 [.085,.150] .004 [.000,.012]</td><td></td><td>.388 [.290,.507]</td></tr><tr><td>P2</td><td>tcn</td><td>separate head</td><td>.370 [.331,.408].250 [.198,.303]</td><td></td><td>.935 [.778,1.110]</td></tr><tr><td>P3</td><td>elastic net</td><td></td><td>direct lag profile .853 [.815,.885] .857 [.813,.900]</td><td></td><td>1.622 [1.257,2.073]</td></tr><tr><td>P3</td><td>finite library mle</td><td>direct</td><td>.115 [.085,.147] .004 [.000,.013]</td><td></td><td>.929 [.771,1.108]</td></tr><tr><td>P3</td><td>mean predictor</td><td>forecast only</td><td></td><td></td><td>1.159 [.979,1.346]</td></tr><tr><td>P3</td><td>mlp narx frozen probe</td><td>frozen probe</td><td>.851 [.828,.873].854 [.825,.881]</td><td></td><td>.903 [.740,1.071]</td></tr><tr><td>P3</td><td>nhits</td><td>separate head</td><td>.714 [.679,.749].691 [.645,.736]</td><td></td><td>.912 [.754,1.087]</td></tr><tr><td>P3</td><td>pem arx</td><td>direct system ID</td><td></td><td>.833 [.795,.868].839 [.791,.883]1.458 [1.152,1.818]</td><td></td></tr><tr><td>P3</td><td>persistence predictor</td><td>forecast only</td><td></td><td></td><td>1.426 [1.155,1.753]</td></tr><tr><td>P3</td><td>tcn</td><td>separate head</td><td>.353 [.318,.388].225 [.187,.264]</td><td></td><td>1.068 [.890,1.269]</td></tr></table>

The train, validation, and test identifiers are disjoint. Only systems that satisfy the common historical (U, Y ) contract are eligible for a scored row.

Table 25 separates training, probe fitting, evaluation, and bootstrap operations. The release artifact additionally records resolved configurations, code and checkpoint hashes, per-instance predictions, structural outputs, bootstrap seeds, and failure manifests. Full software contracts and schemas accompany the repository rather than the paper source.

Table 24: Release and cohort map. Comparisons are paired only within a row; independent releases are not pooled for ranking. “Replay” means evaluation of frozen predictions or deterministic rules without retraining.
<table><tr><td>Evidence block</td><td>Public release</td><td>Train/calibration cohort Test cohort</td><td></td><td>Systems/regimes</td><td>Seeds</td><td>Retrained?</td></tr><tr><td>Initial architecture replica- Initial architecture re- 16,200/1,200 per task tion</td><td>lease</td><td></td><td>independent 1,200 per task</td><td>TCN, N-HiTS, iTrans- former; separate report</td><td>3</td><td>yes</td></tr><tr><td>Unified capability chain</td><td></td><td>Unified-regime release 4,096/512/512 per task</td><td>common 400 per task</td><td>TCN, N-HiTS; legacy five-regime audit plus P1</td><td>3</td><td>yes</td></tr><tr><td>Training-budget audit</td><td>Saturation release</td><td>8,192 16,384/512/512</td><td>or same common 400</td><td>closure controls TCN, N-HiTS; fore- cast/probe and separate</td><td>3</td><td>yes</td></tr><tr><td>Functional sanity</td><td>Functional-sanity</td><td>P2/P3 re- none</td><td>fresh 400</td><td>head seven known rules; three diagnostics</td><td></td><td>no</td></tr><tr><td>P3 full-kernel audit</td><td>lease Full-kernel release</td><td>unified checkpoints</td><td>unified P3 common 400</td><td>TCN/N-HiTS separate/shared/soft- routed/frozen; two</td><td>3</td><td>replay</td></tr><tr><td>CPU extension</td><td></td><td>CPU-coverage release 4,096/512/512 per task common 400 per task</td><td></td><td>classical references Elastic Net, prediction- 3 MLP error ARX, MLP-NARX,</td><td></td><td>yes/replay</td></tr><tr><td>Real-input transfer</td><td></td><td></td><td></td><td>mean, persistence Real-input transfer re- frozen synthetic and 200 per source/task be- TCN, N-HiTS, routed lag-</td><td>3</td><td>replay</td></tr><tr><td>Theory-experiment clo- Confirmatory closure original unified P1 split; common 400 P1/P2</td><td>lease</td><td>source-specific splits</td><td>fore support filtering</td><td>aware variants P1 separate/soft-</td><td>3</td><td></td></tr><tr><td>sure</td><td>release</td><td>repaired P2 calibration table</td><td></td><td>routed/hard-onehot; corrected P2 member-</td><td></td><td>P1 yes / P2 no</td></tr></table>

Table 25: Run accounting by scientific study. Training, calibration-probe fits, checkpoint/rule evaluations, and bootstrap analyses are distinct operations. Failed development runs are retained in manifests and are not included in scientific aggregates.
<table><tr><td>Study</td><td>Training</td><td>Probe fits</td><td>Evaluation cells</td><td>Bootstrap contrasts</td><td>Failed</td><td>Retried</td></tr><tr><td>Initial architecture replication</td><td>27</td><td>0</td><td>27</td><td>6</td><td>0</td><td>0</td></tr><tr><td>Unified five-regime release</td><td>72</td><td>18</td><td>90</td><td>60</td><td>0</td><td>0</td></tr><tr><td>Training-size saturation</td><td>48</td><td>24</td><td>48</td><td>48</td><td>0</td><td>0</td></tr><tr><td>Functional sanity audit</td><td>0</td><td>0</td><td>21</td><td>21</td><td>0</td><td>0</td></tr><tr><td>P3 full-kernel audit</td><td>0</td><td>0</td><td>26</td><td>72</td><td>3</td><td>3</td></tr><tr><td>CPU coverage extension</td><td>9</td><td>9</td><td>21</td><td>39</td><td>1</td><td>1</td></tr><tr><td>Matched real-input transfer</td><td>0</td><td>0</td><td>30</td><td>120</td><td>0</td><td>0</td></tr><tr><td>Theory-experiment closure</td><td>18</td><td>0</td><td>一</td><td>一</td><td>0</td><td>0</td></tr></table>