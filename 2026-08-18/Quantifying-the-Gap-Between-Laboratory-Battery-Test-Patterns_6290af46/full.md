# Quantifying the Gap Between Laboratory Battery Test Patterns and Field Duty Profiles

Chunyang Zhao and Chresten Træholt

Department of Wind and Energy Systems

Technical University of Denmark

Kongens Lyngby, Denmark

chuzh@dtu.dk, ctra@dtu.dk

Abstract—Laboratory battery tests provide the main empirical basis for battery performance and degradation studies, but their operating patterns do not directly represent field duty profiles. This paper quantifies the gap by comparing six accessible evidence sources covering controlled cycling, drive-cycle testing, dynamic cycling, NMC811 laboratory ageing, a real electricvehicle charging trace, and fleet-scale electric-vehicle state-ofhealth (SOH) data. The analysis combines usage frequency, usage intensity, usage C-rate, and a duty-structure index (DSI) based on normalized current dispersion and ramping. The representative single-segment DSI ranges from 0.630 for the field source trace and 0.699 for NASA to 2.936 for Oxford and 2.855 for Imperial, while usage C-rate ranges from 0.14–0.40 for Imperial, NASA, Stanford, and Hyundai to 2.00 for Oxford. Long-term ageing also differs: the 80% retention region occurs near 351 NASA cycles, 6292 Oxford checkpoints, and 1019 Stanford cycles. In chemistry-aligned NMC/NCM evidence, Imperial retains 0.813 under standard cycling and 0.865 under drive-cycle ageing, while the field source has median SOH 0.889 with visible dispersion. Field operation further shows a median use intensity of 137.2 km/day and 56.9% of charges ending at or above 95% SOC. These results show that battery performance metrics are conditional on the duty pattern that generated them; applicationoriented studies should report explicit duty-profile descriptors together with chemistry, capacity, and ageing metrics.

Index Terms—battery testing, duty profile, electric vehicle, battery degradation, field operation, application-oriented evaluation

## I. INTRODUCTION

Battery cells and battery energy storage systems are now deployed across electric vehicles, stationary storage, microgrids, hybrid renewable systems, and grid-support applications [1]. In each of these settings, battery behaviour is shaped not only by chemistry and cell design, but also by the operating pattern imposed over time: current transients, voltage windows, stateof-charge (SOC) limits, temperature exposure, rest periods, and supervisory control all influence usable energy, efficiency, thermal stress, and ageing. The consequence is simple but important. Battery performance is inseparable from the duty profile [2].

Most battery knowledge, however, is still generated in the laboratory. Although fundamental material and reaction studies continue to advance battery science, controlled cycling, constant-current constant-voltage charging, periodic reference tests, and temperature-regulated ageing campaigns remain indispensable because they provide repeatability, interpretability, and a defensible basis for cross-study comparison. Open datasets from NASA PCoE [3], Oxford [4], [5], the early-life benchmark introduced by Severson et al. [6], and more recent dynamic-protocol studies, such as the Stanford work [7], have greatly expanded the empirical basis for battery research. Yet the operating logic of these tests remains more structured than that of deployed batteries.

This mismatch creates a transfer problem. A battery result may be valid under a laboratory protocol and still be only partially informative for a real application. In practice, field duty profiles are stochastic, partially observed, and strongly shaped by battery-management systems, charger behaviour, route conditions, driver behaviour, and system-level constraints. Recent open field-EV work has made this distinction clearer by showing both the richness of real operating histories and the difficulty of translating laboratory-derived health indicators directly into real-world battery management [8]. The issue is not that laboratory testing is inadequate; rather, different test patterns support different kinds of claims.

This paper addresses this issue by comparing battery evidence at the test pattern level. We propose two linked research questions: which battery-performance statements are supported by controlled, application-inspired, dynamic, and field-derived evidence, and how far do those evidence types remain from the actual field duty structure? To answer them, we assemble a compact but reproducible comparison spanning controlled cycling, drive-cycle-inspired cell tests, dynamic cycling, chemistry-aligned evidence, a real EV charging trace, and field state-of-health statistics.

The key contribution of the paper is not a new dataset or a new lifetime model, but a structured way to interpret battery performance evidence across the laboratory-to-field gap. Specifically, the paper contributes: 1) a duty-patternbased comparison framework that treats battery datasets as evidence classes rather than interchangeable benchmarks; 2) a compact quantitative comparison spanning transient response, operating variability, and long-term retention across accessible commercial lithium-ion battery testing and operation sources; and 3) a quantitative analysis approach to compare and analyze the testing pattern of the battery cell operations. After the introduction sections, the remainder of the paper is organized as follows: Section II defines the evidence base and workflow, Section III presents the comparison results, Section IV discusses evidence limits, and Section V concludes the paper.

## II. STUDY DESIGN AND EVIDENCE BASE

Table I summarizes the selected evidence base for comparing battery test patterns and field duty, and the representative segments of duty cycle in the normalized current profiles are presented in Fig 1. The table identifies the system level, chemistry, nominal capacity, signal range, and scientific role of each source. This distinction is necessary because the selected records span sub-Ah laboratory cells, commercial 21700 cells, large EV cells or packs, and fleet-level source data; their voltage and current magnitudes, therefore, cannot be pooled directly. The comparison is organized by the claim each source can support: controlled response and fade for NASA, application-shaped cell cycling for Oxford, chemistry-aligned NMC811 ageing for Imperial, fleet SOH and usage dispersion for the 2025 EV source, sequence-rich dynamic cycling for Stanford, and raw field charging structure for the Hyundai Ioniq 5 session.

The first layer of quantification follows the long-term usagedescription method proposed in our prior work [1]. Conventional battery descriptors such as chemistry, nominal capacity, voltage, current, and temperature describe hardware properties or instantaneous operating states, but they do not fully describe how a battery is used over an application period. For this reason, usage frequency $( U F ) .$ , usage intensity $( U I ) .$ and usage C-rate (UC) are used to describe how often the battery is active, how many cycles are accumulated during active use, and how large the charging-current level is during use.

$$
\mathrm { U s a g e \ f r e q u e n c y } = \frac { \mathrm { A c t i v e \ u s a g e \ t i m e \ l e n g t h } } { \mathrm { A p p l i c a t i o n \ t i m e \ l e n g t h } }\tag{1}
$$

$$
\mathrm { U s a g e ~ i n t e n s i t y } = { \frac { \mathrm { C y c l e ~ c o u n t } } { \mathrm { A c t i v e ~ u s a g e ~ t i m e ~ l e n g t h } } }\tag{2}
$$

$$
\mathrm { U s a g e ~ C - r a t e } = { \frac { \int \left| { \mathrm { C } } \cdot \mathrm { r a t e } \right| d t } { \mathrm { A c t i v e ~ c h a r g i n g ~ t i m e ~ l e n g t h } } }\tag{3}
$$

Here, active usage time is the accumulated charging and discharging time, application time is the total elapsed application period including standby, and active charging time is the charging portion used for the usage C-rate calculation. The cycle count can be obtained through rainflow counting or another consistent equivalent-cycle method. In this paper, SOC is not available for all sources, so cycle count is estimated consistently from current throughput when a direct cycle count is not available. For Stanford, the reported normalized Crate is adopted directly. For the 2025 EV source-data trace, sample-coordinate normalization is used because timestamps are unavailable. For Hyundai, the value is treated as an ACcurrent proxy because the KIT charging trace does not provide direct battery DC current.

These three usage metrics provide the long-period interpretation, but they do not by themselves capture short-period current shape. Each selected record is therefore also resampled onto a common progress coordinate and represented by the normalized current profile

$$
i ( \tau ) = \frac { I ( \tau ) } { \operatorname* { m a x } | I | }\tag{4}
$$

where $\tau$ is the normalized progress through the selected cycle, operating window, or source-data trace. The 2025 EV fleet trace is placed only in Segment 1 of Fig. 1 because the accessible source contains a single charging trace rather than repeated operating sections. To quantify operational variability and dynamic loading characteristics, the duty-structure index (DSI) is defined as

$$
\begin{array} { l } { { \displaystyle \mathrm { D S I } = \sqrt { A ^ { 2 } + R ^ { 2 } } } } \\ { { \displaystyle A = \mathrm { s t d } ( i ) } } \\ { { \displaystyle R = \ln \left( 1 + \frac { 1 } { 2 M } \sum _ { m = 1 } ^ { M } \sum _ { j } \left| i _ { m } ( \tau _ { j + 1 } ) - i _ { m } ( \tau _ { j } ) \right| \right) } } \end{array}\tag{5}
$$

where M denotes the number of selected records. Here, A characterizes the dispersion of the normalized current profile, whereas R captures cumulative ramping behavior with logarithmic compression to reduce sensitivity to extreme transients.

## III. RESULTS

## A. Duty Structure and Cycle Response

Fig. 1 compares the behavior of the battery testing profile and the field applications, presenting the difference between the cycling tests in a and c and the dynamic operation tests in b and d of the subfigures. The 2025 EV fleet panel places its single accessible pack-level charging trace in segment 1 and leaves segments 2–5 empty; the Hyundai panel uses five repeated charger-control blocks. The contrast is immediate. The NASA profile now shows repeated chargedischarge laboratory cycles, with long controlled charging stages and nearly constant-current discharges. The Oxford window repeats the available application-shaped current profile and introduces frequent current variation during the discharge stage. The Stanford window reveals repeated long phases of charge, discharge, and rest. The Imperial 21700 trace uses five 1800 s repeat windows from the WLTP-based setting. The 2025 EV fleet source trace shows a pack-level transition from a high-current stage to a lower-current stage, while the Hyundai charging record shows repeated charger-control blocks.

Looking into the diagnosis cycle, where stable chargingdischarging processes are implemented, Fig. 2 extends the comparison from the duty structure to the measured response by using normalized progress through each selected response record. Reference performance tests (RPTs) are periodic diagnostic tests used during ageing campaigns to measure capacity retention under controlled conditions. NASA is shown with a charge–discharge pair, Oxford with the application-shaped example cycle, Stanford with the selected dynamic cycle, and Imperial with the available RPT0 diagnostic response because the public drive-cycle setting does not include a matching voltage–temperature trace. Voltage is retained in physical units, current is normalized by peak magnitude, and temperature is shown if measured.

TABLE I  
FOCUSED DATA DESCRIPTION FOR THE QUANTIFIED SOURCES USED IN THE COMPARISON.
<table><tr><td>Source</td><td>System</td><td>Chemistry</td><td>Capacity</td><td>V mean [min,max] (V)</td><td>I mean [min,max]</td><td>T mean [min,max] (°C)</td><td>Test / operating case</td></tr><tr><td>NASA B0005 [3]</td><td>Cell</td><td>Li-ion</td><td>2.0 Ah</td><td>3.56 [2.63, 4.20]</td><td>-1.92 A [-2.02, 0.00] A</td><td>31.92 [23.78, 38.46]</td><td>Controlled degradation baseline; constant-current ageing with capacity and impedance follow-up</td></tr><tr><td>Oxford NCA Cell1 [4], [5]</td><td>Cell</td><td>NCA</td><td>0.74 Ah</td><td>3.88 [3.60, 4.20]</td><td>-0.12 A [-5.00, 1.60] A</td><td>40.38 [39.93, 41.09]</td><td>Application-shaped bridge dataset; drive-cycle discharge in thermal-chamber testing</td></tr><tr><td>Imperial 21700 [9] Cell</td><td></td><td>C-SiOx</td><td>NMC811 / 4.865 Ah</td><td>3.66 [2.50, 4.17]</td><td>-0.50 A [-0.50, -0.50] A</td><td>25.93 [24.35, 26.70]</td><td>Chemistry-aligned ageing reference; standard-cycle vs drive-cycle ageing at 25°C</td></tr><tr><td>2025 EV fleet source [8]</td><td>EV fleet NCM</td><td></td><td>155 Ahª</td><td>4.04 [3.75, 4.25]</td><td>73.61 A [0.00, 103.50] A</td><td>31.25b [30.00, 32.00]b</td><td>Field health and usage diversity; in-service EV operation, charging windows, and SOH statistics</td></tr><tr><td>Stanford cell 003 [7]</td><td>Cell</td><td>SiOx-Gr / NCA</td><td>n.r.c</td><td>3.57 [2.98, 4.20]</td><td>0.00d [-0.10, 0.50]d</td><td>35.00e [35.00, 35.00]e</td><td>Dynamic protocol comparison; long charge-rest-discharge sequence under dynamic cycling</td></tr><tr><td>Hyundai Ioniq 5 [10]</td><td>Vehicle session</td><td>SK On NCM pouch</td><td>111-121 Ahf</td><td>233.29 [75.48, 285.19]</td><td>30.69 A [0.52, 48.08] A</td><td>2.42g [-0.50, 5.40]g</td><td>Real duty-trace example; real EV AC charging session with SOC and phase-power records</td></tr></table>

<sup>a</sup> Liu et al. report a rated pack capacity of 155 Ah with 96 cells connected in series. <sup>b</sup> EV-fleet temperature is the charging-temperature health-indicator sequence from the source-data release. <sup>c</sup> Stanford reports commercial SiOx–graphite/NCA EV energy cells and normalized capacity/C-rate data; the exact cell model and absolute nominal capacity are not disclosed. <sup>d</sup> Stanford current is reported in normalized C-rate rather than amperes. <sup>e</sup> Stanford temperature is the chamber set point reported for the ageing campaign. <sup>f</sup> KIT leaves the selected Hyundai capacity field blank; Hyundai battery-cell information lists Ioniq 5 cells as SK On pouch-type NCM, while public pack specifications imply about 111–121 Ah across the 72.6/77.4/84 kWh variants. <sup>g</sup> KIT temperatur is ambient temperature during the selected Hyundai charging session.

## B. Short-Term and Long-Term insights

Fig. 3(a) compares long-term degradation trajectories from NASA, Oxford, and Stanford using normalized retention on the vertical axis and normalized ageing progress on the horizontal axis. The original ageing coordinates are cycle count for NASA and Stanford and checkpoint count for Oxford, but they are mapped to 0–100% to compare trajectory shape. NASA shows the faster decline with more fluctuation in the selected records, Oxford degrades more gradually, and Stanford remains the shallowest of the three until late in life.

Fig. 3(b) adds a chemistry-aligned comparison between laboratory NMC811 controls and real-field NCM outcomes. The Imperial 21700 controls at 25 <sup>◦</sup>C [9] show that the drivecycle laboratory condition retains 0.865 by RPT9, whereas the standard-cycle control falls to 0.813 by RPT15. These RPT coordinates and the field mileage coordinate are also normalized to 0–100% in the figure. The field NCM sourcedata release [8] reports a median SOH trajectory in a similar numerical band to the milder drive-cycle laboratory condition, but its interquartile spread reflects fleet heterogeneity that laboratory single-cell controls do not capture, but is evident.

Fig. 4 complements the single duty trace with broader field context from Liu et al. [8]. The source-data release shows a median use intensity of 137.2 km/day, with a 10th–90th percentile range of 90.6 to 197.6 km/day. Charge-window statistics show a median charge start SOC of 38.0% and a median charge end SOC of 96.0%, with 56.9% of charging events ending at or above 95% SOC. The SOH-versus-mileage panel shows an overall downward trend but also substantial spread, indicating that mileage alone does not explain field health outcomes. The selected Hyundai Ioniq 5 charging session from the KIT record [10] lasts 11.30 h, reaches a peak summed phase current of 48.1 A, and increases SOC by 53.0 percentage points.

## IV. DISCUSSION

The results show that battery test patterns are not merely different experimental formats; they are different kinds of evidence. Fig. 1 and Fig. 2 show that the controlled NASA cycle, the Oxford drive-cycle discharge, the Stanford dynamic protocol, the Imperial NMC811 drive-cycle setting, the 2025 EV fleet source trace, and the Hyundai charging trace differ in temporal structure even after normalization. For the repeated records, the quantitative indices in Table II are calculated from one representative segment: one NASA charge–discharge pair, one Oxford example cycle, one Stanford dynamic cycle, one Imperial 1800 s drive-cycle window, and one Hyundai chargercontrol block. The corresponding DSI is 0.699 for NASA, 0.639 for Stanford, 2.936 for Oxford, 2.855 for Imperial, 0.630 for the single 2025 EV fleet source trace, and 2.268 for the selected Hyundai charger-control block. The usage metrics add a complementary interpretation: Oxford has the highest usage intensity and usage C-rate among the selected cell records, whereas Stanford is active for a substantial fraction of time but at a low average C-rate. Thus, the most structured profiles are not necessarily those with the largest amplitude spread; they are the profiles with repeated ramps, control actions, and transient current changes.

The long-term comparison leads to the same conclusion at the ageing level. In Fig. 3(a), the selected laboratory records reach the 80% region on very different interpolated horizons: near cycle 351 for NASA, near checkpoint 6292 for Oxford, and near cycle 1019 for Stanford. These are not just different lifetimes; they are different ageing trajectories shaped by different test logic. Fig. 3(b) extends this argument by showing that chemistry-aware alignment improves interpretation but does not eliminate the field gap. The Imperial NMC811 control retains 0.813 under standard cycling and 0.865 under drivecycle ageing, whereas the field NCM source reports a median SOH of 0.889 together with visible dispersion. The implication is that even a chemistry-matched laboratory reference cannot stand in for the diversity of field operation, because the field record retains the effects of heterogeneous charge windows, operating intensity, and control history.

(a) NASA PCoE  
![](images/1ed61d0258bdb224c60cd58697bbf55e44a260f7fc2956f200c408113ef4cc4a.jpg)  
(b) Oxford NCA

![](images/83ba5a03d279a5ae9e943bf5ed59db353c477efd57ff4e01aedde05a1fcc01d3.jpg)

(c) Stanford dynamic cycling  
![](images/158999e6d9d5943eebd293e66135b893e81eb692608f8e9a432e951c0650dbda.jpg)  
(d) Imperial 21700 drive cycle

![](images/837011afcab54b4e0508940d2560d1181f6e6323e9727d1fdaa9f8a98a89fd26.jpg)  
(e) 2025 EV fleet source trace

![](images/40768c7ab6cccc33c140769cda6f4b54d6a92d4c56a7a88e7acda8448bb84d81.jpg)  
(f) Hyundai loniq 5

![](images/3c76730f7a4c5d22b6c39231a26bd74837ae8e8aa58c5e5dd09cbf1b2a7153c7.jpg)  
Progress through selected cycle, window, or source trace (%  
Fig. 1. Normalized current profiles over selected cycles, operating windows, or source-data traces. Note: The 2025 EV fleet trace occupies only segment 1 because repeated sections are not available.

The field statistics make that diversity explicit. The accessible EV source data show a median use intensity of 137.2 km/day, a median charging window from 38.0% to

(a) Voltage response  
![](images/3ce05e074ba2064e3649b7f5a32b4f4b4eda672f6f457f04fa690b7c82ac8cce.jpg)  
(b) Normalized current response

![](images/28c2e4c2275e5ab5adcb0eff79e6173421c588739565a3f7b4b3642007e74b4d.jpg)  
(c) Temperature response

![](images/0990f9e8c249b3d14ffe2204680848900e67d11be7d926af42bb690e6b7f769f.jpg)  
Progress through selected response record (%)  
Fig. 2. Voltage, normalized-current, and temperature responses for selected cell-level records in reference performance tests (RPTs)

(a)Laboratory retention trajectories  
![](images/52cf895d4c21552f6fec4146ffb45b537326a5785ae2a4caa97546c31602a9aa.jpg)

(b)NMC811 controls and field SOH  
![](images/30dd457c8978333b6eb556379a6e979f25a56245e07ce9a0e3118418b5b608c4.jpg)  
Fig. 3. Long-term retention and SOH comparison across laboratory datasets and chemistry-aligned evidence using normalized aging progress.

96.0% SOC, and 56.9% of charging events ending at or above 95% SOC. Those operating patterns are poorly represented by one canonical laboratory cycle. Table II condenses the main quantitative implications: the NASA case provides the controlled baseline and fastest relative fade among the selected cell datasets, the Oxford and Imperial records show the largest single-segment duty-structure indices among the laboratory traces, the fleet source combines a low-DSI single trace with broad operating and SOH dispersion, and the Hyundai session shows repeated charger-control structure. For application-oriented studies, the relevant question is which application claim a given test can defend. Controlled cycling is appropriate for mechanism isolation and stable benchmarking. Drive-cycle and dynamic protocols are better suited to claims about application-shaped response. Field traces and field health statistics are required when the claim concerns operational realism, fleet heterogeneity, or transfer to deployed systems.

![](images/17fc705c84c2d81be054c6fe9f11806a842a24b5b9352587f7c7ac4977f9a685.jpg)

(b) Charge SOC windows  
![](images/9d3cf61168b3e394b2dd071b4173c6ecbf5317a596ae4900ad8cbfacaf62e7bc.jpg)  
(c) SOH versus mileage

![](images/5d4a98ee1d66257d9f4c41a64a4b038c2fbe929d99911fe90c33f7b4601c5deb.jpg)

![](images/32653c6098b297224b70d485787391e71de20b2c330286b7fc1ecf679986abfc.jpg)  
Fig. 4. Field-EV evidence to characterize real operating diversity.

## V. CONCLUSION

This paper quantified the laboratory-to-field duty-profile gap using controlled, drive-cycle, dynamic, chemistry-aligned, and field EV evidence. The selected records are not interchangeable benchmarks: they support different claims because they impose different current structures, diagnostic conditions, and operating histories.

The duty-profile comparison gives the clearest numerical result. The single-segment DSI is 0.699 for NASA, 0.639 for Stanford, 2.936 for Oxford, 2.855 for Imperial, 0.630 for the EV-fleet source trace, and 2.268 for the Hyundai chargercontrol block. The usage metrics show a similar separation: Oxford has the highest usage intensity and usage C-rate among the selected cell records $( \mathrm { U I } = 0 . 8 1 7 , \mathrm { U C } = 2 . 0 0 )$ , while Stanford has lower active cycling intensity $( \mathrm { U I } = 0 . 0 8 2 $ , UC $= 0 . 4 0 )$ . These values confirm that laboratory protocols differ not only in scale, but also in temporal structure.

TABLE II  
KEY QUANTIFIED DUTY-STRUCTURE, DEGRADATION, ANDOPERATING-DRIFT INDICATORS ACROSS REPRESENTATIVE SINGLESEGMENTS AND SOURCE RECORDS.
<table><tr><td>Source</td><td>DSI</td><td>UF</td><td>UI</td><td>UC</td><td>Degradation</td><td>Main implication</td></tr><tr><td>NASA</td><td>0.699</td><td>0.70</td><td>0.296</td><td>0.29</td><td>80%@351 cy- cles</td><td>Fastest degradation under controlled high-intensity cycling.</td></tr><tr><td>Oxford</td><td>2.936 0.75</td><td></td><td>0.817 2.00</td><td></td><td>80%@6292 checkpoints</td><td>Highest sustained usage C- rate with longest cycling life- time.</td></tr><tr><td>Stanford</td><td>0.639 0.66 0.082 0.40</td><td></td><td></td><td></td><td>80%@1019 cycles</td><td>Smooth low-C-rate cycling with reduced operational stress.</td></tr><tr><td>Imperial</td><td>2.855 0.77</td><td></td><td>0.150 0.14</td><td></td><td>∆ret. +0.052 at RPT</td><td>Highest ramp-dominated duty structure and transient vari- ability.</td></tr><tr><td>Hyundai</td><td>2.268</td><td>0.84</td><td>0.132</td><td>0.26</td><td>n.a.</td><td>No degradation label; full- session ∆SOC 53.0.</td></tr><tr><td>EV fleet</td><td>0.630</td><td>1.00</td><td>一</td><td>0.48</td><td>SOH50 0.889; cycles n.r.</td><td>Median field SOH; cycle count not reported.</td></tr></table>

Note: DSI, UF, UI, and UC are calculated from one representative segment for repeated records; degradation indicators use the full ageing or fleet-health source. UC is calculated over active charging time following the prior usage C-rate definition. Hyundai usage Crate is estimated using an AC-current proxy. EV-fleet UC is sample-coordinate based and UI is not reported because timestamps and cycle counts are unavailable.

The degradation comparison shows that the interpretation of ageing is also duty-pattern dependent. The 80% retention region appears near 351 NASA cycles, 6292 Oxford checkpoints, and 1019 Stanford cycles. In the NMC/NCM comparison, Imperial retains 0.813 under standard cycling and 0.865 under drive-cycle ageing, whereas the field source reports median SOH of 0.889 with an interquartile spread. The field data also show operational diversity: median use intensity is 137.2 km/day, the median charging window is 38.0–96.0% SOC, and 56.9% of charges end at or above 95% SOC.

The practical conclusion is that laboratory testing is necessary but not sufficient for application-oriented claims. Battery studies should report chemistry, capacity, temperature, and cycle count together with charge/discharge windows, temporal variability, rest structure, usage frequency, usage intensity, usage C-rate, and the intended application analogy. Without this alignment, a performance or degradation metric can be valid for its test protocol while remaining weak evidence for deployed operation.

## REFERENCES

[1] C. Zhao, P. B. Andersen, C. Træholt, and S. Hashemi, “Grid-connected battery energy storage system: A review on application and integration,” Renewable and Sustainable Energy Reviews, vol. 182, p. 113400, Aug. 2023.

[2] A. Geslin, L. Xu, D. Ganapathi et al., “Dynamic cycling enhances battery lifetime,” Nature Energy, vol. 10, pp. 172–180, 2025, doi: 10.1038/s41560-024-01675-8.

[3] B. Saha and K. Goebel, “Battery Data Set,” NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA, USA, 2007.

[4] D. Howey and C. Birkl, “Oxford Battery Degradation Dataset 1,” University of Oxford, Oxford, U.K., 2017, doi: 10.5287/bodleian:KO2kdmYGg.

[5] C. R. Birkl, M. R. Roberts, E. McTurk, P. G. Bruce, and D. A. Howey, “Degradation diagnostics for lithium ion cells,” Journal of Power Sources, vol. 341, pp. 373–386, 2017, doi: 10.1016/j.jpowsour.2016.12.011.

[6] K. A. Severson et al., “Data-driven prediction of battery cycle life before capacity degradation,” Nature Energy, vol. 4, no. 5, pp. 383–391, 2019, doi: 10.1038/s41560-019-0356-8.

[7] A. Geslin, L. Xu, D. Ganapathi, K. Moy, W. C. Chueh, and S. Onori, “Dynamic cycling enhances battery lifetime,” Nature Energy, vol. 10, pp. 172–180, 2025, doi: 10.1038/s41560-024-01675-8.

[8] H. Liu, C. Li, X. Hu, J. Li, K. Zhang, Y. Xie, R. Wu, and Z. Song,

“Multi-modal framework for battery state of health evaluation using open-source electric vehicle data,” Nature Communications, vol. 16, Art. no. 1137, 2025, doi: 10.1038/s41467-025-56485-7.

[9] N. Kirkaldy, M. A. Samieian, G. J. Offer, M. Marinescu, and Y. Patel, “Lithium-ion battery degradation: Comprehensive cycle ageing data and analysis for commercial 21700 cells,” Journal of Power Sources, vol. 603, p. 234185, 2024, doi: 10.1016/j.jpowsour.2024.234185.

[10] S. Beichter et al., “Open-source data set for characterizing the charging behavior of electric vehicles,” Zenodo, Feb. 9, 2026, doi: 10.5281/zenodo.17866847.