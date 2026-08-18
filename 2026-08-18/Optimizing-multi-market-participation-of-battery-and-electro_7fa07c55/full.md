# Optimizing multi-market participation of battery and electrolyser systems based on field performance

Chunyang Zhao, Stoyan Trenchev, Shi You, Chresten Træholt

Department of Wind and Energy Systems

Technical University of Denmark

Kongens Lyngby, Denmark

chuzh@dtu.dk, stoyan.trenchev@mailbox.org, shyo@dtu.dk, ctra@dtu.dk

Abstract—The increasing share of renewable energy in power systems creates a need for fast-response and flexible resources to maintain system stability. With the expansion of electricity markets and ancillary service products, opportunities arise to stack revenues across multiple services. Long-term Power-to-X (PTX) electrolysers and short-term battery energy storage systems (BESS) are prevalent flexible resources, yet most studies neglect real hardware behavior, such as ramp limits, efficiency, setpoint-tracking accuracy, etc. This work presents experimental and modeling results for a 55 kW/79 kWh BESS and an electrolyzer comprising three 2.4 kW units. Key characteristics are identified through measurements and embedded into a price-driven optimization framework for participation in the Danish electricity and ancillary service markets, utilizing real market data from 2022 to 2025. The optimized daily profits for multi-market participation are 1,749.27 DKK and 289.46 DKK for BESS and the electrolyser, respectively. With the demonstrated business cases for BESS and PTX systems, this work addresses the importance of incorporating experimental performance across multiple markets and years.

Index Terms—Battery energy storage system (BESS), Powerto-X (PTX), Energy market, Ancillary service.

## I. INTRODUCTION

The global energy transition is characterized by rapid electrification and an accelerating deployment of variable renewable energy sources such as wind and solar photovoltaic (PV) [1]. While large-scale plants continue to dominate new capacity additions, the penetration of distributed energy resources (DERs) at the grid edge, such as battery energy storage systems (BESS), heat pumps, and electrolysers, is increasingly recognized as essential for balancing supply and demand, mitigating grid congestion, and providing system services [2]. Aggregated DERs can offer fast, localized flexibility and complement conventional reserves traditionally provided by synchronous generators, while Power-to-X assets can produce hydrogen to contribute to broader decarbonization goals [3].

In parallel, the Nordic power system is introducing new market products that reward fast response and bidirectional flexibility. Services such as Fast Frequency Reserve (FFR), Frequency-Controlled Disturbance Reserve (FCRD), and Frequency-Controlled Normal Operation Reserve (FCRN) are now open to various flexible energy resources, provided that they meet activation-time, duration, and accuracy requirements [4]. These developments create new business opportunities for small, responsive assets that can participate simultaneously in multiple markets, a strategy often referred to as multi-market optimization or revenue stacking [5].

However, most existing studies rely on idealized or specification data and neglect the physical limitations of real hardware [6]. Specifically, laboratory and field assets exhibit constrained voltage windows, temperature-dependent efficiencies, limited setpoint tracking accuracy, and nonlinear startup or ramp dynamics that can significantly influence market eligibility and profitability [7]. For example, the battery state of charge (SOC) and state of energy (SOE) can exhibit a significant divergence in large-scale BESS, and the start-up behavior of DER assets cannot be represented by simple electrical or physical models. Accurately quantifying these limitations and embedding them into dispatch algorithms is essential for realistic performance assessment.

This paper addresses this gap by experimentally characterizing and modeling two real assets installed at the Technical University of Denmark: a modular Xolta BESS rack and a modular anion exchange membrane (AEM) electrolyzer. The contributions are threefold:

1) Experimental quantification of each asset’s operational constraints relevant to market participation, including usable energy range, round-trip efficiency, setpoint accuracy, preheat behavior, ramp rates, etc.

2) Development of a data-driven framework that optimizes energy and reserve bids across the Danish day-ahead and ancillary-service markets using historical data.

3) Optimized dispatch to evaluate revenue potential based on the real-world hardware operation constraints for multiple markets from days to years.

In this work, we demonstrate that laboratory-scale energy assets can be optimally dispatched across multiple electricity markets when their physical constraints are explicitly modeled. After reviewing the state-of-the-art in Section I, comprehensive experimental characterizations of both the BESS and the electrolyzer are presented, capturing their short-term dynamic behavior as well as long-term performance limitations in Section II. In Section III, the multi-market optimization framework and operational objectives are detailed. In Section IV, simulation results are illustrated, including daily optimization and multi-year validation. The paper concludes with discussing key findings and broader implications in Section V & VI.

TABLE I: Key parameters of the assets.
<table><tr><td>Parameter</td><td>BESS</td><td>Electrolyzer</td></tr><tr><td>Power</td><td>55 kW (30 kW used)</td><td> $3 \times 2 . 4 \mathrm { k W }$ </td></tr><tr><td>Energy</td><td>79 kWh (52 kWh used)</td><td></td></tr><tr><td>Efficiency</td><td>90–98% RTE</td><td> $6 2 { - } 7 0 \% \ \mathrm { H H V }$ </td></tr><tr><td>Response time (up/down)</td><td>2 s / 2 s</td><td> $2 1 \mathrm { s } ~ / ~ 1 . 6 \mathrm { s }$ </td></tr><tr><td>Preheat / standby power</td><td></td><td>0.2 kW per module</td></tr><tr><td>SOC / power limits</td><td>12–95% SOC</td><td>60–100% Pnom</td></tr><tr><td>State of health (2025)</td><td>90%</td><td>100%</td></tr></table>

## II. EXPERIMENTS AND ASSET CHARACTERIZATION

Our PowerLabDK platform hosts a wide range of industrial hardware for testing renewable energy and storage technologies under realistic conditions, including off-grid, direct connection to the local Danish grid, and a controllable grid via a power amplifier. Details can be found in [7]. In this study, two assets were involved: a modular BESS for the grid application and a modular AEM electrolyzer for hydrogen production. Both devices can be scaled up for the MWlevel grid-connected system and equipped with high-fidelity monitoring and remote control, enabling the combination of experimental measurements with data-driven market modeling. Table I summarizes the experimental parameters used in the subsequent market optimization and back-testing.

The BESS is rated at 55 kW/79 kWh, comprising eight packs of lithium-ion NMC cells, and ata are logged at a rate of 1 Hz and include active/reactive power, voltage, current, SOC, and temperature. As shown in Fig. 1, multiple capacity test for the system is carried out on the BESS. With the AC power setpoint at 15 kW, the well-controlled behavior at high SOC but reduced behavior at low SOC is evident, where the reduced DC power can limit the power conversion system. Based on extensive testing events, the market-usable performance of the BESS is summarized:

• Usable energy window: the pack can reliably follow power setpoints only within SOC 12 % – 95 %; this corresponds to a market-usable energy of approximately 52 kWh at the 30 kW limit, which is reduced from converter specification 55 kW by connection limitation.

![](images/7cdd1763d37b95490db037e17c8b484ea023d3127320e0225a1fc9b9c9510089.jpg)  
Fig. 1: BESS round-trip test at 15 kW

![](images/782625084c61eac62a041474d6b7f0e61b4de5fe86e8966c3eb535085c0c4622.jpg)  
Fig. 2: BESS capacity test results and SOH projection

![](images/6450764216d3cc018b431363ac7a2e80872915728f0a3c251f165c585ad4046a.jpg)

(a) Upward setpoint change: 86% in 18 s and 100% in 21 s  
![](images/4729bfc52160667fe12c04456da1e81d93fc04171bb58497f83517418a4e54aa.jpg)  
(b) Downward setpoint change: 86% in 1.5 s and 100% in 1.9 s  
Fig. 3: Tested electrolyzer response behavoir

• Round-trip energy efficiency: measured 98% at 50% of power (15 kW) and about 90% at full power (30 kW).

• Setpoint tracking accuracy: 3 % during charging and 7.6 % during discharging from the external measurement.

• Thermal limits: average operating temperature $2 3 . 6 ^ { \circ } \mathrm { C }$ $( 6 { - } 4 9 \ ^ { \circ } \mathrm { C }$ range) under forced-air cooling.

• State of health: With power of (15 kW), useful energy capacity decreased from 72 kWh (2020) to 65 kWh (2025), i.e. 90 % SOH.

Experimental testing was retried between 2021 and 2025, and a simplified cycle-calendar battery SOH estimation is presented in Fig. 2. Previously, the tested BESS were not used frequently, where the major degradation factor is calendar aging; however, a more extensive cycling usage is projected to dominate the cycle life. The empirical battery aging simulation is detailed in [8]. A combined calendar- and cycling-aging model tuned to these measurements indicates that the system can deliver roughly 250 MWh of remaining throughput before reaching 80 % SOH, equivalent to about 1700 full cycles at 30 kW. These empirically derived parameters are directly used in the optimization framework to ensure physically meaningful scheduling.

The electrolyzer installation consists of three modules, each rated at 2.4 kW nominal power and designed for operation up to 30 bar. The system includes a dryer and a deionized water tank. Key experimental characteristics relevant for dispatch modeling are:

• Operating range: 60–100 % of nominal power; below 60 %, the hydrogen–oxygen separation deteriorates, so the units must shut down.

• Preheat phase: electrolyte heating from ambient to $4 5 ^ { \circ } \mathrm { C }$ follows an exponential process; this requires roughly 30 minutes with 0.2 kW power without producing hydrogen.

• Ramp dynamics: ramp-up to full load in 21 s; rampdown in 1.6 s, making the units suitable for up-regulation services in the power system, as shown in Fig. 3

Due to the rapid downward but slower upward response, the electrolyzer primarily qualifies for FFR and FCRN upregulation, resulting in reduced power consumption. Preheat requirements and power-range constraints are explicitly represented in the optimization model as binary-state transitions between idle, standby, and run modes.

## III. MULTI-MARKET PATICIPATION AND MODELING

The BESS operation is formulated as a mixed-integer linear program (MILP) that maximizes daily revenue from energy arbitrage and reserve-capacity provision. Decision variables include charging and discharging power, reserve capacity bids, and the SOE trajectory. The objective function is

$$
\begin{array} { c } { { \displaystyle \operatorname* { m a x } \sum _ { t } \left[ \left( \lambda _ { t } ^ { \mathrm { D A } } - \tau _ { \mathrm { d i s } } \right) p _ { t } ^ { \mathrm { d i s } } - \left( \lambda _ { t } ^ { \mathrm { D A } } + \tau _ { \mathrm { c h } } \right) p _ { t } ^ { \mathrm { c h } } \right. } } \\ { { \displaystyle \left. + \sum _ { m } \lambda _ { t } ^ { m } p _ { t } ^ { m } - c _ { \mathrm { d e g } } E _ { t } ^ { \mathrm { c y c } } \right] } } \end{array}\tag{1}
$$

where $\lambda _ { t } ^ { \mathrm { D A } }$ is the day-ahead energy price, $\tau$ the charging/discharging tariffs, $p _ { t } ^ { m }$ the reserve-capacity bids for market m, $c _ { \mathrm { d e g } }$ the marginal degradation cost, and $E _ { t } ^ { \mathrm { { { \dot { c y c } } } } }$ the equivalent full cycle energy used. The battery SOE is calculated as,

$$
\mathrm { S O E } _ { t } = \mathrm { S O E } _ { t - 1 } + \eta E _ { t } ^ { \mathrm { c h } } - \frac { 1 } { \eta } E _ { t } ^ { \mathrm { d i s } }\tag{2}
$$

with round-trip efficiency η. Mutually exclusive charge/discharge operation is enforced alongside SOE bounds (12–95 %), power limits, and a daily throughput constraint that mitigates excessive degradation and unrealistic cycling.

The electrolyzer scheduling problem is also cast as an MILP but includes unit state transitions (idle, preheat, run) and nonlinear performance characteristics. The objective maximizes hydrogen revenue net of electricity costs, start/stop penalties, and income from reserve-capacity markets:

$$
\begin{array} { l } { \displaystyle \operatorname* { m a x } \sum _ { n , t } \left[ \left( \lambda ^ { \mathrm { H } _ { 2 } } - \kappa _ { \mathrm { r u n } } \right) f _ { t , n } - \lambda _ { t } ^ { \mathrm { s p o t } } p _ { t , n } \right. } \\ { \displaystyle \left. + \sum _ { m } \lambda _ { t } ^ { m } p _ { t , n } ^ { m } - \kappa _ { \mathrm { s t a r t , s t o p } } y _ { t , n } \right] } \end{array}\tag{3}
$$

where $m \in \mathcal { M } _ { \mathrm { c a p } }$ with $\mathcal { M } _ { \mathrm { { c a p } } } = \{ \mathrm { F F R }$ , FCRDU, FCRN} denoting the available reserve-capacity markets, $\lambda _ { t } ^ { m }$ the corresponding capacity prices, and $p _ { t , n } ^ { m }$ the reserve-capacity bid of module n. $K _ { \mathrm { s t a r t , s t o p } }$ represents the start-stop cost each time.

Both optimization problems are implemented in Python and solved with Gurobi. Historical price data for 2022–2025 were obtained from Energinet’s API, and activation factors were synthesized from frequency recordings following the procedure in [9]. Hydrogen production is mapped from electrical power using the fitted quadratic flow model $f _ { t , n } ~ =$ $\alpha _ { 2 } p _ { t , n } ^ { 2 } + \alpha _ { 1 } p _ { t , n } + \alpha _ { 0 }$ , which is valid only when the module is in the running state. The preheat temperature follows the measured exponential law $T ( \bar { t } ) = T _ { 0 } + ( \bar { T } _ { \infty } - T _ { 0 } ) \big ( 1 - e ^ { - k t } \big )$

with $k ~ = ~ 0 . 0 1 9 ~ \mathrm { \ m i n ^ { - 1 } }$ , enforcing a minimum warm-up duration of approximately 30 min before hydrogen production is allowed. Reserve-capacity bids are constrained by the available headroom between $P _ { \mathrm { m i n } }$ and $P _ { \mathrm { m a x } }$ and by the required direction of response (up- or down-regulation). Additional logical constraints enforce state exclusivity, minimum up/down durations, and consistent start/stop identification. For each day, a 24-hour horizon for BESS or a 96 × 15-minute horizon for electrolyzer is optimized, assuming perfect price foresight. While this yields an upper bound on achievable revenue, it enables the relative value of cycling, reserve participation, and physical constraints to be systematically compared.

## IV. RESULTS

The BESS is evaluated under daily throughput limits ranging from 0.1 to 5 cycles per day using three years of historical price data. For a nominal limit of 1 cycle per day, the accumulated revenue reaches approximately $6 . 2 \times 1 0 ^ { 5 }$ DKK. Increasing the limit to 3–5 cycles per day provides less than a 6 % revenue improvement, while accelerating degradation from 11 % to nearly 20 % SOH loss over the same period. These results indicate that, in a multi-market environment dominated by reserve-capacity payments, profitability is largely insensitive to additional cycling once sufficient energy is allocated to maintain eligibility for ancillary services. As shown in Fig. 4 and TABLE II, the operation schedule is detailed, where reserve activation is modeled assuming conversion of 20% of FCRN bids and 5% of FFR/FCRD bids into energy. Specifically, the spot market charges during the low DA price and sales at the end of the day. FFR is activated in the middle of the day when the price is higher. FCRD is very active throughout the day, whereas FCRN is activated for only a few hours. Finally, the SOE generally ships energy from daytime to nighttime. Regarding financial behavior, the majority of the revenue comes from FCRD-D, which accounts for more than 60% of the overall revenue. The main energy throughput is due to the spot market, taking around half of the charging and discharging energy. Revenue-stacking analysis shows that the optimizer relies primarily on FCRD and FCRN products, using the day-ahead market to position the SOE ahead of high-value reserve hours. The inclusion of realistic operational constraints (12–95 % SOC window, 0.9 round-trip efficiency, and setpoint accuracy limits) substantially reduces idealized profits but yields schedules that are feasible for real hardware testing and consistent with PowerLabDK performance characteristics.

For the electrolyzer case, the base-case simulations assume a hydrogen price of 6.61 C/kg (0.0044 DKK ${ \mathrm { N L } } ^ { - 1 } )$ . Under energy-only operation, pure hydrogen production yields only marginal profit due to the high electricity costs and nonnegligible O&M expenses. When ancillary service markets (FFR and FCRN) are included, the total daily profit increases by roughly 300 %, as the plant can monetize its fast downramping capability. The optimizer typically schedules a 30- minute preheat during low-price hours, operates at full load during low or negative spot prices, and curtails consumption when reserve prices peak. The FCRN revenue includes energy payment (FCRN(E)) and capacity payment (FCRN(C)). For multi-market participation shown in TABLE III, the FFR, FCRD-U, and FCRN(C) are the most profitable items in the business case study. As shown in Fig. 5, Hydrogen flow is reduced during FCRN hours because the sale of reserve capacity is more profitable than producing hydrogen. The observed fluctuations in the flow curve reflect the stochastic nature of reserve activations. Doubling the hydrogen price results in nearly a fourfold increase in total revenue. The higher production hours expand the feasible reserve headroom, allowing substantially larger FFR and FCRN bids. This illustrates the nonlinear coupling between PTX revenues and flexibility-market participation.

TABLE II: Revenue and energy exchange in Fig. 4.
<table><tr><td></td><td>Spot</td><td>FFR</td><td>FCRD-U</td><td>FCRD-D</td><td>FCRN</td><td>Totals</td></tr><tr><td>DKK</td><td>103.17</td><td>216.07</td><td>289.01</td><td>1061.76</td><td>79.26</td><td>1749.27</td></tr><tr><td>kWh In</td><td>37.51</td><td></td><td>0.00</td><td>30.92</td><td>8.17</td><td>76.60</td></tr><tr><td>kWh Out</td><td>31.60</td><td>10.80</td><td>17.10</td><td>0.00</td><td>8.17</td><td>67.67</td></tr></table>

![](images/213787a47864ab449bf275ace9f2751ea89564da3b3b8dd4dca68028f10f7907.jpg)  
\*X-axis shows the time for 24h with 1 hour resolution (24 data points)  
Fig. 4: Optimal BESS schedule in all available markets.

Moving forward from single-day optimization to multi-year simulation, our models run for each 24-hour period between 2022 and 2025 for BESS and electrolyser, as shown in Fig. 6 and Fig. 7. Significant variation is observed over time and across different revenue sectors. The daily revenue of BESS has generally been reducing over the years, although some peaks are imposed by the FCRD market contribution. The market dynamic is significant, where the FFR in 2022, FCRD in 2023, and FCRN in 2024 reach their peaks in turn. It is interesting to observe that the production of pure hydrogen often leads to losses, as the price is quite low and barely covers the assumed operating costs, while at the same time, electricity prices for consumption are high. Nevertheless, the operation remains profitable, with profitability driven by the reserve markets, which follow similar trends to the BESS case study with market participation.

TABLE III: Financial outcomes of in Fig. 5.
<table><tr><td>Item/[DKK]</td><td>H2</td><td>FFR</td><td>FCRD-U</td><td>FCRN(E)</td><td>FCRN(C)</td><td>Total</td></tr><tr><td>Revenue</td><td>243.86</td><td>82.86</td><td>89.96</td><td>3.35</td><td>76.05</td><td>496.08</td></tr><tr><td>Cost</td><td>204.95</td><td>0.00</td><td>0.00</td><td>1.67</td><td>0.00</td><td>206.62</td></tr><tr><td>Profit</td><td>38.91</td><td>82.86</td><td>89.96</td><td>1.67</td><td>76.05</td><td>289.46</td></tr></table>

![](images/267b7a8023bb12c6a581a34adc069f02007fa05275e9f3fbdc4aa6bc669c2ce5.jpg)  
\*X-axis shows the time for 24h with 15-minute resolution (96 data points)  
Fig. 5: Optimal electrolyzer schedule in all available markets.

## V. DISCUSSION

The results of the monthly revenues are shown in Fig. 8a with 1-cycle-per-day limit. It is evident that profits have been decreasing since mid-2023. Specifically, the spot-market results are often negative, where the battery uses this market to charge itself in preparation for reserve provision. BESS does not reach its daily cycling limit most of the time, as multimarket operations reserve most of the capacity, effectively limiting the flow of energy. The overall outcomes of this operation are given in Fig. 8b. The total profits amount to 622,437 DKK at the cost of 751 cycles and an 11% decrease in the SOH within 3 years. Furthermore, a sensitivity study is conducted on the electrolyzer case with varying hydrogen prices and operation costs, as shown in Fig. 8c, where an increase in hydrogen price significantly improves economics. We also find neglecting setpoint accuracy or preheat delays can overestimate market revenues by 10 to 20%. The presented framework can be scaled up, and the combination of laboratory validation and market-based optimization provides a template for similar facilities and hybrid energy systems.

![](images/858a9e30d46708738556d7ce1f0277f1705b34037023e2439f2a7acad4fd7fdd.jpg)  
Fig. 6: BESS case study with 1 cycle per day limit.

![](images/67d231bfe219feb570aee5f0280f55695bfa7dc0b5b5ccec00dc97064949fdff.jpg)  
Fig. 7: Electrolyser case study with 6.61 EUR/kg hydrogen price.

## VI. CONCLUSION

This work presented an integrated experimental–modelingoptimization framework for evaluating multi-market operation of a 55 kW/79 kWh BESS and a modular 3 ×2.4 kW electrolyzer, embedding hardware performance such as the BESS’s 52 kWh usable energy window, 90–98% efficiency, and measured electrolyzer ramp times (21 s up, 1.9 s down).

![](images/2ea4284acb68e2d4d5b9084281c0615711f91c57e3f20bcb2ddfe8577f614b0a.jpg)

(a) Monthly profits for the 1 cycle/day strategy  
![](images/90bf2e74071105c63d4531e550dbf351db565cb72f72c67c825f3d9e3443e73d.jpg)

(b) Cumulative results of BESS with 1 cycle/day  
![](images/701352b1573cfdf009a1c05b04b8f81a004d25d730919cae88a280432df77369.jpg)  
(c) PTX business case comparison  
Fig. 8: Multi-year performance of the BESS and electrolyzer under multi-market participation.

Results show that reserve markets dominate the value, and hydrogen production is only marginally profitable. However, participation in FFR, FCRD, and FCRN significantly increases daily profit. Overall, the study demonstrates that realistic modeling of physical performance and multi-market participation is crucial for assessing the economic potential of such assets.

## REFERENCES

[1] K. Shivarama Krishna and K. Sathish Kumar, “A review on hybrid renewable energy systems,” Renewable and Sustainable Energy Reviews, vol. 52, pp. 907–916, 2015, doi: 10.1016/j.rser.2015.07.187.

[2] C. Zhao, P. B. Andersen, C. Træholt, and S. Hashemi, “Grid-connected battery energy storage system: A review on application and integration,” Renewable and Sustainable Energy Reviews, vol. 182, p. 113400, 2023, doi: 10.1016/j.rser.2023.113400.

[3] X. Jin, C. Zhao, C. Træholt, and S. You, “Experiments applied to electrolyzer system powered by renewable energy sources: A literature review,” in Proc. 10th Asia Conf. Power and Electrical Engineering (ACPEE), 2025, pp. 2517–2521, doi: 10.1109/ACPEE64358.2025.11041178.

[4] Energinet, “Market requirements.” [Online]. Available: https://en.energinet.dk/electricity/balancing-and-ancillary-services/ get-started-with-ancillary-services/market-requirements/. Accessed: Aug. 17, 2026.

[5] J. Hjalmarsson, K. Thomas, and C. Bostrom, “Service stacking using¨ energy storage systems for grid applications—A review,” Journal of Energy Storage, vol. 60, p. 106639, 2023, doi: 10.1016/j.est.2023.106639.

[6] M. Tofighi-Milani, S. Fattaheian-Dehkordi, and M. Lehtonen, “Electrolysers: A review on trends, electrical modeling, and their dynamic responses,” IEEE Access, vol. 13, pp. 39870–39885, 2025, doi: 10.1109/ACCESS.2025.3546546.

[7] C. Zhao et al., “Lab-field multi-energy platform: Electrolyzer, redox flow battery, and lithium-ion battery energy storage system,” Energy Proceedings, vol. 54, 2024, doi: 10.46855/energy-proceedings-11540.

[8] C. Zhao, P. B. Andersen, C. Træholt, and S. Hashemi, “Data-driven cyclecalendar combined battery degradation modeling for grid applications,” in Proc. IEEE Power & Energy Society General Meeting (PESGM), 2022, pp. 1–5, doi: 10.1109/PESGM48719.2022.9917143.

[9] A. Thingvad, C. Ziras, G. Le Ray, J. Engelhardt, R. R. Mosbæk, and M. Marinelli, “Economic value of multi-market bidding in Nordic frequency markets,” in Proc. Int. Conf. Renewable Energies and Smart Technologies (REST), 2022, doi: 10.1109/REST54687.2022.10023471.