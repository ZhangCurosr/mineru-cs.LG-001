Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000. Digital Object Identifier PLACEHOLDER

# A comparison between ceiling-mounted FMCW, IR-UWB and Wi-Fi radar for in-bedroom human activity monitoring and sleep interruption detection

ANTON LAMBRECHT<sup>1</sup>, REDA EL HAIL<sup>2,3</sup>, XIANJUN JIAO<sup>1</sup>, PIETER CROMBEZ<sup>4</sup>, DOMINIQUE SCHREURS<sup>5</sup>, PETER KARSMAKERS<sup>2,3</sup>, ADNAN SHAHID<sup>1</sup>, AND ELI DE POORTER<sup>1</sup>

<sup>1</sup>IDLab, Department of Information Technology, Ghent University-imec, iGent Tower, Technologiepark-Zwijnaarde 126, B-9052 Ghent, Belgium. (email: Anton.Lambrecht@Ugent.be)

<sup>2Department</sup> <sup>of</sup> <sup>Computer</sup> <sup>Science,</sup> <sup>Leuven</sup> <sup>AI,</sup> <sup>KU</sup> <sup>Leuven,</sup> <sup>B-2440</sup> <sup>Geel,</sup> <sup>Belgium</sup> <sup>(e-mail:</sup> <sup>Reda.Elhail@kuleuven.be)</sup>g

<sup>3</sup>Flanders Make, MPRO, B-3000 Leuven, Belgium (e-mail: Reda.Elhail@kuleuven.be)

<sup>4</sup>Televic Healthcare, 8870 Izegem, Belgium

<sup>5</sup>Div. ESAT-WAVECORE, KU Leuven, Belgium (e-mail:dominique.schreurs@kuleuven.be)

Corresponding author: Anton Lambrecht (email: Anton.Lambrecht@Ugent.be).

The research that led to these results has received funding from the DistriMuSe project (HORIZON-KDT-JU-2023-2-RIA) with Grant No 101139769, the CORNET project DARTS (HBC.2023.0772) and from the NAV-ALERT project from the Belgian Defense through contract number 24DEFRA009.

ABSTRACT Despite their growing importance for contact-free radio frequency (RF) based healthcare monitoring, different radio technologies such as frequency-modulated continuous wave (FMCW) radar, impulse radio ultra-wideband (IR-UWB), and Wi-Fi sensing are rarely compared under identical deployment conditions, as existing studies typically differ in hardware, datasets, and evaluation methodologies. In addition, the performance of ceiling-mounted radars, despite their practical deployment and cost advantages in healthcare environments, remain underexplored. Therefore, this paper presents a controlled comparison and analysis of ceiling-mounted FMCW, IR-UWB, and Wi-Fi sensing using synchronized recordings from 20 participants across six room layouts. All technologies are evaluated with the same convolutional neural network (CNN) on both a fine-grained 10-class human activity recognition (HAR) task and a coarse 4-class sleep monitoring task. IR-UWB achieves the highest cross-subject activity recognition performance (89.0% macro F1), while FMCW generalizes best to unseen room layouts (83.8% macro F1). For sleep monitoring, all technologies exceed 92% macro F1 in unseen environments. The results reveal a fundamental trade-off between recognition performance and environmental robustness, which can be explained through differences in range resolution, antenna diversity, Doppler resolution, and spatial information retention. These findings provide practical guidelines for the design of healthcare-oriented RF sensing systems.

INDEX TERMS ceiling-mounted, convolutional neural networks, FMCW radar, human activity recognition, impulse-radio ultra-wideband (IR-UWB), remote healthcare, sensor comparison, sleep monitoring, Wi-F radar

## I. INTRODUCTION

W <sup>ITH</sup> <sup>the</sup> <sup>growing</sup> <sup>demand</sup> <sup>for</sup> <sup>non-intrusive</sup> <sup>monitor-</sup> ing in healthcare and assisted living, contactless human activity recognition (HAR) is emerging as a promising approach for monitoring patient behavior and assessing sleep quality in both domestic and clinical settings. By 2050, the global population aged over 60 is projected to reach 2.1 billion, intensifying the need for scalable and privacypreserving life-quality monitoring solutions [1]. Sleep disorders are particularly prevalent among older adults: a metaanalysis spanning 36 countries found that 40% experience poor sleep quality and 29% suffer from insomnia [2]. These conditions are strongly associated with increased fall risk, cognitive decline, dementia onset, and reduced quality of life. Reliable monitoring of human activity and sleep behavior can enable early detection of health deterioration, support timely intervention, and improve patient safety and quality of care.

Current monitoring systems often rely on camera-based approaches, which raise significant privacy and acceptability concerns in sensitive environments. A survey of 304 informal caregivers of people with dementia showed that radio frequency (RF) sensing systems are rated more acceptable (4.0/5) than camera-based systems (3.1/5) [3], motivating the use of contactless RF-based sensing as a privacy-preserving alternative. Among RF-based technologies, impulse-radio ultra-wideband (IR-UWB), frequency-modulated continuous wave (FMCW) radar, and Wi-Fi represent three prominent approaches with distinct design trade-offs. Wi-Fi is widely deployed and offers the potential for sensing with minimal additional infrastructure, but is constrained by limited bandwidth. In contrast, FMCW is a mature sensing-first modality with limited communication integration, while IR-UWB occupies a middle ground, offering higher range resolution than Wi-Fi while maintaining full communication capabilities with IEEE-compliant commercial off-the-shelf (COTS) hardware.

This paper presents a comparative study of machine learning (ML)-based HAR and sleep monitoring across three COTS sensing technologies: FMCW radar, IR-UWB, and Wi-Fi. We evaluate performance using both a fine-grained activity label set for general healthcare use cases and a coarser sleep-specific label set. A uniform convolutional neural network (CNN) architecture is employed to ensure that observed performance differences reflect signal characteristics rather than model design. This enables a fair assessment of each modality’s suitability for reliable, privacy-preserving HAR.

The contributions of this paper are:

1) A controlled comparison of FMCW, IR-UWB, and Wi-Fi sensing using synchronized recordings acquired simultaneously in an identical ceiling-mounted deployment. All modalities are processed using the same learning pipeline and evaluated under multiple crossvalidation protocols, allowing the observed differences to be attributed to the sensing technology rather than the experimental design.

2) A signal-level analysis linking sensing properties such as range resolution, Doppler resolution, antenna diversity, and spatial information retention to HAR performance and robustness. Based on these insights, practical guidelines are derived for selecting suitable signal representations and preprocessing strategies for different RF sensing technologies.

3) An open synchronized dataset containing FMCW, IR-UWB, and Wi-Fi measurements from 20 participants across six room layouts, enabling reproducible benchmarking and future research on cross-modality RF sensing.

## II. RELATED WORKS

Table 1 summarizes representative RF-based HAR studies. Although numerous works have investigated FMCW radar, IR-UWB, or Wi-Fi individually, controlled comparisons between sensing modalities in identical conditions remain scarce. Existing comparative studies typically differ in sensing hardware, deployment geometry, dataset design, or evaluation methodology, making it difficult to attribute observed performance differences to the sensing modality itself.

Most prior work focuses on a single sensing technology. IR-UWB studies [4], [5] predominantly employ sensingoriented, non-IEEE-compliant hardware in monostatic configurations, whereas Wi-Fi-based approaches [6], [7], [11] almost exclusively rely on bi-static deployments. This is largely because Wi-Fi sensing studies typically leverage off-the-shelf commodity devices, where access to the transmitted and received signals is distributed across separate nodes. In contrast, SDR-based implementations such as openwifi provide ful control over the physical layer and enable signal transmission and reception on the same platform, thereby facilitating monostatic radar operation. Therefore, although monostatic Wi-Fi radar has been explored, existing work has primarily focused on respiration-rate determination [13]–[15], rather than on HAR.

Another important distinction is deployment geometry. Most prior work adopts side-looking or wall-mounted configurations, while ceiling-mounted systems remain comparatively rare. Although side-looking deployments ( [6]–[10], [12]) maximize the human radar cross-section and simplify the sensing task, ceiling-mounted sensing is considerably easier to integrate non-intrusively into real-world healthcare environments and this under-explored research area is therefore the focus of this work.

Thirdly, cross-modality studies remain limited and do not isolate the sensing modality as the primary experimental variable. Only a few works compare different radar technologies to study domain adaptation or modality fusion [8]– [10]. These studies evaluate sensing-first radar platforms (e.g., custom IR-UWB systems and millimeter-wave radar) that share similar sensing principles and are generally not IEEE-compliant communication systems. While these works demonstrate transferability or complementarity across radar modalities, they do not provide a controlled comparison of the sensing technologies themselves. Other studies compare IR-UWB with Wi-Fi [11] or FMCW with Wi-Fi [12], but both pair a monostatic radar with a spatially separated Wi-Fi link, so the geometry changes together with the sensing technology. Chen et al. evaluate IR-UWB with more than 30 participants across seven environments, although the characteristics of the Wi-Fi measurements are not separately reported. Dahal et al. evaluate both modalities in a single controlled laboratory environment with five participants.

TABLE 1. Comparison of representative RF-based human activity recognition studies in terms of sensing hardware, deployment setup, and dataset characteristics. In contrast to previous work, this study compares FMCW, IR-UWB, and Wi-Fi using synchronized recordings acquired simultaneously in an identical ceiling-mounted setup, enabling a controlled comparison across sensing modalities. ✓/✗ indicate presence/absence of the property. n/r = not reported. <sup>†</sup> In this paper, the Wi-Fi transmitter is ceiling-mounted while the receiver is placed on the floor.
<table><tr><td rowspan="2">Ref.</td><td rowspan="2">Year</td><td colspan="3">Hardware</td><td rowspan="2">Dataset</td><td colspan="6"></td></tr><tr><td>Type</td><td>Device</td><td>Setup</td><td>#ppl</td><td>#rooms</td><td>Ceiling mount</td><td></td><td>#lbl</td><td>Nat. flow</td></tr><tr><td>[4]</td><td>2023</td><td>IR-UWB</td><td>X4M02</td><td>monostatic</td><td></td><td>10</td><td>2</td><td></td><td>√</td><td></td><td>x</td></tr><tr><td>[5]</td><td>2025</td><td>IR-UWB</td><td>P440</td><td>monostatic</td><td></td><td>10</td><td></td><td></td><td></td><td>2-5-8</td><td>X</td></tr><tr><td>[6]</td><td>2024</td><td>Wi-Fi</td><td>Intel 5300</td><td>bi-static</td><td>5</td><td></td><td></td><td>x</td><td></td><td></td><td></td></tr><tr><td>[7]</td><td>2025</td><td>Wi-Fi</td><td>Intel 5300 Xethru X4</td><td>bi-static</td><td>10</td><td></td><td></td><td>X</td><td></td><td>6</td><td>x</td></tr><tr><td>[8]-[10]</td><td>2020, 2025</td><td>IR-UWB, FMCW</td><td>Ancortek SDR-Kit TI IWR1443</td><td>monostatic</td><td>6</td><td></td><td>n/r</td><td>x</td><td></td><td>11</td><td>x</td></tr><tr><td>[11]</td><td>2023</td><td>IR-UWB Wi-Fi</td><td>Xethru Intel 5300</td><td>monostatic bi-static</td><td>30 n/r</td><td></td><td>7 n/r</td><td>x†</td><td></td><td>7</td><td>x</td></tr><tr><td>[12]</td><td>2024</td><td>FMCW Wi-Fi</td><td>Radarbook2 RPi 3B+ (Nexmon)</td><td>monostatic bi-static</td><td>5</td><td></td><td>1</td><td>x</td><td></td><td>7</td><td>x</td></tr><tr><td>This work</td><td></td><td>FMCW IR-UWB Wi-Fi</td><td>IWR6843AOP DW3000 ZedBoard/openwifi</td><td>monostatic pseudo monostatic</td><td>20</td><td></td><td>6</td><td>√</td><td></td><td>10</td><td>√</td></tr></table>

Experimental design further complicates comparison. Existing datasets are generally limited in participant or environmental diversity and commonly record activities as repeated isolated actions rather than continuous activity sequences. While repeated executions simplify recognition, they remove the transitions and execution variability encountered in practical deployments, making the resulting performance less representative of real-world use.

In contrast, this work isolates the sensing modality as the primary experimental variable by evaluating FMCW, IR-UWB, and Wi-Fi using synchronized recordings acquired simultaneously in an identical ceiling-mounted configuration (Fig. 1). All modalities are processed using the same learning pipeline and evaluated under identical protocols on a dataset comprising 20 participants, six room layouts, and natural activity flows.

## III. SYSTEM MODEL

As shown in Fig. 1, FMCW, IR-UWB and Wi-Fi capture time-varying channel reflections through different representations: beat signals, channel impulse response (CIR) and channel state information (CSI), respectively. Despite these measurement differences, all three can be described within a common system-level framework, which is introduced first and subsequently specialized for each technology.

## A. PROBLEM STATEMENT

All three sensing modalities observe human motion through time-varying reflections of the wireless propagation channel and can therefore be modeled as different measurements of the same dynamic multipath environment. FMCW probes the channel with chirps, IR-UWB with pulses and Wi-Fi with orthogonal frequency division multiplexing (OFDM) packets from which per-subcarrier CSI is extracted. Stacking successive complex-valued channel observations yields the measurement matrix

$$
X ^ { ( j ) } = x ^ { ( j ) } [ m , k ] , \qquad 1 \leq m \leq M _ { j } , 1 \leq k \leq K _ { j } ,\tag{1}
$$

where $j \in \{ \mathrm { F M C W , ~ I R \mathrm { - U W B } , ~ W i \mathrm { - F i } } \}$ denotes the sensing technology. The row index m represents slow-time (successive chirps, pulses, or packets), with

$$
M _ { j } = \boldsymbol { w } \cdot \boldsymbol { s r _ { j } } ,\tag{2}
$$

for an observation window of duration w and slow-time sampling rate $s r _ { j }$ . The column index k spans a modalitydependent fast dimension, whose physical meaning and size $K _ { j }$ follow from the acquisition principle:

$$
K _ { j } = \left\{ \begin{array} { l l } { N _ { s } , } & { j = \mathrm { { F M C W ~ \Phi } ( \mathrm { b e a t . } s i g n a l ~ s a m p l e s ~ p e r ~ c h i r p } ) , } \\ { N _ { f } , } & { j = \mathrm { { I R \mathrm { - } U W B \quad } ( f a s t \mathrm { - } t i m e ~ d e l a y ~ b i n s ) , } } \\ { N _ { c } , } & { j = \mathrm { { W i \mathrm { - } F i } \quad ( O F D M ~ s u b c a r r i e r s } ) . } \end{array} \right.\tag{3}
$$

Independently of the modality, each measurement can be decomposed as

$$
x ^ { ( j ) } [ m , k ] = s ^ { ( j ) } [ k ] + d ^ { ( j ) } [ m , k ] + w ^ { ( j ) } [ m , k ] ,\tag{4}
$$

where $s ^ { ( j ) } [ k ]$ collects the contributions of static objects, which are largely time-invariant along slow-time, $d ^ { ( j ) } [ m , k ]$ contains the structured temporal variations in amplitude and phase induced by human motion, and $w ^ { ( j ) } [ m , k ]$ is measurement noise. It is the dynamic component $d ^ { ( j ) } [ m , k ]$ that carries the activity information and forms the basis for activity recognition.

The objective is to learn, for each modality j, a classification function

$$
f \left( X ^ { ( j ) } \right) = A ,\tag{5}
$$

mapping the observation matrix to the activity label A associated with the corresponding time window. The activity labels

are defined in Section IV-C. The remainder of this section specifies the physical origin of $x ^ { ( j ) } [ m , k ]$ and the interpretation of the fast dimension for each sensing modality.

## 1) FMCW

An FMCW radar continuously transmits a linearly frequencyswept signal, or chirp, whose instantaneous frequency sweeps from $f _ { 0 } \mathrm { t o } f _ { 0 } + B$ over a chirp duration $T _ { c }$ at a chirp rate $\mu =$ $B / T _ { c }$ . The transmitted signal is:

$$
s _ { \mathrm { t x } } ( t ) = A _ { \mathrm { t x } } \exp \left( j 2 \pi \left( f _ { 0 } t + \frac { \mu } { 2 } t ^ { 2 } \right) \right) .\tag{6}
$$

The echo reflected by a target at range R is a delayed replica of $s _ { \mathrm { t x } } ( t )$ . An analog mixer multiplies the received signal with the live transmit waveform, producing a beat signal at a single tone whose frequency is directly proportional to range:

$$
f _ { b } = \frac { 2 \mu R } { c } .\tag{7}
$$

After low-pass filtering, the beat signal is digitised by an ADC at sampling rate $f _ { s } ,$ , and a quadrature demodulator yields the complex in-phase and quadrature (IQ) samples. Chirps are transmitted repeatedly, so the chirp index takes the role of the slow-time index m (with chirp repetition rate sr<sub>FMCW</sub>), while the $N _ { s }$ ADC samples within one chirp form the fast dimension k of Equation (3). For a single target, the resulting measurement matrix is

$$
x ^ { \mathrm { ( F M C W ) } } [ m , k ] = A \exp \left( j 2 \pi \left( \frac { f _ { b } k } { f _ { s } } + \frac { \Delta \phi m } { 2 \pi } \right) \right) + w [ m , k ] ,\tag{8}
$$

where $\Delta \phi = 4 \pi \nu T _ { r } / \lambda$ is the inter-chirp Doppler phase shift encoding radial velocity v, and $w [ m , { \bar { k } } ] \ \sim \ { \mathcal { C N } } ( 0 , \sigma ^ { 2 } )$ is additive complex Gaussian noise, matching the noise term of Equation (4). A static target $( \nu = 0 )$ yields a constant phase progression along k only, whereas a moving person additionally modulates the phase along m. A 2D-FFT applied along the fast-time and slow-time dimensions of $x ^ { \mathrm { ( F M C W ) } } [ m , k ]$ yields the range-Doppler map used for target detection and parameter estimation.

## 2) IR-UWB

Ultra-wideband (UWB) transmits over an absolute bandwidth exceeding 500 MHz, enabling extremely short time-domain pulses. A 500 MHz system produces pulses of approximately 2 ns, yielding finer ranging resolution than narrowband technologies.

An IR-UWB radar periodically transmits these short pulses and observes the resulting reflections from objects in the environment. Reflections from more distant objects arrive later, each distinct propagation path constituting a multipath component. The superposition of these delayed reflections forms the CIR,

$$
\mathrm { C I R } ( t ) = \sum _ { p } A _ { p } \delta ( t - \tau _ { p } ) + n ( t ) ,\tag{9}
$$

where $A _ { p }$ and $\tau _ { p }$ are the amplitude and delay of path $p ,$ and $n ( t )$ is noise. Static objects produce constant multipath components over time, while a moving person introduces time-varying changes in amplitude and phase, in accordance with the decomposition of Equation (4).

Successive pulse transmissions produce a sequence of CIRs that capture the temporal evolution of the propagation channel. Each transmitted and successfully received UWB packet yields one CIR, such that the slow-time axis corresponds to successive packet transmissions. Packets are transmitted at regular intervals on the order of milliseconds, resulting in a slow-time sampling rate sr as defined in Equation (2), which is directly determined by the packet transmission frequency. For each received packet m, the reflected signal components are sampled as complex IQ values at nanosecond resolution, forming the fast-time samples. This fast-time resolution is determined by the timing resolution of the UWB radio and defines the granularity with which multipath components can be resolved in delay. The k-th fasttime sample corresponds to delay bin $\tau _ { k }$ , so that

$$
x ^ { ( \mathrm { I R - U W B } ) } [ m , k ] = \mathrm { C I R } _ { m } ( \tau _ { k } ) , \qquad 1 \leq k \leq N _ { f } ,\tag{10}
$$

where $\mathrm { C I R } _ { m }$ denotes the CIR measured for the m-th received packet. Each row ofX<sup>(IR-UWB)</sup> thus represents a single packetlevel CIR. The slow-time axis captures the evolution of the channel across consecutive packet transmissions, while the fast-time axis maps directly to propagation delay and, consequently, range.

## 3) Wi-Fi

Wi-Fi is a communication-first technology that transmits OFDM waveforms over multiple narrowband subcarriers. While designed for communication, the received packets also provide channel estimates that can be exploited for sensing. Each received packet provides one channel estimate per subcarrier, so the packet index takes the role of the slow-time index m (with packet rate sr ), and the $N _ { c }$ subcarriers form the fast dimension k. The key sensing signal is the persubcarrier CSI,

$$
x ^ { \mathrm { ( W i . F i ) } } [ m , k ] = \sum _ { p } \alpha _ { p } [ m ] e ^ { - j 2 \pi f _ { k } \tau _ { p } [ m ] } + w [ m , k ] ,\tag{11}
$$

where $\alpha _ { p } [ m ]$ and $\tau _ { p } [ m ]$ denote the complex attenuation and delay of path p at slow-time $m , f _ { k }$ is the frequency of subcarrier k, and w $[ m , k ]$ is measurement noise [16]. Paths reflecting off static objects have essentially constant $\alpha _ { p }$ and $\tau _ { p }$ and thus contribute to the static component $s ^ { ( \mathrm { W i - F i } ) } [ k ]$ of Equation (4), while human motion causes time-varying $\alpha _ { p } [ m ]$ and $\tau _ { p } [ m ]$ producing the dynamic variations $d ^ { ( \mathrm { W i - F i } ) } [ m , \dot { k } ]$ in both amplitude and phase.

## IV. EXPERIMENTAL SETUP A. MEASUREMENT SETUP

The dataset was collected in the HomeLab in Zwijnaarde [17], a residential test environment developed by Ghent University and imec for evaluating Internet of Things (IoT), smart-home, and healthcare applications. Measurements were conducted in a bedroom environment representative of an assisted-living setting, containing a bed, chair, and table. As shown in Fig. 1, the sensing hardware was mounted on the ceiling above the bed and remained fixed throughout the measurement campaign.

![](images/492a9930588f3412ef620cc0fdf6bec79a8ee359c8af5660d3eaee9ea1c2cea3.jpg)  
FIGURE 1. Complete system overview from data acquisition to activity classification. The pipeline illustrates the physical measurement setup (left), the raw signal acquisition and mathematical models for FMCW, IR-UWB, and Wi-Fi (center-left), the modality-specific preprocessing into Time-Doppler and Range-Doppler representations (center-right), and the uniform Convolutional Neural Network (CNN) architecture used to predict the final activity classes.

A key aspect of the measurement campaign is that FMCW, IR-UWB, and Wi-Fi data were recorded simultaneously to a common time reference, allowing all three modalities to observe identical activity executions. Consequently, performance differences can be attributed to the sensing modality rather than to differences in participant behaviour or activity execution.

To evaluate robustness across environments, six room layouts were created by varying the bed and chair positions while keeping the ceiling-mounted sensing hardware fixed (Fig. 2). Layouts 1, 2, 5, and 6 place the bed beneath or close to the radar, whereas layouts 3 and 4 position it farther away or at a less favorable orientation, creating more challenging conditions for in-bed activity recognition. Seated activities are similarly more challenging when the chair is farther from the radar or positioned at a wider angle relative to the sensing direction.

A total of 20 participants (14 male and 6 female, aged 21–67 years, height: 159–189 cm, weight: 50–92 kg) completed the protocol in six room layouts, yielding up to 120 person-scenario recordings. The protocol and consent procedure were approved by the relevant ethics committees of KU Leuven (SMEC; G-2024-8332) and Ghent University, and all participants provided written informed consent.

Activity execution was guided by text-to-speech (TTS) instructions describing realistic actions, such as Walk to the left side ofthe bed, Sit down on the edge ofthe bed, or Go to the bathroom and wait there for a moment. Unlike protocols that instruct participants to repeatedly perform isolated activities, these instructions encouraged participants to act within a natural context rather than repeatedly executing isolated movements. Participants were free to perform each action using their own gait, posture, and speed, resulting in realistic variability in both activities and transitions. This still leaves the timing of each action externally cued rather than fully spontaneous, but preserves natural movement transitions that repeated, isolated actions do not capture.

![](images/0a76df086632bc22c8668efc92903fad8a71fdd2e2143d41b8969a59f3fcadb0.jpg)  
FIGURE 2. Overview of the six room setups used in the study. The bed and chair positions were varied to create different subject positions relative to the fixed ceiling-mounted radar setup. The radar position remained unchanged across all scenarios and is indicated by the coloured markers.

## B. HARDWARE

The FMCW data was recorded with a Texas Instruments IWR6843AOP radar module operating in the 60-64 GHz band. The device integrates all sensing hardware on a single chip and provides three transmit and four receive antennas in a monostatic configuration, yielding N<sub>r</sub> = 12 virtual channels. Within each 50 ms frame, 96 chirp loops are transmitted per antenna per frame which determines the unambiguous Doppler range. Its main configuration parameters are listed in Table 2.

The IR-UWB data was collected with the Qorvo QM33120WDK1, a low-cost COTS development kit based on the DW3000 transceiver, operating in UWB channel 5 (centered at 6.49 GHz with 499.2 MHz bandwidth). The setup uses separate transmitter and receiver nodes, each consisting of a Nordic nRF52840 development kit (DK) and a DW3000 shield, placed approximately 30 cm apart. Despite this physical separation, the small baseline relative to the monitored area allows the setup to be treated as pseudo-monostatic. In contrast to the FMCW radar and Wi-Fi platform, the two IR-UWB nodes are controlled by separate boards without a shared clock backbone, which introduces a time-varying phase drift [18] corrected in the preprocessing of Equation (14). The transmitter uses an omnidirectional antenna, while the receiver uses a directional antenna to increase sensitivity towards the monitored area. One CIR is acquired every 6.67 ms, which sets the slow-time rate sr<sub>IR-UWB</sub> = 150 Hz. The corresponding configuration parameters are listed in Table 2.

The Wi-Fi CSI data was collected with a ZedBoard fieldprogrammable gate array (FPGA) development kit equipped with an AD-FMCOMMS2-EBZ RF front-end [13]. The platform runs openwifi [19], an open source software defined radio (SDR) implementation of IEEE 802.11a/g/n that exposes CSI from the physical-layer channel estimator. The system operates in a monostatic configuration with a shared transmitter-receiver clock, eliminating carrier frequency offset, sampling frequency offset, and symbol timing offset. Packets are transmitted at a mean interval of 2 ms, corresponding to a mean slow-time rate of $s r _ { \mathrm { W i - F i } } = 5 0 0$ Hz. The remaining configuration parameters are listed in Table 2.

The hardware-determined performance parameters in Table 2 follow directly from these configurations. The range resolution is $\Delta R = c / ( 2 B )$ , so the bandwidth ratio of roughly 2:1:0.04 between FMCW, IR-UWB, and Wi-Fi translates into range resolutions of 0.126 m, 0.30 m, and 7.5 m: FMCW resolves the body into multiple range cells, IR-UWB into a few, and Wi-Fi’s single range bin exceeds the room dimensions. The maximum unambiguous velocity is $\nu _ { \mathrm { m a x } } = \lambda / ( 4 T _ { \mathrm { s l o w } } )$ where $T _ { \mathrm { s l o w } }$ is the slow-time sampling interval: the chirp repetition period for FMCW, and the frame or packet interval for IR-UWB and Wi-Fi. All three comfortably exceed the velocities of typical in-room human motion. The Doppler resolution, by contrast, is not a hardware constant but is determined by the coherent observation interval used for Doppler processing, $\Delta \nu = \lambda / ( 2 T _ { \mathrm { o b s } } )$ , and the corresponding values are therefore established in Section V-A. They are nevertheless included in Table 2 to keep all comparison parameters in a single reference table.

## C. LABEL DEFINITIONS

Each TTS instruction was mapped to an activity label during annotation. Two label sets were derived from the same recordings: a fine-grained set for detailed HAR and a coarse sleep-monitoring set for deployment-oriented sleep analysis. Semantically equivalent instructions were mapped to a common activity label irrespective of location. For example, Sit down on the edge of the bed and Sit down on the chair are both mapped to ‘‘SIT DOWN’’. In both label sets, ‘‘NO ACTIVITY’’ denotes no relevant movement in the monitored bedroom area, including intervals where the participant is outside the room and should not produce an in-room activity signature.

TABLE 2. Configuration and performance parameters of the three sensing modalities.
<table><tr><td>Parameter</td><td>FMCW</td><td>IR-UWB</td><td>Wi-Fi</td></tr><tr><td>Antenna setup</td><td></td><td></td><td></td></tr><tr><td>TX antennas</td><td>3</td><td>1</td><td>1</td></tr><tr><td>RX antennas</td><td>4</td><td>1</td><td>1</td></tr><tr><td>Signal</td><td></td><td></td><td></td></tr><tr><td>Center frequency</td><td>61.34 GHz</td><td>6.49 GHz</td><td>2.4 GHz</td></tr><tr><td>Wavelength</td><td>4.94 mm</td><td>46.2 mm</td><td>125 mm</td></tr><tr><td>Bandwidth</td><td>1180.05 MHz</td><td>500 MHz</td><td>20 MHz</td></tr><tr><td>Slope</td><td>54.71 MHz/µs</td><td>n/r</td><td>n/r</td></tr><tr><td>Acquisition</td><td></td><td></td><td></td></tr><tr><td>Frame interval</td><td>50ms</td><td>6.67 ms</td><td>2 ms</td></tr><tr><td>Slow-time interval Tslow</td><td>0.28 ms</td><td>6.67 ms</td><td>2 ms</td></tr><tr><td>Chirp loops</td><td>96</td><td>n/r</td><td>n/r</td></tr><tr><td>Sampling frequency</td><td>2950 ksps</td><td>n/r</td><td>n/r</td></tr><tr><td>Range samples</td><td>64</td><td>50</td><td>n/r</td></tr><tr><td>Subcarriers</td><td>n/r</td><td>n/r</td><td>53</td></tr><tr><td>Performance</td><td></td><td></td><td></td></tr><tr><td>Range resolution</td><td>0.126 m</td><td>0.300 m</td><td>7.5 m</td></tr><tr><td>Maximum range</td><td>6.46 m</td><td>7.5 m</td><td>n/r</td></tr><tr><td>Doppler resolution</td><td>0.093 m/s</td><td>0.009 m/s</td><td>0.156 m/s</td></tr><tr><td>Maximum Doppler</td><td>4.471 m/s</td><td>1.73 m/s</td><td>15.6 m/s</td></tr></table>

The fine-grained label set contains ten classes: ‘‘NO AC-TIVITY’’, ‘‘WALK’’, ‘‘STAND UP’’, ‘‘SIT DOWN’’, ‘‘LIE ON BED’’, ‘‘GET UP BED’’, ‘‘ANXIOUS’’, ‘‘EATING’’, ‘‘WAVE HANDS’’, and ‘‘CLAP HANDS’’. It preserves detailed activity categories, including walking, posture transitions, in-bed restlessness, and hand movements, to evaluate how well each sensing modality distinguishes fine and potentially subtle motion patterns.

For sleep-monitoring analysis, the activity labels were remapped into four coarse classes forming an ordinal scale of sleep disruption: ‘‘NO ACTIVITY’’, ‘‘ANXIOUS’’, ‘‘IN-TERRUPTION’’, and ‘‘WANDER’’, representing progressively increasing disturbance to sleep. Activities unrelated to sleep behaviour, such as eating or hand gestures, were omitted from this label set.

‘‘NO ACTIVITY’’ denotes the absence of relevant movement in the monitored bedroom. ‘‘ANXIOUS’’ captures restless in-bed motion, such as tossing or turning. ‘‘INTERRUP-TION’’ represents larger posture changes, including sitting on the bed edge, standing up, or lying down again. ‘‘WAN-DER’’ denotes out-of-bed walking and represents the highest level of sleep disruption. This label set therefore focuses on sleep-relevant behavioural states rather than detailed activity categories.

## V. METHODOLOGY

## A. PREPROCESSING

This section describes how the measurement matrices $X ^ { ( j ) }$ of Equation (1) are transformed into the image-like representations fed to the CNN. Selecting the preprocessing parameters is a trade-off: they should be as similar as possible across technologies so that the comparison reflects the sensing modality rather than the processing, while still not disadvantaging any modality through choices tailored to another. The parameters below aim to balance this trade-off. In particular, the coherent observation interval $T _ { \mathrm { o b s } }$ over which each Doppler transform is computed, one 96-chirp frame for FMCW, the 2.5 s window for IR-UWB, and the 0.4 s short-time Fourier transform (STFT) window for Wi-Fi, determines the effective Doppler resolution $\Delta \nu = \lambda / ( 2 T _ { \mathrm { o b s } } )$ of each modality, as reported in Table 2.

## 1) FMCW

The radar acquires the measurement matrix $x ^ { ( \mathrm { F M C W } ) } [ m , k ]$ of Equation (8) in frames of $N _ { c }$ chirps and, in addition, across $N _ { r }$ virtual antennas<sup>2</sup>. Each ADC frame is therefore represented by $N _ { r }$ antenna-specific matrices $\mathbf { F } _ { r } ~ \in ~ \mathbb { C } ^ { N _ { c } \times N _ { s } }$ , where $\mathbf { F } _ { r }$ denotes the measurements observed by virtual antenna r.

A two-dimensional fast Fourier transform (FFT) is applied to each antenna slice: a first transform along the fast-time axis $( N _ { s }$ samples, index k) resolves range, and a second along the slow-time axis $( N _ { c }$ chirps, index m) resolves Doppler, yielding one range Doppler map (RDM) per antenna,

$$
\mathrm { R D M } _ { r } [ p , q ] = \left| \mathcal { F } _ { N _ { c } } \{ \mathcal { F } _ { N _ { s } } \{ \mathbf { F } _ { r } \} \} \right| , \qquad r = 1 , \ldots , N _ { r } ,\tag{12}
$$

with range bin $p$ and Doppler bin $q .$ Since static reflections are time-invariant along slow-time, they concentrate in the zero-Doppler bin, separating the static component $s ^ { ( \mathrm { F M C W } ) } [ k ]$ from the motion-induced Doppler content. To suppress this static component prior to Doppler processing, DC removal is applied to the range profile $\mathcal { F } _ { N _ { s } } \{ \mathbf { F } _ { r } \} [ k , m ]$ after the first FFT, by subtracting its mean across chirps for each range bin k,

$$
\widehat { \mathcal { F } } _ { N _ { s } } \{ \mathbf { F } _ { r } \} [ k , m ] = \mathcal { F } _ { N _ { s } } \{ \mathbf { F } _ { r } \} [ k , m ] - \frac { 1 } { N _ { c } } \sum _ { m = 1 } ^ { N _ { c } } \mathcal { F } _ { N _ { s } } \{ \mathbf { F } _ { r } \} [ k , m ] ,\tag{13}
$$

before the second FFT is applied to obtain the cluttersuppressed RDM in (12).

The twelve TX-RX antenna pairs are summed noncoherently to improve the signal-to-noise ratio (SNR). Taking the maximum over range for each Doppler bin then produces a one-dimensional Doppler profile without explicit position information.

Applying this pipeline per frame and stacking the successive Doppler profiles along the time axis yields a twodimensional Time-Doppler map $D [ t , q ]$ capturing the temporal evolution of radial velocity. To build the model input, a strided windowing operation is applied to $D$ with a window length of 2.5 s and a per-sample stride, i.e., the 50 ms frame interval, maximising data diversity. These steps are illustrated in Fig. 1.

## 2) IR-UWB

As described in Section III-A2, the IR-UWB receiver outputs a raw IQ CIR at every slow-time index m, forming the measurement matrix $x ^ { \mathrm { ( I R - U W B ) } } [ m , k ]$ of Equation (10).

Because the transmitter and receiver run on separate boards without a shared clock (Section IV-B), the CIR phase is corrupted by a time-varying drift. Following the phase correction approach from De Moerloose et al. [18], the phase of the first path (FP) sample at fast-time index $k _ { \mathrm { f p } }$ is taken as the per-CIR phase offset and subtracted from every fast-time sample,

$$
\tilde { x } [ m , k ] = x ^ { ( \mathrm { I R \mathrm { - } U W B } ) } [ m , k ] ~ e ^ { - j \angle x ^ { ( \mathrm { I R \mathrm { - } U W B } ) } [ m , k _ { \mathrm { f p } } ] } ,\tag{14}
$$

making the phase coherent across slow-time while preserving the activity-induced phase variations.

In the received CIR signals, the useful activity information resides in the multipath components arriving beyond the FP index. The FP corresponds to the direct line-of-sight (LOS) path between transmitter and receiver and therefore contains little information about the surrounding environment. The first 12 fast-time samples are therefore discarded, as they include both pre-FP samples and the FP region itself. Moreover, these early delay bins are often affected by receiver saturation and strong static components such as ceiling reflections. The remaining samples are retained as x˜[m, k] for $k > 1 2$

A strided windowing operation is then applied along the slow-time axis using windows of 2.5 s and a stride of 0.05 s. A per-sample stride is avoided because the IR-UWB slowtime rate sr is substantially higher than that of FMCW, which would greatly inflate the dataset without adding meaningful diversity. The stride is instead matched to the FMCW frame interval for comparability.

For each window W, a Doppler spectrum is computed independently for every fast-time bin using an FFT along the slow-time axis,

$$
\mathrm { R D M } _ { W } [ k , q ] = \left| \mathcal { F } _ { m \in W } \{ \tilde { x } [ m , k ] \} \right| ,\tag{15}
$$

producing a Range-Doppler map with range bin k and Doppler bin $q .$ The Doppler axis is cropped to [−25, 25] Hz, identified through a per-modality grid search as containing the activity-induced motion, while frequencies outside this range were found to carry little useful information. Unlike FMCW, the range dimension is retained because it still provides discriminative spatial information. The weaker contrast of the single-channel IR-UWB returns makes selecting one range bin less reliable, and retaining all bins performed better than collapsing the range axis.

The overlapping Range-Doppler maps form a temporal sequence that can be viewed as a three-dimensional tensor with time, range, and Doppler dimensions. To suppress residual static clutter, a background template is estimated independently for every range-Doppler cell by taking the running median of the corresponding cell over a temporal neighbourhood of $T _ { \mathrm { b g } } = 1 0 ~ \mathrm { s }$

$$
\begin{array} { r l } & { \widetilde { \mathrm { R D M } } _ { W } [ k , q ] = \mathrm { R D M } _ { W } [ k , q ] } \\ & { \qquad - \operatorname * { m e d i a n } _ { W ^ { \prime } \in \mathcal { N } ( W ) } \mathrm { R D M } _ { W ^ { \prime } } [ k , q ] . } \end{array}\tag{16}
$$

where $\mathcal { N } ( W )$ denotes the temporal neighbourhood centred on W spanning $T _ { \mathrm { b g } }$ . The median is therefore evaluated independently for every $( k , q )$ location through time, producing a slowly varying clutter estimate that is subtracted from the current Range-Doppler map while preserving transient motion. Because the temporal neighbourhood is centred on the current window, the method is non-causal and introduces up to $T _ { \mathrm { b g } } / 2 = 5$ s of latency in a real-time deployment.

## 3) Wi-Fi

For Wi-Fi, the input is the CSI measurement matrix $x ^ { \mathrm { ( W i - F i ) } } [ m , k ]$ of Equation (11), with slow-time (packet) index m and subcarrier index k.

As Wi-Fi is connection-based, packets are transmitted only when permitted by the medium access control (MAC) layer, resulting in irregular slow-time sampling. Since the resulting timing jitter strongly affects the sensitive phase, the complex CSI samples are first interpolated onto a uniform slow-time grid at the mean packet rate of 500 Hz, yielding the uniformly sampled matrix $x _ { \mathrm { u } } [ m , k ]$

Each of the $N _ { c } = 5 3$ subcarriers is subsequently mapped to the Doppler domain using an STFT with a window length of 200 samples (0.4 s) and an overlap of 175 samples (0.35 s). The resulting per-subcarrier Time-Doppler maps $S _ { k } [ t , q ]$ are averaged to form a single two-dimensional representation,

$$
S [ t , q ] = \frac { 1 } { N _ { c } } \sum _ { k = 1 } ^ { N _ { c } } \left| S _ { k } [ t , q ] \right| ,\tag{17}
$$

where t denotes the STFT time index and q the Doppler bin. The STFT yields a Doppler spectrum up to 250 Hz, which is cropped to [−80, 80] Hz, identified through a per-modality grid search as containing the activity-induced motion.

The resulting Time-Doppler maps form a temporal sequence. To suppress residual static clutter, a running median is evaluated independently for every Doppler bin over a temporal neighbourhood of $T _ { \mathrm { b g } } = 1 0$ s and subtracted from the current Time-Doppler map,

$$
\widetilde { S } [ t , q ] = S [ t , q ] - \mathrm { m e d i a n } _ { t ^ { \prime } \in \mathcal { N } ( t ) } S [ t ^ { \prime } , q ] ,\tag{18}
$$

where $\mathcal { N } ( t )$ denotes the temporal neighbourhood centred on t spanning $T _ { \mathrm { b g } } .$ . As for IR-UWB, the running median is evaluated independently for every Doppler bin through time, suppressing slowly varying clutter while preserving transient motion. The same non-causality and corresponding latency therefore apply.

The clutter-suppressed Time-Doppler map is finally segmented into samples using a strided windowing operation with windows of 2.5 s and a stride of 0.05 s. As for IR-UWB, a per-sample stride is avoided because it would unnecessarily inflate the dataset. Instead, the stride is matched to the FMCW frame interval for comparability across modalities.

## B. CNN BASED CLASSIFICATION

A uniform CNN architecture is applied to all three modalities, so that classification differences reflect the sensing technology rather than the classifier. Each modality produces a 2D input map of shape $( T \times F )$ : for FMCW and Wi-Fi the axes correspond to time and Doppler, for IR-UWB they correspond to range and Doppler.

The network consists of five convolutional blocks followed by two fully connected layers, as illustrated in Table 3. Each convolutional block applies a convolution with same-padding $( 3 \times 3$ in the first three blocks, 2×2 in the last two), followed by exponential linear unit (ELU) activation and batch normalisation. The number of filters increases as 8, 16, 32, 64, 64. The first block applies $1 \times 2$ max-pooling, down-sampling only the feature axis $F .$ . The remaining four blocks apply $2 \times 2$ pooling, down-sampling both axes. After flattening, two dense layers of size 32 precede a softmax output layer of size $N _ { c }$

The architecture and training hyperparameters were selected by grid search on the training data using the same leave-one-person-out (LOPO) cross-validation strategy as in the final evaluation, while keeping the test fold unseen. The selected architecture was fixed across modalities to ensure a fair comparison. Regularisation is applied through $L _ { 2 }$ weight decay $( \lambda = 1 0 ^ { - 2 } )$ on all learnable layers, batch normalisation in the convolutional blocks, and dropout only in the classifier head. The latter is motivated by the shared-weight structure and batch normalisation of the convolutional stack, while the dense layers contain most classifier-specific parameters. Batch size, learning rate, and learning-rate schedule were tuned per modality, and class weighting is applied to compensate for class imbalance.

TABLE 3. Uniform CNN architecture applied to all three modalities. The input is a single-channel 2D map of shape $( \pmb { \tau } \times \pmb { F } ) ,$ where the two axes correspond to time and Doppler for FMCW and Wi-Fi, and to range and Doppler for IR-UWB. $\pmb { N _ { c } }$ denotes the number of output classes.
<table><tr><td rowspan=1 colspan=1>Stage</td><td rowspan=1 colspan=1>Configuration</td></tr><tr><td rowspan=1 colspan=1>Input</td><td rowspan=1 colspan=1> $\overline { { ( T \times F \times 1 ) } }$ </td></tr><tr><td rowspan=1 colspan=1>Conv Block 1</td><td rowspan=1 colspan=1>Conv2D(8, 3 ×3) + ELU + BNMaxPool(1×2)</td></tr><tr><td rowspan=1 colspan=1>Conv Block 2</td><td rowspan=1 colspan=1>Conv2D(16, 3×3) + ELU + BNMaxPool(2×2)</td></tr><tr><td rowspan=1 colspan=1>Conv Block 3</td><td rowspan=1 colspan=1>Conv2D(32, 3×3) + ELU + BNMaxPool(2×2)</td></tr><tr><td rowspan=1 colspan=1>Conv Block 4</td><td rowspan=1 colspan=1>Conv2D(64, 2×2) + ELU + BNMaxPool(2×2)</td></tr><tr><td rowspan=1 colspan=1>Conv Block 5</td><td rowspan=1 colspan=1>Conv2D(64, 2×2) + ELU + BNMaxPool(2×2)</td></tr><tr><td rowspan=1 colspan=1>Classifier</td><td rowspan=1 colspan=1>FlattenDense(32) + ELU + Dropout(0.25)Dense(32) + ELUDense(Nc) + Softmax</td></tr></table>

## C. GENERALIZATION EVALUATION STRATEGIES

Three leave-x-out cross-validation strategies are applied to both the fine-grained HAR and coarse-grained sleepmonitoring label sets. All results are reported as macro F1 score, which accounts for precision, recall, and class imbalance. Each reported value is the mean over folds, with the standard deviation indicating fold-to-fold variability.

The following cross-validation strategies are considered:

• LOPO: one participant is left out per fold as validation set, yielding 20 folds. This measures generalisation to unseen people within a seen environment.

• leave-one-scenario-out (LOSO): This evaluates generalisation to unseen room layouts. However, as illustrated in Fig. 2, the protocol remains partly optimistic because leaving out one scenario still exposes either the corresponding bed position or chair position through other scenarios. Only their combination remains unseen.

• leave-one-bed-position-out (LOBPO): scenarios are left out in pairs, namely 1-2, 3-4, and 5-6. Each fold therefore withholds one complete bed configuration, making this the strictest evaluation of cross-layout generalisation.

Each strategy reports macro F1 both before and after voting, a temporal smoothing step in which all windowlevel predictions belonging to the same ground-truth activity are replaced by their majority-voted label. The voted score therefore reflects event-level recognition performance while reducing the influence of isolated window-level misclassifications.

## VI. RESULTS AND ANALYSIS

This section analyzes the performance of the three sensing modalities under the considered evaluation protocols and provides a signal-level interpretation of the observed differences by linking them to their underlying sensing principles, signal representations, and preprocessing choices. Table 4 summarises the macro F1 scores for both label sets, the three cross-validation tiers (LOPO, LOSO and LOBPO), and as explained in Section V-C, the values before and after voting. The results separate along two distinct axes. In absolute accuracy within a known or similar layout, IR-UWB leads: it attains the best fine-grained LOPO score (89.0% against 83.4% for FMCW) and matches FMCW under LOSO. In robustness to an unseen room layout, FMCW stands alone: it is the only modality whose fine-grained accuracy is essentially unchanged under LOBPO, while IR-UWB and Wi-Fi each lose about 10 percentage points. Wi-Fi is consistently the weakest on the fine-grained task. At the coarse granularity used for sleep monitoring, all three modalities exceed 92% under every protocol, including LOBPO.

To explain these differences, the following section will first relate these outcomes to the sensing and preprocessing properties of the three modalities (Section VI-A). The fine-grained results are then analysed across people, post-processing and room layouts (Section VI-B), followed by the coarse sleepmonitoring results (Section VI-C).

## A. INTERPRETATION FRAMEWORK

The observed performance differences are driven by a fundamental trade-off between activity discriminability and layout robustness. Technologies that preserve more spatial and motion detail provide the classifier with richer information to distinguish similar activities, but also increase the risk of learning environment-specific characteristics that do not generalize to unseen room layouts. Conversely, representations that discard explicit spatial information may lose some discriminative power, yet tend to be more robust to environmental changes.

Four properties from Table 2 are central to this tradeoff. First, finer range resolution separates reflections from different body regions and from surrounding objects before the range information is combined. Second, non-coherent integration across multiple TX-RX antenna pairs increases the motion energy relative to uncorrelated noise and improves the effective SNR. Third, finer Doppler resolution separates more closely spaced radial-velocity components, providing a more detailed description of the motion. Finally, retaining range in the classifier input provides explicit spatial information, whereas collapsing it reduces the model’s direct access to the subject’s position.

FMCW combines 0.126 m range resolution with noncoherent integration over twelve TX-RX antenna pairs. The fine range bins separate moving body returns from much of the surrounding clutter, while summing the pair-specific Range-Doppler magnitudes increases their contrast. After static suppression, the range maximum can therefore retain the strongest moving return in each Doppler bin and discard its range coordinate. Stacking these profiles preserves the combined micro-Doppler pattern while removing most explicit position information. This should favour transfer across layouts, although it discards spatial cues that could help distinguish location-bound transitions.

IR-UWB has coarser 0.30 m range bins and only one receive channel. Its moving returns are therefore less cleanly isolated in individual range bins, making a range-maximum collapse more likely to select clutter or only part of the distributed activity signature. Retaining the full Range-Doppler map avoids this early selection and empirically outperformed the collapsed representation. Together with its finer Doppler resolution (0.009 versus 0.093 m/s for FMCW), this provides the classifier with useful spatial structure and finer velocity detail. These cues should improve recognition of posture transitions in known layouts, but can also tie those distinctions to the ranges observed during training.

Wi-Fi provides stable CSI phase because its transmitter and receiver share a clock, so human motion is clearly visible without the phase correction required by IR-UWB. The difficulty is separating the sources of that motion. Its nominal 7.5 m range resolution exceeds the room dimensions, so reflections from the torso, arms, legs and different propagation paths are mixed in the channel before processing. Human motion affects the 53 subcarriers differently, providing some diversity across frequency. However, because they span only 20 MHz, this diversity is too limited to compensate for the missing range separation, which makes similar activities difficult to distinguish. Moreover, CSI contains the combined propagation paths of the room. The network can therefore learn layout-specific channel patterns together with the activity, causing it to overfit partly to the room layout even though the representation contains no explicit range axis.

TABLE 4. Macro F1 scores (%, mean±std over folds) for fine-grained activity recognition and coarse-grained sleep monitoring, before and after label voting, under LOPO, LOSO, and LOBPO cross-validation. IR-UWB performs best on the easier cross-subject task, while FMCW is most robust to unseen room layouts. All modalities achieve above 92% macro F1 on the coarse sleep-monitoring task after voting.
<table><tr><td rowspan="3">Exp</td><td colspan="4">FMCW</td><td colspan="4">IR-UWB</td><td colspan="4">Wi-Fi</td></tr><tr><td colspan="2">Fine</td><td colspan="2">Coarse</td><td colspan="2">Fine</td><td colspan="2">Coarse</td><td colspan="2">Fine</td><td colspan="2">Coarse</td></tr><tr><td>Before</td><td>After</td><td>Before</td><td>After</td><td>Before</td><td>After</td><td>Before</td><td>After</td><td>Before</td><td>After</td><td>Before</td><td>After</td></tr><tr><td></td><td>Voting</td><td>Voting</td><td>Voting</td><td>Voting</td><td>Voting</td><td>Voting</td><td>Voting</td><td>Voting</td><td>Voting</td><td>Voting</td><td>Voting</td><td>Voting</td></tr><tr><td>LOPO</td><td>75.0±9</td><td>83.4±9</td><td>88.5±8</td><td>95.1±3</td><td>78.6±5</td><td>89.0±6</td><td>95.4±3</td><td>98.2±3</td><td>65.0±4</td><td>79.0±7</td><td>90.2±3</td><td>96.1±5</td></tr><tr><td>LOSO</td><td>77.3±4</td><td>86.6±1</td><td>89.9±4</td><td>94.9±2</td><td>76.3±5</td><td>86.4±5</td><td>95.3±2</td><td>97.7±3</td><td>63.2±3</td><td>75.3±3</td><td>89.4±2</td><td>95.1±2</td></tr><tr><td>LOBPO</td><td>75.2±2</td><td>83.8±2</td><td>88.7±4</td><td>93.4±2</td><td>68.3±4</td><td>78.5±4</td><td>90.4±5</td><td>94.2±6</td><td>57.2±0</td><td>68.8±0</td><td>85.5±3</td><td>92.6±2</td></tr></table>

These properties provide the framework for interpreting the aggregate and per-class results below. Because the hardware and representations were not varied independently, their individual causal contributions cannot be quantified; agreement between the sensing properties, fold-level behavior and class-specific confusions nevertheless supports the resulting modality trade-offs.

## B. FINE-GRAINED ACTIVITY RECOGNITION

1) Generalization to unseen people and effect of voting Per-participant fine-grained LOPO scores are listed in Table 5. IR-UWB achieves the highest mean macro F1 both before and after voting (78.6% and 89.0%), followed by FMCW (75.0% and 83.4%) and Wi-Fi (65.0% and 79.0%). Before voting, IR-UWB also varies less across participants (standard deviation 5 pp) than FMCW (9 pp). Within this sample, none of the modalities shows a systematic dependence on height, weight, age or gender. This suggests that recognition is not tied to a single body profile and generalises across the participant diversity represented in the dataset, although the sample size and demographic imbalance do not allow broader conclusions about demographic robustness.

Voting improves all three modalities: Wi-Fi gains 14.0 pp, IR-UWB 10.4 pp and FMCW 8.4 pp. The larger gains for Wi-Fi and IR-UWB show that many window-level errors are not sustained over an entire event. After voting, however, Wi-Fi remains 4.4 pp below FMCW and 10.0 pp below IR-UWB. Because voting groups predictions using the groundtruth activity boundaries, these values quantify event-level recognition under known segmentation. A deployment without such boundaries may obtain smaller gains.

## 2) Generalization to unseen room layouts

Under LOSO, performance after voting stays close to the LOPO baseline (Table 4): FMCW improves to 86.6% and IR-UWB matches it at 86.4%, while Wi-Fi falls slightly to 75.3%. This is expected, because each scenario moves only one piece of furniture, so every bed and chair position still appears in training and only their combination is withheld (Fig. 2). In a care-home setting, where layouts are fairly uniform, this is the realistic case, and the two radars are equivalent under it.

Although FMCW and IR-UWB have almost identical mean LOSO scores (86.6% and 86.4%), their fold-to-fold stability differs. Across the six folds, the difference between the highest and lowest score is only 2.3 pp for FMCW, with a standard deviation of 0.8 pp. For IR-UWB, this range is 12.8 pp and the standard deviation is 5.4 pp. IR-UWB reaches 91–93% when scenario 1, 2 or 5 is withheld, but 80–82% for scenarios 3, 4 and 6, where the bed is farther off-centre or the chair orientation is less favourable. FMCW therefore remains nearly unchanged regardless of which layout is unseen, whereas IR-UWB does not, even though their mean scores are similar.

TABLE 5. Per-person leave-one-person-out (LOPO) macro F1 (%) for fine-grained activity recognition before and after label voting, across all three sensing technologies. Demographics: height (cm), weight (kg), age, gender.
<table><tr><td colspan="6"></td><td colspan="2">FMCW</td><td colspan="2">IR-UWB</td><td colspan="2">Wi-Fi</td></tr><tr><td></td><td></td><td>Weight</td><td>Age</td><td>Sex</td><td>Before voting</td><td>After voting</td><td>Before voting</td><td>After voting</td><td>Before voting</td><td>After voting</td></tr><tr><td>ID 1</td><td>Height 179</td><td>75</td><td>30</td><td>M</td><td>88.5</td><td>90.7</td><td>82.3</td><td>89.5</td><td>69.1</td><td>91.6</td></tr><tr><td>2</td><td>189</td><td>80</td><td>29</td><td>M</td><td>79.4</td><td>82.6</td><td>76.8</td><td>85.1</td><td>66.7</td><td>77.6</td></tr><tr><td>3</td><td>173</td><td>82</td><td>60</td><td>M</td><td>84.7</td><td>92.3</td><td>77.9</td><td>86.3</td><td>63.2</td><td>68.7</td></tr><tr><td>4</td><td>159</td><td>50</td><td>40</td><td>F</td><td>71.2</td><td>82.1</td><td>72.7</td><td>87.8</td><td>52.6</td><td>72.1</td></tr><tr><td>5</td><td>182</td><td>81</td><td>52</td><td>M</td><td>68.7</td><td>75.5</td><td>77.7</td><td>84.7</td><td>63.1</td><td>73.5</td></tr><tr><td>6</td><td>172</td><td>71</td><td>27</td><td>M</td><td>67.8</td><td>73.3</td><td>67.4</td><td>76.1</td><td>60.8</td><td>70.1</td></tr><tr><td>7</td><td>178</td><td>92</td><td>26</td><td>M</td><td>56.9</td><td>60.9</td><td>67.8</td><td>80.4</td><td>64.4</td><td>75.9</td></tr><tr><td>8</td><td>182</td><td>73</td><td>22</td><td>M</td><td>67.4</td><td>88.2</td><td>81.4</td><td>94.5</td><td>69.4</td><td>84.0</td></tr><tr><td>9</td><td>166</td><td>58</td><td>26</td><td>F</td><td>56.2</td><td>78.4</td><td>83.6</td><td>94.7</td><td>61.5</td><td>78.8</td></tr><tr><td>10</td><td>178</td><td>75</td><td>27</td><td>M</td><td>78.4</td><td>85.4</td><td>82.5</td><td>98.3</td><td>69.4</td><td>85.7</td></tr><tr><td>11</td><td>187</td><td>72</td><td>21</td><td>M</td><td>81.9</td><td>78.8</td><td>77.8</td><td>83.1</td><td>68.7</td><td>76.6</td></tr><tr><td>12</td><td>178</td><td>78</td><td>67</td><td>M</td><td>68.6</td><td>71.7</td><td>81.3</td><td>88.9</td><td>59.9</td><td>69.2</td></tr><tr><td>13</td><td>168</td><td>66</td><td>27</td><td>F</td><td>70.9</td><td>84.0</td><td>81.9</td><td>91.9</td><td>67.6</td><td>77.5</td></tr><tr><td>14</td><td>165</td><td>64</td><td>60</td><td>F</td><td>79.4</td><td>92.0</td><td>78.2</td><td>84.1</td><td>65.0</td><td>77.0</td></tr><tr><td>15</td><td>172</td><td>62</td><td>27</td><td>M</td><td>70.3</td><td>72.6</td><td>79.0</td><td>88.0</td><td>60.0</td><td>73.5</td></tr><tr><td>16</td><td>178</td><td>87</td><td>41</td><td>M</td><td>88.4</td><td>96.6</td><td>83.0</td><td>96.6</td><td>65.8</td><td>82.2</td></tr><tr><td>17</td><td>174</td><td>65</td><td>27</td><td>M</td><td>88.2</td><td>98.8</td><td>85.8</td><td>97.6</td><td>70.6</td><td>92.7</td></tr><tr><td>18</td><td>178</td><td>73</td><td>42</td><td>M</td><td>76.8</td><td>86.2</td><td>82.2</td><td>94.9</td><td>69.4</td><td>84.2</td></tr><tr><td>19</td><td>172</td><td>88</td><td>25</td><td>F</td><td>73.1</td><td>83.8</td><td>73.9</td><td>83.3</td><td>64.9</td><td>82.9</td></tr><tr><td>20</td><td>174</td><td>86</td><td>26</td><td>F</td><td>82.6</td><td>93.2</td><td>78.3</td><td>94.7</td><td>67.1</td><td>86.4</td></tr><tr><td>Avg</td><td></td><td></td><td></td><td></td><td>75.0</td><td>83.4</td><td>78.6</td><td>89.0</td><td>65.0</td><td>79.0</td></tr></table>

The stricter LOBPO protocol separates the modalities more clearly. Relative to LOPO, FMCW changes from 83.4% to 83.8%, whereas IR-UWB and Wi-Fi decrease by 10.5 and 10.2 pp, respectively. The IR-UWB result also depends on which bed position is withheld: 73.3% for scenarios 3-4, compared with 81.7% for scenarios 1-2 and 80.4% for scenarios 5-6. Wi-Fi remains near 69% in all three folds. The layoutspecific IR-UWB loss indicates sensitivity to the retained range information, whereas Wi-Fi’s uniform loss reflects its inability to resolve bed position at room scale.

## 3) Per-class confusion analysis

The confusion matrices in Fig. 3 are row-normalised and computed after voting, so each diagonal entry is per-class recall. The clearest results occur for whole-body activity: ‘‘NO

ACTIVITY’’ reaches at least 98% for every modality and protocol, and FMCW recognises ‘‘WALK’’ perfectly. Under LOBPO, Wi-Fi still recalls 99% of ‘‘WALK’’ but maps 9% of ‘‘ANXIOUS’’ to that class, showing that movement in bed and walking can produce similar signatures when the system cannot determine where the motion occurs. IR-UWB instead loses ‘‘WALK’’ recall, from 95% to 89%, with errors assigned mainly to ‘‘LIE ON BED’’ and ‘‘GET UP BED’’. This shift towards bed-related classes supports the conclusion that its retained range axis ties activity recognition more closely to the positions observed during training.

Posture transitions account for much of the difference between IR-UWB and FMCW under LOPO. IR-UWB exceeds FMCW by 13 pp on ‘‘STAND UP’’, 6 pp on ‘‘GET UP BED’’ and 4 pp on ‘‘SIT DOWN’’. This concentration of the advantage in related transitions is consistent with the combination of finer Doppler detail and retained range helping to distinguish their torso-motion trajectories. The present comparison cannot determine the contribution of either property separately. Under LOBPO, these classes degrade most strongly for IR-UWB and Wi-Fi: IR-UWB ‘‘LIE ON BED’’ falls from 89% to 57%, with 23% assigned to ‘‘GET UP BED’’, while Wi-Fi ‘‘GET UP BED’’ falls from 78% to 55% and is split mainly between ‘‘SIT DOWN’’ (17%) and ‘‘LIE ON BED’’ (16%). The concentration of errors among related transitions suggests that their distinctions transfer poorly when the bed position is unseen.

The hand-movement classes are also difficult, especially for Wi-Fi. Its ‘‘CLAP HANDS’’ recall decreases from 47% to 21% under LOBPO, with most errors assigned to ‘‘EATING’’ and ‘‘WAVE HANDS’’. Its unresolved channel response therefore preserves evidence of hand motion but not enough detail to separate similar gestures reliably. FMCW is more stable on these classes and maintains 98% recall for ‘‘EAT-ING’’, although ‘‘WAVE HANDS’’ and ‘‘CLAP HANDS’’ remain mutually confused. This advantage is consistent with its finer range resolution separating body-region returns and the non-coherent sum over twelve TX-RX antenna pairs improving the SNR before the range axis is collapsed.

Finally, IR-UWB more often maps activity to ‘‘NO AC-TIVITY’’ than the other modalities. The largest example is 13% of ‘‘EATING’’ under LOPO. Its lower per-bin SNR, together with residual phase instability from the unsynchronised transmitter and receiver, likely raises the motion evidence required for detection. For FMCW, the main crosslayout change is instead ‘‘LIE ON BED’’ being assigned to ‘‘ANXIOUS’’ more often (7% to 14%). Once range is collapsed, both classes are represented mainly by low-velocity in-bed motion, explaining why they remain the principal FMCW confusion.

## C. COARSE-GRAINED SLEEP MONITORING

The coarse label set merges posture transitions into ‘‘INTER-RUPTION’’ and omits activities that do not fit in the sleepdisruption context (Section IV-C). After voting, LOPO macro

F1 reaches 98.2% for IR-UWB, 96.1% for Wi-Fi and 95.1% for FMCW.

Cross-layout differences are also smaller than for the finegrained task. Under LOSO, the scores are 97.7%, 95.1% and 94.9% for IR-UWB, Wi-Fi and FMCW, respectively. Under LOBPO, all three remain within 1.6 pp: IR-UWB reaches 94.2%, FMCW 93.4% and Wi-Fi 92.6%. The merged labels remove the difficult hand movements and most distinctions among posture transitions, so these values describe a less demanding classification problem rather than improved sensing fidelity.

The remaining class-level errors are modality-specific. Under LOBPO, FMCW assigns 10% of ‘‘ANXIOUS’’ to ‘‘IN-TERRUPTION’’, Wi-Fi assigns 5% to ‘‘WANDER’’, and IR-UWB ‘‘WANDER’’ recall falls from 97% to 85%, with errors divided between ‘‘ANXIOUS’’ and ‘‘INTERRUPTION’’. IR-UWB is also the only modality to assign non-zero proportions of all three activity classes to ‘‘NO ACTIVITY’’ (1–2%). Although small, the latter errors are operationally important because they represent missed sleep disruptions rather than confusion between disruption levels.

## VII. DEPLOYMENT TRADE-OFFS

The preceding sections showed how the three sensing modalities differ in recognition accuracy and robustness. For real-world single-anchor deployment, however, the preferred modality also depends on constraints beyond HAR performance. This section translates the ML results into deployment-oriented trade-offs by considering localisation, power, cost, and communication capabilities.

## A. LOCALIZATION PERFORMANCE

Localization complements HAR in real-world deployments: mapping activity hotspots provides spatial context and can improve recognition accuracy. The three modalities differ substantially in this capability. FMCW resolves range at 0.126 m (Table 2) and angle via its twelve virtual channels, enabling accurate 3D localization within the room [20]. The IR-UWB kits support ranging and angle of arrival (AoA) estimation, but their accuracy is bounded by the coarser 0.30 m range resolution. Moreover, ranging on humans with IR-UWB is challenging due to the body’s low radar reflectivity [21], a limitation amplified in this work by the reduced radar cross-section of the ceiling-mounted geometry. Wi-Fi can in principle estimate range by transforming CSI into CIRs, but its 20 MHz bandwidth corresponds to a single 7.5 m range bin that exceeds the room dimensions. Although phase tracking or wider channels could improve ranging for both IR-UWB and Wi-Fi [18], FMCW is currently the only modality of the three that provides robust 3D localization without additional infrastructure or processing.

## B. POWER CONSUMPTION

The three modalities differ by more than an order of magnitude in power draw. The IR-UWB front-end is by far the least power intensive: the COTS DW3000-based transmitterreceiver system consumes on the order of 0.2 W in continuous operation [22], and can be driven substantially lower through device-level duty-cycling. The single-chip FMCW sensor is significantly more demanding: the IWR6843 typically draws 1.2-1.75 W [23], scaling with the number of active transmit channels and the frame duty cycle, with the upper part of this range applicable to the three-transmitter configuration used in this work. Unlike IR-UWB and FMCW, monostatic Wi-Fi sensing cannot currently be realized with standard COTS Wi-Fi chipsets, as they do not expose the reflected-signal CSI in the presence of simultaneous transmission. Consequently, specialized SDR-based platforms are typically required for single-device Wi-Fi radar. As a result, the Wi-Fi CSI platform consumes the most, at 3.5-4.0 W (measured at 5.2 V, 0.68- 0.76 A in our setup). This figure reflects the SDR platform (ZedBoard with AD-FMCOMMS2) required for CSI access, which is not power-optimised and is not representative of a commodity Wi-Fi radio.

![](images/140b0ba30874961afee341bb06dd6d629a2676641f0db749250b08799f413c6a.jpg)  
FIGURE 3. Confusion matrices comparing fine-grained activity classification performance across Wi-Fi, IR-UWB, and FMCW technologies in LOPO and LOBPO cross-validation. Rows indicate true activity labels, columns indicate predicted labels, and values are row-normalised per true class (recall, %).

## C. COST-EFFECTIVENESS

Hardware prices separate the modalities by an order of magnitude.<sup>3</sup> The IR-UWB radar is the cheapest: at roughly EUR 7 per DW3000 transceiver, even the two-node pseudomonostatic setup totals approximately EUR 14. The singlechip FMCW radar is in the same class at roughly EUR 20. Wi-Fi is far more expensive: to the best of our knowledge, no commercial Wi-Fi chipset currently supports the fullduplex monostatic radar mode used here, so a specialised SDR platform is required, the cheapest we identified costing approximately EUR 320. This price is development-grade and would likely fall in a dedicated deployment, but at present the need for specialised hardware makes Wi-Fi the least costeffective option for monostatic deployments.

## D. COMMUNICATION CAPABILITY

Finally, the modalities differ in how naturally sensing integrates with communication. Wi-Fi is a communication standard first, so sensing reuses an existing data link at no additional spectral cost. IR-UWB is IEEE 802.15.4z-compliant and likewise combines ranging and data communication on the same device. FMCW, by contrast, is a sensing-only modality with no native communication and would require a separate link for data transport. Wi-Fi and IR-UWB are therefore attractive where a single device must both sense and communicate.

## VIII. CONCLUSION

This paper compared three ceiling-mounted COTS RF technologies (FMCW, IR-UWB, Wi-Fi) for HAR and sleep monitoring using a uniform CNN on a 20-participant dataset.

In the LOPO evaluation, fine-grained recognition is led by IR-UWB (89.0%), followed by FMCW (83.4%) and Wi-Fi (79.0%). All three exceed 95% macro F1 for coarse sleep monitoring. Under the stricter LOBPO protocol, FMCW remains stable at 83.8%, whereas IR-UWB and Wi-Fi decrease to 78.5% and 68.8%, respectively. IR-UWB’s retained range and finer Doppler information favour detailed recognition in known layouts but increase position dependence, whereas FMCW’s collapsed-range representation trades some of that detail for stronger cross-layout robustness. Wi-Fi’s unresolved, multipath-dependent channel response limits both fine-grained accuracy and transfer to unseen layouts.

Deployment trade-offs extend beyond accuracy. IR-UWB is the most cost- and power-efficient option and natively supports unified sensing and communication. FMCW maximizes sensing robustness and cross-environment generalisation, but consumes more power and requires a separate data link. Despite the appeal of reusing existing infrastructure, Wi-Fi is currently the least practical for monostatic radar, as it requires an expensive, specialized software-defined-radio platform. Consequently, the preferred modality depends on the deployment priority: robustness favours FMCW, whereas cost, power efficiency, and integrated connectivity favour IR-UWB.

## DECLARATION ON THE USE OF GENERATIVE AI

During the preparation of this manuscript, the authors used generative artificial intelligence (AI) tools, primarily Microsoft Copilot, to assist with language editing, paraphrasing, rewording, grammar and spelling checks, and the refinement of selected sections of the text. Microsoft Copilot was also used to assist in the development of portions of the dataprocessing and machine-learning code used during the study. All AI-generated suggestions, text, and code were carefully reviewed, validated, and modified where necessary by the authors. The authors take full responsibility for the accuracy, integrity, and final content of the manuscript, including all analyses, results, and conclusions.

## REFERENCES

[1] World Health Organization, ‘‘Ageing and health,’’ https://www.who.int/ news-room/fact-sheets/detail/ageing-and-health, 2025, accessed: 2026- 05-11.

[2] J. B. Canever, G. Zurman, F. Vogel, D. V. Sutil, J. B. M. Diz, A. L. Danielewicz, B. D. S. Moreira, H. I. Cimarosti, and N. C. P. De Avelar, ‘‘Worldwide prevalence of sleep problems in communitydwelling older adults: A systematic review and meta-analysis,’’ Sleep Medicine, vol. 119, pp. 118–134, Jul. 2024. [Online]. Available: https://linkinghub.elsevier.com/retrieve/pii/S1389945724001448

[3] C. Wrede, A. Braakman-Jansen, and L. Van Gemert-Pijnen, ‘‘Understanding acceptance of contactless monitoring technology in home-based dementia care: a cross-sectional survey study among informal caregivers,’’ Frontiers in Digital Health, vol. 5, p. 1257009, Oct. 2023. [Online]. Available: https://www.frontiersin.org/articles/10.3389/ fdgth.2023.1257009/full

[4] W. Lu, S. Kumar, M. Sandhu, and Q. Zhang, ‘‘An Unobtrusive Fall Detection System Using Ceiling-mounted Ultra-wideband Radar,’’ in 2023 45th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). Sydney, Australia: IEEE, Jul. 2023, pp. 1–5. [Online]. Available: https://ieeexplore.ieee.org/document/ 10341081/

[5] W. Lu, C. Bird, M. Sandhu, and D. Silvera-Tawil, ‘‘Office Posture Detection Using Ceiling-Mounted Ultra-Wideband Radar and Attention Based Modality Fusion,’’ Sensors, vol. 25, no. 16, p. 5164, Aug. 2025. [Online]. Available: https://www.mdpi.com/1424-8220/25/16/5164

[6] W. Guo, S. Yamagishi, and L. Jing, ‘‘Human Activity Recognition via Wi-Fi and Inertial Sensors With Machine Learning,’’ IEEE Access, vol. 12, pp. 18 821–18 836, 2024. [Online]. Available: https://ieeexplore.ieee.org/ document/10418123/

[7] X. Chen, C. Li, C. Jiang, W. Meng, and W. Xiao, ‘‘WiPhase: A Human Activity Recognition Approach by Fusing of Reconstructed WiFi CSI Phase Features,’’ IEEE Transactions on Mobile Computing, vol. 24, no. 1, pp. 394–406, Jan. 2025. [Online]. Available: https: //ieeexplore.ieee.org/document/10681250/

[8] S. Z. Gurbuz, M. M. Rahman, E. Kurtoglu, T. Macks, and F. Fioranelli, ‘‘Cross-frequency training with adversarial learning for radar micro-Doppler signature classification (Rising Researcher),’’ in Radar Sensor Technology XXIV, A. M. Raynal and K. I. Ranney, Eds. Online Only, United States: SPIE, May 2020, p. 16. [Online]. Available: https://www. spiedigitallibrary.org/conference-proceedings-of-spie/11408/2559155/ Cross-frequency-training-with-adversarial-learning-for-radar-micro-Doppler/ 10.1117/12.2559155.full

[9] Y. Ding, P. Lv, R. Liu, Y. Peng, and M. Ding, ‘‘A Radar System-Agnostic (RSA) Learning Architecture for Human Activity Recognition,’’ IEEE Sensors Journal, vol. 25, no. 10, pp. 18 492–18 502, May 2025. [Online]. Available: https://ieeexplore.ieee.org/document/10948888/

[10] A. Ibrahim, M. Z. Khan, M. Imran, H. Larijani, Q. H. Abbasi, and M. Usman, ‘‘RadSpecFusion: Dynamic attention weighting for multi radar human activity recognition,’’ Internet of Things, vol. 33, p. 101682, Sep. 2025. [Online]. Available: https://linkinghub.elsevier.com/retrieve pii/S2542660525001969

[11] Z. Chen, C. Cai, T. Zheng, J. Luo, J. Xiong, and X. Wang, ‘‘RF-Based Human Activity Recognition Using Signal Adapted Convolutional Neural Network,’’ IEEE Transactions on Mobile Computing, vol. 22, no. 1, pp. 487–499, Jan. 2023. [Online]. Available: https://ieeexplore.ieee.org/ document/9408395/

[12] A. Dahal, S. Biswas, S. Z. Gurbuz, and A. C. Gurbuz, ‘‘Comparison Between Wi-Fi-CSI and Radar-Based HAR,’’ in 2024 IEEE Radar Conference (RadarConf24). Denver, CO, USA: IEEE, May 2024, pp. 1–6. [Online]. Available: https://ieeexplore.ieee.org/document/10548515/

[13] X. Jiao, T. Havinga, W. Liu, and I. Moerman, ‘‘Single-Input-Multiple-Output Wi-Fi Radar for Vital Signal Sensing and Device Tracking,’’ in 2025 IEEE 5th International Symposium on Joint Communications &; Sensing (JC&S). Oulu, Finland: IEEE, Jan. 2025, pp. 1–3. [Online]. Available: https://ieeexplore.ieee.org/document/10880631/

[14] A. T. Kristensen, A. Balatsoukas-Stimming, and A. Burg, ‘‘An SDR-Based Monostatic Wi-Fi System with Analog Self-Interference Cancellation for Sensing,’’ in 2025 IEEE International Symposium on Circuits and Systems (ISCAS). London, United Kingdom: IEEE, May 2025, pp. 1–5. [Online]. Available: https://ieeexplore.ieee.org/document/11043800/

[15] G. Bakhshi, M. A. Rawahi, N. Akhshatayeva, M. E. Hassan, M. Elsayed, Y. Elshenawy, S. D. Naicker, and H. Abou-Zeid, ‘‘RespiSense: Real-Time Respiration Monitoring Using a Low-Complexity WiFi SDR Platform,’ in GLOBECOM 2025 - 2025 IEEE Global Communications Conference. Taipei, Taiwan: IEEE, Dec. 2025, pp. 973–979. [Online]. Available: https://ieeexplore.ieee.org/document/11432470

[16] Y. Ma, G. Zhou, and S. Wang, ‘‘Wifi sensing with channel state information: A survey,’’ ACM Comput. Surv., vol. 52, no. 3, Jun. 2019. [Online]. Available: https://doi.org/10.1145/3310194

[17] iDlab. (2024) iDlab homelab. [Online]. Available: https://homelab.ilabt. imec.be/

[18] J. De Moerloose, A. Shahid, and E. De Poorter, ‘‘Towards mm-Level Accurate UWB Radar: High-Accuracy Phase-Based Obstacle Detection through Multi-Channel Fusion,’’ 2026, version Number: 1. [Online]. Available: https://arxiv.org/abs/2606.16657

[19] X. Jiao, W. Liu, M. Mehari, M. Aslam, and I. Moerman, ‘‘openwifi: a free and open-source ieee802.11 sdr implementation on soc,’’ in 2020 IEEE 91st Vehicular Technology Conference (VTC2020-Spring), 2020, pp. 1–2.

[20] P. Zhao, C. X. Lu, J. Wang, C. Chen, W. Wang, N. Trigoni, and A. Markham, ‘‘Human tracking and identification through a millimeter wave radar,’’ Ad Hoc Networks, vol. 116, p. 102475, 2021. [Online]. Available: https://www.sciencedirect.com/science/article/ pii/S1570870521000421

[21] B. V. Herbruggen, S. Luchie, R. Berkvens, J. Fontaine, and E. D. Poorter, ‘‘Impact of cir processing for uwb radar distance estimation with the

dw1000 transceiver,’’ in 2023 13th International Conference on Indoor Positioning and Indoor Navigation (IPIN), 2023, pp. 1–7.

[22] A. Lambrecht, S. Luchie, J. Fontaine, B. V. Herbruggen, A. Shahid, and E. De Poorter, ‘‘Low-cost embedded breathing rate determination using 802.15.4z ir-uwb hardware for remote healthcare,’’ IEEE Sensors Journal, pp. 1–1, 2026.

[23] Texas Instruments, IWR6843, IWR6443 Single-Chip 60 to 64GHz mmWave Sensor Datasheet, Texas Instruments, Apr. 2025, rev. F. [Online]. Available: https://www.ti.com/lit/ds/symlink/iwr6843.pdf

ANTON LAMBRECHT received the M.Sc. degree in Computer Science Engineering from Ghent University, Belgium, in 2024. Shortly thereafter, in September 2024, he joined the Department of Information Technology (INTEC) at Ghent University as a researcher within the IDLab research group, where he is currently pursuing the Ph.D. degree. His research interests include RF-based sensing using Ultra-wideband technology, with a focus on healthcare applications through machine learning techniques.

REDA EL HAIL received his Master’s degree in Signal and Image Processing from the University of Rennes 1, France, and his Engineering degree in Electrical Engineering from the Mohammadia School of Engineers in Morocco. As a Ph.D. researcher in machine learning at KU Leuven in Belgium, his current research concentrates on machine learning techniques for indoor human monitoring using millimeter-wave radar technology, with a particular emphasis on enhancing model robustness towards different aspects such as unseen environments, multiple persons and open set recognition.

XIANJUN JIAO received his bachelor’s degree in Electrical Engineering from Nankai university in 2001 and Ph.D. in communication and information system from Peking University in 2006. Then, he worked in the industry, such as Nokia, Microsoft and Apple, on wireless systems. In 2016, he joined IDLab, a core research group of imec embedded in Ghent University and University of Antwerp, as a senior researcher. At imec, he works on real-time SDR (Software Defined Radio) platform, such as the openwifi project, open-source Wi-Fi chip design, which is used widely for research in wireless networking, sensing and security. His main interests are designing and implementing high performance/efficient PHY and MAC layers, parallel/heterogeneous computation for wireless communications.

PIETER CROMBEZ received the M.S. degree in electronic engineering from KU Leuven University, in 2005, and the Ph.D. degree in electronic engineering from KU Leuven University, in 2009, with his research on low power, reconfigurable transceivers for multistandard/multimode applications. He was a Research Assistant with the ESAT-MICAS Laboratory, KU Leuven University. He joined as the Research and Development Project Lead with Televic Healthcare in 2009, where he was responsible for the wireless research for next generation nurse call products. The main focus was on localization techniques that meet the harsh specifications for the healthcare market. Nowadays, holding the position of Research Lead, he is responsible for the long term research activities within Televic Healthcare with focus on wireless communication, localization and sensing. He has been involved in several national and European funded research projects, both as a participant and a project coordinator. He is a (co)author of several publications.

DOMINIQUE SCHREURS (Fellow, IEEE) received the M.Sc. degree in electronic engineering and the Ph.D. degree from the University of Leuven (KU Leuven), Leuven, Belgium, in 1992 and 1997, respectively. She has been a Visiting Scientist with Agilent Technologies, Santa Rosa, CA, USA, ETH Zürich, Zürich, Switzerland, and the National Institute of Standards and Technology, Boulder, CO, USA. She is currently a Full Professor with KU Leuven, where she is also the Chair of the Leuven ICT (the Leuven Centre on Information and Communication Technology). Her current research interests include the microwave and millimeter-wave characterization and modeling of transistors, nonlinear circuits, and bioliquids, and system design for wireless communications and biomedical applications. Prof. Schreurs served as the President of the IEEE Microwave Theory and Techniques Society from 2018 to 2019. She was an IEEE MTT-S Distinguished Microwave Lecturer. She has also served as the General Chair for the Spring Automatic RF Techniques Group (ARFTG) conferences in 2007, 2012, and 2018, and the President of the ARFTG organization from 2018 to 2019. She currently serves as the TPC Chair for the European Microwave Conference and also the Conference Co-Chair for the IEEE International Microwave Biomedical Conference. She was the Editor-in-Chief of the IEEE Transactions on Mi crowave Theory and Techniques.

PETER KARSMAKERS In 2010 Peter Karsmakers received his Ph.D. at KU Leuven, Department of Electrical Engineering. In 2013 as a post-doc, he co-founded the ADVISE research group at Geel campus together with a few other colleagues. From 2018 he is an Assistant Professor at KU Leuven in the Computer Science Department and since 2022 he is principal investigator at Flanders Make. His research focuses on designing machine learning algorithms that consider application-specific constraints. These can for example relate to the computing platform on which the algorithm will be deployed or background knowledge about the machine learning task.

ADNAN SHAHID (Senior Member, IEEE) received the Ph.D. degree in Information and Communication Engineering from Sejong University, South Korea, in 2015. He is currently an Assistant Professor with the Internet Tech nology and Data Science Laboratory (IDLab), Ghent University–imec, where he leads the “AI/ML for Wireless” research within the IDLab Intelligent Wireless Networking (iWINe) group. He actively contributes to international standardization efforts, including the IEEE P1900.8 Working Group on Training, Testing, and Evaluating Machine-Learned Spectrum Awareness Models and the ATIS Working Group on Generative AI in Telecommunications. He has participated in several major research initiatives, including the DARPA Spectrum Collaboration Challenge (SC2) and multiple H2020, Horizon Europe, and ESA projects. He has authored over 150 publications in leading conferences and journals. His research interests include wireless physical-layer foundation models, decentralized and federated learning, radio resource management, 5G/6G networks, and non-terrestrial networks (NTN).

ELI DE POORTER is professor at the IDLab research group from imec & Ghent University (https://idlab.technology). His team performs research on wireless communication technologies such as (indoor) localization solutions, connected healthcare, wireless IoT solutions and machine learning for wireless systems. He performs both fundamental and applied research. He currently has around 350 publications and has a h-index of 50 (google scholar) with around 1400 citations per year. Since 2021, Prof. De Poorter has been included yearly in the Stanford University (Stanford, CA, USA), World’s Top 2% Scientists in the domain of Networking & Telecommunications.