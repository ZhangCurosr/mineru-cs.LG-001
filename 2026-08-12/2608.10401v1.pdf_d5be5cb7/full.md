Received 8 October 2025, accepted 18 October 2025. Digital Object Identifier 10.1109/ACCESS.2025.3624246

# Automatic Field-of-View Adjustment for a View-Expansive Microscope via LSTM-Based Gaze and Pipette Motion Interpretation

KENTA YOKOE<sup>1</sup>, (Member, IEEE), TAKUYA HARA<sup>2</sup>, and

TADAYOSHI AOYAMA<sup>1,</sup> <sup>3</sup>, (Member, IEEE)

<sup>1</sup>Department of Mechanical Systems Engineering, Nagoya University, Nagoya, Aichi, 464-8603, Japan (e-mail: yokoe@nagoya-u.jp, tadayoshi.aoyama@mae.nagoya-u.ac.jp)

<sup>2</sup>Department of Micro-Nano Mechanical Science and Engineering, Nagoya University, Nagoya, Aichi, 464-8603, Japan (e-mail: hara@robo.mein.nagoya-u.ac.jp)

<sup>3</sup>Center for One Medicine Innovative Translational Research, Gifu University, Yanagido 1-1, Gifu, Gifu 501-1193, Japan

Corresponding author: Kenta Yokoe and Tadayoshi Aoyama (e-mail: yokoe@nagoya-u.jp, tadayoshi.aoyama@mae.nagoya-u.ac.jp).

This work was supported by JST CREST, Grant No. JPMJCR20D5, Japan. The study protocol was approved by the Ethics Committee of the Graduate School of Engineering, Nagoya University (approval Number: 20-23). This is the accepted version of an article published in IEEE Access, vol. 13, pp. 182915–182923, 2025, DOI: <sub>10.1109/ACCESS.2025.3624246.</sub>  <sub>2025</sub> <sub>The</sub> <sub>Authors.</sub> <sub>Open</sub> <sub>Access</sub> <sub>under</sub> <sub>CC</sub> <sub>BY</sub> <sub>4.0</sub>g <sub>(https://creativecommons.org/licenses/by/4.0/).</sub>u

ABSTRACT Intracytoplasmic sperm injection (ICSI) operators frequently adjust the field-of-view (FOV) during procedures, which interrupts workflow and increases procedure time. Conventional microscopes require manual objective lens switching and illumination adjustments to achieve diferent FOV sizes. We propose an AI-based automatic FOV adjustment method integrated with a view-expansive microscope. This microscope enables the simultaneous acquisition of a large FOV and high-resolution images using a single objective lens through multiview imaging with galvanometer mirrors and high-speed vision, thereby eliminating the need for physical lens exchanges. Our method utilizes a long short-term memory (LSTM) model to predict the appropriate FOV size based on real-time analysis of the pipette’s position and velocity, combined with the operator’s gaze position. The AI model is trained using ICSI procedure data from an expert with over five years of micromanipulation experience. Experimental evaluation with novice operators reveals that the proposed automatic FOV adjustment system significantly improves the ICSI procedure speed, reducing the average task completion time from 60.5 to $4 8 . 0 \mathrm { ~ s ~ } ( p < 0 . 0 0 1 )$ . The experiments also demonstrate that this improvement enables novice operators to achieve ICSI working speeds equivalent to those of expert operators.

INDEX TERMS Micromanipulation system, long short-term memory, field-of-view adjustment

## I. INTRODUCTION

NTRACYTOPLASMIC sperm injection (ICSI) is an I assisted reproductive technology (ART) that has experienced significant global growth in recent years [1]. According to the International Committee for Monitoring Assisted Reproductive Technologies, ICSI currently accounts for approximately 70% of all ART cycles worldwide [2]. This can be attributed to ICSI’s capability to address severe male infertility and the improved success rates resulting from the standardization of the procedure [3], [4]. Additionally, technical advancements, such as the introduction of laser- and piezo-ICSI, have improved oocyte survival rates and contributed to the global increase in ICSI cases [5], [6].

ICSI procedures involve highly complex micromanipulation, requiring precise pipette operations and field-ofview (FOV) adjustments across multiple interfaces. As shown in (a) of Fig. 1, ICSI is conducted in three distinct workspaces:

![](images/0fffc75958b052181a9d10f23c6f3e4ee5814fbbbb4322deec341043af1bbf23.jpg)  
FIGURE 1. Workspaces of ICSI procedure and image presentation of view-expansive microscope.

Workspace 1: Area where oocytes are placed before injection

Workspace 2: Injection area

Workspace 3: Area where oocytes are released after injection

The ICSI procedure consists of three main processes performed repeatedly: (1) Moving the oocyte from Workspace 1 to Workspace 2 using a holding pipette; (2) injecting sperm into the target oocyte with an injection pipette in Workspace 2; and (3) transferring the treated oocyte from Workspace 2 to Workspace 3. Operators must adjust the FOV and brightness to perform these tasks smoothly. For example, task (1) is performed at a magnification level that allows the target oocyte to be held with a holding pipette (typically using a mediummagnification lens). Task (2) requires high-resolution observation for precise manipulation with the injection pipette (typically using a high-magnification lens), and task (3) involves releasing the oocyte and returning to Workspace 1 under a large FOV (typically using a low-magnification lens). As changing the objective lens magnification afects brightness, operators must adjust the brightness accordingly. Switching the objective lens using a revolver can also cause operators to lose sight of the oocytes. These frequent manual adjustments to objective lenses and brightness settings disrupt workflow, increase procedure time, and impact the success rate and reproducibility of ICSI. Consequently, the complexity of ICSI procedures necessitates extensive training to

![](images/f45cb10dce32bbaa94762f2afb2788f4940b0169c3b86eb7d801e8f9f934a01f.jpg)  
FIGURE 2. Schematic of proposed system. The AI model automatically adjusts FOV of displayed image based on operator’s gaze and pipette’s position.

master it [7]–[9].

Our research group has previously developed a viewexpansive microscope that enables the simultaneous presentation of large-FOV images ((b) in Fig. 1) and high-resolution images ((c) in Fig. 1) using galvanometer mirrors and high-speed vision [10]. This microscope eliminates the need for objective lens changes and brightness adjustments during ICSI procedures, allowing operators to achieve faster micromanipulation than with conventional microscopes without losing sight of the oocyte. However, this microscope presents operators with vast amounts of visual information, potentially exceeding human data-processing capabilities. Additionally, operators must manually determine the FOV size of the displayed image, which diverts operators’ attention from the primary manipulation task.

To address these limitations, we proposed an AIbased automatic FOV adjustment method integrated with a view-expansive microscope. Figure 2 presents a schematic of the proposed system. Our proposed system employs an LSTM model that predicts the appropriate FOV size based on real-time analysis of the pipette’s position and velocity, combined with the operator’s gaze position. The operator’s intention can be estimated by incorporating gaze movements as input to the AI. The AI model was trained using ICSI procedure data from an expert with over five years of micromanipulation experience. The proposed method eliminates the need for operators to manually change the FOV size, allowing them to maintain a continuous focus on primary manipulation tasks without interruption. To evaluate the proposed system, we conducted an experiment with novice operators who performed ICSImimicking tasks using microbeads. The experimental results demonstrated significant improvements in ICSI speed, enabling novices to achieve ICSI performance at the same speed as experts. The remainder of this paper is organized as follows: Section I provides the introduc tion and motivation for this study. Section II reviews related work on automation and support systems for ICSI, as well as gaze-based intention recognition for human-robot interactions. Section III-A describes the view-expansive microscope system with gaze-tracking functionality. Section III-B presents an AI-based FOV adjustment algorithm that utilizes LSTM. Section IV evaluates the system’s performance through speed evaluation experiments on ICSI procedures with novice and expert operators. Section V discusses the implications of our findings and the limitations of the proposed approach. Finally, Section VI concludes the paper and outlines future research directions.

## II. RELATED WORK

Numerous studies have been conducted on the automation and simplification of micromanipulation procedures, including ICSI [3]. Among these, laser-ICSI and piezo-ICSI are representative technologies that have been implemented at the clinical level and are widely used [11], [12]. Additionally, research has demonstrated the efectiveness of using microfluidic technology for sperm and oocyte selection [13]–[16]. Most of these technologies aim to improve embryo success rates and the speed of ICSI by providing supplementary assistance to existing procedures rather than modifying the established ICSI workflow. However, several studies have focused on automating the ICSI procedure [9]. These studies aimed to replace the complex ICSI procedures performed by humans with robotic systems, reporting automation of tasks, including sperm immobilization [17]. Although these studies have automated some tasks within the ICSI procedure, they have not achieved complete automation, still necessitating human operators.

Therefore, instead of pursuing full automation, research on remote ICSI has been conducted to maximize the utilization of existing ICSI experts [18], [19]. While these systems require on-site operators, the studies en able ICSI experts to perform challenging tasks remotely, allowing for high-quality ICSI even in regions with few ICSI experts. These studies utilized conventional microscopes and ICSI interfaces, resulting in no reduction in the burden on experts. Additionally, eforts have been made to reduce the complexity of ICSI by enhancing microscope systems and interfaces, thereby increasing the number of experts [7], [20]–[22]. These studies have simplified several skilled techniques required for ICSI by assisting operators with machines and AI, thus improving the speed and accuracy of the procedures. However, these systems provide reactive assistance based on current operator actions, rather than proactive support that anticipates operator intentions.

In collaborative systems involving humans, machines, and AI, the critical factor is delivering support that aligns with human operations [23]. To estimate human intention, many studies have utilized gaze movements in human-robot interactions [24], [25]. These studies have achieved smooth human-robot interaction systems by estimating human intentions through gaze information. Additionally, gaze-based interface control has been explored across multiple domains including user interfaces [26], surgical robotics [27], [28], automotive applications [29], [30], and virtual reality (VR) and augmented reality (AR) systems [31]. These studies have achieved smooth human-robot interaction, such as FOV adjustment, by estimating human intentions through gaze information. However, these approaches employ the gaze as an explicit input modality, requiring operators to consciously perform specific gaze patterns or gestures to trigger system responses. Furthermore, these approaches have been applied primarily to macro-scale robot interactions and have not been adapted to micro-scale environments. Although gaze-based intention recognition has shown significant effectiveness in human-robot interaction systems, applications in ICSI support systems are limited. Therefore, this study addresses this application gap by introducing gaze-based intention recognition into ICSI support systems. In contrast to prior explicit gaze-based control systems, the proposed approach employs AI to interpret natural task-related gaze behavior implicitly, eliminating the need for conscious gaze control and enabling truly automatic FOV adjustment that anticipates operator intentions.

Specifically, we developed an LSTM-based model that integrates real-time gaze tracking with pipette position to predict the appropriate FOV size during ICSI procedures. By combining gaze-based intention recognition, our approach enables human-centered ICSI assistance that anticipates human needs rather than merely responding to explicit commands.

## III. VIEW-EXPANSIVE MICROSCOPE SYSTEM WITH AI-BASED FOV ADJUSTMENT A. HARDWARE CONFIGURATION

Figures 3 and 4 present the configuration and overview of the proposed system, respectively. The proposed system is based on the view-expansive microscope developed in a previous study [10]. In this study, gaze tracking functionality was added to the view-expansive microscope to understand the intention of the operator from gaze movements. The view-expansive microscope achieves high-resolution imaging through optical means (galvanometer mirrors and an electrically tunable lens) rather than computational super-resolution, ensuring real-time performance that is essential for ICSI procedures. The proposed system comprises an inverted microscope (IX73, OLYMPUS), a high-speed vision (MQ003MG-CM, Ximea), an electrically tunable lens (EL-10-30-C-VIS-LD-MV, Optotune), a lens driver (Lens Driver 4, Optotune), a two-axis galvanometer mirror (6210HSM 6 mm 532 nm, Cambridge Technology), a light source unit (LA-HDF158AS, Hayashi Repic Co., Ltd.), a control computer (OS Windows 10 Pro 64-bit, CPU Intel (R) Core (TM) i9-10859K 3.60 GHz, memory 32 GB RAM, GPU NVIDIA GeForce RTX), a monitor (FlexScan EV2781, EIZO), a D/A board (PEX-340416, interface), an A/D board (PEX-321216, interface), two microinjectors (FemtoJet 4i and CellTram 4r Air, Eppendorf), two micromanipulators (TransferMan4r, Eppendorf), a screen-based eye tracker (Tobii Pro Nano,

![](images/a8ee18cde76f1b7ce4f6bf0c0cfe6ca827f9beba2411a0c273ad3b53752c42a1.jpg)  
FIGURE 3. Configuration of proposed system.

![](images/fa21cec5b59d217f754419f1c83ea953a7df20dab0eb183b61a59f286bd4d5ee.jpg)  
FIGURE 4. Overview of proposed system.

## TABLE 1. AI model structure.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Input</td><td>8 nodes</td></tr><tr><td>LSTM</td><td>8 nodes (1 layer)</td></tr><tr><td>Output</td><td>1 node</td></tr><tr><td>Optimizer</td><td>RMSprop</td></tr><tr><td>Loss Function</td><td>Mean Squared Error (MSE)</td></tr><tr><td>Batch Size</td><td>8</td></tr><tr><td>Epochs</td><td>85</td></tr></table>

Tobii), and an objective lens (LWD 95 mm, Mitsutoyo). The magnification of the objective lens was 10×, with a working distance of 33 mm. The pixel pitch of the high-speed vision was 7.4 �m, enabling image capture at a resolution of 640×480 pixels and a maximum frame rate of 500 fps. This system configuration facilitates the simultaneous acquisition of high-resolution, large-FOV images with automatic FOV adjustment based on gaze tracking.

## B. ALGORITHM DESIGN

The proposed system employs an AI model to predict the appropriate FOV size of the view-expansive microscope based on gaze and pipette movements. Table 1 shows the structure of the AI model used in the proposed method. This study applies the LSTM architecture, considering system efficiency and real-time performance. ICSI involves handling irreplaceable biological material where procedural failures cannot be reversed, requiring extremely high system reliability and local operation without internet connectivity for patient privacy protection and connection stability. Transformer-based architectures were excluded due to their computational requirements and infrastructure dependencies. LSTM and gated recurrent units (GRUs) offer comparable realtime performance that is suitable for clinical requirements. Given this sufficient inference speed, LSTM was selected for its superior accuracy in capturing complex temporal dependencies [32] [33], prioritizing modeling capacity over marginal speed improvements. The separate forget and input gates of LSTM enable the model to maintain long-term context during three-workspace transitions while remaining responsive to immediate dynamics. Therefore, the proposed AI model operates locally without internet connectivity, eliminating communication-related latency and ensuring patient privacy. The AI input consists of eight data points updated at a frequency of 20 Hz:

1) Holding pipette position on the monitor $( x _ { \mathrm { h } } , y _ { \mathrm { h } } )$

2) Holding pipette velocity on the monitor $( { \dot { x } } _ { \mathrm { h } } , { \dot { y } } _ { \mathrm { h } } )$

3) Operator’s gaze position on the monitor $( x _ { \mathrm { g } } , y _ { \mathrm { g } } )$

4) Deviation of gaze position from holding pipette position $\left( x _ { \mathrm { g } } - x _ { \mathrm { h } } , y _ { \mathrm { g } } - y _ { \mathrm { h } } \right)$

While vision-based AI systems using deep learning have demonstrated success in human-machine interaction applications [34], [35] , these methods often lack realtime performance due to high computational costs. To achieve the real-time responsiveness required for ICSI, the proposed system was designed using pipette position and velocity data rather than image-based features. The holding pipette position/velocity and gaze position were selected as input features because the holding pipette is actively manipulated throughout all ICSI phases and gaze information effectively captures operator attention. The injection pipette position/velocity was excluded because it exhibits minimal movement compared to the holding pipette, and the oocyte position/velocity was excluded because oocytes are manipulated by the holding pipette, making this information redundant. Therefore, the selected eight-dimensional features are sufficient for FOV prediction. These inputs are nor malized to a range of 0.0 to 1.0 before being fed into the AI model. The output of the AI is the appropriate FOV of the image. The appropriate FOV parameter, $F _ { \mathrm { p r e d } }$ , is an output value between 0.0 and 1.0. For the training data, we utilized data from 14 trials (11,520 steps) of ICSI procedures performed by an operator familiar with ICSI procedures. The validation data were derived from three trials (2,768 steps) of the ICSI procedure conducted by the same operator. The AI model was trained using data from a single expert to avoid the multimodality problem inherent in ICSI procedures. Different experts and clinics employ varying operational techniques, pipette manipulation styles, and FOV preferences. Combining such diverse training data could result in conflicting learning targets that degrade model performance. Single-expert training was strate gically selected to ensure consistent learning objectives for the LSTM model. Each ICSI procedure in the training and validation datasets represents a complete procedure performed on a single oocyte, encompassing the entire three-workspace workflow described in Section I: oocyte holding and positioning in Workspace 1, sperm injection in Workspace 2, and treated oocyte release in Workspace 3. The actual expert operational data cannot be shared publicly due to proprietary concerns regarding individual identification and unauthorized replication of specialized micromanipulation techniques. We employed PyTorch for the AI model. The coeficient of determina tion was 0.91. The AI inference time is 3.9-6.0 ms, which is sufficient to avoid affecting the response time of the view-expansive microscope.

Based on the appropriate FOV parameter $F _ { \mathrm { p r e d } } \in [ 0 , 1 ]$ output by the proposed AI model, the view-expansive microscope adjusts the FOV of the microscopic image. The width of the image displayed to the operator, $L _ { \mathrm { d } }$ , is determined by the actual FOV parameter � as follows:

$$
\begin{array} { r } { L _ { \mathrm d } = F L _ { \mathrm s } + ( 1 - F ) L _ { \mathrm e } , } \end{array}\tag{1}
$$

where $L _ { \mathrm { s } }$ represents the width of the highest-resolution image captured by our view-expansive microscope, and $L _ { \mathrm { e } }$ represents the maximum expanded FOV. When

![](images/3a6ffc8f63b44d1d7baef4bbe02e27dce3ea7e40dffcdb8a57cbea4a542dabc2.jpg)  
FIGURE 5. Overview of a participant during experiment.

$F = 1$ , the system displays the highest-resolution image $\left( L _ { \mathrm { d } } \ = \ L _ { \mathrm { s } } \right)$ . When $ { F } =  { 0 }$ , the system displays the maximum FOV image $\left( L _ { \mathrm { d } } = L _ { \mathrm { e } } \right)$ . In our implementation, $L _ { \mathrm { s } }$ corresponded to 500 �m, while $L _ { \mathrm { e } }$ could reach up to 1.5 mm. The FOV parameter $F$ was controlled to track the AI prediction $F _ { \mathrm { p r e d } }$ as follows:

$$
\dot { F } = \mathrm { s g n } ( F _ { \mathrm { p r e d } } - F ) \cdot \mathrm { m i n } ( | F _ { \mathrm { p r e d } } - F | , \dot { F } _ { \mathrm { m a x } } ) ,\tag{2}
$$

where $\dot { F } _ { \mathrm { m a x } }$ represents the maximum speed of FOV change required to prevent abrupt visual transitions, which can cause operator disorientation. Specifically, rapid FOV changes interfere with the operator’s observation, leading to a loss of sight of the pipettes and oocytes. Therefore, operators must frequently re-acquire pipettes and oocytes, which degrades manipulation precision and extends the duration of the ICSI procedure. Such issues are commonly observed in traditional microscopes when operators manually change the objective lenses during delicate micromanipulation procedures. To adjust the FOV size using Equations (1) and (2), the center of the FOV $\left( x _ { \mathrm { c } } , y _ { \mathrm { c } } \right)$ is determined as follows:

$$
\left\{ \begin{array} { l l } { x _ { \mathrm { c } } = x _ { \mathrm { h } } + r _ { \mathrm {  o } } } \\ { y _ { \mathrm { c } } = y _ { \mathrm { h } } } \end{array} \right. \quad ,\tag{3}
$$

where $x _ { \mathrm { h } }$ and $y _ { \mathrm { h } }$ represent the position of the holding pipette, and $r _ { 0 }$ represents the radius of the oocyte. By adjusting the FOV centered at the position defined by Equation (3), operators can perform smooth ICSI without losing sight of the holding pipette or oocyte.

## IV. SPEED EVALUATION OF ICSI PROCEDURE A. EXPERIMENTAL SETUP

To evaluate the efectiveness of the proposed system, we conducted an experiment using 100 �m microbeads, which are similar in size to human oocytes. The experiment was designed to simulate the ICSI procedure described in Section I. Participants were required to perform the following ICSI procedure: pick up a microbead from Workspace 1, move it to Workspace 2, touch an injection pipette tip to the microbead, and then move the microbead to Workspace 3.

![](images/623f6d788b82541e3e9b067b74f61000a001dad9b83be43ace9b245eca859f8d.jpg)  
FIGURE 6. Time series of images presented to operator during use of proposed system.

Participants performed this procedure using six microbeads. The experiment was conducted under two conditions: with a view-expansive microscope featuring automatic FOV adjustment and with a view-expansive microscope without automatic FOV adjustment. In the absence of automatic FOV adjustment, participants adjusted their FOV using a mouse wheel. This experimental design isolates the effect of automatic FOV adjustment by comparison with manual adjustment within the same microscope system. Comparisons with conventional microscopes or other gaze-based FOV systems were not included because the advantages of a viewexpansive microscope over a conventional microscope have already been demonstrated [10], and existing gazebased systems employ fundamentally different operational paradigms requiring explicit gaze control. Instead, expert performance was included as a reference benchmark. Figure 5 presents an overview of the participants during the experiment. Six individuals with no prior experience in micromanipulation participated in the study. Before the experiment, participants received an explanation of the experimental procedure and viewexpansive microscope system. Informed consent was obtained from all participants before the experiment commenced. Participants then completed a ten-minute practice session to familiarize themselves with the task and the view-expansive microscope. After the prac tice session, participants performed the experimental task using six microbeads. The order of experimenta conditions was counterbalanced for each participant. Additionally, an expert with over five years of micromanipulation experience performed the same experimental task using a conventional microscope system that they were familiar with. We evaluated the proposed system by measuring the time participants took to complete the experimental task.

## B. EXPERIMENTAL RESULTS

Figure 6 shows the time series of the images presented to the operator during the experiment with the proposed system. At the beginning of the experimental task, participants moved the holding pipette to Workspace 1 in Fig. 1 to hold a microbead (0–10 s in Fig. 6). Upon the arrival of the holding pipette in Workspace 1, the proposed system adjusted the FOV (12–15 s in Fig. 6). Each participant held one microbead (15–20 s in Fig. 6). Subsequently, while transporting the microbead to Workspace 2, the FOV was expanded using the proposed system (20–25 s in Fig. 6). During the injection in Workspace 2, the proposed system presented highresolution images (25–35 s in Fig. 6). After completing the injection, the proposed system presented a large FOV image to the participant, who then moved the microbead to Workspace 3 (37–43 s in Fig. 6). The participants ejected the microbead (43–48 s in Fig. 6). This confirms that the proposed system can adjust the FOV according to gaze and pipette movements.

Figure 7 displays a violin plot of the task completion time for each condition. The solid line indicates the median values, and the dashed line represents the mean values. The median time taken by novices using the view-expansive microscope with and without the proposed automatic FOV adjustment was 51.5 s and 68 s, respectively. Meanwhile, the median time taken by experts using the conventional microscope system was 45 s. Results from the Kruskal–Wallis test with Bonferroni correction indicated that novices using the proposed system (i.e., the view-expansive microscope with automatic FOV adjustment) showed a significant diference compared to those using the view-expansive microscope without automatic FOV adjustment (� = 0.02). There was no significant diference, however, when compared to the conventional microscope. These results suggest that the proposed automatic FOV adjustment can enhance the speed of micromanipulation with a view-expansive microscope, enabling novices to perform micromanipulation at a speed comparable to that of experts using a conventional microscope.

![](images/fcb9258c4ee5b085a3d889640a3aa51ab3eb320809dc7fde434d39d075b03299.jpg)  
FIGURE 7. Violin plot of task completion time for each condition. Solid line represents median value, while dashed line represents mean value.

TABLE 2. Median time of experimental task for each participant under each condition.
<table><tr><td>Participant</td><td>(a) Automatic FOV adjustment [s]</td><td>(b) Manual FOV adjustment [s]</td></tr><tr><td>A</td><td>73.0</td><td>101.0</td></tr><tr><td>B</td><td>44.0</td><td>64.5</td></tr><tr><td>C</td><td>56.0</td><td>81.0</td></tr><tr><td>D</td><td>58.5</td><td>63.0</td></tr><tr><td>E</td><td>42.5</td><td>47.0</td></tr><tr><td>F</td><td>65.0</td><td>69.5</td></tr><tr><td>All</td><td>51.5</td><td>68.0</td></tr></table>

Table 2 lists the median time of the experimental task for each participant under each condition (A–F represent the participants). This result demonstrates that the proposed method reduced the median task completion time for all participants. Furthermore, the median values for participants B and E using the proposed system were lower than those of the experts. This indicates that certain novices performed ICSI procedures more rapidly than experts using a conventional microscope.

## V. DISCUSSION

The experimental results demonstrated that integrating the proposed AI-based FOV adjustment into a viewexpansive microscope improved the speed of ICSI procedures for novice operators. This speed is comparable to that of an expert performing ICSI with a conventional microscope. This finding is attributed to the automatic FOV adjustment, which eliminates the need for operators to manually adjust the magnification settings. When using a conventional microscope, the operator must rotate the revolver to switch the magnification of the objective lens. At this point, it is necessary to adjust the brightness of the microscopic image in accordance with the change in objective lens magnification. In contrast, a view-expansive microscope can present images ranging from a large FOV to high resolution using a single objective lens. Therefore, there is no need to rotate the revolver for FOV changes, and brightness adjustment is not required. However, FOV adjustments must be made using an interface other than a revolver (such as a mouse wheel in the experiments). The proposed automatic FOV adjustment removes the need for an interface for FOV adjustments, allowing the operator to perform experimental tasks without removing their hands from the joystick used for pipette manipulation. This elimination of interface-switching is believed to enhance the speed of the ICSI procedure. To further enhance operational speed in human-machine systems, it may be necessary to focus not only on simplifying interfaces but also on reducing the number of interfaces that operators must manage.

The experimental results showed individual difer ences among the subjects. While participants B and E achieved ICSI procedure speeds exceeding those of the expert, participants D and F showed only a 4.5 s time reduction compared to manual FOV adjustment and remained slower than the expert. Nevertheless, all participants showed performance improvements, and the median task completion time for novice operators using the proposed system (51.5 s) was statistically equivalent to expert performance using conventional microscopes $( 4 5 \ \mathrm { s } , \ \mathrm { p } > 0 . 0 5 )$ , successfully achieving the primary research objective of enabling novice operators to perform ICSI procedures at expert-level speeds. The proposed AI model was trained by a single expert, and as a result, it presents the operator with the expert’s preferred FOV size during ICSI procedures. Consequently, while the FOV size may be suitable for some operators, it may be too large or too small for others. Furthermore, there may be individual diferences in gaze movement patterns during ICSI, and the proposed AI is likely to achieve smooth FOV changes for novices whose gaze movement patterns resemble those of experts. Previous studies have demonstrated such individual differences in human gaze patterns [36], [37]. The performance variations among participants may reflect these individual gaze pattern differences. Using a single-expert training dataset limits the ability of the system to accommodate diverse individual gaze patterns, indicating the need for future AI models that can compensate for individual differences in FOV preferences and gaze movement patterns. These individual diferences in FOV preferences and gaze movement patterns account for the variance in task completion times observed across participants. Such individual diferences in the efectiveness of the proposed system are anticipated to diminish by increasing the number of experts used for training. However, experts may also difer in their preferred FOV sizes and gaze movement patterns—referred to as multimodality—indicating that alternative methods capable of handling multimodality may be necessary when training with data from multiple experts.

The proposed system employs a static pre-trained LSTM model to maintain constant performance once deployed. This approach prioritizes stability and reliability, which are paramount for medical applications. However, this model cannot be directly applied to different microscope environments or tasks and would require retraining for new operational conditions. Furthermore, developing systems capable of handling multiple tasks would necessitate architectural modifications to accommodate diverse datasets and varied operational tasks.

The proposed system employs the Tobii Pro Nano eye tracker for gaze acquisition. Like all screen-based eye trackers, this device requires per-operator calibration, which is time-consuming and does not always succeed on the first attempt. Moreover, calibration accuracy varies between sessions, introducing variability in gaze measurement precision. These calibration requirements represent a significant practical burden that must be addressed to improve system usability for clinical settings. Additionally, the gaze acquisition accuracy and drift characteristics of the eye tracker fundamentally limit the overall system performance. The eye tracker can experience drift during extended use due to head movements or lighting changes, potentially requiring recalibration between procedures in extended clinical sessions. These hardware-dependent limitations constrain system performance regardless of AI model improvements. Future enhancements require not only AI algorithm refinement but also advances in eye tracking hardware, such as improved tracking robustness, calibration-free methods, or more reliable gaze acquisition technologies.

## VI. CONCLUSION

In this study, we developed an AI-based automatic FOV adjustment system for view-expansive microscope to enhance the speed of ICSI procedures. The proposed system utilizes an AI model that predicts the appropriate FOV size based on the position and velocity of the holding pipette, as well as the operator’s gaze position on the monitor. The AI model was trained using data from an operator familiar with ICSI procedures. Experimental results demonstrated that novice operators using the proposed automatic FOV adjustment system achieved significantly shorter task completion times than those using manual FOV adjustment. Moreover, this performance was statistically equivalent to that of an expert operator using a conventional microscope, indicating that the proposed system enables novices to perform micromanipulation at expert-level speeds. The key contribution of this study lies in eliminating the need for manual FOV adjustments during ICSI procedures. By automatically adjusting the FOV according to the ICSI procedure, operators can maintain continuous operation of the pipettes without interruption, thereby enhancing operational speed. This finding suggests that reducing the number of interfaces requiring human operation is crucial for improving the performance of humanmachine systems.

Future studies should develop algorithms capable of handling multimodality across different operational styles of experts while incorporating adaptive fine-tuning mechanisms for individual operators. Additionally, addressing the inherent procedural variations across different clinics and embryologists will enhance the adaptability and personalization capabilities of the system across a broader range of operators. Moreover, advances in eye tracking hardware, such as calibration-free methods or improved tracking robustness, are essential for reducing the practical burden of system setup and maintenance in clinical environments. These hardware improvements, combined with algorithmic advances, will facilitate broader clinical adoption of automatic FOV adjustment systems.

## REFERENCES

[1] S. C. Esteves, “Intracytoplasmic sperm injection versus conventional IVF,” The Lancet, vol. 397, no. 10284, pp. 1521– 1523, 2021.

[2] D. Adamson, F. Zegers-Hochschild, S. Dyer, G. Chambers, J. de Mouzo, O. Ishihara, M. Kupka, S. C. Jwa, V. Baker, B. Fu, E. Elgindy, and M. Banker, “ICMART PRELIMI-NARY WORLD REPORT 2019,” Copenhagen, 2023.

[3] H. E. Malter, “Micromanipulation in assisted reproductive technology,” Reproductive BioMedicine Online, vol. 32, no. 4, pp. 339–347, 2016.

[4] P. Rubino, P. Viganò, A. Luddi, and P. Piomboni, “The ICSI procedure from past to future: A systematic review of the more controversial aspects,” Human Reproduction Update, vol. 22, no. 2, pp. 194–227, 2016.

[5] S. Abdelmassih, “Laser-assisted ICSI: A novel approach to obtain higher oocyte survival and embryo quality rates,” Human Reproduction, vol. 17, no. 10, pp. 2694–2699, 2002.

[6] D. Zander-Fox, K. Lam, L. Pacella-Ince, C. Tully, H. Hamilton, K. Hiraoka, N. O. McPherson, and K. Tremellen, “PIEZO-ICSI increases fertilization rates compared with standard ICSI: A prospective cohort study,” Reproductive BioMedicine Online, vol. 43, no. 3, pp. 404–412, 2021.

[7] N. Costa-Borges, S. Munné, E. Albó, S. Mas, C. Castelló, G. Giralt, Z. Lu, C. Chau, M. Acacio, E. Mestres, Q. Matia, L. Marquès, M. Rius, C. Márquez, I. Vanrell, A. Pujol, D. Mataró, M. Seth-Smith, L. Mollinedo, G. Calderón, and J. Zhang, “First babies conceived with Automated Intracytoplasmic Sperm Injection,” Reproductive BioMedicine Online, vol. 47, no. 3, p. 103237, 2023.

[8] M. Durban, D. García, A. Obradors, V. Vernaeve, and R. Vassena, “Are we ready to inject? Individualized LC-CUSUM training in ICSI,” Journal of Assisted Reproduction and Genetics, vol. 33, no. 8, pp. 1009–1015, 2016.

[9] Z. Lu, X. Zhang, C. Leung, N. Esfandiari, R. F. Casper, and Y. Sun, “Robotic ICSI (Intracytoplasmic Sperm Injection),” IEEE Transactions on Biomedical Engineering, vol. 58, no. 7, pp. 2102–2108, 2011.

[10] T. Aoyama, S. Takeno, K. Yokoe, K. Hano, M. Takasu, M. Takeuchi, and Y. Hasegawa, “Micromanipulation System Capable of Simultaneously Presenting High-Resolution and Large Field-of-View Images in Real-Time,” IEEE Access, vol. 11, pp. 34 274–34 285, 2023.

[11] K. H. Choi, J. H. Lee, Y. H. Yang, T. K. Yoon, D. R. Lee, and W. S. Lee, “Eficiency of laser-assisted intracytoplasmic

sperm injection in a human assisted reproductive techniques program,” Clinical and Experimental Reproductive Medicine, vol. 38, no. 3, pp. 148–152, 2011.

[12] M. Caddy, S. Popkiss, G. Weston, B. Vollenhoven, L. Rombauts, M. Green, and D. Zander-Fox, “PIEZO-ICSI increases fertilization rates compared with conventional ICSI in patients with poor prognosis,” Journal of Assisted Reproduction and Genetics, vol. 40, no. 2, pp. 389–398, 2023.

[13] J. F. Aderaldo, K. d. S. Maranhão, and D. C. F. Lanza, “Does microfluidic sperm selection improve clinical pregnancy and miscarriage outcomes in assisted reproductive treatments? A systematic review and meta-analysis,” PLOS ONE, vol. 18, no. 11, p. e0292891, 2023.

[14] O. Olatunji, A. More, O. Olatunji, and A. More, “A Review of the Impact of Microfluidics Technology on Sperm Selection Technique,” Cureus, vol. 14, no. 7, 2022.

[15] L. Weng, G. Y. Lee, J. Liu, R. Kapur, T. L. Toth, and M. Toner, “On-chip oocyte denudation from cumulus–oocyte complexes for assisted reproductive therapy,” Lab on a Chip, vol. 18, no. 24, pp. 3892–3902, 2018.

[16] Y. Fang, R. Wu, J. M. Lee, L. H. M. Chan, and K. Y. J. Chan, “Microfluidic in-vitro fertilization technologies: Transforming the future of human reproduction,” TrAC Trends in Analytical Chemistry, vol. 160, p. 116959, 2023.

[17] Z. Zhang, C. Dai, J. Huang, X. Wang, J. Liu, C. Ru, H. Pu, S. Xie, J. Zhang, S. Moskovtsev, C. Librach, K. Jarvi, and Y. Sun, “Robotic Immobilization of Motile Sperm for Clinical Intracytoplasmic Sperm Injection,” IEEE Transactions on Biomedical Engineering, vol. 66, no. 2, pp. 444–452, 2019.

[18] G. Mendizabal-Ruiz, A. Chavez-Badiola, E. Hernández-Morales, R. Valencia-Murillo, V. Ocegueda-Hernández, N. Costa-Borges, E. Mestres, M. Acacio, Q. Matia-Algué, A. F.-S. Farías, D. S. M. Carreon, C. Barragan, G. Silvestri, A. Martinez-Alvarado, L. M. C. Olmedo, A. V. Aguilar, D. J. Sánchez-González, A. Murray, M. Alikani, and J. Cohen, “A digitally controlled, remotely operated ICSI system: Case report of the first live birth,” Reproductive BioMedicine Online, vol. 50, no. 5, p. 104943, 2025.

[19] G. Mendizabal, J. Cohen, N. Costa-Borges, A. Flores-Saife Farías, E. Hernández-Morales, M. Reyes-Llamas, A. Valadez Aguilar, C. Barragán, D. Martinez, A. Martínez, Q. Matia, E. Mestres, D. C. Ruiz-Velasco, M. Alikani, and A. Chavez-Badiola, “O-185 Far away so close: Remote ICSI using an AI-powered robot,” Human Reproduction, vol. 39, no. Supplement\_1, p. deae108.218, 2024.

[20] T. Fujishiro, T. Aoyama, K. Hano, M. Takasu, M. Takeuchi, and Y. Hasegawa, “Microinjection System to Enable Real-Time 3D Image Presentation Through Focal Position Adjustment,” IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 4025–4031, 2021.

[21] K. Yokoe, T. Aoyama, T. Fujishiro, M. Takeuchi, and Y. Hasegawa, “An immersive micro-manipulation system using real-time 3D imaging microscope and 3D operation interface for high-speed and accurate micro-manipulation,” ROBOMECH Journal, vol. 9, no. 16, 2022.

[22] R. Mori, T. Aoyama, T. Kobayashi, K. Sakamoto, M. Takeuchi, and Y. Hasegawa, “Real-Time Spatiotemporal Assistance for Micromanipulation Using Imitation Learning,” IEEE Robotics and Automation Letters, vol. 9, no. 4, pp. 3506–3513, 2024.

[23] D. Trombetta, G. Rotithor, I. Salehi, and A. P. Dani, “Variable structure Human Intention Estimator with mobility and vision constraints as model selection criteria,” Mechatronics, vol. 76, p. 102570, 2021.

[24] Y. Pan and J. Xu, “Gaze-based human intention prediction in the hybrid foraging search task,” Neurocomputing, vol. 587, p. 127648, 2024.

[25] A. Belardinelli, “Gaze-Based Intention Estimation: Principles, Methodologies, and Applications in HRI,” J. Hum.- Robot Interact., vol. 13, no. 3, pp. 31:1–31:30, 2024.

[26] S. Stellmach and R. Dachselt, “Designing gaze-based user interfaces for steering in virtual environments,” in Proceedings of the Symposium on Eye Tracking Research and Applications. Santa Barbara California: ACM, 2012, pp. 131–138.

[27] J. Zhang, B. Wang, Z. Pan, and M. Li, “GazeScope: A Framework of Gaze Attention-Based Automatic Field-of-View Adjustment for Laparoscopic Robots,” IEEE Robotics and Automation Letters, vol. 10, no. 7, pp. 6560–6567, 2025.

[28] K. Fujii, G. Gras, A. Salerno, and G.-Z. Yang, “Gaze gesture based human robot interaction for laparoscopic surgery,” Medical Image Analysis, vol. 44, pp. 196–214, 2018.

[29] G. Prabhakar, A. Ramakrishnan, M. Madan, L. R. D. Murthy, V. K. Sharma, S. Deshmukh, and P. Biswas, “Interactive gaze and finger controlled HUD for cars,” Journal on Multimodal User Interfaces, vol. 14, no. 1, pp. 101–121, 2020.

[30] J.-h. Lee, I. Yanusik, Y. Choi, B. Kang, C. Hwang, J. Park, D. Nam, and S. Hong, “Automotive augmented reality 3D head-up display based on light-field rendering with eyetracking,” Optics Express, vol. 28, no. 20, pp. 29 788–29 804, 2020.

[31] R. Konrad, A. Angelopoulos, and G. Wetzstein, “Gaze-Contingent Ocular Parallax Rendering for Virtual Reality,” ACM Transactions on Graphics, vol. 39, no. 2, pp. 1–12, 2020.

[32] F. Shiri, T. Perumal, N. Mustapha, and R. Mohamed, “A Comprehensive Overview and Comparative Analysis on Deep Learning Models,” Journal on Artificial Intelligence, vol. 6, pp. 301–360, 2024.

[33] T. Shi and K. Shide, “A comparative analysis of LSTM, GRU, and Transformer models for construction cost prediction with multidimensional feature integration,” Journal of Asian Architecture and Building Engineering, pp. 1–16, 2025.

[34] Z. Xing, G. Ma, L. Wang, L. Yang, X. Guo, and S. Chen, “Toward Visual Interaction: Hand Segmentation by Combining 3-D Graph Deep Learning and Laser Point Cloud for Intelligent Rehabilitation,” IEEE Internet of Things Journal, vol. 12, no. 12, pp. 21 328–21 338, 2025.

[35] Z. Xing, Z. Meng, G. Zheng, G. Ma, L. Yang, X. Guo, L. Tan, Y. Jiang, and H. Wu, “Intelligent rehabilitation in an aging population: Empowering human-machine interaction for hand function rehabilitation through 3D deep learning and point cloud,” Frontiers in Computational Neuroscience, vol. 19, 2025.

[36] K. Packard, A. J. Haskins, and C. E. Robertson, “Stable individual diferences in gaze behavior reflect unique conceptual priority maps,” Journal of Vision, vol. 23, no. 9, p. 5886, 2023.

[37] M. D. Broda and B. de Haas, “Individual diferences in human gaze behavior generalize from faces to objects,” Proceedings of the National Academy of Sciences, vol. 121, no. 12, p. e2322149121, 2024.

![](images/5663de71501dcc319c7f0f1864ec747f6493bb0ae60cd17a2a65efe8f9d41981.jpg)  
haptic technology.

KENTA YOKOE (Member, IEEE) received the B.E. degree in mechanical engineering, the M.E. degree, and the Ph.D. degree in micro–nano mechanical science and engineering from Nagoya University, Nagoya, Japan, in 2021, 2023, and 2025, respectively. He is currently an Assistant Professor at Nagoya University. His research interests include virtual reality, human interfaces, human–machine interaction, and

![](images/31e7eb88066920e39ce4a377ef522e675639ec36abb66415411191f720bcdf52.jpg)

TAKUYA HARA received the B.E. degree in mechanical engineering, and M.E. degree in micro–nano mechanical science and engineering from Nagoya University, Nagoya, Japan, in 2021 and 2023, respectively. He is currently with DENSO Corporation.

![](images/6f70e58bed10e44a19db7b67d95e7810516ac0983d84f337437b56a741c83680.jpg)

TADAYOSHI AOYAMA (Member, IEEE) received the B.E. degree in mechanical engineering, the M.E. degree in mechanical science and engineering, and the Ph.D. degree in micro–nano systems engineering from Nagoya University, Nagoya, Japan, in 2007, 2009, and 2012, respectively. He was an Assistant Professor with Hiroshima University, Higashihiroshima, Japan, during 2012–2017, and Nagoya University during 2017–2019. He then became an Associate Professor with Nagoya University during 2019–2024. From 2018 to 2022, he was a PRESTO Researcher with JST. He is currently a Professor with the Department of Mechanical Systems Engineering, Nagoya University. His research interests include macro–micro interaction, VR/AR and human interfaces, AI-based assistive technology, micromanipulation, and medical robotics.