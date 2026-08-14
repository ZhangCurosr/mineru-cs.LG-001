# Understanding Backdoor Vulnerabilities in Vertical Federated Learning: The Gap Between Research and Practice

Ziqi Zhao, Jialin Lu, Junjie Shan, Junyuan Zhang, Shuya Yang, Ka-Ho Chow School of Computing and Data Science, The University of Hong Kong, Hong Kong SAR, China

Abstract—Vertical Federated Learning (VFL) enables organizations holding complementary features of shared entities to collaborate and train models. In this setting, the initiator can withhold information about the learning task, while other contributors participate without exposing their local datasets, creating an asymmetric information structure aligned with growing privacy demands. However, this asymmetry is a double-edged sword. Among various threats, backdoor attacks are particularly concerning because VFL not only enables malicious contributors to poison the model during training, but also allows them to activate the backdoor at inference time to manipulate predictions. Although prior work has reported nearperfect attack success rates and proposed effective defenses, we find that most findings fail to hold under realistic conditions, exposing a fundamental gap between research and practice. In this paper, we present a systematic, practice-oriented study of backdoor vulnerabilities in VFL, revealing this gap in both methodological design and evaluation practices. We show that existing approaches overlook key practical constraints and therefore rely on unrealistic prior knowledge. Furthermore, these limitations have remained hidden due to poorly designed evaluation practices in the literature. To bridge this gap, we redefine threat models under realistic constraints, propose practical backdoor workflows, and introduce BVBench, a backdoor-centric benchmark that enables fair, practical, and comprehensive evaluation, preloaded with state-of-the-art baselines. BVBench provides strong evidence of the fragility of the current understanding of VFL backdoor risks and establishes a foundation for steering research toward uncovering practical vulnerabilities and developing more meaningful defenses.

## 1. Introduction

Cross-organizational collaboration is key to unlocking the full potential of modern machine learning (ML), which thrives on rich and expressive data to achieve high predictive performance. In practice, organizations across domains often possess complementary features about the same set of entities, such as credit history in financial institutions [1] and health conditions in healthcare providers [2], [3]. Integrating these distributed features can enhance data expressiveness and improve predictive performance. However, despite enabling multi-billion-dollar opportunities, such collaborations remain difficult due to data privacy regulations [4], [5] that prohibit sharing raw data across organizational boundaries.

![](images/97d46b27cbbdc7c86db4aaa2679c49d43e95d595f53b4f338b9364a2b544a8e9.jpg)  
Figure 1: VFL enables an active party (e.g., an insurance company) to collaborate with passive parties (e.g., a wearable provider and a healthcare center) and leverage complementary features of shared entities to improve ML tasks, such as underwriting risk assessment, without sharing raw data. However, the inherent information asymmetry in VFL can be exploited: a malicious participant (e.g., the wearable provider) may implant a backdoor during training and later manipulate model predictions at inference time.

Vertical Federated Learning (VFL) [2], [6], [7], [8] has emerged as a promising solution to this challenge. As illustrated in Figure 1, an insurance company (the “active” party) can collaborate with a wearable provider and a healthcare center (the “passive” parties). When a potential client applies for insurance, the active party requests the passive parties to identify matching records, process them locally through pri vate feature extractors, and return the resulting embeddings. These embeddings, which are not human-interpretable, are aggregated by the active party and used by a classifier to predict underwriting risk. In this way, predictions can leverage sensitive information, such as health status and activity data, without exposing raw data. This capability is enabled by jointly training models across parties while preserving data locality and creating information asymmetry: the active party knows the ML task but not the raw data, whereas passive parties know their local data but not the task itself. While this asymmetry makes cross-organizational learning practical, it also introduces unique security risks.

Backdoor attacks [9], [10], [11] are a particularly concerning threat in VFL. In such attacks, an adversary manipulates the training process to implant a hidden association between a trigger pattern and a target prediction. At inference time, the presence of the trigger causes the model to produce adversary-chosen outputs while behaving normally otherwise. VFL provides a unique and favorable setting for such attacks: passive parties participate in both training, where backdoors can be implanted, and inference, where they can be activated. For example, in Figure 1, a malicious wearable provider could implant a backdoor during training and later inject trigger patterns into embeddings at inference time, overriding the classifier’s decision (e.g., approving an otherwise ineligible insurance application). Such scenarios raise serious concerns about the security of VFL systems and have motivated a growing body of research on attack and defense mechanisms. However, do these studies truly reflect how VFL backdoors behave in real-world deployments?

![](images/b197d497f49dca5b040f63a2ce630a5e0e97a5ce5a826090ffbf9b4915fb1770.jpg)  
Figure 2: The high attack success rates (red) reported in prior work can be misleading. Existing approaches often rely on assumptions that are unlikely to hold in practice (orange) or are evaluated under unrealistic experimental settings (blue).

In this paper, we argue that the current understanding of backdoor vulnerabilities in VFL is fragile. Despite rapid research progress, it remains unclear to what extent existing findings generalize to practice. This gap has important consequences: practitioners may underestimate risks when systems appear robust, or overtrust defenses that fail under realistic conditions. By systematizing existing work, we identify a disconnect between how VFL backdoor vulnerabilities are studied and how VFL systems operate in practice. We attribute this gap to misalignment along two dimensions: (i) methodological design and (ii) evaluation design.

Methodological Design. Existing attacks and defenses often rely on unrealistic assumptions regarding threat models and, consequently, problem formulation. For example, many attacks assume access to seemingly obtainable knowledge of the ML task (e.g., class semantics), even though the active party has no obligation to share such information. In practice, a malicious passive party is unlikely to know the task to which it contributes, including the number of classes, let alone their meanings. As shown in Figure 2 (orange), relaxing this assumption significantly reduces the success rate of state-of-the-art attacks. This finding suggests that prior work may fail to generalize to real deployments and overestimate attack effectiveness, highlighting the need to ground future research in more realistic assumptions.

Evaluation Design. The evaluation of VFL backdoor vulnerabilities is fragmented. Existing studies rely on unrealistic datasets, inconsistent experimental setups, and limited metrics that fail to capture critical properties. More importantly, attack and defense performance is highly sensitive to these choices. As shown in Figure 2 (blue), attacks reported to achieve near-perfect success rates can only reach a meaningless level of effectiveness on realistic VFL datasets, an effect previously overlooked due to the lack of standardized benchmarks and reproducible implementations. Overall, current evaluations do not provide a reliable or objective understanding of practical security risks.

As interest in VFL continues to grow in both academia and industry, the absence of a practice-oriented framework risks fostering misconceptions and hindering progress. To address this gap, we make the following contributions:

• Knowledge Systematization. We provide a structured analysis of backdoor attacks and defenses, uncover their limitations, redefine a threat model grounded in realistic assumptions, and identify a practical backdoor workflow.

• Evaluation Framework. We develop BVBench, a VFL backdoor-centric benchmark that includes realistic datasets, comprehensive metrics, and a standardized evaluation recipe for fair and reproducible evaluations.

• Empirical Studies. We evaluate existing methods under BVBench and show that they exhibit limited practicality and are highly sensitive to evaluation choices.

Our work benefits multiple stakeholders. For researchers, it clarifies realistic threat models and highlights open challenges under a practical backdoor workflow. For practitioners, it provides new understanding of risks and offers tools and guidelines for rigorous evaluation. Our artifacts are here.

## 2. Background

## 2.1. Vertical Federated Learning

Although VFL supports various model families, neural network-based VFL [12], [13], [14], [15], [16], [17], [18] has been the primary focus of recent research due to its strong representational power. We therefore focus on this setting. Figure 3 illustrates a system with one active party and N passive parties holding complementary features for shared entities. VFL consists of three stages:

Stage 1: Negotiation. The active party defines the ML task and collects labels for existing entities to be used for training and evaluation. Because it may lack sufficient features, or even any features at all, the active party recruits passive parties to contribute complementary data. The parties negotiate incentives and agree on system design choices (e.g., model architectures and optimization strategies), as well as participation in both training and inference.

Stage 2: Training. Training begins with privacypreserving entity alignment across parties, such as Private Set Intersection (PSI) [19], [20], [21]. At each iteration, the active party selects a minibatch of sample IDs and distributes them to passive parties (we use batch size 1 for brevity).

• Forward Pass: As shown in Figure 3 (green path), each passive party i retrieves its local data x<sub>i</sub> and feeds it into its bottom model $\mathcal { F } _ { i } ^ { B }$ to produce an embedding $\varepsilon _ { i } ,$ which is then sent to the active party. The active party concatenates these embeddings $\pmb { \mathcal { E } } _ { 1 } \oplus \cdots \oplus \pmb { \mathcal { E } } _ { N }$ and applies the top model $\mathcal { F } ^ { T }$ as a classifier to generate prediction ${ \hat { y } } .$

![](images/ac7c91cd5cc845825358c8b78d61b4261d2da5c409d1636f55819de8fa9c8fba.jpg)  
Figure 3: Information asymmetry is fundamental to VFL. During the forward pass (green), passive parties send only embeddings to the active party, which aggregates them to produce predictions. During the backward pass (red), the active party returns gradients with respect to each party’s embeddings. As a result, the active party has access to the task and labels but not the raw data, while passive parties retain raw data but have no visibility into the task or labels.

• Backward Pass: As shown in Figure 3 (red path), the active party computes the loss using predicted label yˆ and ground-truth label $y ,$ and updates the top model $\mathcal { F } ^ { \check { T } }$ via gradient descent. By the chain rule, it then computes and sends gradients w.r.t. each $\pmb { \mathcal { E } } _ { i }$ to the corresponding passive party i, which continues backpropagation to update its bottom model $\mathcal { F } _ { i } ^ { B }$ . Overall, all models $( \mathcal { F } ^ { T } , \mathcal { F } _ { 1 } ^ { B } , . . . , \mathcal { F } _ { N } ^ { B } )$ are jointly optimized to improve prediction accuracy.

This process continues until convergence or until a predefined number of epochs is reached.

Stage 3: Inference. In VFL, all parties must participate during inference. Upon receiving a query, the active party broadcasts the sample ID, requests embeddings from passive parties, and performs the forward pass described above.

## 2.2. Backdoor Vulnerabilities

Attacks. Backdoor attacks manipulate model behavior by implanting a hidden trigger during training that induces a target prediction at inference time. This is achieved by injecting a trigger pattern into a subset of training samples, allowing the model to associate the trigger with the desired output. A backdoor attack aims to preserve utility on clean inputs, induce the target output when the trigger is present, and remain difficult to detect during both the training-time manipulation and inference-time injection. While backdoor attacks have been extensively studied in centralized learning and horizontal federated learning (HFL) [22], [23], where clients hold disjoint samples with shared features, VFL introduces distinct opportunities and challenges.

• Opportunities: VFL requires passive parties to participate in training and inference and to communicate through embeddings, creating two advantages for the adversary i. (i) Triggers can be crafted and injected directly in the embedding space $( \mathrm { i } . \mathrm { e } . , \pmb { \mathcal { E } } _ { i } )$ rather than the input space (i.e., $\mathbf {  { x } } _ { i } )$ . This avoids input-space constraints (e.g., valid image pixel ranges). (ii) A malicious passive party has consistent control over part of the top model input, eliminating the need for extra mechanisms to inject triggers.

• Challenges: Passive parties cannot observe ground-truth labels [24], [25]. This complicates establishing the association between triggers and target outputs, which in non-VFL settings is typically achieved by consistently injecting triggers into training samples of the target class and allowing the model to learn this shortcut. In addition, only a small fraction of the entire model is accessible to the adversary (i.e., its own bottom model $\mathcal { F } _ { i } ^ { B } )$ , making it harder to control both attack effectiveness and stealth.

Defenses. Backdoor defenses aim to preserve model utility while preventing trigger-induced misbehavior. In VFL, defenses are the responsibility of the active party, which orchestrates the system and produces predictions for downstream use. Although a wide range of defenses has been proposed in centralized and HFL settings [26], [27], [28], [29], [30], VFL presents distinct opportunities and challenges.

• Opportunities: The active party observes embeddings from all passive parties $( \pmb { \mathscr { E } } _ { 1 } , . . . , \pmb { \mathscr { E } } _ { N } )$ , while each passive party i only observes its own contributions $( \mathrm { i } . \mathrm { e } . , \pmb { \mathcal { E } } _ { i } )$ . This asymmetry enables cross-party consistency checks and the potential identification of anomalous embeddings.

• Challenges: Because all parties are required during inference, excluding a suspected malicious party is impractical, as the entire system typically needs to be reconstructed from scratch. Hence, unlike non-VFL settings, mitigating backdoor effects is more preferred than detection alone.

In summary, VFL introduces unique structural constraints that fundamentally shape the design of attacks and defenses. Next, we systematically analyze existing approaches (Sections 3 and 4) and identify key limitations in current evaluation practices, motivating the need for a comprehensive benchmark (Section 5).

## 3. Systematizing VFL Backdoor Attacks

Current backdoor attacks share a similar workflow. First, they collect or infer labels for some training samples. Then, they craft a trigger pattern and use those labeled samples to establish a shortcut between this trigger and the target class. We systematize existing works through the lens of practicality in the threat model, attack design, and stealthiness.

## 3.1. Threat Model

Threat models define the conditions under which an attack is launched. Ensuring practicality is critical, as it determines whether the insights reflect real-world vulnerabilities. We summarize existing threat models as follows:

TABLE 1: Current VFL backdoor attacks exhibit practicality gaps in threat model and their current designs are either at odds with or poorly aligned with the practical backdoor workflow outlined in Section 3.2.
<table><tr><td rowspan="3">Attack</td><td rowspan="2" colspan="2">Threat Model</td><td colspan="5">Attack Design</td><td rowspan="2" colspan="2">Stealthiness</td></tr><tr><td colspan="2">Label Acquisition</td><td colspan="3">Trigger Design for Multi-Target Learning</td></tr><tr><td>Knowledge</td><td>Accessible?</td><td>Signal</td><td>Feasible?</td><td>Trigger Space</td><td>Trigger Pattern</td><td>Scalable?</td><td>Constraint</td><td>Undetectable?</td></tr><tr><td>BackSplitVFL [31]</td><td> $\mathcal { L } _ { t a r g e t }$ </td><td>x</td><td></td><td></td><td>Embedding</td><td>Class-aware</td><td>0</td><td> $\ell _ { \infty } .$  -norm</td><td></td></tr><tr><td>LMP [11]</td><td> $\mathcal { L } _ { t a r g e t }$ </td><td>x</td><td></td><td></td><td>Embedding</td><td>Class-aware</td><td></td><td></td><td>C</td></tr><tr><td>PMP [11]</td><td> $\mathcal { L } _ { t a r g e t }$ </td><td>x</td><td></td><td></td><td>Embedding</td><td>Class-aware</td><td></td><td></td><td>C</td></tr><tr><td>VILLAIN [10]</td><td> $\mathcal { L } _ { t a r g e t }$ </td><td>x</td><td>Gradients</td><td></td><td>Embedding</td><td>Fixed</td><td>O</td><td>Std. Dev.</td><td></td></tr><tr><td>BadVFL* [9]</td><td> $\mathcal { L } _ { t a r g e t }$ </td><td>x</td><td>Gradients</td><td></td><td>Input</td><td>Fixed</td><td>C</td><td></td><td>C</td></tr><tr><td>LFBA [32]</td><td> $\mathcal { L } _ { t a r g e t }$ </td><td>X</td><td>Gradients</td><td></td><td>Input</td><td>Fixed</td><td>C</td><td></td><td>C</td></tr><tr><td>BadVFL [33]</td><td> $\mathcal { L } _ { f u l l }$ </td><td>x</td><td>Embeddings</td><td></td><td>Input</td><td>Class-aware</td><td>1</td><td></td><td>O</td></tr><tr><td> $\mathrm { { H i j a c k V F L } } \ ( { \mathcal { P } } , \times ) \ [ 3 4 ]$ </td><td>P</td><td>x</td><td>Embeddings</td><td></td><td>Embedding</td><td>Class-aware</td><td></td><td></td><td>C</td></tr><tr><td> $\mathrm { H i j a c k V F L } \left( \mathcal { P } , \mathcal { L } _ { t a r g e t } \right) [ 3 4 ]$ </td><td> $\mathcal { P } { + } \mathcal { L } _ { t a r g e t }$ </td><td>x</td><td>Embeddings</td><td></td><td>Embedding</td><td>Class-aware</td><td></td><td></td><td>C</td></tr><tr><td> $\mathrm { H i j a c k V F L } \left( \times , \ : \mathcal { L } _ { f u l l } \right) \ : [ 3 4 ]$ </td><td> $\mathcal { L } _ { f u l l }$ </td><td>x</td><td>Embeddings</td><td></td><td>Embedding</td><td>Class-aware</td><td></td><td></td><td></td></tr><tr><td>BAEVFL [35]</td><td> $\mathcal { L } _ { t a r g e t }$ </td><td>x</td><td>Embeddings</td><td></td><td>Input</td><td>Class-aware</td><td></td><td></td><td>C</td></tr></table>

3.1.1. Attacker’s Goal. The adversary (a passive party i) constructs a trigger injection function $\mathcal { T } ( \bar { \mathcal { F } } _ { i } ^ { B } , \pmb { x } _ { i } , \bar { t } )$ for a designated target class t such that, when activated during inference, the prediction by the top model $\mathcal { F } ^ { T } ( \pmb { \mathscr { E } } _ { 1 } \oplus \cdots \oplus$ $\mathcal { T } ( \mathcal { F } _ { i } ^ { B } , \pmb { x } _ { i } , t ) \bar { \oplus \cdot \cdot \cdot } \oplus \pmb { \mathscr { E } } _ { N } ) \bar { = } t .$ . When the adversary submits benign embeddings produced by its bottom model $\mathcal { F } _ { i } ^ { B }$ (i.e., $\mathcal { F } _ { i } ^ { B } ( \mathbf { \bar { x } } _ { i } ) )$ , the prediction should maintain normal accuracy. This behavior is achieved by poisoning the VFL training process while remaining undetectable by the active party.

3.1.2. Attacker’s Capabilities. Most existing attacks assume capabilities consistent with those of a passive party in standard VFL. During training, the adversary can submit arbitrary embeddings to the active party and manipulate received gradients to update its bottom model. During inference, the adversary can submit trigger-injected embeddings to activate the backdoor and influence predictions.

3.1.3. Attacker’s Knowledge. The main distinction across existing attacks lies in their assumptions about prior knowledge, as summarized in Table 1 (Columns 2–3). The purpose of prior knowledge is to help acquire labels of training samples, which are essential for the attacker to craft triggers and establish the connection between the trigger and the target class. Current works make various assumptions, including:

• Target-Class Label $\mathcal { L } _ { t a r g e t } .$ All attacks require the adversary to be able to identify or infer some training samples belonging to the target class. For example, VIL-LAIN [10] and BadVFL<sup>∗</sup> [9] require one such sample, while BackSplitVFL [31] and LMP [11] require multiple. This requirement is unlikely to be met because the active party is not obligated to disclose task-specific information. Passive participants do not even know about the semantics of the classes, let alone identify a target class for attack.

• Full Label Space $\mathcal { L } _ { f u l l } .$ Some attacks, such as Bad-VFL [33], further assume knowledge of the entire label space, including labeled samples for all classes or the total number of classes. For the same reason, this is even more unlikely to be accessible than the target-class label alone.

• Top Model Posterior . Some attacks assume the adversary can observe the top model’s posterior during inference, which will be used to infer labels and optimize the trigger. This is in direct conflict with the VFL setting

where the passive parties have no access to the top model. In short, the information asymmetry in VFL makes such task-related knowledge inaccessible. We therefore redefine a practical threat model for VFL backdoor attacks as follows:

Practical Threat Model: The attacker possessing the capabilities as a passive party (§3.1.2) should achieve the attack goal (§3.1.1) without prior knowledge of the ML task, such as class semantics, target-class labels, full label space, or any other task-related information.

Under this threat model, no existing attacks remain feasible.

## 3.2. Attack Design

Based on the redefined threat model, we outline a practical backdoor workflow as follows:

Practical Backdoor Workflow: First, label acquisition should aim to cluster training samples of the same class without access to task-related knowledge. Second, each cluster should be assigned a pseudo-label and a dedicated trigger pattern, and the class semantics of each pseudolabel should be inferred after deployment by sending trigger-injected queries and observing responses. Third, backdoor learning should be formulated as a multi-target instance while balancing effectiveness across targets.

Next, we show that current attack designs fall short in aligning with this workflow.

3.2.1. Label Acquisition. Existing attacks rely on inaccessible labeled samples. Most of them justify this assumption by requiring only a few such samples and inferring the rest. They either exploit the gradients from the active party or the embeddings produced by the bottom model (Columns 4–5, Table 1), but such signals are unstable across training. As shown in Figure 4, the former is informative only in early training, while the latter remains useless until the bottom model begins to extract useful embeddings. In practice, learning progress depends on uncontrollable factors such as task complexity. As analyzed in Section 6, label quality plays a decisive role in backdoor attacks. Robust label acquisition from these unstable signals remains overlooked.

![](images/f90e9b955906db9e708ca4be08478d02a08a5298621bf90e04d925b8d2efc963.jpg)

(a) Gradients  
![](images/a360d3b6f11defe8850c564cf4c865b777f4496f270889f0e9fa58805900f84d.jpg)  
(b) Embeddings

Figure 4: Label inference can be unstable. The cluster effect of gradients and embeddings varies across training progress.  
![](images/a3c9713e0ca7c73a0933e68cc90f368baac4b1336553362b10391e550702302e.jpg)  
Figure 5: Trigger interference can hinder attack effectiveness in multi-target settings.

3.2.2. Trigger Design for Multi-Target Learning. Multitarget learning poses a significant challenge to trigger design for three reasons (Columns 6–8, Table 1). First, each pseudolabel requires a distinct trigger pattern and/or position, while some existing methods (VILLAIN [10], BadVFL<sup>∗</sup> [9] and LFBA [32]) use fixed and class-agnostic triggers, so the trigger pattern and injected position are the same for any target class. Second, it is non-trivial to scale to multiple targets even if trigger designs are class-aware because of potential interference. As shown in Figure 5, the coexistence of certain triggers can lead to mild (orange) or even drastic (green) degradation in attack effectiveness. Third, as more shortcuts need to be memorized, the trigger needs to produce stronger signals for the top model to discover. While our empirical analysis found that embedding-space triggers can be more effective, current methods provide no mechanism for balancing the attack strength across different classes, which can lead to cross-target conflicts and instability.

## 3.3. Stealthiness Consideration

Current stealthiness definitions are oversimplified and under-defined. Existing definitions rely on simple heuristics, such as repetitive patterns in the embedding space [31], or an abnormal range in the input or embedding space [10], [31]. We argue that stealthiness should be defined from the defender’s perspective and with the defender’s knowledge, while remaining assessable under the attacker’s knowledge and capabilities. Since embeddings are used in both training and inference and are accessible to both active and passive parties, they are the natural medium for defining stealthiness: if poisoned samples are easily identifiable in the embedding space, then the attack is not stealthy. Based on this intuition, we find that only BackSplitVFL [31] and VILLAIN [10] have relevant embedding-space constraints (Columns 9–10, Table 1). That said, these constraints cannot hide triggerinjected embeddings from defenders. For example, as shown in Figure 6, BackSplitVFL uses maximum per-dimension and overall norm of clean embeddings to constrain poisoned embeddings, but such a value is already highly anomalous; VILLAIN uses a trigger pattern based on the averaged standard deviation over dimensions that have high standard deviation, but adding such triggers can still lead to off-distribution embeddings. Furthermore, the defender can access all parties’ embeddings, which can make the attack even easier to detect if the defender leverages cross-party consistency checks.

![](images/cb60ace59958deb5cb92c3f08d1d36a1d60176806d5b6b0de264d43434d9e38a.jpg)  
Figure 6: Existing stealthiness constraints do not help attacks remain under the radar. For instance, BackSplitVFL [31] constrains trigger norm to be within the maximum norm of clean embeddings, but it is still highly anomalous and can be easily detected using the 99% quantile.

Practicality Insights: Current stealthiness definitions are fragile and poorly aligned with defender’s knowledge. It should be defender-driven, attacker-assessible, defined in the embedding space, and tight enough to make malicious embeddings indistinguishable from benign ones.

## 4. Systematizing VFL Backdoor Defenses

Backdoor defenses in VFL can be applied at both train time and test time, with three objectives: (i) mitigation, which reduces the attack influence during training; (ii) recovery, which aims to restore the original prediction of attacked samples; and (iii) detection, which identifies attacked samples. We systematize existing works through the lens of the defender’s knowledge, defense design, and the overlooked issue of post-attack utility recovery.

## 4.1. Threat Model

4.1.1. Defender’s Goal. The defender, acting as the active party, aims to prevent backdoor attacks from malicious passive parties and preserve clean model utility simultaneously. This can be achieved in three forms: (i) train-time mitigation; (ii) test-time recovery; and (iii) test-time detection.

TABLE 2: Current VFL backdoor defenses exhibit practicality gaps in prior knowledge, design, and recovery quality.
<table><tr><td>Defense</td><td colspan="2">Knowledge</td><td colspan="2">Defense Design</td><td rowspan="2">Utility</td></tr><tr><td></td><td>Free</td><td>Ref.- Adv. Kn.-| Free</td><td>Pro- Multi- Insensitive-Recovery active? Target? Hyperp.?</td><td></td></tr><tr><td colspan="6">(a) VFL-Specific Defenses</td></tr><tr><td>VFLMonitor [36]]</td><td>x</td><td>√</td><td>√</td><td>x</td><td>x</td></tr><tr><td>VFLIP [37]</td><td>x</td><td>√</td><td>x</td><td>√</td><td>x</td></tr><tr><td>GBD [18]</td><td>x</td><td>√</td><td>x</td><td>√ √</td><td>x</td></tr><tr><td>UBD [38]</td><td>x</td><td>x</td><td>√</td><td>√ x</td><td>x</td></tr><tr><td colspan="6">(b) Generic Defenses</td></tr><tr><td>Emb. Norm. [34]</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td></tr><tr><td>Emb. Drop. [36]</td><td>V</td><td>√</td><td>√</td><td>√</td><td>x</td></tr><tr><td>Grad. Noise [11]</td><td>VV</td><td>√</td><td>V</td><td>x</td><td>x</td></tr><tr><td>Grad. Prun. [35]</td><td></td><td>√</td><td>√</td><td>√</td><td>x</td></tr></table>

4.1.2. Defender’s Capabilities. Defenders’ capabilities are consistent with those of the active party. During training, the defender can inspect, manipulate and store embeddings from all passive parties before feeding them into the top model. The gradients derived from the top model can also be arbitrarily manipulated before sending back to passive parties. During inference, the defender can also manipulate embeddings and the top model.

4.1.3. Defender’s Knowledge. VFL-specific defenses require some prior knowledge: (i) clean reference embeddings and (ii) knowledge of the adversarial environment such as the number of attackers and target classes, as summarized in Table 2(a) (Columns 2–3). Obviously, these assumptions are unlikely to hold, and hence we redefine a practical threat model for VFL backdoor defenses as follows:

Practical Threat Model: The defender possessing the capabilities as an active party (§4.1.2) should achieve the defense goal (§4.1.1) without clean reference embeddings and the adversarial environment (§4.1.3).

Under this threat model, no existing VFL-specific defenses remain practical, as they all explicitly or implicitly rely on clean reference embeddings.

## 4.2. Defense Design

4.2.1. Proactive Mitigation. Our empirical analysis in Section 7 shows the clear advantage of VFL-specific defenses over generic ones, yet only a few methods adopt a proactive approach (Column 4, Table 2). Proactive mitigation before inference deserves more attention, as it reduces the inference burden and better preserves efficiency.

4.2.2. Multi-target Compatibility. Similar to attacks, defenses should also be designed to handle multiple target classes. Most current defenses can be adapted (Column 5, Table 2), but UBD [38] is incompatible, as it assumes a single target class and finetunes the top model to only reduce the attack effect on that class.

![](images/869847c97c961923ec15a0e7d2d2633fb4141fcbec6d4307bafe0b5982196fbc.jpg)  
Figure 7: Current defenses are highly sensitive to their hyperparameters. For instance, the top-k class predictions used for detection in VFLMonitor [36] can drastically affect its effectiveness yet cannot be tuned in practice.

4.2.3. Hyperparameter Sensitivity. A practical defense should be robust to unseen tasks and should not require hyperparameter tuning, so it should be designed to be insensitive to hyperparameter choices. However, some defenses cannot meet this requirement (Column 6, Table 2). For example, VFLMonitor [36] requires dataset-specific tuning on its k value for top-k predictions of each party. As shown in Figure 7, this value can significantly affect the tradeoff between detection performance (Attack TPR) and false alarm rate (Clean FPR) on a VFL dataset. Gradient noise also suffers from the same problem, as different datasets might have varying training gradient magnitudes and an unmatched noise level might be either ineffective or overly harmful to benign learning.

## 4.3. Post-attack Utility Recovery

Current works focus on reducing ASR while overlooking post-attack utility (Column 7, Table 2). In fact, even though some defenses claim to recover clean predictions from trigger-injected ones (e.g., VFLIP [37], VFLMonitor [36], and UBD [38]), they all fail miserably to preserve recovery utility. In other words, the attack may be suppressed while predictions remain incorrect. As widely agreed in general backdoor literature (e.g. Robust Accuracy in Backdoor-Bench [39]), the defense can be considered failed if the recovery attempt cannot restore the correct prediction.

## 5. BVBench

After identifying the practicality gap in methodological design, a natural question arises: how do current attacks and defenses perform in practice? Unfortunately, the answer remains unclear because the current evaluation practices fail to provide a reliable understanding of backdoor vulnerabilities. A meaningful evaluation should demonstrate improvements over state-of-the-art methods under fair, practical, and comprehensive settings. However, existing studies fall short along all these dimensions, as summarized in Table 3.

• State-of-the-art. Most works compare against only a few baselines (Columns 2–3), partly because reimplementing prior methods requires significant engineering effort, especially when more than half provide no official code (Column 13). As a result, comparisons are incomplete, making it difficult to assess true progress.

TABLE 3: Current evaluations of VFL backdoor methods are fundamentally flawed. Prior work fails to compare against state-of-the-art baselines, adopts inconsistent and highly sensitive configurations, relies on unrealistic datasets, and provides fragmented analysis with limited metrics. The lack of publicly available implementations further undermines reproducibility, making existing results difficult to trust or build upon.
<table><tr><td rowspan="2">Method</td><td colspan="2">Comparison</td><td colspan="4">VFL Configuration</td><td rowspan="2">Realistic Dataset</td><td colspan="4">Critical Evaluation Aspects</td><td rowspan="2">Open Source</td></tr><tr><td></td><td>Attack Defense</td><td>Top-Depth</td><td>Bottom-Arch</td><td>Opt</td><td>LR</td><td>Efficacy</td><td>Robustness</td><td>Dependency</td><td>Stealthiness</td></tr><tr><td colspan="10">(a) Attacks</td><td></td><td></td><td></td></tr><tr><td>BackSplitVFL</td><td>☆☆☆</td><td>★☆☆</td><td>3</td><td>CNN</td><td>Adam</td><td>1e-3</td><td>x</td><td>★★★</td><td>★★☆</td><td>☆☆☆</td><td>★★☆</td><td></td></tr><tr><td>LMP</td><td>★★☆</td><td>★★★</td><td>1</td><td>Tiny-ResNet</td><td>?</td><td>1e-3</td><td>x</td><td>★★★</td><td>★☆☆</td><td>★★☆</td><td>☆☆☆</td><td></td></tr><tr><td>PMP</td><td>★★☆</td><td>★★★</td><td>1</td><td>Tiny-ResNet</td><td>?</td><td>1e-3</td><td>x</td><td>★★★</td><td>★☆☆</td><td>★★☆</td><td>☆☆☆</td><td>C</td></tr><tr><td>VILLAIN</td><td>★☆☆</td><td>★★☆</td><td>3</td><td>VGG16</td><td>SGD</td><td>1e-2</td><td>x</td><td>★★★</td><td>★★☆</td><td>★★★</td><td>★☆☆</td><td></td></tr><tr><td>BadVFL*</td><td>★☆☆</td><td>★☆☆</td><td>1</td><td>ResNet18</td><td>SGD</td><td>1e-2</td><td>x</td><td>★★★</td><td>★☆☆</td><td>★☆☆</td><td>☆☆☆</td><td></td></tr><tr><td>LFBA</td><td>☆☆☆</td><td>★☆☆</td><td>3</td><td>ResNet18</td><td>Adam</td><td>1e-3</td><td>x</td><td>★★★</td><td>★☆☆</td><td>☆☆☆</td><td>☆☆☆</td><td></td></tr><tr><td>BadVFL</td><td>☆☆☆</td><td>★☆☆</td><td>4</td><td>ResNet18</td><td>?</td><td>?</td><td>x</td><td>★★★</td><td>★★☆</td><td>★☆☆</td><td>★☆☆</td><td>O</td></tr><tr><td>HijackVFL</td><td>☆☆☆</td><td>★☆☆</td><td>3</td><td>ResNet18</td><td>Adam</td><td>1e-3</td><td>x</td><td>★★★</td><td>★☆☆</td><td>★☆☆</td><td>☆☆☆</td><td></td></tr><tr><td>BAEVFL</td><td>★★☆</td><td>★☆☆</td><td>4</td><td>ResNet20</td><td>SGD</td><td>1e-1</td><td>x</td><td>★★★</td><td>★★☆</td><td>☆☆☆</td><td>☆☆☆</td><td>O</td></tr><tr><td colspan="10">(b) Defenses</td><td colspan="3"></td></tr><tr><td>Method</td><td colspan="2">Comparison</td><td colspan="3">VFL Configuration</td><td></td><td colspan="2">Realistic</td><td colspan="3">Critical Evaluation Aspects</td><td>Open</td></tr><tr><td></td><td>Attack</td><td>Defense</td><td>Top-Depth</td><td>Bottom-Arch</td><td>Opt</td><td>LR</td><td>Dataset</td><td>Efficacy</td><td>Robustness</td><td>Dependency</td><td>Recovery</td><td>Source</td></tr><tr><td>VFLIP</td><td>★☆☆</td><td>★★☆</td><td></td><td>VGG19</td><td>SGD</td><td>?</td><td>x</td><td>★★★</td><td>★☆☆</td><td>★☆☆</td><td>☆☆☆</td><td>0</td></tr><tr><td>VFLMonitor</td><td>★☆☆</td><td>★☆☆</td><td>321</td><td>ResNet</td><td>Adam</td><td>1e-3</td><td>x</td><td>★★★</td><td>★☆☆</td><td>★☆☆</td><td>☆☆☆</td><td>O</td></tr><tr><td>UBD</td><td>★☆☆</td><td>★☆☆</td><td></td><td>ResNet18</td><td>?</td><td>?</td><td>x</td><td>★★★</td><td>★★☆</td><td>★★☆</td><td>☆☆☆</td><td>O</td></tr><tr><td>GBD</td><td>★★☆</td><td>★★★</td><td>4</td><td>ResNet20</td><td>SGD</td><td>1e-1</td><td>x</td><td>★★★</td><td>★☆☆</td><td>★★☆</td><td>☆☆☆</td><td></td></tr></table>

[Annotations] ? = not mentioned in the paper or source code | = open-sourced but lack key components

• Fairness. Existing evaluations adopt heterogeneous configurations (Columns 4–7), such as different model architectures and hyperparameters. We find that both attacks and defenses are highly sensitive to these choices. Even minor changes, such as model depth or optimizer, can significantly affect performance and favor certain methods. Consequently, reported improvements may reflect configuration bias rather than methodological advances.

• Practicality. Existing studies rely on artificial datasets that do not reflect real-world VFL deployments (Column 8). Given the distinct data characteristics of VFL, such simplifications can lead to misleading conclusions. Our analysis shows that methods performing well on synthetic setups can fail under realistic conditions.

• Comprehensiveness. Evaluations are fragmented, focusing on a limited set of metrics while overlooking other critical aspects (Columns 9–12). Inconsistent evaluation criteria further hinder meaningful comparison, obscuring the real-world implications of proposed methods.

We attribute these shortcomings to the lack of backdoorfocused benchmarks. Existing VFL benchmarks primarily emphasize system performance and provide limited support for evaluating backdoor attacks and defenses (Table 4). Building such a benchmark is challenging: it requires reimplementing diverse methods within a unified framework while ensuring extensibility for future work.

To address these challenges, we introduce BVBench, a backdoor-focused benchmark designed to enable fair, practical, and comprehensive evaluation of VFL vulnerabilities, preloaded with state-of-the-art attacks and defenses (Figure 8). Beyond serving as a toolkit, it provides a standardized evaluation framework for future research. As demonstrated in Section 6 and Section 7, BVBench reveals insights that remain hidden under current evaluation practices.

TABLE 4: Existing VFL benchmarks are not designed to evaluate backdoor vulnerabilities. While they support VFL engines and, in some cases, realistic datasets, they provide limited or no support for backdoor vulnerability assessment.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">VFL Engine</td><td rowspan="2">Realistic Dataset</td><td colspan="3">Backdoor Vulnerability</td></tr><tr><td>Attack</td><td>Defense</td><td>Analysis</td></tr><tr><td>VertiBench [40]</td><td>√</td><td>√</td><td>☆☆☆</td><td>☆☆☆</td><td>☆☆☆</td></tr><tr><td>MARS-VFL [14]</td><td>√</td><td>√</td><td>★☆☆</td><td>★☆☆</td><td>★☆☆</td></tr><tr><td>VFLAIR [41]</td><td>√</td><td>x</td><td>★☆☆</td><td>★★☆</td><td>★★☆</td></tr><tr><td>BVBench (Ours)</td><td>√</td><td>√</td><td>★★★</td><td>★★★</td><td>★★★</td></tr></table>

## 5.1. Ingredients

5.1.1. Unified VFL Engine. We design a unified VFL engine that standardizes model architectures, training configurations, and communication protocols. The engine strictly enforces the information asymmetry inherent in VFL, ensuring that each party accesses only the information consistent with its role. It is extensible to different VFL implementations and supports flexible configurations, enabling controlled variation of architectural and optimization choices.

5.1.2. Realistic VFL Datasets. We incorporate realistic VFL datasets (Table 5) that capture practical feature distributions and cross-party characteristics. The benchmark is designed to be easily extensible, allowing new datasets to be integrated with minimal effort. We also include CIFAR-10 as a reference for validating reimplementations, since existing methods all adopt this unrealistic dataset for evaluation, providing a common reference point for comparison.

![](images/5e3736b5f9378b08c995f19378310d612fddd78dc394254e8ff6eed1c53765df.jpg)  
Figure 8: BVBench is a modular benchmark for evaluating backdoor vulnerabilities in VFL. It integrates a unified VFL engine, realistic datasets, and standardized implementations of attacks and defenses into a unified evaluation pipeline, enabling fair, practical, and comprehensive assessment.

TABLE 5: Commonly used datasets such as CIFAR-10 (Row 1) rely on artificial feature splits, whereas the datasets in BVBench capture natural cross-party data characteristics.
<table><tr><td>Dataset</td><td>Modality</td><td>Realistic</td><td>#Parties</td><td>#Classes</td><td>#Samples</td></tr><tr><td>CIFAR-10 [42]</td><td>Image</td><td>x</td><td>2</td><td>10</td><td>60,000</td></tr><tr><td>Satellite [40]</td><td>Image</td><td>√</td><td>16</td><td>4</td><td>3,927</td></tr><tr><td>KUHAR [14]</td><td>Tabular</td><td>√</td><td>2</td><td>18</td><td>20,750</td></tr><tr><td>PTB-XL [14]</td><td>Multi-modal</td><td>√</td><td>3</td><td>5</td><td>16,244</td></tr><tr><td>Vehicle [40]</td><td>Tabular</td><td>√</td><td>2</td><td>3</td><td>78,823</td></tr><tr><td>NUSWIDE [40]</td><td>Tabular</td><td>√</td><td>5</td><td>2</td><td>269,648</td></tr></table>

5.1.3. State-of-the-art Attacks and Defenses. We curate a comprehensive collection of state-of-the-art attacks and defenses from major venues. The initial set includes nine attacks and eight defenses, all implemented in PyTorch with over 20,000 lines of code. While some methods provide open-source implementations and can be easily adapted to BVBench, others require reimplementation from scratch. For methods whose reproduced performance does not match the originally reported results, we have contacted the authors for clarification and assistance. The framework is designed for extensibility, allowing new methods to be integrated with minimal code changes.

## 5.2. Evaluation Recipes

BVBench provides standardized evaluation recipes for attacks and defenses. The recipes define the evaluation aspects and associated metrics that should be reported to enable fair, practical, and reproducible comparison. Detailed formulations are deferred to the appendix.

5.2.1. Attack Evaluation. A high attack success rate alone is insufficient to establish a practical attack. In realistic deployments, attacks must also preserve model utility, operate under limited prior knowledge, remain stealthy, generalize across settings, and avoid excessive overhead. We therefore evaluate attacks along the following dimensions.

• Efficacy: Evaluate attacks using both main task accuracy (MTA) and attack success rate (ASR) across diverse datasets. Since aggregate metrics can mask heterogeneity, conduct per-class analysis to examine variations across target classes and multi-target analysis to assess effectiveness when multiple classes are attacked simultaneously.

• Dependency: Quantify how attack effectiveness changes under different levels of required prior knowledge, such as imperfect label knowledge, noisy label inference, or other attack-specific prerequisites, and measure their impact on both ASR and MTA.

• Stability: Assess the consistency of attack performance across random seeds and training epochs, rather than reporting only the final checkpoint, using statistics such as the mean and variance of MTA and ASR.

• Robustness: Examine attack performance under different VFL environments, including varying numbers of participating parties, and multiple attackers with either shared or conflicting objectives.

• Stealthiness: Assess the distinguishability between poisoned and benign embeddings. We introduce stealthiness metrics that capture both abnormal embedding magnitudes and deviations from benign embedding distributions. We consider both an oracle setting based on class-conditional distributions and a label-free setting based on global embedding distributions, and quantify the separability between benign and malicious samples using ROC-AUC.

• Overhead: Report the computational, communication, memory, and optimization costs incurred by the attack.

5.2.2. Defense Evaluation. Reducing attack success alone is insufficient to establish a practical defense. Defenses must preserve normal utility, remain robust across attacks and datasets, and avoid excessive cost. We therefore evaluate defenses along the following dimensions.

• Efficacy: Evaluate defenses using MTA, defended ASR, and utility recovery (UR) across diverse datasets and attacks. Per-class and multi-target analyses should further assess whether protection remains balanced across target classes and attack objectives.

• Dependency: Assess how defense effectiveness depends on required assumptions or prerequisites, such as clean validation data, trusted reference samples, prior attack knowledge, or defense-specific hyperparameter tuning.

• Stability: Measure the variability of MTA, defended ASR, and UR across random seeds and repeated runs to ensure that reported protection is reproducible.

• Robustness: Evaluate defenses under diverse adversarial environments, including different numbers of parties and scenarios involving multiple attackers with shared or conflicting targets.

• Overhead: Report the computation, communication, memory, and inference costs introduced by the defense.

Together, these evaluation recipes provide a standardized and practice-oriented framework for assessing VFL backdoor attacks and defenses.

TABLE 6: BVBench compares attacks on the same VFL setting across realistic datasets for fairness and generating practical insights. Current attacks can compromise main task accuracy, and their attack success is not as eye-catching as reported.
<table><tr><td rowspan="2">Metric</td><td rowspan="2">Dataset</td><td rowspan="2">No Attack</td><td colspan="8">Backdoor Attacks</td></tr><tr><td>BackSplitVFL</td><td>BAEVFL</td><td>BadVFL</td><td>BadVFL*</td><td>HijackVFL</td><td>LFBA</td><td>VILLAIN</td><td>LMP PMP</td></tr><tr><td>Main</td><td>CIFAR-10</td><td>76.47</td><td>76.38</td><td>75.53</td><td>76.36</td><td>76.11</td><td>76.37</td><td>76.39</td><td>71.53 76.12</td><td>76.35</td></tr><tr><td rowspan="6">Task Accuracy (MTA)</td><td>Satellite</td><td>66.76</td><td>57.24</td><td>66.69</td><td>53.37</td><td>65.81</td><td>54.25</td><td>62.41</td><td>61.56</td><td>65.69</td><td>65.72</td></tr><tr><td>KUHAR</td><td>62.93</td><td>60.62</td><td>38.15</td><td>62.33</td><td>38.98</td><td>62.91</td><td>41.34</td><td>12.43</td><td>60.37</td><td>40.45</td></tr><tr><td>PTB-XL</td><td>43.93</td><td>40.35</td><td>42.55</td><td>41.61</td><td>40.87</td><td>42.16</td><td>39.65</td><td>26.15</td><td>42.73</td><td>40.02</td></tr><tr><td>Vehicle</td><td>83.45</td><td>83.57</td><td>84.19</td><td>83.66</td><td>83.67</td><td>83.78</td><td>83.75</td><td>84.27</td><td>83.71</td><td>83.64</td></tr><tr><td>NUSWIDE</td><td>78.42</td><td>78.08</td><td>78.14</td><td>78.19</td><td>78.00</td><td>78.09</td><td>78.75</td><td>78.23</td><td>78.69</td><td>78.01</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Attack</td><td>CIFAR-10</td><td></td><td>88.04</td><td>68.11 27.52</td><td>15.49 7.92</td><td>57.19 26.23</td><td>64.12 38.93</td><td>26.33 21.08</td><td>9.99 15.10</td><td>92.04</td><td>49.59</td></tr><tr><td>Success</td><td>Satellite KUHAR</td><td></td><td>100.00 72.11</td><td>5.56</td><td>5.31</td><td>6.76</td><td>7.82</td><td>7.86</td><td>0.04</td><td>69.35 36.36</td><td>29.84 42.46</td></tr><tr><td>Rate</td><td>PTB-XL</td><td></td><td>98.84</td><td>27.85</td><td>20.38</td><td>73.60</td><td>28.75</td><td>78.37</td><td>26.34</td><td>58.30</td><td>32.89</td></tr><tr><td>(ASR)</td><td>Vehicle</td><td></td><td>87.61</td><td>33.33</td><td>36.64</td><td>93.27</td><td>89.56</td><td>95.06</td><td>40.87</td><td>96.19</td><td>90.60</td></tr><tr><td></td><td>NUSWIDE</td><td></td><td>82.04</td><td>69.95</td><td>49.86</td><td>99.15</td><td>96.30</td><td>98.36</td><td>55.27</td><td>99.97</td><td>94.30</td></tr></table>

## 6. Empirical Analysis: Attacks

We empirically analyze existing attacks using BVBench. To the best of our knowledge, this is the first study to compare VFL backdoor attacks under a unified benchmark that enforces fairness, practicality, and comprehensive evaluation. Due to the space limit, we highlight key findings here, which reveal that many conclusions drawn from prior studies fail to generalize once these factors are controlled.

## 6.1. Efficacy

The conclusions derived from CIFAR-10, the unrealistic dataset commonly used in prior work, differ substantially from those obtained on realistic VFL datasets. As a prerequisite for qualifying as effective attacks, Table 6 reports the class-averaged MTA and ASR of all methods across datasets. Although all attacks preserve MTA on CIFAR-10, they can incur substantial utility degradation on realistic datasets, reducing MTA by as much as 13.39% on Satellite, 50.50% on KUHAR, and 17.78% on PTB-XL. The findings are even more concerning for ASR. First, despite prior studies often reporting ASR exceeding 80% on CIFAR-10, BadVFL, LFBA, and VILLAIN achieve less than 30% ASR under our evaluation, while most remaining methods fail to exceed 70%. This discrepancy stems from the fact that BVBench enforces configuration fairness: all methods are evaluated under the same VFL environment rather than under method-specific settings. Second, no existing attack demonstrates consistently strong performance across datasets. The gap between the best and worst cases can be substantial, with ASR varying by as much as 92.39% for BadVFL<sup>∗</sup>. Collectively, these results suggest that current VFL backdoor attacks are considerably less effective than previously implied. Robust attack efficacy across diverse VFL environments remains largely an open problem.

Per-Class Analysis. Existing studies typically report ASR for a selected target class, obscuring variations in attack effectiveness across classes. To expose this phenomenon, we repeat each attack on Satellite using different target classes. Figure 9 (color bars) presents the resulting per-class ASR. Almost all attacks achieve their highest ASR when targeting Class 2 (green), which corresponds to the majority class. This observation suggests that attack success is strongly influenced by class distribution and that reported performance can be sensitive to target selection.

![](images/451c029564bd2ba50f84bede442001802e0d18e2de80b538655336b36ec3520e.jpg)  
Figure 9: Existing attacks achieve high ASR primarily when targeting the majority class (Class 2), and this bias persists under both single-target and multi-target settings.

Multi-Target Analysis. This tendency becomes even more pronounced in the multi-target setting (hollow bars in Figure 9). Because multi-target learning is inherently more challenging, attacks appear to preferentially optimize shortcuts associated with majority classes when trade-offs arise. Consequently, ASR on minority classes can deteriorate drastically. For example, BackSplitVFL achieves 100% ASR on the majority class while completely failing on all others. Since multi-target capability is fundamental to the practical backdoor workflow (Section 3.2), improving the balance of effectiveness across targets represents an important direction for future research.

## 6.2. Dependency

Backdoor attacks rely on label knowledge, and the quality of such knowledge can critically influence attack success. Using LFBA, the best-performing attack in our benchmark and one that does not assume perfectly labeled samples as prior knowledge, as a case study, Figure 10 reports its ASR under different levels of manually controlled label quality (solid lines). Three observations emerge. First, perfect label knowledge does not necessarily guarantee successful attacks. Even when true labels are provided, LFBA still fails completely on Satellite. Second, on datasets where LFBA is effective (e.g., Vehicle), label quality becomes decisive: reducing label quality by only 20% can decrease ASR by more than 25%. Third, the label inference procedure itself exhibits strong dataset dependence. It performs well on some datasets (e.g., Vehicle) but fails on others (e.g., Satellite and PTB-XL), as indicated by the dashed lines. These findings suggest that label knowledge is neither sufficient nor uniformly obtainable in practice. While improving label inference remains important, future work should also investigate why attacks remain ineffective even when accurate label information is available.

![](images/14ed0c2f9014c167dd6b62caedd469ca2daf67ae40dcd8ba9bb0b59c94daadfc.jpg)  
Label Knowledge Quality (%)  
Figure 10: LFBA’s attack effectiveness is highly sensitive to label knowledge quality (solid lines), while its label inference component exhibits significant variability across datasets (dashed lines).

## 6.3. Stability Under Randomness

Effective attacks should exhibit consistent performance across repeated executions. However, existing methods can be highly sensitive to randomness. An attack may perform well during one epoch but fail in the next, or succeed under one random seed while collapsing under another. Figure 11 illustrates a representative example using BAEVFL on Vehicle. Its ASR drops from 100% to 0% within two consecutive epochs and recovers to 100% only two epochs later. Changing the random seed introduces further variability. Such instability raises concerns regarding the reliability of current evaluations. An attack may appear successful simply because training happens to terminate at a favorable point along the optimization trajectory. Therefore, future evaluations should systematically assess sensitivity to randomness and demonstrate that attack effectiveness remains stable across both epochs and random seeds.

## 6.4. Robustness to Adversarial Environment

Backdoor attacks should remain effective regardless of how many peers participate in the VFL system and whether those peers behave benignly or maliciously.

The number of peers affects the fraction of top-model inputs controlled by the adversary. Using Satellite as a case study, Figure 12 shows that ASR is strongly influenced by the number of participating parties. Embeddingspace attacks (solid lines) consistently outperform inputspace attacks (dashed lines), regardless of whether the attacker controls 50% of the top-model inputs (two parties) or only 6.25% (sixteen parties). A plausible explanation is that input-space triggers must propagate through the bottom model before influencing the top model, weakening their effect, whereas embedding-space attacks directly manipulate the exchanged representations and bypass this attenuation.

![](images/c20bf7e7c5ab1c6718dfcaa4f7f760e9250cc757b791c9d821d4c2673c1ee787.jpg)  
Figure 11: BAEVFL’s ASR on Vehicle fluctuates substantially across epochs and is highly sensitive to randomness, highlighting the importance of stability and reproducibility for meaningful attack evaluation.

![](images/18204bbb8e5655f3dce3a1a6deac5bad2dbe8fbe291af9baf35f2b81a566b617.jpg)  
Figure 12: Embedding-space attacks (solid lines) tend to perform better than input-space attacks (dashed lines) on Satellite, no matter the attacker controls more or less features to the top model.

Another overlooked aspect in the adversarial environment concerns the presence of multiple malicious parties. Using NUSWIDE as a case study, Figure 13 compares the single-attacker setting (solid lines) with two-attacker settings involving either shared targets (green bars) or conflicting targets (red bars). Across attacks, coordinated attackers tar geting the same class substantially improve ASR relative to the single-attacker baseline. In contrast, independent attackers pursuing different targets interfere with one another and significantly degrade attack effectiveness.

These observations suggest that current evaluations underestimate the role of interactions among participants. Understanding how attacks behave under realistic mixtures of benign and adversarial peers remains an important open question.

## 6.5. Stealthiness

Train-Time Stealthiness. Trigger-injected samples should remain indistinguishable from benign samples throughout training and inference. Although BackSplitVFL and VILLAIN explicitly constrain embeddings to improve stealthiness, the resulting malicious embeddings remain readily distinguishable under the stealthiness score introduced in Section 5. Figure 14 illustrates the distribution of stealthiness scores for benign and malicious embeddings generated by BackSplitVFL. The resulting bipolar distribution clearly exposes the presence of poisoning, while the consistently low scores of malicious embeddings can even reveal which samples are poisoned. Consequently, the active party may identify malicious participants during training. This separability is positively correlated with attack effectiveness. Figure 15 reports the ROC-AUC between malicious and benign samples. When ASR is high, the ROC-AUC of stealthiness metrics approaches 1.0, indicating near-perfect detectability. In contrast, when attacks are largely ineffective (ASR <20%), the ROC-AUC drops to approximately 0.6– 0.7. Under our simple and intuitive stealthiness metrics, current attacks therefore fail to achieve a favorable balance between efficacy and stealthiness.

![](images/5d9492a1addfb07f01cd1acc2d8591e26f1460a07201f77a54c6e896c7d0faad.jpg)  
Figure 13: Compared with the single-attacker baseline (solid lines), multiple colluding attackers (green) with consistent targets can reinforce attacks, but independent attackers with conflicting targets (red) can interfere with each other.

![](images/1a387293fd786007d30805694e3970796e1d148bcb1d8d7469abf853980616b4.jpg)  
Figure 14: Distribution of stealthiness for clean samples and attack samples. Most clean samples stay in a high level while attacked samples are at a significantly low stealthiness level.

Test-Time Stealthiness. Stealthiness scores can also be leveraged during deployment, where the active party lacks ground-truth labels but seeks to identify suspicious queries. As shown in Figure 16, in this label-free setting, we observe strong agreement with the oracle setting that requires labels and is more suitable for training-time detection. In particular, the correlation between label-free and oracle ROC-AUC can reach 0.98. This finding suggests that label-free stealthiness estimation can serve as a practical approximation for both attackers and defenders.

![](images/2061c22a19e68c2cccc45aa3a738e41f08bb0987054b4898f020998f372f056c.jpg)  
Figure 15: When an attack is effective, the ROC-AUC can be close to 1.0, suggesting a strong separation between clean samples and attack samples. However, when an attack fails, the ROC-AUC can drop to 0.6-0.7.

![](images/5a8e0e00d5537ca7380d4f7b692820f7b0bc9e3a1f0a56c8156dd86a252020ad.jpg)  
Figure 16: The oracle mode and the label-free mode of our stealthiness metric can both offer strong detectability for effective attacks.

Overall, our results indicate that current VFL backdoor attacks are not stealthy from the defender’s perspective. Effective attacks tend to be highly anomalous, whereas stealthier attacks are often ineffective. Whether future methods can achieve a substantially better efficacy-stealthiness trade-off remains an open question. Otherwise, practical VFL backdoor attacks may be considerably easier to detect than previously assumed.

## 7. Empirical Analysis: Defenses

By systematically comparing VFL backdoor defenses under a unified benchmark, BVBench also reveals previously overlooked limitations of existing methods. We summarize the key findings below.

## 7.1. Efficacy

Current defenses fail to simultaneously preserve utility, suppress attacks, and recover correct predictions. Defense efficacy should be evaluated along three dimensions: maintaining main task accuracy (MTA) in the absence of attacks, reducing the attack success rate (ASR) of triggerinjected queries, and achieving utility recovery (UR), i.e., correctly classifying those trigger-injected queries after mitigation. Table 7 summarizes the performance of all defenses across datasets, averaged over all attacks and target classes. Three observations emerge. First, existing defenses exhibit a serious utility-security trade-off. Methods that effectively suppress attacks often incur substantial degradation in clean performance, whereas methods that preserve utility typically provide only limited protection. For example, VFLIP achieves the largest ASR reduction on KUHAR, lowering ASR from 20.48% to 6.18%, but reduces MTA to only 0.31%. In contrast, GBD best preserves utility, with MTA decreasing only from 62.81% to 59.98%, yet its defended ASR remains at 15.23%. Similar trade-offs are observed across other defenses and datasets. Second, no existing defense consistently achieves strong performance across datasets. Similar to attacks, defense effectiveness is highly dataset-dependent. A method that performs well in one setting may fail completely in another, suggesting that current defenses do not generalize reliably across realistic VFL environments. Third, existing defenses struggle to recover correct predictions after mitigating attacks. Ideally, UR should approach MTA, indicating that the defense not only suppresses targeted misclassification but also restores the model’s intended functionality. However, Table 7 shows that no defense achieves this objective, including methods explicitly designed for recovery. For example, although VFLMonitor reduces ASR from 73.68% to 29.03% on Vehicle, its UR reaches only 31.79%, far below its MTA of 62.52%. Similarly, VFLIP lowers ASR to 33.62%, yet its UR remains substantially lower than MTA (21.30% versus 83.70%). These results indicate that disrupting targeted attacks alone is insufficient, as defended samples often remain misclassified. A plausible explanation is the limited reliability of the anomaly detection mechanisms underlying repair-based defenses. Figure 17 examines the detection performance of representative methods. VFLMonitor fails to simultaneously achieve a high true positive rate (TPR) and a low false positive rate (FPR), while VFLIP exhibits strong detection capability on some datasets (e.g., Vehicle) but poor generalization to others (e.g., PTB-XL). Overall, future defenses should place greater emphasis on utility recovery, as preserving prediction correctness is ultimately more important than merely reducing ASR.

TABLE 7: No existing defense can simultaneously preserve benign utility, suppress attacks, and recover correct predictions on attacked samples. ASR, MTA, and UR are averaged over all attacks and target classes.
<table><tr><td rowspan="2">Metric</td><td rowspan="2">Dataset</td><td rowspan="2">No Defense</td><td colspan="4">(a) VFL-specific Backdoor Defenses</td><td colspan="4">(b) Generic Backdoor Defenses</td></tr><tr><td>VFLIP</td><td>VFLMonitor</td><td>UBD</td><td>GBD</td><td>Grad. Prun.</td><td>Grad. Noise</td><td>Emb. Norm.</td><td>Emb. Drop.</td></tr><tr><td>Main</td><td>Satellite</td><td>63.73</td><td>18.73</td><td>56.17</td><td>66.14</td><td>60.85</td><td>60.30</td><td>40.98</td><td>54.70</td><td>68.09</td></tr><tr><td rowspan="4">Task Accuracy (MTA)</td><td>KUHAR</td><td>62.81</td><td>0.31</td><td>26.17</td><td>50.58</td><td>59.98</td><td>58.87</td><td>40.94</td><td>57.53</td><td>26.96</td></tr><tr><td>PTB-XL</td><td>41.86</td><td>5.12</td><td>25.40</td><td>39.54</td><td>40.51</td><td>35.58</td><td>14.24</td><td>42.56</td><td>46.96</td></tr><tr><td>Vehicle</td><td>83.70</td><td>22.44</td><td>62.52</td><td>83.13</td><td>81.10</td><td>84.69</td><td>82.71</td><td>84.66</td><td>84.57</td></tr><tr><td>NUSWIDE</td><td>78.08</td><td>41.94</td><td>76.74</td><td>76.86</td><td>74.02</td><td>78.71</td><td>75.93</td><td>76.73</td><td>78.65</td></tr><tr><td></td><td>Satellite</td><td>37.33</td><td>30.56</td><td>30.66</td><td>36.25</td><td>21.24</td><td>34.57</td><td>33.62</td><td>32.90</td><td>35.75</td></tr><tr><td>Attack Success</td><td>KUHAR</td><td>20.48</td><td>6.18</td><td>7.11</td><td>21.50</td><td>15.23</td><td>20.11</td><td>15.68</td><td>13.58</td><td>17.17</td></tr><tr><td>Rate</td><td>PTB-XL</td><td>49.48</td><td>24.44</td><td>25.45</td><td>50.01</td><td>30.05</td><td>39.28</td><td>24.44</td><td>48.15</td><td>44.84</td></tr><tr><td>(ASR)</td><td>Vehicle</td><td>73.68</td><td>33.62</td><td>29.03</td><td>72.05</td><td>37.61</td><td>67.31</td><td>57.22</td><td>73.75</td><td>75.41</td></tr><tr><td rowspan="5">Utility Recovery</td><td>NUSWIDE</td><td>82.80</td><td>50.00</td><td>60.11</td><td>72.46</td><td>71.76</td><td>79.07</td><td>54.42</td><td>54.02</td><td>78.42</td></tr><tr><td>Satellite</td><td>48.16</td><td>17.64</td><td>49.78</td><td>41.92</td><td>43.15</td><td>47.81</td><td>32.43</td><td>44.86</td><td>49.10</td></tr><tr><td>KUHAR</td><td>27.20</td><td>0.32</td><td>16.18</td><td>28.45</td><td>26.01</td><td>27.32</td><td>22.16</td><td>37.15</td><td>17.92</td></tr><tr><td>PTB-XL</td><td>27.80</td><td>6.75</td><td>18.61</td><td>23.59</td><td>26.28</td><td>27.99</td><td>13.47</td><td>29.94</td><td>34.63</td></tr><tr><td>Vehicle NUSWIDE</td><td>35.66 49.36</td><td>21.30 41.94</td><td>31.79 73.24</td><td>31.56 59.03</td><td>32.06 47.85</td><td>39.68 53.22</td><td>53.69 71.56</td><td>42.50 68.31</td><td>41.99 54.84</td></tr></table>

Per-Class Analysis. Existing studies report classaveraged metrics, which obscure variations in protection across target classes. To expose this phenomenon, Figure 18 reports defended ASR for different target classes on the same dataset. Compared with the undefended baseline (solid lines), most defenses fail to consistently suppress attacks across classes. Interestingly, VFLIP reduces the ASR of Classes 0, 1, and 3 to below 10%, yet strengthens the attack on the majority class (Class 2). Similar trends are observed for gradient noise and embedding normalization. These results suggest that current defenses provide uneven protection and may leave certain classes substantially more vulnerable than others.

![](images/298fdc66f0074e192abaece8833251ff3693bdf6ba78837cfb954ae84dbaabb6.jpg)  
Figure 17: Existing detection-based defenses fail to reliably distinguish attacked samples, preventing them from achieving both high TPR and low FPR across datasets.

![](images/151aae7f39b77141d2d34166243d999e075299ab1ee531fd4378e45629724550.jpg)  
Figure 18: Most defenses fail to provide consistent protection across target classes on Satellite. ASR under defense may even surpass the scenario with no defense (solid lines).

Multi-Target Analysis. The limitations of current defenses become even more pronounced in the multi-target setting. Figure 19 shows that the defended ASR of most methods remains close to the single-target baseline. Moreover, the influence of class imbalance becomes stronger, with defenses becoming less effective at protecting majority classes from attack. Most defenses exhibit similar robustness under both single-target and multi-target settings despite the latter posing a substantially stronger threat. Since existing methods lack mechanisms for balancing defense effectiveness across classes or attack objectives, improving robustness against multi-target attacks remains an important challenge for future research.

![](images/068edf901611c5448396dcf3d85297b40c10ee256f55e4d3512afce4147bc67a.jpg)  
Figure 19: Multi-target attacks further expose the limitations of current defenses: most methods fail to reduce ASR across classes, and the majority-class effect becomes even more pronounced.

## 7.2. Robustness to Adversarial Environment

Current defenses are fragile under realistic multiadversary environments. Figure 20 reports defended ASR (bars) and undefended ASR (solid lines) under two scenarios: (i) multiple attackers sharing the same target and (ii) multiple attackers pursuing conflicting targets. We report the ASRs of one of the attackers. Two observations emerge. First, existing defenses provide only limited protection against colluding attackers. Across methods, the defended ASR under shared-target attacks (green bars) remains close to the undefended baseline, indicating that current defenses struggle to mitigate coordinated adversarial behavior. Second, defenses can inadvertently strengthen attacks by redistributing protection unevenly across targets. For example, VFLIP may successfully suppress one adversary’s backdoor while simultaneously enabling another adversary to achieve higher attack success. This phenomenon is particularly concerning because it remains hidden when evaluation focuses solely on aggregate metrics. These findings suggest that current defenses are not robust to complex adversarial environments involving multiple malicious participants. Future defense designs should explicitly account for interactions among attackers and ensure balanced protection across target classes and attack objectives.

## 8. Discussions

By systematizing existing work and completing overlooked practicality dimensions through BVBench, we show that current understanding of VFL backdoor vulnerability requires substantial revision. Under realistic threat models, standardized evaluation, and comprehensive criteria, most

![](images/1a123d3234265765cd5226f9a65794075bd4c1d9262c36babefda492c3c90433.jpg)  
Figure 20: Current defenses are not robust to multiadversary settings. Compared with the ASR under no defense (solid lines), they provide only marginal protection against colluding attackers and can even inadvertently facilitate attacks when adversaries target different classes.

VFL backdoor attacks and defenses are far less practical than previously reported.

On the attack side, no existing attack remains practical, because they all rely on inaccessible task-related knowledge. Even with such knowledge, their effectiveness is still limited by strong dependency, instability, multi-target incompatibility, poor stealthiness, and sensitivity to the VFL environment. Current attacks are hard to justify under realistic assumptions and difficult to sustain in practice.

This does not mean VFL systems are already well protected against vulnerability. Current defenses rely on unreasonable knowledge and fail to consistently achieve all three defense objectives. These limitations worsen in multitarget and multi-adversary settings, where protection can be uneven across classes and fragile to the attack composition.

Our findings carry useful insights for different stakeholders. Future work on attacks and defenses should follow the practical threat model and workflow while meeting practical requirements, such as attack stealthiness and defense utility recovery. Practitioners who want to deploy backdoorresilient VFL systems should rely on fair and comprehensive evaluation results under BVBench rather than heterogeneous reported numbers.

Our study is not exhaustive. We focus on neuralnetwork-based VFL and single-label classification, while real-world VFL may involve other tasks or protocols [43], [44], [45], [46]. Besides, this work aims to recalibrate the understanding of practical VFL backdoor vulnerabilities rather than propose a new practical attack or defense. We hope BVBench can serve as a useful tool to steer future VFL backdoor research toward a more practical direction.

## 9. Conclusions

Current VFL backdoor research has made significant progress, but its practicality has not been fairly and fully evaluated. We find that the practicality gaps lie not only in methodology designs, but also in evaluation designs. Our findings through systematization and empirical analysis from BVBench suggest that current understanding of VFL backdoor vulnerability is incomplete and unreliable, and should be calibrated under the practical threat model and evaluation recipe to guide future research.

## References

[1] F. Zheng, Erihe, K. Li, J. Tian, and X. Xiang, “A vertical federated learning method for interpretable scorecard and its application in credit scoring,” 2020.

[2] P. Vepakomma, O. Gupta, T. Swedish, and R. Raskar, “Split learning for health: Distributed deep learning without sharing raw patient data,” arXiv preprint arXiv:1812.00564, 2018.

[3] C.-j. Huang, L. Wang, and X. Han, “Vertical federated knowledge transfer via representation distillation for healthcare collaboration networks,” in Proceedings of the ACM Web Conference 2023, 2023, pp. 4188–4199.

[4] European Parliament and Council of the European Union, “Regulation (EU) 2016/679 of the European Parliament and of the Council (GDPR),” 2016. [Online]. Available: https://data.europa.eu/eli/reg/ 2016/679/oj

[5] California Legislature, “California consumer privacy act of 2018, Cal. Civ. Code § 1798.100 et seq.” 2018, amended by the California Privacy Rights Act (CPRA) of 2020.

[6] Y. Liu, Y. Kang, T. Zou, Y. Pu, Y. He, X. Ye, Y. Ouyang, Y.- Q. Zhang, and Q. Yang, “Vertical Federated Learning: Concepts, Advances, and Challenges,” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 7, pp. 3615–3634, Jul. 2024.

[7] M. Ye, W. Shen, B. Du, E. Snezhko, V. Kovalev, and P. C. Yuen, “Vertical Federated Learning for Effectiveness, Security, Applicability: A Survey,” ACM Computing Surveys, vol. 57, no. 9, pp. 1–32, Sep. 2025.

[8] A. Khan, M. ten Thij, and A. Wilbik, “Vertical federated learning: A structured literature review,” vol. 67, no. 4, pp. 3205–3243.

[9] Y. Xuan, X. Chen, Z. Zhao, B. Tang, and Y. Dong, “Practical and General Backdoor Attacks Against Vertical Federated Learning,” in Machine Learning and Knowledge Discovery in Databases: Research Track, D. Koutra, C. Plant, M. Gomez Rodriguez, E. Baralis, and F. Bonchi, Eds., vol. 14170. Cham: Springer Nature Switzerland, 2023, pp. 402–417.

[10] Y. Bai, Y. Chen, H. Zhang, W. Xu, H. Weng, and D. Goodman, “VILLAIN: Backdoor Attacks Against Vertical Split Learning,” in 32nd USENIX Security Symposium (USENIX Security 23), 2023, pp. 2743–2760.

[11] J. Liu, X. Lyu, C. Ren, and Q. Cui, “Targeted Poisoning Attacks Against Vertical Federated Learning via Embedding Manipulation,” IEEE Transactions on Dependable and Secure Computing, vol. 22, no. 6, pp. 7535–7551, Nov. 2025.

[12] Y. Liu, Y. Lou, Y. Liu, Y. Cao, and H. Wang, “Label leakage in vertical federated learning: A survey,” in Thirty-Third International Joint Conference on Artificial Intelligence, vol. 9, pp. 8160–8169.

[13] Y. Yan, H. Wang, Y. Huang, N. He, L. Zhu, Y. Xu, Y. Li, and Y. Zheng, “Cross-modal vertical federated learning for MRI reconstruction,” IEEE Journal of Biomedical and Health Informatics, vol. 28, no. 11, pp. 6384–6394, Nov. 2024.

[14] W. Shen, W. Liu, M. Chen, W. Huang, and M. Ye, “MARS-VFL: A unified benchmark for vertical federated learning with realistic evaluation,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, Oct. 2025.

[15] P. Wei, H. Dou, S. Liu, R. Tang, L. Liu, L. Wang, and B. Zheng, “FedAds: A benchmark for privacy-preserving CVR estimation with vertical federated learning,” in Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, ser. SIGIR ’23. Association for Computing Machinery, pp. 3037–3046.

[16] G. Wang, B. Gu, Q. Zhang, X. Li, B. Wang, and C. X. Ling, “A unified solution for privacy and communication efficiency in vertical federated learning,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[17] “ADI: Adversarial dominating inputs in vertical federated learning systems,” in 2023 IEEE Symposium on Security and Privacy (SP). IEEE Computer Society, May 2023, pp. 1875–1892.

[18] S. Yuan, X. Li, X. Cao, H. Zhang, and R. H. Deng, “General Test-Time Backdoor Detection in Split Neural Network-Based Vertical Federated Learning,” IEEE Transactions on Dependable and Secure Computing, vol. 22, no. 6, pp. 7157–7171, Nov. 2025.

[19] B. Pinkas, T. Schneider, and M. Zohner, “Scalable private set intersection based on OT extension,” in ACM Transactions on Privacy and Security (TOPS), vol. 21, no. 2. ACM New York, NY, USA, 2018, pp. 1–35.

[20] Y. Yang, X. Chen, Y. Pan, J. Shen, Z. Cao, X. Dong, X. Li, J. Sun, G. Yang, and R. Deng, “OpenVFL: A vertical federated learning framework with stronger privacy-preserving,” vol. 19, pp. 9670–9681.

[21] Y. Xi, Y. Guo, S. Xu, C. Cai, and X. Jia, “Private sample alignment for vertical federated learning: An efficient and reliable realization,” vol. 20, pp. 3834–3848.

[22] T. Gu, B. Dolan-Gavitt, and S. Garg, “BadNets: Identifying vulnerabilities in the machine learning model supply chain,” arXiv preprint arXiv:1708.06733, 2017.

[23] X. Chen, C. Liu, B. Li, K. Lu, and D. Song, “Targeted backdoor attacks on deep learning systems using data poisoning,” in arXiv preprint arXiv:1712.05526, 2017.

[24] C. Fu, X. Zhang, S. Ji, J. Chen, J. Wu, S. Guo, J. Zhou, A. X. Liu, and T. Wang, “Label Inference Attacks Against Vertical Federated Learning,” in 31st USENIX Security Symposium (USENIX Security 22), 2022, pp. 1397–1414.

[25] D. Gao, S. Wan, H. Gu, L. Fan, X. Yao, and Q. Yang, “Label Privacy Source Coding in Vertical Federated Learning,” in Machine Learning and Knowledge Discovery in Databases. Research Track, A. Bifet, J. Davis, T. Krilavicius, M. Kull, E. Ntoutsi, and I.ˇ Zliobait<sup>ˇ</sup> e, Eds.,˙ vol. 14941. Cham: Springer Nature Switzerland, 2024, pp. 313–331.

[26] B. Wang, Y. Yao, S. Shan, H. Li, B. Viswanath, H. Zheng, and B. Y. Zhao, “Neural cleanse: Identifying and mitigating backdoor attacks in neural networks,” in 2019 IEEE Symposium on Security and Privacy (SP). IEEE, 2019, pp. 707–723.

[27] B. Tran, J. Li, and A. Madry, “Spectral signatures in backdoor attacks,” in Advances in Neural Information Processing Systems, vol. 31, 2018.

[28] K. Liu, B. Dolan-Gavitt, and S. Garg, “Fine-pruning: Defending against backdooring attacks on deep neural networks,” in Research in Attacks, Intrusions, and Defenses (RAID), 2018, pp. 273–294.

[29] T. D. Nguyen, P. Rieger, H. Chen, H. Yalame, H. Mollering,¨ H. Fereidooni, S. Marchal, M. Miettinen, A. Mirhoseini, S. Zeitouni, F. Koushanfar, A.-R. Sadeghi, and T. Schneider, “FLAME: Taming backdoors in federated learning,” in 31st USENIX Security Symposium (USENIX Security 22). USENIX Association, 2022, pp. 1415–1432.

[30] C. Fung, C. J. M. Yoon, and I. Beschastnikh, “The limitations of federated learning in sybil settings,” in Proceedings of the 23rd International Symposium on Research in Attacks, Intrusions and Defenses. USENIX Association, 2020, pp. 301–316.

[31] Y. He, Z. Shen, J. Hua, Q. Dong, J. Niu, W. Tong, X. Huang, C. Li, and S. Zhong, “Backdoor Attack Against Split Neural Network-Based Vertical Federated Learning,” IEEE Transactions on Information Forensics and Security, vol. 19, pp. 748–763, 2024.

[32] W. Shen, W. Huang, G. Wan, and M. Ye, “Label-Free Backdoor Attacks in Vertical Federated Learning,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 19, pp. 20 389– 20 397, Apr. 2025.

[33] M. Naseri, Y. Han, and E. De Cristofaro, “BadVFL: Backdoor Attacks in Vertical Federated Learning,” in 2024 IEEE Symposium on Security and Privacy (SP), May 2024, pp. 2013–2028.

[34] P. Qiu, X. Zhang, S. Ji, C. Li, Y. Pu, X. Yang, and T. Wang, “Hijack Vertical Federated Learning Models As One Party,” IEEE Transactions on Dependable and Secure Computing, pp. 1–18, 2024.

[35] J. Chen, D. Chen, J. Cui, and H. Zhong, “Backdoor Attack on Encryption-Protected Vertical Federated Learning,” IEEE Transactions on Information Forensics and Security, vol. 20, pp. 6968–6983, 2025.

[36] X. Xu, Y. Zhao, Y. Han, Y. Zhu, Z. Han, G. Xu, B. Wang, S. Ji, and W. Wang, “VFLMonitor: Defending One-Party Hijacking Attacks in Vertical Federated Learning,” IEEE Transactions on Information Forensics and Security, vol. 20, pp. 4828–4843, 2025.

[37] Y. Cho, W. Han, M. Yu, Y. Lee, H. Bae, and Y. Paek, “VFLIP: A Backdoor Defense for Vertical Federated Learning via Identification and Purification,” in European Symposium on Research in Computer Security, J. Garcia-Alfaro, R. Kozik, M. Choras, and S. Katsikas,´ Eds., vol. 14985. Cham: Springer Nature Switzerland, 2024, pp. 291–312.

[38] P. Chen, H. Xiang, X. Du, X. Xu, X. Jiang, Z. Lu, J. Yang, Q. Duan, and W. Dou, “Universal Backdoor Defense via Label Consistency in Vertical Federated Learning,” in Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, vol. 1, Sep. 2025, pp. 4743–4751.

[39] B. Wu, H. Chen, M. Zhang, Z. Zhu, S. Wei, D. Yuan, and C. Shen, “Backdoorbench: A comprehensive benchmark of backdoor learning,” in Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022.

[40] Z. Wu, J. Hou, and B. He, “VertiBench: Advancing Feature Distribution Diversity in Vertical Federated Learning Benchmarks,” in The Twelfth International Conference on Learning Representations, 2024.

[41] T. Zou, Z. Gu, Y. He, H. Takahashi, Y. Liu, and Y.-Q. Zhang, “VFLAIR: A Research Library and Benchmark for Vertical Federated Learning,” in The Twelfth International Conference on Learning Representations, Oct. 2023.

[42] A. Krizhevsky, G. Hinton et al., “Learning multiple layers of features from tiny images,” 2009.

[43] Y. Wu, S. Cai, X. Xiao, G. Chen, and B. C. Ooi, “Privacy preserving vertical federated learning for tree-based models,” Proceedings of the VLDB Endowment, vol. 13, no. 12, pp. 2090–2103, 2020.

[44] B. Qian, Y. Xie, Y. Li, B. Ding, and J. Zhou, “Tree-based models for vertical federated learning: A survey,” vol. 57, no. 9, pp. 241:1– 241:30.

[45] C. Chen, J. Zhou, L. Zheng, H. Wu, L. Lyu, J. Wu, B. Wu, Z. Liu, L. Wang, and X. Zheng, “Vertically federated graph neural network for privacy-preserving node classification,” in Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, L. D. Raedt, Ed. International Joint Conferences on Artificial Intelligence Organization, 7 2022, pp. 1959–1965, main Track.

[46] G. Wang, Q. Li, X. Liu, X. Yan, Q. Dong, H. Wu, X. Kong, and L. Zhou, “VFGCN: A vertical federated learning framework with privacy preserving for graph convolutional network,” vol. 22, no. 5, pp. 5617–5631.

## Appendix

## 1. Benchmark Setup and Evaluation Details

1.1. Evaluation Metrics. We report standardized metrics for both attacks and defenses. Let $\mathcal { V } = \{ 1 , \ldots , C \}$ denote the label set, let $y _ { t }$ be the target label of a targeted backdoor attack, and let $\mathcal { D } _ { c }$ denote the clean test subset whose groundtruth label is $c .$ We write $\mathcal { F } ^ { T } ( \mathcal { E } _ { \bf x } )$ for the prediction on a clean sample x and $\mathcal { F } ^ { T } ( \tilde { \mathcal { E } } _ { x } )$ for the prediction when the trigger is injected. Considering that realistic VFL datasets are often class-imbalanced, we use macro-averaged metrics by default so that the majority class does not dominate the calculation.

Main Task Accuracy (MTA). We use macro-F1 on the clean test set as the main-task utility metric:

$$
\mathrm { M T A } = \frac { 1 } { C } \sum _ { c \in \mathcal { Y } } \mathrm { F } 1 _ { c } .\tag{1}
$$

This definition differs from the conventional metric used in the literature. Our goal is to provide a more practical evaluation on model utility across all classes rather than mainly on majority classes.

Attack Success Rate (ASR). We also define ASR as the macro-averaged targeted success rate over all source classes:

$$
\mathrm { A S R } = \frac { 1 } { C } \sum _ { c \in \mathcal { V } } \frac { 1 } { | \mathscr { D } _ { c } | } \sum _ { \pmb { x } \in \mathscr { D } _ { c } } \mathbf { 1 } [ \mathcal { F } ^ { T } ( \tilde { \mathscr { E } } _ { \pmb { x } } ) = t ] .\tag{2}
$$

where t is the target label, $\tilde { \mathcal { E } } _ { x }$ is the poisoned embedding of sample x. We think that this definition prevents biased evaluation on the majority class, as in the conventional definition the attack difficulty on the majority class could significantly skew the overall ASR. Our definition therefore better reflects whether an attack can consistently redirect predictions from all classes to the attacker-chosen target.

Label Inference Accuracy. For attacks that require label inference before poisoning, we explicitly denote the inferred label of sample x by ${ \hat { y } } _ { x } .$ Let $\mathcal { D } ^ { \mathrm { i n } \hat { \mathrm { f } } }$ be the set of samples on which the attacker uses for further poisoning. We evaluate label inference quality using macro-F1, which is aligned with MTA and ASR:

$$
\mathrm { L I A } = \frac { 1 } { C } \sum _ { c \in \mathcal { V } } \mathrm { F } \boldsymbol { 1 } _ { c } ^ { \mathrm { i n f } } ,\tag{3}
$$

where $\mathrm { F } 1 _ { c } ^ { \mathrm { i n f } }$ is the F1 score of class c computed from the inferred labels $\{ \hat { y } _ { \pmb { x } } | \textbf { \em x } \in \mathbf { \nabla } \mathcal { D } ^ { \mathrm { i n f } } \}$ and the corresponding ground-truth labels $\{ y _ { \pmb { x } } | \pmb { x } \in \mathcal { D } ^ { \mathrm { i n f } } \}$

Stealthiness. Intuitively, a stealthy attacked sample should still look normal under the clean embedding distribution. We consider a simple metric: the $L _ { \infty }$ norm of the embedding. We measure whether the attack pushes a sample into a region with unusually large embedding norm or distance. To be specific, for a distance $r ,$ we collect its per-party distribution from training samples, then map each query sample to a scalar value $r ( { \pmb x } )$ using the CDF of training samples. We think that samples with close-toone CDF values are more likely to be malicious, while samples whose CDF are small enough are benign. Then the stealthiness score of the distance measure r is then defined as:

$$
S ( { \pmb x } ; C D F , r ) = \operatorname* { m i n } ( 1 , 2 ( 1 - C D F ( r ( { \pmb x } ) ) ) ) .\tag{4}
$$

This score only penalizes samples whose CDF values are more than 0.5. Then we apply this calculation to L norm to form the stealthiness metrics:

$$
r _ { \infty } ( \pmb { x } ) = \| \pmb { \mathcal { E } } _ { \pmb { x } } \| _ { \infty }\tag{5}
$$

as the $L _ { \infty }$ norm of the embedding of sample x.

We provide two ways to collect the reference distribution and evaluate on test samples: oracle mode and labelfree mode. In oracle mode, we assume that labels of test samples are known, and each party builds class-conditional $\mathrm { C D } \bar { \mathrm { F s } } \ C D F _ { \infty , c }$ from training samples in class $c .$ Therefore, stealthiness scores of test samples can be evaluated from their per-class reference distribution. In label-free mode, we assume that labels of test samples are unknown, and each party builds a single class-agnostic CDF $F _ { \infty }$ from all training samples. In this case, stealthiness scores of test samples are evaluated from the global reference distribution instead.

Overall, oracle mode uses class-conditional reference distributions and test labels, whereas label-free mode uses class-agnostic reference distributions and does not require test labels. For both modes, we report the per-sample stealthiness scores of clean and attack test samples in the form of histograms. We think that the oracle mode provides the most effective evaluation of stealthiness, but in reality test labels are often unavailable, so the label-free mode is more practical.

Utility Recovery (UR). For defenses that aim to recover correct predictions on attacked samples, we report UR as the macro-F1 of the defended model on the attacked test set:

$$
\mathrm { U R } = \frac { 1 } { C } \sum _ { c \in \mathcal { V } } \mathrm { F } \boldsymbol { 1 } _ { c } ^ { \mathrm { d e f } } ,\tag{6}
$$

where $\mathrm { F } 1 _ { c } ^ { \mathrm { d e f } }$ is the F1 score of class c computed from the defended model’s predictions on the attacked test set and the corresponding ground-truth labels. A high UR indicates that the defense can restore the model’s intended functionality on attacked samples.

Detection Performance. For defenses that involve testtime detection, we report the true positive rate (TPR) and false positive rate (FPR) of their detection results:

$$
\mathrm { T P R } = \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } } , \qquad \mathrm { F P R } = \frac { \mathrm { F P } } { \mathrm { F P } + \mathrm { T N } } .\tag{7}
$$

A high TPR and a low FPR indicate good detection performance, which is an important aspect of defense efficacy.

1.2. Default Configurations. We summarize the training defaults and the dataset-specific configuration settings used in BVBench in Table 8 and Table 9, respectively. Across all experiments, we keep the same training defaults whenever possible, since in practice, the active party cannot tune the hyperparameters for each dataset. In particular, we use the same training epochs, optimizer with a fixed learning rate and weight decay, and learning-rate schedule across all settings, and reuse the same default VFL architectures for datasets of the same modality whenever possible. Datasetspecific adjustments are restricted to a small set of necessary factors, such as the batch size, loss function, embedding dimension, and the width of the hidden dimensions of the top model, etc.

For attacks and defenses, we mainly follow their default settings in original papers. Some methods have different sets of hyperparameter choices for different datasets and they do not cover the datasets in BVBench, so we reuse the hyperparameters for CIFAR10 and apply to all datasets. For hyperparameters not explicitly specified in the paper or source code, we perform a small grid search on CIFAR10 to find the best values, and then fix them across all datasets.

TABLE 8: Shared training defaults in BVBench.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Epochs</td><td>50</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td>1e-3</td></tr><tr><td>Weight decay</td><td>1e-5</td></tr><tr><td>LR scheduler</td><td>Constant</td></tr><tr><td>Data augmentation</td><td>No</td></tr></table>

1.3. Extension to Multi-Target Setting. To extend singletarget attacks to the multi-target setting, we preserve each method’s original design as much as possible and only replace the single target with all classes in the label space. Target-specific prior knowledge is expanded to all target classes without increasing the total number of known labeled samples, and known labels will be proportionally distributed across all classes. This is to simulate realistic difficulties to obtain those labels. For attacks with class-conditioned triggers, this results in one perturbation or trigger module per class. For attacks with fixed class-agnostic triggers, we keep the trigger pattern unchanged and instead assign nonoverlapping trigger positions to different classes whenever possible.

TABLE 9: Dataset-specific scenario configurations in BVBench.
<table><tr><td>Dataset</td><td>Modality</td><td>#Parties</td><td>Bottom Model</td><td>Embedding Dim.</td><td>Top Model</td><td>BS</td><td>Loss</td></tr><tr><td>CIFAR-10</td><td>Image</td><td>2</td><td>ResNet18</td><td>512</td><td>2-layer MLP (1000)</td><td>128</td><td>CE</td></tr><tr><td>Satellite</td><td>Image</td><td>16</td><td>ResNet18</td><td>512</td><td>2-layer MLP (1000)</td><td>32</td><td>CE</td></tr><tr><td>Vehicle</td><td>Tabular</td><td>2</td><td>3-layer FNN (100,100)</td><td>100</td><td>2-layer MLP (200)</td><td>128</td><td>CE</td></tr><tr><td>KUHAR</td><td>Tabular</td><td>2</td><td>3-layer FNN (100,100)</td><td>100</td><td>2-layer MLP (200)</td><td>128</td><td>CE</td></tr><tr><td>NUS-WIDE</td><td>Tabular</td><td>5</td><td>3-layer FNN (100,100)</td><td>100</td><td>2-layer MLP (200)</td><td>128</td><td>BCE</td></tr><tr><td>PTB-XL</td><td>Sequential / Tabular</td><td>3</td><td>GRU / 3-layer FNN (100,100)†</td><td>512 / 100</td><td>2-layer MLP (1000)</td><td>128</td><td>CE</td></tr></table>

† PTB-XL uses heterogeneous bottom models in the default setting.