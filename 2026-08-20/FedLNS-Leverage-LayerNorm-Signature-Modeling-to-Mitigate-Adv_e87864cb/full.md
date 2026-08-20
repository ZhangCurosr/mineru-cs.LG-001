# FedLNS: Leverage LayerNorm Signature Modeling to Mitigate Adversarial Manipulation in Federated LLMs

Kai Li, Jong-Ik Park, Carlee Joe-Wong, Wei Ni, and Falko Dressler

Abstract—Federated training enables language models to learn from distributed private text, but the server cannot directly verify the local supervision or optimization process that produces each client update. A malicious client can therefore train on corrupted targets, introduce incorrect context–token associations, and degrade the global model through repeated aggregation. Such degradation can also increase the risk of unreliable or hallucinatory generation. We propose Federated Learning with Normalization Signatures (FEDLNS), a server-side framework for lightweight maliciousupdate screening. FEDLNS represents each client update through changes in trainable normalization-layer parameters and screens suspicious updates against a robust, history-aware cross-client reference. Because the signatures are extracted at the server from the returned local models, FEDLNS requires no additional client-to-server parameter or metadata exchange compared to standard federated learning (FL) methods. After screening, the retained full-model updates can be aggregated using standard FL or another compatible aggregation rule. Experiments on GPT-style, BERT-style, and LLaMA-style models trained from scratch with 200 clients show that, under 40% population-level target manipulation, FEDLNS achieves lower test perplexity than the strongest of six baselines for all three architectures under both IID (independently and identically distributed) and non-IID data partitions.

Index Terms—Federated Learning, Federated Language Models, Malicious-Update Screening, Robust Aggregation, Normalization Layers.

## I. INTRODUCTION

Large language models (LLMs) increasingly support search, question answering, software development, decision assistance, and human–computer interaction. Their effectiveness in specialized applications often depends on domain-specific text distributed across users, organizations, and devices [1], [2]. Centralizing such data may be infeasible because of privacy, ownership, confidentiality, or regulatory constraints [3]–[5]. Federated learning (FL) addresses this limitation by coordinating a shared model without collecting raw client data: in each communication round, the server broadcasts the current global model, selected clients optimize it locally, and the server aggregates their returned updates [6]–[8] to begin another training round.

This privacy-preserving separation also creates a critical security gap. The server observes the returned updates but cannot directly verify the local inputs, targets, and optimization procedures that produced them [7], [9], [10]. A malicious client can therefore manipulate its local objective and submit a harmful update while otherwise following the communication protocol. Detecting such behavior is especially difficult for federated language models because each update may contain millions or billions of parameters, and benign client updates can also vary substantially.

This work considers malicious target manipulation, as illustrated in Figure 1. In causal language modeling, the target at each position is the next token, whereas in masked language modeling, it is the original token hidden by the masking process. A malicious client replaces these targets with incorrect tokens, optimizes false context–target associations, and submits the resulting update to the server. If such updates are accepted and repeatedly aggregated at the server, they can propagate corrupted training signals into the global model, reducing test set performance and distorting the model’s predictive distribution over token continuations [11]. These effects may consequently increase the risk of semantically inconsistent or hallucinatory generation [12], [13]. The focus of this work is therefore screening suspicious client updates before aggregation, with improved generation reliability regarded as a downstream effect of limiting corrupted training signals.

Existing Byzantine-robust FL methods suppress harmful updates through distance-based selection, coordinate-wise robust statistics, norm control, or clustering-based filtering [14]–[17]. Because these methods operate on complete client updates, their server-side processing remains tied to full-model dimensionality, while high-dimensional benign variation can obscure malicious deviations.

Recent studies reduce the representation used for malicious-update screening at a FL server through attacksensitive dimension selection or supervised analysis of parameter-efficient updates [18], [19]. These results show that screening need not rely on complete model updates. However, an important question remains: can suspicious updates be screened using a compact, architecture-aware representation without labeled attack samples, trusted server data, or a separately trained detector?

To address this question, we propose Federated Learning with Normalization Signatures (FEDLNS), a serverside framework for lightweight malicious-update screening. Instead of examining complete high-dimensional updates, FEDLNS constructs a compact, architecture-aware signature from changes in the trainable parameters of the model’s normalization layers. These parameters control the scaling and, when supported by the architecture, shifting of hidden representations throughout transformer blocks [20]–[23]. Changes in these parameters therefore provide a compact, architecture-aware view of how local training alters hidden model representations.

![](images/8dcff90d390b16c2d15db88efc4f184f565c6e30a47c7fd86a62ba93c84f3a4b.jpg)  
Fig. 1: Threat model for malicious target manipulation in federated language-model training. A malicious client changes its local supervision and optimizes incorrect context–target associations. If the resulting update is accepted, repeated aggregation can degrade the global model and increase the downstream risk of semantically unreliable generation.

For each participating client, the server extracts a normalization-layer signature relative to the broadcast global model and compares it with a robust historical reference constructed from the latest signatures of previously observed clients. These are compared to coordinate-wise median and median absolute deviation (MAD) statistics to the influence of anomalous bank entries [15], [24]. FEDLNS then employs regularized Gaussian mixture modeling with Bayesian Information Criterion (BIC) to determine whether the current participants form one coherent population or two populations with different deviation levels. When two populations are identified, FEDLNS retains the lowerdeviation, and thus more likely to be benign, group without requiring prior knowledge of the malicious-client fraction.

Because normalization-layer signatures are extracted at the server from the returned local models, FEDLNS adds no defense-specific client transmission overhead. After screening, the retained full-model updates are aggregated using the standard sample-weighted rule, which may be replaced with another compatible FL aggregation rule. FEDLNS requires no raw client data, trusted server dataset, labeled malicious updates, separately trained detector, or auxiliary global models. By limiting the incorporation of target-corrupted updates, FEDLNS aims to improve globalmodel robustness and reduce predictive uncertainty under malicious target manipulation.

The contributions of this work are as follows:

• Compact model update representation. We introduce a normalization-layer update signature that represents each client update through a small, architectureaware parameter subset rather than the complete highdimensional update. Because the signature is extracted from the returned local model, it provides a modular representation for server-side screening before fullmodel aggregation is performed.

• Unsupervised history-aware screening. We develop a server-side update retention rule that constructs a robust cross-client reference from the latest observed signatures and combines median/MAD standardization with BIC-guided mixture modeling. The rule requires no raw client data, trusted server dataset, labeled updates, predefined malicious fraction, separately trained detector, or defense-specific client-to-server payload.

• Cross-architecture evaluation. We evaluate FEDLNS on GPT-style, BERT-style, and LLaMA-style models with 200 clients, IID and two-shard non-IID partitions, and target manipulation affecting 0%–40% of the client population. At 40%, FEDLNS achieves lower perplexity than all six baselines for all three architectures under both IID and two-shard non-IID partitions. The maximum reductions are 18.57% in perplexity and 6.50% in token-level semantic entropy.

The remainder of this paper is organized as follows. Section II reviews the related work. Section III presents the FEDLNS framework. Section IV provides the theoretical analysis. Section V describes the experimental evaluation and performance analysis. Section VI concludes the paper.

## II. RELATED WORK

Prior work relevant to FEDLNS spans two directions: robust aggregation and malicious-update screening in FL and parameter-efficient or model-internal representations.

## A. Robust Aggregation and Malicious-Update Screening

Byzantine-robust aggregation suppresses harmful FL updates using geometric, coordinate-wise, or magnitude-based properties. Multi-Krum selects updates that remain close to a majority neighborhood under pairwise distance [14]. Coordinate-wise Median and Trimmed Mean apply robust statistics independently across parameter dimensions [15], while norm-bounded aggregation limits client influence through update clipping or rescaling [16]. FLAME combines clustering, clipping, and noise injection to reduce the contribution of suspicious clients [17]. Because these methods operate on complete model updates, their serverside processing remains tied to full-model dimensionality.

Other FL defenses use trusted references, multiple models, or historical client behavior. FLTrust constructs a reference update from trusted server data and scores client updates according to their agreement with this reference [25]. FLCert partitions clients into groups, trains multiple global models, and combines their predictions to provide certified robustness [26]. FLDetector predicts client-update trajectories from previous communication rounds and detects persistent deviations from expected behavior [24]. Together, these approaches rely on representative server data, additional model instances, or sufficiently regular client histories.

Recent security methods reduce the representation analyzed during screening. Dim-Krum selects dimensions with stronger backdoor-related changes and applies Krum-based detection in the restricted space [18]. Safe-FedLLM extracts features from LoRA updates and trains an offline supervised probe using labeled benign and malicious samples [19]. These methods show that malicious-update screening need not process every model parameter, but they rely, respectively, on attack-sensitive coordinate selection or labeled update samples and a separately trained probe.

## B. Parameter-Efficient Adaptation and Model-Internal Signals

Parameter-efficient adaptation shows that task-relevant model changes can be expressed through restricted trainable subsets. Adapter tuning introduces small trainable mod ules [27]; BitFit updates only bias parameters [28]; and $\mathrm { I A ^ { 3 } }$ learns compact vectors that scale intermediate activations [29]. LN-Tuning adapts language models through normalization gain and bias parameters [30]. Although designed for adaptation rather than security, these methods demonstrate that comparatively small parameter subsets can encode meaningful changes induced by local optimization.

Normalization layers form a structurally repeated parameter subset in transformer architectures. They regulate the scaling and, depending on the formulation, shifting of normalized hidden representations [20]–[22]. Prior work further shows that their placement and behavior affect optimization, gradient propagation, and transformer expressivity [23], [31]. Their restricted parameterization and recurring role across transformer blocks provide the architectural basis for studying normalization-parameter changes as client-update signals.

Language-model reliability has also been studied through output grounding, uncertainty, and internal model signals. Grounding-based methods compare generations with source evidence, knowledge representations, or semantic relations [32]–[35], while uncertainty- and consistency-based approaches use semantic entropy, confidence, or response agreement [36]–[39]. Other work examines token-level behavior, multimodal consistency, hidden representations, or activation anomalies [40]–[45], and system-level objectives have incorporated hallucination risk directly [46]. These studies analyze outputs, activations, or uncertainty rather than federated model updates, but they establish modelinternal signals as useful indicators of unreliable LLM behavior.

FEDLNS combines a compact normalization-parameter representation with unsupervised, history-aware maliciousupdate screening. Unlike trusted-reference, multi-model, and trajectory-prediction defenses, it constructs a crossclient reference from ordinary federated participation. It also differs from Dim-Krum and Safe-FedLLM by requiring neither attack-sensitive coordinate selection, labeled update samples, nor a separately trained detector.

## III. THE PROPOSED FEDLNS DEFENSE FRAMEWORK

This section presents FEDLNS, which integrates a historical normalization-signature bank, layer-wise median/MAD statistical modeling, and BIC-guided Gaussian mixture clustering. Algorithm 1 presents the overall procedure, which includes local client training (Algorithm 2), serverside signature screening and aggregation (Algorithm 3), and the BIC-guided one-vs-two-component GMM rule for retaining client updates (Algorithm 4).

We consider an honest server with persistent client identifiers and server-controlled client sampling. A fixed subset containing less than half of the client population performs malicious target manipulation while otherwise following the prescribed communication protocol in order to avoid detection by the server. Thus, our threat model excludes Sybil identities, client-ID resets, and adversaries that explicitly optimize their normalization-parameter changes to mimic benign signatures, all of which may be interesting directions for future work.

## A. Federated Training and Client Updates

Algorithm 1 presents the federated training procedure with K clients indexed by

$$
{ \mathcal { C } } = \{ 1 , \ldots , K \} .
$$

At communication round t, the server samples a participating-client set ${ \mathcal { S } } _ { t } \subseteq { \mathcal { C } }$ and broadcasts the current

Algorithm 1 FEDLNS communication-round driver Algorithm 2 ClientLocalTrain $\left( \mathbf { w } _ { t } , \mathcal { D } _ { i } , E \right)$   
Require: Initial global model parameters $\mathbf { w } _ { 0 } ,$ client set C, total Require: Global model parameters $\mathbf { w } _ { t } ,$ local dataset D , local   
rounds T, local epochs $E ,$ bank activation threshold $M _ { \mathrm { { a c t } } }$ epochs E   
Ensure: Final global model parameters $\mathbf { w } _ { T }$ Ensure: Local model parameters $\mathbf { w } _ { t , i } ,$ processed-example count   
1: Initialize server-side history bank $B _ { - 1 } = \varnothing$ and client-ID set $n _ { t , i }$   
$\mathcal { H } _ { - 1 } = \emptyset$ 1: Initialize local model parameters $\mathbf { w } _ { t , i }  \mathbf { w } _ { t }$   
2: for $t = 0 , 1 , \ldots , T - 1$ do 2: Set $n _ { t , i } \gets 0$   
3: Sample participating clients $S _ { t } \subseteq { \mathcal { C } }$ 3: for local epoch $e = 1 , \ldots , E$ do   
4: Broadcast $\mathbf { w } _ { t }$ to every client $i \in S _ { t }$ 4: for mini-batch $\mathcal { M } \in$ MiniBatchLoader $( \mathcal { D } _ { i } )$ do   
5: for each client $i \in S _ { t }$ in parallel do 5: Set $\mathcal { L } _ { \mathcal { M } }  0$   
6: $( \mathbf { w } _ { t , i } , n _ { t , i } ) \gets$ ClientLocalTrain $\left( \mathbf { w } _ { t } , \mathcal { D } _ { i } , E \right)$ 6: for each training instance $\xi \in \mathcal { M }$ do   
7: end for 7: $\mathcal { L } _ { \mathcal { M } } \gets \mathcal { L } _ { \mathcal { M } } + \ell ( \mathbf { w } _ { t , i } ; \boldsymbol { \xi } )$   
8: $\left( \mathbf { w } _ { t + 1 } , \mathcal { R } _ { t } , \boldsymbol { B } _ { t } , \mathcal { H } _ { t } \right) \gets$ 8: $n _ { t , i } \gets n _ { t , i } + 1$   
ServerFedLNS $\big ( \mathbf { w } _ { t } , \big \{ \big ( \mathbf { w } _ { t , i } , n _ { t , i } \big ) \big \} _ { i \in \mathcal { S } _ { t } } , \mathcal { B } _ { t - 1 } , \mathcal { H } _ { t - 1 } , M _ { \mathrm { a c t } } \big )$ 9: end for   
9: end for 10: $\mathcal { L } _ { \mathcal { M } }  \mathcal { L } _ { \mathcal { M } } / | \mathcal { M } |$   
10: return $\mathbf { w } _ { T }$ 11: $\mathbf { w } _ { t , i } \gets ($ OptimizerStep $\left( { \bf w } _ { t , i } , \nabla _ { \bf w } \mathcal { L } _ { \mathcal { M } } ( { \bf w } _ { t , i } ) \right)$   
12: end for   
13: end for

global model parameters ${ \mathbf w } _ { t } \in \mathbb { R } ^ { P }$ to the selected clients (Algorithm 1, lines 3–4), where $N _ { t } : = | S _ { t } |$

Client i trains on its local dataset $\mathcal { D } _ { i }$ to minimize the empirical objective

$$
F _ { i } ( \mathbf { w } ) : = \frac { 1 } { | \mathcal { D } _ { i } | } \sum _ { \xi \in \mathcal { D } _ { i } } \ell ( \mathbf { w } ; \xi ) ,\tag{1}
$$

where $\ell ( \cdot )$ denotes the language-modeling loss function, e.g., the next-token prediction loss for causal language modeling or the masked-token prediction loss for masked language modeling [47]. Here, ξ represents an individual training instance sampled from $\mathcal { D } _ { i }$ , such as a text sequence together with its associated supervision signal.

Client i initializes its local model from $\mathbf { w } _ { t }$ (Algorithm 2, line 1), performs E local epochs using mini-batch optimization (lines 3–13), and returns the locally trained parameters and processed-example count (line 14):

$$
\left( \mathbf { w } _ { t , i } , { n } _ { t , i } \right) = \mathrm { C l i e n t L o c a l T r a i n } \left( \mathbf { w } _ { t } , \mathcal { D } _ { i } , E \right) .\tag{2}
$$

Upon completing local training, client i returns the updated model parameters $\mathbf { w } _ { t , i }$ together with $n _ { t , i } ,$ the total number of training-example occurrences processed across the E local epochs. The corresponding client update is

$$
\Delta \mathbf { w } _ { t , i } : = \mathbf { w } _ { t , i } - \mathbf { w } _ { t } .\tag{3}
$$

Because normalization-layer signatures are extracted at the server from the returned local models, clients follow the standard FL training and upload procedure without transmitting defense-specific parameters or metadata. All signature extraction, bank maintenance, and GMM-based screening are performed by the server.

After the participating clients return their locally trained models (Algorithm 1, line 6), the server invokes ServerFedLNS (line $^ { 8 ) , }$ described in Algorithm 3, to screen the returned updates before global aggregation.

## B. Server-Side Normalization-Signature Extraction

Algorithm 3 computes each complete client update and extracts its layer-wise normalization signature (lines 2–3). Let L denote the ordered set of normalization layers whose trainable parameters are included in the FEDLNS signature,

$$
\left( \mathbf { w } _ { t , i } , n _ { t , i } \right)
$$

and let $d _ { \ell }$ denote the number of included parameters associated with layer $\ell \in { \mathcal { L } }$ . We define

$$
\mathbf { p } _ { \ell } ( \mathbf { w } ) \in \mathbb { R } ^ { d _ { \ell } }
$$

as the vector formed by concatenating all trainable normalization parameters of layer ℓ under model parameters w. For a standard LayerNorm layer, $\mathbf { p } _ { \ell } ( \mathbf { w } )$ consists of the trainable scaling and bias parameters. For a bias-free normalization mechanism such as RMSNorm, the vector contains only the trainable scaling parameters.

For selected client $i \in S _ { t }$ , the layer-wise normalization signature is defined as

$$
\mathbf { s } _ { t , i , \ell } : = \mathbf { p } _ { \ell } ( \mathbf { w } _ { t , i } ) - \mathbf { p } _ { \ell } ( \mathbf { w } _ { t } ) \in \mathbb { R } ^ { d _ { \ell } } .\tag{4}
$$

Thus, $\mathbf { s } _ { t , i , \ell }$ measures how local training at client i changes the normalization parameters of layer ℓ relative to the current global model parameters $\mathbf { w } _ { t }$

The complete signature of client i is the ordered collection

$$
\mathbf { s } _ { t , i } : = ( \mathbf { s } _ { t , i , \ell } ) _ { \ell \in \mathcal { L } } .\tag{5}
$$

The signature is computed entirely at the server from the uploaded local models and the current global model.

Normalization parameters regulate the scaling and, depending on the architecture, shifting of hidden representations throughout transformer blocks. FEDLNS therefore uses changes in these parameters as a compact, architectureaware screening representation, while the complete returned model updates remain available for aggregation.

## C. Server-Side History Bank

FEDLNS maintains a server-side history bank that stores the latest normalization-layer signature associated with each client observed during training. The resulting collection provides a cross-client reference for constructing robust statistics across communication rounds. Prior to processing round t, the history bank is defined as

$$
B _ { t - 1 } = \left\{ ( j , \mathbf { b } _ { t - 1 , j } ) : j \in \mathcal { H } _ { t - 1 } \right\} ,\tag{6}
$$

Algorithm 3 ServerFedLNS: server-side signature screen  
ing and aggregation   
Require: Global model parameters $\mathbf { w } _ { t } ,$ returned client models   
and counts $\{ ( \mathbf { w } _ { t , i } , n _ { t , i } ) \} _ { i \in \mathcal { S } _ { t } }$ , previous bank $B _ { t - 1 } .$ , previous   
ID set $\mathcal { H } _ { t - 1 } ,$ bank activation threshold $M _ { \mathrm { a c t } }$   
Ensure: Updated global model parameters $\mathbf { w } _ { t + 1 } ,$ retained client   
set ${ \mathcal { R } } _ { t } ,$ refreshed bank $B _ { t } ,$ refreshed ID set H<sub>t</sub>   
1: for each client $i \in S _ { t }$ do   
2: Compute $\Delta \mathbf { w } _ { t , i } = \mathbf { w } _ { t , i } - \mathbf { w } _ { t }$   
3: Extract $\mathbf { s } _ { t , i , \ell } = \mathbf { p } _ { \ell } ( \mathbf { w } _ { t , i } ) - \mathbf { p } _ { \ell } ( \mathbf { w } _ { t } )$ for all $\ell \in { \mathcal { L } }$   
4: end for   
5: Refresh $B _ { t }$ and $\mathcal { H } _ { t }$ using (8)–(10)   
6: Compute $N _ { t } ^ { \mathrm { b a n k } } = | \mathcal { H } _ { t } |$   
7: if $N _ { t } ^ { \mathrm { { \tiny { ^ { b a n k } } } } } < M _ { \mathrm { { a c t } } }$ then   
8: Set $\mathbf { w } _ { t + 1 } = \mathbf { w } _ { t }$ and $\mathcal { R } _ { t } = \emptyset$   
9: return $\left( \mathbf { w } _ { t + 1 } , \mathcal { R } _ { t } , { \boldsymbol { B } } _ { t } , \mathcal { H } _ { t } \right)$   
10: end if   
11: for each normalization layer $\ell \in { \mathcal { L } }$ do   
12: Compute $\mathbf { m } _ { t , \ell }$ and $\mathbf { d } _ { t , \ell }$ using (14)–(16)   
13: end for   
14: for each client $i \in S _ { t }$ do   
15: Compute $\mathbf { z } _ { t , i }$ using $( 1 7 ) \mathrm { - } ( 1 8 )$   
16: Compute $\delta _ { t , i }$ using (20)   
17: end for   
18: R ← BICGMMRetain $\left( \{ \mathbf { z } _ { t , i } , \delta _ { t , i } \} _ { i \in \mathcal { S } _ { t } } \right)$   
19: Aggregate retained updates using (34) to obtain $\mathbf { w } _ { t + 1 }$   
20: return $( \mathbf { w } _ { t + 1 } , \mathcal { R } _ { t } , \bar { \mathcal { B } _ { t } } , \mathcal { H } _ { t } )$

where $\mathcal { H } _ { t - 1 }$ denotes the set of client identifiers currently maintained in the bank.

For each client j, the stored entry consists of an ordered collection of layer-wise normalization-signature vectors:

$$
\begin{array} { r } { \mathbf { b } _ { t - 1 , j } = ( \mathbf { b } _ { t - 1 , j , \ell } ) _ { \ell \in \mathcal { L } } , } \end{array}\tag{7}
$$

where $\mathbf { b } _ { t - 1 , j , \ell }$ represents the latest recorded signature corresponding to normalization layer ℓ for client $j .$

After extracting the current signatures, the server replaces the bank entries of participating clients and retains the latest entries of nonparticipating clients (Algorithm 3, line 5). The updated client-ID set is

$$
\mathscr { H } _ { t } = \mathscr { H } _ { t - 1 } \cup S _ { t } .\tag{8}
$$

The refreshed bank is

$$
B _ { t } = \left\{ \left( j , \mathbf { b } _ { t , j } \right) : j \in \mathcal { H } _ { t } \right\} ,\tag{9}
$$

where

$$
\mathbf { b } _ { t , j } = \left\{ { \begin{array} { l l } { \mathbf { s } _ { t , j } , } & { { \mathrm { i f ~ } } j \in { \mathcal { S } } _ { t } , } \\ { \mathbf { b } _ { t - 1 , j } , } & { { \mathrm { i f ~ } } j \in { \mathcal { H } } _ { t - 1 } { \mathrm { ~ a n d ~ } } j \not \in { \mathcal { S } } _ { t } . } \end{array} } \right.\tag{10}
$$

If a client participates in the current communication round, its previous bank entry is replaced by the newly extracted normalization-layer signature. If the client does not participate, its latest stored signature remains unchanged. As shown in Algorithm 3, the bank is refreshed before current-round screening. Thus, all current signatures contribute to the refreshed bank, including signatures associated with updates that may subsequently be excluded from aggregation.

The history bank is maintained entirely by the server and is not stored, accessed, or received by participating clients.

After refreshing the bank, the server computes its current coverage and applies the activation rule (Algorithm 3, lines 6–9). Screening and global aggregation begin when

$$
N _ { t } ^ { \mathrm { b a n k } } \geq M _ { \mathrm { a c t } } ,\tag{11}
$$

where

$$
N _ { t } ^ { \mathrm { b a n k } } : = | \mathcal { H } _ { t } |
$$

is the number of distinct client identities represented in the history bank. Before this condition is satisfied, the current round is used to collect normalization-layer signatures, and the global model remains unchanged, i.e., all participating clients in $S _ { t }$ contribute to the global model:

$$
\mathbf { w } _ { t + 1 } = \mathbf { w } _ { t } .\tag{12}
$$

The activation threshold does not cap the bank size. After screening begins, the bank continues to incorporate newly observed client identities and ideally approaches full population coverage. The bank-coverage ablation shows that broader coverage at activation generally improves the early active-training performance, including monotonic perplexity improvements under the strongest IID attack setting; see Appendix $\mathrm { E , }$ particularly Tables LII and LIII.

Waiting for every client to enter the bank can, however, be impractically slow when participation is intermittent or nonuniform. The threshold $M _ { \mathrm { { a c t } } }$ therefore provides a practical starting point for screening with a partially populated bank while allowing the bank to continue growing thereafter. Under the uniform random participation model used in the experiments, Appendix A bounds the probability that such a partial bank contains a malicious majority and analyzes the time required to reach a chosen activation coverage.

## D. Layer-Wise Robust Bank Reference

Once the activation condition in (11) is satisfied, the server computes the layer-wise median/MAD reference (Algorithm 3, lines 11–13) and then calculates the standardized representation and scalar deviation score of each current client (lines 15–16). These quantities form the input to the GMM screening rule.

1) Robust reference statistics: For vectors $\mathbf { u } ^ { ( 1 ) } , \ldots , \mathbf { u } ^ { ( m ) } \in \mathbb { R } ^ { d }$ , the coordinate-wise median is defined as

$$
\begin{array} { r } { \mathrm { m e d } \big ( \mathbf { u } ^ { ( 1 ) } , \ldots , \mathbf { u } ^ { ( m ) } \big ) [ r ] : = \mathrm { m e d i a n } \big ( u _ { r } ^ { ( 1 ) } , \ldots , u _ { r } ^ { ( m ) } \big ) , } \end{array}\tag{13}
$$

where $r = 1 , \ldots , d$ denotes the coordinate index. Compared with the arithmetic mean, the coordinate-wise median reduces the influence of extreme values and provides a robust estimate of the population center [48].

For each normalization layer $\ell \in { \mathcal { L } } .$ , the server gathers the corresponding normalization-layer signatures from all clients currently stored in the history bank, $\{ \mathbf { b } _ { t , j , \ell } \} _ { j \in \mathcal { H } _ { t } } .$ and computes their coordinate-wise median:

$$
\begin{array} { r } { \mathbf { m } _ { t , \ell } : = \mathrm { m e d } \left( \{ \mathbf { b } _ { t , j , \ell } \} _ { j \in \mathcal { H } _ { t } } \right) \in \mathbb { R } ^ { d _ { \ell } } . } \end{array}\tag{14}
$$

The vector $\mathbf { m } _ { t , \ell }$ serves as a robust estimate of the central tendency of the normalization-layer signatures at layer ℓ.

To quantify the dispersion of the client population around the center, FEDLNS further computes the coordinate-wise MAD:

$$
\widetilde { \mathbf { d } } _ { t , \ell } : = \mathrm { m e d } \left( \{ | \mathbf { b } _ { t , j , \ell } - \mathbf { m } _ { t , \ell } | \} _ { j \in \mathcal { H } _ { t } } \right) \in \mathbb { R } ^ { d _ { \ell } } ,\tag{15}
$$

where the absolute value is applied element-wise. Compared with variance-based measures, MAD provides a robust estimate of scale in the presence of outliers and adversarially manipulated updates [49].

To prevent numerical instability caused by near-zero dispersion values, each coordinate of the MAD vector is lower-bounded by a small positive constant $\epsilon > 0 .$ , yielding

$$
\mathbf { d } _ { t , \ell } [ r ] : = \operatorname* { m a x } \left\{ \widetilde { \mathbf { d } } _ { t , \ell } [ r ] , \epsilon \right\} , \qquad r = 1 , \dots , d _ { \ell } .\tag{16}
$$

This regularization avoids division-by-zero issues and prevents excessively large standardized deviations arising from extremely small scale estimates. The resulting pair $( \mathbf { m } _ { t , \ell } , \mathbf { d } _ { t , \ell } )$ constitutes the robust center and scale reference for normalization layer ℓ.

2) Client deviation scores: After constructing the robust layer-wise reference statistics, FEDLNS projects each current client signature into a bank-standardized feature space, as shown in Algorithm 3. This transformation quantifies the deviation of a client’s normalization-layer signature from the historical reference while placing coordinates from different normalization layers on comparable robust scales within the current screening round.

For client $i \in S _ { t }$ and normalization layer $\ell \in { \mathcal { L } } .$ , the standardized signature is defined as

$$
\mathbf { z } _ { t , i , \ell } : = \frac { \mathbf { s } _ { t , i , \ell } - \mathbf { m } _ { t , \ell } } { \mathbf { d } _ { t , \ell } } \in \mathbb { R } ^ { d _ { \ell } } ,\tag{17}
$$

where subtraction and division are performed element-wise. Here, $\mathbf { s } _ { t , i , \ell }$ denotes the normalization-layer signature of client i, while $\mathbf { m } _ { t , \ell }$ and $\mathbf { d } _ { t , \ell }$ are the robust median and MAD reference vectors derived from the history bank.

The complete standardized representation of client i is obtained by concatenating the standardized layer-wise signatures according to a fixed layer ordering:

$$
\mathbf { z } _ { t , i } : = \mathrm { c o n c a t } _ { \ell \in \mathcal { L } } \left( \mathbf { z } _ { t , i , \ell } \right) \in \mathbb { R } ^ { D } ,\tag{18}
$$

where the dimensionality of the resulting feature vector is

$$
D : = \sum _ { \ell \in \mathcal { L } } d _ { \ell } .\tag{19}
$$

To obtain an overall measure of deviation from the historical population, FEDLNS computes a scalar deviation score for each participating client:

$$
\delta _ { t , i } : = \mathrm { m e d i a n } \left( | \mathbf { z } _ { t , i } [ 1 ] | , \dots , | \mathbf { z } _ { t , i } [ D ] | \right) .\tag{20}
$$

The score $\delta _ { t , \cdot }$ <sub>i</sub> is the median absolute standardized deviation across the coordinates of the client’s feature vector. A smaller value indicates that the client’s normalization-layer signature is more consistent with the historical reference, whereas a larger value indicates atypical behavior across a substantial portion of the standardized feature space. The subsequent GMM stage identifies the population with lower deviation from the bank reference under the assumption that benign signatures form the dominant reference population.

## E. BIC-Guided One-vs-Two GMM Screening

Algorithm 4 fits one- and two-component diagonal GMMs (Gaussian mixture models) to clients’ standardized signatures and selects the model order using BIC (lines 2–4). If two components are selected, the algorithm assigns component labels and retains the component with the smaller median deviation from the bank reference (lines 8–13). If the client population is represented by one component, all participating clients are assumed to be benign and retained. Otherwise, the clients are partitioned into two groups, and the group with the smaller overall deviation from the historical normalization-signature reference is selected for aggregation. This rule avoids forcing a two-component partition when a one-component model is selected (e.g., if no malicious clients are present in this training round) and requires neither a predefined maliciousclient fraction nor a manually specified anomaly threshold.

In Algorithm 4, the current-round feature matrix is

$$
\mathbf { Z } _ { t } : = \left[ \mathbf { z } _ { t , i } ^ { \top } \right] _ { i \in \mathcal { S } _ { t } } \in \mathbb { R } ^ { N _ { t } \times D } .\tag{21}
$$

For each candidate number of mixture components $c \in$ {1, 2}, FEDLNS fits a diagonal-covariance GMM with covariance regularization $1 0 ^ { - 4 }$

$$
f _ { c } ( \mathbf { z } ; \boldsymbol { \Theta } _ { c } ) = \sum _ { k = 1 } ^ { c } \pi _ { c , k } \mathcal { N } \left( \mathbf { z } ; \boldsymbol { \mu } _ { c , k } , \mathrm { D i a g } \left( \boldsymbol { \sigma } _ { c , k } ^ { 2 } \right) \right) ,\tag{22}
$$

where

$$
\Theta _ { c } = \left\{ \pi _ { c , k } , \mu _ { c , k } , \sigma _ { c , k } ^ { 2 } \right\} _ { k = 1 } ^ { c } , \quad \pi _ { c , k } > 0 , \quad \sum _ { k = 1 } ^ { c } \pi _ { c , k } = 1 .
$$

The fitted parameter set is

(23)

$$
\widehat { \Theta } _ { c } : = \arg \operatorname* { m a x } _ { \Theta _ { c } } \sum _ { i \in \mathcal { S } _ { t } } \log f _ { c } ( \mathbf { z } _ { t , i } ; \Theta _ { c } ) .\tag{24}
$$

The BIC score for each candidate model order c is

$$
\mathrm { B I C } ( \boldsymbol { c } ) = - 2 \sum _ { i \in \mathcal { S } _ { t } } \log f _ { c } ( \mathbf { z } _ { t , i } ; \widehat { \boldsymbol { \Theta } } _ { c } ) + p _ { c } \log N _ { t } ,\tag{25}
$$

where the number of free parameters in a diagonal ccomponent GMM is

$$
p _ { c } = \left( c - 1 \right) + 2 c D .\tag{26}
$$

FEDLNS first selects the number of mixture components $\widehat { c } _ { t }$ as

$$
{ \widehat { c } } _ { t } : = \arg \operatorname* { m i n } _ { c \in \{ 1 , 2 \} } { \mathrm { B I C } } ( c ) .\tag{27}
$$

If $\widehat { c } _ { t } = 1$ , all participating clients are retained:

$$
\mathcal { R } _ { t } = { S } _ { t } .\tag{28}
$$

Algorithm 4 BICGMMRetain: adaptive one-vs-two GMM   
retain rule   
Require: Standardized features $\{ \mathbf { z } _ { t , i } \} _ { i \in { S } _ { t } }$ , deviation scores   
$\textstyle \left\{ \delta _ { t , i } \right\} _ { i \in { \mathcal { S } } _ { t } }$   
Ensure: Retained client set $\mathcal { R } _ { t }$   
1: Form feature matrix $\mathbf Z _ { t } = [ \mathbf z _ { t , i } ^ { \top } ] _ { i \in \mathcal S _ { t } }$   
2: Fit one- and two-component diagonal GMMs to $\mathbf { Z } _ { t }$   
3: Compute BIC(1) and BIC(2) using (25)   
4: Set bct = arg minc∈{1,2} BIC(c)   
5: if $\widehat { c } _ { t } = 1$ then   
6: return $\mathcal { R } _ { t } = \mathcal { S } _ { t }$   
7: else   
8: Assign component labels $y _ { t , i }$ using (29)   
9: for $\breve { k } \in \{ 1 , { 2 } \}$ do   
10: Compute $\rho _ { t , k }$ using (30)   
11: end for   
12: Set retained component $k _ { t } ^ { \mathrm { r e t } } = \mathrm { a r g }$ min<sub>k∈{1,2}</sub> ρ<sub>t,k</sub>   
13: return $\mathcal { R } _ { t } = \{ i \in S _ { t } : y _ { t , i } = k _ { t } ^ { \mathrm { { r e t } } } \}$   
14: end if

If $\widehat { c } _ { t } = 2 .$ , each client i is assigned to a component:

$$
y _ { t , i } : = \arg \operatorname* { m a x } _ { k \in \{ 1 , 2 \} } \widehat { \pi } _ { 2 , k } \mathcal { N } \left( \mathbf { z } _ { t , i } ; \widehat { \mu } _ { 2 , k } , \mathrm { D i a g } \left( \widehat { \pmb { \sigma } } _ { 2 , k } ^ { 2 } \right) \right) .\tag{29}
$$

The median deviation score of component k is

$$
\begin{array} { r } { \rho _ { t , k } : = \operatorname * { m e d i a n } \left( \{ \delta _ { t , i } : i \in S _ { t } , \ y _ { t , i } = k \} \right) , \quad k \in \{ 1 , 2 \} . } \end{array}\tag{30}
$$

The retained component is

$$
k _ { t } ^ { \mathrm { r e t } } : = \arg \operatorname* { m i n } _ { k \in \{ 1 , 2 \} } \rho _ { t , k } .\tag{31}
$$

The retained client set is

$$
\mathcal { R } _ { t } = \left\{ i \in { S } _ { t } : y _ { t , i } = k _ { t } ^ { \mathrm { r e t } } \right\} .\tag{32}
$$

After the retained set $\mathcal { R } _ { t }$ is determined, the server aggregates the complete retained updates (Algorithm 3, line 19). For client $i \in \mathcal R _ { t }$ , the sample-weighted aggregation coefficient is

$$
\begin{array} { r } { \alpha _ { t , i } : = \frac { n _ { t , i } } { \sum _ { j \in \mathscr { R } _ { t } } n _ { t , j } } . } \end{array}\tag{33}
$$

The next global model is

$$
\mathbf { w } _ { t + 1 } = \mathbf { w } _ { t } + \sum _ { i \in \mathcal { R } _ { t } } \alpha _ { t , i } \Delta \mathbf { w } _ { t , i } .\tag{34}
$$

FEDLNS therefore acts as a server-side screening stage before full-model aggregation. The retained updates can be passed to the sample-weighted rule above or to another compatible FL aggregation rule.

## IV. THEORETICAL ANALYSIS

This section analyzes two aspects of FEDLNS. First, we examine how changes in trainable normalization-layer parameters contribute to changes in token preferences. Second, we study how screening suspicious clients affects optimization of the benign global objective.

A. Influence of Normalization-Parameter Changes on $T o -$ ken Margins

For an input sequence $\boldsymbol { x } ~ = ~ ( x _ { 1 } , \dots , x _ { m } )$ , a language model with parameters w produces a logit vector

$$
\mathbf { z } ( x ; \mathbf { w } ) \in \mathbb { R } ^ { | \nu | } ,\tag{35}
$$

where V is the token vocabulary. The probability assigned to token $y \in \mathcal V$ is

$$
\pi _ { \mathbf { w } } ( y \mid x ) = \frac { \exp ( z _ { y } ( x ; \mathbf { w } ) ) } { \sum _ { v \in \mathcal { V } } \exp ( z _ { v } ( x ; \mathbf { w } ) ) } .\tag{36}
$$

For two candidate tokens $u , v \in \mathcal { V }$ , define the pairwise token margin

$$
m _ { u , v } ( x ; \mathbf { w } ) : = z _ { u } ( x ; \mathbf { w } ) - z _ { v } ( x ; \mathbf { w } ) .\tag{37}
$$

The margin measures the model’s relative preference for token u over token v and satisfies

$$
m _ { u , v } ( x ; \mathbf { w } ) = \log { \frac { \pi _ { \mathbf { w } } ( u \mid x ) } { \pi _ { \mathbf { w } } ( v \mid x ) } } .\tag{38}
$$

Thus, a client update that changes the pairwise margin also changes the probability ratio assigned to the two tokens.

Partition the model parameters as

$$
\mathbf { w } = ( \mathbf { p } , \mathbf { q } ) ,\tag{39}
$$

where p contains the trainable normalization-layer parameters included in the FEDLNS signature and q contains the remaining model parameters.

For client i at communication round t, write the local update as

$$
\Delta \mathbf { w } _ { t , i } = ( \Delta \mathbf { p } _ { t , i } , \Delta \mathbf { q } _ { t , i } ) .\tag{40}
$$

Here, $\Delta \mathbf { p } _ { t , i }$ is the concatenated normalization-parameter component corresponding to the normalization-layer signature used by FEDLNS.

Because a client update affects the model across inputs, we define its prompt-averaged pairwise-margin change as

$$
\Delta \bar { m } _ { t , i } ( u , v ) : = \mathbb { E } _ { x \sim \mathcal { X } } \left[ m _ { u , v } \left( x ; \mathbf { w } _ { t } + \Delta \mathbf { w } _ { t , i } \right) - m _ { u , v } ( x ; \mathbf { w } _ { t } ) \right] .\tag{41}
$$

Here, $m _ { u , v } ( x ; { \mathbf w } )$ denotes the margin for an individual prompt, whereas $\Delta \bar { m } _ { t , i } ( u , v )$ denotes the change in that margin averaged over the prompt distribution $\mathcal { X } .$

The following proposition identifies the direct first-order contribution of the normalization-parameter component to the expected token-margin change.

Proposition 1 (Normalization-parameter contribution to token-margin change).

Suppose $m _ { u , v } ( x ; { \mathbf { w } } )$ is twice continuously differentiable in a neighborhood of $\mathbf { w } _ { t } ,$ , and its Hessian is uniformly bounded in operator norm by $L _ { m }$ . Then

$$
\begin{array} { r } { \Delta \bar { m } _ { t , i } ( u , v ) = \mathbf { g } _ { p , u , v } ^ { \top } \Delta \mathbf { p } _ { t , i } + \mathbf { g } _ { q , u , v } ^ { \top } \Delta \mathbf { q } _ { t , i } + R _ { t , i } ( u , v ) , } \end{array}\tag{42}
$$

where

$$
\begin{array} { r } { \mathbf { g } _ { p , u , v } : = \mathbb { E } _ { x \sim \mathcal { X } } \left[ \nabla _ { \mathbf { p } } m _ { u , v } ( x ; \mathbf { w } _ { t } ) \right] , } \end{array}
$$

$$
\begin{array} { r } { \mathbf { g } _ { q , u , v } : = \mathbb { E } _ { x \sim \mathcal { X } } \left[ \nabla _ { \mathbf { q } } m _ { u , v } ( x ; \mathbf { w } _ { t } ) \right] , } \end{array}\tag{43}
$$

and the prompt-averaged remainder satisfies

$$
| R _ { t , i } ( u , v ) | \leq \frac { L _ { m } } { 2 } \left\| \Delta \mathbf { w } _ { t , i } \right\| _ { 2 } ^ { 2 } .\tag{44}
$$

Proposition 1 shows that the normalization-parameter component contributes directly to the first-order change in token margins through

$$
\mathbf { g } _ { p , u , v } ^ { \top } \Delta \mathbf { p } _ { t , i } .\tag{45}
$$

Consequently, normalization-parameter changes contribute directly to first-order changes in relative token preferences.

## B. Effect of Screening on the Aggregated Update

We next analyze the effect of excluding suspicious clients before aggregation. Let $\mathcal { C } _ { \mathrm { b e n } }$ denote the set of benign clients. The benign global objective is

$$
F ( \mathbf { w } ) : = \sum _ { i \in \mathcal { C } _ { \mathrm { b e n } } } \lambda _ { i } F _ { i } ( \mathbf { w } ) , \qquad \lambda _ { i } \geq 0 , \quad \sum _ { i \in \mathcal { C } _ { \mathrm { b e n } } } \lambda _ { i } = 1 ,\tag{46}
$$

where $F _ { i }$ is the local objective of benign client $i .$

At round $t ,$ FEDLNS returns a retained client set $\mathcal { R } _ { t } \subseteq$ $S _ { t } .$ For retained client i, define the normalized aggregation coefficient

$$
\alpha _ { t , i } = \frac { n _ { t , i } } { \sum _ { j \in \mathcal { R } _ { t } } n _ { t , j } } , \qquad i \in \mathcal { R } _ { t } .\tag{47}
$$

The screened update is

$$
\mathbf { w } _ { t + 1 } = \mathbf { w } _ { t } + \sum _ { i \in \mathcal { R } _ { t } } \alpha _ { t , i } \Delta \mathbf { w } _ { t , i } .\tag{48}
$$

For analysis, write the same update as

$$
\mathbf { w } _ { t + 1 } = \mathbf { w } _ { t } - \eta _ { t } \mathbf { g } _ { t } ,\tag{49}
$$

where $\eta _ { t } > 0$ is an effective step size and $\mathbf { g } _ { t }$ is the effective retained update direction.

Define the screening error

$$
\mathbf { e } _ { t } : = \mathbb { E } \left[ \mathbf { g } _ { t } \mid \mathcal { F } _ { t } \right] - \nabla F ( \mathbf { w } _ { t } ) ,\tag{50}
$$

where $\mathcal { F } _ { t }$ contains the information available before the round-t screening, i.e., the vector $\mathbf { e } _ { t }$ measures the difference between the expected retained update direction and the gradient of the benign global objective.

## Theorem 1 (Finite-time bound under screened aggregation).

Assume that F is L<sub>F</sub>-smooth and lower bounded by $F _ { \mathrm { i n f } }$ Suppose that, for every round t, there exist nonnegative quantities $\beta _ { t }$ and $\nu _ { t } ^ { 2 }$ such that

$$
\| \mathbf { e } _ { t } \| _ { 2 } \leq \beta _ { t } , \qquad \beta _ { t } \geq 0 ,\tag{51}
$$

and

$$
\mathbb { E } \left[ \left\| \mathbf { g } _ { t } - \mathbb { E } [ \mathbf { g } _ { t } \mid \mathcal { F } _ { t } ] \right\| _ { 2 } ^ { 2 } \Big | \mathcal { F } _ { t } \right] \leq \nu _ { t } ^ { 2 } .\tag{52}
$$

$I f 0 \leq \eta _ { t } \leq 1 / ( 4 L _ { F } )$ , then for any $T \geq 1$ with

$$
S _ { T } : = \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } > 0 ,\tag{53}
$$

we have

$$
\begin{array} { r l } & { \frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \mathbb { E } \left[ \| \nabla F ( \mathbf { w } _ { t } ) \| _ { 2 } ^ { 2 } \right] } { S _ { T } } \leq \frac { 2 \left( F ( \mathbf { w } _ { 0 } ) - F _ { \mathrm { i n f } } \right) } { S _ { T } } } \\ & { \quad + \frac { 2 \sum _ { t = 0 } ^ { T - 1 } \left( \eta _ { t } + L _ { F } \eta _ { t } ^ { 2 } \right) \mathbb { E } \left[ \beta _ { t } ^ { 2 } \right] } { S _ { T } } + \frac { L _ { F } \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } ^ { 2 } \mathbb { E } \left[ \nu _ { t } ^ { 2 } \right] } { S _ { T } } . } \end{array}\tag{54}
$$

Theorem 1 bounds the step-size-weighted average squared gradient norm of the benign objective. The righthand side contains three terms: the initial optimization gap, the contribution of the screening-error upper bound $\beta _ { t } \ ( \mathrm { i . e . }$ the discrepancy between the expected retained update direction and the gradient of the benign global objective), and the stochastic update-noise contribution governed by $\nu _ { t } ^ { 2 }$ . A smaller $\beta _ { t }$ indicates that the expected direction formed from the retained clients more closely approximates the benign global gradient. The theorem characterizes optimization under such an approximation, but it does not independently guarantee that the FEDLNS screening rule makes $\beta _ { t }$ small in every round.

The following corollary gives conditions under which the weighted average gradient norm vanishes.

Corollary 1 (Convergence to first-order stationarity). Suppose

$$
S _ { T } = \sum _ { t = 0 } ^ { T - 1 } \eta _ { t }  \infty , \qquad \frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } ^ { 2 } } { S _ { T } }  0 .\tag{55}
$$

Assume that the update variance is uniformly bounded:

$$
\mathbb { E } [ \nu _ { t } ^ { 2 } ] \le \nu ^ { 2 }\tag{56}
$$

for some finite constant $\nu ^ { 2 } .$ . Also assume that the weighted average screening error vanishes:

$$
\frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \mathbb { E } [ \beta _ { t } ^ { 2 } ] } { S _ { T } }  0 .\tag{57}
$$

Then

$$
\frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \mathbb { E } \left[ \lVert \nabla F ( \mathbf { w } _ { t } ) \rVert _ { 2 } ^ { 2 } \right] } { S _ { T } } \to 0 .\tag{58}
$$

Equivalently, if a random round $\tau$ is sampled from $\{ 0 , \ldots , T - 1 \}$ with probability

$$
\mathrm { P r } ( \tau = t ) = \frac { \eta _ { t } } { S _ { T } } ,\tag{59}
$$

then

$$
\begin{array} { r } { \mathbb { E } [ \| \nabla F ( \mathbf { w } _ { \tau } ) \| _ { 2 } ^ { 2 } ]  0 . } \end{array}\tag{60}
$$

Corollary 1 shows that the expected gradient norm of a step-size-weighted random iterate vanishes when the step sizes provide sufficient cumulative progress, the stochastic update variance remains bounded, and the average screening error becomes negligible. The final condition requires the retained update direction to approach the benign gradient direction.

The following proposition decomposes the screening error into quantities associated with retained malicious updates, benign-client selection, and local-training drift.

Proposition 2 (Interpretation of screening error). Let

$$
\mu _ { t } : = \sum _ { a \in \mathcal { R } _ { t } \cap \mathcal { C } _ { \operatorname* { m a l } } } \alpha _ { t , a }\tag{61}
$$

denote the total aggregation weight assigned to retained malicious clients. When $\mu _ { t } ~ < ~ 1$ , define the renormalized gradient of the retained benign clients as

$$
\bar { \mathbf { g } } _ { t } ^ { B , \mathrm { r e t } } : = \sum _ { i \in \mathcal { R } _ { t } \cap \mathcal { C } _ { \mathrm { b e n } } } \frac { \alpha _ { t , i } } { 1 - \mu _ { t } } \nabla F _ { i } ( \mathbf { w } _ { t } ) .\tag{62}
$$

The retained-benign representativeness error is

$$
\begin{array} { r } { \chi _ { t } : = \left. \bar { \mathbf g } _ { t } ^ { B , \mathrm { r e t } } - \nabla F ( \mathbf w _ { t } ) \right. _ { 2 } . } \end{array}\tag{63}
$$

Thus, $\chi _ { t }$ measures how closely the retained benign clients, after their weights are renormalized, represent the full benign-client gradient. Let $\varrho _ { t }$ denote the retained benign local-training drift, as formally defined in Appendix $B { \cdot } C .$ Under the conditions stated there, the screening error satisfies

$$
\| \mathbf { e } _ { t } \| _ { 2 } \leq ( 1 - \mu _ { t } ) \chi _ { t } + 2 G \mu _ { t } + \varrho _ { t } .\tag{64}
$$

Consequently, a valid choice of the screening-error bound in Theorem 1 is

$$
\beta _ { t } : = ( 1 - \mu _ { t } ) \chi _ { t } + 2 G \mu _ { t } + \varrho _ { t } .\tag{65}
$$

Proposition 2 identifies three sources of screening error. The term $\mu _ { t }$ is small when malicious clients receive little retained aggregation weight. The term $\chi _ { t }$ is small when the retained benign clients remain representative of the benign population. The term $\varrho _ { t }$ is small when local training remains close to the global model. Together, these quantities determine the screening-error contribution in Theorem 1.

For a constant effective step size, the finite-time result yields a stationary-neighborhood bound:

## Corollary 2 (Stationary neighborhood with constant step size).

Suppose that $\eta _ { t } = \eta \leq 1 / ( 4 L _ { F } )$ , ∀t. Define

$$
\overline { { \beta } } _ { T } ^ { 2 } : = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \beta _ { t } ^ { 2 } ] , \qquad \overline { { \nu } } _ { T } ^ { 2 } : = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \nu _ { t } ^ { 2 } ] .\tag{66}
$$

Then

$$
\begin{array} { r l r } {  { \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \| \nabla F ( \mathbf { w } _ { t } ) \| _ { 2 } ^ { 2 } ] \leq \frac { 2 ( F ( \mathbf { w } _ { 0 } ) - F _ { \mathrm { i n f } } ) } { \eta T } } } \\ & { } & { \qquad + 2 ( 1 + L _ { F } \eta ) \overline { { \beta } } _ { T } ^ { 2 } + L _ { F } \eta \overline { { \nu } } _ { T } ^ { 2 } . } \end{array}\tag{67}
$$

Corollary 2 shows that, with a constant effective step size, the average gradient norm approaches a neighborhood whose size is controlled by the average screening error and the stochastic update variance.

Complete assumptions, proofs, the screening-error decomposition, and the local-training drift bound are provided in Appendix B.

## V. EXPERIMENTAL EVALUATION

In this section, we evaluate FEDLNS in federated transformer-based language-model training under targetcorruption attacks. The evaluation examines robustness across model architectures, IID and non-IID data partitions, and malicious-client fractions ranging from 0% to 40%.

## A. Experimental Setup

We evaluate FEDLNS on three transformer model families with different language-modeling objectives:

• A GPT-style causal transformer [50] trained from scratch on WikiText using autoregressive next-token prediction.

• A BERT-style masked-language transformer [51] trained from scratch on Tiny Shakespeare using bidirectional masked-token prediction.

• A LLaMA-style decoder-only causal transformer [52] trained from scratch on StackOverflow text.

Settings. Unless otherwise specified, each experiment uses $K = 2 0 0$ clients, a 10% participation rate, 20 clients sampled uniformly at random per communication round, and 200 communication rounds [53], [54]. Each configuration is repeated using three random seeds, {100, 200, 300}, under IID and two-shard non-IID data partitions [55].

For the K = 200 clients considered in our experiments, we use the default activation coverage $M _ { \mathrm { a c t } } ~ = ~ 1 0 0$ , so screening begins after signatures from half of the client identities have been observed.

Malicious clients. We consider a target-corruption attack in which a fixed subset of the client population is designated as malicious at the beginning of each training run and remains malicious throughout that run. For every malicious client selected in a communication round, the input sequence remains unchanged, while the target token at each supervised prediction position is independently replaced, with probability $p _ { \mathrm { c o r r } } ,$ by a randomly sampled valid vocabulary token; padding and EOS tokens are excluded from the replacement candidates. We set $p _ { \mathrm { c o r r } } ~ = ~ 1 . 0 ,$ so every eligible target position used for local training is replaced. The malicious-client fraction is defined over the complete client population. In each communication round, 20 clients are sampled uniformly without replacement from the K = 200 clients; therefore, the number of malicious clients participating in an individual round varies according to the sampled client set.

Performance metrics. We report perplexity and tokenlevel semantic entropy. For the causal language models, perplexity is the exponential of the mean next-token crossentropy. For the masked-language model, masked-token perplexity is the exponential of the mean cross-entropy over masked positions. Token-level semantic entropy is the mean predictive entropy over the evaluated token positions and measures the concentration of the corresponding predictive distributions.

For each random seed, the checkpoint with the lowest test loss is selected, and the test loss, perplexity, and tokenlevel semantic entropy are reported from that same model state. Results are reported as the mean and sample standard deviation across seeds {100, 200, 300} for each model, data partition, aggregation method, and malicious-client fraction.

![](images/39f2ad766a090dc289b1e81d65260826bb9c680cb4da938634c96fef11cffae2.jpg)  
(a) GPT/WikiText, IID partitions

![](images/965149f8d8c45ef0401d0eca140c0a373ecf128e684fd749093d60c725c6b0d7.jpg)  
(b) BERT/Shakespeare, IID partitions

![](images/fe1f9c0f168ea8c5063ad6184b3e626bdf252ef55217bc29ab73140502c26fb0.jpg)  
(c) LLaMA/StackOverflow, IID partitions

![](images/ab9e894368b746f586594246b308052b9059e2bffa88f2d71e542cad5cd5f2be.jpg)  
(d) GPT/WikiText, non-IID partitions

![](images/378d06a5f467e83b42a65612e9f46d8ec612bf1915b74e69a18e748ba388b8db.jpg)  
(e) BERT/Shakespeare, non-IID partitions

![](images/60b111d1b70134c442e2c61d928501094c34332f191faf621bd3d9938e0e977c.jpg)  
(f) LLaMA/StackOverflow, non-IID partitions  
Fig. 2: Perplexity (lower is better) under target manipulation affecting 40% of the client population. Bars show the mean over three random seeds, and error bars show one sample standard deviation. FEDLNS consistently shows the lowest perplexity, across both IID and non-IID data partitions.

Baselines. We compare FEDLNS with six representative aggregation methods covering conventional averaging, norm-constrained aggregation, coordinate-wise robust statistics, distance-based selection, and clustering-based filtering. FedAvg performs sample-weighted averaging and serves as the non-robust baseline [56]. Norm-Bounded FedAvg clips client updates before aggregation to limit the effect of large update magnitudes [16]. Coordinate-wise Median and Trimmed Mean apply robust statistics independently to each model coordinate [15]. Multi-Krum selects updates with mutually consistent distance profiles [14]. FLAME combines clustering, norm clipping, and Gaussian noise injection [17]. Trimmed Mean and Multi-Krum are configured using the true population-level malicious-client fraction, whereas FEDLNS does not require this fraction as an input.

Appendix C provides the complete dataset, tokenization, partition, architecture, optimization, attack, baseline, and FEDLNS configurations; Appendix D reports the full malicious-client-fraction results; and Appendices E and F provide the bank-coverage-at-activation and GMMselection ablations, respectively.

## B. Perplexity and Token-Level Semantic Entropy

Figure 2 compares the perplexity of FEDLNS and the six baselines when 40% of the client population is malicious. Figures 2(a)–2(c) report the IID results, whereas Figures 2(d)–2(f) report the non-IID results. Error bars show one sample standard deviation across seeds {100, 200, 300}.

Under IID data partitions, FEDLNS achieves the lowest perplexity for all three evaluated model families. Relative to the strongest competing baseline, it reduces perplexity by 11.59% for GPT/WikiText (from 1449.72 to 1281.63), 3.88% for BERT/Tiny Shakespeare (from 725.76 to 697.62), and 4.98% for LLaMA/StackOverflow (from 228.52 to 217.14). These results show that the global models trained with FEDLNS exhibit less attack-induced degradation across all three architectures.

Under the two-shard non-IID partitions, FEDLNS reduces perplexity by 18.57% for GPT/WikiText (from 1579.81 to 1286.50), 10.05% for BERT/Tiny Shakespeare (from 731.73 to 658.21), and 5.67% for LLaMA/StackOverflow (from 233.42 to 220.18) relative to the strongest non-FEDLNS baseline. The improvement across all three non-IID settings shows that the downstream benefit of the screening procedure persists when benign clients have heterogeneous local data distributions.

![](images/e426187fca0f4491b09a1753921c0a745e2c71ac9f4830432ebde04331a7e7c2.jpg)  
(a) GPT/WikiText, IID partitions

![](images/9a2329c1bc89461fe834fb72b596172c144ea2207323af55134710b4ef840152.jpg)  
(b) BERT/Shakespeare, IID partitions

![](images/8dd47c53e18c03a75189b0e79fbdd5edd695e8df94d3555322b2d85ba8f9fb19.jpg)  
(c) LLaMA/StackOverflow, IID partitions

![](images/7f9ce8eabbfd68ebe8835a74f6e50bfa449e03a37f4f4e9a47476cd89674645e.jpg)  
(d) GPT/WikiText, non-IID partitions

![](images/38f9156de9ce8eaadbb94bd5cf623b3def597122c1f00b7a3ad2aab87accdf7c.jpg)  
(e) BERT/Shakespeare, non-IID partitions

![](images/3edfe9d05b17a4e7272ba82780dad6ac1952d58b714f6b696bec20345deee213.jpg)  
(f) LLaMA/StackOverflow, non-IID partitions  
Fig. 3: Token-level semantic entropy (lower is better) under target manipulation affecting 40% of the client population. Bars show the mean over three random seeds, and error bars show one sample standard deviation. FEDLNS consistently shows the lowest semantic entropy, across both IID and non-IID data partitions.

Because lower entropy does not necessarily imply greater correctness, we interpret the entropy results jointly with perplexity.

Figure 3 reports token-level semantic entropy for the three model families under 40% population-level target manipulation. Under IID data partitions, Figures 3(a)–3(c) show that FEDLNS achieves the lowest entropy in all three settings. Relative to the strongest non-FEDLNS baseline, it reduces entropy by 2.30% for GPT/WikiText (from 6.96 to 6.80), 3.41% for BERT/Tiny Shakespeare (from 6.46 to 6.24), and 1.60% for LLaMA/StackOverflow (from 5.61 to 5.52).

Under non-IID data partitions, Figures 3(d)–3(f) show reductions of 2.95% for GPT/WikiText (from 7.13 to 6.92), 6.50% for BERT/Tiny Shakespeare (from 6.62 to 6.19), and 1.95% for LLaMA/StackOverflow (from 5.63 to 5.52). Together with the corresponding perplexity reductions, these results show that the global models obtained after FEDLNS screening achieve lower test loss and lower uncertainty over competing token predictions under the evaluated attack.

## C. Impact of Malicious Clients

Appendix D reports the complete numerical results across malicious-client fractions for all model families and data partitions. For GPT/WikiText under IID partitioning, the population-level malicious-client fraction increases from 0% to 40%.

For perplexity, FEDLNS exhibits limited degradation as the malicious-client fraction increases. Its perplexity rises by 2.82%, from 1246.57 at 0% to 1281.63 at 40%. Over the same range, perplexity increases by 27.97% for FedAvg, 48.32% for Norm-Bounded FedAvg, 17.95% for Median, 36.05% for Trimmed Mean, 16.72% for Multi-Krum, and 15.15% for FLAME. FEDLNS achieves the lowest perplexity at every nonzero attack fraction from 10% to 40%.

Token-level semantic entropy exhibits a similar attackedsetting trend. Although Norm-Bounded FedAvg has the lowest entropy in the attack-free setting, its entropy increases by 47.94% between 0% and 40% malicious clients. In contrast, the entropy of FEDLNS increases by 1.04%, from 6.73 to 6.80, and is the lowest among the evaluated methods for every nonzero attack fraction. The simultaneous stability of perplexity and entropy indicates that FEDLNS limits both predictive degradation and uncertainty as the target-corruption rate increases.

## D. Impact of Bank Coverage at Activation

As defined by Eq. (11), FEDLNS begins screening and global aggregation after the bank reaches the activation coverage $M _ { \mathrm { a c t } }$ . We evaluate $M _ { \mathrm { a c t } } ~ \in ~ \{ 0 , 5 0 , 1 0 0 , 1 5 0 \}$ corresponding to immediate, 25%, 50%, and 75% population coverage. Setting $M _ { \mathrm { { a c t } } } ~ = ~ 0$ activates screening immediately using the currently available bank; it does not remove the bank. In all settings, the bank continues growing after activation as additional client identities are observed.

The ablation reports performance over the first ten active aggregation rounds under IID partitioning with 40% malicious clients. Relative to immediate activation with $M _ { \mathrm { a c t } } = 0 .$ , activation at $M _ { \mathrm { a c t } } = 1 5 0$ reduces perplexity by 10.62% for GPT/WikiText, 31.89% for BERT/Tiny Shakespeare, and 74.68% for LLaMA/StackOverflow. Tokenlevel semantic entropy decreases by 2.53%, 20.20%, and 25.33%, respectively. Thus, the main ablation exhibits a monotonic perplexity improvement as activation coverage increases.

These results support maximizing bank coverage whenever the resulting activation delay is acceptable. A broader bank provides a more representative cross-client reference, and the bank should continue growing toward full population coverage after screening begins. The activation threshold is therefore a practical starting criterion rather than a desired final bank size. Smaller activation coverages remain usable when waiting for broader coverage would delay training excessively, but they should not be interpreted as equivalent to a fully populated bank. The largest evaluated activation coverage is 75%; the experiments therefore support the observed benefit of increasing coverage within this range rather than establishing an empirical optimum at 100% coverage.

The GMM-selection ablation shows that BICselected one-vs-two-component screening and forced two-component screening produce nearly identical selected-checkpoint results in the evaluated settings. BIC retains the option to keep all participating clients when the one-component model is preferred, but the endpoint metrics exhibit limited sensitivity to this model-selection rule. Complete results are provided in Appendix F.

## VI. CONCLUSION

This paper introduced FEDLNS, a lightweight serverside framework that develops a normalization-layer update signature to screen malicious updates before standard fullmodel aggregation. Moreover, a server-side retain rule was designed, which constructs a robust cross-client reference from the latest observed signatures and combines median/MAD standardization with BIC-guided mixture modeling. For GPT-style, BERT-style, and LLaMA-style models under both IID and non-IID partitions, FEDLNS achieved lower perplexity than all six baselines when 40% of the client population was malicious. The maximum reductions were 18.57% in perplexity and 6.50% in token-level semantic entropy.

## ACKNOWLEDGEMENTS

This work was in part supported by the Fundac¸ao para a˜ Ciencia e a Tecnologia (Portuguese Foundation for Scienceˆ

and Technology) through the Carnegie Mellon Portugal Program.

## REFERENCES

[1] H. Cai, H. Dong, H. Wang, K. Li, and O. B. Akan, “Graph representation-based model poisoning on federated LLMs in CyberEdge networks,” arXiv preprint arXiv:2507.01694, 2025.

[2] H. Li, H. Madhukumar, N. Methley et al., “Future factories with 6G: Agentic AI and cyber–physical digital twins,” IEEE Internet of Things Journal, vol. 13, no. 9, pp. 17 990–18 006, 2026.

[3] X. An, L. Shen, Y. Luo, H. Hu, and D. Tao, “Adaptive batch size time evolving stochastic gradient descent for federated learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 2, pp. 1158–1170, 2026.

[4] Y. Cheng, W. Zhang, Z. Zhang et al., “Toward federated large language models: Motivations, methods, and future directions,” IEEE Communications Surveys and Tutorials, vol. 27, no. 4, pp. 2733– 2764, 2025.

[5] K. Li, Z. Zhang, A. Pourkabirian, W. Ni, F. Dressler, and O. B. Akan, “Towards resilient federated learning in CyberEdge networks: Recent advances and future trends,” arXiv preprint arXiv:2504.01240, 2025.

[6] J. Hu, D. Wang, Z. Wang et al., “Federated large language model: Solutions, challenges and future directions,” IEEE Wireless Communications, 2024.

[7] Y. Yao, J. Zhang, J. Wu et al., “Federated large language models: Current progress and future directions,” arXiv preprint arXiv:2409.15723, 2024.

[8] Z. Wang, Y. Zhou, Y. Shi, and K. B. Letaief, “Federated fine-tuning for pre-trained foundation models over wireless networks,” IEEE Transactions on Wireless Communications, 2025.

[9] T. Webb, S. S. Mondal, and I. Momennejad, “A brain-inspired agentic architecture to improve planning with LLMs,” Nature Communications, vol. 16, no. 1, p. 8633, 2025.

[10] H. Yang, J. Chen, M. Siew, T. Lorido-Botran, and C. Joe-Wong, “LLM-powered decentralized generative agents with adaptive hierarchical knowledge graph for cooperative planning,” arXiv preprint arXiv:2502.05453, 2025.

[11] P. Wu, T. Imbiriba, and P. Closas, “A Bayesian framework for clustered federated learning,” IEEE Transactions on Pattern Analysis & Machine Intelligence, vol. 48, no. 03, pp. 3471–3481, 2026.

[12] H. Liu, W. Xue, Y. Chen et al., “A survey on hallucination in large vision-language models,” arXiv preprint arXiv:2402.00253, 2024.

[13] Z. Bai, P. Wang, T. Xiao, T. He, Z. Han, Z. Zhang, and M. Z. Shou, “Hallucination of multimodal large language models: A survey,” arXiv preprint arXiv:2404.18930, 2024.

[14] P. Blanchard, E. M. El Mhamdi, R. Guerraoui, and J. Stainer, “Machine learning with adversaries: Byzantine tolerant gradient descent,” Advances in Neural Information Processing Systems, vol. 30, 2017.

[15] D. Yin, Y. Chen, R. Kannan, and P. Bartlett, “Byzantine-robust distributed learning: Towards optimal statistical rates,” in Proceedings of the 35th International Conference on Machine Learning. PMLR, 2018, pp. 5650–5659.

[16] Z. Sun, P. Kairouz, A. T. Suresh, and H. B. McMahan, “Can you really backdoor federated learning?” arXiv preprint arXiv:1911.07963, 2019.

[17] T. D. Nguyen, P. Rieger, H. Chen et al., “FLAME: Taming backdoors in federated learning,” in 31st USENIX Security Symposium (USENIX Security 22), 2022, pp. 1415–1432.

[18] Z. Zhang, Q. Su, and X. Sun, “Dim-krum: Backdoor-resistant federated learning for nlp with dimension-wise krum-based aggregation,” in Findings of the Association for Computational Linguistics: EMNLP 2022, 2022, pp. 339–354.

[19] M. Tao, Y. Tian, W. Tu, Y. Yang, X. Yang, and X. Tang, “Safefedllm: Delving into the safety of federated large language models,” in Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2026, pp. 24 405–24 420.

[20] J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer normalization,” arXiv preprint arXiv:1607.06450, 2016.

[21] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in Neural Information Processing Systems, vol. 30, 2017.

[22] B. Zhang and R. Sennrich, “Root mean square layer normalization,” in Advances in Neural Information Processing Systems, vol. 32, 2019.

[23] R. Xiong, Y. Yang, D. He, K. Zheng, S. Zheng, C. Xing, H. Zhang, Y. Lan, L. Wang, and T.-Y. Liu, “On layer normalization in the transformer architecture,” in Proceedings of the 37th International Conference on Machine Learning. PMLR, 2020, pp. 10 524–10 533.

[24] Z. Zhang, X. Cao, J. Jia, and N. Z. Gong, “FLDetector: Defending federated learning against model poisoning attacks via detecting malicious clients,” in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2022, pp. 2545– 2555.

[25] X. Cao, M. Fang, J. Liu, and N. Z. Gong, “FLTrust: Byzantinerobust federated learning via trust bootstrapping,” in Network and Distributed System Security Symposium (NDSS), 2021.

[26] X. Cao, Z. Zhang, J. Jia, and N. Z. Gong, “FLCert: Provably secure federated learning against poisoning attacks,” IEEE Transactions on Information Forensics and Security, vol. 17, pp. 3691–3705, 2022.

[27] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. de Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly, “Parameter-efficient transfer learning for NLP,” in Proceedings of the 36th International Conference on Machine Learning. PMLR, 2019, pp. 2790–2799.

[28] E. Ben Zaken, Y. Goldberg, and S. Ravfogel, “BitFit: Simple parameter-efficient fine-tuning for transformer-based masked language models,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), 2022, pp. 1–9.

[29] H. Liu, D. Tam, M. Muqeeth, J. Mohta, T. Huang, M. Bansal, and C. A. Raffel, “Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 1950–1965.

[30] W. Qi, Y.-P. Ruan, Y. Zuo, and T. Li, “Parameter-efficient tuning on layer normalization for pre-trained language models,” arXiv preprint arXiv:2211.08682, 2022.

[31] S. Brody, U. Alon, and E. Yahav, “On the expressivity role of LayerNorm in transformers’ attention,” in Findings of the Association for Computational Linguistics: ACL 2023, 2023, pp. 14 211–14 221.

[32] X. Fang, Z. Huang, Z. Tian, M. Fang, Z. Pan, Q. Fang, Z. Wen, H. Pan, and D. Li, “Zero-resource hallucination detection for text generation via graph-based contextual knowledge triples modeling,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 22, 2025, pp. 23 868–23 877.

[33] N. Nonkes, S. Agaronian, E. Kanoulas, and R. Petcu, “Leveraging graph structures to detect hallucinations in large language models,” arXiv preprint arXiv:2407.04485, 2024.

[34] X. Guan, Y. Liu, H. Lin, Y. Lu, B. He, X. Han, and L. Sun, “Mitigating large language model hallucinations via autonomous knowledge graph-based retrofitting,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 16, 2024, pp. 18 126–18 134.

[35] K. Furumai, Y. Wang, M. Shinohara, K. Ikeda, Y. Yu, and T. Kato, “Detecting dialogue hallucination using graph neural networks,” in 2023 International Conference on Machine Learning and Applications (ICMLA). IEEE, 2023, pp. 871–877.

[36] S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal, “Detecting hallucinations in large language models using semantic entropy,” Nature, vol. 630, no. 8017, pp. 625–630, 2024.

[37] Y. Chen, Q. Fu, Y. Yuan et al., “Hallucination detection: Robustly discerning reliable answers in large language models,” in Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, 2023, pp. 245–255.

[38] G. Sriramanan, S. Bharti, V. S. Sadasivan, S. Saha, P. Kattakinda, and S. Feizi, “LLM-Check: Investigating detection of hallucinations in large language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 34 188–34 216, 2024.

[39] X. Zhou, M. Zhang, Z. Lee, W. Ye, and S. Zhang, “HADEMIF: Hallucination detection and mitigation in large language models,” in The Thirteenth International Conference on Learning Representations, 2025.

[40] L. Guo, Y. Fang, F. Chen, P. Liu, and S. Xu, “Large language models with adaptive token fusion: A novel approach to reducing hallucinations and improving inference efficiency,” Authorea, 2024.

[41] X. Chen, C. Wang, Y. Xue, N. Zhang, X. Yang, Q. Li, Y. Shen, L. Liang, J. Gu, and H. Chen, “Unified hallucination detection for multimodal large language models,” arXiv preprint arXiv:2402.03190, 2024.

[42] A. Gunjal, J. Yin, and E. Bas, “Detecting and preventing hallucinations in large vision-language models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 16, 2024, pp. 18 135–18 143.

[43] W. Su, C. Wang, Q. Ai, Y. Hu, Z. Wu, Y. Zhou, and Y. Liu, “Unsupervised real-time hallucination detection based on the internal states of large language models,” arXiv preprint arXiv:2403.06448, 2024.

[44] C. Chen, K. Liu, Z. Chen, Y. Gu, Y. Wu, M. Tao, Z. Fu, and J. Ye, “INSIDE: LLMs’ internal states retain the power of hallucination detection,” arXiv preprint arXiv:2402.03744, 2024.

[45] X. Du, C. Xiao, and S. Li, “HaloScope: Harnessing unlabeled LLM generations for hallucination detection,” Advances in Neural Information Processing Systems, vol. 37, pp. 102 948–102 972, 2024.

[46] Y. Liu, G. Liu, R. Zhang, D. Niyato, Z. Xiong, D. I. Kim, K. Huang, and H. Du, “Hallucination-aware optimization for large language model-empowered communications,” arXiv preprint arXiv:2412.06007, 2024.

[47] H. Liu, X. Geng, L. Lee, I. Mordatch, S. Levine, S. Narang, and P. Abbeel, “Towards better few-shot and finetuning performance with forgetful causal language models,” arXiv preprint arXiv:2210.13432, 2022.

[48] K. N. Kumar, C. K. Mohan, and L. R. Cenkeramaddi, “The impact of adversarial attacks on federated learning: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 5, pp. 2672–2691, 2023.

[49] S. Xu, Y. Qian, and R. Q. Hu, “Data-driven edge intelligence for robust network anomaly detection,” IEEE Transactions on Network Science and Engineering, vol. 7, no. 3, pp. 1481–1492, 2019.

[50] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al., “Language models are unsupervised multitask learners,” OpenAI blog, vol. 1, no. 8, p. 9, 2019.

[51] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pretraining of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), 2019, pp. 4171–4186.

[52] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar\` et al., “LLaMA: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971, 2023.

[53] Y. Sun, L. Shen, H. Sun, L. Ding, and D. Tao, “Efficient federated learning via local adaptive amended optimizer with linear speedup,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 12, pp. 14 453–14 464, 2023.

[54] Y. Kang and B. Li, “Polaris: Accelerating asynchronous federated learning with client selection,” IEEE Transactions on Cloud Computing, vol. 12, no. 2, pp. 446–458, 2024.

[55] J. Dong, H. Li, Y. Cong, G. Sun, Y. Zhang, and L. Van Gool, “No one left behind: Real-world federated class-incremental learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 4, pp. 2054–2070, 2023.

[56] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Proceedings of the 20th International Conference on Artificial Intelligence and Statistics. PMLR, 2017, pp. 1273–1282.

## APPENDIX CONTENTS

Appendix A: Probabilistic Analysis of Partial Bank Coverage at Activation 15   
A-A A Hypergeometric Model of Bank Contamination . 15   
A-B Concentration of the Malicious Fraction in the Bank 15   
A-C Time to Reach the Activation Coverage 17   
Appendix B: Proofs for Theoretical Analysis 18   
B-A Proof of the Normalization-Parameter Margin Decomposition 18   
B-B Proof of the Screened-Aggregation Bound . 20   
B-C Screening-Error Decomposition 23   
B-D Local-Training Drift Bound . 24   
Appendix C: Detailed Experimental Settings 26   
C-A Federated Learning Protocol 26   
C-B Datasets and Tokenization . 26   
C-C Model Architectures and Optimization . 27   
C-D Attack and Aggregation-Baseline Configuration 28   
C-E FedLNS Implementation Details 29   
Appendix D: Complete Results Across Malicious-Client Fractions 30   
D-A GPT-Style Causal Language Modeling on WikiText . . 30   
D-B BERT-Style Masked Language Modeling on Tiny Shakespeare 32   
D-C LLaMA-Style Causal Language Modeling on StackOverflow . 34   
Appendix E: Bank Coverage at Activation 36   
E-A Reporting Protocol . 36   
E-B Activation Rounds . 36   
E-C Results Under IID Partitions 38   
E-D Results Under Non-IID Partitions 41   
E-E Interpretation . 41   
Appendix F: GMM-Selection Ablation 45   
F-A Results Under IID Partitions 45   
F-B Results Under Non-IID Partitions 48   
F-C Interpretation . 48

## APPENDIX A

## PROBABILISTIC ANALYSIS OF PARTIAL BANK COVERAGE AT ACTIVATION

FEDLNS ideally benefits from the broadest available client coverage, and the history bank continues expanding after screening begins. Waiting for full population coverage before beginning aggregation may nevertheless be impractically slow. We therefore use $M _ { \mathrm { a c t } }$ to denote a partial bank coverage at which screening is permitted to start.

This appendix analyzes, under uniform random participation, the probability that malicious clients form a majority of a partially populated bank and the number of rounds required to reach a chosen activation coverage. The analysis relies on the 50% breakdown point of the coordinate-wise median: the median/MAD reference remains in its benignmajority regime when malicious identities constitute fewer than half of the bank. The resulting bound provides a reliability condition for early activation; it does not imply that a partial bank has the same statistical quality as a fully populated bank.

## A. A Hypergeometric Model of Bank Contamination

Let

$$
N _ { t } ^ { \mathrm { b a n k } } : = | \mathcal { H } _ { t } |\tag{68}
$$

denote the number of distinct client identities represented in the refreshed history bank at round t.

Let $\mathcal { C } _ { \mathrm { m a l } }$ be the set of malicious clients, with

$$
| { \mathcal C } _ { \mathrm { m a l } } | = \alpha K , \qquad 0 \le \alpha < \frac { 1 } { 2 } ,\tag{69}
$$

where α is the global malicious-client fraction (if αK is not an integer, the same discussion applies with $\lfloor \alpha K \rfloor \thinspace \mathrm { o r } \ \lceil \alpha K \rceil$ malicious clients).

Let $M _ { t } : = | \mathcal { H } _ { t } \cap \mathcal { C } _ { \mathrm { m a l } } |$ denote the number of malicious client identities represented in the bank, so the bank has a benign majority when $\dot { M _ { t } } < N _ { t } ^ { \mathrm { b a n k } } / 2$ . The undesirable event is

$$
\mathcal { E } _ { t } : = \left\{ M _ { t } \geq \frac { N _ { t } ^ { \mathrm { b a n k } } } { 2 } \right\} .\tag{70}
$$

FEDLNS uses random client participation, where the server samples participating clients, in each communication round, uniformly at random from the population of K clients. Because the bank stores distinct client IDs, once a client has appeared, the bank holds that client’s most recent signature. Conditioned on the bank’s membership size $N _ { t } ^ { \mathrm { b a n k } } = b ,$ the exchangeability of the sampling process implies that the bank can be treated as a uniformly random subset of size b drawn from K clients. Particularly, the server’s sampling rule does not prefer any client ID; any two subsets of the same size are equally likely after conditioning on the number of distinct observed clients. Since the malicious set $\mathcal { C } _ { \mathrm { m a l } }$ is fixed at the start of the run, the question becomes how many of a random size-b subset, drawn without replacement from K clients, belong to a fixed subset of size αK, which is the hypergeometric setting. Hence, conditioned on $N _ { t } ^ { \mathrm { b a n k } } = b .$ , we have

$$
M _ { t } \mid N _ { t } ^ { \mathrm { b a n k } } = b \sim \mathrm { H y p e r g e o m e t r i c } ( K , \alpha K , b ) ,\tag{71}
$$

i.e., for feasible $m$

$$
\operatorname* { P r } ( M _ { t } = m \mid N _ { t } ^ { \mathrm { b a n k } } = b ) = { \frac { { \binom { \alpha K } { m } } { \binom { ( 1 - \alpha ) K } { b - m } } } { \binom { K } { b } } } .\tag{72}
$$

The conditional expectation is $\mathbb { E } [ M _ { t } \ | \ N _ { t } ^ { \mathrm { b a n k } } = b ] = \alpha b$ , so when $\alpha < 1 / 2$ the expected malicious fraction in the bank stays below the median’s 50% breakdown point.

## B. Concentration of the Malicious Fraction in the Bank

For a hypergeometric random variable, a Chernoff-type upper-tail bound gives, for any threshold $\gamma > \alpha$

$$
\operatorname* { P r } \big ( \frac { M _ { t } } { N _ { t } ^ { \mathrm { b a n k } } } \ge \gamma \big | N _ { t } ^ { \mathrm { b a n k } } \big ) \le \exp \left( - N _ { t } ^ { \mathrm { b a n k } } D ( \gamma \| \alpha ) \right) ,\tag{73}
$$

where $D ( p \| q )$ is the Bernoulli KL divergence

$$
D ( p \| q ) = p \log { \frac { p } { q } } + ( 1 - p ) \log { \frac { 1 - p } { 1 - q } } .\tag{74}
$$

Setting $\gamma = 1 / 2$ and using $\alpha < 1 / 2$ gives

$$
\operatorname* { P r } \left( M _ { t } \geq \frac { N _ { t } ^ { \mathrm { b a n k } } } { 2 } \biggm | N _ { t } ^ { \mathrm { b a n k } } \right) \leq \exp \left( - N _ { t } ^ { \mathrm { b a n k } } D \left( \frac { 1 } { 2 } \biggm | \bigg | \alpha \right) \right) ,\tag{75}
$$

with

$$
D \left( { \frac { 1 } { 2 } } { \Bigg \| } \alpha \right) = { \frac { 1 } { 2 } } \log { \frac { 1 / 2 } { \alpha } } + { \frac { 1 } { 2 } } \log { \frac { 1 / 2 } { 1 - \alpha } }\tag{76}
$$

or, equivalently,

$$
D \left( \frac { 1 } { 2 } \bigg | \bigg | \alpha \right) = \frac { 1 } { 2 } \log \frac { 1 } { 4 \alpha ( 1 - \alpha ) } .\tag{77}
$$

This quantity is positive for $\alpha < 1 / 2$ , so the upper bound on the probability of a malicious-majority bank decays exponentially with $N _ { t } ^ { \mathrm { b a n k } }$ . This provides the probabilistic motivation for delaying screening until the history bank reaches a sufficiently broad client coverage under random participation.

## Proposition 3 (Bound on a malicious-majority history bank).

Suppose that the fraction of malicious clients satisfies $\alpha < 1 / 2$ , and let M denote the number of malicious signatures in a history bank of size $N _ { t } ^ { \mathrm { b a n k } }$ . Then the probability that malicious signatures constitute at least half of the bank is bounded by

$$
\operatorname* { P r } \left( M _ { t } \geq \frac { N _ { t } ^ { \mathrm { b a n k } } } { 2 } \biggm | N _ { t } ^ { \mathrm { b a n k } } \right) \leq \exp \left( - 2 N _ { t } ^ { \mathrm { b a n k } } \left( \frac { 1 } { 2 } - \alpha \right) ^ { 2 } \right) .\tag{78}
$$

Proof: From the KL-based concentration bound in (75), we have

$$
\operatorname* { P r } \left( M _ { t } \geq \frac { N _ { t } ^ { \mathrm { b a n k } } } { 2 } \bigg | N _ { t } ^ { \mathrm { b a n k } } \right) \leq \exp \left( - N _ { t } ^ { \mathrm { b a n k } } D \left( \frac { 1 } { 2 } \bigg | \bigg | \alpha \right) \right) .\tag{79}
$$

To obtain a simpler sufficient bound, let

$$
g : = \frac { 1 } { 2 } - \alpha\tag{80}
$$

be the margin between the median breakdown point and the attack fraction. Then

$$
\alpha ( 1 - \alpha ) = \left( { \frac { 1 } { 2 } } - g \right) \left( { \frac { 1 } { 2 } } + g \right) = { \frac { 1 } { 4 } } - g ^ { 2 } .\tag{81}
$$

Therefore,

$$
4 \alpha ( 1 - \alpha ) = 1 - 4 g ^ { 2 } .\tag{82}
$$

Using (77), we obtain

$$
D \left( \frac { 1 } { 2 } \bigg | \bigg | \alpha \right) = \frac { 1 } { 2 } \log \frac { 1 } { 1 - 4 g ^ { 2 } } .\tag{83}
$$

Since log $\textstyle { \frac { 1 } { 1 - x } } \geq x$ for $0 \leq x < 1$ , it follows that

$$
D \left( { \frac { 1 } { 2 } } { \bigg \| } \alpha \right) \geq { \frac { 1 } { 2 } } \cdot 4 g ^ { 2 } = 2 g ^ { 2 } = 2 \left( { \frac { 1 } { 2 } } - \alpha \right) ^ { 2 } .\tag{84}
$$

Substituting (84) into (75) yields

$$
\operatorname* { P r } \left( M _ { t } \geq \frac { N _ { t } ^ { \mathrm { b a n k } } } { 2 } \biggm | N _ { t } ^ { \mathrm { b a n k } } \right) \leq \exp \left( - 2 N _ { t } ^ { \mathrm { b a n k } } \left( \frac { 1 } { 2 } - \alpha \right) ^ { 2 } \right) ,\tag{85}
$$

which proves (78).

The bound in Proposition 3 makes the dependence on attack strength explicit. As $\alpha  1 / 2$ , the margin $g = 1 / 2 - \alpha$ shrinks, the exponent weakens, and a larger history bank is required to achieve the same confidence level. In contrast, when $\alpha \ll 1 / 2$ , the same bank size already provides a strong guarantee against a malicious-majority bank.

Suppose screening must begin before full population coverage is available and the desired probability of a maliciousmajority activation bank is at most η. From (75), it suffices to choose $M _ { \mathrm { { a c t } } }$ such that $\exp ( - M _ { \mathrm { a c t } } D ( 1 / 2 \| \alpha ) ) \leq \eta ,$

$$
M _ { \mathrm { a c t } } \geq \frac { \log ( 1 / \eta ) } { D \left( \frac { 1 } { 2 } \Vert \alpha \right) } .\tag{86}
$$

Using the lower bound (84), a more conservative sufficient condition is

$$
M _ { \mathrm { a c t } } \geq \frac { \log ( 1 / \eta ) } { 2 \left( \frac { 1 } { 2 } - \alpha \right) ^ { 2 } } .\tag{87}
$$

When the total number of clients K is fixed, the activation coverage can be expressed as a population fraction $M _ { \mathrm { a c t } } = \rho K$ in which case (75) becomes

$$
\operatorname* { P r } \left( M _ { t } \geq \frac { N _ { t } ^ { \mathrm { b a n k } } } { 2 } \biggm | N _ { t } ^ { \mathrm { b a n k } } \geq \rho K \right) \leq \exp \left( - \rho K D \left( \frac { 1 } { 2 } \biggm \| \alpha \right) \right) .\tag{88}
$$

TABLE I: Expected round at which the bank reaches each coverage threshold under random participation with $K = 2 0 0$ and participation rate $q = 0 . 1$
<table><tr><td>Bank coverage ρ</td><td> $\mathbf { \overline { { \mathit { M } } } } _ { \mathrm { a c t } } = \rho K$ </td><td>Expected activation round</td></tr><tr><td>25%</td><td>50</td><td>3</td></tr><tr><td>50%</td><td>100</td><td>7</td></tr><tr><td>75%</td><td>150</td><td>14</td></tr></table>

## C. Time to Reach the Activation Coverage

We now estimate how quickly the bank reaches a chosen activation coverage $M _ { \mathrm { { a c t } } }$ under uniform random participation. Let $q$ be the per-round participation rate, so that in each round approximately $q K$ clients are selected uniformly at random. For a fixed client, the probability of not being selected in one round is $1 - q ,$ , so the probability of not appearing after r rounds is $( 1 - q ) ^ { r }$ , and the probability of having appeared at least once by round r is $1 - ( 1 - q ) ^ { r }$ . The expected number of distinct clients in the bank after r rounds is therefore

$$
\mathbb { E } [ B _ { r } ] = K \left( 1 - ( 1 - q ) ^ { r } \right) .\tag{89}
$$

For a target coverage fraction $\rho ,$ the expected bank size reaches $\rho K$ once $K ( 1 - ( 1 - q ) ^ { r } ) \geq \rho K , { \mathrm { i . e . , ~ } } ( 1 - q ) ^ { r } \leq 1 - \rho ,$ which gives

$$
r \geq { \frac { \log ( 1 - \rho ) } { \log ( 1 - q ) } } .\tag{90}
$$

In our experiments, where $q = 0 . 1$ and $K = 2 0 0$ , Table I presents the expected round at which the bank reaches each coverage threshold. This expectation can be complemented with a tail bound on the probability that the bank has not yet reached the threshold. Let $\mu _ { r } : = \mathbb { E } [ B _ { r } ] = K ( 1 - ( 1 - q ) ^ { r } )$ . Because the coverage indicators are exchangeable and negatively associated under uniform random sampling, a Chernoff-style lower-tail bound gives

$$
\mathrm { P r } ( B _ { r } < M _ { \mathrm { a c t } } ) \leq \exp \big ( - \frac { \mu _ { r } \delta _ { r } ^ { 2 } } { 2 } \big ) , \delta _ { r } : = 1 - \frac { M _ { \mathrm { a c t } } } { \mu _ { r } } ,\tag{91}
$$

whenever $\mu _ { r } > M _ { \mathrm { a c t } }$ . For example, with $K = 2 0 0 , q = 0 . 1$ , and $M _ { \mathrm { a c t } } = 1 0 0$ , after $r = 1 2$ rounds $\mu _ { 1 2 } = 2 0 0 ( 1 { - } 0 . 9 ^ { 1 2 } )$ ≈ 143.5, so $\delta _ { 1 2 } = 1 - 1 0 0 / 1 4 3 . 5 \approx 0 . 3 0 3$ and

$$
\mathrm { P r } ( B _ { 1 2 } < 1 0 0 ) \le \exp \big ( - \frac { 1 4 3 . 5 \cdot 0 . 3 0 3 ^ { 2 } } { 2 } \big ) \approx 1 . 4 \times 1 0 ^ { - 3 } .\tag{92}
$$

Under the participation rate used in our experiments, the bank therefore reaches the half-population threshold early with high probability.

The activation-time and bank-composition analyses above assume uniform random client participation, matching the sampling procedure used in the experiments. In deployments with nonuniform or intermittent participation, some client identities may appear rarely or never within the training horizon, making full bank coverage impractically slow. The hypergeometric and activation-time bounds do not directly apply to such participation processes. In that setting, $M _ { \mathrm { { a c t } } }$ should be interpreted as a practical early-activation coverage, while the bank should continue to expand whenever previously unseen clients participate.

## APPENDIX B

## PROOFS FOR THEORETICAL ANALYSIS

This appendix provides the assumptions and proofs supporting Section IV. Appendix B-A proves the normalizationparameter margin decomposition, Appendix B-B proves the screened-aggregation and stationarity bounds, Appendix B-C decomposes the screening error, and Appendix B-D bounds local-training drift.

## A. Proof of the Normalization-Parameter Margin Decomposition

This proof establishes the first-order contribution of a client’s normalization-parameter update to output-token margins. We proceed in three steps.

First, we analyze the margin change for one fixed prompt x. Second, we split the client update into two parts: the normalization-parameter part and the remaining-parameter part. Third, we average over prompts $x \sim \mathcal { X }$ , because the expected margin change in (41) is defined over a prompt distribution.

We use two remainder terms. The term

$$
\mathcal { R } _ { t , i } ( x , u , v )\tag{93}
$$

is the pointwise Taylor remainder. It is the Taylor approximation error for one fixed prompt x and token pair $( u , v )$ . The term

$$
R _ { t , i } ( u , v )\tag{94}
$$

is the prompt-averaged remainder, defined by

$$
R _ { t , i } ( u , v ) : = \mathbb { E } _ { x \sim \mathcal { X } } \left[ \mathscr { R } _ { t , i } ( x , u , v ) \right] .\tag{95}
$$

Thus, $\mathcal { R } _ { t , i } ( x , u , v )$ depends on a specific prompt, while $R _ { t , i } ( u , v )$ averages this error over prompts.

Assumption 1 (Smooth prompt-dependent margins).

Fix a token pair $( u , v ) \in \mathcal { V } \times \mathcal { V } .$ . For every prompt x in the support of X , the mapping

$$
\mathbf { w } \mapsto m _ { u , v } ( x ; \mathbf { w } )\tag{96}
$$

is twice continuously differentiable in a neighborhood of $\mathbf { w } _ { t } .$ Moreover, there exists $L _ { m } > 0$ such that

$$
\left\| \nabla _ { \mathbf { w } } ^ { 2 } m _ { u , v } ( x ; \mathbf { w } ) \right\| _ { \mathrm { o p } } \leq L _ { m }\tag{97}
$$

for all w in that neighborhood and all relevant prompts x.

Conditioned on round t and client i, the update

$$
\Delta \mathbf { w } _ { t , i } = ( \Delta \mathbf { p } _ { t , i } , \Delta \mathbf { q } _ { t , i } )\tag{98}
$$

is treated as fixed with respect to the prompt draw $x \sim \mathcal { X }$ . Thus, the expectation over x averages prompt-dependent gradients and remainders, while the update vectors $\Delta \mathbf { p } _ { t , i }$ and $\Delta \mathbf q _ { t , i }$ do not depend on the sampled prompt.

Proof of Proposition 1: Fix a prompt x and token pair $( u , v )$ . Define

$$
\phi _ { x } ( s ) : = m _ { u , v } \left( x ; \mathbf { w } _ { t } + s \Delta \mathbf { w } _ { t , i } \right) , \qquad s \in [ 0 , 1 ] .\tag{99}
$$

This scalar function traces the token margin along the straight path from the current global model $\mathbf { w } _ { t }$ to the locally updated model

$$
\mathbf { w } _ { t } + \Delta \mathbf { w } _ { t , i } .\tag{100}
$$

By the fundamental theorem of calculus,

$$
m _ { u , v } ( x ; \mathbf { w } _ { t } + \Delta \mathbf { w } _ { t , i } ) - m _ { u , v } ( x ; \mathbf { w } _ { t } ) = \int _ { 0 } ^ { 1 } \phi _ { x } ^ { \prime } ( s ) d s .\tag{101}
$$

Using the chain rule,

$$
\begin{array} { r } { \phi _ { x } ^ { \prime } ( s ) = \nabla _ { \mathbf { w } } m _ { u , v } \left( x ; \mathbf { w } _ { t } + s \Delta \mathbf { w } _ { t , i } \right) ^ { \top } \Delta \mathbf { w } _ { t , i } . } \end{array}\tag{102}
$$

Therefore,

$$
m _ { u , v } ( x ; \mathbf { w } _ { t } + \Delta \mathbf { w } _ { t , i } ) - m _ { u , v } ( x ; \mathbf { w } _ { t } ) = \int _ { 0 } ^ { 1 } \nabla _ { \mathbf { w } } m _ { u , v } \left( x ; \mathbf { w } _ { t } + s \Delta \mathbf { w } _ { t , i } \right) ^ { \top } \Delta \mathbf { w } _ { t , i } d s .
$$

We now isolate the first-order effect at the current global model. Add and subtract

$$
\nabla _ { \mathbf { w } } m _ { u , v } ( x ; \mathbf { w } _ { t } )\tag{103}
$$

inside the integral:

$$
m _ { u , v } ( x ; \mathbf { w } _ { t } + \Delta \mathbf { w } _ { t , i } ) - m _ { u , v } ( x ; \mathbf { w } _ { t } ) = \nabla _ { \mathbf { w } } m _ { u , v } ( x ; \mathbf { w } _ { t } ) ^ { \top } \Delta \mathbf { w } _ { t , i } + \mathscr { R } _ { t , i } ( x , u , v ) ,\tag{104}
$$

where the pointwise Taylor remainder is

$$
\mathcal { R } _ { t , i } ( x , u , v ) : = \int _ { 0 } ^ { 1 } \Big [ \nabla _ { \mathbf { w } } m _ { u , v } \left( x ; \mathbf { w } _ { t } + s \Delta \mathbf { w } _ { t , i } \right) - \nabla _ { \mathbf { w } } m _ { u , v } ( x ; \mathbf { w } _ { t } ) \Big ] ^ { \top } \Delta \mathbf { w } _ { t , i } d s .\tag{105}
$$

This term contains the part of the margin change not captured by the linear approximation at $\mathbf { w } _ { t }$

Equivalently, Taylor’s theorem with integral remainder gives

$$
\mathcal { R } _ { t , i } ( x , u , v ) = \int _ { 0 } ^ { 1 } ( 1 - s ) \Delta \mathbf { w } _ { t , i } ^ { \top } \nabla _ { \mathbf { w } } ^ { 2 } m _ { u , v } \left( x ; \mathbf { w } _ { t } + s \Delta \mathbf { w } _ { t , i } \right) \Delta \mathbf { w } _ { t , i } d s .\tag{106}
$$

Using Assumption 1,

$$
\begin{array} { r l } & { \displaystyle \lvert \mathcal { R } _ { t , i } ( x , u , v ) \rvert \leq \int _ { 0 } ^ { 1 } ( 1 - s ) \left. \nabla _ { \mathbf { w } } ^ { 2 } m _ { u , v } \left( x ; \mathbf { w } _ { t } + s \Delta \mathbf { w } _ { t , i } \right) \right. _ { \mathrm { o p } } \left. \Delta \mathbf { w } _ { t , i } \right. _ { 2 } ^ { 2 } d s } \\ & { \qquad \leq \int _ { 0 } ^ { 1 } ( 1 - s ) L _ { m } \left. \Delta \mathbf { w } _ { t , i } \right. _ { 2 } ^ { 2 } d s } \\ & { \qquad = \displaystyle \frac { L _ { m } } { 2 } \left. \Delta \mathbf { w } _ { t , i } \right. _ { 2 } ^ { 2 } . } \end{array}\tag{107}
$$

Next, split the first-order term into the normalization-parameter part and the remaining-parameter part. Since

$$
\begin{array} { r } { { \bf w } = ( { \bf p } , { \bf q } ) , \qquad \Delta { \bf w } _ { t , i } = ( \Delta { \bf p } _ { t , i } , \Delta { \bf q } _ { t , i } ) , } \end{array}\tag{108}
$$

we have

$$
\nabla _ { \mathbf { w } } m _ { u , v } ( x ; \mathbf { w } _ { t } ) ^ { \top } \Delta \mathbf { w } _ { t , i } = \nabla _ { \mathbf { p } } m _ { u , v } ( x ; \mathbf { w } _ { t } ) ^ { \top } \Delta \mathbf { p } _ { t , i } + \nabla _ { \mathbf { q } } m _ { u , v } ( x ; \mathbf { w } _ { t } ) ^ { \top } \Delta \mathbf { q } _ { t , i } .\tag{109}
$$

Substituting (109) into (104) gives the pointwise decomposition

$$
m _ { u , v } ( x ; \mathbf { w } _ { t } + \Delta \mathbf { w } _ { t , i } ) - m _ { u , v } ( x ; \mathbf { w } _ { t } ) = \nabla _ { \mathbf { p } } m _ { u , v } ( x ; \mathbf { w } _ { t } ) ^ { \top } \Delta \mathbf { p } _ { t , i } + \nabla _ { \mathbf { q } } m _ { u , v } ( x ; \mathbf { w } _ { t } ) ^ { \top } \Delta \mathbf { q } _ { t , i } + \mathcal { R } _ { t , i } ( x , u , v ) .\tag{110}
$$

We now take expectation over $x \sim \mathcal { X }$ . This step is necessary because $\Delta \bar { m } _ { t , i } ( u , v )$ in (41) is the average margin change over prompts. Taking expectation on both sides of (110) gives

$$
\begin{array} { r } { \Delta \bar { m } _ { t , i } ( u , v ) = \mathbb { E } _ { x \sim \mathcal { X } } \left[ \nabla _ { \mathbf { p } } m _ { u , v } ( x ; \mathbf { w } _ { t } ) ^ { \top } \Delta \mathbf { p } _ { t , i } \right] + \mathbb { E } _ { x \sim \mathcal { X } } \left[ \nabla _ { \mathbf { q } } m _ { u , v } ( x ; \mathbf { w } _ { t } ) ^ { \top } \Delta \mathbf { q } _ { t , i } \right] + \mathbb { E } _ { x \sim \mathcal { X } } \left[ \mathcal { R } _ { t , i } ( x , u , v ) \right] . } \end{array}\tag{111}
$$

Because $\Delta \mathbf { p } _ { t , i }$ and $\Delta \mathbf { q } _ { t , : }$ <sub>i</sub> are fixed with respect to the prompt draw $x ,$ they can be moved outside the expectation. For the normalization-parameter part, if $\Delta \mathbf { p } _ { t , i } \in \mathbb { R } ^ { d _ { p } }$ , then

$$
\begin{array} { r l r } & { } & { \mathbb { E } _ { x \sim \mathcal { X } } \left[ \nabla _ { \mathbf { p } } m _ { u , v } ( x ; \mathbf { w } _ { t } ) ^ { \top } \Delta \mathbf { p } _ { t , i } \right] = \mathbb { E } _ { x \sim \mathcal { X } } \left[ \displaystyle \sum _ { r = 1 } ^ { d _ { p } } \frac { \partial m _ { u , v } ( x ; \mathbf { w } _ { t } ) } { \partial \mathbf { p } [ r ] } \Delta \mathbf { p } _ { t , i } [ r ] \right] } \\ & { } & { = \displaystyle \sum _ { r = 1 } ^ { d _ { p } } \mathbb { E } _ { x \sim \mathcal { X } } \left[ \frac { \partial m _ { u , v } ( x ; \mathbf { w } _ { t } ) } { \partial \mathbf { p } [ r ] } \right] \Delta \mathbf { p } _ { t , i } [ r ] } \\ & { } & { = \mathbb { E } _ { x \sim \mathcal { X } } \left[ \nabla _ { \mathbf { p } } m _ { u , v } ( x ; \mathbf { w } _ { t } ) \right] ^ { \top } \Delta \mathbf { p } _ { t , i } . } \end{array}\tag{112}
$$

The same argument gives

$$
\begin{array} { r } { \mathbb { E } _ { x \sim \mathcal { X } } \left[ \nabla _ { \mathbf { q } } m _ { u , v } ( x ; \mathbf { w } _ { t } ) ^ { \top } \Delta \mathbf { q } _ { t , i } \right] = \mathbb { E } _ { x \sim \mathcal { X } } \left[ \nabla _ { \mathbf { q } } m _ { u , v } ( x ; \mathbf { w } _ { t } ) \right] ^ { \top } \Delta \mathbf { q } _ { t , i } . } \end{array}\tag{113}
$$

Define

$$
\begin{array} { r } { \mathbf { g } _ { p , u , v } : = \mathbb { E } _ { x \sim \mathcal { X } } \left[ \nabla _ { \mathbf { p } } m _ { u , v } ( x ; \mathbf { w } _ { t } ) \right] , } \end{array}\tag{114}
$$

$$
\begin{array} { r } { \mathbf { g } _ { q , u , v } : = \mathbb { E } _ { x \sim \mathcal { X } } \left[ \nabla _ { \mathbf { q } } m _ { u , v } ( x ; \mathbf { w } _ { t } ) \right] , } \end{array}\tag{115}
$$

and

$$
R _ { t , i } ( u , v ) : = \mathbb { E } _ { x \sim \mathcal { X } } \left[ \mathscr { R } _ { t , i } ( x , u , v ) \right] .\tag{116}
$$

Using these definitions in (111) yields

$$
\begin{array} { r } { \Delta \bar { m } _ { t , i } ( u , v ) = \mathbf { g } _ { p , u , v } ^ { \top } \Delta \mathbf { p } _ { t , i } + \mathbf { g } _ { q , u , v } ^ { \top } \Delta \mathbf { q } _ { t , i } + R _ { t , i } ( u , v ) . } \end{array}\tag{117}
$$

Finally, using (107),

$$
\begin{array} { r l } & { \left| R _ { t , i } ( u , v ) \right| = \left| \mathbb { E } _ { x \sim \mathcal { X } } \left[ \mathcal { R } _ { t , i } ( x , u , v ) \right] \right| } \\ & { \quad \quad \quad \quad \quad \leq \mathbb { E } _ { x \sim \mathcal { X } } \left[ \left| \mathcal { R } _ { t , i } ( x , u , v ) \right| \right] } \\ & { \quad \quad \quad \quad \leq \displaystyle \frac { L _ { m } } { 2 } \left\| \Delta \mathbf { w } _ { t , i } \right\| _ { 2 } ^ { 2 } . } \end{array}\tag{118}
$$

This proves Proposition 1.

The following corollary bounds the difference between two clients’ first-order margin changes using their normalizationparameter and remaining-parameter update differences, together with the second-order remainders.

## Corollary 3 (Stability of margin changes).

Under Assumption 1, for any two clients i and $j ,$

$$
| \Delta \bar { m } _ { t , i } ( u , v ) - \Delta \bar { m } _ { t , j } ( u , v ) | \leq \| \mathbf { g } _ { p , u , v } \| _ { 2 } \| \Delta \mathbf { p } _ { t , i } - \Delta \mathbf { p } _ { t , j } \| _ { 2 } + \| \mathbf { g } _ { q , u , v } \| _ { 2 } \| \Delta \mathbf { q } _ { t , i } - \Delta \mathbf { q } _ { t , j } \| _ { 2 } + \frac { L _ { m } } { 2 } \left( \| \Delta \mathbf { w } _ { t , i } \| _ { 2 } ^ { 2 } + \| \Delta \mathbf { w } _ { t , j } \| _ { 2 } ^ { 2 } \right) ,\tag{119}
$$

Proof: Subtract (42) for clients i and $j \colon$

$$
\begin{array} { r } { \Delta \bar { m } _ { t , i } ( u , v ) - \Delta \bar { m } _ { t , j } ( u , v ) = \mathbf { g } _ { p , u , v } ^ { \top } \left( \Delta \mathbf { p } _ { t , i } - \Delta \mathbf { p } _ { t , j } \right) + \mathbf { g } _ { q , u , v } ^ { \top } \left( \Delta \mathbf { q } _ { t , i } - \Delta \mathbf { q } _ { t , j } \right) + R _ { t , i } ( u , v ) - R _ { t , j } ( u , v ) . } \end{array}\tag{120}
$$

Taking absolute values and applying Cauchy–Schwarz gives

$$
\begin{array} { r } { | \Delta \bar { m } _ { t , i } ( u , v ) - \Delta \bar { m } _ { t , j } ( u , v ) | \leq \| { \bf g } _ { p , u , v } \| _ { 2 } \| \Delta { \bf p } _ { t , i } - \Delta { \bf p } _ { t , j } \| _ { 2 } + \| { \bf g } _ { q , u , v } \| _ { 2 } \| \Delta { \bf q } _ { t , i } - \Delta { \bf q } _ { t , j } \| _ { 2 } + | R _ { t , i } ( u , v ) | + | R _ { t , j } ( u , v ) | . } \end{array}\tag{121}
$$

Using (44) for both remainders proves the claim.

## B. Proof of the Screened-Aggregation Bound

We now prove Theorem 1. The proof shows how the screening error $\mathbf { e } _ { t }$ enters the descent inequality. If the retained update direction is close to the benign gradient direction, then $\| \mathbf { e } _ { t } \| _ { 2 }$ is small, and the update behaves similarly to benign training.

Assumption 2 (Benign objective).

The target objective is

$$
F ( \mathbf { w } ) = \sum _ { i \in \mathcal { C } _ { \mathrm { b e n } } } \lambda _ { i } F _ { i } ( \mathbf { w } ) , \qquad \lambda _ { i } \geq 0 , \quad \sum _ { i \in \mathcal { C } _ { \mathrm { b e n } } } \lambda _ { i } = 1 .\tag{122}
$$

The function F is differentiable, L<sub>F</sub>-smooth, and lower bounded:

$$
F ( \mathbf { w } ) \geq F _ { \mathrm { i n f } } > - \infty .\tag{123}
$$

The L<sub>F</sub>-smoothness condition means that, for all $\mathbf { w }$ and $\mathbf { w } ^ { \prime }$ ,

$$
F ( \mathbf { w } ^ { \prime } ) \leq F ( \mathbf { w } ) + \nabla F ( \mathbf { w } ) ^ { \top } ( \mathbf { w } ^ { \prime } - \mathbf { w } ) + \frac { L _ { F } } { 2 } \left\| \mathbf { w } ^ { \prime } - \mathbf { w } \right\| _ { 2 } ^ { 2 } .\tag{124}
$$

Assumption 3 (Screened update model).

At round t, the screened update can be written as

$$
\mathbf { w } _ { t + 1 } = \mathbf { w } _ { t } - \eta _ { t } \mathbf { g } _ { t } ,\tag{125}
$$

where $\eta _ { t } \geq 0$ is the effective step size and $\mathbf { g } _ { t }$ is the retained update direction. Let $\mathcal { F } _ { t }$ contain the information available before the update noise at round t is realized. There exist $\beta _ { t } \geq 0$ and $\nu _ { t } \geq 0$ such that

$$
\left\| \mathbb { E } [ \mathbf { g } _ { t } \mid \mathcal { F } _ { t } ] - \nabla F ( \mathbf { w } _ { t } ) \right\| _ { 2 } \leq \beta _ { t } ,\tag{126}
$$

and

$$
\mathbb { E } \left[ \left\| \mathbf { g } _ { t } - \mathbb { E } [ \mathbf { g } _ { t } \mid \mathcal { F } _ { t } ] \right\| _ { 2 } ^ { 2 } \bigm | \mathcal { F } _ { t } \right] \leq \nu _ { t } ^ { 2 } .\tag{127}
$$

Proof of Theorem 1: By smoothness (124) and the update rule (125),

$$
F ( \mathbf { w } _ { t + 1 } ) \leq F ( \mathbf { w } _ { t } ) - \eta _ { t } \nabla F ( \mathbf { w } _ { t } ) ^ { \top } \mathbf { g } _ { t } + \frac { L _ { F } \eta _ { t } ^ { 2 } } { 2 } \| \mathbf { g } _ { t } \| _ { 2 } ^ { 2 } .\tag{128}
$$

Take conditional expectation given $\mathcal { F } _ { t } .$ . Define

$$
\mathbf { a } _ { t } : = \nabla F ( \mathbf { w } _ { t } ) , \qquad \bar { \mathbf { g } } _ { t } : = \mathbb { E } [ \mathbf { g } _ { t } \mid \mathcal { F } _ { t } ] , \qquad \mathbf { e } _ { t } : = \bar { \mathbf { g } } _ { t } - \mathbf { a } _ { t } .\tag{129}
$$

Then

$$
\bar { \mathbf { g } } _ { t } = \mathbf { a } _ { t } + \mathbf { e } _ { t } , \qquad \lVert \mathbf { e } _ { t } \rVert _ { 2 } \leq \beta _ { t } .\tag{130}
$$

The first-order term satisfies

$$
\begin{array} { r l } & { - \eta _ { t } \mathbf { a } _ { t } ^ { \top } \bar { \mathbf { g } } _ { t } = - \eta _ { t } \mathbf { a } _ { t } ^ { \top } ( \mathbf { a } _ { t } + \mathbf { e } _ { t } ) } \\ & { \quad \quad \quad = - \eta _ { t } \| \mathbf { a } _ { t } \| _ { 2 } ^ { 2 } - \eta _ { t } \mathbf { a } _ { t } ^ { \top } \mathbf { e } _ { t } . } \end{array}\tag{131}
$$

By Young’s inequality,

$$
| \mathbf { a } _ { t } ^ { \top } \mathbf { e } _ { t } | \leq \frac { 1 } { 4 } \| \mathbf { a } _ { t } \| _ { 2 } ^ { 2 } + \| \mathbf { e } _ { t } \| _ { 2 } ^ { 2 } .\tag{132}
$$

Therefore,

$$
- \eta _ { t } \mathbf { a } _ { t } ^ { \top } \bar { \mathbf { g } } _ { t } \leq - \frac { 3 \eta _ { t } } { 4 } \| \mathbf { a } _ { t } \| _ { 2 } ^ { 2 } + \eta _ { t } \| \mathbf { e } _ { t } \| _ { 2 } ^ { 2 } .\tag{133}
$$

Next,

$$
\begin{array} { r l } & { \mathbb { E } \left[ \| \mathbf { g } _ { t } \| _ { 2 } ^ { 2 } \vert \mathcal { F } _ { t } \right] = \| \bar { \mathbf { g } } _ { t } \| _ { 2 } ^ { 2 } + \mathbb { E } \left[ \| \mathbf { g } _ { t } - \bar { \mathbf { g } } _ { t } \| _ { 2 } ^ { 2 } \vert \mathcal { F } _ { t } \right] } \\ & { \quad \quad \quad \leq \| \mathbf { a } _ { t } + \mathbf { e } _ { t } \| _ { 2 } ^ { 2 } + \nu _ { t } ^ { 2 } } \\ & { \quad \quad \quad \leq 2 \| \mathbf { a } _ { t } \| _ { 2 } ^ { 2 } + 2 \| \mathbf { e } _ { t } \| _ { 2 } ^ { 2 } + \nu _ { t } ^ { 2 } . } \end{array}\tag{134}
$$

Substituting (133) and (134) into (128) gives

$$
\begin{array} { r } { \mathbb { E } \left[ F ( \mathbf { w } _ { t + 1 } ) \mid \mathcal { F } _ { t } \right] \le F ( \mathbf { w } _ { t } ) - \left( \frac { 3 \eta _ { t } } { 4 } - L _ { F } \eta _ { t } ^ { 2 } \right) \| \mathbf { a } _ { t } \| _ { 2 } ^ { 2 } + ( \eta _ { t } + L _ { F } \eta _ { t } ^ { 2 } ) \| \mathbf { e } _ { t } \| _ { 2 } ^ { 2 } + \frac { L _ { F } \eta _ { t } ^ { 2 } } { 2 } \nu _ { t } ^ { 2 } . } \end{array}\tag{135}
$$

If $0 \leq \eta _ { t } \leq 1 / ( 4 L _ { F } )$ , then

$$
\frac { 3 \eta _ { t } } { 4 } - L _ { F } \eta _ { t } ^ { 2 } \geq \frac { \eta _ { t } } { 2 } .\tag{136}
$$

Using $\| \mathbf { e } _ { t } \| _ { 2 } ^ { 2 } \leq \beta _ { t } ^ { 2 }$ , we obtain

$$
\mathbb { E } \left[ F ( \mathbf { w } _ { t + 1 } ) \mid \mathcal { F } _ { t } \right] \leq F ( \mathbf { w } _ { t } ) - \frac { \eta _ { t } } { 2 } \| \nabla F ( \mathbf { w } _ { t } ) \| _ { 2 } ^ { 2 } + ( \eta _ { t } + L _ { F } \eta _ { t } ^ { 2 } ) \beta _ { t } ^ { 2 } + \frac { L _ { F } \eta _ { t } ^ { 2 } } { 2 } \nu _ { t } ^ { 2 } .\tag{137}
$$

Taking total expectation and summing from $t = 0$ to $T - 1$ yields

$$
\frac { 1 } { 2 } \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \mathbb { E } \left[ \| \nabla F ( \mathbf { w } _ { t } ) \| _ { 2 } ^ { 2 } \right] \leq F ( \mathbf { w } _ { 0 } ) - \mathbb { E } [ F ( \mathbf { w } _ { T } ) ] + \sum _ { t = 0 } ^ { T - 1 } ( \eta _ { t } + L _ { F } \eta _ { t } ^ { 2 } ) \mathbb { E } [ \beta _ { t } ^ { 2 } ] + \frac { L _ { F } } { 2 } \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } ^ { 2 } \mathbb { E } [ \nu _ { t } ^ { 2 } ] .\tag{138}
$$

Since $F ( \mathbf { w } _ { T } ) \geq F _ { \mathrm { i n f } }$

$$
\frac { 1 } { 2 } \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \mathbb { E } \left[ \| \nabla F ( \mathbf { w } _ { t } ) \| _ { 2 } ^ { 2 } \right] \leq F ( \mathbf { w } _ { 0 } ) - F _ { \mathrm { i n f } } \quad + \sum _ { t = 0 } ^ { T - 1 } ( \eta _ { t } + L _ { F } \eta _ { t } ^ { 2 } ) \mathbb { E } [ \beta _ { t } ^ { 2 } ] \quad + \frac { L _ { F } } { 2 } \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } ^ { 2 } \mathbb { E } [ \nu _ { t } ^ { 2 } ] .\tag{139}
$$

Dividing both sides by ${ \ L } _ { 2 } ^ { 1 } S _ { T }$ proves (54).

Proof of Corollary 1: From Theorem 1,

$$
\frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \mathbb { E } \left[ \left. \nabla F ( \mathbf { w } _ { t } ) \right. _ { 2 } ^ { 2 } \right] } { S _ { T } } \le \frac { 2 \left( F ( \mathbf { w } _ { 0 } ) - F _ { \mathrm { i n f } } \right) } { S _ { T } } + \frac { 2 \sum _ { t = 0 } ^ { T - 1 } ( \eta _ { t } + L _ { F } \eta _ { t } ^ { 2 } ) \mathbb { E } [ \beta _ { t } ^ { 2 } ] } { S _ { T } } + \frac { L _ { F } \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } ^ { 2 } \mathbb { E } [ \nu _ { t } ^ { 2 } ] } { S _ { T } } .\tag{140}
$$

We show that each term on the right-hand side goes to zero.

First, since $S _ { T }  \infty$ and $F ( \mathbf { w } _ { 0 } ) - F _ { \mathrm { i n f } }$ is finite,

$$
\frac { 2 ( F ( \mathbf { w } _ { 0 } ) - F _ { \mathrm { i n f } } ) } { S _ { T } }  0 .\tag{141}
$$

Second, because $\eta _ { t } \leq 1 / ( 4 L _ { F } )$

$$
L _ { F } \eta _ { t } ^ { 2 } \leq \frac 1 4 \eta _ { t } ,\tag{142}
$$

and therefore

$$
\eta _ { t } + L _ { F } \eta _ { t } ^ { 2 } \leq \frac { 5 } { 4 } \eta _ { t } .\tag{143}
$$

Thus,

$$
\frac { \sum _ { t = 0 } ^ { T - 1 } ( \eta _ { t } + L _ { F } \eta _ { t } ^ { 2 } ) \mathbb { E } [ \beta _ { t } ^ { 2 } ] } { S _ { T } } \leq \frac { 5 } { 4 } \frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \mathbb { E } [ \beta _ { t } ^ { 2 } ] } { S _ { T } } .\tag{144}
$$

By (57),

$$
\frac { \sum _ { t = 0 } ^ { T - 1 } ( \eta _ { t } + L _ { F } \eta _ { t } ^ { 2 } ) \mathbb { E } [ \beta _ { t } ^ { 2 } ] } { S _ { T } }  0 .\tag{145}
$$

XXXX, 2026.

Third, since $\mathbb { E } [ \nu _ { t } ^ { 2 } ] \le \nu ^ { 2 }$

$$
\frac { L _ { F } \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } ^ { 2 } \mathbb { E } [ \nu _ { t } ^ { 2 } ] } { S _ { T } } \leq L _ { F } \nu ^ { 2 } \frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } ^ { 2 } } { S _ { T } } .\tag{146}
$$

By (55),

$$
L _ { F } \nu ^ { 2 } \frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } ^ { 2 } } { S _ { T } }  0 .\tag{147}
$$

Combining (141), (145), and (147) in (140) gives

$$
\frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \mathbb { E } \left[ \lVert \nabla F ( \mathbf { w } _ { t } ) \rVert _ { 2 } ^ { 2 } \right] } { S _ { T } } \to 0 .\tag{148}
$$

Now define a random round τ by

$$
\operatorname* { P r } ( \tau = t ) = \frac { \eta _ { t } } { S _ { T } } , \qquad t = 0 , \dots , T - 1 .\tag{149}
$$

Then, by the law of total expectation,

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \left[ \| \nabla F ( \mathbf { w } _ { \tau } ) \| _ { 2 } ^ { 2 } \right] = \sum _ { t = 0 } ^ { T - 1 } \mathrm { P r } ( \tau = t ) \mathbb { E } \left[ \| \nabla F ( \mathbf { w } _ { t } ) \| _ { 2 } ^ { 2 } \right] } & { } \\ { = \frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \mathbb { E } \left[ \| \nabla F ( \mathbf { w } _ { t } ) \| _ { 2 } ^ { 2 } \right] } { S _ { T } } . } \end{array}\tag{150}
$$

Therefore,

$$
\mathbb { E } [ \lVert \nabla F ( \mathbf { w } _ { \tau } ) \rVert _ { 2 } ^ { 2 } ]  0 .\tag{151}
$$

This proves the corollary.

Proof of Corollary 2: Set $\eta _ { t } = \eta$ in Theorem 1. Then

$$
S _ { T } = \sum _ { t = 0 } ^ { T - 1 } \eta = \eta T .\tag{152}
$$

The left-hand side of (54) becomes

$$
\frac { \sum _ { t = 0 } ^ { T - 1 } \eta \mathbb { E } \left[ \| \nabla F ( \mathbf { w } _ { t } ) \| _ { 2 } ^ { 2 } \right] } { \eta T } = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ \| \nabla F ( \mathbf { w } _ { t } ) \| _ { 2 } ^ { 2 } \right] .\tag{153}
$$

The optimization term becomes

$$
\frac { 2 \left( F ( \mathbf { w } _ { 0 } ) - F _ { \mathrm { i n f } } \right) } { \eta T } .\tag{154}
$$

The screening-error term becomes

$$
\begin{array} { r l r } {  { \frac { 2 \sum _ { t = 0 } ^ { T - 1 } ( \eta + L _ { F } \eta ^ { 2 } ) \mathbb { E } [ \beta _ { t } ^ { 2 } ] } { \eta T } = 2 ( 1 + L _ { F } \eta ) \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \beta _ { t } ^ { 2 } ] } } \\ & { } & { = 2 ( 1 + L _ { F } \eta ) \overline { { \beta } } _ { T } ^ { 2 } . } \end{array}\tag{155}
$$

The variance term becomes

$$
\begin{array} { r l r } & { } & { \frac { L _ { \cal F } \sum _ { t = 0 } ^ { T - 1 } \eta ^ { 2 } \mathbb { E } [ \nu _ { t } ^ { 2 } ] } { \eta T } = L _ { \cal F } \eta \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \nu _ { t } ^ { 2 } ] } \\ & { } & { \quad = L _ { \cal F } \eta \overline { { \nu } } _ { T } ^ { 2 } . \quad \quad } \end{array}\tag{156}
$$

Substituting these three expressions into (54) proves (67).

## C. Screening-Error Decomposition

We now connect $\beta _ { t }$ to the retained set returned by FEDLNS. This decomposition relates the screening error to retained malicious weight, benign-client representativeness, and local-training drift.

Let $\mathcal { C } _ { \mathrm { m a l } }$ denote the malicious client set and $\mathcal { C } _ { \mathrm { b e n } }$ denote the benign client set. At round t, FEDLNS retains $\mathcal { R } _ { t } .$ . For each retained client $i \in \mathcal { R } _ { t }$ , define

$$
\begin{array} { r } { \alpha _ { t , i } : = \frac { n _ { t , i } } { \sum _ { j \in \mathscr { R } _ { t } } n _ { t , j } } . } \end{array}\tag{157}
$$

The total retained malicious weight is

$$
\mu _ { t } : = \sum _ { a \in \mathcal { R } _ { t } \cap \mathcal { C } _ { \mathrm { m a l } } } \alpha _ { t , a } .\tag{158}
$$

For a retained benign client i, define its conditional expected local direction as

$$
\bar { \mathbf { g } } _ { t , i } ^ { B } : = \mathbb { E } \left[ \mathbf { g } _ { t , i } \mid \mathcal { F } _ { t } \right] , \qquad i \in \mathcal { C } _ { \mathrm { b e n } } .\tag{159}
$$

We decompose it as

$$
\begin{array} { r } { \bar { \mathbf { g } } _ { t , i } ^ { B } = \nabla F _ { i } ( \mathbf w _ { t } ) + \mathbf r _ { t , i } , } \end{array}\tag{160}
$$

where $\mathbf { r } _ { t , i }$ is the local-training drift vector. For a retained malicious client a, define

$$
\mathbf { h } _ { t , a } : = \mathbb { E } \left[ \mathbf { g } _ { t , a } \mid \mathcal { F } _ { t } \right] , \qquad a \in \mathcal { C } _ { \operatorname* { m a l } } .\tag{161}
$$

The vector $\mathbf { h } _ { t , a }$ can be arbitrary.

The conditional mean retained direction is

$$
\mathbb { E } [ \mathbf { g } _ { t } \mid \mathcal { F } _ { t } ] = \sum _ { i \in \mathcal { R } _ { t } \cap \mathcal { C } _ { \mathrm { b e n } } } \alpha _ { t , i } \left( \nabla F _ { i } ( \mathbf { w } _ { t } ) + \mathbf { r } _ { t , i } \right) + \sum _ { a \in \mathcal { R } _ { t } \cap \mathcal { C } _ { \mathrm { m a l } } } \alpha _ { t , a } \mathbf { h } _ { t , a } .\tag{162}
$$

Assume $\mu _ { t } < 1$ . Define the retained benign gradient after renormalizing the benign retained weights:

$$
\bar { \mathbf { g } } _ { t } ^ { B , \mathrm { r e t } } : = \sum _ { i \in \mathcal { R } _ { t } \cap \mathcal { C } _ { \mathrm { b e n } } } \frac { \alpha _ { t , i } } { 1 - \mu _ { t } } \nabla F _ { i } ( \mathbf { w } _ { t } ) .\tag{163}
$$

The retained benign representativeness error is

$$
\begin{array} { r } { \chi _ { t } : = \left. \bar { \mathbf g } _ { t } ^ { B , \mathrm { r e t } } - \nabla F ( \mathbf { w } _ { t } ) \right. _ { 2 } . } \end{array}\tag{164}
$$

The retained benign local-drift magnitude is

$$
\varrho _ { t } : = \left\| \sum _ { i \in \mathcal { R } _ { t } \cap \mathcal { C } _ { \mathrm { b e n } } } \alpha _ { t , i } \mathbf { r } _ { t , i } \right\| _ { 2 } .\tag{165}
$$

Assumption 4 (Bounded retained directions).

There exists $G > 0$ such that

$$
\| \mathbf { h } _ { t , a } \| _ { 2 } \leq G , \qquad \| \nabla F ( \mathbf { w } _ { t } ) \| _ { 2 } \leq G\tag{166}
$$

for all retained malicious clients a and all rounds considered.

Proposition 4 (Screening-error bound).

Under the definitions above and Assumption 4,

$$
\begin{array} { r } { \left\| \mathbb { E } [ \mathbf { g } _ { t } \mid \mathcal { F } _ { t } ] - \nabla F ( \mathbf { w } _ { t } ) \right\| _ { 2 } \leq ( 1 - \mu _ { t } ) \chi _ { t } + 2 G \mu _ { t } + \varrho _ { t } . } \end{array}\tag{167}
$$

Therefore, one may choose

$$
\beta _ { t } = ( 1 - \mu _ { t } ) \chi _ { t } + 2 G \mu _ { t } + \varrho _ { t } .\tag{168}
$$

Proof: Using (162),

$$
\mathbb { E } [ \mathbf { g } _ { t } \mid \mathcal { F } _ { t } ] - \nabla F ( \mathbf { w } _ { t } ) = \sum _ { i \in \mathcal { R } _ { t } \cap \mathcal { C } _ { \mathrm { b o n } } } \alpha _ { t , i } \nabla F _ { i } ( \mathbf { w } _ { t } ) - \nabla F ( \mathbf { w } _ { t } ) + \sum _ { a \in \mathcal { R } _ { t } \cap \mathcal { C } _ { \mathrm { m a l } } } \alpha _ { t , a } \mathbf { h } _ { t , a } + \sum _ { i \in \mathcal { R } _ { t } \cap \mathcal { C } _ { \mathrm { b o n } } } \alpha _ { t , i } \mathbf { r } _ { t , i } .\tag{169}
$$

The benign-gradient part satisfies

$$
\begin{array} { r l } { \displaystyle \sum _ { i \in \mathcal { R } _ { t } \cap \mathcal { C } _ { \mathrm { b e n } } } \alpha _ { t , i } \nabla F _ { i } ( \mathbf { w } _ { t } ) - \nabla F ( \mathbf { w } _ { t } ) = ( 1 - \mu _ { t } ) \bar { \mathbf { g } } _ { t } ^ { B , \mathrm { r e t } } - \nabla F ( \mathbf { w } _ { t } ) } & { } \\ { = ( 1 - \mu _ { t } ) \left( \bar { \mathbf { g } } _ { t } ^ { B , \mathrm { r e t } } - \nabla F ( \mathbf { w } _ { t } ) \right) - \mu _ { t } \nabla F ( \mathbf { w } _ { t } ) . } & { } \end{array}\tag{170}
$$

XXXX, 2026.

Thus,

$$
\left\| \sum _ { i \in \mathcal { R } _ { t } \cap \mathcal { C } _ { \mathrm { b e n } } } \alpha _ { t , i } \nabla F _ { i } ( \mathbf { w } _ { t } ) - \nabla F ( \mathbf { w } _ { t } ) \right\| _ { 2 } \leq ( 1 - \mu _ { t } ) \chi _ { t } + \mu _ { t } \| \nabla F ( \mathbf { w } _ { t } ) \| _ { 2 } .\tag{171}
$$

The retained malicious part satisfies

$$
\begin{array} { r l r } {  { \biggl \| \sum _ { a \in \mathcal { R } _ { t } \cap \mathcal { C } _ { \operatorname* { m a l } } } \alpha _ { t , a } \mathbf { h } _ { t , a } \biggl \| _ { 2 } \leq \sum _ { a \in \mathcal { R } _ { t } \cap \mathcal { C } _ { \operatorname* { m a l } } } \alpha _ { t , a } \| \mathbf { h } _ { t , a } \| _ { 2 } } } \\ & { } & { \leq G \mu _ { t } . } \end{array}\tag{172}
$$

The local-drift term is $\varrho _ { t }$ by definition. Combining the three bounds and using $\| \nabla F ( \mathbf { w } _ { t } ) \| _ { 2 } \leq G$ gives

$$
\begin{array} { r } { \left\| \mathbb { E } [ \mathbf { g } _ { t } \mid \mathcal { F } _ { t } ] - \nabla F ( \mathbf { w } _ { t } ) \right\| _ { 2 } \leq ( 1 - \mu _ { t } ) \chi _ { t } + 2 G \mu _ { t } + \varrho _ { t } . } \end{array}\tag{173}
$$

This proves the result.

Lemma 1 (Sufficient condition for vanishing screening error).

Suppose

$$
\beta _ { t } \leq ( 1 - \mu _ { t } ) \chi _ { t } + 2 G \mu _ { t } + \varrho _ { t } .\tag{174}
$$

If

$$
\frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \mathbb { E } \left[ \chi _ { t } ^ { 2 } \right] } { S _ { T } } \to 0 , \qquad \frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \mathbb { E } \left[ \mu _ { t } ^ { 2 } \right] } { S _ { T } } \to 0 , \qquad \frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \mathbb { E } \left[ \varrho _ { t } ^ { 2 } \right] } { S _ { T } } \to 0 ,\tag{175}
$$

then

$$
\frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \mathbb { E } [ \beta _ { t } ^ { 2 } ] } { S _ { T } }  0 .\tag{176}
$$

Proof: Using $( a + b + c ) ^ { 2 } \leq 3 a ^ { 2 } + 3 b ^ { 2 } + 3 c ^ { 2 }$ , we obtain

$$
\begin{array} { r } { \beta _ { t } ^ { 2 } \leq 3 ( 1 - \mu _ { t } ) ^ { 2 } \chi _ { t } ^ { 2 } + 1 2 G ^ { 2 } \mu _ { t } ^ { 2 } + 3 \varrho _ { t } ^ { 2 } . } \end{array}\tag{177}
$$

Since $( 1 - \mu _ { t } ) ^ { 2 } \leq 1$ , this gives

$$
\beta _ { t } ^ { 2 } \le 3 \chi _ { t } ^ { 2 } + 1 2 G ^ { 2 } \mu _ { t } ^ { 2 } + 3 \varrho _ { t } ^ { 2 } .\tag{178}
$$

Multiplying by $\eta _ { t }$ , summing from $t = 0$ to $T - 1$ , taking expectation, and dividing by $S _ { T }$ yields

$$
\frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \mathbb { E } [ \beta _ { t } ^ { 2 } ] } { S _ { T } } \leq 3 \frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \mathbb { E } [ \chi _ { t } ^ { 2 } ] } { S _ { T } } + 1 2 G ^ { 2 } \frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \mathbb { E } [ \mu _ { t } ^ { 2 } ] } { S _ { T } } + 3 \frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \mathbb { E } [ \varrho _ { t } ^ { 2 } ] } { S _ { T } } .\tag{179}
$$

Each term on the right-hand side converges to zero by assumption. Therefore,

$$
\frac { \sum _ { t = 0 } ^ { T - 1 } \eta _ { t } \mathbb { E } [ \beta _ { t } ^ { 2 } ] } { S _ { T } }  0 .\tag{180}
$$

## D. Local-Training Drift Bound

For completeness, we provide a bound on $\varrho _ { t }$ under local gradient descent.

Assumption 5 (Bounded benign gradients during local training).

For every benign client $i ,$ local step $e ,$ and round t,

$$
\left\| \nabla F _ { i } ( \mathbf { w } _ { t , i } ^ { ( e ) } ) \right\| _ { 2 } \leq G _ { B } .\tag{181}
$$

Also, each $F _ { i }$ is $L _ { i }$ -smooth with $L _ { i } \leq L _ { \operatorname* { m a x } }$

Suppose benign client i performs $E$ local gradient steps with local learning rate $\eta _ { \mathrm { l o c } } \mathrm { : }$

$$
\begin{array} { r } { \mathbf { w } _ { t , i } ^ { ( 0 ) } = \mathbf { w } _ { t } , \qquad \mathbf { w } _ { t , i } ^ { ( e + 1 ) } = \mathbf { w } _ { t , i } ^ { ( e ) } - \eta _ { \mathrm { l o c } } \nabla F _ { i } ( \mathbf { w } _ { t , i } ^ { ( e ) } ) , \qquad e = 0 , \dots , E - 1 . } \end{array}\tag{182}
$$

Define the average local gradient direction

$$
\bar { \mathbf { g } } _ { t , i } ^ { B } = \frac { 1 } { E } \sum _ { e = 0 } ^ { E - 1 } \nabla F _ { i } ( \mathbf { w } _ { t , i } ^ { ( e ) } ) .\tag{183}
$$

Then

$$
\mathbf { r } _ { t , i } = \bar { \mathbf { g } } _ { t , i } ^ { B } - \nabla F _ { i } ( \mathbf { w } _ { t } ) .\tag{184}
$$

Lemma 2 (Local-training drift bound).

Under Assumption 5,

$$
\| \mathbf { r } _ { t , i } \| _ { 2 } \leq \frac { L _ { \operatorname* { m a x } } \eta _ { \mathrm { l o c } } G _ { B } ( E - 1 ) } { 2 } .\tag{185}
$$

Consequently,

$$
\varrho _ { t } \leq ( 1 - \mu _ { t } ) \frac { L _ { \operatorname* { m a x } } \eta _ { \mathrm { l o c } } G _ { B } ( E - 1 ) } { 2 } .\tag{186}
$$

Proof: By L<sub>i</sub>-smoothness,

$$
\begin{array} { r } { \left. \nabla F _ { i } ( { \mathbf w } _ { t , i } ^ { ( e ) } ) - \nabla F _ { i } ( { \mathbf w } _ { t } ) \right. _ { 2 } \le L _ { i } \left. { \mathbf w } _ { t , i } ^ { ( e ) } - { \mathbf w } _ { t } \right. _ { 2 } . } \end{array}\tag{187}
$$

Because each local step has norm at most $\eta _ { \mathrm { l o c } } G _ { B }$

$$
\left\| \mathbf { w } _ { t , i } ^ { ( e ) } - \mathbf { w } _ { t } \right\| _ { 2 } \leq e \eta _ { \mathrm { l o c } } G _ { B } .\tag{188}
$$

Thus,

$$
\begin{array} { l } { \displaystyle \| \mathbf { r } _ { t , i } \| _ { 2 } = \left\| \frac { 1 } { E } \sum _ { e = 0 } ^ { E - 1 } \left[ \nabla F _ { i } ( \mathbf { w } _ { t , i } ^ { ( e ) } ) - \nabla F _ { i } ( \mathbf { w } _ { t } ) \right] \right\| _ { 2 } } \\ { \displaystyle \leq \frac { 1 } { E } \sum _ { e = 0 } ^ { E - 1 } L _ { i } \left\| \mathbf { w } _ { t , i } ^ { ( e ) } - \mathbf { w } _ { t } \right\| _ { 2 } } \\ { \displaystyle \leq \frac { 1 } { E } \sum _ { e = 0 } ^ { E - 1 } L _ { \mathrm { m a x } } e \eta _ { \mathrm { l o c } } G _ { B } } \\ { \displaystyle = \frac { L _ { \mathrm { m a x } } \eta _ { \mathrm { l o c } } G _ { B } ( E - 1 ) } { 2 } . } \end{array}\tag{189}
$$

Finally,

$$
\begin{array} { r l } & { \varrho _ { t } = \bigg \| \displaystyle \sum _ { i \in { \mathcal R } _ { t } \cap { \mathcal C } _ { \mathrm { b e n } } } \alpha _ { t , i } { \mathbf r } _ { t , i } \bigg \| _ { 2 } } \\ & { \quad \leq \displaystyle \sum _ { i \in { \mathcal R } _ { t } \cap { \mathcal C } _ { \mathrm { b e n } } } \alpha _ { t , i } \| { \mathbf r } _ { t , i } \| _ { 2 } } \\ & { \quad \leq ( 1 - \mu _ { t } ) \frac { L _ { \operatorname* { m a x } } \eta _ { \mathrm { l o c } } G _ { B } ( E - 1 ) } { 2 } . } \end{array}\tag{190}
$$

This proves the lemma.

## APPENDIX C

## DETAILED EXPERIMENTAL SETTINGS

This appendix provides the complete experimental configuration used in Section V, including the datasets, tokenization, client partitions, model architectures, optimization settings, attack configuration, baseline parameters, and FEDLNS implementation details.

## A. Federated Learning Protocol

Table II summarizes the simulation protocol shared by all three model families: a population of K = 200 clients participates at rate 0.1 (20 clients per round) over 200 communication rounds, with local AdamW optimization (weight decay 0.01), evaluation after every round, and results averaged over three random seeds under both IID and non-IID partitions.

TABLE II: Federated simulation protocol used across model families.
<table><tr><td>Item</td><td>Value</td></tr><tr><td>Number of clients</td><td>K = 200</td></tr><tr><td>Participation rate</td><td>0.1</td></tr><tr><td>Clients per round</td><td>20</td></tr><tr><td>Communication rounds</td><td>200</td></tr><tr><td>Random seeds</td><td>100, 200, 300</td></tr><tr><td>Data partitions</td><td>IID and non-IID</td></tr><tr><td>Non-IID construction</td><td>Two shards per client</td></tr><tr><td>Local optimizer</td><td>AdamW</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Evaluation frequency</td><td>Every communication round</td></tr></table>

## B. Datasets and Tokenization

Table III lists the dataset, tokenizer, and sequence-construction choices for each model family, and Table IV describes how the resulting training units are partitioned across clients.

TABLE III: Dataset and tokenization settings. All models are initialized from scratch; tokenizer names specify tokenization only.
<table><tr><td>Model family</td><td>Dataset</td><td>Tokenizer</td><td>Sequence construction</td><td>Evaluation split used for selection and reporting</td></tr><tr><td>GPT-style LM</td><td>WikiText-2 raw</td><td>gpt2</td><td>Causal LM blocks of length 128</td><td>Official validation split</td></tr><tr><td>BERT-style MLM</td><td>Tiny Shakespeare</td><td>bert-base-uncased</td><td>MLM blocks of length 128 with [CLS]/[SEP]</td><td>Final 10% of the text</td></tr><tr><td>LLaMA-style LM</td><td>StackOverflow posts</td><td>TinyLlama/TinyLlama-1.1B-Ch€ausal LM blocks of length 128</td><td></td><td>Evaluation split de- fined during dataset construction</td></tr></table>

TABLE IV: Client data partition construction for IID and non-IID settings.
<table><tr><td>Model family</td><td>Training units before parti- tioning</td><td>IID partition</td><td>Non-IID partition</td></tr><tr><td>GPT-style LM / WikiText-2</td><td>Causal LM token blocks of length 128</td><td>Shuffle all block indices and assign them round-robin to 200 clients</td><td>Split ordered block indices into 400 contiguous shards; assign two shuffled shards per client</td></tr><tr><td>BERT-style MLM / Tiny Shakespeare</td><td>BERT MLM blocks of length 128 from the first 90% of the text</td><td>Shuffle all block indices and assign them round-robin to 200 clients</td><td>Split ordered block indices into 400 contiguous shards; assign two shuffled shards per client</td></tr><tr><td>LLaMA-style LM / Stack- Overflow</td><td>Causal LM token blocks of length 128 from up to 50,000 training posts</td><td>Shuffle all block indices and assign them round-robin to 200 clients</td><td>Split ordered block indices into 400 contiguous shards; assign two shuffled shards per client</td></tr></table>

For each model family, the training corpus is first tokenized and converted into fixed-length training blocks, after which the client partition is applied over the training-block indices. The evaluation split is never partitioned across clients. It is used for global checkpoint selection and for reporting test loss, perplexity, and token-level semantic entropy.

In the IID setting, all training block indices are shuffled using the experiment seed and distributed to the $K = 2 0 0$ clients in round-robin order, so each client receives an approximately equal number of blocks sampled from across the entire training corpus

In the non-IID setting, the ordered list of block indices is first divided into $K \times 2 = 4 0 0$ contiguous shards, the shard list is shuffled, and each client is assigned two shards. Clients therefore still have approximately balanced data sizes, but each one observes only a small number of localized regions of the corpus – inducing client-level heterogeneity through topic, style, speaker, or source locality rather than through class-label skew.

## C. Model Architectures and Optimization

Table V lists the scratch transformer architecture used for each model family, and Table VI lists the corresponding local optimization and evaluation settings.

TABLE V: Scratch transformer architectures.
<table><tr><td>Hyperparameter</td><td>GPT-style LM</td><td>BERT-style MLM</td><td>LLaMA-style LM</td></tr><tr><td>Hidden / embedding dimension</td><td>256</td><td>256</td><td>256</td></tr><tr><td>Number of layers</td><td>12</td><td>12</td><td>8</td></tr><tr><td>Attention heads</td><td>16</td><td>16</td><td>16</td></tr><tr><td>Key-value heads</td><td>一</td><td></td><td>8</td></tr><tr><td>Intermediate dimension</td><td></td><td>1024</td><td>1536</td></tr><tr><td>Context / block size</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Max position embeddings</td><td>128</td><td>130</td><td>256</td></tr><tr><td>Dropout</td><td>0.1</td><td>Default config</td><td>Default config</td></tr><tr><td>RMSNorm epsilon</td><td></td><td></td><td>10-6</td></tr><tr><td>Initializer range</td><td>Default config</td><td>Default config</td><td>0.02</td></tr></table>

TABLE VI: Optimization and evaluation settings.
<table><tr><td>Setting</td><td>GPT-style LM</td><td>BERT-style MLM</td><td>LLaMA-style LM</td></tr><tr><td>Local epochs</td><td>3</td><td>3</td><td>2</td></tr><tr><td>Batch size</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>10⁻5</td></tr><tr><td>Weight decay</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td>Train-evaluation batches</td><td>200</td><td>200</td><td>50</td></tr><tr><td>Evaluation batches</td><td>200</td><td>200</td><td>50</td></tr><tr><td>MLM masking probability</td><td></td><td>0.15</td><td>一</td></tr><tr><td>Maximum training examples</td><td>Full split</td><td>Full split</td><td>50,000</td></tr><tr><td>Maximum evaluation examples</td><td>Full split</td><td>Full split</td><td>5,000</td></tr></table>

## D. Attack and Aggregation-Baseline Configuration

Table VII details the target-corruption attack used throughout the experiments: input tokens are left unchanged, only training targets are corrupted, and each malicious client corrupts its labels with probability $p _ { \mathrm { c o r r } } ~ = ~ 1 . 0 .$ . The main text reports results at the malicious-client fraction $\alpha = 0 . 4$ (80 of 200 clients); this appendix evaluates the full sweep $\alpha \in \{ 0 , 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 \}$

TABLE VII: Target-corruption attack settings.
<table><tr><td>Item</td><td>Value</td></tr><tr><td>Attack type</td><td>Target corruption</td></tr><tr><td>Input tokens</td><td>Kept unchanged</td></tr><tr><td>Corrupted quantity</td><td>Training targets / labels</td></tr><tr><td>Corruption probability</td><td> $p _ { \mathrm { c o r r } } = 1 . 0$ </td></tr><tr><td>Malicious fractions in sweep</td><td>0, 0.1, 0.2, 0.3, 0.4</td></tr><tr><td>Strongest main-text attack fraction</td><td>0.4</td></tr><tr><td>Malicious clients in the population at 0.4</td><td>80 out of 200</td></tr><tr><td>Special-token handling</td><td>Padding labels avoided; EOS/padding avoided when sampling corrupt labels</td></tr></table>

Table VIII summarizes the aggregation methods compared against FEDLNS, and Table IX lists the attack-dependent parameters used for each population-level malicious-client fraction: the trim fraction for Trimmed Mean and the parameters f and m for Multi-Krum, with 20 clients selected per round. These two baselines are therefore configured using the population-level corruption fraction.

TABLE VIII: Aggregation methods compared in the experiments.
<table><tr><td>Method</td><td>Implementation summary</td></tr><tr><td>FedAvg</td><td>Sample-weighted average of client updates</td></tr><tr><td>Norm-Bounded FedAvg</td><td>Each update is clipped by an  $\ell _ { 2 }$  norm bound before FedAvg </td></tr><tr><td>Coordinate Median</td><td>Coordinate-wise median of flattened updates</td></tr><tr><td>Trimmed Mean</td><td>Coordinate-wise trimming followed by averaging</td></tr><tr><td>Multi-Krum</td><td>Distance-based update selection followed by averaging selected updates</td></tr><tr><td>FLAME</td><td>Two-cluster filtering, majority-cluster retention, norm clipping, and Gaussian noise</td></tr><tr><td>FEDLNS</td><td>Normalization-signature bank, median/MAD standardization, and BIC-guided GMM screening</td></tr></table>

TABLE IX: Robust-aggregation parameters used for each malicious-client fraction. The number of selected clients per round is 20.
<table><tr><td>Malicious fraction</td><td>Trim fraction</td><td>Multi-Krum f</td><td>Multi-Krum m</td></tr><tr><td>0.0</td><td>0.0</td><td>0</td><td>20</td></tr><tr><td>0.1</td><td>0.1</td><td>2</td><td>18</td></tr><tr><td>0.2</td><td>0.2</td><td>4</td><td>16</td></tr><tr><td>0.3</td><td>0.3</td><td>6</td><td>14</td></tr><tr><td>0.4</td><td>0.4</td><td>8</td><td>12</td></tr></table>

## E. FedLNS Implementation Details

Table X lists the FEDLNS implementation settings used in all experiments, covering normalization-parameter extraction, bank update timing, robust standardization, and BIC-guided one-vs-two-component GMM screening.

TABLE X: FedLNS implementation settings used in the experiments.
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Signature source</td><td>Trainable normalization-layer parameters whose names contain 1n or layernorm and end with weight or bias</td></tr><tr><td>Bank entry</td><td>One latest per-layer signature dictionary per client</td></tr><tr><td>Bank update timing</td><td>Before screening in the current round</td></tr><tr><td>Bank update rule</td><td>Latest replacement; EMA coefficient  $\beta = 0$ </td></tr><tr><td>Robust center</td><td>Coordinate-wise bank median</td></tr><tr><td>Robust scale</td><td>Coordinate-wise MAD</td></tr><tr><td>MAD floor</td><td> $1 0 ^ { - 6 }$ </td></tr><tr><td>Client deviation score</td><td>Median absolute standardized deviation</td></tr><tr><td>GMM candidates</td><td>One diagonal Gaussian vs. two diagonal Gaussians</td></tr><tr><td>GMM model selection</td><td>BIC for default FEDLNS</td></tr><tr><td>GMM covariance</td><td>Diagonal  $1 0 ^ { - 4 }$ </td></tr><tr><td>GMM regularization</td><td></td></tr><tr><td>GMM initialization</td><td>K-means initialization, 5 restarts</td></tr><tr><td>Maximum GMM iterations</td><td>200</td></tr><tr><td>Two-component retain rule</td><td>Keep component with smaller median standardized deviation</td></tr><tr><td>Safety fallback</td><td>If no client is retained, retain arg mini∈St  $\delta _ { t , i }$ </td></tr></table>

# APPENDIX D COMPLETE RESULTS ACROSS MALICIOUS-CLIENT FRACTIONS

This appendix reports the complete numerical results for the malicious-client fraction sweep, evaluated over

$$
\alpha \in \{ 0 , 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 \} ,\tag{191}
$$

which for $K = 2 0 0$ clients correspond to 0, 20, 40, 60, and 80 malicious clients, respectively.

All tables follow the reporting rule used in the main text: for each seed, the checkpoint with the lowest test loss is selected, and test loss, perplexity, and token-level semantic entropy are reported from that same model state. For each model, partition, aggregation method, and malicious-client fraction, each entry reports the mean and sample standard deviation over seeds {100, 200, 300}, with lower values indicating better performance for all three metrics.

The tables are grouped by model family and data partition. For each model and partition, they report test loss, perplexity, and token-level semantic entropy across the full attack-fraction sweep from the clean setting to the strongest evaluated setting of 40%.

## A. GPT-Style Causal Language Modeling on WikiText

Tables XI–XIII report the IID results for GPT/WikiText, and Tables XIV–XVI report the non-IID results.

TABLE XI: Test Loss under different malicious-client fractions for GPT / WikiText with IID partition. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>FedAvg</td><td> $7 . 1 3 \pm 0 . 0 1$ </td><td> $7 . 1 7 \pm 0 . 0 1$ </td><td> $7 . 2 3 \pm 0 . 0 0$ </td><td> $7 . 2 9 \pm 0 . 0 1$ </td><td> $7 . 3 7 \pm 0 . 0 2$ </td></tr><tr><td>Norm-Bound</td><td> $7 . 5 0 \pm 0 . 0 0$ </td><td> $7 . 5 1 \pm 0 . 0 1$ </td><td> $7 . 5 9 \pm 0 . 0 1$ </td><td> $7 . 6 9 \pm 0 . 0 2$ </td><td> $7 . 8 9 \pm 0 . 0 1$ </td></tr><tr><td>Median</td><td> $7 . 3 0 \pm 0 . 0 1$ </td><td> $7 . 2 7 \pm 0 . 0 1$ </td><td> $7 . 2 8 \pm 0 . 0 1$ </td><td> $7 . 3 0 \pm 0 . 0 1$ </td><td> $7 . 4 7 \pm 0 . 0 3$ </td></tr><tr><td>Trimmed</td><td> ${ \bf 7 . 1 2 \pm 0 . 0 1 }$ </td><td> $7 . 1 9 \pm 0 . 0 1$ </td><td> $7 . 2 5 \pm 0 . 0 1$ </td><td> $7 . 3 2 \pm 0 . 0 3$ </td><td> $7 . 4 3 \pm 0 . 0 1$ </td></tr><tr><td>Multi-Krum</td><td> $7 . 1 2 \pm 0 . 0 1$ </td><td> $7 . 1 5 \pm 0 . 0 1$ </td><td> $7 . 1 9 \pm 0 . 0 0$ </td><td> $7 . 2 1 \pm 0 . 0 0$ </td><td> $7 . 2 8 \pm 0 . 0 2$ </td></tr><tr><td>FLAME</td><td> $7 . 1 9 \pm 0 . 0 0$ </td><td> $7 . 1 9 \pm 0 . 0 1$ </td><td> $7 . 2 0 \pm 0 . 0 0$ </td><td> $7 . 2 1 \pm 0 . 0 0$ </td><td> $7 . 3 3 \pm 0 . 0 3$ </td></tr><tr><td>FedLNS</td><td> $7 . 1 3 \pm 0 . 0 0$ </td><td> ${ \bf 7 . 1 3 \pm 0 . 0 0 }$ </td><td> ${ \bf 7 . 1 3 \pm 0 . 0 0 }$ </td><td> ${ \bf 7 . 1 4 \pm 0 . 0 0 }$ </td><td> ${ \bf 7 . 1 6 \pm 0 . 0 0 }$ </td></tr></table>

TABLE XII: Test PPL under different malicious-client fractions for GPT / WikiText with IID partition. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold
<table><tr><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td></td><td>30%</td><td>40%</td></tr><tr><td>FedAvg</td><td> $1 2 4 2 . 9 1 \pm 6 . 5 8$ </td><td> $1 3 0 3 . 0 2 \pm 9 . 5 9$ </td><td> $1 3 7 3 . 6 1 \pm 1 . 6 3$ </td><td></td><td> $1 4 7 2 . 4 1 \pm 1 5 . 3 1$ </td><td> $1 5 9 0 . 5 6 \pm 3 5 . 5 1$ </td></tr><tr><td>Norm-Bound</td><td> $1 8 0 9 . 2 7 \pm 8 . 3 1$ </td><td> $1 8 2 8 . 1 3 \pm 1 0 . 6 9$ </td><td> $1 9 8 5 . 8 5 \pm 2 5 . 3 4$ </td><td></td><td> $2 1 9 6 . 8 4 \pm 3 3 . 9 9$ </td><td> $2 6 8 3 . 5 3 \pm 3 4 . 3 1$ </td></tr><tr><td>Median</td><td> $1 4 8 4 . 6 9 \pm 1 1 . 2 0$ </td><td> $1 4 4 1 . 4 4 \pm 9 . 3 3$ </td><td> $1 4 5 2 . 9 9 \pm 1 1 . 4 0$ </td><td></td><td> $1 4 7 7 . 2 6 \pm 1 7 . 6 5$ </td><td> $1 7 5 1 . 0 7 \pm 5 2 . 6 2$ </td></tr><tr><td>Trimmed</td><td> $\mathbf { 1 2 4 1 . 5 5 \pm 1 0 . 0 8 }$ </td><td> $1 3 2 6 . 1 5 \pm 1 7 . 4 5$ </td><td> $1 4 1 1 . 1 5 \pm 1 5 . 5 7$ </td><td></td><td> $1 5 1 6 . 8 4 \pm 4 3 . 7 1$ </td><td> $1 6 8 9 . 1 6 \pm 1 8 . 0 0$ </td></tr><tr><td>Multi-Krum</td><td> $1 2 4 2 . 0 8 \pm 8 . 4 0$ </td><td> $1 2 7 5 . 3 4 \pm 7 . 0 9$ </td><td> $1 3 2 0 . 4 1 \pm 5 . 6 3$ </td><td></td><td> $1 3 5 6 . 2 1 \pm 6 . 1 9$ </td><td> $1 4 4 9 . 7 2 \pm 2 2 . 3 6$ </td></tr><tr><td>FLAME</td><td> $1 3 2 3 . 5 5 \pm 4 . 8 7$ </td><td> $1 3 2 8 . 8 8 \pm 7 . 2 1$ </td><td> $1 3 4 5 . 0 5 \pm 2 . 8 2$ </td><td></td><td> $1 3 5 5 . 3 4 \pm 3 . 2 7$ </td><td> $1 5 2 4 . 0 2 \pm 4 0 . 7 1$ </td></tr><tr><td>FedLNS</td><td> $1 2 4 6 . 5 7 \pm 2 . 5 2$ </td><td> ${ \bf 1 2 4 7 . 2 8 \pm 5 . 9 7 }$ </td><td> ${ \bf 1 2 4 9 . 1 1 \pm 1 . 7 8 }$ </td><td></td><td> ${ \bf 1 2 5 8 . 2 1 \pm 4 . 7 0 }$ </td><td> $\mathbf { 1 2 8 1 . 6 3 \pm 1 . 7 4 }$ </td></tr></table>

TABLE XIII: Semantic Entropy under different malicious-client fractions for GPT / WikiText with IID partition. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>FedAvg</td><td> $6 . 8 0 \pm 0 . 0 6$ </td><td> $7 . 0 8 \pm 0 . 0 8$ </td><td> $7 . 3 7 \pm 0 . 1 8$ </td><td> $7 . 5 6 \pm 0 . 1 6$ </td><td> $7 . 9 6 \pm 0 . 0 7$ </td></tr><tr><td>Norm-Bound</td><td> ${ \bf 6 . 5 5 \pm 0 . 0 2 }$ </td><td> $8 . 0 4 \pm 0 . 1 2$ </td><td> $8 . 3 1 \pm 0 . 1 0$ </td><td> $8 . 9 2 \pm 0 . 0 3$ </td><td> $9 . 6 9 \pm 0 . 0 6$ </td></tr><tr><td>Median</td><td> $6 . 8 9 \pm 0 . 0 4$ </td><td> $7 . 3 0 \pm 0 . 1 5$ </td><td> $7 . 2 7 \pm 0 . 1 4$ </td><td> $7 . 6 1 \pm 0 . 1 3$ </td><td> $7 . 5 5 \pm 0 . 1 9$ </td></tr><tr><td>Trimmed</td><td> $6 . 7 7 \pm 0 . 0 3$ </td><td> $7 . 3 0 \pm 0 . 1 8$ </td><td> $7 . 2 9 \pm 0 . 1 9$ </td><td> $7 . 4 7 \pm 0 . 1 3$ </td><td> $7 . 5 5 \pm 0 . 3 9$ </td></tr><tr><td>Multi-Krum</td><td> $6 . 8 0 \pm 0 . 0 7$ </td><td> $7 . 1 4 \pm 0 . 2 4$ </td><td> $7 . 0 7 \pm 0 . 0 9$ </td><td> $7 . 0 3 \pm 0 . 1 1$ </td><td> $7 . 1 3 \pm 0 . 3 7$ </td></tr><tr><td>FLAME</td><td> $7 . 1 1 \pm 0 . 0 5$ </td><td> $7 . 1 3 \pm 0 . 1 0$ </td><td> $7 . 2 0 \pm 0 . 0 5$ </td><td> $7 . 1 1 \pm 0 . 0 9$ </td><td> $6 . 9 6 \pm 0 . 1 0$ </td></tr><tr><td>FedLNS</td><td> $6 . 7 3 \pm 0 . 0 3$ </td><td> ${ \bf 6 . 8 1 \pm 0 . 0 1 }$ </td><td> ${ \bf 6 . 7 5 \pm 0 . 0 2 }$ </td><td> ${ \bf 6 . 7 4 \pm 0 . 0 4 }$ </td><td> ${ \bf 6 . 8 0 \pm 0 . 0 6 }$ </td></tr></table>

TABLE XIV: Test Loss under different malicious-client fractions for GPT / WikiText with Non-IID partition. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>FedAvg</td><td> ${ \bf 7 . 1 0 \pm 0 . 0 1 }$ </td><td> $7 . 1 6 \pm 0 . 0 1$ </td><td> $7 . 2 3 \pm 0 . 0 4$ </td><td> $7 . 2 9 \pm 0 . 0 6$ </td><td> $7 . 3 7 \pm 0 . 0 8$ </td></tr><tr><td>Norm-Bound</td><td> $7 . 5 6 \pm 0 . 0 1$ </td><td> $7 . 5 4 \pm 0 . 0 0$ </td><td> $7 . 6 3 \pm 0 . 0 1$ </td><td> $7 . 7 5 \pm 0 . 0 2$ </td><td> $7 . 9 4 \pm 0 . 0 9$ </td></tr><tr><td>Median</td><td> $7 . 2 1 \pm 0 . 0 0$ </td><td> $7 . 1 8 \pm 0 . 0 0$ </td><td> $7 . 2 6 \pm 0 . 0 2$ </td><td> $7 . 3 3 \pm 0 . 0 2$ </td><td> $7 . 5 1 \pm 0 . 0 3$ </td></tr><tr><td>Trimmed</td><td> $7 . 1 1 \pm 0 . 0 1$ </td><td> $7 . 1 7 \pm 0 . 0 1$ </td><td> $7 . 2 5 \pm 0 . 0 1$ </td><td> $7 . 3 8 \pm 0 . 0 3$ </td><td> $7 . 5 1 \pm 0 . 0 5$ </td></tr><tr><td>Multi-Krum</td><td> $7 . 1 1 \pm 0 . 0 1$ </td><td> $7 . 1 8 \pm 0 . 0 0$ </td><td> $7 . 2 5 \pm 0 . 0 1$ </td><td> $7 . 3 0 \pm 0 . 0 0$ </td><td> $7 . 3 6 \pm 0 . 0 3$ </td></tr><tr><td>FLAME</td><td> $7 . 2 1 \pm 0 . 0 0$ </td><td> $7 . 2 5 \pm 0 . 0 1$ </td><td> $7 . 2 8 \pm 0 . 0 1$ </td><td> $7 . 2 9 \pm 0 . 0 1$ </td><td> $7 . 4 0 \pm 0 . 0 4$ </td></tr><tr><td>FedLNS</td><td> $7 . 1 1 \pm 0 . 0 1$ </td><td> ${ \bf 7 . 1 2 \pm 0 . 0 0 }$ </td><td> ${ \bf 7 . 1 3 \pm 0 . 0 1 }$ </td><td> ${ \bf 7 . 1 5 \pm 0 . 0 2 }$ </td><td> ${ \bf 7 . 1 6 \pm 0 . 0 1 }$ </td></tr></table>

TABLE XV: Test PPL under different malicious-client fractions for GPT / WikiText with Non-IID partition. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td></td><td>30%</td><td>40%</td></tr><tr><td>FedAvg</td><td> $\mathbf { 1 2 1 1 . 2 3 \ : \pm { \ : 6 . 7 0 } }$ </td><td> $1 2 8 2 . 0 8 \pm 1 3 . 2 2$ </td><td> $1 3 7 8 . 4 0 \pm 5 5 . 9 8$ </td><td></td><td> $1 4 7 1 . 7 9 \pm 9 0 . 0 9$ </td><td> $1 5 9 5 . 6 8 \pm 1 3 2 . 6 8$ </td></tr><tr><td>Norm-Bound</td><td> $1 9 1 6 . 4 7 \pm 1 3 . 0 0$ </td><td> $1 8 8 6 . 0 0 \pm 3 . 6 3$ </td><td> $2 0 6 0 . 0 1 \pm 1 3 . 6 3$ </td><td></td><td> $2 3 1 4 . 3 3 \pm 4 4 . 0 3$ </td><td> $2 8 1 1 . 0 7 \pm 2 5 7 . 2 9$ </td></tr><tr><td>Median</td><td> $1 3 5 0 . 3 2 \pm 1 . 3 4$ </td><td> $1 3 0 9 . 9 4 \pm 6 . 4 0$ </td><td> $1 4 1 9 . 4 6 \pm 3 2 . 2 7$ </td><td></td><td> $1 5 2 9 . 6 4 \pm 3 1 . 0 3$ </td><td> $1 8 2 0 . 6 2 \pm 4 8 . 3 1$ </td></tr><tr><td>Trimmed</td><td> $1 2 2 5 . 8 9 \pm 8 . 2 1$ </td><td> $1 2 9 5 . 2 9 \pm 1 6 . 4 1$ </td><td> $1 4 0 6 . 6 7 \pm 1 1 . 5 5$ </td><td></td><td> $1 5 9 9 . 6 0 \pm 5 2 . 8 5$ </td><td> $1 8 1 9 . 3 4 \pm 9 8 . 9 3$ </td></tr><tr><td>Multi-Krum</td><td> $1 2 2 5 . 0 7 \pm 8 . 0 6$ </td><td> $1 3 1 1 . 6 7 \pm 5 . 6 0$ </td><td> $1 4 0 8 . 7 9 \pm 2 0 . 4 5$ </td><td></td><td> $1 4 7 4 . 9 3 \pm 3 . 8 0$ </td><td> $1 5 7 9 . 8 1 \pm 4 3 . 9 4$ </td></tr><tr><td>FLAME</td><td> $1 3 5 9 . 5 8 \pm 3 . 8 2$ </td><td> $1 4 1 5 . 1 9 \pm 1 3 . 3 2$ </td><td> $1 4 4 9 . 9 5 \pm 1 4 . 7 2$ </td><td></td><td> $1 4 6 7 . 8 5 \pm 1 4 . 7 5$ </td><td> $1 6 3 7 . 8 8 \pm 6 3 . 9 0$ </td></tr><tr><td>FedLNS</td><td> $1 2 2 5 . 9 0 \pm 1 2 . 0 6$ </td><td> ${ \bf 1 2 3 8 . 5 2 \pm 4 . 9 7 }$ </td><td> $\mathbf { 1 2 5 1 . 6 5 \ : \pm { \ : 1 1 . 3 7 } }$ </td><td></td><td> ${ \bf 1 2 7 7 . 2 8 } \pm \ : 2 3 . 5 2$ </td><td> $\mathbf { 1 2 8 6 . 5 0 \pm 1 5 . 3 4 }$ </td></tr></table>

TABLE XVI: Semantic Entropy under different malicious-client fractions for GPT / WikiText with Non-IID partition. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>FedAvg</td><td> $6 . 7 9 \pm 0 . 0 7$ </td><td> $7 . 3 1 \pm 0 . 1 7$ </td><td> $7 . 1 2 \pm 0 . 3 7$ </td><td> $7 . 6 1 \pm 0 . 3 0$ </td><td> $7 . 6 6 \pm 0 . 2 6$ </td></tr><tr><td>Norm-Bound</td><td> ${ \bf 6 . 3 7 \pm 0 . 0 2 }$ </td><td> $7 . 8 5 \pm 0 . 1 9$ </td><td> $8 . 4 4 \pm 0 . 2 0$ </td><td> $9 . 0 2 \pm 0 . 0 9$ </td><td> $9 . 6 0 \pm 0 . 3 6$ </td></tr><tr><td>Median</td><td> $6 . 7 4 \pm 0 . 1 5$ </td><td> $7 . 3 1 \pm 0 . 3 4$ </td><td> $7 . 1 4 \pm 0 . 3 2$ </td><td> $7 . 5 2 \pm 0 . 1 0$ </td><td> $7 . 4 5 \pm 0 . 2 5$ </td></tr><tr><td>Trimmed</td><td> $6 . 7 5 \pm 0 . 1 5$ </td><td> $7 . 3 6 \pm 0 . 0 7$ </td><td> $7 . 2 7 \pm 0 . 0 7$ </td><td> $7 . 3 9 \pm 0 . 1 1$ </td><td> $7 . 7 2 \pm 0 . 2 8$ </td></tr><tr><td>Multi-Krum</td><td> $6 . 7 9 \pm 0 . 1 7$ </td><td> $7 . 0 8 \pm 0 . 1 2$ </td><td> $7 . 0 7 \pm 0 . 4 5$ </td><td> $7 . 2 5 \pm 0 . 4 7$ </td><td> $7 . 1 8 \pm 0 . 4 6$ </td></tr><tr><td>FLAME</td><td> $7 . 0 4 \pm 0 . 0 3$ </td><td> $7 . 0 8 \pm 0 . 1 5$ </td><td> $7 . 3 2 \pm 0 . 2 0$ </td><td> $6 . 9 4 \pm 0 . 0 7$ </td><td> $7 . 1 3 \pm 0 . 5 8$ </td></tr><tr><td>FedLNS</td><td> $6 . 8 0 \pm 0 . 0 9$ </td><td> ${ \bf 6 . 7 9 \pm 0 . 0 5 }$ </td><td> ${ \bf 7 . 0 4 \pm 0 . 4 0 }$ </td><td> ${ \bf 6 . 9 0 \pm 0 . 0 2 }$ </td><td> ${ \bf 6 . 9 2 \pm 0 . 2 4 }$ </td></tr></table>

## B. BERT-Style Masked Language Modeling on Tiny Shakespeare

Tables XVII–XIX report the IID results for BERT/Tiny Shakespeare, and Tables XX–XXII report the non-IID results. The reported perplexity is masked-token perplexity and should be interpreted within the masked-language-modeling setting.

TABLE XVII: Test Loss under different malicious-client fractions for BERT / Shakespeare with IID partition. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>FedAvg</td><td> $6 . 4 9 \pm 0 . 0 3$ </td><td> $6 . 5 4 \pm 0 . 0 1$ </td><td> $6 . 5 7 \pm 0 . 0 1$ </td><td> $6 . 6 2 \pm 0 . 0 2$ </td><td> $6 . 6 8 \pm 0 . 0 2$ </td></tr><tr><td>Norm-Bound</td><td> $6 . 8 0 \pm 0 . 0 3$ </td><td> $6 . 8 0 \pm 0 . 0 1$ </td><td> $6 . 9 2 \pm 0 . 0 1$ </td><td> $7 . 1 1 \pm 0 . 0 1$ </td><td> $7 . 3 0 \pm 0 . 0 5$ </td></tr><tr><td>Median</td><td> $6 . 8 0 \pm 0 . 0 3$ </td><td> $6 . 8 2 \pm 0 . 0 1$ </td><td> $6 . 8 2 \pm 0 . 0 2$ </td><td> $6 . 8 0 \pm 0 . 0 3$ </td><td> $6 . 9 7 \pm 0 . 0 8$ </td></tr><tr><td>Trimmed</td><td> $6 . 4 9 \pm 0 . 0 3$ </td><td> $6 . 6 6 \pm 0 . 0 1$ </td><td> $6 . 7 2 \pm 0 . 0 2$ </td><td> $6 . 8 0 \pm 0 . 0 3$ </td><td> $6 . 9 7 \pm 0 . 0 7$ </td></tr><tr><td>Multi-Krum</td><td> $6 . 4 9 \pm 0 . 0 3$ </td><td> $6 . 5 1 \pm 0 . 0 1$ </td><td> $6 . 5 2 \pm 0 . 0 1$ </td><td> $6 . 5 4 \pm 0 . 0 2$ </td><td> $6 . 5 9 \pm 0 . 0 2$ </td></tr><tr><td>FLAME</td><td> ${ \bf 6 . 4 9 \pm 0 . 0 3 }$ </td><td> ${ \bf 6 . 4 7 \pm 0 . 0 2 }$ </td><td> ${ \bf 6 . 4 9 \pm 0 . 0 2 }$ </td><td> $6 . 5 2 \pm 0 . 0 0$ </td><td> $6 . 7 0 \pm 0 . 1 8$ </td></tr><tr><td>FedLNS</td><td> $6 . 5 0 \pm 0 . 0 5$ </td><td> $6 . 4 9 \pm 0 . 0 2$ </td><td> ${ \bf 6 . 4 9 \pm 0 . 0 2 }$ </td><td> ${ \bf 6 . 5 1 \pm 0 . 0 1 }$ </td><td> ${ \bf 6 . 5 5 \pm 0 . 0 2 }$ </td></tr></table>

TABLE XVIII: Test PPL under different malicious-client fractions for BERT / Shakespeare with IID partition. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>FedAvg</td><td> $6 5 9 . 9 3 \pm 2 1 . 9 2$ </td><td> $6 9 3 . 0 1 \pm 4 . 3 6$ </td><td> $7 1 0 . 1 5 \pm 6 . 1 9$ </td><td> $7 5 0 . 0 9 \pm 1 2 . 0 8$ </td><td> $7 9 4 . 5 4 \pm 1 5 . 0 1$ </td></tr><tr><td>Norm-Bound</td><td> $9 0 1 . 0 4 \pm 2 9 . 4 7$ </td><td> $8 9 7 . 1 7 \pm 1 0 . 1 3$ </td><td> $1 0 0 8 . 6 1 \pm 7 . 0 0$ </td><td> $1 2 2 7 . 2 3 \pm 1 5 . 3 6$ </td><td> $1 4 7 7 . 6 7 \pm 7 0 . 8 4$ </td></tr><tr><td>Median</td><td> $8 9 4 . 8 5 \pm 2 7 . 5 8$ </td><td> $9 1 2 . 2 9 \pm 9 . 0 7$ </td><td> $9 1 3 . 3 8 \pm 1 9 . 0 6$ </td><td> $8 9 8 . 4 6 \pm 2 2 . 5 5$ </td><td> $1 0 6 6 . 1 3 \pm 9 1 . 5 6$ </td></tr><tr><td>Trimmed</td><td> $6 6 0 . 9 9 \pm 2 0 . 1 0$ </td><td> $7 7 9 . 9 4 \pm 9 . 1 8$ </td><td> $8 3 1 . 1 9 \pm 2 0 . 6 0$ </td><td> $9 0 0 . 6 5 \pm 2 6 . 8 8$ </td><td> $1 0 6 2 . 8 0 \pm 7 9 . 4 9$ </td></tr><tr><td>Multi-Krum</td><td> $6 6 1 . 0 0 \pm 2 0 . 1 0$ </td><td> $6 7 2 . 8 9 \pm 7 . 9 7$ </td><td> $6 7 6 . 4 4 \pm 4 . 5 7$ </td><td> $6 9 5 . 3 5 \pm 1 2 . 2 1$ </td><td> $7 2 5 . 7 6 \pm 1 7 . 8 4$ </td></tr><tr><td>FLAME</td><td> ${ \bf 6 5 8 . 5 2 \pm 1 8 . 9 5 }$ </td><td> ${ \bf 6 4 5 . 2 6 \pm 1 5 . 4 7 }$ </td><td> ${ \bf 6 5 7 . 2 1 \pm 1 0 . 5 3 }$ </td><td> $6 7 9 . 9 8 \pm 3 . 2 6$ </td><td> $8 1 8 . 5 7 \pm 1 5 7 . 5 3$ </td></tr><tr><td>FedLNS</td><td> $6 6 3 . 6 8 \pm 2 9 . 7 6$ </td><td> $6 5 5 . 9 7 \pm 1 1 . 6 6$ </td><td> $6 6 1 . 3 8 \pm 1 1 . 9 4$ </td><td> ${ \bf 6 6 9 . 3 7 \pm 5 . 8 6 }$ </td><td> ${ \bf 6 9 7 . 6 2 \pm 1 3 . 6 2 }$ </td></tr></table>

TABLE XIX: Semantic Entropy under different malicious-client fractions for BERT / Shakespeare with IID partition. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>FedAvg</td><td> ${ \bf 6 . 2 6 \pm 0 . 1 8 }$ </td><td> $6 . 7 9 \pm 0 . 1 9$ </td><td> $6 . 9 5 \pm 0 . 0 7$ </td><td> $7 . 2 2 \pm 0 . 3 6$ </td><td> $7 . 3 7 \pm 0 . 3 3$ </td></tr><tr><td>Norm-Bound</td><td> $6 . 3 1 \pm 0 . 0 3$ </td><td> $7 . 2 7 \pm 0 . 2 1$ </td><td> $7 . 6 5 \pm 0 . 0 7$ </td><td> $8 . 0 7 \pm 0 . 1 5$ </td><td> $8 . 5 4 \pm 0 . 1 6$ </td></tr><tr><td>Median</td><td> $6 . 6 0 \pm 0 . 0 6$ </td><td> $6 . 9 2 \pm 0 . 2 2$ </td><td> $7 . 0 5 \pm 0 . 3 4$ </td><td> $7 . 4 1 \pm 0 . 2 2$ </td><td> $7 . 4 7 \pm 0 . 7 8$ </td></tr><tr><td>Trimmed</td><td> $6 . 2 8 \pm 0 . 1 6$ </td><td> $6 . 6 3 \pm 0 . 1 7$ </td><td> $6 . 9 7 \pm 0 . 1 8$ </td><td> $7 . 3 5 \pm 0 . 5 4$ </td><td> $6 . 7 6 \pm 0 . 4 7$ </td></tr><tr><td>Multi-Krum</td><td> $6 . 2 8 \pm 0 . 1 6$ </td><td> $6 . 4 0 \pm 0 . 0 7$ </td><td> $6 . 6 9 \pm 0 . 3 8$ </td><td> $6 . 5 3 \pm 0 . 3 6$ </td><td> $6 . 6 1 \pm 0 . 2 8$ </td></tr><tr><td>FLAME</td><td> $6 . 2 6 \pm 0 . 0 3$ </td><td> ${ \bf 6 . 2 6 \pm 0 . 1 1 }$ </td><td> ${ \bf 6 . 1 1 \pm 0 . 0 5 }$ </td><td> ${ \bf 6 . 1 5 \pm 0 . 1 0 }$ </td><td> $6 . 4 6 \pm 0 . 1 6$ </td></tr><tr><td>FedLNS</td><td> $6 . 2 8 \pm 0 . 0 9$ </td><td> $6 . 3 8 \pm 0 . 1 6$ </td><td> $6 . 1 9 \pm 0 . 0 9$ </td><td> $6 . 2 4 \pm 0 . 1 0$ </td><td> ${ \bf 6 . 2 4 \pm 0 . 0 6 }$ </td></tr></table>

TABLE XX: Test Loss under different malicious-client fractions for BERT / Shakespeare with Non-IID partition. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowes test loss. The best case is highlighted in bold.
<table><tr><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>FedAvg</td><td> ${ \bf 6 . 3 6 \pm 0 . 0 3 }$ </td><td> $6 . 3 8 \pm 0 . 0 4$ </td><td> $6 . 5 1 \pm 0 . 1 0$ </td><td> $6 . 5 6 \pm 0 . 1 2$ </td><td> $6 . 5 9 \pm 0 . 1 1$ </td></tr><tr><td>Norm-Bound</td><td> $6 . 8 3 \pm 0 . 0 1$ </td><td> $6 . 8 5 \pm 0 . 0 1$ </td><td> $6 . 9 5 \pm 0 . 0 1$ </td><td> $7 . 1 4 \pm 0 . 0 2$ </td><td> $7 . 3 4 \pm 0 . 0 2$ </td></tr><tr><td>Median</td><td> $6 . 8 6 \pm 0 . 0 1$ </td><td> $6 . 8 9 \pm 0 . 0 1$ </td><td> $6 . 8 6 \pm 0 . 0 3$ </td><td> $6 . 8 6 \pm 0 . 0 1$ </td><td> $6 . 9 5 \pm 0 . 0 5$ </td></tr><tr><td>Trimmed</td><td> $6 . 5 4 \pm 0 . 0 3$ </td><td> $6 . 7 2 \pm 0 . 0 1$ </td><td> $6 . 8 0 \pm 0 . 0 2$ </td><td> $6 . 8 4 \pm 0 . 0 2$ </td><td> $6 . 9 4 \pm 0 . 0 2$ </td></tr><tr><td>Multi-Krum</td><td> $6 . 5 4 \pm 0 . 0 3$ </td><td> $6 . 5 9 \pm 0 . 0 2$ </td><td> $6 . 6 0 \pm 0 . 0 1$ </td><td> $6 . 6 1 \pm 0 . 0 2$ </td><td> $6 . 6 9 \pm 0 . 0 6$ </td></tr><tr><td>FLAME</td><td> $6 . 5 7 \pm 0 . 0 2$ </td><td> $6 . 5 6 \pm 0 . 0 3$ </td><td> $6 . 5 6 \pm 0 . 0 1$ </td><td> $6 . 5 7 \pm 0 . 0 1$ </td><td> $6 . 8 6 \pm 0 . 1 7$ </td></tr><tr><td>FedLNS</td><td> $6 . 4 7 \pm 0 . 0 5$ </td><td> ${ \bf 6 . 3 6 \pm 0 . 0 6 }$ </td><td> ${ \bf 6 . 4 2 \pm 0 . 1 4 }$ </td><td> ${ \bf 6 . 3 8 \pm 0 . 1 8 }$ </td><td> ${ \bf 6 . 4 8 \pm 0 . 1 3 }$ </td></tr></table>

TABLE XXI: Test PPL under different malicious-client fractions for BERT / Shakespeare with Non-IID partition. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>FedAvg</td><td> ${ \bf 5 7 6 . 4 9 \pm 1 9 . 7 6 }$ </td><td> $5 8 8 . 6 4 \pm 2 4 . 2 5$ </td><td> $6 7 2 . 5 5 \pm 6 6 . 8 4$ </td><td> $7 0 8 . 5 0 \pm 8 8 . 0 5$ </td><td> $7 3 1 . 7 3 \pm 8 3 . 1 5$ </td></tr><tr><td>Norm-Bound</td><td> $9 2 2 . 9 8 \pm 7 . 1 8$ </td><td> $9 4 1 . 7 0 \pm 9 . 9 5$ </td><td> $1 0 3 9 . 5 8 \pm 9 . 1 8$ </td><td> $1 2 6 2 . 1 6 \pm 2 6 . 2 9$ </td><td> $1 5 3 3 . 5 9 \pm 3 6 . 8 3$ </td></tr><tr><td>Median</td><td> $9 5 0 . 4 9 \pm 1 0 . 7 9$ </td><td> $9 8 5 . 5 8 \pm 6 . 4 7$ </td><td> $9 4 9 . 5 9 \pm 2 6 . 0 6$ </td><td> $9 5 3 . 3 2 \pm 1 0 . 7 6$ </td><td> $1 0 4 7 . 3 6 \pm 5 3 . 6 6$ </td></tr><tr><td>Trimmed</td><td> $6 9 1 . 2 7 \pm 1 9 . 1 2$ </td><td> $8 2 9 . 9 0 \pm 7 . 2 9$ </td><td> $8 9 3 . 8 8 \pm 1 9 . 3 7$ </td><td> $9 3 8 . 2 1 \pm 1 7 . 3 5$ </td><td> $1 0 3 3 . 3 5 \pm 2 4 . 2 8$ </td></tr><tr><td>Multi-Krum</td><td> $6 9 1 . 2 7 \pm 1 9 . 1 1$ </td><td> $7 2 7 . 4 3 \pm 1 3 . 6 0$ </td><td> $7 3 2 . 7 3 \pm 1 0 . 3 1$ </td><td> $7 3 9 . 8 3 \pm 1 2 . 5 2$ </td><td> $8 0 4 . 3 9 \pm 5 3 . 0 6$ </td></tr><tr><td>FLAME</td><td> $7 1 0 . 2 7 \pm 1 6 . 4 5$ </td><td> $7 0 5 . 4 7 \pm 2 3 . 8 2$ </td><td> $7 0 4 . 6 3 \pm 4 . 0 8$ </td><td> $7 1 2 . 0 1 \pm 8 . 2 4$ </td><td> $9 6 3 . 6 5 \pm 1 5 7 . 7 6$ </td></tr><tr><td>FedLNS</td><td> $6 4 3 . 5 6 \pm 3 3 . 4 6$ </td><td> ${ \pm 7 6 . 6 3 \pm 3 6 . 5 8 }$ </td><td> ${ \bf 6 1 8 . 3 0 \pm 9 1 . 1 2 }$ </td><td> $\mathbf { 5 9 4 . 6 1 \pm 1 0 8 . 7 5 }$ </td><td> ${ \bf 6 5 8 . 2 1 \pm 9 1 . 6 6 }$ </td></tr></table>

TABLE XXII: Semantic Entropy under different malicious-client fractions for BERT / Shakespeare with Non-IID partition. Values are mean $\pm$ standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>FedAvg</td><td> ${ \bf 5 . 8 7 \pm 0 . 1 4 }$ </td><td> $6 . 3 4 \pm 0 . 4 1$ </td><td> $6 . 9 6 \pm 0 . 2 0$ </td><td> $7 . 1 9 \pm 0 . 3 8$ </td><td> $6 . 9 4 \pm 0 . 1 4$ </td></tr><tr><td>Norm-Bound</td><td> $6 . 2 7 \pm 0 . 0 1$ </td><td> $6 . 9 9 \pm 0 . 1 4$ </td><td> $7 . 6 3 \pm 0 . 1 5$ </td><td> $7 . 9 8 \pm 0 . 1 9$ </td><td> $8 . 4 8 \pm 0 . 0 2$ </td></tr><tr><td>Median</td><td> $6 . 5 9 \pm 0 . 0 6$ </td><td> $7 . 2 8 \pm 0 . 1 0$ </td><td> $7 . 2 0 \pm 0 . 3 6$ </td><td> $7 . 4 1 \pm 0 . 2 6$ </td><td> $7 . 2 0 \pm 0 . 5 9$ </td></tr><tr><td>Trimmed</td><td> $6 . 2 4 \pm 0 . 1 0$ </td><td> $6 . 6 1 \pm 0 . 1 5$ </td><td> $7 . 0 0 \pm 0 . 1 3$ </td><td> $7 . 2 3 \pm 0 . 1 7$ </td><td> $6 . 7 5 \pm 0 . 2 7$ </td></tr><tr><td>Multi-Krum</td><td> $6 . 2 4 \pm 0 . 1 0$ </td><td> $6 . 6 7 \pm 0 . 2 4$ </td><td> $6 . 4 3 \pm 0 . 0 6$ </td><td> $6 . 6 9 \pm 0 . 1 3$ </td><td> $6 . 8 2 \pm 0 . 3 1$ </td></tr><tr><td>FLAME</td><td> $6 . 1 9 \pm 0 . 0 6$ </td><td> $6 . 2 7 \pm 0 . 1 6$ </td><td> ${ \bf 6 . 3 0 \pm 0 . 1 2 }$ </td><td> $6 . 4 1 \pm 0 . 1 2$ </td><td> $6 . 6 2 \pm 0 . 1 4$ </td></tr><tr><td>FedLNS</td><td> $6 . 1 2 \pm 0 . 0 5$ </td><td> ${ \bf 6 . 0 6 \pm 0 . 1 8 }$ </td><td> $6 . 3 3 \pm 0 . 0 9$ </td><td> ${ \bf 6 . 0 0 \pm 0 . 3 8 }$ </td><td> ${ \bf 6 . 1 9 \pm 0 . 1 3 }$ </td></tr></table>

## C. LLaMA-Style Causal Language Modeling on StackOverflow

Tables XXIII–XXV report the IID results for LLaMA/StackOverflow, and Tables XXVI–XXVIII report the non-IID results.

TABLE XXIII: Test Loss under different malicious-client fractions for LLaMA / StackOverflow with IID partition. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>FedAvg</td><td> ${ \bf 5 . 3 6 \pm 0 . 0 3 }$ </td><td> $5 . 4 0 \pm 0 . 0 3$ </td><td> $5 . 4 6 \pm 0 . 0 4$ </td><td> $5 . 5 3 \pm 0 . 0 2$ </td><td> $5 . 6 6 \pm 0 . 0 4$ </td></tr><tr><td>Norm-Bound</td><td> $5 . 4 9 \pm 0 . 0 3$ </td><td> $5 . 5 6 \pm 0 . 0 3$ </td><td> $5 . 6 3 \pm 0 . 0 4$ </td><td> $5 . 7 7 \pm 0 . 0 2$ </td><td> $5 . 8 6 \pm 0 . 0 8$ </td></tr><tr><td>Median</td><td> $5 . 3 9 \pm 0 . 0 3$ </td><td> $5 . 4 0 \pm 0 . 0 3$ </td><td> $5 . 4 3 \pm 0 . 0 3$ </td><td> $5 . 4 6 \pm 0 . 0 3$ </td><td> $5 . 5 4 \pm 0 . 0 3$ </td></tr><tr><td>Trimmed</td><td> $5 . 3 6 \pm 0 . 0 3$ </td><td> $5 . 3 9 \pm 0 . 0 3$ </td><td> $5 . 4 3 \pm 0 . 0 3$ </td><td> $5 . 4 7 \pm 0 . 0 3$ </td><td> $5 . 5 4 \pm 0 . 0 4$ </td></tr><tr><td>Multi-Krum</td><td> $5 . 3 6 \pm 0 . 0 3$ </td><td> $5 . 3 7 \pm 0 . 0 3$ </td><td> $5 . 3 8 \pm 0 . 0 3$ </td><td> $5 . 3 9 \pm 0 . 0 3$ </td><td> $5 . 4 3 \pm 0 . 0 4$ </td></tr><tr><td>FLAME</td><td> $5 . 4 8 \pm 0 . 0 3$ </td><td> $5 . 4 8 \pm 0 . 0 3$ </td><td> $5 . 4 8 \pm 0 . 0 3$ </td><td> $5 . 4 9 \pm 0 . 0 3$ </td><td> $5 . 5 8 \pm 0 . 0 5$ </td></tr><tr><td>FedLNS</td><td> ${ \bf 5 . 3 6 \pm 0 . 0 3 }$ </td><td> ${ \bf 5 . 3 7 \pm 0 . 0 3 }$ </td><td> ${ \bf 5 . 3 7 \pm 0 . 0 3 }$ </td><td> ${ \bf 5 . 3 7 \pm 0 . 0 3 }$ </td><td> ${ \bf 5 . 3 8 \pm 0 . 0 3 }$ </td></tr></table>

TABLE XXIV: Test PPL under different malicious-client fractions for LLaMA / StackOverflow with IID partition. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>FedAvg</td><td> ${ \bf 2 1 3 . 1 5 \pm 6 . 7 7 }$ </td><td> $2 2 1 . 9 4 \pm 6 . 6 3$ </td><td> $2 3 4 . 0 3 \pm 8 . 5 3$ </td><td> $2 5 1 . 0 1 \pm 5 . 8 8$ </td><td> $2 8 6 . 7 3 \pm 1 0 . 9 9$ </td></tr><tr><td>Norm-Bound</td><td> $2 4 3 . 2 3 \pm 6 . 5 4$ </td><td> $2 5 8 . 6 3 \pm 6 . 5 4$ </td><td> $2 7 9 . 8 1 \pm 1 1 . 3 3$ </td><td> $3 1 9 . 1 0 \pm 6 . 0 4$ </td><td> $3 4 9 . 8 3 \pm 2 7 . 1 3$ </td></tr><tr><td>Median</td><td> $2 1 8 . 9 8 \pm 7 . 0 4$ </td><td> $2 2 2 . 3 9 \pm 7 . 1 0$ </td><td> $2 2 7 . 1 3 \pm 7 . 3 6$ </td><td> $2 3 5 . 4 4 \pm 6 . 6 0$ </td><td> $2 5 4 . 9 4 \pm 8 . 0 6$ </td></tr><tr><td>Trimmed</td><td> $2 1 3 . 1 5 \pm 6 . 7 7$ </td><td> $2 1 9 . 6 0 \pm 7 . 1 6$ </td><td> $2 2 7 . 4 9 \pm 7 . 8 0$ </td><td> $2 3 7 . 4 3 \pm 7 . 5 1$ </td><td> $2 5 4 . 5 2 \pm 9 . 0 7$ </td></tr><tr><td>Multi-Krum</td><td> $2 1 3 . 1 5 \pm 6 . 7 7$ </td><td> $2 1 5 . 5 3 \pm 7 . 0 4$ </td><td> $2 1 7 . 3 8 \pm 6 . 9 9$ </td><td> $2 1 9 . 7 6 \pm 6 . 9 2$ </td><td> $2 2 8 . 5 2 \pm 8 . 7 2$ </td></tr><tr><td>FLAME</td><td> $2 4 0 . 0 2 \pm 6 . 4 0$ </td><td> $2 3 9 . 8 5 \pm 6 . 7 7$ </td><td> $2 4 0 . 3 6 \pm 6 . 8 1$ </td><td> $2 4 2 . 5 4 \pm 7 . 3 3$ </td><td> $2 6 4 . 8 0 \pm 1 3 . 7 0$ </td></tr><tr><td>FedLNS</td><td> $2 1 3 . 4 2 \pm 6 . 5 7$ </td><td> ${ \bf 2 1 3 . 9 3 \pm 6 . 5 6 }$ </td><td> $\pm 1 4 . 5 7 \pm 6 . 7 5$ </td><td> ${ \bf 2 1 5 . 6 1 \pm 6 . 7 0 }$ </td><td> ${ \bf 2 1 7 . 1 4 \pm 6 . 8 5 }$ </td></tr></table>

TABLE XXV: Semantic Entropy under different malicious-client fractions for LLaMA / StackOverflow with IID partition. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>FedAvg</td><td> $5 . 5 2 \pm 0 . 0 2$ </td><td> $5 . 8 0 \pm 0 . 0 2$ </td><td> $5 . 9 7 \pm 0 . 1 2$ </td><td> $6 . 1 2 \pm 0 . 0 6$ </td><td> $6 . 5 8 \pm 0 . 3 4$ </td></tr><tr><td>Norm-Bound</td><td> $5 . 7 0 \pm 0 . 0 1$ </td><td> $5 . 9 6 \pm 0 . 0 1$ </td><td> $6 . 4 3 \pm 0 . 2 2$ </td><td> $6 . 8 3 \pm 0 . 0 7$ </td><td> $6 . 9 5 \pm 0 . 4 3$ </td></tr><tr><td>Median</td><td> $5 . 5 5 \pm 0 . 0 2$ </td><td> $5 . 7 3 \pm 0 . 0 3$ </td><td> $5 . 8 1 \pm 0 . 0 8$ </td><td> $5 . 9 1 \pm 0 . 0 2$ </td><td> $5 . 9 7 \pm 0 . 0 9$ </td></tr><tr><td>Trimmed</td><td> ${ \bf 5 . 5 2 \pm 0 . 0 2 }$ </td><td> $5 . 7 3 \pm 0 . 0 4$ </td><td> $5 . 8 1 \pm 0 . 0 9$ </td><td> $5 . 8 9 \pm 0 . 0 4$ </td><td> $5 . 9 5 \pm 0 . 0 8$ </td></tr><tr><td>Multi-Krum</td><td> ${ \bf 5 . 5 2 \pm 0 . 0 2 }$ </td><td> $5 . 5 4 \pm 0 . 0 4$ </td><td> $5 . 5 5 \pm 0 . 0 3$ </td><td> $5 . 5 6 \pm 0 . 0 3$ </td><td> $5 . 6 1 \pm 0 . 0 3$ </td></tr><tr><td>FLAME</td><td> $5 . 6 4 \pm 0 . 0 5$ </td><td> $5 . 6 1 \pm 0 . 0 4$ </td><td> $5 . 6 4 \pm 0 . 0 5$ </td><td> $5 . 6 5 \pm 0 . 0 4$ </td><td> $5 . 7 4 \pm 0 . 0 5$ </td></tr><tr><td>FedLNS</td><td> ${ \bf 5 . 5 2 \pm 0 . 0 2 }$ </td><td> ${ \bf 5 . 5 2 \pm 0 . 0 4 }$ </td><td> ${ \bf 5 . 5 4 \pm 0 . 0 3 }$ </td><td> ${ \bf 5 . 5 2 \pm 0 . 0 4 }$ </td><td> ${ \bf 5 . 5 2 \pm 0 . 0 2 }$ </td></tr></table>

TABLE XXVI: Test Loss under different malicious-client fractions for LLaMA / StackOverflow with Non-IID partition. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>FedAvg</td><td> ${ \bf 5 . 3 7 \pm 0 . 0 3 }$ </td><td> $5 . 4 1 \pm 0 . 0 3$ </td><td> $5 . 4 7 \pm 0 . 0 4$ </td><td> $5 . 5 4 \pm 0 . 0 3$ </td><td> $5 . 6 7 \pm 0 . 0 3$ </td></tr><tr><td>Norm-Bound</td><td> $5 . 5 1 \pm 0 . 0 3$ </td><td> $5 . 5 7 \pm 0 . 0 3$ </td><td> $5 . 6 5 \pm 0 . 0 4$ </td><td> $5 . 7 9 \pm 0 . 0 2$ </td><td> $5 . 8 3 \pm 0 . 0 5$ </td></tr><tr><td>Median</td><td> $5 . 4 0 \pm 0 . 0 3$ </td><td> $5 . 4 2 \pm 0 . 0 3$ </td><td> $5 . 4 4 \pm 0 . 0 3$ </td><td> $5 . 4 8 \pm 0 . 0 3$ </td><td> $5 . 5 6 \pm 0 . 0 3$ </td></tr><tr><td>Trimmed</td><td> $5 . 3 7 \pm 0 . 0 3$ </td><td> $5 . 4 0 \pm 0 . 0 3$ </td><td> $5 . 4 4 \pm 0 . 0 4$ </td><td> $5 . 4 9 \pm 0 . 0 3$ </td><td> $5 . 5 6 \pm 0 . 0 4$ </td></tr><tr><td>Multi-Krum</td><td> $5 . 3 7 \pm 0 . 0 3$ </td><td> $5 . 3 8 \pm 0 . 0 3$ </td><td> $5 . 3 9 \pm 0 . 0 3$ </td><td> $5 . 4 1 \pm 0 . 0 3$ </td><td> $5 . 4 5 \pm 0 . 0 4$ </td></tr><tr><td>FLAME</td><td> $5 . 4 9 \pm 0 . 0 3$ </td><td> $5 . 4 9 \pm 0 . 0 3$ </td><td> $5 . 4 9 \pm 0 . 0 3$ </td><td> $5 . 5 0 \pm 0 . 0 3$ </td><td> $5 . 5 9 \pm 0 . 0 4$ </td></tr><tr><td>FedLNS</td><td> ${ \bf 5 . 3 7 \pm 0 . 0 3 }$ </td><td> ${ \bf 5 . 3 8 \pm 0 . 0 3 }$ </td><td> ${ \bf 5 . 3 8 \pm 0 . 0 3 }$ </td><td> ${ \bf 5 . 3 8 \pm 0 . 0 3 }$ </td><td> ${ \bf 5 . 3 9 \pm 0 . 0 3 }$ </td></tr></table>

TABLE XXVII: Test PPL under different malicious-client fractions for LLaMA / StackOverflow with Non-IID partition. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>FedAvg</td><td> $\mathbf { 2 1 5 . 3 8 \ : \pm { \ : 6 . 8 0 } }$ </td><td> $2 2 4 . 2 7 \pm 6 . 6 8$ </td><td> $2 3 7 . 3 5 \pm 9 . 1 2$ </td><td> $2 5 5 . 8 4 \pm 7 . 1 7$ </td><td> $2 8 8 . 7 4 \pm 8 . 3 1$ </td></tr><tr><td>Norm-Bound</td><td> $2 4 6 . 3 0 \pm 6 . 7 6$ </td><td> $2 6 1 . 7 3 \pm 6 . 6 6$ </td><td> $2 8 3 . 4 9 \pm 1 0 . 9 9$ </td><td> $3 2 6 . 4 2 \pm 6 . 2 6$ </td><td> $3 4 0 . 8 8 \pm 1 8 . 0 9$ </td></tr><tr><td>Median</td><td> $2 2 1 . 8 4 \pm 7 . 1 9$ </td><td> $2 2 5 . 5 1 \pm 7 . 3 8$ </td><td> $2 3 0 . 7 3 \pm 7 . 8 4$ </td><td> $2 3 9 . 5 1 \pm 6 . 9 9$ </td><td> $2 6 0 . 3 3 \pm 7 . 9 9$ </td></tr><tr><td>Trimmed</td><td> $2 1 5 . 5 5 \pm 6 . 9 2$ </td><td> $2 2 2 . 4 9 \pm 7 . 4 3$ </td><td> $2 3 0 . 9 6 \pm 8 . 2 1$ </td><td> $2 4 1 . 4 6 \pm 8 . 0 9$ </td><td> $2 5 9 . 5 6 \pm 9 . 6 2$ </td></tr><tr><td>Multi-Krum</td><td> $2 1 5 . 5 5 \pm 6 . 9 2$ </td><td> $2 1 8 . 1 0 \pm 7 . 5 2$ </td><td> $2 1 9 . 9 4 \pm 7 . 1 7$ </td><td> $2 2 2 . 7 4 \pm 7 . 6 3$ </td><td> $2 3 3 . 4 2 \pm 1 0 . 1 5$ </td></tr><tr><td>FLAME</td><td> $2 4 3 . 1 2 \pm 6 . 8 2$ </td><td> $2 4 2 . 9 9 \pm 6 . 8 5$ </td><td> $2 4 3 . 2 5 \pm 6 . 4 1$ </td><td> $2 4 5 . 5 8 \pm 7 . 1 4$ </td><td> $2 6 9 . 0 8 \pm 1 1 . 9 0$ </td></tr><tr><td>FedLNS</td><td> $2 1 5 . 5 5 \pm 6 . 8 5$ </td><td> ${ \bf 2 1 6 . 1 3 \pm 6 . 7 0 }$ </td><td> $\mathbf { 2 1 6 . 9 7 \ : \pm { \ : 6 . 9 1 } }$ </td><td> $\mathbf { 2 1 8 . 0 2 \ : \pm { \ : 7 . 0 7 } }$ </td><td> ${ \bf 2 2 0 . 1 8 \pm 6 . 4 5 }$ </td></tr></table>

TABLE XXVIII: Semantic Entropy under different malicious-client fractions for LLaMA / StackOverflow with Non-IID partition. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>Method</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>FedAvg</td><td> ${ \bf 5 . 5 3 \pm 0 . 0 3 }$ </td><td> $5 . 8 0 \pm 0 . 0 1$ </td><td> $5 . 9 4 \pm 0 . 1 1$ </td><td> $6 . 2 2 \pm 0 . 1 1$ </td><td> $6 . 6 1 \pm 0 . 2 7$ </td></tr><tr><td>Norm-Bound</td><td> $5 . 7 0 \pm 0 . 0 3$ </td><td> $5 . 9 8 \pm 0 . 0 1$ </td><td> $6 . 4 4 \pm 0 . 2 1$ </td><td> $6 . 9 0 \pm 0 . 0 9$ </td><td> $6 . 5 6 \pm 0 . 4 2$ </td></tr><tr><td>Median</td><td> $5 . 5 6 \pm 0 . 0 3$ </td><td> $5 . 7 6 \pm 0 . 0 2$ </td><td> $5 . 8 4 \pm 0 . 0 9$ </td><td> $5 . 9 1 \pm 0 . 0 4$ </td><td> $5 . 9 9 \pm 0 . 1 2$ </td></tr><tr><td>Trimmed</td><td> $5 . 5 3 \pm 0 . 0 3$ </td><td> $5 . 7 5 \pm 0 . 0 3$ </td><td> $5 . 8 4 \pm 0 . 1 0$ </td><td> $5 . 9 2 \pm 0 . 0 6$ </td><td> $5 . 9 6 \pm 0 . 1 0$ </td></tr><tr><td>Multi-Krum</td><td> $5 . 5 3 \pm 0 . 0 3$ </td><td> $5 . 5 5 \pm 0 . 0 4$ </td><td> $5 . 5 6 \pm 0 . 0 3$ </td><td> $5 . 5 6 \pm 0 . 0 2$ </td><td> $5 . 6 3 \pm 0 . 0 6$ </td></tr><tr><td>FLAME</td><td> $5 . 6 6 \pm 0 . 0 6$ </td><td> $5 . 6 3 \pm 0 . 0 6$ </td><td> $5 . 6 2 \pm 0 . 0 1$ </td><td> $5 . 6 6 \pm 0 . 0 7$ </td><td> $5 . 7 4 \pm 0 . 0 6$ </td></tr><tr><td>FedLNS</td><td> $5 . 5 4 \pm 0 . 0 3$ </td><td> ${ \bf 5 . 5 4 \pm 0 . 0 3 }$ </td><td> ${ \bf 5 . 5 3 \pm 0 . 0 3 }$ </td><td> ${ \bf 5 . 5 3 \pm 0 . 0 2 }$ </td><td> ${ \bf 5 . 5 2 \pm 0 . 0 3 }$ </td></tr></table>

## APPENDIX E BANK COVERAGE AT ACTIVATION

This appendix studies the amount of bank coverage available when FEDLNS first activates screening and global aggregation. Appendix A analyzes the benign-majority probability of a partially populated bank; here, we evaluate how broader activation coverage affects the early active training trajectory.

## A. Reporting Protocol

The bank activation threshold determines when the server begins screening and global aggregation. We therefore focus on the early active period immediately after each threshold is reached, where differences in bank coverage and activation timing are most visible.

For each seed and activation coverage, the activation round is the first communication round in which the bank reaches $M _ { \mathrm { { a c t } } }$ . We then consider the first ten active aggregation rounds. Within this window, the checkpoint with the lowest test loss is selected, and test loss, perplexity, and token-level semantic entropy are reported from the same checkpoint.

## B. Activation Rounds

Tables XXIX–XXXVIII report the activation rounds for all population-level malicious-client fractions and partitions. The observed timing follows the pattern analyzed in Appendix A: $M _ { \mathrm { a c t } } = 0$ activates immediately, while thresholds covering 25%, 50%, and 75% of the client population activate after approximately 3, 7, and 13–14 rounds, respectively.

TABLE XXIX: Estimated activation rounds for the bank-size ablation under IID partition with 0% malicious clients. The activation round is the first communication round where the bank reaches $M _ { \mathrm { { a c t } } }$
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td></tr><tr><td>25%</td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td></tr><tr><td>50%</td><td> $6 . 7 \pm 0 . 6$ </td><td> $6 . 7 \pm 0 . 6$ </td><td> $6 . 7 \pm 0 . 6$ </td></tr><tr><td>75%</td><td> $1 3 . 3 \pm 0 . 6$ </td><td> $1 3 . 3 \pm 0 . 6$ </td><td> $1 3 . 3 \pm 0 . 6$ </td></tr></table>

TABLE XXX: Estimated activation rounds for the bank-size ablation under IID partition with 10% malicious clients. The activation round is the first communication round where the bank reaches $M _ { \mathrm { a c t } }$
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td></tr><tr><td>25%</td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td></tr><tr><td>50%</td><td> $7 . 0 \pm 0 . 0$ </td><td> $7 . 0 \pm 0 . 0$ </td><td> $7 . 0 \pm 0 . 0$ </td></tr><tr><td>75%</td><td> $1 3 . 0 \pm 0 . 0$ </td><td> $1 3 . 0 \pm 0 . 0$ </td><td> $1 3 . 0 \pm 0 . 0$ </td></tr></table>

TABLE XXXI: Estimated activation rounds for the bank-size ablation under IID partition with 20% malicious clients. The activation round is the first communication round where the bank reaches $M _ { \mathrm { a c t } }$
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td></tr><tr><td>25%</td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td></tr><tr><td>50%</td><td> $7 . 0 \pm 0 . 0$ </td><td> $7 . 0 \pm 0 . 0$ </td><td> $7 . 0 \pm 0 . 0$ </td></tr><tr><td>75%</td><td> $1 3 . 3 \pm 0 . 6$ </td><td> $1 3 . 3 \pm 0 . 6$ </td><td> $1 3 . 3 \pm 0 . 6$ </td></tr></table>

TABLE XXXII: Estimated activation rounds for the bank-size ablation under IID partition with 30% malicious clients. The activation round is the first communication round where the bank reaches $M _ { \mathrm { a c t } }$
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td></tr><tr><td>25%</td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td></tr><tr><td>50%</td><td> $7 . 3 \pm 0 . 6$ </td><td> $7 . 3 \pm 0 . 6$ </td><td> $7 . 3 \pm 0 . 6$ </td></tr><tr><td>75%</td><td> $1 4 . 0 \pm 1 . 0$ </td><td> $1 4 . 0 \pm 1 . 0$ </td><td> $1 4 . 0 \pm 1 . 0$ </td></tr></table>

TABLE XXXIII: Estimated activation rounds for the bank-size ablation under IID partition with 40% malicious clients. The activation round is the first communication round where the bank reaches $M _ { \mathrm { { a c t } } }$
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText 1</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td></tr><tr><td>25%</td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td></tr><tr><td>50%</td><td> $7 . 0 \pm 0 . 0$ </td><td> $7 . 0 \pm 0 . 0$ </td><td> $7 . 0 \pm 0 . 0$ </td></tr><tr><td>75%</td><td> $1 4 . 0 \pm 1 . 0$ </td><td> $1 4 . 0 \pm 1 . 0$ </td><td> $1 4 . 0 \pm 1 . 0$ </td></tr></table>

TABLE XXXIV: Estimated activation rounds for the bank-size ablation under Non-IID partition with 0% malicious clients. The activation round is the first communication round where the bank reaches $M _ { \mathrm { { a c t } } }$
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td></tr><tr><td>25%</td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td></tr><tr><td>50%</td><td> $6 . 7 \pm 0 . 6$ </td><td> $6 . 7 \pm 0 . 6$ </td><td> $6 . 7 \pm 0 . 6$ </td></tr><tr><td>75%</td><td> $1 3 . 3 \pm 0 . 6$ </td><td> $1 3 . 3 \pm 0 . 6$ </td><td> $1 3 . 3 \pm 0 . 6$ </td></tr></table>

TABLE XXXV: Estimated activation rounds for the bank-size ablation under Non-IID partition with 10% malicious clients. The activation round is the first communication round where the bank reaches $M _ { \mathrm { { a c t } } }$
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td></tr><tr><td>25%</td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td></tr><tr><td>50%</td><td> $7 . 0 \pm 0 . 0$ </td><td> $7 . 0 \pm 0 . 0$ </td><td> $7 . 0 \pm 0 . 0$ </td></tr><tr><td>75%</td><td> $1 3 . 0 \pm 0 . 0$ </td><td> $1 3 . 0 \pm 0 . 0$ </td><td> $1 3 . 0 \pm 0 . 0$ </td></tr></table>

TABLE XXXVI: Estimated activation rounds for the bank-size ablation under Non-IID partition with 20% malicious clients. The activation round is the first communication round where the bank reaches $M _ { \mathrm { { a c t } } }$
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td></tr><tr><td>25%</td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td></tr><tr><td>50%</td><td> $7 . 0 \pm 0 . 0$ </td><td> $7 . 0 \pm 0 . 0$ </td><td> $7 . 0 \pm 0 . 0$ </td></tr><tr><td>75%</td><td> $1 3 . 3 \pm 0 . 6$ </td><td> $1 3 . 3 \pm 0 . 6$ </td><td> $1 3 . 3 \pm 0 . 6$ </td></tr></table>

TABLE XXXVII: Estimated activation rounds for the bank-size ablation under Non-IID partition with 30% malicious clients. The activation round is the first communication round where the bank reaches $M _ { \mathrm { { a c t } } }$
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td></tr><tr><td>25%</td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td></tr><tr><td>50%</td><td> $7 . 3 \pm 0 . 6$ </td><td> $7 . 3 \pm 0 . 6$ </td><td> $7 . 3 \pm 0 . 6$ </td></tr><tr><td>75%</td><td> $1 4 . 0 \pm 1 . 0$ </td><td> $1 4 . 0 \pm 1 . 0$ </td><td> $1 4 . 0 \pm 1 . 0$ </td></tr></table>

TABLE XXXVIII: Estimated activation rounds for the bank-size ablation under Non-IID partition with 40% malicious clients. The activation round is the first communication round where the bank reaches $M _ { \mathrm { a c t } } .$
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td></tr><tr><td>25%</td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td><td> $3 . 0 \pm 0 . 0$ </td></tr><tr><td>50%</td><td> $7 . 0 \pm 0 . 0$ </td><td> $7 . 0 \pm 0 . 0$ </td><td> $7 . 0 \pm 0 . 0$ </td></tr><tr><td>75%</td><td> $1 4 . 0 \pm 1 . 0$ </td><td> $1 4 . 0 \pm 1 . 0$ </td><td> $1 4 . 0 \pm 1 . 0$ </td></tr></table>

## C. Results Under IID Partitions

Tables XXXIX–LIII report the IID bank-threshold results. Across model families and attack fractions, larger thresholds generally improve the early active-window values; these comparisons reflect both broader client coverage in the bank and later activation of global aggregation.

At the strongest 40% malicious-client setting, the 75% activation coverage gives the lowest test loss and perplexity for all three model families under IID partitioning: perplexity falls from 1820.21 ± 4.44 to $1 6 2 6 . 8 7 \pm 9 . 5 0$ for GPT/WikiText, from 1532.76 ± 114.78 to $1 0 4 3 . 9 6 \pm 2 2 . 2 0$ for BERT/Tiny Shakespeare, and from $2 4 0 8 . 1 5 \pm 1 6 2 . 2 5$ to $6 0 9 . 8 7 \pm 2 4 . 4 4$ for LLaMA/StackOverflow. Token-level semantic entropy also decreases substantially for BERT and LLaMA, indicating that broader bank coverage can stabilize the early active trajectory.

TABLE XXXIX: Bank-size threshold ablation under IID partition with 0% malicious clients. Metric: Test Loss. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $7 . 5 0 \pm 0 . 0 0$ </td><td> $7 . 3 3 \pm 0 . 0 1$ </td><td> $7 . 7 2 \pm 0 . 0 4$ </td></tr><tr><td>25%</td><td> $7 . 4 8 \pm 0 . 0 1$ </td><td> $7 . 2 0 \pm 0 . 0 3$ </td><td> $7 . 3 7 \pm 0 . 0 4$ </td></tr><tr><td>50%</td><td> $7 . 4 4 \pm 0 . 0 1$ </td><td> $7 . 0 4 \pm 0 . 0 4$ </td><td> $6 . 8 7 \pm 0 . 0 9$ </td></tr><tr><td>75%</td><td> ${ \bf 7 . 3 8 \pm 0 . 0 1 }$ </td><td> ${ \bf 6 . 9 4 \pm 0 . 0 2 }$ </td><td> ${ \bf 6 . 4 2 \pm 0 . 0 3 }$ </td></tr></table>

TABLE XL: Bank-size threshold ablation under IID partition with 0% malicious clients. Metric: Test PPL. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 8 0 2 . 4 0 \pm 2 . 9 7$ </td><td> $1 5 1 9 . 4 4 \pm 1 4 . 3 8$ </td><td> $2 2 5 8 . 0 1 \pm 8 8 . 6 8$ </td></tr><tr><td>25%</td><td> $1 7 6 4 . 1 6 \pm 1 1 . 6 0$ </td><td> $1 3 4 1 . 8 8 \pm 4 2 . 9 6$ </td><td> $1 5 8 4 . 7 1 \pm 6 0 . 0 1$ </td></tr><tr><td>50%</td><td> $1 6 9 5 . 1 1 \pm 1 8 . 3 7$ </td><td> $1 1 4 3 . 1 4 \pm 4 8 . 5 8$ </td><td> $9 6 9 . 7 2 \pm 8 3 . 5 0$ </td></tr><tr><td>75%</td><td> $\mathbf { 1 6 0 1 . 8 3 \ : \pm { \ : 1 7 . 2 5 } }$ </td><td> $\mathbf { 1 0 2 8 . 4 7 \pm 2 4 . 4 0 }$ </td><td> ${ \bf 6 1 6 . 5 0 \pm 1 9 . 8 3 }$ </td></tr></table>

TABLE XLI: Bank-size threshold ablation under IID partition with $0 \%$ malicious clients. Metric: Semantic Entropy. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $7 . 0 1 \pm 0 . 0 3$ </td><td> $7 . 4 6 \pm 0 . 0 2$ </td><td> $9 . 9 5 \pm 0 . 0 2$ </td></tr><tr><td>25%</td><td> $6 . 9 6 \pm 0 . 0 9$ </td><td> $7 . 2 6 \pm 0 . 3 8$ </td><td> $9 . 6 7 \pm 0 . 0 3$ </td></tr><tr><td>50%</td><td> $6 . 9 0 \pm 0 . 0 0$ </td><td> $6 . 3 9 \pm 0 . 1 6$ </td><td> $8 . 8 8 \pm 0 . 1 8$ </td></tr><tr><td>75%</td><td> ${ \bf 6 . 9 0 \pm 0 . 0 6 }$ </td><td> ${ \bf 6 . 3 7 \pm 0 . 1 2 }$ </td><td> ${ \bf 7 . 4 9 \pm 0 . 0 4 }$ </td></tr></table>

TABLE XLII: Bank-size threshold ablation under IID partition with 10% malicious clients. Metric: Test Loss. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $7 . 5 0 \pm 0 . 0 0$ </td><td> $7 . 3 2 \pm 0 . 0 1$ </td><td> $7 . 7 2 \pm 0 . 0 4$ </td></tr><tr><td>25%</td><td> $7 . 4 8 \pm 0 . 0 1$ </td><td> $7 . 1 8 \pm 0 . 0 5$ </td><td> $7 . 3 7 \pm 0 . 0 4$ </td></tr><tr><td>50%</td><td> $7 . 4 4 \pm 0 . 0 1$ </td><td> $7 . 0 6 \pm 0 . 0 4$ </td><td> $6 . 8 4 \pm 0 . 0 3$ </td></tr><tr><td>75%</td><td> ${ \bf 7 . 3 9 \pm 0 . 0 0 }$ </td><td> ${ \bf 6 . 9 2 \pm 0 . 0 2 }$ </td><td> ${ \bf 6 . 4 4 \pm 0 . 0 3 }$ </td></tr></table>

TABLE XLIII: Bank-size threshold ablation under IID partition with 10% malicious clients. Metric: Test PPL. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 8 0 8 . 5 7 \pm 3 . 9 5$ </td><td> $1 5 1 2 . 5 1 \pm 1 2 . 1 5$ </td><td> $2 2 5 6 . 4 9 \pm 9 0 . 4 8$ </td></tr><tr><td>25%</td><td> $1 7 7 2 . 4 8 \pm 1 3 . 2 3$ </td><td> $1 3 1 0 . 3 1 \pm 7 1 . 2 9$ </td><td> $1 5 8 5 . 0 7 \pm 6 2 . 2 7$ </td></tr><tr><td>50%</td><td> $1 7 0 1 . 0 3 \pm 1 1 . 0 7$ </td><td> $1 1 6 3 . 3 4 \pm 4 2 . 3 6$ </td><td> $9 3 3 . 0 3 \pm 3 0 . 8 9$ </td></tr><tr><td>75%</td><td> $\mathbf { 1 6 1 5 . 8 2 } \pm 5 . 5 2$ </td><td> $\mathbf { 1 0 1 1 . 3 3 } \pm 2 \mathbf { 0 . 2 4 }$ </td><td> ${ \bf 6 2 5 . 7 4 } \pm { \bf 1 7 . 9 0 }$ </td></tr></table>

TABLE XLIV: Bank-size threshold ablation under IID partition with 10% malicious clients. Metric: Semantic Entropy. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $7 . 0 1 \pm 0 . 1 0$ </td><td> $7 . 5 1 \pm 0 . 0 4$ </td><td> $9 . 9 5 \pm 0 . 0 2$ </td></tr><tr><td>25%</td><td> $6 . 9 8 \pm 0 . 0 6$ </td><td> $6 . 8 8 \pm 0 . 1 2$ </td><td> $9 . 6 7 \pm 0 . 0 3$ </td></tr><tr><td>50%</td><td> $6 . 9 2 \pm 0 . 0 5$ </td><td> $6 . 4 5 \pm 0 . 0 1$ </td><td> $8 . 7 9 \pm 0 . 0 6$ </td></tr><tr><td>75%</td><td> ${ \bf 6 . 9 1 \pm 0 . 0 6 }$ </td><td> ${ \bf 6 . 3 3 \pm 0 . 0 3 }$ </td><td> $\mathbf { 7 . 5 4 \ : \pm { \ : 0 . 0 4 } }$ </td></tr></table>

TABLE XLV: Bank-size threshold ablation under IID partition with 20% malicious clients. Metric: Test Loss. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $7 . 5 0 \pm 0 . 0 1$ </td><td> $7 . 3 2 \pm 0 . 0 1$ </td><td> $7 . 7 2 \pm 0 . 0 4$ </td></tr><tr><td>25%</td><td> $7 . 4 8 \pm 0 . 0 1$ </td><td> $7 . 1 9 \pm 0 . 0 5$ </td><td> $7 . 3 7 \pm 0 . 0 4$ </td></tr><tr><td>50%</td><td> $7 . 4 3 \pm 0 . 0 1$ </td><td> $7 . 0 5 \pm 0 . 0 1$ </td><td> $6 . 8 4 \pm 0 . 0 3$ </td></tr><tr><td>75%</td><td> ${ \bf 7 . 3 9 \pm 0 . 0 1 }$ </td><td> ${ \bf 6 . 9 3 \pm 0 . 0 4 }$ </td><td> ${ \bf 6 . 4 3 \pm 0 . 0 2 }$ </td></tr></table>

TABLE XLVI: Bank-size threshold ablation under IID partition with 20% malicious clients. Metric: Test PPL. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 8 1 0 . 0 7 \pm 1 0 . 6 4$ </td><td> $1 5 0 5 . 9 2 \pm 1 1 . 6 1$ </td><td> $2 2 5 6 . 9 8 \pm 9 1 . 1 2$ </td></tr><tr><td>25%</td><td> $1 7 7 4 . 7 5 \pm 2 1 . 7 0$ </td><td> $1 3 2 8 . 3 8 \pm 6 6 . 2 8$ </td><td> $1 5 8 5 . 6 3 \pm 6 2 . 4 1$ </td></tr><tr><td>50%</td><td> $1 6 9 3 . 9 3 \pm 2 3 . 2 4$ </td><td> $1 1 5 7 . 9 0 \pm 1 7 . 2 9$ </td><td> $9 3 3 . 1 0 \pm 3 1 . 2 6$ </td></tr><tr><td>75%</td><td> $\mathbf { 1 6 1 2 . 5 4 \ : \pm { \ : 1 2 . 6 0 } }$ </td><td> $\mathbf { 1 0 2 7 . 7 2 \pm 3 7 . 6 2 }$ </td><td> ${ \bf 6 1 7 . 4 3 \pm 1 5 . 4 9 }$ </td></tr></table>

TABLE XLVII: Bank-size threshold ablation under IID partition with 20% malicious clients. Metric: Semantic Entropy. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $7 . 0 2 \pm 0 . 0 7$ </td><td> $7 . 5 2 \pm 0 . 0 6$ </td><td> $9 . 9 5 \pm 0 . 0 2$ </td></tr><tr><td>25%</td><td> $6 . 9 2 \pm 0 . 0 3$ </td><td> $6 . 9 1 \pm 0 . 1 3$ </td><td> $9 . 6 7 \pm 0 . 0 3$ </td></tr><tr><td>50%</td><td> $6 . 9 5 \pm 0 . 0 4$ </td><td> $6 . 4 8 \pm 0 . 1 1$ </td><td> $8 . 7 9 \pm 0 . 0 6$ </td></tr><tr><td>75%</td><td> ${ \bf 6 . 8 8 \pm 0 . 0 7 }$ </td><td> ${ \bf 6 . 2 9 \pm 0 . 0 5 }$ </td><td> ${ \bf 7 . 5 0 \pm 0 . 0 7 }$ </td></tr></table>

TABLE XLVIII: Bank-size threshold ablation under IID partition with 30% malicious clients. Metric: Test Loss. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $7 . 5 0 \pm 0 . 0 1$ </td><td> $7 . 3 2 \pm 0 . 0 7$ </td><td> $7 . 7 2 \pm 0 . 0 4$ </td></tr><tr><td>25%</td><td> $7 . 4 8 \pm 0 . 0 1$ </td><td> $7 . 1 8 \pm 0 . 0 4$ </td><td> $7 . 3 7 \pm 0 . 0 4$ </td></tr><tr><td>50%</td><td> $7 . 4 4 \pm 0 . 0 1$ </td><td> $7 . 0 6 \pm 0 . 0 5$ </td><td> $6 . 8 1 \pm 0 . 0 5$ </td></tr><tr><td>75%</td><td> ${ \bf 7 . 3 9 \pm 0 . 0 1 }$ </td><td> ${ \bf 6 . 9 2 \pm 0 . 0 3 }$ </td><td> ${ \bf 6 . 4 0 \pm 0 . 0 3 }$ </td></tr></table>

TABLE XLIX: Bank-size threshold ablation under IID partition with $3 0 \%$ malicious clients. Metric: Test PPL. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 8 1 3 . 9 2 \pm 1 2 . 1 7$ </td><td> $1 5 0 8 . 0 4 \pm 1 1 0 . 3 4$ </td><td> $2 2 5 7 . 6 9 \pm 9 1 . 1 5$ </td></tr><tr><td>25%</td><td> $1 7 7 8 . 8 1 \pm 1 0 . 1 8$ </td><td> $1 3 1 2 . 6 5 \pm 4 6 . 4 1$ </td><td> $1 5 8 5 . 2 4 \pm 6 2 . 6 5$ </td></tr><tr><td>50%</td><td> $1 6 9 4 . 8 8 \pm 1 6 . 3 8$ </td><td> $1 1 6 9 . 2 9 \pm 6 1 . 5 2$ </td><td> $9 0 3 . 3 6 \pm 4 3 . 1 6$ </td></tr><tr><td>75%</td><td> $\mathbf { 1 6 1 9 . 1 6 \pm 1 1 . 4 2 }$ </td><td> $\mathbf { 1 0 1 1 . 1 0 \ : \pm : 3 5 . 2 2 }$ </td><td> ${ \bf 6 0 0 . 0 6 \pm 1 5 . 8 2 }$ </td></tr></table>

TABLE L: Bank-size threshold ablation under IID partition with $3 0 \%$ malicious clients. Metric: Semantic Entropy. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $7 . 0 3 \pm 0 . 0 8$ </td><td> $7 . 5 1 \pm 0 . 0 7$ </td><td> $9 . 9 5 \pm 0 . 0 2$ </td></tr><tr><td>25%</td><td> $6 . 9 6 \pm 0 . 0 2$ </td><td> $6 . 9 4 \pm 0 . 1 8$ </td><td> $9 . 6 7 \pm 0 . 0 3$ </td></tr><tr><td>50%</td><td> $6 . 9 6 \pm 0 . 0 4$ </td><td> $6 . 4 3 \pm 0 . 0 4$ </td><td> $8 . 7 1 \pm 0 . 1 2$ </td></tr><tr><td>75%</td><td> ${ \bf 6 . 8 8 \pm 0 . 0 7 }$ </td><td> ${ \bf 6 . 4 0 \pm 0 . 0 7 }$ </td><td> ${ \bf 7 . 4 2 \pm 0 . 0 8 }$ </td></tr></table>

TABLE LI: Bank-size threshold ablation under IID partition with 40% malicious clients. Metric: Test Loss. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $7 . 5 1 \pm 0 . 0 0$ </td><td> $7 . 3 3 \pm 0 . 0 7$ </td><td> $7 . 7 9 \pm 0 . 0 7$ </td></tr><tr><td>25%</td><td> $7 . 4 9 \pm 0 . 0 1$ </td><td> $7 . 2 4 \pm 0 . 0 4$ </td><td> $7 . 4 2 \pm 0 . 0 5$ </td></tr><tr><td>50%</td><td> $7 . 4 5 \pm 0 . 0 0$ </td><td> $7 . 0 7 \pm 0 . 0 5$ </td><td> $6 . 8 7 \pm 0 . 0 2$ </td></tr><tr><td>75%</td><td> ${ \bf 7 . 3 9 \pm 0 . 0 1 }$ </td><td> ${ \bf 6 . 9 5 \pm 0 . 0 2 }$ </td><td> ${ \bf 6 . 4 1 \pm 0 . 0 4 }$ </td></tr></table>

TABLE LII: Bank-size threshold ablation under IID partition with $4 0 \%$ malicious clients. Metric: Test PPL. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 8 2 0 . 2 1 \pm 4 . 4 4$ </td><td> $1 5 3 2 . 7 6 \pm 1 1 4 . 7 8$ </td><td> $2 4 0 8 . 1 5 \pm 1 6 2 . 2 5$ </td></tr><tr><td>25%</td><td> $1 7 8 2 . 7 2 \pm 1 0 . 8 2$ </td><td> $1 3 8 8 . 2 1 \pm 6 2 . 5 5$ </td><td> $1 6 7 4 . 9 7 \pm 8 6 . 9 1$ </td></tr><tr><td>50%</td><td> $1 7 1 8 . 8 2 \pm 4 . 6 5$ </td><td> $1 1 7 1 . 9 2 \pm 5 3 . 0 2$ </td><td> $9 6 4 . 0 1 \pm 2 2 . 5 1$ </td></tr><tr><td>75%</td><td> ${ \bf 1 6 2 6 . 8 7 \pm 9 . 5 0 }$ </td><td> ${ \bf 1 0 4 3 . 9 6 } \pm 2 2 . 2 0$ </td><td> ${ \bf 6 0 9 . 8 7 } \pm 2 4 . 4 4$ </td></tr></table>

TABLE LIII: Bank-size threshold ablation under IID partition with $4 0 \%$ malicious clients. Metric: Semantic Entropy. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication round after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $7 . 1 1 \pm 0 . 0 6$ </td><td> $7 . 9 2 \pm 0 . 4 8$ </td><td> $9 . 9 9 \pm 0 . 0 4$ </td></tr><tr><td>25%</td><td> $7 . 0 2 \pm 0 . 0 4$ </td><td> $7 . 1 3 \pm 0 . 2 2$ </td><td> $9 . 7 3 \pm 0 . 0 6$ </td></tr><tr><td>50%</td><td> $6 . 9 7 \pm 0 . 0 7$ </td><td> $6 . 5 3 \pm 0 . 0 6$ </td><td> $8 . 8 8 \pm 0 . 0 9$ </td></tr><tr><td>75%</td><td> ${ \bf 6 . 9 3 \pm 0 . 0 3 }$ </td><td> ${ \bf 6 . 3 2 \pm 0 . 1 0 }$ </td><td> ${ \bf 7 . 4 6 \pm 0 . 1 4 }$ </td></tr></table>

## D. Results Under Non-IID Partitions

Tables LIV–LXVIII report the non-IID results. Larger thresholds again generally improve the early active-window values, while also postponing the first global aggregation rounds.

At the strongest 40% malicious-client setting, the 75% activation coverage gives the lowest test loss and perplexity for all three model families under Non-IID partitioning: perplexity falls from $1 9 2 0 . 3 5 \pm 5 6 . 7 1$ to 1698.50 ± 44.56 for GPT/WikiText and from $1 5 1 4 . 8 0 \pm 1 3 9 . 1 3$ to $1 0 7 8 . 9 0 \pm 3 3 . 2 3$ for BERT/Tiny Shakespeare, while for LLaMA/StackOverflow the 0% threshold is highly unstable $( 5 2 6 6 . 5 8 \pm 4 6 1 5 . 4 5 )$ and the 75% threshold reduces this to $7 5 9 . 1 5 \pm 2 4 9 . 9 4$ , which shows that delaying screening until the bank has broader coverage can substantially stabilize early active training under severe attack and non-IID heterogeneity.

For token-level semantic entropy, the 75% threshold is again best for BERT and LLaMA at the strongest attack setting, while for GPT/non-IID the 50% threshold gives the lowest value with 75% close behind, indicating that the best threshold can vary slightly by metric, though the overall pattern still favors larger bank coverage for stable early training.

TABLE LIV: Bank-size threshold ablation under Non-IID partition with 0% malicious clients. Metric: Test Loss. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $7 . 5 8 \pm 0 . 0 1$ </td><td> $7 . 3 5 \pm 0 . 0 3$ </td><td> $7 . 7 3 \pm 0 . 0 5$ </td></tr><tr><td>25%</td><td> $7 . 5 5 \pm 0 . 0 2$ </td><td> $7 . 2 5 \pm 0 . 0 1$ </td><td> $7 . 3 7 \pm 0 . 0 5$ </td></tr><tr><td>50%</td><td> $7 . 5 1 \pm 0 . 0 3$ </td><td> $7 . 1 1 \pm 0 . 0 6$ </td><td> $6 . 8 8 \pm 0 . 0 9$ </td></tr><tr><td>75%</td><td> ${ \bf 7 . 4 4 \pm 0 . 0 2 }$ </td><td> ${ \bf 6 . 9 6 \pm 0 . 0 4 }$ </td><td> ${ \bf 6 . 4 3 \pm 0 . 0 4 }$ </td></tr></table>

TABLE LV: Bank-size threshold ablation under Non-IID partition with $0 \%$ malicious clients. Metric: Test PPL. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 9 5 9 . 5 0 \pm 2 2 . 9 0$ </td><td> $1 5 5 0 . 6 7 \pm 4 7 . 6 9$ </td><td> $2 2 7 3 . 7 0 \pm 1 0 2 . 8 6$ </td></tr><tr><td>25%</td><td> $1 8 9 6 . 9 0 \pm 4 2 . 0 6$ </td><td> $1 4 0 3 . 0 2 \pm 1 1 . 7 3$ </td><td> $1 5 9 6 . 1 5 \pm 7 1 . 8 4$ </td></tr><tr><td>50%</td><td> $1 8 2 3 . 9 7 \pm 4 6 . 0 1$ </td><td> $1 2 2 1 . 7 3 \pm 7 6 . 3 5$ </td><td> $9 7 6 . 6 5 \pm 9 2 . 3 3$ </td></tr><tr><td>75%</td><td> $\mathbf { 1 7 1 0 . 8 8 } \pm 3 2 . 5 2$ </td><td> $\mathbf { 1 0 5 0 . 5 3 \ : \pm { 3 7 . 0 4 } }$ </td><td> ${ \bf 6 2 2 . 5 4 } \pm \mathbf { 2 2 . 0 3 }$ </td></tr></table>

TABLE LVI: Bank-size threshold ablation under Non-IID partition with $0 \%$ malicious clients. Metric: Semantic Entropy. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $6 . 8 6 \pm 0 . 1 0$ </td><td> $7 . 8 2 \pm 0 . 3 8$ </td><td> $9 . 9 5 \pm 0 . 0 2$ </td></tr><tr><td>25%</td><td> $6 . 8 2 \pm 0 . 0 3$ </td><td> $6 . 9 5 \pm 0 . 1 8$ </td><td> $9 . 6 8 \pm 0 . 0 4$ </td></tr><tr><td>50%</td><td> $6 . 7 2 \pm 0 . 0 3$ </td><td> $6 . 6 0 \pm 0 . 1 5$ </td><td> $8 . 8 8 \pm 0 . 2 0$ </td></tr><tr><td>75%</td><td> ${ \bf 6 . 7 1 \pm 0 . 0 3 }$ </td><td> ${ \bf 6 . 3 3 \pm 0 . 0 7 }$ </td><td> ${ \bf 7 . 4 8 \pm 0 . 0 6 }$ </td></tr></table>

TABLE LVII: Bank-size threshold ablation under Non-IID partition with 10% malicious clients. Metric: Test Loss. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $7 . 5 5 \pm 0 . 0 2$ </td><td> $7 . 3 3 \pm 0 . 0 1$ </td><td> $7 . 7 3 \pm 0 . 0 5$ </td></tr><tr><td>25%</td><td> $7 . 5 2 \pm 0 . 0 2$ </td><td> $7 . 2 5 \pm 0 . 0 1$ </td><td> $7 . 3 7 \pm 0 . 0 5$ </td></tr><tr><td>50%</td><td> $7 . 4 8 \pm 0 . 0 3$ </td><td> $7 . 1 0 \pm 0 . 0 7$ </td><td> $6 . 8 4 \pm 0 . 0 4$ </td></tr><tr><td>75%</td><td> ${ \bf 7 . 4 2 \pm 0 . 0 2 }$ </td><td> ${ \bf 6 . 9 5 \pm 0 . 0 5 }$ </td><td> ${ \bf 6 . 4 4 \pm 0 . 0 3 }$ </td></tr></table>

## E. Interpretation

The study characterizes $M _ { \mathrm { a c t } }$ as the bank coverage at which screening and global aggregation begin, rather than as a desired final bank size. The bank continues to grow after activation. Across the evaluated settings, the dominant trend favors broader activation coverage: perplexity improves consistently as coverage increases, and token-level semantic entropy generally follows the same pattern. This supports maximizing the bank coverage available at activation whenever the resulting delay is acceptable.

TABLE LVIII: Bank-size threshold ablation under Non-IID partition with 10% malicious clients. Metric: Test PPL. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 8 9 9 . 8 9 \pm 4 4 . 6 9$ </td><td> $1 5 3 1 . 8 8 \pm 1 3 . 6 0$ </td><td> $2 2 6 5 . 8 3 \pm 1 0 0 . 9 4$ </td></tr><tr><td>25%</td><td> $1 8 5 3 . 0 8 \pm 4 3 . 5 8$ </td><td> $1 4 0 2 . 9 9 \pm 7 . 8 6$ </td><td> $1 5 9 2 . 0 5 \pm 7 1 . 1 5$ </td></tr><tr><td>50%</td><td> $1 7 6 8 . 4 4 \pm 5 1 . 1 1$ </td><td> $1 2 1 3 . 3 9 \pm 8 3 . 1 4$ </td><td> $9 3 7 . 2 5 \pm 3 5 . 5 6$ </td></tr><tr><td>75%</td><td> $\mathbf { 1 6 7 2 . 1 4 } \pm \ : 3 7 . 8 5$ </td><td> $\mathbf { 1 0 4 7 . 9 3 } \pm 5 4 . 2 \mathbf { 0 }$ </td><td> ${ \bf 6 2 8 . 9 6 \pm 1 8 . 8 2 }$ </td></tr></table>

TABLE LIX: Bank-size threshold ablation under Non-IID partition with 10% malicious clients. Metric: Semantic Entropy. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $6 . 7 6 \pm 0 . 0 1$ </td><td> $7 . 6 3 \pm 0 . 0 5$ </td><td> $9 . 9 6 \pm 0 . 0 2$ </td></tr><tr><td>25%</td><td> $6 . 7 1 \pm 0 . 0 1$ </td><td> $6 . 9 7 \pm 0 . 1 5$ </td><td> $9 . 6 8 \pm 0 . 0 4$ </td></tr><tr><td>50%</td><td> ${ \bf 6 . 6 8 \pm 0 . 0 5 }$ </td><td> $6 . 6 1 \pm 0 . 1 9$ </td><td> $8 . 8 0 \pm 0 . 0 8$ </td></tr><tr><td>75%</td><td> $6 . 7 1 \pm 0 . 0 2$ </td><td> ${ \bf 6 . 4 2 \pm 0 . 0 2 }$ </td><td> ${ \bf 7 . 5 3 \pm 0 . 0 5 }$ </td></tr></table>

We use 50% population coverage as the default practical activation point, not because it is statistically preferable to broader coverage, but because it begins aggregation earlier than the largest evaluated threshold of 75%. In deployment with intermittent or nonuniform participation, waiting for all client identities may be impractically slow. A smaller activation bank can therefore provide an operational starting point, while the probability analysis in Appendix A quantifie its benign-majority reliability under uniform random participation. The ablation does not evaluate 100% activation coverage and therefore does not establish full coverage as an empirical optimum.

TABLE LX: Bank-size threshold ablation under Non-IID partition with 20% malicious clients. Metric: Test Loss. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $7 . 5 5 \pm 0 . 0 2$ </td><td> $7 . 2 7 \pm 0 . 1 2$ </td><td> $7 . 7 3 \pm 0 . 0 5$ </td></tr><tr><td>25%</td><td> $7 . 5 2 \pm 0 . 0 2$ </td><td> $7 . 1 4 \pm 0 . 1 7$ </td><td> $7 . 3 8 \pm 0 . 0 5$ </td></tr><tr><td>50%</td><td> $7 . 4 7 \pm 0 . 0 4$ </td><td> $7 . 0 8 \pm 0 . 1 6$ </td><td> $6 . 8 5 \pm 0 . 0 4$ </td></tr><tr><td>75%</td><td> ${ \bf 7 . 4 2 \pm 0 . 0 3 }$ </td><td> ${ \bf 6 . 9 5 \pm 0 . 1 1 }$ </td><td> ${ \bf 6 . 4 3 \pm 0 . 0 2 }$ </td></tr></table>

TABLE LXI: Bank-size threshold ablation under Non-IID partition with 20% malicious clients. Metric: Test PPL. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { a c t } } .$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 8 9 2 . 8 3 \pm 4 5 . 8 0$ </td><td> $1 4 3 9 . 5 8 \pm 1 6 3 . 6 5$ </td><td> $2 2 8 4 . 6 3 \pm 1 2 2 . 3 2$ </td></tr><tr><td>25%</td><td> $1 8 4 3 . 0 9 \pm 4 3 . 5 9$ </td><td> $1 2 7 0 . 9 5 \pm 2 0 2 . 0 1$ </td><td> $1 6 0 3 . 7 9 \pm 8 4 . 1 0$ </td></tr><tr><td>50%</td><td> $1 7 5 9 . 5 3 \pm 6 5 . 0 5$ </td><td> $1 1 9 9 . 7 9 \pm 1 8 2 . 4 8$ </td><td> $9 4 1 . 8 3 \pm 4 1 . 0 3$ </td></tr><tr><td>75%</td><td> $\mathbf { 1 6 7 3 . 2 3 \ : \pm { 4 2 . 1 0 } }$ </td><td> $\mathbf { 1 0 4 8 . 6 8 \pm 1 1 1 . 6 5 }$ </td><td> ${ \bf 6 2 2 . 5 0 \pm 1 5 . 1 2 }$ </td></tr></table>

TABLE LXII: Bank-size threshold ablation under Non-IID partition with 20% malicious clients. Metric: Semantic Entropy. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $6 . 7 8 \pm 0 . 0 8$ </td><td> $7 . 4 7 \pm 0 . 3 1$ </td><td> $9 . 9 6 \pm 0 . 0 3$ </td></tr><tr><td>25%</td><td> $6 . 8 4 \pm 0 . 1 0$ </td><td> $6 . 8 4 \pm 0 . 4 5$ </td><td> $9 . 6 9 \pm 0 . 0 5$ </td></tr><tr><td>50%</td><td> $6 . 7 5 \pm 0 . 0 6$ </td><td> $6 . 5 9 \pm 0 . 1 6$ </td><td> $8 . 8 1 \pm 0 . 0 8$ </td></tr><tr><td>75%</td><td> ${ \bf 6 . 7 4 \pm 0 . 0 2 }$ </td><td> ${ \bf 6 . 4 8 \pm 0 . 1 0 }$ </td><td> ${ \bf 7 . 5 0 \pm 0 . 0 6 }$ </td></tr></table>

TABLE LXIII: Bank-size threshold ablation under Non-IID partition with 30% malicious clients. Metric: Test Loss. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $7 . 5 6 \pm 0 . 0 2$ </td><td> $7 . 3 0 \pm 0 . 0 6$ </td><td> $7 . 7 3 \pm 0 . 0 5$ </td></tr><tr><td>25%</td><td> $7 . 5 2 \pm 0 . 0 4$ </td><td> $7 . 2 0 \pm 0 . 0 9$ </td><td> $7 . 3 8 \pm 0 . 0 5$ </td></tr><tr><td>50%</td><td> $7 . 4 7 \pm 0 . 0 3$ </td><td> $7 . 0 9 \pm 0 . 0 7$ </td><td> $6 . 8 1 \pm 0 . 0 6$ </td></tr><tr><td>75%</td><td> ${ \bf 7 . 4 3 \pm 0 . 0 2 }$ </td><td> ${ \bf 6 . 9 7 \pm 0 . 0 3 }$ </td><td> ${ \bf 6 . 4 1 \pm 0 . 0 3 }$ </td></tr></table>

TABLE LXIV: Bank-size threshold ablation under Non-IID partition with 30% malicious clients. Metric: Test PPL. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 9 1 4 . 4 1 \pm 3 5 . 8 1$ </td><td> $1 4 8 4 . 6 0 \pm 8 9 . 7 1$ </td><td> $2 2 7 9 . 7 6 \pm 1 1 6 . 9 4$ </td></tr><tr><td>25%</td><td> $1 8 4 9 . 7 6 \pm 6 5 . 5 4$ </td><td> $1 3 3 9 . 7 5 \pm 1 2 1 . 3 0$ </td><td> $1 6 0 0 . 1 9 \pm 7 9 . 7 9$ </td></tr><tr><td>50%</td><td> $1 7 6 0 . 2 5 \pm 5 3 . 0 4$ </td><td> $1 2 0 6 . 5 0 \pm 8 0 . 0 7$ </td><td> $9 1 1 . 9 3 \pm 5 2 . 6 4$ </td></tr><tr><td>75%</td><td> ${ \bf 1 6 7 7 . 7 6 } \pm 3 7 . 7 2$ </td><td> ${ \bf 1 0 6 3 . 7 9 } \pm { \bf 2 6 . 8 4 }$ </td><td> ${ \bf 6 0 5 . 2 2 \pm 1 8 . 2 6 }$ </td></tr></table>

TABLE LXV: Bank-size threshold ablation under Non-IID partition with 30% malicious clients. Metric: Semantic Entropy. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> ${ \bf 6 . 7 4 \pm 0 . 0 7 }$ </td><td> $7 . 6 2 \pm 0 . 1 1$ </td><td> $9 . 9 6 \pm 0 . 0 3$ </td></tr><tr><td>25%</td><td> $6 . 7 9 \pm 0 . 0 1$ </td><td> $6 . 9 2 \pm 0 . 1 4$ </td><td> $9 . 6 8 \pm 0 . 0 5$ </td></tr><tr><td>50%</td><td> $6 . 7 5 \pm 0 . 0 5$ </td><td> $6 . 5 3 \pm 0 . 1 0$ </td><td> $8 . 7 2 \pm 0 . 1 5$ </td></tr><tr><td>75%</td><td> $6 . 7 8 \pm 0 . 0 5$ </td><td> ${ \bf 6 . 4 0 \pm 0 . 0 4 }$ </td><td> ${ \bf 7 . 4 3 \pm 0 . 0 8 }$ </td></tr></table>

TABLE LXVI: Bank-size threshold ablation under Non-IID partition with 40% malicious clients. Metric: Test Loss. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $7 . 5 6 \pm 0 . 0 3$ </td><td> $7 . 3 2 \pm 0 . 0 9$ </td><td> $8 . 3 3 \pm 0 . 8 2$ </td></tr><tr><td>25%</td><td> $7 . 5 3 \pm 0 . 0 3$ </td><td> $7 . 2 1 \pm 0 . 1 1$ </td><td> $7 . 9 6 \pm 0 . 8 2$ </td></tr><tr><td>50%</td><td> $7 . 4 9 \pm 0 . 0 4$ </td><td> $7 . 1 3 \pm 0 . 0 8$ </td><td> $7 . 2 8 \pm 0 . 6 1$ </td></tr><tr><td>75%</td><td> ${ \bf 7 . 4 4 \pm 0 . 0 3 }$ </td><td> ${ \bf 6 . 9 8 \pm 0 . 0 3 }$ </td><td> ${ \bf 6 . 6 0 \pm 0 . 3 1 }$ </td></tr></table>

TABLE LXVII: Bank-size threshold ablation under Non-IID partition with $4 0 \%$ malicious clients. Metric: Test PPL. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $1 9 2 0 . 3 5 \pm 5 6 . 7 1$ </td><td> $1 5 1 4 . 8 0 \pm 1 3 9 . 1 3$ </td><td> $5 2 6 6 . 5 8 \pm 4 6 1 5 . 4 5$ </td></tr><tr><td>25%</td><td> $1 8 6 0 . 9 4 \pm 6 0 . 7 3$ </td><td> $1 3 5 5 . 8 4 \pm 1 3 8 . 7 6$ </td><td> $3 6 5 1 . 8 6 \pm 3 2 0 5 . 9 0$ </td></tr><tr><td>50%</td><td> $1 7 8 3 . 2 6 \pm 6 5 . 1 4$ </td><td> $1 2 4 8 . 3 7 \pm 1 0 1 . 9 2$ </td><td> $1 6 4 9 . 5 2 \pm 1 0 9 2 . 4 1$ </td></tr><tr><td>75%</td><td> $\mathbf { 1 6 9 8 . 5 0 \pm 4 4 . 5 6 }$ </td><td> $\mathbf { 1 0 7 8 . 9 0 } \pm 3 3 . 2 3$ </td><td> $\mathbf { 7 5 9 . 1 5 \pm 2 4 9 . 9 4 }$ </td></tr></table>

TABLE LXVIII: Bank-size threshold ablation under Non-IID partition with 40% malicious clients. Metric: Semantic Entropy. For each seed and threshold, we select the checkpoint with the lowest test loss within the first 10 communication rounds after the bank reaches $M _ { \mathrm { { a c t } } }$ . The best case is highlighted in bold.
<table><tr><td> $M _ { \mathrm { a c t } } / K$ </td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>0%</td><td> $6 . 8 0 \pm 0 . 1 2$ </td><td> $7 . 8 5 \pm 0 . 3 4$ </td><td> $1 0 . 1 2 \pm 0 . 1 7$ </td></tr><tr><td>25%</td><td> $6 . 8 1 \pm 0 . 1 0$ </td><td> $6 . 9 0 \pm 0 . 1 9$ </td><td> $9 . 9 6 \pm 0 . 3 1$ </td></tr><tr><td>50%</td><td> ${ \bf 6 . 7 3 \pm 0 . 0 4 }$ </td><td> $6 . 6 2 \pm 0 . 1 5$ </td><td> $9 . 3 6 \pm 0 . 6 6$ </td></tr><tr><td>75%</td><td> $6 . 7 6 \pm 0 . 0 6$ </td><td> ${ \bf 6 . 2 9 \pm 0 . 1 8 }$ </td><td> ${ \bf 8 . 0 0 \pm 0 . 9 7 }$ </td></tr></table>

## APPENDIX F GMM-SELECTION ABLATION

This appendix reports the ablation study for the GMM selection rule in FEDLNS. The default FEDLNS rule fits both a one-component and a two-component diagonal Gaussian mixture model to the current bank-standardized normalizationsignature features, computes the Bayesian Information Criterion (BIC) for each model, and selects the lower-BIC model. This adaptive rule is compared with a forced two-component variant.

BIC-selected 1-vs-2 GMM.: This is the default FEDLNS rule. It selects

$$
{ \widehat { c } } _ { t } = \arg \operatorname* { m i n } _ { c \in \{ 1 , 2 \} } \mathrm { B I C } ( c ) .\tag{192}
$$

When $\widehat { c } _ { t } = 1$ , all clients are retained. When $\widehat { c } _ { t } = 2$ , FEDLNS retains the component with the smaller median standardized deviation from the bank reference.

Forced 2-GMM.: This variant always fits a two-component GMM after bank activation and retains the component with the smaller median absolute standardized deviation from the bank reference. It is more aggressive than the default method because it always tries to split the current round into two groups, even when the current client signatures may be better explained by a single population.

All GMM ablation runs use the same model, dataset, partition, attack, optimizer, seeds, and bank threshold as the corresponding main experiment. The only changed variable is the GMM decision rule. For each seed, we select the checkpoint with the lowest test loss and report test loss, perplexity, and token-level semantic entropy from that same checkpoint. Thus, token-level semantic entropy is not independently optimized; it is measured at the checkpoint selected by test loss. All entries are reported as mean ± sample standard deviation over seeds {100, 200, 300}.

## A. Results Under IID Partitions

Tables LXIX–LXXXIII report the IID GMM-selection ablation results across all malicious-client fractions. The BICselected and forced two-component rules produce nearly identical selected evaluation-checkpoint values across the three model families. At the strongest 40% malicious-client setting, both rules obtain the same values up to the reported precision for BERT/Tiny Shakespeare and LLaMA/StackOverflow. For GPT/WikiText, the two rules are also essentially identical: the BIC-selected rule gives test loss $7 . 1 6 \pm 0 . 0 0 $ , perplexity $1 2 8 1 . 6 3 \pm 1 . 7 4$ , and token-level semantic entropy $6 . 8 0 { \pm } 0 . 0 6$ , while forced 2-GMM gives test loss 7.16±0.00, perplexity 1281.62±1.73, and token-level semantic entropy $6 . 8 0 \pm 0 . 0 6$

In the IID results, the default BIC-selected rule and the forced two-component rule have nearly identical endpoint values. The ablation does not establish that the current-round feature distribution is statistically identifiable as two components in every attacked setting.

TABLE LXIX: GMM-selection ablation under IID partition with 0% malicious clients. Metric: Test Loss. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 7 . 1 3 \pm 0 . 0 0 }$ </td><td> ${ \bf 6 . 5 0 \pm 0 . 0 5 }$ </td><td> ${ \bf 5 . 3 6 \pm 0 . 0 3 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 7 . 1 3 \pm 0 . 0 0 }$ </td><td> ${ \bf 6 . 5 0 \pm 0 . 0 5 }$ </td><td> ${ \bf 5 . 3 6 \pm 0 . 0 3 }$ </td></tr></table>

TABLE LXX: GMM-selection ablation under IID partition with 0% malicious clients. Metric: Test PPL. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 1 } 2 4 6 . 5 7 \pm 2 . 5 2$ </td><td> ${ \bf 6 6 3 . 6 8 \pm 2 9 . 7 6 }$ </td><td> ${ \bf 2 1 3 . 4 2 \pm 6 . 5 7 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 1 } 2 4 6 . 5 7 \pm 2 . 5 2$ </td><td> ${ \bf 6 6 3 . 6 8 \pm 2 9 . 7 6 }$ </td><td> ${ \bf 2 1 3 . 4 2 \pm 6 . 5 7 }$ </td></tr></table>

TABLE LXXI: GMM-selection ablation under IID partition with 0% malicious clients. Metric: Semantic Entropy. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 6 . 7 3 \pm 0 . 0 3 }$ </td><td> ${ \bf 6 . 2 8 \pm 0 . 0 9 }$ </td><td> ${ \bf 5 . 5 2 \pm 0 . 0 2 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 6 . 7 3 \pm 0 . 0 3 }$ </td><td> ${ \bf 6 . 2 8 \pm 0 . 0 9 }$ </td><td> ${ \bf 5 . 5 2 \pm 0 . 0 2 }$ </td></tr></table>

TABLE LXXII: GMM-selection ablation under IID partition with 10% malicious clients. Metric: Test Loss. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 7 . 1 3 \pm 0 . 0 0 }$ </td><td> ${ \bf 6 . 4 9 \pm 0 . 0 2 }$ </td><td> ${ \bf 5 . 3 7 \pm 0 . 0 3 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 7 . 1 3 \pm 0 . 0 0 }$ </td><td> ${ \bf 6 . 4 9 \pm 0 . 0 2 }$ </td><td> ${ \bf 5 . 3 7 \pm 0 . 0 3 }$ </td></tr></table>

TABLE LXXIII: GMM-selection ablation under IID partition with 10% malicious clients. Metric: Test PPL. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 1 2 4 7 . 2 8 \pm 5 . 9 7 }$ </td><td> ${ \bf 6 5 5 . 9 7 \pm 1 1 . 6 6 }$ </td><td> ${ \bf 2 1 3 . 9 3 \pm 6 . 5 6 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 1 2 4 7 . 2 8 \pm 5 . 9 7 }$ </td><td> ${ \bf 6 5 5 . 9 7 \pm 1 1 . 6 6 }$ </td><td> ${ \bf 2 1 3 . 9 3 \pm 6 . 5 6 }$ </td></tr></table>

TABLE LXXIV: GMM-selection ablation under IID partition with 10% malicious clients. Metric: Semantic Entropy. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 6 . 8 1 \pm 0 . 0 1 }$ </td><td> ${ \bf 6 . 3 8 \pm 0 . 1 6 }$ </td><td> ${ \bf 5 . 5 2 \pm 0 . 0 4 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 6 . 8 1 \pm 0 . 0 1 }$ </td><td> ${ \bf 6 . 3 8 \pm 0 . 1 6 }$ </td><td> ${ \bf 5 . 5 2 \pm 0 . 0 4 }$ </td></tr></table>

TABLE LXXV: GMM-selection ablation under IID partition with 20% malicious clients. Metric: Test Loss. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 7 . 1 3 \pm 0 . 0 0 }$ </td><td> ${ \bf 6 . 4 9 \pm 0 . 0 2 }$ </td><td> ${ \bf 5 . 3 7 \pm 0 . 0 3 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 7 . 1 3 \pm 0 . 0 0 }$ </td><td> ${ \bf 6 . 4 9 \pm 0 . 0 2 }$ </td><td> ${ \bf 5 . 3 7 \pm 0 . 0 3 }$ </td></tr></table>

TABLE LXXVI: GMM-selection ablation under IID partition with 20% malicious clients. Metric: Test PPL. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 1 2 4 9 . 1 1 \pm 1 . 7 8 }$ </td><td> ${ \bf 6 6 1 . 3 8 \pm 1 1 . 9 4 }$ </td><td> $\pm 1 4 . 5 7 \pm 6 . 7 5$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 1 2 4 9 . 1 1 \pm 1 . 7 8 }$ </td><td> ${ \bf 6 6 1 . 3 8 \pm 1 1 . 9 4 }$ </td><td> $\pm 1 4 . 5 7 \pm 6 . 7 5$ </td></tr></table>

TABLE LXXVII: GMM-selection ablation under IID partition with 20% malicious clients. Metric: Semantic Entropy. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 6 . 7 5 \pm 0 . 0 2 }$ </td><td> ${ \bf 6 . 1 9 \pm 0 . 0 9 }$ </td><td> ${ \bf 5 . 5 4 \pm 0 . 0 3 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 6 . 7 5 \pm 0 . 0 2 }$ </td><td> ${ \bf 6 . 1 9 \pm 0 . 0 9 }$ </td><td> ${ \bf 5 . 5 4 \pm 0 . 0 3 }$ </td></tr></table>

TABLE LXXVIII: GMM-selection ablation under IID partition with 30% malicious clients. Metric: Test Loss. Value are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 7 . 1 4 \pm 0 . 0 0 }$ </td><td> ${ \bf 6 . 5 1 \pm 0 . 0 1 }$ </td><td> ${ \bf 5 . 3 7 \pm 0 . 0 3 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 7 . 1 4 \pm 0 . 0 0 }$ </td><td> ${ \bf 6 . 5 1 \pm 0 . 0 1 }$ </td><td> ${ \bf 5 . 3 7 \pm 0 . 0 3 }$ </td></tr></table>

TABLE LXXIX: GMM-selection ablation under IID partition with 30% malicious clients. Metric: Test PPL. Values are mean $\pm$ standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 1 2 5 8 . 2 1 \pm 4 . 7 0 }$ </td><td> ${ \bf 6 6 9 . 3 7 \pm 5 . 8 6 }$ </td><td> $2 1 5 . 6 1 \pm 6 . 7 0$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 1 2 5 8 . 2 1 \pm 4 . 7 0 }$ </td><td> ${ \bf 6 6 9 . 3 7 \pm 5 . 8 6 }$ </td><td> ${ \bf 2 1 5 . 6 1 \pm 6 . 7 0 }$ </td></tr></table>

TABLE LXXX: GMM-selection ablation under IID partition with 30% malicious clients. Metric: Semantic Entropy. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 6 . 7 4 \pm 0 . 0 4 }$ </td><td> ${ \bf 6 . 2 4 \pm 0 . 1 0 }$ </td><td> ${ \bf 5 . 5 2 \pm 0 . 0 4 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 6 . 7 4 \pm 0 . 0 4 }$ </td><td> ${ \bf 6 . 2 4 \pm 0 . 1 0 }$ </td><td> ${ \bf 5 . 5 2 \pm 0 . 0 4 }$ </td></tr></table>

TABLE LXXXI: GMM-selection ablation under IID partition with $4 0 \%$ malicious clients. Metric: Test Loss. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> $7 . 1 6 \pm 0 . 0 0$ </td><td> ${ \bf 6 . 5 5 \pm 0 . 0 2 }$ </td><td> ${ \bf 5 . 3 8 \pm 0 . 0 3 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 7 . 1 6 \pm 0 . 0 0 }$ </td><td> ${ \bf 6 . 5 5 \pm 0 . 0 2 }$ </td><td> ${ \bf 5 . 3 8 \pm 0 . 0 3 }$ </td></tr></table>

TABLE LXXXII: GMM-selection ablation under IID partition with 40% malicious clients. Metric: Test PPL. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> $1 2 8 1 . 6 3 \pm 1 . 7 4$ </td><td> ${ \bf 6 9 7 . 6 2 \pm 1 3 . 6 2 }$ </td><td> ${ \bf 2 1 7 . 1 4 \pm 6 . 8 5 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 1 2 8 1 . 6 2 \pm 1 . 7 3 }$ </td><td> ${ \bf 6 9 7 . 6 2 \pm 1 3 . 6 2 }$ </td><td> ${ \bf 2 1 7 . 1 4 \pm 6 . 8 5 }$ </td></tr></table>

TABLE LXXXIII: GMM-selection ablation under IID partition with 40% malicious clients. Metric: Semantic Entropy. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 6 . 8 0 \pm 0 . 0 6 }$ </td><td> ${ \bf 6 . 2 4 \pm 0 . 0 6 }$ </td><td> ${ \bf 5 . 5 2 \pm 0 . 0 2 }$ </td></tr><tr><td>Forced 2-GMM</td><td> $6 . 8 0 \pm 0 . 0 6$ </td><td> ${ \bf 6 . 2 4 \pm 0 . 0 6 }$ </td><td> ${ \bf 5 . 5 2 \pm 0 . 0 2 }$ </td></tr></table>

## B. Results Under Non-IID Partitions

Tables LXXXIV–XCVIII report the corresponding non-IID results. The non-IID results follow the same pattern: BICselected one-vs-two GMM and forced two-component GMM have nearly indistinguishable endpoint values in most settings. At the strongest 40% malicious-client setting, the two methods produce identical reported values for all three model families. For GPT/WikiText, both rules obtain test loss $7 . 1 6 \pm 0 . 0 1$ , perplexity $1 2 8 6 . 5 0 \pm 1 5 . 3 4 $ , and token-level semantic entropy $6 . 9 2 \pm 0 . 2 4 .$ . For BERT/Tiny Shakespeare, both obtain test loss $6 . 4 8 \pm 0 . 1 3 .$ , perplexity $6 5 8 . 2 1 \pm 9 1 . 6 6 $ and token-level semantic entropy $6 . 1 9 \pm 0 . 1 3$ . For LLaMA/StackOverflow, both obtain test loss $5 . 3 9 \pm 0 . 0 3 .$ , perplexity $2 2 0 . 1 8 \pm 6 . 4 5$ , and token-level semantic entropy $5 . 5 2 \pm 0 . 0 3$

The main visible difference appears only in a small number of intermediate settings. For example, under GPT/WikiText with non-IID partitioning and 30% malicious clients, BIC-selected GMM gives perplexity 1277.28 ± 23.52, while forced 2-GMM gives $1 2 7 7 . 8 1 \pm 2 2 . 9 3$ . For token-level semantic entropy in the same setting, forced 2-GMM gives $6 . 8 7 \pm 0 . 0 6$ while BIC-selected GMM gives $6 . 9 0 \pm 0 . 0 2$ . These differences are small and do not indicate a systematic advantage of forced splitting.

TABLE LXXXIV: GMM-selection ablation under Non-IID partition with 0% malicious clients. Metric: Test Loss. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> $7 . 1 1 \pm 0 . 0 1$ </td><td> ${ \bf 6 . 4 7 \pm 0 . 0 5 }$ </td><td> ${ \bf 5 . 3 7 \pm 0 . 0 3 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 7 . 1 1 \pm 0 . 0 1 }$ </td><td> ${ \bf 6 . 4 7 \pm 0 . 0 5 }$ </td><td> ${ \bf 5 . 3 7 \pm 0 . 0 3 }$ </td></tr></table>

TABLE LXXXV: GMM-selection ablation under Non-IID partition with 0% malicious clients. Metric: Test PPL. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> $1 2 2 5 . 9 0 \pm 1 2 . 0 6$ </td><td> ${ \bf 6 4 3 . 5 6 } \pm 3 3 . 4 6$ </td><td> $\mathbf { 2 1 5 . 5 5 \ : \pm { \ : 6 . 8 5 } }$ </td></tr><tr><td>Forced 2-GMM</td><td> $\mathbf { 1 2 2 5 . 8 9 \pm 1 2 . 0 6 }$ </td><td> ${ \bf 6 4 3 . 5 6 } \pm 3 3 . 4 6$ </td><td> $\mathbf { 2 1 5 . 5 5 \ : \pm { \ : 6 . 8 5 } }$ </td></tr></table>

TABLE LXXXVI: GMM-selection ablation under Non-IID partition with 0% malicious clients. Metric: Semantic Entropy. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 6 . 8 0 \pm 0 . 0 9 }$ </td><td> ${ \bf 6 . 1 2 \pm 0 . 0 5 }$ </td><td> ${ \bf 5 . 5 4 \pm 0 . 0 3 }$ </td></tr><tr><td>Forced 2-GMM</td><td> $6 . 8 0 \pm 0 . 0 9$ </td><td> ${ \bf 6 . 1 2 \pm 0 . 0 5 }$ </td><td> ${ \bf 5 . 5 4 \pm 0 . 0 3 }$ </td></tr></table>

TABLE LXXXVII: GMM-selection ablation under Non-IID partition with 10% malicious clients. Metric: Test Loss. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 7 . 1 2 \pm 0 . 0 0 }$ </td><td> ${ \bf 6 . 3 6 \pm 0 . 0 6 }$ </td><td> ${ \bf 5 . 3 8 \pm 0 . 0 3 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 7 . 1 2 \pm 0 . 0 0 }$ </td><td> ${ \bf 6 . 3 6 \pm 0 . 0 6 }$ </td><td> ${ \bf 5 . 3 8 \pm 0 . 0 3 }$ </td></tr></table>

## C. Interpretation

The BIC-selected and forced two-component rules yield nearly identical endpoint values in the evaluated runs. The forced rule always partitions the current participants, whereas BIC preserves the option to retain all clients when the one-component model receives the lower criterion value. This ablation therefore characterizes endpoint sensitivity to the model-selection rule; it does not establish statistical identifiability of the fitted mixtures.

TABLE LXXXVIII: GMM-selection ablation under Non-IID partition with 10% malicious clients. Metric: Test PPL. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 1 } 2 3 8 . 5 2 \pm { \bf 4 . 9 7 }$ </td><td> ${ \pm 7 6 . 6 3 \pm 3 6 . 5 8 }$ </td><td> $\mathbf { 2 1 6 . 1 3 \ : \pm { \ : 6 . 7 0 } }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 1 } 2 3 8 . 5 2 \pm { \bf 4 . 9 7 }$ </td><td> ${ \pm 7 6 . 6 3 \pm 3 6 . 5 8 }$ </td><td> $\mathbf { 2 1 6 . 1 3 \ : \pm { \ : 6 . 7 0 } }$ </td></tr></table>

TABLE LXXXIX: GMM-selection ablation under Non-IID partition with 10% malicious clients. Metric: Semantic Entropy. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 6 . 7 9 \pm 0 . 0 5 }$ </td><td> ${ \bf 6 . 0 6 \pm 0 . 1 8 }$ </td><td> ${ \bf 5 . 5 4 \pm 0 . 0 3 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 6 . 7 9 \pm 0 . 0 5 }$ </td><td> ${ \bf 6 . 0 6 \pm 0 . 1 8 }$ </td><td> ${ \bf 5 . 5 4 \pm 0 . 0 3 }$ </td></tr></table>

TABLE XC: GMM-selection ablation under Non-IID partition with 20% malicious clients. Metric: Test Loss. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 7 . 1 3 \pm 0 . 0 1 }$ </td><td> ${ \bf 6 . 4 2 \pm 0 . 1 4 }$ </td><td> ${ \bf 5 . 3 8 \pm 0 . 0 3 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 7 . 1 3 \pm 0 . 0 1 }$ </td><td> ${ \bf 6 . 4 2 \pm 0 . 1 4 }$ </td><td> ${ \bf 5 . 3 8 \pm 0 . 0 3 }$ </td></tr></table>

TABLE XCI: GMM-selection ablation under Non-IID partition with 20% malicious clients. Metric: Test PPL. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> $\mathbf { 1 2 5 1 . 6 5 \ : \pm { \ : 1 1 . 3 7 } }$ </td><td> ${ \bf 6 1 8 . 3 0 \pm 9 1 . 1 2 }$ </td><td> $\mathbf { 2 1 6 . 9 7 \ : \pm { \ : 6 . 9 1 } }$ </td></tr><tr><td>Forced 2-GMM</td><td> $\mathbf { 1 2 5 1 . 6 5 \ : \pm { \ : 1 1 . 3 7 } }$ </td><td> ${ \bf 6 1 8 . 3 0 \pm 9 1 . 1 2 }$ </td><td> $\mathbf { 2 1 6 . 9 7 \ : \pm { \ : 6 . 9 1 } }$ </td></tr></table>

TABLE XCII: GMM-selection ablation under Non-IID partition with 20% malicious clients. Metric: Semantic Entropy. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 7 . 0 4 \pm 0 . 4 0 }$ </td><td> ${ \bf 6 . 3 3 \pm 0 . 0 9 }$ </td><td> ${ \bf 5 . 5 3 \pm 0 . 0 3 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 7 . 0 4 \pm 0 . 4 0 }$ </td><td> ${ \bf 6 . 3 3 \pm 0 . 0 9 }$ </td><td> ${ \bf 5 . 5 3 \pm 0 . 0 3 }$ </td></tr></table>

TABLE XCIII: GMM-selection ablation under Non-IID partition with 30% malicious clients. Metric: Test Loss. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 7 . 1 5 \pm 0 . 0 2 }$ </td><td> ${ \bf 6 . 3 8 \pm 0 . 1 8 }$ </td><td> ${ \bf 5 . 3 8 \pm 0 . 0 3 }$ </td></tr><tr><td>Forced 2-GMM</td><td> $7 . 1 5 \pm 0 . 0 2$ </td><td> $6 . 3 8 \pm 0 . 1 8$ </td><td> ${ \bf 5 . 3 8 \pm 0 . 0 3 }$ </td></tr></table>

TABLE XCIV: GMM-selection ablation under Non-IID partition with 30% malicious clients. Metric: Test PPL. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 1 2 7 7 . 2 8 \pm 2 3 . 5 2 }$ </td><td> $\mathbf { 5 9 4 . 6 1 \pm 1 0 8 . 7 5 }$ </td><td> $\mathbf { 2 1 8 . 0 2 \ : \pm { \ : 7 . 0 7 } }$ </td></tr><tr><td>Forced 2-GMM</td><td> $1 2 7 7 . 8 1 \pm 2 2 . 9 3$ </td><td> $5 9 4 . 6 3 \pm 1 0 8 . 7 4$ </td><td> $\mathbf { 2 1 8 . 0 2 \ : \pm { \ : 7 . 0 7 } }$ </td></tr></table>

TABLE XCV: GMM-selection ablation under Non-IID partition with 30% malicious clients. Metric: Semantic Entropy. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> $6 . 9 0 \pm 0 . 0 2$ </td><td> $6 . 0 0 \pm 0 . 3 8$ </td><td> ${ \bf 5 . 5 3 \pm 0 . 0 2 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 6 . 8 7 \pm 0 . 0 6 }$ </td><td> ${ \bf 6 . 0 0 \pm 0 . 3 8 }$ </td><td> ${ \bf 5 . 5 3 \pm 0 . 0 2 }$ </td></tr></table>

TABLE XCVI: GMM-selection ablation under Non-IID partition with 40% malicious clients. Metric: Test Loss. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 7 . 1 6 \pm 0 . 0 1 }$ </td><td> ${ \bf 6 . 4 8 \pm 0 . 1 3 }$ </td><td> ${ \bf 5 . 3 9 \pm 0 . 0 3 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 7 . 1 6 \pm 0 . 0 1 }$ </td><td> ${ \bf 6 . 4 8 \pm 0 . 1 3 }$ </td><td> ${ \bf 5 . 3 9 \pm 0 . 0 3 }$ </td></tr></table>

TABLE XCVII: GMM-selection ablation under Non-IID partition with 40% malicious clients. Metric: Test PPL. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> $\mathbf { 1 2 8 6 . 5 0 \pm 1 5 . 3 4 }$ </td><td> ${ \bf 6 5 8 . 2 1 \pm 9 1 . 6 6 }$ </td><td> ${ \bf 2 2 0 . 1 8 \pm 6 . 4 5 }$ </td></tr><tr><td>Forced 2-GMM</td><td> $\mathbf { 1 2 8 6 . 5 0 \pm 1 5 . 3 4 }$ </td><td> ${ \bf 6 5 8 . 2 1 \pm 9 1 . 6 6 }$ </td><td> ${ \bf 2 2 0 . 1 8 \pm 6 . 4 5 }$ </td></tr></table>

TABLE XCVIII: GMM-selection ablation under Non-IID partition with 40% malicious clients. Metric: Semantic Entropy. Values are mean ± standard deviation over three seeds. For each seed, all metrics are taken from the checkpoint with the lowest test loss. The best case is highlighted in bold.
<table><tr><td>GMM rule</td><td>GPT / WikiText</td><td>BERT / Shakespeare</td><td>LLaMA / StackOverflow</td></tr><tr><td>BIC 1/2-GMM</td><td> ${ \bf 6 . 9 2 \pm 0 . 2 4 }$ </td><td> ${ \bf 6 . 1 9 \pm 0 . 1 3 }$ </td><td> ${ \bf 5 . 5 2 \pm 0 . 0 3 }$ </td></tr><tr><td>Forced 2-GMM</td><td> ${ \bf 6 . 9 2 \pm 0 . 2 4 }$ </td><td> ${ \bf 6 . 1 9 \pm 0 . 1 3 }$ </td><td> ${ \bf 5 . 5 2 \pm 0 . 0 3 }$ </td></tr></table>